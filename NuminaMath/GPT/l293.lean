import Mathlib

namespace unique_point_exists_l293_293940

variable {ℝ : Type} [LinearOrderedField ℝ]

structure Point :=
(x : ℝ)
(y : ℝ)

def distance (P Q : Point) : ℝ :=
((P.x - Q.x)^2 + (P.y - Q.y)^2) ^ (1/2)

theorem unique_point_exists (A B C : Point)
  (h : ¬ collinear ℝ A B C) :
  ∃! (X : Point), distance X A ^ 2 + distance X B ^ 2 + distance A B ^ 2 =
                  distance X B ^ 2 + distance X C ^ 2 + distance B C ^ 2 ∧
                  distance X C ^ 2 + distance X A ^ 2 + distance C A ^ 2 :=
sorry

end unique_point_exists_l293_293940


namespace collinear_points_x_eq_4_point_M_on_line_OC_l293_293969

-- Problem 1: Prove that if A, B, and C are collinear, then x = 4
theorem collinear_points_x_eq_4 (A B C : Point) 
  (OA : (ℝ × ℝ)) (OB : (ℝ × ℝ)) (OC : (ℝ × ℝ)) : 
  (OA = (1, 4)) → (OB = (2, 3)) → (OC = (x, 1)) → collinear OA OB OC → x = 4 :=
sorry

-- Problem 2: When x = 3, there exists a point M on the line OC such that 
-- the dot product of vectors MA and MB achieves its minimum value and M has coordinates (12/5, 4/5)
theorem point_M_on_line_OC (A B C M : Point) 
  (OA : (ℝ × ℝ)) (OB : (ℝ × ℝ)) (OM : (ℝ × ℝ)) :
  (OA = (1, 4)) → (OB = (2, 3)) → (OM = (3*λ, λ)) → (x = 3) → 
  exists (M : Point), (OM = (12/5, 4/5)) ∧ 
  ∀ μ : ℝ, (dot_product (OA - OM) (OB - OM)) ≥ (dot_product (OA - (3*λ, λ)) (OB - (3*λ, λ))) :=
sorry

end collinear_points_x_eq_4_point_M_on_line_OC_l293_293969


namespace perfect_squares_difference_l293_293577

theorem perfect_squares_difference : 
  let N : ℕ := 20000;
  let diff_squared (b : ℤ) : ℤ := (b+2)^2 - b^2;
  ∃ k : ℕ, (1 ≤ k ∧ k ≤ 70) ∧ (∀ m : ℕ, (m < N) → (∃ b : ℤ, m = diff_squared b) → m = (2 * k)^2)
:= sorry

end perfect_squares_difference_l293_293577


namespace unique_ordered_quadruples_l293_293869

theorem unique_ordered_quadruples :
  ∃! (a b c d : ℝ), 
    (a + 1) * (d + 1) - b * c ≠ 0 ∧ 
    (by rw [matrix.inv_def, matrix.eq_iff, det_smul, matrix.mul_apply]; 
         exact (∀ i j, a i j * matrix.inv_def ⟨1/(a i j + 1) , 1/(b i j + 1), 1/(c i j + 1), 1/(d i j + 1)⟩ = 1)) :=
sorry

end unique_ordered_quadruples_l293_293869


namespace simplest_square_root_l293_293823

-- Definitions of the conditions
def root8 := Real.sqrt 8
def root1_over_9 := Real.sqrt (1 / 9)
def root_a_squared (a : ℝ) := Real.sqrt (a^2)
def root_a_squared_plus_3 (a : ℝ) := Real.sqrt (a^2 + 3)

-- Statement that root_a_squared_plus_3 is the simplest form
theorem simplest_square_root (a : ℝ) : 
(root_a_squared_plus_3 a) = Real.sqrt (a^2 + 3) := 
sorry

end simplest_square_root_l293_293823


namespace find_k_l293_293536

noncomputable def S : ℕ → ℤ := λ n => n^2 - 7 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

def P : ℕ → Prop := λ k => 16 < a k + a (k + 1) ∧ a k + a (k + 1) < 22

theorem find_k : ∃ k : ℕ, P k ∧ k = 8 :=
by
  sorry

end find_k_l293_293536


namespace identify_heaviest_and_lightest_coin_in_13_weighings_l293_293312

theorem identify_heaviest_and_lightest_coin_in_13_weighings :
  ∀ (coins : Finₓ 10 → ℝ) 
    (balance_weighing : ∀ (a b : Finₓ 10), Prop), 
    (∀ i j, coins i ≠ coins j) → 
    (∃ strategy : ℕ → (Finₓ 10 × Finₓ 10),
      ∃ h : ℕ,
        h ≤ 13 ∧
        (∃ heaviest lightest : Finₓ 10,
          (∀ i, coins heaviest ≥ coins i) ∧ (∀ j, coins lightest ≤ coins j))) :=
by
  sorry

end identify_heaviest_and_lightest_coin_in_13_weighings_l293_293312


namespace jeffs_last_trip_speed_l293_293989

theorem jeffs_last_trip_speed (d₁ d₂ total_distance time₁ time₂ time₃ speed₁ speed₂ last_speed : ℕ)
  (h₁ : speed₁ = 80) (h₂ : time₁ = 6) (h₃ : speed₂ = 60) (h₄ : time₂ = 4) (h₅ : time₃ = 2) (h₆ : total_distance = 800) 
  (h₇ : d₁ = speed₁ * time₁) (h₈ : d₂ = speed₂ * time₂)
  (h₉ : total_distance - (d₁ + d₂) = last_speed * time₃) :
  last_speed = 40 :=
by
  simp at *
  sorry

end jeffs_last_trip_speed_l293_293989


namespace sum_of_constants_l293_293647

def P (x : ℝ) : ℝ := x^2 - 4 * x - 4

theorem sum_of_constants :
  let a := 61
  let b := 81
  let c := 145
  let d := 43
  let e := 17
  in a + b + c + d + e = 347 := by
  let a := 61
  let b := 81
  let c := 145
  let d := 43
  let e := 17
  show a + b + c + d + e = 347
  sorry

end sum_of_constants_l293_293647


namespace initial_speed_is_correct_l293_293806

def initial_speed (v : ℝ) : Prop :=
  let D_total : ℝ := 70 * 5
  let D_2 : ℝ := 85 * 2
  let D_1 := v * 3
  D_total = D_1 + D_2

theorem initial_speed_is_correct :
  ∃ v : ℝ, initial_speed v ∧ v = 60 :=
by
  sorry

end initial_speed_is_correct_l293_293806


namespace equation_of_line_through_point_with_equal_intercepts_l293_293503

open LinearAlgebra

theorem equation_of_line_through_point_with_equal_intercepts :
  ∃ (a b c : ℝ), (a * 1 + b * 2 + c = 0) ∧ (a * b < 0) ∧ ∀ x y : ℝ, 
  (a * x + b * y + c = 0 ↔ (2 * x - y = 0 ∨ x + y - 3 = 0)) :=
sorry

end equation_of_line_through_point_with_equal_intercepts_l293_293503


namespace smallest_n_multiple_of_15_l293_293656

def f (n : ℕ) : ℕ :=
  Nat.find (λ k => Nat.factorial k ∣ n)

theorem smallest_n_multiple_of_15 (n : ℕ) (hn : n % 15 = 0) (hf : 15 < f n) : n = 255 :=
by
  sorry

end smallest_n_multiple_of_15_l293_293656


namespace odd_divisors_perfect_squares_less_than_50_l293_293463

theorem odd_divisors_perfect_squares_less_than_50 : 
  ∃ n, n = { i | i < 50 ∧ ∃ j, j * j = i }.card ∧ n = 7 := by 
  sorry

end odd_divisors_perfect_squares_less_than_50_l293_293463


namespace percentage_of_mortality_l293_293591

theorem percentage_of_mortality
  (P : ℝ) -- The population size could be represented as a real number
  (affected_fraction : ℝ) (dead_fraction : ℝ)
  (h1 : affected_fraction = 0.15) -- 15% of the population is affected
  (h2 : dead_fraction = 0.08) -- 8% of the affected population died
: (affected_fraction * dead_fraction) * 100 = 1.2 :=
by
  sorry

end percentage_of_mortality_l293_293591


namespace anne_find_bottle_caps_l293_293453

theorem anne_find_bottle_caps 
  (n_i n_f : ℕ) (h_initial : n_i = 10) (h_final : n_f = 15) : n_f - n_i = 5 :=
by
  sorry

end anne_find_bottle_caps_l293_293453


namespace find_heaviest_and_lightest_l293_293326

-- Definition of the main problem conditions
def coins : ℕ := 10
def max_weighings : ℕ := 13
def distinct_weights (c : ℕ) : Prop := ∀ (i j : ℕ), i ≠ j → i < c → j < c → weight i ≠ weight j

-- Noncomputed property representing the weight of each coin
noncomputable def weight : ℕ → ℝ := sorry

-- The main theorem statement
theorem find_heaviest_and_lightest (c : ℕ) (mw : ℕ) (dw : distinct_weights c) : c = coins ∧ mw = max_weighings
  → ∃ (h l : ℕ), h < c ∧ l < c ∧ (∀ (i : ℕ), i < c → weight i ≤ weight h ∧ weight i ≥ weight l) :=
by
  sorry

end find_heaviest_and_lightest_l293_293326


namespace tan_neg_1140_eq_neg_sqrt3_l293_293092

theorem tan_neg_1140_eq_neg_sqrt3 
  (tan_neg : ∀ θ : ℝ, Real.tan (-θ) = -Real.tan θ)
  (tan_periodicity : ∀ θ : ℝ, ∀ n : ℤ, Real.tan (θ + n * 180) = Real.tan θ)
  (tan_60 : Real.tan 60 = Real.sqrt 3) :
  Real.tan (-1140) = -Real.sqrt 3 := 
sorry

end tan_neg_1140_eq_neg_sqrt3_l293_293092


namespace evaluate_ceiling_sums_l293_293032

theorem evaluate_ceiling_sums : 
  (⌈real.sqrt 3⌉ + ⌈real.sqrt 33⌉ + ⌈real.sqrt 333⌉) = 27 :=
by
  have h1 : 1 < real.sqrt 3 ∧ real.sqrt 3 < 2 :=
    ⟨by norm_num, by norm_num⟩,
  have h2 : 5 < real.sqrt 33 ∧ real.sqrt 33 < 6 :=
    ⟨by norm_num, by norm_num⟩,
  have h3 : 18 < real.sqrt 333 ∧ real.sqrt 333 < 19 :=
    ⟨by norm_num, by norm_num⟩,
  sorry

end evaluate_ceiling_sums_l293_293032


namespace factorize_x_squared_plus_2x_l293_293061

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by
  sorry

end factorize_x_squared_plus_2x_l293_293061


namespace one_less_than_neg_two_is_neg_three_l293_293821

theorem one_less_than_neg_two_is_neg_three :
  ∃ x : ℤ, x = -2 - 1 ∧ x = -3 :=
begin
  use -3,
  split,
  { 
    refl,
  },
  {
    refl,
  },
end

end one_less_than_neg_two_is_neg_three_l293_293821


namespace simplest_square_root_is_B_l293_293392

-- Definitions of the square root terms
def optionA (a : ℝ) : ℝ := Real.sqrt (16 * a)
def optionB (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)
def optionC (a b : ℝ) : ℝ := Real.sqrt (b / a)
def optionD : ℝ := Real.sqrt 45

-- Simplest square root among the options
theorem simplest_square_root_is_B (a b : ℝ) : optionB a b = Real.sqrt (a^2 + b^2) := by
  sorry

end simplest_square_root_is_B_l293_293392


namespace heaviest_and_lightest_in_13_weighings_l293_293315

/-- Given ten coins of different weights and a balance scale.
    Prove that it is possible to identify the heaviest and the lightest coin
    within 13 weighings. -/
theorem heaviest_and_lightest_in_13_weighings
  (coins : Fin 10 → ℝ)
  (h_different: ∀ i j : Fin 10, i ≠ j → coins i ≠ coins j)
  : ∃ (heaviest lightest : Fin 10),
      (heaviest ≠ lightest) ∧
      (∀ i : Fin 10, coins i ≤ coins heaviest) ∧
      (∀ i : Fin 10, coins lightest ≤ coins i) :=
sorry

end heaviest_and_lightest_in_13_weighings_l293_293315


namespace RS_plus_ST_l293_293704

theorem RS_plus_ST {a b c d e : ℕ} 
  (h1 : a = 68) 
  (h2 : b = 10) 
  (h3 : c = 7) 
  (h4 : d = 6) 
  : e = 3 :=
sorry

end RS_plus_ST_l293_293704


namespace complex_numbers_equilateral_triangle_l293_293477

noncomputable def isEquilateralTriangle (p q r : ℂ) : Prop :=
  ∃ z : ℂ, z ^ 2 - z + 1 = 0 ∧ ((r - p) / (q - p) = z ∨ (r - p) / (q - p) = z.conj)

theorem complex_numbers_equilateral_triangle
  (p q r : ℂ)
  (h1 : isEquilateralTriangle p q r)
  (h2 : complex.abs (p + q + r) = 48)
  : complex.abs (p * q + p * r + q * r) = 768 :=
sorry

end complex_numbers_equilateral_triangle_l293_293477


namespace find_numbers_l293_293516

theorem find_numbers :
  ∃ a b : ℕ, a + b = 60 ∧ Nat.gcd a b + Nat.lcm a b = 84 :=
by
  sorry

end find_numbers_l293_293516


namespace gold_beads_cannot_be_determined_without_cost_per_bead_l293_293002

-- Carly's bead conditions
def purple_rows : ℕ := 50
def purple_beads_per_row : ℕ := 20
def blue_rows : ℕ := 40
def blue_beads_per_row : ℕ := 18
def total_cost : ℝ := 180

-- The calculation of total purple and blue beads
def purple_beads : ℕ := purple_rows * purple_beads_per_row
def blue_beads : ℕ := blue_rows * blue_beads_per_row
def total_beads_without_gold : ℕ := purple_beads + blue_beads

-- Given the lack of cost per bead, the number of gold beads cannot be determined
theorem gold_beads_cannot_be_determined_without_cost_per_bead :
  ¬ (∃ cost_per_bead : ℝ, ∃ gold_beads : ℕ, (purple_beads + blue_beads + gold_beads) * cost_per_bead = total_cost) :=
sorry

end gold_beads_cannot_be_determined_without_cost_per_bead_l293_293002


namespace inequality_proof_l293_293406

theorem inequality_proof (x y z : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (h2 : 1 ≤ x → x^2 + (y^2 + z^2) / x^3 < x^2 + y^2 + z^2)
  (h3 : xyz ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + 
  (y^5 - y^2) / (y^5 + z^2 + x^2) + 
  (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 
  :=
begin
  sorry
end

end inequality_proof_l293_293406


namespace point_in_fourth_quadrant_l293_293553

theorem point_in_fourth_quadrant 
  (A B C : ℝ) 
  (h0 : 0 < A ∧ A < π / 2)
  (h1 : 0 < B ∧ B < π / 2)
  (h2 : 0 < C ∧ C < π / 2)
  (h3 : A + B + C = π) :
  (sin A - cos B > 0) ∧ (cos A - sin C < 0) :=
begin
  sorry
end

end point_in_fourth_quadrant_l293_293553


namespace least_number_with_remainders_l293_293404

theorem least_number_with_remainders :
  ∃ x, (x ≡ 4 [MOD 5]) ∧ (x ≡ 4 [MOD 6]) ∧ (x ≡ 4 [MOD 9]) ∧ (x ≡ 4 [MOD 18]) ∧ x = 94 := 
by 
  sorry

end least_number_with_remainders_l293_293404


namespace ceil_sqrt_sum_l293_293047

theorem ceil_sqrt_sum : 
  (⌈Real.sqrt 3⌉ = 2) ∧ 
  (⌈Real.sqrt 33⌉ = 6) ∧ 
  (⌈Real.sqrt 333⌉ = 19) → 
  2 + 6 + 19 = 27 :=
by 
  intro h
  cases h with h3 h
  cases h with h33 h333
  rw [h3, h33, h333]
  norm_num

end ceil_sqrt_sum_l293_293047


namespace rate_of_current_l293_293297

theorem rate_of_current (speed_in_still_water : ℝ) (distance_downstream : ℝ) (time_downstream : ℝ) (rate_of_current : ℝ) :
  speed_in_still_water = 22 → distance_downstream = 10.4 → time_downstream = 24 / 60 → rate_of_current = 4 := by
  intros h_speed h_distance h_time
  have eq1 : distance_downstream = (speed_in_still_water + rate_of_current) * time_downstream := sorry
  rw [h_speed, h_distance, h_time] at eq1
  have : 10.4 = (22 + rate_of_current) * 0.4 := by exact eq1
  have eq2 : 10.4 / 0.4 = 22 + rate_of_current := sorry
  rw [this] at eq2
  have : 26 - 22 = rate_of_current := sorry
  exact this

end rate_of_current_l293_293297


namespace number_of_correct_propositions_l293_293487

-- Definitions of the original statement, converse, inverse, and contrapositive
def original_statement (x : ℝ) : Prop := x^2 > 0 → x > 0
def converse_statement (x : ℝ) : Prop := x > 0 → x^2 > 0
def inverse_statement (x : ℝ) : Prop := x^2 ≤ 0 → x ≤ 0
def contrapositive_statement (x : ℝ) : Prop := x ≤ 0 → x^2 ≤ 0

-- Main theorem to prove the number of correct propositions
theorem number_of_correct_propositions : 
  (¬ (∀ x, original_statement x) ∧ 
   (∀ x, converse_statement x) ∧ 
   (∀ x, inverse_statement x) ∧ 
   ¬ (∀ x, contrapositive_statement x)) →
  2 :=
by
  sorry

end number_of_correct_propositions_l293_293487


namespace line_through_point_equal_intercepts_l293_293507

-- Definitions based on conditions
def passes_through (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l p.1 p.2

def equal_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a, a ≠ 0 ∧ (∀ x y, l x y ↔ x + y = a) ∨ (∀ x y, l x y ↔ y = 2 * x)

-- Theorem statement based on the problem
theorem line_through_point_equal_intercepts :
  ∃ l, passes_through (1, 2) l ∧ equal_intercepts l ∧
  (∀ x y, l x y ↔ 2 * x - y = 0) ∨ (∀ x y, l x y ↔ x + y - 3 = 0) :=
sorry

end line_through_point_equal_intercepts_l293_293507


namespace sum_of_six_digits_l293_293273

open Finset

theorem sum_of_six_digits 
(vars_cols : Finset ℕ) (vars_rows : Finset ℕ) 
(h1 : vars_cols ⊆ {2, 4, 6, 8}) (h2 : vars_rows ⊆ {1, 3, 5, 7, 9})
(h3 : vars_cols.sum id = 22) (h4 : vars_rows.sum id = 14)
(h5 : (vars_cols ∪ vars_rows).card = 6):
  (vars_cols ∪ vars_rows).sum id = 30 := 
  sorry

end sum_of_six_digits_l293_293273


namespace jackpot_and_bonus_probability_l293_293614

noncomputable def probability_jackpot_and_bonus := 
  let MegaBallProb := 1 / 30
  let orderedSubsetProb := 1 / (50 * 49)
  let unorderedSubsetProb := 1 / (binom 48 5)
  MegaBallProb * orderedSubsetProb * unorderedSubsetProb

theorem jackpot_and_bonus_probability :
  probability_jackpot_and_bonus = 1 / 125,703,480,000 := by
  sorry

end jackpot_and_bonus_probability_l293_293614


namespace find_n_l293_293935

theorem find_n : 
  ∃ (n : ℕ), (1986 + n - 255 = 1994) ∧ n = 312 :=
begin
  -- This is the statement.
  sorry
end

end find_n_l293_293935


namespace ceil_sum_of_sqr_roots_l293_293041

theorem ceil_sum_of_sqr_roots : 
  (⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27) := 
by {
  -- Definitions based on conditions
  have h1 : 1^2 < 3 ∧ 3 < 2^2, by norm_num,
  have h2 : 5^2 < 33 ∧ 33 < 6^2, by norm_num,
  have h3 : 18^2 < 333 ∧ 333 < 19^2, by norm_num,
  sorry
}

end ceil_sum_of_sqr_roots_l293_293041


namespace max_value_of_expression_l293_293663

noncomputable def maximum_value (x y z : ℝ) := 8 * x + 3 * y + 10 * z

theorem max_value_of_expression :
  ∀ (x y z : ℝ), 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 → maximum_value x y z ≤ (Real.sqrt 481) / 6 :=
by
  sorry

end max_value_of_expression_l293_293663


namespace problem_r_of_3_eq_88_l293_293251

def q (x : ℤ) : ℤ := 2 * x - 5
def r (x : ℤ) : ℤ := x^3 + 2 * x^2 - x - 4

theorem problem_r_of_3_eq_88 : r 3 = 88 :=
by
  sorry

end problem_r_of_3_eq_88_l293_293251


namespace jay_change_l293_293985

theorem jay_change (book_price pen_price ruler_price payment : ℕ) (h1 : book_price = 25) (h2 : pen_price = 4) (h3 : ruler_price = 1) (h4 : payment = 50) : 
(book_price + pen_price + ruler_price ≤ payment) → (payment - (book_price + pen_price + ruler_price) = 20) :=
by
  intro h
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end jay_change_l293_293985


namespace angle_between_vectors_l293_293569

variables {V : Type} [inner_product_space ℝ V]
variables (a b : V)

noncomputable def correct_angle (a b : V) : real :=
  if 2 * ‖a‖ = ‖b‖ then acos (1/2) else 0

theorem angle_between_vectors : 2 * ‖a‖ = ‖b‖ ∧ 2 • a - b ≠ 0 → 
  acos (inner_product_space.real_inner a b / (‖a‖ * ‖b‖)) = π / 6 :=
begin
  sorry
end

end angle_between_vectors_l293_293569


namespace coordinates_of_A_and_C_l293_293501

-- Definitions based on conditions
def B : ℝ × ℝ := (-1, 0)
def OB : ℝ := real.abs (B.1)
def OA : ℝ := 4 * OB
def OC : ℝ := 4 * OB

-- Definitions for A and C coordinates
def A : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (0, -4)

-- The statement to prove
theorem coordinates_of_A_and_C :
  OA = 4 * OB ∧ OC = 4 * OB ∧ A = (4, 0) ∧ C = (0, -4) :=
by
  sorry

end coordinates_of_A_and_C_l293_293501


namespace ceil_sum_of_sqr_roots_l293_293038

theorem ceil_sum_of_sqr_roots : 
  (⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27) := 
by {
  -- Definitions based on conditions
  have h1 : 1^2 < 3 ∧ 3 < 2^2, by norm_num,
  have h2 : 5^2 < 33 ∧ 33 < 6^2, by norm_num,
  have h3 : 18^2 < 333 ∧ 333 < 19^2, by norm_num,
  sorry
}

end ceil_sum_of_sqr_roots_l293_293038


namespace area_of_gray_region_l293_293367

theorem area_of_gray_region (d_s : ℝ) (h1 : d_s = 4) (h2 : ∀ r_s r_l : ℝ, r_s = d_s / 2 → r_l = 3 * r_s) : ℝ :=
by
  let r_s := d_s / 2
  let r_l := 3 * r_s
  let area_larger := π * r_l^2
  let area_smaller := π * r_s^2
  let area_gray := area_larger - area_smaller
  have hr_s : r_s = 2 := by sorry
  have hr_l : r_l = 6 := by sorry
  have ha_larger : area_larger = 36 * π := by sorry
  have ha_smaller : area_smaller = 4 * π := by sorry
  have ha_gray : area_gray = 32 * π := by sorry
  exact ha_gray

end area_of_gray_region_l293_293367


namespace sum_of_sequence_l293_293730

theorem sum_of_sequence:
  (∀ n ≥ 2, S (n) + S (n - 1) = 2 * n - 1) → S (2) = 3 → a (1) + a (3) = -1 :=
by
  intro h1 h2
  sorry

end sum_of_sequence_l293_293730


namespace largest_m_exists_l293_293666

noncomputable def quadratic_function (a b c : ℝ) (h : a ≠ 0) : ℝ → ℝ :=
λ x, a*x^2 + b*x + c

theorem largest_m_exists (a b c : ℝ) (h_a : a ≠ 0) 
    (h_symm : ∀ x : ℝ, quadratic_function a b c h_a (x - 4) = quadratic_function a b c h_a (2 - x))
    (h_ge_x : ∀ x : ℝ, quadratic_function a b c h_a x ≥ x)
    (h_le_bound : ∀ x, 0 < x ∧ x < 2 → quadratic_function a b c h_a x ≤ ((x + 1) / 2)^2)
    (h_inf_zero : ∀ y : ℝ, y = infi (λ x, quadratic_function a b c h_a x) → y = 0) :
    ∃ m > 1, ∃ t : ℝ, ∀ x ∈ set.Icc 1 m, quadratic_function a b c h_a (x + t) ≤ x :=
sorry

end largest_m_exists_l293_293666


namespace max_value_m_l293_293953

theorem max_value_m (a b : ℝ) (ha : a > 0) (hb : b > 0) (m : ℝ)
  (h : (2 / a) + (1 / b) ≥ m / (2 * a + b)) : m ≤ 9 :=
sorry

end max_value_m_l293_293953


namespace distance_between_A_and_B_l293_293604

noncomputable def distance_between_points (v_A v_B : ℝ) (t_meet t_A_to_B_after_meet : ℝ) : ℝ :=
  let t_total_A := t_meet + t_A_to_B_after_meet
  let t_total_B := t_meet + (t_meet - t_A_to_B_after_meet)
  let D := v_A * t_total_A + v_B * t_total_B
  D

-- Given conditions
def t_meet : ℝ := 4
def t_A_to_B_after_meet : ℝ := 3
def speed_difference : ℝ := 20

-- Function to calculate speeds based on given conditions
noncomputable def calculate_speeds (v_B : ℝ) : ℝ × ℝ :=
  let v_A := v_B + speed_difference
  (v_A, v_B)

-- Statement of the problem in Lean 4
theorem distance_between_A_and_B : ∃ (v_B v_A : ℝ), 
  v_A = v_B + speed_difference ∧
  distance_between_points v_A v_B t_meet t_A_to_B_after_meet = 240 :=
by 
  sorry

end distance_between_A_and_B_l293_293604


namespace find_b_l293_293652

open_locale matrix

def a : ℝ^3 := ![5, -3, -6]
def c : ℝ^3 := ![-3, -2, 3]
def b : ℝ^3 := ![-1, -3/4, 3/4]

theorem find_b (h1 : ∃ k : ℝ, b = k • a ∨ b = k • c)
(h2: inner_product_space.angle a b = inner_product_space.angle b c):
  b = ![-1, -3/4, 3/4] :=
sorry

end find_b_l293_293652


namespace minimum_value_l293_293135

theorem minimum_value (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 2) 
  (h4 : a + b = 1) : 
  ∃ L, L = (3 * a * c / b) + (c / (a * b)) + (6 / (c - 2)) ∧ L = 1 / (a * (1 - a)) := sorry

end minimum_value_l293_293135


namespace factorization_correct_l293_293064

-- Define the expression
def expression (x : ℝ) : ℝ := x^2 + 2 * x

-- State the theorem to prove the factorized form is equal to the expression
theorem factorization_correct (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by {
  -- Lean will skip the proof because of sorry, ensuring the statement compiles correctly.
  sorry
}

end factorization_correct_l293_293064


namespace sundae_cost_l293_293582

def ice_cream_cost := 2.00
def topping_cost := 0.50
def number_of_toppings := 10

theorem sundae_cost : ice_cream_cost + topping_cost * number_of_toppings = 7.00 := 
by
  sorry

end sundae_cost_l293_293582


namespace no_a_for_x4_l293_293590

theorem no_a_for_x4 : ∃ a : ℝ, (1 / (4 + a) + 1 / (4 - a) = 1 / (4 - a)) → false :=
  by sorry

end no_a_for_x4_l293_293590


namespace align_circles_l293_293168

theorem align_circles (C : ℝ) (n : ℕ) (A : ℝ) :
  C = 100 ∧ n = 100 ∧ A < 1 →
  ∃ θ : ℝ, ∀ i : ℕ, i < n → θ ∉ set.range (λ x, x + i * (C / n)) :=
by
  sorry

end align_circles_l293_293168


namespace conic_sections_of_equation_l293_293030

theorem conic_sections_of_equation (x y : ℝ) :
  y^4 - 9 * x^4 = 3 * y^2 - 1 →
  (∃ a b : ℝ, y^2 - a * x^2 = b ∧ conic_section (y^2 - a * x^2 = b) = conic_section.hyperbola) ∧
  (∃ c d : ℝ, y^2 + c * x^2 = d ∧ conic_section (y^2 + c * x^2 = d) = conic_section.ellipse) :=
by
  sorry

end conic_sections_of_equation_l293_293030


namespace derivative_evaluation_at_pi_over_3_l293_293923

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2 * x) + Real.tan x

theorem derivative_evaluation_at_pi_over_3 :
  deriv f (Real.pi / 3) = 3 :=
sorry

end derivative_evaluation_at_pi_over_3_l293_293923


namespace max_palindromic_even_extensions_l293_293180

-- Define the conditions for the telephone extensions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_palindrome (n : ℕ) : Prop :=
  let digits := (List.ofDigits 10 (List.digits 10 n));
  digits = digits.reverse

-- The main theorem we want to prove
theorem max_palindromic_even_extensions : 
  let digits := [1, 2, 3, 8, 9],
  let extensions := { n : ℕ | n < 10000 ∧ 
                               ∀ d, d ∈ List.digits 10 n → d ∈ digits ∧ 
                               (is_even n) ∧ 
                               (is_palindrome n) 
                             },
  -- It is required in the Lean theorem statement that we represent the cardinality (number)
  -- of elements satisfying the conditions. Thus, the type of extensions is a set.
  extensions.card ≤ 12 ∧ ∃ n ∈ extensions, extensions.card = 12 := by sorry

end max_palindromic_even_extensions_l293_293180


namespace count_even_integers_with_unique_digits_l293_293576

theorem count_even_integers_with_unique_digits :
  let S := {0, 1, 3, 4, 6, 7}
  ∃ n : ℕ, n = 40 ∧
    card {x : ℕ | 300 ≤ x ∧ x < 800 ∧ 
      (∃ a b c : ℕ, x = a * 100 + b * 10 + c ∧
      a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      c % 2 = 0)} = n :=
by
  use 40
  sorry -- proof goes here

end count_even_integers_with_unique_digits_l293_293576


namespace identify_heaviest_and_lightest_coin_in_13_weighings_l293_293310

theorem identify_heaviest_and_lightest_coin_in_13_weighings :
  ∀ (coins : Finₓ 10 → ℝ) 
    (balance_weighing : ∀ (a b : Finₓ 10), Prop), 
    (∀ i j, coins i ≠ coins j) → 
    (∃ strategy : ℕ → (Finₓ 10 × Finₓ 10),
      ∃ h : ℕ,
        h ≤ 13 ∧
        (∃ heaviest lightest : Finₓ 10,
          (∀ i, coins heaviest ≥ coins i) ∧ (∀ j, coins lightest ≤ coins j))) :=
by
  sorry

end identify_heaviest_and_lightest_coin_in_13_weighings_l293_293310


namespace locus_inside_sphere_G1_l293_293893

variables (G : sphere) (O : point) (r : ℝ) (P : point) (AB : chord) (F : point)

def is_midpoint (F : point) (A B : point) : Prop := dist O F = r / 2 ∧ midpoint O A B = F

noncomputable def locus_of_points : set point :=
  {P | ∃ (A B : point), is_midpoint F A B ∧ dist P F < dist A F / 2}

theorem locus_inside_sphere_G1 (G : sphere) (O : point) (r : ℝ) :
  ∀ P : point, (∃ A B : point, is_midpoint F A B ∧ dist P F < dist A F / 2) → dist O P < sqrt 2 * r :=
  sorry

end locus_inside_sphere_G1_l293_293893


namespace Petya_wins_optimal_play_l293_293680

-- Define the game conditions
def checkerboard_size (n : ℕ) : Prop :=
  n > 1

inductive Cell
| white : Cell
| black : Cell

structure Board := (cells : List (List Cell))

def initial_board (n : ℕ) : Board :=
  Board.mk ((List.repeat Cell.black 1 :: List.repeat (List.repeat Cell.white (n - 1)) (n - 1)))

structure Rook := (location : (ℕ × ℕ))

structure Game :=
  (board : Board)
  (rook : Rook)
  (petya_turn : Bool)

-- Function to check if a rook move is legal
def is_legal_move (board : Board) (rook : Rook) (to : ℕ × ℕ) : Prop :=
  let (i, j) := rook.location
  let (r, c) := to
  (i = r ∨ j = c) ∧
  (i ≠ r ∨ j ≠ c) ∧
  List.all (List.map (fun k => board.cells (if i = r then i else k) (if i = r then k else j)) (List.range (abs (i - r) + abs (j - c) + 1))) (λ cell => cell = Cell.white)

-- Transition function
def move_rook (g : Game) (to : ℕ × ℕ) : Game :=
  let (i, j) := g.rook.location
  let Board.cells := g.board.cells
  let new_cells := 
    if i = to.1 then
      cells.map_with_index (λ row_idx row => if row_idx = i then row.map_with_index (λ col_idx col => if col_idx ≤ max j to.2 ∧ col_idx ≥ min j to.2 then Cell.black else col) else row)
    else
      cells.map_with_index (λ row_idx row => if row_idx ≥ min i to.1 ∧ row_idx <= max i to.1 then row.map_with_index (λ col_idx col => if col_idx = j then Cell.black else col) else row)
  { g with board := { cells := new_cells }, rook := { location := to }, petya_turn := not g.petya_turn }

-- Function to check if there are any legal moves left
def has_moves (g : Game) : Bool :=
  let possible_moves := List.bind (List.range (g.board.cells.length)) (λ i => List.bind (List.range (g.board.cells.head.length)) (λ j => if h : is_legal_move g.board g.rook (i, j) then [(i, j)] else [] ))
  ¬ possible_moves.isEmpty

-- Define the game-winning strategy
def optimal_play_win (g : Game) : Bool :=
  if ¬has_moves g then g.petya_turn ∨ (¬g.petya_turn ∧ has_moves (move_rook g (g.rook.location))) else
  g.petya_turn ∧ ∀i j, is_legal_move g.board g.rook (i, j) → ¬has_moves (move_rook g (i, j)) 

-- Theorem: Petya wins with optimal play
theorem Petya_wins_optimal_play (n : ℕ) (h : checkerboard_size n) : optimal_play_win { board := initial_board n, rook := { location := (0, 0) }, petya_turn := true } :=
sorry

end Petya_wins_optimal_play_l293_293680


namespace find_ratio_of_arithmetic_sequences_l293_293905

variable {a_n b_n : ℕ → ℕ}
variable {A_n B_n : ℕ → ℝ}

def arithmetic_sums (a_n b_n : ℕ → ℕ) (A_n B_n : ℕ → ℝ) : Prop :=
  ∀ n, A_n n = (n * (2 * a_n 1 + (n - 1) * (a_n 8 - a_n 7))) / 2 ∧
         B_n n = (n * (2 * b_n 1 + (n - 1) * (b_n 8 - b_n 7))) / 2

theorem find_ratio_of_arithmetic_sequences 
    (h : ∀ n, A_n n / B_n n = (5 * n - 3) / (n + 9)) :
    ∃ r : ℝ, r = 3 := by
  sorry

end find_ratio_of_arithmetic_sequences_l293_293905


namespace complex_roots_real_l293_293743

theorem complex_roots_real (z : ℂ) (h : z^30 = 1) : 
  {z : ℂ | z^30 = 1}.count (λ z, z^5 ∈ ℝ) = 10 :=
sorry

end complex_roots_real_l293_293743


namespace largest_number_in_sample_l293_293103

theorem largest_number_in_sample
  (n : ℕ) (students : fin n) (sampling_interval : ℕ) (first_sample second_sample largest_sample : ℕ)
  (h1 : n = 5000)
  (h2 : students = fin.of_nat n)
  (h3 : first_sample = 18)
  (h4 : second_sample = 68)
  (h5 : sampling_interval = second_sample - first_sample)
  (h6 : sampling_interval > 0)
  (h7 : largest_sample = first_sample + sampling_interval * (5000 / sampling_interval - 1)) :
  largest_sample = 4968 :=
by
  sorry

end largest_number_in_sample_l293_293103


namespace parabola_symmetry_product_l293_293967

theorem parabola_symmetry_product (a p m : ℝ) 
  (hpr1 : a ≠ 0) 
  (hpr2 : p > 0) 
  (hpr3 : ∀ (x₀ y₀ : ℝ), y₀^2 = 2*p*x₀ → (a*(y₀ - m)^2 - 3*(y₀ - m) + 3 = x₀ + m)) :
  a * p * m = -3 := 
sorry

end parabola_symmetry_product_l293_293967


namespace ratio_blue_gill_to_bass_l293_293421

theorem ratio_blue_gill_to_bass (bass trout blue_gill : ℕ) 
  (h1 : bass = 32)
  (h2 : trout = bass / 4)
  (h3 : bass + trout + blue_gill = 104) 
: blue_gill / bass = 2 := 
sorry

end ratio_blue_gill_to_bass_l293_293421


namespace rectangle_perimeter_of_equal_area_l293_293347

theorem rectangle_perimeter_of_equal_area (a b c : ℕ) (area_triangle width length : ℕ) :
  a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2 ∧ (2 * area_triangle = a * b) ∧
  (width = 6) ∧ (area_triangle = width * length) -> 
  2 * (length + width) = 30 :=
by
  intros h,
  sorry

end rectangle_perimeter_of_equal_area_l293_293347


namespace number_of_pairs_arithmetic_progression_l293_293844

theorem number_of_pairs_arithmetic_progression : 
  ∃ (x y : ℤ), (∃ x y, (x = 6 + y / 2) ∧ (y = -2 ∨ y = -6)) ∧ (x + xy = 2y) :=
by
  sorry

end number_of_pairs_arithmetic_progression_l293_293844


namespace max_ones_in_grid_l293_293206

theorem max_ones_in_grid : ∀ (grid : Fin 9 × Fin 9 → ℕ),
  (∀ i j, (grid ⟨i.val, sorry⟩ = 0 ∨ grid ⟨i.val, sorry⟩ = 1) ∧
           (i.val < 9 ∧ j.val < 9) →
  (∀ x y, (0 ≤ x ∧ x < 8 ∧ 0 ≤ y ∧ y < 8) →
            (((grid ⟨x, sorry⟩ + grid ⟨x+1, sorry⟩ + grid ⟨x, sorry⟩ + grid ⟨x+1, sorry⟩) % 2 = 1))) →
  (∑ x, ∑ y, grid ⟨x, sorry⟩  ≤ 65) := sorry

end max_ones_in_grid_l293_293206


namespace eval_sum_sqrt_ceil_l293_293043

theorem eval_sum_sqrt_ceil:
  ∀ (x : ℝ), 
  (1 < sqrt 3 ∧ sqrt 3 < 2) ∧
  (5 < sqrt 33 ∧ sqrt 33 < 6) ∧
  (18 < sqrt 333 ∧ sqrt 333 < 19) →
  (⌈ sqrt 3 ⌉ + ⌈ sqrt 33 ⌉ + ⌈ sqrt 333 ⌉ = 27) :=
by
  intro x
  sorry

end eval_sum_sqrt_ceil_l293_293043


namespace reduced_price_is_16_l293_293752

noncomputable def reduced_price_per_kg (P : ℝ) (r : ℝ) : ℝ :=
  0.9 * (P * (1 + r))

theorem reduced_price_is_16 (P r : ℝ) (h₀ : (0.9 : ℝ) * (P * (1 + r)) = 16) : 
  reduced_price_per_kg P r = 16 :=
by
  -- We have the hypothesis and we need to prove the result
  exact h₀

end reduced_price_is_16_l293_293752


namespace ceil_sum_of_sqr_roots_l293_293037

theorem ceil_sum_of_sqr_roots : 
  (⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27) := 
by {
  -- Definitions based on conditions
  have h1 : 1^2 < 3 ∧ 3 < 2^2, by norm_num,
  have h2 : 5^2 < 33 ∧ 33 < 6^2, by norm_num,
  have h3 : 18^2 < 333 ∧ 333 < 19^2, by norm_num,
  sorry
}

end ceil_sum_of_sqr_roots_l293_293037


namespace equal_pair_c_l293_293460

theorem equal_pair_c : (-4)^3 = -(4^3) := 
by {
  sorry
}

end equal_pair_c_l293_293460


namespace sequence_formula_l293_293724

noncomputable def S_n (n : ℕ) : ℝ := 2 * n - (1 + (2 ^ n - 1) / (2 ^ (n - 1)))

theorem sequence_formula (n : ℕ) (hn : n > 0) : 
  let a_n := (2^n - 1) / (2^(n-1)) in
  S_n n = 2 * n - a_n := 
by 
  sorry

end sequence_formula_l293_293724


namespace no_real_roots_range_l293_293138

theorem no_real_roots_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 ≠ 0) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end no_real_roots_range_l293_293138


namespace passengers_taken_at_second_station_l293_293815

noncomputable def initial_passengers : ℕ := 270
noncomputable def passengers_dropped_first_station := initial_passengers / 3
noncomputable def passengers_after_first_station := initial_passengers - passengers_dropped_first_station + 280
noncomputable def passengers_dropped_second_station := passengers_after_first_station / 2
noncomputable def passengers_after_second_station (x : ℕ) := passengers_after_first_station - passengers_dropped_second_station + x
noncomputable def passengers_at_third_station := 242

theorem passengers_taken_at_second_station : ∃ x : ℕ,
  passengers_after_second_station x = passengers_at_third_station ∧ x = 12 :=
by
  sorry

end passengers_taken_at_second_station_l293_293815


namespace area_of_triangle_ABC_l293_293627

-- Definitions according to the given conditions
def small_triangle_area : ℝ := 1
def number_of_small_triangles : ℝ := 24
def total_area : ℝ := number_of_small_triangles * small_triangle_area

-- Subtractions as per solution steps
def subtracted_area : ℝ := (1/2 * 4) + (1/2 * 6) + 1 + (1/2 * 4) + 6

-- Correct final area of triangle ABC
def final_area : ℝ := total_area - subtracted_area

-- Proof statement
theorem area_of_triangle_ABC : final_area = 10 := 
by {
  -- A placeholder for the proof. In practice, one would replace 'sorry' with the actual proof.
  sorry
}

end area_of_triangle_ABC_l293_293627


namespace license_plate_combinations_l293_293472

-- Definitions representing the conditions
def valid_license_plates_count : ℕ :=
  let letter_combinations := Nat.choose 26 2 -- Choose 2 unique letters
  let letter_arrangements := Nat.choose 4 2 * 2 -- Arrange the repeated letters
  let digit_combinations := 10 * 9 * 8 -- Choose different digits
  letter_combinations * letter_arrangements * digit_combinations

-- The theorem representing the problem statement
theorem license_plate_combinations :
  valid_license_plates_count = 2808000 := 
  sorry

end license_plate_combinations_l293_293472


namespace factorize_x_squared_plus_2x_l293_293058

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) :=
by sorry

end factorize_x_squared_plus_2x_l293_293058


namespace subset_S_A_inter_B_nonempty_l293_293255

open Finset

-- Definitions of sets A and B
def A : Finset ℕ := {1, 2, 3, 4, 5, 6}
def B : Finset ℕ := {4, 5, 6, 7, 8}

-- Definition of the subset S and its condition
def S : Finset ℕ := {5, 6}

-- The statement to be proved
theorem subset_S_A_inter_B_nonempty : S ⊆ A ∧ S ∩ B ≠ ∅ :=
by {
  sorry -- proof to be provided
}

end subset_S_A_inter_B_nonempty_l293_293255


namespace acute_triangle_incorrect_option_l293_293200

theorem acute_triangle_incorrect_option (A B C : ℝ) (hA : 0 < A ∧ A < 90) (hB : 0 < B ∧ B < 90) (hC : 0 < C ∧ C < 90)
  (angle_sum : A + B + C = 180) (h_order : A > B ∧ B > C) : ¬(B + C < 90) :=
sorry

end acute_triangle_incorrect_option_l293_293200


namespace pyramid_sectional_areas_relationship_l293_293810

theorem pyramid_sectional_areas_relationship
    (base_area : ℝ) (S_1 S_2 S_3 : ℝ)
    (h1 : S_1 = (base_area / 4))
    (h2 : S_1 < S_2) 
    (h3 : S_2 < (base_area / 2))
    (h4 : S_2 < S_3) 
    (h5 : S_3 = base_area * (∛ 2)^2 / 4) :
    S_1 < S_2 ∧ S_2 < S_3 :=
by {
    sorry
}

end pyramid_sectional_areas_relationship_l293_293810


namespace assignment_count_l293_293674

def set_of_geologists : Finset ℕ := {0, 1, 2, 3, 4, 5}
def set_of_schools : Finset ℕ := {0, 1, 2, 3}

theorem assignment_count : 
  (∑ t in (set_of_geologists.powerset.filter (λ s, s.card = 3)), 
    ∑ u in ((set_of_geologists \ t).powerset.filter (λ s, s.card = 2)), 
      (set_of_geologists \ t \ u).card = 1) +
  (∑ t in (set_of_geologists.powerset.filter (λ s, s.card = 2)), 
    ∑ u in ((set_of_geologists \ t).powerset.filter (λ s, s.card = 2)),
      (set_of_geologists \ t \ u).card = 2) =
  1560 :=
sorry

end assignment_count_l293_293674


namespace simplify_fraction_l293_293272

theorem simplify_fraction :
  (5 : ℚ) / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 27) = Real.sqrt 3 / 12 := by
sorry

end simplify_fraction_l293_293272


namespace cube_wire_not_possible_minimum_cuts_to_construct_cube_l293_293399

theorem cube_wire_not_possible
  (wire_length : ℝ) (cube_edge : ℝ) (cube_edges : ℕ)
  (vertices : ℕ)
  (vertex_degree : ℕ) :
  wire_length = 120 → cube_edge = 10 → cube_edges = 12 → vertices = 8 → vertex_degree = 3 →
  ¬ (exists (path : ℝ → ℝ), path 0 = 0 ∧ path 120 = 120 ∧ (∀ t, t ∈ set.Icc 0 120 → 
  (∃ (i j : ℕ), i ≠ j ∧ (abs (path t - path (t+ cube_edge / 2))) = 10)) ∧
  (∀ t, t ∈ set.Icc 0 120 → (abs (path t - path (t + cube_edge / 2) : ℝ) = 10 → (t + cube_edge / 2) ∈ set.Icc 0 120))) :=
by sorry

theorem minimum_cuts_to_construct_cube
  (vertices : ℕ)
  (vertex_degree : ℕ) :
  vertices = 8 → vertex_degree = 3 → 4 :=
by sorry

end cube_wire_not_possible_minimum_cuts_to_construct_cube_l293_293399


namespace circle_tangent_ratio_l293_293778

theorem circle_tangent_ratio
  (O : Type) [metric_space O]
  (A B X T1 T2 S1 S2 : O) 
  (H_AB_diameter : ∀ (C: O), dist C A = dist C B)
  (H_AX_3BX : dist A X = 3 * dist B X)
  (tangent_to_circle_O : ∀ (ω : Type), ∃ (T : O), ω = T ∨ ω = ~T)
  (lines_intersect_at : ∀ (l1 l2 : O), exists S : O, l1 = l2 → dist S1 S2 = dist A B)
  : dist T1 T2 / dist S1 S2 = (3 : ℝ) / 5 :=
sorry

end circle_tangent_ratio_l293_293778


namespace rectangle_perimeter_is_30_l293_293353

noncomputable def triangle_DEF_sides := (9 : ℕ, 12 : ℕ, 15 : ℕ)
noncomputable def rectangle_width := (6 : ℕ)

theorem rectangle_perimeter_is_30 :
  let area_triangle_DEF := (triangle_DEF_sides.1 * triangle_DEF_sides.2) / 2
  let rectangle_length := area_triangle_DEF / rectangle_width
  let rectangle_perimeter := 2 * (rectangle_width + rectangle_length)
  rectangle_perimeter = 30 := by
  sorry

end rectangle_perimeter_is_30_l293_293353


namespace seq_form_l293_293535

def seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else 3 * 4^(n-2)

theorem seq_form : ∀ n : ℕ, 
  (n = 1 → seq n = 1) ∧ (n ≥ 2 → seq n = 3 * 4^(n-2)) :=
by
  intro n
  split
  → intro h1
    rw [h1]
    simp only [seq]
  → intro h2
    simp only [seq]
    sorry

end seq_form_l293_293535


namespace distinct_digit_sum_l293_293592

theorem distinct_digit_sum (a b c d : ℕ) (h1 : a + c = 10) (h2 : b + c = 9) (h3 : a + d = 1)
  (h4 : a ≠ b) (h5 : a ≠ c) (h6 : a ≠ d) (h7 : b ≠ c) (h8 : b ≠ d) (h9 : c ≠ d)
  (h10 : a < 10) (h11 : b < 10) (h12 : c < 10) (h13 : d < 10)
  (h14 : 0 ≤ a) (h15 : 0 ≤ b) (h16 : 0 ≤ c) (h17 : 0 ≤ d) :
  a + b + c + d = 18 :=
sorry

end distinct_digit_sum_l293_293592


namespace relation_between_a_b_c_l293_293900

def a : ℝ := (Real.sqrt 2) / 2 * (Real.sin (Real.pi * 17 / 180) + Real.cos (Real.pi * 17 / 180))
def b : ℝ := 2 * (Real.cos (Real.pi * 13 / 180))^2 - 1
def c : ℝ := Real.sin (Real.pi * 37 / 180) * Real.sin (Real.pi * 67 / 180) + Real.sin (Real.pi * 53 / 180) * Real.sin (Real.pi * 23 / 180)

theorem relation_between_a_b_c : c < a ∧ a < b :=
by
  sorry

end relation_between_a_b_c_l293_293900


namespace measure_of_angle_A_values_of_b_and_c_l293_293957

variable (a b c : ℝ) (A : ℝ)

-- Declare the conditions as hypotheses
def condition1 (a b c : ℝ) := a^2 - c^2 = b^2 - b * c
def condition2 (a : ℝ) := a = 2
def condition3 (b c : ℝ) := b + c = 4

-- Proof that A = 60 degrees when the conditions are satisfied
theorem measure_of_angle_A (h : condition1 a b c) : A = 60 := by
  sorry

-- Proof that b and c are 2 when given conditions are satisfied
theorem values_of_b_and_c (h1 : condition1 2 b c) (h2 : condition3 b c) : b = 2 ∧ c = 2 := by
  sorry

end measure_of_angle_A_values_of_b_and_c_l293_293957


namespace subproblem1_subproblem2_l293_293109

-- Sub-Problem (I)
theorem subproblem1 (x y r : ℝ) (M : ℝ × ℝ) (l : ℝ → ℝ) 
  (hC : ∀ x y, x^2 + (y - 4)^2 = r^2) 
  (hM : M = (-2, 0)) 
  (hr : r = 2) 
  (hl : ∀ x, l (-2) = 0) :
  l = ((λ x, -2) ∨ l = (λ x, (3 / 4) x + 3 / 4 * 2)) := sorry

-- Sub-Problem (II)
theorem subproblem2 (x y r : ℝ) (M : ℝ × ℝ) (l : ℝ → ℝ) 
  (hC : ∀ x y, x^2 + (y - 4)^2 = r^2) 
  (hM : M = (-2, 0)) 
  (hα : ∀ x, l = λ x, -x - 2) 
  (hAB : ∀ A B, dist A B = 2 * sqrt(2)) :
  r^2 = 20 := sorry

end subproblem1_subproblem2_l293_293109


namespace determine_q_l293_293277

noncomputable def q (x : ℝ) : ℝ := x^3 - (58/13) * x^2 + (49/13) * x + 20

theorem determine_q :
  (∃ (p : ℝ → ℝ), (∀ x : ℝ, q(x) = p(x)) ∧ -- Asserts that there exists a polynomial p equivalent to q
                (p(3 - 2*I) = 0) ∧           -- Condition that 3-2i is a root
                (p(3 + 2*I) = 0) ∧           -- Implied condition, equivalent to real coefficients property
                (p(0) = -20)) := 
begin
  use q,
  split,
  {
    intro x,
    refl,
  },
  {
    split,
    {
      -- Here you would prove that q(3-2i) = 0
      sorry
    },
    {
      split,
      {
        -- Here you would prove that q(3+2i) = 0
        sorry
      },
      {
        -- Here you would prove that q(0) = -20
        sorry
      }
    }
  }
end

end determine_q_l293_293277


namespace MrsSantiagoHas58Roses_l293_293672

variable (rosesGarrett rosesSantiago : ℕ)

-- Conditions as definitions in Lean
def MrsGarrettRoses := 24
def moreRoses := 34
def MrsSantiagoRoses := rosesGarrett + moreRoses

-- Theorem statement
theorem MrsSantiagoHas58Roses :
  MrsGarrettRoses = 24 →
  moreRoses = 34 →
  rosesSantiago = MrsGarrettRoses + moreRoses →
  rosesSantiago = 58 := by
  intros
  rw [‹MrsGarrettRoses = 24›, ‹moreRoses = 34›]
  sorry

end MrsSantiagoHas58Roses_l293_293672


namespace carlos_cycles_more_than_diana_l293_293282

theorem carlos_cycles_more_than_diana :
  let slope_carlos := 1
  let slope_diana := 0.75
  let rate_carlos := slope_carlos * 20
  let rate_diana := slope_diana * 20
  let distance_carlos_after_3_hours := 3 * rate_carlos
  let distance_diana_after_3_hours := 3 * rate_diana
  distance_carlos_after_3_hours - distance_diana_after_3_hours = 15 :=
sorry

end carlos_cycles_more_than_diana_l293_293282


namespace at_least_one_f_nonnegative_l293_293356

theorem at_least_one_f_nonnegative 
  (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m * n > 1) : 
  (m^2 - m ≥ 0) ∨ (n^2 - n ≥ 0) :=
by sorry

end at_least_one_f_nonnegative_l293_293356


namespace geometric_sequence_sum_l293_293241

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

variables {a : ℕ → ℝ}

theorem geometric_sequence_sum (h1 : is_geometric_sequence a) (h2 : a 1 * a 2 = 8 * a 0)
  (h3 : (a 3 + 2 * a 4) / 2 = 20) :
  (a 0 * (2^5 - 1)) = 31 :=
by
  sorry

end geometric_sequence_sum_l293_293241


namespace simplify_polynomial_l293_293269

def polynomial1 := 2 * x ^ 5 - 3 * x ^ 4 + x ^ 3 + 5 * x ^ 2 - 2 * x + 8
def polynomial2 := x ^ 4 - 2 * x ^ 3 + 3 * x ^ 2 + 4 * x - 16
def simplifiedPolynomial := 2 * x ^ 5 - 2 * x ^ 4 - x ^ 3 + 8 * x ^ 2 + 2 * x - 8 

theorem simplify_polynomial : 
  polynomial1 + polynomial2 = simplifiedPolynomial :=
by
  sorry

end simplify_polynomial_l293_293269


namespace num_divisors_1200_l293_293221

def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem num_divisors_1200 : 
  ∃ (count : ℕ), count = (List.range 1200).count (λ n, is_divisor n 1200) ∧ count = 30 :=
by
  sorry

end num_divisors_1200_l293_293221


namespace no_possible_arrangement_l293_293701

theorem no_possible_arrangement :
  ¬ ∃ (a : Fin 9 → ℕ),
    (∀ i, 1 ≤ a i ∧ a i ≤ 9) ∧
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    (∀ i, (a i + a ((i + 1) % 9) + a ((i + 2) % 9)) % 3 = 0) ∧
    (∀ i, (a i + a ((i + 1) % 9) + a ((i + 2) % 9)) > 12) :=
  sorry

end no_possible_arrangement_l293_293701


namespace length_AD_l293_293337

noncomputable theory
open Classical

variables (A B C D O P : ℝ × ℝ)
variables (BC CD AD : ℝ)
variables (trapezoid : Prop)
variables (is_midpoint : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop)
variables (is_perpendicular : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop)

def trapezoid_ABCD : Prop :=
  let BC := dist B C in
  let CD := dist C D in
  let AD := dist A D in
  BC = 39 ∧ CD = 39 ∧
  (∃ O, ∃ P, O = midpoint A C ∧ P = midpoint B C ∧ OP = 10 ∧ is_perpendicular A D B C)

theorem length_AD (h : trapezoid_ABCD A B C D O P BC CD AD is_midpoint is_perpendicular) :
  AD = 5 * sqrt 76 :=
sorry

end length_AD_l293_293337


namespace identify_heaviest_and_lightest_coin_within_13_weighings_l293_293318

-- Lean 4 statement to encapsulate the given problem
theorem identify_heaviest_and_lightest_coin_within_13_weighings :
  ∃ (weighings: list (ℕ × ℕ)) (heaviest lightest: ℕ),
    (length weighings ≤ 13) ∧
    (∀ i j, 1 ≤ i ∧ i ≤ 10 → 1 ≤ j ∧ j ≤ 10 → i ≠ j) ∧
    (∀ (comp: ℕ × ℕ), comp ∈ weighings → comp.1 ≠ comp.2 ∧ 1 ≤ comp.1 ∧ comp.1 ≤ 10 ∧ 1 ≤ comp.2 ∧ comp.2 ≤ 10) ∧
    heaviest ≠ lightest ∧
    (∀ (i: ℕ), 1 ≤ i ∧ i ≤ 10 → 
      (i = heaviest ∨ i = lightest))
: sorry

end identify_heaviest_and_lightest_coin_within_13_weighings_l293_293318


namespace measure_of_angle_YZX_l293_293012

noncomputable def problem_statement : Prop :=
  ∃ (A B C X Y Z : ℝ),
    X ∈ (segment B C) ∧
    Y ∈ (segment A B) ∧
    Z ∈ (segment A C) ∧
    angle A = 50 ∧
    angle B = 70 ∧
    angle C = 60 ∧
    let Γ := incircle ABC in
    let γ := circumcircle XYZ in
    incircle ABC = Γ ∧ 
    circumcircle XYZ = γ ∧ 
    ∠YZX = 115

theorem measure_of_angle_YZX (A B C X Y Z : ℝ) :
  X ∈ (segment B C) →
  Y ∈ (segment A B) →
  Z ∈ (segment A C) →
  angle A = 50 →
  angle B = 70 →
  angle C = 60 →
  let Γ := incircle ABC in
  let γ := circumcircle XYZ in
  incircle ABC = Γ →
  circumcircle XYZ = γ →
  ∠YZX = 115 :=
by
  sorry

end measure_of_angle_YZX_l293_293012


namespace central_angle_is_180_l293_293601

noncomputable def central_angle_of_sector (r l: ℝ) (A_l A_b: ℝ) (hlateral: A_l = π * r * l)
  (hbase: A_b = π * r^2) (harea: A_l = 2 * A_b) : ℝ :=
  let α := 360 * (2 * π * r) / (2 * π * l) in α

theorem central_angle_is_180 (r l: ℝ) (A_l A_b: ℝ) (hl: l = 2 * r)
  (hlateral: A_l = π * r * l) (hbase: A_b = π * r^2) (harea: A_l = 2 * A_b) :
  central_angle_of_sector r l A_l A_b hlateral hbase harea = 180 :=
by
  sorry

end central_angle_is_180_l293_293601


namespace probability_at_least_half_girls_l293_293219

theorem probability_at_least_half_girls (n : ℕ) (hn : n = 6) :
  (∑ k in finset.range (n + 1), if k ≥ 3 then nat.choose n k * (1 / 2)^n else 0) = 21 / 32 :=
by
  sorry

end probability_at_least_half_girls_l293_293219


namespace average_of_shifted_sample_l293_293167

theorem average_of_shifted_sample (x1 x2 x3 : ℝ) (hx_avg : (x1 + x2 + x3) / 3 = 40) (hx_var : ((x1 - 40) ^ 2 + (x2 - 40) ^ 2 + (x3 - 40) ^ 2) / 3 = 1) : 
  ((x1 + 40) + (x2 + 40) + (x3 + 40)) / 3 = 80 :=
sorry

end average_of_shifted_sample_l293_293167


namespace sum_of_possible_b_values_l293_293489

/--
Given a quadratic function g(x) = x^2 - b * x + 3 * b,
Prove that the sum of all possible values of b for which g(x) has integer zeroes is 8.
-/
theorem sum_of_possible_b_values : 
  (∑ (b : ℤ) in {b : ℤ | ∃ r s : ℤ, r + s = b ∧ r * s = 3 * b}, b) = 8 := by
  sorry

end sum_of_possible_b_values_l293_293489


namespace circumcircles_touch_l293_293827

theorem circumcircles_touch
  (ABC : Type*)
  [triangle ABC]
  (I : incenter ABC)
  (A1 B1 C1 : Point)
  (AA1 : line A A1)
  (BB1 : line B B1)
  (CC1 : line C C1)
  (A0 C0 : Point)
  (perpendicular_bisector_BB1 : ∀ (P : Point), P ∈ perp_bisector B B1 → P ∈ AA1 ∨ P ∈ CC1)
  (circumcircle_ABC : Circle ABC)
  (circumcircle_A0IC0 : Circle A0 I C0) :
  tangent circumcircle_ABC circumcircle_A0IC0 :=
sorry

end circumcircles_touch_l293_293827


namespace arithmetic_sequence_product_l293_293542

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n+1) = a n + d

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n + 1) * (a 0) + ((n * (n + 1)) / 2) * (a 1 - a 0)

theorem arithmetic_sequence_product (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_seq : is_arithmetic_sequence a)
  (h_sum : ∀ n, S n = sequence_sum a n)
  (h_a7 : a 7 < 0)
  (h_a8 : a 8 > 0)
  (h_a8_gt_abs_a7 : a 8 > |a 7|) :
  (S 13) * (S 14) < 0 := by
  sorry

end arithmetic_sequence_product_l293_293542


namespace num_ways_fill_boxes_l293_293751

theorem num_ways_fill_boxes :
  (∃ f : Fin 6 → Bool, (∃ i : Fin 6, f i = tt) ∧
  ∀ i j : Fin 6, (i ≤ j - 1) → (f i = tt → f j = tt)) →
  (finset.card {f : Fin 6 → Bool | (∃ i : Fin 6, f i = tt) ∧
  ∀ i j : Fin 6, (i ≤ j - 1) → (f i = tt → f j = tt)} = 21) :=
by {
  sorry
}

end num_ways_fill_boxes_l293_293751


namespace max_value_of_c_holds_l293_293907

noncomputable def c_max : ℝ := ( (real.sqrt 6 + 3 * real.sqrt 2) / 2 ) * real.sqrt 3 ^ (1/4)

theorem max_value_of_c_holds (x y z : ℝ) (h : x ≥ 0) (h2: y ≥ 0) (h3: z ≥ 0) :
  x^3 + y^3 + z^3 - 3 * x * y * z ≥ c_max * (abs ((x - y) * (y - z) * (z - x))) :=
sorry

end max_value_of_c_holds_l293_293907


namespace jack_gallons_per_batch_l293_293636

noncomputable def ounces_per_2_days : ℕ := 96
noncomputable def time_to_make_coffee : ℕ := 20
noncomputable def total_hours_making_coffee : ℕ := 120
noncomputable def total_days : ℕ := 24
noncomputable def ounces_per_gallon : ℕ := 128

theorem jack_gallons_per_batch :
  (ounces_per_2_days / 2 * total_days / ounces_per_gallon) / 
  (total_hours_making_coffee / time_to_make_coffee) = 1.5 := 
sorry

end jack_gallons_per_batch_l293_293636


namespace identify_heaviest_and_lightest_coin_within_13_weighings_l293_293322

-- Lean 4 statement to encapsulate the given problem
theorem identify_heaviest_and_lightest_coin_within_13_weighings :
  ∃ (weighings: list (ℕ × ℕ)) (heaviest lightest: ℕ),
    (length weighings ≤ 13) ∧
    (∀ i j, 1 ≤ i ∧ i ≤ 10 → 1 ≤ j ∧ j ≤ 10 → i ≠ j) ∧
    (∀ (comp: ℕ × ℕ), comp ∈ weighings → comp.1 ≠ comp.2 ∧ 1 ≤ comp.1 ∧ comp.1 ≤ 10 ∧ 1 ≤ comp.2 ∧ comp.2 ≤ 10) ∧
    heaviest ≠ lightest ∧
    (∀ (i: ℕ), 1 ≤ i ∧ i ≤ 10 → 
      (i = heaviest ∨ i = lightest))
: sorry

end identify_heaviest_and_lightest_coin_within_13_weighings_l293_293322


namespace rectangle_perimeter_is_30_l293_293351

noncomputable def triangle_DEF_sides := (9 : ℕ, 12 : ℕ, 15 : ℕ)
noncomputable def rectangle_width := (6 : ℕ)

theorem rectangle_perimeter_is_30 :
  let area_triangle_DEF := (triangle_DEF_sides.1 * triangle_DEF_sides.2) / 2
  let rectangle_length := area_triangle_DEF / rectangle_width
  let rectangle_perimeter := 2 * (rectangle_width + rectangle_length)
  rectangle_perimeter = 30 := by
  sorry

end rectangle_perimeter_is_30_l293_293351


namespace find_original_cost_l293_293719

noncomputable def original_cost_price (final_price : ℝ) (discount : ℝ) (markup : ℝ) 
                                     (assembly_fee : ℝ) (shipping_fee : ℝ) (tax : ℝ) : ℝ :=
  final_price / ((1 - discount) * (1 + markup + assembly_fee + shipping_fee) * (1 + tax))

theorem find_original_cost :
  original_cost_price 6400 0.10 0.15 0.05 0.10 0.08 ≈ 4938.27 :=
by
  sorry

end find_original_cost_l293_293719


namespace sum_of_radii_l293_293835

theorem sum_of_radii (R r : ℝ) (hRr : R > r) (hr0 : r > 0) (h1 : R - r = 5) (h2 : R^2 - r^2 = 100) :
  R + r = 20 :=
begin
  sorry
end

end sum_of_radii_l293_293835


namespace identify_heaviest_and_lightest_coin_in_13_weighings_l293_293308

theorem identify_heaviest_and_lightest_coin_in_13_weighings :
  ∀ (coins : Finₓ 10 → ℝ) 
    (balance_weighing : ∀ (a b : Finₓ 10), Prop), 
    (∀ i j, coins i ≠ coins j) → 
    (∃ strategy : ℕ → (Finₓ 10 × Finₓ 10),
      ∃ h : ℕ,
        h ≤ 13 ∧
        (∃ heaviest lightest : Finₓ 10,
          (∀ i, coins heaviest ≥ coins i) ∧ (∀ j, coins lightest ≤ coins j))) :=
by
  sorry

end identify_heaviest_and_lightest_coin_in_13_weighings_l293_293308


namespace ceil_sum_sqrt_evaluation_l293_293052

theorem ceil_sum_sqrt_evaluation :
  (⌈real.sqrt 3⌉ + ⌈real.sqrt 33⌉ + ⌈real.sqrt 333⌉ = 27) :=
begin
  have h1 : 1 < real.sqrt 3 ∧ real.sqrt 3 < 2 := sorry,
  have h2 : 5 < real.sqrt 33 ∧ real.sqrt 33 < 6 := sorry,
  have h3 : 18 < real.sqrt 333 ∧ real.sqrt 333 < 19 := sorry,
  sorry,
end

end ceil_sum_sqrt_evaluation_l293_293052


namespace find_p_value_l293_293159

variable (p q : ℝ)
variable (p_pos : p > 0) (q_pos : q > 0)

theorem find_p_value
  (hq : ∀ x, x * x + p * x - q = 0)
  (root_diff : ∀ r1 r2, r1 ≠ r2 → abs (r1 - r2) = 2) :
  p = sqrt (4 - 4 * q) :=
by
  sorry

end find_p_value_l293_293159


namespace perpendicular_if_and_only_if_condition_l293_293191

theorem perpendicular_if_and_only_if_condition
  (ABC : Type)
  [triangle ABC]
  [acute ABC]
  (A B C D E F : ABC)
  (h_angle_B_greater_C : ∠ B > ∠ C)
  (h_D_foot : is_foot_of_altitude A D BC)
  (h_E_foot : is_foot_of_perpendicular D E AC)
  (h_F_on_DE : F ∈ line_segment D E) :
  (is_perpendicular (line_through A F) (line_through B F)) ↔ (dist E F * dist D C = dist B D * dist D E) :=
sorry

end perpendicular_if_and_only_if_condition_l293_293191


namespace sequence_integer_values_l293_293783

def sequence (m : ℕ) : ℕ → ℕ
| 0     := 1
| (n+1) := if (sequence m n) < 2^m then (sequence m n)^2 + 2^m 
                            else (sequence m n) / 2

theorem sequence_integer_values (m : ℕ) (h_m : m > 0) : 
  (a1 : ℕ) → (h_a1_pos: a1 > 0) → 
  (∀ n ≥ 1, (sequence m (n-1)) ∈ ℕ) ↔ ∃ l : ℕ, a1 = 2^l :=
sorry

end sequence_integer_values_l293_293783


namespace sum_of_solutions_quadratic_eq_l293_293376

theorem sum_of_solutions_quadratic_eq : 
  (∑ x in {x | x^2 = 10 * x - 24}, x) = 10 :=
by
  sorry

end sum_of_solutions_quadratic_eq_l293_293376


namespace Sn_divisible_by_3_Sn_divisible_by_9_iff_even_l293_293661

-- Define \( S_n \) following the given conditions
def S (n : ℕ) : ℕ :=
  let a := 2^n + 1 -- first term
  let b := 2^(n+1) - 1 -- last term
  let m := b - a + 1 -- number of terms
  (m * (a + b)) / 2 -- sum of the arithmetic series

-- The first part: Prove that \( S_n \) is divisible by 3 for all positive integers \( n \)
theorem Sn_divisible_by_3 (n : ℕ) (hn : 0 < n) : 3 ∣ S n := sorry

-- The second part: Prove that \( S_n \) is divisible by 9 if and only if \( n \) is even
theorem Sn_divisible_by_9_iff_even (n : ℕ) (hn : 0 < n) : 9 ∣ S n ↔ Even n := sorry

end Sn_divisible_by_3_Sn_divisible_by_9_iff_even_l293_293661


namespace odd_functions_suff_not_necessary_l293_293154

noncomputable def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = - (f x)

theorem odd_functions_suff_not_necessary (f g : ℝ → ℝ) (hf : is_odd f) (hg : is_odd g) :
  is_odd (λ x, f x + g x) ∧ ¬ (∀ (f g : ℝ → ℝ), is_odd (λ x, f x + g x) → is_odd f ∧ is_odd g) :=
by
  sorry

end odd_functions_suff_not_necessary_l293_293154


namespace minimum_PA_plus_PQ_is_9_l293_293908

def parabola (x y : ℝ) : Prop := x^2 = 4 * y

def projection_on_x (P : ℝ × ℝ) : ℝ × ℝ := (P.1, 0)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

noncomputable def minimum_distance_from_parabola_to_point_A : ℝ :=
  let P : ℝ × ℝ := sorry in -- A point on the parabola
  let Q : ℝ × ℝ := projection_on_x P in
  let A : ℝ × ℝ := (8, 7) in
  distance P A + distance P Q

theorem minimum_PA_plus_PQ_is_9 : minimum_distance_from_parabola_to_point_A = 9 := sorry

end minimum_PA_plus_PQ_is_9_l293_293908


namespace complement_A_union_B_range_of_m_l293_293123

def setA : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.sqrt (x^2 - 5*x - 14) }
def setB : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.log (-x^2 - 7*x - 12) }
def setC (m : ℝ) : Set ℝ := { x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1 }

theorem complement_A_union_B :
  (A ∪ B)ᶜ = Set.Ioo (-2 : ℝ) 7 :=
sorry

theorem range_of_m (m : ℝ) :
  (A ∪ setC m = A) → (m < 2 ∨ m ≥ 6) :=
sorry

end complement_A_union_B_range_of_m_l293_293123


namespace farm_area_l293_293811

theorem farm_area (W : ℕ) (cost_fencing_per_meter : ℕ) (total_cost : ℕ) (total_length_fence : ℕ)
                  (hW : W = 30) (h_cost : cost_fencing_per_meter = 12) (h_total_cost : total_cost = 1440)
                  (h_fence_length : total_length_fence = total_cost / cost_fencing_per_meter) :
  W * (total_length_fence - W - (Int.sqrt(total_length_fence * total_length_fence - 2 * total_length_fence * W + W * W).toNat)) = 1200 :=
by
  sorry

end farm_area_l293_293811


namespace identify_heaviest_and_lightest_coin_within_13_weighings_l293_293320

-- Lean 4 statement to encapsulate the given problem
theorem identify_heaviest_and_lightest_coin_within_13_weighings :
  ∃ (weighings: list (ℕ × ℕ)) (heaviest lightest: ℕ),
    (length weighings ≤ 13) ∧
    (∀ i j, 1 ≤ i ∧ i ≤ 10 → 1 ≤ j ∧ j ≤ 10 → i ≠ j) ∧
    (∀ (comp: ℕ × ℕ), comp ∈ weighings → comp.1 ≠ comp.2 ∧ 1 ≤ comp.1 ∧ comp.1 ≤ 10 ∧ 1 ≤ comp.2 ∧ comp.2 ≤ 10) ∧
    heaviest ≠ lightest ∧
    (∀ (i: ℕ), 1 ≤ i ∧ i ≤ 10 → 
      (i = heaviest ∨ i = lightest))
: sorry

end identify_heaviest_and_lightest_coin_within_13_weighings_l293_293320


namespace parking_lot_wheels_l293_293198

-- Define the conditions
def num_cars : Nat := 10
def num_bikes : Nat := 2
def wheels_per_car : Nat := 4
def wheels_per_bike : Nat := 2

-- Define the total number of wheels
def total_wheels : Nat := (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike)

-- State the theorem
theorem parking_lot_wheels : total_wheels = 44 :=
by
  sorry

end parking_lot_wheels_l293_293198


namespace planted_fraction_l293_293497

theorem planted_fraction (side_x : ℝ) (a b c A_BC : ℝ) :
  let A := 5
  let B := 12
  let hypotenuse := Real.sqrt(A^2 + B^2)
  let total_area := (1/2) * A * B
  let unplanted_square_side := side_x
  let unplanted_square_distance_to_hypotenuse := 4
  unplanted_square_distance_to_hypotenuse = 4 → 
  (total_area - unplanted_square_side ^ 2) / total_area = 734 / 750 := 
by
  sorry

end planted_fraction_l293_293497


namespace derivative_at_zero_l293_293148

def f (x : ℝ) : ℝ := Real.exp (x + 1) - 3 * x

theorem derivative_at_zero : deriv f 0 = Real.exp 1 - 3 := by
  sorry

end derivative_at_zero_l293_293148


namespace simplify_sqrt_expression_l293_293693

theorem simplify_sqrt_expression :
  sqrt (6 + 4 * sqrt 3) + sqrt (6 - 4 * sqrt 3) = 2 * sqrt 6 :=
by 
  sorry

end simplify_sqrt_expression_l293_293693


namespace coefficient_of_x_pow_4_l293_293709

theorem coefficient_of_x_pow_4 : 
  let exp := (x^2 + (1/x))^5 in
  (∃ r : ℕ, r = 2 ∧ ∑ i in range 6, binomial 5 i * (x^2)^(5-i) * (1/x)^i = 10 * x^4) :=
begin
  sorry
end

end coefficient_of_x_pow_4_l293_293709


namespace isosceles_triangle_with_three_colors_l293_293552

open Function

def is_coprime (a b : ℕ) : Prop := gcd a b = 1

theorem isosceles_triangle_with_three_colors
  (n : ℕ)
  (h_coprime : is_coprime n 6)
  (coloring : Fin n → Fin 3)
  (h_odd_appearance : ∀ c : Fin 3, Odd (Nat.card (coloring ⁻¹' {c}))) :
  ∃ (triangle : Fin 3 → Fin n), turn_test_triangle coloring triangle :=
by
  sorry

end isosceles_triangle_with_three_colors_l293_293552


namespace find_ffour_times_l293_293874

noncomputable def f (z : ℂ) : ℂ :=
if h : z.im = 0 then -z^2 else z^2

theorem find_ffour_times (z : ℂ) (h : z = 2 + I) : f (f (f (f z))) = 164833 + 354816 * I :=
by {
  rw h,
  sorry
}

end find_ffour_times_l293_293874


namespace sequence_1234_to_500_not_divisible_by_9_l293_293980

-- Definition for the sum of the digits of concatenated sequence
def sum_of_digits (n : ℕ) : ℕ :=
  -- This is a placeholder for the actual function calculating the sum of digits
  -- of all numbers from 1 to n concatenated together.
  sorry 

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem sequence_1234_to_500_not_divisible_by_9 : ¬ is_divisible_by_9 (sum_of_digits 500) :=
by
  -- Placeholder indicating the solution facts and methods should go here.
  sorry

end sequence_1234_to_500_not_divisible_by_9_l293_293980


namespace region_area_l293_293022

def abs (x : ℝ) : ℝ := if x < 0 then -x else x

def bounded_region (x y : ℝ) : Prop :=
  abs (x - 4) ≤ y ∧ y ≤ 5 - abs (x + 1)

def area_of_bounded_region : ℝ := 1

theorem region_area :
  ∃ region_area : ℝ, (∀ (x y : ℝ), bounded_region x y → region_area = 1) :=
sorry

end region_area_l293_293022


namespace cubic_expression_l293_293948

theorem cubic_expression (x : ℝ) (hx : x + 1/x = -7) : x^3 + 1/x^3 = -322 :=
by sorry

end cubic_expression_l293_293948


namespace correct_option_l293_293774

-- Define the constants
def e := 2.718
def π := 3.14
def ln2 := 0.69

-- Define the inequalities
def optionA := 3 * e ^ ((π - 3) / 2) < π
def optionB := (4 * Real.sqrt 2) / 3 < e * ln2
def optionC := (e ^ Real.cos 1) / (Real.cos 2 + 1) < 2 * Real.sqrt e
def optionD := Real.sin 1 < 2 / π

theorem correct_option :
  ¬optionA ∧ ¬optionB ∧ optionC ∧ ¬optionD := by
  sorry

end correct_option_l293_293774


namespace simplify_fraction_a_b_l293_293518

noncomputable def a_n (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), 1 / (Nat.choose n k)

noncomputable def b_n (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), (k^2 : ℝ) / (Nat.choose n k)

theorem simplify_fraction_a_b (n : ℕ) (hn : 0 < n) : a_n n / b_n n = 2 / (n^2) :=
by
  -- proof would go here
  sorry

end simplify_fraction_a_b_l293_293518


namespace prove_modulus_one_l293_293560

open Complex

-- Defining the hypothesis
structure ComplexCondition (z : ℂ) : Prop :=
(eq : 11 * z ^ 10 + 10 * I * z ^ 9 + 10 * I * z - 11 = 0)

-- The main theorem statement
theorem prove_modulus_one (z : ℂ) (h : ComplexCondition z) : |z| = 1 :=
sorry

end prove_modulus_one_l293_293560


namespace measure_of_angle_YZX_l293_293014

noncomputable def problem_statement : Prop :=
  ∃ (A B C X Y Z : ℝ),
    X ∈ (segment B C) ∧
    Y ∈ (segment A B) ∧
    Z ∈ (segment A C) ∧
    angle A = 50 ∧
    angle B = 70 ∧
    angle C = 60 ∧
    let Γ := incircle ABC in
    let γ := circumcircle XYZ in
    incircle ABC = Γ ∧ 
    circumcircle XYZ = γ ∧ 
    ∠YZX = 115

theorem measure_of_angle_YZX (A B C X Y Z : ℝ) :
  X ∈ (segment B C) →
  Y ∈ (segment A B) →
  Z ∈ (segment A C) →
  angle A = 50 →
  angle B = 70 →
  angle C = 60 →
  let Γ := incircle ABC in
  let γ := circumcircle XYZ in
  incircle ABC = Γ →
  circumcircle XYZ = γ →
  ∠YZX = 115 :=
by
  sorry

end measure_of_angle_YZX_l293_293014


namespace distance_dormitory_to_city_l293_293402

variable (D : ℝ)
variable (c : ℝ := 12)
variable (f := (1/5) * D)
variable (b := (2/3) * D)

theorem distance_dormitory_to_city (h : f + b + c = D) : D = 90 := by
  sorry

end distance_dormitory_to_city_l293_293402


namespace total_profit_l293_293796

-- Definitions based on conditions
def Investment (B : ℕ) : ℕ := B
def Period (B : ℕ) : ℕ := B * 2
def Profit_share (investment period : ℕ) : ℕ := investment * period
def B_profit : ℕ := 5000

-- Main theorem statement
theorem total_profit (B_invest : ℕ) (B_period : ℕ) (A_invest := 3 * B_invest) (A_period := 2 * B_period) :
  let B_share := Profit_share B_invest B_period;
  let A_share := Profit_share A_invest A_period;
  let total_profit := B_share + A_share in
  total_profit = 35000 :=
by
  admit

end total_profit_l293_293796


namespace Laura_bought_one_kg_of_potatoes_l293_293995

theorem Laura_bought_one_kg_of_potatoes :
  let price_salad : ℝ := 3
  let price_beef_per_kg : ℝ := 2 * price_salad
  let price_potato_per_kg : ℝ := price_salad * (1 / 3)
  let price_juice_per_liter : ℝ := 1.5
  let total_cost : ℝ := 22
  let num_salads : ℝ := 2
  let num_beef_kg : ℝ := 2
  let num_juice_liters : ℝ := 2
  let cost_salads := num_salads * price_salad
  let cost_beef := num_beef_kg * price_beef_per_kg
  let cost_juice := num_juice_liters * price_juice_per_liter
  (total_cost - (cost_salads + cost_beef + cost_juice)) / price_potato_per_kg = 1 :=
sorry

end Laura_bought_one_kg_of_potatoes_l293_293995


namespace binary_product_correct_l293_293871

-- Definitions based on the conditions
def bin1 : ℕ := 0b110011
def bin2 : ℕ := 0b1101
def product : ℕ := 0b10011000101

-- The Lean 4 statement for the proof problem
theorem binary_product_correct : bin1 * bin2 = product := by
  sorry

end binary_product_correct_l293_293871


namespace find_A_l293_293545

theorem find_A (A B C D : ℕ) 
  (distinctABCD : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (distSums : ∀ x ∈ {A + C, B + C, B + D, D + A}, x ∈ {1, 2, 3, 4, 5, 6, 7, 8}) 
  (distinctSums : (A + C ≠ B + C) ∧ (A + C ≠ B + D) ∧ (A + C ≠ D + A) ∧
                  (B + C ≠ B + D) ∧ (B + C ≠ D + A) ∧ (B + D ≠ D + A))
  (A_largest : A > B ∧ A > C ∧ A > D) :
  A = 12 :=
by
  sorry

end find_A_l293_293545


namespace target_problem_l293_293820

-- Definitions of the given functions
def f1 (x : ℝ) : ℝ := (1 / 2) ^ x
def f2 (x : ℝ) : ℝ := x ^ -2
def f3 (x : ℝ) : ℝ := x ^ 2 + 1
def f4 (x : ℝ) : ℝ := log 3 (-x)

-- Conditions to check
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)
def is_increasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop := ∀ ⦃a b⦄, a ∈ s → b ∈ s → a < b → f a < f b

-- The target problem statement
theorem target_problem :
  is_even f2 ∧ is_increasing_on f2 (set.Iio 0) :=
sorry

end target_problem_l293_293820


namespace supporting_pillars_concrete_is_1890_l293_293220

variable (roadwayDeck : ℕ) (oneAnchorConcrete : ℕ) (totalConcrete : ℕ) (increasePercentage : ℕ)

def totalAnchorsConcrete : ℕ := oneAnchorConcrete * 2

def supportingPillarsConcrete : ℕ := totalConcrete - roadwayDeck - totalAnchorsConcrete

def increasedRoadwayDeck : ℕ := roadwayDeck + (roadwayDeck * increasePercentage) / 100
def increasedSupportingPillarsConcrete : ℕ := supportingPillarsConcrete + (supportingPillarsConcrete * increasePercentage) / 100

theorem supporting_pillars_concrete_is_1890
    (h1 : roadwayDeck = 1600)
    (h2 : oneAnchorConcrete = 700)
    (h3 : totalConcrete = 4800)
    (h4 : increasePercentage = 5) :
    increasedSupportingPillarsConcrete roadwayDeck oneAnchorConcrete totalConcrete increasePercentage = 1890 := by
  sorry

end supporting_pillars_concrete_is_1890_l293_293220


namespace price_decreases_l293_293718

variables (a : ℝ) (p : ℝ) (m : ℕ) -- variables definitions

def price_function (x : ℕ) : ℝ :=
  a * (1 - p / 100)^x

theorem price_decreases : 
  ∀ x, 0 ≤ x ∧ x ≤ m → price_function a p x = a * (1 - p / 100)^x :=
by
  sorry

end price_decreases_l293_293718


namespace domain_and_decrease_a1_increasing_f_iff_l293_293565

section
variables {a x : ℝ}

-- Definition of function f
def f (x : ℝ) (a : ℝ) : ℝ := log 2 (x^2 - 4 * a * x + 3)

-- Part (Ⅰ): For a = 1, the domain of f(x)
def domain_f_when_a_is_one : Set ℝ :=
{x | x ∈ (Set.Ioo (-∞) 1) ∪ (Set.Ioo 3 ∞)}

-- Part (Ⅰ): For a = 1, the interval where f(x) is decreasing
def interval_f_decreasing_when_a_is_one : Set ℝ :=
{x | x ∈ (Set.Ioo (-∞) 1)}

-- The main theorem statement for Part (Ⅰ)
theorem domain_and_decrease_a1 : 
  ∀ x : ℝ, f x 1 ≠ 0 → (x ∈ domain_f_when_a_is_one) ∧ (x ∈ interval_f_decreasing_when_a_is_one) :=
sorry

-- Part (Ⅱ): f(x) is increasing on (1, +∞) if and only if a ∈ (-∞, 1/2]
def range_of_a_for_increasing_f : Set ℝ :=
 {a | a ≤ 1 / 2}

-- The main theorem statement for Part (Ⅱ)
theorem increasing_f_iff (a : ℝ) : 
  (∀ x ∈ Set.Ioo 1 (∞ : ℝ), ∂ (λ x, f x a) / ∂ x > 0) ↔ a ∈ range_of_a_for_increasing_f :=
sorry

end

end domain_and_decrease_a1_increasing_f_iff_l293_293565


namespace volume_of_extended_parallelepiped_l293_293018

theorem volume_of_extended_parallelepiped :
  let main_box_volume := 3 * 3 * 6
  let external_boxes_volume := 2 * (3 * 3 * 1 + 3 * 6 * 1 + 3 * 6 * 1)
  let spheres_volume := 8 * (1 / 8) * (4 / 3) * Real.pi * (1 ^ 3)
  let cylinders_volume := 12 * (1 / 4) * Real.pi * 1^2 * 3 + 12 * (1 / 4) * Real.pi * 1^2 * 6
  main_box_volume + external_boxes_volume + spheres_volume + cylinders_volume = (432 + 52 * Real.pi) / 3 :=
by
  sorry

end volume_of_extended_parallelepiped_l293_293018


namespace split_tip_evenly_l293_293227

noncomputable def total_cost (julie_order : ℝ) (letitia_order : ℝ) (anton_order : ℝ) : ℝ :=
  julie_order + letitia_order + anton_order

noncomputable def total_tip (meal_cost : ℝ) (tip_rate : ℝ) : ℝ :=
  tip_rate * meal_cost

noncomputable def tip_per_person (total_tip : ℝ) (num_people : ℝ) : ℝ :=
  total_tip / num_people

theorem split_tip_evenly :
  let julie_order := 10 in
  let letitia_order := 20 in
  let anton_order := 30 in
  let tip_rate := 0.20 in
  let num_people := 3 in
  tip_per_person (total_tip (total_cost julie_order letitia_order anton_order) tip_rate) num_people = 4 :=
by
  sorry

end split_tip_evenly_l293_293227


namespace find_five_digit_number_l293_293383

theorem find_five_digit_number :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∃ rev_n : ℕ, rev_n = (n % 10) * 10000 + (n / 10 % 10) * 1000 + (n / 100 % 10) * 100 + (n / 1000 % 10) * 10 + (n / 10000) ∧ 9 * n = rev_n) ∧ n = 10989 :=
  sorry

end find_five_digit_number_l293_293383


namespace number_of_valid_programs_l293_293443

def course := String

def English : course := "English"
def Algebra : course := "Algebra"
def Geometry : course := "Geometry"
def Physics : course := "Physics"
def History : course := "History"
def Art : course := "Art"
def Latin : course := "Latin"

-- Define the set of possible courses
def course_list := {English, Algebra, Geometry, Physics, History, Art, Latin}

-- Define conditions
def includes_math (prog : set course) : Prop :=
  ({Algebra, Geometry} ⊆ prog) ∨ ({Algebra, Geometry} ∩ prog ⊇ {Algebra}) ∨ ({Algebra, Geometry} ∩ prog ⊇ {Geometry})

def includes_english (prog : set course) : Prop :=
  English ∈ prog

def includes_science (prog : set course) : Prop :=
  Physics ∈ prog

def valid_program (prog : set course) : Prop :=
  includes_english prog ∧ includes_math prog ∧ includes_science prog

-- Define the proof statement
theorem number_of_valid_programs : (finset.filter valid_program (finset.powerset course_list)).card = 4 :=
by
  sorry

end number_of_valid_programs_l293_293443


namespace ellipse_standard_equation_l293_293714

theorem ellipse_standard_equation
  (a b c : ℝ)
  (h1 : (3 * a) / (-a) + 16 / b = 1)
  (h2 : (3 * a) / c + 16 / (-b) = 1)
  (h3 : a > 0)
  (h4 : b > 0)
  (h5 : a > b)
  (h6 : a^2 = b^2 + c^2) : 
  (a = 5 ∧ b = 4 ∧ c = 3) ∧ (∀ x y, x^2 / 25 + y^2 / 16 = 1 ↔ (a = 5 ∧ b = 4)) := 
sorry

end ellipse_standard_equation_l293_293714


namespace factorization_correct_l293_293066

-- Define the expression
def expression (x : ℝ) : ℝ := x^2 + 2 * x

-- State the theorem to prove the factorized form is equal to the expression
theorem factorization_correct (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by {
  -- Lean will skip the proof because of sorry, ensuring the statement compiles correctly.
  sorry
}

end factorization_correct_l293_293066


namespace parabola_evolute_ellipse_evolute_l293_293502

-- Definition of the parabola and the solution for its evolute
def parabola (x y : ℝ) : Prop := x^2 = 2 * (1 - y)
def evolute_parabola (X Y : ℝ) : Prop := 27 * X^2 = -8 * Y^3

-- Proof statement for the parabola
theorem parabola_evolute (x y X Y : ℝ) (h_parabola : parabola x y) :
  evolute_parabola X Y :=
sorry

-- Definition of the ellipse and the solution for its evolute
def ellipse (t a b x y : ℝ) : Prop :=
  x = a * cos t ∧ y = b * sin t

def evolute_ellipse (t a b X Y : ℝ) (c_sq : ℝ := a^2 - b^2) : Prop :=
  X = -c_sq / a * (cos t)^3 ∧ Y = -c_sq / b * (sin t)^3

-- Proof statement for the ellipse
theorem ellipse_evolute (t a b x y X Y : ℝ)
  (h_ellipse : ellipse t a b x y) :
  evolute_ellipse t a b X Y :=
sorry

end parabola_evolute_ellipse_evolute_l293_293502


namespace complex_roots_real_implies_12_real_z5_l293_293746

theorem complex_roots_real_implies_12_real_z5 :
  (∃ (z : ℂ) (h : z ^ 30 = 1), is_real (z ^ 5)) → (finset.card ((finset.filter (λ z, is_real (z ^ 5)) (finset.univ.filter (λ z, z ^ 30 = 1)))) = 12) := by
  sorry

end complex_roots_real_implies_12_real_z5_l293_293746


namespace rhombus_dot_product_l293_293534

variables (a : ℝ) -- the side length of rhombus ABCD
variables (A B C D : Type) [AddCommGroup B] [Module ℝ B]
variables (AB BC BD : B) (θ : ℝ)

-- Rhombus properties with side length a
def is_rhombus (a : ℝ) (AB BC BD CD : B) : Prop :=
  ∥AB∥ = a ∧ ∥BC∥ = a ∧ ∥BD∥ = a ∧ ∥CD∥ = a ∧ ∃ (θ : ℝ), θ = π / 3

-- Computing dot product for vectors in a rhombus given angle 60°
theorem rhombus_dot_product (h : is_rhombus a AB BC BD CD) :
  BD ⬝ CD = 3 / 2 * a^2 :=
sorry -- proof skipped

end rhombus_dot_product_l293_293534


namespace infinite_relatively_prime_pairs_l293_293686

def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

noncomputable def positive_int_fraction (x y : ℕ) : Prop := ∃ k : ℕ, k > 0 ∧ x = k * y

theorem infinite_relatively_prime_pairs :
  ∃ (a b : ℕ), relatively_prime a b ∧ positive_int_fraction (a^2 - 5) b ∧ positive_int_fraction (b^2 - 5) a ∧ (∀ n, ∃ (a b : ℕ), relatively_prime a b ∧ positive_int_fraction (a^2 - 5) b ∧ positive_int_fraction (b^2 - 5) a) := by
  sorry

end infinite_relatively_prime_pairs_l293_293686


namespace factorize_x_squared_plus_2x_l293_293060

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) :=
by sorry

end factorize_x_squared_plus_2x_l293_293060


namespace intersection_of_sets_l293_293185

variable (M : Set ℝ) (N : Set ℝ)

def setM := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def setN := {x : ℝ | x^2 - 2*x ≤ 0}

-- Prove that the intersection of setM and setN is [0, 1]
theorem intersection_of_sets (M = setM) (N = setN) : M ∩ N = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end intersection_of_sets_l293_293185


namespace find_m_n_l293_293977

variable {XYZ : Type}
variables (X Y Z D M P : XYZ)

def length (a b : XYZ) : ℝ := sorry
def angle_bisector (a b c : XYZ) : Prop := sorry
def midpoint (a b : XYZ) (m : XYZ) : Prop := sorry
def intersects (a b c : XYZ) : XYZ := sorry

-- Conditions
axiom XY_length : length X Y = 15
axiom XZ_length : length X Z = 9
axiom angle_bisector_X : angle_bisector X Y Z D
axiom midpoint_XD : midpoint X D M
axiom intersection_BM_XZ : intersects X Z M = P

-- To prove
theorem find_m_n : let m := 8 in let n := 5 in m + n = 13 :=
by
  sorry

end find_m_n_l293_293977


namespace distance_between_points_l293_293943

noncomputable def distance_AB (r : ℝ) (O1O2 : ℝ) (α : ℝ) : ℝ :=
  if (r > 1/2) then Real.sin α else 0

theorem distance_between_points
  (r : ℝ)
  (O1O2 : ℝ)
  (α : ℝ)
  (h1 : r > 1/2)
  (h2 : O1O2 = 1) :
  distance_AB r O1O2 α = Real.sin α :=
by
  -- Proof placeholder
  sorry

end distance_between_points_l293_293943


namespace sequence_equality_l293_293231

theorem sequence_equality (n : ℕ) (h_pos : 0 < n) :
  let A := { seq : list ℕ // (∀ i, 1 ≤ i → i < seq.length → seq[i] ≥ seq[i + 1]) ∧ 
                                      seq.sum = n ∧ 
                                      ∀ i, 1 ≤ i → i < seq.length → ∃ m, seq[i] + 1 = 2 ^ m },
      B := { seq : list ℕ // (∀ j, 0 ≤ j → j < seq.length - 1 → seq[j] ≥ 2 * seq[j + 1]) ∧ 
                                      seq.sum = n } in
  A.card = B.card :=
sorry

end sequence_equality_l293_293231


namespace eval_sum_sqrt_ceil_l293_293042

theorem eval_sum_sqrt_ceil:
  ∀ (x : ℝ), 
  (1 < sqrt 3 ∧ sqrt 3 < 2) ∧
  (5 < sqrt 33 ∧ sqrt 33 < 6) ∧
  (18 < sqrt 333 ∧ sqrt 333 < 19) →
  (⌈ sqrt 3 ⌉ + ⌈ sqrt 33 ⌉ + ⌈ sqrt 333 ⌉ = 27) :=
by
  intro x
  sorry

end eval_sum_sqrt_ceil_l293_293042


namespace journey_speed_is_24_l293_293805

def speed_of_second_half_journey : Prop :=
  ∀ (d1 d2 v1 t total_time v2 : ℝ),
    d1 = 112 →
    d2 = 112 →
    v1 = 21 →
    total_time = 10 →
    d1 + d2 = 224 →
    t = d1 / v1 →
    total_time = t + d2 / v2 →
    v2 = 24

theorem journey_speed_is_24 : speed_of_second_half_journey :=
by
  intros d1 d2 v1 t total_time v2 h_d1 h_d2 h_v1 h_total_time h_distance h_t h_total_time_eq
  rw [h_d1, h_d2, h_v1, h_total_time, h_distance] at *,
  rw h_t at h_total_time_eq,
  simp at h_total_time_eq,
  sorry

end journey_speed_is_24_l293_293805


namespace correct_transformation_l293_293452

theorem correct_transformation :
  (∀ a b c : ℝ, c ≠ 0 → (a / c = b / c ↔ a = b)) ∧
  (∀ x : ℝ, ¬ (x / 4 + x / 3 = 1 ∧ 3 * x + 4 * x = 1)) ∧
  (∀ a b c : ℝ, ¬ (a * b = b * c ∧ a ≠ c)) ∧
  (∀ x a : ℝ, ¬ (4 * x = a ∧ x = 4 * a)) := sorry

end correct_transformation_l293_293452


namespace major_axis_length_l293_293289

theorem major_axis_length (x y : ℝ) (h : 16 * x^2 + 9 * y^2 = 144) : 8 = 8 :=
by
  sorry

end major_axis_length_l293_293289


namespace line_intersects_574_squares_and_circles_l293_293834

def is_lattice_point (p : ℕ × ℕ) : Prop := ∃ k : ℕ, p = (7 * k, 3 * k)

def intersects_square (square_center_p : ℕ × ℕ) (line_func : ℝ → ℝ × ℝ) : Prop :=
  let side_len := 1 / 5 in
  let p := (square_center_p.1 : ℝ, square_center_p.2 : ℝ) in
  let square := {
    bottom_left := (p.1 - side_len / 2, p.2 - side_len / 2),
    top_right := (p.1 + side_len / 2, p.2 + side_len / 2)
  } in
  ∃ t : ℝ, (line_func t).1 ≥ square.bottom_left.1 ∧
            (line_func t).1 ≤ square.top_right.1 ∧
            (line_func t).2 ≥ square.bottom_left.2 ∧
            (line_func t).2 ≤ square.top_right.2

def intersects_circle (circle_center_p : ℕ × ℕ) (line_func : ℝ → ℝ × ℝ) : Prop :=
  let radius := 1 / 10 in
  let p := (circle_center_p.1 : ℝ, circle_center_p.2 : ℝ) in
  ∃ t : ℝ, (line_func t).1 ^ 2 + (line_func t).2 ^ 2 ≤ radius ^ 2

noncomputable def num_intersections (line_func : ℝ → ℝ × ℝ) : ℕ :=
  let lattice_points := Nat.upto 144 in
  let centers := lattice_points.map (λ k, (7 * k, 3 * k)) in
  let squares := centers.filter (λ p, intersects_square p line_func) in
  let circles := centers.filter (λ p, intersects_circle p line_func) in
  squares.length + circles.length

theorem line_intersects_574_squares_and_circles :
  let line_func := λ t : ℝ, (1001 * t, 429 * t),
  num_intersections line_func = 574 :=
by sorry

end line_intersects_574_squares_and_circles_l293_293834


namespace vectors_parallel_perpendicular_l293_293572

theorem vectors_parallel_perpendicular (t t1 t2 : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) 
    (h_a : a = (2, t)) (h_b : b = (1, 2)) :
    ((2 * 2 = t * 1) → t1 = 4) ∧ ((2 * 1 + 2 * t = 0) → t2 = -1) :=
by 
  sorry

end vectors_parallel_perpendicular_l293_293572


namespace sphere_volume_ratio_l293_293603

theorem sphere_volume_ratio (r1 r2 : ℝ) (S1 S2 V1 V2 : ℝ) 
(h1 : S1 = 4 * Real.pi * r1^2)
(h2 : S2 = 4 * Real.pi * r2^2)
(h3 : V1 = (4 / 3) * Real.pi * r1^3)
(h4 : V2 = (4 / 3) * Real.pi * r2^3)
(h_surface_ratio : S1 / S2 = 2 / 3) :
V1 / V2 = (2 * Real.sqrt 6) / 9 :=
by
  sorry

end sphere_volume_ratio_l293_293603


namespace tina_spent_on_books_l293_293757

theorem tina_spent_on_books : 
  ∀ (saved_in_june saved_in_july saved_in_august spend_on_books spend_on_shoes money_left : ℤ),
  saved_in_june = 27 →
  saved_in_july = 14 →
  saved_in_august = 21 →
  spend_on_shoes = 17 →
  money_left = 40 →
  (saved_in_june + saved_in_july + saved_in_august) - spend_on_books - spend_on_shoes = money_left →
  spend_on_books = 5 :=
by
  intros saved_in_june saved_in_july saved_in_august spend_on_books spend_on_shoes money_left
  intros h_june h_july h_august h_shoes h_money_left h_eq
  sorry

end tina_spent_on_books_l293_293757


namespace union_eq_C_l293_293790

def A: Set ℝ := { x | x > 2 }
def B: Set ℝ := { x | x < 0 }
def C: Set ℝ := { x | x * (x - 2) > 0 }

theorem union_eq_C : (A ∪ B) = C :=
by
  sorry

end union_eq_C_l293_293790


namespace sequence_polynomial_degree_l293_293981

theorem sequence_polynomial_degree
  (k : ℕ)
  (v : ℕ → ℤ)
  (u : ℕ → ℤ)
  (h_diff_poly : ∃ p : Polynomial ℤ, ∀ n, v n = Polynomial.eval (n : ℤ) p)
  (h_diff_seq : ∀ n, v n = (u (n + 1) - u n)) :
  ∃ q : Polynomial ℤ, ∀ n, u n = Polynomial.eval (n : ℤ) q := 
sorry

end sequence_polynomial_degree_l293_293981


namespace pq_over_ef_l293_293268

noncomputable def point := (ℝ × ℝ)
noncomputable def rectangle_abcd := (8 : ℝ) × (6 : ℝ) × (8 : ℝ) × (6 : ℝ)

noncomputable def segment_overline_ab := (6 : ℝ)
noncomputable def point_e := (6 : ℝ) × (6 : ℝ)
noncomputable def segment_overline_bc := (4 : ℝ)
noncomputable def point_g := (8 : ℝ) × (2 : ℝ)
noncomputable def segment_overline_cd := (3 : ℝ)
noncomputable def point_f := (3 : ℝ) × (0 : ℝ)

noncomputable def point_p := ((48 / 11) : ℝ) × ((30 / 11) : ℝ)
noncomputable def point_q := ((24 / 5) : ℝ) × ((18 / 5) : ℝ)

noncomputable def length_ef : ℝ :=
  real.sqrt ((6 - 3)^2  + (6 - 0)^2)

noncomputable def length_pq : ℝ := 
  real.sqrt (((48 / 11) - (24 / 5))^2 + ((30 / 11) - (18 / 5))^2)

theorem pq_over_ef : 
  (length_pq / length_ef) = (8 / 55) :=
sorry

end pq_over_ef_l293_293268


namespace right_triangle_geometric_proof_l293_293202

theorem right_triangle_geometric_proof
  (A B C P Q X Y : Point)
  (h₁ : ∠BAC = 90°)
  (h₂ : perpendicular_from C A B P)
  (h₃ : perpendicular_from B A C Q)
  (h₄ : line_intersects_circumcircle_two_points PQ (circumcircle_triangle A B C) X Y)
  (h₅ : dist X P = 12)
  (h₆ : dist P Q = 20)
  (h₇ : dist Q Y = 8)
  (AB AC : ℝ)
  (h₈ : AB = 5)
  (h₉ : AC = 3)
  : AB * AC = 15 :=
by 
  sorry

end right_triangle_geometric_proof_l293_293202


namespace sum_of_coordinates_l293_293807

def Point := (ℝ × ℝ)

def is_midpoint (M A B : Point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def parallelogram_vertices (A B C D : Point) : Prop :=
  let M_ad := ((A.1 + D.1) / 2, (A.2 + D.2) / 2)
  let M_bc := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  in A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ D ≠ A ∧ D ≠ B ∧ D ≠ C ∧ is_midpoint M_ad A D ∧ is_midpoint M_bc B C ∧ M_ad = M_bc

theorem sum_of_coordinates (A B D : Point) (C : Point)
  (h : parallelogram_vertices A B C D)
  (h_A : A = (-4, 1)) 
  (h_B : B = (2, -3)) 
  (h_D : D = (8, 5)) :
  C.1 + C.2 = 11 := 
sorry

end sum_of_coordinates_l293_293807


namespace range_of_m_l293_293528

theorem range_of_m (m : ℝ) (h1 : 0 < m)
  (h2 : ∃ P : ℝ × ℝ, ((P.1 - 3)^2 + (P.2 - 4)^2 = 1) ∧
                      ((-m - P.1)^2 + (P.2)^2 = ℝ) ∧ 
                      ((m - P.1)^2 + (P.2)^2 = ℝ) ∧ 
                      real.angle (-m, 0) (P.1, P.2) (m, 0) = real.pi / 2)
  : 4 ≤ m ∧ m ≤ 6 :=
sorry

end range_of_m_l293_293528


namespace complex_roots_real_l293_293741

theorem complex_roots_real (z : ℂ) (h : z^30 = 1) : 
  {z : ℂ | z^30 = 1}.count (λ z, z^5 ∈ ℝ) = 10 :=
sorry

end complex_roots_real_l293_293741


namespace complex_roots_real_l293_293744

theorem complex_roots_real (z : ℂ) (h : z^30 = 1) : 
  {z : ℂ | z^30 = 1}.count (λ z, z^5 ∈ ℝ) = 10 :=
sorry

end complex_roots_real_l293_293744


namespace minimum_value_of_expression_l293_293082

theorem minimum_value_of_expression :
  ∃ x y : ℝ, ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 7 ≥ 28 := by
  sorry

end minimum_value_of_expression_l293_293082


namespace mary_flour_l293_293669

-- Defining the conditions
def total_flour : ℕ := 11
def total_sugar : ℕ := 7
def flour_difference : ℕ := 2

-- The problem we want to prove
theorem mary_flour (F : ℕ) (C : ℕ) (S : ℕ)
  (h1 : C + 2 = S)
  (h2 : total_flour = F + C)
  (h3 : S = total_sugar) :
  F = 2 :=
by
  sorry

end mary_flour_l293_293669


namespace stabilize_house_colors_l293_293792

noncomputable def majority_color_changes_can_stabilize : Prop :=
  ∃ (dwarves : Fin 12 → Prop),
  ∃ (color : Fin 12 → Prop),
  (∀ (i : Fin 12), ∃ (friends : Fin 12 → Prop), (∃ (majority_color : Prop),
    (∀ (j : Fin 12), friends j → (color j = majority_color)) ∧ (color i ≠ majority_color)) →
    (color i = majority_color)) →
  (∀ (s : ℕ), ∃ (t : ℕ), t > s ∧ ¬(∃ (i : Fin 12), (∀ (friends : Fin 12 → Prop),
     (∃ (majority_color : Prop),
     (∀ (j : Fin 12), friends j → (color j = majority_color)) ∧ (color i ≠ majority_color)))).


theorem stabilize_house_colors :
  majority_color_changes_can_stabilize :=
sorry

end stabilize_house_colors_l293_293792


namespace number_of_correct_propositions_is_3_l293_293822

theorem number_of_correct_propositions_is_3 :
  ∃ p1 p2 p3 p4 p5 : Prop,
  (p1 = (∀ (T : Triangle), is_isosceles T → (distance_from_midpoint_to_legs T = true))) ∧
  (p2 = (∀ (T : Triangle), is_isosceles T → (height_eq_median_eq_bisector T = true))) ∧
  (p3 = (∀ (T1 T2 : Triangle), (is_isosceles T1 ∧ is_isosceles T2 ∧ 
                                 corresponding_angles_equal T1 T2 ∧ base_angles_equal T1 T2) →
                                 (T1 ≅ T2))) ∧
  (p4 = (∀ (T : Triangle), (∃ (b : Angle), b = 60° ∧ b ∈ angles T) → (is_equilateral T = false))) ∧
  (p5 = (∀ (T : Triangle), is_isosceles T → (axis_of_symmetry_bisects_vertex T = true))) ∧
  (num_correct_propositions [p1, p2, p3, p4, p5] = 3) :=
by
  sorry

end number_of_correct_propositions_is_3_l293_293822


namespace point_coordinates_l293_293554

theorem point_coordinates (M : ℝ × ℝ) 
  (hx : abs M.2 = 3) 
  (hy : abs M.1 = 2) 
  (h_first_quadrant : 0 < M.1 ∧ 0 < M.2) : 
  M = (2, 3) := 
sorry

end point_coordinates_l293_293554


namespace all_draws_l293_293960

variable (n : ℕ) (n_pos : n > 1) (P : (team : Fin (2 * n)) → ℕ)
variable (G : (team : Fin (2 * n)) → Fin (2 * n - 1) → ℕ)

-- Total number of teams
define teams := Fin (2 * n)

-- Number of points and games each team before the last round
def PointsBeforeLastRound (team : teams) : ℕ := (range (2 * n)) sum fun x => P x
def PointsAfterLastRound (team : teams) : ℕ := (PointsBeforeLastRound team) + (G team (2 * n - 1))

-- The ratio condition definition
def RatioCondition (team : teams) : Prop :=
  PointsAfterLastRound team / (2 * n - 1) = PointsBeforeLastRound team / (2 * n - 2)

-- Goal
theorem all_draws (team : teams) (PointsBeforeLastRound) (PointsAfterLastRound) (RatioCondition) : 
  ∀ team : teams, P team = 1 :=
begin
  sorry
end

end all_draws_l293_293960


namespace cube_faces_sum_eq_neg_3_l293_293802

theorem cube_faces_sum_eq_neg_3 
    (a b c d e f : ℤ)
    (h1 : a = -3)
    (h2 : b = a + 1)
    (h3 : c = b + 1)
    (h4 : d = c + 1)
    (h5 : e = d + 1)
    (h6 : f = e + 1)
    (h7 : a + f = b + e)
    (h8 : b + e = c + d) :
  a + b + c + d + e + f = -3 := sorry

end cube_faces_sum_eq_neg_3_l293_293802


namespace trapezoid_is_isosceles_trapezoid_not_necessarily_isosceles_l293_293119

-- Part a: Proof that the trapezoid is isosceles
theorem trapezoid_is_isosceles 
(trapezoid ABCD : Trapezoid)
(mid_M : is_midpoint M A D)
(mid_N : is_midpoint N B C)
(inter_perp_bisects_MN : point_Lies_On_Segment (intersection (perp_bisector AB) (perp_bisector CD)) M N) :
is_isosceles_trapezoid ABCD := sorry

-- Part b: Providing a counterexample
theorem trapezoid_not_necessarily_isosceles 
(trapezoid ABCD : Trapezoid)
(mid_M : is_midpoint M A D)
(mid_N : is_midpoint N B C)
(inter_perp_bisects_line_MN : point_Lies_On_Line (intersection (perp_bisector AB) (perp_bisector CD)) M N) :
¬ is_isosceles_trapezoid ABCD := sorry

end trapezoid_is_isosceles_trapezoid_not_necessarily_isosceles_l293_293119


namespace line_circle_distance_converse_line_circle_distance_l293_293263

structure TangentLineCircle (circle: Circle) (line: Line): Prop :=
(non_intersect: ¬∃ p: Point, Line.contains line p ∧ Circle.contains circle p)
(points_tangent: ∀ (a b: Point), Line.contains line a → Line.contains line b → 
  ∃ c d: Point, TangentPoint c circle a line ∧ TangentPoint d circle b line
  ∧ |a - b| ≤ |a - c| + |b - d|)

theorem line_circle_distance
  {circle: Circle} {line: Line}
  (h: TangentLineCircle circle line):
  ∀ (a b: Point), Line.contains line a  → Line.contains line b →
  |a - b| ≤ |a - c| + |b - d| := sorry

theorem converse_line_circle_distance
  {circle: Circle} {line: Line}
  (h: TangentLineCircle circle line):
  ∀ (a b: Point), Line.contains line a  → Line.contains line b →
  |a - b| > |a - d| - |b - d| → ∃ p: Point, Line.contains line p ∧ Circle.contains circle p := sorry

end line_circle_distance_converse_line_circle_distance_l293_293263


namespace range_of_f_l293_293888

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_pos : ∀ x : ℝ, 0 < x → 0 < f(x)
axiom f_ineq : ∀ x : ℝ, 0 < x → f(x) < deriv (deriv f) x ∧ deriv (deriv f) x < 2 * f(x)

theorem range_of_f : (1 / Real.exp 2) < f 1 / f 2 ∧ f 1 / f 2 < (1 / Real.exp 1) :=
by sorry

end range_of_f_l293_293888


namespace relationship_among_abc_l293_293111

variable {α : Type}
variable (f : ℝ → ℝ)

-- conditions
axiom symmetry_condition : ∀ x, f(1 - x) = f(1 + x)
axiom monotonic_decreasing : ∀ x y, x < y ∧ y ≤ 1 → f(y) ≤ f(x)

-- definitions
def a := f (-1/2)
def b := f (-1)
def c := f 2

-- statement
theorem relationship_among_abc : c < a ∧ a < b := sorry

end relationship_among_abc_l293_293111


namespace annual_income_calculation_l293_293828

theorem annual_income_calculation :
  let stock1_investment := 6800
  let stock1_div_rate := 0.4
  let stock1_price := 136
  let stock1_income := (stock1_investment / stock1_price) * (stock1_div_rate * 100)
  
  let stock2_investment := 3500
  let stock2_div_rate := 0.5
  let stock2_price := 125
  let stock2_income := (stock2_investment / stock2_price) * (stock2_div_rate * 100)
  
  let stock3_investment := 4500
  let stock3_div_rate := 0.3
  let stock3_price := 150
  let stock3_income := (stock3_investment / stock3_price) * (stock3_div_rate * 100)
  
  let total_income := stock1_income + stock2_income + stock3_income

  total_income = 4300 :=
begin
  dsimp [stock1_investment, stock1_div_rate, stock1_price, stock1_income,
         stock2_investment, stock2_div_rate, stock2_price, stock2_income,
         stock3_investment, stock3_div_rate, stock3_price, stock3_income,
         total_income],
  norm_num,
  sorry
end

end annual_income_calculation_l293_293828


namespace range_of_q_l293_293961

noncomputable def a1 : ℝ := 1 / 64
noncomputable def logBase := 0.5

def an (q : ℝ) (n : ℕ) := a1 * q^(n - 1)
def bn (q : ℝ) (n : ℕ) := Real.logBase logBase (an q n)

def B4 (q : ℝ) : ℝ :=
  let b1 := bn q 1
  let b2 := bn q 2
  let b3 := bn q 3
  let b4 := bn q 4
  b1 + b2 + b3 + b4

theorem range_of_q (q : ℝ) :
  (∀ n : ℕ, (n ≠ 4 → B4 q ≤ (∑ k in Finset.range n, bn q k + 1)) ∧
   (B4 q > 0) ∧
   (B4 q < 0)) ↔ (2 * Real.sqrt 2 < q ∧ q < 4) :=
by sorry

end range_of_q_l293_293961


namespace part_1_part_2_l293_293657

-- Define the conditions p and q
def p (a x : ℝ) := x^2 - 4 * a * x + 3 * a^2 < 0
def q (m : ℝ) (x : ℝ) := x = (1 / 2)^(m - 1)

-- Define the statements to be proved
theorem part_1 (x : ℝ) (a : ℝ) (m : ℝ) 
  (h_a : a = 1/4) (h_q : q m x ) 
  (h_m : m ∈ set.Ioo (1:ℝ) 2): 
  p a x → 1/2 < x ∧ x < 3/4 := 
sorry

theorem part_2 (a : ℝ) :
  (∀ (x : ℝ) (m : ℝ), q m x → p a x) ∧ ¬(∀ (x : ℝ) (m : ℝ), p a x → q m x) → 
  1/3 ≤ a ∧ a ≤ 1/2 :=
sorry

end part_1_part_2_l293_293657


namespace Fe_mass_percentage_in_Fe2O3_is_approx_70_l293_293075

def molar_mass_Fe : ℝ := 55.845  -- g/mol
def molar_mass_O : ℝ := 15.999  -- g/mol

def molar_mass_Fe2O3 : ℝ := (2 * molar_mass_Fe) + (3 * molar_mass_O)

def mass_percentage_Fe_in_Fe2O3 : ℝ :=
  ((2 * molar_mass_Fe) / molar_mass_Fe2O3) * 100

theorem Fe_mass_percentage_in_Fe2O3_is_approx_70 :
  abs (mass_percentage_Fe_in_Fe2O3 - 70) < 1 :=
sorry

end Fe_mass_percentage_in_Fe2O3_is_approx_70_l293_293075


namespace intersection_points_chord_length_l293_293932

noncomputable def line_eq (x : ℝ) : ℝ := x
noncomputable def circle_eq (x y : ℝ) : ℝ := (x - 2) ^ 2 + (y - 4) ^ 2

theorem intersection_points :
  (∃ x y : ℝ, line_eq x = y ∧ circle_eq x y = 10 ∧ ((x, y) = (5, 5) ∨ (x, y) = (1, 1))) :=
by
  sorry

theorem chord_length :
  let d := abs (2 - 4) / real.sqrt 2 in
  let r := real.sqrt 10 in
  let length := 2 * real.sqrt (r^2 - d^2) in
  length = 4 * real.sqrt 2 :=
by
  sorry

end intersection_points_chord_length_l293_293932


namespace area_of_gray_region_l293_293365

theorem area_of_gray_region
  (r : ℝ)
  (R : ℝ)
  (diameter_small_circle : ℝ)
  (diameter_small_circle_eq : diameter_small_circle = 4)
  (radius_small_circle : r = diameter_small_circle / 2)
  (radius_large_circle : R = 3 * r) :
  let As := π * r^2,
      AL := π * R^2 in
  AL - As = 32 * π :=
by
  -- Definitions for readability and decoration
  have radius_smaller := diameter_small_circle_eq ▸ radius_small_circle,
  have radius_larger := congr_arg (λ x, 3 * x) radius_smaller,
  let area_smaller := π * (r^2),
  let area_larger := π * (R^2),
  sorry

end area_of_gray_region_l293_293365


namespace fraction_representing_repeating_decimal_l293_293178

theorem fraction_representing_repeating_decimal (x a b : ℕ) (h : x = 35) (h1 : 100 * x - x = 35) 
(h2 : ∃ (a b : ℕ), x = a / b ∧ gcd a b = 1 ∧ a + b = 134) : a + b = 134 := 
sorry

end fraction_representing_repeating_decimal_l293_293178


namespace part1_equation_of_ellipse_part2_rhombus_l293_293896

theorem part1_equation_of_ellipse (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (e : ℝ)
    (h3 : e = 1 / 2) (h4 : (a / 2) * b = √3) : 
    (a = 2 * b) → (c = a / 2) → (a = 2 * sqrt 3) → 
    (b = sqrt 3) → 
    (∀ x y : ℝ, (x / 2) * (x / 2) + (y / sqrt 3) * (y / sqrt 3) = 1) :=
by sorry

theorem part2_rhombus (F₂ : ℝ × ℝ) (A₁ A₂ : ℝ × ℝ) (M N : ℝ × ℝ) 
    (P Q : ℝ × ℝ) (O : ℝ × ℝ) (h1 : F₂ = (1, 0)) (h2 : A₁ = (-2, 0)) 
    (h3 : A₂ = (2, 0)) (h4 : (P, Q) ∈ l₁ := x = 1) :
    quadrilateral_rhombus O P A₂ Q :=
by sorry

end part1_equation_of_ellipse_part2_rhombus_l293_293896


namespace lift_19_times_exceeds_usual_weight_l293_293702

def usual_weight_lifted := 2 * 12 * 20 + 2 * 8 * 10
def new_weight_per_lift := 20 + 15

theorem lift_19_times_exceeds_usual_weight :
  19 * new_weight_per_lift > usual_weight_lifted :=
by
  have h1 : usual_weight_lifted = 640 := by norm_num
  have h2 : new_weight_per_lift = 35 := by norm_num
  calc
    19 * 35 = 665 : by norm_num
    665 > 640 : by norm_num
  qed

end lift_19_times_exceeds_usual_weight_l293_293702


namespace problem_statement_l293_293248

noncomputable def F1 : ℝ × ℝ := (-4, 2 - real.sqrt 2)
noncomputable def F2 : ℝ × ℝ := (-4, 2 + real.sqrt 2)

theorem problem_statement : 
  ∃ h k a b : ℝ, 
  h = -4 ∧ k = 2 ∧ a = 1 ∧ b = 1 ∧ h + k + a + b = 0 :=
by {
  use [-4, 2, 1, 1],
  repeat { split <|> rfl <|> norm_num },
}

end problem_statement_l293_293248


namespace factorize_x_squared_plus_2x_l293_293063

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by
  sorry

end factorize_x_squared_plus_2x_l293_293063


namespace angle_YZX_l293_293004

theorem angle_YZX {A B C : Type} {γ : Circle} {X Y Z : Point} (h_incircle : γ.incircle A B C) 
  (h_circumcircle : γ.circumcircle X Y Z) (hX : X ∈ segment B C) (hY : Y ∈ segment A B) 
  (hZ : Z ∈ segment A C) (angle_A : angle A = 50) (angle_B : angle B = 70) (angle_C : angle C = 60) : 
  angle (segment Y Z) (segment Z X) = 65 := 
sorry

end angle_YZX_l293_293004


namespace compose_shapes_from_quadrilaterals_l293_293031

theorem compose_shapes_from_quadrilaterals :
  (∃ (A B C D : ℝ × ℝ), A = (0,0) ∧ B = (2,0) ∧ C = (2,2) ∧ D = (0,2)) ∧
  (∃ (E F G H : ℝ × ℝ), E = (2,2) ∧ F = (4,2) ∧ G = (4,4) ∧ H = (2,4)) ∧ (
    (∃ (A1 B1 C1 : ℝ × ℝ), A1 = (0, 0) ∧ B1 = (2, 0) ∧ C1 = (2, 2)) ∧ 
    (∃ (A2 B2 C2 D2 E2 : ℝ × ℝ), A2 = (0, 0) ∧ B2 = (2, 0) ∧ C2 = (2, 2) ∧ D2 = (2, 4) ∧ E2 = (0, 2)
  ) ∧ 
    (∃ (A3 B3 C3 : ℝ × ℝ), A3 = (0, 0) ∧ B3 = (2, 0) ∧ C3 = (2, 2)) ∧
    (∃ (A4 B4 C4 D4 : ℝ × ℝ), A4 = (0, 0) ∧ B4 = (2, 0) ∧ C4 = (2, 2) ∧ D4 = (0, 2)) ∧
    (∃ (A5 B5 C5 D5 E5 : ℝ × ℝ), A5 = (0, 0) ∧ B5 = (2, 0) ∧ C5 = (2, 2) ∧ D5 = (2, 4) ∧ E5 = (0, 2)
  ).
Proof: sorry

end compose_shapes_from_quadrilaterals_l293_293031


namespace sum_of_squares_l293_293645

theorem sum_of_squares (n : ℕ) (p : ℕ) (h1 : n > 0) (h2 : Nat.Prime p) (h3 : ∃ x : ℕ, np + 1 = x^2) :
  ∃ (a : ℕ), (n + 1) = ∑ i in finset.range p, (a i)^2 :=
sorry

end sum_of_squares_l293_293645


namespace vector_magnitude_l293_293131

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Given conditions
axiom norm_a : ∥a∥ = 2
axiom norm_b : ∥b∥ = 2
axiom dot_product_condition : ⟪a, b - a⟫ = -2

-- The theorem to be proven
theorem vector_magnitude (h1 : ∥a∥ = 2) (h2 : ∥b∥ = 2) (h3 : ⟪a, b - a⟫ = -2) :
  ∥2 • a - b∥ = 2 * real.sqrt 3 :=
sorry

end vector_magnitude_l293_293131


namespace precise_point_when_perpendicular_l293_293210

-- Define the width of the strips and the properties of the intersection.
variables (w : ℝ) (h : ℝ) (θ : ℝ)
-- Define the condition where the strips intersect at an angle.
def strips_intersect_at_angle (w h θ : ℝ) : Prop :=
  θ ≠ 0 ∧ θ ≠ π / 2 ∧ θ ≠ π

-- Define the properties of the rhombus formed by the intersection of two strips.
def rhombus_intersection (w h : ℝ) : Prop :=
  w = h

-- Define the condition where the intersection is a square.
def intersection_is_square (w h : ℝ) :=
  w = h ∧ θ = π / 2

-- Theorem stating that the point is determined as accurately as possible when strips are perpendicular.
theorem precise_point_when_perpendicular (w h : ℝ) :
  strips_intersect_at_angle w h (π / 2) → intersection_is_square w h :=
by
  intro _,
  sorry

end precise_point_when_perpendicular_l293_293210


namespace conclusion_is_other_l293_293917

-- Define the conditions as premises
variables {Square Rectangle : Type} 
variables (is_square : Square → Prop) (is_rectangle : Rectangle → Prop)
variables (diagonals_equal_square : ∀ (sq : Square), is_square sq → True) -- premise 1
variables (diagonals_equal_rect : ∀ (rect : Rectangle), is_rectangle rect → True) -- premise 2
variables (square_is_rectangle : ∀ (sq : Square), is_square sq → is_rectangle sq) -- premise 3

-- State the theorem: the conclusion based on the premises
theorem conclusion_is_other (sq : Square) : 
  (diagonals_equal_square sq (by assumption) ∧
   diagonals_equal_rect sq (square_is_rectangle sq (by assumption))) 
  → True :=
sorry

end conclusion_is_other_l293_293917


namespace ceil_sum_of_sqr_roots_l293_293039

theorem ceil_sum_of_sqr_roots : 
  (⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27) := 
by {
  -- Definitions based on conditions
  have h1 : 1^2 < 3 ∧ 3 < 2^2, by norm_num,
  have h2 : 5^2 < 33 ∧ 33 < 6^2, by norm_num,
  have h3 : 18^2 < 333 ∧ 333 < 19^2, by norm_num,
  sorry
}

end ceil_sum_of_sqr_roots_l293_293039


namespace unique_solution_l293_293845

noncomputable def valid_solutions (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + (2 - a) * x + 1 = 0 ∧ -1 < x ∧ x ≤ 3 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2

theorem unique_solution (a : ℝ) :
  (valid_solutions a) ↔ (a = 4.5 ∨ (a < 0) ∨ (a > 16 / 3)) := 
sorry

end unique_solution_l293_293845


namespace find_five_digit_number_l293_293382

theorem find_five_digit_number :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∃ rev_n : ℕ, rev_n = (n % 10) * 10000 + (n / 10 % 10) * 1000 + (n / 100 % 10) * 100 + (n / 1000 % 10) * 10 + (n / 10000) ∧ 9 * n = rev_n) ∧ n = 10989 :=
  sorry

end find_five_digit_number_l293_293382


namespace linear_function_passing_through_point_and_intersecting_another_line_area_of_triangle_l293_293140

theorem linear_function_passing_through_point_and_intersecting_another_line (
  k b : ℝ)
  (h1 : (∀ x y : ℝ, y = k * x + b → ((x = 3 ∧ y = -3) ∨ (x = 3/4 ∧ y = 0))))
  (h2 : (∀ x : ℝ, 0 = (4 * x - 3) → x = 3/4))
  : k = -4 / 3 ∧ b = 1 := 
sorry

theorem area_of_triangle (
  k b : ℝ)
  (h1 : k = -4 / 3 ∧ b = 1)
  : 1 / 2 * 3 / 4 * 1 = 3 / 8 := 
sorry

end linear_function_passing_through_point_and_intersecting_another_line_area_of_triangle_l293_293140


namespace find_largest_number_l293_293521

theorem find_largest_number (w x y z : ℕ) 
  (h1 : w + x + y = 190) 
  (h2 : w + x + z = 210) 
  (h3 : w + y + z = 220) 
  (h4 : x + y + z = 235) : 
  max (max w x) (max y z) = 95 := 
sorry

end find_largest_number_l293_293521


namespace period_of_time_l293_293765

-- We define the annual expense and total amount spent as constants
def annual_expense : ℝ := 2
def total_amount_spent : ℝ := 20

-- Theorem to prove the period of time (in years)
theorem period_of_time : total_amount_spent / annual_expense = 10 :=
by 
  -- Placeholder proof
  sorry

end period_of_time_l293_293765


namespace fourth_term_integer_l293_293637

def jacob_sequence : ℕ → ℤ
| 0       := 7 -- Starting term
| (n + 1) :=
  let prev := jacob_sequence n in
  if coin_flip = "heads" then
    if prev % 2 = 0 then -- even
      2 * prev - 1
    else -- odd
      3 * prev + 1
  else -- tails
    if prev % 2 = 0 then -- even
      prev / 2
    else -- odd
      (prev - 1) / 2

theorem fourth_term_integer : 
  ∀ coin_sequence : vector string 3, -- three coin flips
  ∃ (a : ℤ), jacob_sequence 4 = a ∧ a ∈ ℤ := 
sorry

end fourth_term_integer_l293_293637


namespace complex_z_count_l293_293859

-- Translate conditions as definitions
definition is_on_unit_circle (z : ℂ) : Prop :=
  complex.abs z = 1

-- Translate the problem statement
theorem complex_z_count (z : ℂ) (h : is_on_unit_circle z) :
  (z ^ 7.factorial - z ^ 6.factorial).im = 0 ↔
  ∃ n : ℕ, n = 7200 :=
sorry

end complex_z_count_l293_293859


namespace correct_sampling_methods_l293_293787

-- Definitions from the conditions
def families : Nat := 800
def high_income_families : Nat := 200
def middle_income_families : Nat := 480
def low_income_families : Nat := 120
def sample_size : Nat := 100
def students : Nat := 10
def students_to_select : Nat := 3

-- Sampling methods
inductive SamplingMethod
| simple_random 
| systematic 
| stratified

-- Problem statements
def sample_families_method : SamplingMethod := SamplingMethod.stratified
def select_students_method : SamplingMethod := SamplingMethod.simple_random

theorem correct_sampling_methods :
  (sample_families_method = SamplingMethod.stratified) ∧ 
  (select_students_method = SamplingMethod.simple_random) :=
by
  split
  · sorry
  · sorry

end correct_sampling_methods_l293_293787


namespace altitude_inequality_altitude_inequality_condition_l293_293116

-- Definitions of the conditions
variables {A B C : Type} (triangle_ABC : Triangle A B C)
variable [obtuse_angle_A : ObtuseAngle triangle_ABC A]

-- Definition of the altitudes
variables (h k : ℝ)

-- Definitions of the sides of the triangle
variables (a b c : ℝ) -- lengths of sides opposite to angles at vertices A, B, and C respectively

-- Proving the required inequality with the given conditions
theorem altitude_inequality
  (h_def : h = b * sin C)
  (k_def : k = a * sin C) :
  a + h ≥ b + k :=
by sorry

-- Condition for equality
theorem altitude_inequality_condition
  (h_def : h = b * sin C)
  (k_def : k = a * sin C)
  (obtuse_condition : ObtuseAngle triangle_ABC A) :
  a + h = b + k ↔ a = b :=
by sorry

end altitude_inequality_altitude_inequality_condition_l293_293116


namespace tangent_line_to_circle_tangent_lines_through_point_l293_293898

theorem tangent_line_to_circle (a : ℝ) :
  ∀ (M : ℝ × ℝ) (C : ℝ × ℝ) (R : ℝ),
  M = (3, 1) →
  C = (1, 2) →
  R = 2 →
  line_tangent_to_circle (ax - y + 4 = 0) ((x - 1)^2 + (y - 2)^2 = 4) →
  (a = 0 ∨ a = 4 / 3) :=
begin
  intros M C R hM hC hR hTangent,
  sorry
end

theorem tangent_lines_through_point (M : ℝ × ℝ) (C : ℝ × ℝ) (R : ℝ) :
  M = (3, 1) →
  C = (1, 2) →
  R = 2 →
  tangent_lines_to_circle_through_point ((x - 1)^2 + (y - 2)^2 = 4) M =
  { x = 3, 3x - 4y - 5 = 0 } :=
begin
  intros hM hC hR,
  sorry
end

end tangent_line_to_circle_tangent_lines_through_point_l293_293898


namespace ratio_KC_div_MD_l293_293963

-- Noncomputable theory due to use of ratios
noncomputable theory

-- Define the points and their relationships in the parallelogram
variable (A B C D K M E: Type)
variable [AddCommGroup A] [VectorSpace ℝ A]

-- Points A, B, C, and D forming the parallelogram
variable [∀(P : A), AffineSpace ℝ A]

-- Define the conditions of the problem
variable (ABCD: A)
variable (hABCD_parallelogram : IsParallelogram ABCD)

-- Define midpoints
variable (K_mid_AB: K = midpoint A B)
variable (M_mid_BC: M = midpoint B C)

-- Define the intersection
variable (E_intersection: E = intersection (KC) (MD))

-- The statement to be proven
theorem ratio_KC_div_MD (hABCD_parallelogram: IsParallelogram ABCD)
                         (K_mid_AB: K = midpoint A B)
                         (M_mid_BC: M = midpoint B C)
                         (E_intersection: E = intersection (KC) (MD)) :
    let ME := dist M E in
    let ED := dist E D in
    ME / ED = 1 / 4 :=
begin
  sorry
end

end ratio_KC_div_MD_l293_293963


namespace point_on_angle_bisector_l293_293593

theorem point_on_angle_bisector (a : ℝ) 
  (h : (2 : ℝ) * a + (3 : ℝ) = a) : a = -3 :=
sorry

end point_on_angle_bisector_l293_293593


namespace largest_mersenne_prime_under_1000_l293_293795

def is_prime (n : ℕ) : Prop := Sorry -- Assume a definition for primality

def mersenne_prime (p : ℕ) : Prop := is_prime p ∧ ∃ (n : ℕ), is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_under_1000 : ∀ (p : ℕ), mersenne_prime p ∧ p < 1000 → p ≤ 127 :=
by
  sorry

end largest_mersenne_prime_under_1000_l293_293795


namespace find_heaviest_and_lightest_l293_293327

-- Definition of the main problem conditions
def coins : ℕ := 10
def max_weighings : ℕ := 13
def distinct_weights (c : ℕ) : Prop := ∀ (i j : ℕ), i ≠ j → i < c → j < c → weight i ≠ weight j

-- Noncomputed property representing the weight of each coin
noncomputable def weight : ℕ → ℝ := sorry

-- The main theorem statement
theorem find_heaviest_and_lightest (c : ℕ) (mw : ℕ) (dw : distinct_weights c) : c = coins ∧ mw = max_weighings
  → ∃ (h l : ℕ), h < c ∧ l < c ∧ (∀ (i : ℕ), i < c → weight i ≤ weight h ∧ weight i ≥ weight l) :=
by
  sorry

end find_heaviest_and_lightest_l293_293327


namespace DanAgeIs12_l293_293841

def DanPresentAge (x : ℕ) : Prop :=
  (x + 18 = 5 * (x - 6))

theorem DanAgeIs12 : ∃ x : ℕ, DanPresentAge x ∧ x = 12 :=
by
  use 12
  unfold DanPresentAge
  sorry

end DanAgeIs12_l293_293841


namespace s_range_l293_293877

noncomputable def s (n : ℕ) : ℕ :=
if h : n > 1 ∧ Nat.PrimeFactorization.prod (n.factorization) = n ∧ 3 ∉ n.factorization.support
then n.factorization.sum (λ p k, p * k)
else 0

theorem s_range :
  {m : ℕ | ∃ n : ℕ, n > 1 ∧ 3 ∉ n.factorization.support ∧ Nat.PrimeFactorization.prod (n.factorization) = n ∧ s(n) = m}
  = {m : ℕ | m > 3} :=
sorry

end s_range_l293_293877


namespace complex_modulus_l293_293144

theorem complex_modulus (z : ℂ) (h : (1 + complex.I) / z = 1 - complex.I) : ∥z∥ = 1 :=
sorry

end complex_modulus_l293_293144


namespace reroll_two_dice_probability_l293_293217

theorem reroll_two_dice_probability : 
  (∃ d1 d2 d3 : ℕ, 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6 ∧ 
  let sum := d1 + d2 + d3 in 
    sum = 9 ∧ 
    (d1 = 3 ∧ d2 = 6 ∧ d3 = 6 ∨ 
     d1 = 3 ∧ d2 = 5 ∧ d3 = 5 ∨ 
     d1 = 2 ∧ d2 = 6 ∧ d3 = 7 ∨ 
     -- ... other conditions here ...
     true)) →
  (probability_of_rerolling_two_dice = 1 / 6) := sorry

end reroll_two_dice_probability_l293_293217


namespace eval_sum_sqrt_ceil_l293_293044

theorem eval_sum_sqrt_ceil:
  ∀ (x : ℝ), 
  (1 < sqrt 3 ∧ sqrt 3 < 2) ∧
  (5 < sqrt 33 ∧ sqrt 33 < 6) ∧
  (18 < sqrt 333 ∧ sqrt 333 < 19) →
  (⌈ sqrt 3 ⌉ + ⌈ sqrt 33 ⌉ + ⌈ sqrt 333 ⌉ = 27) :=
by
  intro x
  sorry

end eval_sum_sqrt_ceil_l293_293044


namespace sundae_cost_l293_293580

theorem sundae_cost (ice_cream_cost toppings_cost : ℕ) (num_toppings : ℕ) :
  ice_cream_cost = 200  →
  toppings_cost = 50 →
  num_toppings = 10 →
  ice_cream_cost + num_toppings * toppings_cost = 700 := by
  sorry

end sundae_cost_l293_293580


namespace wendy_total_sales_l293_293360

noncomputable def apple_price : ℝ := 1.50
noncomputable def orange_price : ℝ := 1.00
noncomputable def morning_apples : ℕ := 40
noncomputable def morning_oranges : ℕ := 30
noncomputable def afternoon_apples : ℕ := 50
noncomputable def afternoon_oranges : ℕ := 40

theorem wendy_total_sales :
  (morning_apples * apple_price + morning_oranges * orange_price) +
  (afternoon_apples * apple_price + afternoon_oranges * orange_price) = 205 := by
  sorry

end wendy_total_sales_l293_293360


namespace fraction_unshaded_area_l293_293964

theorem fraction_unshaded_area (s : ℝ) :
  let P := (s / 2, 0)
  let Q := (s, s / 2)
  let top_left := (0, s)
  let area_triangle : ℝ := 1 / 2 * (s / 2) * (s / 2)
  let area_square : ℝ := s * s
  let unshaded_area : ℝ := area_square - area_triangle
  let fraction_unshaded : ℝ := unshaded_area / area_square
  fraction_unshaded = 7 / 8 := 
by 
  sorry

end fraction_unshaded_area_l293_293964


namespace ceil_sqrt_sum_l293_293050

theorem ceil_sqrt_sum : 
  (⌈Real.sqrt 3⌉ = 2) ∧ 
  (⌈Real.sqrt 33⌉ = 6) ∧ 
  (⌈Real.sqrt 333⌉ = 19) → 
  2 + 6 + 19 = 27 :=
by 
  intro h
  cases h with h3 h
  cases h with h33 h333
  rw [h3, h33, h333]
  norm_num

end ceil_sqrt_sum_l293_293050


namespace min_m_for_integral_solutions_l293_293771

theorem min_m_for_integral_solutions :
  ∃ m : ℕ, (∀ x : ℚ, 10 * x^2 - m * x + 780 = 0 → x ∈ ℤ) ∧ m = 190 :=
by
  sorry

end min_m_for_integral_solutions_l293_293771


namespace problem_statement_l293_293341

noncomputable def perimeter_rectangle 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (area_triangle : ℝ := (1/2) * a * b) 
  (area_rectangle : ℝ := area_triangle) 
  (l : ℝ := area_rectangle / w) : ℝ :=
2 * (w + l)

theorem problem_statement 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h_a : a = 9) 
  (h_b : b = 12) 
  (h_c : c = 15) 
  (h_w : w = 6) : 
  perimeter_rectangle a b c w h1 = 30 :=
by 
  sorry

end problem_statement_l293_293341


namespace factorize_polynomial_l293_293496

theorem factorize_polynomial :
  ∀ (x : ℝ), x^4 + 2021 * x^2 + 2020 * x + 2021 = (x^2 + x + 1) * (x^2 - x + 2021) :=
by
  intros x
  sorry

end factorize_polynomial_l293_293496


namespace range_of_a_in_fourth_quadrant_l293_293156

noncomputable def z1 (a : ℝ) : ℂ := 3 - a * complex.I
noncomputable def z2 : ℂ := 1 + 2 * complex.I
noncomputable def z (a : ℝ) : ℂ := z1 a / z2

theorem range_of_a_in_fourth_quadrant (a : ℝ) :
  (0 < (z a).re ∧ (z a).im < 0) ↔ (-6 < a ∧ a < 3 / 2) := by
  sorry

end range_of_a_in_fourth_quadrant_l293_293156


namespace measure_YZX_l293_293009

-- Define the triangle ABC with given angles
axiom triangle_ABC (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) : Prop :=
  angle_A = 50 ∧ angle_B = 70 ∧ angle_C = 60 ∧ angle_A + angle_B + angle_C = 180

-- Define the points X, Y, Z on the sides of triangle ABC
axiom points_on_sides (X Y Z : Type) (B C A B' C' A' : Type) : Prop :=
  X ∈ (B' ∩ C') ∧ Y ∈ (A' ∩ B) ∧ Z ∈ (A ∩ C')

-- Define the circle Gamma as incircle and circumcircle of triangles
axiom circle_incircle_circumcircle (Gamma : Type) (triangle_ABC triangle_XYZ : Type) : Prop :=
  incircle Gamma triangle_ABC ∧ circumcircle Gamma triangle_XYZ

-- Define the measure of YZX angle
def angle_YZX (angle_B angle_C : ℝ) : ℝ :=
  180 - (angle_B / 2 + angle_C / 2)

-- The main theorem to prove
theorem measure_YZX : ∀ (A B C X Y Z : Type) (Gamma : Type),
    triangle_ABC A B C 50 70 60 →
    points_on_sides X Y Z B C A B C A →
    circle_incircle_circumcircle Gamma (triangle_ABC A B C) (triangle_XYZ X Y Z) →
    angle_YZX 70 60 = 115 := 
by 
  intros A B C X Y Z Gamma hABC hXYZ hGamma
  sorry

end measure_YZX_l293_293009


namespace inner_cone_volume_l293_293354

noncomputable def volumeInnerCone (R : ℝ) (α : ℝ) : ℝ :=
  (1 / 3) * π * R^3 * (cos ((π / 4) - (α / 2)))^3 * cot α

theorem inner_cone_volume (R : ℝ) (α : ℝ) 
  (hα : 0 < α ∧ α < π) 
  (h_lateral : let S_outer_total := π * R^2 * (1 / sin α + 1) in 
               let S_inner_lateral := (1 / 2) * S_outer_total in 
               S_inner_lateral = ((π * R^2 * (1 + sin α)) / (2 * sin α))) :
  let S_inner := R * cos ((π / 4) - (α / 2)) in
  let V_inner := (1 / 3) * π * (S_inner^2) * (S_inner * cot α) in
  V_inner = volumeInnerCone R α := sorry

end inner_cone_volume_l293_293354


namespace equation_of_line_through_point_with_equal_intercepts_l293_293504

open LinearAlgebra

theorem equation_of_line_through_point_with_equal_intercepts :
  ∃ (a b c : ℝ), (a * 1 + b * 2 + c = 0) ∧ (a * b < 0) ∧ ∀ x y : ℝ, 
  (a * x + b * y + c = 0 ↔ (2 * x - y = 0 ∨ x + y - 3 = 0)) :=
sorry

end equation_of_line_through_point_with_equal_intercepts_l293_293504


namespace log3_81_between_consecutive_ints_l293_293732

theorem log3_81_between_consecutive_ints :
  ∃ a b : ℤ, (a < log 3 81 ∧ log 3 81 < b) ∧ a + b = 9 :=
by {
  use [4, 5],
  split,
  { 
    exact ⟨by norm_num, by norm_num⟩},
  { 
    exact rfl}
}

end log3_81_between_consecutive_ints_l293_293732


namespace carbonated_water_solution_l293_293437

variable (V V_1 V_2 : ℝ)
variable (C2 : ℝ)

def carbonated_water_percent (V V1 V2 C2 : ℝ) : Prop :=
  0.8 * V1 + C2 * V2 = 0.6 * V

theorem carbonated_water_solution :
  ∀ (V : ℝ),
  (V1 = 0.1999999999999997 * V) →
  (V2 = 0.8000000000000003 * V) →
  carbonated_water_percent V V1 V2 C2 →
  C2 = 0.55 :=
by
  intros V V1_eq V2_eq carbonated_eq
  sorry

end carbonated_water_solution_l293_293437


namespace AB_in_rectangle_l293_293619

theorem AB_in_rectangle (ABCD_is_rectangle : rectangle ABCD)
                        (P_on_BC : P ∈ line_of_segment (B, C))
                        (BP_eq : dist B P = 12)
                        (CP_eq : dist C P = 4)
                        (tan_APD : tan (angle A P D) = 3 / 2) : 
  dist A B = 14 :=
sorry

end AB_in_rectangle_l293_293619


namespace charlie_and_dana_proof_l293_293814

noncomputable def charlie_and_dana_ways 
    (cookies : ℕ) (smoothies : ℕ) (total_items : ℕ) 
    (distinct_charlie : ℕ) 
    (repeatable_dana : ℕ) : ℕ :=
    if cookies = 8 ∧ smoothies = 5 ∧ total_items = 5 ∧ distinct_charlie = 0 
       ∧ repeatable_dana = 0 then 27330 else 0

theorem charlie_and_dana_proof :
  charlie_and_dana_ways 8 5 5 0 0 = 27330 := 
  sorry

end charlie_and_dana_proof_l293_293814


namespace simplify_fraction_expression_l293_293271

theorem simplify_fraction_expression :
  5 * (12 / 7) * (49 / (-60)) = -7 := 
sorry

end simplify_fraction_expression_l293_293271


namespace num_zeros_of_function_l293_293150

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then 2^x else Real.log2 x

noncomputable def g : ℝ → ℝ :=
λ x, f (f x) - (1 / 2)

def num_zeros (g : ℝ → ℝ) : ℕ :=
set.count {x : ℝ | g x = 0}

theorem num_zeros_of_function :
  num_zeros g = 2 :=
sorry

end num_zeros_of_function_l293_293150


namespace second_player_always_wins_l293_293816

theorem second_player_always_wins (n : ℕ) (hn : n > 1) : 
  ∃ strategy : (ℕ → ℕ), (∀ first_move, ∀ (second_move := strategy first_move), ∃ longest_arc : ℝ, 
    second_player_arc longest_arc ∧ longest_arc > first_player_longest_arc first_move) := sorry

def second_player_arc (arc_length : ℝ) : Prop := -- Definitions follow from conditions
  -- Conditions involving arc length and properties of marking
  sorry

def first_player_longest_arc (first_move : ℕ) : ℝ := 
  -- Function giving the longest arc for the first player's move
  sorry

end second_player_always_wins_l293_293816


namespace total_amount_spent_correct_l293_293176

theorem total_amount_spent_correct :
  let price_life_journey := 100
      price_day_life := 50
      price_rescind := 85
      num_life_journey := 3
      num_day_life := 4
      num_rescind := 2
      total_before_discount := (num_life_journey * price_life_journey) +
                               (num_day_life * price_day_life) +
                               (num_rescind * price_rescind)
      total_cds := num_life_journey + num_day_life + num_rescind
      discount := if total_cds >= 10 then 0.2
                  else if total_cds >= 6 then 0.15
                  else if total_cds >= 3 then 0.10
                  else 0.0
  in total_before_discount * (1 - discount) = 569.50 :=
by
  sorry

end total_amount_spent_correct_l293_293176


namespace decreasing_interval_of_g_l293_293664

noncomputable def f : ℝ → ℝ := sorry

def g (x : ℝ) : ℝ := x^2 * f (x - 1)

theorem decreasing_interval_of_g : (∀ x ∈ Icc (0 : ℝ) (1 : ℝ), ∃ ε > 0, x + ε ∈ Icc (0 : ℝ) (1 : ℝ) → g (x + ε) < g x) :=
sorry

end decreasing_interval_of_g_l293_293664


namespace greatest_perimeter_triangle_l293_293199

theorem greatest_perimeter_triangle :
  ∃ (x : ℕ), (x > (16 / 5)) ∧ (x < (16 / 3)) ∧ ((x = 4 ∨ x = 5) → 4 * x + x + 16 = 41) :=
by
  sorry

end greatest_perimeter_triangle_l293_293199


namespace equation_of_line_through_point_with_equal_intercepts_l293_293505

open LinearAlgebra

theorem equation_of_line_through_point_with_equal_intercepts :
  ∃ (a b c : ℝ), (a * 1 + b * 2 + c = 0) ∧ (a * b < 0) ∧ ∀ x y : ℝ, 
  (a * x + b * y + c = 0 ↔ (2 * x - y = 0 ∨ x + y - 3 = 0)) :=
sorry

end equation_of_line_through_point_with_equal_intercepts_l293_293505


namespace log_value_between_integers_l293_293303

theorem log_value_between_integers : (1 : ℤ) < Real.log 25 / Real.log 10 ∧ Real.log 25 / Real.log 10 < (2 : ℤ) → 1 + 2 = 3 :=
by
  sorry

end log_value_between_integers_l293_293303


namespace complex_multiplication_l293_293207

theorem complex_multiplication {z : ℂ} (hz : z = 2 - complex.I) : 
  complex.I^3 * z = -1 - 2 * complex.I := 
sorry

end complex_multiplication_l293_293207


namespace sum_of_first_25_even_numbers_l293_293087

theorem sum_of_first_25_even_numbers : ∑ i in finset.range 25, (2 * (i + 1)) = 650 := by
  sorry

end sum_of_first_25_even_numbers_l293_293087


namespace number_of_juniors_l293_293962

variables (J S x : ℕ)

theorem number_of_juniors (h1 : (2 / 5 : ℚ) * J = x)
                          (h2 : (1 / 4 : ℚ) * S = x)
                          (h3 : J + S = 30) :
  J = 11 :=
sorry

end number_of_juniors_l293_293962


namespace find_values_l293_293179

open Real

noncomputable def positive_numbers (x y : ℝ) := x > 0 ∧ y > 0

noncomputable def given_condition (x y : ℝ) := (sqrt (12 * x) * sqrt (20 * x) * sqrt (4 * y) * sqrt (25 * y) = 50)

theorem find_values (x y : ℝ) 
  (h1: positive_numbers x y) 
  (h2: given_condition x y) : 
  x * y = sqrt (25 / 24) := 
sorry

end find_values_l293_293179


namespace triangle_ineq_l293_293243

variable {a b c S : ℝ}

axiom law_of_cosines (a b c : ℝ) (C : ℝ) : a^2 + b^2 - c^2 = 2 * a * b * Real.cos C
axiom triangle_area (a b c : ℝ) (S : ℝ) : S = 1 / 2 * a * b * Real.sin (Real.acos ((a^2 + b^2 - c^2) / (2 * a * b)))

theorem triangle_ineq (a b c S : ℝ)
  (cond1 : a^2 + b^2 - c^2 = 2 * a * b * Real.cos C)
  (cond2 : S = 1 / 2 * a * b * Real.sin (Real.acos ((a^2 + b^2 - c^2) / (2 * a * b)))):
  c^2 - a^2 - b^2 + 4 * a * b ≥ 4 * Real.sqrt 3 * S := 
sorry

end triangle_ineq_l293_293243


namespace share_apples_l293_293690

theorem share_apples (h : 9 / 3 = 3) : true :=
sorry

end share_apples_l293_293690


namespace mary_repayment_time_l293_293781

theorem mary_repayment_time :
  ∀ (S a r : ℕ), S = 819200 → a = 400 → r = 2 → (∃ n, S = a * (r^n - 1) / (r - 1) ∧ n = 11) :=
by
  intros S a r hS ha hr
  simp [hS, ha, hr]
  -- The following is the problem statement converted to an existential equality match
  use 11
  split
  sorry
  exact rfl

end mary_repayment_time_l293_293781


namespace no_such_quadruple_exists_l293_293866

theorem no_such_quadruple_exists :
  ∀ (a b c d : ℝ), 
    (Matrix.det (Matrix.of ![![a, b], ![c, d]]) ≠ 0) →
    (Matrix.inv (Matrix.of ![![a, b], ![c, d]]) = Matrix.of ![![1/(a + 1), 1/(b + 1)], ![1/(c + 1), 1/(d + 1)]]) →
    false := 
begin
  assume a b c d h_det h_inv,
  sorry
end

end no_such_quadruple_exists_l293_293866


namespace jay_change_l293_293984

theorem jay_change (book_price pen_price ruler_price payment : ℕ) (h1 : book_price = 25) (h2 : pen_price = 4) (h3 : ruler_price = 1) (h4 : payment = 50) : 
(book_price + pen_price + ruler_price ≤ payment) → (payment - (book_price + pen_price + ruler_price) = 20) :=
by
  intro h
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end jay_change_l293_293984


namespace memory_card_cost_l293_293991

noncomputable def cost_per_memory_card (total_pictures : ℤ) (pictures_per_card : ℤ) (total_cost : ℤ) : ℤ :=
  total_cost / (total_pictures / pictures_per_card)

theorem memory_card_cost
  (pictures_per_day : ℤ)
  (days_per_year : ℤ)
  (years : ℤ)
  (pictures_per_card : ℤ)
  (total_cost : ℤ) :
  cost_per_memory_card (pictures_per_day * days_per_year * years) pictures_per_card total_cost = 60 :=
by
  sorry

#eval memory_card_cost 10 365 3 50 13140

end memory_card_cost_l293_293991


namespace rectangle_perimeter_l293_293343

open Real

def triangle_DEF_sides : ℝ × ℝ × ℝ := (9, 12, 15) -- sides of the triangle DEF

def rectangle_width : ℝ := 6 -- width of the rectangle

theorem rectangle_perimeter (a b c width : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : width = 6) :
  2 * (54 / width + width) = 30 :=
by
  sorry -- Proof is omitted as required

end rectangle_perimeter_l293_293343


namespace num_solutions_correct_l293_293863

noncomputable def num_solutions : ℕ :=
  let factorial (n : ℕ) := (nat.factorial n)
  let z_is_on_unit_circle (z : ℂ) : Prop := abs z = 1
  let expression_is_real (z : ℂ) : Prop := z^(factorial 7) - z^(factorial 6) ∈ ℝ
  
  nat.card { z : ℂ // z_is_on_unit_circle z ∧ expression_is_real z }

theorem num_solutions_correct : num_solutions = 44 := by
  sorry

end num_solutions_correct_l293_293863


namespace max_handshakes_27_people_l293_293411

theorem max_handshakes_27_people : ∀ {n : ℕ}, n = 27 → (n * (n - 1)) / 2 = 351 :=
by
  intros n h
  rw [h]
  norm_num

end max_handshakes_27_people_l293_293411


namespace find_population_Y_l293_293358

noncomputable def current_population_of_village_Y (P : ℕ) : Prop :=
  let population_X := 74000 - 1200 * 16
  in let population_Y := P + 800 * 16
  in population_Y = population_X

theorem find_population_Y : ∃ P : ℕ, current_population_of_village_Y P ∧ P = 42000 :=
by
  use 42000
  unfold current_population_of_village_Y
  simp
  sorry

end find_population_Y_l293_293358


namespace triangle_is_isosceles_l293_293539

theorem triangle_is_isosceles
  (α β γ x y z w : ℝ)
  (h1 : α + β + γ = 180)
  (h2 : α + β = x)
  (h3 : β + γ = y)
  (h4 : γ + α = z)
  (h5 : x + y + z + w = 360) : 
  (α = β ∧ β = γ) ∨ (α = γ ∧ γ = β) ∨ (β = α ∧ α = γ) := by
  sorry

end triangle_is_isosceles_l293_293539


namespace angle_YZX_l293_293005

theorem angle_YZX {A B C : Type} {γ : Circle} {X Y Z : Point} (h_incircle : γ.incircle A B C) 
  (h_circumcircle : γ.circumcircle X Y Z) (hX : X ∈ segment B C) (hY : Y ∈ segment A B) 
  (hZ : Z ∈ segment A C) (angle_A : angle A = 50) (angle_B : angle B = 70) (angle_C : angle C = 60) : 
  angle (segment Y Z) (segment Z X) = 65 := 
sorry

end angle_YZX_l293_293005


namespace quadrilateral_area_80_l293_293628

structure Quadrilateral :=
(A B C D : ℝ)
(angle_BCD_right : angle B C D = 90)
(AB BC CD AD : ℝ)
(AB : 15) 
(BC : 5) 
(CD : 12) 
(AD : 13)

theorem quadrilateral_area_80 (quad: Quadrilateral) : 
  let area (quad: Quadrilateral) := 
    1 / 2 * quad.BC * quad.CD + 
    let s := (quad.AB + 13 + quad.AD) / 2 in
    Real.sqrt (s * (s - quad.AB) * (s - 13) * (s - quad.AD))
  in area quad = 80 := 
sorry

end quadrilateral_area_80_l293_293628


namespace explicit_formula_minimum_φ_l293_293573

noncomputable def m (a x : ℝ) : ℝ × ℝ := (2 * a * cos x, sin x)
noncomputable def n (x b : ℝ) : ℝ × ℝ := (cos x, b * cos x)
noncomputable def f (a b x : ℝ) : ℝ := (m a x).1 * (n x b).1 + (m a x).2 * (n x b).2 - (sqrt 3 / 2)

def intercept_condition (a : ℝ) : Prop := f a 1 0 = sqrt 3 / 2
def highest_point_condition (a b : ℝ) : Prop := f a b (π / 12) = 1
def formula_condition (a : ℝ) (b := 1) : Prop := ∀ x, f a b x = sin (2 * x + π / 3)

def transformed_formula_condition (φ : ℝ) : Prop :=
  ∀ x, f (sqrt 3 / 2) 1 (x / 2 - φ) = sin x

theorem explicit_formula (a : ℝ) (b : ℝ) : intercept_condition a → highest_point_condition a b → formula_condition a b := by
  sorry

theorem minimum_φ (φ : ℝ) : φ > 0 → transformed_formula_condition φ → φ = 5 * π / 6 := by
  sorry

end explicit_formula_minimum_φ_l293_293573


namespace double_sum_example_l293_293494

theorem double_sum_example :
  (∑ i in Finset.range(50)+1, ∑ j in Finset.range(50)+1, 2 * (i + j)) = 255000 := by
  sorry

end double_sum_example_l293_293494


namespace range_y_l293_293587

variable (x : ℝ) (hx1 : 0 < x) (hx2 : x ≤ π / 3)

def y (x : ℝ) : ℝ := (Real.sin x * Real.cos x + 1) / (Real.sin x + Real.cos x)

theorem range_y :
  set.range (y ∘ λ x, (hx1 : 0 < x, hx2 : x ≤ π / 3)) = set.Ioc (3 / 2) (3 * Real.sqrt 2 / 4) := sorry

end range_y_l293_293587


namespace centroid_circle_area_l293_293258

theorem centroid_circle_area (A B C : Point) (r : ℝ)
  (h1 : AB = 32)
  (h2 : is_diameter A B)
  (h3 : C ∈ circle_centered_on A B)
  (h4 : C ≠ A)
  (h5 : C ≠ B) : 
  area_of_curve_traced_by_centroid A B C = (256 * real.pi) / 9 :=
sorry

end centroid_circle_area_l293_293258


namespace log_T_approximation_l293_293236

noncomputable def T (n : ℕ) : ℝ :=
  (2 + complex.i)^n + (2 - complex.i)^n

theorem log_T_approximation (n : ℕ) :
  let t := (T n) / 2
  in log t / log 10 = n * log 2 / log 10 :=
by
  sorry

end log_T_approximation_l293_293236


namespace heaviest_and_lightest_in_13_weighings_l293_293317

/-- Given ten coins of different weights and a balance scale.
    Prove that it is possible to identify the heaviest and the lightest coin
    within 13 weighings. -/
theorem heaviest_and_lightest_in_13_weighings
  (coins : Fin 10 → ℝ)
  (h_different: ∀ i j : Fin 10, i ≠ j → coins i ≠ coins j)
  : ∃ (heaviest lightest : Fin 10),
      (heaviest ≠ lightest) ∧
      (∀ i : Fin 10, coins i ≤ coins heaviest) ∧
      (∀ i : Fin 10, coins lightest ≤ coins i) :=
sorry

end heaviest_and_lightest_in_13_weighings_l293_293317


namespace real_a_from_pure_imaginary_l293_293127

theorem real_a_from_pure_imaginary (a : ℝ) (h : ∃ b : ℝ, (4 + (a - 2) * complex.I) / complex.I = b * complex.I) : a = 2 :=
sorry

end real_a_from_pure_imaginary_l293_293127


namespace total_length_figure_2_l293_293818

-- Define the conditions for Figure 1
def left_side_figure_1 := 10
def right_side_figure_1 := 7
def top_side_figure_1 := 3
def bottom_side_figure_1_seg1 := 2
def bottom_side_figure_1_seg2 := 1

-- Define the conditions for Figure 2 after removal
def left_side_figure_2 := left_side_figure_1
def right_side_figure_2 := right_side_figure_1
def top_side_figure_2 := 0
def bottom_side_figure_2 := top_side_figure_1 + bottom_side_figure_1_seg1 + bottom_side_figure_1_seg2

-- The Lean statement proving the total length in Figure 2
theorem total_length_figure_2 : 
  left_side_figure_2 + right_side_figure_2 + top_side_figure_2 + bottom_side_figure_2 = 23 := by
  sorry

end total_length_figure_2_l293_293818


namespace find_range_a_l293_293286

def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := 
  ∀ x y ∈ s, x < y → f x > f y

noncomputable def f (a : ℝ) : ℝ → ℝ := 
λ x, if 0 ≤ x then (a - 5) * x - 1 else (x + a) / (x - 1)

theorem find_range_a :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ∈ Ioc (-1 : ℝ) 1 := 
sorry

end find_range_a_l293_293286


namespace prove_sin_value_l293_293105

variable (α : ℝ)
variable (cos_condition : cos(α + π / 3) = 1 / 3)
variable (alpha_condition : 0 < α ∧ α < π / 2)

noncomputable def sin_2alpha_plus_pi_over_6 : ℝ :=
  sin (2 * α + π / 6)

theorem prove_sin_value : sin_2alpha_plus_pi_over_6 α = 7 / 9 :=
by
  have h1 : cos (α + π / 3) = 1 / 3 := cos_condition
  have h2 : 0 < α ∧ α < π / 2 := alpha_condition
  sorry

end prove_sin_value_l293_293105


namespace minimize_distance_l293_293533

-- Definitions provided as conditions in the problem
variables {A B C D E F G X Y : Point}
variables (b e : ℝ) -- lengths defined in the solution

-- Assume X is on segment EG, and Y is the reflection of X through G
variable [hX : on_segment E G X]
variable [hY : reflection_across_point G X Y]

theorem minimize_distance :
  minimize_sum (dist AX + dist DX + dist XY + dist YB + dist YC) = if (e / sqrt 3) >= b
    then G -- when $\frac{e}{\sqrt{3}} \geq b$,
    else b - (e / sqrt 3) -- otherwise $x = b - \frac{e}{\sqrt{3}}$.
:= 
begin
  sorry, -- the proof is omitted as per instructions
end

end minimize_distance_l293_293533


namespace sundae_cost_l293_293583

def ice_cream_cost := 2.00
def topping_cost := 0.50
def number_of_toppings := 10

theorem sundae_cost : ice_cream_cost + topping_cost * number_of_toppings = 7.00 := 
by
  sorry

end sundae_cost_l293_293583


namespace count_real_z5_of_z30_eq_1_l293_293738

theorem count_real_z5_of_z30_eq_1 : 
  ∃ zs : Finset ℂ, (zs.card = 30) ∧ (∀ z ∈ zs, z ^ 30 = 1) ∧ Finset.card ({z ∈ zs | ∃ r : ℝ, (z ^ 5 : ℂ) = r}) = 12 := 
sorry

end count_real_z5_of_z30_eq_1_l293_293738


namespace line_through_point_with_equal_intercepts_l293_293511

-- Definition of the conditions
def point := (1 : ℝ, 2 : ℝ)
def eq_intercepts (line : ℝ → ℝ) := ∃ a b : ℝ, a = b ∧ (∀ x, line x = b - x * (b/a))

-- The proof statement
theorem line_through_point_with_equal_intercepts (line : ℝ → ℝ) : 
  (line 1 = 2 ∧ eq_intercepts line) → (line = (λ x, 2 * x) ∨ line = (λ x, 3 - x)) :=
by
  sorry

end line_through_point_with_equal_intercepts_l293_293511


namespace factors_are_divisors_l293_293410

theorem factors_are_divisors (a b c d : ℕ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) (h4 : d = 5) : 
  a ∣ 30 ∧ b ∣ 30 ∧ c ∣ 30 ∧ d ∣ 30 :=
by
  sorry

end factors_are_divisors_l293_293410


namespace minimum_value_l293_293566

noncomputable def f : ℝ → ℝ
| x => if h : 0 < x ∧ x ≤ 1 then x^2 - x else
         if h : 1 < x ∧ x ≤ 2 then -2 * (x - 1)^2 + 6 * (x - 1) - 5
         else 0 -- extend as appropriate outside given ranges

noncomputable def g (x : ℝ) : ℝ := x - 1

theorem minimum_value (x_1 x_2 : ℝ) (h1 : 1 < x_1 ∧ x_1 ≤ 2) : 
  (x_1 - x_2)^2 + (f x_1 - g x_2)^2 = 49 / 128 :=
sorry

end minimum_value_l293_293566


namespace find_m_n_l293_293978

variable {XYZ : Type}
variables (X Y Z D M P : XYZ)

def length (a b : XYZ) : ℝ := sorry
def angle_bisector (a b c : XYZ) : Prop := sorry
def midpoint (a b : XYZ) (m : XYZ) : Prop := sorry
def intersects (a b c : XYZ) : XYZ := sorry

-- Conditions
axiom XY_length : length X Y = 15
axiom XZ_length : length X Z = 9
axiom angle_bisector_X : angle_bisector X Y Z D
axiom midpoint_XD : midpoint X D M
axiom intersection_BM_XZ : intersects X Z M = P

-- To prove
theorem find_m_n : let m := 8 in let n := 5 in m + n = 13 :=
by
  sorry

end find_m_n_l293_293978


namespace first_quadrant_condition_l293_293247

def complex_point_in_first_quadrant (a : ℝ) : Prop :=
  let z := a + (a + 1) * complex.I in
  z.re > 0 ∧ z.im > 0

theorem first_quadrant_condition (a : ℝ) :
  complex_point_in_first_quadrant a ↔ a > 0 := by
  -- proof goes here
  sorry

end first_quadrant_condition_l293_293247


namespace geometric_sequence_product_l293_293972

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n + 1) = a n * r) 
  (h_log : log 2 (a 2 * a 98) = 4) : a 40 * a 60 = 16 :=
by
  sorry

end geometric_sequence_product_l293_293972


namespace orthocenters_on_circle_l293_293190

-- Defining the radius and centers of the circles
variables {r_b r_k d : ℝ} 
variables {O I : euclidean_space ℝ 2}

-- Condition on distance using Euler's formula
def eulers_formula (d r_k r_b : ℝ) : Prop :=
  d^2 = r_k^2 + 2 * r_k * r_b ∨ d^2 = r_k^2 - 2 * r_k * r_b

-- Statement of the problem
theorem orthocenters_on_circle 
  (hc : eulers_formula d r_k r_b) 
  (htri : ∃ t : triangle, t.inscribed_in (circle O r_b) ∧ t.circumscribed_around (circle I r_k)): 
  ∃ c : euclidean_space ℝ 2, ∃ R : ℝ, 
    (∀ (t : triangle), t.inscribed_in (circle O r_b) ∧ t.circumscribed_around (circle I r_k) → orthocenter t ∈ circle c R) 
    ∧ distance O c = distance I c 
    ∧ (R = 2 * r_b + r_k ∨ R = -2 * r_b + r_k) := 
by sorry

end orthocenters_on_circle_l293_293190


namespace flippy_numbers_divisible_by_25_l293_293174

theorem flippy_numbers_divisible_by_25 : 
  ∃ (n : ℕ), 
  (n = 16) ∧ 
  ∀ (num : ℕ), 
    (num.to_digits 10).length = 5 ∧ 
    (∀ i, i < 4 → 
      ((num.to_digits 10).nth i ≠ (num.to_digits 10).nth (i+1))) ∧
    (num % 25 = 0) → num ∈ set.univ :=
sorry

end flippy_numbers_divisible_by_25_l293_293174


namespace subproblem1_subproblem2_l293_293108

-- Sub-Problem (I)
theorem subproblem1 (x y r : ℝ) (M : ℝ × ℝ) (l : ℝ → ℝ) 
  (hC : ∀ x y, x^2 + (y - 4)^2 = r^2) 
  (hM : M = (-2, 0)) 
  (hr : r = 2) 
  (hl : ∀ x, l (-2) = 0) :
  l = ((λ x, -2) ∨ l = (λ x, (3 / 4) x + 3 / 4 * 2)) := sorry

-- Sub-Problem (II)
theorem subproblem2 (x y r : ℝ) (M : ℝ × ℝ) (l : ℝ → ℝ) 
  (hC : ∀ x y, x^2 + (y - 4)^2 = r^2) 
  (hM : M = (-2, 0)) 
  (hα : ∀ x, l = λ x, -x - 2) 
  (hAB : ∀ A B, dist A B = 2 * sqrt(2)) :
  r^2 = 20 := sorry

end subproblem1_subproblem2_l293_293108


namespace cos_inequality_sin_inequality_l293_293685

theorem cos_inequality (x : ℝ) (hx : x > 0) : 
  cos x > 1 - x^2 / 2 := 
sorry

theorem sin_inequality (x : ℝ) (hx : x > 0) : 
  sin x > x - x^3 / 6 := 
sorry

end cos_inequality_sin_inequality_l293_293685


namespace increasing_interval_cosine_l293_293717

theorem increasing_interval_cosine (k : ℤ) :
  ∃ (a b : ℝ), 
    a = 4 * (k * π) + (2 / 3) * π ∧ 
    b = 4 * (k * π) + (8 / 3) * π ∧ 
    ∀ x, a ≤ x ∧ x ≤ b → -cos((x / 2) - (π / 3)) is increasing :=
sorry

end increasing_interval_cosine_l293_293717


namespace find_a_l293_293885

variable (x y a : ℝ)
variable (h1 : x = 1)
variable (h2 : y = -2)
variable (h3 : 2 * x - a * y = 3)

theorem find_a : a = 1 / 2 := by
  have h : 2 * (1 : ℝ) - a * (-2 : ℝ) = 3 := by
    rw [h1, h2, h3]
  norm_num at h
  linarith
  sorry

end find_a_l293_293885


namespace solve_problem1_solve_problem2_l293_293610

noncomputable def problem1 (a b : ℝ) : Prop :=
  let c := 2
  let C := Real.pi / 3
  (√3 = 1 / 2 * a * b * Real.sin C) ∧ (4 = a^2 + b^2 - a * b * Real.cos C) →
  (a = 2 ∧ b = 2)

noncomputable def problem2 (a b : ℝ) : Prop :=
  let c := 2
  let C := Real.pi / 3
  (Real.sin C + Real.sin (Real.pi - (Real.arccos (b/(2*a)) - Real.arccos (2*a/b))) = Real.sin (2 * Real.arccos (a/b))) →
  (a = 2 ∧ b = 2) ∨ (a = 4 * Real.sqrt 3 / 3 ∧ b = 2 * Real.sqrt 3 / 3)

theorem solve_problem1 (a b : ℝ) : problem1 a b :=
sorry

theorem solve_problem2 (a b : ℝ) : problem2 a b :=
sorry

end solve_problem1_solve_problem2_l293_293610


namespace cosx_cos2x_not_rational_l293_293476

noncomputable def cosx_sqrt2_rational (x : ℝ) : Prop :=
  (∃ (a : ℚ), cos x + Real.sqrt 2 = a)

noncomputable def cos2x_sqrt2_rational (x : ℝ) : Prop :=
  (∃ (b : ℚ), cos (2*x) + Real.sqrt 2 = b)

theorem cosx_cos2x_not_rational : ∀ (x : ℝ), ¬(cosx_sqrt2_rational x ∧ cos2x_sqrt2_rational x) :=
by
  sorry

end cosx_cos2x_not_rational_l293_293476


namespace angle_bisector_l293_293605

def ellipse (a b : ℝ) : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | (9 * p.1^2 + 25 * p.2^2 = 225)}

def tangent_points (K H : ℝ × ℝ) : Prop := 
  K = (3, 12 / 5) ∧ H = (-4, 9 / 5)

def foci (G : ℝ × ℝ) : Prop := 
  G = (4, 0) ∨ G = (-4, 0)

theorem angle_bisector (A G K H : ℝ × ℝ)
  (hA : A = (-25 / 19, 15 / 4))
  (hG : foci G)
  (hEllipse : ellipse 5 3)
  (hTangentPts : tangent_points K H) :
  LineSegment A G bisects (Angle K G H) :=
sorry

end angle_bisector_l293_293605


namespace simplest_square_root_is_B_l293_293391

-- Definitions of the square root terms
def optionA (a : ℝ) : ℝ := Real.sqrt (16 * a)
def optionB (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)
def optionC (a b : ℝ) : ℝ := Real.sqrt (b / a)
def optionD : ℝ := Real.sqrt 45

-- Simplest square root among the options
theorem simplest_square_root_is_B (a b : ℝ) : optionB a b = Real.sqrt (a^2 + b^2) := by
  sorry

end simplest_square_root_is_B_l293_293391


namespace unique_parallel_line_through_point_l293_293773

open Locale.ModelQueue

variables (α β : Type) [Plane α ] [Plane β ] (m : Line α) (M : point β)

theorem unique_parallel_line_through_point :
  parallel α β → ∃! l : Line β, passes_through l M ∧ parallel l m :=
sorry

end unique_parallel_line_through_point_l293_293773


namespace seq_is_arithmetic_sum_inv_a_n_l293_293973

-- Define the sequence a_n
def a : ℕ → ℕ
| 0       := 0   -- a_0 is defined as 0 since sequences usually start from a_1
| 1       := 4   -- a_1 = 4 according to the problem condition
| (n+2) := (2:ℕ) * (n + 1) * (n + 2) + (n + 1) * a (n + 1)

-- Define the sequence a_n / n as a_n_n
def a_n_n (n : ℕ) : ℕ := a n / n

-- Define the sequence 1 / a_n as inv_a_n
noncomputable def inv_a_n (n : ℕ) : ℝ := 1 / (a n : ℝ)

-- Define the sum of the first n terms of inv_a_n as S_n
noncomputable def S_n (n : ℕ) : ℝ :=
∑ i in Finset.range n, inv_a_n (i + 1)

-- Prove that {a_n / n} is an arithmetic sequence
theorem seq_is_arithmetic : ∃ (d : ℕ), ∀ n, a_n_n (n + 1) - a_n_n n = d := sorry

-- Prove the sum of the first n terms of the sequence {1 / a_n} is n / (2n + 2)
theorem sum_inv_a_n (n : ℕ) : S_n n = (n : ℝ) / (2 * n + 2) := sorry

end seq_is_arithmetic_sum_inv_a_n_l293_293973


namespace arithmetic_sequence_a10_l293_293970

theorem arithmetic_sequence_a10 
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (h_seq : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h_S4 : S 4 = 10)
  (h_S9 : S 9 = 45) :
  a 10 = 10 :=
sorry

end arithmetic_sequence_a10_l293_293970


namespace market_value_correct_l293_293397

noncomputable def face_value : ℝ := 100
noncomputable def dividend_per_share : ℝ := 0.14 * face_value
noncomputable def yield : ℝ := 0.08

theorem market_value_correct :
  (dividend_per_share / yield) * 100 = 175 := by
  sorry

end market_value_correct_l293_293397


namespace altitude_of_triangle_l293_293447

theorem altitude_of_triangle
  (a b c : ℝ)
  (h₁ : a = 13)
  (h₂ : b = 15)
  (h₃ : c = 22)
  (h₄ : a + b > c)
  (h₅ : a + c > b)
  (h₆ : b + c > a) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h := (2 * A) / c
  h = (30 * Real.sqrt 10) / 11 :=
by
  sorry

end altitude_of_triangle_l293_293447


namespace sol_earnings_in_a_week_l293_293695

theorem sol_earnings_in_a_week :
  let first_day_sales := 10
  let increase_per_day := 4
  let number_of_days := 6
  let price_per_candy_bar := 0.10
  let sales := list.sum [first_day_sales, first_day_sales + increase_per_day, first_day_sales + 2 * increase_per_day, first_day_sales + 3 * increase_per_day, first_day_sales + 4 * increase_per_day, first_day_sales + 5 * increase_per_day]
  sales * price_per_candy_bar = 12.00 :=
by
  sorry

end sol_earnings_in_a_week_l293_293695


namespace NWF_USD_share_l293_293027

theorem NWF_USD_share :
  ∀ (total_NWF : ℝ) (amounts_in_curr : list ℝ) (prev_share_USD : ℝ), 
    total_NWF = 794.26 ∧ 
    amounts_in_curr = [34.72, 8.55, 600.30, 110.54, 0.31] ∧
    prev_share_USD = 49.17 -> 
    let USD_04 := (total_NWF - amounts_in_curr.sum) in
    let share_USD_04 := (USD_04 / total_NWF) * 100 in
    let change_share_USD := share_USD_04 - prev_share_USD in
    abs (share_USD_04 - 5.02) < 0.01 ∧ abs (change_share_USD + 44) < 0.01 :=
by
  intros total_NWF amounts_in_curr prev_share_USD h_cond
  rcases h_cond with ⟨h_total, h_amounts, h_prev⟩
  let USD_04 := 794.26 - (34.72 + 8.55 + 600.30 + 110.54 + 0.31)
  let share_USD_04 := (USD_04 / 794.26) * 100
  let change_share_USD := share_USD_04 - 49.17
  have : USD_04 = 39.84 := by sorry
  have : share_USD_04 ≈ 5.02 := by sorry
  have : change_share_USD ≈ -44 := by sorry
  exact ⟨this, this_1⟩

end NWF_USD_share_l293_293027


namespace diamond_evaluation_l293_293875

def diamond (x y z : ℝ) : ℝ := x / (y + z)

theorem diamond_evaluation :
  diamond (diamond 3 4 5) (diamond 4 5 3) (diamond 5 3 4) = 14 / 51 := by
  sorry

end diamond_evaluation_l293_293875


namespace cosine_double_angle_l293_293126

theorem cosine_double_angle : ∀ x : ℝ, cos x = 3/4 → cos (2*x) = 1/8 :=
by
  intros x h
  sorry

end cosine_double_angle_l293_293126


namespace arithmetic_expression_result_l293_293832

theorem arithmetic_expression_result :
  (10 - 9 * 8 / 4 + 7 - 6 * 5 + 3 - 2 * 1 : ℤ) = -30 :=
begin
  sorry
end

end arithmetic_expression_result_l293_293832


namespace solve_for_b_l293_293133

theorem solve_for_b (a b : ℚ) :
  (∀ x : ℂ, x^3 + a * x^2 + b * x - 12 = 0 → 
    x = 2 + sqrt 3 ∨ x = 2 - sqrt 3 ∨ x = -12) →
  b = -47 :=
by
  intros h
  -- Translation of all provided information to Lean
  sorry

end solve_for_b_l293_293133


namespace theorem_cross_product_l293_293170

noncomputable def point : Type :=
  ℝ × ℝ 

structure cross_product (M T1 T2 : point) :=
(dot_prod: ℝ)
(final_value: dot_prod ≥ 2*real.sqrt 2 - 3 ∧ dot_prod ≤ 0)

def line_passing_through (A B : point) : set point := sorry
def circle (O : point) (r : ℝ) : set point := sorry
def tangent_lines (M : point) (O : point) (r : ℝ) : set point := sorry

theorem theorem_cross_product 
(pointA point B : point)
(central_angle : ∠AOB = 90)
(P : point) (Q : point)
(P := (2, 0))
(Q := (0, 2))
(M := line_passing_through P pointA ∩ line_passing_through Q pointB)
(M ∉ circle (0,0) 1)
(T1 T2: point)
(T1 ∈ tangent_lines M (0, 0) 1)
(T2 ∈ tangent_lines M (0, 0) 1)
:
cross_product M T1 T2
:= 
begin
  sorry,
end

end theorem_cross_product_l293_293170


namespace calculate_weight_Cu2CO3_2_l293_293146

def molar_mass_Cu : ℝ := 63.55
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00

def molar_mass_Cu2CO3_2 : ℝ :=
  2 * molar_mass_Cu + 2 * molar_mass_C + 6 * molar_mass_O

def balanced_chemical_equation :=
  2 * 1 + 3 * 1 = 1 * 1 + 6 * 1 -- Simplified representation for Lean

noncomputable def weight_Cu2CO3_2 (moles_Cu2CO3_2 : ℝ) : ℝ :=
  moles_Cu2CO3_2 * molar_mass_Cu2CO3_2

theorem calculate_weight_Cu2CO3_2 :
  (moles_CuNO3_2 = 1.85) ∧ (moles_Na2CO3 = 3.21) →
  (balanced_chemical_equation) →
  weight_Cu2CO3_2 (1.85 / 2) = 228.586 :=
begin
  sorry
end

end calculate_weight_Cu2CO3_2_l293_293146


namespace hyperbola_eccentricity_l293_293546

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (F1 F2 : ℝ × ℝ) (h3 : F1 = (-sqrt (a^2 + b^2), 0)) (h4 : F2 = (sqrt (a^2 + b^2), 0)) 
    (A B : ℝ × ℝ) (h5 : A = (sqrt (a^2 + b^2) / 2, sqrt 3 / 2 * sqrt (a^2 + b^2))) 
    (h6 : B = (sqrt (a^2 + b^2) / 2, -sqrt 3 / 2 * sqrt (a^2 + b^2))) 
    (h7 : dist F1 A = dist F1 B) (h8 : dist F2 A = dist F2 B) (h9 : dist F1 A = dist A B) : 
    sqrt (3 : ℝ) + 1 = (sqrt (a^2 + b^2) / a) := 
by 
  sorry

end hyperbola_eccentricity_l293_293546


namespace intersection_of_M_and_N_l293_293165

def M : Set ℝ := { x | |x + 1| ≤ 1}

def N : Set ℝ := {-1, 0, 1}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0} :=
by
  sorry

end intersection_of_M_and_N_l293_293165


namespace vanaspati_percentage_l293_293203

theorem vanaspati_percentage (Q : ℝ) (h1 : 0.60 * Q > 0) (h2 : Q + 10 > 0) (h3 : Q = 10) :
    let total_ghee := Q + 10
    let pure_ghee := 0.60 * Q + 10
    let pure_ghee_fraction := pure_ghee / total_ghee
    pure_ghee_fraction = 0.80 → 
    let vanaspati_fraction := 1 - pure_ghee_fraction
    vanaspati_fraction * 100 = 40 :=
by
  intros
  sorry

end vanaspati_percentage_l293_293203


namespace heaviest_and_lightest_in_13_weighings_l293_293314

/-- Given ten coins of different weights and a balance scale.
    Prove that it is possible to identify the heaviest and the lightest coin
    within 13 weighings. -/
theorem heaviest_and_lightest_in_13_weighings
  (coins : Fin 10 → ℝ)
  (h_different: ∀ i j : Fin 10, i ≠ j → coins i ≠ coins j)
  : ∃ (heaviest lightest : Fin 10),
      (heaviest ≠ lightest) ∧
      (∀ i : Fin 10, coins i ≤ coins heaviest) ∧
      (∀ i : Fin 10, coins lightest ≤ coins i) :=
sorry

end heaviest_and_lightest_in_13_weighings_l293_293314


namespace convenient_triangle_angles_l293_293667

-- Definition of a convenient triangle
def is_convenient_triangle (T : Triangle ℝ) : Prop :=
  ∀ P : Point ℝ, ¬ P ∈ Plane T → is_triangle (segments_from_projection_to_plane T P)

-- Proposition to prove
theorem convenient_triangle_angles (T : Triangle ℝ) (h : is_convenient_triangle T) :
  T.angle A B C = 60 ∧ T.angle B C A = 60 ∧ T.angle C A B = 60 :=
sorry

end convenient_triangle_angles_l293_293667


namespace option_A_option_B_option_C_option_D_l293_293457

theorem option_A : (-(-1) : ℤ) ≠ -|(-1 : ℤ)| := by
  sorry

theorem option_B : ((-3)^2 : ℤ) ≠ -(3^2 : ℤ) := by
  sorry

theorem option_C : ((-4)^3 : ℤ) = -(4^3 : ℤ) := by
  sorry

theorem option_D : ((2^2 : ℚ)/3) ≠ ((2/3)^2 : ℚ) := by
  sorry

end option_A_option_B_option_C_option_D_l293_293457


namespace tangent_line_eq_l293_293284

theorem tangent_line_eq (f : ℝ → ℝ) (L : ℝ → ℝ) (df : ℝ → ℝ)
  (point_on_f : (1, 1) = (1 : ℝ, f 1))
  (f_eq : ∀ x, f x = x^2)
  (tangent_line : ∀ x1 y1 (m : ℝ), L = fun x => m * (x - x1) + y1)
  (df_eq : ∀ x, df x = 2 * x)
  (tangent_cond : ∀ x0, df x0 = (2 : ℝ) ∧ L x0 = f x0) :
  ∀ x : ℝ, L x = 2 * x - 1 :=
by 
sorry

end tangent_line_eq_l293_293284


namespace polar_equation_perpendicular_line_l293_293017

theorem polar_equation_perpendicular_line 
  (r θ : ℝ)
  (h₀ : r = 2)
  (h₁ : θ = Real.pi / 3) 
  (x y t : ℝ)
  (param_eq : (x = t) ∧ (y = t - 1)) :
  ∃ ρ : ℝ, ρ = (1 + Real.sqrt 3) / (Real.sin θ + Real.cos θ) :=
by 
  have P_cartesian : (x y).pair = (1, sqrt 3) := sorry
  have line_slope : (∃ k, k = 1) := sorry
  have perp_line : (∃ k', k' = -1) := sorry
  have perp_passing_P : (∃ eq, eq = x + y - sqrt 3 - 1) := sorry
  have polar_form : (∃ ρ, ρ = (1 + sqrt 3) / (sin θ + cos θ)) := sorry
  use ρ
  exact polar_form

end polar_equation_perpendicular_line_l293_293017


namespace PA_PB_geq_2r2_l293_293997

variable {r : ℝ}
variable {O₁ O₂ A B P : EuclideanSpace ℝ (Fin 2)}
variable {k₁ k₂ : set (EuclideanSpace ℝ (Fin 2))}

noncomputable def circle (O : EuclideanSpace ℝ (Fin 2)) (r : ℝ) := 
  {X | (X - O).norm = r}

def symmetric_about (A B O₁ O₂ : EuclideanSpace ℝ (Fin 2)) :=
  ∃ M : EuclideanSpace ℝ (Fin 2), (A + B) / 2 = M ∧ M ∈ line_through O₁ O₂

-- Conditions
axiom circle_k₁ : k₁ = circle O₁ r
axiom circle_k₂ : k₂ = circle O₂ r
axiom distance_O₁O₂ : (O₁ - O₂).norm = r
axiom symmetric_A_B : symmetric_about A B O₁ O₂
axiom P_on_k₂ : P ∈ k₂

theorem PA_PB_geq_2r2 : (P - A).norm^2 + (P - B).norm^2 ≥ 2 * r^2 := by
  sorry

end PA_PB_geq_2r2_l293_293997


namespace prime_divisibility_l293_293163

def sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 2 ∧ 
  a 1 = 1 ∧ 
  ∀ n, a (n + 1) = a n + a (n - 1)

theorem prime_divisibility (a : ℕ → ℕ) (k : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)] :
  sequence a →
  p ∣ (a (2 * k) - 2) →
  p ∣ (a (2 * k + 1) - 1) :=
by
  intro h_seq h_div
  sorry

end prime_divisibility_l293_293163


namespace range_of_a_l293_293600

noncomputable def has_common_tangent (f g : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, f x₁ = g x₂ ∧ f' x₁ = g' x₂

theorem range_of_a (a : ℝ) (h : 0 < a) :
  has_common_tangent (λ x : ℝ, x^2 + 1) (λ x : ℝ, a * Real.exp x + 1) →
  0 < a ∧ a ≤ 4 / Real.exp 2 :=
sorry

end range_of_a_l293_293600


namespace liam_book_pages_l293_293668

theorem liam_book_pages :
  let pages_first_three_days := 3 * 40 in
  let pages_next_three_days := 3 * 50 in
  let pages_first_session_seventh_day := 15 in
  let pages_second_session_seventh_day := 2 * 15 in
  pages_first_three_days + pages_next_three_days + 
  pages_first_session_seventh_day + pages_second_session_seventh_day = 315 :=
by
  let pages_first_three_days := 3 * 40
  let pages_next_three_days := 3 * 50
  let pages_first_session_seventh_day := 15
  let pages_second_session_seventh_day := 2 * 15
  calc
    pages_first_three_days + pages_next_three_days + 
    pages_first_session_seventh_day + pages_second_session_seventh_day
    = 120 + 150 + 45 : by rw [pages_first_three_days, pages_next_three_days, pages_first_session_seventh_day, pages_second_session_seventh_day]
    ... = 315 : by norm_num

end liam_book_pages_l293_293668


namespace simplify_expression_l293_293270

variable (x y : ℝ)

theorem simplify_expression : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 9 * y = 45 * x + 9 * y := 
by sorry

end simplify_expression_l293_293270


namespace calc_expression_l293_293831

theorem calc_expression :
  (3 * Real.sqrt 48 - 2 * Real.sqrt 12) / Real.sqrt 3 = 8 :=
sorry

end calc_expression_l293_293831


namespace measure_YZX_l293_293008

-- Define the triangle ABC with given angles
axiom triangle_ABC (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) : Prop :=
  angle_A = 50 ∧ angle_B = 70 ∧ angle_C = 60 ∧ angle_A + angle_B + angle_C = 180

-- Define the points X, Y, Z on the sides of triangle ABC
axiom points_on_sides (X Y Z : Type) (B C A B' C' A' : Type) : Prop :=
  X ∈ (B' ∩ C') ∧ Y ∈ (A' ∩ B) ∧ Z ∈ (A ∩ C')

-- Define the circle Gamma as incircle and circumcircle of triangles
axiom circle_incircle_circumcircle (Gamma : Type) (triangle_ABC triangle_XYZ : Type) : Prop :=
  incircle Gamma triangle_ABC ∧ circumcircle Gamma triangle_XYZ

-- Define the measure of YZX angle
def angle_YZX (angle_B angle_C : ℝ) : ℝ :=
  180 - (angle_B / 2 + angle_C / 2)

-- The main theorem to prove
theorem measure_YZX : ∀ (A B C X Y Z : Type) (Gamma : Type),
    triangle_ABC A B C 50 70 60 →
    points_on_sides X Y Z B C A B C A →
    circle_incircle_circumcircle Gamma (triangle_ABC A B C) (triangle_XYZ X Y Z) →
    angle_YZX 70 60 = 115 := 
by 
  intros A B C X Y Z Gamma hABC hXYZ hGamma
  sorry

end measure_YZX_l293_293008


namespace find_R4_l293_293233

theorem find_R4 {ABCD : Type*} (inscribed : inscribed_circle ABCD)
  (P : Type*) (intersection : P = diagonals_intersection ABCD)
  (R1 R2 R3 R4 : ℝ)
  (H1 : R1 = 31) (H2 : R2 = 24) (H3 : R3 = 12)
  (circumradius_condition : R1 + R3 = R2 + R4) :
  R4 = 19 :=
by
  sorry

end find_R4_l293_293233


namespace no_such_quadruple_exists_l293_293867

theorem no_such_quadruple_exists :
  ∀ (a b c d : ℝ), 
    (Matrix.det (Matrix.of ![![a, b], ![c, d]]) ≠ 0) →
    (Matrix.inv (Matrix.of ![![a, b], ![c, d]]) = Matrix.of ![![1/(a + 1), 1/(b + 1)], ![1/(c + 1), 1/(d + 1)]]) →
    false := 
begin
  assume a b c d h_det h_inv,
  sorry
end

end no_such_quadruple_exists_l293_293867


namespace four_digit_numbers_with_at_most_one_even_l293_293879

/-
  Question: Prove the number of four-digit numbers that can be formed using the digits 
  {1, 2, 3, 4, 5, 6, 7, 8, 9} without repetition, with at most one even digit, is 1080.
-/

theorem four_digit_numbers_with_at_most_one_even :
  (∑ (length : ℕ) in {120, 960}.to_finset, length) = 1080 :=
by sorry

end four_digit_numbers_with_at_most_one_even_l293_293879


namespace convert_kg_to_tons_convert_minutes_to_hours_convert_kg_to_grams_l293_293794

-- Define the conversion factors
def kg_to_tons (kg : ℝ) : ℝ := kg / 1000
def minutes_to_hours (min : ℝ) : ℝ := min / 60
def kg_to_grams (kg : ℝ) : ℝ := kg * 1000

-- Declare the three parts to be proven
theorem convert_kg_to_tons : kg_to_tons 56 = 0.056 := sorry

theorem convert_minutes_to_hours : minutes_to_hours 45 = 0.75 := sorry

theorem convert_kg_to_grams : kg_to_grams 0.3 = 300 := sorry

end convert_kg_to_tons_convert_minutes_to_hours_convert_kg_to_grams_l293_293794


namespace arc_length_l293_293529

theorem arc_length (C : ℝ) (theta : ℝ) (hC : C = 100) (htheta : theta = 30) :
  (theta / 360) * C = 25 / 3 :=
by sorry

end arc_length_l293_293529


namespace probability_diagonals_intersect_l293_293194

theorem probability_diagonals_intersect (n : ℕ) (h : n > 0) :
  let vertices := 2 * n + 1 in
  let total_diagonals := (vertices.choose 2) - vertices in
  let ways_to_choose_2_diagonals := (total_diagonals.choose 2) in
  let ways_to_choose_4_vertices := (vertices.choose 4) in
  let intersection_probability := ways_to_choose_4_vertices.to_rat / ways_to_choose_2_diagonals.to_rat in
  intersection_probability = n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) :=
by sorry

end probability_diagonals_intersect_l293_293194


namespace initial_ratio_of_milk_to_water_l293_293192

-- Define the capacity of the can, the amount of milk added, and the ratio when full.
def capacity : ℕ := 72
def additionalMilk : ℕ := 8
def fullRatioNumerator : ℕ := 2
def fullRatioDenominator : ℕ := 1

-- Define the initial amounts of milk and water in the can.
variables (M W : ℕ)

-- Define the conditions given in the problem.
def conditions : Prop :=
  M + W + additionalMilk = capacity ∧
  (M + additionalMilk) * fullRatioDenominator = fullRatioNumerator * W

-- Define the expected result, the initial ratio of milk to water in the can.
def expected_ratio : ℕ × ℕ :=
  (5, 3)

-- The theorem to prove the initial ratio of milk to water given the conditions.
theorem initial_ratio_of_milk_to_water (M W : ℕ) (h : conditions M W) :
  (M / Nat.gcd M W, W / Nat.gcd M W) = expected_ratio :=
sorry

end initial_ratio_of_milk_to_water_l293_293192


namespace find_m_n_sum_l293_293975

variables {X Y Z D M P : Type}
variables {XY XZ : ℝ}
variables {YZ YD DZ ZP PX : ℝ}
variables (m n : ℕ)

-- Conditions
axiom tri_XYZ : ∃ triangle : X → Y → Z, true
axiom len_XY : XY = 15
axiom len_XZ : XZ = 9
axiom angle_bisector_intersection : ∃ D, ∃ YD DZ : ℝ, ZP / PX = 8 / 5
axiom midpoint_M : ∃ M, ∃ XD XD' : ℝ, true

-- Problem (final goal)
theorem find_m_n_sum : m + n = 13 :=
sorry

end find_m_n_sum_l293_293975


namespace geometry_inequalities_match_figure_b_ii_l293_293800

/-- Given a circle centered at the origin with radius 2 units, prove that the inequalities
|x| + |y| ≤ 4 ≤ 2(x^2 + y^2) ≤ 8 * max(|x|, |y|) match the geometric figure B) II. -/
theorem geometry_inequalities_match_figure_b_ii :
  (∀ (x y : ℝ), |x| + |y| ≤ 4 ∧ 4 ≤ 2 * (x^2 + y^2) ∧ 2 * (x^2 + y^2) ≤ 8 * max (|x|) (|y|)) →
  (figure : Type) (is_figure_b_ii : figure → Prop), is_figure_b_ii figure :=
begin
  sorry
end

end geometry_inequalities_match_figure_b_ii_l293_293800


namespace range_of_a_l293_293141

noncomputable def f (x : ℝ) : ℝ := sorry -- f(x) is an odd and monotonically increasing function, to be defined later.

noncomputable def g (x a : ℝ) : ℝ :=
  f (x^2) + f (a - 2 * |x|)

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, g x1 a = 0 ∧ g x2 a = 0 ∧ g x3 a = 0 ∧ g x4 a = 0 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ↔
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l293_293141


namespace split_tip_evenly_l293_293226

noncomputable def total_cost (julie_order : ℝ) (letitia_order : ℝ) (anton_order : ℝ) : ℝ :=
  julie_order + letitia_order + anton_order

noncomputable def total_tip (meal_cost : ℝ) (tip_rate : ℝ) : ℝ :=
  tip_rate * meal_cost

noncomputable def tip_per_person (total_tip : ℝ) (num_people : ℝ) : ℝ :=
  total_tip / num_people

theorem split_tip_evenly :
  let julie_order := 10 in
  let letitia_order := 20 in
  let anton_order := 30 in
  let tip_rate := 0.20 in
  let num_people := 3 in
  tip_per_person (total_tip (total_cost julie_order letitia_order anton_order) tip_rate) num_people = 4 :=
by
  sorry

end split_tip_evenly_l293_293226


namespace cesaroSum_51terms_l293_293517

noncomputable def sequence : Type := List ℝ

def cesaroSum (B : sequence) : ℝ :=
  let fnSums := List.scanl (+) 0 B
  (fnSums.foldl (+) 0) / (B.length : ℝ)

variable (b : sequence) (h_blen : b.length = 50)
(h_bcesaro : cesaroSum b = 500)

theorem cesaroSum_51terms (h_bcesaro : cesaroSum b = 500) :
  cesaroSum (2 :: b) = 492 :=
sorry

end cesaroSum_51terms_l293_293517


namespace value_of_x_minus_y_l293_293949

theorem value_of_x_minus_y (x y a : ℝ) (h₁ : x + y > 0) (h₂ : a < 0) (h₃ : a * y > 0) : x - y > 0 :=
sorry

end value_of_x_minus_y_l293_293949


namespace slope_of_line_d_l293_293630

def C1 (x y : ℝ) := (x + 1) ^ 2 + y ^ 2 = 1
def C2 (x y : ℝ) := (x - 2) ^ 2 + y ^ 2 = 4
def A := (-3/2 : ℝ, Real.sqrt 3 / 2)
def B := (3 : ℝ, Real.sqrt 3)
def C := (1 : ℝ, Real.sqrt 3)
def D := (3 : ℝ, Real.sqrt 3)

theorem slope_of_line_d : 
  let d := (λ p q : ℝ × ℝ, (q.snd - p.snd) / (q.fst - p.fst))
  ∃ m : ℝ, 
    C1 A.1 A.2 ∧ 
    C2 B.1 B.2 ∧ 
    C2 C.1 C.2 ∧ 
    C1 D.1 D.2 ∧ 
    angle O B C = 60 ∧ 
    m = d A D ∧ 
    m = Real.sqrt 3 / 9 :=
begin
  sorry
end

end slope_of_line_d_l293_293630


namespace no_such_four_points_exist_l293_293265

theorem no_such_four_points_exist 
  (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB CD AC BD AD BC : ℝ)
  (h1 : AB = 8) (h2 : CD = 8) (h3 : AC = 10) (h4 : BD = 10) (h5 : AD = 13) (h6 : BC = 13) :
  ¬ (∃ A B C D : A, dist A B = AB ∧ dist C D = CD ∧ dist A C = AC ∧ dist B D = BD ∧ dist A D = AD ∧ dist B C = BC) :=
sorry

end no_such_four_points_exist_l293_293265


namespace sum_of_decimals_l293_293474

theorem sum_of_decimals :
  let a := 0.3
  let b := 0.08
  let c := 0.007
  a + b + c = 0.387 :=
by
  sorry

end sum_of_decimals_l293_293474


namespace factorization_correct_l293_293065

-- Define the expression
def expression (x : ℝ) : ℝ := x^2 + 2 * x

-- State the theorem to prove the factorized form is equal to the expression
theorem factorization_correct (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by {
  -- Lean will skip the proof because of sorry, ensuring the statement compiles correctly.
  sorry
}

end factorization_correct_l293_293065


namespace arithmetic_sequence_a9_l293_293624

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Assume arithmetic sequence: a(n) = a1 + (n-1)d
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) (n : ℕ) : ℤ := a 1 + (n - 1) * d

-- Given conditions
axiom condition1 : arithmetic_sequence a d 5 + arithmetic_sequence a d 7 = 16
axiom condition2 : arithmetic_sequence a d 3 = 1

-- Prove that a₉ = 15
theorem arithmetic_sequence_a9 : arithmetic_sequence a d 9 = 15 := by
  sorry

end arithmetic_sequence_a9_l293_293624


namespace Solution_l293_293530

noncomputable def problem_statement : Prop :=
∀ (A B C D : Point) (s₁ s₂ : Circle),
  CyclicQuadrilateral A B C D →
  (s₁.pass_through A ∧ s₁.pass_through B ∧ s₁.tangent AC) →
  (s₂.pass_through C ∧ s₂.pass_through D ∧ s₂.tangent AC) →
  ∃ (O : Point), Collinear A C O ∧ Collinear B D O ∧ Tangent s₁ s₂ O

theorem Solution : problem_statement :=
sorry

end Solution_l293_293530


namespace area_of_square_l293_293678

open Classical

-- Definitions and conditions of the problem
variable (AG DH GB HC : ℕ)
variable (Square_ABCD : Type)

axiom (hAGDH : AG = 10)
axiom (hGBHC : GB = 20)
axiom (hEquivalence : DH = AG ∧ HC = GB)
axiom (side_length_square : ∃ x : ℕ, x = AG + GB)

-- Let's state the problem
theorem area_of_square (hAGDH : AG = 10) (hGBHC : GB = 20) (hEquivalence : DH = AG ∧ HC = GB) (side_len : ∃ x : ℕ, x = AG + GB) : 
  ∃ area : ℕ, area = 900 :=
by 
  obtain ⟨x, hx⟩ := side_len
  have side_len_value : x = 30 := by rw [hAGDH, hGBHC]; rfl
  have area := x^2
  have area_value : area = 900 := by norm_num
  exact ⟨area, area_value⟩

end area_of_square_l293_293678


namespace angle_B_is_60_deg_l293_293609

-- Initial conditions
variable {A B C : ℝ} -- Define the angles A, B, C
variable {a b : ℝ} -- Define the sides a, b

-- Given condition in the problem
def given_condition : Prop := sqrt 3 * a = 2 * b * sin A

-- Statement to prove B = 60 degrees (π/3 radians) given the condition
theorem angle_B_is_60_deg (h : √3 * a = 2 * b * sin A) : B = π / 3 := 
sorry

end angle_B_is_60_deg_l293_293609


namespace rectangle_perimeter_l293_293345

open Real

def triangle_DEF_sides : ℝ × ℝ × ℝ := (9, 12, 15) -- sides of the triangle DEF

def rectangle_width : ℝ := 6 -- width of the rectangle

theorem rectangle_perimeter (a b c width : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : width = 6) :
  2 * (54 / width + width) = 30 :=
by
  sorry -- Proof is omitted as required

end rectangle_perimeter_l293_293345


namespace count_valid_integers_l293_293878

-- Define the expression that needs to be considered
def is_integer_expression (n : ℕ) : Prop :=
  (n ≥ 1 ∧ n ≤ 60) ∧ ((nat.factorial (n ^ 3 - 1)) % (nat.factorial (n ^ 2) ^ n) = 0)

-- Define the theorem to prove the number of valid integers n is 2
theorem count_valid_integers : (finset.filter is_integer_expression (finset.range 61)).card = 2 := sorry

end count_valid_integers_l293_293878


namespace binom_coefficient_divisible_by_at_least_two_primes_l293_293684

theorem binom_coefficient_divisible_by_at_least_two_primes
  (n k : ℕ) (h1 : 1 < k) (h2 : k < n - 1) :
  ∃ p q : ℕ, p ≠ q ∧ prime p ∧ prime q ∧ p ∣ (nat.choose n k) ∧ q ∣ (nat.choose n k) := 
sorry

end binom_coefficient_divisible_by_at_least_two_primes_l293_293684


namespace task_completion_prob_l293_293394

theorem task_completion_prob :
  let P_A1 := 2 / 3
  let P_A2 := 3 / 5
  let P_A3 := 4 / 7
  let P_not_A2 := 1 - P_A2
  P_A1 * P_not_A2 * P_A3 = 16 / 105 :=
by
  have P_A1 := (2 : ℚ) / 3
  have P_A2 := (3 : ℚ) / 5
  have P_A3 := (4 : ℚ) / 7
  have P_not_A2 := 1 - P_A2
  calc
    P_A1 * P_not_A2 * P_A3
      = (2 / 3) * ((1 : ℚ) - (3 / 5)) * (4 / 7) : by congr; norm_num
  ... = (2 / 3) * (2 / 5) * (4 / 7)             : by congr; ring
  ... = 16 / 105                                : by norm_num

end task_completion_prob_l293_293394


namespace area_PQRS_l293_293435

-- Define the lengths of the sides of the quadrilateral and the diagonal
def PQ := 4.0
def QR := 5.0
def RS := 3.0
def SP := 6.0
def PR := 7.0

-- Define the semi-perimeters of triangles PQR and PRS
def s₁ := (PQ + QR + PR) / 2
def s₂ := (PR + RS + SP) / 2

-- Define the areas of the triangles using Heron's formula
def area_PQR := Real.sqrt(s₁ * (s₁ - PQ) * (s₁ - QR) * (s₁ - PR))
def area_PRS := Real.sqrt(s₂ * (s₂ - PR) * (s₂ - RS) * (s₂ - SP))

-- Define the total area of the quadrilateral
def total_area := area_PQR + area_PRS

-- Theorem stating that the total area is 18.7 square miles
theorem area_PQRS : total_area = 18.7 :=
by
  sorry

end area_PQRS_l293_293435


namespace solve_for_x_l293_293025

theorem solve_for_x (x : ℝ) (h_pos : x > 0) 
  (h_eq : log (x + 1) / log 3 + log (x^2 + 1) / (log 3 / 2) + log (x + 1) / (-log 3) = 2) : 
  x = real.sqrt 2 := 
  sorry

end solve_for_x_l293_293025


namespace vector_magnitude_problem_l293_293136

variables (a b : ℝ^3) -- We'll use ℝ^3 for the vectors

-- Define the conditions
def is_unit_vector (v : ℝ^3) : Prop := ‖v‖ = 1
def angle_between (u v : ℝ^3) (θ : ℝ) : Prop := u.dot v = ‖u‖ * ‖v‖ * Real.cos θ

-- Main statement of the theorem
theorem vector_magnitude_problem (ha : is_unit_vector a) (hb : is_unit_vector b)
  (hangle : angle_between a b (Real.pi / 4)) :
  ‖a - (Real.sqrt 2) • b‖ = 1 :=
sorry

end vector_magnitude_problem_l293_293136


namespace sin_sum_identity_l293_293882

theorem sin_sum_identity 
  (α : ℝ) 
  (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) : 
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 := 
by 
  sorry

end sin_sum_identity_l293_293882


namespace find_k_l293_293084

theorem find_k (x y z k : ℝ) (h1 : 8 / (x + y + 1) = k / (x + z + 2)) (h2 : k / (x + z + 2) = 12 / (z - y + 3)) : k = 20 := by
  sorry

end find_k_l293_293084


namespace problem_a_problem_c_l293_293525

variable {a b : ℝ}

theorem problem_a (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 1) : ab ≤ 1 / 8 :=
by
  sorry

theorem problem_c (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 1) : 1 / a + 2 / b ≥ 9 :=
by
  sorry

end problem_a_problem_c_l293_293525


namespace new_water_percentage_l293_293430

theorem new_water_percentage (initial_volume : ℕ) (initial_percentage_water : ℝ) (added_water : ℕ) :
  initial_volume = 150 → initial_percentage_water = 0.10 → added_water = 30 →
  let initial_water := initial_percentage_water * initial_volume in
  let new_total_volume := initial_volume + added_water in
  let new_water_volume := initial_water + added_water in
  (new_water_volume / new_total_volume) * 100 = 25 :=
sorry

end new_water_percentage_l293_293430


namespace identify_heaviest_and_lightest_l293_293330

   def coin : Type := ℕ  -- let's represent coins as natural numbers for simplicity.

   def has_different_weights (coins : list coin) : Prop := 
     ∀ (c1 c2 : coin), c1 ∈ coins → c2 ∈ coins → c1 ≠ c2 → weight(c1) ≠ weight(c2)

   def weight : coin → ℝ := -- assume a function that gives the weight corresponding to a coin.
     sorry 

   theorem identify_heaviest_and_lightest (coins : list coin) 
     (h₁ : length coins = 10)
     (h₂ : has_different_weights coins) : 
     ∃ (heaviest lightest : coin), 
       (heaviest ∈ coins) ∧ (lightest ∈ coins) ∧
       (∀ c ∈ coins, weight c ≤ weight heaviest) ∧
       (∀ c ∈ coins, weight c ≥ weight lightest) :=
   by 
     sorry
   
end identify_heaviest_and_lightest_l293_293330


namespace problem_statement_l293_293339

noncomputable def perimeter_rectangle 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (area_triangle : ℝ := (1/2) * a * b) 
  (area_rectangle : ℝ := area_triangle) 
  (l : ℝ := area_rectangle / w) : ℝ :=
2 * (w + l)

theorem problem_statement 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h_a : a = 9) 
  (h_b : b = 12) 
  (h_c : c = 15) 
  (h_w : w = 6) : 
  perimeter_rectangle a b c w h1 = 30 :=
by 
  sorry

end problem_statement_l293_293339


namespace propositions_evaluation_l293_293543

theorem propositions_evaluation
  (p1 : ∀ x : ℝ, deriv (λ x, 2^x - 2^(-x)) x > 0)
  (p2 : ¬ (∀ x : ℝ, deriv (λ x, 2^x + 2^(-x)) x < 0)) :
  (p1 ∨ p2) ∧ ¬(p1 ∧ p2) ∧ ¬(¬p1 ∨ p2) ∧ (p1 ∨ ¬p2) :=
by
  sorry

end propositions_evaluation_l293_293543


namespace array_sums_distinct_l293_293099

/-- 
For any natural number n, if there exists an n x n array of entries 0, ±1 
such that all rows and columns have different sums, then n must be even. 
-/
theorem array_sums_distinct (n : ℕ) 
  (exists_array : ∃ A : Fin n → Fin n → ℤ, 
      (∀ (i j : Fin n), A i j = 0 ∨ A i j = 1 ∨ A i j = -1) ∧ 
      Function.Injective (λ i, ∑ j, A i j) ∧ 
      Function.Injective (λ j, ∑ i, A i j)) :
  n % 2 = 0 := 
sorry

end array_sums_distinct_l293_293099


namespace power_function_identity_l293_293903

def is_power_function (f : ℝ → ℝ) : Prop :=
∃ k c : ℝ, ∀ x : ℝ, f(x) = k * x^c

theorem power_function_identity 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : f = λ x : ℝ, a * x^(2 * a + 1) - b + 1)
  (h2 : is_power_function f) :
  a + b = 2 :=
sorry

end power_function_identity_l293_293903


namespace count_real_z5_of_z30_eq_1_l293_293739

theorem count_real_z5_of_z30_eq_1 : 
  ∃ zs : Finset ℂ, (zs.card = 30) ∧ (∀ z ∈ zs, z ^ 30 = 1) ∧ Finset.card ({z ∈ zs | ∃ r : ℝ, (z ^ 5 : ℂ) = r}) = 12 := 
sorry

end count_real_z5_of_z30_eq_1_l293_293739


namespace third_side_length_l293_293910

def is_odd (n : ℕ) := n % 2 = 1

theorem third_side_length (x : ℕ) (h1 : 2 + 5 > x) (h2 : x + 2 > 5) (h3 : is_odd x) : x = 5 :=
by
  sorry

end third_side_length_l293_293910


namespace measure_of_angle_YZX_l293_293015

noncomputable def problem_statement : Prop :=
  ∃ (A B C X Y Z : ℝ),
    X ∈ (segment B C) ∧
    Y ∈ (segment A B) ∧
    Z ∈ (segment A C) ∧
    angle A = 50 ∧
    angle B = 70 ∧
    angle C = 60 ∧
    let Γ := incircle ABC in
    let γ := circumcircle XYZ in
    incircle ABC = Γ ∧ 
    circumcircle XYZ = γ ∧ 
    ∠YZX = 115

theorem measure_of_angle_YZX (A B C X Y Z : ℝ) :
  X ∈ (segment B C) →
  Y ∈ (segment A B) →
  Z ∈ (segment A C) →
  angle A = 50 →
  angle B = 70 →
  angle C = 60 →
  let Γ := incircle ABC in
  let γ := circumcircle XYZ in
  incircle ABC = Γ →
  circumcircle XYZ = γ →
  ∠YZX = 115 :=
by
  sorry

end measure_of_angle_YZX_l293_293015


namespace max_stones_alex_can_place_l293_293819

theorem max_stones_alex_can_place : 
  let grid_side := 20,
      cell_side := 1,
      d := sqrt 5, 
      total_cells := grid_side * grid_side
  in ∃ K, (K <= total_cells / 4) ∧ ∀ alex stones, 
     ensure_distance (stones, d) → 
     ∃ stones', (|stones'| ≥ K ∧ (∀ s ∈ stones', within_grid(s, grid_side) ∧ not_overlap(stones'))) := 100 :=
sorry

end max_stones_alex_can_place_l293_293819


namespace num_coprime_with_15_l293_293024

theorem num_coprime_with_15 : finset.card (finset.filter (λ a : ℕ, a < 15 ∧ Nat.gcd a 15 = 1) (finset.range 15)) = 8 :=
by
  sorry

end num_coprime_with_15_l293_293024


namespace reflected_coordinates_l293_293711

-- Define the coordinates of point P
def point_P : ℝ × ℝ := (-2, -3)

-- Define the function for reflection across the origin
def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- State the theorem to prove
theorem reflected_coordinates :
  reflect_origin point_P = (2, 3) := by
  sorry

end reflected_coordinates_l293_293711


namespace twentieth_permutation_of_1389_is_9183_l293_293754

theorem twentieth_permutation_of_1389_is_9183 :
  let digits := [1, 3, 8, 9]
  let permutations := List.permutations digits
  let sorted_numbers := List.map (λ l, l.foldl (λ sum d, sum * 10 + d) 0) permutations |>.sorted (<)
  ∃ l ∈ permutations, l.foldl (λ sum d, sum * 10 + d) 0 = 9183 ∧ List.indexOf (l.foldl (λ sum d, sum * 10 + d) 0) sorted_numbers = 19 := 
by
  sorry

end twentieth_permutation_of_1389_is_9183_l293_293754


namespace total_letters_l293_293612

-- Define the conditions
def B := 16
def S := B + 30      -- S is defined from the condition S - B = 30
def D := B + 4       -- D is defined from the condition D - B = 4

-- The statement of the theorem
theorem total_letters : D + S - B = 50 := by
  calc
    D + S - B = (B + 4) + (B + 30) - B : by rw [D, S]
         ... = B + 4 + B + 30 - B  : by rw [D, S]
         ... = B + B + 4 + 30 - B  : by simp
         ... = 50                : by simp [B]

-- sorry as a placeholder for the proof
sorry

end total_letters_l293_293612


namespace complement_union_eq_complement_l293_293999

open Set

section ComplementUnion

variable (k : ℤ)

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def UniversalSet : Set ℤ := univ
def ComplementUnion : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem complement_union_eq_complement :
  UniversalSet \ (SetA ∪ SetB) = ComplementUnion :=
by
  sorry

end ComplementUnion

end complement_union_eq_complement_l293_293999


namespace a2_value_for_cubic_expansion_l293_293095

theorem a2_value_for_cubic_expansion (x a0 a1 a2 a3 : ℝ) : 
  (x ^ 3 = a0 + a1 * (x - 2) + a2 * (x - 2) ^ 2 + a3 * (x - 2) ^ 3) → a2 = 6 := by
  sorry

end a2_value_for_cubic_expansion_l293_293095


namespace range_of_a_l293_293145

noncomputable def is_tangent_at_positive_points (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ ∀ x, f'(x) = x^2 - x + a ∧ x^2 - x + a = 3

theorem range_of_a (a : ℝ) :
  is_tangent_at_positive_points (λ x, (1/3) * x^3 - (1/2) * x^2 + a * x + 3) a → 
  3 < a ∧ a < 13/4 :=
sorry

end range_of_a_l293_293145


namespace solve_for_x_l293_293700

theorem solve_for_x (x : ℝ) (h : x - 5.90 = 9.28) : x = 15.18 :=
by
  sorry

end solve_for_x_l293_293700


namespace wendys_sales_are_205_l293_293362

def price_of_apple : ℝ := 1.5
def price_of_orange : ℝ := 1.0
def apples_sold_morning : ℕ := 40
def oranges_sold_morning : ℕ := 30
def apples_sold_afternoon : ℕ := 50
def oranges_sold_afternoon : ℕ := 40

/-- Wendy's total sales for the day are $205 given the conditions about the prices of apples and oranges,
and the number of each sold in the morning and afternoon. -/
def wendys_total_sales : ℝ :=
  let total_apples_sold := apples_sold_morning + apples_sold_afternoon
  let total_oranges_sold := oranges_sold_morning + oranges_sold_afternoon
  let sales_from_apples := total_apples_sold * price_of_apple
  let sales_from_oranges := total_oranges_sold * price_of_orange
  sales_from_apples + sales_from_oranges

theorem wendys_sales_are_205 : wendys_total_sales = 205 := by
  sorry

end wendys_sales_are_205_l293_293362


namespace rectangle_perimeter_of_equal_area_l293_293346

theorem rectangle_perimeter_of_equal_area (a b c : ℕ) (area_triangle width length : ℕ) :
  a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2 ∧ (2 * area_triangle = a * b) ∧
  (width = 6) ∧ (area_triangle = width * length) -> 
  2 * (length + width) = 30 :=
by
  intros h,
  sorry

end rectangle_perimeter_of_equal_area_l293_293346


namespace sum_of_solutions_l293_293662

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x + (2 / 3) ^ x + (5 / 6) ^ x

theorem sum_of_solutions :
  let t := f x in
  (∀ x : ℝ, (t = 1 ∨ t = 2 ∨ t = 3) → x ∈ {0, 1, 3}) →
  (∑ x in {0, 1, 3}, x) = 4
:=
by
  sorry

end sum_of_solutions_l293_293662


namespace geometric_sequence_sum_of_first_five_l293_293568

theorem geometric_sequence_sum_of_first_five :
  (∃ (a : ℕ → ℝ) (r : ℝ),
    (∀ n, n > 0 → a n > 0) ∧
    a 2 = 2 ∧
    a 4 = 8 ∧
    r = 2 ∧
    a 1 = 1 ∧
    a 3 = a 1 * r^2 ∧
    a 5 = a 1 * r^4 ∧
    (a 1 + a 2 + a 3 + a 4 + a 5 = 31)
  ) :=
sorry

end geometric_sequence_sum_of_first_five_l293_293568


namespace intersection_area_of_two_circles_l293_293965

theorem intersection_area_of_two_circles :
  let a := 2 in
  let r := 1 in
  let R := (a * Real.sqrt 3) / 6 in
    (π * R^2 / 3 - R^2 * Real.sin (2 * π / 3) / 2)
  + (π * r^2 / 6 - r^2 * Real.sin (2 * π / 3) / 2)
  = (5 * π - 6 * Real.sqrt 3) / 18 := 
by
  sorry

end intersection_area_of_two_circles_l293_293965


namespace rectangle_perimeter_is_30_l293_293350

noncomputable def triangle_DEF_sides := (9 : ℕ, 12 : ℕ, 15 : ℕ)
noncomputable def rectangle_width := (6 : ℕ)

theorem rectangle_perimeter_is_30 :
  let area_triangle_DEF := (triangle_DEF_sides.1 * triangle_DEF_sides.2) / 2
  let rectangle_length := area_triangle_DEF / rectangle_width
  let rectangle_perimeter := 2 * (rectangle_width + rectangle_length)
  rectangle_perimeter = 30 := by
  sorry

end rectangle_perimeter_is_30_l293_293350


namespace ideal_number_502_l293_293891

noncomputable def sequence {α : Type*} [add_comm_group α] (a : ℕ → α) : ℕ → α
| 0 := 0
| (n + 1) := a (n + 1) + sequence a n

noncomputable def sum_first_n {α : Type*} [add_comm_group α] (a : ℕ → α) (n : ℕ) : α :=
∑ i in finset.range (n + 1), (sequence a i)

noncomputable def ideal_number {α : Type*} [field α] (a : ℕ → α) (n : ℕ) : α :=
(sum_first_n a n) / (n + 1)

variables (a : ℕ → ℝ) (n : ℕ)

theorem ideal_number_502 :
  (ideal_number a 501 = 2012) →
  (ideal_number (λ i, if i = 0 then 2 else a (i - 1)) 502 = 2010) :=
by 
  sorry

end ideal_number_502_l293_293891


namespace calculate_grand_total_profit_l293_293334

-- Definitions based on conditions
def cost_per_type_A : ℕ := 8 * 10
def sell_price_type_A : ℕ := 125
def cost_per_type_B : ℕ := 12 * 18
def sell_price_type_B : ℕ := 280
def cost_per_type_C : ℕ := 15 * 12
def sell_price_type_C : ℕ := 350

def num_sold_type_A : ℕ := 45
def num_sold_type_B : ℕ := 35
def num_sold_type_C : ℕ := 25

-- Definition of profit calculations
def profit_per_type_A : ℕ := sell_price_type_A - cost_per_type_A
def profit_per_type_B : ℕ := sell_price_type_B - cost_per_type_B
def profit_per_type_C : ℕ := sell_price_type_C - cost_per_type_C

def total_profit_type_A : ℕ := num_sold_type_A * profit_per_type_A
def total_profit_type_B : ℕ := num_sold_type_B * profit_per_type_B
def total_profit_type_C : ℕ := num_sold_type_C * profit_per_type_C

def grand_total_profit : ℕ := total_profit_type_A + total_profit_type_B + total_profit_type_C

-- Statement to be proved
theorem calculate_grand_total_profit : grand_total_profit = 8515 := by
  sorry

end calculate_grand_total_profit_l293_293334


namespace time_to_cross_bridge_l293_293444

-- Definitions of conditions
def train_length : ℝ := 135
def train_speed_kmph : ℝ := 45
def bridge_length : ℝ := 240

-- Conversion of speed to m/s
def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

-- Total distance to be covered
def total_distance : ℝ := train_length + bridge_length

-- Theorem: Time to cross the bridge is 30 seconds
theorem time_to_cross_bridge : 
  total_distance / train_speed_mps = 30 :=
by
  sorry

end time_to_cross_bridge_l293_293444


namespace exists_consecutive_composites_l293_293100

noncomputable theory

def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ p * q = n

theorem exists_consecutive_composites (a t d r : ℕ) 
    (ha : is_composite a) 
    (ht : is_composite t) 
    (hd : is_composite d) 
    (hr : is_composite r) :
  ∃ x : ℕ, ∀ j : ℕ, 1 ≤ j ∧ j ≤ r → is_composite (a * t^(x + j) + d) :=
by sorry

end exists_consecutive_composites_l293_293100


namespace cross_area_correct_l293_293767

noncomputable def cross_area : ℕ :=
  let i := 4 in  -- Number of interior lattice points
  let b := 6 in  -- Number of boundary lattice points
  i + (b / 2) - 1

theorem cross_area_correct : cross_area = 6 := by
  unfold cross_area
  norm_num
  sorry

end cross_area_correct_l293_293767


namespace det_of_cross_product_matrix_l293_293653

noncomputable def matrix_det (a b c : Vector ℝ 3) : ℝ :=
  a ⬝ (b ×ₗ c)

noncomputable def new_matrix_det (a b c : Vector ℝ 3) : ℝ :=
  (b ×ₗ c) ⬝ ((c ×ₗ a) ×ₗ (a ×ₗ b))

theorem det_of_cross_product_matrix (a b c : Vector ℝ 3) :
  let D := matrix_det a b c
  in new_matrix_det a b c = D ^ 2 :=
by
  sorry

end det_of_cross_product_matrix_l293_293653


namespace find_x_from_eq_l293_293589

theorem find_x_from_eq :
  ∃ x : ℝ, 400 * x = 28000 * 100^1 ∧ x = 7000 := 
by
  use 7000
  split
  . calc
    400 * 7000 = 2800000       : by norm_num
            ... = 28000 * 100^1 : by norm_num
  . sorry

end find_x_from_eq_l293_293589


namespace book_pages_count_l293_293183

theorem book_pages_count (n : ℕ) (h : ∑ i in Finset.range (n + 1), (n.digitsCount 1) = 171) : n = 318 := 
sorry

end book_pages_count_l293_293183


namespace fat_content_proof_l293_293416

def fat_percentage_non_whole_milk : ℝ := 2
def fat_percentage_whole_milk : ℝ := 10 / 3

theorem fat_content_proof :
  fat_percentage_non_whole_milk = 0.6 * fat_percentage_whole_milk :=
by
  -- this is where the proof would go
  sorry

end fat_content_proof_l293_293416


namespace value_of_a0_plus_a8_l293_293883

/-- Theorem stating the value of a0 + a8 from the given polynomial equation -/
theorem value_of_a0_plus_a8 (a_0 a_8 : ℤ) :
  (∀ x : ℤ, (1 + x) ^ 10 = a_0 + a_1 * (1 - x) + a_2 * (1 - x) ^ 2 + 
              a_3 * (1 - x) ^ 3 + a_4 * (1 - x) ^ 4 + a_5 * (1 - x) ^ 5 +
              a_6 * (1 - x) ^ 6 + a_7 * (1 - x) ^ 7 + a_8 * (1 - x) ^ 8 + 
              a_9 * (1 - x) ^ 9 + a_10 * (1 - x) ^ 10) →
  a_0 + a_8 = 1204 :=
by
  sorry

end value_of_a0_plus_a8_l293_293883


namespace pattern_count_l293_293575

-- Define the problem conditions
def grid := fin 4 × fin 4 -- 4x4 grid

def is_pattern_equivalent {A B : set grid} : Prop :=
  ∃ f : grid → grid, (bihojective f) ∧ (A = f '' B) ∧ (f '' B ≠ ∅)  -- equivalence by flips and/or turns

def is_valid_pattern (P : set grid) : Prop :=
  P.card = 3 -- pattern must have exactly 3 shaded squares

-- Statement of the problem
theorem pattern_count : 
  ∃ S : set (set grid), (∀ P ∈ S, is_valid_pattern P) 
  ∧ (card S = 10) 
  ∧ (∀ P1 P2 ∈ S, is_pattern_equivalent P1 P2 → P1 = P2) :=
sorry

end pattern_count_l293_293575


namespace range_of_k_l293_293152

noncomputable def f (x : ℝ) : ℝ := 
  Real.log ((1 + x) / (1 - x)) + x^3

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < b ∧ a < y ∧ y < b ∧ x < y → f(x) < f(y)

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

theorem range_of_k (k : ℝ) :
  is_increasing f (-1) 1 ∧ is_odd f ∧ 
  ∃ x1 x2, x1 ≠ x2 ∧ f(x1) + f(k - x1^2) = 0 ∧ f(x2) + f(k - x2^2) = 0 →
  - 1 / 4 < k ∧ k < 0 :=
sorry

end range_of_k_l293_293152


namespace _l293_293608

noncomputable theorem triangle_problem (a b c A B C : ℝ) 
  (h1 : C = π / 3) 
  (h2 : let CA := 3 * sqrt 6 in 
        let CB := 3 * sqrt 6 in 
        (CA • (CA - CB)) = -27) 
  (h3 : let AB := sqrt (CA^2 + CB^2 - 2 * 27) in 
        AB >= 3 * sqrt 6) : 
  (C = π / 3) ∧ (minAB ≥ 3 * sqrt 6) :=
by
  sorry

end _l293_293608


namespace tan_alpha_minus_pi_div_4_sin2alpha_cosalpha_expr_value_l293_293132

noncomputable def alpha : ℝ := sorry  -- since α needs to be within the right interval, we should generally consider this carefully

def sin_alpha (α : ℝ) : Prop := α ∈ (π/2, π) ∧ sin α = 3/5

theorem tan_alpha_minus_pi_div_4 (α : ℝ) (h : sin_alpha α) : 
  ∃ α_val, α = α_val ∧ tan (α - π/4) = -7 :=
sorry

theorem sin2alpha_cosalpha_expr_value (α : ℝ) (h : sin_alpha α) : 
  ∃ α_val, α = α_val ∧ (sin (2 * α) - cos α) / (1 + cos (2 * α)) = -1/8 :=
sorry

end tan_alpha_minus_pi_div_4_sin2alpha_cosalpha_expr_value_l293_293132


namespace tangents_to_circle_l293_293847

-- Definition of the circle centered at (1, 1) with radius 1
def is_on_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Definition of the lines passing through the point (2, 3)
def line1 : ℝ → ℝ → Prop := λ x y, 3 * x - 4 * y + 6 = 0
def line2 : ℝ → ℝ → Prop := λ x y, x = 2

-- Problem statement: Prove that the tangents to the circle at (1, 1) passing through (2, 3) are line1 and line2
theorem tangents_to_circle : 
  (∀ x y, is_on_circle x y → (line1 x y ∨ line2 x y)) ∧
  line1 2 3 ∧ line2 2 3 :=
by 
  sorry

end tangents_to_circle_l293_293847


namespace num_solutions_correct_l293_293864

noncomputable def num_solutions : ℕ :=
  let factorial (n : ℕ) := (nat.factorial n)
  let z_is_on_unit_circle (z : ℂ) : Prop := abs z = 1
  let expression_is_real (z : ℂ) : Prop := z^(factorial 7) - z^(factorial 6) ∈ ℝ
  
  nat.card { z : ℂ // z_is_on_unit_circle z ∧ expression_is_real z }

theorem num_solutions_correct : num_solutions = 44 := by
  sorry

end num_solutions_correct_l293_293864


namespace distance_from_origin_to_line_l293_293155

structure Point where
  x : ℝ
  y : ℝ

def on_hyperbola (M : Point) : Prop :=
  2 * M.x^2 - M.y^2 = 1

def on_ellipse (N : Point) : Prop :=
  4 * N.x^2 + N.y^2 = 1

def orthogonal (M N : Point) : Prop :=
  M.x * N.x + M.y * N.y = 0

theorem distance_from_origin_to_line (M N : Point) (hM : on_hyperbola M) (hN : on_ellipse N) (h_perp : orthogonal M N) : 
  distance_to_line (Point.mk 0 0) M N = Math.sqrt(3) / 3 := 
sorry

end distance_from_origin_to_line_l293_293155


namespace principal_amount_l293_293186

-- Define the conditions and required result
theorem principal_amount
  (P R T : ℝ)
  (hR : R = 0.5)
  (h_diff : (P * R * (T + 4) / 100) - (P * R * T / 100) = 40) :
  P = 2000 :=
  sorry

end principal_amount_l293_293186


namespace gcd_1151_3079_l293_293074

def a : ℕ := 1151
def b : ℕ := 3079

theorem gcd_1151_3079 : gcd a b = 1 := by
  sorry

end gcd_1151_3079_l293_293074


namespace lottery_win_amount_l293_293994

theorem lottery_win_amount (total_tax : ℝ) (federal_tax_rate : ℝ) (local_tax_rate : ℝ) (tax_paid : ℝ) :
  total_tax = tax_paid →
  federal_tax_rate = 0.25 →
  local_tax_rate = 0.15 →
  tax_paid = 18000 →
  ∃ x : ℝ, x = 49655 :=
by
  intros h1 h2 h3 h4
  use (tax_paid / (federal_tax_rate + local_tax_rate * (1 - federal_tax_rate))), by
    norm_num at h1 h2 h3 h4
    sorry

end lottery_win_amount_l293_293994


namespace sequence_is_aperiodic_l293_293812

noncomputable def sequence_a (a : ℕ → ℕ) : Prop :=
∀ k n : ℕ, k < 2^n → a k ≠ a (k + 2^n)

theorem sequence_is_aperiodic (a : ℕ → ℕ) (h_a : sequence_a a) : ¬(∃ p : ℕ, ∀ n k : ℕ, a k = a (k + n * p)) :=
sorry

end sequence_is_aperiodic_l293_293812


namespace hawks_points_l293_293611

theorem hawks_points (E H : ℕ) (h₁ : E + H = 82) (h₂ : E = H + 18) (h₃ : H ≥ 9) : H = 32 :=
sorry

end hawks_points_l293_293611


namespace find_a_l293_293584

theorem find_a (a : ℝ) (h : 3 ∈ {a + 3, 2 * a + 1, a^2 + a + 1}) : a = -2 :=
by
  sorry

end find_a_l293_293584


namespace train_crossing_time_l293_293631

-- Define given conditions
def train_length : ℝ := 200  -- Length of the train in meters
def train_speed_kmh : ℝ := 180  -- Speed of the train in km/hr
def conversion_factor : ℝ := 1000 / 3600  -- Conversion factor from km/hr to m/s

-- Define the problem statement
theorem train_crossing_time :
  let train_speed_ms := train_speed_kmh * conversion_factor in
  let crossing_time := train_length / train_speed_ms in
  crossing_time = 4 := by
  -- Proof goes here
  sorry

end train_crossing_time_l293_293631


namespace length_of_AB_l293_293113

noncomputable def line_eqn (t : ℝ) : ℝ × ℝ :=
(2 + (Real.sqrt 2 / 2) * t, 1 + (Real.sqrt 2 / 2) * t)

noncomputable def circle_eqn (x y : ℝ) : Prop :=
x^2 + y^2 = 4

theorem length_of_AB :
  ∀ t1 t2 : ℝ, 
  line_eqn t1 = (x1, y1) → circle_eqn x1 y1 →
  line_eqn t2 = (x2, y2) → circle_eqn x2 y2 →
  t1 + t2 = -3 * Real.sqrt 2 ∧ t1 * t2 = 1 →
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = Real.sqrt 14 :=
begin
  sorry
end

end length_of_AB_l293_293113


namespace find_a_square_of_binomial_l293_293068

theorem find_a_square_of_binomial :
  ∃ a : ℝ, (∀ x : ℝ, ax^2 + 12 * x + 9 = (2 * x + 3)^2 ↔ a = 4) :=
begin
  sorry
end

end find_a_square_of_binomial_l293_293068


namespace even_function_periodic_odd_function_period_generalized_period_l293_293660

-- Problem 1
theorem even_function_periodic (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (-x) = f x) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 2 * a) = f x :=
by sorry

-- Problem 2
theorem odd_function_period (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 4 * a) = f x :=
by sorry

-- Problem 3
theorem generalized_period (f : ℝ → ℝ) (a m n : ℝ) (h₁ : ∀ x : ℝ, 2 * n - f x = f (2 * m - x)) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 4 * (m - a)) = f x :=
by sorry

end even_function_periodic_odd_function_period_generalized_period_l293_293660


namespace purely_imaginary_implies_a_neg2_l293_293597

variable (a : ℝ)
def z := a^2 - 4 + (a - 2) * complex.I

theorem purely_imaginary_implies_a_neg2
  (hz : z a = (0 : ℝ) + complex.I * (a - 2))
  (hi_ne_zero : a - 2 ≠ 0):
  a = -2 :=
sorry

end purely_imaginary_implies_a_neg2_l293_293597


namespace convert_to_spherical_l293_293482

-- Definitions of the conditions for the spherical coordinates
def rectangular_coords := (0 : ℝ, 3 : ℝ, -3 * Real.sqrt 3 : ℝ)
def spherical_coords := (6 : ℝ, Real.pi / 2, 5 * Real.pi / 6)

-- The spherical coordinate conditions
def rho (x y z : ℝ) := Real.sqrt (x^2 + y^2 + z^2)
def phi (z rho : ℝ) := Real.arccos (z / rho)
def theta (x y : ℝ) := if (x = 0) then Real.pi / 2 else Real.atan2 y x

-- Main theorem statement
theorem convert_to_spherical :
  ∃ (ρ θ φ : ℝ), 
    ρ > 0 ∧ 
    0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
    0 ≤ φ ∧ φ ≤ Real.pi ∧ 
    (ρ, θ, φ) = spherical_coords :=
by
  sorry

end convert_to_spherical_l293_293482


namespace stamp_arrangements_15_cents_l293_293846

-- Definition of the stamp values
def stamp_values : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Condition for arranging 15 cents worth of stamps
def is_valid_arrangement (arrangement : List ℕ) : Prop :=
  arrangement.sum = 15

-- Unique arrangements modulo equivalent permutations (rotations, etc.)
def count_unique_arrangements (arrs : List (List ℕ)) : ℕ :=
  arrs.filter is_valid_arrangement |>.length

-- The main theorem stating the number of such distinct arrangements
theorem stamp_arrangements_15_cents :
  count_unique_arrangements (stamp_values.powerset) = 48 :=
sorry

end stamp_arrangements_15_cents_l293_293846


namespace odd_function_a_eq_minus_1_l293_293598

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * (x + a) / x

theorem odd_function_a_eq_minus_1 (a : ℝ) :
  (∀ x : ℝ, f (-x) a = -f x a) → a = -1 :=
by
  intros h
  sorry

end odd_function_a_eq_minus_1_l293_293598


namespace differential_equation_solution_l293_293979

theorem differential_equation_solution (x y : ℝ) (C : ℝ) :
  (∀ dx dy, 2 * x * y * dx + x^2 * dy = 0) → x^2 * y = C :=
sorry

end differential_equation_solution_l293_293979


namespace katie_sold_cupcakes_l293_293520

theorem katie_sold_cupcakes :
  ∀ (initial additional left sold : ℕ),
  initial = 26 → additional = 20 → left = 26 →
  (initial + additional - left = sold) → sold = 20 :=
by
  intros initial additional left sold h_initial h_additional h_left h_equation
  rw [h_initial, h_additional, h_left] at h_equation
  exact h_equation

end katie_sold_cupcakes_l293_293520


namespace car_travel_inequality_l293_293414

variable (x : ℕ)

theorem car_travel_inequality (hx : 8 * (x + 19) > 2200) : 8 * (x + 19) > 2200 :=
by
  sorry

end car_travel_inequality_l293_293414


namespace radius_of_circle_eqn_zero_l293_293026

def circle_eqn (x y : ℝ) := x^2 + 8*x + y^2 - 4*y + 20 = 0

theorem radius_of_circle_eqn_zero :
  ∀ x y : ℝ, circle_eqn x y → ∃ r : ℝ, r = 0 :=
by
  intros x y h
  -- Sorry to skip the proof as per instructions
  sorry

end radius_of_circle_eqn_zero_l293_293026


namespace line_through_point_with_equal_intercepts_l293_293510

-- Definition of the conditions
def point := (1 : ℝ, 2 : ℝ)
def eq_intercepts (line : ℝ → ℝ) := ∃ a b : ℝ, a = b ∧ (∀ x, line x = b - x * (b/a))

-- The proof statement
theorem line_through_point_with_equal_intercepts (line : ℝ → ℝ) : 
  (line 1 = 2 ∧ eq_intercepts line) → (line = (λ x, 2 * x) ∨ line = (λ x, 3 - x)) :=
by
  sorry

end line_through_point_with_equal_intercepts_l293_293510


namespace number_of_solutions_l293_293860

-- Define the condition for z being a complex number with |z| = 1
def is_on_unit_circle (z : ℂ) : Prop := complex.abs z = 1

-- Define the condition for z^{7!} - z^{6!} being a real number
def is_real_output (z : ℂ) : Prop := ∃ (a : ℝ), z ^ 5040 - z ^ 720 = a

-- The main theorem we need to prove
theorem number_of_solutions : 
  (finset.univ.filter (λ z : ℂ, is_on_unit_circle z ∧ is_real_output z)).card = 7200 := 
sorry

end number_of_solutions_l293_293860


namespace painter_collects_186_dollars_l293_293442

def arith_seq (a d n : ℕ) : ℕ := a + (n - 1) * d

def cost_one_digit (n : ℕ) : ℕ :=
  if n < 10 then 1 else 0

def cost_two_digits_below_50 (n : ℕ) : ℕ :=
  if 10 ≤ n ∧ n < 50 then 2 else 0

def cost_two_digits_50_and_above (n : ℕ) : ℕ :=
  if 50 ≤ n ∧ n < 100 then 4 else 0

def cost_three_digits (n : ℕ) : ℕ :=
  if n ≥ 100 then 6 else 0

def cost_per_house (n : ℕ) : ℕ :=
  cost_one_digit n +
  cost_two_digits_below_50 n +
  cost_two_digits_50_and_above n +
  cost_three_digits n

def total_cost : ℕ :=
  let south_addresses := list.range 25.map (λ n => arith_seq 5 8 (n + 1))
  let north_addresses := list.range 25.map (λ n => arith_seq 3 8 (n + 1))
  (south_addresses ++ north_addresses).sum cost_per_house

theorem painter_collects_186_dollars :
  total_cost = 186 :=
by
  sorry

end painter_collects_186_dollars_l293_293442


namespace caffeine_per_energy_drink_l293_293290

variable (amount_of_caffeine_per_drink : ℕ)

def maximum_safe_caffeine_per_day := 500
def drinks_per_day := 4
def additional_safe_amount := 20

theorem caffeine_per_energy_drink :
  4 * amount_of_caffeine_per_drink + additional_safe_amount = maximum_safe_caffeine_per_day →
  amount_of_caffeine_per_drink = 120 :=
by
  sorry

end caffeine_per_energy_drink_l293_293290


namespace unique_singleton_value_l293_293094

theorem unique_singleton_value (P : ℤ → ℤ) (infinite_set : set ℤ)
  (h1 : ∃ inf_set : set ℤ, set.infinite inf_set ∧ ∀ a ∈ inf_set, ∃ x y : ℤ, x ≠ y ∧ P x = a ∧ P y = a) :
  ∃ a : ℤ, ∀ x y : ℤ, x ≠ y → P x = a → P y ≠ a :=
by
  sorry

end unique_singleton_value_l293_293094


namespace rhombus_perimeter_is_correct_l293_293278

-- Define the rhombus with diagonals 12 and 16 inches
structure Rhombus (d1 d2 : ℝ) :=
  (d1_pos : 0 < d1)
  (d2_pos : 0 < d2)

-- Given a rhombus with diagonals of specific lengths
def given_rhombus := Rhombus.mk 12 16 (by norm_num) (by norm_num)

-- The function to calculate the side length of a rhombus given the diagonals
def rhombus_side_length (d1 d2 : ℝ) : ℝ :=
  real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)

-- The function to calculate the perimeter of a rhombus
def rhombus_perimeter (r : Rhombus) : ℝ :=
  4 * (rhombus_side_length r.d1 r.d2)

-- The theorem to prove
theorem rhombus_perimeter_is_correct : rhombus_perimeter given_rhombus = 40 := 
by {
  -- Details of the proof
  sorry
}

end rhombus_perimeter_is_correct_l293_293278


namespace largest_real_number_l293_293235

theorem largest_real_number (n : ℕ) (h : n ≥ 3) (x : ℕ → ℝ)
  (h_pos: ∀ i < n, 0 < x i) :
  ∃ y : ℕ → ℝ, (∀ i < n, y i = x i) →
  (∀ i, y (i + n) = y i) →
  (∑ i in finset.range n, y i ^ 2 / (y (i + 1) ^ 2 - y (i + 1) * y (i + 2) + y (i + 2) ^ 2)) ≥ n - 1 :=
sorry

end largest_real_number_l293_293235


namespace rachel_total_score_l293_293688

theorem rachel_total_score (points_per_treasure : ℕ) (level1_treasures level2_treasures : ℕ) : 
  (points_per_treasure = 9) → 
  (level1_treasures = 5) → 
  (level2_treasures = 2) → 
  points_per_treasure * level1_treasures + points_per_treasure * level2_treasures = 63 :=
by {
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end rachel_total_score_l293_293688


namespace driving_time_l293_293760

-- Conditions from problem
variable (distance1 : ℕ) (time1 : ℕ) (distance2 : ℕ)
variable (same_speed : distance1 / time1 = distance2 / (5 : ℕ))

-- Statement to prove
theorem driving_time (h1 : distance1 = 120) (h2 : time1 = 3) (h3 : distance2 = 200)
  : distance2 / (40 : ℕ) = (5 : ℕ) := by
  sorry

end driving_time_l293_293760


namespace median_group_two_l293_293172

theorem median_group_two (freq_one freq_two freq_three freq_four total_students : ℕ) 
  (h_freq_one : freq_one = 10) (h_freq_two : freq_two = 20) (h_freq_three : freq_three = 12) (h_freq_four : freq_four = 8) 
  (h_total : total_students = 50) :
  1 ≤ median_weekly_labor_time freq_one freq_two freq_three freq_four < 2 :=
by
  sorry

end median_group_two_l293_293172


namespace range_m_monotonic_increasing_l293_293149

def f (m x : ℝ) := x^3 - 3 * m * x^2 + 9 * m * x + 1

theorem range_m_monotonic_increasing (m : ℝ) : 
  (1 < x → f m x ≤ f m (x + 1)) ↔ (m ∈ Icc (-1 : ℝ) 3) :=
begin
  sorry
end

end range_m_monotonic_increasing_l293_293149


namespace smallest_n_l293_293028

theorem smallest_n (n : ℕ) :
  (∀ k : ℕ, (k > 0) →  (∏ k in Finset.range (n + 1), ((100:ℝ) ^ (k / 15))) > (10 ^ 6 : ℝ)) ↔ n = 10 := 
sorry

end smallest_n_l293_293028


namespace y_in_terms_of_x_l293_293586

theorem y_in_terms_of_x (p x y : ℝ) (h1 : x = 2 + 2^p) (h2 : y = 1 + 2^(-p)) : 
  y = (x-1)/(x-2) :=
by
  sorry

end y_in_terms_of_x_l293_293586


namespace even_half_binom_count_l293_293097

def binom (n k : ℕ) : ℕ :=
if k > n then 0
else (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

noncomputable def count_even_half_binom : ℕ :=
(list.range 1000).count (\(n : ℕ) → is_even (binom (2 * (n + 1)) (n + 1) / 2))

theorem even_half_binom_count : count_even_half_binom = 990 := by sorry

end even_half_binom_count_l293_293097


namespace pairs_satisfying_equation_l293_293021

theorem pairs_satisfying_equation (a b : ℝ) : 
  (∀ n : ℕ, n > 0 → a * ⌊b * n⌋ = b * ⌊a * n⌋) ↔ 
  (a = 0 ∨ b = 0 ∨ a = b ∨ ∃ k : ℤ, a = k ∧ b = k) := 
by
  sorry

end pairs_satisfying_equation_l293_293021


namespace pinecone_ratio_l293_293749

-- Definitions from the conditions
def total_pinecones : ℕ := 2000
def percentage_eaten_by_reindeer : ℕ := 20
def percentage_collected_for_fires : ℕ := 25
def left_pinecones_after_squirrels : ℕ := 600

-- Theorem statement to prove
theorem pinecone_ratio : 
  let eaten_by_reindeer := (percentage_eaten_by_reindeer * total_pinecones) / 100 in
  let left_after_reindeer := total_pinecones - eaten_by_reindeer in
  let collected_for_fires := (percentage_collected_for_fires * left_after_reindeer) / 100 in
  let left_after_fires := left_after_reindeer - collected_for_fires in
  let eaten_by_squirrels := left_after_fires - left_pinecones_after_squirrels in
  eaten_by_squirrels / eaten_by_reindeer = 3 / 2 :=
by
  sorry

end pinecone_ratio_l293_293749


namespace simplify_fraction_solution_l293_293694

noncomputable def simplify_fraction (x : ℝ) (h : x ≠ 0) : ℝ :=
  (5 / (4 * x^(-2))) - ((4 * x^3) / 5)

theorem simplify_fraction_solution (x : ℝ) (h : x ≠ 0) :
  simplify_fraction x h = (25 * x^2 - 16 * x^3) / 20 := 
by
  sorry

end simplify_fraction_solution_l293_293694


namespace sum_of_first_2017_terms_l293_293532

variables {R : Type*} [LinearOrderedField R]
variables {f : R → R} {a : ℕ → R}

-- Conditions
def functional_eq (f : R → R) : Prop := ∀ x, f x = f (2 - x)
def monotonic_on (f : R → R) (s : Set R) : Prop := ∀ x ∈ s, ∀ y ∈ s, x ≤ y → f x ≤ f y
def arithmetic_seq (a : ℕ → R) : Prop := ∃ d, d ≠ 0 ∧ ∀ (n : ℕ), a (n + 1) = a n + d

-- Given
variables (h1 : functional_eq f)
variables (h2 : monotonic_on f (Set.Ici 1))
variables (h3 : arithmetic_seq a)
variables (h4 : f (a 5) = f (a 2011))  -- Using 0-based index for Lean's sequence notation

-- Goal
theorem sum_of_first_2017_terms (h1 : functional_eq f) (h2 : monotonic_on f (Set.Ici 1))
  (h3 : arithmetic_seq a) (h4 : f (a 5) = f (a 2011)) :
  (finset.range 2017).sum a = 2017 :=
  sorry

end sum_of_first_2017_terms_l293_293532


namespace quotient_calculation_l293_293769

theorem quotient_calculation : ∀ (dividend divisor remainder : ℕ), 
  dividend = 158 →
  divisor = 17 →
  remainder = 5 →
  (dividend - remainder) / divisor = 9 :=
by
  intros dividend divisor remainder h_dividend h_divisor h_remainder
  have h1 : dividend - remainder = 153 := by
    rw [h_dividend, h_remainder]
  have h2 : 153 / divisor = 9 := by
    rw [h_divisor]
    norm_num
  rw [h1, h2]
  sorry

end quotient_calculation_l293_293769


namespace convert_spherical_to_rectangular_l293_293839

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 15 (3 * Real.pi / 4) (Real.pi / 2) = 
    (-15 * Real.sqrt 2 / 2, 15 * Real.sqrt 2 / 2, 0) :=
by 
  sorry

end convert_spherical_to_rectangular_l293_293839


namespace directrix_of_parabola_l293_293909

theorem directrix_of_parabola (x_P y_P p : ℝ) (h1 : x_P = 1)
  (h2 : x_P^2 - 4 * x_P + y_P^2 = 0)
  (h3 : x_P^ 2 = -2 * p * y_P) (h4 : p > 0) :
    y_P = -real.sqrt 3 → (p = real.sqrt 3 / 6 → (-p = real.sqrt 3 / 12)) :=
by
  intro hy_P
  rw [hy_P] at h2 h3
  cases h2
  cases h3
  sorry

end directrix_of_parabola_l293_293909


namespace equal_pair_c_l293_293459

theorem equal_pair_c : (-4)^3 = -(4^3) := 
by {
  sorry
}

end equal_pair_c_l293_293459


namespace total_percentage_reduction_l293_293412

theorem total_percentage_reduction (P : ℝ) :
  let P1 := P * 0.7 in
  let P2 := P1 * 0.7 in
  ((P - P2) / P) * 100 = 51 := by sorry

end total_percentage_reduction_l293_293412


namespace fraction_meaningful_condition_l293_293710

theorem fraction_meaningful_condition (x : ℝ) : (∃ y, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end fraction_meaningful_condition_l293_293710


namespace angle_with_negative_vector_l293_293594

-- Define a function to represent the angle between two vectors
def angle_between (a b : EuclideanSpace ℝ) : ℝ := sorry

-- Assume the condition given in the problem
def given_condition (a b : EuclideanSpace ℝ) : Prop :=
  angle_between a b = 60

-- The main theorem we need to prove
theorem angle_with_negative_vector (a b : EuclideanSpace ℝ) (h : given_condition a b) :
  angle_between a (-b) = 120 :=
begin
  sorry
end

end angle_with_negative_vector_l293_293594


namespace sin_theta_value_l293_293515

-- Let θ be an angle in the first quadrant
variable {θ : ℝ}

-- Define the terminal side condition of θ lying on the line
def terminal_side_condition (x y : ℝ) : Prop := (5 * y - 3 * x = 0)

-- Define the correctness of the problem statement
theorem sin_theta_value 
  (h1 : terminal_side_condition x y)
  (h2 : 0 < x ∧ 0 < y)
  : sin θ = (3 / sqrt 34) :=
sorry

end sin_theta_value_l293_293515


namespace guesthouse_rolls_probability_l293_293424

theorem guesthouse_rolls_probability :
  let rolls := 12
  let guests := 3
  let types := 4
  let rolls_per_guest := 3
  let total_probability : ℚ := (12 / 12) * (9 / 11) * (6 / 10) * (3 / 9) *
                               (8 / 8) * (6 / 7) * (4 / 6) * (2 / 5) *
                               1
  let simplified_probability : ℚ := 24 / 1925
  total_probability = simplified_probability := sorry

end guesthouse_rolls_probability_l293_293424


namespace sin_30_sub_one_plus_pi_zero_l293_293833

theorem sin_30_sub_one_plus_pi_zero :
  let sin_30 := 1 / 2
  let one_plus_pi_zero := 1
  sin_30 - one_plus_pi_zero = -1 / 2 :=
by
  assume sin_30 : ℝ := 1 / 2
  assume one_plus_pi_zero : ℝ := 1
  show sin_30 - one_plus_pi_zero = -1 / 2 from sorry

end sin_30_sub_one_plus_pi_zero_l293_293833


namespace jack_runs_faster_than_paul_l293_293635

noncomputable def convert_km_hr_to_m_s (v : ℝ) : ℝ :=
  v * (1000 / 3600)

noncomputable def speed_difference : ℝ :=
  let v_J_km_hr := 20.62665  -- Jack's speed in km/hr
  let v_J_m_s := convert_km_hr_to_m_s v_J_km_hr  -- Jack's speed in m/s
  let distance := 1000  -- distance in meters
  let time_J := distance / v_J_m_s  -- Jack's time in seconds
  let time_P := time_J + 1.5  -- Paul's time in seconds
  let v_P_m_s := distance / time_P  -- Paul's speed in m/s
  let speed_diff_m_s := v_J_m_s - v_P_m_s  -- speed difference in m/s
  let speed_diff_km_hr := speed_diff_m_s * (3600 / 1000)  -- convert to km/hr
  speed_diff_km_hr

theorem jack_runs_faster_than_paul : speed_difference = 0.18225 :=
by
  -- Proof is omitted
  sorry

end jack_runs_faster_than_paul_l293_293635


namespace num_factors_of_32_l293_293946

theorem num_factors_of_32 : ∀ (n : ℕ), n = 32 → (∃ k, k = 6 ∧ ∀ d : ℕ, d ∣ n → d ∈ {1, 2, 4, 8, 16, 32}) :=
by
  intro n
  intro h
  use 6
  constructor
  . 
  exact 
  . 
  intro d
  intro h1
  simp
  sorry

end num_factors_of_32_l293_293946


namespace equal_pair_c_l293_293462

theorem equal_pair_c : (-4)^3 = -(4^3) := 
by {
  sorry
}

end equal_pair_c_l293_293462


namespace part1_part2_l293_293914

theorem part1 (a b : ℝ) 
  (h : ∀ x : ℝ, ax^2 - 3x + 2 > 0 ↔ x < 1 ∨ x > b) : a = 1 ∧ b = 2 :=
by sorry

theorem part2 (c : ℝ) 
  (h : ∀ x : ℝ, x^2 - (c + 2)*x + 2 * c ≤ 0 ↔ (c < 2 ∧ c ≤ x ∧ x ≤ 2) ∨ (c = 2 ∧ x = 2) ∨ (c > 2 ∧ 2 ≤ x ∧ x ≤ c)) : 
  part2_sol (c : ℝ) : a = 1 ∧ b = 2 → 
    (c < 2 → set_of (λ x : ℝ, x^2 - (c + 2) * x + 2 * c ≤ 0) = set.Icc c 2) ∧
    (c = 2 → set_of (λ x : ℝ, x^2 - (c + 2) * x + 2 * c ≤ 0) = {2}) ∧
    (c > 2 → set_of (λ x : ℝ, x^2 - (c + 2) * x + 2 * c ≤ 0) = set.Icc 2 c)
 :=
by sorry

end part1_part2_l293_293914


namespace josh_elimination_l293_293640

noncomputable def mark_out_every_second (lst : List ℕ) (start_idx : ℕ) : List ℕ :=
  lst.enum.foldl (λ acc x, if (acc.size + start_idx).odd then acc.push x.snd else acc) []

noncomputable def last_remaining_number (lst : List ℕ) : ℕ :=
  if lst.length ≤ 1 then lst.head? else last_remaining_number (mark_out_every_second lst 1)

theorem josh_elimination :
  last_remaining_number (List.range 150).map (· + 1) = 73 :=
by
  sorry

end josh_elimination_l293_293640


namespace jay_change_l293_293988

def cost_book : ℝ := 25
def cost_pen : ℝ := 4
def cost_ruler : ℝ := 1
def payment : ℝ := 50

theorem jay_change : (payment - (cost_book + cost_pen + cost_ruler) = 20) := sorry

end jay_change_l293_293988


namespace remaining_sugar_l293_293261

-- Conditions as definitions
def total_sugar : ℝ := 9.8
def spilled_sugar : ℝ := 5.2

-- Theorem to prove the remaining sugar
theorem remaining_sugar : total_sugar - spilled_sugar = 4.6 := by
  sorry

end remaining_sugar_l293_293261


namespace infinite_series_convergence_l293_293836

theorem infinite_series_convergence :
  ∃ L : ℝ, Tendsto (fun n => (Finset.range n).sum (λ n, (3 * (n : ℝ) + 2) / ((n : ℝ) * (n + 1) * (n + 3)))) atTop (𝓝 L) :=
sorry

end infinite_series_convergence_l293_293836


namespace not_possible_total_47_l293_293301

open Nat

theorem not_possible_total_47 (h c : ℕ) : ¬ (13 * h + 5 * c = 47) :=
  sorry

end not_possible_total_47_l293_293301


namespace median_of_transformed_data_set_l293_293291

open List

theorem median_of_transformed_data_set :
  (∀ X, mode ([15, X, 9, 11, 7] : List ℝ) = 11) →
  median ([10, 11, 14, 8, X] : List ℝ) = 11 :=
by
  intro h

  have hX_11 : X = 11 :=
    by
    sorry

  rw [hX_11]

  have := List.sort [10, 11, 14, 8, 11]
  have := List.nth_le ([8, 10, 11, 11, 14] : List ℝ) 2 _
  sorry

end median_of_transformed_data_set_l293_293291


namespace maximum_candies_l293_293307

theorem maximum_candies (n : ℕ) (h1 : n = 31) : 
  (∑ i in finset.range (n - 1), (n - i) * i) = 465 :=
by 
  -- Conditions derived directly from the problem statement
  sorry

end maximum_candies_l293_293307


namespace sum_over_difference_l293_293889

variable {a : ℕ → ℝ}

def non_zero_sequence := ∀ n : ℕ, a n ≠ 0
def satisfies_relation := ∀ n : ℕ, a (n+1)^2 = a n * a (n+2)
def specific_condition := 32 * a 8 - a 3 = 0

theorem sum_over_difference (h1 : non_zero_sequence a) (h2 : satisfies_relation a) (h3 : specific_condition a) :
  let S (n : ℕ) := ∑ k in Finset.range n, a k in
  (S 6) / (a 1 - S 3) = -21 / 8 :=
by
  let S (n : ℕ) := ∑ k in Finset.range n, a k
  sorry

end sum_over_difference_l293_293889


namespace isosceles_triangle_l293_293537

theorem isosceles_triangle
  (α β γ x y z w : ℝ)
  (h_triangle : α + β + γ = 180)
  (h_quad : x + y + z + w = 360)
  (h_conditions : (x = α + β) ∧ (y = β + γ) ∧ (z = γ + α) ∨ (w = α + β) ∧ (x = β + γ) ∧ (y = γ + α) ∨ (z = α + β) ∧ (w = β + γ) ∧ (x = γ + α) ∨ (y = α + β) ∧ (z = β + γ) ∧ (w = γ + α))
  : α = β ∨ β = γ ∨ γ = α := 
sorry

end isosceles_triangle_l293_293537


namespace line_equation_l293_293283

theorem line_equation (p : ℝ × ℝ) (a : ℝ × ℝ) :
  p = (4, -4) →
  a = (1, 2 / 7) →
  ∃ (m b : ℝ), m = 2 / 7 ∧ b = -36 / 7 ∧ ∀ x y : ℝ, y = m * x + b :=
by
  intros hp ha
  sorry

end line_equation_l293_293283


namespace three_digit_sum_permutations_l293_293257

theorem three_digit_sum_permutations (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) (h₃ : 1 ≤ b) (h₄ : b ≤ 9) (h₅ : 1 ≤ c) (h₆ : c ≤ 9)
  (h₇ : n = 100 * a + 10 * b + c)
  (h₈ : 222 * (a + b + c) - n = 1990) :
  n = 452 :=
by
  sorry

end three_digit_sum_permutations_l293_293257


namespace perpendicularity_condition_l293_293894

theorem perpendicularity_condition 
  (A B C D E F k b : ℝ) 
  (h1 : b ≠ 0)
  (line : ∀ (x : ℝ), y = k * x + b)
  (curve : ∀ (x y : ℝ), A * x^2 + 2 * B * x * y + C * y^2 + 2 * D * x + 2 * E * y + F = 0):
  A * b^2 - 2 * D * k * b + F * k^2 + C * b^2 + 2 * E * b + F = 0 :=
sorry

end perpendicularity_condition_l293_293894


namespace part1_part2_l293_293478

variable (S : Type) (f : S → S → S)

-- Conditions
axiom binary_property : ∀ a b c d : S, f (f a b) (f c d) = f a d
axiom ab_eq_c : ∀ a b c : S, f a b = c

-- Part 1
theorem part1 (a b c : S) (h : f a b = c) : f c c = c := by
  have h1 : f (f a b) (f a b) = f a a := by rw [binary_property]
  rw [h] at h1
  exact h1

-- Part 2
theorem part2 (a b c d : S) (h : f a b = c) : f a d = f c d := by
  have h2 : f (f a b) d = f a d := by rw [binary_property]
  rw [h] at h2
  exact h2

end part1_part2_l293_293478


namespace a5_a6_less_than_a4_squared_l293_293142

variable {a : ℕ → ℝ}
variable {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem a5_a6_less_than_a4_squared
  (h_geo : is_geometric_sequence a q)
  (h_cond : a 5 * a 6 < (a 4) ^ 2) :
  0 < q ∧ q < 1 :=
sorry

end a5_a6_less_than_a4_squared_l293_293142


namespace complex_z_count_l293_293857

-- Translate conditions as definitions
definition is_on_unit_circle (z : ℂ) : Prop :=
  complex.abs z = 1

-- Translate the problem statement
theorem complex_z_count (z : ℂ) (h : is_on_unit_circle z) :
  (z ^ 7.factorial - z ^ 6.factorial).im = 0 ↔
  ∃ n : ℕ, n = 7200 :=
sorry

end complex_z_count_l293_293857


namespace card_swapping_cost_l293_293755

noncomputable def swap_cost (x y : ℕ) : ℕ := 2 * (x - y).natAbs

theorem card_swapping_cost (n : ℕ) (a : Fin n → Fin n) (perm : ∀ i, 1 ≤ a i.val.succ ∧ a i.val.succ <= n) :
  ∃ (swap_seq : List (Fin n × Fin n)), 
  (∀ (swap : Fin n × Fin n), swap ∈ swap_seq → 
      swap_cost swap.1.val swap.2.val <= |swap.1.val + 1 - swap.2.val + 1| ) ∧ 
  swap_seq.map (λ swap, swap_cost swap.1.val swap.2.val).sum ≤ ∑ i, (a i).val ∣ i := 
begin
  sorry,
end

end card_swapping_cost_l293_293755


namespace compare_sum_perimeter_l293_293761

noncomputable def point := ℝ × ℝ
noncomputable def triangle := point × point × point

def P : point := (-2, 3)
def Q : point := (4, 5)
def R : point := (1, -4)

def S : point := ((-2 + 4 + 1) / 3, (3 + 5 - 4) / 3)

def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def perimeter (t : triangle) : ℝ :=
  let (p1, p2, p3) := t
  distance p1 p2 + distance p2 p3 + distance p3 p1

def T : triangle := (P, Q, R)

def sum : ℝ := 10 * S.1 + S.2

theorem compare_sum_perimeter :
  sum = 34 / 3 ∧ sum ≤ perimeter T := by
  sorry

end compare_sum_perimeter_l293_293761


namespace open_box_volume_calculation_l293_293429

-- Definitions of the conditions
def original_length : ℝ := 48
def original_width : ℝ := 38
def thickness : ℝ := 0.5
def cut_square_side : ℝ := 8

-- Function to calculate the resulting volume of the open box
noncomputable def calculate_internal_volume (l w t s: ℝ) : ℝ :=
  let new_length := l - 2 * s
      new_width := w - 2 * s
      internal_length := new_length - 2 * t
      internal_width := new_width - 2 * t
      height := s
  in internal_length * internal_width * height

-- Statement of the theorem to be proved
theorem open_box_volume_calculation :
  calculate_internal_volume original_length original_width thickness cut_square_side = 5208 := sorry


end open_box_volume_calculation_l293_293429


namespace probability_x_greater_than_6_l293_293916

-- Given that the variable x follows a normal distribution N(4, σ^2)
def x_distribution (σ : ℝ) : MeasureTheory.ProbabilityMeasure ℝ :=
  MeasureTheory.ProbabilityMeasure.normal 4 σ

-- Given that P(x > 2) = 0.6
axiom P_x_greater_than_2 (σ : ℝ) : MeasureTheory.ProbabilityMeasure.prob (x_distribution σ) { x : ℝ | x > 2 } = 0.6

-- Prove that P(x > 6) = 0.4
theorem probability_x_greater_than_6 (σ : ℝ) : 
  MeasureTheory.ProbabilityMeasure.prob (x_distribution σ) { x : ℝ | x > 6 } = 0.4 :=
by
  sorry

end probability_x_greater_than_6_l293_293916


namespace wendy_total_sales_l293_293359

noncomputable def apple_price : ℝ := 1.50
noncomputable def orange_price : ℝ := 1.00
noncomputable def morning_apples : ℕ := 40
noncomputable def morning_oranges : ℕ := 30
noncomputable def afternoon_apples : ℕ := 50
noncomputable def afternoon_oranges : ℕ := 40

theorem wendy_total_sales :
  (morning_apples * apple_price + morning_oranges * orange_price) +
  (afternoon_apples * apple_price + afternoon_oranges * orange_price) = 205 := by
  sorry

end wendy_total_sales_l293_293359


namespace find_heaviest_and_lightest_l293_293324

-- Definition of the main problem conditions
def coins : ℕ := 10
def max_weighings : ℕ := 13
def distinct_weights (c : ℕ) : Prop := ∀ (i j : ℕ), i ≠ j → i < c → j < c → weight i ≠ weight j

-- Noncomputed property representing the weight of each coin
noncomputable def weight : ℕ → ℝ := sorry

-- The main theorem statement
theorem find_heaviest_and_lightest (c : ℕ) (mw : ℕ) (dw : distinct_weights c) : c = coins ∧ mw = max_weighings
  → ∃ (h l : ℕ), h < c ∧ l < c ∧ (∀ (i : ℕ), i < c → weight i ≤ weight h ∧ weight i ≥ weight l) :=
by
  sorry

end find_heaviest_and_lightest_l293_293324


namespace parabola_vertex_l293_293304

def parabola_conditions
  (a b c : ℝ)
  (vertex_cond1 : (4 * a * c - b^2) / (4 * a) = -11)
  (vertex_cond2 : -b / (2 * a) = 4)
  (x_intersect_cond1 : b^2 - 4 * a * c > 0)
  (x_intersect_cond2 : c / a < 0) : Prop :=
a > 0 ∧ b < 0 ∧ c < 0

theorem parabola_vertex
  (a b c : ℝ)
  (vertex_cond1 : (4 * a * c - b^2) / (4 * a) = -11)
  (vertex_cond2 : -b / (2 * a) = 4)
  (x_intersect_cond1 : b^2 - 4 * a * c > 0)
  (x_intersect_cond2 : c / a < 0) : parabola_conditions a b c :=
by {
  sorry -- Proof goes here
}

end parabola_vertex_l293_293304


namespace number_of_solutions_l293_293861

-- Define the condition for z being a complex number with |z| = 1
def is_on_unit_circle (z : ℂ) : Prop := complex.abs z = 1

-- Define the condition for z^{7!} - z^{6!} being a real number
def is_real_output (z : ℂ) : Prop := ∃ (a : ℝ), z ^ 5040 - z ^ 720 = a

-- The main theorem we need to prove
theorem number_of_solutions : 
  (finset.univ.filter (λ z : ℂ, is_on_unit_circle z ∧ is_real_output z)).card = 7200 := 
sorry

end number_of_solutions_l293_293861


namespace sequence_a_n_2013_l293_293162

theorem sequence_a_n_2013 (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 6)
  (h : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 2013 = 3 :=
sorry

end sequence_a_n_2013_l293_293162


namespace john_flights_of_stairs_l293_293990

theorem john_flights_of_stairs (x : ℕ) : 
    let flight_height := 10
    let rope_height := flight_height / 2
    let ladder_height := rope_height + 10
    let total_height := 70
    10 * x + rope_height + ladder_height = total_height → x = 5 :=
by
    intro h
    sorry

end john_flights_of_stairs_l293_293990


namespace math_problem_solution_l293_293887

theorem math_problem_solution : ∀ x y : ℝ, (|x - 3| + real.sqrt (x - y + 1) = 0) → x = 3 ∧ y = 4 → real.sqrt (x^2 * y + x * y^2 + 1/4 * y^3) = 10 :=
by
  intros x y h₁ h₂
  sorry

end math_problem_solution_l293_293887


namespace num_solutions_correct_l293_293865

noncomputable def num_solutions : ℕ :=
  let factorial (n : ℕ) := (nat.factorial n)
  let z_is_on_unit_circle (z : ℂ) : Prop := abs z = 1
  let expression_is_real (z : ℂ) : Prop := z^(factorial 7) - z^(factorial 6) ∈ ℝ
  
  nat.card { z : ℂ // z_is_on_unit_circle z ∧ expression_is_real z }

theorem num_solutions_correct : num_solutions = 44 := by
  sorry

end num_solutions_correct_l293_293865


namespace sphere_surface_area_l293_293544

theorem sphere_surface_area (A B C P O : Type)
  (h1 : A ∈ O ∧ B ∈ O ∧ C ∈ O ∧ P ∈ O)
  (h2 : ∀ a b c : O, equilateral_triangle a b c → a.dist b = 4 * sqrt 3)
  (h3 : ∀ a b c p : O, maximum_volume_pyramid a b c p = 32 * sqrt 3) :
  surface_area O = 100 * π := sorry

end sphere_surface_area_l293_293544


namespace area_of_gray_region_l293_293369

theorem area_of_gray_region (d_s : ℝ) (h1 : d_s = 4) (h2 : ∀ r_s r_l : ℝ, r_s = d_s / 2 → r_l = 3 * r_s) : ℝ :=
by
  let r_s := d_s / 2
  let r_l := 3 * r_s
  let area_larger := π * r_l^2
  let area_smaller := π * r_s^2
  let area_gray := area_larger - area_smaller
  have hr_s : r_s = 2 := by sorry
  have hr_l : r_l = 6 := by sorry
  have ha_larger : area_larger = 36 * π := by sorry
  have ha_smaller : area_smaller = 4 * π := by sorry
  have ha_gray : area_gray = 32 * π := by sorry
  exact ha_gray

end area_of_gray_region_l293_293369


namespace ceil_sum_sqrt_evaluation_l293_293053

theorem ceil_sum_sqrt_evaluation :
  (⌈real.sqrt 3⌉ + ⌈real.sqrt 33⌉ + ⌈real.sqrt 333⌉ = 27) :=
begin
  have h1 : 1 < real.sqrt 3 ∧ real.sqrt 3 < 2 := sorry,
  have h2 : 5 < real.sqrt 33 ∧ real.sqrt 33 < 6 := sorry,
  have h3 : 18 < real.sqrt 333 ∧ real.sqrt 333 < 19 := sorry,
  sorry,
end

end ceil_sum_sqrt_evaluation_l293_293053


namespace most_reasonable_sampling_method_l293_293789

-- Define the conditions
variables (primary_vision junior_vision senior_vision : Type)
variables (vision_diff_stages : primary_vision ≠ junior_vision ∧ junior_vision ≠ senior_vision ∧ primary_vision ≠ senior_vision)
variables (vision_diff_gender : Prop := false)

-- State the theorem to be proved
theorem most_reasonable_sampling_method 
  (primary_vision junior_vision senior_vision : Type)
  (vision_diff_stages : primary_vision ≠ junior_vision ∧ junior_vision ≠ senior_vision ∧ primary_vision ≠ senior_vision)
  (vision_diff_gender : Prop := false) : 
  {method : Type // method = "Stratified sampling by educational stage"} :=
begin
  -- The proof is skipped with a placeholder 
  exact ⟨"Stratified sampling by educational stage", rfl⟩,
end

end most_reasonable_sampling_method_l293_293789


namespace proof_equation_D_is_true_l293_293387

theorem proof_equation_D_is_true :
  (-(+3) ≠ 3) ∧ (-(-2) ≠ +(-2)) ∧ (-|-4| ≠ 4) ∧ (-|+5| = -|-5|) :=
by
  sorry

end proof_equation_D_is_true_l293_293387


namespace find_f_find_min_max_l293_293555

-- Given conditions
def h (x : ℝ) := x + 1 / x + 2
def is_symmetric (f g : ℝ → ℝ) (A : ℝ × ℝ) :=
  ∀ x y, f x = y → g (-x) = 2 - y

-- Problem statements
theorem find_f (f : ℝ → ℝ) (A : ℝ × ℝ) :
  is_symmetric f h A ∧ A = (0, 1) → f = (λ x, x + 1 / x) :=
by
  sorry

theorem find_min_max (f : ℝ → ℝ) :
  f = (λ x, x + 1 / x) → 
  (∀ x, 0 < x ∧ x ≤ 8 → 2 ≤ f x ∧ f x ≤ (65 / 8)) :=
by
  sorry

end find_f_find_min_max_l293_293555


namespace intersection_complement_M_N_l293_293166

open Set

noncomputable def U : Set ℝ := set.univ

noncomputable def M : Set ℝ := {x | ∃ (y : ℝ), y = real.log (1 - x)}

noncomputable def N : Set ℝ := {x | 2 ^ (x * (x - 2)) < 1}

noncomputable def complement_M_in_U : Set ℝ := U \ M

noncomputable def target_set : Set ℝ := (complement_M_in_U ∩ N)

theorem intersection_complement_M_N :
  target_set = {x | 1 ≤ x ∧ x < 2} :=
sorry

end intersection_complement_M_N_l293_293166


namespace geometric_sequence_common_ratio_l293_293112

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : S 2 = 2 * a 2 + 3)
  (h2 : S 3 = 2 * a 3 + 3)
  (h3 : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) : q = 2 := 
by
  sorry

end geometric_sequence_common_ratio_l293_293112


namespace gnomos_have_at_least_336_keys_l293_293625

-- Define the number of gnomos and locks
constant num_gnomos : ℕ := 7
constant num_locks : ℕ := 144

-- Define the property that any three gnomos can open all locks
constant can_open_all_locks : (finset (fin num_gnomos)) → Prop

axiom any_three_gnomos_open_all_locks :
  ∀ (g1 g2 g3 : fin num_gnomos), 
  can_open_all_locks {g1, g2, g3}

-- State the theorem
theorem gnomos_have_at_least_336_keys : 
  ∃ (keys : finset (fin num_locks)) (own_keys : fin num_gnomos → finset (fin num_locks)), 
  (∀ g1 g2 g3 : fin num_gnomos, (own_keys g1 ∪ own_keys g2 ∪ own_keys g3) = finset.univ) →
  (∑ i, (own_keys i).card ≥ 336) := 
sorry

end gnomos_have_at_least_336_keys_l293_293625


namespace correct_transformation_l293_293451

theorem correct_transformation :
  (∀ a b c : ℝ, c ≠ 0 → (a / c = b / c ↔ a = b)) ∧
  (∀ x : ℝ, ¬ (x / 4 + x / 3 = 1 ∧ 3 * x + 4 * x = 1)) ∧
  (∀ a b c : ℝ, ¬ (a * b = b * c ∧ a ≠ c)) ∧
  (∀ x a : ℝ, ¬ (4 * x = a ∧ x = 4 * a)) := sorry

end correct_transformation_l293_293451


namespace minimize_sum_of_a_b_l293_293548

-- Definitions and conditions
variables {a b : ℝ}
def is_positive (x : ℝ) : Prop := x > 0
def m := (a, 4 : ℝ)
def n := (b, b - 1 : ℝ)
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

-- Lean 4 statement for the proof problem
theorem minimize_sum_of_a_b (ha : is_positive a) (hb : is_positive b) (h_parallel : parallel m n) : a + b = 9 :=
by sorry

end minimize_sum_of_a_b_l293_293548


namespace circle_reflections_l293_293708

def reflect_y_eq_x (p : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (p.snd, p.fst)

def reflect_y_eq_neg_x (p : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (-p.snd, -p.fst)

theorem circle_reflections {x y : ℝ} (h : (x, y) = (3, -7)) :
  let p1 := reflect_y_eq_x (x, y)
  let p2 := reflect_y_eq_neg_x p1
  in p2 = (3, 7) :=
by
  sorry

end circle_reflections_l293_293708


namespace minimal_degree_of_g_l293_293843

noncomputable def g_degree_minimal (f g h : Polynomial ℝ) (deg_f : ℕ) (deg_h : ℕ) (h_eq : (5 : ℝ) • f + (7 : ℝ) • g = h) : Prop :=
  Polynomial.degree f = deg_f ∧ Polynomial.degree h = deg_h → Polynomial.degree g = 12

theorem minimal_degree_of_g (f g h : Polynomial ℝ) (h_eq : (5 : ℝ) • f + (7 : ℝ) • g = h)
    (deg_f : Polynomial.degree f = 5) (deg_h : Polynomial.degree h = 12) :
    Polynomial.degree g = 12 := by
  sorry

end minimal_degree_of_g_l293_293843


namespace expr_value_l293_293782

theorem expr_value : 2 ^ (1 + 2 + 3) - (2 ^ 1 + 2 ^ 2 + 2 ^ 3) = 50 :=
by
  sorry

end expr_value_l293_293782


namespace find_rate_l293_293401

-- Given conditions
def principal : ℝ := 750
def amount : ℝ := 1200
def time : ℕ := 5

-- Mathematical equivalence: A = P + (P * R * T) / 100
def interest_rate : ℝ := 12 / 100

theorem find_rate (P A : ℝ) (T : ℕ) (hP : P = principal) (hA : A = amount) (hT : T = time) :
  A = P + P * interest_rate * T :=
sorry

end find_rate_l293_293401


namespace dice_square_factor_probability_l293_293950

theorem dice_square_factor_probability :
  let dice_faces := {1, 2, 3, 4, 5, 6}
  in let no_square_factor_probability := ((4:ℚ) / 6) ^ 6
  in let square_factor_probability := 1 - no_square_factor_probability
  in square_factor_probability = 665 / 729 :=
by
  let dice_faces := {1, 2, 3, 4, 5, 6}
  let no_square_factor_probability := ((4:ℚ) / 6) ^ 6
  let square_factor_probability := 1 - no_square_factor_probability
  have : square_factor_probability = 665 / 729 := sorry
  exact this

end dice_square_factor_probability_l293_293950


namespace number_of_children_l293_293492

-- Definitions of the conditions
def crayons_per_child : ℕ := 8
def total_crayons : ℕ := 56

-- Statement of the problem
theorem number_of_children : total_crayons / crayons_per_child = 7 := by
  sorry

end number_of_children_l293_293492


namespace moles_of_CaCO3_used_l293_293855

-- Definitions according to the conditions
def balancedEquation : Prop := 
  ∀ (HCl CaCO3 CaCl2 H2O CO2 : Type),
  (2 * HCl + CaCO3) → (CaCl2 + H2O + CO2)

def molarMass_H2O : ℝ := 18

def mass_H2O : ℝ := 18

def moles_H2O := mass_H2O / molarMass_H2O

-- Theorem statement
theorem moles_of_CaCO3_used (HCl CaCO3 CaCl2 H2O CO2 : Type) :
  balancedEquation HCl CaCO3 CaCl2 H2O CO2 →
  moles_H2O = 1 →
  ∃ n : ℝ, n = 1 := 
by
  sorry

end moles_of_CaCO3_used_l293_293855


namespace camp_cedar_total_counselors_l293_293475

def number_of_boys : ℕ := 48
def number_of_girls : ℕ := 4 * number_of_boys - 12
def counselors_for_boys : ℕ := number_of_boys / 6
def counselors_for_girls : ℕ := number_of_girls / 10

theorem camp_cedar_total_counselors : counselors_for_boys + counselors_for_girls = 26 := by
  have h_boys : number_of_boys = 48 := rfl
  have h_girls : number_of_girls = 4 * number_of_boys - 12 := rfl
  have h_cfb : counselors_for_boys = number_of_boys / 6 := rfl
  have h_cfg : counselors_for_girls = number_of_girls / 10 := rfl
  -- Now calculate the total number of counselors
  calc
    counselors_for_boys + counselors_for_girls = 8 + 18 := by
      rw [←h_cfb, ←h_cfg, ←h_boys, ←h_girls]
      norm_num
    ... = 26 := by norm_num

end camp_cedar_total_counselors_l293_293475


namespace close_1000_roads_no_odd_closed_route_l293_293305

theorem close_1000_roads_no_odd_closed_route (cities : Fin 2000) (roads : cities → Fin 3) :
  ∃ closed_roads : Fin 1000 → cities × cities, ∀ (c1 c2 c3 : closed_roads), ¬(odd_number_of_roads c1 c2 c3) := by
  sorry

end close_1000_roads_no_odd_closed_route_l293_293305


namespace find_five_digit_number_l293_293385

def reverse_number (n : ℕ) : ℕ :=
  let digits := n.digits 10 in
  digits.reverse.to_nat

theorem find_five_digit_number :
  ∃ n : ℕ, (9999 < n ∧ n < 100000) ∧ (9 * n = reverse_number n) ∧ n = 10989 :=
by
  sorry

end find_five_digit_number_l293_293385


namespace ceil_sum_sqrt_evaluation_l293_293054

theorem ceil_sum_sqrt_evaluation :
  (⌈real.sqrt 3⌉ + ⌈real.sqrt 33⌉ + ⌈real.sqrt 333⌉ = 27) :=
begin
  have h1 : 1 < real.sqrt 3 ∧ real.sqrt 3 < 2 := sorry,
  have h2 : 5 < real.sqrt 33 ∧ real.sqrt 33 < 6 := sorry,
  have h3 : 18 < real.sqrt 333 ∧ real.sqrt 333 < 19 := sorry,
  sorry,
end

end ceil_sum_sqrt_evaluation_l293_293054


namespace identify_heaviest_and_lightest_l293_293332

   def coin : Type := ℕ  -- let's represent coins as natural numbers for simplicity.

   def has_different_weights (coins : list coin) : Prop := 
     ∀ (c1 c2 : coin), c1 ∈ coins → c2 ∈ coins → c1 ≠ c2 → weight(c1) ≠ weight(c2)

   def weight : coin → ℝ := -- assume a function that gives the weight corresponding to a coin.
     sorry 

   theorem identify_heaviest_and_lightest (coins : list coin) 
     (h₁ : length coins = 10)
     (h₂ : has_different_weights coins) : 
     ∃ (heaviest lightest : coin), 
       (heaviest ∈ coins) ∧ (lightest ∈ coins) ∧
       (∀ c ∈ coins, weight c ≤ weight heaviest) ∧
       (∀ c ∈ coins, weight c ≥ weight lightest) :=
   by 
     sorry
   
end identify_heaviest_and_lightest_l293_293332


namespace orange_harvest_exists_l293_293491

theorem orange_harvest_exists :
  ∃ (A B C D : ℕ), A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ A + B + C + D = 56 :=
by
  use 10
  use 15
  use 16
  use 15
  repeat {split};
  sorry

end orange_harvest_exists_l293_293491


namespace outlier_attribute_l293_293057

/-- Define the given attributes of the Dragon -/
def one_eyed := "одноокий"
def two_eared := "двуухий"
def three_tailed := "треххвостый"
def four_legged := "четырехлапый"
def five_spiked := "пятиглый"

/-- Define a predicate to check if an attribute contains doubled letters -/
def has_doubled_letters (s : String) : Bool :=
  let chars := s.toList
  chars.any (fun ch => chars.count ch > 1)

/-- Prove that "четырехлапый" (four-legged) does not fit the pattern of containing doubled letters -/
theorem outlier_attribute : ¬ has_doubled_letters four_legged :=
by
  -- Proof would be inserted here
  sorry

end outlier_attribute_l293_293057


namespace factorize_x_squared_plus_2x_l293_293059

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) :=
by sorry

end factorize_x_squared_plus_2x_l293_293059


namespace ratio_aerobics_to_weight_training_l293_293993

def time_spent_exercising : ℕ := 250
def time_spent_aerobics : ℕ := 150
def time_spent_weight_training : ℕ := 100

theorem ratio_aerobics_to_weight_training :
    (time_spent_aerobics / gcd time_spent_aerobics time_spent_weight_training) = 3 ∧
    (time_spent_weight_training / gcd time_spent_aerobics time_spent_weight_training) = 2 :=
by
    sorry

end ratio_aerobics_to_weight_training_l293_293993


namespace sum_of_rationals_l293_293253

-- Let a1, a2, a3, a4 be 4 rational numbers such that the set of products of distinct pairs is given.
def valid_products (a1 a2 a3 a4 : ℚ) : Prop :=
  {a1 * a2, a1 * a3, a1 * a4, a2 * a3, a2 * a4, a3 * a4} = {-24, -2, -3/2, -1/8, 1, 3}

-- Define the theorem which asserts the sum of these rational numbers is either 9/4 or -9/4.
theorem sum_of_rationals (a1 a2 a3 a4 : ℚ) (h : valid_products a1 a2 a3 a4) :
  a1 + a2 + a3 + a4 = 9/4 ∨ a1 + a2 + a3 + a4 = -9/4 :=
sorry

end sum_of_rationals_l293_293253


namespace evaluate_ceiling_sums_l293_293036

theorem evaluate_ceiling_sums : 
  (⌈real.sqrt 3⌉ + ⌈real.sqrt 33⌉ + ⌈real.sqrt 333⌉) = 27 :=
by
  have h1 : 1 < real.sqrt 3 ∧ real.sqrt 3 < 2 :=
    ⟨by norm_num, by norm_num⟩,
  have h2 : 5 < real.sqrt 33 ∧ real.sqrt 33 < 6 :=
    ⟨by norm_num, by norm_num⟩,
  have h3 : 18 < real.sqrt 333 ∧ real.sqrt 333 < 19 :=
    ⟨by norm_num, by norm_num⟩,
  sorry

end evaluate_ceiling_sums_l293_293036


namespace inequality_solution_l293_293853

theorem inequality_solution
  : {x : ℝ | (x^2 / (x + 2)^2) ≥ 0} = {x : ℝ | x ≠ -2} :=
by
  sorry

end inequality_solution_l293_293853


namespace weight_group_9_kg_l293_293423

theorem weight_group_9_kg {weights : Finset ℕ} :
  weights = Finset.range 13 \ {0} →
  ∃ (G1 G2 G3 : Finset ℕ), 
      G1.card = 4 ∧ G2.card = 4 ∧ G3.card = 4 ∧ 
      G1.sum id = 41 ∧ G2.sum id = 26 ∧ 
      (9 ∈ G2) ∧ (7 ∈ G2) ∧
      (G1 ∪ G2 ∪ G3 = weights) ∧ 
      (G1 ∩ G2 ∩ G3 = ∅) := 
by
    sorry

end weight_group_9_kg_l293_293423


namespace measure_of_angle_YZX_l293_293013

noncomputable def problem_statement : Prop :=
  ∃ (A B C X Y Z : ℝ),
    X ∈ (segment B C) ∧
    Y ∈ (segment A B) ∧
    Z ∈ (segment A C) ∧
    angle A = 50 ∧
    angle B = 70 ∧
    angle C = 60 ∧
    let Γ := incircle ABC in
    let γ := circumcircle XYZ in
    incircle ABC = Γ ∧ 
    circumcircle XYZ = γ ∧ 
    ∠YZX = 115

theorem measure_of_angle_YZX (A B C X Y Z : ℝ) :
  X ∈ (segment B C) →
  Y ∈ (segment A B) →
  Z ∈ (segment A C) →
  angle A = 50 →
  angle B = 70 →
  angle C = 60 →
  let Γ := incircle ABC in
  let γ := circumcircle XYZ in
  incircle ABC = Γ →
  circumcircle XYZ = γ →
  ∠YZX = 115 :=
by
  sorry

end measure_of_angle_YZX_l293_293013


namespace percentage_denied_from_west_side_high_l293_293852

theorem percentage_denied_from_west_side_high 
  (denied_riverside : ℕ := (0.20 * 120).toNat) 
  (denied_mountaintop : ℕ := (0.50 * 50).toNat) 
  (kids_got_in : ℕ := 148) 
  (total_kids_all_schools : ℕ := 260) 
  (total_kids_west_side : ℕ := 90) 
  (W : ℕ := total_kids_all_schools - kids_got_in - denied_riverside - denied_mountaintop)
  : (W / total_kids_west_side * 100 = 70) := 
  by
  sorry

end percentage_denied_from_west_side_high_l293_293852


namespace area_of_gray_region_l293_293364

theorem area_of_gray_region
  (r : ℝ)
  (R : ℝ)
  (diameter_small_circle : ℝ)
  (diameter_small_circle_eq : diameter_small_circle = 4)
  (radius_small_circle : r = diameter_small_circle / 2)
  (radius_large_circle : R = 3 * r) :
  let As := π * r^2,
      AL := π * R^2 in
  AL - As = 32 * π :=
by
  -- Definitions for readability and decoration
  have radius_smaller := diameter_small_circle_eq ▸ radius_small_circle,
  have radius_larger := congr_arg (λ x, 3 * x) radius_smaller,
  let area_smaller := π * (r^2),
  let area_larger := π * (R^2),
  sorry

end area_of_gray_region_l293_293364


namespace placing_balls_into_boxes_l293_293880

noncomputable def total_ways_placing_balls (balls : Finset ℕ) (boxes : Finset ℕ) : ℕ :=
   if h : 5 = balls.card ∧ 3 = boxes.card then 180 else 0

theorem placing_balls_into_boxes :
  ∀ (balls boxes : Finset ℕ), 
    5 = balls.card ∧ 3 = boxes.card → 
    (∃ b1 b2 b3 : Finset ℕ, b1 ∪ b2 ∪ b3 = balls ∧ b1 ∩ b2 = ∅ ∧ b2 ∩ b3 = ∅ ∧ b1 ∩ b3 = ∅ ∧
    b1.nonempty ∧ b2.nonempty ∧ b3.nonempty) →
    total_ways_placing_balls balls boxes = 180 := 
by sorry

end placing_balls_into_boxes_l293_293880


namespace isosceles_triangle_l293_293538

theorem isosceles_triangle
  (α β γ x y z w : ℝ)
  (h_triangle : α + β + γ = 180)
  (h_quad : x + y + z + w = 360)
  (h_conditions : (x = α + β) ∧ (y = β + γ) ∧ (z = γ + α) ∨ (w = α + β) ∧ (x = β + γ) ∧ (y = γ + α) ∨ (z = α + β) ∧ (w = β + γ) ∧ (x = γ + α) ∨ (y = α + β) ∧ (z = β + γ) ∧ (w = γ + α))
  : α = β ∨ β = γ ∨ γ = α := 
sorry

end isosceles_triangle_l293_293538


namespace oblique_line_plane_angle_range_l293_293293

/-- 
An oblique line intersects the plane at an angle other than a right angle. 
The angle cannot be $0$ radians or $\frac{\pi}{2}$ radians.
-/
theorem oblique_line_plane_angle_range (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) : 
  0 < θ ∧ θ < π / 2 :=
by {
  exact ⟨h₀, h₁⟩
}

end oblique_line_plane_angle_range_l293_293293


namespace number_of_solutions_l293_293862

-- Define the condition for z being a complex number with |z| = 1
def is_on_unit_circle (z : ℂ) : Prop := complex.abs z = 1

-- Define the condition for z^{7!} - z^{6!} being a real number
def is_real_output (z : ℂ) : Prop := ∃ (a : ℝ), z ^ 5040 - z ^ 720 = a

-- The main theorem we need to prove
theorem number_of_solutions : 
  (finset.univ.filter (λ z : ℂ, is_on_unit_circle z ∧ is_real_output z)).card = 7200 := 
sorry

end number_of_solutions_l293_293862


namespace abs_five_minus_sqrt_11_eq_l293_293493

noncomputable def sqrt_11 := Real.sqrt 11

theorem abs_five_minus_sqrt_11_eq : abs (5 - sqrt_11) = 1.683 := by
  have h_sqrt_11 : sqrt_11 = 3.317 := sorry
  calc
    abs (5 - sqrt_11)
        = abs (5 - 3.317) : by rw [h_sqrt_11]
    ... = abs (1.683) : by norm_num
    ... = 1.683 : abs_of_pos (by norm_num)

end abs_five_minus_sqrt_11_eq_l293_293493


namespace parallelogram_KONJ_l293_293541

variables {A B C K L M O H N J : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C]
variables [Inhabited K] [Inhabited L] [Inhabited M]
variables [Inhabited O] [Inhabited H] [Inhabited N] [Inhabited J]

def acute_triangle (ABC : Type) (O H : Type) := true

def inside_triangle (K : Type) (ABC : Type) := true

def parallelogram (P Q R S : Type) := true

def midpoint (J H K : Type) := true

def intersect (P Q R S T : Type) := true

theorem parallelogram_KONJ
    (ABC : Type) (O H : Type) (K : Type) (L : Type) (M : Type)
    (N : Type) (J : Type)
    [acute_triangle ABC O H] 
    [inside_triangle K ABC] 
    [parallelogram A K C L] 
    [parallelogram A K B M]
    [intersect B L C M N] 
    [midpoint J H K] : 
    parallelogram K O N J :=
sorry

end parallelogram_KONJ_l293_293541


namespace problem_statement_l293_293246

def f (x : ℝ) : ℝ := x^5 - x^3 + 1
def g (x : ℝ) : ℝ := x^2 - 2

theorem problem_statement (x1 x2 x3 x4 x5 : ℝ) 
  (h_roots : ∀ x, f x = 0 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5) :
  g x1 * g x2 * g x3 * g x4 * g x5 = -7 := 
sorry

end problem_statement_l293_293246


namespace percent_of_amount_l293_293380

theorem percent_of_amount (Part Whole : ℝ) (hPart : Part = 120) (hWhole : Whole = 80) :
  (Part / Whole) * 100 = 150 :=
by
  rw [hPart, hWhole]
  sorry

end percent_of_amount_l293_293380


namespace correct_ranking_l293_293992

-- Definitions for the colleagues
structure Colleague :=
  (name : String)
  (seniority : ℕ)

-- Colleagues: Julia, Kevin, Lana
def Julia := Colleague.mk "Julia" 1
def Kevin := Colleague.mk "Kevin" 0
def Lana := Colleague.mk "Lana" 2

-- Statements definitions
def Statement_I (c1 c2 c3 : Colleague) := c2.seniority < c1.seniority ∧ c1.seniority < c3.seniority 
def Statement_II (c1 c2 c3 : Colleague) := c1.seniority > c3.seniority
def Statement_III (c1 c2 c3 : Colleague) := c1.seniority ≠ c1.seniority

-- Exactly one of the statements is true
def Exactly_One_True (s1 s2 s3 : Prop) := (s1 ∨ s2 ∨ s3) ∧ ¬(s1 ∧ s2 ∨ s1 ∧ s3 ∨ s2 ∧ s3) ∧ ¬(s1 ∧ s2 ∧ s3)

-- The theorem to be proved
theorem correct_ranking :
  Exactly_One_True (Statement_I Kevin Lana Julia) (Statement_II Kevin Lana Julia) (Statement_III Kevin Lana Julia) →
  (Kevin.seniority < Lana.seniority ∧ Lana.seniority < Julia.seniority) := 
  by  sorry

end correct_ranking_l293_293992


namespace jack_jill_total_difference_l293_293295

theorem jack_jill_total_difference :
  let original_price := 90.00
  let discount_rate := 0.20
  let tax_rate := 0.06

  -- Jack's calculation
  let jack_total :=
    let price_with_tax := original_price * (1 + tax_rate)
    price_with_tax * (1 - discount_rate)
  
  -- Jill's calculation
  let jill_total :=
    let discounted_price := original_price * (1 - discount_rate)
    discounted_price * (1 + tax_rate)

  -- Equality check
  jack_total = jill_total := 
by
  -- Place the proof here
  sorry

end jack_jill_total_difference_l293_293295


namespace marcy_water_amount_l293_293259

theorem marcy_water_amount :
  ∀ (time_per_sip total_time minutes_per_liter sips_per_liter : ℕ),
    time_per_sip = 5 →
    total_time = 250 →
    minutes_per_liter = 1000 →
    sips_per_liter = 1 →
    ((total_time / time_per_sip) * 40) / minutes_per_liter * sips_per_liter = 2 :=
by
  intros time_per_sip total_time minutes_per_liter sips_per_liter
  assume h1 : time_per_sip = 5
  assume h2 : total_time = 250
  assume h3 : minutes_per_liter = 1000
  assume h4 : sips_per_liter = 1
  sorry

end marcy_water_amount_l293_293259


namespace shaded_area_and_sum_is_correct_l293_293467

def equilateral_triangle_side := 10
def radius := 5 -- derived as half of the side since diameter = side length
def sector_angle := 120
def triangle_area := 25 * Real.sqrt 3
def total_area := (50 / 3) * Real.pi - 25 * Real.sqrt 3
def sum_abc := (50 / 3 : ℚ) + 25 + 3

theorem shaded_area_and_sum_is_correct :
  ∃ (a b c : ℚ), total_area = a * Real.pi - b * Real.sqrt c ∧ a + b + c = 45 :=
by
  use (50 / 3 : ℚ), 25, 3
  simp [total_area, Real.sqrt, Real.pi, sum_abc]
  sorry

end shaded_area_and_sum_is_correct_l293_293467


namespace find_five_digit_number_l293_293384

def reverse_number (n : ℕ) : ℕ :=
  let digits := n.digits 10 in
  digits.reverse.to_nat

theorem find_five_digit_number :
  ∃ n : ℕ, (9999 < n ∧ n < 100000) ∧ (9 * n = reverse_number n) ∧ n = 10989 :=
by
  sorry

end find_five_digit_number_l293_293384


namespace count_lines_in_equation_number_of_lines_l293_293287

theorem count_lines_in_equation (x y : ℝ) : 
  (x^4 = x^2 * y^2) → (x = 0 ∨ y = x ∨ y = -x) :=
by
  sorry

theorem number_of_lines : ∃ n, n = 3 ∧ 
  ∀ (x y : ℝ), (x^4 = x^2 * y^2) → (x = 0 ∨ y = x ∨ y = -x) :=
by
  use 3
  split
  . rfl
  . exact count_lines_in_equation

end count_lines_in_equation_number_of_lines_l293_293287


namespace problem_l293_293720

theorem problem (n : ℕ) (a : Fin n → ℝ) (k : ℝ)
  (h1 : ∀ i, 0 < a i)
  (h_sum : (∑ i, a i) = 3 * k)
  (h_sum_sq : (∑ i, (a i)^2) = 3 * k^2)
  (h_sum_cub : (∑ i, (a i)^3) > 3 * k^3 + k) :
  ∃ i j, 0 ≤ i < n → 0 ≤ j < n → |a i - a j| > 1 :=
by 
  sorry

end problem_l293_293720


namespace integral_f_l293_293926

noncomputable def f : ℝ → ℝ :=
  λ x, if h : -1 ≤ x ∧ x ≤ 0 then (x + 1) ^ 2 else if 0 < x ∧ x ≤ 1 then real.sqrt (1 - x ^ 2) else 0

theorem integral_f :
  ∫ x in -1..1, f x = (4 + 3 * real.pi) / 12 :=
begin
  sorry
end

end integral_f_l293_293926


namespace find_heaviest_and_lightest_l293_293323

-- Definition of the main problem conditions
def coins : ℕ := 10
def max_weighings : ℕ := 13
def distinct_weights (c : ℕ) : Prop := ∀ (i j : ℕ), i ≠ j → i < c → j < c → weight i ≠ weight j

-- Noncomputed property representing the weight of each coin
noncomputable def weight : ℕ → ℝ := sorry

-- The main theorem statement
theorem find_heaviest_and_lightest (c : ℕ) (mw : ℕ) (dw : distinct_weights c) : c = coins ∧ mw = max_weighings
  → ∃ (h l : ℕ), h < c ∧ l < c ∧ (∀ (i : ℕ), i < c → weight i ≤ weight h ∧ weight i ≥ weight l) :=
by
  sorry

end find_heaviest_and_lightest_l293_293323


namespace find_b_collinear_and_bisects_l293_293649

def a := (5 : ℤ, -3 : ℤ, -6 : ℤ)
def c := (-3 : ℤ, -2 : ℤ, 3 : ℤ)
def b := (1 : ℚ, -12/5 : ℚ, 3/5 : ℚ)

def collinear (a b c : α × α × α) [CommRing α] : Prop :=
  ∃ k : α, b = (a.1 + k * (c.1 - a.1), a.2 + k * (c.2 - a.2), a.3 + k * (c.3 - a.3))

def bisects_angle (a b c : ℚ × ℚ × ℚ) : Prop :=
  let dot_product (x y : ℚ × ℚ × ℚ) := x.1 * y.1 + x.2 * y.2 + x.3 * y.3
  let norm (x : ℚ × ℚ × ℚ) := real.sqrt (dot_product x x)
  (dot_product a b) / (norm a * norm b) = (dot_product b c) / (norm b * norm c)

theorem find_b_collinear_and_bisects :
  collinear a b c ∧ bisects_angle a b c :=
by
  sorry

end find_b_collinear_and_bisects_l293_293649


namespace percentage_error_in_area_l293_293400

noncomputable def side_with_error (s : ℝ) : ℝ := 1.04 * s

noncomputable def actual_area (s : ℝ) : ℝ := s ^ 2

noncomputable def calculated_area (s : ℝ) : ℝ := (side_with_error s) ^ 2

noncomputable def percentage_error (actual : ℝ) (calculated : ℝ) : ℝ :=
  ((calculated - actual) / actual) * 100

theorem percentage_error_in_area (s : ℝ) :
  percentage_error (actual_area s) (calculated_area s) = 8.16 := by
  sorry

end percentage_error_in_area_l293_293400


namespace triangle_DEF_area_l293_293809

theorem triangle_DEF_area (a b c : ℕ) (t_a t_b t_c : ℕ) (Q : Type) (area_ta area_tb area_tc : ℕ)
  (h1 : t_a = 16) (h2 : t_b = 25) (h3 : t_c = 36) :
  a * b - c * Q.area DEF = 225 :=
by
  sorry

end triangle_DEF_area_l293_293809


namespace actual_selling_price_l293_293417

-- Define the original price m
variable (m : ℝ)

-- Define the discount rate
def discount_rate : ℝ := 0.2

-- Define the selling price
def selling_price := m * (1 - discount_rate)

-- The theorem states the relationship between the original price and the selling price after discount
theorem actual_selling_price : selling_price m = 0.8 * m :=
by
-- Proof step would go here
sorry

end actual_selling_price_l293_293417


namespace inscribed_sphere_radius_l293_293813

-- Define the problem statement
theorem inscribed_sphere_radius :
  let side_length : ℝ := 1
  let sqrt2 := real.sqrt 2
  let sqrt3 := real.sqrt 3
  let volume := (1 / 3) * (1 / 2) * side_length * side_length * (sqrt2 / 2)
  let area := 2 * (1 / 2) * side_length * side_length + 2 * (sqrt3 / 4)
  let r := sqrt2 - (real.sqrt 6 / 2)
  volume = (1 / 3) * area * r :=
sorry

end inscribed_sphere_radius_l293_293813


namespace exists_rhombus_with_given_side_and_diag_sum_l293_293020

-- Define the context of the problem
variables (a s : ℝ)

-- Necessary definitions for a rhombus
structure Rhombus (side diag_sum : ℝ) :=
  (side_length : ℝ)
  (diag_sum : ℝ)
  (d1 d2 : ℝ)
  (side_length_eq : side_length = side)
  (diag_sum_eq : d1 + d2 = diag_sum)
  (a_squared : 2 * (side_length)^2 = d1^2 + d2^2)

-- The proof problem
theorem exists_rhombus_with_given_side_and_diag_sum (a s : ℝ) : 
  ∃ (r : Rhombus a (2*s)), r.side_length = a ∧ r.diag_sum = 2 * s :=
by
  sorry

end exists_rhombus_with_given_side_and_diag_sum_l293_293020


namespace find_m_monotonicity_f_0_2_monotonicity_f_2_infty_range_a_l293_293564

def f (x : ℝ) (m : ℝ) : ℝ := x + m / x

theorem find_m (m : ℝ) (h : f 1 m = 5) : m = 4 :=
by
  sorry

theorem monotonicity_f_0_2 (x : ℝ) (h1 : 0 < x) (h2 : x < 2) : 
  f x 4 ≥ f x 4 :=
by
  sorry -- No proof required for monotonicity

theorem monotonicity_f_2_infty (x : ℝ) (h1 : 2 ≤ x) : 
  f x 4 ≤ f x 4 :=
by
  sorry -- No proof required for monotonicity

theorem range_a (a : ℝ) (h : ∀ x > 0, x^2 + 4 ≥ a * x) : a ≤ 4 :=
by
  sorry

end find_m_monotonicity_f_0_2_monotonicity_f_2_infty_range_a_l293_293564


namespace correct_propositions_l293_293920

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

end correct_propositions_l293_293920


namespace identify_heaviest_and_lightest_l293_293331

   def coin : Type := ℕ  -- let's represent coins as natural numbers for simplicity.

   def has_different_weights (coins : list coin) : Prop := 
     ∀ (c1 c2 : coin), c1 ∈ coins → c2 ∈ coins → c1 ≠ c2 → weight(c1) ≠ weight(c2)

   def weight : coin → ℝ := -- assume a function that gives the weight corresponding to a coin.
     sorry 

   theorem identify_heaviest_and_lightest (coins : list coin) 
     (h₁ : length coins = 10)
     (h₂ : has_different_weights coins) : 
     ∃ (heaviest lightest : coin), 
       (heaviest ∈ coins) ∧ (lightest ∈ coins) ∧
       (∀ c ∈ coins, weight c ≤ weight heaviest) ∧
       (∀ c ∈ coins, weight c ≥ weight lightest) :=
   by 
     sorry
   
end identify_heaviest_and_lightest_l293_293331


namespace parallel_vectors_l293_293570

def vector (F : Type*) := (F × F)

theorem parallel_vectors (n : ℤ) (a b c : vector ℤ)
  (h_a : a = (n, -1))
  (h_b : b = (-1, 1))
  (h_c : c = (-1, 2))
  (h_par : ∃ k : ℤ, (fst (a.1 + b.1)) = k * (fst c.1) ∧ (snd (a.2 + b.2)) = k * (snd c.2)) :
  n = 1 :=
by
  sorry

end parallel_vectors_l293_293570


namespace quadratic_equation_solutions_l293_293726

theorem quadratic_equation_solutions (x : ℝ) : x * (x - 7) = 0 ↔ x = 0 ∨ x = 7 :=
by
  sorry

end quadratic_equation_solutions_l293_293726


namespace first_number_percentage_of_second_l293_293599

theorem first_number_percentage_of_second (X : ℝ) (h1 : First = 0.06 * X) (h2 : Second = 0.18 * X) : 
  (First / Second) * 100 = 33.33 := 
by 
  sorry

end first_number_percentage_of_second_l293_293599


namespace sector_area_l293_293595

theorem sector_area (θ c : ℝ) (hθ : θ = 2) (hc : c = 2) : 
  let r := 1 / Real.sin(1) in 
  1 / (Real.sin 1) ^ 2 = 1 / (Real.sin 1) ^ 2 :=
by
  -- Given conditions: central angle θ = 2 and chord length c = 2
  have hθ : θ = 2 := hθ
  have hc : c = 2 := hc
  -- Radius formula: c = 2r * sin(θ/2)
  let r := 1 / Real.sin(1)
  -- Area formula: A = (1/2) * r^2 * θ
  sorry

end sector_area_l293_293595


namespace streetlight_comb_l293_293197

/-
  We have 12 streetlights, out of which 3 can be turned off with the following rules:
  1. The first and last lights cannot be turned off.
  2. No two adjacent lights can be turned off.

  We must prove the number of ways to turn off 3 lights is 56.
-/

theorem streetlight_comb: 
  ∃ (n k: ℕ), n = 8 ∧ k = 3 ∧ (n.choose k) = 56 :=
begin
  use [8, 3],
  split, { refl },
  split, { refl },
  apply nat.choose_eq,
  norm_num,
end

end streetlight_comb_l293_293197


namespace min_cos_C_l293_293959

variable {a b c : ℝ}
variable {A B C : ℝ}
variable {triangle_ABC : (0 < A ∧ A < π) ∧ (0 < B ∧ B < π) ∧ (0 < C ∧ C < π) ∧ (A + B + C = π)}
variable {sides_ABC : a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2}
variable {cos_rule_A : cos A = (b^2 + c^2 - a^2) / (2 * b * c)}
variable {cos_rule_B : cos B = (a^2 + c^2 - b^2) / (2 * a * c)}

def given_equation (a b A B : ℝ) : Prop :=
  a * (4 - 2 * sqrt 7 * cos B) = b * (2 * sqrt 7 * cos A - 5)

theorem min_cos_C (h1 : given_equation a b A B) : 
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  in cos_C >= -1/2 :=
  sorry

end min_cos_C_l293_293959


namespace power_mod_equivalence_l293_293770

theorem power_mod_equivalence : (7^700) % 100 = 1 := 
by 
  -- Given that (7^4) % 100 = 1
  have h : 7^4 % 100 = 1 := by sorry
  -- Use this equivalence to prove the statement
  sorry

end power_mod_equivalence_l293_293770


namespace evaluate_ceiling_sums_l293_293034

theorem evaluate_ceiling_sums : 
  (⌈real.sqrt 3⌉ + ⌈real.sqrt 33⌉ + ⌈real.sqrt 333⌉) = 27 :=
by
  have h1 : 1 < real.sqrt 3 ∧ real.sqrt 3 < 2 :=
    ⟨by norm_num, by norm_num⟩,
  have h2 : 5 < real.sqrt 33 ∧ real.sqrt 33 < 6 :=
    ⟨by norm_num, by norm_num⟩,
  have h3 : 18 < real.sqrt 333 ∧ real.sqrt 333 < 19 :=
    ⟨by norm_num, by norm_num⟩,
  sorry

end evaluate_ceiling_sums_l293_293034


namespace no_arrangement_of_circle_l293_293292

theorem no_arrangement_of_circle :
  ¬ ∃ (f : ℕ → ℕ), (∀ n, 1 ≤ f n ∧ f n ≤ 100) ∧ (∀ n, 30 ≤ |f n - f (n + 1) % 100| ∧ |f n - f (n + 1) % 100| ≤ 50) :=
sorry

end no_arrangement_of_circle_l293_293292


namespace monotonic_increasing_interval_l293_293716

noncomputable def f (x : ℝ) : ℝ := log (1 / 2) (x^2 - 4)

theorem monotonic_increasing_interval :
  ∀ x₁ x₂ : ℝ, x₁ ∈ Iio (-2) → x₂ ∈ Iio (-2) → x₁ < x₂ → f x₁ < f x₂ :=
by
  -- proof could be added here
  sorry

end monotonic_increasing_interval_l293_293716


namespace travel_days_l293_293434

variable (a b d : ℕ)

theorem travel_days (h1 : a + d = 11) (h2 : b + d = 21) (h3 : a + b = 12) : a + b + d = 22 :=
by sorry

end travel_days_l293_293434


namespace part1_eq1_part1_eq2_values_of_a_b_monotonicity_on_neg_min_value_l293_293562

def f (a b x : ℝ) := 2^x + 2^(a * x + b)

section
variables {a b x : ℝ}

theorem part1_eq1 (h1 : f a b 1 = 5 / 2) : a + b = -1 :=
sorry

theorem part1_eq2 (h2 : f a b 2 = 17 / 4) : 2 * a + b = -2 :=
sorry

theorem values_of_a_b (h1 : f a b 1 = 5 / 2) (h2 : f a b 2 = 17 / 4) : a = -1 ∧ b = 0 :=
sorry

theorem monotonicity_on_neg (a b : ℝ) (h1 : f a b 1 = 5 / 2) (h2 : f a b 2 = 17 / 4) :
  ∀ x1 x2, x1 < x2 → x2 ≤ 0 → f a b x1 > f a b x2 :=
sorry

theorem min_value (a b : ℝ) (h1 : f a b 1 = 5 / 2) (h2 : f a b 2 = 17 / 4) : ∃ x, f a b x = 2 :=
sorry
end

end part1_eq1_part1_eq2_values_of_a_b_monotonicity_on_neg_min_value_l293_293562


namespace complex_difference_l293_293556

-- Define the complex modulus condition
def modulus_one (z : ℂ) : Prop := complex.abs z = 1

-- Define maximum and minimum conditions
def z_max (z : ℂ) : Prop := ∀ w : ℂ, complex.abs w = 1 → complex.abs (z + 1 + complex.I) ≥ complex.abs (w + 1 + complex.I)
def z_min (z : ℂ) : Prop := ∀ w : ℂ, complex.abs w = 1 → complex.abs (z + 1 + complex.I) ≤ complex.abs (w + 1 + complex.I)

theorem complex_difference (z₁ z₂ : ℂ) (h₁ : modulus_one z₁) (h₂ : modulus_one z₂) (h₃ : z_max z₁) (h₄ : z_min z₂) :
  z₁ - z₂ = complex.sqrt 2 + complex.sqrt 2 * complex.I :=
sorry

end complex_difference_l293_293556


namespace identify_heaviest_and_lightest_coin_in_13_weighings_l293_293309

theorem identify_heaviest_and_lightest_coin_in_13_weighings :
  ∀ (coins : Finₓ 10 → ℝ) 
    (balance_weighing : ∀ (a b : Finₓ 10), Prop), 
    (∀ i j, coins i ≠ coins j) → 
    (∃ strategy : ℕ → (Finₓ 10 × Finₓ 10),
      ∃ h : ℕ,
        h ≤ 13 ∧
        (∃ heaviest lightest : Finₓ 10,
          (∀ i, coins heaviest ≥ coins i) ∧ (∀ j, coins lightest ≤ coins j))) :=
by
  sorry

end identify_heaviest_and_lightest_coin_in_13_weighings_l293_293309


namespace find_R_l293_293906

theorem find_R (a b : ℝ) (Q R : ℝ) (hQ : Q = 4)
  (h1 : 1/a + 1/b = Q/(a + b))
  (h2 : a/b + b/a = R) : R = 2 :=
by
  sorry

end find_R_l293_293906


namespace min_value_reciprocal_sum_l293_293902

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hmean : (a + b) / 2 = 1 / 2) : 
  ∃ c, c = (1 / a + 1 / b) ∧ c ≥ 4 := 
sorry

end min_value_reciprocal_sum_l293_293902


namespace jay_change_l293_293983

theorem jay_change (book_price pen_price ruler_price payment : ℕ) (h1 : book_price = 25) (h2 : pen_price = 4) (h3 : ruler_price = 1) (h4 : payment = 50) : 
(book_price + pen_price + ruler_price ≤ payment) → (payment - (book_price + pen_price + ruler_price) = 20) :=
by
  intro h
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end jay_change_l293_293983


namespace no_solution_eqn_l293_293071

theorem no_solution_eqn (a : ℝ) : 
  (∀ x : ℝ, 8 * |x - 4 * a| + |x - a^2| + 7 * x - 3 * a ≠ 0) ↔ 
  a ∈ (-∞, -21) ∪ 0 ∪ (0, ∞) :=
begin
  sorry
end

end no_solution_eqn_l293_293071


namespace true_inverse_D_is_only_l293_293388

-- Define the propositions
def prop_A (a b : ℝ) : Prop := a = b → a^2 = b^2
def inverse_A (a b : ℝ) : Prop := a^2 = b^2 → a = b

def prop_B (a b : ℝ) : Prop := a = b → |a| = |b|
def inverse_B (a b : ℝ) : Prop := |a| = |b| → a = b

def prop_C (a b : ℝ) : Prop := a = 0 → a * b = 0
def inverse_C (a b : ℝ) : Prop := a * b = 0 → a = 0

def prop_D := ∀ {Δ1 Δ2 : TriangleShape}, congruent Δ1 Δ2 → (∀ s ∈ Δ1.sides, Δ2.sides.contains s)
def inverse_D := ∀ {Δ1 Δ2 : TriangleShape}, (∀ s ∈ Δ1.sides, Δ2.sides.contains s) → congruent Δ1 Δ2

-- The corresponding proof problem
theorem true_inverse_D_is_only :
  (∀ a b : ℝ, ¬ inverse_A a b) ∧
  (∀ a b : ℝ, ¬ inverse_B a b) ∧
  (∀ a b : ℝ, ¬ inverse_C a b) ∧
  inverse_D :=
by {
  sorry
}

end true_inverse_D_is_only_l293_293388


namespace b_alone_days_l293_293779

-- Given definitions
def combined_work_rate (a_rate b_rate : ℕ → ℕ) (days : ℕ) := 1 / (a_rate days + b_rate days) 
def days_worked_by_a (a_rate : ℕ → ℕ) := 15
def days_worked_by_ab (a_rate b_rate: ℕ → ℕ) := 10

-- Define the function for work rate of A
def work_rate_a := λ days : ℕ, 1 / days

-- Theorem statement to prove
theorem b_alone_days (b_will_complete : ℕ) : 
    let b_rate := λ days : ℕ, 1 / days in
    combined_work_rate work_rate_a b_rate 10 = 1 / 10 ∧
    work_rate_a 15 = 1 / 15 →
    b_will_complete = 30 :=
by
  sorry

end b_alone_days_l293_293779


namespace find_min_value_of_f_in_interval_l293_293928

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := sin (2 * x) * cos φ + cos (2 * x) * sin φ

theorem find_min_value_of_f_in_interval (φ : ℝ) (h1 : abs φ < π / 2) :
  (∀ x ∈ set.Icc (0 : ℝ) (π / 2), f x φ ≥ -sqrt 3 / 2) ∧ 
  (∃ x ∈ set.Icc (0 : ℝ) (π / 2), f x φ = -sqrt 3 / 2) :=
by
  sorry

end find_min_value_of_f_in_interval_l293_293928


namespace jay_change_l293_293987

def cost_book : ℝ := 25
def cost_pen : ℝ := 4
def cost_ruler : ℝ := 1
def payment : ℝ := 50

theorem jay_change : (payment - (cost_book + cost_pen + cost_ruler) = 20) := sorry

end jay_change_l293_293987


namespace jane_distance_l293_293638

def distance_from_start (east1 : ℝ) (angle : ℝ) (hypotenuse : ℝ) : ℝ :=
  let east2 := hypotenuse * Math.cos angle
  let north := hypotenuse * Math.sin angle
  Math.sqrt ((east1 + east2)^2 + north^2)

theorem jane_distance :
  distance_from_start 3 (Real.pi / 4) 8 = Math.sqrt 73 := by
  sorry

end jane_distance_l293_293638


namespace triangle_area_l293_293956

theorem triangle_area (k x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 2*x) (h3 : k*x - y + 1 ≥ 0)
    (h_triangle : ∃ p1 p2 p3 : ℝ × ℝ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p1 ∧
        (collinear p1 p2 ∧ collinear p2 p3 ∧ collinear p3 p1) ∧ 
        (∠p1p2p3 = π/2 ∨ ∠p2p3p1 = π/2 ∨ ∠p3p1p2 = π/2)) :
    (∃ a : ℝ, a = 1/5 ∨ a = 1/4) :=
by
    sorry

end triangle_area_l293_293956


namespace not_directly_nor_inversely_proportional_l293_293029

theorem not_directly_nor_inversely_proportional :
  ∀ (x y : ℝ),
    ((2 * x + y = 5) ∨ (2 * x + 3 * y = 12)) ∧
    ((¬ (∃ k : ℝ, x = k * y)) ∧ (¬ (∃ k : ℝ, x * y = k))) := sorry

end not_directly_nor_inversely_proportional_l293_293029


namespace determine_a_if_f_is_odd_l293_293147

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def my_function (f : ℝ → ℝ) (a : ℝ) : Prop :=
f = λ x, 2^x - 2^(-x) * log a (2)

theorem determine_a_if_f_is_odd :
  (∀ f : ℝ → ℝ, is_odd_function f ∧ my_function f a) → a = 10 :=
by
  sorry

end determine_a_if_f_is_odd_l293_293147


namespace mean_of_three_added_numbers_l293_293706

theorem mean_of_three_added_numbers (x y z : ℝ) :
  (∀ (s : ℝ), (s / 7 = 75) → (s + x + y + z) / 10 = 90) → (x + y + z) / 3 = 125 :=
by
  intro h
  sorry

end mean_of_three_added_numbers_l293_293706


namespace angle_C_and_area_l293_293215

variables {A B C : ℝ} {a b c : ℝ}

theorem angle_C_and_area (h1 : c * Real.cos B + (b - 2 * a) * Real.cos C = 0)
                        (h2 : c = 2)
                        (h3 : a + b = a * b) :
  (C = π / 3) ∧ (1 / 2 * a * b * Real.sin (π / 3)) = sqrt 3 := by
sorry

end angle_C_and_area_l293_293215


namespace reflect_P_across_x_axis_l293_293622

def point_reflection_over_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem reflect_P_across_x_axis : 
  point_reflection_over_x_axis (-3, 1) = (-3, -1) :=
  by
    sorry

end reflect_P_across_x_axis_l293_293622


namespace pyaterochka_store_placement_economics_l293_293262

theorem pyaterochka_store_placement_economics :
  ∀ (location : Type) (strategy : location → Prop),
    (∃ (reasons : list (location → Prop)), 
      reasons = [ 
        (λ loc : location, strategy loc → "barrier to entry for competitors"),
        (λ loc : location, strategy loc → "reduction in transportation costs"),
        (λ loc : location, strategy loc → "franchise strategy leading to optimal individual placement")
        ] ∧
      ∀ reason ∈ reasons, strategy (arbitrary location) → reason (arbitrary location)
    ) ∧ 
    (∃ (drawbacks : list (location → Prop)),
      drawbacks = [
        (λ loc : location, strategy loc → "risk of cannibalization of sales between stores"),
        (λ loc : location, strategy loc → "legal repercussions due to anti-monopoly laws")
        ] ∧
      ∀ drawback ∈ drawbacks, strategy (arbitrary location) → drawback (arbitrary location)
    ) :=
sorry

end pyaterochka_store_placement_economics_l293_293262


namespace problem_statement_l293_293340

noncomputable def perimeter_rectangle 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (area_triangle : ℝ := (1/2) * a * b) 
  (area_rectangle : ℝ := area_triangle) 
  (l : ℝ := area_rectangle / w) : ℝ :=
2 * (w + l)

theorem problem_statement 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h_a : a = 9) 
  (h_b : b = 12) 
  (h_c : c = 15) 
  (h_w : w = 6) : 
  perimeter_rectangle a b c w h1 = 30 :=
by 
  sorry

end problem_statement_l293_293340


namespace can_reach_2021_and_2021_cannot_reach_2022_and_2022_l293_293260

-- Define the allowed action in Lean
def allowed_action (a b : ℕ) : (ℕ × ℕ) :=
  (a + (Nat.digitSum b), b + (Nat.digitSum a))

-- Define the starting and target states
def initial_state : ℕ × ℕ := (1, 2)
def target_state_a : ℕ × ℕ := (2021, 2021)
def target_state_b : ℕ × ℕ := (2022, 2022)

-- Function to check if a sequence of actions can reach the target state
def can_reach_target (initial target : ℕ × ℕ) : Prop :=
  ∃ steps : List (ℕ × ℕ), 
    steps.head? = some initial ∧
    steps.getLast? = some target ∧
    ∀ s t, (s, t) ∈ List.zip steps (List.tail steps) → t = allowed_action s.fst s.snd

-- Problem (a): Can we turn 1 and 2 into 2021 and 2021?
theorem can_reach_2021_and_2021 : can_reach_target initial_state target_state_a := sorry

-- Problem (b): Can we turn 1 and 2 into 2022 and 2022?
theorem cannot_reach_2022_and_2022 : ¬ can_reach_target initial_state target_state_b := sorry

end can_reach_2021_and_2021_cannot_reach_2022_and_2022_l293_293260


namespace probability_diff_by_one_l293_293522

open Classical

theorem probability_diff_by_one : 
  let s := {1, 2, 3, 4, 5}.powerset.filter (λ t, t.card = 2) in
  let pairs := (s.filter (λ t, (t.toList.nth 1).iget - (t.toList.head).iget = 1 ∨ (t.toList.head).iget - (t.toList.nth 1).iget = 1)) in
  (pairs.card : ℝ) / (s.card : ℝ) = 2 / 5 :=
by
  sorry

end probability_diff_by_one_l293_293522


namespace monthly_salary_l293_293427

variable (S : ℝ)
variable (Saves : ℝ)
variable (NewSaves : ℝ)

open Real

theorem monthly_salary (h1 : Saves = 0.30 * S)
                       (h2 : NewSaves = Saves - 0.25 * Saves)
                       (h3 : NewSaves = 400) :
    S = 1777.78 := by
    sorry

end monthly_salary_l293_293427


namespace range_sum_six_l293_293955

noncomputable def f (x : ℝ) : ℝ := 3 + (2^x - 1) / (2^x + 1) + sin (2 * x)

theorem range_sum_six (k : ℝ) (h : k > 0) (m n : ℝ) :
  ∃ m n, (∀ x ∈ [-k, k], f(x) ∈ [m, n]) ∧ (m + n = 6) := 
sorry

end range_sum_six_l293_293955


namespace percentage_value_l293_293418

variables {P a b c : ℝ}

theorem percentage_value (h1 : (P / 100) * a = 12) (h2 : (12 / 100) * b = 6) (h3 : c = b / a) : c = P / 24 :=
by
  sorry

end percentage_value_l293_293418


namespace passengers_with_round_trip_tickets_l293_293403

theorem passengers_with_round_trip_tickets (P R : ℝ) : 
  (0.40 * R = 0.25 * P) → (R / P = 0.625) :=
by
  intro h
  sorry

end passengers_with_round_trip_tickets_l293_293403


namespace find_abs_w_l293_293252

noncomputable def complex_problem (z w : ℂ) (h1 : complex.abs (3 * z - w) = 12) 
                                  (h2 : complex.abs (z + 3 * w) = 9) 
                                  (h3 : complex.abs (z - w) = 7) : ℝ :=
|w|

theorem find_abs_w (z w : ℂ) (h1 : complex.abs (3 * z - w) = 12) 
                   (h2 : complex.abs (z + 3 * w) = 9) 
                   (h3 : complex.abs (z - w) = 7) : complex_problem z w h1 h2 h3 = 6.0625 :=
by 
  -- proof will go here 
  sorry

end find_abs_w_l293_293252


namespace largest_common_term_in_sequences_l293_293464

/-- An arithmetic sequence starts with 3 and has a common difference of 10. A second sequence starts
with 5 and has a common difference of 8. In the range of 1 to 150, the largest number common to 
both sequences is 133. -/
theorem largest_common_term_in_sequences : ∃ (b : ℕ), b < 150 ∧ (∃ (n m : ℤ), b = 3 + 10 * n ∧ b = 5 + 8 * m) ∧ (b = 133) := 
by
  sorry

end largest_common_term_in_sequences_l293_293464


namespace min_value_expression_l293_293526

theorem min_value_expression (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2) :
  (∃ θ_min : ℝ, θ_min = π / 6 ∧ 1 / sin θ_min + 3 * sqrt 3 / cos θ_min = 8) := 
sorry

end min_value_expression_l293_293526


namespace collinear_centers_exists_l293_293756

-- Define the set of coins
def coins := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10 : ℕ}

-- State the theorem to be proved
theorem collinear_centers_exists : 
  ∃ (a b x y : ℕ), a ∈ coins ∧ b ∈ coins ∧ x ∈ coins ∧ y ∈ coins ∧ 
  (b - a) * (a * x + a * y + 2 * x * y) = a * (a + b) * (2 * a + x + y) :=
by 
  sorry

end collinear_centers_exists_l293_293756


namespace inequality_solution_l293_293500

theorem inequality_solution :
  ∀ x : ℝ, (5 / 24 + |x - 11 / 48| < 5 / 16 ↔ (1 / 8 < x ∧ x < 1 / 3)) :=
by
  intro x
  sorry

end inequality_solution_l293_293500


namespace xiaomings_mother_money_l293_293777

-- Definitions for the conditions
def price_A : ℕ := 6
def price_B : ℕ := 9
def units_more_A := 2

-- Main statement to prove
theorem xiaomings_mother_money (x : ℕ) (M : ℕ) :
  M = 6 * x ∧ M = 9 * (x - 2) → M = 36 :=
by
  -- Assuming the conditions are given
  rintro ⟨hA, hB⟩
  -- The proof is omitted
  sorry

end xiaomings_mother_money_l293_293777


namespace square_plot_area_l293_293381

theorem square_plot_area (s : ℕ) 
  (cost_per_foot : ℕ) 
  (total_cost : ℕ) 
  (H1 : cost_per_foot = 58) 
  (H2 : total_cost = 1624) 
  (H3 : total_cost = 232 * s) : 
  s * s = 49 := 
  by sorry

end square_plot_area_l293_293381


namespace sum_of_possible_values_of_b_l293_293837

theorem sum_of_possible_values_of_b :
  ( ∀ x : ℝ, 6 * x^2 + b * x + 12 * x + 18 = 0 → 
  solveDiscriminant (6 * x^2 + (b+12) * x + 18 = 0) = 0 ) →
  (∀ sum_b : ℝ, (b^2 + 24 * b - 288 = 0) → sum_b = -24) :=
  sorry

end sum_of_possible_values_of_b_l293_293837


namespace find_matrix_N_l293_293076

theorem find_matrix_N :
  ∃ N : matrix (fin 2) (fin 2) ℚ, N * !![2, -3; 5, -1] = !![-30, -9; 11, 1] ∧ N = !![5, -8; 7/13, 35/13] :=
by
  sorry

end find_matrix_N_l293_293076


namespace parallel_resistors_l293_293201
noncomputable def resistance_R (x y z w : ℝ) : ℝ :=
  1 / (1/x + 1/y + 1/z + 1/w)

theorem parallel_resistors :
  resistance_R 5 7 3 9 = 315 / 248 :=
by
  sorry

end parallel_resistors_l293_293201


namespace number_composition_l293_293793

theorem number_composition (n : ℕ) (h : n = 50900300) : 
  ∃ tens thousands hundreds : ℕ, 
    (tens = 5 ∧ thousands = 9 ∧ hundreds = 3) ∧ 
    (n = 5090 * 10000 + 300 * 1) :=
by
  use 5, 9, 3
  split
  . split 
    . exact rfl 
    . exact rfl 
  . exact rfl
  sorry

end number_composition_l293_293793


namespace sequence_term_formula_sequence_sum_formula_l293_293936

noncomputable def s (n : ℕ) : ℚ := (1/2 : ℚ) * n^2 + (1/2 : ℚ) * n

def a (n : ℕ) : ℕ := n

noncomputable def T (n : ℕ) : ℚ := (1/2 : ℚ) * (1 + (1 : ℚ) / 2 - (1 : ℚ) / (n + 1) - (1 : ℚ) / (n + 2))

theorem sequence_term_formula (n : ℕ) (h : 0 < n) : 
  (a n = s n - s (n-1)) :=
by
  sorry

theorem sequence_sum_formula (n : ℕ) (h : 0 < n) :
  (\sum i in range n, 1 / ((a i) * (a (i+2)))) = T n :=
by
  sorry

end sequence_term_formula_sequence_sum_formula_l293_293936


namespace cats_sold_l293_293808

theorem cats_sold (original_siamese : ℕ) (original_house : ℕ) (cats_left : ℕ) (total_cats := original_siamese + original_house) :
  original_siamese = 13 → original_house = 5 → cats_left = 8 → total_cats - cats_left = 10 :=
by
  intros h1 h2 h3
  rw [h1, h2] at total_cats
  norm_num at total_cats
  rw [total_cats, h3]
  norm_num
  sorry

end cats_sold_l293_293808


namespace cargo_transport_possible_l293_293798

theorem cargo_transport_possible 
  (total_cargo_weight : ℝ) 
  (weight_limit_per_box : ℝ) 
  (number_of_trucks : ℕ) 
  (max_load_per_truck : ℝ)
  (h1 : total_cargo_weight = 13.5)
  (h2 : weight_limit_per_box = 0.35)
  (h3 : number_of_trucks = 11)
  (h4 : max_load_per_truck = 1.5) :
  ∃ (n : ℕ), n ≤ number_of_trucks ∧ (total_cargo_weight / max_load_per_truck) ≤ n :=
by
  sorry

end cargo_transport_possible_l293_293798


namespace bananas_per_friend_l293_293639

theorem bananas_per_friend (bananas : ℕ) (friends : ℕ) (h_bananas : bananas = 21) (h_friends : friends = 3) :
  bananas / friends = 7 :=
by {
  rw [h_bananas, h_friends],
  norm_num,
  sorry
}

end bananas_per_friend_l293_293639


namespace find_a_b_find_range_k_l293_293954

-- Define the function f(x) and specify the conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (b - 2^x) / (a + 2^x)

-- Prove the values of a and b given f is odd and defined over ℝ
theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, f(-x, a, b) = -f(x, a, b)) → a = 1 ∧ b = 1 :=
sorry

-- Prove the range of k given the specified inequality
theorem find_range_k (k : ℝ) :
  (∀ t : ℝ, f(t^2 - 2 * t, 1, 1) < f(-2 * t^2 + k, 1, 1)) → k < -1/3 :=
sorry

end find_a_b_find_range_k_l293_293954


namespace part1_part2_l293_293130

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

theorem part1 (x : ℝ) (h : 0 < f(1 - 2 * x) - f(x) ∧ f(1 - 2 * x) - f(x) < 1) :
  -2/3 < x ∧ x < 1/3 :=
sorry

noncomputable def g (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then f x else f (2 - x)

theorem part2 (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) :
  ∃ y, g y = x ∧ 0 ≤ y ∧ y ≤ Real.log 2 ∧ y = 3 - 10 ^ x :=
sorry

end part1_part2_l293_293130


namespace smallest_prime_sum_of_three_different_primes_is_19_l293_293374

theorem smallest_prime_sum_of_three_different_primes_is_19 :
  ∃ (p : ℕ), Prime p ∧ p = 19 ∧ (∀ a b c : ℕ, a ≠ b → b ≠ c → a ≠ c → Prime a → Prime b → Prime c → a + b + c = p → p ≥ 19) :=
by
  sorry

end smallest_prime_sum_of_three_different_primes_is_19_l293_293374


namespace measure_YZX_l293_293010

-- Define the triangle ABC with given angles
axiom triangle_ABC (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) : Prop :=
  angle_A = 50 ∧ angle_B = 70 ∧ angle_C = 60 ∧ angle_A + angle_B + angle_C = 180

-- Define the points X, Y, Z on the sides of triangle ABC
axiom points_on_sides (X Y Z : Type) (B C A B' C' A' : Type) : Prop :=
  X ∈ (B' ∩ C') ∧ Y ∈ (A' ∩ B) ∧ Z ∈ (A ∩ C')

-- Define the circle Gamma as incircle and circumcircle of triangles
axiom circle_incircle_circumcircle (Gamma : Type) (triangle_ABC triangle_XYZ : Type) : Prop :=
  incircle Gamma triangle_ABC ∧ circumcircle Gamma triangle_XYZ

-- Define the measure of YZX angle
def angle_YZX (angle_B angle_C : ℝ) : ℝ :=
  180 - (angle_B / 2 + angle_C / 2)

-- The main theorem to prove
theorem measure_YZX : ∀ (A B C X Y Z : Type) (Gamma : Type),
    triangle_ABC A B C 50 70 60 →
    points_on_sides X Y Z B C A B C A →
    circle_incircle_circumcircle Gamma (triangle_ABC A B C) (triangle_XYZ X Y Z) →
    angle_YZX 70 60 = 115 := 
by 
  intros A B C X Y Z Gamma hABC hXYZ hGamma
  sorry

end measure_YZX_l293_293010


namespace find_f_of_f_neg4_l293_293924

def f (x : ℝ) : ℝ := 
  if x ≥ 0 then real.sqrt x 
  else (1 / 2) ^ (-x)

theorem find_f_of_f_neg4 : f (f (-4)) = 4 := 
by sorry

end find_f_of_f_neg4_l293_293924


namespace imaginary_part_of_z_l293_293182

theorem imaginary_part_of_z (z : ℂ) (h : (complex.I * z = 2)) : complex.imag z = -2 :=
sorry

end imaginary_part_of_z_l293_293182


namespace number_of_starting_positions_l293_293240

def hyperbola (x y : ℝ) : Prop :=
  y^2 - 4 * x^2 = 4

def line_through (p : ℝ × ℝ) (slope : ℝ) (x y : ℝ) : Prop :=
  (y - p.2) = slope * (x - p.1)

def projection (x : ℝ × ℝ) : ℝ × ℝ :=
  (0, x.2)

def next_point (P : ℝ × ℝ) : ℝ × ℝ :=
  let ⟨x, y⟩ := P in
  let intersect_x := (y + 2 * projection (x, y).2 - 4 * (y + 2 * projection (x, y).2) / 4) in
  projection (intersect_x, projection (intersect_x, projection (0, projection (intersect_x, (0, y))).2).2).2)

def sequence (P0 : ℝ × ℝ) (n : ℕ) : ℝ × ℝ :=
  Nat.rec_on n P0 (fun n' P' => next_point P')

theorem number_of_starting_positions : ∃(P0 : ℝ), sequence P0 1024 = sequence P0 0 ∧
  (set.range (λ k, k * (Real.pi / (2^1024 - 1)))).card = 2^1024 - 2 := sorry

end number_of_starting_positions_l293_293240


namespace line_through_point_with_equal_intercepts_l293_293509

-- Definition of the conditions
def point := (1 : ℝ, 2 : ℝ)
def eq_intercepts (line : ℝ → ℝ) := ∃ a b : ℝ, a = b ∧ (∀ x, line x = b - x * (b/a))

-- The proof statement
theorem line_through_point_with_equal_intercepts (line : ℝ → ℝ) : 
  (line 1 = 2 ∧ eq_intercepts line) → (line = (λ x, 2 * x) ∨ line = (λ x, 3 - x)) :=
by
  sorry

end line_through_point_with_equal_intercepts_l293_293509


namespace complex_roots_real_implies_12_real_z5_l293_293748

theorem complex_roots_real_implies_12_real_z5 :
  (∃ (z : ℂ) (h : z ^ 30 = 1), is_real (z ^ 5)) → (finset.card ((finset.filter (λ z, is_real (z ^ 5)) (finset.univ.filter (λ z, z ^ 30 = 1)))) = 12) := by
  sorry

end complex_roots_real_implies_12_real_z5_l293_293748


namespace belty_position_l293_293357

theorem belty_position :
  let letters := ['B', 'E', 'L', 'T', 'Y']
  let word := ['B', 'E', 'L', 'T', 'Y']
  (sorted_perm_index letters word) = 1 :=
by
  sorry

noncomputable def sorted_perm_index (letters : List Char) (word : List Char) : Nat :=
  sorry

end belty_position_l293_293357


namespace barbara_candies_left_l293_293825

def initial_candies: ℝ := 18.5
def candies_used_to_make_dessert: ℝ := 4.2
def candies_received_from_friend: ℝ := 6.8
def candies_eaten: ℝ := 2.7

theorem barbara_candies_left : 
  initial_candies - candies_used_to_make_dessert + candies_received_from_friend - candies_eaten = 18.4 := 
by
  sorry

end barbara_candies_left_l293_293825


namespace exists_hexagon_in_square_l293_293677

structure Point (α : Type*) :=
(x : α)
(y : α)

def is_in_square (p : Point ℕ) : Prop :=
p.x ≤ 4 ∧ p.y ≤ 4

def area_of_hexagon (vertices : List (Point ℕ)) : ℝ :=
-- placeholder for actual area calculation of a hexagon
sorry

theorem exists_hexagon_in_square : ∃ (p1 p2 : Point ℕ), 
  is_in_square p1 ∧ is_in_square p2 ∧ 
  area_of_hexagon [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 0⟩, ⟨4, 4⟩, p1, p2] = 6 :=
sorry

end exists_hexagon_in_square_l293_293677


namespace lines_intersect_l293_293488

-- Define the coefficients of the lines
def A1 : ℝ := 3
def B1 : ℝ := -2
def C1 : ℝ := 5

def A2 : ℝ := 1
def B2 : ℝ := 3
def C2 : ℝ := 10

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := A1 * x + B1 * y + C1 = 0
def line2 (x y : ℝ) : Prop := A2 * x + B2 * y + C2 = 0

-- Mathematical problem to prove
theorem lines_intersect : ∃ (x y : ℝ), line1 x y ∧ line2 x y :=
by
  sorry

end lines_intersect_l293_293488


namespace range_g_l293_293838

def g (x : ℝ) : ℝ := ⌊⌊x⌋ - x⌋

theorem range_g :
  let fx := ⌊x⌋ - x in
  fx ∈ Ioc (-1 : ℝ) (0 : ℝ) →
  set.range g = {-1, 0} :=
sorry

end range_g_l293_293838


namespace option_A_option_B_option_C_option_D_l293_293455

theorem option_A : (-(-1) : ℤ) ≠ -|(-1 : ℤ)| := by
  sorry

theorem option_B : ((-3)^2 : ℤ) ≠ -(3^2 : ℤ) := by
  sorry

theorem option_C : ((-4)^3 : ℤ) = -(4^3 : ℤ) := by
  sorry

theorem option_D : ((2^2 : ℚ)/3) ≠ ((2/3)^2 : ℚ) := by
  sorry

end option_A_option_B_option_C_option_D_l293_293455


namespace line_through_point_equal_intercepts_l293_293506

-- Definitions based on conditions
def passes_through (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l p.1 p.2

def equal_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a, a ≠ 0 ∧ (∀ x y, l x y ↔ x + y = a) ∨ (∀ x y, l x y ↔ y = 2 * x)

-- Theorem statement based on the problem
theorem line_through_point_equal_intercepts :
  ∃ l, passes_through (1, 2) l ∧ equal_intercepts l ∧
  (∀ x y, l x y ↔ 2 * x - y = 0) ∨ (∀ x y, l x y ↔ x + y - 3 = 0) :=
sorry

end line_through_point_equal_intercepts_l293_293506


namespace range_of_k_l293_293158

def is_ellipse (a b : ℝ) : Prop := a > 0 ∧ b > 0
def is_hyperbola (a b : ℝ) : Prop := a * b < 0

def p (k : ℝ) : Prop := is_ellipse (2 * k - 1) (k - 1)
def q (k: ℝ) : Prop := is_hyperbola (4 - k) (k - 3)

def pq_logic (k : ℝ) : Prop := (p k ∨ q k) ∧ ¬(p k ∧ q k)

theorem range_of_k (k : ℝ) : pq_logic k → (k ≤ 1 ∨ (3 ≤ k ∧ k ≤ 4)) :=
begin
  sorry
end

end range_of_k_l293_293158


namespace pauls_age_is_47_or_53_l293_293699

theorem pauls_age_is_47_or_53 (guesses : List ℕ) (age : ℕ) (h1 : guesses = [25, 29, 33, 35, 39, 42, 45, 46, 50, 52, 54])
(h2 : ∃ count_low, count_low ≥ 6 ∧ count_low = (guesses.filter (λ g, g < age)).length)
(h3 : ∃ count_off_by_one, count_off_by_one = (guesses.filter (λ g, abs (g - age) = 1)).length ∧ count_off_by_one = 3)
(h4 : Nat.Prime age) : age = 47 ∨ age = 53 := sorry

end pauls_age_is_47_or_53_l293_293699


namespace sum_of_roots_of_quadratic_l293_293378

theorem sum_of_roots_of_quadratic (a b c : ℕ) (h : a = 1 ∧ b = -10 ∧ c = 24)
  (h_eq : ∀ x, x^2 = 10 * x - 24 ↔ a * x^2 + b * x + c = 0) :
  -b = 10 := 
by 
  sorry

end sum_of_roots_of_quadratic_l293_293378


namespace geom_sum_first_eight_terms_l293_293091

theorem geom_sum_first_eight_terms (a r : ℚ) (h_a : a = 1/3) (h_r : r = 1/3) :
    ∑ k in finset.range 8, a * r^k = 9840/19683 := by
  sorry

end geom_sum_first_eight_terms_l293_293091


namespace sum_of_variables_l293_293951

theorem sum_of_variables (a b c d : ℝ) (h₁ : a * c + a * d + b * c + b * d = 68) (h₂ : c + d = 4) : a + b + c + d = 21 :=
sorry

end sum_of_variables_l293_293951


namespace ceil_sum_of_sqr_roots_l293_293040

theorem ceil_sum_of_sqr_roots : 
  (⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27) := 
by {
  -- Definitions based on conditions
  have h1 : 1^2 < 3 ∧ 3 < 2^2, by norm_num,
  have h2 : 5^2 < 33 ∧ 33 < 6^2, by norm_num,
  have h3 : 18^2 < 333 ∧ 333 < 19^2, by norm_num,
  sorry
}

end ceil_sum_of_sqr_roots_l293_293040


namespace minimum_term_of_b_l293_293164

noncomputable theory
open_locale big_operators

open finset

def a (n : ℕ) : ℕ := 2 * n - 1

def S (n : ℕ) : ℕ := ∑ i in range n, a (i + 1)

def b (n : ℕ) : ℚ := 2 ^ (a n) / (S n) ^ 2

theorem minimum_term_of_b : ∀ n : ℕ, (n ≥ 1) → 
  (∀ m : ℕ, (m ≠ 3) → b 3 ≤ b m) :=
by
  intro n hn m hm,
  sorry

end minimum_term_of_b_l293_293164


namespace product_zero_of_sum_replacement_l293_293892

theorem product_zero_of_sum_replacement (s : Fin 1997 → ℝ) 
  (h : ∀ i, s i = ∑ j in Finset.univ.erase i, s j) : 
  (∏ i, s i) = 0 := 
by
  sorry

end product_zero_of_sum_replacement_l293_293892


namespace clock_angle_at_3_45_l293_293373

theorem clock_angle_at_3_45 :
  let minute_angle_rate := 6.0 -- degrees per minute
  let hour_angle_rate := 0.5  -- degrees per minute
  let initial_angle := 90.0   -- degrees at 3:00
  let minutes_passed := 45.0  -- minutes since 3:00
  let angle_difference_rate := minute_angle_rate - hour_angle_rate
  let angle_change := angle_difference_rate * minutes_passed
  let final_angle := initial_angle - angle_change
  let smaller_angle := if final_angle < 0 then 360.0 + final_angle else final_angle
  smaller_angle = 157.5 :=
by
  sorry

end clock_angle_at_3_45_l293_293373


namespace probability_of_two_green_marbles_l293_293425

-- Define the problem
def total_marbles : ℕ := 3 + 4 + 10 + 5
def green_marbles : ℕ := 4

-- Define the event for drawing two green marbles in succession without replacement
def prob_draw_two_green_without_replacement : ℚ :=
  (green_marbles.to_rat / total_marbles.to_rat) * ((green_marbles - 1).to_rat / (total_marbles - 1).to_rat)

-- The theorem to be proved
theorem probability_of_two_green_marbles :
  prob_draw_two_green_without_replacement = 2 / 77 :=
by
  -- Lean expects proof here
  sorry

end probability_of_two_green_marbles_l293_293425


namespace stacy_berries_l293_293275

theorem stacy_berries (total_berries : ℕ) 
  (sylar_berries : ℕ) (stacy_to_steve : ℕ → ℕ) (steve_to_sylar : ℕ → ℕ) :
  total_berries = 1100 ∧ stacy_to_steve (steve_to_sylar sylar_berries) = 8 * sylar_berries ∧ stacy_to_steve = (λ n, 4 * n) ∧ steve_to_sylar = (λ n, 2 * n) →
  stacy_to_steve (steve_to_sylar sylar_berries) = 800 :=
by
  sorry

end stacy_berries_l293_293275


namespace printer_time_l293_293776

theorem printer_time (Tx : ℝ) 
  (h1 : ∀ (Ty Tz : ℝ), Ty = 10 → Tz = 20 → 1 / Ty + 1 / Tz = 3 / 20) 
  (h2 : ∀ (T_combined : ℝ), T_combined = 20 / 3 → Tx / T_combined = 2.4) :
  Tx = 16 := 
by 
  sorry

end printer_time_l293_293776


namespace finite_set_of_n_l293_293644

open Int

theorem finite_set_of_n (a b : ℤ) (ha : 0 < a) (hb : 0 < b) :
  {n : ℕ | ∃ (n : ℕ),  int (a + 1 / 2) ^ n + int (b + 1 / 2) ^ n ∈ ℤ}.finite :=
sorry

end finite_set_of_n_l293_293644


namespace product_a_b_l293_293188

variable (a b c : ℝ)
variable (h_pos_a : a > 0)
variable (h_pos_b : b > 0)
variable (h_pos_c : c > 0)
variable (h_c : c = 3)
variable (h_a : a = b^2)
variable (h_bc : b + c = b * c)

theorem product_a_b : a * b = 27 / 8 :=
by
  -- We need to prove that given the above conditions, a * b = 27 / 8
  sorry

end product_a_b_l293_293188


namespace number_of_integer_solutions_l293_293485

theorem number_of_integer_solutions : 
  let sol_set := {x : ℤ | (-4 : ℤ) * x ≥ 2 * x + 10 ∧ -3 * x ≤ 15 ∧ -5 * x ≥ 3 * x + 24} in
  finset.card sol_set = 3 :=
by
  sorry

end number_of_integer_solutions_l293_293485


namespace equal_distances_l293_293237

-- Given conditions as definitions in Lean
variables (Γ ω : Circle) (A B P Q S : Point)
variable h1 : OnCircle A Γ
variable h2 : OnCircle B Γ
variable h3 : TangentAt ω Γ P
variable h4 : TangentAt ω (LineSegment A B) Q
variable h5 : Intersection (Line PQ) Γ = {P, S}

-- The statement to be proved
theorem equal_distances (h1 : OnCircle A Γ) (h2 : OnCircle B Γ) 
  (h3 : TangentAt ω Γ P) (h4 : TangentAt ω (LineSegment A B) Q)
  (h5 : Intersection (LineSegment PQ) Γ = {P, S}) : dist S A = dist S B := by
  sorry

end equal_distances_l293_293237


namespace part_I_part_II_i_part_II_ii_l293_293153

def f (x : ℝ) (a : ℝ) := a - 1/x - Real.log x
def g (x : ℝ) (a : ℝ) (p : ℝ) := a - 1/x - 2 * (x - p) / (x + p) - f x a - Real.log p

theorem part_I (a : ℝ) :
  (∃ x : ℝ, f x a = 0) → ∀ x : ℝ, f x a = 0 → a = 1 :=
sorry

theorem part_II_i (a : ℝ) (p : ℝ) (hp : 0 < p) :
  ∀ x₁ x₂ : ℝ, (x₁ < x₂) → (f x₁ a = 0) → (f x₂ a = 0) →
    (g x₁ a p = g x₂ a p) :=
sorry

theorem part_II_ii (a : ℝ) (x₁ x₂ : ℝ) (h₀ : x₁ < x₂) (h₁ : f x₁ a = 0) (h₂ : f x₂ a = 0) :
  x₁ + x₂ < 3 * Real.exp (a - 1) - 1 :=
sorry

end part_I_part_II_i_part_II_ii_l293_293153


namespace product_lt_square_l293_293386

theorem product_lt_square :
  let n := 1234568 in
  (n - 1) * (n + 1) < n^2 :=
by
  let n := 1234568
  sorry

end product_lt_square_l293_293386


namespace Wang_diff_restaurants_X_distribution_expectation_X_P_M_conditional_l293_293395

-- Definitions based on given conditions
def days (Wang : ℕ × ℕ × ℕ × ℕ) (Zhang : ℕ × ℕ × ℕ × ℕ) := (Wang, Zhang)

-- Condition: Independent choices
axiom Wang_Zhang_indep : Prop

-- Part 1: Probability of Wang choosing different restaurants for lunch and dinner
def P_C (Wang : ℕ × ℕ × ℕ × ℕ) : ℚ :=
  let AA := 9
  let AB := 6
  let BA := 12
  let BB := 3
  (AB + BA) / 30

theorem Wang_diff_restaurants (Wang : ℕ × ℕ × ℕ × ℕ)
  (hWang : Wang = (9, 6, 12, 3)) : P_C Wang = 0.6 := sorry

-- Part 2: Distribution and expectation of X
def P_X_1 : ℚ := 0.1
def P_X_2 : ℚ := 0.9

theorem X_distribution (P_X_1 : ℚ) (P_X_2 : ℚ)
  (hP : P_X_1 = 0.1 ∧ P_X_2 = 0.9) : 
  ∃ X : ℕ → ℚ, (X 1 = P_X_1) ∧ (X 2 = P_X_2) := sorry

theorem expectation_X (P_X_1 : ℚ) (P_X_2 : ℚ)
  (hP : P_X_1 = 0.1 ∧ P_X_2 = 0.9) : 
  (1 * P_X_1 + 2 * P_X_2 = 1.9) := sorry

-- Part 3: Conditional probabilities and proof
axiom P_M_pos : Prop
axiom P_N_given_M (P : ℚ) : Prop
axiom P_N_given_not_M (P : ℚ) : Prop

theorem P_M_conditional (P_N_given_M : ℚ) (P_N_given_not_M : ℚ)
  (h_cond : P_N_given_M > P_N_given_not_M) : 
  P(M|N) > P(M|\overline{N}) := sorry

end Wang_diff_restaurants_X_distribution_expectation_X_P_M_conditional_l293_293395


namespace identify_heaviest_and_lightest_coin_in_13_weighings_l293_293311

theorem identify_heaviest_and_lightest_coin_in_13_weighings :
  ∀ (coins : Finₓ 10 → ℝ) 
    (balance_weighing : ∀ (a b : Finₓ 10), Prop), 
    (∀ i j, coins i ≠ coins j) → 
    (∃ strategy : ℕ → (Finₓ 10 × Finₓ 10),
      ∃ h : ℕ,
        h ≤ 13 ∧
        (∃ heaviest lightest : Finₓ 10,
          (∀ i, coins heaviest ≥ coins i) ∧ (∀ j, coins lightest ≤ coins j))) :=
by
  sorry

end identify_heaviest_and_lightest_coin_in_13_weighings_l293_293311


namespace area_of_gray_region_l293_293366

theorem area_of_gray_region
  (r : ℝ)
  (R : ℝ)
  (diameter_small_circle : ℝ)
  (diameter_small_circle_eq : diameter_small_circle = 4)
  (radius_small_circle : r = diameter_small_circle / 2)
  (radius_large_circle : R = 3 * r) :
  let As := π * r^2,
      AL := π * R^2 in
  AL - As = 32 * π :=
by
  -- Definitions for readability and decoration
  have radius_smaller := diameter_small_circle_eq ▸ radius_small_circle,
  have radius_larger := congr_arg (λ x, 3 * x) radius_smaller,
  let area_smaller := π * (r^2),
  let area_larger := π * (R^2),
  sorry

end area_of_gray_region_l293_293366


namespace ceil_sum_sqrt_evaluation_l293_293056

theorem ceil_sum_sqrt_evaluation :
  (⌈real.sqrt 3⌉ + ⌈real.sqrt 33⌉ + ⌈real.sqrt 333⌉ = 27) :=
begin
  have h1 : 1 < real.sqrt 3 ∧ real.sqrt 3 < 2 := sorry,
  have h2 : 5 < real.sqrt 33 ∧ real.sqrt 33 < 6 := sorry,
  have h3 : 18 < real.sqrt 333 ∧ real.sqrt 333 < 19 := sorry,
  sorry,
end

end ceil_sum_sqrt_evaluation_l293_293056


namespace remainder_when_1_stmt_l293_293085

-- Define the polynomial g(s)
def g (s : ℚ) : ℚ := s^15 + 1

-- Define the remainder theorem statement in the context of this problem
theorem remainder_when_1_stmt (s : ℚ) : g 1 = 2 :=
  sorry

end remainder_when_1_stmt_l293_293085


namespace area_outside_two_small_squares_l293_293426

theorem area_outside_two_small_squares (L S : ℝ) (hL : L = 9) (hS : S = 4) :
  let large_square_area := L^2
  let small_square_area := S^2
  let combined_small_squares_area := 2 * small_square_area
  large_square_area - combined_small_squares_area = 49 :=
by
  sorry

end area_outside_two_small_squares_l293_293426


namespace multiplicative_inverse_mod_l293_293016

-- We define our variables
def a := 154
def m := 257
def inv_a := 20

-- Our main theorem stating that inv_a is indeed the multiplicative inverse of a modulo m
theorem multiplicative_inverse_mod : (a * inv_a) % m = 1 := by
  sorry

end multiplicative_inverse_mod_l293_293016


namespace complex_z_count_l293_293858

-- Translate conditions as definitions
definition is_on_unit_circle (z : ℂ) : Prop :=
  complex.abs z = 1

-- Translate the problem statement
theorem complex_z_count (z : ℂ) (h : is_on_unit_circle z) :
  (z ^ 7.factorial - z ^ 6.factorial).im = 0 ↔
  ∃ n : ℕ, n = 7200 :=
sorry

end complex_z_count_l293_293858


namespace monotonicity_of_f_inequality_f_derivative_at_mid_l293_293563

def f (a x : ℝ) : ℝ := Real.exp x - a * x

theorem monotonicity_of_f (a : ℝ) (x : ℝ) :
    ∃ l : ℝ, (a ≤ 0 → ∀ x, f' (a x) > 0) ∧
             (a > 0 → ∃ l : ℝ, ∀ x < l, f' (a x) < 0 ∧ ∀ x > l, f' (a x) > 0) := sorry

theorem inequality_f (a x : ℝ) (h : x > 0) :
    f a (Real.log a + x) > f a (Real.log a - x) := sorry

theorem derivative_at_mid (a : ℝ) (x₁ x₂ : ℝ)
    (h1 : f a x₁ = 0) (h2 : f a x₂ = 0) (h3 : x₁ < x₂) :
    (x₁ + x₂) / 2 < Real.log a → f' a ((x₁ + x₂) / 2) < 0 := sorry

end monotonicity_of_f_inequality_f_derivative_at_mid_l293_293563


namespace find_y_l293_293659

theorem find_y (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) 
  (h : (2 * a) ^ (2 * b ^ 2) = (a ^ b + y ^ b) ^ 2) : y = 4 * a ^ 2 - a := 
sorry

end find_y_l293_293659


namespace class_discussion_integer_l293_293801

theorem class_discussion_integer :
  ∃ M : ℕ, (∀ n ∈ (List.range 30).erase 22 ++ (List.range 25).drop 23, M % n = 0) ∧ M = 1237834741500 :=
by
  let M := 2^3 * 3^3 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29
  use M
  have : ∀ n ∈ (List.range 30).erase 23 ++ (List.range 26).drop 24, M % n = 0,
  { intros n hn,
    fin_cases hn,
    sorry },
  split; assumption

end class_discussion_integer_l293_293801


namespace length_of_each_episode_l293_293759

theorem length_of_each_episode : 
  ∀ (episodes : ℕ) (hours_per_day days : ℕ) (minutes_per_hour : ℕ),
    episodes = 90 →
    hours_per_day = 2 →
    days = 15 →
    minutes_per_hour = 60 →
    (hours_per_day * days * minutes_per_hour) / episodes = 20 :=
by
  intros episodes hours_per_day days minutes_per_hour h_episodes h_hours_per_day h_days h_minutes_per_hour
  rw [h_episodes, h_hours_per_day, h_days, h_minutes_per_hour]
  norm_num
  sorry

end length_of_each_episode_l293_293759


namespace volume_of_one_wedge_is_162π_l293_293438

-- Define the diameter of the sphere.
def diameter : ℝ := 18

-- Define the radius (r) using the diameter.
def radius : ℝ := diameter / 2

-- Calculate the volume of the sphere.
def volume_sphere : ℝ := (4 / 3) * Real.pi * (radius ^ 3)

-- The volume of one wedge.
def volume_wedge : ℝ := volume_sphere / 6

-- The theorem to be proved, stating that the volume of one wedge is 162π cubic inches.
theorem volume_of_one_wedge_is_162π :
  volume_wedge = 162 * Real.pi :=
sorry

end volume_of_one_wedge_is_162π_l293_293438


namespace hyperbola_eccentricity_l293_293557

theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ), (b = 5) → (c = 3) → (c^2 = a^2 + b) → (a > 0) →
  (a + c = 3) → (e = c / a) → (e = 3 / 2) :=
by
  intros a b c hb hc hc2 ha hac he
  sorry

end hyperbola_eccentricity_l293_293557


namespace fewer_pushups_l293_293473

theorem fewer_pushups (sets: ℕ) (pushups_per_set : ℕ) (total_pushups : ℕ) 
  (h1 : sets = 3) (h2 : pushups_per_set = 15) (h3 : total_pushups = 40) :
  sets * pushups_per_set - total_pushups = 5 :=
by
  sorry

end fewer_pushups_l293_293473


namespace split_tips_evenly_l293_293230

theorem split_tips_evenly :
  let julie_cost := 10
  let letitia_cost := 20
  let anton_cost := 30
  let total_cost := julie_cost + letitia_cost + anton_cost
  let tip_rate := 0.2
  let total_tip := total_cost * tip_rate
  let tip_per_person := total_tip / 3
  tip_per_person = 4 := by
  sorry

end split_tips_evenly_l293_293230


namespace candidateAVotes_correct_l293_293780

def totalVotes : ℕ := 560000
def invalidVotesPercent : ℝ := 0.15
def validVotesPercent : ℝ := 1 - invalidVotesPercent
def candidateAVotesPercent : ℝ := 0.65

def validVotes : ℕ := (validVotesPercent * totalVotes : ℝ).to_nat
def candidateAVotes : ℕ := (candidateAVotesPercent * validVotes : ℝ).to_nat

theorem candidateAVotes_correct : candidateAVotes = 309400 := by
  have validVotes_eq : validVotes = 476000 := by
    rw [validVotes, validVotesPercent, totalVotes]
    norm_num
    exact (476000 : ℝ).to_nat_eq.2 rfl
  rw [candidateAVotes, validVotes_eq, candidateAVotesPercent]
  norm_num  
  exact (309400 : ℝ).to_nat_eq.2 rfl

end candidateAVotes_correct_l293_293780


namespace jay_change_l293_293986

def cost_book : ℝ := 25
def cost_pen : ℝ := 4
def cost_ruler : ℝ := 1
def payment : ℝ := 50

theorem jay_change : (payment - (cost_book + cost_pen + cost_ruler) = 20) := sorry

end jay_change_l293_293986


namespace find_positive_real_solutions_l293_293069

variable {x_1 x_2 x_3 x_4 x_5 : ℝ}

theorem find_positive_real_solutions
  (h1 : (x_1^2 - x_3 * x_5) * (x_2^2 - x_3 * x_5) ≤ 0)
  (h2 : (x_2^2 - x_4 * x_1) * (x_3^2 - x_4 * x_1) ≤ 0)
  (h3 : (x_3^2 - x_5 * x_2) * (x_4^2 - x_5 * x_2) ≤ 0)
  (h4 : (x_4^2 - x_1 * x_3) * (x_5^2 - x_1 * x_3) ≤ 0)
  (h5 : (x_5^2 - x_2 * x_4) * (x_1^2 - x_2 * x_4) ≤ 0)
  (hx1 : 0 < x_1)
  (hx2 : 0 < x_2)
  (hx3 : 0 < x_3)
  (hx4 : 0 < x_4)
  (hx5 : 0 < x_5) :
  x_1 = x_2 ∧ x_2 = x_3 ∧ x_3 = x_4 ∧ x_4 = x_5 :=
by
  sorry

end find_positive_real_solutions_l293_293069


namespace max_g_value_on_interval_l293_293080

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_g_value_on_interval : ∃ x, 0 ≤ x ∧ x ≤ 2 ∧ ∀ y,  0 ≤ y ∧ y ≤ 2 → g x ≥ g y ∧ g x = 3 :=
-- Proof goes here
sorry

end max_g_value_on_interval_l293_293080


namespace disneyland_attractions_ordered_l293_293982

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def valid_sequences (total_attractions : ℕ) (parade_idx : ℕ) (fireworks_idx : ℕ) : ℕ :=
  if parade_idx < fireworks_idx then
    factorial total_attractions / 2
  else
    0

theorem disneyland_attractions_ordered (total_attractions : ℕ) (parade_idx : ℕ) (fireworks_idx : ℕ) :
  total_attractions = 6 ∧ parade_idx < fireworks_idx →
  valid_sequences total_attractions parade_idx fireworks_idx = 360 :=
by
  intros h
  have h1 : factorial total_attractions = 720 := by sorry
  have h2 : valid_sequences total_attractions parade_idx fireworks_idx = 720 / 2 := by sorry
  show valid_sequences total_attractions parade_idx fireworks_idx = 360, from h2

end disneyland_attractions_ordered_l293_293982


namespace find_angle_l293_293731

variable (x : ℝ)

theorem find_angle (h1 : x + (180 - x) = 180) (h2 : x + (90 - x) = 90) (h3 : 180 - x = 3 * (90 - x)) : x = 45 := 
by
  sorry

end find_angle_l293_293731


namespace problem_statement_l293_293648

noncomputable def a (n : ℕ) : ℕ := n
noncomputable def S (n : ℕ) : ℕ := n * (n + 1) / 2
noncomputable def b (n : ℕ) : ℕ := Int.floor (Real.log10 n)

theorem problem_statement :
  a 1 = 1 ∧ S 7 = 28 ∧ b 1 = 0 ∧ b 11 = 1 ∧ b 101 = 2 ∧ (∑ n in Finset.range 1000, b (n + 1)) = 1893 :=
by {
  sorry
}

end problem_statement_l293_293648


namespace product_of_y_coordinates_l293_293681

theorem product_of_y_coordinates :
  ∀ y : ℝ, ((-4, y) on_line : ℝ × ℝ) ∧ dist (-4, y) (3, -1) = 13 → y ∈ { -1 + 2 * real.sqrt 30, -1 - 2 * real.sqrt 30 } →
  ∏ (y in ({ -1 + 2 * real.sqrt 30, -1 - 2 * real.sqrt 30 }), y) = -119 := by
  sorry

end product_of_y_coordinates_l293_293681


namespace length_of_platform_correct_l293_293398

/-- Define a structure to represent a train crossing problem -/
structure TrainCrossingPlatform where
  train_length : ℝ
  time_to_cross_pole : ℝ
  time_to_cross_platform : ℝ

def example_problem : TrainCrossingPlatform := {
  train_length := 300,
  time_to_cross_pole := 18,
  time_to_cross_platform := 48
}

/-- Define a constant for the answer -/
def platform_length (p : TrainCrossingPlatform) : ℝ :=
  let speed := p.train_length / p.time_to_cross_pole
  let total_distance := speed * p.time_to_cross_platform
  total_distance - p.train_length
  
theorem length_of_platform_correct (p : TrainCrossingPlatform) : platform_length p = 500.16 :=
  by
    sorry

end length_of_platform_correct_l293_293398


namespace zachary_cans_first_day_l293_293396

theorem zachary_cans_first_day :
  ∃ (first_day_cans : ℕ),
    ∃ (second_day_cans : ℕ),
      ∃ (third_day_cans : ℕ),
        ∃ (seventh_day_cans : ℕ),
          second_day_cans = 9 ∧
          third_day_cans = 14 ∧
          (∀ (n : ℕ), 2 ≤ n ∧ n < 7 → third_day_cans = second_day_cans + 5) →
          seventh_day_cans = 34 ∧
          first_day_cans = second_day_cans - 5 ∧
          first_day_cans = 4 :=

by
  sorry

end zachary_cans_first_day_l293_293396


namespace point_in_polar_coordinates_l293_293840

noncomputable def convert_to_polar : ℝ × ℝ → ℝ × ℝ
| (x, y) := (Real.sqrt (x^2 + y^2), if y < 0 then 2*Real.pi - Real.atan (y / x) else Real.atan (y / x))

theorem point_in_polar_coordinates :
  convert_to_polar (Real.sqrt 3, -1) = (2, 11 * Real.pi / 6) :=
by
  -- hints for calculating these values are provided in the problem
  sorry

end point_in_polar_coordinates_l293_293840


namespace angle_YZX_l293_293006

theorem angle_YZX {A B C : Type} {γ : Circle} {X Y Z : Point} (h_incircle : γ.incircle A B C) 
  (h_circumcircle : γ.circumcircle X Y Z) (hX : X ∈ segment B C) (hY : Y ∈ segment A B) 
  (hZ : Z ∈ segment A C) (angle_A : angle A = 50) (angle_B : angle B = 70) (angle_C : angle C = 60) : 
  angle (segment Y Z) (segment Z X) = 65 := 
sorry

end angle_YZX_l293_293006


namespace distance_between_centers_l293_293942

variable (P R r : ℝ)
variable (h_tangent : P = R - r)
variable (h_radius1 : R = 6)
variable (h_radius2 : r = 3)

theorem distance_between_centers : P = 3 := by
  sorry

end distance_between_centers_l293_293942


namespace coin_sort_expected_lt_4_8_l293_293750

def coin_weight_sorting_expected_weighings (A B C D : ℝ) (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ D) (hD : D ≠ A) (hAC : A ≠ C) (hBD : B ≠ D) (hAD : A ≠ D) (hBC : B ≠ C) : Prop :=
  ∃ sorting_strategy, expected_weighings sorting_strategy < 4.8

theorem coin_sort_expected_lt_4_8 (A B C D : ℝ) (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ D) (hD : D ≠ A) (hAC : A ≠ C) (hBD : B ≠ D) (hAD : A ≠ D) (hBC : B ≠ C) :
  coin_weight_sorting_expected_weighings A B C D hA hB hC hD hAC hBD hAD hBC :=
sorry

end coin_sort_expected_lt_4_8_l293_293750


namespace triangle_middle_side_at_least_sqrt_two_l293_293446

theorem triangle_middle_side_at_least_sqrt_two
    (a b c : ℝ)
    (h1 : a ≥ b) (h2 : b ≥ c)
    (h3 : ∃ α : ℝ, 0 < α ∧ α < π ∧ 1 = 1/2 * b * c * Real.sin α) :
  b ≥ Real.sqrt 2 :=
sorry

end triangle_middle_side_at_least_sqrt_two_l293_293446


namespace bisector_unit_vector_l293_293238

noncomputable def a : ℝ × ℝ × ℝ := (3, 4, 2)
noncomputable def b : ℝ × ℝ × ℝ := (-1, 1, 1)
noncomputable def v : ℝ × ℝ × ℝ := (-0.72, 0.03, 0.69)

-- Define the norm of vector a
noncomputable def norm_a : ℝ := real.sqrt ((3 * 3) + (4 * 4) + (2 * 2))

-- Unit vector condition and bisector condition
theorem bisector_unit_vector
(h_unit : real.sqrt ((v.1)^2 + (v.2)^2 + (v.3)^2) = 1)
(h_bisect : b = (1/2 :ℝ) * (a.1 + norm_a * v.1, a.2 + norm_a * v.2, a.3 + norm_a * v.3)) :
(v = (-0.72, 0.03, 0.69)) :=
sorry

end bisector_unit_vector_l293_293238


namespace minimum_area_triangle_l293_293941

theorem minimum_area_triangle (A B C : ℝ) (a : ℝ) (c : ℝ) :
  let AB := 2 in
  let sinA := c / real.sqrt (a^2 + c^2) in
  let tanB := c / (A - B) in
  (2 / sinA + 1 / tanB = 2 * real.sqrt 3) → c = 2 * real.sqrt 3 / 3 := sorry

end minimum_area_triangle_l293_293941


namespace reconstruct_pentagon_l293_293193

variables {K L M N A B C D E : Type*}

/-- Given four points K, L, M, N which are the midpoints of four consecutive sides of a
    circumscribed pentagon ABCDE, reconstruct the original pentagon ABCDE --/
theorem reconstruct_pentagon (circumscribed_pentagon : A B C D E)
  (Kmid : midpoint A B = K) (Lmid : midpoint B C = L)
  (Mmid : midpoint C D = M) (Nmid : midpoint D E = N) : 
  exists (A' B' C' D' E' : Type*), 
    circumscribed_pentagon = (A' B' C' D' E') :=
begin
  sorry
end

end reconstruct_pentagon_l293_293193


namespace heaviest_and_lightest_in_13_weighings_l293_293313

/-- Given ten coins of different weights and a balance scale.
    Prove that it is possible to identify the heaviest and the lightest coin
    within 13 weighings. -/
theorem heaviest_and_lightest_in_13_weighings
  (coins : Fin 10 → ℝ)
  (h_different: ∀ i j : Fin 10, i ≠ j → coins i ≠ coins j)
  : ∃ (heaviest lightest : Fin 10),
      (heaviest ≠ lightest) ∧
      (∀ i : Fin 10, coins i ≤ coins heaviest) ∧
      (∀ i : Fin 10, coins lightest ≤ coins i) :=
sorry

end heaviest_and_lightest_in_13_weighings_l293_293313


namespace probability_cosine_positive_l293_293937

theorem probability_cosine_positive (A : Finset ℝ) (hA : A = {0, π/6, π/4, π/3, π/2, 2 * π / 3, 3 * π / 4, 5 * π / 6, π}) :
  (∑ x in A, if Real.cos x > 0 then 1 else 0) / A.card = 4 / 9 :=
by
  sorry

end probability_cosine_positive_l293_293937


namespace triangle_is_isosceles_l293_293540

theorem triangle_is_isosceles
  (α β γ x y z w : ℝ)
  (h1 : α + β + γ = 180)
  (h2 : α + β = x)
  (h3 : β + γ = y)
  (h4 : γ + α = z)
  (h5 : x + y + z + w = 360) : 
  (α = β ∧ β = γ) ∨ (α = γ ∧ γ = β) ∨ (β = α ∧ α = γ) := by
  sorry

end triangle_is_isosceles_l293_293540


namespace max_g_value_on_interval_l293_293081

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_g_value_on_interval : ∃ x, 0 ≤ x ∧ x ≤ 2 ∧ ∀ y,  0 ≤ y ∧ y ≤ 2 → g x ≥ g y ∧ g x = 3 :=
-- Proof goes here
sorry

end max_g_value_on_interval_l293_293081


namespace general_term_a_n_sum_T_n_l293_293913

noncomputable def a_n (n : ℕ) : ℕ := 2 * n

noncomputable def b_n (n : ℕ) : ℕ := n * 2 ^ n

noncomputable def T_n (n : ℕ) : ℕ := (n - 1) * 2^(n + 1) + 2

theorem general_term_a_n (n : ℕ) (a_3_a1 : 2 * (3 - 1) = 4) (S_3_val : 3 * (2 + 2) = 12) :
  a_n n = 2 * n :=
by sorry

theorem sum_T_n (n : ℕ) (b_eq : b_n = λ n, n * 2^n) :
  (finset.range n).sum (λ i, b_n (i + 1)) = T_n n :=
by sorry

end general_term_a_n_sum_T_n_l293_293913


namespace num_ways_express_2009_as_diff_of_squares_l293_293495

theorem num_ways_express_2009_as_diff_of_squares : 
  ∃ (n : Nat), n = 12 ∧ 
  ∃ (a b : Int), ∀ c, 2009 = a^2 - b^2 ∧ 
  (c = 1 ∨ c = -1) ∧ (2009 = (c * a)^2 - (c * b)^2) :=
sorry

end num_ways_express_2009_as_diff_of_squares_l293_293495


namespace rectangle_perimeter_of_equal_area_l293_293349

theorem rectangle_perimeter_of_equal_area (a b c : ℕ) (area_triangle width length : ℕ) :
  a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2 ∧ (2 * area_triangle = a * b) ∧
  (width = 6) ∧ (area_triangle = width * length) -> 
  2 * (length + width) = 30 :=
by
  intros h,
  sorry

end rectangle_perimeter_of_equal_area_l293_293349


namespace geometric_sequence_general_term_l293_293588

theorem geometric_sequence_general_term (x : ℝ) (h : ∀ n : ℕ, (0 < n) -> {a : ℕ -> ℝ |
  (a 1 = x) ∧ (a 2 = x-1) ∧ (a 3 = 2*x-2)} ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q ) :
  ∀ n : ℕ, a n = -2^(n-1) := sorry

end geometric_sequence_general_term_l293_293588


namespace range_of_function_l293_293721

theorem range_of_function (x: ℝ) (h: x ≤ 1): ∃ y, y = (x - real.sqrt (1 - x)) ∧ y ≤ 1 :=
by
sorry

end range_of_function_l293_293721


namespace equal_pair_c_l293_293461

theorem equal_pair_c : (-4)^3 = -(4^3) := 
by {
  sorry
}

end equal_pair_c_l293_293461


namespace factorial_mod_13_l293_293519

open Nat

theorem factorial_mod_13 :
  let n := 10
  let p := 13
  n! % p = 6 := by
sorry

end factorial_mod_13_l293_293519


namespace solution_set_of_abs_inequality_l293_293514

theorem solution_set_of_abs_inequality :
  {x : ℝ | |x - 1| + |x + 2| ≤ 4} = set.Icc (-5 / 2) (3 / 2) :=
by
  sorry

end solution_set_of_abs_inequality_l293_293514


namespace ceil_sum_sqrt_evaluation_l293_293055

theorem ceil_sum_sqrt_evaluation :
  (⌈real.sqrt 3⌉ + ⌈real.sqrt 33⌉ + ⌈real.sqrt 333⌉ = 27) :=
begin
  have h1 : 1 < real.sqrt 3 ∧ real.sqrt 3 < 2 := sorry,
  have h2 : 5 < real.sqrt 33 ∧ real.sqrt 33 < 6 := sorry,
  have h3 : 18 < real.sqrt 333 ∧ real.sqrt 333 < 19 := sorry,
  sorry,
end

end ceil_sum_sqrt_evaluation_l293_293055


namespace sequence_formula_l293_293161

noncomputable def a : ℕ → ℝ
| 0 := 1
| (n+1) := real.sqrt ((a n)^2 + 1)

theorem sequence_formula (n : ℕ) : a n = real.sqrt n :=
begin
  sorry
end

end sequence_formula_l293_293161


namespace peter_drew_8_pictures_l293_293267

theorem peter_drew_8_pictures : 
  ∃ (P : ℕ), ∀ (Q R : ℕ), Q = P + 20 → R = 5 → R + P + Q = 41 → P = 8 :=
by
  sorry

end peter_drew_8_pictures_l293_293267


namespace final_vertices_correct_l293_293468

-- Define the initial vertices of the equilateral triangle
def initial_vertices : set (ℝ × ℝ) := { (1, 0), (0, real.sqrt 3), (-1, 0) }

-- Define the transformation functions
def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def rotate_180_ccw (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Apply the sequence of transformations to the set of vertices
def transformed_vertices : set (ℝ × ℝ) :=
  (initial_vertices.map reflect_over_x_axis).map rotate_180_ccw map reflect_over_y_eq_x

-- The final position of the vertices should be (0, -1), (real.sqrt 3, 0), and (0, 1)
theorem final_vertices_correct :
  transformed_vertices = { (0, -1), (real.sqrt 3, 0), (0, 1) } :=
sorry

end final_vertices_correct_l293_293468


namespace omega_range_l293_293549

noncomputable def symmetry_axis_condition (ω : ℝ) (x : ℝ) : Prop :=
  ¬ (π < ω * x + (π / 6) ∧ ω * x + (π / 6) < 2 * π)

theorem omega_range (ω : ℝ) (hω1 : ω > 1 / 4) (hω2 : ∀ x : ℝ, symmetry_axis_condition ω x) :
  1 / 3 ≤ ω ∧ ω ≤ 2 / 3 :=
by
  sorry

end omega_range_l293_293549


namespace max_consecutive_odd_prime_powers_set_l293_293173

def has_odd_prime_powers (n : ℕ) : Prop :=
  ∀ p in nat.prime_divisors n, (nat.multiplicity p n).get_or_else 0 % 2 = 1

theorem max_consecutive_odd_prime_powers_set : 
  ∃ (m : ℕ), 
  (∀ n ∈ finset.range m, has_odd_prime_powers (n + 1)) ∧
  m = 7 :=
sorry

end max_consecutive_odd_prime_powers_set_l293_293173


namespace subset_elements_sum_four_n_plus_one_l293_293250

theorem subset_elements_sum_four_n_plus_one (n : ℕ) (h : n > 0) :
  ∃ k, ∀ (A : Finset ℕ), A ⊆ Finset.range (2 * n + 1) → A.card = k →
    ∃ (x y z w : ℕ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧ 
    x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ w ∈ A ∧ x + y + z + w = 4 * n + 1 :=
begin
  use (n + 3),
  sorry
end

end subset_elements_sum_four_n_plus_one_l293_293250


namespace K_most_accurate_value_l293_293613

def K : ℝ := 1.78654
def accuracy : ℝ := 0.00443
def K_upper : ℝ := K + accuracy
def K_lower : ℝ := K - accuracy

theorem K_most_accurate_value :
  (K_upper.round (1/100)) = 1.79 ∧ (K_lower.round (1/100)) = 1.78 ∧ (K.round (1/10)) = 1.8 :=
by
  sorry

end K_most_accurate_value_l293_293613


namespace computation_distinct_collections_l293_293676

open Finset

def vowels := {'A', 'E', 'I', 'O', 'U'}  -- The set of all vowels in general.
def letters := "COMPUTATION".toList.toFinset  -- The letters in the word COMPUTATION (treated as a Finset of characters).

-- The specific numbers of each type of letter available:
def counts :=
  ('C', 1) :: ('O', 2) :: ('M', 1) :: ('P', 1) :: ('U', 1) :: ('T', 2) 
  :: ('A', 1) :: ('I', 1) :: ('N', 1) :: []

def distinct_collections_count : ℕ :=
  let vowels_count := ('A', 1) :: ('O', 2) :: ('U', 1) :: ('I', 1) :: []
  let consonants_count := ('C', 1) :: ('M', 1) :: ('P', 1) :: ('T', 2) :: ('N', 1) :: []
  let total_vowels_choices := (combinatorics.binomial 5 3).to_nat * 
    (let choices := (0 :: 1 :: []) in choices.sum (λ t_count, combinatorics.binomial 4 (4 - t_count).to_nat))
  let total_all_t_choices := (combinatorics.binomial 4 2).to_nat * 6
  in total_vowels_choices + total_all_t_choices

theorem computation_distinct_collections :
  distinct_collections_count = 110 :=
begin
  -- Omitted proof
  sorry
end

end computation_distinct_collections_l293_293676


namespace educational_trail_length_l293_293881
  
  noncomputable def length_of_educational_trail : ℕ → Prop
  | 4500 := true
  | 4700 := true
  | _    := false

  theorem educational_trail_length (x y w : ℕ)
    (H1 : w + x + y + y = 7700)
    (H2 : y + y + w = 5800)
    (H3 : x = 1900)
    (H4 : 2 * (x + y) + 1500 = 8800) :
    length_of_educational_trail (x + y) :=
  by
    sorry
  
end educational_trail_length_l293_293881


namespace line_tangent_to_circle_l293_293294

def line (x y : ℝ) : Prop := x + √3 * y - 4 = 0
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

theorem line_tangent_to_circle :
  (∃ x y : ℝ, line x y ∧ circle x y) → 
  ∀ x y : ℝ, circle x y → line x y → √ ((x - 0)^2 + (y - 0)^2) = √ (4) :=
sorry

end line_tangent_to_circle_l293_293294


namespace median_price_of_baskets_l293_293691

-- Define the basket prices
def basket_prices : List ℕ := [5, 4, 6, 4, 4, 7, 5, 6, 8]

-- Define a function to calculate the median of a list of prices
def median (prices : List ℕ) : ℕ :=
  let sorted_prices := List.sort (· ≤ ·) prices
  sorted_prices[(sorted_prices.length / 2)]

-- State the problem as a theorem to be proved
theorem median_price_of_baskets : median basket_prices = 5 := by
  sorry

end median_price_of_baskets_l293_293691


namespace each_person_tip_l293_293223

-- Definitions based on the conditions
def julie_cost : ℝ := 10
def letitia_cost : ℝ := 20
def anton_cost : ℝ := 30
def tip_rate : ℝ := 0.2

-- Theorem statement
theorem each_person_tip (total_cost := julie_cost + letitia_cost + anton_cost)
 (total_tip := total_cost * tip_rate) :
 (total_tip / 3) = 4 := by
  sorry

end each_person_tip_l293_293223


namespace split_tip_evenly_l293_293225

noncomputable def total_cost (julie_order : ℝ) (letitia_order : ℝ) (anton_order : ℝ) : ℝ :=
  julie_order + letitia_order + anton_order

noncomputable def total_tip (meal_cost : ℝ) (tip_rate : ℝ) : ℝ :=
  tip_rate * meal_cost

noncomputable def tip_per_person (total_tip : ℝ) (num_people : ℝ) : ℝ :=
  total_tip / num_people

theorem split_tip_evenly :
  let julie_order := 10 in
  let letitia_order := 20 in
  let anton_order := 30 in
  let tip_rate := 0.20 in
  let num_people := 3 in
  tip_per_person (total_tip (total_cost julie_order letitia_order anton_order) tip_rate) num_people = 4 :=
by
  sorry

end split_tip_evenly_l293_293225


namespace largest_matching_cost_under_500_l293_293336

-- Define the cost function for decimal digits
def cost1 (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the cost function for binary digits
def cost2 (n : ℕ) : ℕ :=
  (n.digits 2).sum

-- The statement to be proved
theorem largest_matching_cost_under_500 : 
  ∃ n, n < 500 ∧ cost1 n = cost2 n ∧ ∀ m, m < 500 ∧ cost1 m = cost2 m → m ≤ n :=
begin
  use 404,
  split,
  exact dec_trivial,
  split,
  { -- Check if 404 has equal cost in both options
    unfold cost1 cost2,
    have : (404.digits 10).sum = 8 := dec_trivial,
    have : (404.digits 2).sum = 4 := dec_trivial,
    sorry },
  { -- Check if 404's cost is the largest under 500
    sorry }
end

end largest_matching_cost_under_500_l293_293336


namespace card_placement_three_digits_min_card_boxes_less_than_forty_impossible_min_card_boxes_less_than_fifty_impossible_card_placement_four_digits_min_card_boxes_k_digits_l293_293106

open Nat

/--
1. Prove that for any 3-digit number card (from 000 to 999), all cards can be placed into 50 boxes based on the given rule.
-/
theorem card_placement_three_digits : numberOfBoxes 3 10 = 50 := sorry

/--
2. Prove that it is impossible to distribute all 3-digit cards into fewer than 40 boxes.
-/
theorem min_card_boxes_less_than_forty_impossible : minNumberOfBoxes 3 10 = 50 := sorry

/--
3. Prove that it is impossible to distribute all 3-digit cards into fewer than 50 boxes.
-/
theorem min_card_boxes_less_than_fifty_impossible : minNumberOfBoxes 3 10 = 50 := sorry

/--
4. Prove that for 4-digit cards (from 0000 to 9999), all cards can be placed into 34 boxes based on the given rule.
-/
theorem card_placement_four_digits : numberOfBoxes 4 10 = 34 := sorry

/--
5. Prove that for k-digit cards, the minimum number of boxes required based on the given rule.
-/
theorem min_card_boxes_k_digits (k : Nat) : k ≥ 4 → minNumberOfBoxes k 10 = minNumBoxes k 10 := sorry

end card_placement_three_digits_min_card_boxes_less_than_forty_impossible_min_card_boxes_less_than_fifty_impossible_card_placement_four_digits_min_card_boxes_k_digits_l293_293106


namespace a_n_formula_b_n_geometric_sequence_and_sum_c_n_sum_l293_293899

noncomputable def a_n(n : ℕ) : ℕ := 2 * n - 1

noncomputable def S_n(n : ℕ) : ℕ := n * (a_n(n) + a_n(1)) / 2
axiom S_9_eq_81 : S_n 9 = 81

axiom a_3_eq_5 : a_n 3 = 5

theorem a_n_formula : ∀ n, a_n n = 2 * n - 1 := sorry

noncomputable def b_n(n : ℕ) : ℝ := 2 ^ a_n n

theorem b_n_geometric_sequence_and_sum (n : ℕ) : 
    (∃ q : ℝ, ∀ m, b_n (m + 1) = q * b_n m) ∧ 
    (∀ n, ∑ i in Finset.range n, b_n (i + 1) = (2 / 3) * (4 ^ n - 1)) := sorry

noncomputable def c_n(n : ℕ) : ℝ := a_n n * b_n n

theorem c_n_sum (n : ℕ) : ∑ i in Finset.range n, c_n (i + 1) = (10 / 9) + ((12 * n - 10) / 9) * 4 ^ n := sorry

end a_n_formula_b_n_geometric_sequence_and_sum_c_n_sum_l293_293899


namespace rational_solution_exists_l293_293633

theorem rational_solution_exists :
  ∃ (a b : ℚ), (a + b) / a + a / (a + b) = b :=
by
  sorry

end rational_solution_exists_l293_293633


namespace sin_theta_value_l293_293239

variables {ℝ : Type} [InnerProductSpace ℝ (EuclideanSpace 3)] -- Specify Euclidean space

-- Define vectors a, b, c
variable (a b c : EuclideanSpace 3)

-- Define norms of the vectors given in the conditions
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 7
axiom norm_c : ∥c∥ = 2

-- Define the condition involving the vector triple product
axiom vec_triple_prod : a × (a × b) = 2 • c

-- Define the angle θ and the statement to be proved
noncomputable def θ := real.angle a b
noncomputable def sin_θ := real.sin θ

theorem sin_theta_value : sin_θ a b = 4 / 7 :=
by sorry

end sin_theta_value_l293_293239


namespace ratio_of_areas_triangle_l293_293974

variable (P Q R S : Type) [Point P] [Point Q] [Point R] [Point S]
variable [Triangle P Q R]

-- Define the lengths of the sides
variables (PQ PR QR : ℝ)
variables [PQ_length : PQ = 18] [PR_length : PR = 27] [QR_length : QR = 23]

-- Assume PS is an angle bisector in triangle PQR
variable [AngleBisector P Q R S]

-- Define the ratio of the areas of triangles PQS and PRS
def ratio_of_areas (P Q R S : Type) [Point P] [Point Q] [Point R] [Point S]
  [Triangle P Q R] [AngleBisector P Q R S] : ℚ :=
  have h₁ : PQ = 18 := PQ_length
  have h₂ : PR = 27 := PR_length
  by sorry -- Assume angle bisector theorem and areas calculation

theorem ratio_of_areas_triangle (P Q R S : Type) [Point P] [Point Q] [Point R] [Point S]
  [Triangle P Q R] [AngleBisector P Q R S] 
  (PQ PR QR : ℝ) [PQ_length : PQ = 18] [PR_length : PR = 27] [QR_length : QR = 23] :
  ratio_of_areas P Q R S = 2 / 3 := 
sorry

end ratio_of_areas_triangle_l293_293974


namespace Ryan_bike_time_l293_293758

-- Definitions of the conditions
variables (B : ℕ)

-- Conditions
def bike_time := B
def bus_time := B + 10
def friend_time := B / 3
def commuting_time := bike_time B + 3 * bus_time B + friend_time B = 160

-- Goal to prove
theorem Ryan_bike_time : commuting_time B → B = 30 :=
by
  intro h
  sorry

end Ryan_bike_time_l293_293758


namespace tetrahedron_equal_volume_l293_293448

/-- Given a tetrahedron ABCD, where M and N are midpoints of edges AB and CD respectively, 
show that any plane passing through M and N divides the tetrahedron into two regions of equal volume. -/
theorem tetrahedron_equal_volume (A B C D M N : ℝ³) (M_mid_AB : M = (A + B) / 2)
                                          (N_mid_CD : N = (C + D) / 2) :
  ∀ (plane : ℝ³ → Prop), (plane M → plane N) → 
  let volumes_split (v1 v2 : ℝ≥0) := (v1 = v2) in
  volumes_split (volume_of_tetra_part A B C D plane) (volume_of_tetra_part A B C D (λ P, ¬ plane P)) :=
by
  sorry

end tetrahedron_equal_volume_l293_293448


namespace ratio_of_areas_l293_293102

variables {A B C P K M N : Type} [ground_field : Type]
variables (a b c k m n : ℝ)

-- all conditions in the problem
def triangle_is_acute (A B C : Type) [triangle A B C] : Prop := sorry
def point_is_inside_triangle (P A B C : Type) [triangle A B C] : Prop := sorry
def perpendiculars_drawn (P : Type) [to_sides {a b c k m n : ℝ}] : Prop := sorry

theorem ratio_of_areas
  (h1 : triangle_is_acute A B C)
  (h2 : point_is_inside_triangle P A B C)
  (h3 : perpendiculars_drawn P a b c k m n) :
  (area_triangle_abc * (m n a + k n b + k m c)) = ((a b c) * area_triangle_formed_perpendiculars) := 
begin
  sorry
end

end ratio_of_areas_l293_293102


namespace sum_of_products_of_digits_l293_293096

def product_of_digits (n : ℕ) : ℕ :=
  (n.to_digits 10).prod

def Tala_sum : ℕ :=
  (List.range' 1 2019).sum (λ n => product_of_digits n)

theorem sum_of_products_of_digits :
  Tala_sum = 184320 := 
by 
  sorry

end sum_of_products_of_digits_l293_293096


namespace number_of_sets_l293_293115

theorem number_of_sets {M : Finset ℕ} : 
  {1, 2} ⊆ M ∧ M ⊆ {1, 2, 3, 4, 5} → (Finset.card {M | {1, 2} ⊆ M ∧ M ⊆ {1, 2, 3, 4, 5}} = 7) :=
by 
  sorry

end number_of_sets_l293_293115


namespace negation_of_p_implication_q_l293_293897

noncomputable def negation_of_conditions : Prop :=
∀ (a : ℝ), (a > 0 → a^2 > a) ∧ (¬(a > 0) ↔ ¬(a^2 > a)) → ¬(a ≤ 0 → a^2 ≤ a)

theorem negation_of_p_implication_q :
  negation_of_conditions :=
by {
  sorry
}

end negation_of_p_implication_q_l293_293897


namespace sum_of_roots_of_quadratic_l293_293377

theorem sum_of_roots_of_quadratic (a b c : ℕ) (h : a = 1 ∧ b = -10 ∧ c = 24)
  (h_eq : ∀ x, x^2 = 10 * x - 24 ↔ a * x^2 + b * x + c = 0) :
  -b = 10 := 
by 
  sorry

end sum_of_roots_of_quadratic_l293_293377


namespace volume_ratio_l293_293723

noncomputable def salinity_bay (salt_bay volume_bay : ℝ) : ℝ :=
  salt_bay / volume_bay

noncomputable def salinity_sea_excluding_bay (salt_sea_excluding_bay volume_sea_excluding_bay : ℝ) : ℝ :=
  salt_sea_excluding_bay / volume_sea_excluding_bay

noncomputable def salinity_whole_sea (salt_sea volume_sea : ℝ) : ℝ :=
  salt_sea / volume_sea

theorem volume_ratio (salt_bay volume_bay salt_sea_excluding_bay volume_sea_excluding_bay : ℝ) 
  (h_bay : salinity_bay salt_bay volume_bay = 240 / 1000)
  (h_sea_excluding_bay : salinity_sea_excluding_bay salt_sea_excluding_bay volume_sea_excluding_bay = 110 / 1000)
  (h_whole_sea : salinity_whole_sea (salt_bay + salt_sea_excluding_bay) (volume_bay + volume_sea_excluding_bay) = 120 / 1000) :
  (volume_bay + volume_sea_excluding_bay) / volume_bay = 13 := 
sorry

end volume_ratio_l293_293723


namespace smallest_solution_l293_293086

noncomputable def frac_part (x : ℝ) : ℝ := x - (floor x)

theorem smallest_solution : ∃ x : ℝ, (floor x = 7 + 50 * frac_part x) ∧ (0 ≤ frac_part x) ∧ (frac_part x < 1) ∧ (x = 343 / 50) :=
by
  sorry

end smallest_solution_l293_293086


namespace factorization_problem_1_factorization_problem_2_l293_293850

-- Problem 1: Factorize 2(m-n)^2 - m(n-m) and show it equals (n-m)(2n - 3m)
theorem factorization_problem_1 (m n : ℝ) :
  2 * (m - n)^2 - m * (n - m) = (n - m) * (2 * n - 3 * m) :=
by
  sorry

-- Problem 2: Factorize -4xy^2 + 4x^2y + y^3 and show it equals y(2x - y)^2
theorem factorization_problem_2 (x y : ℝ) :
  -4 * x * y^2 + 4 * x^2 * y + y^3 = y * (2 * x - y)^2 :=
by
  sorry

end factorization_problem_1_factorization_problem_2_l293_293850


namespace number_of_squares_factors_of_2000_is_6_l293_293306

theorem number_of_squares_factors_of_2000_is_6 : 
  (finset.filter (λ n, n^2 ∣ 2000) (finset.range 2001)).card = 6 := 
by
  sorry

end number_of_squares_factors_of_2000_is_6_l293_293306


namespace angle_of_inclination_eq_pi_div_4_l293_293204

theorem angle_of_inclination_eq_pi_div_4 :
  ∃ θ : ℝ, (∃ (m : ℝ), m = 1 ∧ tan θ = m) ∧ 0 ≤ θ ∧ θ < π ∧ θ = π / 4 :=
by
  sorry

end angle_of_inclination_eq_pi_div_4_l293_293204


namespace count_complex_numbers_l293_293083

def isReal (z : ℂ) : Prop := z.im = 0

theorem count_complex_numbers (n m : ℕ) (h1 : |z| = 1) (h2 : isReal (z ^ n - z ^ m) = true) (h3 : isReal (z ^ m - z ^ (m div 6)) = true) 
    (h4 : n = 7! ∧ m = 6!) : 
    ∃ k, 0 ≤ k ∧ k < 7 ∧ (∀ θ ∈ {0..360}, |θ| = 1 ∧ isReal (θ ^ n - θ ^ m) = true ∧ isReal (θ ^ m - θ ^ (m div 6)) = true) ↔ (θ ∈ {0, 360}) := sorry

end count_complex_numbers_l293_293083


namespace drawing_10_balls_always_includes_all_colors_l293_293266

def bag_contains_sufficient_balls
    (w r b n : Nat)
    (hw : w = 5)
    (hr : r = 4)
    (hb : b = 3)
    (hn : n = 10) : Prop :=
  ∀ draw : Fin n → Fin (w + r + b), ∃ i j k, 
    i < w ∧ j < r ∧ k < b ∧ 
    ∃ t, draw t = i ∨ draw t = j ∨ draw t = k

theorem drawing_10_balls_always_includes_all_colors :
  bag_contains_sufficient_balls 5 4 3 10
sorry

end drawing_10_balls_always_includes_all_colors_l293_293266


namespace wendys_sales_are_205_l293_293361

def price_of_apple : ℝ := 1.5
def price_of_orange : ℝ := 1.0
def apples_sold_morning : ℕ := 40
def oranges_sold_morning : ℕ := 30
def apples_sold_afternoon : ℕ := 50
def oranges_sold_afternoon : ℕ := 40

/-- Wendy's total sales for the day are $205 given the conditions about the prices of apples and oranges,
and the number of each sold in the morning and afternoon. -/
def wendys_total_sales : ℝ :=
  let total_apples_sold := apples_sold_morning + apples_sold_afternoon
  let total_oranges_sold := oranges_sold_morning + oranges_sold_afternoon
  let sales_from_apples := total_apples_sold * price_of_apple
  let sales_from_oranges := total_oranges_sold * price_of_orange
  sales_from_apples + sales_from_oranges

theorem wendys_sales_are_205 : wendys_total_sales = 205 := by
  sorry

end wendys_sales_are_205_l293_293361


namespace value_of_a_l293_293256

-- Conditions
def A (a : ℝ) : Set ℝ := {2, a}
def B (a : ℝ) : Set ℝ := {-1, a^2 - 2}

-- Theorem statement asserting the condition and the correct answer
theorem value_of_a (a : ℝ) : (A a ∩ B a).Nonempty → a = -2 :=
by
  sorry

end value_of_a_l293_293256


namespace rectangle_perimeter_of_equal_area_l293_293348

theorem rectangle_perimeter_of_equal_area (a b c : ℕ) (area_triangle width length : ℕ) :
  a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2 ∧ (2 * area_triangle = a * b) ∧
  (width = 6) ∧ (area_triangle = width * length) -> 
  2 * (length + width) = 30 :=
by
  intros h,
  sorry

end rectangle_perimeter_of_equal_area_l293_293348


namespace distance_from_Q_to_EH_l293_293697

structure Square :=
(E F G H : ℝ × ℝ)
(side_length : ℝ)
(H_side_len : dist E F = side_length ∧ dist F G = side_length ∧ dist G H = side_length ∧ dist H E = side_length)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the square EFGH with given points and side length 5
-- H is at (0, 0), G at (5, 0), F at (5, 5), E at (0, 5)
noncomputable def EF (s : Square) : (ℝ × ℝ) := (s.E, s.F)
noncomputable def GH (s : Square) : (ℝ × ℝ) := (s.G, s.H)

-- Given square EFGH with the provided coordinates
def EFGH_square := { 
  E := (0, 5), 
  F := (5, 5), 
  G := (5, 0), 
  H := (0, 0), 
  side_length := 5, 
  H_side_len := and.intro (by norm_num) 
    (and.intro (by norm_num) 
    (and.intro (by norm_num) 
    (by norm_num))) 
}

-- Midpoint N of segment GH
def N : ℝ × ℝ := midpoint EFGH_square.G EFGH_square.H

-- Circle centered at N with radius 3
def circle_N (x y : ℝ) := (x - N.1)^2 + y^2 = 9

-- Circle centered at E with radius 5
def circle_E (x y : ℝ) := x^2 + (y - EFGH_square.E.2)^2 = 25

-- Define the statement of the proof problem in Lean
theorem distance_from_Q_to_EH :
  ∃ Q : ℝ × ℝ, (Q ∈ (circle_N Q.1 Q.2 : set (ℝ × ℝ))) ∧ (Q ∈ (circle_E Q.1 Q.2 : set (ℝ × ℝ))) ∧ Q.2 = 3 := by
  sorry

end distance_from_Q_to_EH_l293_293697


namespace eval_sum_sqrt_ceil_l293_293045

theorem eval_sum_sqrt_ceil:
  ∀ (x : ℝ), 
  (1 < sqrt 3 ∧ sqrt 3 < 2) ∧
  (5 < sqrt 33 ∧ sqrt 33 < 6) ∧
  (18 < sqrt 333 ∧ sqrt 333 < 19) →
  (⌈ sqrt 3 ⌉ + ⌈ sqrt 33 ⌉ + ⌈ sqrt 333 ⌉ = 27) :=
by
  intro x
  sorry

end eval_sum_sqrt_ceil_l293_293045


namespace shaded_area_is_20_l293_293439

theorem shaded_area_is_20 (large_square_side : ℕ) (num_small_squares : ℕ) 
  (shaded_squares : ℕ) 
  (h1 : large_square_side = 10) (h2 : num_small_squares = 25) 
  (h3 : shaded_squares = 5) : 
  (large_square_side^2 / num_small_squares) * shaded_squares = 20 :=
by
  sorry

end shaded_area_is_20_l293_293439


namespace max_triangle_area_l293_293768

theorem max_triangle_area :
  ∃ a b c : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 1 ≤ b ∧ b ≤ 2 ∧ 2 ≤ c ∧ c ≤ 3 ∧ 
  (a + b > c ∧ a + c > b ∧ b + c > a) ∧ (1 ≤ 0.5 * a * b) := sorry

end max_triangle_area_l293_293768


namespace roll_seven_dice_at_least_one_pair_no_three_l293_293791

noncomputable def roll_seven_dice_probability : ℚ :=
  let total_outcomes := (6^7 : ℚ)
  let one_pair_case := (6 * 21 * 120 : ℚ)
  let two_pairs_case := (15 * 21 * 10 * 24 : ℚ)
  let successful_outcomes := one_pair_case + two_pairs_case
  successful_outcomes / total_outcomes

theorem roll_seven_dice_at_least_one_pair_no_three :
  roll_seven_dice_probability = 315 / 972 :=
by
  unfold roll_seven_dice_probability
  -- detailed steps to show the proof would go here
  sorry

end roll_seven_dice_at_least_one_pair_no_three_l293_293791


namespace max_value_HMMT_l293_293077

theorem max_value_HMMT :
  ∀ (H M T : ℤ), H * M ^ 2 * T = H + 2 * M + T → H * M ^ 2 * T ≤ 8 :=
by
  sorry

end max_value_HMMT_l293_293077


namespace dice_probability_l293_293826

theorem dice_probability :
  let P_even := (1 / 2 : ℝ) in
  let P_odd := (1 / 2 : ℝ) in
  let num_ways := Nat.choose 6 3 in
  let total_prob := num_ways * (P_even ^ 3) * (P_odd ^ 3) in
  total_prob = (5 / 16 : ℝ) :=
by
  have P_even := (1 / 2 : ℝ)
  have P_odd := (1 / 2 : ℝ)
  have num_ways := Nat.choose 6 3
  have total_prob := num_ways * (P_even ^ 3) * (P_odd ^ 3)
  sorry

end dice_probability_l293_293826


namespace complement_A_l293_293104

open Set

variable (A : Set ℝ) (x : ℝ)
def A_def : Set ℝ := { x | x ≥ 1 }

theorem complement_A : Aᶜ = { y | y < 1 } :=
by
  sorry

end complement_A_l293_293104


namespace average_sum_permutations_eq_21_l293_293876

theorem average_sum_permutations_eq_21 :
  let a8_perms := {L : List ℕ // L.nodup ∧ L.perm [1, 2, 3, 4, 5, 6, 7, 8]}
  let sum_fn := λ L : a8_perms, |L.elem 1 - L.elem 2| + |L.elem 3 - L.elem 4| + |L.elem 5 - L.elem 6| + |L.elem 7 - L.elem 8|
  (List.foldl (λ (acc : ℚ) (L : a8_perms), acc + sum_fn L) 0 a8_perms.to_list) / ↑a8_perms.size = (20:ℚ) →
  20 + 1 = 21 :=
by
  intros a8_perms sum_fn h
  -- The proof steps would typically follow here.
  sorry

end average_sum_permutations_eq_21_l293_293876


namespace balance_remove_two_weights_no_balance_remove_two_weights_l293_293407

-- Define the problem conditions and corresponding proof statement.

-- Part (a)
theorem balance_remove_two_weights :
  ∀ (weights : Multiset ℕ), 
    weights = {1, 2, ..., 100} ∧ 
    (∃ (pan1 pan2 : Multiset ℕ), pan1 ∪ pan2 = weights ∧ pan1.sum = pan2.sum) →
      (∃ (pan1' pan2' : Multiset ℕ), (∃ w1 w2 w3 w4, w1 + w2 + w3 + w4 = pan1.sum = pan2.sum) :=
  sorry

-- Part (b)
theorem no_balance_remove_two_weights :
  ∃ (n : ℕ), n > 3 ∧ (∀ (weights : Multiset ℕ), weights =  Multiset.range (n+1) ∧
    (∃ (pan1 pan2 : Multiset ℕ), pan1 ∪ pan2 = weights ∧ pan1.sum = pan2.sum) →
      ¬ (∃ (pan1' pan2' : Multiset ℕ), (∃ w1 w2 w3 w4, w1 + w2 + w3 + w4 = pan1.sum = pan2.sum) :=
  sorry

end balance_remove_two_weights_no_balance_remove_two_weights_l293_293407


namespace triangles_not_congruent_l293_293523

theorem triangles_not_congruent (A B C C_1 : Type) [EuclideanGeometry A B C] [EuclideanGeometry A B C_1] 
  (AB AC : ℝ) (angle_ABC angle_ABC1 : Real) :
  (AB = AB) ∧ (AC = AC) ∧ (angle_ABC = angle_ABC1) ∧ (¬ ∃ (f : EuclideanGeometry A B C ≃ EuclideanGeometry A B C_1), f.isometry) :=
by sorry

end triangles_not_congruent_l293_293523


namespace unique_ordered_quadruples_l293_293868

theorem unique_ordered_quadruples :
  ∃! (a b c d : ℝ), 
    (a + 1) * (d + 1) - b * c ≠ 0 ∧ 
    (by rw [matrix.inv_def, matrix.eq_iff, det_smul, matrix.mul_apply]; 
         exact (∀ i j, a i j * matrix.inv_def ⟨1/(a i j + 1) , 1/(b i j + 1), 1/(c i j + 1), 1/(d i j + 1)⟩ = 1)) :=
sorry

end unique_ordered_quadruples_l293_293868


namespace geometric_sequence_term_207_l293_293195

/-- In a geometric sequence where the first term is 8 and the common ratio is -1,
    prove that the 207th term is 8. -/
theorem geometric_sequence_term_207 :
  let a1 := 8 in
  let r := -1 in
  ∀ (n : ℕ), n = 207 → a1 * r^(n - 1) = 8 :=
by {
  -- Defining the first term and common ratio
  let a1 := 8;
  let r := -1;
  -- Assume n equals 207 and prove the term formula
  intro n h;
  have hn : n = 207 := h;
  rw hn;
  -- Simplify the expression to find the value for the 207th term
  sorry
}

end geometric_sequence_term_207_l293_293195


namespace rectangle_perimeter_l293_293344

open Real

def triangle_DEF_sides : ℝ × ℝ × ℝ := (9, 12, 15) -- sides of the triangle DEF

def rectangle_width : ℝ := 6 -- width of the rectangle

theorem rectangle_perimeter (a b c width : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : width = 6) :
  2 * (54 / width + width) = 30 :=
by
  sorry -- Proof is omitted as required

end rectangle_perimeter_l293_293344


namespace q_range_l293_293658

noncomputable def q (x : ℝ) : ℝ :=
  if Nat.prime (Int.floor x) then x + 2
  else
    let y := (Nat.prime_factors (Int.floor x)).max
    q y + (x + 2 - Int.floor x)

-- Restating the goal using Lean's interval notation and set operations
theorem q_range : Set.Icc 4 10 ∪ Set.Icc 12 15 ∪ {18.01} = Set.Range q :=
sorry

end q_range_l293_293658


namespace quadratic_no_real_roots_l293_293121

variables {p q a b c d : ℝ}

-- Conditions
variables (hp : p > 0) (hq : q > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hpq : p ≠ q)
variables (hgp : a ^ 2 = p * q) (hap : b = p + d) (hap' : c = p + 2 * d) (hq' : q = p + 3 * d)

theorem quadratic_no_real_roots (h : d ≠ 0) :
  let Δ := (2 * a) ^ 2 - 4 * b * c in Δ < 0 :=
by
  sorry

end quadratic_no_real_roots_l293_293121


namespace savings_correct_l293_293713

def income : ℝ := 15000
def ratio_income_exp : ℝ := 15 / 8
def expenditure : ℝ := income * (8 / 15)
def savings : ℝ := income - expenditure

theorem savings_correct : savings = 7000 := by
  sorry

end savings_correct_l293_293713


namespace find_a_l293_293214

-- Define the conditions
def b : ℝ := 7
def c : ℝ := 6
def cos_B_minus_C : ℝ := 71 / 80

-- Define the final statement to be proven
theorem find_a (a : ℝ) (h_b : b = 7) (h_c : c = 6) (h_cos : cos_B_minus_C = 71 / 80) : a = Real.sqrt 55 := 
sorry

end find_a_l293_293214


namespace min_x_y_sum_l293_293904

theorem min_x_y_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/(x+1) + 1/y = 1/2) : x + y ≥ 7 := 
by 
  sorry

end min_x_y_sum_l293_293904


namespace factorize_expression_l293_293851

theorem factorize_expression (x y : ℝ) : x^2 + x * y + x = x * (x + y + 1) := 
by
  sorry

end factorize_expression_l293_293851


namespace slope_angle_line_l293_293296

theorem slope_angle_line (θ : ℝ) (h_θ : θ ∈ (0 : ℝ) ^ 180) 
  (h_line : ∃ x y : ℝ, 2 * x + 2 * y = 1) :
  θ = 135 :=
sorry

end slope_angle_line_l293_293296


namespace length_of_courtyard_l293_293335

-- Define the dimensions and properties of the courtyard and paving stones
def width := 33 / 2
def numPavingStones := 132
def pavingStoneLength := 5 / 2
def pavingStoneWidth := 2

-- Total area covered by paving stones
def totalArea := numPavingStones * (pavingStoneLength * pavingStoneWidth)

-- To prove: Length of the courtyard
theorem length_of_courtyard : totalArea / width = 40 := by
  sorry

end length_of_courtyard_l293_293335


namespace sum_of_geometric_sequence_l293_293088

theorem sum_of_geometric_sequence :
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  let n : ℕ := 8
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 3280 / 6561 :=
by
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  let n : ℕ := 8
  let S_n := a * (1 - r^n) / (1 - r)
  sorry

end sum_of_geometric_sequence_l293_293088


namespace ceil_sqrt_sum_l293_293049

theorem ceil_sqrt_sum : 
  (⌈Real.sqrt 3⌉ = 2) ∧ 
  (⌈Real.sqrt 33⌉ = 6) ∧ 
  (⌈Real.sqrt 333⌉ = 19) → 
  2 + 6 + 19 = 27 :=
by 
  intro h
  cases h with h3 h
  cases h with h33 h333
  rw [h3, h33, h333]
  norm_num

end ceil_sqrt_sum_l293_293049


namespace range_a_l293_293288

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 - |x| + a - 1

-- Theorem to prove the range of a given the intersection condition
theorem range_a (a : ℝ) (h_intersection : ∃ x1 x2 x3 x4 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0 ∧ f x4 a = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) :
  1 < a ∧ a < 5/4 :=
begin
  sorry
end

end range_a_l293_293288


namespace number_of_real_z5_l293_293733

theorem number_of_real_z5 (z : ℂ) (h : z ^ 30 = 1) :
  {z : ℂ | z ^ 30 = 1 ∧ z ^ 5 ∈ ℝ}.to_finset.card = 10 :=
sorry

end number_of_real_z5_l293_293733


namespace distance_downstream_correct_l293_293727

noncomputable def speed_boat_still_water : ℝ := 20
noncomputable def rate_current : ℝ := 5
noncomputable def time_travelled_minutes : ℝ := 12
noncomputable def distance_travelled_downstream : ℝ :=
  let effective_speed := speed_boat_still_water + rate_current
  let time_in_hours := time_travelled_minutes / 60
  effective_speed * time_in_hours

theorem distance_downstream_correct :
  distance_travelled_downstream = 5 := by
  let effective_speed := speed_boat_still_water + rate_current
  let time_in_hours := time_travelled_minutes / 60
  have h1 : effective_speed = 25 := rfl
  have h2 : time_in_hours = 0.2 := by norm_num
  have h3 : 25 * 0.2 = 5 := by norm_num
  show distance_travelled_downstream = 5 from by simp [distance_travelled_downstream, h1, h2, h3]

end distance_downstream_correct_l293_293727


namespace quadrilateral_5_has_inscribed_circle_l293_293101

-- Consider a convex quadrilateral divided into nine smaller quadrilaterals
-- The points of intersection of the segments lie on the diagonals of the quadrilateral
-- Define the necessary geometric setup and the tangential condition

variables {P Q R S T: Type} [MetricSpace P] {A B C D: P} {AC BD: Line P}

-- Define the condition that quadrilaterals 1, 2, 3, and 4 have inscribed circles
def has_inscribed_circle (quad: Quadrilateral) : Prop :=
  quad.opposite_sides_equal

-- The main theorem to prove
theorem quadrilateral_5_has_inscribed_circle 
  (quad1 quad2 quad3 quad4 quad5 : Quadrilateral)
  (h1 : has_inscribed_circle quad1)
  (h2 : has_inscribed_circle quad2)
  (h3 : has_inscribed_circle quad3)
  (h4 : has_inscribed_circle quad4)
  -- Additional geometric relations from the intersections of diagonals
  (intersect_diag_cond : points lie on diagonals) :
  has_inscribed_circle quad5 := 
sorry

end quadrilateral_5_has_inscribed_circle_l293_293101


namespace rectangle_side_length_l293_293632

-- Define the conditions
variable (A B C D E F : Point)
variable (AB : ℝ) (r : ℝ)
variable (L : ℝ)

-- Main theorem statement
theorem rectangle_side_length
  (condition1 : congruent_segments E A D ∧ congruent_segments E F B ∧ congruent_segments E F C)
  (condition2 : AB = 22)
  (condition3 : circumcircle_radius_triangle A F D = 10) :
  side_length B C = 16 :=
sorry

end rectangle_side_length_l293_293632


namespace problem_statement_l293_293212

noncomputable theory

open_locale real

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

def is_cube (A B C D A₁ B₁ C₁ D₁ : Point3D) : Prop :=
-- Definition of a cube goes here (omitted for brevity)
sorry

def angle_between (u v : Point3D) : real.angle :=
-- Definition to calculate the angle (omitted for brevity)
sorry

def not_angle_60_deg (angle : real.angle) : Prop :=
angle ≠ real.angle.of_real (60 * (real.pi / 180))

theorem problem_statement (A B C D A₁ B₁ C₁ D₁ : Point3D)
  (h_cube : is_cube A B C D A₁ B₁ C₁ D₁) :
  not_angle_60_deg (angle_between 
    ⟨D.x - A.x, D.y - A.y, D.z - A.z⟩ 
    ⟨B.x - C.x, B.y - C.y, B.z - C.z⟩) :=
sorry

end problem_statement_l293_293212


namespace gray_area_of_circles_l293_293372

theorem gray_area_of_circles (d : ℝ) (h1 : d = 4) (h2 : ∀ r : ℝ, r = d / 2) (h3 : ∀ R : ℝ, R = 3 * (d / 2)) :
  π * (3 * (d / 2))^2 - π * (d / 2)^2 = 32 * π :=
by
  -- According to h1, diameter of the smaller circle is 4.
  have r_def : d / 2 = 2, from
    calc d / 2 = 4 / 2 : by rw h1
         ... = 2 : by norm_num,
  -- According to h3, radius of the larger circle is 3 times the radius of the smaller circle.
  have R_def : 3 * (d / 2) = 6, from
    calc 3 * (d / 2) = 3 * 2 : by rw r_def
         ... = 6 : by norm_num,
  -- Calculate the area of the larger circle.
  have area_large : π * (3 * (d / 2))^2 = 36 * π, from
    calc π * (3 * (d / 2))^2 = π * 6^2 : by rw R_def
                         ... = 36 * π : by norm_num,
  -- Calculate the area of the smaller circle.
  have area_small : π * (d / 2)^2 = 4 * π, from
    calc π * (d / 2)^2 = π * 2^2 : by rw r_def
                    ... = 4 * π : by norm_num,
  -- Calculate the area of the gray region.
  calc π * (3 * (d / 2))^2 - π * (d / 2)^2 = 36 * π - 4 * π : by rw [area_large, area_small]
                                         ... = 32 * π : by norm_num

end gray_area_of_circles_l293_293372


namespace all_members_real_product_is_2_l293_293480

-- We define the set S_n recursively as described in the problem
noncomputable def S (n : ℕ) : set ℝ :=
  if n = 0 then {-2, 2}
  else { x : ℝ | ∃ y ∈ S (n - 1), x = 2 + sqrt y ∨ x = 2 - sqrt y }

-- Part (a): Proving all members of S_n are real
theorem all_members_real (n : ℕ) (x : ℝ) (hx : x ∈ S n) : Real x :=
by
  sorry

-- Part (b): Proving the product P_n of elements of S_n is 2
noncomputable def product_S_n (n : ℕ) : ℝ :=
  ∏ x in S n, x

theorem product_is_2 (n : ℕ) : product_S_n n = 2 :=
by
  sorry

end all_members_real_product_is_2_l293_293480


namespace greatest_x_value_l293_293606

noncomputable theory

def satisfies_conditions (x : ℤ) : Prop :=
  6.1 * 10^x < 620 ∧ ¬ Nat.Prime (Int.natAbs x)

theorem greatest_x_value :
  ∃ x : ℤ, satisfies_conditions x ∧ ∀ y : ℤ, satisfies_conditions y → y ≤ x ∧ x = 1 :=
by
  sorry

end greatest_x_value_l293_293606


namespace option_A_option_B_option_C_option_D_l293_293456

theorem option_A : (-(-1) : ℤ) ≠ -|(-1 : ℤ)| := by
  sorry

theorem option_B : ((-3)^2 : ℤ) ≠ -(3^2 : ℤ) := by
  sorry

theorem option_C : ((-4)^3 : ℤ) = -(4^3 : ℤ) := by
  sorry

theorem option_D : ((2^2 : ℚ)/3) ≠ ((2/3)^2 : ℚ) := by
  sorry

end option_A_option_B_option_C_option_D_l293_293456


namespace rachel_class_choices_l293_293618

theorem rachel_class_choices : (Nat.choose 8 3) = 56 :=
by
  sorry

end rachel_class_choices_l293_293618


namespace initial_money_l293_293817

-- Definitions in accordance with the conditions
def airplane_cost : ℝ := 4.28
def change_received : ℝ := 0.72

-- Theorem to prove the initial money Adam had
theorem initial_money (airplane_cost : ℝ) (change_received : ℝ) : ℝ :=
  airplane_cost + change_received = 5.00

-- Assertion of the theorem using the given conditions
example : initial_money 4.28 0.72 :=
by
  simp [initial_money, airplane_cost, change_received]
  sorry

end initial_money_l293_293817


namespace initial_markup_percentage_l293_293432

variables (C M : ℝ)

def S1 := C * (1 + M)
def S2 := S1 * 1.25
def S3 := S2 * 0.75

theorem initial_markup_percentage :
  S3 = C * 1.125 → M = 0.2 :=
by
  intros h
  let eq1 := S1
  let eq2 := S2
  let eq3 := S3

  sorry

end initial_markup_percentage_l293_293432


namespace ticket_price_increase_one_day_later_l293_293433

noncomputable def ticket_price : ℝ := 1050
noncomputable def days_before_departure : ℕ := 14
noncomputable def daily_increase_rate : ℝ := 0.05

theorem ticket_price_increase_one_day_later :
  ∀ (price : ℝ) (days : ℕ) (rate : ℝ), price = ticket_price → days = days_before_departure → rate = daily_increase_rate →
  price * rate = 52.50 :=
by
  intros price days rate hprice hdays hrate
  rw [hprice, hrate]
  exact sorry

end ticket_price_increase_one_day_later_l293_293433


namespace third_trial_point_l293_293629

variable (a b : ℝ) (x₁ x₂ x₃ : ℝ)

axiom experimental_range : a = 2 ∧ b = 4
axiom method_0618 : ∀ x1 x2, (x1 = 2 + 0.618 * (4 - 2) ∧ x2 = 2 + (4 - x1)) ∨ 
                              (x1 = (2 + (4 - 3.236)) ∧ x2 = 3.236)
axiom better_result (x₁ x₂ : ℝ) : x₁ > x₂  -- Assuming better means strictly greater

axiom x1_value : x₁ = 3.236 ∨ x₁ = 2.764
axiom x2_value : x₂ = 2.764 ∨ x₂ = 3.236
axiom x3_cases : (x₃ = 4 - 0.618 * (4 - x₁)) ∨ (x₃ = 2 + (4 - x₂))

theorem third_trial_point : x₃ = 3.528 ∨ x₃ = 2.472 :=
by
  sorry

end third_trial_point_l293_293629


namespace max_value_fraction_l293_293098

theorem max_value_fraction (x y z w : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) :
  (xyz(x + y + z + w)) / ((x + y + z)^2 * (y + z + w)^2) ≤ 1 / 4 :=
sorry

end max_value_fraction_l293_293098


namespace sum_of_squares_inequality_l293_293682

theorem sum_of_squares_inequality {n : ℕ} (a : ℕ → ℝ) 
  (h1 : ∀ i j : ℕ, i ≤ j → a i ≥ a j) 
  (h2 : ∀ i : ℕ, i ≤ n → a i ≥ 0) :
  ∑ i in finset.range (n + 1), (a i)^2 
  ≤ (∑ i in finset.range (n + 1), a i / (Real.sqrt i + Real.sqrt (i - 1)))^2 :=
by
  sorry


end sum_of_squares_inequality_l293_293682


namespace rowing_time_from_A_to_B_and_back_l293_293218

-- Define the problem parameters and conditions
def rowing_speed_still_water : ℝ := 5
def distance_AB : ℝ := 12
def stream_speed : ℝ := 1

-- Define the problem to prove
theorem rowing_time_from_A_to_B_and_back :
  let downstream_speed := rowing_speed_still_water + stream_speed
  let upstream_speed := rowing_speed_still_water - stream_speed
  let time_downstream := distance_AB / downstream_speed
  let time_upstream := distance_AB / upstream_speed
  let total_time := time_downstream + time_upstream
  total_time = 5 :=
by
  sorry

end rowing_time_from_A_to_B_and_back_l293_293218


namespace b_n_recurrence_a_n_arithmetic_T_n_sum_l293_293558

theorem b_n_recurrence (S_n : ℕ → ℝ) (b_n : ℕ → ℝ) (n : ℕ) (hn : n > 0)
  (H1 : ∀ n, b_n n = 2 - S_n n)
  (H2 : S_n 1 = b_n 1)
  (H3 : ∀ n, S_n (n + 1) = S_n n + b_n (n + 1)) :
  b_n (n + 1) = (1 / 2) ^ n := sorry

theorem a_n_arithmetic (n : ℕ) (a_n : ℕ → ℕ)
  (H1 : a_n 5 = 11)
  (H2 : a_n 8 = 17) :
  a_n n = 2 * n + 1 := sorry

theorem T_n_sum (n : ℕ) (a_n b_n c_n : ℕ → ℝ) (T_n : ℕ → ℝ)
  (H1 : ∀ n, a_n n = 2 * n + 1)
  (H2 : ∀ n, b_n n = (1 / 2) ^ (n - 1))
  (H3 : ∀ n, c_n n = a_n n * b_n n)
  (H4 : ∀ n, T_n n = ∑ i in range n, c_n i) :
  T_n n = 14 - (2 * n + 3) * (1 / 2) ^ (n - 1) := sorry

end b_n_recurrence_a_n_arithmetic_T_n_sum_l293_293558


namespace find_b_collinear_and_bisects_l293_293650

def a := (5 : ℤ, -3 : ℤ, -6 : ℤ)
def c := (-3 : ℤ, -2 : ℤ, 3 : ℤ)
def b := (1 : ℚ, -12/5 : ℚ, 3/5 : ℚ)

def collinear (a b c : α × α × α) [CommRing α] : Prop :=
  ∃ k : α, b = (a.1 + k * (c.1 - a.1), a.2 + k * (c.2 - a.2), a.3 + k * (c.3 - a.3))

def bisects_angle (a b c : ℚ × ℚ × ℚ) : Prop :=
  let dot_product (x y : ℚ × ℚ × ℚ) := x.1 * y.1 + x.2 * y.2 + x.3 * y.3
  let norm (x : ℚ × ℚ × ℚ) := real.sqrt (dot_product x x)
  (dot_product a b) / (norm a * norm b) = (dot_product b c) / (norm b * norm c)

theorem find_b_collinear_and_bisects :
  collinear a b c ∧ bisects_angle a b c :=
by
  sorry

end find_b_collinear_and_bisects_l293_293650


namespace integral_x_squared_plus_sin_l293_293788

theorem integral_x_squared_plus_sin :
  ∫ x in -1..1, (x^2 + sin x) = (2/3) := by
  sorry

end integral_x_squared_plus_sin_l293_293788


namespace leo_amount_after_settling_debts_l293_293996

theorem leo_amount_after_settling_debts (total_amount : ℝ) (ryan_share : ℝ) (ryan_owes_leo : ℝ) (leo_owes_ryan : ℝ) 
  (h1 : total_amount = 48) 
  (h2 : ryan_share = (2 / 3) * total_amount) 
  (h3 : ryan_owes_leo = 10) 
  (h4 : leo_owes_ryan = 7) : 
  (total_amount - ryan_share) + (ryan_owes_leo - leo_owes_ryan) = 19 :=
by
  sorry

end leo_amount_after_settling_debts_l293_293996


namespace product_of_numbers_l293_293300

variable (x y z : ℝ)

theorem product_of_numbers :
  x + y + z = 36 ∧ x = 3 * (y + z) ∧ y = 6 * z → x * y * z = 268 := 
by
  sorry

end product_of_numbers_l293_293300


namespace solution_set_l293_293550

variable (f : ℝ → ℝ)
variable (h_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f(x₁) < f(x₂))
variable (h_f0 : f 0 = -1)
variable (h_f3 : f 3 = 1)

theorem solution_set : {x : ℝ | |f (x + 1)| < 1} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end solution_set_l293_293550


namespace count_real_z5_of_z30_eq_1_l293_293737

theorem count_real_z5_of_z30_eq_1 : 
  ∃ zs : Finset ℂ, (zs.card = 30) ∧ (∀ z ∈ zs, z ^ 30 = 1) ∧ Finset.card ({z ∈ zs | ∃ r : ℝ, (z ^ 5 : ℂ) = r}) = 12 := 
sorry

end count_real_z5_of_z30_eq_1_l293_293737


namespace subtract_value_l293_293602

theorem subtract_value (N x : ℤ) (h1 : (N - x) / 7 = 7) (h2 : (N - 6) / 8 = 6) : x = 5 := 
by 
  sorry

end subtract_value_l293_293602


namespace prop2_prop4_prop5_l293_293551

variables {m n l : Line} {α β γ : Plane}

-- Proposition 2
theorem prop2 (h1 : α ∥ β) (h2 : α ∩ γ = m) (h3 : β ∩ γ = n) : m ∥ n := sorry

-- Proposition 4
theorem prop4 (h1 : α ∩ β = m) (h2 : m ∥ n) 
             (h3 : n ⊈ α) (h4 : n ⊈ β) : n ∥ α ∧ n ∥ β := sorry

-- Proposition 5
theorem prop5 (h1 : α ∩ β = m) (h2 : β ∩ γ = n) (h3 : α ∩ γ = l)
             (h4 : α ⊥ β) (h5 : α ⊥ γ) (h6 : β ⊥ γ) : m ⊥ n ∧ m ⊥ l ∧ n ⊥ l := sorry

end prop2_prop4_prop5_l293_293551


namespace Problem1_l293_293873

theorem Problem1 (x y : ℝ) (h : x^2 + y^2 = 1) : x^6 + 3*x^2*y^2 + y^6 = 1 := 
by
  sorry

end Problem1_l293_293873


namespace isosceles_triangle_perimeter_l293_293870

-- Define the quadratic equation and its roots
def quadratic_eq (x : ℝ) : Prop := x^2 - 9 * x + 18 = 0

-- Define an isosceles triangle with base a and legs b
structure IsoscelesTriangle :=
(base leg : ℝ)
(perimeter : ℝ := base + 2 * leg)

-- Suppose a and b are roots of the quadratic equation x^2 - 9x + 18 = 0
-- Given the conditions of the problem
theorem isosceles_triangle_perimeter : 
  ∃ (a b : ℝ), quadratic_eq a ∧ quadratic_eq b ∧ ((a + b = 9) ∧ (a * b = 18)) → 
  IsoscelesTriangle.mk a b a.b + 2 * b = 12 :=
sorry

end isosceles_triangle_perimeter_l293_293870


namespace slope_AA_not_one_l293_293019

theorem slope_AA_not_one (p q r s t u : ℝ) (hp : 0 ≤ p) (hq : 0 ≤ q) (hr : 0 ≤ r) (hs : 0 ≤ s) (ht : 0 ≤ t) (hu : 0 ≤ u) :
  p + q ≠ 0 → (q - p) / (q + p) ≠ 1 := 
by 
  intro h1
  have h2 : (q - p) / (q + p) = 1 ↔ q - p = q + p := by field_simp [h1]
  sorry

end slope_AA_not_one_l293_293019


namespace ordering_abc_l293_293655

noncomputable def a : ℝ := Real.sqrt 1.01
noncomputable def b : ℝ := Real.exp 0.01 / 1.01
noncomputable def c : ℝ := Real.log (1.01 * Real.exp 1)

theorem ordering_abc : b < a ∧ a < c := by
  -- Proof of the theorem goes here
  sorry

end ordering_abc_l293_293655


namespace math_proof_problem_l293_293804

-- Define the function and its properties
variable (f : ℝ → ℝ)
axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodicity : ∀ x : ℝ, f (x + 1) = -f x
axiom increasing_on_interval : ∀ x y : ℝ, (-1 ≤ x ∧ x < y ∧ y ≤ 0) → f x < f y

-- Theorem statement expressing the questions and answers
theorem math_proof_problem :
  (∀ x : ℝ, f (x + 2) = f x) ∧
  (∀ x : ℝ, f (1 - x) = f (1 + x)) ∧
  (f 2 = f 0) :=
by
  sorry

end math_proof_problem_l293_293804


namespace expression_result_l293_293829

theorem expression_result :
  (1 * 3 * 5 * 7 * 9 * 10 * 12 * 14 * 16 * 18) / ((5 * 6 * 7 * 8 * 9) ^ 2) = 2 := 
begin
  sorry
end

end expression_result_l293_293829


namespace quadratic_eq_form_l293_293211

theorem quadratic_eq_form (p q : ℚ) (x1 x2 : ℚ) :
  (x1^2 + p * x1 + q = 0) ∧ (x2^2 + p * x2 + q = 0) ∧ irrational x1 ∧ irrational x2 ∧ (x1 = x2^3) →
  (p = 0 ∧ q = 1) :=
begin
  sorry
end

end quadratic_eq_form_l293_293211


namespace correct_transformation_l293_293449

theorem correct_transformation (a b c : ℝ) (h : c ≠ 0) (h1 : a / c = b / c) : a = b :=
by 
  -- Actual proof would go here, but we use sorry for the scaffold.
  sorry

end correct_transformation_l293_293449


namespace sum_of_extrema_l293_293299

theorem sum_of_extrema (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ 9) :
  let y := 2 * sin ((π * x / 6) - (π / 3)) in
  let max_y := 2 in
  let min_y := -sqrt 3 in
  max_y + min_y = 2 - sqrt 3 :=
by
  sorry

end sum_of_extrema_l293_293299


namespace find_d_to_make_divisible_by_11_l293_293363

-- Define that the number is 54321d
def digits_54321d (d : ℕ) : list ℕ := [5, 4, 3, 2, 1, d]

-- Define the sum of digits in odd positions
def sum_digits_odd_positions (d : ℕ) : ℕ :=
  digits_54321d d |>.nth 0 |>.getOrElse 0 +
  digits_54321d d |>.nth 2 |>.getOrElse 0 +
  digits_54321d d |>.nth 4 |>.getOrElse 0

-- Define the sum of digits in even positions
def sum_digits_even_positions (d : ℕ) : ℕ :=
  digits_54321d d |>.nth 1 |>.getOrElse 0 +
  digits_54321d d |>.nth 3 |>.getOrElse 0 +
  digits_54321d d |>.nth 5 |>.getOrElse 0

-- Define the difference condition for divisibility by 11
def divisible_by_11_condition (d : ℕ) : Prop :=
  let difference := (sum_digits_odd_positions d - sum_digits_even_positions d).natAbs
  difference = 0 ∨ difference % 11 = 0

-- The goal is to prove that d = 3 satisfies this condition
theorem find_d_to_make_divisible_by_11 : divisible_by_11_condition 3 :=
by
  sorry

end find_d_to_make_divisible_by_11_l293_293363


namespace train_crossing_time_l293_293574

-- Definitions of the given conditions
def length_of_train : ℝ := 110
def speed_of_train_kmph : ℝ := 72
def length_of_bridge : ℝ := 175

noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * 1000 / 3600

noncomputable def total_distance : ℝ := length_of_train + length_of_bridge

noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross_bridge = 14.25 := 
sorry

end train_crossing_time_l293_293574


namespace inequality_solution_real_roots_range_l293_293930

noncomputable def f (x : ℝ) : ℝ :=
|2 * x - 4| - |x - 3|

theorem inequality_solution :
  ∀ x, f x ≤ 2 → x ∈ Set.Icc (-1 : ℝ) 3 :=
sorry

theorem real_roots_range (k : ℝ) :
  (∃ x, f x = 0) → k ∈ Set.Icc (-1 : ℝ) 3 :=
sorry

end inequality_solution_real_roots_range_l293_293930


namespace max_value_h3_solve_for_h_l293_293114

-- Definition part for conditions
def quadratic_function (h : ℝ) (x : ℝ) : ℝ :=
  -(x - h) ^ 2

-- Part (1): When h = 3, proving the maximum value of the function within 2 ≤ x ≤ 5 is 0.
theorem max_value_h3 : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 5 → quadratic_function 3 x ≤ 0 :=
by
  sorry

-- Part (2): If the maximum value of the function is -1, then the value of h is 6 or 1.
theorem solve_for_h (h : ℝ) : 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 5 → quadratic_function h x ≤ -1) ↔ h = 6 ∨ h = 1 :=
by
  sorry

end max_value_h3_solve_for_h_l293_293114


namespace A_3_2_eq_29_l293_293483

def A : ℕ → ℕ → ℕ
| 0, n     => n + 1
| (m + 1), 0 => A m 1
| (m + 1), (n + 1) => A m (A (m + 1) n)

theorem A_3_2_eq_29 : A 3 2 = 29 := by
  sorry

end A_3_2_eq_29_l293_293483


namespace max_dot_product_l293_293944

variables {a b : EuclideanSpace ℝ (Fin 2)}

theorem max_dot_product
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : inner a b = (‖a‖ * ‖b‖) * real.cos (real.pi / 3))
  (h4 : ‖a - (2 : ℝ) • b‖ = 2) :
  ∃ c, c = 1 ∧ ∀ d, inner a b ≤ d → d ≤ c := 
sorry

end max_dot_product_l293_293944


namespace min_eccentricity_value_l293_293124

-- Given problem conditions encapsulated in definitions
variables (F1 F2 P : Type) (dist : F1 → F2 → ℝ)
variables (e1 e2 : ℝ)

-- Given conditions as axioms
axiom ellipse_hyperbola_common_foci : dist F1 F2 > 0
axiom common_point_p : dist P F2 > dist P F1
axiom eccentricity_ellipse : e1 > 0
axiom eccentricity_hyperbola : e2 > 0
axiom distance_condition : dist P F1 = dist F1 F2

-- The equivalent problem
theorem min_eccentricity_value : 3 / e1 + e2 / 3 = 8 :=
sorry

end min_eccentricity_value_l293_293124


namespace total_students_correct_l293_293722

def students_in_school : ℕ :=
  let students_per_class := 23
  let classes_per_grade := 12
  let grades_per_school := 3
  students_per_class * classes_per_grade * grades_per_school

theorem total_students_correct :
  students_in_school = 828 :=
by
  sorry

end total_students_correct_l293_293722


namespace find_stu_l293_293245

open Complex

theorem find_stu (p q r s t u : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (h1 : p = (q + r) / (s - 3))
  (h2 : q = (p + r) / (t - 3))
  (h3 : r = (p + q) / (u - 3))
  (h4 : s * t + s * u + t * u = 8)
  (h5 : s + t + u = 4) :
  s * t * u = 10 := 
sorry

end find_stu_l293_293245


namespace average_geq_l293_293642

variables (m n : ℕ) (a : fin m → fin n.succ)

/-- The conditions: -/
def valid_set (a : fin m → fin n.succ) : Prop :=
  (∀ i j : fin m, 1 ≤ i.1 ∧ i.1 ≤ j.1 ∧ a i + a j ≤ n → ∃ k : fin m, a i + a j = a k) ∧
  (a eq : fin m → fin n.succ := λ i, a i) ∧
  (∀ i j : fin m, i ≠ j → a i ≠ a j)

/-- The main theorem: -/
theorem average_geq (h1 : 0 < m) (h2 : 0 < n) (hc : valid_set m n a) :
  (∑ i, a i : ℕ ) / m ≥ (n + 1) / 2 := sorry

end average_geq_l293_293642


namespace each_person_tip_l293_293222

-- Definitions based on the conditions
def julie_cost : ℝ := 10
def letitia_cost : ℝ := 20
def anton_cost : ℝ := 30
def tip_rate : ℝ := 0.2

-- Theorem statement
theorem each_person_tip (total_cost := julie_cost + letitia_cost + anton_cost)
 (total_tip := total_cost * tip_rate) :
 (total_tip / 3) = 4 := by
  sorry

end each_person_tip_l293_293222


namespace lines_perpendicular_and_intersect_l293_293945

variable {a b : ℝ}

theorem lines_perpendicular_and_intersect 
  (h_ab_nonzero : a * b ≠ 0)
  (h_orthogonal : a + b = 0) : 
  ∃ p, p ≠ 0 ∧ 
    (∀ x y, x = -y * b^2 → y = 0 → p = (x, y)) ∧ 
    (∀ x y, y = x / a^2 → x = 0 → p = (x, y)) ∧ 
    (∀ x y, x = -y * b^2 ∧ y = x / a^2 → x = 0 ∧ y = 0) := 
sorry

end lines_perpendicular_and_intersect_l293_293945


namespace eight_letter_good_words_count_l293_293842

def good_word (w : List Char) : Bool :=
  (∀ i, i < w.length - 1 → 
    (w[i] = 'A' → ¬ w[i+1] = 'B') ∧
    (w[i] = 'B' → ¬ w[i+1] = 'C') ∧
    (w[i] = 'C' → ¬ w[i+1] = 'D') ∧
    (w[i] = 'D' → ¬ w[i+1] = 'A'))

def count_good_words (n : Nat) : Nat :=
  if n = 0 then 1
  else 4 * 3^(n-1)

theorem eight_letter_good_words_count : count_good_words 8 = 8748 := 
  by 
    sorry

end eight_letter_good_words_count_l293_293842


namespace correct_transformation_l293_293450

theorem correct_transformation (a b c : ℝ) (h : c ≠ 0) (h1 : a / c = b / c) : a = b :=
by 
  -- Actual proof would go here, but we use sorry for the scaffold.
  sorry

end correct_transformation_l293_293450


namespace find_b_l293_293651

open_locale matrix

def a : ℝ^3 := ![5, -3, -6]
def c : ℝ^3 := ![-3, -2, 3]
def b : ℝ^3 := ![-1, -3/4, 3/4]

theorem find_b (h1 : ∃ k : ℝ, b = k • a ∨ b = k • c)
(h2: inner_product_space.angle a b = inner_product_space.angle b c):
  b = ![-1, -3/4, 3/4] :=
sorry

end find_b_l293_293651


namespace find_m_value_l293_293139

def m_value (m : ℝ) : Prop :=
  1 > m ∧ (sqrt (1 - m)) / 1 = (sqrt 3) / 2

theorem find_m_value : ∃ m : ℝ, m_value m ∧ m = 1 / 4 :=
by
  use 1 / 4
  split
  · -- Prove the conditions
    sorry
  · -- Prove m = 1/4
    refl

end find_m_value_l293_293139


namespace multiply_abs_value_l293_293001

theorem multiply_abs_value : -2 * |(-3 : ℤ)| = -6 := by
  sorry

end multiply_abs_value_l293_293001


namespace max_min_magnitude_of_sum_l293_293585

open Real

-- Define the vectors a and b and their magnitudes
variables {a b : ℝ × ℝ}
variable (h_a : ‖a‖ = 5)
variable (h_b : ‖b‖ = 2)

-- Define the constant 7 and 3 for the max and min values
noncomputable def max_magnitude : ℝ := 7
noncomputable def min_magnitude : ℝ := 3

-- State the theorem
theorem max_min_magnitude_of_sum (h_a : ‖a‖ = 5) (h_b : ‖b‖ = 2) :
  ‖a + b‖ ≤ max_magnitude ∧ ‖a + b‖ ≥ min_magnitude :=
by {
  sorry -- Proof goes here
}

end max_min_magnitude_of_sum_l293_293585


namespace parabola_intersection_value_l293_293911

theorem parabola_intersection_value (a : ℝ) (h : a^2 - a - 1 = 0) : a^2 - a + 2014 = 2015 :=
by
  sorry

end parabola_intersection_value_l293_293911


namespace area_of_OAM_l293_293715

-- Define the given parabola and points F and M
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def F : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (0, 1)
def O : ℝ × ℝ := (0, 0)

-- Define the area computation of triangle given vertices O, A, and M
def triangle_area (O A M : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (O.fst * (A.snd - M.snd) + A.fst * (M.snd - O.snd) + M.fst * (O.snd - A.snd))

-- The theorem to be proved
theorem area_of_OAM : 
  ∃ A : ℝ × ℝ, (parabola A.fst A.snd ∧ 
  ∃ k : ℝ, (A.snd = -A.fst + k ∧ 
  k = 1)) ∧ 
  triangle_area O A M = (3 / 2 - real.sqrt 2) :=
sorry

end area_of_OAM_l293_293715


namespace sum_of_15th_set_is_1695_l293_293939

def sum_of_n_th_set (n : ℕ) : ℕ :=
let first_element := 1 + (n * (n - 1)) / 2 in
let last_element := first_element + n - 1 in
(n * (first_element + last_element)) / 2

theorem sum_of_15th_set_is_1695 : sum_of_n_th_set 15 = 1695 :=
by
  -- proof to be provided
  sorry

end sum_of_15th_set_is_1695_l293_293939


namespace exponent_and_root_condition_root_implication_sufficient_but_not_necessary_condition_l293_293785

theorem exponent_and_root_condition (a b : ℝ) : 
  (2 ^ a > 2 ^ b ∧ 2 ^ b > 1) → (a > b → (a > 0 ∧ b > 0)) :=
begin
  sorry
end

theorem root_implication (a b : ℝ) : 
  (a > b → ∃ (c : ℝ), c = 0) :=
begin
  sorry
end

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  ((2 ^ a > 2 ^ b ∧ 2 ^ b > 1) → (a > b)) ∧ ¬ ((a > b) → (2 ^ a > 2 ^ b ∧ 2 ^ b > 1)) :=
begin
  split,
  { intros h h₁,
    sorry
  },
  { intro h,
    sorry
  }
end

end exponent_and_root_condition_root_implication_sufficient_but_not_necessary_condition_l293_293785


namespace average_age_increase_l293_293707

theorem average_age_increase (A : ℝ) :
  (∃ A, (10 * A - 10 - 12 + 21 + 21) / 10 - A = 2) :=
by
  use A
  have h₁ : ((10 * A - 10 - 12 + 21 + 21) / 10 = (10 * A + 20) / 10) := sorry
  have h₂ : (10 * A + 20) / 10 - A = 10 * A / 10 + 20 / 10 - A := sorry
  have h₃ : 10 * A / 10 + 20 / 10 - A = A + 2 - A := sorry
  have h₄ : A + 2 - A = 2 := sorry
  exact (h₁.trans (h₂.trans (h₃.trans h₄)))

end average_age_increase_l293_293707


namespace domain_of_my_function_l293_293486

def my_function (x : ℝ) : ℝ := (x - 2) / (x ^ 2 - 4)

theorem domain_of_my_function :
  {x : ℝ | ∃ y : ℝ, y = my_function x} =
  {x : ℝ | x ≠ 2 ∧ x ≠ -2} :=
sorry

end domain_of_my_function_l293_293486


namespace min_positive_period_of_f_max_value_of_f_min_value_of_f_l293_293567

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + 3 * Real.pi / 4) * cos x

theorem min_positive_period_of_f : ∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi :=
sorry

theorem max_value_of_f :
  ∀ x ∈ Set.Icc (-5 * Real.pi / 4) (5 * Real.pi / 4), f x ≤ 1 + Real.sqrt 3 :=
sorry

theorem min_value_of_f :
  ∀ x ∈ Set.Icc (-5 * Real.pi / 4) (5 * Real.pi / 4), f x ≥ -3 + Real.sqrt 3 :=
sorry

end min_positive_period_of_f_max_value_of_f_min_value_of_f_l293_293567


namespace waiter_customers_l293_293784

variable (initial_customers left_customers new_customers : ℕ)

theorem waiter_customers 
  (h1 : initial_customers = 33)
  (h2 : left_customers = 31)
  (h3 : new_customers = 26) :
  (initial_customers - left_customers + new_customers = 28) := 
by
  sorry

end waiter_customers_l293_293784


namespace count_real_z5_of_z30_eq_1_l293_293740

theorem count_real_z5_of_z30_eq_1 : 
  ∃ zs : Finset ℂ, (zs.card = 30) ∧ (∀ z ∈ zs, z ^ 30 = 1) ∧ Finset.card ({z ∈ zs | ∃ r : ℝ, (z ^ 5 : ℂ) = r}) = 12 := 
sorry

end count_real_z5_of_z30_eq_1_l293_293740


namespace modulus_complex_number_l293_293918

theorem modulus_complex_number :
  let z := (2 * Complex.I) / (1 - Complex.I)
  in Complex.abs z = Real.sqrt 2 :=
sorry

end modulus_complex_number_l293_293918


namespace each_person_tip_l293_293224

-- Definitions based on the conditions
def julie_cost : ℝ := 10
def letitia_cost : ℝ := 20
def anton_cost : ℝ := 30
def tip_rate : ℝ := 0.2

-- Theorem statement
theorem each_person_tip (total_cost := julie_cost + letitia_cost + anton_cost)
 (total_tip := total_cost * tip_rate) :
 (total_tip / 3) = 4 := by
  sorry

end each_person_tip_l293_293224


namespace geometric_sequence_ratio_l293_293725

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (S3 : ℝ) 
  (h1 : a 1 = 1) (h2 : S3 = 3 / 4) 
  (h3 : S3 = a 1 + a 1 * q + a 1 * q^2) :
  q = -1 / 2 := 
by
  sorry

end geometric_sequence_ratio_l293_293725


namespace floor_sqrt_sum_eq_floor_sqrt_sum_two_l293_293683

theorem floor_sqrt_sum_eq_floor_sqrt_sum_two (n : ℕ) : 
  ⌊√n + √(n + 1)⌋ = ⌊√(4 * n + 2)⌋ :=
by sorry

end floor_sqrt_sum_eq_floor_sqrt_sum_two_l293_293683


namespace solution_set_of_inequality_minimum_value_2a_plus_b_l293_293929

noncomputable def f (x : ℝ) : ℝ := x + 1 + |3 - x|

theorem solution_set_of_inequality :
  {x : ℝ | x ≥ -1 ∧ f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem minimum_value_2a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 8 * a * b = a + 2 * b) :
  2 * a + b = 9 / 8 :=
by
  sorry

end solution_set_of_inequality_minimum_value_2a_plus_b_l293_293929


namespace initial_purchase_correct_max_profit_additional_correct_l293_293703

open Real

-- Define the initial conditions and variables
variables (a b : ℕ)
variables (pA pB : ℕ) -- number of additional A and B phones
variables (total_cost total_profit : ℕ)

-- Initial purchase costs and profits
def costA := 3000 * a
def costB := 3500 * b
def revenueA := 3400 * a
def revenueB := 4000 * b
def totalSpent := costA + costB
def totalProfit := (revenueA + revenueB) - totalSpent

-- Additional purchase and profit determination
def additionalPhones : ℕ := 30
def maxProfitAdditional (pA : ℕ) : ℕ := (400 * pA + 500 * (additionalPhones - pA))

-- Inequality constraint for additional purchases
def constraint (pA : ℕ) : Prop := (additionalPhones - pA) ≤ 2 * pA

-- Proof statements
theorem initial_purchase_correct :
  costA + costB = 32000 ∧ (revenueA + revenueB) - (costA + costB) = 4400 →
  a = 6 ∧ b = 4 :=
sorry

theorem max_profit_additional_correct :
  (∃ pA, constraint pA) →
  (∀ pA b, constraint pA → pA = 10 ∧ maxProfitAdditional pA = 14000) := 
sorry

end initial_purchase_correct_max_profit_additional_correct_l293_293703


namespace temp_neg_represents_below_zero_l293_293971

-- Definitions based on the conditions in a)
def above_zero (x: ℤ) : Prop := x > 0
def below_zero (x: ℤ) : Prop := x < 0

-- Proof problem derived from c)
theorem temp_neg_represents_below_zero (t1 t2: ℤ) 
  (h1: above_zero t1) (h2: t1 = 10) 
  (h3: below_zero t2) (h4: t2 = -3) : 
  -t2 = 3 :=
by
  sorry

end temp_neg_represents_below_zero_l293_293971


namespace eval_sum_sqrt_ceil_l293_293046

theorem eval_sum_sqrt_ceil:
  ∀ (x : ℝ), 
  (1 < sqrt 3 ∧ sqrt 3 < 2) ∧
  (5 < sqrt 33 ∧ sqrt 33 < 6) ∧
  (18 < sqrt 333 ∧ sqrt 333 < 19) →
  (⌈ sqrt 3 ⌉ + ⌈ sqrt 33 ⌉ + ⌈ sqrt 333 ⌉ = 27) :=
by
  intro x
  sorry

end eval_sum_sqrt_ceil_l293_293046


namespace option_A_option_B_option_C_option_D_l293_293458

theorem option_A : (-(-1) : ℤ) ≠ -|(-1 : ℤ)| := by
  sorry

theorem option_B : ((-3)^2 : ℤ) ≠ -(3^2 : ℤ) := by
  sorry

theorem option_C : ((-4)^3 : ℤ) = -(4^3 : ℤ) := by
  sorry

theorem option_D : ((2^2 : ℚ)/3) ≠ ((2/3)^2 : ℚ) := by
  sorry

end option_A_option_B_option_C_option_D_l293_293458


namespace profit_percentage_correct_l293_293419

noncomputable theory

-- Given conditions
def SP : ℝ := 850
def profit : ℝ := 255
def salesTaxPercent : ℝ := 0.07
def discountPercent : ℝ := 0.05

-- Definitions based on conditions
def CP := SP - profit
def salesTax := salesTaxPercent * SP
def discount := discountPercent * SP
def ASP := SP - discount
def netAmountReceived := ASP - salesTax
def actualProfit := netAmountReceived - CP
def profitPercentage := (actualProfit / CP) * 100

-- The theorem to prove
theorem profit_percentage_correct : profitPercentage ≈ 25.71 := by
  sorry

end profit_percentage_correct_l293_293419


namespace speed_of_sound_correct_l293_293428

-- Define the given conditions
def heard_second_blast_after : ℕ := 30 * 60 + 24 -- 30 minutes and 24 seconds in seconds
def time_sound_travelled : ℕ := 24 -- The sound traveled for 24 seconds
def distance_travelled : ℕ := 7920 -- Distance in meters

-- Define the expected answer for the speed of sound 
def expected_speed_of_sound : ℕ := 330 -- Speed in meters per second

-- The proposition that states the speed of sound given the conditions
theorem speed_of_sound_correct : (distance_travelled / time_sound_travelled) = expected_speed_of_sound := 
by {
  -- use division to compute the speed of sound
  sorry
}

end speed_of_sound_correct_l293_293428


namespace number_of_real_z5_l293_293734

theorem number_of_real_z5 (z : ℂ) (h : z ^ 30 = 1) :
  {z : ℂ | z ^ 30 = 1 ∧ z ^ 5 ∈ ℝ}.to_finset.card = 10 :=
sorry

end number_of_real_z5_l293_293734


namespace directed_Kn_no_triangles_R3_bound_exists_c_l293_293408

-- Part (a)
theorem directed_Kn_no_triangles (n : ℕ) :
  (1 - 2/8) ^ Nat.choose n 3 = (3/4) ^ Nat.choose n 3 :=
sorry

-- Part (b)
theorem R3_bound_exists_c (c : ℝ) (h₁ : 0 < c) :
  ∃ c, ∀ n : ℕ, R_3(4, n) ≥ 2^(c * n) :=
sorry

end directed_Kn_no_triangles_R3_bound_exists_c_l293_293408


namespace one_odd_one_even_l293_293181

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

theorem one_odd_one_even (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prime : is_prime a) (h_eq : a^2 + b^2 = c^2) : 
(is_odd b ∧ is_even c) ∨ (is_even b ∧ is_odd c) :=
sorry

end one_odd_one_even_l293_293181


namespace thomas_weekly_wage_l293_293333

theorem thomas_weekly_wage (monthly_wage : ℕ) (weeks_in_month : ℕ) (weekly_wage : ℕ) 
    (h1 : monthly_wage = 19500) (h2 : weeks_in_month = 4) :
    weekly_wage = 4875 :=
by
  have h3 : weekly_wage = monthly_wage / weeks_in_month := sorry
  rw [h1, h2] at h3
  exact h3

end thomas_weekly_wage_l293_293333


namespace tray_height_correct_l293_293440

noncomputable def tray_height : ℝ :=
  let side_length := 120
  let wedge_distance := Real.sqrt 20
  let cut_angle := 45
  have sqrt_40 : Real.sqrt (AM^2 + AN^2) = Real.sqrt 40, from sorry
  let AR := sqrt_40
  let height := AR / 2
  Real.sqrt 10

theorem tray_height_correct :
  tray_height = Real.sqrt 10 :=
sorry

end tray_height_correct_l293_293440


namespace P_X_le_0_eq_028_l293_293912

noncomputable def P (X : ℝ → Prop) : ℝ := sorry

def normal_distribution (μ σ : ℝ) (X : ℝ) : Prop := sorry

def P_X_le_2 : ℝ := 0.72

theorem P_X_le_0_eq_028 (σ : ℝ) :
  (∀ X, normal_distribution 1 σ X → P (λ t, t ≤ 2) = P_X_le_2) →
  P (λ t, t ≤ 0) = 0.28 :=
sorry

end P_X_le_0_eq_028_l293_293912


namespace AM_GM_inequality_example_l293_293120

theorem AM_GM_inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 6) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 1 / 2 :=
by
  sorry

end AM_GM_inequality_example_l293_293120


namespace JerryAge_l293_293670

-- Given definitions
def MickeysAge : ℕ := 20
def AgeRelationship (M J : ℕ) : Prop := M = 2 * J + 10

-- Proof statement
theorem JerryAge : ∃ J : ℕ, AgeRelationship MickeysAge J ∧ J = 5 :=
by
  sorry

end JerryAge_l293_293670


namespace sum_of_rationals_l293_293254

-- Let a1, a2, a3, a4 be 4 rational numbers such that the set of products of distinct pairs is given.
def valid_products (a1 a2 a3 a4 : ℚ) : Prop :=
  {a1 * a2, a1 * a3, a1 * a4, a2 * a3, a2 * a4, a3 * a4} = {-24, -2, -3/2, -1/8, 1, 3}

-- Define the theorem which asserts the sum of these rational numbers is either 9/4 or -9/4.
theorem sum_of_rationals (a1 a2 a3 a4 : ℚ) (h : valid_products a1 a2 a3 a4) :
  a1 + a2 + a3 + a4 = 9/4 ∨ a1 + a2 + a3 + a4 = -9/4 :=
sorry

end sum_of_rationals_l293_293254


namespace sundae_cost_l293_293581

theorem sundae_cost (ice_cream_cost toppings_cost : ℕ) (num_toppings : ℕ) :
  ice_cream_cost = 200  →
  toppings_cost = 50 →
  num_toppings = 10 →
  ice_cream_cost + num_toppings * toppings_cost = 700 := by
  sorry

end sundae_cost_l293_293581


namespace midpoint_of_intersection_l293_293934

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 2 * t)

noncomputable def polar_curve (θ : ℝ) : ℝ :=
  2 / Real.sqrt (1 + 3 * Real.sin θ ^ 2)

theorem midpoint_of_intersection :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, 
      A = parametric_line t₁ ∧ 
      B = parametric_line t₂ ∧ 
      (A.1 ^ 2 / 4 + A.2 ^ 2 = 1) ∧ 
      (B.1 ^ 2 / 4 + B.2 ^ 2 = 1)) ∧
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (4 / 5, -1 / 5) :=
sorry

end midpoint_of_intersection_l293_293934


namespace quadratic_solution_trig_expression_value_l293_293786

-- Define the quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 4 * x + 3 = 0

-- Define the trigonometric values
def sin30 := 1 / 2
def cos45 := Real.sqrt 2 / 2
def tan60 := Real.sqrt 3

-- Proof statements
theorem quadratic_solution :
  {x : ℝ | quadratic_eq x} = {1, 3} :=
by sorry

theorem trig_expression_value :
  4 * sin30 - Real.sqrt 2 * cos45 + Real.sqrt 3 * tan60 = 4 :=
by sorry

end quadratic_solution_trig_expression_value_l293_293786


namespace column_product_2014_l293_293107

theorem column_product_2014
  (a : Fin 2014 → ℝ) (b : Fin 2014 → ℝ)
  (h_distinct_a : Function.Injective a)
  (h_distinct_b : Function.Injective b)
  (h_row_product : ∀ i : Fin 2014, ∏ j : Fin 2014, (a i + b j) = 2014)
  : ∀ j : Fin 2014, ∏ i : Fin 2014, (a i + b j) = 2014 :=
by sorry

end column_product_2014_l293_293107


namespace family_eggs_count_l293_293420

theorem family_eggs_count : 
  ∀ (initial_eggs parent_use child_use : ℝ) (chicken1 chicken2 chicken3 chicken4 : ℝ), 
    initial_eggs = 25 →
    parent_use = 7.5 + 2.5 →
    chicken1 = 2.5 →
    chicken2 = 3 →
    chicken3 = 4.5 →
    chicken4 = 1 →
    child_use = 1.5 + 0.5 →
    (initial_eggs - parent_use + (chicken1 + chicken2 + chicken3 + chicken4) - child_use) = 24 :=
by
  intros initial_eggs parent_use child_use chicken1 chicken2 chicken3 chicken4 
         h_initial_eggs h_parent_use h_chicken1 h_chicken2 h_chicken3 h_chicken4 h_child_use
  -- Proof goes here
  sorry

end family_eggs_count_l293_293420


namespace identify_heaviest_and_lightest_coin_within_13_weighings_l293_293319

-- Lean 4 statement to encapsulate the given problem
theorem identify_heaviest_and_lightest_coin_within_13_weighings :
  ∃ (weighings: list (ℕ × ℕ)) (heaviest lightest: ℕ),
    (length weighings ≤ 13) ∧
    (∀ i j, 1 ≤ i ∧ i ≤ 10 → 1 ≤ j ∧ j ≤ 10 → i ≠ j) ∧
    (∀ (comp: ℕ × ℕ), comp ∈ weighings → comp.1 ≠ comp.2 ∧ 1 ≤ comp.1 ∧ comp.1 ≤ 10 ∧ 1 ≤ comp.2 ∧ comp.2 ≤ 10) ∧
    heaviest ≠ lightest ∧
    (∀ (i: ℕ), 1 ≤ i ∧ i ≤ 10 → 
      (i = heaviest ∨ i = lightest))
: sorry

end identify_heaviest_and_lightest_coin_within_13_weighings_l293_293319


namespace hex_prism_paintings_l293_293431

def num_paintings : ℕ :=
  -- The total number of distinct ways to paint a hex prism according to the conditions
  3 -- Two colors case: white-red, white-blue, red-blue
  + 6 -- Three colors with pattern 121213
  + 1 -- Three colors with identical opposite faces: 123123
  + 3 -- Three colors with non-identical opposite faces: 123213

theorem hex_prism_paintings : num_paintings = 13 := by
  sorry

end hex_prism_paintings_l293_293431


namespace intersection_coordinates_moving_point_polar_eq_l293_293968

-- Part 1: Intersection coordinates of C1 and C2
theorem intersection_coordinates (theta ρ : ℝ) (h1 : ρ * cos theta = 3) (h2 : ρ = 4 * cos theta) 
  (h_theta_range : 0 ≤ theta ∧ theta < π / 2) : 
  ρ = 2 * sqrt 3 ∧ theta = π / 6 :=
by
  sorry

-- Part 2: Polar coordinate equation of moving point P
theorem moving_point_polar_eq (ρ θ ρ₀ θ₀ : ℝ) 
  (hQ_on_C2 : ρ₀ = 4 * cos θ₀) (hθ₀_range : 0 ≤ θ₀ ∧ θ₀ < π / 2) 
  (h_OQ_QP : ρ₀ = (2 / 5) * ρ) (h_theta_equality : θ₀ = θ) :
  (ρ = 10 * cos θ ∧ 0 ≤ θ ∧ θ < π / 2) :=
by
  sorry

end intersection_coordinates_moving_point_polar_eq_l293_293968


namespace line_through_point_equal_intercepts_l293_293508

-- Definitions based on conditions
def passes_through (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l p.1 p.2

def equal_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a, a ≠ 0 ∧ (∀ x y, l x y ↔ x + y = a) ∨ (∀ x y, l x y ↔ y = 2 * x)

-- Theorem statement based on the problem
theorem line_through_point_equal_intercepts :
  ∃ l, passes_through (1, 2) l ∧ equal_intercepts l ∧
  (∀ x y, l x y ↔ 2 * x - y = 0) ∨ (∀ x y, l x y ↔ x + y - 3 = 0) :=
sorry

end line_through_point_equal_intercepts_l293_293508


namespace simplify_expression_l293_293285

theorem simplify_expression (z : ℝ) : (7 - real.sqrt (z^2 - 49))^2 = z^2 - 14 * real.sqrt (z^2 - 49) :=
by sorry

end simplify_expression_l293_293285


namespace probability_of_shaded_triangle_l293_293196

theorem probability_of_shaded_triangle 
  (triangles : Finset ℝ) 
  (shaded_triangles : Finset ℝ)
  (h1 : triangles = {1, 2, 3, 4, 5})
  (h2 : shaded_triangles = {1, 4})
  : (shaded_triangles.card / triangles.card) = 2 / 5 := 
  by
  sorry

end probability_of_shaded_triangle_l293_293196


namespace binomial_expansion_properties_l293_293177

-- Define the binomial coefficient
def binomial_coefficient (n k : ℕ) := nat.choose n k

-- State the problem using the provided conditions and the derived answers
theorem binomial_expansion_properties (x : ℝ) (n : ℕ) 
  (h : 2 * binomial_coefficient n 2 = binomial_coefficient n 1 + binomial_coefficient n 3) :
  n = 7 ∧ (∀ r : ℕ, r ≠ 3 ∧ r ≠ 4 → (binomial_coefficient 7 r * (x ^ ((7 - 2 * r) / 6)) ≠ 1)) :=
by
  sorry

end binomial_expansion_properties_l293_293177


namespace vectors_dot_product_l293_293171

noncomputable def dot_product (a b : ℝ) (θ : ℝ) : ℝ :=
  a * b * Real.cos θ

theorem vectors_dot_product :
  ∀ (a b : ℝ) (θ : ℝ), a = 1 ∧ b = sqrt 2 ∧ θ = Real.pi / 3 →
  dot_product a b θ = sqrt 2 / 2 :=
by
  intros a b θ h
  cases h with ha h
  cases h with hb hθ
  simp [dot_product, ha, hb, hθ]
  sorry

end vectors_dot_product_l293_293171


namespace average_age_is_23_l293_293762

-- Define the conditions
def N : ℕ := 11
def captain_age : ℕ := 25
def wicket_keeper_age : ℕ := 30

-- Youngest player condition
lemma youngest_player_age (youngest_player_age : ℕ) : youngest_player_age ≤ 15 :=
by sorry

-- Average age condition
lemma remaining_avg_age (A : ℕ) : (captain_age + wicket_keeper_age + (A - 1) * (N - 2)) / N = A :=
by sorry

-- Theorem statement
theorem average_age_is_23 : ∃ A : ℕ, A = 23 :=
by {
  use 23,
  have := remaining_avg_age 23,
  sorry
}

end average_age_is_23_l293_293762


namespace range_of_a_l293_293531

def f (a x : ℝ) : ℝ :=
if x ≤ 1 then a ^ x + 1 else 2 * x ^ 2 - (a + 1) * x + 5

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → (x1 - x2) * (f a x1 - f a x2) > 0) ↔ (1 < a ∧ a ≤ 2.5) :=
by sorry

end range_of_a_l293_293531


namespace max_sqrt_sum_l293_293093

theorem max_sqrt_sum (x : ℝ) (hx : -49 ≤ x ∧ x ≤ 49) : 
  sqrt (49 + x) + sqrt (49 - x) ≤ 14 :=
sorry

end max_sqrt_sum_l293_293093


namespace coefficient_of_x3_l293_293280

noncomputable def coefficient_of_x3_in_expansion : ℚ :=
  let poly := (1 - Polynomial.X)^5 * (1 + Polynomial.X)^3
  poly.coeff 3

theorem coefficient_of_x3 : coefficient_of_x3_in_expansion = 6 :=
  sorry

end coefficient_of_x3_l293_293280


namespace max_value_of_a_l293_293143

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 9/5 → sqrt (2 * x) - a ≥ sqrt (9 - 5 * x)) → a ≤ -3 :=
by
  intros h
  have h_ineq : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 9/5 → sqrt (2 * x) - sqrt (9 - 5 * x) ≤ -3 :=
  sorry
  sorry

end max_value_of_a_l293_293143


namespace problem_statement_l293_293338

noncomputable def perimeter_rectangle 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (area_triangle : ℝ := (1/2) * a * b) 
  (area_rectangle : ℝ := area_triangle) 
  (l : ℝ := area_rectangle / w) : ℝ :=
2 * (w + l)

theorem problem_statement 
  (a b c w : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h_a : a = 9) 
  (h_b : b = 12) 
  (h_c : c = 15) 
  (h_w : w = 6) : 
  perimeter_rectangle a b c w h1 = 30 :=
by 
  sorry

end problem_statement_l293_293338


namespace cindy_correct_answer_l293_293003

theorem cindy_correct_answer (x : ℤ) (h : (x - 7) / 5 = 37) : (x - 5) / 7 = 26 :=
sorry

end cindy_correct_answer_l293_293003


namespace shaded_area_correct_l293_293209

-- Define the conditions
def UV := 5
def VW := 5
def WX := 5
def XY := 5
def YZ := 5
def UZ := UV + VW + WX + XY + YZ

-- Function to calculate the area of a semicircle given the diameter
def semicircleArea (d : ℝ) : ℝ :=
  (1 / 8) * Real.pi * d^2

-- The shaded area calculation
def shaded_area : ℝ :=
  semicircleArea UZ + semicircleArea UV

-- Lean statement to assert the correctness of the area calculation
theorem shaded_area_correct : shaded_area = (325 / 4) * Real.pi :=
by
  -- Remove the proof with sorry
  sorry

end shaded_area_correct_l293_293209


namespace min_value_m_l293_293205

open scoped Real Nat

noncomputable def a_n (n : ℕ) : ℝ := 4 * n - 3

noncomputable def seq_recip_sum (n : ℕ) : ℝ :=
  ∑ i in range n, 1 / a_n (i + 1)

theorem min_value_m (m : ℕ) :
  (∀ n : ℕ+, seq_recip_sum (2 * n + 1) - seq_recip_sum n ≤ m / 15) → m ≥ 5 :=
sorry

end min_value_m_l293_293205


namespace area_ratio_is_1_l293_293966

noncomputable theory

open Nat
open Real

def isosceles_right_triangle (A B C : Point) : Prop :=
  ∠B = π / 2 ∧ dist A B = dist A C

def midpoint (D E : Point) (A B : Point) : Prop :=
  dist D A = dist D B ∧ dist E A = dist E C

def intersection (X : Point) (C D E B : Point) : Prop :=
  collinear X C D ∧ collinear X B E

theorem area_ratio_is_1 :
  ∀ (A B C D E X : Point), isosceles_right_triangle A B C ∧
                           dist A B = 10 ∧
                           dist A C = 10 ∧
                           midpoint D B A ∧
                           midpoint E C A ∧
                           intersection X C D E B →
  area (polygon A E X D) / area (triangle B X C) = 1 := by
  intros A B C D E X h
  sorry

end area_ratio_is_1_l293_293966


namespace min_packs_to_buy_120_cans_l293_293274

/-- Prove that the minimum number of packs needed to buy exactly 120 cans of soda,
with packs available in sizes of 8, 15, and 30 cans, is 4. -/
theorem min_packs_to_buy_120_cans : 
  ∃ n, n = 4 ∧ ∀ x y z: ℕ, 8 * x + 15 * y + 30 * z = 120 → x + y + z ≥ n :=
sorry

end min_packs_to_buy_120_cans_l293_293274


namespace split_tips_evenly_l293_293229

theorem split_tips_evenly :
  let julie_cost := 10
  let letitia_cost := 20
  let anton_cost := 30
  let total_cost := julie_cost + letitia_cost + anton_cost
  let tip_rate := 0.2
  let total_tip := total_cost * tip_rate
  let tip_per_person := total_tip / 3
  tip_per_person = 4 := by
  sorry

end split_tips_evenly_l293_293229


namespace binomial_expansion_coeff_x3_l293_293281

theorem binomial_expansion_coeff_x3 :
  let T := λ (n k : ℕ) (a b : ℝ), (nat.choose n k) * (a ^ (n - k)) * (b ^ k)
  in T 7 3 (sqrt 3) (-2) * (-1) ^ 3 = -2520 :=
begin
  -- The proof will go here, but currently it is omitted.
  sorry
end

end binomial_expansion_coeff_x3_l293_293281


namespace students_in_either_but_not_both_l293_293848

-- Definitions and conditions
def both : ℕ := 18
def geom : ℕ := 35
def only_stats : ℕ := 16

-- Correct answer to prove
def total_not_both : ℕ := geom - both + only_stats

theorem students_in_either_but_not_both : total_not_both = 33 := by
  sorry

end students_in_either_but_not_both_l293_293848


namespace rectangle_perimeter_l293_293342

open Real

def triangle_DEF_sides : ℝ × ℝ × ℝ := (9, 12, 15) -- sides of the triangle DEF

def rectangle_width : ℝ := 6 -- width of the rectangle

theorem rectangle_perimeter (a b c width : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : width = 6) :
  2 * (54 / width + width) = 30 :=
by
  sorry -- Proof is omitted as required

end rectangle_perimeter_l293_293342


namespace average_sine_l293_293264

theorem average_sine (sums : ℕ → ℝ)
  (h₀ : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 90 → sums k = (2 * k) * Real.sin (2 * k * Real.pi / 180)) :
  (∑ k in Finset.range 91 \ Finset.range 1, sums k) / 90 = Real.cot (Real.pi / 180) :=
by
  sorry

end average_sine_l293_293264


namespace tan_B_values_count_l293_293232

theorem tan_B_values_count :
  ∀ (A B C : ℝ), (A + B + C = π) →
  let tanA := Real.tan A
  let tanB := Real.tan B
  let tanC := Real.tan C
  geometric_sequence : tanB^2 = tanA * tanC →
  1 ≤ tanA + tanB + tanC ∧ tanA + tanB + tanC ≤ 2015 →
  ∃ b, b ∈ ({2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} : Set ℕ) ∧
       (tanB = ↑b)  :=
sorry

end tan_B_values_count_l293_293232


namespace count_valid_quadratic_polynomials_l293_293578

theorem count_valid_quadratic_polynomials :
  ∃ (count : ℕ), count = 4 ∧
  ∀ (a b c : ℝ) (r s : ℝ), a ≠ 0 →
    ({a, b, c} = {r, s}) →
    (r + s = -b/a ∧ rs = c/a ∧
    ∃ (valid_polynomials : List (ℝ × ℝ × ℝ)), valid_polynomials.length = count) := by
  sorry

end count_valid_quadratic_polynomials_l293_293578


namespace evaluate_ceiling_sums_l293_293033

theorem evaluate_ceiling_sums : 
  (⌈real.sqrt 3⌉ + ⌈real.sqrt 33⌉ + ⌈real.sqrt 333⌉) = 27 :=
by
  have h1 : 1 < real.sqrt 3 ∧ real.sqrt 3 < 2 :=
    ⟨by norm_num, by norm_num⟩,
  have h2 : 5 < real.sqrt 33 ∧ real.sqrt 33 < 6 :=
    ⟨by norm_num, by norm_num⟩,
  have h3 : 18 < real.sqrt 333 ∧ real.sqrt 333 < 19 :=
    ⟨by norm_num, by norm_num⟩,
  sorry

end evaluate_ceiling_sums_l293_293033


namespace diameter_le_midsegment_length_l293_293687

-- Defining the necessary entities and conditions
noncomputable def circumscribed_quadrilateral (A B C D M N : Point) (ω : Circle) (r : ℝ) : Prop :=
∀ (p : ℝ), touching_circle_circumscribed_abcd A B C D ω ∧
  midpoint B C M ∧ midpoint A D N ∧ radius ω = r ∧
  area_quadrilateral A B C D = p * r

-- The theorem to prove
theorem diameter_le_midsegment_length (A B C D M N : Point) (ω : Circle) (r : ℝ) (h : circumscribed_quadrilateral A B C D M N ω r) : 
  segment_length M N ≥ 2 * r :=
sorry

end diameter_le_midsegment_length_l293_293687


namespace right_triangle_median_hypotenuse_distance_l293_293620

noncomputable def distance_to_midpoint (F DE DF EF : ℝ) : ℝ :=
  let M := DE / 2
  M

theorem right_triangle_median_hypotenuse_distance
  (DE DF EF : ℝ)
  (h : DE^2 = DF^2 + EF^2)
  (hypDE : DE = 15)
  (hypDF : DF = 9)
  (hypEF : EF = 12) : distance_to_midpoint 0 DE DF EF = 7.5 :=
by {
  have hyp1 : DE = 15 := hypDE,
  have hyp2 : DF = 9 := hypDF,
  have hyp3 : EF = 12 := hypEF,
  have pythagorean : (15 : ℝ)^2 = (9 : ℝ)^2 + (12 : ℝ)^2 := h,
  -- We know by the median to hypotenuse property in right triangle
  rw [distance_to_midpoint, hyp1],
  norm_num,
  sorry
}

end right_triangle_median_hypotenuse_distance_l293_293620


namespace contradict_D_l293_293134

-- Define the basic entities: a plane and two lines
axiom Plane : Type
axiom Line : Type
axiom α : Plane
axiom m : Line
axiom n : Line

-- The conditions provided in the problem
axiom perpendicular : Line → Plane → Prop
axiom parallel : Line → Line → Prop
axiom parallelToPlane : Line → Plane → Prop

-- Given conditions
axiom m_perp_alpha : perpendicular m α

-- The statement to show that the given condition contradicts the statement (D)
theorem contradict_D (H : perpendicular m n) : ¬ (parallelToPlane n α) := 
  sorry

end contradict_D_l293_293134


namespace ceil_sqrt_sum_l293_293051

theorem ceil_sqrt_sum : 
  (⌈Real.sqrt 3⌉ = 2) ∧ 
  (⌈Real.sqrt 33⌉ = 6) ∧ 
  (⌈Real.sqrt 333⌉ = 19) → 
  2 + 6 + 19 = 27 :=
by 
  intro h
  cases h with h3 h
  cases h with h33 h333
  rw [h3, h33, h333]
  norm_num

end ceil_sqrt_sum_l293_293051


namespace range_of_a_l293_293122

def p (a : ℝ) : Prop := a > -1
def q (a : ℝ) : Prop := ∀ m : ℝ, -2 ≤ m ∧ m ≤ 4 → a^2 - a ≥ 4 - m

theorem range_of_a (a : ℝ) : (p a ∧ ¬q a) ∨ (¬p a ∧ q a) ↔ (-1 < a ∧ a < 3) ∨ a ≤ -2 := by
  sorry

end range_of_a_l293_293122


namespace compound_interest_after_two_years_l293_293469

-- Define the conditions given in the problem
def principal : ℝ := 5000
def annual_interest_rate : ℝ := 0.10
def compounds_per_year : ℕ := 1
def investment_duration : ℝ := 2

-- Statement to prove: the amount A after two years is $6050
theorem compound_interest_after_two_years :
  let A := principal * (1 + annual_interest_rate / compounds_per_year)^(compounds_per_year * investment_duration)
  in A = 6050 :=
by
  sorry

end compound_interest_after_two_years_l293_293469


namespace avg_three_numbers_l293_293753

theorem avg_three_numbers (A B C : ℝ) 
  (h1 : A + B = 53)
  (h2 : B + C = 69)
  (h3 : A + C = 58) : 
  (A + B + C) / 3 = 30 := 
by
  sorry

end avg_three_numbers_l293_293753


namespace number_of_real_z5_l293_293736

theorem number_of_real_z5 (z : ℂ) (h : z ^ 30 = 1) :
  {z : ℂ | z ^ 30 = 1 ∧ z ^ 5 ∈ ℝ}.to_finset.card = 10 :=
sorry

end number_of_real_z5_l293_293736


namespace trigonometry_problem_l293_293886

noncomputable def f (a b x : ℝ) : ℝ := a * (Real.sin x) ^ 3 + b * Real.tan x + 1

theorem trigonometry_problem (a b : ℝ) (h : f a b 2 = 3) : f a b (2 * Real.pi - 2) = -1 :=
by
  unfold f at *
  sorry

end trigonometry_problem_l293_293886


namespace exists_nat_lt_100_two_different_squares_l293_293490

theorem exists_nat_lt_100_two_different_squares :
  ∃ n : ℕ, n < 100 ∧ 
    ∃ a b c d : ℕ, a^2 + b^2 = n ∧ c^2 + d^2 = n ∧ (a ≠ c ∨ b ≠ d) ∧ a ≠ b ∧ c ≠ d :=
by
  sorry

end exists_nat_lt_100_two_different_squares_l293_293490


namespace find_m_n_sum_l293_293976

variables {X Y Z D M P : Type}
variables {XY XZ : ℝ}
variables {YZ YD DZ ZP PX : ℝ}
variables (m n : ℕ)

-- Conditions
axiom tri_XYZ : ∃ triangle : X → Y → Z, true
axiom len_XY : XY = 15
axiom len_XZ : XZ = 9
axiom angle_bisector_intersection : ∃ D, ∃ YD DZ : ℝ, ZP / PX = 8 / 5
axiom midpoint_M : ∃ M, ∃ XD XD' : ℝ, true

-- Problem (final goal)
theorem find_m_n_sum : m + n = 13 :=
sorry

end find_m_n_sum_l293_293976


namespace triangle_angles_median_bisector_altitude_l293_293216

theorem triangle_angles_median_bisector_altitude {α β γ : ℝ} 
  (h : α + β + γ = 180) 
  (median_angle_condition : α / 4 + β / 4 + γ / 4 = 45) -- Derived from 90/4 = 22.5
  (median_from_C : 4 * α = γ) -- Given condition that angle is divided into 4 equal parts
  (median_angle_C : γ = 90) -- Derived that angle @ C must be right angle (90°)
  (sum_angles_C : α + β = 90) : 
  α = 22.5 ∧ β = 67.5 ∧ γ = 90 :=
by
  sorry

end triangle_angles_median_bisector_altitude_l293_293216


namespace initial_tax_rate_l293_293184

variable (R : ℝ)

theorem initial_tax_rate
  (income : ℝ := 48000)
  (new_rate : ℝ := 0.30)
  (savings : ℝ := 7200)
  (tax_savings : income * (R / 100) - income * new_rate = savings) :
  R = 45 := by
  sorry

end initial_tax_rate_l293_293184


namespace find_minimum_value_max_value_when_g_half_l293_293073

noncomputable def f (a x : ℝ) : ℝ := 1 - 2 * a - 2 * a * (Real.cos x) - 2 * (Real.sin x) ^ 2

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a <= 2 then -a^2 / 2 - 2 * a - 1
  else 1 - 4 * a

theorem find_minimum_value (a : ℝ) :
  ∃ g_val, g_val = g a :=
  sorry

theorem max_value_when_g_half : 
  g (-1) = 1 / 2 →
  ∃ max_val, max_val = (max (f (-1) π) (f (-1) 0)) :=
  sorry

end find_minimum_value_max_value_when_g_half_l293_293073


namespace rectangle_perimeter_is_30_l293_293352

noncomputable def triangle_DEF_sides := (9 : ℕ, 12 : ℕ, 15 : ℕ)
noncomputable def rectangle_width := (6 : ℕ)

theorem rectangle_perimeter_is_30 :
  let area_triangle_DEF := (triangle_DEF_sides.1 * triangle_DEF_sides.2) / 2
  let rectangle_length := area_triangle_DEF / rectangle_width
  let rectangle_perimeter := 2 * (rectangle_width + rectangle_length)
  rectangle_perimeter = 30 := by
  sorry

end rectangle_perimeter_is_30_l293_293352


namespace equilateral_triangle_100_degrees_invalid_l293_293393

theorem equilateral_triangle_100_degrees_invalid :
  ¬ ∃ (T : Triangle), T.isEquilateral ∧ (∃ (A : ℝ), A = 100 ∧ T.hasAngle A) :=
by
  sorry

end equilateral_triangle_100_degrees_invalid_l293_293393


namespace num_real_values_equal_roots_l293_293513

theorem num_real_values_equal_roots : 
  {p : ℝ // ∃ h : (root1 = root2), x^2 - p * x + p^2 = 0 } = 1 :=
sorry

end num_real_values_equal_roots_l293_293513


namespace num_distinct_ellipses_l293_293023

theorem num_distinct_ellipses : 
  let S := {-3, -2, -1, 1, 2, 3}
  -- We define the set of all triples of distinct elements from S
  let triples := {t : S × S × S // t.1 ≠ t.2 ∧ t.2 ≠ t.3 ∧ t.1 ≠ t.3}
  -- We define the condition for an ellipse: -c/a > 0 and -c/b > 0
  let is_ellipse (t : S × S × S) : Prop := 
    let a := t.1
    let b := t.2
    let c := t.3
    (-(c.toInt / a.toInt) > 0) ∧ (-(c.toInt / b.toInt) > 0)
  -- We count the number of distinct triples that satisfy the ellipse condition
  (triples.filter (λ t, is_ellipse t)).card = 18 :=
begin
  sorry
end

end num_distinct_ellipses_l293_293023


namespace number_of_knights_l293_293675

-- Definitions for knights and liars
constant Inhabitant : Type
constant is_knight : Inhabitant → Prop
constant is_liar : Inhabitant → Prop

-- Conditions
constant total_inhabitants : ℕ := 1001
constant stance : (Inhabitant → ℕ) → Prop

-- Assume each inhabitant is either a knight or a liar
axiom honesty_axiom : ∀ x : Inhabitant, is_knight x ∨ is_liar x

-- Each person says "The ten people following me are all liars"
axiom statement_axiom : ∀ x : Inhabitant, 
                         stance (λ y : Inhabitant, is_liar y) x ↔ 
                         (∀ y : Inhabitant, y ∈ following_ten x → is_liar y)

-- Knight always tells the truth
axiom knight_truth : ∀ x : Inhabitant, is_knight x → (statement_axiom x)

-- Liar always lies
axiom liar_falsehood : ∀ x : Inhabitant, is_liar x → ¬ (statement_axiom x)

-- Define the following_ten (circular list)
noncomputable def following_ten (x : Inhabitant) : List Inhabitant := sorry

-- Proof statement
theorem number_of_knights : 
  ∃ knights : ℕ, 
  knights = 91 := 
sorry

end number_of_knights_l293_293675


namespace find_angle_A_find_sides_b_c_l293_293901

variable {a b c A B C : ℝ}

theorem find_angle_A 
  (h1 : c = sqrt 3 * a * sin C - c * cos A) 
  : A = π / 3 := sorry

theorem find_sides_b_c 
  (ha : a = 2) 
  (area : (1 / 2) * b * c * sin A = sqrt 3) 
  (angle_A : A = π / 3) 
  : b = 2 ∧ c = 2 := sorry

end find_angle_A_find_sides_b_c_l293_293901


namespace find_heaviest_and_lightest_l293_293325

-- Definition of the main problem conditions
def coins : ℕ := 10
def max_weighings : ℕ := 13
def distinct_weights (c : ℕ) : Prop := ∀ (i j : ℕ), i ≠ j → i < c → j < c → weight i ≠ weight j

-- Noncomputed property representing the weight of each coin
noncomputable def weight : ℕ → ℝ := sorry

-- The main theorem statement
theorem find_heaviest_and_lightest (c : ℕ) (mw : ℕ) (dw : distinct_weights c) : c = coins ∧ mw = max_weighings
  → ∃ (h l : ℕ), h < c ∧ l < c ∧ (∀ (i : ℕ), i < c → weight i ≤ weight h ∧ weight i ≥ weight l) :=
by
  sorry

end find_heaviest_and_lightest_l293_293325


namespace cubic_polynomial_root_properties_l293_293160

theorem cubic_polynomial_root_properties
  (a b : ℝ) (h_a_neg : a < 0) (h_b_pos : b > 0)
  (h_roots : ∃ (x1 x2 x3 : ℝ), x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ (x1 + x2 + x3 = 1) ∧ (x1 * x2 + x2 * x3 + x3 * x1 = -a) ∧ (x1 * x2 * x3 = b)) :
  ∃ (y1 : ℝ) (y2 y3 : ℂ), y1 > 0 ∧ complex.conjugate y2 y3 ∧ (y1 + (y2 + y3).re = 1) ∧ (y1 * y2.re + y2.re * y3.re + y3.re * y1 = b) ∧ (y1 * y2 * y3.re = -a) := 
sorry

end cubic_polynomial_root_properties_l293_293160


namespace proof1_proof2_l293_293665

-- Definitions based on the conditions given in the problem description.

def f1 (x : ℝ) (a : ℝ) : ℝ := abs (3 * x - 1) + a * x + 3

theorem proof1 (x : ℝ) : (f1 x 1 ≤ 4) ↔ 0 ≤ x ∧ x ≤ 1 / 2 := 
by
  sorry

theorem proof2 (a : ℝ) : (-3 ≤ a ∧ a ≤ 3) ↔ 
  ∃ (x : ℝ), ∀ y : ℝ, f1 x a ≤ f1 y a := 
by
  sorry

end proof1_proof2_l293_293665


namespace train_crossing_time_l293_293445

variable (a b v : ℝ)

theorem train_crossing_time (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < v) : 
  (a + b) / v = (∃ t : ℝ, t = (a + b) / v) :=
by sorry

end train_crossing_time_l293_293445


namespace volume_of_cylindrical_pool_l293_293830

-- Define the conditions
def diameter : ℝ := 20
def depth : ℝ := 5

-- Define the radius based on the diameter
def radius : ℝ := diameter / 2

-- Define the volume formula for a cylinder
def volume (r h : ℝ) : ℝ := π * r^2 * h

-- State the theorem (problem) that needs to be proven
theorem volume_of_cylindrical_pool : volume radius depth = 500 * π :=
by
  sorry

end volume_of_cylindrical_pool_l293_293830


namespace unique_solution_log_l293_293405

noncomputable def unique_solution_eq (a : ℝ) :=
  ∀ x : ℝ, a^x = real.log x / real.log a → ∃! x, a^x = x

theorem unique_solution_log {a : ℝ} (h : 1 < a) :
  unique_solution_eq a ↔ a = real.exp (1 / real.exp 1) :=
by
  sorry

end unique_solution_log_l293_293405


namespace A_n_eq_B_n_l293_293646

open Real

noncomputable def A_n (n : ℕ) : ℝ :=
  1408 * (1 - (1 / (2 : ℝ) ^ n))

noncomputable def B_n (n : ℕ) : ℝ :=
  (3968 / 3) * (1 - (1 / (-2 : ℝ) ^ n))

theorem A_n_eq_B_n : A_n 5 = B_n 5 := sorry

end A_n_eq_B_n_l293_293646


namespace sammy_lawn_mowing_time_l293_293692

theorem sammy_lawn_mowing_time :
  ∃ S : ℝ, let combined_time := 1.71428571429,
           let laura_time := 4,
           let sammy_rate := 1 / S,
           let laura_rate := 1 / laura_time,
           let combined_rate := 1 / combined_time in
  combined_rate = sammy_rate + laura_rate → S ≈ 3 :=
begin
  sorry
end

end sammy_lawn_mowing_time_l293_293692


namespace number_of_possible_committees_l293_293471

-- Define the problem statement and conditions
def num_professors_per_department : ℕ := 6
def num_departments : ℕ := 3
def num_male_professors_per_department : ℕ := 3
def num_female_professors_per_department : ℕ := 3
def committee_size : ℕ := 6
def num_men_in_committee : ℕ := 3
def num_women_in_committee : ℕ := 3

-- Define a senior professor condition
def has_senior_professor (professors : list ℕ) : Prop := 
  ∃ senior, senior ∈ professors

-- Formalize the main theorem statement
theorem number_of_possible_committees :
  ∃ committee : set ℕ,
    (committee.card = committee_size) ∧ 
    (committee.filter (λ prof, prof < num_male_professors_per_department * num_departments)).card = num_men_in_committee ∧
    (committee.filter (λ prof, prof ≥ num_male_professors_per_department * num_departments)).card = num_women_in_committee ∧
    (∀ department, (∃ prof in committee, prof / num_professors_per_department = department) ∧ 
                   (committee.filter (λ prof, prof / num_professors_per_department = department)).card ≥ 2) ∧
    has_senior_professor (committee.to_list) →
  ∃ n, (n = 972) :=
begin
  sorry
end

end number_of_possible_committees_l293_293471


namespace regular_price_of_shrimp_l293_293422

theorem regular_price_of_shrimp 
  (discounted_price : ℝ) 
  (discount_rate : ℝ) 
  (quarter_pound_price : ℝ) 
  (full_pound_price : ℝ) 
  (price_relation : quarter_pound_price = discounted_price * (1 - discount_rate) / 4) 
  (discounted_value : quarter_pound_price = 2) 
  (given_discount_rate : discount_rate = 0.6) 
  (given_discounted_price : discounted_price = full_pound_price) 
  : full_pound_price = 20 :=
by {
  sorry
}

end regular_price_of_shrimp_l293_293422


namespace triangle_solution_l293_293958

noncomputable def solve_triangle (a b : ℝ) (A : ℝ) : ℝ × ℝ × ℝ :=
  if B = 60 then (60, 90, 12) else (120, 30, 6)

theorem triangle_solution :
  ∃ B C c, (solve_triangle 6 (6 * Real.sqrt 3) 30 = (B, C, c)) ∧
           ((B = 60 ∧ C = 90 ∧ c = 12) ∨ (B = 120 ∧ C = 30 ∧ c = 6)) :=
by sorry

end triangle_solution_l293_293958


namespace evaluate_expression_at_values_l293_293379

theorem evaluate_expression_at_values :
  let x := 2
  let y := -1
  let z := 3
  2 * x^2 + 3 * y^2 - 4 * z^2 + 5 * x * y = -35 := by
    sorry

end evaluate_expression_at_values_l293_293379


namespace rationalizing_denominator_l293_293689

noncomputable def a := Real.root 4 4
noncomputable def b := Real.root 4 2

/-- Rationalizing the denominator of the given expression and computing the sum of the components of the simplified numerator and denominator. -/
theorem rationalizing_denominator :
  let X := 64
      Y := 32
      Z := 16
      W := 8
      D := 14 in
  X + Y + Z + W + D = 134 :=
by
  sorry

end rationalizing_denominator_l293_293689


namespace min_dist_of_PQ_l293_293547

open Real

theorem min_dist_of_PQ :
  ∀ (P Q : ℝ × ℝ),
    (P.fst - 3)^2 + (P.snd + 1)^2 = 4 →
    Q.fst = -3 →
    ∃ (min_dist : ℝ), min_dist = 4 :=
by
  sorry

end min_dist_of_PQ_l293_293547


namespace identify_heaviest_and_lightest_coin_within_13_weighings_l293_293321

-- Lean 4 statement to encapsulate the given problem
theorem identify_heaviest_and_lightest_coin_within_13_weighings :
  ∃ (weighings: list (ℕ × ℕ)) (heaviest lightest: ℕ),
    (length weighings ≤ 13) ∧
    (∀ i j, 1 ≤ i ∧ i ≤ 10 → 1 ≤ j ∧ j ≤ 10 → i ≠ j) ∧
    (∀ (comp: ℕ × ℕ), comp ∈ weighings → comp.1 ≠ comp.2 ∧ 1 ≤ comp.1 ∧ comp.1 ≤ 10 ∧ 1 ≤ comp.2 ∧ comp.2 ≤ 10) ∧
    heaviest ≠ lightest ∧
    (∀ (i: ℕ), 1 ≤ i ∧ i ≤ 10 → 
      (i = heaviest ∨ i = lightest))
: sorry

end identify_heaviest_and_lightest_coin_within_13_weighings_l293_293321


namespace angle_YZX_l293_293007

theorem angle_YZX {A B C : Type} {γ : Circle} {X Y Z : Point} (h_incircle : γ.incircle A B C) 
  (h_circumcircle : γ.circumcircle X Y Z) (hX : X ∈ segment B C) (hY : Y ∈ segment A B) 
  (hZ : Z ∈ segment A C) (angle_A : angle A = 50) (angle_B : angle B = 70) (angle_C : angle C = 60) : 
  angle (segment Y Z) (segment Z X) = 65 := 
sorry

end angle_YZX_l293_293007


namespace divisors_not_divisible_by_3_l293_293947

theorem divisors_not_divisible_by_3 (a b c d : ℕ) (h_a : a ≤ 1) (h_b : b ≤ 1) (h_c : c ≤ 1) (h_d : d ≤ 1) :
  let num_divisors := 2 * 2 * 2
  in (∃ x : ℕ, x = 2 ^ a * 3 ^ b * 5 ^ c * 7 ^ d ∧ x ∣ 210 ∧ b = 0) → num_divisors = 8 :=
by {
  sorry
}

end divisors_not_divisible_by_3_l293_293947


namespace call_center_agents_ratio_l293_293413

noncomputable def fraction_of_agents (calls_A calls_B total_agents total_calls : ℕ) : ℚ :=
  let calls_A_per_agent := calls_A / total_agents
  let calls_B_per_agent := calls_B / total_agents
  let ratio_calls_A_B := (3: ℚ) / 5
  let fraction_calls_B := (8: ℚ) / 11
  let fraction_calls_A := (3: ℚ) / 11
  let ratio_of_agents := (5: ℚ) / 11
  if (calls_A_per_agent * fraction_calls_A = ratio_calls_A_B * calls_B_per_agent) then ratio_of_agents else 0

theorem call_center_agents_ratio (calls_A calls_B total_agents total_calls agents_A agents_B : ℕ) :
  (calls_A : ℚ) / (calls_B : ℚ) = (3 / 5) →
  (calls_B : ℚ) = (8 / 11) * total_calls →
  (agents_A : ℚ) = (5 / 11) * (agents_B : ℚ) :=
sorry

end call_center_agents_ratio_l293_293413


namespace sum_of_geometric_sequence_l293_293089

theorem sum_of_geometric_sequence :
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  let n : ℕ := 8
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 3280 / 6561 :=
by
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  let n : ℕ := 8
  let S_n := a * (1 - r^n) / (1 - r)
  sorry

end sum_of_geometric_sequence_l293_293089


namespace cycle_length_top_card_after_74_shuffles_l293_293849

-- Define the initial sequence of cards
def initial_deck : List Char := ['A', 'B', 'C', 'D', 'E']

-- Define the shuffling operation
def shuffle_once (deck : List Char) : List Char :=
match deck with
| (x::y::rest) => rest ++ [y, x]
| _ => deck -- This case handles the degenerate case, though it's unnecessary with our constraints

-- Compute the number of shuffles needed to return to the initial configuration is 6
theorem cycle_length : ∀ (deck : List Char), cycle_shuffles deck 6 = initial_deck :=
sorry

-- Prove that the top card after 74 shuffles is 'E'
theorem top_card_after_74_shuffles : (shuffle_n_times 74 initial_deck).head = 'E' :=
sorry

end cycle_length_top_card_after_74_shuffles_l293_293849


namespace max_area_of_triangle_l293_293895

-- Define the problem conditions and the maximum area S
theorem max_area_of_triangle
  (A B C : ℝ)
  (a b c S : ℝ)
  (h1 : 4 * S = a^2 - (b - c)^2)
  (h2 : b + c = 8) :
  S ≤ 8 :=
sorry

end max_area_of_triangle_l293_293895


namespace find_a8_of_geometric_sequence_l293_293931

noncomputable def is_geometric (a: ℕ → ℝ) : Prop := ∀ n : ℕ, a(n+2) = a(n+1)^2 / a(n)

theorem find_a8_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : is_geometric a)
  (h_a4 : a 4 = 7)
  (h_a6 : a 6 = 21) :
  a 8 = 63 :=
by
  sorry

end find_a8_of_geometric_sequence_l293_293931


namespace stacy_berries_l293_293276

theorem stacy_berries (total_berries : ℕ) 
  (sylar_berries : ℕ) (stacy_to_steve : ℕ → ℕ) (steve_to_sylar : ℕ → ℕ) :
  total_berries = 1100 ∧ stacy_to_steve (steve_to_sylar sylar_berries) = 8 * sylar_berries ∧ stacy_to_steve = (λ n, 4 * n) ∧ steve_to_sylar = (λ n, 2 * n) →
  stacy_to_steve (steve_to_sylar sylar_berries) = 800 :=
by
  sorry

end stacy_berries_l293_293276


namespace right_angled_triangle_lines_l293_293189

theorem right_angled_triangle_lines (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y + 4 = 0 → x - 2 * y + 5 = 0 → m * x - 3 * y + 12 = 0 → 
    (exists x₁ y₁ : ℝ, 2 * x₁ - 1 * y₁ + 4 = 0 ∧ (x₁ - 5) ^ 2 / 4 + y₁ / (4) = (2^(1/2))^2) ∨ 
    (exists x₂ y₂ : ℝ, 1/2 * x₂ * y₂ - y₂ / 3 + 1 / 6 = 0 ∧ (x₂ + 5) ^ 2 / 9 + y₂ / 4 = small)) → 
    (m = -3 / 2 ∨ m = -6) :=
sorry

end right_angled_triangle_lines_l293_293189


namespace probability_no_less_than_one_meter_l293_293890

theorem probability_no_less_than_one_meter (length_of_rope : ℝ) (cut_position : ℝ) :
    length_of_rope = 3 ∧ 0 ≤ cut_position ∧ cut_position ≤ length_of_rope →
    (∃ probability, probability = 1 / 3) :=
begin
    sorry,
end

end probability_no_less_than_one_meter_l293_293890


namespace gray_area_of_circles_l293_293370

theorem gray_area_of_circles (d : ℝ) (h1 : d = 4) (h2 : ∀ r : ℝ, r = d / 2) (h3 : ∀ R : ℝ, R = 3 * (d / 2)) :
  π * (3 * (d / 2))^2 - π * (d / 2)^2 = 32 * π :=
by
  -- According to h1, diameter of the smaller circle is 4.
  have r_def : d / 2 = 2, from
    calc d / 2 = 4 / 2 : by rw h1
         ... = 2 : by norm_num,
  -- According to h3, radius of the larger circle is 3 times the radius of the smaller circle.
  have R_def : 3 * (d / 2) = 6, from
    calc 3 * (d / 2) = 3 * 2 : by rw r_def
         ... = 6 : by norm_num,
  -- Calculate the area of the larger circle.
  have area_large : π * (3 * (d / 2))^2 = 36 * π, from
    calc π * (3 * (d / 2))^2 = π * 6^2 : by rw R_def
                         ... = 36 * π : by norm_num,
  -- Calculate the area of the smaller circle.
  have area_small : π * (d / 2)^2 = 4 * π, from
    calc π * (d / 2)^2 = π * 2^2 : by rw r_def
                    ... = 4 * π : by norm_num,
  -- Calculate the area of the gray region.
  calc π * (3 * (d / 2))^2 - π * (d / 2)^2 = 36 * π - 4 * π : by rw [area_large, area_small]
                                         ... = 32 * π : by norm_num

end gray_area_of_circles_l293_293370


namespace sin_cos_value_l293_293128

theorem sin_cos_value (α : ℝ) (h : sin (3 * π - α) = -2 * sin (π / 2 + α)) : 
  sin α * cos α = 2 / 5 ∨ sin α * cos α = -2 / 5 :=
by
  sorry

end sin_cos_value_l293_293128


namespace cosine_sum_identity_l293_293129

variable (A B C : ℝ)

theorem cosine_sum_identity (h : tan (A / 2) * tan (B / 2) + tan (B / 2) * tan (C / 2) + tan (A / 2) * tan (C / 2) = 1) :
  cos (A + B + C) = -1 :=
by
  sorry

end cosine_sum_identity_l293_293129


namespace problem_l293_293922

variables {a_n : ℕ → ℝ} {a : ℝ} {m : ℝ} {A B : ℝ} {f : ℝ → ℝ}
def is_geometric (s : ℕ → ℝ) := ∃ q : ℝ, ∀ n, s (n+1) = q * s n
def is_pairwise_geometric (s : ℕ → ℝ) := ∃ q : ℝ, ∀ n, s (n+1) * s (n+2) = q * s n * s (n+1)

theorem problem (P1 P4 : Prop) : 
  ((is_geometric a_n → is_pairwise_geometric a_n) ∧ 
  (¬(is_pairwise_geometric a_n → is_geometric a_n)) → P1) ∧
  ((∃ x, x ≠ 2 ∧ ∀ z, ((z ≥ 2 → z - x ≥ 0) ∨ (z ≤ 2 → z - x ≤ 0)) → ¬ (x = 2)) ∧
  (∃ x, ∀ z, x = 2 → (z ≥ 2 → f z = z - x) → ¬ (f z = z - a)) → ¬ P1)  ∧
  ((∃ n, ((n+3)*x + n*y - 2 = 0) ∧ (n*x - 6*y + 5 = 0) ∧ n ≠ 3) → ¬ P1) ∧
  ((1 = a ∧ b = sqrt 3 ∧ A = 30 ∧ (B = 60 → 1/sin A = sqrt 3/sin B) → P4) ∧
  ((0 < B < 120)  ∧ ((B ≠ 60 ∧ A = 30) → false) → P4) := sorry

end problem_l293_293922


namespace gray_area_of_circles_l293_293371

theorem gray_area_of_circles (d : ℝ) (h1 : d = 4) (h2 : ∀ r : ℝ, r = d / 2) (h3 : ∀ R : ℝ, R = 3 * (d / 2)) :
  π * (3 * (d / 2))^2 - π * (d / 2)^2 = 32 * π :=
by
  -- According to h1, diameter of the smaller circle is 4.
  have r_def : d / 2 = 2, from
    calc d / 2 = 4 / 2 : by rw h1
         ... = 2 : by norm_num,
  -- According to h3, radius of the larger circle is 3 times the radius of the smaller circle.
  have R_def : 3 * (d / 2) = 6, from
    calc 3 * (d / 2) = 3 * 2 : by rw r_def
         ... = 6 : by norm_num,
  -- Calculate the area of the larger circle.
  have area_large : π * (3 * (d / 2))^2 = 36 * π, from
    calc π * (3 * (d / 2))^2 = π * 6^2 : by rw R_def
                         ... = 36 * π : by norm_num,
  -- Calculate the area of the smaller circle.
  have area_small : π * (d / 2)^2 = 4 * π, from
    calc π * (d / 2)^2 = π * 2^2 : by rw r_def
                    ... = 4 * π : by norm_num,
  -- Calculate the area of the gray region.
  calc π * (3 * (d / 2))^2 - π * (d / 2)^2 = 36 * π - 4 * π : by rw [area_large, area_small]
                                         ... = 32 * π : by norm_num

end gray_area_of_circles_l293_293371


namespace integer_solutions_l293_293498

theorem integer_solutions (x y z : ℤ) : 
  x + y + z = 3 ∧ x^3 + y^3 + z^3 = 3 ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 4 ∧ y = 4 ∧ z = -5) ∨
  (x = 4 ∧ y = -5 ∧ z = 4) ∨
  (x = -5 ∧ y = 4 ∧ z = 4) := 
sorry

end integer_solutions_l293_293498


namespace binom_18_9_l293_293125

theorem binom_18_9 :
  (binom 16 7 = 11440) →
  (binom 16 8 = 12870) →
  (binom 16 9 = 11440) →
  binom 18 9 = 48620 := by
  sorry

end binom_18_9_l293_293125


namespace work_completed_in_days_l293_293454

def total_days_to_complete_work (amit_days : ℕ) (ananthu_days : ℕ) (amit_worked_days : ℕ) : ℕ :=
  let amit_rate := 1 / amit_days
  let ananthu_rate := 1 / ananthu_days
  let amit_work := amit_rate * amit_worked_days
  let remaining_work := 1 - amit_work
  let ananthu_worked_days := remaining_work / ananthu_rate
  amit_worked_days + ananthu_worked_days

theorem work_completed_in_days : total_days_to_complete_work 10 20 2 = 18 :=
by
  simp [total_days_to_complete_work]
  let amit_rate := 1 / 10
  let ananthu_rate := 1 / 20
  let amit_work := amit_rate * 2
  let remaining_work := 1 - amit_work
  let ananthu_worked_days := remaining_work / ananthu_rate
  have h : amit_work = 1 / 5 := by simp [amit_rate]
  have h2 : remaining_work = 4 / 5 := by simp [remaining_work, h]
  have h3 : ananthu_worked_days = 16 := by simp [ananthu_worked_days, ananthu_rate]
  exact calc
    2 + ananthu_worked_days = 2 + 16 : by simp [h3]
    ... = 18 : by simp

end work_completed_in_days_l293_293454


namespace ceil_sqrt_sum_l293_293048

theorem ceil_sqrt_sum : 
  (⌈Real.sqrt 3⌉ = 2) ∧ 
  (⌈Real.sqrt 33⌉ = 6) ∧ 
  (⌈Real.sqrt 333⌉ = 19) → 
  2 + 6 + 19 = 27 :=
by 
  intro h
  cases h with h3 h
  cases h with h33 h333
  rw [h3, h33, h333]
  norm_num

end ceil_sqrt_sum_l293_293048


namespace measure_YZX_l293_293011

-- Define the triangle ABC with given angles
axiom triangle_ABC (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) : Prop :=
  angle_A = 50 ∧ angle_B = 70 ∧ angle_C = 60 ∧ angle_A + angle_B + angle_C = 180

-- Define the points X, Y, Z on the sides of triangle ABC
axiom points_on_sides (X Y Z : Type) (B C A B' C' A' : Type) : Prop :=
  X ∈ (B' ∩ C') ∧ Y ∈ (A' ∩ B) ∧ Z ∈ (A ∩ C')

-- Define the circle Gamma as incircle and circumcircle of triangles
axiom circle_incircle_circumcircle (Gamma : Type) (triangle_ABC triangle_XYZ : Type) : Prop :=
  incircle Gamma triangle_ABC ∧ circumcircle Gamma triangle_XYZ

-- Define the measure of YZX angle
def angle_YZX (angle_B angle_C : ℝ) : ℝ :=
  180 - (angle_B / 2 + angle_C / 2)

-- The main theorem to prove
theorem measure_YZX : ∀ (A B C X Y Z : Type) (Gamma : Type),
    triangle_ABC A B C 50 70 60 →
    points_on_sides X Y Z B C A B C A →
    circle_incircle_circumcircle Gamma (triangle_ABC A B C) (triangle_XYZ X Y Z) →
    angle_YZX 70 60 = 115 := 
by 
  intros A B C X Y Z Gamma hABC hXYZ hGamma
  sorry

end measure_YZX_l293_293011


namespace sin_alpha_value_l293_293559

noncomputable def sin_alpha_through_point (x y : ℤ) (r : ℝ) (α : ℝ) : ℝ :=
  if x = 2 ∧ y = -3 ∧ r = Real.sqrt (x^2 + y^2) then -3 * Real.sqrt 13 / 13 else 0

theorem sin_alpha_value :
  let x := 2
  let y := -3
  let r := Real.sqrt (x^2 + y^2)
  let α := Real.arcsin (y / r)
  sin_alpha_through_point x y r α = -3 * Real.sqrt 13 / 13 := 
by
  sorry

end sin_alpha_value_l293_293559


namespace pythagorean_special_case_l293_293213

-- Definitions of points and the triangle geometry
variable {α : Type*} [EuclideanGeometry α]

-- Definitions of right triangle vertices and the right angle at B
variables (A B C M N O : α)
variables (hABC : Triangle ABC) (h_right : ∠ABC = 90) (h_midpoint : midpoint O A C)
variables (hM : segment AB M) (hN : segment BC N) (h_right_MON : ∠MON = 90)

-- The proof goal
theorem pythagorean_special_case : AM ^ 2 + CN ^ 2 = MN ^ 2 :=
sorry

end pythagorean_special_case_l293_293213


namespace parallel_lines_slope_l293_293169

theorem parallel_lines_slope {a : ℝ} 
    (line1 : ∀ (x : ℝ), (y : ℝ) → y = a * x - 2) 
    (line2 : ∀ (x : ℝ), (y : ℝ) → y = (2 - a) * x + 1) 
    (parallel : ∀ (x : ℝ), (y1 y2 : ℝ), line1 x y1 → line2 x y2 → (1 : ℝ) = a / (2 - a)) :
  a = 1 :=
  by
    sorry

end parallel_lines_slope_l293_293169


namespace projection_magnitude_correct_l293_293571

variables (a : ℝ × ℝ) (b : ℝ × ℝ) 

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def projection_magnitude (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (norm b)

theorem projection_magnitude_correct (a : ℝ × ℝ) (b : ℝ × ℝ) (ha : a = (2, 3)) (hb : b = (-2, 1)) :
  projection_magnitude a b = -Real.sqrt(5) / 5 :=
  sorry

end projection_magnitude_correct_l293_293571


namespace sum_of_solutions_quadratic_eq_l293_293375

theorem sum_of_solutions_quadratic_eq : 
  (∑ x in {x | x^2 = 10 * x - 24}, x) = 10 :=
by
  sorry

end sum_of_solutions_quadratic_eq_l293_293375


namespace correct_statements_l293_293481

variables (L m n : Type) [LinearOrder L] [LinearOrder m] [LinearOrder n]
variables (α β γ : Type) [Plane α] [Plane β] [Plane γ]
variables (L_parallel_m : is_parallel L m) (m_perp_alpha : is_perpendicular m α)
variables (m_parallel_alpha : is_parallel m α)
variables (alpha_inter_beta : α ∩ β = L) (beta_inter_gamma : β ∩ γ = m) (gamma_inter_alpha : γ ∩ α = n)

theorem correct_statements :
  (∀ L m α, (is_parallel L m) ∧ (is_perpendicular m α) → (is_perpendicular L α)) ∧
  (¬ ∀ L m α, (is_parallel L m) ∧ (is_parallel m α) → (is_parallel L α)) ∧
  (¬ ∀ α β γ L m n, (α ∩ β = L) ∧ (β ∩ γ = m) ∧ (γ ∩ α = n) → (is_parallel L m) ∧ (is_parallel m n)) := sorry

end correct_statements_l293_293481


namespace new_salary_l293_293466

theorem new_salary (increase : ℝ) (percent_increase : ℝ) (S_new : ℝ) :
  increase = 25000 → percent_increase = 38.46153846153846 → S_new = 90000 :=
by
  sorry

end new_salary_l293_293466


namespace intersection_A_B_l293_293938

open Set

def A : Set ℤ := {x : ℤ | ∃ y : ℝ, y = Real.sqrt (1 - (x : ℝ)^2)}
def B : Set ℤ := {y : ℤ | ∃ x : ℤ, x ∈ A ∧ y = 2 * x - 1}

theorem intersection_A_B : A ∩ B = {-1, 1} := 
by {
  sorry
}

end intersection_A_B_l293_293938


namespace men_sent_to_other_project_l293_293634

-- Let the initial number of men be 50
def initial_men : ℕ := 50
-- Let the time to complete the work initially be 10 days
def initial_days : ℕ := 10
-- Calculate the total work in man-days
def total_work : ℕ := initial_men * initial_days

-- Let the total time taken after sending some men to another project be 30 days
def new_days : ℕ := 30
-- Let the number of men sent to another project be x
variable (x : ℕ)
-- Let the new number of men be (initial_men - x)
def new_men : ℕ := initial_men - x

theorem men_sent_to_other_project (x : ℕ):
total_work = new_men x * new_days -> x = 33 :=
by
  sorry

end men_sent_to_other_project_l293_293634


namespace number_of_real_z5_l293_293735

theorem number_of_real_z5 (z : ℂ) (h : z ^ 30 = 1) :
  {z : ℂ | z ^ 30 = 1 ∧ z ^ 5 ∈ ℝ}.to_finset.card = 10 :=
sorry

end number_of_real_z5_l293_293735


namespace pythagorean_triple_correct_l293_293775

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_correct : 
  is_pythagorean_triple 9 12 15 ∧ ¬ is_pythagorean_triple 3 4 6 ∧ ¬ is_pythagorean_triple 1 2 3 ∧ ¬ is_pythagorean_triple 6 12 13 :=
by
  sorry

end pythagorean_triple_correct_l293_293775


namespace prob_only_one_passes_l293_293698

open Probability

axiom prob_A : ℝ
axiom prob_B : ℝ
axiom prob_A_val : prob_A = 1 / 2
axiom prob_B_val : prob_B = 1 / 3
axiom independent_events : is_independent prob_A prob_B

noncomputable def prob_C := 
  let prob_not_B := 1 - prob_B
  let prob_not_A := 1 - prob_A
  prob_A * prob_not_B + prob_not_A * prob_B

theorem prob_only_one_passes :
  prob_C = 1 / 2 :=
sorry

end prob_only_one_passes_l293_293698


namespace total_lake_glacial_monoliths_is_80_l293_293465

noncomputable def totalMonoliths := 143
def probSandyLoam := 2 / 11
def probMarineLoam := 7 / 13

def numSandyLoamMonoliths := totalMonoliths * probSandyLoam
def numLoamMonoliths := totalMonoliths - numSandyLoamMonoliths
def numMarineLoamMonoliths := numLoamMonoliths * probMarineLoam
def numLakeGlacialLoamMonoliths := numLoamMonoliths - numMarineLoamMonoliths
def numLakeGlacialMonoliths := numSandyLoamMonoliths + numLakeGlacialLoamMonoliths

theorem total_lake_glacial_monoliths_is_80 :
  numLakeGlacialMonoliths = 80 :=
by
  have h1 : numSandyLoamMonoliths = 26 := sorry
  have h2 : numLoamMonoliths = 117 := sorry
  have h3 : numMarineLoamMonoliths = 63 := sorry
  have h4 : numLakeGlacialLoamMonoliths = 54 := sorry
  sorry

end total_lake_glacial_monoliths_is_80_l293_293465


namespace find_n_modulo_l293_293766

theorem find_n_modulo :
  ∀ n : ℤ, (0 ≤ n ∧ n < 25 ∧ -175 % 25 = n % 25) → n = 0 :=
by
  intros n h
  sorry

end find_n_modulo_l293_293766


namespace split_tips_evenly_l293_293228

theorem split_tips_evenly :
  let julie_cost := 10
  let letitia_cost := 20
  let anton_cost := 30
  let total_cost := julie_cost + letitia_cost + anton_cost
  let tip_rate := 0.2
  let total_tip := total_cost * tip_rate
  let tip_per_person := total_tip / 3
  tip_per_person = 4 := by
  sorry

end split_tips_evenly_l293_293228


namespace find_value_l293_293952

variables (a b : ℝ)

def equation_1 : Prop := 30^a = 2
def equation_2 : Prop := 30^b = 3

theorem find_value (h1 : equation_1 a) (h2 : equation_2 b) :
  15^((1 - a - b) / (2 * (1 - b))) = 15 :=
by
  sorry

end find_value_l293_293952


namespace parallel_resistor_problem_l293_293617

theorem parallel_resistor_problem
  (x : ℝ)
  (r : ℝ := 2.2222222222222223)
  (y : ℝ := 5) : 
  (1 / r = 1 / x + 1 / y) → x = 4 :=
by sorry

end parallel_resistor_problem_l293_293617


namespace potato_plant_is_tetraploid_l293_293763

-- Define the conditions in the problem
def anther_culture_produces_haploid (plant : Type) : Prop :=
  ∃ haploid_plant : Type, somatic_cells_eq_gametes haploid_plant

def somatic_cells_eq_gametes (plant : Type) : Prop :=
  ∃ chromosome_count : ℕ, chrom_count_eq_gametes plant chromosome_count

def chrom_count_eq_gametes (plant : Type) (count : ℕ) : Prop :=
  true  -- Assuming correct chrom_count_eq_gametes is already defined

def haploid_meiosis_pairs (plant : Type) (pairs : ℕ) : Prop :=
  ∃ meiosis_count : ℕ, meiosis_count = pairs

-- The proof problem
theorem potato_plant_is_tetraploid (plant : Type) : Prop :=
  (anther_culture_produces_haploid plant) ∧
  (somatic_cells_eq_gametes plant) ∧
  (haploid_meiosis_pairs plant 12) →
  potato_plant_ploidy plant = 4 -- Assuming potato_plant_ploidy is already defined

end potato_plant_is_tetraploid_l293_293763


namespace no_integer_solution_l293_293470

theorem no_integer_solution (a b : ℤ) : ¬(a^2 + b^2 = 10^100 + 3) :=
sorry

end no_integer_solution_l293_293470


namespace imaginary_part_of_z_l293_293527

-- Define a complex number z and an imaginary unit i. 
-- State the initial condition and the target of the problem.

-- Let z be a complex number
variable {z : ℂ}

-- Given the condition
def given_condition (z : ℂ) : Prop :=
  z * (1 - complex.I) = complex.abs(1 - complex.I) + complex.I

-- The statement to be proven
theorem imaginary_part_of_z (hz : given_condition z) : complex.im z = (sqrt 2 + 1) / 2 :=
sorry

end imaginary_part_of_z_l293_293527


namespace king_midas_gold_l293_293641

theorem king_midas_gold (G x a : ℝ) (hx : x ≠ 0) :
  (a = 100 / (x - 1)) ↔ (
    let spent := (100 / x) / 100 * G in
    let remaining := G - spent in
    let required := (a / 100) * remaining in
    remaining + required = G
  ) := by
  sorry

end king_midas_gold_l293_293641


namespace no_preimage_range_l293_293933

open Set

def f (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem no_preimage_range :
  { k : ℝ | ∀ x : ℝ, f x ≠ k } = Iio 2 := by
  sorry

end no_preimage_range_l293_293933


namespace complex_roots_real_implies_12_real_z5_l293_293745

theorem complex_roots_real_implies_12_real_z5 :
  (∃ (z : ℂ) (h : z ^ 30 = 1), is_real (z ^ 5)) → (finset.card ((finset.filter (λ z, is_real (z ^ 5)) (finset.univ.filter (λ z, z ^ 30 = 1)))) = 12) := by
  sorry

end complex_roots_real_implies_12_real_z5_l293_293745


namespace four_pow_four_mul_five_pow_four_l293_293067

theorem four_pow_four_mul_five_pow_four : (4 ^ 4) * (5 ^ 4) = 160000 := by
  sorry

end four_pow_four_mul_five_pow_four_l293_293067


namespace problem_f_double_apply_l293_293151

def f (x : ℝ) : ℝ :=
if x > 0 then Real.log x / Real.log 3 else 2^x

theorem problem_f_double_apply : f (f ((1 : ℝ)/9)) = 1/4 := by
  sorry

end problem_f_double_apply_l293_293151


namespace first_player_max_area_l293_293117

-- Define the conditions of the problem
variable {ABC : Type} [Triangle ABC]
variable (S_ABC : ℝ) (h_ABC : S_ABC = 1)

def belongs_to_side {T : Type} [Triangle T] (p : point) (side : line T) : Prop := sorry

def point : Type := sorry

def Triangle.side_AB := sorry
def Triangle.side_BC := sorry
def Triangle.side_AC := sorry

noncomputable def midpoint (p1 p2 : point) : point := sorry

-- Define points chosen by the players
variable (X : point) (hX : belongs_to_side X Triangle.side_AB)
variable (Y : point) (hY : belongs_to_side Y Triangle.side_BC)
variable (Z : point) (hZ : belongs_to_side Z Triangle.side_AC)

-- Function to calculate area of subtriangle XYZ
noncomputable def area_XYZ (X Y Z : point) : ℝ := sorry

-- Proposition to be proved
theorem first_player_max_area :
  (∀ X : point, belongs_to_side X Triangle.side_AB → 
    ∀ Y : point, belongs_to_side Y Triangle.side_BC → 
    ∃ Z : point, belongs_to_side Z Triangle.side_AC ∧ 
      area_XYZ X Y Z ≥ 1 / 4) := sorry

end first_player_max_area_l293_293117


namespace range_of_a_l293_293654

noncomputable def f (a x : ℝ) : ℝ := x + (a^2) / x
noncomputable def g (x : ℝ) : ℝ := x - Real.log x

theorem range_of_a (a : ℝ) (h : a > 0) 
  (h_fg : ∀ x1 x2 : ℝ, x1 ∈ Set.Icc 1 Real.exp 1 → x2 ∈ Set.Icc 1 Real.exp 1 → f a x1 ≥ g x2) : 
  a ≥ Real.sqrt (Real.exp 1 - 2) :=
by
  sorry

end range_of_a_l293_293654


namespace max_value_g_l293_293078

noncomputable def g (x : ℝ) : ℝ := 4*x - x^4

theorem max_value_g : ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 2 → g y ≤ g x ∧ g x = 3 :=
sorry

end max_value_g_l293_293078


namespace solve_for_y_l293_293279

def diamond (a b : ℕ) : ℕ := 2 * a + b

theorem solve_for_y (y : ℕ) (h : diamond 4 (diamond 3 y) = 17) : y = 3 :=
by sorry

end solve_for_y_l293_293279


namespace sum_of_first_2015_digits_l293_293298

noncomputable def repeating_decimal : List ℕ := [1, 4, 2, 8, 5, 7]

def sum_first_n_digits (digits : List ℕ) (n : ℕ) : ℕ :=
  let repeat_length := digits.length
  let full_cycles := n / repeat_length
  let remaining_digits := n % repeat_length
  full_cycles * (digits.sum) + (digits.take remaining_digits).sum

theorem sum_of_first_2015_digits :
  sum_first_n_digits repeating_decimal 2015 = 9065 :=
by
  sorry

end sum_of_first_2015_digits_l293_293298


namespace fair_schedules_larger_than_bound_l293_293764

-- Definitions based on problem conditions
def schedule (V N : List ℕ) : Prop :=
  V.length = 2020 ∧ N.length = 2020 ∧
  (∀ i < 2020, V.nth i ≠ N.nth i) ∧
  (∀ i, V.count i = 1 ∧ N.count i = 1)

def fair_schedule (V N : List ℕ) := 
  let VbN := V.countp (λ i, V.indexOf i < N.indexOf i)
  let NbV := N.countp (λ i, N.indexOf i < V.indexOf i)
  VbN = NbV

def number_of_schedules_exceeds (schedules : List (List ℕ × List ℕ)) (bound : ℕ): Prop :=
  schedules.countp (λ (V, N), schedule V N ∧ fair_schedule V N) > bound

noncomputable def bound : ℕ := 2020! * (2^1010 + (1010!)^2)

theorem fair_schedules_larger_than_bound
  (schedules: List (List ℕ × List ℕ)) :
  number_of_schedules_exceeds schedules bound :=
by sorry

end fair_schedules_larger_than_bound_l293_293764


namespace max_value_of_g_in_interval_l293_293512

noncomputable def g (x : ℝ) : ℝ := 2 * x - x^3

theorem max_value_of_g_in_interval :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = (2 * (3 * real.sqrt 3 - real.sqrt 2) * real.sqrt (2 / 3)) / (3 * real.sqrt 3) :=
by
  sorry

end max_value_of_g_in_interval_l293_293512


namespace BR_squared_is_160_17_final_result_l293_293643

noncomputable def square_side_length := 4
noncomputable def point_B := (4 : ℝ, 4 : ℝ)
noncomputable def point_A := (0 : ℝ, 4 : ℝ)
noncomputable def point_D := (0 : ℝ, 0 : ℝ)
noncomputable def point_C := (4 : ℝ, 0 : ℝ)
noncomputable def BP := 3
noncomputable def BQ := 1
noncomputable def point_P := (1 : ℝ, 4 : ℝ)
noncomputable def point_Q := (4 : ℝ, 3 : ℝ)

noncomputable def point_R : ℝ × ℝ := 
  let x := 16 / 17
  let y := 64 / 17
  (x, y)

noncomputable def BR_sq : ℝ := 
  let (bx, by) := point_B
  let (rx, ry) := point_R
  ((bx - rx)^2 + (by - ry)^2)

theorem BR_squared_is_160_17 : BR_sq = 160 / 17 := by
  -- calculation steps skipped
  sorry

theorem final_result : 160 + 17 = 177 := by
  -- calculation steps skipped
  sorry

end BR_squared_is_160_17_final_result_l293_293643


namespace starship_reaches_boundary_l293_293729

theorem starship_reaches_boundary
  (R : ℝ) -- distance from the half-space boundary
  (in_half_space : ∃ (p : ℝ × ℝ × ℝ), p ∈ half_space) -- initial position in half-space
  (boundary_sensor : ℝ × ℝ × ℝ → Prop) -- sensor signals when boundary is reached
  (reachable : ∃ (trajectory : (ℝ × ℝ × ℝ) → ℝ), (∀ p, 0 ≤ trajectory p ∧ trajectory p ≤ R)) -- valid trajectories
  : ∃ (p : ℝ × ℝ × ℝ), boundary_sensor p :=
sorry

end starship_reaches_boundary_l293_293729


namespace possible_perimeters_l293_293441

theorem possible_perimeters (a b c: ℝ) (h1: a = 1) (h2: b = 1) 
  (h3: c = 1) (h: ∀ x y z: ℝ, x = y ∧ y = z):
  ∃ x y: ℝ, (x = 8/3 ∧ y = 5/2) := 
  by
    sorry

end possible_perimeters_l293_293441


namespace simplest_square_root_is_b_l293_293390

theorem simplest_square_root_is_b (a b : ℝ) : 
  (\sqrt{a^2 + b^2} = \sqrt{a^2 + b^2}) ∧ 
  (4√a ≠ \sqrt{a^2 + b^2}) ∧ 
  (\sqrt{\frac{b}{a}} ≠ \sqrt{a^2 + b^2}) ∧ 
  (3√5 ≠ \sqrt{a^2 + b^2}) :=
by
  -- Prove that the given square root is the simplest
  sorry

end simplest_square_root_is_b_l293_293390


namespace applejack_max_trees_bucked_l293_293772

-- Define the initial conditions and the objective.
def applejack_initial_energy : ℕ := 100
def minutes_available : ℕ := 60

noncomputable def max_trees_bucked (initial_energy : ℕ) (minutes : ℕ) : ℕ :=
  let x := 6 in  -- Given the problem's steps, optimal resting time is evaluated to 6 or 7 minutes.
  let series_sum := (minutes - x) * (141 + 3 * x) / 2 in
  series_sum

theorem applejack_max_trees_bucked : max_trees_bucked applejack_initial_energy minutes_available = 4293 :=
by
  sorry

end applejack_max_trees_bucked_l293_293772


namespace arithmetic_sequence_geometric_sequence_l293_293118

noncomputable def a_n (n : ℕ) : ℚ := (n : ℚ) / 2

theorem arithmetic_sequence (a_n : ℕ → ℚ) (S : ℕ → ℚ)
    (h1 : a_n 2 = 1) 
    (h2 : S 11 = 33) 
    (h3 : ∀ n, S n = (n * (a_n 1) + (n * (n-1) / 2) * (a_n 2 - a_n 1))) :
  (∀ n, a_n n = (n : ℚ) / 2) :=
sorry

noncomputable def b_n (a_n : ℕ → ℚ) (n : ℕ) : ℚ := (1/4) ^ (a_n n : ℚ)

theorem geometric_sequence (b_n : ℕ → ℚ)
    (h4 : ∀ n, b_n n = (1 / 2) ^ n) :
  (∃ r : ℚ, r = 1 / 2 ∧ ∀ m n, b_n (m+1) / b_n m = r) :=
sorry

end arithmetic_sequence_geometric_sequence_l293_293118


namespace domain_of_f_l293_293072

-- Define the function f(x) as specified in the conditions
def f (x : ℝ) : ℝ := real.sqrt (2 - real.sqrt (4 - real.sqrt (5 - x)))

-- Prove the domain of f(x) is between -11 and 5 inclusive
theorem domain_of_f : ∀ (x : ℝ), (f x) = f x → -11 ≤ x ∧ x ≤ 5 :=
by
  sorry

end domain_of_f_l293_293072


namespace sum_of_integers_sum_of_even_integers_l293_293728

theorem sum_of_integers (x : ℤ) (h1 : Even x) (h2 : x^2 = 200 + x) : x = 20 ∨ x = -10 :=
by
  sorry

theorem sum_of_even_integers (S : Set ℤ) (h : ∀ x, x ∈ S ↔ Even x ∧ x^2 = 200 + x) :
  ∑ x in S, x = 10 :=
by
  have h0 : S = {20, -10} := sorry
  rw [h0]
  dsimp
  norm_num
  sorry

end sum_of_integers_sum_of_even_integers_l293_293728


namespace proof_problem_l293_293249

-- We first define a and b are odd numbers and are positive integers.
variables {a b : ℕ}
variable (a_odd : odd a)
variable (b_odd : odd b)
variable (a_pos : a > 0)
variable (b_pos : b > 0)

-- Now, we state that for any positive integer n, there exists an m such that either of the 
-- equations is satisfied.
theorem proof_problem (n : ℕ) (n_pos : n > 0) :
  ∃ m : ℕ, 2^n ∣ (a^m * b^2 - 1) ∨ 2^n ∣ (b^m * a^2 - 1) :=
  sorry

end proof_problem_l293_293249


namespace evaluate_ceiling_sums_l293_293035

theorem evaluate_ceiling_sums : 
  (⌈real.sqrt 3⌉ + ⌈real.sqrt 33⌉ + ⌈real.sqrt 333⌉) = 27 :=
by
  have h1 : 1 < real.sqrt 3 ∧ real.sqrt 3 < 2 :=
    ⟨by norm_num, by norm_num⟩,
  have h2 : 5 < real.sqrt 33 ∧ real.sqrt 33 < 6 :=
    ⟨by norm_num, by norm_num⟩,
  have h3 : 18 < real.sqrt 333 ∧ real.sqrt 333 < 19 :=
    ⟨by norm_num, by norm_num⟩,
  sorry

end evaluate_ceiling_sums_l293_293035


namespace max_value_of_f_on_interval_l293_293856

noncomputable def f (x : ℝ) : ℝ := 2^x + x * Real.log (1/4)

theorem max_value_of_f_on_interval :
  ∃ x ∈ Set.Icc (-2:ℝ) 2, f x = (1/4:ℝ) + 4 * Real.log 2 := 
sorry

end max_value_of_f_on_interval_l293_293856


namespace geom_sum_first_eight_terms_l293_293090

theorem geom_sum_first_eight_terms (a r : ℚ) (h_a : a = 1/3) (h_r : r = 1/3) :
    ∑ k in finset.range 8, a * r^k = 9840/19683 := by
  sorry

end geom_sum_first_eight_terms_l293_293090


namespace toppings_combination_l293_293673

-- Define the combination function
def combination (n k : ℕ) : ℕ := n.choose k

theorem toppings_combination :
  combination 9 3 = 84 := by
  sorry

end toppings_combination_l293_293673


namespace problem_statement_l293_293244

noncomputable def a : ℝ := 0.3^2
noncomputable def b : ℝ := 2^0.3
noncomputable def c : ℝ := Real.log 4 / Real.log 0.3 -- log_base(x) = (log(x) / log(base))

theorem problem_statement : c < a ∧ a < b :=
by
  have hyp_a : a = 0.3^2 := rfl
  have hyp_b : b = 2^0.3 := rfl
  have hyp_c : c = Real.log 4 / Real.log 0.3 := rfl
  sorry

end problem_statement_l293_293244


namespace heaviest_and_lightest_in_13_weighings_l293_293316

/-- Given ten coins of different weights and a balance scale.
    Prove that it is possible to identify the heaviest and the lightest coin
    within 13 weighings. -/
theorem heaviest_and_lightest_in_13_weighings
  (coins : Fin 10 → ℝ)
  (h_different: ∀ i j : Fin 10, i ≠ j → coins i ≠ coins j)
  : ∃ (heaviest lightest : Fin 10),
      (heaviest ≠ lightest) ∧
      (∀ i : Fin 10, coins i ≤ coins heaviest) ∧
      (∀ i : Fin 10, coins lightest ≤ coins i) :=
sorry

end heaviest_and_lightest_in_13_weighings_l293_293316


namespace minimum_time_to_find_faulty_bulb_l293_293824

theorem minimum_time_to_find_faulty_bulb:
  ∀ (n : ℕ) (unscrew_time screw_time : ℕ), (n = 4) → (unscrew_time = 10) → (screw_time = 10) →
  ∃ (min_time : ℕ), min_time = 60 :=
by
  intro n unscrew_time screw_time h1 h2 h3
  exists 60
  sorry

end minimum_time_to_find_faulty_bulb_l293_293824


namespace area_of_scaled_vectors_l293_293705

variables (u v : ℝ^3)

-- Given condition
def given_area : ℝ := 12

-- Statement to prove
theorem area_of_scaled_vectors :
  ‖(3 * u - 2 * v) × (4 * u + 5 * v)‖ = 276 :=
by 
  sorry

end area_of_scaled_vectors_l293_293705


namespace monotonous_integer_count_l293_293484

/--
Define a positive integer as monotonous if its digits, when read from left to right, form either a strictly increasing or a strictly decreasing sequence.
Allowable digits are from 0 to 8 and for decreasing sequences, the digit 9 can be appended to the end of any sequence.
Prove that there are 1542 monotonous positive integers.
-/
theorem monotonous_integer_count : 
  let monotonous := λ (digits : List ℕ), 
                      (digits.all (λ d, d ∈ [0,1,2,3,4,5,6,7,8])) ∧ 
                      (strict_mono digits ∨ (strict_anti (drop_last digits) ∧ digits.last = some 9))
  in
  ∃ (count : ℕ), count = 1542 ∧ count = (List.range 9).powerset.to_finset.card + 
                                       2 * (List.range 9).powerset.to_finset.card - 1 
:=
begin
  -- Sorry is used to skip the proof
  sorry,
end

end monotonous_integer_count_l293_293484


namespace A_visits_all_seats_iff_even_l293_293415

def move_distance_unique (n : ℕ) : Prop := 
  ∀ k l : ℕ, (1 ≤ k ∧ k < n) → (1 ≤ l ∧ l < n) → k ≠ l → (k ≠ l % n)

def visits_all_seats (n : ℕ) : Prop := 
  ∃ A : ℕ → ℕ, 
  (∀ (k : ℕ), 0 ≤ k ∧ k < n → (0 ≤ A k ∧ A k < n)) ∧ 
  (∀ (k : ℕ), 0 ≤ k ∧ k < n → ∃ (m : ℕ), m ≠ n ∧ A k ≠ (A m % n))

theorem A_visits_all_seats_iff_even (n : ℕ) :
  (move_distance_unique n ∧ visits_all_seats n) ↔ (n % 2 = 0) := 
sorry

end A_visits_all_seats_iff_even_l293_293415


namespace max_value_g_l293_293079

noncomputable def g (x : ℝ) : ℝ := 4*x - x^4

theorem max_value_g : ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 2 → g y ≤ g x ∧ g x = 3 :=
sorry

end max_value_g_l293_293079


namespace rectangle_area_tangent_circle_l293_293436

open Real

theorem rectangle_area_tangent_circle (x r : ℝ) (h : 2 * r = x) : 
  let w := 3 * x -- width of the rectangle
  let h := x -- height of the rectangle
  in w * h = 12 * r^2 :=
by
  let w := 3 * x
  let h := x
  have : w * h = 3 * x * x := by ring
  rw [← h] at this
  rw [h] at this
  simp [h.symm, mul_assoc] at this
  sorry

end rectangle_area_tangent_circle_l293_293436


namespace highest_profit_plan_l293_293803

section Production

variables (x y : ℕ) (a b : ℝ)

-- Conditions
def raw_material_A := 66 : ℝ
def raw_material_B := 66.4 : ℝ
def total_products := 90 : ℕ
def raw_material_A_per_A := 0.5 : ℝ
def raw_material_B_per_A := 0.8 : ℝ
def raw_material_A_per_B := 1.2 : ℝ
def raw_material_B_per_B := 0.6 : ℝ
def profit_per_A := 30 : ℝ
def profit_per_B := 20 : ℝ

-- Constraints
def constraints_satisfied (x y : ℕ) : Prop :=
  x + y = total_products ∧
  raw_material_A_per_A * x + raw_material_A_per_B * y ≤ raw_material_A ∧
  raw_material_B_per_A * x + raw_material_B_per_B * y ≤ raw_material_B

-- Possible production plans
def possible_plans :=
  { (60, 30), (61, 29), (62, 28) }

-- Calculation of profit
def profit (x : ℕ) : ℝ :=
  profit_per_A * x + profit_per_B * (total_products - x)

-- Highest profit
noncomputable def max_profit : ℝ :=
  max (profit 60) (max (profit 61) (profit 62))

theorem highest_profit_plan :
  (max_profit = 2420) ∧
  ((constraints_satisfied 60 30 ∨ constraints_satisfied 61 29 ∨ constraints_satisfied 62 28) ∧
   max_profit = profit 62) :=
by sorry

end Production

end highest_profit_plan_l293_293803


namespace magnitude_of_complex_z_l293_293596

theorem magnitude_of_complex_z (z : ℂ) (h : complex.I * z = 3 + complex.I) : complex.abs z = real.sqrt 10 :=
by
  sorry

end magnitude_of_complex_z_l293_293596


namespace factorize_x_squared_plus_2x_l293_293062

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by
  sorry

end factorize_x_squared_plus_2x_l293_293062


namespace problem_l293_293919

noncomputable def eccentricity : ℝ := 1 / 2

noncomputable def ellipse : ℝ × ℝ → Prop := 
  λ p, (p.1 ^ 2) / 4 + (p.2 ^ 2) / 3 = 1

noncomputable def left_focus_distance : ℝ := 5 / 2

noncomputable def right_focus_distance : ℝ := 4 - left_focus_distance

theorem problem : ∀ P : ℝ × ℝ, ellipse P → ∃ d : ℝ, 
  right_focus_distance / d = eccentricity ∧ d = 3 := 
by
  intro P hP,
  use 3,
  split,
  -- 1st part: right_focus_distance / 3 = eccentricity
  sorry,
  -- 2nd part: d = 3
  refl

end problem_l293_293919


namespace value_of_2x_l293_293607

theorem value_of_2x (x y z : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_eq : 2 * x = 6 * z) (h_sum : x + y + z = 26) : 2 * x = 6 := 
by
  sorry

end value_of_2x_l293_293607


namespace dominos_tiling_impossible_l293_293479

theorem dominos_tiling_impossible :
  let chessboard := (fin 8 × fin 8)
  ∃ (black_squares_removed : set (fin 8 × fin 8)),
    black_squares_removed = {(0, 0), (7, 7)} ∧
    (∀ (f : (fin 8 × fin 8) → option (fin 31)),
      ∃ a b : fin 31, (f a = none ∧ f b = none)) ∧
    (|black_squares_removed| = 2) ∧
    (∃ (remaining_squares : set (fin 8 × fin 8)),
      remaining_squares = (chessboard \ black_squares_removed) ∧
      card remaining_squares = 62) →
    ¬∃ (domino_tiling : (fin 8 × fin 8) → (fin 8 × fin 8)),
      bijective domino_tiling ∧
      (∀ (p : fin 8 × fin 8),
        p ∈ remaining_squares →
        p.1 % 2 ≠ p.2 % 2 →
        domino_tiling p = (p.1.succ % 8, p.2) ∨
        domino_tiling p = (p.1, p.2.succ % 8)) :=
by {
  intros _ _ _,
  have h1 : card (fin 8 × fin 8) = 64 := sorry, -- given condition
  have h2 : |black_squares_removed| = 2 := by sorry,
  have h3 : 64 - 2 = 62 := by sorry,
  have h4 : ∃ (remaining_squares : set (fin 8 × fin 8)),
    remaining_squares = (fin 8 × fin 8) \ black_squares_removed ∧
    card remaining_squares = 62 := by sorry,
  have h5 : |remaining_squares| = 62 ∧ (∀ (f : (fin 8 × fin 8) → option (fin 31)),
    ∃ a b : fin 31, f a = none ∧ f b = none) := by sorry,
  exact h5,
  have not_possible_to_tile := by sorry,
  exact not_possible_to_tile,
}

end dominos_tiling_impossible_l293_293479


namespace lines_are_skew_l293_293070

open Classical

def vector (α : Type*) := α × α × α  -- Representing 3D vectors
def line (point dir : vector ℚ) (t : ℚ) := 
  (point.1 + t * dir.1, point.2 + t * dir.2, point.3 + t * dir.3)

-- Definitions of the two lines
def line1 (b : ℚ) := λ t : ℚ, line (2, 1, b) (3, 4, 2) t
def line2 := λ u : ℚ, line (5, 3, 2) (1, -1, 2) u

-- The system of equations for the lines to intersect
def intersect_system (t u : ℚ) (b : ℚ) := 
    (2 + 3 * t = 5 + u) ∧ 
    (1 + 4 * t = 3 - u) ∧ 
    (b + 2 * t = 2 + 2 * u)

-- The proof problem: finding all values of b where the lines do not intersect, hence they are skew
theorem lines_are_skew (b : ℚ) : 
  ¬∃ t u : ℚ, intersect_system t u b ↔ b ≠ 2 / 7 :=
by
  sorry

end lines_are_skew_l293_293070


namespace wam_gm_gt_hm_l293_293242

noncomputable def wam (w v a b : ℝ) : ℝ := w * a + v * b
noncomputable def gm (a b : ℝ) : ℝ := Real.sqrt (a * b)
noncomputable def hm (a b : ℝ) : ℝ := (2 * a * b) / (a + b)

theorem wam_gm_gt_hm
  (a b w v : ℝ)
  (h1 : 0 < a ∧ 0 < b)
  (h2 : 0 < w ∧ 0 < v)
  (h3 : w + v = 1)
  (h4 : a ≠ b) :
  wam w v a b > gm a b ∧ gm a b > hm a b :=
by
  -- Proof omitted
  sorry

end wam_gm_gt_hm_l293_293242


namespace part1_part2_part3_l293_293000

-- Problem 1
theorem part1 (x : ℕ) : (x - 1) * (∑ k in finset.range (7), x^k) = x^7 - 1 := sorry

-- Problem 2
theorem part2 (x : ℕ) (n : ℕ) : (x - 1) * (∑ k in finset.range (n), x^k) = x^n - 1 := sorry

-- Problem 3
theorem part3 : (∑ k in finset.range (36), 2^k) = 2^36 - 1 := sorry

end part1_part2_part3_l293_293000


namespace tan_half_angle_l293_293884

theorem tan_half_angle (x y : ℝ) (h1 : cos (x + y) * sin x - sin (x + y) * cos x = 12 / 13)
  (h2 : ∃ (k : ℤ), 2 * k * π + 3 * π / 2 < y ∧ y < 2 * k * π + 2 * π) :
  tan (y / 2) = -2 / 3 := 
sorry

end tan_half_angle_l293_293884


namespace complex_roots_real_l293_293742

theorem complex_roots_real (z : ℂ) (h : z^30 = 1) : 
  {z : ℂ | z^30 = 1}.count (λ z, z^5 ∈ ℝ) = 10 :=
sorry

end complex_roots_real_l293_293742


namespace largest_angle_in_triangle_l293_293615

theorem largest_angle_in_triangle 
  (A B C : ℝ)
  (h_sum_angles: 2 * A + 20 = 105)
  (h_triangle_sum: A + (A + 20) + C = 180)
  (h_A_ge_0: A ≥ 0)
  (h_B_ge_0: B ≥ 0)
  (h_C_ge_0: C ≥ 0) : 
  max A (max (A + 20) C) = 75 := 
by
  -- Placeholder proof
  sorry

end largest_angle_in_triangle_l293_293615


namespace measure_angle_BPC_l293_293626

/-- Given a square ABCD with side length 5 and an equilateral triangle ABE.
Line segments BE and AC intersect at P. Point Q is on BC such that PQ is perpendicular to BC
and PQ = y. We need to prove the measure of ∠BPC equals 105°. -/
theorem measure_angle_BPC (A B C D E P Q : Point) (AB_CD_square : is_square A B C D)
  (AB_length : dist A B = 5) (ABE_equilateral : is_equilateral_triangle A B E)
  (BE_AC_intersect_P : line BE ∩ line AC = P) (Q_on_BC : Q ∈ segment B C) 
  (PQ_perpendicular_BC : is_perpendicular PQ BC) (PQ_length : dist P Q = y) :
  measure (angle B P C) = 105 :=
sorry

end measure_angle_BPC_l293_293626


namespace solution_interval_for_x_l293_293499

theorem solution_interval_for_x (x : ℝ) : 
  (⌊x * ⌊x⌋⌋ = 48) ↔ (48 / 7 ≤ x ∧ x < 49 / 7) :=
by sorry

end solution_interval_for_x_l293_293499


namespace no_positive_integer_triples_l293_293854

theorem no_positive_integer_triples (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) : ¬ (x^2 + y^2 + 41 = 2^n) :=
  sorry

end no_positive_integer_triples_l293_293854


namespace dot_product_right_triangle_midpoint_l293_293621

open Real EuclideanGeometry 

theorem dot_product_right_triangle_midpoint :
  ∀ (A B C D : Point)
    (h_triangle : Triangle A B C)
    (h_right : Angle A C B = π / 2)
    (h_angle_A : Angle B A C = π / 6)
    (h_length_BC : dist B C = 1)
    (h_midpoint_D : Midpoint (LineSegment A B) D),
  (vector AB).dot (vector CD) = -1 := by sorry

end dot_product_right_triangle_midpoint_l293_293621


namespace identify_heaviest_and_lightest_l293_293329

   def coin : Type := ℕ  -- let's represent coins as natural numbers for simplicity.

   def has_different_weights (coins : list coin) : Prop := 
     ∀ (c1 c2 : coin), c1 ∈ coins → c2 ∈ coins → c1 ≠ c2 → weight(c1) ≠ weight(c2)

   def weight : coin → ℝ := -- assume a function that gives the weight corresponding to a coin.
     sorry 

   theorem identify_heaviest_and_lightest (coins : list coin) 
     (h₁ : length coins = 10)
     (h₂ : has_different_weights coins) : 
     ∃ (heaviest lightest : coin), 
       (heaviest ∈ coins) ∧ (lightest ∈ coins) ∧
       (∀ c ∈ coins, weight c ≤ weight heaviest) ∧
       (∀ c ∈ coins, weight c ≥ weight lightest) :=
   by 
     sorry
   
end identify_heaviest_and_lightest_l293_293329


namespace pure_imaginary_number_a_l293_293157

theorem pure_imaginary_number_a (a : ℝ) 
  (h1 : a^2 + 2 * a - 3 = 0)
  (h2 : a^2 - 4 * a + 3 ≠ 0) : a = -3 :=
sorry

end pure_imaginary_number_a_l293_293157


namespace abs_diff_x_y_l293_293998

open Real

noncomputable def floor_fract_sum_eq (x y : ℝ) :=
  int.floor x + fract y = 7.2

noncomputable def fract_floor_sum_eq (x y : ℝ) :=
  fract x + int.floor y = 10.3

theorem abs_diff_x_y (x y : ℝ) (h1 : floor_fract_sum_eq x y) 
                      (h2 : fract_floor_sum_eq x y) : |x - y| = 2.9 := 
by 
  sorry

end abs_diff_x_y_l293_293998


namespace arc_length_calculation_sector_optimization_l293_293137

noncomputable def rad_of_deg (d : ℝ) : ℝ := d * Real.pi / 180
def arc_length (α r : ℝ) : ℝ := α * r
def perimeter_eq (l r : ℝ) : Prop := l + 2 * r = 24
def sector_area (l r : ℝ) : ℝ := (1 / 2) * l * r

theorem arc_length_calculation :
  arc_length (rad_of_deg 120) 6 = 4 * Real.pi := 
by
  sorry

theorem sector_optimization :
  ∃ α S, α = 2 ∧ S = 36 ∧ 
  (∀ r l, perimeter_eq l r → sector_area (24 - 2 * r) r ≤ S) :=
by
  sorry

end arc_length_calculation_sector_optimization_l293_293137


namespace no_nat_solutions_l293_293579

theorem no_nat_solutions (x y : ℕ) : (2 * x + y) * (2 * y + x) ≠ 2017 ^ 2017 := by sorry

end no_nat_solutions_l293_293579


namespace find_tan_alpha_l293_293623

noncomputable def line_l (t α : ℝ) : ℝ × ℝ :=
(2 + t * Real.cos α, 1 + t * Real.sin α)

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
(4 * Real.cos θ * Real.cos θ, 4 * Real.cos θ * Real.sin θ)

def point_P : ℝ × ℝ := (2, 1)

theorem find_tan_alpha (α : ℝ) (hα : 0 ≤ α ∧ α < Real.pi) :
  (∃ t1 t2, line_l t1 α ∈ set_of (λ p, ∃ θ, p = curve_C θ) ∧
            line_l t2 α ∈ set_of (λ p, ∃ θ, p = curve_C θ) ∧
            vector_rel (line_l t1 α) point_P = 2 * vector_rel (line_l t2 α) point_P)
  → Real.tan α = √(3 / 5) ∨ Real.tan α = -√(3 / 5) :=
by
  sorry


end find_tan_alpha_l293_293623


namespace constant_term_of_first_equation_l293_293921

theorem constant_term_of_first_equation
  (y z : ℤ)
  (h1 : 2 * 20 - y - z = 40)
  (h2 : 3 * 20 + y - z = 20)
  (hx : 20 = 20) :
  4 * 20 + y + z = 80 := 
sorry

end constant_term_of_first_equation_l293_293921


namespace no_intersection_of_lines_l293_293872

theorem no_intersection_of_lines (k : ℝ) (t s : ℝ) :
  (1 + t * 5 = 1 + s * -3) ∧ (3 + t * -2 = 1 + s * k) → k ≠ (6 / 5) :=
by 
  intro h
  have h_eq : k = 6 / 5, sorry
  contradiction

end no_intersection_of_lines_l293_293872


namespace area_of_gray_region_l293_293368

theorem area_of_gray_region (d_s : ℝ) (h1 : d_s = 4) (h2 : ∀ r_s r_l : ℝ, r_s = d_s / 2 → r_l = 3 * r_s) : ℝ :=
by
  let r_s := d_s / 2
  let r_l := 3 * r_s
  let area_larger := π * r_l^2
  let area_smaller := π * r_s^2
  let area_gray := area_larger - area_smaller
  have hr_s : r_s = 2 := by sorry
  have hr_l : r_l = 6 := by sorry
  have ha_larger : area_larger = 36 * π := by sorry
  have ha_smaller : area_smaller = 4 * π := by sorry
  have ha_gray : area_gray = 32 * π := by sorry
  exact ha_gray

end area_of_gray_region_l293_293368


namespace probability_of_sum_lt_six_l293_293799

noncomputable def calc_probability : ℚ :=
  let A := {1, 3, 5, 7, 9, 11}
  let B := {2, 4, 6, 8, 10}
  let favorable_sums : Finset ℕ := 
    (Finset.product A B).filter (λ ab, ab.1 + ab.2 < 6).map (λ ab, ab.1 + ab.2)
  favorable_sums.card / ((A.card:ℕ) * (B.card:ℕ))

theorem probability_of_sum_lt_six : calc_probability = 1 / 10 :=
  by sorry

end probability_of_sum_lt_six_l293_293799


namespace num_ways_distinct_letters_l293_293175

def letters : List String := ["A₁", "A₂", "A₃", "N₁", "N₂", "N₃", "B₁", "B₂"]

theorem num_ways_distinct_letters : (letters.permutations.length = 40320) := by
  sorry

end num_ways_distinct_letters_l293_293175


namespace parity_and_monotonicity_exists_a_n_l293_293927

noncomputable def f (a x : ℝ) : ℝ := log a ((x + 1) / (x - 1))

theorem parity_and_monotonicity (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : a > 1) :
  (∀ x : ℝ, x > 1 → f a x = -f a (-x)) ∧ (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f a x₁ > f a x₂) :=
by sorry

theorem exists_a_n (x : ℝ) (a : ℝ) (n : ℝ) (h₀ : x ∈ set.Ioo n (a - 2))
  (h₁ : n = 1) (h₂ : a = 2 + Real.sqrt 3) : 
  ∃ a n, ∀ y ∈ set.Ioo n (a - 2), 1 < f a y :=
by sorry

end parity_and_monotonicity_exists_a_n_l293_293927


namespace max_factors_b_pow_n_l293_293302

theorem max_factors_b_pow_n (b n : ℕ) (hb : 1 ≤ b ∧ b ≤ 10) (hn : 1 ≤ n ∧ n ≤ 10) : 
  ∃ k, (Nat.factorsMultiset b).card = k ∧
       (∀ m, m ∈ Nat.factorsMultiset (b ^ n) → m ∈ Nat.factorsMultiset b) ∧
       (b = 2^3 ∧ n = 10 → k * 10 + 1 = 31) := 
by
  intros
  sorry

end max_factors_b_pow_n_l293_293302


namespace part1_geom_sequence_and_general_term_part2_inequality_l293_293915

-- Conditions
variable {a S : ℕ → ℝ}
variable {n : ℕ+} -- ℕ+ representing positive natural numbers

axiom condition_1 : ∀ (n : ℕ+), 2 * S n = 3 * a n - 2 * (n : ℕ)

-- Proof Goals
theorem part1_geom_sequence_and_general_term :
    (∀ (n : ℕ+), a n + 1 = 3^(n - 1).val - 1) ∧
    (a n + 1 = 3 * (a (n - 1) + 1)) :=
sorry

theorem part2_inequality (b : ℕ+ → ℝ) (h_b : ∀ (n : ℕ+), b n = a n + 2^n.val + 1) :
    (∑ i in Finset.range n.val.succ, (1 / (b (i + 1)))) < (1 / 2) - (1 / (2^(n.val + 1))) :=
sorry

end part1_geom_sequence_and_general_term_part2_inequality_l293_293915


namespace complex_roots_real_implies_12_real_z5_l293_293747

theorem complex_roots_real_implies_12_real_z5 :
  (∃ (z : ℂ) (h : z ^ 30 = 1), is_real (z ^ 5)) → (finset.card ((finset.filter (λ z, is_real (z ^ 5)) (finset.univ.filter (λ z, z ^ 30 = 1)))) = 12) := by
  sorry

end complex_roots_real_implies_12_real_z5_l293_293747


namespace inequality_correct_l293_293524

theorem inequality_correct (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c :=
sorry

end inequality_correct_l293_293524


namespace snow_added_on_third_day_l293_293797

def snow_depth_day_four (x : ℝ) :=
  let day_one := 20
  let day_two := day_one / 2
  let day_three := day_two + x
  let day_four := day_three + 18
  day_four

theorem snow_added_on_third_day :
  ∃ x, snow_depth_day_four x = 34 ∧ x = 6 :=
by
  use 6
  split
  · unfold snow_depth_day_four
    norm_num
  · norm_num

end snow_added_on_third_day_l293_293797


namespace find_ellipse_equation_max_AB_value_l293_293561

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x = sqrt 3 ∧ y = 1 / 2) → (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def eccentricity (a b : ℝ) : Prop :=
  sqrt (a^2 - b^2) / a = sqrt 3 / 2

theorem find_ellipse_equation (a b : ℝ) (h₀ : a > b > 0)
  (h₁ : ellipse_equation a b) (h₂ : eccentricity a b) :
  a = 2 ∧ b = 1 → ∀ x y : ℝ, x^2 / 4 + y^2 = 1 :=
by
  sorry

noncomputable def tangent_line_condition (k m : ℝ) : Prop :=
  m^2 = k^2 + 1

theorem max_AB_value (k m a b : ℝ) (h₀ : a > b > 0)
  (h₁ : tangent_line_condition k m)
  (h₂ : a = 2 ∧ b = 1)
  (h₃ : a^2 / b^2 + k^2 = 1) :
  (4 * sqrt (1 + k^2) * sqrt(k^2 + 1) / (1 + 4 * k^2) ≤ 2) :=
by
  sorry

end find_ellipse_equation_max_AB_value_l293_293561


namespace fraction_zero_condition_l293_293187

theorem fraction_zero_condition (x : ℝ) (h : (abs x - 2) / (2 - x) = 0) : x = -2 :=
by
  sorry

end fraction_zero_condition_l293_293187


namespace find_x_l293_293208

namespace Geometry

-- Define the points and angles based on the conditions
def angle_QPR : Real := 75
def angle_PRS : Real := 125

-- Collinear points imply sum of angles on a straight line equals 180 degrees
theorem find_x:
  ∀ (x : Real),
  -- Given conditions
  (angle_PRS = 125) →
  (angle_QPR = 75) →
  -- Straight line
  (λ angle_PRQ, angle_PRQ + angle_PRS = 180) →
  -- Sum of angles in triangle
  (λ angle_PRQ, x + angle_QPR + angle_PRQ = 180) →
  -- Conclusion
  x = 50 :=
by
  intros x h1 h2 h_collinear h_sum_triangle
  sorry

end Geometry

end find_x_l293_293208


namespace identify_heaviest_and_lightest_l293_293328

   def coin : Type := ℕ  -- let's represent coins as natural numbers for simplicity.

   def has_different_weights (coins : list coin) : Prop := 
     ∀ (c1 c2 : coin), c1 ∈ coins → c2 ∈ coins → c1 ≠ c2 → weight(c1) ≠ weight(c2)

   def weight : coin → ℝ := -- assume a function that gives the weight corresponding to a coin.
     sorry 

   theorem identify_heaviest_and_lightest (coins : list coin) 
     (h₁ : length coins = 10)
     (h₂ : has_different_weights coins) : 
     ∃ (heaviest lightest : coin), 
       (heaviest ∈ coins) ∧ (lightest ∈ coins) ∧
       (∀ c ∈ coins, weight c ≤ weight heaviest) ∧
       (∀ c ∈ coins, weight c ≥ weight lightest) :=
   by 
     sorry
   
end identify_heaviest_and_lightest_l293_293328


namespace find_lengths_l293_293234

-- Given definitions and conditions
variables (A B C D P K : Point)
variable (α : ℝ)
variable (AB AC BC : ℝ)
variable [Nonempty ℝ]

-- Define that D is the foot of the altitude from B
def is_altitude_foot (B A C D : Point) : Prop :=
  altitude_from B = D

-- Define specific lengths and their properties
def is_midpoint (A C K : Point) : Prop := midpoint A C K
def is_incenter (BCD P : Triangle) : Prop := incenter B C D = P
def is_centroid (ABC P : Triangle) : Prop := centroid A B C = P

-- Main theorem statement
theorem find_lengths (h_altitude_foot : is_altitude_foot B A C D)
    (h_AB : AB = 1)
    (h_incenter_centroid : is_incenter B C D P ∧ is_centroid A B C P) :
    AC = sqrt (5 / 2) ∧ BC = sqrt (5 / 2) :=
begin
  sorry, -- The proof would go here
end

end find_lengths_l293_293234


namespace min_people_like_both_mozart_and_bach_l293_293355

theorem min_people_like_both_mozart_and_bach (total : ℕ) (like_mozart : ℕ) 
(like_bach : ℕ) (num_people_both : ℕ) (h1 : total = 200) (h2 : like_mozart = 160) 
(h3 : like_bach = 150) (h4 : num_people_both = 110) : 
total ≥ like_mozart + like_bach - num_people_both
:= by {
  rw [h1, h2, h3, h4],
  linarith,
}

end min_people_like_both_mozart_and_bach_l293_293355


namespace equation_equiv_product_zero_l293_293712

theorem equation_equiv_product_zero (a b x y : ℝ) :
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 1) →
  ∃ (m n p : ℤ), (a^m * x - a^n) * (a^p * y - a^3) = a^5 * b^5 ∧ m * n * p = 0 :=
by
  intros h
  sorry

end equation_equiv_product_zero_l293_293712


namespace intersection_property_l293_293616

open Real EuclideanGeometry

noncomputable theory

def acute_triangle (A B C : Point) : Prop :=
  ∃ a b c : ℝ,
    ∆ABC a b c ∧
    a + b + c = 180 ∧
    a < 90 ∧ b < 90 ∧ c < 90

def circle_with_diameter (B C : Point) : set Point :=
  { P | dist B P * dist P C = (dist B C / 2)^2 }

theorem intersection_property 
  {A B C D E : Point} 
  (h_acute : acute_triangle A B C)
  (h_circle : ∀ P ∈ circle_with_diameter B C, P = D ∨ P = E)
  : dist E B * dist A B + dist D C * dist A C = dist B D ^ 2 := 
sorry

end intersection_property_l293_293616


namespace necessary_and_sufficient_condition_l293_293110

-- Definitions of geometrical elements
variables {A B C E F P O : Type*} [DecidableEq A] [Nonempty A]
variables {circleΓ : Type*} [DecidableEq circleΓ] [Nonempty circleΓ]

-- Parameters and hypotheses
variable (ABC : Triangle A B C)
variable (circleΓ_contains_A : circleΓ ∈ Vertex A ABC)
variables (circleΓ_intersects_AC_at_E : circleΓ ∈ side AC ∧ E ∈ circleΓ)
variables (circleΓ_intersects_AB_at_F : circleΓ ∈ side AB ∧ F ∈ circleΓ)
variable (circumcircleABC_intersects_circleΓ_at_P : circumcircle ABC ∩ circleΓ = {P})
variable (reflection_P_EF_lies_on_BC : (reflection P line EF) ∈ line BC)
variable (circumcenter_AB_enabled : circumcenter ABC = O)

-- The proof statement
theorem necessary_and_sufficient_condition :
  (reflection P line EF ∈ line BC ↔ circleΓ ∈ circumcenter_of (Triangle A B C)) :=
begin
  sorry
end

end necessary_and_sufficient_condition_l293_293110


namespace proof_monotonically_decreasing_proof_min_value_of_a_l293_293925

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

def g (a : ℝ) (x : ℝ) : ℝ := f a x + x

def tangent_line_condition (a : ℝ) : Prop :=
  let g' := λ (x : ℝ), 3 - a - 2 / x
  let g_tangent := λ (x : ℝ), (1 - a) * (x - 1) + 1
  g_tangent 0 = 2

def decreasing_interval (a : ℝ) : Set ℝ := { x | 0 < x ∧ x < 2 ∧ (3 - a - 2 / x < 0)}

def no_zeros_condition (a : ℝ) : Prop :=
  ∀ x, 0 < x ∧ x < 1/2 → (2 - a) * (x - 1) - 2 * Real.log x > 0

def min_value_of_a : Prop := 
  ∃ (a : ℝ), (∀ x, 0 < x ∧ x < 1/2 → (2 - a) * (x - 1) - 2 * Real.log x > 0) 
              ∧ a = 2 - 4 * Real.log 2

theorem proof_monotonically_decreasing (a : ℝ) (a_condition : tangent_line_condition a) :
  (∀ x, x ∈ decreasing_interval a) :=
sorry

theorem proof_min_value_of_a : min_value_of_a :=
sorry

end proof_monotonically_decreasing_proof_min_value_of_a_l293_293925


namespace simplest_square_root_is_b_l293_293389

theorem simplest_square_root_is_b (a b : ℝ) : 
  (\sqrt{a^2 + b^2} = \sqrt{a^2 + b^2}) ∧ 
  (4√a ≠ \sqrt{a^2 + b^2}) ∧ 
  (\sqrt{\frac{b}{a}} ≠ \sqrt{a^2 + b^2}) ∧ 
  (3√5 ≠ \sqrt{a^2 + b^2}) :=
by
  -- Prove that the given square root is the simplest
  sorry

end simplest_square_root_is_b_l293_293389


namespace probability_of_earning_2400_l293_293679

noncomputable def spinner_labels := ["Bankrupt", "$700", "$900", "$200", "$3000", "$800"]
noncomputable def total_possibilities := (spinner_labels.length : ℕ) ^ 3
noncomputable def favorable_outcomes := 6

theorem probability_of_earning_2400 :
  (favorable_outcomes : ℚ) / total_possibilities = 1 / 36 := by
  sorry

end probability_of_earning_2400_l293_293679


namespace complex_division_l293_293409

theorem complex_division : (10 * complex.I) / (2 - complex.I) = -2 + 4 * complex.I := 
by
  sorry

end complex_division_l293_293409


namespace solve_for_z_l293_293696

theorem solve_for_z :
  ∃ z : ℂ, 2 - (3 + complex.i) * z = 1 - (3 - complex.i) * z ∧ z = complex.i / 2 := 
by
  sorry

end solve_for_z_l293_293696


namespace lens_discount_l293_293671

def new_camera_cost (old_camera_cost : ℝ) : ℝ := 1.3 * old_camera_cost
def old_camera_cost : ℝ := 4000
def original_lens_cost : ℝ := 400
def total_paid : ℝ := 5400

theorem lens_discount : 
  let cost_of_new_camera := new_camera_cost old_camera_cost
      amount_paid_for_camera_and_lens := total_paid
      amount_paid_for_camera := cost_of_new_camera
      amount_paid_for_lens := total_paid - cost_of_new_camera
      discount := original_lens_cost - amount_paid_for_lens
  in discount = 200 :=
by
  sorry

end lens_discount_l293_293671
