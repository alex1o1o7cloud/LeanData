import Mathlib

namespace log_eq_six_implies_x_sixteen_l20_20135

theorem log_eq_six_implies_x_sixteen (x : ℝ) (hx : log 2 x + log 4 x = 6) : x = 16 :=
sorry

end log_eq_six_implies_x_sixteen_l20_20135


namespace find_principal_l20_20213

variable (P R : ℝ)
variable (condition1 : P + (P * R * 2) / 100 = 660)
variable (condition2 : P + (P * R * 7) / 100 = 1020)

theorem find_principal : P = 516 := by
  sorry

end find_principal_l20_20213


namespace combinations_of_5_choose_2_l20_20179

theorem combinations_of_5_choose_2 : nat.choose 5 2 = 10 :=
by sorry

end combinations_of_5_choose_2_l20_20179


namespace abc_inequality_l20_20099

open Real

noncomputable def posReal (x : ℝ) : Prop := x > 0

theorem abc_inequality (a b c : ℝ) 
  (hCond1 : posReal a) 
  (hCond2 : posReal b) 
  (hCond3 : posReal c) 
  (hCond4 : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
by
  sorry

end abc_inequality_l20_20099


namespace one_thirteenth_150th_digit_l20_20631

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20631


namespace one_div_thirteen_150th_digit_l20_20810

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20810


namespace percentage_reduction_correct_l20_20253

-- Define the initial conditions
def original_sheets : ℤ := 750
def original_lines_per_sheet : ℤ := 150
def original_characters_per_line : ℤ := 200
def retyped_lines_per_sheet : ℤ := 250
def retyped_characters_per_line : ℤ := 220

-- Define the total number of characters in the original document
def total_characters_original : ℤ := original_sheets * original_lines_per_sheet * original_characters_per_line

-- Define the number of characters per retyped sheet
def characters_per_retyped_sheet : ℤ := retyped_lines_per_sheet * retyped_characters_per_line

-- Calculate the number of retyped sheets needed without rounding
def retyped_sheets_exact : ℝ := total_characters_original.to_real / characters_per_retyped_sheet.to_real

-- The number of retyped sheets needed, rounded up to the nearest integer
def retyped_sheets : ℤ := retyped_sheets_exact.to_ceil.int_ceil

-- Calculate the reduction in the number of sheets
def reduction_in_sheets : ℤ := original_sheets - retyped_sheets

-- Calculate the percentage reduction in the number of sheets
def reduction_percentage : ℝ := (reduction_in_sheets.to_real / original_sheets.to_real) * 100

-- The theorem to be proved
theorem percentage_reduction_correct : reduction_percentage ≈ 45.33 := by
  sorry

end percentage_reduction_correct_l20_20253


namespace shaltaev_boltaev_proof_l20_20226

variable (S B : ℝ)

axiom cond1 : 175 * S > 125 * B
axiom cond2 : 175 * S < 126 * B

theorem shaltaev_boltaev_proof : 3 * S + B ≥ 1 :=
by {
  sorry
}

end shaltaev_boltaev_proof_l20_20226


namespace molecular_weight_of_compound_l20_20920

def n_weight : ℝ := 14.01
def h_weight : ℝ := 1.01
def br_weight : ℝ := 79.90

def molecular_weight : ℝ := (1 * n_weight) + (4 * h_weight) + (1 * br_weight)

theorem molecular_weight_of_compound :
  molecular_weight = 97.95 :=
by
  -- proof steps go here if needed, but currently, we use sorry to complete the theorem
  sorry

end molecular_weight_of_compound_l20_20920


namespace sampling_method_is_stratified_l20_20269

-- Definitions according to conditions in a)
def male_students : ℕ := 400
def female_students : ℕ := 600
def sampled_male_students : ℕ := 40
def sampled_female_students : ℕ := 60
def population_ratio : ℕ × ℕ := (4, 6)
def sampling_ratio : ℕ × ℕ := (4, 6)

-- Proof statement
theorem sampling_method_is_stratified
    (males : ℕ)
    (females : ℕ)
    (sampled_males : ℕ)
    (sampled_females : ℕ)
    (pop_ratio : ℕ × ℕ)
    (samp_ratio : ℕ × ℕ)
    (h1 : males = 400)
    (h2 : females = 600)
    (h3 : sampled_males = 40)
    (h4 : sampled_females = 60)
    (h5 : pop_ratio = (4, 6))
    (h6 : samp_ratio = (4, 6))
    : samp_ratio = pop_ratio → stratified_sampling (samp_ratio = pop_ratio) :=
by sorry

-- Additional placeholder to define stratified_sampling criteria
def stratified_sampling (h : Prop) : Prop := 
sorry

end sampling_method_is_stratified_l20_20269


namespace sum_of_radical_conjugates_l20_20294

theorem sum_of_radical_conjugates : 
  (8 - Real.sqrt 1369) + (8 + Real.sqrt 1369) = 16 :=
by
  sorry

end sum_of_radical_conjugates_l20_20294


namespace digit_150_after_decimal_of_one_thirteenth_l20_20869

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20869


namespace florist_roses_l20_20221

theorem florist_roses : (initial_roses sold_roses picked_roses : ℕ) (total_roses : ℕ) 
  (h1 : initial_roses = 37) (h2 : sold_roses = 16) (h3 : picked_roses = 19) (h4 : total_roses = initial_roses - sold_roses + picked_roses) : 
  total_roses = 40 :=
by
  sorry

end florist_roses_l20_20221


namespace greatest_int_value_not_satisfy_condition_l20_20915

/--
For the inequality 8 - 6x > 26, the greatest integer value 
of x that satisfies this is -4.
-/
theorem greatest_int_value (x : ℤ) : 8 - 6 * x > 26 → x ≤ -4 :=
by sorry

theorem not_satisfy_condition (x : ℤ) : x > -4 → ¬ (8 - 6 * x > 26) :=
by sorry

end greatest_int_value_not_satisfy_condition_l20_20915


namespace length_ED_l20_20222

structure Trapezoid (A B C D E : Type) (len_AB len_CD len_BD len_ED : ℝ) :=
  (is_parallel : ∀ AB CD : ℝ, AB = 3 * CD)
  (diagonals_intersect : ∀ A C B D : ℝ, ∃ E, E ∈ (diagonals_intersect A C B D))
  (len_BD : len_BD = 15)

theorem length_ED {ABCD : Trapezoid} (A B C D E : Type) (len_ED : ℝ) : len_ED = 15 / 4 :=
by
  sorry

end length_ED_l20_20222


namespace digit_150_of_1_div_13_l20_20796

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20796


namespace ratio_of_girls_to_boys_l20_20052

theorem ratio_of_girls_to_boys (g b : ℕ) (h1 : g = b + 6) (h2 : g + b = 36) : g / b = 7 / 5 := by sorry

end ratio_of_girls_to_boys_l20_20052


namespace abs_diff_A_B_l20_20992

def A : ℕ :=
  (Finset.range 24).sum (λ n, (2 * n + 1) * (2 * n + 2)) + 2 * 49

def B : ℕ :=
  (1 + (Finset.range 24).sum (λ n, (2 * n + 2) * (2 * n + 3))) + 48 * 49

theorem abs_diff_A_B :
  |A - B| = 1200 :=
by
  sorry

end abs_diff_A_B_l20_20992


namespace popsicle_sticks_left_l20_20303

-- Defining the conditions
def total_money : ℕ := 10
def cost_of_molds : ℕ := 3
def cost_of_sticks : ℕ := 1
def cost_of_juice_bottle : ℕ := 2
def popsicles_per_bottle : ℕ := 20
def initial_sticks : ℕ := 100

-- Statement of the problem
theorem popsicle_sticks_left : 
  let remaining_money := total_money - cost_of_molds - cost_of_sticks
  let bottles_of_juice := remaining_money / cost_of_juice_bottle
  let total_popsicles := bottles_of_juice * popsicles_per_bottle
  let sticks_left := initial_sticks - total_popsicles
  sticks_left = 40 := by
  sorry

end popsicle_sticks_left_l20_20303


namespace arithmetic_sequence_value_l20_20380

-- Definitions for the conditions
def arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n m : ℕ, a(n) = a(m) * q^(n-m)

variables {q : ℝ} (q_ne_one : q ≠ 1)
variables {a : ℕ → ℝ}
variables (h_seq : arithmetic_sequence a q)
variables (h_sum : a 0 + a 1 + a 2 + a 3 + a 4 = 6)
variables (h_sum_sq : a 0^2 + a 1^2 + a 2^2 + a 3^2 + a 4^2 = 18)

theorem arithmetic_sequence_value :
  a 0 - a 1 + a 2 - a 3 + a 4 = 3 :=
sorry

end arithmetic_sequence_value_l20_20380


namespace initial_volume_of_kola_solution_l20_20961

theorem initial_volume_of_kola_solution 
    (V : ℚ)
    (H1: 0.88 * V + 0.05 * V + 0.07 * V = V)
    (H2: 7.5% = ((0.07 * V + 3.2) / (V + 3.2 + 10 + 6.8))) : 
    V = 340 := 
sorry

end initial_volume_of_kola_solution_l20_20961


namespace max_power_speed_l20_20560

variables (C S ρ v₀ v : ℝ)

def force (C S ρ v₀ v : ℝ) : ℝ :=
  (C * S * ρ * (v₀ - v)^2) / 2

def power (C S ρ v₀ v : ℝ) : ℝ :=
  (C * S * ρ / 2) * v * (v₀^2 - 2 * v₀ * v + v^2)

theorem max_power_speed (C S ρ v₀ : ℝ) (hρ : ρ > 0) (hC : C > 0) (hS : S > 0) :
  ∃ v : ℝ, power C S ρ v₀ v = (C * S * ρ / 2) * (v₀^2 * (v₀/3) - 2 * v₀ * (v₀/3)^2 + (v₀/3)^3) := 
  sorry

end max_power_speed_l20_20560


namespace one_thirteen_150th_digit_l20_20907

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20907


namespace digit_150_of_decimal_1_div_13_l20_20668

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20668


namespace monotonicity_of_f_range_of_x_l20_20011
noncomputable theory

open Real

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * 2^x + b * 3^x

-- Theorem 1: Monotonicity when ab > 0
theorem monotonicity_of_f (a b : ℝ) (h : a * b > 0) :
  (∀ x y : ℝ, x < y → f a b x < f a b y) ∨ (∀ x y : ℝ, x < y → f a b x > f a b y) :=
sorry

-- Theorem 2: Range of x for f(x + 1) > f(x) when ab < 0
theorem range_of_x (a b : ℝ) (h : a * b < 0) :
  (a < 0 ∧ b > 0 → ∀ x : ℝ, f a b (x + 1) > f a b x → x > log (2 * b / -a) (3/2)) ∧
  (a > 0 ∧ b < 0 → ∀ x : ℝ, f a b (x + 1) > f a b x → x < log (2 * b / -a) (3/2)) :=
sorry

end monotonicity_of_f_range_of_x_l20_20011


namespace sum_of_squares_of_solutions_l20_20334

theorem sum_of_squares_of_solutions :
  let c := (1 : ℝ) / 2010
  let f := λ x : ℝ, x^2 - 2*x + c
  (|f x| = c) → (∑ s : {x // |f x| = c}, s.val^2) = 16108 / 2010 :=
by
  intros c f h,
  sorry

end sum_of_squares_of_solutions_l20_20334


namespace digit_150_of_1_over_13_is_3_l20_20769

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20769


namespace value_of_philosophy_l20_20549

theorem value_of_philosophy (value_philosophy : Prop)
  (cond1 : ∀ (P : Prop), P → P)
  (cond2 : ¬ (∀ (P : Prop), P ∧ true))
  (cond3 : ∀ (P : Prop), P → P)
  (cond4 : ∀ (P Q : Prop), (P ∧ Q) → P)
  (cond5 : ∀ (P Q : Prop), (P ∧ Q) → Q)
  (cond6 : (∃ (P Q : Prop), P ∧ Q) → Prop) :
  value_philosophy = ("Philosophy is the art of guiding people to live better lives") := by
  sorry

end value_of_philosophy_l20_20549


namespace digit_150_of_1_div_13_l20_20797

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20797


namespace one_div_thirteen_150th_digit_l20_20820

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20820


namespace rational_with_smallest_absolute_value_is_zero_l20_20579

theorem rational_with_smallest_absolute_value_is_zero (r : ℚ) :
  (forall r : ℚ, |r| ≥ 0) →
  (forall r : ℚ, r ≠ 0 → |r| > 0) →
  |r| = 0 ↔ r = 0 := sorry

end rational_with_smallest_absolute_value_is_zero_l20_20579


namespace probability_sum_even_l20_20607

/-- Suppose we have two wheels:
    - The first wheel has 6 segments with 3 even numbers and 3 odd numbers.
    - The second wheel has 5 segments with 2 even numbers and 3 odd numbers.
  The probability that the sum of the numbers from the two wheels is even is 1/2. -/
theorem probability_sum_even :
  let p_even_first_wheel := 3 / 6,
      p_odd_first_wheel := 3 / 6,
      p_even_second_wheel := 2 / 5,
      p_odd_second_wheel := 3 / 5 in
  (p_even_first_wheel * p_even_second_wheel +
   p_odd_first_wheel * p_odd_second_wheel = 1 / 2) :=
by sorry

end probability_sum_even_l20_20607


namespace digit_150_after_decimal_of_one_thirteenth_l20_20873

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20873


namespace digit_150_of_decimal_1_div_13_l20_20661

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20661


namespace decimal_150th_digit_l20_20890

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20890


namespace perfect_square_divisors_product_l20_20037

theorem perfect_square_divisors_product (n : ℕ) 
  (h_n_pos : n > 0) 
  (h_n_square : ∃ m : ℕ, n = m * m)
  (h_product_divisors : ∏ d in (finset.filter (λ k, n % k = 0) (finset.range (n + 1))), d = 1024) : 
  n = 16 :=
sorry

end perfect_square_divisors_product_l20_20037


namespace digit_150_after_decimal_of_one_thirteenth_l20_20867

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20867


namespace train_crossing_time_l20_20251

-- Definitions for the conditions
def speed_kmph : Float := 72
def speed_mps : Float := speed_kmph * (1000 / 3600)
def length_train_m : Float := 240.0416
def length_platform_m : Float := 280
def total_distance_m : Float := length_train_m + length_platform_m

-- The problem statement
theorem train_crossing_time :
  (total_distance_m / speed_mps) = 26.00208 :=
by
  sorry

end train_crossing_time_l20_20251


namespace triangle_area_sum_l20_20602

theorem triangle_area_sum (P Q R : triangle) (r R : ℝ) (h₁ : r = 6) (h₂ : R = 17)
  (h₃ : 3 * (cos (angle Q)) = cos (angle P) + cos (angle R)) :
  ∃ (p q r : ℕ), (area P Q R = p * sqrt q / r) ∧ p.gcd r = 1 ∧ q.nat_abs.sqrt_free ∧ p + q + r = 152 :=
by
  sorry

end triangle_area_sum_l20_20602


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20732

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20732


namespace solve_for_x_l20_20540

theorem solve_for_x : ∃ (x : ℤ), 24 * 2 - 6 = 3 * x + 6 ∧ x = 12 :=
by
  use 12
  split
  sorry
  rfl

end solve_for_x_l20_20540


namespace pure_imaginary_iff_m_eq_3_first_quadrant_iff_m_range_l20_20101

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def in_first_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im > 0

def complex_z (m : ℝ) : ℂ := complex.mk (m^2 - 2 * m - 3) (m^2 + 3 * m + 2)

theorem pure_imaginary_iff_m_eq_3 (m : ℝ) : 
  is_pure_imaginary (complex_z m) ↔ m = 3 := 
by sorry

theorem first_quadrant_iff_m_range (m : ℝ) :
  in_first_quadrant (complex_z m) ↔ m < -2 ∨ m > 3 := 
by sorry

end pure_imaginary_iff_m_eq_3_first_quadrant_iff_m_range_l20_20101


namespace smallest_k_674_l20_20355

theorem smallest_k_674 :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 2017) → (S.card = 674) → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ (672 < a - b) ∧ (a - b < 1344) ∨ (672 < b - a) ∧ (b - a < 1344) :=
by sorry

end smallest_k_674_l20_20355


namespace digit_150_after_decimal_of_one_thirteenth_l20_20875

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20875


namespace sample_correlation_negative_one_l20_20375

noncomputable def sample_correlation {n : ℕ} (x y : Fin n → ℝ) : ℝ :=
  sorry  -- replace with actual implementation

theorem sample_correlation_negative_one
  (n : ℕ) (x y : Fin n → ℝ) 
  (hne : ∃i j: Fin n, i ≠ j ∧ x i ≠ x j)
  (hline : ∀ i, y i = -1/2 * x i + 1) :
  n ≥ 2 → sample_correlation x y = -1 :=
by
  sorry

end sample_correlation_negative_one_l20_20375


namespace scientific_notation_two_sig_figs_l20_20572

def land_area := 148000000

theorem scientific_notation_two_sig_figs :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ land_area = a * 10^n ∧ (round (a * 10^1) / 10 = 1.5) ∧ n = 8 := by
sorry

end scientific_notation_two_sig_figs_l20_20572


namespace max_ab_l20_20426

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ f : ℝ → ℝ, (∀ x, f x = 4 * x ^ 3 - a * x ^ 2 - 2 * b * x + 2) → (f' 1 = 0)) :
  ab ≤ 9 :=
begin
  sorry
end

end max_ab_l20_20426


namespace digit_150_after_decimal_of_one_thirteenth_l20_20863

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20863


namespace gwen_did_not_recycle_2_bags_l20_20026

def points_per_bag : ℕ := 8
def total_bags : ℕ := 4
def points_earned : ℕ := 16

theorem gwen_did_not_recycle_2_bags : total_bags - points_earned / points_per_bag = 2 := by
  sorry

end gwen_did_not_recycle_2_bags_l20_20026


namespace transmitted_word_l20_20950

noncomputable def f (x y : ℕ) : ℕ := (x + 4 * y) % 10

theorem transmitted_word :
  ∃ (m₁ m₂ m₃ m₄ m₅ m₆ : ℕ), 
  f m₁ 1 = 3 ∧
  f m₂ (f m₁ 1) = 3 ∧
  f m₃ (f m₂ (f m₁ 1)) = 7 ∧
  f m₄ (f m₃ (f m₂ (f m₁ 1))) = 7 ∧
  f m₅ (f m₄ (f m₃ (f m₂ (f m₁ 1)))) = 4 ∧
  f m₆ (f m₅ (f m₄ (f m₃ (f m₂ (f m₁ 1))))) = 1 ∧
  (m₁, m₂, m₃) = (12, 4, 2).

end transmitted_word_l20_20950


namespace distinctThreeDigitIntegers_count_l20_20028

def digitSet := {1, 1, 4, 4, 4, 7, 8}
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def hasValidDigits (n : ℕ) : Prop := (∀ digit ∈ toList n, digit ∈ digitSet) ∧
                                      (∀ d, count d (toList n) ≤ count d digitSet)
def numDistinctThreeDigitIntegers : ℕ := 43

theorem distinctThreeDigitIntegers_count :
  (∑ n in finset.filter (λ n, isThreeDigit n ∧ hasValidDigits n) (finset.range 1000)) = numDistinctThreeDigitIntegers := sorry

end distinctThreeDigitIntegers_count_l20_20028


namespace digit_150_after_decimal_of_one_thirteenth_l20_20861

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20861


namespace part1_part2_part3_l20_20010

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (1/9)^x - 2 * a * (1/3)^x + 3

-- Define the minimum value function h(a)
noncomputable def h (a : ℝ) : ℝ :=
if a <= (1/3) then (28/9) - (2*a/3)
else if a < 3 then 3 - a^2
else 12 - 6*a

-- Proving part 1: Range of f when a = 1
theorem part1 : ∀ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → f x 1 ∈ Set.Icc 2 6 :=
sorry

-- Proving part 2: The minimum value function h(a)
theorem part2 : ∀ (a : ℝ), h(a) = 
(if a <= (1/3) then (28/9) - (2*a/3)
else if a < 3 then 3 - a^2
else 12 - 6*a) :=
sorry

-- Proving part 3: Non-existence of such m and n
theorem part3 : ¬ (∃ (m n : ℝ), m > n ∧ n > 3 ∧ (∀ a ∈ Set.Icc n m, h(a) ∈ Set.Icc (n^2) (m^2))) :=
sorry

end part1_part2_part3_l20_20010


namespace range_of_f_l20_20171

def f (x : ℝ) : ℝ := Real.sqrt (3 - x^2) + Real.sqrt (x^2 - 3)

theorem range_of_f :
  {y : ℝ | ∃ (x : ℝ), (3 - x^2 ≥ 0) ∧ (x^2 - 3 ≥ 0) ∧ (y = f x)} = {0} :=
by
  sorry

end range_of_f_l20_20171


namespace presidency_meeting_ways_l20_20963

-- Defining the number of schools and members per school
def number_of_schools : ℕ := 4
def members_per_school : ℕ := 6

-- Using the given conditions
def total_members : ℕ := number_of_schools * members_per_school

-- Define a function that calculates the number of ways to choose k representatives out of n members
def choose (n k : ℕ) : ℕ := n.choose k

-- Main theorem statement
theorem presidency_meeting_ways : 
  ∃ (total_ways : ℕ), total_ways = 
    choose number_of_schools 1 * 
    choose members_per_school 3 * 
    members_per_school ^ (number_of_schools - 1) := 
begin
  -- Specify the total number of ways in the existence statement
  use 17280,
  sorry
end

end presidency_meeting_ways_l20_20963


namespace converse_false_inverse_false_contrapositive_true_l20_20543

theorem converse_false (x : ℝ) : (x^2 - 5*x + 6 = 0) → (x = 2) = False :=
by
  assume h1 : x^2 - 5*x + 6 = 0
  have h2 : (x - 2)*(x - 3) = 0 := by sorry -- Proof of factorization
  have h3 : x = 2 ∨ x = 3 := by sorry -- Roots of the equation
  show (x = 2) = False := by sorry -- Proof that x could also be 3 

theorem inverse_false (x : ℝ) : (x ≠ 2) → (x^2 - 5*x + 6 ≠ 0) = False :=
by
  assume h1 : x ≠ 2
  have h2 : x = 3 := by sorry -- Since x ≠ 2 and 3 satisfies the equation
  have h3 : x^2 - 5*x + 6 = 0 := by sorry -- Verification
  show (x^2 - 5*x + 6 ≠ 0) = False := by sorry -- Proof 

theorem contrapositive_true (x : ℝ) : (x^2 - 5*x + 6 ≠ 0) → (x ≠ 2) :=
by
  assume h1 : x^2 - 5*x + 6 ≠ 0
  have h2 : x = 2 ∨ x = 3 := by sorry -- Given the solutions from the quadratic equation
  show x ≠ 2 := by sorry -- Proof by exclusionary factor as x cannot be 2 if the equation isn't 0

end converse_false_inverse_false_contrapositive_true_l20_20543


namespace perp_lines_k_value_l20_20434

theorem perp_lines_k_value (k : ℝ) :
  (∀ x : ℝ, y_1 x = 2 * x - 1 ∧ y_2 x = k * x → y_1 x * y_2 x = -1) → k = -1 / 2 :=
by
  intros h
  have h1 : ( ∀ x : ℝ, 2 * x * (k * x) = -1), from sorry
  exact sorry

end perp_lines_k_value_l20_20434


namespace find_a13_l20_20585

-- Define the arithmetic sequence a_n
def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

-- Define the sum of the first n terms S_n of the arithmetic sequence
def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ := n * a1 + n * (n - 1) / 2 * d

theorem find_a13 (a1 d : ℝ) (h1 : arithmetic_sequence a1 d 5 = 3) 
  (h2 : sum_arithmetic_sequence a1 d 5 = 10) : arithmetic_sequence a1 d 13 = 7 := 
by 
  sorry

end find_a13_l20_20585


namespace one_over_thirteen_150th_digit_l20_20678

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20678


namespace shaded_rectangle_area_is_one_fourth_l20_20152

namespace RectangleArea

-- Definitions based on conditions
def AB : ℝ := 1 -- Length of AB in meters
def AD : ℝ := 4 -- Length of AD in meters
def AE : ℝ := AD / 2 -- Midpoint E of AD
def AG : ℝ := AB / 2 -- Midpoint G of AB
def AF : ℝ := AE / 2 -- Midpoint F of AE
def AH : ℝ := AG / 2 -- Midpoint H of AG

-- Define the area calculation of the shaded rectangle
def area_shaded_rectangle : ℝ := AF * AH

-- The proof statement
theorem shaded_rectangle_area_is_one_fourth :
  area_shaded_rectangle = 1 / 4 := by
  sorry

end RectangleArea

end shaded_rectangle_area_is_one_fourth_l20_20152


namespace one_thirteenth_150th_digit_l20_20627

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20627


namespace correct_statements_count_l20_20083

noncomputable def probability_of_events := sorry

theorem correct_statements_count :
  let M N : Type
  let P : M ∩ N → ℝ
  let MutuallyExclusiveEvents := ∀ a b : M × N, a ∩ b = ∅ → P(a) + P(b) := 9/20
  let IndependentEvents := ∀ a b : M × N, P(a ∩ b) = P(a) * P(b) := 1/6
  (P(M) = 1/5 ∧ P(N) = 1/4 → P(M ∪ N) = 9/20) ∧
  (P(M) = 1/2 ∧ P(N) = 1/3 ∧ P(M ∩ N) = 1/6 → IndependentEvents) ∧
  (P(¬M) = 1/2 ∧ P(N) = 1/3 ∧ P(M ∩ N) = 1/6 → IndependentEvents) ∧
  (P(M) = 1/2 ∧ P(¬N) = 1/3 ∧ P(M ∩ N) = 1/6 → ¬IndependentEvents) ∧
  (P(M) = 1/2 ∧ P(N) = 1/3 ∧ P(¬(M ∩ N)) = 5/6 → ¬IndependentEvents)
→ 3 = 3 :=
begin
  sorry
end

end correct_statements_count_l20_20083


namespace arc_intersections_l20_20342

def valid_k (k : ℕ) : Prop :=
  k < 100 ∧ ¬ (k + 1) % 8 = 0

theorem arc_intersections (k : ℕ) :
  valid_k k → 
  ∃ (arcs : Finset ℕ), 
    arcs.card = 100 ∧ 
    (∀ i ∈ arcs, ∃ (intersections : Finset ℕ), intersections.card = k ∧ i ≠ intersections ∧ intersections ⊆ arcs) 
:= by
  sorry

end arc_intersections_l20_20342


namespace final_position_3000_l20_20972

def initial_position : ℤ × ℤ := (0, 0)
def moves_up_first_minute (pos : ℤ × ℤ) : ℤ × ℤ := (pos.1, pos.2 + 1)

def next_position (n : ℕ) (pos : ℤ × ℤ) : ℤ × ℤ :=
  if n % 4 = 0 then (pos.1 + n, pos.2)
  else if n % 4 = 1 then (pos.1, pos.2 + n)
  else if n % 4 = 2 then (pos.1 - n, pos.2)
  else (pos.1, pos.2 - n)

def final_position (minutes : ℕ) : ℤ × ℤ := sorry

theorem final_position_3000 : final_position 3000 = (0, 27) :=
by {
  -- logic to compute final_position
  sorry -- proof exists here
}

end final_position_3000_l20_20972


namespace find_r_interval_find_100p_plus_q_l20_20128

noncomputable theory
open Complex

def repetitive (r : ℝ) :=
  ∃ (z1 z2 : ℂ), |z1| = 1 ∧ |z2| = 1 ∧ z1 ≠ z2 ∧ {z1, z2} ≠ {-Complex.i, Complex.i} ∧ 
  z1 * (z1^3 + z1^2 + (r : ℂ) * z1 + 1) = z2 * (z2^3 + z2^2 + (r : ℂ) * z2 + 1)

theorem find_r_interval :
  ∃ a b : ℝ, (∀ r : ℝ, repetitive r ↔ a < r ∧ r ≤ b) ∧ |a| + |b| = 25/4 :=
sorry

theorem find_100p_plus_q :
  let ⟨a, b, hab, hab_sum⟩ := find_r_interval in
  ∃ (p q : ℕ), Nat.gcd p q = 1 ∧ 
               (|a| + |b| = p / q) ∧ 
               (100 * p + q = 2504) :=
sorry

end find_r_interval_find_100p_plus_q_l20_20128


namespace remainder_of_exponentiation_l20_20489

theorem remainder_of_exponentiation (n : ℕ) : (3 ^ (2 * n) + 8) % 8 = 1 := 
by sorry

end remainder_of_exponentiation_l20_20489


namespace digit_150_of_one_thirteenth_l20_20838

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20838


namespace smallest_k_exists_l20_20346

theorem smallest_k_exists (s : Finset ℕ) :
  (∀ a b ∈ s, a ≠ b → (672 < |a - b| ∧ |a - b| < 1344)) →
  (∀ k, k < 674 → ∃ s : Finset ℕ, s.card = k ∧ (∀ a b ∈ s, a ≠ b → ¬ (672 < |a - b| ∧ |a - b| < 1344))) → False :=
begin
  sorry
end

end smallest_k_exists_l20_20346


namespace one_over_thirteen_150th_digit_l20_20679

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20679


namespace least_months_for_duplicate_committee_l20_20513

open Nat

def volunteer_group_members : ℕ := 13
def women_in_group : ℕ := 6
def men_in_group : ℕ := 7
def committee_size : ℕ := 5

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

lemma binomial_coefficient_13_5 : choose volunteer_group_members committee_size = 1287 := by sorry
lemma binomial_coefficient_6_5 : choose women_in_group committee_size = 6 := by sorry
lemma binomial_coefficient_7_5 : choose men_in_group committee_size = 21 := by sorry

def valid_committees : ℕ := choose volunteer_group_members committee_size - choose women_in_group committee_size - choose men_in_group committee_size

lemma valid_committees_count : valid_committees = 1260 := by sorry

theorem least_months_for_duplicate_committee :
  ∃ m : ℕ, m = valid_committees + 1 ∧ m = 1261 := by
  existsi 1261
  split
  . refl
  . exact valid_committees_count.symm

end least_months_for_duplicate_committee_l20_20513


namespace solve_equation_l20_20136

theorem solve_equation : 
  ∀ x : ℝ, x ≠ 1 → 
  (1 + 3 * (x - 1) = -3 * x) ↔ ((1 / (x - 1) + 3) * (x - 1) = (3 * x) / (1 - x) * (x - 1)) :=
by 
  intros x hx,
  split,
  { -- Prove the forward direction, i.e., from simplified form to original form
    intro h,
    sorry },
  { -- Prove the backward direction, i.e., from original form to simplified form
    intro h,
    sorry }

end solve_equation_l20_20136


namespace tangent_line_eqn_extreme_values_l20_20013

/-- The tangent line to the function f at (0, 5) -/
theorem tangent_line_eqn (f : ℝ → ℝ) (h : ∀ x, f x = (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 - 2 * x + 5) :
  (∃ k b, (∀ x, f x = k * x + b) ∧ k = -2 ∧ b = 5) ∧ (2 * 0 + 5 - 5 = 0) := by
  sorry

/-- The function f has a local maximum at x = -1 and a local minimum at x = 2 -/
theorem extreme_values (f : ℝ → ℝ) (h : ∀ x, f x = (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 - 2 * x + 5) :
  (∃ x₁ x₂, x₁ = -1 ∧ f x₁ = 37 / 6 ∧ x₂ = 2 ∧ f x₂ = 5 / 3) := by
  sorry

end tangent_line_eqn_extreme_values_l20_20013


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20741

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20741


namespace intersection_eq_l20_20019

def setA : set ℝ := {x | x < 1}
def setB : set ℝ := {x | x^2 - x - 6 < 0}

theorem intersection_eq : (setA ∩ setB) = {x | -2 < x ∧ x < 1} :=
by sorry

end intersection_eq_l20_20019


namespace one_div_thirteen_150th_digit_l20_20779

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20779


namespace shaded_area_eq_l20_20237

noncomputable def radius_small : ℝ := 2
noncomputable def radius_large : ℝ := 3
noncomputable def area_shaded : ℝ := (5.65 / 3) * Real.pi - 2.9724 * Real.sqrt 5

theorem shaded_area_eq : 
  (∃ A B C : ℝ × ℝ, 
    ∀ (x : ℝ × ℝ), x ∈ { p | Real.norm (p.1 - C.1, p.2 - C.2) = radius_small } → 
    (Real.norm (x.1 - A.1, x.2 - A.2) = radius_large ∧ Real.norm (x.1 - B.1, x.2 - B.2) = radius_large) ∧ 
    Real.norm (A.1 - B.1, A.2 - B.2) = 2 * radius_small) → 
  ∃ R : ℝ, R = area_shaded := 
by {
  sorry
}

end shaded_area_eq_l20_20237


namespace ratio_of_distances_l20_20207

-- Definitions based on conditions in a)
variables (x y w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_w : 0 ≤ w)
variables (h_w_ne_zero : w ≠ 0) (h_y_ne_zero : y ≠ 0) (h_eq_times : y / w = x / w + (x + y) / (9 * w))

-- The proof statement
theorem ratio_of_distances (x y w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y)
  (h_nonneg_w : 0 ≤ w) (h_w_ne_zero : w ≠ 0) (h_y_ne_zero : y ≠ 0)
  (h_eq_times : y / w = x / w + (x + y) / (9 * w)) :
  x / y = 4 / 5 :=
sorry

end ratio_of_distances_l20_20207


namespace vector_sum_l20_20082

variables {V : Type*} [add_comm_group V] [vector_space ℝ V] 

-- Define vector GA, GB, GC as elements of the vector space V.
variables (G A B C : V)

-- Suppose G is the centroid of the triangle ABC.
def is_centroid (G A B C : V) : Prop := 
  G = (1/3 : ℝ) • (A + B + C)

-- Given the fundamental property of centroid.
axiom centroid_property (G A B C : V) (h : is_centroid G A B C) : (G - A) + (G - B) + (G - C) = 0

-- The proof problem statement.
theorem vector_sum (G A B C : V) (h : (G - A) + (G - B) + (G - C) = 0) : 
  (G - A) + 2 • (G - B) + 3 • (G - C) = C - A :=
sorry

end vector_sum_l20_20082


namespace digit_150th_of_fraction_l20_20713

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20713


namespace find_certain_number_l20_20586

theorem find_certain_number (N : ℝ) 
  (h : 3.6 * N * 2.50 / (0.12 * 0.09 * 0.5) = 800.0000000000001)
  : N = 0.48 :=
sorry

end find_certain_number_l20_20586


namespace integer_expression_l20_20522

theorem integer_expression (m : ℤ) : ∃ k : ℤ, k = (m / 3) + (m^2 / 2) + (m^3 / 6) :=
sorry

end integer_expression_l20_20522


namespace sally_cards_problem_l20_20529

-- Condition definitions
def red_cards := {1, 2, 3, 4, 5, 6} -- Red cards numbered 1 through 6
def blue_cards := {2, 3, 4, 5, 6, 7, 8} -- Blue cards numbered 2 through 8
def alternates (l : List Nat) : Prop := 
  ∀ (i : Nat), i < l.length - 1 → (red_cards.contains (l.get i) ↔ blue_cards.contains (l.get (i + 1)))

def divides_correctly (l : List Nat) : Prop := 
  ∀ (i j : Nat), i < l.length ∧ j = i + 1 → 
    (red_cards.contains (l.get i) ∧ blue_cards.contains (l.get j) → l.get j % l.get i = 0) ∧ 
    (blue_cards.contains (l.get i) ∧ red_cards.contains (l.get j) → l.get j % l.get i = 0)

def middle_cards_sum_correct (l : List Nat) : Prop :=
  (l.length = 9 ∧ l.nth 3.getD 0 + l.nth 4.getD 0 + l.nth 5.getD 0 = 17)

theorem sally_cards_problem :
  ∃ (l : List Nat), alternates l ∧ divides_correctly l ∧ middle_cards_sum_correct l :=
sorry

end sally_cards_problem_l20_20529


namespace digit_150_after_decimal_of_one_thirteenth_l20_20866

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20866


namespace part_one_part_two_l20_20317

noncomputable def e := real.exp 1

def f (x : ℝ) : ℝ := x / e + 1 / (e * x)

theorem part_one (x₁ x₂ : ℝ) (h₁ : 1 ≤ x₁) (h₂ : 1 ≤ x₂) (h₃ : x₁ ≠ x₂) :
  (f x₂ - f x₁) / (x₂ - x₁) > 0 := sorry

theorem part_two (a : ℝ) (h : f (|a| + 3) > f (|a - 4| + 1)) : a > 1 := sorry

end part_one_part_two_l20_20317


namespace find_S30_l20_20060

variable {S : ℕ → ℝ} -- Assuming S is a function from natural numbers to real numbers

-- Arithmetic sequence is defined such that the sum of first n terms follows a specific format
def is_arithmetic_sequence (S : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, S (n + 1) - S n = d

-- Given conditions
axiom S10 : S 10 = 4
axiom S20 : S 20 = 20
axiom S_arithmetic : is_arithmetic_sequence S

-- The equivalent proof problem
theorem find_S30 : S 30 = 48 :=
by
  sorry

end find_S30_l20_20060


namespace digit_150_of_1_over_13_is_3_l20_20773

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20773


namespace regular_hexagon_has_greatest_lines_of_symmetry_l20_20204

-- Definitions for the various shapes and their lines of symmetry.
def regular_pentagon_lines_of_symmetry : ℕ := 5
def parallelogram_lines_of_symmetry : ℕ := 0
def oval_ellipse_lines_of_symmetry : ℕ := 2
def right_triangle_lines_of_symmetry : ℕ := 0
def regular_hexagon_lines_of_symmetry : ℕ := 6

-- Theorem stating that the regular hexagon has the greatest number of lines of symmetry.
theorem regular_hexagon_has_greatest_lines_of_symmetry :
  regular_hexagon_lines_of_symmetry > regular_pentagon_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > parallelogram_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > oval_ellipse_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > right_triangle_lines_of_symmetry :=
by
  sorry

end regular_hexagon_has_greatest_lines_of_symmetry_l20_20204


namespace multiply_binomials_l20_20112

variable (x : ℝ)

theorem multiply_binomials :
  (4 * x - 3) * (x + 7) = 4 * x ^ 2 + 25 * x - 21 :=
by
  sorry

end multiply_binomials_l20_20112


namespace one_thirteen_150th_digit_l20_20896

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20896


namespace digit_150_of_one_thirteenth_l20_20837

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20837


namespace digit_150th_of_fraction_l20_20724

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20724


namespace face_with_fewer_than_six_sides_l20_20521

theorem face_with_fewer_than_six_sides (P : ℕ) (Γ : ℕ) (B : ℕ) 
  (Γ_t : ℕ → ℕ) 
  (h1 : 2 * P = ∑ t in Finset.range (Γ + 1), t * Γ_t t)
  (h2 : P = Γ + B - 2) :
  ∃ t, t < 6 ∧ Γ_t t > 0 :=
by
  sorry

end face_with_fewer_than_six_sides_l20_20521


namespace grades_assignment_l20_20273

theorem grades_assignment (n : ℕ) (grades : finset char) (h_n : n = 12) (h_grades : grades.card = 3) : 
  (grades.card ^ n) = 531441 := by
  sorry

end grades_assignment_l20_20273


namespace points_on_decreasing_line_y1_gt_y2_l20_20389
-- Import the necessary library

-- Necessary conditions and definitions
variable {x y : ℝ}

-- Given points P(3, y1) and Q(4, y2)
def y1 : ℝ := -2*3 + 4
def y2 : ℝ := -2*4 + 4

-- Lean statement to prove y1 > y2
theorem points_on_decreasing_line_y1_gt_y2 (h1 : y1 = -2 * 3 +4) (h2 : y2 = -2 * 4 + 4) : 
  y1 > y2 :=
sorry  -- Proof steps go here

end points_on_decreasing_line_y1_gt_y2_l20_20389


namespace principal_is_correct_l20_20978

-- Define the conditions as variables/constants
variables (SI : ℝ) (R : ℝ) (T : ℝ)

-- Given conditions with specified values
def given_conditions : Prop :=
  SI = 4016.25 ∧ R = 9 ∧ T = 5

-- Define the formula for Simple Interest
def simple_interest (P : ℝ) : ℝ :=
  (P * R * T) / 100

-- The goal to prove that the Principal (P) is 8925
theorem principal_is_correct (P : ℝ) (h : given_conditions) :
  simple_interest P = SI → P = 8925 :=
by
  -- Just outline the proof structure, actual proof details will be filled in by proving
  sorry

end principal_is_correct_l20_20978


namespace three_digit_numbers_count_l20_20535

theorem three_digit_numbers_count :
  (∑ (s : Finset (ℕ × ℕ × ℕ)) in
    { (a, b, c) : ℕ × ℕ × ℕ |
      (a ∈ {0, 2, 4}) ∧ (b, c) ∈ ({1, 3, 5} × {1, 3, 5}).filter (λ bc, bc.1 ≠ bc.2) ∧ 
      a ≠ b ∧ a ≠ c ∧
      (a = 0 → b ≠ 0 ∧ c ≠ 0) ∧ 100 * a + 10 * b + c ∈ (100 * {0, 2, 4} + 10 * {1, 3, 5} + {1, 3, 5}) },
    1) = 48 := sorry

end three_digit_numbers_count_l20_20535


namespace max_radius_of_circle_in_triangle_inscribed_l20_20079

theorem max_radius_of_circle_in_triangle_inscribed (ω : Set (ℝ × ℝ)) (hω : ∀ (P : ℝ × ℝ), P ∈ ω → P.1^2 + P.2^2 = 1)
  (O : ℝ × ℝ) (hO : O = (0, 0)) (P : ℝ × ℝ) (hP : P ∈ ω) (A : ℝ × ℝ) 
  (hA : A = (P.1, 0)) : 
  (∃ r : ℝ, r = (Real.sqrt 2 - 1) / 2) :=
by
  sorry

end max_radius_of_circle_in_triangle_inscribed_l20_20079


namespace pumpkins_eaten_l20_20533

-- Definitions for the conditions
def originalPumpkins : ℕ := 43
def leftPumpkins : ℕ := 20

-- Theorem statement
theorem pumpkins_eaten : originalPumpkins - leftPumpkins = 23 :=
  by
    -- Proof steps are omitted
    sorry

end pumpkins_eaten_l20_20533


namespace ab_sum_l20_20067

theorem ab_sum (A B C D : Nat) (h_digits: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_mult : A * (10 * C + D) = 1001 + 100 * A + 10 * B + A) : A + B = 1 := 
  sorry

end ab_sum_l20_20067


namespace divisor_is_twelve_l20_20953

theorem divisor_is_twelve (d : ℕ) (h : 64 = 5 * d + 4) : d = 12 := 
sorry

end divisor_is_twelve_l20_20953


namespace second_group_men_count_l20_20225

theorem second_group_men_count : 
  ∀ (M : ℕ), (16 * 28 = M * 22.4) → M = 20 :=
by
  sorry

end second_group_men_count_l20_20225


namespace trapezoid_area_isosceles_l20_20557

theorem trapezoid_area_isosceles
  (BC : ℝ)
  (P : ℝ)
  (isosceles : ∃ (A B C D : ℝ), B = C ∧ ∠BCA = ∠ACD ∧ ∠BCA = ∠CAD)
  (diagonal_bisects : ∀ A B C D : ℝ, isosceles → ∠BCA = ∠ACD)
  (shorter_base : BC = 3)
  (perimeter : P = 42) :
  (∃ area : ℝ, area = 96) :=
sorry

end trapezoid_area_isosceles_l20_20557


namespace area_of_combined_rectangle_l20_20343

theorem area_of_combined_rectangle
  (short_side : ℝ) (num_small_rectangles : ℕ) (total_area : ℝ)
  (h1 : num_small_rectangles = 4)
  (h2 : short_side = 7)
  (h3 : total_area = (3 * short_side + short_side) * (2 * short_side)) :
  total_area = 392 := by
  sorry

end area_of_combined_rectangle_l20_20343


namespace quadratic_inequality_solution_l20_20321

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 3 * x - 18 < 0 ↔ -6 < x ∧ x < 3 := 
sorry

end quadratic_inequality_solution_l20_20321


namespace sam_money_left_l20_20125

theorem sam_money_left (initial_amount : ℕ) (book_cost : ℕ) (number_of_books : ℕ) (initial_amount_eq : initial_amount = 79) (book_cost_eq : book_cost = 7) (number_of_books_eq : number_of_books = 9) : initial_amount - book_cost * number_of_books = 16 :=
by
  rw [initial_amount_eq, book_cost_eq, number_of_books_eq]
  norm_num
  sorry

end sam_money_left_l20_20125


namespace decimal_1_div_13_150th_digit_is_3_l20_20845

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20845


namespace train_crossing_time_l20_20942

-- Define the length of the train in meters
def train_length : ℝ := 100

-- Define the speed of the train in meters per second (converted from 162 km/hr)
def train_speed : ℝ := 45

-- Define the expected time to cross the pole
def expected_time : ℝ := 2.22

-- Prove that the time taken by the train to cross the electric pole is approximately the expected time
theorem train_crossing_time :
  train_length / train_speed ≈ expected_time :=
sorry

end train_crossing_time_l20_20942


namespace digit_150_of_one_thirteenth_l20_20843

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20843


namespace decimal_150th_digit_l20_20642

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20642


namespace average_speed_l20_20217

theorem average_speed (d1 d2 t1 t2 : ℝ) :
  d1 = 98 ∧ d2 = 60 ∧ t1 = 1 ∧ t2 = 1 → (d1 + d2) / (t1 + t2) = 79 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  rw [h1, h3, h5, h6]
  norm_num

end average_speed_l20_20217


namespace min_value_of_a_l20_20376

noncomputable def a_seq : ℕ → ℚ
| 1 := 1/3
| (n+1) := a_seq n * a_seq 1

def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
  
theorem min_value_of_a (a : ℝ) (h : ∀ n : ℕ, (S_n n : ℝ) < a) : a ≥ 1/2 :=
sorry

end min_value_of_a_l20_20376


namespace non_empty_proper_subsets_of_A_l20_20017

def A := {2, 3}

theorem non_empty_proper_subsets_of_A :
  (∃ S, S ⊂ A ∧ S = {2}) ∧ (∃ S, S ⊂ A ∧ S = {3}) :=
by
  sorry

end non_empty_proper_subsets_of_A_l20_20017


namespace one_div_thirteen_150th_digit_l20_20776

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20776


namespace solution_set_of_inequality_l20_20405

def f (x : ℝ) : ℝ := (1 / 3) * x^3 + x

theorem solution_set_of_inequality :
  ∀ f : ℝ → ℝ, (∀ x : ℝ, f (-x) = -f x) → (∀ x y : ℝ, x < y → f x < f y) →
  ∀ x : ℝ, f (2 - x^2) + f (2 * x + 1) > 0 ↔ -1 < x ∧ x < 3 :=
by
  intros f odd monotone x
  sorry

end solution_set_of_inequality_l20_20405


namespace first_player_wins_l20_20450

-- Definition of the chessboard and movement
def position : Type := ℕ × ℕ

def move_right (p : position) : position := (p.1 + 1, p.2)
def move_up (p : position) : position := (p.1, p.2 + 1)
def move_diagonal (p : position) : position := (p.1 + 1, p.2 + 1)

-- The starting position and the goal position
def start_pos : position := (1, 1)
def goal_pos : position := (8, 8)

-- Definition of a player winning from a position
def wins (p : position) : Prop :=
  ∃ path : list position, 
    path.head = some p ∧ path.last = some goal_pos ∧
    ∀ i < path.length - 1, 
      (path.nth i = some p ∧ 
      (move_right p = path.nth (i + 1).some) ∨ 
      (move_up p = path.nth (i + 1).some) ∨ 
      (move_diagonal p = path.nth (i + 1).some))

-- Theorem stating the first player with perfect strategy wins
theorem first_player_wins : wins start_pos :=
sorry 

end first_player_wins_l20_20450


namespace kirill_height_l20_20472

theorem kirill_height (K B : ℕ) (h1 : K = B - 14) (h2 : K + B = 112) : K = 49 :=
by
  sorry

end kirill_height_l20_20472


namespace smallest_part_of_80_divided_by_proportion_l20_20424

theorem smallest_part_of_80_divided_by_proportion (x : ℕ) (h1 : 1 * x + 3 * x + 5 * x + 7 * x = 80) : x = 5 :=
sorry

end smallest_part_of_80_divided_by_proportion_l20_20424


namespace rectangle_is_cyclic_l20_20280

theorem rectangle_is_cyclic (Q : Type*) [quadrilateral Q] :
  (Q = rectangle) → (∃ P : point, ∀ V ∈ vertices Q, distance P V = r) :=
by
  sorry

end rectangle_is_cyclic_l20_20280


namespace circle_line_intersection_symmetric_l20_20044

theorem circle_line_intersection_symmetric (m n p x y : ℝ)
    (h_intersects : ∃ x y, x = m * y - 1 ∧ x^2 + y^2 + m * x + n * y + p = 0)
    (h_symmetric : ∀ A B : ℝ × ℝ, A = (x, y) ∧ B = (y, x) → y = x) :
    p < -3 / 2 :=
by
  sorry

end circle_line_intersection_symmetric_l20_20044


namespace luke_total_points_l20_20105

theorem luke_total_points (rounds : ℕ) (points_per_round : ℕ) (total_points : ℕ) : 
  rounds = 177 → points_per_round = 46 → total_points = 8142 → rounds * points_per_round = total_points :=
by intros h1 h2 h3
   rw [h1, h2]
   exact h3


end luke_total_points_l20_20105


namespace one_div_thirteen_150th_digit_l20_20811

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20811


namespace digit_150th_of_fraction_l20_20719

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20719


namespace find_n_l20_20447

noncomputable def arithmetic_sequence (a : ℕ → ℕ) := 
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_n (a : ℕ → ℕ) (n d : ℕ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1)
  (h3 : a 2 + a 5 = 12)
  (h4 : a n = 25) : 
  n = 13 := 
sorry

end find_n_l20_20447


namespace total_distance_covered_l20_20210

theorem total_distance_covered :
  let t1 := 30 / 60 -- time in hours for first walking session
  let s1 := 3       -- speed in mph for first walking session
  let t2 := 20 / 60 -- time in hours for running session
  let s2 := 8       -- speed in mph for running session
  let t3 := 10 / 60 -- time in hours for second walking session
  let s3 := 2       -- speed in mph for second walking session
  let d1 := s1 * t1 -- distance for first walking session
  let d2 := s2 * t2 -- distance for running session
  let d3 := s3 * t3 -- distance for second walking session
  d1 + d2 + d3 = 4.5 :=
by
  sorry

end total_distance_covered_l20_20210


namespace complement_intersection_l20_20023

-- Definitions of sets and complements
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}
def C_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}
def C_U_B : Set ℕ := {x | x ∈ U ∧ x ∉ B}

-- The proof statement
theorem complement_intersection {U A B C_U_A C_U_B : Set ℕ} (h1 : U = {1, 2, 3, 4, 5}) (h2 : A = {1, 2, 3}) (h3 : B = {2, 5}) (h4 : C_U_A = {x | x ∈ U ∧ x ∉ A}) (h5 : C_U_B = {x | x ∈ U ∧ x ∉ B}) : 
  (C_U_A ∩ C_U_B) = {4} :=
by 
  sorry

end complement_intersection_l20_20023


namespace insurance_covers_80_percent_of_lenses_l20_20074

/--
James needs to get a new pair of glasses. 
His frames cost $200 and the lenses cost $500. 
Insurance will cover a certain percentage of the cost of lenses and he has a $50 off coupon for frames. 
Everything costs $250. 
Prove that the insurance covers 80% of the cost of the lenses.
-/

def frames_cost : ℕ := 200
def lenses_cost : ℕ := 500
def total_cost_after_discounts_and_insurance : ℕ := 250
def coupon : ℕ := 50

theorem insurance_covers_80_percent_of_lenses :
  ((frames_cost - coupon + lenses_cost - total_cost_after_discounts_and_insurance) * 100 / lenses_cost) = 80 := 
  sorry

end insurance_covers_80_percent_of_lenses_l20_20074


namespace one_div_thirteen_150th_digit_l20_20757

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20757


namespace transform_quadratic_equation_l20_20278

theorem transform_quadratic_equation :
  ∀ x : ℝ, (x^2 - 8 * x - 1 = 0) → ((x - 4)^2 = 17) :=
by
  intro x
  intro h
  sorry

end transform_quadratic_equation_l20_20278


namespace one_div_thirteen_150th_digit_l20_20786

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20786


namespace digit_150th_of_fraction_l20_20717

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20717


namespace digit_150_after_decimal_of_one_thirteenth_l20_20872

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20872


namespace candy_last_days_l20_20951

variable (candy_from_neighbors candy_from_sister candy_per_day : ℕ)

theorem candy_last_days
  (h_candy_from_neighbors : candy_from_neighbors = 66)
  (h_candy_from_sister : candy_from_sister = 15)
  (h_candy_per_day : candy_per_day = 9) :
  let total_candy := candy_from_neighbors + candy_from_sister  
  (total_candy / candy_per_day) = 9 := by
  sorry

end candy_last_days_l20_20951


namespace prime_sequence_divisibility_l20_20493

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def ceil (x : ℚ) : ℤ :=
  (x : ℝ).ceil.to_int

def sequence_a (p : ℕ) (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 1 then 2 else a (n - 1) + ceil (p * a (n - 1) / n)

theorem prime_sequence_divisibility
  (p : ℕ)
  (h1 : isPrime p)
  (h2 : isPrime (p + 2))
  (h3 : p > 3)
  (a : ℕ → ℕ)
  (a_def : ∀ n : ℕ, a n = sequence_a p a n) :
  ∀ n : ℕ, 3 ≤ n → n < p → n ∣ (p * a (n - 1) + 1) :=
by
  sorry

end prime_sequence_divisibility_l20_20493


namespace fraction_students_say_dislike_actually_like_l20_20285

theorem fraction_students_say_dislike_actually_like (total_students : ℕ) (like_dancing_fraction : ℚ) 
  (like_dancing_say_dislike_fraction : ℚ) (dislike_dancing_say_dislike_fraction : ℚ) : 
  (∃ frac : ℚ, frac = 40.7 / 100) :=
by
  let total_students := (200 : ℕ)
  let like_dancing_fraction := (70 / 100 : ℚ)
  let like_dancing_say_dislike_fraction := (25 / 100 : ℚ)
  let dislike_dancing_say_dislike_fraction := (85 / 100 : ℚ)
  
  let total_like_dancing := total_students * like_dancing_fraction
  let total_dislike_dancing :=  total_students * (1 - like_dancing_fraction)
  let like_dancing_say_dislike := total_like_dancing * like_dancing_say_dislike_fraction
  let dislike_dancing_say_dislike := total_dislike_dancing * dislike_dancing_say_dislike_fraction
  let total_say_dislike := like_dancing_say_dislike + dislike_dancing_say_dislike
  let fraction_say_dislike_actually_like := like_dancing_say_dislike / total_say_dislike
  
  existsi fraction_say_dislike_actually_like
  sorry

end fraction_students_say_dislike_actually_like_l20_20285


namespace digit_150_of_1_over_13_is_3_l20_20764

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20764


namespace infinitely_many_primes_4k1_l20_20196

theorem infinitely_many_primes_4k1 (h : ∀ m : ℕ, ∃ p : ℕ, p.prime ∧ p = 4 * k + 1 ∧ p ∣ (4 * m + 1)) : 
  ∃ inf : ℕ → ℕ, ∀ n : ℕ, (inf n).prime ∧ inf n = 4 * k + 1 := 
sorry

end infinitely_many_primes_4k1_l20_20196


namespace smallest_k_l20_20353

theorem smallest_k (k : ℕ) (numbers : set ℕ) (h₁ : ∀ x ∈ numbers, x ≤ 2016) (h₂ : numbers.card = k) :
  (∃ a b ∈ numbers, 672 < abs (a - b) ∧ abs (a - b) < 1344) ↔ k ≥ 674 := 
by
  sorry

end smallest_k_l20_20353


namespace find_m_monotonic_increase_interval_find_bc_sum_l20_20406
noncomputable def f (x : ℝ) : ℝ := (real.sqrt 3) * real.sin (2 * x) + 2 * real.cos x ^ 2 + 3

theorem find_m :
  ∃ m, (∀ x ∈ Icc (0 : ℝ) (real.pi / 2), f x ≤ 6) ∧ (∃ x ∈ Icc (0 : ℝ) (real.pi / 2), f x = 6) :=
sorry

theorem monotonic_increase_interval (k : ℤ) :
  ∀ x ∈ Icc (-real.pi / 3 + k * real.pi) (real.pi / 6 + k * real.pi),
  ∃ m, ∀ y ∈ Icc (-real.pi / 3 + k * real.pi) (x + k * real.pi),
    f y ≤ f x :=
sorry

noncomputable def f_A : ℝ := 2 * real.sin (2 * (real.pi / 3) + real.pi / 6) + 4

theorem find_bc_sum (a : ℝ) (area : ℝ) (A : ℝ) :
  (a = 4 ∧ area = real.sqrt 3 ∧ f_A = 5 ∧ A = real.pi / 3) →
  ∃ b c : ℝ, b + c = 2 * real.sqrt 7 :=
sorry

end find_m_monotonic_increase_interval_find_bc_sum_l20_20406


namespace no_solutions_l20_20494

theorem no_solutions (x y : ℤ) (h : 8 * x + 3 * y^2 = 5) : False :=
by
  sorry

end no_solutions_l20_20494


namespace john_sixth_quiz_score_l20_20077

noncomputable def sixth_quiz_score_needed : ℤ :=
  let scores := [86, 91, 88, 84, 97]
  let desired_average := 95
  let number_of_quizzes := 6
  let total_score_needed := number_of_quizzes * desired_average
  let total_score_so_far := scores.sum
  total_score_needed - total_score_so_far

theorem john_sixth_quiz_score :
  sixth_quiz_score_needed = 124 := 
by
  sorry

end john_sixth_quiz_score_l20_20077


namespace curve_is_hyperbola_l20_20313

open Real

def curve (t : ℝ) (ht : t ≠ 0) : ℝ × ℝ :=
  ((t^2 + 1) / t, (t^2 - 1) / t)

theorem curve_is_hyperbola : 
  ∀ t : ℝ, t ≠ 0 → ∃ (a b : ℝ), ∀ (x y : ℝ), 
  (x, y) = ((t^2 + 1) / t, (t^2 - 1) / t) → 
  a * x^2 - b * y^2 = c :=
sorry

end curve_is_hyperbola_l20_20313


namespace qs_length_l20_20190

theorem qs_length
  (PQR : Triangle)
  (PQ QR PR : ℝ)
  (h1 : PQ = 7)
  (h2 : QR = 8)
  (h3 : PR = 9)
  (bugs_meet_half_perimeter : PQ + QR + PR = 24)
  (bugs_meet_distance : PQ + qs = 12) :
  qs = 5 :=
by
  sorry

end qs_length_l20_20190


namespace henry_money_after_transactions_l20_20027

theorem henry_money_after_transactions :
  ∃ (initial birthday_spending spent_on_game final_money : ℤ),
    initial = 11 ∧
    birthday_spending = 18 ∧
    spent_on_game = 10 ∧
    final_money = initial + birthday_spending - spent_on_game ∧
    final_money = 19 :=
by
  use 11, 18, 10, 19
  simp
  sorry

end henry_money_after_transactions_l20_20027


namespace decimal_150th_digit_l20_20888

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20888


namespace Lisa_weight_l20_20281

theorem Lisa_weight : ∃ l a : ℝ, a + l = 240 ∧ l - a = l / 3 ∧ l = 144 :=
by
  sorry

end Lisa_weight_l20_20281


namespace largest_negative_a_l20_20918

theorem largest_negative_a {a : ℝ} :
  (∀ x : ℝ, x ∈ Set.Ioo (-3 * Real.pi) (-5 * Real.pi / 2) →
    ((Real.cbrt (Real.cos x) - Real.cbrt (Real.sin x)) / (Real.cbrt (Real.tan x) - Real.cbrt (Real.tan x)) > a)) →
  a = -0.45 :=
sorry

end largest_negative_a_l20_20918


namespace excursion_min_parents_l20_20594

theorem excursion_min_parents 
  (students : ℕ) 
  (car_capacity : ℕ)
  (h_students : students = 30)
  (h_car_capacity : car_capacity = 5) 
  : ∃ (parents_needed : ℕ), parents_needed = 8 := 
by
  sorry -- proof goes here

end excursion_min_parents_l20_20594


namespace polynomial_q_form_l20_20545

noncomputable def q (x : ℝ) : ℝ := x^3 - (79/16) * x^2 - (17/8) * x + 81

theorem polynomial_q_form :
  (∀ x, q x = x^3 - (79/16) * x^2 - (17/8) * x + 81) ∧
  q(5 - 3 * Complex.i) = 0 ∧
  q(5 + 3 * Complex.i) = 0 ∧
  q 0 = 81 :=
by
  sorry

end polynomial_q_form_l20_20545


namespace max_value_of_f_f_lt_x3_minus_2x2_l20_20394

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^2 + Real.log x + b

theorem max_value_of_f (a b : ℝ) (h_a : a = -1) (h_b : b = -1 / 4) :
  f a b (Real.sqrt 2 / 2) = - (3 + 2 * Real.log 2) / 4 := by
  sorry

theorem f_lt_x3_minus_2x2 (a b : ℝ) (h_a : a = -1) (h_b : b = -1 / 4) (x : ℝ) (hx : 0 < x) :
  f a b x < x^3 - 2 * x^2 := by
  sorry

end max_value_of_f_f_lt_x3_minus_2x2_l20_20394


namespace one_div_thirteen_150th_digit_l20_20752

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20752


namespace cosine_shift_equivalence_l20_20601

theorem cosine_shift_equivalence : 
  ∀ x, cos (x / 2 - π / 3) = cos ((x - 2 * π / 3) / 2) :=
by
  intro x
  sorry

end cosine_shift_equivalence_l20_20601


namespace john_drive_time_l20_20465

theorem john_drive_time
  (t : ℝ)
  (h1 : 60 * t + 90 * (15 / 4 - t) = 300)
  (h2 : 1 / 4 = 15 / 60)
  (h3 : 4 = 15 / 4 + t + 1 / 4)
  :
  t = 1.25 :=
by
  -- This introduces the hypothesis and begins the Lean proof.
  sorry

end john_drive_time_l20_20465


namespace count_valid_pairs_l20_20268

open Nat

-- Define the conditions
def room_conditions (p q : ℕ) : Prop :=
  q > p ∧
  (∃ (p' q' : ℕ), p = p' + 6 ∧ q = q' + 6 ∧ p' * q' = 48)

-- State the theorem to prove the number of valid pairs (p, q)
theorem count_valid_pairs : 
  (∃ l : List (ℕ × ℕ), 
    (∀ pq ∈ l, room_conditions pq.fst pq.snd) ∧ 
    l.length = 5) := 
sorry

end count_valid_pairs_l20_20268


namespace shooter_with_more_fluctuation_l20_20580

noncomputable def variance (scores : List ℕ) (mean : ℕ) : ℚ :=
  (List.sum (List.map (λ x => (x - mean) * (x - mean)) scores) : ℚ) / scores.length

theorem shooter_with_more_fluctuation :
  let scores_A := [7, 9, 8, 6, 10]
  let scores_B := [7, 8, 9, 8, 8]
  let mean := 8
  variance scores_A mean > variance scores_B mean :=
by
  sorry

end shooter_with_more_fluctuation_l20_20580


namespace equal_segments_on_AC_l20_20603

-- Define the points and circles geometry
variables {A B C X Y D E : Point}
variables {circle1 circle2 : Circle}
variable hABC : Angle ABC -- Angle at B between points A and C
variable hCircle1Inscribed : Inscribed circle1 hABC -- circle1 inscribed in angle ABC
variable hCircle2Inscribed : Inscribed circle2 hABC -- circle2 inscribed in angle ABC
variable hTouch1 : Tangent circle1 A -- circle1 touches AB at A
variable hTouch2 : Tangent circle2 C -- circle2 touches BC at C
variable hIntersect1 : IntersectionPoints circle1 lineAC X -- circle1 and line AC intersect at X
variable hIntersect2 : IntersectionPoints circle2 lineAC Y -- circle2 and line AC intersect at Y
variable hTangency1 : TangentPoint circle1 D AB -- Tangency point D on AB for circle1
variable hTangency2 : TangentPoint circle2 E BC -- Tangency point E on BC for circle2

-- Define distances
variable hDistance1 : Distance A D = Distance C E -- Distances from points of tangency are equal

-- Prove the intersecting segments on AC are equal
theorem equal_segments_on_AC : Distance C X = Distance A Y :=
by sorry

end equal_segments_on_AC_l20_20603


namespace find_c_for_degree_3_l20_20300

noncomputable def f : Polynomial ℚ := 2 - 15 * Polynomial.X + 4 * Polynomial.X^2 - 3 * Polynomial.X^3 + 6 * Polynomial.X^4
noncomputable def g : Polynomial ℚ := 4 - 3 * Polynomial.X + 1 * Polynomial.X^2 - 7 * Polynomial.X^3 + 10 * Polynomial.X^4

theorem find_c_for_degree_3 :
  ∃ (c : ℚ), Polynomial.degree (f + c • g) = 3 :=
sorry

end find_c_for_degree_3_l20_20300


namespace decimal_150th_digit_l20_20656

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20656


namespace theta_range_l20_20383

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem theta_range (k : ℤ) (θ : ℝ) : 
  (2 * ↑k * π - 5 * π / 6 < θ ∧ θ < 2 * ↑k * π - π / 6) →
  (f (1 / (Real.sin θ)) + f (Real.cos (2 * θ)) < f π - f (1 / π)) :=
by
  intros h
  sorry

end theta_range_l20_20383


namespace volume_of_earth_dug_out_l20_20962
-- Import the necessary library

-- Define the conditions
def diameter : ℝ := 4
def depth : ℝ := 24
def radius : ℝ := diameter / 2
def pi : ℝ := Real.pi

-- Define the question: the volume of the cylindrical well
def volume_of_well : ℝ := pi * radius * radius * depth

-- The theorem we need to prove, stating that the volume is 96π
theorem volume_of_earth_dug_out :
  volume_of_well = 96 * pi :=
by
  -- Proof goes here
  sorry

end volume_of_earth_dug_out_l20_20962


namespace ratio_RS_ST_l20_20987

noncomputable def area_triangle (base height : ℕ) : ℕ := (base * height) / 2

noncomputable def area_above_RS (total_area : ℕ) : ℕ := (2 * total_area) / 3

noncomputable def area_below_RS (total_area : ℕ) : ℕ := total_area / 3

theorem ratio_RS_ST :
  let base := 6,
      height := 3,
      unit_squares := 12,
      triangle_area := area_triangle base height,
      total_area := unit_squares + triangle_area,
      area_below := area_below_RS total_area,
      area_above := area_above_RS total_area,
      RS := 2,
      ST := base - RS in
  area_below + area_above = total_area ∧ area_above = 2 * area_below ∧ (RS : ℚ) / (ST : ℚ) = 1 / 2 :=
by {
  sorry
}

end ratio_RS_ST_l20_20987


namespace right_triangle_properties_l20_20191

theorem right_triangle_properties (a b : ℝ) (h : a = 9) (h' : b = 12) :
    let hypotenuse := real.sqrt (a^2 + b^2)
    hypotenuse = 15 ∧ (1/2 * a * b) = 54 := by
  sorry

end right_triangle_properties_l20_20191


namespace one_thirteen_150th_digit_l20_20902

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20902


namespace smallest_k_exists_l20_20347

theorem smallest_k_exists (s : Finset ℕ) :
  (∀ a b ∈ s, a ≠ b → (672 < |a - b| ∧ |a - b| < 1344)) →
  (∀ k, k < 674 → ∃ s : Finset ℕ, s.card = k ∧ (∀ a b ∈ s, a ≠ b → ¬ (672 < |a - b| ∧ |a - b| < 1344))) → False :=
begin
  sorry
end

end smallest_k_exists_l20_20347


namespace one_div_thirteen_150th_digit_l20_20751

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20751


namespace one_div_thirteen_150th_digit_l20_20780

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20780


namespace one_div_thirteen_150th_digit_l20_20753

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20753


namespace digit_150_of_decimal_1_div_13_l20_20669

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20669


namespace output_value_l20_20528

theorem output_value (a b : ℕ) (h1 : a = 1) (h2 : b = 2) : let a := a + b in a = 3 :=
by
  sorry

end output_value_l20_20528


namespace shaded_area_eq_l20_20238

noncomputable def radius_small : ℝ := 2
noncomputable def radius_large : ℝ := 3
noncomputable def area_shaded : ℝ := (5.65 / 3) * Real.pi - 2.9724 * Real.sqrt 5

theorem shaded_area_eq : 
  (∃ A B C : ℝ × ℝ, 
    ∀ (x : ℝ × ℝ), x ∈ { p | Real.norm (p.1 - C.1, p.2 - C.2) = radius_small } → 
    (Real.norm (x.1 - A.1, x.2 - A.2) = radius_large ∧ Real.norm (x.1 - B.1, x.2 - B.2) = radius_large) ∧ 
    Real.norm (A.1 - B.1, A.2 - B.2) = 2 * radius_small) → 
  ∃ R : ℝ, R = area_shaded := 
by {
  sorry
}

end shaded_area_eq_l20_20238


namespace decimal_150th_digit_l20_20885

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20885


namespace digit_150_of_one_thirteenth_l20_20836

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20836


namespace one_thirteenth_150th_digit_l20_20629

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20629


namespace range_of_m_l20_20409

theorem range_of_m (m : ℝ) :
  (m ≤ 0 ∨ m ≥ 4) ↔ (∀ θ : ℝ, m^2 + (cos θ ^ 2 - 5) * m + 4 * sin θ ^ 2 ≥ 0) :=
by
  sorry

end range_of_m_l20_20409


namespace max_power_at_v0_div_3_l20_20567

variable (C S ρ v₀ : ℝ)

def force_on_sail (v : ℝ) : ℝ :=
  (C * S * ρ * (v₀ - v) ^ 2) / 2

def power (v : ℝ) : ℝ :=
  (force_on_sail C S ρ v₀ v) * v

theorem max_power_at_v0_div_3 : ∃ v : ℝ, power C S ρ v₀ v = (2 * C * S * ρ * v₀ ^ 3) / 27 ∧ v = v₀ / 3 :=
by {
  sorry
}

end max_power_at_v0_div_3_l20_20567


namespace find_integer_pairs_l20_20331

theorem find_integer_pairs :
  (∃ xs ys: List ℤ, 
    ∀ y x, 
      (y ∈ ys) ↔ 
      (x ∈ xs) ∧ 
      y ≥ 2^x + 3 * 2^34 ∧
      y < 76 + 2 * (2^32 - 1) * x) 
  ∧ xs.length = 31 := 
sorry

end find_integer_pairs_l20_20331


namespace perimeter_of_shaded_region_l20_20451

theorem perimeter_of_shaded_region (O P Q : Point) (h1 : P ≠ Q) (h2 : dist O P = 5) (h3 : dist O Q = 5) (h4 : ∠ Q O P = real.pi / 2) : 
  perimeter (shaded_region P Q O) = 10 + 15 * real.pi / 2 := by
  sorry

end perimeter_of_shaded_region_l20_20451


namespace smallest_k_l20_20350

theorem smallest_k (k : ℕ) (numbers : set ℕ) (h₁ : ∀ x ∈ numbers, x ≤ 2016) (h₂ : numbers.card = k) :
  (∃ a b ∈ numbers, 672 < abs (a - b) ∧ abs (a - b) < 1344) ↔ k ≥ 674 := 
by
  sorry

end smallest_k_l20_20350


namespace petya_friends_l20_20518

theorem petya_friends : ∀ (n : ℕ), n = 28 → (∀ i j : fin (n+1), i ≠ j → i.val ≠ j.val) → ∃ k : ℕ, k = 14 := 
by
    intros n n_is_28 unique_friends
    have h : n = 28 := n_is_28
    sorry

end petya_friends_l20_20518


namespace evaluate_expression_l20_20318

theorem evaluate_expression :
  sqrt ((4 / 25) + (9 / 49)) = sqrt (421 / 1225) := by
    sorry

end evaluate_expression_l20_20318


namespace one_div_thirteen_150th_digit_l20_20815

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20815


namespace team_size_eight_l20_20274

variable {S : ℝ} (team_size : ℕ)

-- Define the conditions
def meadows (a b : ℝ) : Prop :=
  a = S ∧ b = 2 * S

def half_day_work (mowers : ℕ) (remaining_large_meadow : ℝ) : Prop :=
  remaining_large_meadow = 2 * S / 3

def split_work (mowers_half : ℕ) (remaining_large_meadow : ℝ) (cut_small_meadow : ℝ) : Prop :=
  (mowers_half *  remaining_large_meadow) = S / 3 ∧ (mowers_half * cut_small_meadow) = S / 2

def remaining_work_finished_by_one_mower (remaining_small_meadow : ℝ) : Prop :=
  remaining_small_meadow = S / 2

-- Prove team size is 8 given the conditions
theorem team_size_eight : ∃ N, 
  (meadows S S) ∧
  (half_day_work team_size _) ∧
  (split_work (team_size / 2) _ S) ∧
  (remaining_work_finished_by_one_mower S / 2) ∧
  N = 8 :=
begin
  sorry
end

end team_size_eight_l20_20274


namespace sailboat_speed_max_power_l20_20565

-- Define the parameters
variables (C S ρ v0 : ℝ)

-- Define the force function
def force (v : ℝ) : ℝ :=
  (C * S * ρ * (v0 - v) ^ 2) / 2

-- Define the power function
def power (v : ℝ) : ℝ :=
  (force C S ρ v0 v) * v

-- Define the statement to be proven
theorem sailboat_speed_max_power : ∃ v : ℝ, (power C S ρ v0 v = Term.max (power C S ρ v0)) ∧ v = v0 / 3 :=
by
  sorry

end sailboat_speed_max_power_l20_20565


namespace find_a_in_csc_equation_l20_20990

theorem find_a_in_csc_equation (a : ℝ) (x : ℝ) : 
  (∀ x, y = 3 * csc (2 * x - π)) → a = 3 :=
by
  sorry

end find_a_in_csc_equation_l20_20990


namespace digit_150_of_1_div_13_l20_20800

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20800


namespace students_on_couch_per_room_l20_20976

def total_students : ℕ := 30
def total_rooms : ℕ := 6
def students_per_bed : ℕ := 2
def beds_per_room : ℕ := 2
def students_in_beds_per_room : ℕ := beds_per_room * students_per_bed

theorem students_on_couch_per_room :
  (total_students / total_rooms) - students_in_beds_per_room = 1 := by
  sorry

end students_on_couch_per_room_l20_20976


namespace average_height_l20_20547

def heights : List ℕ := [145, 142, 138, 136, 143, 146, 138, 144, 137, 141]

theorem average_height :
  (heights.sum : ℕ) / heights.length = 141 := by
  sorry

end average_height_l20_20547


namespace people_in_company_l20_20590

theorem people_in_company (M S Z None P : ℕ) (hM : M = 16) (hS : S = 18) (hZ : Z = 11)
  (hNone : None ≤ 26) (hSZ : ∀ SZ, SZ = 0): P = M + S + Z + None := by
  have h1 : P = 16 + 18 + 11 + 26 := by
    rw [hM, hS, hZ, hNone] 
    simp only [hSZ]
  exact h1

end people_in_company_l20_20590


namespace largest_binomial_coefficient_l20_20481

theorem largest_binomial_coefficient (a : Fin 8 → ℤ) (x : ℤ) :
  (∀ n, a n = (-1) ^ n * Nat.choose 7 n) →
  max (Finset.image a Finset.univ) = a 4 := 
by
  intros h
  -- Proof code to demonstrate the maximum value will be here
  sorry

end largest_binomial_coefficient_l20_20481


namespace one_thirteenth_150th_digit_l20_20625

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20625


namespace simplest_sqrt_is_5_l20_20205

def is_simplified_sqrt (n : ℕ) (x : ℝ) : Prop :=
  sqrt n = x

def is_prime_or_non_perfect_square (x : ℝ) : Prop :=
  x = real.sqrt 5

theorem simplest_sqrt_is_5 :
  (is_simplified_sqrt 5 (real.sqrt 5)) ∧ ¬ (is_simplified_sqrt 9 (real.sqrt 9) ∧ is_prime_or_non_perfect_square 9) ∧ 
  ¬ (is_simplified_sqrt 18 (real.sqrt 18) ∧ is_prime_or_non_perfect_square 18) ∧ 
  ¬ (is_simplified_sqrt (1 / 2) (real.sqrt (1 / 2)) ∧ is_prime_or_non_perfect_square (1 / 2)) :=
by sorry

end simplest_sqrt_is_5_l20_20205


namespace polyhedron_with_equal_square_faces_not_necessarily_cube_l20_20460

-- Define what it means to be a polyhedron with all faces equal squares
structure Polyhedron (P : Type) :=
  (faces : list (set P))
  (equality_of_faces : ∀ f ∈ faces, is_square f)
  (equal_faces : ∀ f₁ f₂ ∈ faces, f₁ = f₂)

-- Define what it means to be a cube
def is_cube {P : Type} (poly : Polyhedron P) : Prop :=
  ∃ (vertices : list P), vertices.length = 8 ∧
  ∃ (edges : list (P × P)), edges.length = 12 ∧
  ∃ (faces : list (set P)), faces.length = 6

-- Define the main theorem
theorem polyhedron_with_equal_square_faces_not_necessarily_cube :
  ∃ (P : Type) (poly : Polyhedron P), ¬ is_cube poly :=
sorry

end polyhedron_with_equal_square_faces_not_necessarily_cube_l20_20460


namespace decimal_150th_digit_l20_20652

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20652


namespace possible_triple_roots_l20_20974

def integer_polynomial (P : ℤ[X]) : Prop :=
∃ b₄ b₃ b₂ b₁ : ℤ, P = X^5 + C b₄ * X^4 + C b₃ * X^3 + C b₂ * X^2 + C b₁ * X + 24

def triple_root_condition (P : ℤ[X]) (r : ℤ) : Prop :=
(X - C r) ^ 3 ∣ P

theorem possible_triple_roots :
  ∀ (P : ℤ[X]), integer_polynomial P → ∀ (r : ℤ), triple_root_condition P r → r ∈ [-2, -1, 1, 2] :=
by
  intros P h_poly r h_triple_root
  sorry

end possible_triple_roots_l20_20974


namespace even_multiples_of_4_product_zero_count_l20_20339
open Complex Real -- Open the relevant namespaces for complex numbers and real numbers.
  
theorem even_multiples_of_4_product_zero_count :
  (∃ n ∈ Icc 1 2020, (∏ k in finset.range n, ((1 + exp (2 * π * I * k / n))^n + 1) = 0)) ↔ (252 : ℕ) :=
sorry

end even_multiples_of_4_product_zero_count_l20_20339


namespace digit_150th_of_fraction_l20_20720

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20720


namespace digit_150_of_1_div_13_l20_20795

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20795


namespace one_div_thirteen_150th_digit_l20_20782

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20782


namespace stones_max_value_50_l20_20288

-- Define the problem conditions in Lean
def value_of_stones (x y z : ℕ) : ℕ := 14 * x + 11 * y + 2 * z

def weight_of_stones (x y z : ℕ) : ℕ := 5 * x + 4 * y + z

def max_value_stones {x y z : ℕ} (h_w : weight_of_stones x y z ≤ 18) (h_x : x ≥ 0) (h_y : y ≥ 0) (h_z : z ≥ 0) : Prop :=
  value_of_stones x y z ≤ 50

theorem stones_max_value_50 : ∃ (x y z : ℕ), weight_of_stones x y z ≤ 18 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ value_of_stones x y z = 50 :=
by
  sorry

end stones_max_value_50_l20_20288


namespace unique_root_and_three_distinct_roots_l20_20155

noncomputable def P (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^2 + b * x + c

noncomputable def solve_equation : Set ℝ :=
  { x | P (P (P x b c) b c) b c = 0 }

theorem unique_root_and_three_distinct_roots (b c : ℝ)
  (h1 : b^2 = 4 * c)
  (h2 : ∀ x1 x2 x3 : ℝ, x1 ≠ x2 → x1 ≠ x3 → x2 ≠ x3 →
    x1 ∈ solve_equation ∧ x2 ∈ solve_equation ∧ x3 ∈ solve_equation) :
  (1 : ℝ) ∈ solve_equation ∧ (1 + Real.sqrt 2) ∈ solve_equation ∧ (1 - Real.sqrt 2) ∈ solve_equation :=
begin
  sorry,
end

end unique_root_and_three_distinct_roots_l20_20155


namespace one_div_thirteen_150th_digit_l20_20812

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20812


namespace digit_150_of_1_over_13_is_3_l20_20766

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20766


namespace decimal_150th_digit_l20_20640

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20640


namespace decimal_150th_digit_l20_20643

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20643


namespace game_show_prizes_l20_20250

theorem game_show_prizes :
  let digits := [1, 1, 2, 2, 3, 3, 3, 3]
  let permutations := Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 2)
  let partitions := Nat.choose 7 3
  permutations * partitions = 14700 :=
by
  let digits := [1, 1, 2, 2, 3, 3, 3, 3]
  let permutations := Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 2)
  let partitions := Nat.choose 7 3
  exact sorry

end game_show_prizes_l20_20250


namespace seven_in_M_l20_20104

-- Define the universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- Define the set M complement with respect to U
def compl_U_M : Set ℕ := {1, 3, 5}

-- Define the set M
def M : Set ℕ := U \ compl_U_M

-- Prove that 7 is an element of M
theorem seven_in_M : 7 ∈ M :=
by {
  sorry
}

end seven_in_M_l20_20104


namespace advertisement_probability_l20_20230

theorem advertisement_probability
  (ads_time_hour : ℕ)
  (total_time_hour : ℕ)
  (h1 : ads_time_hour = 20)
  (h2 : total_time_hour = 60) :
  ads_time_hour / total_time_hour = 1 / 3 :=
by
  sorry

end advertisement_probability_l20_20230


namespace general_formula_secondary_sum_formula_l20_20086

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a d n : ℕ) : ℕ := n * a + n * (n - 1) * d / 2

-- Problem Conditions
variable (a d : ℕ)
axiom a3a5 : arithmetic_sequence a d 3 * arithmetic_sequence a d 5 = 3 * arithmetic_sequence a d 7
axiom S3_eq_9 : sum_first_n_terms a d 3 = 9

-- General formula for the sequence
theorem general_formula : ∀ n, arithmetic_sequence a d 1 = 1 ∧ d = 2 → arithmetic_sequence 1 2 n = 2 * n - 1 := 
by {
  sorry
}

-- Define the secondary sequence and its sum
def secondary_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  1 / (arithmetic_sequence a d n * arithmetic_sequence a d (n + 1))

def sum_secondary_sequence_first_n (a d n : ℕ) : ℕ :=
  ∑ k in range n, secondary_sequence a d k

-- Calculate T_n
theorem secondary_sum_formula :
  ∀ n,
    (arithmetic_sequence a d 1 = 1 ∧ d = 2) →
    sum_secondary_sequence_first_n 1 2 n = n / (2 * n + 1) :=
by {
  sorry
}

end general_formula_secondary_sum_formula_l20_20086


namespace cos_squared_plus_cos_fourth_l20_20425

theorem cos_squared_plus_cos_fourth (α : ℝ) (h : sin α + (sin α) ^ 2 = 1) :
  cos α ^ 2 + (cos α) ^ 4 = 1 := 
sorry

end cos_squared_plus_cos_fourth_l20_20425


namespace solve_cubic_equation_l20_20137

theorem solve_cubic_equation : 
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^3 - y^3 = 999 ∧ (x, y) = (12, 9) ∨ (x, y) = (10, 1) := 
  by
  sorry

end solve_cubic_equation_l20_20137


namespace value_of_square_reciprocal_l20_20423

theorem value_of_square_reciprocal (x : ℝ) (h : 18 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = Real.sqrt 20 := by
  sorry

end value_of_square_reciprocal_l20_20423


namespace count_even_three_digit_numbers_less_than_600_l20_20618

-- Define the digits
def digits : List ℕ := [1, 2, 3, 4, 5, 6]

-- Condition: the number must be less than 600, i.e., hundreds digit in {1, 2, 3, 4, 5}
def valid_hundreds (d : ℕ) : Prop := d ∈ [1, 2, 3, 4, 5]

-- Condition: the units (ones) digit must be even
def valid_units (d : ℕ) : Prop := d ∈ [2, 4, 6]

-- Problem: total number of valid three-digit numbers
def total_valid_numbers : ℕ :=
  List.product (List.product [1, 2, 3, 4, 5] digits) [2, 4, 6] |>.length

-- Proof statement
theorem count_even_three_digit_numbers_less_than_600 :
  total_valid_numbers = 90 := by
  sorry

end count_even_three_digit_numbers_less_than_600_l20_20618


namespace infinite_primes_dividing_a_pow_n_plus_b_pow_n_minus_c_pow_n_l20_20477

theorem infinite_primes_dividing_a_pow_n_plus_b_pow_n_minus_c_pow_n
  (a b c : ℕ)
  (h1 : a ≠ c)
  (h2 : b ≠ c) :
  ∃ᶠ p in filter.at_top Prime,
    ∃ n : ℕ, p ∣ a^n + b^n - c^n :=
by sorry

end infinite_primes_dividing_a_pow_n_plus_b_pow_n_minus_c_pow_n_l20_20477


namespace right_triangle_AB_length_l20_20448

theorem right_triangle_AB_length
  (A B C : Type)
  (angle_B_90 : ∠B = 90)
  (BC : ℝ) (hBC : BC = 1)
  (AC : ℝ) (hAC : AC = 2)
  (AB : ℝ)
  (pythagorean_theorem : AB^2 + BC^2 = AC^2) :
  AB = Real.sqrt 3 :=
by
  sorry

end right_triangle_AB_length_l20_20448


namespace quad_factor_value_l20_20559

theorem quad_factor_value (c d : ℕ) (h1 : c + d = 14) (h2 : c * d = 40) (h3 : c > d) : 4 * d - c = 6 :=
sorry

end quad_factor_value_l20_20559


namespace dealer_sold_135_BMWs_l20_20248

theorem dealer_sold_135_BMWs
  (total_cars : ℕ)
  (perc_audi perc_toyota perc_acura : ℝ)
  (h1 : total_cars = 300)
  (h2 : perc_audi = 0.12)
  (h3 : perc_toyota = 0.25)
  (h4 : perc_acura = 0.18) :
  let remaining_perc := 1 - (perc_audi + perc_toyota + perc_acura)
  let num_BMWs := total_cars * remaining_perc
  in num_BMWs = 135 := by
  sorry

end dealer_sold_135_BMWs_l20_20248


namespace digit_150_of_1_over_13_is_3_l20_20775

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20775


namespace find_g_five_l20_20163

theorem find_g_five 
  (g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
  (h2 : ∀ x : ℝ, g x ≠ 0)
  (h3 : g 0 = 1) : g 5 = Real.exp 5 :=
sorry

end find_g_five_l20_20163


namespace total_interest_percentage_l20_20275

theorem total_interest_percentage (inv_total : ℝ) (rate1 rate2 : ℝ) (inv2 : ℝ)
  (h_inv_total : inv_total = 100000)
  (h_rate1 : rate1 = 0.09)
  (h_rate2 : rate2 = 0.11)
  (h_inv2 : inv2 = 24999.999999999996) :
  (rate1 * (inv_total - inv2) + rate2 * inv2) / inv_total * 100 = 9.5 := 
sorry

end total_interest_percentage_l20_20275


namespace digit_150_of_one_thirteenth_l20_20834

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20834


namespace one_div_thirteen_150th_digit_l20_20787

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20787


namespace one_thirteen_150th_digit_l20_20895

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20895


namespace total_revenue_calculation_l20_20986

open Real

def truck_data (weight: ℝ) (damage: ℝ) := (weight, damage)

def discount_scheme := 
    ((1, 9), 75) ::
        ((10, 19), 72) ::
        ((20, 50), 68) ::
        []

def price_per_bag (truck_bags: ℝ) : ℝ :=
  if truck_bags < 10 then 75
  else if truck_bags < 20 then 72
  else 68

def undamaged_weight (weight: ℝ) (damage_percentage: ℝ) : ℝ :=
  weight * (1 - damage_percentage / 100)

def number_of_bags (weight: ℝ) : ℕ :=
  Nat.floor (weight / 50)

def revenue (truck: (ℝ × ℝ)) : ℝ :=
  let bags := number_of_bags (undamaged_weight (truck.1) (truck.2)) in
  bags * price_per_bag (bags)

theorem total_revenue_calculation : 
  revenue (truck_data 3000 3) + 
  revenue (truck_data 4000 4.5) + 
  revenue (truck_data 2500 2.5) + 
  revenue (truck_data 2500 5) = 16024 := by
  sorry

end total_revenue_calculation_l20_20986


namespace digit_150_of_one_thirteenth_l20_20829

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20829


namespace one_div_thirteen_150th_digit_l20_20745

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20745


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20726

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20726


namespace carol_seq_last_three_digits_l20_20993

/-- Carol starts to make a list, in increasing order, of the positive integers that have 
    a first digit of 2. She writes 2, 20, 21, 22, ...
    Prove that the three-digit number formed by the 1198th, 1199th, 
    and 1200th digits she wrote is 218. -/
theorem carol_seq_last_three_digits : 
  (digits_1198th_1199th_1200th = 218) :=
by
  sorry

end carol_seq_last_three_digits_l20_20993


namespace Max_wins_count_l20_20291

-- Definitions for Chloe and Max's wins and their ratio
def Chloe_wins := 24
def Max_wins (Y : ℕ) := 8 * Y = 3 * Chloe_wins

-- The theorem to be proven
theorem Max_wins_count : ∃ Y : ℕ, Max_wins Y ∧ Y = 9 :=
by
  existsi 9
  simp [Max_wins, Chloe_wins]
  sorry

end Max_wins_count_l20_20291


namespace smallest_x_remainder_l20_20926

theorem smallest_x_remainder : ∃ x : ℕ, x > 0 ∧ 
    x % 6 = 5 ∧
    x % 7 = 6 ∧
    x % 8 = 7 ∧
    x = 167 :=
by
  sorry

end smallest_x_remainder_l20_20926


namespace circle_area_equals_100_l20_20206

noncomputable def side_length_of_square : ℝ := 25

def perimeter_of_square (s : ℝ) : ℝ := 4 * s

def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

theorem circle_area_equals_100 :
  (∃ r : ℝ, area_of_circle r = perimeter_of_square side_length_of_square) -> area_of_circle (Real.sqrt (100 / Real.pi)) = 100 :=
by
  intro h
  cases h with r hr
  have r_eq_sqrt := Real.sqrt_eq r (100 / Real.pi) (by linarith) sorry
  rw [r_eq_sqrt] at hr
  exact sorry

end circle_area_equals_100_l20_20206


namespace vector_sum_magnitude_zero_l20_20392

variables (e1 e2 e3 : EuclideanSpace ℝ (Fin 3))

-- Conditions: e1, e2, and e3 are unit vectors
axiom unit_vector_e1 : ∥e1∥ = 1
axiom unit_vector_e2 : ∥e2∥ = 1
axiom unit_vector_e3 : ∥e3∥ = 1

-- Condition: The angle between any two vectors is 120 degrees (2π/3 radians)
axiom angle_e1_e2 : inner e1 e2 = -1/2
axiom angle_e1_e3 : inner e1 e3 = -1/2
axiom angle_e2_e3 : inner e2 e3 = -1/2

-- Theorem: The magnitude of e1 + e2 + e3 is 0
theorem vector_sum_magnitude_zero : ∥e1 + e2 + e3∥ = 0 := by
  sorry

end vector_sum_magnitude_zero_l20_20392


namespace alternating_binomial_sum_l20_20336

open BigOperators Finset

theorem alternating_binomial_sum :
  ∑ k in range 34, (-1 : ℤ)^k * (Nat.choose 99 (3 * k)) = -1 := by
  sorry

end alternating_binomial_sum_l20_20336


namespace hexagon_area_l20_20542

theorem hexagon_area (a b c d : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_sum_of_legs : a + b = d) :
  let hex_area := a^2 + d^2 in
  hex_area = a^2 + d^2 :=
by sorry

end hexagon_area_l20_20542


namespace one_thirteen_150th_digit_l20_20908

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20908


namespace largest_k_divisibility_by_2_l20_20085

theorem largest_k_divisibility_by_2 (Q : ℕ) (hQ : Q = List.prod (List.map (λ n, 2 * n) (List.range 50))) :
    ∃ k, k = 97 ∧ 2^k ∣ Q :=
by
  use 97
  -- Details of the proof are filled here
  sorry

end largest_k_divisibility_by_2_l20_20085


namespace digit_150_of_decimal_1_div_13_l20_20671

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20671


namespace probability_even_upper_face_of_six_sided_cube_l20_20928

theorem probability_even_upper_face_of_six_sided_cube : 
  let outcomes := {1, 2, 3, 4, 5, 6} in
  let even_outcomes := {2, 4, 6} in
  (even_outcomes.card : ℚ) / (outcomes.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_even_upper_face_of_six_sided_cube_l20_20928


namespace survivor_probability_same_tribe_l20_20172

theorem survivor_probability_same_tribe :
  let total_people := 18
  let people_in_tribe := 6
  let total_combinations :=
    @fintype.card (finset (fin total_people)) (finset.fintype (@finset.image (fin 3) (finset.univ : finset (fin total_people)) id))
  let favorable_combinations := 3 * @fintype.card (finset (fin people_in_tribe)) (finset.fintype (@finset.image (fin 3) (finset.univ : finset (fin people_in_tribe)) id))
  (favorable_combinations : ℚ) / (total_combinations : ℚ) = 5 / 68 := 
begin
  sorry
end

end survivor_probability_same_tribe_l20_20172


namespace greatest_lines_of_symmetry_l20_20929

/-- Definition of figures and their lines of symmetry -/
def lines_of_symmetry (shape : String) : ℕ∞ :=
  match shape with
  | "equilateral_triangle" => 3
  | "regular_pentagon" => 5
  | "non_square_rectangle" => 2
  | "circle" => ∞
  | "square" => 4
  | _ => 0

/-- Proving that the circle has the greatest number of lines of symmetry -/
theorem greatest_lines_of_symmetry :
  lines_of_symmetry "circle" = ∞ ∧
  lines_of_symmetry "circle" > lines_of_symmetry "equilateral_triangle" ∧
  lines_of_symmetry "circle" > lines_of_symmetry "regular_pentagon" ∧
  lines_of_symmetry "circle" > lines_of_symmetry "non_square_rectangle" ∧
  lines_of_symmetry "circle" > lines_of_symmetry "square" := by
  sorry

end greatest_lines_of_symmetry_l20_20929


namespace one_div_thirteen_150th_digit_l20_20778

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20778


namespace broken_calculator_probability_fixed_calculator_probability_l20_20214

-- Definition for the condition when multiplication sign is broken
def broken_calculator_even_odd_probability : Prop :=
  let p_even := 0.5
  let p_odd := 0.5
  p_even = p_odd

-- Definition for the condition when multiplication sign is fixed
noncomputable def fixed_calculator_even_more_likely : Prop :=
  -- Given complex expressions, we simplify and conclude the greater likelihood of even result
  ∃ k : ℕ, let p_even := 1 - (1 / 2) ^ (k + 1) in
  p_even > 0.5

-- Problem statement for part (a)
theorem broken_calculator_probability :
  broken_calculator_even_odd_probability := 
by
  sorry

-- Problem statement for part (b)
theorem fixed_calculator_probability :
  fixed_calculator_even_more_likely := 
by
  sorry

end broken_calculator_probability_fixed_calculator_probability_l20_20214


namespace inequality_solution_a_eq_1_inequality_solution_a_gt_2_inequality_solution_a_eq_2_inequality_solution_a_lt_2_l20_20014

noncomputable def solve_inequality_a_eq_1 (x : ℝ) : Prop :=
  1 < x ∧ x < 2 ↔ x^2 - 3 * x + 2 < 0

noncomputable def solve_inequality_a_gt_2 (x a : ℝ) (h : a > 2) : Prop :=
  2 < x ∧ x < a ↔ x^2 - (a + 2) * x + 2 * a < 0

noncomputable def solve_inequality_a_eq_2 (x : ℝ) : Prop :=
  false ↔ x^2 - 4 * x + 4 < 0

noncomputable def solve_inequality_a_lt_2 (x a : ℝ) (h : a < 2) : Prop :=
  a < x ∧ x < 2 ↔ x^2 - (a + 2) * x + 2 * a < 0

theorem inequality_solution_a_eq_1 : ∀ x : ℝ, solve_inequality_a_eq_1 x := by
  sorry

theorem inequality_solution_a_gt_2 : ∀ x a : ℝ, a > 2 → solve_inequality_a_gt_2 x a := by
  sorry

theorem inequality_solution_a_eq_2 : ∀ x : ℝ, solve_inequality_a_eq_2 x := by
  sorry

theorem inequality_solution_a_lt_2 : ∀ x a : ℝ, a < 2 → solve_inequality_a_lt_2 x a := by
  sorry

end inequality_solution_a_eq_1_inequality_solution_a_gt_2_inequality_solution_a_eq_2_inequality_solution_a_lt_2_l20_20014


namespace smallest_angle_of_triangle_l20_20168

theorem smallest_angle_of_triangle (a b c : ℕ) 
    (h1 : a = 60) (h2 : b = 70) (h3 : a + b + c = 180) : 
    c = 50 ∧ min a (min b c) = 50 :=
by {
    sorry
}

end smallest_angle_of_triangle_l20_20168


namespace inequality_d_l20_20262

-- We define the polynomial f with integer coefficients
variable (f : ℤ → ℤ)

-- The function for f^k iteration
def iter (f: ℤ → ℤ) : ℕ → ℤ → ℤ
| 0, x => x
| (n + 1), x => f (iter f n x)

-- Definition of d(a, k) based on the problem statement
def d (a : ℤ) (k : ℕ) : ℝ := |(iter f k a : ℤ) - a|

-- Given condition that d(a, k) is positive
axiom d_pos (a : ℤ) (k : ℕ) : 0 < d f a k

-- The statement to be proved
theorem inequality_d (a : ℤ) (k : ℕ) : d f a k ≥ ↑k / 3 := by
  sorry

end inequality_d_l20_20262


namespace books_read_l20_20111

-- Given conditions
def chapters_per_book : ℕ := 17
def total_chapters_read : ℕ := 68

-- Statement to prove
theorem books_read : (total_chapters_read / chapters_per_book) = 4 := 
by sorry

end books_read_l20_20111


namespace fraction_inequality_l20_20033

theorem fraction_inequality {a b : ℝ} (h1 : a < b) (h2 : b < 0) : (1 / a) > (1 / b) :=
by
  sorry

end fraction_inequality_l20_20033


namespace decimal_150th_digit_l20_20883

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20883


namespace greatest_number_of_fridays_in_66_days_l20_20200

theorem greatest_number_of_fridays_in_66_days : 
  ∃ n, n = 9 ∧ ∀ weeks days, weeks = 66 / 7 ∧ days = 66 % 7 ∧ Fridays(weeks) + AdditionalFridaysFrom(days) = n :=
begin
  sorry
end

end greatest_number_of_fridays_in_66_days_l20_20200


namespace one_thirteenth_150th_digit_l20_20634

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20634


namespace quadrilateral_theorem_l20_20474

variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Angle definitions
variables (α β γ δ ε : ℝ)
variables (DAB ADB ACB DBC DBA : A → B → C → ℝ)

-- Given conditions
axiom hα : DAB A B C = α
axiom hβ : ADB A B C = β 
axiom hγ : ACB A B C = γ 
axiom hδ : DBC A B C = δ 
axiom hε : DBA A B C = ε 
axiom hα_lt_pi_div_2 : α < π / 2
axiom hβ_plus_γ : β + γ = π / 2
axiom hδ_plus_2ε : δ + 2 * ε = π

-- Points and distances
variables (DB BC AD AC : ℝ)

axiom h_DB : dist D B = DB
axiom h_BC : dist B C = BC
axiom h_AD : dist A D = AD
axiom h_AC : dist A C = AC

theorem quadrilateral_theorem : (DB + BC)^2 = AD^2 + AC^2 :=
sorry

end quadrilateral_theorem_l20_20474


namespace problem_l20_20365

noncomputable def f (x : ℝ) := 2^x - 2^(-x)
def a := (7 / 9 : ℝ)^(-1 / 2)
def b := (7 / 9 : ℝ)^( 1 / 2)
def c := Real.log2 (7 / 9 : ℝ)

theorem problem :
  f c < f b ∧ f b < f a :=
by
  sorry

end problem_l20_20365


namespace perpendicular_vectors_l20_20416

-- Define the vectors m and n
def m : ℝ × ℝ := (1, 2)
def n : ℝ × ℝ := (-3, 2)

-- Define the conditions to be checked
def km_plus_n (k : ℝ) : ℝ × ℝ := (k * m.1 + n.1, k * m.2 + n.2)
def m_minus_3n : ℝ × ℝ := (m.1 - 3 * n.1, m.2 - 3 * n.2)

-- The dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove that for k = 19, the two vectors are perpendicular
theorem perpendicular_vectors (k : ℝ) (h : k = 19) : dot_product (km_plus_n k) (m_minus_3n) = 0 := by
  rw [h]
  simp [km_plus_n, m_minus_3n, dot_product]
  sorry

end perpendicular_vectors_l20_20416


namespace decimal_150th_digit_of_1_div_13_l20_20699

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20699


namespace basketball_total_distance_l20_20228

def total_distance_traveled (initial_height : ℝ) (rebound_ratio: ℝ) (bounces: ℕ) : ℝ :=
  let descent := (0 to bounces.succ).map (λ n, initial_height * rebound_ratio^(n-1)) |>.sum
  let ascent := (0 to bounces).map (λ n, initial_height * rebound_ratio^n) |>.sum
  descent + ascent

theorem basketball_total_distance : total_distance_traveled 150 (2/5 : ℝ) 6 = 347.952 := sorry

end basketball_total_distance_l20_20228


namespace circle_area_l20_20435

theorem circle_area (r : ℝ) (h : r = 1/2) : π * r^2 = π / 4 :=
by
  rw [h]
  norm_num

end circle_area_l20_20435


namespace ME_eq_DN_l20_20220

-- Define the concepts of point, line, triangle, altitude, and perpendicular foot
variables (A B C D E M N : Type) [Point A] [Point B] [Point C] [Point D] [Point E] [Point M] [Point N]

-- Assume the conditions for the acute-angled triangle and altitudes
variables (is_acute_triangle : IsAcuteTriangle A B C)
          (altitude_AD : IsAltitude A D B C)
          (altitude_CE : IsAltitude C E A B)
          (perpendicular_AM : IsPerpendicularFoot A M D E)
          (perpendicular_CN : IsPerpendicularFoot C N D E)

-- The main proof statement
theorem ME_eq_DN : 
  distance M E = distance D N :=
sorry

end ME_eq_DN_l20_20220


namespace three_digit_even_less_than_600_count_l20_20612

theorem three_digit_even_less_than_600_count : 
  let digits := {1, 2, 3, 4, 5, 6} 
  let hundreds := {d ∈ digits | d < 6}
  let tens := digits 
  let units := {d ∈ digits | d % 2 = 0}
  ∑ (h : ℕ) in hundreds, ∑ (t : ℕ) in tens, ∑ (u : ℕ) in units, 1 = 90 :=
by
  sorry

end three_digit_even_less_than_600_count_l20_20612


namespace determine_t_l20_20480

variable (n : ℕ)

-- Define the polynomial (1 + x + x^3)^n
def g (x : ℂ) : ℂ := (1 + x + x^3)^n

-- Define the sum of certain coefficients
def t := (polynomial.coeffs (polynomial.monomial 1 1 + polynomial.monomial 1 3 + polynomial.X) ^ n).filter (fun ⟨i, _⟩ => i % 3 = 0).sum ⟨0, _⟩

-- Theorem statement that t = 2^n
theorem determine_t : t n = 2^n :=
sorry

end determine_t_l20_20480


namespace find_maximum_magnitude_of_a_l20_20475

-- Define complex numbers a and b which satisfy the given conditions
def satisfies_conditions (a b : ℂ) : Prop :=
  a^3 - 3 * a * b^2 = 36 ∧ b^3 - 3 * b * a^2 = 28 * complex.I

-- Define the main theorem stating the problem in Lean 4
theorem find_maximum_magnitude_of_a (a b : ℂ) (h : satisfies_conditions a b) :
  let M := 3 in
  |a| = M ↔ (a = 3 ∨ a = 3 * complex.exp (2 * real.pi * complex.I / 3) ∨ a = 3 * complex.exp (4 * real.pi * complex.I / 3)) :=
sorry

end find_maximum_magnitude_of_a_l20_20475


namespace sin_shift_eq_cos_l20_20600

theorem sin_shift_eq_cos (x : ℝ) : 
  ∃ (phi : ℝ), phi = π / 3 ∧ sin (2 * (x + phi) - π / 6) = cos (2 * x) := 
by
  use π / 3
  split
  . refl
  . rw [add_mul, mul_add, ← add_sub_assoc, sin_add, sin_sub, mul_div_cancel', cos_add, cos_sub, sin_pi, cos_pi, mul_div_cancel', mul_div_cancel]
  sorry

end sin_shift_eq_cos_l20_20600


namespace digit_150_of_decimal_1_div_13_l20_20657

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20657


namespace count_isosceles_triangles_l20_20954

variable (A B C D E F O : Type) [RegularHexagon A B C D E F O]

theorem count_isosceles_triangles : 
  num_isosceles_triangles A B C D E F O = 20 := sorry

end count_isosceles_triangles_l20_20954


namespace mode_and_median_l20_20234

-- Define ages and corresponding frequencies
def ages : List Nat := [12, 12, 14, 14, 15, 15, 15, 16]

-- Define the proof statement
theorem mode_and_median :
  let sorted_ages := ages.sorted
  let mode_value := 15
  let median_value := 14
  (mode_value = 15) ∧ (median_value = 14) :=
by
  sorry

end mode_and_median_l20_20234


namespace bird_average_l20_20109

theorem bird_average (a b c : ℤ) (h1 : a = 7) (h2 : b = 11) (h3 : c = 9) :
  (a + b + c) / 3 = 9 :=
by
  sorry

end bird_average_l20_20109


namespace coprime_and_divisible_l20_20093

theorem coprime_and_divisible (n : ℤ) (chosen : Finset ℤ)
  (h_chosen_size : chosen.card = n + 1)
  (h_chosen_subset : ∀ x ∈ chosen, x ∈ Finset.range (2 * n + 1)) :
  (∃ a b ∈ chosen, Int.gcd a b = 1) ∧ (∃ a b ∈ chosen, a ∣ b ∨ b ∣ a) :=
by sorry

end coprime_and_divisible_l20_20093


namespace quadrilateral_trapezoid_or_parallelogram_l20_20263

noncomputable section

open EuclideanGeometry

variables {A B C D I M N : Point}
variables {a b c d im in : ℝ}

def is_circumscribed_around_circle (ABCD : Quadrilateral) (I : Point) : Prop := 
  let ⟨A, B, C, D⟩ := ABCD in ∃ r, 
    dist I A = r ∧ dist I B = r ∧ dist I C = r ∧ dist I D = r

def midpoint (P Q R: Point) : Prop := 
  dist P Q = dist P R ∧ dist Q R = dist Q R / 2

def IM_eq_ratio (IM : ℝ) (AB : ℝ) (IN : ℝ) (CD : ℝ) : Prop :=
  IM / AB = IN / CD

axiom quadrilateral_circumscribed (ABCD : Quadrilateral) (I : Point) :
  is_circumscribed_around_circle ABCD I

axiom points_midpoints (A B C D M N : Point) :
  midpoint M A B ∧ midpoint N C D

axiom given_ratios (IM : ℝ) (a : ℝ) (IN : ℝ) (c : ℝ) :
  IM_eq_ratio IM a IN c

theorem quadrilateral_trapezoid_or_parallelogram
  (ABCD : Quadrilateral) (I : Point) (A B C D M N : Point)
  (IM : ℝ) (a : ℝ) (IN : ℝ) (c : ℝ):
  is_circumscribed_around_circle ABCD I →
  midpoint M A B ∧ midpoint N C D →
  IM_eq_ratio IM a IN c →
  (trapezoid ABCD ∨ parallelogram ABCD) :=
by
  intros h1 h2 h3
  sorry

end quadrilateral_trapezoid_or_parallelogram_l20_20263


namespace one_div_thirteen_150th_digit_l20_20818

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20818


namespace compare_abc_l20_20995

noncomputable def a : ℕ := 3^3^3^3^3^3^3^3^3^3^3^3^3^3^3^3^3^3^3^3^3^3^3^3^3
noncomputable def b : ℕ := 4^4^4^4^4^4^4^4^4^4^4^4^4^4^4^4^4^4^4^4
noncomputable def c : ℕ := 3125 -- 5^5

theorem compare_abc : c < a ∧ a < b := 
by
  -- Using greater than signs within Lean notation for combining steps
  show c < a, by sorry,
  show a < b, by sorry

end compare_abc_l20_20995


namespace card_probability_l20_20184

theorem card_probability :
  let total_cards := 52
  let hearts := 13
  let clubs := 13
  let spades := 13
  let prob_heart_first := hearts / total_cards
  let remaining_after_heart := total_cards - 1
  let prob_club_second := clubs / remaining_after_heart
  let remaining_after_heart_and_club := remaining_after_heart - 1
  let prob_spade_third := spades / remaining_after_heart_and_club
  (prob_heart_first * prob_club_second * prob_spade_third) = (2197 / 132600) :=
  sorry

end card_probability_l20_20184


namespace tip_percentage_l20_20558

theorem tip_percentage 
  (total_bill : ℕ) 
  (silas_payment : ℕ) 
  (remaining_friend_payment_with_tip : ℕ) 
  (num_remaining_friends : ℕ) 
  (num_friends : ℕ)
  (h1 : total_bill = 150) 
  (h2 : silas_payment = total_bill / 2) 
  (h3 : num_remaining_friends = 5)
  (h4 : remaining_friend_payment_with_tip = 18)
  : (remaining_friend_payment_with_tip - (total_bill / 2 / num_remaining_friends) * num_remaining_friends) / total_bill * 100 = 10 :=
by
  sorry

end tip_percentage_l20_20558


namespace spy_probabilistic_detection_failure_l20_20272

noncomputable def undetectable_probability (L: ℝ) : ℝ :=
  2 - (Real.sqrt 3 / 2) - (Real.pi / 3)

-- Define the conditions: square forest, four direction finders, one non-functional finder
variable (L : ℝ)
axiom forest_square : L = 10

-- The final statement of the problem in Lean
theorem spy_probabilistic_detection_failure:
  undetectable_probability 10 = 0.087 :=
by
  sorry

end spy_probabilistic_detection_failure_l20_20272


namespace smallest_x_division_remainder_l20_20924

theorem smallest_x_division_remainder :
  ∃ x : ℕ, x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7 ∧ x = 167 := by
  sorry

end smallest_x_division_remainder_l20_20924


namespace one_thirteen_150th_digit_l20_20904

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20904


namespace solve_for_x_l20_20134

theorem solve_for_x (x : ℝ) (h : 10 - x = 15) : x = -5 :=
by
  sorry

end solve_for_x_l20_20134


namespace digit_150_of_decimal_1_div_13_l20_20672

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20672


namespace digit_150th_of_fraction_l20_20722

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20722


namespace circle_equation_l20_20398

-- Definitions of the conditions
def passes_through (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (r : ℝ) : Prop :=
  (c - a) ^ 2 + (d - b) ^ 2 = r ^ 2

def center_on_line (a : ℝ) (b : ℝ) : Prop :=
  a - b - 4 = 0

-- Statement of the problem to be proved
theorem circle_equation 
  (a b r : ℝ) 
  (h1 : passes_through a b (-1) (-4) r)
  (h2 : passes_through a b 6 3 r)
  (h3 : center_on_line a b) :
  -- Equation of the circle
  (a = 3 ∧ b = -1 ∧ r = 5) → ∀ x y : ℝ, 
    (x - 3)^2 + (y + 1)^2 = 25 :=
sorry

end circle_equation_l20_20398


namespace one_over_thirteen_150th_digit_l20_20690

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20690


namespace product_of_squares_is_perfect_square_l20_20476

theorem product_of_squares_is_perfect_square (a b c : ℤ) (h : a * b + b * c + c * a = 1) :
    ∃ k : ℤ, (1 + a^2) * (1 + b^2) * (1 + c^2) = k^2 :=
sorry

end product_of_squares_is_perfect_square_l20_20476


namespace digit_150_of_1_div_13_l20_20808

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20808


namespace one_div_thirteen_150th_digit_l20_20749

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20749


namespace decimal_150th_digit_l20_20878

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20878


namespace digit_150_after_decimal_of_one_thirteenth_l20_20870

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20870


namespace center_of_circle_in_polar_coordinates_l20_20455

theorem center_of_circle_in_polar_coordinates :
  ∀ θ ρ, 
  (∃ (x y : ℝ), ρ = (x ^ 2 + y ^ 2) ∧ x = ρ * cos θ ∧ y = ρ * sin θ ∧ (x^2 + (y - 1/2)^2 = 1/4))
  → ρ = 1/2 ∧ θ = π/2 :=
by
  sorry

end center_of_circle_in_polar_coordinates_l20_20455


namespace one_over_thirteen_150th_digit_l20_20682

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20682


namespace decimal_150th_digit_of_1_div_13_l20_20691

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20691


namespace base4_vs_base9_digits_difference_l20_20421

noncomputable def digits_count (base n : ℕ) : ℕ := 
  Nat.find (λ k, base^k > n)

theorem base4_vs_base9_digits_difference : digits_count 4 2023 - digits_count 9 2023 = 2 :=
by
  sorry

end base4_vs_base9_digits_difference_l20_20421


namespace weekly_cost_l20_20187

def cost_per_hour : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 7
def number_of_bodyguards : ℕ := 2

theorem weekly_cost :
  (cost_per_hour * hours_per_day * number_of_bodyguards * days_per_week) = 2240 := by
  sorry

end weekly_cost_l20_20187


namespace max_power_at_v0_div_3_l20_20568

variable (C S ρ v₀ : ℝ)

def force_on_sail (v : ℝ) : ℝ :=
  (C * S * ρ * (v₀ - v) ^ 2) / 2

def power (v : ℝ) : ℝ :=
  (force_on_sail C S ρ v₀ v) * v

theorem max_power_at_v0_div_3 : ∃ v : ℝ, power C S ρ v₀ v = (2 * C * S * ρ * v₀ ^ 3) / 27 ∧ v = v₀ / 3 :=
by {
  sorry
}

end max_power_at_v0_div_3_l20_20568


namespace probability_palindrome_divisible_by_11_l20_20259

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 = d5 ∧ d2 = d4

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def count_all_palindromes : ℕ :=
  9 * 10 * 10

def count_palindromes_div_by_11 : ℕ :=
  9 * 10

theorem probability_palindrome_divisible_by_11 :
  (count_palindromes_div_by_11 : ℚ) / count_all_palindromes = 1 / 10 :=
by sorry

end probability_palindrome_divisible_by_11_l20_20259


namespace find_a_l20_20147

-- Define the Fibonacci sequence.
noncomputable def Fibonacci : ℕ → ℕ
| 1     := 1
| 2     := 1
| (n + 3) := Fibonacci (n + 2) + Fibonacci (n + 1)

-- Define what it means for three Fibonacci numbers to form an increasing arithmetic sequence.
def isArithmeticSeq (a b c : ℕ) : Prop :=
  2 * b = a + c

-- Define the conditions given in the problem.
def conditions (a : ℕ) : Prop :=
  let b := a + 3 in
  let c := a + 4 in
  a + b + c = 3000 ∧ isArithmeticSeq (Fibonacci a) (Fibonacci b) (Fibonacci c)

-- The theorem we want to prove.
theorem find_a : ∃ (a : ℕ), conditions a ∧ a = 997 := sorry

end find_a_l20_20147


namespace cost_of_apples_and_bananas_l20_20177

variable (a b : ℝ) -- Assume a and b are real numbers.

theorem cost_of_apples_and_bananas (a b : ℝ) : 
  (3 * a + 2 * b) = 3 * a + 2 * b :=
by 
  sorry -- Proof placeholder

end cost_of_apples_and_bananas_l20_20177


namespace semicircle_radius_l20_20943

theorem semicircle_radius (P : ℝ) (r : ℝ) (h₁ : P = π * r + 2 * r) (h₂ : P = 198) :
  r = 198 / (π + 2) :=
sorry

end semicircle_radius_l20_20943


namespace determinant_with_d_l20_20081

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b c d : V)

-- The original determinant D
def D : ℝ := (a : V) ⬝ (b ×ₗ c)

theorem determinant_with_d (a b c d : V) : 
  let M := Matrix.vec_cons (a + d) (Matrix.vec_cons (b + d) 
    (Matrix.vec_cons (c + d) Matrix.vec_empty)) in
  Matrix.det M = D a b c + d ⬝ (b ×ₗ c) :=
sorry

end determinant_with_d_l20_20081


namespace area_difference_l20_20458

theorem area_difference (A B a b : ℝ) : (A * b) - (a * B) = A * b - a * B :=
by {
  -- proof goes here
  sorry
}

end area_difference_l20_20458


namespace sum_sins_eq_tan_l20_20393

-- Given conditions:
def sumsins (n : ℕ) : ℝ := ∑ k in finset.range n, Real.sin (4 * (k + 1))

theorem sum_sins_eq_tan (h : ∑ k in finset.range 20, Real.sin (4 * (k + 1)) = Real.tan (42)) :
  ∃ (p q : ℕ), Nat.coprime p q ∧ (p : ℝ) / q = 42 ∧ p + q = 43 :=
by
  use 42
  use 1
  split
  { exact Nat.coprime_one_right 42 }
  split
  { norm_num }
  { norm_num } 
  sorry

end sum_sins_eq_tan_l20_20393


namespace abs_T_sum_is_correct_l20_20087

def isFactorableOverIntegers (c : ℤ) : Prop :=
  ∃ (u v : ℤ), u + v = -c ∧ u * v = 2010 * c

noncomputable def T_sum : ℤ :=
  ∑ c in (finset.univ.filter isFactorableOverIntegers), c

theorem abs_T_sum_is_correct : |T_sum| = 325620 := 
sorry

end abs_T_sum_is_correct_l20_20087


namespace regular_dodecahedra_types_proof_regular_icosahedra_types_proof_l20_20939

noncomputable def regular_dodecahedra_types : List String := ["Small stellated dodecahedron", "Great dodecahedron", "Great stellated dodecahedron"]
noncomputable def regular_icosahedra_types : List String := ["Great icosahedron"]

theorem regular_dodecahedra_types_proof :
  (∀ (P : Type) [has_faces P] [has_regular_faces P] [has_edges P],
    is_regular_dodecahedron P → 
    (∃ (t : List String), t = regular_dodecahedra_types)) := 
sorry

theorem regular_icosahedra_types_proof :
  (∀ (P : Type) [has_faces P] [has_regular_faces P] [has_edges P],
    is_regular_icosahedron P → 
    (∃ (t : List String), t = regular_icosahedra_types)) := 
sorry

class has_faces (P : Type) :=
  (faces : P → Type)

class has_regular_faces (P : Type) [has_faces P] :=
  (regular : ∀ p : P, has_faces.faces p → Prop)

class has_edges (P : Type) :=
  (edges : P → Type)

def is_regular_dodecahedron (P : Type) [has_faces P] [has_regular_faces P] [has_edges P] := 
    -- Conditions to satisfy being a regular dodecahedron
    sorry

def is_regular_icosahedron (P : Type) [has_faces P] [has_regular_faces P] [has_edges P] :=
    -- Conditions to satisfy being a regular icosahedron
    sorry

end regular_dodecahedra_types_proof_regular_icosahedra_types_proof_l20_20939


namespace decimal_150th_digit_l20_20655

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20655


namespace digit_150_of_1_over_13_is_3_l20_20762

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20762


namespace S_n_eq_square_a_n_eq_formula_T_n_formula_l20_20379

-- Define the sequence a_n and its sum S_n
def a_n (n : ℕ) : ℕ := 2 * n - 1
def S_n (n : ℕ) : ℕ := n^2

-- Define the sequence b_n and the sum T_n
def b_n (n : ℕ) : ℚ := (-1)^(n - 1) * (a_n (n + 1) / (S_n n + n))
def T_n (n : ℕ) : ℚ := ∑ i in Finset.range n, b_n (i + 1)

-- Prove the statements for S_n and a_n
theorem S_n_eq_square (n : ℕ) : S_n n = n^2 := by
  sorry

theorem a_n_eq_formula (n : ℕ) : a_n n = 2 * n - 1 := by
  sorry

-- Prove the statement for T_n
theorem T_n_formula (n : ℕ) : T_n n = (n + 1 + (-1)^(n + 1)) / (n + 1) := by
  sorry

end S_n_eq_square_a_n_eq_formula_T_n_formula_l20_20379


namespace generalized_trigonometric_identity_l20_20511

theorem generalized_trigonometric_identity (x : Real) :
  sin^2 x + sin x * cos (Real.pi / 6 + x) + cos^2 (Real.pi / 6 + x) = 3 / 4 :=
  sorry

end generalized_trigonometric_identity_l20_20511


namespace interval_of_monotonic_increase_range_of_values_l20_20366

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  cos x * (m * sin x - cos x) + sin ^ 2 (π + x)

theorem interval_of_monotonic_increase (m : ℝ) (k : ℤ) (h : 0 < m): ∃ I, I = set.Icc (k * π - π / 6) (k * π + π / 3) :=
  sorry

theorem range_of_values (a b c A B C : ℝ) (h₁ : b * cos A = 2 * c * cos A - a * cos B) (h₂ : 0 < A ∧ A < π) : 
  ∃ I, I = set.Ioc (-1) 2 :=
  sorry

end interval_of_monotonic_increase_range_of_values_l20_20366


namespace A_F_O_E_concyclic_l20_20049

-- Define the geometric configuration
variables (A B C D O F E : Type) [Geometry A B C D O F E]

-- Conditions
axiom circumcenter (O : Type) (triangle : Triangle A B C) : Circumcenter O triangle
axiom point_on_BC (D : Type) (line : Line B C) : PointOn D line
axiom perp_bisector_BD (perp_bisector : PerpBisector B D) (intersection : Intersects perp_bisector AB) : IntersectPoint perp_bisector AB F
axiom perp_bisector_CD (perp_bisector : PerpBisector C D) (intersection : Intersects perp_bisector AC) : IntersectPoint perp_bisector AC E

-- Goal to prove
theorem A_F_O_E_concyclic (triangle : Triangle A B C) (O : Circumcenter O triangle) (D : PointOn D BC)
  (F : IntersectPoint (PerpBisector B D) AB)
  (E : IntersectPoint (PerpBisector C D) AC) : Concyclic A F O E :=
sorry

end A_F_O_E_concyclic_l20_20049


namespace students_neither_math_nor_physics_l20_20512

theorem students_neither_math_nor_physics :
  let total_students := 150
  let students_math := 80
  let students_physics := 60
  let students_both := 20
  total_students - (students_math - students_both + students_physics - students_both + students_both) = 30 :=
by
  sorry

end students_neither_math_nor_physics_l20_20512


namespace decimal_150th_digit_of_1_div_13_l20_20704

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20704


namespace horner_v2_value_l20_20609

def polynomial : ℤ → ℤ := fun x => 208 + 9 * x^2 + 6 * x^4 + x^6

def horner (x : ℤ) : ℤ :=
  let v0 := 1
  let v1 := v0 * x
  let v2 := v1 * x + 6
  v2

theorem horner_v2_value (x : ℤ) : x = -4 → horner x = 22 :=
by
  intro h
  rw [h]
  rfl

end horner_v2_value_l20_20609


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20733

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20733


namespace complex_sum_traces_ellipse_l20_20553

noncomputable def shape_of_trace (z : ℂ) (h : ∥z∥ = 3) : Prop :=
  let x := z.re + (z.re / (z.abs^2))
  let y := z.im - (z.im / (z.abs^2))
  (x ^ 2 / (11 / 3) ^ 2) + (y ^ 2 / (7 / 3) ^ 2) = 1

theorem complex_sum_traces_ellipse (z : ℂ) (h : ∥z∥ = 3) :
  shape_of_trace z h := sorry

end complex_sum_traces_ellipse_l20_20553


namespace digit_150_of_1_over_13_is_3_l20_20763

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20763


namespace length_DG_l20_20121

variables (a b k l : ℕ) (S : ℕ) (BC DG : ℕ)

noncomputable def area_equal (BC : ℕ) : Prop :=
  ∃ (a b k l : ℕ), a * k = b * l ∧ S = 13 * (a + b) ∧ a * k = S ∧ b * l = S ∧ BC = 13 ∧
  k < l ∧ (k - 13) * (l - 13) = 169 

theorem length_DG : area_equal BC → DG = 182 :=
by
  intros h
  have h_ex := h.some_spec.some_spec.some_spec.some_spec
  sorry

end length_DG_l20_20121


namespace log_inequality_solution_l20_20999

theorem log_inequality_solution (m : ℝ) (x : ℝ) (h1 : |m| ≤ 1) 
  (h2 : log x^2 - (2 + m) * log x + m - 1 > 0) :
  (0 < x ∧ x < 1/10) ∨ (10^3 < x) :=
sorry

end log_inequality_solution_l20_20999


namespace find_digits_l20_20619

theorem find_digits (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_digits : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (h_sum : 100 * a + 10 * b + c = (10 * a + b) + (10 * b + c) + (10 * c + a)) :
  a = 1 ∧ b = 9 ∧ c = 8 := by
  sorry

end find_digits_l20_20619


namespace digit_150_of_one_thirteenth_l20_20828

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20828


namespace no_terms_form_3_alpha_5_beta_l20_20080

def sequence (n : ℕ) : ℤ :=
  Nat.rec_on n 0 (λ n f, if n = 0 then 1 else 8 * f - sequence (n - 1))

theorem no_terms_form_3_alpha_5_beta :
  ∀ (n : ℕ), ∀ (α β : ℕ), sequence n ≠ (3 ^ α * 5 ^ β : ℤ) :=
by
  sorry

end no_terms_form_3_alpha_5_beta_l20_20080


namespace integer_triples_soln_l20_20325

theorem integer_triples_soln (x y z : ℤ) :
  (x^3 + y^3 + z^3 - 3*x*y*z = 2003) ↔ ( (x = 668 ∧ y = 668 ∧ z = 667) ∨ (x = 668 ∧ y = 667 ∧ z = 668) ∨ (x = 667 ∧ y = 668 ∧ z = 668) ) := 
by
  sorry

end integer_triples_soln_l20_20325


namespace one_div_thirteen_150th_digit_l20_20744

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20744


namespace digit_150th_of_fraction_l20_20718

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20718


namespace final_number_is_one_l20_20146

theorem final_number_is_one (n : ℕ) (h : ∀ k, 1 ≤ k ∧ k ≤ 97 → 1 ≤ k ∧ k ≤ 97):
  ∃ x, (∏ k in finset.range 97, (2 * (49 / (k + 1)) - 1) = 1) →
  x = 1 :=
begin
  sorry
end

end final_number_is_one_l20_20146


namespace infinite_primes_4k_plus_1_l20_20194

theorem infinite_primes_4k_plus_1 :
  ∃ (infinitely_many : ∀ n : ℕ, ∃ p : ℕ, prime p ∧ p = 4 * n + 1) :=
sorry

end infinite_primes_4k_plus_1_l20_20194


namespace max_power_speed_l20_20561

variables (C S ρ v₀ v : ℝ)

def force (C S ρ v₀ v : ℝ) : ℝ :=
  (C * S * ρ * (v₀ - v)^2) / 2

def power (C S ρ v₀ v : ℝ) : ℝ :=
  (C * S * ρ / 2) * v * (v₀^2 - 2 * v₀ * v + v^2)

theorem max_power_speed (C S ρ v₀ : ℝ) (hρ : ρ > 0) (hC : C > 0) (hS : S > 0) :
  ∃ v : ℝ, power C S ρ v₀ v = (C * S * ρ / 2) * (v₀^2 * (v₀/3) - 2 * v₀ * (v₀/3)^2 + (v₀/3)^3) := 
  sorry

end max_power_speed_l20_20561


namespace function_domain_l20_20153

theorem function_domain : 
  (∀ x : ℝ, (1 - x ≥ 0 ∧ x + 1 ≠ 0) ↔ (x ∈ set.Iic 1 \ {-1})) :=
by intros x; split; intro h; sorry

end function_domain_l20_20153


namespace decimal_150th_digit_l20_20645

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20645


namespace shaded_region_area_correct_l20_20241

-- Define the geometrical essentials
def circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | dist p center = radius}

-- Conditions
def circle_a := circle (0, 0) 3
def circle_b := circle (4, 0) 3
def smaller_circle := circle (2, 0) 2

-- Define the correct answer
def shaded_area := 2 * Real.pi - 4 * Real.sqrt 5

-- Theorem statement
theorem shaded_region_area_correct :
  (area of the region outside smaller_circle and inside both circles A and B) = shaded_area := sorry

end shaded_region_area_correct_l20_20241


namespace people_per_chair_l20_20589

theorem people_per_chair (c : ℕ) (r : ℕ) (p : ℕ) (h1 : c = 6) (h2 : r = 20) (h3 : p = 600) : (p / (c * r)) = 5 :=
by
  rw [h1, h2, h3]
  simp
  sorry

end people_per_chair_l20_20589


namespace decimal_150th_digit_l20_20654

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20654


namespace log_a_range_l20_20164

open Real

theorem log_a_range (a : ℝ) (h : ∀ x ∈ Icc (0 : ℝ) 3, abs (log a ((x - 1)^2 + 2)) < 1) :
  a ∈ (Ioi 6) ∪ Ioc 0 (1 / 6) :=
by
  sorry

end log_a_range_l20_20164


namespace Max_wins_count_l20_20292

-- Definitions for Chloe and Max's wins and their ratio
def Chloe_wins := 24
def Max_wins (Y : ℕ) := 8 * Y = 3 * Chloe_wins

-- The theorem to be proven
theorem Max_wins_count : ∃ Y : ℕ, Max_wins Y ∧ Y = 9 :=
by
  existsi 9
  simp [Max_wins, Chloe_wins]
  sorry

end Max_wins_count_l20_20292


namespace symmetric_line_equation_l20_20008

theorem symmetric_line_equation (l : ℝ → ℝ → Prop)
  (hx : ∀ x y : ℝ, l x y ↔ x - y + 1 = 0)
  (symmetric_l : ℝ → ℝ → Prop)
  (hs : ∀ x y : ℝ, symmetric_l x y ↔ ∃ u v: ℝ, l u v ∧ 4 - u = x ∧ v = y):
  (∀ x y : ℝ, symmetric_l x y ↔ x + y - 5 = 0) :=
begin
  intros x y,
  simp [hx, hs],
  split,
  { rintro ⟨u, v, hul, hux, hyv⟩,
    rw hx at hul,
    subst hux, subst hyv,
    linarith },
  { intro h,
    use [4 - x, y],
    rw hx,
    split,
    { linarith },
    { simp } }
end

end symmetric_line_equation_l20_20008


namespace no_such_increasing_seq_exists_l20_20118

theorem no_such_increasing_seq_exists :
  ¬(∃ (a : ℕ → ℕ), (∀ m n : ℕ, a (m * n) = a m + a n) ∧ (∀ n : ℕ, a n < a (n + 1))) :=
by
  sorry

end no_such_increasing_seq_exists_l20_20118


namespace three_digit_even_less_than_600_count_l20_20611

theorem three_digit_even_less_than_600_count : 
  let digits := {1, 2, 3, 4, 5, 6} 
  let hundreds := {d ∈ digits | d < 6}
  let tens := digits 
  let units := {d ∈ digits | d % 2 = 0}
  ∑ (h : ℕ) in hundreds, ∑ (t : ℕ) in tens, ∑ (u : ℕ) in units, 1 = 90 :=
by
  sorry

end three_digit_even_less_than_600_count_l20_20611


namespace rabbit_time_2_miles_l20_20265

def rabbit_travel_time (distance : ℕ) (rate : ℕ) : ℕ :=
  (distance * 60) / rate

theorem rabbit_time_2_miles : rabbit_travel_time 2 5 = 24 := by
  sorry

end rabbit_time_2_miles_l20_20265


namespace freq_sixth_group_is_correct_l20_20286

-- Definitions
def total_students : ℕ := 40
def freq_group1 : ℕ := 10
def freq_group2 : ℕ := 5
def freq_group3 : ℕ := 7
def freq_group4 : ℕ := 6
def freq_group5 : ℕ := 0.20 * total_students

-- Assertion to prove
theorem freq_sixth_group_is_correct : 
  (total_students - (freq_group1 + freq_group2 + freq_group3 + freq_group4 + freq_group5)) = 0.1 * total_students :=
by
  sorry

end freq_sixth_group_is_correct_l20_20286


namespace smallest_k_l20_20351

theorem smallest_k (k : ℕ) (numbers : set ℕ) (h₁ : ∀ x ∈ numbers, x ≤ 2016) (h₂ : numbers.card = k) :
  (∃ a b ∈ numbers, 672 < abs (a - b) ∧ abs (a - b) < 1344) ↔ k ≥ 674 := 
by
  sorry

end smallest_k_l20_20351


namespace fraction_meaningful_l20_20432

variable (x : ℝ)

theorem fraction_meaningful (x : ℝ) : x ≠ 4 ↔ ((x + 3)/(x - 4)).Denominator ≠ 0 :=
by sorry

end fraction_meaningful_l20_20432


namespace area_of_region_R_l20_20525

noncomputable def region_R_area (a b c d : ℝ) (A B C D : ℝ) : ℝ :=
  if a = 3 ∧ b = 3 ∧ c = 3 ∧ d = 3 ∧ A = 150 ∧ B = 150 ∧ C = 150 ∧ D = 150 then
    (9 * (Real.sqrt 6 - Real.sqrt 2)) / 8
  else
    0

theorem area_of_region_R :
  ∀ (a b c d : ℝ) (A B C D : ℝ),
    a = 3 →
    b = 3 →
    c = 3 →
    d = 3 →
    A = 150 →
    B = 150 →
    C = 150 →
    D = 150 →
  region_R_area a b c d A B C D = (9 * (Real.sqrt 6 - Real.sqrt 2)) / 8 :=
by
  intros a b c d A B C D ha hb hc hd hA hB hC hD
  simp [region_R_area, ha, hb, hc, hd, hA, hB, hC, hD]
  sorry

end area_of_region_R_l20_20525


namespace f_is_odd_f_is_increasing_l20_20569

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := x + (2 / x)

-- Proof statement for Part 1: f(x) is odd
theorem f_is_odd : ∀ (x : ℝ), f (-x) = -f(x) := by
  intro x
  unfold f
  sorry

-- Proof statement for Part 2: f(x) is increasing on [sqrt(2), +∞)
theorem f_is_increasing : ∀ (x1 x2 : ℝ), (x1 ≥ Real.sqrt 2) → (x2 ≥ Real.sqrt 2) → 
  (x1 < x2) → (f x1 < f x2) := by
  intros x1 x2 hx1 hx2 h
  unfold f
  sorry

end f_is_odd_f_is_increasing_l20_20569


namespace arithmetic_sequence_8th_term_l20_20198

theorem arithmetic_sequence_8th_term :
  ∃ (a1 a15 n : ℕ) (d a8 : ℝ),
  a1 = 3 ∧ a15 = 48 ∧ n = 15 ∧
  d = (a15 - a1) / (n - 1) ∧
  a8 = a1 + 7 * d ∧
  a8 = 25.5 :=
by
  sorry

end arithmetic_sequence_8th_term_l20_20198


namespace basketball_team_lineups_l20_20958

theorem basketball_team_lineups :
  (let centers := 2 in 
   let right_forwards := 2 in 
   let left_forwards := 2 in 
   let right_guard := 1 in 
   let both_guards := 3 in 
   let total_positions := centers * right_forwards * left_forwards * (right_guard + both_guards) * both_guards in
   total_positions / both_guards == 72) :=
by
  sorry

end basketball_team_lineups_l20_20958


namespace remainder_3_pow_2n_plus_8_l20_20487

theorem remainder_3_pow_2n_plus_8 (n : Nat) : (3 ^ (2 * n) + 8) % 8 = 1 := by
  sorry

end remainder_3_pow_2n_plus_8_l20_20487


namespace one_div_thirteen_150th_digit_l20_20784

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20784


namespace yoongi_class_male_students_not_wearing_glasses_l20_20176

theorem yoongi_class_male_students_not_wearing_glasses
  (total_students : ℕ)
  (ratio_male : ℚ)
  (ratio_male_glasses : ℚ)
  (male_students := ratio_male * total_students)
  (male_students_glasses := ratio_male_glasses * male_students)
  (male_students_not_glasses := male_students - male_students_glasses) : 
  total_students = 32 → ratio_male = 5 / 8 → ratio_male_glasses = 3 / 4 → male_students_not_glasses = 5 := 
by
  intros
  rw [←rat.cast_coe_nat, Int.ofNat_eq_cast, rat.cast_coe_int, rat.coe_int_mk]
  norm_num
  sorry

end yoongi_class_male_students_not_wearing_glasses_l20_20176


namespace smallest_k_for_difference_l20_20360

theorem smallest_k_for_difference (s : Finset ℕ) (h₁ : ∀ x ∈ s, x ≤ 2016) (h₂ : s.card = 674) :
  ∃ a b ∈ s, 672 < abs (a - b) ∧ abs (a - b) < 1344 :=
by
  sorry

end smallest_k_for_difference_l20_20360


namespace find_a_l20_20042

theorem find_a (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  (∀ x : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) → 
    a^(2*x) + 2*a^x - 9 ≤ 6) ∧ 
  (∃ x : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ 
    a^(2*x) + 2*a^x - 9 = 6) → 
  a = 3 ∨ a = 1 / 3 :=
sorry

end find_a_l20_20042


namespace luna_total_monthly_budget_l20_20505

noncomputable def monthlyBudget (H F P T E : ℝ) :=
  H + F + T + P + E

theorem luna_total_monthly_budget (H F P T E : ℝ)
  (hF : F = 0.60 * H)
  (hP : P = 0.10 * F)
  (hT : T = 0.25 * H)
  (hE : E = 0.15 * (F + T))
  (hTotal : H + F + T = 300) :
  monthlyBudget H F P T E ≈ 330.41 := sorry

end luna_total_monthly_budget_l20_20505


namespace count_even_three_digit_numbers_less_than_600_l20_20616

-- Define the digits
def digits : List ℕ := [1, 2, 3, 4, 5, 6]

-- Condition: the number must be less than 600, i.e., hundreds digit in {1, 2, 3, 4, 5}
def valid_hundreds (d : ℕ) : Prop := d ∈ [1, 2, 3, 4, 5]

-- Condition: the units (ones) digit must be even
def valid_units (d : ℕ) : Prop := d ∈ [2, 4, 6]

-- Problem: total number of valid three-digit numbers
def total_valid_numbers : ℕ :=
  List.product (List.product [1, 2, 3, 4, 5] digits) [2, 4, 6] |>.length

-- Proof statement
theorem count_even_three_digit_numbers_less_than_600 :
  total_valid_numbers = 90 := by
  sorry

end count_even_three_digit_numbers_less_than_600_l20_20616


namespace digit_150_of_1_div_13_l20_20798

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20798


namespace one_thirteenth_150th_digit_l20_20637

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20637


namespace digit_150_of_1_div_13_l20_20799

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20799


namespace limit_example_l20_20949

open Real

theorem limit_example : 
  Tendsto (λ n : ℕ, ((n + 1 : ℝ) ^ 3 + (n - 1 : ℝ) ^ 3) / (n ^ 3 + 1))
    atTop (𝓝 2) :=
by
  sorry

end limit_example_l20_20949


namespace side_length_of_ABC_l20_20245

def is_equilateral (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

variables (O A B C : Point)
variables (OA_dist : dist O A = 10)
variables (area_circle : π * (radius O)^2 = 314 * π)
variables (is_equil_ABC : is_equilateral A B C)
variables (BC_chord_circle : ∃ M : Point, is_midpoint M B C ∧ dist O M = radius O)
variables (O_outside_ABC : ¬ (incircle A B C) O)

theorem side_length_of_ABC : dist B C = 10 :=
by
  sorry

end side_length_of_ABC_l20_20245


namespace question_1_question_2_question_3_question_4_l20_20048

-- Define each condition as a theorem
theorem question_1 (explanation: String) : explanation = "providing for the living" :=
  sorry

theorem question_2 (usage: String) : usage = "structural auxiliary word, placed between subject and predicate, negating sentence independence" :=
  sorry

theorem question_3 (explanation: String) : explanation = "The Shang dynasty called it 'Xu,' and the Zhou dynasty called it 'Xiang.'" :=
  sorry

theorem question_4 (analysis: String) : analysis = "The statement about the 'ultimate ideal' is incorrect; the original text states that 'enabling people to live and die without regret' is 'the beginning of the King's Way.'" :=
  sorry

end question_1_question_2_question_3_question_4_l20_20048


namespace parabola_roots_difference_l20_20261

def parabola_vertex (a b c : ℝ) := (3:ℝ, -9:ℝ)
def point_on_parabola (a b c : ℝ) := (5:ℝ, 6:ℝ)

noncomputable def root_difference_proof (a b c p q : ℝ) (hpq : p > q) : ℝ :=
p - q = (4*(15:ℝ)^(1/2))/5

theorem parabola_roots_difference :
  ∀ (a b c p q : ℝ),
  parabola_vertex a b c = (3, -9) →
  point_on_parabola a b c = (5, 6) →
  p > q →
  root_difference_proof a b c p q (by assumption) = (4*(15:ℝ)^(1/2))/5 :=
by
  intros a b c p q hvertex hpoint hdiff
  sorry

end parabola_roots_difference_l20_20261


namespace units_digit_of_factorial_sum_l20_20032

theorem units_digit_of_factorial_sum :
  let T := (Finset.range 60).sum (λ k, Nat.factorial (k + 1))
  (T % 10) = 3 := 
by
  let T := (Finset.range 60).sum (λ k, Nat.factorial (k + 1))
  sorry

end units_digit_of_factorial_sum_l20_20032


namespace decimal_1_div_13_150th_digit_is_3_l20_20846

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20846


namespace intersection_of_lines_l20_20329

theorem intersection_of_lines :
  ∃ (x y : ℚ), y = 5 * x + 1 ∧ 2 * y - 3 = -6 * x ∧ x = -1 / 4 ∧ y = -1 / 4 :=
by
  use (-1 / 4), (-1 / 4)
  split
  · -- Show y = 5x + 1
    exact rfl
  split
  · -- Show 2y - 3 = -6x
    exact rfl
  split
  · -- Show x = -1/4
    exact rfl
  · -- Show y = -1/4
    exact rfl

end intersection_of_lines_l20_20329


namespace smallest_k_for_difference_l20_20361

theorem smallest_k_for_difference (s : Finset ℕ) (h₁ : ∀ x ∈ s, x ≤ 2016) (h₂ : s.card = 674) :
  ∃ a b ∈ s, 672 < abs (a - b) ∧ abs (a - b) < 1344 :=
by
  sorry

end smallest_k_for_difference_l20_20361


namespace problem_solution_l20_20414

theorem problem_solution (x y z : ℝ)
  (h1 : 1/x + 1/y + 1/z = 2)
  (h2 : 1/x^2 + 1/y^2 + 1/z^2 = 1) :
  1/(x*y) + 1/(y*z) + 1/(z*x) = 3/2 :=
sorry

end problem_solution_l20_20414


namespace intersection_of_sets_l20_20413

open Set Real

theorem intersection_of_sets :
  let U := univ : Set ℝ
  let A := {x : ℝ | 2^x > 1}
  let B := {x : ℝ | x^2 - 3 * x - 4 > 0}
  A ∩ B = {x : ℝ | x > 4} := by
  sorry

end intersection_of_sets_l20_20413


namespace one_thirteenth_150th_digit_l20_20626

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20626


namespace common_point_geometric_progression_passing_l20_20982

theorem common_point_geometric_progression_passing
  (a b c : ℝ) (r : ℝ) (h_b : b = a * r) (h_c : c = a * r^2) :
  ∃ x y : ℝ, (∀ a ≠ 0, a * x + (a * r) * y = a * r^2) → (x = 0 ∧ y = 1) :=
by
  sorry

end common_point_geometric_progression_passing_l20_20982


namespace digit_150_of_1_div_13_l20_20807

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20807


namespace sequenceProbabilitySum_l20_20271

def countValidSequences (n : Nat) : Nat :=
  let rec aux : Nat → Nat
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | k => aux (k - 1) + aux (k - 2) + aux (k - 3)
  aux n

def totalSequences (n : Nat) : Nat :=
  2 ^ n

def validOverTotal (n : Nat) : Nat × Nat :=
  let valid := countValidSequences n
  let total := totalSequences n
  let gcd := Nat.gcd valid total
  (valid / gcd, total / gcd)

theorem sequenceProbabilitySum (n : Nat) (hn : n = 12) :
  let (m, t) := validOverTotal n
  m + t = 1305 := by
  -- The proof would go here
  sorry

end sequenceProbabilitySum_l20_20271


namespace expression_undefined_at_9_l20_20314

theorem expression_undefined_at_9 (x : ℝ) : (3 * x ^ 3 - 5) / (x ^ 2 - 18 * x + 81) = 0 → x = 9 :=
by sorry

end expression_undefined_at_9_l20_20314


namespace proof_t_minus_s_l20_20975

def class_sizes := [60, 40, 30, 10, 5, 5]
def total_students := 150
def number_of_teachers := 6

def t : ℚ := (class_sizes.sum : ℚ) / number_of_teachers

def s : ℚ := class_sizes.map (λ n, n * (n : ℚ) / total_students).sum

theorem proof_t_minus_s : t - s = -16.68 := 
by
  have t_value : t = 25 := by
    dsimp [t]
    rw List.sum_cons_eq_add _ _ (60::40::30::10::5::5::[])
    norm_num
  have s_value : s = 41.68 := by
    dsimp [s]
    repeat { rw List.map_cons }
    repeat { rw List.sum_cons_eq_add }
    norm_num [class_sizes, total_students]
  rw [t_value, s_value]
  norm_num
  sorry

end proof_t_minus_s_l20_20975


namespace one_div_thirteen_150th_digit_l20_20785

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20785


namespace range_of_f_l20_20408

def f (x : ℝ) : ℝ := Real.log (Real.exp x + Real.exp (-x)) + x^2

theorem range_of_f (x : ℝ) : f (2 * x) > f (x + 3) ↔ x < -1 ∨ x > 3 :=
by
  sorry

end range_of_f_l20_20408


namespace digit_150th_of_fraction_l20_20715

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20715


namespace decimal_150th_digit_l20_20651

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20651


namespace backpack_pencil_case_combinations_l20_20064

variables (backpacks pencil_cases : ℕ)

theorem backpack_pencil_case_combinations (h1 : backpacks = 2) (h2 : pencil_cases = 2) :
  backpacks * pencil_cases = 4 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end backpack_pencil_case_combinations_l20_20064


namespace distance_on_monday_l20_20548

theorem distance_on_monday
  (dist_tuesday : ℝ)
  (dist_wednesday : ℝ)
  (dist_thursday : ℝ)
  (avg_distance_per_day : ℝ)
  (num_days : ℕ) :
  dist_tuesday = 3.8 →
  dist_wednesday = 3.6 →
  dist_thursday = 4.4 →
  avg_distance_per_day = 4 →
  num_days = 4 →
  ∃ dist_monday : ℝ, dist_monday = 4.2 :=
begin
  intros h_tuesday h_wednesday h_thursday h_avg h_days,
  sorry
end

end distance_on_monday_l20_20548


namespace digit_150_of_1_over_13_is_3_l20_20770

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20770


namespace equal_circles_of_orthocenter_l20_20519

noncomputable def orthocenter_of_triangle : Type := sorry
noncomputable def circle_through_points (A B C : Type) : Type := sorry

theorem equal_circles_of_orthocenter (A B C O : Type) [h1 : acute_triangle A B C] [h2 : orthocenter_of_triangle O A B C]:
  circle_through_points O A B = circle_through_points O B C ∧
  circle_through_points O A B = circle_through_points O C A ∧
  circle_through_points O B C = circle_through_points O C A :=
sorry

end equal_circles_of_orthocenter_l20_20519


namespace complex_sum_traces_ellipse_l20_20554

noncomputable def shape_of_trace (z : ℂ) (h : ∥z∥ = 3) : Prop :=
  let x := z.re + (z.re / (z.abs^2))
  let y := z.im - (z.im / (z.abs^2))
  (x ^ 2 / (11 / 3) ^ 2) + (y ^ 2 / (7 / 3) ^ 2) = 1

theorem complex_sum_traces_ellipse (z : ℂ) (h : ∥z∥ = 3) :
  shape_of_trace z h := sorry

end complex_sum_traces_ellipse_l20_20554


namespace decimal_150th_digit_l20_20882

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20882


namespace find_digits_l20_20453

theorem find_digits :
  ∃ (A B C D : ℕ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧
  (A * 1000 + B * 100 + C * 10 + D = 1098) :=
by {
  sorry
}

end find_digits_l20_20453


namespace melanie_total_weight_l20_20510

def weight_of_brie : ℝ := 8 / 16 -- 8 ounces converted to pounds
def weight_of_bread : ℝ := 1
def weight_of_tomatoes : ℝ := 1
def weight_of_zucchini : ℝ := 2
def weight_of_chicken : ℝ := 1.5
def weight_of_raspberries : ℝ := 8 / 16 -- 8 ounces converted to pounds
def weight_of_blueberries : ℝ := 8 / 16 -- 8 ounces converted to pounds

def total_weight : ℝ := weight_of_brie + weight_of_bread + weight_of_tomatoes + weight_of_zucchini +
                        weight_of_chicken + weight_of_raspberries + weight_of_blueberries

theorem melanie_total_weight : total_weight = 7 := by
  sorry

end melanie_total_weight_l20_20510


namespace digit_150_of_one_thirteenth_l20_20835

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20835


namespace simplify_expression_l20_20131

theorem simplify_expression (a c d x y z : ℝ) :
  (cx * (a^3 * x^3 + 3 * a^3 * y^3 + c^3 * z^3) + dz * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (cx + dz) =
  a^3 * x^3 + c^3 * z^3 + (3 * cx * a^3 * y^3 / (cx + dz)) + (3 * dz * c^3 * x^3 / (cx + dz)) :=
by
  sorry

end simplify_expression_l20_20131


namespace digit_150_of_1_over_13_is_3_l20_20767

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20767


namespace minimum_parents_needed_l20_20596

/-- 
Given conditions:
1. There are 30 students going on the excursion.
2. Each car can accommodate 5 people, including the driver.
Prove that the minimum number of parents needed to be invited on the excursion is 8.
-/
theorem minimum_parents_needed (students : ℕ) (car_capacity : ℕ) (drivers_needed : ℕ) 
  (h1 : students = 30) (h2 : car_capacity = 5) (h3 : drivers_needed = 1) 
  : ∃ (parents : ℕ), parents = 8 :=
by
  existsi 8
  sorry

end minimum_parents_needed_l20_20596


namespace shaded_region_area_correct_l20_20239

-- Define the geometrical essentials
def circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | dist p center = radius}

-- Conditions
def circle_a := circle (0, 0) 3
def circle_b := circle (4, 0) 3
def smaller_circle := circle (2, 0) 2

-- Define the correct answer
def shaded_area := 2 * Real.pi - 4 * Real.sqrt 5

-- Theorem statement
theorem shaded_region_area_correct :
  (area of the region outside smaller_circle and inside both circles A and B) = shaded_area := sorry

end shaded_region_area_correct_l20_20239


namespace integer_solutions_cubic_eq_prime_l20_20322

theorem integer_solutions_cubic_eq_prime (x y : ℤ) (p : ℕ) (hp_prime : prime p) (hp_mod : p % 4 = 3) :
  y^2 = x^3 - p^2 * x → 
  (x, y) = (0, 0) ∨ (x, y) = (p, 0) ∨ (x, y) = (-p, 0) ∨
  (x = ((p^2 + 1) / 2)^2 ∧ y = 0) :=
by sorry

end integer_solutions_cubic_eq_prime_l20_20322


namespace red_balls_count_l20_20372

theorem red_balls_count (R : ℕ) (h : 2 / (2 + R) = 0.5) : R = 2 :=
by
    sorry

end red_balls_count_l20_20372


namespace raised_arm_length_exceeds_head_l20_20931

variables (h s s' x : ℝ)

def xiaogang_height := 1.7
def shadow_without_arm := 0.85
def shadow_with_arm := 1.1

theorem raised_arm_length_exceeds_head :
  h = xiaogang_height → s = shadow_without_arm → s' = shadow_with_arm → 
  x / (s' - s) = h / s → x = 0.5 :=
by
  intros h_eq s_eq s'_eq prop
  sorry

end raised_arm_length_exceeds_head_l20_20931


namespace number_of_good_permutations_l20_20492

theorem number_of_good_permutations (n : ℕ) (hn : n ≥ 3) :
  ∃ perms : finset (fin n → ℕ), 
  (∀ a ∈ perms, (∀ k ∈ finset.range (n + 1), 2 * (finset.range k).sum (λ i, a i) % k = 0)) ∧
  perms.card = 3 * 2^(n - 2) :=
sorry

end number_of_good_permutations_l20_20492


namespace popsicle_sticks_left_l20_20304

/-- Danielle has $10 for supplies. She buys one set of molds for $3, 
a pack of 100 popsicle sticks for $1. Each bottle of juice makes 20 popsicles and costs $2.
Prove that the number of popsicle sticks Danielle will be left with after making as many popsicles as she can is 40. -/
theorem popsicle_sticks_left (initial_money : ℕ)
    (mold_cost : ℕ) (sticks_cost : ℕ) (initial_sticks : ℕ)
    (juice_cost : ℕ) (popsicles_per_bottle : ℕ)
    (final_sticks : ℕ) :
    initial_money = 10 →
    mold_cost = 3 → 
    sticks_cost = 1 → 
    initial_sticks = 100 →
    juice_cost = 2 →
    popsicles_per_bottle = 20 →
    final_sticks = initial_sticks - (popsicles_per_bottle * (initial_money - mold_cost - sticks_cost) / juice_cost) →
    final_sticks = 40 :=
by
  intros h_initial_money h_mold_cost h_sticks_cost h_initial_sticks h_juice_cost h_popsicles_per_bottle h_final_sticks
  rw [h_initial_money, h_mold_cost, h_sticks_cost, h_initial_sticks, h_juice_cost, h_popsicles_per_bottle] at h_final_sticks
  norm_num at h_final_sticks
  exact h_final_sticks

end popsicle_sticks_left_l20_20304


namespace decimal_150th_digit_of_1_div_13_l20_20702

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20702


namespace sam_money_left_l20_20126

theorem sam_money_left (initial_amount : ℕ) (book_cost : ℕ) (number_of_books : ℕ) (initial_amount_eq : initial_amount = 79) (book_cost_eq : book_cost = 7) (number_of_books_eq : number_of_books = 9) : initial_amount - book_cost * number_of_books = 16 :=
by
  rw [initial_amount_eq, book_cost_eq, number_of_books_eq]
  norm_num
  sorry

end sam_money_left_l20_20126


namespace minimum_value_sqrt_sum_l20_20024

open Real

-- Define the two circles
def circle1 (P : ℝ × ℝ) : Prop := P.1^2 + P.2^2 = 1
def circle2 (P : ℝ × ℝ) : Prop := (P.1 - 2)^2 + (P.2 - 4)^2 = 1

-- Define the minimum value condition
theorem minimum_value_sqrt_sum {a b : ℝ} (h : a^2 + b^2 = (a - 5)^2 + (b + 1)^2) :
  ∀ P : ℝ × ℝ, (circle1 P ∨ circle2 P) → 
  (√(a^2 + b^2) + √((a - 5)^2 + (b + 1)^2) ≥ √34) := 
sorry

end minimum_value_sqrt_sum_l20_20024


namespace one_div_thirteen_150th_digit_l20_20819

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20819


namespace smallest_k_674_l20_20356

theorem smallest_k_674 :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 2017) → (S.card = 674) → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ (672 < a - b) ∧ (a - b < 1344) ∨ (672 < b - a) ∧ (b - a < 1344) :=
by sorry

end smallest_k_674_l20_20356


namespace Jovana_problem_l20_20469

def Jovana_initial_weight (x : ℝ) : Prop :=
  let y := x + (x + 2.5) + (x + 3.7) - (x - 1.3)
  in (3 * y = 72.6) → (x = 5.56666667)

theorem Jovana_problem (x : ℝ) : Jovana_initial_weight x :=
by sorry

end Jovana_problem_l20_20469


namespace bird_average_l20_20108

theorem bird_average (a b c : ℤ) (h1 : a = 7) (h2 : b = 11) (h3 : c = 9) :
  (a + b + c) / 3 = 9 :=
by
  sorry

end bird_average_l20_20108


namespace geometric_sequence_min_value_l20_20091

theorem geometric_sequence_min_value
  (s : ℝ) (b1 b2 b3 : ℝ)
  (h1 : b1 = 2)
  (h2 : b2 = 2 * s)
  (h3 : b3 = 2 * s ^ 2) :
  ∃ (s : ℝ), 3 * b2 + 4 * b3 = -9 / 8 :=
by
  sorry

end geometric_sequence_min_value_l20_20091


namespace samantha_original_cans_l20_20127

theorem samantha_original_cans : 
  ∀ (cans_per_classroom : ℚ),
  (cans_per_classroom = (50 - 38) / 5) →
  (50 / cans_per_classroom) = 21 := 
by
  sorry

end samantha_original_cans_l20_20127


namespace profit_per_meter_is_15_l20_20979

def sellingPrice (meters : ℕ) : ℕ := 
    if meters = 85 then 8500 else 0

def costPricePerMeter : ℕ := 85

def totalCostPrice (meters : ℕ) : ℕ := 
    meters * costPricePerMeter

def totalProfit (meters : ℕ) (sellingPrice : ℕ) (costPrice : ℕ) : ℕ := 
    sellingPrice - costPrice

def profitPerMeter (profit : ℕ) (meters : ℕ) : ℕ := 
    profit / meters

theorem profit_per_meter_is_15 : profitPerMeter (totalProfit 85 (sellingPrice 85) (totalCostPrice 85)) 85 = 15 := 
by sorry

end profit_per_meter_is_15_l20_20979


namespace digit_150th_of_fraction_l20_20723

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20723


namespace digit_150_after_decimal_of_one_thirteenth_l20_20876

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20876


namespace digit_150_of_1_over_13_is_3_l20_20759

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20759


namespace find_a_values_l20_20040

noncomputable def function_a_max_value (a : ℝ) : ℝ :=
  a^2 + 2 * a - 9

theorem find_a_values (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : function_a_max_value a = 6) : 
    a = 3 ∨ a = 1/3 :=
  sorry

end find_a_values_l20_20040


namespace pi_div_two_minus_alpha_in_third_quadrant_l20_20429

theorem pi_div_two_minus_alpha_in_third_quadrant (α : ℝ) (k : ℤ) (h : ∃ k : ℤ, (π + 2 * k * π < α) ∧ (α < 3 * π / 2 + 2 * k * π)) : 
  ∃ k : ℤ, (π + 2 * k * π < (π / 2 - α)) ∧ ((π / 2 - α) < 3 * π / 2 + 2 * k * π) :=
sorry

end pi_div_two_minus_alpha_in_third_quadrant_l20_20429


namespace one_over_thirteen_150th_digit_l20_20674

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20674


namespace decimal_1_div_13_150th_digit_is_3_l20_20852

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20852


namespace digit_150_of_one_thirteenth_l20_20833

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20833


namespace traces_ellipse_z_plus_two_over_z_l20_20555

-- Definitions
def is_circle (z : ℂ) (r : ℝ) : Prop := abs z = r

def traces_ellipse (f : ℂ → ℂ) : Prop := 
  ∃ (a b : ℝ), ∀ (z : ℂ), is_circle z 3 → is_circle (f z) 1

-- The main theorem statement
theorem traces_ellipse_z_plus_two_over_z :
  traces_ellipse (λ z : ℂ, z + (2 / z)) :=
sorry

end traces_ellipse_z_plus_two_over_z_l20_20555


namespace raised_arm_length_exceeds_head_l20_20932

variables (h s s' x : ℝ)

def xiaogang_height := 1.7
def shadow_without_arm := 0.85
def shadow_with_arm := 1.1

theorem raised_arm_length_exceeds_head :
  h = xiaogang_height → s = shadow_without_arm → s' = shadow_with_arm → 
  x / (s' - s) = h / s → x = 0.5 :=
by
  intros h_eq s_eq s'_eq prop
  sorry

end raised_arm_length_exceeds_head_l20_20932


namespace visited_both_countries_l20_20440

theorem visited_both_countries {Total Iceland Norway Neither Both : ℕ} 
  (h1 : Total = 50) 
  (h2 : Iceland = 25)
  (h3 : Norway = 23)
  (h4 : Neither = 23) 
  (h5 : Total - Neither = 27) 
  (h6 : Iceland + Norway - Both = 27) : 
  Both = 21 := 
by
  sorry

end visited_both_countries_l20_20440


namespace circle_radius_is_sqrt_6_l20_20282

noncomputable def ellipse := {a b c : ℝ // a = 6 ∧ b = 3 ∧ c = 3 * Real.sqrt 3}

noncomputable def circle_tangent_to_ellipse (r : ℝ) : Prop :=
  ∃ (x y : ℝ), ((x - 3 * Real.sqrt 3) ^ 2 + y ^ 2 = r ^ 2) ∧ ((x ^ 2) / 36 + (y ^ 2) / 9 = 1)

theorem circle_radius_is_sqrt_6 :
  ∃ (r : ℝ), circle_tangent_to_ellipse r ∧ r = Real.sqrt 6 :=
sorry

end circle_radius_is_sqrt_6_l20_20282


namespace decimal_150th_digit_of_1_div_13_l20_20700

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20700


namespace find_height_of_brick_l20_20335

-- Declare the variables
variable (height : ℝ) (length width : ℝ) (surfaceArea : ℝ)

-- Define the conditions
def conditions (length width surfaceArea : ℝ) (height : ℝ) :=
  length = 8 ∧ width = 6 ∧ surfaceArea = 152

-- Define the surface area of the rectangular prism
def surface_area_formula (l w h : ℝ) :=
  2 * l * w + 2 * l * h + 2 * w * h

-- Define the theorem stating the height
theorem find_height_of_brick (h : ℝ) :
  conditions 8 6 152 h ∧ surface_area_formula 8 6 h = 152 → h = 2 :=
  by
    intro cond surface_area
    sorry -- Proof steps are skipped

end find_height_of_brick_l20_20335


namespace digit_150_after_decimal_of_one_thirteenth_l20_20877

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20877


namespace find_line_through_midpoint_of_hyperbola_l20_20969

theorem find_line_through_midpoint_of_hyperbola
  (x1 y1 x2 y2 : ℝ)
  (P : ℝ × ℝ := (4, 1))
  (A : ℝ × ℝ := (x1, y1))
  (B : ℝ × ℝ := (x2, y2))
  (H_midpoint : P = ((x1 + x2) / 2, (y1 + y2) / 2))
  (H_hyperbola_A : (x1^2 / 4 - y1^2 = 1))
  (H_hyperbola_B : (x2^2 / 4 - y2^2 = 1)) :
  ∃ m b : ℝ, (m = 1) ∧ (b = 3) ∧ (∀ x y : ℝ, y = m * x + b → x - y - 3 = 0) := by
  sorry

end find_line_through_midpoint_of_hyperbola_l20_20969


namespace rabbit_travel_time_l20_20267

noncomputable def rabbit_speed : ℝ := 5 -- speed of the rabbit in miles per hour
noncomputable def rabbit_distance : ℝ := 2 -- distance traveled by the rabbit in miles

theorem rabbit_travel_time :
  let t := (rabbit_distance / rabbit_speed) * 60 in
  t = 24 :=
by
  sorry

end rabbit_travel_time_l20_20267


namespace intersectionIsFocus_l20_20381

-- Given definitions based on the initial conditions
def isEllipse (M : ℝ → ℝ → Prop) (a b : ℝ) : Prop :=
  ∀ x y, M x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)

def isParallelogram (O A B C : ℝ × ℝ) : Prop :=
  B.1 = A.1 / 2 ∧ B.2 = sqrt(3) / 2 * B.2 ∧
  abs (B.2 * A.1 * sqrt(3) / 2 * B.2 / 3) = 6

def isReflectionAboutXAxis (Q E : ℝ × ℝ) : Prop :=
  E = (Q.1, -Q.2)

def lineIntersectsEllipse (PQ : ℝ → ℝ) (M : ℝ → ℝ → Prop) (P Q: ℝ × ℝ): Prop :=
  PQ P.1 = P.2 ∧ PQ Q.1 = Q.2 ∧ M P.1 P.2 ∧ M Q.1 Q.2

def lineThroughPoint (PQ : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  PQ p.1 = p.2

-- The main theorem
theorem intersectionIsFocus :
  ∃ (M : ℝ → ℝ → Prop) (a b : ℝ),
  isEllipse M a b ∧ a = 2 ∧ b = sqrt 3 ∧
  (∀ x y, M x y → x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ A B C, (0, 0) = O ∧
    isParallelogram O A B C ∧
    ∃ PQ, lineThroughPoint PQ (4, 0) ∧
      ∃ P Q E, lineIntersectsEllipse PQ M P Q ∧
        isReflectionAboutXAxis Q E ∧
        (∃ x, E = (x, 0) → x = 1)) :=
by sorry

end intersectionIsFocus_l20_20381


namespace popsicle_sticks_left_l20_20301

-- Defining the conditions
def total_money : ℕ := 10
def cost_of_molds : ℕ := 3
def cost_of_sticks : ℕ := 1
def cost_of_juice_bottle : ℕ := 2
def popsicles_per_bottle : ℕ := 20
def initial_sticks : ℕ := 100

-- Statement of the problem
theorem popsicle_sticks_left : 
  let remaining_money := total_money - cost_of_molds - cost_of_sticks
  let bottles_of_juice := remaining_money / cost_of_juice_bottle
  let total_popsicles := bottles_of_juice * popsicles_per_bottle
  let sticks_left := initial_sticks - total_popsicles
  sticks_left = 40 := by
  sorry

end popsicle_sticks_left_l20_20301


namespace one_div_thirteen_150th_digit_l20_20747

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20747


namespace triangle_dot_product_l20_20050

variable {A B C : Type}

-- Assume a triangle with sides opposite to angles A, B, and C
variables {a b c : ℝ} {cosB cosC : ℝ}

-- Given conditions
def given_conditions (a b c cosB cosC : ℝ) : Prop :=
  a = 2 ∧ c = 3 ∧ (2 * a - c) * cosB = b * cosC

-- To prove
theorem triangle_dot_product (a b c cosB cosC : ℝ) (h : given_conditions a b c cosB cosC) :
  (a = 2 ∧ c = 3 ∧ (2 * a - c) * cosB = b * cosC) → 
  let ab := a in
  let bc := c in
  let result := -ab * bc * cosB in
  result = -3 :=
by
  intro h₁
  let ab := 2
  let bc := 3
  sorry

end triangle_dot_product_l20_20050


namespace burger_cost_l20_20989

/-
Given:
1. A burger meal (burger + french fries + soft drink) costs $9.50.
2. A kid's meal (kid's burger + kid's french fries + kid's juice box) costs $5.
3. Mr. Parker buys:
   • 2 burger meals for himself and his wife (costing $19).
   • 2 burger meals for his children (costing $19).
   • 2 kid's meals for his children (costing $10).
4. Total cost for 6 meals is $48.
5. Mr. Parker saves $10 by buying the meals versus buying individual food items.
6. Cost of individual items: 
   - For burger meals: 4B + 12 + 12
   - For kid's meals: 6 + 4 + 4 
7. Total cost of individual items is $58.

Prove that the cost of a burger (B) is $5.
-/

theorem burger_cost (B : ℝ)  
  (h1 : 2 * 9.50 + 2 * 9.50 + 2 * 5 = 48)
  (h2 : 2 * (9.50) + (cost_kids_meal := 2 * 5) = 48 - 10)
  (h3 : 4 * B + 4 * 3 + 4 * 3 + 2 * 3 + 2 * 2 + 2 * 2 = 58)
  : B = 5 :=
by {
  sorry
}

end burger_cost_l20_20989


namespace sin_cos_sum_l20_20397

theorem sin_cos_sum (θ : ℝ) (hθ1 : θ ∈ Set.Icc (π/2) π) (hθ2 : tan (θ - π/4) = 3) :
  sin θ + cos θ = sqrt 5 / 5 :=
sorry

end sin_cos_sum_l20_20397


namespace one_over_thirteen_150th_digit_l20_20676

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20676


namespace sum_of_center_coords_l20_20333

theorem sum_of_center_coords (x y : ℝ) (h : x^2 + y^2 = 4 * x - 6 * y + 9) : 2 + (-3) = -1 :=
by
  sorry

end sum_of_center_coords_l20_20333


namespace decimal_1_div_13_150th_digit_is_3_l20_20860

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20860


namespace digit_150_after_decimal_of_one_thirteenth_l20_20862

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20862


namespace bus_speed_excluding_stoppages_l20_20320

theorem bus_speed_excluding_stoppages
  (speed_including_stoppages : ℝ)
  (stoppages_per_hour : ℝ)
  (speed_including_stoppages_eq : speed_including_stoppages = 35)
  (stoppages_per_hour_eq : stoppages_per_hour = 18) :
  let time_in_motion_per_hour : ℝ := 60 - stoppages_per_hour in
  let fraction_of_hour_in_motion : ℝ := time_in_motion_per_hour / 60 in
  let distance_covered_without_stoppages : ℝ := speed_including_stoppages * fraction_of_hour_in_motion in
  let speed_excluding_stoppages : ℝ := distance_covered_without_stoppages / fraction_of_hour_in_motion in
  speed_excluding_stoppages = 35 :=
by
  sorry

end bus_speed_excluding_stoppages_l20_20320


namespace quadratic_roots_proof_l20_20921

noncomputable def quadratic_roots_statement : Prop :=
  ∃ (x1 x2 : ℝ), 
    (x1 ≠ x2 ∨ x1 = x2) ∧ 
    (x1 = -20 ∧ x2 = -20) ∧ 
    (x1^2 + 40 * x1 + 300 = -100) ∧ 
    (x1 - x2 = 0 ∧ x1 * x2 = 400)  

theorem quadratic_roots_proof : quadratic_roots_statement :=
sorry

end quadratic_roots_proof_l20_20921


namespace find_second_interest_rate_l20_20527

theorem find_second_interest_rate (total_amount : ℕ) (P1 : ℕ) (interest_rate_P1 : ℚ) 
    (total_income : ℕ) (expected_interest_rate_P2 : ℚ) : 
    total_amount = 2600 →
    P1 = 1600 →
    interest_rate_P1 = 0.05 →
    total_income = 140 →
    expected_interest_rate_P2 = 0.06 :=
begin
  intros h1 h2 h3 h4,
  have P2 : ℕ := total_amount - P1, 
  have interest_from_P1 : ℚ := P1 * interest_rate_P1, 
  have interest_from_P2 : ℚ := total_income - interest_from_P1,
  have second_rate : ℚ := interest_from_P2 / P2,
  have result : second_rate = expected_interest_rate_P2, {
    calc
      second_rate = interest_from_P2 / P2 : rfl
               ... = 60 / 1000           : by { simp [P2, interest_from_P2, h1, h2, h3, h4]}
               ... = 0.06                : by norm_num,
  },
  exact result,
end

end find_second_interest_rate_l20_20527


namespace area_of_region_R_l20_20575

theorem area_of_region_R (a b : ℕ) :
  let J := (2, 7)
  let K := (5, 3)
  let L (r t : ℕ) := (r, t)
  let triangle_area (J K L : (ℕ × ℕ) × (ℕ × ℕ) × (ℕ × ℕ)) :=
    1 / 2 * abs ((L.1 - J.1) * (K.2 - J.2) - (L.2 - J.2) * (K.1 - J.1))
  let area_R := 78.25
  in
  triangle_area(J, K, L(r, t)) ≤ 10 ∧ 0 ≤ r ∧ r ≤ 10 ∧ 0 ≤ t ∧ t ≤ 10 → (300 + a) / (40 - b) = 313 / 4 → a + b = 49 :=
by
  sorry

end area_of_region_R_l20_20575


namespace palindrome_divisible_by_11_prob_l20_20257

def is_palindrome (n : ℕ) : Prop :=
  let digits := List.ofDigits [n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  in digits.reverse = digits

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def palindrome_in_range (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def random_palindrome_divisible_by_11 : Prop :=
  ∃ n, palindrome_in_range n ∧ is_palindrome n ∧ is_divisible_by_11 n

theorem palindrome_divisible_by_11_prob : 
  let total_palindromes := 9 * 10 * 10
  let valid_palindromes := 90
  let probability := valid_palindromes / total_palindromes
  probability = 1 / 10 :=
sorry

end palindrome_divisible_by_11_prob_l20_20257


namespace overlap_difference_correct_l20_20941

-- Definition of the given conditions
variables (total_students : ℕ) (geom_students : ℕ) (bio_students : ℕ)
hypothesis (h1 : total_students = 232) 
hypothesis (h2 : geom_students = 144) 
hypothesis (h3 : bio_students = 119)

-- Definition of the function to calculate the difference between the greatest and smallest overlap
def overlap_difference (total geom bio : ℕ) : ℕ :=
  let max_overlap := min geom bio in
  let min_overlap := geom + bio - total in
  max_overlap - min_overlap

-- The proof problem statement
theorem overlap_difference_correct : overlap_difference 232 144 119 = 88 := by
  sorry

end overlap_difference_correct_l20_20941


namespace variance_boys_girls_l20_20439

def boys_scores : List ℝ := [86, 94, 88, 92, 90]
def girls_scores : List ℝ := [88, 93, 93, 88, 93]

def mean (scores : List ℝ) : ℝ :=
  (scores.sum) / (scores.length)

def variance (scores : List ℝ) : ℝ :=
  (scores.map (λ x, (x - mean scores)^2).sum) / (scores.length)

theorem variance_boys_girls :
  variance boys_scores > variance girls_scores :=
by
  unfold variance boys_scores girls_scores
  sorry

end variance_boys_girls_l20_20439


namespace ratio_of_part_diminished_by_4_l20_20256

theorem ratio_of_part_diminished_by_4 (N P : ℕ) (h1 : N = 160)
    (h2 : (1/5 : ℝ) * N + 4 = P - 4) : (P - 4) / N = 9 / 40 := 
by
  sorry

end ratio_of_part_diminished_by_4_l20_20256


namespace problem_solution_l20_20437

noncomputable def triangle_inequality (a b c d1 d2 d3 : ℝ) (S : ℝ) : Prop :=
  BC = a ∧ AC = b ∧ AB = c ∧
  d_1 = d1 ∧ d_2 = d2 ∧ d_3 = d3 ∧
  S = (1/2) * a * d1 ∧ S = (1/2) * b * d2 ∧ S = (1/2) * c * d3 →
  (a / d1 + b / d2 + c / d3) ≥ ((a + b + c)^2) / (2 * S)

theorem problem_solution (a b c d1 d2 d3 S: ℝ) (h : triangle_inequality a b c d1 d2 d3 S) : 
  (a / d1 + b / d2 + c / d3) ≥ ((a + b + c)^2) / (2 * S) :=
  by sorry

end problem_solution_l20_20437


namespace average_birds_seen_correct_l20_20107

-- Define the number of birds seen by each person
def birds_seen_by_marcus : ℕ := 7
def birds_seen_by_humphrey : ℕ := 11
def birds_seen_by_darrel : ℕ := 9

-- Define the number of people
def number_of_people : ℕ := 3

-- Calculate the total number of birds seen
def total_birds_seen : ℕ := birds_seen_by_marcus + birds_seen_by_humphrey + birds_seen_by_darrel

-- Calculate the average number of birds seen
def average_birds_seen : ℕ := total_birds_seen / number_of_people

-- Proof statement
theorem average_birds_seen_correct :
  average_birds_seen = 9 :=
by
  -- Leaving the proof out as instructed
  sorry

end average_birds_seen_correct_l20_20107


namespace trigonometric_identity_l20_20363

variable {α : ℝ}

theorem trigonometric_identity (h : Real.tan α = 3) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 :=
by
  sorry

end trigonometric_identity_l20_20363


namespace one_thirteenth_150th_digit_l20_20624

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20624


namespace decimal_150th_digit_l20_20641

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20641


namespace decimal_150th_digit_l20_20644

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20644


namespace cot_B_minus_cot_C_l20_20071

theorem cot_B_minus_cot_C (ABC : Triangle)
  (D : Point) (AD_median : median ABC A D)
  (angle_AD_BC : angle AD BC = 60) :
  |cot B - cot C| = 3 / 2 :=
sorry

end cot_B_minus_cot_C_l20_20071


namespace clock_hour_rotation_l20_20345

theorem clock_hour_rotation (start end division_deg : ℕ) (degree_each : ℕ) 
  (h1 : end - start = 3) (h2 : division_deg = 30)
  (h3 : degree_each = 1) : 
  (end - start) * division_deg * degree_each = 90 :=
by
  sorry

end clock_hour_rotation_l20_20345


namespace probability_compensation_l20_20235

-- Define the probabilities of each vehicle getting into an accident
def p1 : ℚ := 1 / 20
def p2 : ℚ := 1 / 21

-- Define the probability of the complementary event
def comp_event : ℚ := (1 - p1) * (1 - p2)

-- Define the overall probability that at least one vehicle gets into an accident
def comp_unit : ℚ := 1 - comp_event

-- The theorem to be proved: the probability that the unit will receive compensation from this insurance within a year is 2 / 21
theorem probability_compensation : comp_unit = 2 / 21 :=
by
  -- giving the proof is not required
  sorry

end probability_compensation_l20_20235


namespace polynomial_root_evaluation_l20_20593

theorem polynomial_root_evaluation (Q : Polynomial ℚ) (hdeg : Q.degree = 6) (hcoeff : Q.leadingCoeff = 1)
  (hroot : Polynomial.eval (√3 + √7) Q = 0) : Polynomial.eval 1 Q = 0 := 
sorry

end polynomial_root_evaluation_l20_20593


namespace consecutive_integers_count_l20_20504

theorem consecutive_integers_count : 
  (∃ K : List ℤ, 
    (∀ i j : ℤ, i ∈ K → j ∈ K → (j - i) ∈ (list.range (list.foldr max 1 K + 1))),
    -3 ∈ K ∧ ∃ n : ℤ, n ∈ K ∧ 1 ∈ K..n ∧ (n - 1) = 4) → (K.length = 9) :=
sorry

end consecutive_integers_count_l20_20504


namespace digit_150_of_one_thirteenth_l20_20841

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20841


namespace problem_1_problem_2_l20_20404

-- Define the given function
def f (x : ℝ) := |x - 1|

-- Problem 1: Prove if f(x) + f(1 - x) ≥ a always holds, then a ≤ 1
theorem problem_1 (a : ℝ) : 
  (∀ x : ℝ, f x + f (1 - x) ≥ a) → a ≤ 1 :=
  sorry

-- Problem 2: Prove if a + 2b = 8, then f(a)^2 + f(b)^2 ≥ 5
theorem problem_2 (a b : ℝ) : 
  (a + 2 * b = 8) → (f a)^2 + (f b)^2 ≥ 5 :=
  sorry

end problem_1_problem_2_l20_20404


namespace one_thirteenth_150th_digit_l20_20638

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20638


namespace sin_alpha_value_l20_20483

-- Declare the parameters in a general form
variables (α : ℝ) (x : ℝ)

-- Declare the conditions
def in_second_quadrant (α : ℝ) : Prop := π/2 < α ∧ α < π
def cos_condition (α x : ℝ) : Prop := cos α = (real.sqrt 2 / 4) * x
def point_P (x : ℝ) : Prop := x ≠ 0

-- State the theorem
theorem sin_alpha_value (h1 : in_second_quadrant α) (h2 : point_P x) (h3 : cos_condition α x) : sin α = real.sqrt 10 / 4 :=
by
  simp [h1, h2, h3]
  sorry

end sin_alpha_value_l20_20483


namespace one_div_thirteen_150th_digit_l20_20750

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20750


namespace count_six_digits_in_1_to_100_l20_20030

def has_digit_six (n : ℕ) : Prop :=
  n % 10 = 6 ∨ (n / 10) % 10 = 6

def count_integers_with_digit_six (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).countp has_digit_six

theorem count_six_digits_in_1_to_100 : count_integers_with_digit_six 1 100 = 19 := 
by
  sorry

end count_six_digits_in_1_to_100_l20_20030


namespace distinct_m_values_and_largest_sum_to_3_l20_20418

theorem distinct_m_values_and_largest_sum_to_3 (a b c : ℝ) (h1 : a * b * c > 0) (h2 : a + b + c = 0) : 
  let m := (| -c | / c) + (2 * | -a | / a) + (3 * | -b | / b) in
  let x := 3 in  -- Manual step for finding x
  let y := 0 in  -- Manual step for finding y
  x + y = 3 :=
by
  sorry

end distinct_m_values_and_largest_sum_to_3_l20_20418


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20730

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20730


namespace one_thirteen_150th_digit_l20_20905

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20905


namespace one_div_thirteen_150th_digit_l20_20813

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20813


namespace digit_150_of_1_div_13_l20_20809

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20809


namespace general_term_of_sequence_l20_20410

noncomputable def sequence (n : ℕ) : ℕ → ℝ 
| 1 := 1
| (n+2) := (sequence (n+1)) / (sequence (n+1) + 1)

theorem general_term_of_sequence (n : ℕ) (hn_pos : 0 < n) : 
  (sequence n) = 1 / n := 
sorry

end general_term_of_sequence_l20_20410


namespace circle_value_of_m_l20_20154

theorem circle_value_of_m (m : ℝ) : (∃ a b r : ℝ, r > 0 ∧ (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2) ↔ m < 1/2 := by
  sorry

end circle_value_of_m_l20_20154


namespace angle_l20_20114

open EuclideanGeometry

variables {A B C B₁ C₁ X : Point}

/--
Points  C₁, B₁  on sides  AB, AC  respectively of triangle  ABC  are such that  BB₁ ⊥  CC₁.
Point  X  lying inside the triangle is such that  ∠ XBC = ∠ B₁BA ∧ ∠ XCB = ∠ C₁CA.
Prove that  ∠ B₁XC₁ = 90° - ∠ A.
-/
theorem angle B₁XC₁_eq_90_sub_angle_A
  (h₀ : Point_in_triangle A B C B₁)
  (h₁ : Point_in_triangle A B C C₁)
  (h₂ : ∠(B B₁) = 90°)
  (h₃ : Point_in_triangle A B C X)
  (h₄ : (∠ X B C = ∠ B₁ B A) ∧ (∠ X C B = ∠ C₁CA))
  : ∠ B₁ X C₁ = 90° - ∠ A :=
by
  sorry

end angle_l20_20114


namespace max_value_of_f_l20_20298

noncomputable def f (x a : ℝ) : ℝ := -x^2 + 4 * x + a

theorem max_value_of_f (a : ℝ) (h_min : min (f 0 a) (f 1 a) = -2) :
  max (f 0 a) (f 1 a) = 1 :=
by
  sorry

end max_value_of_f_l20_20298


namespace decimal_1_div_13_150th_digit_is_3_l20_20851

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20851


namespace max_friendly_groups_19_max_friendly_groups_20_l20_20956

def friendly_group {Team : Type} (beat : Team → Team → Prop) (A B C : Team) : Prop :=
  beat A B ∧ beat B C ∧ beat C A

def max_friendly_groups_19_teams : ℕ := 285
def max_friendly_groups_20_teams : ℕ := 330

theorem max_friendly_groups_19 {Team : Type} (n : ℕ) (h : n = 19) (beat : Team → Team → Prop) :
  ∃ (G : ℕ), G = max_friendly_groups_19_teams := sorry

theorem max_friendly_groups_20 {Team : Type} (n : ℕ) (h : n = 20) (beat : Team → Team → Prop) :
  ∃ (G : ℕ), G = max_friendly_groups_20_teams := sorry

end max_friendly_groups_19_max_friendly_groups_20_l20_20956


namespace jane_can_make_nine_glasses_l20_20075

theorem jane_can_make_nine_glasses (total_lemons : ℝ) (lemons_per_glass : ℝ) (h1 : total_lemons = 18.0) (h2 : lemons_per_glass = 2.0) :
  total_lemons / lemons_per_glass = 9 :=
by
  rw [h1, h2]
  norm_num
  sorry

end jane_can_make_nine_glasses_l20_20075


namespace one_thirteen_150th_digit_l20_20906

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20906


namespace cos_angle_computation_l20_20396

theorem cos_angle_computation (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π/2)
  (h3 : sin (θ + 15 * real.pi / 180) = 4 / 5) :
  cos (2 * θ - 15 * real.pi / 180) = 17 * real.sqrt 2 / 50 := by
  sorry

end cos_angle_computation_l20_20396


namespace multiple_of_second_number_l20_20516

def main : IO Unit := do
  IO.println s!"Proof problem statement in Lean 4."

theorem multiple_of_second_number (x m : ℕ) 
  (h1 : 19 = m * x + 3) 
  (h2 : 19 + x = 27) : 
  m = 2 := 
sorry

end multiple_of_second_number_l20_20516


namespace increase_speed_for_safe_overtake_l20_20188

-- Definition of initial conditions
def speed_car_B : ℝ := 40 -- speed of car B in mph
def speed_car_C : ℝ := 50 -- speed of car C in mph
def dist_needed : ℝ := 30 / 5280 -- 30 feet converted to miles
def dist_car_C_start : ℝ := 210 / 5280 -- 210 feet converted to miles

-- The theorem to prove
theorem increase_speed_for_safe_overtake (r : ℝ) :
  (∃ d : ℝ, d = (1500 + 30 * r) / (5280 * (10 + r)) ∧ d = (10500 + 210 * r) / (5280 * (100 + r)))
  → r = 5 :=
by
  sorry

end increase_speed_for_safe_overtake_l20_20188


namespace total_apples_in_stack_l20_20252

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def number_of_apples_in_layer (layers : ℕ) : ℕ := 
  list.sum (list.map triangular_number (list.range layers))

theorem total_apples_in_stack :
  number_of_apples_in_layer 6 = 56 :=
by sorry

end total_apples_in_stack_l20_20252


namespace infinitely_many_primes_4k1_l20_20195

theorem infinitely_many_primes_4k1 (h : ∀ m : ℕ, ∃ p : ℕ, p.prime ∧ p = 4 * k + 1 ∧ p ∣ (4 * m + 1)) : 
  ∃ inf : ℕ → ℕ, ∀ n : ℕ, (inf n).prime ∧ inf n = 4 * k + 1 := 
sorry

end infinitely_many_primes_4k1_l20_20195


namespace problem1_problem2_problem3_l20_20407

open Real

def f (x : ℝ) := exp x + exp (-x)

-- Problem 1
theorem problem1 : ∀ x : ℝ, f(-x) = f(x) :=
by sorry

-- Problem 2
theorem problem2 (m : ℝ) : (∀ x > 0, m * f x ≤ exp (-x) + m - 1) → m ≤ -1/3 :=
by sorry

-- Problem 3
theorem problem3 (a x0 : ℝ) (h_a : a > 1/2 * (exp 1 + exp (-1))) (h_x0 : x0 ≥ 1) (h_ineq : f(x0) < a * (-x0^3 + 3 * x0)) :
  (a ∈ Ioo (1/2 * (exp 1 + exp (-1))) (exp 1) → exp (a - 1) < a ^ (exp 1 - 1)) ∧
  (a = exp 1 → exp (a - 1) = a ^ (exp 1 - 1)) ∧
  (a ∈ Ioo (exp 1) (⊤) → exp (a - 1) > a ^ (exp 1 - 1)) :=
by sorry

end problem1_problem2_problem3_l20_20407


namespace exponentiation_problem_l20_20036

theorem exponentiation_problem (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 2^a * 2^b = 8) : (2^a)^b = 4 := 
sorry

end exponentiation_problem_l20_20036


namespace triangle_inequality_cosine_rule_l20_20456

theorem triangle_inequality_cosine_rule (a b c : ℝ) (A B C : ℝ)
  (hA : Real.cos A = (b^2 + c^2 - a^2) / (2 * b * c))
  (hB : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c))
  (hC : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  a^3 * Real.cos A + b^3 * Real.cos B + c^3 * Real.cos C ≤ (3 / 2) * a * b * c := 
sorry

end triangle_inequality_cosine_rule_l20_20456


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20735

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20735


namespace decimal_150th_digit_l20_20891

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20891


namespace absolute_difference_AB_l20_20419

noncomputable def A : Real := 12 / 7
noncomputable def B : Real := 20 / 7

theorem absolute_difference_AB : |A - B| = 8 / 7 := by
  sorry

end absolute_difference_AB_l20_20419


namespace platyfish_white_balls_l20_20592

theorem platyfish_white_balls :
  (∃ n : ℕ, 3 * 10 + 10 * n = 80) → ∃ n : ℕ, n = 5 :=
begin
 sorry
end

end platyfish_white_balls_l20_20592


namespace arithmetic_sequence_S30_l20_20062

theorem arithmetic_sequence_S30
  (S : ℕ → ℕ)
  (h_arith_seq: ∀ m : ℕ, 2 * (S (2 * m) - S m) = S m + S (3 * m) - S (2 * m))
  (h_S10: S 10 = 4)
  (h_S20: S 20 = 20) :
  S 30 = 48 := 
by
  sorry

end arithmetic_sequence_S30_l20_20062


namespace digit_150th_of_fraction_l20_20716

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20716


namespace monotonic_f_iff_l20_20400

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - a * x + 5 else 1 + 1 / x

theorem monotonic_f_iff {a : ℝ} :  
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (2 ≤ a ∧ a ≤ 4) :=
by
  sorry

end monotonic_f_iff_l20_20400


namespace problem_1_problem_2_l20_20499

theorem problem_1 (a b c: ℝ) (h1: a > 0) (h2: b > 0) :
  a^3 + b^3 ≥ a^2 * b + a * b^2 :=
by
  sorry

theorem problem_2 (a b c: ℝ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
by
  sorry

end problem_1_problem_2_l20_20499


namespace find_A_time_l20_20139

noncomputable def work_rate_equations (W : ℝ) (A B C : ℝ) : Prop :=
  B + C = W / 2 ∧ A + B = W / 2 ∧ C = W / 3

theorem find_A_time {W A B C : ℝ} (h : work_rate_equations W A B C) :
  W / A = 3 :=
sorry

end find_A_time_l20_20139


namespace one_div_thirteen_150th_digit_l20_20756

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20756


namespace mode_is_11_l20_20232

noncomputable def mode_of_production : Nat :=
  let production_count : List (Nat × Nat) := [
    (10, 1), 
    (11, 5), 
    (12, 4), 
    (13, 3), 
    (14, 2), 
    (15, 1)]
  let frequency : Nat -> Nat := fun x => (production_count.filter (fun p => p.fst = x)).headD (0, 0).snd
  (List.maximumBy (compare on frequency) (production_count.map Prod.fst)).getD 0

theorem mode_is_11 : mode_of_production = 11 := by
  sorry

end mode_is_11_l20_20232


namespace tortoise_distance_l20_20449

-- Define the initial conditions
def a₁ : ℝ := 100
def q : ℝ := 1 / 10
def a_n : ℝ := 0.01

-- Define the sum of the geometric series formula
noncomputable def S_n : ℝ := (a₁ - a_n * q) / (1 - q)

-- Prove that the total distance the tortoise has traveled is (10^5 - 1) / 900
theorem tortoise_distance :
  S_n = (10^5 - 1) / 900 :=
by
  sorry

end tortoise_distance_l20_20449


namespace part_I_part_II_l20_20025

-- Define the vectors and the dot product function
def vector_m (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (x / 4), 1)
def vector_n (x : ℝ) : ℝ × ℝ := (cos (x / 4), cos (x / 4) ^ 2)
def f (x : ℝ) : ℝ := (vector_m x).1 * (vector_n x).1 + (vector_m x).2 * (vector_n x).2

-- Part (Ⅰ)
theorem part_I (x : ℝ) (h : f x = 1) : cos (x + π / 3) = 1 / 2 :=
sorry

-- Define the acute triangle and the condition
variables {A B C : ℝ} {a b c : ℝ}
def is_acute_triangle (A B C : ℝ) : Prop := 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2

theorem part_II (h_acute : is_acute_triangle A B C) (ha : (2 * a - c) * cos B = b * cos C) :
  (sin (2 * A + π / 6) + 1 / 2) ∈ (set.Ioo ((sqrt 3 + 1) / 2) (3 / 2)) :=
sorry

end part_I_part_II_l20_20025


namespace math_problem_l20_20090

variable {α : Type*} [LinearOrderedField α]

theorem math_problem 
  (a : ℕ → α) (n : ℕ) (A : α)
  (h1 : 1 < n)
  (h2 : A + ∑ i in Finset.range n, a i ^ 2 < (1 / (n - 1)) * (∑ i in Finset.range n, a i) ^ 2) :
  ∀ i j : ℕ, (1 ≤ i) → (i < j) → (j ≤ n) → A < 2 * a i * a j :=
by
  sorry

end math_problem_l20_20090


namespace circle_tangent_to_line_eqn_l20_20000

theorem circle_tangent_to_line_eqn
    (tangent_point : { p : ℝ × ℝ // p = (2, 2) })
    (tangent_line : { l : ℝ × ℝ → Prop // l = λ p : ℝ × ℝ, 3 * p.1 + 4 * p.2 - 14 = 0 })
    (center_line : { l : ℝ × ℝ → Prop // l = λ p : ℝ × ℝ, p.1 + p.2 - 11 = 0 }) :
    ∃ (C : ℝ × ℝ → Prop), 
        (C = λ p : ℝ × ℝ, (p.1 - 5) ^ 2 + (p.2 - 6) ^ 2 = 25)
    :=
    sorry

end circle_tangent_to_line_eqn_l20_20000


namespace decimal_1_div_13_150th_digit_is_3_l20_20855

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20855


namespace decimal_1_div_13_150th_digit_is_3_l20_20844

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20844


namespace digit_150_of_decimal_1_div_13_l20_20664

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20664


namespace total_hours_verification_l20_20307

def total_hours_data_analytics : ℕ := 
  let weekly_class_homework_hours := (2 * 3 + 1 * 4 + 4) * 24 
  let lab_project_hours := 8 * 6 + (10 + 14 + 18)
  weekly_class_homework_hours + lab_project_hours

def total_hours_programming : ℕ :=
  let weekly_hours := (2 * 2 + 2 * 4 + 6) * 24
  weekly_hours

def total_hours_statistics : ℕ :=
  let weekly_class_lab_project_hours := (2 * 3 + 1 * 2 + 3) * 24
  let exam_study_hours := 9 * 5
  weekly_class_lab_project_hours + exam_study_hours

def total_hours_all_courses : ℕ :=
  total_hours_data_analytics + total_hours_programming + total_hours_statistics

theorem total_hours_verification : 
    total_hours_all_courses = 1167 := 
by 
    sorry

end total_hours_verification_l20_20307


namespace inequality_proof_l20_20479

variable (n : ℕ)
variable (a : Fin n → ℝ)

theorem inequality_proof (hpos : ∀ i, 0 < a i) (hcyc : a (⟨n, sorry⟩.1) = a 0) :
  (∑ i : Fin n, a ((i + 1) % n) / a i) ≥ 
  (∑ i : Fin n, Real.sqrt((a ((i + 1) % n))^2 + 1)/Real.sqrt(a i^2 + 1)) :=
sorry

end inequality_proof_l20_20479


namespace NP_NQ_sum_constant_l20_20403

/-- Given the ellipse C₁ with equation x²/4 + y² = 1, eccentricity √3/2, and a > b > 0,
    and the parabola C₂ with equation y² = 2px and focus (sqrt(3), 0),
    where the chord length through the midpoint is 2√6,
    we want to show:
      1. The equation of the ellipse is x²/4 + y² = 1.
      2. Under the condition |OT| = λ|OA| + 2|OB| and the product of slopes of OA and OB is -1/4,
         |NP| + |NQ| is constant and equal to 2. -/
def ellipse_C1_standard_equation : Prop :=
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧ a > b ∧ 
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1)

noncomputable def parabola_C2_chord_length : Prop :=
  ∃ (p : ℝ), 
    (y^2 = 2 * p * x) ∧
    (let F₂ := (⟨sqrt(3), (0 : ℝ)⟩ : ℝ × ℝ) in
    ∀ x : ℝ, 2 * sqrt(6) = x * (2 * sqrt(3))) ∧
    (∀ λ μ : ℝ, (∃ ε ϕ : ℝ,  ∀ (O A B T P Q : ℝ), 
    λ ^ 2 + 4 * μ ^ 2 = 1 ∧
    λ * ε + 2 * μ * ϕ = 1 ∧
    (O A B : ℝ), Σi, npq := |2|)) 

theorem NP_NQ_sum_constant : ellipse_C1_standard_equation ∧ parabola_C2_chord_length → 
  ∀ N P Q : ℝ, (|P - Q| + |N - P| = 2) :=
sorry

end NP_NQ_sum_constant_l20_20403


namespace largest_negative_a_l20_20916

noncomputable def largest_a := -0.45

theorem largest_negative_a :
  ∀ x ∈ Set.Ioo (-3 * Real.pi) (-5 * Real.pi / 2),
  (Exists (fun x => True) → (largest_a)) ∧ 
  (¬(∃ (ϵ : ℝ) (h : ϵ > 0), (∀ x ∈ Set.Ioo (-3 * Real.pi) (-5 * Real.pi / 2),
  (largest_a) < ϵ))
follows_from_condition :=
  begin
    sorry
  end

end largest_negative_a_l20_20916


namespace character_digit_representation_l20_20066

noncomputable def valid_digit_representing := ∀ (h: ℕ), h ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def unique_characters := ∀ (c1 c2: char), c1 ≠ c2 → ∀ (d1 d2: ℕ), c1 ≠ c2 → d1 ≠ d2

theorem character_digit_representation :
  (∃ (好: ℕ), valid_digit_representing 好 ∧
  (∃ (other_digits: finset ℕ), ({好} ∪ other_digits).card = 3 ∧ unique_characters '好' ∧
  111 * 好 = 222)) :=
begin
  use 2,
  split,
  { simp [valid_digit_representing] },
  {
    use finset.of_list [3, 7],
    split,
    { simp },
    { exact unique_characters '好' }
    { exact 111 * 2 = 222 },
  }
end

end character_digit_representation_l20_20066


namespace real_solution_count_l20_20332

noncomputable def f (x : ℝ) : ℝ := 
  (Finset.range 50).sum (λ i, (i+1) / (x - (i+1)))

theorem real_solution_count : 
  ∃! x : ℝ, (51 : ℕ) = (Finset.range 51).count (λ x, f x = x + 5) :=
sorry

end real_solution_count_l20_20332


namespace digit_150_after_decimal_of_one_thirteenth_l20_20865

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20865


namespace decimal_1_div_13_150th_digit_is_3_l20_20859

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20859


namespace find_differentials_l20_20570

noncomputable def u (x y : ℝ) : ℝ := sorry
noncomputable def v (x y : ℝ) : ℝ := sorry

def given_system (x y u v : ℝ) : Prop :=
  (x * u - y * v = 1) ∧ (x - y + u - v = 0)

theorem find_differentials (x y dx dy : ℝ) (h : ∃ u v, given_system x y u v) :
  ∃ (du dv : ℝ) (ux' uy' vx' vy' : ℝ) (d2u : ℝ → ℝ → ℝ), 
  (du = (u - v) / (y - x) * dy + (y - v) / (y - x) * dx) ∧ 
  (dv = (u - x) / (y - x) * dx + (x - v) / (y - x) * dy) ∧ 
  (ux' = (u - y) / (y - x)) ∧
  (uy' = (y - v) / (y - x)) ∧
  (vx' = (u - x) / (y - x)) ∧
  (vy' = (x - v) / (y - x)) ∧
  (d2u = λ dx dy, 
    2 * (u - y) / (y - x)^2 * dx^2 + 
    2 * (x + y - v - u) / (y - x)^2 * dx * dy + 
    2 * (v - x) / (y - x)^2 * dy^2) :=
by sorry

end find_differentials_l20_20570


namespace at_least_100_pairs_l20_20178

-- Definitions based on the given problem
def sizes := {41, 42, 43}
def total_boots := 600
def left_boots := 300
def right_boots := 300
def boots_per_size := 200

-- Define the counts of left and right boots for each size
variables (L41 L42 L43 R41 R42 R43 : ℕ)

-- Conditions
axiom size_distribution : L41 + L42 + L43 = left_boots ∧ R41 + R42 + R43 = right_boots
axiom size_allocation : L41 ≤ boots_per_size ∧ L42 ≤ boots_per_size ∧ L43 ≤ boots_per_size ∧ R41 ≤ boots_per_size ∧ R42 ≤ boots_per_size ∧ R43 ≤ boots_per_size

-- Theorem stating the minimum number of valid pairs
theorem at_least_100_pairs : L41 + L42 + L43 = left_boots ∧ R41 + R42 + R43 = right_boots ∧
                             L41 ≤ boots_per_size ∧ L42 ≤ boots_per_size ∧ L43 ≤ boots_per_size ∧
                             R41 ≤ boots_per_size ∧ R42 ≤ boots_per_size ∧ R43 ≤ boots_per_size →
                             ∃ (pairs : ℕ), pairs ≥ 100 :=
begin
  sorry
end

end at_least_100_pairs_l20_20178


namespace asymptote_sum_l20_20299

theorem asymptote_sum
  (A B C : ℤ)
  (f : ℝ → ℝ)
  (h1 : ∀ x > 5, f x > 0.6)
  (h2 : ∀ x, f x = x^2 / (A * x^2 + B * x + C)) :
  (A * x + B) * (x + 3) = A * (x + 3) * (x - 2) [x = -3 ∨ x = 2] →
  A + B + C = -6 :=
sorry

end asymptote_sum_l20_20299


namespace find_omega_l20_20367

noncomputable def f (ω x : ℝ) := 3 * sin (ω * x + π / 3)

theorem find_omega 
  (ω : ℝ) (h1 : ω > 0)
  (h2 : f ω (π / 6) = f ω (π / 3))
  (h3 : ∃ x, f ω x = 3 * sin(ω * (π / 4) + π / 3) 
    ∧ ∀ (x ∈ set.Ioc (π / 6) (π / 3)), (differentiable f ω x 
      ∧ deriv (f ω) x = 0 → not (x = π / 4))) :
  ω = 14 / 3 := 
sorry

end find_omega_l20_20367


namespace fifteen_a_eq_one_prob_between_point_five_and_point_eight_prob_between_point_one_and_point_five_prob_at_one_l20_20500

namespace ProbabilityProof

variable (ξ : ℕ → ℚ) (a : ℚ)

-- Condition: The distribution of the random variable ξ
axiom h_distribution : ∀ k : ℕ, k ∈ {1, 2, 3, 4, 5} → ξ k = k / 5

-- Condition: Probability function
def P (event : ℚ → Prop) : ℚ :=
  (Finset.filter (fun k => event (k / 5)) (Finset.range 6)).sum fun k => a * k

-- Proofs
theorem fifteen_a_eq_one : 15 * a = 1 :=
  sorry

theorem prob_between_point_five_and_point_eight : P (λ ξ, 0.5 < ξ ∧ ξ < 0.8) = 0.2 :=
  sorry

theorem prob_between_point_one_and_point_five : P (λ ξ, 0.1 < ξ ∧ ξ < 0.5) = 0.2 :=
  sorry

theorem prob_at_one : P (λ ξ, ξ = 1) ≠ 0.3 :=
  sorry

end ProbabilityProof

end fifteen_a_eq_one_prob_between_point_five_and_point_eight_prob_between_point_one_and_point_five_prob_at_one_l20_20500


namespace remainder_of_exponentiation_l20_20490

theorem remainder_of_exponentiation (n : ℕ) : (3 ^ (2 * n) + 8) % 8 = 1 := 
by sorry

end remainder_of_exponentiation_l20_20490


namespace star_m_equals_12_l20_20484

def sum_of_digits (x : ℕ) : ℕ :=
  x.digits.sum

def S : Finset ℕ :=
  {n | sum_of_digits n = 10 ∧ n < 1000000}.to_finset

def m : ℕ := S.card

theorem star_m_equals_12 : sum_of_digits m = 12 := by
  sorry

end star_m_equals_12_l20_20484


namespace trapezoid_perpendicular_diagonals_l20_20115

theorem trapezoid_perpendicular_diagonals 
  (a b c d : ℝ)
  (h_perp_diagonals : ∀ E, E ∈ Set.Inter (Line (AB) (CD)) ∩ (Line (BC) (DA)) → (E ∈ (Line (AB) (BC)) ∧ E ∈ (Line (CD) (DA))) ∧ ⟪AB - E, CD - E⟫ = 0) :
  b * d ≥ a * c :=
sorry

end trapezoid_perpendicular_diagonals_l20_20115


namespace not_all_trig_eq_holds_l20_20923

-- Definitions by conditions
def trig_eq_holds (x : ℝ) : Prop := 
  cos x + cos (2 * x) + cos (4 * x) = 0

def trig_eq_holds_double (x : ℝ) : Prop := 
  cos (2 * x) + cos (4 * x) + cos (8 * x) = 0

-- Theorem statement
theorem not_all_trig_eq_holds :
  ∃ x : ℝ, trig_eq_holds x ∧ ¬ trig_eq_holds_double x := 
sorry

end not_all_trig_eq_holds_l20_20923


namespace solve_for_x_l20_20541

theorem solve_for_x : ∀ x : ℝ, (9 ^ x) * (9 ^ x) * (9 ^ x) * (9 ^ x) = 81 ^ 6 → x = 3 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l20_20541


namespace pq_combined_l20_20166

-- Definitions and conditions
def p (x : ℝ) : ℝ := a * x + b
def q (x : ℝ) : ℝ := c * x ^ 2 + d * x + e

variable (a b c d e : ℝ)
variable (h1 : p(-1) = -2)
variable (h2 : q(1) = 3)
variable (h3 : c ≠ 0)
variable (h4 : d = 0)
variable (h5 : e = -c)

-- The goal is to prove p(x) + q(x) equals to the given polynomial
theorem pq_combined (x : ℝ) : p x + q x = 1.5 * x ^ 2 + 4 * x - 3.5 :=
sorry  -- proof to be filled in

end pq_combined_l20_20166


namespace linear_function_m_value_l20_20035

theorem linear_function_m_value (m : ℝ) (h : abs (m + 1) = 1) : m = -2 :=
sorry

end linear_function_m_value_l20_20035


namespace find_side_length_of_largest_square_l20_20965

theorem find_side_length_of_largest_square (A : ℝ) (hA : A = 810) :
  ∃ a : ℝ, (5 / 8) * a ^ 2 = A ∧ a = 36 := by
  sorry

end find_side_length_of_largest_square_l20_20965


namespace members_not_playing_either_instrument_l20_20441

theorem members_not_playing_either_instrument (total_members guitar_players piano_players both_players : ℕ)
  (h1 : total_members = 80)
  (h2 : guitar_players = 45)
  (h3 : piano_players = 30)
  (h4 : both_players = 18) :
  total_members - (guitar_players - both_players + piano_players - both_players + both_players) = 23 :=
by
  rw [h1, h2, h3, h4]
  sorry

end members_not_playing_either_instrument_l20_20441


namespace smallest_n_such_that_no_n_digit_is_11_power_l20_20620

theorem smallest_n_such_that_no_n_digit_is_11_power (log_11 : Real) (h : log_11 = 1.0413) : 
  ∃ n > 1, ∀ k : ℕ, ¬ (10 ^ (n - 1) ≤ 11 ^ k ∧ 11 ^ k < 10 ^ n) :=
sorry

end smallest_n_such_that_no_n_digit_is_11_power_l20_20620


namespace solve_inequality_l20_20003

open Real

theorem solve_inequality (f : ℝ → ℝ)
  (h_cos : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f (cos x) ≥ 0) :
  ∀ k : ℤ, ∀ x, (2 * ↑k * π ≤ x ∧ x ≤ 2 * ↑k * π + π) → f (sin x) ≥ 0 :=
by
  intros k x hx
  sorry

end solve_inequality_l20_20003


namespace find_S30_l20_20059

variable {S : ℕ → ℝ} -- Assuming S is a function from natural numbers to real numbers

-- Arithmetic sequence is defined such that the sum of first n terms follows a specific format
def is_arithmetic_sequence (S : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, S (n + 1) - S n = d

-- Given conditions
axiom S10 : S 10 = 4
axiom S20 : S 20 = 20
axiom S_arithmetic : is_arithmetic_sequence S

-- The equivalent proof problem
theorem find_S30 : S 30 = 48 :=
by
  sorry

end find_S30_l20_20059


namespace shaded_region_area_l20_20244

structure Circle where
  radius : ℝ
  center : ℝ × ℝ

def is_tangent_internally (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  (dist c1.center p = c1.radius) ∧ (dist c2.center p = c2.radius) ∧ (dist c1.center c2.center = c2.radius - c1.radius)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem shaded_region_area :
  ∀ (A B C : ℝ × ℝ),
  ∀ (r1 r2 : ℝ),
  (r1 = 2) →
  (r2 = 3) →
  let smaller_circle := Circle.mk r1 C in
  let larger_circle_A := Circle.mk r2 A in
  let larger_circle_B := Circle.mk r2 B in
  is_tangent_internally smaller_circle larger_circle_A A →
  is_tangent_internally smaller_circle larger_circle_B B →
  (dist A B = 2 * r1) →
  let shaded_area := 3 * π - 3 * real.arccos (1 / 3) * π - 2 * real.sqrt 5 in
  sorry

end shaded_region_area_l20_20244


namespace people_left_after_first_stop_l20_20181

def initial_people_on_train : ℕ := 48
def people_got_off_train : ℕ := 17

theorem people_left_after_first_stop : (initial_people_on_train - people_got_off_train) = 31 := by
  sorry

end people_left_after_first_stop_l20_20181


namespace roots_subtraction_l20_20216

theorem roots_subtraction (a b : ℝ) (h_roots : a * b = 20 ∧ a + b = 12) (h_order : a > b) : a - b = 8 :=
sorry

end roots_subtraction_l20_20216


namespace one_over_thirteen_150th_digit_l20_20683

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20683


namespace relationship_of_y_l20_20038

theorem relationship_of_y (k y1 y2 y3 : ℝ)
  (hk : k < 0)
  (hy1 : y1 = k / -2)
  (hy2 : y2 = k / 1)
  (hy3 : y3 = k / 2) :
  y2 < y3 ∧ y3 < y1 := by
  -- Proof omitted
  sorry

end relationship_of_y_l20_20038


namespace sqrt_6_approx_l20_20491

noncomputable def newton_iteration (x : ℝ) : ℝ :=
  (1 / 2) * x + (3 / x)

theorem sqrt_6_approx :
  let x0 : ℝ := 2
  let x1 : ℝ := newton_iteration x0
  let x2 : ℝ := newton_iteration x1
  let x3 : ℝ := newton_iteration x2
  abs (x3 - 2.4495) < 0.0001 :=
by
  sorry

end sqrt_6_approx_l20_20491


namespace one_thirteenth_150th_digit_l20_20635

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20635


namespace mario_garden_total_blossoms_l20_20506

def hibiscus_growth (initial_flowers growth_rate weeks : ℕ) : ℕ :=
  initial_flowers + growth_rate * weeks

def rose_growth (initial_flowers growth_rate weeks : ℕ) : ℕ :=
  initial_flowers + growth_rate * weeks

theorem mario_garden_total_blossoms :
  let weeks := 2
  let hibiscus1 := hibiscus_growth 2 3 weeks
  let hibiscus2 := hibiscus_growth (2 * 2) 4 weeks
  let hibiscus3 := hibiscus_growth (4 * (2 * 2)) 5 weeks
  let rose1 := rose_growth 3 2 weeks
  let rose2 := rose_growth 5 3 weeks
  hibiscus1 + hibiscus2 + hibiscus3 + rose1 + rose2 = 64 := 
by
  sorry

end mario_garden_total_blossoms_l20_20506


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20727

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20727


namespace wipes_per_pack_l20_20120

theorem wipes_per_pack (days : ℕ) (wipes_per_day : ℕ) (packs : ℕ) (total_wipes : ℕ) (n : ℕ)
    (h1 : days = 360)
    (h2 : wipes_per_day = 2)
    (h3 : packs = 6)
    (h4 : total_wipes = wipes_per_day * days)
    (h5 : total_wipes = n * packs) : 
    n = 120 := 
by 
  sorry

end wipes_per_pack_l20_20120


namespace one_div_thirteen_150th_digit_l20_20823

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20823


namespace profit_percent_is_42_point_5_l20_20948

-- Define the parameters
variables {P C : ℝ}

-- Given condition
def condition := (2 / 3) * P = 0.85 * C

-- Define the profit
def profit (P C : ℝ) := P - C

-- Define the profit percent
def profit_percent (P C : ℝ) := (profit P C / C) * 100

-- The goal to prove
theorem profit_percent_is_42_point_5 (h : condition) : profit_percent P C = 42.5 := 
by sorry

end profit_percent_is_42_point_5_l20_20948


namespace ratio_milk_water_larger_vessel_l20_20945

-- Definitions for the conditions given in the problem
def ratio_volume (V1 V2 : ℝ) : Prop := V1 / V2 = 3 / 5
def ratio_milk_water_vessel1 (M1 W1 : ℝ) : Prop := M1 / W1 = 1 / 2
def ratio_milk_water_vessel2 (M2 W2 : ℝ) : Prop := M2 / W2 = 3 / 2

-- The final goal to prove
theorem ratio_milk_water_larger_vessel (V1 V2 M1 W1 M2 W2 : ℝ)
  (h1 : ratio_volume V1 V2) 
  (h2 : V1 = M1 + W1) 
  (h3 : V2 = M2 + W2) 
  (h4 : ratio_milk_water_vessel1 M1 W1) 
  (h5 : ratio_milk_water_vessel2 M2 W2) :
  (M1 + M2) / (W1 + W2) = 1 :=
by
  -- Proof is omitted
  sorry

end ratio_milk_water_larger_vessel_l20_20945


namespace percentage_problem_l20_20233

theorem percentage_problem (n : ℕ) (h1 : n = 4800) (h2 : 0.30 * 0.50 * n = 108) : 
  ∃ P : ℝ, P = 15 ∧ (P / 100) * (0.30 * 0.50 * n) = 108 :=
by
  -- Using the conditions and the number, prove the percentage P is 15.
  sorry

end percentage_problem_l20_20233


namespace chocolate_cookies_sold_l20_20209

variable {C : ℕ} -- The number of chocolate cookies sold

theorem chocolate_cookies_sold :
  (C : ℕ) + 2 * 70 = 360 → C = 220 :=
begin
  intro h,
  linarith,
end

end chocolate_cookies_sold_l20_20209


namespace candy_distribution_l20_20344

/--
Given four people A, B, C, and D, and the conditions:
1. A gets 10 more pieces than twice what B gets.
2. A gets 18 more pieces than three times what C gets.
3. A gets 55 fewer pieces than five times what D gets.
4. They all together have 2013 pieces of candy.
Prove that the number of candies A gets is 990.
-/
theorem candy_distribution : 
  ∃ (A B C D : ℕ), 
    A = 2 * B + 10 ∧ 
    A = 3 * C + 18 ∧ 
    A = 5 * D - 55 ∧ 
    A + B + C + D = 2013 ∧ 
    A = 990 :=
begin
  sorry
end

end candy_distribution_l20_20344


namespace minimize_sum_of_squares_l20_20223

theorem minimize_sum_of_squares (a b c : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 16) :
  a^2 + b^2 + c^2 ≥ 86 :=
sorry

end minimize_sum_of_squares_l20_20223


namespace pumpkins_eaten_l20_20534

-- Definitions for the conditions
def originalPumpkins : ℕ := 43
def leftPumpkins : ℕ := 20

-- Theorem statement
theorem pumpkins_eaten : originalPumpkins - leftPumpkins = 23 :=
  by
    -- Proof steps are omitted
    sorry

end pumpkins_eaten_l20_20534


namespace union_A_B_complement_intersection_A_B_l20_20411

def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}

def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

theorem union_A_B : A ∪ B = { x | x ≥ 3 } := 
by
  sorry

theorem complement_intersection_A_B : (A ∩ B)ᶜ = { x | x < 4 } ∪ { x | x ≥ 10 } := 
by
  sorry

end union_A_B_complement_intersection_A_B_l20_20411


namespace temperature_on_Monday_l20_20551

theorem temperature_on_Monday 
  (M T W Th F : ℝ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : F = 31) : 
  M = 39 :=
by
  sorry

end temperature_on_Monday_l20_20551


namespace product_of_three_fair_dice_eq_72_l20_20599

noncomputable def prob_product_72 : ℚ :=
  -- The correct answer that we need to prove
  1 / 24

-- Definitions and mathematical context
def fair_dice_face : set ℕ := {1, 2, 3, 4, 5, 6}

theorem product_of_three_fair_dice_eq_72 : 
  (∑ a in fair_dice_face, ∑ b in fair_dice_face, ∑ c in fair_dice_face,
    if a * b * c = 72 then 1 else 0) 
  / real.to_rat(6 * 6 * 6) = prob_product_72 :=
sorry

end product_of_three_fair_dice_eq_72_l20_20599


namespace sine_rule_circumcircle_l20_20117

theorem sine_rule_circumcircle (A B C : Type*)
  [inner_product_space ℝ A] 
  [metric_space B]
  [inner_product_space ℝ B]
  [metric_space C]
  [inner_product_space ℝ C]
  (a b c : ℝ) 
  (α β γ : ℝ) -- these are angles in radians
  (R : ℝ)    -- radius of the circumscribed circle around triangle ABC
  (h : a = 2 * R * sin(α)) :
  ∀ (A B C : Point)
  [is_triangle A B C]
  (h1 : α = angle A B C)
  (h2 : α < π)
  (δ : diameter_circumcircle A B C = 2 * R),
  (a / sin(α) = 2 * R) :=
sorry

end sine_rule_circumcircle_l20_20117


namespace max_value_expression_l20_20100

theorem max_value_expression (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : x^2 + y^2 + z^2 = 1) : 
  3 * x * z * real.sqrt 3 + 9 * y * z ≤ real.sqrt ((29 * 54) / 5) :=
sorry

end max_value_expression_l20_20100


namespace area_union_of_transformed_triangle_l20_20069

theorem area_union_of_transformed_triangle
  {A B C G A' B' C' : Type}
  (AB BC AC : ℝ)
  (h_AB : AB = 9)
  (h_BC : BC = 40)
  (h_AC : AC = 41)
  (G : Point)
  (is_median_intersection : is_intersection_of_medians A B C G)
  (A' B' C' : Point)
  (h_rotate_translate : 
    ∃ θ (translation : Point → Point), 
      θ = 180 ∧ (translation (rotate_point θ G A) = A' ∧
                 translation (rotate_point θ G B) = B' ∧
                 translation (rotate_point θ G C) = C') ∧
      translation.x_shift = 5) :
  area_union_of_triangles A B C A' B' C' = 369 :=
by
  sorry

end area_union_of_transformed_triangle_l20_20069


namespace one_over_thirteen_150th_digit_l20_20680

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20680


namespace probability_of_sum_17_l20_20157

def decagonal_die_faces : Set ℕ := Set.Icc 1 10

def num_successful_outcomes : Nat :=
  (decagonal_die_faces.filter (λ x => 17 - x ∈ decagonal_die_faces)).card

def total_outcomes : Nat := decagonal_die_faces.card * decagonal_die_faces.card

theorem probability_of_sum_17 :
  (num_successful_outcomes : ℚ) / total_outcomes = 1 / 25 := by
  sorry

end probability_of_sum_17_l20_20157


namespace part1_part2_l20_20386

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part (1)
theorem part1 (a : ℝ) (h : a = 3) : (P 3)ᶜ ∩ Q = {x | -2 ≤ x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) : (∀ x, x ∈ P a → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P a) → 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end part1_part2_l20_20386


namespace decimal_150th_digit_l20_20647

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20647


namespace one_thirteen_150th_digit_l20_20910

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20910


namespace ratio_of_perimeters_l20_20143

theorem ratio_of_perimeters (h w : ℝ) (h = 3) (w1 w2 w3: ℝ)
  (w1 = 3) (w2 = 2) (w3 = 3)
  (P_small P_medium P_large : ℝ)
  (P_small = 2 * (w1 + h))
  (P_medium = 2 * (w2 + h))
  (P_large = 2 * (w3 + h)) :
  P_medium / P_large = 5 / 6 :=
sorry

end ratio_of_perimeters_l20_20143


namespace probability_palindrome_divisible_by_11_l20_20260

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 = d5 ∧ d2 = d4

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def count_all_palindromes : ℕ :=
  9 * 10 * 10

def count_palindromes_div_by_11 : ℕ :=
  9 * 10

theorem probability_palindrome_divisible_by_11 :
  (count_palindromes_div_by_11 : ℚ) / count_all_palindromes = 1 / 10 :=
by sorry

end probability_palindrome_divisible_by_11_l20_20260


namespace trajectory_equation_range_of_k_when_m_eq_1_range_of_k_and_m_l20_20057

noncomputable def point (x y : ℝ) := (x, y)
noncomputable def fixed_points := (point 0 (-Real.sqrt 3), point 0 (Real.sqrt 3))
noncomputable def curve_P (P : ℝ × ℝ) :=
  let (x, y) := P in
  let F₁ := point 0 (-Real.sqrt 3) in
  let F₂ := point 0 (Real.sqrt 3) in
  Real.abs (Real.sqrt ((x - 0)^2 + (y - (-Real.sqrt 3))^2)) - 
  Real.abs (Real.sqrt ((x - 0)^2 + (y - Real.sqrt 3)^2)) = 2

noncomputable def curve_C (x y : ℝ) := y^2 - x^2 / 2 = 1 ∧ y ≥ 1

theorem trajectory_equation :
  ∀ P : ℝ × ℝ, curve_P P → curve_C P.fst P.snd := sorry

noncomputable def line_l (x y : ℝ) (k m : ℝ) := y = k * x + m

theorem range_of_k_when_m_eq_1 :
  ∀ k : ℝ, ∃ x y : ℝ, 
  (curve_C x y ∧ (line_l x y k 1)) → k ≠ Real.sqrt 2 / 2 := sorry

theorem range_of_k_and_m :
  ∃ k m : ℝ, 
  (-Real.sqrt 2 / 2 < k ∧ k < Real.sqrt 2 / 2) ∧ 
  (-Real.sqrt 3 < m ∧ m < Real.sqrt 3) ∧
  (∃ x1 y1 x2 y2 : ℝ, 
    curve_C x1 y1 ∧ curve_C x2 y2 ∧
    line_l x1 y1 k m ∧ line_l x2 y2 k m ∧
    (Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = Real.sqrt (x1^2 + y1^2) + Real.sqrt (x2^2 + y2^2))) := sorry

end trajectory_equation_range_of_k_when_m_eq_1_range_of_k_and_m_l20_20057


namespace hexagons_formed_square_z_l20_20985

theorem hexagons_formed_square_z (a b s z : ℕ) (hexagons_congruent : a = 9 ∧ b = 16 ∧ s = 12 ∧ z = 4): 
(z = 4) := by
  sorry

end hexagons_formed_square_z_l20_20985


namespace min_period_sin_squared_plus_b_sin_plus_c_l20_20284

theorem min_period_sin_squared_plus_b_sin_plus_c (b c : ℝ) :
  ∃ p > 0, ∀ x, f (x + p) = f x ∧ (∀ q > 0, ( ∀ x, f (x + q) = f x) → p ≤ q) :=
  ∀ b c : ℝ, min_period (f : ℝ → ℝ) (λ x, sin x)^2 + b * sin x + c = sorry

end min_period_sin_squared_plus_b_sin_plus_c_l20_20284


namespace ducklings_distance_l20_20538

noncomputable def ducklings_swim (r : ℝ) (n : ℕ) : Prop :=
  ∀ (ducklings : Fin n → ℝ × ℝ), (∀ i, (ducklings i).1 ^ 2 + (ducklings i).2 ^ 2 = r ^ 2) →
    ∃ (i j : Fin n), i ≠ j ∧ (ducklings i - ducklings j).1 ^ 2 + (ducklings i - ducklings j).2 ^ 2 ≤ r ^ 2

theorem ducklings_distance :
  ducklings_swim 5 6 :=
by sorry

end ducklings_distance_l20_20538


namespace projection_of_b_on_a_l20_20399

open Real EuclideanSpace

noncomputable def vec_a : EuclideanSpace ℝ (Fin 2) := ![2, 0]
noncomputable def vec_b (magnitude_b : ℝ) (θ : ℝ) : EuclideanSpace ℝ (Fin 2) := 
  ![magnitude_b * cos θ, magnitude_b * sin θ]

theorem projection_of_b_on_a
    (angle_ab : ℝ := (2 / 3) * π)
    (magnitude_a : ℝ := 2)
    (dot_product_condition : ∀ b : EuclideanSpace ℝ (Fin 2),
      (vec_a + b) ⬝ (vec_a - 2 • b) = 0)
    (result : ℝ := - (Real.sqrt 33 + 1) / 8)
    (magnitude_b : ℝ) : 
  magnitude_a ≠ 0 ∧ 
  ⟪vec_a, vec_b magnitude_b angle_ab⟫ = magnitude_a * magnitude_b * cos angle_ab ∧
  ∃ b : EuclideanSpace ℝ (Fin 2), 
    b = vec_b magnitude_b angle_ab ∧ 
    ⟨vec_a, b⟩ / ⟨vec_a, vec_a⟩ = result :=
begin
  sorry
end

end projection_of_b_on_a_l20_20399


namespace product_of_ratios_l20_20180

theorem product_of_ratios (x y : ℕ → ℝ) (h : ∀ i, (x i)^3 - 3 * (x i) * (y i)^2 = 2017 ∧ (y i)^3 - 3 * (x i)^2 * (y i) = 2016) : 
  ∏ i in finset.range 4, (1 - x i / y i) = -1 / 1008 := 
by sorry

end product_of_ratios_l20_20180


namespace one_div_thirteen_150th_digit_l20_20777

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20777


namespace find_cost_price_l20_20215

noncomputable def cost_price (selling_price profit_percentage : ℝ) : ℝ :=
  selling_price / (1 + profit_percentage / 100)

theorem find_cost_price (SP PP : ℝ) (hSP : SP = 120) (hPP : PP = 25) :
  cost_price SP PP = 96 :=
by
  rw [hSP, hPP]
  have h : cost_price 120 25 = 120 / 1.25 := rfl
  rw [h]
  norm_num

end find_cost_price_l20_20215


namespace proof_sets_equal_l20_20390

open Set

def A : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 1}
def B : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4b + 5}

theorem proof_sets_equal : A = B :=
by
  sorry

end proof_sets_equal_l20_20390


namespace one_over_thirteen_150th_digit_l20_20688

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20688


namespace least_number_remainder_l20_20947

open Nat

theorem least_number_remainder (n : ℕ) :
  (n ≡ 4 [MOD 5]) →
  (n ≡ 4 [MOD 6]) →
  (n ≡ 4 [MOD 9]) →
  (n ≡ 4 [MOD 12]) →
  n = 184 :=
by
  intros h1 h2 h3 h4
  sorry

end least_number_remainder_l20_20947


namespace digit_150_of_one_thirteenth_l20_20840

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20840


namespace negation_equivalence_l20_20170

theorem negation_equivalence {Triangle : Type} (has_circumcircle : Triangle → Prop) :
  ¬ (∃ (t : Triangle), ¬ has_circumcircle t) ↔ (∀ (t : Triangle), has_circumcircle t) :=
by
  sorry

end negation_equivalence_l20_20170


namespace traces_ellipse_z_plus_two_over_z_l20_20556

-- Definitions
def is_circle (z : ℂ) (r : ℝ) : Prop := abs z = r

def traces_ellipse (f : ℂ → ℂ) : Prop := 
  ∃ (a b : ℝ), ∀ (z : ℂ), is_circle z 3 → is_circle (f z) 1

-- The main theorem statement
theorem traces_ellipse_z_plus_two_over_z :
  traces_ellipse (λ z : ℂ, z + (2 / z)) :=
sorry

end traces_ellipse_z_plus_two_over_z_l20_20556


namespace decimal_1_div_13_150th_digit_is_3_l20_20854

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20854


namespace digit_150_after_decimal_of_one_thirteenth_l20_20868

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20868


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20728

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20728


namespace one_over_thirteen_150th_digit_l20_20687

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20687


namespace phone_number_C_value_l20_20442

/-- 
In a phone number formatted as ABC-DEF-GHIJ, each letter symbolizes a distinct digit.
Digits in each section ABC, DEF, and GHIJ are in ascending order i.e., A < B < C, D < E < F, and G < H < I < J.
Moreover, D, E, F are consecutive odd digits, and G, H, I, J are consecutive even digits.
Also, A + B + C = 15. Prove that the value of C is 9. 
-/
theorem phone_number_C_value :
  ∃ (A B C D E F G H I J : ℕ), 
  A < B ∧ B < C ∧ D < E ∧ E < F ∧ G < H ∧ H < I ∧ I < J ∧
  (D % 2 = 1) ∧ (E % 2 = 1) ∧ (F % 2 = 1) ∧
  (G % 2 = 0) ∧ (H % 2 = 0) ∧ (I % 2 = 0) ∧ (J % 2 = 0) ∧
  (E = D + 2) ∧ (F = D + 4) ∧ (H = G + 2) ∧ (I = G + 4) ∧ (J = G + 6) ∧
  A + B + C = 15 ∧
  C = 9 := by 
  sorry

end phone_number_C_value_l20_20442


namespace one_over_thirteen_150th_digit_l20_20685

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20685


namespace decimal_150th_digit_l20_20650

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20650


namespace cheese_cutting_l20_20270

theorem cheese_cutting (α : Type) [LinearOrderedField α] :
  (∀ (a : α), 0 < a ∧ a < 1 → ∃ (N : ℕ) (b : α), 0 < b ∧ b < 1 ∧ 
    ∀ ε > 0, ∃ n ≥ N, |b^(1/n) - b^(1/(n+1))| < ε) →
  (∀ (a : α), a ∈ Ioo (0 : α) 1 ∃ (x: α), Ioo (a, a + 0.001) x) :=
by
  intro a ha h
  sorry

end cheese_cutting_l20_20270


namespace digit_150_of_1_div_13_l20_20805

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20805


namespace total_pay_is_correct_l20_20971

-- Define the constants and conditions
def regular_rate := 3  -- $ per hour
def regular_hours := 40  -- hours
def overtime_multiplier := 2  -- overtime pay is twice the regular rate
def overtime_hours := 8  -- hours

-- Calculate regular and overtime pay
def regular_pay := regular_rate * regular_hours
def overtime_rate := regular_rate * overtime_multiplier
def overtime_pay := overtime_rate * overtime_hours

-- Calculate total pay
def total_pay := regular_pay + overtime_pay

-- Prove that the total pay is $168
theorem total_pay_is_correct : total_pay = 168 := by
  -- The proof goes here
  sorry

end total_pay_is_correct_l20_20971


namespace cosine_of_angleC_l20_20445

-- Definition of the conditions
def angleC : ℝ := Real.arcsin (4/7)

def angleA_lt_50 (A : ℝ) : Prop := A < 50
def angleB_lt_70 (B : ℝ) : Prop := B < 70

-- The sum of angles in a triangle
def angle_sum (A B C : ℝ) : Prop := A + B + C = 180

-- Prove that the cosine of angle C is -√33/7
theorem cosine_of_angleC (A B C : ℝ) 
  (hA : angleA_lt_50 A)
  (hB : angleB_lt_70 B)
  (h_sum : angle_sum A B angleC) :
  Real.cos angleC = - (Real.sqrt 33) / 7 := by 
  sorry

end cosine_of_angleC_l20_20445


namespace prob_heart_club_spade_l20_20182

-- Definitions based on the conditions
def total_cards : ℕ := 52
def cards_per_suit : ℕ := 13

-- Definitions based on the question
def prob_first_heart : ℚ := cards_per_suit / total_cards
def prob_second_club : ℚ := cards_per_suit / (total_cards - 1)
def prob_third_spade : ℚ := cards_per_suit / (total_cards - 2)

-- The main proof statement to be proved
theorem prob_heart_club_spade :
  prob_first_heart * prob_second_club * prob_third_spade = 169 / 10200 :=
by
  sorry

end prob_heart_club_spade_l20_20182


namespace work_completed_in_8_days_l20_20959

theorem work_completed_in_8_days 
  (A_complete : ℕ → Prop)
  (B_complete : ℕ → Prop)
  (C_complete : ℕ → Prop)
  (A_can_complete_in_10_days : A_complete 10)
  (B_can_complete_in_20_days : B_complete 20)
  (C_can_complete_in_30_days : C_complete 30)
  (A_leaves_5_days_before_completion : ∀ x : ℕ, x ≥ 5 → A_complete (x - 5))
  (C_leaves_3_days_before_completion : ∀ x : ℕ, x ≥ 3 → C_complete (x - 3)) :
  ∃ x : ℕ, x = 8 := sorry

end work_completed_in_8_days_l20_20959


namespace unique_integer_for_mod2027_l20_20544

-- Define a monic polynomial P of degree 2023 satisfying the given functional equation.
noncomputable def P (x : ℚ) : ℚ := sorry

-- Assume and state the main functional condition of P.
axiom monic_P : ∀ (k : ℚ), (1 ≤ k ∧ k ≤ 2023) → P(k) = k^2023 * P(1 - 1/k)

-- stating the proof problem
theorem unique_integer_for_mod2027 (n : ℤ) : 
  P(-1) = 0 → 
  (0 ≤ n ∧ n < 2027) → 
  2027 ∣ n :=
by
  intro h
  intro
  exact Int.dvd_of_eq_zero (by
    rw [h, mul_zero, zero_sub, zero_add]
    refl)

end unique_integer_for_mod2027_l20_20544


namespace infinite_primes_4k_plus_1_l20_20193

theorem infinite_primes_4k_plus_1 :
  ∃ (infinitely_many : ∀ n : ℕ, ∃ p : ℕ, prime p ∧ p = 4 * n + 1) :=
sorry

end infinite_primes_4k_plus_1_l20_20193


namespace decimal_150th_digit_l20_20884

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20884


namespace digit_150_of_1_div_13_l20_20803

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20803


namespace area_of_trapezoid_l20_20457

-- Define the conditions specific to our problem
variables (m n : ℝ)
variable (ABCD : Type)
variable [trapezoid ABCD]

-- Define the properties of the trapezoid ABCD
variable (CD : ABCD → ℝ)
variable (midpoint_distance : ABCD → ℝ)

-- Assume the given conditions
axiom CD_length : ∀ t : ABCD, CD t = m
axiom midpoint_to_CD : ∀ t : ABCD, midpoint_distance t = n

-- Define the area of the trapezoid
def trapezoid_area (t : ABCD) : ℝ :=
  CD t * midpoint_distance t

-- The goal is to prove the area of the trapezoid equals m * n
theorem area_of_trapezoid (t : ABCD) :
  trapezoid_area t = m * n := by
  sorry

end area_of_trapezoid_l20_20457


namespace cubic_difference_pos_l20_20427

theorem cubic_difference_pos {a b : ℝ} (h : a > b) : a^3 - b^3 > 0 :=
sorry

end cubic_difference_pos_l20_20427


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20734

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20734


namespace hypotenuse_length_l20_20167

theorem hypotenuse_length (x y : ℝ) (h1 : y = 3 * x - 3) (h2 : 1 / 2 * x * y = 84) : (x^2 + y^2 = 505) :=
by
  -- Definitions by conditions
  have h_area : 1 / 2 * x * (3 * x - 3) = 84 := by rw h1; exact h2
  -- Sorry to skip the proof
  sorry

end hypotenuse_length_l20_20167


namespace max_product_AF_BF_l20_20142

theorem max_product_AF_BF (x y a : ℝ) (F : ℝ × ℝ) (A B : ℝ × ℝ)
  (h1 : a > 0) 
  (h2 : a ≠ 2) 
  (h3 : (A.1^2 / a^2 + A.2^2 / 16 = 1)) 
  (h4 : (B.1^2 / a^2 + B.2^2 / 16 = 1)) 
  (h5 : dist A B = 3) 
  (h6 : dist (A, F) * dist (B, F) = x)
  : a = 8 / 3 ∨ a = real.sqrt 3 := 
  sorry

end max_product_AF_BF_l20_20142


namespace integral_value_l20_20001

theorem integral_value (a : ℝ) (h : 3 * a * (3 / 36) = 1 / 2) :
  ∫ x in 1..a, (1 / x + real.sqrt (2 * x - x^2)) = real.log 2 + real.pi / 4 :=
by
  sorry

end integral_value_l20_20001


namespace inequality_holds_l20_20370

theorem inequality_holds (a : ℝ) : 
  (∀ x ∈ set.Icc (-2 : ℝ) 1, a * x ^ 3 - x ^ 2 + 4 * x + 3 ≥ 0) ↔ (-6 ≤ a ∧ a ≤ -2) :=
sorry

end inequality_holds_l20_20370


namespace one_div_thirteen_150th_digit_l20_20743

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20743


namespace decimal_150th_digit_of_1_div_13_l20_20706

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20706


namespace det_A_is_neg9_l20_20998

noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ := ![![-7, 5], ![6, -3]]

theorem det_A_is_neg9 : Matrix.det A = -9 := 
by 
  sorry

end det_A_is_neg9_l20_20998


namespace number_of_non_empty_proper_subsets_of_A_l20_20018

noncomputable def A : Set ℤ := { x : ℤ | -1 < x ∧ x ≤ 2 }

theorem number_of_non_empty_proper_subsets_of_A : 
  (∃ (A : Set ℤ), A = { x : ℤ | -1 < x ∧ x ≤ 2 }) → 
  ∃ (n : ℕ), n = 6 := by
  sorry

end number_of_non_empty_proper_subsets_of_A_l20_20018


namespace kirill_height_l20_20473

theorem kirill_height (K B : ℕ) (h1 : K = B - 14) (h2 : K + B = 112) : K = 49 :=
by
  sorry

end kirill_height_l20_20473


namespace max_tiles_on_floor_l20_20523

def tile_size : ℕ × ℕ := (20, 30)
def floor_size : ℕ × ℕ := (100, 150)

theorem max_tiles_on_floor (tile : ℕ × ℕ) (floor : ℕ × ℕ) (no_overlap : ℕ × ℕ → ℕ × ℕ → Prop) :
  tile = tile_size →
  floor = floor_size →
  (∀ t₁ t₂, t₁ ≠ t₂ → no_overlap t₁ t₂) →
  (∀ t₁, t₁.1 ≤ floor.1 ∧ t₁.2 ≤ floor.2) →
  (∀ t₀, no_overlap t₀ t₀) →
  ∃ n, n = 25 
:= by
  intros
  use 25
  sorry

end max_tiles_on_floor_l20_20523


namespace max_power_speed_l20_20562

variables (C S ρ v₀ v : ℝ)

def force (C S ρ v₀ v : ℝ) : ℝ :=
  (C * S * ρ * (v₀ - v)^2) / 2

def power (C S ρ v₀ v : ℝ) : ℝ :=
  (C * S * ρ / 2) * v * (v₀^2 - 2 * v₀ * v + v^2)

theorem max_power_speed (C S ρ v₀ : ℝ) (hρ : ρ > 0) (hC : C > 0) (hS : S > 0) :
  ∃ v : ℝ, power C S ρ v₀ v = (C * S * ρ / 2) * (v₀^2 * (v₀/3) - 2 * v₀ * (v₀/3)^2 + (v₀/3)^3) := 
  sorry

end max_power_speed_l20_20562


namespace number_of_arrangements_l20_20227

-- Define the conditions
inductive Student
| GirlA : Student
| GirlB : Student
| Boy1 : Student
| Boy2 : Student
| Boy3 : Student

-- Main theorem statement based on conditions provided
theorem number_of_arrangements : 
  let students := [Student.GirlA, Student.GirlB, Student.Boy1, Student.Boy2, Student.Boy3] in
  -- condition: Girl A does not stand at either end
  -- assumption that exactly 2 boys must stand next to each other
  ∃ (arrangements : Finset (List Student)),
    (∀ l ∈ arrangements, l.length = 5) ∧
    (∀ l ∈ arrangements, l.head ≠ some Student.GirlA) ∧
    (∀ l ∈ arrangements, l.last ≠ some Student.GirlA) ∧
    (∀ l ∈ arrangements, 
      let boy_positions := l.enum.filter (λ ⟨_, x⟩, x ≠ Student.GirlA ∧ x ≠ Student.GirlB) in
      boy_positions.length = 3 ∧
      ∃ (p1 p2 p3 : ℕ), 
        boy_positions[p1] = boy_positions[p2] ∧ 
        boy_positions[p2] ∼ boy_positions[p3]) ∧
    arrangements.card = 48 :=
sorry

end number_of_arrangements_l20_20227


namespace decimal_150th_digit_l20_20887

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20887


namespace digit_150_of_decimal_1_div_13_l20_20667

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20667


namespace limit_sequence_a_l20_20078

noncomputable 
def sequence_a (n : ℕ) : ℝ :=
  (finset.range n).prod (λ k, (2 * k + 1 : ℕ) / (2 * (k + 1) : ℕ))

theorem limit_sequence_a : 
  filter.tendsto sequence_a filter.at_top (nhds 0) :=
sorry

end limit_sequence_a_l20_20078


namespace find_a_l20_20043

theorem find_a (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  (∀ x : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) → 
    a^(2*x) + 2*a^x - 9 ≤ 6) ∧ 
  (∃ x : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ 
    a^(2*x) + 2*a^x - 9 = 6) → 
  a = 3 ∨ a = 1 / 3 :=
sorry

end find_a_l20_20043


namespace nth_term_arithmetic_seq_l20_20912

theorem nth_term_arithmetic_seq (a b n t count : ℕ) (h1 : count = 25) (h2 : a = 3) (h3 : b = 75) (h4 : n = 8) :
    t = a + (n - 1) * ((b - a) / (count - 1)) → t = 24 :=
by
  intros
  sorry

end nth_term_arithmetic_seq_l20_20912


namespace ratio_of_areas_l20_20202
open Real

-- Define the height of an equilateral triangle and side length of the inscribed square in the triangle
def equilateral_triangle_height (s : ℝ) : ℝ := (sqrt 3 * s) / 2
def inscribed_square_side_in_triangle (s : ℝ) : ℝ := s * sqrt 3 / 6

-- Define the area of the inscribed square in the equilateral triangle and the outer square
def area_of_inscribed_square_in_triangle (s : ℝ) : ℝ := (inscribed_square_side_in_triangle s) ^ 2
def area_of_inscribed_square_in_square (s : ℝ) : ℝ := s ^ 2

-- The ratio of the areas
theorem ratio_of_areas (s : ℝ) (hs : s ≠ 0) : 
  (area_of_inscribed_square_in_triangle s) / (area_of_inscribed_square_in_square s) = 1 / 12 := 
by
  sorry

end ratio_of_areas_l20_20202


namespace range_of_x_range_of_a_l20_20094

-- Definitions of the conditions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- Part (1)
theorem range_of_x (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
by sorry

-- Part (2)
theorem range_of_a (a : ℝ) (h : ∀ x, ¬ (p x a) → ¬ (q x)) : 1 < a ∧ a ≤ 2 :=
by sorry

end range_of_x_range_of_a_l20_20094


namespace digit_150_of_1_div_13_l20_20804

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20804


namespace digit_150_of_1_div_13_l20_20793

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20793


namespace one_div_thirteen_150th_digit_l20_20826

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20826


namespace y1_increasing_on_0_1_l20_20983

noncomputable def y1 (x : ℝ) : ℝ := |x|
noncomputable def y2 (x : ℝ) : ℝ := 3 - x
noncomputable def y3 (x : ℝ) : ℝ := 1 / x
noncomputable def y4 (x : ℝ) : ℝ := -x^2 + 4

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem y1_increasing_on_0_1 :
  is_increasing_on y1 0 1 ∧
  ¬ is_increasing_on y2 0 1 ∧
  ¬ is_increasing_on y3 0 1 ∧
  ¬ is_increasing_on y4 0 1 :=
by
  sorry

end y1_increasing_on_0_1_l20_20983


namespace area_of_triangle_given_parallelogram_l20_20116

variables {A B C D E F : Type} [euclidean_space A B C D E F]

def is_parallelogram (A B C D : Type) [euclidean_space A B C D] : Prop :=
  midpoint A C = midpoint B D

def midpoint (x y : A) : A :=
  (x + y) / 2

def area (x y z : A) : ℝ :=
  -- some formula to calculate the area of a triangle given vertices x, y, z
  sorry

theorem area_of_triangle_given_parallelogram :
  is_parallelogram A B C D →
  midpoint E B = midpoint B C →
  midpoint F C = midpoint C D →
  area A E F = (3 / 8) * area A B C D :=
by
  sorry

end area_of_triangle_given_parallelogram_l20_20116


namespace probability_intersection_l20_20430

variables (A B : Type → Prop)

-- Assuming we have a measure space (probability) P
variables {P : Type → Prop}

-- Given probabilities
def p_A := 0.65
def p_B := 0.55
def p_Ac_Bc := 0.20

-- The theorem to be proven
theorem probability_intersection :
  (p_A + p_B - (1 - p_Ac_Bc) = 0.40) :=
by
  sorry

end probability_intersection_l20_20430


namespace john_annual_patients_l20_20466

-- Definitions for the various conditions
def first_hospital_patients_per_day := 20
def second_hospital_patients_per_day := first_hospital_patients_per_day + (first_hospital_patients_per_day * 20 / 100)
def third_hospital_patients_per_day := first_hospital_patients_per_day + (first_hospital_patients_per_day * 15 / 100)
def total_patients_per_day := first_hospital_patients_per_day + second_hospital_patients_per_day + third_hospital_patients_per_day
def workdays_per_week := 5
def total_patients_per_week := total_patients_per_day * workdays_per_week
def working_weeks_per_year := 50 - 2 -- considering 2 weeks of vacation
def total_patients_per_year := total_patients_per_week * working_weeks_per_year

-- The statement to prove
theorem john_annual_patients : total_patients_per_year = 16080 := by
  sorry

end john_annual_patients_l20_20466


namespace digit_150_of_one_thirteenth_l20_20830

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20830


namespace integral_eval_eq_13_l20_20996

noncomputable def integral_value : ℝ :=
  ∫ x in 1..sqrt(3), (x^(2 * x^2 + 1) + real.log (x^(2 * x^(2 * x^2 + 1))))

theorem integral_eval_eq_13 : integral_value = 13 :=
sorry

end integral_eval_eq_13_l20_20996


namespace exists_unique_circle_l20_20591

structure Circle := (center : ℝ × ℝ) (radius : ℝ)

def diametrically_opposite_points (C : Circle) (P : ℝ × ℝ) : Prop :=
  let (cx, cy) := C.center
  let (px, py) := P
  (px - cx) ^ 2 + (py - cy) ^ 2 = (C.radius ^ 2)

def intersects_at_diametrically_opposite_points (K A : Circle) : Prop :=
  ∃ P₁ P₂ : ℝ × ℝ, diametrically_opposite_points A P₁ ∧ diametrically_opposite_points A P₂ ∧
  P₁ ≠ P₂ ∧ diametrically_opposite_points K P₁ ∧ diametrically_opposite_points K P₂

theorem exists_unique_circle (A B C : Circle) :
  ∃! K : Circle, intersects_at_diametrically_opposite_points K A ∧
  intersects_at_diametrically_opposite_points K B ∧
  intersects_at_diametrically_opposite_points K C := sorry

end exists_unique_circle_l20_20591


namespace sqrt_sqr_l20_20536

theorem sqrt_sqr (x : ℝ) (hx : 0 ≤ x) : (Real.sqrt x) ^ 2 = x := 
by sorry

example : (Real.sqrt 3) ^ 2 = 3 := 
by apply sqrt_sqr; linarith

end sqrt_sqr_l20_20536


namespace calculate_money_lacking_l20_20422

noncomputable def cost_mp3 := 135
noncomputable def cost_cd := 25
noncomputable def cost_headphones := 50
noncomputable def cost_case := 30
noncomputable def savings := 55
noncomputable def father_contribution := 20

def total_cost := cost_mp3 + cost_cd + cost_headphones + cost_case
def total_money_available := savings + father_contribution
def money_lacking := total_cost - total_money_available

theorem calculate_money_lacking : money_lacking = 165 :=
by
  sorry

end calculate_money_lacking_l20_20422


namespace contrapositive_sin_l20_20930

theorem contrapositive_sin (x y : ℝ) : (¬ (sin x = sin y) → ¬(x = y)) :=
by
  -- Placeholder for the proof. The statement will be proven true.
  sorry

end contrapositive_sin_l20_20930


namespace parallelogram_area_fraction_l20_20446

def coord1 := (3, 3)
def coord2 := (6, 5)
def coord3 := (3, 7)
def coord4 := (0, 5)

def grid_size := 8

/-- Given an 8 by 8 grid of points, a shaded parallelogram is formed by connecting four points at coordinates
(3,3), (6,5), (3,7), and (0,5). Prove that the fraction of the larger square's area that is inside the shaded parallelogram 
is 3/16. -/
theorem parallelogram_area_fraction :
  let area_parallelogram := 1 / 2 * |(3 * 5 + 6 * 7 + 3 * 5 + 0 * 3) - (3 * 6 + 5 * 3 + 7 * 0 + 5 * 3)|,
      area_square := (grid_size:ℕ)^2
  in area_parallelogram / area_square = 3 / 16 :=
by sorry

end parallelogram_area_fraction_l20_20446


namespace range_of_t_l20_20415

open Real

variables (e1 e2 : ℝ^3)
variables (t : ℝ)

-- Conditions
axiom e1_norm : ∥e1∥ = 2
axiom e2_norm : ∥e2∥ = 1
axiom angle_e1_e2 : ∀ (a : ℝ),
  a ≠ 0 → 
  cos (a • e1 e2) = cos (pi / 3)
axiom obtuse_angle_condition :
  (2 * t • e1 + 7 • e2) • (e1 + t • e2) < 0

-- Theorem to prove
theorem range_of_t : 
  (-7 < t ∧ t < -Real.sqrt 14 / 2) ∨ (-Real.sqrt 14 / 2 < t ∧ t < -1 / 2) :=
sorry

end range_of_t_l20_20415


namespace quadrilateral_angle_and_volume_l20_20283

-- Define the quadrilateral conditions and calculations
def quadrilateral_properties :=
  let height := 3
  let side_length := 2
  let center_projection := (side_length / 2, side_length / 2)
  let midpoint_SC := (height / 2) in
  sorry -- include properties in a more detailed implementation if necessary

-- The proof problem with the conditions
theorem quadrilateral_angle_and_volume :
  (∀ S A B C D K M N : Point,
    -- Given conditions
    (height S A B C D = 3) ∧
    (is_square A B C D) ∧
    (side_length A B = 2) ∧
    (projection S center_of_square A B C) ∧
    (midpoint K S C) ∧
    (plane_intersects M N :=
      plane_through (A, K) ∧
      intersects_at M (S, B) ∧
      intersects_at N (S, D)) →
    -- Proof goals
    (sin_angle (A, K) (plane S B C) = (2 * sqrt 30) / 15) ∧
    (volume_ratio (S, M, K, N) (S, B, C, D) = (dist S M * dist S K * dist S N) / (dist S B * dist S C * dist S D)) ∧
    ((midpoint M S B) →
    volume (S, A, M, K, N) = 3 / 2)) :=
begin
  sorry
end

end quadrilateral_angle_and_volume_l20_20283


namespace solve_equation_l20_20138

theorem solve_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 0) :
  (2 / (x - 1) - (x + 2) / (x * (x - 1)) = 0) ↔ x = 2 :=
by
  sorry

end solve_equation_l20_20138


namespace digit_150_of_one_thirteenth_l20_20839

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20839


namespace hyperbola_statements_true_l20_20382

-- Definitions and theorem statement based on conditions
def equilateralHyperbola := ∃ t : ℝ, t > 0 ∧ (∀ (x y : ℝ), x^2 / t^2 - y^2 / t^2 = 1)

variables (F : ℝ × ℝ) (A B D : ℝ × ℝ)
variable length_real_axis : ℝ

-- Hypotheses translated from the conditions
hypothesis H1 : equilateralHyperbola F
hypothesis H2 : length_real_axis = 2 * real.sqrt 2
hypothesis H3 : F = (2, 0)
hypothesis H4 : B = (D.1, -D.2)

-- Theorem statement
theorem hyperbola_statements_true
  (H1 : equilateralHyperbola F)
  (H2 : length_real_axis = 2 * real.sqrt 2)
  (H3 : F = (2, 0))
  (H4 : B = (D.1, -D.2)) :
  (∃ t : ℝ, t = real.sqrt 2 ∧ ∀ (x y : ℝ), x^2 / 2 - y^2 / 2 = 1) ∧ -- A is true
  (∀ (m : ℝ), m ≠ 0 → m = 2 → |(A.1 - B.1 + A.2 - B.2)| = 10 * real.sqrt 2 / 3) ∧ -- B is true
  (¬(∃ (l : line), (A.1 + F.1) / 2 = l.A)) ∧ -- C is false
  (∀ (m : ℝ), m ≠ 0 → P = (1,0) ∧ D.1 = m * D.2 + 2) := -- D is true
sorry -- Proof not required

end hyperbola_statements_true_l20_20382


namespace JacobProof_l20_20073

def JacobLadders : Prop :=
  let costPerRung : ℤ := 2
  let costPer50RungLadder : ℤ := 50 * costPerRung
  let num50RungLadders : ℤ := 10
  let totalPayment : ℤ := 3400
  let cost1 : ℤ := num50RungLadders * costPer50RungLadder
  let remainingAmount : ℤ := totalPayment - cost1
  let numRungs20Ladders : ℤ := remainingAmount / costPerRung
  numRungs20Ladders = 1200

theorem JacobProof : JacobLadders := by
  sorry

end JacobProof_l20_20073


namespace decimal_150th_digit_l20_20649

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20649


namespace length_of_AB_l20_20068

noncomputable theory

variables {AB CD : ℝ}

def area_ratio (AB CD : ℝ) : Prop := (4 * CD = AB)

def sum_length (AB CD : ℝ) : Prop := (AB + CD = 180)

theorem length_of_AB (h1 : area_ratio AB CD) (h2 : sum_length AB CD) : AB = 144 :=
by
  -- by skipping the proof step.
  sorry

end length_of_AB_l20_20068


namespace decimal_150th_digit_l20_20886

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20886


namespace six_letter_great_words_count_l20_20309

def is_great_word (w : List Char) : Prop :=
  ∀ (i : ℕ), i < w.length - 1 → (
    (w[i] = 'D' → ¬ (w[i + 1] = 'E'))
    ∧ (w[i] = 'E' → ¬ (w[i + 1] = 'F'))
    ∧ (w[i] = 'F' → ¬ (w[i + 1] = 'G'))
    ∧ (w[i] = 'G' → ¬ (w[i + 1] = 'D'))
  )

def count_six_letter_great_words : ℕ :=
  4 * 3^5

theorem six_letter_great_words_count :
  ∃ w : List Char, w.length = 6 ∧ is_great_word w ∧ count_six_letter_great_words = 972 := 
sorry

end six_letter_great_words_count_l20_20309


namespace bead_arrangement_probability_l20_20208

def total_beads := 6
def red_beads := 2
def white_beads := 2
def blue_beads := 2

def total_arrangements : ℕ := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

def valid_arrangements : ℕ := 6  -- Based on valid patterns RWBRWB, RWBWRB, and all other permutations for each starting color

def probability_valid := valid_arrangements / total_arrangements

theorem bead_arrangement_probability : probability_valid = 1 / 15 :=
  by
  -- The context and details of the solution steps are omitted as they are not included in the Lean theorem statement.
  -- This statement will skip the proof
  sorry

end bead_arrangement_probability_l20_20208


namespace one_div_thirteen_150th_digit_l20_20790

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20790


namespace policeman_catches_thief_l20_20218

/- Define the speeds and initial conditions -/
def speed_thief : ℝ := 1
def speed_policeman : ℝ := 2 * speed_thief

/- Define the position functions for thief and policeman -/
def position_thief (t : ℝ) : ℝ := t
def position_policeman (k : ℕ) : ℝ :=
  let rec_aux (n : ℕ) (x : ℝ) : ℝ :=
    if n = 0 then x
    else rec_aux (n - 1) (6 * (-4)^(n - 1) + x)
  rec_aux k 0

/- Statement of the theorem to prove -/
theorem policeman_catches_thief :
  ∃ t : ℝ, ∃ k : ℕ, position_thief t = position_policeman k := 
sorry

end policeman_catches_thief_l20_20218


namespace find_x1_l20_20391

variable (x1 x2 x3 : ℝ)

theorem find_x1 (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 0.8)
    (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1 / 3) : 
    x1 = 3 / 4 :=
  sorry

end find_x1_l20_20391


namespace value_of_x_plus_y_l20_20316

noncomputable def sequence := list ℝ

theorem value_of_x_plus_y {x y : ℝ} (r : ℝ) (seq : sequence) (h1 : r = 1 / 4)
  (h2 : seq = [4096, 1024, 256, 64, x, y, 4, 1, 1/4]) : x + y = 20 :=
  by
  sorry

end value_of_x_plus_y_l20_20316


namespace surface_area_of_given_cube_l20_20174

-- Define the edge length condition
def edge_length_of_cube (sum_edge_lengths : ℕ) :=
  sum_edge_lengths / 12

-- Define the surface area of a cube given an edge length
def surface_area_of_cube (edge_length : ℕ) :=
  6 * (edge_length * edge_length)

-- State the theorem
theorem surface_area_of_given_cube : 
  edge_length_of_cube 36 = 3 ∧ surface_area_of_cube 3 = 54 :=
by
  -- We leave the proof as an exercise.
  sorry

end surface_area_of_given_cube_l20_20174


namespace maximum_t_value_l20_20378

noncomputable def a : ℕ → ℕ
| 0       := 0  -- we define a_0 but it won't be used
| 1       := 1
| (n+2)   := 3 * a (n+1) + 4

theorem maximum_t_value :
  ∃ (t : ℝ), t = 10 / 9 ∧ ∀ n : ℕ, n > 0 → n * (2 * n + 1) ≥ t * (a n + 2) := 
  by 
    sorry

end maximum_t_value_l20_20378


namespace popsicle_sticks_left_l20_20302

-- Defining the conditions
def total_money : ℕ := 10
def cost_of_molds : ℕ := 3
def cost_of_sticks : ℕ := 1
def cost_of_juice_bottle : ℕ := 2
def popsicles_per_bottle : ℕ := 20
def initial_sticks : ℕ := 100

-- Statement of the problem
theorem popsicle_sticks_left : 
  let remaining_money := total_money - cost_of_molds - cost_of_sticks
  let bottles_of_juice := remaining_money / cost_of_juice_bottle
  let total_popsicles := bottles_of_juice * popsicles_per_bottle
  let sticks_left := initial_sticks - total_popsicles
  sticks_left = 40 := by
  sorry

end popsicle_sticks_left_l20_20302


namespace distance_point_to_plane_l20_20443

def point := (2, 1, -3 : ℝ)
def plane_A := 1
def plane_B := 2
def plane_C := 3
def plane_D := 3

theorem distance_point_to_plane :
  (let d := (| plane_A * 2 + plane_B * 1 + plane_C * -3 + plane_D |) / (Real.sqrt (plane_A^2 + plane_B^2 + plane_C^2))
   in d = (Real.sqrt 14) / 7) :=
by
  sorry

end distance_point_to_plane_l20_20443


namespace digit_150_of_decimal_1_div_13_l20_20662

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20662


namespace digit_150_after_decimal_of_one_thirteenth_l20_20864

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20864


namespace angle_bisectors_intersection_on_CD_l20_20495

noncomputable def cyclic_quadrilateral (A B C D : Point) : Prop :=
∃ O : Point, Circle O O.r (mkLine A B) ∧ Circle O O.r (mkLine B C) ∧ Circle O O.r (mkLine C D) ∧ Circle O O.r (mkLine D A)

def condition1 (A B C D : Point) (DC AD BC : ℝ) : Prop :=
DC = AD + BC

theorem angle_bisectors_intersection_on_CD (A B C D : Point) (DC AD BC : ℝ) (h_cyclic: cyclic_quadrilateral A B C D) (h_cond: condition1 A B C D DC AD BC) :
  ∃ E : Point, line E C D ∧ ∃ F : Point, angle_bisector E A ∧ angle_bisector E B := 
sorry


end angle_bisectors_intersection_on_CD_l20_20495


namespace inheritance_amount_l20_20598

def federalTaxRate : ℝ := 0.25
def stateTaxRate : ℝ := 0.15
def totalTaxPaid : ℝ := 16500

theorem inheritance_amount :
  ∃ x : ℝ, (federalTaxRate * x + stateTaxRate * (1 - federalTaxRate) * x = totalTaxPaid) → x = 45500 := by
  sorry

end inheritance_amount_l20_20598


namespace one_div_thirteen_150th_digit_l20_20788

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20788


namespace solution_set_of_inequality_l20_20582

theorem solution_set_of_inequality : 
  { x : ℝ | (3 - 2 * x) * (x + 1) ≤ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 / 2 } :=
sorry

end solution_set_of_inequality_l20_20582


namespace largest_negative_a_l20_20919

theorem largest_negative_a {a : ℝ} :
  (∀ x : ℝ, x ∈ Set.Ioo (-3 * Real.pi) (-5 * Real.pi / 2) →
    ((Real.cbrt (Real.cos x) - Real.cbrt (Real.sin x)) / (Real.cbrt (Real.tan x) - Real.cbrt (Real.tan x)) > a)) →
  a = -0.45 :=
sorry

end largest_negative_a_l20_20919


namespace base_nine_to_mod_five_l20_20552

-- Define the base-nine number N
def N : ℕ := 2 * 9^10 + 7 * 9^9 + 0 * 9^8 + 0 * 9^7 + 6 * 9^6 + 0 * 9^5 + 0 * 9^4 + 0 * 9^3 + 0 * 9^2 + 5 * 9^1 + 2 * 9^0

-- Theorem statement
theorem base_nine_to_mod_five : N % 5 = 3 :=
by
  sorry

end base_nine_to_mod_five_l20_20552


namespace digit_150_of_1_over_13_is_3_l20_20771

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20771


namespace complementSetM_l20_20021

open Set Real

-- The universal set U is the set of all real numbers
def universalSet : Set ℝ := univ

-- The set M is defined as {x | |x - 1| ≤ 2}
def setM : Set ℝ := {x : ℝ | |x - 1| ≤ 2}

-- We need to prove that the complement of M with respect to U is {x | x < -1 ∨ x > 3}
theorem complementSetM :
  (universalSet \ setM) = {x : ℝ | x < -1 ∨ x > 3} :=
by
  sorry

end complementSetM_l20_20021


namespace geometry_problem_l20_20459

theorem geometry_problem
  {A B C X Y : Type}
  [has_angle A B C 105]
  [has_angle C B X 70]
  [has_distance_eq B X B C]
  [has_distance_eq B Y B A]
  : AX + AY ≥ CY :=
sorry

end geometry_problem_l20_20459


namespace unique_ordered_triple_l20_20088

theorem unique_ordered_triple (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ab : Nat.lcm a b = 500) (h_bc : Nat.lcm b c = 2000) (h_ca : Nat.lcm c a = 2000) :
  (a = 100 ∧ b = 2000 ∧ c = 2000) :=
by
  sorry

end unique_ordered_triple_l20_20088


namespace one_over_thirteen_150th_digit_l20_20677

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20677


namespace decimal_150th_digit_of_1_div_13_l20_20694

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20694


namespace digit_150_of_decimal_1_div_13_l20_20670

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20670


namespace fraction_is_meaningful_l20_20578

theorem fraction_is_meaningful (x : ℝ) : x ≠ 1 ↔ ∃ y : ℝ, y = 8 / (x - 1) :=
by
  sorry

end fraction_is_meaningful_l20_20578


namespace no_solution_for_ab_ba_l20_20955

theorem no_solution_for_ab_ba (a b x : ℕ)
  (ab ba : ℕ)
  (h_ab : ab = 10 * a + b)
  (h_ba : ba = 10 * b + a) :
  (ab^x - 2 = ba^x - 7) → false :=
by
  sorry

end no_solution_for_ab_ba_l20_20955


namespace one_thirteenth_150th_digit_l20_20633

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20633


namespace vertical_asymptotes_A_plus_B_plus_C_l20_20165

noncomputable def A : ℤ := -6
noncomputable def B : ℤ := 5
noncomputable def C : ℤ := 12

theorem vertical_asymptotes_A_plus_B_plus_C :
  (x + 1) * (x - 3) * (x - 4) = x^3 + A*x^2 + B*x + C ∧ A + B + C = 11 := by
  sorry

end vertical_asymptotes_A_plus_B_plus_C_l20_20165


namespace decimal_150th_digit_l20_20879

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20879


namespace one_over_thirteen_150th_digit_l20_20681

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20681


namespace remainder_3_pow_2n_plus_8_l20_20488

theorem remainder_3_pow_2n_plus_8 (n : Nat) : (3 ^ (2 * n) + 8) % 8 = 1 := by
  sorry

end remainder_3_pow_2n_plus_8_l20_20488


namespace terminal_side_in_third_quadrant_l20_20462

theorem terminal_side_in_third_quadrant (α : ℝ) (tan_alpha cos_alpha : ℝ) 
  (h1 : Point (tan α) (cos α) ∈ third_quadrant) : 
  terminal_side_angle α ∈ third_quadrant :=
sorry

end terminal_side_in_third_quadrant_l20_20462


namespace pumpkins_eaten_l20_20531

theorem pumpkins_eaten (initial: ℕ) (left: ℕ) (eaten: ℕ) (h1 : initial = 43) (h2 : left = 20) : eaten = 23 :=
by {
  -- We are skipping the proof as per the requirement
  sorry
}

end pumpkins_eaten_l20_20531


namespace sailboat_speed_max_power_l20_20563

-- Define the parameters
variables (C S ρ v0 : ℝ)

-- Define the force function
def force (v : ℝ) : ℝ :=
  (C * S * ρ * (v0 - v) ^ 2) / 2

-- Define the power function
def power (v : ℝ) : ℝ :=
  (force C S ρ v0 v) * v

-- Define the statement to be proven
theorem sailboat_speed_max_power : ∃ v : ℝ, (power C S ρ v0 v = Term.max (power C S ρ v0)) ∧ v = v0 / 3 :=
by
  sorry

end sailboat_speed_max_power_l20_20563


namespace decimal_1_div_13_150th_digit_is_3_l20_20853

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20853


namespace decimal_1_div_13_150th_digit_is_3_l20_20857

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20857


namespace correct_propositions_l20_20935

theorem correct_propositions (props: Finset ℕ) :
  -- Conditions
  (∀ x, x ∈ props → x = 1 ∨ x = 4) ∧
  -- Propositions based on conditions
  (∀ x, x = 1 → (min_positive_period (λ x, cos x ^ 2 - 1 / 2) = π)) ∧
  (∀ x, x = 2 → ∀ k, (terminal_side_angles_based_on_y_axis k ≠ 0)) ∧
  (∀ x, x = 3 → ¬ (graph_symmetric_about_point θ 4 2 (π / 6, 0))) ∧
  (∀ x, x = 4 → ∃ I, I = (-π/12, 5π/12) ∧ function_increasing_interval (λ x, 3 * sin (2*x - π/3)) I) ∧
  (∀ x, x = 5 → ¬ (graph_shift_eq (λ x, 4 * cos 2*x) (λ x, 4 * sin 2*x) (π / 4)))
⊢ props = {1, 4} := sorry

end correct_propositions_l20_20935


namespace probability_black_or_white_l20_20229

-- Defining the probabilities of drawing red and white balls
def prob_red : ℝ := 0.45
def prob_white : ℝ := 0.25

-- Defining the total probability
def total_prob : ℝ := 1.0

-- Define the probability of drawing a black or white ball
def prob_black_or_white : ℝ := total_prob - prob_red

-- The theorem stating the required proof
theorem probability_black_or_white : 
  prob_black_or_white = 0.55 := by
    sorry

end probability_black_or_white_l20_20229


namespace one_thirteenth_150th_digit_l20_20630

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20630


namespace digit_150_of_decimal_1_div_13_l20_20665

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20665


namespace power_sum_w_equals_w_minus_1_l20_20096

-- Let w be a complex number such that w^2 - w + 1 = 0
variable {w : ℂ}
variable (hw : w^2 - w + 1 = 0)

-- Statement: For w satisfying the given condition, the sum of specific powers of w equals w - 1
theorem power_sum_w_equals_w_minus_1 : 
  w^{98} + w^{99} + w^{100} + w^{101} + w^{102} = w - 1 := 
sorry

end power_sum_w_equals_w_minus_1_l20_20096


namespace T_conditions_T_cross_product_T_conditions_1_T_conditions_2_find_T_218_l20_20502

noncomputable def T : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ := sorry

theorem T_conditions (a b : ℝ) (v w : ℝ × ℝ × ℝ) :
  T (a • v.1 + b • w.1, a • v.2 + b • w.2, a • v.3 + b • w.3) = 
    (a • T v).1 + (b • T w).1, 
    (a • T v).2 + (b • T w).2, 
    (a • T v).3 + (b • T w).3 := sorry

theorem T_cross_product (v w : ℝ × ℝ × ℝ) :
  T (v.2 * w.3 - v.3 * w.2, v.3 * w.1 - v.1 * w.3, v.1 * w.2 - v.2 * w.1) = 
    (T v).2 * (T w).3 - (T v).3 * (T w).2,
    (T v).3 * (T w).1 - (T v).1 * (T w).3,
    (T v).1 * (T w).2 - (T v).2 * (T w).1 := sorry

theorem T_conditions_1 :
  T (4, 2, 6) = (1, 5, -2) := sorry

theorem T_conditions_2 :
  T (3, -3, 2) = (-3, 1, 4) := sorry

theorem find_T_218 :
  T (2, 1, 8) = ? -- to be filled with the computed vector := sorry

end T_conditions_T_cross_product_T_conditions_1_T_conditions_2_find_T_218_l20_20502


namespace probability_even_xy_l20_20047

open_locale big_operators

def x_set : finset ℕ := {1, 2, 3, 4, 7, 11}
def y_set : finset ℕ := {5, 6, 9, 13, 17}

def is_even (n : ℕ) : Prop := n % 2 = 0

theorem probability_even_xy : 
  ∑ x in x_set, ∑ y in y_set, if is_even (x * y) then 1 else 0 =
  6 * 5 * (7 / 15) := 
by  -- Note: This part is usually where the proof would begin.
-- The proof steps would involve showing the sum resulting in the desired probability
-- Since the exact steps are not considered here, we leave it as an unfinished proof.
  sorry

end probability_even_xy_l20_20047


namespace equal_values_difference_pi_l20_20478

variables {n : ℕ} (a : fin n → ℝ) (x x1 x2 : ℝ)

noncomputable def f (a : fin n → ℝ) (x : ℝ) : ℝ :=
  ∑ i in finset.range n, (cos (a i + x)) / 2^i

theorem equal_values_difference_pi (h : f a x1 = 0) (h2 : f a x2 = 0) :
  ∃ m : ℤ, x1 - x2 = m * π :=
sorry

end equal_values_difference_pi_l20_20478


namespace prove_triangle_inequality_l20_20496

def triangle_inequality (a b c a1 a2 b1 b2 c1 c2 : ℝ) : Prop := 
  a * a1 * a2 + b * b1 * b2 + c * c1 * c2 ≥ a * b * c

theorem prove_triangle_inequality 
  (a b c a1 a2 b1 b2 c1 c2 : ℝ)
  (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c)
  (h4: 0 ≤ a1) (h5: 0 ≤ a2) 
  (h6: 0 ≤ b1) (h7: 0 ≤ b2)
  (h8: 0 ≤ c1) (h9: 0 ≤ c2) : triangle_inequality a b c a1 a2 b1 b2 c1 c2 :=
sorry

end prove_triangle_inequality_l20_20496


namespace popsicle_sticks_left_l20_20306

/-- Danielle has $10 for supplies. She buys one set of molds for $3, 
a pack of 100 popsicle sticks for $1. Each bottle of juice makes 20 popsicles and costs $2.
Prove that the number of popsicle sticks Danielle will be left with after making as many popsicles as she can is 40. -/
theorem popsicle_sticks_left (initial_money : ℕ)
    (mold_cost : ℕ) (sticks_cost : ℕ) (initial_sticks : ℕ)
    (juice_cost : ℕ) (popsicles_per_bottle : ℕ)
    (final_sticks : ℕ) :
    initial_money = 10 →
    mold_cost = 3 → 
    sticks_cost = 1 → 
    initial_sticks = 100 →
    juice_cost = 2 →
    popsicles_per_bottle = 20 →
    final_sticks = initial_sticks - (popsicles_per_bottle * (initial_money - mold_cost - sticks_cost) / juice_cost) →
    final_sticks = 40 :=
by
  intros h_initial_money h_mold_cost h_sticks_cost h_initial_sticks h_juice_cost h_popsicles_per_bottle h_final_sticks
  rw [h_initial_money, h_mold_cost, h_sticks_cost, h_initial_sticks, h_juice_cost, h_popsicles_per_bottle] at h_final_sticks
  norm_num at h_final_sticks
  exact h_final_sticks

end popsicle_sticks_left_l20_20306


namespace one_thirteen_150th_digit_l20_20903

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20903


namespace sum_black_cells_even_l20_20150

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

end sum_black_cells_even_l20_20150


namespace PQ_bisects_BC_l20_20374

-- Define all the necessary points, lines, and properties from the conditions.
variables {A B C H D P Q M : Type*}
variables [metric_space A] [metric_space B] [metric_space C]
variables [orthocenter H A B C] [circumcircle Omega A B C]
variables [tangent_line la Omega A] [tangent_line lb Omega B] [tangent_line lc Omega C]
variables [intersection_point D lb lc]
variables [foot_perpendicular H la P] [foot_perpendicular H (line_segment A D) Q]
variables [midpoint M B C]

-- State the main theorem to be proved.
theorem PQ_bisects_BC : collinear P Q M := sorry

end PQ_bisects_BC_l20_20374


namespace prob_sum_is_five_prob_at_least_one_odd_prob_within_circle_l20_20122

-- Define the probability space for a die roll
def die_roll : Type := fin 6
-- Define the sample space for two independent die rolls
def sample_space : Type := die_roll × die_roll

-- Define the event that the sum of the two numbers is 5
def event_sum_is_five (x y : sample_space) : Prop :=
  (x.1.succ + x.2.succ = 5)

-- Define the event that at least one number is odd
def event_at_least_one_odd (x : sample_space) : Prop :=
  (x.1.succ % 2 = 1) ∨ (x.2.succ % 2 = 1)

-- Define the event that the point lies within the circle x^2 + y^2 = 15
def event_within_circle (x : sample_space) : Prop :=
  (x.1.succ * x.1.succ + x.2.succ * x.2.succ < 15)

-- The main theorem statements:
theorem prob_sum_is_five : 
  (fun f, f (λ x, event_sum_is_five x) sample_space) = 1 / 9 := sorry

theorem prob_at_least_one_odd : 
  (fun f, f (λ x, event_at_least_one_odd x) sample_space) = 3 / 4 := sorry

theorem prob_within_circle : 
  (fun f, f (λ x, event_within_circle x) sample_space) = 2 / 9 := sorry

end prob_sum_is_five_prob_at_least_one_odd_prob_within_circle_l20_20122


namespace one_thirteenth_150th_digit_l20_20639

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20639


namespace part1_part2_l20_20385

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part (1)
theorem part1 (a : ℝ) (h : a = 3) : (P 3)ᶜ ∩ Q = {x | -2 ≤ x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) : (∀ x, x ∈ P a → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P a) → 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end part1_part2_l20_20385


namespace one_div_thirteen_150th_digit_l20_20822

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20822


namespace one_over_thirteen_150th_digit_l20_20675

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20675


namespace tangent_line_polar_equation_at_point_l20_20015

noncomputable def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (Real.sqrt 2 * Real.cos t, Real.sqrt 2 * Real.sin t)

def point_on_curve := (1, 1)

theorem tangent_line_polar_equation_at_point :
  ∃ l : ℝ → ℝ → Prop,
    l 1 1 ∧
    (∀ x y, l x y ↔ x + y = 2) ∧
    (∀ ρ θ, l (ρ * Real.cos θ) (ρ * Real.sin θ) ↔ ρ * Real.sin (θ + π / 4) = Real.sqrt 2) :=
begin
  sorry
end

end tangent_line_polar_equation_at_point_l20_20015


namespace sum_of_zeros_l20_20498

def f (x : ℝ) : ℝ :=
if x < 0 then 3*x - 6 else -x/3 + 2

theorem sum_of_zeros : (∑ x in {x : ℝ | f x = 0}.to_finset, x) = 6 :=
by
  sorry

end sum_of_zeros_l20_20498


namespace investor_receives_7260_l20_20197

-- Define the initial conditions
def principal : ℝ := 6000
def annual_rate : ℝ := 0.10
def compoundings_per_year : ℝ := 1
def years : ℝ := 2

-- Define the compound interest formula
noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- State the theorem: The investor will receive $7260 after two years
theorem investor_receives_7260 : compound_interest principal annual_rate compoundings_per_year years = 7260 := by
  sorry

end investor_receives_7260_l20_20197


namespace mean_median_temperatures_l20_20574

open List

-- Define the list of temperatures
def temperatures : List ℝ := [78, 77, 79, 83, 85, 87, 86, 87, 84]

-- Define the mean function
def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

-- Define the function to find the median
def median (l : List ℝ) : ℝ :=
  let sorted := l.sort
  sorted.get! (l.length / 2)

-- The proof problem statement
theorem mean_median_temperatures (temps : List ℝ) (h : temps = temperatures) :
  mean temps = 82.8 ∧ median temps = 84 :=
by
  -- Place proof here
  sorry

end mean_median_temperatures_l20_20574


namespace log_base_sufficient_but_not_necessary_l20_20952

theorem log_base_sufficient_but_not_necessary (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) : 
  ∃ b : ℝ, ∃ a : ℝ, (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ log a b > 0 :=
sorry

end log_base_sufficient_but_not_necessary_l20_20952


namespace even_multiples_of_4_product_zero_count_l20_20338
open Complex Real -- Open the relevant namespaces for complex numbers and real numbers.
  
theorem even_multiples_of_4_product_zero_count :
  (∃ n ∈ Icc 1 2020, (∏ k in finset.range n, ((1 + exp (2 * π * I * k / n))^n + 1) = 0)) ↔ (252 : ℕ) :=
sorry

end even_multiples_of_4_product_zero_count_l20_20338


namespace min_value_l20_20212

theorem min_value (t : ℤ) : 
  let k := 9 * t + 8 in 
  (¬(9 ∣ (5 * (9 * t + 8) * (9 * 25 * t + 222))) → R_min = 10) := 
begin
  intros k hk,
  sorry
end

end min_value_l20_20212


namespace sum_of_digits_l20_20577

theorem sum_of_digits (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
                      (h_range : 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9)
                      (h_product : a * b * c * d = 810) :
  a + b + c + d = 23 := sorry

end sum_of_digits_l20_20577


namespace digit_150th_of_fraction_l20_20714

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20714


namespace find_integer_triples_l20_20323

theorem find_integer_triples (x y z : ℤ) :
  (x^3 + y^3 + z^3 - 3 * x * y * z = 2003) →
  (x, y, z) ∈ {(668, 668, 667) | (668, 667, 668) | (667, 668, 668)} :=
begin
  sorry
end

end find_integer_triples_l20_20323


namespace circumcircle_trapezoid_triang_l20_20276

noncomputable theory

open EuclideanGeometry

/-- Given a trapezoid ABCD inscribed in a circle with center O,
    and tangents PA and PB drawn from point P to the circle,
    prove that the circumcircle of triangle PAB passes through 
    the midpoint M of the base CD. -/
theorem circumcircle_trapezoid_triang (O P A B C D M : Point)
  (h1 : inscribed_in_circle ABCD O)
  (h2 : tangent PA (circumscribed_circle PAB))
  (h3 : tangent PB (circumscribed_circle PAB))
  (h4 : midpoint M CD)
  (h5 : is_trapezoid ABCD) :
  passes_through (circumscribed_circle PAB) M :=
sorry

end circumcircle_trapezoid_triang_l20_20276


namespace feed_puppies_days_l20_20530

theorem feed_puppies_days (puppies : ℕ) (total_portions : ℕ) (feed_times_per_day : ℕ) : 
  puppies = 7 → total_portions = 105 → feed_times_per_day = 3 → 
  total_portions / (puppies * feed_times_per_day) = 5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end feed_puppies_days_l20_20530


namespace find_integer_triples_l20_20324

theorem find_integer_triples (x y z : ℤ) :
  (x^3 + y^3 + z^3 - 3 * x * y * z = 2003) →
  (x, y, z) ∈ {(668, 668, 667) | (668, 667, 668) | (667, 668, 668)} :=
begin
  sorry
end

end find_integer_triples_l20_20324


namespace kirill_height_l20_20470

theorem kirill_height (B : ℕ) (h1 : ∃ B, B - 14 = kirill_height) (h2 : B + (B - 14) = 112) : kirill_height = 49 :=
sorry

end kirill_height_l20_20470


namespace minimum_distance_curve_line_l20_20482

noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x

def line (x : ℝ) : ℝ := x - 2

theorem minimum_distance_curve_line : 
  ∃ P : ℝ × ℝ, P.2 = curve P.1 ∧ 
  (∀ Q : ℝ × ℝ, Q.2 = curve Q.1 → dist P ⟨Q.1, line Q.1⟩ ≥ dist P ⟨P.1, line P.1⟩) ∧ 
  dist P ⟨P.1, line P.1⟩ = Real.sqrt 2 :=
begin
  sorry
end

end minimum_distance_curve_line_l20_20482


namespace average_age_combined_l20_20149

theorem average_age_combined (
  avg_age_RoomA : ℕ → ℕ,
  avg_age_RoomB : ℕ → ℕ
) (nA nB tA tB : ℕ)
  (h1 : nA = 7)
  (h2 : nB = 5)
  (h3 : avg_age_RoomA nA = 35) 
  (h4 : avg_age_RoomB nB = 30) 
  (h5 : tA = nA * avg_age_RoomA nA) 
  (h6 : tB = nB * avg_age_RoomB nB) :
  ((tA + tB) / (nA + nB) = 33)
  :=
  by sorry

end average_age_combined_l20_20149


namespace shaded_area_eq_l20_20236

noncomputable def radius_small : ℝ := 2
noncomputable def radius_large : ℝ := 3
noncomputable def area_shaded : ℝ := (5.65 / 3) * Real.pi - 2.9724 * Real.sqrt 5

theorem shaded_area_eq : 
  (∃ A B C : ℝ × ℝ, 
    ∀ (x : ℝ × ℝ), x ∈ { p | Real.norm (p.1 - C.1, p.2 - C.2) = radius_small } → 
    (Real.norm (x.1 - A.1, x.2 - A.2) = radius_large ∧ Real.norm (x.1 - B.1, x.2 - B.2) = radius_large) ∧ 
    Real.norm (A.1 - B.1, A.2 - B.2) = 2 * radius_small) → 
  ∃ R : ℝ, R = area_shaded := 
by {
  sorry
}

end shaded_area_eq_l20_20236


namespace decimal_150th_digit_l20_20648

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20648


namespace square_root_relation_cube_root_of_m_squared_plus_2_l20_20436

theorem square_root_relation (m : ℤ) (x : ℤ) 
(h₁ : x = (m - 3) ^ 2)
(h₂ : x = (m - 7) ^ 2) : 
x = 4 :=
by sorry

theorem cube_root_of_m_squared_plus_2 (m : ℤ)
(hm : m = 5) : 
Int.cbrt (m ^ 2 + 2) = 3 :=
by sorry

end square_root_relation_cube_root_of_m_squared_plus_2_l20_20436


namespace determine_a_b_l20_20045

variables {a b : ℝ}
def line_parametric (t : ℝ) := 
  (t + a, (b / 2) * t + 1)

def point_P := (0, 2)
def point_Q := (1, 3)

theorem determine_a_b : a = -1 ∧ b = 2 :=
sorry

end determine_a_b_l20_20045


namespace lara_has_largest_answer_l20_20467

/-- Define the final result for John, given his operations --/
def final_john (n : ℕ) : ℕ :=
  let add_three := n + 3
  let double := add_three * 2
  double - 4

/-- Define the final result for Lara, given her operations --/
def final_lara (n : ℕ) : ℕ :=
  let triple := n * 3
  let add_five := triple + 5
  add_five - 6

/-- Define the final result for Miguel, given his operations --/
def final_miguel (n : ℕ) : ℕ :=
  let double := n * 2
  let subtract_two := double - 2
  subtract_two + 2

/-- Main theorem to be proven --/
theorem lara_has_largest_answer :
  final_lara 12 > final_john 12 ∧ final_lara 12 > final_miguel 12 :=
by {
  sorry
}

end lara_has_largest_answer_l20_20467


namespace decimal_150th_digit_of_1_div_13_l20_20693

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20693


namespace problem1_problem2_l20_20364

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

-- Problem 1
theorem problem1 (a : ℝ) :
  (∀ x : ℝ, 0 < x → f a x > 0) ↔ (a ∈ set.Iic (-real.sqrt 5) ∪ set.Ioo (-real.sqrt 5) (real.sqrt 5)) :=
begin
  sorry
end

-- Problem 2
theorem problem2 (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, 1 ≤ x → x ≤ a → f a x ∈ set.Icc 1 a) → a = 2 :=
begin
  sorry
end

end problem1_problem2_l20_20364


namespace decimal_1_div_13_150th_digit_is_3_l20_20849

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20849


namespace decimal_150th_digit_of_1_div_13_l20_20697

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20697


namespace product_of_two_numbers_l20_20175

variable {x y : ℝ}

theorem product_of_two_numbers (h1 : x + y = 25) (h2 : x - y = 7) : x * y = 144 := by
  sorry

end product_of_two_numbers_l20_20175


namespace zeros_difference_l20_20295

-- Definitions based on the given conditions
def vertex : ℝ × ℝ := (3, -3)
def point_on_parabola : ℝ × ℝ := (5, 15)
def a : ℝ := 4.5
def zero1 : ℝ := 3 + Real.sqrt(2 / 3)
def zero2 : ℝ := 3 - Real.sqrt(2 / 3)

-- The proof problem
theorem zeros_difference : (vertex = (3, -3)) ∧ (point_on_parabola = (5, 15)) →
  a = 4.5 →
  zero1 - zero2 = 2 * Real.sqrt(2 / 3) :=
sorry

end zeros_difference_l20_20295


namespace one_div_thirteen_150th_digit_l20_20746

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20746


namespace one_div_thirteen_150th_digit_l20_20814

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20814


namespace Xiaogang_raised_arm_exceeds_head_l20_20933

theorem Xiaogang_raised_arm_exceeds_head :
  ∀ (height shadow_no_arm shadow_with_arm : ℝ),
    height = 1.7 → shadow_no_arm = 0.85 → shadow_with_arm = 1.1 →
    (height / shadow_no_arm) = ((shadow_with_arm - shadow_no_arm) * (height / shadow_no_arm)) →
    shadow_with_arm - shadow_no_arm = 0.25 →
    ((height / shadow_no_arm) * 0.25) = 0.5 :=
by
  intros height shadow_no_arm shadow_with_arm h_eq1 h_eq2 h_eq3 h_eq4 h_eq5
  sorry

end Xiaogang_raised_arm_exceeds_head_l20_20933


namespace volume_of_parallelepiped_l20_20587

variables {R : Type*} [Field R] 
variables (a b c : R^3)

theorem volume_of_parallelepiped 
  (h : |a.dot (b.cross c)| = 8) : 
  |(2 • a + 3 • b).dot ((b + 2 • c).cross (c - 4 • a))| = 208 := by
  sorry

end volume_of_parallelepiped_l20_20587


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20738

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20738


namespace count_even_three_digit_numbers_less_than_600_l20_20617

-- Define the digits
def digits : List ℕ := [1, 2, 3, 4, 5, 6]

-- Condition: the number must be less than 600, i.e., hundreds digit in {1, 2, 3, 4, 5}
def valid_hundreds (d : ℕ) : Prop := d ∈ [1, 2, 3, 4, 5]

-- Condition: the units (ones) digit must be even
def valid_units (d : ℕ) : Prop := d ∈ [2, 4, 6]

-- Problem: total number of valid three-digit numbers
def total_valid_numbers : ℕ :=
  List.product (List.product [1, 2, 3, 4, 5] digits) [2, 4, 6] |>.length

-- Proof statement
theorem count_even_three_digit_numbers_less_than_600 :
  total_valid_numbers = 90 := by
  sorry

end count_even_three_digit_numbers_less_than_600_l20_20617


namespace right_drawing_num_triangles_l20_20573

-- Given the conditions:
-- 1. Nine distinct lines in the right drawing
-- 2. Any combination of 3 lines out of these 9 forms a triangle
-- 3. Count of intersections of these lines where exactly three lines intersect

def num_triangles : Nat := 84 -- Calculated via binomial coefficient
def num_intersections : Nat := 61 -- Given or calculated from the problem

-- The target theorem to prove that the number of triangles is equal to 23
theorem right_drawing_num_triangles :
  num_triangles - num_intersections = 23 :=
by
  -- Proof would go here, but we skip it as per the instructions
  sorry

end right_drawing_num_triangles_l20_20573


namespace cos_sine_sum_l20_20997

theorem cos_sine_sum : (∑ n in Finset.range 91, (Real.cos (n * Real.pi / 180))^6) = 229 / 8 := 
by
  sorry

end cos_sine_sum_l20_20997


namespace first_problem_dy_dx_zero_at_t_zero_second_problem_second_derivative_third_problem_third_derivative_l20_20224

-- First Problem Statement
theorem first_problem_dy_dx_zero_at_t_zero (k t : ℝ) :
  (x = k * sin t + sin (k * t) ∧ y = k * cos t + cos (k * t)) → 
    ((dx/dx = dy/dx) t=0) := sorry

-- Second Problem Statement
theorem second_problem_second_derivative (α : ℝ) :
  (x = α^2 + 2 * α ∧ y = log (α + 1)) →
    (d^2 y/d x^2 = -1/(2 * (α + 1)^4)) := sorry

-- Third Problem Statement
theorem third_problem_third_derivative (a φ : ℝ) :
  (x = 1 + exp (a * φ) ∧ y = a * φ + exp (-a * φ)) → 
    (d^3 y/d x^3 = 2 * exp (-3 * a * φ) - 6 * exp (-4 * a * φ)) := sorry

end first_problem_dy_dx_zero_at_t_zero_second_problem_second_derivative_third_problem_third_derivative_l20_20224


namespace jerry_trays_l20_20076

theorem jerry_trays :
  ∀ (trays_from_table1 trays_from_table2 trips trays_per_trip : ℕ),
  trays_from_table1 = 9 →
  trays_from_table2 = 7 →
  trips = 2 →
  trays_from_table1 + trays_from_table2 = 16 →
  trays_per_trip = (trays_from_table1 + trays_from_table2) / trips →
  trays_per_trip = 8 :=
by
  intros
  sorry

end jerry_trays_l20_20076


namespace decimal_150th_digit_l20_20881

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20881


namespace one_div_thirteen_150th_digit_l20_20817

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20817


namespace decimal_150th_digit_of_1_div_13_l20_20692

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20692


namespace card_probability_l20_20185

theorem card_probability :
  let total_cards := 52
  let hearts := 13
  let clubs := 13
  let spades := 13
  let prob_heart_first := hearts / total_cards
  let remaining_after_heart := total_cards - 1
  let prob_club_second := clubs / remaining_after_heart
  let remaining_after_heart_and_club := remaining_after_heart - 1
  let prob_spade_third := spades / remaining_after_heart_and_club
  (prob_heart_first * prob_club_second * prob_spade_third) = (2197 / 132600) :=
  sorry

end card_probability_l20_20185


namespace paris_total_study_hours_semester_l20_20160

-- Definitions
def weeks_in_semester := 15
def weekday_study_hours_per_day := 3
def weekdays_per_week := 5
def saturday_study_hours := 4
def sunday_study_hours := 5

-- Theorem statement
theorem paris_total_study_hours_semester :
  weeks_in_semester * (weekday_study_hours_per_day * weekdays_per_week + saturday_study_hours + sunday_study_hours) = 360 := 
sorry

end paris_total_study_hours_semester_l20_20160


namespace dollar_neg3_4_eq_neg27_l20_20308

-- Define the operation $$
def dollar (a b : ℤ) : ℤ := a * (b + 1) + a * b

-- Theorem stating the value of (-3) $$ 4
theorem dollar_neg3_4_eq_neg27 : dollar (-3) 4 = -27 := 
by
  sorry

end dollar_neg3_4_eq_neg27_l20_20308


namespace one_div_thirteen_150th_digit_l20_20791

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20791


namespace find_sequence_l20_20254

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

variable (a : ℕ → ℝ)
axiom cond1 : determinant a 1 (1/2) 2 1 = 1
axiom cond2 : ∀ n : ℕ, determinant 3 3 (a n) (a (n+1)) = 12

theorem find_sequence : a 1 = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + 4 := sorry

end find_sequence_l20_20254


namespace find_ordered_pair_l20_20141

noncomputable def find_cd : ℝ × ℝ :=
  let c : ℝ := 1
  let d : ℝ := -2
  (c, d)

theorem find_ordered_pair (c d : ℝ) (h : c ≠ 0 ∧ d ≠ 0) (h1 : ∀ x : ℝ, x^2 + c * x + d = 0 → x = c ∨ x = d) : (c, d) = find_cd :=
by
  have Vieta_sum : c + d = -c := sorry
  have Vieta_prod : c * d = d := sorry
  -- Implementation of the intermediate steps and proof would go here
  assumption -- This is a placeholder to make the theorem compile successfully

end find_ordered_pair_l20_20141


namespace length_of_platform_is_350_l20_20938

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

end length_of_platform_is_350_l20_20938


namespace ways_to_sum_31_as_two_primes_l20_20055

theorem ways_to_sum_31_as_two_primes : 
  let p := 31 
  (p = 2 + 29 ∧ prime 2 ∧ prime 29) ∨ (p = 11 + 19 ∧ prime 11 ∧ prime 19) 
  → nat.num_possible_sums_of_two_primes p = 2 :=
by sorry

end ways_to_sum_31_as_two_primes_l20_20055


namespace tom_charges_per_lawn_l20_20189

theorem tom_charges_per_lawn (gas_cost earnings_from_weeding total_profit lawns_mowed : ℕ) (charge_per_lawn : ℤ) 
  (h1 : gas_cost = 17)
  (h2 : earnings_from_weeding = 10)
  (h3 : total_profit = 29)
  (h4 : lawns_mowed = 3)
  (h5 : total_profit = ((lawns_mowed * charge_per_lawn) + earnings_from_weeding) - gas_cost) :
  charge_per_lawn = 12 := 
by
  sorry

end tom_charges_per_lawn_l20_20189


namespace digit_150_of_decimal_1_div_13_l20_20660

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20660


namespace excursion_min_parents_l20_20595

theorem excursion_min_parents 
  (students : ℕ) 
  (car_capacity : ℕ)
  (h_students : students = 30)
  (h_car_capacity : car_capacity = 5) 
  : ∃ (parents_needed : ℕ), parents_needed = 8 := 
by
  sorry -- proof goes here

end excursion_min_parents_l20_20595


namespace place_value_difference_l20_20946

def numeral := 135.21

def hundreds_place := 10^2
def tenths_place := 10^(-1)

theorem place_value_difference : hundreds_place - tenths_place = 99.9 := by
  sorry

end place_value_difference_l20_20946


namespace determine_C_D_l20_20297

-- Define the given conditions and the statement to be proved
theorem determine_C_D :
  ∃ (C D : ℚ), 
  (C, D) = (-3/17 : ℚ, 81/17 : ℚ) ∧ 
  ∀ y : ℚ,
  (6 * y - 15) / (3 * y^3 - 13 * y^2 + 4 * y + 12) = 
  (C / (y + 3)) + (D / (3 * y^2 - 10 * y + 4)) := by
  sorry

end determine_C_D_l20_20297


namespace line_does_not_pass_through_third_quadrant_l20_20433

theorem line_does_not_pass_through_third_quadrant (k : ℝ) :
  (∀ x : ℝ, ¬ (x > 0 ∧ (-3 * x + k) < 0)) ∧ (∀ x : ℝ, ¬ (x < 0 ∧ (-3 * x + k) > 0)) → k ≥ 0 :=
by
  sorry

end line_does_not_pass_through_third_quadrant_l20_20433


namespace smallest_x_remainder_l20_20927

theorem smallest_x_remainder : ∃ x : ℕ, x > 0 ∧ 
    x % 6 = 5 ∧
    x % 7 = 6 ∧
    x % 8 = 7 ∧
    x = 167 :=
by
  sorry

end smallest_x_remainder_l20_20927


namespace k_value_l20_20373

-- Conditions
def line (k : ℝ) : (ℝ × ℝ) → Prop := λ p, p.2 = k * (p.1 - 2)
def parabola : (ℝ × ℝ) → Prop := λ p, (p.2)^2 = 8 * p.1
def focus : ℝ × ℝ := (2, 0)

-- Given the line and parabola intersect at points A and B where |AF| = 2|BF|
def points_A_B_on_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A ∧ parabola B ∧ line (2 * ℝ.sqrt 2) A ∧ line (2 * ℝ.sqrt 2) B

def distance (p q : ℝ × ℝ) : ℝ := Mathlib.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

axiom distance_AF_2_BF (A B : ℝ × ℝ) : points_A_B_on_parabola A B ∧ distance A focus = 2 * distance B focus

-- Prove that k == 2√2 given |AF| = 2|BF|
theorem k_value :
  ∃ k : ℝ, k = 2 * ℝ.sqrt 2 := sorry

end k_value_l20_20373


namespace smallest_k_for_difference_l20_20359

theorem smallest_k_for_difference (s : Finset ℕ) (h₁ : ∀ x ∈ s, x ≤ 2016) (h₂ : s.card = 674) :
  ∃ a b ∈ s, 672 < abs (a - b) ∧ abs (a - b) < 1344 :=
by
  sorry

end smallest_k_for_difference_l20_20359


namespace no_such_function_exists_l20_20315

def f (n : ℕ) : ℕ := sorry

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, n > 1 → (f n = f (f (n - 1)) + f (f (n + 1))) :=
by
  sorry

end no_such_function_exists_l20_20315


namespace decimal_150th_digit_of_1_div_13_l20_20696

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20696


namespace eulers_formula_l20_20098

noncomputable def exp_complex (a b : ℝ) : ℂ :=
  Complex.ez (a : ℂ) + I * (b : ℂ)

theorem eulers_formula (a b : ℝ) :
  (λ x : ℂ, Complex.exp x) (Complex.ofReal a + Complex.I * (Complex.ofReal b))
  = Complex.exp (Complex.ofReal a) * (Complex.cos (Complex.ofReal b) + Complex.I * Complex.sin (Complex.ofReal b)) :=
  by
  sorry

end eulers_formula_l20_20098


namespace range_of_f_compare_magnitude_l20_20012

noncomputable def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

-- Part (1): Proving the range of the function f(x)
theorem range_of_f :
  ∀ (y : ℝ), y ∈ set.range f ↔ y ∈ set.Icc (-3) 3 :=
sorry

-- Part (2): For a, b ∈ {y | y = f(x)}, compare 3|a + b| and |ab + 9|
theorem compare_magnitude (a b : ℝ) (ha : a ∈ set.Icc (-3) 3) (hb : b ∈ set.Icc (-3) 3) :
  3 * |a + b| ≤ |a * b + 9| :=
sorry

end range_of_f_compare_magnitude_l20_20012


namespace decimal_150th_digit_of_1_div_13_l20_20705

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20705


namespace correct_conclusions_l20_20384

variable (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0)

def f (x : ℝ) : ℝ := x^2

theorem correct_conclusions (h_distinct : x1 ≠ x2) :
  (f x1 * x2 = f x1 * f x2) ∧
  ((f x1 - f x2) / (x1 - x2) > 0) ∧
  (f ((x1 + x2) / 2) < (f x1 + f x2) / 2) :=
by
  sorry

end correct_conclusions_l20_20384


namespace remaining_amount_is_16_l20_20124

-- Define initial amount of money Sam has.
def initial_amount : ℕ := 79

-- Define cost per book.
def cost_per_book : ℕ := 7

-- Define the number of books.
def number_of_books : ℕ := 9

-- Define the total cost of books.
def total_cost : ℕ := cost_per_book * number_of_books

-- Define the remaining amount of money after buying the books.
def remaining_amount : ℕ := initial_amount - total_cost

-- Prove the remaining amount is 16 dollars.
theorem remaining_amount_is_16 : remaining_amount = 16 := by
  rfl

end remaining_amount_is_16_l20_20124


namespace ratio_CD_DB_l20_20070

-- Define the triangle and points on segments
variables (A B C D E T : Type) 
variables [triangle : Triangle A B C] [segmentDE : OnLineSegment D (LineSegment B C)] [segmentAE : OnLineSegment E (LineSegment A C)]

-- Conditions given in the problem
variables (AT DT BE ET : ℝ) (h1 : AT = 2 * DT) (h2 : BT = 5 * ET)

-- Goal to prove
theorem ratio_CD_DB (hAD : AD = AT + DT) (hBE : BE = BT + ET) : CD / DB = 1 / 5 :=
sorry

end ratio_CD_DB_l20_20070


namespace graph_of_equation_l20_20914

theorem graph_of_equation :
  ∀ x y : ℝ, (2 * x - 3 * y) ^ 2 = 4 * x ^ 2 + 9 * y ^ 2 → (x = 0 ∨ y = 0) :=
by
  intros x y h
  sorry

end graph_of_equation_l20_20914


namespace Xiaogang_raised_arm_exceeds_head_l20_20934

theorem Xiaogang_raised_arm_exceeds_head :
  ∀ (height shadow_no_arm shadow_with_arm : ℝ),
    height = 1.7 → shadow_no_arm = 0.85 → shadow_with_arm = 1.1 →
    (height / shadow_no_arm) = ((shadow_with_arm - shadow_no_arm) * (height / shadow_no_arm)) →
    shadow_with_arm - shadow_no_arm = 0.25 →
    ((height / shadow_no_arm) * 0.25) = 0.5 :=
by
  intros height shadow_no_arm shadow_with_arm h_eq1 h_eq2 h_eq3 h_eq4 h_eq5
  sorry

end Xiaogang_raised_arm_exceeds_head_l20_20934


namespace even_function_g_correct_l20_20368

theorem even_function_g_correct (g : ℝ → ℝ) (f : ℝ → ℝ) (x : ℝ) :
  (f = (λ x, x^3 * g x)) →
  (∀ x, f x = f (-x)) →
  (g = (λ x, 3^x - 3^(-x)) ∨ g = (λ x, Real.log (Real.sqrt (x^2 + 1) + x))) :=
by
  intros h1 h2
  sorry

end even_function_g_correct_l20_20368


namespace max_wins_l20_20290

theorem max_wins (Chloe_wins Max_wins : ℕ) (h1 : Chloe_wins = 24) (h2 : 8 * Max_wins = 3 * Chloe_wins) : Max_wins = 9 := by
  sorry

end max_wins_l20_20290


namespace decimal_150th_digit_l20_20893

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20893


namespace problem_f_symmetric_l20_20428

theorem problem_f_symmetric (f : ℝ → ℝ) (k : ℝ) (h : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + k * f b) (h_not_zero : ∃ x : ℝ, f x ≠ 0) :
  ∀ x : ℝ, f (-x) = f x :=
sorry

end problem_f_symmetric_l20_20428


namespace digit_150th_of_fraction_l20_20721

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20721


namespace decimal_150th_digit_l20_20889

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20889


namespace decimal_1_div_13_150th_digit_is_3_l20_20858

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20858


namespace decimal_150th_digit_l20_20653

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20653


namespace similar_triangles_height_l20_20192

theorem similar_triangles_height (h₁ h₂ : ℝ) (a₁ a₂ : ℝ) 
  (ratio_area : a₁ / a₂ = 1 / 9) (height_small : h₁ = 4) :
  h₂ = 12 :=
sorry

end similar_triangles_height_l20_20192


namespace total_study_time_l20_20158

theorem total_study_time
  (weeks : ℕ) (weekday_hours : ℕ) (weekend_saturday_hours : ℕ) (weekend_sunday_hours : ℕ)
  (H1 : weeks = 15)
  (H2 : ∀ i : ℕ, i < 5 → weekday_hours = 3)
  (H3 : weekend_saturday_hours = 4)
  (H4 : weekend_sunday_hours = 5) :
  let total_weekday_hours := 5 * weekday_hours in
  let total_weekend_hours := weekend_saturday_hours + weekend_sunday_hours in
  let total_week_hours := total_weekday_hours + total_weekend_hours in
  let total_semester_hours := total_week_hours * weeks in
  total_semester_hours = 360 := by
    sorry

end total_study_time_l20_20158


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20725

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20725


namespace intersection_of_M_and_N_l20_20412

-- Define the set M
def M : Set ℤ := { x | x^2 ≤ 1 }

-- Define the set N
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Define the intersection of M and N as a proof goal
theorem intersection_of_M_and_N : (↑M : Set ℝ) ∩ N = {0, 1} :=
by
  sorry

end intersection_of_M_and_N_l20_20412


namespace one_thirteenth_150th_digit_l20_20623

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20623


namespace find_common_ratio_l20_20065

variable {α : Type*} [linear_ordered_field α]
variable {a : ℕ → α}

-- Conditions
def is_geometric_sequence (a : ℕ → α) (q : α) :=
  ∀ n, a (n + 1) = q * a n

def a_2 : α := 8
def a_5 : α := 64

-- Proof Statement
theorem find_common_ratio (q : α) (a : ℕ → α) (h1 : is_geometric_sequence a q) (h2 : a 2 = a_2) (h3 : a 5 = a_5) :
  q = 2 := by
  sorry

end find_common_ratio_l20_20065


namespace count_valid_n_l20_20337

-- Definition: A fraction a/b has terminating decimal representation if and only if the denominator b has no prime factors other than 2 or 5 when simplified. 
def terminates_as_decimal (n b : ℕ) : Prop :=
  ∃ m k : ℕ, b = 2^m * 5^k * n

-- Definition: The number 1800 and its prime factorization
def denomin := 1800

-- The problem specifies n in [1,500] where the denominator only has factors 2 or 5 after simplification
def valid_n (n : ℕ) := n ∈ finset.Icc 1 500 ∧ ∃ k, n = 9 * k

-- Main theorem statement
theorem count_valid_n : finset.card (finset.filter valid_n (finset.Icc 1 500)) = 55 := by
  sorry

end count_valid_n_l20_20337


namespace digit_150_of_one_thirteenth_l20_20831

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20831


namespace dice_number_divisible_by_7_l20_20132

theorem dice_number_divisible_by_7 :
  ∃ a b c : ℕ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) 
               ∧ (1001 * (100 * a + 10 * b + c)) % 7 = 0 :=
by
  sorry

end dice_number_divisible_by_7_l20_20132


namespace one_thirteenth_150th_digit_l20_20636

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20636


namespace one_div_thirteen_150th_digit_l20_20781

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20781


namespace neighbor_packs_l20_20110

theorem neighbor_packs (n : ℕ) :
  let milly_balloons := 3 * 6 -- Milly and Floretta use 3 packs of their own
  let neighbor_balloons := n * 6 -- some packs of the neighbor's balloons, each contains 6 balloons
  let total_balloons := milly_balloons + neighbor_balloons -- total balloons
  -- They split balloons evenly; Milly takes 7 extra, then Floretta has 8 left
  total_balloons / 2 + 7 = total_balloons - 15
  → n = 2 := sorry

end neighbor_packs_l20_20110


namespace limit_sequence_eq_one_fifth_l20_20219

theorem limit_sequence_eq_one_fifth :
  (filter.tendsto (λ n : ℕ, ((2 * (n: ℝ) - 3)^3 - (↑n + 5)^3) / ((3 * ↑n - 1)^3 + (2 * ↑n + 3)^3))
    filter.at_top (nhds (1 / 5))) :=
begin
  sorry -- Proof is omitted as per instructions
end

end limit_sequence_eq_one_fifth_l20_20219


namespace correct_equation_l20_20293

-- Definitions based on conditions
def total_students := 98
def transfer_students := 3
def original_students_A (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ total_students
def students_B (x : ℕ) := total_students - x

-- Equation set up based on translation of the proof problem
theorem correct_equation (x : ℕ) (h : original_students_A x) :
  students_B x + transfer_students = x - transfer_students ↔ (98 - x) + 3 = x - 3 :=
by
  sorry
  
end correct_equation_l20_20293


namespace digit_150_of_decimal_1_div_13_l20_20658

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20658


namespace smallest_k_l20_20352

theorem smallest_k (k : ℕ) (numbers : set ℕ) (h₁ : ∀ x ∈ numbers, x ≤ 2016) (h₂ : numbers.card = k) :
  (∃ a b ∈ numbers, 672 < abs (a - b) ∧ abs (a - b) < 1344) ↔ k ≥ 674 := 
by
  sorry

end smallest_k_l20_20352


namespace sequence_arithmetic_and_sum_l20_20377

theorem sequence_arithmetic_and_sum (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h_sum_terms : ∀ n, a n + 2 * S n * S (n - 1) = 0)
  (h_a1 : a 1 = 1 / 2) :
  (∀ n, 1 / S n - 1 / S (n - 1) = 2) ∧
  (let b := λ n, 2^n / S n in ∀ n, sum (λ k, b k) (finset.range n) = 2^(n+2) * (n - 1) + 4) := 
sorry

end sequence_arithmetic_and_sum_l20_20377


namespace paris_total_study_hours_semester_l20_20161

-- Definitions
def weeks_in_semester := 15
def weekday_study_hours_per_day := 3
def weekdays_per_week := 5
def saturday_study_hours := 4
def sunday_study_hours := 5

-- Theorem statement
theorem paris_total_study_hours_semester :
  weeks_in_semester * (weekday_study_hours_per_day * weekdays_per_week + saturday_study_hours + sunday_study_hours) = 360 := 
sorry

end paris_total_study_hours_semester_l20_20161


namespace valid_n_count_l20_20341

def count_valid_n : ℕ :=
  (1 to 2020).count (λ n => (∃ k (1 ≤ k ∧ k ≤ n - 1), ((1 + Complex.exp (2 * Real.pi * Complex.I * k / n)) ^ n + 1) = 0))

theorem valid_n_count : count_valid_n = 337 := sorry

end valid_n_count_l20_20341


namespace decimal_1_div_13_150th_digit_is_3_l20_20847

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20847


namespace lastNumberIsOneOverSeven_l20_20113

-- Definitions and conditions
def seq (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, 2 ≤ k ∧ k ≤ 99 → a k = a (k - 1) * a (k + 1)

def nonZeroSeq (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 100 → a k ≠ 0

def firstSeq7 (a : ℕ → ℝ) : Prop :=
  a 1 = 7

-- Theorem statement
theorem lastNumberIsOneOverSeven (a : ℕ → ℝ) :
  seq a → nonZeroSeq a → firstSeq7 a → a 100 = 1 / 7 :=
by
  sorry

end lastNumberIsOneOverSeven_l20_20113


namespace integer_triples_soln_l20_20326

theorem integer_triples_soln (x y z : ℤ) :
  (x^3 + y^3 + z^3 - 3*x*y*z = 2003) ↔ ( (x = 668 ∧ y = 668 ∧ z = 667) ∨ (x = 668 ∧ y = 667 ∧ z = 668) ∨ (x = 667 ∧ y = 668 ∧ z = 668) ) := 
by
  sorry

end integer_triples_soln_l20_20326


namespace find_length_BC_l20_20515

noncomputable def length_BC {O A B M C: Type} [metric_space O] [circ : circular_order O] 
  (r : ℝ) (alpha : ℝ) (sin_alpha : sin alpha = (real.sqrt 35) / 6) 
  (AO : A ≠ O) (AM : M ≠ A) (OMC : angle O M C = alpha) 
  (AMB : angle A M B = alpha) : ℝ :=
  let BC := 2 * r * real.cos alpha in
  BC

theorem find_length_BC {O A B M C: Type} [metric_space O] [circ : circular_order O] 
  (r : ℝ) (alpha : ℝ) (sin_alpha : sin alpha = (real.sqrt 35) / 6)
  (radius: r = 12) (AO : A ≠ O) (AM : M ≠ A) (OMC : angle O M C = alpha) 
  (AMB : angle A M B = alpha) : length_BC r alpha sin_alpha AO AM OMC AMB = 4 :=
by
  sorry

end find_length_BC_l20_20515


namespace sum_palindromic_primes_less_than_60_l20_20517

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_palindromic_prime (n : ℕ) : Prop :=
  is_prime n ∧ n ≥ 10 ∧ n < 60 ∧ is_prime (nat.reverse n)

def palindromic_primes_under_60 : list ℕ := list.filter is_palindromic_prime [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

theorem sum_palindromic_primes_less_than_60 : list.sum palindromic_primes_under_60 = 55 := by sorry

end sum_palindromic_primes_less_than_60_l20_20517


namespace smallest_k_for_difference_l20_20358

theorem smallest_k_for_difference (s : Finset ℕ) (h₁ : ∀ x ∈ s, x ≤ 2016) (h₂ : s.card = 674) :
  ∃ a b ∈ s, 672 < abs (a - b) ∧ abs (a - b) < 1344 :=
by
  sorry

end smallest_k_for_difference_l20_20358


namespace decimal_150th_digit_of_1_div_13_l20_20701

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20701


namespace find_M_pos_int_l20_20922

theorem find_M_pos_int (M : ℕ) (hM : 33^2 * 66^2 = 15^2 * M^2) :
    M = 726 :=
by
  -- Sorry, skipping the proof.
  sorry

end find_M_pos_int_l20_20922


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20740

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20740


namespace average_score_of_girls_l20_20140

-- Define the variables involved
variables {f₁ l₁ f₂ l₂ : ℝ}

-- Define the conditions from the problem as Lean hypotheses
def cond1 : Prop := (71 * f₁ + 76 * l₁) / (f₁ + l₁) = 74
def cond2 : Prop := (81 * f₂ + 90 * l₂) / (f₂ + l₂) = 84
def cond3 : Prop := (71 * f₁ + 81 * f₂) / (f₁ + f₂) = 79

-- Rewrite the proof problem as a Lean theorem
theorem average_score_of_girls (h1 : cond1) (h2 : cond2) (h3 : cond3) :
  (76 * l₁ + 90 * l₂) / (l₁ + l₂) = 84 :=
sorry

end average_score_of_girls_l20_20140


namespace initial_marble_count_l20_20468

axiom JoshLostMarbles : ℕ := 7
axiom JoshCurrentMarbles : ℕ := 9

theorem initial_marble_count : ∃ n : ℕ, n - JoshLostMarbles = JoshCurrentMarbles ∧ n = 16 :=
by
  let n := 16
  have h : n - JoshLostMarbles = JoshCurrentMarbles := by sorry
  exact ⟨n, h, rfl⟩

end initial_marble_count_l20_20468


namespace problem_statement_l20_20231

-- Defining the data of Type A and Type B
def typeA : List ℝ := [2, 4, 5, 6, 8]
def typeB : List ℝ := [3, 4, 4, 4, 5]

-- Defining the means of Type A and Type B
def mean (xs : List ℝ) : ℝ := xs.sum / xs.length

-- Defining the covariance and variance functions
def covariance (xs ys : List ℝ) : ℝ :=
  let n := xs.length
  List.sum (List.map₂ (λ x y, (x - mean xs) * (y - mean ys)) xs ys) / n

def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  List.sum (List.map (λ x, (x - m) ^ 2) xs) / xs.length

-- Correlation coefficient
def correlation_coefficient (xs ys : List ℝ) : ℝ :=
  covariance xs ys / (Real.sqrt (variance xs) * Real.sqrt (variance ys))

-- Probability calculation
def count_greater_pairs (pairs : List (ℝ × ℝ)) : ℕ :=
  pairs.count (λ (a, b), a > b)

def probability_greater (pairs : List (ℝ × ℝ)) : ℝ :=
  let greater := count_greater_pairs pairs
  let total := pairs.length
  greater / total

theorem problem_statement : correlation_coefficient typeA typeB = sqrt (9/10) ∧
                             abs (correlation_coefficient typeA typeB) > 0.75 ∧
                             probability_greater [(2,3), (2,4), (2,4), (2,4), (2,5),
                                                  (4,3), (4,4), (4,4), (4,4), (4,5),
                                                  (5,3), (5,4), (5,4), (5,4), (5,5),
                                                  (6,3), (6,4), (6,4), (6,4), (6,5),
                                                  (8,3), (8,4), (8,4), (8,4), (8,5)] = 3/10 :=
by sorry  -- Proof of the theorem is omitted

end problem_statement_l20_20231


namespace sum_b_100_l20_20016

-- Define the sequence a_n with the given conditions
def a : ℕ → ℚ
| 0       := 1 -- Note: a_1 corresponds to index 0 in Lean
| (n+1)   := 1 / (n+1 : ℚ)

-- Define b_n based on a_{2n-1} * a_{2n+1}
def b (n : ℕ) : ℚ := a (2*n) * a (2*n+2)

-- Sum of the first 100 terms of b_n sequence
noncomputable def sum_b (k : ℕ) : ℚ :=
∑ n in finset.range k, b n

-- The theorem statement
theorem sum_b_100 : sum_b 100 = 100 / 201 := 
by
  sorry   -- Proof not required as per instructions

end sum_b_100_l20_20016


namespace frog_probability_0_4_l20_20249

-- Definitions and conditions
def vertices : List (ℤ × ℤ) := [(1,1), (1,6), (5,6), (5,1)]
def start_position : ℤ × ℤ := (2,3)

-- Probabilities for transition, boundary definitions, this mimics the recursive nature described
def P : ℤ × ℤ → ℝ
| (x, 1) => 1   -- Boundary condition for horizontal sides
| (x, 6) => 1   -- Boundary condition for horizontal sides
| (1, y) => 0   -- Boundary condition for vertical sides
| (5, y) => 0   -- Boundary condition for vertical sides
| (x, y) => sorry  -- General case for other positions

-- The theorem to prove
theorem frog_probability_0_4 : P (2, 3) = 0.4 :=
by
  sorry

end frog_probability_0_4_l20_20249


namespace rational_product_sum_l20_20606

theorem rational_product_sum (x y : ℚ) 
  (h1 : x * y < 0) 
  (h2 : x + y < 0) : 
  |y| < |x| ∧ y < 0 ∧ x > 0 ∨ |x| < |y| ∧ x < 0 ∧ y > 0 :=
by
  sorry

end rational_product_sum_l20_20606


namespace three_digit_numbers_units_digit_at_least_three_times_tens_l20_20420

theorem three_digit_numbers_units_digit_at_least_three_times_tens :
  ∃ (nums : Finset ℕ), 
  (∀ (n : ℕ), n ∈ nums ↔ 
    (100 ≤ n ∧ n < 1000) ∧
    (let u := n % 10 in
     let t := (n / 10) % 10 in
     let h := n / 100 in
     h > 0 ∧ 3 * t ≤ u)) ∧
  nums.card = 189 :=
begin
  sorry
end

end three_digit_numbers_units_digit_at_least_three_times_tens_l20_20420


namespace decimal_1_div_13_150th_digit_is_3_l20_20850

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20850


namespace digit_150_of_1_div_13_l20_20802

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20802


namespace one_div_thirteen_150th_digit_l20_20783

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20783


namespace christine_price_l20_20994

theorem christine_price 
  (original_price : ℝ)
  (discount_rate : ℝ) 
  (h1 : original_price = 25)
  (h2 : discount_rate = 0.25) :
  let discount_amount := original_price * discount_rate in
  let final_price := original_price - discount_amount in
  final_price = 18.75 :=
by 
  sorry

end christine_price_l20_20994


namespace find_f_minus_one_l20_20486

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then 2^x + sin (π * x / 2) + b else -(2^(-x) + sin (π * (-x) / 2) + b)

theorem find_f_minus_one (b : ℝ) (h : ∀ x : ℝ, f (-x) = -f x) :
  f (-1) = -2 :=
by
  sorry

end find_f_minus_one_l20_20486


namespace reflection_through_plane_l20_20084

noncomputable def reflection_matrix (u : ℝ^3) :=
  let n := ![2, -1, 1] in
  let proj := ((u ⬝ n) / (n ⬝ n)) • n in
  let q := u - proj in
  2 • q - u

theorem reflection_through_plane {u : ℝ^3} :
  reflection_matrix u = ![-(1/3), (4/3), (4/3)] • ![1,0,0] +
                        ![(1/3), (2/3), (4/3)] • ![0,1,0] +
                        ![(1/3), (2/3), (2/3)] • ![0,0,1] * u := sorry

end reflection_through_plane_l20_20084


namespace increasing_function_in_interval_l20_20984

noncomputable def y₁ (x : ℝ) : ℝ := abs (x + 1)
noncomputable def y₂ (x : ℝ) : ℝ := 3 - x
noncomputable def y₃ (x : ℝ) : ℝ := 1 / x
noncomputable def y₄ (x : ℝ) : ℝ := -x^2 + 4

theorem increasing_function_in_interval : ∀ x, (0 < x ∧ x < 1) → 
  y₁ x > y₁ (x - 0.1) ∧ y₂ x < y₂ (x - 0.1) ∧ y₃ x < y₃ (x - 0.1) ∧ y₄ x < y₄ (x - 0.1) :=
by {
  sorry
}

end increasing_function_in_interval_l20_20984


namespace rhombus_area_l20_20002

theorem rhombus_area (a b : ℝ) (s : ℝ) (d1 d2 : ℝ) (h1 : d1 / 2 = s) (h2 : d2 / 2 = s + 5) (h3 : d1 ^ 2 + d2 ^ 2 = (sqrt 165) ^ 2) (h4 : abs (d1 - d2) = 10) :
  (d1 * d2) / 2 = 305 / 4 := 
sorry -- Proof goes here

end rhombus_area_l20_20002


namespace max_min_f_value_l20_20395

theorem max_min_f_value (x : ℝ) (h : log (1/2) x^2 ≥ log (1/2) (3*x - 2)) :
  ∃ a b, 
  (∀ y, log 2 (y / 4) * log 2 (y / 2) ≤ a)
  ∧ (∀ y, log 2 (y / 4) * log 2 (y / 2) ≥ b)
  ∧ a = 2 ∧ b = 0 := by
sorry

end max_min_f_value_l20_20395


namespace number_of_non_empty_subsets_of_even_subset_l20_20029

-- Define the given set of numbers
def original_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the subset containing only even numbers from the original set
def even_subset : Set ℕ := {2, 4, 6, 8}

-- Define the number of non-empty subsets
def number_of_non_empty_subsets (s : Set ℕ) : ℕ :=
  2^s.card - 1

-- State the theorem
theorem number_of_non_empty_subsets_of_even_subset :
  number_of_non_empty_subsets even_subset = 15 :=
by {
  -- Skip the proof for now
  sorry
}

end number_of_non_empty_subsets_of_even_subset_l20_20029


namespace average_of_numbers_is_correct_l20_20913

theorem average_of_numbers_is_correct :
  let nums := [12, 13, 14, 510, 520, 530, 1120, 1, 1252140, 2345]
  let sum_nums := 1253205
  let count_nums := 10
  (sum_nums / count_nums.toFloat) = 125320.5 :=
by {
  sorry
}

end average_of_numbers_is_correct_l20_20913


namespace min_value_of_tan_reciprocals_l20_20072

theorem min_value_of_tan_reciprocals 
  (A B C : ℝ) [hABC : ∠ A + ∠ B + ∠ C = π] [hCosB : cos B = 1/4] :
  ∃ k : ℝ, k = (2 * sqrt 15) / 5 ∧ k = (1 / tan A) + (1 / tan C) :=
begin
  sorry
end

end min_value_of_tan_reciprocals_l20_20072


namespace kirill_height_l20_20471

theorem kirill_height (B : ℕ) (h1 : ∃ B, B - 14 = kirill_height) (h2 : B + (B - 14) = 112) : kirill_height = 49 :=
sorry

end kirill_height_l20_20471


namespace problem_correct_options_l20_20524

noncomputable def f (x : ℝ) : ℝ := 
  sqrt 3 * sin (2 * x - (π/6)) + 2 * (sin (x - (π/12)))^2

theorem problem_correct_options : 
  ¬(∀ x, f x ≤ 2) ∧ 
  ¬ (∀ x, x ∈ Ioo (-π/12) (5 * π/12) → f x) ∧
  (∀ x, y = 2 * sin (2 * x) + 1 → ∃ x', f x' = y) ∧ 
  (∀ m, ∀ x1 ∈ Icc (π/12) (π/2), ∀ x2 ∈ Icc (π/12) (π/2), 
    f x1 = m → f x2 = m → m ∈ Icc (sqrt 3 + 1) 3)
:=
by
  sorry

end problem_correct_options_l20_20524


namespace problem_statement_l20_20497

def g (x : ℝ) : ℝ :=
if x > 9 then real.sqrt x else x^2

theorem problem_statement : g (g (g 2)) = 4 := 
sorry

end problem_statement_l20_20497


namespace no_drifting_point_reciprocal_has_drifting_point_exponential_drifting_point_logarithm_range_l20_20046

-- Part 1: Prove that f(x) = 1/x does not have a drifting point.
theorem no_drifting_point_reciprocal (x0 : ℝ) : ¬ (1 / (x0 + 1) = 1 / x0 + 1) :=
sorry

-- Part 2: Prove that f(x) = x^2 + 2^x has a drifting point in (0,1).
theorem has_drifting_point_exponential : ∃ x0 ∈ set.Ioo 0 1, (x0 + 1)^2 + 2^(x0 + 1) = x0^2 + 2^x0 + 2 :=
sorry

-- Part 3: Determine the range of values for 'a' such that f(x) = log(a / (x^2 + 1)) has a drifting point in (0, +∞).
theorem drifting_point_logarithm_range (a : ℝ) : (∃ x0 > 0, Real.log (a / ((x0 + 1)^2 + 1)) = Real.log (a / (x0^2 + 1)) + Real.log (a / 2)) ↔ 0 < a ∧ a < 2 :=
sorry

end no_drifting_point_reciprocal_has_drifting_point_exponential_drifting_point_logarithm_range_l20_20046


namespace decimal_1_div_13_150th_digit_is_3_l20_20848

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20848


namespace missing_digit_is_0_l20_20162

/- Define the known digits of the number. -/
def digit1 : ℕ := 6
def digit2 : ℕ := 5
def digit3 : ℕ := 3
def digit4 : ℕ := 4

/- Define the condition that ensures the divisibility by 9. -/
def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

/- The main theorem to prove: the value of the missing digit d is 0. -/
theorem missing_digit_is_0 (d : ℕ) 
  (h : is_divisible_by_9 (digit1 + digit2 + digit3 + digit4 + d)) : 
  d = 0 :=
sorry

end missing_digit_is_0_l20_20162


namespace one_div_thirteen_150th_digit_l20_20792

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20792


namespace area_of_twelve_sided_figure_l20_20277

def vertices : list (ℝ × ℝ) :=
  [(1,3), (2,4), (2,5), (3,6), (4,6), (5,5), (6,4), (6,3), (5,2), (4,1), (3,1), (2,2)]

noncomputable def area_of_figure (vertices : list (ℝ × ℝ)) : ℝ := sorry

theorem area_of_twelve_sided_figure :
  area_of_figure vertices = 16 := sorry

end area_of_twelve_sided_figure_l20_20277


namespace one_div_thirteen_150th_digit_l20_20789

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l20_20789


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20737

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20737


namespace part1_part2_l20_20103

noncomputable def seq (S a : ℕ → ℚ) (n : ℕ) : Prop :=
  S n + a n = (n - 1) / (n^2 + n)

noncomputable def an (S: ℕ → ℚ) (n : ℕ) : ℚ :=
  if n = 1 then S 1 else S n - S (n - 1)

theorem part1 (S : ℕ → ℚ) (a : ℕ → ℚ) (h : ∀ n : ℕ, seq S a n) :
  ∃ (c r : ℚ), r ≠ 0 ∧
    (∀ n ≥ 1, S n - 1 / (n + 1) = c * r ^ (n - 1)) ∧
    c = -(1 / 2) ∧ r = 1 / 2 :=
by
  sorry

noncomputable def bn (S: ℕ → ℚ) (n : ℕ) : ℚ :=
  1 / ((1 / (n + 1)) - S n)

theorem part2 (S : ℕ → ℚ) (h : ∀ n : ℕ, seq S (an S) n) (bn : ℕ → ℚ):
  (∀ n : ℕ, bn n = 1 / ((1 / (n + 1)) - S n)) →
  ∑ i in finset.range n, bn i / ((bn i - 1) * (bn (i + 1) - 1)) =
    1 - (1 / (2^(n+1) - 1)) :=
by
  sorry

end part1_part2_l20_20103


namespace final_income_is_60000_l20_20973

def percentage_of (percentage: ℝ) (amount: ℝ) := (percentage / 100) * amount

def remaining_income (total_income: ℝ) (children_percentage: ℝ) (wife_percentage: ℝ) : ℝ :=
  total_income * (1 - (children_percentage + wife_percentage) / 100)

def final_amount (remaining: ℝ) (donation_percentage: ℝ) : ℝ :=
  remaining * (1 - donation_percentage / 100)

theorem final_income_is_60000 (total_income: ℝ) (children_percentage: ℝ) (wife_percentage: ℝ) (donation_percentage: ℝ) :
  total_income = 266666.67 → children_percentage = 15 * 3 → wife_percentage = 30 → donation_percentage = 10 →
  final_amount (remaining_income total_income children_percentage wife_percentage) donation_percentage = 60000 :=
  
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end final_income_is_60000_l20_20973


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20739

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20739


namespace gcd_546_210_l20_20009

theorem gcd_546_210 : Nat.gcd 546 210 = 42 := by
  sorry -- Proof is required to solve

end gcd_546_210_l20_20009


namespace final_price_percentage_l20_20977

theorem final_price_percentage (P : ℝ) (h₀ : P > 0)
  (h₁ : ∃ P₁, P₁ = 0.80 * P)
  (h₂ : ∃ P₂, P₁ = 0.80 * P ∧ P₂ = P₁ - 0.10 * P₁) :
  P₂ = 0.72 * P :=
by
  sorry

end final_price_percentage_l20_20977


namespace geometric_sequence_sum_l20_20102

theorem geometric_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) (n : ℕ) (hS : ∀ n, S n = t - 3 * 2^n) (h_geom : ∀ n, a (n + 1) = a n * r) :
  t = 3 :=
by
  sorry

end geometric_sequence_sum_l20_20102


namespace highest_power_of_3_dividing_consecutive_number_l20_20145

theorem highest_power_of_3_dividing_consecutive_number :
  let N := (List.range' 21 79).foldl (λ acc n => acc * 100 + n) 0 in
  ∃ k : ℕ, k = 2 ∧ 3^k ∣ N ∧ ∀ m : ℕ, 3^(m + 1) ∣ N → m < k :=
by
  sorry

end highest_power_of_3_dividing_consecutive_number_l20_20145


namespace minimum_elements_in_X_l20_20584

def digit_set := {x : ℕ // x < 10}
def pair_set := {p : digit_set × digit_set // p.1.val <= p.2.val}
def sequence := ℕ → digit_set

def satisfies_condition (X : set (digit_set × digit_set)) (s : sequence) : Prop :=
  ∃ n : ℕ, (s n, s (n + 1)) ∈ X

noncomputable def minimum_X_cardinality : ℕ := 55

theorem minimum_elements_in_X (X : set (digit_set × digit_set)) :
  (∀ s : sequence, satisfies_condition X s) → X.cardinality ≥ minimum_X_cardinality :=
by sorry

end minimum_elements_in_X_l20_20584


namespace cafeteria_optimization_l20_20255

-- Define the conditions as parameters
parameter (rice_price : ℝ := 1500)
parameter (transportation_fee : ℝ := 100)
parameter (storage_cost_per_day : ℝ := 2)
parameter (daily_rice_need : ℝ := 1)
parameter (discount_threshold : ℝ := 20)
parameter (discount_rate : ℝ := 0.95)

-- We need to show:
-- 1. Optimal purchase frequency to minimize the total average daily cost without discount
def optimal_purchase_frequency_without_discount (n : ℝ) : Prop :=
n = 10

-- 2. Demonstrate whether the cafeteria should accept the discount 
def should_accept_discount (m : ℝ) : Prop :=
20 ≤ m ∧ m + (transportation_fee / m) + 1426 < daily_rice_need * rice_price + (storage_cost_per_day * daily_rice_need * (daily_rice_need + 1)) / 2 + transportation_fee / daily_rice_need

theorem cafeteria_optimization :
  (∀ n, optimal_purchase_frequency_without_discount n) ∧ 
  (∀ m, should_accept_discount m) :=
sorry

end cafeteria_optimization_l20_20255


namespace decimal_150th_digit_l20_20646

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l20_20646


namespace alex_total_cost_l20_20960

noncomputable def base_cost : ℝ := 25
noncomputable def cost_per_text : ℝ := 0.05
noncomputable def cost_per_minute_over_50_hours : ℝ := 0.15
noncomputable def number_of_texts : ℝ := 200
noncomputable def total_hours : ℝ := 51
noncomputable def included_hours : ℝ := 50

theorem alex_total_cost :
  let total_cost := base_cost + cost_per_text * number_of_texts + cost_per_minute_over_50_hours * ((total_hours - included_hours) * 60)
  in total_cost = 44 := by
  sorry

end alex_total_cost_l20_20960


namespace decimal_150th_digit_of_1_div_13_l20_20703

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20703


namespace max_area_rectangle_l20_20201

theorem max_area_rectangle :
  ∀ x y : ℝ,
  (|y+1| * (y^2 + 2*y + 28) + |x-2| = 9 * (y^2 + 2*y + 4)) →
  (area := -4 * x * (x - 3)^3) →
  ∃ (max_area : ℝ), max_area = 34.171875 :=
begin
  sorry
end

end max_area_rectangle_l20_20201


namespace decimal_150th_digit_l20_20894

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20894


namespace lambda_sufficient_condition_l20_20417

def vec1 : ℝ × ℝ := (1, 2)
def vec2 : ℝ × ℝ := (2, 3)
def vec3 : ℝ × ℝ := (3, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem lambda_sufficient_condition (λ : ℝ) (hλ : λ < -4) : 
  ∃ λ : ℝ, ∀ (λ : ℝ), λ < -4 → dot_product (λ * vec1.1 + vec2.1, λ * vec1.2 + vec2.2) vec3 < 0 :=
by
  sorry

end lambda_sufficient_condition_l20_20417


namespace total_fruits_picked_l20_20097

variable (L M P B : Nat)

theorem total_fruits_picked (hL : L = 25) (hM : M = 32) (hP : P = 12) (hB : B = 18) : L + M + P = 69 :=
by
  sorry

end total_fruits_picked_l20_20097


namespace maximize_S_n_l20_20063

-- Define the necessary terms and conditions
variable {a_n : ℕ → ℝ}
variable {S_n : ℕ → ℝ}

-- Given conditions
axiom a1_pos (a1 : ℝ) : a1 > 0
axiom S_n_def (n : ℕ) : S_n n = (n * a1 + n * (n - 1) / 2 * (a_n 2 - a_n 1))
axiom S9_eq_S18 : S_n 9 = S_n 18

-- Goal: Proving that S_n is maximized when n = 13 or n = 14
theorem maximize_S_n : S_n 13 = S_n 14 := sorry

end maximize_S_n_l20_20063


namespace pumpkins_eaten_l20_20532

theorem pumpkins_eaten (initial: ℕ) (left: ℕ) (eaten: ℕ) (h1 : initial = 43) (h2 : left = 20) : eaten = 23 :=
by {
  -- We are skipping the proof as per the requirement
  sorry
}

end pumpkins_eaten_l20_20532


namespace a_500_is_2176_a_500_l20_20053

-- Define the sequence using a function a : ℕ → ℤ
def a : ℕ → ℤ
| 0 := 2010
| 1 := 2012
| n+2 := n + 3 - (a n + a (n+1))

-- State the theorem
theorem a_500_is_2176 :
  a 499 + a 500 + a 501 = 502 :=
begin
  sorry
end

theorem a_500 :
  a 500 = 2176 :=
begin
  sorry
end

end a_500_is_2176_a_500_l20_20053


namespace sqrt_pm_one_l20_20004

theorem sqrt_pm_one (a b : ℤ) (h1: (a + 11) = 1) (h2: 1 - b = 16) : 
  a = -10 ∧ b = -15 ∧ Int.cbrt (2 * a + 7 * b) = -5 := by
  have ha : a = 1 - 11 := by
    rw [h1]
  have hb : b = 1 - 16 := by
    rw [h2]
  rw [ha, hb]
  split
  case h => exact rfl
  case h => split
    case h => exact rfl
    case h3 =>
      have hcalc : 2 * (-10) + 7 * (-15) = -125 := by norm_num
      rw [hcalc]
      exact rfl

end sqrt_pm_one_l20_20004


namespace first_digit_base_6_870_l20_20199

theorem first_digit_base_6_870 : ∀ (n : ℕ), n = 870 → nat.digits 6 n ≠ [] ∧ list.head (nat.digits 6 n) = some 4 := by
  intro n hn
  rw hn
  sorry

end first_digit_base_6_870_l20_20199


namespace bobby_paid_for_shoes_l20_20991

theorem bobby_paid_for_shoes :
  let mold_cost := 250
  let hourly_labor_rate := 75
  let hours_worked := 8
  let discount_rate := 0.80
  let materials_cost := 150
  let tax_rate := 0.10

  let labor_cost := hourly_labor_rate * hours_worked
  let discounted_labor_cost := discount_rate * labor_cost
  let total_cost_before_tax := mold_cost + discounted_labor_cost + materials_cost
  let tax := total_cost_before_tax * tax_rate
  let total_cost_with_tax := total_cost_before_tax + tax

  total_cost_with_tax = 968 :=
by
  sorry

end bobby_paid_for_shoes_l20_20991


namespace cleaning_time_l20_20526

theorem cleaning_time (richard_time : ℕ) (cory_extra : ℕ) (blake_less : ℕ) (cleaning_times_per_week : ℕ)
  (H1 : richard_time = 22)
  (H2 : cory_extra = 3)
  (H3 : blake_less = 4)
  (H4 : cleaning_times_per_week = 2) :
  let cory_time := richard_time + cory_extra,
      blake_time := cory_time - blake_less,
      total_time := (richard_time + cory_time + blake_time) * cleaning_times_per_week
  in total_time = 136 :=
by 
  sorry

end cleaning_time_l20_20526


namespace rabbit_time_2_miles_l20_20264

def rabbit_travel_time (distance : ℕ) (rate : ℕ) : ℕ :=
  (distance * 60) / rate

theorem rabbit_time_2_miles : rabbit_travel_time 2 5 = 24 := by
  sorry

end rabbit_time_2_miles_l20_20264


namespace find_x_value_l20_20431

/-- Given x, y, z such that x ≠ 0, z ≠ 0, (x / 2) = y^2 + z, and (x / 4) = 4y + 2z, the value of x is 120. -/
theorem find_x_value (x y z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) (h1 : x / 2 = y^2 + z) (h2 : x / 4 = 4 * y + 2 * z) : x = 120 := 
sorry

end find_x_value_l20_20431


namespace K1K2_eq_one_over_four_l20_20970

theorem K1K2_eq_one_over_four
  (K1 : ℝ) (hK1 : K1 ≠ 0)
  (K2 : ℝ)
  (x1 y1 x2 y2 : ℝ)
  (hx1y1 : x1^2 - 4 * y1^2 = 4)
  (hx2y2 : x2^2 - 4 * y2^2 = 4)
  (hx0 : x0 = (x1 + x2) / 2)
  (hy0 : y0 = (y1 + y2) / 2)
  (K1_eq : K1 = (y1 - y2) / (x1 - x2))
  (K2_eq : K2 = y0 / x0) :
  K1 * K2 = 1 / 4 :=
sorry

end K1K2_eq_one_over_four_l20_20970


namespace digit_150_of_one_thirteenth_l20_20832

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20832


namespace one_thirteenth_150th_digit_l20_20628

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20628


namespace steering_wheel_translational_l20_20608

def drives_on_straight_road (uncle_zhang : Prop) : Prop :=
  uncle_zhang → (steering_wheel_translational_motion : Prop)

-- Proof statement 
theorem steering_wheel_translational (uncle_zhang : Prop) : drives_on_straight_road uncle_zhang :=
by 
  sorry

end steering_wheel_translational_l20_20608


namespace three_digit_even_less_than_600_count_l20_20610

theorem three_digit_even_less_than_600_count : 
  let digits := {1, 2, 3, 4, 5, 6} 
  let hundreds := {d ∈ digits | d < 6}
  let tens := digits 
  let units := {d ∈ digits | d % 2 = 0}
  ∑ (h : ℕ) in hundreds, ∑ (t : ℕ) in tens, ∑ (u : ℕ) in units, 1 = 90 :=
by
  sorry

end three_digit_even_less_than_600_count_l20_20610


namespace area_of_union_of_triangles_l20_20133

theorem area_of_union_of_triangles :
  ∀ (s : ℕ) (n : ℕ), s = 3 → n = 6 →
  let area := (3^2 * Real.sqrt 3) / 4
  let total_area := n * area
  let small_triangle_side := s / 2
  let small_triangle_area := (small_triangle_side^2 * Real.sqrt 3) / 4 
  let total_overlapping := (n - 1) * small_triangle_area
  let net_area := total_area - total_overlapping
  net_area = (171 * Real.sqrt 3) / 16 :=
by
  intros s n hs hn
  simp only [hs, hn]
  let area := (3^2 * Real.sqrt 3) / 4
  let total_area := 6 * area
  let small_triangle_side := 3 / 2
  let small_triangle_area := (small_triangle_side^2 * Real.sqrt 3) / 4 
  let total_overlapping := 5 * small_triangle_area
  let net_area := total_area - total_overlapping
  show net_area = (171 * Real.sqrt 3) / 16
  -- Providing the proof here is not necessary
  sorry

end area_of_union_of_triangles_l20_20133


namespace problem_part_1_problem_part_2_l20_20058

noncomputable def vector_A : ℝ × ℝ := (1, 4)
noncomputable def vector_B : ℝ × ℝ := (-2, 3)
noncomputable def vector_C : ℝ × ℝ := (2, -1)
noncomputable def vector_O : ℝ × ℝ := (0, 0)

noncomputable def vector_AB : ℝ × ℝ := (vector_A.1 - vector_B.1, vector_A.2 - vector_B.2)
noncomputable def vector_AC : ℝ × ℝ := (vector_A.1 - vector_C.1, vector_A.2 - vector_C.2)
noncomputable def vector_OC : ℝ × ℝ := (vector_C.1 - vector_O.1, vector_C.2 - vector_O.2)
noncomputable def vector_AB_plus_AC : ℝ × ℝ := (vector_AB.1 + vector_AC.1, vector_AB.2 + vector_AC.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem problem_part_1 :
  dot_product vector_AB vector_AC = 2 ∧ magnitude vector_AB_plus_AC = 2 * real.sqrt 10 :=
by 
  sorry

theorem problem_part_2 : ∃ t : ℝ, (dot_product (vector_AB - (t • vector_OC)) vector_OC = 0) ∧ t = -1 :=
by 
  sorry

end problem_part_1_problem_part_2_l20_20058


namespace range_of_a_for_two_distinct_solutions_l20_20369

theorem range_of_a_for_two_distinct_solutions (x : ℝ) (h : 0 ≤ x ∧ x ≤ π) :
    ∃ a : ℝ, (2 * Real.sin (x + π/3) = a) ∧ (a ∈ Ioo (Real.sqrt 3) 2) := 
sorry

end range_of_a_for_two_distinct_solutions_l20_20369


namespace digit_150th_of_fraction_l20_20710

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20710


namespace decimal_150th_digit_of_1_div_13_l20_20707

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20707


namespace one_thirteen_150th_digit_l20_20909

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20909


namespace digit_150_after_decimal_of_one_thirteenth_l20_20874

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20874


namespace tournament_list_contains_all_names_l20_20054

def tournament (Player : Type) :=
  ∀ (a b : Player), a ≠ b → (a beats b ∨ b beats a)

def player_list (Player : Type) [∀ P : Player,  decidable_rel (λ a b : Player, a beats b)] (p : Player) : set Player :=
  list_of (λ q : Player, p beats q ∨ (∃ r : Player, p beats r ∧ r beats q))

theorem tournament_list_contains_all_names
  (Player : Type)
  [fintype Player]
  [∀ P : Player, decidable_rel (λ a b : Player, a beats b)]
  (h_tournament : tournament Player)
  (h_total : ∀ (a b : Player), a ≠ b → (a beats b ∨ b beats a)) :
  ∃ p : Player, ∀ q : Player, q ≠ p → q ∈ player_list Player p :=
sorry

end tournament_list_contains_all_names_l20_20054


namespace one_div_thirteen_150th_digit_l20_20755

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20755


namespace problem_statement_l20_20022

def U : Set Int := {x | |x| < 5}
def A : Set Int := {-2, 1, 3, 4}
def B : Set Int := {0, 2, 4}

theorem problem_statement : (A ∩ (U \ B)) = {-2, 1, 3} := by
  sorry

end problem_statement_l20_20022


namespace max_power_at_v0_div_3_l20_20566

variable (C S ρ v₀ : ℝ)

def force_on_sail (v : ℝ) : ℝ :=
  (C * S * ρ * (v₀ - v) ^ 2) / 2

def power (v : ℝ) : ℝ :=
  (force_on_sail C S ρ v₀ v) * v

theorem max_power_at_v0_div_3 : ∃ v : ℝ, power C S ρ v₀ v = (2 * C * S * ρ * v₀ ^ 3) / 27 ∧ v = v₀ / 3 :=
by {
  sorry
}

end max_power_at_v0_div_3_l20_20566


namespace arithmetic_sqrt_16_sqrt_sqrt_81_cube_root_neg_64_l20_20148

theorem arithmetic_sqrt_16 : Nat.sqrt 16 = 4 :=
sorry

theorem sqrt_sqrt_81 : Nat.sqrt (Nat.sqrt 81) = 3 ∨ Nat.sqrt (Nat.sqrt 81) = -3 :=
sorry

theorem cube_root_neg_64 : Nat.cbrt (-64) = -4 :=
sorry

end arithmetic_sqrt_16_sqrt_sqrt_81_cube_root_neg_64_l20_20148


namespace digit_150th_of_fraction_l20_20709

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20709


namespace max_wins_l20_20289

theorem max_wins (Chloe_wins Max_wins : ℕ) (h1 : Chloe_wins = 24) (h2 : 8 * Max_wins = 3 * Chloe_wins) : Max_wins = 9 := by
  sorry

end max_wins_l20_20289


namespace digit_150_after_decimal_of_one_thirteenth_l20_20871

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l20_20871


namespace integral_sup_bound_l20_20039

noncomputable theory

open Complex

variable {c : ℕ → ℂ} (hc : ∀ k : ℕ, abs (c k) ≤ 1)

def binary_repr (n : ℕ) : ℕ → ℕ
| i := if n / 2 ^ i % 2 = 1 then 1 else 0

def xor (k n : ℕ) : ℕ :=
nat.bits (λ i, nat.bits (binary_repr k i) (binary_repr n i))

theorem integral_sup_bound (N : ℕ) (hN : 0 < N) :
  ∃ C δ : ℝ, 0 < C ∧ 0 < δ ∧
    ∫⁻ (x y : ℝ) in set.prod (set.interval (-π) π) (set.interval (-π) π),
      ennreal.of_real (⨆ (n : ℕ) (hn : n < N), (1 / N : ℝ) *
        abs (∑ k in finset.range n, c k * (exp (I * ((k : ℝ) * x + (xor k n) * y)))))
    ≤ C * N ^ (-δ) :=
begin
  sorry
end

end integral_sup_bound_l20_20039


namespace even_three_digit_numbers_l20_20613

-- Define the set of digits
def digits : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the conditions
def isEven (n : ℕ) : Prop := n % 2 = 0
def isLessThan600 (n : ℕ) : Prop := n < 600

-- Define the digit constraints for a, b, c
def validHundredsDigit (a : ℕ) : Prop := a ∈ {1, 2, 3, 4, 5}
def validTensDigit (b : ℕ) : Prop := b ∈ digits
def validUnitsDigit (c : ℕ) : Prop := c ∈ {2, 4, 6}

-- Define the number formation
def formNumber (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

-- Main statement
theorem even_three_digit_numbers : 
  {n : ℕ | ∃ a b c : ℕ, 
    validHundredsDigit a ∧ validTensDigit b ∧ validUnitsDigit c ∧ 
    isLessThan600 (formNumber a b c) ∧ isEven (formNumber a b c)}.card = 90 := 
by
  sorry

end even_three_digit_numbers_l20_20613


namespace longest_route_l20_20247

noncomputable def intersections : Finset (ℕ × ℕ) := sorry

noncomputable def streets (i1 i2 : (ℕ × ℕ)) : Prop := sorry

def valid_route (route : List (ℕ × ℕ)) : Bool :=
  (route.head? = some (0, 0)) ∧  -- route starts at A
  (route.reverse.head? = some (0, 0)) ∧  -- route ends at B
  route.Nodup ∧  -- no repeating intersections
  ∀ i j, i < route.length - 1 → streets (route.get i) (route.get (i + 1))

theorem longest_route :
  ∃ route, valid_route route ∧ (route.length - 1 = 34) := sorry

end longest_route_l20_20247


namespace remaining_amount_is_16_l20_20123

-- Define initial amount of money Sam has.
def initial_amount : ℕ := 79

-- Define cost per book.
def cost_per_book : ℕ := 7

-- Define the number of books.
def number_of_books : ℕ := 9

-- Define the total cost of books.
def total_cost : ℕ := cost_per_book * number_of_books

-- Define the remaining amount of money after buying the books.
def remaining_amount : ℕ := initial_amount - total_cost

-- Prove the remaining amount is 16 dollars.
theorem remaining_amount_is_16 : remaining_amount = 16 := by
  rfl

end remaining_amount_is_16_l20_20123


namespace popsicle_sticks_left_l20_20305

/-- Danielle has $10 for supplies. She buys one set of molds for $3, 
a pack of 100 popsicle sticks for $1. Each bottle of juice makes 20 popsicles and costs $2.
Prove that the number of popsicle sticks Danielle will be left with after making as many popsicles as she can is 40. -/
theorem popsicle_sticks_left (initial_money : ℕ)
    (mold_cost : ℕ) (sticks_cost : ℕ) (initial_sticks : ℕ)
    (juice_cost : ℕ) (popsicles_per_bottle : ℕ)
    (final_sticks : ℕ) :
    initial_money = 10 →
    mold_cost = 3 → 
    sticks_cost = 1 → 
    initial_sticks = 100 →
    juice_cost = 2 →
    popsicles_per_bottle = 20 →
    final_sticks = initial_sticks - (popsicles_per_bottle * (initial_money - mold_cost - sticks_cost) / juice_cost) →
    final_sticks = 40 :=
by
  intros h_initial_money h_mold_cost h_sticks_cost h_initial_sticks h_juice_cost h_popsicles_per_bottle h_final_sticks
  rw [h_initial_money, h_mold_cost, h_sticks_cost, h_initial_sticks, h_juice_cost, h_popsicles_per_bottle] at h_final_sticks
  norm_num at h_final_sticks
  exact h_final_sticks

end popsicle_sticks_left_l20_20305


namespace distinct_solutions_count_l20_20092

def f (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem distinct_solutions_count : 
  {c : ℝ | f (f (f (f c))) = 7}.toFinset.card = 5 := 
by sorry

end distinct_solutions_count_l20_20092


namespace one_over_thirteen_150th_digit_l20_20686

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20686


namespace one_div_thirteen_150th_digit_l20_20821

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20821


namespace rectangles_equal_area_implies_value_l20_20173

theorem rectangles_equal_area_implies_value (x y : ℝ) (h1 : x < 9) (h2 : y < 4)
  (h3 : x * (4 - y) = y * (9 - x)) : 360 * x / y = 810 :=
by
  -- We only need to state the theorem, the proof is not required.
  sorry

end rectangles_equal_area_implies_value_l20_20173


namespace digit_150_of_decimal_1_div_13_l20_20673

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20673


namespace one_thirteen_150th_digit_l20_20897

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20897


namespace jerry_claims_years_of_salary_l20_20464

theorem jerry_claims_years_of_salary
  (Y : ℝ)
  (salary_damage_per_year : ℝ := 50000)
  (medical_bills : ℝ := 200000)
  (punitive_damages : ℝ := 3 * (salary_damage_per_year * Y + medical_bills))
  (total_damages : ℝ := salary_damage_per_year * Y + medical_bills + punitive_damages)
  (received_amount : ℝ := 0.8 * total_damages)
  (actual_received_amount : ℝ := 5440000) :
  received_amount = actual_received_amount → Y = 30 := 
by
  sorry

end jerry_claims_years_of_salary_l20_20464


namespace cone_volume_percent_l20_20246

open Real

noncomputable def cone_filled_percent (h r : ℝ) : ℝ :=
  let original_volume := (1 / 3) * π * r ^ 2 * h
  let smaller_cone_volume := (1 / 3) * π * (5/6 * r) ^ 2 * (5/6 * h)
  (smaller_cone_volume / original_volume : ℝ)

theorem cone_volume_percent : 
  ∀ (h r : ℝ), r > 0 → h > 0 → 
  abs (cone_filled_percent h r - 0.5787) < 0.0001 :=
by
  intros h r hr_pos hh_pos
  let original_volume := (1 / 3) * π * r ^ 2 * h
  let smaller_cone_volume := (1 / 3) * π * (5/6 * r) ^ 2 * (5/6 * h)
  let ratio := smaller_cone_volume / original_volume
  have h1 : ratio = 125/216 := sorry
  show abs (ratio - 0.5787) < 0.0001, from sorry

end cone_volume_percent_l20_20246


namespace one_over_thirteen_150th_digit_l20_20689

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20689


namespace melanie_food_total_weight_l20_20508

def total_weight (brie_oz : ℕ) (bread_lb : ℕ) (tomatoes_lb : ℕ) (zucchini_lb : ℕ) 
           (chicken_lb : ℕ) (raspberries_oz : ℕ) (blueberries_oz : ℕ) : ℕ :=
  let brie_lb := brie_oz / 16
  let raspberries_lb := raspberries_oz / 16
  let blueberries_lb := blueberries_oz / 16
  brie_lb + raspberries_lb + blueberries_lb + bread_lb + tomatoes_lb + zucchini_lb + chicken_lb

theorem melanie_food_total_weight : total_weight 8 1 1 2 (3 / 2) 8 8 = 7 :=
by
  -- result placeholder
  sorry

end melanie_food_total_weight_l20_20508


namespace digit_150_of_1_div_13_l20_20806

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20806


namespace mod_cond_l20_20622

theorem mod_cond (n : ℤ) : 0 ≤ n ∧ n < 11 ∧ (-1234 ≡ n [MOD 11]) → n = 9 :=
by 
  sorry

end mod_cond_l20_20622


namespace digit_150_of_1_over_13_is_3_l20_20760

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20760


namespace student_arrangements_l20_20588

noncomputable def num_arrangements : ℕ := 192

theorem student_arrangements :
  ∃ (students : Finset ℕ) (A B C : ℕ),
  students.card = 7 ∧
  ∃ (arrangements : Finset (Finset ℕ)),
  (∀ (arr : Finset ℕ), arr ∈ arrangements → 
    A ∈ arr ∧ arr.card = 7 ∧ 
    ∃ (middle : ℕ), middle = A ∧ 
    ∃ (bc_pos : ℕ), student_pos B ∈ arr ∧ student_pos C ∈ arr ∧ 
    ((student_pos B + 1 = student_pos C) ∨ (student_pos C + 1 = student_pos B))
  ) ∧ arrangements.card = num_arrangements :=
sorry

end student_arrangements_l20_20588


namespace right_triangle_incenter_intersect_l20_20056

variables {A B C D K L E : Type*}
variables [ordered_ring A] [metric_space A]
variables [ordered_ring B] [metric_space B]
variables [ordered_ring C] [metric_space C]
variables [ordered_ring D] [metric_space D]
variables [ordered_ring K] [metric_space K]
variables [ordered_ring L] [metric_space L]
variables [ordered_ring E] [metric_space E]

-- Defining the points and the segments
def altitude (A B C D : Type*) := ∀ {X : Type*} [inhabited X] [metric_space X], 
  (X ∈ Euclidean_space (fin 2)) → orthogonal_projection A B C = D

def intersect_incenter (triangle : Type*) := ∀ {X : Type*} [inhabited X] [metric_space X],
  (X ∈ Euclidean_space (fin 2)) → 
  let incenter_triangle (A B D : Type*) := incenter_pos A B D incenter_triangle

theorem right_triangle_incenter_intersect (A B C D K L E : Type*) 
  [ordered_ring A] [metric_space A] 
  [ordered_ring B] [metric_space B]
  [ordered_ring C] [metric_space C]
  [ordered_ring D] [metric_space D] 
  [ordered_ring K] [metric_space K] 
  [ordered_ring L] [metric_space L] 
  [ordered_ring E] [metric_space E]
  (ABC_right : right_triangle A B C)
  (altitude_AD : altitude A B C D)
  (incenter_ABD : intersect_incenter ABD K)
  (incenter_ACD : intersect_incenter ACD L)
  (intersect_KL_AD : intersects KL E AD) :
  1 / distance A B + 1 / distance A C = 1 / distance A E := 
  sorry

end right_triangle_incenter_intersect_l20_20056


namespace digit_150_of_one_thirteenth_l20_20842

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20842


namespace digit_150_of_decimal_1_div_13_l20_20666

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20666


namespace one_thirteenth_150th_digit_l20_20632

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l20_20632


namespace sailboat_speed_max_power_l20_20564

-- Define the parameters
variables (C S ρ v0 : ℝ)

-- Define the force function
def force (v : ℝ) : ℝ :=
  (C * S * ρ * (v0 - v) ^ 2) / 2

-- Define the power function
def power (v : ℝ) : ℝ :=
  (force C S ρ v0 v) * v

-- Define the statement to be proven
theorem sailboat_speed_max_power : ∃ v : ℝ, (power C S ρ v0 v = Term.max (power C S ρ v0)) ∧ v = v0 / 3 :=
by
  sorry

end sailboat_speed_max_power_l20_20564


namespace exponential_inequality_l20_20031

variable (a b : ℝ)

theorem exponential_inequality (h : -1 < a ∧ a < b ∧ b < 1) : Real.exp a < Real.exp b :=
by
  sorry

end exponential_inequality_l20_20031


namespace shaded_region_area_correct_l20_20240

-- Define the geometrical essentials
def circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | dist p center = radius}

-- Conditions
def circle_a := circle (0, 0) 3
def circle_b := circle (4, 0) 3
def smaller_circle := circle (2, 0) 2

-- Define the correct answer
def shaded_area := 2 * Real.pi - 4 * Real.sqrt 5

-- Theorem statement
theorem shaded_region_area_correct :
  (area of the region outside smaller_circle and inside both circles A and B) = shaded_area := sorry

end shaded_region_area_correct_l20_20240


namespace one_div_thirteen_150th_digit_l20_20758

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20758


namespace digit_150th_of_fraction_l20_20711

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20711


namespace part1_part2_l20_20387

-- Part (1)
theorem part1 (a : ℝ) (P Q : Set ℝ) (hP : P = {x | 4 <= x ∧ x <= 7})
              (hQ : Q = {x | -2 <= x ∧ x <= 5}) :
  (Set.compl P ∩ Q) = {x | -2 <= x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) (P Q : Set ℝ)
              (hP : P = {x | a + 1 <= x ∧ x <= 2 * a + 1})
              (hQ : Q = {x | -2 <= x ∧ x <= 5})
              (h_sufficient : ∀ x, x ∈ P → x ∈ Q) 
              (h_not_necessary : ∃ x, x ∈ Q ∧ x ∉ P) :
  (0 <= a ∧ a <= 2) :=
by
  sorry

end part1_part2_l20_20387


namespace one_div_thirteen_150th_digit_l20_20824

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20824


namespace parabola_focus_l20_20151

open Classical

variable (a : ℝ) (h : a ≠ 0)

def parabola_focus_coordinates :=
  (0, 1 / (16 * a))

theorem parabola_focus :
  ∀ (a : ℝ) (h : a ≠ 0), parabola_focus_coordinates a h = (0, 1 / (16 * a)) :=
by
  intros
  sorry

end parabola_focus_l20_20151


namespace greatest_b_max_b_value_l20_20156

theorem greatest_b (b y : ℤ) (h : b > 0) (hy : y^2 + b*y = -21) : b ≤ 22 :=
sorry

theorem max_b_value : ∃ b : ℤ, (∀ y : ℤ, y^2 + b*y = -21 → b > 0) ∧ (b = 22) :=
sorry

end greatest_b_max_b_value_l20_20156


namespace valid_third_side_l20_20401

theorem valid_third_side (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 8) (h₃ : 5 < c) (h₄ : c < 11) : c = 8 := 
by 
  sorry

end valid_third_side_l20_20401


namespace Youseff_time_difference_l20_20936

theorem Youseff_time_difference 
  (blocks : ℕ)
  (walk_time_per_block : ℕ) 
  (bike_time_per_block_sec : ℕ) 
  (sec_per_min : ℕ)
  (h_blocks : blocks = 12) 
  (h_walk_time_per_block : walk_time_per_block = 1) 
  (h_bike_time_per_block_sec : bike_time_per_block_sec = 20) 
  (h_sec_per_min : sec_per_min = 60) : 
  (blocks * walk_time_per_block) - ((blocks * bike_time_per_block_sec) / sec_per_min) = 8 :=
by 
  sorry

end Youseff_time_difference_l20_20936


namespace valid_n_count_l20_20340

def count_valid_n : ℕ :=
  (1 to 2020).count (λ n => (∃ k (1 ≤ k ∧ k ≤ n - 1), ((1 + Complex.exp (2 * Real.pi * Complex.I * k / n)) ^ n + 1) = 0))

theorem valid_n_count : count_valid_n = 337 := sorry

end valid_n_count_l20_20340


namespace digit_150_of_1_div_13_l20_20794

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20794


namespace price_per_foot_of_fencing_l20_20203

theorem price_per_foot_of_fencing
  (area : ℝ) (total_cost : ℝ) (price_per_foot : ℝ)
  (h1 : area = 36) (h2 : total_cost = 1392) :
  price_per_foot = 58 :=
by
  sorry

end price_per_foot_of_fencing_l20_20203


namespace largest_angle_heptagon_l20_20550

theorem largest_angle_heptagon :
  ∃ (x : ℝ), 4 * x + 4 * x + 4 * x + 5 * x + 6 * x + 7 * x + 8 * x = 900 ∧ 8 * x = (7200 / 38) := 
by 
  sorry

end largest_angle_heptagon_l20_20550


namespace intersection_point_of_perpendicular_lines_l20_20310

theorem intersection_point_of_perpendicular_lines 
    (h1 : ∀ x y : ℝ, y = -3*x + 4 ↔ on_line₁ x y)
    (h2 : ∀ x y : ℝ, y = x/3 - 1 ∧ (3, -2) = pt ↔ on_line₂ x y) :
  ∃ x y : ℝ, on_line₁ x y ∧ on_line₂ x y ∧ (x, y) = (1.5, -0.5) := 
by 
  sorry

def on_line₁ (x y : ℝ) : Prop := y = -3*x + 4

def on_line₂ (x y : ℝ) : Prop := y = x / 3 - 1

def pt : ℝ × ℝ := (3, -2)

end intersection_point_of_perpendicular_lines_l20_20310


namespace arithmetic_sequence_S30_l20_20061

theorem arithmetic_sequence_S30
  (S : ℕ → ℕ)
  (h_arith_seq: ∀ m : ℕ, 2 * (S (2 * m) - S m) = S m + S (3 * m) - S (2 * m))
  (h_S10: S 10 = 4)
  (h_S20: S 20 = 20) :
  S 30 = 48 := 
by
  sorry

end arithmetic_sequence_S30_l20_20061


namespace monotonicity_intervals_l20_20330

def f (x a : ℝ) : ℝ := x * Real.exp x - a * x - (1/2) * a * x^2

def f' (x a : ℝ) : ℝ := (Real.exp x - a) * (x + 1)

theorem monotonicity_intervals (a : ℝ) :
  (a ≤ 0 → (∀ x : ℝ, (x > -1 → f' x a > 0) ∧ (x < -1 → f' x a < 0))) ∧
  ((0 < a ∧ a < 1 / Real.exp 1) → (∀ x : ℝ, (x < Real.log a → f' x a > 0) ∧ (Real.log a < x → x < -1 → f' x a < 0) ∧ (-1 < x → f' x a > 0))) ∧
  (a = 1 / Real.exp 1 → (∀ x : ℝ, f' x a > 0)) ∧
  (a > 1 / Real.exp 1 → (∀ x : ℝ, (x < -1 → f' x a > 0) ∧ (-1 < x → x < Real.log a → f' x a < 0) ∧ (Real.log a < x → f' x a > 0))) :=
by sorry

end monotonicity_intervals_l20_20330


namespace ellipse_standard_eq_and_max_area_l20_20007

theorem ellipse_standard_eq_and_max_area
  (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a > b) 
  (focal_len_eq : 2 = 2 * real.sqrt (a^2 - b^2))
  (point_on_ellipse : (1, 3/2) ∈ { p : ℝ × ℝ | (p.1 ^ 2) / a^2 + (p.2 ^ 2) / b^2 = 1 }) :
  (∃ (a b : ℝ), 
    ∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1) ∧ 
      (∀ (k : ℝ), 
        ∃ (x₁ y₁ x₂ y₂ : ℝ), 
          (x₁, y₁) ∈ { p : ℝ × ℝ | (p.1 ^ 2) / 4 + (p.2 ^ 2) / 3 = 1 } ∧ 
          (x₂, y₂) ∈ { p : ℝ × ℝ | (p.1 ^ 2) / 4 + (p.2 ^ 2) / 3 = 1 } ∧ 
          let ad := real.sqrt (1 + k ^ 2) * real.sqrt (2 * k ^ 2) in
          let area := ad * (4 * k / real.sqrt (1 + k ^ 2)) in 
          area = 6)) :=
sorry

end ellipse_standard_eq_and_max_area_l20_20007


namespace no_solutions_xyz_l20_20311

theorem no_solutions_xyz :
  ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 4 :=
by
  sorry

end no_solutions_xyz_l20_20311


namespace divisors_rectangular_table_l20_20327

theorem divisors_rectangular_table (n : ℕ) (h : n > 0) :
  (∃ a b : ℕ, a * b = (nat.divisors n).card ∧ 
    (∀ i j ∈ finset.range a, ∃! d ∈ nat.divisors n, d = σ (i * j)) ∧ 
    (∀ i ∈ finset.range a, finset.sum (finset.filter (λ (x : ℕ × ℕ), x.1 = i) (finset.product (finset.range a) (finset.range b))).val = 
      (i : ℕ) * (σ n / a)) ∧ 
    (∀ j ∈ finset.range b, finset.sum (finset.filter (λ (x : ℕ × ℕ), x.2 = j) (finset.product (finset.range a) (finset.range b))).val = 
      (j : ℕ) * (σ n / b))
    ) ↔ n = 1 :=
sorry

end divisors_rectangular_table_l20_20327


namespace one_div_thirteen_150th_digit_l20_20754

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20754


namespace speaking_sequences_count_l20_20438

theorem speaking_sequences_count (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 6) :
  ∃ speakers : Finset (Finset ℕ), speakers.card = 4 ∧ 
  ∀ s ∈ speakers, (A ∈ s ∨ B ∈ s) →
  ∑ i in speakers, 4.choose(i.card) * i.card! = 336 :=
by
  sorry

end speaking_sequences_count_l20_20438


namespace smallest_x_division_remainder_l20_20925

theorem smallest_x_division_remainder :
  ∃ x : ℕ, x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7 ∧ x = 167 := by
  sorry

end smallest_x_division_remainder_l20_20925


namespace one_thirteen_150th_digit_l20_20911

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20911


namespace max_salary_l20_20968

theorem max_salary
  (n : Nat)
  (p : Nat → Nat)
  (h_team_size : n = 18)
  (h_min_salary : ∀ i, p i ≥ 12000)
  (h_total_salary : ∑ i in Finset.range n, p i ≤ 480000) :
  ∃ i, p i = 276000 := sorry

end max_salary_l20_20968


namespace range_of_cos_B_l20_20402

-- Definition of circumcenter properties and given condition
variables {A B C O : Type} [RealLinearSpace A] [RealLinearSpace B] [RealLinearSpace C] [RealLinearSpace O]
variables (circumcenter : O → (A × B × C) → Prop)
variables {a b c : ℝ}
variables [Triangle ABC]

-- Assuming the triangle has a circumcenter at O with the given dot product condition
axiom dot_product_condition : 
  (circumcenter O (A, B, C)) →
  (∥(A - O)∥ * ∥(B - C)∥ * Real.cos (angle A O B - angle A O C)) = 
  3 * ∥(B - O)∥ * ∥(A - C)∥ * Real.cos (angle B O A - angle B O C) + 
  4 * ∥(C - O)∥ * ∥(B - A)∥ * Real.cos (angle C O B - angle C O A)

-- Statement to prove the range of values for cos B
theorem range_of_cos_B (h: circumcenter O (A, B, C) ∧
  (∥(A - O)∥ * ∥(B - C)∥ * Real.cos (angle A O B - angle A O C)) = 
  3 * ∥(B - O)∥ * ∥(A - C)∥ * Real.cos (angle B O A - angle B O C) + 
  4 * ∥(C - O)∥ * ∥(B - A)∥ * Real.cos (angle C O B - angle C O A)) :
  ∃ x: ℝ, \(\frac{\sqrt{2}}{3} \leq x < 1 \) ∧ x =  cos (angle B)
:= sorry

end range_of_cos_B_l20_20402


namespace find_abc_sum_l20_20144

theorem find_abc_sum 
  (f : ℤ → ℤ)
  (h1 : ∀ x, f(x+3) = 3*x^2 + 7*x + 4)
  (h2 : ∃ a b c, ∀ x, f(x) = a*x^2 + b*x + c) :
  ∃ a b c, a + b + c = 2 :=
by
  sorry

end find_abc_sum_l20_20144


namespace bag_closest_to_50kg_heaviest_lightest_diff_total_mass_of_bags_l20_20966

noncomputable def closest_to_standard (weights : List ℝ) (standard : ℝ) : ℝ :=
weights.minimumBy (λ w => abs (w - standard))

noncomputable def mass_difference (weights : List ℝ) : ℝ :=
weights.maximum - weights.minimum

noncomputable def total_mass (weights : List ℝ) (standard : ℝ) : ℝ :=
(weights.length * standard) + weights.sum

theorem bag_closest_to_50kg :
  closest_to_standard [+2, +3.5, -1, -0.5, -3, -1, +4, +1, -2, +1.5] 50 = -0.5 := 
by sorry

theorem heaviest_lightest_diff :
  mass_difference [+2, +3.5, -1, -0.5, -3, -1, +4, +1, -2, +1.5] = 7 :=
by sorry

theorem total_mass_of_bags :
  total_mass [+2, +3.5, -1, -0.5, -3, -1, +4, +1, -2, +1.5] 50 = 504.5 := 
by sorry

end bag_closest_to_50kg_heaviest_lightest_diff_total_mass_of_bags_l20_20966


namespace melanie_food_total_weight_l20_20507

def total_weight (brie_oz : ℕ) (bread_lb : ℕ) (tomatoes_lb : ℕ) (zucchini_lb : ℕ) 
           (chicken_lb : ℕ) (raspberries_oz : ℕ) (blueberries_oz : ℕ) : ℕ :=
  let brie_lb := brie_oz / 16
  let raspberries_lb := raspberries_oz / 16
  let blueberries_lb := blueberries_oz / 16
  brie_lb + raspberries_lb + blueberries_lb + bread_lb + tomatoes_lb + zucchini_lb + chicken_lb

theorem melanie_food_total_weight : total_weight 8 1 1 2 (3 / 2) 8 8 = 7 :=
by
  -- result placeholder
  sorry

end melanie_food_total_weight_l20_20507


namespace urn_problem_l20_20988

theorem urn_problem (N : ℕ) (h : (1/2) * (20 / (20 + N)) + (1/2) * (N / (20 + N)) = 0.6) : N = 20 :=
sorry

end urn_problem_l20_20988


namespace intersecting_circles_equal_angles_l20_20604

-- Definitions and assumptions based on the problem conditions
axiom Circle (O : Type*) (r : ℝ) : Type*
axiom Point (O : Type*) : Type*

variables {O1 O2 : Type*} (A B : Point O1) (r1 r2 : ℝ)

-- Conditions:
-- Two circles intersect at two points A and B
axiom circle1 : Circle O1 r1
axiom circle2 : Circle O2 r2

axiom A_on_circle1 : A ∈ circle1
axiom B_on_circle1 : B ∈ circle1

axiom A_on_circle2 : A ∈ circle2
axiom B_on_circle2 : B ∈ circle2

axiom radius1A : distance O1 A = r1
axiom radius1B : distance O1 B = r1 
axiom radius2A : distance O2 A = r2
axiom radius2B : distance O2 B = r2

-- Prove that the angles \( \angle O1AO2 \) and \( \angle O1BO2 \) are equal
theorem intersecting_circles_equal_angles :
  ∠(O1, A, O2) = ∠(O1, B, O2) :=
sorry

end intersecting_circles_equal_angles_l20_20604


namespace smallest_k_674_l20_20357

theorem smallest_k_674 :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 2017) → (S.card = 674) → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ (672 < a - b) ∧ (a - b < 1344) ∨ (672 < b - a) ∧ (b - a < 1344) :=
by sorry

end smallest_k_674_l20_20357


namespace employees_after_hiring_l20_20944

noncomputable def original_employees : ℝ := E
def female_ratio_initial : ℝ := 0.6
def female_ratio_final : ℝ := 0.55
def additional_male_workers : ℝ := 20
def final_employees (E : ℝ) : ℝ := E + additional_male_workers
def female_employees_initial : ℝ := female_ratio_initial * E

theorem employees_after_hiring
    (E : ℝ)
    (h1 : female_employees_initial = female_ratio_final * final_employees E):
  final_employees E = 240 := 
  sorry

end employees_after_hiring_l20_20944


namespace digit_150_of_1_over_13_is_3_l20_20772

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20772


namespace part1_part2_l20_20388

-- Part (1)
theorem part1 (a : ℝ) (P Q : Set ℝ) (hP : P = {x | 4 <= x ∧ x <= 7})
              (hQ : Q = {x | -2 <= x ∧ x <= 5}) :
  (Set.compl P ∩ Q) = {x | -2 <= x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) (P Q : Set ℝ)
              (hP : P = {x | a + 1 <= x ∧ x <= 2 * a + 1})
              (hQ : Q = {x | -2 <= x ∧ x <= 5})
              (h_sufficient : ∀ x, x ∈ P → x ∈ Q) 
              (h_not_necessary : ∃ x, x ∈ Q ∧ x ∉ P) :
  (0 <= a ∧ a <= 2) :=
by
  sorry

end part1_part2_l20_20388


namespace domain_of_f_l20_20957

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f y = x}

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

theorem domain_of_f : domain f = {x | x > 1} := sorry

end domain_of_f_l20_20957


namespace sine_transform_l20_20546

def coord_transform (x y : ℝ) : ℝ × ℝ := (0.5 * x, 3 * y)

theorem sine_transform (x y : ℝ) (h : y = sin x) :
  coord_transform x y = (0.5 * x, 3 * y) → y = 3 * sin (2 * x) :=
by
  intro h1
  sorry

end sine_transform_l20_20546


namespace only_statements_1_and_4_are_true_l20_20296

theorem only_statements_1_and_4_are_true (a x y : ℝ) :
  (a * (x + y) = a * x + a * y) ∧
  ¬ (a ^ (x + y) = a ^ x + a ^ y) ∧
  ¬ (log (x + y) = log x + log y) ∧
  (log x / log y = log x / log y) ∧
  ¬ (a * (x * y) = a * x * a * y) :=
by 
  -- proof part
  sorry

end only_statements_1_and_4_are_true_l20_20296


namespace polygon_is_octahedron_l20_20005

theorem polygon_is_octahedron (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end polygon_is_octahedron_l20_20005


namespace smallest_k_674_l20_20354

theorem smallest_k_674 :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 2017) → (S.card = 674) → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ (672 < a - b) ∧ (a - b < 1344) ∨ (672 < b - a) ∧ (b - a < 1344) :=
by sorry

end smallest_k_674_l20_20354


namespace sum_floor_div_leq_floor_l20_20520

theorem sum_floor_div_leq_floor (x : ℝ) (n : ℕ) : 
  ∑ k in Finset.range (n+1), (⌊(k : ℝ) * x⌋ / k) ≤ ⌊(n : ℝ) * x⌋ := 
by
  sorry

end sum_floor_div_leq_floor_l20_20520


namespace complement_of_A_in_U_l20_20503

-- Define the universal set U as the set of integers
def U : Set ℤ := Set.univ

-- Define the set A as the set of odd integers
def A : Set ℤ := {x : ℤ | ∃ k : ℤ, x = 2 * k + 1}

-- Define the complement of A in U
def complement_A : Set ℤ := U \ A

-- State the equivalence to be proved
theorem complement_of_A_in_U :
  complement_A = {x : ℤ | ∃ k : ℤ, x = 2 * k} :=
by
  sorry

end complement_of_A_in_U_l20_20503


namespace exists_epsilon_sum_divisible_by_1000_l20_20371

theorem exists_epsilon_sum_divisible_by_1000 (a : Fin 10 → ℤ) : 
  ∃ (ε : Fin 10 → ℤ), (∀ i, ε i ∈ {-1, 0, 1}) ∧ (∃ i, ε i ≠ 0) ∧ (∑ i, ε i * a i) % 1000 = 0 :=
by
  sorry

end exists_epsilon_sum_divisible_by_1000_l20_20371


namespace one_thirteen_150th_digit_l20_20898

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20898


namespace exists_strictly_increasing_sequences_l20_20130

theorem exists_strictly_increasing_sequences :
  ∃ u v : ℕ → ℕ, (∀ n, u n < u (n + 1)) ∧ (∀ n, v n < v (n + 1)) ∧ (∀ n, 5 * u n * (u n + 1) = v n ^ 2 + 1) :=
sorry

end exists_strictly_increasing_sequences_l20_20130


namespace shaded_region_area_l20_20242

structure Circle where
  radius : ℝ
  center : ℝ × ℝ

def is_tangent_internally (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  (dist c1.center p = c1.radius) ∧ (dist c2.center p = c2.radius) ∧ (dist c1.center c2.center = c2.radius - c1.radius)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem shaded_region_area :
  ∀ (A B C : ℝ × ℝ),
  ∀ (r1 r2 : ℝ),
  (r1 = 2) →
  (r2 = 3) →
  let smaller_circle := Circle.mk r1 C in
  let larger_circle_A := Circle.mk r2 A in
  let larger_circle_B := Circle.mk r2 B in
  is_tangent_internally smaller_circle larger_circle_A A →
  is_tangent_internally smaller_circle larger_circle_B B →
  (dist A B = 2 * r1) →
  let shaded_area := 3 * π - 3 * real.arccos (1 / 3) * π - 2 * real.sqrt 5 in
  sorry

end shaded_region_area_l20_20242


namespace sqrt_eight_eq_sqrt_three_over_two_eq_two_sqrt_three_squared_eq_l20_20537

theorem sqrt_eight_eq : sqrt 8 = 2 * sqrt 2 := sorry

theorem sqrt_three_over_two_eq : sqrt (3 / 2) = sqrt 6 / 2 := sorry

theorem two_sqrt_three_squared_eq : (2 * sqrt 3) ^ 2 = 12 := sorry

end sqrt_eight_eq_sqrt_three_over_two_eq_two_sqrt_three_squared_eq_l20_20537


namespace digit_150_of_1_over_13_is_3_l20_20774

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20774


namespace one_over_thirteen_150th_digit_l20_20684

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l20_20684


namespace minimum_parents_needed_l20_20597

/-- 
Given conditions:
1. There are 30 students going on the excursion.
2. Each car can accommodate 5 people, including the driver.
Prove that the minimum number of parents needed to be invited on the excursion is 8.
-/
theorem minimum_parents_needed (students : ℕ) (car_capacity : ℕ) (drivers_needed : ℕ) 
  (h1 : students = 30) (h2 : car_capacity = 5) (h3 : drivers_needed = 1) 
  : ∃ (parents : ℕ), parents = 8 :=
by
  existsi 8
  sorry

end minimum_parents_needed_l20_20597


namespace min_value_function_l20_20169

theorem min_value_function :
  let f := λ x : ℝ, (1 / 3) * x ^ 3 - 4 * x + 4 in
  ∃ x : ℝ, x ∈ set.Icc (0 : ℝ) 3 ∧ 
  (∀ y ∈ set.Icc (0 : ℝ) 3, f x ≤ f y) ∧ 
  f x = -(4 / 3) :=
begin
  sorry
end

end min_value_function_l20_20169


namespace digit_150_of_1_over_13_is_3_l20_20761

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20761


namespace compute_expression_l20_20034

theorem compute_expression : 
  let a := Real.log 8
  let b := Real.log 27
  4^(a/b) + 3^(b/a) = 13 := by
  sorry

end compute_expression_l20_20034


namespace one_thirteen_150th_digit_l20_20900

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20900


namespace incenter_excenter_on_circle_l20_20940

theorem incenter_excenter_on_circle {A B C : Point} {S : Circle}
  (h_tangent_A_to_B : tangent A B S)
  (h_tangent_A_to_C : tangent A C S) :
  lies_on (incenter A B C) S ∧ lies_on (excenter A B C) (side B C) S :=
sorry

end incenter_excenter_on_circle_l20_20940


namespace convex_pentagon_regular_l20_20964

-- Definitions for the problem conditions
-- A pentagon, modeled as a list of vertices in the complex plane
def is_convex (x: List Complex) : Prop := 
  ∀ i j k, i < j → j < k → i < k → Angle.of (x.j - x.i) (x.k - x.j) < π

-- All side lengths are rational
def sides_are_rational (x : List Complex) : Prop := 
  ∀ i, dist x.get(i mod 5) x.get((i+1) mod 5) ∈ ℚ

-- All angles are equal
def equal_angles (x : List Complex) : Prop := 
  ∀ i j k, Angle.of (x.j - x.i) (x.k - x.j) = Angle.of (x.(i+1 mod 5) - x.i) (x.(k+1 mod 5) - x.(j+1 mod 5))

-- The conclusion that the pentagon is regular
def is_regular (x : List Complex) : Prop := 
  (∀ i, dist x.get(i mod 5) x.get((i+1) mod 5) = dist x.get(0) x.get(1)) ∧ (equal_angles x)

-- The main theorem
theorem convex_pentagon_regular (x : List Complex) (h1 : is_convex x) (h2 : sides_are_rational x) (h3 : equal_angles x) : is_regular x :=
  sorry

end convex_pentagon_regular_l20_20964


namespace round_trip_time_l20_20583

/-- The speed of the boat in standing water (v_b) -/
def boat_speed : ℝ := 16

/-- The speed of the stream (v_s) -/
def stream_speed : ℝ := 2

/-- The distance to the destination (d) -/
def distance : ℝ := 7740

/-- The effective speed of the boat downstream (v_down) -/
def downstream_speed : ℝ := boat_speed + stream_speed

/-- The effective speed of the boat upstream (v_up) -/
def upstream_speed : ℝ := boat_speed - stream_speed

/-- The time taken to go downstream to the destination (t_down) -/
def time_downstream : ℝ := distance / downstream_speed

/-- The time taken to come back upstream to the starting point (t_up) -/
def time_upstream : ℝ := distance / upstream_speed

/-- The total time for the round trip (t_total) -/
def total_time : ℝ := time_downstream + time_upstream

/-- Proof that the total time for the round trip is 983 hours -/
theorem round_trip_time : total_time = 983 := by
  sorry

end round_trip_time_l20_20583


namespace trig_identity_l20_20362

theorem trig_identity (x : ℝ) (h : (cos x) / ((sin x) - 1) = 1 / 2) : (1 + (sin x)) / (cos x) = -1 / 2 :=
by
  sorry

end trig_identity_l20_20362


namespace palindrome_divisible_by_11_prob_l20_20258

def is_palindrome (n : ℕ) : Prop :=
  let digits := List.ofDigits [n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  in digits.reverse = digits

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def palindrome_in_range (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def random_palindrome_divisible_by_11 : Prop :=
  ∃ n, palindrome_in_range n ∧ is_palindrome n ∧ is_divisible_by_11 n

theorem palindrome_divisible_by_11_prob : 
  let total_palindromes := 9 * 10 * 10
  let valid_palindromes := 90
  let probability := valid_palindromes / total_palindromes
  probability = 1 / 10 :=
sorry

end palindrome_divisible_by_11_prob_l20_20258


namespace tan_domain_l20_20328

theorem tan_domain (x : ℝ) : 
  (∃ (k : ℤ), x = k * Real.pi - Real.pi / 4) ↔ 
  ¬(∃ (k : ℤ), x = k * Real.pi - Real.pi / 4) :=
sorry

end tan_domain_l20_20328


namespace one_div_thirteen_150th_digit_l20_20825

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20825


namespace total_study_time_l20_20159

theorem total_study_time
  (weeks : ℕ) (weekday_hours : ℕ) (weekend_saturday_hours : ℕ) (weekend_sunday_hours : ℕ)
  (H1 : weeks = 15)
  (H2 : ∀ i : ℕ, i < 5 → weekday_hours = 3)
  (H3 : weekend_saturday_hours = 4)
  (H4 : weekend_sunday_hours = 5) :
  let total_weekday_hours := 5 * weekday_hours in
  let total_weekend_hours := weekend_saturday_hours + weekend_sunday_hours in
  let total_week_hours := total_weekday_hours + total_weekend_hours in
  let total_semester_hours := total_week_hours * weeks in
  total_semester_hours = 360 := by
    sorry

end total_study_time_l20_20159


namespace roots_product_sum_l20_20095

-- Define the polynomial and the roots
def cubic_polynomial (x : ℝ) : ℝ := 5 * x^3 - 10 * x^2 + 17 * x - 7

-- Define the statement to prove using Vieta's formulas
theorem roots_product_sum :
  let p q r : ℝ in
  (cubic_polynomial p = 0) ∧ (cubic_polynomial q = 0) ∧ (cubic_polynomial r = 0) →
  (p * q + p * r + q * r) = 17 / 5 :=
by
  sorry

end roots_product_sum_l20_20095


namespace pie_contest_l20_20186

def first_student_pie := 7 / 6
def second_student_pie := 4 / 3
def third_student_eats_from_first := 1 / 2
def third_student_eats_from_second := 1 / 3

theorem pie_contest :
  (first_student_pie - third_student_eats_from_first = 2 / 3) ∧
  (second_student_pie - third_student_eats_from_second = 1) ∧
  (third_student_eats_from_first + third_student_eats_from_second = 5 / 6) :=
by
  sorry

end pie_contest_l20_20186


namespace number_of_correct_statements_is_one_l20_20279

/-- Among the following four statements, prove that the number of correct ones is exactly 1:
1. If two planes have three common points, then these two planes coincide.
2. Two lines can determine a plane.
3. If M ∈ α, M ∈ β, and α ∩ β = l, then M ∈ l.
4. In space, three lines intersecting at the same point are in the same plane. -/
theorem number_of_correct_statements_is_one :
  (∀ (p1 p2 : Plane) (p1 ≠ p2) (A B C : Point), 
    (A ∈ p1 ∧ A ∈ p2) ∧ (B ∈ p1 ∧ B ∈ p2) ∧ (C ∈ p1 ∧ C ∈ p2) → False) ∧
  (∀ (l1 l2 : Line), ∃ (p : Plane), l1 ≠ l2 → l1 ⊆ p ∧ l2 ⊆ p) ∧
  (∀ (M : Point) (α β : Plane) (l : Line), 
    M ∈ α ∧ M ∈ β ∧ α ∩ β = l → M ∈ l) ∧
  (∀ (l1 l2 l3 : Line) (P : Point), 
    (l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3) ∧ (P ∈ l1 ∧ P ∈ l2 ∧ P ∈ l3) → 
    (∃ (p : Plane), l1 ⊆ p ∧ l2 ⊆ p ∧ l3 ⊆ p)) →
  1 :=
sorry

end number_of_correct_statements_is_one_l20_20279


namespace digit_150_of_one_thirteenth_l20_20827

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l20_20827


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20736

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20736


namespace average_birds_seen_correct_l20_20106

-- Define the number of birds seen by each person
def birds_seen_by_marcus : ℕ := 7
def birds_seen_by_humphrey : ℕ := 11
def birds_seen_by_darrel : ℕ := 9

-- Define the number of people
def number_of_people : ℕ := 3

-- Calculate the total number of birds seen
def total_birds_seen : ℕ := birds_seen_by_marcus + birds_seen_by_humphrey + birds_seen_by_darrel

-- Calculate the average number of birds seen
def average_birds_seen : ℕ := total_birds_seen / number_of_people

-- Proof statement
theorem average_birds_seen_correct :
  average_birds_seen = 9 :=
by
  -- Leaving the proof out as instructed
  sorry

end average_birds_seen_correct_l20_20106


namespace f_is_increasing_f_at_1_l20_20211

noncomputable theory

open Real

-- Function f : ℝ → ℝ defined, with continuous derivative for x ≥ 0
variable {f : ℝ → ℝ}

-- Given conditions
axiom h1 : ∀ x, 0 ≤ x → ∃ (f : ℝ → ℝ), f (0) = 1 ∧ deriv f (0) = 0 ∧ ( ∀ x, 0 ≤ x → (1 + f x) * deriv (deriv f)) x = 1 + x )

-- You need to prove f is increasing for x ≥ 0
theorem f_is_increasing : ∀ x, 0 ≤ x → deriv f x > 0 :=
by 
  sorry

-- You need to show f(1) <= 4/3
theorem f_at_1 : f 1 ≤ 4/3 :=
by
  sorry

end f_is_increasing_f_at_1_l20_20211


namespace arithmetic_sequence_S7_eq_28_l20_20006

/--
Given the arithmetic sequence \( \{a_n\} \) and the sum of its first \( n \) terms is \( S_n \),
if \( a_3 + a_4 + a_5 = 12 \), then prove \( S_7 = 28 \).
-/
theorem arithmetic_sequence_S7_eq_28
  (a : ℕ → ℤ) -- Sequence a_n
  (S : ℕ → ℤ) -- Sum sequence S_n
  (h1 : a 3 + a 4 + a 5 = 12)
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) -- Sum formula
  : S 7 = 28 :=
sorry

end arithmetic_sequence_S7_eq_28_l20_20006


namespace one_thirteen_150th_digit_l20_20899

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20899


namespace bug_position_after_2021_jumps_l20_20129

theorem bug_position_after_2021_jumps : 
  let prime_points := [2, 3, 5, 7]
  let non_prime_points := [1, 4, 6]
  let start := 7
  let movement (p : ℕ) := if p ∈ prime_points then 2 else 3
  let next_position (p : ℕ) := (p + movement p) % 7
  (iterate next_position 2021 start) == 2 := by
    sorry

end bug_position_after_2021_jumps_l20_20129


namespace initial_passengers_l20_20980

theorem initial_passengers (P : ℝ) :
  (1/2 * (2/3 * P + 280) + 12 = 242) → P = 270 :=
by
  sorry

end initial_passengers_l20_20980


namespace digit_150_of_1_div_13_l20_20801

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l20_20801


namespace even_three_digit_numbers_l20_20614

-- Define the set of digits
def digits : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the conditions
def isEven (n : ℕ) : Prop := n % 2 = 0
def isLessThan600 (n : ℕ) : Prop := n < 600

-- Define the digit constraints for a, b, c
def validHundredsDigit (a : ℕ) : Prop := a ∈ {1, 2, 3, 4, 5}
def validTensDigit (b : ℕ) : Prop := b ∈ digits
def validUnitsDigit (c : ℕ) : Prop := c ∈ {2, 4, 6}

-- Define the number formation
def formNumber (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

-- Main statement
theorem even_three_digit_numbers : 
  {n : ℕ | ∃ a b c : ℕ, 
    validHundredsDigit a ∧ validTensDigit b ∧ validUnitsDigit c ∧ 
    isLessThan600 (formNumber a b c) ∧ isEven (formNumber a b c)}.card = 90 := 
by
  sorry

end even_three_digit_numbers_l20_20614


namespace problem_statement_l20_20501

noncomputable def f (x : ℝ) (a b α β : ℝ) : ℝ := a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f 2013 a b α β = 5) :
  f 2014 a b α β = 3 :=
by
  sorry

end problem_statement_l20_20501


namespace polynomial_root_sum_l20_20485

theorem polynomial_root_sum 
  (c d : ℂ) 
  (h1 : c + d = 6) 
  (h2 : c * d = 10) 
  (h3 : c^2 - 6 * c + 10 = 0) 
  (h4 : d^2 - 6 * d + 10 = 0) : 
  c^3 + c^5 * d^3 + c^3 * d^5 + d^3 = 16156 := 
by sorry

end polynomial_root_sum_l20_20485


namespace decimal_150th_digit_of_1_div_13_l20_20695

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20695


namespace even_three_digit_numbers_l20_20615

-- Define the set of digits
def digits : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the conditions
def isEven (n : ℕ) : Prop := n % 2 = 0
def isLessThan600 (n : ℕ) : Prop := n < 600

-- Define the digit constraints for a, b, c
def validHundredsDigit (a : ℕ) : Prop := a ∈ {1, 2, 3, 4, 5}
def validTensDigit (b : ℕ) : Prop := b ∈ digits
def validUnitsDigit (c : ℕ) : Prop := c ∈ {2, 4, 6}

-- Define the number formation
def formNumber (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

-- Main statement
theorem even_three_digit_numbers : 
  {n : ℕ | ∃ a b c : ℕ, 
    validHundredsDigit a ∧ validTensDigit b ∧ validUnitsDigit c ∧ 
    isLessThan600 (formNumber a b c) ∧ isEven (formNumber a b c)}.card = 90 := 
by
  sorry

end even_three_digit_numbers_l20_20615


namespace repeating_decimal_to_fraction_l20_20621

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = (433 / 990) ∧ x = (4 + 37 / 99) / 10 := by
  sorry

end repeating_decimal_to_fraction_l20_20621


namespace rods_needed_to_complete_6_step_pyramid_l20_20444

def rods_in_step (n : ℕ) : ℕ :=
  16 * n

theorem rods_needed_to_complete_6_step_pyramid (rods_1_step rods_2_step : ℕ) :
  rods_1_step = 16 → rods_2_step = 32 → rods_in_step 6 - rods_in_step 4 = 32 :=
by
  intros h1 h2
  sorry

end rods_needed_to_complete_6_step_pyramid_l20_20444


namespace probability_not_seated_beside_partner_l20_20605

theorem probability_not_seated_beside_partner:
  let total_arrangements := 5!
  let favorable_arrangements := (3! * 2 * 2)
  total_arrangements = 120 ∧ favorable_arrangements = 24 →
  let probability_all_together := favorable_arrangements / total_arrangements
  let probability_at_least_one_not_together := 1 - probability_all_together
  probability_at_least_one_not_together = 0.8 :=
by
  intros total_arrangements favorable_arrangements h
  let probability_all_together := favorable_arrangements / total_arrangements
  let probability_at_least_one_not_together := 1 - probability_all_together
  have : probability_at_least_one_not_together = 0.8,
  sorry

end probability_not_seated_beside_partner_l20_20605


namespace decimal_1_div_13_150th_digit_is_3_l20_20856

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l20_20856


namespace range_of_g_l20_20312

noncomputable def g (x : ℝ) : ℝ := 1 / (x^2 + 4)

theorem range_of_g : set_of (y : ℝ) (∃ x : ℝ, g x = y) = set.Ioc 0 (1/4) :=
by
  sorry

end range_of_g_l20_20312


namespace product_pass_rate_l20_20576

variable (a b : ℝ)

theorem product_pass_rate (h1 : 0 ≤ a) (h2 : a < 1) (h3 : 0 ≤ b) (h4 : b < 1) : 
  (1 - a) * (1 - b) = 1 - (a + b - a * b) :=
by sorry

end product_pass_rate_l20_20576


namespace correct_option_for_logical_judgment_l20_20454

-- Define the structures in the logical structure of algorithms
inductive AlgorithmStructure
| Sequential
| Conditional
| Loop

-- Define the options available
inductive Option
| A
| B
| C
| D

-- Define the correctness condition based on the question and conditions
def requiresLogicalJudgment (option : Option) : Prop :=
  option = Option.B -- Option B corresponds to Conditional structure and loop structure

-- Formal statement of the problem to be proved
theorem correct_option_for_logical_judgment : requiresLogicalJudgment Option.B :=
by 
  -- This is the placeholder for the proof which is not required in this problem translation
  sorry

end correct_option_for_logical_judgment_l20_20454


namespace digit_150_of_1_over_13_is_3_l20_20768

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20768


namespace digit_150_of_decimal_1_div_13_l20_20659

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20659


namespace digit_150th_of_fraction_l20_20708

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20708


namespace product_of_fractions_equals_one_div_75287520_l20_20287

def fraction_product : ℚ :=
  (∏ k in Finset.range 95, ((k + 1) : ℚ) / (k + 6))

theorem product_of_fractions_equals_one_div_75287520 :
  fraction_product = 1 / 75287520 := 
sorry

end product_of_fractions_equals_one_div_75287520_l20_20287


namespace decimal_150th_digit_l20_20880

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20880


namespace find_initial_bottle_caps_l20_20319

def initial_bottle_caps : ℤ := sorry

theorem find_initial_bottle_caps 
  (found_bottle_caps : ℤ)
  (total_bottle_caps : ℤ)
  (h : total_bottle_caps = initial_bottle_caps + found_bottle_caps) : initial_bottle_caps = 18 := 
by
  sorry

-- Given values:
example : find_initial_bottle_caps 63 81 :=
by
  sorry

end find_initial_bottle_caps_l20_20319


namespace largest_negative_a_l20_20917

noncomputable def largest_a := -0.45

theorem largest_negative_a :
  ∀ x ∈ Set.Ioo (-3 * Real.pi) (-5 * Real.pi / 2),
  (Exists (fun x => True) → (largest_a)) ∧ 
  (¬(∃ (ϵ : ℝ) (h : ϵ > 0), (∀ x ∈ Set.Ioo (-3 * Real.pi) (-5 * Real.pi / 2),
  (largest_a) < ϵ))
follows_from_condition :=
  begin
    sorry
  end

end largest_negative_a_l20_20917


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20731

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20731


namespace digit_150th_of_fraction_l20_20712

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l20_20712


namespace smallest_k_exists_l20_20349

theorem smallest_k_exists (s : Finset ℕ) :
  (∀ a b ∈ s, a ≠ b → (672 < |a - b| ∧ |a - b| < 1344)) →
  (∀ k, k < 674 → ∃ s : Finset ℕ, s.card = k ∧ (∀ a b ∈ s, a ≠ b → ¬ (672 < |a - b| ∧ |a - b| < 1344))) → False :=
begin
  sorry
end

end smallest_k_exists_l20_20349


namespace decimal_150th_digit_l20_20892

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l20_20892


namespace one_thirteen_150th_digit_l20_20901

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l20_20901


namespace inequality_proof_l20_20119

theorem inequality_proof (x y z : ℝ) (hx : 2 < x) (hx4 : x < 4) (hy : 2 < y) (hy4 : y < 4) (hz : 2 < z) (hz4 : z < 4) :
  (x / (y^2 - z) + y / (z^2 - x) + z / (x^2 - y)) > 1 :=
by
  sorry

end inequality_proof_l20_20119


namespace odd_if_and_only_if_m_even_l20_20089

theorem odd_if_and_only_if_m_even (p m : ℤ) (hp : odd p) : odd (p^2 + 3 * m * p) ↔ even m := 
  sorry

end odd_if_and_only_if_m_even_l20_20089


namespace one_div_thirteen_150th_digit_l20_20742

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20742


namespace find_a_values_l20_20041

noncomputable def function_a_max_value (a : ℝ) : ℝ :=
  a^2 + 2 * a - 9

theorem find_a_values (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : function_a_max_value a = 6) : 
    a = 3 ∨ a = 1/3 :=
  sorry

end find_a_values_l20_20041


namespace diana_hits_seven_l20_20539

-- Define the participants
inductive Player 
| Alex 
| Brooke 
| Carlos 
| Diana 
| Emily 
| Fiona

open Player

-- Define a function to get the total score of a participant
def total_score (p : Player) : ℕ :=
match p with
| Alex => 20
| Brooke => 23
| Carlos => 28
| Diana => 18
| Emily => 26
| Fiona => 30

-- Function to check if a dart target is hit within the range and unique
def is_valid_target (x y z : ℕ) :=
x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 1 ≤ x ∧ x ≤ 12 ∧ 1 ≤ y ∧ y ≤ 12 ∧ 1 ≤ z ∧ z ≤ 12

-- Check if the sum equals the score of the player
def valid_score (p : Player) (x y z : ℕ) :=
is_valid_target x y z ∧ x + y + z = total_score p

-- Lean 4 theorem statement, asking if Diana hits the region 7
theorem diana_hits_seven : ∃ x y z, valid_score Diana x y z ∧ (x = 7 ∨ y = 7 ∨ z = 7) :=
sorry

end diana_hits_seven_l20_20539


namespace james_money_left_no_foreign_currency_needed_l20_20463

noncomputable def JameMoneyLeftAfterPurchase : ℝ :=
  let usd_bills := 50 + 20 + 5 + 1 + 20 + 10 -- USD bills and coins
  let euro_in_usd := 5 * 1.20               -- €5 bill to USD
  let pound_in_usd := 2 * 1.35 - 0.8 / 100 * (2 * 1.35) -- £2 coin to USD after fee
  let yen_in_usd := 100 * 0.009 - 1.5 / 100 * (100 * 0.009) -- ¥100 coin to USD after fee
  let franc_in_usd := 2 * 1.08 - 1 / 100 * (2 * 1.08) -- 2₣ coins to USD after fee
  let total_usd := usd_bills + euro_in_usd + pound_in_usd + yen_in_usd + franc_in_usd
  let present_cost_with_tax := 88 * 1.08   -- Present cost after 8% tax
  total_usd - present_cost_with_tax        -- Amount left after purchasing the present

theorem james_money_left :
  JameMoneyLeftAfterPurchase = 22.6633 :=
by
  sorry

theorem no_foreign_currency_needed :
  (0 : ℝ)  = 0 :=
by
  sorry

end james_money_left_no_foreign_currency_needed_l20_20463


namespace max_intersections_with_cos_l20_20571

noncomputable def circle (h k r : ℝ) : set (ℝ × ℝ) :=
  { p | (p.1 - h) ^ 2 + (p.2 - k) ^ 2 = r ^ 2 }

def cos_graph : set (ℝ × ℝ) :=
  { p | p.2 = Real.cos p.1 }

def circleA : set (ℝ × ℝ) := circle 0 0 2
def circleB : set (ℝ × ℝ) := circle 0 1 1
def circleC : set (ℝ × ℝ) := circle 2 0 0.5
def circleD : set (ℝ × ℝ) := circle Real.pi 0 2

theorem max_intersections_with_cos :
  let num_intersections (c : set (ℝ × ℝ)) :=
    { p | p ∈ cos_graph ∧ p ∈ c }.finite.to_finset.card in
  num_intersections circleD >
  num_intersections circleA ∧
  num_intersections circleD >
  num_intersections circleB ∧
  num_intersections circleD >
  num_intersections circleC :=
sorry

end max_intersections_with_cos_l20_20571


namespace sample_size_is_59_l20_20967

def totalStudents : Nat := 295
def samplingRatio : Nat := 5

theorem sample_size_is_59 : totalStudents / samplingRatio = 59 := 
by
  sorry

end sample_size_is_59_l20_20967


namespace clock_hand_alignment_l20_20461

theorem clock_hand_alignment :
  ∃ t (ht : 0 ≤ t ∧ t < 60), 
    let minute_hand_position := 6 * (t + 7)
    let hour_hand_position := 90 + 0.5 * (t + 4)
    minute_hand_position = hour_hand_position →
    t = 9 + 5/60 :=
by
  sorry

end clock_hand_alignment_l20_20461


namespace decimal_150th_digit_of_1_div_13_l20_20698

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l20_20698


namespace rabbit_travel_time_l20_20266

noncomputable def rabbit_speed : ℝ := 5 -- speed of the rabbit in miles per hour
noncomputable def rabbit_distance : ℝ := 2 -- distance traveled by the rabbit in miles

theorem rabbit_travel_time :
  let t := (rabbit_distance / rabbit_speed) * 60 in
  t = 24 :=
by
  sorry

end rabbit_travel_time_l20_20266


namespace shaded_region_area_l20_20243

structure Circle where
  radius : ℝ
  center : ℝ × ℝ

def is_tangent_internally (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  (dist c1.center p = c1.radius) ∧ (dist c2.center p = c2.radius) ∧ (dist c1.center c2.center = c2.radius - c1.radius)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem shaded_region_area :
  ∀ (A B C : ℝ × ℝ),
  ∀ (r1 r2 : ℝ),
  (r1 = 2) →
  (r2 = 3) →
  let smaller_circle := Circle.mk r1 C in
  let larger_circle_A := Circle.mk r2 A in
  let larger_circle_B := Circle.mk r2 B in
  is_tangent_internally smaller_circle larger_circle_A A →
  is_tangent_internally smaller_circle larger_circle_B B →
  (dist A B = 2 * r1) →
  let shaded_area := 3 * π - 3 * real.arccos (1 / 3) * π - 2 * real.sqrt 5 in
  sorry

end shaded_region_area_l20_20243


namespace melanie_total_weight_l20_20509

def weight_of_brie : ℝ := 8 / 16 -- 8 ounces converted to pounds
def weight_of_bread : ℝ := 1
def weight_of_tomatoes : ℝ := 1
def weight_of_zucchini : ℝ := 2
def weight_of_chicken : ℝ := 1.5
def weight_of_raspberries : ℝ := 8 / 16 -- 8 ounces converted to pounds
def weight_of_blueberries : ℝ := 8 / 16 -- 8 ounces converted to pounds

def total_weight : ℝ := weight_of_brie + weight_of_bread + weight_of_tomatoes + weight_of_zucchini +
                        weight_of_chicken + weight_of_raspberries + weight_of_blueberries

theorem melanie_total_weight : total_weight = 7 := by
  sorry

end melanie_total_weight_l20_20509


namespace RX_length_l20_20452

-- Declaring the segments and lengths as given in the problem
variables (CD WX CX DR RW CR RX : ℝ)

-- Conditions given in the problem
axiom CD_parallel_WX : CD ∥ WX
axiom CX_length : CX = 56
axiom DR_length : DR = 16
axiom RW_length : RW = 32

-- Prove the length of segment RX
theorem RX_length : RX = 37 + 1 / 3 :=
by 
  sorry

end RX_length_l20_20452


namespace John_pushup_count_l20_20937

-- Definitions arising from conditions
def Zachary_pushups : ℕ := 51
def David_pushups : ℕ := Zachary_pushups + 22
def John_pushups : ℕ := David_pushups - 4

-- Theorem statement
theorem John_pushup_count : John_pushups = 69 := 
by 
  sorry

end John_pushup_count_l20_20937


namespace equal_segments_on_trapezoid_base_l20_20514

theorem equal_segments_on_trapezoid_base
  (A B C D O P Q M N: Type*)
  [Trapezoid A B C D]
  [Segment_Intersects_Diagonals AC BD O]
  [Segment AP OC]
  [Segment DQ OB]
  [Segment_Intercepted AM on AD by A B P]
  [Segment_Intercepted DN on AD by D C Q]
  : AM = DN := 
  sorry

end equal_segments_on_trapezoid_base_l20_20514


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l20_20729

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l20_20729


namespace digit_150_of_decimal_1_div_13_l20_20663

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l20_20663


namespace inequality_l20_20581

def seq (n : ℕ) : ℕ 
| 0 := 1
| 1 := 2
| (n+2) := seq (n+1) + seq n

theorem inequality (n : ℕ) : 
  Real.root n (seq (n+1)) ≥ 1 + 1 / Real.root n (seq n) :=
sorry

end inequality_l20_20581


namespace ratio_of_areas_ABD_ABC_l20_20981

-- Definitions of the points and the angles
variables {A B C D : Type} -- assuming points A, B, C, D are of some type

-- Given conditions
def all_edges_tangent_to_sphere (A B C D : Type) : Prop := sorry -- sphere tangency property

def midpoints_of_skew_edges_are_equal (A B C D : Type) : Prop := sorry -- midpoint segments equality

def angle_DBC_is_50_degrees (A B C D : Type) : Prop := sorry -- given angle measure

def angle_BCD_greater_BDC (A B C D : Type) : Prop := sorry -- one angle is greater than the other

-- The final theorem
theorem ratio_of_areas_ABD_ABC (A B C D : Type)
  (h1 : all_edges_tangent_to_sphere A B C D)
  (h2 : midpoints_of_skew_edges_are_equal A B C D)
  (h3 : angle_DBC_is_50_degrees A B C D)
  (h4 : angle_BCD_greater_BDC A B C D) :
  ratio_of_areas (face_ABD A B D) (face_ABC A B C) = sqrt 3 * tan (40 * pi / 180) := 
sorry

end ratio_of_areas_ABD_ABC_l20_20981


namespace triangle_ABC_perimeter_l20_20051

noncomputable def triangle_perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem triangle_ABC_perimeter
  (a b c : ℝ)
  (h₁ : c ^ 2 = a ^ 2 + b ^ 2)
  (h₂ : b = 5 * Real.sqrt 2)
  (h₃ : a = 5 * Real.sqrt 2)
  (h₄ : ∀ A B C D E F G H I O, A = B ∧ B = C ∧ D = E ∧ E = F ∧ F = G ∧ G = H ∧ H = I ∧ I = O ∧ O = A) :
  triangle_perimeter a b c = 15 * Real.sqrt 2 :=
by
  sorry

end triangle_ABC_perimeter_l20_20051


namespace count_valid_mappings_l20_20020

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {2, 3, 4, 5}

-- Define the predicate for our mapping f to make x + f(x) + x * f(x) odd
def valid_mapping (f : ℤ → ℤ) : Prop :=
  ∀ x ∈ M, (x + f x + x * f x) % 2 = 1

-- The number of such mappings
def num_valid_mappings : ℕ :=
  (finset.univ.filter (λ f, valid_mapping f)).card

theorem count_valid_mappings :
  num_valid_mappings = 32 :=
sorry

end count_valid_mappings_l20_20020


namespace prob_heart_club_spade_l20_20183

-- Definitions based on the conditions
def total_cards : ℕ := 52
def cards_per_suit : ℕ := 13

-- Definitions based on the question
def prob_first_heart : ℚ := cards_per_suit / total_cards
def prob_second_club : ℚ := cards_per_suit / (total_cards - 1)
def prob_third_spade : ℚ := cards_per_suit / (total_cards - 2)

-- The main proof statement to be proved
theorem prob_heart_club_spade :
  prob_first_heart * prob_second_club * prob_third_spade = 169 / 10200 :=
by
  sorry

end prob_heart_club_spade_l20_20183


namespace smallest_k_exists_l20_20348

theorem smallest_k_exists (s : Finset ℕ) :
  (∀ a b ∈ s, a ≠ b → (672 < |a - b| ∧ |a - b| < 1344)) →
  (∀ k, k < 674 → ∃ s : Finset ℕ, s.card = k ∧ (∀ a b ∈ s, a ≠ b → ¬ (672 < |a - b| ∧ |a - b| < 1344))) → False :=
begin
  sorry
end

end smallest_k_exists_l20_20348


namespace digit_150_of_1_over_13_is_3_l20_20765

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l20_20765


namespace one_div_thirteen_150th_digit_l20_20816

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l20_20816


namespace one_div_thirteen_150th_digit_l20_20748

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l20_20748
