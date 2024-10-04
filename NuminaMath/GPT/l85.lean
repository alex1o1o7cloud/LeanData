import Mathlib

namespace trailing_zeros_2007_factorial_last_non_zero_digit_2007_factorial_l85_85440

theorem trailing_zeros_2007_factorial :
  ∑ k in (range 1 ((2007 / 5).log 5).ceil.succ), (2007 / 5 ^ k).floor = 500 := sorry

theorem last_non_zero_digit_2007_factorial :
  ∃ d, d ≠ 0 ∧ factorial 2007 % 10 = d := sorry

end trailing_zeros_2007_factorial_last_non_zero_digit_2007_factorial_l85_85440


namespace probability_sum_less_than_13_l85_85309

/--
Two fair, eight-sided dice are rolled. Prove that the probability
that the sum of the two numbers showing is less than 13 is 27/32.
-/
theorem probability_sum_less_than_13 (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 8) (hb : 1 ≤ b ∧ b ≤ 8) :
  (∑ x in finset.range 8, ∑ y in finset.range 8, if (x + 1) + (y + 1) < 13 then 1 else 0) / (8 * 8) = 27 / 32 := sorry

end probability_sum_less_than_13_l85_85309


namespace cylindrical_container_invariant_volume_l85_85363

theorem cylindrical_container_invariant_volume {r h y : ℝ} (hr : r = 6) (hh : h = 5) :
    (∀ y > 0, π * (r + y)^2 * h - π * r^2 * h = π * r^2 * (h + y) - π * r^2 * h) → y = 2.16 :=
by
  intro hy
  sorry

end cylindrical_container_invariant_volume_l85_85363


namespace binary_111_eq_7_l85_85998

theorem binary_111_eq_7 : (1 * 2^0 + 1 * 2^1 + 1 * 2^2) = 7 :=
by
  sorry

end binary_111_eq_7_l85_85998


namespace conditional_probability_l85_85291

-- Given conditions
variables (P A B : Prop)
variable [ProbabilityMeasure P]

axiom prob_A : P(A) = 0.8
axiom prob_B : P(B) = 0.4
axiom prob_A_inter_B : P(A ∧ B) = 0.4

-- Question: Prove conditional probability
theorem conditional_probability (A B : Prop) [ProbabilityMeasure P] :
  P(B|A) = 0.5 :=
by
  sorry

end conditional_probability_l85_85291


namespace six_divides_aA_l85_85478

theorem six_divides_aA (k : ℤ) (a : ℤ) (A : ℤ) (hk : k ≥ 3) (ha : a = (2^k) % 10) (hA : 10 * A + a = 2^k) : 6 ∣ (a * A) :=
by
  -- The proof is intentionally omitted, marked with sorry.
  sorry

end six_divides_aA_l85_85478


namespace largest_divisor_n_l85_85529

theorem largest_divisor_n (n : ℕ) (h₁ : n > 0) (h₂ : 650 ∣ n^3) : 130 ∣ n :=
sorry

end largest_divisor_n_l85_85529


namespace value_of_ln_2_l85_85580

noncomputable def f (x : ℝ) : ℝ := sorry

axiom mono_increasing : ∀ x y : ℝ, x ≤ y → f x ≤ f y

axiom functional_eq : ∀ x : ℝ, f (f x - exp x) = Real.exp 1 + 1

theorem value_of_ln_2 : f (Real.log 2) = 3 := by sorry

end value_of_ln_2_l85_85580


namespace like_term_l85_85613

theorem like_term (a : ℝ) : ∃ (a : ℝ), a * x ^ 5 * y ^ 3 = a * x ^ 5 * y ^ 3 :=
by sorry

end like_term_l85_85613


namespace total_shaded_area_l85_85004

theorem total_shaded_area (S T : ℝ) (h1 : 16 / S = 4) (h2 : S / T = 4) : 
    S^2 + 16 * T^2 = 32 := 
by {
    sorry
}

end total_shaded_area_l85_85004


namespace correct_option_D_l85_85703

theorem correct_option_D : (-8) / (-4) = 8 / 4 := 
by
  exact (rfl

end correct_option_D_l85_85703


namespace area_triang_DEF_l85_85406

theorem area_triang_DEF (ABC : Type)
  [triangle ABC]
  (D : point)
  (midpoint : is_midpoint D (BC : segment))
  (E F : point)
  (H1 : line_contains (AB : line) E)
  (H2 : line_contains (AB : line) F)
  (ratios : AE / AB = 1 / 3 ∧ BF / AB = 1 / 4)
  (area_ABC : area ABC = 2018) :
  area DEF = 2018 / 24 :=
by sorry

end area_triang_DEF_l85_85406


namespace calculate_perimeter_l85_85634

-- Define the given conditions
def region_area := 512 -- area of region in square centimeters
def num_squares := 8 -- number of squares
def arrangement := (rows := 2, columns := 4) -- arrangement of squares

-- Define the derived values
def area_per_square := region_area / num_squares -- area of one square
def side_length := (area_per_square.toFloat).sqrt -- side length of one square

-- Define the perimeter calculation
def perimeter_of_region := 2 * (arrangement.rows * side_length + arrangement.columns * side_length)

-- The goal to prove
theorem calculate_perimeter (region_area = 512) (num_squares = 8) (arrangement = (2, 4)) : perimeter_of_region = 128 :=
by
  sorry

end calculate_perimeter_l85_85634


namespace shooting_levels_comparison_l85_85310

noncomputable def shooterA_distribution := [(8, 0.2), (9, 0.6), (10, 0.2)]
noncomputable def shooterB_distribution := [(8, 0.4), (9, 0.2), (10, 0.4)]

noncomputable def E (dist: List (ℕ × ℝ)) : ℝ :=
  dist.foldl (λ acc x, acc + x.1 * x.2) 0

noncomputable def V (dist: List (ℕ × ℝ)) : ℝ :=
  let mean := E dist
  dist.foldl (λ acc x, acc + (x.1 - mean)^2 * x.2) 0

theorem shooting_levels_comparison :
  V shooterA_distribution < V shooterB_distribution :=
by
  sorry

end shooting_levels_comparison_l85_85310


namespace grounded_days_for_lying_l85_85019

def extra_days_per_grade_below_b : ℕ := 3
def grades_below_b : ℕ := 4
def total_days_grounded : ℕ := 26

theorem grounded_days_for_lying : 
  (total_days_grounded - (grades_below_b * extra_days_per_grade_below_b) = 14) := 
by 
  sorry

end grounded_days_for_lying_l85_85019


namespace total_potatoes_l85_85728

theorem total_potatoes (monday_to_friday_potatoes : ℕ) (double_potatoes : ℕ) 
(lunch_potatoes_mon_fri : ℕ) (lunch_potatoes_weekend : ℕ)
(dinner_potatoes_mon_fri : ℕ) (dinner_potatoes_weekend : ℕ)
(h1 : monday_to_friday_potatoes = 5)
(h2 : double_potatoes = 10)
(h3 : lunch_potatoes_mon_fri = 25)
(h4 : lunch_potatoes_weekend = 20)
(h5 : dinner_potatoes_mon_fri = 40)
(h6 : dinner_potatoes_weekend = 26)
  : monday_to_friday_potatoes * 5 + double_potatoes * 2 + dinner_potatoes_mon_fri * 5 + (double_potatoes + 3) * 2 = 111 := 
sorry

end total_potatoes_l85_85728


namespace trig_identity_simplification_l85_85108

theorem trig_identity_simplification (θ : ℝ) (hθ : θ = 15 * Real.pi / 180) :
  (Real.sqrt 3 / 2 - Real.sqrt 3 * (Real.sin θ) ^ 2) = 3 / 4 := 
by sorry

end trig_identity_simplification_l85_85108


namespace kathleen_paint_time_l85_85924

-- Define the conditions as Lean definitions
def kathleen_rate (k : ℝ) := 1 / k
def anthony_rate : ℝ := 1 / 4
def combined_time := 3.428571428571429
def rooms_painted := 2

-- Define the theorem to prove
theorem kathleen_paint_time :
  ∃ (k : ℝ), (kathleen_rate k + anthony_rate) * combined_time = rooms_painted → k = 3 := by
  -- Proof will be added here
  sorry

end kathleen_paint_time_l85_85924


namespace relatively_prime_integers_product_l85_85252

theorem relatively_prime_integers_product (n : ℕ) (hpos : 0 < n) :
  ∃ (k : Fin (n + 1) → ℕ), (∀ i j : Fin (n + 1), i ≠ j → Nat.gcd (k i) (k j) = 1) ∧
    (∀ i : Fin (n + 1), 1 < k i) ∧
    (∃ x y : ℕ, k 0 * k 1 * ... * k n - 1 = x * (x + 1)) := sorry

end relatively_prime_integers_product_l85_85252


namespace base5_sum_correct_l85_85446

theorem base5_sum_correct : 
  ∀ (a b c : ℕ), 
  (a = 123) → (b = 432) → (c = 214) → 
  let sum_base10 := (5^0) * (a % 10) + (5^1) * ((a / 10) % 10) + (5^2) * (a / 100) +
                    (5^0) * (b % 10) + (5^1) * ((b / 10) % 10) + (5^2) * (b / 100) +
                    (5^0) * (c % 10) + (5^1) * ((c / 10) % 10) + (5^2) * (c / 100) in
  let sum_base5 :=  (sum_base10 % 5) + 
                    (5 * ((sum_base10 / 5) % 5)) + 
                    (5^2 * ((sum_base10 / (5^2)) % 5)) +
                    (5^3 * (sum_base10 / (5^3))) in
  sum_base5 = 1 * (5^3) + 3 * (5^2) + 2 * (5^1) + 4 * (5^0) := 
by
  intros a b c ha hb hc
  sorry

end base5_sum_correct_l85_85446


namespace third_function_is_inverse_l85_85306

-- Assume the existence of the function f
variables {α β : Type*} [Nonempty α] [Nonempty β] [TopologicalSpace α] [TopologicalSpace β] {f : α → β}

-- Assume f is invertible and its inverse is called f_inv
noncomputable def f_inv (y : β) : α := sorry
axiom f_left_inv : ∀ (x : α), f_inv (f x) = x
axiom f_right_inv : ∀ (y : β), f (f_inv y) = y

-- Define the inverse function of y = 2 - f(-x)
noncomputable def g_inv (y : β) : α := sorry

-- The function y = -f⁻¹(2 - x)
def third_function (x : β) : α := -f_inv (2 - x)

-- The function y = 2 - f(-x)
def inverse_candidate (x : β) : α := 2 - f (-x)

-- Prove that the third function is the inverse of the candidate function
theorem third_function_is_inverse :
  ∀ x, third_function (inverse_candidate x) = x ∧ 
       inverse_candidate (third_function x) = x :=
by
  -- (Steps not required)
  sorry

end third_function_is_inverse_l85_85306


namespace length_of_NC_l85_85027

noncomputable def semicircle_radius (AB : ℝ) : ℝ := AB / 2

theorem length_of_NC : 
  ∀ (AB CD AN NB N M C NC : ℝ),
    AB = 10 ∧ AB = CD ∧ AN = NB ∧ AN + NB = AB ∧ M = N ∧ AB / 2 = semicircle_radius AB ∧ (NC^2 + semicircle_radius AB^2 = (2 * semicircle_radius AB)^2) →
    NC = 5 * Real.sqrt 3 := 
by 
  intros AB CD AN NB N M C NC h 
  rcases h with ⟨hAB, hCD, hAN, hSumAN, hMN, hRadius, hPythag⟩
  sorry

end length_of_NC_l85_85027


namespace mul_value_proof_l85_85989

theorem mul_value_proof :
  ∃ x : ℝ, (8.9 - x = 3.1) ∧ ((x * 3.1) * 2.5 = 44.95) :=
by
  sorry

end mul_value_proof_l85_85989


namespace mass_percentage_Ca_in_mixture_l85_85050

open_locale big_operators

-- Definitions for molar masses of elements
def molar_mass_Ca : ℝ := 40.08
def molar_mass_O : ℝ := 16.00
def molar_mass_H : ℝ := 1.01
def molar_mass_C : ℝ := 12.01

-- Definitions for molar masses of compounds
def molar_mass_CaOH2 : ℝ := molar_mass_Ca + 2 * molar_mass_O + 2 * molar_mass_H
def molar_mass_CaCO3 : ℝ := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O

-- Given masses of compounds
def mass_CaOH2 : ℝ := 25.0
def mass_CaCO3 : ℝ := 50.0

-- Total mass of the mixture
def total_mass_mixture : ℝ := mass_CaOH2 + mass_CaCO3

-- Calculate the mass of Ca in each compound
def mass_Ca_in_CaOH2 : ℝ := (molar_mass_Ca / molar_mass_CaOH2) * mass_CaOH2
def mass_Ca_in_CaCO3 : ℝ := (molar_mass_Ca / molar_mass_CaCO3) * mass_CaCO3

-- Total mass of Ca
def total_mass_Ca : ℝ := mass_Ca_in_CaOH2 + mass_Ca_in_CaCO3

-- Mass percentage of Ca in the mixture
def mass_percentage_Ca : ℝ := (total_mass_Ca / total_mass_mixture) * 100

theorem mass_percentage_Ca_in_mixture :
  mass_percentage_Ca = 44.75 := by
  sorry

end mass_percentage_Ca_in_mixture_l85_85050


namespace spherical_to_rectangular_coords_l85_85744

-- Given conditions
variables (ρ θ φ : ℝ)
variables (x y z : ℝ) (h1 : x = ρ * sin φ * cos θ) (h2 : y = ρ * sin φ * sin θ) (h3 : z = ρ * cos φ)

-- Equivalent proof problem statement
theorem spherical_to_rectangular_coords (h1 : x = ρ * sin φ * cos θ) (h2 : y = ρ * sin φ * sin θ) (h3 : z = ρ * cos φ) :
  (-ρ * sin φ * cos θ, -ρ * sin φ * sin θ, ρ * cos φ) = (-3, -8, -6) :=
sorry

end spherical_to_rectangular_coords_l85_85744


namespace min_value_expression_l85_85051

theorem min_value_expression :
  (∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ ((tan x + cot x)^2 + (sin x + cos x)^2 = 5)) ∧ 
  ∀ y : ℝ, (0 < y ∧ y < π / 2) → ((tan y + cot y)^2 + (sin y + cos y)^2) ≥ 5 :=
sorry

end min_value_expression_l85_85051


namespace lana_picked_37_roses_l85_85927

def total_flowers_picked (used : ℕ) (extra : ℕ) := used + extra

def picked_roses (total : ℕ) (tulips : ℕ) := total - tulips

theorem lana_picked_37_roses :
    ∀ (tulips used extra : ℕ), tulips = 36 → used = 70 → extra = 3 → 
    picked_roses (total_flowers_picked used extra) tulips = 37 :=
by
  intros tulips used extra htulips husd hextra
  sorry

end lana_picked_37_roses_l85_85927


namespace problem_1_part1_problem_1_part2_l85_85494

def f (x : ℝ) : ℝ := cos x ^ 2 + sqrt 3 * sin x * cos x + 1

theorem problem_1_part1 :
  (∃ p, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧ p = π) ∧
  (∃ M, (∀ x : ℝ, f x ≤ M) ∧ M = 5 / 2) ∧
  (∃ m, (∀ x : ℝ, f x ≥ m) ∧ m = 1 / 2) :=
by
  -- proof to be filled in
  sorry

theorem problem_1_part2 : 
  (∀ x : ℝ, (0 < x ∧ x ≤ π / 6 ∨ 2 * π / 3 ≤ x ∧ x < π) → ∀ y, (0 < y ∧ y <= π) → x < y → f x < f y) ∧
  (∀ x : ℝ, (π / 6 ≤ x ∧ x ≤ 2 * π / 3) → ∀ y, (0 < y ∧ y <= π) → x < y → f x > f y) :=
by
  -- proof to be filled in
  sorry

end problem_1_part1_problem_1_part2_l85_85494


namespace maximum_minus_minimum_value_l85_85581

open Real

theorem maximum_minus_minimum_value (x y k : ℝ) (h₀ : k > 0) :
  let m := 0 in
  let M := 1 in
  M - m = 1 :=
by
  sorry

end maximum_minus_minimum_value_l85_85581


namespace exponentiation_division_l85_85778

variable {a : ℝ} (h1 : (a^2)^3 = a^6) (h2 : a^6 / a^2 = a^4)

theorem exponentiation_division : (a^2)^3 / a^2 = a^4 := 
by 
  sorry

end exponentiation_division_l85_85778


namespace sum_inequality_l85_85260

theorem sum_inequality (n : ℕ) (h : 0 < n) : 
  ∑ k in finset.range (n + 1), (2 * n + 1 - k) / (k + 1) ^ k ≤ 2 ^ n := 
by {
  sorry,
}

end sum_inequality_l85_85260


namespace ratio_of_cows_to_horses_l85_85773

-- Step 1: Define the number of cows and horses
def cows := 21
def horses := 6

-- Step 2: Define the greatest common divisor (for context)
def gcd (a b : ℕ) : ℕ := sorry  -- Lean has built-in gcd functionality, consider this symbolic for context

-- Step 3: State the ratio reduction equivalence
theorem ratio_of_cows_to_horses (h_gcd: gcd 21 6 = 3) : (21 / 6) = (7 / 2) :=
by
  sorry

end ratio_of_cows_to_horses_l85_85773


namespace path_length_of_M_l85_85769

theorem path_length_of_M (BC : ℝ) (BC_eq : BC = 4 / real.pi) :
  let M_path_length := 2 in
  M_path_length = 2 :=
by
  -- Definitions have been used directly from problem conditions
  -- The solution steps are not considered
  sorry

end path_length_of_M_l85_85769


namespace no_n_nat_powers_l85_85980

theorem no_n_nat_powers (n : ℕ) : ∀ n : ℕ, ¬∃ m k : ℕ, k ≥ 2 ∧ n * (n + 1) = m ^ k := 
by 
  sorry

end no_n_nat_powers_l85_85980


namespace problem_statement_l85_85784

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

-- Lean 4 statements for the proof problem
theorem problem_statement :
  (f = λ x, 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  ¬ (∀ k, f (x + 2 * Real.pi * k) = f x) ∧
  (∀ x, f (x) = f (-2 * (-x - Real.pi / 6) / 2 + Real.pi / 3)) ∧
  ¬ (∀ x, f (2 * (x - 5 * Real.pi / 6) / 2 + Real.pi / 3) = f (2 * (x + 5 * Real.pi / 6) / 2 + Real.pi / 3))
:= by sorry

end problem_statement_l85_85784


namespace find_m_in_arith_seq_l85_85472

noncomputable def arith_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem find_m_in_arith_seq (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0) 
  (h_seq : arith_seq a d) 
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32) 
  (h_am : ∃ m, a m = 8) : 
  ∃ m, m = 8 := 
sorry

end find_m_in_arith_seq_l85_85472


namespace find_a_l85_85882

variable (b : ℝ) (x a : ℝ)

theorem find_a 
  (hb : b > 1)
  (hsinx_pos : Real.sin x > 0)
  (hcosx_pos : Real.cos x > 0)
  (hsinx : Real.sin x = b^(-a))
  (ha : a > 0) :
  Real.cos x = b^(-2 * a) → 
  a = -(1 / (2 * Real.log b)) * Real.log ((Real.sqrt 5 - 1) / 2) :=
begin
  sorry
end

end find_a_l85_85882


namespace lake_crystal_frogs_percentage_l85_85562

noncomputable def percentage_fewer_frogs (frogs_in_lassie_lake total_frogs : ℕ) : ℕ :=
  let P := (total_frogs - frogs_in_lassie_lake) * 100 / frogs_in_lassie_lake
  P

theorem lake_crystal_frogs_percentage :
  let frogs_in_lassie_lake := 45
  let total_frogs := 81
  percentage_fewer_frogs frogs_in_lassie_lake total_frogs = 20 :=
by
  sorry

end lake_crystal_frogs_percentage_l85_85562


namespace total_sheep_l85_85754

variable (x y : ℕ)
/-- Initial condition: After one ram runs away, the ratio of rams to ewes is 7:5. -/
def initial_ratio (x y : ℕ) : Prop := 5 * (x - 1) = 7 * y
/-- Second condition: After the ram returns and one ewe runs away, the ratio of rams to ewes is 5:3. -/
def second_ratio (x y : ℕ) : Prop := 3 * x = 5 * (y - 1)
/-- The total number of sheep in the flock initially is 25. -/
theorem total_sheep (x y : ℕ) 
  (h1 : initial_ratio x y) 
  (h2 : second_ratio x y) : 
  x + y = 25 := 
by sorry

end total_sheep_l85_85754


namespace area_under_abs_sin_l85_85270

noncomputable def f (x : ℝ) : ℝ := |Real.sin x|

theorem area_under_abs_sin : 
  ∫ x in -Real.pi..Real.pi, f x = 4 :=
by
  sorry

end area_under_abs_sin_l85_85270


namespace probability_prime_and_multiple_of_11_l85_85610

theorem probability_prime_and_multiple_of_11 (h1 : 11.prime) (h2 : 11 ∣ 11) :
  (1 : ℚ) / 100 = 1 / 100 :=
by
  -- Conditions that are given in the problem
  have h_total_cards : 100 > 0 := by norm_num
  -- Card 11 is the only prime and multiple of 11 in the range 1-100
  have h_unique_card : ∃ (n : ℕ), n = 11 := ⟨11, rfl⟩
  -- Probability calculation
  sorry -- proof is not required

end probability_prime_and_multiple_of_11_l85_85610


namespace graduation_ceremony_chairs_l85_85173

theorem graduation_ceremony_chairs (g p t a : ℕ) 
  (h_g : g = 50) 
  (h_p : p = 2 * g) 
  (h_t : t = 20) 
  (h_a : a = t / 2) : 
  g + p + t + a = 180 :=
by
  sorry

end graduation_ceremony_chairs_l85_85173


namespace number_of_children_at_reunion_l85_85013

theorem number_of_children_at_reunion (A C : ℕ) 
    (h1 : 3 * A = C)
    (h2 : 2 * A / 3 = 10) : 
  C = 45 :=
by
  sorry

end number_of_children_at_reunion_l85_85013


namespace part_a_part_b_l85_85567

open Classical

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def isConcatenationForm (n : ℕ) : Prop :=
  ∃ A : ℕ, let digitCount := (Nat.log10 (A + 1)) + 1 in n = (A * (10 ^ digitCount) + A)

def S : Set ℕ := { n | isPerfectSquare n ∧ isConcatenationForm n }

theorem part_a : ∃ (infSeq : ℕ → ℕ), ∀ n, infSeq n ∈ S :=
sorry

theorem part_b : ∃ f : ℕ → ℕ → ℕ, (∀ a b c, a ∈ S → b ∈ S → c ∈ S → a ∣ c → b ∣ c → f a b ∈ S → f a b ∣ c) :=
sorry

end part_a_part_b_l85_85567


namespace sum_of_products_lt_one_l85_85461

noncomputable def S (a : Fin 1959 → ℝ) : ℝ :=
  ∑ (k : Finset (Fin 1959)) in Finset.choose 1959 1000, k.prod (λ i, a i)

theorem sum_of_products_lt_one {a : Fin 1959 → ℝ}
  (hpos : ∀ i, 0 < a i) (hsum : ∑ i, a i = 1) : S a < 1 :=
sorry

end sum_of_products_lt_one_l85_85461


namespace resisting_force_of_wood_l85_85357

noncomputable def bullet_speed := 500 -- in m/s
noncomputable def bullet_mass := 0.015 -- in kg
noncomputable def penetration_depth := 0.45 -- in m
noncomputable def acceleration_due_to_gravity := 9.81 -- in m/s²

theorem resisting_force_of_wood :
  let v := bullet_speed
      m := bullet_mass
      d := penetration_depth
      g := acceleration_due_to_gravity
      KE := (1/2 : ℝ) * m * v^2
      F := KE / d
      F_kgf := F / g
  in F_kgf = 424.74 := 
by
  sorry

end resisting_force_of_wood_l85_85357


namespace g_at_8_equals_minus_30_l85_85644

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_8_equals_minus_30 :
  (∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2) →
  g 8 = -30 :=
by
  intro h
  sorry

end g_at_8_equals_minus_30_l85_85644


namespace trig_identity_proof_l85_85335

theorem trig_identity_proof :
  (sin (24 * real.pi / 180) * cos (6 * real.pi / 180) - sin (6 * real.pi / 180) * cos (66 * real.pi / 180)) / 
  (sin (21 * real.pi / 180) * cos (39 * real.pi / 180) - sin (39 * real.pi / 180) * cos (21 * real.pi / 180)) = -1 :=
by
  sorry

end trig_identity_proof_l85_85335


namespace largest_incomposable_exists_l85_85547

noncomputable def largest_incomposable_amount (n : ℕ) : ℤ :=
  2 * 4^(n + 1) - 3^(n + 2)

theorem largest_incomposable_exists (n : ℕ) : ∃ s : ℕ, 
  (s = largest_incomposable_amount n) ∧
  ¬ (∃ k : ℕ, ∃ a b : list ℕ, 
    (∀ x ∈ a, x ∈ [0, 1, 2, ..., 3^n, 3^(n-1) * 4, 3^(n-2) * 4^2, ..., 4^n]) ∧
    (∀ x ∈ b, x ∈ [0, 1, 2, ..., 4^n]) ∧
    k = a.sum * 3 ^ n + b.sum * 4^n ∧ k = s) :=
sorry

end largest_incomposable_exists_l85_85547


namespace identify_counterfeit_coin_l85_85009

constant Coin : Type
constant Mass : Type
constant is_genuine : Coin -> Prop -- Coin is genuine if it has the same mass as all genuine coins
constant is_counterfeit : Coin -> Prop -- Coin is counterfeit if it has a different mass
constant mass_of : Coin -> Mass -- Function that assigns a mass to a coin
constant balance_scale : Coin -> Coin -> Prop -- Predicate that states if two coins balance on the scale

constant C1 C2 C3 C4 C5 : Coin -- Five coins

axiom one_counterfeit : is_counterfeit C1 ∨ is_counterfeit C2 ∨ is_counterfeit C3 ∨ is_counterfeit C4 ∨ is_counterfeit C5
axiom unique_counterfeit : (is_counterfeit C1 → ¬ is_counterfeit C2 ∧ ¬ is_counterfeit C3 ∧ ¬ is_counterfeit C4 ∧ ¬ is_counterfeit C5) ∧ 
                           (is_counterfeit C2 → ¬ is_counterfeit C1 ∧ ¬ is_counterfeit C3 ∧ ¬ is_counterfeit C4 ∧ ¬ is_counterfeit C5) ∧ 
                           (is_counterfeit C3 → ¬ is_counterfeit C1 ∧ ¬ is_counterfeit C2 ∧ ¬ is_counterfeit C4 ∧ ¬ is_counterfeit C5) ∧
                           (is_counterfeit C4 → ¬ is_counterfeit C1 ∧ ¬ is_counterfeit C2 ∧ ¬ is_counterfeit C3 ∧ ¬ is_counterfeit C5) ∧
                           (is_counterfeit C5 → ¬ is_counterfeit C1 ∧ ¬ is_counterfeit C2 ∧ ¬ is_counterfeit C3 ∧ ¬ is_counterfeit C4)
axiom genuine_coins_same_mass : ∀ {c1 c2 : Coin}, is_genuine c1 → is_genuine c2 → mass_of c1 = mass_of c2
axiom counterfeit_different_mass : ∀ c, is_counterfeit c → ∀ g, is_genuine g → mass_of c ≠ mass_of g
axiom balance_behavior : ∀ {c1 c2 : Coin}, balance_scale c1 c2 ↔ mass_of c1 = mass_of c2

theorem identify_counterfeit_coin : ∃ c, (is_counterfeit c) ∧ (∀ g, is_genuine g → mass_of c ≠ mass_of g) :=
sorry

end identify_counterfeit_coin_l85_85009


namespace evaluate_i_powers_l85_85431

theorem evaluate_i_powers : (complex.I ^ 45) + (complex.I ^ 123) = 0 := by 
  sorry

end evaluate_i_powers_l85_85431


namespace longest_possible_height_l85_85283

theorem longest_possible_height (a b c : ℕ) (ha : a = 3 * c) (hb : b * 4 = 12 * c) (h_tri : a - c < b) (h_unequal : ¬(a = c)) :
  ∃ x : ℕ, (4 < x ∧ x < 6) ∧ x = 5 :=
by
  sorry

end longest_possible_height_l85_85283


namespace incorrect_calculation_l85_85705

theorem incorrect_calculation :
  ¬ (real.sqrt 8 / real.sqrt 2 = real.sqrt 2) :=
by sorry

end incorrect_calculation_l85_85705


namespace convert_spherical_to_rectangular_correct_l85_85037

-- Define the spherical coordinates and the conversion formulas.
def ρ : ℝ := -5
def θ : ℝ := (7 * Real.pi) / 4
def φ : ℝ := Real.pi / 3

-- Define the conversion to rectangular coordinates.
def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

-- Define the expected rectangular coordinates.
def expected_rectangular : ℝ × ℝ × ℝ :=
  (- (5 * Real.sqrt 6) / 4, - (5 * Real.sqrt 6) / 4, -5 / 2)

-- State the theorem: the spherical point converts to the expected rectangular coordinates.
theorem convert_spherical_to_rectangular_correct :
  spherical_to_rectangular ρ θ φ = expected_rectangular :=
  sorry

end convert_spherical_to_rectangular_correct_l85_85037


namespace incorrect_statement_tripling_triangle_altitude_l85_85245

theorem incorrect_statement_tripling_triangle_altitude :
  ∀ (b h : ℝ), (3 * h * b / 2 = 4 * (h * b / 2)) → false :=
by
  assume b h
  sorry

end incorrect_statement_tripling_triangle_altitude_l85_85245


namespace solution_of_equation_l85_85986

def solve_equation (x : ℚ) : Prop := 
  (x^2 + 3 * x + 4) / (x + 5) = x + 6

theorem solution_of_equation : solve_equation (-13/4) := 
by
  sorry

end solution_of_equation_l85_85986


namespace lucy_clean_aquariums_l85_85600

-- Define the conditions
def lucy_cleaning_rate : ℝ := 2 / 3 -- Lucy's rate of cleaning aquariums (aquariums per hour)
def lucy_work_hours : ℕ := 24 -- Lucy's working hours this week

-- Define the goal
theorem lucy_clean_aquariums : (lucy_cleaning_rate * lucy_work_hours) = 16 := by
  sorry

end lucy_clean_aquariums_l85_85600


namespace each_friend_gets_9_marbles_l85_85237

/-- Definition of Lori's total marbles -/
def total_marbles : ℕ := 60

/-- Definition of the percentage of marbles shared -/
def percentage_shared : ℝ := 0.75

/-- Definition of the number of friends -/
def number_of_friends : ℕ := 5

/-- Definition of the marbles each friend gets -/
def marbles_each_friend_gets (total: ℕ) (percent: ℝ) (friends: ℕ) : ℕ :=
  (percent * total).toNat / friends

/-- Theorem stating the actual marbles each friend gets -/
theorem each_friend_gets_9_marbles :
  marbles_each_friend_gets total_marbles percentage_shared number_of_friends = 9 :=
sorry

end each_friend_gets_9_marbles_l85_85237


namespace tangents_to_curve_l85_85105

theorem tangents_to_curve (P : ℝ × ℝ) (h : P = (1, 1) ∨ P = (0, 0) ∨ P = (0, 1) ∨ P = (-2, -1)) :
  (∃! m : ℝ, tangent (fun x => x^3) P m) ↔ P = (1, 1) :=
by
  sorry

end tangents_to_curve_l85_85105


namespace profit_margin_difference_eq_71640_l85_85186

def R (x : ℕ) : ℕ := 3000 * x - 20 * x^2
def C (x : ℕ) : ℕ := 500 * x + 4000
def P (x : ℕ) : ℕ := R x - C x
def MP (x : ℕ) : ℕ := P (x + 1) - P x

-- Conditions
axiom N_pos : ℕ → Prop
axiom company_produces_no_more_than_100 (x : ℕ) : x ≤ 100

-- Problem statement
theorem profit_margin_difference_eq_71640 :
  ∀ (x : ℕ), N_pos x → company_produces_no_more_than_100 x → 
  (let maxP := max (P 62) (P 63) in
   let maxMP := MP 0 in
   maxP - maxMP = 71640) :=
by
  intros x H1 H2
  let maxP := max (P 62) (P 63)
  let maxMP := MP 0
  have h1 : P 62 = -20 * 62^2 + 2500 * 62 - 4000 := by sorry
  have h2 : P 63 = -20 * 63^2 + 2500 * 63 - 4000 := by sorry
  have maxP_eq : maxP = 74120 := by sorry
  have maxMP_eq : maxMP = 2480 := by sorry
  exact calc
    maxP - maxMP = 74120 - 2480 := by rw [maxP_eq, maxMP_eq]
    ... = 71640

end profit_margin_difference_eq_71640_l85_85186


namespace smallest_positive_integer_n_l85_85320

theorem smallest_positive_integer_n (n : ℕ) (h : 527 * n ≡ 1083 * n [MOD 30]) : n = 2 :=
sorry

end smallest_positive_integer_n_l85_85320


namespace conjugate_div_modulus_l85_85821

-- Given conditions
def z : ℂ := 4 + 3 * complex.i

-- The statement we want to prove
theorem conjugate_div_modulus :
  (complex.conj z) / complex.abs z = 4 / 5 - (3 / 5) * complex.i :=
by
  sorry

end conjugate_div_modulus_l85_85821


namespace perpendicular_lines_condition_l85_85638

theorem perpendicular_lines_condition (a : ℝ) :
  (a = 2) ↔ (∃ m1 m2 : ℝ, (m1 = -1/(4 : ℝ)) ∧ (m2 = (4 : ℝ)) ∧ (m1 * m2 = -1)) :=
by sorry

end perpendicular_lines_condition_l85_85638


namespace skateboard_total_distance_l85_85002

theorem skateboard_total_distance :
  ∀ (a₁ d : ℤ) (n : ℕ), a₁ = 48 → d = -12 → n = 5 →
  (∀ i, (0 ≤ i ∧ i < n) → a₁ + (i : ℤ) * d > 0) →
  (∑ i in Finset.range n, a₁ + (i : ℤ) * d) = 120 :=
begin
  intros a₁ d n h₁ h₂ h₃ h₄,
  rw [h₁, h₂, h₃], -- Substitute given values
  sorry
end

end skateboard_total_distance_l85_85002


namespace lucy_clean_aquariums_l85_85599

-- Define the conditions
def lucy_cleaning_rate : ℝ := 2 / 3 -- Lucy's rate of cleaning aquariums (aquariums per hour)
def lucy_work_hours : ℕ := 24 -- Lucy's working hours this week

-- Define the goal
theorem lucy_clean_aquariums : (lucy_cleaning_rate * lucy_work_hours) = 16 := by
  sorry

end lucy_clean_aquariums_l85_85599


namespace zou_wins_5_out_of_6_l85_85709

theorem zou_wins_5_out_of_6 (P_win_after_win P_win_after_loss : ℚ)
  (h_win : P_win_after_win = 2/3) (h_loss : P_win_after_loss = 1/3) :
  let prob := 4 * (1 / 3) * (2 / 3) ^ 4 + (2 / 3) ^ 4 * (1 / 3) in
  prob.denom = 243 ∧ prob.num = 80 →
  prob.num + prob.denom = 323 :=
by
  sorry

end zou_wins_5_out_of_6_l85_85709


namespace sum_fifth_roots_inverse_magnitude_l85_85780

open Complex

noncomputable def fifth_roots_of_i := 
  { z : ℂ | z^5 = I }

theorem sum_fifth_roots_inverse_magnitude :
  ∑ z in fifth_roots_of_i, (1 / |I - z|) = 5 * (sqrt 2 / sqrt (6 - sqrt 5)) :=
by
  -- proof goes here
  sorry

end sum_fifth_roots_inverse_magnitude_l85_85780


namespace general_formula_for_arithmetic_sequence_l85_85836

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m k, (a n) * (a k) = (a m) ^ 2

theorem general_formula_for_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_not_zero : ∃ d, d ≠ 0 ∧ ∀ n, a (n + 1) - a n = d)
  (a1 : a 1 = 1)
  (h_geom : geometric_sequence (λ n, a (if n = 0 then 1 else if n = 1 then 3 else if n = 2 then 9 else 1))) : 
  ∀ n, a n = n := 
begin 
  sorry 
end

end general_formula_for_arithmetic_sequence_l85_85836


namespace yardwork_payment_l85_85815

theorem yardwork_payment :
  let earnings := [15, 20, 25, 40]
  let total_earnings := List.sum earnings
  let equal_share := total_earnings / earnings.length
  let high_earner := 40
  high_earner - equal_share = 15 :=
by
  sorry

end yardwork_payment_l85_85815


namespace definite_integral_value_l85_85340

/-- 
Problem: Calculate the definite integral 
∫_{π/4}^{arccos (1 / sqrt 26)} (36 / ((6 - tan x) * sin (2 * x))) dx 
given the substitution t = tan x.
Expected Result: 6 * ln 5
-/
theorem definite_integral_value :
  ∫ x in real.pi / 4..real.arccos (1 / real.sqrt 26), 
    (36 / ((6 - real.tan x) * real.sin (2 * x))) = 6 * real.log 5 := 
sorry

end definite_integral_value_l85_85340


namespace subset_implies_range_a_intersection_implies_range_a_l85_85125

noncomputable def setA : Set ℝ := {x | -1 < x ∧ x < 2}
noncomputable def setB (a : ℝ) : Set ℝ := {x | 2 * a - 1 < x ∧ x < 2 * a + 3}

theorem subset_implies_range_a (a : ℝ) : (setA ⊆ setB a) → (-1/2 ≤ a ∧ a ≤ 0) :=
by
  sorry

theorem intersection_implies_range_a (a : ℝ) : (setA ∩ setB a = ∅) → (a ≤ -2 ∨ a ≥ 3/2) :=
by
  sorry

end subset_implies_range_a_intersection_implies_range_a_l85_85125


namespace new_cross_maintains_nine_diamonds_l85_85249

/-- 
Given the initial arrangement and the modified total count constraints,
prove that the new arrangement maintains a total count of 9 diamonds 
despite the removal of 2 diamonds.
-/
theorem new_cross_maintains_nine_diamonds
  (original_count : ℕ)
  (diamonds_removed : ℕ)
  (new_count : ℕ) :
  original_count = 9 →
  diamonds_removed = 2 →
  new_count = 9 :=
by
  intro h1 h2
  have : original_count - diamonds_removed = new_count,
    -- Proof to be provided
    sorry
  exact Eq.trans (Nat.sub_add_cancel h2) h1

end new_cross_maintains_nine_diamonds_l85_85249


namespace quadratic_completing_square_l85_85623

theorem quadratic_completing_square :
  ∃ p q : ℝ, (∀ x : ℝ, 4 * x^2 + 8 * x - 448 = 0 ↔ (x + p)^2 = q) ∧ q = 113 :=
by
  let p := 1
  let q := 113
  use p, q
  split
  { intro h
    simp
    sorry }
  { refl }

end quadratic_completing_square_l85_85623


namespace cake_sector_chord_length_l85_85731

noncomputable def sector_longest_chord_square (d : ℝ) (n : ℕ) : ℝ :=
  let r := d / 2
  let theta := (360 : ℝ) / n
  let chord_length := 2 * r * Real.sin (theta / 2 * Real.pi / 180)
  chord_length ^ 2

theorem cake_sector_chord_length :
  sector_longest_chord_square 18 5 = 111.9473 := by
  sorry

end cake_sector_chord_length_l85_85731


namespace circle_equation_and_lines_l85_85074

noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (6, 2)
noncomputable def B : ℝ × ℝ := (4, 4)
noncomputable def C_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 10

structure Line (κ β: ℝ) where
  passes_through : ℝ × ℝ → Prop
  definition : Prop

def line_passes_through_point (κ β : ℝ) (p : ℝ × ℝ) : Prop := p.2 = κ * p.1 + β

theorem circle_equation_and_lines : 
  (∀ p : ℝ × ℝ, p = O ∨ p = A ∨ p = B → C_eq p.1 p.2) ∧
  ((∀ p : ℝ × ℝ, line_passes_through_point 0 2 p → C_eq 2 6 ∧ (∃ x1 x2 y : ℝ, C_eq x1 y ∧ C_eq x2 y ∧ ((x1 - x2)^2 + (y - y)^2) = 4)) ∧
   (∀ p : ℝ × ℝ, line_passes_through_point (-7 / 3) (32 / 3) p → C_eq 2 6 ∧ (∃ x1 x2 y : ℝ, C_eq x1 y ∧ C_eq x2 y ∧ ((x1 - x2)^2 + (y - y)^2) = 4))) :=
by 
  sorry

end circle_equation_and_lines_l85_85074


namespace correct_divisor_l85_85171

theorem correct_divisor (D : ℕ) (X : ℕ) (H1 : X = 70 * (D + 12)) (H2 : X = 40 * D) : D = 28 := 
by 
  sorry

end correct_divisor_l85_85171


namespace projection_of_a_onto_b_is_sqrt_5_l85_85873

noncomputable def vector_a : ℝ × ℝ := (3, -1)
noncomputable def vector_b : ℝ × ℝ := (1, -2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def norm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def positive_projection (v1 v2 : ℝ × ℝ) : ℝ :=
  (dot_product v1 v2) / (norm v2)

theorem projection_of_a_onto_b_is_sqrt_5 :
  positive_projection vector_a vector_b = Real.sqrt 5 :=
by
  sorry

end projection_of_a_onto_b_is_sqrt_5_l85_85873


namespace unique_function_and_sum_calculate_n_times_s_l85_85218

def f : ℝ → ℝ := sorry

theorem unique_function_and_sum :
  (∀ x y z : ℝ, f (x^2 + 2 * f y) = x * f x + y * f z) →
  (∃! g : ℝ → ℝ, ∀ x, f x = g x) ∧ f 3 = 0 :=
sorry

theorem calculate_n_times_s :
  ∃ n s : ℕ, (∃! f : ℝ → ℝ, ∀ x y z : ℝ, f (x^2 + 2 * f y) = x * f x + y * f z) ∧ n = 1 ∧ s = (0 : ℝ) ∧ n * s = 0 :=
sorry

end unique_function_and_sum_calculate_n_times_s_l85_85218


namespace circle_passing_given_points_l85_85048

theorem circle_passing_given_points :
  ∃ (D E F : ℚ), (F = 0) ∧ (E = - (9 / 5)) ∧ (D = 19 / 5) ∧
  (∀ (x y : ℚ), x^2 + y^2 + D * x + E * y + F = 0 ↔ (x = 0 ∧ y = 0) ∨ (x = -2 ∧ y = 3) ∨ (x = -4 ∧ y = 1)) :=
by
  sorry

end circle_passing_given_points_l85_85048


namespace boiling_point_fahrenheit_l85_85312

-- Define the conditions as hypotheses
def boils_celsius : ℝ := 100
def melts_celsius : ℝ := 0
def melts_fahrenheit : ℝ := 32
def pot_temp_celsius : ℝ := 55
def pot_temp_fahrenheit : ℝ := 131

-- Theorem to prove the boiling point in Fahrenheit
theorem boiling_point_fahrenheit : ∀ (boils_celsius : ℝ) (melts_celsius : ℝ) (melts_fahrenheit : ℝ) 
                                    (pot_temp_celsius : ℝ) (pot_temp_fahrenheit : ℝ),
  boils_celsius = 100 →
  melts_celsius = 0 →
  melts_fahrenheit = 32 →
  pot_temp_celsius = 55 →
  pot_temp_fahrenheit = 131 →
  ∃ boils_fahrenheit : ℝ, boils_fahrenheit = 212 :=
by
  intros
  existsi 212
  sorry

end boiling_point_fahrenheit_l85_85312


namespace intersection_of_asymptotes_l85_85442

theorem intersection_of_asymptotes : ∃ (x y : ℝ), 
  (y = 1) ∧ (x = 2) ∧ (y = (x^2 - 4 * x + 3) / (x^2 - 4 * x + 4)) :=
by {
  use [2, 1],
  split,
  { exact rfl },
  split,
  { exact rfl },
  {
    simp,
    sorry -- Skipping the detailed proof steps as per instructions
  }
}

end intersection_of_asymptotes_l85_85442


namespace chord_line_equation_l85_85079

theorem chord_line_equation
  (A B : ℝ × ℝ)
  (h_circle : ∀ (p : ℝ × ℝ), (p = A ∨ p = B) → p.1^2 + p.2^2 + 2 * p.1 - 4 * p.2 + 1 = 0)
  (h_midpoint : ∀ (p : ℝ × ℝ), (p = ((A.1 + B.1)/2, (A.2 + B.2)/2)) → (p = (-2, 3))) :
  ∃ (m : ℝ), (∀ (p : ℝ × ℝ), (p = A ∨ p = B) → (p.2 - 3) = m * (p.1 + 2)) ∧ m = 1 :=
begin
  sorry
end

end chord_line_equation_l85_85079


namespace moon_speed_conversion_correct_l85_85713

-- Define the conversions
def kilometers_per_second_to_miles_per_hour (kmps : ℝ) : ℝ :=
  kmps * 0.621371 * 3600

-- Condition: The moon's speed
def moon_speed_kmps : ℝ := 1.02

-- Correct answer in miles per hour
def expected_moon_speed_mph : ℝ := 2281.34

-- Theorem stating the equivalence of converted speed to expected speed
theorem moon_speed_conversion_correct :
  kilometers_per_second_to_miles_per_hour moon_speed_kmps = expected_moon_speed_mph :=
by 
  sorry

end moon_speed_conversion_correct_l85_85713


namespace find_v_l85_85808

variable (v : ℝ × ℝ)

def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let k := (u.1 * v.1 + u.2 * v.2) / (u.1^2 + u.2^2)
  (k * u.1, k * u.2)

theorem find_v (h1 : proj ⟨2, 1⟩ v = ⟨38/5, 19/5⟩) (h2 : proj ⟨2, 3⟩ v = ⟨58/13, 87/13⟩) :
  v = ⟨7, 5⟩ :=
  sorry

end find_v_l85_85808


namespace boat_downstream_distance_l85_85725

/-- A boat travels downstream in 4 hours, 75 km upstream in 15 hours, and
the speed of the stream is 10 km/h. Prove that the distance the boat goes
downstream is 100 km. -/
theorem boat_downstream_distance (V_b V_s : ℝ) (t_downstream t_upstream : ℝ)
  (distance_upstream : ℝ) (V_s' : V_s = 10) (t_downstream' : t_downstream = 4)
  (t_upstream' : t_upstream = 15) (distance_upstream' : distance_upstream = 75) :
  let distance_downstream := (V_b + V_s) * t_downstream in
  distance_downstream = 100 :=
by
  sorry

end boat_downstream_distance_l85_85725


namespace triangular_pyramid_volume_l85_85635

-- Define the conditions and variables
variables (c : ℝ) (hypotenuse_length : c > 0)

-- Volume of the triangular pyramid given the conditions
theorem triangular_pyramid_volume (h : hypotenuse_length) :
  let base_area := (c^2 * real.sqrt 3) / 8 in
  let height := (c * real.sqrt 3) / 4 in
  let volume := (1 / 3) * base_area * height in
  volume = c^3 / 32 :=
by {
  -- Proof is omitted
  sorry
}

end triangular_pyramid_volume_l85_85635


namespace min_n_for_coprimes_l85_85928

theorem min_n_for_coprimes (N : ℕ) (hN : 0 < N) (n : ℕ) 
  (h : ∀ m, N < m → ∃ A : Finset ℕ, Finset.card A = 7 ∧ (∀ (x ∈ A) (y ∈ A), x ≠ y → Nat.coprime x y)) : 22 ≤ n := 
sorry

end min_n_for_coprimes_l85_85928


namespace ctg_beta_values_l85_85917

theorem ctg_beta_values (α β : ℝ) 
  (h1 : Real.tan (2 * α - β) + 6 * Real.tan (2 * α) + Real.tan β = 0)
  (h2 : Real.tan α = 2) :
  Real.cot β = 1 ∨ Real.cot β = 1 / 7 := 
by
sorry

end ctg_beta_values_l85_85917


namespace length_of_BC_l85_85764

theorem length_of_BC (a : ℝ) 
  (h_parabola_B : ∃ xB, B = (-a, 2 * a^2)) 
  (h_parabola_C : ∃ xC, C = (a, 2 * a^2))
  (h_origin_A : A = (0, 0))
  (h_BC_parallel : B.2 = C.2)
  (h_area : 2 * a^3 = 128) :
  2 * a = 8 :=
sorry

end length_of_BC_l85_85764


namespace chord_parabola_constant_l85_85729

theorem chord_parabola_constant (k : ℝ) : 
  ∀ (A B F : ℝ × ℝ), 
  (∃ x, y = (1 / 4) * x^2) ∧ 
  (A = (x₁, y₁)) ∧ 
  (B = (x₂, y₂)) ∧ 
  F = (0, 1) ∧ 
  slope = k → 
  (y₁ + y₂ = 4 * k^2 + 2) ∧ 
  (y₁ * y₂ = 1) ∧ 
  (1 / AF + 1 / BF = 1) :=
sorry

end chord_parabola_constant_l85_85729


namespace alice_total_cost_usd_is_correct_l85_85763

def tea_cost_yen : ℕ := 250
def sandwich_cost_yen : ℕ := 350
def conversion_rate : ℕ := 100
def total_cost_usd (tea_cost_yen sandwich_cost_yen conversion_rate : ℕ) : ℕ :=
  (tea_cost_yen + sandwich_cost_yen) / conversion_rate

theorem alice_total_cost_usd_is_correct :
  total_cost_usd tea_cost_yen sandwich_cost_yen conversion_rate = 6 := 
by
  sorry

end alice_total_cost_usd_is_correct_l85_85763


namespace diagonal_of_rectangular_solid_l85_85988

-- Define the lengths of the edges
def a : ℝ := 2
def b : ℝ := 3
def c : ℝ := 4

-- Prove that the diagonal of the rectangular solid with edges a, b, and c is sqrt(29)
theorem diagonal_of_rectangular_solid (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : (a^2 + b^2 + c^2) = 29 := 
by 
  rw [h1, h2, h3]
  norm_num

end diagonal_of_rectangular_solid_l85_85988


namespace max_liars_17_people_circle_l85_85302

def maximum_liars (n : ℕ) : ℕ :=
  if n = 17 then 11 else 0

theorem max_liars_17_people_circle : maximum_liars 17 = 11 :=
begin
  -- the proof steps would go here,
  -- but we'll skip it for this example 
  sorry
end

end max_liars_17_people_circle_l85_85302


namespace find_sum_of_squares_l85_85942

theorem find_sum_of_squares (x y : ℝ) (h1: x * y = 16) (h2: x^2 + y^2 = 34) : (x + y) ^ 2 = 66 :=
by sorry

end find_sum_of_squares_l85_85942


namespace max_value_expression_l85_85091

variable (x y z : ℝ)

theorem max_value_expression (h : x^2 + y^2 + z^2 = 4) :
  (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2 ≤ 28 :=
sorry

end max_value_expression_l85_85091


namespace trainA_distance_when_meet_l85_85339

noncomputable def train_distance := (distance : ℕ) (v_A v_B : ℕ) : ℕ :=
  let closing_speed := v_A + v_B
  let time := distance / closing_speed
  v_A * time

theorem trainA_distance_when_meet (distance v_A v_B d_A : ℕ) (h₀ : distance = 200) (h₁ : v_A = 20) (h₂ : v_B = 20) (h₃ : v_A = d_A / (distance / (v_A + v_B))) : d_A = 100 :=
  by
    rw [h₀, h₁, h₂]
    simp
    sorry

end trainA_distance_when_meet_l85_85339


namespace distance_from_P_to_x_axis_l85_85840

noncomputable def hyperbola := ∀ (x y : ℝ), x^2 / 2 - y^2 / 4 = 1
noncomputable def asymptote : (ℝ → ℝ) := fun x => sqrt 2 * x
noncomputable def point_on_asymptote (P : ℝ × ℝ) := ∃ m : ℝ, P = (m, sqrt 2 * m)
noncomputable def foci (F1 F2 : ℝ × ℝ) := F1 = (-sqrt 6, 0) ∧ F2 = (sqrt 6, 0)
noncomputable def scalar_product_zero (P F1 F2 : ℝ × ℝ) := 
  (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = 0

theorem distance_from_P_to_x_axis (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (h1 : hyperbola P.1 P.2) 
(h2 : point_on_asymptote P) (h3 : foci F1 F2) (h4 : scalar_product_zero P F1 F2) : 
  P.2 = 2 := sorry

end distance_from_P_to_x_axis_l85_85840


namespace all_dominos_can_be_arranged_horizontally_l85_85246

-- Definitions:

-- A chessboard of size 8x8, initially completely covered by 32 dominos
def Chessboard : Type := sorry

-- Condition: One extra cell on the board
def extraCell (b : Chessboard) : Prop := sorry

-- Condition: A domino can be moved to adjacent empty cells
def moveableDomino (b : Chessboard) (d: Domino) : Prop := sorry

-- Condition: Non-overlapping domino placements
def nonOverlapping (b : Chessboard) : Prop := sorry

-- Proving the statement
theorem all_dominos_can_be_arranged_horizontally (b : Chessboard) 
  (h_extraCell: extraCell b) 
  (h_moveableDomino : ∀ d, moveableDomino b d) 
  (h_nonOverlapping : nonOverlapping b) : 
  ∀ d, horizontalDomino b d :=
sorry

end all_dominos_can_be_arranged_horizontally_l85_85246


namespace problem_l85_85232

theorem problem (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 := 
sorry

end problem_l85_85232


namespace right_triangle_area_l85_85730

theorem right_triangle_area (a b c : ℝ)
    (h1 : a = 16)
    (h2 : ∃ r, r = 6)
    (h3 : c = Real.sqrt (a^2 + b^2)) 
    (h4 : a^2 + b^2 = c^2) :
    1/2 * a * b = 240 := 
by
  -- given:
  -- a = 16
  -- ∃ r, r = 6
  -- c = Real.sqrt (a^2 + b^2)
  -- a^2 + b^2 = c^2
  -- Prove: 1/2 * a * b = 240
  sorry

end right_triangle_area_l85_85730


namespace tangent_line_through_M_to_circle_l85_85756

noncomputable def M : ℝ × ℝ := (2, -1)
noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem tangent_line_through_M_to_circle :
  ∀ {x y : ℝ}, circle_eq x y → M = (2, -1) → 2*x - y - 5 = 0 :=
sorry

end tangent_line_through_M_to_circle_l85_85756


namespace by_how_much_were_the_numerator_and_denominator_increased_l85_85274

noncomputable def original_fraction_is_six_over_eleven (n : ℕ) : Prop :=
  n / (n + 5) = 6 / 11

noncomputable def resulting_fraction_is_seven_over_twelve (n x : ℕ) : Prop :=
  (n + x) / (n + 5 + x) = 7 / 12

theorem by_how_much_were_the_numerator_and_denominator_increased :
  ∃ (n x : ℕ), original_fraction_is_six_over_eleven n ∧ resulting_fraction_is_seven_over_twelve n x ∧ x = 1 :=
by
  sorry

end by_how_much_were_the_numerator_and_denominator_increased_l85_85274


namespace num_ways_to_divide_friends_l85_85522

theorem num_ways_to_divide_friends :
  (4 : ℕ) ^ 8 = 65536 := by
  sorry

end num_ways_to_divide_friends_l85_85522


namespace arithmetic_sequence_common_difference_l85_85543

theorem arithmetic_sequence_common_difference
  (a1 a4 : ℤ) (d : ℤ) 
  (h1 : a1 + (a1 + 4 * d) = 10)
  (h2 : a1 + 3 * d = 7) : 
  d = 2 :=
sorry

end arithmetic_sequence_common_difference_l85_85543


namespace sum_of_two_numbers_l85_85656

theorem sum_of_two_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y = 9) (h2 : (1 / x) = 4 * (1 / y)) : x + y = 15 / 2 :=
  sorry

end sum_of_two_numbers_l85_85656


namespace find_range_of_a_l85_85499

noncomputable def quadratic_zero_in_interval (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ c ∈ set.Icc (-1 : ℝ) 1, f c = 0

theorem find_range_of_a (a : ℝ) :
  quadratic_zero_in_interval a (λ x, 2 * a * x^2 + 2 * x - 3 - a) ↔
  a ≤ (-(3 : ℝ) - real.sqrt 7) / 2 ∨ 1 ≤ a := 
sorry

end find_range_of_a_l85_85499


namespace pyramid_height_l85_85774

-- Definitions to model the problem conditions
structure RightAngledTriangle (a b c : ℕ) : Prop :=
  (hypotenuse : c^2 = a^2 + b^2)
  (sides_345 : {a, b, c} = {3, 4, 5})

structure InclinedLateralFace (angle : ℕ) : Prop :=
  (angle_45deg : angle = 45)

-- Theorem statement
theorem pyramid_height (a b c angle height : ℕ)
  (h_triangle : RightAngledTriangle a b c)
  (h_inclined : InclinedLateralFace angle) :
  height = 1 ∨ height = 2 ∨ height = 3 ∨ height = 6 :=
sorry

end pyramid_height_l85_85774


namespace digit_8_occurrences_from_1_to_700_l85_85144

theorem digit_8_occurrences_from_1_to_700 : 
  let occurrences (d : ℕ) (n : ℕ) := (finset.range d).sum (λ x, (nat.digits 10 x).count n) in
  occurrences 701 8 = 240 :=
by
  sorry

end digit_8_occurrences_from_1_to_700_l85_85144


namespace seventh_term_arithmetic_sequence_l85_85078

theorem seventh_term_arithmetic_sequence :
  ∃ a d : ℝ, 
    (∑ i in (finset.range 7), (a + (i - 4) * d)^5 = 0) ∧
    (∑ i in (finset.range 7), (a + (i - 4) * d)^4 = 2000) →
    (a + 3 * d = 3 * real.root 10 (4 : ℕ)) :=
sorry

end seventh_term_arithmetic_sequence_l85_85078


namespace max_number_square_l85_85193

theorem max_number_square (A B C D E : ℕ) (h_distinct: ∀ x y: ℕ, x ≠ y -> (x ∈ {5, 6, 7, 8, 9}) -> (y ∈ {5, 6, 7, 8, 9}) -> (x ≠ y)) :
  A ∈ {5, 6, 7, 8, 9} ∧ B ∈ {5, 6, 7, 8, 9} ∧ C ∈ {5, 6, 7, 8, 9} ∧
  D ∈ {5, 6, 7, 8, 9} ∧ E ∈ {5, 6, 7, 8, 9} →
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
  C ≠ D ∧ C ≠ E ∧
  D ≠ E) →
  let F := A * B * C in
  let G := B * C * D in
  let H := C * D * E in
  let I := F + G + H in
  I ≤ 1251 :=
sorry

end max_number_square_l85_85193


namespace least_positive_integer_l85_85692

-- Necessary to define the conditions
def conditions (n : ℕ) :=
  (∀ i ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10}, n % i = 1) ∧ n > 1

-- Theorem statement that uses the conditions definition
theorem least_positive_integer (n : ℕ) (h : conditions n) : n = 2521 :=
by
  sorry

end least_positive_integer_l85_85692


namespace infinite_rectangles_cover_plane_infinite_squares_cover_plane_l85_85918

-- Define the property of interest for the set of rectangles or squares
def can_cover_plane (rectangles : Set (Set ℝ × ℝ)) (can_overlap : Bool) : Prop :=
  ∀ (S : ℝ), ∃ subset ∈ rectangles, (total_area subset > S) → (covers_plane subset can_overlap)

-- Part (a) statement: infinite set of rectangles can cover the plane
theorem infinite_rectangles_cover_plane (rectangles : Set (Set ℝ × ℝ)) :
  (∀ S : ℝ, ∃ subset ∈ rectangles, total_area subset > S) →
  can_cover_plane rectangles true :=
sorry

-- Part (b) statement: infinite set of squares can cover the plane
theorem infinite_squares_cover_plane (squares : Set (Set ℝ)) :
  (∀ S : ℝ, ∃ subset ∈ squares, total_area subset > S) →
  can_cover_plane squares true :=
sorry

-- Definitions for total_area and covers_plane can be added according to specific needs
noncomputable def total_area (subset : Set (ℝ × ℝ)) : ℝ :=
  sorry

def covers_plane (subset : Set (ℝ × ℝ)) (can_overlap : Bool) : Prop :=
  sorry

end infinite_rectangles_cover_plane_infinite_squares_cover_plane_l85_85918


namespace area_of_quadrilateral_AEOF_l85_85772

theorem area_of_quadrilateral_AEOF
  {A B C D E F O : Type}
  (h1 : rectangle A B C D)
  (h2 : area A B C D = 60)
  (h3 : E ≠ B)
  (h4 : collinear A E B)
  (h5 : collinear F A D)
  (h6 : EB = 2 * AE)
  (h7 : AF = FD)
  : area_AEOF = 7 := by
  sorry

end area_of_quadrilateral_AEOF_l85_85772


namespace product_lcm_gcd_eq_2160_l85_85694

theorem product_lcm_gcd_eq_2160 :
  let a := 36
  let b := 60
  lcm a b * gcd a b = 2160 := by
  sorry

end product_lcm_gcd_eq_2160_l85_85694


namespace lillian_cupcakes_l85_85953

theorem lillian_cupcakes (home_sugar : ℕ) (bags : ℕ) (sugar_per_bag : ℕ) (batter_sugar_per_dozen : ℕ) (frosting_sugar_per_dozen : ℕ) :
  home_sugar = 3 → bags = 2 → sugar_per_bag = 6 → batter_sugar_per_dozen = 1 → frosting_sugar_per_dozen = 2 →
  ((home_sugar + bags * sugar_per_bag) / (batter_sugar_per_dozen + frosting_sugar_per_dozen)) = 5 := 
by 
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end lillian_cupcakes_l85_85953


namespace find_AD_in_convex_quad_l85_85077

theorem find_AD_in_convex_quad (ABCD : Type) [convex_quadrilateral ABCD]
  (A B C D X : Point)
  (h_midpoint : midpoint X A C)
  (h_parallel : parallel CD BX)
  (h_length_BX : length BX = 3)
  (h_length_BC : length BC = 7)
  (h_length_CD : length CD = 6) :
  length AD = 14 :=
by
  sorry

end find_AD_in_convex_quad_l85_85077


namespace number_of_ways_to_divide_friends_l85_85518

theorem number_of_ways_to_divide_friends :
  (4 ^ 8 = 65536) := by
  -- result obtained through the multiplication principle
  sorry

end number_of_ways_to_divide_friends_l85_85518


namespace correct_answer_l85_85617

-- Definitions
def is_sine_function (f : ℝ → ℝ) : Prop := 
  ∃ a : ℝ, a ≠ 0 ∧ ∃ b : ℝ, f = (λ x, sin (a * x + b))

def is_piecewise_function (f : ℝ → ℝ) : Prop := 
  ∃ g h : ℝ → ℝ, ∃ p : ℝ → Prop, ∀ x, (p x → f x = g x) ∧ (¬p x → f x = h x)

-- Original proposition
def original_proposition : Prop := 
  ∀ f : ℝ → ℝ, is_sine_function f → ¬is_piecewise_function f

-- Correct answer statement
theorem correct_answer :
  (∀ f : ℝ → ℝ, is_piecewise_function f → ¬is_sine_function f) :=
begin
  sorry
end

end correct_answer_l85_85617


namespace alice_bob_sitting_arrangements_l85_85541

theorem alice_bob_sitting_arrangements :
  ∃ (n : ℕ), n = 10 
  ∧ ∃ (arrangements : ℕ), arrangements = (10 - 1)! / (10 - 1) * 2
  ∧ arrangements = 80,640 :=
begin
  -- There are 10 people.
  use 10,
  split,
  -- Alice and Bob must sit next to each other.
  exact rfl,
  use ((10 - 1)! / (10 - 1)) * 2,
  split,
  -- Compute the arrangement
  -- (9!) / 9 * 2 = (8!) * 2
  -- 8! = 40320, so the total is 40320 * 2 = 80640
  exact rfl,
  -- The total is 80640
  norm_num,
  sorry,
end

end alice_bob_sitting_arrangements_l85_85541


namespace problem_l85_85470

-- Definitions for sequence a_n and related properties
def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧
  a 2 = 6 ∧
  ∀ n, (a (n + 2) - a (n + 1)) - (a (n + 1) - a n) = 2

noncomputable def seq_sum (a : ℕ → ℕ) :=
  ∑ k in Finset.range 2016, 2016 / (a (k + 1))

theorem problem (a : ℕ → ℕ) (h : sequence a) :
  ⌊seq_sum a⌋ = 2015 := sorry

end problem_l85_85470


namespace geometric_progression_common_ratio_l85_85172

theorem geometric_progression_common_ratio:
  ∀ {a r : ℝ}, a > 0 → (∀ n : ℕ, a * r ^ n = a * r ^ (n + 1) + a * r ^ (n + 2)) → 
    r = (sqrt 5 - 1) / 2 :=
begin
  sorry
end

end geometric_progression_common_ratio_l85_85172


namespace general_term_a_n_sum_b_n_terms_l85_85591

-- Given definitions based on the conditions
def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := (2^(2*n-1))^2

def b_sum (n : ℕ) : (ℕ → ℕ) := 
  (fun b : ℕ => match b with 
                | 1 => 4 
                | 2 => 64 
                | _ => (4^(2*(b - 2 + 1) - 1)))

def T (n : ℕ) : ℕ := (4 / 15) * (16^n - 1)

-- First part: Proving the general term of {a_n} is 2^(n-1)
theorem general_term_a_n (n : ℕ) : a n = 2^(n-1) := by
  sorry

-- Second part: Proving the sum of the first n terms of {b_n} is (4/15)*(16^n - 1)
theorem sum_b_n_terms (n : ℕ) : T n = (4 / 15) * (16^n - 1) := by 
  sorry

end general_term_a_n_sum_b_n_terms_l85_85591


namespace largest_non_provable_amount_is_correct_l85_85545

def limonia_coin_denominations (n : ℕ) : List ℕ :=
  List.range (n + 1) |> List.map (λ i => 3^(n - i) * 4^i)

def largest_non_provable_amount (n : ℕ) : ℕ :=
  2 * 4^(n+1) - 3^(n+2)

theorem largest_non_provable_amount_is_correct (n : ℕ) :
  ∀ s : ℕ, (¬∃ c : List ℕ, 
    (c ∈ List.powerset (limonia_coin_denominations n)) ∧ (s = c.sum)) 
    ↔ s = largest_non_provable_amount n := 
by
  sorry

end largest_non_provable_amount_is_correct_l85_85545


namespace find_y_l85_85432

-- Define the main hypothesis
theorem find_y (y : ℝ) (h : 2 * arctan (1 / 3) + arctan (1 / 15) + arctan (1 / y) = π / 3) :
  y = 13.25 :=
sorry

end find_y_l85_85432


namespace find_f_e_l85_85462

noncomputable def f (x : ℝ) : ℝ := 2 * f' 1 * Real.log x + x / Real.exp 1

theorem find_f_e : f (Real.exp 1) = 1 - 2 / Real.exp 1 := by
  sorry

end find_f_e_l85_85462


namespace area_triangle_l85_85196

theorem area_triangle (PQ PR PS PT : ℝ) (a b θ φ : ℝ)
  (h_eq : PQ = PR)
  (h_altitude : PS^2 + (QR/2)^2 = PQ^2)
  (h_extension : PT = 12)
  (h_geometric_tan : (real.tan (θ - φ)) * (real.tan (θ + φ)) = (real.tan θ)^2)
  (h_arithmetic_cot : (real.cot (π/4)) + ((a + b) / (b - a)) = 2 * (real.cot φ))
  (h_tan_eq_one : real.tan θ = 1)
  (area_eq : a = PQ / 3 ∧ b = 6 * real.sqrt 2) : 
  let area := PQ * PS / 2 in 
  area = 24 := 
sorry

end area_triangle_l85_85196


namespace two_zeros_of_function_l85_85864

theorem two_zeros_of_function (a : ℝ) :
  (-1 < a ∧ a < 0) ↔ (∀ f : ℝ → ℝ, f = (λ x, a * Real.log x + x^2 - (a + 2) * x) →
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0) :=
by
  sorry

end two_zeros_of_function_l85_85864


namespace secretary_typing_orders_l85_85179

theorem secretary_typing_orders :
  let S := {1, 2, 3, 4, 5, 6, 7, 8}
  in (∑ k in finset.range 9, (nat.choose 8 k) * (k + 2)) = 1536 :=
by sorry

end secretary_typing_orders_l85_85179


namespace apps_deleted_more_than_added_l85_85787

theorem apps_deleted_more_than_added 
  (initial_apps : ℕ) 
  (added_apps : ℕ) 
  (left_apps : ℕ) 
  (h1 : initial_apps = 15) 
  (h2 : added_apps = 71) 
  (h3 : left_apps = 14) :
  (initial_apps + added_apps - left_apps - added_apps = 1) :=
by {
  rw [h1, h2, h3],
  norm_num,
}

end apps_deleted_more_than_added_l85_85787


namespace find_four_digit_squares_l85_85045

theorem find_four_digit_squares (N : ℕ) (a b : ℕ) 
    (h1 : 100 ≤ N ∧ N < 10000)
    (h2 : 10 ≤ a ∧ a < 100)
    (h3 : 0 ≤ b ∧ b < 100)
    (h4 : N = 100 * a + b)
    (h5 : N = (a + b) ^ 2) : 
    N = 9801 ∨ N = 3025 ∨ N = 2025 :=
    sorry

end find_four_digit_squares_l85_85045


namespace square_area_l85_85755

-- Definition of the vertices' coordinates
def y_coords := ({-3, 2, 2, -3} : Set ℤ)
def x_coords_when_y2 := ({0, 5} : Set ℤ)

-- The statement we need to prove
theorem square_area (h1 : y_coords = {-3, 2, 2, -3}) 
                     (h2 : x_coords_when_y2 = {0, 5}) : 
                     ∃ s : ℤ, s^2 = 25 :=
by
  sorry

end square_area_l85_85755


namespace find_f_expression_l85_85853

-- Define the function f and the condition it satisfies
variable (f : ℝ → ℝ)
variable (h : ∀ x : ℝ, f(x) + 2 * f(1 / x) = 1 / x + 2)

-- State the theorem
theorem find_f_expression : ∀ x : ℝ, f(x) = (2/3:ℝ) * x - (1/3:ℝ) * (1 / x) + (2/3:ℝ) :=
by
  -- Placeholder for the proof
  sorry

end find_f_expression_l85_85853


namespace ivans_number_is_5303_l85_85060

def is_valid_two_digit (n : ℕ) : Prop := n >= 10 ∧ n <= 99

def is_valid_segment (s : String) : Prop :=
  let n := s.to_nat! -- convert string to natural number
  is_valid_two_digit n

def is_valid_combination (seq : String) (ivan_num : String) : Prop :=
  ivan_num.length = 4 ∧
  let remaining := seq.drop ivan_num.length
  remaining.length = 8 ∧
  is_valid_segment (remaining.take 2) ∧
  is_valid_segment (remaining.drop 2 |>.take 2) ∧
  is_valid_segment (remaining.drop 4 |>.take 2) ∧
  is_valid_segment (remaining.drop 6 |>.take 2)

noncomputable def ivans_number_proof : Prop :=
  ∃ ivan_num, ivan_num = "5303" ∧ is_valid_combination "132040530321" ivan_num

theorem ivans_number_is_5303 : ivans_number_proof :=
by
  exists "5303"
  sorry

end ivans_number_is_5303_l85_85060


namespace exists_polynomial_l85_85829

theorem exists_polynomial (a : ℕ) (ha : 0 < a) (n : ℕ) (hn : 0 < n):
  ∃ p : Polynomial ℤ, (∀ k : ℕ, k ≤ n → ∃ l : ℤ, (p.eval k = 2 * (a ^ l) + 3)) :=
by
  sorry

end exists_polynomial_l85_85829


namespace a_arithmetic_sequence_Tn_formula_l85_85471

variables {a : ℕ → ℕ} {S : ℕ → ℕ} {b : ℕ → ℚ} {T : ℕ → ℚ}

-- Define conditions
axiom a_pos (n : ℕ) (h : n ≠ 0) : a n > 0
axiom S_def (n : ℕ) (h : n ≠ 0) : S n = a n * (a n + 1) / 2

-- Prove that the sequence a_n is an arithmetic sequence
theorem a_arithmetic_sequence (n : ℕ) (h : n ≠ 0) :
  ∃ d : ℤ, ∀ m : ℕ, a (m + 1) = a m + d := sorry

-- Define bn and Tn
def b (n : ℕ) : ℚ := 1 / (2 * S n)
def T (n : ℕ) : ℚ := ∑ i in finset.range n, b (i + 1)

-- Prove the formula for Tn
theorem Tn_formula (n : ℕ) (h : n ≠ 0) :
  T n = n / (n + 1) := sorry

end a_arithmetic_sequence_Tn_formula_l85_85471


namespace find_M_M_superset_N_M_intersection_N_l85_85505

-- Define the set M as per the given condition
def M : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }

-- Define the set N based on parameters a and b
def N (a b : ℝ) : Set ℝ := { x : ℝ | a < x ∧ x < b }

-- Prove that M = (-1, 2)
theorem find_M : M = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

-- Prove that if M ⊇ N, then a ≥ -1
theorem M_superset_N (a b : ℝ) (h : M ⊇ N a b) : -1 ≤ a :=
sorry

-- Prove that if M ∩ N = M, then b ≥ 2
theorem M_intersection_N (a b : ℝ) (h : M ∩ (N a b) = M) : 2 ≤ b :=
sorry

end find_M_M_superset_N_M_intersection_N_l85_85505


namespace bookstore_purchase_prices_equal_l85_85355

variable (x : ℝ)

theorem bookstore_purchase_prices_equal
  (h1 : 500 > 0)
  (h2 : 700 > 0)
  (h3 : x > 0)
  (h4 : x + 4 > 0)
  (h5 : ∃ p₁ p₂ : ℝ, p₁ = 500 / x ∧ p₂ = 700 / (x + 4) ∧ p₁ = p₂) :
  500 / x = 700 / (x + 4) :=
by
  sorry

end bookstore_purchase_prices_equal_l85_85355


namespace one_over_x_plus_one_over_y_eq_two_l85_85847

theorem one_over_x_plus_one_over_y_eq_two 
  (x y : ℝ)
  (h1 : 3^x = Real.sqrt 12)
  (h2 : 4^y = Real.sqrt 12) : 
  1 / x + 1 / y = 2 := 
by 
  sorry

end one_over_x_plus_one_over_y_eq_two_l85_85847


namespace interesting_sets_l85_85227

noncomputable def is_interesting_set (p : ℕ) (S : Finset ℕ) : Prop :=
  p.Prime ∧ S.card = p + 2 ∧
  ∀ T : Finset ℕ, T.card = p → ∃ a b : ℕ, a ∈ S \ T ∧ b ∈ S \ T ∧ (∀ x ∈ T, x ∣ a) ∧ (∀ x ∈ T, x ∣ b)

theorem interesting_sets (p : ℕ) (S : Finset ℕ) :
  is_interesting_set p S ↔ (∃ (a : ℕ), S = (Finset.range (p+2)).map (λ _, a) ∨ S = (insert (p * a) ((Finset.range p).map (λ _, a)))) :=
sorry

end interesting_sets_l85_85227


namespace student_A_claps_twice_in_20_numbers_student_A_claps_10th_at_39th_number_l85_85447

-- Define Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fib n + fib (n+1)

-- Condition (student's claps and fibonacci sequence)
def claps (n : ℕ) := (fib n) % 3 = 0

-- Question 1: Prove that student A claps 2 times in first 20 numbers.
theorem student_A_claps_twice_in_20_numbers :
  (finset.filter claps (finset.range 20)) .card = 2 := 
  sorry

-- Question 2: Prove that by the 10th time student A claps,
-- students have counted up to the 39th number.
theorem student_A_claps_10th_at_39th_number (n : ℕ) :
  (finset.filter claps (finset.range n)) .card  = 10 → n = 39 :=
  sorry

end student_A_claps_twice_in_20_numbers_student_A_claps_10th_at_39th_number_l85_85447


namespace solve_for_y_l85_85325

theorem solve_for_y (y : ℝ) :
  (40 / 60 = real.sqrt (y / 60)) → y = 80 / 3 :=
by
  intro h
  -- proof goes here
  sorry

end solve_for_y_l85_85325


namespace insects_meet_at_S_l85_85684

-- Define the points of the triangle and the respective side lengths
def Triangle := {PQ QR PR : ℕ // PQ = 7 ∧ QR = 8 ∧ PR = 9}

-- State the problem in terms of what needs to be proved
theorem insects_meet_at_S (T : Triangle) : QS = 5 := 
  sorry

end insects_meet_at_S_l85_85684


namespace sum_positive_l85_85810

-- Define the sum of digits of the binary representation of n
def s (n : ℕ) : ℕ := n.binary_digits.sum

-- Define the sum we want to prove is positive
def sum_to_prove : ℝ :=
  ∑ n in finset.range (2^2022), (-1) ^ s n / (2022 + n)

theorem sum_positive : sum_to_prove > 0 :=
sorry

end sum_positive_l85_85810


namespace max_distinct_prime_factors_l85_85625

theorem max_distinct_prime_factors 
  (a b : ℕ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_gcd : nat.prime_factors (gcd a b).card = 7) 
  (h_lcm : nat.prime_factors (nat.lcm a b).card = 28) 
  (h_fewer_factors : nat.prime_factors a.card < nat.prime_factors b.card) : 
  nat.prime_factors a.card ≤ 17 := 
sorry

end max_distinct_prime_factors_l85_85625


namespace fortieth_number_is_twelve_l85_85280

/--
Given a sequence where the $n$-th row contains $2n$ occurrences of the number $2n$,
prove that the $40^{\mathrm{th}}$ number in this sequence is $12$.
-/
theorem fortieth_number_is_twelve :
  let sequence := List.concat (List.range 20).map (λ n => List.replicate (2*(n+1)) (2*(n+1)))
  sequence.get? 39 = some 12 :=
by
  sorry

end fortieth_number_is_twelve_l85_85280


namespace population_of_town_l85_85180

theorem population_of_town (F : ℝ) (males : ℕ) (female_glasses : ℝ) (percentage_glasses : ℝ) (total_population : ℝ) 
  (h1 : males = 2000) 
  (h2 : percentage_glasses = 0.30) 
  (h3 : female_glasses = 900) 
  (h4 : percentage_glasses * F = female_glasses) 
  (h5 : total_population = males + F) :
  total_population = 5000 :=
sorry

end population_of_town_l85_85180


namespace rectangle_side_lengths_l85_85789

variables (x y m n S : ℝ) (hx_y_ratio : x / y = m / n) (hxy_area : x * y = S)

theorem rectangle_side_lengths :
  x = Real.sqrt (m * S / n) ∧ y = Real.sqrt (n * S / m) :=
sorry

end rectangle_side_lengths_l85_85789


namespace least_three_digit_seven_heavy_l85_85760

def is_seven_heavy (n : ℕ) : Prop := n % 7 > 4

-- Define what it means to be a three-digit number.
def is_three_digits (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- State the main theorem
theorem least_three_digit_seven_heavy : ∃ n : ℕ, is_seven_heavy(n) ∧ is_three_digits(n) ∧ ∀ m : ℕ, is_seven_heavy(m) ∧ is_three_digits(m) → n ≤ m :=
sorry

end least_three_digit_seven_heavy_l85_85760


namespace tan_half_beta_sin_alpha_l85_85457

-- Define the conditions
variables {α β : ℝ}
axiom h1 : 0 < α ∧ α < π / 2
axiom h2 : π / 2 < β ∧ β < π
axiom h3 : cos β = -1 / 3
axiom h4 : sin (α + β) = 7 / 9

-- Prove the first requirement
theorem tan_half_beta : tan (β / 2) = sqrt 2 :=
by sorry

-- Prove the second requirement
theorem sin_alpha : sin α = 1 / 3 :=
by sorry

end tan_half_beta_sin_alpha_l85_85457


namespace trig_inequality_l85_85094

variable {f : ℝ → ℝ}
variable {f'' : ℝ → ℝ}
variable {A B : ℝ}

def second_derivative (y : ℝ) := true -- Representation for the existence of second derivative

theorem trig_inequality (h1 : ∀ x, second_derivative x) 
    (h2 : ∀ x, x * f'' x - 2 * f x > 0)
    (h3 : 0 < sin B ∧ sin B < cos A):
    f (cos A) * (sin B) ^ 2 > f (sin B) * (cos A) ^ 2 := 
sorry

end trig_inequality_l85_85094


namespace exponential_property_l85_85905

noncomputable theory

variables {α : Type*} [ordered_comm_group α]

def h (a : α) (x : α) : α := a ^ x

theorem exponential_property (a : α) (x₁ x₂ : α) (h₁ : 0 < a) (h₂ : a ≠ 1) :
  h a (x₁ + x₂) = h a x₁ * h a x₂ :=
by {
  unfold h,
  exact pow_add a x₁ x₂
}

end exponential_property_l85_85905


namespace AF_perpendicular_to_BC_l85_85916

noncomputable def TriangleABC (A B C D E F : Type) :=
  ∃ (angleBAC : ℝ) (angleABC : ℝ) (angleCBD : ℝ) (angleBCE : ℝ),
  angleBAC = 40 ∧
  angleABC = 60 ∧
  angleCBD = 40 ∧
  angleBCE = 70 ∧
  intersection BD CE = F → 
  perpendicular AF BC

theorem AF_perpendicular_to_BC :
  ∀ A B C D E F : Type,
  TriangleABC A B C D E F →
  (perpendicular AF BC) := by
  sorry

end AF_perpendicular_to_BC_l85_85916


namespace nonnegative_difference_between_roots_l85_85316

open Real

theorem nonnegative_difference_between_roots :
  ∀ x : ℝ, ∀ a b : ℝ, (x^2 + 40*x + 348 = 0) →
    (∀ r1 r2 : ℝ, (x - r1) * (x - r2) = x^2 + 40*x + 348 → 
    (r1 = -12 ∨ r1 = -28) ∧ (r2 = -12 ∨ r2 = -28) ∧ r1 ≠ r2 → 
    abs(r1 - r2) = 16) :=
begin
  intros,
  sorry
end

end nonnegative_difference_between_roots_l85_85316


namespace calculate_mass_fraction_l85_85950

variable (n_Na2CO3 : ℝ)
variable (M_Na2CO3 : ℝ)
variable (m_solution_Na2CO3 : ℝ)

def mass_Na2CO3 (n : ℝ) (M : ℝ) : ℝ := n * M

def mass_fraction_Na2CO3 (mass : ℝ) (m_solution : ℝ) : ℝ :=
  (mass * 100) / m_solution

theorem calculate_mass_fraction 
  (hn_Na2CO3 : n_Na2CO3 = 0.125)
  (hM_Na2CO3 : M_Na2CO3 = 106)
  (hm_solution_Na2CO3 : m_solution_Na2CO3 = 132.5) :
  (mass_Na2CO3 n_Na2CO3 M_Na2CO3 = 13.25) ∧ 
  (mass_fraction_Na2CO3 13.25 m_solution_Na2CO3 = 10) :=
by
  sorry

end calculate_mass_fraction_l85_85950


namespace correct_option_D_l85_85704

theorem correct_option_D : (-8) / (-4) = 8 / 4 := 
by
  exact (rfl

end correct_option_D_l85_85704


namespace triangle_perimeter_l85_85441

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def perimeter_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  distance A B + distance B C + distance C A

theorem triangle_perimeter :
  perimeter_of_triangle (1, 2) (4, 8) (5, 5) = 3 * Real.sqrt 5 + Real.sqrt 10 + 5 :=
by sorry

end triangle_perimeter_l85_85441


namespace inverse_of_composition_l85_85645

variables {X Y Z W T : Type*}
variables (r : T → X) (s : X → Y) (p : Y → Z) (q : Z → W)

noncomputable def f : T → W := q ∘ p ∘ s ∘ r

theorem inverse_of_composition 
  (hr : function.bijective r) (hs : function.bijective s)
  (hp : function.bijective p) (hq : function.bijective q) :
  function.left_inverse ((r ∘ s ∘ p ∘ q)⁻¹) f ∧
  function.right_inverse ((r ∘ s ∘ p ∘ q)⁻¹) f :=
sorry

end inverse_of_composition_l85_85645


namespace fourth_roll_eight_probability_l85_85373

noncomputable def calculate_probability_fourth_eight : ℚ :=
  let fair_prob_three_eights := (1/8)^3 in
  let biased_prob_three_eights := (3/4)^3 in
  let total_prob_three_eights := (1/2 * fair_prob_three_eights + 1/2 * biased_prob_three_eights) in
  let prob_fair_given_three_eights := (1/2 * fair_prob_three_eights) / total_prob_three_eights in
  let prob_biased_given_three_eights := (1/2 * biased_prob_three_eights) / total_prob_three_eights in
  prob_fair_given_three_eights * (1/8) + prob_biased_given_three_eights * (3/4)

theorem fourth_roll_eight_probability :
  calculate_probability_fourth_eight = 1297 / 1736 :=
by sorry

end fourth_roll_eight_probability_l85_85373


namespace not_prime_if_2pow_n_plus_one_l85_85619

theorem not_prime_if_2pow_n_plus_one (n m k : ℕ) (h1 : n = k * m) (h2 : m > 1) (h3 : odd m) : ¬ prime (2^n + 1) :=
by sorry

end not_prime_if_2pow_n_plus_one_l85_85619


namespace find_remainder_division_l85_85971

/--
Given:
1. A dividend of 100.
2. A quotient of 9.
3. A divisor of 11.

Prove: The remainder \( r \) when dividing 100 by 11 is 1.
-/
theorem find_remainder_division :
  ∀ (q d r : Nat), q = 9 → d = 11 → 100 = (d * q + r) → r = 1 :=
by
  intros q d r hq hd hdiv
  -- Proof steps would go here
  sorry

end find_remainder_division_l85_85971


namespace equations_of_lines_l85_85341

theorem equations_of_lines 
  (a_intercepts : (0, 12) ∧ (-12, 0)) 
  (c_equation : ∀ x : ℝ, (0, 0) → y = -2 * x)
  (b_steeper_than_c : ∀ x : ℝ, (0, 12) → y < -2 * x) 
  (d_intersects_b: (12, -24) ∧ 
                   (-12, 0) → (∀ x : ℝ, y = -x - 12)) :
  (∀ x : ℝ, (x, y) ∈ b → y = -3 * x + 12) ∧ (∀ x : ℝ, (x, y) ∈ d → y = -x - 12) :=
by
  sorry

end equations_of_lines_l85_85341


namespace pirate_treasure_probability_l85_85376

/--
Suppose there are 8 islands. Each island has a 1/3 chance of having buried treasure and no traps,
a 1/6 chance of having traps but no treasure, and a 1/2 chance of having neither traps nor treasure.
Prove that the probability that while searching all 8 islands, the pirate will encounter exactly 4 islands 
with treasure and none with traps is 35/648.
-/
theorem pirate_treasure_probability :
  let p_treasure_no_trap := 1/3
      p_trap_no_treasure := 1/6
      p_neither := 1/2
      choose_8_4 := Nat.choose 8 4
      p_4_treasure_no_trap := p_treasure_no_trap^4
      p_4_neither := p_neither^4
  in choose_8_4 * p_4_treasure_no_trap * p_4_neither = 35/648 := 
by
  sorry

end pirate_treasure_probability_l85_85376


namespace math_problem_l85_85132

noncomputable def problem_1 (λ : ℝ) : Prop :=
  let m := (λ - 1, 1)
  let n := (λ - 2, 2)
  (m.1 * n.2 = m.2 * n.1) → (λ = 0)

noncomputable def problem_2 (λ : ℝ) : Prop :=
  let m := (λ - 1, 1)
  let n := (λ - 2, 2)
  let sum := (m.1 + n.1, m.2 + n.2)
  let diff := (m.1 - n.1, m.2 - n.2)
  (sum.1 * diff.1 + sum.2 * diff.2 = 0) → (λ = 3)

-- We would combine these into a single theorem statement
theorem math_problem (λ : ℝ) : ((problem_1 λ) ∧ (problem_2 λ)) := 
  by sorry

end math_problem_l85_85132


namespace lillian_can_bake_and_ice_5_dozen_cupcakes_l85_85957

-- Define conditions as constants and variables
constant cups_at_home : ℕ := 3
constant bags_of_sugar : ℕ := 2
constant cups_per_bag : ℕ := 6
constant batter_sugar_per_dozen : ℕ := 1
constant frosting_sugar_per_dozen : ℕ := 2

-- Calculation based on conditions
def total_sugar : ℕ := cups_at_home + bags_of_sugar * cups_per_bag
def sugar_per_dozen_cupcakes : ℕ := batter_sugar_per_dozen + frosting_sugar_per_dozen
def dozen_cupcakes_possible (sugar : ℕ) : ℕ := sugar / sugar_per_dozen_cupcakes

theorem lillian_can_bake_and_ice_5_dozen_cupcakes :
  dozen_cupcakes_possible total_sugar = 5 :=
by
  sorry

end lillian_can_bake_and_ice_5_dozen_cupcakes_l85_85957


namespace max_norm_c_l85_85838

noncomputable section

variables {V : Type*} [InnerProductSpace ℝ V]
variables (a b c : V)

-- Conditions
def mutually_perpendicular_unit_vectors : Prop :=
  (∥a∥ = 1) ∧ (∥b∥ = 1) ∧ (a ⬝ b = 0)

def condition_on_c : Prop := 
  (a - c) ⬝ (b - c) = 0

-- Statement
theorem max_norm_c (h1 : mutually_perpendicular_unit_vectors a b)
  (h2 : condition_on_c a b c) : ∥c∥ ≤ sqrt 2 := sorry

end max_norm_c_l85_85838


namespace negation_equivalence_l85_85650

-- Define the proposition P stating 'there exists an x in ℝ such that x^2 - 2x + 4 > 0'
def P : Prop := ∃ x : ℝ, x^2 - 2*x + 4 > 0

-- Define the proposition Q which is the negation of proposition P
def Q : Prop := ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0

-- State the proof problem: Prove that the negation of proposition P is equivalent to proposition Q
theorem negation_equivalence : ¬ P ↔ Q := by
  -- Proof to be provided.
  sorry

end negation_equivalence_l85_85650


namespace seats_per_row_is_eight_l85_85384

def number_of_seats_per_row (S : ℕ) := 
  let cost_per_seat := 30
  let discount_per_10_seats := 0.1 * (10 * cost_per_seat)
  let cost_per_seat_with_discount := cost_per_seat - (discount_per_10_seats / 10)
  let total_rows := 5
  let total_cost := 1080
  let total_cost_computed := (total_rows * cost_per_seat_with_discount * S)
  total_cost_computed = total_cost

theorem seats_per_row_is_eight : number_of_seats_per_row 8 := 
by {
  let cost_per_seat := 30,
  let discount_per_10_seats := 0.1 * (10 * cost_per_seat),
  let cost_per_seat_with_discount := cost_per_seat - (discount_per_10_seats / 10),
  let total_cost := 1080,
  let total_cost_computed := (5 * cost_per_seat_with_discount * 8),
  have : total_cost_computed = total_cost, sorry
}

end seats_per_row_is_eight_l85_85384


namespace arithmetic_and_geometric_sequence_statement_l85_85085

-- Arithmetic sequence definitions
def arithmetic_seq (a b d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Conditions
def a_2 : ℕ := 9
def a_5 : ℕ := 21

-- General formula and solution for part (Ⅰ)
def general_formula_arithmetic_sequence : Prop :=
  ∃ (a d : ℕ), (a + d = a_2 ∧ a + 4 * d = a_5) ∧ ∀ n : ℕ, arithmetic_seq a d n = 4 * n + 1

-- Definitions and conditions for geometric sequence derived from arithmetic sequence
def b_n (n : ℕ) : ℕ := 2 ^ (4 * n + 1)

-- Sum of the first n terms of the sequence {b_n}
def S_n (n : ℕ) : ℕ := (32 * (2 ^ (4 * n) - 1)) / 15

-- Statement that needs to be proven
theorem arithmetic_and_geometric_sequence_statement :
  general_formula_arithmetic_sequence ∧ (∀ n, S_n n = (32 * (2 ^ (4 * n) - 1)) / 15) := by
  sorry

end arithmetic_and_geometric_sequence_statement_l85_85085


namespace graph_intersection_points_l85_85628

open Function

theorem graph_intersection_points (g : ℝ → ℝ) (h_inv : Involutive (invFun g)) : 
  ∃! (x : ℝ), x = 0 ∨ x = 1 ∨ x = -1 → g (x^2) = g (x^6) :=
by sorry

end graph_intersection_points_l85_85628


namespace telepathic_connection_probability_l85_85251

-- Define the possible values for a and b
def values : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the predicate for "telepathic connection"
def telepathic_connection (a b : ℕ) : Prop := |a - b| ≤ 1

-- Calculate the number of pairs (a, b) such that |a - b| ≤ 1
def telepathic_event_count : ℕ := 
  (values.product values).count (λ ab, telepathic_connection ab.1 ab.2)

-- The total number of possible pairs (a, b)
def total_event_count : ℕ := 36

-- The probability that Person A and Person B have a "telepathic connection"
def telepathic_probability : ℝ := telepathic_event_count / total_event_count

-- The theorem statement
theorem telepathic_connection_probability :
  telepathic_probability = 4 / 9 :=
by
  sorry

end telepathic_connection_probability_l85_85251


namespace prob_exceeds_175_l85_85816

-- Definitions from the conditions
def prob_less_than_160 (p : ℝ) : Prop := p = 0.2
def prob_160_to_175 (p : ℝ) : Prop := p = 0.5

-- The mathematical equivalence proof we need
theorem prob_exceeds_175 (p₁ p₂ p₃ : ℝ) 
  (h₁ : prob_less_than_160 p₁) 
  (h₂ : prob_160_to_175 p₂) 
  (H : p₃ = 1 - (p₁ + p₂)) :
  p₃ = 0.3 := 
by
  -- Placeholder for proof
  sorry

end prob_exceeds_175_l85_85816


namespace negation_of_every_function_has_parity_l85_85707

-- Assume the initial proposition
def every_function_has_parity := ∀ f : ℕ → ℕ, ∃ (p : ℕ), p = 0 ∨ p = 1

-- Negation of the original proposition
def exists_function_without_parity := ∃ f : ℕ → ℕ, ∀ p : ℕ, p ≠ 0 ∧ p ≠ 1

-- The theorem to prove
theorem negation_of_every_function_has_parity : 
  ¬ every_function_has_parity ↔ exists_function_without_parity := 
by
  unfold every_function_has_parity exists_function_without_parity
  sorry

end negation_of_every_function_has_parity_l85_85707


namespace diagonals_intersect_at_midpoint_of_opposite_vertices_l85_85976

theorem diagonals_intersect_at_midpoint_of_opposite_vertices (x1 y1 x2 y2 : ℝ):
  (x1, y1) = (2, -3) → (x2, y2) = (14, 9) → 
  (∃ m n : ℝ, (m, n) = (8, 3)) :=
by
  intros h1 h2
  use (8, 3)
  sorry

end diagonals_intersect_at_midpoint_of_opposite_vertices_l85_85976


namespace base4_to_base10_conversion_l85_85030

-- We define a base 4 number as follows:
def base4_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10 in
  let n := n / 10 in
  let d1 := n % 10 in
  let n := n / 10 in
  let d2 := n % 10 in
  let n := n / 10 in
  let d3 := n % 10 in
  let n := n / 10 in
  let d4 := n % 10 in
  (d4 * 4^4 + d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0)

-- Mathematical proof problem statement:
theorem base4_to_base10_conversion : base4_to_base10 21012 = 582 :=
  sorry

end base4_to_base10_conversion_l85_85030


namespace cupcakes_baking_l85_85959

theorem cupcakes_baking (sugar_at_home : ℕ) (bags_bought : ℕ) (cups_per_bag : ℕ)
    (batter_sugar_per_dozen : ℕ) (frosting_sugar_per_dozen : ℕ) :
  sugar_at_home = 3 → bags_bought = 2 → cups_per_bag = 6 →
  batter_sugar_per_dozen = 1 → frosting_sugar_per_dozen = 2 →
  (sugar_at_home + bags_bought * cups_per_bag) / (batter_sugar_per_dozen + frosting_sugar_per_dozen) = 5 :=
by
  intros sugar_at_home_eq bags_bought_eq cups_per_bag_eq batter_sugar_per_dozen_eq frosting_sugar_per_dozen_eq
  rw [sugar_at_home_eq, bags_bought_eq, cups_per_bag_eq, batter_sugar_per_dozen_eq, frosting_sugar_per_dozen_eq]
  simp
  sorry

end cupcakes_baking_l85_85959


namespace tan_tan_interval_solutions_l85_85513

-- Define the function T(x)
def T (x : ℝ) : ℝ := Real.tan x - x - (Real.pi / 4)

-- Problem statement: proving there are 600 solutions
theorem tan_tan_interval_solutions : 
  ∃ (solutions : ℕ), solutions = 600 ∧ 
  ∀ x, 0 ≤ x ∧ x ≤ Real.arctan 1884 → Real.tan x = Real.tan (Real.tan x + (Real.pi / 4)) → 
  ∃ n : ℤ, {(x : ℝ) | T(x) = (n : ℝ) * Real.pi}.nonempty := sorry

end tan_tan_interval_solutions_l85_85513


namespace problem_solution_l85_85579

-- Define the equation given in the problem.
def equation (x : ℝ) : Prop :=
  (4 / (x - 4)) + (6 / (x - 6)) + (18 / (x - 18)) + (20 / (x - 20)) = x^2 - 13x - 6

-- Define the condition that n is the largest real solution to the equation.
def largest_solution (n : ℝ) : Prop :=
  equation n ∧ ∀ x, equation x → x ≤ n

-- State the proposition we need to prove.
theorem problem_solution : ∃ p q r : ℕ, largest_solution (13 + Real.sqrt 72) ∧ 13 + Real.sqrt 72 = p + Real.sqrt (q + Real.sqrt r) ∧ p + q + r = 309 :=
by
  sorry

end problem_solution_l85_85579


namespace circle_area_irrational_l85_85887

theorem circle_area_irrational (m n : ℤ) (hn : n ≠ 0) :
  let d := (m : ℚ) / n in
  let r := d / 2 in
  let A := π * r^2 in
  0 < n → ¬is_rational A :=
by
  intros h
  let d := (m : ℚ) / n
  let r := d / 2
  let A := π * r^2
  sorry

end circle_area_irrational_l85_85887


namespace card_arrangement_l85_85611

theorem card_arrangement (cards : Finset ℕ) (envelopes : Finset ℕ) 
    (cond1 : cards = {1, 2, 3, 4, 5, 6})
    (cond2 : envelopes.card = 3) 
    (cond3 : ∀ e ∈ envelopes, ((cards ∩ e).card = 2))
    (cond4 : ∀ e ∈ envelopes, (1 ∈ e) → (2 ∈ e)) :
  (∃ n : ℕ, n = 36) := 
by
  sorry

end card_arrangement_l85_85611


namespace problem_l85_85115

noncomputable def f (x : ℝ) := Real.log x + (x + 1) / x

noncomputable def g (x : ℝ) := x - 1/x - 2 * Real.log x

theorem problem 
  (x : ℝ) (hx : x > 0) (hxn1 : x ≠ 1) :
  f x > (x + 1) * Real.log x / (x - 1) :=
by
  sorry

end problem_l85_85115


namespace trapezoid_area_l85_85348

-- Define the given conditions
def AB : ℝ := 10
def CD : ℝ := 15
def radius : ℝ := 6
def arc_angle : ℝ := 120

-- Define the problem
theorem trapezoid_area :
  ∃ (A : ℝ), A = (AB + CD) / 2 * 9 ∧ A = 225 / 2 := by
sory

end trapezoid_area_l85_85348


namespace find_m_n_l85_85941

-- Definitions of the conditions
def has_exactly_four_divisors (m : ℕ) : Prop :=
  ∃ (p q : ℕ), p.prime ∧ q.prime ∧ (m = p^2 ∨ m = p * q)

def has_exactly_five_divisors (n : ℕ) : Prop :=
  ∃ (p : ℕ), p.prime ∧ n = p^4

def smallest_with_four_divisors : ℕ :=
  Nat.find (Exists.intro 4
    ⟨has_exactly_four_divisors 4, by 
      dsimp [has_exactly_four_divisors] 
      use [2, 3]; 
      tauto⟩)

def largest_with_five_divisors_lt_50 : ℕ :=
  Nat.find (Exists.intro 16
    ⟨has_exactly_five_divisors 16, by 
      dsimp [has_exactly_five_divisors] 
      use 2; 
      tauto⟩)

-- Theorem to prove the desired result
theorem find_m_n : smallest_with_four_divisors + largest_with_five_divisors_lt_50 = 20 :=
begin
  -- Proof is omitted
  sorry
end

end find_m_n_l85_85941


namespace curve_equation_max_EF_l85_85946

theorem curve_equation 
(points A B : ℝ × ℝ) (P : ℝ × ℝ) 
(hA : A = (-2, 0)) 
(hB : B = (2, 0)) 
(hP : (P.1, P.2)) 
(h_prod_slope : (P.2 / (P.1 + 2)) * (P.2 / (P.1 - 2)) = -1 / 4) : 
P.1 ^ 2 / 4 + P.2 ^ 2 = 1 := 
sorry

theorem max_EF 
(m x₁ x₂ y₁ y₂ k : ℝ) 
(htangent : ∀ x₁ x₂ y₁ y₂, ((x₁, y₁), (x₂, y₂)) ∈ (tangent_points m k)) 
(hm : |m| > 1) 
(hEF_len : EF_length x₁ y₁ x₂ y₂ = |EF| ) : 
|EF| = 2 :=
sorry

end curve_equation_max_EF_l85_85946


namespace shift_down_4_correct_l85_85008

def f (x : ℝ) : ℝ := -3 * x

def g (x : ℝ) : ℝ := -3 * x - 4

theorem shift_down_4_correct :
  ∀ (x : ℝ), g(x) = f(x) - 4 :=
by
  -- implementation goes here
  sorry

end shift_down_4_correct_l85_85008


namespace maximum_a_monotonically_increasing_l85_85533

theorem maximum_a_monotonically_increasing :
  ∃ a : Real, (a = (sqrt 5 - 1)/2) ∧ (∀ x ∈ Icc a (a + 1), (e^x * (-x^2 + 2*x + a)' >= 0)) :=
sorry

end maximum_a_monotonically_increasing_l85_85533


namespace min_value_of_expression_l85_85837

theorem min_value_of_expression
    (a b : ℝ)
    (pos_a : 0 < a)
    (pos_b : 0 < b)
    (tangent_condition : ∃ (m : ℝ), (m - 2 * a = log (m + b)) ∧ (1 = 1 / (m + b))) :
    (∃ (min_value : ℝ), min_value = 8 ∧ ∀ a b : ℝ, 0 < a ∧ 0 < b ∧ ∃ m : ℝ, (m - 2 * a = log (m + b)) ∧ (1 = 1 / (m + b)) → min_value ≤ (1 / a + 2 / b)) :=
by
  sorry

end min_value_of_expression_l85_85837


namespace sqrt_bc_bd_l85_85023

theorem sqrt_bc_bd (C1 C2 : Type) [MetricSpace C1] [MetricSpace C2] 
  (X Y A B C D : C1) (radius1 : ℝ) (radius2 : ℝ)
  (h1 : 2016 * dist A B = dist B C)
  (h2 : dist X Y ≠ 0)
  (h3 : is_tangent A C1)
  (h4 : is_tangent B C2)
  (h5 : A ≠ B)
  (h6 : X ≠ Y)
  (h7 : radius1 = 20) (h8 : radius2 = 16) :
  sqrt (1 + (dist B C / dist B D)) = 2017 :=
sorry

end sqrt_bc_bd_l85_85023


namespace remainder_of_division_l85_85277

theorem remainder_of_division (x : ℕ) (r : ℕ) :
  1584 - x = 1335 ∧ 1584 = 6 * x + r → r = 90 := by
  sorry

end remainder_of_division_l85_85277


namespace keystone_arch_larger_angle_l85_85284

theorem keystone_arch_larger_angle (n : ℕ) (h1 : n = 12) (isosceles_trapezoids : Type)
  (h2 : ∀ t : isosceles_trapezoids, extends_meeting_at_center t) :
  ∃ x : ℝ, x = 97.5 :=
by
  sorry

end keystone_arch_larger_angle_l85_85284


namespace find_x_plus_y_l85_85944

theorem find_x_plus_y (x y : ℝ)
  (h1 : (x - 1)^3 + 2015 * (x - 1) = -1)
  (h2 : (y - 1)^3 + 2015 * (y - 1) = 1)
  : x + y = 2 :=
sorry

end find_x_plus_y_l85_85944


namespace degree_n_plus_1_degree_n_l85_85716

noncomputable def block_similar (n : ℕ) (P Q : ℝ → ℝ) : Prop :=
  ∀ i ∈ finset.range n, (finset.range 2015).image (λ k, P (2015 * i + k)).perm (finset.range 2015).image (λ k, Q (2015 * i + k))

theorem degree_n_plus_1 (n : ℕ) (h : 2 ≤ n) : 
  ∃ (P Q : ℝ → ℝ), (polynomial.degree P = ↑(n+1)) ∧ (polynomial.degree Q = ↑(n+1)) ∧ (block_similar n P Q) ∧ (P ≠ Q) :=
sorry

theorem degree_n (n : ℕ) (h : 2 ≤ n) : 
  ¬ ∃ (P Q : ℝ → ℝ), (polynomial.degree P = ↑n) ∧ (polynomial.degree Q = ↑n) ∧ (block_similar n P Q) ∧ (P ≠ Q) :=
sorry

end degree_n_plus_1_degree_n_l85_85716


namespace no_move_possible_theorem_infinite_moves_possible_theorem_l85_85584

-- Definition for the number of stones such that no move is possible
def no_move_possible (n : ℕ) : ℕ := 2 * n^2 - 2 * n

-- Definition for the number of stones for an infinite sequence of moves
def infinite_moves_possible (n k : ℕ) : Prop := k >= 2 * n^2 - 2 * n

theorem no_move_possible_theorem (n : ℕ) (hn : n > 0) :
  ∃ k, k = no_move_possible n ∧ ∀ (config : fin n × fin n → ℕ), ¬(∃ i j, config ⟨i, j⟩ ≥ 4) :=
sorry

theorem infinite_moves_possible_theorem (n k : ℕ) (hn : n > 0) (hk : k >= 2 * n^2 - 2 * n) :
  ∃ (config : fin n × fin n → ℕ),
    (∀ (i j : fin n), config ⟨i, j⟩ ≥ 4 → exists_adj_move config ⟨i, j⟩) ∧ 
    (∀ (i j : fin n), config ⟨i, j⟩ ≥ 2) :=
sorry

-- Auxiliary definitions and lemmas can be defined here if needed to formalize the moves and configurations. 


end no_move_possible_theorem_infinite_moves_possible_theorem_l85_85584


namespace proposition_verification_l85_85109

theorem proposition_verification :
  let p1: Prop := (∃ x, 2 * sin (2 * x - π / 3) = 2 ∧ x = 5 * π / 12)
  let p2: Prop := (∃ x y, (x, y) = (π / 2, 0))
  let p3: Prop := (∀ x, 0 ≤ x ∧ x ≤ π / 2 → 0 ≤ sin x ∧ sin x ≤ 1)  -- This uses a weaker form for the properties of sine in the first quadrant.
  let p4: Prop := (∀ x1 x2, sin (2 * x1 - π / 4) = sin (2 * x2 - π / 4) → x1 - x2 = k * π ∨ x1 + x2 = k * π + 3 * π / 4)  -- Expanded form for k in integers
  (p1 ∧ p2) ∧ ¬(p3 ∧ p4) := 
begin
  sorry
end

end proposition_verification_l85_85109


namespace hexagon_midpoints_form_equilateral_l85_85368

open EuclideanGeometry

variables {A B C D E F O P Q R : Point}
variables {R : ℝ}

noncomputable def midpoint (X Y : Point) : Point := sorry

noncomputable def is_equilateral (X Y Z : Point) : Prop := sorry

noncomputable def is_inscribed_in_circle (hexagon : List Point) (O : Point) (R : ℝ) : Prop := sorry

theorem hexagon_midpoints_form_equilateral
  (hexagon_inscribed : is_inscribed_in_circle [A, B, C, D, E, F] O R)
  (AB_eq_R : dist A B = R)
  (CD_eq_R : dist C D = R)
  (EF_eq_R : dist E F = R) :
  is_equilateral (midpoint B C) (midpoint D E) (midpoint F A) :=
sorry

end hexagon_midpoints_form_equilateral_l85_85368


namespace prize_winners_l85_85796

variable (Elaine Frank George Hannah : Prop)

axiom ElaineImpliesFrank : Elaine → Frank
axiom FrankImpliesGeorge : Frank → George
axiom GeorgeImpliesHannah : George → Hannah
axiom OnlyTwoWinners : (Elaine ∧ Frank ∧ ¬George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ Frank ∧ George ∧ ¬Hannah) ∨ (¬Elaine ∧ Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah)

theorem prize_winners : (Elaine ∧ Frank ∧ ¬George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ Frank ∧ George ∧ ¬Hannah) ∨ (¬Elaine ∧ Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah) → (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah) :=
by
  sorry

end prize_winners_l85_85796


namespace like_term_exists_l85_85615

variable (a b : ℝ) (x y : ℝ)

theorem like_term_exists : ∃ b : ℝ, b * x^5 * y^3 = 3 * x^5 * y^3 ∧ b ≠ a :=
by
  -- existence of b
  use 3
  -- proof is omitted
  sorry

end like_term_exists_l85_85615


namespace omicron_variant_diameter_in_scientific_notation_l85_85276

/-- Converting a number to scientific notation. -/
def to_scientific_notation (d : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  d = a * 10 ^ n

theorem omicron_variant_diameter_in_scientific_notation :
  to_scientific_notation 0.00000011 1.1 (-7) :=
by
  sorry

end omicron_variant_diameter_in_scientific_notation_l85_85276


namespace mark_bread_baking_time_l85_85966

/--
Mark is baking bread. 
He has to let it rise for 120 minutes twice. 
He also needs to spend 10 minutes kneading it and 30 minutes baking it. 
Prove that the total time Mark takes to finish making the bread is 280 minutes.
-/
theorem mark_bread_baking_time :
  let rising_time := 120 * 2
  let kneading_time := 10
  let baking_time := 30
  rising_time + kneading_time + baking_time = 280 := 
by
  let rising_time := 120 * 2
  let kneading_time := 10
  let baking_time := 30
  have rising_time_eq : rising_time = 240 := rfl
  have kneading_time_eq : kneading_time = 10 := rfl
  have baking_time_eq : baking_time = 30 := rfl
  calc
    rising_time + kneading_time + baking_time
        = 240 + 10 + 30 : by rw [rising_time_eq, kneading_time_eq, baking_time_eq]
    ... = 280 : by norm_num

end mark_bread_baking_time_l85_85966


namespace num_of_four_digit_numbers_not_greater_than_5104_l85_85817

/--
Among four-digit numbers formed without repeating digits from the set {0, 1, 4, 5, 8}, 
prove that the total number of four-digit numbers not greater than 5104 is 55.
-/
theorem num_of_four_digit_numbers_not_greater_than_5104 :
  let digits := {0, 1, 4, 5, 8}
  in count (λ n, n < 5105 ∧ ∀ d ∈ digits, digit_occur d n ≤ 1) = 55 := 
sorry

end num_of_four_digit_numbers_not_greater_than_5104_l85_85817


namespace log_expression_eq_one_l85_85779

theorem log_expression_eq_one : log 5 / log 2 * log 2 / log 3 * log 3 / log 5 = 1 :=
by sorry

end log_expression_eq_one_l85_85779


namespace allan_has_4_more_balloons_than_jake_l85_85398

namespace BalloonProblem

def initial_balloons_allan : Nat := 6
def initial_balloons_jake : Nat := 2
def additional_balloons_jake : Nat := 3
def additional_balloons_allan : Nat := 4
def given_balloons_jake : Nat := 2
def given_balloons_allan : Nat := 3

def final_balloons_allan : Nat := (initial_balloons_allan + additional_balloons_allan) - given_balloons_allan
def final_balloons_jake : Nat := (initial_balloons_jake + additional_balloons_jake) - given_balloons_jake

theorem allan_has_4_more_balloons_than_jake :
  final_balloons_allan = final_balloons_jake + 4 :=
by
  -- proof is skipped with sorry
  sorry

end BalloonProblem

end allan_has_4_more_balloons_than_jake_l85_85398


namespace ways_to_divide_8_friends_l85_85521

theorem ways_to_divide_8_friends : (4 : ℕ) ^ 8 = 65536 := by
  sorry

end ways_to_divide_8_friends_l85_85521


namespace exists_x0_l85_85888

open Real

variable {f : ℝ → ℝ}

theorem exists_x0 (h_cont : ContinuousOn f (Set.Icc 1 2))
    (h_int : ∫ x in 1..2, f x = 73 / 24) :
    ∃ x0 ∈ Set.Ioo 1 2, (x0^2 < f x0) ∧ (f x0 < x0^3) :=
by
  sorry

end exists_x0_l85_85888


namespace gcd_lcm_240_l85_85700

theorem gcd_lcm_240 (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 240) : 
  ∃ n, ∃ gcds : Finset ℕ, (gcds.card = n) ∧ (Nat.gcd a b ∈ gcds) :=
by
  sorry

end gcd_lcm_240_l85_85700


namespace function_properties_l85_85087

variable {α : Type _} [LinearOrder α] [AddCommGroup α] [HasSmul ℝ α]

-- Given conditions
def is_even (f : α → α) : Prop := ∀ x, f(x) = f(-x)
def increasing_interval (f : α → α) (a b : α) : Prop := ∀ x y, a < x ∧ y < b → x < y → f x ≤ f y
def domain (f : α → α) (a b : α) : Prop := ∀ x, a < x ∧ x < b

-- Translated theorem statement
theorem function_properties (f : ℝ → ℝ) :
  is_even f →
  domain f (-10 : ℝ) 10 →
  increasing_interval f 2 6 →
  increasing_interval (λ x, f (2 - x)) 4 8 ∧
  ∀ x, f(2-x) = f(x-2) → ¬ f x = f(-x) →
  is_even (λ x, f (x - 2)) :=
by
  sorry

end function_properties_l85_85087


namespace count_valid_n_l85_85058

-- Define the problem conditions and the correct answer
noncomputable def isValidN (n : ℕ) : Prop :=
  ∀ (t : ℝ), (Complex.sin t + Complex.cos t * Complex.I)^(2*n) = Complex.sin (2*n*t) + Complex.cos (2*n*t) * Complex.I

theorem count_valid_n :
  (Finset.filter (λ n, isValidN n) (Finset.range 501)).card = 125 := 
by 
  -- Placeholder for the proof
  sorry

end count_valid_n_l85_85058


namespace entrance_exam_correct_answers_l85_85185

theorem entrance_exam_correct_answers (c w : ℕ) 
  (h1 : c + w = 70) 
  (h2 : 3 * c - w = 38) : 
  c = 27 := 
sorry

end entrance_exam_correct_answers_l85_85185


namespace monotonic_function_zeros_l85_85889

theorem monotonic_function_zeros (f : ℝ → ℝ) (h_mono : monotone f ∨ monotone (λ x, -f x)) : 
  ∃! x, f x = 0 :=
sorry

end monotonic_function_zeros_l85_85889


namespace base4_to_base10_conversion_l85_85032

theorem base4_to_base10_conversion : 
  2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582 :=
by 
  sorry

end base4_to_base10_conversion_l85_85032


namespace average_weight_increase_l85_85901

theorem average_weight_increase (A : ℝ) :
  let initial_count := 10
  let old_weight := 70
  let new_weight := 110
  let old_total_weight := initial_count * A
  let weight_increase := new_weight - old_weight
  let new_total_weight := old_total_weight + weight_increase in
  (new_total_weight / initial_count) - A = 4 := by
  sorry

end average_weight_increase_l85_85901


namespace only_true_statement_l85_85419

-- Definitions for conditions
def period_sin_cos (x : ℝ) : Prop := ∃ p > 0, ∀ x, sin x ^ 4 - cos x ^ 4 = - cos (2 * x) ∧ p = pi
def shift_3sin (x : ℝ) : Prop := ∀ b, 3 * sin b ≠ 3 * sin (2 * b + x)
def decreasing_sin (x : ℝ) : Prop := ¬(∀ x ∈ set.Icc 0 pi, deriv sin x < 0)

-- Problem restating
theorem only_true_statement is_1 : period_sin_cos ∧ shift_3sin ∧ decreasing_sin → is_1 := by
  sorry

end only_true_statement_l85_85419


namespace sara_quarters_final_l85_85257

def initial_quarters : ℕ := 21
def quarters_from_dad : ℕ := 49
def quarters_spent_at_arcade : ℕ := 15
def dollar_bills_from_mom : ℕ := 2
def quarters_per_dollar : ℕ := 4

theorem sara_quarters_final :
  (initial_quarters + quarters_from_dad - quarters_spent_at_arcade + dollar_bills_from_mom * quarters_per_dollar) = 63 :=
by
  sorry

end sara_quarters_final_l85_85257


namespace distinct_factors_of_number_l85_85410

theorem distinct_factors_of_number : 
  let n := 2^6 * 3^5 * 5^3 * 7^2 in
  nat.divisor_count n = 504 := 
by
  let n := 2^6 * 3^5 * 5^3 * 7^2
  sorry

end distinct_factors_of_number_l85_85410


namespace number_of_boys_is_12500_l85_85425

-- Define the number of boys and girls in the school
def numberOfBoys (B : ℕ) : ℕ := B
def numberOfGirls : ℕ := 5000

-- Define the total attendance
def totalAttendance (B : ℕ) : ℕ := B + numberOfGirls

-- Define the condition for the percentage increase from boys to total attendance
def percentageIncreaseCondition (B : ℕ) : Prop :=
  totalAttendance B = B + Int.ofNat numberOfGirls

-- Statement to prove
theorem number_of_boys_is_12500 (B : ℕ) (h : totalAttendance B = B + numberOfGirls) : B = 12500 :=
sorry

end number_of_boys_is_12500_l85_85425


namespace problem_CorrectOption_l85_85126

def setA : Set ℝ := {y | ∃ x : ℝ, y = |x| - 1}
def setB : Set ℝ := {x | x ≥ 2}

theorem problem_CorrectOption : setA ∩ setB = setB := 
  sorry

end problem_CorrectOption_l85_85126


namespace inverse_of_49_mod_103_l85_85092

theorem inverse_of_49_mod_103 (h : 7⁻¹ ≡ 55 [MOD 103]) : 49⁻¹ ≡ 38 [MOD 103] :=
sorry

end inverse_of_49_mod_103_l85_85092


namespace find_p_l85_85489

noncomputable def binomial_parameter (n : ℕ) (p : ℚ) (E : ℚ) (D : ℚ) : Prop :=
  E = n * p ∧ D = n * p * (1 - p)

theorem find_p (n : ℕ) (p : ℚ) 
  (hE : n * p = 50)
  (hD : n * p * (1 - p) = 30)
  : p = 2 / 5 :=
sorry

end find_p_l85_85489


namespace quadrilateral_diagonals_fixed_point_l85_85026

theorem quadrilateral_diagonals_fixed_point
  (O : Point) (r : ℝ) (K : Circle O r) (A : Point)
  (hA_outside : ¬ A ∈ K) (epsilon : Line)
  (hAO_ne : epsilon ≠ Line_through A O)
  (B C : Point) (hBC : K.contains B ∧ K.contains C ∧ betweenness A B C epsilon)
  (symmetric_epsilon : Line)
  (hSymm : symmetric_epsilon = symmetric_line epsilon (Line_through A O))
  (E D : Point) (hED : K.contains E ∧ K.contains D ∧ betweenness A E D symmetric_epsilon)
  : ∃ P : Point, intersection_point (Line_through B D) (Line_through C E) = P ∧ 
    P ∈ Line_through A O := sorry

end quadrilateral_diagonals_fixed_point_l85_85026


namespace range_of_a_l85_85947

def A (x : ℝ) : Prop := (x - 1) * (x - 2) ≥ 0
def B (a x : ℝ) : Prop := x ≥ a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, A x ∨ B a x) ↔ a ≤ 1 :=
sorry

end range_of_a_l85_85947


namespace drying_time_for_whites_l85_85765

-- Define the washing and drying times in minutes.
def whites_wash_time : ℕ := 72
def darks_wash_time : ℕ := 58
def darks_dry_time : ℕ := 65
def colors_wash_time : ℕ := 45
def colors_dry_time : ℕ := 54
def total_time : ℕ := 344

-- Define the theorem that needs to be proved.
theorem drying_time_for_whites : 
  let W := total_time - (whites_wash_time + darks_wash_time + darks_dry_time + colors_wash_time + colors_dry_time)
  in W = 50 := 
by
  -- Required proof will go here.
  sorry

end drying_time_for_whites_l85_85765


namespace probability_divisible_by_4_l85_85758

theorem probability_divisible_by_4 {a b : ℕ} (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) :
  let N := 100 * a + 10 * b + 7 in
  (N % 4 = 0) = false :=
by
  sorry

end probability_divisible_by_4_l85_85758


namespace cos_pi_minus_theta_l85_85490

-- We define the conditions given in the problem
def point : ℝ × ℝ := (4, -3)

-- We extract the x and y coordinates from the point
def x : ℝ := point.1
def y : ℝ := point.2

-- Define the radius r using the Pythagorean theorem
noncomputable def r : ℝ := real.sqrt (x^2 + y^2)

-- Define cos(theta) using the definition of cosine in a right triangle
noncomputable def cos_theta : ℝ := x / r

-- State the final goal using the trigonometric identity for cos(pi - theta)
theorem cos_pi_minus_theta : cos (π - θ) = -4 / 5 :=
by 
  sorry -- proofs and further derivations are omitted

end cos_pi_minus_theta_l85_85490


namespace leftover_grass_seed_coverage_l85_85430

/-
Question: How many extra square feet could the leftover grass seed cover after Drew reseeds his lawn?

Conditions:
- One bag of grass seed covers 420 square feet of lawn.
- The lawn consists of a rectangular section and a triangular section.
- Rectangular section:
    - Length: 32 feet
    - Width: 45 feet
- Triangular section:
    - Base: 25 feet
    - Height: 20 feet
- Triangular section requires 1.5 times the standard coverage rate.
- Drew bought seven bags of seed.

Answer: The leftover grass seed coverage is 1125 square feet.
-/

theorem leftover_grass_seed_coverage
  (bag_coverage : ℕ := 420)
  (rect_length : ℕ := 32)
  (rect_width : ℕ := 45)
  (tri_base : ℕ := 25)
  (tri_height : ℕ := 20)
  (coverage_multiplier : ℕ := 15)  -- Using 15 instead of 1.5 for integer math
  (bags_bought : ℕ := 7) :
  (bags_bought * bag_coverage - 
    (rect_length * rect_width + tri_base * tri_height * coverage_multiplier / 20) = 1125) :=
  by {
    -- Placeholder for proof steps
    sorry
  }

end leftover_grass_seed_coverage_l85_85430


namespace integral_of_g_l85_85206

noncomputable def f (x : ℝ) : ℝ := x * 2^(-x)

noncomputable def g (t : ℝ) : ℝ :=
if h : t ≤ 1 / Real.log 2 - 1 then f t else f (t + 1)

theorem integral_of_g : (∫ t in 0..2, g t) = (∫ t in 0..(1 / Real.log 2 - 1), f t) + (∫ t in (1 / Real.log 2 - 1)..2, f (t + 1)) :=
by
  sorry

end integral_of_g_l85_85206


namespace number_of_real_roots_of_f_f_eq_0_l85_85222

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else real.sqrt x - 1

theorem number_of_real_roots_of_f_f_eq_0 :
  (set.to_finset {x : ℝ | f (f x) = 0}).card = 3 :=
sorry

end number_of_real_roots_of_f_f_eq_0_l85_85222


namespace find_angle_B_find_range_sqrt3a_minus_c_l85_85914

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {triangle : Type}

def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  (2 * a - real.sqrt 3 * c) ^ 2 = 4 * b ^ 2 - c ^ 2 ∧ 
  0 < B ∧ B < π ∧ a = 2 * real.sin A ∧ c = 2 * real.sin C ∧
  b = 1

theorem find_angle_B (h : triangleABC a b c A B C) : B = π / 6 := by
  sorry

theorem find_range_sqrt3a_minus_c (h : triangleABC a b c A B C) : -1 < sqrt 3 * a - c ∧ sqrt 3 * a - c ≤ 2 := by
  sorry

end find_angle_B_find_range_sqrt3a_minus_c_l85_85914


namespace base_conversion_correct_l85_85264

theorem base_conversion_correct : ∃ C D : ℕ, C = 5 ∧ D = 3 ∧ 8 * C + D = 43 :=
by {
  use 5,
  use 3,
  split,
  { reflexivity },
  split,
  { reflexivity },
  sorry
}

end base_conversion_correct_l85_85264


namespace loss_percentage_is_8_l85_85395

def selling_price_gain (cp : ℝ) : ℝ :=
  cp * 1.04

def original_selling_price (sp_gain : ℝ) : ℝ :=
  sp_gain - 140

def loss_percentage (cp sp : ℝ) : ℝ :=
  (cp - sp) / cp * 100

theorem loss_percentage_is_8 :
  let CP := 1166.67 in
  let SP_gain := selling_price_gain CP in
  let SP := original_selling_price SP_gain in
  loss_percentage CP SP = 8 :=
by
  let CP := 1166.67
  let SP_gain := selling_price_gain CP
  let SP := original_selling_price SP_gain
  have : loss_percentage CP SP = 8 := sorry
  exact this

end loss_percentage_is_8_l85_85395


namespace cos_pi_minus_theta_l85_85493

-- Define the point (4, -3)
def point := (4 : ℕ, -3 : ℤ)

-- Define the cosine value of θ using the point (4, -3)
def cos_theta : ℚ := 4 / real.sqrt (4 ^ 2 + (-3 : ℚ) ^ 2)

-- The goal is to prove that cos (π - θ) = -4/5 given the point condition
theorem cos_pi_minus_theta (θ : ℝ) (h : (cos θ = 4 / 5)) : cos (π - θ) = -4 / 5 :=
by {
  sorry
}

end cos_pi_minus_theta_l85_85493


namespace new_average_of_remaining_students_l85_85996

theorem new_average_of_remaining_students 
  (avg_initial_score : ℝ)
  (num_initial_students : ℕ)
  (dropped_score : ℝ)
  (num_remaining_students : ℕ)
  (new_avg_score : ℝ) 
  (h_avg : avg_initial_score = 62.5)
  (h_num_initial : num_initial_students = 16)
  (h_dropped : dropped_score = 55)
  (h_num_remaining : num_remaining_students = 15)
  (h_new_avg : new_avg_score = 63) :
  let total_initial_score := avg_initial_score * num_initial_students
  let total_remaining_score := total_initial_score - dropped_score
  let calculated_new_avg := total_remaining_score / num_remaining_students
  calculated_new_avg = new_avg_score := 
by
  -- The proof will be provided here
  sorry

end new_average_of_remaining_students_l85_85996


namespace viewers_difference_l85_85904

theorem viewers_difference :
  let second_game := 80
  let first_game := second_game - 20
  let third_game := second_game + 15
  let fourth_game := third_game + (third_game / 10)
  let total_last_week := 350
  let total_this_week := first_game + second_game + third_game + fourth_game
  total_this_week - total_last_week = -10 := 
by
  sorry

end viewers_difference_l85_85904


namespace Carver_school_earnings_l85_85984

noncomputable def total_earnings_Carver_school : ℝ :=
  let base_payment := 20
  let total_payment := 900
  let Allen_days := 7 * 3
  let Balboa_days := 5 * 6
  let Carver_days := 4 * 10
  let total_student_days := Allen_days + Balboa_days + Carver_days
  let adjusted_total_payment := total_payment - 3 * base_payment
  let daily_wage := adjusted_total_payment / total_student_days
  daily_wage * Carver_days

theorem Carver_school_earnings : 
  total_earnings_Carver_school = 369.6 := 
by 
  sorry

end Carver_school_earnings_l85_85984


namespace arithmetic_seq_first_term_l85_85932

theorem arithmetic_seq_first_term (S : ℕ → ℚ) (a : ℚ) (n : ℕ) (h1 : ∀ n, S n = (n * (2 * a + (n - 1) * 5)) / 2)
  (h2 : ∀ n, S (4 * n) / S n = 16) : a = 5 / 2 := 
sorry

end arithmetic_seq_first_term_l85_85932


namespace expected_value_of_fair_eight_sided_die_l85_85734

def fair_eight_sided_die_outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def win_amount (n : ℕ) : ℕ := n ^ 3

def expected_value (outcomes : List ℕ) (win : ℕ → ℕ) : ℚ :=
  (1 / outcomes.length : ℚ) * outcomes.sum_by (λ n => win n)

theorem expected_value_of_fair_eight_sided_die :
  expected_value fair_eight_sided_die_outcomes win_amount = 162 := by
  sorry

end expected_value_of_fair_eight_sided_die_l85_85734


namespace tan_omega_decreasing_l85_85119

theorem tan_omega_decreasing (ω : ℝ) :
  (∀ x ∈ Ioo (-(π/2)) (π/2), deriv (λ x, tan (ω * x)) x < 0) → (-1 ≤ ω ∧ ω < 0) :=
by
  intro h
  -- Proof steps would go here
  sorry

end tan_omega_decreasing_l85_85119


namespace mark_bread_time_l85_85963

def rise_time1 : Nat := 120
def rise_time2 : Nat := 120
def kneading_time : Nat := 10
def baking_time : Nat := 30

def total_time : Nat := rise_time1 + rise_time2 + kneading_time + baking_time

theorem mark_bread_time : total_time = 280 := by
  sorry

end mark_bread_time_l85_85963


namespace tetrahedron_volume_correct_l85_85827

noncomputable def tetrahedron_volume (A B C D : Type) [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C] [EuclideanSpace ℝ D]
  (AB : dist A B = 2)
  (CD : dist C D = 2 * real.sqrt 5)
  (AC : dist A C = 3)
  (BD : dist B D = 3)
  (AD : dist A D = real.sqrt 5)
  (BC : dist B C = real.sqrt 5) : ℝ :=
  (1 / 3) * (1 / 2) * 2 * real.sqrt 5 * (4 / real.sqrt 5)

theorem tetrahedron_volume_correct 
  (A B C D : Type) [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C] [EuclideanSpace ℝ D]
  (h1 : dist A B = 2)
  (h2 : dist C D = 2 * real.sqrt 5)
  (h3 : dist A C = 3)
  (h4 : dist B D = 3)
  (h5 : dist A D = real.sqrt 5)
  (h6 : dist B C = real.sqrt 5) :
  tetrahedron_volume A B C D h1 h2 h3 h4 h5 h6 = 4 / 3 :=
sorry

end tetrahedron_volume_correct_l85_85827


namespace equilateral_triangle_isosceles_triangle_l85_85459

noncomputable def is_equilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c

noncomputable def is_isosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem equilateral_triangle (a b c : ℝ) (h : abs (a - b) + abs (b - c) = 0) : is_equilateral a b c :=
  sorry

theorem isosceles_triangle (a b c : ℝ) (h : (a - b) * (b - c) = 0) : is_isosceles a b c :=
  sorry

end equilateral_triangle_isosceles_triangle_l85_85459


namespace swap_values_l85_85990

theorem swap_values : ∀ (a b : ℕ), a = 3 → b = 2 → 
  (∃ c : ℕ, c = b ∧ (b = a ∧ (a = c ∨ a = 2 ∧ b = 3))) :=
by
  sorry

end swap_values_l85_85990


namespace SharonOranges_l85_85561

-- Define the given conditions
def JanetOranges : Nat := 9
def TotalOranges : Nat := 16

-- Define the statement that needs to be proven
theorem SharonOranges (J : Nat) (T : Nat) (S : Nat) (hJ : J = 9) (hT : T = 16) (hS : S = T - J) : S = 7 := by
  -- (proof to be filled in later)
  sorry

end SharonOranges_l85_85561


namespace fraction_power_mult_equality_l85_85415

-- Define the fraction and the power
def fraction := (1 : ℚ) / 3
def power : ℚ := fraction ^ 4

-- Define the multiplication
def result := 8 * power

-- Prove the equality
theorem fraction_power_mult_equality : result = 8 / 81 := by
  sorry

end fraction_power_mult_equality_l85_85415


namespace eggs_eaten_in_afternoon_l85_85775

theorem eggs_eaten_in_afternoon (initial : ℕ) (morning : ℕ) (final : ℕ) (afternoon : ℕ) :
  initial = 20 → morning = 4 → final = 13 → afternoon = initial - morning - final → afternoon = 3 :=
by
  intros h_initial h_morning h_final h_afternoon
  rw [h_initial, h_morning, h_final] at h_afternoon
  linarith

end eggs_eaten_in_afternoon_l85_85775


namespace minimum_p_plus_q_l85_85861

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then 4 * Real.log x + 1 else 2 * x - 1

theorem minimum_p_plus_q (p q : ℝ) (hpq : p ≠ q) (hf : f p + f q = 2) :
  p + q = 3 - 2 * Real.log 2 := by
  sorry

end minimum_p_plus_q_l85_85861


namespace minimum_value_of_f_l85_85071

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_of_f (x : ℝ) (h : x ≥ 5 / 2) : ∃ y ≥ (5 / 2), f y = 1 := by
  sorry

end minimum_value_of_f_l85_85071


namespace gym_class_students_correct_l85_85949

noncomputable def check_gym_class_studens :=
  let P1 := 15
  let P2 := 5
  let P3 := 12.5
  let P4 := 9.166666666666666
  let P5 := 8.333333333333334
  P1 = P2 + 10 ∧
  P2 = 2 * P3 - 20 ∧
  P3 = P4 + P5 - 5 ∧
  P4 = (1 / 2) * P5 + 5

theorem gym_class_students_correct : check_gym_class_studens := by
  simp [check_gym_class_studens]
  sorry

end gym_class_students_correct_l85_85949


namespace minimum_value_of_f_l85_85072

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_of_f (x : ℝ) (h : x ≥ 5 / 2) : ∃ y ≥ (5 / 2), f y = 1 := by
  sorry

end minimum_value_of_f_l85_85072


namespace statement_b_statement_d_l85_85063

-- Define the function f
variable {f : ℝ → ℝ}

-- Define the given conditions
axiom cond1 (a b : ℝ) : a + b = 4
axiom cond2 : ∀ x : ℝ, f(x + 2) = -f(-x - 2)
axiom cond3 : ∀ x1 x2 : ℝ, (2 ≤ x1 ∧ 2 ≤ x2 ∧ x1 < x2) → (f(x2) - f(x1)) / (x2 - x1) > 0

-- Statement to prove monotonicity on R
theorem statement_b : ∀ x1 x2 : ℝ, (x1 < x2) → f(x1) ≤ f(x2) := by
  sorry

-- Statement to prove the solution set of the inequality
theorem statement_d (a b x : ℝ) (h : f(a) + f(b) + f(x) < 0) : x < 2 := by
  sorry

end statement_b_statement_d_l85_85063


namespace like_term_exists_l85_85614

variable (a b : ℝ) (x y : ℝ)

theorem like_term_exists : ∃ b : ℝ, b * x^5 * y^3 = 3 * x^5 * y^3 ∧ b ≠ a :=
by
  -- existence of b
  use 3
  -- proof is omitted
  sorry

end like_term_exists_l85_85614


namespace question1_question2_l85_85118

noncomputable def f (x : ℝ) := 2*x^3 - 12*x

theorem question1 (a b c : ℝ) (h1 : a ≠ 0) (h2 :  ∀ x, f (-x) = -f (x)) (h3 : f (1) = 2*(1)^3 - 12*(1) ∧ f' (1) = -6) :
  a = 2 ∧ b = -12 ∧ c = 0 := 
sorry

theorem question2 :
  (∀ x, f' (x) > 0 ↔ x ∈ Ioo (-∞) (-sqrt 2) ∪ Ioo (sqrt 2) (∞)) ∧ 
  is_max_on f [interval (-1) 3] =
  18 ∧ is_min_on f [interval (-1) 3] = -8* sqrt(2) := 
sorry

end question1_question2_l85_85118


namespace num_subsets_of_intersection_l85_85507

def M : Set ℤ := {-4, -3, -2, -1}
def N : Set ℤ := {x | 3^x < (1/9)}

def M_inter_N : Set ℤ := M ∩ {x | x < -2}

theorem num_subsets_of_intersection : (M_inter_N.card : ℤ) = 4 :=
by sorry -- Skip the proof here

end num_subsets_of_intersection_l85_85507


namespace golden_ratio_problem_l85_85907

-- Definitions based on conditions
def m : ℝ := 2 * Real.sin (Real.pi * 18 / 180)
def n : ℝ := 4 - m^2

-- The main statement to prove
theorem golden_ratio_problem :
  (m + Real.sqrt n) / Real.sin (Real.pi * 63 / 180) = 2 * Real.sqrt 2 :=
by
  sorry

end golden_ratio_problem_l85_85907


namespace semicircle_contains_three_of_four_points_hemisphere_contains_four_of_five_points_l85_85720

-- Definition and Theorem for Part 1
def circle : Type := sorry -- Define a circle type
def points_on_circle (p1 p2 p3 p4 : circle) : Prop := sorry -- Define 4 points on a circle
def exists_semicircle_contains_three (p1 p2 p3 p4 : circle) : Prop :=
  ∃ semicircle, ∀ p ∈ {p1, p2, p3, p4}, p ∈ semicircle → (p1 ∈ semicircle ∧ p2 ∈ semicircle ∧ p3 ∈ semicircle)

theorem semicircle_contains_three_of_four_points (p1 p2 p3 p4 : circle) (h : points_on_circle p1 p2 p3 p4) :
  exists_semicircle_contains_three p1 p2 p3 p4 :=
sorry

-- Definition and Theorem for Part 2
def sphere : Type := sorry -- Define a sphere type
def points_on_sphere (p1 p2 p3 p4 p5 : sphere) : Prop := sorry -- Define 5 points on a sphere
def exists_hemisphere_contains_four (p1 p2 p3 p4 p5 : sphere) : Prop :=
  ∃ hemisphere, ∀ p ∈ {p1, p2, p3, p4, p5}, p ∈ hemisphere → (p1 ∈ hemisphere ∧ p2 ∈ hemisphere ∧ p3 ∈ hemisphere ∧ p4 ∈ hemisphere)

theorem hemisphere_contains_four_of_five_points (p1 p2 p3 p4 p5 : sphere) (h : points_on_sphere p1 p2 p3 p4 p5) :
  exists_hemisphere_contains_four p1 p2 p3 p4 p5 :=
sorry

end semicircle_contains_three_of_four_points_hemisphere_contains_four_of_five_points_l85_85720


namespace price_decrease_l85_85665

theorem price_decrease (P : ℝ) (h₁ : 1.25 * P = P * 1.25) (h₂ : 1.10 * P = P * 1.10) :
  1.25 * P * (1 - 12 / 100) = 1.10 * P :=
by
  sorry

end price_decrease_l85_85665


namespace range_of_a_l85_85862
noncomputable def f (a x : ℝ) : ℝ := log (x + 2) - (x^2) / (2 * a)

theorem range_of_a (a : ℝ) (h_non_zero : a ≠ 0) (h_extreme : ∃ x₀, (∃ f' = (λ (x : ℝ), 1 / (x + 2) - x / a), f' x₀ = 0) ∧ x₀ ∉ set.Icc (Real.exp 1 + 2) (Real.exp 2 + 2))
(h_non_neg : ∀ x ∈ set.Icc (Real.exp 1 + 2) (Real.exp 2 + 2), f a x ≥ 0) : a > Real.exp 4 + 2 * Real.exp 2 :=
sorry

end range_of_a_l85_85862


namespace carl_drives_total_hours_in_two_weeks_l85_85681

theorem carl_drives_total_hours_in_two_weeks
  (daily_hours : ℕ) (additional_weekly_hours : ℕ) :
  (daily_hours = 2) 
  → (additional_weekly_hours = 6)
  → (14 * daily_hours + 2 * additional_weekly_hours = 40) := 
by
  intros h1 h2
  rw [h1, h2]
  calc
  14 * 2 + 2 * 6 = 28 + 12 := by ring
                ... = 40   := by rfl

end carl_drives_total_hours_in_two_weeks_l85_85681


namespace hyperbola_equation_l85_85852

theorem hyperbola_equation (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0)
  (dist_condition : a + p / 2 = 3)
  (intersection_condition_1 : -p / 2 = -1)
  (intersection_condition_2 : -b/a * (p / 2) = -1) :
  (a = 2) ∧ (b = 2) ∧ (p = 2) → ( ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (x^2 / 4 - y^2 / 4 = 1)) :=
begin
  intro h,
  sorry
end

end hyperbola_equation_l85_85852


namespace unique_ways_to_divide_half_day_l85_85737

-- Lean statement for the problem
theorem unique_ways_to_divide_half_day
  (n m : ℕ) -- n and m are positive integers
  (h1 : n * m = 43200)
  (h2 : n > 0)
  (h3 : m > 0) :
  (nat.divisors 43200).length = 105 :=
by
  sorry

end unique_ways_to_divide_half_day_l85_85737


namespace largest_prime_divisor_of_sum_of_squares_l85_85802

def a : ℕ := 35
def b : ℕ := 84

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, Prime p ∧ p = 13 ∧ (a^2 + b^2) % p = 0 := by
  sorry

end largest_prime_divisor_of_sum_of_squares_l85_85802


namespace ac_greater_than_bd_iff_l85_85587

variable (A B C D : Point)
variable [InCircle A B C D] -- This is conceptual pseudocode, as Lean/Mathlib may have a different signature for inscribed quadrilaterals or circles.

theorem ac_greater_than_bd_iff (h : inscribed_quadrilateral A B C D) :
  dist A C > dist B D ↔ (dist A D - dist B C) * (dist A B - dist C D) > 0 := 
sorry

end ac_greater_than_bd_iff_l85_85587


namespace find_abc_l85_85935

theorem find_abc :
  ∃ a b c : ℝ, (∀ x : ℝ, (x < -6 ∨ |x - 30| ≤ 2) ↔ ((x - a) * (x - b) / (x - c) ≤ 0)) ∧ a < b ∧ a + 2 * b + 3 * c = 74 :=
by
  sorry

end find_abc_l85_85935


namespace second_sheet_width_l85_85999

noncomputable def width_of_second_sheet (W : ℝ) : Prop := 
  let first_sheet_area := 2 * (11 * 19)
  let second_sheet_area := 2 * (11 * W)
  first_sheet_area = second_sheet_area + 100

theorem second_sheet_width : ∃ W : ℝ, width_of_second_sheet W ∧ W ≈ 14.45 :=
by 
  sorry

end second_sheet_width_l85_85999


namespace megatek_manufacturing_percentage_l85_85993

theorem megatek_manufacturing_percentage :
  ∀ (total_degrees manufacturing_degrees total_percentage : ℝ),
  total_degrees = 360 → manufacturing_degrees = 216 → total_percentage = 100 →
  (manufacturing_degrees / total_degrees) * total_percentage = 60 :=
by
  intros total_degrees manufacturing_degrees total_percentage H1 H2 H3
  rw [H1, H2, H3]
  sorry

end megatek_manufacturing_percentage_l85_85993


namespace jean_average_speed_correct_l85_85020

-- Definitions of constants and variables based on conditions
def half_distance_speed_chantal_initial : ℝ := 5
def half_distance_speed_chantal_reduction : ℝ := 2.5
def descent_speed_chantal : ℝ := 4
def distance_halfway : ℝ := d
def total_distance_to_fire_tower : ℝ := 2 * distance_halfway

noncomputable def time_chantal_first_half : ℝ := distance_halfway / half_distance_speed_chantal_initial
noncomputable def time_chantal_second_half : ℝ := distance_halfway / half_distance_speed_chantal_reduction
noncomputable def time_chantal_descent : ℝ := distance_halfway / descent_speed_chantal
noncomputable def total_time_chantal_meet_jean : ℝ := time_chantal_first_half + time_chantal_second_half + time_chantal_descent

noncomputable def jean_average_speed : ℝ := distance_halfway / total_time_chantal_meet_jean

theorem jean_average_speed_correct (d : ℝ) :
  jean_average_speed = 20 / 17 := by
  -- proof would go here
  sorry

end jean_average_speed_correct_l85_85020


namespace problem_1_problem_2_problem_3_l85_85262

-- Problem 1
theorem problem_1 (C : ℝ) (y x : ℝ) :
  deriv y - y * cot x = sin x → y = (x + C) * sin x := sorry

-- Problem 2
theorem problem_2 (C : ℝ) (y x : ℝ) :
  x^2 * y^2 * deriv y + x * y^3 = 1 → y = root 3 (3/2 * x^2 + C) := sorry

-- Problem 3
theorem problem_3 (C : ℝ) (x y : ℝ) :
  y * (deriv x) - (3 * x + 1 + log y) * (deriv y) = 0 ∧ y (-1/3) = 1 → 
  x = y^3 / 9 - 4 / 9 - 1/3 * log y := sorry

end problem_1_problem_2_problem_3_l85_85262


namespace base4_to_base10_conversion_l85_85036

theorem base4_to_base10_conversion : (2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582) :=
by {
  -- The proof is omitted
  sorry
}

end base4_to_base10_conversion_l85_85036


namespace largest_incomposable_exists_l85_85548

noncomputable def largest_incomposable_amount (n : ℕ) : ℤ :=
  2 * 4^(n + 1) - 3^(n + 2)

theorem largest_incomposable_exists (n : ℕ) : ∃ s : ℕ, 
  (s = largest_incomposable_amount n) ∧
  ¬ (∃ k : ℕ, ∃ a b : list ℕ, 
    (∀ x ∈ a, x ∈ [0, 1, 2, ..., 3^n, 3^(n-1) * 4, 3^(n-2) * 4^2, ..., 4^n]) ∧
    (∀ x ∈ b, x ∈ [0, 1, 2, ..., 4^n]) ∧
    k = a.sum * 3 ^ n + b.sum * 4^n ∧ k = s) :=
sorry

end largest_incomposable_exists_l85_85548


namespace function_intersection_range_l85_85912

theorem function_intersection_range (a : ℝ) (h : a ≠ 0) : 
  (∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    (sqrt(a * x₁ + 4) = (x₁^2 - 4) / a) ∧ 
    (sqrt(a * x₂ + 4) = (x₂^2 - 4) / a) ∧ 
    (sqrt(a * x₃ + 4) = (x₃^2 - 4) / a)) → 
  (- 4 * sqrt(3) / 3 < a ∧ a ≤ -2) :=
sorry

end function_intersection_range_l85_85912


namespace like_term_l85_85612

theorem like_term (a : ℝ) : ∃ (a : ℝ), a * x ^ 5 * y ^ 3 = a * x ^ 5 * y ^ 3 :=
by sorry

end like_term_l85_85612


namespace divisibility_problem_l85_85578

theorem divisibility_problem (n : ℕ) : n-1 ∣ n^n - 7*n + 5*n^2024 + 3*n^2 - 2 := 
by
  sorry

end divisibility_problem_l85_85578


namespace expansion_constant_term_is_neg_672_l85_85897

noncomputable def binomial_expansion_constant_term (a : ℝ) : ℝ :=
  let c := a in
  if (1 - a)^9 = -1 then
    let r := 3 in
    (-1 : ℝ) * (Nat.choose 9 r) * (2 ^ r)
  else
    0

theorem expansion_constant_term_is_neg_672 :
  binomial_expansion_constant_term 2 = -672 :=
by
  sorry

end expansion_constant_term_is_neg_672_l85_85897


namespace find_a5_a6_l85_85552

-- Define the conditions for the geometric sequence
variables (a : ℕ → ℝ) (q : ℝ)
-- Let {a_n} be a geometric sequence with common ratio q
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

-- Assume the given conditions
axiom sum_a1_a2 : a 1 + a 2 = 324
axiom sum_a3_a4 : a 3 + a 4 = 36

-- Prove that a_5 + a_6 = 4
theorem find_a5_a6 (hgeom : geometric_sequence a q) (h1 : sum_a1_a2) (h3 : sum_a3_a4) : a 5 + a 6 = 4 :=
  sorry

end find_a5_a6_l85_85552


namespace exists_prime_pair_l85_85021

open Nat

-- Define our proof problem
theorem exists_prime_pair :
  ∃ (x y : ℕ), (10 < x ∧ x < 30) ∧ (10 < y ∧ y < 30) ∧ (x ≠ y) ∧ Prime x ∧ Prime y ∧ (x * y - (x + y) = 119) :=
by
  -- Define the range of prime numbers and verify the condition for being a prime
  let primes := [11, 13, 17, 19, 23, 29]
  
  -- Define the target value
  let target := 119
  
  -- Check all pairs
  have : ∃ (x y : ℕ), ((x ∈ primes) ∧ (y ∈ primes) ∧ (x ≠ y) ∧ (x * y - (x + y) = 119))
  {
    use (11, 13)
    simp [primes]
    -- Check all conditions
    exact ⟨by decide, by decide, by decide, by norm_num⟩
  }
    
  exact exists.elim this (λ x hx, ⟨x.1, x.2, hx⟩)

-- Add sorry to skip the proof
sorry

end exists_prime_pair_l85_85021


namespace quadratic_inequality_no_solution_l85_85867

theorem quadratic_inequality_no_solution (a b c : ℝ) (h : a ≠ 0)
  (hnsol : ∀ x : ℝ, ¬(a * x^2 + b * x + c ≥ 0)) :
  a < 0 ∧ b^2 - 4 * a * c < 0 :=
sorry

end quadratic_inequality_no_solution_l85_85867


namespace number_of_permutations_l85_85216

open Finset
open Function
open Equiv.Perm

noncomputable def inversion_number_of (σ : Perm (Fin 8)) (i : Fin 8) : Nat :=
  (Finset.filter (λ j, j < i ∧ σ j < σ i) (Finset.range 8)).card

def valid_permutation (σ: Perm (Fin 8)) : Prop :=
  inversion_number_of σ 7 = 2 ∧
  inversion_number_of σ 6 = 3 ∧
  inversion_number_of σ 4 = 3

theorem number_of_permutations : (Finset.filter valid_permutation (univ : Finset (Perm (Fin 8)))).card = 144 :=
  sorry

end number_of_permutations_l85_85216


namespace twenty_five_percent_of_five_hundred_l85_85435

theorem twenty_five_percent_of_five_hundred : 0.25 * 500 = 125 := 
by 
  sorry

end twenty_five_percent_of_five_hundred_l85_85435


namespace solve_quadratic_equation_l85_85622

noncomputable def solve_log_equation (x : ℝ) : Prop :=
  log (1 / 3 : ℝ) (x^2 + 3 * x - 4) = log (1 / 3 : ℝ) (2 * x + 2)

theorem solve_quadratic_equation :
  solve_log_equation 2 :=
by
  sorry

end solve_quadratic_equation_l85_85622


namespace graduation_ceremony_chairs_l85_85175

theorem graduation_ceremony_chairs (num_graduates num_teachers: ℕ) (half_as_administrators: ℕ) :
  (∀ num_graduates = 50) →
  (∀ num_teachers = 20) →
  (∀ half_as_administrators = num_teachers / 2) →
  (2 * num_graduates + num_graduates + num_teachers + half_as_administrators = 180) :=
begin
  intros,
  sorry
end

end graduation_ceremony_chairs_l85_85175


namespace chessboard_selection_l85_85298

-- Assume a chessboard model where each cell is numbered from 1 to 32,
-- and each number appears exactly twice.
def Chessboard := Array (Array ℕ) -- Simplified representation

-- Conditions: each number from 1 to 32 appears exactly twice on the chessboard
def valid_board (board : Chessboard) : Prop :=
  (∀ n, n ∈ {1, 2, ..., 32} → ∃! (r c : ℕ), 0 ≤ r < 8 ∧ 0 ≤ c < 8 ∧ board[r][c] = n)

-- Problem statement: Select 32 cells with each number being unique such that
-- each row and each column contains at least two such cells.
def has_valid_selection (board : Chessboard) : Prop :=
  ∃ selected : Array (Array Bool), 
    (∀ n, n ∈ {1, 2, ..., 32} → ∃! (r c : ℕ), 0 ≤ r < 8 ∧ 0 ≤ c < 8 ∧ selected[r][c] ∧ board[r][c] = n) ∧
    (∀ r : ℕ, 0 ≤ r < 8 → (∑ c in (0:8), if selected[r][c] then 1 else 0) ≥ 2) ∧
    (∀ c : ℕ, 0 ≤ c < 8 → (∑ r in (0:8), if selected[r][c] then 1 else 0) ≥ 2)

theorem chessboard_selection (board : Chessboard) (h : valid_board board) :
  has_valid_selection board :=
  sorry

end chessboard_selection_l85_85298


namespace avg_tickets_sold_by_males_100_l85_85899

theorem avg_tickets_sold_by_males_100 
  (female_avg : ℕ := 70) 
  (nonbinary_avg : ℕ := 50) 
  (overall_avg : ℕ := 66) 
  (male_ratio : ℕ := 2) 
  (female_ratio : ℕ := 3) 
  (nonbinary_ratio : ℕ := 5) : 
  ∃ (male_avg : ℕ), male_avg = 100 := 
by 
  sorry

end avg_tickets_sold_by_males_100_l85_85899


namespace carol_to_cathy_ratio_l85_85595

-- Define the number of cars owned by Cathy, Lindsey, Carol, and Susan
def cathy_cars : ℕ := 5
def lindsey_cars : ℕ := cathy_cars + 4
def carol_cars : ℕ := cathy_cars
def susan_cars : ℕ := carol_cars - 2

-- Define the total number of cars in the problem statement
def total_cars : ℕ := 32

-- Theorem to prove the ratio of Carol's cars to Cathy's cars is 1:1
theorem carol_to_cathy_ratio : carol_cars = cathy_cars := by
  sorry

end carol_to_cathy_ratio_l85_85595


namespace right_triangle_example_find_inverse_450_mod_3599_l85_85289

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def multiplicative_inverse (a b m : ℕ) : Prop :=
  (a * b) % m = 1

theorem right_triangle_example : is_right_triangle 60 221 229 :=
by
  sorry

theorem find_inverse_450_mod_3599 : ∃ n, 0 ≤ n ∧ n < 3599 ∧ multiplicative_inverse 450 n 3599 :=
by
  use 8
  sorry

end right_triangle_example_find_inverse_450_mod_3599_l85_85289


namespace range_of_a_l85_85839

variable (a : ℝ) (h₁ : a ≠ 0)
def setA : Set ℝ := {x | x^2 - x - 6 < 0}
def setB : Set ℝ := {x | x^2 + 2x - 8 ≥ 0}
def setC : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}
def notSetB : Set ℝ := {x | ¬(x ∈ setB)}
def intersectSet : Set ℝ := setA ∩ notSetB

theorem range_of_a (h₂ : setC ⊆ intersectSet) : a ∈ Ioo 0 (2/3) ∪ Ico (-2/3) 0 :=
sorry

#check range_of_a

end range_of_a_l85_85839


namespace liam_homework_probability_l85_85653

theorem liam_homework_probability:
  let p_complete := 5 / 9
  let p_not_complete := 1 - p_complete
  in p_not_complete = 4 / 9 :=
by
  -- steps to prove the theorem
  sorry

end liam_homework_probability_l85_85653


namespace ahn_largest_number_l85_85762

def largest_number_ahn_can_get : ℕ :=
  let n := 10
  2 * (200 - n)

theorem ahn_largest_number :
  (10 ≤ 99) →
  (10 ≤ 99) →
  largest_number_ahn_can_get = 380 := 
by
-- Conditions: n is a two-digit integer with range 10 ≤ n ≤ 99
-- Proof is skipped
  sorry

end ahn_largest_number_l85_85762


namespace correct_division_algorithm_l85_85701

theorem correct_division_algorithm : (-8 : ℤ) / (-4 : ℤ) = (8 : ℤ) / (4 : ℤ) := 
by 
  sorry

end correct_division_algorithm_l85_85701


namespace find_counterexample_l85_85719

open Nat

def is_non_prime_not_prime_sub_5_counterexample : Prop :=
  ∃ n : ℕ, 15 ≤ n ∧ n ≤ 30 ∧ ¬ isPrime n ∧ isPrime (n - 5)

theorem find_counterexample : is_non_prime_not_prime_sub_5_counterexample :=
  sorry

end find_counterexample_l85_85719


namespace calculate_value_l85_85592

-- Define points A and B
def A : ℝ × ℝ := (-6, 9)
def B : ℝ × ℝ := (8, -3)

-- Define midpoint calculation
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define point C as the midpoint of A and B
def C : ℝ × ℝ := midpoint A B

-- Define coordinates x and y for point C
def x : ℝ := C.1
def y : ℝ := C.2

-- Prove that 3x - 2y = -3 given our conditions
theorem calculate_value : 3 * x - 2 * y = -3 := by
  sorry

end calculate_value_l85_85592


namespace train_passing_time_l85_85338

theorem train_passing_time :
  ∀ (length speed_kmph : ℕ), length = 200 → speed_kmph = 72 →
  (length / (speed_kmph * 1000 / 3600) = 10) :=
by
  intros length speed_kmph len_eq spd_eq
  rw [len_eq, spd_eq]
  sorry

end train_passing_time_l85_85338


namespace max_value_leq_one_max_value_equal_one_l85_85208

noncomputable def max_value (a b c d : ℝ) : ℝ :=
  real.sqrt(abcd) ^ (1 / 4) + real.sqrt((1 - a) * (1 - b) * (1 - c) * (1 - d))

theorem max_value_leq_one (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (ha1 : a ≤ 1) (hb1 : b ≤ 1) (hc1 : c ≤ 1) (hd1 : d ≤ 1) :
    max_value a b c d ≤ 1 :=
sorry

theorem max_value_equal_one :
  max_value 0 0 0 0 = 1 :=
sorry

end max_value_leq_one_max_value_equal_one_l85_85208


namespace single_elimination_tournament_games_l85_85018

theorem single_elimination_tournament_games (players : ℕ) (h : players = 300) :
  ∃ games : ℕ, games = 299 :=
by
  use 299
  have h1 : players - 1 = 299, by linarith
  exact h1
  sorry

end single_elimination_tournament_games_l85_85018


namespace max_value_of_f_l85_85040

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^4 - 3*x^2 - 6*x + 13) - Real.sqrt (x^4 - x^2 + 1)

theorem max_value_of_f : ∃ x : ℝ, f x = Real.sqrt 10 :=
sorry

end max_value_of_f_l85_85040


namespace Mike_found_seashells_l85_85241

/-!
# Problem:
Mike found some seashells on the beach, he gave Tom 49 of his seashells.
He has thirteen seashells left. How many seashells did Mike find on the beach?

# Conditions:
1. Mike gave Tom 49 seashells.
2. Mike has 13 seashells left.

# Proof statement:
Prove that Mike found 62 seashells on the beach.
-/

/-- Define the variables and conditions -/
def seashells_given_to_Tom : ℕ := 49
def seashells_left_with_Mike : ℕ := 13

/-- Prove that Mike found 62 seashells on the beach -/
theorem Mike_found_seashells : 
  seashells_given_to_Tom + seashells_left_with_Mike = 62 := 
by
  -- This is where the proof would go
  sorry

end Mike_found_seashells_l85_85241


namespace exists_prime_q_not_div_n_p_minus_p_l85_85931

variable (p : ℕ) [Fact (Nat.Prime p)]

theorem exists_prime_q_not_div_n_p_minus_p :
  ∃ q : ℕ, Nat.Prime q ∧ q ≠ p ∧ ∀ n : ℕ, ¬ q ∣ (n ^ p - p) :=
sorry

end exists_prime_q_not_div_n_p_minus_p_l85_85931


namespace bug_total_distance_l85_85356

/-- 
A bug starts at position 3 on a number line. It crawls to -4, then to 7, and finally to 1.
The total distance the bug crawls is 24 units.
-/
theorem bug_total_distance : 
  let start := 3
  let first_stop := -4
  let second_stop := 7
  let final_position := 1
  let distance := abs (first_stop - start) + abs (second_stop - first_stop) + abs (final_position - second_stop)
  distance = 24 := 
by
  sorry

end bug_total_distance_l85_85356


namespace area_of_sector_l85_85886

-- Definitions for conditions
def radius (s : ℝ) (θ : ℝ) : ℝ := s / θ

def sector_area (r : ℝ) (θ : ℝ) : ℝ := 1/2 * r^2 * θ

-- The theorem to be proved
theorem area_of_sector (s θ : ℝ) (h1 : θ = 2) (h2 : s = 4) : sector_area (radius s θ) θ = 4 := 
by {
  rw [radius, sector_area, h1, h2],
  norm_num,
}

end area_of_sector_l85_85886


namespace problem_solution_l85_85792

open Real

noncomputable def numberOfSolutions : ℝ := 400

theorem problem_solution :
  ∃ (s : Set ℝ), (∀ x ∈ s, x ∈ (Set.Ioo 0 (200 * π)) ∧ abs (sin x) = (1/2) ^ x) ∧ s.finite ∧ s.card = numberOfSolutions :=
by
  sorry

end problem_solution_l85_85792


namespace part_I_part_II_l85_85129

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (λ : ℝ) : ℝ × ℝ := (2, λ)

def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2)
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := (v.1^2 + v.2^2).sqrt

theorem part_I (λ : ℝ) (h : parallel a (b λ)) : λ = -4 := sorry

theorem part_II (θ : ℝ) (λ : ℝ)
  (h1 : magnitude (b λ) = (5:ℝ).sqrt / 2)
  (h2 : perpendicular ((1, 2) + 2•(b λ)) (2•(1, 2) - (b λ))) :
  θ = Real.pi :=
sorry

end part_I_part_II_l85_85129


namespace triangle_area_l85_85557

noncomputable def sin30 : ℝ := 1 / 2

theorem triangle_area (A B C : Point) (hAB : dist A B = 4) (hAC : dist A C = 3) (hAngleA : angle A B C = 30) : 
  area A B C = 3 :=
by
  -- Use appropriate definitions and formulas (for area of triangle and sin function)
  sorry

end triangle_area_l85_85557


namespace problem_statement_l85_85583

noncomputable def a_sequence : ℕ → ℕ :=
sorry

noncomputable def b_sequence (m : ℕ) : ℕ :=
Inf { n | a_sequence n ≥ m }

theorem problem_statement : 
  (∀ n, a_sequence n ≤ a_sequence (n + 1)) -- non-decreasing sequence
  → a_sequence 19 = 85 -- Given condition
  → let b_sum := (Finset.range 85).sum b_sequence in
     let a_sum := (Finset.range 19).sum a_sequence in
     a_sum + b_sum ≤ 1700 -- maximum value
 :=
sorry

end problem_statement_l85_85583


namespace cos_theta_l85_85934

def vector1 : ℝ × ℝ × ℝ := (3, -4, 1)
def vector2 : ℝ × ℝ × ℝ := (9, -12, -4)

noncomputable def cosine_angle_between_planes : ℝ :=
  let dot_product := vector1.1 * vector2.1 + vector1.2 * vector2.2 + vector1.3 * vector2.3
  let norm_vector1 := real.sqrt (vector1.1 ^ 2 + vector1.2 ^ 2 + vector1.3 ^ 2)
  let norm_vector2 := real.sqrt (vector2.1 ^ 2 + vector2.2 ^ 2 + vector2.3 ^ 2)
  dot_product / (norm_vector1 * norm_vector2)

theorem cos_theta (θ : ℝ) (h : θ = real.atanh ((vector1.1 * vector2.1 + vector1.2 * vector2.2 + vector1.3 * vector2.3) / 
  (real.sqrt (vector1.1 ^ 2 + vector1.2 ^ 2 + vector1.3 ^ 2) * real.sqrt (vector2.1 ^ 2 + vector2.2 ^ 2 + vector2.3 ^ 2)))) :
  cosine_angle_between_planes = 71 / (real.sqrt 26 * real.sqrt 241) :=
sorry

end cos_theta_l85_85934


namespace solve_for_x_l85_85711

theorem solve_for_x (x : ℝ) (h : 5 + 7 / x = 6 - 5 / x) : x = 12 :=
by
  -- reduce the problem to its final steps
  sorry

end solve_for_x_l85_85711


namespace limit_for_regular_pay_l85_85740

theorem limit_for_regular_pay 
  (x : ℕ) 
  (regular_pay_rate : ℕ := 3) 
  (overtime_pay_rate : ℕ := 6) 
  (total_pay : ℕ := 186) 
  (overtime_hours : ℕ := 11) 
  (H : 3 * x + (6 * 11) = 186) 
  :
  x = 40 :=
sorry

end limit_for_regular_pay_l85_85740


namespace option_one_cost_effective_for_45_same_cost_for_40_l85_85001

noncomputable def ticket_price : ℕ → ℕ := 
λ n, 30 * n

noncomputable def option_one_cost (n : ℕ) : ℕ :=
21 * n

noncomputable def option_two_cost (n : ℕ) : ℕ :=
24 * (n - 5)

theorem option_one_cost_effective_for_45 : option_one_cost 45 < option_two_cost 45 :=
by
  sorry

theorem same_cost_for_40 : ∀ n, option_one_cost n = option_two_cost n ↔ n = 40 :=
by
  sorry

end option_one_cost_effective_for_45_same_cost_for_40_l85_85001


namespace number_of_subsets_of_A_l85_85668

theorem number_of_subsets_of_A (A : set ℕ) (h : A = {1, 2}) : A.powerset.card = 4 :=
by 
  -- Proof goes here
  sorry

end number_of_subsets_of_A_l85_85668


namespace F_atoms_in_compound_l85_85733

-- Given conditions
def atomic_weight_Al : Real := 26.98
def atomic_weight_F : Real := 19.00
def molecular_weight : Real := 84

-- Defining the assertion: number of F atoms in the compound
def number_of_F_atoms (n : Real) : Prop :=
  molecular_weight = atomic_weight_Al + n * atomic_weight_F

-- Proving the assertion that the number of F atoms is approximately 3
theorem F_atoms_in_compound : number_of_F_atoms 3 :=
  by
  sorry

end F_atoms_in_compound_l85_85733


namespace max_m_404_l85_85295

def sequence (n : ℕ) : ℕ :=
  nat.rec_on n 2022 (λ k x_k, 7 * x_k + 5)

def expression (n m : ℕ) : ℕ := 
  (sequence n).choose m

theorem max_m_404 {n m : ℕ} (h : m > 404) :
  ¬ (7 ∣ expression n m) :=
sorry

end max_m_404_l85_85295


namespace inequality_solution_set_l85_85896

theorem inequality_solution_set (a b : ℝ) (h1 : a > 0) (h2 : ∀ x : ℝ, ax^2 + bx - 1 < 0 ↔ -1/2 < x ∧ x < 1) :
  ∀ x : ℝ, (2 * x + 2) / (-x + 1) < 0 ↔ (x < -1 ∨ x > 1) :=
by sorry

end inequality_solution_set_l85_85896


namespace problem_l85_85449

def f (x : ℝ) : ℝ :=
  sorry

def a (n : ℕ) : ℝ :=
  (f n) ^ 2 - f n

def c (n : ℕ) : ℝ :=
  sorry

theorem problem 
  (h₀ : ∀ x : ℝ, f (x + 1) = real.sqrt (f x - (f x) ^ 2) + 1 / 2) 
  (h₁ : ∑ i in finset.range 15, a i = -31 / 16)
  (h₂ : ∀ n : ℕ, c n + c (n + 1) = (f 2019) ^ n) :
  (c 1 = 1 / 5) ∨ (c 1 = 3 / 7) := 
sorry

end problem_l85_85449


namespace sum_of_prime_factors_of_7pow7_sub_7pow3_l85_85806

theorem sum_of_prime_factors_of_7pow7_sub_7pow3 : 
  ∑ p in (Nat.factors (7^7 - 7^3)).toFinset, p = 17 := by
sorry

end sum_of_prime_factors_of_7pow7_sub_7pow3_l85_85806


namespace area_of_sine_curve_l85_85994

theorem area_of_sine_curve :
  let f := (fun x => Real.sin x)
  let a := -Real.pi
  let b := 2 * Real.pi
  (∫ x in a..b, f x) = 6 :=
by
  sorry

end area_of_sine_curve_l85_85994


namespace base6_to_base10_l85_85767

theorem base6_to_base10 : 
  let n := nat.mk_num_base 6 [2, 6, 5] in
  n = 113 :=
by
  let n := 2 * 6^2 + 6 * 6^1 + 5 * 6^0
  show n = 113
  from sorry

end base6_to_base10_l85_85767


namespace growth_rate_of_yield_l85_85359

-- Let x be the growth rate of the average yield per acre
variable (x : ℝ)

-- Initial conditions
def initial_acres := 10
def initial_yield := 20000
def final_yield := 60000

-- Relationship between the growth rates
def growth_relation := x * initial_acres * (1 + 2 * x) * (1 + x) = final_yield / initial_yield

theorem growth_rate_of_yield (h : growth_relation x) : x = 0.5 :=
  sorry

end growth_rate_of_yield_l85_85359


namespace b_is_geometric_sequence_sum_S_n_l85_85082

-- Define the sequence {a_n} and the relevant initial term and recurrence relation
def a (n : ℕ) : ℝ :=
  if h : n = 0 then 0          -- Handling for undefined 0th term for convention
  else Nat.recOn (n - 1) (2 / 3) (λ m a_m => 2 * a_m / (a_m + 1))

-- Define the sequence {b_n} such that b_n = 1/a_n - 1
def b (n : ℕ) : ℝ :=
  if h : n = 0 then 0          -- Handling for undefined 0th term for convention
  else 1 / a n - 1

-- Problem statement (1): Prove that b_n is a geometric sequence
theorem b_is_geometric_sequence : ∃ (r : ℝ), ∀ (n : ℕ), b (n + 1) = r * b n :=
sorry

-- Problem statement (2): Prove the sum S_n of the first n terms of the sequence {n/b_n}
theorem sum_S_n (n : ℕ) : ∑ i in Finset.range n, (i + 1) / b (i + 1) = 2 + (n^2 + n) / 2 - (2 + n) / 2^n :=
sorry

end b_is_geometric_sequence_sum_S_n_l85_85082


namespace reciprocal_power_l85_85894

theorem reciprocal_power (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 :=
by sorry

end reciprocal_power_l85_85894


namespace solution_set_of_inequality_l85_85192

variable {x : ℝ}

theorem solution_set_of_inequality : {x : ℝ | abs (abs (x - 2) - 1) ≤ 1} = set.Icc 0 4 :=
sorry

end solution_set_of_inequality_l85_85192


namespace isosceles_triangle_inequality_l85_85360

theorem isosceles_triangle_inequality
    (A B C M N : Point)
    (h_iso : is_isosceles_triangle A B C AC)
    (h_circle : is_circle_through_tangent_intersect A B C M N BC)
    (h_M_tangent : tangent_point_on_circle M BC)
    (h_N_intersect : intersect_point_on_circle N AB) :
    length (segment AN) > length (segment CM) := 
sorry

end isosceles_triangle_inequality_l85_85360


namespace white_cells_adjacent_even_l85_85970

def cell := prod int int
def is_black : cell → Prop
def is_white (c : cell) : Prop := ¬ is_black c
def adjacent (c1 c2 : cell) : Prop :=
  (abs (c1.fst - c2.fst) = 1 ∧ c1.snd = c2.snd) ∨ (abs (c1.snd - c2.snd) = 1 ∧ c1.fst = c2.fst)

def black_cells_even_white_neighbors (G : set cell) : Prop :=
  ∀ c ∈ G, is_black c → (filter (λ c', adjacent c c' ∧ is_white c') G).length % 2 = 0

theorem white_cells_adjacent_even (G : set cell) (hG : black_cells_even_white_neighbors G) :
  ∀ c ∈ G, is_white c → ∃ n, n = (filter (λ c', adjacent c c' ∧ is_white c') G).length ∧ n % 2 = 0 :=
sorry

end white_cells_adjacent_even_l85_85970


namespace arrangement_count_l85_85515

theorem arrangement_count :
  let word := ["C", "O₁", "L₁", "O₂", "R₂", "F₃", "U₄", "L₃"] in
  multiset.card word = 7 ∧
  multiset.distinct word →
  (multiset.perm_with_index_eq word).card = 5040 :=
by sorry

end arrangement_count_l85_85515


namespace ways_to_divide_8_friends_l85_85520

theorem ways_to_divide_8_friends : (4 : ℕ) ^ 8 = 65536 := by
  sorry

end ways_to_divide_8_friends_l85_85520


namespace distance_AB_CD_l85_85090

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := ⟨1, 0, 1⟩
def B : Point3D := ⟨-2, 2, 1⟩
def C : Point3D := ⟨2, 0, 3⟩
def D : Point3D := ⟨0, 4, -2⟩

def vector (p1 p2 : Point3D) : Point3D :=
  ⟨p2.x - p1.x, p2.y - p1.y, p2.z - p1.z⟩

def AB : Point3D := vector A B
def CD : Point3D := vector C D

def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def cross_product (v1 v2 : Point3D) : Point3D :=
  ⟨v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x⟩

def magnitude (v : Point3D) : ℝ :=
  real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

def distance_between_lines (p1 p2 p3 p4 : Point3D) : ℝ :=
  let n := cross_product (vector p1 p2) (vector p3 p4) in
  let distance := abs (dot_product n (vector p1 p3)) / magnitude n in
  distance

theorem distance_AB_CD : distance_between_lines A B C D = 26 / real.sqrt 389 :=
by sorry

end distance_AB_CD_l85_85090


namespace area_of_inscribed_quadrilateral_l85_85649

theorem area_of_inscribed_quadrilateral (R : ℝ) :
  let α := 9 * (Real.pi / 180) in
  let area := (1 / 2) * R^2 * (Real.sin α + Real.sin (3 * α) + Real.sin (9 * α) + Real.sin (27 * α)) in
  area = (R^2 * Real.sqrt 2) / 4 :=
sorry

end area_of_inscribed_quadrilateral_l85_85649


namespace sequence_solution_inequality_solution_l85_85753

-- Part (1): Prove that the value of λ is 1 and the formula for aₙ is 2ⁿ given the conditions.
theorem sequence_solution (a : ℕ → ℤ) (λ : ℤ)
  (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + λ * 2^n) 
  (h3 : a 1, a 2 + 1, a 3 form an arithmetic sequence) : 
  λ = 1 ∧ ∀ n : ℕ, n > 0 → a n = 2^n := 
sorry

-- Part (2): Prove that the value of integer p is 3 given the inequality holds for exactly four natural numbers n.
theorem inequality_solution (p : ℕ) 
  (h : ∃! n : ℕ, n > 0 ∧ (p / (2n - 5 : ℕ) ≤ (2p + 16) / (2^n))) : 
  p = 3 :=
sorry

end sequence_solution_inequality_solution_l85_85753


namespace parabola_eq_of_min_sum_of_distances_value_of_AB_plus_CD_l85_85122

-- Given and conditions
variables {p : ℝ} {P M F : ℝ × ℝ} (h_p_pos : p > 0)
variable (M : ℝ × ℝ := (2,3))
def parabola := {P | ∃ x : ℝ , P = (x, real.sqrt (2 * p * x))}
variable (directrix : set (ℝ × ℝ))
def distance (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Prove that parabola E has the equation y^2 = 4x given the condition of minimum sum of distances
theorem parabola_eq_of_min_sum_of_distances (h_min_dist : ∀ P ∈ parabola, P.0, distance P M + distance P (proj.directrix directrix P) = sqrt 10) :
  parabola = {P | ∃ x : ℝ, P = (x, real.sqrt (4 * x))} :=
sorry

-- Prove the value of |AB| + |CD|
theorem value_of_AB_plus_CD (b>0) (circle : set (ℝ × ℝ) := {P | P.0^2 + P.1^2 = 9})
  (h_intersect : ∀ b < 2, ∃ A B C D : ℝ × ℝ, A ∈ circle ∧ B ∈ parabola ∧ C ∈ circle ∧ D ∈ parabola
  ∧ fst A < fst B ∧ fst B < fst C ∧ fst C < fst D)
  (h_complementary : ∀ B ∈ parabola, D ∈ parabola, (slope B F) + (slope D F) = 0) :
  |distance A B + distance C D| = (36 * real.sqrt 5)/5 :=
sorry

end parabola_eq_of_min_sum_of_distances_value_of_AB_plus_CD_l85_85122


namespace intersection_A_B_l85_85832

def A := {x : ℤ | x < 6}
def B := {-3, 5, 6, 8}

theorem intersection_A_B : A ∩ B = {-3, 5} :=
by
  sorry

end intersection_A_B_l85_85832


namespace number_of_lucky_tickets_l85_85006

def is_leningrad_lucky (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  a₁ + a₂ + a₃ = a₄ + a₅ + a₆

def is_moscow_lucky (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  a₂ + a₄ + a₆ = a₁ + a₃ + a₅

def is_symmetric (a₂ a₅ : ℕ) : Prop :=
  a₂ = a₅

def is_valid_ticket (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  is_leningrad_lucky a₁ a₂ a₃ a₄ a₅ a₆ ∧
  is_moscow_lucky a₁ a₂ a₃ a₄ a₅ a₆ ∧
  is_symmetric a₂ a₅

theorem number_of_lucky_tickets : 
  ∃ n : ℕ, n = 6700 ∧ 
  (∀ a₁ a₂ a₃ a₄ a₅ a₆ : ℕ, 
    0 ≤ a₁ ∧ a₁ ≤ 9 ∧
    0 ≤ a₂ ∧ a₂ ≤ 9 ∧
    0 ≤ a₃ ∧ a₃ ≤ 9 ∧
    0 ≤ a₄ ∧ a₄ ≤ 9 ∧
    0 ≤ a₅ ∧ a₅ ≤ 9 ∧
    0 ≤ a₆ ∧ a₆ ≤ 9 →
    is_valid_ticket a₁ a₂ a₃ a₄ a₅ a₆ →
    n = 6700) := sorry

end number_of_lucky_tickets_l85_85006


namespace probability_event_A_l85_85948

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then real.log (x + 1) else 2^(-x) - 1

noncomputable def event_A (x : ℝ) : Prop :=
  f x ≤ 1

theorem probability_event_A :
  let I := set.Icc (-real.exp 1) (real.exp 1) in 
  let A := {x ∈ I | event_A x} in
  (measure_theory.volume A / measure_theory.volume I) = 1 / 2 :=
begin
  sorry
end

end probability_event_A_l85_85948


namespace distinct_real_roots_implies_positive_derivative_l85_85234

noncomputable def f (x a : ℝ) : ℝ := x^2 - (a-2)*x - a*log x
noncomputable def f' (x a : ℝ) : ℝ := (2*x - (a-2) - a/x)

theorem distinct_real_roots_implies_positive_derivative 
  (a c x1 x2 : ℝ) (h1 : f x1 a = c)
  (h2 : f x2 a = c) (h3 : x1 ≠ x2) (h4 : 0 < x1) (h5 : 0 < x2) (h6 : a > 0) :
  f' ((x1 + x2) / 2) a > 0 :=
sorry

end distinct_real_roots_implies_positive_derivative_l85_85234


namespace toys_in_box_time_l85_85601

-- Conditions
def total_toys : ℕ := 40
def toys_per_cycle (mom_toys brother_toys : ℕ) : ℕ := (mom_toys - 3) * 2 - brother_toys
def mom_time_per_cycle : ℕ := 20
def brother_time_per_cycle : ℕ := 40
def cycles_needed (total_toys net_gain toys_needed_per_cycle : ℕ) : ℕ := total_toys - 3

-- Ensure Lean keeps track of the non-computable factor where necessary
noncomputable def total_time_in_seconds (cycles_needed : ℕ) (mom_time : ℕ) : ℕ :=
  cycles_needed * mom_time + 20

theorem toys_in_box_time :
  let net_gain := toys_per_cycle 4 1 in
  let needed_cycles := total_toys - 3 in
  (total_time_in_seconds needed_cycles brother_time_per_cycle / 60) = 25 :=
by
  let net_gain := toys_per_cycle 4 1
  let needed_cycles := total_toys - 3
  have := total_time_in_seconds needed_cycles brother_time_per_cycle / 60
  sorry

end toys_in_box_time_l85_85601


namespace pool_depth_is_10_feet_l85_85992

-- Definitions based on conditions
def hoseRate := 60 -- cubic feet per minute
def poolWidth := 80 -- feet
def poolLength := 150 -- feet
def drainingTime := 2000 -- minutes

-- Proof goal: the depth of the pool is 10 feet
theorem pool_depth_is_10_feet :
  ∃ (depth : ℝ), depth = 10 ∧ (hoseRate * drainingTime) = (poolWidth * poolLength * depth) :=
by
  use 10
  sorry

end pool_depth_is_10_feet_l85_85992


namespace robot_goods_robot_purchase_and_cost_l85_85358

-- Part 1: Amount of goods carried per robot
theorem robot_goods (x y : ℤ) (h1 : x = y + 20) (h2 : 3 * x + 2 * y = 460) :
  x = 100 ∧ y = 80 := sorry

-- Part 2: Number of robots to purchase and minimum cost
theorem robot_purchase_and_cost (m : ℤ) (total_robots : ℤ := 20) (goods_per_day : ℤ := 1820)
  (a_cost : ℤ := 30000) (b_cost : ℤ := 20000)
  (h1 : ∀ m, 100 * m + 80 * (total_robots - m) ≥ goods_per_day) (h2 : ∀ m, m ≥ 11)
  (h3 : ∀ m, w := 10 * m + 400) :
  (m = 11 ∧ w = 510) := sorry

end robot_goods_robot_purchase_and_cost_l85_85358


namespace probability_sum_greater_than_9_l85_85687

def two_dice_roll_probability : ℚ := 
  let possible_outcomes := {(a, b) | a <- [1, 2, 3, 4, 5, 6], b <- [1, 2, 3, 4, 5, 6]}
  let favorable_outcomes := {(a, b) | (a, b) ∈ possible_outcomes ∧ a + b > 9}
  ↑(Set.card favorable_outcomes) / ↑(Set.card possible_outcomes)

theorem probability_sum_greater_than_9 : two_dice_roll_probability = 1 / 6 :=
  sorry

end probability_sum_greater_than_9_l85_85687


namespace find_x_l85_85818

def vector := (ℝ × ℝ)

-- Define the vectors a and b
def a (x : ℝ) : vector := (x, 3)
def b : vector := (3, 1)

-- Define the perpendicular condition
def perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Prove that under the given conditions, x = -1
theorem find_x (x : ℝ) (h : perpendicular (a x) b) : x = -1 :=
  sorry

end find_x_l85_85818


namespace boat_length_calculation_l85_85354

-- Define the given conditions as constants
def breadth := 2 -- meters
def sink_height := 0.01 -- meters (1 cm converted to meters)
def man_mass := 80 -- kg
def gravity := 9.81 -- m/s²
def water_density := 1000 -- kg/m³

-- Define the statement that the length of the boat is 4 meters given these conditions
theorem boat_length_calculation : ∃ (L : ℝ), L = 4 ∧
  (let V := L * breadth * sink_height in
    let W := man_mass * gravity in
    let W_water := water_density * V * gravity in
    W = W_water) :=
begin
  sorry
end

end boat_length_calculation_l85_85354


namespace length_of_AB_l85_85910

theorem length_of_AB
  (triangle_ABC_is_isosceles : ∀ A B C : Type, angle A B C = angle B A C → (distance A C = distance B C))
  (triangle_CBD_is_isosceles : ∀ C B D : Type, angle C B D = angle C D B → (distance C D = distance B C))
  (perimeter_CBD_is_18 : ∀ C B D : Type, distance B D + distance B C + distance C D = 18)
  (perimeter_ABC_is_24 : ∀ A B C : Type, distance A B + distance B C + distance A C = 24)
  (length_BD_is_8 : ∀ B D : Type, distance B D = 8)
  (equal_angles_ABC_CBD : ∀ A B C D : Type, angle A B C = angle C B D) :
  ∃ AB : ℝ, AB = 14 := by
  sorry

end length_of_AB_l85_85910


namespace ellipse_equation_min_length_AB_l85_85107

-- Define the conditions for the ellipse problem
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1

def point_on_ellipse (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  P.2 ^ 2 = b ^ 2

def eccentricity (c a : ℝ) : Prop :=
  c / a = 2 / 3

def foci_relation (a b c : ℝ) : Prop :=
  a ^ 2 = b ^ 2 + c ^ 2

noncomputable def A := (4: ℝ, 0: ℝ)

-- First part: prove the equation of the ellipse
theorem ellipse_equation : 
  ∀ (x y a b : ℝ),
    let P := (0:ℝ, sqrt 5:ℝ) in
    let c := 2:ℝ in
    a > b ∧ b > 0 ∧
    point_on_ellipse P a b ∧
    eccentricity c a ∧
    foci_relation a b c →
    ellipse x y 3 (sqrt 5) :=
by
  -- Proof is omitted
  sorry

-- Second part: prove the minimum length of segment AB
theorem min_length_AB :
  ∀ (B : ℝ × ℝ),
    let C := (0:ℝ, sqrt 5:ℝ) in
    let e := 2/3 in
    ellipse B.1 B.2 3 (sqrt 5) ∧
    (4 * B.1 + B.2 * 0 = 0) →
    ∃ (AB_min : ℝ), AB_min = sqrt 21 :=
by
  -- Proof is omitted
  sorry

end ellipse_equation_min_length_AB_l85_85107


namespace math_bonanza_2016_3_2_l85_85205

/-- Define the sequence a_n given a_0 = 1 and the recurrence relation -/
def a : ℕ → ℝ
| 0     := 1
| (n+1) := (sqrt 3 * a n - 1) / (a n + sqrt 3)

/-- The goal to prove that a_{2017} is equal to 2 - sqrt 3 and thus, that the sum of components a + b + c is 4 -/
theorem math_bonanza_2016_3_2 :
  a 2017 = 2 - sqrt 3 ∧ (2 - 1 + 3 = 4) :=
by sorry

end math_bonanza_2016_3_2_l85_85205


namespace number_of_real_solutions_l85_85782

-- Define the conditions of the problem
def system_of_equations (x y z w : ℝ) : Prop :=
  x = z + w - z*w*x + 1 ∧ 
  y = w + x - w*x*y + 1 ∧
  z = x + y - x*y*z + 1 ∧
  w = y + z - y*z*w + 1

-- Statement to prove the number of solutions
theorem number_of_real_solutions : 
  ∃ (S : Finset (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (sol : ℝ × ℝ × ℝ × ℝ), sol ∈ S ↔ system_of_equations sol.1 sol.2 sol.3 sol.4) ∧ 
    S.card = 5 :=
by sorry

end number_of_real_solutions_l85_85782


namespace parabola_equation_and_perpendicularity_l85_85467

-- Define the parabola conditions and the intersection condition
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def passes_through (x y : ℝ) : Prop := x = 1 ∧ y = 2
def is_on_focus (x : ℝ) : Prop := x ≠ 0

-- Define the perpendicular condition
def perpendicular (A B O : ℝ × ℝ) : Prop :=
let (x1, y1) := A,
    (x2, y2) := B,
    (x0, y0) := O in
(y1 - y0) / (x1 - x0) * (y2 - y0) / (x2 - x0) = -1

-- Prove that the standard equation is y^2 = 4x and OA ⊥ OB
theorem parabola_equation_and_perpendicularity :
  ∃ (p : ℝ), 
    (parabola 2 1 2) ∧
    (∀ x y, parabola p x y → y^2 = 4 * x) ∧
    ∃ (A B : ℝ × ℝ),
    (parabola 2 (fst A) (snd A)) ∧ (parabola 2 (fst B) (snd B)) ∧
    (perpendicular A B (0, 0)) := 
begin
  sorry
end

end parabola_equation_and_perpendicularity_l85_85467


namespace distance_light_travels_in_half_a_year_l85_85297

def speed_of_light : ℕ := 299792  -- kilometers per second
def days_in_half_a_year : ℝ := 182.5  -- days

noncomputable def seconds_in_one_day : ℕ := 24 * 3600
noncomputable def seconds_in_half_a_year : ℕ := (days_in_half_a_year * seconds_in_one_day).to_nat

noncomputable def distance_traveled_in_half_a_year : ℝ := (speed_of_light * seconds_in_half_a_year).to_real

theorem distance_light_travels_in_half_a_year :
  distance_traveled_in_half_a_year ≈ 4.73 * 10^12 :=
sorry

end distance_light_travels_in_half_a_year_l85_85297


namespace union_M_N_equals_set_x_ge_1_l85_85124

-- Definitions of M and N based on the conditions from step a)
def M : Set ℝ := { x | x - 2 > 0 }

def N : Set ℝ := { y | ∃ x : ℝ, y = Real.sqrt (x^2 + 1) }

-- Statement of the theorem
theorem union_M_N_equals_set_x_ge_1 : (M ∪ N) = { x : ℝ | x ≥ 1 } := 
sorry

end union_M_N_equals_set_x_ge_1_l85_85124


namespace problem1_problem2_problem3_l85_85469

-- Definitions of transformations and final sequence S
def transformation (A : List ℕ) : List ℕ := 
  match A with
  | x :: y :: xs => (x + y) :: transformation (y :: xs)
  | _ => []

def nth_transform (A : List ℕ) (n : ℕ) : List ℕ :=
  Nat.iterate (λ L => transformation L) n A

def final_sequence (A : List ℕ) : ℕ :=
  match nth_transform A (A.length - 1) with
  | [x] => x
  | _ => 0

-- Proof Statements

theorem problem1 : final_sequence [1, 2, 3] = 8 := sorry

theorem problem2 (n : ℕ) : final_sequence (List.range (n+1)) = (n + 2) * 2 ^ (n - 1) := sorry

theorem problem3 (A B : List ℕ) (h : A = List.range (B.length)) (h_perm : B.permutations.contains A) : 
  final_sequence B = final_sequence A := by
  sorry

end problem1_problem2_problem3_l85_85469


namespace mary_initial_pokemon_cards_l85_85239

theorem mary_initial_pokemon_cards (x : ℕ) (torn_cards : ℕ) (new_cards : ℕ) (current_cards : ℕ) 
  (h1 : torn_cards = 6) 
  (h2 : new_cards = 23) 
  (h3 : current_cards = 56) 
  (h4 : current_cards = x - torn_cards + new_cards) : 
  x = 39 := 
by
  sorry

end mary_initial_pokemon_cards_l85_85239


namespace simplify_expression_l85_85621

theorem simplify_expression (m : ℤ) : 
  (3^(m + 5) - 3 * 3^m) / (3 * 3^(m + 4)) = 80 / 81 := 
by
  sorry

end simplify_expression_l85_85621


namespace surface_area_pyramid_l85_85575

-- Definitions of the pyramid DABC and the given conditions
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)
structure Triangle := (A B C : Point)
structure Pyramid := (D A B C : Point)

-- Conditions for the pyramid
def hasSpecifiedEdgeLengths (DABC : Pyramid) : Prop :=
  ∀ (p1 p2 : Point), (p1 = DABC.D ∨ p1 = DABC.A ∨ p1 = DABC.B ∨ p1 = DABC.C) ∧
  (p2 = DABC.D ∨ p2 = DABC.A ∨ p2 = DABC.B ∨ p2 = DABC.C) ∧
  p1 ≠ p2 → let length := dist p1 p2 in length = 17 ∨ length = 39

def noEquilateralFace (DABC : Pyramid) : Prop :=
  ∀ (t : Triangle),
  (t = ⟨DABC.A, DABC.B, DABC.C⟩ ∨
   t = ⟨DABC.D, DABC.A, DABC.B⟩ ∨
   t = ⟨DABC.D, DABC.B, DABC.C⟩ ∨
   t = ⟨DABC.D, DABC.C, DABC.A⟩) →
  let lengths := [dist t.A t.B, dist t.B t.C, dist t.C t.A] in
  ¬ (lengths.nodup_list ∧ lengths.all (λ x, x = 17) ∨
     lengths.all (λ x, x = 39))

-- The problem statement
theorem surface_area_pyramid (DABC : Pyramid)
  (h1 : hasSpecifiedEdgeLengths DABC)
  (h2 : noEquilateralFace DABC) :
  let base_area := 323.51 in
  4 * base_area = 1294.04 := by
  sorry

end surface_area_pyramid_l85_85575


namespace minimum_pieces_needed_l85_85301

/--
There are 2020 x 2020 squares on a board. A square (a, b) is on the 
diagonals containing the square (c, d) when |a - c| = |b - d|.
Given that at most one piece can be placed in each square,
prove that the minimum number of pieces to place such that
each square has at least two other pieces on its diagonals is 2020.
-/
theorem minimum_pieces_needed (n : ℕ) (hn : n = 2020) :
  ∃ p : ℕ, 
    (∀ i j : ℕ, 
      i < n ∧ j < n → 
      ∃ a1 a2 : ℕ × ℕ, 
        a1 ≠ (i,j) ∧ a2 ≠ (i,j) ∧ 
        |i - a1.1| = |j - a1.2| ∧ 
        |i - a2.1| = |j - a2.2| ∧
        piece_exists_at a1 ∧ piece_exists_at a2)
    ∧ p = 2020 := sorry

end minimum_pieces_needed_l85_85301


namespace area_of_isosceles_right_triangle_l85_85158

open_locale classical

noncomputable def area_of_triangle (a b c : ℝ) : ℝ := (1 / 2) * a * b

theorem area_of_isosceles_right_triangle (h : ∀ X Y Z : ℝ, X = Y ∧ Y = 2 * Z) (EF : ℝ) (hEF : EF = 2) : area_of_triangle (2 * 2) (2 * 2) (2 * 2) = 4 :=
by
  -- altitude is given as 2 cm
  have h1 : EF = 2, from hEF,
  -- all sides of isosceles right triangle are equal: X = Y = 2 * sqrt(2)
  exact calc
    area_of_triangle (2 * 2) (2 * 2) (2 * 2)
        = (1 / 2) * (2 * 2) * (2 * 2) : rfl
    ... = (1 / 2) * 8 : by ring
    ... = 4 : by ring


end area_of_isosceles_right_triangle_l85_85158


namespace faster_train_speed_l85_85714

theorem faster_train_speed (length_train : ℝ) (time_cross : ℝ) (speed_ratio : ℝ) (total_distance : ℝ) (relative_speed : ℝ) :
  length_train = 100 → 
  time_cross = 8 → 
  speed_ratio = 2 → 
  total_distance = 2 * length_train → 
  relative_speed = (1 + speed_ratio) * (total_distance / time_cross) → 
  (1 + speed_ratio) * (total_distance / time_cross) / 3 * 2 = 8.33 := 
by
  intros
  sorry

end faster_train_speed_l85_85714


namespace exists_small_triangle_l85_85181

theorem exists_small_triangle (points : set (point R)) (h_points : points ⊆ (set.univ : set (point R))) :
  (∃ t : triangle R, (t ∈ triangles_of_points points) ∧ (area_of_triangle t < 1/100)) :=
sorry

-- Definitions to support the theorem
definition point (R : Type) := R × R  -- A point is represented as a pair (a, b) in R^2
definition triangle (R : Type) := (point R) × (point R) × (point R)  -- A triangle is represented by 3 points

-- The finite set of 102 points in the unit square
constant points : set (point ℝ) := sorry

-- Triangles formed by these points
definition triangles_of_points {R : Type} (S : set (point R)) := 
  {t | ∃ a b c. a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ t = (a, b, c)}

-- The area of a triangle given by vertices (a, b, c)
noncomputable definition area_of_triangle {R : Type} [field R] (t : triangle R) :=
  let ((x1, y1), (x2, y2), (x3, y3)) := t in
  0.5 * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

end exists_small_triangle_l85_85181


namespace standard_equation_of_circle_tangent_to_x_axis_l85_85253

theorem standard_equation_of_circle_tangent_to_x_axis :
  ∀ (x y : ℝ), ((x + 3) ^ 2 + (y - 4) ^ 2 = 16) :=
by
  -- Definitions based on the conditions
  let center_x := -3
  let center_y := 4
  let radius := 4

  sorry

end standard_equation_of_circle_tangent_to_x_axis_l85_85253


namespace pythagorean_triple_third_number_l85_85084

theorem pythagorean_triple_third_number (x : ℕ) (h1 : x^2 + 8^2 = 17^2) : x = 15 :=
sorry

end pythagorean_triple_third_number_l85_85084


namespace minimum_value_f_l85_85069

def f (x : ℝ) : ℝ :=
  (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_f (x : ℝ) (hx : x ≥ 5 / 2) : ∃ m, m = 1 ∧ ∀ y, y ≥ 5 / 2 → f y ≥ m :=
by 
  sorry

end minimum_value_f_l85_85069


namespace angle_of_inclination_l85_85039

theorem angle_of_inclination (θ : ℝ) : 
  (∀ x y : ℝ, x - y + 3 = 0 → ∃ θ : ℝ, Real.tan θ = 1 ∧ θ = Real.pi / 4) := by
  sorry

end angle_of_inclination_l85_85039


namespace average_score_group2_l85_85732

-- Total number of students
def total_students : ℕ := 50

-- Overall average score
def overall_average_score : ℝ := 92

-- Number of students from 1 to 30
def group1_students : ℕ := 30

-- Average score of students from 1 to 30
def group1_average_score : ℝ := 90

-- Total number of students - group1_students = 50 - 30 = 20
def group2_students : ℕ := total_students - group1_students

-- Lean 4 statement to prove the average score of students with student numbers 31 to 50 is 95
theorem average_score_group2 :
  (overall_average_score * total_students = group1_average_score * group1_students + x * group2_students) →
  x = 95 :=
sorry

end average_score_group2_l85_85732


namespace range_of_a_l85_85159

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 3| + |x - a| ≥ 3) ↔ a ≤ 0 ∨ a ≥ 6 :=
by
  sorry

end range_of_a_l85_85159


namespace number_of_ways_to_divide_friends_l85_85517

theorem number_of_ways_to_divide_friends :
  (4 ^ 8 = 65536) := by
  -- result obtained through the multiplication principle
  sorry

end number_of_ways_to_divide_friends_l85_85517


namespace students_part_both_l85_85666

variable (num_students total_part_volleyball total_part_track_and_field part_neither part_both : ℕ)

theorem students_part_both :
  num_students = 45 →
  total_part_volleyball = 12 →
  total_part_track_and_field = 20 →
  part_neither = 19 →
  part_both = 6 :=
by
  intros h1 h2 h3 h4
  -- Here we convert the conditions to the equations used in solving the problem and show the expected result matches the solution
  have h_total_part := num_students - part_neither,
  have h_total_part_eq := by rw [h1, h4]; exact h_total_part,
  have h_both := total_part_volleyball + total_part_track_and_field - h_total_part_eq,
  sorry

end students_part_both_l85_85666


namespace cherries_regular_price_l85_85389

theorem cherries_regular_price (y : ℝ) (h : 0.175 * y = 2) : y = 11.43 :=
by
  have h₁ : y = 2 / 0.175 := by sorry
  norm_num at h₁
  exact h₁

end cherries_regular_price_l85_85389


namespace probability_includes_chinese_l85_85011

open ProbabilityMassFunction

variable (Individual : Type) [Fintype Individual]

axiom two_americans_one_frenchman_one_chinese (A1 A2 F C : Individual) :
  ∃ (individuals : Finset Individual),
    individuals = {A1, A2, F, C} ∧ 
    ∀ (x : Individual), x ∈ individuals → x = A1 ∨ x = A2 ∨ x = F ∨ x = C

theorem probability_includes_chinese (A1 A2 F C : Individual) :
  ∃ (individuals : Finset Individual)
    (pairs : Finset (Finset Individual))
    (pairs_with_chinese : Finset (Finset Individual)),
    individuals = {A1, A2, F, C} ∧
    pairs = individuals.powerset.filter (λ s, s.card = 2) ∧
    pairs_with_chinese = pairs.filter (λ s, C ∈ s) ∧
    (pairs_with_chinese.card : ℚ) / pairs.card = 1 / 2 :=
by
  sorry

end probability_includes_chinese_l85_85011


namespace actual_time_when_clock_reads_5_l85_85975

theorem actual_time_when_clock_reads_5:
  ∀ (t₀ t₁ actual_t₀ actual_t₁: ℕ),
  t₀ = 8 * 60 →
  t₁ = 10 * 60 →
  actual_t₀ = 8 * 60 →
  actual_t₁ = 10 * 60 →
  (t₁ - t₀) - (actual_t₁ - actual_t₀) = 8 →
  ∃ t: ℕ, t = 17 * 60 + 38 + 34 /60 ∧
  (actual_t₀ + 578.57) = t :=
by {
  sorry
}

end actual_time_when_clock_reads_5_l85_85975


namespace smallest_root_l85_85444

-- Define the equation
def equation (x chi : ℝ) : Prop := (sin (π * x) + tan chi = x + x^3)

-- Define the condition tg(chi) = 0
def condition (chi : ℝ) : Prop := (tan chi = 0)

-- Prove that the smallest root is 0 given the above conditions
theorem smallest_root (chi : ℝ) (h: condition chi) : ∃ x : ℝ, equation x chi ∧ ∀ y : ℝ, equation y chi → x ≤ y :=
begin
  use 0,
  split,
  {
    -- Show that 0 is a root
    unfold equation,
    rw [h, mul_zero, sin_zero, add_zero, zero_add, pow_succ, mul_zero],
  },
  {
    -- For all y, prove that 0 is the smallest root
    intros y hy,
    sorry,  -- Skipping detailed proof steps
  }
end

end smallest_root_l85_85444


namespace tasty_filling_probability_sum_l85_85783

def isNotInOrder (l : List ℕ) : Prop :=
  ¬ (l = l.sorted || l = l.sorted.reverse)

def isTastyFilling (filling : List (List ℕ)) : Prop :=
  (isNotInOrder filling.head!) &&
  (isNotInOrder filling.tail!.head !) &&
  (isNotInOrder (List.transpose filling).head!) &&
  (isNotInOrder (List.transpose filling).tail!.head!) &&
  (isNotInOrder (List.transpose filling).tail!.tail!.head!)

def numValidFillings : ℕ := 720

def probabilityTastyFilling : ℕ × ℕ :=
  (numValidFillings, 720)

theorem tasty_filling_probability_sum :
  let ⟨m, n⟩ := probabilityTastyFilling in
  Nat.coprime m n → m + n = 2 := by
  sorry

end tasty_filling_probability_sum_l85_85783


namespace minimum_value_sum_maximum_value_sum_l85_85229

-- Given conditions
variable (n : ℕ) (x : Fin n → ℝ)
axiom nonneg_x : ∀ i : Fin n, 0 ≤ x i
axiom constraint :
  ∑ i in Finset.range n, (x i) ^ 2 + 
  2 * ∑ i in Finset.range n, ∑ j in Finset.range i, Real.sqrt (i / (j + 1)) * (x i) * (x j + 1) = 1

-- Minimum value of the sum
theorem minimum_value_sum : 
  ∑ i in Finset.range n, x i ≥ 1 := sorry

-- Maximum value of the sum
theorem maximum_value_sum :
  ∑ i in Finset.range n, x i ≤ Real.sqrt ( ∑ k in Finset.range n, (Real.sqrt k - Real.sqrt (k - 1)) ^ 2 ) := sorry

end minimum_value_sum_maximum_value_sum_l85_85229


namespace problem_equivalent_lean_l85_85481

-- Defining the types for lines and planes
variable (Line Plane : Type) 

-- Declaring the necessary relations
variable (Perpendicular Parallel : Line → Plane → Prop) 
variable (LineParallel : Line → Line → Prop) 

-- Given conditions
variables (m n : Line) 
variables (α β : Plane) 
variable (different_lines : m ≠ n)
variable (different_planes : α ≠ β)
variable (m_perpendicular_α : Perpendicular m α)
variable (n_parallel_β : Parallel n β)

-- Statement equivalent to the problem conditions
theorem problem_equivalent_lean :
(∃ (m n : Line) (α β : Plane), 
  m ≠ n ∧ α ≠ β ∧ 
  Perpendicular m α ∧ 
  Parallel n β ∧ 
  (LineParallel m n → Perpendicular α β) ∧ 
  (Perpendicular m β → LineParallel m n)) :=
begin
  use [m, n, α, β],
  split, assumption, -- m ≠ n
  split, assumption, -- α ≠ β
  split, assumption, -- m ⊥ α
  split, assumption, -- n ∥ β
  split,
  { -- Proof of: If m ∥ n, then α ⊥ β.
    intro h,
    sorry },
  { -- Proof of: If m ⊥ β, then m ⊥ n.
    intro h,
    sorry }
end

end problem_equivalent_lean_l85_85481


namespace password_factors_l85_85550

theorem password_factors (m n : ℕ) :
  let p := λ x : ℕ, x^3 + (m - n) * x^2 + n * x in
  p 10 = 101213 ↔ (m = 11 ∧ n = 6) := by
sorry

end password_factors_l85_85550


namespace mark_bread_time_l85_85964

def rise_time1 : Nat := 120
def rise_time2 : Nat := 120
def kneading_time : Nat := 10
def baking_time : Nat := 30

def total_time : Nat := rise_time1 + rise_time2 + kneading_time + baking_time

theorem mark_bread_time : total_time = 280 := by
  sorry

end mark_bread_time_l85_85964


namespace all_black_possible_l85_85568

-- Define the problem context
variable (n : ℕ)

-- The initial configuration where all numbers are white
def initial_configuration : Fin n → Bool := fun _ => false

-- Define the flipping operation
def flip_color (colors : Fin n → Bool) (i : Fin n) : Fin n → Bool :=
  fun j => if (Nat.gcd (i + 1) (j + 1) = 1 ) then !colors j else colors j

-- Lean theorem stating the problem
theorem all_black_possible (n : ℕ) :
  ∃ flips : List (Fin n), (flips.foldl flip_color initial_configuration) = (fun _ => true) :=
sorry

end all_black_possible_l85_85568


namespace count_common_elements_l85_85667

def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else a(n - 1) + a(n - 2)

def b (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 1
  else b(n - 1) + b(n - 2)

theorem count_common_elements : 
  {x : ℕ | ∃ n m : ℕ, a n = x ∧ b m = x}.card = 2 :=
sorry

end count_common_elements_l85_85667


namespace base4_to_base10_conversion_l85_85033

theorem base4_to_base10_conversion : 
  2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582 :=
by 
  sorry

end base4_to_base10_conversion_l85_85033


namespace probability_of_condition_l85_85265

def chosen_set : Set ℕ := {n | 1 ≤ n ∧ n ≤ 20}

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_divisible_by_five (a b : ℕ) : Prop := (a + b) % 5 = 0

theorem probability_of_condition :
  let total_choices := (Finset.card (Finset.powersetLen 2 (Finset.range 21))) -- total combinations
  let odd_numbers := {n | n ∈ chosen_set ∧ is_odd n}
  let odd_pairs := (Finset.powersetLen 2 (Finset.filter odd_numbers.toSet (Finset.range 21))) 
  let valid_pairs := Finset.filter (λ p, sum_divisible_by_five p.1 p.2) odd_pairs
  (Finset.card valid_pairs : ℚ) / (Finset.card total_choices) = 9 / 95 :=
by
  sorry

end probability_of_condition_l85_85265


namespace expected_revenue_day_14_plan_1_more_reasonable_plan_l85_85757

-- Define the initial conditions
def initial_valuation : ℕ := 60000
def rain_probability : ℚ := 0.4
def no_rain_probability : ℚ := 0.6
def hiring_cost : ℕ := 32000

-- Calculate the expected revenue if Plan ① is adopted
def expected_revenue_plan_1_day_14 : ℚ :=
  (initial_valuation / 10000) * (1/2 * rain_probability + no_rain_probability)

-- Calculate the total revenue for Plan ①
def total_revenue_plan_1 : ℚ :=
  (initial_valuation / 10000) + 2 * expected_revenue_plan_1_day_14

-- Calculate the total revenue for Plan ②
def total_revenue_plan_2 : ℚ :=
  3 * (initial_valuation / 10000) - (hiring_cost / 10000)

-- Define the lemmas to prove
theorem expected_revenue_day_14_plan_1 :
  expected_revenue_plan_1_day_14 = 4.8 := 
  by sorry

theorem more_reasonable_plan :
  total_revenue_plan_1 > total_revenue_plan_2 :=
  by sorry

end expected_revenue_day_14_plan_1_more_reasonable_plan_l85_85757


namespace Sīyǔ_correct_score_l85_85642

def Statement1 : Prop := ∀ x : ℝ, x < 0 → ¬∃ y : ℝ, y * y = x
def Statement2 : Prop := ∀ x : ℝ, ∃ y : ℝ, y * y * y = x → x = 0 ∨ x = 1 ∨ x = -1
def Statement3 : Prop := ∃ x : ℝ, x * x * x = 27 ∧ x ≠ 3
def Statement4 : Prop := ∃ x : ℝ, x = real.sqrt 7 ∧ 2 < x ∧ x < 3

def Correctness1 : Statement1 := by sorry
def Correctness2 : Statement2 := by sorry
def Correctness3 : ¬Statement3 := by sorry
def Correctness4 : Statement4 := by sorry

def score_1 (ans : Prop) (correct : Prop) : ℕ := if ans = correct then 25 else 0
def score_2 (ans : Prop) (correct : Prop) : ℕ := if ans = correct then 25 else 0
def score_3 (ans : Prop) (correct : Prop) : ℕ := if ans = correct then 25 else 0
def score_4 (ans : Prop) (correct : Prop) : ℕ := if ans = correct then 25 else 0

def Sīyǔ_score : ℕ :=
  score_1 true Correctness1 +
  score_2 false Correctness2 +
  score_3 true Correctness3 +
  score_4 false Correctness4

theorem Sīyǔ_correct_score : Sīyǔ_score = 50 := by sorry

end Sīyǔ_correct_score_l85_85642


namespace total_shirts_l85_85136

def hazel_shirts : ℕ := 6
def razel_shirts : ℕ := 2 * hazel_shirts

theorem total_shirts : hazel_shirts + razel_shirts = 18 := by
  sorry

end total_shirts_l85_85136


namespace range_of_eccentricity_l85_85502

noncomputable def hyperbola : Type := sorry

variables {a b c e : ℝ} {P : hyperbola}

axiom hyperbola_equation (x y : ℝ) :
  a > 0 ∧ b > 0 ∧ c = sqrt (a^2 + b^2) ∧
  (∃ P : (ℝ × ℝ), P ∈ hyperbola ∧
    (a / (Real.sin (angle P (-c, 0) (c, 0))) = c / (Real.sin (angle P (c, 0) (-c, 0))))) ∧
   (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)

theorem range_of_eccentricity :
  1 < e ∧ e < sqrt 2 + 1 :=
sorry

end range_of_eccentricity_l85_85502


namespace exists_n_for_dvd_ka_pow_n_add_n_l85_85204

theorem exists_n_for_dvd_ka_pow_n_add_n 
  (a k : ℕ) (a_pos : 0 < a) (k_pos : 0 < k) (d : ℕ) (d_pos : 0 < d) :
  ∃ n : ℕ, 0 < n ∧ d ∣ k * (a ^ n) + n :=
by
  sorry

end exists_n_for_dvd_ka_pow_n_add_n_l85_85204


namespace total_cards_l85_85921

theorem total_cards (Brenda_card Janet_card Mara_card Michelle_card : ℕ)
  (h1 : Janet_card = Brenda_card + 9)
  (h2 : Mara_card = 7 * Janet_card / 4)
  (h3 : Michelle_card = 4 * Mara_card / 5)
  (h4 : Mara_card = 210 - 60) :
  Janet_card + Brenda_card + Mara_card + Michelle_card = 432 :=
by
  sorry

end total_cards_l85_85921


namespace balls_count_neither_red_nor_blue_l85_85304

theorem balls_count_neither_red_nor_blue : 
  (total_balls red_balls blue_balls remaining_balls balls_neither_red_nor_blue : ℕ)
  (h1 : total_balls = 360) 
  (h2 : red_balls = total_balls / 4)
  (h3 : remaining_balls = total_balls - red_balls)
  (h4 : blue_balls = remaining_balls / 5)
  (h5 : balls_neither_red_nor_blue = total_balls - (red_balls + blue_balls)) :
  balls_neither_red_nor_blue = 216 := by
  sorry

end balls_count_neither_red_nor_blue_l85_85304


namespace inequality1_inequality2_l85_85616

-- First condition and statement
theorem inequality1 (a b c : ℝ) : a^2 + b^2 + c^2 ≥ ab + ac + bc :=
sorry

-- Second condition and statement
theorem inequality2 : sqrt 6 + sqrt 7 > 2 * sqrt 2 + sqrt 5 :=
sorry

end inequality1_inequality2_l85_85616


namespace find_f_1_l85_85152

def f (x : ℝ) : ℝ := sorry

axiom condition : ∀ x : ℝ, f(x) + 3 * f(-x) = Real.log (x + 3) / Real.log 2

theorem find_f_1 : f 1 = 1 / 8 :=
by 
  -- proof steps would go here
  sorry

end find_f_1_l85_85152


namespace part1_part2_l85_85133

-- Define the vectors m and n and the function f
def vector_m (x : ℝ) : ℝ × ℝ × ℝ := (sqrt 3, sin (x / 4), 1)
def vector_n (x : ℝ) : ℝ × ℝ := (cos (x / 4), cos (x / 4) ^ 2)
def f (x : ℝ) : ℝ := (sqrt 3) * (sin (x / 4)) * (cos (x / 4)) + (cos (x / 4) ^ 2)

-- Part 1: Prove cos (2 * pi / 3 - x) = -1 / 2 if f(x) = 1
theorem part1 (x : ℝ) (h : f x = 1) : cos ((2 * Real.pi) / 3 - x) = -1 / 2 :=
by
  sorry

-- Part 2: Prove the range of f(B) given a cos C + (1 / 2) * c = b
def triangle_condition (a b c : ℝ) : Prop := a * (cos C) + (1 / 2) * c = b

theorem part2 (a b c : ℝ) (h : triangle_condition a b c) (B : ℝ) :
  1 < f B ∧ f B < 3 / 2 :=
by
  sorry

end part1_part2_l85_85133


namespace cube_4_edge_trips_l85_85287

theorem cube_4_edge_trips (A B : Type) (cube : set (Type)) (e : cube → cube → Prop) :
  (forall (u : cube), ∃ (v w : cube), e u v ∧ e u w ∧ v ≠ w) →
  (forall (u : cube), ∃ (v : cube), e u v) →
  (e AB) →
  (shortest_trip_length A B 4)
  ∃ (paths : list (list cube)), (all_elements_of_length_4 paths) ∧ (number_of_paths paths) = 12 := sorry

end cube_4_edge_trips_l85_85287


namespace sum_ia_leq_half_n_minus_one_l85_85233

theorem sum_ia_leq_half_n_minus_one (a : ℕ → ℝ) (n : ℕ) 
  (h1 : ∑ i in finset.range n, a i = 0) 
  (h2 : ∑ i in finset.range n, |a i| = 1) : 
  |∑ i in finset.range n, i * a i| ≤ (n - 1) / 2 := 
sorry

end sum_ia_leq_half_n_minus_one_l85_85233


namespace triangle_angle_values_l85_85558

theorem triangle_angle_values (A B C P Q : Type)
  [triangle A B C]
  (ha1 : ∃ P, angle_bisector A P B C)
  (ha2 : ∃ Q, angle_bisector B Q C A)
  (ha3 : angle A B C = 60)
  (h_eq : length A B + length B P = length A Q + length Q B) :
  (angle B A C = 80 ∧ angle C A B = 40) :=
by
  sorry

end triangle_angle_values_l85_85558


namespace min_value_product_l85_85080

noncomputable def minimum_product (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  ∏ i, (Real.sin (x i) + 1 / Real.sin (x i))

theorem min_value_product (n : ℕ) (x : Fin n → ℝ) (h1 : 2 ≤ n) 
  (h2 : ∀ i, 0 < x i) (h3 : ∑ i, x i = Real.pi) :
  minimum_product n x = (Real.sin (Real.pi / n) + 1 / Real.sin (Real.pi / n)) ^ n :=
  sorry

end min_value_product_l85_85080


namespace find_angle_l85_85068

variables (a b : ℝ^3) (θ : ℝ)

-- Defining the magnitude conditions
def magnitude_a : Prop := ∥a∥ = 3
def magnitude_b : Prop := ∥b∥ = 4

-- Condition involving the dot product
def dot_product_condition : Prop := (2 • a - b) ⬝ (a + 2 • b) ≥ 4

-- Angle condition to prove
def theta_condition : Prop := θ ∈ set.Icc 0 (π / 3)

-- The theorem stating the result
theorem find_angle (ha : magnitude_a a) (hb : magnitude_b b) (hdot : dot_product_condition a b) : theta_condition θ := 
sorry

end find_angle_l85_85068


namespace sum_cos_2x_2y_2z_l85_85943

theorem sum_cos_2x_2y_2z (x y z : ℝ)
  (h1 : cos x + cos y + cos z = 0)
  (h2 : sin x + sin y + sin z = 0) :
  cos (2 * x) + cos (2 * y) + cos (2 * z) = 0 := 
by
  sorry

end sum_cos_2x_2y_2z_l85_85943


namespace base4_to_base10_conversion_l85_85031

theorem base4_to_base10_conversion : 
  2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582 :=
by 
  sorry

end base4_to_base10_conversion_l85_85031


namespace total_money_spent_l85_85336

variable (X : ℝ)

axiom fresh_fruits_vegetables : X * (1/2)
axiom meat_products : X * (1/3)
axiom bakery_products : X * (1/10)
axiom candy_products : X - (X * (1/2) + X * (1/3) + X * (1/10)) = 8

theorem total_money_spent (X : ℝ) : X = 120 :=
by
  sorry

end total_money_spent_l85_85336


namespace triangle_area_l85_85915

theorem triangle_area (a b c : ℝ) (A B C : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) 
  (h_c : c = 2) (h_C : C = π / 3)
  (h_sin : Real.sin B = 2 * Real.sin A) :
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3 / 3 :=
sorry

end triangle_area_l85_85915


namespace Rank_Regions_Consumption_l85_85423

def RankRegionsBasedOnConsumption (west nonWest russia : ℕ) : Prop :=
  ∃ rankWest rankNonWest rankRussia : ℕ, 
  rankWest = 1 ∧ 
  rankNonWest = 2 ∧ 
  rankRussia = 3 ∧ 
  west <= nonWest ∧ 
  nonWest <= russia

theorem Rank_Regions_Consumption:
  RankRegionsBasedOnConsumption 21428 26848.55 302790.13 :=
begin
  sorry
end

end Rank_Regions_Consumption_l85_85423


namespace sum_geometric_sequence_no_arithmetic_sequence_consecutive_terms_existence_term_ge_2016_l85_85464

section GeometricSequence

variables (a q : ℝ) (n : ℕ)
hypothesis (h_q_nonzero : q ≠ 0)
hypothesis (h_q_nonone : q ≠ 1)

-- Part 1: Prove the formula for the sum of a geometric series
theorem sum_geometric_sequence :
  S n = a * (1 - q^n) / (1 - q) :=
sorry

-- Part 2: Prove the nonexistence of consecutive terms forming an arithmetic sequence
theorem no_arithmetic_sequence_consecutive_terms :
  ¬ ∃ k, 2 * (a * q^k) = (a * q^k) + (a * q^(k + 2)) :=
sorry

end GeometricSequence

section SpecificGeometricSequence

-- Part 3: Specific case, check the existence of T_n >= 2016 for given a=q=2
variables (n : ℕ)
noncomputable def T (n : ℕ) : ℝ := ∑ i in range (n+1), i * 2^(i)

theorem existence_term_ge_2016 :
  (∃ n : ℕ, T n ≥ 2016) ↔ (n ≥ 8) :=
sorry

end SpecificGeometricSequence

end sum_geometric_sequence_no_arithmetic_sequence_consecutive_terms_existence_term_ge_2016_l85_85464


namespace tank_capacity_l85_85391

theorem tank_capacity (x : ℝ) (h : (5/12) * x = 150) : x = 360 :=
by
  sorry

end tank_capacity_l85_85391


namespace car_enters_and_leaves_storm_l85_85727

theorem car_enters_and_leaves_storm (
  t_1 t_2 : ℝ
  (h1 : (∀ t : ℝ, t ≥ 0 → ∃ x y : ℝ, x = t ∧ y = 0)) -- Car travels due east at 1 mile per minute
  (h2 : ∀ t : ℝ, t ≥ 0 → ∃ x y : ℝ, x = t/2 ∧ y = 150 - t/2) -- Storm moves southeast at sqrt(2)/2 miles per minute
  (h3 : ∀ t : ℝ, (0, t) = (0, 0) := 150) -- Initial position of storm 150 miles due north of the car
  (h4 : ∃ t : ℝ, t = t_1 ∨ t = t_2 ∧ (sqrt ((t - t/2)^2 + (150 - t/2)^2) = 60))) -- Car enters and leaves storm circle
: 1/2 * (t_1 + t_2) = 150 :=
sorry

end car_enters_and_leaves_storm_l85_85727


namespace ball_redistribution_impossible_l85_85344

noncomputable def white_boxes_initial_ball_count := 31
noncomputable def black_boxes_initial_ball_count := 26
noncomputable def white_boxes_new_ball_count := 21
noncomputable def black_boxes_new_ball_count := 16
noncomputable def white_boxes_target_ball_count := 15
noncomputable def black_boxes_target_ball_count := 10

theorem ball_redistribution_impossible
  (initial_white_boxes : ℕ)
  (initial_black_boxes : ℕ)
  (new_white_boxes : ℕ)
  (new_black_boxes : ℕ)
  (total_white_boxes : ℕ)
  (total_black_boxes : ℕ) :
  initial_white_boxes * white_boxes_initial_ball_count +
    initial_black_boxes * black_boxes_initial_ball_count =
  total_white_boxes * white_boxes_target_ball_count +
    total_black_boxes * black_boxes_target_ball_count →
  (new_white_boxes, new_black_boxes) = (total_white_boxes - initial_white_boxes, total_black_boxes - initial_black_boxes) →
  ¬(∃ total_white_boxes total_black_boxes, 
    total_white_boxes * white_boxes_target_ball_count +
    total_black_boxes * black_boxes_target_ball_count =
    initial_white_boxes * white_boxes_initial_ball_count +
    initial_black_boxes * black_boxes_initial_ball_count) :=
by sorry

end ball_redistribution_impossible_l85_85344


namespace koala_fiber_eaten_l85_85926

def koala_fiber_absorbed (fiber_eaten : ℝ) : ℝ := 0.30 * fiber_eaten

theorem koala_fiber_eaten (absorbed : ℝ) (fiber_eaten : ℝ) 
  (h_absorbed : absorbed = koala_fiber_absorbed fiber_eaten) : fiber_eaten = 40 :=
by {
  have h1 : fiber_eaten * 0.30 = absorbed,
  rw h_absorbed,
  have : 12 = absorbed,
  rw this,
  sorry,
}

end koala_fiber_eaten_l85_85926


namespace problem_l85_85100

noncomputable def center_on_line (a : ℝ) : Prop := ∃ c : ℝ × ℝ, c = (a, -2 * a)

def passes_through (c : ℝ × ℝ) (p : ℝ × ℝ) (r : ℝ) : Prop :=
  (c.1 - p.1)^2 + (c.2 - p.2)^2 = r^2

def tangent_to_line (c : ℝ × ℝ) (line : ℝ → ℝ → ℝ) (r : ℝ) : Prop :=
  (line c.1 c.2) ^ 2 = r^2 * (1 + 1)

def equation_of_circle (c : ℝ × ℝ) (r : ℝ) : ℝ → ℝ → ℝ := λ x y, (x - c.1)^2 + (y - c.2)^2 - r^2

def slope_of_tangent (c : ℝ × ℝ) (p : ℝ × ℝ) (r : ℝ) : ℝ → Prop :=
  let k := abs((p.2 - c.2) / (p.1 - c.1)) in
  k^2 - 7 = 0

theorem problem :
  (center_on_line 1) →
  (passes_through (1, -2) (2, -1) (sqrt 2)) →
  (tangent_to_line (1, -2) (λ x y, x + y - 1) (sqrt 2)) →
  (∃ eq : (ℝ → ℝ → ℝ), eq = equation_of_circle (1, -2) (sqrt 2)) ∧
  (slope_of_tangent (1, -2) (1, 2) (sqrt 2)) :=
by
  sorry

end problem_l85_85100


namespace lambda_mu_sum_l85_85870

theorem lambda_mu_sum 
  (p : ℝ) (hp: p > 0)
  (m : ℝ) (hm: m ≠ 0)
  (x0 x1 x2 y0 y1 y2: ℝ)
  (lambda mu: ℝ)
  (hPM: (x1 - x0, y1 - y0) = lambda * (m - x1, -y1))
  (hPN: (x2 - x0, y2 - y0) = mu * (m - x2, -y2))
  (hParabolaM: y1^2 = 2 * p * x1)
  (hParabolaN: y2^2 = 2 * p * x2)
  : lambda + mu = -1 :=
sorry

end lambda_mu_sum_l85_85870


namespace negation_of_p_l85_85503

variable (a : ℝ)

def proposition_p := ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0

theorem negation_of_p : ¬ proposition_p ↔ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0 := by
  sorry

end negation_of_p_l85_85503


namespace find_unique_function_l85_85799

theorem find_unique_function (f : ℝ → ℝ) (hf1 : ∀ x, 0 ≤ x → 0 ≤ f x)
    (hf2 : ∀ x, 0 ≤ x → f (f x) + f x = 12 * x) :
    ∀ x, 0 ≤ x → f x = 3 * x := 
  sorry

end find_unique_function_l85_85799


namespace line_equation_l85_85488

theorem line_equation (x y : ℝ) (h_perpendicular : ∀ x y : ℝ, x - 2 * y - 1 = 0) (h_passes_through : (1,1)) :
  2 * x + y - 3 = 0 :=
sorry

end line_equation_l85_85488


namespace line_perpendicular_transitivity_l85_85835

-- Given definitions

variables (α β : Plane) (l : Line) 
variables (l_perp_alpha : l ⟂ α) (alpha_parallel_beta : α ∥ β)

-- The proof statement

theorem line_perpendicular_transitivity :
  l ⟂ β :=
sorry

end line_perpendicular_transitivity_l85_85835


namespace number_of_children_l85_85795

def pencils_per_child : ℕ := 2
def total_pencils : ℕ := 16

theorem number_of_children : total_pencils / pencils_per_child = 8 :=
by
  sorry

end number_of_children_l85_85795


namespace quadrilateral_area_inequality_l85_85381

theorem quadrilateral_area_inequality 
  (T : ℝ) (a b c d e f : ℝ) (φ : ℝ) 
  (hT : T = (1/2) * e * f * Real.sin φ) 
  (hptolemy : e * f ≤ a * c + b * d) : 
  2 * T ≤ a * c + b * d := 
sorry

end quadrilateral_area_inequality_l85_85381


namespace count_valid_numbers_l85_85416

theorem count_valid_numbers : 
  ∃ n : ℕ, n = 70 ∧ 
  (∀ a b c : ℕ, 
    (∃ k : ℕ, 100001 * a + 10010 * b + 1100 * c = 7 * k) ∧ 
    b % 2 = 1 ∧ 
    ∀ m: ℕ, m ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → 
    ((∃ p q: ℕ, p = a ∧ q = c ∧ q = p ∨ q = p + 7 ∨ q = p - 7))
  ) := sorry

end count_valid_numbers_l85_85416


namespace vector_dot_product_l85_85822

variables (a b : ℝ^n)

-- Definitions
def norm (v : ℝ^n) : ℝ := real.sqrt (v ⬝ v)
def perpendicular (v w : ℝ^n) : Prop := v ⬝ w = 0

-- Given conditions as hypotheses
variables (ha : norm a = 4) (hb : norm b = 5) (hab : perpendicular a b)

-- Statement to prove
theorem vector_dot_product (a b : ℝ^n) (ha : norm a = 4) (hb : norm b = 5) (hab : perpendicular a b) : a ⬝ b = 0 :=
by
  sorry

end vector_dot_product_l85_85822


namespace perpendicular_vectors_solution_l85_85131

variables (a b : ℝ × ℝ) (λ : ℝ)

def perp_vectors (v w : ℝ × ℝ) : Prop := 
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_vectors_solution (h_a : a = (3, 3)) (h_b : b = (1, -1)) :
  perp_vectors (a.1 + λ * b.1, a.2 + λ * b.2) (a.1 - λ * b.1, a.2 - λ * b.2) ↔ λ = 3 ∨ λ = -3 :=
sorry

end perpendicular_vectors_solution_l85_85131


namespace min_value_of_f_minus_2_l85_85859

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 else x + 6/x - 6

theorem min_value_of_f_minus_2 : ∃ x : ℝ, f(x) - 2 = (-(1/2)) := sorry

end min_value_of_f_minus_2_l85_85859


namespace multiply_by_reciprocal_l85_85911

theorem multiply_by_reciprocal (x : ℝ) : 0.3 * x = 45 → x = 150 :=
by
  intros h
  have h₀ : 0.3 = 3 / 10 := by norm_num
  rw [h₀] at h
  have h₁ : (3 / 10) * x * (10 / 3) = 45 * (10 / 3) := by rw [h]
  norm_num at h₁
  exact h₁

example : x = 150 → 0.3 * x = 45 := assume (h : x = 150) => by rw [h]; norm_num

end multiply_by_reciprocal_l85_85911


namespace select_four_person_committee_l85_85385

open Nat

theorem select_four_person_committee 
  (n : ℕ)
  (h1 : (n * (n - 1) * (n - 2)) / 6 = 21) 
  : (n = 9) → Nat.choose n 4 = 126 :=
by
  sorry

end select_four_person_committee_l85_85385


namespace probability_odd_product_sum_divisible_by_5_l85_85268

theorem probability_odd_product_sum_divisible_by_5 :
  (∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 20 ∧ 1 ≤ b ∧ b ≤ 20 ∧ a ≠ b ∧ (a * b % 2 = 1 ∧ (a + b) % 5 = 0)) →
  ∃ (p : ℚ), p = 3 / 95 :=
by
  sorry

end probability_odd_product_sum_divisible_by_5_l85_85268


namespace ellipse_centroid_locus_l85_85834

noncomputable def ellipse_equation (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1
noncomputable def centroid_locus (x y : ℝ) : Prop := (9 * x^2) / 4 + 3 * y^2 = 1 ∧ y ≠ 0

theorem ellipse_centroid_locus (x y : ℝ) (h : ellipse_equation x y) : centroid_locus (x / 3) (y / 3) :=
  sorry

end ellipse_centroid_locus_l85_85834


namespace dog_revs_l85_85365

theorem dog_revs (r₁ r₂ : ℝ) (n₁ : ℕ) (n₂ : ℕ) (h₁ : r₁ = 48) (h₂ : n₁ = 40) (h₃ : r₂ = 12) :
  n₂ = 160 := 
sorry

end dog_revs_l85_85365


namespace number_of_students_in_third_batch_l85_85303

theorem number_of_students_in_third_batch
  (avg1 avg2 avg3 : ℕ)
  (total_avg : ℚ)
  (students1 students2 : ℕ)
  (h_avg1 : avg1 = 45)
  (h_avg2 : avg2 = 55)
  (h_avg3 : avg3 = 65)
  (h_total_avg : total_avg = 56.333333333333336)
  (h_students1 : students1 = 40)
  (h_students2 : students2 = 50) :
  ∃ x : ℕ, (students1 * avg1 + students2 * avg2 + x * avg3 = total_avg * (students1 + students2 + x) ∧ x = 60) :=
by
  sorry

end number_of_students_in_third_batch_l85_85303


namespace diagonals_intersect_at_midpoint_of_opposite_vertices_l85_85977

theorem diagonals_intersect_at_midpoint_of_opposite_vertices (x1 y1 x2 y2 : ℝ):
  (x1, y1) = (2, -3) → (x2, y2) = (14, 9) → 
  (∃ m n : ℝ, (m, n) = (8, 3)) :=
by
  intros h1 h2
  use (8, 3)
  sorry

end diagonals_intersect_at_midpoint_of_opposite_vertices_l85_85977


namespace inequality_log_l85_85454

theorem inequality_log (a b c : ℝ) (h1: 0 < a) (h2: a < b) (h3: b < 1) (h4: c > 1) : 
  a * Real.log c (1 / b) < b * Real.log c (1 / a) :=
sorry

end inequality_log_l85_85454


namespace max_x1_sq_plus_x2_sq_l85_85228

theorem max_x1_sq_plus_x2_sq (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = k - 2) 
  (h2 : x1 * x2 = k^2 + 3 * k + 5)
  (h3 : -4 ≤ k ∧ k ≤ -4 / 3) : 
  x1^2 + x2^2 ≤ 18 :=
by sorry

end max_x1_sq_plus_x2_sq_l85_85228


namespace minimum_friend_circles_l85_85674

-- Define the context and parameters

noncomputable def handshake_graph : Type := sorry -- Placeholder for the graph definition
noncomputable def num_vertices : Nat := 24
noncomputable def num_edges : Nat := 216
noncomputable def max_non_triangle_edges (u v : vertex) : Nat := 10

-- Define the minimum friend circles number

theorem minimum_friend_circles (G : handshake_graph) 
  (H1 : G.vertex_set.card = num_vertices)
  (H2 : G.edge_set.card = num_edges)
  (H3 : ∀ (u v : vertex), u ≠ v → G.adj u v → 
        (card {w : vertex | G.adj u w ∧ ¬ G.adj v w ∨ G.adj v w ∧ ¬ G.adj u w}) ≤ max_non_triangle_edges u v):
  ∃ (min_friend_circles : Nat), min_friend_circles = 864 :=
sorry

end minimum_friend_circles_l85_85674


namespace merchant_articles_l85_85742

theorem merchant_articles (N CP SP : ℝ) 
  (h1 : N * CP = 16 * SP)
  (h2 : SP = CP * 1.375) : 
  N = 22 :=
by
  sorry

end merchant_articles_l85_85742


namespace ratio_Steve_to_Mike_l85_85967

variable (P : ℝ) -- The price Steve paid for the DVD online excluding shipping
variable (Price_Mike : ℝ := 5) -- The price Mike paid at the store
variable (Shipping_Cost_Rate : ℝ := 0.8) -- The shipping cost rate

-- Total amount Steve paid
def Total_Payment_Steve := P + Shipping_Cost_Rate * P

-- The given condition that Steve's total payment is $18
axiom Steve_Total_Payment : Total_Payment_Steve P = 18

theorem ratio_Steve_to_Mike : (P / Price_Mike) = 2 :=
by
  have hP : P = 10 := by
    have h1 : 1.8 * P = 18 := by
      simp [Total_Payment_Steve, Steve_Total_Payment]
    have h2 : P = 18 / 1.8 := by
      rw ←mul_div_assoc at h1
      apply eq_of_mul_eq_mul_right
      norm_num
      exact h1
    rw h2
    norm_num
  simp [hP]
  norm_num

end ratio_Steve_to_Mike_l85_85967


namespace triangle_area_l85_85426

/-- 
  Given:
  - A smaller rectangle OABD with OA = 4 cm, AB = 4 cm
  - A larger rectangle ABEC with AB = 12 cm, BC = 12 cm
  - Point O at (0,0)
  - Point A at (4,0)
  - Point B at (16,0)
  - Point C at (16,12)
  - Point D at (4,12)
  - Point E is on the line from A to C
  
  Prove the area of the triangle CDE is 54 cm²
-/
theorem triangle_area (OA AB OB DE DC : ℕ) : 
  OA = 4 ∧ AB = 4 ∧ OB = 16 ∧ DE = 12 - 3 ∧ DC = 12 → (1 / 2) * DE * DC = 54 := by 
  intros h
  sorry

end triangle_area_l85_85426


namespace average_value_of_powers_l85_85436

theorem average_value_of_powers (z : ℝ) : 
  (z^2 + 3*z^2 + 6*z^2 + 12*z^2 + 24*z^2) / 5 = 46*z^2 / 5 :=
by
  sorry

end average_value_of_powers_l85_85436


namespace base_salary_l85_85752

theorem base_salary (income1 income2 income3 income4 income5 comm next_comms avg_income total_weeks : ℕ)
  (h_income1 : income1 = 406)
  (h_income2 : income2 = 413)
  (h_income3 : income3 = 420)
  (h_income4 : income4 = 436)
  (h_income5 : income5 = 395)
  (h_comm : comm = 345)
  (h_next_comms : next_comms = 2)
  (h_avg_income : avg_income = 500)
  (h_total_weeks : total_weeks = 7) :
(base_salary : ℕ) = 370 :=
by
  sorry

end base_salary_l85_85752


namespace expected_value_equals_1_75_l85_85242

noncomputable def expected_value_of_winnings : ℚ :=
  let winnings := [2, 3, 5, 7, 0, 0, -4, 1] in
  (List.sum (List.map (λ x, (1 / 8 : ℚ) * x) winnings))

theorem expected_value_equals_1_75 :
  expected_value_of_winnings = 1.75 :=
by
  sorry

end expected_value_equals_1_75_l85_85242


namespace find_X_for_isosceles_triangle_l85_85828

-- Definitions for the geometric setup
variable {P : Type*} [EuclideanGeometry P]

-- Given conditions: acute angle MON, points A and B within the angle
variables (O M N A B : P)
variable (X : P) -- Point on OM
variable (Y Z : P) -- Intersections of XA and XB with ON

-- Given: acute angle MON and specific positioning constraints
variable (acute_MON : angle MON < 90)
variable (inside_A : inside_angle O M N A)
variable (inside_B : inside_angle O M N B)

-- Projection onto ON and conditions for point X
variable (proj_X_ON_between : proj_on_line O N X between (proj_on_line O N A) (proj_on_line O N B))
variable (angle_AXB : angle A' X B = 180 - 2 * angle MON)

-- Prove that triangle XYZ is isosceles
theorem find_X_for_isosceles_triangle :
  is_isosceles_triangle X Y Z :=
sorry

end find_X_for_isosceles_triangle_l85_85828


namespace ABCD_parallelogram_l85_85978

open_locale classical

variables {A B C D K N O A₁ D₁ : Type*}
variables [has_coe ℝ (set A)] [has_coe ℝ (set B)] [has_coe ℝ (set C)] [has_coe ℝ (set D)]
variables [has_coe ℝ (set K)] [has_coe ℝ (set N)] [has_coe ℝ (set O)] [has_coe ℝ (set A₁)] [has_coe ℝ (set D₁)]
variables [is_midpoint A B K] [is_midpoint C D N] [line_through A O A₁] [line_through D O D₁]
variables [line_divide_in_three A₁ B C D₁]

-- define midpoint predicate
def is_midpoint (A B K : Type*) [has_coe ℝ (set A)] [has_coe ℝ (set B)] [has_coe ℝ (set K)] : Prop :=
  ∃ (M : set (ℝ × ℝ)), M = (K : Type*) ∧ (K : Type*) = (A + B)/2

-- define lines intersecting/dividing predicate
def line_through (P Q R : Type*) [has_coe ℝ (set P)] [has_coe ℝ (set Q)] [has_coe ℝ (set R)] : Prop :=
  ∃ (L : set (ℝ × ℝ)), L = { x | ∃ y : ℝ, y*x = (P + Q + R)}

-- define line dividing segment into three equal parts predicate
def line_divide_in_three (A₁ B C D₁ : Type*) [has_coe ℝ (set A₁)] [has_coe ℝ (set B)] [has_coe ℝ (set C)] [has_coe ℝ (set D₁)] : Prop :=
  ∃ (L₁ L₂ L₃: set (ℝ × ℝ)), L₁ = { x | ∃ y : ℝ, y*x = 1/3*(A₁+B+C+D₁)} ∧ L₂ = { x | ∃ y : ℝ, y*x = 2/3*(A₁+B+C+D₁)}

theorem ABCD_parallelogram
  (K_midpoint : is_midpoint A B K)
  (N_midpoint : is_midpoint C D N)
  (A1_intersect : line_through A O A₁)
  (D1_intersect : line_through D O D₁)
  (BC_three_parts : line_divide_in_three A₁ B C D₁) :
  is_parallelogram A B C D :=
sorry

end ABCD_parallelogram_l85_85978


namespace probability_prime_and_multiple_of_11_l85_85609

theorem probability_prime_and_multiple_of_11 (h1 : 11.prime) (h2 : 11 ∣ 11) :
  (1 : ℚ) / 100 = 1 / 100 :=
by
  -- Conditions that are given in the problem
  have h_total_cards : 100 > 0 := by norm_num
  -- Card 11 is the only prime and multiple of 11 in the range 1-100
  have h_unique_card : ∃ (n : ℕ), n = 11 := ⟨11, rfl⟩
  -- Probability calculation
  sorry -- proof is not required

end probability_prime_and_multiple_of_11_l85_85609


namespace equivalent_statement_l85_85706

variable (R G : Prop)

theorem equivalent_statement (h : ¬ R → ¬ G) : G → R := by
  intro hG
  by_contra hR
  exact h hR hG

end equivalent_statement_l85_85706


namespace fencing_cost_approx_l85_85712

noncomputable def cost_of_fencing (area_hectares : ℝ) (cost_per_meter : ℝ) : ℝ :=
let area_m2 := area_hectares * 10000 in
let radius := (area_m2 / Real.pi).sqrt in
let circumference := 2 * Real.pi * radius in
circumference * cost_per_meter

theorem fencing_cost_approx (area : ℝ) (cost_per_meter : ℝ) (h_area : area = 17.56) (h_cost : cost_per_meter = 4) :
  abs (cost_of_fencing area cost_per_meter - 5938) < 1 :=
sorry

end fencing_cost_approx_l85_85712


namespace correct_propositions_count_l85_85508

variables {m n : Line} {a b : Plane}

def proposition1 (m n : Line) (a b : Plane) : Prop :=
  m ⊥ a ∧ n ⊥ b ∧ m ⊥ n → a ⊥ b

def proposition2 (m n : Line) (a b : Plane) : Prop :=
  m ∥ a ∧ n ∥ b ∧ m ∥ n → a ∥ b

def proposition3 (m n : Line) (a b : Plane) : Prop :=
  m ⊥ a ∧ n ∥ b ∧ m ⊥ n → a ⊥ b

def proposition4 (m n : Line) (a b : Plane) : Prop :=
  m ⊥ a ∧ n ∥ b ∧ m ∥ n → a ∥ b

theorem correct_propositions_count :
  (proposition1 m n a b ∨ proposition2 m n a b ∨ proposition3 m n a b ∨ proposition4 m n a b) ∧
  ((proposition1 m n a b ∧ ¬ proposition2 m n a b ∧ proposition3 m n a b ∧ ¬ proposition4 m n a b) → 
    #(prop1 is true and neither 2 nor 4 are true) = 1)


end correct_propositions_count_l85_85508


namespace certain_event_l85_85766

-- Definitions of the events
def event1 : Prop := ∀ (P : ℝ), P ≠ 20.0
def event2 : Prop := ∀ (x : ℤ), x ≠ 105 ∧ x ≤ 100
def event3 : Prop := ∃ (r : ℝ), 0 ≤ r ∧ r ≤ 1 ∧ ¬(r = 0 ∨ r = 1)
def event4 (a b : ℝ) : Prop := ∃ (area : ℝ), area = a * b

-- Statement to prove that event4 is the only certain event
theorem certain_event (a b : ℝ) : (event4 a b) := 
by
  sorry

end certain_event_l85_85766


namespace min_value_of_expression_l85_85066

theorem min_value_of_expression (x y : ℝ) (h : x^2 + y^2 + x * y = 315) :
  ∃ m : ℝ, m = x^2 + y^2 - x * y ∧ m ≥ 105 :=
by
  sorry

end min_value_of_expression_l85_85066


namespace largest_divisor_product_die_l85_85402

-- Define the eight-sided die numbers set
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the factorial for n
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define the product Q of seven out of eight numbers visible on the die
def product_of_visible (n : ℕ) (h : n ∈ die_faces) : Finset ℕ → ℕ :=
  fun s => if s = die_faces.erase n then s.prod id else 0 

-- The largest common divisor guaranteed to divide any of the products Q
theorem largest_divisor_product_die : (∀ n ∈ die_faces, gcd_a (product_of_visible (die_faces.erase n)) Q 960 :=
by 
  sorry

end largest_divisor_product_die_l85_85402


namespace point_on_curve_E_max_area_triangle_l85_85486

noncomputable def trajectory_eq := 
  ∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1)

theorem point_on_curve_E (x y : ℝ) (P Q A B F F' : ℝ × ℝ)
  (hC : (x - 1)^2 + y^2 = 16)
  (hF'_sym : F' = (-1, 0))
  (hP : (x^2 / 4 + y^2 / 3 = 1))
  (hA : A = (4,0))
  (hPQ_perp_x : Q.2 = -P.2)
  (hQA_PF_intersect : ∃ B, (AQ_intersects PF))
  : (x_B^2 / 4 + y_B^2 / 3 = 1 ) :=
sorry

theorem max_area_triangle (P A B : ℝ × ℝ) (hP : (x^2 / 4 + y^2 / 3 = 1)) 
  (hA : A = (4,0)) : 
  ∃ max_area : ℝ, max_area = 9 / 2 :=
sorry

end point_on_curve_E_max_area_triangle_l85_85486


namespace equation_solution_l85_85717

theorem equation_solution (x : ℝ) (hx : x > 0) : x = 4 ↔ x^(x^(real.sqrt x)) = x^(real.sqrt (x^x)) :=
by
  sorry

end equation_solution_l85_85717


namespace digit_8_occurrences_from_1_to_700_l85_85145

theorem digit_8_occurrences_from_1_to_700 : 
  let occurrences (d : ℕ) (n : ℕ) := (finset.range d).sum (λ x, (nat.digits 10 x).count n) in
  occurrences 701 8 = 240 :=
by
  sorry

end digit_8_occurrences_from_1_to_700_l85_85145


namespace betty_total_stones_l85_85409

def stones_per_bracelet : ℕ := 14
def number_of_bracelets : ℕ := 10
def total_stones : ℕ := stones_per_bracelet * number_of_bracelets

theorem betty_total_stones : total_stones = 140 := by
  sorry

end betty_total_stones_l85_85409


namespace min_water_price_for_increased_revenue_min_revenue_l85_85168

variable (a k x : ℝ)

-- Part 1: Analytical Expression for Revenue
def revenue (a k x : ℝ) : ℝ := 
  (a + k / (x - 2)) * (x - 1.8)

-- Part 2: Minimum Water Price for at Least 20% Revenue Increase
theorem min_water_price_for_increased_revenue (h₁ : 2.3 ≤ x) (h₂ : x ≤ 2.6) (k_eq : k = 0.4 * a) :
  (a + 0.4 * a / (x - 2)) * (x - 1.8) ≥ 1.2 * a → x = 2.4 := 
sorry

-- Part 3: Water Price for Minimum Revenue and Minimum Revenue
theorem min_revenue (h₁ : 2.3 ≤ x) (h₂ : x ≤ 2.6) (k_eq : k = 0.8 * a) :
  (x = 2.4 ∧ revenue a k x = 1.8 * a) :=
sorry

end min_water_price_for_increased_revenue_min_revenue_l85_85168


namespace number_to_remove_l85_85328

theorem number_to_remove (l : List ℕ) (target_avg : ℚ) (x : ℕ)
  (h : l = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  (h_avg : (l.sum - x) / (l.length - 1) = target_avg) :
  x = 6 :=
begin
  have l_length : l.length = 11, by { rw h, simp },
  have l_sum : l.sum = 77, by { rw h, simp },
  have remaining_sum : 71 = 10 * 7.1, by { norm_num },
  rw [l_sum, remaining_sum] at h_avg,
  simp at h_avg,
  norm_num at h_avg,
  exact h_avg
end

end number_to_remove_l85_85328


namespace value_of_expression_l85_85842

theorem value_of_expression (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - y^2 = 53) :
  x^3 - y^3 - 2 * (x + y) + 10 = 2011 :=
sorry

end value_of_expression_l85_85842


namespace least_value_of_x_is_840_l85_85594

noncomputable def least_value_of_x : ℕ :=
  let evn_prime := 2
  let x := evn_prime * 12 * p * q
  if 2 * p - q = 3 ∧ p.prime ∧ q.prime ∧ p ≠ q ∧ q > p then x else 0

theorem least_value_of_x_is_840 (p q : ℕ) (hp : p.prime) (hq : q.prime) (h1 : p ≠ q) (h2 : q > p) (h3 : 2 * p - q = 3) :
  (∃ x : ℕ, (x = 24 * p * q ∧ even_prime (x / (12 * p * q)) = 2 ∧ 0 < x)) → (least_value_of_x p q = 840) :=
sorry

end least_value_of_x_is_840_l85_85594


namespace min_surface_area_of_sphere_l85_85825

theorem min_surface_area_of_sphere (a b c : ℝ) (volume : ℝ) (height : ℝ) 
  (h_volume : a * b * c = volume) (h_height : c = height) 
  (volume_val : volume = 12) (height_val : height = 4) : 
  ∃ r : ℝ, 4 * π * r^2 = 22 * π := 
by
  sorry

end min_surface_area_of_sphere_l85_85825


namespace number_of_female_officers_l85_85973

theorem number_of_female_officers (total_on_duty : ℕ) (female_on_duty : ℕ) (percentage_on_duty : ℚ) : 
  total_on_duty = 500 → 
  female_on_duty = 250 → 
  percentage_on_duty = 1/4 → 
  (female_on_duty : ℚ) = percentage_on_duty * (total_on_duty / 2 : ℚ) →
  (total_on_duty : ℚ) = 4 * female_on_duty →
  total_on_duty = 1000 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end number_of_female_officers_l85_85973


namespace mixed_candy_price_l85_85741

theorem mixed_candy_price (x y : ℝ) (hx : 3 * x = 2 * y): 
  ((3 * x + 2 * y) / (x + y)) = 2.4 :=
by
  have hxy : y = 3 * x / 2 := by linarith,
  calc
    (3 * x + 2 * y) / (x + y)
        = (3 * x + 2 * (3 * x / 2)) / (x + 3 * x / 2) : by rw [hxy]
    ... = (3 * x + 3 * x) / (x + 1.5 * x)             : by linarith
    ... = 6 * x / (2.5 * x)                           : by linarith
    ... = 2.4                                       : by norm_num

end mixed_candy_price_l85_85741


namespace square_area_ratio_l85_85390

theorem square_area_ratio 
  (r : ℝ) -- radius of the circle
  (s : ℝ) -- side length of the inscribed square
  (h_inscribed : s = r * (real.sqrt 2)) -- inscribed square diagonal equal to circle diameter
  (h_circumscribed : r * (real.sqrt 2) = s * (real.sqrt 2)) -- circumscribed square side length equal to circle diameter
  : (s * (real.sqrt 2))^2 / s^2 = 2 :=
sorry

end square_area_ratio_l85_85390


namespace mail_distribution_l85_85370

def pieces_per_block (total_pieces blocks : ℕ) : ℕ := total_pieces / blocks

theorem mail_distribution : pieces_per_block 192 4 = 48 := 
by { 
    -- Proof skipped
    sorry 
}

end mail_distribution_l85_85370


namespace triangle_is_isosceles_l85_85606

/--
Given a triangle \( ABC \), with \( BM \) as the median,
a point \( K \) on \( BM \) such that \( AK = BC \),
and the ray \( AK \) intersects side \( BC \) at point \( P \).
Prove that triangle \( B K P \) is isosceles.
-/
theorem triangle_is_isosceles
  (A B C M K P : Type)
  [triangle A B C]              -- A B C forms a triangle
  (BM : Segment B M)            -- BM is a segment
  (is_median : is_median_of BM) -- BM is a median of triangle ABC
  (K_on_BM : on_segment K BM)   -- K is on the segment BM
  (eq_AK_BC : Segment_Equal (Segment A K) (Segment B C)) -- AK = BC
  (P_on_AK : on_ray P (Ray A K)) -- P is on the ray AK
  (P_on_BC : on_segment P (Segment B C)) -- P is on the segment BC
: is_isosceles (Triangle B K P) := sorry

end triangle_is_isosceles_l85_85606


namespace carl_drives_total_hours_in_two_weeks_l85_85680

theorem carl_drives_total_hours_in_two_weeks
  (daily_hours : ℕ) (additional_weekly_hours : ℕ) :
  (daily_hours = 2) 
  → (additional_weekly_hours = 6)
  → (14 * daily_hours + 2 * additional_weekly_hours = 40) := 
by
  intros h1 h2
  rw [h1, h2]
  calc
  14 * 2 + 2 * 6 = 28 + 12 := by ring
                ... = 40   := by rfl

end carl_drives_total_hours_in_two_weeks_l85_85680


namespace range_of_m_l85_85891

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m^2 * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x) ↔ -2 < m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l85_85891


namespace driving_hours_l85_85683

theorem driving_hours (days_in_week : ℕ) (initial_hours_per_day weekly_additional_hours : ℕ) :
  days_in_week = 7 → initial_hours_per_day = 2 → weekly_additional_hours = 6 → 
  2 * days_in_week + weekly_additional_hours = 20 →
  2 * (initial_hours_per_day * days_in_week + weekly_additional_hours) = 40 :=
by
  intros h_days h_initial h_weekly h_week_hours
  rw [h_days, h_initial, h_weekly, h_week_hours]
  simp
  sorry

end driving_hours_l85_85683


namespace max_cars_div_10_l85_85969

noncomputable def max_cars (car_length : ℕ) (distance_for_speed : ℕ → ℕ) (speed : ℕ → ℕ) : ℕ :=
  let k := 2000
  2000 -- Maximum number of cars passing the sensor

theorem max_cars_div_10 (car_length : ℕ) (distance_for_speed : ℕ → ℕ) (speed : ℕ → ℕ) :
  car_length = 5 →
  (∀ k : ℕ, distance_for_speed k = k) →
  (∀ k : ℕ, speed k = 10 * k) →
  (max_cars car_length distance_for_speed speed) = 2000 → 
  (max_cars car_length distance_for_speed speed) / 10 = 200 := by
  intros
  sorry

end max_cars_div_10_l85_85969


namespace concurrency_lines_l85_85685

open EuclideanGeometry

variable {Point : Type}
variable [IncidencePlane Point]

/-- Definition of an equilateral triangle -/
def equilateral_triangle (A B C D E F M N P : Point) :=
  triangle A B C ∧
  midpoint D B C ∧
  midpoint E C A ∧
  midpoint F A B ∧
  midpoint M B F ∧
  midpoint N D F ∧
  midpoint P D C

/-- The concurrency of lines AN, FP, and EM in the context of given conditions -/
theorem concurrency_lines
  (A B C D E F M N P : Point)
  (h_eq_triangle : equilateral_triangle A B C D E F M N P) :
  concurrent (line_through A N) (line_through F P) (line_through E M) :=
sorry

end concurrency_lines_l85_85685


namespace shaded_fraction_of_large_square_l85_85647

theorem shaded_fraction_of_large_square :
  let large_square_area := 1
  let unit_squares := 16
  let unit_square_area := large_square_area / unit_squares
  let triangles_per_shaded_square := 2
  let shaded_triangles := 4
  let shaded_triangle_area := unit_square_area / triangles_per_shaded_square
  (shaded_triangle_area * shaded_triangles) / large_square_area = 1 / 8 :=
by
  -- Define areas and conditions
  let large_square_area := 1 : ℝ
  let unit_squares := 16 : ℕ
  let unit_square_area := large_square_area / (unit_squares : ℝ)
  let triangles_per_shaded_square := 2 : ℕ
  let shaded_triangles := 4 : ℕ
  let shaded_triangle_area := unit_square_area / (triangles_per_shaded_square : ℝ)
  
  -- Compute the shaded area of triangles
  have total_shaded_area := shaded_triangle_area * (shaded_triangles : ℝ)
  
  -- Simplify and compare with 1/8
  show total_shaded_area / large_square_area = 1 / 8
  sorry

end shaded_fraction_of_large_square_l85_85647


namespace day_of_18th_day_of_month_is_tuesday_l85_85991

theorem day_of_18th_day_of_month_is_tuesday
  (day_of_24th_is_monday : ℕ → ℕ)
  (mod_seven : ∀ n, n % 7 = n)
  (h24 : day_of_24th_is_monday 24 = 1) : day_of_24th_is_monday 18 = 2 :=
by
  sorry

end day_of_18th_day_of_month_is_tuesday_l85_85991


namespace smallest_norm_v_l85_85214

noncomputable def v : Type := ℝ × ℝ

theorem smallest_norm_v
  (v : v)
  (h : ∥v + ⟨4, 2⟩∥ = 10) :
  ∥v∥ = 10 - 2 * Real.sqrt 5 :=
sorry

end smallest_norm_v_l85_85214


namespace tan_22_5_equiv_l85_85662

theorem tan_22_5_equiv : 
  ∃ a b c d : ℕ, a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a + b + c + d = 4 ∧ 
  (tan (real.pi / 8) = real.sqrt a - real.sqrt b + real.sqrt c - d) :=
by
  have h1 : real.sin (real.pi / 4) = real.sqrt 2 / 2 := sorry,
  have h2 : real.cos (real.pi / 4) = real.sqrt 2 / 2 := sorry,
  have tan_half_angle : tan (real.pi / 8) = (1 - real.sqrt 2 / 2) / (real.sqrt 2 / 2), from sorry,
  have tan_val : tan (real.pi / 8) = real.sqrt 2 - 1, from sorry,
  existsi [2, 1, 0, 1],
  split,
  { -- Verify inequalities
    repeat { split }; linarith },
  split,
  { -- Sum of variables
    norm_num },
  -- Check the expression equivalence
  exact tan_val

end tan_22_5_equiv_l85_85662


namespace num_ways_to_sum_3060_with_2s_and_3s_l85_85876

theorem num_ways_to_sum_3060_with_2s_and_3s : 
  (∃ n : ℕ, list.sum (list.replicate (2 * n) 2 ++ list.replicate (3 * (510 - n)) 3) = 3060) ∧
  fintype.card {n : ℕ | 0 ≤ n ∧ n ≤ 510 ∧ list.sum (list.replicate (2 * n) 2 ++ list.replicate (3 * (510 - n)) 3) = 3060 } = 511 :=
sorry

end num_ways_to_sum_3060_with_2s_and_3s_l85_85876


namespace Rank_Regions_Consumption_l85_85424

def RankRegionsBasedOnConsumption (west nonWest russia : ℕ) : Prop :=
  ∃ rankWest rankNonWest rankRussia : ℕ, 
  rankWest = 1 ∧ 
  rankNonWest = 2 ∧ 
  rankRussia = 3 ∧ 
  west <= nonWest ∧ 
  nonWest <= russia

theorem Rank_Regions_Consumption:
  RankRegionsBasedOnConsumption 21428 26848.55 302790.13 :=
begin
  sorry
end

end Rank_Regions_Consumption_l85_85424


namespace functional_eq_to_negx_l85_85433

def f (x : ℤ) : ℤ := sorry 

theorem functional_eq_to_negx (f: ℤ → ℤ) (h: ∀ x y : ℤ, f(x + f(y)) = f(x) - y) : 
  f = (λ x, -x) :=
by 
  sorry

end functional_eq_to_negx_l85_85433


namespace problem_x_y_value_l85_85573

def floor (z : ℝ) : ℤ := Int.floor z

theorem problem_x_y_value (x y : ℝ) (h1 : y = 3 * floor x + 4) (h2 : y = 4 * floor (x - 3) + 7) (h3 : ¬ ∃ n : ℤ, x = n) :
  40 < x + y ∧ x + y < 41 :=
by
  sorry

end problem_x_y_value_l85_85573


namespace find_y_l85_85843

variable (x y : ℝ)
noncomputable def i : ℂ := complex.I -- Imaginary unit definition

theorem find_y (h : (2*x + i) * (1 - i) = y) : y = 2 := by
  sorry

end find_y_l85_85843


namespace max_axes_of_symmetry_three_segments_l85_85693

-- We define the problem statement using Lean language constructs.

def maxAxesOfSymmetry (A B C : Set Point) : Nat :=
max (axesOfSymmetry A ∪ axesOfSymmetry B ∪ axesOfSymmetry C)

theorem max_axes_of_symmetry_three_segments (A B C : Set Point) :
  maxAxesOfSymmetry A B C ≤ 6 :=
sorry

end max_axes_of_symmetry_three_segments_l85_85693


namespace part1_balanced_1_part1_balanced_2_part2_not_balanced_l85_85480

section BalancedNumbers

variable {a b m : ℝ}
 
-- Definition of balanced numbers
def is_balanced (a b : ℝ) := a + b = 2

-- Part (1)
theorem part1_balanced_1 : is_balanced (-1) 3 :=
by
  dsimp [is_balanced]
  linarith

theorem part1_balanced_2 : is_balanced (1 - sqrt 2) (1 + sqrt 2) :=
by
  dsimp [is_balanced]
  rw [add_assoc, add_right_neg, add_comm, add_sub_assoc, sqrt_eq_2_rpow_half]
  ring

-- Part (2)
theorem part2_not_balanced {m : ℝ} (h: (sqrt 3 + m) * (sqrt 3 - 1) = 2) : ¬ is_balanced (m + sqrt 3) (2 - sqrt 3) :=
by
  have h1 : m = 1 :=
    by
      have h2 : (sqrt 3) * (sqrt 3 - 1) = 3 - sqrt 3
      field_simp [sqrt_ne_zero.2, ne_of_lt, hi.sqrt 3 (ne_of_gt one_lt_three)]
      rw [mul_sub, mul_one, one_mul, sub_eq_add_neg, ← sub_eq_add_neg]
      linarith
    linarith
  dsimp [is_balanced]
  rw [h1]
  linarith

end BalancedNumbers

end part1_balanced_1_part1_balanced_2_part2_not_balanced_l85_85480


namespace horner_method_multiplications_additions_count_l85_85777

-- Define the polynomial function
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 - 2 * x^2 + 4 * x - 6

-- Define the property we want to prove
theorem horner_method_multiplications_additions_count : 
  ∃ (multiplications additions : ℕ), multiplications = 4 ∧ additions = 4 := 
by
  sorry

end horner_method_multiplications_additions_count_l85_85777


namespace quadratic_inequality_l85_85153

variable {a b c : ℝ}
def f (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem quadratic_inequality (h1 : a > 0) 
                             (h2 : ∀ x, f x > 0 ↔ x < -2 ∨ x > 4) :
  f 2 < f (-1) ∧ f (-1) < f 5 :=
by
  sorry

end quadratic_inequality_l85_85153


namespace no_squares_with_side_at_least_8_l85_85296

def H : set (ℤ × ℤ) := {p | 2 ≤ |p.1| ∧ |p.1| ≤ 6 ∧ 2 ≤ |p.2| ∧ |p.2| ≤ 6}

theorem no_squares_with_side_at_least_8 :
  ∃ (n : ℕ), (∀ (square : list (ℤ × ℤ)), square.length = 4 → 
                (∀ 1 ≤ i ≤ 3, dist square[i] square[(i+1)%4] = 8) → 
                (∀ p ∈ square, p ∈ H) → n = 0) :=
by
  sorry

end no_squares_with_side_at_least_8_l85_85296


namespace octagon_area_equals_eight_one_plus_sqrt_two_l85_85003

theorem octagon_area_equals_eight_one_plus_sqrt_two
  (a b : ℝ)
  (h1 : 4 * a = 8 * b)
  (h2 : a ^ 2 = 16) :
  2 * (1 + Real.sqrt 2) * b ^ 2 = 8 * (1 + Real.sqrt 2) :=
by
  sorry

end octagon_area_equals_eight_one_plus_sqrt_two_l85_85003


namespace graph_intersection_points_l85_85629

open Function

theorem graph_intersection_points (g : ℝ → ℝ) (h_inv : Involutive (invFun g)) : 
  ∃! (x : ℝ), x = 0 ∨ x = 1 ∨ x = -1 → g (x^2) = g (x^6) :=
by sorry

end graph_intersection_points_l85_85629


namespace graph_properties_l85_85555

theorem graph_properties (k b : ℝ) (hk : k ≠ 0) (hb : b ≠ 0) (positive_kb : k * b > 0) :
  (∃ (f g : ℝ → ℝ),
    (∀ x, f x = k * x + b) ∧
    (∀ x (hx : x ≠ 0), g x = k * b / x) ∧
    -- Under the given conditions, the graphs must match option (B)
    (True)) := sorry

end graph_properties_l85_85555


namespace first_digit_base_4_of_853_l85_85691

theorem first_digit_base_4_of_853 : 
  ∃ (d : ℕ), d = 3 ∧ (d * 256 ≤ 853 ∧ 853 < (d + 1) * 256) :=
by
  sorry

end first_digit_base_4_of_853_l85_85691


namespace freds_sister_borrowed_3_dimes_l85_85452

-- Define the conditions
def original_dimes := 7
def remaining_dimes := 4

-- Define the question and answer
def borrowed_dimes := original_dimes - remaining_dimes

-- Statement to prove
theorem freds_sister_borrowed_3_dimes : borrowed_dimes = 3 := by
  sorry

end freds_sister_borrowed_3_dimes_l85_85452


namespace vec_dot_product_problem_l85_85846

noncomputable def vec_dot_product {n : ℕ} (a b : EuclideanSpace ℝ (Fin n)) : ℝ :=
  (a • b : ℝ)

theorem vec_dot_product_problem (a b : EuclideanSpace ℝ (Fin 2))
  (ha : ‖a‖ = 2) (hb : ‖b‖ = 1) (angle_ab : real.angle a b = real.angle.pi_div_three) :
  vec_dot_product a (a - b) = 3 :=
by 
  sorry

end vec_dot_product_problem_l85_85846


namespace distance_CD_l85_85417

open Real

def ellipse_eq (x y : ℝ) : Prop := 4 * (x - 2) ^ 2 + 16 * y ^ 2 = 64

def major_axis_endpoint : (ℝ × ℝ) := (6, 0) -- C, one endpoint of the major axis
def minor_axis_endpoint : (ℝ × ℝ) := (2, 2) -- D, one endpoint of the minor axis

theorem distance_CD :
  let C := major_axis_endpoint in
  let D := minor_axis_endpoint in
  dist C D = 2 * sqrt 5 :=
by
  sorry

end distance_CD_l85_85417


namespace number_of_parents_l85_85678

theorem number_of_parents (P : ℕ) (h : P + 177 = 238) : P = 61 :=
by
  sorry

end number_of_parents_l85_85678


namespace largest_non_provable_amount_is_correct_l85_85546

def limonia_coin_denominations (n : ℕ) : List ℕ :=
  List.range (n + 1) |> List.map (λ i => 3^(n - i) * 4^i)

def largest_non_provable_amount (n : ℕ) : ℕ :=
  2 * 4^(n+1) - 3^(n+2)

theorem largest_non_provable_amount_is_correct (n : ℕ) :
  ∀ s : ℕ, (¬∃ c : List ℕ, 
    (c ∈ List.powerset (limonia_coin_denominations n)) ∧ (s = c.sum)) 
    ↔ s = largest_non_provable_amount n := 
by
  sorry

end largest_non_provable_amount_is_correct_l85_85546


namespace propositions_validity_l85_85399

theorem propositions_validity :
  (¬(x^2 + y^2 ≠ 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  ((x^2 + 2 * x + q = 0 → q < 2) ↔ q < 2) ∧
  (¬(¬ ∃ (T₁ T₂ : Type), congruent T₁ T₂ → Area T₁ = Area T₂)) ∧
  ((∀ a : ℝ, a > 1 → log a > 0) ↔ (∀ a : ℝ, log a ≤ 0 → a ≤ 1))
  :=
by
  -- proof skipped
  sorry

end propositions_validity_l85_85399


namespace intersection_points_parabola_l85_85164

noncomputable def parabola : ℝ → ℝ := λ x => x^2

noncomputable def directrix : ℝ → ℝ := λ x => -1

noncomputable def other_line (m c : ℝ) : ℝ → ℝ := λ x => m * x + c

theorem intersection_points_parabola {m c : ℝ} (h1 : ∃ x1 x2 : ℝ, other_line m c x1 = parabola x1 ∧ other_line m c x2 = parabola x2) :
  (∃ x1 x2 : ℝ, parabola x1 = other_line m c x1 ∧ parabola x2 = other_line m c x2 ∧ x1 ≠ x2) → 
  (∃ x1 x2 : ℝ, parabola x1 = other_line m c x1 ∧ parabola x2 = other_line m c x2 ∧ x1 = x2) := 
by
  sorry

end intersection_points_parabola_l85_85164


namespace range_of_slope_angle_l85_85739

theorem range_of_slope_angle (k : ℝ) :
  ∀ P : ℝ × ℝ,
  P = (-real.sqrt 3, -1) →
  (∃ x y : ℝ, (x^2 + y^2 = 1) ∧ (k * x - y + (real.sqrt 3) * k - 1 = 0)) →
  0 ≤ k ∧ k ≤ real.sqrt 3 ∧ 0 ≤ real.arctan k ∧ real.arctan k ≤ real.pi / 3 :=
by
  intros P hP hexists
  sorry

end range_of_slope_angle_l85_85739


namespace power_of_a_l85_85892

theorem power_of_a (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 := sorry

end power_of_a_l85_85892


namespace complete_the_square_problem_l85_85785

theorem complete_the_square_problem :
  ∃ r s : ℝ, (r = -2) ∧ (s = 9) ∧ (r + s = 7) ∧ ∀ x : ℝ, 15 * x ^ 2 - 60 * x - 135 = 0 ↔ (x + r) ^ 2 = s := 
by
  sorry

end complete_the_square_problem_l85_85785


namespace mark_bread_baking_time_l85_85965

/--
Mark is baking bread. 
He has to let it rise for 120 minutes twice. 
He also needs to spend 10 minutes kneading it and 30 minutes baking it. 
Prove that the total time Mark takes to finish making the bread is 280 minutes.
-/
theorem mark_bread_baking_time :
  let rising_time := 120 * 2
  let kneading_time := 10
  let baking_time := 30
  rising_time + kneading_time + baking_time = 280 := 
by
  let rising_time := 120 * 2
  let kneading_time := 10
  let baking_time := 30
  have rising_time_eq : rising_time = 240 := rfl
  have kneading_time_eq : kneading_time = 10 := rfl
  have baking_time_eq : baking_time = 30 := rfl
  calc
    rising_time + kneading_time + baking_time
        = 240 + 10 + 30 : by rw [rising_time_eq, kneading_time_eq, baking_time_eq]
    ... = 280 : by norm_num

end mark_bread_baking_time_l85_85965


namespace tan_22_5_expression_l85_85660

theorem tan_22_5_expression :
  let a := 2
  let b := 1
  let c := 0
  let d := 0
  let t := Real.tan (Real.pi / 8)
  a ≥ b ∧ b ≥ c ∧ c ≥ d ∧
  t = (Real.sqrt a) - (Real.sqrt b) + (Real.sqrt c) - d → 
  a + b + c + d = 3 :=
by
  intros
  exact sorry

end tan_22_5_expression_l85_85660


namespace ellipse_standard_eqn_max_major_axis_l85_85869

-- Definitions for Lean equivalent proof

def is_eccentricity (a b : ℝ) (e : ℝ) : Prop := e = real.sqrt 3 / 3 ∧ a > b ∧ b > 0

def focal_distance (a : ℝ) : Prop := 2 * real.sqrt (a ^ 2 - (real.sqrt 3 / 3 * a) ^ 2) = 2

def orthogonal (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

def in_eccentricity_range (e : ℝ) : Prop :=
  (1 / 2 : ℝ) ≤ e ∧ e ≤ real.sqrt 2 / 2

-- Problem 1: Equation of the ellipse
theorem ellipse_standard_eqn (a b : ℝ) (e c : ℝ) 
  (h1 : is_eccentricity a b e) 
  (h2 : focal_distance a)
  : (a = real.sqrt 3 ∧ b = real.sqrt 2 ∧ e = real.sqrt 3 / 3) → 
    (ellipse_eqn : \(x: ℝ, y: ℝ) -> (x^2 / 3 + y^2 / 2 = 1) := sorry

-- Problem 2: Maximum length of the major axis
theorem max_major_axis (a : ℝ) (e : ℝ)
  (h1 : in_eccentricity_range e) 
  : (major_axis : \(2 * a) ∈ \(\left[\frac{\sqrt{42}}{3}, \sqrt{6}\right]\))
    → (max_length = \(\sqrt{6}\)) := sorry

end ellipse_standard_eqn_max_major_axis_l85_85869


namespace monotonically_increasing_interval_l85_85288

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x : ℝ, (k * Real.pi - 5 * Real.pi / 12 <= x ∧ x <= k * Real.pi + Real.pi / 12) →
    (∃ r : ℝ, f (x + r) > f x ∨ f (x + r) < f x) := by
  sorry

end monotonically_increasing_interval_l85_85288


namespace cakes_in_november_l85_85630

-- Define the function modeling the number of cakes baked each month
def num_of_cakes (initial: ℕ) (n: ℕ) := initial + 2 * n

-- Given conditions
def cakes_in_october := 19
def cakes_in_december := 23
def cakes_in_january := 25
def cakes_in_february := 27
def monthly_increase := 2

-- Prove that the number of cakes baked in November is 21
theorem cakes_in_november : num_of_cakes cakes_in_october 1 = 21 :=
by
  sorry

end cakes_in_november_l85_85630


namespace odd_function_result_l85_85848

variable {ℝ : Type*} [linear_ordered_field ℝ]

-- Define the problem context
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

noncomputable def f (x : ℝ) (a : ℝ) :=
if x ≥ 0 then x^2 - (a + 4) * x + a else - x^2 - 4 * x

-- Main statement
theorem odd_function_result {f : ℝ → ℝ} (a : ℝ) :
  (is_odd f) →
  (∀ x : ℝ, 0 ≤ x → f x = x^2 - (a + 4) * x + a) →
  (a = 0 ∧ (∀ x : ℝ, f x = (if x ≥ 0 then x^2 - 4 * x else - x^2 - 4 * x))) :=
sorry

end odd_function_result_l85_85848


namespace find_n_l85_85347

theorem find_n (n : ℕ) : (8 : ℝ)^(1/3) = (2 : ℝ)^n → n = 1 := by
  sorry

end find_n_l85_85347


namespace solve_equation_x_squared_eq_16x_l85_85669

theorem solve_equation_x_squared_eq_16x :
  ∀ x : ℝ, x^2 = 16 * x ↔ (x = 0 ∨ x = 16) :=
by 
  intro x
  -- Complete proof here
  sorry

end solve_equation_x_squared_eq_16x_l85_85669


namespace polynomial_simplification_l85_85261

-- Define the polynomials
def poly1 : ℕ → ℤ
| 0 := 15
| 1 := 0
| 2 := 1
| 3 := 0
| 4 := -3
| 5 := 1
| _ := 0

def poly2 : ℕ → ℤ
| 0 := 18
| 1 := 0
| 2 := 2
| 3 := -3
| 4 := 0
| 5 := 2
| _ := 0

-- Define the expected simplified polynomial
def simplified_poly : ℕ → ℤ
| 0 := -3
| 1 := 0
| 2 := -1
| 3 := 3
| 4 := -3
| 5 := -1
| _ := 0

-- Define the subtraction of two polynomials
def subtract_polys (p1 p2 : ℕ → ℤ) : ℕ → ℤ :=
λ n, p1 n - p2 n

-- Lean statement to prove the polynomial simplification
theorem polynomial_simplification :
  subtract_polys poly1 poly2 = simplified_poly :=
sorry

end polynomial_simplification_l85_85261


namespace playground_children_count_l85_85718

theorem playground_children_count (boys girls : ℕ) (h_boys : boys = 27) (h_girls : girls = 35) : boys + girls = 62 := by
  sorry

end playground_children_count_l85_85718


namespace find_phi_l85_85863

open Real

def f (x : ℝ) : ℝ := sin (2 * x + (π / 6))

theorem find_phi (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π / 2) 
  (h3 : ∀ x : ℝ, f (x - φ) = f (- (x - φ))) : φ = π / 3 :=
by
  sorry

end find_phi_l85_85863


namespace quadratic_real_roots_range_l85_85123

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, (m-1)*x^2 + x + 1 = 0) → (m ≤ 5/4 ∧ m ≠ 1) :=
by
  sorry

end quadratic_real_roots_range_l85_85123


namespace fraction_not_integer_l85_85226

def containsExactlyTwoOccurrences (d : List ℕ) : Prop :=
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7], d.count n = 2

theorem fraction_not_integer
  (k m : ℕ)
  (hk : 14 = (List.length (Nat.digits 10 k)))
  (hm : 14 = (List.length (Nat.digits 10 m)))
  (hkd : containsExactlyTwoOccurrences (Nat.digits 10 k))
  (hmd : containsExactlyTwoOccurrences (Nat.digits 10 m))
  (hkm : k ≠ m) :
  ¬ ∃ d : ℕ, k = m * d := 
sorry

end fraction_not_integer_l85_85226


namespace real_roots_count_l85_85589

noncomputable def f₀ (x : ℝ) : ℝ := (1 / 2) ^ |x|

noncomputable def f₁ (x : ℝ) : ℝ := |f₀ x - 1 / 2|

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  if n = 0 then f₀ x
  else if n = 1 then f₁ x
  else |f (n - 1) x - (1 / 2) ^ n|

theorem real_roots_count (n : ℕ) : 
  ∃ S : set ℝ, S.card = 2 ^ (n + 1) ∧ ∀ x ∈ S, f n x = (1 / (n + 2)) ^ n :=
sorry

end real_roots_count_l85_85589


namespace verify_Proposition1_verify_Proposition2_verify_Proposition3_verify_Proposition4_verify_Proposition5_l85_85197

noncomputable def Proposition1 (a b c : ℝ) (C : ℝ) :=
  a + b > 2 * c → C < π / 3

noncomputable def Proposition2 (a b c : ℝ) :=
  a^2 + b^2 > c^2 → ¬(∠ABC is acute)

noncomputable def Proposition3 (a b c A : ℝ) :=
  cos^2 (A / 2) = (b + c) / (2 * c) → ∠ABC is right

noncomputable def Proposition4 (a b A B : ℝ) :=
  (a / cos B) = (b / cos A) → ¬(∠ABC is isosceles)

noncomputable def Proposition5 (A B C : ℝ) :=
  (triangle ABC is acute) → ¬(sin A + sin B + tan C > (1/2) * cos A + (1/3) * cos B + (1/4) * cot C)

-- Proving these statements
theorem verify_Proposition1 (a b c C : ℝ) (h : a + b > 2 * c) : C < π / 3 := 
  sorry

theorem verify_Proposition2 (a b c : ℝ) (h : a^2 + b^2 > c^2) : ¬(∠ABC is acute) :=
  sorry

theorem verify_Proposition3 (a b c A : ℝ) (h : cos^2 (A / 2) = (b + c) / (2 * c)) : ∠ABC is right :=
  sorry

theorem verify_Proposition4 (a b A B : ℝ) (h : (a / cos B) = (b / cos A)) : ¬(∠ABC is isosceles) :=
  sorry

theorem verify_Proposition5 (A B C : ℝ) (h : (triangle ABC is acute)) : ¬(sin A + sin B + tan C > (1/2) * cos A + (1/3) * cos B + (1/4) * cot C) :=
  sorry

end verify_Proposition1_verify_Proposition2_verify_Proposition3_verify_Proposition4_verify_Proposition5_l85_85197


namespace find_f2_l85_85497

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x y : ℝ, x * f y = y * f x) (h10 : f 10 = 30) : f 2 = 6 := 
by
  sorry

end find_f2_l85_85497


namespace max_S_value_l85_85586

noncomputable def S (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :=
min x (min (y + 1/x) (1/y))

theorem max_S_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  S x y hx hy ≤ real.sqrt 2 :=
sorry

end max_S_value_l85_85586


namespace max_sin_y_minus_cos_squared_x_l85_85065

theorem max_sin_y_minus_cos_squared_x (x y : ℝ) :
  sin x + sin y = 1 / 3 →
  (sin y - cos x ^ 2) ≤ 4 / 9 :=
  sorry

end max_sin_y_minus_cos_squared_x_l85_85065


namespace volume_of_given_sphere_l85_85273

noncomputable def volume_of_sphere (A d : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (Real.sqrt (d^2 + A / Real.pi))^3

theorem volume_of_given_sphere
  (hA : 2 * Real.pi = 2 * Real.pi)
  (hd : 1 = 1):
  volume_of_sphere (2 * Real.pi) 1 = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end volume_of_given_sphere_l85_85273


namespace polynomial_irreducibility_l85_85342

def polynomial {R : Type*} [CommRing R] [IsDomain R] := R[X]

noncomputable def problem (b k : ℕ) (f : polynomial ℤ) : Prop :=
  (1 < k ∧ k < b) ∧
  (∀ i : ℕ, i ≤ nat_degree f → coeff f i ≥ 0) ∧
  (∃ p : ℤ, nat.prime (int.to_nat p) ∧ eval ↑b f = k * p) ∧
  (∀ r : ℂ, eval r f = 0 → abs (r - ↑b) > real.sqrt (↑k))

theorem polynomial_irreducibility (b k : ℕ) (f : polynomial ℤ) :
  problem b k f → irreducible f :=
sorry

end polynomial_irreducibility_l85_85342


namespace fixed_point_coordinates_l85_85849

theorem fixed_point_coordinates (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : 
  ∃ P : ℝ × ℝ, P = (1, 3) ∧ ∀ x, f x = 2 + a^(x-1) :=
by
  -- Define the function f(x)
  let f := λ x, 2 + a^(x-1)
  -- Define the fixed point P
  let P := (1, 3)
  -- State that the function passes through P
  have H : f (1) = 3,
  { sorry },
  -- Return the coordinates of P
  exact ⟨P, rfl, H⟩

end fixed_point_coordinates_l85_85849


namespace average_speed_ratio_l85_85559

theorem average_speed_ratio
  (distance : ℝ)
  (jack_time jill_time jamie_time : ℝ)
  (h_distance : distance = 42)
  (h_jack_time : jack_time = 5)
  (h_jill_time : jill_time = 4.2)
  (h_jamie_time : jamie_time = 3.5) :
  let jack_speed := distance / jack_time,
      jill_speed := distance / jill_time,
      jamie_speed := distance / jamie_time,
      min_speed := min jack_speed (min jill_speed jamie_speed) in
  (jack_speed / min_speed) : (jill_speed / min_speed) : (jamie_speed / min_speed) = 100 : 119 : 143 := 
sorry

end average_speed_ratio_l85_85559


namespace find_a₃_l85_85215

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

-- Condition: a is a geometric sequence with common ratio 2
def geom_sequence (a : ℕ → ℚ) (r : ℚ) : Prop :=
  ∀ n, a (n + 1) = r * a n

-- Sum of the first n terms of a sequence
def sum_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n, S n = (finset.range n).sum a

-- Given conditions
axiom a_geom_seq : geom_sequence a 2
axiom sum_to_S : sum_sequence a S
axiom condition : S 4 = 2 * S 2 + 1

-- Prove a₃ = 4/9
theorem find_a₃ : a 3 = 4 / 9 := 
by sorry

end find_a₃_l85_85215


namespace find_b_range_l85_85281

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^3 - 3*b*x + 3*b

theorem find_b_range (b : ℝ) :
  (∃ x, 0 < x ∧ x < 1 ∧ (∀ x₁, 0 < x₁ ∧ x₁ < 1 → f(x₁, b) > f(x, b))) →
  (3 * (0 : ℝ)^2 - 3 * b < 0 ∧ 3 * (1 : ℝ)^2 - 3 * b > 0) :=
by sorry

end find_b_range_l85_85281


namespace range_of_a_l85_85531

noncomputable def A : set ℝ := { x | (x - 3) / (x + 1) ≥ 0 }
noncomputable def B (a : ℝ) : set ℝ := { x | a * x + 1 ≤ 0 }

theorem range_of_a (a : ℝ) : B a ⊆ A ↔ -1/3 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l85_85531


namespace balloon_arrangements_l85_85139

open Finset

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem balloon_arrangements : 
  let n := 7
      p1 := 2
  in  (factorial n) / ((factorial p1)) = 2520 := by
{
  sorry
}

end balloon_arrangements_l85_85139


namespace find_distance_AB_l85_85689

variable (vA vB : ℝ) -- speeds of Person A and Person B
variable (x : ℝ) -- distance between points A and B
variable (t1 t2 : ℝ) -- time variables

-- Conditions
def startTime := 0
def meetDistanceBC := 240
def returnPointBDistantFromA := 120
def doublingSpeedFactor := 2

-- Main questions and conditions
theorem find_distance_AB
  (h1 : vA > vB)
  (h2 : t1 = x / vB)
  (h3 : t2 = 2 * (x - meetDistanceBC) / vA) 
  (h4 : x = meetDistanceBC + returnPointBDistantFromA + (t1 * (doublingSpeedFactor * vB) - t2 * vA) / (doublingSpeedFactor - 1)) :
  x = 420 :=
sorry

end find_distance_AB_l85_85689


namespace lines_divide_circle_into_four_arcs_l85_85831

theorem lines_divide_circle_into_four_arcs (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 1 → y = x + a ∨ y = x + b) →
  a^2 + b^2 = 2 :=
by
  sorry

end lines_divide_circle_into_four_arcs_l85_85831


namespace number_of_self_inverse_subsets_is_15_l85_85157

-- Define the set M
def M : Set ℚ := ({-1, 0, 1/2, 1/3, 1, 2, 3, 4} : Set ℚ)

-- Definition of self-inverse set
def is_self_inverse (A : Set ℚ) : Prop := ∀ x ∈ A, 1/x ∈ A

-- Theorem stating the number of non-empty self-inverse subsets of M
theorem number_of_self_inverse_subsets_is_15 :
  (∃ S : Finset (Set ℚ), S.card = 15 ∧ ∀ A ∈ S, A ⊆ M ∧ is_self_inverse A) :=
sorry

end number_of_self_inverse_subsets_is_15_l85_85157


namespace sachin_younger_than_rahul_l85_85256

theorem sachin_younger_than_rahul
  (S R : ℝ)
  (h1 : S = 24.5)
  (h2 : S / R = 7 / 9) :
  R - S = 7 := 
by sorry

end sachin_younger_than_rahul_l85_85256


namespace n_is_prime_l85_85930

theorem n_is_prime (p : ℕ) (h : ℕ) (n : ℕ)
  (hp : Nat.Prime p)
  (hh : h < p)
  (hn : n = p * h + 1)
  (div_n : n ∣ (2^(n-1) - 1))
  (not_div_n : ¬ n ∣ (2^h - 1)) : Nat.Prime n := sorry

end n_is_prime_l85_85930


namespace min_slope_of_tangent_l85_85401

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

theorem min_slope_of_tangent : (∀ x : ℝ, 3 * (x + 1)^2 + 3 ≥ 3) :=
by 
  sorry

end min_slope_of_tangent_l85_85401


namespace number_of_males_correct_l85_85350

variable (TotalPopulation FemalesWithGlasses : ℕ)

def percentage_of_females_with_glasses : ℝ := 0.30

noncomputable def females (FemalesWithGlasses : ℕ) : ℕ :=
  FemalesWithGlasses / 0.30

noncomputable def males (TotalPopulation FemalesWithGlasses : ℕ) : ℕ :=
  TotalPopulation - females FemalesWithGlasses

theorem number_of_males_correct
  (h1 : TotalPopulation = 5000)
  (h2 : percentage_of_females_with_glasses = 0.30)
  (h3 : FemalesWithGlasses = 900) :
  males TotalPopulation FemalesWithGlasses = 2000 :=
  by
  sorry

end number_of_males_correct_l85_85350


namespace probability_at_5_5_equals_1_over_243_l85_85743

-- Define the base probability function P
def P : ℕ → ℕ → ℚ
| 0, 0       => 1
| x+1, 0     => 0
| 0, y+1     => 0
| x+1, y+1   => (1/3 : ℚ) * P x (y+1) + (1/3 : ℚ) * P (x+1) y + (1/3 : ℚ) * P x y

-- Theorem statement that needs to be proved
theorem probability_at_5_5_equals_1_over_243 : P 5 5 = 1 / 243 :=
sorry

end probability_at_5_5_equals_1_over_243_l85_85743


namespace arithmetic_seq_a2_l85_85104

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n m : ℕ, a m = a (n + 1) + d * (m - (n + 1))

theorem arithmetic_seq_a2 
  (a : ℕ → ℤ) (d a1 : ℤ)
  (h_arith: ∀ n : ℕ, a n = a1 + n * d)
  (h_sum: a 3 + a 11 = 50)
  (h_a4: a 4 = 13) :
  a 2 = 5 :=
sorry

end arithmetic_seq_a2_l85_85104


namespace E_card_comparison_l85_85569

def E (T : Set ℕ) (p : ℕ) : Set (Fin (p - 1) → ℕ) :=
  { x | (∀ i, x i ∈ T) ∧ (∑ i in Finset.range (p - 1), (i + 1) * x ⟨i, Fin.is_lt _⟩) % p = 0 }

def cardinality {α : Type} (s : Set α) : ℕ :=
  if h : Nonempty α then Finset.card (s.toFinset h) else 0

theorem E_card_comparison (p : ℕ) (hp : Nat.Prime p) (hp3 : 3 < p) :
  cardinality (E {0, 1, 3} p) ≥ cardinality (E {0, 1, 2} p) ∧
  (cardinality (E {0, 1, 3} p) = cardinality (E {0, 1, 2} p) ↔ p = 5) := by
  sorry

end E_card_comparison_l85_85569


namespace sequence_strictly_increasing_from_14_l85_85294

def a (n : ℕ) : ℤ := n^4 - 20 * n^2 - 10 * n + 1

theorem sequence_strictly_increasing_from_14 :
  ∀ n : ℕ, n ≥ 14 → a (n + 1) > a n :=
by
  sorry

end sequence_strictly_increasing_from_14_l85_85294


namespace range_f_g_f_eq_g_implies_A_l85_85872

open Set

noncomputable def f (x : ℝ) : ℝ := x^2 + 1
noncomputable def g (x : ℝ) : ℝ := 4 * x + 1

theorem range_f_g :
  (range f ∩ Icc 1 17 = Icc 1 17) ∧ (range g ∩ Icc 1 17 = Icc 1 17) :=
sorry

theorem f_eq_g_implies_A :
  ∀ A ⊆ Icc 0 4, (∀ x ∈ A, f x = g x) → A = {0} ∨ A = {4} ∨ A = {0, 4} :=
sorry

end range_f_g_f_eq_g_implies_A_l85_85872


namespace triangle_is_isosceles_l85_85830

/- Define the points A, B, C, D as elements of a plane -/
variables (A B C D : Point) 

/- Assume that they are distinct -/
axiom distinct_points : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A ≠ C ∧ B ≠ D

/- Assume the given vector equation -/
axiom given_equation : 
  let DB := B - D in 
  let DC := C - D in 
  let DA := A - D in 
  let AB := B - A in 
  let AC := C - A in 
  (DB + DC - 2 * DA) • (AB - AC) = 0

/- Prove that triangle ABC is isosceles -/
theorem triangle_is_isosceles : 
  isosceles_triangle A B C :=
sorry

end triangle_is_isosceles_l85_85830


namespace number_of_roots_l85_85439

theorem number_of_roots : 
  ∃ (solutions : set ℝ), 
    (∀ x ∈ solutions, sqrt (9 - x) = 2 * x * sqrt (9 - x)) ∧ 
    (solutions = {9, 1 / 2}) ∧ 
    solutions.card = 2 := 
by
  sorry

end number_of_roots_l85_85439


namespace slope_of_line_best_fits_data_l85_85333

variable {α : Type*} [LinearOrderedField α]
variables (x1 x2 x3 x4 y1 y2 y3 y4 : α)

def mean (l : List α) : α :=
  l.sum / l.length

def x_values := [x1, x2, x3, x4]
def y_values := [y1, y2, y3, y4]

axiom h_distinct : x1 < x2 ∧ x2 < x3 ∧ x3 < x4
axiom h_unequal_diffs : ¬(x2 - x1 = x3 - x2 ∧ x3 - x2 = x4 - x3)

noncomputable def x_mean := mean x_values
noncomputable def y_mean := mean y_values

noncomputable def best_fit_slope : α :=
  (List.sum (List.map (λ (xi yi : α), (xi - x_mean) * (yi - y_mean)) x_values y_values)) / 
  (List.sum (List.map (λ xi : α, (xi - x_mean) ^ 2) x_values))

theorem slope_of_line_best_fits_data :
  best_fit_slope x1 x2 x3 x4 y1 y2 y3 y4 =
    (∑ i in [x1, x2, x3, x4].zip_with (λ xi, (xi - x_mean)) y_values, (xi.1 - x_mean) * (xi.2 - y_mean)) /
    (∑ i in x_values.map (λ xi, (xi - x_mean) ^ 2)) := 
by 
  sorry

end slope_of_line_best_fits_data_l85_85333


namespace ratio_of_areas_quadrilaterals_l85_85292

noncomputable def ratioOfAreas (A B C D K L N M : Point) 
  (h1 : isTrisection A B K) 
  (h2 : isTrisection A D L) 
  (h3 : isTrisection C B N) 
  (h4 : isTrisection C D M) 
  (quadABCD : isQuadrilateral A B C D) 
  (quadKLMN : isQuadrilateral K L M N) : Real :=
  let areaABCD := area A B C D
  let areaKLMN := area K L M N
  areaKLMN / areaABCD

theorem ratio_of_areas_quadrilaterals (A B C D K L N M : Point) :
  isTrisection A B K → isTrisection A D L → isTrisection C B N → isTrisection C D M → 
  isQuadrilateral A B C D → isQuadrilateral K L M N → ratioOfAreas A B C D K L N M = 4 / 9 :=
by
  intro h1 h2 h3 h4 quadABCD quadKLMN
  -- proof goes here
  sorry

end ratio_of_areas_quadrilaterals_l85_85292


namespace num_valid_colorings_l85_85923

namespace ColoringGrid

-- Definition of the grid and the constraint.
-- It's easier to represent with simply 9 nodes and adjacent constraints, however,
-- we will declare the conditions and result as discussed.

def Grid := Fin 3 × Fin 3
def Colors := Fin 2

-- Define adjacency relationship
def adjacent (a b : Grid) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

-- Condition stating no two adjacent squares can share the same color
def valid_coloring (f : Grid → Colors) : Prop :=
  ∀ a b : Grid, adjacent a b → f a ≠ f b

-- The main theorem stating the number of valid colorings
theorem num_valid_colorings : ∃ (n : ℕ), n = 2 ∧ ∀ (f : Grid → Colors), valid_coloring f → n = 2 :=
by sorry

end ColoringGrid

end num_valid_colorings_l85_85923


namespace unique_solution_x_l85_85455

theorem unique_solution_x (c d : ℝ) : 
  (∃ x y : ℝ, 4 * x - 7 + c = d * x + 2 * y + 4) -> 
  d ≠ 4 :=
by
  intros h
  cases h with x hxy
  sorry

end unique_solution_x_l85_85455


namespace probability_prime_and_multiple_of_11_l85_85607

noncomputable def is_prime (n : ℕ) : Prop := nat.prime n
def is_multiple_of_11 (n : ℕ) : Prop := n % 11 = 0

theorem probability_prime_and_multiple_of_11 :
  (1 / 100 : ℝ) = 
  let qualifying_numbers := {n | n ∈ finset.range 101 ∧ is_prime n ∧ is_multiple_of_11 n} in
  let number_of_qualifying := finset.card qualifying_numbers in
  (number_of_qualifying / 100 : ℝ) :=
by
  sorry

end probability_prime_and_multiple_of_11_l85_85607


namespace chord_length_of_circle_on_line_l85_85738

theorem chord_length_of_circle_on_line :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x = 0) ∧ (y = (sqrt 3) * x) → 2 * sqrt ((2^2) - (sqrt 3)^2) = 2 :=
by
  intros x y h
  sorry

end chord_length_of_circle_on_line_l85_85738


namespace fair_dice_roll_six_times_four_not_necessarily_appear_l85_85900

-- Define a fair dice
def fair_dice : Ω := {1, 2, 3, 4, 5, 6}
-- Define the event that the number 4 appears when a fair dice is rolled.
def event_four (ω : Ω) := ω = 4

-- Define the random variable for rolling a fair dice 6 times
def roll_six_times : list Ω := replicate 6 (count fair_dice)

-- Statement to prove that the number 4 does not necessarily appear
theorem fair_dice_roll_six_times_four_not_necessarily_appear :
  ¬ ∀ (ωlist : list Ω), ωlist ∈ roll_six_times → ∃ ω ∈ ωlist, event_four ω :=
sorry

end fair_dice_roll_six_times_four_not_necessarily_appear_l85_85900


namespace find_fraction_l85_85735

theorem find_fraction (x y : ℕ) (h₁ : x / (y + 1) = 1 / 2) (h₂ : (x + 1) / y = 1) : x = 2 ∧ y = 3 := by
  sorry

end find_fraction_l85_85735


namespace candy_sold_tuesday_correct_l85_85560

variable (pieces_sold_monday pieces_left_by_wednesday initial_candy total_pieces_sold : ℕ)
variable (pieces_sold_tuesday : ℕ)

-- Conditions
def initial_candy_amount := 80
def candy_sold_on_monday := 15
def candy_left_by_wednesday := 7

-- Total candy sold by Wednesday
def total_candy_sold_by_wednesday := initial_candy_amount - candy_left_by_wednesday

-- Candy sold on Tuesday
def candy_sold_on_tuesday : ℕ := total_candy_sold_by_wednesday - candy_sold_on_monday

-- Proof statement
theorem candy_sold_tuesday_correct : candy_sold_on_tuesday = 58 := sorry

end candy_sold_tuesday_correct_l85_85560


namespace shortest_path_from_vertex_to_center_of_non_adjacent_face_l85_85768

noncomputable def shortest_path_on_cube (edge_length : ℝ) : ℝ :=
  edge_length + (edge_length * Real.sqrt 2 / 2)

theorem shortest_path_from_vertex_to_center_of_non_adjacent_face :
  shortest_path_on_cube 1 = 1 + Real.sqrt 2 / 2 :=
by
  sorry

end shortest_path_from_vertex_to_center_of_non_adjacent_face_l85_85768


namespace probability_at_least_two_same_l85_85318

theorem probability_at_least_two_same :
  let total_outcomes := (8 ^ 4 : ℕ)
  let num_diff_outcomes := (8 * 7 * 6 * 5 : ℕ)
  let probability_diff := (num_diff_outcomes : ℝ) / total_outcomes
  let probability_at_least_two := 1 - probability_diff
  probability_at_least_two = (151 : ℝ) / 256 :=
by
  sorry

end probability_at_least_two_same_l85_85318


namespace P_not_divisible_by_3_P_divisible_by_3_list_l85_85812

noncomputable section

def harmonic_sum (n : ℕ) : ℚ :=
  ∑ i in Finset.range (n + 1) |i > 0, (1 / i : ℚ)

def relatively_prime (a b : ℤ) : Prop :=
  Int.gcd a b = 1

def P (n : ℕ) : ℤ :=
  let ⟨p, _⟩ := (harmonic_sum n).num_denom in p

def Q (n : ℕ) : ℤ :=
  let ⟨_, q⟩ := (harmonic_sum n).num_denom in q

theorem P_not_divisible_by_3 (n : ℕ) (h₁ : relatively_prime (P n) (Q n)) : P 67 % 3 ≠ 0 :=
sorry

theorem P_divisible_by_3_list (n : ℕ) (h₁ : relatively_prime (P n) (Q n)) :
  {n | P n % 3 = 0} = {2, 7, 22} :=
sorry

end P_not_divisible_by_3_P_divisible_by_3_list_l85_85812


namespace find_k_value_l85_85485

noncomputable def solve_for_k (k : ℝ) : Prop :=
  (∀ x y : ℝ, y = k * x → (x - real.sqrt 3) ^ 2 + y ^ 2 = 1) ↔ k = real.sqrt 2 / 2

theorem find_k_value :
  ∃ k : ℝ, k > 0 ∧ solve_for_k k :=
by
  sorry

end find_k_value_l85_85485


namespace middle_dimension_of_crate_l85_85362

-- Define the dimensions and conditions of the crate and the pillar.
structure Crate :=
  (d1 d2 d3 : ℝ) -- dimensions of the crate

structure Pillar :=
  (radius height : ℝ) -- dimensions of the pillar

-- The pillar has to fit upright in the crate.
def fits_in (c : Crate) (p : Pillar) : Prop :=
  (p.radius * 2 ≤ c.d1 ∨ p.radius * 2 ≤ c.d2 ∨ p.radius * 2 ≤ c.d3) ∧
  (p.height ≤ c.d1 ∨ p.height ≤ c.d2 ∨ p.height ≤ c.d3)

-- Given conditions
def given_crate : Crate := {d1 := 3, d2 := _, d3 := 12}
def given_pillar : Pillar := {radius := 3, height := _}

-- Statement to prove
theorem middle_dimension_of_crate :
  ∃ (d2 : ℝ), d2 = 6 ∧ fits_in given_crate { given_pillar with height := d2 } :=
sorry

end middle_dimension_of_crate_l85_85362


namespace solve_for_y_l85_85324

theorem solve_for_y (y : ℚ) : (40 / 60 = real.sqrt (y / 60)) → y = 80 / 3 :=
by
  intro h
  sorry

end solve_for_y_l85_85324


namespace correct_division_algorithm_l85_85702

theorem correct_division_algorithm : (-8 : ℤ) / (-4 : ℤ) = (8 : ℤ) / (4 : ℤ) := 
by 
  sorry

end correct_division_algorithm_l85_85702


namespace car_x_travel_distance_l85_85412

theorem car_x_travel_distance :
  ∀ (t : ℕ), (speed_X speed_Y: ℝ),
    speed_X = 35 →
    speed_Y = 41 →
    ∀ (d_X d_Y: ℝ),
      d_X = speed_X * t →
      d_Y = speed_Y * t →
      35 * 7 = 245 :=
by {
  sorry
}

end car_x_travel_distance_l85_85412


namespace max_k_l85_85198

-- Definitions and conditions
def original_number (A B : ℕ) : ℕ := 10 * A + B
def new_number (A C B : ℕ) : ℕ := 100 * A + 10 * C + B

theorem max_k (A C B k : ℕ) (hA : A ≠ 0) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : 0 ≤ B ∧ B ≤ 9) (h3: 0 ≤ C ∧ C ≤ 9) :
  ((original_number A B) * k = (new_number A C B)) → 
  (∀ (A: ℕ), 1 ≤ k) → 
  k ≤ 19 :=
by
  sorry

end max_k_l85_85198


namespace polynomial_expansion_has_5_nonzero_terms_l85_85790

noncomputable def polynomial_expansion : Nat :=
  let p1 := (x - 3) * (3 * x ^ 3 + 2 * x ^ 2 - 4 * x + 1)
  let p2 := 4 * (x ^ 4 + x ^ 3 - 2 * x ^ 2 + x)
  let p3 := -5 * (x ^ 3 - 3 * x + 1)
  let result := p1 + p2 + p3
  result.coeffs.count_nonzero_terms  -- Assuming this function returns the count of nonzero terms

theorem polynomial_expansion_has_5_nonzero_terms : polynomial_expansion = 5 := sorry

end polynomial_expansion_has_5_nonzero_terms_l85_85790


namespace solve_for_y_l85_85326

theorem solve_for_y (y : ℝ) :
  (40 / 60 = real.sqrt (y / 60)) → y = 80 / 3 :=
by
  intro h
  -- proof goes here
  sorry

end solve_for_y_l85_85326


namespace max_binomial_coeff_term_inequality_f2x_f2_exists_a_l85_85938

section proofs

variables (n : ℕ) (x : ℕ) (n_gt_one : 1 < n)
open_locale big_operators

noncomputable def f (x : ℕ) : ℝ := (1 + 1 / n.to_real)^x

-- Statement for (Ⅰ)
theorem max_binomial_coeff_term :
  ∃ (t : ℝ), t = (nat.choose 6 3) * (1 ^ 3) * ((1 / (n.to_real)) ^ 3) ∧ 
             t = 20 / (n.to_real ^ 3) :=
sorry

-- Statement for (Ⅱ)
theorem inequality_f2x_f2 :
  ∀ (x : ℝ), (f (2 * int.of_nat x) + f 2) / 2 > (deriv f) x :=
sorry

-- Statement for (Ⅲ)
theorem exists_a (a : ℕ) :
  2 = a ∧ ∀ (n : ℕ), (1 < n) → (a * n < ∑ k in finset.range (n + 1), 1 + 1 / k.to_real < (a + 1) * n) :=
sorry

end proofs

end max_binomial_coeff_term_inequality_f2x_f2_exists_a_l85_85938


namespace express_y_in_terms_of_x_l85_85448

variable (x y : ℝ)

theorem express_y_in_terms_of_x (h : x + y = -1) : y = -1 - x := 
by 
  sorry

end express_y_in_terms_of_x_l85_85448


namespace determine_vector_p_l85_85134

structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def vector_operation (m p : Vector2D) : Vector2D :=
  Vector2D.mk (m.x * p.x + m.y * p.y) (m.x * p.y + m.y * p.x)

theorem determine_vector_p (p : Vector2D) : 
  (∀ (m : Vector2D), vector_operation m p = m) → p = Vector2D.mk 1 0 :=
by
  sorry

end determine_vector_p_l85_85134


namespace pirate_treasure_probability_l85_85374

theorem pirate_treasure_probability :
  let num_islands := 8
  let prob_treasure_no_traps := 1 / 3
  let prob_traps_no_treasure := 1 / 6
  let prob_neither := 1 / 2 
  (nat.choose 8 4 * (1/3)^4 * (1/2)^4) = 35 / 648 :=
by
  sorry

end pirate_treasure_probability_l85_85374


namespace jace_total_distance_l85_85199

noncomputable def total_distance (s1 s2 s3 s4 s5 : ℝ) (t1 t2 t3 t4 t5 : ℝ) : ℝ :=
  s1 * t1 + s2 * t2 + s3 * t3 + s4 * t4 + s5 * t5

theorem jace_total_distance :
  total_distance 50 65 60 75 55 3 4.5 2.75 1.8333 2.6667 = 891.67 := by
  sorry

end jace_total_distance_l85_85199


namespace inequality_example_l85_85811

theorem inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a ^ 2 + 8 * b * c)) + (b / Real.sqrt (b ^ 2 + 8 * c * a)) + (c / Real.sqrt (c ^ 2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end inequality_example_l85_85811


namespace product_of_positive_n_satisfying_quadratic_is_304_l85_85443

theorem product_of_positive_n_satisfying_quadratic_is_304 :
  ∀ n : ℕ, (∃ p : ℕ, prime p ∧ n^2 - 35 * n + 306 = p) → ∃ N : ℕ, N = 304 :=
by
  sorry

end product_of_positive_n_satisfying_quadratic_is_304_l85_85443


namespace molecular_formula_is_Al2O3_l85_85315

-- Define the atomic weights
def atomicWeightAl : ℝ := 26.98
def atomicWeightO : ℝ := 16.00

-- Define the molecular weight of the compound
def molecularWeight : ℝ := 102.00

-- Define the number of oxygen atoms
def numOxygenAtoms : ℕ := 3

-- Define the weight of oxygen atoms in the compound
def weightOxygenAtoms : ℝ := numOxygenAtoms * atomicWeightO

-- Define the weight of aluminum in the compound
def weightAluminum : ℝ := molecularWeight - weightOxygenAtoms

-- Define the number of aluminum atoms approximately
def numAluminumAtoms : ℝ := weightAluminum / atomicWeightAl

-- Statement of the theorem
theorem molecular_formula_is_Al2O3 :
  approximate (numAluminumAtoms, 2) ∧ molecularWeight = 102 :=
by
  sorry

end molecular_formula_is_Al2O3_l85_85315


namespace other_asymptote_eq_l85_85974

-- Definitions based on given conditions
def asymptote1 : (ℝ → ℝ) := λ x, 2 * x + 1
def focus_x : ℝ := 4

-- Prove that given the first asymptote and the x-coordinate of the foci,
-- the equation of the other asymptote is y = -2x + 17.
theorem other_asymptote_eq :
  ∃ (m b : ℝ), (∀ x : ℝ, (y = 2 * x + 1 → y = -2 * x + 17)) := 
sorry

end other_asymptote_eq_l85_85974


namespace sum_converges_to_zero_l85_85230

noncomputable def G : ℕ → ℝ
| 0        := 1
| 1        := 2
| (n + 2)  := 3 * G (n + 1) - (1/2) * G n

theorem sum_converges_to_zero : (∑' n, 1 / 2 ^ (G n)) = 0 :=
sorry

end sum_converges_to_zero_l85_85230


namespace sum_of_roots_l85_85844

theorem sum_of_roots (x1 x2 : ℝ) (h1 : x1^2 + 5*x1 - 3 = 0) (h2 : x2^2 + 5*x2 - 3 = 0) (h3 : x1 ≠ x2) :
  x1 + x2 = -5 :=
sorry

end sum_of_roots_l85_85844


namespace problem_solution_l85_85151

theorem problem_solution (b x : ℝ) (hb : b > 1) (hx : x > 0) :
  (4 * x)^(Real.log b 4) - (5 * x)^(Real.log b 5) = 0 → x = 1 / 5 :=
by
  -- proof goes here
  sorry

end problem_solution_l85_85151


namespace trigonometric_identity_simplification_l85_85620

theorem trigonometric_identity_simplification (theta : ℝ) :
  cos (2 * π - theta) * cos (2 * theta) + sin (theta) * sin (π + 2 * theta) = cos (3 * theta) :=
by
  -- Add the proof here
  sorry

end trigonometric_identity_simplification_l85_85620


namespace joe_average_speed_l85_85201

noncomputable def average_speed (total_distance total_time : ℝ) : ℝ :=
  total_distance / total_time

theorem joe_average_speed :
  let distance1 := 420
  let speed1 := 60
  let distance2 := 120
  let speed2 := 40
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  average_speed total_distance total_time = 54 := by
sorry

end joe_average_speed_l85_85201


namespace tiles_needed_l85_85383
open Function

theorem tiles_needed (room_tiles_8cm : ℕ) (side_length_8cm : ℕ) (side_length_6cm : ℕ) : 
    room_tiles_8cm = 90 → side_length_8cm = 8 → side_length_6cm = 6 → 
    ∃ room_tiles_6cm : ℕ, room_tiles_6cm = 160 :=
by
  intros h1 h2 h3
  use 160
  have eq1 : 6 * 6 * 160 = 8 * 8 * 90 := by sorry
  exact eq1

end tiles_needed_l85_85383


namespace correct_statements_l85_85841

section

variables {m n : Line} {α β γ : Plane}

-- Conditions directly appearing in the problem:
def statement_1 : Prop := (α ⊥ β) ∧ (α ∩ β = m) ∧ (n ⊥ m) → (n ⊥ α ∨ n ⊥ β)
def statement_2 : Prop := (α ‖ β) ∧ (α ∩ γ = m) ∧ (β ∩ γ = n) → (m ‖ n)
def statement_3 : Prop := ¬(m ⊥ α) → ¬(∀ l ∈ (λ x, x.1 = α), m ⊥ l)
def statement_4 : Prop := (α ∩ β = m) ∧ (n ‖ m) ∧ ¬(n ⊆ α) ∧ ¬(n ⊆ β) → (n ‖ α ∧ n ‖ β)

-- Correct answers:
theorem correct_statements : (statement_2 ∧ statement_4) :=
by {
    sorry,
}

end

end correct_statements_l85_85841


namespace two_m_plus_three_n_l85_85801

open Nat

theorem two_m_plus_three_n (x y : ℕ) 
  (h₁ : log (10 : ℝ) x + 2 * log (10 : ℝ) (gcd x y) = 100) 
  (h₂ : log (10 : ℝ) y + 2 * log (10 : ℝ) (lcm x y) = 900) :
  let m := x.factorization.support.card
  let n := y.factorization.support.card
  2 * m + 3 * n = 1980 :=
sorry

end two_m_plus_three_n_l85_85801


namespace base4_to_base10_conversion_l85_85035

theorem base4_to_base10_conversion : (2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582) :=
by {
  -- The proof is omitted
  sorry
}

end base4_to_base10_conversion_l85_85035


namespace sin_double_angle_l85_85073

theorem sin_double_angle {α : ℝ} (hα : 0 < α ∧ α < π) (h : tan (π / 4 - α) = 1 / 3) : 
  sin (2 * α) = 4 / 5 :=
by
  sorry

end sin_double_angle_l85_85073


namespace fraction_difference_l85_85776

theorem fraction_difference :
  let x : ℚ := 8 / 11 in
  let y : ℚ := 726 / 1000 in
  x - y = 14 / 11000 :=
by
  sorry

end fraction_difference_l85_85776


namespace six_digit_permutation_reverse_div_by_11_l85_85572

theorem six_digit_permutation_reverse_div_by_11 
  (a b c : ℕ)
  (h_a : 1 ≤ a ∧ a ≤ 9)
  (h_b : 0 ≤ b ∧ b ≤ 9)
  (h_c : 0 ≤ c ∧ c ≤ 9)
  (X : ℕ)
  (h_X : X = 100001 * a + 10010 * b + 1100 * c) :
  11 ∣ X :=
by 
  sorry

end six_digit_permutation_reverse_div_by_11_l85_85572


namespace number_of_cookies_per_bag_l85_85332

def cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) : Prop :=
  (total_cookies / num_bags) = 11

theorem number_of_cookies_per_bag (h1 : total_cookies = 33) (h2 : num_bags = 3) : cookies_per_bag 33 3 :=
by
  rw [h1, h2]
  simp
  sorry

end number_of_cookies_per_bag_l85_85332


namespace find_BD_in_triangle_l85_85537

theorem find_BD_in_triangle :
  ∀ (A B C D : Type) [EuclideanGeometry A] (a b c : A),
    distance b c = 24 ∧ distance a b = 26 ∧ distance a c = 26 ∧ (midpoint a c = d) →
    distance b d = sqrt 457 :=
by 
  intros A B C D h a b c hb hc ha mid D_eq_d,
  sorry

end find_BD_in_triangle_l85_85537


namespace reciprocal_power_l85_85895

theorem reciprocal_power (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 :=
by sorry

end reciprocal_power_l85_85895


namespace MinkyungHeight_is_correct_l85_85135

noncomputable def HaeunHeight : ℝ := 1.56
noncomputable def NayeonHeight : ℝ := HaeunHeight - 0.14
noncomputable def MinkyungHeight : ℝ := NayeonHeight + 0.27

theorem MinkyungHeight_is_correct : MinkyungHeight = 1.69 :=
by
  sorry

end MinkyungHeight_is_correct_l85_85135


namespace arccos_range_l85_85096

theorem arccos_range (a : ℝ) (x : ℝ) (h₀ : x = Real.sin a) 
  (h₁ : -Real.pi / 4 ≤ a ∧ a ≤ 3 * Real.pi / 4) :
  ∀ y, y = Real.arccos x → 0 ≤ y ∧ y ≤ 3 * Real.pi / 4 := 
sorry

end arccos_range_l85_85096


namespace inequality_am_gm_holds_l85_85476

theorem inequality_am_gm_holds 
    (a b c : ℝ) 
    (ha : a > 0) 
    (hb : b > 0) 
    (hc : c > 0) 
    (h : a^3 + b^3 = c^3) : 
  a^2 + b^2 - c^2 > 6 * (c - a) * (c - b) := 
sorry

end inequality_am_gm_holds_l85_85476


namespace find_fraction_l85_85352

-- Define the weight of the bag and the fraction
def weight : ℝ := 12
def fraction_weight (x : ℝ) : ℝ := weight / x

-- Theorem statement
theorem find_fraction (x : ℝ) (h : fraction_weight x = 12): x = 1 := 
by
  sorry

end find_fraction_l85_85352


namespace seq_an_geometric_seq_bn_arithmetic_sum_an_bn_seq_l85_85463

noncomputable def geometric_seq_a : ℕ → ℚ
| 1 => 1
| 2 => 2
| (n+1) => 2 * geometric_seq_a n

def arith_cond (a b c : ℚ) := a + c = 2 * (b + 1)

def S (n : ℕ) : ℕ := n^2 + n

noncomputable def seq_b (n : ℕ) : ℕ :=
if n = 1 then 2 else 2 * n

theorem seq_an_geometric :
  geometric_seq_a 2 = 2 ∧
  arith_cond (geometric_seq_a 2) (geometric_seq_a 3) (geometric_seq_a 4) ∧
  (∀ n, seq_b (n+1) = S (n+1) - S n) →
  (∀ n, geometric_seq_a n = 2^(n-1)) :=
sorry

theorem seq_bn_arithmetic :
  (∀ n, seq_b (n+1) = S (n+1) - S n) ∧
  seq_b 1 = 2 →
  (∀ n, seq_b n = 2 * n) :=
sorry

theorem sum_an_bn_seq :
  (∀ n, geometric_seq_a n = 2^(n-1)) ∧
  (∀ n, seq_b n = 2 * n) →
  (∀ n, ∑ i in Finset.range n, geometric_seq_a (i + 1) + 
    (4 / (seq_b (i + 1) * seq_b (i + 2))) = 2^n - 1 / (n+1)) :=
sorry

end seq_an_geometric_seq_bn_arithmetic_sum_an_bn_seq_l85_85463


namespace correct_propositions_l85_85819

namespace MathProof

-- Definitions of planes and lines
variables {Plane : Type} {Line : Type} 
variable (α β : Plane)
variable (m n : Line)

-- Definitions of geometric predicates
variables (perpendicular : Line → Plane → Prop)
variables (subset : Line → Plane → Prop)
variables (parallel : Line → Plane → Prop)

-- Proposition P1
def P1 := ∀ (m : Line) (α β : Plane), perpendicular m α → subset m β → perpendicular α β

-- Proposition P2
def P2 := ∀ (m n : Line) (α : Plane), perpendicular m n → perpendicular m α → parallel n α

-- Proposition P3
def P3 := ∀ (m : Line) (α β : Plane), parallel m α → perpendicular α β → perpendicular m β

-- Proposition P4
def P4 := ∀ (m n : Line) (α β : Plane), 
  (α ∩ β = m) → 
  parallel n m → 
  ¬ subset n α → 
  ¬ subset n β → 
  parallel n α ∧ parallel n β

-- Main theorem statement
theorem correct_propositions :
  ∀ (α β : Plane) (m n : Line) 
  (perpendicular : Line → Plane → Prop)
  (subset : Line → Plane → Prop)
  (parallel : Line → Plane → Prop),
  P1 perpendicular subset perpendicular α β m → 
  ¬ P2 perpendicular parallel α m n → 
  ¬ P3 parallel perpendicular m α β → 
  P4 parallel subset α β m n →
  (P1 perpendicular subset perpendicular ∧ P4 parallel subset α β m n)
:= 
by {
  intros,
  sorry
}

end MathProof

end correct_propositions_l85_85819


namespace solve_for_y_l85_85323

theorem solve_for_y (y : ℚ) : (40 / 60 = real.sqrt (y / 60)) → y = 80 / 3 :=
by
  intro h
  sorry

end solve_for_y_l85_85323


namespace tangent_line_at_a_eq_1_monotonic_intervals_range_of_a_l85_85865

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x + (a - 2) / x + 2 - 2 * a

theorem tangent_line_at_a_eq_1 :
  f 1 2 = 3 / 2 ∧ (deriv (f 1) 2 = 5 / 4) →
  ∃ (y : ℝ), (y - (3 / 2)) = (5 / 4) * (2 - 2) ∧ 5 * 2 - 4 * y - 4 = 0 :=
sorry

theorem monotonic_intervals (a : ℝ) (h : a > 0) :
  (0 < a ∧ a ≤ 2 → 
    ∀ x, (x < 0 ∨ x > 0 → deriv (f a) x ≥ 0)) ∧ 
  (a > 2 → 
    ∀ x, (x < -sqrt ((a-2)/a) ∨ x > sqrt ((a-2)/a) ∧ deriv (f a) x > 0) ∧ 
         (-sqrt ((a-2)/a) < x ∧ x < 0 ∨ 0 < x ∧ x < sqrt ((a-2)/a) ∧ deriv (f a) x < 0)) := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x → f a x ≥ 2 * ln x) ↔ (1 ≤ a) :=
sorry

end tangent_line_at_a_eq_1_monotonic_intervals_range_of_a_l85_85865


namespace minimize_J_l85_85156

noncomputable def H (p q : ℝ) : ℝ := -3 * p * q + 4 * p * (1 - q) + 2 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) : ℝ := max (H p 0) (H p 1)

theorem minimize_J : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ (∀ p' : ℝ, 0 ≤ p' ∧ p' ≤ 1 → J p ≤ J p') ∧ p = 1 / 2 :=
by
  sorry

end minimize_J_l85_85156


namespace percentage_of_democrats_l85_85539

variable (D R : ℝ)

theorem percentage_of_democrats (h1 : D + R = 100) (h2 : 0.75 * D + 0.20 * R = 53) :
  D = 60 :=
by
  sorry

end percentage_of_democrats_l85_85539


namespace sqrt_8_consecutive_integers_l85_85149

theorem sqrt_8_consecutive_integers (a b : ℤ) (h_consec: b = a + 1) (h : (a : ℝ) < Real.sqrt 8 ∧ Real.sqrt 8 < (b : ℝ)) : b^a = 9 :=
by
  sorry

end sqrt_8_consecutive_integers_l85_85149


namespace remainder_of_x120_l85_85097

theorem remainder_of_x120 :
  ∃ (a b : ℝ), (∀ x : ℝ, x ^ 120 % (x ^ 2 - 4 * x + 3) = a * x + b) ∧
    (3 * a + b = 3 ^ 120) ∧
    (a + b = 1) ∧
    (a = (3 ^ 120 - 1) / 2) ∧
    (b = (3 - 3 ^ 120) / 2) :=
by
  use [(3 ^ 120 - 1) / 2, (3 - 3 ^ 120) / 2]
  split
  sorry
  split
  sorry
  split
  sorry
  split
  sorry

end remainder_of_x120_l85_85097


namespace one_odd_one_even_a_negative_same_sign_roots_l85_85919

-- Define the necessary elements and conditions
variable (a : ℝ)
variable (x1 x2 : ℤ)
variable (h_eq : (x1 : ℝ) + (x2 : ℝ) = 3)
variable (h_roots: x1 * x2 = a + 4)

-- Question 1: Prove that one of these integer roots is odd, and the other is even
theorem one_odd_one_even (h_int : (∃ x1 x2 : ℤ, x1^2 - 3 * x1 + a + 4 = 0) ) :
  ((x1 % 2 = 1 ∧ x2 % 2 = 0) ∨ (x1 % 2 = 0 ∧ x2 % 2 = 1)) := sorry

-- Question 2: Prove that a is a negative number
theorem a_negative : a ≤ -7 / 4 := sorry

-- Question 3: Find the value of a and these two roots when they have the same sign
theorem same_sign_roots (h_same_sign : signum x1 = signum x2) : (a = -2 ∧ x1 = 1 ∧ x2 = 2) := sorry

end one_odd_one_even_a_negative_same_sign_roots_l85_85919


namespace prop1_prop2_l85_85937

universe u
variables {R : Type u} [Real R]
variables {f g h : R → R} {T : R}

-- Condition for proposition ①
def periodic (f : R → R) (T : R) := ∀ x, f(x) = f(x + T)

-- Condition for proposition ②
def increasing (f : R → R) := ∀ x y, x < y → f(x) ≤ f(y)

-- Proposition ①: periodic functions
theorem prop1 (hfg : periodic (λ x, f x + g x) T) (hfh : periodic (λ x, f x + h x) T) (hgh : periodic (λ x, g x + h x) T) :
  periodic f T ∧ periodic g T ∧ periodic h T := sorry

-- Proposition ②: increasing functions
theorem prop2 (hfg_inc : increasing (λ x, f x + g x)) (hfh_inc : increasing (λ x, f x + h x)) (hgh_inc : increasing (λ x, g x + h x)) :
  ¬ (increasing g) := sorry

end prop1_prop2_l85_85937


namespace coefficient_x4_expansion_l85_85163

noncomputable def coefficient_of_x4_in_expansion : ℤ :=
  let n := 6
  let term := λ r, ((-2)^r) * Nat.choose n r
  term 1

theorem coefficient_x4_expansion : coefficient_of_x4_in_expansion = -12 := by
  sorry

end coefficient_x4_expansion_l85_85163


namespace sum_of_thousands_and_units_digit_of_product_l85_85654

def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10
def units_digit (n : ℕ) : ℕ := n % 10

-- Define the two 102-digit numbers
def num1 : ℕ := 404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404
def num2 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

-- Define their product
def product : ℕ := num1 * num2

-- Define the conditions for the problem
def A := thousands_digit product
def B := units_digit product

-- Define the problem statement
theorem sum_of_thousands_and_units_digit_of_product : A + B = 13 := 
by
  sorry

end sum_of_thousands_and_units_digit_of_product_l85_85654


namespace table_length_is_77_l85_85748

theorem table_length_is_77 :
  ∀ (x : ℝ), 
  (∀ (y : ℝ), 
  (x >= 0) ∧ 
  (y = 80) ∧ 
  (∀ (w : ℝ), (w = 8)) ∧ 
  (∀ (h : ℝ), (h = 5)) ∧ 
  (∀ (i : ℕ), 
    (i₀ := 0) ∧ 
    (j₀ := 0) ∧ 
    (∀ i, w + i * 1 = y) ∧ 
    (∀ i, h + i * 1 = x) ∧ 
    (i = 72))) → 
  (x = 77) :=
by
  intros
  sorry

end table_length_is_77_l85_85748


namespace probability_of_condition_l85_85266

def chosen_set : Set ℕ := {n | 1 ≤ n ∧ n ≤ 20}

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_divisible_by_five (a b : ℕ) : Prop := (a + b) % 5 = 0

theorem probability_of_condition :
  let total_choices := (Finset.card (Finset.powersetLen 2 (Finset.range 21))) -- total combinations
  let odd_numbers := {n | n ∈ chosen_set ∧ is_odd n}
  let odd_pairs := (Finset.powersetLen 2 (Finset.filter odd_numbers.toSet (Finset.range 21))) 
  let valid_pairs := Finset.filter (λ p, sum_divisible_by_five p.1 p.2) odd_pairs
  (Finset.card valid_pairs : ℚ) / (Finset.card total_choices) = 9 / 95 :=
by
  sorry

end probability_of_condition_l85_85266


namespace cos_pi_minus_theta_l85_85491

-- We define the conditions given in the problem
def point : ℝ × ℝ := (4, -3)

-- We extract the x and y coordinates from the point
def x : ℝ := point.1
def y : ℝ := point.2

-- Define the radius r using the Pythagorean theorem
noncomputable def r : ℝ := real.sqrt (x^2 + y^2)

-- Define cos(theta) using the definition of cosine in a right triangle
noncomputable def cos_theta : ℝ := x / r

-- State the final goal using the trigonometric identity for cos(pi - theta)
theorem cos_pi_minus_theta : cos (π - θ) = -4 / 5 :=
by 
  sorry -- proofs and further derivations are omitted

end cos_pi_minus_theta_l85_85491


namespace nancy_tortilla_chips_l85_85968

theorem nancy_tortilla_chips :
  ∀ (total_chips chips_brother chips_herself chips_sister : ℕ),
    total_chips = 22 →
    chips_brother = 7 →
    chips_herself = 10 →
    chips_sister = total_chips - chips_brother - chips_herself →
    chips_sister = 5 :=
by
  intros total_chips chips_brother chips_herself chips_sister
  intro h_total h_brother h_herself h_sister
  rw [h_total, h_brother, h_herself] at h_sister
  simp at h_sister
  assumption

end nancy_tortilla_chips_l85_85968


namespace projection_of_a_onto_b_l85_85130

-- Define the vectors a and b
def a := (Real.sqrt 3, 1 : ℝ)
def b := (1, 0 : ℝ)

-- Define the dot product 
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the projection of vector a onto vector b
def projection (a b : ℝ × ℝ) : ℝ := (dot_product a b) / (magnitude b)

-- The theorem to be proved
theorem projection_of_a_onto_b : projection a b = Real.sqrt 3 :=
by
  sorry

end projection_of_a_onto_b_l85_85130


namespace comparison_abc_l85_85460

def ln (x : ℝ) : ℝ := Real.log x

noncomputable def a := ln 2 / 2
noncomputable def b := ln π / π
noncomputable def c := ln 5 / 5

theorem comparison_abc : b > a ∧ a > c := by
  sorry

end comparison_abc_l85_85460


namespace lcm_gcd_product_l85_85696

theorem lcm_gcd_product (a b : ℕ) (ha : a = 36) (hb : b = 60) : 
  Nat.lcm a b * Nat.gcd a b = 2160 :=
by
  rw [ha, hb]
  sorry

end lcm_gcd_product_l85_85696


namespace barry_count_is_24_l85_85170

noncomputable def number_of_people_named_barry 
  (total_nice_people : ℕ)
  (nice_kevin_fraction nice_julie_fraction nice_joe_fraction : ℚ)
  (total_kevin total_julie total_joe : ℕ) : ℕ :=
total_nice_people - (nice_kevin_fraction * total_kevin + nice_julie_fraction * total_julie + nice_joe_fraction * total_joe).to_nat

theorem barry_count_is_24 (total_nice_people : ℕ)
  (nice_kevin_fraction nice_julie_fraction nice_joe_fraction : ℚ)
  (total_kevin total_julie total_joe : ℕ)
  (nice_people_count : ℕ) :
  nice_kevin_fraction = 1/2 ∧ 
  nice_julie_fraction = 3/4 ∧ 
  nice_joe_fraction = 1/10 ∧ 
  total_nice_people = 99 ∧ 
  total_kevin = 20 ∧ 
  total_julie = 80 ∧ 
  total_joe = 50 ∧ 
  nice_people_count = 24 →
  number_of_people_named_barry total_nice_people nice_kevin_fraction nice_julie_fraction nice_joe_fraction total_kevin total_julie total_joe = nice_people_count :=
by
  intros 
  simp [number_of_people_named_barry]
  sorry

end barry_count_is_24_l85_85170


namespace hyperparallelepiped_diagonal_sum_l85_85723

variable {V : Type*} [InnerProductSpace ℝ V]

theorem hyperparallelepiped_diagonal_sum 
  (u v w x : V) :
  (∥u + v + w∥^2 + ∥v + w + x∥^2 + ∥u + w + x∥^2 + ∥u + v + x∥^2) /
  (∥u∥^2 + ∥v∥^2 + ∥w∥^2 + ∥x∥^2) = 4 :=
by
  sorry

end hyperparallelepiped_diagonal_sum_l85_85723


namespace smoking_lung_cancer_relation_l85_85913

theorem smoking_lung_cancer_relation (study_valid: Prop) (confidence_level: Real) (prob_mistake: Real) :
  study_valid → confidence_level = 0.99 → prob_mistake ≤ 0.01 →
  (∃ (correct_option: Prop), correct_option = "Among 100 smokers, it's possible that not a single person has lung cancer") :=
by {
  intros study_valid confidence_level prob_mistake h1 h2 h3,
  existsi ("Among 100 smokers, it's possible that not a single person has lung cancer" : Prop),
  sorry
}

end smoking_lung_cancer_relation_l85_85913


namespace kendall_total_change_l85_85202

-- Definition of values of coins
def value_of_quarters (q : ℕ) : ℝ := q * 0.25
def value_of_dimes (d : ℕ) : ℝ := d * 0.10
def value_of_nickels (n : ℕ) : ℝ := n * 0.05

-- Conditions
def quarters := 10
def dimes := 12
def nickels := 6

-- Theorem statement
theorem kendall_total_change : 
  value_of_quarters quarters + value_of_dimes dimes + value_of_nickels nickels = 4.00 :=
by
  sorry

end kendall_total_change_l85_85202


namespace smallest_n_exists_unique_k_l85_85319

/- The smallest positive integer n for which there exists
   a unique integer k such that 9/16 < n / (n + k) < 7/12 is n = 1. -/

theorem smallest_n_exists_unique_k :
  ∃! (n : ℕ), n > 0 ∧ (∃! (k : ℤ), (9 : ℚ)/16 < (n : ℤ)/(n + k) ∧ (n : ℤ)/(n + k) < (7 : ℚ)/12) :=
sorry

end smallest_n_exists_unique_k_l85_85319


namespace lillian_cupcakes_l85_85952

theorem lillian_cupcakes (home_sugar : ℕ) (bags : ℕ) (sugar_per_bag : ℕ) (batter_sugar_per_dozen : ℕ) (frosting_sugar_per_dozen : ℕ) :
  home_sugar = 3 → bags = 2 → sugar_per_bag = 6 → batter_sugar_per_dozen = 1 → frosting_sugar_per_dozen = 2 →
  ((home_sugar + bags * sugar_per_bag) / (batter_sugar_per_dozen + frosting_sugar_per_dozen)) = 5 := 
by 
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end lillian_cupcakes_l85_85952


namespace part_I_part_II_l85_85219

noncomputable def f (a x : ℝ) : ℝ := |x - 1| + a * |x - 2|

theorem part_I (a : ℝ) (h_min : ∃ m, ∀ x, f a x ≥ m) : -1 ≤ a ∧ a ≤ 1 :=
sorry

theorem part_II (a : ℝ) (h_bound : ∀ x, f a x ≥ 1/2) : a = 1/3 :=
sorry

end part_I_part_II_l85_85219


namespace lollipop_ratio_l85_85983

/-- Sarah bought 12 lollipops for a total of 3 dollars. Julie gave Sarah 75 cents to pay for the shared lollipops.
Prove that the ratio of the number of lollipops shared to the total number of lollipops bought is 1:4. -/
theorem lollipop_ratio
  (h1 : 12 = lollipops_bought)
  (h2 : 3 = total_cost_dollars)
  (h3 : 75 = amount_paid_cents)
  : (75 / 25) / lollipops_bought = 1/4 :=
sorry

end lollipop_ratio_l85_85983


namespace cycle_Xk_l85_85585
-- Define the conditions
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

def X0 (n : ℕ) : Vector ℕ n :=
  if h : n > 1 then
    ⟨[1] ++ List.replicate (n - 2) 0 ++ [1], sorry⟩
  else 
    ⟨[], sorry⟩ -- n must be > 1

def x_next (xi xi_plus1 : ℕ) : ℕ :=
  if xi = xi_plus1 then 0 else 1

def Xk (n k : ℕ) (Xk_minus1 : Vector ℕ n) : Vector ℕ n :=
  ⟨List.ofFn (λ i, x_next (Xk_minus1.nth i) (Xk_minus1.nth ((i + 1) % n))), sorry⟩

-- Prove that n divides m given X_m = X_0
theorem cycle_Xk (n m : ℕ) (hn_odd : is_odd n) (hn_gt1 : n > 1) :
  let X0n := X0 n in
  let Xm := Nat.iterate m (Xk n) X0n in
  Xm = X0n → n ∣ m :=
by
  sorry

end cycle_Xk_l85_85585


namespace concyclic_MKFL_l85_85330

open EuclideanGeometry

-- Definitions and conditions translated into Lean
variable {A B C M L K F : Point}
variable [circle (A, B)] [circle (B, C)] -- Circles with diameters AB and BC

noncomputable def angle_MBA_eq_LBC := ∠(M, B, A) = ∠(L, B, C)
noncomputable def BK_eq_BC := (B, K) = (B, C)
noncomputable def BF_eq_AB := (B, F) = (A, B)
noncomputable def collinear_ABC := collinear A B C

-- Statement of the proof problem
theorem concyclic_MKFL :
  collinear_ABC → angle_MBA_eq_LBC → BK_eq_BC → BF_eq_AB → cyclic_quadrilateral M K F L :=
by
  sorry

end concyclic_MKFL_l85_85330


namespace highest_possible_characteristic_l85_85290

noncomputable def characteristic (n : ℕ) (A : ℕ → ℕ → ℕ) : ℚ :=
  let fractions := { (A i j) / (A k l) | i j k l : ℕ, i = k ∨ j = l, (A i j) > (A k l) }
  fractions.min' sorry

theorem highest_possible_characteristic (n : ℕ) (h : 2 ≤ n) (A : ℕ → ℕ → ℕ) :
  characteristic n A ≤ (n + 1) / n :=
  sorry

end highest_possible_characteristic_l85_85290


namespace quadratic_inequality_roots_a_eq_neg1_quadratic_inequality_for_all_real_a_range_l85_85868

-- Proof Problem (1)
theorem quadratic_inequality_roots_a_eq_neg1
  (a : ℝ)
  (h : ∀ x, (-1 < x ∧ x < 3) → ax^2 - 2 * a * x + 3 > 0) :
  a = -1 :=
sorry

-- Proof Problem (2)
theorem quadratic_inequality_for_all_real_a_range
  (a : ℝ)
  (h : ∀ x, ax^2 - 2 * a * x + 3 > 0) :
  0 ≤ a ∧ a < 3 :=
sorry

end quadratic_inequality_roots_a_eq_neg1_quadratic_inequality_for_all_real_a_range_l85_85868


namespace laptop_sticker_price_l85_85511

open Real

theorem laptop_sticker_price (x : ℝ) 
  (h1 : ∀ x, storeA x = 0.80 * x - 120)
  (h2 : ∀ x, storeB x = 0.70 * x) 
  (h3 : ∀ x, storeB x = storeA x + 18) : x = 1020 := 
sorry


end laptop_sticker_price_l85_85511


namespace num_real_roots_of_g_eq_0_l85_85223

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else real.sqrt x - 1

def g (x : ℝ) : ℝ := f (f x)

theorem num_real_roots_of_g_eq_0 : 
  ({x : ℝ | g x = 0}.to_finset.card = 3) :=
by
  sorry

end num_real_roots_of_g_eq_0_l85_85223


namespace circle_and_parabola_no_intersection_l85_85851

theorem circle_and_parabola_no_intersection (m : ℝ) (h : m ≠ 0) :
  (m > 0 ∨ m < -4) ↔
  ∀ x y : ℝ, (x^2 + y^2 - 4 * x = 0) → (y^2 = 4 * m * x) → x ≠ -m := 
sorry

end circle_and_parabola_no_intersection_l85_85851


namespace ratio_of_two_numbers_l85_85646

-- Definitions of gcd and lcm for two number assertions as per given conditions:
def gcd (a b : Nat) : Nat := Nat.gcd a b
def lcm (a b : Nat) : Nat := Nat.lcm a b

theorem ratio_of_two_numbers (A B : Nat) (h₁ : gcd A B = 84) (h₂ : lcm A B = 21) (h₃ : A = 84) : 
  A / B = 4 :=
by
  sorry

end ratio_of_two_numbers_l85_85646


namespace right_triangle_leg_square_l85_85750

theorem right_triangle_leg_square (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = a + 2) : 
  b^2 = 4 * a + 4 := 
by 
  sorry

end right_triangle_leg_square_l85_85750


namespace cupcakes_baking_l85_85960

theorem cupcakes_baking (sugar_at_home : ℕ) (bags_bought : ℕ) (cups_per_bag : ℕ)
    (batter_sugar_per_dozen : ℕ) (frosting_sugar_per_dozen : ℕ) :
  sugar_at_home = 3 → bags_bought = 2 → cups_per_bag = 6 →
  batter_sugar_per_dozen = 1 → frosting_sugar_per_dozen = 2 →
  (sugar_at_home + bags_bought * cups_per_bag) / (batter_sugar_per_dozen + frosting_sugar_per_dozen) = 5 :=
by
  intros sugar_at_home_eq bags_bought_eq cups_per_bag_eq batter_sugar_per_dozen_eq frosting_sugar_per_dozen_eq
  rw [sugar_at_home_eq, bags_bought_eq, cups_per_bag_eq, batter_sugar_per_dozen_eq, frosting_sugar_per_dozen_eq]
  simp
  sorry

end cupcakes_baking_l85_85960


namespace sum_possible_values_l85_85570

-- helper definitions capturing volumes and surface areas
def volume (a b c : ℕ) := a * b * c
def edge_sum (a b c : ℕ) := 4 * (a + b + c)
def surface_area (a b c : ℕ) := 2 * (a * b + b * c + c * a)

-- main theorem statement
theorem sum_possible_values (A B C : ℕ) : 
  (∑ s in {s | ∃ a b c, a + b + c = s ∧ volume a b c = edge_sum a b c + surface_area a b c}, id) = 647 :=
by sorry

end sum_possible_values_l85_85570


namespace num_integers_with_2_and_6_between_300_and_700_l85_85874

theorem num_integers_with_2_and_6_between_300_and_700 : 
  card ({n : ℕ | 300 ≤ n ∧ n < 700 ∧ (∃ t u, (n = 300 * t + 10 * u + 6 * (t ≠ u))) ∧ (∃ d1 d2, (d1 ∈ {2, 6}) ∧ (d2 ∈ {2, 6}) ∧ d1 ≠ d2)}) = 8 :=
sorry

end num_integers_with_2_and_6_between_300_and_700_l85_85874


namespace vertices_degree_difference_at_most_m_minus_one_l85_85465

open Finset

section

variables {V : Type} [Fintype V] [DecidableEq V]

def vertex_degree (G : V → V → Prop) (v : V) : ℕ :=
  Fintype.card { u : V // G v u }

theorem vertices_degree_difference_at_most_m_minus_one
  {G : V → V → Prop} {n m : ℕ} (h1 : Fintype.card V = n) (h2 : m < n) :
  ∃ (S : Finset V), S.card = (m + 1) ∧
    ∃ (d_max d_min : ℕ), d_max - d_min ≤ (m - 1) ∧
    (∀ v ∈ S, vertex_degree G v = d_max ∨ vertex_degree G v = d_min) :=
sorry

end

end vertices_degree_difference_at_most_m_minus_one_l85_85465


namespace geometric_sequence_sum_l85_85194

theorem geometric_sequence_sum
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (r : ℝ)
  (h1 : r ≠ 1)
  (h2 : ∀ n, S n = a 0 * (1 - r^(n + 1)) / (1 - r))
  (h3 : S 5 = 3)
  (h4 : S 10 = 9) :
  S 15 = 21 :=
sorry

end geometric_sequence_sum_l85_85194


namespace brenda_sally_track_length_l85_85015

theorem brenda_sally_track_length
  (c d : ℝ) 
  (h1 : c / 4 * 3 = d) 
  (h2 : d - 120 = 0.75 * c - 120) 
  (h3 : 0.75 * c + 60 <= 1.25 * c - 180) 
  (h4 : (c - 120 + 0.25 * c - 60) = 1.25 * c - 180):
  c = 766.67 :=
sorry

end brenda_sally_track_length_l85_85015


namespace units_digit_of_product_composites_l85_85322

def is_composite (n : ℕ) : Prop := 
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

theorem units_digit_of_product_composites (h1 : is_composite 9) (h2 : is_composite 10) (h3 : is_composite 12) :
  (9 * 10 * 12) % 10 = 0 :=
by
  sorry

end units_digit_of_product_composites_l85_85322


namespace smallest_norm_of_v_eq_l85_85211

noncomputable def v : ℝ² := sorry -- assume we have some vector v satisfying the condition

-- the main theorem statement
theorem smallest_norm_of_v_eq : 
  ‖(v + ![4, 2])‖ = 10 → ‖v‖ = 10 - 2 * Real.sqrt 5 := 
by
  intro h
  sorry

end smallest_norm_of_v_eq_l85_85211


namespace cos_pi_minus_theta_l85_85492

-- Define the point (4, -3)
def point := (4 : ℕ, -3 : ℤ)

-- Define the cosine value of θ using the point (4, -3)
def cos_theta : ℚ := 4 / real.sqrt (4 ^ 2 + (-3 : ℚ) ^ 2)

-- The goal is to prove that cos (π - θ) = -4/5 given the point condition
theorem cos_pi_minus_theta (θ : ℝ) (h : (cos θ = 4 / 5)) : cos (π - θ) = -4 / 5 :=
by {
  sorry
}

end cos_pi_minus_theta_l85_85492


namespace product_lcm_gcd_eq_2160_l85_85695

theorem product_lcm_gcd_eq_2160 :
  let a := 36
  let b := 60
  lcm a b * gcd a b = 2160 := by
  sorry

end product_lcm_gcd_eq_2160_l85_85695


namespace ratio_proof_l85_85982

variable (s : ℝ) -- side length of square IJKL
variable (x y : ℝ) -- lengths AB and AD of rectangle ABCD
variable (ratio : ℝ) -- ratio AB over AD

-- Conditions
def shared_area_square: ℝ := 0.25 * s^2
def shared_area_rectangle: ℝ := 0.4 * x * y

-- Given that the shared areas are equal
def shared_area_condition := shared_area_square = shared_area_rectangle

-- Calculating the ratio between AB and AD
def calculate_ratio := ratio = x / y

-- Given the side y is a fifth of the square side length
def y_condition := y = s / 5

-- Given that AB is 1.6 times the side length s
def x_condition := x = 1.6 * s

theorem ratio_proof (s x y ratio : ℝ)
  (h_shared_area : shared_area_square s = shared_area_rectangle s x y)
  (h_y : y = s / 5)
  (h_x : x = 1.6 * s) :
  ratio = 8 := by
  sorry

end ratio_proof_l85_85982


namespace no_consecutive_positive_integers_with_no_real_solutions_l85_85791

theorem no_consecutive_positive_integers_with_no_real_solutions :
  ∀ b c : ℕ, (c = b + 1) → (b^2 - 4 * c < 0) → (c^2 - 4 * b < 0) → false :=
by
  intro b c
  sorry

end no_consecutive_positive_integers_with_no_real_solutions_l85_85791


namespace problem_equivalent_l85_85883

variable {m n : ℝ}
theorem problem_equivalent : (mn = m + 3) -> (3m - 3mn + 10 = 1) :=
by
  intro h
  sorry

end problem_equivalent_l85_85883


namespace find_value_of_a_l85_85106

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def tangent_line (x : ℝ) : ℝ := x - 1
noncomputable def quadratic_curve (x a : ℝ) : ℝ := x^2 + a

theorem find_value_of_a :
  let a : ℝ := -3/4 in
  ∀ x : ℝ, tangent_line x = quadratic_curve x a → (4 * a + 3 = 0) :=
by
  intro a x h;
  have h1 : 4 * a + 3 = 0 := sorry;
  exact h1

end find_value_of_a_l85_85106


namespace odd_n_no_cycle_l85_85169

-- Define a structure for our graph
structure Graph (V : Type) :=
(edges : V → V → Prop)
(no_cycle_n : Π {n : ℕ}, n % 2 = 1 → (∀ v w : V, ¬(∃ p : list V, length p = n ∧ p.headI = v ∧ p.lastI = w ∧ (list.pairwise (edges) p) ∧ (∀ (u : V), u ∈ p → u ≠ v ∧ u ≠ w))))
(path_length_100 : Π (v w : V), ∃ p : list V, length p = 100 ∧ p.headI = v ∧ p.lastI = w ∧ (list.pairwise (edges) p) ∧ (∀ (u : V), count u p ≤ 1))

-- Define the theorem
theorem odd_n_no_cycle {V : Type} (G : Graph V) (n : ℕ) (h1 : n % 2 = 1) :
  n ≠ 101 :=
by
  intro h2
  rw [← h2] at h1
  have : G.no_cycle_n h1 := G.no_cycle_n h1
  have path_100 := G.path_length_100 
  sorry

end odd_n_no_cycle_l85_85169


namespace area_of_intersection_exactly_three_triangles_l85_85972

-- Define the shapes and their properties
def rectangle (a b : ℝ) := {len := a, width := b, area := a * b}
def equilateral_triangle (s : ℝ) := {side := s, area := (s * s * sqrt 3) / 4}

-- Noncomputable as exact calculation with real numbers
noncomputable def calculateAreaThreeTrianglesIntersection (a b : ℝ) : ℝ := 
  -- The function that calculates the area as described
  sorry

-- Problem statement
theorem area_of_intersection_exactly_three_triangles (A B : ℝ) (h : A = 6 ∧ B = 8) : 
  calculateAreaThreeTrianglesIntersection A B = (288.0 - 154.0 * (sqrt 3)) / 3.0 :=
  sorry

end area_of_intersection_exactly_three_triangles_l85_85972


namespace winner_won_by_324_votes_l85_85184

theorem winner_won_by_324_votes
  (total_votes : ℝ)
  (winner_percentage : ℝ)
  (winner_votes : ℝ)
  (h1 : winner_percentage = 0.62)
  (h2 : winner_votes = 837) :
  (winner_votes - (0.38 * total_votes) = 324) :=
by
  sorry

end winner_won_by_324_votes_l85_85184


namespace ways_to_divide_8_friends_l85_85519

theorem ways_to_divide_8_friends : (4 : ℕ) ^ 8 = 65536 := by
  sorry

end ways_to_divide_8_friends_l85_85519


namespace processed_apples_eq_stems_l85_85369

-- Define the constants
constant n : ℕ -- n is the number of stems seen after 2 hours
constant applesProcessed : ℕ -- applesProcessed is the number of apples processed

-- Theorem statement
theorem processed_apples_eq_stems (h : applesProcessed = n) : applesProcessed = n := by
  sorry

end processed_apples_eq_stems_l85_85369


namespace prism_volume_is_correct_l85_85382

noncomputable def prism_volume 
  (a b c : ℝ) 
  (hab : a * b = 15) 
  (hbc : b * c = 18) 
  (hca : c * a = 20) 
  (hc_longest : c = 2 * a) 
  : ℝ :=
  a * b * c

theorem prism_volume_is_correct 
  (a b c : ℝ) 
  (hab : a * b = 15) 
  (hbc : b * c = 18) 
  (hca : c * a = 20) 
  (hc_longest : c = 2 * a) 
  : prism_volume a b c hab hbc hca hc_longest = 30 * Real.sqrt 10 :=
sorry

end prism_volume_is_correct_l85_85382


namespace number_of_real_roots_of_f_f_eq_0_l85_85221

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else real.sqrt x - 1

theorem number_of_real_roots_of_f_f_eq_0 :
  (set.to_finset {x : ℝ | f (f x) = 0}).card = 3 :=
sorry

end number_of_real_roots_of_f_f_eq_0_l85_85221


namespace regression_sum_of_squares_l85_85536

theorem regression_sum_of_squares (SST SSE SSR : ℝ) (h1 : SST = 256) (h2 : SSE = 32) :
  SSR = SST - SSE → SSR = 224 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end regression_sum_of_squares_l85_85536


namespace int_solutions_count_l85_85141

noncomputable def number_of_integer_solutions : ℕ :=
  Set.count { x : ℤ // (x - 3)^(36 - x^2) = 1 }

theorem int_solutions_count : number_of_integer_solutions = 4 := by
  sorry

end int_solutions_count_l85_85141


namespace b_pow_a_eq_nine_l85_85147

theorem b_pow_a_eq_nine (a b : ℤ) (h1 : a < Real.sqrt 8) (h2 : Real.sqrt 8 < b) (h3 : a + 1 = b) : b ^ a = 9 :=
by
  sorry

end b_pow_a_eq_nine_l85_85147


namespace misha_second_session_score_l85_85345

theorem misha_second_session_score (x : ℝ) (hx : x ≥ 0) (first_session : ℝ := 8) :
  let first_score := first_session * 1 in
  let second_score := 2 * first_score in
  let third_score := 1.5 * second_score in
  second_score = 16 := by
  let first_score := first_session * 1 
  have hfirst : first_score = 8 := by
    simp [first_session]
  let second_score := 2 * first_score
  have hsecond : second_score = 16 := by
    simp [first_score, hfirst]
  exact hsecond


end misha_second_session_score_l85_85345


namespace corners_have_different_colors_l85_85636

def painted_board (n : ℕ) := matrix (fin n) (fin n) (fin 4)

def valid_painting (n : ℕ) (board : painted_board n) : Prop :=
  ∀ i j : fin n, ∀ d : ℤ, 
  (d = -1 ∨ d = 0 ∨ d = 1) → 
  (i + d >= 0 ∧ i + d < n ∧ j + d >= 0 ∧ j + d < n) → 
  board i j ≠ board (i + d) j ∧ board i j ≠ board i (j + d)

theorem corners_have_different_colors {n : ℕ} (h_n : n = 100) (board : painted_board n) 
  (h_valid : valid_painting n board) :
  board 0 0 ≠ board 0 (n - 1) ∧
  board 0 0 ≠ board (n - 1) 0 ∧
  board 0 0 ≠ board (n - 1) (n - 1) ∧
  board 0 (n - 1) ≠ board (n - 1) 0 ∧
  board 0 (n - 1) ≠ board (n - 1) (n - 1) ∧
  board (n - 1) 0 ≠ board (n - 1) (n - 1) :=
sorry

end corners_have_different_colors_l85_85636


namespace sequence_patterns_l85_85803

theorem sequence_patterns :
  (∀ (n : ℕ), seq1 n = seq1 (n - 1) * 2 + 1) →
  (∀ (n : ℕ), seq2 (2 * n + 2) = seq2 (2 * n) + 2) →
  (∀ (n : ℕ), seq2 (2 * n + 1) = seq2 (2 * n + 1) + 1) →
  seq1 5 = 127 ∧ seq1 6 = 255 ∧ seq2 1 = 7 ∧ seq2 2 = 11 :=
by
  intros h_seq1 h_seq2_ev h_seq2_odd
  sorry

end sequence_patterns_l85_85803


namespace UncleVanya_travel_time_l85_85346

-- Define the conditions
variables (x y z : ℝ)
variables (h1 : 2 * x + 3 * y + 20 * z = 66)
variables (h2 : 5 * x + 8 * y + 30 * z = 144)

-- Question: how long will it take to walk 4 km, cycle 5 km, and drive 80 km
theorem UncleVanya_travel_time : 4 * x + 5 * y + 80 * z = 174 :=
sorry

end UncleVanya_travel_time_l85_85346


namespace lisa_goal_l85_85236

theorem lisa_goal 
  (total_quizzes : ℕ) 
  (target_percentage : ℝ) 
  (completed_quizzes : ℕ) 
  (earned_A : ℕ) 
  (remaining_quizzes : ℕ) : 
  total_quizzes = 40 → 
  target_percentage = 0.9 → 
  completed_quizzes = 25 → 
  earned_A = 20 → 
  remaining_quizzes = (total_quizzes - completed_quizzes) → 
  (earned_A + remaining_quizzes ≥ target_percentage * total_quizzes) → 
  remaining_quizzes - (total_quizzes * target_percentage - earned_A) = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end lisa_goal_l85_85236


namespace six_digit_numbers_exactly_one_zero_l85_85138

def digits := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem six_digit_numbers_exactly_one_zero : 
  (9 * 10 ^ 5) - ∑ i in [1, 2, 3, 4, 5, 6], (∑ d in digits, ∑ e in digits, ∑ f in digits, ∑ g in digits, ∑ h in digits, ite ((i = 1 ∧ d = 0) ∨ (i = 2 ∧ e = 0) ∨ (i = 3 ∧ f = 0) ∨ (i = 4 ∧ g = 0) ∨ (i = 5 ∧ h = 0)) 1 0) = 295245 :=
by
  sorry

end six_digit_numbers_exactly_one_zero_l85_85138


namespace probability_53_mondays_in_leap_year_l85_85052

theorem probability_53_mondays_in_leap_year : 
  let total_days := 366
  let total_weeks := total_days / 7
  let extra_days := total_days % 7
  (extra_days = 2) →
  let combinations := [
    ("Sunday", "Monday"), 
    ("Monday", "Tuesday"), 
    ("Tuesday", "Wednesday"), 
    ("Wednesday", "Thursday"), 
    ("Thursday", "Friday"), 
    ("Friday", "Saturday"), 
    ("Saturday", "Sunday")
  ]
  let mondays_in_combinations := (("Sunday", "Monday"), ("Monday", "Tuesday"))
  let probability := mondays_in_combinations.length / combinations.length
  probability = 2 / 7 :=
by
  sorry

end probability_53_mondays_in_leap_year_l85_85052


namespace same_solution_ordered_pair_l85_85563

theorem same_solution_ordered_pair (b c : ℤ) : 
  (∀ x : ℝ, |x-5| = 2 ↔ x = 3 ∨ x = 7) -> 
  (x^2 + b * x + c = 0 ↔ (x = 3 ∨ x = 7)) -> 
  (b, c) = (-10, 21) := 
by 
  intros h1 h2
  have : x^2 - 10 * x + 21 = 0 ↔ (x = 3 ∨ x = 7) := by 
    sorry
  have : b = -10 ∧ c = 21 := by 
    sorry
  exact ⟨this.1, this.2⟩

end same_solution_ordered_pair_l85_85563


namespace find_S₉_l85_85189

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℝ := 3 + (n - 1) * 1

-- Sum of the first 9 terms in the sequence {a_n}
def S₉ := (9 * (a 1 + a 9)) / 2

-- Given condition for the problem
axiom condition : a 3 + a 5 + a 7 = 15

-- Prove that S₉ = 45 given the condition
theorem find_S₉ : S₉ = 45 :=
by
  sorry

end find_S₉_l85_85189


namespace probability_at_least_two_tails_l85_85007

def fair_coin_prob (n : ℕ) : ℚ :=
  (1 / 2 : ℚ)^n

def at_least_two_tails_in_next_three_flips : ℚ :=
  1 - (fair_coin_prob 3 + 3 * fair_coin_prob 3)

theorem probability_at_least_two_tails :
  at_least_two_tails_in_next_three_flips = 1 / 2 := 
by
  sorry

end probability_at_least_two_tails_l85_85007


namespace area_stage_8_l85_85528

/-- If a 4'' by 4'' square is added every second stage starting from Stage 1,
    then the area of the figure at Stage 8 is 64 square inches. -/
theorem area_stage_8 : 
  (∑ i in {1, 3, 5, 7}, (4 * 4)) = 64 :=
by
  sorry

end area_stage_8_l85_85528


namespace base4_to_base10_conversion_l85_85029

-- We define a base 4 number as follows:
def base4_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10 in
  let n := n / 10 in
  let d1 := n % 10 in
  let n := n / 10 in
  let d2 := n % 10 in
  let n := n / 10 in
  let d3 := n % 10 in
  let n := n / 10 in
  let d4 := n % 10 in
  (d4 * 4^4 + d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0)

-- Mathematical proof problem statement:
theorem base4_to_base10_conversion : base4_to_base10 21012 = 582 :=
  sorry

end base4_to_base10_conversion_l85_85029


namespace height_of_mother_is_l85_85331

-- Defining the given conditions
variables (xiaoming_height stool_height taller_than_mother : ℝ)
variables (height_of_mother : ℝ)

-- Assigning values to conditions
def xiaoming_height := 1.30
def stool_height := 0.40
def taller_than_mother := 0.08

-- Stating the theorem
theorem height_of_mother_is :
  height_of_mother = xiaoming_height + stool_height - taller_than_mother :=
sorry

end height_of_mother_is_l85_85331


namespace max_area_triangle_ABC_l85_85553

def points : Type := ℝ × ℝ

variables (A B C P : points)
noncomputable def dist (x y : points) : ℝ :=
real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

variables (PA PB PC BC : ℝ)
  (hPA : dist P A = 5)
  (hPB : dist P B = 8)
  (hPC : dist P C = 6)
  (hBC : dist B C = 10)

theorem max_area_triangle_ABC : 
  ∃ A B C P : points, dist P A = 5 ∧ dist P B = 8 ∧ dist P C = 6 ∧ dist B C = 10 ∧ 
  49 = let h_A := PA + (2 * 24 / BC) in 0.5 * BC * h_A :=
begin
  sorry,
end

end max_area_triangle_ABC_l85_85553


namespace find_ratio_l85_85643

theorem find_ratio (f : ℝ → ℝ) (h : ∀ a b : ℝ, b^2 * f a = a^2 * f b) (h3 : f 3 ≠ 0) :
  (f 7 - f 3) / f 3 = 40 / 9 :=
sorry

end find_ratio_l85_85643


namespace max_non_attacking_rooks_l85_85361

theorem max_non_attacking_rooks (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 299) (h3 : 1 ≤ b) (h4 : b ≤ 299) :
  ∃ max_rooks : ℕ, max_rooks = 400 :=
  sorry

end max_non_attacking_rooks_l85_85361


namespace range_of_b_l85_85576

theorem range_of_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b)
  (h4 : tendsto (λ n, (a ^ (n + 1) - b ^ (n + 1)) / (a ^ n + b ^ n)) at_top (nhds 2)) : 
  b ∈ set.Ioo 0 2 := sorry

end range_of_b_l85_85576


namespace power_of_a_l85_85893

theorem power_of_a (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 := sorry

end power_of_a_l85_85893


namespace max_distance_ellipse_to_line_l85_85908

open Real

-- Define the ellipse and the line
def ellipse (x y : ℝ) := (x^2) / 16 + (y^2) / 9 = 1
def line (x y : ℝ) := x - y - 5 = 0

-- State the theorem to find the maximum distance
theorem max_distance_ellipse_to_line :
  ∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧ (∀ Q : ℝ × ℝ, ellipse Q.1 Q.2 → dist (Q.1, Q.2) (line Q.1 Q.2) ≤ 5 * sqrt 2) :=
sorry

end max_distance_ellipse_to_line_l85_85908


namespace cos_pi_plus_2alpha_l85_85820

theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin ((Real.pi / 2) + α) = 1 / 3) : Real.cos (Real.pi + 2 * α) = 7 / 9 :=
by
  sorry

end cos_pi_plus_2alpha_l85_85820


namespace sin_x_correct_l85_85877

noncomputable def sin_x (a b c : ℝ) (x : ℝ) : ℝ :=
  2 * a * b * c / Real.sqrt (a^4 + 2 * a^2 * b^2 * (c^2 - 1) + b^4)

theorem sin_x_correct (a b c x : ℝ) 
  (h₁ : a > b) 
  (h₂ : b > 0) 
  (h₃ : c > 0) 
  (h₄ : 0 < x ∧ x < Real.pi / 2) 
  (h₅ : Real.tan x = 2 * a * b * c / (a^2 - b^2)) :
  Real.sin x = sin_x a b c x :=
sorry

end sin_x_correct_l85_85877


namespace total_weight_of_rice_l85_85618

theorem total_weight_of_rice :
  (29 * 4) / 16 = 7.25 := by
sorry

end total_weight_of_rice_l85_85618


namespace fraction_spent_december_correct_l85_85396

variable {S : ℝ} -- S is Abi's monthly salary

-- Conditions
def january_savings := 0.10 * S
def february_savings := 0.12 * S
def march_savings := 0.15 * S
def april_savings := 0.20 * S
def may_savings := 0.15 * S
def june_savings := 0.16 * S
def july_savings := 0.17 * S
def august_savings := 0.18 * S
def september_savings := 0.19 * S
def october_savings := 0.20 * S
def november_savings := 0.21 * S
def december_savings := 0.22 * S

def total_savings := january_savings + february_savings + march_savings + april_savings +
                     may_savings + june_savings + july_savings + august_savings +
                     september_savings + october_savings + november_savings + december_savings

def december_spending_fraction := 39 / 50

def fraction_spent_in_december (D : ℝ) :=  D / S = december_spending_fraction

theorem fraction_spent_december_correct (D : ℝ) (h1 : total_savings = 4 * D) :
  fraction_spent_in_december D :=
by
  sorry

end fraction_spent_december_correct_l85_85396


namespace constant_term_expansion_eq_l85_85427

theorem constant_term_expansion_eq : 
  let general_term (r : ℕ) := (binom 6 r : ℚ) * (-(1/2))^r * (x^2)^(6-r) * (-(1/2x))^r in
  let constant_term := \(- \frac{1}{2}\) := 4 * binom(6, 4) = (15 / 16 : ℚ) 

end constant_term_expansion_eq_l85_85427


namespace scientific_notation_correct_l85_85285

noncomputable def scientific_notation (x : ℕ) : Prop :=
  x = 3010000000 → 3.01 * (10 ^ 9) = 3.01 * (10 ^ 9)

theorem scientific_notation_correct : 
  scientific_notation 3010000000 :=
by
  intros h
  sorry

end scientific_notation_correct_l85_85285


namespace a3_pm_2b3_not_div_by_37_l85_85278

theorem a3_pm_2b3_not_div_by_37 {a b : ℤ} (ha : ¬ (37 ∣ a)) (hb : ¬ (37 ∣ b)) :
  ¬ (37 ∣ (a^3 + 2 * b^3)) ∧ ¬ (37 ∣ (a^3 - 2 * b^3)) :=
  sorry

end a3_pm_2b3_not_div_by_37_l85_85278


namespace diagonal_bisected_l85_85275

open EuclideanGeometry

/-
  The problem setup:
  Let ABCD be a convex quadrilateral.
  O is the intersection point of its diagonals.
  The sum of areas of triangles formed by diagonals is equal.
-/

theorem diagonal_bisected (A B C D O : Point) 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_diagonal_intersection A B C D O)
  (equal_areas : area (triangle A O B) + area (triangle C O D) = area (triangle B O C) + area (triangle D O A)) :
  (is_bisected A C O) ∨ (is_bisected B D O) :=
  sorry

end diagonal_bisected_l85_85275


namespace fifth_selected_is_11_l85_85745

noncomputable def population : List Nat := List.range 1 21

noncomputable def random_number_table : List (List Nat) := 
  [[7816, 6572, 0802, 6314, 0702, 4311], 
   [3204, 9234, 4935, 8200, 3623, 4869]]

noncomputable def extract_valid_numbers (table : List (List Nat)) : List Nat :=
  table.join.filter (λ n, n <= 20)

noncomputable def selected_individuals : List Nat :=
  extract_valid_numbers random_number_table

theorem fifth_selected_is_11 : selected_individuals.nth 4 = some 11 := by
  sorry

end fifth_selected_is_11_l85_85745


namespace profit_percentage_correct_l85_85726

theorem profit_percentage_correct :
  ∃ p : ℝ, p ≈ 19.99 ∧
  let sp := 260, cp := 216.67, profit := sp - cp in
  p = (profit / cp) * 100 :=
by
  sorry

end profit_percentage_correct_l85_85726


namespace find_c_for_polygon_l85_85929

noncomputable def c (n : ℕ) (P : EuclideanGeometry.Polygon (2 * n + 2)) (hP : P.area = 1) : ℝ :=
  sqrt (2 / Real.pi)

theorem find_c_for_polygon (n : ℕ) (h : n = 1011) (P : EuclideanGeometry.Polygon (2 * n + 2))
  (hP : P.area = 1) (A B : EuclideanGeometry.Point)
  (hA : A ∈ P.vertices) (hB : B ∈ P.vertices) :
  ProbabilityTheory.probability (dist A B ≥ c n P hP) = 1 / 2 :=
sorry

end find_c_for_polygon_l85_85929


namespace sum_of_20th_and_30th_triangular_numbers_l85_85793

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_20th_and_30th_triangular_numbers :
  triangular_number 20 + triangular_number 30 = 675 :=
by
  sorry

end sum_of_20th_and_30th_triangular_numbers_l85_85793


namespace power_function_value_l85_85534

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem power_function_value (a : ℝ) (h : f 2 a = (Real.sqrt 2) / 2) : f 4 a = 1 / 2 :=
by
  have h : 2 ^ a = (Real.sqrt 2) / 2 := sorry  -- This is the given condition
  have a_eq : a = -1 / 2 := sorry  -- This is solved from the given condition
  show 4 ^ (-1 / 2) = 1 / 2  -- The goal from the question
  sorry  -- Placeholder for the actual proof

end power_function_value_l85_85534


namespace price_of_scooter_l85_85961

-- Assume upfront_payment and percentage_upfront are given
def upfront_payment : ℝ := 240
def percentage_upfront : ℝ := 0.20

noncomputable
def total_price (upfront_payment : ℝ) (percentage_upfront : ℝ) : ℝ :=
  (upfront_payment / percentage_upfront)

theorem price_of_scooter : total_price upfront_payment percentage_upfront = 1200 :=
  by
    sorry

end price_of_scooter_l85_85961


namespace probability_abs_diff_gt_one_l85_85254

open Probability

-- Let x and y be random variables following the described distributions
noncomputable def dist_x : Measure ℝ := 
  0.5 • Measure.dirac 0 + -- heads on first flip + heads on second flip
  0.25 • Measure.dirac 2 + -- heads on first flip + tails on second flip
  0.25 • Measure.dirac (uniform_measure 0 2) -- heads on first flip

noncomputable def dist_y : Measure ℝ := 
  0.5 • Measure.dirac 0 + -- heads on first flip + heads on second flip
  0.25 • Measure.dirac 2 + -- heads on first flip + tails on second flip
  0.25 • Measure.dirac (uniform_measure 0 2) -- heads on first flip

noncomputable def prob_abs_diff_gt_one : Real := 
  ∫⁻ (x y : ℝ), indicator (λ p : ℝ × ℝ, ∥p.1 - p.2∥ > 1) sorry (dist_x.prod dist_y)

theorem probability_abs_diff_gt_one :
  prob_abs_diff_gt_one = 5/8 :=
sorry

end probability_abs_diff_gt_one_l85_85254


namespace find_x_l85_85878

def F (a b c d : ℤ) : ℤ := a ^ b + c * d

theorem find_x :
  (F 2 x 4 11 = 300) → (x = 8) :=
by
  -- proof goes here
  sorry

end find_x_l85_85878


namespace solve_for_N_l85_85527

theorem solve_for_N :
    (481 + 483 + 485 + 487 + 489 + 491 = 3000 - N) → (N = 84) :=
by
    -- Proof is omitted
    sorry

end solve_for_N_l85_85527


namespace largest_fraction_less_than_l85_85203

noncomputable def largest_fraction (m n : ℕ) (h1 : m + n ≤ 2005) (h2 : 23 * m < 16 * n) : ℚ := m / n

theorem largest_fraction_less_than (m n : ℕ) (h1 : m + n ≤ 2005) (h2 : 23 * m < 16 * n) :
  largest_fraction 816 1189 h1 h2 = 816 / 1189 :=
by
  sorry

end largest_fraction_less_than_l85_85203


namespace geometric_series_sum_l85_85327

theorem geometric_series_sum :
  let a := (1 : ℚ) / 3
  let r := -(1 / 3)
  let n := 5
  let S₅ := (a * (1 - r ^ n)) / (1 - r)
  S₅ = 61 / 243 := by
  let a := (1 : ℚ) / 3
  let r := -(1 / 3)
  let n := 5
  let S₅ := (a * (1 - r ^ n)) / (1 - r)
  sorry

end geometric_series_sum_l85_85327


namespace percentage_of_a_l85_85161

theorem percentage_of_a (a : ℕ) (x : ℕ) (h1 : a = 190) (h2 : (x * a) / 100 = 95) : x = 50 := by
  sorry

end percentage_of_a_l85_85161


namespace area_of_quadrilateral_MBNP_l85_85715

def is_triangle (A B C: Point) : Prop :=
∃ Δ, Area(Δ) = 30 ∧ PointsOnLine(Δ, A, B, C) 

def Point_on_segment (A B M: Point) (r : ℝ) : Prop :=
M ∈ Segment (A B) ∧ AM = r * MB

def Point_on_segment (B C N: Point) (s : ℝ) : Prop :=
N ∈ Segment (B C) ∧ BN = s * NC

def lines_intersect (A N C M P: Point): Prop :=
Segment (A N) ∩ Segment (C M) = {P}

theorem area_of_quadrilateral_MBNP
  (A B C M N P : Point)
  (h₁ : is_triangle A B C)
  (h₂ : Point_on_segment A B M 2)
  (h₃ : Point_on_segment B C N 1)
  (h₄ : lines_intersect A N C M P):
  Area (Quadrilateral M B N P) = 7 := sorry

end area_of_quadrilateral_MBNP_l85_85715


namespace intersection_fixed_line_l85_85473

open Real

-- Define the ellipse C with semi-major axis a, semi-minor axis b
def ellipse (x y : ℝ) := x^2 / 25 + y^2 / 9 = 1

-- Define focal points F1 and F2
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)

-- Define the given point D
def D : ℝ × ℝ := (41 / 8, 0)

-- Define the general line equation l passing through F2
def line_l (m y : ℝ) := 4 + m * y

-- Intersection point of lines l' and BD is always on the fixed straight line x = 25/4
theorem intersection_fixed_line (m y1 y2 : ℝ)
  (h1 : -y1 - y2 = 72 * m / (9 * m^2 + 25))
  (h2 : y1 * y2 = -81 / (9 * m^2 + 25)) :
  ∃ x, line_l m y2 = x ∧ x = 25 / 4 :=
sorry

end intersection_fixed_line_l85_85473


namespace find_original_radius_l85_85633

theorem find_original_radius (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) :
  r = n * (Real.sqrt 3 - 2) / 2 :=
by
  sorry

end find_original_radius_l85_85633


namespace sum_first_n_terms_l85_85101

noncomputable def f : ℝ → ℝ
| x => sorry

lemma f_add (a b : ℝ) : f (a + b) = f a * f b :=
sorry

lemma f_one : f 1 = 2 :=
sorry

lemma f_ne_zero (x : ℝ) : f x ≠ 0 :=
sorry

def a_n (n : ℕ) :=
(f n)^2 + f (2 * n) / f (2 * n - 1)

theorem sum_first_n_terms (n : ℕ) :
  (∑ i in Finset.range n, a_n i.succ) = 4 * n :=
sorry

end sum_first_n_terms_l85_85101


namespace ratio_2_10_as_percent_l85_85664

-- Define the problem conditions as given
def ratio_2_10 := 2 / 10

-- Express the question which is to show the percentage equivalent of the ratio 2:10
theorem ratio_2_10_as_percent : (ratio_2_10 * 100) = 20 :=
by
  -- Starting statement
  sorry -- Proof is not required here

end ratio_2_10_as_percent_l85_85664


namespace students_opted_for_both_math_and_science_l85_85337

theorem students_opted_for_both_math_and_science (T M_not S_not none: ℕ) 
(hT: T = 40)
(hMN: M_not = 10)
(hSN: S_not = 15)
(hNone: none = 2) : 
∃ (B: ℕ), B = (T - M_not) + (T - S_not) - (T - none) := 
by 
  have hM : ℕ := T - M_not
  have hS : ℕ := T - S_not 
  have hU : ℕ := T - none
  use (hM + hS - hU)

end students_opted_for_both_math_and_science_l85_85337


namespace linear_function_eq_l85_85854
noncomputable def linear_function : ℝ → ℝ := λ x, (-2 : ℝ) * x + 10

theorem linear_function_eq (x y : ℝ) : 
  (∃ (b : ℝ), y = -2 * x + b ∧ (2, 6) ∈ (λ x, (-2 : ℝ) * x + b) '' {(x : ℝ, x = 2)})
  → linear_function x = y :=
by simp [linear_function]; sorry

end linear_function_eq_l85_85854


namespace intersect_count_l85_85626

def g (x : ℝ) : ℝ := sorry
def g_inv (y : ℝ) : ℝ := sorry

axiom g_invertible : ∀ (y : ℝ), g (g_inv y) = y

theorem intersect_count : {x : ℝ | g (x^2) = g (x^6)}.finite.card = 3 := by
  sorry

end intersect_count_l85_85626


namespace tangent_line_at_point_range_of_k_l85_85940

-- Define the function f
def f (x : ℝ) (k : ℝ) : ℝ := log x - k * x

-- Conditions
variable (k : ℝ)
variable hk_pos : k = 2
variable hk_noroots : ¬ ∃ x : ℝ, f x k = 0

-- The tangent line problem
theorem tangent_line_at_point : 
  x + y + 1 = 0 :=
sorry

-- The range of values for k problem
theorem range_of_k :
  k > 1 / real.exp 1 :=
sorry

end tangent_line_at_point_range_of_k_l85_85940


namespace combination_seven_choose_three_l85_85012

-- Define the combination formula
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the problem-specific values
def n : ℕ := 7
def k : ℕ := 3

-- Problem statement: Prove that the number of combinations of 3 toppings from 7 is 35
theorem combination_seven_choose_three : combination 7 3 = 35 :=
  by
    sorry

end combination_seven_choose_three_l85_85012


namespace train_length_correct_l85_85759

noncomputable def length_of_train (speed_kmh : ℕ) (crossing_time_sec : ℕ) (bridge_length_m : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600 in
  let total_distance := speed_ms * crossing_time_sec in
  total_distance - bridge_length_m

theorem train_length_correct : length_of_train 45 30 230 = 145 := by
  sorry

end train_length_correct_l85_85759


namespace balls_in_consecutive_bins_probability_l85_85307

noncomputable def probability_consecutive_bins : ℚ :=
  let prob_ball_in_bin (i : ℕ) : ℚ := if i ≥ 1 then 3^(-i : ℤ) else 0
  let prob_consecutive_bins (a : ℕ) : ℚ := prob_ball_in_bin a * prob_ball_in_bin (a+1) * prob_ball_in_bin (a+2)
  let sum_prob_consecutive_bins : ℚ := ∑' (a : ℕ), prob_consecutive_bins a
  let permutations : ℚ := 3!

-- Main statement of the problem
theorem balls_in_consecutive_bins_probability : 
  ((permutations * sum_prob_consecutive_bins) : ℚ) = 1 / 117 := sorry

end balls_in_consecutive_bins_probability_l85_85307


namespace ratio_x0_a_l85_85866

-- Definitions of the function and conditions
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + a^2 + (Real.exp (2 * x + 2) - 2 * a) ^ 2

-- The main theorem statement
theorem ratio_x0_a {x0 a : ℝ} (h : f x0 a ≤ 9 / 5) : x0 / a  = -5 := 
sorry

end ratio_x0_a_l85_85866


namespace tan_22_5_equiv_l85_85661

theorem tan_22_5_equiv : 
  ∃ a b c d : ℕ, a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a + b + c + d = 4 ∧ 
  (tan (real.pi / 8) = real.sqrt a - real.sqrt b + real.sqrt c - d) :=
by
  have h1 : real.sin (real.pi / 4) = real.sqrt 2 / 2 := sorry,
  have h2 : real.cos (real.pi / 4) = real.sqrt 2 / 2 := sorry,
  have tan_half_angle : tan (real.pi / 8) = (1 - real.sqrt 2 / 2) / (real.sqrt 2 / 2), from sorry,
  have tan_val : tan (real.pi / 8) = real.sqrt 2 - 1, from sorry,
  existsi [2, 1, 0, 1],
  split,
  { -- Verify inequalities
    repeat { split }; linarith },
  split,
  { -- Sum of variables
    norm_num },
  -- Check the expression equivalence
  exact tan_val

end tan_22_5_equiv_l85_85661


namespace decagon_side_length_in_rectangle_l85_85906

theorem decagon_side_length_in_rectangle
  (AB CD : ℝ)
  (AE FB : ℝ)
  (s : ℝ)
  (cond1 : AB = 10)
  (cond2 : CD = 15)
  (cond3 : AE = 5)
  (cond4 : FB = 5)
  (regular_decagon : ℝ → Prop)
  (h : regular_decagon s) : 
  s = 5 * (Real.sqrt 2 - 1) :=
by 
  sorry

end decagon_side_length_in_rectangle_l85_85906


namespace arrangements_5_rooms_5_workers_not_adjacent_l85_85770

noncomputable def number_of_arrangements : ℕ :=
  ( (nat.choose 5 1 * nat.choose 4 1 * nat.choose 3 3 / nat.fact 2) 
  + (nat.choose 5 2 * nat.choose 3 2 * nat.choose 1 1 / nat.fact 2) )
  * nat.fact 3 * nat.choose 4 2

theorem arrangements_5_rooms_5_workers_not_adjacent :
  number_of_arrangements = 900 := 
by {
  -- placeholder for proof
  sorry 
}

end arrangements_5_rooms_5_workers_not_adjacent_l85_85770


namespace tan_22_5_expression_l85_85659

theorem tan_22_5_expression :
  let a := 2
  let b := 1
  let c := 0
  let d := 0
  let t := Real.tan (Real.pi / 8)
  a ≥ b ∧ b ≥ c ∧ c ≥ d ∧
  t = (Real.sqrt a) - (Real.sqrt b) + (Real.sqrt c) - d → 
  a + b + c + d = 3 :=
by
  intros
  exact sorry

end tan_22_5_expression_l85_85659


namespace interesting_coeffs_of_Q_l85_85235

def is_interesting (r : ℝ) : Prop :=
∃ (a b : ℤ), r = a + b * real.sqrt 2

def has_interesting_coeffs (P : Polynomial ℝ) : Prop :=
∀ i, is_interesting (Polynomial.coeff P i)

theorem interesting_coeffs_of_Q
  (A B Q : Polynomial ℝ)
  (hA : has_interesting_coeffs A)
  (hB : has_interesting_coeffs B)
  (hB_const : Polynomial.coeff B 0 = 1)
  (h : A = B * Q)
  : has_interesting_coeffs Q :=
sorry

end interesting_coeffs_of_Q_l85_85235


namespace enclosed_area_l85_85272

theorem enclosed_area (arc_length : ℝ) (side_length : ℝ) (num_arcs : ℕ) (theta : ℝ) (cot_pi_over_five : ℝ) :
  arc_length = 5 * Real.pi / 6 →
  side_length = 3 →
  num_arcs = 5 →
  theta = 5 * Real.pi / 6 →
  cot_pi_over_five ≈ 1.37638 →
  let r := arc_length / theta in
  let sector_area := (theta / (2 * Real.pi)) * Real.pi * r^2 in
  let total_sector_area := num_arcs * sector_area in
  let pentagon_area := (5 / 4) * side_length^2 * cot_pi_over_five in
  pentagon_area + total_sector_area = 15.50475 + 25 / 12 * Real.pi :=
by
  intros h1 h2 h3 h4 h5 r sector_area total_sector_area pentagon_area
  sorry

end enclosed_area_l85_85272


namespace derivative_y_x_l85_85437

noncomputable def x (t : ℝ) := (t / Real.sqrt (1 - t^2)) * Real.arcsin t + Real.log (Real.sqrt (1 - t^2))
noncomputable def y (t : ℝ) := t / Real.sqrt (1 - t^2)

theorem derivative_y_x (t : ℝ) (h : t ∈ Ioo (-1 : ℝ) (1 : ℝ)) : 
  Real.has_deriv_at (λ t, y t) (1 / Real.arcsin t) t → y t / x t = 1 / Real.arcsin t :=
sorry

end derivative_y_x_l85_85437


namespace part1_part2_part3_l85_85496

-- Definition of f and g
def f (x : ℝ) : ℝ := Real.log (abs x)
def g (x a : ℝ) : ℝ := (1 / (deriv f x)) + a * (deriv f x)

-- Proof statements as Lean axioms (no proof provided)
theorem part1 (x : ℝ) (a : ℝ) (h : x ≠ 0) : g x a = x + a / x :=
sorry

theorem part2 (a : ℝ) : (∀ x > 0, x + a / x ≥ 2) → (∃ x > 0, x + a / x = 2) → a = 1 :=
sorry

theorem part3 : ∫ x in (3 / 2 : ℝ)..2, (-(x / 3) + (7 / 6) - (1 / x)) = (7 / 24) + real.log 3 - 2 * real.log 2 :=
sorry

end part1_part2_part3_l85_85496


namespace zero_of_function_l85_85055

theorem zero_of_function : ∃ x : ℝ, (2 * x - 3) = 0 ∧ x = 3 / 2 :=
by
  use (3 / 2)
  split
  · -- Show that the function value at 3/2 is zero
    sorry -- Proof will be inserted here
  · -- Show that this x is indeed 3/2
    sorry -- Proof will be inserted here

end zero_of_function_l85_85055


namespace integer_solutions_count_eq_4_l85_85143

theorem integer_solutions_count_eq_4 : 
  ∃ S : Finset ℤ, (∀ x ∈ S, (x - 3) ^ (36 - x ^ 2) = 1) ∧ S.card = 4 := 
sorry

end integer_solutions_count_eq_4_l85_85143


namespace area_of_triangle_RDE_l85_85788

-- Define the points R, D, E and the variable m
variables (m : ℝ)
def R := (0 : ℝ, 9 : ℝ)
def D := (4 : ℝ, 9 : ℝ)
def E := (0 : ℝ, m)

-- Define the base RD and height RE
def base_RD := D.1 - R.1
def height_RE := R.2 - E.2

-- Define the area of the triangle RDE
def area_RDE := 1 / 2 * base_RD * height_RE

-- The proof statement
theorem area_of_triangle_RDE : area_RDE m = 2 * (9 - m) :=
by
  sorry

end area_of_triangle_RDE_l85_85788


namespace no_corresponding_x_implies_k_range_l85_85456

-- Given: A = B = R
def A := ℝ
def B := ℝ

-- Given: y = x^2 - 2x - 2 is a function from set A to set B
def f (x : A) : B := x^2 - 2 * x - 2

-- To prove: The range of k if there is no corresponding x in A such that f(x) = k is (-∞, -3)
theorem no_corresponding_x_implies_k_range : 
  (∀ k : B, (∀ x : A, f x ≠ k) → k ∈ set.Iio (-3)) :=
sorry

end no_corresponding_x_implies_k_range_l85_85456


namespace perfect_square_divisor_probability_15_fact_l85_85379

theorem perfect_square_divisor_probability_15_fact :
  let total_divisors := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * 2 * 2,
      perfect_square_divisors := 6 * 4 * 2 * 2 * 1 * 1,
      probability := perfect_square_divisors / total_divisors in
  probability = 1 / 42 :=
by
  sorry

end perfect_square_divisor_probability_15_fact_l85_85379


namespace pirate_treasure_probability_l85_85375

theorem pirate_treasure_probability :
  let num_islands := 8
  let prob_treasure_no_traps := 1 / 3
  let prob_traps_no_treasure := 1 / 6
  let prob_neither := 1 / 2 
  (nat.choose 8 4 * (1/3)^4 * (1/2)^4) = 35 / 648 :=
by
  sorry

end pirate_treasure_probability_l85_85375


namespace mean_after_removal_l85_85995

variable {n : ℕ}
variable {S : ℝ}
variable {S' : ℝ}
variable {mean_original : ℝ}
variable {size_original : ℕ}
variable {x1 : ℝ}
variable {x2 : ℝ}

theorem mean_after_removal (h_mean_original : mean_original = 42)
    (h_size_original : size_original = 60)
    (h_x1 : x1 = 50)
    (h_x2 : x2 = 60)
    (h_S : S = mean_original * size_original)
    (h_S' : S' = S - (x1 + x2)) :
    S' / (size_original - 2) = 41.55 :=
by
  sorry

end mean_after_removal_l85_85995


namespace max_trees_cut_down_l85_85248

-- Definitions for the grid size
def grid_size : ℕ := 100
def total_trees : ℕ := grid_size * grid_size

-- Definitions for rows and columns
def is_odd (n : ℕ) : Prop := n % 2 = 1
def odd_rows : finset ℕ := (finset.range grid_size).filter is_odd
def odd_columns : finset ℕ := (finset.range grid_size).filter is_odd

-- Define the maximum number of trees that can be cut down
def max_cut_down_trees := odd_rows.card * odd_columns.card

-- Theorem to prove the maximum number of trees that can be cut down
theorem max_trees_cut_down (n : ℕ) (h_grid : n = total_trees) :
  max_cut_down_trees = 2500 := by
  sorry

end max_trees_cut_down_l85_85248


namespace eq_of_divisible_l85_85088

theorem eq_of_divisible (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : a + b ∣ 5 * a + 3 * b) : a = b :=
sorry

end eq_of_divisible_l85_85088


namespace original_curve_given_rotation_l85_85474

variable {x y : ℝ}

def rotate_vector (θ : ℝ) (x y : ℝ) : ℝ × ℝ := 
  (x * Real.cos θ - y * Real.sin θ, x * Real.sin θ + y * Real.cos θ)

theorem original_curve_given_rotation :
  (∀ (x y : ℝ), rotate_vector (Real.pi / 4) x y ∈ {p : ℝ × ℝ | (p.fst)^2 - (p.snd)^2 = 2}) →
  ∀ (x y : ℝ), x * y = -1 :=
by
  sorry

end original_curve_given_rotation_l85_85474


namespace last_four_digits_5_pow_2015_l85_85244

theorem last_four_digits_5_pow_2015 :
  (5^2015) % 10000 = 8125 :=
by
  sorry

end last_four_digits_5_pow_2015_l85_85244


namespace train_crossing_platform_time_l85_85351

theorem train_crossing_platform_time (train_length : ℝ) (platform_length : ℝ) (time_cross_post : ℝ) :
  train_length = 300 → platform_length = 350 → time_cross_post = 18 → 
  (train_length + platform_length) / (train_length / time_cross_post) = 39 :=
by
  intros
  sorry

end train_crossing_platform_time_l85_85351


namespace smallest_four_digit_multiple_of_15_l85_85445

theorem smallest_four_digit_multiple_of_15 :
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 15 = 0) ∧ (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 15 = 0) → n ≤ m) ∧ n = 1005 :=
sorry

end smallest_four_digit_multiple_of_15_l85_85445


namespace cylinder_shape_l85_85903

-- Define the conditions in Lean
variable {r θ z : ℝ}  -- Coordinates in cylindrical system
variable {c : ℝ}      -- Positive constant

-- Main theorem statement
theorem cylinder_shape (h1 : 0 < c) (h2 : r = c) : 
  ∃ z θ, r * cos θ = c ∧ r * sin θ = c ∧ r = c := 
by
  sorry

end cylinder_shape_l85_85903


namespace exists_small_triangle_l85_85182

-- Definitions and conditions based on the identified problem points
def square_side_length : ℝ := 1
def total_points : ℕ := 53
def vertex_points : ℕ := 4
def interior_points : ℕ := 49
def total_area : ℝ := square_side_length ^ 2
def max_triangle_area : ℝ := 0.01

-- The main theorem statement
theorem exists_small_triangle
  (sq_side : ℝ := square_side_length)
  (total_pts : ℕ := total_points)
  (vertex_pts : ℕ := vertex_points)
  (interior_pts : ℕ := interior_points)
  (total_ar : ℝ := total_area)
  (max_area : ℝ := max_triangle_area)
  (h_side : sq_side = 1)
  (h_pts : total_pts = 53)
  (h_vertex : vertex_pts = 4)
  (h_interior : interior_pts = 49)
  (h_total_area : total_ar = 1) :
  ∃ (t : ℝ), t ≤ max_area :=
sorry

end exists_small_triangle_l85_85182


namespace right_triangle_logarithm_l85_85648

theorem right_triangle_logarithm (a b h : ℝ) 
  (h1 : a = log 27 / log 4) 
  (h2 : b = log 9 / log 2) 
  (h3 : a^2 + b^2 = h^2) : 
  4^h = 243 := by
sorry

end right_triangle_logarithm_l85_85648


namespace ellipse_equation_max_area_triangle_l85_85833

open Real

-- Definition of the ellipse E with endpoints along major axis A and B, F = (√3, 0) is one of its foci, and vector condition
variable (a b : ℝ) (a_gt_b_gt_0 : a > b ∧ b > 0)
variable (E : set (ℝ × ℝ)) (is_ellipse : ∀ (x y : ℝ), (x, y) ∈ E ↔ (x^2 / a^2 + y^2 / b^2 = 1))
variable (F : ℝ × ℝ) (F_eq : F = (sqrt 3, 0))
variable (A B : ℝ × ℝ) (A_B_major_axis : ∃ (vx vy : ℝ), A = (vx, vy) ∧ B = (-vx, -vy) ∧ vx ^ 2 = a ^ 2 - b ^ 2)
variable (AF_BF_dot : (fst A - fst F) * (fst B - fst F) + (snd A - snd F) * (snd B - snd F) = -1)

-- Question 1: Find the equation of the ellipse E
theorem ellipse_equation : E = {p : ℝ × ℝ | (fst p) ^ 2 / 4 + (snd p) ^ 2 = 1} :=
sorry

-- Question 2: If a line intersects the ellipse E at points M,N, and a circle passing through origin O has MN as its diameter, find the maximum area of ∆OMN
variable (k : ℝ) (k_ge_0 : k ≥ 0)
variable (M N : ℝ × ℝ)

-- Points on ellipse intersected by lines
variable (M_eq : M = (2 / sqrt (1 + 4 * k ^ 2), 2 * k / sqrt (1 + 4 * k ^ 2)))
variable (N_eq : N = (2 * k / sqrt (4 + k ^ 2), -2 / sqrt (4 + k ^2)))
variable (O : ℝ × ℝ) (O_eq : O = (0,0))

-- Question 2: Find the maximum area of triangle ∆OMN
theorem max_area_triangle : ∃ k, max_area k = 1 :=
sorry

end ellipse_equation_max_area_triangle_l85_85833


namespace ratio_circumscribed_circle_area_triangle_area_l85_85286

open Real

theorem ratio_circumscribed_circle_area_triangle_area (h R : ℝ) (h_eq : R = h / 2) :
  let circle_area := π * R^2
  let triangle_area := (h^2) / 4
  (circle_area / triangle_area) = π :=
by
  sorry

end ratio_circumscribed_circle_area_triangle_area_l85_85286


namespace geometric_vs_arithmetic_l85_85293

-- Definition of a positive geometric progression
def positive_geometric_progression (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = q * a n ∧ q > 0

-- Definition of an arithmetic progression
def arithmetic_progression (b : ℕ → ℝ) (d : ℝ) := ∀ n, b (n + 1) = b n + d

-- Theorem statement based on the problem and conditions
theorem geometric_vs_arithmetic
  (a : ℕ → ℝ) (b : ℕ → ℝ) (q : ℝ) (d : ℝ)
  (h1 : positive_geometric_progression a q)
  (h2 : arithmetic_progression b d)
  (h3 : a 6 = b 7) :
  a 3 + a 9 ≥ b 4 + b 10 := 
by 
  sorry

end geometric_vs_arithmetic_l85_85293


namespace original_price_of_shirt_l85_85408

-- Define the conditions as given in the problem
def discount_percent : ℝ := 0.15
def payment_after_discount : ℝ := 68
def discount_factor : ℝ := 1 - discount_percent

-- Prove that original price is $80
theorem original_price_of_shirt : 
  ∃ (P : ℝ), discount_factor * P = payment_after_discount ∧ P = 80 :=
begin
  sorry  -- Proof is skipped
end

end original_price_of_shirt_l85_85408


namespace graduation_ceremony_chairs_l85_85174

theorem graduation_ceremony_chairs (g p t a : ℕ) 
  (h_g : g = 50) 
  (h_p : p = 2 * g) 
  (h_t : t = 20) 
  (h_a : a = t / 2) : 
  g + p + t + a = 180 :=
by
  sorry

end graduation_ceremony_chairs_l85_85174


namespace inverse_of_A_sq_l85_85879

open Matrix

variables {α : Type*} [DecidableEq α] [Fintype α]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, -1],
  ![2, 1]
]

theorem inverse_of_A_sq : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := inverse (A_inv) in
  inverse (A * A) = ![
    ![7, -4],
    ![8, -1]
  ] := by
  sorry

end inverse_of_A_sq_l85_85879


namespace min_score_seventh_shot_to_break_record_shots_hitting_10_to_break_record_when_7th_shot_is_8_necessary_shot_of_10_when_7th_shot_is_10_l85_85387

-- Definitions for the problem conditions
def initial_points : ℕ := 52
def record_points : ℕ := 89
def max_shots : ℕ := 10
def points_range : Finset ℕ := Finset.range 11 \ {0}

-- Lean statement for the first question
theorem min_score_seventh_shot_to_break_record (x₇ : ℕ) (h₁: x₇ ∈ points_range) :
  initial_points + x₇ + 30 > record_points ↔ x₇ ≥ 8 :=
by sorry

-- Lean statement for the second question
theorem shots_hitting_10_to_break_record_when_7th_shot_is_8 (x₈ x₉ x₁₀ : ℕ)
  (h₂ : 8 ∈ points_range) 
  (h₃ : x₈ ∈ points_range) (h₄ : x₉ ∈ points_range) (h₅ : x₁₀ ∈ points_range) :
  initial_points + 8 + x₈ + x₉ + x₁₀ > record_points ↔ (x₈ = 10 ∧ x₉ = 10 ∧ x₁₀ = 10) :=
by sorry

-- Lean statement for the third question
theorem necessary_shot_of_10_when_7th_shot_is_10 (x₈ x₉ x₁₀ : ℕ)
  (h₆ : 10 ∈ points_range)
  (h₇ : x₈ ∈ points_range) (h₈ : x₉ ∈ points_range) (h₉ : x₁₀ ∈ points_range) :
  initial_points + 10 + x₈ + x₉ + x₁₀ > record_points ↔ (x₈ = 10 ∨ x₉ = 10 ∨ x₁₀ = 10) :=
by sorry

end min_score_seventh_shot_to_break_record_shots_hitting_10_to_break_record_when_7th_shot_is_8_necessary_shot_of_10_when_7th_shot_is_10_l85_85387


namespace algebraic_expression_value_l85_85062

theorem algebraic_expression_value (m : ℝ) (h : (2018 + m) * (2020 + m) = 2) : (2018 + m)^2 + (2020 + m)^2 = 8 :=
by
  sorry

end algebraic_expression_value_l85_85062


namespace compute_abc_l85_85880

theorem compute_abc (a b c : ℤ) (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h₁ : a + b + c = 30) 
  (h₂ : (1 : ℚ)/a + (1 : ℚ)/b + (1 : ℚ)/c + 300/(a * b * c) = 1) : a * b * c = 768 := 
by 
  sorry

end compute_abc_l85_85880


namespace fifteen_men_job_completion_time_l85_85154

theorem fifteen_men_job_completion_time (job_days_10_men : ℕ) (men_initial : ℕ) (men_final : ℕ) (prep_days : ℕ) (total_man_days : ℕ) :
  men_initial = 10 → job_days_10_men = 15 → men_final = 15 → prep_days = 2 → total_man_days = men_initial * job_days_10_men →
  (total_man_days / men_final + prep_days = 12) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end fifteen_men_job_completion_time_l85_85154


namespace consumption_order_l85_85421

theorem consumption_order :
  let west_consumption := 21428
  let non_west_consumption := 26848.55
  let russia_consumption := 302790.13
  (west_consumption < non_west_consumption) ∧ (non_west_consumption < russia_consumption) :=
by
  let west_consumption := 21428
  let non_west_consumption := 26848.55
  let russia_consumption := 302790.13
  have h1 : west_consumption < non_west_consumption := sorry
  have h2 : non_west_consumption < russia_consumption := sorry
  exact ⟨h1, h2⟩

end consumption_order_l85_85421


namespace exists_polynomials_floor_l85_85565

theorem exists_polynomials_floor (k : ℕ) (hk : 0 < k) :
  ∃ (P : ℕ → ℕ → ℕ), ∀ n : ℕ,
    (nat.floor ((n : ℚ) / k)) ^ k =
    P 0 n + P 1 n * (nat.floor ((n : ℚ) / k)) +
    P 2 n * (nat.floor ((n : ℚ) / k)) ^ 2 +
    ⋯ +
    P (k - 1) n * (nat.floor ((n : ℚ) / k)) ^ (k - 1) := sorry

end exists_polynomials_floor_l85_85565


namespace cone_rolls_path_l85_85749

theorem cone_rolls_path (r h m n : ℝ) (rotations : ℕ) 
  (h_rotations : rotations = 20)
  (h_ratio : h / r = 3 * Real.sqrt 133)
  (h_m : m = 3)
  (h_n : n = 133) : 
  m + n = 136 := 
by sorry

end cone_rolls_path_l85_85749


namespace find_h_l85_85658

variable {a b c : ℝ}
variable {n h k : ℝ}

-- Condition: ax^2 + bx + c = 5 * (x - 5)^2 - 3
def quadratic_eq_condition := ∀ x : ℝ, a * x^2 + b * x + c = 5 * (x - 5)^2 - 3

-- Definition of the quadratic expression transformed by multiplying by 4
def transformed_quadratic (a b c : ℝ) := 4 * a * x^2 + 4 * b * x + 4 * c

-- Expression of the transformed quadratic in the form n(x - h)^2 + k
def transformed_quadratic_form := n * (x - h)^2 + k

-- The theorem to prove
theorem find_h 
  (a b c : ℝ)
  (h : ℝ)
  (quadratic_eq_condition : ∀ x : ℝ, a * x^2 + b * x + c = 5 * (x - 5)^2 - 3)
  : h = 5 := 
sorry

end find_h_l85_85658


namespace find_other_root_of_z_squared_eq_neg75_add_65i_l85_85250

noncomputable def complex_second_root (z : ℂ) :=
  if h : z^2 = -75 + 65 * complex.I then
    - (4 + 9 * complex.I)
  else 
    0

theorem find_other_root_of_z_squared_eq_neg75_add_65i : ∀ z : ℂ,
  z^2 = -75 + 65 * complex.I →
  (z = 4 + 9 * complex.I ∨ z = -4 - 9 * complex.I) :=
begin
  intros z hz,
  have h1 : (4 + 9 * complex.I)^2 = -75 + 65i,
  { calc
        (4 + 9 * complex.I)^2 = 16 - 81 + 72 * complex.I
                            = -75 + 65 * complex.I },
  have h2 : (-4 - 9 * complex.I)^2 = -75 + 65i,
  { calc
        (-4 - 9 * complex.I)^2 = 16 - 81 + 72 * complex.I
                             = -75 + 65 * complex.I },
  by_cases h1z : z = 4 + 9 * complex.I,
  { left, exact h1z },
  { right,
    have key : z = - (4 + 9*complex.I),
    { calc
          z = -(4 + 9 * complex.I) : by sorry },
    exact key }
end

end find_other_root_of_z_squared_eq_neg75_add_65i_l85_85250


namespace sum_first_n_terms_l85_85081

noncomputable def a_n : ℕ → ℕ
| 0     := 0  -- Adding the base case for zero-indexing, although not used in problem specifics
| (n+1) := 2 * (n + 1) - 1

def S_n (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, 1 / ((a_n i) * (a_n (i + 1)))

theorem sum_first_n_terms (n : ℕ) : S_n n = n / (2 * n + 1) := by
  sorry

end sum_first_n_terms_l85_85081


namespace determine_k_range_l85_85500

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x
noncomputable def g (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def h (x : ℝ) : ℝ := (Real.log x) / (x * x)

theorem determine_k_range :
  (∀ x : ℝ, x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) → f k x = g x) →
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ x2 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1)) →
  k ∈ Set.Ico (1 / (Real.exp 1) ^ 2) (1 / (2 * Real.exp 1)) := 
  sorry

end determine_k_range_l85_85500


namespace proj_v_q_l85_85574

noncomputable def p : ℝ × ℝ := (2, 1)
noncomputable def q : ℝ × ℝ := (2, 1)
noncomputable def v : ℝ × ℝ := (4, 2)

-- Assume p and q are orthogonal
axiom ortho_pq : p.1 * q.1 + p.2 * q.2 = 0

-- Assume the projection of v on p is p itself
axiom proj_v_p : proj p v = p

-- Define the statement to prove
theorem proj_v_q :
  proj q v = q := sorry

end proj_v_q_l85_85574


namespace jen_dials_correct_number_probability_l85_85200

noncomputable def jen_prob_correct_number : ℚ :=
  let possible_first_three := {324, 327, 328} in
  let last_five_digits : Finset ℕ := {0, 2, 5, 8, 9} in 
  let total_combinations := possible_first_three.card * last_five_digits.1.perm 5 in
  1 / total_combinations

theorem jen_dials_correct_number_probability :
  jen_prob_correct_number = 1 / 360 :=
sorry

end jen_dials_correct_number_probability_l85_85200


namespace muffin_price_proof_l85_85258

noncomputable def price_per_muffin (s m t : ℕ) (contribution : ℕ) : ℕ :=
  contribution / (s + m + t)

theorem muffin_price_proof :
  ∀ (sasha_muffins melissa_muffins : ℕ) (h1 : sasha_muffins = 30) (h2 : melissa_muffins = 4 * sasha_muffins)
  (tiffany_muffins total_muffins : ℕ) (h3 : total_muffins = sasha_muffins + melissa_muffins)
  (h4 : tiffany_muffins = total_muffins / 2)
  (h5 : total_muffins = sasha_muffins + melissa_muffins + tiffany_muffins)
  (contribution : ℕ) (h6 : contribution = 900),
  price_per_muffin sasha_muffins melissa_muffins tiffany_muffins contribution = 4 :=
by
  intros sasha_muffins melissa_muffins h1 h2 tiffany_muffins total_muffins h3 h4 h5 contribution h6
  simp [price_per_muffin]
  sorry

end muffin_price_proof_l85_85258


namespace calc_factorial_sum_l85_85016

theorem calc_factorial_sum : 5 * Nat.factorial 5 + 4 * Nat.factorial 4 + Nat.factorial 4 = 720 := by
  sorry

end calc_factorial_sum_l85_85016


namespace equal_real_imaginary_parts_l85_85593

theorem equal_real_imaginary_parts : 
  ∀ (a : ℝ), (let z : ℂ := (⟨1, 2⟩ * (⟨a, 1⟩ : ℂ)) in z.re = z.im) → a = -3 :=
by 
  intros a h_eq;
  sorry

end equal_real_imaginary_parts_l85_85593


namespace sum_of_positive_integers_l85_85805

/-- Let τ(n) be the number of positive divisors of n.
    Prove that the sum of all positive integers n such that τ(n)^2 = 2n is 98. -/
theorem sum_of_positive_integers (τ : ℕ → ℕ) (hτ : ∀ n, τ n = nat.divisors n . card) :
  ∑ n in finset.filter (λ n, τ n * τ n = 2 * n) (finset.range 1000), n = 98 :=
by {
  have h1 : τ 1 = 1 := rfl,
  have h2 : τ 2 = 2 := rfl,
  have h3 : τ 8 = 4 := rfl,
  have h4 : τ 18 = 6 := rfl,
  have h5 : τ 72 = 12 := rfl,
  sorry
}

end sum_of_positive_integers_l85_85805


namespace range_of_m_l85_85160

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (-2 < x ∧ x ≤ 2) → x ≤ m) → m ≥ 2 :=
by
  intro h
  -- insert necessary proof steps here
  sorry

end range_of_m_l85_85160


namespace David_pushups_calculation_l85_85038

variable (Zachary_pushups : ℕ)
variable (David_pushups : ℕ)
variable (more_pushups : ℕ)
variable (Zachary_condition : Zachary_pushups = 35)
variable (more_condition : more_pushups = 9)

theorem David_pushups_calculation (h1 : Zachary_pushups = 35) (h2 : more_pushups = 9) (h3 : David_pushups = Zachary_pushups + more_pushups) : David_pushups = 44 := by
  rw [h1, h2, h3]
  sorry

end David_pushups_calculation_l85_85038


namespace max_alpha_min_beta_l85_85343

theorem max_alpha_min_beta 
  (M : ℝ) (α β : ℝ) 
  (hM : ∀ x y z : ℝ, M = ∑ cyc(√(x^2 + x * y + y^2) * √(y^2 + y * z + z^2)))
  (h_conditions : ∀ x y z : ℝ, α * (x * y + y * z + z * x) ≤ M ∧ M ≤ β * (x^2 + y^2 + z^2)) 
  :  α ≤ 3 ∧ β ≥ 3 := 
by 
  sorry

end max_alpha_min_beta_l85_85343


namespace max_streetlights_l85_85751

theorem max_streetlights {road_length streetlight_length : ℝ} 
  (h1 : road_length = 1000)
  (h2 : streetlight_length = 1)
  (fully_illuminated : ∀ (n : ℕ), (n * streetlight_length) < road_length)
  : ∃ max_n, max_n = 1998 ∧ (∀ n, n > max_n → (∃ i, streetlight_length * i > road_length)) :=
sorry

end max_streetlights_l85_85751


namespace wheel_rpm_l85_85663

theorem wheel_rpm (radius_cm : ℕ) (speed_kmph : ℕ) : 
  radius_cm = 250 → speed_kmph = 66 → 
  (distance_per_min : ℕ := speed_kmph * 100000 / 60) →
  (circumference_cm : ℕ := 2 * 3.1416 * radius_cm) →
  distance_per_min / circumference_cm = 70.03 :=
begin
  sorry
end

end wheel_rpm_l85_85663


namespace number_and_sum_of_f2_l85_85577

def f (m : ℤ) : ℤ := sorry

axiom functional_eq (m n : ℤ) : f(m + n) + f(m * n - 1) = f(m) * f(n) + 2

theorem number_and_sum_of_f2 : 
  let n := 1 in 
  let s := 5 in 
  n * s = 5 :=
by sorry

end number_and_sum_of_f2_l85_85577


namespace probability_odd_product_sum_divisible_by_5_l85_85267

theorem probability_odd_product_sum_divisible_by_5 :
  (∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 20 ∧ 1 ≤ b ∧ b ≤ 20 ∧ a ≠ b ∧ (a * b % 2 = 1 ∧ (a + b) % 5 = 0)) →
  ∃ (p : ℚ), p = 3 / 95 :=
by
  sorry

end probability_odd_product_sum_divisible_by_5_l85_85267


namespace Milly_took_extra_balloons_l85_85603

theorem Milly_took_extra_balloons :
  let total_packs := 3 + 2
  let balloons_per_pack := 6
  let total_balloons := total_packs * balloons_per_pack
  let even_split := total_balloons / 2
  let Floretta_balloons := 8
  let Milly_extra_balloons := even_split - Floretta_balloons
  Milly_extra_balloons = 7 := by
  sorry

end Milly_took_extra_balloons_l85_85603


namespace probability_one_card_per_suit_l85_85885

theorem probability_one_card_per_suit :
  let total_cards := 52
  let total_suits := 4
  let total_draws := 4
  let first_card_prob := 1
  let second_card_prob := (13 / (total_cards - 1))
  let third_card_prob := (13 / (total_cards - 2 - 1))
  let fourth_card_prob := (13 / (total_cards - 3 - 1))
  in (first_card_prob * second_card_prob * third_card_prob * fourth_card_prob) = (2197 / 20825) :=
by 
  sorry

end probability_one_card_per_suit_l85_85885


namespace largest_w_for_11_200_factorial_l85_85884

theorem largest_w_for_11_200_factorial :
  ∃ w, 11^w ∣ Nat.factorial 200 ∧ ∀ w', 11^w' ∣ Nat.factorial 200 → w' ≤ 19 :=
begin
  sorry
end

end largest_w_for_11_200_factorial_l85_85884


namespace graduation_ceremony_chairs_l85_85176

theorem graduation_ceremony_chairs (num_graduates num_teachers: ℕ) (half_as_administrators: ℕ) :
  (∀ num_graduates = 50) →
  (∀ num_teachers = 20) →
  (∀ half_as_administrators = num_teachers / 2) →
  (2 * num_graduates + num_graduates + num_teachers + half_as_administrators = 180) :=
begin
  intros,
  sorry
end

end graduation_ceremony_chairs_l85_85176


namespace find_f_of_five_thirds_l85_85939

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem find_f_of_five_thirds (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_fun : ∀ x : ℝ, f (1 + x) = f (-x))
  (h_val : f (-1 / 3) = 1 / 3) : 
  f (5 / 3) = 1 / 3 :=
  sorry

end find_f_of_five_thirds_l85_85939


namespace pencil_pen_eraser_cost_l85_85640

-- Define the problem conditions and question
theorem pencil_pen_eraser_cost 
  (p q : ℝ)
  (h1 : 3 * p + 2 * q = 4.10)
  (h2 : 2 * p + 3 * q = 3.70) :
  p + q + 0.85 = 2.41 :=
sorry

end pencil_pen_eraser_cost_l85_85640


namespace sum_of_positive_integers_eq_zero_l85_85321

noncomputable def lcm (a b : ℕ) : ℕ := (a * b) / (Nat.gcd a b)

theorem sum_of_positive_integers_eq_zero : 
  ∑ n in Finset.range 1001, if lcm n 200 = Nat.gcd n 200 + 1000 then n else 0 = 0 := 
by
  sorry

end sum_of_positive_integers_eq_zero_l85_85321


namespace num_ways_to_divide_friends_l85_85524

theorem num_ways_to_divide_friends :
  (4 : ℕ) ^ 8 = 65536 := by
  sorry

end num_ways_to_divide_friends_l85_85524


namespace pirates_treasure_l85_85736

noncomputable def lcm_of_divisors_of_fifteen : ℕ :=
  List.lcm (List.finRange 15).map (λ k, 15 / (k + 1))

theorem pirates_treasure (x := 15^14) :
  x = lcm_of_divisors_of_fifteen →
  let initial_coins := x in
  let coins_before_last_pirate := initial_coins * (14.factorial / 15^14) in
  let coins_taken_by_last_pirate := coins_before_last_pirate in
  coins_taken_by_last_pirate = 87178291200 :=
begin
  sorry
end

end pirates_treasure_l85_85736


namespace semicircle_contains_three_of_four_points_hemisphere_contains_four_of_five_points_l85_85721

-- Definition and Theorem for Part 1
def circle : Type := sorry -- Define a circle type
def points_on_circle (p1 p2 p3 p4 : circle) : Prop := sorry -- Define 4 points on a circle
def exists_semicircle_contains_three (p1 p2 p3 p4 : circle) : Prop :=
  ∃ semicircle, ∀ p ∈ {p1, p2, p3, p4}, p ∈ semicircle → (p1 ∈ semicircle ∧ p2 ∈ semicircle ∧ p3 ∈ semicircle)

theorem semicircle_contains_three_of_four_points (p1 p2 p3 p4 : circle) (h : points_on_circle p1 p2 p3 p4) :
  exists_semicircle_contains_three p1 p2 p3 p4 :=
sorry

-- Definition and Theorem for Part 2
def sphere : Type := sorry -- Define a sphere type
def points_on_sphere (p1 p2 p3 p4 p5 : sphere) : Prop := sorry -- Define 5 points on a sphere
def exists_hemisphere_contains_four (p1 p2 p3 p4 p5 : sphere) : Prop :=
  ∃ hemisphere, ∀ p ∈ {p1, p2, p3, p4, p5}, p ∈ hemisphere → (p1 ∈ hemisphere ∧ p2 ∈ hemisphere ∧ p3 ∈ hemisphere ∧ p4 ∈ hemisphere)

theorem hemisphere_contains_four_of_five_points (p1 p2 p3 p4 p5 : sphere) (h : points_on_sphere p1 p2 p3 p4 p5) :
  exists_hemisphere_contains_four p1 p2 p3 p4 p5 :=
sorry

end semicircle_contains_three_of_four_points_hemisphere_contains_four_of_five_points_l85_85721


namespace cricketer_wickets_now_l85_85710

-- Conditions: Original average, match performance, and new average
variables (W R : ℕ)
variables (original_average : Float) (runs_in_match : ℕ) (wickets_in_match : ℕ)
variables (new_average_decrease : Float)

-- Assumptions (given conditions)
axiom h1 : original_average = 12.4
axiom h2 : runs_in_match = 26
axiom h3 : wickets_in_match = 5
axiom h4 : new_average_decrease = 0.4

-- Prove the number of wickets now
theorem cricketer_wickets_now : 
    let new_wickets := W + wickets_in_match in
    let new_average := original_average - new_average_decrease in
    (R / W = 12.4) ->
    ((R + runs_in_match) / new_wickets = 12.0) ->
    new_wickets = 90 :=
by
    sorry

end cricketer_wickets_now_l85_85710


namespace balls_count_neither_red_nor_blue_l85_85305

theorem balls_count_neither_red_nor_blue : 
  (total_balls red_balls blue_balls remaining_balls balls_neither_red_nor_blue : ℕ)
  (h1 : total_balls = 360) 
  (h2 : red_balls = total_balls / 4)
  (h3 : remaining_balls = total_balls - red_balls)
  (h4 : blue_balls = remaining_balls / 5)
  (h5 : balls_neither_red_nor_blue = total_balls - (red_balls + blue_balls)) :
  balls_neither_red_nor_blue = 216 := by
  sorry

end balls_count_neither_red_nor_blue_l85_85305


namespace count_zeros_after_decimal_point_l85_85641

theorem count_zeros_after_decimal_point :
  let x := 1 / (20 ^ 22)
  ∃ k : ℕ, k = 28 ∧ x = 9 * 10 ^ (-(k + 1)) + y ∧ |y| < 10 ^ (-(k + 1)) := 
begin
  sorry
end

end count_zeros_after_decimal_point_l85_85641


namespace num_rectangles_equilateral_triangle_l85_85209

theorem num_rectangles_equilateral_triangle (ABC : Triangle) (h : ABC.is_equilateral) :
  ∃ n : ℕ, n = 6 ∧ num_rectangles_in_plane_with_two_vertices_of_triangle_and_sides_parallel (ABC) n :=
by
  sorry

end num_rectangles_equilateral_triangle_l85_85209


namespace count_triangles_in_figure_l85_85514

/-- 
The figure is a rectangle divided into 8 columns and 2 rows with additional diagonal and vertical lines.
We need to prove that there are 76 triangles in total in the figure.
-/
theorem count_triangles_in_figure : 
  let columns := 8 
  let rows := 2 
  let num_triangles := 76 
  ∃ total_triangles, total_triangles = num_triangles :=
by
  sorry

end count_triangles_in_figure_l85_85514


namespace driving_hours_l85_85682

theorem driving_hours (days_in_week : ℕ) (initial_hours_per_day weekly_additional_hours : ℕ) :
  days_in_week = 7 → initial_hours_per_day = 2 → weekly_additional_hours = 6 → 
  2 * days_in_week + weekly_additional_hours = 20 →
  2 * (initial_hours_per_day * days_in_week + weekly_additional_hours) = 40 :=
by
  intros h_days h_initial h_weekly h_week_hours
  rw [h_days, h_initial, h_weekly, h_week_hours]
  simp
  sorry

end driving_hours_l85_85682


namespace find_digits_for_multiple_of_11_l85_85102

theorem find_digits_for_multiple_of_11 :
  {a b : ℕ // a < 10 ∧ b < 10 ∧ (∃k:ℤ, a - b - 8 = 11 * k)} ->
  (a, b) ∈ {(8, 0), (9, 1), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9)} :=
begin
  assume h : {a b : ℕ // a < 10 ∧ b < 10 ∧ (a - (b + 15)) % 11 = -8 % 11},
  sorry
end

end find_digits_for_multiple_of_11_l85_85102


namespace travel_distance_l85_85605

-- Define the Cartesian points for Nia, Mia, and Lia
def Nia : (ℝ × ℝ) := (10, -30)
def Mia : (ℝ × ℝ) := (5, 22)
def Lia : (ℝ × ℝ) := (6, 8)

-- Noncomputable definition for the midpoint of Nia and Mia
noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Noncomputable definition for calculating the Euclidean distance
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

-- Main theorem: proving the correct total distance traveled by Nia and Mia
theorem travel_distance:
  distance (midpoint Nia Mia) Lia = real.sqrt 146.25 := by
  sorry

end travel_distance_l85_85605


namespace even_n_vector_sum_zero_odd_n_vector_sum_finite_zero_points_l85_85468

-- Definitions from conditions
variables {n : ℕ} (A : Fin n → ℝ × ℝ) (M : ℝ × ℝ)

-- Question a
theorem even_n_vector_sum_zero (h_even : Even n) :
  ∃ (ε : Fin n → ℝ), (∀ i, ε i = 1 ∨ ε i = -1) ∧ Σ i, ε i • (A i -ᵥ M) = 0 := 
sorry

-- Question b
theorem odd_n_vector_sum_finite_zero_points (h_odd : Odd n) :
  ∃ (finite_set : Set (ℝ × ℝ)), (finite_set.finite ∧ ∃ M ∈ finite_set, Σ i, ε i • (A i -ᵥ M) = 0) :=
sorry

end even_n_vector_sum_zero_odd_n_vector_sum_finite_zero_points_l85_85468


namespace abc_line_intersects_cd_at_l85_85945

def point : Type := ℝ × ℝ

def A : point := (0, 0)
def B : point := (2, 3)
def C : point := (6, 3)
def D : point := (7, 0)

def line_through (p1 p2 : point) : (ℝ × ℝ → Prop) :=
  λ (x y : ℝ), (y - p1.2) / (x - p1.1) = (p2.2 - p1.2) / (p2.1 - p1.1)

def divides_into_equal_areas (line : ℝ × ℝ → Prop) : Prop :=
  let area := 6 in
  -- Instruction to find the area bounded by line with the quadrilateral
  sorry

def intersection (line1 line2 : ℝ × ℝ → Prop) : point :=
  -- Function to find the intersection point of two lines
  sorry

theorem abc_line_intersects_cd_at : 
  ∃ p q r s : ℕ, (
    intersection (line_through A some_line_through_A) (line_through C D) = (p / q, r / s) ∧ 
    p + q + r + s = 11) := 
sorry

end abc_line_intersects_cd_at_l85_85945


namespace circle_equation_standard_form_l85_85299

theorem circle_equation_standard_form (x y : ℝ) :
  (∃ (center : ℝ × ℝ), center.1 = -1 ∧ center.2 = 2 * center.1 ∧ (center.2 = -2) ∧ (center.1 + 1)^2 + center.2^2 = 4 ∧ (center.1 = -1) ∧ (center.2 = -2)) ->
  (x + 1)^2 + (y + 2)^2 = 4 :=
sorry

end circle_equation_standard_form_l85_85299


namespace greatest_number_of_pieces_leftover_l85_85137

theorem greatest_number_of_pieces_leftover (y : ℕ) (q r : ℕ) 
  (h : y = 6 * q + r) (hrange : r < 6) : r = 5 := sorry

end greatest_number_of_pieces_leftover_l85_85137


namespace cost_price_of_cricket_bat_for_A_l85_85386

-- Define the cost price of the cricket bat for A as a variable
variable (CP_A : ℝ)

-- Define the conditions given in the problem
def condition1 := CP_A * 1.20 -- B buys at 20% profit
def condition2 := CP_A * 1.20 * 1.25 -- B sells at 25% profit
def totalCost := 231 -- C pays $231

-- The theorem we need to prove
theorem cost_price_of_cricket_bat_for_A : (condition2 = totalCost) → CP_A = 154 := by
  intros h
  sorry

end cost_price_of_cricket_bat_for_A_l85_85386


namespace sin_pi_div_2_minus_pi_div_6_l85_85673

noncomputable def sin_diff (α β : ℝ) : ℝ := Real.sin (α - β)

theorem sin_pi_div_2_minus_pi_div_6 : sin_diff (Real.pi / 2) (Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end sin_pi_div_2_minus_pi_div_6_l85_85673


namespace min_one_over_a_plus_one_over_b_min_b_over_a_cubed_plus_a_over_b_cubed_l85_85458

-- Define conditions
variables (a b : ℝ)
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom sum_squares : a^2 + b^2 = 1

-- Question 1: Prove the minimum value of (1/a) + (1/b) is 2√2
theorem min_one_over_a_plus_one_over_b : (∃ x, x = 1/a + 1/b ∧ ∀ y, y ≤ x → y = 2*real.sqrt 2) :=
  sorry

-- Question 2: Prove the minimum value of (b/a^3) + (a/b^3) is 4
theorem min_b_over_a_cubed_plus_a_over_b_cubed : (∃ x, x = b/a^3 + a/b^3 ∧ ∀ y, y ≤ x → y = 4) :=
  sorry

end min_one_over_a_plus_one_over_b_min_b_over_a_cubed_plus_a_over_b_cubed_l85_85458


namespace visible_shaded_area_l85_85898

noncomputable def side_of_small_square : ℝ := 3
noncomputable def grid_size : ℕ := 6
noncomputable def total_squares : ℕ := grid_size * grid_size
noncomputable def side_of_unshaded_square : ℝ := 1.5
noncomputable def A : ℝ := 324
noncomputable def B : ℝ := 11.25
noncomputable def phi : ℝ := 1
noncomputable def value_to_prove : ℝ := A + B

theorem visible_shaded_area (A B phi : ℝ) (hA : A = 324) (hB : B = 11.25) (hphi : phi = 1) :
    A + B = 335.25 :=
by
  rw [hA, hB]
  exact rfl

end visible_shaded_area_l85_85898


namespace propositions_valid_l85_85110

def proposition1 (m : ℝ) : Prop :=
  (m > 0) → ∃ x : ℝ, x^2 + 2 * x - m = 0

def proposition2 : Prop :=
  ∃ x : ℝ, x = 1 ∧ x^2 - 3 * x + 2 = 0 ∧ ¬(∀ x : ℝ, x^2 - 3 * x + 2 = 0 → x = 1)

def proposition3 : Prop :=
  (∀ (a b c d : ℝ), a = c ∧ b = d → a^2 + b^2 = c^2 + d^2) →

  (∀ (a b c d : ℝ), (a^2 + b^2 = c^2 + d^2) → a = c ∧ b = d)

def proposition4 : Prop :=
  (¬ (∀ x : ℝ, x^2 + x + 3 > 0)) ↔ ∃ x : ℝ, x^2 + x + 3 ≤ 0

theorem propositions_valid :
  proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ proposition4 := by
  sorry

end propositions_valid_l85_85110


namespace dihedral_angle_is_60_degrees_l85_85191

def point (x y z : ℝ) := (x, y, z)

noncomputable def dihedral_angle (P Q R S T : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem dihedral_angle_is_60_degrees :
  dihedral_angle 
    (point 1 0 0)  -- A
    (point 1 1 0)  -- B
    (point 0 0 0)  -- D
    (point 1 0 1)  -- A₁
    (point 0 0 1)  -- D₁
 = 60 :=
sorry

end dihedral_angle_is_60_degrees_l85_85191


namespace allowance_spent_on_burgers_l85_85525

theorem allowance_spent_on_burgers (total_allowance : ℕ) (frac_movies frac_music frac_ice_cream : ℝ) :
  total_allowance = 30 → frac_movies = 1 / 3 → frac_music = 3 / 10 → frac_ice_cream = 1 / 5 →
  let spent_on_movies := frac_movies * total_allowance,
      spent_on_music := frac_music * total_allowance,
      spent_on_ice_cream := frac_ice_cream * total_allowance,
      total_spent := spent_on_movies + spent_on_music + spent_on_ice_cream,
      spent_on_burgers := total_allowance - total_spent
  in spent_on_burgers = 5 :=
by
  intros
  sorry

end allowance_spent_on_burgers_l85_85525


namespace bound_on_ai_l85_85155

theorem bound_on_ai (n : ℕ) (a : ℕ → ℕ) (k : ℕ) :
  (n ≥ a 1) ∧ (∀ i, 1 ≤ i ∧ i < k → a i > a (i + 1)) ∧ (∀ i j, 1 ≤ i ∧ i ≤ k ∧ 1 ≤ j ∧ j ≤ k → Nat.lcm (a i) (a j) ≤ n) →
  ∀ i, 1 ≤ i ∧ i ≤ k → i * a i ≤ n :=
by
  sorry

end bound_on_ai_l85_85155


namespace find_number_l85_85985

theorem find_number 
  (x : ℚ) 
  (h : (3 / 4) * x - (8 / 5) * x + 63 = 12) : 
  x = 60 := 
by
  sorry

end find_number_l85_85985


namespace martha_initial_crayons_l85_85238

theorem martha_initial_crayons : ∃ (x : ℕ), (x / 2 + 20 = 29) ∧ x = 18 :=
by
  sorry

end martha_initial_crayons_l85_85238


namespace union_A_B_l85_85477

open Set

def A := {x : ℝ | x * (x - 2) < 3}
def B := {x : ℝ | 5 / (x + 1) ≥ 1}
def U := {x : ℝ | -1 < x ∧ x ≤ 4}

theorem union_A_B : A ∪ B = U := 
sorry

end union_A_B_l85_85477


namespace lcm_gcd_product_l85_85697

theorem lcm_gcd_product (a b : ℕ) (ha : a = 36) (hb : b = 60) : 
  Nat.lcm a b * Nat.gcd a b = 2160 :=
by
  rw [ha, hb]
  sorry

end lcm_gcd_product_l85_85697


namespace find_P_l85_85300

theorem find_P (P : ℕ) (h : P^2 + P = 30) : P = 5 :=
sorry

end find_P_l85_85300


namespace radius_of_circle_l85_85655

theorem radius_of_circle (r : ℝ) (h : 3 * (2 * Real.pi * r) = 2 * (Real.pi * r ^ 2)) : r = 3 := by
  sorry

end radius_of_circle_l85_85655


namespace rational_inequality_solution_l85_85800

theorem rational_inequality_solution (x : ℝ) (h : x ≠ 4) :
  (4 < x ∧ x ≤ 5) ↔ (x - 2) / (x - 4) ≤ 3 :=
sorry

end rational_inequality_solution_l85_85800


namespace divides_n_plus_one_l85_85207

theorem divides_n_plus_one (p q n : ℕ) (hp : p.prime) (hq : q.prime) (hpq : p ≠ q) 
(h1 : p * q ∣ n ^ (p * q) + 1) (h2 : p^3 * q^3 ∣ n ^ (p * q) + 1) : 
  p^2 ∣ n + 1 ∨ q^2 ∣ n + 1 :=
by
  sorry

end divides_n_plus_one_l85_85207


namespace probability_units_digit_of_2_pow_a_sub_5_pow_b_has_units_digit_3_l85_85809

def pow_units_cycle (base : ℕ) : List ℕ :=
  match base % 10 with
  | 2 => [2, 4, 8, 6]
  | 5 => [5]
  | _ => []

noncomputable def probability_units_digit_3 : ℚ :=
  let a_vals := List.range 50
  let b_vals := List.range 50
  let counts := a_vals.filter (fun a =>
    (2^a % 10 - 5^5 % 10) % 10 == 3).length
  counts / (50 * 50)

theorem probability_units_digit_of_2_pow_a_sub_5_pow_b_has_units_digit_3 :
  probability_units_digit_3 = 6 / 25 := 
sorry

end probability_units_digit_of_2_pow_a_sub_5_pow_b_has_units_digit_3_l85_85809


namespace min_r_for_three_coloring_l85_85933

-- Define the set S as the set of points inside and on the boundary of a regular hexagon with side length 1
def S : set (ℝ × ℝ) := {p | p ∈ hexagon_with_side_1}

-- Define the property P such that a configuration satisfies three-coloring constraints
def P (r : ℝ) : Prop :=
  ∃ (f : (ℝ × ℝ) → fin 3), ∀ (p q : ℝ × ℝ), p ∈ S → q ∈ S → f p = f q → dist p q < r

-- Main theorem stating the minimum r for which P(r) holds
theorem min_r_for_three_coloring : 
  ∃ min_r, P min_r ∧ (∀ r, P r → r ≥ min_r) ∧ min_r = 3 / 2 :=
sorry

end min_r_for_three_coloring_l85_85933


namespace shuttle_buses_transport_435_total_l85_85388

noncomputable def total_tourists : ℕ :=
  let n := 6
  let a := 80
  let d := -3
  n * a + n * (n - 1) * d / 2

theorem shuttle_buses_transport_435_total:
  total_tourists = 435 :=
by
  sorry

end shuttle_buses_transport_435_total_l85_85388


namespace complement_A_eq_l85_85506

def set_A : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

theorem complement_A_eq :
  (set.univ \ set_A) = (set.Iic 0 ∪ set.Ici 1) := 
by sorry

end complement_A_eq_l85_85506


namespace num_pos_multiples_of_six_is_150_l85_85512

theorem num_pos_multiples_of_six_is_150 : 
  ∃ (n : ℕ), (∀ k, (n = 150) ↔ (102 + (k - 1) * 6 = 996 ∧ 102 ≤ 6 * k ∧ 6 * k ≤ 996)) :=
sorry

end num_pos_multiples_of_six_is_150_l85_85512


namespace find_a_l85_85112

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.logb 3 x else 3^x

theorem find_a (a : ℝ) (h : f a = 1/3) : a = Real.sqrtN 3 3 ∨ a = -1 := by
  sorry

end find_a_l85_85112


namespace shaded_areas_are_different_l85_85786

theorem shaded_areas_are_different :
  let shaded_area_I := 3 / 8
  let shaded_area_II := 1 / 3
  let shaded_area_III := 1 / 2
  (shaded_area_I ≠ shaded_area_II) ∧ (shaded_area_I ≠ shaded_area_III) ∧ (shaded_area_II ≠ shaded_area_III) :=
by
  sorry

end shaded_areas_are_different_l85_85786


namespace measure_of_angle_R_l85_85187

variable (S T A R : ℝ) -- Represent the angles as real numbers.

-- The conditions given in the problem.
axiom angles_congruent : S = T ∧ T = A ∧ A = R
axiom angle_A_equals_angle_S : A = S

-- Statement: Prove that the measure of angle R is 108 degrees.
theorem measure_of_angle_R : R = 108 :=
by
  sorry

end measure_of_angle_R_l85_85187


namespace f_value_third_quadrant_l85_85823

noncomputable def f (α : ℝ) : ℝ :=
  (Real.cos (Real.pi / 2 + α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.sin (-Real.pi - α) * Real.sin (3 * Real.pi / 2 + α))

theorem f_value_third_quadrant (α : ℝ) (h1 : (3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)) (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 :=
sorry

end f_value_third_quadrant_l85_85823


namespace minimum_distance_parabola_line_l85_85121

noncomputable def distance_from_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / real.sqrt (A^2 + B^2)

theorem minimum_distance_parabola_line :
  let parabola_point (t : ℝ) := (t, t^2 / 2)
      distance (t : ℝ) := distance_from_point_to_line t (t^2 / 2) (-1) 1 2 in
  ∃ (min_d : ℝ), min_d = (3 * real.sqrt 2) / 4 ∧
    ∀ t : ℝ, distance t ≥ min_d :=
begin
  sorry
end

end minimum_distance_parabola_line_l85_85121


namespace max_diff_consecutive_slightly_unlucky_l85_85951

def is_slightly_unlucky (n : ℕ) : Prop := (n.digits 10).sum % 13 = 0

theorem max_diff_consecutive_slightly_unlucky :
  ∃ n m : ℕ, is_slightly_unlucky n ∧ is_slightly_unlucky m ∧ (m > n) ∧ ∀ k, (is_slightly_unlucky k ∧ k > n ∧ k < m) → false → (m - n) = 79 :=
sorry

end max_diff_consecutive_slightly_unlucky_l85_85951


namespace vertex_in_fourth_quadrant_l85_85146

theorem vertex_in_fourth_quadrant (a : ℝ) (ha : a < 0) :  
  let x_vertex := -a / 4
  let y_vertex := (-40 - a^2) / 8
  x_vertex > 0 ∧ y_vertex < 0 := by
  let x_vertex := -a / 4
  let y_vertex := (-40 - a^2) / 8
  have hx : x_vertex > 0 := by sorry
  have hy : y_vertex < 0 := by sorry
  exact And.intro hx hy

end vertex_in_fourth_quadrant_l85_85146


namespace fraction_power_mult_equality_l85_85414

-- Define the fraction and the power
def fraction := (1 : ℚ) / 3
def power : ℚ := fraction ^ 4

-- Define the multiplication
def result := 8 * power

-- Prove the equality
theorem fraction_power_mult_equality : result = 8 / 81 := by
  sorry

end fraction_power_mult_equality_l85_85414


namespace cole_average_speed_l85_85024

theorem cole_average_speed (speed_to_work : ℝ) (total_time : ℝ) (time_to_work_minutes : ℝ) (correct_speed : ℝ) :
    speed_to_work = 80 →
    total_time = 2 →
    time_to_work_minutes = 72 →
    correct_speed = 120 →
    let time_to_work := time_to_work_minutes / 60 in
    let distance_to_work := speed_to_work * time_to_work in
    let time_back_home := total_time - time_to_work in
    let speed_back_home := distance_to_work / time_back_home in 
    speed_back_home = correct_speed := 
begin
  intros h1 h2 h3 h4,
  sorry
end

end cole_average_speed_l85_85024


namespace sqrt_sum_of_products_le_sum_of_sqrt_products_l85_85981

theorem sqrt_sum_of_products_le_sum_of_sqrt_products (n : ℕ) (x y : ℕ → ℝ) (hx : ∀ i, 0 < x i) (hy : ∀ i, 0 < y i) :
  ∑ i in Finset.range n, Real.sqrt (x i * y i) ≤ Real.sqrt (∑ i in Finset.range n, x i) * Real.sqrt (∑ i in Finset.range n, y i) :=
sorry

end sqrt_sum_of_products_le_sum_of_sqrt_products_l85_85981


namespace ratio_of_triangle_side_to_rectangle_width_l85_85010

theorem ratio_of_triangle_side_to_rectangle_width
  (t w : ℕ)
  (ht : 3 * t = 24)
  (hw : 6 * w = 24) :
  t / w = 2 := by
  sorry

end ratio_of_triangle_side_to_rectangle_width_l85_85010


namespace find_multiple_of_t_l85_85165

variable (t : ℝ)
variable (x y : ℝ)

theorem find_multiple_of_t (h1 : x = 1 - 4 * t)
  (h2 : ∃ m : ℝ, y = m * t - 2)
  (h3 : t = 0.5)
  (h4 : x = y) : ∃ m : ℝ, (m = 2) :=
by
  sorry

end find_multiple_of_t_l85_85165


namespace total_dress_designs_l85_85364

theorem total_dress_designs:
  let colors := 5
  let patterns := 4
  let sleeve_lengths := 2
  colors * patterns * sleeve_lengths = 40 := 
by
  sorry

end total_dress_designs_l85_85364


namespace length_of_MN_l85_85530

theorem length_of_MN (x1 y1 z1 x2 y2 z2 : ℝ) (hM : x1 = 3 ∧ y1 = 4 ∧ z1 = 1)
  (hN : x2 = 0 ∧ y2 = 0 ∧ z2 = 1) : 
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2) = 5 :=
by 
  sorry

end length_of_MN_l85_85530


namespace find_error_bit_l85_85353

variable (x : Fin 7 → Bool) (y : Fin 7 → Bool)
variable (k : ℕ)

def parity_check_1 : Prop := x 3 ⊕ x 4 ⊕ x 5 ⊕ x 6 = true
def parity_check_2 : Prop := x 1 ⊕ x 2 ⊕ x 5 ⊕ x 6 = true
def parity_check_3 : Prop := x 0 ⊕ x 2 ⊕ x 4 ⊕ x 6 = true

def bit_flip (b : Bool) : Bool := bnot b

def y_from_x (k : Fin 7) : Fin 7 → Bool :=
  λ i, if i = k then bit_flip (x i) else x i

theorem find_error_bit :
  (parity_check_1 x) → (parity_check_2 x) → (parity_check_3 x) → 
  y = y_from_x x (Fin.mk 4 sorry) →
  k = 5 :=
by
  sorry

end find_error_bit_l85_85353


namespace ordered_pairs_satisfy_conditions_l85_85046

theorem ordered_pairs_satisfy_conditions :
  ∀ (a b : ℕ), 0 < a → 0 < b → (a^2 + b^2 + 25 = 15 * a * b) → Nat.Prime (a^2 + a * b + b^2) →
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by
  intros a b ha hb h1 h2
  sorry

end ordered_pairs_satisfy_conditions_l85_85046


namespace minimum_value_f_l85_85070

def f (x : ℝ) : ℝ :=
  (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_f (x : ℝ) (hx : x ≥ 5 / 2) : ∃ m, m = 1 ∧ ∀ y, y ≥ 5 / 2 → f y ≥ m :=
by 
  sorry

end minimum_value_f_l85_85070


namespace cheerleader_total_l85_85637

theorem cheerleader_total 
  (size2 : ℕ)
  (size6 : ℕ)
  (size12 : ℕ)
  (h1 : size2 = 4)
  (h2 : size6 = 10)
  (h3 : size12 = size6 / 2) :
  size2 + size6 + size12 = 19 :=
by
  sorry

end cheerleader_total_l85_85637


namespace determine_c_l85_85794

theorem determine_c : ∃ c : ℚ, ∀ x : ℚ, 9 * x^2 - 21 * x + c = (3 * x - 7 / 2)^2 := 
by
  use 49 / 4
  intro x
  calc
    9 * x^2 - 21 * x + 49 / 4 = (3 * x - 7 / 2) * (3 * x - 7 / 2) : sorry

end determine_c_l85_85794


namespace quadratic_polynomial_unique_l85_85053

noncomputable def q (x : ℝ) : ℝ := x^2 + 1

theorem quadratic_polynomial_unique :
  q (-2) = 5 ∧ q (1) = 2 ∧ q (3) = 10 :=
by 
  split; 
  { norm_num;
    -- Verifying each condition
  }

end quadratic_polynomial_unique_l85_85053


namespace probability_factors_less_than_l85_85317

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

def factors_less_than_x (n : ℕ) (x : ℕ) : finset ℕ := 
  (finset.range x).filter (λ d, n % d = 0)

theorem probability_factors_less_than (n x : ℕ) (h : x > 0) (h90 : n = 90) :
  (factors_less_than_x n x).card / (num_factors n) = 1 / 3 :=
by
  sorry

end probability_factors_less_than_l85_85317


namespace smallest_norm_v_l85_85213

noncomputable def v : Type := ℝ × ℝ

theorem smallest_norm_v
  (v : v)
  (h : ∥v + ⟨4, 2⟩∥ = 10) :
  ∥v∥ = 10 - 2 * Real.sqrt 5 :=
sorry

end smallest_norm_v_l85_85213


namespace factor_expression_l85_85025

theorem factor_expression (a : ℝ) :
  (8 * a^3 + 105 * a^2 + 7) - (-9 * a^3 + 16 * a^2 - 14) = a^2 * (17 * a + 89) + 21 :=
by
  sorry

end factor_expression_l85_85025


namespace base4_to_base10_conversion_l85_85034

theorem base4_to_base10_conversion : (2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582) :=
by {
  -- The proof is omitted
  sorry
}

end base4_to_base10_conversion_l85_85034


namespace triangles_share_incenter_l85_85549

theorem triangles_share_incenter {A B C P O E F D K : Type*} 
  [has_inscribed_triangle P A B C]
  [circle_passing_through_two_points O A B]
  [intersects_circle O AC E]
  [intersects_circle O BC F]
  [line_intersect AF DE D]
  [line_intersect OD P K] :
  ∃ I, is_incenter I K B E ∧ is_incenter I K A F :=
by sorry

end triangles_share_incenter_l85_85549


namespace sequence_identity_l85_85826

open Nat

-- Define the base function
def f1 (x : ℝ) : ℝ := x / Real.sqrt (1 + x^2)

-- Define the recursive sequence of functions
def f (n : ℕ) (x : ℝ) : ℝ :=
  if n = 0 then x else
  (List.repeat f1 n).foldl (λ acc fn => fn acc) x

-- Define the conjectured expression
def f_conjecture (n : ℕ) (x : ℝ) : ℝ := x / Real.sqrt (1 + n * x^2)

-- State the theorem to prove
theorem sequence_identity (n : ℕ) (x : ℝ) (h : x > 0) : f (n + 1) x = f_conjecture (n + 1) x :=
sorry

end sequence_identity_l85_85826


namespace carol_has_35_nickels_l85_85269

def problem_statement : Prop :=
  ∃ (n d : ℕ), 5 * n + 10 * d = 455 ∧ n = d + 7 ∧ n = 35

theorem carol_has_35_nickels : problem_statement := by
  -- Proof goes here
  sorry

end carol_has_35_nickels_l85_85269


namespace number_of_possible_D_values_l85_85188

-- Definitions of distinct digits and valid equation
def distinct (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

def valid_digits (x : ℕ) : Prop := x >= 0 ∧ x ≤ 9

def valid_sum (A B C D : ℕ) : Prop :=
  valid_digits A ∧ valid_digits B ∧ valid_digits C ∧ valid_digits D ∧
  distinct A B C D ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + C) + 
  (10000 * C + 1000 * B + 100 * A + 10 * D + B) = 
  (10000 * D + 1000 * B + 100 * D + 10 * D + D)

-- The main theorem statement
theorem number_of_possible_D_values : 
  ∃ (n : ℕ), n = 10 ∧ ∀ D : ℕ, valid_digits D → ("%1 ≤ D ∧ D ≤ n" implies valid_sum _ _ _ D) := sorry

end number_of_possible_D_values_l85_85188


namespace unique_zero_in_interval_f_geq_e_inv_l85_85111

-- Define the function f(x)
def f (a x : ℝ) : ℝ := (x ^ 2 - x + 1) * Real.exp (-x) + (a / 2) * x ^ 2 - a * x + (a / 2)

-- Proof Problem 1
theorem unique_zero_in_interval (a : ℝ) (h : a = 0) :
  ∃! x ∈ Icc (-2 : ℝ) 1, (f a x - 2 = 0) :=
sorry

-- Proof Problem 2
theorem f_geq_e_inv (a : ℝ) (h : a ≥ 1 / Real.exp 3) :
  ∀ x ≥ 1, f a x ≥ 1 / Real.exp 1 :=
sorry

end unique_zero_in_interval_f_geq_e_inv_l85_85111


namespace A_2019th_number_l85_85313

/--
Three children named A, B, and C play a counting game where:
1. A starts with the number 1.
2. B follows with the next two numbers.
3. C follows with the next three numbers, and so on.
Each child says one more number than the previous one in a cyclic manner up to 10000.

This process continues and we want to find the 2019th number A says.

Theorem: The 2019th number A says is 5979.
-/
theorem A_2019th_number : 
  (∃ f : ℕ → ℕ, f 2019 = 5979 ∧ ∀ n, (if n % 3 = 0 then f n = n div 3 + 1 * (n - 1)
                                       -- Other corresponding sequence conditions would be defined here.
                                       )
                                      ) :=
sorry

end A_2019th_number_l85_85313


namespace inscribed_circle_in_pyramid_base_l85_85824

theorem inscribed_circle_in_pyramid_base
  (pyramid : Type) [quadrilateral_pyramid pyramid]
  (sphere : Type) [inscribed_sphere sphere pyramid]
  (center_on_height : center_of_sphere_on_height sphere) :
  ∃ (circle : Type), inscribed_circle_in_base circle pyramid :=
sorry

end inscribed_circle_in_pyramid_base_l85_85824


namespace tangent_lines_eqn_l85_85438

-- Define the center and radius of the circle
def center : (ℝ × ℝ) := (0, 0)
def radius : ℝ := sqrt 5

-- Define the equation of the circle
def circle (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line parallel to x + 2y + 1 = 0
def parallel_line (x y b : ℝ) : Prop := x + 2 * y + b = 0

-- Define the distance from a point to a line
def distance_from_origin (b : ℝ) : ℝ := |b| / sqrt (1^2 + 2^2)

-- Define the tangency condition
def is_tangent (b : ℝ) : Prop := distance_from_origin b = sqrt 5

-- Theorem stating the problem
theorem tangent_lines_eqn (b : ℝ) :
  (is_tangent b ↔ b = 5 ∨ b = -5) :=
sorry

end tangent_lines_eqn_l85_85438


namespace gcd_le_two_l85_85282

theorem gcd_le_two (a m n : ℕ) (h1 : a > 0) (h2 : m > 0) (h3 : n > 0) (h4 : Odd n) :
  Nat.gcd (a^n - 1) (a^m + 1) ≤ 2 := 
sorry

end gcd_le_two_l85_85282


namespace proof_problem_l85_85495

def f (x : ℝ) : ℝ := 1 + Real.tan x

theorem proof_problem (a : ℝ) (h : f a = 3) : f (-a) = -1 := by
  sorry

end proof_problem_l85_85495


namespace arithmetic_sequence_common_difference_l85_85542

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Statement of the problem
theorem arithmetic_sequence_common_difference
  (h1 : a 2 + a 6 = 8)
  (h2 : a 3 + a 4 = 3)
  (h_arith : ∀ n, a (n+1) = a n + d) :
  d = 5 := by
  sorry

end arithmetic_sequence_common_difference_l85_85542


namespace two_subsets_count_l85_85311

-- Definitions from the problem conditions
def S : Set (Fin 5) := {0, 1, 2, 3, 4}

-- Main statement
theorem two_subsets_count : 
  (∃ A B : Set (Fin 5), A ∪ B = S ∧ A ∩ B = {a, b} ∧ A ≠ B) → 
  (number_of_ways = 40) :=
sorry

end two_subsets_count_l85_85311


namespace largest_integer_is_190_l85_85997

theorem largest_integer_is_190 (A B C D : ℤ) 
  (h1 : A < B) (h2 : B < C) (h3 : C < D) 
  (h4 : (A + B + C + D) / 4 = 76) 
  (h5 : A = 37) 
  (h6 : B = 38) 
  (h7 : C = 39) : 
  D = 190 := 
sorry

end largest_integer_is_190_l85_85997


namespace point_slope_form_of_perpendicular_line_l85_85850

theorem point_slope_form_of_perpendicular_line :
  ∀ (l1 l2 : ℝ → ℝ) (P : ℝ × ℝ),
    (l2 x = x + 1) →
    (P = (2, 1)) →
    (∀ x, l2 x = -1 * l1 x) →
    (∀ x, l1 x = -x + 3) :=
by
  intros l1 l2 P h1 h2 h3
  sorry

end point_slope_form_of_perpendicular_line_l85_85850


namespace find_f_neg_19_over_3_l85_85487

noncomputable def f : ℝ → ℝ :=
  λ x, if 0 < x ∧ x < 1 then 8^x else
         if x + 2 ≠ x then f (x - 2 * (floor (x / 2))) else sorry

theorem find_f_neg_19_over_3 :
  (∀ x : ℝ, f (x + 2) = f x) →
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, 0 < x → x < 1 → f x = 8^x) →
  f (-19/3) = -2 :=
by
  intros h1 h2 h3
  sorry

end find_f_neg_19_over_3_l85_85487


namespace consumption_order_l85_85422

theorem consumption_order :
  let west_consumption := 21428
  let non_west_consumption := 26848.55
  let russia_consumption := 302790.13
  (west_consumption < non_west_consumption) ∧ (non_west_consumption < russia_consumption) :=
by
  let west_consumption := 21428
  let non_west_consumption := 26848.55
  let russia_consumption := 302790.13
  have h1 : west_consumption < non_west_consumption := sorry
  have h2 : non_west_consumption < russia_consumption := sorry
  exact ⟨h1, h2⟩

end consumption_order_l85_85422


namespace smallest_possible_obscured_number_l85_85393

theorem smallest_possible_obscured_number (a b : ℕ) (cond : 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) :
  2 * a = b - 9 →
  42 + 25 + 56 + 10 * a + b = 4 * (4 + 2 + 2 + 5 + 5 + 6 + a + b) →
  10 * a + b = 79 :=
sorry

end smallest_possible_obscured_number_l85_85393


namespace exists_such_that_f_s_lt_zero_l85_85813

variables {a b c d t : ℝ} (h_not_all_zero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0)
variables (h_ft_eq_4a : a + b * cos (2 * t) + c * sin (5 * t) + d * cos (8 * t) = 4 * a)

noncomputable def f (x : ℝ) : ℝ := a + b * cos (2 * x) + c * sin (5 * x) + d * cos (8 * x)

theorem exists_such_that_f_s_lt_zero : ∃ s : ℝ, f a b c d s < 0 := 
sorry

end exists_such_that_f_s_lt_zero_l85_85813


namespace min_value_l85_85588

noncomputable def minimum_expression_value (a b c d e f : ℝ) (h_pos : ∀ x ∈ [a, b, c, d, e, f], 0 < x)
(h_sum : a + b + c + d + e + f = 8) : ℝ :=
  (1/a) + (9/b) + (4/c) + (25/d) + (16/e) + (49/f)

theorem min_value (a b c d e f : ℝ) : 
  (∀ x ∈ [a, b, c, d, e, f], 0 < x) → 
  a + b + c + d + e + f = 8 → 
  minimum_expression_value a b c d e f (by intros x h; exact sorry) (by simp [h_sum]; exact sorry) = 1352 := 
sorry

end min_value_l85_85588


namespace solution_set_l85_85220

variable {f : ℝ → ℝ}

-- Conditions
axiom differentiable_f : ∀ x < 0, differentiable_at ℝ f x
axiom derivative_f : ∀ x, 0 > x → deriv f x = f' x
axiom inequality_condition : ∀ x, 0 > x → 2 * f x + x * f' x > x^2

theorem solution_set (x : ℝ) :
  ((x + 2016)^2 * f (x + 2016) - 9 * f (-3) > 0) ↔ (x < -2019) :=
by
  sorry

end solution_set_l85_85220


namespace sqrt_8_consecutive_integers_l85_85150

theorem sqrt_8_consecutive_integers (a b : ℤ) (h_consec: b = a + 1) (h : (a : ℝ) < Real.sqrt 8 ∧ Real.sqrt 8 < (b : ℝ)) : b^a = 9 :=
by
  sorry

end sqrt_8_consecutive_integers_l85_85150


namespace oranges_per_crate_correct_l85_85675

def total_crates : ℕ := 12
def boxes : ℕ := 16
def nectarines_per_box : ℕ := 30
def total_fruit : ℕ := 2280
def total_nectarines := boxes * nectarines_per_box
def total_oranges := total_fruit - total_nectarines
def oranges_per_crate := total_oranges / total_crates

theorem oranges_per_crate_correct : oranges_per_crate = 150 := by
  unfold total_nectarines
  unfold total_oranges
  unfold oranges_per_crate
  calc
    oranges_per_crate = (total_fruit - (boxes * nectarines_per_box)) / total_crates := by rfl
    ... = (2280 - (16 * 30)) / 12 := by rfl
    ... = (2280 - 480) / 12 := by rfl
    ... = 1800 / 12 := by rfl
    ... = 150 := by rfl

end oranges_per_crate_correct_l85_85675


namespace hyperbola_eccentricity_l85_85466

theorem hyperbola_eccentricity (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : 
    let c := sqrt (a^2 + b^2),
        f1 := (c, 0),
        f2 := (-c, 0),
        A := (some A_point : R^2),
        B := (some B_point : R^2)
    in
    ( dist A f1 > 0 ∧ dist A f2 > 0 ) ∧ 
    ( A_point • f1 = 0 ) ∧ 
    ( f2 + 2 • A = 0 ) →
    ( c / a = sqrt(17) / 3 ) :=
by
  sorry

end hyperbola_eccentricity_l85_85466


namespace marbles_per_boy_l85_85538

theorem marbles_per_boy (boys marbles : ℕ) (h1 : boys = 5) (h2 : marbles = 35) : marbles / boys = 7 := by
  sorry

end marbles_per_boy_l85_85538


namespace trapezoid_parallel_segment_length_l85_85049

variables (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)

theorem trapezoid_parallel_segment_length : 
  let L := (2 * a * b) / (a + b) in L =
  (2 * a * b) / (a + b) := by 
  sorry

end trapezoid_parallel_segment_length_l85_85049


namespace minimum_value_of_quadratic_l85_85532

def quadratic_function (x : ℝ) (m : ℝ) : ℝ := x^2 - 2 * x + m

theorem minimum_value_of_quadratic (m : ℝ) :
  (∀ x ∈ set.Ici (2 : ℝ), quadratic_function x m ≥ -2) ∧ 
  (∃ x ∈ set.Ici (2 : ℝ), quadratic_function x m = -2) →
  m = -2 :=
by sorry

end minimum_value_of_quadratic_l85_85532


namespace matias_books_sold_l85_85240

theorem matias_books_sold :
  ∃ T : ℕ, (let W := 3 * T in let Th := 3 * W in T + W + Th = 91) ∧ T = 7 :=
by
  sorry

end matias_books_sold_l85_85240


namespace cone_ratio_l85_85166

theorem cone_ratio {r_1 r_2 s : ℝ} 
  (hemisphere_vol : (2/3) * π * r_1^3)
  (cone_vol : (π / 3) * ((r_1^2 + r_1 * r_2 + r_2^2) * sqrt (r_1 * r_2)))
  (sphere_vol : (4/3) * π * s^3)
  (combined_vol : cone_vol + hemisphere_vol = 3 * sphere_vol)
  (s_eq_sqrt : s = sqrt (r_1 * r_2)) :
  r_1 / r_2 = (5 + sqrt 21) / 2 :=
by
  sorry

end cone_ratio_l85_85166


namespace triangle_perimeter_l85_85652

variable (r A p : ℝ)

-- Define the conditions from the problem
def inradius (r : ℝ) := r = 3
def area (A : ℝ) := A = 30
def perimeter (A r p : ℝ) := A = r * (p / 2)

-- The theorem stating the problem
theorem triangle_perimeter (h1 : inradius r) (h2 : area A) (h3 : perimeter A r p) : p = 20 := 
by
  -- Proof is provided by the user, so we skip it with sorry
  sorry

end triangle_perimeter_l85_85652


namespace probability_prime_and_multiple_of_11_l85_85608

noncomputable def is_prime (n : ℕ) : Prop := nat.prime n
def is_multiple_of_11 (n : ℕ) : Prop := n % 11 = 0

theorem probability_prime_and_multiple_of_11 :
  (1 / 100 : ℝ) = 
  let qualifying_numbers := {n | n ∈ finset.range 101 ∧ is_prime n ∧ is_multiple_of_11 n} in
  let number_of_qualifying := finset.card qualifying_numbers in
  (number_of_qualifying / 100 : ℝ) :=
by
  sorry

end probability_prime_and_multiple_of_11_l85_85608


namespace total_wheels_in_parking_lot_l85_85902

def num_cars : ℕ := 14
def num_bikes : ℕ := 10
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

theorem total_wheels_in_parking_lot :
  (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike) = 76 :=
by
  sorry

end total_wheels_in_parking_lot_l85_85902


namespace polygon_sides_eq_n_l85_85535

theorem polygon_sides_eq_n
  (sum_except_two_angles : ℝ)
  (angle_equal : ℝ)
  (h1 : sum_except_two_angles = 2970)
  (h2 : angle_equal * 2 < 180)
  : ∃ n : ℕ, 180 * (n - 2) = 2970 + 2 * angle_equal ∧ n = 19 :=
by
  sorry

end polygon_sides_eq_n_l85_85535


namespace arithmetic_sequence_sum_n_4_l85_85670

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

-- Definitions for the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

variable (a : ℕ → ℝ)

-- Conditions
axiom a2 : a 2 = 1
axiom a3 : a 3 = 3

-- Sum of the first n terms of an arithmetic sequence
noncomputable def S (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, a (k + 1)

-- Proof statement
theorem arithmetic_sequence_sum_n_4 (h_arith : is_arithmetic_sequence a) : 
  S 4 = 8 :=
by {
  sorry
}

end arithmetic_sequence_sum_n_4_l85_85670


namespace determine_k_l85_85041

variable (x y z w : ℝ)

theorem determine_k
  (h₁ : 9 / (x + y + w) = k / (x + z + w))
  (h₂ : k / (x + z + w) = 12 / (z - y)) :
  k = 21 :=
sorry

end determine_k_l85_85041


namespace prime_factors_l85_85054

theorem prime_factors (some_number : ℕ):
  let prime_factors_in_some_number := nat.prime_factors some_number;
  let num_prime_factors := prime_factors_in_some_number.length;
  num_prime_factors + 13 + 5 = 29 → num_prime_factors = 11 :=
begin
  let prime_factors_count := num_prime_factors + 13 + 5,
  assume h : prime_factors_count = 29,
  have h1 : num_prime_factors = 11, by linarith,
  exact h1,
end

end prime_factors_l85_85054


namespace shift_parabola_upwards_l85_85162

theorem shift_parabola_upwards (x : ℝ) :
  (∀ y, y = -x^2 -> y + 2 = -x^2 + 2) :=
by
  intros y h
  rw h
  sorry

end shift_parabola_upwards_l85_85162


namespace value_of_S_l85_85564

theorem value_of_S :
  let S := (1 / (4 - Real.sqrt 15:ℝ)) - (1 / (Real.sqrt 15 - Real.sqrt 14)) 
          + (1 / (Real.sqrt 14 - Real.sqrt 13)) - (1 / (Real.sqrt 13 - Real.sqrt 12)) 
          + (1 / (Real.sqrt 12 - 3))
  in S = 7 := by
  sorry

end value_of_S_l85_85564


namespace AM_parallel_DN_l85_85405

variables {O A B C D E F P M N : ℝ}

-- Definitions of conditions
def hexagon_inscribed_in_circle (O A B C D E F : ℝ) : Prop := -- condition 1
sorry

def tangents_intersect_at_P (O A D P : ℝ) : Prop := -- condition 2
sorry

def BC_intersects_PD_at_M (B C P D M : ℝ) : Prop := -- condition 3
sorry

def EF_intersects_PA_at_N (E F P A N : ℝ) : Prop := -- condition 4
sorry

def angles_APF_eq_DPC (A P F D C : ℝ) : Prop := -- condition 5
sorry

def AB_parallel_DE (A B D E : ℝ) : Prop := -- condition 6
sorry

-- Theorem statement
theorem AM_parallel_DN 
  (h1 : hexagon_inscribed_in_circle O A B C D E F)
  (h2 : tangents_intersect_at_P O A D P)
  (h3 : BC_intersects_PD_at_M B C P D M)
  (h4 : EF_intersects_PA_at_N E F P A N)
  (h5 : angles_APF_eq_DPC A P F D C)
  (h6 : AB_parallel_DE A B D E) :
  AM_parallel_DN A M D N :=
sorry

end AM_parallel_DN_l85_85405


namespace unique_n_with_prime_squares_l85_85057

theorem unique_n_with_prime_squares :
  ∃! n : ℕ, n > 1 ∧ prime (nat.sqrt n) ∧ (n = (nat.sqrt n)^2) ∧ ((n + 64) = (nat.sqrt (n + 64))^2) ∧ prime (nat.sqrt (n + 64)) :=
sorry

end unique_n_with_prime_squares_l85_85057


namespace triangle_existence_condition_equality_case_l85_85420

noncomputable def is_right_triangle {A B C : Type*} [normed_space ℝ A] (a b c : A) (w : ℝ) (h : w < 90) : Prop :=
∃ M : A, midpoint ℝ b c M ∧ ∠ a M b = w

theorem triangle_existence_condition {A B C : Type*} [normed_space ℝ A] (b c : ℝ) (w : ℝ) (h : w < 90) :
  ∃ (A B C : A), (∠ A (midpoint ℝ B C) B) = w ∧ AC = b ∧ AB = c ↔ b * tan (w / 2) ≤ c ∧ c < b :=
sorry

theorem equality_case {A B C : Type*} [normed_space ℝ A] (b c : ℝ) (w : ℝ) (h : w < 90) :
  b * tan (w / 2) = c → (is_right_triangle A B C w h) :=
sorry

end triangle_existence_condition_equality_case_l85_85420


namespace unique_acute_prime_triangle_l85_85183

noncomputable def is_prime (n : ℕ) : Prop := sorry -- We assume a prime-checking definition

variable (α β γ : ℕ) -- Angles of the triangle in degrees
variable (ABC : Triangle) -- Triangle ABC

-- Conditions
def acute_triangle (ABC : Triangle) : Prop :=
  α < 90 ∧ β < 90 ∧ γ < 90

def internal_angles_prime (α β γ : ℕ) : Prop :=
  is_prime α ∧ is_prime β ∧ is_prime γ

-- Theorem: Proving the uniqueness and isosceles property
theorem unique_acute_prime_triangle (ABC : Triangle) (α β γ : ℕ)
  (h1: acute_triangle ABC)
  (h2 : α + β + γ = 180)
  (h3 : internal_angles_prime α β γ) :
  (α = 2 ∧ β = 89 ∧ γ = 89) ∨ (α = 89 ∧ β = 2 ∧ γ = 89) ∨ (α = 89 ∧ β = 89 ∧ γ = 2) ∧
  (isosceles_triangle ABC) :=
by
  sorry

end unique_acute_prime_triangle_l85_85183


namespace isosceles_triangle_two_longest_altitudes_sum_l85_85875

def isosceles_triangle_altitudes_sum (a b c : ℝ) (h_isosceles : a = b) (h_sides : {a, b, c} = {8, 8, 15}) : ℝ :=
  16

theorem isosceles_triangle_two_longest_altitudes_sum :
  ∀ (a b c : ℝ), a = b → ({a, b, c} = {8, 8, 15}) → isosceles_triangle_altitudes_sum a b c a (a = b) ({a, b ,c} = {8, 8, 15}) = 16 :=
by
  intros a b c h_isosceles h_sides
  dsimp [isosceles_triangle_altitudes_sum]
  rw [h_isosceles, h_sides]
  sorry

end isosceles_triangle_two_longest_altitudes_sum_l85_85875


namespace maria_interview_probability_l85_85962

theorem maria_interview_probability
  (total_students : ℕ)
  (robotics_students : ℕ)
  (drama_students : ℕ)
  (students_in_both_clubs : ℕ)
  (students_only_robotics : ℕ := robotics_students - students_in_both_clubs)
  (students_only_drama : ℕ := drama_students - students_in_both_clubs)
  (total_interviews : ℕ := nat.choose total_students 2)
  (robotics_interviews : ℕ := nat.choose students_only_robotics 2)
  (drama_interviews : ℕ := nat.choose students_only_drama 2)
  (both_club_probability : ℚ := 1 - (robotics_interviews + drama_interviews) / total_interviews)
  (correct_probability : ℚ := 352 / 435) :
  total_students = 30 →
  robotics_students = 22 →
  drama_students = 19 →
  students_in_both_clubs = (robotics_students + drama_students - total_students) →
  both_club_probability = correct_probability :=
by
  intros
  sorry

end maria_interview_probability_l85_85962


namespace ratio_area_of_triangles_l85_85086

open Real

noncomputable def equilateral_triangle := 
  { A B C : Point | dist A B = 11 ∧ dist B C = 11 ∧ dist C A = 11 }

noncomputable def points_on_sides := 
  { A B C A1 B1 C1 : Point 
  | dist A1 C = 5 ∧ dist B1 A = 5 ∧ dist C1 B = 5 }

theorem ratio_area_of_triangles 
  (A B C A1 B1 C1 : Point)
  (h1 : equilateral_triangle A B C)
  (h2 : dist A1 C = 5)
  (h3 : dist B1 A = 5)
  (h4 : dist C1 B = 5) :
  let ABC := Triangle A B C,
      DEF := Triangle (line_through_points A A1 ∩ line_through_points B B1)
                      (line_through_points B B1 ∩ line_through_points C C1)
                      (line_through_points C C1 ∩ line_through_points A A1) 
in (area ABC) / (area DEF) = 91 :=
sorry

end ratio_area_of_triangles_l85_85086


namespace calculation_results_in_a_pow_5_l85_85329

variable (a : ℕ)

theorem calculation_results_in_a_pow_5 : a^3 * a^2 = a^5 := 
  by sorry

end calculation_results_in_a_pow_5_l85_85329


namespace intersect_count_l85_85627

def g (x : ℝ) : ℝ := sorry
def g_inv (y : ℝ) : ℝ := sorry

axiom g_invertible : ∀ (y : ℝ), g (g_inv y) = y

theorem intersect_count : {x : ℝ | g (x^2) = g (x^6)}.finite.card = 3 := by
  sorry

end intersect_count_l85_85627


namespace lucy_cleans_aquariums_l85_85597

theorem lucy_cleans_aquariums :
  (∃ rate : ℕ, rate = 2 / 3) →
  (∃ hours : ℕ, hours = 24) →
  (∃ increments : ℕ, increments = 24 / 3) →
  (∃ aquariums : ℕ, aquariums = (2 * (24 / 3))) →
  aquariums = 16 :=
by
  sorry

end lucy_cleans_aquariums_l85_85597


namespace area_of_triangle_CD_l85_85566

def radius := 10
def diameter := 2 * radius
def AC := 16

-- Circle \(\Omega\) and points A, X
def A := (-radius, 0)
def X := (radius, 0)

-- Point C lies on \(\Omega\) such that AC = 16
def C := (2.8, 9.6) -- one of the possible coordinates, simplified

-- Point D lies on \(\Omega\) such that CX = CD
def D := (-2.8, 9.6) -- one of the possible coordinates, based on reflection

-- Midpoint of AC
def AC_midpoint := ( (-radius + 2.8) / 2, (0 + 9.6) / 2 )

-- Point D' is the reflection of D across the midpoint of AC
def D' := (2 * AC_midpoint.1 - D.1, 2 * AC_midpoint.2 - D.2)

-- Midpoint of CD
def CD_midpoint := ( (C.1 + D.1) / 2, (C.2 + D.2) / 2 )

-- Point X' is the reflection of X across the midpoint of CD
def X' := (2 * CD_midpoint.1 - X.1, 2 * CD_midpoint.2 - X.2)

-- Determinant formula for area of triangle CD'X'
def area := (1 / 2) * abs (
    C.1 * (D'.2 - X'.2) +
    D'.1 * (X'.2 - C'.2) +
    X'.1 * (C.2 - D'.2)
)

theorem area_of_triangle_CD'X' : area = 96 := sorry

end area_of_triangle_CD_l85_85566


namespace sequence_value_a8_b8_l85_85243

theorem sequence_value_a8_b8
(a b : ℝ) 
(h1 : a + b = 1) 
(h2 : a^2 + b^2 = 3) 
(h3 : a^3 + b^3 = 4) 
(h4 : a^4 + b^4 = 7) 
(h5 : a^5 + b^5 = 11) 
(h6 : a^6 + b^6 = 18) : 
a^8 + b^8 = 47 :=
sorry

end sequence_value_a8_b8_l85_85243


namespace minimum_value_l85_85095

theorem minimum_value (x : ℝ) (h₀ : x > 0) (y : ℝ) (h₁ : y = x ^ (-2)) : 
  x + y ≥ (3 * real.cbrt (2) / 2) :=
by
  sorry

end minimum_value_l85_85095


namespace proportion_of_ones_l85_85922

theorem proportion_of_ones (m n : ℕ) (h : Nat.gcd m n = 1) : 
  m + n = 275 :=
  sorry

end proportion_of_ones_l85_85922


namespace b_pow_a_eq_nine_l85_85148

theorem b_pow_a_eq_nine (a b : ℤ) (h1 : a < Real.sqrt 8) (h2 : Real.sqrt 8 < b) (h3 : a + 1 = b) : b ^ a = 9 :=
by
  sorry

end b_pow_a_eq_nine_l85_85148


namespace area_to_be_painted_correct_l85_85602

-- Define the dimensions and areas involved
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window_height : ℕ := 3
def window_length : ℕ := 5
def painting_height : ℕ := 2
def painting_length : ℕ := 2

-- Calculate the areas
def wall_area : ℕ := wall_height * wall_length
def window_area : ℕ := window_height * window_length
def painting_area : ℕ := painting_height * painting_length
def area_not_painted : ℕ := window_area + painting_area
def area_to_be_painted : ℕ := wall_area - area_not_painted

-- Theorem: The area to be painted is 131 square feet
theorem area_to_be_painted_correct : area_to_be_painted = 131 := by
  sorry

end area_to_be_painted_correct_l85_85602


namespace find_y_l85_85807

theorem find_y (x y : ℤ) (h1 : x^2 - 5 * x + 8 = y + 6) (h2 : x = -8) : y = 106 := by
  sorry

end find_y_l85_85807


namespace exists_point_dist_greater_than_l85_85247

open Int

def is_coprime (x y : ℤ) : Prop :=
  gcd x y = 1

theorem exists_point_dist_greater_than (n : ℕ) (h : n > 0) :
  ∃ (a b : ℤ), ∀ (x y : ℤ), is_coprime x y → (a - x)^2 + (b - y)^2 > n^2 :=
by
  sorry

end exists_point_dist_greater_than_l85_85247


namespace problem_l85_85857

theorem problem {x y : ℝ} (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) :
  let S := x^2 + y^2 in
  1 / (arg_max (λ S, ∃θ, S = x^2 + y^2)) + 1 / (arg_min (λ S, ∃θ, S = x^2 + y^2)) = 8 / 5 :=
sorry


end problem_l85_85857


namespace fair_coin_toss_consecutive_heads_l85_85366

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

end fair_coin_toss_consecutive_heads_l85_85366


namespace subtraction_of_negatives_l85_85411

theorem subtraction_of_negatives :
  -2 - (-3) = 1 := 
by
  sorry

end subtraction_of_negatives_l85_85411


namespace max_area_of_triangle_l85_85858

theorem max_area_of_triangle
  (C1 : ∀ {x y : ℝ}, x^2 + y^2 + 4 * x - 4 * y - 3 = 0)
  (C2 : ∀ {x y : ℝ}, x^2 + y^2 - 4 * x - 12 = 0) :
  ∃ P : ℝ × ℝ, P ∈ C2 → 
  let C1_center := (-2, 2),
      C2_center := (2, 0),
      distance_C1_C2 := 2 * √5 in
  ∀ d_PC2, (PC2 : ℝ × ℝ) = d_PC2 * (1, 0) → -- maximum perpendicular distance in direction
  abs (((C1_center.1 - PC2.1) * (C2_center.2 - PC2.2) - (C1_center.2 - PC2.2) * (C2_center.1 - PC2.1)) / 2) = 4 * √5 := 
by
  sorry

end max_area_of_triangle_l85_85858


namespace increasing_decreasing_intervals_l85_85428

noncomputable def f (x : ℝ) : ℝ := Real.sin (-2 * x + 3 * Real.pi / 4)

theorem increasing_decreasing_intervals : (∀ k : ℤ, 
    ∀ x, 
      ((k : ℝ) * Real.pi + 5 * Real.pi / 8 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + 9 * Real.pi / 8) 
      → 0 < f x ∧ f x < 1) 
  ∧ 
    (∀ k : ℤ, 
    ∀ x, 
      ((k : ℝ) * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + 5 * Real.pi / 8) 
      → -1 < f x ∧ f x < 0) :=
by
  sorry

end increasing_decreasing_intervals_l85_85428


namespace max_four_topping_pizzas_l85_85378

theorem max_four_topping_pizzas (n k : ℕ) (h_n : n = 6) (h_k : k = 4) : binomial n k = 15 :=
by 
  sorry

end max_four_topping_pizzas_l85_85378


namespace lillian_can_bake_and_ice_5_dozen_cupcakes_l85_85956

-- Define conditions as constants and variables
constant cups_at_home : ℕ := 3
constant bags_of_sugar : ℕ := 2
constant cups_per_bag : ℕ := 6
constant batter_sugar_per_dozen : ℕ := 1
constant frosting_sugar_per_dozen : ℕ := 2

-- Calculation based on conditions
def total_sugar : ℕ := cups_at_home + bags_of_sugar * cups_per_bag
def sugar_per_dozen_cupcakes : ℕ := batter_sugar_per_dozen + frosting_sugar_per_dozen
def dozen_cupcakes_possible (sugar : ℕ) : ℕ := sugar / sugar_per_dozen_cupcakes

theorem lillian_can_bake_and_ice_5_dozen_cupcakes :
  dozen_cupcakes_possible total_sugar = 5 :=
by
  sorry

end lillian_can_bake_and_ice_5_dozen_cupcakes_l85_85956


namespace find_expression_find_range_of_m_l85_85103

-- Definitions from the conditions
def f (x : ℝ) : ℝ := 2 * x^2 - 10 * x
def max_f_on_interval (I : set ℝ) := ∀ x ∈ I, f x ≤ 12
def solution_set_f_lt_zero (xs : set ℝ) := ∀ x ∈ xs, f x < 0

-- Statements to prove
theorem find_expression :
  (∀ x ∈ (set.Icc (-1) 4), f x ≤ 12) →
  (∀ x ∈ (set.Ioo 0 5), f x < 0) →
  f = (λ x, 2 * x^2 - 10 * x) :=
sorry

theorem find_range_of_m (m : ℝ) :
  (∀ x ∈ set.Icc (-1) 4, f (2 - 2 * real.cos x) < f (1 - real.cos x - m)) →
  m ∈ (-∞, -5) ∪ (1, ∞) :=
sorry

end find_expression_find_range_of_m_l85_85103


namespace koala_fiber_eaten_l85_85925

def koala_fiber_absorbed (fiber_eaten : ℝ) : ℝ := 0.30 * fiber_eaten

theorem koala_fiber_eaten (absorbed : ℝ) (fiber_eaten : ℝ) 
  (h_absorbed : absorbed = koala_fiber_absorbed fiber_eaten) : fiber_eaten = 40 :=
by {
  have h1 : fiber_eaten * 0.30 = absorbed,
  rw h_absorbed,
  have : 12 = absorbed,
  rw this,
  sorry,
}

end koala_fiber_eaten_l85_85925


namespace area_of_region_bounded_by_equation_l85_85271

theorem area_of_region_bounded_by_equation :
  ∃ (m n : ℤ), (x y : ℝ) (h : x^2 + y^2 = 4 * abs (x - y) + 2 * abs (x + y)),
    (m : ℝ) + (n : ℝ) * real.pi = 40 * real.pi := 
sorry

end area_of_region_bounded_by_equation_l85_85271


namespace Albaszu_machine_productivity_l85_85657

theorem Albaszu_machine_productivity (x : ℝ) 
  (h1 : 1.5 * x = 25) : x = 16 := 
by 
  sorry

end Albaszu_machine_productivity_l85_85657


namespace complement_of_A_l85_85128

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}

theorem complement_of_A : U \ A = {2, 4, 6} := 
by 
  sorry

end complement_of_A_l85_85128


namespace digit_5_not_in_97th_increasing_number_l85_85380

def is_increasing_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 : ℕ),
    n = d1*10^4 + d2*10^3 + d3*10^2 + d4*10 + d5 ∧
    1 ≤ d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 ≤ 9

def count_increasing_numbers_from (start_digit remaining_digits : ℕ) : ℕ :=
  nat.choose (9 - start_digit) remaining_digits

def nth_increasing_number (n : ℕ) : ℕ :=
  -- Node: Placeholder for function calculating the nth increasing number
  sorry

theorem digit_5_not_in_97th_increasing_number :
  ¬ (5 ∈ (nth_increasing_number 97).digits 10) :=
by
  sorry

end digit_5_not_in_97th_increasing_number_l85_85380


namespace problem1_proof_problem2_proof_l85_85117

-- Problem 1
noncomputable def f1 (x : ℝ) : ℝ := abs (3 * x - 1) + abs (x + 3)
def problem1_statement : Set ℝ := {x | f1 x >= 4}
def problem1_solution : Set ℝ := { x | x <= -3 ∨ x >= 1 / 2 }

theorem problem1_proof : problem1_statement = problem1_solution := sorry

-- Problem 2
variables (b c : ℝ)
noncomputable def f2 (x : ℝ) : ℝ := abs (x - b) + abs (x + c)

def problem2_statement : Prop := (∀ x, f2 x ≥ 1) ∧ (f2 1 = 1) ∧ b > 0 ∧ c > 0
def problem2_solution : ℝ := 4

theorem problem2_proof (h : problem2_statement) : (1 / b + 1 / c) = problem2_solution := sorry

end problem1_proof_problem2_proof_l85_85117


namespace measure_of_angle_B_area_of_triangle_l85_85099

variable {a b c : ℝ}
variables (A B C : ℝ) (S : ℝ)

-- Proof Problem 1
theorem measure_of_angle_B (h1 : b * Real.sin A + √3 * a * Real.cos B = 0) : 
    B = 120 ∨ B = 240 :=
sorry

-- Proof Problem 2
theorem area_of_triangle (h1 : b * Real.sin A + √3 * a * Real.cos B = 0) 
  (h2 : a = 2) 
  (h3 : let BD := sqrt 3 
        BD^2 = (1/4) * (c^2 + (a^2) 
        + 2 * (a) * (c) * Real.cos (120 * Real.pi/180))) : 
    S = 2 * sqrt 3 :=
sorry

end measure_of_angle_B_area_of_triangle_l85_85099


namespace log_diff_condition_l85_85056

theorem log_diff_condition (a : ℕ → ℝ) (d e : ℝ) (H1 : ∀ n : ℕ, n > 1 → a n = Real.log n / Real.log 3003)
  (H2 : d = a 2 + a 3 + a 4 + a 5 + a 6) (H3 : e = a 15 + a 16 + a 17 + a 18 + a 19) :
  d - e = -Real.log 1938 / Real.log 3003 := by
  sorry

end log_diff_condition_l85_85056


namespace cost_per_sq_meter_l85_85747

theorem cost_per_sq_meter (length width road_width total_cost : ℕ)
  (h_length : length = 80) (h_width : width = 40)
  (h_road_width : road_width = 10) (h_total_cost : total_cost = 3300) :
  let area_road1 := road_width * length,
      area_road2 := road_width * width,
      area_intersection := road_width * road_width,
      total_area := (area_road1 + area_road2) - area_intersection,
      cost_per_sq_meter := total_cost / total_area
  in cost_per_sq_meter = 3 :=
by
  -- Identifying the values for conditions
  have h_area_road1 : area_road1 = 800 := by simp [h_road_width, h_length]
  have h_area_road2 : area_road2 = 400 := by simp [h_road_width, h_width]
  have h_area_intersection : area_intersection = 100 := by simp [h_road_width]
  have h_total_area : total_area = 1100 := by simp [h_area_road1, h_area_road2, h_area_intersection]
  have h_cost_per_sq_meter : cost_per_sq_meter = 3 := by simp [h_total_cost, h_total_area]
  -- Finishing the proof
  exact h_cost_per_sq_meter

end cost_per_sq_meter_l85_85747


namespace find_x_l85_85510

def vec_a (x : ℝ) : ℝ × ℝ := (3 * x / 2, 1 / 2)
def vec_b (x : ℝ) : ℝ × ℝ := (-1, x)

def dot_prod (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def norm_squared (v : ℝ × ℝ) : ℝ :=
  dot_prod v v

def norm_diff_squared (v1 v2 : ℝ × ℝ) : ℝ :=
  norm_squared (2 * v1.1 - v2.1, 2 * v1.2 - v2.2)

open Real in
theorem find_x (x : ℝ) :
  norm_diff_squared (vec_a x) (vec_b x) = 4 * norm_squared (vec_a x) + norm_squared (vec_b x) + 2 →
  x = 1 / 2 :=
by
  sorry

end find_x_l85_85510


namespace vector_scalar_multiplication_l85_85044

namespace VectorMath

def vec1 : ℤ × ℤ × ℤ := (-3, 2, -1)
def vec2 : ℤ × ℤ × ℤ := (1, 5, -3)
def scalar : ℤ := 3

theorem vector_scalar_multiplication :
  scalar * (vec1 + vec2) = (-6, 21, -12) :=
by
  -- Proof goes here
  sorry

end VectorMath

end vector_scalar_multiplication_l85_85044


namespace intersection_or_parallel_lines_l85_85453

theorem intersection_or_parallel_lines
  (A B C P A₁ A₂ A₃ B₁ B₂ B₃ C₁ C₂ C₃ : Point)
  (hPA1: perp (P - A₁) (B - C))
  (hPA2: perp (P - A₂) (A - A₃))
  (hPB1: perp (P - B₁) (A - C))
  (hPB2: perp (P - B₂) (B - B₃))
  (hPC1: perp (P - C₁) (A - B))
  (hPC2: perp (P - C₂) (C - C₃)) :
  concurrent_or_parallel (line (A₁ - A₂)) (line (B₁ - B₂)) (line (C₁ - C₂)) :=
begin
  sorry
end

end intersection_or_parallel_lines_l85_85453


namespace total_elements_in_C_l85_85259

-- Defining the sets C and D
variables {C D : Set α}

-- Total number of elements in set C is three times the total number of elements in set D
def three_times_elements (h : |C| = 3 * |D|) : Prop := h

-- Total number of elements in the union of C and D
def union_elements (h : |C ∪ D| = 5040) : Prop := h

-- Intersection of sets C and D contains 840 elements
def intersection_elements (h : |C ∩ D| = 840) : Prop := h

-- Prove the total number of elements in set C is 4410
theorem total_elements_in_C
  (h1 : three_times_elements |C|)
  (h2 : union_elements |C ∪ D|)
  (h3 : intersection_elements |C ∩ D|) :
  |C| = 4410 :=
sorry

end total_elements_in_C_l85_85259


namespace new_ratio_after_adding_ten_l85_85688

theorem new_ratio_after_adding_ten 
  (x : ℕ) 
  (h_ratio : 3 * x = 15) 
  (new_smaller : ℕ := x + 10) 
  (new_larger : ℕ := 15) 
  : new_smaller / new_larger = 1 :=
by sorry

end new_ratio_after_adding_ten_l85_85688


namespace frequency_of_third_group_l85_85484

theorem frequency_of_third_group (total_data first_group second_group fourth_group third_group : ℕ) 
    (h1 : total_data = 40)
    (h2 : first_group = 5)
    (h3 : second_group = 12)
    (h4 : fourth_group = 8) :
    third_group = 15 :=
by
  sorry

end frequency_of_third_group_l85_85484


namespace max_colored_regions_l85_85429

theorem max_colored_regions (n : ℕ) (h : n ≥ 2) :
  ∀ (num_colored_regions : ℕ),
  (∃ k, -- k is the number of colored regions such that
   ∀ i j : ℕ, i ≠ j → ¬(regions share_common_boundary i j)) →
  num_colored_regions ≤ 1 / 3 * (n ^ 2 + n) :=
sorry

end max_colored_regions_l85_85429


namespace intervals_of_monotonicity_min_value_a_l85_85120

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (a x : ℝ) : ℝ := a / x

noncomputable def F (a x : ℝ) : ℝ := f x + g a x

theorem intervals_of_monotonicity (a : ℝ) (ha : 0 < a) :
  (∀ x, a < x → 0 < (F a).deriv x) ∧ (∀ x, 0 < x ∧ x < a → (F a).deriv x < 0) := sorry

theorem min_value_a (ha : ∀ (x : ℝ), 0 < x ∧ x ≤ 3 → (F a).deriv x ≤ 1/2) :
  a ≥ 1/2 := sorry

end intervals_of_monotonicity_min_value_a_l85_85120


namespace length_of_platform_l85_85724

theorem length_of_platform
  (train_length : ℝ)
  (signal_pole_time : ℝ)
  (platform_time : ℝ)
  (speed : ℝ := train_length / signal_pole_time)
  (distance_crossing_platform : ℝ := speed * platform_time)
  (platform_length : ℝ := distance_crossing_platform - train_length) :
  train_length = 450 → signal_pole_time = 28 → platform_time = 52 → platform_length ≈ 385.71 :=
begin
  intros h1 h2 h3,
  dsimp [train_length, signal_pole_time, platform_time, speed, distance_crossing_platform, platform_length],
  rw [h1, h2, h3],
  norm_num,
  exact approx_rfl,
end

end length_of_platform_l85_85724


namespace hexagon_division_parts_area_small_hexagon_l85_85177

-- Part a
theorem hexagon_division_parts (h : ∀ a b : ℝ, pnat.mul_comm 6 4 = 24) (area : ℝ) 
(h_area : area = 144): 
  divides_into_parts : ∃ parts : ℕ, parts = 24 := 
  sorry

-- Part b
theorem area_small_hexagon (h : ∀ a b : ℝ, pnat.div_mul 144 3 = 48) (area : ℝ) 
(h_area : area = 144): 
  area_small_hex : ∃ smaller_area : ℝ , smaller_area = 48 := 
  sorry

end hexagon_division_parts_area_small_hexagon_l85_85177


namespace probability_at_least_one_boy_one_girl_l85_85632

def total_members : ℕ := 20
def boys : ℕ := 10
def girls : ℕ := 10
def committee_size : ℕ := 4

theorem probability_at_least_one_boy_one_girl :
  (let total_committees := (Nat.choose total_members committee_size)
   let all_boys_committees := (Nat.choose boys committee_size)
   let all_girls_committees := (Nat.choose girls committee_size)
   let unwanted_committees := all_boys_committees + all_girls_committees
   let desired_probability := 1 - (unwanted_committees / total_committees : ℚ)
   in desired_probability = (295 / 323 : ℚ)) := 
by
  sorry

end probability_at_least_one_boy_one_girl_l85_85632


namespace lucy_cleans_aquariums_l85_85598

theorem lucy_cleans_aquariums :
  (∃ rate : ℕ, rate = 2 / 3) →
  (∃ hours : ℕ, hours = 24) →
  (∃ increments : ℕ, increments = 24 / 3) →
  (∃ aquariums : ℕ, aquariums = (2 * (24 / 3))) →
  aquariums = 16 :=
by
  sorry

end lucy_cleans_aquariums_l85_85598


namespace min_value_f_range_a_l85_85475

-- Define the given functions
def f (x : ℝ) : ℝ := x * Real.log x
def g (a x : ℝ) : ℝ := -x^2 + a * x - 2

-- Define the conditions for the intervals
def interval_t (t : ℝ) : Set ℝ := { x | t ≤ x ∧ x ≤ t + 1 }
def interval_1_e : Set ℝ := { x | 1 ≤ x ∧ x ≤ Real.exp 1 }

-- The first problem: minimum value of f(x) on the interval [t, t+1] for t > 0
theorem min_value_f (t : ℝ) (ht : 0 < t) :
  (∀ x ∈ interval_t t, f x ≥ t * Real.log t) ∧ 
  (∀ x ∈ interval_t 1, f x = 0) :=
sorry

-- The second problem: range of a such that f(x₀) ≥ g(x₀) for some x₀ ∈ [1, e]
theorem range_a (a : ℝ) :
  (∃ x₀ ∈ interval_1_e, f x₀ ≥ g a x₀) ↔ a ≤ 4 :=
sorry

end min_value_f_range_a_l85_85475


namespace vitamin_A_supplements_per_pack_l85_85604

theorem vitamin_A_supplements_per_pack {A x y : ℕ} (h1 : A * x = 119) (h2 : 17 * y = 119) : A = 7 :=
by
  sorry

end vitamin_A_supplements_per_pack_l85_85604


namespace count_ways_to_distribute_items_l85_85014

theorem count_ways_to_distribute_items :
  ∃ (ways : ℕ), ways = 51 ∧
    (∀ (items : ℕ) (bags : ℕ),
      items = 5 → bags = 3 → 
      ways = (λ (items bags : ℕ), -- a function that returns correct number of ways
        -- Note: This specific function or formula isn't implemented in the statement, symbolic placeholder used for proof sketch
        51)) := 
  sorry

end count_ways_to_distribute_items_l85_85014


namespace power_function_convex_upwards_l85_85059

noncomputable def f (x : ℝ) : ℝ :=
  x ^ (4 / 5)

theorem power_function_convex_upwards (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) :
  f ((x1 + x2) / 2) > (f x1 + f x2) / 2 :=
sorry

end power_function_convex_upwards_l85_85059


namespace range_of_m_correct_l85_85210

noncomputable def range_of_m {m : ℝ} : Prop :=
  (∀ x : ℝ, x > m → (1 ≤ x ∧ x < 3)) → m < 1

-- To formally state the lemma
theorem range_of_m_correct (m : ℝ) : range_of_m :=
  sorry

end range_of_m_correct_l85_85210


namespace correct_conclusion_l85_85116

-- Definitions and conditions
def f (a : ℝ) (x : ℝ) := a^x
def g (x : ℝ) (a : ℝ) :=
  if x ∈ set.Icc (-2 : ℝ) 2 then f a x else sorry

theorem correct_conclusion (a : ℝ) (x : ℝ) (y : ℝ)
  (ha : a > 0 ∧ a ≠ 1)
  (h_inverse_pass : ∃ x y, (f (1/a) y = x ∧ f a y = x) ∧ (x = sqrt 2 / 2) ∧ (y = 1/2))
  (h_even : ∀ x, g (x + 2) a = g (x + 2) a)
  (h_decreasing : ∀ x, x ∈ set.Icc (-2 : ℝ) 2 → g (x) a = f a x ∧ ∀ x y, x < y → g x a > g y a)
  : g (sqrt 2) a < g (3) a ∧ g (3) a < g (π) a := by
  sorry

end correct_conclusion_l85_85116


namespace problem_solution_l85_85434

noncomputable def positiveRealSolution : ℝ :=
  (75 + Real.sqrt 5681) / 2

theorem problem_solution :
  ∃ x : ℝ, x > 0 ∧ (1 / 2 * (4 * x ^ 2 - 2) = (x ^ 2 - 75 * x - 15) * (x ^ 2 + 35 * x + 7)) :=
begin
  use positiveRealSolution,
  sorry
end

end problem_solution_l85_85434


namespace evaluate_expression_l85_85798

theorem evaluate_expression (a : ℕ) (h : a = 3) : a^2 * a^5 = 2187 :=
by sorry

end evaluate_expression_l85_85798


namespace profit_per_piece_correct_sales_volume_correct_maximum_monthly_profit_optimum_selling_price_is_130_l85_85651

-- Define the selling price and cost price
def cost_price : ℝ := 60
def sales_price (x : ℝ) := x

-- 1. Prove the profit per piece
def profit_per_piece (x : ℝ) : ℝ := sales_price x - cost_price

theorem profit_per_piece_correct (x : ℝ) : profit_per_piece x = x - 60 :=
by 
  -- it follows directly from the definition of profit_per_piece
  sorry

-- 2. Define the linear function relationship between monthly sales volume and selling price
def sales_volume (x : ℝ) : ℝ := -2 * x + 400

theorem sales_volume_correct (x : ℝ) : sales_volume x = -2 * x + 400 :=
by 
  -- it follows directly from the definition of sales_volume
  sorry

-- 3. Define the monthly profit and prove the maximized profit
def monthly_profit (x : ℝ) : ℝ := profit_per_piece x * sales_volume x

theorem maximum_monthly_profit (x : ℝ) : 
  monthly_profit x = -2 * x^2 + 520 * x - 24000 :=
by 
  -- it follows directly from the definition of monthly_profit
  sorry

theorem optimum_selling_price_is_130 : ∃ (x : ℝ), (monthly_profit x = 9800) ∧ (x = 130) :=
by
  -- solve this using the properties of quadratic functions
  sorry

end profit_per_piece_correct_sales_volume_correct_maximum_monthly_profit_optimum_selling_price_is_130_l85_85651


namespace part1_part2_l85_85098

noncomputable def z1 : ℂ := - (1 / 2) - (complex.I * (real.sqrt 3) / 2)
noncomputable def z2 : ℂ := - (1 / 2) + (complex.I * (real.sqrt 3) / 2)

theorem part1 (h1 : z1^2 + z1 + 1 = 0) (h2 : z2^2 + z2 + 1 = 0) :
  (1 / z1) + (1 / z2) = -1 :=
by 
  have h_sum : z1 + z2 = -1, from by sorry,
  have h_prod : z1 * z2 = 1, from by sorry,
  calc
    (1 / z1) + (1 / z2) = (z1 + z2) / (z1 * z2) : by sorry
    ... = -1 / 1 : by sorry
    ... = -1 : by sorry

theorem part2 (a : ℝ) (h : z1 * (a + complex.I) = complex.I * b) :
  a = real.sqrt 3 :=
by
  have : z1 = - (1 / 2) - (complex.I * (real.sqrt 3) / 2), from by sorry,
  sorry

end part1_part2_l85_85098


namespace value_of_sum_l85_85067

theorem value_of_sum (a x y : ℝ) (h1 : 17 * x + 19 * y = 6 - a) (h2 : 13 * x - 7 * y = 10 * a + 1) : 
  x + y = 1 / 3 := 
sorry

end value_of_sum_l85_85067


namespace circumference_greater_than_100_l85_85677

def running_conditions (A B : ℝ) (C : ℝ) (P : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ A ≠ B ∧ P = 0 ∧ C > 0

theorem circumference_greater_than_100 (A B C P : ℝ) (h : running_conditions A B C P):
  C > 100 :=
by
  sorry

end circumference_greater_than_100_l85_85677


namespace y_increase_by_30_when_x_increases_by_12_l85_85540

theorem y_increase_by_30_when_x_increases_by_12
  (h : ∀ x y : ℝ, x = 4 → y = 10)
  (x_increase : ℝ := 12) :
  ∃ y_increase : ℝ, y_increase = 30 :=
by
  -- Here we assume the condition h and x_increase
  let ratio := 10 / 4  -- Establish the ratio of increase
  let expected_y_increase := x_increase * ratio
  exact ⟨expected_y_increase, sorry⟩  -- Prove it is 30

end y_increase_by_30_when_x_increases_by_12_l85_85540


namespace impossible_to_create_tabletop_l85_85372

theorem impossible_to_create_tabletop : 
  ∀ (side1 side2 side3 target : ℕ),
  side1 = 12 ∧ side2 = 15 ∧ side3 = 16 ∧ target = 25 →
  let area1 := side1 * side1
      area2 := side2 * side2
      area3 := side3 * side3
      area_target := target * target in
  (area1 + area2 + area3 = area_target) →
  ¬ (∃ (pieces : Finset (ℕ × ℕ)) (n : ℕ),
      n = 5 ∧
      (∀ piece ∈ pieces, piece.1 * piece.2 > 0) ∧ 
      (pieces.sum (λ p, p.1 * p.2) = area_target ∧ pieces.card = n)) := 
sorry

end impossible_to_create_tabletop_l85_85372


namespace find_triples_l85_85450

noncomputable def factorial : ℕ → ℕ 
| 0 => 1
| (n + 1) => (n + 1) * factorial n

theorem find_triples (x y : ℕ) (z : ℤ) : 
    (16 * z + 2017 = factorial x + factorial y) → 
    (odd z) → 
    ((x = 1 ∧ y = 6 ∧ z = -81) ∨ (x = 6 ∧ y = 1 ∧ z = -81) ∨ 
     (x = 1 ∧ y = 7 ∧ z = 189) ∨ (x = 7 ∧ y = 1 ∧ z = 189)) :=
begin
    sorry
end

end find_triples_l85_85450


namespace sum_of_star_angles_l85_85686

theorem sum_of_star_angles :
  let n := 12
  let angle_per_arc := 360 / n
  let arcs_per_tip := 3
  let internal_angle_per_tip := 360 - arcs_per_tip * angle_per_arc
  let sum_of_angles := n * (360 - internal_angle_per_tip)
  sum_of_angles = 1080 :=
by
  sorry

end sum_of_star_angles_l85_85686


namespace coin_game_probability_l85_85690

theorem coin_game_probability :
  let total_outcomes := 2^9
  let arrangements_4_heads_4_tails_in_8_flips := Nat.choose 8 4
  let probability_ends_on_9th_flip := (arrangements_4_heads_4_tails_in_8_flips * (1 / 256))
  let probability_before_9th_flip := 1 - probability_ends_on_9th_flip
  (probability_before_9th_flip = 93 / 128) := 
by 
  unfold total_outcomes arrangements_4_heads_4_tails_in_8_flips probability_ends_on_9th_flip probability_before_9th_flip
  sorry

end coin_game_probability_l85_85690


namespace probability_divisible_by_396_l85_85804

theorem probability_divisible_by_396 :
  ∀ (digits : List ℕ), 
  digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] →
  (∀ perm : List ℕ, perm = digits.permutations →
    let number := 5 * 10 ^ 23 + 
                  3 * 10 ^ 22 +
                  8 * 10 ^ 21 +
                  3 * 10 ^ 20 +
                  8 * 10 ^ 19 +
                  2 * 10 ^ 18 +
                  9 * 10 ^ 17 +
                  3 * 10 ^ 16 +
                  6 * 10 ^ 15 +
                  5 * 10 ^ 14 +
                  8 * 10 ^ 13 +
                  2 * 10 ^ 12 +
                  0 * 10 ^ 11 +
                  3 * 10 ^ 10 +
                  9 * 10 ^ 9 +
                  3 * 10 ^ 8 +
                  7 * 10 ^ 7 +
                  6 in 
    number % 396 = 0) :=
by sorry

end probability_divisible_by_396_l85_85804


namespace exists_ratios_eq_l85_85979

theorem exists_ratios_eq (a b z : ℕ) (ha : 0 < a) (hb : 0 < b) (hz : 0 < z) (h : a * b = z^2 + 1) :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ (a : ℚ) / b = (x^2 + 1) / (y^2 + 1) :=
by
  sorry

end exists_ratios_eq_l85_85979


namespace ratio_of_larger_to_smaller_l85_85672

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x > y) (h4 : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end ratio_of_larger_to_smaller_l85_85672


namespace angle_A_determination_l85_85856

theorem angle_A_determination (a R : ℝ) (h_a : a = 3) (h_R : R = 3) :
  ∃ A : ℝ, (sin A = 1 / 2) ∧ (A = 30 ∨ A = 150) :=
by
  -- Use the given conditions
  have h1 : a / (sin 30) = 2 * R := sorry
  -- Replace a and R with their given values
  have h2 : 3 / (sin 30) = 2 * 3 := sorry
  -- Simplify the expression
  have h3 : (sin 30) = 1 / 2 := sorry
  -- State the possible angle values
  existsi 30
  existsi 150
  split
  sorry
  -- Proof that the sine value equals 1/2
  split
  sorry
  sorry
  -- Proof that the angles are either 30° or 150°
  sorry

end angle_A_determination_l85_85856


namespace pairs_characterization_l85_85582

noncomputable def valid_pairs (A : ℝ) : Set (ℕ × ℕ) :=
  { p | ∃ x : ℝ, x > 0 ∧ (1 + x) ^ p.1 = (1 + A * x) ^ p.2 }

theorem pairs_characterization (A : ℝ) (hA : A > 1) :
  valid_pairs A = { p | p.2 < p.1 ∧ p.1 < A * p.2 } :=
by
  sorry

end pairs_characterization_l85_85582


namespace num_ways_to_divide_friends_l85_85523

theorem num_ways_to_divide_friends :
  (4 : ℕ) ^ 8 = 65536 := by
  sorry

end num_ways_to_divide_friends_l85_85523


namespace number_of_family_members_l85_85334

-- Define the cost of soda and pizza, and the total amount spent
def cost_soda := 0.5
def cost_pizza := 1.0
def total_amount := 9.0

-- Define the number of people as Zoe plus x family members
def total_people (x : ℕ) := x + 1

-- Define the cost per person
def cost_per_person := cost_soda + cost_pizza

-- Define the total cost in terms of the number of family members
def total_cost (x : ℕ) := total_people x * cost_per_person

-- The theorem to prove
theorem number_of_family_members : ∃ x : ℕ, total_cost x = total_amount ∧ x = 5 := 
by
  sorry

end number_of_family_members_l85_85334


namespace lillian_can_bake_and_ice_5_dozen_cupcakes_l85_85955

-- Define conditions as constants and variables
constant cups_at_home : ℕ := 3
constant bags_of_sugar : ℕ := 2
constant cups_per_bag : ℕ := 6
constant batter_sugar_per_dozen : ℕ := 1
constant frosting_sugar_per_dozen : ℕ := 2

-- Calculation based on conditions
def total_sugar : ℕ := cups_at_home + bags_of_sugar * cups_per_bag
def sugar_per_dozen_cupcakes : ℕ := batter_sugar_per_dozen + frosting_sugar_per_dozen
def dozen_cupcakes_possible (sugar : ℕ) : ℕ := sugar / sugar_per_dozen_cupcakes

theorem lillian_can_bake_and_ice_5_dozen_cupcakes :
  dozen_cupcakes_possible total_sugar = 5 :=
by
  sorry

end lillian_can_bake_and_ice_5_dozen_cupcakes_l85_85955


namespace sisterPassesMeInOppositeDirection_l85_85771

noncomputable def numberOfPasses (laps_sister : ℕ) : ℕ :=
if laps_sister > 1 then 2 * laps_sister else 0

theorem sisterPassesMeInOppositeDirection
  (my_laps : ℕ) (laps_sister : ℕ) (passes_in_same_direction : ℕ) :
  my_laps = 1 ∧ passes_in_same_direction = 2 ∧ laps_sister > 1 →
  passes_in_same_direction * 2 = 4 :=
by intros; sorry

end sisterPassesMeInOppositeDirection_l85_85771


namespace giraffe_ratio_l85_85676

theorem giraffe_ratio (g ng : ℕ) (h1 : g = 300) (h2 : g = ng + 290) : g / ng = 30 :=
by
  sorry

end giraffe_ratio_l85_85676


namespace smallest_possible_value_l85_85526

theorem smallest_possible_value (x : ℝ) (hx : 11 = x^2 + 1 / x^2) :
  x + 1 / x = -Real.sqrt 13 :=
by
  sorry

end smallest_possible_value_l85_85526


namespace area_of_gray_region_l85_85022

def center_C : ℝ × ℝ := (4, 6)
def radius_C : ℝ := 6
def center_D : ℝ × ℝ := (14, 6)
def radius_D : ℝ := 6

theorem area_of_gray_region :
  let area_of_rectangle := (14 - 4) * 6
  let quarter_circle_area := (π * 6 ^ 2) / 4
  let area_to_subtract := 2 * quarter_circle_area
  area_of_rectangle - area_to_subtract = 60 - 18 * π := 
by {
  sorry
}

end area_of_gray_region_l85_85022


namespace sum_of_fourth_powers_of_even_perfect_squares_lt_500_l85_85698

theorem sum_of_fourth_powers_of_even_perfect_squares_lt_500 :
  (∑ n in {k : ℕ | ∃ m : ℕ, m % 2 = 0 ∧ m^2 = k ∧ k^4 < 500}, n^4) = 272 :=
by
  sorry

end sum_of_fourth_powers_of_even_perfect_squares_lt_500_l85_85698


namespace percent_gain_on_transaction_l85_85000

theorem percent_gain_on_transaction:
  (n total_cows sold_cows remaining_cows price_per_cow : ℕ)
  (cost_per_cow total_cost total_revenue profit : ℝ) :
  total_cows = 625 →
  sold_cows = 600 →
  remaining_cows = 25 →
  price_per_cow = (total_cost : ℝ) / sold_cows →
  total_cost = (cost_per_cow : ℝ) * total_cows →
  total_revenue = (price_per_cow * sold_cows) + (price_per_cow * remaining_cows) →
  profit = total_revenue - total_cost →
  100 * (profit / total_cost) = 4.17 :=
sorry

end percent_gain_on_transaction_l85_85000


namespace teresa_total_spending_l85_85631

def price_fancy_sandwich : ℝ := 7.75
def num_fancy_sandwiches : ℝ := 2
def price_salami : ℝ := 4
def price_brie := 3 * price_salami
def price_per_pound_olives : ℝ := 10
def weight_olives : ℝ := 0.25
def price_per_pound_feta : ℝ := 8
def weight_feta : ℝ := 0.5
def price_french_bread : ℝ := 2
def discount_rate_brie : ℝ := 0.10
def discount_rate_sandwiches : ℝ := 0.15
def sales_tax_rate : ℝ := 0.05

def total_spending_before_tax : ℝ :=
  (num_fancy_sandwiches * price_fancy_sandwich * (1 - discount_rate_sandwiches)) +
  price_salami +
  (price_brie * (1 - discount_rate_brie)) +
  (weight_olives * price_per_pound_olives) +
  (weight_feta * price_per_pound_feta) +
  price_french_bread

def total_spending_including_tax : ℝ :=
  total_spending_before_tax * (1 + sales_tax_rate)

theorem teresa_total_spending :
  (total_spending_including_tax).round = 38.30 :=
by
  sorry

end teresa_total_spending_l85_85631


namespace cupcakes_baking_l85_85958

theorem cupcakes_baking (sugar_at_home : ℕ) (bags_bought : ℕ) (cups_per_bag : ℕ)
    (batter_sugar_per_dozen : ℕ) (frosting_sugar_per_dozen : ℕ) :
  sugar_at_home = 3 → bags_bought = 2 → cups_per_bag = 6 →
  batter_sugar_per_dozen = 1 → frosting_sugar_per_dozen = 2 →
  (sugar_at_home + bags_bought * cups_per_bag) / (batter_sugar_per_dozen + frosting_sugar_per_dozen) = 5 :=
by
  intros sugar_at_home_eq bags_bought_eq cups_per_bag_eq batter_sugar_per_dozen_eq frosting_sugar_per_dozen_eq
  rw [sugar_at_home_eq, bags_bought_eq, cups_per_bag_eq, batter_sugar_per_dozen_eq, frosting_sugar_per_dozen_eq]
  simp
  sorry

end cupcakes_baking_l85_85958


namespace common_ratio_q_l85_85551

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} {q : ℝ}

axiom a5_condition : a_n 5 = 2 * S_n 4 + 3
axiom a6_condition : a_n 6 = 2 * S_n 5 + 3

theorem common_ratio_q : q = 3 :=
by
  sorry

end common_ratio_q_l85_85551


namespace find_complex_z_modulus_of_z_l85_85483

open Complex

theorem find_complex_z (z : ℂ) (h1 : (z - (0 + 3 * I)).im = 0) (h2 : ((z - (0 + 5 * I)) / (2 - I)).re = 0) : 
    z = -1 + 3 * I := by 
  sorry

theorem modulus_of_z (z : ℂ) (h1 : (z - (0 + 3 * I)).im = 0) (h2 : ((z - (0 + 5 * I)) / (2 - I)).re = 0) : 
    Complex.abs (z / (1 - I)) = Real.sqrt 5 := by 
  sorry

end find_complex_z_modulus_of_z_l85_85483


namespace probability_of_success_probability_code_cracked_minimum_C_people_l85_85679

open ProbabilityTheory

variable (A B C : Event) (P_A P_B P_C : ℝ)
#check ∑ a b c, P_A * (1 - P_B) * (1 - P_C) + (1 - P_A) * P_B * (1 - P_C) + (1 - P_A) * (1 - P_B) * P_C

theorem probability_of_success [measure A] [measure B] [measure C] (P_A : ℝ) (P_B : ℝ) (P_C : ℝ) (hA: P_A = 0.5) (hB: P_B = (3 / 5)) (hC: P_C = (3 / 4)):
(∑ a b c, P_A * (1 - P_B) * (1 - P_C) + (1 - P_A) * P_B * (1 - P_C) + (1 - P_A) * (1 - P_B) * P_C) = 11 / 40 :=
begin
  sorry
end

theorem probability_code_cracked [measure A] [measure B] [measure C] (P_A : ℝ) (P_B : ℝ) (P_C : ℝ) (hA: P_A = 0.5) (hB: P_B = (3 / 5)) (hC: P_C = (3 / 4)):
(P_A * P_B * P_C) + (P_A * P_B * (1 - P_C)) + (P_A * (1 - P_B) * P_C) + ((1 - P_A) * P_B * P_C) = 19 / 20 :=
begin
  sorry
end

theorem minimum_C_people (P_C : ℝ) (hP_C: P_C = (3 / 4)) :
(minimum n, 1 - (1 - P_C)^n ≥ 0.95) = 3 :=
begin
  sorry
end

end probability_of_success_probability_code_cracked_minimum_C_people_l85_85679


namespace evaluate_f_f_neg1_l85_85860

def f (x : ℝ) : ℝ :=
  if h : x > 3 then Real.log x / Real.log 2
  else 3 - x

theorem evaluate_f_f_neg1 : f (f (-1)) = 2 :=
by
  sorry

end evaluate_f_f_neg1_l85_85860


namespace total_pieces_of_candy_l85_85392

-- Define the given conditions
def students : ℕ := 43
def pieces_per_student : ℕ := 8

-- Define the goal, which is proving the total number of pieces of candy is 344
theorem total_pieces_of_candy : students * pieces_per_student = 344 :=
by
  sorry

end total_pieces_of_candy_l85_85392


namespace magnitude_conjugate_l85_85075

theorem magnitude_conjugate (z : ℂ) (h : (1 + (complex.I)^2023) * z = 1 + complex.I) : 
  complex.abs (conj z) = 1 :=
sorry

end magnitude_conjugate_l85_85075


namespace Laura_income_l85_85554

-- Defining Laura's income and total tax paid
def A : ℝ
def T : ℝ

-- Given percentage for tax calculations
def p : ℝ

-- Defining the tax calculation conditions
def tax_first_35000 := 0.01 * p * 35000
def tax_above_35000 := 0.01 * (p + 3) * (A - 35000)
def tax_total := tax_first_35000 + tax_above_35000

-- Given condition for the total tax paid
def tax_condition := 0.01 * (p + 0.3) * A

-- Theorem stating Laura's annual income
theorem Laura_income :
  tax_total = tax_condition → A = 39000 := by
  -- Proof is omitted
  sorry

end Laura_income_l85_85554


namespace count_colorings_l85_85279

-- Define the types for colors and positions in the hexagonal grid
inductive Color
  | Red | Yellow | Green | Blue
  deriving DecidableEq

structure HexPosition where
  row : Int
  col : Int

def adjacent (p1 p2 : HexPosition) : Bool :=
  match (p1, p2) with
  | (⟨r1, c1⟩, ⟨r2, c2⟩) => 
      -- Define adjacency based on hexagonal grid structure (detailed model omitted for simplicity)
      (r1 - r2).abs <= 1 ∧ (c1 - c2).abs <= 1 ∧ (r1, c1) ≠ (r2, c2)

-- Define coloring constraints based on adjacency
def valid_coloring (coloring : HexPosition → Color) : Prop :=
  ∀ (p1 p2 : HexPosition), adjacent p1 p2 → coloring p1 ≠ coloring p2

-- The central hexagon is initially colored Red
def center : HexPosition := ⟨0, 0⟩
def fixed_center_coloring (coloring : HexPosition → Color) : Prop :=
  coloring center = Color.Red

-- The problem essentially reduces to counting the number of valid colorings:
theorem count_colorings : ∃ (coloring : HexPosition → Color), 
  valid_coloring coloring ∧ fixed_center_coloring coloring ∧
  (∃ S : Finset (HexPosition → Color), S.card ≥ (3^6) ∧ ∀ f ∈ S, valid_coloring f ∧ fixed_center_coloring f) :=
sorry

end count_colorings_l85_85279


namespace third_place_prize_is_120_l85_85746

noncomputable def prize_for_third_place (total_prize : ℕ) (first_place_prize : ℕ) (second_place_prize : ℕ) (prize_per_novel : ℕ) (num_novels_receiving_prize : ℕ) : ℕ :=
  let remaining_prize := total_prize - first_place_prize - second_place_prize
  let total_other_prizes := num_novels_receiving_prize * prize_per_novel
  remaining_prize - total_other_prizes

theorem third_place_prize_is_120 : prize_for_third_place 800 200 150 22 15 = 120 := by
  sorry

end third_place_prize_is_120_l85_85746


namespace annual_pension_in_terms_of_cdrs_l85_85404

variable (c d r s : ℝ)

def pension_increase_condition1 (P y : ℝ) : Prop :=
  P + r = P + sqrt(y + c)

def pension_increase_condition2 (P y : ℝ) : Prop :=
  P + s = P + sqrt(y + d)

theorem annual_pension_in_terms_of_cdrs
  (h1 : ∀ P y, pension_increase_condition1 c d r s P y)
  (h2 : ∀ P y, pension_increase_condition2 c d r s P y)
  (h3 : d ≠ c) :
  ∃ (P : ℝ), P = (cs^2 - dr^2) / (2 * (dr - cs)) :=
sorry

end annual_pension_in_terms_of_cdrs_l85_85404


namespace general_term_and_sum_l85_85845

noncomputable theory

open BigOperators

theorem general_term_and_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ n, ∃ q : ℝ, a n = a 1 * q ^ (n - 1)) ∧
  a 1 = 2 ∧
  ∃ q : ℝ, 2 * (a 3 + 1) = a 1 + a 4 ∧ a 3 = a 1 * q ^ 2 ∧ a 4 = a 1 * q ^ 3 →
  ∀ n, a n = 2 ^ n ∧
  b n = (2 * n - 1) * a n ∧
  S n = ∑ i in Finset.range n, b (i + 1) →
  S n = 6 + (2 * n - 3) * 2 ^ (n + 1) :=
sorry

end general_term_and_sum_l85_85845


namespace trajectory_is_straight_line_l85_85089

noncomputable def distance_point_line (M : ℝ × ℝ) (P : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  let dist_point := (M.1 - P.1)^2 + (M.2 - P.2)^2
  let dist_line := abs (M.1 - M.2 + 1) / sqrt(2)
  dist_point = dist_line

theorem trajectory_is_straight_line :
  ∀ (M : ℝ × ℝ), distance_point_line M (2, 3) (λ x y, x - y + 1 = 0) →
    ∃ m b, ∀ x y, x - y = 1 ∧ (2, 3).1 * m + (2, 3).2 = b :=
by
  sorry

end trajectory_is_straight_line_l85_85089


namespace summation_problem_l85_85590

theorem summation_problem 
  (f : ℕ → ℝ)
  (h1 : ∀ m n : ℕ, f (m + n) = f m * f n)
  (h2 : f 1 = 2) : 
  (Finset.range 2010).sum (λ i, f (i + 2) / f (i + 1)) = 4020 := 
by
  sorry

end summation_problem_l85_85590


namespace geometric_sequence_sufficient_not_necessary_l85_85479

theorem geometric_sequence_sufficient_not_necessary (a b c : ℝ) :
  (∃ r : ℝ, a = b * r ∧ b = c * r) → (b^2 = a * c) ∧ ¬ ( (b^2 = a * c) → (∃ r : ℝ, a = b * r ∧ b = c * r) ) :=
by
  sorry

end geometric_sequence_sufficient_not_necessary_l85_85479


namespace solution_set_l85_85936

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

axiom f_derivative : ∀ x : ℝ, has_deriv_at f (f' x) x
axiom condition1 : ∀ x : ℝ, f(x) + f'(x) < 1
axiom condition2 : f(0) = 2018

theorem solution_set : { x : ℝ | e^x * f(x) > e^x + 2017 } = Iio 0 :=
sorry

end solution_set_l85_85936


namespace max_garden_area_proof_l85_85255

noncomputable def max_garden_area : ℝ :=
  let w := 120 in
  let l := 480 - 2 * w in
  l * w

theorem max_garden_area_proof (l w : ℝ) (h_fencing : l + 2 * w = 480) (h_length : l ≥ 2 * w) : 
  max_garden_area = 28800 :=
by
  let w : ℝ := 120
  let l : ℝ := 480 - 2 * w
  have h₁ : l + 2 * w = 480 := by simp [l, w]
  have h₂ : l ≥ 2 * w := by simp [l, w]
  have h₃ : max_garden_area = l * w := rfl
  simp [h₁, h₂, h₃]
  exact 28800

#eval max_garden_area

end max_garden_area_proof_l85_85255


namespace correct_calculation_given_conditions_l85_85708

variable (number : ℤ)

theorem correct_calculation_given_conditions 
  (h : number + 16 = 64) : number - 16 = 32 := by
  sorry

end correct_calculation_given_conditions_l85_85708


namespace base4_to_base10_conversion_l85_85028

-- We define a base 4 number as follows:
def base4_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10 in
  let n := n / 10 in
  let d1 := n % 10 in
  let n := n / 10 in
  let d2 := n % 10 in
  let n := n / 10 in
  let d3 := n % 10 in
  let n := n / 10 in
  let d4 := n % 10 in
  (d4 * 4^4 + d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0)

-- Mathematical proof problem statement:
theorem base4_to_base10_conversion : base4_to_base10 21012 = 582 :=
  sorry

end base4_to_base10_conversion_l85_85028


namespace find_y_values_l85_85047

def A (y : ℝ) : ℝ := 1 - y - 2 * y^2

theorem find_y_values (y : ℝ) (h₁ : y ≤ 1) (h₂ : y ≠ 0) (h₃ : y ≠ -1) (h₄ : y ≠ 0.5) :
  y^2 * A y / (y * A y) ≤ 1 ↔
  y ∈ Set.Iio (-1) ∪ Set.Ioo (-1) (1/2) ∪ Set.Ioc (1/2) 1 :=
by
  -- proof is omitted
  sorry

end find_y_values_l85_85047


namespace quadratic_roots_product_l85_85217

open Real

theorem quadratic_roots_product (d e : ℝ) (h : 5*d^2 - 4*d - 1 = 0) (he : 5*e^2 - 4*e - 1 = 0) :
  (d - 2) * (e - 2) = 2.2 :=
by
  have sum_roots := by nlinarith [h, he]
  have prod_roots := by nlinarith [h, he]
  sorry

end quadratic_roots_product_l85_85217


namespace A_beats_B_by_160_meters_l85_85167

-- Definitions used in conditions
def distance_A := 400 -- meters
def time_A := 60 -- seconds
def distance_B := 400 -- meters
def time_B := 100 -- seconds
def speed_B := distance_B / time_B -- B's speed in meters/second
def time_for_B_in_A_time := time_A -- B's time for the duration A took to finish the race
def distance_B_in_A_time := speed_B * time_for_B_in_A_time -- Distance B covers in A's time

-- Statement to prove
theorem A_beats_B_by_160_meters : distance_A - distance_B_in_A_time = 160 :=
by
  -- This is a placeholder for an eventual proof
  sorry

end A_beats_B_by_160_meters_l85_85167


namespace arithmetic_sequence_a4_eight_l85_85909

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, a (n + m) = a n + m * (a 2 - a 1)

variable {a : ℕ → ℤ}

theorem arithmetic_sequence_a4_eight (h_arith_sequence : arithmetic_sequence a)
    (h_cond : a 3 + a 5 = 16) : a 4 = 8 :=
by
  sorry

end arithmetic_sequence_a4_eight_l85_85909


namespace lillian_cupcakes_l85_85954

theorem lillian_cupcakes (home_sugar : ℕ) (bags : ℕ) (sugar_per_bag : ℕ) (batter_sugar_per_dozen : ℕ) (frosting_sugar_per_dozen : ℕ) :
  home_sugar = 3 → bags = 2 → sugar_per_bag = 6 → batter_sugar_per_dozen = 1 → frosting_sugar_per_dozen = 2 →
  ((home_sugar + bags * sugar_per_bag) / (batter_sugar_per_dozen + frosting_sugar_per_dozen)) = 5 := 
by 
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end lillian_cupcakes_l85_85954


namespace twelfth_valid_number_l85_85397

def digits_sum (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

def is_valid_num (n : ℕ) : Prop :=
  digits_sum n = 12

def valid_nums : List ℕ :=
  List.filter is_valid_num (List.range 1000) -- Considering all numbers less than 1000

theorem twelfth_valid_number : valid_nums.nth 11 = some 165 := by
  sorry

end twelfth_valid_number_l85_85397


namespace break_even_income_l85_85624

noncomputable def sold_out_performance_cost := 16000

theorem break_even_income 
    (overhead_cost : ℕ) 
    (production_cost_per_performance : ℕ) 
    (performances_to_break_even : ℕ) 
    (total_cost : ℕ) 
    (income_per_performance : ℕ) :
  overhead_cost = 81000 →
  production_cost_per_performance = 7000 →
  performances_to_break_even = 9 →
  total_cost = overhead_cost + (production_cost_per_performance * performances_to_break_even) →
  income_per_performance = total_cost / performances_to_break_even →
  income_per_performance = sold_out_performance_cost :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  exact (h5.trans h4.symm)

end break_even_income_l85_85624


namespace diff_12_358_7_2943_l85_85017

theorem diff_12_358_7_2943 : 12.358 - 7.2943 = 5.0637 :=
by
  -- Proof is not required, so we put sorry
  sorry

end diff_12_358_7_2943_l85_85017


namespace geom_sum_proof_l85_85671

-- Let the conditions be:
-- a: the first term of the geometric sequence
-- r: the common ratio
-- S(n): the sum of the first n terms in the geometric sequence

def geom_sum (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

-- Given conditions
def sum_6033 (a r : ℝ) : ℝ := geom_sum a r 6033
def sum_12066 (a r : ℝ) : ℝ := geom_sum a r 12066

-- Main goal
def sum_18099 (a r : ℝ) : ℝ := geom_sum a r 18099

theorem geom_sum_proof (a r : ℝ) (h1 : sum_6033 a r = 600) (h2 : sum_12066 a r = 1140) : 
  sum_18099 a r = 1626 :=
by
  sorry

end geom_sum_proof_l85_85671


namespace arithmetic_sequence_k_value_l85_85190

theorem arithmetic_sequence_k_value 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) = a n + d) 
  (h_first_term : a 1 = 0) 
  (h_nonzero_diff : d ≠ 0) 
  (h_sum : ∃ k, a k = a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7) : 
  ∃ k, k = 22 := 
by 
  sorry

end arithmetic_sequence_k_value_l85_85190


namespace common_integer_root_l85_85451

theorem common_integer_root (a x : ℤ) : (a * x + a = 7) ∧ (3 * x - a = 17) → a = 1 :=
by
    sorry

end common_integer_root_l85_85451


namespace power_function_no_origin_l85_85890

theorem power_function_no_origin (m : ℝ) :
  (m = 1 ∨ m = 2) → 
  (m^2 - 3 * m + 3 ≠ 0 ∧ (m - 2) * (m + 1) ≤ 0) :=
by
  intro h
  cases h
  case inl =>
    -- m = 1 case will be processed here
    sorry
  case inr =>
    -- m = 2 case will be processed here
    sorry

end power_function_no_origin_l85_85890


namespace sqrt_inequality_and_equality_condition_l85_85225

variable {a b c : ℝ}

theorem sqrt_inequality_and_equality_condition (a_nonneg : a ≥ c) (b_nonneg : b ≥ c) (c_pos : c > 0) :
  ( sqrt (c * (a - c)) + sqrt (c * (b - c)) ≤ sqrt (a * b) ) ∧
  ( sqrt (c * (a - c)) + sqrt (c * (b - c)) = sqrt (a * b) ↔ a * b = c * (a + b) ) :=
by sorry

end sqrt_inequality_and_equality_condition_l85_85225


namespace ellipse_major_axis_length_l85_85403

def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

noncomputable def major_axis_length (f₁ f₂ : ℝ × ℝ) (tangent_point : ℝ × ℝ) : ℝ :=
  2 * distance (reflect_over_x tangent_point) f₂

noncomputable def reflect_over_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem ellipse_major_axis_length :
  let f₁ := (2, 10)
  let f₂ := (26, 35)
  tangent_point := (f₁.1, 0)
  distance (reflect_over_x f₁) f₂ = 51 →
  distance (reflect_over_x f₁) f₂ + distance tangent_point f₂ = 102 :=
by
  intros
  refl

end ellipse_major_axis_length_l85_85403


namespace lucy_needs_to_buy_l85_85596

/- 
  Define the initial amounts of each ingredient.
-/
def initial_flour : ℕ := 500
def initial_sugar : ℕ := 300
def initial_chocolate_chips : ℕ := 400

/- 
  Define the amounts used on different days.
-/
def monday_flour_used : ℕ := 150
def monday_sugar_used : ℕ := 120
def monday_chocolate_chips_used : ℕ := 200

def tuesday_flour_used : ℕ := 240
def tuesday_sugar_used : ℕ := 90
def tuesday_chocolate_chips_used : ℕ := 150

def wednesday_flour_used : ℕ := 100
def wednesday_chocolate_chips_used : ℕ := 90

/- 
  Define how much of each ingredient is needed to restock to full bags.
-/
def full_bag_flour : ℕ := 500
def full_bag_sugar : ℕ := 300
def full_bag_chocolate_chips : ℕ := 400

/- 
  Define the final amounts Lucy needs to buy to restock.
-/
def flour_needed_to_buy : ℕ := 545
def sugar_needed_to_buy : ℕ := 210
def chocolate_chips_needed_to_buy : ℕ := 440

theorem lucy_needs_to_buy :
  (let remaining_flour := (initial_flour - monday_flour_used - tuesday_flour_used) / 2 - wednesday_flour_used
   in full_bag_flour + (- remaining_flour))
  = flour_needed_to_buy
  ∧
  (let remaining_sugar := initial_sugar - monday_sugar_used - tuesday_sugar_used
   in full_bag_sugar - remaining_sugar)
  = sugar_needed_to_buy
  ∧
  (let remaining_chocolate_chips := initial_chocolate_chips - monday_chocolate_chips_used - tuesday_chocolate_chips_used - wednesday_chocolate_chips_used
   in full_bag_chocolate_chips + (- remaining_chocolate_chips))
  = chocolate_chips_needed_to_buy := by
  sorry

end lucy_needs_to_buy_l85_85596


namespace crossing_time_approx_l85_85005

-- Define the speed of the train in kilometers per hour.
def speed_kmph : ℝ := 90

-- Define the length of the train in meters.
def length_m : ℝ := 125.01

-- Conversion factor from kilometers per hour to meters per second.
def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

-- Calculate the speed of the train in meters per second.
def speed_mps : ℝ := kmph_to_mps speed_kmph

-- Calculate the time in seconds for the train to cross the pole.
def crossing_time : ℝ := length_m / speed_mps

-- Prove that the time is approximately 5 seconds.
theorem crossing_time_approx : crossing_time ≈ 5 := by
  sorry

end crossing_time_approx_l85_85005


namespace john_needs_2_sets_l85_85814

-- Definition of the conditions
def num_bars_per_set : ℕ := 7
def total_bars : ℕ := 14

-- The corresponding proof problem statement
theorem john_needs_2_sets : total_bars / num_bars_per_set = 2 :=
by
  sorry

end john_needs_2_sets_l85_85814


namespace integer_solutions_count_eq_4_l85_85142

theorem integer_solutions_count_eq_4 : 
  ∃ S : Finset ℤ, (∀ x ∈ S, (x - 3) ^ (36 - x ^ 2) = 1) ∧ S.card = 4 := 
sorry

end integer_solutions_count_eq_4_l85_85142


namespace monotonic_intervals_minimum_m_value_l85_85113

noncomputable def f (x : ℝ) (a : ℝ) := (2 * Real.exp 1 + 1) * Real.log x - (3 * a / 2) * x + 1

theorem monotonic_intervals (a : ℝ) : 
  if a ≤ 0 then ∀ x ∈ Set.Ioi 0, 0 < (2 * Real.exp 1 + 1) / x - (3 * a / 2) 
  else ∀ x ∈ Set.Ioc 0 ((2 * (2 * Real.exp 1 + 1)) / (3 * a)), (2 * Real.exp 1 + 1) / x - (3 * a / 2) > 0 ∧
       ∀ x ∈ Set.Ioi ((2 * (2 * Real.exp 1 + 1)) / (3 * a)), (2 * Real.exp 1 + 1) / x - (3 * a / 2) < 0 := sorry

noncomputable def g (x : ℝ) (m : ℝ) := x * Real.exp x + m - ((2 * Real.exp 1 + 1) * Real.log x + x - 1)

theorem minimum_m_value :
  ∀ (m : ℝ), (∀ (x : ℝ), 0 < x → g x m ≥ 0) ↔ m ≥ - Real.exp 1 := sorry

end monotonic_intervals_minimum_m_value_l85_85113


namespace max_value_of_m_l85_85314

theorem max_value_of_m (S : Finset ℕ) (hS1 : ∀ j ∈ S, 0 < j)
  (hS2 : S.card = 19)
  (hS3 : ∑ j in S, j^2 = 2500) : 
  ∀ T : Finset ℕ, (∀ t ∈ T, 0 < t) → 
    (∑ t in T, t^2 = 2500) → T.card ≤ 19 := 
by
  sorry

end max_value_of_m_l85_85314


namespace initial_number_of_girls_l85_85987

noncomputable theory

-- Definitions for initial conditions
def initial_percentage_girls (total_students : ℕ) : ℕ := 3 * total_students / 10
def changed_percentage_girls (total_students : ℕ) (girls_left : ℕ) : ℕ :=
  (initial_percentage_girls total_students) - girls_left

-- The proof problem
theorem initial_number_of_girls (p : ℕ) (h1 : 3 * p / 10 > 0) (h2 : 3 * p / 10 - 3 = 2 * p / 10) : 
  initial_percentage_girls p = 9 :=
by 
  sorry

end initial_number_of_girls_l85_85987


namespace sin_double_angle_l85_85061

open Real

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Set.Ioc (π / 2) π) (h2 : sin α = 4 / 5) :
  sin (2 * α) = -24 / 25 :=
by
  sorry

end sin_double_angle_l85_85061


namespace parabola_vertex_l85_85639

theorem parabola_vertex :
  ∃ (x y : ℤ), ((∀ x : ℝ, 2 * x^2 - 4 * x - 7 = y) ∧ x = 1 ∧ y = -9) := 
sorry

end parabola_vertex_l85_85639


namespace exists_divisible_pair_l85_85263

theorem exists_divisible_pair (S : Finset ℕ) (hS : S.card = 51) (hU : S ⊆ Finset.range 101) :
  ∃ x y ∈ S, x ∣ y ∨ y ∣ x :=
by
  sorry

end exists_divisible_pair_l85_85263


namespace equidistant_point_min_side_length_height_l85_85556

variables {a : ℝ} -- Side length of the square
variables {hA hB hC hD : ℝ} -- Heights of the trees at vertices

theorem equidistant_point_min_side_length_height (
  hA_eq : hA = 7
  hB_eq : hB = 13
  hC_eq : hC = 17
) : ∃ a > real.sqrt 120, hD = 13 :=
  let hD := 13 in exists.intro (real.sqrt 120 + ε) (and.intro trivial sorry)

end equidistant_point_min_side_length_height_l85_85556


namespace probability_all_green_apples_l85_85920

theorem probability_all_green_apples :
  let total_ways := Nat.choose 10 3
  let green_ways := Nat.choose 4 3
  let probability := (green_ways : ℚ) / (total_ways : ℚ)
  probability = 1 / 30 := 
by
  have total_ways_eq : Nat.choose 10 3 = 120 := by sorry
  have green_ways_eq : Nat.choose 4 3 = 4 := by sorry
  have probability_eq : (4 : ℚ) / (120 : ℚ) = 1 / 30 := by sorry
  rw [total_ways_eq, green_ways_eq, probability_eq]
  rfl

end probability_all_green_apples_l85_85920


namespace exponent_power_identity_l85_85881

variable {G : Type*} [Group G]

theorem exponent_power_identity {a : G} {m n : ℕ} (h₁ : a ^ m = 2) (h₂ : a ^ n = 1) : a ^ (m + 2 * n) = 2 := by
  sorry

end exponent_power_identity_l85_85881


namespace problem_I_problem_II_l85_85501

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4 * a * x + 1
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := 6 * a^2 * Real.log x + 2 * b + 1
noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := f a x + g a b x

theorem problem_I (a : ℝ) (ha : a > 0) :
  ∃ b, b = 5 / 2 * a^2 - 3 * a^2 * Real.log a ∧ ∀ b', b' ≤ 3 / 2 * Real.exp (2 / 3) :=
sorry

theorem problem_II (a x₁ x₂ : ℝ) (ha : a ≥ Real.sqrt 3 - 1) (hx : 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂) :
  (h a b x₂ - h a b x₁) / (x₂ - x₁) > 8 :=
sorry

end problem_I_problem_II_l85_85501


namespace age_ratio_l85_85781

variable (Cindy Jan Marcia Greg: ℕ)

theorem age_ratio 
  (h1 : Cindy = 5)
  (h2 : Jan = Cindy + 2)
  (h3: Greg = 16)
  (h4 : Greg = Marcia + 2)
  (h5 : ∃ k : ℕ, Marcia = k * Jan) 
  : Marcia / Jan = 2 := 
    sorry

end age_ratio_l85_85781


namespace find_x_for_parallel_vectors_l85_85509

open Real

theorem find_x_for_parallel_vectors :
  ∃ x : ℝ, let a := (4 : ℝ, 2 : ℝ), b := (x, 3 : ℝ) in
  (a.1 * b.2 - a.2 * b.1 = 0) → (x = 6) :=
begin
  intro h,
  use 6,
  sorry
end

end find_x_for_parallel_vectors_l85_85509


namespace roots_sum_ln_abs_x_minus_2_eq_m_l85_85482

open Real

theorem roots_sum_ln_abs_x_minus_2_eq_m (m : ℝ) (x1 x2 : ℝ)
  (h1 : ln (abs (x1 - 2)) = m)
  (h2 : ln (abs (x2 - 2)) = m)
  (h3 : x1 ≠ x2) :
  x1 + x2 = 4 :=
sorry

end roots_sum_ln_abs_x_minus_2_eq_m_l85_85482


namespace evaluate_sum_series_l85_85043

noncomputable def sum_infinite_series : ℝ :=
  ∑' n, (n^3 + n^2 - n) / ((n + 3)! : ℝ)

theorem evaluate_sum_series :
  sum_infinite_series = 1 / 6 :=
by
  sorry

end evaluate_sum_series_l85_85043


namespace max_value_of_T_l85_85571

noncomputable def a_seq (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 3) ^ n

noncomputable def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a_seq (i + 1)

noncomputable def b_seq (n : ℕ) : ℝ :=
  (3 / 2) * Real.log (3) (1 - 2 * S n) + 10

noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b_seq (i + 1)

theorem max_value_of_T : ∃ n : ℕ, T n = 57 / 2 :=
by
  sorry

end max_value_of_T_l85_85571


namespace range_omega_for_two_zeros_l85_85114

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.cos (ω * x + π / 6) - 1

theorem range_omega_for_two_zeros (ω : ℝ) :
  (∃! x ∈ Ioo 0 π, f ω x = 0) ↔ (3 / 2 < ω ∧ ω ≤ 13 / 6) :=
sorry

end range_omega_for_two_zeros_l85_85114


namespace number_of_ways_to_divide_friends_l85_85516

theorem number_of_ways_to_divide_friends :
  (4 ^ 8 = 65536) := by
  -- result obtained through the multiplication principle
  sorry

end number_of_ways_to_divide_friends_l85_85516


namespace speed_of_man_rowing_upstream_l85_85371

theorem speed_of_man_rowing_upstream (V_m V_downstream V_upstream V_s : ℝ) 
  (h1 : V_m = 25) 
  (h2 : V_downstream = 38) :
  V_upstream = V_m - (V_downstream - V_m) :=
by
  sorry

end speed_of_man_rowing_upstream_l85_85371


namespace octagon_triangle_probability_l85_85178

theorem octagon_triangle_probability : 
  let octagon_vertices := finset.range 8
  let total_triangles := (octagon_vertices.card.choose 3)
  let valid_triangles := 40
  total_triangles = 56 → 
  valid_triangles.fdiv total_triangles = (5 : ℚ) / 7 :=
by {
  let octagon_vertices := finset.range 8
  have h1 : octagon_vertices.card = 8 := sorry,
  have h2 : total_triangles = octagon_vertices.card.choose 3 := rfl,
  have h3 : total_triangles = 56 := by 
    calc
      total_triangles 
        = 8.choose 3 : by rw [h2, h1]
    ... = 56 : by decide,
  have h4 : valid_triangles = 40 := rfl,
  show valid_triangles.fdiv total_triangles = (5 : ℚ) / 7, from
    by rw [h4, h3]; exact sorry
}

end octagon_triangle_probability_l85_85178


namespace triangle_probability_l85_85042

def lengths : List ℕ := [2, 3, 4, 6, 9, 10, 12, 15]

def isTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_triangles (l : List ℕ) : List (ℕ × ℕ × ℕ) :=
  let triples := l.nthLe "choose-three combinations" sorry 
  triples.filter (λ (x : ℕ × ℕ × ℕ), isTriangle x.1 x.2 x.3)

def total_combinations (n : ℕ) (k : ℕ) : ℕ :=
  let f := nat.factorial
  f n / (f k * f (n - k))

theorem triangle_probability : 
  (valid_triangles lengths).length = 17 → total_combinations 8 3 = 56 → 
  (float (valid_triangles lengths).length / float (total_combinations 8 3)) = 17/56 :=
sorry

end triangle_probability_l85_85042


namespace find_cost_price_l85_85394

-- Define the known data
def cost_price_80kg (C : ℝ) := 80 * C
def cost_price_20kg := 20 * 20
def selling_price_mixed := 2000
def total_cost_price_mixed (C : ℝ) := cost_price_80kg C + cost_price_20kg

-- Using the condition for 25% profit
def selling_price_of_mixed (C : ℝ) := 1.25 * total_cost_price_mixed C

-- The main theorem
theorem find_cost_price (C : ℝ) : selling_price_of_mixed C = selling_price_mixed → C = 15 :=
by
  sorry

end find_cost_price_l85_85394


namespace conclusion_1_conclusion_2_conclusion_3_conclusion_4_number_of_correct_conclusions_l85_85498

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3) + 1

-- Conclusion ①
theorem conclusion_1 : ∀ x : ℝ, f (x + π) = f x := sorry

-- Conclusion ②
theorem conclusion_2 : ¬∀ x : ℝ, (2 * sin (2 * (x + π / 6) - π / 3) + 1) 
   = (2 * sin (2 * x) + 1) := sorry

-- Conclusion ③
theorem conclusion_3 : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 5 * π / 12 → f x1 < f x2 := sorry

-- Conclusion ④
theorem conclusion_4 : ∀ x1 x2 x3 : ℝ, (π / 3 ≤ x1 ∧ x1 ≤ π / 2) → 
  (π / 3 ≤ x2 ∧ x2 ≤ π / 2) → (π / 3 ≤ x3 ∧ x3 ≤ π / 2) → 
  f x1 + f x3 > f x2 := sorry

-- The main theorem
theorem number_of_correct_conclusions :
  (conclusion_1 ∧ conclusion_3 ∧ conclusion_4) ∧ ¬conclusion_2 → 
   3 = (if (conclusion_1 ∧ conclusion_3 ∧ conclusion_4 ∧ ¬conclusion_2) then 3 else 0) :=
sorry

end conclusion_1_conclusion_2_conclusion_3_conclusion_4_number_of_correct_conclusions_l85_85498


namespace dodecahedron_surface_area_increase_l85_85699

theorem dodecahedron_surface_area_increase (a : ℝ) :
  let A := 3 * real.sqrt(25 + 10 * real.sqrt 5) * a^2 in
  let a' := 1.20 * a in
  let A' := 3 * real.sqrt(25 + 10 * real.sqrt 5) * a'^2 in
  ((A' - A) / A) * 100 = 44 :=
by
  sorry

end dodecahedron_surface_area_increase_l85_85699


namespace gear_squeak_interval_l85_85367

theorem gear_squeak_interval 
  (N : ℕ) (M : ℕ) (T : ℕ) (hN : N = 12) (hM : M = 32) (hT : T = 3) :
  let lcm_val := Nat.lcm N M in
  (lcm_val / M) * T = 9 :=
by
  sorry

end gear_squeak_interval_l85_85367


namespace solution_set_l85_85855

section 
variables {f : ℝ → ℝ}

/-- Given conditions -/
variables (h1 : ∀ x ∈ ℝ, f (-x) = -f x)  -- f is an odd function
variables (h2 : ∀ x, x > 0 → f x > f (-x))  -- f is monotonically increasing on (0, +∞)
variable  (h3 : f (-2) = 0)  -- f(-2)=0

/-- Prove the solution set of x * f x < 0 is (-2, 0) ∪ (0, 2) -/
theorem solution_set :
  {x : ℝ | x * f x < 0} = set.Ioo (-2 : ℝ) 0 ∪ set.Ioo 0 2 := 
sorry

end

end solution_set_l85_85855


namespace subset_problem_l85_85127

theorem subset_problem (a : ℝ) (P S : Set ℝ) :
  P = { x | x^2 - 2 * x - 3 = 0 } →
  S = { x | a * x + 2 = 0 } →
  (S ⊆ P) →
  (a = 0 ∨ a = 2 ∨ a = -2 / 3) :=
by
  intro hP hS hSubset
  sorry

end subset_problem_l85_85127


namespace pirate_treasure_probability_l85_85377

/--
Suppose there are 8 islands. Each island has a 1/3 chance of having buried treasure and no traps,
a 1/6 chance of having traps but no treasure, and a 1/2 chance of having neither traps nor treasure.
Prove that the probability that while searching all 8 islands, the pirate will encounter exactly 4 islands 
with treasure and none with traps is 35/648.
-/
theorem pirate_treasure_probability :
  let p_treasure_no_trap := 1/3
      p_trap_no_treasure := 1/6
      p_neither := 1/2
      choose_8_4 := Nat.choose 8 4
      p_4_treasure_no_trap := p_treasure_no_trap^4
      p_4_neither := p_neither^4
  in choose_8_4 * p_4_treasure_no_trap * p_4_neither = 35/648 := 
by
  sorry

end pirate_treasure_probability_l85_85377


namespace cars_between_black_and_white_l85_85349

-- Define the given conditions
def total_cars : ℕ := 20
def black_car_position_from_right : ℕ := 16
def white_car_position_from_left : ℕ := 11

-- Calculate the position of the black car from the left
def black_car_position_from_left : ℕ := total_cars - black_car_position_from_right + 1

-- Define the theorem to prove
theorem cars_between_black_and_white :
  black_car_position_from_left = 5 → white_car_position_from_left = 11 → (white_car_position_from_left - black_car_position_from_left - 1) = 5 :=
by
  intros h1 h2
  simp [h1, h2]
  sorry

end cars_between_black_and_white_l85_85349


namespace total_points_first_four_games_l85_85413

-- Define the scores for the first three games
def score1 : ℕ := 10
def score2 : ℕ := 14
def score3 : ℕ := 6

-- Define the score for the fourth game as the average of the first three games
def score4 : ℕ := (score1 + score2 + score3) / 3

-- Define the total points scored in the first four games
def total_points : ℕ := score1 + score2 + score3 + score4

-- State the theorem to prove
theorem total_points_first_four_games : total_points = 40 :=
  sorry

end total_points_first_four_games_l85_85413


namespace find_distance_CD_l85_85418

-- Define the ellipse and the required points
def ellipse (x y : ℝ) : Prop := 16 * (x-3)^2 + 4 * (y+2)^2 = 64

-- Define the center and the semi-axes lengths
noncomputable def center : (ℝ × ℝ) := (3, -2)
noncomputable def semi_major_axis_length : ℝ := 4
noncomputable def semi_minor_axis_length : ℝ := 2

-- Define the points C and D on the ellipse
def point_C (x y : ℝ) : Prop := ellipse x y ∧ (x = 3 + semi_major_axis_length ∨ x = 3 - semi_major_axis_length) ∧ y = -2
def point_D (x y : ℝ) : Prop := ellipse x y ∧ x = 3 ∧ (y = -2 + semi_minor_axis_length ∨ y = -2 - semi_minor_axis_length)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Main theorem to prove
theorem find_distance_CD : 
  ∃ C D : ℝ × ℝ, 
    (point_C C.1 C.2 ∧ point_D D.1 D.2) → 
    distance C D = 2 * Real.sqrt 5 := 
sorry

end find_distance_CD_l85_85418


namespace num_real_roots_of_g_eq_0_l85_85224

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else real.sqrt x - 1

def g (x : ℝ) : ℝ := f (f x)

theorem num_real_roots_of_g_eq_0 : 
  ({x : ℝ | g x = 0}.to_finset.card = 3) :=
by
  sorry

end num_real_roots_of_g_eq_0_l85_85224


namespace pivot_distance_l85_85722

noncomputable def rod_length : ℝ := 1.4
noncomputable def mass : ℝ := 3.0
noncomputable def desired_speed : ℝ := 1.6
noncomputable def acceleration_due_to_gravity : ℝ := 9.8

theorem pivot_distance 
    (l = rod_length)
    (m = mass)
    (v = desired_speed)
    (g = acceleration_due_to_gravity) :
    ∃ x : ℝ, 0 < x ∧ x < l ∧
    (let k := (l - x) / x in 
    45.84 - 60 * x + 3.84 * k^2 = 0) ∧
    x = 0.8 :=
begin
  sorry
end

end pivot_distance_l85_85722


namespace smallest_norm_of_v_eq_l85_85212

noncomputable def v : ℝ² := sorry -- assume we have some vector v satisfying the condition

-- the main theorem statement
theorem smallest_norm_of_v_eq : 
  ‖(v + ![4, 2])‖ = 10 → ‖v‖ = 10 - 2 * Real.sqrt 5 := 
by
  intro h
  sorry

end smallest_norm_of_v_eq_l85_85212


namespace length_MN_l85_85871

-- Define the circles C1 and C2
def C1 (x y : ℝ) := (x - 1)^2 + (y - 2)^2 = 4
def C2 (x y : ℝ) := (x - 2)^2 + (y - 1)^2 = 2

-- Define points A and B as intersection points (for the sake of argument clarity)
variable (A B : ℝ × ℝ)

-- Define the line l tangent to C2 and parallel to AB
variable (l : ℝ → ℝ)

-- Introduce conditions extracted from the problem statement
axiom intersects_C1 : ∃ x y, l x = y ∧ C1 x y
axiom intersects_C2 : ∃ x y, l x = y ∧ C2 x y

-- State the proof problem to find the length of MN
theorem length_MN : 
  let M := Classical.some intersects_C1,
      N := Classical.some intersects_C2 in
      dist M N = 4 :=
sorry

end length_MN_l85_85871


namespace find_items_l85_85308

-- Define a structure for the gnomes
structure Gnome :=
(name : String)

-- Define the gnomes
def Eli : Gnome := ⟨"Eli"⟩
def Pili : Gnome := ⟨"Pili"⟩
def Spali : Gnome := ⟨"Spali"⟩

-- Define the conditions
axiom Eli_has_red_hood : true
axiom Eli_beard_longer_than_Pili : true
axiom Basin_found_by_gnome_with_longest_beard_and_blue_hood : true
axiom Diamond_found_by_gnome_with_shortest_beard : true

-- Define result types
inductive Item
| Diamond | Topaz | Basin

-- Assign findings according to the proof problem
def WhoFoundWhat (G : Gnome) : Item :=
if G = Eli then Item.Topaz
else if G = Pili then Item.Diamond
else Item.Basin

-- The main theorem encapsulating the proof problem
theorem find_items (hEliRedHood : true) (hEliBeard : true)
                  (hBasinBlueHood : true) (hDiamondShortestBeard : true) :
  WhoFoundWhat Pilu = Item.Diamond ∧
  WhoFoundWhat Spali = Item.Basin ∧
  WhoFoundWhat Eli = Item.Topaz :=
sorry

end find_items_l85_85308


namespace min_coach_handshakes_l85_85761

-- Definitions based on the problem conditions
def total_gymnasts : ℕ := 26
def total_handshakes : ℕ := 325

/- 
  The main theorem stating that the fewest number of handshakes 
  the coaches could have participated in is 0.
-/
theorem min_coach_handshakes (n : ℕ) (h : 0 ≤ n ∧ n * (n - 1) / 2 = total_handshakes) : 
  n = total_gymnasts → (total_handshakes - n * (n - 1) / 2) = 0 :=
by 
  intros h_n_eq_26
  sorry

end min_coach_handshakes_l85_85761


namespace sum_S_2019_l85_85083

noncomputable def a (n : ℕ) : ℝ := sorry

noncomputable def S (n : ℕ) : ℝ := (-1)^n * a n + 1 / 2^n

theorem sum_S_2019 : (∑ i in finset.range 2019, S (i + 1)) = (5 / 12 - 1 / 2 * 1 / 4^1010) := 
sorry

end sum_S_2019_l85_85083


namespace find_n_max_l85_85093

-- Define the arithmetic sequence conditions
variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}

-- Conditions of the problem
def arithmetic_sequence (a : ℕ → ℤ) := 
  ∃ d : ℤ, ∀ n, a(n+1) = a n + d

def a3 (a : ℕ → ℤ) : Prop := 
  a 2 = 7

def a1a7 (a : ℕ → ℤ) : Prop := 
  a 0 + a 6 = 10

-- The sum of the first n terms of the sequence
def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * a 0 + (n * (n - 1) / 2) * (a(1) - a(0))

-- The condition for maximizing S_n
def S_n_max (S : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ m, S m ≤ S n

-- Lean statement to prove
theorem find_n_max : 
  ∀ (a : ℕ → ℤ) (S : ℕ → ℤ), 
  arithmetic_sequence a → a3 a → a1a7 a → sum_of_first_n_terms a S → ∃ n, S_n_max S n ∧ n = 6 :=
by sorry

end find_n_max_l85_85093


namespace int_solutions_count_l85_85140

noncomputable def number_of_integer_solutions : ℕ :=
  Set.count { x : ℤ // (x - 3)^(36 - x^2) = 1 }

theorem int_solutions_count : number_of_integer_solutions = 4 := by
  sorry

end int_solutions_count_l85_85140


namespace solution_l85_85400

-- Define the functions 
def fA (x : ℝ) : ℝ := 1 / x
def fB (x : ℝ) : ℝ := Real.exp (-x)
def fC (x : ℝ) : ℝ := 1 - x^2
def fD (x : ℝ) : ℝ := x^2

-- Even function predicate
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Monotonically decreasing predicate
def is_monotonically_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≥ f y

variable x : ℝ

-- Function fD (x^2) is an even function
def check_fD_even : Prop := is_even_function fD

-- Function fD (x^2) is monotonically decreasing on (-∞, 0)
def check_fD_monotonic : Prop := is_monotonically_decreasing_on fD (Set.Iio 0)

-- Combining both properties
def fD_satisfies_conditions : Prop :=
  check_fD_even ∧ check_fD_monotonic

theorem solution : fD_satisfies_conditions :=
  sorry

end solution_l85_85400


namespace abs_sum_leq_abs_l85_85064

theorem abs_sum_leq_abs (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  |a| + |b| ≤ |a + b| :=
sorry

end abs_sum_leq_abs_l85_85064


namespace calculate_crushing_load_l85_85797

namespace CrushingLoad

-- Definitions for given conditions
def T : ℝ := 5
def H : ℝ := 10
def K : ℝ := 2
def formula (T H K : ℝ) := (30 * T ^ 3 * K) / (H ^ 3)

-- The Lean theorem statement equivalent to the proof of L = 15 / 2
theorem calculate_crushing_load :
  formula T H K = 15 / 2 :=
by
  sorry

end CrushingLoad

end calculate_crushing_load_l85_85797


namespace general_formula_l85_85504

open Nat

def a (n : ℕ) : ℚ :=
  if n = 0 then 7/6 else 0 -- Recurrence initialization with dummy else condition

-- Defining the recurrence relation as a function
lemma recurrence_relation {n : ℕ} (h : n > 0) : 
    a n = (1 / 2) * a (n - 1) + (1 / 3) := 
sorry

-- Proof of the general formula
theorem general_formula (n : ℕ) : a n = (1 / (2^n : ℚ)) + (2 / 3) :=
sorry

end general_formula_l85_85504


namespace best_fitting_model_l85_85195

theorem best_fitting_model :
  ∀ (R2_model1 R2_model2 R2_model3 R2_model4 : ℝ),
  R2_model1 = 0.99 ∧ R2_model2 = 0.88 ∧ R2_model3 = 0.50 ∧ R2_model4 = 0.20 →
  R2_model1 > R2_model2 ∧ R2_model1 > R2_model3 ∧ R2_model1 > R2_model4 :=
by
  intros R2_model1 R2_model2 R2_model3 R2_model4 h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest2,
  cases h_rest2 with h3 h4,
  split;
  rw [h1, h2, h3, h4],
  all_goals { norm_num },
  sorry

end best_fitting_model_l85_85195


namespace monic_P_Q_real_root_of_no_real_root_P_eq_Q_l85_85231

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry

theorem monic_P_Q_real_root_of_no_real_root_P_eq_Q (hP : P.monic) 
  (hQ : Q.monic) (degP : P.natDegree = 10) (degQ : Q.natDegree = 10)
  (no_real_root_PQ : ∀ x : ℝ, P.eval x ≠ Q.eval x) :
  ∃ x : ℝ, (P.comp Polynomial.X.add_one - Q.comp Polynomial.X.sub_one).eval x = 0 :=
sorry

end monic_P_Q_real_root_of_no_real_root_P_eq_Q_l85_85231


namespace magnitude_of_z_l85_85076

def z : ℂ := (3 + complex.i) / (1 + complex.i)

theorem magnitude_of_z : complex.abs z = real.sqrt 5 := by
  sorry

end magnitude_of_z_l85_85076


namespace coffee_processing_completed_l85_85407

-- Define the initial conditions
def CoffeeBeansProcessed (m n : ℕ) : Prop :=
  let mass: ℝ := 1
  let days_single_machine: ℕ := 5
  let days_both_machines: ℕ := 4
  let half_mass: ℝ := mass / 2
  let total_ground_by_June_10 := (days_single_machine * m + days_both_machines * (m + n)) = half_mass
  total_ground_by_June_10

-- Define the final proof problem
theorem coffee_processing_completed (m n : ℕ) (h: CoffeeBeansProcessed m n) : ∃ d : ℕ, d = 15 := by
  -- Processed in 15 working days
  sorry

end coffee_processing_completed_l85_85407


namespace point_in_fourth_quadrant_l85_85544

noncomputable def magnitude (z : ℂ) : ℝ := complex.abs z

theorem point_in_fourth_quadrant :
  let z := 3 + 4 * complex.I in
  let mag_z := magnitude z in
  let w := 2 + complex.I in
  let div_w := (mag_z / w) * complex.conj w / complex.norm_sq w in
  div_w = 2 - complex.I → (div_w.re > 0 ∧ div_w.im < 0) :=
by
  intros
  sorry

end point_in_fourth_quadrant_l85_85544
