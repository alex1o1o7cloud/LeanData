import Mathlib

namespace solve_for_n_l101_101764

theorem solve_for_n (n : ℚ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : 
    n = 3.5 :=
  sorry

end solve_for_n_l101_101764


namespace f_neg_eq_f_l101_101649

def f (x : ℝ) : ℝ := (x^2 + 4) / (x^2 - 4)

theorem f_neg_eq_f (x : ℝ) (h : x^2 ≠ 4) : f(-x) = f(x) :=
by
  sorry

end f_neg_eq_f_l101_101649


namespace deck_cost_correct_l101_101283

def rare_cards := 19
def uncommon_cards := 11
def common_cards := 30
def rare_card_cost := 1
def uncommon_card_cost := 0.5
def common_card_cost := 0.25

def total_deck_cost := rare_cards * rare_card_cost + uncommon_cards * uncommon_card_cost + common_cards * common_card_cost

theorem deck_cost_correct : total_deck_cost = 32 :=
  by
  -- The proof would go here.
  sorry

end deck_cost_correct_l101_101283


namespace probability_DEQ_greater_than_DFQ_and_EFQ_l101_101374

namespace EquilateralTriangle

noncomputable def probability_in_quadrilateral (DEF : Type) [triangle DEF] : ℝ :=
  let centroid := centroid DEF
  let quadrilateral_area := area (quadrilateral FGHI)
  let total_area := area DEF
  quadrilateral_area / total_area

theorem probability_DEQ_greater_than_DFQ_and_EFQ (Q : Point DEF) :
  Q ∈ interior DEF →
  (area (triangle DEQ) > area (triangle DFQ)) → 
  (area (triangle DEQ) > area (triangle EFQ)) →
  probability_in_quadrilateral DEF = 1 / 3 := 
by sorry

end EquilateralTriangle

end probability_DEQ_greater_than_DFQ_and_EFQ_l101_101374


namespace count_injective_functions_with_unique_descent_l101_101354

open Nat

theorem count_injective_functions_with_unique_descent 
  (n m : ℕ) (h₁ : 2 ≤ n) (h₂ : n ≤ m) :
  (∃ (f : Fin n → Fin m), (Injective f) ∧ (∃! i : Fin (n - 1), f i > f (i + 1)))
  = Nat.choose m n * (2^n - (n+1)) :=
sorry

end count_injective_functions_with_unique_descent_l101_101354


namespace probability_all_choose_paper_l101_101281

-- Given conditions
def probability_choice_is_paper := 1 / 3

-- The theorem to be proved
theorem probability_all_choose_paper :
  probability_choice_is_paper ^ 3 = 1 / 27 :=
sorry

end probability_all_choose_paper_l101_101281


namespace number_drawn_in_seventh_group_l101_101132

theorem number_drawn_in_seventh_group (m : ℕ) (h1 : m = 6) :
  ∃ n ∈ (finset.range 70 \ 60), n % 10 = (m + 7) % 10 := 
by {
  use 63,
  split,
  {
    split,
    { linarith, },
    { linarith, },
  },
  exact rfl,
}

end number_drawn_in_seventh_group_l101_101132


namespace max_m_value_and_position_l101_101828

variables (r : ℝ) (h_r : r > 0) (x : ℝ) (h_x : x > 0) (m : ℝ) (A B O M A' B' P : Type)

-- Assume the necessary geometric setup
noncomputable def is_semi_circle_with_tangent (A B O P : Type) := sorry
noncomputable def is_tangent_at_point (M : Type) := sorry
noncomputable def intersects_tangents (A' B' M : Type) := sorry
noncomputable def perpendicular_projection (P : Type) := sorry
noncomputable def rotation_solid_volumes (x m r : ℝ) := sorry

noncomputable def max_m_value := 2/3

theorem max_m_value_and_position (r : ℝ) (x : ℝ) (m : ℝ)
  (h1 : r > 0)
  (h2 : x > 0)
  (h3 : is_semi_circle_with_tangent A B O P)
  (h4 : is_tangent_at_point M)
  (h5 : intersects_tangents A' B' M)
  (h6 : perpendicular_projection P)
  (h7 : rotation_solid_volumes x m r) :

  m = max_m_value ∧ x = 1 :=
begin
  sorry
end

end max_m_value_and_position_l101_101828


namespace eccentricity_minor_axis_range_of_m_l101_101547

open Real

noncomputable def ellipse (m : ℝ) := {p : ℝ × ℝ | (p.1^2 / 4) + (p.2^2 / m) = 1 ∧ m > 0}

theorem eccentricity_minor_axis {m : ℝ} (hm : m = 2):
  (sqrt (4 - 2) / 2 = (sqrt 2) / 2) ∧ (2 * sqrt 2 = 2 * sqrt 2) :=
by sorry

theorem range_of_m (m : ℝ) :
  (∀ k : ℝ, ∃ x₁ x₂ y₁ y₂ : ℝ, 
    let t1 := (m + 4 * k^2) * x₁^2 + 8 * k^2 * x₁ + 4 * k^2 - 4 * m in 
    (t1 = 0 ∧ x₁ * x₂ + y₁ * y₂ = 0 ∧ (1 + k^2) * x₁ * x₂ + k^2 * (x₁ + x₂) + k^2 = 0) ∧
    (k^2 = 4 * m / (4 - 3 * m) ∧ (4 * m / (4 - 3 * m)) ≥ 0 ∧ m > 0 → 0 < m ∧ m ≤ 4 / 3 ∨ - 1 < y₁ ∧ (1 / 4 + 1 / m = 1 ∧ m = 4 / 3) ) := 
by sorry

end eccentricity_minor_axis_range_of_m_l101_101547


namespace multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l101_101094

theorem multiples_of_7_with_units_digit_7 (n : ℕ) : 
  (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) ↔ 
  n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  sorry

theorem number_of_multiples_of_7_with_units_digit_7 : 
  ∃ m, m = 3 ∧ ∀ n : ℕ, (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) → n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  use 3
  intros n
  intros hn
  split
  intro h
  cases h
  sorry

end multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l101_101094


namespace find_w_l101_101410

variables (w x y z : ℕ)

-- conditions
def condition1 : Prop := x = w / 2
def condition2 : Prop := y = w + x
def condition3 : Prop := z = 400
def condition4 : Prop := w + x + y + z = 1000

-- problem to prove
theorem find_w (h1 : condition1 w x) (h2 : condition2 w x y) (h3 : condition3 z) (h4 : condition4 w x y z) : w = 200 :=
by sorry

end find_w_l101_101410


namespace gcd_binomial_coeffs_correct_l101_101908

noncomputable def gcd_binomial_coeffs (n : ℕ) (h : n > 0) (k : ℕ) (h1 : 2^k ∣ n) (h2 : ¬ 2^(k+1) ∣ n) : ℕ :=
  let binom := λ i, Nat.choose (2 * n) i in
  Nat.gcd (List.gcd (List.map binom (List.filter (λ i, i % 2 = 1) (List.range (2 * n))))) (2^(k+1))

theorem gcd_binomial_coeffs_correct (n : ℕ) (h : n > 0) (k : ℕ) (h1 : 2^k ∣ n) (h2 : ¬ 2^(k+1) ∣ n) : gcd_binomial_coeffs n h k h1 h2 = 2^(k+1) :=
  sorry

end gcd_binomial_coeffs_correct_l101_101908


namespace hidden_dots_sum_l101_101706

-- Lean 4 equivalent proof problem definition
theorem hidden_dots_sum (d1 d2 d3 d4 : ℕ)
    (h1 : d1 ≠ d2 ∧ d1 + d2 = 7)
    (h2 : d3 ≠ d4 ∧ d3 + d4 = 7)
    (h3 : d1 = 2 ∨ d1 = 4 ∨ d2 = 2 ∨ d2 = 4)
    (h4 : d3 + 4 = 7) :
    d1 + 7 + 7 + d3 = 24 :=
sorry

end hidden_dots_sum_l101_101706


namespace find_x_l101_101114

theorem find_x (x : ℕ) (h : 2^x - 2^(x - 2) = 3 * 2^10) : x = 12 :=
by 
  sorry

end find_x_l101_101114


namespace factor_tree_value_l101_101992

theorem factor_tree_value :
  ∀ (X Y Z F G : ℕ),
  X = Y * Z → 
  Y = 7 * F → 
  F = 2 * 5 → 
  Z = 11 * G → 
  G = 7 * 3 → 
  X = 16170 := 
by
  intros X Y Z F G
  sorry

end factor_tree_value_l101_101992


namespace percentage_increase_l101_101588

variable {x y : ℝ}
variable {P : ℝ} -- percentage

theorem percentage_increase (h1 : y = x * (1 + P / 100)) (h2 : x = y * 0.5882352941176471) : P = 70 := 
by
  sorry

end percentage_increase_l101_101588


namespace not_factorial_tails_count_l101_101456

def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) -- you can extend further if needed

def is_factorial_tail (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ f(m) = n

theorem not_factorial_tails_count : 
  ∃ c : ℕ, c = 500 ∧ ∀ k : ℕ, k < 2500 → ¬is_factorial_tail k → k < 2500 - c :=
by
  sorry

end not_factorial_tails_count_l101_101456


namespace g_at_1001_l101_101686

open Function

variable (g : ℝ → ℝ)

axiom g_property : ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x
axiom g_at_1 : g 1 = 3

theorem g_at_1001 : g 1001 = -997 :=
by
  sorry

end g_at_1001_l101_101686


namespace distinct_pairs_sum_l101_101277

theorem distinct_pairs_sum (a1 b1 a2 b2 : ℕ)
  (h1 : a1 < b1) (h2 : a2 < b2)
  (h3 : (a1 + b1 * complex.I) * (b1 - a1 * complex.I) = 2020)
  (h4 : (a2 + b2 * complex.I) * (b2 - a2 * complex.I) = 2020)
  (h5 : a1 * b1 = 1010) (h6 : a2 * b2 = 1010) :
  a1 + b1 + a2 + b2 = 714 :=
  sorry

end distinct_pairs_sum_l101_101277


namespace alt_series_converges_l101_101750

noncomputable def alt_series (n : ℕ) : ℝ := (-1)^(n-1) / Real.sqrt n

theorem alt_series_converges :
  (∀ n : ℕ, n > 0 → alt_series n > 0) ∧ 
  (∀ n : ℕ, n > 0 → alt_series (n + 1) < alt_series n) ∧ 
  (Tendsto (fun n => alt_series n) atTop (𝓝 0)) →
  CauchySeq (partial_sum alt_series) ∧
  ∃ N, N ≥ 9999 := 
sorry

end alt_series_converges_l101_101750


namespace minimal_positive_period_of_function_l101_101703

theorem minimal_positive_period_of_function :
  minimal_positive_period (λ x, (Real.sin (4 * x)) / (1 + Real.cos (4 * x))) = π / 2 :=
by
  sorry

end minimal_positive_period_of_function_l101_101703


namespace common_pts_above_curve_l101_101157

open Real

theorem common_pts_above_curve {x y t : ℝ} (h1 : 0 ≤ x ∧ x ≤ 1) (h2 : 0 ≤ y ∧ y ≤ 1) (h3 : 0 < t ∧ t < 1) :
  (∀ t, y ≥ (t-1)/t * x + 1 - t) ↔ (sqrt x + sqrt y ≥ 1) := 
by
  sorry

end common_pts_above_curve_l101_101157


namespace collinear_vectors_l101_101543

theorem collinear_vectors (x : ℝ) :
  (∃ k : ℝ, (2, 4) = (k * 2, k * 4) ∧ (k * 2 = x ∧ k * 4 = 6)) → x = 3 :=
by
  intros h
  sorry

end collinear_vectors_l101_101543


namespace sum_divisors_of_37_is_38_l101_101322

theorem sum_divisors_of_37_is_38 (h : Nat.Prime 37) : (∑ d in (Finset.filter (λ d, 37 % d = 0) (Finset.range (37 + 1))), d) = 38 := by
  sorry

end sum_divisors_of_37_is_38_l101_101322


namespace sin_A_in_right_triangle_cos_given_l101_101996

theorem sin_A_in_right_triangle_cos_given {A : ℝ} (h : real.cos A = 1/2) : real.sin A = real.sqrt(3)/2 :=
sorry

end sin_A_in_right_triangle_cos_given_l101_101996


namespace negation_of_proposition_l101_101237

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 - x - 1 = 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≠ 0) :=
by sorry

end negation_of_proposition_l101_101237


namespace deck_total_cost_is_32_l101_101288

-- Define the costs of each type of card
def rare_cost := 1
def uncommon_cost := 0.50
def common_cost := 0.25

-- Define the quantities of each type of card in Tom's deck
def rare_quantity := 19
def uncommon_quantity := 11
def common_quantity := 30

-- Define the total cost calculation
def total_cost : ℝ := (rare_quantity * rare_cost) + (uncommon_quantity * uncommon_cost) + (common_quantity * common_cost)

-- Prove that the total cost of the deck is $32
theorem deck_total_cost_is_32 : total_cost = 32 := by
  sorry

end deck_total_cost_is_32_l101_101288


namespace not_factorial_tails_l101_101465

noncomputable def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) + ...

theorem not_factorial_tails :
  let n := 2500 in ∃ (k : ℕ), k = 500 ∧ ∀ m < n, ¬(f m = k) :=
by {
  sorry
}

end not_factorial_tails_l101_101465


namespace evaluate_polynomial_l101_101014

theorem evaluate_polynomial (x : ℝ) (h : x^2 - 3 * x - 9 = 0) (hx : 0 < x) : 
  x^4 - 3 * x^3 - 9 * x^2 + 27 * x - 8 = -8 := 
sorry

end evaluate_polynomial_l101_101014


namespace constant_term_proof_l101_101938

noncomputable def constant_term (n : ℕ) (x : ℝ) : ℝ :=
  (∑ r in Finset.range (n + 1), 
      if 6 - (3/2 : ℝ) * r = 0 then 
        (-1)^r * 2^(n-r) * Nat.choose n r else 0)

theorem constant_term_proof :
  ∑ i in Finset.range (6 + 1), Nat.choose 6 i = 64 →
  constant_term 6 1 = 60 :=
begin
  sorry
end

end constant_term_proof_l101_101938


namespace ratio_EZ_FZ_l101_101748

-- Define the main structures and conditions
variables {Point Line : Type} (A B C D E F W X Y Z : Point)
variables (lineBC lineWX lineEF lineAB lineAC lineDE : Line)
variables [EquilateralTriangle ABC] [EquilateralTriangle DEF]
variables [SameSide lineBC ABC DEF]
variables [Collinear E B C F] [LineIntersects lineAB lineWX W]
variables [LineIntersects lineAC lineWX X]
variables [LineIntersects lineDE lineWX Y]
variables [LineIntersects lineEF lineWX Z]

-- Given ratios
variables (h1 : AW / WB = 2 / 9)
variables (h2 : AX / CX = 5 / 6)
variables (h3 : DY / EY = 9 / 2)

-- Prove the ratio and find m + n
theorem ratio_EZ_FZ :
  ∃ (m n : ℕ), (gcd m n = 1) ∧ m + n = 33 ∧ EZ / FZ = m / n :=
sorry

end ratio_EZ_FZ_l101_101748


namespace red_car_speed_l101_101858

noncomputable def speed_blue : ℕ := 80
noncomputable def speed_green : ℕ := 8 * speed_blue
noncomputable def speed_red : ℕ := 2 * speed_green

theorem red_car_speed : speed_red = 1280 := by
  unfold speed_red
  unfold speed_green
  unfold speed_blue
  sorry

end red_car_speed_l101_101858


namespace maximum_height_l101_101822

noncomputable def h (t : ℝ) : ℝ :=
  -20 * t ^ 2 + 100 * t + 30

theorem maximum_height : 
  ∃ t : ℝ, h t = 155 ∧ ∀ t' : ℝ, h t' ≤ 155 := 
sorry

end maximum_height_l101_101822


namespace spotlight_distance_l101_101406

open Real

-- Definitions for the ellipsoid parameters
def ellipsoid_parameters (a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2 ∧ a - c = 1.5

-- Given conditions as input parameters
variables (a b c : ℝ)
variables (h_a : a = 2.7) -- semi-major axis half length
variables (h_c : c = 1.5) -- focal point distance

-- Prove that the distance from F2 to F1 is 12 cm
theorem spotlight_distance (h : ellipsoid_parameters a b c) : 2 * a - (a - c) = 12 :=
by sorry

end spotlight_distance_l101_101406


namespace inequality_verified_l101_101777

noncomputable def verify_inequality
  (n : ℕ)
  (a : Fin n → ℝ)
  (h_n : 0 < n) : Prop :=
  sqrt (∑ i, a i ^ 2 / n) ≥ ∑ i, a i / n

theorem inequality_verified
  (n : ℕ)
  (a : Fin n → ℝ)
  (h_n : 0 < n)
  : verify_inequality n a h_n :=
  sorry

end inequality_verified_l101_101777


namespace zoo_feeding_sequences_l101_101387

def num_alternating_sequences (n : ℕ) (start_gender : bool) (end_gender : bool) :=
  if n = 5 ∧ start_gender = true ∧ end_gender = false then 4 * 4 * 3 * 3 * 2 * 2 * 1 else 0

theorem zoo_feeding_sequences : 
  num_alternating_sequences 5 tt ff = 576 :=
by
  -- Initial condition: 5 pairs of animals, starting with the male lion (true), ending with the female bear (false)
  have h : num_alternating_sequences 5 tt ff = 4 * 4 * 3 * 3 * 2 * 2 * 1,
  -- proof omitted
  sorry
  -- Final result
  exact h

end zoo_feeding_sequences_l101_101387


namespace sum_of_geometric_sequence_first_9000_terms_l101_101253

noncomputable def geomSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_of_geometric_sequence_first_9000_terms (a r : ℝ) (h1 : geomSum a r 3000 = 500) (h2 : geomSum a r 6000 = 950) :
  geomSum a r 9000 = 1355 :=
by
  sorry

end sum_of_geometric_sequence_first_9000_terms_l101_101253


namespace ratio_of_female_democrats_l101_101276

theorem ratio_of_female_democrats 
    (M F : ℕ) 
    (H1 : M + F = 990)
    (H2 : M / 4 + 165 = 330) 
    (H3 : 165 = 165) : 
    165 / F = 1 / 2 := 
sorry

end ratio_of_female_democrats_l101_101276


namespace fifi_hangers_l101_101983

-- Definitions based on conditions
def P : ℕ := 7
def G : ℕ    -- green hangers
def B : ℕ := G - 1
def Y : ℕ := G - 2

-- The main theorem statement
theorem fifi_hangers : P + G + B + Y = 16 -> G = 4 := by
  sorry

end fifi_hangers_l101_101983


namespace abs_add_lt_abs_sub_l101_101648

-- Define the conditions
variables {a b : ℝ} (h1 : a * b < 0)

-- Prove the statement
theorem abs_add_lt_abs_sub (h1 : a * b < 0) : |a + b| < |a - b| := sorry

end abs_add_lt_abs_sub_l101_101648


namespace problem1_l101_101414

theorem problem1 : -3^2 + (2023 - real.pi)^0 + (1/2)^(-(3:ℤ)) = 0 :=
by sorry

end problem1_l101_101414


namespace not_factorial_tails_l101_101462

noncomputable def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) + ...

theorem not_factorial_tails :
  let n := 2500 in ∃ (k : ℕ), k = 500 ∧ ∀ m < n, ¬(f m = k) :=
by {
  sorry
}

end not_factorial_tails_l101_101462


namespace a_in_range_l101_101077

noncomputable def kOM (t : ℝ) : ℝ := (Real.log t) / t
noncomputable def kON (a t : ℝ) : ℝ := (a + a * t - t^2) / t

theorem a_in_range (a : ℝ) : 
  (∀ t ∈ Set.Ici 1, 0 ≤ (1 - Real.log t + a) / t^2 + 1) →
  a ∈ Set.Ici (-2) := 
by
  sorry

end a_in_range_l101_101077


namespace complex_division_l101_101779

theorem complex_division :
  (⟨5, -1⟩ : ℂ) / (⟨1, -1⟩ : ℂ) = (⟨3, 2⟩ : ℂ) :=
sorry

end complex_division_l101_101779


namespace sqrt_factorial_l101_101757

theorem sqrt_factorial : Real.sqrt (Real.ofNat (Nat.factorial 5) * Real.ofNat (Nat.factorial 5)) = 120 := 
by 
  sorry

end sqrt_factorial_l101_101757


namespace smallest_non_multiple_of_5_abundant_l101_101005

def properDivisors (n : ℕ) : List ℕ :=
  (List.range (n - 1 + 1) ).filter (fun d => d < n ∧ n % d = 0)

def isAbundant (n : ℕ) : Prop :=
  properDivisors n |>.sum > n

def isNotMultipleOfFive (n : ℕ) : Prop :=
  n % 5 ≠ 0

theorem smallest_non_multiple_of_5_abundant :
  ∃ n, isAbundant n ∧ isNotMultipleOfFive n ∧ 
       ∀ m, isAbundant m ∧ isNotMultipleOfFive m → n ≤ m :=
  sorry

end smallest_non_multiple_of_5_abundant_l101_101005


namespace area_of_sector_l101_101340

/-- The area of a sector of a circle with radius 10 meters and central angle 42 degrees is 35/3 * pi square meters. -/
theorem area_of_sector (r θ : ℕ) (h_r : r = 10) (h_θ : θ = 42) : 
  (θ / 360 : ℝ) * (Real.pi : ℝ) * (r : ℝ)^2 = (35 / 3 : ℝ) * (Real.pi : ℝ) :=
by {
  sorry
}

end area_of_sector_l101_101340


namespace tile_area_l101_101715

-- Define the properties and conditions of the tile

structure Tile where
  sides : Fin 9 → ℝ 
  six_of_length_1 : ∀ i : Fin 6, sides i = 1 
  congruent_quadrilaterals : Fin 3 → Quadrilateral

structure Quadrilateral where
  length : ℝ
  width : ℝ

-- Given the tile structure, calculate the area
noncomputable def area_of_tile (t: Tile) : ℝ := sorry

-- Statement: Prove the area of the tile given the conditions
theorem tile_area (t : Tile) : area_of_tile t = (4 * Real.sqrt 3 / 3) :=
  sorry

end tile_area_l101_101715


namespace well_diameter_correct_l101_101891

-- The conditions given in the problem
def depth_of_well : ℝ := 14
def cost_per_cubic_meter : ℝ := 17
def total_cost : ℝ := 1682.32
def volume_of_well : ℝ := total_cost / cost_per_cubic_meter

-- The goal is to compute and prove the diameter of the well
def diameter_of_well : ℝ := 2 * real.sqrt (volume_of_well / (real.pi * depth_of_well))

theorem well_diameter_correct : diameter_of_well = 2.996 :=
by 
  -- The detailed proof steps go here
  sorry

end well_diameter_correct_l101_101891


namespace optionA_is_incorrect_l101_101105

theorem optionA_is_incorrect :
  ¬((deriv (λ x : ℝ, x^(-2))) = (λ x : ℝ, -2 * x^(-1))) :=
by sorry

end optionA_is_incorrect_l101_101105


namespace inequality_solution_l101_101887

noncomputable def f (x : ℝ) : ℝ :=
  (2 / (x + 2)) + (4 / (x + 8))

theorem inequality_solution {x : ℝ} :
  f x ≥ 1/2 ↔ ((-8 < x ∧ x ≤ -4) ∨ (-2 ≤ x ∧ x ≤ 2)) :=
sorry

end inequality_solution_l101_101887


namespace largest_possible_value_of_s_l101_101163

theorem largest_possible_value_of_s (r s : Nat) (h1 : r ≥ s) (h2 : s ≥ 3)
  (h3 : (r - 2) * s * 61 = (s - 2) * r * 60) : s = 121 :=
sorry

end largest_possible_value_of_s_l101_101163


namespace func_necessary_sufficient_l101_101001

variable {ℝ : Type*} [NonEmpty ℝ] [OrderedCommGroup ℝ]

-- Define the function f and its properties
variable (f : ℝ → ℝ)

noncomputable def func_cond1 := ∀ x : ℝ, f(x) = f(2 - x)

noncomputable def func_cond2 := ∀ x : ℝ, (f' x) * (x - 1) > 0

-- Statement of the theorem to be proved
theorem func_necessary_sufficient (h1 : func_cond1 f) (h2 : func_cond2 f) : 
  ∀ x1 x2 : ℝ, (x1 < x2) → (f(x1) > f(x2) ↔ x1 + x2 < 2) := 
sorry

end func_necessary_sufficient_l101_101001


namespace minimum_perimeter_of_tiled_rectangle_in_meters_l101_101300

-- Define the problem conditions
def numberOfTiles : Nat := 24
def sideLengthCm : Nat := 40
def sideLengthM : Float := 0.4

-- Define the proof statement to verify the minimum perimeter
theorem minimum_perimeter_of_tiled_rectangle_in_meters :
  ∀ (n s: Nat), n = numberOfTiles → s = sideLengthCm → 
  ∃ (p : Float), p = 8 :=
by
  intro n s hn hs
  use 8
  sorry

end minimum_perimeter_of_tiled_rectangle_in_meters_l101_101300


namespace product_of_distances_l101_101198

theorem product_of_distances (A B C I I_A : Point)
    (incenter_I : is_incenter I A B C)
    (excenter_I_A : is_excenter I_A A B C)
    (r R : ℝ)
    (inradius_r : inradius I A B C r)
    (circumradius_R : circumradius A B C R) :
    distance I C * distance I I_A = 4 * R * r := 
sorry

end product_of_distances_l101_101198


namespace solution_verification_l101_101021

-- Define the differential equation
def diff_eq (y y' y'': ℝ → ℝ) : Prop :=
  ∀ x, y'' x - 4 * y' x + 5 * y x = 2 * Real.cos x + 6 * Real.sin x

-- General solution form
def general_solution (C₁ C₂ : ℝ) (y: ℝ → ℝ) : Prop :=
  ∀ x, y x = Real.exp (2 * x) * (C₁ * Real.cos x + C₂ * Real.sin x) + Real.cos x + 1/2 * Real.sin x

-- Proof problem statement
theorem solution_verification (C₁ C₂ : ℝ) (y y' y'': ℝ → ℝ) :
  (∀ x, y' x = deriv y x) →
  (∀ x, y'' x = deriv (deriv y) x) →
  diff_eq y y' y'' →
  general_solution C₁ C₂ y :=
by
  intros h1 h2 h3
  sorry

end solution_verification_l101_101021


namespace focus_parabola_l101_101892

theorem focus_parabola (f : ℝ) (d : ℝ) (y : ℝ) :
  (∀ y, ((- (1 / 8) * y^2 - f) ^ 2 + y^2 = (- (1 / 8) * y^2 - d) ^ 2)) → 
  (d - f = 4) → 
  (f^2 = d^2) → 
  f = -2 :=
by
  sorry

end focus_parabola_l101_101892


namespace sum_first_9000_terms_l101_101248

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_9000_terms (a r : ℝ) :
  geometric_sequence_sum a r 3000 = 500 →
  geometric_sequence_sum a r 6000 = 950 →
  geometric_sequence_sum a r 9000 = 1355 :=
by
  intros h1 h2
  sorry

end sum_first_9000_terms_l101_101248


namespace unique_poly_degree_4_l101_101736

theorem unique_poly_degree_4 
  (Q : ℚ[X]) 
  (h_deg : Q.degree = 4) 
  (h_lead_coeff : Q.leading_coeff = 1)
  (h_root : Q.eval (sqrt 3 + sqrt 7) = 0) : 
  Q = X^4 - 12 * X^2 + 16 ∧ Q.eval 2 = 0 :=
by sorry

end unique_poly_degree_4_l101_101736


namespace exists_k_x0_max_V_2_to_781_l101_101106

-- Definition of greatest prime factor
def greatestPrimeFactor (n : ℕ) : ℕ := sorry

-- Definition of the sequence x_i
noncomputable def sequence (x₀ : ℕ) : ℕ → ℕ
| 0 => x₀
| (n + 1) => sequence n - greatestPrimeFactor (sequence n)

-- Theorem 1: For any integer \(x_0\) greater than 1, there exists a natural number \(k(x_0)\) such that \(x_{k(x_0)+1} = 0\)
theorem exists_k_x0 (x₀ : ℕ) (hx : 1 < x₀) :
  ∃ k : ℕ, sequence x₀ k + 1 = 0 := sorry

-- Function V(x₀)
def V (x₀ : ℕ) : ℕ := sorry

-- Theorem 2: The largest number in \(V(2), V(3), \cdots, V(781)\) is 2
theorem max_V_2_to_781 : ∀ x, (2 ≤ x ∧ x ≤ 781) → V x ≤ 2 := sorry

end exists_k_x0_max_V_2_to_781_l101_101106


namespace chess_tournament_games_l101_101987

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) :
  n = 24 → total_games = 276 → (n - 1) = 23 :=
by
  intros h1 h2
  rw [h1] at h2
  sorry

end chess_tournament_games_l101_101987


namespace sum_first_9000_terms_l101_101246

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_9000_terms (a r : ℝ) :
  geometric_sequence_sum a r 3000 = 500 →
  geometric_sequence_sum a r 6000 = 950 →
  geometric_sequence_sum a r 9000 = 1355 :=
by
  intros h1 h2
  sorry

end sum_first_9000_terms_l101_101246


namespace pheromone_correct_choice_l101_101403

noncomputable def pheromone_formula := (10, 18, 1) -- Represent (C count, H count, O count)
def carbon_count (formula : Nat × Nat × Nat) := formula.1
def hydrogen_count (formula : Nat × Nat × Nat) := formula.2
def oxygen_count (formula : Nat × Nat × Nat) := formula.3

def statement_1 (formula : Nat × Nat × Nat) : Prop :=
  carbon_count formula > 0 ∧ hydrogen_count formula > 0 ∧ oxygen_count formula > 0

def mass_ratio (formula : Nat × Nat × Nat) : (Nat × Nat × Nat) :=
  (12 * carbon_count formula, hydrogen_count formula, 16 * oxygen_count formula)

def statement_2 (formula : Nat × Nat × Nat) : Prop :=
  mass_ratio formula = (120, 18, 16)

def statement_3 : Prop := -- Compound vs elements semantics
  False -- Given the problem's definition

def total_atoms (formula : Nat × Nat × Nat) : Nat :=
  carbon_count formula + hydrogen_count formula + oxygen_count formula

def statement_4 (formula : Nat × Nat × Nat) : Prop :=
  total_atoms formula = 29

def correct_choice (formula : Nat × Nat × Nat) : Prop :=
  statement_1 formula ∧ ¬statement_2 formula ∧ ¬statement_3 ∧ statement_4 formula

theorem pheromone_correct_choice :
  correct_choice pheromone_formula := by
  sorry

end pheromone_correct_choice_l101_101403


namespace exists_8_equilateral_triangles_with_conditions_l101_101428

noncomputable def equilateral_triangle_exists 
  (O1 O2 : Point) (r1 r2 : ℝ) 
  (C: Point → Bool) (radical_axis : Line) : Prop :=
    ∃ t1 t2 t3 t4 t5 t6 t7 t8 : (Triangle), 
      (∀ t, t ∈ {t1, t2, t3, t4, t5, t6, t7, t8} -> 
        (equilateral t ∧ tangent_touches_circle t.side1 (circle O1 r1) ∧ tangent_touches_circle t.side2 (circle O2 r2) ∧ on_radical_axis t.vertex3 O1 O2))

-- Define placeholder structures and properties to avoid incomplete references
structure Point
constant Line : Type
structure Circle (center : Point) (radius : ℝ)
structure Triangle :=
  (side1: Line)
  (side2: Line)
  (vertex3: Point)
def equilateral : Triangle → Prop := sorry
def tangent_touches_circle : Line → Circle → Prop := sorry
def on_radical_axis : Point → Point → Point → Prop := sorry

-- Statement of the mathematically equivalent proof problem as a Lean 4 statement
theorem exists_8_equilateral_triangles_with_conditions
  (O1 O2 : Point) (r1 r2 : ℝ) 
  (radical_axis : Line) : 
  ∃ t : Set Triangle, 
    (∀ t, t ∈ t → 
      ∃ C : Point → Prop, 
        C t.vertex3 ∧ 
        equilateral t ∧ 
        tangent_touches_circle t.side1 (circle O1 r1) ∧ 
        tangent_touches_circle t.side2 (circle O2 r2) ∧ 
        on_radical_axis t.vertex3 O1 O2 ∧ 
        8 = (finite.to_finset t).card) :=
begin
  -- The detailed proof is not required
  sorry
end

end exists_8_equilateral_triangles_with_conditions_l101_101428


namespace range_of_k_l101_101577

-- Conditions
def f (k x : ℝ) : ℝ := (√3) * Real.sin ((π * x) / k)
def circle (k x y : ℝ) : Prop := x^2 + y^2 = k^2

-- Question: Prove the range of k such that |k| ≥ 2
theorem range_of_k (k : ℝ) :
  (∃ x_max, f k x_max = √3 ∧ circle k x_max (√3)) →
  (∃ x_min, f k x_min = -√3 ∧ circle k x_min (-√3)) →
  |k| ≥ 2 :=
sorry

end range_of_k_l101_101577


namespace find_a_for_distinct_solutions_l101_101889

theorem find_a_for_distinct_solutions :
  ∃ (a : ℝ), 
  ((∃ x1 x2 : ℝ, -π/6 ≤ x1 ∧ x1 ≤ 3*π/2 ∧ -π/6 ≤ x2 ∧ x2 ≤ 3*π/2 ∧ x1 ≠ x2 ∧
  (2 * sin x1 + a^2 + a)^3 - (cos (2 * x1) + 3 * a * sin x1 + 11)^3 = 
  12 - 2 * sin^2 x1 + (3 * a - 2) * sin x1 - a^2 - a ∧
  (2 * sin x2 + a^2 + a)^3 - (cos (2 * x2) + 3 * a * sin x2 + 11)^3 = 
  12 - 2 * sin^2 x2 + (3 * a - 2) * sin x2 - a^2 - a))  ∨
  (∃ (a : ℝ), ((2.5 ≤ a ∧ a < 4) ∨ (-5 ≤ a ∧ a < -2))) :=
sorry

end find_a_for_distinct_solutions_l101_101889


namespace num_valid_x0_values_l101_101044

theorem num_valid_x0_values : 
  ∃ (s : Finset ℝ), (∀ x_0 ∈ s, 0 ≤ x_0 ∧ x_0 < 1 ∧ let f (x : ℝ) := if 2 * x < 1 then 2 * x else 2 * x - 1 
                                                                in f (f (f x_0)) = x_0)
  ∧ s.card = 8 := 
sorry

end num_valid_x0_values_l101_101044


namespace tetrahedral_intersection_parallelogram_l101_101196

-- Definitions for the problem
variables {Point Plane Line : Type}
variables (TetrahedralAngle : Point → Plane → Prop)
variables (intersection_line : Plane → Plane → Line)
variables (parallel_planes : Plane → Plane → Prop)

-- The problem statement
theorem tetrahedral_intersection_parallelogram
  (α α' β β' : Plane)
  (a b : Line)
  (vertex : Point)
  (hαα' : intersection_line α α' = a)
  (hββ' : intersection_line β β' = b)
  (h_parallel α α' : parallel_planes α α')
  (h_parallel β β' : parallel_planes β β')
  (h_a_through_vertex : TetrahedralAngle vertex α)
  (h_b_through_vertex : TetrahedralAngle vertex β)
  : ∃ (π : Plane), ∃ (lα lα' lβ lβ' : Line),
    lα = intersection_line α π ∧
    lα' = intersection_line α' π ∧
    lβ = intersection_line β π ∧
    lβ' = intersection_line β' π ∧
    lα = a ∧
    lα' = a ∧
    lβ = b ∧
    lβ' = b ∧
    parallel_planes lα lα' ∧
    parallel_planes lβ lβ' := 
sorry

end tetrahedral_intersection_parallelogram_l101_101196


namespace inverse_proportion_relation_l101_101922

theorem inverse_proportion_relation :
  let y1 := 3 / (-2)
  let y2 := 3 / (-1)
  let y3 := 3 / 1
  y2 < y1 ∧ y1 < y3 :=
by
  -- Variable definitions according to conditions
  let y1 := 3 / (-2)
  let y2 := 3 / (-1)
  let y3 := 3 / 1
  -- Proof steps go here (not required for the statement)
  -- Since proof steps are omitted, we use sorry to indicate it
  sorry

end inverse_proportion_relation_l101_101922


namespace not_perfect_square_l101_101653

noncomputable theory

-- Define the relevant parameters and conditions
variables (a b : ℕ)
variables (ha : a ≠ 0) (hb : b ≠ 0)

-- Define the main theorem stating that the given integer is not a perfect square
theorem not_perfect_square : ¬ ∃ c : ℤ, a^2 + ⌈(4 * a^2 : ℚ) / b⌉ = c^2 := 
sorry

end not_perfect_square_l101_101653


namespace critics_voted_same_actor_actress_l101_101787

theorem critics_voted_same_actor_actress :
  ∃ (critic1 critic2 : ℕ) 
  (actor_vote1 actor_vote2 actress_vote1 actress_vote2 : ℕ),
  1 ≤ critic1 ∧ critic1 ≤ 3366 ∧
  1 ≤ critic2 ∧ critic2 ≤ 3366 ∧
  (critic1 ≠ critic2) ∧
  ∃ (vote_count : Fin 100 → ℕ) 
  (actor actress : Fin 3366 → Fin 100),
  (∀ n : Fin 100, ∃ act : Fin 100, vote_count act = n + 1) ∧
  actor critic1 = actor_vote1 ∧ actress critic1 = actress_vote1 ∧
  actor critic2 = actor_vote2 ∧ actress critic2 = actress_vote2 ∧
  actor_vote1 = actor_vote2 ∧ actress_vote1 = actress_vote2 :=
by
  -- Proof omitted
  sorry

end critics_voted_same_actor_actress_l101_101787


namespace cos_minus_sin_l101_101904

theorem cos_minus_sin (α : ℝ) (h1 : sin α * cos α = 1 / 8) (h2 : π / 4 < α ∧ α < π / 2) :
  cos α - sin α = - (real.sqrt 3) / 2 :=
sorry

end cos_minus_sin_l101_101904


namespace curve_line_intersect_range_l101_101579

theorem curve_line_intersect_range :
  ∀ k : ℝ, (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ y_sq_unit_circle x1 = y_line k x1 ∧ y_sq_unit_circle x2 = y_line k x2) →
  0 < k ∧ k ≤ 1/2 :=
by
  -- Definitions
  let y_sq_unit_circle (x : ℝ) : ℝ := sqrt (1 - x^2)
  let y_line (k x : ℝ) : ℝ := k * (x - 1) + 1
  sorry

end curve_line_intersect_range_l101_101579


namespace quadratic_is_perfect_square_l101_101273

theorem quadratic_is_perfect_square (a b c : ℤ) 
  (h : ∀ x : ℤ, ∃ d e : ℤ, a*x^2 + b*x + c = (d*x + e)^2) : 
  ∃ d e : ℤ, a = d^2 ∧ b = 2*d*e ∧ c = e^2 :=
sorry

end quadratic_is_perfect_square_l101_101273


namespace international_postage_surcharge_l101_101860

theorem international_postage_surcharge 
  (n_letters : ℕ) 
  (std_postage_per_letter : ℚ) 
  (n_international : ℕ) 
  (total_cost : ℚ) 
  (cents_per_dollar : ℚ) 
  (std_total_cost : ℚ) 
  : 
  n_letters = 4 →
  std_postage_per_letter = 108 / 100 →
  n_international = 2 →
  total_cost = 460 / 100 →
  cents_per_dollar = 100 →
  std_total_cost = n_letters * std_postage_per_letter →
  (total_cost - std_total_cost) / n_international * cents_per_dollar = 14 := 
sorry

end international_postage_surcharge_l101_101860


namespace sues_nuts_percentage_l101_101212

-- Define the conditions as formal Lean statements
variables (N: ℝ) -- percentage of nuts in Sue's trail mix
variable (S_dried_fruit : ℝ := 70) -- percentage of dried fruit in Sue's trail mix
variable (J_nuts : ℝ := 60) -- percentage of nuts in Jane's trail mix
variable (J_chocolate_chips : ℝ := 40) -- percentage of chocolate chips in Jane's trail mix
variable (combined_nuts : ℝ := 45) -- percentage of nuts in the combined mixture
variable (combined_dried_fruit : ℝ := 35) -- percentage of dried fruit in the combined mixture

-- Theorem statement to prove
theorem sues_nuts_percentage : 
  (N + J_nuts) / 2 = combined_nuts →
  (combined_dried_fruit) = S_dried_fruit / 2 →
  N = 30 :=
begin
  sorry,
end

end sues_nuts_percentage_l101_101212


namespace cos_double_angle_l101_101041

variable (θ : ℝ)

theorem cos_double_angle (h : Real.tan (θ + Real.pi / 4) = 3) : Real.cos (2 * θ) = 3 / 5 :=
sorry

end cos_double_angle_l101_101041


namespace median_AD_eq_altitude_BH_eq_l101_101989

noncomputable def Point := (ℝ × ℝ)
noncomputable def Line := {l : ℝ × ℝ × ℝ // l ≠ 0}

def A : Point := (1, 3)
def B : Point := (5, 1)
def C : Point := (-1, -1)

-- Definitions for the midpoints and slope calculations
def midpoint (p1 p2 : Point) : Point :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  ((x1 + x2) / 2, (y1 + y2) / 2)

def slope (p1 p2 : Point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (y2 - y1) / (x2 - x1)

def line_from_points (p1 p2 : Point) : Line :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  ⟨(y2 - y1, x1 - x2, x2*y1 - x1*y2), by sorry⟩

def line_from_point_slope (p : Point) (m : ℝ) : Line :=
  let (x1, y1) := p
  ⟨(m, -1, y1 - m*x1), by sorry⟩

-- Defining the points for computation
def D : Point := midpoint B C -- Midpoint of BC
def kAC : ℝ := slope A C -- Slope of AC
def kBH : ℝ := -(1 / kAC) -- Slope of BH, the negative reciprocal of AC

-- Lean statements for the proof problems
theorem median_AD_eq : line_from_points A D = ⟨(3, 1, -6), by sorry⟩ := sorry

theorem altitude_BH_eq : line_from_point_slope B kBH = ⟨(1, 2, -7), by sorry⟩ := sorry

end median_AD_eq_altitude_BH_eq_l101_101989


namespace deck_total_cost_is_32_l101_101287

-- Define the costs of each type of card
def rare_cost := 1
def uncommon_cost := 0.50
def common_cost := 0.25

-- Define the quantities of each type of card in Tom's deck
def rare_quantity := 19
def uncommon_quantity := 11
def common_quantity := 30

-- Define the total cost calculation
def total_cost : ℝ := (rare_quantity * rare_cost) + (uncommon_quantity * uncommon_cost) + (common_quantity * common_cost)

-- Prove that the total cost of the deck is $32
theorem deck_total_cost_is_32 : total_cost = 32 := by
  sorry

end deck_total_cost_is_32_l101_101287


namespace color_lines_no_blue_perimeter_l101_101854

theorem color_lines_no_blue_perimeter (n : ℕ) :
  ∃ k, k ≥ ⌊√(n / 2)⌋ ∧
       ∀ (lines : fin n → line), (∀ i j, i ≠ j → ¬parallel (lines i) (lines j)) ∧
       (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬concurrent (lines i) (lines j) (lines k)) →
       ∃ (coloring : fin n → bool), (coloring_count coloring = k) ∧
                                     (∀ region, is_finite_region region → 
                                                 perimeter_color region coloring ≠ all_blue) := sorry

-- Definitions to make this precise would follow.

end color_lines_no_blue_perimeter_l101_101854


namespace coffee_cups_weekend_l101_101362

theorem coffee_cups_weekend (brew_per_hour : ℕ) (hours_per_day : ℕ)
                            (weekdays : ℕ) (total_cups_per_week : ℕ)
                            (weekend_cups : ℕ) :
    brew_per_hour = 10 →
    hours_per_day = 5 →
    weekdays = 5 →
    total_cups_per_week = 370 →
    weekend_cups = total_cups_per_week - (brew_per_hour * hours_per_day * weekdays) →
    weekend_cups = 120 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, mul_assoc] at h5
  simp at h5
  assumption

end coffee_cups_weekend_l101_101362


namespace martha_blue_butterflies_l101_101666

variables (B Y : Nat)

theorem martha_blue_butterflies (h_total : B + Y + 5 = 11) (h_twice : B = 2 * Y) : B = 4 :=
by
  sorry

end martha_blue_butterflies_l101_101666


namespace value_of_2_sin_alpha_l101_101970

theorem value_of_2_sin_alpha (α : ℝ) (hα : α = real.pi / 4) : 2 * real.sin α = real.sqrt 2 :=
by
  sorry

end value_of_2_sin_alpha_l101_101970


namespace generalized_trig_identity_l101_101548

theorem generalized_trig_identity (θ : ℝ) : 
    sin(θ)^2 + cos(θ + (30 * (π / 180)))^2 + sin(θ) * cos(θ + (30 * (π / 180))) = 3 / 4 := 
by
  sorry

end generalized_trig_identity_l101_101548


namespace range_of_t_l101_101660

theorem range_of_t
  (f : ℝ → ℝ)
  (a t : ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, x < y → f x ≤ f y)
  (h_f_neg1 : f (-1) = -1)
  (h_a : a ∈ set.Icc (-1:ℝ) 1)
  (h_ineq : ∀ x ∈ set.Icc (-1:ℝ) 1, f x ≤ t^2 - 2 * a * t + 1) :
  t ≥ 2 ∨ t ≤ -2 ∨ t = 0 :=
sorry

end range_of_t_l101_101660


namespace zero_point_probability_l101_101824

open Set

theorem zero_point_probability :
  ∀ a ∈ Icc (-2 : ℝ) 2, 
  (∃ x, 4^x - a * 2^(x + 1) + 1 = 0) → 
  (measure_of (Icc 1 2) (measure_space.volume.to_measure (by {})) / measure_of (Icc (-2 : ℝ) 2) (measure_space.volume.to_measure (by {}))) = 1 / 4 :=
sorry

end zero_point_probability_l101_101824


namespace problem_solution_l101_101563

-- Definitions of the complex numbers and conditions
variables (a b c d e f : ℝ)

def complex1 : ℂ := ⟨a, b⟩
def complex2 : ℂ := ⟨c, d⟩
def complex3 : ℂ := ⟨e, f⟩

-- Stating the problem conditions
def condition_b : Prop := b = 4
def condition_e : Prop := e = -2 * a - c
def condition_sum : Prop := complex1 + complex2 + complex3 = ⟨0, 5⟩

-- The proof problem statement
theorem problem_solution (h_b : condition_b) (h_e : condition_e) (h_sum : condition_sum) : d + 2 * f = 1 :=
sorry

end problem_solution_l101_101563


namespace Q_evaluation_at_2_l101_101735

noncomputable def Q : Polynomial ℚ := 
  (Polynomial.X^2 + Polynomial.C 4)^2

theorem Q_evaluation_at_2 : 
  Q.eval 2 = 64 :=
by 
  -- We'll skip the proof as per the instructions.
  sorry

end Q_evaluation_at_2_l101_101735


namespace general_term_sequence_x_l101_101630

-- Definitions used in Lean statement corresponding to the conditions.
noncomputable def sequence_a (n : ℕ) : ℝ := sorry

noncomputable def sequence_x (n : ℕ) : ℝ := sorry

axiom condition_1 : ∀ n : ℕ, 
  ((sequence_a (n + 2))⁻¹ = ((sequence_a n)⁻¹ + (sequence_a (n + 1))⁻¹) / 2)

axiom condition_2 {n : ℕ} : sequence_x n > 0

axiom condition_3 : sequence_x 1 = 3

axiom condition_4 : sequence_x 1 + sequence_x 2 + sequence_x 3 = 39

axiom condition_5 (n : ℕ) : (sequence_x n)^(sequence_a n) = 
  (sequence_x (n + 1))^(sequence_a (n + 1)) ∧ 
  (sequence_x (n + 1))^(sequence_a (n + 1)) = 
  (sequence_x (n + 2))^(sequence_a (n + 2))

-- Theorem stating that the general term of sequence {x_n} is 3^n.
theorem general_term_sequence_x : ∀ n : ℕ, sequence_x n = 3^n :=
by
  sorry

end general_term_sequence_x_l101_101630


namespace maximum_length_of_third_side_l101_101214

noncomputable def maximum_third_side (a b : ℝ) (A B C : ℝ) : ℝ :=
  if A = 60 ∨ B = 60 ∨ C = 60 then
    let c_squared := a^2 + b^2 - 2 * a * b * Real.cos (Real.pi / 3)
    in Real.sqrt c_squared
  else 0

theorem maximum_length_of_third_side {A B C : ℝ} {a b : ℝ}
  (h1 : a = 7) (h2 : b = 24)
  (h3 : Real.cos (2 * A) + Real.cos (2 * B) + Real.cos (2 * C) = 1)
  (h4 : A = 60 ∨ B = 60 ∨ C = 60) :
  maximum_third_side 7 24 A B C = Real.sqrt 457 :=
by {
  sorry
}

end maximum_length_of_third_side_l101_101214


namespace min_length_of_normal_chord_l101_101707

/-- 
Given a parabola P with focus a distance m from the directrix,
and a chord AB normal to P at A, the minimum length of AB is 3√3 * p.
-/
theorem min_length_of_normal_chord 
  (p m : ℝ) (h_focus_dist : m = p / 2) 
  (A B : ℝ × ℝ) 
  (hA_on_parabola : A.2 ^ 2 = 2 * p * A.1)
  (hAB_normal : A.2 = -2 * A.2 * (B.1 - A.1) + 4 * p * (A.2 ^ 3) + 2 * p * A.2)
  : dist A B >= 3 * real.sqrt 3 * p := sorry

end min_length_of_normal_chord_l101_101707


namespace largest_k_log3_l101_101475

-- Define the tower function T 
def T : ℕ → ℝ 
| 1 := 3
| (n + 1) := 3 ^ (T n)

-- Definitions of A and B
def A := T 4
def B := A ^ A

-- Prove that the largest integer k for which the nested log_3 is defined is 4
theorem largest_k_log3 (k : ℕ) : 
  (k = 4) → ∀ B, ∃ A, A = 3 ^ (3 ^ (3 ^ 27)) → B = A ^ A → ¬ defined (iterate (log 3) k B) := sorry

end largest_k_log3_l101_101475


namespace complex_division_product_l101_101929

theorem complex_division_product
  (i : ℂ)
  (h_exp: i * i = -1)
  (a b : ℝ)
  (h_div: (1 + 7 * i) / (2 - i) = a + b * i)
  : a * b = -3 := 
sorry

end complex_division_product_l101_101929


namespace count_multiples_of_7_with_units_digit_7_less_than_150_l101_101084

-- Definition of the problem's conditions and expected result
theorem count_multiples_of_7_with_units_digit_7_less_than_150 : 
  let multiples_of_7 := { n : Nat // n < 150 ∧ n % 7 = 0 }
  let units_digit_7_multiples := { n : Nat // n < 150 ∧ n % 7 = 0 ∧ n % 10 = 7 }
  List.length (units_digit_7_multiples.members) = 3 :=
by
  sorry

end count_multiples_of_7_with_units_digit_7_less_than_150_l101_101084


namespace value_of_n_l101_101763

theorem value_of_n (n : ℝ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : n = 3.5 :=
by sorry

end value_of_n_l101_101763


namespace general_formula_an_sum_b_formula_l101_101661

-- Definitions for the given conditions
def seq_a (n : ℕ) : ℕ := 
  if n = 0 then 0 -- Not defined for n=0, but ensuring n ∈ ℕ as per Lean's type system.
  else if n = 1 then 1
  else 4 * 3^(n - 2)

def sum_first_n (n : ℕ) : ℕ := 
  nat.rec_on n 0 (λ n sum_n, sum_n + seq_a (n + 1))

axiom initial_condition : seq_a 1 = 1
axiom recursive_condition : ∀ n : ℕ, n > 0 → sum_first_n (n + 1) = 3 * (sum_first_n n) + 1

def seq_b (n : ℕ) : ℚ := 
  if n = 1 then 8 / (seq_a 2 - seq_a 1)
  else 8 * n / (seq_a (n + 1) - seq_a n)

def sum_b (n : ℕ) : ℚ := 
  (finset.range n).sum (λ i, seq_b (i + 1))

-- Proof problems
theorem general_formula_an : 
  ∀ n : ℕ, 
  if n = 1 then seq_a n = 1
  else seq_a n = 4 * 3^(n - 2) :=
sorry

theorem sum_b_formula : 
  ∀ n : ℕ, 
  sum_b n = 77 / 12 - (2 * n + 3) / (4 * 3^(n - 2)) :=
sorry

end general_formula_an_sum_b_formula_l101_101661


namespace unicorn_tether_l101_101832

theorem unicorn_tether (a b c : ℕ) (h_c_prime : Prime c) :
  (∃ (a b c : ℕ), c = 1 ∧ (25 - 15 = 10 ∧ 10^2 + 10^2 = 15^2 ∧ 
  a = 10 ∧ b = 125) ∧ a + b + c = 136) :=
  sorry

end unicorn_tether_l101_101832


namespace sum_of_first_9000_terms_of_geometric_sequence_l101_101261

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_9000_terms_of_geometric_sequence 
  (a r : ℝ)
  (h₁ : geometric_sum a r 3000 = 500)
  (h₂ : geometric_sum a r 6000 = 950)
  : geometric_sum a r 9000 = 1355 :=
sorry

end sum_of_first_9000_terms_of_geometric_sequence_l101_101261


namespace range_of_a_l101_101112

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Iic (1 / 2), deriv (λ x, a * x^2 - (2 - a) * x + 1) x ≤ 0) ↔ 0 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l101_101112


namespace count_multiples_of_7_with_units_digit_7_less_than_150_l101_101085

-- Definition of the problem's conditions and expected result
theorem count_multiples_of_7_with_units_digit_7_less_than_150 : 
  let multiples_of_7 := { n : Nat // n < 150 ∧ n % 7 = 0 }
  let units_digit_7_multiples := { n : Nat // n < 150 ∧ n % 7 = 0 ∧ n % 10 = 7 }
  List.length (units_digit_7_multiples.members) = 3 :=
by
  sorry

end count_multiples_of_7_with_units_digit_7_less_than_150_l101_101085


namespace geometric_sequence_sum_l101_101269

noncomputable theory

-- Define the conditions and the problem statement
theorem geometric_sequence_sum (a r : ℝ) (h₁ : a * (1 - r^3000) / (1 - r) = 500) 
  (h₂ : a * (1 - r^6000) / (1 - r) = 950) : 
  a * (1 - r^9000) / (1 - r) = 1355 := 
sorry

end geometric_sequence_sum_l101_101269


namespace abs_sum_binom_expansion_l101_101943

theorem abs_sum_binom_expansion :
  let a := (1 - 3 * x)^9,
  let sum_absolutes := ∑ k in Finset.range 10, |(Nat.choose 9 k) * (-3)^k|,
  sum_absolutes = 4^9 :=
by
  sorry

end abs_sum_binom_expansion_l101_101943


namespace Nishita_preferred_shares_l101_101192

variable (P : ℕ)

def preferred_share_dividend : ℕ := 5 * P
def common_share_dividend : ℕ := 3500 * 3  -- 3.5 * 1000

theorem Nishita_preferred_shares :
  preferred_share_dividend P + common_share_dividend = 16500 → P = 1200 :=
by
  unfold preferred_share_dividend common_share_dividend
  intro h
  sorry

end Nishita_preferred_shares_l101_101192


namespace problem_solution_l101_101658

theorem problem_solution (x y : ℝ)
  (h : x * y + x / y + y / x = 3) :
  (∃ a ∈ {3, 4}, (x + 1) * (y + 1) = a) → 
  (x + 1) * (y + 1) ∈ {3, 4} → 
  3 + 4 = 7 :=
by
  -- The proof is omitted
  sorry

end problem_solution_l101_101658


namespace multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l101_101098

theorem multiples_of_7_with_units_digit_7 (n : ℕ) : 
  (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) ↔ 
  n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  sorry

theorem number_of_multiples_of_7_with_units_digit_7 : 
  ∃ m, m = 3 ∧ ∀ n : ℕ, (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) → n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  use 3
  intros n
  intros hn
  split
  intro h
  cases h
  sorry

end multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l101_101098


namespace angle_equality_l101_101900

-- Definitions for the problem conditions
variables {A B C M N : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space M] [metric_space N]

-- Assume that triangle ABC is a right triangle at B
axiom right_triangle (A B C : Type) : A ≠ B → B ≠ C → A ≠ C → angle B A C = 90

-- Assume M is an arbitrary point on the side BC
axiom point_on_BC (M : Type) : M ∈ segment B C

-- Assume MN is perpendicular to AB
axiom perp_MN_to_AB (M N A B : Type) : N ∈ line A B → angle M N A = 90

-- The proof goal
theorem angle_equality {A B C M N : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space M] [metric_space N]
  (h₁ : right_triangle A B C)
  (h₂ : point_on_BC M)
  (h₃ : perp_MN_to_AB M N A B) :
  angle M A N = angle M C N :=
sorry

end angle_equality_l101_101900


namespace reena_interest_paid_l101_101200

-- Definitions based on conditions
def principal : ℝ := 1200
def rate : ℝ := 0.03
def time : ℝ := 3

-- Definition of simple interest calculation based on conditions
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Statement to prove that Reena paid $108 as interest
theorem reena_interest_paid : simple_interest principal rate time = 108 := by
  sorry

end reena_interest_paid_l101_101200


namespace find_a_range_l101_101072

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then |x| + 2 else x + 2 / x

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x / 2 + a|) ↔ (-2 ≤ a ∧ a ≤ 2) :=
by
  sorry

end find_a_range_l101_101072


namespace minimal_sum_of_matrix_elements_l101_101173

theorem minimal_sum_of_matrix_elements (a b c d : ℤ) 
    (h1 : a ≠ 0) 
    (h2 : b ≠ 0) 
    (h3 : c ≠ 0) 
    (h4 : d ≠ 0) 
    (h5 : (Matrix.vecCons (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
                         (Matrix.vecCons (Matrix.vecCons c (Matrix.vecCons d Matrix.vecEmpty))
                         (Matrix.vecCons 0 (Matrix.vecCons 0 Matrix.vecEmpty))))
          ^2 = 
    (Matrix.vecCons (Matrix.vecCons 12 (Matrix.vecCons 0 Matrix.vecEmpty))
                   (Matrix.vecCons (Matrix.vecCons 0 (Matrix.vecCons 12 Matrix.vecEmpty))
                   (Matrix.vecCons 0 (Matrix.vecCons 0 Matrix.vecEmpty))))) :
    ∃ a b c d, |a| + |b| + |c| + |d| = 10 :=
sorry

end minimal_sum_of_matrix_elements_l101_101173


namespace Q_at_2_l101_101725

-- Define the polynomial Q(x)
def Q (x : ℚ) : ℚ := x^4 - 8 * x^2 + 16

-- Define the properties of Q(x)
def is_special_polynomial (P : (ℚ → ℚ)) : Prop := 
  degree P = 4 ∧ leading_coeff P = 1 ∧ P.is_root(√3 + √7)

-- Problem: Prove Q(2) = 0 given the polynomial Q meets the conditions.
theorem Q_at_2 (Q : ℚ → ℚ) (h1 : degree Q = 4) (h2 : leading_coeff Q = 1) (h3 : is_root Q (√3 + √7)) : 
  Q 2 = 0 := by
  sorry

end Q_at_2_l101_101725


namespace number_of_integer_points_in_triangle_l101_101240

theorem number_of_integer_points_in_triangle : 
  let triangle_OAB := {p : ℤ × ℤ | 0 < p.1 ∧ p.1 < 100 ∧ 0 < p.2 ∧ p.2 < 2 * p.1} in
  finset.card (finset.filter (λ p, p ∈ triangle_OAB) (finset.Icc (0, 0) (100, 200))) = 9801 :=
by 
  sorry

end number_of_integer_points_in_triangle_l101_101240


namespace find_a_b_g_increasing_l101_101550

noncomputable def f (a b x : ℝ) : ℝ := (1 + a * x^2) / (x + b)

noncomputable def g (a b x : ℝ) : ℝ := x * f a b x

theorem find_a_b (a b : ℝ) (h₁ : f a b 1 = 3) (h₂ : ∀ x : ℝ, g a b x = g a b (-x)) :
  a = 2 ∧ b = 0 := 
  sorry

theorem g_increasing (a b : ℝ) (h₁ : f a b 1 = 3) (h₂ : ∀ x : ℝ, g a b x = g a b (-x)) 
  (ha : a = 2) (hb : b = 0) : 
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → g a b x₁ < g a b x₂ := 
  sorry

end find_a_b_g_increasing_l101_101550


namespace diagonal_of_rectangle_l101_101752

noncomputable def L : ℝ := 40 * Real.sqrt 3
noncomputable def W : ℝ := 30 * Real.sqrt 3
noncomputable def d : ℝ := Real.sqrt (L^2 + W^2)

theorem diagonal_of_rectangle :
  d = 50 * Real.sqrt 3 :=
by sorry

end diagonal_of_rectangle_l101_101752


namespace geometric_sequence_sum_l101_101256

theorem geometric_sequence_sum
  (a r : ℝ)
  (h1 : a * (1 - r ^ 3000) / (1 - r) = 500)
  (h2 : a * (1 - r ^ 6000) / (1 - r) = 950) :
  a * (1 - r ^ 9000) / (1 - r) = 1355 :=
sorry

end geometric_sequence_sum_l101_101256


namespace sqrt_factorial_mul_self_eq_sqrt_factorial_mul_self_value_l101_101759

theorem sqrt_factorial_mul_self_eq :
  sqrt ((5!) * (5!)) = 5! :=
by sorry

theorem sqrt_factorial_mul_self_value :
  sqrt ((5!) * (5!)) = 120 :=
by {
  rw sqrt_factorial_mul_self_eq,
  norm_num,
  exact rfl,
  sorry
}

end sqrt_factorial_mul_self_eq_sqrt_factorial_mul_self_value_l101_101759


namespace truck_speed_in_mph_l101_101243

-- Definitions based on the conditions
def truck_length : ℝ := 66  -- Truck length in feet
def tunnel_length : ℝ := 330  -- Tunnel length in feet
def exit_time : ℝ := 6  -- Exit time in seconds
def feet_to_miles : ℝ := 5280  -- Feet per mile

-- Problem statement
theorem truck_speed_in_mph :
  ((tunnel_length + truck_length) / exit_time) * (3600 / feet_to_miles) = 45 := 
sorry

end truck_speed_in_mph_l101_101243


namespace rectangle_area_and_diagonal_length_l101_101141

theorem rectangle_area_and_diagonal_length
  (A B C D : Point)
  (AB AC AD BD : ℝ)
  (hAB : AB = 15)
  (hAC : AC = 17)
  (hRect : rectangle A B C D)
  (hDiag : AC = distance A C) :
  area (rectangle A B C D) = 120 ∧ distance B D = 17 := by
  sorry

end rectangle_area_and_diagonal_length_l101_101141


namespace log_relationship_l101_101038

noncomputable def log_m (m x : ℝ) : ℝ := Real.log x / Real.log m

theorem log_relationship (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1) :
  log_m m 0.3 > log_m m 0.5 :=
by
  sorry

end log_relationship_l101_101038


namespace roots_quadratic_reciprocal_l101_101685

theorem roots_quadratic_reciprocal (x1 x2 : ℝ) (h1 : x1 + x2 = -8) (h2 : x1 * x2 = 4) :
  (1 / x1) + (1 / x2) = -2 :=
sorry

end roots_quadratic_reciprocal_l101_101685


namespace general_formula_sequence_sum_of_first_n_terms_maximum_value_S_n_l101_101662

noncomputable def a_n (n : ℕ) := 48 - 8 * n

noncomputable def S_n (n : ℕ) := -4 * n^2 + 44 * n

theorem general_formula_sequence (a_3 : ℕ := 24) (S_11 : ℕ := 0) :
  (a_n 3 = 24) ∧ (S_n 11 = 0) → ∀ n, a_n n = 48 - 8 * n := sorry

theorem sum_of_first_n_terms (a_3 : ℕ := 24) (S_11 : ℕ := 0) :
  (a_n 3 = 24) ∧ (S_n 11 = 0) → ∀ n, S_n n = -4 * n^2 + 44 * n := sorry

theorem maximum_value_S_n (a_3 : ℕ := 24) (S_11 : ℕ := 0) :
  (a_n 3 = 24) ∧ (S_n 11 = 0) →
  (S_n 5 = 120 ∧ S_n 6 = 120) := sorry

end general_formula_sequence_sum_of_first_n_terms_maximum_value_S_n_l101_101662


namespace water_left_in_cooler_l101_101329

def initial_gallons := 3
def ounces_per_gallon := 128
def dixie_cup_ounces := 6
def rows_of_chairs := 5
def chairs_per_row := 10

theorem water_left_in_cooler :
  let initial_ounces := initial_gallons * ounces_per_gallon,
      total_chairs := rows_of_chairs * chairs_per_row,
      total_ounces_used := total_chairs * dixie_cup_ounces,
      final_ounces := initial_ounces - total_ounces_used
  in final_ounces = 84 :=
  sorry

end water_left_in_cooler_l101_101329


namespace correct_option_l101_101325

theorem correct_option (a b : ℂ) :
  ¬ (a^2 + a^2 = a^4) ∧ ¬ ((a^2)^3 = a^5) ∧ ¬ (a + 2 = 2a) ∧ ((ab)^3 = a^3 * b^3) := 
by
  split
  · intro h
    apply h
    sorry 
  split
  · intro h
    apply h
    sorry
  split
  · intro h
    apply h
    sorry
  · exact sorry

end correct_option_l101_101325


namespace light_travel_time_l101_101714

open Real

def speedOfLight := 300000 -- speed of light in kilometers per second
def distanceSunToEarth := 150000000 -- distance from the Sun to the Earth in kilometers

theorem light_travel_time :
  (distanceSunToEarth / (speedOfLight * 60)).round = 8.3 :=
by
  -- Proof omitted, only the statement is required
  sorry

end light_travel_time_l101_101714


namespace math_problem_l101_101783

theorem math_problem :
  ((34.2735 * 18.9251) / 6.8307 + 128.0021 - 56.1193) ≈ 166.8339 :=
by
  sorry

end math_problem_l101_101783


namespace at_least_two_babies_speak_l101_101120

theorem at_least_two_babies_speak (p : ℚ) (h : p = 1/5) :
  let q := 1 - p in
  let none_speak := q^7 in
  let one_speaks := (7.choose 1) * p * q^6 in
  let at_most_one_speaks := none_speak + one_speaks in
  let result := 1 - at_most_one_speaks in
  result = 50477 / 78125 :=
by
  simp [h]
  sorry

end at_least_two_babies_speak_l101_101120


namespace sqrt_factorial_l101_101756

theorem sqrt_factorial : Real.sqrt (Real.ofNat (Nat.factorial 5) * Real.ofNat (Nat.factorial 5)) = 120 := 
by 
  sorry

end sqrt_factorial_l101_101756


namespace length_AB_circ_quad_l101_101615

theorem length_AB_circ_quad (α β γ δ : ℝ) (a b c d : ℝ)
  (A B C D E : Type) [circle_quad A B C D]
  (h1 : α = 90) (h2 : β = 120) (h3 : γ = 120) (h4 : c = 1):
  ∃ x, x = 2 - real.sqrt 3 :=
begin
  sorry
end

end length_AB_circ_quad_l101_101615


namespace coeff_of_term_with_inverse_x_l101_101542

-- Given condition
def sum_of_binomial_coeff (n : ℕ) : ℕ := 2^n

-- Statement of the problem without providing a direct proof
theorem coeff_of_term_with_inverse_x 
: (sum_of_binomial_coeff 7 = 128) → 
  (binom_coeff : ∀ r : ℕ, r ≤ 7 → 
  Nat.choose 7 r * (2^((7 - r) * 2)) * (-1)^r * x^((7 - r) * 2 - r) → Int) → 
  (r = 5) → 
  (binom_coeff 5 ≤ 7 = -84) := 
by sorry

end coeff_of_term_with_inverse_x_l101_101542


namespace average_students_l101_101717

def ClassGiraffe : ℕ := 225

def ClassElephant (giraffe: ℕ) : ℕ := giraffe + 48

def ClassRabbit (giraffe: ℕ) : ℕ := giraffe - 24

theorem average_students (giraffe : ℕ) (elephant : ℕ) (rabbit : ℕ) :
  giraffe = 225 → elephant = giraffe + 48 → rabbit = giraffe - 24 →
  (giraffe + elephant + rabbit) / 3 = 233 := by
  sorry

end average_students_l101_101717


namespace workshop_workers_l101_101219

noncomputable def average_salary_all : ℝ := 850
noncomputable def num_technicians : ℕ := 7
noncomputable def average_salary_technicians : ℝ := 1000
noncomputable def average_salary_non_technicians : ℝ := 780
noncomputable def total_number_of_workers : ℕ := 22

theorem workshop_workers :
  ∃ W : ℝ, W = total_number_of_workers ∧ 
  (average_salary_all * W = (num_technicians * average_salary_technicians) + 
                           ((W - num_technicians) * average_salary_non_technicians)) :=
by
  use 22
  split
  · rfl
  · sorry

end workshop_workers_l101_101219


namespace impossible_to_make_all_points_red_l101_101718

-- Definitions taken from step a)
def Circle := { points : List (Bool) // points ≠ [] }
-- True represents a red point and False represents a blue point

-- Statement for the mathematically equivalent proof problem
theorem impossible_to_make_all_points_red (circle : Circle) : 
  ∃ blue_point ∈ circle.points, true :=
begin
  sorry

end impossible_to_make_all_points_red_l101_101718


namespace value_of_ω_value_of_φ_sum_S_30_l101_101946

-- First, we define the parameters and constraints.
variables {ω : ℝ} {φ : ℝ}

-- Given function: f(x) = 2 * sin(ω * x + φ)
def f (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- Given constraints:
axiom ω_pos : ω > 0
axiom φ_bound : |φ| < Real.pi
axiom passes_through_1 : f (Real.pi / 12) = -2
axiom passes_through_2 : f (7 * Real.pi / 12) = 2
axiom monotonic_in_interval : ∀ x y, (Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ 7 * Real.pi / 12) → f x ≤ f y

-- Question (I): Prove the values of ω and φ.
theorem value_of_ω : ω = 2 := sorry
theorem value_of_φ : φ = -2 * Real.pi / 3 := sorry

-- Definition of a_n and its sequence.
def a_n (n : ℕ) : ℝ := n * f (n * Real.pi / 3)

-- Sum of the first 30 terms of the sequence {a_n}
def S_30 : ℝ := (Finset.range 30).sum (λ n, a_n (n + 1))

-- Question (II): Prove the sum S_30.
theorem sum_S_30 : S_30 = -10 * Real.sqrt 3 := sorry

end value_of_ω_value_of_φ_sum_S_30_l101_101946


namespace mono_increasing_omega_range_l101_101551

noncomputable def omega (ω : ℝ) : Prop :=
  (0 < ω) ∧ (ω ≤ 3 / 4)

theorem mono_increasing_omega_range
  (ω : ℝ) (hω : omega ω) :
  ∀ x ∈ set.Icc (-π / 4) (2 * π / 3), 
  0 ≤ 2 * ω * real.cos (ω * x) :=
by
  sorry

end mono_increasing_omega_range_l101_101551


namespace sqrt_factorial_l101_101755

theorem sqrt_factorial : Real.sqrt (Real.ofNat (Nat.factorial 5) * Real.ofNat (Nat.factorial 5)) = 120 := 
by 
  sorry

end sqrt_factorial_l101_101755


namespace three_digit_subtraction_l101_101272

theorem three_digit_subtraction (c d : ℕ) (H1 : 0 ≤ c ∧ c ≤ 9) (H2 : 0 ≤ d ∧ d ≤ 9) :
  (745 - (300 + c * 10 + 4) = (400 + d * 10 + 1)) ∧ ((4 + 1) - d % 11 = 0) → 
  c + d = 14 := 
sorry

end three_digit_subtraction_l101_101272


namespace total_handshakes_l101_101845

theorem total_handshakes (sets_of_twins : ℕ) (sets_of_triplets : ℕ)
  (total_twins : ℕ) (total_triplets : ℕ)
  (handshakes_among_twins : ℕ) (handshakes_among_triplets : ℕ)
  (cross_handshakes_twins : ℕ) (cross_handshakes_triplets : ℕ) 
  (total_handshakes : ℕ):
  sets_of_twins = 10 → sets_of_triplets = 7 →
  total_twins = sets_of_twins * 2 →
  total_triplets = sets_of_triplets * 3 →
  handshakes_among_twins = total_twins * (total_twins - 2) / 2 →
  handshakes_among_triplets = total_triplets * (total_triplets - 3) / 2 →
  cross_handshakes_twins = total_twins * (total_triplets / 3) →
  cross_handshakes_triplets = total_triplets * (total_twins / 4) →
  total_handshakes = handshakes_among_twins + handshakes_among_triplets + cross_handshakes_twins + cross_handshakes_triplets →
  total_handshakes = 614 :=
by {
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9,
  subst h1,
  subst h2,
  subst h3,
  subst h4,
  subst h5,
  subst h6,
  subst h7,
  subst h8,
  subst h9,
  -- Additional proof details would go here
  sorry
}

end total_handshakes_l101_101845


namespace relationship_y₁_y₂_y₃_l101_101924

variables (y₁ y₂ y₃ : ℝ)

def inverse_proportion (x : ℝ) : ℝ := 3 / x

-- Given points A(-2, y₁), B(-1, y₂), C(1, y₃)
-- and y₁ = inverse_proportion(-2), y₂ = inverse_proportion(-1), y₃ = inverse_proportion(1)
theorem relationship_y₁_y₂_y₃ : 
  let y₁ := inverse_proportion (-2),
      y₂ := inverse_proportion (-1),
      y₃ := inverse_proportion (1) in
  y₂ < y₁ ∧ y₁ < y₃ :=
by
  sorry

end relationship_y₁_y₂_y₃_l101_101924


namespace distinct_patterns_count_l101_101788

def initial_table := matrix (fin 3) (fin 3) bool

def toggle (m : matrix (fin 3) (fin 3) bool) (i j : fin 2) : matrix (fin 3) (fin 3) bool :=
  λ x y => if (x.val = i.val ∨ x.val = i.val + 1) ∧ (y.val = j.val ∨ y.val = j.val + 1) then not (m x y) else (m x y)

def toggle_sequence (seq : list (fin 2 × fin 2)) (m : matrix (fin 3) (fin 3) bool) : matrix (fin 3) (fin 3) bool :=
  seq.foldl (λ m p => toggle m p.1 p.2) m

def patterns (seqs : list (list (fin 2 × fin 2))) : set (matrix (fin 3) (fin 3) bool) :=
  seqs.map (toggle_sequence initial_table) |>.to_finset.to_set

def count_patterns : ℕ :=
  patterns (list.fin_enum (list {p : fin 2 × fin 2 // true})).card

theorem distinct_patterns_count : count_patterns = 16 := sorry

end distinct_patterns_count_l101_101788


namespace unsatisfactory_grade_fraction_l101_101361

theorem unsatisfactory_grade_fraction 
  (num_A num_B num_C num_D num_F : ℕ) 
  (htotal : num_A = 6 ∧ num_B = 5 ∧ num_C = 4 ∧ num_D = 2 ∧ num_F = 8)
  (hsatisfactory : (grades : ℕ) → grades = num_A + num_B + num_C + num_D + num_F = 25) :
  num_F / (num_A + num_B + num_C + num_D + num_F) = 8 / 25 := 
by
  sorry

end unsatisfactory_grade_fraction_l101_101361


namespace closed_set_B_l101_101002

-- Given definition of a closed set
def isClosed (A : Set ℤ) : Prop :=
∀ a b ∈ A, (a + b) ∈ A ∧ (a - b) ∈ A

-- Specific set B
def B : Set ℤ := {n | ∃ k : ℤ, n = 3 * k}

-- The proof problem statement
theorem closed_set_B : isClosed B :=
sorry

end closed_set_B_l101_101002


namespace fifth_term_is_zero_l101_101696

variable (x y : ℝ)

-- First four terms of the arithmetic sequence
def first_term : ℝ := x + 3 * y
def second_term : ℝ := x - 3 * y
def third_term : ℝ := 2 * x * y
def fourth_term : ℝ := 2 * x / y

-- Common difference
def common_difference : ℝ := second_term x y - first_term x y

-- Expression for the fifth term
def fifth_term : ℝ := fourth_term x y + common_difference x y

-- Ensure x and y values satisfy the conditions
def valid_xy : Prop := (x = 9 * y / (1 - 2 * y)) ∧ (y = -1)

-- The fifth term is equal to 0 given the conditions on x and y
theorem fifth_term_is_zero (h : valid_xy x y) : fifth_term x y = 0 :=
by {
  sorry
}

end fifth_term_is_zero_l101_101696


namespace FourColorTheorem_l101_101692

-- Define the condition of the normal map
def NormalMap (V : Type) (E : Type) : Prop :=
  ∃ (regions : Finset V) (borders : Finset E),
  (∀ e ∈ borders, ∃ n : ℕ, 1 ≤ n ∧ n ≤ 3)
  ∧ (∀ r₁ r₂ ∈ regions, r₁ ≠ r₂ → ∃ e ∈ borders, connects e r₁ r₂)

-- Define the function stating the coloring condition
def ProperlyColored (V : Type) (E : Type) (colors : V → ℕ) : Prop :=
  ∀ (r₁ r₂ : V), (∃ e : E, connects e r₁ r₂) → colors r₁ ≠ colors r₂

-- Type definitions for regions and borders
constant V : Type -- Type for regions
constant E : Type -- Type for borders
constant connects : E → V → V → Prop -- Binary relation for connectivity of borders

-- Statement of the theorem
theorem FourColorTheorem : ∀ (G : Type),
  (NormalMap V E) →
  ∃ (colors : V → ℕ), (∀ r : V, colors r ≤ 4) ∧ ProperlyColored V E colors := 
sorry

end FourColorTheorem_l101_101692


namespace evaluation_of_expression_l101_101012

theorem evaluation_of_expression : (((-2 : ℝ)⁻² - (-3 : ℝ)⁻¹)⁻¹) = 12 / 7 := 
by
  sorry

end evaluation_of_expression_l101_101012


namespace domain_of_sqrt_function_l101_101020

noncomputable def f (x : ℝ) := real.sqrt (2 * x - 1)

theorem domain_of_sqrt_function :
  ∀ x : ℝ, (∃ y : ℝ, f y = x) ↔ (x ≥ 1/2) :=
sorry

end domain_of_sqrt_function_l101_101020


namespace intersection_of_sets_l101_101182

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 ≥ 0
def B (x : ℝ) : Prop := -2 ≤ x ∧ x < 2

theorem intersection_of_sets :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
by
  sorry

end intersection_of_sets_l101_101182


namespace exists_function_with_properties_l101_101673

open Classical

noncomputable def f (n : ℕ) : ℕ := (padic_val_rat 2 n.to_rat).natAbs % 1000 + 1

theorem exists_function_with_properties :
  (∃ f : ℕ → ℕ, (∀ x : ℕ, f x = (padic_val_rat 2 x.to_rat).natAbs % 1000 + 1) ∧
    (∃ N : ℕ, ∀ n m : ℕ, n ≠ m → (f n ≠ f m ⟶ n ≠ m)) ∧
    (∀ (x1 x2 ... x1000 : ℤ), x1 ≠ 0 ∧ x2 ≠ 0 ∧ ... ∧ x1000 ≠ 0 ∧
      f (|x1|.to_nat) = f (|x2|.to_nat) ∧ ... ∧ f (|x1000|.to_nat) →
        x1 + 2 * x2 + 2^2 * x3 + ... + 2^999 * x1000 ≠ 0)) :=
by {
  let f := λ n : ℕ, (padic_val_rat 2 n.to_rat).natAbs % 1000 + 1,
  use f,
  split,
  { intro n,
    refl },
  split,
  { obtain ⟨N, hN⟩ : ∃ N : ℕ, ∀ n m : ℕ, n ≠ m → (f n ≠ f m ⟶ n ≠ m),
    sorry,
    exact ⟨N, hN⟩ },
  { intros x1 x2 ... x1000 h1 h2,
    sorry,
  }
}

end exists_function_with_properties_l101_101673


namespace exists_even_n_square_maker_partition_l101_101575

/-- Conditions for being a square maker -/
def is_square_maker (a b : ℕ) : Prop :=
  ∃ k : ℕ, ab + 1 = k^2

/-- Main theorem for proving the conditions -/
theorem exists_even_n_square_maker_partition (n : ℕ) :
  (∃ (pairs : Finset (ℕ × ℕ)), 
    pairs.card = n ∧ 
    (∀ p ∈ pairs, (p:ℕ × ℕ).fst < (p:ℕ × ℕ).snd ∧ is_square_maker (p:ℕ × ℕ).fst (p:ℕ × ℕ).snd) ∧ 
    ∀ x ∈ (pairs.bUnion (λ p, {p.1, p.2})), 1 ≤ x ∧ x ≤ 2 * n ∧ x ∈ Finset.Icc 1 (2 * n)) ↔ 
    ∃ k : ℕ, n = 2 * k :=
sorry

end exists_even_n_square_maker_partition_l101_101575


namespace maximum_at_vertex_l101_101037

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem maximum_at_vertex (a b c x_0 : ℝ) (h_a : a < 0) (h_x0 : 2 * a * x_0 + b = 0) :
  ∀ x : ℝ, quadratic_function a b c x ≤ quadratic_function a b c x_0 :=
sorry

end maximum_at_vertex_l101_101037


namespace sequence_converges_l101_101159

noncomputable def sequence : ℕ → ℝ
| 0     := 2021
| (n+1) := (sequence n)^2 + 2021 / (2 * (sequence n + 1))

theorem sequence_converges :
  ∃ l : ℝ, (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequence n - l| < ε) ∧ l = sqrt 2022 - 1 := 
  sorry

end sequence_converges_l101_101159


namespace find_a_for_tangent_line_l101_101940

/--
Given the curve \( y = ax - \ln(x+1) \) has a tangent line at the point (0,0) with the equation \( y = 2x \),
prove that \( a = 3 \).
-/
theorem find_a_for_tangent_line (a : ℝ) :
  (∀ x : ℝ, y = a * x - real.log (x + 1)) ∧ (∀ x : ℝ, y = 2 * x) →
  a = 3 :=
sorry

end find_a_for_tangent_line_l101_101940


namespace min_value_of_y_l101_101003

noncomputable def y (x : ℝ) : ℝ := x^2 + 26 * x + 7

theorem min_value_of_y : ∃ x : ℝ, y x = -162 :=
by
  use -13
  sorry

end min_value_of_y_l101_101003


namespace geometric_sequence_sum_l101_101267

noncomputable theory

-- Define the conditions and the problem statement
theorem geometric_sequence_sum (a r : ℝ) (h₁ : a * (1 - r^3000) / (1 - r) = 500) 
  (h₂ : a * (1 - r^6000) / (1 - r) = 950) : 
  a * (1 - r^9000) / (1 - r) = 1355 := 
sorry

end geometric_sequence_sum_l101_101267


namespace minimum_value_of_function_l101_101074

open Real

noncomputable def input_function (x : ℝ) : ℝ :=
  x + 2 / (x - 1)

theorem minimum_value_of_function : 
  ∃ x ∈ Ioo (1:ℝ) (⊤:ℝ), input_function x = 2 * sqrt 2 + 1 :=
sorry

end minimum_value_of_function_l101_101074


namespace negative_870_in_third_quadrant_l101_101271

noncomputable def angle_in_third_quadrant (theta : ℝ) : Prop :=
  180 < theta ∧ theta < 270

theorem negative_870_in_third_quadrant:
  angle_in_third_quadrant 210 :=
by
  sorry

end negative_870_in_third_quadrant_l101_101271


namespace sin_cos_half_angle_sum_l101_101903

theorem sin_cos_half_angle_sum 
  (θ : ℝ)
  (hcos : Real.cos θ = -7/25) 
  (hθ : θ ∈ Set.Ioo (-Real.pi) 0) : 
  Real.sin (θ/2) + Real.cos (θ/2) = -1/5 := 
sorry

end sin_cos_half_angle_sum_l101_101903


namespace star_m_eq_22_l101_101170

def star (x : ℕ) : ℕ := x.digits.sum

def S : set ℕ := {n | star n = 15 ∧ 100 ≤ n ∧ n < 10^5}

noncomputable def m : ℕ := S.card

theorem star_m_eq_22 : star m = 22 := by
  sorry

end star_m_eq_22_l101_101170


namespace multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l101_101095

theorem multiples_of_7_with_units_digit_7 (n : ℕ) : 
  (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) ↔ 
  n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  sorry

theorem number_of_multiples_of_7_with_units_digit_7 : 
  ∃ m, m = 3 ∧ ∀ n : ℕ, (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) → n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  use 3
  intros n
  intros hn
  split
  intro h
  cases h
  sorry

end multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l101_101095


namespace construct_line_l101_101143

variable (Point Line Plane : Type)
variable (a b c : Line)
variable distance : Point → Point → ℝ
variable parallel : Line → Line → Prop
variable intersects : Line → Line → Prop
variable contains : Plane → Line → Prop
variable intersection : Plane → Plane → Line

-- Conditions: Definitions and axioms
axiom skew_lines (l1 l2 : Line) : ¬(intersects l1 l2)
axiom exists_parallel_line (l p : Line) (pt : Point) : parallel l p → ∃ l', parallel l l' ∧ ∃ pt', contains l pt'

-- Problem statement: Existence of the line m
theorem construct_line (h_skew_a_b : skew_lines a b) (h_exists_parallel_c : ∃ pt, ∃ l, parallel l c ∧ ∃ pt', contains l pt'):
  ∃ m, parallel m c ∧ intersects m a ∧ intersects m b :=
begin
  sorry
end

end construct_line_l101_101143


namespace vector_magnitude_cosine_angle_l101_101043

open Real

variables (a b : ℝ → ℝ → ℝ) -- Assuming vectors a and b in 2D for simplicity.

-- Given: |a| = 4, |b| = 2, and the angle between a and b is 120 degrees.
axiom magnitude_a : |a| = 4
axiom magnitude_b : |b| = 2
axiom angle_ab : angle a b = 120

-- Proving: |3a - b| = 2*sqrt(43)
theorem vector_magnitude : |3 * a - b| = 2 * sqrt 43 :=
by sorry

-- Proving: The cosine value of the angle between a and (a - b) is 5*sqrt(7)/14
theorem cosine_angle : cos (angle a (a - b)) = (5 * sqrt 7) / 14 :=
by sorry

end vector_magnitude_cosine_angle_l101_101043


namespace problem1_problem2_l101_101848

-- Problem 1
theorem problem1 : 
  (-2.8) - (-3.6) + (-1.5) - (3.6) = -4.3 := 
by 
  sorry

-- Problem 2
theorem problem2 :
  (- (5 / 6 : ℚ) + (1 / 3 : ℚ) - (3 / 4 : ℚ)) * (-24) = 30 := 
by 
  sorry

end problem1_problem2_l101_101848


namespace find_lambda_l101_101293

noncomputable def line (λ : ℝ) : ℝ → ℝ → ℝ := λ x y, 3 * x - 4 * y + λ
noncomputable def translated_line (λ : ℝ) : ℝ → ℝ → ℝ := λ x y, 3 * (x + 1) - 4 * y + λ
noncomputable def circle : ℝ → ℝ → ℝ := λ x y, x^2 + y^2 - 2 * x - 4 * y + 4

theorem find_lambda : ∃ λ : ℝ, (λ = -3 ∨ λ = 7) ∧ ∀ x y, translated_line λ x y = 0 ↔ circle x y = 0 :=
sorry

end find_lambda_l101_101293


namespace curve_intersects_itself_l101_101838

theorem curve_intersects_itself :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ (t₁^2 - 3, t₁^3 - 6 * t₁ + 4) = (3, 4) ∧ (t₂^2 - 3, t₂^3 - 6 * t₂ + 4) = (3, 4) :=
sorry

end curve_intersects_itself_l101_101838


namespace polygon_num_sides_l101_101587

theorem polygon_num_sides (s : ℕ) (h : 180 * (s - 2) > 2790) : s = 18 :=
sorry

end polygon_num_sides_l101_101587


namespace ratio_final_to_original_l101_101979

-- Given conditions
variable (d : ℝ)
variable (h1 : 364 = d * 1.30)

-- Problem statement
theorem ratio_final_to_original : (364 / d) = 1.3 := 
by sorry

end ratio_final_to_original_l101_101979


namespace least_positive_integer_property_l101_101893

theorem least_positive_integer_property : 
  ∃ (n d : ℕ) (p : ℕ) (h₁ : 1 ≤ d) (h₂ : d ≤ 9) (h₃ : p ≥ 2), 
  (10^p * d = 24 * n) ∧ (∃ k : ℕ, (n = 100 * 10^(p-2) / 3) ∧ (900 = 8 * 10^p + 100 / 3 * 10^(p-2))) := sorry

end least_positive_integer_property_l101_101893


namespace locus_of_right_angle_vertices_l101_101495

-- Definition of points and segment
variable (A B C : Point)
variable (BC : LineSegment B C)

-- Definition of the closed balls with diameters AB and AC
def B1 := closedBall (midpoint A B) (dist A B / 2)
def B2 := closedBall (midpoint A C) (dist A C / 2)

-- Locus of points forming the vertices of described right angles
def locus (O : Point) : Prop := O ∈ symmetricDifference B1 B2

-- The theorem statement
theorem locus_of_right_angle_vertices (A B C : Point) (BC : LineSegment B C) :
    ∀ O : Point, (∃ K : Point, K ∈ BC ∧ (angle A O K = π / 2)) → O ∈ symmetricDifference (closedBall (midpoint A B) (dist A B / 2)) (closedBall (midpoint A C) (dist A C / 2)) :=
by
  intro O h
  sorry

end locus_of_right_angle_vertices_l101_101495


namespace volume_of_pyramid_l101_101674

variables (P A B C D : Type)
variables (h1 : dist A B = 10)
variables (h2 : dist B C = 5)
variables (h3 : ∀ x, perpendicular (vector PA) (vector AD))
variables (h4 : ∀ x, perpendicular (vector PA) (vector AB))
variables (h5 : dist P B = 20)

theorem volume_of_pyramid : 
  volume (pyramid P A B C D) = 500 * real.sqrt 3 / 3 :=
sorry

end volume_of_pyramid_l101_101674


namespace multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l101_101097

theorem multiples_of_7_with_units_digit_7 (n : ℕ) : 
  (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) ↔ 
  n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  sorry

theorem number_of_multiples_of_7_with_units_digit_7 : 
  ∃ m, m = 3 ∧ ∀ n : ℕ, (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) → n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  use 3
  intros n
  intros hn
  split
  intro h
  cases h
  sorry

end multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l101_101097


namespace parallelogram_sum_p_plus_a_l101_101879

variable (x1 y1 x2 y2 x3 y3 x4 y4 : ℤ)
variable (p a : ℕ)

-- Given the conditions
def vertices := (x1, y1) = (0, 0) ∧ (x2, y2) = (7, 0) ∧ (x3, y3) = (3, 4) ∧ (x4, y4) = (10, 4)
def perimeter := p = 24
def area := a = 28

-- The proof statement
theorem parallelogram_sum_p_plus_a (h1 : vertices)
    (h2 : perimeter)
    (h3 : area) : p + a = 52 :=
by
  -- Proof omitted
  sorry

end parallelogram_sum_p_plus_a_l101_101879


namespace problem_statement_l101_101110

noncomputable def P1 (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
noncomputable def P2 (β : ℝ) : ℝ × ℝ := (Real.cos β, -Real.sin β)
noncomputable def P3 (α β : ℝ) : ℝ × ℝ := (Real.cos (α + β), Real.sin (α + β))
noncomputable def A : ℝ × ℝ := (1, 0)

theorem problem_statement (α β : ℝ) :
  (Prod.fst (P1 α))^2 + (Prod.snd (P1 α))^2 = 1 ∧
  (Prod.fst (P2 β))^2 + (Prod.snd (P2 β))^2 = 1 ∧
  (Prod.fst (P1 α) * Prod.fst (P2 β) + Prod.snd (P1 α) * Prod.snd (P2 β)) = Real.cos (α + β) :=
by
  sorry

end problem_statement_l101_101110


namespace angle_B_measure_l101_101629

theorem angle_B_measure (a b : ℝ) (A B : ℝ) (h₁ : a = 4) (h₂ : b = 4 * Real.sqrt 3) (h₃ : A = Real.pi / 6) : 
  B = Real.pi / 3 ∨ B = 2 * Real.pi / 3 :=
by
  sorry

end angle_B_measure_l101_101629


namespace min_cube_count_is_5_l101_101796

def front_view : List (Nat × Nat) := [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 3), (3, 2), (3, 1), (3, 0)]
def side_view : List (Nat × Nat) := [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 1), (2, 2), (3, 3), (3, 2)]

def valid_cube_placement (x y z: Nat) : Prop :=
  ((x, y) ∈ front_view ∧ (y, z) ∈ side_view)
  ∧ (cube x y z).shares_face_with_another_cube

theorem min_cube_count_is_5 :
  ∃ placement : List (Nat × Nat × Nat),
    (∀ c ∈ placement, valid_cube_placement c.1 c.2 c.3) ∧ placement.length = 5 :=
sorry

end min_cube_count_is_5_l101_101796


namespace zoo_total_animals_l101_101395

def num_tiger_enclosures : ℕ := 4
def num_zebra_enclosures_per_tiger : ℕ := 2
def num_giraffe_enclosures_per_zebra : ℕ := 3
def num_tigers_per_enclosure : ℕ := 4
def num_zebras_per_enclosure : ℕ := 10
def num_giraffes_per_enclosure : ℕ := 2

theorem zoo_total_animals :
  let num_zebra_enclosures := num_tiger_enclosures * num_zebra_enclosures_per_tiger,
      num_giraffe_enclosures := num_zebra_enclosures * num_giraffe_enclosures_per_zebra,
      total_tigers := num_tiger_enclosures * num_tigers_per_enclosure,
      total_zebras := num_zebra_enclosures * num_zebras_per_enclosure,
      total_giraffes := num_giraffe_enclosures * num_giraffes_per_enclosure,
      total_animals := total_tigers + total_zebras + total_giraffes
  in total_animals = 144 :=
by
  intros
  let num_zebra_enclosures := num_tiger_enclosures * num_zebra_enclosures_per_tiger
  let num_giraffe_enclosures := num_zebra_enclosures * num_giraffe_enclosures_per_zebra
  let total_tigers := num_tiger_enclosures * num_tigers_per_enclosure
  let total_zebras := num_zebra_enclosures * num_zebras_per_enclosure
  let total_giraffes := num_giraffe_enclosures * num_giraffes_per_enclosure
  let total_animals := total_tigers + total_zebras + total_giraffes
  exact eq.refl 144

end zoo_total_animals_l101_101395


namespace div_by_72_l101_101119

theorem div_by_72 (x : ℕ) (y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : x = 4)
    (h3 : 0 ≤ y ∧ y ≤ 9) (h4 : y = 6) : 
    72 ∣ (9834800 + 1000 * x + 10 * y) :=
by 
  sorry

end div_by_72_l101_101119


namespace not_factorial_tails_l101_101461

noncomputable def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) + ...

theorem not_factorial_tails :
  let n := 2500 in ∃ (k : ℕ), k = 500 ∧ ∀ m < n, ¬(f m = k) :=
by {
  sorry
}

end not_factorial_tails_l101_101461


namespace find_original_number_l101_101802

-- Variables and assumptions
variables (N y x : ℕ)

-- Conditions
def crossing_out_condition : Prop := N = 10 * y + x
def sum_condition : Prop := N + y = 54321
def remainder_condition : Prop := x = 54321 % 11
def y_value : Prop := y = 4938

-- The final proof problem statement
theorem find_original_number (h1 : crossing_out_condition N y x) (h2 : sum_condition N y) 
  (h3 : remainder_condition x) (h4 : y_value y) : N = 49383 :=
sorry

end find_original_number_l101_101802


namespace pants_cut_is_0_50_l101_101385

-- Define the conditions
def skirt_cut : ℝ := 0.75
def x : ℝ -- Amount cut from the pants
def condition : Prop := skirt_cut = x + 0.25

-- We need to show that the pants cut is 0.5 inches
theorem pants_cut_is_0_50 (h : condition) : x = 0.50 := by
  sorry

end pants_cut_is_0_50_l101_101385


namespace isosceles_triangle_large_angles_l101_101606

theorem isosceles_triangle_large_angles (y : ℝ) (h : 2 * y + 40 = 180) : y = 70 :=
by
  sorry

end isosceles_triangle_large_angles_l101_101606


namespace inequality_solution_l101_101888

theorem inequality_solution (x : ℝ) : 
  (3 / 20 + abs (2 * x - 5 / 40) < 9 / 40) → (1 / 40 < x ∧ x < 1 / 10) :=
by
  sorry

end inequality_solution_l101_101888


namespace rihlelo_has_4_lines_of_symmetry_l101_101225

-- Definition of the rihlèlò design based on symmetry
def rihlelo_design : Type := { d : Set (Point × Point) // ∃ (s : Finset Line), s.card = 4 ∧ ∀ l ∈ s, is_symmetry l }

-- Proof statement
theorem rihlelo_has_4_lines_of_symmetry (d : rihlelo_design) : ∃ s : Finset Line, s.card = 4 ∧ ∀ l ∈ s, is_symmetry l :=
begin
  sorry
end

end rihlelo_has_4_lines_of_symmetry_l101_101225


namespace multiples_of_7_units_digit_7_l101_101090

theorem multiples_of_7_units_digit_7 (n : ℕ) (h1 : n < 150) (h2 : ∃ (k : ℕ), n = 7 * k) (h3 : n % 10 = 7) : 
    ∃ (m : ℕ), m = 2 := 
by
  sorry

end multiples_of_7_units_digit_7_l101_101090


namespace equal_distribution_possible_l101_101841

theorem equal_distribution_possible 
  (boxes : ℕ) (balls : ℕ) 
  (initial_distribution : Fin boxes → ℕ)
  (add_six_balls : (Fin boxes → ℕ) → (Fin boxes → ℕ)) :
  boxes = 95 ∧ balls = 19 ∧ 
  (∀ (f : Fin boxes → ℕ), initial_distribution.sum (λ i, f i) = balls) ∧
  (∀ (f : Fin boxes → ℕ), 
     ∃ n : ℕ, 
     (iterate n add_six_balls initial_distribution).sum (λ i, f i) % boxes = 0) 
→ 
  ∃ k : ℕ, ∀ i : Fin boxes, (iterate n add_six_balls initial_distribution) i = k := 
sorry

end equal_distribution_possible_l101_101841


namespace black_balls_number_l101_101046

-- Define the given conditions and the problem statement as Lean statements
theorem black_balls_number (n : ℕ) (h : (2 : ℝ) / (n + 2 : ℝ) = 0.4) : n = 3 :=
by
  sorry

end black_balls_number_l101_101046


namespace shaded_region_area_l101_101148

-- Given conditions
def radius := 5
def diameter_AB := 2 * radius
def diameter_CD := 2 * radius
def are_perpendicular (AB CD : ℝ) := true

-- Goal
theorem shaded_region_area (r : ℝ) (h1 : r = radius) (h2 : diameter_AB = 2 * r) (h3 : diameter_CD = 2 * r) (h4 : are_perpendicular diameter_AB diameter_CD = true) :
  let area := 25 + 12.5 * Real.pi in area = 25 + 12.5 * Real.pi := 
by
  sorry

end shaded_region_area_l101_101148


namespace emily_total_beads_l101_101881

-- Let's define the given conditions
def necklaces : ℕ := 11
def beads_per_necklace : ℕ := 28

-- The statement to prove
theorem emily_total_beads : (necklaces * beads_per_necklace) = 308 := by
  sorry

end emily_total_beads_l101_101881


namespace possible_integer_roots_l101_101376

theorem possible_integer_roots (a3 a2 a1 : ℤ) :
  (∃ x : ℤ, x^4 + a3 * x^3 + a2 * x^2 + a1 * x - 27 = 0) →
  (x ∈ {-27, -9, -3, -1, 1, 3, 9, 27} ∨ ¬ ∃ x : ℤ, x^4 + a3 * x^3 + a2 * x^2 + a1 * x - 27 = 0) :=
by
  sorry

end possible_integer_roots_l101_101376


namespace range_of_a_l101_101552

noncomputable def f (x : ℝ) : ℝ :=
  (Finset.sum (Finset.range 2013) (λ n, |x - ↑n|) + Finset.sum (Finset.range 2012) (λ n, |x + (n + 1)|))

theorem range_of_a (a : ℝ) (ha : f (a^2 + 2*a + 2) > f a) : a < -2 ∨ a > -1 :=
by
  sorry

end range_of_a_l101_101552


namespace trig_identity_1_evaluate_function_f_l101_101352

-- Proof Problem 1: Prove that the expression \(\sin (-120^{\circ})\cos 210^{\circ}+\cos (-300^{\circ})\sin (-330^{\circ}) = 1\)
theorem trig_identity_1 : 
  sin (-120 * real.pi / 180) * cos (210 * real.pi / 180) + cos (-300 * real.pi / 180) * sin (-330 * real.pi / 180) = 1 := sorry

-- Proof Problem 2: Prove that \( f\left( -\frac{23\pi}{6} \right) = \sqrt{3} \)
noncomputable def f (α : ℝ) : ℝ :=
  (2 * sin (real.pi + α) * cos (real.pi - α) - cos (real.pi + α)) /
  (1 + sin (α) * sin (α) + cos ((3 * real.pi / 2) + α) - sin ((real.pi / 2) + α) * sin ((real.pi / 2) + α))

theorem evaluate_function_f : f (-23 * real.pi / 6) = real.sqrt 3 := sorry

end trig_identity_1_evaluate_function_f_l101_101352


namespace cube_surface_area_correct_l101_101793

def edge_length : ℝ := 11

def cube_surface_area (e : ℝ) : ℝ := 6 * e^2

theorem cube_surface_area_correct : cube_surface_area edge_length = 726 := by
  sorry

end cube_surface_area_correct_l101_101793


namespace geo_series_sum_eight_terms_l101_101483

theorem geo_series_sum_eight_terms :
  let a_0 := 1 / 3
  let r := 1 / 3 
  let S_8 := a_0 * (1 - r^8) / (1 - r)
  S_8 = 3280 / 6561 :=
by
  /- :: Proof Steps Omitted. -/
  sorry

end geo_series_sum_eight_terms_l101_101483


namespace initial_number_of_players_l101_101280

theorem initial_number_of_players (players_quit remaining_lives_per_player total_remaining_lives : ℕ) 
  (h1 : players_quit = 7)
  (h2 : remaining_lives_per_player = 8)
  (h3 : total_remaining_lives = 24) :
  let remaining_players := total_remaining_lives / remaining_lives_per_player in
  let initial_players := remaining_players + players_quit in
  initial_players = 10 :=
by
  sorry

end initial_number_of_players_l101_101280


namespace reflection_through_x_axis_l101_101952

theorem reflection_through_x_axis (f : ℝ → ℝ) (H : f 4 = 2) : 
    ∃ x y, (x, y) = (4, -2) := 
by
  use (4, -2)
  unfold_projs
  rw H
  sorry

end reflection_through_x_axis_l101_101952


namespace center_and_radius_of_circle_M_equation_of_symmetric_circle_equation_of_reflected_light_ray_l101_101368

/-
Proof Problem Statement in Lean 4
-/

def circle_M_eq : (x y : ℝ) → Prop :=
  λ x y, x^2 + y^2 - 4*x - 4*y + 7 = 0

def symmetric_circle_eq : (x y : ℝ) → Prop :=
  λ x y, (x - 2)^2 + (y + 2)^2 = 1

def light_ray_eq : (x y : ℝ) → Prop :=
  λ x y, 4*x - 3*y + 9 = 0

-- Prove the center and radius
theorem center_and_radius_of_circle_M :
  ∃ c : ℝ × ℝ, ∃ r : ℝ, (∀ x y : ℝ, circle_M_eq x y ↔ (x - c.1)^2 + (y - c.2)^2 = r^2) ∧ c = (2, 2) ∧ r = 1 :=
sorry

-- Prove the equation of the symmetric circle
theorem equation_of_symmetric_circle :
  ∀ x y : ℝ, symmetric_circle_eq x y ↔ (x - 2)^2 + (y + 2)^2 = 1 :=
sorry

-- Prove the equation of the reflected light ray
theorem equation_of_reflected_light_ray :
  ∀ x y : ℝ, light_ray_eq x y ↔ 4*x - 3*y + 9 = 0 :=
sorry

end center_and_radius_of_circle_M_equation_of_symmetric_circle_equation_of_reflected_light_ray_l101_101368


namespace solve_for_a_l101_101572

theorem solve_for_a (x a : ℤ) (h : x = 3) (heq : 2 * x - 10 = 4 * a) : a = -1 := by
  sorry

end solve_for_a_l101_101572


namespace sum_of_geometric_sequence_first_9000_terms_l101_101254

noncomputable def geomSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_of_geometric_sequence_first_9000_terms (a r : ℝ) (h1 : geomSum a r 3000 = 500) (h2 : geomSum a r 6000 = 950) :
  geomSum a r 9000 = 1355 :=
by
  sorry

end sum_of_geometric_sequence_first_9000_terms_l101_101254


namespace find_xy_l101_101840

-- Defining the initial conditions
variable (x y : ℕ)

-- Defining the rectangular prism dimensions and the volume equation
def prism_volume_original : ℕ := 15 * 5 * 4 -- Volume = 300
def remaining_volume : ℕ := 120

-- The main theorem statement to prove the conditions and their solution
theorem find_xy (h1 : prism_volume_original - 5 * y * x = remaining_volume)
    (h2 : x < 4) 
    (h3 : y < 15) : 
    x = 3 ∧ y = 12 := sorry

end find_xy_l101_101840


namespace range_of_a_for_distinct_real_roots_l101_101581

theorem range_of_a_for_distinct_real_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 2 * x + 1 = 0 ∧ (a - 1) * y^2 - 2 * y + 1 = 0) ↔ (a < 2 ∧ a ≠ 1) :=
by
  sorry

end range_of_a_for_distinct_real_roots_l101_101581


namespace solve_real_x_l101_101886

theorem solve_real_x (x : ℝ) : 
  (16^x + 81^x) / (24^x + 36^x) = 8 / 7 ↔ 
  x = Real.log(15 / 7) / (2 * Real.log(3 / 2)) :=
by sorry

end solve_real_x_l101_101886


namespace magnitude_of_complex_number_l101_101506

theorem magnitude_of_complex_number (z : ℂ) (h : conj z * (3 + 4 * complex.I) = 4 + 3 * complex.I) : |z| = 1 :=
sorry

end magnitude_of_complex_number_l101_101506


namespace gain_percent_l101_101371

theorem gain_percent (CP SP : ℕ) (h1 : CP = 20) (h2 : SP = 25) : 
  (SP - CP) * 100 / CP = 25 := by
  sorry

end gain_percent_l101_101371


namespace unique_common_element_l101_101054

variable (A B : Set ℝ)
variable (a : ℝ)

theorem unique_common_element :
  A = {1, 3, a} → 
  B = {4, 5} →
  A ∩ B = {4} →
  a = 4 := 
by
  intro hA hB hAB
  sorry

end unique_common_element_l101_101054


namespace count_multiples_of_7_with_units_digit_7_less_than_150_l101_101088

-- Definition of the problem's conditions and expected result
theorem count_multiples_of_7_with_units_digit_7_less_than_150 : 
  let multiples_of_7 := { n : Nat // n < 150 ∧ n % 7 = 0 }
  let units_digit_7_multiples := { n : Nat // n < 150 ∧ n % 7 = 0 ∧ n % 10 = 7 }
  List.length (units_digit_7_multiples.members) = 3 :=
by
  sorry

end count_multiples_of_7_with_units_digit_7_less_than_150_l101_101088


namespace find_a_l101_101906

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then exp x + a * x
  else if x < 0 then exp (-x) - a * x
  else 0

theorem find_a :
  (∃ c1 c2 : ℝ, c1 > 0 ∧ c2 > 0 ∧ (c1 ≠ 0 ∨ c2 ≠ 0) ∧
                f (-e) c1 = 0 ∧ f (-e) (-c2) = 0) →
                ∀ x : ℝ, f (-e) x = 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 :=
begin
sorry
end

end find_a_l101_101906


namespace product_of_possible_values_of_b_l101_101863

noncomputable def f (b x : ℝ) := b / (3 * x - 4)

theorem product_of_possible_values_of_b (b : ℝ) :
  (f b 3 = (λ y, f y b) (2 * b - 1)) → 
  let eq : ℝ := (20 / 6) in eq = 10 / 3 :=
sorry

end product_of_possible_values_of_b_l101_101863


namespace collinear_triples_count_l101_101872

open Set

def lattice_4x4x4 : Set (ℕ × ℕ × ℕ) :=
  {p | p.1 ∈ {0, 1, 2, 3} ∧ p.2 ∈ {0, 1, 2, 3} ∧ p.3 ∈ {0, 1, 2, 3}}

def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ l : ℝ, ∃ m : ℝ, ∃ n : ℝ, ∀ (x y z : ℝ), (x - y) * (p1.1 - p2.1) = l * (p1.2 - p2.2)
  ∧ (y - z) * (p1.2 - p3.2) = m * (p1.3 - p2.3)
  ∧ (x - z) * (p1.3 - p3.3) = n * (p1.1 - p2.1)
  -- Ensure the points are not all identical and are collinear

def count_collinear_triples (points : Set (ℝ × ℝ × ℝ)) : ℕ :=
  Card {t | ∃ (p1 p2 p3 : ℝ × ℝ × ℝ), t = {p1, p2, p3} ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ collinear p1 p2 p3}

theorem collinear_triples_count : count_collinear_triples lattice_4x4x4 = 376 := by
  sorry

end collinear_triples_count_l101_101872


namespace find_original_number_l101_101806

-- Variables and assumptions
variables (N y x : ℕ)

-- Conditions
def crossing_out_condition : Prop := N = 10 * y + x
def sum_condition : Prop := N + y = 54321
def remainder_condition : Prop := x = 54321 % 11
def y_value : Prop := y = 4938

-- The final proof problem statement
theorem find_original_number (h1 : crossing_out_condition N y x) (h2 : sum_condition N y) 
  (h3 : remainder_condition x) (h4 : y_value y) : N = 49383 :=
sorry

end find_original_number_l101_101806


namespace magnitude_of_sum_l101_101963

variables (k : ℝ)
def vector_a := (1 : ℝ, 2 : ℝ)
def vector_b := (-2 : ℝ, k)

-- Definition of collinearity
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ (λ : ℝ), a.1 = λ * b.1 ∧ a.2 = λ * b.2

-- Prove that the magnitude of 3a + b is √5 given collinear condition
theorem magnitude_of_sum (h : collinear vector_a vector_b)
  (h1 : k = -4) :
  let vector_sum := (3 * 1 + (-2), 3 * 2 + (-4)) in
  ∥vector_sum∥ = real.sqrt 5 := 
by
  sorry -- Proof goes here

end magnitude_of_sum_l101_101963


namespace liam_markers_liam_first_markers_over_500_l101_101184

def seq (n : ℕ) : ℕ := 5 * 3^n

theorem liam_markers (n : ℕ) (h1 : seq 0 = 5) (h2 : seq 1 = 10) (h3 : ∀ k < n, 5 * 3^k ≤ 500) : 
  seq n > 500 := by sorry

theorem liam_first_markers_over_500 (h1 : seq 0 = 5) (h2 : seq 1 = 10) :
  ∃ n, seq n > 500 ∧ ∀ k < n, seq k ≤ 500 := by sorry

end liam_markers_liam_first_markers_over_500_l101_101184


namespace car_speed_is_8_times_walking_speed_l101_101279

theorem car_speed_is_8_times_walking_speed :
  ∀ (d t_car t_walk t_meet t_arrive : ℕ), 
  d = 2 → t_car = 30 → t_walk = 40 → t_meet = 10 → t_arrive = 20 →
  let car_speed := d * 2 / (t_arrive - t_meet) in
  let walk_speed := d / t_walk in
  car_speed = 8 * walk_speed :=
by
  sorry

end car_speed_is_8_times_walking_speed_l101_101279


namespace distance_light_travels_250_years_l101_101061

def distance_light_travels_one_year : ℝ := 5.87 * 10^12
def years : ℝ := 250

theorem distance_light_travels_250_years :
  distance_light_travels_one_year * years = 1.4675 * 10^15 :=
by
  sorry

end distance_light_travels_250_years_l101_101061


namespace solution_set_inequality_l101_101226

variable {f : ℝ → ℝ}

-- Declare the conditions as definitions and assumptions
def condition1 : Prop := ∀ x, f x + deriv f x < 2
def condition2 : Prop := f 1 = 3

theorem solution_set_inequality (h1 : condition1) (h2 : condition2) :
  {x : ℝ | exp x * f x > 2 * exp x + exp 1} = {x : ℝ | x < 1} :=
sorry

end solution_set_inequality_l101_101226


namespace zoo_total_animals_l101_101393

theorem zoo_total_animals (tiger_enclosure : ℕ) (zebra_enclosure_per_tiger : ℕ) 
  (giraffe_enclosures_ratio : ℕ) (tigers_per_enclosure : ℕ) 
  (zebras_per_enclosure : ℕ) (giraffes_per_enclosure : ℕ) 
  (Htiger : tiger_enclosure = 4) (Hzebra_per_tiger : zebra_enclosure_per_tiger = 2) 
  (Hgiraffe_ratio : giraffe_enclosures_ratio = 3) (Ht_pe : tigers_per_enclosure = 4) 
  (Hz_pe : zebras_per_enclosure = 10) (Hg_pe : giraffes_per_enclosure = 2) : 
  let zebra_enclosures := tiger_enclosure * zebra_enclosure_per_tiger in
  let giraffe_enclosures := zebra_enclosures * giraffe_enclosures_ratio in
  let total_tigers := tiger_enclosure * tigers_per_enclosure in
  let total_zebras := zebra_enclosures * zebras_per_enclosure in
  let total_giraffes := giraffe_enclosures * giraffes_per_enclosure in
  total_tigers + total_zebras + total_giraffes = 144 :=
by
  sorry

end zoo_total_animals_l101_101393


namespace min_value_dot_product_l101_101511

-- Side length of the square
def side_length: ℝ := 1

-- Definition of points in vector space
variables {A B C D O M N P: Type}

-- Definitions assuming standard Euclidean geometry
variables (O P : ℝ) (a b c : ℝ)

-- Points M and N on the edges AD and BC respectively, line MN passes through O
-- Point P satisfies 2 * vector OP = l * vector OA + (1-l) * vector OB
theorem min_value_dot_product (l : ℝ) (O P M N : ℝ) :
  (2 * (O + P)) = l * (O - a) + (1 - l) * (b + c) ∧
  ((O - P) * (O + P) - ((l^2 - l + 1/2) / 4) = -7/16) :=
by
  sorry

end min_value_dot_product_l101_101511


namespace sum_of_geometric_sequence_first_9000_terms_l101_101250

noncomputable def geomSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_of_geometric_sequence_first_9000_terms (a r : ℝ) (h1 : geomSum a r 3000 = 500) (h2 : geomSum a r 6000 = 950) :
  geomSum a r 9000 = 1355 :=
by
  sorry

end sum_of_geometric_sequence_first_9000_terms_l101_101250


namespace value_of_n_l101_101762

theorem value_of_n (n : ℝ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : n = 3.5 :=
by sorry

end value_of_n_l101_101762


namespace route_x_distance_route_y_distance_l101_101670

-- Definitions based on conditions:
def scale (map_inches : ℝ) : ℝ := 24 / 1.5 * map_inches -- 1 inch represents 16 miles

def route_x_map_dist_cm : ℝ := 44 -- Route X: 44 cm
def route_y_map_dist_cm : ℝ := 62 -- Route Y: 62 cm

def inches_to_cm (inches : ℝ) : ℝ := inches * 2.54
def cm_to_inches (cm : ℝ) : ℝ := cm / 2.54

-- Converted distances in inches:
def route_x_map_dist_inch := cm_to_inches route_x_map_dist_cm
def route_y_map_dist_inch := cm_to_inches route_y_map_dist_cm

-- Expected ground distances:
def expected_route_x_miles := 277.12
def expected_route_y_miles := 390.56

-- Calculations using the scale:
def ground_distance (map_dist_inch: ℝ) : ℝ := map_dist_inch * 16 -- 16 miles per inch

-- Proofs for route distances:
theorem route_x_distance :
  ground_distance route_x_map_dist_inch = expected_route_x_miles := by
    sorry

theorem route_y_distance :
  ground_distance route_y_map_dist_inch = expected_route_y_miles := by
    sorry

end route_x_distance_route_y_distance_l101_101670


namespace calculate_X_value_l101_101109

theorem calculate_X_value : 
  let M := (2025 : ℝ) / 3
  let N := M / 4
  let X := M - N
  X = 506.25 :=
by 
  sorry

end calculate_X_value_l101_101109


namespace eccentricity_of_ellipse_l101_101063

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a > b ∧ b > 0) : ℝ :=
  let A := (-a, 0)
  let B := (0, b)
  let AB := λ (x y : ℝ), b*x - a*y + a*b = 0
  let F := (-sqrt (a^2 - b^2), 0)
  let distance := abs (b * sqrt (a^2 - b^2) + a*b) / sqrt (a^2 + b^2)
  in if distance = b / sqrt 7 then 1/2
     else sorry

theorem eccentricity_of_ellipse (a b : ℝ) (h : a > b ∧ b > 0) :
  let e := ellipse_eccentricity a b h in
  e = 1/2 :=
by rw [ellipse_eccentricity] ; sorry

end eccentricity_of_ellipse_l101_101063


namespace find_original_number_l101_101799

open Int

theorem find_original_number (N y x : ℕ) (h1 : N = 10 * y + x) (h2 : N + y = 54321) (h3 : x = 54321 % 11) (h4 : y = 4938) : N = 49383 := by
  sorry

end find_original_number_l101_101799


namespace original_number_l101_101816

theorem original_number (N y x : ℕ) 
  (h1: N + y = 54321)
  (h2: N = 10 * y + x)
  (h3: 11 * y + x = 54321)
  (h4: x = 54321 % 11)
  (hy: y = 4938) : 
  N = 49383 := 
  by 
  sorry

end original_number_l101_101816


namespace original_five_digit_number_l101_101810

theorem original_five_digit_number :
  ∃ N y x : ℕ, (N = 10 * y + x) ∧ (N + y = 54321) ∧ (N = 49383) :=
by
  -- The proof script goes here
  sorry

end original_five_digit_number_l101_101810


namespace power_function_unique_l101_101957

theorem power_function_unique (f : ℝ → ℝ) (h : f 27 = 3) : f = (λ x, x^(1/3)) :=
sorry

end power_function_unique_l101_101957


namespace zoo_total_animals_l101_101396

def num_tiger_enclosures : ℕ := 4
def num_zebra_enclosures_per_tiger : ℕ := 2
def num_giraffe_enclosures_per_zebra : ℕ := 3
def num_tigers_per_enclosure : ℕ := 4
def num_zebras_per_enclosure : ℕ := 10
def num_giraffes_per_enclosure : ℕ := 2

theorem zoo_total_animals :
  let num_zebra_enclosures := num_tiger_enclosures * num_zebra_enclosures_per_tiger,
      num_giraffe_enclosures := num_zebra_enclosures * num_giraffe_enclosures_per_zebra,
      total_tigers := num_tiger_enclosures * num_tigers_per_enclosure,
      total_zebras := num_zebra_enclosures * num_zebras_per_enclosure,
      total_giraffes := num_giraffe_enclosures * num_giraffes_per_enclosure,
      total_animals := total_tigers + total_zebras + total_giraffes
  in total_animals = 144 :=
by
  intros
  let num_zebra_enclosures := num_tiger_enclosures * num_zebra_enclosures_per_tiger
  let num_giraffe_enclosures := num_zebra_enclosures * num_giraffe_enclosures_per_zebra
  let total_tigers := num_tiger_enclosures * num_tigers_per_enclosure
  let total_zebras := num_zebra_enclosures * num_zebras_per_enclosure
  let total_giraffes := num_giraffe_enclosures * num_giraffes_per_enclosure
  let total_animals := total_tigers + total_zebras + total_giraffes
  exact eq.refl 144

end zoo_total_animals_l101_101396


namespace sqrt_factorial_mul_self_eq_sqrt_factorial_mul_self_value_l101_101760

theorem sqrt_factorial_mul_self_eq :
  sqrt ((5!) * (5!)) = 5! :=
by sorry

theorem sqrt_factorial_mul_self_value :
  sqrt ((5!) * (5!)) = 120 :=
by {
  rw sqrt_factorial_mul_self_eq,
  norm_num,
  exact rfl,
  sorry
}

end sqrt_factorial_mul_self_eq_sqrt_factorial_mul_self_value_l101_101760


namespace delta_f_l101_101427

open BigOperators

def f (n : ℕ) : ℕ := ∑ i in Finset.range n, (i + 1) * (n - i)

theorem delta_f (k : ℕ) : f (k + 1) - f k = ∑ i in Finset.range (k + 1), (i + 1) :=
by
  sorry

end delta_f_l101_101427


namespace find_cos_A_l101_101134

variable (A B C : Type) [RealNotorderöd]

-- Assume triangle ABC is acute
def acute_triangle (a b c : Real) : Prop := a < π / 2 ∧ b < π / 2 ∧ c < π / 2

-- Assume area of triangle ABC is given
def area_triangle (ab ac sinA : Real) := (1 / 2) * ab * ac * sinA = 3 * sqrt 3 / 2

noncomputable def cos_a (a b c : Real) (ab ac : Real) (area : Real) : Prop :=
  let sinA := (area * 2) / (ab * ac)
  sinA = sqrt 3 / 2 → cos a = 1 / 2

-- Prove cos A
theorem find_cos_A (A B C : Real) (h1 : acute_triangle A B C) (h2 : area_triangle 2 3) : cos A = 1 / 2 :=
by
  sorry

end find_cos_A_l101_101134


namespace inverse_proportional_l101_101684

theorem inverse_proportional (p q : ℝ) (k : ℝ) 
  (h1 : ∀ (p q : ℝ), p * q = k)
  (h2 : p = 25)
  (h3 : q = 6) 
  (h4 : q = 15) : 
  p = 10 := 
by
  sorry

end inverse_proportional_l101_101684


namespace max_volume_of_tetrahedron_l101_101512

-- Define the conditions of the problem
variable (P A B C : Type) [MetricSpace P]
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable [HasDist A] [HasDist B] [HasDist C]

variable (d_PA : ℚ) (d_PB : ℚ) (d_AB : ℚ) (d_BC : ℚ) (d_CA : ℚ)
variable (max_volume : ℚ)

hypothesis h1 : d_PA = 3
hypothesis h2 : d_PB = 3
hypothesis h3 : d_AB = 2
hypothesis h4 : d_BC = 2
hypothesis h5 : d_CA = 2

-- Define the problem to prove
theorem max_volume_of_tetrahedron : 
  max_volume = (2 * Real.sqrt 6) / 3 := by
  sorry

end max_volume_of_tetrahedron_l101_101512


namespace multiples_of_7_with_unit_digit_7_and_lt_150_l101_101101

/--
Given a positive integer n,
Let \(7n\) be a multiple of 7,
\(7n < 150\),
and \(7n \mod 10 = 7\),
prove there are 11 such multiples of 7.
-/
theorem multiples_of_7_with_unit_digit_7_and_lt_150 :
  ∃! (n : ℕ), n ≤ 11 ∧ ∀ (m : ℕ), (1 ≤ m ∧ m ≤ n) → (7 * (2 * m - 1) < 150 ∧ (7 * (2 * m - 1)) % 10 = 7) := by
sorry

end multiples_of_7_with_unit_digit_7_and_lt_150_l101_101101


namespace min_value_dist_sq_l101_101917

noncomputable def circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (1, 0)

def dist_sq (p₁ p₂ : ℝ × ℝ) : ℝ :=
  (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2

theorem min_value_dist_sq {x y : ℝ} (hP : circle x y) :
  (dist_sq point_A (x, y) + dist_sq point_B (x, y)) = 20 := sorry

end min_value_dist_sq_l101_101917


namespace coefficient_of_x3_term_l101_101617

-- Definition of polynomial conditions
def polynomial : ℤ[X] := (X^2 - 2 * X) * (1 + X)^6

-- Statement to prove the coefficient of the x^3 term
theorem coefficient_of_x3_term : polynomial.coeff 3 = -24 := by
  -- This is a placeholder to indicate that the proof itself is to be provided
  sorry

end coefficient_of_x3_term_l101_101617


namespace tax_deduction_is_correct_l101_101850

-- Define the hourly wage and tax rate
def hourly_wage_dollars : ℝ := 25
def tax_rate : ℝ := 0.021

-- Define the conversion from dollars to cents
def dollars_to_cents (dollars : ℝ) : ℝ := dollars * 100

-- Calculate the hourly wage in cents
def hourly_wage_cents : ℝ := dollars_to_cents hourly_wage_dollars

-- Calculate the tax deducted in cents per hour
def tax_deduction_cents (wage : ℝ) (rate : ℝ) : ℝ := rate * wage

-- State the theorem that needs to be proven
theorem tax_deduction_is_correct :
  tax_deduction_cents hourly_wage_cents tax_rate = 52.5 :=
by
  sorry

end tax_deduction_is_correct_l101_101850


namespace solve_proof_problem_l101_101523

noncomputable def proof_problem (alpha : ℝ) :=
  𝚜𝚎𝚌𝚘𝚗𝚍𝚀𝚞𝚊𝚍𝚛𝚊𝚗𝚝 : ∃ k : ℤ, α ∈ set.Ioo (k * (2 * Real.pi) + Real.pi / 2) (k * (2 * Real.pi) + Real.pi) ∧
  (h : Real.sin (alpha + Real.pi / 6) = 1 / 3),
  Real.sin (2 * alpha + Real.pi / 3) = -4 * Real.sqrt 2 / 9

theorem solve_proof_problem (alpha : ℝ) (h1 : ∃ k : ℤ, alpha ∈ set.Ioo (k * (2 * Real.pi) + Real.pi / 2) (k * (2 * Real.pi) + Real.pi)) 
                             (h2 : Real.sin (alpha + Real.pi / 6) = 1 / 3) :
  Real.sin (2 * alpha + Real.pi / 3) = -4 * Real.sqrt 2 / 9 :=
sorry

end solve_proof_problem_l101_101523


namespace logs_in_stack_l101_101384

theorem logs_in_stack (a l : ℕ) (n : ℕ) (S : ℕ)
  (a_eq_five : a = 5)
  (l_eq_fifteen : l = 15)
  (n_eq_eleven : n = 15 - 5 + 1)
  (S_eq_sum : S = n / 2 * (a + l)) :
  S = 110 :=
by
  rw [a_eq_five, l_eq_fifteen, n_eq_eleven, S_eq_sum]
  sorry

end logs_in_stack_l101_101384


namespace Q_at_2_l101_101724

-- Define the polynomial Q(x)
def Q (x : ℚ) : ℚ := x^4 - 8 * x^2 + 16

-- Define the properties of Q(x)
def is_special_polynomial (P : (ℚ → ℚ)) : Prop := 
  degree P = 4 ∧ leading_coeff P = 1 ∧ P.is_root(√3 + √7)

-- Problem: Prove Q(2) = 0 given the polynomial Q meets the conditions.
theorem Q_at_2 (Q : ℚ → ℚ) (h1 : degree Q = 4) (h2 : leading_coeff Q = 1) (h3 : is_root Q (√3 + √7)) : 
  Q 2 = 0 := by
  sorry

end Q_at_2_l101_101724


namespace intersection_points_l101_101059

def periodic_func (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def func_def (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 ≤ x ∧ x < 2) → f x = x^3 - x

theorem intersection_points (f : ℝ → ℝ)
  (h_periodic : periodic_func f 2)
  (h_def : func_def f) :
  (∃ s : set ℝ, s = { x ∈ set.Icc 0 6 | f x = 0 } ∧ s.to_finset.card = 7) :=
sorry

end intersection_points_l101_101059


namespace m_minus_n_is_4_l101_101070

def f (x : ℝ) (m n : ℝ) : ℝ :=
  if x > 0 then m * Real.log 2017 x + 3 * Real.sin x
  else Real.log 2017 (-x) + n * Real.sin x

theorem m_minus_n_is_4 {m n : ℝ} 
  (hf : ∀ x : ℝ, f x m n = f (-x) m n) : m - n = 4 :=
by
  have h_pos : ∀ x > 0, f x m n = f (-x) m n := λ x hx, hf x
  have h_neg : ∀ x < 0, f x m n = f (-x) m n := λ x hx, hf x

  -- Show that m = 1
  have m_eq_1 : m = 1 := 
    by
      -- taking any positive x (e.g., x = 1)
      specialize h_pos 1 (by linarith)
      simp at h_pos
      have : m * Real.log 2017 1 + 3 * Real.sin 1 = Real.log 2017 1 - n * Real.sin 1 := h_pos
      have log1 : Real.log 2017 1 = 0 := sorry
      have sin1 : 3 * Real.sin 1 + n * Real.sin 1 = 0 := sorry
      have : m * 0 + 3 * Real.sin 1 = 0 - n * Real.sin 1 := sorry
      linarith

  -- Show that n = -3
  have n_eq_n3 : n = -3 := 
    by
      specialize h_pos 1 (by linarith)
      simp at h_pos
      sorry

  simp [m_eq_1, n_eq_n3]
  linarith

end m_minus_n_is_4_l101_101070


namespace martha_blue_butterflies_l101_101664

-- Definitions based on conditions
variables (total_butterflies : ℕ) (black_butterflies : ℕ)
variables (yellow_butterflies : ℕ) (blue_butterflies : ℕ)

def total_is_11 : Prop := total_butterflies = 11
def black_is_5 : Prop := black_butterflies = 5
def blue_is_twice_yellow : Prop := blue_butterflies = 2 * yellow_butterflies
def remaining_is_blue_and_yellow : Prop := total_butterflies - black_butterflies = blue_butterflies + yellow_butterflies

-- The statement we want to prove
theorem martha_blue_butterflies (h1 : total_is_11) (h2 : black_is_5) 
    (h3 : blue_is_twice_yellow) (h4 : remaining_is_blue_and_yellow) :
    blue_butterflies = 4 := 
begin
  sorry
end

end martha_blue_butterflies_l101_101664


namespace triangle_area_DE_F_l101_101123

theorem triangle_area_DE_F (DE EF DF : ℝ) (hDE : DE = 35) (hEF : EF = 35) (hDF : DF = 54) :
  let s := (DE + EF + DF) / 2 in
  let area := Real.sqrt (s * (s - DE) * (s - EF) * (s - DF)) in
  area = 602 := by
  sorry

end triangle_area_DE_F_l101_101123


namespace sin_eq_product_one_eighth_l101_101024

open Real

theorem sin_eq_product_one_eighth :
  (∀ (n k m : ℕ), 1 ≤ n → n ≤ 5 → 1 ≤ k → k ≤ 5 → 1 ≤ m → m ≤ 5 →
    sin (π * n / 12) * sin (π * k / 12) * sin (π * m / 12) = 1 / 8) ↔ (n = 2 ∧ k = 2 ∧ m = 2) := by
  sorry

end sin_eq_product_one_eighth_l101_101024


namespace number_of_partitions_l101_101423

-- Definition: n-staircase
def n_staircase (n : ℕ) : set (ℕ × ℕ) :=
  {p | 1 ≤ p.1 ∧ p.1 ≤ p.2 ∧ p.2 ≤ n}

-- The main theorem to be proven
theorem number_of_partitions (n : ℕ) :
  (∃ partitions : list (set (ℕ × ℕ)), 
    (∀ rect ∈ partitions, rect ⊆ n_staircase n ∧ rect.is_rectangle) ∧ 
    (partitions.pairwise_disjoint ∧ partitions.areas_are_distinct)) → 
    partitions.length = 2^(n-1) :=
by sorry

end number_of_partitions_l101_101423


namespace complex_modulus_example_correct_l101_101489

noncomputable def complex_modulus_example : ℂ := -3 + (8 / 5.0) * I

theorem complex_modulus_example_correct :
  complex.norm complex_modulus_example = 17 / 5 := 
by 
  sorry

end complex_modulus_example_correct_l101_101489


namespace count_positive_integers_in_range_l101_101899

theorem count_positive_integers_in_range (a b : ℕ) (hx : a = 144) (hy : b = 256) :
  {x : ℕ | a ≤ x^2 ∧ x^2 ≤ b}.card = 5 :=
by
  sorry

end count_positive_integers_in_range_l101_101899


namespace skirts_to_add_l101_101188

-- Definitions based on conditions
def skirt_price : ℕ := 20
def blouse_price : ℕ := 15
def pant_price : ℕ := 30
def total_budget : ℕ := 180
def num_blouses : ℕ := 5
def num_pants : ℕ := 2

-- Main theorem stating that Marcia can buy 3 skirts
theorem skirts_to_add : 
  let total_blouse_cost := num_blouses * blouse_price in
  let discounted_pant_cost := pant_price + pant_price / 2 in
  let total_pant_cost := discounted_pant_cost in
  let total_spent := total_blouse_cost + total_pant_cost in
  let remaining_budget := total_budget - total_spent in
  let num_skirts := remaining_budget / skirt_price in
  num_skirts = 3 :=
by
  -- Here the proof steps would go
  sorry

end skirts_to_add_l101_101188


namespace bucket_water_l101_101431

theorem bucket_water (oz1 oz2 oz3 oz4 oz5 total1 total2: ℕ) 
  (h1 : oz1 = 11)
  (h2 : oz2 = 13)
  (h3 : oz3 = 12)
  (h4 : oz4 = 16)
  (h5 : oz5 = 10)
  (h_total : total1 = oz1 + oz2 + oz3 + oz4 + oz5)
  (h_second_bucket : total2 = 39)
  : total1 - total2 = 23 :=
sorry

end bucket_water_l101_101431


namespace directrix_of_parabola_l101_101227

theorem directrix_of_parabola (p : ℝ) (hp : 2 * p = 4) : 
  (∃ x : ℝ, x = -1) :=
by
  sorry

end directrix_of_parabola_l101_101227


namespace not_factorial_tails_l101_101463

noncomputable def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) + ...

theorem not_factorial_tails :
  let n := 2500 in ∃ (k : ℕ), k = 500 ∧ ∀ m < n, ¬(f m = k) :=
by {
  sorry
}

end not_factorial_tails_l101_101463


namespace monotonicity_x_pow_2_over_3_l101_101704

noncomputable def x_pow_2_over_3 (x : ℝ) : ℝ := x^(2/3)

theorem monotonicity_x_pow_2_over_3 : ∀ x y : ℝ, 0 < x → x < y → x_pow_2_over_3 x < x_pow_2_over_3 y :=
by
  intros x y hx hxy
  sorry

end monotonicity_x_pow_2_over_3_l101_101704


namespace sum_first_9000_terms_l101_101245

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_9000_terms (a r : ℝ) :
  geometric_sequence_sum a r 3000 = 500 →
  geometric_sequence_sum a r 6000 = 950 →
  geometric_sequence_sum a r 9000 = 1355 :=
by
  intros h1 h2
  sorry

end sum_first_9000_terms_l101_101245


namespace remainder_of_polynomial_division_l101_101753

theorem remainder_of_polynomial_division :
  let y := (x : ℝ) → x^5 - 8*x^4 + 15*x^3 + 30*x^2 - 47*x + 20
  y 2 = 104 :=
by
  let y := (x : ℝ) → x^5 - 8*x^4 + 15*x^3 + 30*x^2 - 47*x + 20
  sorry

end remainder_of_polynomial_division_l101_101753


namespace parkway_elementary_students_l101_101622

/-- The total number of students in the fifth grade at Parkway Elementary School is 420,
given the following conditions:
1. There are 312 boys.
2. 250 students are playing soccer.
3. 78% of the students that play soccer are boys.
4. There are 53 girl students not playing soccer. -/
theorem parkway_elementary_students (boys : ℕ) (playing_soccer : ℕ) (percent_boys_playing : ℝ) (girls_not_playing_soccer : ℕ)
  (h1 : boys = 312)
  (h2 : playing_soccer = 250)
  (h3 : percent_boys_playing = 0.78)
  (h4 : girls_not_playing_soccer = 53) :
  ∃ total_students : ℕ, total_students = 420 :=
by
  sorry

end parkway_elementary_students_l101_101622


namespace four_friendly_hands_with_largest_7_l101_101593

-- Definition of a friendly hand
def is_friendly_hand (cards : List ℕ) : Prop :=
  cards.length = 4 ∧ cards.sum = 24

-- Definition of having 7 as the largest number
def has_largest_7 (cards : List ℕ) : Prop :=
  List.maximum cards = some 7

-- The problem statement: There are 4 distinct friendly hands with the largest number being 7
theorem four_friendly_hands_with_largest_7 :
  (finset.filter (λ cards, is_friendly_hand cards ∧ has_largest_7 cards)
    (finset.powerset (finset.range 10)).filter (λ cards, cards.cardinality = 4)).card = 4 :=
by
  sorry

end four_friendly_hands_with_largest_7_l101_101593


namespace rotate_3_minus_sqrt3i_by_pi_over_3_l101_101616

def rotate_complex_clockwise (z : Complex) (theta : ℝ) : Complex :=
  z * Complex.conj (Complex.exp (-Complex.I * theta))

theorem rotate_3_minus_sqrt3i_by_pi_over_3 :
  rotate_complex_clockwise (3 - Real.sqrt 3 * Complex.I) (Real.pi / 3) = -2 * Real.sqrt 3 * Complex.I :=
by
  -- Placeholder for the proof
  sorry

end rotate_3_minus_sqrt3i_by_pi_over_3_l101_101616


namespace martha_blue_butterflies_l101_101665

-- Definitions based on conditions
variables (total_butterflies : ℕ) (black_butterflies : ℕ)
variables (yellow_butterflies : ℕ) (blue_butterflies : ℕ)

def total_is_11 : Prop := total_butterflies = 11
def black_is_5 : Prop := black_butterflies = 5
def blue_is_twice_yellow : Prop := blue_butterflies = 2 * yellow_butterflies
def remaining_is_blue_and_yellow : Prop := total_butterflies - black_butterflies = blue_butterflies + yellow_butterflies

-- The statement we want to prove
theorem martha_blue_butterflies (h1 : total_is_11) (h2 : black_is_5) 
    (h3 : blue_is_twice_yellow) (h4 : remaining_is_blue_and_yellow) :
    blue_butterflies = 4 := 
begin
  sorry
end

end martha_blue_butterflies_l101_101665


namespace right_angled_pyramid_property_1_right_angled_pyramid_property_2_l101_101377

/-- Definition of a right-angled pyramid -/
structure RightAngledPyramid (V : Type*) [NormedAddCommGroup V] [NormedSpace ℝ V] : Type* :=
(lateral_edges_perpendicular : ∀ v1 v2 v3 : V, v1 ⊥ v2 ∧ v2 ⊥ v3 ∧ v1 ⊥ v3)
(lateral_faces_right_angled)
(base : Set V) -- This represents the oblique face

/-- Definition of the mid-face of the oblique face -/
def MidFaceOfObliqueFace (pyramid : RightAngledPyramid V) : Set V := 
-- Mid-face passing through the vertex and midpoints of two sides of the oblique face.
sorry

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem right_angled_pyramid_property_1 (pyramid : RightAngledPyramid V) :
  let A₁ A₂ A₃ A₄ := sorry in -- areas of the right-angled faces and the oblique face
  (A₁^2 + A₂^2 + A₃^2 = A₄^2) :=
sorry

theorem right_angled_pyramid_property_2 (pyramid : RightAngledPyramid V) :
  let A₄ := sorry in -- area of the oblique face
  let A_mid := sorry in -- area of the mid-face of the oblique face
  (A_mid = (1/4) * A₄) :=
sorry

end right_angled_pyramid_property_1_right_angled_pyramid_property_2_l101_101377


namespace maintain_ratio_l101_101189

def ratio_ingredients (flour salt : ℕ) : Prop := flour = 2 * salt

theorem maintain_ratio (sugar : ℕ) : 
  ∀ (initial_flour initial_salt desired_flour : ℕ), 
  ratio_ingredients desired_flour initial_salt → 
  initial_flour = 2 → initial_salt = 2 → 
  initial_flour + 2 = desired_flour + initial_salt := 
by
  intros
  rw [ratio_ingredients] at H
  rw [H]
  rw [H] at *
  rw [H at H_0]
  assumption
  sorry

end maintain_ratio_l101_101189


namespace grisha_wins_with_104_l101_101332

theorem grisha_wins_with_104 : 
  ∀ (P : ℕ), P ∈ (1:ℕ)..(105:ℕ) ∧ P % 2 = 0 → 
  (∃ w:number_pairs P, w = max_number_pairs_in_range)
--- conditions
-- P ∈ 1..105 
-- P % 2 = 0 (P must be even)
-- number_pairs P is defined as: 
-- number_pairs P = ⌊ P/4 ⌋
-- max_number_pairs_in_range is the maximum number of pairs within the given range (1..105)
--- statement
-- Grisha wins with the number 104
by sorry

end grisha_wins_with_104_l101_101332


namespace min_distance_transform_g_correct_l101_101052

noncomputable def z1 (x : ℝ) := Complex.mk (Real.cos x) 1
noncomputable def z2 (x : ℝ) := Complex.mk 1 (-Real.sin x)
noncomputable def distance (x : ℝ) := Complex.abs (z1 x - z2 x)

theorem min_distance : ∃ x : ℝ, distance x = Real.sqrt 2 - 1 := 
sorry

noncomputable def f (x : ℝ) := 1 - (1 / 2) * Real.sin (2 * x)
noncomputable def g (x : ℝ) := 1 + (1 / 2) * Real.cos x

theorem transform_g_correct : ∀ x : ℝ, 
  let y1 := 1 - (1 / 2) * Real.sin x;
  let g_transformed := y1 (x - (Real.pi / 2)) in
  g_transformed = g x := 
sorry

end min_distance_transform_g_correct_l101_101052


namespace determine_valid_m_l101_101950

-- The function given in the problem
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x + m + 2

-- The range of values for m
def valid_m (m : ℝ) : Prop := -1/4 ≤ m ∧ m ≤ 0

-- The condition that f is increasing on (-∞, 2)
def increasing_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < a → f x ≤ f y

-- The main statement we want to prove
theorem determine_valid_m (m : ℝ) :
  increasing_on_interval (f m) 2 ↔ valid_m m :=
sorry

end determine_valid_m_l101_101950


namespace evaluate_Q_at_2_l101_101727

-- Define the polynomial Q(x)
noncomputable def Q (x : ℚ) : ℚ := x^4 - 20 * x^2 + 16

-- Define the root condition of sqrt(3) + sqrt(7)
def root_condition (x : ℚ) : Prop := (x = ℚ.sqrt(3) + ℚ.sqrt(7))

-- Theorem statement
theorem evaluate_Q_at_2 : Q 2 = -32 :=
by
  -- Given conditions
  have h1 : degree Q = 4 := sorry,
  have h2 : leading_coeff Q = 1 := sorry,
  have h3 : root_condition (ℚ.sqrt(3) + ℚ.sqrt(7)) := sorry,
  -- Goal
  exact sorry

end evaluate_Q_at_2_l101_101727


namespace fifty_percent_of_number_l101_101576

noncomputable def given_number (N : ℝ) : Prop :=
  1.15 * ((1/4) * (1/3) * (2/5) * N) = 23

theorem fifty_percent_of_number (N : ℝ) (h : given_number N) : 0.5 * N = 300 :=
begin
  sorry
end

end fifty_percent_of_number_l101_101576


namespace min_a1a3_value_l101_101912

noncomputable def minimum_a1a3 (a : ℕ → ℝ) : ℝ :=
  let x := a 1 + 1
  let y := a 3 + 3 in
  x * y - 3 * x - y + 3

theorem min_a1a3_value (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_a2 : a 2 = 6)
  (h_arith_seq : (1 / (a 1 + 1) + 1 / (a 3 + 3)) = 1 / (a 2 + 2)) :
  minimum_a1a3 a = 19 + 8 * real.sqrt 3 :=
by
  sorry

end min_a1a3_value_l101_101912


namespace octagon_Q_area_l101_101425

noncomputable def octagon_area (apothem : ℝ) : ℝ :=
  let side_length := 6 * (Real.sqrt 2 - 1) in
  let smaller_side_length := side_length / 2 in
  2 * (smaller_side_length ^ 2) * (1 + Real.sqrt 2)

theorem octagon_Q_area :
  octagon_area 3 = 36 * Real.sqrt 2 - 36 :=
by
  sorry

end octagon_Q_area_l101_101425


namespace john_more_needed_l101_101156

def john_needs : ℝ := 2.5
def john_has : ℝ := 0.75
def john_needs_more : ℝ := 1.75

theorem john_more_needed : (john_needs - john_has) = john_needs_more :=
by
  sorry

end john_more_needed_l101_101156


namespace zero_count_in_interval_l101_101241

def f (x : ℝ) : ℝ := Real.tan (1935 * x) - Real.tan (2021 * x) + Real.tan (2107 * x)

theorem zero_count_in_interval :
  ∃ (n : ℕ), n = 2022 ∧ ∀ x ∈ Set.Icc 0 Real.pi, f x = 0 → x = (k : ℕ) * Real.pi / 2021 ∧ k ∈ Finset.range 2022 := 
sorry

end zero_count_in_interval_l101_101241


namespace ratio_volumes_of_spheres_l101_101584

theorem ratio_volumes_of_spheres (r R : ℝ) (hratio : (4 * π * r^2) / (4 * π * R^2) = 4 / 9) :
    (4 / 3 * π * r^3) / (4 / 3 * π * R^3) = 8 / 27 := 
by {
  sorry
}

end ratio_volumes_of_spheres_l101_101584


namespace no_intersections_l101_101871

noncomputable def abs_eq (a b : ℝ) : ℝ := abs (a - b)

def f1 (x : ℝ) : ℝ := abs (3 * x + 6)
def f2 (x : ℝ) : ℝ := -abs (4 * x - 3)

theorem no_intersections :
  ∀ x y : ℝ, (y = f1 x) → (y = f2 x) → False :=
by
  assume x y h1 h2
  sorry

end no_intersections_l101_101871


namespace number_of_games_in_chess_tournament_l101_101975

theorem number_of_games_in_chess_tournament 
  (n : ℕ) (h : n = 19) : (∑ i in finset.range n, i) / 2 = 171 :=
by
  sorry

end number_of_games_in_chess_tournament_l101_101975


namespace multiples_of_7_units_digit_7_l101_101089

theorem multiples_of_7_units_digit_7 (n : ℕ) (h1 : n < 150) (h2 : ∃ (k : ℕ), n = 7 * k) (h3 : n % 10 = 7) : 
    ∃ (m : ℕ), m = 2 := 
by
  sorry

end multiples_of_7_units_digit_7_l101_101089


namespace find_k_exists_p3_p5_no_number_has_p2_and_p4_l101_101573

def has_prop_pk (n k : ℕ) : Prop := ∃ lst : List ℕ, (∀ x ∈ lst, x > 1) ∧ (lst.length = k) ∧ (lst.prod = n)

theorem find_k_exists_p3_p5 :
  ∃ (k : ℕ), (k = 3) ∧ ∃ (n : ℕ), has_prop_pk n k ∧ has_prop_pk n (k + 2) :=
by {
  sorry
}

theorem no_number_has_p2_and_p4 :
  ¬ ∃ (n : ℕ), has_prop_pk n 2 ∧ has_prop_pk n 4 :=
by {
  sorry
}

end find_k_exists_p3_p5_no_number_has_p2_and_p4_l101_101573


namespace functional_equation_solution_l101_101866

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f(2 * x + f(y)) = x + y + f(x)) → (∀ x : ℝ, f(x) = x) :=
by
  intros f H
  sorry

end functional_equation_solution_l101_101866


namespace modulus_of_Z_l101_101781

noncomputable def Z : ℂ := Complex.ofReal (2 * (Complex.I)) / Complex.ofReal (1 - Complex.I)

-- Prove that the modulus of the complex number Z is √2 given that (1-i)Z = 2i.
theorem modulus_of_Z : (1 - Complex.I) * Z = 2 * Complex.I → Complex.abs Z = Real.sqrt 2 :=
by
  intro hyp
  -- omitting the proof steps, only the statement is required
  sorry

end modulus_of_Z_l101_101781


namespace distance_along_stream_l101_101138
-- Define the problem in Lean 4

noncomputable def speed_boat_still : ℝ := 11   -- Speed of the boat in still water
noncomputable def distance_against_stream : ℝ := 9  -- Distance traveled against the stream in one hour

theorem distance_along_stream : 
  ∃ (v_s : ℝ), (speed_boat_still - v_s = distance_against_stream) ∧ (11 + v_s) * 1 = 13 := 
by
  use 2
  sorry

end distance_along_stream_l101_101138


namespace min_people_all_items_l101_101602

noncomputable def minimum_people_wearing_all_items (n : ℕ) (gloves hats scarves : ℕ) :=
  1 / 3 * n = gloves ∧ 2 / 3 * n = hats ∧ 1 / 2 * n = scarves → gloves + hats + scarves - (gloves + hats - 6) - (gloves + scarves - 6) - (hats + scarves - 6) + (gloves + hats + scarves - n) = 6

theorem min_people_all_items (n : ℕ) (gloves hats scarves : ℕ) :
  n = 12 → 1 / 3 * n = 4 → 2 / 3 * n = 8 → 1 / 2 * n = 6 → minimum_people_wearing_all_items n gloves hats scarves = 6 :=
by 
  sorry

end min_people_all_items_l101_101602


namespace sum_of_divisors_37_l101_101317

theorem sum_of_divisors_37 : ∑ d in (finset.filter (λ d, 37 % d = 0) (finset.range 38)), d = 38 :=
by {
  sorry
}

end sum_of_divisors_37_l101_101317


namespace find_original_number_l101_101801

open Int

theorem find_original_number (N y x : ℕ) (h1 : N = 10 * y + x) (h2 : N + y = 54321) (h3 : x = 54321 % 11) (h4 : y = 4938) : N = 49383 := by
  sorry

end find_original_number_l101_101801


namespace values_of_a_and_b_l101_101876

noncomputable def a_value := -(75 / 2)
noncomputable def b_value := (59 / 2)
noncomputable def f := (a_value : ℚ) * X^4 + (b_value : ℚ) * X^3 + 38 * X^2 - 12 * X + 15
noncomputable def g := 3 * X^2 - 2 * X + 2

theorem values_of_a_and_b :
  (∃ (u : Polynomial ℚ), f = g * u) →
  a_value = -(75 / 2) ∧ b_value = (59 / 2) :=
by
  sorry

end values_of_a_and_b_l101_101876


namespace partition_intersect_lines_l101_101778

variables (N : ℕ)
variables (a : ℕ → ℕ) (b : ℕ → ℕ)

theorem partition_intersect_lines (h_partition : ∀ n, n ≥ 1 → ∀ l ∈ partition_sets n, all_parallel_lines l ∈ partition_sets n)
  (h_intersect : ∀ n, n ≥ 2 → ∀ p ∈ intersect_points n, exactly_n_lines_intersect_at p n) :
  ∑ n in (finset.range (N + 1)).filter (λ n, n ≥ 2), (a n + b n) * nat.choose n 2 = nat.choose N 2 :=
by
  sorry

end partition_intersect_lines_l101_101778


namespace chord_length_probability_l101_101348

noncomputable theory
open_locale classical

/-- 
  Given a circle with radius R and a fixed point M on its circumference,
  the probability that a randomly chosen point N on the circumference results in
  the chord MN having length greater than sqrt(3) * R is 1/3.
-/
theorem chord_length_probability (R : ℝ) (h_r_pos : R > 0) (M N : {p: ℝ × ℝ // p.1^2 + p.2^2 = R^2}) :
  let chord_length_exceeds := dist M.1 N.1 > sqrt(3) * R
  in ℙ (chord_length_exceeds) = 1 / 3 :=
sorry

end chord_length_probability_l101_101348


namespace IH_perp_AD_l101_101405

variables {A B C O M P Q H I N D : Type}
variables [circumcircle : circle O (triangle A B C)]
variables [AB_lt_AC : AB < AC]
variables [angle_BAC : ∠BAC = 120°]
variables [midpoint_M : is_midpoint_of_arc M A C (circumcircle)]
variables [tangents_PA_PB : is_tangent PA circumcircle ∧ is_tangent PB circumcircle]
variables [tangents_QA_QC : is_tangent QA circumcircle ∧ is_tangent QC circumcircle]
variables [orthocenter_H : is_orthocenter H (triangle P O Q)]
variables [incenter_I : is_incenter I (triangle P O Q)]
variables [midpoint_N : is_midpoint N O I]
variables [second_intersection_D : is_second_intersection D (line M N) circumcircle]

theorem IH_perp_AD : ⦃ H I A D : Type ⦄ (IH : IH_perp (line I H) (line A D)) :=
  sorry

end IH_perp_AD_l101_101405


namespace geometric_sequence_a9_value_l101_101529

theorem geometric_sequence_a9_value {a : ℕ → ℝ} (q a1 : ℝ) 
  (h_geom : ∀ n, a n = a1 * q ^ n)
  (h_a3 : a 3 = 2)
  (S : ℕ → ℝ)
  (h_S : ∀ n, S n = a1 * (1 - q ^ n) / (1 - q))
  (h_sum : S 12 = 4 * S 6) : a 9 = 2 := 
by 
  sorry

end geometric_sequence_a9_value_l101_101529


namespace find_numbers_l101_101236

def is_reverse (n m : ℕ) : Prop :=
  let digits := n.digits 10
  let reversed_digits := digits.reverse
  m = reversed_digits.foldl (λ acc d, 10 * acc + d) 0

def product_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).foldl (λ acc d, acc * d) 1

theorem find_numbers (X : ℕ) (h1 : X.digits 10 ≠ 0) :
  let P := product_of_digits X
  let r := (λ X, [X % 10, (X / 10) % 10, X / 100]).reverse
    X * r = 1000 + P
  P = 24 ∨ P = 42 :=
  sorry

end find_numbers_l101_101236


namespace largest_term_l101_101645

-- Given conditions
def U : ℕ := 2 * (2010 ^ 2011)
def V : ℕ := 2010 ^ 2011
def W : ℕ := 2009 * (2010 ^ 2010)
def X : ℕ := 2 * (2010 ^ 2010)
def Y : ℕ := 2010 ^ 2010
def Z : ℕ := 2010 ^ 2009

-- Proposition to prove
theorem largest_term : 
  (U - V) > (V - W) ∧ 
  (U - V) > (W - X + 100) ∧ 
  (U - V) > (X - Y) ∧ 
  (U - V) > (Y - Z) := 
by 
  sorry

end largest_term_l101_101645


namespace hyperbola_parabola_asymptotes_l101_101557

theorem hyperbola_parabola_asymptotes (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : ∃ (x y : ℝ), (-3, 0) = (x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2, y ^ 2 = -12 * x)) 
  (h₃ : ∀ (x : ℝ), (x = a * sqrt 2 ∨ x = -a * sqrt 2)) :
  a = sqrt 3 := 
sorry

end hyperbola_parabola_asymptotes_l101_101557


namespace probability_angle_acute_l101_101345

theorem probability_angle_acute :
  let O : Point
  let A B : Point
  (angle (A, O, B)).degree < 90 :=
sorry

end probability_angle_acute_l101_101345


namespace root_quadratic_expression_value_l101_101531

theorem root_quadratic_expression_value (m : ℝ) (h : m^2 - m - 3 = 0) : 2023 - m^2 + m = 2020 := 
by 
  sorry

end root_quadratic_expression_value_l101_101531


namespace total_cost_of_deck_l101_101291

def num_rare_cards : ℕ := 19
def num_uncommon_cards : ℕ := 11
def num_common_cards : ℕ := 30

def cost_per_rare_card : ℕ := 1
def cost_per_uncommon_card : ℝ := 0.50
def cost_per_common_card : ℝ := 0.25

def cost_of_rare_cards : ℕ := num_rare_cards * cost_per_rare_card
def cost_of_uncommon_cards : ℝ := num_uncommon_cards * cost_per_uncommon_card
def cost_of_common_cards : ℝ := num_common_cards * cost_per_common_card

def total_cost : ℝ := cost_of_rare_cards + cost_of_uncommon_cards + cost_of_common_cards

theorem total_cost_of_deck : total_cost = 32 := by
  -- We will need to convert integers to real numbers for this addition
  have h1 : (cost_of_rare_cards : ℝ) = 19 := by norm_cast
  rw [h1]
  have h2 : (num_uncommon_cards: ℝ) * cost_per_uncommon_card = 5.5 := by norm_num
  have h3 : (num_common_cards: ℝ) * cost_per_common_card = 7.5 := by norm_num
  calc
    (19 : ℝ) + 5.5 + 7.5 = 32 := by norm_num

end total_cost_of_deck_l101_101291


namespace virus_affected_computers_l101_101607

theorem virus_affected_computers (m n : ℕ) (h1 : 5 * m + 2 * n = 52) : m = 8 :=
by
  sorry

end virus_affected_computers_l101_101607


namespace inclination_angle_of_line_l101_101937

theorem inclination_angle_of_line :
  let k := 1 in
  let α := Real.arctan k in
  (α * 180 / Real.pi) = 45 :=
by
  let k := 1
  let α := Real.arctan k
  have h1 : α = Real.pi / 4 := sorry
  have h2 : (Real.pi / 4 * 180 / Real.pi) = 45 := sorry
  exact h2

end inclination_angle_of_line_l101_101937


namespace maxOmega_is_2_l101_101678

noncomputable def maxOmega (ω : ℝ) : ℝ :=
  if ((ω > 0) ∧ 2*sin(ω*(-π/6)) ≥ 2*sin(-π/2) ∧ 2*sin(ω*(π/4)) ≤ 2*sin(π/2)) then 2 else 0

theorem maxOmega_is_2 (ω : ℝ) (h₁ : ω > 0) (h₂ : 2*sin(ω*(-π/6)) ≥ 2*sin(-π/2))
  (h₃ : 2*sin(ω*(π/4)) ≤ 2*sin(π/2)) : maxOmega ω = 2 :=
by
  sorry

end maxOmega_is_2_l101_101678


namespace sum_abs_diff_mod_10_l101_101713

theorem sum_abs_diff_mod_10 {a b : Fin 999 → ℕ} 
  (h1 : ∀ x, 1 ≤ a x ∧ a x ≤ 1998) 
  (h2 : ∀ x, 1 ≤ b x ∧ b x ≤ 1998)
  (h3 : Set.toFinset (Set.Range a) ∪ Set.toFinset (Set.Range b) = Finset.range 1 1999)
  (h4 : ∀ i, (|a i - b i| = 1) ∨ (|a i - b i| = 6)) :
  (Finset.univ.sum (λ i, |a i - b i|)) % 10 = 9 :=
by
  sorry

end sum_abs_diff_mod_10_l101_101713


namespace triangle_side_length_l101_101149

noncomputable def triangle_BC_length (A B C : Type) [EuclideanGeometry A] (angle_A : A) (side_AB : B) (area_ABC : C) : ℝ :=
  let ∠A := angle_A
  let AB := side_AB
  let area := area_ABC
  sorry

theorem triangle_side_length (A B C : Type) [EuclideanPlane A B C]
  (angle_A : A) (AB_length : ℝ) (area_ABC : ℝ) (BC_length : ℝ) :
  angle_A = 60 ∧ AB_length = 2 ∧ area_ABC = sqrt(3)/2 → BC_length = sqrt(3) :=
by
  intro h
  cases h with h1 h
  cases h with h2 h3
  have h_angle_A : angle_A = 60 := h1
  have h_AB_length : AB_length = 2 := h2
  have h_area_ABC : area_ABC = sqrt(3) / 2 := h3
  -- Placeholder for the proof
  sorry

end triangle_side_length_l101_101149


namespace min_sum_abs_elements_l101_101176

theorem min_sum_abs_elements (a b c d : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h : matrix.mul (matrix.from_blocks (matrix.scalar 2 1) (matrix.scalar 2 0) (matrix.scalar 2 0) (matrix.scalar 2 1)) 
  (matrix.from_blocks (matrix.scalar 2 1) (matrix.scalar 2 0) (matrix.scalar 2 0) (matrix.scalar 2 1)) 
  = matrix.from_blocks (matrix.scalar 2 12) (matrix.scalar 2 0) (matrix.scalar 2 0) (matrix.scalar 2 12)) :
  |a| + |b| + |c| + |d| = 10 := 
by
  sorry

end min_sum_abs_elements_l101_101176


namespace tan_theta_solution_l101_101108

theorem tan_theta_solution (θ : ℝ)
  (h : 2 * Real.sin (θ + Real.pi / 3) = 3 * Real.sin (Real.pi / 3 - θ)) :
  Real.tan θ = Real.sqrt 3 / 5 := sorry

end tan_theta_solution_l101_101108


namespace max_districts_in_park_l101_101720

theorem max_districts_in_park :
  ∀ (side_length park_side : ℕ) (district_length district_width : ℕ), 
    side_length = 14 →
    district_length = 8 →
    district_width = 2 →
    let park_area := side_length * park_side in 
    let district_area := district_length * district_width in 
    park_area / district_area = 12 :=
by
  intros side_length park_side district_length district_width h_side_length h_district_length h_district_width
  simp [h_side_length, h_district_length, h_district_width]
  let park_area := side_length * side_length
  let district_area := district_length * district_width
  suffices : park_area = 196 ∧ district_area = 16, sorry
  sorry

end max_districts_in_park_l101_101720


namespace jackson_final_grade_l101_101634

def jackson_hours_playing_video_games : ℕ := 9

def ratio_study_to_play : ℚ := 1 / 3

def time_spent_studying (hours_playing : ℕ) (ratio : ℚ) : ℚ := hours_playing * ratio

def points_per_hour_studying : ℕ := 15

def jackson_grade (time_studied : ℚ) (points_per_hour : ℕ) : ℚ := time_studied * points_per_hour

theorem jackson_final_grade :
  jackson_grade
    (time_spent_studying jackson_hours_playing_video_games ratio_study_to_play)
    points_per_hour_studying = 45 :=
by
  sorry

end jackson_final_grade_l101_101634


namespace abs_diff_61st_term_l101_101746

-- Define sequences C and D
def seqC (n : ℕ) : ℤ := 20 + 15 * (n - 1)
def seqD (n : ℕ) : ℤ := 20 - 15 * (n - 1)

-- Prove the absolute value of the difference between the 61st terms is 1800
theorem abs_diff_61st_term : (abs (seqC 61 - seqD 61) = 1800) :=
by
  sorry

end abs_diff_61st_term_l101_101746


namespace simplify_complex_fraction_l101_101681

open Complex

theorem simplify_complex_fraction :
  (3 + 8 * Complex.i) / (1 - 4 * Complex.i) = 
  - (29 : ℂ) / 17 + (20 : ℂ) / 17 * Complex.i :=
by
  sorry

end simplify_complex_fraction_l101_101681


namespace no_three_digit_number_exists_l101_101877

theorem no_three_digit_number_exists (a b c : ℕ) (h₁ : 0 ≤ a ∧ a < 10) (h₂ : 0 ≤ b ∧ b < 10) (h₃ : 0 ≤ c ∧ c < 10) (h₄ : a ≠ 0) :
  ¬ ∃ k : ℕ, k^2 = 99 * (a - c) :=
by
  sorry

end no_three_digit_number_exists_l101_101877


namespace triangle_cosine_sine_inequality_l101_101985

theorem triangle_cosine_sine_inequality (A B C : ℝ) (h : A + B + C = Real.pi) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hA_lt_pi : A < Real.pi)
  (hB_lt_pi : B < Real.pi)
  (hC_lt_pi : C < Real.pi) :
  Real.cos A * (Real.sin B + Real.sin C) ≥ -2 * Real.sqrt 6 / 9 := 
by
  sorry

end triangle_cosine_sine_inequality_l101_101985


namespace furniture_cost_correct_l101_101486

-- Define the initial amount Emma got
def initial_amount : ℝ := 2000

-- Define the amount Emma has left after giving 3/4 of the remaining money to her friend
def amount_left_after_giving (remaining : ℝ) : ℝ := (1 / 4) * remaining

-- Define the amount Emma kept for herself
def amount_kept : ℝ := 400

-- Define the cost of the furniture
def furniture_cost : ℝ := initial_amount - (amount_kept * 4)

-- Problem statement: Prove that the cost of the furniture is $400 given the conditions
theorem furniture_cost_correct : furniture_cost = 400 :=
by
  sorry

end furniture_cost_correct_l101_101486


namespace probability_longest_segment_at_least_z_times_l101_101360

theorem probability_longest_segment_at_least_z_times
  (circumference : ℝ) (cuts : Fin 2 → ℝ) (z : ℝ) 
  (h_circumference : circumference = 1)
  (h_cuts_range : ∀ i, 0 ≤ cuts i ∧ cuts i ≤ circumference)
  (h_cuts_diff : cuts 0 ≠ cuts 1)
  (h_z_nonneg : z > 0) :
  let A := min (cuts 0) (min (cuts 1) (circumference - cuts 0 - cuts 1)) in
  let B := min (abs (cuts 0 - cuts 1)) (min (circumference - cuts 0) (circumference - cuts 1)) in
  let C := circumference - A - B in
  (∃ i, i = 0 ∨ i = 1 ∨ i = 2) ∧ (A + B + C = circumference) ∧
  ((A ≥ z * B ∧ A ≥ z * C) ∨ (B ≥ z * A ∧ B ≥ z * C) ∨ (C ≥ z * A ∧ C ≥ z * B)) →
  (1/3) * (circumference / z + 1) = (3 / (z + 1)) :=
sorry

end probability_longest_segment_at_least_z_times_l101_101360


namespace number_of_intersections_l101_101056

def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x < 2 then x^3 - x else f(x - 2)

def is_root (f : ℝ → ℝ) (x : ℝ) : Prop := f(x) = 0

open Set

theorem number_of_intersections :
  ∀ (f : ℝ → ℝ), (∀ x, f (x + 2) = f x) →
  (∀ x, 0 ≤ x ∧ x < 2 → f x = x^3 - x) →
  count (λ x, is_root f x ∧ (0 ≤ x ∧ x ≤ 6)) 7 := sorry

end number_of_intersections_l101_101056


namespace hyperbola_equation_l101_101559

theorem hyperbola_equation (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
    (c_focus : c = 2) (eccentricity : e = 2) 
    (hyp_eq : ∀ x y, (x^2) / (a^2) - (y^2) / (b^2) = 1) 
    (shared_focus : (2, 0) is a focus of the hyperbola) :
  x^2 - (y^2 / 3) = 1 :=
sorry

end hyperbola_equation_l101_101559


namespace pencil_arrangements_count_l101_101744

theorem pencil_arrangements_count:
  ∃ (n : Nat), n = 132 ∧ (∃ (pencils : List ℕ), 
    pencils.length = 12 ∧ 
    (∃ (rows : List (List ℕ)), 
      rows.length = 2 ∧
      ∀ row ∈ rows, row.length = 6 ∧ 
      ∀ row ∈ rows, List.pairwise (>) row ∧ 
      ∀ (p2 ∈ rows.head!) (p1 ∈ rows[1]), p2 > p1)) :=
sorry

end pencil_arrangements_count_l101_101744


namespace complementary_angle_decrease_l101_101242

theorem complementary_angle_decrease :
  (ratio : ℚ := 3 / 7) →
  let total_angle := 90
  let small_angle := (ratio * total_angle) / (1+ratio)
  let large_angle := total_angle - small_angle
  let new_small_angle := small_angle * 1.2
  let new_large_angle := total_angle - new_small_angle
  let decrease_percent := (large_angle - new_large_angle) / large_angle * 100
  decrease_percent = 8.57 :=
by
  sorry

end complementary_angle_decrease_l101_101242


namespace vector_magnitude_sum_l101_101536

open Real

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (angle_ab : ℝ)
variables (norm_b : ℝ)

def vector_a : EuclideanSpace ℝ (Fin 2) := ![1, sqrt 3]

theorem vector_magnitude_sum :
  angle_ab = π * (2 / 3) →
  b.norm = 1 →
  ∥vector_a + b∥ = sqrt 3 :=
by
  intro h_angle h_b_norm
  -- statements regarding the norms and dot products can be defined here
  sorry

end vector_magnitude_sum_l101_101536


namespace ship_lighthouse_distance_l101_101380

-- Definitions for conditions
def speed : ℝ := 15 -- speed of the ship in km/h
def time : ℝ := 4  -- time the ship sails eastward in hours
def angle_A : ℝ := 60 -- angle at point A in degrees
def angle_C : ℝ := 30 -- angle at point C in degrees

-- Main theorem statement
theorem ship_lighthouse_distance (d_A_C : ℝ) (d_C_B : ℝ) : d_A_C = speed * time → d_C_B = 60 := 
by sorry

end ship_lighthouse_distance_l101_101380


namespace point_C_coordinates_line_MN_equation_area_triangle_ABC_l101_101981

-- Define the points A and B
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (7, 3)

-- Let C be an unknown point that we need to determine
variables (x y : ℝ)

-- Define the conditions given in the problem
axiom midpoint_M : (x + 5) / 2 = 0 ∧ (y + 3) / 2 = 0 -- Midpoint M lies on the y-axis
axiom midpoint_N : (x + 7) / 2 = 1 ∧ (y + 3) / 2 = 0 -- Midpoint N lies on the x-axis

-- The problem consists of proving three assertions
theorem point_C_coordinates :
  ∃ (x y : ℝ), (x, y) = (-5, -3) :=
by
  sorry

theorem line_MN_equation :
  ∃ (a b c : ℝ), a = 5 ∧ b = -2 ∧ c = -5 :=
by
  sorry

theorem area_triangle_ABC :
  ∃ (S : ℝ), S = 841 / 20 :=
by
  sorry

end point_C_coordinates_line_MN_equation_area_triangle_ABC_l101_101981


namespace number_of_intersections_l101_101057

def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x < 2 then x^3 - x else f(x - 2)

def is_root (f : ℝ → ℝ) (x : ℝ) : Prop := f(x) = 0

open Set

theorem number_of_intersections :
  ∀ (f : ℝ → ℝ), (∀ x, f (x + 2) = f x) →
  (∀ x, 0 ≤ x ∧ x < 2 → f x = x^3 - x) →
  count (λ x, is_root f x ∧ (0 ≤ x ∧ x ≤ 6)) 7 := sorry

end number_of_intersections_l101_101057


namespace horner_correct_l101_101295

variable {α : Type*} [Field α]

def horner (a : List α) (x : α) : α :=
  a.foldr (λ a_i acc, acc * x + a_i) 0

theorem horner_correct (a : List α) (n : ℕ) (x_0 : α) :
  n = a.length - 1 → 
  (let f (x : α) := List.sum (List.map (λ i, a[n-i] * x^i) (List.range (n + 1)))) 
  (∀ k, k ≤ n → 
  (let rec_seq := List.mapWithIndex (λ i v, if i = 0 then a[n] else v * x_0 + a[n - i]) (List.range (n + 1))) 
  (rec_seq.k = horner a x_0)) :=
begin
  sorry
end

end horner_correct_l101_101295


namespace deck_total_cost_is_32_l101_101286

-- Define the costs of each type of card
def rare_cost := 1
def uncommon_cost := 0.50
def common_cost := 0.25

-- Define the quantities of each type of card in Tom's deck
def rare_quantity := 19
def uncommon_quantity := 11
def common_quantity := 30

-- Define the total cost calculation
def total_cost : ℝ := (rare_quantity * rare_cost) + (uncommon_quantity * uncommon_cost) + (common_quantity * common_cost)

-- Prove that the total cost of the deck is $32
theorem deck_total_cost_is_32 : total_cost = 32 := by
  sorry

end deck_total_cost_is_32_l101_101286


namespace oil_bill_january_l101_101775

theorem oil_bill_january 
  (F J : ℝ) 
  (h1 : 2 * F = 3 * J) 
  (h2 : 3 * (F + 10) = 5 * J) : 
  J = 60 := 
begin
  sorry,
end

end oil_bill_january_l101_101775


namespace find_constants_l101_101019

theorem find_constants (t s : ℤ) :
  (∀ x : ℤ, (3 * x^2 - 4 * x + 9) * (5 * x^2 + t * x + s) = 15 * x^4 - 22 * x^3 + (41 + s) * x^2 - 34 * x + 9 * s) →
  t = -2 ∧ s = s :=
by
  intros h
  sorry

end find_constants_l101_101019


namespace conjugate_of_z_l101_101507

open Complex

theorem conjugate_of_z
  (z : ℂ)
  (h : z * (2 + I) = 3 + I) :
  conj z = (7 / 5) + ((1 / 5) * I) :=
sorry

end conjugate_of_z_l101_101507


namespace find_a_b_find_extreme_point_g_num_zeros_h_l101_101530

-- (1) Proving the values of a and b
theorem find_a_b (a b : ℝ)
  (h1 : (3 + 2 * a + b = 0))
  (h2 : (3 - 2 * a + b = 0)) : 
  a = 0 ∧ b = -3 :=
sorry

-- (2) Proving the extreme points of g(x)
theorem find_extreme_point_g (x : ℝ) : 
  x = -2 :=
sorry

-- (3) Proving the number of zeros of h(x)
theorem num_zeros_h (c : ℝ) (h : -2 ≤ c ∧ c ≤ 2) :
  (|c| = 2 → ∃ y, y = 5) ∧ (|c| < 2 → ∃ y, y = 9) :=
sorry

end find_a_b_find_extreme_point_g_num_zeros_h_l101_101530


namespace solve_for_x_l101_101780

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else -2 * x

theorem solve_for_x (x : ℝ) (h : f x = 10) : x = -3 :=
sorry

end solve_for_x_l101_101780


namespace polynomial_degree_expression_l101_101931

noncomputable def degree (p : polynomial ℝ) : ℕ := p.nat_degree

theorem polynomial_degree_expression :
  (∀ A B C : polynomial ℝ, degree A = 672 ∧ degree B = 672 ∧ degree C = 671 →
  ((let m := degree (A + B) in
    let n := degree (A - C) in
    |m - n| + |2 * m - n - 672| + |-3 * m - 2| = 2018))) :=
by {
  -- Define polynomials A, B, and C
  intro A,
  intro B,
  intro C,
  -- Define the conditions
  intros h,
  cases h with h1 h',
  cases h' with h2 h3,
  -- Define m and n
  let m := degree (A + B),
  let n := degree (A - C),
  -- Begin the proof
  sorry
}

end polynomial_degree_expression_l101_101931


namespace find_a_range_l101_101945

noncomputable def f (a : ℝ) : ℝ → ℝ := 
  λ x, if x > 2 then a * x ^ 2 + x - 1 else -x + 1

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≥ f y

theorem find_a_range (a : ℝ) :
  is_monotonically_decreasing (f a) → a ≤ -1/2 :=
begin
  sorry
end

end find_a_range_l101_101945


namespace number_of_families_l101_101614

theorem number_of_families (x : ℕ) (h1 : x + x / 3 = 100) : x = 75 :=
sorry

end number_of_families_l101_101614


namespace sum_of_geometric_sequence_first_9000_terms_l101_101251

noncomputable def geomSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_of_geometric_sequence_first_9000_terms (a r : ℝ) (h1 : geomSum a r 3000 = 500) (h2 : geomSum a r 6000 = 950) :
  geomSum a r 9000 = 1355 :=
by
  sorry

end sum_of_geometric_sequence_first_9000_terms_l101_101251


namespace polynomial_remainder_l101_101641

def Q (x : ℝ) : ℝ := sorry -- Define Q(x) as some polynomial (unknown)

-- Conditions given in the problem
def cond1 : Prop := ∃ R : ℝ → ℝ, Q(x) = (x - 15) * R(x) + 8
def cond2 : Prop := ∃ S : ℝ → ℝ, Q(x) = (x - 10) * S(x) + 3

-- Question: the remainder when Q(x) is divided by (x-10)(x-15)
theorem polynomial_remainder : cond1 ∧ cond2 → ∃ R : ℝ → ℝ, Q(x) = (x - 10) * (x - 15) * R(x) + (x - 7) := 
by
  sorry

end polynomial_remainder_l101_101641


namespace not_factorial_tails_count_l101_101459

def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) -- you can extend further if needed

def is_factorial_tail (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ f(m) = n

theorem not_factorial_tails_count : 
  ∃ c : ℕ, c = 500 ∧ ∀ k : ℕ, k < 2500 → ¬is_factorial_tail k → k < 2500 - c :=
by
  sorry

end not_factorial_tails_count_l101_101459


namespace correct_sum_of_integers_l101_101190

theorem correct_sum_of_integers (a b : ℕ) (h1 : a - b = 4) (h2 : a * b = 63) : a + b = 18 := 
  sorry

end correct_sum_of_integers_l101_101190


namespace no_eulerian_path_l101_101508

-- Definitions for conditions
variables (A B C D E F H: Type) [fintype A] [fintype B] [fintype C] [fintype D] [fintype E] [fintype F] [fintype H]

-- The degrees of each region (represented mathematically)
noncomputable def degree_A : ℕ := 5
noncomputable def degree_B : ℕ := 5
noncomputable def degree_C : ℕ := 4
noncomputable def degree_D : ℕ := 5
noncomputable def degree_E : ℕ := 4
noncomputable def degree_F : ℕ := 4
noncomputable def degree_H : ℕ := 4

-- The proof statement
theorem no_eulerian_path :
  ¬(∃ (curve : A → C), (∀ (seg : B → D), curve seg ∧ curve seg) = true ∧
  ¬(curve seg = A ∨ curve seg = D)) :=
begin
  sorry -- skipping proof steps as per instruction
end

end no_eulerian_path_l101_101508


namespace volume_of_sphere_l101_101980

theorem volume_of_sphere (R : ℝ) (h : 4 * Real.pi * R ^ 2 = 4 * Real.pi) : 
  (4 / 3) * Real.pi * R ^ 3 = (4 / 3) * Real.pi :=
by
  have h1 : R ^ 2 = 1 := by
    rw [mul_assoc] at h
    exact (eq_div_iff (4 * Real.pi ≠ 0 _)).mp h
  have h2 : R = 1 := by
    exact eq_of_sq_eq_sq (Real.sqrt_nonneg _) h1
  rw [h2]
  exact rfl

end volume_of_sphere_l101_101980


namespace determine_f1_l101_101935

-- Definitions
def func (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f(x + y) = f(x) + f(y)

-- Theorem statement
theorem determine_f1 (f : ℝ → ℝ) (h1 : func f) (h2 : f 2 = 4) : f 1 = 2 := 
by 
  sorry

end determine_f1_l101_101935


namespace total_selection_schemes_l101_101144

-- Definitions of individuals and tasks
inductive Person
| XiaoZhang
| XiaoZhao
| XiaoLi
| XiaoLuo
| XiaoWang

inductive Task
| Translation
| TourGuiding
| Etiquette
| Driving

-- Function to check if a person can perform a task
def canPerform : Person → Task → Prop
| Person.XiaoZhang, Task.Translation => true
| Person.XiaoZhang, Task.TourGuiding => true
| Person.XiaoZhao, Task.Translation => true
| Person.XiaoZhao, Task.TourGuiding => true
| _, _ => true

-- Lean 4 statement for the problem
theorem total_selection_schemes : 
  (∃ (P1 P2 P3 P4 : Person) (T1 T2 T3 T4 : Task),
    canPerform P1 T1 ∧ canPerform P2 T2 ∧ canPerform P3 T3 ∧ canPerform P4 T4 ∧
    P1 ≠ P2 ∧ P1 ≠ P3 ∧ P1 ≠ P4 ∧
    P2 ≠ P3 ∧ P2 ≠ P4 ∧ P3 ≠ P4 ∧
    T1 ≠ T2 ∧ T1 ≠ T3 ∧ T1 ≠ T4 ∧
    T2 ≠ T3 ∧ T2 ≠ T4 ∧ T3 ≠ T4) = 36 
 := sorry

end total_selection_schemes_l101_101144


namespace probability_one_absent_two_present_l101_101127

theorem probability_one_absent_two_present : 
  let p_absent := 1 / 15 
  let p_present := 1 - p_absent
  let p_exactly_one_absent := 3 * (p_absent * p_present * p_present)
  (p_exactly_one_absent * 100 : ℝ) ≈ 17.4 :=
by
  let p_absent := 1 / 15
  let p_present := 1 - p_absent
  let p_exactly_one_absent := 3 * (p_absent * p_present * p_present)
  have : (p_exactly_one_absent * 100 : ℝ) ≈ 17.4, from sorry
  exact this

end probability_one_absent_two_present_l101_101127


namespace length_hypotenuse_QR_l101_101142

-- Definitions and conditions based on the problem statement
variables (a b : ℝ)
def PS := a / 4
def SQ := 3 * a / 4
def PT := b / 4
def TR := 3 * b / 4
def QT := 20
def SR := 35

-- Statements of the conditions translated to Lean
def condition1 := (PS ^ 2 + b ^ 2 = SR ^ 2)
def condition2 := (a ^ 2 + PT ^ 2 = QT ^ 2)

-- The theorem statement to be proven
theorem length_hypotenuse_QR (h1 : condition1 a b) (h2 : condition2 a b) : 
  a ^ 2 + b ^ 2 = 1529.41 :=
sorry

end length_hypotenuse_QR_l101_101142


namespace total_workers_is_22_l101_101217

-- Defining the average salaries and the related conditions
def average_salary_all : ℝ := 850
def average_salary_technicians : ℝ := 1000
def average_salary_rest : ℝ := 780

-- Given number of technicians
def T : ℕ := 7

-- Total number of workers in the workshop
def total_number_of_workers : ℕ :=
  let R := 15 in
  7 + R

-- Total number of workers proof
theorem total_workers_is_22 : total_number_of_workers = 22 :=
by
  -- Calculation to be filled in proof
  sorry

end total_workers_is_22_l101_101217


namespace find_function_expression_l101_101944

theorem find_function_expression
  (A ω ϕ k : ℝ)
  (hA : A > 0)
  (hω : ω > 0)
  (hϕ : |ϕ| < π / 2)
  (h_high : ∃ x, x = 2 ∧ f x = 2)
  (h_low : ∃ x, x = 8 ∧ f x = -4)
  (h_A : A = (2 - (-4)) / 2)
  (h_k : k = (2 + (-4)) / 2)
  (h_ω : ω = π / 6)
  (h_ϕ : ϕ = π / 6) :
  ∀ x, f x = A * sin (ω * x + ϕ) + k :=
by
  sorry

def f (x : ℝ) : ℝ :=
  3 * sin (π / 6 * x + π / 6) - 1

end find_function_expression_l101_101944


namespace problem1_problem2_l101_101145

-- Problem 1
theorem problem1 (m n : ℤ) (A B : ℤ × ℤ) (hA : A = (3, 2 * m - 1)) (hB : B = (n + 1, -1)) :
  (A.1 = B.1) → n = 2 :=
by
  -- sorry, proof goes here

-- Problem 2
theorem problem2 (m n : ℤ) (A B : ℤ × ℤ) (hA : A = (3, 2 * m - 1)) (hB : B = (n + 1, -1)) :
  (A.2 + 2 = B.2) ∧ (A.1 - 3 = B.1) → m = -1 ∧ n = -1 :=
by
  -- sorry, proof goes here

end problem1_problem2_l101_101145


namespace min_product_eq_neg480_l101_101304

open Fin

def min_possible_product_of_three : Set ℕ :=
  {-10, -5, -3, 0, 2, 4, 6, 8}

theorem min_product_eq_neg480 :
  ∃ a b c ∈ min_possible_product_of_three, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = -480 :=
by
  let s := min_possible_product_of_three
  have hs : Set.Finite s := Set.finite_of_fintype _
  use -10, 8, 6
  have ha : -10 ∈ s := Set.mem_singleton (-10)
  have hb : 8 ∈ s := Set.mem_singleton 8
  have hc : 6 ∈ s := Set.mem_singleton 6
  exact ⟨
    λ h, Set.mem_of_mem_of_subset h ha,
    λ h, Set.mem_of_mem_of_subset h hb,
    λ h, Set.mem_of_mem_of_subset h hc,
    (ne_of_mem_of_not_mem hb ha).2,
    (ne_of_mem_of_not_mem hc hb).2,
    (ne_of_mem_of_not_mem hc ha).2,
    by refl ⟩
  sorry

end min_product_eq_neg480_l101_101304


namespace value_of_5_S_3_l101_101864

def operation_S (a b : ℝ) : ℝ := 4 * a + 6 * b - 2 * a * b

theorem value_of_5_S_3 : operation_S 5 3 = 8 :=
by
  sorry

end value_of_5_S_3_l101_101864


namespace graveling_cost_is_correct_l101_101335

noncomputable def cost_of_graveling (lawn_length : ℕ) (lawn_breadth : ℕ) 
(road_width : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  let area_road_parallel_to_length := road_width * lawn_breadth
  let area_road_parallel_to_breadth := road_width * lawn_length
  let area_overlap := road_width * road_width
  let total_area := area_road_parallel_to_length + area_road_parallel_to_breadth - area_overlap
  total_area * cost_per_sq_m

theorem graveling_cost_is_correct : cost_of_graveling 90 60 10 3 = 4200 := by
  sorry

end graveling_cost_is_correct_l101_101335


namespace boxes_per_class_l101_101274

variable (boxes : ℕ) (classes : ℕ)

theorem boxes_per_class (h1 : boxes = 3) (h2 : classes = 4) : 
  (boxes : ℚ) / (classes : ℚ) = 3 / 4 :=
by
  rw [h1, h2]
  norm_num

end boxes_per_class_l101_101274


namespace imaginary_part_of_z_l101_101231

theorem imaginary_part_of_z (i : ℂ) (hi : i^2 = -1) : (z : ℂ) → z = i^2 * (1 + i) → z.im = -1 :=
by
  intros i hi z hz 
  sorry

end imaginary_part_of_z_l101_101231


namespace max_triangle_area_ellipse_l101_101069

theorem max_triangle_area_ellipse
  (a b : ℝ) (h : a > b) (h_pos : b > 0)
  (focus : ℝ × ℝ) (chord : ℝ × ℝ → ℝ × ℝ → Prop)
  (ellipse : ℝ → ℝ → Prop) :
  (ellipse (x : ℝ) (y: ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1) →
  (focus = (real.sqrt (a^2 - b^2), 0)) →
  (chord A B := A.1 = -B.1 ∧ A.2 = -B.2) →
  ∃ ABF : ℝ,
    ∀ A B : ℝ × ℝ, ellipse A.1 A.2 → ellipse B.1 B.2 → chord A B →
    let dx := A.1 - focus.1,
        dy := A.2 - focus.2
    in ABF ≤ b * real.sqrt(a^2 - b^2) :=
sorry

end max_triangle_area_ellipse_l101_101069


namespace ellipse_max_OM_l101_101517

theorem ellipse_max_OM :
  ∀ (a b : ℝ),
    a > b ∧ b > 0 ∧
    (c : ℝ) (h_c : c / a = sqrt 3 / 2) ∧
    b = 1 ∧
    ∃ M : ℝ × ℝ, is_midpoint M
where is_midpoint M :=
  let (M_x, M_y) := M in
  let O := (0, 0) in
  ∀ (A B : ℝ × ℝ),
    is_tangent_tangent A B ∧
    midpoint_of A B = M ∧
    (O_first : M_x ^ 2 + M_y ^ 2 ≤ 25 / 16)

end ellipse_max_OM_l101_101517


namespace find_original_number_l101_101800

open Int

theorem find_original_number (N y x : ℕ) (h1 : N = 10 * y + x) (h2 : N + y = 54321) (h3 : x = 54321 % 11) (h4 : y = 4938) : N = 49383 := by
  sorry

end find_original_number_l101_101800


namespace find_m_l101_101541

theorem find_m (x m : ℝ) (h1 : 3 * x - 2 * m = 4) (h2 : x = m) : m = 4 :=
sorry

end find_m_l101_101541


namespace unique_poly_degree_4_l101_101740

theorem unique_poly_degree_4 
  (Q : ℚ[X]) 
  (h_deg : Q.degree = 4) 
  (h_lead_coeff : Q.leading_coeff = 1)
  (h_root : Q.eval (sqrt 3 + sqrt 7) = 0) : 
  Q = X^4 - 12 * X^2 + 16 ∧ Q.eval 2 = 0 :=
by sorry

end unique_poly_degree_4_l101_101740


namespace original_number_l101_101812

theorem original_number (N y x : ℕ) 
  (h1: N + y = 54321)
  (h2: N = 10 * y + x)
  (h3: 11 * y + x = 54321)
  (h4: x = 54321 % 11)
  (hy: y = 4938) : 
  N = 49383 := 
  by 
  sorry

end original_number_l101_101812


namespace area_triangle_ABC_l101_101949

def f (x : ℝ) : ℝ := sin (2*x - π/6) + 2*(cos x)^2 - 1

variable (a b c A : ℝ) (h_a : a = 1) (h_bc : b + c = 2) (h_fA : f A = 1/2)

theorem area_triangle_ABC : 
  let cos_A := cos A
  let sin_A := sin A
  let bc := b * c
  let area := 1/2 * bc * sin_A
  A = π/3 ∧ area = sqrt(3)/4 := 
sorry

end area_triangle_ABC_l101_101949


namespace number_divisible_by_396_l101_101896

def is_divisible_by (n d : Nat) : Prop := d ∣ n

def candidates : List Nat := [453420, 413424]

theorem number_divisible_by_396 (n : Nat) : n ∈ candidates → is_divisible_by n 396 :=
by
  intros h
  have div_by_4 : is_divisible_by n 4 :=
    by
      cases h with
      | inl _ => sorry
      | inr _ => sorry
  have div_by_99 : is_divisible_by n 99 :=
    by
      cases h with
      | inl _ => sorry
      | inr _ => sorry
  sorry

end number_divisible_by_396_l101_101896


namespace part1_part2_l101_101078

-- Definitions for Part 1
variables {p : ℝ} (h_pos : p > 0)
def focus : ℝ × ℝ := (0, p / 2)
def parabola (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Definitions for Part 2
def curve_eq (x y : ℝ) : Prop := y = real.sqrt(1 - x^2)
def parabola_spec (x y : ℝ) : Prop := x^2 = 4 * y

-- Part 1: Proving point G lies on a fixed line
theorem part1
  (l : ℝ → ℝ)
  (hl : ∀ x y, l x = y → parabola x y → l x = y)
  (P Q : ℝ × ℝ)
  (hP : parabola P.1 P.2)
  (hQ : parabola Q.1 Q.2)
  (tangent_P tangent_Q : ℝ → ℝ)
  (h_tangent_P : ∀ x, tangent_P x = (x - P.1) * P.2 / P.1 + P.2)
  (h_tangent_Q : ∀ x, tangent_Q x = (x - Q.1) * Q.2 / Q.1 + Q.2)
  (G : ℝ × ℝ)
  (hG : ∀ x, tangent_P x = tangent_Q x → G = (x, tangent_P x)) :
  G.2 = (-p / 2) :=
sorry

-- Part 2: Finding the range of values for the area of triangle MPQ
theorem part2
  (p_eq : p = 2)
  (M : ℝ × ℝ)
  (hM : curve_eq M.1 M.2)
  (mid_MP mid_MQ : ℝ × ℝ)
  (h_mid_MP : parabola_spec mid_MP.1 mid_MP.2)
  (h_mid_MQ : parabola_spec mid_MQ.1 mid_MQ.2)
  (area_MPQ : ℝ)
  (h_area : area_MPQ = (3 * real.sqrt 2 / 4) * (M.1^2 - 4 * M.2)^3 / 2) :
  (area_MPQ ∈ set.Icc (3 * real.sqrt 2 / 4) (6 * real.sqrt 2)) :=
sorry

end part1_part2_l101_101078


namespace problem_statement_l101_101652

open Complex

theorem problem_statement (x y : ℂ) (h : (x + y) / (x - y) - (3 * (x - y)) / (x + y) = 2) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 8320 / 4095 := 
by 
  sorry

end problem_statement_l101_101652


namespace coefficient_of_x4_l101_101620

theorem coefficient_of_x4 : 
  (Finset.sum (Finset.range 6) 
    (λ r, (Nat.choose 5 r) * (2:ℕ)^(5-r) * (-1:ℤ)^r * (x:ℤ)^(10 - 3 * r))) = 80 * x^4 :=
  sorry

end coefficient_of_x4_l101_101620


namespace benjamin_distance_l101_101411

def speed := 10  -- Speed in kilometers per hour
def time := 8    -- Time in hours

def distance (s t : ℕ) := s * t  -- Distance formula

theorem benjamin_distance : distance speed time = 80 :=
by
  -- proof omitted
  sorry

end benjamin_distance_l101_101411


namespace melanie_balloons_l101_101635

theorem melanie_balloons (joan_balloons melanie_balloons total_balloons : ℕ)
  (h_joan : joan_balloons = 40)
  (h_total : total_balloons = 81) :
  melanie_balloons = total_balloons - joan_balloons :=
by
  sorry

end melanie_balloons_l101_101635


namespace evaluate_f_f_f_1_l101_101659

def f (x : ℝ) : ℝ :=
  if x >= 3 then x^2 + 1 else real.sqrt (x + 1)

theorem evaluate_f_f_f_1 : f (f (f 1)) = real.sqrt (real.sqrt (real.sqrt 2 + 1) + 1) :=
by sorry

end evaluate_f_f_f_1_l101_101659


namespace find_DY_length_l101_101604

noncomputable def angle_bisector_theorem (DE DY EF FY : ℝ) : ℝ :=
  (DE * FY) / EF

theorem find_DY_length :
  ∀ (DE EF FY : ℝ), DE = 26 → EF = 34 → FY = 30 →
  angle_bisector_theorem DE DY EF FY = 22.94 := 
by
  intros
  sorry

end find_DY_length_l101_101604


namespace find_mnp_l101_101064

noncomputable theory

def perpendicular_lines (m n : ℝ) : Prop :=
  (m / -4) * (2 / -5) = -1

def foot_of_perpendicular (m p : ℝ) : Prop :=
  (10 * 1 + 4 * p - 2 = 0)

theorem find_mnp (m n p : ℝ) 
  (h1 : perpendicular_lines m n)
  (h2 : foot_of_perpendicular m p)
  (h3 : 2 * 1 - 5 * p + n = 0) : 
  m + n - p = 0 := 
by 
  sorry

end find_mnp_l101_101064


namespace count_multiples_of_7_with_units_digit_7_less_than_150_l101_101087

-- Definition of the problem's conditions and expected result
theorem count_multiples_of_7_with_units_digit_7_less_than_150 : 
  let multiples_of_7 := { n : Nat // n < 150 ∧ n % 7 = 0 }
  let units_digit_7_multiples := { n : Nat // n < 150 ∧ n % 7 = 0 ∧ n % 10 = 7 }
  List.length (units_digit_7_multiples.members) = 3 :=
by
  sorry

end count_multiples_of_7_with_units_digit_7_less_than_150_l101_101087


namespace unique_poly_degree_4_l101_101738

theorem unique_poly_degree_4 
  (Q : ℚ[X]) 
  (h_deg : Q.degree = 4) 
  (h_lead_coeff : Q.leading_coeff = 1)
  (h_root : Q.eval (sqrt 3 + sqrt 7) = 0) : 
  Q = X^4 - 12 * X^2 + 16 ∧ Q.eval 2 = 0 :=
by sorry

end unique_poly_degree_4_l101_101738


namespace cuboid_point_distance_l101_101591

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem cuboid_point_distance 
  (points : Fin 2001 → ℝ × ℝ × ℝ) 
  (h : ∀ i, (points i).1 ≥ 0 ∧ (points i).1 ≤ 5 ∧
             (points i).2 ≥ 0 ∧ (points i).2 ≤ 5 ∧
             (points i).3 ≥ 0 ∧ (points i).3 ≤ 10) :
  ∃ (i j : Fin 2001), i ≠ j ∧ distance (points i) (points j) < 0.7 := 
sorry

end cuboid_point_distance_l101_101591


namespace quadrilateral_possible_with_2_2_2_l101_101770

theorem quadrilateral_possible_with_2_2_2 :
  ∀ (s1 s2 s3 s4 : ℕ), (s1 = 2) → (s2 = 2) → (s3 = 2) → (s4 = 5) →
  s1 + s2 + s3 > s4 :=
by
  intros s1 s2 s3 s4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Proof omitted
  sorry

end quadrilateral_possible_with_2_2_2_l101_101770


namespace isosceles_triangle_DEF_area_l101_101492

noncomputable def triangle_area {α : Type*} [linear_ordered_field α] 
  (a b c : α) (A : real.angle) : α :=
1/2 * a * b * (real.sin A)

noncomputable def DEF_area : real := triangle_area 5 5 (real.angle.of_deg 120)

theorem isosceles_triangle_DEF_area :
  DEF_area = 250 / 9 :=
by 
  sorry

end isosceles_triangle_DEF_area_l101_101492


namespace num_not_factorial_tails_lt_2500_l101_101470

-- Definition of the function f(m)
def f (m : ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625 + m / 78125 + m / 390625 + m / 1953125

-- The theorem stating the main result
theorem num_not_factorial_tails_lt_2500 : 
  let count_non_tails := ∑ k in finset.range 2500, if ∀ m, f m ≠ k then 1 else 0
  in count_non_tails = 500 :=
by
  sorry

end num_not_factorial_tails_lt_2500_l101_101470


namespace ellipse_problem_l101_101915

noncomputable def ellipse_equation (x y a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_problem :
  (∀ x y : ℝ, ellipse_equation x y 2 1 ↔ (x^2 / 4 + y^2 = 1))
  ∧
  (∃ m : ℝ, m = 17 / 8 ∧ ∀ k : ℝ, let x1, x2 := 1, 1,
    let y1, y2 := √3 / 2, -√3 / 2 in
    let PE := (m - x1, -y1) in
    let QE := (m - x2, -y2) in
    (PE.1 * QE.1 + PE.2 * QE.2) = 33 / 64) := 
sorry

end ellipse_problem_l101_101915


namespace maximum_value_of_ab_l101_101566

noncomputable def max_ab : ℝ :=
  let y1 := λ x : ℝ, x^2 - 2 * x + 2
  let y2 := λ x a b : ℝ, -x^2 + a * x + b
  let dy1 := λ x : ℝ, 2 * x - 2
  let dy2 := λ x a : ℝ, -2 * x + a
  let f := λ a b : ℝ, a + b
  max f {ab : ℝ // ∃ (a b x0 : ℝ), a > 0 ∧ b > 0 ∧
    (dy1 x0) * (dy2 x0 a) = -1 ∧
    y1 x0 = y2 x0 a b ∧
    x0 > 0}

theorem maximum_value_of_ab 
  (a b x0 : ℝ) (h : a > 0 ∧ b > 0)
  (inter_tangent_perpendicular : (2 * x0 - 2) * (-2 * x0 + a) = -1)
  (intersection : (x0^2 - 2*x0 + 2) = (-x0^2 + a*x0 + b)) :
  (a + b = 5 / 2) → (ab ≤ 25 / 16) := 
sorry

end maximum_value_of_ab_l101_101566


namespace line_passes_through_point_l101_101232

theorem line_passes_through_point : 
  ∃ k : ℝ, (1 + 4 * k) * 2 - (2 - 3 * k) * 2 + 2 - 14 * k = 0 :=
begin
  sorry
end

end line_passes_through_point_l101_101232


namespace second_square_area_l101_101594

noncomputable def initial_square_side_length := real.sqrt 169

noncomputable def triangle_leg_length := 2 * initial_square_side_length

noncomputable def triangle_hypotenuse := triangle_leg_length * real.sqrt 2

noncomputable def second_square_side_length := triangle_hypotenuse / (1 + real.sqrt 2) * (real.sqrt 2 - 1)

theorem second_square_area :
  second_square_side_length ^ 2 = 676 :=
by sorry

end second_square_area_l101_101594


namespace find_100th_index_neg_b_l101_101865

def sequence_b (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), Real.cos (k : ℝ)

theorem find_100th_index_neg_b :
  ∃ n : ℕ, n = 632 ∧ ∀ k < 100, (sequence_b (n + k)) < 0 := sorry

end find_100th_index_neg_b_l101_101865


namespace tangent_triples_l101_101162

theorem tangent_triples
  (ABC : Triangle)
  (Γ : Circumcircle ABC)
  (ω : Incircle ABC)
  (P Q R : Point)
  (hP : P ∈ Γ)
  (hQ : Q ∈ Γ)
  (hR : R ∈ Γ)
  (hPQ_tangent : Tangent PQ ω)
  (hQR_tangent : Tangent QR ω) :
  Tangent RP ω :=
by sorry

end tangent_triples_l101_101162


namespace greatest_k_value_l101_101712

noncomputable def greatest_possible_k : ℝ :=
  let k_squared := 117 in
  real.sqrt k_squared

theorem greatest_k_value (k : ℝ) (h : ∀ x : ℝ, x^2 + k * x + 8 = 0 → ∃ a b : ℝ, a - b = real.sqrt 85) :
  k = greatest_possible_k :=
by
  sorry

end greatest_k_value_l101_101712


namespace tangent_perpendicular_l101_101545

open Real

def curve (m : ℝ) (x : ℝ) : ℝ := exp x - m * x + 1

theorem tangent_perpendicular (m : ℝ) :
  (∃ x : ℝ, deriv (curve m) x = deriv (λ x, exp x) x ∧ 
    deriv (curve m) x = -1 / deriv (λ x, exp x) x) → 
  m > 1 / exp 1 := by
  sorry

end tangent_perpendicular_l101_101545


namespace water_left_in_cooler_l101_101330

theorem water_left_in_cooler : 
  let gallons_in_cooler := 3 in
  let ounces_per_cup := 6 in
  let rows := 5 in
  let chairs_per_row := 10 in
  let ounces_per_gallon := 128 in
  let initial_ounces := gallons_in_cooler * ounces_per_gallon in
  let total_chairs := rows * chairs_per_row in
  let total_ounces_needed := total_chairs * ounces_per_cup in
  let remaining_ounces := initial_ounces - total_ounces_needed in
  remaining_ounces = 84 :=
by 
  -- introduce variables
  let gallons_in_cooler := 3
  let ounces_per_cup := 6
  let rows := 5
  let chairs_per_row := 10
  let ounces_per_gallon := 128
  let initial_ounces := gallons_in_cooler * ounces_per_gallon
  let total_chairs := rows * chairs_per_row
  let total_ounces_needed := total_chairs * ounces_per_cup
  let remaining_ounces := initial_ounces - total_ounces_needed
  -- prove the theorem
  sorry

end water_left_in_cooler_l101_101330


namespace intersection_of_A_and_B_l101_101960

noncomputable def A : set ℝ := {x | x^2 - 2*x - 3 < 0}
noncomputable def B : set ℝ := {x | (1 - x) / x < 0}

theorem intersection_of_A_and_B : 
  A ∩ B = {x | (-1 < x ∧ x < 0) ∨ (1 < x ∧ x < 3)} :=
by sorry

end intersection_of_A_and_B_l101_101960


namespace vector_line_equation_l101_101025

theorem vector_line_equation {x y : ℝ} (hv : proj ⟨7, 3⟩ ⟨x, y⟩ = ⟨-7/2, -3/2⟩) :
  y = -(7/3 : ℝ) * x - (29/3 : ℝ) :=
sorry

end vector_line_equation_l101_101025


namespace not_factorial_tails_count_l101_101454

def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) -- you can extend further if needed

def is_factorial_tail (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ f(m) = n

theorem not_factorial_tails_count : 
  ∃ c : ℕ, c = 500 ∧ ∀ k : ℕ, k < 2500 → ¬is_factorial_tail k → k < 2500 - c :=
by
  sorry

end not_factorial_tails_count_l101_101454


namespace line_equation_l101_101535

def line_through_intersection (l : ℝ → Prop) : Prop :=
  (∃ x y : ℝ, 2 * x + y - 5 = 0 ∧ x - 2 * y = 0 ∧ l x)

def point_distance (l : ℝ → Prop) (P : (ℝ × ℝ)) (d : ℝ) : Prop :=
  ∃ k : ℝ, 
    (l = λ ⟨x, y⟩, k * x - y + 1 - 2 * k = 0) ∧
    abs (5 * k + 1 - 2 * k) / (Real.sqrt (k^2 + 1)) = d

theorem line_equation {l : ℝ → Prop} {P : (ℝ × ℝ)} : 
  line_through_intersection l → point_distance l P 3 → 
  (l = λ ⟨x, y⟩, 4 * x - 3 * y - 5 = 0) ∨ (l = λ ⟨x, y⟩, x - 2 = 0) :=
by sorry

end line_equation_l101_101535


namespace not_factorial_tails_l101_101466

noncomputable def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) + ...

theorem not_factorial_tails :
  let n := 2500 in ∃ (k : ℕ), k = 500 ∧ ∀ m < n, ¬(f m = k) :=
by {
  sorry
}

end not_factorial_tails_l101_101466


namespace multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l101_101096

theorem multiples_of_7_with_units_digit_7 (n : ℕ) : 
  (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) ↔ 
  n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  sorry

theorem number_of_multiples_of_7_with_units_digit_7 : 
  ∃ m, m = 3 ∧ ∀ n : ℕ, (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) → n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  use 3
  intros n
  intros hn
  split
  intro h
  cases h
  sorry

end multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l101_101096


namespace count_non_factorial_tails_lt_2500_l101_101451

def f (m : ℕ) : ℕ := 
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + ... -- Continue as needed

theorem count_non_factorial_tails_lt_2500 : 
  #{ n : ℕ | n < 2500 ∧ ¬ (∃ m : ℕ, f(m) = n) } = 4 :=
by sorry

end count_non_factorial_tails_lt_2500_l101_101451


namespace quadrant_of_z_l101_101479

open Complex

def z := (-8 + I) * I

theorem quadrant_of_z : (Re z < 0) ∧ (Im z < 0) := by
  sorry

end quadrant_of_z_l101_101479


namespace sum_of_first_9000_terms_of_geometric_sequence_l101_101260

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_9000_terms_of_geometric_sequence 
  (a r : ℝ)
  (h₁ : geometric_sum a r 3000 = 500)
  (h₂ : geometric_sum a r 6000 = 950)
  : geometric_sum a r 9000 = 1355 :=
sorry

end sum_of_first_9000_terms_of_geometric_sequence_l101_101260


namespace complement_intersection_l101_101178

open Set

variable (A B U : Set ℕ) 

theorem complement_intersection (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4, 5}) (hU : U = A ∪ B) :
  (U \ A) ∩ B = {4, 5} :=
by sorry

end complement_intersection_l101_101178


namespace incorrect_statements_l101_101817

open List

-- Define each statement as a condition
def Statement_A (locus : Type) [DecidablePred locus] (P : locus → Prop) : Prop :=
  (∀ x, locus x → P x) ∧ (∀ x, ¬locus x → ¬P x)

def Statement_B (locus : Type) [DecidablePred locus] (P : locus → Prop) : Prop :=
  (∀ x, ¬P x → locus x) ∧ (∀ x, locus x → P x)

def Statement_C (locus : Type) [DecidablePred locus] (P : locus → Prop) : Prop :=
  (∀ x, P x → locus x) ∧ (∀ x, ¬locus x → P x)

def Statement_D (locus : Type) [DecidablePred locus] (P : locus → Prop) : Prop :=
  (∀ x, ¬locus x → ¬P x) ∧ (∀ x, ¬P x → ¬locus x)

def Statement_E (locus : Type) [DecidablePred locus] (P : locus → Prop) : Prop :=
  (∀ x, P x → locus x) ∧ (∀ x, ¬P x → ¬locus x)

-- Define the theorem that these statements are incorrect
theorem incorrect_statements (locus : Type) [DecidablePred locus] (P : locus → Prop) : 
  ¬Statement_B locus P ∧ ¬Statement_C locus P :=
by
  sorry

end incorrect_statements_l101_101817


namespace collinear_Q1_Q2_Q3_l101_101544

variables {P₁ P₂ P₃ P₁' P₂' P₃' Q₁ Q₂ Q₃ : Type*}
variables [Point P₁] [Point P₂] [Point P₃] [Point P₁'] [Point P₂'] [Point P₃']
variables [Point Q₁] [Point Q₂] [Point Q₃] [Line P P_diagonal] [Line p]

-- Assume that the points define a complete quadrilateral
variable (complete_quadrilateral : complete_quadrilateral P₁ P₂ P₃ P₁' P₂' P₃')
-- Assume that line p intersects the diagonals P₁ P₁', P₂ P₂', P₃ P₃' at Q₁, Q₂, Q₃
variable (p_intersects_diagonals : intersects P_diagonal p Q₁ Q₂ Q₃)

-- Assume that Q_i' is the fourth harmonic point corresponding to P_i P_i' Q_i
variables {Q₁' Q₂' Q₃' : Type*} [FourthHarmonicPoint P₁ P₁' Q₁ Q₁']
  [FourthHarmonicPoint P₂ P₂' Q₂ Q₂']
  [FourthHarmonicPoint P₃ P₃' Q₃ Q₃']

-- The theorem we want to state
theorem collinear_Q1_Q2_Q3 :
  collinear Q₁ Q₂ Q₃ :=
sorry

end collinear_Q1_Q2_Q3_l101_101544


namespace union_sets_l101_101961

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_sets : M ∪ N = {0, 1, 3, 9} :=
by
  sorry

end union_sets_l101_101961


namespace original_five_digit_number_l101_101808

theorem original_five_digit_number :
  ∃ N y x : ℕ, (N = 10 * y + x) ∧ (N + y = 54321) ∧ (N = 49383) :=
by
  -- The proof script goes here
  sorry

end original_five_digit_number_l101_101808


namespace suzhou_metro_scientific_notation_l101_101689

theorem suzhou_metro_scientific_notation : 
  (∃(a : ℝ) (n : ℤ), 
    1 ≤ abs a ∧ abs a < 10 ∧ 15.6 * 10^9 = a * 10^n) → 
    (a = 1.56 ∧ n = 9) := 
by
  sorry

end suzhou_metro_scientific_notation_l101_101689


namespace sqrt_12_lt_4_l101_101851

theorem sqrt_12_lt_4 : Real.sqrt 12 < 4 := sorry

end sqrt_12_lt_4_l101_101851


namespace sin_cos_difference_eq_sqrt3_div2_l101_101875

theorem sin_cos_difference_eq_sqrt3_div2 :
  sin (40 : ℝ) * cos (20 : ℝ) - cos (220 : ℝ) * sin (20 : ℝ) = (√3 / 2 : ℝ) :=
by
  sorry

end sin_cos_difference_eq_sqrt3_div2_l101_101875


namespace distribute_balls_l101_101968

theorem distribute_balls : 
  ∃ (ways : ℕ), ways = 104 ∧  ∀ (balls : ℕ) (boxes : ℕ), balls = 7 ∧ boxes = 4 → ways = (sum 
      ([
      {balls_alloc := (7, 0, 0, 0); ways := 4},
      {balls_alloc := (6, 1, 0, 0); ways := 12},
      {balls_alloc := (5, 2, 0, 0); ways := 12},
      {balls_alloc := (4, 3, 0, 0); ways := 12},
      {balls_alloc := (4, 2, 1, 0); ways := 24},
      {balls_alloc := (3, 3, 1, 0); ways := 12},
      {balls_alloc := (3, 2, 2, 0); ways := 12},
      {balls_alloc := (3, 2, 1, 1); ways := 12},
      {balls_alloc := (2, 2, 2, 1); ways := 4}
      ])) :=
begin
  use 104,
  split,
  {refl},
  {intros balls boxes h,
   cases h with ball_cond box_cond,
   subst ball_cond,
   subst box_cond,
   sorry}
end

end distribute_balls_l101_101968


namespace simplify_expression_l101_101205

theorem simplify_expression (x : ℝ) (h : x = Real.tan (Float.pi / 3)) :
  ((x + 1 - 8 / (x - 1)) / ((x^2 - 6*x + 9) / (x^2 - x)) * (3 - x)) = -3 - 3 * Real.sqrt 3 :=
by
  -- Proof would be here
  sorry

end simplify_expression_l101_101205


namespace domain_of_g_l101_101477

noncomputable def g (x : ℝ) := log 6 (log 2 (log 3 (log 5 (log 7 x))))

theorem domain_of_g : ∀ x : ℝ, x > 7^78125 → ∃ y : ℝ, g x = y :=
by
  sorry

end domain_of_g_l101_101477


namespace sum_of_first_9000_terms_of_geometric_sequence_l101_101264

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_9000_terms_of_geometric_sequence 
  (a r : ℝ)
  (h₁ : geometric_sum a r 3000 = 500)
  (h₂ : geometric_sum a r 6000 = 950)
  : geometric_sum a r 9000 = 1355 :=
sorry

end sum_of_first_9000_terms_of_geometric_sequence_l101_101264


namespace committee_roles_distribution_l101_101693

theorem committee_roles_distribution (n r : ℕ) (h_n : n = 7) (h_r : r = 2) :
  fintype.card (permutations (fin n)) / (fintype.card (permutations (fin n) // (permutations (fin r)))) = 42 :=
by
  sorry

end committee_roles_distribution_l101_101693


namespace problem1_problem2_problem3_l101_101028

section
variable (f1 f2 F : ℝ → ℝ)
variable (x : ℝ)

/-- 
(1) Problem:
    Determine whether the following functions have a "lower bound".

    (i) For \( f_1(x) = 1 - 2x \), with \( x > 0 \).
    (ii) For \( f_2(x) = x + \frac{16}{x} \), with \( 0 < x ≤ 5 \).
-/
def no_lower_bound (f1 : ℝ → ℝ) := ∀ L : ℝ, ∃ x > 0, f1 x < L
def has_lower_bound (f2 : ℝ → ℝ) (B : ℝ) := ∀ x ∈ set.Icc 0 5, B ≤ f2 x

theorem problem1 (f1 : ℝ → ℝ) (f2 : ℝ → ℝ) :
  (f1 = λ x, 1 - 2 * x) → no_lower_bound f1
  ∧ (f2 = λ x, x + 16 / x) → has_lower_bound f2 8 := sorry

/-- 
(2) Problem:
    (i) Define upper bound for a function \( f(x) \) on an interval \( D \).
    (ii) Prove whether \( f_2(x) = |x - \frac{16}{x}| \) has an upper bound on \( (0, 5] \).
-/
def has_upper_bound (f : ℝ → ℝ) (D : set ℝ) (U : ℝ) := ∃ x0 ∈ D, ∀ x ∈ D, f x ≤ U
def no_upper_bound (f : ℝ → ℝ) := ∀ U : ℝ, ∃ x ∈ set.Ioc 0 5, f x > U

theorem problem2 (f2 : ℝ → ℝ) :
  (∀ x ∈ set.Ioc 0 5, f2 x = abs (x - 16 / x)) → no_upper_bound f2 := sorry

/-- 
(3) Problem:
    Explore whether \( F(x) = x|x - 2a| + 3 \), for \( a ≤ 1/2 \), is bounded on the interval [1, 2]. 

    Deduce the amplitude of it if it is bounded.
-/
def bounded_on_interval (F : ℝ → ℝ) (a : ℝ) := ∃ L U : ℝ, ∀ x ∈ set.Icc 1 2, L ≤ F x ∧ F x ≤ U
def amplitude (F : ℝ → ℝ) (a : ℝ) := ∃ M : ℝ, M = 3 - 2 * a

theorem problem3 (F : ℝ → ℝ) (a : ℝ) :
  (a ≤ 1 / 2) → (F = λ x, x * abs (x - 2 * a) + 3) → bounded_on_interval F a ∧ amplitude F a := sorry

end

end problem1_problem2_problem3_l101_101028


namespace non_factorial_tails_lt_2500_l101_101433

-- Define the function f(m)
def f (m: ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625

-- Define what it means to be a factorial tail
def is_factorial_tail (n: ℕ) : Prop :=
  ∃ m : ℕ, f(m) = n

-- Define the statement to be proved
theorem non_factorial_tails_lt_2500 : 
  (finset.range 2500).filter (λ n, ¬ is_factorial_tail n).card = 499 :=
by
  sorry

end non_factorial_tails_lt_2500_l101_101433


namespace ellipse_properties_l101_101895

-- Definitions used in the statement
noncomputable def length_major_axis (m : ℝ) (h : 0 < m) : ℝ := 2 / m
noncomputable def length_minor_axis (m : ℝ) (h : 0 < m) : ℝ := 1 / m
noncomputable def foci_coords (m : ℝ) (h : 0 < m) : Set (ℝ × ℝ) := { (sqrt 3 / (2 * m), 0), (- sqrt 3 / (2 * m), 0)}
noncomputable def vertex_coords (m : ℝ) (h : 0 < m) : Set (ℝ × ℝ) := { (1 / m, 0), (-1 / m, 0), (0, 1 / (2 * m)), (0, -1 / (2 * m))}
noncomputable def eccentricity (m : ℝ) (h : 0 < m) : ℝ := sqrt 3 / 2

theorem ellipse_properties (m : ℝ) (h : 0 < m) :
  length_major_axis m h = 2 / m ∧
  length_minor_axis m h = 1 / m ∧
  foci_coords m h = { (sqrt 3 / (2 * m), 0), (- sqrt 3 / (2 * m), 0)} ∧
  vertex_coords m h = { (1 / m, 0), (-1 / m, 0), (0, 1 / (2 * m)), (0, -1 / (2 * m))} ∧
  eccentricity m h = sqrt 3 / 2 :=
sorry

end ellipse_properties_l101_101895


namespace smallest_abundant_not_multiple_of_five_l101_101007

def is_proper_divisor (d n : ℕ) : Prop := d < n ∧ n % d = 0

def sum_proper_divisors (n : ℕ) : ℕ :=
  Finset.sum (Finset.filter (λ d => is_proper_divisor d n) (Finset.range n)) (λ d => d)

def is_abundant (n : ℕ) : Prop := sum_proper_divisors n > n

def is_not_multiple_of_five (n : ℕ) : Prop := n % 5 ≠ 0

theorem smallest_abundant_not_multiple_of_five : ∃ n : ℕ, is_abundant n ∧ is_not_multiple_of_five n ∧ n = 12 := by
  sorry

end smallest_abundant_not_multiple_of_five_l101_101007


namespace final_pen_count_l101_101771

theorem final_pen_count
  (initial_pens : ℕ := 7) 
  (mike_given_pens : ℕ := 22) 
  (doubled_pens : ℕ := 2)
  (sharon_given_pens : ℕ := 19) :
  let total_after_mike := initial_pens + mike_given_pens
  let total_after_cindy := total_after_mike * doubled_pens
  let final_count := total_after_cindy - sharon_given_pens
  final_count = 39 :=
by
  sorry

end final_pen_count_l101_101771


namespace find_tangent_line_polar_eq_l101_101533

noncomputable def tangent_line_polar_eq {θ ρ : ℝ} : Prop :=
  ∃ (P : ℝ × ℝ), 
    P = (1 + 2 * (Real.cos θ), sqrt 3 + 2 * (Real.sin θ)) ∧
    (P.1 = 3 ∧ P.2 = sqrt 3) → 
    ρ = 1 / (Real.cos (θ + 60 / 180 * Real.pi))
    
theorem find_tangent_line_polar_eq (θ ρ : ℝ) :
  tangent_line_polar_eq :=
by
  -- Proof step is omitted as instructed
  sorry

end find_tangent_line_polar_eq_l101_101533


namespace count_non_factorial_tails_lt_2500_l101_101452

def f (m : ℕ) : ℕ := 
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + ... -- Continue as needed

theorem count_non_factorial_tails_lt_2500 : 
  #{ n : ℕ | n < 2500 ∧ ¬ (∃ m : ℕ, f(m) = n) } = 4 :=
by sorry

end count_non_factorial_tails_lt_2500_l101_101452


namespace min_value_of_f_l101_101234

def f (x : ℝ) : ℝ := cos x ^ 2 - sin x ^ 2

theorem min_value_of_f : ∃ x : ℝ, f x = -1 := sorry

end min_value_of_f_l101_101234


namespace vince_savings_l101_101298

-- Define constants and conditions
def earnings_per_customer : ℕ := 18
def monthly_expenses : ℕ := 280
def percentage_for_recreation : ℕ := 20
def number_of_customers : ℕ := 80

-- Function to calculate total earnings
def total_earnings (customers : ℕ) (earnings_per_customer : ℕ) : ℕ :=
  customers * earnings_per_customer

-- Function to calculate amount allocated for recreation and relaxation
def recreation_amount (total_earnings : ℕ) (percentage : ℕ) : ℕ :=
  total_earnings * percentage / 100

-- Function to calculate total expenses
def total_monthly_expenses (base_expenses : ℕ) (recreation : ℕ) : ℕ :=
  base_expenses + recreation

-- Function to calculate savings
def savings (total_earnings : ℕ) (total_expenses : ℕ) : ℕ :=
  total_earnings - total_expenses

theorem vince_savings :
  let earnings := total_earnings number_of_customers earnings_per_customer,
      recreation := recreation_amount earnings percentage_for_recreation,
      expenses := total_monthly_expenses monthly_expenses recreation
  in savings earnings expenses = 872 :=
by
  sorry

end vince_savings_l101_101298


namespace problem_statement_l101_101168

theorem problem_statement (x y z w : ℝ) (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -1 := 
begin
  sorry
end

end problem_statement_l101_101168


namespace evaluate_Q_at_2_l101_101726

-- Define the polynomial Q(x)
noncomputable def Q (x : ℚ) : ℚ := x^4 - 20 * x^2 + 16

-- Define the root condition of sqrt(3) + sqrt(7)
def root_condition (x : ℚ) : Prop := (x = ℚ.sqrt(3) + ℚ.sqrt(7))

-- Theorem statement
theorem evaluate_Q_at_2 : Q 2 = -32 :=
by
  -- Given conditions
  have h1 : degree Q = 4 := sorry,
  have h2 : leading_coeff Q = 1 := sorry,
  have h3 : root_condition (ℚ.sqrt(3) + ℚ.sqrt(7)) := sorry,
  -- Goal
  exact sorry

end evaluate_Q_at_2_l101_101726


namespace arithmetic_sequence_geometric_condition_sum_first_n_b_seq_l101_101513

noncomputable def a_seq (n : ℕ) : ℕ := 2 * n + 4

def b_seq (n : ℕ) : ℚ := 2 / ((n + 1) * (a_seq n))

theorem arithmetic_sequence_geometric_condition (a_eq : ℕ → ℕ) (d : ℕ) (h1 : d ≠ 0) (h2 : a_eq 1 = 6) (h3 : a_eq 6 * a_eq 6 = a_eq 2 * a_eq 14) :
  a_eq n = 2 * n + 4 :=
sorry

theorem sum_first_n_b_seq (S : ℕ → ℚ) (b_eq : ℕ → ℚ) (hS : ∀ n, S n = ∑ i in range n, b_eq i) :
  ∀ n, S n = n / (2 * (n + 2)) :=
sorry

end arithmetic_sequence_geometric_condition_sum_first_n_b_seq_l101_101513


namespace workshop_workers_l101_101220

noncomputable def average_salary_all : ℝ := 850
noncomputable def num_technicians : ℕ := 7
noncomputable def average_salary_technicians : ℝ := 1000
noncomputable def average_salary_non_technicians : ℝ := 780
noncomputable def total_number_of_workers : ℕ := 22

theorem workshop_workers :
  ∃ W : ℝ, W = total_number_of_workers ∧ 
  (average_salary_all * W = (num_technicians * average_salary_technicians) + 
                           ((W - num_technicians) * average_salary_non_technicians)) :=
by
  use 22
  split
  · rfl
  · sorry

end workshop_workers_l101_101220


namespace min_elements_in_M_l101_101510

open Set Finite

theorem min_elements_in_M {α : Type} [Fintype α] (A B : Fin 20 → Finset α) :
  (∀ i, (A i).Nonempty) →
  (∀ i, (B i).Nonempty) →
  (∀ i j, i ≠ j → Disjoint (A i) (A j)) →
  (∀ i j, i ≠ j → Disjoint (B i) (B j)) →
  (⋃ i, A i = ⋃ i, B i) →
  (∀ i j, Disjoint (A i) (B j) → 18 ≤ (A i ∪ B j).card) →
  (⋃ i, A i).card = 180 :=
by
  intros
  sorry

end min_elements_in_M_l101_101510


namespace projection_is_same_l101_101429

-- Define the triangle ABC where AC is 6 cm and the triangle is a right isosceles triangle
def isosceles_right_triangle (A B C : Type) :=
  (AC : ℝ) (BC : ℝ) (AB : ℝ) (hAC : AC = 6)
  (hBC : BC = 6)
  (hRightAngle : ∠ACB = 90)
  (hIsosceles : AC = BC)
  (hHypotenuse : AB = AC * sqrt 2)

-- The projection of triangle ABC onto the projection plane is equal to itself
theorem projection_is_same (A B C : Type) : 
  isosceles_right_triangle A B C →
  ∀ AC BC AB, (∠ACB = 90) ∧ (AC = 6) ∧ (BC = 6) ∧ (AB = 6 * sqrt 2) → 
  isosceles_right_triangle A B C :=
by
  sorry

end projection_is_same_l101_101429


namespace non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101439

def g (m : Nat) : Nat :=
  let f k := m / k
  f 5 + f 25 + f 125 + f 625 + f 3125 + f 15625 + f 78125 + f 390625 + f 1953125 + f 9765625 + f 48828125 + f 244140625 + f 1220703125 + f 6103515625 + f 30517578125 -- continue as needed

def is_factorial_tail (n : Nat) : Prop :=
  ∃ m : Nat, g m = n

def count_factorial_tails (N : Nat) : Nat :=
  (List.range N).filter (λ n, is_factorial_tail n).length

theorem non_factorial_tails_lemma : count_factorial_tails 2500 = 2000 := sorry

theorem count_non_factorial_tails (N : Nat) : Prop :=
  let total := N - 1
  total - count_factorial_tails N = 499

theorem solution : count_non_factorial_tails 2500 := sorry

end non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101439


namespace arrange_numbers_l101_101901

variable {a : ℝ}

theorem arrange_numbers (h1 : -1 < a) (h2 : a < 0) : (1 / a < a) ∧ (a < a ^ 2) ∧ (a ^ 2 < |a|) :=
by 
  sorry

end arrange_numbers_l101_101901


namespace coefficient_of_x_squared_in_expansion_l101_101826

noncomputable def polynomial_expansion (p : Polynomial ℚ) (n : ℕ) : Polynomial ℚ :=
Polynomial.X ^ n + (-3) * Polynomial.C 1 * Polynomial.X ^ (n/2) + Polynomial.X * Polynomial.C 2 * Polynomial.X

theorem coefficient_of_x_squared_in_expansion : 
  (polynomial_expansion (Polynomial.X ^ 2 - 3 * Polynomial.X + Polynomial.C 2) 4).coeff 2 = 248 :=
sorry

end coefficient_of_x_squared_in_expansion_l101_101826


namespace boundary_length_of_pattern_l101_101383

theorem boundary_length_of_pattern (area : ℝ) (num_points : ℕ) 
(points_per_side : ℕ) : 
area = 144 → num_points = 4 → points_per_side = 4 →
∃ length : ℝ, length = 92.5 :=
by
  intros
  sorry

end boundary_length_of_pattern_l101_101383


namespace range_of_m_l101_101976

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - m * x + m > 0) ↔ 0 < m ∧ m < 4 :=
by sorry

end range_of_m_l101_101976


namespace solutions_of_system_l101_101476

theorem solutions_of_system (x y z : ℝ) :
    (x^2 - y = z^2) → (y^2 - z = x^2) → (z^2 - x = y^2) →
    (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
    (x = 1 ∧ y = 0 ∧ z = -1) ∨ 
    (x = 0 ∧ y = -1 ∧ z = 1) ∨ 
    (x = -1 ∧ y = 1 ∧ z = 0) := by
  sorry

end solutions_of_system_l101_101476


namespace min_sum_abs_elements_l101_101175

theorem min_sum_abs_elements (a b c d : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h : matrix.mul (matrix.from_blocks (matrix.scalar 2 1) (matrix.scalar 2 0) (matrix.scalar 2 0) (matrix.scalar 2 1)) 
  (matrix.from_blocks (matrix.scalar 2 1) (matrix.scalar 2 0) (matrix.scalar 2 0) (matrix.scalar 2 1)) 
  = matrix.from_blocks (matrix.scalar 2 12) (matrix.scalar 2 0) (matrix.scalar 2 0) (matrix.scalar 2 12)) :
  |a| + |b| + |c| + |d| = 10 := 
by
  sorry

end min_sum_abs_elements_l101_101175


namespace sum_of_first_9000_terms_of_geometric_sequence_l101_101263

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_9000_terms_of_geometric_sequence 
  (a r : ℝ)
  (h₁ : geometric_sum a r 3000 = 500)
  (h₂ : geometric_sum a r 6000 = 950)
  : geometric_sum a r 9000 = 1355 :=
sorry

end sum_of_first_9000_terms_of_geometric_sequence_l101_101263


namespace average_age_combined_l101_101690

-- Definitions of the given conditions
def avg_age_fifth_graders := 10
def number_fifth_graders := 40
def avg_age_parents := 40
def number_parents := 60

-- The theorem we need to prove
theorem average_age_combined : 
  (avg_age_fifth_graders * number_fifth_graders + avg_age_parents * number_parents) / (number_fifth_graders + number_parents) = 28 := 
by
  sorry

end average_age_combined_l101_101690


namespace find_c_l101_101167

-- Definitions of r and s
def r (x : ℝ) : ℝ := 4 * x - 9
def s (x : ℝ) (c : ℝ) : ℝ := 5 * x - c

-- Given and proved statement
theorem find_c (c : ℝ) : r (s 2 c) = 11 → c = 5 := 
by 
  sorry

end find_c_l101_101167


namespace similar_triangles_l101_101343

theorem similar_triangles (A B C H K M : EuclideanGeometry.Point) (triangle_ABC : EuclideanGeometry.Triangle A B C) 
  (BH : EuclideanGeometry.Altitude A B C H) 
  (AM : EuclideanGeometry.Median A B C M) 
  (CK : EuclideanGeometry.Median C B A K) :
  EuclideanGeometry.Similar (EuclideanGeometry.Triangle K H M) (EuclideanGeometry.Triangle A B C) := 
by
  sorry

end similar_triangles_l101_101343


namespace sin_angle_PAB_l101_101195

-- Axioms specifying lengths of the sides of triangle ABC
axiom AB : ℝ
axiom BC : ℝ
axiom CA : ℝ

-- Conditions from problem statement
axiom hAB : AB = 16
axiom hBC : BC = 17
axiom hCA : CA = 18

-- Definition of the angles being congruent
axiom ω : ℝ
axiom angle_congruent : ∀ P : Type, (angle P A B = ω) ∧ (angle P B C = ω) ∧ (angle P C A = ω)

-- The goal to prove
theorem sin_angle_PAB : sin ω = 0.5046 :=
sorry  -- Proof would be filled here

end sin_angle_PAB_l101_101195


namespace reciprocal_sum_of_roots_l101_101181

theorem reciprocal_sum_of_roots (c d : ℝ) (h1 : 7 * c^2 + 4 * c + 9 = 0)
  (h2 : 7 * d^2 + 4 * d + 9 = 0) (h3 : c ≠ d):
  (1 / c) + (1 / d) = -4 / 9 :=
by
  -- Using the relationships c + d = -4/7 and cd = 9/7
  have sum_roots : c + d = -4 / 7, from sorry,
  have prod_roots : c * d = 9 / 7, from sorry,
  -- Therefore, the sum of the reciprocals is given by
  calc (1 / c + 1 / d)
       = ((c + d) / (c * d)) : by sorry
    ... = (-4 / 7) / (9 / 7) : by sorry
    ... = -4 / 9 : sorry

end reciprocal_sum_of_roots_l101_101181


namespace polynomial_remainder_l101_101643

open Polynomial

theorem polynomial_remainder (Q : Polynomial ℝ) (h1 : Q.eval 15 = 8) (h2 : Q.eval 10 = 3) :
  ∃ a b : ℝ, Q = (X - 10) * (X - 15) * (Q.div_XsubC_Q (10, 15)) + Polynomial.C b + Polynomial.X * Polynomial.C a ∧ a = 1 ∧ b = -7 :=
by
  sorry

end polynomial_remainder_l101_101643


namespace intersecting_graphs_l101_101699

theorem intersecting_graphs (a b c d : ℝ) (h₁ : (3, 6) = (3, -|3 - a| + b))
  (h₂ : (9, 2) = (9, -|9 - a| + b))
  (h₃ : (3, 6) = (3, |3 - c| + d))
  (h₄ : (9, 2) = (9, |9 - c| + d)) : 
  a + c = 12 := 
sorry

end intersecting_graphs_l101_101699


namespace hundredth_non_multiple_of_three_or_five_eq_187_l101_101835

def is_neither_multiple_of_three_nor_five (n : ℕ) : Prop :=
  ¬(n % 3 = 0) ∧ ¬(n % 5 = 0)

def filtered_list (from to : ℕ) : List ℕ :=
  (List.range' from (to - from + 1)).filter is_neither_multiple_of_three_nor_five

theorem hundredth_non_multiple_of_three_or_five_eq_187 : 
  (filtered_list 1 200).nth 99 = some 187 := 
by
  sorry

end hundredth_non_multiple_of_three_or_five_eq_187_l101_101835


namespace population_net_increase_l101_101600

theorem population_net_increase (birth_rate death_rate : ℕ) (duration_seconds : ℕ) (seconds_in_a_day : ℕ) :
  birth_rate = 6 → death_rate = 2 → duration_seconds = 2 → seconds_in_a_day = 86400 →
  let net_increase_per_second := (birth_rate / duration_seconds) - (death_rate / duration_seconds) in
  net_increase_per_second * seconds_in_a_day = 172800 :=
by
  intros h1 h2 h3 h4
  let net_increase_per_second := (birth_rate / duration_seconds) - (death_rate / duration_seconds)
  calc
    net_increase_per_second * seconds_in_a_day
    = ((birth_rate / duration_seconds) - (death_rate / duration_seconds)) * seconds_in_a_day : by rfl
    ... = 2 * 86400 : by sorry
    ... = 172800 : by rfl

end population_net_increase_l101_101600


namespace f_sum_l101_101695

noncomputable def f : ℝ → ℝ := sorry

axiom domain_f : ∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, f y = x
axiom non_decreasing_f : ∀ x1 x2, 0 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 1 → f(x1) ≤ f(x2)
axiom f_0 : f(0) = 0
axiom f_frac : ∀ x, 0 ≤ x ∧ x ≤ 1 → f(x / 3) = (1 / 2) * f(x)
axiom f_sym : ∀ x, 0 ≤ x ∧ x ≤ 1 → f(1 - x) = 1 - f(x)

theorem f_sum : f(1 / 3) + f(1 / 8) = 3 / 4 := sorry

end f_sum_l101_101695


namespace MP_NQ_perpendicular_and_equal_length_l101_101671

noncomputable def isosceles_right_triangle (A B C : ℂ) : Prop :=
  ∃ (M : ℂ), (abs (M - A) = abs (M - B)) ∧ ((M - A) * (M - B).conj = 0)

noncomputable def are_perpendicular_and_same_length (P Q R S: ℂ) : Prop :=
  (abs (P - R) = abs (Q - S)) ∧
  ((P - R) * (Q - S).conj = 0)

theorem MP_NQ_perpendicular_and_equal_length
  (A B C D M N P Q : ℂ)
  (h1 : isosceles_right_triangle A M B)
  (h2 : isosceles_right_triangle B N C)
  (h3 : isosceles_right_triangle C P D)
  (h4 : isosceles_right_triangle D Q A)
  : are_perpendicular_and_same_length M P N Q :=
begin
  -- Proof goes here
  sorry
end

end MP_NQ_perpendicular_and_equal_length_l101_101671


namespace problem_l101_101080

noncomputable def sequence_a (n : ℕ) : ℝ :=
if n = 1 then 1 / 2 else 1 - sequence_b (n - 1)

noncomputable def sequence_b (n : ℕ) : ℝ :=
if n = 1 then 1 / 2 else sequence_b (n - 1) / (1 - (sequence_a (n - 1))^2)

theorem problem
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h1 : a 1 = 1 / 2)
  (h2 : ∀ n, a n + b n = 1)
  (h3 : ∀ n, b (n + 1) = b n / (1 - (a n)^2)) :
  b 2017 = 2017 / 2018 :=
sorry

end problem_l101_101080


namespace ellipse_equation_l101_101934

theorem ellipse_equation (h1 : ∃ (c : ℝ), parabola_focus=(-1,0) ∧ y^2 = -4*x) 
(h2 : ∃ (e : ℝ), e = 1/2): 
∃ (a b : ℝ), ellipse_equation = (x^2/4) + (y^2/3) := 
by 
  sorry

end ellipse_equation_l101_101934


namespace zoo_total_animals_l101_101390

theorem zoo_total_animals :
  let tiger_enclosures := 4
  let zebra_enclosures := tiger_enclosures * 2
  let giraffe_enclosures := zebra_enclosures * 3
  let tigers := tiger_enclosures * 4
  let zebras := zebra_enclosures * 10
  let giraffes := giraffe_enclosures * 2
  let total_animals := tigers + zebras + giraffes
  in total_animals = 144 := by
  sorry

end zoo_total_animals_l101_101390


namespace additional_charge_per_international_letter_l101_101861

-- Definitions based on conditions
def standard_postage_per_letter : ℕ := 108
def num_international_letters : ℕ := 2
def total_cost : ℕ := 460
def num_letters : ℕ := 4

-- Theorem stating the question
theorem additional_charge_per_international_letter :
  (total_cost - (num_letters * standard_postage_per_letter)) / num_international_letters = 14 :=
by
  sorry

end additional_charge_per_international_letter_l101_101861


namespace coefficient_of_x4_l101_101621

theorem coefficient_of_x4 : 
  (Finset.sum (Finset.range 6) 
    (λ r, (Nat.choose 5 r) * (2:ℕ)^(5-r) * (-1:ℤ)^r * (x:ℤ)^(10 - 3 * r))) = 80 * x^4 :=
  sorry

end coefficient_of_x4_l101_101621


namespace number_of_real_values_of_p_l101_101166

theorem number_of_real_values_of_p (p : ℝ) :
  (∃ n : ℕ, n = 2 ∧ ∀ p : ℝ, (p = 1 + 2 * Real.sqrt 2 ∨ p = 1 - 2 * Real.sqrt 2) ↔ (∀ x : ℝ, (x^2 - (p + 1) * x + (p + 2) = 0 → discrim_eqz : x^2 - (p + 1) * x + (p + 2) = 0 ))) := 
sorry

end number_of_real_values_of_p_l101_101166


namespace geometric_sequence_sum_l101_101257

theorem geometric_sequence_sum
  (a r : ℝ)
  (h1 : a * (1 - r ^ 3000) / (1 - r) = 500)
  (h2 : a * (1 - r ^ 6000) / (1 - r) = 950) :
  a * (1 - r ^ 9000) / (1 - r) = 1355 :=
sorry

end geometric_sequence_sum_l101_101257


namespace trains_passing_time_l101_101294

-- Definitions based on the conditions
def train1_speed_kmh : ℝ := 50
def train2_speed_kmh : ℝ := 40
def train1_length_m : ℝ := 125
def train2_length_m : ℝ := 125.02

-- Conversions and calculations
def kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600
def relative_speed_mps : ℝ := kmh_to_mps train1_speed_kmh - kmh_to_mps train2_speed_kmh
def total_length_m : ℝ := train1_length_m + train2_length_m

theorem trains_passing_time :
  (total_length_m / relative_speed_mps) = 90 :=
by sorry

end trains_passing_time_l101_101294


namespace number_of_CO2_moles_l101_101496

/-
Define the conditions and prove that combining some moles of CO2 with 3 moles of H2O forms 3 moles of H2CO3,
therefore the moles of CO2 required are 3.
-/

variable (CO2 H2O H2CO3 : Type)
variable (moles_CO2 moles_H2O moles_H2CO3 : ℕ)

-- Define the balanced chemical equation relation
def reaction (nCO2 nH2O nH2CO3 : ℕ) : Prop :=
  nCO2 = nH2CO3 ∧ nH2O = nH2CO3

-- Given conditions
axiom water_moles : moles_H2O = 3
axiom carbonic_acid_moles : moles_H2CO3 = 3

-- Proof statement
theorem number_of_CO2_moles :
  reaction moles_CO2 moles_H2O moles_H2CO3 → moles_CO2 = 3 := 
by
  assume h : reaction moles_CO2 moles_H2O moles_H2CO3
  rw [←h.1, carbonic_acid_moles]
  sorry

end number_of_CO2_moles_l101_101496


namespace root_in_interval_l101_101397

noncomputable def validate_root_interval (a b c : ℝ) (h : a ≠ 0) : Prop :=
  let f := λ x : ℝ, a * x^2 + b * x + c in
  (f 0.5 = -0.25 ∧ f 0.6 = 0.16 ∧ f 0.4 = -0.64 ∧ f 0.7 = 0.59)
  → ∃ x : ℝ, 0.5 < x ∧ x < 0.6 ∧ f x = 0

theorem root_in_interval (a b c: ℝ) (h: a ≠ 0)
  (h0: a * 0.4^2 + b * 0.4 + c = -0.64)
  (h1: a * 0.5^2 + b * 0.5 + c = -0.25)
  (h2: a * 0.6^2 + b * 0.6 + c = 0.16)
  (h3: a * 0.7^2 + b * 0.7 + c = 0.59) :
  ∃ x: ℝ, 0.5 < x ∧ x < 0.6 ∧ a * x^2 + b * x + c = 0 :=
by
  sorry

end root_in_interval_l101_101397


namespace final_height_of_helicopter_total_fuel_consumed_l101_101009

noncomputable def height_changes : List Float := [4.1, -2.3, 1.6, -0.9, 1.1]

def total_height_change (changes : List Float) : Float :=
  changes.foldl (λ acc x => acc + x) 0

theorem final_height_of_helicopter :
  total_height_change height_changes = 3.6 :=
by
  sorry

noncomputable def fuel_consumption (changes : List Float) : Float :=
  changes.foldl (λ acc x => if x > 0 then acc + 5 * x else acc + 3 * -x) 0

theorem total_fuel_consumed :
  fuel_consumption height_changes = 43.6 :=
by
  sorry

end final_height_of_helicopter_total_fuel_consumed_l101_101009


namespace relationship_y₁_y₂_y₃_l101_101923

variables (y₁ y₂ y₃ : ℝ)

def inverse_proportion (x : ℝ) : ℝ := 3 / x

-- Given points A(-2, y₁), B(-1, y₂), C(1, y₃)
-- and y₁ = inverse_proportion(-2), y₂ = inverse_proportion(-1), y₃ = inverse_proportion(1)
theorem relationship_y₁_y₂_y₃ : 
  let y₁ := inverse_proportion (-2),
      y₂ := inverse_proportion (-1),
      y₃ := inverse_proportion (1) in
  y₂ < y₁ ∧ y₁ < y₃ :=
by
  sorry

end relationship_y₁_y₂_y₃_l101_101923


namespace xyz_sum_l101_101709

-- Necessary definitions for the problem setup
variables (P Q R S : Type*) 
variables (dist : P → P → ℝ) 
variables (radius : ℝ) (dPQ : dist P Q = 20) (dQR : dist Q R = 21) (dRP : dist R P = 29)
variables (is_on_sphere : ∀ (X : P), dist X S = 25)

-- The distance formula from the sphere's center to the plane of the triangle
def distance_from_center_to_plane (h : ℝ) : ℝ := 266 * Real.sqrt 154 / 14

-- x, y, z as defined in the problem with specified properties
noncomputable def x : ℕ := 266
noncomputable def y : ℕ := 154
noncomputable def z : ℕ := 14

-- Given x, y, z values, we need to prove the final result
theorem xyz_sum : 
  (x + y + z = 434) :=
by 
  -- Specific conditions and calculations will be instantiated here
  sorry

end xyz_sum_l101_101709


namespace polynomial_remainder_l101_101642

open Polynomial

theorem polynomial_remainder (Q : Polynomial ℝ) (h1 : Q.eval 15 = 8) (h2 : Q.eval 10 = 3) :
  ∃ a b : ℝ, Q = (X - 10) * (X - 15) * (Q.div_XsubC_Q (10, 15)) + Polynomial.C b + Polynomial.X * Polynomial.C a ∧ a = 1 ∧ b = -7 :=
by
  sorry

end polynomial_remainder_l101_101642


namespace volunteers_correct_l101_101792

-- Definitions of given conditions and the required result
def sheets_per_member : ℕ := 10
def cookies_per_sheet : ℕ := 16
def total_cookies : ℕ := 16000

-- Number of members who volunteered
def members : ℕ := total_cookies / (sheets_per_member * cookies_per_sheet)

-- Proof statement
theorem volunteers_correct :
  members = 100 :=
sorry

end volunteers_correct_l101_101792


namespace num_not_factorial_tails_lt_2500_l101_101471

-- Definition of the function f(m)
def f (m : ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625 + m / 78125 + m / 390625 + m / 1953125

-- The theorem stating the main result
theorem num_not_factorial_tails_lt_2500 : 
  let count_non_tails := ∑ k in finset.range 2500, if ∀ m, f m ≠ k then 1 else 0
  in count_non_tails = 500 :=
by
  sorry

end num_not_factorial_tails_lt_2500_l101_101471


namespace train_speed_l101_101355

theorem train_speed (length : ℕ) (time : ℕ) (k : ℕ) : 
  (length = 250) → 
  (time = 12) → 
  (k = 3.6) →
  ((length / time : ℕ) * k = 75) :=
by
  sorry

end train_speed_l101_101355


namespace eq_of_complex_parts_equal_l101_101540

theorem eq_of_complex_parts_equal (a : ℝ) 
  (h : (2 + a : ℂ) / (5 : ℂ) = (2a - 1 : ℂ) / (5 : ℂ)) : a = 3 := 
  by
  -- sorry is used to skip the proof
  sorry

end eq_of_complex_parts_equal_l101_101540


namespace vector_magnitude_l101_101568

theorem vector_magnitude (x : ℝ) :
  let a := (1, -2)
  let b := (x, 2) in
  a.1 * b.1 + a.2 * b.2 = 0 → (real.sqrt (b.1^2 + b.2^2) = 2 * real.sqrt 5) :=
by
  intro h
  sorry

end vector_magnitude_l101_101568


namespace regular_polygon_sides_l101_101574

theorem regular_polygon_sides (exterior_angle : ℝ) (h : exterior_angle = 45) : (∃ n : ℕ, n = 360 / exterior_angle) :=
by {
  existsi 8,
  simp [h],
  norm_num,
}

end regular_polygon_sides_l101_101574


namespace coefficient_x2y3z_in_expansion_l101_101323

theorem coefficient_x2y3z_in_expansion :
  ∀ (x y z : ℕ), (x = 2) → (y = 3) → (z = 1) → 
  nat.choose 6 x * nat.choose (6 - x) y * nat.choose (6 - x - y) z = 60 :=
by 
  intros x y z hx hy hz
  rw [hx, hy, hz]
  norm_num
  sorry -- Placeholder for the complete proof

end coefficient_x2y3z_in_expansion_l101_101323


namespace seedlings_planted_l101_101677

theorem seedlings_planted (x : ℕ) (h1 : 2 * x + x = 1200) : x = 400 :=
by {
  sorry
}

end seedlings_planted_l101_101677


namespace fractionOf_Product_Of_Fractions_l101_101302

noncomputable def fractionOfProductOfFractions := 
  let a := (2 : ℚ) / 9 * (5 : ℚ) / 6 -- Define the product of the fractions
  let b := (3 : ℚ) / 4 -- Define another fraction
  a / b = 20 / 81 -- Statement to be proven

theorem fractionOf_Product_Of_Fractions: fractionOfProductOfFractions :=
by sorry

end fractionOf_Product_Of_Fractions_l101_101302


namespace inverse_proportion_relation_l101_101921

theorem inverse_proportion_relation :
  let y1 := 3 / (-2)
  let y2 := 3 / (-1)
  let y3 := 3 / 1
  y2 < y1 ∧ y1 < y3 :=
by
  -- Variable definitions according to conditions
  let y1 := 3 / (-2)
  let y2 := 3 / (-1)
  let y3 := 3 / 1
  -- Proof steps go here (not required for the statement)
  -- Since proof steps are omitted, we use sorry to indicate it
  sorry

end inverse_proportion_relation_l101_101921


namespace cistern_fill_time_l101_101334

theorem cistern_fill_time (F : ℝ) (E : ℝ) (net_rate : ℝ) (time : ℝ)
  (h_F : F = 1 / 4)
  (h_E : E = 1 / 8)
  (h_net : net_rate = F - E)
  (h_time : time = 1 / net_rate) :
  time = 8 := 
sorry

end cistern_fill_time_l101_101334


namespace sum_divisors_of_37_is_38_l101_101321

theorem sum_divisors_of_37_is_38 (h : Nat.Prime 37) : (∑ d in (Finset.filter (λ d, 37 % d = 0) (Finset.range (37 + 1))), d) = 38 := by
  sorry

end sum_divisors_of_37_is_38_l101_101321


namespace cone_apex_angle_60_l101_101324

def lateral_area (r l : ℝ) : ℝ := π * r * l
def base_area (r : ℝ) : ℝ := π * r ^ 2

def cone_apex_angle (r l : ℝ) (h : ℝ) : Prop :=
  (l = 2 * r) →
  (h = √(l ^ 2 - r ^ 2)) →
  (∃ θ : ℝ, tan (θ / 2) = r / h ∧ θ = 60)

theorem cone_apex_angle_60 (r l h : ℝ) : cone_apex_angle r l h :=
sorry

end cone_apex_angle_60_l101_101324


namespace hazel_ratio_is_correct_father_ratio_is_correct_l101_101082

variables (hazelA hazelB fatherA fatherB : ℕ)
variables (hazelRatio fatherRatio : ℚ)

-- Conditions for Hazel
def hazel_conditions : Prop :=
  hazelA = 48 ∧ hazelB = 32

-- Condition for Hazel's ratio
def hazel_ratio_conditions : Prop :=
  hazelRatio = hazelA / hazelB

-- Conditions for Hazel's father
def father_conditions : Prop :=
  fatherA = 46 ∧ fatherB = 24

-- Condition for Hazel's father's ratio
def father_ratio_conditions : Prop :=
  fatherRatio = fatherA / fatherB

theorem hazel_ratio_is_correct (hA : hazelA = 48) (hB : hazelB = 32) : hazelRatio = 3 / 2 :=
  sorry

theorem father_ratio_is_correct (fA : fatherA = 46) (fB : fatherB = 24) : fatherRatio = 23 / 12 :=
  sorry

end hazel_ratio_is_correct_father_ratio_is_correct_l101_101082


namespace limit_of_fraction_l101_101013

theorem limit_of_fraction (n : ℕ) :
  (real tendsto (λ n, (5 * n^2 - 2) / ((n - 3) * (n + 1))) at_top (𝓝 5)) := 
sorry

end limit_of_fraction_l101_101013


namespace hexagon_problem_l101_101347

theorem hexagon_problem (ABCDEF : RegularHexagon) (O : Point) (A EF D E C P Q R : Point) (h1 : center O ABCDEF) 
(h2 : perpendicular A EF P) (h3 : perpendicular A D Q) (h4 : perpendicular A C R) (OP_eq : dist O P = 1) :
  dist O A + dist O Q + dist O R = 5 :=
sorry

end hexagon_problem_l101_101347


namespace smallest_non_multiple_of_5_abundant_l101_101006

def properDivisors (n : ℕ) : List ℕ :=
  (List.range (n - 1 + 1) ).filter (fun d => d < n ∧ n % d = 0)

def isAbundant (n : ℕ) : Prop :=
  properDivisors n |>.sum > n

def isNotMultipleOfFive (n : ℕ) : Prop :=
  n % 5 ≠ 0

theorem smallest_non_multiple_of_5_abundant :
  ∃ n, isAbundant n ∧ isNotMultipleOfFive n ∧ 
       ∀ m, isAbundant m ∧ isNotMultipleOfFive m → n ≤ m :=
  sorry

end smallest_non_multiple_of_5_abundant_l101_101006


namespace johns_total_payment_l101_101774

theorem johns_total_payment
  (P : ℝ) (r : ℝ) (t : ℝ) 
  (hP : P = 6650) 
  (hr : r = 6) 
  (ht : t = 10) : 
  let rebate_amount := (r / 100) * P in
  let price_after_rebate := P - rebate_amount in
  let sales_tax := (t / 100) * price_after_rebate in
  let total_amount := price_after_rebate + sales_tax in
  total_amount = 6876.10 := by
  sorry

end johns_total_payment_l101_101774


namespace sequence_sum_terms_l101_101561

theorem sequence_sum_terms (a : ℕ → ℝ) (n : ℕ) (h : (∑ i in Finset.range n, (i+1) * a (i+1)) = 2 * n) :
  (∑ i in Finset.range n, a (i+1) * a (i+2)) = 4 * n / (n + 1) :=
by
  sorry

end sequence_sum_terms_l101_101561


namespace evaluate_Q_at_2_l101_101728

-- Define the polynomial Q(x)
noncomputable def Q (x : ℚ) : ℚ := x^4 - 20 * x^2 + 16

-- Define the root condition of sqrt(3) + sqrt(7)
def root_condition (x : ℚ) : Prop := (x = ℚ.sqrt(3) + ℚ.sqrt(7))

-- Theorem statement
theorem evaluate_Q_at_2 : Q 2 = -32 :=
by
  -- Given conditions
  have h1 : degree Q = 4 := sorry,
  have h2 : leading_coeff Q = 1 := sorry,
  have h3 : root_condition (ℚ.sqrt(3) + ℚ.sqrt(7)) := sorry,
  -- Goal
  exact sorry

end evaluate_Q_at_2_l101_101728


namespace proof_ellipse_l101_101515

noncomputable def ellipse_equation_and_fixed_point : Prop :=
  ∃ (a b x y : ℝ), a > b ∧ b > 0 ∧ e = sqrt (3) / 2 ∧
  (x = 2 ∧ y = 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
  ∀ (M N: ℝ × ℝ),
    M ≠ (2, 0) ∧ N ≠ (2, 0) ∧
    (M.1^2 / 4 + M.2^2 = 1) ∧ (N.1^2 / 4 + N.2^2 = 1) ∧
    ((M.2 / (M.1 - 2)) * (N.2 / (N.1 - 2)) = -1/4) →
    (∃ x : ℝ, x = 0 ∧ (x = 0) ) ∧
    (∃ y : ℝ, y = 0 ∧ (y = 0) )

theorem proof_ellipse (h : ellipse_equation_and_fixed_point) :
  ∃ a b : ℝ, a = 2 ∧ b = 1 ∧ ( ∀ (M N: ℝ × ℝ), 
    M ≠ (2, 0) ∧ N ≠ (2, 0) ∧ 
    (M.1^2 / 4 + M.2^2 = 1) ∧ (N.1^2 / 4 + N.2^2 = 1) ∧ 
    ((M.2 / (M.1 - 2)) * (N.2 / (N.1 - 2)) = -1/4) → 
    (∃ x : ℝ, x = 0 ∧ (x = 0) ) ∧
    (∃ y : ℝ, y = 0 ∧ (y = 0) )) :=
begin
  sorry
end

end proof_ellipse_l101_101515


namespace roots_of_equation_l101_101004

theorem roots_of_equation : ∃ x₁ x₂ : ℝ, (3 ^ x₁ = Real.log (x₁ + 9) / Real.log 3) ∧ 
                                     (3 ^ x₂ = Real.log (x₂ + 9) / Real.log 3) ∧ 
                                     (x₁ < 0) ∧ (x₂ > 0) := 
by {
  sorry
}

end roots_of_equation_l101_101004


namespace complex_number_in_first_quadrant_l101_101688

theorem complex_number_in_first_quadrant :
  (let z := (i / exp(Real.pi / 4 * Complex.I)) in
  0 < z.re ∧ 0 < z.im) := 
by
  sorry

end complex_number_in_first_quadrant_l101_101688


namespace monotonicity_of_f_inequality_solution_set_l101_101905

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (x + 1)
  else -x^2 + 2 * x

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

theorem inequality_solution_set :
  { x : ℝ | f (2 * x - 1) > f (2 - x) } = {x : ℝ | x > 1 / 3} :=
by
  sorry

end monotonicity_of_f_inequality_solution_set_l101_101905


namespace original_number_l101_101815

theorem original_number (N y x : ℕ) 
  (h1: N + y = 54321)
  (h2: N = 10 * y + x)
  (h3: 11 * y + x = 54321)
  (h4: x = 54321 % 11)
  (hy: y = 4938) : 
  N = 49383 := 
  by 
  sorry

end original_number_l101_101815


namespace find_shaded_area_l101_101379

-- Define the side length of the larger hexagon
def side_length_large_hex : ℝ := 4

-- Define the radius of each semicircle
def radius_semi : ℝ := side_length_large_hex / 2

-- Define the area of the larger hexagon
def area_large_hex : ℝ := 3 * Real.sqrt 3 * side_length_large_hex^2 / 2

-- Define the area of one semicircle
def area_one_semi : ℝ := Real.pi * radius_semi^2 / 2

-- Define the total area of all semicircles
def total_area_semis : ℝ := 6 * area_one_semi

-- Define the side length of the smaller hexagon
def side_length_small_hex : ℝ := radius_semi * Real.sqrt 3

-- Define the area of the smaller hexagon
def area_small_hex : ℝ := 3 * Real.sqrt 3 * side_length_small_hex^2 / 2

-- Define the total shaded area inside the larger hexagon but outside the semicircles and smaller hexagon
def area_shaded : ℝ := area_large_hex - total_area_semis - area_small_hex

-- Theorem statement for the given problem
theorem find_shaded_area : area_shaded = 6 * Real.sqrt 3 - 48 * Real.pi :=
by
  sorry

end find_shaded_area_l101_101379


namespace smallest_k_l101_101651

-- Sequence definition
def u : ℕ → ℝ
| 0     := 1 / 3
| (k+1) := 3 * u k - 3 * (u k)^3

-- Limit definition
def L := 1 / 3

-- Proof statement
theorem smallest_k (k : ℕ) (hk : k ≥ 1) : 
  |u k - L| ≤ 1 / 3 ^ 300 :=
sorry

end smallest_k_l101_101651


namespace stable_equilibrium_condition_l101_101370

theorem stable_equilibrium_condition
  (a b : ℝ)
  (h_condition1 : a > b)
  (h_condition2 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  : (b / a) < (1 / Real.sqrt 2) :=
sorry

end stable_equilibrium_condition_l101_101370


namespace theta_in_second_quadrant_l101_101116

theorem theta_in_second_quadrant (θ : ℝ) :
  (-sin θ < 0) ∧ (cos θ < 0) → (π/2 < θ ∧ θ < π) :=
by
  intro h
  -- Proof goes here
  sorry

end theta_in_second_quadrant_l101_101116


namespace range_of_k_l101_101958

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x^2 + 2 * x - 1 = 0) ↔ (k ≥ 0 ∧ k ≠ 1) :=
by
  sorry

end range_of_k_l101_101958


namespace can_cut_apples_l101_101716

-- Define the conditions given in the problem
def apple_weight := 0.025  -- weight of each apple in kg
def total_apples := 100
def total_weight := 10  -- total weight of all apples in kg
def child_weight := 0.1  -- weight each child should get in kg
def children := 100

-- State the theorem to be proved
theorem can_cut_apples (h1 : total_apples = 100)
                       (h2 : total_weight = 10)
                       (h3 : ∀ k, 1 ≤ k ∧ k ≤ total_apples → apple_weight ≥ 0.025) :
  ∃ (cutting : List (Fin total_apples → ℝ)), 
    (∀ i, 0 < i ∧ i ≤ children → (cutting.nth i).getD 0 ≥ child_weight) ∧
    (∀ piece, piece ∈ cutting → piece ≥ 0.025) ∧ 
    (cutting.sum = total_weight) :=
sorry

end can_cut_apples_l101_101716


namespace thabo_books_l101_101215

/-- Thabo's book count puzzle -/
theorem thabo_books (H P F : ℕ) (h1 : P = H + 20) (h2 : F = 2 * P) (h3 : H + P + F = 200) : H = 35 :=
by
  -- sorry is used to skip the proof, only state the theorem.
  sorry

end thabo_books_l101_101215


namespace chromosomal_variation_incorrect_statement_l101_101837

theorem chromosomal_variation_incorrect_statement 
  (A : Prop := "Chromosomal variation includes changes in chromosomal structure and changes in the number of chromosomes.")
  (B : Prop := "The exchange of segments between two chromosomes both fall under chromosomal variation.")
  (C : Prop := "Patients with Cri-du-chat syndrome have a partial deletion of chromosome 5 compared to normal people.")
  (D : Prop := "Changes in chromosomal structure can alter the number or arrangement of genes located on the chromosome.")
  (hA : A)
  (hC : C)
  (hD : D) : ¬ B :=
sorry

end chromosomal_variation_incorrect_statement_l101_101837


namespace sin_identity_proof_l101_101528

theorem sin_identity_proof (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 4) :
  Real.sin (5 * π / 6 - x) + Real.sin (π / 3 - x) ^ 2 = 19 / 16 :=
by
  sorry

end sin_identity_proof_l101_101528


namespace sum_of_roots_l101_101897

theorem sum_of_roots (x : ℝ) :
  (x^2 - 2003 * x - 2004 = 0) → (∑ x, x^2 - 2003 * x - 2004 = 2003) :=
begin
  sorry
end

end sum_of_roots_l101_101897


namespace problem1_problem1_not_linear_control_problem2_problem3_l101_101047

open Real

-- Problem 1
theorem problem1
  (f : ℝ → ℝ) (h_f: f = sin) :
  (∀ x : ℝ, |deriv f x| ≤ 1) :=
sorry

theorem problem1_not_linear_control
  (g : ℝ → ℝ) (h_g: g = exp) :
  ¬(∀ x : ℝ, |deriv g x| ≤ 1) :=
sorry

-- Problem 2
theorem problem2 (f : ℝ → ℝ)
  (h_linear: ∀ x : ℝ, |deriv f x| ≤ 1)
  (h_increasing: ∀ x y : ℝ, x < y → f x < f y)
  (x₁ x₂ : ℝ) (h_x: x₁ < x₂) :
  0 < (f x₂ - f x₁) / (x₂ - x₁) ∧ (f x₂ - f x₁) / (x₂ - x₁) ≤ 1 :=
sorry

-- Problem 3
theorem problem3 (f : ℝ → ℝ)
  (h_linear: ∀ x : ℝ, |deriv f x| ≤ 1)
  (h_periodic : ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x)
  (x₁ x₂ : ℝ) :
  |f x₁ - f x₂| ≤ h_periodic.some :=
sorry

end problem1_problem1_not_linear_control_problem2_problem3_l101_101047


namespace simplest_quadratic_radical_is_sqrt15_l101_101327

def is_simplest_quadratic_radical (r : ℝ) : Prop :=
  ∀ (s : ℝ), r = s * s → r = s

def sqrt_15_is_simplest : Prop :=
  let rA := real.sqrt 15
  let rB := real.sqrt 12
  let rC := real.sqrt (1 / 3)
  let rD := real.cbrt 12
  is_simplest_quadratic_radical rA ∧
  ¬ is_simplest_quadratic_radical rB ∧
  ¬ is_simplest_quadratic_radical rC ∧
  ¬ is_simplest_quadratic_radical rD

theorem simplest_quadratic_radical_is_sqrt15 : sqrt_15_is_simplest := by
  sorry

end simplest_quadratic_radical_is_sqrt15_l101_101327


namespace count_non_factorial_tails_lt_2500_l101_101447

def f (m : ℕ) : ℕ := 
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + ... -- Continue as needed

theorem count_non_factorial_tails_lt_2500 : 
  #{ n : ℕ | n < 2500 ∧ ¬ (∃ m : ℕ, f(m) = n) } = 4 :=
by sorry

end count_non_factorial_tails_lt_2500_l101_101447


namespace correct_calculation_l101_101768

theorem correct_calculation :
  (3 * Real.sqrt 2) * (2 * Real.sqrt 3) = 6 * Real.sqrt 6 :=
by sorry

end correct_calculation_l101_101768


namespace diving_competition_scores_l101_101986

theorem diving_competition_scores (A B C D E : ℝ) (hA : 1 ≤ A ∧ A ≤ 10)
  (hB : 1 ≤ B ∧ B ≤ 10) (hC : 1 ≤ C ∧ C ≤ 10) (hD : 1 ≤ D ∧ D ≤ 10) 
  (hE : 1 ≤ E ∧ E ≤ 10) (degree_of_difficulty : ℝ) (h_diff : degree_of_difficulty = 3.2)
  (point_value : ℝ) (h_point_value : point_value = 79.36) :
  A = max A (max B (max C (max D E))) →
  E = min A (min B (min C (min D E))) →
  (B + C + D) = (point_value / degree_of_difficulty) :=
by sorry

end diving_competition_scores_l101_101986


namespace functional_eq_solve_l101_101869

theorem functional_eq_solve (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (2*x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := 
sorry

end functional_eq_solve_l101_101869


namespace sum_of_ages_l101_101597

-- Define the ages of the animals and the aging process
def lioness_age (L : ℝ) : Prop := L = 12 
def hyena_age (H : ℝ) : Prop := L = 2 * H
def leopard_age (P : ℝ) : Prop := P = 3 * H

def baby_age (A : ℝ) : ℝ := A / 2

def aging_rate (animal_rate : ℝ) (human_years : ℝ) : ℝ := animal_rate * human_years
def new_baby_age (baby : ℝ) (rate : ℝ) (human_years : ℝ) : ℝ := baby + (aging_rate rate human_years / 2)

-- Calculation for sum of the ages of the babies in five human years
theorem sum_of_ages (L H P : ℝ) (BL BH BP NBL NBH NBP : ℝ) :
  lioness_age L → hyena_age H → leopard_age P → 
  BL = baby_age L → BH = baby_age H → BP = baby_age P →
  NBL = new_baby_age BL 1.5 5 → NBH = new_baby_age BH 1.25 5 → NBP = new_baby_age BP 2 5 → 
  (NBL + NBH + NBP = 29.875) :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  rw [h1, h2, h3, h4, h5, h6, h7, h8, h9]
  sorry

end sum_of_ages_l101_101597


namespace correct_propositions_l101_101399

theorem correct_propositions (a b c α β : ℝ) (f : ℝ → ℝ) :
  (c^2 > 0 → (ac^2 > bc^2 → a > b)) ∧
  (¬(sin α = sin β → α = β)) ∧
  ((∀ y: ℝ, x - 2*a*y = 1 ∧ 2*x - 2*a*y = 1 → a = 0) ↔ 
  (∀ y: ℝ, x - 2*a*y = 1 ∧ 2*x - 2*a*y ≠ 1 → a ≠ 0)) ∧
  (∀ x, f x = real.log x → f (abs x) = real.log (abs x)) :=
by
  sorry

end correct_propositions_l101_101399


namespace cuberoot_sum_eq_a_has_solution_l101_101017

theorem cuberoot_sum_eq_a_has_solution (a : ℝ) (h : a ≥ 0) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ (∛(1 + x) + ∛(1 - x) = a)) ↔ (a ∈ set.Icc (∛2) 2) :=
by 
  noncomputable theory
  sorry

end cuberoot_sum_eq_a_has_solution_l101_101017


namespace dishonest_dealer_weight_l101_101364

variable (CostPrice SellingPrice Profit Weight : ℝ)

theorem dishonest_dealer_weight :
  Profit = 66.67 / 100 → 
  CostPrice = 100 → 
  SellingPrice = CostPrice * (1 + Profit) → 
  Weight = CostPrice / SellingPrice →
  Weight ≈ 0.6 :=
by sorry

end dishonest_dealer_weight_l101_101364


namespace solve_arccos_eq_l101_101207

theorem solve_arccos_eq (x : ℝ) (h1 : -1 ≤ 3 * x ∧ 3 * x ≤ 1) (h2 : -1 ≤ x ∧ x ≤ 1) :
  (arccos (3 * x) - arccos x = π / 6) ↔ (x = 0.1578 ∨ x = -0.1578) :=
by
  -- The proof will be filled in later
  sorry

end solve_arccos_eq_l101_101207


namespace tagged_fish_ratio_l101_101592

theorem tagged_fish_ratio (tagged_first_catch : ℕ) 
(tagged_second_catch : ℕ) (total_second_catch : ℕ) 
(h1 : tagged_first_catch = 30) (h2 : tagged_second_catch = 2) 
(h3 : total_second_catch = 50) : tagged_second_catch / total_second_catch = 1 / 25 :=
by
  sorry

end tagged_fish_ratio_l101_101592


namespace number_of_correct_statements_l101_101400

-- Definitions according to conditions
def rhombuses_not_necessarily_similar : Prop := ¬∀ (R1 R2 : Type), similar_shapes R1 R2
def equilateral_triangles_similar : Prop := ∀ (T1 T2 : Type), equilateral_triangle T1 → equilateral_triangle T2 → similar_shapes T1 T2
def squares_similar : Prop := ∀ (S1 S2 : Type), square S1 → square S2 → similar_shapes S1 S2
def rectangles_not_necessarily_similar : Prop := ¬∀ (R1 R2 : Type), rectangle R1 → rectangle R2 → similar_shapes R1 R2
def congruent_triangles_similar : Prop := ∀ (T1 T2 : Type), congruent_triangles T1 T2 → similar_shapes T1 T2
def right_angled_triangles_not_necessarily_similar : Prop := ¬∀ (T1 T2 : Type), right_angled_triangle T1 → right_angled_triangle T2 → similar_shapes T1 T2

-- Proof problem
theorem number_of_correct_statements :
  (¬∀ (R1 R2 : Type), similar_shapes R1 R2) ∧
  (∀ (T1 T2 : Type), equilateral_triangle T1 → equilateral_triangle T2 → similar_shapes T1 T2) ∧
  (∀ (S1 S2 : Type), square S1 → square S2 → similar_shapes S1 S2) ∧
  (¬∀ (R1 R2 : Type), rectangle R1 → rectangle R2 → similar_shapes R1 R2) ∧
  (∀ (T1 T2 : Type), congruent_triangles T1 T2 → similar_shapes T1 T2) ∧
  (¬∀ (T1 T2 : Type), right_angled_triangle T1 → right_angled_triangle T2 → similar_shapes T1 T2) →
  num_correct_statements = 3 :=
by
  sorry

end number_of_correct_statements_l101_101400


namespace SquareArea_l101_101745

theorem SquareArea (s : ℝ) (θ : ℝ) (h1 : s = 3) (h2 : θ = π / 4) : s * s = 9 := 
by 
  sorry

end SquareArea_l101_101745


namespace intersection_points_l101_101058

def periodic_func (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def func_def (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 ≤ x ∧ x < 2) → f x = x^3 - x

theorem intersection_points (f : ℝ → ℝ)
  (h_periodic : periodic_func f 2)
  (h_def : func_def f) :
  (∃ s : set ℝ, s = { x ∈ set.Icc 0 6 | f x = 0 } ∧ s.to_finset.card = 7) :=
sorry

end intersection_points_l101_101058


namespace problem1_l101_101782

theorem problem1 (α : Real) (h : Real.tan (Real.pi / 4 + α) = 2) : 
  (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 := 
sorry

end problem1_l101_101782


namespace gardening_arrangement_l101_101282

theorem gardening_arrangement :
  ∃ x y : ℕ, 
    (20 ≤ x ∧ x ≤ 22) ∧ 
    y = 50 - x ∧ 
    70 * x + 40 * y ≤ 2660 ∧ 
    30 * x + 80 * y ≤ 3000 ∧ 
    (∀ x' y' : ℕ, (20 ≤ x' ∧ x' ≤ 22) ∧ y' = 50 - x' →
      70 * x' + 40 * y' ≤ 2660 ∧ 
      30 * x' + 80 * y' ≤ 3000 → 
      800 * x + 960 * y ≤ 800 * x' + 960 * y') :=
begin
  sorry
end

end gardening_arrangement_l101_101282


namespace part_a_part_b_l101_101337

noncomputable def is_regular_tetrahedron (a b c a₁ b₁ c₁ : ℝ) : Prop := 
  a = b ∧ b = c ∧ c = a₁ ∧ a₁ = b₁ ∧ b₁ = c₁

noncomputable def sum_of_edges_squared (a b c a₁ b₁ c₁ : ℝ) : ℝ := 
  a^2 + b^2 + c^2 + a₁^2 + b₁^2 + c₁^2

noncomputable def sum_of_faces_areas_squared (Δ : ℝ) : ℝ := 
  Δ  -- assuming Δ represents this sum directly.

theorem part_a (a b c a₁ b₁ c₁ Δ : ℝ) :
  is_regular_tetrahedron a b c a₁ b₁ c₁ → 
  sum_of_edges_squared a b c a₁ b₁ c₁ = 3 * real.sqrt(3) * Δ :=
sorry

theorem part_b (a b c a₁ b₁ c₁ Δ : ℝ) :
  sum_of_edges_squared a b c a₁ b₁ c₁ ≥ 3 * real.sqrt(3) * Δ 
    + 1/2 * ((a + a₁ - b - b₁)^2 + (a + a₁ - c - c₁)^2 + (b + b₁ - c - c₁)^2)
    + 3/4 * ((a - a₁)^2 + (b - b₁)^2 + (c - c₁)^2) := 
sorry

end part_a_part_b_l101_101337


namespace coefficient_x4_expansion_l101_101619

theorem coefficient_x4_expansion :
  let expr := (2 * x ^ 2 - 1 / x) ^ 5
  ∃ (c : ℤ), c = 80 ∧ ∃ (t : ℕ → ℝ) (r : ℕ),
    t r ≠ 0 ∧
    expr.expandCoef r = c ∧
    expr.expandPower r = 4 :=
sorry

end coefficient_x4_expansion_l101_101619


namespace functional_equation_solution_l101_101016

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 / (1 + a * x)

theorem functional_equation_solution (a : ℝ) (x y : ℝ)
  (ha : 0 < a) (hx : 0 < x) (hy : 0 < y) :
  f a x * f a (y * f a x) = f a (x + y) :=
sorry

end functional_equation_solution_l101_101016


namespace minimal_sum_of_matrix_elements_l101_101174

theorem minimal_sum_of_matrix_elements (a b c d : ℤ) 
    (h1 : a ≠ 0) 
    (h2 : b ≠ 0) 
    (h3 : c ≠ 0) 
    (h4 : d ≠ 0) 
    (h5 : (Matrix.vecCons (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
                         (Matrix.vecCons (Matrix.vecCons c (Matrix.vecCons d Matrix.vecEmpty))
                         (Matrix.vecCons 0 (Matrix.vecCons 0 Matrix.vecEmpty))))
          ^2 = 
    (Matrix.vecCons (Matrix.vecCons 12 (Matrix.vecCons 0 Matrix.vecEmpty))
                   (Matrix.vecCons (Matrix.vecCons 0 (Matrix.vecCons 12 Matrix.vecEmpty))
                   (Matrix.vecCons 0 (Matrix.vecCons 0 Matrix.vecEmpty))))) :
    ∃ a b c d, |a| + |b| + |c| + |d| = 10 :=
sorry

end minimal_sum_of_matrix_elements_l101_101174


namespace coin_flip_probability_l101_101974

-- Define the required variables and conditions
def n : ℕ := 3 -- number of flips
def k : ℕ := 2 -- number of desired heads
def p : ℝ := 0.5 -- probability of getting heads

-- Binomial coefficient calculation
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement of the problem
theorem coin_flip_probability :
  (binomial_coeff n k) * (p ^ k) * ((1 - p) ^ (n - k)) = 0.375 :=
by
  -- Proof goes here
  sorry

end coin_flip_probability_l101_101974


namespace num_not_factorial_tails_lt_2500_l101_101467

-- Definition of the function f(m)
def f (m : ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625 + m / 78125 + m / 390625 + m / 1953125

-- The theorem stating the main result
theorem num_not_factorial_tails_lt_2500 : 
  let count_non_tails := ∑ k in finset.range 2500, if ∀ m, f m ≠ k then 1 else 0
  in count_non_tails = 500 :=
by
  sorry

end num_not_factorial_tails_lt_2500_l101_101467


namespace melanie_balloons_l101_101636

theorem melanie_balloons (joan_balloons melanie_balloons total_balloons : ℕ)
  (h_joan : joan_balloons = 40)
  (h_total : total_balloons = 81) :
  melanie_balloons = total_balloons - joan_balloons :=
by
  sorry

end melanie_balloons_l101_101636


namespace max_dance_pairs_l101_101128

-- Define the capacities of boys and girls in the dance ensemble.
def boys := 8
def girls := 16

-- Statement: there exists a maximum number of pairs such that in each pair,
-- at least one of the partners is not part of any other pair.
theorem max_dance_pairs :
  ∃ (max_pairs : ℕ), max_pairs = 22 ∧
  (∀ (pairs : Π (i : ℕ), (i < max_pairs) → (ℕ × ℕ)), 
    (∀ i, (i < max_pairs) → let (b, g) := pairs i (by linarith) 
                            in b ≠ 0 ∧ g ≠ 0 →
    (b ≠ b' ∧ g ≠ g' ∀ j, j ≠ i → pairs j (by linarith) = (b', g'))) → 
  ) :=
begin
  sorry
end

end max_dance_pairs_l101_101128


namespace sum_of_divisors_37_l101_101313

theorem sum_of_divisors_37 : ∑ d in (finset.filter (λ d, 37 % d = 0) (finset.range 38)), d = 38 :=
by {
  sorry
}

end sum_of_divisors_37_l101_101313


namespace small_branches_count_l101_101358

theorem small_branches_count (x : ℕ) (h : x^2 + x + 1 = 91) : x = 9 := 
  sorry

end small_branches_count_l101_101358


namespace carly_shipping_cost_l101_101417

noncomputable def total_shipping_cost (flat_fee cost_per_pound weight : ℝ) : ℝ :=
flat_fee + cost_per_pound * weight

theorem carly_shipping_cost : 
  total_shipping_cost 5 0.80 5 = 9 :=
by 
  unfold total_shipping_cost
  norm_num

end carly_shipping_cost_l101_101417


namespace number_of_pins_l101_101186

def divisible_by_2019_squared (n : Nat) : Prop :=
  ∃ x : Nat, n = x ^ 2 ∧ x = 2019

theorem number_of_pins 
  (A B : Finset (Set (ℝ × ℝ)))
  (h₁ : ¬ A.empty)
  (h₂ : ¬ B.empty)
  (hA : A.card = 2019^2)
  (hB : B.card = 2019^2)
  (hA_areas : ∀ a ∈ A, measure_theory.measure_space.volume a = 1)
  (hB_areas : ∀ b ∈ B, measure_theory.measure_space.volume b = 1)
  (divisibility : divisible_by_2019_squared (A.card))
  : ∃ (f : A → B), function.bijective f ∧ ∀ a ∈ A, ∃ (p : ℝ × ℝ), p ∈ a ∧ p ∈ f a :=
begin
  sorry
end

end number_of_pins_l101_101186


namespace triangle_AEB_area_l101_101609

def Point := ℝ × ℝ
def Rectangle (A B C D : Point) : Prop := 
  A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ D.2 = C.2 ∧ 
  (B.1 - A.1) = 8 ∧ (C.2 - B.2) = 4

def OnSegment (P D C : Point) : Prop :=
  (min D.1 C.1 ≤ P.1 ∧ P.1 ≤ max D.1 C.1) ∧ 
  (min D.2 C.2 ≤ P.2 ∧ P.2 ≤ max D.2 C.2)

def Intersects (AF BG : Set Point) (E : Point) : Prop :=
  E ∈ AF ∧ E ∈ BG

def length (P Q : Point) : ℝ :=
  sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

def TriangleArea (A B C : Point) : ℝ :=
  abs (A.1*(B.2 - C.2) + B.1*(C.2 - A.2) + C.1*(A.2 - B.2)) / 2

theorem triangle_AEB_area (A B C D F G E : Point) (hRect : Rectangle A B C D)
  (hPointF : OnSegment F D C) (hPointG : OnSegment G D C)
  (hDF : length D F = 2) (hGC : length G C = 3)
  (hIntersects : Intersects ({ x : Point | ∃ l : ℝ, x = (l * F.1 + (1-l) * A.1, l * F.2 + (1-l) * A.2) } : Set Point) 
  ({ x : Point | ∃ l : ℝ, x = (l * G.1 + (1-l) * B.1, l * G.2 + (1-l) * B.2) } : Set Point) E) :
  TriangleArea A E B = 6 := sorry

end triangle_AEB_area_l101_101609


namespace geometric_sequence_sum_l101_101258

theorem geometric_sequence_sum
  (a r : ℝ)
  (h1 : a * (1 - r ^ 3000) / (1 - r) = 500)
  (h2 : a * (1 - r ^ 6000) / (1 - r) = 950) :
  a * (1 - r ^ 9000) / (1 - r) = 1355 :=
sorry

end geometric_sequence_sum_l101_101258


namespace non_factorial_tails_lt_2500_l101_101432

-- Define the function f(m)
def f (m: ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625

-- Define what it means to be a factorial tail
def is_factorial_tail (n: ℕ) : Prop :=
  ∃ m : ℕ, f(m) = n

-- Define the statement to be proved
theorem non_factorial_tails_lt_2500 : 
  (finset.range 2500).filter (λ n, ¬ is_factorial_tail n).card = 499 :=
by
  sorry

end non_factorial_tails_lt_2500_l101_101432


namespace f_monotone_decreasing_f_odd_function_f_range_on_interval_l101_101910

-- Defining the function f and conditions
variable (f : ℝ → ℝ)

axiom additivity (s t : ℝ) : f (s + t) = f s + f t
axiom negativity_pos (x : ℝ) (hx : x > 0) : f x < 0
axiom specific_value : f 3 = -3

-- Prove that f is monotonically decreasing on ℝ
theorem f_monotone_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₂ < f x₁ := 
by
  sorry

-- Prove that f is an odd function
theorem f_odd_function : ∀ x : ℝ, f (-x) = -f x := 
by
  sorry

-- Prove the range of f over an interval [m, n] where m, n ∈ ℤ and m > 0
theorem f_range_on_interval (m n : ℤ) (hm : 0 < m) : set.range (λ x : ℝ, if m ≤ x ∧ x ≤ n then f x else 0) = set.Icc (f n) (f m) :=
by
  sorry

end f_monotone_decreasing_f_odd_function_f_range_on_interval_l101_101910


namespace sequence_increasing_l101_101042

noncomputable def sequence (x : ℕ → ℝ) : Prop :=
∀ n, x (n + 1) = (x n * (x n ^ 2 + 3)) / (3 * (x n ^ 2) + 1)

theorem sequence_increasing (x : ℕ → ℝ) (h1 : x 1 > 0) (h2 : x 1 ≠ 1) (h_seq : sequence x) :
  (¬ ∀ n > 0, x n < x (n + 1)) ↔ ∃ n > 0, x n ≥ x (n + 1) :=
sorry

end sequence_increasing_l101_101042


namespace area_quadrilateral_BEIH_l101_101856

def rectangle_ABCD (A B C D : ℝ × ℝ) : Prop := 
  (A = (0, 2)) ∧ 
  (B = (0, 0)) ∧ 
  (C = (3, 0)) ∧ 
  (D = (3, 2)) ∧ 
  (dist A B = 3) ∧ 
  (dist B C = 2)

def point_E (A B : ℝ × ℝ) : ℝ × ℝ := (0, 4/3)

def point_F (B C : ℝ × ℝ) : ℝ × ℝ := (3/4, 0)

def line_AF (A F : ℝ × ℝ) (x : ℝ) := -8/3 * x + 2

def line_DE (D E : ℝ × ℝ) (x : ℝ) := 2/3 * x

def intersection_AF_DE (A F D E : ℝ × ℝ) : ℝ × ℝ := (2/3, 2/9)  -- Point I

def line_BD (B D : ℝ × ℝ) (x : ℝ) := 2/3 * x

def intersection_AF_BD (A F B D : ℝ × ℝ) : ℝ × ℝ := (6/5, 4/5)  -- Point H

theorem area_quadrilateral_BEIH :
  ∀ (A B C D E F I H : ℝ × ℝ),
  rectangle_ABCD A B C D → 
  E = point_E A B → 
  F = point_F B C → 
  I = intersection_AF_DE A F D E → 
  H = intersection_AF_BD A F B D → 
  (∃ (BEIH : ℝ), BEIH = 8 / 15) :=
by
  intros
  sorry

end area_quadrilateral_BEIH_l101_101856


namespace evaluate_98_times_98_mental_calculation_l101_101488

theorem evaluate_98_times_98_mental_calculation :
  (98 : ℕ) * 98 = 9604 :=
by
  calc
    98 * 98 = (100 - 2) * (100 - 2) : by rw [nat.sub_self]
          ... = 100 * 100 - 2 * 100 * 2 + 2 * 2 : by rw [nat.sub_self]
          ... = 10000 - 400 + 4 : by norm_num
          ... = 9604 : by norm_num

end evaluate_98_times_98_mental_calculation_l101_101488


namespace milk_for_flour_l101_101031

theorem milk_for_flour (milk flour use_flour : ℕ) (h1 : milk = 75) (h2 : flour = 300) (h3 : use_flour = 900) : (use_flour/flour * milk) = 225 :=
by sorry

end milk_for_flour_l101_101031


namespace complex_number_in_third_quadrant_l101_101939

def z : ℂ := 1 + complex.i

def w : ℂ := 5 / (z * z) - z

theorem complex_number_in_third_quadrant : w.re < 0 ∧ w.im < 0 := by
  sorry

end complex_number_in_third_quadrant_l101_101939


namespace least_positive_integer_property_l101_101894

theorem least_positive_integer_property : 
  ∃ (n d : ℕ) (p : ℕ) (h₁ : 1 ≤ d) (h₂ : d ≤ 9) (h₃ : p ≥ 2), 
  (10^p * d = 24 * n) ∧ (∃ k : ℕ, (n = 100 * 10^(p-2) / 3) ∧ (900 = 8 * 10^p + 100 / 3 * 10^(p-2))) := sorry

end least_positive_integer_property_l101_101894


namespace area_triangle_BEC_l101_101603

-- Definition of the trapezoid ABCD and given conditions
variables {A B C D E : Type}
variables [Trapezoid A B C D]
variables (AD DC : Real) (AB : Real)
variable [Perpendicular AD DC]
variable (BE : Real)
variable [Parallel BE AD]
variable (angle_BAD : Real)
variable [right_angle : angle_BAD = 90]

-- Conditions
axiom AD_len : AD = 4
axiom AB_len : AB = 4
axiom DC_len : DC = 8
axiom BE_len : BE = 4
axiom EC_len : EC = 4
axiom E_on_DC : E ∈ (DC)

-- The proof problem
theorem area_triangle_BEC : triangle_area B E C = 8 := sorry

end area_triangle_BEC_l101_101603


namespace tangent_line_circle_b_l101_101978

theorem tangent_line_circle_b (b : ℝ) : 
  (∃ b, (3 * 1 + 4 * 1 - b) / real.sqrt (3^2 + 4^2) = 1) ↔ (b = 2 ∨ b = 12) := 
by
  sorry

end tangent_line_circle_b_l101_101978


namespace proof_problem_l101_101412

def is_prime (n : ℕ) : Prop := nat.prime n

def third_smallest_prime (p : ℕ) : Prop :=
  is_prime p ∧ ∃ (a b : ℕ), a = 2 ∧ b = 3 ∧ p > a ∧ p > b ∧
  (∀ (q : ℕ), q < p → q = a ∨ q = b → is_prime q → true)

def fourth_smallest_prime (p : ℕ) : Prop :=
  is_prime p ∧ ∃ (a b c : ℕ), a = 2 ∧ b = 3 ∧ c = 5 ∧ p > a ∧ p > b ∧ p > c ∧
  (∀ (q : ℕ), q < p → q = a ∨ q = b ∨ q = c → is_prime q → true)

theorem proof_problem :
  ∃ (a b : ℕ),
    third_smallest_prime a ∧
    fourth_smallest_prime b ∧
    (a ^ 2) ^ 3 * b = 109375 :=
by
  sorry

end proof_problem_l101_101412


namespace line_circle_relationship_l101_101534

-- Given conditions
def circleC (x y : ℝ) : Prop := x^2 + y^2 = 25
def pointP : ℝ × ℝ := (4, 3)
def lineL (x y : ℝ) : Prop := ∃ m b : ℝ, y = m * x + b

-- Proof problem statement
theorem line_circle_relationship :
  (lineL pointP.1 pointP.2 ∧ circleC pointP.1 pointP.2) → 
  (∀ x y : ℝ, lineL x y → circleC x y → (∃ u : ℝ, distance (0, 0) (x, y) ≤ 5)) ∨
  (∀ x y : ℝ, lineL x y → circleC x y → (∃ v : ℝ, distance (0, 0) (x, y) = 5)) := 
sorry

end line_circle_relationship_l101_101534


namespace no_polynomial_satisfies_conditions_l101_101083

theorem no_polynomial_satisfies_conditions :
  ∀ (f : polynomial ℝ), (∃ n, polynomial.degree f = n ∧ n ≥ 1) →
  (∀ x, f(x^2) = (f(x))^3) →
  (∀ x, f(f(x)) = (f(x))^2) →
  false :=
by sorry

end no_polynomial_satisfies_conditions_l101_101083


namespace count_non_factorial_tails_lt_2500_l101_101449

def f (m : ℕ) : ℕ := 
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + ... -- Continue as needed

theorem count_non_factorial_tails_lt_2500 : 
  #{ n : ℕ | n < 2500 ∧ ¬ (∃ m : ℕ, f(m) = n) } = 4 :=
by sorry

end count_non_factorial_tails_lt_2500_l101_101449


namespace complement_of_union_correct_l101_101962

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

-- Defining the universal set
def universal_set : Set ℕ := {1, 2, 3, 4, 5}

-- Defining set A
def set_A : Set ℕ := { x | x^2 - 3*x + 2 = 0 }

-- Defining set B
def set_B : Set ℕ := { x | ∃ a ∈ set_A, x = 2 * a}

-- Defining the complement of A ∪ B with respect to U
def complement_of_union : Set ℕ := universal_set \ (set_A ∪ set_B)

-- The theorem to prove
theorem complement_of_union_correct :
  complement_of_union universal_set set_A set_B = {3, 5} :=
sorry

end complement_of_union_correct_l101_101962


namespace Jane_age_l101_101193

theorem Jane_age {n x y : ℕ} (h1 : n = x^3 + 1) (h2 : n = y^2 - 4) (h3 : x^3 - y^2 = -5) : n = 1332 :=
by
  sorry

end Jane_age_l101_101193


namespace volume_apple_juice_correct_l101_101363

noncomputable def volumeAppleJuice
    (container_height : ℝ)
    (container_diameter : ℝ)
    (fill_ratio : ℝ)
    (juice_ratio : ℝ)
    (water_ratio : ℝ)
    : ℝ :=
  let radius := container_diameter / 2
  let volume_cylinder := Real.pi * radius^2 * (container_height * fill_ratio)
  let juice_fraction := juice_ratio / (juice_ratio + water_ratio)
  volume_cylinder * juice_fraction

theorem volume_apple_juice_correct :
    volumeAppleJuice 9 3 (1 / 3) 2 5 ≈ 6.06 :=
by
  sorry

end volume_apple_juice_correct_l101_101363


namespace jackson_grade_l101_101632

open Function

theorem jackson_grade :
  ∃ (grade : ℕ), 
  ∀ (hours_playing hours_studying : ℕ), 
    (hours_playing = 9) ∧ 
    (hours_studying = hours_playing / 3) ∧ 
    (grade = hours_studying * 15) →
    grade = 45 := 
by {
  sorry
}

end jackson_grade_l101_101632


namespace find_m_of_hyperbola_l101_101558

noncomputable def eccen_of_hyperbola (a b : ℝ) : ℝ := Real.sqrt (1 + (b^2) / (a^2))

theorem find_m_of_hyperbola :
  ∃ (m : ℝ), (m > 0) ∧ (eccen_of_hyperbola 2 m = Real.sqrt 3) ∧ (m = 2 * Real.sqrt 2) :=
by
  sorry

end find_m_of_hyperbola_l101_101558


namespace min_max_eq_four_min_abs_sum_eq_six_l101_101926

variables {a b c : ℝ}

theorem min_max_eq_four (h1 : a + b + c = 2) (h2 : a * b * c = 4) : 
  ∃ d, (d = max a (max b c)) ∧ d = 4 :=
begin
  sorry
end

theorem min_abs_sum_eq_six (h1 : a + b + c = 2) (h2 : a * b * c = 4) : 
  ∃ s, (s = |a| + |b| + |c|) ∧ s = 6 :=
begin
  sorry
end

end min_max_eq_four_min_abs_sum_eq_six_l101_101926


namespace right_triangle_shorter_leg_l101_101133

theorem right_triangle_shorter_leg (a b : ℕ) (h : a^2 + b^2 = 25^2) : a = 7 ∨ a = 20 :=
begin
  sorry
end

end right_triangle_shorter_leg_l101_101133


namespace expectation_of_2ξ_plus_1_variance_of_2ξ_plus_1_l101_101065

variable (ξ : ℝ)

-- Given conditions
def E_ξ : ℝ := 3
def D_ξ : ℝ := 4

-- Questions and corresponding correct answers
theorem expectation_of_2ξ_plus_1 : 
  E (2 * ξ + 1) = 7 := by sorry

theorem variance_of_2ξ_plus_1 : 
  D (2 * ξ + 1) = 16 := by sorry

end expectation_of_2ξ_plus_1_variance_of_2ξ_plus_1_l101_101065


namespace people_per_family_l101_101786

theorem people_per_family
  (families : ℕ) (days : ℕ) (towels_per_day : ℕ) (loads : ℕ) (towels_per_load : ℕ)
  (h1 : families = 3) (h2 : days = 7) (h3 : towels_per_day = 1)
  (h4 : towels_per_load = 14) (h5 : loads = 6) :
  (loads * towels_per_load / (days * families)) = 4 := by
  calc
    loads * towels_per_load / (days * families)
        = 6 * 14 / (7 * 3) : by 
        rw [h1, h2, h3, h4, h5]
    ... = 84 / 21 : by norm_num
    ... = 4 : by norm_num

end people_per_family_l101_101786


namespace line_does_not_pass_through_third_quadrant_l101_101702

theorem line_does_not_pass_through_third_quadrant (x y : ℝ) (h : y = -x + 1) :
  ¬(x < 0 ∧ y < 0) :=
sorry

end line_does_not_pass_through_third_quadrant_l101_101702


namespace smallest_positive_value_S_n_l101_101146

theorem smallest_positive_value_S_n
  (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 11 / a 10 < -1) 
  (h3 : ∃ N, ∀ n, S n ≤ S N) : 
  (∃ n, S n > 0 ∧ ∀ m, m < n → S m ≤ 0) → n = 19 :=
sorry

end smallest_positive_value_S_n_l101_101146


namespace total_area_of_triangles_l101_101595

theorem total_area_of_triangles :
    let AB := 12
    let DE := 8 * Real.sqrt 2
    let area_ABC := (1 / 2) * AB * AB
    let area_DEF := (1 / 2) * DE * DE * 2
    area_ABC + area_DEF = 136 := by
  sorry

end total_area_of_triangles_l101_101595


namespace geometric_sequence_sum_l101_101266

noncomputable theory

-- Define the conditions and the problem statement
theorem geometric_sequence_sum (a r : ℝ) (h₁ : a * (1 - r^3000) / (1 - r) = 500) 
  (h₂ : a * (1 - r^6000) / (1 - r) = 950) : 
  a * (1 - r^9000) / (1 - r) = 1355 := 
sorry

end geometric_sequence_sum_l101_101266


namespace sum_of_coefficients_shifted_function_l101_101350

def original_function (x : ℝ) : ℝ :=
  3*x^2 - 2*x + 6

def shifted_function (x : ℝ) : ℝ :=
  original_function (x + 5)

theorem sum_of_coefficients_shifted_function : 
  let a := 3
  let b := 28
  let c := 71
  a + b + c = 102 :=
by
  -- Placeholder for the proof
  sorry

end sum_of_coefficients_shifted_function_l101_101350


namespace find_m_l101_101121

open Set

variable {m : ℝ}
def A := {0, m}
def B := {1, 2}

theorem find_m (h : A ∩ B = {1}) : m = 1 := 
  sorry

end find_m_l101_101121


namespace problem1_problem2_l101_101920

def point (x y : ℝ) := (x, y)

def slope (P Q : ℝ × ℝ) : ℝ := (Q.snd - P.snd) / (Q.fst - P.fst)

def slopes_product_neg_quarter (P Q : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  slope P M * slope Q M = -1 / 4

def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 8 + y^2 / 2 = 1 ∧ x ≠ 2 ∧ x ≠ -2

def triangle_area (P A O : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.fst * (A.snd - O.snd) + A.fst * (O.snd - P.snd) + O.fst * (P.snd - A.snd))

def area_range (A P : ℝ × ℝ) : Prop :=
  0 < triangle_area P A (0,0) ∧ triangle_area P A (0,0) ≤ 2

theorem problem1 : 
  ∀ (P : ℝ × ℝ) (Q M : ℝ × ℝ),
  P = (2, 1) → Q = (-2, -1) →
  slopes_product_neg_quarter P Q M →
  trajectory_equation M.fst M.snd := 
by
  sorry

theorem problem2 :
  ∀ (P A : ℝ × ℝ),
  P = (2,1) →
  ∃ k : ℝ, ∀ l : set (ℝ × ℝ), l = {A | A.snd = k * (A.fst - 2) + 1} →
  A ∈ l →
  trajectory_equation A.fst A.snd →
  area_range A P :=
by
  sorry

end problem1_problem2_l101_101920


namespace length_of_AC_l101_101126

variables {A B C D E : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]

-- Assume segment lengths and bisector property
def triangle_data (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (BD DE EC AB AC : ℕ) :=
  (BD = 2) ∧ (DE = 3) ∧ (EC = 9) ∧ (AB = 10) ∧ (AD bisects ∠BAC)

noncomputable def angle_bisector_theorem (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (BD DE EC AB AC BD_DC_ratio AB_AC_ratio : ℕ) :=
  (BD / (DE + EC) = AB / AC) ∧ (BD = 2) ∧ (DE = 3) ∧ (EC = 9) ∧ (AB = 10) → (AC = 60)

theorem length_of_AC {A B C D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (BD DE EC AB AC : ℕ) (h : triangle_data A B C D E BD DE EC AB AC) :
  AC = 60 :=
  by sorry

end length_of_AC_l101_101126


namespace nonneg_integer_solutions_l101_101885

open Nat

theorem nonneg_integer_solutions (x y : ℕ) : 
  (x! + 2^y = (x + 1)! ) ↔ ((x = 1 ∧ y = 0) ∨ (x = 2 ∧ y = 1)) := by
  sorry

end nonneg_integer_solutions_l101_101885


namespace part1_part2_part3_l101_101073

noncomputable def f (x : ℝ) := (4^x - 2^x)

def range_f : Set ℝ := {y | ∃ x, x ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ y = f x}

def g (a : ℝ) : ℝ := (a^2 - a) / 2

def D : Set ℝ := Ioo 1 2

theorem part1 : range_f = Icc (-1 / 4 : ℝ) 2 := 
sorry

theorem part2 (s t : ℝ) (h1 : f s + f t = 0) (a = 2^s + 2^t) (b = 2^(s+t)) :
  b = g a ∧ a ∈ D :=
sorry

theorem part3 (x1 m : ℝ) (h1 : x1 ∈ D) (h2 : ∃ x2, x2 ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ g x1 = f x2 + m) :
  m ∈ Icc (-1 : ℝ) (1 / 4) :=
sorry

end part1_part2_part3_l101_101073


namespace not_factorial_tails_l101_101464

noncomputable def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) + ...

theorem not_factorial_tails :
  let n := 2500 in ∃ (k : ℕ), k = 500 ∧ ∀ m < n, ¬(f m = k) :=
by {
  sorry
}

end not_factorial_tails_l101_101464


namespace angle_E_measure_l101_101378

theorem angle_E_measure (H F G E : ℝ) 
  (h1 : E = 2 * F) (h2 : F = 2 * G) (h3 : G = 1.25 * H) 
  (h4 : E + F + G + H = 360) : E = 150 := by
  sorry

end angle_E_measure_l101_101378


namespace weight_of_7th_person_l101_101776

-- Defining the constants and conditions
def num_people_initial : ℕ := 6
def avg_weight_initial : ℝ := 152
def num_people_total : ℕ := 7
def avg_weight_total : ℝ := 151

-- Calculating the total weights from the given average weights
def total_weight_initial := num_people_initial * avg_weight_initial
def total_weight_total := num_people_total * avg_weight_total

-- Theorem stating the weight of the 7th person
theorem weight_of_7th_person : total_weight_total - total_weight_initial = 145 := 
sorry

end weight_of_7th_person_l101_101776


namespace trigonometric_expression_identity_l101_101230

theorem trigonometric_expression_identity :
  (2 * Real.sin (100 * Real.pi / 180) - Real.cos (70 * Real.pi / 180)) / Real.cos (20 * Real.pi / 180)
  = 2 * Real.sqrt 3 - 1 :=
sorry

end trigonometric_expression_identity_l101_101230


namespace Q_at_2_l101_101723

-- Define the polynomial Q(x)
def Q (x : ℚ) : ℚ := x^4 - 8 * x^2 + 16

-- Define the properties of Q(x)
def is_special_polynomial (P : (ℚ → ℚ)) : Prop := 
  degree P = 4 ∧ leading_coeff P = 1 ∧ P.is_root(√3 + √7)

-- Problem: Prove Q(2) = 0 given the polynomial Q meets the conditions.
theorem Q_at_2 (Q : ℚ → ℚ) (h1 : degree Q = 4) (h2 : leading_coeff Q = 1) (h3 : is_root Q (√3 + √7)) : 
  Q 2 = 0 := by
  sorry

end Q_at_2_l101_101723


namespace min_perimeter_triangle_l101_101125

theorem min_perimeter_triangle (a b c : ℝ) (cosC : ℝ) :
  a + b = 10 ∧ cosC = -1/2 ∧ c^2 = (a - 5)^2 + 75 →
  a + b + c = 10 + 5 * Real.sqrt 3 :=
by
  sorry

end min_perimeter_triangle_l101_101125


namespace range_of_a_max_area_of_triangle_l101_101954

variable (p a : ℝ) (h : p > 0)

def parabola_eq (x y : ℝ) := y ^ 2 = 2 * p * x
def line_eq (x y : ℝ) := y = x - a
def intersects_parabola (A B : ℝ × ℝ) := parabola_eq p A.fst A.snd ∧ line_eq a A.fst A.snd ∧ parabola_eq p B.fst B.snd ∧ line_eq a B.fst B.snd
def ab_length_le_2p (A B : ℝ × ℝ) := (Real.sqrt ((A.fst - B.fst)^2 + (A.snd - B.snd)^2) ≤ 2 * p)

theorem range_of_a
  (A B : ℝ × ℝ)
  (h_intersects : intersects_parabola a p A B)
  (h_ab_length : ab_length_le_2p p A B) :
  - p / 2 < a ∧ a ≤ - p / 4 := sorry

theorem max_area_of_triangle
  (A B : ℝ × ℝ) (N : ℝ × ℝ)
  (h_intersects : intersects_parabola a p A B)
  (h_ab_length : ab_length_le_2p p A B)
  (h_N : N.snd = 0) :
  ∃ (S : ℝ), S = Real.sqrt 2 * p^2 := sorry

end range_of_a_max_area_of_triangle_l101_101954


namespace sum_of_geometric_sequence_first_9000_terms_l101_101252

noncomputable def geomSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_of_geometric_sequence_first_9000_terms (a r : ℝ) (h1 : geomSum a r 3000 = 500) (h2 : geomSum a r 6000 = 950) :
  geomSum a r 9000 = 1355 :=
by
  sorry

end sum_of_geometric_sequence_first_9000_terms_l101_101252


namespace math_problem_l101_101015

noncomputable def f (x : ℝ) : ℝ := 3 * real.sqrt 3 / (3 + real.sqrt 3 * x)
noncomputable def g (x : ℝ) : ℝ := -27 / (3 + real.sqrt 3 * x)^3

theorem math_problem (x : ℝ) (h₀ : 0 ≤ x) :
  let k := real.sqrt 3 in
  (∀ x, 0 ≤ x → f(x) = k + ∫ t in 0..x, g(t) / f(t)) ∧
  (∀ x, 0 ≤ x → g(x) = -k - ∫ t in 0..x, f(t) * g(t)) ∧
  f(0) = k ∧
  (differentiable ℝ f) ∧
  f' 0 = -k^2 / 3 ∧
  (∀ x, 0 ≤ x → f(x) ≠ 0) :=
by 
  sorry

end math_problem_l101_101015


namespace orthocenter_distance_sum_eq_twice_circumradius_l101_101994

variable (A B C : Type) 
variable [EuclideanGeometry A B C]

theorem orthocenter_distance_sum_eq_twice_circumradius 
  (ABC_triangle : Triangle A B C)
  (orthocenter_H : Orthocenter ABC_triangle)
  (circumcenter_O : Circumcenter ABC_triangle)
  (circumradius_R : real)
  (H_distance_to_vertices : 
    dist orthocenter_H A + dist orthocenter_H B + dist orthocenter_H C = 2 * circumradius_R) : 
  dist orthocenter_H A + dist orthocenter_H B + dist orthocenter_H C = 2 * circumradius_R :=
sorry

end orthocenter_distance_sum_eq_twice_circumradius_l101_101994


namespace Q_at_2_l101_101721

-- Define the polynomial Q(x)
def Q (x : ℚ) : ℚ := x^4 - 8 * x^2 + 16

-- Define the properties of Q(x)
def is_special_polynomial (P : (ℚ → ℚ)) : Prop := 
  degree P = 4 ∧ leading_coeff P = 1 ∧ P.is_root(√3 + √7)

-- Problem: Prove Q(2) = 0 given the polynomial Q meets the conditions.
theorem Q_at_2 (Q : ℚ → ℚ) (h1 : degree Q = 4) (h2 : leading_coeff Q = 1) (h3 : is_root Q (√3 + √7)) : 
  Q 2 = 0 := by
  sorry

end Q_at_2_l101_101721


namespace count_multiples_of_7_with_units_digit_7_less_than_150_l101_101086

-- Definition of the problem's conditions and expected result
theorem count_multiples_of_7_with_units_digit_7_less_than_150 : 
  let multiples_of_7 := { n : Nat // n < 150 ∧ n % 7 = 0 }
  let units_digit_7_multiples := { n : Nat // n < 150 ∧ n % 7 = 0 ∧ n % 10 = 7 }
  List.length (units_digit_7_multiples.members) = 3 :=
by
  sorry

end count_multiples_of_7_with_units_digit_7_less_than_150_l101_101086


namespace measure_angle_B_in_triangle_l101_101627

theorem measure_angle_B_in_triangle (x : ℝ) (h1 : ∠ C = 3 * ∠ A) (h2 : ∠ B = 2 * ∠ A) (h_angle_sum : ∠ A + ∠ B + ∠ C = 180) : 
∠ B = 60 :=
by
  -- Lean only needs to take the conditions and the conclusion directly.
  sorry

end measure_angle_B_in_triangle_l101_101627


namespace find_m_if_z_is_pure_imaginary_l101_101578

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem find_m_if_z_is_pure_imaginary (m : ℝ) (z : ℂ) (i : ℂ) (h_i_unit : i^2 = -1) (h_z : z = (1 + i) / (1 - i) + m * (1 - i)) :
  is_pure_imaginary z → m = 0 := 
by
  sorry

end find_m_if_z_is_pure_imaginary_l101_101578


namespace profit_percentage_l101_101829

theorem profit_percentage (initial_value : ℝ) (percentage_lost : ℝ) (overall_loss : ℝ) (expected_profit : ℝ) :
  initial_value = 100 →
  percentage_lost = 0.5 →
  overall_loss = 0.45 →
  let remaining_value := initial_value * (1 - percentage_lost) in
  let loss_value := initial_value * overall_loss in
  let required_value := initial_value - loss_value in
  let equation := remaining_value * (1 + expected_profit / 100) = required_value in
  equation →
  expected_profit = 10 := 
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end profit_percentage_l101_101829


namespace rectangle_area_error_percent_l101_101137

theorem rectangle_area_error_percent 
  (L W : ℝ)
  (hL: L > 0)
  (hW: W > 0) :
  let original_area := L * W
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let measured_area := measured_length * measured_width
  let error := measured_area - original_area
  let error_percent := (error / original_area) * 100
  error_percent = 0.7 := by
  let original_area := L * W
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let measured_area := measured_length * measured_width
  let error := measured_area - original_area
  let error_percent := (error / original_area) * 100
  sorry

end rectangle_area_error_percent_l101_101137


namespace rajesh_work_completion_l101_101187

noncomputable def work_completion_time_by_rajesh (W : ℝ) : ℝ :=
  let mahesh_work_rate := W / 50
  let suresh_work_rate := W / 60
  let work_by_mahesh := 20 * mahesh_work_rate
  let remaining_work := W - work_by_mahesh
  let combined_work_rate := remaining_work / 30
  let rajesh_work_rate := combined_work_rate - suresh_work_rate in
  W / rajesh_work_rate

theorem rajesh_work_completion (W : ℝ) (H : W > 0) :
  work_completion_time_by_rajesh W = 300 :=
by
  -- Detailed proof goes here
  sorry

end rajesh_work_completion_l101_101187


namespace fk_monotonically_increasing_l101_101509

noncomputable def f (x : ℝ) : ℝ := 2^(-|x|)

noncomputable def f_K (K : ℝ) (x : ℝ) : ℝ :=
if f x <= K then f x else K

theorem fk_monotonically_increasing :
  let K := 1 / 2,
      f_k := λ x, f_K K x in
  ∀ x1 x2 : ℝ, x1 < x2 → x1 < -1 → x2 < -1 → f_k x1 ≤ f_k x2 :=
by
  let K := 1 / 2
  let f_k := λ x, f_K K x
  sorry

end fk_monotonically_increasing_l101_101509


namespace total_cost_of_deck_l101_101290

def num_rare_cards : ℕ := 19
def num_uncommon_cards : ℕ := 11
def num_common_cards : ℕ := 30

def cost_per_rare_card : ℕ := 1
def cost_per_uncommon_card : ℝ := 0.50
def cost_per_common_card : ℝ := 0.25

def cost_of_rare_cards : ℕ := num_rare_cards * cost_per_rare_card
def cost_of_uncommon_cards : ℝ := num_uncommon_cards * cost_per_uncommon_card
def cost_of_common_cards : ℝ := num_common_cards * cost_per_common_card

def total_cost : ℝ := cost_of_rare_cards + cost_of_uncommon_cards + cost_of_common_cards

theorem total_cost_of_deck : total_cost = 32 := by
  -- We will need to convert integers to real numbers for this addition
  have h1 : (cost_of_rare_cards : ℝ) = 19 := by norm_cast
  rw [h1]
  have h2 : (num_uncommon_cards: ℝ) * cost_per_uncommon_card = 5.5 := by norm_num
  have h3 : (num_common_cards: ℝ) * cost_per_common_card = 7.5 := by norm_num
  calc
    (19 : ℝ) + 5.5 + 7.5 = 32 := by norm_num

end total_cost_of_deck_l101_101290


namespace value_of_n_l101_101761

theorem value_of_n (n : ℝ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : n = 3.5 :=
by sorry

end value_of_n_l101_101761


namespace geometric_sequence_sum_l101_101255

theorem geometric_sequence_sum
  (a r : ℝ)
  (h1 : a * (1 - r ^ 3000) / (1 - r) = 500)
  (h2 : a * (1 - r ^ 6000) / (1 - r) = 950) :
  a * (1 - r ^ 9000) / (1 - r) = 1355 :=
sorry

end geometric_sequence_sum_l101_101255


namespace percentage_reduction_in_women_l101_101202

theorem percentage_reduction_in_women
    (total_people : Nat) (men_in_office : Nat) (women_in_office : Nat)
    (men_in_meeting : Nat) (women_in_meeting : Nat)
    (even_men_women : men_in_office = women_in_office)
    (total_people_condition : total_people = men_in_office + women_in_office)
    (meeting_condition : total_people = 60)
    (men_meeting_condition : men_in_meeting = 4)
    (women_meeting_condition : women_in_meeting = 6) :
    ((women_in_meeting * 100) / women_in_office) = 20 :=
by
  sorry

end percentage_reduction_in_women_l101_101202


namespace num_5_digit_numbers_is_six_l101_101356

-- Define that we have the digits 2, 45, and 68
def digits : List Nat := [2, 45, 68]

-- Function to generate all permutations of given digits
def permute : List Nat → List (List Nat)
| [] => [[]]
| (x::xs) =>
  List.join (List.map (λ ys =>
    List.map (λ zs => x :: zs) (permute xs)) (permute xs))

-- Calculate the number of distinct 5-digit numbers
def numberOf5DigitNumbers : Int := 
  (permute digits).length

-- Theorem to prove the number of distinct 5-digit numbers formed
theorem num_5_digit_numbers_is_six : numberOf5DigitNumbers = 6 := by
  sorry

end num_5_digit_numbers_is_six_l101_101356


namespace BD_length_is_40_l101_101743

-- Definitions for the conditions
structure Trapezoid (A B C D O P : Type) :=
(parallel_AB_CD : A // AB → Prop)
(equal_AB_CD : ∀{a b : AB}, a = b → b = a → Prop)
(equal_BC_AD : ∀{a d : BC AD}, a = d → Prop)
(intersection_AC_BD : ∀ (a b c d o : Point), intersect AC BD o → Prop)
(centroid_BOD : Point)
(length_OP : length OP = 10)

-- The main statement we want to prove
theorem BD_length_is_40 (A B C D O P : Point) 
    (trapezoid : Trapezoid A B C D O P) :
  length BD = 40 := sorry

end BD_length_is_40_l101_101743


namespace Annette_more_than_Sara_l101_101402

variable (A C S : ℕ)

-- Define the given conditions as hypotheses
def Annette_Caitlin_weight : Prop := A + C = 95
def Caitlin_Sara_weight : Prop := C + S = 87

-- The theorem to prove: Annette weighs 8 pounds more than Sara
theorem Annette_more_than_Sara (h1 : Annette_Caitlin_weight A C)
                               (h2 : Caitlin_Sara_weight C S) :
  A - S = 8 := by
  sorry

end Annette_more_than_Sara_l101_101402


namespace no_triangle_sides_exist_l101_101878

theorem no_triangle_sides_exist (x y z : ℝ) (h_triangle_sides : x > 0 ∧ y > 0 ∧ z > 0)
  (h_triangle_inequality : x < y + z ∧ y < x + z ∧ z < x + y) :
  x^3 + y^3 + z^3 ≠ (x + y) * (y + z) * (z + x) :=
sorry

end no_triangle_sides_exist_l101_101878


namespace S10_equals_21_l101_101959

variable {a : ℕ → ℝ}

-- The given conditions
def initial_condition : a 1 = 3 := by sorry
def recursive_relation (n : ℕ) (h : 2 ≤ n) : a (n - 1) + a n + a (n + 1) = 6 := by sorry

-- Definition of the partial sum
def S (n : ℕ) : ℝ := (Finset.range n).sum a

-- The proof goal
theorem S10_equals_21 : S 10 = 21 := by sorry

end S10_equals_21_l101_101959


namespace equal_parts_fraction_l101_101430

theorem equal_parts_fraction (wire_length : ℝ) (parts : ℕ) (fraction : ℝ) 
  (h1 : wire_length = 4/5) (h2 : parts = 3) (h3 : fraction = 1/3) : 
  let total_length := wire_length / parts in 
  fraction = total_length / wire_length :=
by
  sorry

end equal_parts_fraction_l101_101430


namespace value_of_m_l101_101113

theorem value_of_m (m x : ℝ) (h : x = 3) (h_eq : 3 * m - 2 * x = 6) : m = 4 := by
  -- Given x = 3
  subst h
  -- Now we have to show m = 4
  sorry

end value_of_m_l101_101113


namespace sam_milk_amount_l101_101484

variable (initial_milk : ℚ) (rachel_fraction : ℚ) (sam_fraction : ℚ)

def milk_rachel_drinks := rachel_fraction * initial_milk
def milk_remaining := initial_milk - milk_rachel_drinks
def milk_sam_drinks := sam_fraction * milk_remaining

theorem sam_milk_amount :
  initial_milk = 3/4 ∧
  rachel_fraction = 1/2 ∧
  sam_fraction = 1/3 →
  milk_sam_drinks initial_milk rachel_fraction sam_fraction = 1/8 := by
    intro h
    cases h with h1 h2
    cases h2 with h3 h4
    rw [h1, h3, h4]
    dsimp [milk_rachel_drinks, milk_remaining, milk_sam_drinks]
    simp
    norm_num,
    sorry

end sam_milk_amount_l101_101484


namespace deck_cost_correct_l101_101285

def rare_cards := 19
def uncommon_cards := 11
def common_cards := 30
def rare_card_cost := 1
def uncommon_card_cost := 0.5
def common_card_cost := 0.25

def total_deck_cost := rare_cards * rare_card_cost + uncommon_cards * uncommon_card_cost + common_cards * common_card_cost

theorem deck_cost_correct : total_deck_cost = 32 :=
  by
  -- The proof would go here.
  sorry

end deck_cost_correct_l101_101285


namespace sqrt_factorial_mul_self_eq_sqrt_factorial_mul_self_value_l101_101758

theorem sqrt_factorial_mul_self_eq :
  sqrt ((5!) * (5!)) = 5! :=
by sorry

theorem sqrt_factorial_mul_self_value :
  sqrt ((5!) * (5!)) = 120 :=
by {
  rw sqrt_factorial_mul_self_eq,
  norm_num,
  exact rfl,
  sorry
}

end sqrt_factorial_mul_self_eq_sqrt_factorial_mul_self_value_l101_101758


namespace martha_blue_butterflies_l101_101667

variables (B Y : Nat)

theorem martha_blue_butterflies (h_total : B + Y + 5 = 11) (h_twice : B = 2 * Y) : B = 4 :=
by
  sorry

end martha_blue_butterflies_l101_101667


namespace number_of_equidistant_planes_l101_101918

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def areNonCoplanar (A B C D : Point3D) : Prop := 
  -- A condition to check if four points are not coplanar.
  ∃ (volume : ℝ), volume ≠ 0
    ∧ volume = det ![
          ![(B.x - A.x), (B.y - A.y), (B.z - A.z)],
          ![(C.x - A.x), (C.y - A.y), (C.z - A.z)],
          ![(D.x - A.x), (D.y - A.y), (D.z - A.z)]
        ]

def equidistantPlanesCount (A B C D : Point3D) (nonCoplanar : areNonCoplanar A B C D) : ℕ :=
  7

theorem number_of_equidistant_planes (A B C D : Point3D) : 
  areNonCoplanar A B C D → equidistantPlanesCount A B C D (areNonCoplanar A B C D) = 7 := 
by 
  intros nonCoplanar
  exact rfl

end number_of_equidistant_planes_l101_101918


namespace max_knights_l101_101010

theorem max_knights (people : Fin 10 → ℕ) (knight : Fin 10 → Prop) (liar : Fin 10 → Prop)
  (h1 : ∀ i, knight i ↔ (people i > i + 1)) 
  (h2 : ∀ i, liar i ↔ ¬ (people i > i + 1)) 
  (h_knight_liar : ∀ i, knight i ∨ liar i) 
  (h_unique_knight : ∀ i j, knight i → knight j → i = j) 
  (h_unique_liar : ∀ i j, liar i → liar j → i = j) :
  (∑ i, if knight i then 1 else 0) = 8 :=
sorry

end max_knights_l101_101010


namespace fraction_uninterested_students_interested_l101_101409

theorem fraction_uninterested_students_interested 
  (students : Nat)
  (interest_ratio : ℚ)
  (express_interest_ratio_if_interested : ℚ)
  (express_disinterest_ratio_if_not_interested : ℚ) 
  (h1 : students > 0)
  (h2 : interest_ratio = 0.70)
  (h3 : express_interest_ratio_if_interested = 0.75)
  (h4 : express_disinterest_ratio_if_not_interested = 0.85) :
  let interested_students := students * interest_ratio
  let not_interested_students := students * (1 - interest_ratio)
  let express_interest_and_interested := interested_students * express_interest_ratio_if_interested
  let not_express_interest_and_interested := interested_students * (1 - express_interest_ratio_if_interested)
  let express_disinterest_and_not_interested := not_interested_students * express_disinterest_ratio_if_not_interested
  let express_interest_and_not_interested := not_interested_students * (1 - express_disinterest_ratio_if_not_interested)
  let not_express_interest_total := not_express_interest_and_interested + express_disinterest_and_not_interested
  let fraction := not_express_interest_and_interested / not_express_interest_total
  fraction = 0.407 := 
by
  sorry

end fraction_uninterested_students_interested_l101_101409


namespace not_factorial_tails_l101_101460

noncomputable def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) + ...

theorem not_factorial_tails :
  let n := 2500 in ∃ (k : ℕ), k = 500 ∧ ∀ m < n, ¬(f m = k) :=
by {
  sorry
}

end not_factorial_tails_l101_101460


namespace find_height_of_pyramid_l101_101223

noncomputable def volume (B h : ℝ) : ℝ := (1/3) * B * h
noncomputable def area_of_isosceles_right_triangle (leg : ℝ) : ℝ := (1/2) * leg * leg

theorem find_height_of_pyramid (leg : ℝ) (h : ℝ) (V : ℝ) (B : ℝ)
  (Hleg : leg = 3)
  (Hvol : V = 6)
  (Hbase : B = area_of_isosceles_right_triangle leg)
  (Hvol_eq : V = volume B h) :
  h = 4 :=
by
  sorry

end find_height_of_pyramid_l101_101223


namespace jerry_spent_l101_101155

theorem jerry_spent :
  ∀ (original_amount spent_amount remaining_amount : ℕ),
    original_amount = 18 →
    remaining_amount = 12 →
    spent_amount = original_amount - remaining_amount →
    spent_amount = 6 :=
by
  intros original_amount spent_amount remaining_amount
  assume h1 : original_amount = 18
  assume h2 : remaining_amount = 12
  assume h3 : spent_amount = original_amount - remaining_amount
  rw [h1, h2] at h3
  exact h3
  sorry

end jerry_spent_l101_101155


namespace percentage_of_number_l101_101972

theorem percentage_of_number (N P : ℝ) (h1 : 0.20 * N = 1000) (h2 : (P / 100) * N = 6000) : P = 120 :=
sorry

end percentage_of_number_l101_101972


namespace sum_divisors_of_37_is_38_l101_101318

theorem sum_divisors_of_37_is_38 (h : Nat.Prime 37) : (∑ d in (Finset.filter (λ d, 37 % d = 0) (Finset.range (37 + 1))), d) = 38 := by
  sorry

end sum_divisors_of_37_is_38_l101_101318


namespace beautiful_fold_probability_l101_101194

-- Define the square and random point F
variable {a : ℝ} -- side length of the square
variable {F : ℝ × ℝ} -- random point F within the square
variable {ABCD : set (ℝ × ℝ)} -- the set representing the square

-- Define the conditions of the problem
def is_square (ABCD : set (ℝ × ℝ)) (a : ℝ) : Prop :=
  ∃ (A B C D : ℝ × ℝ),
  A = (0, 0) ∧ B = (a, 0) ∧ C = (a, a) ∧ D = (0, a) ∧
  ABCD = {p | p.1 >= 0 ∧ p.1 <= a ∧ p.2 >= 0 ∧ p.2 <= a }

def is_on_boundary (F : ℝ × ℝ) (a : ℝ) : Prop :=
  (F.1 = 0 ∨ F.1 = a ∨ F.2 = 0 ∨ F.2 = a) ∧ (0 <= F.1 ∧ F.1 <= a) ∧ (0 <= F.2 ∧ F.2 <= a)

def is_beautiful_fold (F : ℝ × ℝ) (a : ℝ) : Prop :=
  F.1 = a/2 ∨ F.2 = a/2

-- The probability calculation statement
theorem beautiful_fold_probability (ABCD : set (ℝ × ℝ)) (F : ℝ × ℝ) (a : ℝ)
  (h_square : is_square ABCD a)
  (h_point : is_on_boundary F a) :
  ∃ p, p = 1 / 2 ∧
  (∀ F, F ∈ ABCD → ∃ beautiful_fold, beautiful_fold = is_beautiful_fold F a) :=
sorry

end beautiful_fold_probability_l101_101194


namespace math_problem_l101_101941

noncomputable def ellipse_equation (a b : ℝ) (C : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, C (x, y) ↔ x^2 / a^2 + y^2 / b^2 = 1

noncomputable def is_triangle_area (a b c : ℝ) (area : ℝ) : Prop :=
  ∃ b c, a^2 = b^2 + c^2 ∧ (b * 2 * c / 2) = area

theorem math_problem
  (eccentricity : ℝ := (Real.sqrt 6) / 3)
  (area : ℝ := (5 * Real.sqrt 2) / 3)
  (a b c : ℝ)
  (C : ℝ × ℝ → Prop)
  (M : ℝ × ℝ := (-7 / 3, 0)) :
  (is_triangle_area a b c area ∧ (c / a) = eccentricity ∧ a^2 = 5 ∧ b^2 = 5 / 3) →
  ellipse_equation a b C ∧
  (∀ k, (∃ A B : ℝ × ℝ, C A ∧ C B ∧ 
  ((A.1 + B.1) / 2 = -1 / 2) ∧ ((A.2 - B.2) > 0) ∧ (k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3))) ∧
  (∀ A B : ℝ × ℝ, C A ∧ C B ∧ ∃ k, 
  ((A.1 + B.1) / 2 = -1 / 2) ∧ 
  ((A.2 = k * (A.1 + 1)) ∧ (B.2 = k * (B.1 + 1))) ∧ 
  (let MA := (A.1 + 7 / 3, A.2 - 0); MB := (B.1 + 7 / 3, B.2 - 0)
   in (MA.1 * MB.1 + MA.2 * MB.2) = 4 / 9)) :=
sorry

end math_problem_l101_101941


namespace sequence_general_term_l101_101229

theorem sequence_general_term (n : ℕ) (h : n > 0) : 
  ∃ a : ℕ → ℚ, (∀ n, a n = 1 / n) :=
by
  sorry

end sequence_general_term_l101_101229


namespace find_angle_B_l101_101589

noncomputable def cosine_value : ℝ := 13 / 14
noncomputable def side_ratio (a b : ℝ) := 7 * a = 3 * b

theorem find_angle_B (A B : ℝ) (a b c : ℝ) (h1 : cos A = cosine_value) (h2 : side_ratio a b) :
  B = π / 3 ∨ B = 2 * π / 3 :=
by
  sorry

end find_angle_B_l101_101589


namespace problem_statement_l101_101172

variable (A B C D : Type) [Add A] [Div A] [HasCharZeroA] (K L M N O P Q R : A)

def midpoint (P Q : A) := (P + Q) / 2

def centroid (A B C D : A) := (A + B + C + D) / 4

noncomputable def convex_quadrilateral :=
  K = midpoint A B ∧
  L = midpoint B C ∧
  M = midpoint C D ∧
  N = midpoint D A ∧
  O = centroid A B C D ∧
  O = midpoint K M ∧
  O = midpoint L N ∧
  P = midpoint A C ∧
  Q = midpoint B D ∧
  O = midpoint P Q

theorem problem_statement (A B C D : Type) [Add A] [Div A] [HasCharZeroA] (K L M N O P Q : A) :
  convex_quadrilateral A B C D K L M N O P Q →
  O =
  centroid (A B C D) /\
  O = midpoint K M /\
  O = midpoint L N /\
  O = midpoint P Q :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩
  sorry

end problem_statement_l101_101172


namespace tan_sum_identity_l101_101527

noncomputable def sin_α : ℝ := 3/5

noncomputable def α : ℝ := sorry -- we assume the existence of such α

axiom α_in_range : 0 < α ∧ α < π/2

theorem tan_sum_identity : 
  tan (α + π/4) = 7 :=
by
  sorry

end tan_sum_identity_l101_101527


namespace angle_B_measure_range_of_b_l101_101151

variable {A B C : Real}
variable {a b c : Real}

-- Conditions of the problem.
axiom triangle_conditions :
  ∀ {A B C a b c : Real},
    a = 1 - c →
    B = π / 3 →
    ∃ k: Real, ∃ k': Real, cos C + (cos A - sqrt 3 * sin A) * cos B = 0

-- Questions to prove.
theorem angle_B_measure :
  tan B = sqrt 3 → B = π / 3 :=
by
  sorry

theorem range_of_b (h₁ : a + c = 1) (h₂ : B = π / 3) :
  1 / 2 ≤ b ∧ b < 1 :=
by
  sorry

end angle_B_measure_range_of_b_l101_101151


namespace length_of_room_is_correct_l101_101701

-- We state the conditions as definitions or hypotheses
def width_of_room : ℝ := 3.75
def cost_of_paving : ℝ := 16500
def rate_per_sq_meter : ℝ := 800
def total_area : ℝ := cost_of_paving / rate_per_sq_meter

-- The goal is to prove that the length of the room is 5.5 meters
theorem length_of_room_is_correct : total_area / width_of_room = 5.5 := by
  sorry

end length_of_room_is_correct_l101_101701


namespace truck_tank_percentage_increase_l101_101129

-- Declaration of the initial conditions (as given in the problem)
def service_cost : ℝ := 2.20
def fuel_cost_per_liter : ℝ := 0.70
def num_minivans : ℕ := 4
def num_trucks : ℕ := 2
def total_cost : ℝ := 395.40
def minivan_tank_size : ℝ := 65.0

-- Proof statement with the conditions declared above
theorem truck_tank_percentage_increase :
  ∃ p : ℝ, p = 120 ∧ (minivan_tank_size * (p + 100) / 100 = 143) :=
sorry

end truck_tank_percentage_increase_l101_101129


namespace sin_15_cos_15_eq_one_fourth_l101_101413

theorem sin_15_cos_15_eq_one_fourth : sin (15 * Real.pi / 180) * cos (15 * Real.pi / 180) = 1 / 4 := by
  sorry

end sin_15_cos_15_eq_one_fourth_l101_101413


namespace arrangement_count_six_people_l101_101719

theorem arrangement_count_six_people (A B C D E F : Type) 
  (arrangements : Finset (List (Type))) 
  (total_arrangements : arrangements.card = 720) 
  (valid_arrangements : Finset (List (Type))) 
  (valid_count : valid_arrangements.card = 480) :
  (valid_count = (2 / 3 : ℝ) * total_arrangements) :=
sorry

end arrangement_count_six_people_l101_101719


namespace abs_inequality_solution_l101_101210

theorem abs_inequality_solution (x : ℝ) : |x - 1| + |x - 3| < 8 ↔ -2 < x ∧ x < 6 :=
by sorry

end abs_inequality_solution_l101_101210


namespace distance_point_to_line_l101_101625

-- Define the polar coordinates for point M
def M_polar := (2, Real.pi / 3)

-- Define the polar equation of the line l
def line_l_polar (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = Real.sqrt 2 / 2

-- Define the Cartesian coordinates conversion for point M
def M_cartesian : ℝ × ℝ := (2 * Real.cos (Real.pi / 3), 2 * Real.sin (Real.pi / 3))

-- Define the Cartesian equation of the line l
def line_l_cartesian (x y : ℝ) : Prop := x + y = 1

-- Distance from a point to a line in Cartesian coordinates
def distance_from_point_to_line (x0 y0 A B C : ℝ) : ℝ := abs (A * x0 + B * y0 + C) / Real.sqrt (A^2 + B^2)

-- Statement of the theorem in Lean
theorem distance_point_to_line : 
  let M := M_cartesian in
  let (xM, yM) := M in
  distance_from_point_to_line xM yM 1 1 (-1) = Real.sqrt 6 / 2 :=
by 
  unfold M_cartesian 
  unfold distance_from_point_to_line 
  apply congrArg 
  sorry

end distance_point_to_line_l101_101625


namespace adults_count_l101_101408

theorem adults_count (number_of_children : ℕ) (meal_cost : ℕ) (total_bill : ℕ) :
  (number_of_children = 5) → (meal_cost = 3) → (total_bill = 21) → 
  ∃ A : ℕ, A = 2 :=
by
  intros h1 h2 h3
  have h4 : 5 * meal_cost = 15 := by rw [←h2]; exact rfl
  have h5 : total_bill - 15 = 6 := by rw [←h3, ←h4]; exact rfl
  have h6 : 6 / meal_cost = 2 := by rw [←h2]; exact rfl
  use 2
  exact h6


end adults_count_l101_101408


namespace find_a_plus_b_l101_101546

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

def xi_values := {1, 2, 3}
def P (k : ℝ) := a * k + b
def E_xi := ∑ k in xi_values, k * P k

theorem find_a_plus_b
  (h1 : E_xi = 7 / 3)
  (h2 : ∑ k in xi_values, P k = 1) :
  a + b = 1 / 6 :=
by
  sorry

end find_a_plus_b_l101_101546


namespace find_dot_product_ad_l101_101165

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c d : V)

-- Definitions based on conditions
def unit_vectors : Prop := 
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥c∥ = 1 ∧ ∥d∥ = 1

def dot_products : Prop := 
  inner a b = -1/7 ∧ 
  inner a c = -1/7 ∧ 
  inner b c = -1/7 ∧ 
  inner b d = -1/7 ∧ 
  inner c d = -1/7

-- Problem statement
theorem find_dot_product_ad (h_unit: unit_vectors a b c d) (h_dot: dot_products a b c d) : 
  inner a d = -37/42 :=
sorry

end find_dot_product_ad_l101_101165


namespace concyclic_of_tangent_circles_l101_101164

noncomputable def concyclic_points (Γ1 Γ2 Γ3 Γ4 : Circle)
  (A : Point) (B : Point) (C : Point) (D : Point) : Prop :=
  Γ1.tangent Γ2 A ∧
  Γ2.tangent Γ3 B ∧
  Γ3.tangent Γ4 C ∧
  Γ4.tangent Γ1 D ∧ 
  Circle.concyclic A B C D

theorem concyclic_of_tangent_circles
  (Γ1 Γ2 Γ3 Γ4 : Circle)
  (A : Point) (B : Point) (C : Point) (D : Point)
  (h1 : Circles_disjoint_Interiors Γ1 Γ2 Γ3 Γ4)
  (h2 : Γ1.tangent Γ2 A)
  (h3 : Γ2.tangent Γ3 B)
  (h4 : Γ3.tangent Γ4 C)
  (h5 : Γ4.tangent Γ1 D) :
  Circle.concyclic A B C D :=
sorry

end concyclic_of_tangent_circles_l101_101164


namespace solve_for_base_b_l101_101382

-- Definitions that directly come from the conditions of the problem
def is_square (E F G H : ℝ × ℝ) : Prop :=
  let ⟨x1, y1⟩ := E
  let ⟨x2, y2⟩ := F
  let ⟨x3, y3⟩ := G
  let ⟨x4, y4⟩ := H
  (x1 - x2)^2 + (y1 - y2)^2 = 25 ∧ 
  (x2 - x3)^2 + (y2 - y3)^2 = 25 ∧
  (x3 - x4)^2 + (y3 - y4)^2 = 25 ∧
  (x4 - x1)^2 + (y4 - y1)^2 = 25

def parallel_to_y_axis (E F : ℝ × ℝ) : Prop :=
  let ⟨x1, y1⟩ := E
  let ⟨x2, y2⟩ := F
  x1 = x2

noncomputable def y_log (b x : ℝ) : ℝ := log b x
noncomputable def y_3log (b x : ℝ) : ℝ := 3 * log b x
noncomputable def x_b2y (b y : ℝ) : ℝ := b ^ (2 * y)

theorem solve_for_base_b (b : ℝ) (E F G H : ℝ × ℝ)
  (h1 : is_square E F G H)
  (h2 : parallel_to_y_axis E F)
  (he : E.2 = y_log b E.1)
  (hf : F.2 = y_3log b F.1)
  (hh : H.1 = x_b2y b H.2) :
  b = 1.5 := 
sorry

end solve_for_base_b_l101_101382


namespace train_crosses_platform_in_given_time_l101_101333

-- Define the given conditions
def length_of_train : ℕ := 140  -- in meters
def length_of_platform : ℕ := 520  -- in meters
def speed_of_train_km_per_hr : ℕ := 55  -- in km/hr

-- Convert speed from km/hr to m/s
def speed_of_train_m_per_s : ℝ := (speed_of_train_km_per_hr * 1000) / 3600

-- Calculate the total distance (length_of_train + length_of_platform)
def total_distance : ℕ := length_of_train + length_of_platform 

-- The calculation of expected time taken ignoring the slight approximation
def expected_time : ℝ := total_distance / speed_of_train_m_per_s

-- The statement to be proved
theorem train_crosses_platform_in_given_time :
  expected_time ≈ 43.2 := by
  sorry

end train_crosses_platform_in_given_time_l101_101333


namespace correct_statements_l101_101932

-- Definitions of planes and lines
variables {Point Line Plane : Type*}

-- Defining perpendicularity and parallelism relationships
variables (l m : Line) (α β : Plane)

-- Given conditions
axiom perpendicular_to_plane : ∀ {l : Line} {α : Plane}, l ⥮ α
axiom line_in_plane : ∀ {m : Line} {β : Plane}, m ⊆ β

-- Proven statements
theorem correct_statements (h1 : l ⥮ α) (h2 : m ⊆ β) :
  (α ‖ β → l ⥮ m) ∧
  (α ⥮ β → ¬ (l ‖ m)) ∧
  (l ‖ m → α ⥮ β) ∧
  (l ⥮ m → ¬ (α ‖ β)) :=
by
  sorry

end correct_statements_l101_101932


namespace chess_pieces_on_board_l101_101301

theorem chess_pieces_on_board :
  (∀ (board : ℕ → ℕ → ℕ),
    (∀ x y, 0 ≤ x ∧ x < 7 → 0 ≤ y ∧ y < 7 → 
      board x y + board (x+1) y + board x (y+1) + board (x+1) (y+1) = 4 * board 0 0)
    ∧ (∀ x y, (0 ≤ x ∧ x < 6 ∧ 0 ≤ y ∧ y < 8 →
      board x y + board (x+1) y + board (x+2) y = 3 * board 0 0) ∨
      (0 ≤ x ∧ x < 8 ∧ 0 ≤ y ∧ y < 6 →
      board x y + board x (y+1) + board x (y+2) = 3 * board 0 0)) →
  ∃ n : ℕ, n = 0 ∨ n = 64
:=
begin
  sorry
end

end chess_pieces_on_board_l101_101301


namespace trees_died_in_typhoon_l101_101081

theorem trees_died_in_typhoon :
  ∀ (original_trees left_trees died_trees : ℕ), 
  original_trees = 20 → 
  left_trees = 4 → 
  died_trees = original_trees - left_trees → 
  died_trees = 16 :=
by
  intros original_trees left_trees died_trees h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end trees_died_in_typhoon_l101_101081


namespace count_non_factorial_tails_lt_2500_l101_101450

def f (m : ℕ) : ℕ := 
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + ... -- Continue as needed

theorem count_non_factorial_tails_lt_2500 : 
  #{ n : ℕ | n < 2500 ∧ ¬ (∃ m : ℕ, f(m) = n) } = 4 :=
by sorry

end count_non_factorial_tails_lt_2500_l101_101450


namespace additional_charge_per_international_letter_l101_101862

-- Definitions based on conditions
def standard_postage_per_letter : ℕ := 108
def num_international_letters : ℕ := 2
def total_cost : ℕ := 460
def num_letters : ℕ := 4

-- Theorem stating the question
theorem additional_charge_per_international_letter :
  (total_cost - (num_letters * standard_postage_per_letter)) / num_international_letters = 14 :=
by
  sorry

end additional_charge_per_international_letter_l101_101862


namespace Q_evaluation_at_2_l101_101731

noncomputable def Q : Polynomial ℚ := 
  (Polynomial.X^2 + Polynomial.C 4)^2

theorem Q_evaluation_at_2 : 
  Q.eval 2 = 64 :=
by 
  -- We'll skip the proof as per the instructions.
  sorry

end Q_evaluation_at_2_l101_101731


namespace range_of_b_l101_101076

-- Define the functions
def f (x : ℝ) : ℝ := Real.exp x - 1
def g (x : ℝ) : ℝ := -x^2 + 4 * x - 3

-- Formalizing the main statement
theorem range_of_b (a b : ℝ) (h : f a = g b) : 1 < b ∧ b < 3 :=
by
  sorry

end range_of_b_l101_101076


namespace find_triples_of_positive_integers_l101_101018

theorem find_triples_of_positive_integers :
  ∀ (x y z : ℕ), 2 * (x + y + z + 2 * x * y * z)^2 = (2 * x * y + 2 * y * z + 2 * z * x + 1)^2 + 2023 ↔ 
  (x = 3 ∧ y = 3 ∧ z = 2) ∨
  (x = 3 ∧ y = 2 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 3) ∨
  (x = 3 ∧ y = 2 ∧ z = 3) ∨
  (x = 3 ∧ y = 3 ∧ z = 2) := 
by 
  sorry

end find_triples_of_positive_integers_l101_101018


namespace rounding_addition_to_100_correct_l101_101663

theorem rounding_addition_to_100_correct : 
  (125 + 96) % 100 = 21 ∧ (125 + 96) / 100 = 2 →
  round (125 + 96) / 100 * 100 = 200 := 
by 
  intros h 
  have h1 : 125 + 96 = 221 := by norm_num
  have h2 : 221 % 100 = 21 := by rw h1; norm_num
  have h3 : 221 / 100 = 2 := by rw h1; norm_num
  exact h

end rounding_addition_to_100_correct_l101_101663


namespace multiples_of_7_with_unit_digit_7_and_lt_150_l101_101099

/--
Given a positive integer n,
Let \(7n\) be a multiple of 7,
\(7n < 150\),
and \(7n \mod 10 = 7\),
prove there are 11 such multiples of 7.
-/
theorem multiples_of_7_with_unit_digit_7_and_lt_150 :
  ∃! (n : ℕ), n ≤ 11 ∧ ∀ (m : ℕ), (1 ≤ m ∧ m ≤ n) → (7 * (2 * m - 1) < 150 ∧ (7 * (2 * m - 1)) % 10 = 7) := by
sorry

end multiples_of_7_with_unit_digit_7_and_lt_150_l101_101099


namespace range_of_a_l101_101638

noncomputable def is_odd_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = -f(x)

theorem range_of_a :
  (∀ f : ℝ → ℝ, (is_odd_function f) ∧ (∀ x : ℝ, 0 ≤ x → f x = x^2) →
  (∀ x : ℝ, a ≤ x ∧ x ≤ a + 2 → f (x + a) ≥ 2 * f x) → a ∈ Set.Ici (Real.sqrt 2)) :=
begin
  sorry
end

end range_of_a_l101_101638


namespace jackson_final_grade_l101_101633

def jackson_hours_playing_video_games : ℕ := 9

def ratio_study_to_play : ℚ := 1 / 3

def time_spent_studying (hours_playing : ℕ) (ratio : ℚ) : ℚ := hours_playing * ratio

def points_per_hour_studying : ℕ := 15

def jackson_grade (time_studied : ℚ) (points_per_hour : ℕ) : ℚ := time_studied * points_per_hour

theorem jackson_final_grade :
  jackson_grade
    (time_spent_studying jackson_hours_playing_video_games ratio_study_to_play)
    points_per_hour_studying = 45 :=
by
  sorry

end jackson_final_grade_l101_101633


namespace travel_time_ratio_l101_101211

theorem travel_time_ratio 
  (T_NY_SF : ℕ)
  (layover : ℕ)
  (total_time : ℕ)
  (T_NY_SF_val : T_NY_SF = 24)
  (layover_val : layover = 16)
  (total_time_val : total_time = 58)
  (eqn : ∀ (T_NO_NY : ℕ), T_NO_NY + layover + T_NY_SF = total_time) : 
  (T_NO_NY / T_NY_SF = 3/4) :=
by
  have T_NY_SF := T_NY_SF_val
  have layover := layover_val
  have total_time := total_time_val
  specialize eqn (total_time - layover - T_NY_SF)
  calc 
    (total_time - layover - T_NY_SF) / T_NY_SF
        = 18 / 24 : by sorry
        = 3 / 4   : by sorry
  sorry -- Proof should be added here

end travel_time_ratio_l101_101211


namespace circumcircle_fixed_point_l101_101955

open EuclideanGeometry

/-- Given the parabola P: y^2 = x, the tangents at two moving points A and B on it intersect at point C. 
Let D be the circumcenter of triangle ΔABC. Prove that the circumcircle of triangle ΔABD passes through 
the fixed point (1/4, 0). -/
theorem circumcircle_fixed_point (A B C D : Point) (F : Point) :
  ∃ F : Point, (F.x = 1/4) ∧ (F.y = 0) →
  (on_parabola A) ∧ (on_parabola B) ∧
  (tangent_intersects_at A B C) ∧
  (circumcenter ABC D) →
  passes_through_circumcircle ABD F :=
begin
  sorry
end

def on_parabola (P : Point) : Prop := 
  P.y ^ 2 = P.x

def tangent_intersects_at (A B C : Point) : Prop := 
  ∃ k1 k2 : ℝ, 
  is_tangent A k1 ∧
  is_tangent B k2 ∧
  intersects A B C k1 k2

def circumcenter (A B C D : Point) : Prop := 
  is_circumcenter A B C D

def passes_through_circumcircle (A B D F : Point) : Prop := 
  is_on_circumcircle A B D F

attribute [instance] Mathlib_contact.transforms.circle

end circumcircle_fixed_point_l101_101955


namespace probability_at_least_one_trip_l101_101485

theorem probability_at_least_one_trip (p_A_trip : ℚ) (p_B_trip : ℚ)
  (h1 : p_A_trip = 1/4) (h2 : p_B_trip = 1/5) :
  (1 - ((1 - p_A_trip) * (1 - p_B_trip))) = 2/5 :=
by
  sorry

end probability_at_least_one_trip_l101_101485


namespace total_cost_of_deck_l101_101289

def num_rare_cards : ℕ := 19
def num_uncommon_cards : ℕ := 11
def num_common_cards : ℕ := 30

def cost_per_rare_card : ℕ := 1
def cost_per_uncommon_card : ℝ := 0.50
def cost_per_common_card : ℝ := 0.25

def cost_of_rare_cards : ℕ := num_rare_cards * cost_per_rare_card
def cost_of_uncommon_cards : ℝ := num_uncommon_cards * cost_per_uncommon_card
def cost_of_common_cards : ℝ := num_common_cards * cost_per_common_card

def total_cost : ℝ := cost_of_rare_cards + cost_of_uncommon_cards + cost_of_common_cards

theorem total_cost_of_deck : total_cost = 32 := by
  -- We will need to convert integers to real numbers for this addition
  have h1 : (cost_of_rare_cards : ℝ) = 19 := by norm_cast
  rw [h1]
  have h2 : (num_uncommon_cards: ℝ) * cost_per_uncommon_card = 5.5 := by norm_num
  have h3 : (num_common_cards: ℝ) * cost_per_common_card = 7.5 := by norm_num
  calc
    (19 : ℝ) + 5.5 + 7.5 = 32 := by norm_num

end total_cost_of_deck_l101_101289


namespace ellipse_foci_sum_l101_101639

-- Definitions for the problem's conditions:
def P (x y : ℝ) := (x^2 / 25 + y^2 / 16 = 1)

def a : ℝ := 5
def foci_dist := 10  -- because 2a = 10

-- Statement of the problem in Lean
theorem ellipse_foci_sum (x y : ℝ) (F1 F2 : ℝ) (hP : P x y) :
  |PF_1 + PF_2| = foci_dist :=
by sorry

end ellipse_foci_sum_l101_101639


namespace multiples_of_7_units_digit_7_l101_101091

theorem multiples_of_7_units_digit_7 (n : ℕ) (h1 : n < 150) (h2 : ∃ (k : ℕ), n = 7 * k) (h3 : n % 10 = 7) : 
    ∃ (m : ℕ), m = 2 := 
by
  sorry

end multiples_of_7_units_digit_7_l101_101091


namespace seahorse_penguin_ratio_l101_101843

theorem seahorse_penguin_ratio :
  ∃ S P : ℕ, S = 70 ∧ P = S + 85 ∧ Nat.gcd 70 (S + 85) = 5 ∧ 70 / Nat.gcd 70 (S + 85) = 14 ∧ (S + 85) / Nat.gcd 70 (S + 85) = 31 :=
by
  sorry

end seahorse_penguin_ratio_l101_101843


namespace equal_BE_CF_l101_101902

open_locale classical
noncomputable theory

variables {A B C D E F M : Point}
variables (triangle_ABC : Triangle A B C)
variables (angle_bisector_AD : AngleBisector A D (LineThrough B C))
variables (CE_parallel_MB : Parallel (LineThrough C E) (LineThrough M B))
variables (BF_parallel_MC : Parallel (LineThrough B F) (LineThrough M C))

theorem equal_BE_CF
  (H1 : M ∈ angle_bisector_AD)
  (H2 : CE_parallel_MB)
  (H3 : BF_parallel_MC) :
  distance B E = distance C F :=
by
  sorry

end equal_BE_CF_l101_101902


namespace polynomial_remainder_l101_101640

def Q (x : ℝ) : ℝ := sorry -- Define Q(x) as some polynomial (unknown)

-- Conditions given in the problem
def cond1 : Prop := ∃ R : ℝ → ℝ, Q(x) = (x - 15) * R(x) + 8
def cond2 : Prop := ∃ S : ℝ → ℝ, Q(x) = (x - 10) * S(x) + 3

-- Question: the remainder when Q(x) is divided by (x-10)(x-15)
theorem polynomial_remainder : cond1 ∧ cond2 → ∃ R : ℝ → ℝ, Q(x) = (x - 10) * (x - 15) * R(x) + (x - 7) := 
by
  sorry

end polynomial_remainder_l101_101640


namespace find_alpha_l101_101964

-- Given vectors
def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 2 * Real.sin x)
def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)

-- Define the function f
def f (x : ℝ) : ℝ :=
  let dot_product := (a x).1 * (b x).1 + (a x).2 * (b x).2
  dot_product - Real.sqrt 3

-- Given conditions
def cond1 : Prop :=
  f (α / 2 - π / 6) - f (α / 2 + π / 12) = Real.sqrt 6

def cond2 (α : ℝ) : Prop :=
  α ∈ Set.Ioo (π / 2) π

-- Prove α = 7π/12 or α = 11π/12 given the conditions
theorem find_alpha (α : ℝ) (h1 : cond1 α) (h2 : cond2 α) : α = 7 * π / 12 ∨ α = 11 * π / 12 :=
  sorry

end find_alpha_l101_101964


namespace bus_speed_calculation_l101_101883

noncomputable def bus_speed_excluding_stoppages : ℝ :=
  let effective_speed_with_stoppages := 50 -- kmph
  let stoppage_time_in_minutes := 13.125 -- minutes per hour
  let stoppage_time_in_hours := stoppage_time_in_minutes / 60 -- convert to hours
  let effective_moving_time := 1 - stoppage_time_in_hours -- effective moving time in one hour
  let bus_speed := (effective_speed_with_stoppages * 60) / (60 - stoppage_time_in_minutes) -- calculate bus speed
  bus_speed

theorem bus_speed_calculation : bus_speed_excluding_stoppages = 64 := by
  sorry

end bus_speed_calculation_l101_101883


namespace angle_q_of_extended_sides_of_regular_decagon_l101_101680

theorem angle_q_of_extended_sides_of_regular_decagon :
  ∀ (A B C D E F G H I J Q : Type) [regular_decagon A B C D E F G H I J]
    (extended_sides_meet : extended_sides_meet Q A J D E),
  measure_angle Q = 144 :=
by sorry

end angle_q_of_extended_sides_of_regular_decagon_l101_101680


namespace triangle_area_afb_l101_101626

noncomputable def length_of_median (a b c : ℝ) : ℝ :=
  real.sqrt ((2*b^2 + 2*c^2 - a^2) / 4)

theorem triangle_area_afb:
  ∃ (m n : ℕ),
  (∀ (A B C D E F : Type)
    (AD CE AB : ℝ)
    (hA : A ∈ ℝ)
    (hB : B ∈ ℝ) 
    (hC : C ∈ ℝ) 
    (hAD : length_of_median AB BC AC = 18)
    (hCE : length_of_median CA AB BC = 27)
    (hAB : AB = 24)
    (hEF : CE > 0)
    (hF : intersects_circumcircle CE F),
  area (△ A F B) = m * real.sqrt n ∧
  ∀ (p : ℕ) (hp : p.prime), ¬(p^2 ∣ n)) :=
begin
  sorry
end

end triangle_area_afb_l101_101626


namespace amc_proposed_by_Dorlir_Ahmeti_Albania_l101_101027

-- Define the problem statement, encapsulating the conditions and the final inequality.
theorem amc_proposed_by_Dorlir_Ahmeti_Albania
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_cond : a * b + b * c + c * a = 3) :
  (a / Real.sqrt (a^3 + 5) + b / Real.sqrt (b^3 + 5) + c / Real.sqrt (c^3 + 5) ≤ Real.sqrt 6 / 2) := 
by 
  sorry -- Proof steps go here, which are omitted as per the requirement.

end amc_proposed_by_Dorlir_Ahmeti_Albania_l101_101027


namespace add_fractions_add_fractions_as_mixed_l101_101833

theorem add_fractions : (3 / 4) + (5 / 6) + (4 / 3) = (35 / 12) := sorry

theorem add_fractions_as_mixed : (3 / 4) + (5 / 6) + (4 / 3) = 2 + 11 / 12 := sorry

end add_fractions_add_fractions_as_mixed_l101_101833


namespace max_area_triangle_obc_l101_101068

theorem max_area_triangle_obc :
  let ellipse := λ x y, (x^2) / 4 + (y^2) / 3 = 1
  let B := (1, 3 / 2)
  ∃ C : ℝ × ℝ, 
    ellipse C.1 C.2 ∧ 
    (∃ l : ℝ × ℝ → Prop, l B ∧ l C) →
    (∃ t : ℝ, l (0, 0) ∧ l (1, 3 / 2) ∧ l C) →
      let area := λ O B C, 1/2 * abs (O.1 * (B.2 - C.2) + B.1 * (C.2 - O.2) + C.1 * (O.2 - B.2))
      area (0, 0) B C = sqrt 3 :=
sorry

end max_area_triangle_obc_l101_101068


namespace count_non_factorial_tails_lt_2500_l101_101446

def f (m : ℕ) : ℕ := 
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + ... -- Continue as needed

theorem count_non_factorial_tails_lt_2500 : 
  #{ n : ℕ | n < 2500 ∧ ¬ (∃ m : ℕ, f(m) = n) } = 4 :=
by sorry

end count_non_factorial_tails_lt_2500_l101_101446


namespace room_width_proof_l101_101228

theorem room_width_proof
  (length_room : ℝ) (veranda_width : ℝ) (area_veranda : ℝ) (w : ℝ) 
  (h1 : length_room = 18)
  (h2 : veranda_width = 2)
  (h3 : area_veranda = 136)
  (h4 : 22 * (w + 4) - 18 * w = 136) :
  w = 12 :=
by
  rw [←h1, ←h2, ←h3] at h4
  sorry

end room_width_proof_l101_101228


namespace max_trees_cut_l101_101598

theorem max_trees_cut (rows cols : ℕ) (total_trees : ℕ) (grid : set (ℕ × ℕ))
  (h1 : rows = 100)
  (h2 : cols = 100)
  (h3 : total_trees = rows * cols)
  (h4 : grid = set.univ.filter (λ p, p.1 < rows ∧ p.2 < cols)) :
  ∃ max_cut_trees : ℕ, max_cut_trees = 2500 ∧ 
  (∀ stumps : set (ℕ × ℕ), stumps ⊆ grid ∧ stumps.size = max_cut_trees → 
    ∀ s1 s2 ∈ stumps, s1 ≠ s2 → s1.1 ≠ s2.1 ∧ s1.2 ≠ s2.2) :=
begin
  sorry
end

end max_trees_cut_l101_101598


namespace find_n_solution_l101_101524

open Real

noncomputable def find_n (x : ℝ) (n : ℝ) : Prop :=
  ln (sin x) + ln (cos x) = -1 ∧
  tan x = sqrt 3 ∧
  ln (sin x + cos x) = (1 / 3) * (ln n - 1)

theorem find_n_solution : ∃ n, ∃ x, find_n x (exp (3 * (-1/2 + ln (1 + 1 / sqrt (sqrt 3))) + 1)) :=
by
  sorry

end find_n_solution_l101_101524


namespace triangle_inequality_l101_101051

theorem triangle_inequality
  (A B C P : Point)
  (S S_a S_b S_c : ℝ)
  (PA PB PC : ℝ)
  (R : ℝ)
  (triangle_ABC : Triangle A B C)
  (P_inside_ABC : PointInsideTriangle P triangle_ABC)
  (area_ABC : Area triangle_ABC = S)
  (area_BPC : Area (Triangle B P C) = S_a)
  (area_CPA : Area (Triangle C P A) = S_b)
  (area_APB : Area (Triangle A P B) = S_c)
  (circumradius : Circumradius triangle_ABC = R) :
  (S_a / PA^2) + (S_b / PB^2) + (S_c / PC^2) ≥ (S / R^2) := 
sorry

end triangle_inequality_l101_101051


namespace ratio_of_lions_to_penguins_is_simplified_correctly_l101_101844

variable (L P : ℕ)
variable (h1 : L = 30)
variable (h2 : P = 112)

theorem ratio_of_lions_to_penguins_is_simplified_correctly : L = 30 → P = 112 → Nat.gcd L P = 2 → (L / 2) = 15 ∧ (P / 2) = 56 :=
by
  intros
  have hL : L = 30 := by assumption
  have hP : P = 112 := by assumption
  have gcd_2 : Nat.gcd L P = 2 := by assumption
  split
  { rw [hL]
    norm_num }
  { rw [hP]
    norm_num }


end ratio_of_lions_to_penguins_is_simplified_correctly_l101_101844


namespace min_expression_value_l101_101026

theorem min_expression_value (a b c : ℝ) (h₀ : c > b) (h₁ : b > a) (h₂ : c ≠ 0) :
  ∃ k : ℝ, k = 2 ∧ (∀ a b c, c > b → b > a → c ≠ 0 → k ≤ (a+b)^2+(b+c)^2+(c-a)^2 / c^2) :=
begin
  use 2,
  split,
  { refl, },
  { intros a b c hc1 hb1 hc2,
    let expr := (a + b)^2 + (b + c)^2 + (c - a)^2,
    let denom := c^2,
    have expr_nonneg : expr ≥ 0, from
      add_nonneg (add_nonneg (sq_nonneg _) (sq_nonneg _)) (sq_nonneg _),
    have ratio_nonneg : expr / denom ≥ 0, from div_nonneg expr_nonneg (sq_nonneg _),
    have : ∀ (a b c : ℝ), c > b → b > a → c ≠ 0 → 2 ≤ (a + b)^2 + (b + c)^2 + (c - a)^2 / c^2, 
      sorry,
    from this _ _ _ hc1 hb1 hc2, }
end

end min_expression_value_l101_101026


namespace num_white_black_balls_prob_2_black_balls_dist_exp_black_balls_l101_101357

-- Problem 1: Number of white and black balls
theorem num_white_black_balls (n m : ℕ) (h1 : n + m = 10)
  (h2 : (10 - m) = 4) : n = 4 ∧ m = 6 :=
by sorry

-- Problem 2: Probability of drawing exactly 2 black balls with replacement
theorem prob_2_black_balls (p_black_draw : ℕ → ℕ → ℚ)
  (h1 : ∀ n m, p_black_draw n m = (6/10)^(n-m) * (4/10)^m)
  (h2 : p_black_draw 2 3 = 54/125) : p_black_draw 2 3 = 54 / 125 :=
by sorry

-- Problem 3: Distribution and Expectation of number of black balls drawn without replacement
theorem dist_exp_black_balls (prob_X : ℕ → ℚ) (expect_X : ℚ)
  (h1 : prob_X 0 = 2/15) (h2 : prob_X 1 = 8/15) (h3 : prob_X 2 = 1/3)
  (h4 : expect_X = 6 / 5) : ∀ k, prob_X k = match k with
    | 0 => 2/15
    | 1 => 8/15
    | 2 => 1/3
    | _ => 0 :=
by sorry

end num_white_black_balls_prob_2_black_balls_dist_exp_black_balls_l101_101357


namespace G_F_E_collinear_l101_101139

noncomputable def cyclic_quad (A B C D : Type) := sorry

theorem G_F_E_collinear (A B C D E F G H I J : Type) (circumcircle : ∀ P Q R : Type, Prop) (inscribed : circumcircle A B C)
  (intersect_AC_BD_F : the_diagonals A C (D) intersect_at F)
  (angle_bisectors_intersect_BAC_CDC_G : the_angle_bisectors BAC CDB intersect G)
  (AG_intersects_BD_H : AG intersects BD at H)
  (DG_intersects_AC_I : DG intersects AC at I)
  (EG_intersects_AD_J : EG intersects AD at J)
  (cyclic_quad_FHGI : cyclic_quad F H G I)
  (JA_FC_GH_JD_FB_GI : JA * FC * GH = JD * FB * GI) :
  collinear G F E := sorry

end G_F_E_collinear_l101_101139


namespace carA_total_distance_2016_meetings_l101_101416

noncomputable def distanceTravelledByCarA : ℕ → ℝ 
| 2016 := 1813900

theorem carA_total_distance_2016_meetings
  (a_to_b_speed : ℝ) (b_to_a_speed : ℝ)
  (b_to_a_speed_carB : ℝ) (a_to_b_speed_carB : ℝ)
  (distance_a_b : ℝ) :
  a_to_b_speed = 40 ∧ b_to_a_speed = 50 ∧
  b_to_a_speed_carB = 50 ∧ a_to_b_speed_carB = 40 ∧
  distance_a_b = 900 → 
  distanceTravelledByCarA 2016 = 1813900 :=
by
  sorry

end carA_total_distance_2016_meetings_l101_101416


namespace ratio_volumes_of_spheres_l101_101583

theorem ratio_volumes_of_spheres (r R : ℝ) (hratio : (4 * π * r^2) / (4 * π * R^2) = 4 / 9) :
    (4 / 3 * π * r^3) / (4 / 3 * π * R^3) = 8 / 27 := 
by {
  sorry
}

end ratio_volumes_of_spheres_l101_101583


namespace find_angle_C_find_sin_A_l101_101152

noncomputable def triangleABC {A B C : ℝ} (a b c : ℝ) : Prop :=
  2 * c * cos A = 2 * b - sqrt 3 * a

theorem find_angle_C (a b c : ℝ) (A : ℝ) (h : triangleABC a b c) : C = π / 6 :=
  sorry

theorem find_sin_A (a b c : ℝ) (A C : ℝ) (area : ℝ) (h_area : area = 2 * sqrt 3) 
  (h_b : b = 2) (h_C : C = π / 6) (h : triangleABC a b c) : sin A = sqrt 7 / 7 :=
  sorry

end find_angle_C_find_sin_A_l101_101152


namespace rational_linear_independent_sqrt_prime_l101_101336

theorem rational_linear_independent_sqrt_prime (p : ℕ) (hp : Nat.Prime p) (m n m1 n1 : ℚ) :
  m + n * Real.sqrt p = m1 + n1 * Real.sqrt p → m = m1 ∧ n = n1 :=
sorry

end rational_linear_independent_sqrt_prime_l101_101336


namespace sum_of_divisors_37_l101_101310

theorem sum_of_divisors_37 : ∑ d in (Finset.filter (fun d => d > 0 ∧ 37 % d = 0) (Finset.range (37 + 1))), d = 38 :=
by
  sorry

end sum_of_divisors_37_l101_101310


namespace greatest_integer_sum_divisors_2013_l101_101033

-- Let s(n) be a function that returns the sum of squares of positive integers less or equal to n that are relatively prime to n
def s (n : ℕ) : ℕ := 
  ∑ k in finset.filter (λ k, Nat.gcd k n = 1) (finset.range (n + 1)), k^2

-- Define the proposition that given the prime factorization and divisors of 2013, 
-- the greatest integer less than or equal to the sum of s(n)/n^2 over all divisors equals 345
theorem greatest_integer_sum_divisors_2013 :
  ∑ n in finset.filter (λ d, 2013 % d = 0) (finset.range (2013 + 1)), (s n : ℚ) / (n ^ 2) ≤ 345 :=
  sorry

end greatest_integer_sum_divisors_2013_l101_101033


namespace prime_power_of_n_l101_101683

theorem prime_power_of_n (n : ℕ) (h : Nat.Prime (4^n + 2^n + 1)) : ∃ k : ℕ, n = 3^k := 
sorry

end prime_power_of_n_l101_101683


namespace pyramid_volume_transformation_l101_101823

theorem pyramid_volume_transformation (V : ℝ) (l w h : ℝ) 
  (hV : V = (1 / 3) * l * w * h) 
  (new_l : ℝ := 3 * l)
  (new_w : ℝ := (1 / 2) * w)
  (new_h : ℝ := 1.25 * h) :
  (1 / 3) * new_l * new_w * new_h = 225 := 
by
  have hV' := V -- original volume, given V = 60
  have l_extended := 3
  have w_extended := 0.5
  have h_extended := 1.25
  calc
    (1 / 3) * (3 * l) * ((1/2) * w) * (1.25 * h)
        = V * l_extended * w_extended * h_extended : by simp [hV, mul_assoc, mul_comm, mul_left_comm]
    ... = 225 : by { rw hV, norm_num }

end pyramid_volume_transformation_l101_101823


namespace dot_product_of_CA_CB_l101_101124

variable {α : Type*} [Field α] [LinearOrderedField α]

noncomputable def triangle_dot_product (a b c : α) (sin B sin C : α) 
  (h1 : a^2 + b^2 - c^2 = sqrt 3 * a * b)
  (h2 : a * c * sin B = 2 * sqrt 3 * sin C) : α :=
  let ab := 2 * sqrt 3 in
  ab * (sqrt 3 / 2)

theorem dot_product_of_CA_CB 
  {a b c sin_B sin_C : α}
  (h1 : a^2 + b^2 - c^2 = sqrt 3 * a * b)
  (h2 : a * c * sin_B = 2 * sqrt 3 * sin_C) : 
  triangle_dot_product a b c sin_B sin_C h1 h2 = 3 := by
  sorry

end dot_product_of_CA_CB_l101_101124


namespace distance_between_parallel_lines_l101_101741

theorem distance_between_parallel_lines (r d : ℝ) 
  (h1 : ∃ p1 p2 p3 : ℝ, p1 = 40 ∧ p2 = 40 ∧ p3 = 36) 
  (h2 : ∀ θ : ℝ, ∃ A B C D : ℝ → ℝ, 
    (A θ - B θ) = 40 ∧ (C θ - D θ) = 36) : d = 6 :=
sorry

end distance_between_parallel_lines_l101_101741


namespace calculate_m_plus_n_l101_101855

theorem calculate_m_plus_n :
  let product := ∏ k in finset.range (22), (csc ((4 * k + 1 - 3 : ℚ) * π / 180)) ^ 2
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ product = m^n ∧ (m + n = 23) :=
by
  let product := ∏ k in finset.range (22), (csc ((4 * k + 1 - 3 : ℚ) * π / 180)) ^ 2
  use 2
  use 21
  split
  { exact nat.succ_pos' 1 }
  split
  { exact nat.succ_pos' 20 }
  split
  { sorry }
  { refl }

end calculate_m_plus_n_l101_101855


namespace socks_pair_count_l101_101988

theorem socks_pair_count :
  let white := 5
  let brown := 5
  let blue := 3
  let green := 2
  (white * brown) + (white * blue) + (white * green) + (brown * blue) + (brown * green) + (blue * green) = 81 :=
by
  intros
  sorry

end socks_pair_count_l101_101988


namespace solve_sqrt_equation_l101_101209

theorem solve_sqrt_equation (x : ℝ) :
  (sqrt (9 + sqrt (27 + 3 * x)) + sqrt (3 + sqrt (9 + x)) = 3 + 3 * sqrt 3) → x = 1 :=
by
  sorry

end solve_sqrt_equation_l101_101209


namespace zoo_total_animals_l101_101389

theorem zoo_total_animals :
  let tiger_enclosures := 4
  let zebra_enclosures := tiger_enclosures * 2
  let giraffe_enclosures := zebra_enclosures * 3
  let tigers := tiger_enclosures * 4
  let zebras := zebra_enclosures * 10
  let giraffes := giraffe_enclosures * 2
  let total_animals := tigers + zebras + giraffes
  in total_animals = 144 := by
  sorry

end zoo_total_animals_l101_101389


namespace find_XY_squared_l101_101647

axiom acute_scalene_triangle (A B C : Point) : Scalene A B C ∧ Acute A B C

noncomputable def triangle_conditions
  (T X Y : Point) (A B C : Point)
  (BT CT BC TX TY XY : ℝ) : Prop :=
  acute_scalene_triangle A B C ∧
  BT = 18 ∧ CT = 18 ∧ BC = 26 ∧
  (TX^2 + TY^2 + XY^2 = 1420)

theorem find_XY_squared 
  (T X Y A B C : Point) 
  (BT CT BC TX TY XY : ℝ)
  (h : triangle_conditions T X Y A B C BT CT BC TX TY XY) :
  XY^2 = 473 :=
sorry

end find_XY_squared_l101_101647


namespace sum_of_divisors_37_l101_101315

theorem sum_of_divisors_37 : ∑ d in (finset.filter (λ d, 37 % d = 0) (finset.range 38)), d = 38 :=
by {
  sorry
}

end sum_of_divisors_37_l101_101315


namespace sum_condition_a_n_correct_T_n_correct_l101_101514

noncomputable def S_n (n : ℕ) : ℝ := sorry

def a_n (n : ℕ) : ℕ := 2 * n - 1

def b_n (q : ℝ) (n : ℕ) : ℝ := q ^ (a_n n)

def T_n (q : ℝ) (n : ℕ) : ℝ :=
  if q ≠ 1 then 
    (1 / (q^4 - 1)) * (1 - 1 / q^(4 * n))
  else 
    n

theorem sum_condition : S_4 = 16 ∧ S_6 = 36 := sorry

theorem a_n_correct (n : ℕ) : a_n n = 2 * n - 1 :=
  sorry

theorem T_n_correct (q : ℝ) (n : ℕ) (hq : q > 0) : 
  T_n q n = 
    if q ≠ 1 then 
      (1 / (q^4 - 1)) * (1 - 1 / q^(4 * n))
    else 
      n :=
  sorry

end sum_condition_a_n_correct_T_n_correct_l101_101514


namespace non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101440

def g (m : Nat) : Nat :=
  let f k := m / k
  f 5 + f 25 + f 125 + f 625 + f 3125 + f 15625 + f 78125 + f 390625 + f 1953125 + f 9765625 + f 48828125 + f 244140625 + f 1220703125 + f 6103515625 + f 30517578125 -- continue as needed

def is_factorial_tail (n : Nat) : Prop :=
  ∃ m : Nat, g m = n

def count_factorial_tails (N : Nat) : Nat :=
  (List.range N).filter (λ n, is_factorial_tail n).length

theorem non_factorial_tails_lemma : count_factorial_tails 2500 = 2000 := sorry

theorem count_non_factorial_tails (N : Nat) : Prop :=
  let total := N - 1
  total - count_factorial_tails N = 499

theorem solution : count_non_factorial_tails 2500 := sorry

end non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101440


namespace other_number_is_7_l101_101880

-- Given conditions
variable (a b : ℤ)
variable (h1 : 2 * a + 3 * b = 110)
variable (h2 : a = 32 ∨ b = 32)

-- The proof goal
theorem other_number_is_7 : (a = 7 ∧ b = 32) ∨ (a = 32 ∧ b = 7) :=
by
  sorry

end other_number_is_7_l101_101880


namespace average_price_per_racket_l101_101381

theorem average_price_per_racket (total_amount : ℕ) (pairs_sold : ℕ) (expected_average : ℚ) 
  (h1 : total_amount = 637) (h2 : pairs_sold = 65) : 
  expected_average = total_amount / pairs_sold := 
by
  sorry

end average_price_per_racket_l101_101381


namespace total_surface_area_of_resulting_solid_is_12_square_feet_l101_101386

noncomputable def height_of_D :=
  let h_A := 1 / 4
  let h_B := 1 / 5
  let h_C := 1 / 8
  2 - (h_A + h_B + h_C)

theorem total_surface_area_of_resulting_solid_is_12_square_feet :
  let h_A := 1 / 4
  let h_B := 1 / 5
  let h_C := 1 / 8
  let h_D := 2 - (h_A + h_B + h_C)
  let top_and_bottom_area := 4 * 2
  let side_area := 2 * (h_A + h_B + h_C + h_D)
  top_and_bottom_area + side_area = 12 := by
  sorry

end total_surface_area_of_resulting_solid_is_12_square_feet_l101_101386


namespace non_factorial_tails_lt_2500_l101_101438

-- Define the function f(m)
def f (m: ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625

-- Define what it means to be a factorial tail
def is_factorial_tail (n: ℕ) : Prop :=
  ∃ m : ℕ, f(m) = n

-- Define the statement to be proved
theorem non_factorial_tails_lt_2500 : 
  (finset.range 2500).filter (λ n, ¬ is_factorial_tail n).card = 499 :=
by
  sorry

end non_factorial_tails_lt_2500_l101_101438


namespace inverse_variation_l101_101342

theorem inverse_variation :
  (∃ k : ℝ, ∀ y : ℝ, y ≠ 0 → (x = k / y^2)) →
  (x = 1 → (∃ y : ℝ, x = 1 ∧ y = 9 ∧ x = 0.1111111111111111 → y = 3)) :=
begin
  sorry
end

end inverse_variation_l101_101342


namespace multiples_of_7_units_digit_7_l101_101093

theorem multiples_of_7_units_digit_7 (n : ℕ) (h1 : n < 150) (h2 : ∃ (k : ℕ), n = 7 * k) (h3 : n % 10 = 7) : 
    ∃ (m : ℕ), m = 2 := 
by
  sorry

end multiples_of_7_units_digit_7_l101_101093


namespace scientific_notation_of_1040000000_l101_101421

theorem scientific_notation_of_1040000000 : (1.04 * 10^9 = 1040000000) :=
by
  -- Math proof steps can be added here
  sorry

end scientific_notation_of_1040000000_l101_101421


namespace tangent_line_at_1_l101_101948

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x^2 - 4 * x

theorem tangent_line_at_1 :
  let slope := (f' 1)
  let point := (1, f 1)
  let tangent_line_eq (x y : ℝ) := y = slope * (x - point.1) + point.2
  ∃ y, tangent_line_eq (x - 1) = x - y - 3 := sorry

end tangent_line_at_1_l101_101948


namespace students_weight_decrease_l101_101221

theorem students_weight_decrease (N : ℕ) 
  (avg_decrease : 3) 
  (weight_old : 80) 
  (weight_new : 62) 
  (weight_difference : 18) 
  (avg_weight_decrease: 3 * N = 18) : 
  N = 6 :=
by
  sorry

end students_weight_decrease_l101_101221


namespace Q_evaluation_at_2_l101_101734

noncomputable def Q : Polynomial ℚ := 
  (Polynomial.X^2 + Polynomial.C 4)^2

theorem Q_evaluation_at_2 : 
  Q.eval 2 = 64 :=
by 
  -- We'll skip the proof as per the instructions.
  sorry

end Q_evaluation_at_2_l101_101734


namespace ads_not_blocked_not_interesting_l101_101569

theorem ads_not_blocked_not_interesting:
  (let A_blocks := 0.75
   let B_blocks := 0.85
   let C_blocks := 0.95
   let A_let_through := 1 - A_blocks
   let B_let_through := 1 - B_blocks
   let C_let_through := 1 - C_blocks
   let all_let_through := A_let_through * B_let_through * C_let_through
   let interesting := 0.15
   let not_interesting := 1 - interesting
   (all_let_through * not_interesting) = 0.00159375) :=
  sorry

end ads_not_blocked_not_interesting_l101_101569


namespace volume_surface_area_ratio_eq_l101_101199

noncomputable
def sphere_radius (R : ℝ) : ℝ := R

noncomputable
def sphere_volume (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

noncomputable
def sphere_surface_area (R : ℝ) : ℝ := 4 * Real.pi * R^2

noncomputable
def truncated_cone_volume (h r1 r2 : ℝ) : ℝ := (1 / 3) * Real.pi * h * (r1^2 + r1 * r2 + r2^2)

noncomputable
def truncated_cone_surface_area (h r1 r2 : ℝ) : ℝ :=
  let l := Real.sqrt (h^2 + (r2 - r1)^2)
  π * r1^2 + π * r2^2 + π * (r1 + r2) * l

theorem volume_surface_area_ratio_eq (R h r1 r2 : ℝ) 
  (Vs := sphere_volume R)
  (As := sphere_surface_area R)
  (Vc := truncated_cone_volume h r1 r2)
  (Ac := truncated_cone_surface_area h r1 r2) :
  Vs / Vc = As / Ac :=
by sorry

end volume_surface_area_ratio_eq_l101_101199


namespace evaluate_expression_l101_101882

theorem evaluate_expression : 
  ( (5 ^ 2014) ^ 2 - (5 ^ 2012) ^ 2 ) / ( (5 ^ 2013) ^ 2 - (5 ^ 2011) ^ 2 ) = 25 := 
by sorry

end evaluate_expression_l101_101882


namespace inequality_proof_l101_101990

variable {A B C A' B' C' G A₁ B₁ C₁ : Point}
variable {AB BC CA A'B' B'C' C'A' AG GA' GA₁ GB GB' GC GC' CG' : ℝ}
variable {ABC A'B'C' : Triangle}

-- Given conditions
def areCongruent (ABC A'B'C' : Triangle) : Prop :=
  (AB = A'B') ∧ (BC = B'C') ∧ (CA = C'A')

def isCentroid (G : Point) (ABC : Triangle) : Prop :=
  G = (A + B + C) / 3

def intersectionPoint (P : Point) (circle1 circle2 : Circle) : Point :=
  -- Definition of intersection point (contextual definition)
  sorry

-- Define points and circles
def A1 : Point := intersectionPoint A' (circle_with_diameter AA') (circle_with_center_radius A' (distance A' G))
def B1 : Point := intersectionPoint B' (circle_with_diameter BB') (circle_with_center_radius B' (distance B' G))
def C1 : Point := intersectionPoint C' (circle_with_diameter CC') (circle_with_center_radius C' (distance C' G))

-- Final proof problem statement
theorem inequality_proof 
  (h1 : areCongruent ABC A'B'C')
  (h2 : isCentroid G ABC)
  (h3 : AA₁ = distance A A1)
  (h4 : BB₁ = distance B B1)
  (h5 : CC₁ = distance C C1) : 
  AA₁^2 + BB₁^2 + CC₁^2 ≤ AB^2 + BC^2 + CA^2 :=
sorry

end inequality_proof_l101_101990


namespace angle_relationship_l101_101623

open Real

variables (A B C D : Point)
variables (AB AC AD : ℝ)
variables (CAB DAC BDC DBC : ℝ)
variables (k : ℝ)

-- Given conditions
axiom h1 : AB = AC
axiom h2 : AC = AD
axiom h3 : DAC = k * CAB

-- Proof to be shown
theorem angle_relationship : DBC = k * BDC :=
  sorry

end angle_relationship_l101_101623


namespace vince_savings_l101_101299

-- Define constants and conditions
def earnings_per_customer : ℕ := 18
def monthly_expenses : ℕ := 280
def percentage_for_recreation : ℕ := 20
def number_of_customers : ℕ := 80

-- Function to calculate total earnings
def total_earnings (customers : ℕ) (earnings_per_customer : ℕ) : ℕ :=
  customers * earnings_per_customer

-- Function to calculate amount allocated for recreation and relaxation
def recreation_amount (total_earnings : ℕ) (percentage : ℕ) : ℕ :=
  total_earnings * percentage / 100

-- Function to calculate total expenses
def total_monthly_expenses (base_expenses : ℕ) (recreation : ℕ) : ℕ :=
  base_expenses + recreation

-- Function to calculate savings
def savings (total_earnings : ℕ) (total_expenses : ℕ) : ℕ :=
  total_earnings - total_expenses

theorem vince_savings :
  let earnings := total_earnings number_of_customers earnings_per_customer,
      recreation := recreation_amount earnings percentage_for_recreation,
      expenses := total_monthly_expenses monthly_expenses recreation
  in savings earnings expenses = 872 :=
by
  sorry

end vince_savings_l101_101299


namespace multiples_of_7_units_digit_7_l101_101092

theorem multiples_of_7_units_digit_7 (n : ℕ) (h1 : n < 150) (h2 : ∃ (k : ℕ), n = 7 * k) (h3 : n % 10 = 7) : 
    ∃ (m : ℕ), m = 2 := 
by
  sorry

end multiples_of_7_units_digit_7_l101_101092


namespace kitten_tail_percentage_increase_l101_101153

theorem kitten_tail_percentage_increase 
    (L1 : ℝ) (L3 : ℝ) (h1 : L1 = 16) (h3 : L3 = 25) : 
    ∃ P : ℝ, L3 = L1 * (1 + P / 100) ^ 2 ∧ P = 25 :=
by {
    use 25,
    rw [h1, h3],
    norm_num,
    sorry
}

end kitten_tail_percentage_increase_l101_101153


namespace original_number_l101_101813

theorem original_number (N y x : ℕ) 
  (h1: N + y = 54321)
  (h2: N = 10 * y + x)
  (h3: 11 * y + x = 54321)
  (h4: x = 54321 % 11)
  (hy: y = 4938) : 
  N = 49383 := 
  by 
  sorry

end original_number_l101_101813


namespace unique_poly_degree_4_l101_101737

theorem unique_poly_degree_4 
  (Q : ℚ[X]) 
  (h_deg : Q.degree = 4) 
  (h_lead_coeff : Q.leading_coeff = 1)
  (h_root : Q.eval (sqrt 3 + sqrt 7) = 0) : 
  Q = X^4 - 12 * X^2 + 16 ∧ Q.eval 2 = 0 :=
by sorry

end unique_poly_degree_4_l101_101737


namespace multiples_of_7_with_unit_digit_7_and_lt_150_l101_101102

/--
Given a positive integer n,
Let \(7n\) be a multiple of 7,
\(7n < 150\),
and \(7n \mod 10 = 7\),
prove there are 11 such multiples of 7.
-/
theorem multiples_of_7_with_unit_digit_7_and_lt_150 :
  ∃! (n : ℕ), n ≤ 11 ∧ ∀ (m : ℕ), (1 ≤ m ∧ m ≤ n) → (7 * (2 * m - 1) < 150 ∧ (7 * (2 * m - 1)) % 10 = 7) := by
sorry

end multiples_of_7_with_unit_digit_7_and_lt_150_l101_101102


namespace concentric_circles_ratio_l101_101747

theorem concentric_circles_ratio (R r k : ℝ) (hr : r > 0) (hRr : R > r) (hk : k > 0)
  (area_condition : π * (R^2 - r^2) = k * π * r^2) :
  R / r = Real.sqrt (k + 1) :=
by
  sorry

end concentric_circles_ratio_l101_101747


namespace smallest_k_correct_l101_101754

noncomputable def smallest_k (n m : ℕ) (hn : 0 < n) (hm : 0 < m ∧ m ≤ 5) : ℕ :=
    6

theorem smallest_k_correct (n : ℕ) (m : ℕ) (hn : 0 < n) (hm : 0 < m ∧ m ≤ 5) :
  64 ^ smallest_k n m hn hm + 32 ^ m > 4 ^ (16 + n) :=
sorry

end smallest_k_correct_l101_101754


namespace solve_for_n_l101_101766

theorem solve_for_n (n : ℚ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : 
    n = 3.5 :=
  sorry

end solve_for_n_l101_101766


namespace log_a_fraction_inequality_l101_101045

theorem log_a_fraction_inequality (a : ℝ) (h : log a (2 / 3) < 1) : (0 < a ∧ a < 2 / 3) ∨ (1 < a) :=
by {
  sorry
}

end log_a_fraction_inequality_l101_101045


namespace sum_of_divisors_37_l101_101312

theorem sum_of_divisors_37 : ∑ d in (Finset.filter (fun d => d > 0 ∧ 37 % d = 0) (Finset.range (37 + 1))), d = 38 :=
by
  sorry

end sum_of_divisors_37_l101_101312


namespace num_not_factorial_tails_lt_2500_l101_101473

-- Definition of the function f(m)
def f (m : ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625 + m / 78125 + m / 390625 + m / 1953125

-- The theorem stating the main result
theorem num_not_factorial_tails_lt_2500 : 
  let count_non_tails := ∑ k in finset.range 2500, if ∀ m, f m ≠ k then 1 else 0
  in count_non_tails = 500 :=
by
  sorry

end num_not_factorial_tails_lt_2500_l101_101473


namespace problem_statement_l101_101973

theorem problem_statement (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3 * y^2) / 7 = 75 / 7 :=
by 
  -- proof goes here
  sorry

end problem_statement_l101_101973


namespace inverse_function_correct_l101_101700

-- defining the original function for x > 1
def f (x : ℝ) (hx : x > 1) : ℝ := log (2, x / (x - 1))

-- defining the proposed inverse function for x > 0
def g (y : ℝ) (hy : y > 0) : ℝ := 2 ^ y / (2 ^ y - 1)

theorem inverse_function_correct (x : ℝ) (hx : x > 1) : 
  let y := f x hx in g y y > 0 = x := by
  -- proof steps
  sorry

end inverse_function_correct_l101_101700


namespace sin_double_angle_identity_l101_101521

open Real

theorem sin_double_angle_identity {α : ℝ} (h1 : π / 2 < α ∧ α < π) 
    (h2 : sin (α + π / 6) = 1 / 3) :
  sin (2 * α + π / 3) = -4 * sqrt 2 / 9 := 
by 
  sorry

end sin_double_angle_identity_l101_101521


namespace eccentricity_range_l101_101048

noncomputable def hyperbola_eccentricity_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  let c := a * Real.sqrt((b^2/a^2 - 1)) in
  let e := Real.sqrt(1 + (b/a)^2) in
  (Real.sqrt(5) + 1)/2 < e ∧ e < (Real.sqrt(6) + Real.sqrt(2))/2

theorem eccentricity_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  hyperbola_eccentricity_range a b ha hb :=
begin
  sorry
end

end eccentricity_range_l101_101048


namespace evaluate_Q_at_2_l101_101730

-- Define the polynomial Q(x)
noncomputable def Q (x : ℚ) : ℚ := x^4 - 20 * x^2 + 16

-- Define the root condition of sqrt(3) + sqrt(7)
def root_condition (x : ℚ) : Prop := (x = ℚ.sqrt(3) + ℚ.sqrt(7))

-- Theorem statement
theorem evaluate_Q_at_2 : Q 2 = -32 :=
by
  -- Given conditions
  have h1 : degree Q = 4 := sorry,
  have h2 : leading_coeff Q = 1 := sorry,
  have h3 : root_condition (ℚ.sqrt(3) + ℚ.sqrt(7)) := sorry,
  -- Goal
  exact sorry

end evaluate_Q_at_2_l101_101730


namespace feifei_reaches_school_at_828_l101_101842

-- Definitions for all conditions
def start_time : Nat := 8 * 60 + 10  -- Feifei starts walking at 8:10 AM in minutes since midnight
def dog_delay : Nat := 3             -- Dog starts chasing after 3 minutes
def catch_up_200m_time : ℕ := 1      -- Time for dog to catch Feifei at 200 meters
def catch_up_400m_time : ℕ := 4      -- Time for dog to catch Feifei at 400 meters
def school_distance : ℕ := 800       -- Distance from home to school
def feifei_speed : ℕ := 2            -- assumed speed of Feifei where distance covered uniformly
def dog_speed : ℕ := 6               -- dog speed is three times Feifei's speed
def catch_times := [200, 400, 800]   -- Distances (in meters) where dog catches Feifei

-- Derived condition:
def total_travel_time : ℕ := 
  let time_for_200m := catch_up_200m_time + catch_up_200m_time;
  let time_for_400m_and_back := 2* catch_up_400m_time ;
  (time_for_200m + time_for_400m_and_back + (school_distance - 400))

-- The statement we wish to prove:
theorem feifei_reaches_school_at_828 : 
  (start_time + total_travel_time - dog_delay/2) % 60 = 28 :=
sorry

end feifei_reaches_school_at_828_l101_101842


namespace distinct_remainders_l101_101030

theorem distinct_remainders (n : ℕ) (h : n > 1) :
  ∃ (A : Finset ℕ), (A = Finset.range n ∧
  ∀ (i j : ℕ) (hi : i < n) (hj : j < n),
  (i + j) % (n * (n + 1) / 2) ≠ (i' + j') % (n * (n + 1) / 2) 
  → (i = i' ∧ j = j') ∨ (i = j' ∧ j = i')) :=
begin
  use Finset.range n,
  split,
  { refl },
  { intros i j hi hj,
    sorry
  }
end

end distinct_remainders_l101_101030


namespace find_number_l101_101498

theorem find_number (x k : ℕ) (h₁ : x / k = 4) (h₂ : k = 6) : x = 24 := by
  sorry

end find_number_l101_101498


namespace range_a_and_intersections_l101_101553

-- Definitions of the functions f and g
def f (x : ℝ) (a b : ℝ) := (a * Real.log x + b) / x
def g (x : ℝ) (a : ℝ) := a + 2 - x - 2 / x

-- Hypothesis and conditions
variable {a b : ℝ} (ha1 : a ≤ 2) (ha2 : a ≠ 0)
variable (hx : ∀ x : ℝ, x = 1/e → HasExtremum (f x a b))

-- The expression for b derived from the extremum condition
noncomputable def b := 2 * a

-- Statement of the problem in Lean
theorem range_a_and_intersections :
  (∀ x : ℝ, (0 < x ∧ x < 1/e) → (f x a (b a) < 0) ∧ (1/e < x → f x a (b a) > 0) ∨
   (a ∈ (0 : ℝ) ∩ ((-∞) ∪ {2})) ∧
   (∃ x : ℝ, (0 < x ∧ x ≤ 2) ∧ f x a (b a) = g x a ∧
    (a = -1 ∨ a < - 2 / Real.log 2 ∨ 0 < a ∧ a ≤ 2))) :=
sorry

end range_a_and_intersections_l101_101553


namespace sum_of_a_values_l101_101278

theorem sum_of_a_values : 
  (∑ a in {a : ℤ | ∃ x < 0, x^2 + |x - a| - 2 < 0}, a) = -2 :=
by naive_optional
  sorry  -- Proof needs to be filled in

end sum_of_a_values_l101_101278


namespace non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101443

def g (m : Nat) : Nat :=
  let f k := m / k
  f 5 + f 25 + f 125 + f 625 + f 3125 + f 15625 + f 78125 + f 390625 + f 1953125 + f 9765625 + f 48828125 + f 244140625 + f 1220703125 + f 6103515625 + f 30517578125 -- continue as needed

def is_factorial_tail (n : Nat) : Prop :=
  ∃ m : Nat, g m = n

def count_factorial_tails (N : Nat) : Nat :=
  (List.range N).filter (λ n, is_factorial_tail n).length

theorem non_factorial_tails_lemma : count_factorial_tails 2500 = 2000 := sorry

theorem count_non_factorial_tails (N : Nat) : Prop :=
  let total := N - 1
  total - count_factorial_tails N = 499

theorem solution : count_non_factorial_tails 2500 := sorry

end non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101443


namespace angles_equal_l101_101839

-- Definitions based on conditions
structure Triangle (A B C : Type) :=
(equilateral : Prop)

variables {A B C D E H K : Type}
variables [InPlane : A × B × C]
variables (triangle_ABC : Triangle A B C) -- Equilateral triangle ABC
variables (triangle_CDE : Triangle C D E) -- Equilateral triangle CDE
variables (triangle_EHK : Triangle E H K) -- Equilateral triangle EHK
variables (shared_vertex_CE : Prop) -- Triangles share vertices C and E
variables (midpoint_D : D = (AK) / 2) -- D is the midpoint of AK

-- This should prove that the angle DBH equals the angle BDH given the conditions
theorem angles_equal (tABC : triangle_ABC.equilateral) (tCDE : triangle_CDE.equilateral) (tEHK : triangle_EHK.equilateral) 
                      (shared_CE : shared_vertex_CE) (mid_D : midpoint_D) : 
  ∠ DBH = ∠ BDH :=
by sorry

end angles_equal_l101_101839


namespace rock_paper_scissors_probability_l101_101034

noncomputable def probability_beats_all (n k : ℕ) (p : ℚ) : ℚ :=
  let independent_event_prob := p ^ k in
  (n * 3 * independent_event_prob)

theorem rock_paper_scissors_probability :
  probability_beats_all 4 3 (1/3 : ℚ) = (4/27 : ℚ) :=
by
  sorry

end rock_paper_scissors_probability_l101_101034


namespace find_angle_C_range_b_minus_2a_l101_101050

noncomputable def triangle_AC_conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
(a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (C = Real.acos ((a^2 + b^2 - c^2) / (2 * a * b)))

theorem find_angle_C (a b : ℝ) (h₁ : ∀ x : ℝ, 0 < x) : ∃ C, 
  triangle_AC_conditions a b (sqrt 3) A B C ∧ (Real.sin C = sqrt 3 / 2) ∧ (0 < C ∧ C < Real.pi / 2) := 
begin
  sorry
end

theorem range_b_minus_2a (a b : ℝ) (h₁ : ∀ x : ℝ, 0 < x) : 
  ∃ A, (Real.pi / 6 < A) ∧ (A < Real.pi / 2) → 
  (-3 < b - 2 * a) ∧ (b - 2 * a < 0) := 
begin
  sorry
end

end find_angle_C_range_b_minus_2a_l101_101050


namespace journey_total_distance_l101_101115

/--
Given:
- A person covers 3/5 of their journey by train.
- A person covers 7/20 of their journey by bus.
- A person covers 3/10 of their journey by bicycle.
- A person covers 1/50 of their journey by taxi.
- The rest of the journey (4.25 km) is covered by walking.

Prove:
  D = 15.74 km
where D is the total distance of the journey.
-/
theorem journey_total_distance :
  ∀ (D : ℝ), 3/5 * D + 7/20 * D + 3/10 * D + 1/50 * D + 4.25 = D → D = 15.74 :=
by
  intro D
  sorry

end journey_total_distance_l101_101115


namespace solve_servant_payment_problem_l101_101372

/-- A Lean definition to state the problem conditions and the resulting proof statement --/

-- We first define the constants and variables given in the problem.
constant U : ℝ  -- Value of the uniform
constant A : ℝ  -- Agreed amount after one year of service
constant received_amount : ℝ := 250  -- Amount received for 9 months of service

-- Define the statement that expresses the problem conditions
def servant_payment_problem : Prop :=
  -- The servant worked for 9 months, which is 3/4 of a year
  (3 / 4 * A + U = received_amount + U) → 
  -- The agreed amount after one year should be approximately Rs. 333.33
  A ≈ 333.33

-- Insert the "sorry" placeholder to skip the proof
theorem solve_servant_payment_problem : servant_payment_problem :=
  sorry

end solve_servant_payment_problem_l101_101372


namespace application_schemes_eq_l101_101275

noncomputable def number_of_application_schemes (graduates : ℕ) (universities : ℕ) : ℕ :=
  universities ^ graduates

theorem application_schemes_eq : 
  number_of_application_schemes 5 3 = 3 ^ 5 := 
by 
  -- proof goes here
  sorry

end application_schemes_eq_l101_101275


namespace mutually_exclusive_events_l101_101596

-- Definitions based on given conditions
def num_boys := 3
def num_girls := 2
def total_students := num_boys + num_girls

-- Events
def at_least_one_boy (selection : List Nat) : Prop :=
  ∃ b ∈ selection, b < num_boys  -- assuming boys are numbered 0, 1, 2

def all_girls (selection : List Nat) : Prop :=
  ∀ g ∈ selection, g ≥ num_boys  -- assuming girls are numbered 3, 4

-- The proof problem statement
theorem mutually_exclusive_events :
  ∀ (selection : List Nat), 
    (length selection = 2) →
    (at_least_one_boy selection) →
    (all_girls selection) → 
    False :=
by sorry

end mutually_exclusive_events_l101_101596


namespace ways_to_receive_fifth_pass_l101_101818

def players : Fin 3 := ⟨3⟩

def steps : Fin 5 := ⟨5⟩

def pass_to_other (p1 p2 : Fin 3) : Prop := p1 ≠ p2

theorem ways_to_receive_fifth_pass (A B C : Fin 3)
  (h1 : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (first_pass : Fin 3 := A)
  (steps := 5) :
  (∃ (total_ways : Nat), total_ways = 10) :=
sorry

end ways_to_receive_fifth_pass_l101_101818


namespace least_rice_l101_101742

variable (o r : ℝ)

-- Conditions
def condition_1 : Prop := o ≥ 8 + r / 2
def condition_2 : Prop := o ≤ 3 * r

-- The main theorem we want to prove
theorem least_rice (h1 : condition_1 o r) (h2 : condition_2 o r) : r ≥ 4 :=
sorry

end least_rice_l101_101742


namespace num_not_factorial_tails_lt_2500_l101_101469

-- Definition of the function f(m)
def f (m : ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625 + m / 78125 + m / 390625 + m / 1953125

-- The theorem stating the main result
theorem num_not_factorial_tails_lt_2500 : 
  let count_non_tails := ∑ k in finset.range 2500, if ∀ m, f m ≠ k then 1 else 0
  in count_non_tails = 500 :=
by
  sorry

end num_not_factorial_tails_lt_2500_l101_101469


namespace find_complex_conjugate_l101_101525

-- Given conditions
def complex_conjugate (z : ℂ) : ℂ := conj z -- The complex conjugate function
variable (z : ℂ) (h : (1 - complex.i) * z = 1 + complex.i)

-- Proof problem statement: Prove that the complex conjugate of z is -i
theorem find_complex_conjugate : complex_conjugate z = -complex.i := by
  sorry

end find_complex_conjugate_l101_101525


namespace expected_value_red_balls_draws_l101_101687

/-- Let there be a bag containing 4 red balls and 2 white balls,
all of the same size and texture. 
If balls are drawn one after another with replacement from the bag,
and the number of times a red ball is drawn in 6 draws is denoted by 
ξ, then the expected value E(ξ) is 4. -/
theorem expected_value_red_balls_draws :
  let p := 2 / 3 in
  let n := 6 in
  let ξ : ℕ → ℕ := λ k, k  in
  n * p = 4 := 
by
  -- Proof steps would go here
  sorry

end expected_value_red_balls_draws_l101_101687


namespace find_f_x_l101_101040

theorem find_f_x (f : ℝ → ℝ) : (∀ x : ℝ, f(x - 1) = x^2) → (∀ x : ℝ, f(x) = (x + 1)^2) :=
by
  intro h
  sorry

end find_f_x_l101_101040


namespace find_N_l101_101022

def G := 101
def k (N : ℕ) : ℕ := sorry
def m : ℕ := 5161 / G

theorem find_N (N : ℕ) : (N % G = 8) ∧ (5161 % G = 10) → N = 5159 :=
by
  intro h
  cases h with hNmod h5161mod
  have hN : (N = G * k N + 8) := sorry
  have h5161 : (5161 = G * m + 10) := sorry
  sorry

end find_N_l101_101022


namespace caroline_rearrangements_time_l101_101418

theorem caroline_rearrangements_time :
  let rearrangements_count := 8!
  let rearrangements_per_minute := 15
  let total_minutes := rearrangements_count / rearrangements_per_minute
  let total_hours := total_minutes / 60
  total_hours = 44.8 := by
  -- Definitions as conditions
  let rearrangements_count := 8!
  let rearrangements_per_minute := 15
  let total_minutes := rearrangements_count / rearrangements_per_minute
  let total_hours := total_minutes / 60

  -- Sorry to skip the proof
  sorry

end caroline_rearrangements_time_l101_101418


namespace find_min_n_l101_101834

theorem find_min_n (n k : ℕ) (h : 14 * n = k^2) : n = 14 := sorry

end find_min_n_l101_101834


namespace percentage_increase_rate_flow_l101_101697

theorem percentage_increase_rate_flow
  (R1 R2 R3 : ℕ)
  (h1 : R2 = 36)
  (h2 : R3 = 1.25 * R2)
  (h3 : R1 + R2 + R3 = 105) :
  ((R2 - R1) / R1.to_float * 100.0 = 50.0) :=
by 
  sorry

end percentage_increase_rate_flow_l101_101697


namespace solve_alternating_fraction_l101_101481

noncomputable def alternating_fraction : ℝ :=
  3 + 6 / (2 + 6 / (3 + 6 / (2 + 6 / (3 + ...))))

theorem solve_alternating_fraction :
  alternating_fraction = 2 + Real.sqrt 7 :=
sorry

end solve_alternating_fraction_l101_101481


namespace time_spent_on_type_a_problems_l101_101339

theorem time_spent_on_type_a_problems 
  (total_problems : ℕ)
  (exam_time_minutes : ℕ)
  (type_a_problems : ℕ)
  (type_b_problem_time : ℕ)
  (total_time_type_a : ℕ)
  (h1 : total_problems = 200)
  (h2 : exam_time_minutes = 180)
  (h3 : type_a_problems = 50)
  (h4 : ∀ x : ℕ, type_b_problem_time = 2 * x)
  (h5 : ∀ x : ℕ, total_time_type_a = type_a_problems * type_b_problem_time)
  : total_time_type_a = 72 := 
by
  sorry

end time_spent_on_type_a_problems_l101_101339


namespace find_original_number_l101_101797

open Int

theorem find_original_number (N y x : ℕ) (h1 : N = 10 * y + x) (h2 : N + y = 54321) (h3 : x = 54321 % 11) (h4 : y = 4938) : N = 49383 := by
  sorry

end find_original_number_l101_101797


namespace triangle_inequality_proof_l101_101913

theorem triangle_inequality_proof (a b c : ℝ) (h : a + b > c) : a^3 + b^3 + 3 * a * b * c > c^3 :=
by sorry

end triangle_inequality_proof_l101_101913


namespace population_increase_l101_101710

theorem population_increase (P0 P : ℝ) (t : ℝ) (r : ℝ) 
  (h1 : P0 = 60000) 
  (h2 : P = 79860) 
  (h3 : t = 3) 
  (h4 : P = P0 * (1 + r)^t) : r ≈ 0.099 :=
by {
  have h: 79860 = 60000 * (1 + r)^3, from h4,
  calc r ≈ 0.099 : sorry
}

end population_increase_l101_101710


namespace Q_evaluation_at_2_l101_101732

noncomputable def Q : Polynomial ℚ := 
  (Polynomial.X^2 + Polynomial.C 4)^2

theorem Q_evaluation_at_2 : 
  Q.eval 2 = 64 :=
by 
  -- We'll skip the proof as per the instructions.
  sorry

end Q_evaluation_at_2_l101_101732


namespace number_of_cars_sold_next_four_days_cars_sold_each_day_next_four_days_l101_101035

def cars_sold_each_day_first_three_days : ℕ := 5
def days_first_period : ℕ := 3
def quota : ℕ := 50
def cars_remaining_after_next_four_days : ℕ := 23
def days_next_period : ℕ := 4

theorem number_of_cars_sold_next_four_days :
  (quota - cars_sold_each_day_first_three_days * days_first_period) - cars_remaining_after_next_four_days = 12 :=
by
  sorry

theorem cars_sold_each_day_next_four_days :
  (quota - cars_sold_each_day_first_three_days * days_first_period - cars_remaining_after_next_four_days) / days_next_period = 3 :=
by
  sorry

end number_of_cars_sold_next_four_days_cars_sold_each_day_next_four_days_l101_101035


namespace relationship_between_a_b_c_l101_101505

-- Definitions as conditions from a)
def a := 2 / Real.log 4
def b := Real.log 3 / Real.log 2
def c := 3 / 2

-- Statement of the equivalence proof problem
theorem relationship_between_a_b_c : b > c ∧ c > a :=
by {
  sorry
}

end relationship_between_a_b_c_l101_101505


namespace zoo_total_animals_l101_101391

theorem zoo_total_animals (tiger_enclosure : ℕ) (zebra_enclosure_per_tiger : ℕ) 
  (giraffe_enclosures_ratio : ℕ) (tigers_per_enclosure : ℕ) 
  (zebras_per_enclosure : ℕ) (giraffes_per_enclosure : ℕ) 
  (Htiger : tiger_enclosure = 4) (Hzebra_per_tiger : zebra_enclosure_per_tiger = 2) 
  (Hgiraffe_ratio : giraffe_enclosures_ratio = 3) (Ht_pe : tigers_per_enclosure = 4) 
  (Hz_pe : zebras_per_enclosure = 10) (Hg_pe : giraffes_per_enclosure = 2) : 
  let zebra_enclosures := tiger_enclosure * zebra_enclosure_per_tiger in
  let giraffe_enclosures := zebra_enclosures * giraffe_enclosures_ratio in
  let total_tigers := tiger_enclosure * tigers_per_enclosure in
  let total_zebras := zebra_enclosures * zebras_per_enclosure in
  let total_giraffes := giraffe_enclosures * giraffes_per_enclosure in
  total_tigers + total_zebras + total_giraffes = 144 :=
by
  sorry

end zoo_total_animals_l101_101391


namespace largest_prime_factor_divides_sum_l101_101794

theorem largest_prime_factor_divides_sum (seq : List ℕ) (h_seq_len : seq.length = 4)
(h_digits : ∀ i, seq.nth i % 1000 / 100 = seq.nth ((i + 1) % 4) / 1000)
(h_cycle : ∀ i, seq.nth ((i - 1) % 4) % 100 = (seq.nth i % 1000) / 10)
: 101 ∣ seq.sum :=
sorry

end largest_prime_factor_divides_sum_l101_101794


namespace probability_sum9_of_two_dice_is_1_div_9_l101_101305

theorem probability_sum9_of_two_dice_is_1_div_9 :
  let outcomes := { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} },
      favorable := { (x, y) | (x, y) ∈ outcomes ∧ x + y = 9 } in
  (favorable.card : ℚ) / (outcomes.card : ℚ) = 1 / 9 := 
by
  sorry

end probability_sum9_of_two_dice_is_1_div_9_l101_101305


namespace sequence_pairwise_relatively_prime_l101_101907

noncomputable def f (x : ℕ) : ℕ := x^2 - x + 1

theorem sequence_pairwise_relatively_prime (m : ℕ) (h : m > 1) :
  ∀ n k : ℕ, n ≠ k → Nat.Coprime (nat.iterate f n m) (nat.iterate f k m) :=
sorry

end sequence_pairwise_relatively_prime_l101_101907


namespace find_abs_dot_product_l101_101656

variables (a b : ℝ^3)

def norm (v : ℝ^3) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)
def dot (u v : ℝ^3) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
def cross (u v : ℝ^3) : ℝ^3 := (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

noncomputable def abs_dot_product : ℝ :=
  |dot a b|

axioms
  (h1 : norm a = 3)
  (h2 : norm b = 4)
  (h3 : norm (cross a (b + a)) = 15)

theorem find_abs_dot_product :
  abs_dot_product a b = 12 * real.sqrt (1 - 25 / (25 + 6 * abs_dot_product a b)) :=
sorry

end find_abs_dot_product_l101_101656


namespace focus_of_parabola_l101_101494

theorem focus_of_parabola :
  (∃ (x y : ℝ), y = 4 * x ^ 2 - 8 * x - 12 ∧ x = 1 ∧ y = -15.9375) :=
by
  sorry

end focus_of_parabola_l101_101494


namespace abhay_speed_l101_101773

variable {A S : ℝ}

noncomputable theory

theorem abhay_speed (h1 : 30 / A = 30 / S + 2) (h2 : 30 / (2 * A) = 30 / S - 1) : A = 10 :=
sorry

end abhay_speed_l101_101773


namespace least_n_property_l101_101023

theorem least_n_property :
  ∃ (n : ℕ), (∀ (V : finset (ℝ × ℝ)), V.card = 8 ∧ 
    (∀ (p1 p2 p3 : ℝ × ℝ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬collinear p1 p2 p3) →
    (∀ (E : finset (ℝ × ℝ × ℝ × ℝ)), E.card = n →
      ∃ (line : ℝ × ℝ → Prop), (∃ (S : finset (ℝ × ℝ × ℝ × ℝ)), S ⊆ E ∧ S.card = 4 ∧ 
        ∀ (seg : ℝ × ℝ × ℝ × ℝ), seg ∈ S → line (intrPoint seg))))
  := sorry

end least_n_property_l101_101023


namespace solution_set_of_inequality_l101_101071

def f (x : ℝ) : ℝ :=
  2016^x + Real.logBase 2016 (Real.sqrt (x^2 + 1) + x) - 2016^(-x) + 2

theorem solution_set_of_inequality :
  { x : ℝ | f (3 * x + 1) + f x > 4 } = Set.Ioi (-1 / 4) :=
sorry

end solution_set_of_inequality_l101_101071


namespace jackson_grade_l101_101631

open Function

theorem jackson_grade :
  ∃ (grade : ℕ), 
  ∀ (hours_playing hours_studying : ℕ), 
    (hours_playing = 9) ∧ 
    (hours_studying = hours_playing / 3) ∧ 
    (grade = hours_studying * 15) →
    grade = 45 := 
by {
  sorry
}

end jackson_grade_l101_101631


namespace find_original_number_l101_101805

-- Variables and assumptions
variables (N y x : ℕ)

-- Conditions
def crossing_out_condition : Prop := N = 10 * y + x
def sum_condition : Prop := N + y = 54321
def remainder_condition : Prop := x = 54321 % 11
def y_value : Prop := y = 4938

-- The final proof problem statement
theorem find_original_number (h1 : crossing_out_condition N y x) (h2 : sum_condition N y) 
  (h3 : remainder_condition x) (h4 : y_value y) : N = 49383 :=
sorry

end find_original_number_l101_101805


namespace sphere_volume_ratio_l101_101586

theorem sphere_volume_ratio
  (r R : ℝ)
  (h : (4:ℝ) * π * r^2 / (4 * π * R^2) = (4:ℝ) / 9) : 
  (r^3 / R^3 = (8:ℝ) / 27) := by
  sorry

end sphere_volume_ratio_l101_101586


namespace roots_of_quadratic_l101_101122

theorem roots_of_quadratic (b c : ℝ) (h1 : 1 + -2 = -b) (h2 : 1 * -2 = c) : b = 1 ∧ c = -2 :=
by
  sorry

end roots_of_quadratic_l101_101122


namespace n_divisible_by_3_l101_101608

variables {n : ℕ}
variables (S_col S_row : ℕ → ℕ)

def exists_valid_n (n : ℕ) :=
  ∀ k : ℕ, k ≥ 1 ∧ k ≤ n → S_row k = S_col k - 1 ∨ S_row k = S_col k + 2

theorem n_divisible_by_3 (h : exists_valid_n n S_col S_row) : 3 ∣ n :=
sorry

end n_divisible_by_3_l101_101608


namespace identify_false_coin_bag_triangle_area_less_than_100_smallest_whole_number_lucky_ticket_sum_divisible_by_13_l101_101351

-- Problem 1
theorem identify_false_coin_bag (W : ℕ) (W_honest : W =  10 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10)) :
  ∃ n : ℕ,
    (n = W - 549) := 
by sorry

-- Problem 2
theorem triangle_area_less_than_100 
  (a b c h_a h_b h_c : ℝ) 
  (h_a_lt_1 : h_a < 1) 
  (h_b_lt_1 : h_b < 1) 
  (h_c_lt_1 : h_c < 1) :
    ∃ A : ℝ, 
      A < 100 := 
by sorry

-- Problem 3
theorem smallest_whole_number : 
  ∃ x : ℕ, 
    (∃ a b c : ℕ, 
      2 * a * a = x ∧ 3 * b^3 = x ∧ 5 * c^5 = x) ∧  x = 10125000 := 
by sorry

-- Problem 4
theorem lucky_ticket_sum_divisible_by_13 : 
  (∑ i in finset.range 999999, 
    i = N) → 
    N % 13 = 0 := 
by sorry

end identify_false_coin_bag_triangle_area_less_than_100_smallest_whole_number_lucky_ticket_sum_divisible_by_13_l101_101351


namespace change_factor_l101_101216

theorem change_factor (avg1 avg2 : ℝ) (n : ℕ) (h_avg1 : avg1 = 40) (h_n : n = 10) (h_avg2 : avg2 = 80) : avg2 * (n : ℝ) / (avg1 * (n : ℝ)) = 2 :=
by
  sorry

end change_factor_l101_101216


namespace residue_of_f_2015_l101_101857

noncomputable def f : ℕ → ℕ
| 0     := 1
| 1 := 1
| 2 := 1
| (n+3) := f n + f (n + 2) + 1

theorem residue_of_f_2015 : (f 2015) % 4 = 3 := 
sorry

end residue_of_f_2015_l101_101857


namespace unique_n_degree_polynomial_l101_101956

noncomputable def P (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem unique_n_degree_polynomial (a b c : ℝ) (h : a ≠ 0) (n : ℕ) :
  ∃ at_most_one (Q : ℝ → ℝ), degree Q = n ∧ ∀ x : ℝ, (a ≠ 0) → Q (P a b c x) = P a b c (Q x) :=
sorry

end unique_n_degree_polynomial_l101_101956


namespace correct_propositions_are_1_and_3_l101_101564

noncomputable def proposition1 := ∀ (b c : ℝ), ∃ m ∈ ℝ, ∀ x ∈ ℝ, x^2 + b*x + c ≥ m
noncomputable def proposition2 := ¬ (∃ (p1 p2 d1 d2 : ℕ) (h1 : p1 + d1 = 5) (h2 : p2 + d2 = 3),
  (2 * 1 * choose 3 2 / (choose 5 3)) = 1)
noncomputable def proposition3 := 12 = 6 * 2

theorem correct_propositions_are_1_and_3 : proposition1 ∧ proposition3 ∧ proposition2 := 
by
  split
  {
    sorry
  },
  split
  {
    sorry
  },
  sorry

end correct_propositions_are_1_and_3_l101_101564


namespace slopes_product_of_tangents_l101_101177

theorem slopes_product_of_tangents 
  (x₀ y₀ : ℝ) 
  (h_hyperbola : (2 * x₀^2) / 3 - y₀^2 / 6 = 1) 
  (h_outside_circle : x₀^2 + y₀^2 > 2) :
  ∃ (k₁ k₂ : ℝ), 
    k₁ * k₂ = 4 ∧ 
    (y₀ - k₁ * x₀)^2 + k₁^2 = 2 ∧ 
    (y₀ - k₂ * x₀)^2 + k₂^2 = 2 :=
by {
  -- this proof will use the properties of tangents to a circle and the constraints given
  -- we don't need to implement it now, but we aim to show the correct relationship
  sorry
}

end slopes_product_of_tangents_l101_101177


namespace shortest_distance_between_circles_l101_101307

noncomputable def circle1_center : ℝ × ℝ :=
  (3, 4)

noncomputable def circle1_radius : ℝ :=
  real.sqrt 26

noncomputable def circle2_center : ℝ × ℝ :=
  (-5, 3)

noncomputable def circle2_radius : ℝ :=
  real.sqrt 59

noncomputable def distance_between_centers : ℝ :=
  real.sqrt ((3 - (-5))^2 + (4 - 3)^2)

noncomputable def shortest_distance : ℝ :=
  distance_between_centers - (circle1_radius + circle2_radius)

theorem shortest_distance_between_circles : shortest_distance = 0 :=
sorry

end shortest_distance_between_circles_l101_101307


namespace smallest_m_for_2n_roots_of_unity_l101_101644

noncomputable def T : set ℂ := 
  {z : ℂ | ∃ u v : ℝ, z = complex.mk u v ∧ (real.sqrt 3 / 3 ≤ u ∧ u ≤ real.sqrt 3 / 2)}

theorem smallest_m_for_2n_roots_of_unity : 
  ∃ m : ℕ, m = 16 ∧ ∀ n : ℕ, n ≥ m → (∃ z ∈ T, z ^ (2 * n) = 1) :=
begin
  sorry -- Proof goes here
end

end smallest_m_for_2n_roots_of_unity_l101_101644


namespace find_original_number_l101_101798

open Int

theorem find_original_number (N y x : ℕ) (h1 : N = 10 * y + x) (h2 : N + y = 54321) (h3 : x = 54321 % 11) (h4 : y = 4938) : N = 49383 := by
  sorry

end find_original_number_l101_101798


namespace no_extreme_points_l101_101478

noncomputable def f (x a : ℝ) : ℝ := x^3 + 3 * x^2 + 4 * x - a

theorem no_extreme_points (a : ℝ) : ∀ x : ℝ, 3 * x^2 + 6 * x + 4 ≠ 0 :=
by
  intro x
  have h : 3 * (x + 1)^2 + 1 > 0 := by
    simp [pow_two, mul_add, mul_one, add_assoc]
    exact add_pos_of_nonneg_of_pos (mul_nonneg zero_le_three (pow_two_nonneg (x + 1))) zero_lt_one
  linarith
  sorry

end no_extreme_points_l101_101478


namespace imaginary_part_conjugate_l101_101067

noncomputable def i := Complex.I

theorem imaginary_part_conjugate (z : ℂ) (h : (1 - i) * z = i ^ 2015) : Complex.im (Complex.conj z) = 1 / 2 :=
by
  sorry

end imaginary_part_conjugate_l101_101067


namespace solve_for_x_l101_101208

theorem solve_for_x (x : ℝ) (h : log10 (3 * x + 4) = 1) : x = 2 :=
by
  sorry

end solve_for_x_l101_101208


namespace problem_statement_l101_101160

theorem problem_statement 
  (x y : ℕ)
  (h1 : 2^x = 180.gcd 2^180)
  (h2 : 3^y = 180.gcd 3^180) :
  ((1 / 7) : ℝ)^(y - x) = 1 :=
by sorry

end problem_statement_l101_101160


namespace sum_divisors_of_37_is_38_l101_101320

theorem sum_divisors_of_37_is_38 (h : Nat.Prime 37) : (∑ d in (Finset.filter (λ d, 37 % d = 0) (Finset.range (37 + 1))), d) = 38 := by
  sorry

end sum_divisors_of_37_is_38_l101_101320


namespace transformed_inequality_l101_101326

theorem transformed_inequality (x : ℝ) : 
  (x - 3) / 3 < (2 * x + 1) / 2 - 1 ↔ 2 * (x - 3) < 3 * (2 * x + 1) - 6 :=
by
  sorry

end transformed_inequality_l101_101326


namespace find_k_l101_101672

noncomputable def circle_center : ℝ × ℝ := (1, 0)
noncomputable def circle_radius : ℝ := 1
noncomputable def line (k : ℝ) : ℝ × ℝ → Prop := λ P, k * P.1 + P.2 + 3 = 0
noncomputable def area_quadrilateral (P : ℝ × ℝ)(k : ℝ) : ℝ := 
  -- an expression that computes the area of quadrilateral PACB involving point P and k
  -- this expression should be derived based on geometric properties
  sorry

theorem find_k (k : ℝ) (h_pos : k > 0) :
  (∃ P, line k P ∧ area_quadrilateral P k = 2) → k = 2 :=
by sorry

end find_k_l101_101672


namespace range_of_f_l101_101927

open Real

def condition1 (m α β : ℝ) := (tan α + tan β = (1 - 2 * m) / m) ∧ (tan α * tan β = (2 * m - 3) / (2 * m)) ∧ (α ≠ β)
def condition2 (m : ℝ) := m ∈ set.Icc (-1/2) 0 ∪ set.Icc 0 (∞)
noncomputable def f (m : ℝ) : ℝ := 5 * m^2 + 3 * m * (tan ((α + β))) + 4

theorem range_of_f :
  ∀ m : ℝ, condition1 m α β ∧ condition2 m → f(m) ∈ set.Icc (13/4) 4 ∪ set.Icc 4 (∞) := by
  sorry

end range_of_f_l101_101927


namespace count_non_factorial_tails_lt_2500_l101_101448

def f (m : ℕ) : ℕ := 
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + ... -- Continue as needed

theorem count_non_factorial_tails_lt_2500 : 
  #{ n : ℕ | n < 2500 ∧ ¬ (∃ m : ℕ, f(m) = n) } = 4 :=
by sorry

end count_non_factorial_tails_lt_2500_l101_101448


namespace coefficient_x4_expansion_l101_101618

theorem coefficient_x4_expansion :
  let expr := (2 * x ^ 2 - 1 / x) ^ 5
  ∃ (c : ℤ), c = 80 ∧ ∃ (t : ℕ → ℝ) (r : ℕ),
    t r ≠ 0 ∧
    expr.expandCoef r = c ∧
    expr.expandPower r = 4 :=
sorry

end coefficient_x4_expansion_l101_101618


namespace SomeAthletesNotHonorSociety_l101_101407

variable (Athletes HonorSociety : Type)
variable (Discipline : Athletes → Prop)
variable (isMember : Athletes → HonorSociety → Prop)

-- Some athletes are not disciplined
axiom AthletesNotDisciplined : ∃ a : Athletes, ¬Discipline a

-- All members of the honor society are disciplined
axiom AllHonorSocietyDisciplined : ∀ h : HonorSociety, ∀ a : Athletes, isMember a h → Discipline a

-- The theorem to be proved
theorem SomeAthletesNotHonorSociety : ∃ a : Athletes, ∀ h : HonorSociety, ¬isMember a h :=
  sorry

end SomeAthletesNotHonorSociety_l101_101407


namespace minimum_tan_product_l101_101605

theorem minimum_tan_product 
  {A B C : ℝ} 
  (h1 : A + B + C = π)
  (h2 : A < π / 2)
  (h3 : B < π / 2)
  (h4 : C < π / 2)
  (h5 : ∃ a b c : ℝ, 
        b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C)
  (h6 : Real.sin A = 1/2):
  ∃ (B C : ℝ), 
  (π / 6 < B) ∧ (B < π / 2) ∧ (π / 6 < C) ∧ (C < π / 2) ∧ 
  tan A * tan B * tan C = (12 + 7 * Real.sqrt 3) / 3 := 
by
  sorry

end minimum_tan_product_l101_101605


namespace child_grandmother_ratio_l101_101235

def grandmother_weight (G D C : ℝ) : Prop :=
  G + D + C = 160

def daughter_child_weight (D C : ℝ) : Prop :=
  D + C = 60

def daughter_weight (D : ℝ) : Prop :=
  D = 40

theorem child_grandmother_ratio (G D C : ℝ) (h1 : grandmother_weight G D C) (h2 : daughter_child_weight D C) (h3 : daughter_weight D) :
  C / G = 1 / 5 :=
sorry

end child_grandmother_ratio_l101_101235


namespace vince_savings_l101_101296

-- Definitions based on conditions
def earnings_per_customer : ℕ := 18
def monthly_expenses : ℤ := 280
def recreation_percentage : ℚ := 0.20
def customers_served : ℕ := 80

-- Intermediate calculations (explained in the solution steps but not used directly here)
def total_earnings : ℕ := customers_served * earnings_per_customer
def recreation_expenses : ℤ := (recreation_percentage.toReal * total_earnings).to_nat
def total_monthly_expenses : ℤ := monthly_expenses + recreation_expenses
def savings : ℤ := total_earnings - total_monthly_expenses

-- The proof statement we need to verify
theorem vince_savings :
  savings = 872 :=
sorry

end vince_savings_l101_101296


namespace non_factorial_tails_lt_2500_l101_101437

-- Define the function f(m)
def f (m: ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625

-- Define what it means to be a factorial tail
def is_factorial_tail (n: ℕ) : Prop :=
  ∃ m : ℕ, f(m) = n

-- Define the statement to be proved
theorem non_factorial_tails_lt_2500 : 
  (finset.range 2500).filter (λ n, ¬ is_factorial_tail n).card = 499 :=
by
  sorry

end non_factorial_tails_lt_2500_l101_101437


namespace ratio_first_term_common_difference_l101_101480

theorem ratio_first_term_common_difference
  (a d : ℚ)
  (h : (15 / 2) * (2 * a + 14 * d) = 4 * (8 / 2) * (2 * a + 7 * d)) :
  a / d = -7 / 17 := 
by {
  sorry
}

end ratio_first_term_common_difference_l101_101480


namespace maximum_area_of_quadrilateral_l101_101909

theorem maximum_area_of_quadrilateral (R k : ℝ) (k_pos : 0 ≤ k) : 
  ∃ (S_max : ℝ),
    (S_max = (k + 1) * R^2 ∧ k ≤ real.sqrt 2 - 1) ∨
    (S_max = (2 * real.sqrt (k * (k + 2)) / (k + 1)) * R^2 ∧ k > real.sqrt 2 - 1) :=
sorry

end maximum_area_of_quadrilateral_l101_101909


namespace number_of_factors_in_224_l101_101795

def smallest_is_half_largest (n1 n2 : ℕ) : Prop :=
  n1 * 2 = n2

theorem number_of_factors_in_224 :
  ∃ n1 n2 n3 : ℕ, n1 * n2 * n3 = 224 ∧ smallest_is_half_largest (min n1 (min n2 n3)) (max n1 (max n2 n3)) ∧
    (if h : n1 < n2 ∧ n1 < n3 then
      if h2 : n2 < n3 then 
        smallest_is_half_largest n1 n3 
        else 
        smallest_is_half_largest n1 n2 
    else if h : n2 < n1 ∧ n2 < n3 then 
      if h2 : n1 < n3 then 
        smallest_is_half_largest n2 n3 
        else 
        smallest_is_half_largest n2 n1 
    else 
      if h2 : n1 < n2 then 
        smallest_is_half_largest n3 n2 
        else 
        smallest_is_half_largest n3 n1) = true ∧ 
    (if h : n1 < n2 ∧ n1 < n3 then
       if h2 : n2 < n3 then 
         n1 * n2 * n3 
         else 
         n1 * n3 * n2 
     else if h : n2 < n1 ∧ n2 < n3 then 
       if h2 : n1 < n3 then 
         n2 * n1 * n3
         else 
         n2 * n3 * n1 
     else 
       if h2 : n1 < n2 then 
         n3 * n1 * n2 
         else 
         n3 * n2 * n1) = 224 := sorry

end number_of_factors_in_224_l101_101795


namespace count_axisymmetric_shapes_eq_four_l101_101560

-- Define the given shapes
inductive Shape
| line_segment
| right_angle
| isosceles_triangle
| parallelogram
| rectangle

open Shape

-- Define a function to check if a shape is axisymmetric
def is_axisymmetric : Shape → Bool
| line_segment := true
| right_angle := true
| isosceles_triangle := true
| parallelogram := false
| rectangle := true

-- Define a function to count the number of axisymmetric shapes in a list
def count_axisymmetric (shapes : List Shape) : Nat :=
  shapes.filter is_axisymmetric |>.length

-- Define the list of given shapes
def given_shapes : List Shape := [line_segment, right_angle, isosceles_triangle, parallelogram, rectangle]

-- The proof problem statement
theorem count_axisymmetric_shapes_eq_four :
  count_axisymmetric given_shapes = 4 :=
by
  sorry

end count_axisymmetric_shapes_eq_four_l101_101560


namespace water_left_in_cooler_l101_101328

def initial_gallons := 3
def ounces_per_gallon := 128
def dixie_cup_ounces := 6
def rows_of_chairs := 5
def chairs_per_row := 10

theorem water_left_in_cooler :
  let initial_ounces := initial_gallons * ounces_per_gallon,
      total_chairs := rows_of_chairs * chairs_per_row,
      total_ounces_used := total_chairs * dixie_cup_ounces,
      final_ounces := initial_ounces - total_ounces_used
  in final_ounces = 84 :=
  sorry

end water_left_in_cooler_l101_101328


namespace negation_of_implication_iff_l101_101238

variable (a : ℝ)

theorem negation_of_implication_iff (p : a > 1 → a^2 > 1) :
  ¬(a > 1 → a^2 > 1) ↔ (a ≤ 1 → a^2 ≤ 1) :=
by sorry

end negation_of_implication_iff_l101_101238


namespace pencils_given_l101_101487

theorem pencils_given (initial_pencils : ℕ) (total_pencils : ℕ) (received_pencils : ℕ) 
  (h_initial : initial_pencils = 51) (h_total : total_pencils = 57) :
  received_pencils = total_pencils - initial_pencils :=
by
  rw [h_initial, h_total]
  exact Nat.sub_self 51

end pencils_given_l101_101487


namespace angle_between_vectors_l101_101567

variable (a : ℝ × ℝ := (1, sqrt 3))
variable (b : ℝ × ℝ := (3, m))
variable (m : ℝ)
variable (projection_cond : (a.1 * b.1 + a.2 * b.2) / (sqrt (a.1^2 + a.2^2)) = 3)

theorem angle_between_vectors (a : ℝ × ℝ) (b : ℝ × ℝ) (h : (a.1 * b.1 + a.2 * b.2) / (sqrt (a.1^2 + a.2^2)) = 3) : 
  ∃ θ : ℝ, θ = π / 6 :=
by
  -- Placeholder for proof
  sorry

end angle_between_vectors_l101_101567


namespace tangent_line_equation_and_triangle_area_l101_101936

theorem tangent_line_equation_and_triangle_area
  (curve : ℝ → ℝ)
  (tangent_l1 : ℝ → ℝ)
  (tangent_l2 : ℝ → ℝ)
  (tangent_at_point : tangent_l1 = λ x, 3 * x - 3)
  (tangent_curve_relationship : ∀ x, tangent_l1 x = x^2 + x - 2 ∧
                                     tangent_l2 x = (2 * (-2/3) + 1) * x - ((-2/3)^2 + (-2/3) - 2))
  (perpendicular_relationship : ∀ x, (tangent_l1 x).derivative * (tangent_l2 x).derivative = -1) :
  (∀ x, tangent_l2 x = -1/3 * x - 22/9) ∧ 
  (1/2 * 25/3 * 5/2 = 125/12) :=
by
  sorry

end tangent_line_equation_and_triangle_area_l101_101936


namespace no_psafe_numbers_l101_101499

def is_psafe (n p : ℕ) : Prop := 
  ¬ (n % p = 0 ∨ n % p = 1 ∨ n % p = 2 ∨ n % p = 3 ∨ n % p = p - 3 ∨ n % p = p - 2 ∨ n % p = p - 1)

theorem no_psafe_numbers (N : ℕ) (hN : N = 10000) :
  ∀ n, (n ≤ N ∧ is_psafe n 5 ∧ is_psafe n 7 ∧ is_psafe n 11) → false :=
by
  sorry

end no_psafe_numbers_l101_101499


namespace range_m_l101_101224
noncomputable def z (m : ℝ) : ℂ := 1 + complex.I + m / (1 + complex.I)

theorem range_m (m : ℝ) :
  (1 + (m / 2) : ℝ) > 0 ∧ (1 - (m / 2) : ℝ) > 0 ↔ -2 < m ∧ m < 2 :=
by {
  sorry
}

end range_m_l101_101224


namespace non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101442

def g (m : Nat) : Nat :=
  let f k := m / k
  f 5 + f 25 + f 125 + f 625 + f 3125 + f 15625 + f 78125 + f 390625 + f 1953125 + f 9765625 + f 48828125 + f 244140625 + f 1220703125 + f 6103515625 + f 30517578125 -- continue as needed

def is_factorial_tail (n : Nat) : Prop :=
  ∃ m : Nat, g m = n

def count_factorial_tails (N : Nat) : Nat :=
  (List.range N).filter (λ n, is_factorial_tail n).length

theorem non_factorial_tails_lemma : count_factorial_tails 2500 = 2000 := sorry

theorem count_non_factorial_tails (N : Nat) : Prop :=
  let total := N - 1
  total - count_factorial_tails N = 499

theorem solution : count_non_factorial_tails 2500 := sorry

end non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101442


namespace sum_f_sigma_over_permutations_l101_101029

open Finset

-- Definitions for transpositions and permutations
def transposition (n : ℕ) := Σ i j : Fin n, i ≠ j

def f_sigma (σ : Equiv.Perm (Fin 7)) : ℕ :=
  σ.support.card - 1

-- Main statement to prove
theorem sum_f_sigma_over_permutations : 
  (∑ σ in univ (Equiv.Perm (Fin 7)), f_sigma σ) = 22212 :=
sorry

end sum_f_sigma_over_permutations_l101_101029


namespace non_factorial_tails_lt_2500_l101_101434

-- Define the function f(m)
def f (m: ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625

-- Define what it means to be a factorial tail
def is_factorial_tail (n: ℕ) : Prop :=
  ∃ m : ℕ, f(m) = n

-- Define the statement to be proved
theorem non_factorial_tails_lt_2500 : 
  (finset.range 2500).filter (λ n, ¬ is_factorial_tail n).card = 499 :=
by
  sorry

end non_factorial_tails_lt_2500_l101_101434


namespace problem_l101_101562

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

axiom universal_set : U = {1, 2, 3, 4, 5, 6, 7}
axiom set_M : M = {3, 4, 5}
axiom set_N : N = {1, 3, 6}

def complement (U M : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ M}

theorem problem :
  {1, 6} = (complement U M) ∩ N :=
by
  sorry

end problem_l101_101562


namespace ways_to_choose_organizers_and_leaders_l101_101191

theorem ways_to_choose_organizers_and_leaders : 
  ∃ (n k : ℕ), n = 6 ∧ k = 3 ∧ (nat.choose n k) * (k * k) = 180 := 
by
  use 6
  use 3
  split
  { refl }
  split
  { refl }
  simp only [nat.choose, nat.factorial]
  norm_num
  sorry

end ways_to_choose_organizers_and_leaders_l101_101191


namespace find_points_on_number_line_l101_101519

noncomputable def numbers_are_opposite (x y : ℝ) : Prop :=
  x = -y

theorem find_points_on_number_line (A B : ℝ) 
  (h1 : numbers_are_opposite A B) 
  (h2 : |A - B| = 8) 
  (h3 : A < B) : 
  (A = -4 ∧ B = 4) :=
by
  sorry

end find_points_on_number_line_l101_101519


namespace not_factorial_tails_count_l101_101458

def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) -- you can extend further if needed

def is_factorial_tail (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ f(m) = n

theorem not_factorial_tails_count : 
  ∃ c : ℕ, c = 500 ∧ ∀ k : ℕ, k < 2500 → ¬is_factorial_tail k → k < 2500 - c :=
by
  sorry

end not_factorial_tails_count_l101_101458


namespace number_of_towers_l101_101359

-- Variables representing the number of cubes of each color
def num_red_cubes : ℕ := 3
def num_blue_cubes : ℕ := 3
def num_green_cubes : ℕ := 3
def tower_height : ℕ := 7

-- Function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Sum of valid combinations for creating the towers
def valid_combinations_sum : ℕ := 
  (binomial 9 7) * ∑ r in Finset.Icc 0 num_red_cubes,
  ∑ b in Finset.Icc 0 num_blue_cubes,
  ∑ g in Finset.Icc 0 num_green_cubes, 
  if r + b + g = tower_height then
    Nat.factorial tower_height / (Nat.factorial r * Nat.factorial b * Nat.factorial g)
  else
    0

-- Theorem to prove the number of different towers is 140
theorem number_of_towers : valid_combinations_sum = 140 := 
  sorry

end number_of_towers_l101_101359


namespace angle_AMC_equals_angle_BMP_l101_101916

variables {A B C M P : Type}
variables (triangle_ABC : is_isosceles_triangle A B C)
variables (midpoint_M : is_midpoint M A B)
variables (right_angle_A : ∠A = 90)
variables (perpendicular_line : ∀ (line_CM : ∃ line_CM, is_perpendicular line_CM CM ∧ passes_through line_CM A), intersects (pass_through A (perpendicular_to CM)) BC P)

theorem angle_AMC_equals_angle_BMP :
  ∠AMC = ∠BMP :=
sorry

end angle_AMC_equals_angle_BMP_l101_101916


namespace fruit_basket_count_l101_101571

theorem fruit_basket_count :
  let apples := 6
  let oranges := 8
  let min_apples := 2
  let min_fruits := 1
  (0 <= oranges ∧ oranges <= 8) ∧ (min_apples <= apples ∧ apples <= 6) ∧ (min_fruits <= (apples + oranges)) →
  (5 * 9 = 45) :=
by
  intro h
  sorry

end fruit_basket_count_l101_101571


namespace net_gain_of_C_l101_101669

theorem net_gain_of_C : 
  let initial_value : ℕ := 20000,
      selling_price : ℕ := initial_value * 6 / 5, -- 20% profit
      buying_price : ℕ := selling_price * 17 / 20 -- 15% loss
  in selling_price - buying_price = 3600 :=
by 
  sorry

end net_gain_of_C_l101_101669


namespace value_of_place_ratio_l101_101624

theorem value_of_place_ratio :
  let d8_pos := 10000
  let d6_pos := 0.1
  d8_pos = 100000 * d6_pos :=
by
  let d8_pos := 10000
  let d6_pos := 0.1
  sorry

end value_of_place_ratio_l101_101624


namespace true_propositions_l101_101836

theorem true_propositions :
  (("if p then q" and "if ¬q then ¬p" are contrapositive propositions of each other) = true) ∧
  ((am^2 < bm^2 is a necessary and sufficient condition for a < b) = false) ∧
  ((the negation of the proposition "the diagonals of a rectangle are equal in length" is false) = false) ∧
  ((the roots of the equation 2*x^2 - 5*x + 2 = 0 can represent the eccentricities of an ellipse and a hyperbola) = true) ∧
  ((the directrix equation of the parabola y = 4*x^2 is y = -1) = false) ↔
  ([1, 4] = [1, 4]) := 
by
  sorry

end true_propositions_l101_101836


namespace rita_months_needed_l101_101995

noncomputable def total_hours_needed : ℕ := 1500
noncomputable def hours_backstroke : ℕ := 50
noncomputable def hours_breaststroke : ℕ := 9
noncomputable def hours_butterfly : ℕ := 121
noncomputable def monthly_hours : ℕ := 220
noncomputable def completed_hours_so_far : ℕ := hours_backstroke + hours_breaststroke + hours_butterfly
noncomputable def remaining_hours : ℕ := total_hours_needed - completed_hours_so_far
noncomputable def months_needed : ℕ := remaining_hours / monthly_hours

theorem rita_months_needed : months_needed = 6 :=
by
  rw [total_hours_needed, completed_hours_so_far, hours_backstroke, hours_breaststroke, hours_butterfly, monthly_hours, remaining_hours, months_needed]
  exact sorry

end rita_months_needed_l101_101995


namespace ch_sub_ch_add_sh_sub_sh_add_l101_101953

noncomputable def sh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def ch (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

theorem ch_sub (x y : ℝ) : ch (x - y) = ch x * ch y - sh x * sh y := sorry
theorem ch_add (x y : ℝ) : ch (x + y) = ch x * ch y + sh x * sh y := sorry
theorem sh_sub (x y : ℝ) : sh (x - y) = sh x * ch y - ch x * sh y := sorry
theorem sh_add (x y : ℝ) : sh (x + y) = sh x * ch y + ch x * sh y := sorry

end ch_sub_ch_add_sh_sub_sh_add_l101_101953


namespace sum_of_other_endpoint_coordinates_l101_101708

theorem sum_of_other_endpoint_coordinates
  (x₁ y₁ x₂ y₂ : ℝ)
  (hx : (x₁ + x₂) / 2 = 5)
  (hy : (y₁ + y₂) / 2 = -8)
  (endpt1 : x₁ = 7)
  (endpt2 : y₁ = -2) :
  x₂ + y₂ = -11 :=
sorry

end sum_of_other_endpoint_coordinates_l101_101708


namespace proof_problem_l101_101911

noncomputable def line_l (t α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, 2 + t * Real.sin α)

noncomputable def curve_C (θ ρ : ℝ) : Prop :=
  ρ * (Real.cos θ)^2 = 4 * Real.sin θ

theorem proof_problem {α : ℝ} (hα : 0 ≤ α ∧ α < Real.pi) :
  let l_general_eqn := (α = Real.pi / 6) → (∀ t : ℝ, line_l t (Real.pi / 6) = (t * (Real.cos (Real.pi / 6)), 2 + t * (Real.sin (Real.pi / 6)))) → ∃ a b c, a * line_l t (Real.pi / 6) + b * line_l t (Real.pi / 6) + c = 0
  ∧ (∀ θ ρ : ℝ, curve_C θ ρ → ∃ x y, x^2 = 4*y)
  ∧ (∀ A B : ℝ × ℝ, l_general_eqn → curve_C A.2 A.1 → curve_C B.2 B.1 → ∃ min_dist : ℝ, dist A B = 4 * Real.sqrt 2) := sorry

end proof_problem_l101_101911


namespace triangle_area_l101_101079

variables (a b c : ℝ)
noncomputable def sin (x : ℝ) := Real.sin x

theorem triangle_area (h1 : a^2 = b^2 + c^2 - b * c) (h2 : b * c = 16) :
  (1 / 2) * b * c * sin (real.pi / 3) = 4 * Real.sqrt 3 :=
by
  sorry

end triangle_area_l101_101079


namespace line_l_standard_eq_dot_product_constant_l101_101518

section
variables {t α : ℝ} (θ ρ : ℝ)

-- Conditions:
-- Parametric equation of line l
def line_l (t α : ℝ) : ℝ × ℝ := (1 + t * real.cos α, t * real.sin α)

-- Polar equation of curve C
def curve_C (ρ θ : ℝ) : Prop :=
  ρ * real.cos θ ^ 2 + 4 * real.cos θ = ρ ∧ (ρ ≥ 0) ∧ (0 ≤ θ ∧ θ ≤ 2 * real.pi)

-- Prove:
theorem line_l_standard_eq (α : ℝ) (hα : α = real.pi / 3) :
  let (x, y) := line_l t (real.pi / 3) in y = real.sqrt 3 * (x - 1) :=
by
  sorry

theorem dot_product_constant {A B : ℝ × ℝ} (hA : A ∈ set.range (λ (t : ℝ), line_l t (real.pi / 3)))
  (hB : B ∈ set.range (λ (t : ℝ), line_l t (real.pi / 3)))
  (hCA : curve_C (real.sqrt (A.1 ^ 2 + A.2 ^ 2)) (real.atan2 A.2 A.1))
  (hCB : curve_C (real.sqrt (B.1 ^ 2 + B.2 ^ 2)) (real.atan2 B.2 B.1)) :
  A.1 * B.1 + A.2 * B.2 = -3 :=
by 
  sorry

end

end line_l_standard_eq_dot_product_constant_l101_101518


namespace allowance_amount_l101_101398

variable (initial_money spent_money final_money : ℕ)

theorem allowance_amount (initial_money : ℕ) (spent_money : ℕ) (final_money : ℕ) (h1: initial_money = 5) (h2: spent_money = 2) (h3: final_money = 8) : (final_money - (initial_money - spent_money)) = 5 := 
by 
  sorry

end allowance_amount_l101_101398


namespace intensity_on_Thursday_l101_101827

-- Step a) - Definitions from Conditions
def inversely_proportional (i b k : ℕ) : Prop := i * b = k

-- Translation of the proof problem
theorem intensity_on_Thursday (k b : ℕ) (h₁ : k = 24) (h₂ : b = 3) : ∃ i, inversely_proportional i b k ∧ i = 8 := 
by
  sorry

end intensity_on_Thursday_l101_101827


namespace sum_of_divisors_37_l101_101314

theorem sum_of_divisors_37 : ∑ d in (finset.filter (λ d, 37 % d = 0) (finset.range 38)), d = 38 :=
by {
  sorry
}

end sum_of_divisors_37_l101_101314


namespace range_a_empty_intersection_range_a_sufficient_condition_l101_101933

noncomputable def A (x : ℝ) : Prop := -10 < x ∧ x < 2
noncomputable def B (x : ℝ) (a : ℝ) : Prop := x ≥ 1 + a ∨ x ≤ 1 - a
noncomputable def A_inter_B_empty (a : ℝ) : Prop := ∀ x : ℝ, A x → ¬ B x a
noncomputable def neg_p (x : ℝ) : Prop := x ≥ 2 ∨ x ≤ -10
noncomputable def neg_p_implies_q (a : ℝ) : Prop := ∀ x : ℝ, neg_p x → B x a

theorem range_a_empty_intersection : (∀ x : ℝ, A x → ¬ B x 11) → 11 ≤ a := by
  sorry

theorem range_a_sufficient_condition : (∀ x : ℝ, neg_p x → B x 1) → 0 < a ∧ a ≤ 1 := by
  sorry

end range_a_empty_intersection_range_a_sufficient_condition_l101_101933


namespace no_multiple_of_10_l101_101135

theorem no_multiple_of_10 (grid: Fin 8 → Fin 8 → ℕ)
  (operation : (Fin 8 → Fin 8) → (Fin (max 3 4) → Fin (max 3 4)) → (Fin 8 → Fin 8 → ℕ) → (Fin 8 → Fin 8 → ℕ))
  (h_op: ∀ grid, ∀ f: Fin (max 3 4) → Fin (max 3 4), ∀ g: Fin 8 → Fin 8, 
    operation g f grid = λ i j, grid i j + 1) :
  ¬ ∃ n: ℕ, ∀ i j: Fin 8, (operation g f grid i j) % 10 = 0 := 
begin
  sorry
end

end no_multiple_of_10_l101_101135


namespace original_number_l101_101814

theorem original_number (N y x : ℕ) 
  (h1: N + y = 54321)
  (h2: N = 10 * y + x)
  (h3: 11 * y + x = 54321)
  (h4: x = 54321 % 11)
  (hy: y = 4938) : 
  N = 49383 := 
  by 
  sorry

end original_number_l101_101814


namespace angle_equality_l101_101140

variables {A B C D E F : Type*} [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty E] [Nonempty F]
variables (angle : Type*) (∠ : A → B → C → angle)
variables (circumcircle : A → B → D → Type*) (lies_on : E → (circumcircle A B D) → Prop) (intersection_point : A → C → circumcircle = F)

theorem angle_equality
  (h1 : E ∈ BD) (h2 : ∠ E C D = ∠ A C B) 
  (h3 : lies_on F (circumcircle A B D))
  (h4 : ∠ A C B = ∠ D A F) :
  ∠ D F E = ∠ A F B :=
by
  sorry

end angle_equality_l101_101140


namespace trajectory_of_P_l101_101925

-- Define points M and N
def M : ℝ × ℝ := (-3, 0)
def N : ℝ × ℝ := (3, 0)

-- Define the condition |PM| - |PN| = 4
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) - Real.sqrt ((P.1 - N.1)^2 + (P.2 - N.2)^2) = 4

-- Define the target proposition: P lies on the right branch of a hyperbola with foci M and N
def right_branch_hyperbola (P : ℝ × ℝ) : Prop :=
  (sqrt (((M.1 + N.1) / 2 - P.1)^2 + P.2^2) > 3) ∧ (P.1 > (M.1 + N.1) / 2)

theorem trajectory_of_P (P : ℝ × ℝ) (h : satisfies_condition P) :
  right_branch_hyperbola P :=
sorry

end trajectory_of_P_l101_101925


namespace jenny_kenny_meet_time_l101_101154

-- Definitions of initial conditions
def jen_speed := 2 -- Jenny's speed in feet per second
def ken_speed := 4 -- Kenny's speed in feet per second
def path_distance := 300 -- Distance between paths in feet
def building_diameter := 150 -- Diameter of the building in feet
def obstruct_distance_start := 300 -- Initial obstruct distance in feet

-- The radius of the building from its diameter
def building_radius : ℝ := building_diameter / 2

-- Proof statement
theorem jenny_kenny_meet_time :
  (∃ (t : ℝ), 
    t = 48 ∧
    -- condition 5 revised for clarity in the theorem context
    (obstruct_distance_start = path_distance) ∧
    ∀𝑡 (t > 0), 
    (4 * t + 2 * t = 300 + 2 * 150) -- Line obstruction geometry adapted for proof clarity
  ) :=
sorry

end jenny_kenny_meet_time_l101_101154


namespace num_integer_solutions_l101_101965

theorem num_integer_solutions : 
  (∀ x : ℤ, (x^2 - x - 1)^(x + 2) = 1) → (∀ x : ℤ, List.mem x [2, -1, 0, -2]) → (List.length [2, -1, 0, -2] = 4) :=
by
  sorry

end num_integer_solutions_l101_101965


namespace exponentiation_division_l101_101847

variable (a b : ℝ)

theorem exponentiation_division (a b : ℝ) : ((2 * a) / b) ^ 4 = (16 * a ^ 4) / (b ^ 4) := by
  sorry

end exponentiation_division_l101_101847


namespace non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101445

def g (m : Nat) : Nat :=
  let f k := m / k
  f 5 + f 25 + f 125 + f 625 + f 3125 + f 15625 + f 78125 + f 390625 + f 1953125 + f 9765625 + f 48828125 + f 244140625 + f 1220703125 + f 6103515625 + f 30517578125 -- continue as needed

def is_factorial_tail (n : Nat) : Prop :=
  ∃ m : Nat, g m = n

def count_factorial_tails (N : Nat) : Nat :=
  (List.range N).filter (λ n, is_factorial_tail n).length

theorem non_factorial_tails_lemma : count_factorial_tails 2500 = 2000 := sorry

theorem count_non_factorial_tails (N : Nat) : Prop :=
  let total := N - 1
  total - count_factorial_tails N = 499

theorem solution : count_non_factorial_tails 2500 := sorry

end non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101445


namespace ellipse_properties_ellipse_intersection_theorem_l101_101942

noncomputable def ellipse_equation (a b : ℝ) : ℝ × ℝ → Prop :=
  λ p, (p.1^2/a^2 + p.2^2/b^2 = 1)

def focal_distance (a b c : ℝ) : Prop :=
  a^2 - b^2 = c^2

def passes_through_point (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  ellipse_equation a b p

def isonceles_triangle {x E B : ℝ × ℝ} : Prop :=
  E.2 = x.2 + B.2

theorem ellipse_properties :
  ∃ a b : ℝ, ∃ C : ℝ × ℝ → Prop,
  (a > b ∧ b > 0) ∧
  (focal_distance a b (2 * real.sqrt 3)) ∧
  (passes_through_point a b (2 * real.sqrt 3, 1)) ∧
  (C = ellipse_equation 4 2)
:=
begin
  sorry
end

theorem ellipse_intersection_theorem :
  let C : ℝ × ℝ → Prop := ellipse_equation 4 2,
      k : ℝ := real.sqrt 2 / 4,
      B : ℝ × ℝ := (0, -2),
      E := (real.sqrt 2 / 4, 1 - real.sqrt 2),
      F := (- real.sqrt 2 / 4, 1 + real.sqrt 2),
      circle := λ p : ℝ × ℝ, p.1^2 + p.2^2 = 0.5 in
  isonceles_triangle E F B →
  ∀ M₀ M : ℝ × ℝ, 
    (M₀.1 * E.1 + M₀.2 * E.2 = M.1 * E.1 + M.2 * E.2) →
    (M.2 - M₀.2 = k * (M.1 - M₀.1)) →
    (let d := abs 1/(M.1 * C.trace + M.1 * B.trace) Ctrace_circle := let dist := abs 2*B.trace ∧ (dist > real.sqrt 2/2) in intersect : 
  ∀ p; 
    (p = C.trace div circle → Csection := ∅):=
begin
  sorry
end

end ellipse_properties_ellipse_intersection_theorem_l101_101942


namespace point_in_second_quadrant_l101_101969

theorem point_in_second_quadrant (A B C : ℝ) (hA : 0 < A ∧ A < 90) (hB : 0 < B ∧ B < 90) (hC : 0 < C ∧ C < 90) (h_sum : A + B + C = 180) : 
  (-1 * (cos B - sin A) > 0 ∧ sin B - cos A > 0) :=
by
  sorry

end point_in_second_quadrant_l101_101969


namespace non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101441

def g (m : Nat) : Nat :=
  let f k := m / k
  f 5 + f 25 + f 125 + f 625 + f 3125 + f 15625 + f 78125 + f 390625 + f 1953125 + f 9765625 + f 48828125 + f 244140625 + f 1220703125 + f 6103515625 + f 30517578125 -- continue as needed

def is_factorial_tail (n : Nat) : Prop :=
  ∃ m : Nat, g m = n

def count_factorial_tails (N : Nat) : Nat :=
  (List.range N).filter (λ n, is_factorial_tail n).length

theorem non_factorial_tails_lemma : count_factorial_tails 2500 = 2000 := sorry

theorem count_non_factorial_tails (N : Nat) : Prop :=
  let total := N - 1
  total - count_factorial_tails N = 499

theorem solution : count_non_factorial_tails 2500 := sorry

end non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101441


namespace cos_sum_identity_l101_101930

-- Define conditions:
def argument_of (z : ℂ) := complex.arg z

-- Theorem statement:
theorem cos_sum_identity (z : ℂ) (h : z ≠ 0) :
  let α := complex.arg z in
  (complex.cos α + complex.cos (2 * α) + complex.cos (3 * α) = -1 / 2) :=
by
  sorry

end cos_sum_identity_l101_101930


namespace probability_DEQ_greater_than_DFQ_and_EFQ_l101_101375

namespace EquilateralTriangle

noncomputable def probability_in_quadrilateral (DEF : Type) [triangle DEF] : ℝ :=
  let centroid := centroid DEF
  let quadrilateral_area := area (quadrilateral FGHI)
  let total_area := area DEF
  quadrilateral_area / total_area

theorem probability_DEQ_greater_than_DFQ_and_EFQ (Q : Point DEF) :
  Q ∈ interior DEF →
  (area (triangle DEQ) > area (triangle DFQ)) → 
  (area (triangle DEQ) > area (triangle EFQ)) →
  probability_in_quadrilateral DEF = 1 / 3 := 
by sorry

end EquilateralTriangle

end probability_DEQ_greater_than_DFQ_and_EFQ_l101_101375


namespace international_postage_surcharge_l101_101859

theorem international_postage_surcharge 
  (n_letters : ℕ) 
  (std_postage_per_letter : ℚ) 
  (n_international : ℕ) 
  (total_cost : ℚ) 
  (cents_per_dollar : ℚ) 
  (std_total_cost : ℚ) 
  : 
  n_letters = 4 →
  std_postage_per_letter = 108 / 100 →
  n_international = 2 →
  total_cost = 460 / 100 →
  cents_per_dollar = 100 →
  std_total_cost = n_letters * std_postage_per_letter →
  (total_cost - std_total_cost) / n_international * cents_per_dollar = 14 := 
sorry

end international_postage_surcharge_l101_101859


namespace maximum_omega_l101_101075

theorem maximum_omega : 
  ∀ (ω : ℝ),
  (∀ x : ℝ, x ∈ Icc (-π / 12) (π / 4) → -π / 3 ≤ 4 * ω * x ∧ 4 * ω * x ≤ π / 2) →
  ω ≤ 1/2 :=
by sorry

end maximum_omega_l101_101075


namespace find_lambda_l101_101183

variables (a b : Type) [AddCommGroup a] [Module ℝ a]
variables (u v : a) (λ : ℝ)

-- Definition: vectors a and b are non-parallel
def non_parallel (u v : a) : Prop := ¬ (∃ k : ℝ, u = k • v)

-- Definition: vector addition under the given lambda
def parallel_condition (u v : a) (λ : ℝ) : Prop := 
  ∃ k : ℝ, (λ • u + v) = k • (u - 2 • v)

-- Theorem to prove
theorem find_lambda (h1 : non_parallel u v) (h2 : parallel_condition u v λ) : λ = -1 / 2 :=
sorry

end find_lambda_l101_101183


namespace find_radius_l101_101790
-- Definitions for the conditions
def diameter_hole : ℝ := 24
def depth_hole : ℝ := 8
def r : ℝ := diameter_hole / 2
def d (R : ℝ) : ℝ := R - depth_hole

-- The Pythagorean theorem applied to this scenario
lemma radius_of_ball (R : ℝ) : R^2 = r^2 + (d R)^2 := by
  unfold r d

-- Prove that the radius R is 13 based on the given conditions
theorem find_radius : ∃ (R : ℝ), R^2 = 12^2 + (R - 8)^2 ∧ R = 13 := by
  use 13
  split
  -- Prove that 13^2 = 12^2 + (13 - 8)^2
  calc 13^2 = 169 : by norm_num
       ... = 144 + 25 : by norm_num
  sorry

end find_radius_l101_101790


namespace solve_proof_problem_l101_101522

noncomputable def proof_problem (alpha : ℝ) :=
  𝚜𝚎𝚌𝚘𝚗𝚍𝚀𝚞𝚊𝚍𝚛𝚊𝚗𝚝 : ∃ k : ℤ, α ∈ set.Ioo (k * (2 * Real.pi) + Real.pi / 2) (k * (2 * Real.pi) + Real.pi) ∧
  (h : Real.sin (alpha + Real.pi / 6) = 1 / 3),
  Real.sin (2 * alpha + Real.pi / 3) = -4 * Real.sqrt 2 / 9

theorem solve_proof_problem (alpha : ℝ) (h1 : ∃ k : ℤ, alpha ∈ set.Ioo (k * (2 * Real.pi) + Real.pi / 2) (k * (2 * Real.pi) + Real.pi)) 
                             (h2 : Real.sin (alpha + Real.pi / 6) = 1 / 3) :
  Real.sin (2 * alpha + Real.pi / 3) = -4 * Real.sqrt 2 / 9 :=
sorry

end solve_proof_problem_l101_101522


namespace range_f_x_negative_l101_101062

-- We define the conditions: f is an even function, increasing on (-∞, 0), and f(2) = 0.
variables {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_neg_infinity_to_zero (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → x < 0 ∧ y < 0 → f x ≤ f y

def f_at_2_is_zero (f : ℝ → ℝ) : Prop :=
  f 2 = 0

-- The theorem to be proven.
theorem range_f_x_negative (hf_even : even_function f)
  (hf_incr : increasing_on_neg_infinity_to_zero f)
  (hf_at2 : f_at_2_is_zero f) :
  ∀ x, f x < 0 ↔ x < -2 ∨ x > 2 :=
by
  sorry

end range_f_x_negative_l101_101062


namespace problem1_problem2_l101_101039

def f (x a : ℝ) := |x - 1| + |x - a|

/-
  Problem 1:
  Prove that if a = 3, the solution set to the inequality f(x) ≥ 4 is 
  {x | x ≤ 0 ∨ x ≥ 4}.
-/
theorem problem1 (f : ℝ → ℝ → ℝ) (a : ℝ) (h : a = 3) : 
  {x : ℝ | f x a ≥ 4} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := 
sorry

/-
  Problem 2:
  Prove that for any x₁ ∈ ℝ, if f(x₁) ≥ 2 holds true, the range of values for
  a is {a | a ≥ 3 ∨ a ≤ -1}.
-/
theorem problem2 (f : ℝ → ℝ → ℝ) (x₁ : ℝ) :
  (∀ x₁ : ℝ, f x₁ a ≥ 2) ↔ (a ≥ 3 ∨ a ≤ -1) :=
sorry

end problem1_problem2_l101_101039


namespace range_of_c_l101_101555

def cubic_polynomial (a b c : ℝ) : (x : ℝ) → ℝ := λ x, x^3 + a * x^2 + b * x + c

theorem range_of_c {a b c : ℝ}
  (h1 : 0 < cubic_polynomial a b c (-1))
  (h2 : cubic_polynomial a b c (-1) = cubic_polynomial a b c (-2))
  (h3 : cubic_polynomial a b c (-2) = cubic_polynomial a b c (-3))
  (h4 : cubic_polynomial a b c (-1) ≤ 3) :
  6 < c ∧ c ≤ 9 := 
sorry

end range_of_c_l101_101555


namespace sum_first_9000_terms_l101_101249

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_9000_terms (a r : ℝ) :
  geometric_sequence_sum a r 3000 = 500 →
  geometric_sequence_sum a r 6000 = 950 →
  geometric_sequence_sum a r 9000 = 1355 :=
by
  intros h1 h2
  sorry

end sum_first_9000_terms_l101_101249


namespace length_of_train_l101_101011

-- Conditions
variable (L E T : ℝ)
axiom h1 : 300 * E = L + 300 * T
axiom h2 : 90 * E = L - 90 * T

-- The statement to be proved
theorem length_of_train : L = 200 * E :=
by
  sorry

end length_of_train_l101_101011


namespace right_triangle_length_QR_l101_101601

theorem right_triangle_length_QR (Q P R : ℝ) 
  (h1 : tan Q = 0.5) (h2 : ∃ Q' P' R', QP = (Q' - P') = 16) (h3 : right_triangle Q P R) :
  QR = 8 * sqrt 5 := by
  sorry

end right_triangle_length_QR_l101_101601


namespace move_chocolates_l101_101846

theorem move_chocolates {p q : ℕ} (hp_odd : p % 2 = 1) (hq_odd : q % 2 = 1) (h : p > q) :
  let chocolates_to_move := (p - q + 2) / 2
  (q + chocolates_to_move > p - chocolates_to_move) :=
by {
  let chocolates_to_move := (p - q + 2) / 2,
  have h1 : q + chocolates_to_move = q + (p - q + 2) / 2, from rfl,
  have h2 : p - chocolates_to_move = p - (p - q + 2) / 2, from rfl,
  have h3 : q + (p - q + 2) / 2 > p - (p - q + 2) / 2, sorry,
  exact h3
}

end move_chocolates_l101_101846


namespace find_equation_of_ellipse_and_line_l101_101516

variable (a b c m k : ℝ)
variable (A B : ℝ × ℝ)
variable (O : ℝ × ℝ := (0, 0))
variable (F1 F2 : ℝ × ℝ)

def ellipse_C (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def hyperbola_H (x y : ℝ) : Prop :=
  (x^2 / 2) - y^2 = 1

theorem find_equation_of_ellipse_and_line :
  (a > b) →
  (b > 0) →
  (c / a = Real.sqrt 2 / 2) →
  (dist (F1.1, F1.2) (line (1, Real.sqrt 2)) = Real.sqrt 3 / 3) →
  (∀ (x y : ℝ), (line (k, m) x y) → ellipse_C x y) →
  k < 0 →
  (dist O (line k m) = 2 * Real.sqrt 5 / 5) →
  (∃ a b : ℝ, a = Real.sqrt 2 ∧ b = 1 ∧ (ellipse_C x y) ↔ (x^2 / 2 + y^2) = 1) ∧
  (∃ k m : ℝ, k = -1/2 ∧ m = 1 ∧ (line k m x y) ↔ (y = -1/2 * x + 1)) :=
sorry

end find_equation_of_ellipse_and_line_l101_101516


namespace fraction_equivalence_l101_101504

theorem fraction_equivalence (a b : ℝ) (h : ((1 / a) + (1 / b)) / ((1 / a) - (1 / b)) = 2020) : (a + b) / (a - b) = 2020 :=
sorry

end fraction_equivalence_l101_101504


namespace evaluate_Q_at_2_l101_101729

-- Define the polynomial Q(x)
noncomputable def Q (x : ℚ) : ℚ := x^4 - 20 * x^2 + 16

-- Define the root condition of sqrt(3) + sqrt(7)
def root_condition (x : ℚ) : Prop := (x = ℚ.sqrt(3) + ℚ.sqrt(7))

-- Theorem statement
theorem evaluate_Q_at_2 : Q 2 = -32 :=
by
  -- Given conditions
  have h1 : degree Q = 4 := sorry,
  have h2 : leading_coeff Q = 1 := sorry,
  have h3 : root_condition (ℚ.sqrt(3) + ℚ.sqrt(7)) := sorry,
  -- Goal
  exact sorry

end evaluate_Q_at_2_l101_101729


namespace cos_of_odd_multiple_of_90_in_0_720_eq_neg6_l101_101104

theorem cos_of_odd_multiple_of_90_in_0_720_eq_neg6 : 
  ∀ x : ℝ, 
  (0 ≤ x) → (x < 720) → (∃ k : ℤ, x = (2 * k + 1) * 90) → cos (x * (π / 180)) = -0.6 → false :=
by
  intros x h0 h1 h2 hcos
  sorry

end cos_of_odd_multiple_of_90_in_0_720_eq_neg6_l101_101104


namespace find_original_number_l101_101804

-- Variables and assumptions
variables (N y x : ℕ)

-- Conditions
def crossing_out_condition : Prop := N = 10 * y + x
def sum_condition : Prop := N + y = 54321
def remainder_condition : Prop := x = 54321 % 11
def y_value : Prop := y = 4938

-- The final proof problem statement
theorem find_original_number (h1 : crossing_out_condition N y x) (h2 : sum_condition N y) 
  (h3 : remainder_condition x) (h4 : y_value y) : N = 49383 :=
sorry

end find_original_number_l101_101804


namespace find_side_length_S2_l101_101676

-- Define the variables and conditions
variables (r s : ℕ)
def is_solution (r s : ℕ) : Prop :=
  2 * r + s = 2160 ∧ 2 * r + 3 * s = 3450

-- Define the problem statement
theorem find_side_length_S2 (r s : ℕ) (h : is_solution r s) : s = 645 :=
sorry

end find_side_length_S2_l101_101676


namespace number_of_divisors_of_1200_l101_101637

theorem number_of_divisors_of_1200 : 
  (number of d : ℕ | d ∣ 1200 ∧ 1 ≤ d ∧ d ≤ 1200) = 30 :=
sorry

end number_of_divisors_of_1200_l101_101637


namespace calculate_sum_l101_101426

-- Define the right-angled triangle ABC
variables {A B C P Q E F : Point}
variable {c b : ℝ}

-- Definition of right angle and lengths of sides involved
def right_angle_triangle (A B C : Point) (c b : ℝ) :=
  angle A B C = π / 2 ∧ dist A B = c ∧ dist A C = b

-- Points on sides and angles
def specific_points (P Q : Point) (A B C : Point) :=
  P ∈ segment A C ∧ Q ∈ segment A B ∧ angle A P Q = angle A B C ∧ angle A Q P = angle A C B

-- Projections onto sides
def projections (E F P Q B C : Point) :=
  proj B onto (line B C) = E ∧ proj Q onto (line B C) = F

-- Altitude calculation from right-angled triangle properties
def altitude (A B C : Point) (c b : ℝ) : ℝ :=
  dist A B * dist A C / sqrt ((dist A B) ^ 2 + (dist A C) ^ 2)

-- The theorem to prove PQ + PE + QF = 4 * h_a
theorem calculate_sum {A B C P Q E F : Point} {c b : ℝ}
  (h1 : right_angle_triangle A B C c b)
  (h2 : specific_points P Q A B C)
  (h3 : projections E F P Q B C)
  : dist P Q + dist P E + dist Q F = 4 * altitude A B C c b :=
sorry -- the proof is to be completed

end calculate_sum_l101_101426


namespace right_column_sum_eq_ab_l101_101344

theorem right_column_sum_eq_ab (a b : ℤ) (h : a > 0) :
  let f : ℤ × ℤ → ℤ × ℤ := λ p, 
    if even p.1 then (p.1 / 2, p.2 * 2) else ((p.1 - 1) / 2, p.2 * 2)
    in 
    (∑ n in Nat.range (Nat.log2 (Int.toNat a) + 1), 
      if (Nat.bitTest n (Int.toNat a)) 
      then some_fun (f^[n] (a, b)).2 
      else 0 ) = a * b :=
sorry

end right_column_sum_eq_ab_l101_101344


namespace Q_at_2_l101_101722

-- Define the polynomial Q(x)
def Q (x : ℚ) : ℚ := x^4 - 8 * x^2 + 16

-- Define the properties of Q(x)
def is_special_polynomial (P : (ℚ → ℚ)) : Prop := 
  degree P = 4 ∧ leading_coeff P = 1 ∧ P.is_root(√3 + √7)

-- Problem: Prove Q(2) = 0 given the polynomial Q meets the conditions.
theorem Q_at_2 (Q : ℚ → ℚ) (h1 : degree Q = 4) (h2 : leading_coeff Q = 1) (h3 : is_root Q (√3 + √7)) : 
  Q 2 = 0 := by
  sorry

end Q_at_2_l101_101722


namespace f_extreme_values_b_range_if_a_greater_than_zero_and_consistent_on_minus2_infty_maximum_abs_diff_of_a_b_if_consistent_on_a_b_l101_101928

variables (a b : ℝ)

def f (x : ℝ) : ℝ := x^3 + a * x
def g (x : ℝ) : ℝ := x^2 + b * x
def f' (x : ℝ) : ℝ := 3 * x^2 + a
def g' (x : ℝ) : ℝ := 2 * x + b

-- 1.
theorem f_extreme_values:
  if a ≥ 0 then (∀ x, (f' x) ≥ 0) ∧ ∀ x, x = -2 * a * sqrt(-3 * a) / 9 ∧ x = 2 * a * sqrt(-3 * a) / 9 :=
sorry

-- 2.
theorem b_range_if_a_greater_than_zero_and_consistent_on_minus2_infty
  (h_a_pos : a > 0)
  (h_consistent : ∀ x ∈ Icc (-2 : ℝ) (⊤ : ℝ), f' x * g' x ≥ 0):
  b ≥ 4 :=
sorry

-- 3.
theorem maximum_abs_diff_of_a_b_if_consistent_on_a_b
  (h_a_neg : a < 0)
  (h_neq : a ≠ b)
  (h_consistent_on_open : ∀ x ∈ Ioo a b, f' x * g' x ≥ 0) :
  abs (a - b) ≤ 1 / 3 :=
sorry

end f_extreme_values_b_range_if_a_greater_than_zero_and_consistent_on_minus2_infty_maximum_abs_diff_of_a_b_if_consistent_on_a_b_l101_101928


namespace congruent_triangles_overlap_l101_101769

theorem congruent_triangles_overlap (T1 T2 : Triangle) :
  (∀ T3 : Triangle, (T3 ≅ T1) ∧ (T3 ≅ T2) → (T1 ≅ T2)) :=
by
  have T1_congruent_T2 : T1 ≅ T2 := sorry
  exact T1_congruent_T2

end congruent_triangles_overlap_l101_101769


namespace solve_for_n_l101_101765

theorem solve_for_n (n : ℚ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : 
    n = 3.5 :=
  sorry

end solve_for_n_l101_101765


namespace ratio_of_mustang_models_length_l101_101201

theorem ratio_of_mustang_models_length :
  ∀ (full_size_length mid_size_length smallest_model_length : ℕ),
    full_size_length = 240 →
    mid_size_length = full_size_length / 10 →
    smallest_model_length = 12 →
    smallest_model_length / mid_size_length = 1/2 :=
by
  intros full_size_length mid_size_length smallest_model_length h1 h2 h3
  sorry

end ratio_of_mustang_models_length_l101_101201


namespace area_of_quadrilateral_l101_101852

-- Define the vertices of the quadrilateral
def v1 : ℝ × ℝ := (2, 1)
def v2 : ℝ × ℝ := (0, 7)
def v3 : ℝ × ℝ := (5, 5)
def v4 : ℝ × ℝ := (6, 9)

-- Prove that the area of the quadrilateral is 9
theorem area_of_quadrilateral :
  let A := (2 * 7 + 0 * 5 + 5 * 9 + 6 * 1 - (1 * 0 + 7 * 5 + 5 * 6 + 9 * 2)) / 2 in
  abs A = 9 :=
by {
  let A := (2 * 7 + 0 * 5 + 5 * 9 + 6 * 1 - (1 * 0 + 7 * 5 + 5 * 6 + 9 * 2)) / 2,
  have : A = (14 + 0 + 45 + 6 - (0 + 35 + 30 + 18)) / 2, 
  { unfold A, simp },
  have : A = 18 / 2,
  { simp [this] },
  have : A = 9,
  { simp [this] },
  rw [this],
  exact abs_of_nonneg (show 9 ≥ 0, by norm_num),
}

end area_of_quadrilateral_l101_101852


namespace sandy_world_record_length_l101_101203

-- Define the conditions
def sandy_age_current := 12
def sandy_fingernail_length_current := 2.0 -- in inches
def fingernail_growth_rate := 0.1 -- inches per month
def sandy_age_goal := 32

-- Calculate the remaining time until Sandy achieves the record
def years_to_goal := sandy_age_goal - sandy_age_current
def months_to_goal := years_to_goal * 12
def total_growth := months_to_goal * fingernail_growth_rate

-- Define the length of fingernails when the record is achieved
def fingernail_length_goal := sandy_fingernail_length_current + total_growth

-- The theorem stating the goal
theorem sandy_world_record_length : fingernail_length_goal = 26 := by
  sorry

end sandy_world_record_length_l101_101203


namespace number_of_zeros_at_end_of_7_factorial_in_base_9_l101_101239

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem number_of_zeros_at_end_of_7_factorial_in_base_9 : 
  ∃ k : ℕ, (k = 1) ∧ (∃ m : ℕ, 7! = m * 9^k) :=
sorry

end number_of_zeros_at_end_of_7_factorial_in_base_9_l101_101239


namespace area_of_triangle_AEB_l101_101612

noncomputable theory

open_locale classical

variables {AB BC DF GC : ℝ}
variables (A B C D E F G : Type*)

-- Define the properties of points
variables [is_rectangular A B C D]
variables [line_through A F E]
variables [line_through B G E]

def AB_length : AB = 8 := by sorry
def BC_length : BC = 4 := by sorry
def DF_length : DF = 2 := by sorry
def GC_length : GC = 3 := by sorry

theorem area_of_triangle_AEB : area (triangle A E B) = 128/5 :=
by
  sorry

end area_of_triangle_AEB_l101_101612


namespace find_rational_r_l101_101870

theorem find_rational_r (r : ℚ) :
  (∀ x : ℚ, polynomial.aeval x[X] (r * X ^ 2 + (r + 2) * X + (r - 1)) = 0 → x ∈ set.univ → x ∈ ℤ) ↔ r = -1/3 ∨ r = 1 :=
begin
  sorry
end

end find_rational_r_l101_101870


namespace people_who_chose_soda_l101_101984

-- Define the total number of people surveyed and the central angle for "Soda"
def peopleSurveyed : Nat := 500
def sodaAngle : ℕ := 200

-- Define the calculation for the number of people who chose "Soda"
def fractionOfSoda : ℚ := sodaAngle / 360
def numberOfSodaPeople : ℕ := ((peopleSurveyed : ℚ) * fractionOfSoda).round.to_nat

-- The theorem states the number of people who chose "Soda" given the conditions
theorem people_who_chose_soda :
  numberOfSodaPeople = 278 :=
by
  -- Here we would include the actual proof steps, but we're using sorry to skip it.
  sorry

end people_who_chose_soda_l101_101984


namespace sum_of_divisors_37_l101_101316

theorem sum_of_divisors_37 : ∑ d in (finset.filter (λ d, 37 % d = 0) (finset.range 38)), d = 38 :=
by {
  sorry
}

end sum_of_divisors_37_l101_101316


namespace water_left_in_cooler_l101_101331

theorem water_left_in_cooler : 
  let gallons_in_cooler := 3 in
  let ounces_per_cup := 6 in
  let rows := 5 in
  let chairs_per_row := 10 in
  let ounces_per_gallon := 128 in
  let initial_ounces := gallons_in_cooler * ounces_per_gallon in
  let total_chairs := rows * chairs_per_row in
  let total_ounces_needed := total_chairs * ounces_per_cup in
  let remaining_ounces := initial_ounces - total_ounces_needed in
  remaining_ounces = 84 :=
by 
  -- introduce variables
  let gallons_in_cooler := 3
  let ounces_per_cup := 6
  let rows := 5
  let chairs_per_row := 10
  let ounces_per_gallon := 128
  let initial_ounces := gallons_in_cooler * ounces_per_gallon
  let total_chairs := rows * chairs_per_row
  let total_ounces_needed := total_chairs * ounces_per_cup
  let remaining_ounces := initial_ounces - total_ounces_needed
  -- prove the theorem
  sorry

end water_left_in_cooler_l101_101331


namespace lattice_points_count_in_region_l101_101367

def isLatticePoint (p : ℤ × ℤ) : Prop :=
  ∃ x y : ℤ, p = (x, y)

def isInsideRegion (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  y ≤ abs x ∧ y ≤ -x^2 + 8

theorem lattice_points_count_in_region :
  {p : ℤ × ℤ | isLatticePoint p ∧ isInsideRegion p}.card = 27 :=
by sorry

end lattice_points_count_in_region_l101_101367


namespace prism_volume_is_nine_l101_101222

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ :=
  (√3 / 4) * s^2

noncomputable def median_of_equilateral_triangle (s : ℝ) : ℝ :=
  (√3 / 2) * s

noncomputable def height_of_prism (slant_height centroid_to_vertex : ℝ) : ℝ :=
  sqrt ((slant_height^2) - (centroid_to_vertex^2))

noncomputable def volume_of_prism (base_area height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

theorem prism_volume_is_nine :
  let s := 6
  let slant_height := sqrt 15
  let base_area := equilateral_triangle_area s
  let median := median_of_equilateral_triangle s
  let centroid_to_vertex := (2 / 3) * median
  let height := height_of_prism slant_height centroid_to_vertex
  volume_of_prism base_area height = 9 :=
by 
  sorry

end prism_volume_is_nine_l101_101222


namespace f_monotonically_decreasing_range_of_a_tangent_intersection_l101_101556

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 + 2
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - 2 * a * x

-- Part (I)
theorem f_monotonically_decreasing (a : ℝ) (x : ℝ) :
  (a > 0 → 0 < x ∧ x < (2 / 3) * a → f' x a < 0) ∧
  (a = 0 → ¬∃ x, f' x a < 0) ∧
  (a < 0 → (2 / 3) * a < x ∧ x < 0 → f' x a < 0) :=
sorry

-- Part (II)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f' x a ≥ abs x - 3 / 4) → (-1 ≤ a ∧ a ≤ 1) :=
sorry

-- Part (III)
theorem tangent_intersection (a : ℝ) :
  (a = 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ ∃ t : ℝ, (t - x1^3 - 2 = 3 * x1^2 * (2 - x1)) ∧
  (t - x2^3 - 2 = 3 * x2^2 * (2 - x2)) ∧ 2 ≤ t ∧ t ≤ 10 ∧
  ∀ t', (t' - x1^3 - 2 = 3 * x1^2 * (2 - x1)) ∧
  (t' - x2^3 - 2 = 3 * x2^2 * (2 - x2)) → t' ≤ 10) :=
sorry

end f_monotonically_decreasing_range_of_a_tangent_intersection_l101_101556


namespace number_of_valid_integers_l101_101502

theorem number_of_valid_integers :
  {n : ℤ // n ≥ 2 ∧ n < 5}.card = 3 :=
by sorry

end number_of_valid_integers_l101_101502


namespace fixed_point_of_intersection_l101_101161

variable {α : Type}
variable [inner_product_space ℝ α]

theorem fixed_point_of_intersection
  {Γ : set (euclidean_space ℝ (fin 2))}
  (ABC : affine_subspace ℝ (euclidean_space ℝ (fin 2)))
  (P : euclidean_space ℝ (fin 2))
  (hP : P ∈ Γ ∧ P ≠ (ABC.direction.to_affine _ _).affine_span.points)
  (I J : euclidean_space ℝ (fin 2))
  (hI : is_incenter_of_triangle I ABC P)
  (hJ : is_incenter_of_triangle J ABC P)
  (Q : euclidean_space ℝ (fin 2))
  (hQ : Q ∈ (circumcircle_of_triangle PIJ ∩ Γ)) :
  ∀ P ∈ Γ, P ≠ A ∧ P ≠ B → Q = fixed_point :=
sorry

end fixed_point_of_intersection_l101_101161


namespace eccentricity_of_hyperbola_l101_101369

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def eccentricity (a b : ℝ) := 
  let c := real.sqrt (a^2 + b^2)
  c / a

theorem eccentricity_of_hyperbola (a b e : ℝ) 
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_hyperbola : hyperbola a b (real.sqrt (a^2 + b^2)) 0) :
  eccentricity a b = (real.sqrt 5 + 1) / 2 :=
sorry

end eccentricity_of_hyperbola_l101_101369


namespace least_number_to_add_l101_101341

theorem least_number_to_add (n d : ℕ) (r : ℕ) (h₀ : n % d = r) (h₁ : r < d) : ∃ x : ℕ, (n + x) % d = 0 ∧ x = d - r :=
by
  exists (d - r)
  split
  {
    rw [Nat.add_sub_assoc h₁, Nat.add_mod, h₀, Nat.mod_self, Nat.zero_mod]
    exact zero_le d
  }
  {
    refl
  }

def example_problem : ∃ x : ℕ, (1056 + x) % 25 = 0 ∧ x = 19 :=
  least_number_to_add 1056 25 6 (by norm_num) (by norm_num)

end least_number_to_add_l101_101341


namespace no_poly_deg3_satisfies_conditions_l101_101967

-- Define the polynomial function of degree 3
def poly_deg3 (a3 a2 a1 a0 : ℝ) (x : ℝ) : ℝ :=
  a3 * x^3 + a2 * x^2 + a1 * x + a0

-- Define the conditions for the polynomial
def condition1 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x^2) = f(f(x)))

def condition2 (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f'(x^2) = 2 * f(x) * f'(x)

-- Main theorem statement
theorem no_poly_deg3_satisfies_conditions :
  ∀ f : ℝ → ℝ, ∀ f' : ℝ → ℝ,
    (∃ a3 a2 a1 a0 : ℝ, f = poly_deg3 a3 a2 a1 a0) →
    condition1 f →
    condition2 f f' →
    false :=
begin
  intros f f',
  rintro ⟨a3, a2, a1, a0, hf⟩ h1 h2,
  sorry
end

end no_poly_deg3_satisfies_conditions_l101_101967


namespace hexagon_triangle_area_half_l101_101197

open Set

-- Define the problem conditions and the goal to prove
theorem hexagon_triangle_area_half (O : ℝ) (r : ℝ) (A1 A2 A3 B1 B2 B3 : ℝ) 
  (h1 : dist O A1 = dist O B1 ∧ dist O A1 = r ∧ dist O B1 = r)
  (h2 : dist O A2 = dist O B2 ∧ dist O A2 = r ∧ dist O B2 = r)
  (h3 : dist O A3 = dist O B3 ∧ dist O A3 = r ∧ dist O B3 = r)  
  (h4 : straight_line O A1 B1)
  (h5 : straight_line O A2 B2)
  (h6 : straight_line O A3 B3) :
  area (triangle O A1 A3) + area (triangle O A3 B2) + area (triangle O B2 A1) = (area (hexagon A1 A2 A3 B1 B2 B3)) / 2 := 
sorry -- Proof omitted for brevity

end hexagon_triangle_area_half_l101_101197


namespace find_angle_A_find_area_triangle_find_area_triangle_alt_range_b_l101_101628

-- Assumptions and conditions
variables {a b c A B C : ℝ}
variable (eqn1 : (2 * b - real.sqrt 3 * c) * real.cos A = real.sqrt 3 * a * real.cos C)
variables (cond1 : c = real.sqrt 3 * b) (cond2 : B = 2 * real.pi / 3) (cond3 : a = 2)

-- Part (I)
theorem find_angle_A : A = real.pi / 6 :=
sorry

-- Part (II)
theorem find_area_triangle (case1 : cond1) (case2 : cond3) : 
  let S := (b * c * real.sin A) / 2 in 
  S = real.sqrt 3 :=
sorry

theorem find_area_triangle_alt (case3 : cond2) (case4 : cond3) :
  let S := (2 * 2 * real.sin B / 2) in 
  S = real.sqrt 3 :=
sorry

-- Part (III)
theorem range_b (cond : a = 2) : 2 < b ∧ b < 4 :=
sorry

end find_angle_A_find_area_triangle_find_area_triangle_alt_range_b_l101_101628


namespace polynomial_coeff_sum_l101_101180

noncomputable def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem polynomial_coeff_sum (a b c d : ℝ) 
  (h3i : g a b c d (3 * Complex.i) = 0)
  (h1i : g a b c d (1 + Complex.i) = 0)
  (hre : ∀ x : ℂ, g a b c d x = (g a b c d x).conj) :
  a + b + c + d = 9 := 
by 
  sorry

end polynomial_coeff_sum_l101_101180


namespace w_squared_approx_91_44_l101_101107

theorem w_squared_approx_91_44 (w : ℝ) 
  (h : (w + 15)^2 = (4w + 9) * (3w + 6)) : 
  w^2 ≈ 91.44 := 
by
  sorry

end w_squared_approx_91_44_l101_101107


namespace coins_placement_possible_l101_101415

-- Definition of the board and coins
structure Board :=
  (cells : Fin 16 → ℕ)  -- Considering cells indexed from 0 to 15 for simplicity
  (is_silver : Fin 16 → Prop)  -- Predicate to indicate which cells have silver coins (the rest will have gold coins)
  (num_gold : ℕ)  -- Number of gold coins on the board
  (num_silver : ℕ)  -- Number of silver coins on the board

-- Predicate for silver greater than gold in a sub-square
def sub_square_condition (b : Board) (indices : List (Fin 16)) : Prop :=
  let gold_count := indices.countp (λ i, ¬ b.is_silver i) in
  let silver_count := indices.countp b.is_silver in
  silver_count > gold_count

-- Main theorem
theorem coins_placement_possible :
  ∃ (b : Board),
    -- Total gold greater than total silver
    b.num_gold > b.num_silver ∧
    -- All 3x3 sub-squares have more silver than gold
    (sub_square_condition b [0, 1, 2, 4, 5, 6, 8, 9, 10]) ∧
    (sub_square_condition b [1, 2, 3, 5, 6, 7, 9, 10, 11]) ∧
    (sub_square_condition b [4, 5, 6, 8, 9, 10, 12, 13, 14]) ∧
    (sub_square_condition b [5, 6, 7, 9, 10, 11, 13, 14, 15]) ∧
    (sub_square_condition b [0, 1, 2, 4, 5, 6, 3, 7, 11]) ∧
    (sub_square_condition b [0, 4, 8, 5, 9, 13, 1, 2, 6]) ∧
    (sub_square_condition b [4, 8, 12, 5, 9, 13, 6, 7, 10]) ∧
    (sub_square_condition b [8, 12, 13, 9, 14, 15, 4, 5, 9]) ∧
    (sub_square_condition b [1, 5, 6, 9, 10, 13, 2, 6, 7]) ∧
    -- Center 2x2 check
    (b.is_silver 5 ∧ b.is_silver 6 ∧ b.is_silver 9 ∧ b.is_silver 10) :=
sorry

end coins_placement_possible_l101_101415


namespace unique_poly_degree_4_l101_101739

theorem unique_poly_degree_4 
  (Q : ℚ[X]) 
  (h_deg : Q.degree = 4) 
  (h_lead_coeff : Q.leading_coeff = 1)
  (h_root : Q.eval (sqrt 3 + sqrt 7) = 0) : 
  Q = X^4 - 12 * X^2 + 16 ∧ Q.eval 2 = 0 :=
by sorry

end unique_poly_degree_4_l101_101739


namespace periodic_sequences_zero_at_two_l101_101919

variable {R : Type*} [AddGroup R]

def seq_a (a b : ℕ → R) (n : ℕ) : Prop := a (n + 1) = a n + b n
def seq_b (b c : ℕ → R) (n : ℕ) : Prop := b (n + 1) = b n + c n
def seq_c (c d : ℕ → R) (n : ℕ) : Prop := c (n + 1) = c n + d n
def seq_d (d a : ℕ → R) (n : ℕ) : Prop := d (n + 1) = d n + a n

theorem periodic_sequences_zero_at_two
  (a b c d : ℕ → R)
  (k m : ℕ)
  (hk : 1 ≤ k)
  (hm : 1 ≤ m)
  (ha : ∀ n, seq_a a b n)
  (hb : ∀ n, seq_b b c n)
  (hc : ∀ n, seq_c c d n)
  (hd : ∀ n, seq_d d a n)
  (kra : a (k + m) = a m)
  (krb : b (k + m) = b m)
  (krc : c (k + m) = c m)
  (krd : d (k + m) = d m) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 := sorry

end periodic_sequences_zero_at_two_l101_101919


namespace max_p_q_r_l101_101650

open Matrix

def B (p q r : ℤ) : Matrix (Fin 2) (Fin 2) ℚ :=
  (1 / 3 : ℚ) • ![![2, p], ![q, r]]

def B_squared_equals_2I (p q r : ℤ) : Prop :=
  B p q r ⬝ B p q r = (2 : ℚ) • 1

theorem max_p_q_r (p q r : ℤ)
  (hB : B_squared_equals_2I p q r) :
  p + q + r ≤ 13 :=
  sorry

end max_p_q_r_l101_101650


namespace original_five_digit_number_l101_101811

theorem original_five_digit_number :
  ∃ N y x : ℕ, (N = 10 * y + x) ∧ (N + y = 54321) ∧ (N = 49383) :=
by
  -- The proof script goes here
  sorry

end original_five_digit_number_l101_101811


namespace find_m_l101_101977

-- Definitions to detail the problem setup
def is_ellipse (a b : ℝ) : Prop := a > 0 ∧ b > 0

def ellipse_equation (x y a b : ℝ) : Prop :=
  x^2 / a + y^2 / b = 1

def eccentricity (a b e : ℝ) : Prop :=
  e = if a ≥ b then real.sqrt (1 - b^2 / a^2) else real.sqrt (1 - a^2 / b^2)

-- Main problem statement
theorem find_m (m : ℝ) :
  (∀ (x y : ℝ), is_ellipse 5 (5 + m) →
  ellipse_equation x y 5 (5 + m) →
  eccentricity 5 (5 + m) (1/2)) →
  (m = -5/4 ∨ m = 5/3) :=
by
  sorry

end find_m_l101_101977


namespace find_original_number_l101_101803

-- Variables and assumptions
variables (N y x : ℕ)

-- Conditions
def crossing_out_condition : Prop := N = 10 * y + x
def sum_condition : Prop := N + y = 54321
def remainder_condition : Prop := x = 54321 % 11
def y_value : Prop := y = 4938

-- The final proof problem statement
theorem find_original_number (h1 : crossing_out_condition N y x) (h2 : sum_condition N y) 
  (h3 : remainder_condition x) (h4 : y_value y) : N = 49383 :=
sorry

end find_original_number_l101_101803


namespace polynomial_coefficient_l101_101767

theorem polynomial_coefficient :
  ∀ d : ℝ, (2 * (2 : ℝ)^4 + 3 * (2 : ℝ)^3 + d * (2 : ℝ)^2 - 4 * (2 : ℝ) + 15 = 0) ↔ (d = -15.75) :=
by
  sorry

end polynomial_coefficient_l101_101767


namespace at_least_one_positive_l101_101055

variables {x y z : ℝ}

def a := x^2 - 2 * y + (Real.pi / 2)
def b := y^2 - 2 * z + (Real.pi / 3)
def c := z^2 - 2 * x + (Real.pi / 6)

theorem at_least_one_positive : a > 0 ∨ b > 0 ∨ c > 0 :=
sorry

end at_least_one_positive_l101_101055


namespace find_b_l101_101655

def vector (α : Type*) [Add α] [Mul α] := List α

noncomputable def is_collinear {α : Type*} [Field α] (a b c : vector α) : Prop :=
  ∃ t : α, b = a + t • (c - a)

noncomputable def is_angle_bisector
  {α : Type*} [Field α] [Fintype α] [EuclideanSpace α (EuclideanInnerSpace α)]
  (a b c : vector α) : Prop :=
  ∥a∥ * (b ⬝ c) / (∥b∥ * ∥c∥) = ∥c∥ * (a ⬝ b) / (∥b∥ * ∥a∥)

def a : vector ℝ := [8, -3, 1]
def c : vector ℝ := [2, 1, -3]
noncomputable def b : vector ℝ := [2, 1, -3]

theorem find_b :
  is_collinear a b c ∧ is_angle_bisector a b c :=
by
  sorry

end find_b_l101_101655


namespace quadrilateral_not_conclusively_square_l101_101599

/-- The diagonals of a quadrilateral are perpendicular. -/
def perpendicular_diagonals (P Q R S : Point) : Prop :=
  let (D1, D2) := (line_through P R, line_through Q S) in
  D1 ⊥ D2

/-- A quadrilateral can be inscribed in a circle. -/
def inscribable (P Q R S : Point) : Prop := 
  ∃ (O: Point) (r: ℝ), 
    circle O r P ∧ circle O r Q ∧ circle O r R ∧ circle O r S

/-- A quadrilateral can circumscribe a circle. -/
def circumscribable (P Q R S : Point) : Prop :=
  ∃ (I: Point), tangent_to I P ∧ tangent_to I Q ∧ tangent_to I R ∧ tangent_to I S

/-- A quadrilateral is a square. -/
def is_square (P Q R S : Point) : Prop :=
  dist P Q = dist Q R ∧ dist Q R = dist R S ∧ dist R S = dist S P ∧
  angle P Q R = π/2 ∧ angle Q R S = π/2 ∧ angle R S P = π/2 ∧ angle S P Q = π/2

/-- It cannot be conclusively stated that the quadrilateral is a square. -/
theorem quadrilateral_not_conclusively_square {P Q R S : Point} :
  perpendicular_diagonals P Q R S → inscribable P Q R S → circumscribable P Q R S → ¬ is_square P Q R S := 
sorry

end quadrilateral_not_conclusively_square_l101_101599


namespace average_max_two_selected_balls_l101_101131

theorem average_max_two_selected_balls : 
  let outcomes := {(1, 2), (1, 3), (2, 3)} in
  let max_values := outcomes.map (λ pair, max pair.fst pair.snd) in
  let probabilities := max_values.map (λ x, if x = 2 then 1/3 else if x = 3 then 2/3 else 0) in
  let expected_value := (max_values.zip probabilities).sum (λ (x, p), x * p) in
  expected_value = 8 / 3 :=
by sorry

end average_max_two_selected_balls_l101_101131


namespace min_value_f_on_interval_l101_101951

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x + a

theorem min_value_f_on_interval :
  ∃ (a : ℝ), (∀ x ∈ set.interval (-2 : ℝ) (2 : ℝ), f x a ≤ f 2 a) ∧
             (∀ x ∈ set.interval (-2 : ℝ) (2 : ℝ), f x a ≥ (f -1 a)) :=
begin
  use -2,
  split,
  { intro x,
    intro hx,
    -- place proof here
    sorry },
  { intro x,
    intro hx,
    -- place proof here
    sorry },
end

end min_value_f_on_interval_l101_101951


namespace EF_squared_eq_EP_squared_add_FN_squared_collinear_E_M_G_N_l101_101404

-- Definitions based on problem conditions
variables {A B C D E F G P M N : Type*}

-- Definitions based on geometric configuration
variable [InscribedQuadrilateral (Circle O) A B C D]
variable [Intersects (Line A B) (Line D C) E]
variable [Intersects (Line A D) (Line B C) F]
variable [Intersects (Diagonal A C) (Diagonal B D) G]
variable [Tangent (Line E P) (Circle O) P]
variable [TangentThroughPoint (Circle O) F M N]

-- First proof obligation
theorem EF_squared_eq_EP_squared_add_FN_squared (EF EP FN : ℝ)
  (h1 : squared_dist E F = squared_dist E P + squared_dist F N) : 
  EF^2 = EP^2 + FN^2 :=
by sorry

-- Second proof obligation
theorem collinear_E_M_G_N :
  (Collinear E M G N) :=
by sorry

end EF_squared_eq_EP_squared_add_FN_squared_collinear_E_M_G_N_l101_101404


namespace total_workers_is_22_l101_101218

-- Defining the average salaries and the related conditions
def average_salary_all : ℝ := 850
def average_salary_technicians : ℝ := 1000
def average_salary_rest : ℝ := 780

-- Given number of technicians
def T : ℕ := 7

-- Total number of workers in the workshop
def total_number_of_workers : ℕ :=
  let R := 15 in
  7 + R

-- Total number of workers proof
theorem total_workers_is_22 : total_number_of_workers = 22 :=
by
  -- Calculation to be filled in proof
  sorry

end total_workers_is_22_l101_101218


namespace sum_of_integers_equals_75_l101_101705

theorem sum_of_integers_equals_75 
  (n m : ℤ) 
  (h1 : n * (n + 1) * (n + 2) = 924) 
  (h2 : m * (m + 1) * (m + 2) * (m + 3) = 924) 
  (sum_seven_integers : ℤ := n + (n + 1) + (n + 2) + m + (m + 1) + (m + 2) + (m + 3)) :
  sum_seven_integers = 75 := 
  sorry

end sum_of_integers_equals_75_l101_101705


namespace sum_of_divisors_37_l101_101311

theorem sum_of_divisors_37 : ∑ d in (Finset.filter (fun d => d > 0 ∧ 37 % d = 0) (Finset.range (37 + 1))), d = 38 :=
by
  sorry

end sum_of_divisors_37_l101_101311


namespace range_of_f_l101_101497

def g (x : ℝ) : ℝ := -8 - 2 * cos (8 * x) - 4 * cos (4 * x)
def f (x : ℝ) : ℝ := sqrt (36 - g(x) ^ 2)

theorem range_of_f : ∀ x : ℝ, 0 ≤ f(x) ∧ f(x) ≤ sqrt 11 := 
by 
  sorry

end range_of_f_l101_101497


namespace simultaneous_messengers_l101_101694

theorem simultaneous_messengers (m n : ℕ) (h : m * n = 2010) : 
  m ≠ n → ((m, n) = (1, 2010) ∨ (m, n) = (2, 1005) ∨ (m, n) = (3, 670) ∨ 
          (m, n) = (5, 402) ∨ (m, n) = (6, 335) ∨ (m, n) = (10, 201) ∨ 
          (m, n) = (15, 134) ∨ (m, n) = (30, 67)) :=
sorry

end simultaneous_messengers_l101_101694


namespace smallest_N_l101_101830

-- Define the problem parameters and conditions
variable (l m n N : ℕ)

-- Conditions: N 1-cm cubes stacked; 143 cubes hidden when exposing three faces
def hidden_cubes_condition (l m n : ℕ) : Prop := (l-1) * (m-1) * (n-1) = 143

-- The goal to prove the smallest N value
theorem smallest_N (l m n N : ℕ) 
  (hl : l = 2) 
  (hm : m = 12) 
  (hn : n = 14)
  (hc : hidden_cubes_condition l m n) : N = 336 := 
by 
  dsimp [hidden_cubes_condition] at hc
  rw [hl, hm, hn] 
  have h_cube : (2-1) * (12-1) * (14-1) = 143,
  { norm_num },
  exact h_cube ▸ rfl

end smallest_N_l101_101830


namespace find_crease_length_l101_101853

-- Define the given conditions:
def rectangle_width : ℝ := 8 -- The width of the rectangle is 8 inches
variable (θ : ℝ) -- The angle formed at the corner before folding is θ

-- Define the crease length (L) as a function of θ
def crease_length (θ : ℝ) : ℝ :=
  rectangle_width * Real.cos θ

-- Define the theorem to be proven
theorem find_crease_length : crease_length θ = 8 * Real.cos θ :=
by sorry

end find_crease_length_l101_101853


namespace functional_eq_solve_l101_101868

theorem functional_eq_solve (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (2*x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := 
sorry

end functional_eq_solve_l101_101868


namespace find_r_given_conditions_l101_101117

theorem find_r_given_conditions (p c r : ℝ) (h1 : p * r = 360) (h2 : 6 * c * r = 15) (h3 : r = 4) : r = 4 :=
by
  sorry

end find_r_given_conditions_l101_101117


namespace sum_of_divisors_37_l101_101308

theorem sum_of_divisors_37 : ∑ d in (Finset.filter (fun d => d > 0 ∧ 37 % d = 0) (Finset.range (37 + 1))), d = 38 :=
by
  sorry

end sum_of_divisors_37_l101_101308


namespace storks_more_than_birds_l101_101785

-- Definitions based on given conditions
def initial_birds : ℕ := 3
def added_birds : ℕ := 2
def total_birds : ℕ := initial_birds + added_birds
def storks : ℕ := 6

-- Statement to prove the correct answer
theorem storks_more_than_birds : (storks - total_birds = 1) :=
by
  sorry

end storks_more_than_birds_l101_101785


namespace Q_evaluation_at_2_l101_101733

noncomputable def Q : Polynomial ℚ := 
  (Polynomial.X^2 + Polynomial.C 4)^2

theorem Q_evaluation_at_2 : 
  Q.eval 2 = 64 :=
by 
  -- We'll skip the proof as per the instructions.
  sorry

end Q_evaluation_at_2_l101_101733


namespace sin_div_one_minus_tan_eq_neg_three_fourths_l101_101526

variable (α : ℝ)

theorem sin_div_one_minus_tan_eq_neg_three_fourths
  (h : Real.sin (α - Real.pi / 4) = Real.sqrt 2 / 4) :
  (Real.sin α) / (1 - Real.tan α) = -3 / 4 := sorry

end sin_div_one_minus_tan_eq_neg_three_fourths_l101_101526


namespace necessary_but_not_sufficient_condition_l101_101346

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (log 2 x < 2 → x < 4) ∧ (∃ x, x < 4 ∧ ¬ (log 2 x < 2)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l101_101346


namespace find_number_l101_101784

noncomputable def number_solution (x : ℝ) : Prop :=
  1.3333 * x = 4.82

theorem find_number : ∃ x : ℝ, number_solution x ∧ abs (x - 3.615) < 0.001 :=
by
  use 4.82 / 1.3333
  rw [number_solution]
  split
  · norm_num
  · norm_num
  sorry

end find_number_l101_101784


namespace multiples_of_7_with_unit_digit_7_and_lt_150_l101_101103

/--
Given a positive integer n,
Let \(7n\) be a multiple of 7,
\(7n < 150\),
and \(7n \mod 10 = 7\),
prove there are 11 such multiples of 7.
-/
theorem multiples_of_7_with_unit_digit_7_and_lt_150 :
  ∃! (n : ℕ), n ≤ 11 ∧ ∀ (m : ℕ), (1 ≤ m ∧ m ≤ n) → (7 * (2 * m - 1) < 150 ∧ (7 * (2 * m - 1)) % 10 = 7) := by
sorry

end multiples_of_7_with_unit_digit_7_and_lt_150_l101_101103


namespace average_first_19_natural_numbers_l101_101890

theorem average_first_19_natural_numbers : (Finset.range 20).sum / 19 = 10 :=
by
  -- Given that the sum of the first n natural numbers is n(n + 1)/2
  have : Finset.range 20 = (0..19).to_finset, from rfl,
  rw [Finset.sum_range, this, Finset.sum_eq_sum_range, (0..19).sum_eq, Finset.sum_range_succ_comm],
  -- Prove that (0 + 1 + 2 + 3 + ... + 18 + 19) / 19 = 10
  have : ∑ i in (Finset.filter (λ n, n < 20) (0..19).to_finset), i = (19 * (19 + 1)) / 2 :=
    by
      rw [Finset.sum_range_succ_comm, Finset.range_succ_eq_map, Finset.sum_range, Finset.card_range],
      -- Sum of first 19 natural numbers: n(n + 1)/2
      sorry,
  rw [Finset.sum_filter],
  have : (19 * 20) / 2 / 19 = 10,
    by
      rw [Nat.mul_div_cancel_left, Nat.div_div_eq_div_mul],
      -- Simplify the division: (19 * 20) / 2 / 19 = 10
      sorry,
  rw [this],
  exact this

end average_first_19_natural_numbers_l101_101890


namespace smallest_abundant_not_multiple_of_five_l101_101008

def is_proper_divisor (d n : ℕ) : Prop := d < n ∧ n % d = 0

def sum_proper_divisors (n : ℕ) : ℕ :=
  Finset.sum (Finset.filter (λ d => is_proper_divisor d n) (Finset.range n)) (λ d => d)

def is_abundant (n : ℕ) : Prop := sum_proper_divisors n > n

def is_not_multiple_of_five (n : ℕ) : Prop := n % 5 ≠ 0

theorem smallest_abundant_not_multiple_of_five : ∃ n : ℕ, is_abundant n ∧ is_not_multiple_of_five n ∧ n = 12 := by
  sorry

end smallest_abundant_not_multiple_of_five_l101_101008


namespace second_crew_tractors_l101_101884

theorem second_crew_tractors
    (total_acres : ℕ)
    (days : ℕ)
    (first_crew_days : ℕ)
    (first_crew_tractors : ℕ)
    (acres_per_tractor_per_day : ℕ)
    (remaining_days : ℕ)
    (remaining_acres_after_first_crew : ℕ)
    (second_crew_acres_per_tractor : ℕ) :
    total_acres = 1700 → days = 5 → first_crew_days = 2 → first_crew_tractors = 2 → 
    acres_per_tractor_per_day = 68 → remaining_days = 3 → 
    remaining_acres_after_first_crew = total_acres - (first_crew_tractors * acres_per_tractor_per_day * first_crew_days) → 
    second_crew_acres_per_tractor = acres_per_tractor_per_day * remaining_days → 
    (remaining_acres_after_first_crew / second_crew_acres_per_tractor = 7) := 
by
  sorry

end second_crew_tractors_l101_101884


namespace sum_divisors_of_37_is_38_l101_101319

theorem sum_divisors_of_37_is_38 (h : Nat.Prime 37) : (∑ d in (Finset.filter (λ d, 37 % d = 0) (Finset.range (37 + 1))), d) = 38 := by
  sorry

end sum_divisors_of_37_is_38_l101_101319


namespace sqrt_4_times_9_sqrt_49_div_36_cuberoot_a6_sqrt_9_a2_l101_101490

-- Problem (a)
theorem sqrt_4_times_9 : Real.sqrt (4 * 9) = 6 :=
by sorry

-- Problem (b)
theorem sqrt_49_div_36 : Real.sqrt (49 / 36) = 7 / 6 :=
by sorry

-- Problem (c)
theorem cuberoot_a6 (a : Real) : Real.cbrt (a ^ 6) = a ^ 2 :=
by sorry

-- Problem (d)
theorem sqrt_9_a2 (a : Real) : Real.sqrt (9 * a ^ 2) = 3 * a :=
by sorry

end sqrt_4_times_9_sqrt_49_div_36_cuberoot_a6_sqrt_9_a2_l101_101490


namespace range_of_a_l101_101537

noncomputable def f (a x : ℝ) : ℝ := 2 * x^2 - a * x + 5

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y) → a ≤ 4 :=
by
  sorry

end range_of_a_l101_101537


namespace num_true_props_l101_101549

namespace ProofProblem

-- Define the sets M and N
def M (x : ℝ) := 0 < x ∧ x ≤ 3
def N (x : ℝ) := 0 < x ∧ x ≤ 2

-- Proposition definitions
def prop1 := ∀ a, (M a → N a) ∧ ¬(N a → M a)
def prop2 := ∀ a b, (M a → ¬M b) ↔ (M b → ¬M a)
def prop3 := ∀ p q : Prop, ¬(p ∧ q) → (¬p ∧ ¬q)
def prop4 := ¬∃ x : ℝ, x^2 - x - 1 > 0 ↔ ∀ x : ℝ, x^2 - x - 1 ≤ 0

-- Main theorem claiming the number of true propositions
theorem num_true_props : (∃ a, prop1) + (∃ a b, prop2) + (∃ p q : Prop, prop3) + prop4 = 2 := sorry

end ProofProblem

end num_true_props_l101_101549


namespace scientific_notation_l101_101982

theorem scientific_notation : (37_000_000 : ℝ) = 3.7 * 10^7 := 
sorry

end scientific_notation_l101_101982


namespace min_value_expression_l101_101482

theorem min_value_expression 
  (a b c : ℝ)
  (h1 : a + b + c = -1)
  (h2 : a * b * c ≤ -3) : 
  (ab + 1) / (a + b) + (bc + 1) / (b + c) + (ca + 1) / (c + a) ≥ 3 :=
sorry

end min_value_expression_l101_101482


namespace line_segments_equivalent_circles_equivalent_l101_101749

-- Define what it means for two subsets to be equivalent using a bijection
structure EquivalentSets (A B : Set (ℝ × ℝ)) :=
  (exists_bijection : ∃ f : (ℝ × ℝ) → (ℝ × ℝ), ∀ a ∈ A, f a ∈ B ∧ bijective f)

-- Problem (i): Any two line segments in the plane are equivalent
theorem line_segments_equivalent (A B C D : ℝ × ℝ) (AB_segment : λ t : ℝ => (1 - t) • A + t • B)
                            (CD_segment : λ t : ℝ => (1 - t) • C + t • D) :
  EquivalentSets {p | ∃ t, t ∈ Icc (0:ℝ) 1 ∧ p = AB_segment t} {q | ∃ t, t ∈ Icc (0:ℝ) 1 ∧ q = CD_segment t} :=
begin
  sorry
end

-- Problem (ii): Any two circles in the plane are equivalent
theorem circles_equivalent (x_A y_A r_1 x_B y_B r_2 : ℝ) :
  EquivalentSets {p | ∃ θ, θ ∈ Icc (0:ℝ) (2 * real.pi) ∧ p = (x_A + r_1 * real.cos θ, y_A + r_1 * real.sin θ)}
                 {q | ∃ θ, θ ∈ Icc (0:ℝ) (2 * real.pi) ∧ q = (x_B + r_2 * real.cos θ, y_B + r_2 * real.sin θ)} :=
begin
  sorry
end

end line_segments_equivalent_circles_equivalent_l101_101749


namespace ratio_of_share_l101_101791

/-- A certain amount of money is divided amongst a, b, and c. 
The share of a is $122, and the total amount of money is $366. 
Prove that the ratio of a's share to the combined share of b and c is 1 / 2. -/
theorem ratio_of_share (a b c : ℝ) (total share_a : ℝ) (h1 : a + b + c = total) 
  (h2 : total = 366) (h3 : share_a = 122) : share_a / (total - share_a) = 1 / 2 := by
  sorry

end ratio_of_share_l101_101791


namespace sum_of_divisors_37_l101_101309

theorem sum_of_divisors_37 : ∑ d in (Finset.filter (fun d => d > 0 ∧ 37 % d = 0) (Finset.range (37 + 1))), d = 38 :=
by
  sorry

end sum_of_divisors_37_l101_101309


namespace vince_savings_l101_101297

-- Definitions based on conditions
def earnings_per_customer : ℕ := 18
def monthly_expenses : ℤ := 280
def recreation_percentage : ℚ := 0.20
def customers_served : ℕ := 80

-- Intermediate calculations (explained in the solution steps but not used directly here)
def total_earnings : ℕ := customers_served * earnings_per_customer
def recreation_expenses : ℤ := (recreation_percentage.toReal * total_earnings).to_nat
def total_monthly_expenses : ℤ := monthly_expenses + recreation_expenses
def savings : ℤ := total_earnings - total_monthly_expenses

-- The proof statement we need to verify
theorem vince_savings :
  savings = 872 :=
sorry

end vince_savings_l101_101297


namespace non_factorial_tails_lt_2500_l101_101436

-- Define the function f(m)
def f (m: ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625

-- Define what it means to be a factorial tail
def is_factorial_tail (n: ℕ) : Prop :=
  ∃ m : ℕ, f(m) = n

-- Define the statement to be proved
theorem non_factorial_tails_lt_2500 : 
  (finset.range 2500).filter (λ n, ¬ is_factorial_tail n).card = 499 :=
by
  sorry

end non_factorial_tails_lt_2500_l101_101436


namespace actual_value_wrongly_copied_l101_101233

theorem actual_value_wrongly_copied :
  ∀ (n : ℕ) (mean_initial mean_correct : ℚ) (wrong_value correct_value : ℚ),
  n = 20 →
  mean_initial = 150 →
  mean_correct = 151.25 →
  wrong_value = 135 →
  (mean_correct * n - mean_initial * n) + wrong_value = correct_value →
  correct_value = 160 :=
begin
  intros n mean_initial mean_correct wrong_value correct_value,
  assume h_n h_mean_initial h_mean_correct h_wrong_value h_correct_value,
  sorry
end

end actual_value_wrongly_copied_l101_101233


namespace snack_eaters_left_after_second_newcomers_l101_101366

theorem snack_eaters_left_after_second_newcomers
  (initial_snackers : ℕ)
  (new_outsiders_1 : ℕ)
  (half_left_1 : ℕ)
  (new_outsiders_2 : ℕ)
  (final_snackers : ℕ)
  (H1 : initial_snackers = 100)
  (H2 : new_outsiders_1 = 20)
  (H3 : half_left_1 = (initial_snackers + new_outsiders_1) / 2)
  (H4 : new_outsiders_2 = 10)
  (H5 : final_snackers = 20)
  : (initial_snackers + new_outsiders_1 - half_left_1 + new_outsiders_2 - (initial_snackers + new_outsiders_1 - half_left_1 + new_outsiders_2 - final_snackers * 2)) = 30 :=
by 
  sorry

end snack_eaters_left_after_second_newcomers_l101_101366


namespace range_and_period_range_of_m_l101_101554

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos (x + Real.pi / 3) * (Real.sin (x + Real.pi / 3) - Real.sqrt 3 * Real.cos (x + Real.pi / 3))

theorem range_and_period (x : ℝ) :
  (Set.range f = Set.Icc (-2 - Real.sqrt 3) (2 - Real.sqrt 3)) ∧ (∀ x, f (x + Real.pi) = f x) := sorry

theorem range_of_m (x m : ℝ) (h1 : 0 ≤ x ∧ x ≤ Real.pi / 6) (h2 : m * (f x + Real.sqrt 3) + 2 = 0) :
  m ∈ Set.Icc (- 2 * Real.sqrt 3 / 3) (-1) := sorry

end range_and_period_range_of_m_l101_101554


namespace triangle_circumscribed_around_coins_l101_101751

noncomputable def circumscribed_triangle_area (r1 r2 r3 : ℝ) : ℝ :=
  r1*r1*real.cot(π/2) + r2*r2*real.cot(π/2) + r3*r3*real.cot(π/2)
  + (r1 + r2) * real.sqrt (r1 * r2)
  + (r2 + r3) * real.sqrt (r2 * r3)
  + (r3 + r1) * real.sqrt (r1 * r3)
  + real.sqrt ((r1 + r2 + r3) * r1 * r2 * r3)

theorem triangle_circumscribed_around_coins :
  circumscribed_triangle_area 21 19 23 = 1467.5 :=
sorry

end triangle_circumscribed_around_coins_l101_101751


namespace problem_statement_l101_101914

open Nat

variable (a : ℕ → ℝ) -- The arithmetic sequence {a_n}
variable (S : ℕ → ℝ) -- The sum of the first n terms of the sequence {a_n}
variable (n : ℕ)     -- A natural number n

-- Conditions
axiom a2_a5_sum : a 2 + a 5 = 12
axiom S5_value : S 5 = 25

-- Definition of the sum of the first n terms of an arithmetic sequence
def arithmetic_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  n * a 1 + (n * (n - 1) * (a 2 - a 1)) / 2

-- To define the general term for the arithmetic sequence
noncomputable def a_n := 2 * n - 1

-- To define the sequence b_n
noncomputable def b_n : ℕ → ℝ := λ n, 1 / (2 * n - 1) - 1 / (2 * n + 1)

-- To define the sum of the first n terms of b_n
noncomputable def T_n (n : ℕ) := 1 - 1 / (2 * n + 1)

-- The Lean statement to prove the equivalent math problem
theorem problem_statement (a : ℕ → ℝ) (S : ℕ → ℝ) (b_n T_n : ℕ → ℝ) :
  (a 2 + a 5 = 12) →
  (S 5 = 25) →
  (∀ n, a n = 2 * n - 1) →
  (∀ n, b_n n = 1 / (2 * n - 1) - 1 / (2 * n + 1)) →
  (∀ n, T_n n = 1 - 1 / (2 * n + 1)) :=
sorry

end problem_statement_l101_101914


namespace fibonacci_polynomial_property_l101_101646

noncomputable def fibonacci (n : ℕ) : ℕ :=
if n = 1 ∨ n = 2 then 1 else fibonacci (n - 1) + fibonacci (n - 2)

theorem fibonacci_polynomial_property
  (P : ℕ → ℕ)
  (hP_deg : ∀ x, P x = a_x)
  (hP_values : ∀ k, 992 ≤ k ∧ k ≤ 1982 → P k = fibonacci k) :
  P 1983 = fibonacci 1983 - 1 :=
sorry

end fibonacci_polynomial_property_l101_101646


namespace math_proof_problem_l101_101053

noncomputable def problem_1 : Prop :=
  let circle_eq := (x + 1)^2 + y^2 = 8
  let ellipse_eq := x^2 / 2 + y^2 = 1
  ∀ (C : ℝ × ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) (A : ℝ × ℝ) (M : ℝ × ℝ),
    C = (-1, 0) ∧
    ((P.1 + 1)^2 + P.2^2 = 8) ∧
    (∀ t : ℝ, Q = (1 - t) * C + t * P) ∧
    A = (1, 0) ∧
    (∃ t : ℝ, M = (1 - t) * A + t * P ∧ 2 * (M - Q) = 0) ∧
    ((M - Q) • (Q - A) = 0) →
    ellipse_eq

noncomputable def problem_2 : Prop :=
  let line := λ k b x, y = k * x + b
  let circle_tangent_eq := x^2 + y^2 = 1
  let ellipse_eq := x^2 / 2 + y^2 = 1
  ∀ (k : ℝ) (b : ℝ) (F H : ℝ × ℝ),
    (line k b F.1 = F.2) ∧ (line k b H.1 = H.2) ∧
    circle_tangent_eq F.1 F.2 ∧ circle_tangent_eq H.1 H.2 ∧
    ellipse_eq F.1 F.2 ∧ ellipse_eq H.1 H.2 ∧
    (3 / 4 ≤ (F.1 * H.1 + F.2 * H.2) ∧ (F.1 * H.1 + F.2 * H.2) ≤ 4 / 5) →
    (-sqrt(2) / 2 ≤ k ∧ k ≤ -sqrt(3) / 3) ∨ (sqrt(3) / 3 ≤ k ∧ k ≤ sqrt(2) / 2)

theorem math_proof_problem : problem_1 ∧ problem_2 :=
by
  constructor
  · sorry
  · sorry

end math_proof_problem_l101_101053


namespace combine_monomials_x_plus_y_l101_101565

theorem combine_monomials_x_plus_y : ∀ (x y : ℤ),
  7 * x = 2 - 4 * y →
  y + 7 = 2 * x →
  x + y = -1 :=
by
  intros x y h1 h2
  sorry

end combine_monomials_x_plus_y_l101_101565


namespace deck_cost_correct_l101_101284

def rare_cards := 19
def uncommon_cards := 11
def common_cards := 30
def rare_card_cost := 1
def uncommon_card_cost := 0.5
def common_card_cost := 0.25

def total_deck_cost := rare_cards * rare_card_cost + uncommon_cards * uncommon_card_cost + common_cards * common_card_cost

theorem deck_cost_correct : total_deck_cost = 32 :=
  by
  -- The proof would go here.
  sorry

end deck_cost_correct_l101_101284


namespace sum_of_first_9000_terms_of_geometric_sequence_l101_101262

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_9000_terms_of_geometric_sequence 
  (a r : ℝ)
  (h₁ : geometric_sum a r 3000 = 500)
  (h₂ : geometric_sum a r 6000 = 950)
  : geometric_sum a r 9000 = 1355 :=
sorry

end sum_of_first_9000_terms_of_geometric_sequence_l101_101262


namespace geometric_sequence_sum_l101_101265

noncomputable theory

-- Define the conditions and the problem statement
theorem geometric_sequence_sum (a r : ℝ) (h₁ : a * (1 - r^3000) / (1 - r) = 500) 
  (h₂ : a * (1 - r^6000) / (1 - r) = 950) : 
  a * (1 - r^9000) / (1 - r) = 1355 := 
sorry

end geometric_sequence_sum_l101_101265


namespace multiples_of_7_with_unit_digit_7_and_lt_150_l101_101100

/--
Given a positive integer n,
Let \(7n\) be a multiple of 7,
\(7n < 150\),
and \(7n \mod 10 = 7\),
prove there are 11 such multiples of 7.
-/
theorem multiples_of_7_with_unit_digit_7_and_lt_150 :
  ∃! (n : ℕ), n ≤ 11 ∧ ∀ (m : ℕ), (1 ≤ m ∧ m ≤ n) → (7 * (2 * m - 1) < 150 ∧ (7 * (2 * m - 1)) % 10 = 7) := by
sorry

end multiples_of_7_with_unit_digit_7_and_lt_150_l101_101100


namespace zoo_total_animals_l101_101394

def num_tiger_enclosures : ℕ := 4
def num_zebra_enclosures_per_tiger : ℕ := 2
def num_giraffe_enclosures_per_zebra : ℕ := 3
def num_tigers_per_enclosure : ℕ := 4
def num_zebras_per_enclosure : ℕ := 10
def num_giraffes_per_enclosure : ℕ := 2

theorem zoo_total_animals :
  let num_zebra_enclosures := num_tiger_enclosures * num_zebra_enclosures_per_tiger,
      num_giraffe_enclosures := num_zebra_enclosures * num_giraffe_enclosures_per_zebra,
      total_tigers := num_tiger_enclosures * num_tigers_per_enclosure,
      total_zebras := num_zebra_enclosures * num_zebras_per_enclosure,
      total_giraffes := num_giraffe_enclosures * num_giraffes_per_enclosure,
      total_animals := total_tigers + total_zebras + total_giraffes
  in total_animals = 144 :=
by
  intros
  let num_zebra_enclosures := num_tiger_enclosures * num_zebra_enclosures_per_tiger
  let num_giraffe_enclosures := num_zebra_enclosures * num_giraffe_enclosures_per_zebra
  let total_tigers := num_tiger_enclosures * num_tigers_per_enclosure
  let total_zebras := num_zebra_enclosures * num_zebras_per_enclosure
  let total_giraffes := num_giraffe_enclosures * num_giraffes_per_enclosure
  let total_animals := total_tigers + total_zebras + total_giraffes
  exact eq.refl 144

end zoo_total_animals_l101_101394


namespace sum_f_inv_and_f_eq_4045_div_2_l101_101500

noncomputable def f (x : ℝ) : ℝ := x / (1 + x)

theorem sum_f_inv_and_f_eq_4045_div_2 :
  (∑ k in finRange 2023 + 1, f (1 / k) + f k) = 4045 / 2 :=
by
  sorry

end sum_f_inv_and_f_eq_4045_div_2_l101_101500


namespace sum_of_cubes_consecutive_divisible_by_9_l101_101204

theorem sum_of_cubes_consecutive_divisible_by_9 (n : ℤ) : 9 ∣ (n-1)^3 + n^3 + (n+1)^3 :=
  sorry

end sum_of_cubes_consecutive_divisible_by_9_l101_101204


namespace finite_y_for_divisibility_l101_101679

theorem finite_y_for_divisibility :
  ∀ x : ℕ, x > 0 → (∃ y : ℕ, y > 0 ∧ (x^3 + y^3 + 1) % (x + y + 1) = 0) ∧ 
           (∀ x' : ℕ, x' > 0 → ¬ (∃ f : ℕ → ℕ, injective f ∧ ∀ n, (f n) > 0 ∧ (x'^3 + (f n)^3 + 1) % (x' + (f n) + 1) = 0)) := 
begin
  sorry
end

end finite_y_for_divisibility_l101_101679


namespace roots_are_positive_integers_but_not_squares_l101_101532

variable {m n : ℕ}

def roots_eq (x : ℤ) : Prop :=
  x^2 - (m^2 - m + 1)*(x - n^2 - 1) - (n^2 + 1)^2 = 0

def is_not_square (x : ℤ) : Prop :=
  ∀ k : ℤ, x ≠ k^2

theorem roots_are_positive_integers_but_not_squares
  (hmn_pos : m > n)
  (hmn_even : (m + n) % 2 = 0)
  (hm_pos : m > 0)
  (hn_pos : n > 0) :
  ∃ (x y : ℤ),
    roots_eq x ∧ roots_eq y ∧
    0 < x ∧ 0 < y ∧ x ≠ y ∧
    is_not_square x ∧ is_not_square y :=
sorry

end roots_are_positive_integers_but_not_squares_l101_101532


namespace max_volume_for_open_top_box_l101_101874

noncomputable def volume (a x : ℝ) : ℝ := x * (a - 2 * x)^2

theorem max_volume_for_open_top_box (a : ℝ) (ha : 0 < a) :
  ∃ x : ℝ, x = a / 6 ∧ 0 < x ∧ x < a / 2 ∧ volume a x = (2 * a^3 / 27) :=
begin
  -- Prove the statement here
  sorry
end

end max_volume_for_open_top_box_l101_101874


namespace find_a15_l101_101147

namespace ArithmeticSequence

variable {a : ℕ → ℝ}

def arithmetic_sequence (an : ℕ → ℝ) := ∃ (a₁ d : ℝ), ∀ n, an n = a₁ + (n - 1) * d

theorem find_a15
  (h_seq : arithmetic_sequence a)
  (h_a3 : a 3 = 4)
  (h_a9 : a 9 = 10) :
  a 15 = 16 := 
sorry

end ArithmeticSequence

end find_a15_l101_101147


namespace committee_count_l101_101365

theorem committee_count (students : Finset ℕ) (Alice : ℕ) (hAlice : Alice ∈ students) (hCard : students.card = 7) :
  ∃ committees : Finset (Finset ℕ), (∀ c ∈ committees, Alice ∈ c ∧ c.card = 4) ∧ committees.card = 20 :=
sorry

end committee_count_l101_101365


namespace algebraic_expression_value_l101_101971

theorem algebraic_expression_value (m n : ℤ) (h : n - m = 2):
  (m^2 - n^2) / m * (2 * m / (m + n)) = -4 :=
sorry

end algebraic_expression_value_l101_101971


namespace max_cells_cut_by_line_in_chessboard_l101_101303

theorem max_cells_cut_by_line_in_chessboard :
  ∀ (n : ℕ), n = 8 → (∃ L : ℝ → ℝ, (∃ max_cells : ℕ, max_cells = 15 ∧ 
  ∀ θ : ℝ, max_cells <= (count_cells_intersected_by_line n L θ))) :=
sorry

end max_cells_cut_by_line_in_chessboard_l101_101303


namespace sphere_volume_ratio_l101_101585

theorem sphere_volume_ratio
  (r R : ℝ)
  (h : (4:ℝ) * π * r^2 / (4 * π * R^2) = (4:ℝ) / 9) : 
  (r^3 / R^3 = (8:ℝ) / 27) := by
  sorry

end sphere_volume_ratio_l101_101585


namespace sheets_of_paper_used_l101_101668

theorem sheets_of_paper_used :
  let 
    num_classes := 6,
    num_first_classes := 3,
    num_last_classes := 3,
    students_first_class := 22,
    students_last_class := 18,
    sheets_per_student_first := 6,
    sheets_per_student_last := 4,
    extra_copies := 10,
    handout_sheets_per_student := 2,
    total_first_class_sheets :=
      3 * ((students_first_class * sheets_per_student_first) +
           (extra_copies * sheets_per_student_first) +
           (students_first_class * handout_sheets_per_student)),
    total_last_class_sheets :=
      3 * ((students_last_class * sheets_per_student_last) +
           (extra_copies * sheets_per_student_last) +
           (students_last_class * handout_sheets_per_student)),
    total_sheets := total_first_class_sheets + total_last_class_sheets
  in total_sheets = 1152 := by
  sorry

end sheets_of_paper_used_l101_101668


namespace harry_change_l101_101570

theorem harry_change (a : ℕ) :
  (∃ k : ℕ, a = 50 * k + 2 ∧ a < 100) ∧ (∃ m : ℕ, a = 5 * m + 4 ∧ a < 100) →
  a = 52 :=
by sorry

end harry_change_l101_101570


namespace find_f_minus_two_l101_101000

noncomputable def f : ℝ → ℝ := sorry

axiom f_add : ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y
axiom f_one : f 1 = 2

theorem find_f_minus_two : f (-2) = 2 :=
by sorry

end find_f_minus_two_l101_101000


namespace sum_fractional_parts_l101_101171

def fractional_part (x : ℚ) : ℚ := x - (int.floor x)

theorem sum_fractional_parts (p : ℕ) [fact (nat.prime p)] (hp : p % 4 = 1) :
  (∑ k in finset.range (p - 1), fractional_part ((k ^ 2 : ℚ) / p)) = (p - 1) / 4 :=
sorry

end sum_fractional_parts_l101_101171


namespace triangle_AEB_area_l101_101610

def Point := ℝ × ℝ
def Rectangle (A B C D : Point) : Prop := 
  A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ D.2 = C.2 ∧ 
  (B.1 - A.1) = 8 ∧ (C.2 - B.2) = 4

def OnSegment (P D C : Point) : Prop :=
  (min D.1 C.1 ≤ P.1 ∧ P.1 ≤ max D.1 C.1) ∧ 
  (min D.2 C.2 ≤ P.2 ∧ P.2 ≤ max D.2 C.2)

def Intersects (AF BG : Set Point) (E : Point) : Prop :=
  E ∈ AF ∧ E ∈ BG

def length (P Q : Point) : ℝ :=
  sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

def TriangleArea (A B C : Point) : ℝ :=
  abs (A.1*(B.2 - C.2) + B.1*(C.2 - A.2) + C.1*(A.2 - B.2)) / 2

theorem triangle_AEB_area (A B C D F G E : Point) (hRect : Rectangle A B C D)
  (hPointF : OnSegment F D C) (hPointG : OnSegment G D C)
  (hDF : length D F = 2) (hGC : length G C = 3)
  (hIntersects : Intersects ({ x : Point | ∃ l : ℝ, x = (l * F.1 + (1-l) * A.1, l * F.2 + (1-l) * A.2) } : Set Point) 
  ({ x : Point | ∃ l : ℝ, x = (l * G.1 + (1-l) * B.1, l * G.2 + (1-l) * B.2) } : Set Point) E) :
  TriangleArea A E B = 6 := sorry

end triangle_AEB_area_l101_101610


namespace area_of_section_l101_101991

-- Definitions based on problem conditions
variables (a h : ℝ)
-- Conditions
def is_positive_a (a : ℝ) : Prop := a > 0
def is_positive_h (h : ℝ) : Prop := h > a * real.sqrt (6) / 6

-- Correct answer in the proof
def derived_area (a h : ℝ) : ℝ := (3 * a^2 * h) / (4 * real.sqrt (a^2 + 3 * h^2))

-- Lean statement
theorem area_of_section (a h : ℝ) 
  (ha : is_positive_a a) 
  (hh : is_positive_h h) :
  ∃ (area : ℝ), area = derived_area a h := 
sorry

end area_of_section_l101_101991


namespace not_factorial_tails_count_l101_101457

def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) -- you can extend further if needed

def is_factorial_tail (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ f(m) = n

theorem not_factorial_tails_count : 
  ∃ c : ℕ, c = 500 ∧ ∀ k : ℕ, k < 2500 → ¬is_factorial_tail k → k < 2500 - c :=
by
  sorry

end not_factorial_tails_count_l101_101457


namespace zoo_total_animals_l101_101388

theorem zoo_total_animals :
  let tiger_enclosures := 4
  let zebra_enclosures := tiger_enclosures * 2
  let giraffe_enclosures := zebra_enclosures * 3
  let tigers := tiger_enclosures * 4
  let zebras := zebra_enclosures * 10
  let giraffes := giraffe_enclosures * 2
  let total_animals := tigers + zebras + giraffes
  in total_animals = 144 := by
  sorry

end zoo_total_animals_l101_101388


namespace count_undefined_values_l101_101503

-- Define the expression where the denominator causes the undefined values
def expression_denominator (x : ℝ) : ℝ := (x^2 + x - 6) * (x - 4)

-- State the theorem about the number of values that make the expression undefined
theorem count_undefined_values : (finset.univ.filter (λ x, expression_denominator x = 0)).card = 3 := by
  sorry

end count_undefined_values_l101_101503


namespace acute_angles_of_right_triangle_l101_101825

theorem acute_angles_of_right_triangle 
  (A B C F M1 M2 : Point) 
  (h_right_angle : ∠ C = 90°)
  (h_median : is_median C F)
  (h_incircle : inscribed_circle (triangle A B C))
  (h_chord_equal_section : chord_of_circle M1 M2 = section_ending_at_C F C) :
  acute_angles_of_triangle A B C = (86° + 24', 3° + 36') :=
sorry

end acute_angles_of_right_triangle_l101_101825


namespace line_in_first_second_third_quadrants_l101_101539

theorem line_in_first_second_third_quadrants (k b : ℝ) (h : ∀ x : ℝ, (∃ y : ℝ, y = k * x + b) ∧ 
            (by have I : ∃ p : ℝ × ℝ, p.1 ≠ 0 ∧ p.2 ≠ 0 := sorry; exact I)) :
  k > 0 ∧ b > 0 := sorry

end line_in_first_second_third_quadrants_l101_101539


namespace positive_difference_l101_101270

-- Define the conditions given in the problem
def conditions (x y : ℝ) : Prop :=
  x + y = 40 ∧ 3 * y - 4 * x = 20

-- The theorem to prove
theorem positive_difference (x y : ℝ) (h : conditions x y) : abs (y - x) = 11.42 :=
by
  sorry -- proof omitted

end positive_difference_l101_101270


namespace midpoint_BL_on_PQ_l101_101136

open EuclideanGeometry Real

variables {A B C D E L P Q : Point}

-- assuming given conditions
variables 
  (h_acute : AcuteAngledTriangle A B C)
  (h_BL_bisects_ABC : AngleBisector B L (TriangleVertexAngle A B C))
  (h_D_midpoint_arc_AB : Midpoint D (CircumcircleArc A B))
  (h_E_midpoint_arc_BC : Midpoint E (CircumcircleArc B C))
  (h_P_extension_BD : OnLine P (ExtensionOfLineThroughPoints B D))
  (h_Q_extension_BE : OnLine Q (ExtensionOfLineThroughPoints B E))
  (h_angle_APB_90 : ∡ A P B = 90 )
  (h_angle_CQB_90 : ∡ C Q B = 90 )

-- prove that the midpoint of BL lies on line PQ
theorem midpoint_BL_on_PQ :
  Midpoint (SegmentMidpoint B L) (LineThroughPoints P Q) :=
sorry

end midpoint_BL_on_PQ_l101_101136


namespace digit_1234_of_concatenated_sequence_eq_4_l101_101657

def sequence_concat (n : ℕ) : ℕ := 
  let digits := (List.join (List.map (λ i, (i : ℕ).digitCharList) (List.range' 1 (n + 1))) : List Char)
  let digit_as_string := digits.nthD 1233 -- We use index 1233 since Lean lists are 0-indexed
  (digit_as_string.val - '0'.val).toNat

theorem digit_1234_of_concatenated_sequence_eq_4 : sequence_concat 500 = 4 := by
  sorry

end digit_1234_of_concatenated_sequence_eq_4_l101_101657


namespace find_lengths_l101_101613

-- Definitions
variables {A B C D P T S Q R : Type} [MetricSpace X]
variables {AB CD BC AD : X}
variables {PA AQ QP : ℝ}

-- Conditions
def rectangle (ABCD : X) : Prop := -- define what it means for ABCD to be a rectangle
def perpendicular (TS BC : X) : Prop := -- define what it means for TS and BC to be perpendicular
def angle_90 (APD : X) : Prop := -- define what it means for an angle to be 90 degrees
def equal_length (CP PT : X) := -- define what it means for CP and PT to be equal

-- Given Conditions in Lean theorem statement
theorem find_lengths (rectangle ABCD) (angle_90 APD) (perpendicular TS BC) (equal_length CP PT)
  (PA = 18) (AQ = 24) (QP = 15) :
  CP = 2 * real.sqrt 17 ∧ QT = real.sqrt 157 :=
sorry

end find_lengths_l101_101613


namespace cylinder_volume_transformation_l101_101582

theorem cylinder_volume_transformation (π : ℝ) (r h : ℝ) (V : ℝ) (V_new : ℝ)
  (hV : V = π * r^2 * h) (hV_initial : V = 20) : V_new = π * (3 * r)^2 * (4 * h) :=
by
sorry

end cylinder_volume_transformation_l101_101582


namespace range_of_a_condition_l101_101179

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2*x + 4 else 2^x

theorem range_of_a_condition (a : ℝ) (h : f (f a) > f (f a + 1)) : a ∈ Set.Ioo (-5/2) (-2) := 
sorry

end range_of_a_condition_l101_101179


namespace count_irreducible_fractions_l101_101966

theorem count_irreducible_fractions (a b : ℕ) :
  let num := 2015
  let lower_bound := 2015 * 2016
  let upper_bound := 2015 ^ 2 
  (∀ (d : ℕ), lower_bound < d ∧ d ≤ upper_bound ∧ Int.gcd num d = 1) → 
  b = 1440 :=
by
  sorry

end count_irreducible_fractions_l101_101966


namespace scientific_notation_correct_l101_101419

def num : ℝ := 1040000000

theorem scientific_notation_correct : num = 1.04 * 10^9 := by
  sorry

end scientific_notation_correct_l101_101419


namespace triangle_circumcenter_interior_count_l101_101032

theorem triangle_circumcenter_interior_count :
  let types_of_triangles_with_interior_circumcenter := 
    {t : Type} 
      (h : t = "Equilateral" ∨ t = "Isosceles" ∨ t = "Scalene" ∨ t = "Right") in
  (∀ t, t ∈ types_of_triangles_with_interior_circumcenter → 
    (t = "Equilateral" ∨ t = "Isosceles") →
     true) → -- stating these 2 have circumcenters inside
  2 := sorry

end triangle_circumcenter_interior_count_l101_101032


namespace non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101444

def g (m : Nat) : Nat :=
  let f k := m / k
  f 5 + f 25 + f 125 + f 625 + f 3125 + f 15625 + f 78125 + f 390625 + f 1953125 + f 9765625 + f 48828125 + f 244140625 + f 1220703125 + f 6103515625 + f 30517578125 -- continue as needed

def is_factorial_tail (n : Nat) : Prop :=
  ∃ m : Nat, g m = n

def count_factorial_tails (N : Nat) : Nat :=
  (List.range N).filter (λ n, is_factorial_tail n).length

theorem non_factorial_tails_lemma : count_factorial_tails 2500 = 2000 := sorry

theorem count_non_factorial_tails (N : Nat) : Prop :=
  let total := N - 1
  total - count_factorial_tails N = 499

theorem solution : count_non_factorial_tails 2500 := sorry

end non_factorial_tails_lemma_count_non_factorial_tails_solution_l101_101444


namespace original_five_digit_number_l101_101807

theorem original_five_digit_number :
  ∃ N y x : ℕ, (N = 10 * y + x) ∧ (N + y = 54321) ∧ (N = 49383) :=
by
  -- The proof script goes here
  sorry

end original_five_digit_number_l101_101807


namespace sum_of_coefficients_l101_101036

noncomputable def polynomial := (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 + (1 + x)^6 + (1 + x)^7 + (1 + x)^8

theorem sum_of_coefficients :
  let a := 8,
  let sum_of_all_coeff := 510,
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = 502 :=
by
  sorry

end sum_of_coefficients_l101_101036


namespace minimize_expression_in_triangle_l101_101150

theorem minimize_expression_in_triangle
  (A B C : ℝ)
  (a b c : ℝ)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_opposite_angles : a = sin A ∧ b = sin B ∧ c = sin C)
  (h_condition : 3 * (cos (2 * A) - cos (2 * C)) = 1 - cos (2 * B)) :
  ∃ x : ℝ, x = (2 * sqrt 7 / 3) ∧
           ( ∀ A B C : ℝ, 3 * (cos (2 * A) - cos (2 * C)) = 1 - cos (2 * B) →
             let expression := (sin C / (sin A * sin B)) + (cos C / sin C) in
             expression ≥ x ∧ (expression = x) ↔ (tan A = sqrt 7 / 2)) :=
by
  sorry

end minimize_expression_in_triangle_l101_101150


namespace sqrt_problem1_sqrt_problem2_l101_101849

theorem sqrt_problem1 : sqrt 80 - sqrt 20 + sqrt 5 = 3 * sqrt 5 :=
by
  sorry

theorem sqrt_problem2 : (4 * sqrt 2 - 3 * sqrt 6) / (2 * sqrt 2) = 2 - (3 / 2) * sqrt 3 :=
by
  sorry

end sqrt_problem1_sqrt_problem2_l101_101849


namespace range_of_x_l101_101947

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x

theorem range_of_x (x : ℝ) (h : f (x^2 + 2) < f (3 * x)) : 1 < x ∧ x < 2 :=
by sorry

end range_of_x_l101_101947


namespace simplify_and_evaluate_l101_101206

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) : ((x - 2) / (x - 1)) / ((x + 1) - (3 / (x - 1))) = 1 / 5 :=
by
  sorry

end simplify_and_evaluate_l101_101206


namespace scientific_notation_correct_l101_101420

def num : ℝ := 1040000000

theorem scientific_notation_correct : num = 1.04 * 10^9 := by
  sorry

end scientific_notation_correct_l101_101420


namespace sin_double_angle_identity_l101_101520

open Real

theorem sin_double_angle_identity {α : ℝ} (h1 : π / 2 < α ∧ α < π) 
    (h2 : sin (α + π / 6) = 1 / 3) :
  sin (2 * α + π / 3) = -4 * sqrt 2 / 9 := 
by 
  sorry

end sin_double_angle_identity_l101_101520


namespace problem_solution_l101_101698

theorem problem_solution :
  ∃ r s : ℚ → ℚ,
    (∃ k : ℚ, r = λ x, k * (x + 4) * (x - 2) ∧ s = (λ x, (x + 4) * (x - 3))) ∧
    (tendsto (λ x, r x / s x) at_top (nhds (-3))) ∧
    (r 2 = 0) ∧
    (r 0 / s 0 = 2) := 
  sorry

end problem_solution_l101_101698


namespace geometric_sequence_sum_l101_101259

theorem geometric_sequence_sum
  (a r : ℝ)
  (h1 : a * (1 - r ^ 3000) / (1 - r) = 500)
  (h2 : a * (1 - r ^ 6000) / (1 - r) = 950) :
  a * (1 - r ^ 9000) / (1 - r) = 1355 :=
sorry

end geometric_sequence_sum_l101_101259


namespace find_angle_l101_101491

theorem find_angle (A : ℝ) (deg_to_rad : ℝ) :
  (1/2 * Real.sin (A / 2 * deg_to_rad) + Real.cos (A / 2 * deg_to_rad) = 1) →
  (A = 360) :=
sorry

end find_angle_l101_101491


namespace standard_deviation_of_applicants_ages_l101_101691

noncomputable def average_age : ℝ := 30
noncomputable def max_different_ages : ℝ := 15

theorem standard_deviation_of_applicants_ages 
  (σ : ℝ)
  (h : max_different_ages = 2 * σ) 
  : σ = 7.5 :=
by
  sorry

end standard_deviation_of_applicants_ages_l101_101691


namespace not_factorial_tails_count_l101_101453

def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) -- you can extend further if needed

def is_factorial_tail (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ f(m) = n

theorem not_factorial_tails_count : 
  ∃ c : ℕ, c = 500 ∧ ∀ k : ℕ, k < 2500 → ¬is_factorial_tail k → k < 2500 - c :=
by
  sorry

end not_factorial_tails_count_l101_101453


namespace exists_polynomial_H_flow_l101_101501

structure Multigraph (V E : Type) := 
  (adj : E → V × V)
  (edges : ∀ e : E, e → ℕ)
  (vertices : finset V)

variable (G : Multigraph V E)

def finite_abelian_group (H : Type) [add_comm_group H] [fintype H] := H

theorem exists_polynomial_H_flow (G : Multigraph V E) :
  ∃ P : polynomial ℤ, 
    ∀ (H : Type) [finite_abelian_group H],
      number_of_H_flows G H = P ((|H| - 1) : ℤ) :=
sorry

end exists_polynomial_H_flow_l101_101501


namespace increasing_range_l101_101118

theorem increasing_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → (x + a * real.sin x) ≤ (y + a * real.sin y)) ↔ (a ∈ set.Icc (-1) 1) :=
by
  sorry

end increasing_range_l101_101118


namespace monotone_f_implies_a_in_range_l101_101580

-- Define the function and the conditions
def f (x a : ℝ) := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

-- State the theorem we want to prove
theorem monotone_f_implies_a_in_range (a : ℝ) :
  (∀ x : ℝ, 1 - (2 / 3) * Real.cos (2 * x) + a * Real.cos x ≥ 0) →
  a ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ) :=
by
  intro h
  -- Placeholder for proof, which is not required
  sorry

end monotone_f_implies_a_in_range_l101_101580


namespace pure_alcohol_to_add_l101_101789

-- Variables and known values
variables (x : ℝ) -- amount of pure alcohol added
def initial_volume : ℝ := 6 -- initial solution volume in liters
def initial_concentration : ℝ := 0.35 -- initial alcohol concentration
def target_concentration : ℝ := 0.50 -- target alcohol concentration

-- Conditions
def initial_pure_alcohol : ℝ := initial_volume * initial_concentration

-- Statement of the problem
theorem pure_alcohol_to_add :
  (2.1 + x) / (initial_volume + x) = target_concentration ↔ x = 1.8 :=
by
  sorry

end pure_alcohol_to_add_l101_101789


namespace N_gt_M_l101_101169

def number_of_integral_solutions (eq : ℤ → ℤ → ℤ → ℤ → ℤ) (bound : ℤ) : ℤ :=
  ∑ x in finset.Icc 0 bound, 
    ∑ y in finset.Icc 0 bound, 
      ∑ z in finset.Icc 0 bound, 
        ∑ t in finset.Icc 0 bound, 
          if eq x y z t = 0 then 1 else 0

def N : ℤ :=
  number_of_integral_solutions (λ x y z t, x^2 - y^2 - (z^3 - t^3)) 1000000

def M : ℤ :=
  number_of_integral_solutions (λ x y z t, x^2 - y^2 - (z^3 - t^3 + 1)) 1000000

theorem N_gt_M : N > M :=
  sorry

end N_gt_M_l101_101169


namespace math_problem_l101_101353

theorem math_problem :
  3^(5+2) + 4^(1+3) = 39196 ∧
  2^(9+2) - 3^(4+1) = 3661 ∧
  1^(8+6) + 3^(2+3) = 250 ∧
  6^(5+4) - 4^(5+1) = 409977 → 
  5^(7+2) - 2^(5+3) = 1952869 :=
by
  sorry

end math_problem_l101_101353


namespace F_neg_a_eq_l101_101538

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

variables {f g : ℝ → ℝ} {a b : ℝ}

-- given the conditions
axiom h1 : is_odd f
axiom h2 : is_odd g
axiom f_def : ∀ x, F x = f x + 3 * g x + 5
axiom F_a : F a = b

-- proof goal
theorem F_neg_a_eq : F (-a) = 10 - b :=
by sorry

end F_neg_a_eq_l101_101538


namespace zoo_total_animals_l101_101392

theorem zoo_total_animals (tiger_enclosure : ℕ) (zebra_enclosure_per_tiger : ℕ) 
  (giraffe_enclosures_ratio : ℕ) (tigers_per_enclosure : ℕ) 
  (zebras_per_enclosure : ℕ) (giraffes_per_enclosure : ℕ) 
  (Htiger : tiger_enclosure = 4) (Hzebra_per_tiger : zebra_enclosure_per_tiger = 2) 
  (Hgiraffe_ratio : giraffe_enclosures_ratio = 3) (Ht_pe : tigers_per_enclosure = 4) 
  (Hz_pe : zebras_per_enclosure = 10) (Hg_pe : giraffes_per_enclosure = 2) : 
  let zebra_enclosures := tiger_enclosure * zebra_enclosure_per_tiger in
  let giraffe_enclosures := zebra_enclosures * giraffe_enclosures_ratio in
  let total_tigers := tiger_enclosure * tigers_per_enclosure in
  let total_zebras := zebra_enclosures * zebras_per_enclosure in
  let total_giraffes := giraffe_enclosures * giraffes_per_enclosure in
  total_tigers + total_zebras + total_giraffes = 144 :=
by
  sorry

end zoo_total_animals_l101_101392


namespace trigonometric_identity_l101_101772

theorem trigonometric_identity
  (h1 : ∀ x, sin x ^ 2 = (1 - cos (2 * x)) / 2)
  (h2 : ∀ x, cos x ^ 2 = (1 + cos (2 * x)) / 2)
  (h3 : cos (3 * π / 4) = -cos (π / 4))
  (h4 : cos (5 * π / 4) = -cos (π / 4))
  (h5 : cos (7 * π / 4) = cos (π / 4)) :
  sin (π / 8) ^ 4 + cos (3 * π / 8) ^ 4 + sin (5 * π / 8) ^ 4 + cos (7 * π / 8) ^ 4 = 3 / 2 :=
by
  sorry

end trigonometric_identity_l101_101772


namespace abs_value_inequality_l101_101898

theorem abs_value_inequality (x : ℝ) : (2 ≤ |x - 3| ∧ |x - 3| ≤ 8) ↔ (x ∈ Icc (-5 : ℝ) 1 ∨ x ∈ Icc 5 11) :=
by
  sorry

end abs_value_inequality_l101_101898


namespace solve_fn_l101_101244

open Set

def f0 (x : ℝ) := |x|

def f (n : ℕ) : ℝ → ℝ
| 0, x => f0 x
| (n+1), x => |f n x - 2|

noncomputable def solution_set (n : ℕ) : Set ℝ :=
  { x : ℝ | ∃ k : ℤ, x = ↑(2 * k + 1) ∧ |(2 * k + 1 : ℝ)| ≤ 2 * n + 1 }

theorem solve_fn (n : ℕ) : (n > 0) → { x : ℝ | f n x = 1 } = solution_set n := by
  sorry

end solve_fn_l101_101244


namespace sum_first_9000_terms_l101_101247

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_9000_terms (a r : ℝ) :
  geometric_sequence_sum a r 3000 = 500 →
  geometric_sequence_sum a r 6000 = 950 →
  geometric_sequence_sum a r 9000 = 1355 :=
by
  intros h1 h2
  sorry

end sum_first_9000_terms_l101_101247


namespace math_problem_l101_101998

section

variable {x y k₁ k₂ λ : ℝ}
variable {A B M N P : ℝ × ℝ}

-- Define the given points
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define non-vertical line slopes given the coordinate pairs
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the condition of product of slopes
def slopes_product_condition (P : ℝ × ℝ) : Prop := 
  slope A P * slope B P = -3 / 4

-- Define the trajectory equation
def trajectory_equation (P : ℝ × ℝ) : Prop := 
  P.2 ≠ 0 ∧ P.1^2 / 4 + P.2^2 / 3 = 1

-- Define the condition kAM * kAN = lambda
def slopes_MN_condition (M N : ℝ × ℝ) (k₁ : ℝ) : Prop := 
  slope A M * slope A N = λ

-- Define the condition that MN passes through the midpoint of AB
def passes_midpoint_AB (M N : ℝ × ℝ) : Prop := 
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (midpoint.2 = (slope M N * (midpoint.1 - M.1) + M.2) ∧ (midpoint.2 = (slope M N * (midpoint.1 - N.1) + N.2)))

-- Main theorem
theorem math_problem (P : ℝ × ℝ) (M N : ℝ × ℝ) (k₁ : ℝ):
  slopes_product_condition P →
  trajectory_equation P →
  slopes_MN_condition M N k₁ →
  λ = -3 / 4 →
  passes_midpoint_AB M N :=
sorry

end

end math_problem_l101_101998


namespace geometric_sequence_sum_l101_101268

noncomputable theory

-- Define the conditions and the problem statement
theorem geometric_sequence_sum (a r : ℝ) (h₁ : a * (1 - r^3000) / (1 - r) = 500) 
  (h₂ : a * (1 - r^6000) / (1 - r) = 950) : 
  a * (1 - r^9000) / (1 - r) = 1355 := 
sorry

end geometric_sequence_sum_l101_101268


namespace square_ac_lt_2_fg_l101_101654

variables {A B C D E F G : Point}
variable [square ABCD]

-- Definitions & assumptions
def is_internal_point_on_side (E : Point) (AD : Line) : Prop :=
  -- Assume E is an internal point on the line AD
  -- (How we define this mathematically might depend on a broader geometry context)

def is_foot_of_perpendicular (F : Point) (line1 : Line) (line2 : Line) : Prop :=
  -- Assume F is the foot of the perpendicular from a point on line1 to line2
  -- (How we define this mathematically might depend on a broader geometry context)

def collinear (P Q R : Point) : Prop := 
  -- Assume a basic collinearity definition
  P ∈ line Q R

-- Theorem & statement
theorem square_ac_lt_2_fg (hE : is_internal_point_on_side E AD)
                           (hF : is_foot_of_perpendicular F (line_through B C) (line_through C E))
                           (hG : collinear G B F ∧ BG = FG ∧ collinear G (midpoint E F) (parallel_line BC)) :
  dist A C < 2 * dist F G :=
by
  sorry

end square_ac_lt_2_fg_l101_101654


namespace min_value_of_m_l101_101158

noncomputable def m : (ℝ → ℝ → ℝ) → ℝ :=
λ f, Inf (set_of (λ s: ℝ, s ≥ 1) : set ℝ) (λ s, (f (s + 1) (s + 1)) - (f (s + 1) s) - (f s (s + 1)) + (f s s))

theorem min_value_of_m (f : ℝ → ℝ → ℝ) 
  (h1 : ∀ x y, x ≥ 1 → y ≥ 1 →  x * (fderiv! ℝ (λ p : ℝ × ℝ, f p.1 p.2) (x, y)).fst + y * (fderiv! ℝ (λ p : ℝ × ℝ, f p.1 p.2) (x, y)).snd = x * y * log (x * y))
  (h2 : ∀ x y, x ≥ 1 → y ≥ 1 →  x^2 *  (fderiv! ℝ (λ p : ℝ × ℝ, (fderiv! ℝ (λ p : ℝ × ℝ, f p.1 p.2) p).fst) (x, y)).fst + y^2 * (fderiv! ℝ (λ p : ℝ × ℝ, (fderiv! ℝ (λ p : ℝ × ℝ, f p.1 p.2) p).snd) (x, y)).snd = x * y) :
  m f = (1 / 2 : ℝ) + log 4 :=
sorry

end min_value_of_m_l101_101158


namespace geometric_seq_sum_l101_101066

theorem geometric_seq_sum (a : ℝ) (q : ℝ) (ha : a ≠ 0) (hq : q ≠ 1) 
    (hS4 : a * (1 - q^4) / (1 - q) = 1) 
    (hS12 : a * (1 - q^12) / (1 - q) = 13) 
    : a * q^12 * (1 + q + q^2 + q^3) = 27 := 
by
  sorry

end geometric_seq_sum_l101_101066


namespace plane_speed_in_still_air_l101_101821

theorem plane_speed_in_still_air (P W : ℝ) 
  (h1 : (P + W) * 3 = 900) 
  (h2 : (P - W) * 4 = 900) 
  : P = 262.5 :=
by
  sorry

end plane_speed_in_still_air_l101_101821


namespace relationship_between_a_b_c_l101_101111

noncomputable def a := 33
noncomputable def b := 5 * 6^1 + 2 * 6^0
noncomputable def c := 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem relationship_between_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_between_a_b_c_l101_101111


namespace opposite_face_is_D_l101_101424

-- Define the six faces
inductive Face
| A | B | C | D | E | F

open Face

-- Define the adjacency relation
def is_adjacent (x y : Face) : Prop :=
(y = B ∧ x = A) ∨ (y = F ∧ x = A) ∨ (y = C ∧ x = A) ∨ (y = E ∧ x = A)

-- Define the problem statement in Lean
theorem opposite_face_is_D : 
  (∀ (x : Face), is_adjacent A x ↔ x = B ∨ x = F ∨ x = C ∨ x = E) →
  (¬ (is_adjacent A D)) →
  True :=
by
  intro adj_relation non_adj_relation
  sorry

end opposite_face_is_D_l101_101424


namespace average_selling_price_is_86_l101_101819

def selling_prices := [82, 86, 90, 85, 87, 85, 86, 82, 90, 87, 85, 86, 82, 86, 87, 90]

def average (prices : List Nat) : Nat :=
  (prices.sum) / prices.length

theorem average_selling_price_is_86 :
  average selling_prices = 86 :=
by
  sorry

end average_selling_price_is_86_l101_101819


namespace perpendicular_line_equation_l101_101493

theorem perpendicular_line_equation :
∃ (m b : ℝ), (∀ x y : ℝ, y = x^3 - 1 → y = m * x + b) ∧
    m = -1/3 ∧ b = 1/3 :=
begin
  existsi (-1/3 : ℝ),
  existsi (1/3 : ℝ),
  split,
  { intros x y h,
    have h_deriv : deriv (λ x, x^3 - 1) = λ x, 3 * x^2,
    { sorry, },
    have tangent_slope : deriv (λ x, x^3 - 1) 1 = 3,
    { sorry, },
    have perpendicular_slope : -1 / 3,
    { sorry, },
    rw [perpendicular_slope],
    exact (λ x, x * (-1 / 3) + 1 / 3) },
  split; refl
end

end perpendicular_line_equation_l101_101493


namespace num_not_factorial_tails_lt_2500_l101_101472

-- Definition of the function f(m)
def f (m : ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625 + m / 78125 + m / 390625 + m / 1953125

-- The theorem stating the main result
theorem num_not_factorial_tails_lt_2500 : 
  let count_non_tails := ∑ k in finset.range 2500, if ∀ m, f m ≠ k then 1 else 0
  in count_non_tails = 500 :=
by
  sorry

end num_not_factorial_tails_lt_2500_l101_101472


namespace not_factorial_tails_count_l101_101455

def f (m : ℕ) : ℕ :=
  (m / 5) + (m / 25) + (m / 125) + (m / 625) + (m / 3125) + (m / 15625) -- you can extend further if needed

def is_factorial_tail (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ f(m) = n

theorem not_factorial_tails_count : 
  ∃ c : ℕ, c = 500 ∧ ∀ k : ℕ, k < 2500 → ¬is_factorial_tail k → k < 2500 - c :=
by
  sorry

end not_factorial_tails_count_l101_101455


namespace inhabitants_count_smallest_largest_square_average_area_l101_101590

-- Definitions based on the conditions
def is_squareland_square (s : ℕ) : Prop :=
  s % 2 = 0

def has_friends (s : ℕ) : Prop :=
  ∃ f1 f2, is_squareland_square s ∧ is_squareland_square f1 ∧ is_squareland_square f2 ∧ 4 * f1 = 4 * s - 8 ∧ 4 * f2 = 4 * s + 8

def avg_side_length (a : ℕ) (t : ℕ) : Prop :=
  2 * a + 2 * t = 30

def unique_dimensions (squares : list ℕ) : Prop :=
  ∀ x ∈ squares, ∀ y ∈ squares, x ≠ y → x ≠ y

def smallest_largest_relation (a : ℕ) (t : ℕ) : Prop :=
  4 * a = a + 2 * t

-- Questions translated to be proven given the conditions
theorem inhabitants_count : ∀ squares : list ℕ,
  unique_dimensions squares →
  (∀ s ∈ squares, has_friends s) →
  avg_side_length 6 9 →
  smallest_largest_relation 6 9 →
  squares.length = 10 :=
sorry

theorem smallest_largest_square : 
  unique_dimensions [6, 8, 10, 12, 14, 16, 18, 20, 22, 24] →
  (∀ s ∈ [6, 8, 10, 12, 14, 16, 18, 20, 22, 24], has_friends s) →
  avg_side_length 6 9 →
  smallest_largest_relation 6 9 →
  ([6, 24]) =
    [6, 24] :=
sorry

theorem average_area :
  (∑ x in [6, 8, 10, 12, 14, 16, 18, 20, 22, 24], x * x) / 10 = 258 :=
sorry

end inhabitants_count_smallest_largest_square_average_area_l101_101590


namespace range_of_a_for_cos_equation_l101_101873

theorem range_of_a_for_cos_equation : 
  (∃ x : ℝ, cos x ^ 2 - 2 * cos x - a = 0) ↔ -1 ≤ a ∧ a ≤ 3 :=
sorry

end range_of_a_for_cos_equation_l101_101873


namespace reciprocal_of_neg_5_4_l101_101306

theorem reciprocal_of_neg_5_4 : (⁻¹ : ℚ → ℚ) (-5 / 4) = -4 / 5 :=
by
  sorry

end reciprocal_of_neg_5_4_l101_101306


namespace num_not_factorial_tails_lt_2500_l101_101468

-- Definition of the function f(m)
def f (m : ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625 + m / 78125 + m / 390625 + m / 1953125

-- The theorem stating the main result
theorem num_not_factorial_tails_lt_2500 : 
  let count_non_tails := ∑ k in finset.range 2500, if ∀ m, f m ≠ k then 1 else 0
  in count_non_tails = 500 :=
by
  sorry

end num_not_factorial_tails_lt_2500_l101_101468


namespace scientific_notation_of_1040000000_l101_101422

theorem scientific_notation_of_1040000000 : (1.04 * 10^9 = 1040000000) :=
by
  -- Math proof steps can be added here
  sorry

end scientific_notation_of_1040000000_l101_101422


namespace investment_final_amount_l101_101401

-- Define the given constants:
def P : ℝ := 10000
def r : ℝ := 0.045
def n : ℝ := 12
def t : ℝ := 3

-- Define the compound interest formula
def compoundInterest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- State the theorem
theorem investment_final_amount : compoundInterest P r n t = 11446 := by
  sorry

end investment_final_amount_l101_101401


namespace percentage_decrease_is_24_l101_101711

-- Define the given constants Rs. 820 and Rs. 1078.95
def current_price : ℝ := 820
def original_price : ℝ := 1078.95

-- Define the percentage decrease P
def percentage_decrease (P : ℝ) : Prop :=
  original_price - (P / 100) * original_price = current_price

-- Prove that percentage decrease P is approximately 24
theorem percentage_decrease_is_24 : percentage_decrease 24 :=
by
  unfold percentage_decrease
  sorry

end percentage_decrease_is_24_l101_101711


namespace correct_calculation_l101_101130

theorem correct_calculation :
  (- (4 + 2 / 3) - (1 + 5 / 6) - (- (18 + 1 / 2)) + (- (13 + 3 / 4))) = - (7 / 4) :=
by 
  sorry

end correct_calculation_l101_101130


namespace inversely_varied_x3_sqrtx_l101_101213

theorem inversely_varied_x3_sqrtx (k : ℝ) (h : ∀ x : ℝ, (x^3) * (Real.sqrt x) = k) :
  (h 4) = 128 → (h 64) = 128 → y = (1 : ℝ) / 16384 :=
by
  sorry

end inversely_varied_x3_sqrtx_l101_101213


namespace chess_tournament_l101_101993

theorem chess_tournament (n g : ℕ) 
  (hn : n = 50) 
  (hg : g = 61) 
  (h2or3 : ∀ k, k < 50 → (players_game k = 2 ∨ players_game k = 3)) :
  ¬ (∃ (x : ℕ), (x ≤ 50) ∧ (3 * x + 2 * (50 - x) = 2 * g) ∧
      (∀ i j, ((i < 50 ∧ j < 50 ∧ players_game i = 3 ∧ players_game j = 3) → (i ≠ j → ¬ (played_each_other i j))))) :=
begin
  sorry
end

end chess_tournament_l101_101993


namespace non_factorial_tails_lt_2500_l101_101435

-- Define the function f(m)
def f (m: ℕ) : ℕ :=
  m / 5 + m / 25 + m / 125 + m / 625 + m / 3125 + m / 15625

-- Define what it means to be a factorial tail
def is_factorial_tail (n: ℕ) : Prop :=
  ∃ m : ℕ, f(m) = n

-- Define the statement to be proved
theorem non_factorial_tails_lt_2500 : 
  (finset.range 2500).filter (λ n, ¬ is_factorial_tail n).card = 499 :=
by
  sorry

end non_factorial_tails_lt_2500_l101_101435


namespace sum_of_interior_angles_6_find_n_from_300_degrees_l101_101049

-- Definitions and statement for part 1:
def sum_of_interior_angles (n : ℕ) : ℕ :=
  (n - 2) * 180

theorem sum_of_interior_angles_6 :
  sum_of_interior_angles 6 = 720 := 
by
  sorry

-- Definitions and statement for part 2:
def find_n_from_angles (angle : ℕ) : ℕ := 
  (angle / 180) + 2

theorem find_n_from_300_degrees :
  find_n_from_angles 900 = 7 :=
by
  sorry

end sum_of_interior_angles_6_find_n_from_300_degrees_l101_101049


namespace bottle_caps_remaining_l101_101675

-- Define the problem using the conditions and the desired proof.
theorem bottle_caps_remaining (original_count removed_count remaining_count : ℕ) 
    (h_original : original_count = 87) 
    (h_removed : removed_count = 47)
    (h_remaining : remaining_count = original_count - removed_count) :
    remaining_count = 40 :=
by 
  rw [h_original, h_removed] at h_remaining 
  exact h_remaining

end bottle_caps_remaining_l101_101675


namespace distance_swim_downstream_correct_l101_101373

def speed_man_still_water : ℝ := 7
def time_taken : ℝ := 5
def distance_upstream : ℝ := 25

lemma distance_swim_downstream (V_m : ℝ) (t : ℝ) (d_up : ℝ) : 
  t * ((V_m + (V_m - d_up / t)) / 2) = 45 :=
by
  have h_speed_upstream : (V_m - (d_up / t)) = d_up / t := by sorry
  have h_speed_stream : (d_up / t) = (V_m - (d_up / t)) := by sorry
  have h_distance_downstream : t * ((V_m + (V_m - (d_up / t)) / 2)) = t * (V_m + (V_m - (V_m - d_up / t))) := by sorry
  sorry

noncomputable def distance_swim_downstream_value : ℝ :=
  9 * 5

theorem distance_swim_downstream_correct :
  distance_swim_downstream_value = 45 :=
by
  sorry

end distance_swim_downstream_correct_l101_101373


namespace area_of_triangle_AEB_l101_101611

noncomputable theory

open_locale classical

variables {AB BC DF GC : ℝ}
variables (A B C D E F G : Type*)

-- Define the properties of points
variables [is_rectangular A B C D]
variables [line_through A F E]
variables [line_through B G E]

def AB_length : AB = 8 := by sorry
def BC_length : BC = 4 := by sorry
def DF_length : DF = 2 := by sorry
def GC_length : GC = 3 := by sorry

theorem area_of_triangle_AEB : area (triangle A E B) = 128/5 :=
by
  sorry

end area_of_triangle_AEB_l101_101611


namespace triathlon_bike_speed_l101_101185

theorem triathlon_bike_speed :
  let t_swim := (1 / 8 : ℝ) / 1,
      t_run := (2 : ℝ) / 8,
      t_total := t_swim + t_run,
      t_goal := 1.5,
      t_bike := t_goal - t_total,
      d_bike := 8,
      v_bike := d_bike / t_bike
  in v_bike = (64 / 9 : ℝ) :=
by sorry

end triathlon_bike_speed_l101_101185


namespace solve_for_n_l101_101682

theorem solve_for_n (n : ℝ) (h : (n - 5)^3 = (1 / 9)^(-1)) : n = 5 + 3^(2 / 3) :=
by 
  sorry

end solve_for_n_l101_101682


namespace sum_of_sequence_equals_l101_101474

noncomputable def y : ℕ → ℕ
| 0     := 0
| 1     := 217
| (n+2) := y (n+1) * y (n+1) - y (n+1)

noncomputable def sum_sequence (n : ℕ) : ℝ :=
∑ i in Finset.range n, (1 / (y (i + 1) - 1 : ℝ))

theorem sum_of_sequence_equals :
  (∑' n : ℕ, 1 / (y (n + 1) - 1 : ℝ)) = 1 / 216 :=
by sorry

end sum_of_sequence_equals_l101_101474


namespace tony_exercise_time_l101_101292

noncomputable def time_walking (distance walk_speed : ℝ) : ℝ := distance / walk_speed
noncomputable def time_running (distance run_speed : ℝ) : ℝ := distance / run_speed
noncomputable def time_swimming (distance swim_speed : ℝ) : ℝ := distance / swim_speed
noncomputable def total_time_per_day (walk_time run_time swim_time : ℝ) : ℝ := walk_time + run_time + swim_time
noncomputable def total_time_per_week (daily_time : ℝ) := daily_time * 7

theorem tony_exercise_time :
  let walk_distance := 5
  let walk_speed := 2.5
  let run_distance := 15
  let run_speed := 4.5
  let swim_distance := 1.5
  let swim_speed := 1
  let day_time := total_time_per_day 
                   (time_walking walk_distance walk_speed)
                   (time_running run_distance run_speed)
                   (time_swimming swim_distance swim_speed)
  in total_time_per_week day_time = 47.8333 :=
by
  sorry

end tony_exercise_time_l101_101292


namespace greatest_divisor_6215_7373_l101_101338

theorem greatest_divisor_6215_7373 : 
  Nat.gcd (6215 - 23) (7373 - 29) = 144 := by
  sorry

end greatest_divisor_6215_7373_l101_101338


namespace functional_equation_solution_l101_101867

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f(2 * x + f(y)) = x + y + f(x)) → (∀ x : ℝ, f(x) = x) :=
by
  intros f H
  sorry

end functional_equation_solution_l101_101867


namespace original_five_digit_number_l101_101809

theorem original_five_digit_number :
  ∃ N y x : ℕ, (N = 10 * y + x) ∧ (N + y = 54321) ∧ (N = 49383) :=
by
  -- The proof script goes here
  sorry

end original_five_digit_number_l101_101809


namespace range_of_y_l101_101060

theorem range_of_y (m n k y : ℝ)
  (h₁ : 0 ≤ m)
  (h₂ : 0 ≤ n)
  (h₃ : 0 ≤ k)
  (h₄ : m - k + 1 = 1)
  (h₅ : 2 * k + n = 1)
  (h₆ : y = 2 * k^2 - 8 * k + 6)
  : 5 / 2 ≤ y ∧ y ≤ 6 :=
by
  sorry

end range_of_y_l101_101060


namespace problem1_l101_101349

theorem problem1 : 
  (5 + 1/16)^0.5 - 2 * (2 + 10/27)^(-2/3) - 2 * (Real.sqrt(2 + Real.pi))^0 / (3 / 4)^(-2) = 0 :=
by
  sorry

end problem1_l101_101349


namespace arithmetic_sequence_sum_Tn_l101_101997

-- Definitions and Conditions
def point_on_parabola (n : ℕ) : Prop := ∀ (P : ℕ → ℝ × ℝ), 
  (∃ x : ℝ, x ≥ 0 ∧ P n = (x, x^2)) 

def circle_tangent_xaxis (P : ℝ × ℝ) : Prop := 
  P.2 = (P.1)^2

def circles_tangent (P Q : ℝ × ℝ) : Prop :=
  ∃ R Q', (circle_tangent_xaxis P) ∧ (circle_tangent_xaxis Q) ∧ 
          (R = P.2) ∧ (Q'.2 = Q.2) ∧ 
          ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (R + Q'.2)^2)

def decreasing_sequence (P : ℕ → ℝ × ℝ) : Prop :=
  ∀ n, (P (n+1)).1 < (P n).1

-- Convert conditions into functions
def conditions (n : ℕ) : Prop :=
  (∃ P : ℕ → ℝ × ℝ, point_on_parabola n ∧ 
     decreasing_sequence P ∧ circles_tangent (P n) (P (n+1)) ∧ 
     ((P 1).1 = 1)) 

-- Statement 1 : Arithmetic Sequence
theorem arithmetic_sequence (n : ℕ) (P : ℕ → ℝ × ℝ) 
  (h_conditions : conditions n) :
  ∃ d : ℝ, ∀ m, P m.1 ≠ 0 → (1 / (P (m+1)).1 - 1 / (P m).1) = d :=
by sorry

-- Statement 2 : T_n < 3 * sqrt(pi) / 2
noncomputable def Sn (P : ℕ → ℝ × ℝ) (n : ℕ) := 
  π * ((P n).1)^4

noncomputable def Tn (P : ℕ → ℝ × ℝ) (n : ℕ) := 
  ∑ k in (range (n+1)), real.sqrt (Sn P k)

theorem sum_Tn (n : ℕ) (P : ℕ → ℝ × ℝ) 
  (h_conditions : conditions n) :
  Tn P n < 3 * real.sqrt π / 2 :=
by sorry

end arithmetic_sequence_sum_Tn_l101_101997


namespace a_even_is_square_l101_101820

noncomputable def a : ℕ → ℕ
| 1     := 1
| 2     := 1
| 3     := 2
| 4     := 4
| (n+1) := if n ≥ 4 then a n + a (n - 2) + a (n - 3) else a n -- Simplified condition

theorem a_even_is_square (n : ℕ) (h : n ≥ 1) : ∃ k : ℕ, a (2 * n) = k * k :=
sorry -- Proof goes here

end a_even_is_square_l101_101820


namespace train_pass_time_l101_101831

/-
Given a train of length 110 meters, a train speed of 30 km/h, and a man running at 3 km/h in the opposite direction,
prove that the train will pass the man in approximately 12 seconds.
-/
theorem train_pass_time (l v_train v_man : ℕ) (h_l : l = 110) (h_vtrain : v_train = 30) (h_vman : v_man = 3) : 
  (110 / (33 * 1000 / 3600 : ℝ) ≈ 12) :=
by {
  sorry
}

end train_pass_time_l101_101831


namespace find_m_l101_101999

noncomputable def parametric_x (t m : ℝ) := (√3/2) * t + m
noncomputable def parametric_y (t : ℝ) := (1/2) * t

noncomputable def circle_eq (x y : ℝ) := (x - 2)^2 + y^2 = 4

noncomputable def line_eq (x y m : ℝ) := x - √3 * y - m = 0

def is_tangent (m : ℝ) : Prop :=
  let dist := abs ((2 - m) / (Real.sqrt ((√3/2)^2 + (1/2)^2))) in
  dist = 2

theorem find_m (m: ℝ) (t : ℝ):
  (∀ t, line_eq (parametric_x t m) (parametric_y t) m) →
  (circle_eq (parametric_x t m) (parametric_y t)) →
  is_tangent m :=
sorry

end find_m_l101_101999
