import Mathlib

namespace sum_powers_of_5_mod_8_l238_238171

theorem sum_powers_of_5_mod_8 :
  (List.sum (List.map (fun n => (5^n % 8)) (List.range 2011))) % 8 = 4 := 
  sorry

end sum_powers_of_5_mod_8_l238_238171


namespace simplify_complex_fraction_pow_l238_238864

theorem simplify_complex_fraction_pow :
  ( (1 + 2 * Complex.i) / (1 - 2 * Complex.i) ) ^ 1004 = 1 := by
  sorry

end simplify_complex_fraction_pow_l238_238864


namespace sum_consecutive_sides_geq_l238_238785

open Nat

theorem sum_consecutive_sides_geq (p q : ℕ) (h_p : p.Prime) (h_q : q.Prime) (h_p_lt_q : p < q)
  (a : ℕ → ℕ) (h_distinct : ∀ (i j : ℕ), i ≠ j → a i ≠ a j)
  (h_len : ∀ (i : ℕ), 0 ≤ i → i < p * q → 0 < a i)
  (h_sum_equi : (Σ i in finset.range (p * q), a i % i = 0)) :
  ∀ k : ℕ, 1 ≤ k → k ≤ p → (Σ i in finset.Ico 0 k, a i) ≥ (k ^ 3 + k) / 2 := 
sorry

end sum_consecutive_sides_geq_l238_238785


namespace factor_tree_result_l238_238767

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Given primes
axiom prime_2 : prime 2
axiom prime_7 : prime 7
axiom prime_11 : prime 11

variables (Y Z F X : ℕ)
variables (A1 : Y = 7 * 11)
variables (A2 : F = 11 * 2)
variables (A3 : Z = 7 * F)
variables (A4 : X = Y * Z)

theorem factor_tree_result : X = 11858 :=
by
  have hY : Y = 77 := A1
  have hF : F = 22 := by rw [A2]; exact mul_comm 11 2
  have hZ : Z = 154 := by rw [A3, hF]; exact mul_comm 7 22
  have hX : X = 77 * 154 := by rw [A4, hY, hZ]
  exact hX
#check factor_tree_result

end factor_tree_result_l238_238767


namespace billy_sleep_total_l238_238749

def billy_sleep : Prop :=
  let first_night := 6
  let second_night := first_night + 2
  let third_night := second_night / 2
  let fourth_night := third_night * 3
  first_night + second_night + third_night + fourth_night = 30

theorem billy_sleep_total : billy_sleep := by
  sorry

end billy_sleep_total_l238_238749


namespace polynomial_evaluation_l238_238526

theorem polynomial_evaluation (P : ℕ → ℝ) (n : ℕ) 
  (h_degree : ∀ k : ℕ, k ≤ n → P k = k / (k + 1)) 
  (h_poly : ∀ k : ℕ, ∃ a : ℝ, P k = a * k ^ n) : 
  P (n + 1) = (n + 1 + (-1) ^ (n + 1)) / (n + 2) :=
by 
  sorry

end polynomial_evaluation_l238_238526


namespace max_rabbits_l238_238080

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l238_238080


namespace all_matchings_have_same_weight_l238_238478

variables {V : Type*} [Fintype V] [DecidableEq V]
variables {A B : Finset V}
variables (G : SimpleGraph V) [BipartiteGraph G A B]
variables (w : G.EdgeSet → ℕ) (n : ℕ)
variables [Fintype (G.perfectMatchings)]
variables (G' : SimpleGraph V)

noncomputable def minWeightMatching (G : SimpleGraph V) (w : G.EdgeSet → ℕ) : ℕ :=
  Inf (G.perfectMatchings.image (λ m, ∑ e in m.edges, w e))

def minimumWeightEdges (G : SimpleGraph V) (w : G.EdgeSet → ℕ) : G.EdgeSet :=
  { e ∈ G.EdgeSet | ∃ m ∈ G.perfectMatchings, ∑ e in m.edges, w e = minWeightMatching G w }

def G'_construct (G : SimpleGraph V) (w : G.EdgeSet → ℕ) : SimpleGraph V :=
  { adj := λ a b, a ≠ b ∧ (a, b) ∈ minimumWeightEdges G w }

theorem all_matchings_have_same_weight :
  ∀ m1 m2 ∈ G'.perfectMatchings, (∑ e in m1.edges, w e) = (∑ e in m2.edges, w e) :=
sorry

end all_matchings_have_same_weight_l238_238478


namespace exists_line_with_given_segment_ratios_l238_238647

noncomputable def segment_ratios (l l1 l2 l3 l4 : Line) : Prop :=
  ∃ A B C D : Point,
    collinear A B C D ∧
    A ∈ l1 ∧ B ∈ l2 ∧ C ∈ l3 ∧ D ∈ l4 ∧
    ratio (segment_length A B) (segment_length B C) = known_ratio_AB_BC ∧
    ratio (segment_length B C) (segment_length C D) = known_ratio_BC_CD

theorem exists_line_with_given_segment_ratios (l1 l2 l3 l4 : Line) :
  ∃ l : Line, segment_ratios l l1 l2 l3 l4 := sorry

end exists_line_with_given_segment_ratios_l238_238647


namespace numbers_represented_3_units_from_A_l238_238500

theorem numbers_represented_3_units_from_A (A : ℝ) (x : ℝ) (h : A = -2) : 
  abs (x + 2) = 3 ↔ x = 1 ∨ x = -5 := by
  sorry

end numbers_represented_3_units_from_A_l238_238500


namespace count_even_integers_l238_238402

theorem count_even_integers (m : ℤ) (h : m ≠ 0 ∧ even m ∧ abs m ≤ 10) : 
  ∃ n, n = 10 := 
by
  -- The proof would go here.
  sorry

end count_even_integers_l238_238402


namespace minimize_quadratic_l238_238200

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end minimize_quadratic_l238_238200


namespace SamMoreThanAverage_l238_238288

variable (b r s e m : ℕ)
variable (countB countR countS countE countM : ℕ)

def BridgetCount : ℕ := 14
def ReginaldCount (b_count: ℕ) : ℕ := b_count - 2
def SamCount (r_count: ℕ) : ℕ := r_count + 4
def EmmaCount (s_count: ℕ) : ℕ := s_count + 3
def MaxCount (b_count: ℕ) : ℕ := b_count - 7

def TotalStars (b_count r_count s_count e_count m_count: ℕ) : ℕ :=
  b_count + r_count + s_count + e_count + m_count

def AverageStars (total: ℕ) : ℝ :=
  total / 5

theorem SamMoreThanAverage :
  let b_count := BridgetCount
  let r_count := ReginaldCount b_count
  let s_count := SamCount r_count
  let e_count := EmmaCount s_count
  let m_count := MaxCount b_count
  let total := TotalStars b_count r_count s_count e_count m_count
  let average := AverageStars total
  (s_count : ℝ) - average = 2.4 :=
by
  sorry

end SamMoreThanAverage_l238_238288


namespace maximum_rabbits_condition_l238_238104

-- Define the conditions and constraints
variables {N : ℕ}
variables (total_rabbits long_ears jump_far : ℕ)
variables (at_least_three_with_both : Prop)

-- State the conditions with exact values and assumptions
def conditions := 
  total_rabbits = N ∧
  long_ears = 13 ∧
  jump_far = 17 ∧
  at_least_three_with_both = (∃ a b c : ℕ, a >= 3 ∧ b = (long_ears - a) ∧ c = (jump_far - a))

-- State the theorem to be proved
theorem maximum_rabbits_condition :
  ∀ {N : ℕ}, conditions N long_ears jump_far at_least_three_with_both → N ≤ 27 :=
by sorry

end maximum_rabbits_condition_l238_238104


namespace problem_l238_238797

def remainder_when_divided_by_20 (a b : ℕ) : ℕ := (a + b) % 20

theorem problem (a b : ℕ) (n m : ℤ) (h1 : a = 60 * n + 53) (h2 : b = 50 * m + 24) : 
  remainder_when_divided_by_20 a b = 17 := 
by
  -- Proof would go here
  sorry

end problem_l238_238797


namespace minimize_quadratic_l238_238193

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_l238_238193


namespace sqrt_x_minus_3_meaningful_l238_238905

theorem sqrt_x_minus_3_meaningful (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 3)) ↔ x ≥ 3 :=
by 
  sorry

end sqrt_x_minus_3_meaningful_l238_238905


namespace ratio_angle_OBE_BAC_eq_zero_l238_238276

theorem ratio_angle_OBE_BAC_eq_zero
  (A B C O E : Point)
  (h1 : TriangleInscribed A B C O)
  (h2 : ArcLength A B 90)
  (h3 : ArcLength B C 90)
  (h4 : OnArcMinorAC E A C)
  (h5 : Perpendicular (Line O E) (Line A C)) :
  (Angle OB E) / (Angle BA C) = 0 :=
by
  sorry

end ratio_angle_OBE_BAC_eq_zero_l238_238276


namespace scientific_notation_of_0_0000006_l238_238325

theorem scientific_notation_of_0_0000006 : 6 * 10^(-7:ℤ) = 0.0000006 :=
by
  sorry

end scientific_notation_of_0_0000006_l238_238325


namespace find_phi_l238_238380

def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 4)

def g (x : ℝ) : ℝ := Real.cos (4 * x + Real.pi / 4)

def h (x : ℝ) (ϕ : ℝ) : ℝ := Real.cos (4 * x - 4 * ϕ + Real.pi / 4)

theorem find_phi (ϕ : ℝ) (k : ℤ) (ϕ_nonneg : 0 ≤ ϕ) 
  (symmetry_condition : -4 * ϕ + Real.pi / 4 = k * Real.pi + Real.pi / 2) : 
  ϕ = 3 * Real.pi / 16 :=
by
  sorry

end find_phi_l238_238380


namespace sets_equal_l238_238142

def M := { u | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l }
def N := { u | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r }

theorem sets_equal : M = N :=
by sorry

end sets_equal_l238_238142


namespace necessary_but_not_sufficient_condition_l238_238572

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (2 < x ∧ x < 5) → (3 < x ∧ x < 4) →
  (2 < x ∧ x < 5) ∧ ¬((2 < x ∧ x < 5) → (3 < x ∧ x < 4)) :=
begin
  intros h1 h2,
  split,
  { exact h2 },
  { intro h3,
    have h_false : (∃ (x : ℝ), (2 < x ∧ x < 5) ∧ ¬(3 < x ∧ x < 4)),
    {
      use 2.1,
      split,
      { split,
        { linarith },
        { linarith } },
      { intro h4,
        cases h4 with h51 h52,
        linarith }
    },
    cases h_false with x hx,
    cases hx with h5 h6,
    have h7: (3 < x ∧ x < 4) → false,
    { intro h5,
      exact h6 h5 },
    specialize h3 h5,
    exact h7 h3
  }
end

end necessary_but_not_sufficient_condition_l238_238572


namespace average_marks_first_5_subjects_l238_238620

theorem average_marks_first_5_subjects (average_6_subjects : ℕ) (marks_6th_subject : ℕ)
  (average_6_subjects = 78) (marks_6th_subject = 98) :
  (average_6_subjects * 6 - marks_6th_subject) / 5 = 74 := 
by
  sorry

end average_marks_first_5_subjects_l238_238620


namespace proof_problem_l238_238382

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def g (a b c x : ℝ) : ℝ := f a b c x + 2 * a * |x - 1|

theorem proof_problem (a b c : ℝ) (h1 : a ≠ 0) (h2 : ∀ x : ℝ, 2 * x ≤ f a b c x ∧ f a b c x ≤ 1/2 * (x + 1)^2) :
  (f a b c 1 = 2) ∧ 
  (0 < a ∧ a < 1/2) ∧ 
  (∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), g a (2 - (a + c)) a x ≥ -1 → a = 1/5) := by
  sorry

end proof_problem_l238_238382


namespace next_month_eggs_l238_238495

-- Given conditions definitions
def eggs_left_last_month : ℕ := 27
def eggs_after_buying : ℕ := 58
def eggs_eaten_this_month : ℕ := 48

-- Calculate number of eggs mother buys each month
def eggs_bought_each_month : ℕ := eggs_after_buying - eggs_left_last_month

-- Remaining eggs before next purchase
def eggs_left_before_next_purchase : ℕ := eggs_after_buying - eggs_eaten_this_month

-- Final amount of eggs after mother buys next month's supply
def total_eggs_next_month : ℕ := eggs_left_before_next_purchase + eggs_bought_each_month

-- Prove the total number of eggs next month equals 41
theorem next_month_eggs : total_eggs_next_month = 41 := by
  sorry

end next_month_eggs_l238_238495


namespace range_of_abcd_l238_238126

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then |real.log x / real.log 2|
else (2 / 3) * x^2 - 8 * x + 70 / 3

theorem range_of_abcd 
  (a b c d : ℝ) 
  (ha : a ≠ b) 
  (hb : b ≠ c) 
  (hc : c ≠ d) 
  (hd : d ≠ a) 
  (habcd : f a = f b ∧ f a = f c ∧ f a = f d) :
  32 < a * b * c * d ∧ a * b * c * d < 35 :=
sorry

end range_of_abcd_l238_238126


namespace max_remaining_numbers_l238_238836

/-- 
The board initially has numbers 1, 2, 3, ..., 235.
Among the remaining numbers, no number is divisible by the difference of any two others.
Prove that the maximum number of numbers that could remain on the board is 118.
-/
theorem max_remaining_numbers : 
  ∃ S : set ℕ, (∀ a ∈ S, 1 ≤ a ∧ a ≤ 235) ∧ (∀ a b ∈ S, a ≠ b → ¬ ∃ d, d ∣ (a - b)) ∧ 
  ∃ T : set ℕ, S ⊆ T ∧ T ⊆ finset.range 236 ∧ T.card = 118 := 
sorry

end max_remaining_numbers_l238_238836


namespace max_remaining_numbers_l238_238832

/-- 
The board initially has numbers 1, 2, 3, ..., 235.
Among the remaining numbers, no number is divisible by the difference of any two others.
Prove that the maximum number of numbers that could remain on the board is 118.
-/
theorem max_remaining_numbers : 
  ∃ S : set ℕ, (∀ a ∈ S, 1 ≤ a ∧ a ≤ 235) ∧ (∀ a b ∈ S, a ≠ b → ¬ ∃ d, d ∣ (a - b)) ∧ 
  ∃ T : set ℕ, S ⊆ T ∧ T ⊆ finset.range 236 ∧ T.card = 118 := 
sorry

end max_remaining_numbers_l238_238832


namespace typing_time_l238_238918

theorem typing_time (typing_speed : ℕ) (words_per_page : ℕ) (number_of_pages : ℕ) (h1 : typing_speed = 90) (h2 : words_per_page = 450) (h3 : number_of_pages = 10) : 
  (words_per_page / typing_speed) * number_of_pages = 50 := 
by
  sorry

end typing_time_l238_238918


namespace problem1_problem2_l238_238293

-- Problem 1
theorem problem1 (x y : ℝ) :
  2 * x^2 * y - 3 * x * y + 2 - x^2 * y + 3 * x * y = x^2 * y + 2 :=
by sorry

-- Problem 2
theorem problem2 (m n : ℝ) :
  9 * m^2 - 4 * (2 * m^2 - 3 * m * n + n^2) + 4 * n^2 = m^2 + 12 * m * n :=
by sorry

end problem1_problem2_l238_238293


namespace intersection_eq_union_eq_l238_238001

noncomputable def A := {x : ℝ | -2 < x ∧ x <= 3}
noncomputable def B := {x : ℝ | x < -1 ∨ x > 4}

theorem intersection_eq : A ∩ B = {x : ℝ | -2 < x ∧ x < -1} := by
  sorry

theorem union_eq : A ∪ B = {x : ℝ | x <= 3 ∨ x > 4} := by
  sorry

end intersection_eq_union_eq_l238_238001


namespace max_red_balls_l238_238154

theorem max_red_balls (r w : ℕ) (h1 : r = 3 * w) (h2 : r + w ≤ 50) : r = 36 :=
sorry

end max_red_balls_l238_238154


namespace fraction_of_total_students_l238_238426

variables (G B T : ℕ) (F : ℚ)

-- Given conditions
axiom ratio_boys_to_girls : (7 : ℚ) / 3 = B / G
axiom total_students : T = B + G
axiom fraction_equals_two_thirds_girls : (2 : ℚ) / 3 * G = F * T

-- Proof goal
theorem fraction_of_total_students : F = 1 / 5 :=
by
  sorry

end fraction_of_total_students_l238_238426


namespace odd_f_l238_238364

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2^x else if x < 0 then -x^2 + 2^(-x) else 0

theorem odd_f (x : ℝ) : (f (-x) = -f x) :=
by
  sorry

end odd_f_l238_238364


namespace find_ellipse_equation_find_line_equation_l238_238356

noncomputable def ellipse_C_conditions : Prop :=
  ∃ a b : ℝ, (a > b ∧ b > 0) ∧ (∃ (x y : ℝ), (x / a)^2 + (y / b)^2 = 1) ∧ (2 * b * b = 1)

theorem find_ellipse_equation {a b : ℝ} (h_a_gt_b : a > b) (h_b_gt_0 : b > 0) (h_area : b * b = 1) :
  ∀ {x y : ℝ}, (x / a) ^ 2 + (y / b) ^ 2 = 1 →
    (a = sqrt 2 ∧ b = 1) ∧ (∀ x y, (x ^ 2) / 2 + y ^ 2 = 1) :=
sorry

noncomputable def line_l_conditions : Prop :=
  ∃ k : ℝ, (∃ x1 x2 : ℝ, (1 / x1 + 1 / x2 = 3) ∧ (2 * k^2 / (k^2 - 1) = 3))

theorem find_line_equation (k : ℝ) (h_intersection : (2 * k^2 / (k^2 - 1) = 3)) :
  ∃ k : ℝ, (k = sqrt 3 ∨ k = -sqrt 3) ∧ (∀ x,
    (∃ y, y = sqrt 3 * x - sqrt 3) ∨ (∃ y, y = -sqrt 3 * x + sqrt 3)) :=
sorry

end find_ellipse_equation_find_line_equation_l238_238356


namespace polynomial_root_geq_2019_l238_238387

noncomputable def f (a : Fin 6 → ℝ) (x : ℝ) : ℝ :=
  x^7 + ∑ i in Finset.range 6, a ⟨i, Fin.is_lt _⟩ * x^(i + 1) + 3

def all_roots_real (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = 0 → ∃ a : ℝ, f = λ z, (z + a)^7

theorem polynomial_root_geq_2019 (a : Fin 6 → ℝ)
  (h_nonneg : ∀ i : Fin 6, 0 ≤ a i)
  (h_roots_real : all_roots_real (f a)) : f a 2 ≥ 2019 := 
sorry

end polynomial_root_geq_2019_l238_238387


namespace value_of_expression_l238_238347

theorem value_of_expression (x : ℝ) (h : |x| = x + 2) : 19 * x ^ 99 + 3 * x + 27 = 5 :=
by
  have h1: x ≥ -2 := sorry
  have h2: x = -1 := sorry
  sorry

end value_of_expression_l238_238347


namespace smallest_n_for_divisibility_l238_238174

theorem smallest_n_for_divisibility : ∃ n: ℕ, (n > 0) ∧ (n^2 % 24 = 0) ∧ (n^3 % 864 = 0) ∧ ∀ m : ℕ, 
  (m > 0) ∧ (m^2 % 24 = 0) ∧ (m^3 % 864 = 0) → (12 ≤ m) :=
begin
  sorry
end

end smallest_n_for_divisibility_l238_238174


namespace max_handshakes_l238_238955

theorem max_handshakes (n : ℕ) (h : n = 25) : ∃ k : ℕ, k = 300 :=
by
  use (n * (n - 1)) / 2
  have h_eq : n = 25 := h
  rw h_eq
  sorry

end max_handshakes_l238_238955


namespace find_abcde_l238_238903

theorem find_abcde (N : ℕ) (a b c d e f : ℕ) (h : a ≠ 0) 
(h1 : N % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f)
(h2 : (N^2) % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f) :
    a * 10000 + b * 1000 + c * 100 + d * 10 + e = 48437 :=
by sorry

end find_abcde_l238_238903


namespace value_of_3_star_6_l238_238313

theorem value_of_3_star_6
  (a : ℚ)
  (h₁ : ∀ x y : ℚ, x ☆ y = a^2 * x + a * y + 1)
  (h₂ : 1 ☆ 2 = 3) : 3 ☆ 6 = 7 :=
sorry

end value_of_3_star_6_l238_238313


namespace cost_of_fencing_per_meter_l238_238971

def rectangular_farm_area : Real := 1200
def short_side_length : Real := 30
def total_cost : Real := 1440

theorem cost_of_fencing_per_meter : (total_cost / (short_side_length + (rectangular_farm_area / short_side_length) + Real.sqrt ((rectangular_farm_area / short_side_length)^2 + short_side_length^2))) = 12 :=
by
  sorry

end cost_of_fencing_per_meter_l238_238971


namespace sum_first_60_natural_numbers_l238_238292

theorem sum_first_60_natural_numbers : (∑ i in Finset.range 61, i) = 1830 :=
by
  sorry

end sum_first_60_natural_numbers_l238_238292


namespace distance_from_point_to_line_is_two_l238_238878

-- Definitions for the conditions
structure Point (α : Type*) :=
  (x : α)
  (y : α)

structure Line (α : Type*) :=
  (A : α)
  (B : α)
  (C : α)

-- Define the point and the line based on the conditions
noncomputable def P : Point ℝ := { x := 3, y := 0 }
noncomputable def l : Line ℝ := { A := 3, B := 4, C := 1 }

-- Function to compute the point-to-line distance
noncomputable def point_to_line_distance (P : Point ℝ) (l : Line ℝ) : ℝ :=
  abs (l.A * P.x + l.B * P.y + l.C) / sqrt (l.A^2 + l.B^2)

-- The proof statement
theorem distance_from_point_to_line_is_two : point_to_line_distance P l = 2 :=
by {
  sorry,
}

end distance_from_point_to_line_is_two_l238_238878


namespace sue_votes_correct_l238_238138

def total_votes : ℕ := 1000
def percentage_others : ℝ := 0.65
def sue_votes : ℕ := 350

theorem sue_votes_correct :
  sue_votes = (total_votes : ℝ) * (1 - percentage_others) :=
by
  sorry

end sue_votes_correct_l238_238138


namespace systematic_sampling_first_group_l238_238547

theorem systematic_sampling_first_group (S : ℕ) (n : ℕ) (students_per_group : ℕ) (group_number : ℕ)
(h1 : n = 160)
(h2 : students_per_group = 8)
(h3 : group_number = 16)
(h4 : S + (group_number - 1) * students_per_group = 126)
: S = 6 := by
  sorry

end systematic_sampling_first_group_l238_238547


namespace value_of_k_h_10_l238_238407

def h (x : ℝ) : ℝ := 4 * x - 5
def k (x : ℝ) : ℝ := 2 * x + 6

theorem value_of_k_h_10 : k (h 10) = 76 := by
  -- We provide only the statement as required, skipping the proof
  sorry

end value_of_k_h_10_l238_238407


namespace range_of_f_l238_238752

noncomputable def f : ℝ → ℝ := sorry

lemma problem_conditions (x : ℝ) (hx : x > 0) : 
  f x > 0 ∧ f x < deriv f x ∧ deriv f x < 3 * f x :=
sorry

theorem range_of_f : 
  1 / real.exp 6 < f 1 / f 3 ∧ f 1 / f 3 < 1 / real.exp 2 :=
sorry

end range_of_f_l238_238752


namespace circle_area_radius_one_l238_238290

noncomputable def integral_circle_area : ℝ := ∫ x in -1..1, real.sqrt (1 - x^2)

theorem circle_area_radius_one : integral_circle_area * 2 = real.pi :=
by 
  -- The definition and the proof are separated.
  -- integral_circle_area is the area of a semicircle of radius 1.
  -- When multiplied by 2, we get the area of the full circle.
  sorry

end circle_area_radius_one_l238_238290


namespace max_remained_numbers_l238_238848

theorem max_remained_numbers (S : Finset ℕ) (hSubset : S ⊆ Finset.range 236)
  (hCondition : ∀ a b c ∈ S, a ≠ b → a ≠ c → b ≠ c → ¬(b - a ∣ c)) : S.card ≤ 118 := 
sorry

end max_remained_numbers_l238_238848


namespace sum_digits_x_l238_238030

/-- 
Sophia took x U.S. dollars and exchanged them at a rate of 8 Canadian dollars for every 5 U.S.
dollars. After spending 80 Canadian dollars, she had x Canadian dollars left. Prove that the sum of 
the digits of x is 7.
-/
theorem sum_digits_x (x : ℤ) (h : (8 * x / 5) - 80 = x) : x.digits.sum = 7 :=
by
  -- This is a placeholder for the formal proof
  sorry

end sum_digits_x_l238_238030


namespace multimedia_sets_max_profit_l238_238581

-- Definitions of conditions:
def cost_A : ℝ := 3
def cost_B : ℝ := 2.4
def price_A : ℝ := 3.3
def price_B : ℝ := 2.8
def total_sets : ℕ := 50
def total_cost : ℝ := 132
def min_m : ℕ := 11

-- Problem 1: Prove the number of sets based on equations
theorem multimedia_sets (x y : ℕ) (h1 : x + y = total_sets) (h2 : cost_A * x + cost_B * y = total_cost) :
  x = 20 ∧ y = 30 :=
by sorry

-- Problem 2: Prove the maximum profit within a given range
theorem max_profit (m : ℕ) (h_m : 10 < m ∧ m < 20) :
  (-(0.1 : ℝ) * m + 20 = 18.9) ↔ m = min_m :=
by sorry

end multimedia_sets_max_profit_l238_238581


namespace shortest_distance_between_circles_is_zero_l238_238929

open Real

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop :=
  x^2 - 12 * x + y^2 - 8 * y - 12 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop :=
  x^2 + 10 * x + y^2 - 10 * y + 34 = 0

-- Statement of the proof problem: 
-- Prove the shortest distance between the two circles defined by circle1 and circle2 is 0.
theorem shortest_distance_between_circles_is_zero :
    ∀ (x1 y1 x2 y2 : ℝ),
      circle1 x1 y1 →
      circle2 x2 y2 →
      0 = 0 :=
by
  intros x1 y1 x2 y2 h1 h2
  sorry

end shortest_distance_between_circles_is_zero_l238_238929


namespace complex_fraction_simplifies_l238_238511

-- Define the given complex numbers
def num : ℂ := 3 + I
def denom : ℂ := 1 + I
def expr := num / denom
def expected : ℂ := 2 - I

-- Formally restate and prove the equality
theorem complex_fraction_simplifies :
  expr = expected :=
by
  sorry

end complex_fraction_simplifies_l238_238511


namespace solve_for_y_l238_238213

theorem solve_for_y (y : ℚ) : 
  y + 5 / 8 = 2 / 9 + 1 / 2 → 
  y = 7 / 72 := 
by 
  intro h1
  sorry

end solve_for_y_l238_238213


namespace maximum_numbers_no_divisible_difference_l238_238822

theorem maximum_numbers_no_divisible_difference :
  ∃ (S : Finset ℕ), S ⊆ Finset.range 236 ∧ 
  (∀ a b c ∈ S, a ≠ b → b ≠ c → c ≠ a → ¬ (a - b = 0) ∨ ¬ (c ∣ (a - b))) ∧ S.card ≤ 118 :=
by
  sorry

end maximum_numbers_no_divisible_difference_l238_238822


namespace triangle_AB_length_l238_238421

theorem triangle_AB_length
  {A B C : Type*}
  (angle_A : ∠A = 90)
  (tan_B : Real := 5 / 12)
  (AC : Real := 52) :
  AB = 48 := 
sorry

end triangle_AB_length_l238_238421


namespace find_g_of_5_l238_238519

theorem find_g_of_5 (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, g (x * y) = g x * g y) 
  (h2 : g 1 = 2) : 
  g 5 = 32 := 
by 
  sorry

end find_g_of_5_l238_238519


namespace max_remained_numbers_l238_238847

theorem max_remained_numbers (S : Finset ℕ) (hSubset : S ⊆ Finset.range 236)
  (hCondition : ∀ a b c ∈ S, a ≠ b → a ≠ c → b ≠ c → ¬(b - a ∣ c)) : S.card ≤ 118 := 
sorry

end max_remained_numbers_l238_238847


namespace find_point_P_l238_238473

-- Given points A, B, C, D, E
def A : ℝ × ℝ × ℝ := (10, 0, 0)
def B : ℝ × ℝ × ℝ := (0, -6, 0)
def C : ℝ × ℝ × ℝ := (0, 0, 8)
def D : ℝ × ℝ × ℝ := (0, 0, 0)
def E : ℝ × ℝ × ℝ := (5, -3, 4)

noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

theorem find_point_P :
  ∃ P : ℝ × ℝ × ℝ, distance A P = distance D P ∧
                    distance B P = distance D P ∧
                    distance C P = distance D P ∧
                    distance E P = distance D P ∧
                    P = (5, -3, 4) :=
by
  sorry

end find_point_P_l238_238473


namespace find_height_of_triangle_ABC_dropped_from_B_l238_238446

-- Definitions based on the given conditions
variables {A B C K L Q : Type*}
variables {AK KC AL LB : ℝ}
variables {area_AQC : ℝ}
-- Conditions from the problem
variables (H₀ : AK = 1)
variables (H₁ : KC = 3)
variables (H₂ : AL / LB = 2 / 3)
variables (H₃ : ∀ (A B C K L Q : Type*), is_triangle A B C → is_on_line K A C → is_on_line L A B → is_intersection Q (line_through B K) (line_through C L))
variables (H₄ : area_of_triangle A Q C = 1)

-- Stating the problem
theorem find_height_of_triangle_ABC_dropped_from_B
  (H₀ : AK = 1)
  (H₁ : KC = 3)
  (H₂ : AL / LB = 2 / 3)
  (H₃ : ∀ (A B C K L Q : Type*), is_triangle A B C → is_on_line K A C → is_on_line L A B → 
    is_intersection Q (line_through B K) (line_through C L))
  (H₄ : area_of_triangle A Q C = 1) :
  height_of_triangle_dropped_from B A B C = 1.5 :=
sorry

end find_height_of_triangle_ABC_dropped_from_B_l238_238446


namespace trapezoid_area_correct_l238_238462

open_locale big_operators

noncomputable def trapezoid_area (a c : ℝ) : ℝ :=
  (a + c) / 2 * real.sqrt ((a^2 + 6 * a * c - 7 * c^2) / 12)

theorem trapezoid_area_correct (a c : ℝ) :
  is_trapezoid ABCD a c → midpoints AD E BC F → perp_projection E G BC →
  trisection_point G F C →
  area_of_trapezoid ABCD = (a + c) / 2 * real.sqrt ((a^2 + 6 * a * c - 7 * c^2) / 12) :=
begin
  -- Use the given structure and definitions to prove the area
  sorry
end

end trapezoid_area_correct_l238_238462


namespace minimum_possible_deaths_l238_238910

-- Definition of the problem
def gangster_shooting_problem : Prop :=
  ∃ (gangsters : Finset (ℝ × ℝ)) (positions : Finset ℝ),
    gangsters.card = 50 ∧
    (∀ g ∈ gangsters, ∃ closest ∈ gangsters, closest ≠ g ∧ is_closest (gangster_positions g) (gangster_positions closest)) ∧
    minimum_deaths gangsters = 10

-- Function to check if one position is closest to another
def is_closest (pos1 pos2 : (ℝ × ℝ)) : Prop :=
  ∀ (g ∈ gangsters), distance pos1 g ≤ distance pos2 g

-- Pseudo-definition for calculating minimum deaths (to be defined properl):
def minimum_deaths (gangsters : Finset (ℝ × ℝ)) : ℕ :=
  -- This function would include the logic to determine the minimum possible deaths
  sorry

-- The main statement to be proved
theorem minimum_possible_deaths :
  gangster_shooting_problem :=
by
  sorry

end minimum_possible_deaths_l238_238910


namespace cost_for_five_dozen_l238_238282

def price_per_dozen (total_price : ℝ) (total_dozen : ℝ) : ℝ :=
  total_price / total_dozen

def cost_of_apples (price_per_dozen : ℝ) (num_dozen : ℝ) : ℝ :=
  price_per_dozen * num_dozen

theorem cost_for_five_dozen (price_3_dozen : ℝ) (num_dozen_3 : ℝ) (price_5_dozen : ℝ) : 
  price_3_dozen = 23.40 → num_dozen_3 = 3 → price_5_dozen = 5 * (price_3_dozen / num_dozen_3) → price_5_dozen = 39 :=
by
  sorry

end cost_for_five_dozen_l238_238282


namespace maximum_numbers_no_divisible_difference_l238_238821

theorem maximum_numbers_no_divisible_difference :
  ∃ (S : Finset ℕ), S ⊆ Finset.range 236 ∧ 
  (∀ a b c ∈ S, a ≠ b → b ≠ c → c ≠ a → ¬ (a - b = 0) ∨ ¬ (c ∣ (a - b))) ∧ S.card ≤ 118 :=
by
  sorry

end maximum_numbers_no_divisible_difference_l238_238821


namespace players_taking_chemistry_l238_238621

open Finset

variable {U : Type} [DecidableEq U]
variable (players : Finset U) (biology chemistry physics : Finset U)

-- Given Conditions:
-- 30 players taking at least one of biology, chemistry, or physics
axiom h_total : players.card = 30

-- 14 players taking biology
axiom h_bio : biology.card = 14

-- 6 players taking both biology and chemistry
axiom h_bio_chem : (biology ∩ chemistry).card = 6

-- 10 players taking all three subjects
axiom h_all_three : (biology ∩ chemistry ∩ physics).card = 10

-- Define the statement to prove
theorem players_taking_chemistry : (chemistry.card = 26) :=
by
  sorry

end players_taking_chemistry_l238_238621


namespace billy_sleep_total_l238_238742

theorem billy_sleep_total
  (h₁ : ∀ n : ℕ, n = 1 → ∃ h : ℕ, h = 6)
  (h₂ : ∀ n : ℕ, n = 2 → ∃ h : ℕ, h = (6 + 2))
  (h₃ : ∀ n : ℕ, n = 3 → ∃ h : ℕ, h = ((6 + 2) / 2))
  (h₄ : ∀ n : ℕ, n = 4 → ∃ h : ℕ, h = (((6 + 2) / 2) * 3)) :
  ∑ n in {1, 2, 3, 4}, (classical.some (h₁ n 1) + classical.some (h₂ n 2) + classical.some (h₃ n 3) + classical.some (h₄ n 4)) = 30 :=
by sorry

end billy_sleep_total_l238_238742


namespace domain_of_f_l238_238315

theorem domain_of_f (c : ℝ) :
  (∀ x : ℝ, -7 * x^2 + 5 * x + c ≠ 0) ↔ c < -25 / 28 :=
by
  sorry

end domain_of_f_l238_238315


namespace parabola_equation_standard_slope_angle_range_l238_238707

noncomputable def parabola_equation (p : ℝ) : Prop :=
  ∀ (x y : ℝ), y^2 = 2 * p * x

theorem parabola_equation_standard :
  parabola_equation 2 ↔ ∀ (x y : ℝ), y^2 = 4 * x :=
sorry

noncomputable def within_parabola_range (C : ℝ → ℝ → Prop) (l : ℝ → ℝ): Prop :=
  ∃ (α : ℝ), ∀ (x y : ℝ), C (cos α * x - sin α * y) (sin α * x + cos α * y)

theorem slope_angle_range (C : ℝ → ℝ → Prop) (l : ℝ → ℝ) :
  (C = parabola_equation 2) →
  (l = λ x, (x - 1) * tan α) →
  (exists α, ∀ (AB : ℝ), |AB| ≤ 8 ∧ (∃ (x y : ℝ), 3 * x^2 + 2 * y^2 = 2) →
    (π / 4 ≤ α ∧ α ≤ π / 3) ∨ (2 * π / 3 ≤ α ∧ α ≤ 3 * π / 4)) :=
sorry

end parabola_equation_standard_slope_angle_range_l238_238707


namespace max_remaining_numbers_l238_238841

theorem max_remaining_numbers : 
  ∃ s : Finset ℕ, s ⊆ (Finset.range 236) ∧ (∀ x y ∈ s, x ≠ y → ¬ (x - y).abs ∣ x) ∧ s.card = 118 := 
by
  sorry

end max_remaining_numbers_l238_238841


namespace salt_not_heavier_than_cotton_l238_238246

-- Definitions based on conditions
def mass_salt : ℝ := 1 -- 1 kilogram of salt
def mass_cotton : ℝ := 1 -- 1 kilogram of cotton

-- Theorem stating the question and the expected false answer
theorem salt_not_heavier_than_cotton : mass_salt = mass_cotton → ¬ (mass_salt > mass_cotton) :=
by
  intro h
  rw h
  apply not_lt_self
  sorry

end salt_not_heavier_than_cotton_l238_238246


namespace problem_statement_l238_238774

namespace TriangleProof

noncomputable def value_of_a (c : ℝ) (C : ℝ) (condition : ℝ → ℝ → Prop) : ℝ :=
  if h : C = (30 : ℝ) * (Real.pi / 180) ∧ c = 2 ∧ ∃ a b : ℝ, condition a b
  then 4 -- from the problem's solution
  else 0 -- default

noncomputable def area_of_triangle (c : ℝ) (C : ℝ) (condition : ℝ → ℝ → Prop) : ℝ :=
  if h : C = (30 : ℝ) * (Real.pi / 180) ∧ c = 2 ∧ ∃ a b : ℝ, condition a b
  then 2 * Real.sqrt 3 -- from the problem's solution
  else 0 -- default

theorem problem_statement :
  ∀ (c : ℝ) (C : ℝ)
  (condition : ℝ → ℝ → Prop)
  (ha : value_of_a c C condition = 4)
  (hs : area_of_triangle c C condition = 2 * Real.sqrt 3),
  true :=
by
  intro c C condition ha hs
  sorry

end TriangleProof

end problem_statement_l238_238774


namespace minimize_quadratic_function_l238_238182

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end minimize_quadratic_function_l238_238182


namespace angle_equality_l238_238275

noncomputable def triangle := sorry   -- Placeholder, as setting up basic geometric objects would be required.

variables (A B C D E F : Point)
variables (hD : is_midpoint D B C)
variables (hE : is_midpoint E C A)
variables (hF : is_midpoint F A B)

theorem angle_equality (h_triangle : triangle A B C) :
  ∠DAC = ∠ABE ↔ ∠AFC = ∠ADB := sorry

end angle_equality_l238_238275


namespace ice_cream_sundaes_l238_238984

theorem ice_cream_sundaes (flavors : Finset String) (vanilla : String)
  (h_len : flavors.card = 8) (h_vanilla : vanilla ∈ flavors) :
  ∃ sundae_choices, ∀ s ∈ sundae_choices, s ⊆ flavors ∧ vanilla ∈ s ∧ s.card = 2 ∧ sundae_choices.card = 7 :=
by
  sorry

end ice_cream_sundaes_l238_238984


namespace limit_of_expression_l238_238233

noncomputable def limit_expression (a x : ℝ) := (2 - x / a) ^ tan (π * x / (2 * a))

theorem limit_of_expression (a : ℝ) (h : 0 < a) :
  Filter.Tendsto (λ x, limit_expression a x) (nhds a) (nhds (Real.exp (2 / π))) :=
sorry

end limit_of_expression_l238_238233


namespace initial_time_l238_238959

-- Define the conditions
def distance : ℝ := 180
def speed : ℝ := 20

-- Define the time to cover the distance initially as T and the new time
def T : ℝ := 13.5
def new_time (T : ℝ) : ℝ := (2 / 3) * T

-- Proof statement converting the conditions and question to equivalent Lean 4 problem
theorem initial_time (T : ℝ) (new_time : ℝ) (distance : ℝ) (speed : ℝ) :
  new_time = distance / speed → (new_time = (2 / 3) * T) → T = 13.5 :=
by
  intros h1 h2
  sorry

end initial_time_l238_238959


namespace triangle_cos_ratio_sin_l238_238459

theorem triangle_cos_ratio_sin :
  (∃ (A B C : ℝ) (s : ℕ) (t : ℚ),
    ∠A + ∠B + ∠C = 180 ∧
    cos ∠A = cos ∠B ∧
    cos ∠C = 2 * cos ∠A ∧
    sin ∠A = sqrt[s] t ∧
    (s : ℝ) + (t : ℝ) = 19/4) := sorry

end triangle_cos_ratio_sin_l238_238459


namespace product_of_two_greatest_non_achievable_scores_l238_238508

theorem product_of_two_greatest_non_achievable_scores :
  let non_achievable_scores := [31, 39] in
  let product := non_achievable_scores.product in
  product = 1209 :=
begin
  sorry
end

end product_of_two_greatest_non_achievable_scores_l238_238508


namespace max_rabbits_l238_238111

theorem max_rabbits (N : ℕ) (h1 : ∀ k, k = N → k = 27 → true)
    (h2 : ∀ n_l : ℕ, n_l = 13 → n_l <= N)
    (h3 : ∀ n_j : ℕ, n_j = 17 → n_j <= N)
    (h4 : ∀ n_both : ℕ, n_both >= 3 → true) :
  N <= 27 :=
begin
  sorry
end

end max_rabbits_l238_238111


namespace b_n_general_term_a_n_sum_l238_238723

-- Definitions of the sequences with given conditions
def a (n : ℕ) : ℕ := sorry -- The sequence {a_n}
def b (n : ℕ) : ℕ := sorry -- The sequence {b_n}

-- Conditions
axiom a_b_condition1 (n : ℕ) : a (n + 1) + 1 = 2 * a n + n
axiom a_b_condition2 (n : ℕ) : b n - a n = n
axiom b_1 : b 1 = 2

-- The first proof problem: General term of the geometric sequence {b_n}
theorem b_n_general_term : ∀ (n : ℕ), b n = 2 ^ n := 
by sorry

-- The second proof problem: Sum of the first n terms of the sequence {a_n}
def S (n : ℕ) : ℕ := (2 ^ (n + 1)) - 2 - (n * n + n) / 2

theorem a_n_sum : ∑ i in finset.range n, a i = S n :=
by sorry

end b_n_general_term_a_n_sum_l238_238723


namespace simplify_expression_correct_l238_238050

def simplify_expression : ℚ :=
  (5^5 + 5^3) / (5^4 - 5^2)

theorem simplify_expression_correct : simplify_expression = 65 / 12 :=
  sorry

end simplify_expression_correct_l238_238050


namespace cards_ordering_correct_l238_238950

noncomputable def initial_deck := [
   "king", "three", "nine", "six", "queen", "two", "four", "eight", 
   "seven", "ace", "ten", "jack", "five"
]

def word_len (card : String) : Nat :=
  card.length

def move_cards (deck : List String) (n : Nat) : List String :=
  deck.drop n ++ deck.take n

def lay_card (deck : List String) : List String × List String :=
  let n := word_len (deck.head!)
  let deck' := move_cards deck n
  (deck'.tail!, deck'.head!::[])

def lay_all_cards (deck : List String) (result : List String) : List String :=
  if deck.isEmpty then result else
    let (deck', card) := lay_card deck
    lay_all_cards deck' (result ++ card)

def expected_deck := [
   "three", "three", "five hearts", "ace diamonds", "ten diamonds",
   "ten spades", "king clubs", "two diamonds", "king spades", "jack hearts", "five clubs",
   "three diamonds", "jack diamonds", "six hearts", "jack clubs", "four spades", "eight spades", 
   "queen diamonds", "four diamonds", "queen hearts", "seven hearts", "ten clubs", "jack spades", 
   "five diamonds", "ace clubs", "five spades", "king diamonds", "seven clubs", "eight clubs", 
   "six diamonds", "eight hearts", "ace hearts", "king hearts", "four clubs", "seven diamonds", 
   "nine spades", "two hearts", "queen spades", "ace spades", "six spades", "three hearts", 
   "eight diamonds", "nine hearts", "two clubs", "queen clubs", "two spades", 
   "six clubs", "nine clubs", "nine diamonds", "four hearts", "seven spades", "ten hearts"
]

theorem cards_ordering_correct :
  lay_all_cards initial_deck [] = expected_deck :=
by
  sorry -- Proof to be filled in later

-- This Lean 4 code should compile successfully and correctly represents the math problem.

end cards_ordering_correct_l238_238950


namespace logarithm_expression_l238_238574

theorem logarithm_expression :
  2 * log 5 10 + log 5 (1 / 4) + 2 ^ (log 4 3) = 2 + Real.sqrt 3 :=
by
  sorry

end logarithm_expression_l238_238574


namespace coefficients_zero_l238_238901

noncomputable def p (z : ℂ) (a b : ℂ) : ℂ := z^2 + a * z + b

theorem coefficients_zero (a b : ℂ) (h : ∀ z : ℂ, abs z = 1 → abs (p z a b) = 1) : a = 0 ∧ b = 0 :=
by
  sorry

end coefficients_zero_l238_238901


namespace billy_sleep_total_l238_238743

theorem billy_sleep_total
  (h₁ : ∀ n : ℕ, n = 1 → ∃ h : ℕ, h = 6)
  (h₂ : ∀ n : ℕ, n = 2 → ∃ h : ℕ, h = (6 + 2))
  (h₃ : ∀ n : ℕ, n = 3 → ∃ h : ℕ, h = ((6 + 2) / 2))
  (h₄ : ∀ n : ℕ, n = 4 → ∃ h : ℕ, h = (((6 + 2) / 2) * 3)) :
  ∑ n in {1, 2, 3, 4}, (classical.some (h₁ n 1) + classical.some (h₂ n 2) + classical.some (h₃ n 3) + classical.some (h₄ n 4)) = 30 :=
by sorry

end billy_sleep_total_l238_238743


namespace maximum_value_N_27_l238_238103

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end maximum_value_N_27_l238_238103


namespace inverse_subtraction_eq_neg_five_thirds_l238_238644

theorem inverse_subtraction_eq_neg_five_thirds:
  ((5⁻¹ - 2⁻¹)⁻¹ = -10 / 3) := 
by 
  -- Rewriting given conditions
  have h1 : 5⁻¹ = 1 / 5 := by norm_num,
  have h2 : 2⁻¹ = 1 / 2 := by norm_num,
  -- Calculate the difference
  calc
    (5⁻¹ - 2⁻¹)⁻¹ 
    = (1 / 5 - 1 / 2)⁻¹ : by rw [h1, h2]
    = (2 / 10 - 5 / 10)⁻¹ : by norm_num
    = ((2 - 5) / 10)⁻¹ : by rw [sub_div, one_div]
    = (-3 / 10)⁻¹ : by norm_num
    = -10 / 3 : by norm_num


end inverse_subtraction_eq_neg_five_thirds_l238_238644


namespace log_cos_x_l238_238751

variable (b : ℝ) (x : ℝ) (y : ℝ) (a : ℝ)

theorem log_cos_x (h₁ : 1 < b) 
                 (h₂ : sin x ^ 2 = 2 * sin y * cos y)
                 (h₃ : log b (sin x) = a) : 
                 log b (cos x) = 1 / 2 * log b (1 - b^(2 * a)) :=
by 
s sorry

end log_cos_x_l238_238751


namespace vendor_apples_sold_second_day_l238_238603

/-- The vendor's sales and waste calculations as described, leading to the conclusion that he sells 50% of the remaining apples on the second day. -/
theorem vendor_apples_sold_second_day (n : ℕ) (h_nonzero : n ≠ 0) :
  let sold_first_day := n * 30 / 100,
      remaining_after_first_sale := n - sold_first_day,
      thrown_first_day := remaining_after_first_sale * 20 / 100,
      remaining_after_first_day := remaining_after_first_sale - thrown_first_day,
      total_thrown := n * 42 / 100,
      thrown_second_day := total_thrown - thrown_first_day,
      sold_second_day_percent := (remaining_after_first_day - thrown_second_day) * 100 / remaining_after_first_day
  in sold_second_day_percent = 50 :=
by
  sorry

end vendor_apples_sold_second_day_l238_238603


namespace max_rabbits_with_long_ears_and_jumping_far_l238_238091

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l238_238091


namespace minimize_quadratic_function_l238_238179

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end minimize_quadratic_function_l238_238179


namespace maximum_rabbits_condition_l238_238107

-- Define the conditions and constraints
variables {N : ℕ}
variables (total_rabbits long_ears jump_far : ℕ)
variables (at_least_three_with_both : Prop)

-- State the conditions with exact values and assumptions
def conditions := 
  total_rabbits = N ∧
  long_ears = 13 ∧
  jump_far = 17 ∧
  at_least_three_with_both = (∃ a b c : ℕ, a >= 3 ∧ b = (long_ears - a) ∧ c = (jump_far - a))

-- State the theorem to be proved
theorem maximum_rabbits_condition :
  ∀ {N : ℕ}, conditions N long_ears jump_far at_least_three_with_both → N ≤ 27 :=
by sorry

end maximum_rabbits_condition_l238_238107


namespace lee_kim_tied_september_l238_238072

-- Define home runs per month for Lee and Kim
def lee_hrs : List Nat := [1, 9, 14, 12, 9, 12, 15, 5]
def kim_hrs : List Nat := [3, 8, 18, 11, 14, 16, 7, 7]

-- Define cumulative home runs calculation
def cumulative (hrs : List Nat) : List Nat :=
  hrs.foldl (λ acc hr, acc ++ [hr + (acc.reverse.head!.getD 0)]) []

-- Define cumulative home runs for Lee and Kim
def lee_cumulative : List Nat := cumulative lee_hrs
def kim_cumulative : List Nat := cumulative kim_hrs

-- Statement that Lee and Kim had the same number of home runs by the end of September
theorem lee_kim_tied_september : (lee_cumulative.getD 6 0) = (kim_cumulative.getD 6 0) :=
by sorry

end lee_kim_tied_september_l238_238072


namespace always_possible_226_blues_exists_226_blues_no_matter_what_l238_238149

-- Definition of the problem's conditions
def num_points := 2028
def num_values := 676
def num_appearances := 3

-- Condition: Points are numbered from 1 to 676, each appearing exactly 3 times.
axiom points_on_circle : Fin num_points → Fin (num_values + 1)
axiom each_value_appears_three_times : ∀ (n : Fin (num_values + 1)), (points_on_circle n).val = 3

-- Number of triangles formed
def num_triangles := 676

-- Statement 1: Regardless of assignment, always possible to form the 676 triangles such that exactly 226 numbers are colored blue.
theorem always_possible_226_blues (assignment: Fin num_points → Fin num_values): 
  ∃ (triangles : Fin num_triangles → Fin 3 → Fin num_points), 
  (∀ (t1 t2 : Fin num_triangles) (i j : Fin 3), t1 ≠ t2 → triangles t1 i ≠ triangles t2 j) ∧
  (∃ (colored_blue : Fin num_points → Prop), 
    (∀ (t : Fin num_triangles), 
      ∃ (v : Fin num_points), colored_blue v ∧ (v = max (triangles t 0) (triangles t 1) (triangles t 2))
    ) ∧
    (card { v | colored_blue v } = 226))
:= sorry

-- Statement 2: There exists an assignment such that no matter how the triangles are selected, exactly 226 numbers are colored blue.
theorem exists_226_blues_no_matter_what (exists_assignment: ∃ (assignment: Fin num_points → Fin num_values), 
  ∀ (triangles: Fin num_triangles → Fin 3 → Fin num_points), 
  (∀ (t1 t2 : Fin num_triangles) (i j : Fin 3), t1 ≠ t2 → triangles t1 i ≠ triangles t2 j) →
  (∃ (colored_blue: Fin num_points → Prop),
    (∀ (t : Fin num_triangles), 
      ∃ (v : Fin num_points), colored_blue v ∧ (v = max (triangles t 0) (triangles t 1) (triangles t 2))
    ) ∧
    (card { v | colored_blue v } = 226))
)
:= sorry

end always_possible_226_blues_exists_226_blues_no_matter_what_l238_238149


namespace eq_of_neg_reciprocal_l238_238756

variable (a b : ℚ) -- Assuming a and b are rational numbers for simplicity

-- Conditions as hypotheses
hypothesis h1 : -(-1 / a) = 8
hypothesis h2 : -(1 / -b) = 8

-- Theorem stating the equivalence of a and b given the conditions
theorem eq_of_neg_reciprocal (h1 : -(-1 / a) = 8) (h2 : -(1 / -b) = 8) : a = b := by
  sorry

end eq_of_neg_reciprocal_l238_238756


namespace proof_problem_l238_238698

-- Definitions of sets A and B
def A := { x : ℝ | -1 < x ∧ x < 3 }
def B := { x : ℝ | -2 < x ∧ x ≤ 2 }

-- Definition of the complement of set A
def A_complement := { x : ℝ | x ≤ -1 ∨ x ≥ 3 }

-- The intersection of the complement of A and B
def A_complement_inter_B := { x : ℝ | -2 < x ∧ x ≤ -1 }

-- Function definition
def f (x : ℝ) := 2 ^ (2 - x)

-- Statement of the theorem to be proven
theorem proof_problem :
  (A_complement ∩ B) = A_complement_inter_B ∧
  (∀ x, -2 < x ∧ x ≤ -1 → f(x) ∈ set.Ico 8 16) :=
by
  sorry

end proof_problem_l238_238698


namespace cube_volume_l238_238900

/-- Given the perimeter of one face of a cube, proving the volume of the cube -/

theorem cube_volume (h : ∀ (s : ℝ), 4 * s = 28) : (∃ (v : ℝ), v = (7 : ℝ) ^ 3) :=
by
  sorry

end cube_volume_l238_238900


namespace triangle_is_right_l238_238693

variables (m n : ℝ) (x y : ℝ) (F1 F2 P : ℝ × ℝ) (c: ℝ)
hypothesis (h1 : m > 1)
hypothesis (h2 : n = m - 2)
hypothesis (h3 : ∀ (x y : ℝ), x^2 / m + y^2 = 1)
hypothesis (h4 : ∀ (x y : ℝ), x^2 / n - y^2 = 1)
hypothesis (h5 : dist F1 F2 = 2 * real.sqrt (m - 1))
hypothesis (h6 : dist P F1 - dist P F2 = 2 * real.sqrt n)
hypothesis (h7 : dist P F1 + dist P F2 = 2 * real.sqrt m)

theorem triangle_is_right : 
  dist P F1 ^ 2 + dist P F2 ^ 2 = dist F1 F2 ^ 2 :=
sorry

end triangle_is_right_l238_238693


namespace min_lambda_exists_l238_238720

noncomputable def f (x : ℝ) : ℝ := 3 * x - 2

theorem min_lambda_exists :
  ∃ (λ : ℝ), λ = Real.sqrt 2 ∧ ∀ (θ : ℝ), 0 < θ ∧ θ ≤ Real.pi / 2 →
  f (Real.cos θ ^ 2 + λ * Real.sin θ - 1) + 1 / 2 ≥ 0 :=
by
  sorry

end min_lambda_exists_l238_238720


namespace match_black_piece_shape_with_option_A_l238_238267

structure WoodenPrism where
  black_pieces : Set ℕ
  white_pieces : Set ℕ

variable {prism : WoodenPrism}

-- Define the condition that each color must form a shape with four cubes.
def four_cubes_connected (pieces : Set ℕ) : Prop :=
  pieces.card = 4 -- and other conditions ensuring connectedness (omitted here for brevity)

-- Define the condition that one cube from each color is hidden.
def one_cube_hidden (visible : Set ℕ) : Prop :=
  visible.card = 3

-- Define that visible black piece forms part of a larger connected shape.
def is_tetris_like (pieces : Set ℕ) : Prop :=
  -- Placeholder for actual Tetris-like shape condition (omitted here for brevity)
  sorry

-- Main theorem to prove option A matches the black piece shape
theorem match_black_piece_shape_with_option_A
  (prism : WoodenPrism)
  (visible_black : Set ℕ) :
  four_cubes_connected prism.black_pieces →
  one_cube_hidden visible_black →
  is_tetris_like prism.black_pieces →
  -- Options as sets representing positions:
  let option_A : Set ℕ := {1, 2, 3, 4} in -- Hypothetical positions
    prism.black_pieces = option_A :=
  sorry

end match_black_piece_shape_with_option_A_l238_238267


namespace hyperbola_eccentricity_l238_238385

noncomputable def eccentricity_of_hyperbola (a b c : ℝ) (ha : a > 0) (hb : b > 0)
  (h_latus_rectum : ∃ p : ℝ, p = 2 * c)
  (h_intersection : ∃ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ∧ (y^2 = 2 * p * x) ∧ (x = c)) :
  Real := 
  sqrt 2 + 1

theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0)
  (h_latus_rectum : ∃ p : ℝ, p = 2 * c)
  (h_intersection : ∃ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ∧ (y^2 = 2 * p * x) ∧ (x = c)) :
  eccentricity_of_hyperbola a b c ha hb h_latus_rectum h_intersection = sqrt 2 + 1 :=
sorry

end hyperbola_eccentricity_l238_238385


namespace quadratic_inequality_l238_238697

theorem quadratic_inequality (a : ℝ) :
  (¬ (∃ x : ℝ, a * x^2 + 2 * x + 3 ≤ 0)) ↔ (a > 1 / 3) :=
by 
  sorry

end quadratic_inequality_l238_238697


namespace largest_possible_value_l238_238472

/-- Let x, y, z be integers such that:
B = 1/7 * matrix (-5 x, y z)
and B^2 = I (the identity matrix), 
prove the largest possible value of x + y + z is 30. -/
theorem largest_possible_value (x y z : ℤ) 
  (B : Matrix (Fin 2) (Fin 2) ℚ)
  (h1 : B = (1/7 : ℚ) • Matrix ([[ -5, x], [y, z ]]))
  (h2 : B * B = 1) : x + y + z ≤ 30 :=
begin
  sorry
end

end largest_possible_value_l238_238472


namespace cost_prices_max_units_B_possible_scenarios_l238_238582

-- Part 1: Prove cost prices of Product A and B
theorem cost_prices (x : ℝ) (A B : ℝ) 
  (h₁ : B = x ∧ A = x - 2) 
  (h₂ : 80 / A = 100 / B) 
  : B = 10 ∧ A = 8 :=
by 
  sorry

-- Part 2: Prove maximum units of product B that can be purchased
theorem max_units_B (y : ℕ) 
  (h₁ : ∀ y : ℕ, 3 * y - 5 + y ≤ 95) 
  : y ≤ 25 :=
by 
  sorry

-- Part 3: Prove possible scenarios for purchasing products A and B
theorem possible_scenarios (y : ℕ) 
  (h₁ : y > 23 * 9/17 ∧ y ≤ 25) 
  : y = 24 ∨ y = 25 :=
by 
  sorry

end cost_prices_max_units_B_possible_scenarios_l238_238582


namespace imaginary_part_of_z_l238_238676

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + complex.i) = 1) : complex.im z = -1/5 := 
sorry

end imaginary_part_of_z_l238_238676


namespace tom_typing_time_l238_238921

theorem tom_typing_time (typing_speed : ℕ) (words_per_page : ℕ) (number_of_pages : ℕ) 
  (h1 : typing_speed = 90) 
  (h2 : words_per_page = 450) 
  (h3 : number_of_pages = 10) : 
  (words_per_page / typing_speed) * number_of_pages = 50 :=
by simp [h1, h2, h3]; norm_num

end tom_typing_time_l238_238921


namespace eulerian_path_exists_l238_238035

variables {V : Type*} [Fintype V] (G : SimpleGraph V)

noncomputable def exists_eulerian_path (G : SimpleGraph V) : Prop :=
Connected G ∧ (∑ v in Finset.univ, if G.degree v % 2 = 1 then 1 else 0) ≤ 2

theorem eulerian_path_exists (hG : exists_eulerian_path G) : 
  ∃ p : G.Walk, p.isEulerian :=
sorry

end eulerian_path_exists_l238_238035


namespace cyclic_quadrilateral_at_D_inscribed_circle_radii_relation_l238_238157

theorem cyclic_quadrilateral_at_D
  (ABCD : ConvexQuadrilateral)
  (P : Point) (Q : Point)
  (circumscribed_A : IsCircumscribed (adjacentQuadrilateral ABCD A))
  (circumscribed_B : IsCircumscribed (adjacentQuadrilateral ABCD B))
  (circumscribed_C : IsCircumscribed (adjacentQuadrilateral ABCD C)) :
  IsCircumscribed (adjacentQuadrilateral ABCD D) :=
sorry

theorem inscribed_circle_radii_relation
  (ABCD : ConvexQuadrilateral)
  (P : Point) (Q : Point)
  (r_a r_b r_c r_d : ℝ)
  (inscribed_A : IsInscribed (adjacentQuadrilateralRadius ABCD A r_a))
  (inscribed_B : IsInscribed (adjacentQuadrilateralRadius ABCD B r_b))
  (inscribed_C : IsInscribed (adjacentQuadrilateralRadius ABCD C r_c))
  (inscribed_D : IsInscribed (adjacentQuadrilateralRadius ABCD D r_d)) :
  (1 / r_a + 1 / r_c = 1 / r_b + 1 / r_d) :=
sorry

end cyclic_quadrilateral_at_D_inscribed_circle_radii_relation_l238_238157


namespace log_relation_l238_238405

theorem log_relation (a b : ℝ) (h : log 2 a + log (1 / 2) b = 2) : a = 4 * b :=
by sorry

end log_relation_l238_238405


namespace minimize_quadratic_l238_238189

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_l238_238189


namespace log_equation_solution_l238_238737

noncomputable def k : ℕ := 9
noncomputable def x : ℕ := 81

theorem log_equation_solution (h : log k x * log 3 k = 4) (hk : k = 9) : x = 81 :=
by
  sorry

end log_equation_solution_l238_238737


namespace cost_per_sq_meter_l238_238597

def tank_dimensions : ℝ × ℝ × ℝ := (25, 12, 6)
def total_plastering_cost : ℝ := 186
def total_plastering_area : ℝ :=
  let (length, width, height) := tank_dimensions
  let area_bottom := length * width
  let area_longer_walls := length * height * 2
  let area_shorter_walls := width * height * 2
  area_bottom + area_longer_walls + area_shorter_walls

theorem cost_per_sq_meter : total_plastering_cost / total_plastering_area = 0.25 := by
  sorry

end cost_per_sq_meter_l238_238597


namespace smallest_number_divisible_by_6_in_permutations_list_l238_238023

def is_divisible_by_6 (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 6 * k)

noncomputable def permutations_5_digits := 
  [1, 2, 3, 4, 5].permutations.map (λ l => l.foldl (λ acc x => 10 * acc + x) 0)

theorem smallest_number_divisible_by_6_in_permutations_list :
  ∃ n ∈ permutations_5_digits, is_divisible_by_6 n ∧ (∀ m ∈ permutations_5_digits, is_divisible_by_6 m → n ≤ m) :=
sorry

end smallest_number_divisible_by_6_in_permutations_list_l238_238023


namespace train_length_approx_l238_238598

-- Define the known constants: speed and time
def speed := 20 -- Speed in meters per second
def time := 5.999520038396929 -- Time in seconds

-- Define the length of the train as speed multiplied by time
def train_length := speed * time

-- The theorem to be proved: train_length is approximately 119.99 meters
theorem train_length_approx : abs (train_length - 119.99) < 0.01 :=
by sorry

end train_length_approx_l238_238598


namespace complex_sine_sum_l238_238740

theorem complex_sine_sum :
  (∑ n in Finset.range 31, (Complex.i ^ n) * Real.sin (30 + 90 * n) * (Real.pi / 180)) = 7 + 8 * Complex.i * Real.sqrt 3 := 
sorry

end complex_sine_sum_l238_238740


namespace unattainable_y_l238_238710

theorem unattainable_y (x : ℝ) (h : x ≠ -5/4) : ¬∃ y : ℝ, y = (2 - 3 * x) / (4 * x + 5) ∧ y = -3 / 4 :=
by
  sorry

end unattainable_y_l238_238710


namespace max_projection_area_l238_238543

-- Define the conditions
def equilateral_triangle_area (s: ℝ) : ℝ := (sqrt 3 / 4) * s^2

def tetrahedron_projected_area (s: ℝ) (α: ℝ) : ℝ :=
  let S := equilateral_triangle_area s in
  max S (S * cos α + S * cos (2 * real.pi / 3 - α))

-- Define the problem statement
theorem max_projection_area : 
  ∀ s α, s = 1 → α = real.pi / 3 → tetrahedron_projected_area s α = sqrt 3 / 4 :=
by
  intros s α s_eq α_eq
  rw [s_eq, α_eq]
  -- rest of proof omitted
  sorry

end max_projection_area_l238_238543


namespace dot_product_of_collinear_vectors_l238_238706

variable (k : ℚ)

def vector_m : ℚ × ℚ := (2 * k - 1, k)
def vector_n : ℚ × ℚ := (4, 1)

def collinear (v1 v2 : ℚ × ℚ) : Prop :=
  ∃ (λ : ℚ), v1 = (λ * v2.1, λ * v2.2)

theorem dot_product_of_collinear_vectors :
  collinear (vector_m k) vector_n → (vector_m k).fst * vector_n.fst + (vector_m k).snd * vector_n.snd = -17 / 2 :=
by
  intros
  sorry

end dot_product_of_collinear_vectors_l238_238706


namespace geom_seq_triangle_area_l238_238420

noncomputable def area_of_triangle {a b c : ℝ} (B : ℝ) : ℝ :=
  1/2 * a * c * Real.sin B

theorem geom_seq_triangle_area (a b c : ℝ) (B : ℝ) 
  (h_geom_seq : b^2 = a * c) 
  (hb : b = 2) 
  (hB : B = Real.pi / 3) :
  area_of_triangle B = Real.sqrt 3 :=
by
  sorry

end geom_seq_triangle_area_l238_238420


namespace mass_percentage_of_H_in_carbonic_acid_is_3_26_l238_238331

noncomputable def mass_percentage_H_in_H2CO3 (molar_mass_H molar_mass_C molar_mass_O : ℝ) : ℝ :=
  let molar_mass_H2CO3 := (2 * molar_mass_H) + molar_mass_C + (3 * molar_mass_O)
  let total_mass_H := 2 * molar_mass_H
  (total_mass_H / molar_mass_H2CO3) * 100

theorem mass_percentage_of_H_in_carbonic_acid_is_3_26 :
  mass_percentage_H_in_H2CO3 1.01 12.01 16.00 ≈ 3.26 :=
by
  unfold mass_percentage_H_in_H2CO3
  -- We calculate the molar mass of H2CO3
  have h1 : (2 * 1.01 + 12.01 + 3 * 16.00) = 62.03 := by norm_num
  -- We calculate the total mass of H in H2CO3
  have h2 : 2 * 1.01 = 2.02 := by norm_num
  -- Now we compute the mass percentage
  have h3 : (2.02 / 62.03) * 100 ≈ 3.26 := by norm_num
  exact h3
  sorry

end mass_percentage_of_H_in_carbonic_acid_is_3_26_l238_238331


namespace min_pie_pieces_l238_238887

theorem min_pie_pieces (p : ℕ) : 
  (∀ (k : ℕ), (k = 5 ∨ k = 7) → ∃ (m : ℕ), p = k * m ∨ p = m * k) → p = 11 := 
sorry

end min_pie_pieces_l238_238887


namespace author_earnings_calculation_l238_238612

open Real

namespace AuthorEarnings

def paperCoverCopies  : ℕ := 32000
def paperCoverPrice   : ℝ := 0.20
def paperCoverPercent : ℝ := 0.06

def hardCoverCopies   : ℕ := 15000
def hardCoverPrice    : ℝ := 0.40
def hardCoverPercent  : ℝ := 0.12

def total_earnings_paper_cover : ℝ := paperCoverCopies * paperCoverPrice
def earnings_paper_cover : ℝ := total_earnings_paper_cover * paperCoverPercent

def total_earnings_hard_cover : ℝ := hardCoverCopies * hardCoverPrice
def earnings_hard_cover : ℝ := total_earnings_hard_cover * hardCoverPercent

def author_total_earnings : ℝ := earnings_paper_cover + earnings_hard_cover

theorem author_earnings_calculation : author_total_earnings = 1104 := by
  sorry

end AuthorEarnings

end author_earnings_calculation_l238_238612


namespace max_rabbits_with_traits_l238_238118

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l238_238118


namespace shaded_area_correct_l238_238283

-- Define the side lengths of the squares
def side_length_large_square : ℕ := 14
def side_length_small_square : ℕ := 10

-- Define the areas of the squares
def area_large_square : ℕ := side_length_large_square * side_length_large_square
def area_small_square : ℕ := side_length_small_square * side_length_small_square

-- Define the area of the shaded regions
def area_shaded_regions : ℕ := area_large_square - area_small_square

-- State the theorem
theorem shaded_area_correct : area_shaded_regions = 49 := by
  sorry

end shaded_area_correct_l238_238283


namespace change_in_area_is_1_percent_decrease_l238_238216

variable (b h : ℝ)

-- Define the original area
def originalArea : ℝ := (1/2) * b * h

-- Define the new base and height
def newBase : ℝ := 1.1 * b
def newHeight : ℝ := 0.9 * h

-- Define the new area
def newArea : ℝ := (1/2) * newBase * newHeight

-- Define the percentage change in area
def percentageChangeInArea : ℝ := ((newArea - originalArea) / originalArea) * 100

theorem change_in_area_is_1_percent_decrease : percentageChangeInArea b h = -1 :=
by
  sorry

end change_in_area_is_1_percent_decrease_l238_238216


namespace unique_z_exists_l238_238000

-- Variables and axioms
variables {x y p m n : ℤ} (hmn : Nat.gcd m n = 1)
          (hp : p.Prime) (hxm : Nat.modeq p (x^m) (y^n))

-- Main theorem
theorem unique_z_exists
  (hz : ∃ (z : ℕ), Nat.modeq p x (z^n) ∧ Nat.modeq p y (z^m)) :
  ∃! z : ℕ, Nat.modeq p x (z^n) ∧ Nat.modeq p y (z^m) := 
begin
  sorry
end

end unique_z_exists_l238_238000


namespace sin_half_x_increasing_on_neg_pi_to_pi_l238_238882

noncomputable def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem sin_half_x_increasing_on_neg_pi_to_pi :
  is_increasing_on_interval (λ x, sin (1 / 2 * x)) (-π) π :=
sorry

end sin_half_x_increasing_on_neg_pi_to_pi_l238_238882


namespace bicycle_route_total_length_l238_238770

theorem bicycle_route_total_length :
  let horizontal_length := 13 
  let vertical_length := 13 
  2 * horizontal_length + 2 * vertical_length = 52 :=
by
  let horizontal_length := 13
  let vertical_length := 13
  sorry

end bicycle_route_total_length_l238_238770


namespace find_x_l238_238485

-- Define vectors a and b
variables (x : ℝ)
def a : ℝ × ℝ := (x, 2)
def b : ℝ × ℝ := (1, -1)

-- Define the condition that a + b is perpendicular to b
def perp_condition : Prop := 
  let sum := (a x).1 + (b).1, (a x).2 + (b).2
  sum.1 * (b).1 + sum.2 * (b).2 = 0

-- Prove the value of x
theorem find_x (h : perp_condition x) : x = 0 :=
sorry

end find_x_l238_238485


namespace imaginary_part_of_z_l238_238679

-- Definition based on the problem condition
def z_condition (z : ℂ) : Prop := z * (2 + complex.i) = 1

-- Statement of the proof problem
theorem imaginary_part_of_z (z : ℂ) (h : z_condition z) : complex.im z = -1/5 :=
sorry

end imaginary_part_of_z_l238_238679


namespace quadratic_coefficients_l238_238558

theorem quadratic_coefficients (x : ℝ) :
    let a := 5
    let b := -6
    let c := (1/2 : ℝ)
    in 5 * x^2 + (1/2) = 6 * x → a = 5 ∧ b = -6 ∧ c = (1/2 : ℝ) :=
by 
    intro h
    have h1 : 5 * x^2 - 6 * x + (1/2) = 0,
    exact h,
    simp,
    split,
    {refl},
    split,
    {finish},
    {finish},
    sorry -- Completing the proof is not required

end quadratic_coefficients_l238_238558


namespace set_intersection_complement_l238_238390

open Set

variable M : Set ℤ := {-1, 0, 1, 2, 3, 4}
variable A : Set ℤ := {2, 3}
variable AB : Set ℤ := {1, 2, 3, 4}
variable B : Set ℤ

theorem set_intersection_complement (h : A ∪ B = AB) : B ∩ (M \ A) = {1, 4} := 
by sorry

end set_intersection_complement_l238_238390


namespace minimize_quadratic_function_l238_238177

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end minimize_quadratic_function_l238_238177


namespace find_f_l238_238683

noncomputable def f (a b : ℝ) (p : ℝ → ℝ) (x : ℝ) : ℝ :=
  (1/(a^2 - b^2)) * (a * sin x - b * x^2 * sin (1/x) + a * p(x) - b * x^2 * p(1/x))

theorem find_f
  (a b : ℝ)
  (x : ℝ)
  (p : ℝ → ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : a^2 ≠ b^2)
  (h4 : x > 0)
  (h5 : ∀ y, p(y) ≥ 0) :
  a * f a b p x + b * x^2 * f a b p (1/x) ≥ sin x :=
by {
  sorry
}

end find_f_l238_238683


namespace degree_of_f_l238_238168

noncomputable def f (x : ℝ) : ℝ := 3 + 7 * x^5 + 150 - 5 * real.pi * x^3 + real.sqrt 7 * x^2 + 20 * x^(1/2)

theorem degree_of_f : polynomial.degree (7 * polynomial.X ^ 5 + -5 * real.pi * polynomial.X ^ 3 + real.sqrt 7 * polynomial.X ^ 2 + 20 * polynomial.X^(1/2)) = 5 :=
sorry

end degree_of_f_l238_238168


namespace tom_typing_time_l238_238920

theorem tom_typing_time (typing_speed : ℕ) (words_per_page : ℕ) (number_of_pages : ℕ) 
  (h1 : typing_speed = 90) 
  (h2 : words_per_page = 450) 
  (h3 : number_of_pages = 10) : 
  (words_per_page / typing_speed) * number_of_pages = 50 :=
by simp [h1, h2, h3]; norm_num

end tom_typing_time_l238_238920


namespace find_A_l238_238363

variables {a b c : ℝ} {A B C : ℝ}
hypothesis h_triangle : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π
hypothesis h_sides : 2 * b = c + 2 * a * Real.cos C

theorem find_A (h_lt_180: A < π): A = π / 3 :=
by
  sorry

end find_A_l238_238363


namespace product_of_diagonals_of_cube_l238_238258

theorem product_of_diagonals_of_cube (side_length : ℕ) (h : side_length = 1) :
  let face_diagonals := 12
  let space_diagonals := 4
  let face_length := (real.sqrt 2)^12
  let space_length := (real.sqrt 3)^4
  (face_length * space_length) = 576 :=
by
  sorry

end product_of_diagonals_of_cube_l238_238258


namespace shaded_region_area_l238_238988

-- Define the side length of the regular octagon
def side_length : ℝ := 4

-- Define the formula for the area of a regular octagon given its side length
def area_octagon (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

-- Define the formula for the area of one semicircle given its radius
def area_semicircle (r : ℝ) : ℝ := (1/2) * Real.pi * r^2

-- Define the formula for the total area of 8 semicircles
def total_area_semicircles (r : ℝ) : ℝ := 8 * area_semicircle r

-- Define the radius of the semicircles
def radius : ℝ := side_length / 2

-- Prove that the area inside the octagon but outside all semicircles is as specified
theorem shaded_region_area : 
  let octagon_area := area_octagon side_length
  let semicircles_area := total_area_semicircles radius
  octagon_area - semicircles_area = 32 * (1 + Real.sqrt 2) - 16 * Real.pi :=
by
  -- Detailed proof skipped
  sorry

end shaded_region_area_l238_238988


namespace parallel_lines_parallel_lines_solution_l238_238412

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) → a = -1 ∨ a = 2 :=
sorry

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) ∧ 
  ((a = -1 → ∀ x y : ℝ, a * x + 2 * y + 6 = 0 ∧ (x + (a - 1) * y + (a^2 - 1) = 0)) ∨ 
  (a = 2 → ∀ x y : ℝ, a * x + 2 * y + 6 = 0 ∧ (x + (a - 1) * y + (a^2 - 1) = 0))) :=
sorry

end parallel_lines_parallel_lines_solution_l238_238412


namespace g_odd_and_decreasing_l238_238716

-- Define f(x) as a piecewise function
def f (x : ℝ) : ℝ :=
  if x > 0 then 1 else if x < 0 then -1 else 0

-- Define g(x)
def g (x : ℝ) : ℝ := 
  if x = 0 then 0 else f(x) / (x^2)

theorem g_odd_and_decreasing : 
  (∀ x : ℝ, g x = (f x) / (x^2)) ∧
  (∀ x : ℝ, g(-x) = -g(x)) ∧
  (∀ x : ℝ, x > 0 → g(x) < g(x / 2)) ∧ 
  (∀ x : ℝ, x < 0 → g(x) < g(x / 2)) :=
by
  sorry

end g_odd_and_decreasing_l238_238716


namespace solve_quadratic_l238_238872

theorem solve_quadratic : ∃ x : ℝ, (x = 1 ∨ x = 6) ∧ (x^2 - 7*x + 6 = 0) :=
by
  existsi 1
  existsi 6
  sorry

end solve_quadratic_l238_238872


namespace max_rabbits_l238_238113

theorem max_rabbits (N : ℕ) (h1 : ∀ k, k = N → k = 27 → true)
    (h2 : ∀ n_l : ℕ, n_l = 13 → n_l <= N)
    (h3 : ∀ n_j : ℕ, n_j = 17 → n_j <= N)
    (h4 : ∀ n_both : ℕ, n_both >= 3 → true) :
  N <= 27 :=
begin
  sorry
end

end max_rabbits_l238_238113


namespace maximum_rabbits_l238_238087

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l238_238087


namespace largest_interesting_number_l238_238164

def interesting (n : ℕ) : Prop :=
  ∀ i, 1 < i ∧ i < Nat.digits n ∧ i < List.length (Nat.digits n)
  → (Nat.digits n).nth i < ((Nat.digits n).nth (i - 1) + (Nat.digits n).nth (i + 1)) / 2

theorem largest_interesting_number :
  ∃ (n : ℕ), interesting n ∧ ∀ (m : ℕ), interesting m → m ≤ n :=
  ∃ (n : ℕ), interesting n ∧ (n = 96433469)
  sorry

end largest_interesting_number_l238_238164


namespace min_distance_between_points_on_circles_l238_238481

theorem min_distance_between_points_on_circles :
  let C₁ := {p : ℝ × ℝ | p.1^2 + p.2^2 + 4*p.1 + 2*p.2 + 1 = 0}
  let C₂ := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 4*p.2 + 6 = 0}
  ∃ P Q ∈ ℝ × ℝ, P ∈ C₁ ∧ Q ∈ C₂ ∧ dist P Q = 3 - Real.sqrt 2 :=
begin
  sorry
end

end min_distance_between_points_on_circles_l238_238481


namespace swimmers_meet_times_l238_238546

theorem swimmers_meet_times : 
  ∀ (pool_length time_minutes : ℝ) (speed_A speed_B : ℝ), 
    pool_length = 120 ∧
    time_minutes = 15 ∧
    speed_A = 4 ∧
    speed_B = 3 →
    let time_seconds := time_minutes * 60 in
    let LCM_period := Nat.lcm 60 80 in
    let total_full_periods := (time_seconds / LCM_period) in
    let full_meetings := 6 * (total_full_periods : ℕ) in
    let remaining_time := time_seconds % LCM_period in
    let extra_meetings := remaining_time / 240 * 6 in
    full_meetings + extra_meetings = 29 :=
sorry

end swimmers_meet_times_l238_238546


namespace min_value_d_squared_correct_l238_238854

noncomputable def min_value_d_squared (z : ℂ) (area : ℝ) (h_area : area = 12/13) (h_real_pos : 0 < z.re) : ℝ :=
  let dz := z + z⁻¹
  let d := complex.abs dz
  d^2

theorem min_value_d_squared_correct (z : ℂ) (h_area : 2 * abs ((z.imag) * (1/z).real - z.real * (1/z).imag) = 12/13) (h_real_pos : 0 < z.re) :
  min_value_d_squared z (h_area) h_area h_real_pos = 16/13 :=
sorry

end min_value_d_squared_correct_l238_238854


namespace lim_sup_eq_Union_lim_inf_l238_238649

open Set

theorem lim_sup_eq_Union_lim_inf
  (Ω : Type*)
  (A : ℕ → Set Ω) :
  (⋂ n, ⋃ k ≥ n, A k) = ⋃ (n_infty : ℕ → ℕ) (hn : StrictMono n_infty), ⋃ n, ⋂ k ≥ n, A (n_infty k) :=
by
  sorry

end lim_sup_eq_Union_lim_inf_l238_238649


namespace greatest_four_digit_number_divisible_by_3_and_4_l238_238927

theorem greatest_four_digit_number_divisible_by_3_and_4 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (n % 12 = 0) ∧ (∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ (m % 12 = 0) → m ≤ 9996) :=
by sorry

end greatest_four_digit_number_divisible_by_3_and_4_l238_238927


namespace complex_on_ellipse_l238_238349

noncomputable def complex_z (x y : ℝ) : ℂ := x + y * complex.I

theorem complex_on_ellipse (x y : ℝ) (h1 : (x^2 / 9) + (y^2 / 16) = 1)
  (h2 : ∃ a : ℝ, complex.ofReal a = (complex_z x (y - 1) - 1 - complex.I) / (complex_z x y - complex.I)) :
  complex_z x y = complex_z (3 * real.sqrt 15 / 4) 1 ∨ complex_z x y = complex_z (-3 * real.sqrt 15 / 4) 1 :=
begin
  sorry
end

end complex_on_ellipse_l238_238349


namespace problem_solution_l238_238358

def p1 : Prop := ∃ x : ℝ, x^2 + x + 1 < 0
def p2 : Prop := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - 1 ≥ 0

theorem problem_solution : (¬ p1) ∨ (¬ p2) :=
by
  sorry

end problem_solution_l238_238358


namespace author_earnings_calculation_l238_238611

open Real

namespace AuthorEarnings

def paperCoverCopies  : ℕ := 32000
def paperCoverPrice   : ℝ := 0.20
def paperCoverPercent : ℝ := 0.06

def hardCoverCopies   : ℕ := 15000
def hardCoverPrice    : ℝ := 0.40
def hardCoverPercent  : ℝ := 0.12

def total_earnings_paper_cover : ℝ := paperCoverCopies * paperCoverPrice
def earnings_paper_cover : ℝ := total_earnings_paper_cover * paperCoverPercent

def total_earnings_hard_cover : ℝ := hardCoverCopies * hardCoverPrice
def earnings_hard_cover : ℝ := total_earnings_hard_cover * hardCoverPercent

def author_total_earnings : ℝ := earnings_paper_cover + earnings_hard_cover

theorem author_earnings_calculation : author_total_earnings = 1104 := by
  sorry

end AuthorEarnings

end author_earnings_calculation_l238_238611


namespace derivative_of_y_l238_238657

-- Define the function y
def y (x : ℝ) : ℝ := - (1 / (3 * (Real.sin x)^3)) - (1 / (Real.sin x)) + (1 / 2) * Real.log ((1 + Real.sin x) / (1 - Real.sin x))

-- Statement to prove the derivative of y
theorem derivative_of_y (x : ℝ) : deriv y x = 1 / (Real.cos x * (Real.sin x)^4) := by
  sorry

end derivative_of_y_l238_238657


namespace tea_in_box_l238_238024

theorem tea_in_box (tea_per_day ounces_per_week ounces_per_box : ℝ) 
    (H1 : tea_per_day = 1 / 5) 
    (H2 : ounces_per_week = tea_per_day * 7) 
    (H3 : ounces_per_box = ounces_per_week * 20) : 
    ounces_per_box = 28 := 
by
  sorry

end tea_in_box_l238_238024


namespace non_visible_dots_total_l238_238341

theorem non_visible_dots_total (visible_sum total_dots : ℕ) 
                               (h_visible_sum : visible_sum = 32) 
                               (h_total_dots : total_dots = 84) : 
                               total_dots - visible_sum = 52 :=
by
  rw [h_visible_sum, h_total_dots]
  norm_num

end non_visible_dots_total_l238_238341


namespace first_number_is_105_percent_of_second_kilograms_reduced_by_10_percent_l238_238591

-- Proof problem 1: Given a number is 5% more than another number
theorem first_number_is_105_percent_of_second (x y : ℚ) (h : x = y * 1.05) : x = y * (1 + 0.05) :=
by {
  -- proof here
  sorry
}

-- Proof problem 2: 10 kilograms reduced by 10%
theorem kilograms_reduced_by_10_percent (kg : ℚ) (h : kg = 10) : kg * (1 - 0.1) = 9 :=
by {
  -- proof here
  sorry
}

end first_number_is_105_percent_of_second_kilograms_reduced_by_10_percent_l238_238591


namespace werner_ria_sum_l238_238548

/-- 
Problem Statement: Werner wrote a list of numbers with sum 22. 
Ria then subtracted each of Werner's numbers from 7 and wrote down her answers. 
The sum of Ria's numbers was 34. 
Show that the number of numbers Werner wrote down is 8.
-/

theorem werner_ria_sum (n : ℕ) (a : Fin n → ℕ) 
  (h₁ : (∑ i, a i) = 22) 
  (h₂ : (∑ i, (7 - a i)) = 34) : 
  n = 8 :=
begin
  sorry
end

end werner_ria_sum_l238_238548


namespace max_remaining_numbers_l238_238812

theorem max_remaining_numbers : 
  ∃ (S ⊆ {n | 1 ≤ n ∧ n ≤ 235}), (∀ x ∈ S, ∀ y ∈ S, x ≠ y → ¬(x ∣ (y - x))) ∧ card S = 118 :=
by
  sorry

end max_remaining_numbers_l238_238812


namespace Janet_initial_crayons_l238_238492

variable (Michelle_initial Janet_initial Michelle_final : ℕ)

theorem Janet_initial_crayons (h1 : Michelle_initial = 2) (h2 : Michelle_final = 4) (h3 : Michelle_final = Michelle_initial + Janet_initial) :
  Janet_initial = 2 :=
by
  sorry

end Janet_initial_crayons_l238_238492


namespace initial_walnut_trees_l238_238538

theorem initial_walnut_trees (total_trees_after_planting : ℕ) (trees_planted_today : ℕ) (initial_trees : ℕ) : 
  (total_trees_after_planting = 55) → (trees_planted_today = 33) → (initial_trees + trees_planted_today = total_trees_after_planting) → (initial_trees = 22) :=
by
  sorry

end initial_walnut_trees_l238_238538


namespace crow_avg_speed_l238_238966

-- Constants and Conditions
def distance : ℝ := 400 -- distance in meters
def trips : ℝ := 15
def time_hours : ℝ := 1.5
def average_speed_kmh : ℝ := 25 -- average speed in km/h
def against_wind_reduction : ℝ := 0.30
def with_wind_increase : ℝ := 0.20

-- Helper definitions
def average_speed_ms := average_speed_kmh * 1000 / 3600
def speed_against_wind := average_speed_ms * (1 - against_wind_reduction)
def speed_with_wind := average_speed_ms * (1 + with_wind_increase)
def time_against_wind := distance / speed_against_wind
def time_with_wind := distance / speed_with_wind
def total_time_round_trip := time_against_wind + time_with_wind
def total_travel_time := time_hours * 3600
def total_distance := trips * (2 * distance)
def average_speed := total_distance / total_travel_time

theorem crow_avg_speed:
  average_speed * 3600 / 1000 = 8 := by
  sorry

end crow_avg_speed_l238_238966


namespace small_frog_reaches_7th_rung_medium_frog_cannot_reach_1st_rung_large_frog_reaches_3rd_rung_l238_238156

-- 1. Prove that the small frog can reach the 7th rung
theorem small_frog_reaches_7th_rung : ∃ (a b : ℕ), 2 * a + 3 * b = 7 :=
by sorry

-- 2. Prove that the medium frog cannot reach the 1st rung
theorem medium_frog_cannot_reach_1st_rung : ¬(∃ (a b : ℕ), 2 * a + 4 * b = 1) :=
by sorry

-- 3. Prove that the large frog can reach the 3rd rung
theorem large_frog_reaches_3rd_rung : ∃ (a b : ℕ), 6 * a + 9 * b = 3 :=
by sorry

end small_frog_reaches_7th_rung_medium_frog_cannot_reach_1st_rung_large_frog_reaches_3rd_rung_l238_238156


namespace max_remained_numbers_l238_238846

theorem max_remained_numbers (S : Finset ℕ) (hSubset : S ⊆ Finset.range 236)
  (hCondition : ∀ a b c ∈ S, a ≠ b → a ≠ c → b ≠ c → ¬(b - a ∣ c)) : S.card ≤ 118 := 
sorry

end max_remained_numbers_l238_238846


namespace complex_number_properties_l238_238365

def z1_condition (z1 : ℂ) : Prop :=
  let z2 := z1 + 1 / z1 in
  z2.re ∈ Icc (-1 : ℝ) 1

noncomputable def omega (z1 : ℂ) : ℂ :=
  (1 - z1) / (1 + z1)

theorem complex_number_properties (z1 : ℂ) (h : z1_condition z1) :
  abs z1 = 1 ∧ z1.re ∈ Icc (-1 / 2 : ℝ) (1 / 2) ∧ ∃ b : ℝ, omega z1 = b * complex.I :=
sorry

end complex_number_properties_l238_238365


namespace mixed_doubles_teams_l238_238319

theorem mixed_doubles_teams (males females : ℕ) (hm : males = 6) (hf : females = 7) : (males * females) = 42 :=
by
  sorry

end mixed_doubles_teams_l238_238319


namespace num_students_in_second_class_l238_238877

theorem num_students_in_second_class 
  (avg1 : ℕ) (num1 : ℕ) (avg2 : ℕ) (overall_avg : ℕ) (n : ℕ) :
  avg1 = 50 → num1 = 30 → avg2 = 60 → overall_avg = 5625 → 
  (num1 * avg1 + n * avg2) = (num1 + n) * overall_avg → n = 50 :=
by sorry

end num_students_in_second_class_l238_238877


namespace minimum_area_of_triangle_l238_238695

noncomputable def dist (x₁ y₁ x₂ y₂ : ℝ) : ℝ := sqrt ((x₁ - x₂) ^ 2 + (y₁ - y₂) ^ 2)

def circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

def minAreaTriangleABP : ℝ :=
  let A := (0 : ℝ, -3 : ℝ)
  let B := (4 : ℝ, 0 : ℝ)
  let d := dist 0 1 (0 : ℝ) (0 : ℝ)
  let r := 1
  (d - r) * dist 0 (-3) 4 0 / 2

theorem minimum_area_of_triangle :
  ∀ P : ℝ × ℝ, circle P.1 P.2 → minAreaTriangleABP = 11 / 2 :=
begin
  sorry
end

end minimum_area_of_triangle_l238_238695


namespace reading_schedule_correct_l238_238769

-- Defining the conditions
def total_words : ℕ := 34685
def words_day1 (x : ℕ) : ℕ := x
def words_day2 (x : ℕ) : ℕ := 2 * x
def words_day3 (x : ℕ) : ℕ := 4 * x

-- Defining the main statement of the problem
theorem reading_schedule_correct (x : ℕ) : 
  words_day1 x + words_day2 x + words_day3 x = total_words := 
sorry

end reading_schedule_correct_l238_238769


namespace cost_price_of_cupboard_l238_238231

theorem cost_price_of_cupboard (C S S_profit : ℝ) (h1 : S = 0.88 * C) (h2 : S_profit = 1.12 * C) (h3 : S_profit - S = 1650) :
  C = 6875 := by
  sorry

end cost_price_of_cupboard_l238_238231


namespace number_of_digits_of_smallest_n_l238_238790

-- Definitions based on the conditions
def is_divisible (n k : ℕ) : Prop := ∃ m, n = k * m
def is_perfect_square (n : ℕ) : Prop := ∃ k, n = k * k

noncomputable def smallest_n : ℕ :=
  Inf {n : ℕ | is_perfect_square n ∧ is_perfect_square (n * n) ∧ is_divisible n 24}

theorem number_of_digits_of_smallest_n : (smallest_n.toString.length = 3) := by
  sorry

end number_of_digits_of_smallest_n_l238_238790


namespace sixth_student_stickers_l238_238146

-- Define the given conditions.
def first_student_stickers := 29
def increment := 6

-- Define the number of stickers given to each subsequent student.
def stickers (n : ℕ) : ℕ :=
  first_student_stickers + n * increment

-- Theorem statement: the 6th student will receive 59 stickers.
theorem sixth_student_stickers : stickers 5 = 59 :=
by
  sorry

end sixth_student_stickers_l238_238146


namespace premium_percentage_on_shares_l238_238590

theorem premium_percentage_on_shares
    (investment : ℕ)
    (share_price : ℕ)
    (premium_percentage : ℕ)
    (dividend_percentage : ℕ)
    (total_dividend : ℕ)
    (number_of_shares : ℕ)
    (investment_eq : investment = number_of_shares * (share_price + premium_percentage))
    (dividend_eq : total_dividend = number_of_shares * (share_price * dividend_percentage / 100))
    (investment_val : investment = 14400)
    (share_price_val : share_price = 100)
    (dividend_percentage_val : dividend_percentage = 5)
    (total_dividend_val : total_dividend = 600)
    (number_of_shares_val : number_of_shares = 600 / 5) :
    premium_percentage = 20 :=
by
  sorry

end premium_percentage_on_shares_l238_238590


namespace sophia_read_more_pages_l238_238507

variable (total_pages : ℝ) (finished_fraction : ℝ)
variable (pages_read : ℝ) (pages_left : ℝ) (pages_more : ℝ)

theorem sophia_read_more_pages :
  total_pages = 269.99999999999994 ∧
  finished_fraction = 2/3 ∧
  pages_read = finished_fraction * total_pages ∧
  pages_left = total_pages - pages_read →
  pages_more = pages_read - pages_left →
  pages_more = 90 := 
by
  intro h
  sorry

end sophia_read_more_pages_l238_238507


namespace train_crosses_platform_time_l238_238957

theorem train_crosses_platform_time :
  let length_train := 300 -- length of the train in meters
  let time_pole := 16 -- time to cross the signal pole in seconds
  let length_platform := 431.25 -- length of the platform in meters
  let speed_train := length_train / time_pole -- speed of the train in m/s
  let total_distance := length_train + length_platform -- total distance to cross the platform
  total_distance / speed_train = 39 :=
by
  let length_train := 300
  let time_pole := 16
  let length_platform := 431.25
  let speed_train := length_train / time_pole
  let total_distance := length_train + length_platform
  sorry

end train_crosses_platform_time_l238_238957


namespace men_women_equal_after_city_Y_l238_238958

variable (M W M' W' : ℕ)

-- Initial conditions: total passengers, women to men ratio
variable (h1 : M + W = 72)
variable (h2 : W = M / 2)

-- Changes in city Y: men leave, women enter
variable (h3 : M' = M - 16)
variable (h4 : W' = W + 8)

theorem men_women_equal_after_city_Y (h1 : M + W = 72) (h2 : W = M / 2) (h3 : M' = M - 16) (h4 : W' = W + 8) : 
  M' = W' := 
by 
  sorry

end men_women_equal_after_city_Y_l238_238958


namespace max_rabbits_with_traits_l238_238122

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l238_238122


namespace find_f_1_l238_238739

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_1 : (∀ x : ℝ, f x + 3 * f (-x) = Real.logb 2 (x + 3)) → f 1 = 1 / 8 := 
by 
  sorry

end find_f_1_l238_238739


namespace value_of_f_at_2_l238_238406

def f (x : ℝ) : ℝ :=
  x^3 - x - 1

theorem value_of_f_at_2 : f 2 = 5 := by
  -- Proof goes here
  sorry

end value_of_f_at_2_l238_238406


namespace exists_root_in_interval_range_of_a_l238_238576

-- Problem 1: Prove there is a root in the interval [-1, 0]
theorem exists_root_in_interval
  (f : ℝ → ℝ := λ x, 2^x - x^2)
  (a b : ℝ) 
  (h : a ≤ b) 
  (h1 : f a * f b ≤ 0) 
  (h_cont : continuous_on f (set.Icc a b))
  : ∃ x ∈ set.Icc a b, f x = 0 :=
begin
  use intermediate_value_theorem,
  sorry
end

-- Problem 2: Prove the range of a is (2, +∞) for exactly one root in the interval (0, 1)
theorem range_of_a 
  (a : ℝ) 
  (h : a > 2) 
  : ∃! x ∈ set.Ioo 0 1, a * x^2 - x - 1 = 0 :=
begin
  use quadratic_has_exactly_one_solution,
  sorry
end

end exists_root_in_interval_range_of_a_l238_238576


namespace arithmetic_seq_sum_l238_238700

variable (a : ℕ → ℝ)

open ArithSeq

-- Given conditions
axiom h_arith_seq : ∀ n, a (n+1) - a n = a 1 - a 0
axiom h_sum : a 1 + a 5 + a 9 = 6

-- The statement to prove
theorem arithmetic_seq_sum : a 2 + a 8 = 4 := by
  sorry -- proof to be completed

end arithmetic_seq_sum_l238_238700


namespace sum_factorial_eq_l238_238291

theorem sum_factorial_eq (n : ℕ) : (∑ k in Finset.range (n+1), (k+1) * (k+1)!) = (n+2)! - 1 :=
by
  sorry

end sum_factorial_eq_l238_238291


namespace option_A_incorrect_l238_238220

theorem option_A_incorrect {a b m : ℤ} (h : am = bm) : m = 0 ∨ a = b :=
by sorry

end option_A_incorrect_l238_238220


namespace simplify_expression_l238_238057

theorem simplify_expression (x y z : ℝ) (h1 : x = 3) (h2 : y = 2) (h3 : z = 4) :
  (12 * x^2 * y^3 * z) / (4 * x * y * z^2) = 9 :=
by
  sorry

end simplify_expression_l238_238057


namespace pentagon_midpoints_centroids_intersection_l238_238894

-- Definition of a convex pentagon in terms of its vertices
structure Pentagon :=
  (A B C D E : Point)

-- Definition of the midpoint of a side
def midpoint (P Q : Point) : Point := sorry

-- Definition of the centroid of a triangle
def centroid (P Q R : Point) : Point := sorry

-- Description of the convex property
def is_convex (p : Pentagon) : Prop := sorry

-- The Lean 4 theorem statement
theorem pentagon_midpoints_centroids_intersection (P : Pentagon) (h : is_convex P) :
  ∃ G, 
    (∃ M₁ C₁, M₁ = midpoint P.A P.B ∧ C₁ = centroid P.C P.D P.E ∧ G ∈ line_segment M₁ C₁) ∧
    (∃ M₂ C₂, M₂ = midpoint P.B P.C ∧ C₂ = centroid P.A P.D P.E ∧ G ∈ line_segment M₂ C₂) ∧
    (∃ M₃ C₃, M₃ = midpoint P.C P.D ∧ C₃ = centroid P.A P.B P.E ∧ G ∈ line_segment M₃ C₃) ∧
    (∃ M₄ C₄, M₄ = midpoint P.D P.E ∧ C₄ = centroid P.A P.B P.C ∧ G ∈ line_segment M₄ C₄) ∧
    (∃ M₅ C₅, M₅ = midpoint P.E P.A ∧ C₅ = centroid P.A P.B P.D ∧ G ∈ line_segment M₅ C₅) :=
sorry

end pentagon_midpoints_centroids_intersection_l238_238894


namespace largest_number_among_list_l238_238562

theorem largest_number_among_list : 
  let nums : List ℝ := [0.997, 0.9797, 0.97, 0.979, 0.9709] in 
    ∀ n ∈ nums, n ≤ 0.997 := 
begin
  intro nums,
  intros n hn,
  apply List.Mem.elim hn,
  repeat {intro h, rw h},
  all_goals {simp}
end

end largest_number_among_list_l238_238562


namespace triangle_area_zero_when_sides_doubled_l238_238775

theorem triangle_area_zero_when_sides_doubled 
  (AB AC BC : ℝ)
  (hAB : AB = 12)
  (hAC : AC = 7)
  (hBC : BC = 10)
  (AB' := 2 * AB)
  (AC' := 2 * AC) :
  ¬((AB' + AC' > BC) ∧ (AB' + BC > AC') ∧ (AC' + BC > AB')) :=
by
  have hAB' : AB' = 24 := by simp [AB', hAB]
  have hAC' : AC' = 14 := by simp [AC', hAC]
  have hBC : BC = 10 := hBC
  simp [hAB', hAC', hBC]
  split
  all_goals { sorry }

end triangle_area_zero_when_sides_doubled_l238_238775


namespace sqrt_inequality_trig_identities_eq_generalized_trig_identity_l238_238577

-- Prove that √8 - √6 < √5 - √3
theorem sqrt_inequality : Real.sqrt 8 - Real.sqrt 6 < Real.sqrt 5 - Real.sqrt 3 :=
sorry

-- Prove that the given trigonometric expressions all equal 3/4
theorem trig_identities_eq :
   (Real.sin (13 * Real.pi / 180)) ^ 2 + (Real.cos (17 * Real.pi / 180)) ^ 2 - Real.sin (13 * Real.pi / 180) * Real.cos (17 * Real.pi / 180) = 3/4 ∧
   (Real.sin (15 * Real.pi / 180)) ^ 2 + (Real.cos (15 * Real.pi / 180)) ^ 2 - Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 3/4 ∧
   (Real.sin (18 * Real.pi / 180)) ^ 2 + (Real.cos (12 * Real.pi / 180)) ^ 2 - Real.sin (18 * Real.pi / 180) * Real.cos (12 * Real.pi / 180) = 3/4 ∧
   (Real.sin (-18 * Real.pi / 180)) ^ 2 + (Real.cos (48 * Real.pi / 180)) ^ 2 - Real.sin (-18 * Real.pi / 180) * Real.cos (48 * Real.pi / 180) = 3/4 ∧
   (Real.sin (-25 * Real.pi / 180)) ^ 2 + (Real.cos (55 * Real.pi / 180)) ^ 2 - Real.sin (-25 * Real.pi / 180) * Real.cos (55 * Real.pi / 180) = 3/4 :=
sorry

-- Prove the generalized trigonometric identity
theorem generalized_trig_identity (α : ℝ) : 
  sin(α) ^ 2 + cos((30:ℝ) * Real.pi / 180 - α) ^ 2 - sin(α) * cos((30:ℝ) * Real.pi / 180 - α) = 3 / 4 :=
sorry

end sqrt_inequality_trig_identities_eq_generalized_trig_identity_l238_238577


namespace rate_percent_per_annum_l238_238946

theorem rate_percent_per_annum (P : ℝ) (SI_increase : ℝ) (T_increase : ℝ) (R : ℝ) 
  (hP : P = 2000) (hSI_increase : SI_increase = 40) (hT_increase : T_increase = 4) 
  (h : SI_increase = P * R * T_increase / 100) : R = 0.5 :=
by  
  sorry

end rate_percent_per_annum_l238_238946


namespace rectangle_width_squared_l238_238956

theorem rectangle_width_squared (w l : ℝ) (h1 : w^2 + l^2 = 400) (h2 : 4 * w^2 + l^2 = 484) : w^2 = 28 := 
by
  sorry

end rectangle_width_squared_l238_238956


namespace area_between_circles_l238_238532

theorem area_between_circles (C A D B : Point) (r1 r2 : ℝ)
    (h1 : C = center A B D) (h2 : chord_tangent_in_inner_circle A D B)
    (h3 : distance C A = 13) (h4 : distance A D = 20) :
    area_of_annulus r1 r2 = 100 * π :=
by
  sorry

end area_between_circles_l238_238532


namespace costume_processing_time_l238_238961

theorem costume_processing_time (x : ℕ) : 
  (300 - 60) / (2 * x) + 60 / x = 9 → (60 / x) + (240 / (2 * x)) = 9 :=
by
  sorry

end costume_processing_time_l238_238961


namespace percent_increase_is_correct_l238_238245

noncomputable def smaller_diameter := 15  -- Diameter of the smaller track
noncomputable def larger_diameter := 20   -- Diameter of the larger track

noncomputable def radius (d : ℝ) := d / 2

noncomputable def area (r : ℝ) := Real.pi * r^2

noncomputable def percent_increase (small_large : ℝ × ℝ) :=
  let (small, large) := small_large
  (area large - area small) / area small * 100

theorem percent_increase_is_correct :
  percent_increase (radius smaller_diameter, radius larger_diameter) ≈ 77.78 :=
sorry

end percent_increase_is_correct_l238_238245


namespace b_tenth_term_l238_238269

noncomputable def b : ℕ → ℚ
| 1 := 2
| 2 := 5
| (n + 3) := (b (n + 2) + b (n + 1)) / (2 * b (n + 2) - b (n + 1))

theorem b_tenth_term : b 10 = 24 / 5 := 
by 
  sorry

end b_tenth_term_l238_238269


namespace quadratic_polynomial_solution_l238_238654

noncomputable def exists_quadratic_polynomials (a b : ℤ) : Prop :=
  ∃ p q : ℤ[x], ¬(q = 0) ∧ ((p^2) - (Polynomial.C (1:ℤ) * (Polynomial.X^2 + Polynomial.C a * Polynomial.X + Polynomial.C b)) * (q^2) = Polynomial.C 1)

theorem quadratic_polynomial_solution :
  ∀ (a b : ℤ), exists_quadratic_polynomials a b ↔
    (∃ k : ℤ, (a = 2 * k + 1 ∧ b = k^2 + k) ∨
               (a = 2 * k ∧ (b = k^2 + 1 ∨ b = k^2 - 1 ∨ b = k^2 + 2 ∨ b = k^2 - 2))) :=
by
  sorry

end quadratic_polynomial_solution_l238_238654


namespace sphere_radius_correct_l238_238436

noncomputable def sphere_radius (a b θ : ℝ) : ℝ :=
  (Real.sqrt (b^2 - a^2 * (Real.cos θ)^2)) / (2 * Real.sin θ)

theorem sphere_radius_correct (A B C D : ℝ × ℝ × ℝ) 
  (m n : ℝ × ℝ × ℝ → Prop)
  (a b θ : ℝ)
  (ha : dist A B = a)
  (hb : dist C D = b)
  (hθ : ∀ (p q : ℝ × ℝ × ℝ), m p → n q → angle p q = θ)
  (hAC_perp_AB : ∀ (p : ℝ × ℝ × ℝ), m p → ∃ (q : ℝ × ℝ × ℝ), is_perpendicular p q ∧ is_on_line q AB)
  (hBD_perp_AB : ∀ (p : ℝ × ℝ × ℝ), n p → ∃ (q : ℝ × ℝ × ℝ), is_perpendicular p q ∧ is_on_line q AB) :
  radius_of_sphere_passing_through A B C D = sphere_radius a b θ :=
sorry

end sphere_radius_correct_l238_238436


namespace planting_area_l238_238972

variable (x : ℝ)

def garden_length := x + 2
def garden_width := 4
def path_width := 1

def effective_garden_length := garden_length x - 2 * path_width
def effective_garden_width := garden_width - 2 * path_width

theorem planting_area : effective_garden_length x * effective_garden_width = 2 * x := by
  simp [garden_length, garden_width, path_width, effective_garden_length, effective_garden_width]
  sorry

end planting_area_l238_238972


namespace total_amount_of_money_l238_238249

theorem total_amount_of_money (P1 : ℝ) (interest_total : ℝ)
  (hP1 : P1 = 299.99999999999994) (hInterest : interest_total = 144) :
  ∃ T : ℝ, T = 3000 :=
by
  sorry

end total_amount_of_money_l238_238249


namespace final_toy_count_correct_l238_238641

def initial_toy_count : ℝ := 5.3
def tuesday_toys_left (initial: ℝ) : ℝ := initial * 0.605
def tuesday_new_toys : ℝ := 3.6
def wednesday_toys_left (tuesday_total: ℝ) : ℝ := tuesday_total * 0.498
def wednesday_new_toys : ℝ := 2.4
def thursday_toys_left (wednesday_total: ℝ) : ℝ := wednesday_total * 0.692
def thursday_new_toys : ℝ := 4.5

def total_toys (initial: ℝ) : ℝ :=
  let after_tuesday := tuesday_toys_left initial + tuesday_new_toys
  let after_wednesday := wednesday_toys_left after_tuesday + wednesday_new_toys
  let after_thursday := thursday_toys_left after_wednesday + thursday_new_toys
  after_thursday

def toys_lost_tuesday (initial: ℝ) (left: ℝ) : ℝ := initial - left
def toys_lost_wednesday (tuesday_total: ℝ) (left: ℝ) : ℝ := tuesday_total - left
def toys_lost_thursday (wednesday_total: ℝ) (left: ℝ) : ℝ := wednesday_total - left
def total_lost_toys (initial: ℝ) : ℝ :=
  let tuesday_left := tuesday_toys_left initial
  let tuesday_total := tuesday_left + tuesday_new_toys
  let wednesday_left := wednesday_toys_left tuesday_total
  let wednesday_total := wednesday_left + wednesday_new_toys
  let thursday_left := thursday_toys_left wednesday_total
  let lost_tuesday := toys_lost_tuesday initial tuesday_left
  let lost_wednesday := toys_lost_wednesday tuesday_total wednesday_left
  let lost_thursday := toys_lost_thursday wednesday_total thursday_left
  lost_tuesday + lost_wednesday + lost_thursday

def final_toy_count (initial: ℝ) : ℝ :=
  let current_toys := total_toys initial
  let lost_toys := total_lost_toys initial
  current_toys + lost_toys

theorem final_toy_count_correct :
  final_toy_count initial_toy_count = 15.8 := sorry

end final_toy_count_correct_l238_238641


namespace sum_of_angles_solution_l238_238660

theorem sum_of_angles_solution (x : ℝ) (hx : x ∈ set.Icc 0 360):
  (sin x)^3 - (cos x)^2 = 1 / (cos x) - 1 / (sin x) →
  ∃ y ∈ {45, 225}, x = y →
  45 + 225 = 270 :=
by
  intros hx_cond hy_cond hyp_eq
  -- Proof steps will be filled here
  sorry

end sum_of_angles_solution_l238_238660


namespace maximum_rabbits_l238_238086

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l238_238086


namespace largest_on_edge_l238_238029

/-- On a grid, each cell contains a number which is the arithmetic mean of the four numbers around it 
    and all numbers are different. Prove that the largest number is located on the edge of the grid. -/
theorem largest_on_edge 
    (grid : ℕ → ℕ → ℝ) 
    (h_condition : ∀ (i j : ℕ), grid i j = (grid (i+1) j + grid (i-1) j + grid i (j+1) + grid i (j-1)) / 4)
    (h_unique : ∀ (i1 j1 i2 j2 : ℕ), (i1 ≠ i2 ∨ j1 ≠ j2) → grid i1 j1 ≠ grid i2 j2)
    : ∃ (i j : ℕ), (i = 0 ∨ j = 0 ∨ i = max_i ∨ j = max_j) ∧ ∀ (x y : ℕ), grid x y ≤ grid i j :=
sorry

end largest_on_edge_l238_238029


namespace ned_mowed_in_summer_l238_238497

def mowed_in_summer (total_mows spring_mows summer_mows : ℕ) : Prop :=
  total_mows = spring_mows + summer_mows

theorem ned_mowed_in_summer :
  ∀ (total_mows spring_mows summer_mows : ℕ),
  total_mows = 11 →
  spring_mows = 6 →
  mowed_in_summer total_mows spring_mows summer_mows →
  summer_mows = 5 :=
by
  intros total_mows spring_mows summer_mows h_total h_spring h_mowed
  sorry

end ned_mowed_in_summer_l238_238497


namespace number_of_intersection_points_l238_238430

noncomputable section

-- Define a type for Points in the plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Defining the five points
variables (A B C D E : Point)

-- Define the conditions that no three points are collinear
def no_three_collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) ≠ (C.x - A.x) * (B.y - A.y)

-- Define the theorem statement
theorem number_of_intersection_points (h1 : no_three_collinear A B C)
  (h2 : no_three_collinear A B D)
  (h3 : no_three_collinear A B E)
  (h4 : no_three_collinear A C D)
  (h5 : no_three_collinear A C E)
  (h6 : no_three_collinear A D E)
  (h7 : no_three_collinear B C D)
  (h8 : no_three_collinear B C E)
  (h9 : no_three_collinear B D E)
  (h10 : no_three_collinear C D E) :
  ∃ (N : ℕ), N = 40 :=
  sorry

end number_of_intersection_points_l238_238430


namespace smallest_n_divisible_by_24_and_864_l238_238173

theorem smallest_n_divisible_by_24_and_864 :
  ∃ n : ℕ, (0 < n) ∧ (24 ∣ n^2) ∧ (864 ∣ n^3) ∧ (∀ m : ℕ, (0 < m) → (24 ∣ m^2) → (864 ∣ m^3) → (n ≤ m)) :=
sorry

end smallest_n_divisible_by_24_and_864_l238_238173


namespace minimize_quadratic_l238_238192

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_l238_238192


namespace car_original_cost_price_l238_238968

theorem car_original_cost_price (C : ℝ) (SP1 SP2: ℝ) (FinalSalePrice : ℝ) (loss1 gain1 gain2 : ℝ):
  (loss1 = 0.11) → 
  (gain1 = 0.15) → 
  (gain2 = 0.25) → 
  (FinalSalePrice = 75000) → 
  (SP1 = C * (1 - loss1)) → 
  (SP2 = SP1 * (1 + gain1)) → 
  (SP2 * (1 + gain2) = FinalSalePrice) → 
  C ≈ 58744.16 :=
by
  intros h_loss1 h_gain1 h_gain2 h_FinalSalePrice h_SP1 h_SP2 h_SalePriceCalc
  sorry

end car_original_cost_price_l238_238968


namespace imaginary_part_of_z_l238_238675

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + complex.i) = 1) : complex.im z = -1/5 := 
sorry

end imaginary_part_of_z_l238_238675


namespace maximum_value_N_27_l238_238101

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end maximum_value_N_27_l238_238101


namespace triangle_parallel_line_length_ne_none_of_these_l238_238067

open Real

theorem triangle_parallel_line_length_ne_none_of_these :
  ∀ (base : ℝ) (length : ℝ),
  base = 15 ∧
  (∃ (k : ℝ), k > 0 ∧
    ∀ (h h₁ h₂ : ℝ), h > h₁ ∧ h₁ > h₂ →
      Area (Triangle base h) = 3 * Area (Triangle (base * k) h₁) ∧
      Area (Triangle (base * k) h₁) = 3 * Area (Triangle (base * k^2) h₂) ∧
      length = base * k) →
  length ≠ 4 * sqrt 3 ∧ length ≠ 10 ∧ length ≠ 5 * sqrt 6 ∧ length ≠ 7.5 :=
by
  sorry

end triangle_parallel_line_length_ne_none_of_these_l238_238067


namespace inequality_solution_l238_238379

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 1 else 1

theorem inequality_solution (x : ℝ) : 
  f (1 - x^2) > f (2 * x) ↔ -1 < x ∧ x < real.sqrt 2 - 1 := 
by
  sorry

end inequality_solution_l238_238379


namespace min_value_at_3_l238_238206

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end min_value_at_3_l238_238206


namespace four_digit_numbers_with_two_pairs_of_identical_digits_l238_238535

open Finset

theorem four_digit_numbers_with_two_pairs_of_identical_digits (n : ℕ) (digits : Finset ℕ) :
  n = 216 ∧ digits = range 1 10 ∧
  (∃ (a b : ℕ), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ (∀ x ∈ digits, x = a ∨ x = b)) →
  (∃ d : ℕ, d = 4) :=
by
  sorry

end four_digit_numbers_with_two_pairs_of_identical_digits_l238_238535


namespace solve_for_y_l238_238871

theorem solve_for_y (y : ℝ) : (2:ℝ)^(8^y) = (8:ℝ)^(2^y) ↔ y = real.log 3 / 2 :=
by
  sorry

end solve_for_y_l238_238871


namespace find_k_l238_238795

-- Define the set A using a condition on the quadratic equation
def A (k : ℝ) : Set ℝ := {x | k * x ^ 2 + 4 * x + 4 = 0}

-- Define the condition for the set A to have exactly one element
def has_exactly_one_element (k : ℝ) : Prop :=
  ∃ x : ℝ, A k = {x}

-- The problem statement is to find the value of k for which A has exactly one element
theorem find_k : ∃ k : ℝ, has_exactly_one_element k ∧ k = 1 :=
by
  simp [has_exactly_one_element, A]
  sorry

end find_k_l238_238795


namespace minimize_quadratic_function_l238_238178

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end minimize_quadratic_function_l238_238178


namespace spherical_to_cartesian_example_l238_238640

-- Define the spherical to Cartesian coordinate transformation
def spherical_to_cartesian (r θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  let z := r * Real.cos θ
  let x := r * Real.sin θ * Real.cos φ
  let y := r * Real.sin θ * Real.sin φ
  (x, y, z)

theorem spherical_to_cartesian_example :
  spherical_to_cartesian 8 (Real.pi / 3) (Real.pi / 6) = (6, 2 * Real.sqrt 3, 4) :=
by 
  sorry

end spherical_to_cartesian_example_l238_238640


namespace max_pawns_on_chessboard_l238_238552

/-- Define the conditions and calculate the maximum number of pawns -/
theorem max_pawns_on_chessboard :
  let max_pawns := 25 in
  ∀ (P : ℕ) (condition1 : P ≠ getSquare e4) (condition2 : ∀ (s1 s2 : Square), symmetric s1 s2 -> ¬ (P (s1) ∧ P (s2))),
  P <= max_pawns := by
  sorry

end max_pawns_on_chessboard_l238_238552


namespace maximum_rabbits_condition_l238_238109

-- Define the conditions and constraints
variables {N : ℕ}
variables (total_rabbits long_ears jump_far : ℕ)
variables (at_least_three_with_both : Prop)

-- State the conditions with exact values and assumptions
def conditions := 
  total_rabbits = N ∧
  long_ears = 13 ∧
  jump_far = 17 ∧
  at_least_three_with_both = (∃ a b c : ℕ, a >= 3 ∧ b = (long_ears - a) ∧ c = (jump_far - a))

-- State the theorem to be proved
theorem maximum_rabbits_condition :
  ∀ {N : ℕ}, conditions N long_ears jump_far at_least_three_with_both → N ≤ 27 :=
by sorry

end maximum_rabbits_condition_l238_238109


namespace total_roses_l238_238454

theorem total_roses (a b : Nat) (h_a : a = 16) (h_b : b = 7) : a + b = 23 := 
by
  -- Definitions from conditions
  rw [h_a, h_b]
  -- Now the goal is to prove 16 + 7 = 23
  sorry

end total_roses_l238_238454


namespace identify_linear_equation_l238_238219

theorem identify_linear_equation (A B C D : String)
    (eqA : A = "xy=3")
    (eqB : B = "3x+y^2=1")
    (eqC : C = "x+y=5")
    (eqD : D = "1/x+y=2") :
    (C = "x+y=5") ↔ (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (λ x y, a * x + b * y = c) = (5 = 1)) :=
by {
    sorry
}

end identify_linear_equation_l238_238219


namespace max_rabbits_with_long_ears_and_jumping_far_l238_238092

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l238_238092


namespace sum_of_fourth_powers_eq_square_of_sum_of_squares_l238_238857

theorem sum_of_fourth_powers_eq_square_of_sum_of_squares 
  (x1 x2 x3 : ℝ) (p q n : ℝ)
  (h1 : x1^3 + p*x1^2 + q*x1 + n = 0)
  (h2 : x2^3 + p*x2^2 + q*x2 + n = 0)
  (h3 : x3^3 + p*x3^2 + q*x3 + n = 0)
  (h_rel : q^2 = 2 * n * p) :
  x1^4 + x2^4 + x3^4 = (x1^2 + x2^2 + x3^2)^2 := 
sorry

end sum_of_fourth_powers_eq_square_of_sum_of_squares_l238_238857


namespace q_range_l238_238881

def q (x : ℝ) : ℝ := (x^2 - 2)^2

theorem q_range : 
  ∀ y : ℝ, y ∈ Set.range q ↔ 0 ≤ y :=
by sorry

end q_range_l238_238881


namespace max_remaining_numbers_l238_238840

theorem max_remaining_numbers : 
  ∃ s : Finset ℕ, s ⊆ (Finset.range 236) ∧ (∀ x y ∈ s, x ≠ y → ¬ (x - y).abs ∣ x) ∧ s.card = 118 := 
by
  sorry

end max_remaining_numbers_l238_238840


namespace sequence_general_term_and_minimum_k_l238_238378

theorem sequence_general_term_and_minimum_k (f : ℕ → ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n : ℕ, S n = -2 * n^2 + 22 * n) →
  (∀ n : ℕ, f n = -2 * n^2 + 22 * n) →
  (∀ n : ℕ, S n = sum (λ i, a i) n) →
  (∀ p : ℕ × ℕ, (∃ n : ℕ, p = (n, S n)) → f p.1 = p.2) →
  (∀ n : ℕ, a n = 24 - 4 * n) ∧ (∃ k : ℕ, (∀ n : ℕ, 1 ≤ n → sum (λ i, S i / i) n < k) ∧ k = 111) :=
by
  intro hS hf h_sum h_points
  sorry

end sequence_general_term_and_minimum_k_l238_238378


namespace cf_bisects_bg_l238_238432

-- Define the trapezoid ABCD with non-parallel sides AC and BD.
variables {A B C D E F G : Type}

-- Conditions
-- EF is the midline
-- EF is (2/3) of the longer parallel side AB
-- Extend midline from F by half its length to point H
-- H is connected to D by a line that intersects AB at G

-- Let’s consider A, B, C, and D points such that AB is parallel to CD
-- E and F are midpoints of AC and BD respectively
def trapezoid (A B C D E F G : Type) : Prop :=
  ∃ k : ℝ, 2 * k = (dist B D) ∧ 3 * k = (dist A B) ∧ k = (2 / 3) * (dist A B)

-- Now extend from F by half its length and denote the connecting point to D intersects AB at G
def extended_midline (F H : Type) : Prop :=
  dist F H = (1 / 2) * dist F E

-- Finally, expressing the desired proof that CF bisects BG
theorem cf_bisects_bg (A B C D E F G H : Type) [trapezoid A B C D E F G] [extended_midline F H] :
  let CF_bisects_BG := ∃ C1 : Type, dist C1 G = (1 / 2) * dist B G in CF_bisects_BG := sorry

end cf_bisects_bg_l238_238432


namespace sum_exterior_angles_decagon_l238_238907

theorem sum_exterior_angles_decagon : 
  ∀ (p : Type) [fintype p] [decidable_eq p] (polygon : p → Prop) 
  (hp : ∀ v, polygon v → polygon v.is_regular_decagon),
  ∑ v in (finset.univ : finset p), polygon.exterior_angle v = 360 :=
sorry

end sum_exterior_angles_decagon_l238_238907


namespace geometric_product_Pi8_l238_238022

def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

variables {a : ℕ → ℝ}
variable (h_geom : geometric_sequence a)
variable (h_prod : a 4 * a 5 = 2)

theorem geometric_product_Pi8 :
  (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) * (a 8) = 16 :=
by
  sorry

end geometric_product_Pi8_l238_238022


namespace count_even_in_pascal_top_15_rows_l238_238776

/--
  Prove that the number of even integers in the top 15 rows of Pascal's triangle can be calculated.
  We define Pascal's triangle such that the (n, k) element is given by the binomial coefficient C(n, k).
  We need to calculate C(n, k) mod 2 for each element in rows 0 through 14 and count the number of even results.
-/
theorem count_even_in_pascal_top_15_rows : ∃ (count : ℕ), count = (∑ n in Finset.range 15, 
    (∑ k in Finset.range (n+1), if (Nat.choose n k) % 2 = 0 then 1 else 0)) :=
sorry

end count_even_in_pascal_top_15_rows_l238_238776


namespace solution_set_l238_238008

variables {f : ℝ → ℝ}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def condition1 : Prop :=
  is_odd_function f

def condition2 : Prop :=
  f (-2) = 0

def condition3 (x : ℝ) : Prop :=
  x > 0 → (x * f' x - f x) / x^2 > 0

theorem solution_set (f : ℝ → ℝ)
  (odd_f : condition1)
  (cond2 : condition2)
  (cond3 : ∀ x, condition3 x) :
  ∀ x, (x < -2 ∨ x > 2) → x * f x > 0 := 
sorry

end solution_set_l238_238008


namespace fraction_calculation_l238_238925

theorem fraction_calculation :
  (3 / 4) * (1 / 2) * (2 / 5) * 5060 = 759 :=
by
  sorry

end fraction_calculation_l238_238925


namespace sufficient_but_not_necessary_not_necessary_l238_238471

variable (x y : ℝ)

theorem sufficient_but_not_necessary (h1: x ≥ 2) (h2: y ≥ 2): x^2 + y^2 ≥ 4 :=
by
  sorry

theorem not_necessary (hx4 : x^2 + y^2 ≥ 4) : ¬ (x ≥ 2 ∧ y ≥ 2) → ∃ x y, (x^2 + y^2 ≥ 4) ∧ (¬ (x ≥ 2) ∨ ¬ (y ≥ 2)) :=
by
  sorry

end sufficient_but_not_necessary_not_necessary_l238_238471


namespace dist_C1_to_l_BM_mul_BN_value_l238_238438

noncomputable def point_A := (4 * Real.sqrt 2, Real.pi / 4)
noncomputable def line_l_eq (ρ : ℝ) (θ : ℝ) := ρ * Real.cos (θ - Real.pi / 4) = 4 * Real.sqrt 2
noncomputable def curve_C1 (θ : ℝ) := (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)
noncomputable def point_B := (-2, 2)

theorem dist_C1_to_l :
  let d_max := (Real.sqrt 14 + 8 * Real.sqrt 2) / 2
  let d_min := (8 * Real.sqrt 2 - Real.sqrt 14) / 2
  ∃ d_max d_min, ∀ (θ : ℝ), 
    let (x, y) := curve_C1 θ in
    d_min <= (abs (x + y - 8) / Real.sqrt 2) ∧ (abs (x + y - 8) / Real.sqrt 2) <= d_max :=
sorry

theorem BM_mul_BN_value :
  ∃ (BM BN : ℝ), BM * BN = 32 / 7 :=
sorry

end dist_C1_to_l_BM_mul_BN_value_l238_238438


namespace find_volume_of_parallelepiped_l238_238070

noncomputable def volume_of_parallelepiped 
  (x y z : ℝ) 
  (h1 : x^2 + y^2 = 3) 
  (h2 : x^2 + z^2 = 5) 
  (h3 : y^2 + z^2 = 4) : ℝ :=
x * y * z

theorem find_volume_of_parallelepiped : 
  ∃ (x y z : ℝ),
  x^2 + y^2 = 3 ∧ 
  x^2 + z^2 = 5 ∧ 
  y^2 + z^2 = 4 ∧ 
  volume_of_parallelepiped x y z (by {sorry}) (by {sorry}) (by {sorry}) = sqrt 6 :=
begin
  sorry
end

end find_volume_of_parallelepiped_l238_238070


namespace sum_zero_of_cubic_identity_l238_238418

theorem sum_zero_of_cubic_identity (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) (h4 : a^3 + b^3 + c^3 = 3 * a * b * c) : 
  a + b + c = 0 :=
by
  sorry

end sum_zero_of_cubic_identity_l238_238418


namespace bob_pennies_l238_238410

-- Define the conditions
def condition_1 (a b : ℕ) : Prop :=
  b + 2 = 4 * (a - 2)

def condition_2 (a b : ℕ) : Prop :=
  b - 2 = 3 * (a + 2)

-- The proof statement
theorem bob_pennies (a b : ℕ) (h1 : condition_1 a b) (h2 : condition_2 a b) : a = 18 → b = 62 :=
by {
  intro ha,
  rw [ha] at h1 h2,
  sorry -- proof steps would go here
}

end bob_pennies_l238_238410


namespace minimum_hearing_condition_l238_238151

theorem minimum_hearing_condition (n : ℕ) (h : n = 50) : 
  ∀ (participants : Finset ℕ), 
  participants.card = 99 → 
  (∀ p ∈ participants, (∃ q ∈ participants, q ≠ p ∧ heard_of p q)) →
  (∃ a b ∈ participants, a ≠ b ∧ heard_of a b ∧ heard_of b a) := 
sorry

end minimum_hearing_condition_l238_238151


namespace smallest_n_divisible_by_24_and_864_l238_238172

theorem smallest_n_divisible_by_24_and_864 :
  ∃ n : ℕ, (0 < n) ∧ (24 ∣ n^2) ∧ (864 ∣ n^3) ∧ (∀ m : ℕ, (0 < m) → (24 ∣ m^2) → (864 ∣ m^3) → (n ≤ m)) :=
sorry

end smallest_n_divisible_by_24_and_864_l238_238172


namespace percentage_error_in_square_area_l238_238613

theorem percentage_error_in_square_area:
  ∀ (S : ℝ), let S' := 1.04 * S in 
             let A  := S * S in 
             let A' := S' * S' in
             (A' - A) / A * 100 = 8.16 :=
by
  intro S
  let S' := 1.04 * S
  let A := S * S
  let A' := S' * S'
  have h1: A' = 1.04 * 1.04 * S * S := sorry
  have h2: (A' - A) / A * 100 = ((1.04 * 1.04 - 1) * S * S) / (S * S) * 100 := sorry
  have h3: ... := sorry
  exact sorry

end percentage_error_in_square_area_l238_238613


namespace binomial_coeff_8_3_l238_238301

theorem binomial_coeff_8_3 : nat.choose 8 3 = 56 := by
  sorry

end binomial_coeff_8_3_l238_238301


namespace limit_at_a_l238_238236

noncomputable def limit_expression (x a : ℝ) : ℝ :=
  (2 - x / a) ^ Real.tan (π * x / (2 * a))

theorem limit_at_a (a : ℝ) (h : a ≠ 0) :
  filter.tendsto (λ x, limit_expression x a) (nhds a) (nhds (Real.exp (2 / π))) :=
sorry

end limit_at_a_l238_238236


namespace winning_candidate_vote_percentage_l238_238579

theorem winning_candidate_vote_percentage (a b c d e : ℕ) 
    (total_votes : ℕ) 
    (b_second_place : ℕ) 
    (c_third_place : ℕ) 
    (winner_votes : ℕ)
    (h1 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0)
    (h2 : a + b + c + d + e = 50000)
    (h3 : a = b + 12 * b / 100)
    (h4 : b = 30 * 50000 / 100)
    (h5 : c = b - 20 * b / 100) : 
    (a * 100 / total_votes) = 33.6 := by
  sorry

end winning_candidate_vote_percentage_l238_238579


namespace largest_possible_s_l238_238003

theorem largest_possible_s (r s : ℕ) (h1 : r ≥ s) (h2 : s ≥ 3) (h3 : (r - 2) * 180 * s = (s - 2) * 180 * r * 61 / 60) : s = 118 :=
sorry

end largest_possible_s_l238_238003


namespace sufficient_but_not_necessary_condition_m_sufficient_but_not_necessary_l238_238359

noncomputable def y (x m : ℝ) : ℝ := x^2 + m / x
noncomputable def y_prime (x m : ℝ) : ℝ := 2 * x - m / x^2

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x ≥ 1, y_prime x m ≥ 0) ↔ m ≤ 2 :=
sorry  -- Proof skipped as instructed

-- Now, state that m < 1 is a sufficient but not necessary condition
theorem m_sufficient_but_not_necessary (m : ℝ) :
  m < 1 → (∀ x ≥ 1, y_prime x m ≥ 0) :=
sorry  -- Proof skipped as instructed

end sufficient_but_not_necessary_condition_m_sufficient_but_not_necessary_l238_238359


namespace sum_p_q_l238_238274

-- Define the conditions
def is_intersection_point (k : ℝ) : Prop :=
  ∃ (p q : ℤ), k = p + real.sqrt (q) ∧
  |real.logBase 3 k - real.logBase 3 (k - 5)| = 1

-- State that if k = p + sqrt(q) then p + q = 11 given the geometrical constraints
theorem sum_p_q (k : ℝ) (p q : ℤ) (hk : k = p + real.sqrt q) (h_intersection : is_intersection_point k) : p + q = 11 :=
by
  sorry

end sum_p_q_l238_238274


namespace cartesian_equations_and_min_dist_l238_238424

-- Define the parametric curve C1
def C1_parametric (α : ℝ) : ℝ × ℝ :=
  (3 * Real.cos α, Real.sqrt 3 * Real.sin α)

-- Define the polar curve C2
def C2_polar (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Define the Cartesian equation of C1
def C1_cartesian (x y : ℝ) : Prop := (x^2 / 9 + y^2 / 3) = 1

-- Define the Cartesian equation of C2
def C2_cartesian (x y : ℝ) : Prop := ((x - 1)^2 + y^2 = 1)

-- Define the distance |PQ| between a point on C1 and a point on C2
def PQ_distance (α : ℝ) (x y : ℝ) : ℝ :=
  Real.sqrt((3 * Real.cos α - 1)^2 + 3 * Real.sin α^2)

-- Define the minimum distance |PQ|
def min_PQ_distance : ℝ := Real.sqrt 10 / 2 - 1

-- Statement to prove the Cartesian equations and minimum distance
theorem cartesian_equations_and_min_dist :
  (∀ α, C1_cartesian (3 * Real.cos α) (Real.sqrt 3 * Real.sin α)) ∧
  (∀ θ, ∃ x y, C2_cartesian x y ∧ C2_polar θ = Real.sqrt (x^2 + y^2) ∧ θ = Real.atan2 y x) ∧
  (∀ α, PQ_distance α (3 * Real.cos α) (Real.sqrt 3 * Real.sin α) ≥ min_PQ_distance) :=
by sorry

end cartesian_equations_and_min_dist_l238_238424


namespace problem_ACD_l238_238222

theorem problem_ACD :
  (∀ (R: Type) [ring R], ∀ (x y z w: R), x * x + 2 * x * y + y * y = z * z + 2 * z * w + w * w) ∧
  ¬ (∀ (n: ℝ), (λ x, x^n) = (λ x, 0) ∨ ∀ x, (λ x, x^n) x = 0) ∧ 
  (∃ x: ℝ, x^2 - 4 * x + 3 < 0) ∧
  (∀ (p q r: Prop), (p → q ∧ ¬ q → p) ∧ (q ↔ r) → (r → p ∧ ¬ p → r)) :=
by sorry

end problem_ACD_l238_238222


namespace centroid_distance_to_hypotenuse_l238_238890

theorem centroid_distance_to_hypotenuse {A B C : Point} (hABC : right_triangle A B C) 
  (M : Point) (hM : centroid A B C M) 
  (dM1 : dist M (proj M (leg1 hABC)) = 3) 
  (dM2 : dist M (proj M (leg2 hABC)) = 4) : 
  dist M (hypotenuse hABC) = 12 / 5 :=
begin
  sorry,
end

end centroid_distance_to_hypotenuse_l238_238890


namespace max_remaining_numbers_l238_238838

/-- 
The board initially has numbers 1, 2, 3, ..., 235.
Among the remaining numbers, no number is divisible by the difference of any two others.
Prove that the maximum number of numbers that could remain on the board is 118.
-/
theorem max_remaining_numbers : 
  ∃ S : set ℕ, (∀ a ∈ S, 1 ≤ a ∧ a ≤ 235) ∧ (∀ a b ∈ S, a ≠ b → ¬ ∃ d, d ∣ (a - b)) ∧ 
  ∃ T : set ℕ, S ⊆ T ∧ T ⊆ finset.range 236 ∧ T.card = 118 := 
sorry

end max_remaining_numbers_l238_238838


namespace problem1_eval_problem2_eval_l238_238992

theorem problem1_eval : 
  (-2)^2 + 24 * (-1/8 + 2/3 - 5/6) = -3 := by
  sorry

theorem problem2_eval : 
  -2^3 - (2 - 1.5) / (3/8) * abs (-6 - (-3)^2) = -28 := by
  sorry

end problem1_eval_problem2_eval_l238_238992


namespace fraction_smart_integers_divisible_by_11_l238_238399

def is_smart_integer (n : ℕ) : Prop :=
  n > 30 ∧ n < 130 ∧ n % 2 = 1 ∧ (n.digits.sum = 10)

def smart_integers : list ℕ :=
  list.filter is_smart_integer (list.range' 31 99)

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem fraction_smart_integers_divisible_by_11 :
  (list.count divisible_by_11 smart_integers) / (list.length smart_integers) = 1 / 2 :=
sorry

end fraction_smart_integers_divisible_by_11_l238_238399


namespace permutation_divisible_sum_l238_238034

theorem permutation_divisible_sum (n : ℕ) (h : 0 < n) :
  ∃ a : Fin n → ℕ, (∀ k : Fin (n-1), (a k.succ) ∣ (Finset.range k.succ).sum (λ i, a ⟨i, Nat.lt_of_lt_pred k.is_lt⟩)) ∧
  (∀ i j, a i = a j → i = j) ∧
  (∀ i, a i ∈ Finset.range n) :=
sorry

end permutation_divisible_sum_l238_238034


namespace max_remaining_numbers_l238_238804

def numbers (n : ℕ) : set ℕ := {k | 1 ≤ k ∧ k ≤ n}

def valid_subset (s : set ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → (a - b) ≠ 0 → ¬ (a - b) ∣ c

theorem max_remaining_numbers : ∃ s : set ℕ, s ⊆ numbers 235 ∧ valid_subset s ∧ card s = 118 := 
sorry

end max_remaining_numbers_l238_238804


namespace expected_num_matches_variance_num_matches_l238_238586

variable {N : ℕ} (I : Fin N → Prop) [Fintype N]

-- Define the indicator variable
def indicator (k : Fin N) : ℝ :=
if I k then 1 else 0

-- Define the expected value of indicator
def expected_indicator (k : Fin N) : ℝ :=
1 / N

-- Summing up all indicators to get the total number of matches
def S : ℝ :=
∑ k, indicator k

-- The expected number of matches E(S)
theorem expected_num_matches :
  expected_value S = 1 := 
sorry

-- Variance calculation
theorem variance_num_matches :
  variance S = 1 := 
sorry

end expected_num_matches_variance_num_matches_l238_238586


namespace smallest_n_for_sum_exceed_10_pow_5_l238_238530

def a₁ : ℕ := 9
def r : ℕ := 10
def S (n : ℕ) : ℕ := 5 * n^2 + 4 * n
def target_sum : ℕ := 10^5

theorem smallest_n_for_sum_exceed_10_pow_5 : 
  ∃ n : ℕ, S n > target_sum ∧ ∀ m < n, ¬(S m > target_sum) := 
sorry

end smallest_n_for_sum_exceed_10_pow_5_l238_238530


namespace total_pieces_l238_238980

def pieces_from_friend : ℕ := 123
def pieces_from_brother : ℕ := 136
def pieces_needed : ℕ := 117

theorem total_pieces :
  pieces_from_friend + pieces_from_brother + pieces_needed = 376 :=
by
  unfold pieces_from_friend pieces_from_brother pieces_needed
  sorry

end total_pieces_l238_238980


namespace sum_extremes_correct_l238_238494

-- Define the grid and the placement of numbers
def is_grid_valid (grid : list (list ℕ)) : Prop :=
  (∀ i j, 0 ≤ i ∧ i < 10 ∧ 0 ≤ j ∧ j < 10 → 
    grid.nth i >>= (λ row, row.nth j) ≠ none) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 100 → ∃ i j, 0 ≤ i ∧ i < 10 ∧ 0 ≤ j ∧ j < 10 ∧
    grid.nth i >>= (λ row, row.nth j) = some n)

-- Central placement and counterclockwise fill must form a valid grid
def is_counterclockwise_fill (grid : list (list ℕ)) : Prop :=
  is_grid_valid grid ∧
  (grid.nth 4 >>= (λ row, row.nth 5) = some 1)  -- 1 is at (fifth row, sixth column)

-- Numbers in the fifth row
def numbers_in_fifth_row (grid : list (list ℕ)) : list ℕ :=
  grid.nth 4 |>.toList |>.join

-- Sum of the greatest and least number in the fifth row
def sum_of_extremes (l : list ℕ) : ℕ :=
  l.maximum.getD 0 + l.minimum.getD 0

-- The formal problem statement
theorem sum_extremes_correct (grid : list (list ℕ)) (h : is_counterclockwise_fill grid) :
  sum_of_extremes (numbers_in_fifth_row grid) = 110 :=
by
  sorry

end sum_extremes_correct_l238_238494


namespace line_intersects_parabola_l238_238682

noncomputable def point (x y : ℝ) : Prop := True

theorem line_intersects_parabola (a p : ℝ) (h₁ : 0 < a) (h₂ : 0 < p)
  (h₃ : ∀ (P Q : (ℝ × ℝ)), (∃ (x : ℝ), P = (x, sqrt (2 * p * x)) ∧ Q = (x, -sqrt (2 * p * x)))
    → ∃ (k : ℝ), (1 / ((P.1 - a)^2 + P.2^2) + 1 / ((Q.1 - a)^2 + Q.2^2)) = k) :
  a = p :=
sorry

end line_intersects_parabola_l238_238682


namespace count_odd_expressions_l238_238217

theorem count_odd_expressions : 
  let exp1 := 1^2
  let exp2 := 2^3
  let exp3 := 3^4
  let exp4 := 4^5
  let exp5 := 5^6
  (if exp1 % 2 = 1 then 1 else 0) + 
  (if exp2 % 2 = 1 then 1 else 0) + 
  (if exp3 % 2 = 1 then 1 else 0) + 
  (if exp4 % 2 = 1 then 1 else 0) + 
  (if exp5 % 2 = 1 then 1 else 0) = 3 :=
by 
  sorry

end count_odd_expressions_l238_238217


namespace problem_solution_l238_238060

theorem problem_solution (p q r : ℝ) 
    (h1 : (p * r / (p + q) + q * p / (q + r) + r * q / (r + p)) = -8)
    (h2 : (q * r / (p + q) + r * p / (q + r) + p * q / (r + p)) = 9) 
    : (q / (p + q) + r / (q + r) + p / (r + p) = 10) := 
by
  sorry

end problem_solution_l238_238060


namespace coefficient_x_in_expansion_l238_238771

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
match k with
| 0 => 1
| _ => if k > n then 0 else (n * binom (n - 1) (k - 1)) / k

-- Theorem stating the coefficient of x in the expansion
theorem coefficient_x_in_expansion : 
  (∑ k in Finset.range 6, (if k = 3 
                          then (-2)^k * (binom 5 k : ℤ) 
                          else 0)) = -80 := 
sorry

end coefficient_x_in_expansion_l238_238771


namespace derivative_at_pi_over_4_l238_238346

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem derivative_at_pi_over_4 : (deriv f (Real.pi / 4)) = -2 :=
by
  sorry

end derivative_at_pi_over_4_l238_238346


namespace ellipse_equation_l238_238517

-- Define the conditions
variables (b : ℝ) (c : ℝ) (b_cond : b = 3) (c_cond : c = 4)

-- Define the relationship between a, b, and c
def a : ℝ := sqrt (b^2 + c^2)

-- Theorem statement
theorem ellipse_equation : b = 3 → c = 4 → a = 5 ∧ 
                          ellipse_equation := ∀ x y, y^2 / 25 + x^2 / 9 = 1 :=
by
  sorry

end ellipse_equation_l238_238517


namespace sum_of_possible_100th_terms_l238_238617

theorem sum_of_possible_100th_terms : 
  ∃ d ∈ {1, 2, 4}, let a₁ := 7 in 
  let a₁ := 7 + 99 * d in 
  a₁ = 714 :=
by
  sorry

end sum_of_possible_100th_terms_l238_238617


namespace find_k_values_l238_238376

theorem find_k_values (a b k : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b % a = 0) 
  (h₄ : ∀ (m : ℤ), (a : ℤ) = k * (a : ℤ) + m ∧ (8 * (b : ℤ)) = k * (b : ℤ) + m) :
  k = 9 ∨ k = 15 :=
by
  { sorry }

end find_k_values_l238_238376


namespace number_of_donut_selections_l238_238501

-- Definitions for the problem
def g : ℕ := sorry
def c : ℕ := sorry
def p : ℕ := sorry

-- Condition: Pat wants to buy four donuts from three types
def equation : Prop := g + c + p = 4

-- Question: Prove the number of different selections possible
theorem number_of_donut_selections : (∃ n, n = 15) := 
by 
  -- Use combinatorial method to establish this
  sorry

end number_of_donut_selections_l238_238501


namespace num_true_propositions_is_zero_l238_238137

theorem num_true_propositions_is_zero :
  (∃ (p1 p2 p3 p4 : Prop),
   p1 = (∀ (temperature velocity displacement work : ℝ → ℝ),
           (temperature = 0 ∧ work = 0) → (velocity ≠ 0 ∧ displacement ≠ 0)) ∧
   p2 = (∀ (zero_vector : ℝ × ℝ), zero_vector = (0, 0) → ∀ direction : ℝ × ℝ, direction = (1, 0) → direction ≠ zero_vector) ∧
   p3 = (∀ (v : ℝ × ℝ), v ≠ (0, 0) → (fst v)^2 + (snd v)^2 > 0) ∧
   p4 = (∃ (x_axis y_axis : ℝ × ℝ), (x_axis = (1, 0) ∧ y_axis = (0, 1)) →
            (fst x_axis)^2 + (snd x_axis)^2 = 1 ∧ (fst y_axis)^2 + (snd y_axis)^2 = 1) →
  ¬ p1 ∧ ¬ p2 ∧ ¬ p3 ∧ ¬ p4) →
  0 = 0 := sorry

end num_true_propositions_is_zero_l238_238137


namespace range_of_z_l238_238669

theorem range_of_z (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 1) : 
  let z := 2 * x - 3 * y 
  in -5 ≤ z ∧ z ≤ 4 := 
by
  sorry

end range_of_z_l238_238669


namespace dynastic_vertex_bound_l238_238155

open Nat

def is_dynastic (k : ℕ) (forest : List (List ℕ)) (v : ℕ) : Prop :=
  v < forest.length ∧ forest[v].length = 2 ∧ 
  ∀ u ∈ forest[v], ∃ extended : List ℕ, extended.length ≥ k ∧ 
  ∀ w ∈ extended, (w < forest.length ∧ forest[w].length = 2)

def number_of_dynastic_vertices (k : ℕ) (forest : List (List ℕ)) : ℕ :=
  forest.countp (is_dynastic k forest)

theorem dynastic_vertex_bound (n k : ℕ) (forest : List (List ℕ)) :
  forest.length = n →
  number_of_dynastic_vertices k forest ≤ n / (k + 2) :=
by
  sorry

end dynastic_vertex_bound_l238_238155


namespace billy_sleep_total_l238_238744

theorem billy_sleep_total
  (h₁ : ∀ n : ℕ, n = 1 → ∃ h : ℕ, h = 6)
  (h₂ : ∀ n : ℕ, n = 2 → ∃ h : ℕ, h = (6 + 2))
  (h₃ : ∀ n : ℕ, n = 3 → ∃ h : ℕ, h = ((6 + 2) / 2))
  (h₄ : ∀ n : ℕ, n = 4 → ∃ h : ℕ, h = (((6 + 2) / 2) * 3)) :
  ∑ n in {1, 2, 3, 4}, (classical.some (h₁ n 1) + classical.some (h₂ n 2) + classical.some (h₃ n 3) + classical.some (h₄ n 4)) = 30 :=
by sorry

end billy_sleep_total_l238_238744


namespace polynomial_identity_l238_238403

theorem polynomial_identity (a : ℝ) (h₁ : a^5 + 5 * a^4 + 10 * a^3 + 3 * a^2 - 9 * a - 6 = 0) (h₂ : a ≠ -1) : (a + 1)^3 = 7 :=
sorry

end polynomial_identity_l238_238403


namespace distance_to_town_l238_238979

theorem distance_to_town (d : ℝ) : (d < 8) → (d > 7) → (d > 6) → (d < 5) → false → d ∈ set.Ioo 7 8 :=
by
  intros h1 h2 h3 h4 h5
  exfalso
  exact h5

end distance_to_town_l238_238979


namespace leonard_younger_than_nina_by_4_l238_238457

variable (L N J : ℕ)

-- Conditions based on conditions from the problem
axiom h1 : L = 6
axiom h2 : N = 1 / 2 * J
axiom h3 : L + N + J = 36

-- Statement to prove
theorem leonard_younger_than_nina_by_4 : N - L = 4 :=
by 
  sorry

end leonard_younger_than_nina_by_4_l238_238457


namespace find_4a_plus_8b_l238_238408

def quadratic_equation_x_solution (a b : ℝ) : Prop :=
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0

theorem find_4a_plus_8b (a b : ℝ) (h : quadratic_equation_x_solution a b) : 4 * a + 8 * b = -4 := 
  by
    sorry

end find_4a_plus_8b_l238_238408


namespace translated_point_on_sin_graph_l238_238159

theorem translated_point_on_sin_graph (s : ℝ) (t : ℝ) (h1 : s > 0) 
  (h2 : t = sin (2 * (π / 4) - π / 3)) 
  (h3 : sin (2 * (π / 4 - s)) = t) 
  : t = 1 / 2 ∧ s = π / 6 :=
  sorry

end translated_point_on_sin_graph_l238_238159


namespace unique_n_points_with_area_condition_l238_238314

theorem unique_n_points_with_area_condition :
  ∃ n : ℕ, n > 3 ∧ (∀ (A : Fin n → ℝ × ℝ) (r : Fin n → ℝ),
  (∀ i j k : Fin n, i < j → j < k → (area (A i) (A j) (A k) = r i + r j + r k)) ↔ n = 4) :=
by
  sorry

-- Helper function to calculate the area of a triangle given its three vertices
def area (A₁ A₂ A₃ : ℝ × ℝ) : ℝ :=
  |(A₁.1 * (A₂.2 - A₃.2) + A₂.1 * (A₃.2 - A₁.2) + A₃.1 * (A₁.2 - A₂.2)) / 2|

end unique_n_points_with_area_condition_l238_238314


namespace num_quadruples_l238_238470

/-
  Prove that the number of ordered quadruples (x1, x2, x3, x4)
  where x1 is a squared odd integer minus 1, x2, x3, x4 are positive odd integers,
  and the sum of these integers is 100 is 4557.
-/
theorem num_quadruples (n : ℕ) 
  (h₁ : ∃ y1 : ℕ, x1 = (2 * y1 + 1)^2 - 1)
  (h₂ : ∀ i ∈ {2, 3, 4}, ∃ z : ℕ, x i = 2 * z - 1)
  (h₃ : ∑ i in {1, 2, 3, 4}, x i = 100) :
  n = 4557 := sorry

end num_quadruples_l238_238470


namespace number_of_supported_sets_is_1430_l238_238975

def is_supported_set (S : Set ℕ) : Prop :=
  0 ∈ S ∧ (∀ k ∈ S, k + 8 ∈ S ∧ k + 9 ∈ S)

theorem number_of_supported_sets_is_1430 :
  let supported_sets := {S : Set ℕ | is_supported_set S} in
  card supported_sets = 1430 := 
sorry

end number_of_supported_sets_is_1430_l238_238975


namespace maximum_value_N_27_l238_238099

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end maximum_value_N_27_l238_238099


namespace arithmetic_seq_general_term_l238_238373

/-- Given the sum of the first n terms of a sequence S_n = n^2 - 3n,
    prove that the general term a_n is 2n - 4. -/
theorem arithmetic_seq_general_term (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S n = n^2 - 3 * n) →
  (∀ n, a n = S n - S (n - 1)) →
  (a 1 = S 1) →
  (∀ n, a n = 2 * n - 4) :=
begin
  intro hS,
  intro hR,
  intro hA1,
  sorry
end

end arithmetic_seq_general_term_l238_238373


namespace max_remained_numbers_l238_238852

theorem max_remained_numbers (S : Finset ℕ) (hSubset : S ⊆ Finset.range 236)
  (hCondition : ∀ a b c ∈ S, a ≠ b → a ≠ c → b ≠ c → ¬(b - a ∣ c)) : S.card ≤ 118 := 
sorry

end max_remained_numbers_l238_238852


namespace intersection_complement_eq_l238_238724

open Set

-- Definitions from the problem conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | x ≥ 0}
def C_U_N : Set ℝ := {x | x < 0}

-- Statement of the proof problem
theorem intersection_complement_eq : M ∩ C_U_N = {x | -1 ≤ x ∧ x < 0} :=
by
  sorry

end intersection_complement_eq_l238_238724


namespace sum_expression_equals_constant_l238_238733

theorem sum_expression_equals_constant :
  (∑ n in finset.range 1500, (n + 1) * (1500 - n)) = 1500 * 751 * 501 := 
by
  sorry

end sum_expression_equals_constant_l238_238733


namespace max_remaining_numbers_l238_238817

theorem max_remaining_numbers : 
  ∃ (S ⊆ {n | 1 ≤ n ∧ n ≤ 235}), (∀ x ∈ S, ∀ y ∈ S, x ≠ y → ¬(x ∣ (y - x))) ∧ card S = 118 :=
by
  sorry

end max_remaining_numbers_l238_238817


namespace graph_passes_through_point_l238_238883

theorem graph_passes_through_point :
  ∀ (a : ℝ), (0 < a) ∧ (a ≠ 1) → (∃ x y : ℝ, x = 0 ∧ y = 3 ∧ y = a^x + 2) :=
by
  intro a
  assume h : (0 < a) ∧ (a ≠ 1)
  use [0, 3]
  split
  · rfl
  split
  · rfl
  · sorry

end graph_passes_through_point_l238_238883


namespace derivative_of_option_C_is_odd_l238_238608

-- Definitions of the functions
def f_A (x : ℝ) : ℝ := Math.sin x
def f_B (x : ℝ) : ℝ := Math.exp x
def f_C (x : ℝ) : ℝ := Math.cos x - 1/2
def f_D (x : ℝ) : ℝ := Math.log x

-- Definition of odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- The theorem to prove
theorem derivative_of_option_C_is_odd :
  is_odd (fun x => -Math.sin x) :=
sorry

end derivative_of_option_C_is_odd_l238_238608


namespace parabola_equation_constant_expression_l238_238722

-- Given conditions
variables {p m : ℝ} (hp : 0 < p)

-- Parabola Equation
def parabola (x y : ℝ) := y^2 = 2 * p * x
-- Line Equation
def line (x y : ℝ) := x = m * y + 3 
-- Points A and B: Intersection of parabola and line
def intersection (x y : ℝ) := parabola x y ∧ line x y

-- Dot product condition
variable {A B O : ℝ × ℝ}
variable (Hd : (A.1 * B.1 + A.2 * B.2) = 6)

-- Point C
def point_C := (-3 : ℝ, 0 : ℝ)
-- Slopes
variables {k1 k2 : ℝ}
def slope_C (P : ℝ × ℝ) (k : ℝ) := k = (P.2 - point_C.2) / (P.1 - point_C.1)
variable (Hk1 : slope_C A k1)
variable (Hk2 : slope_C B k2)

-- Theorem 1: Equation of Parabola
theorem parabola_equation : parabola = (λ x y, y^2 = x) := sorry

-- Theorem 2: Constant expression
theorem constant_expression : (1 / k1^2) + (1 / k2^2) - 2 * m^2 = 24 := sorry

end parabola_equation_constant_expression_l238_238722


namespace handshake_problem_l238_238953

noncomputable def number_of_handshakes (n : ℕ) : ℕ :=
  n.choose 2

theorem handshake_problem : number_of_handshakes 25 = 300 := 
  by
  sorry

end handshake_problem_l238_238953


namespace max_rabbits_l238_238117

theorem max_rabbits (N : ℕ) (h1 : ∀ k, k = N → k = 27 → true)
    (h2 : ∀ n_l : ℕ, n_l = 13 → n_l <= N)
    (h3 : ∀ n_j : ℕ, n_j = 17 → n_j <= N)
    (h4 : ∀ n_both : ℕ, n_both >= 3 → true) :
  N <= 27 :=
begin
  sorry
end

end max_rabbits_l238_238117


namespace max_rabbits_l238_238077

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l238_238077


namespace average_growth_rate_le_max_growth_rate_l238_238989

variable (P : ℝ) (a : ℝ) (b : ℝ) (x : ℝ)

theorem average_growth_rate_le_max_growth_rate (h : (1 + x)^2 = (1 + a) * (1 + b)) :
  x ≤ max a b := 
sorry

end average_growth_rate_le_max_growth_rate_l238_238989


namespace face_sum_solution_exists_l238_238870

noncomputable def face_sum_problem (a b c d e f : ℕ) : Prop :=
  let v := a + d
  in (abc + aec + abf + aef + dbc + dec + dbf + def = 2002 ∧ v = 22) →
     (a + d) + (b + e) + (c + f) = 42

theorem face_sum_solution_exists : ∃ (a b c d e f : ℕ), face_sum_problem a b c d e f :=
sorry

end face_sum_solution_exists_l238_238870


namespace smallest_sum_is_4_9_l238_238997

theorem smallest_sum_is_4_9 :
  min
    (min
      (min
        (min (1/3 + 1/4) (1/3 + 1/5))
        (min (1/3 + 1/6) (1/3 + 1/7)))
      (1/3 + 1/9)) = 4/9 :=
  by sorry

end smallest_sum_is_4_9_l238_238997


namespace quadratic_roots_difference_squared_l238_238735

theorem quadratic_roots_difference_squared :
  let α β : ℝ in
  (∀ x : ℝ, x^2 - 3 * x + 2 = 0 → x = α ∨ x = β) → (α - β) * (α - β) = 1 :=
by
  sorry

end quadratic_roots_difference_squared_l238_238735


namespace sum_of_common_ratios_l238_238469

variable {k s t : ℝ}

noncomputable def a₂ := k * s
noncomputable def a₃ := k * s^2
noncomputable def b₂ := k * t
noncomputable def b₃ := k * t^2

theorem sum_of_common_ratios (h1 : a₃ - b₃ = 5 * (a₂ - b₂))
(h2 : k ≠ 0) (h3 : s ≠ t) : s + t = 5 :=
by sorry

end sum_of_common_ratios_l238_238469


namespace maximum_numbers_up_to_235_l238_238828

def max_remaining_numbers : ℕ := 118

theorem maximum_numbers_up_to_235 (numbers : set ℕ) (h₁ : ∀ n ∈ numbers, n ≤ 235)
  (h₂ : ∀ a b ∈ numbers, a ≠ b → ¬ (a - b).abs ∣ a) :
  numbers.card ≤ max_remaining_numbers :=
sorry

end maximum_numbers_up_to_235_l238_238828


namespace pie_division_min_pieces_l238_238884

-- Define the problem as a Lean statement
theorem pie_division_min_pieces : ∃ n : ℕ, (∀ m ∈ {5, 7}, n % m = 0) ∧ n = 11 :=
by
  use 11
  split
  -- Prove for 5
  { intro m
    intro hm
    cases hm
    -- m = 5
    { exact Nat.mod_eq_zero_of_dvd (Nat.dvd_trans (Nat.dvd_refl 11) (Nat.dvd_of_mem_divisors hm)) }
    -- m = 7
    { exact Nat.mod_eq_zero_of_dvd (Nat.dvd_trans (Nat.dvd_refl 11) (Nat.dvd_of_mem_divisors hm)) }
    -- Impossible, there are only 5 and 7
    contradiction }
  -- Prove n = 11
  exact rfl

end pie_division_min_pieces_l238_238884


namespace min_value_of_f_on_0_1_max_value_of_f_on_0_1_range_of_a_l238_238384

section Part1

variable {f : ℝ → ℝ} (a : ℝ) (ha : a = 3)
noncomputable def f := λ x : ℝ, a * x^2 - 2 * x + 1

theorem min_value_of_f_on_0_1 : infi (f 3 '' set.Icc (0 : ℝ) 1) = 2 / 3 := sorry

theorem max_value_of_f_on_0_1 : supr (f 3 '' set.Icc (0 : ℝ) 1) = 2 := sorry

end Part1

section Part2

variable {f : ℝ → ℝ} {g : ℝ → ℝ} (a : ℝ)
noncomputable def f := λ x : ℝ, a * x^2 - 2 * x + 1
noncomputable def g := λ x : ℝ, x + 1 / x

theorem range_of_a :
  (∀ x₁ ∈ set.Ioo 2 3, ∃ x₂ ∈ set.Icc (1 : ℝ) (2 : ℝ), f a x₁ < g x₂) ↔ a ∈ set.Icc (0 : ℝ) (5 / 6) := sorry

end Part2

end min_value_of_f_on_0_1_max_value_of_f_on_0_1_range_of_a_l238_238384


namespace sum_digits_exp_grows_without_bound_l238_238014

-- Definition of sum of digits function
def sum_digits (n : ℕ) : ℕ := 
  if n == 0 then 0 
  else n % 10 + sum_digits (n / 10)

theorem sum_digits_exp_grows_without_bound :
  ∀ᶠ n in Filter.atTop, sum_digits (2 ^ n) > n :=
by 
  sorry

end sum_digits_exp_grows_without_bound_l238_238014


namespace billy_sleep_total_l238_238748

def billy_sleep : Prop :=
  let first_night := 6
  let second_night := first_night + 2
  let third_night := second_night / 2
  let fourth_night := third_night * 3
  first_night + second_night + third_night + fourth_night = 30

theorem billy_sleep_total : billy_sleep := by
  sorry

end billy_sleep_total_l238_238748


namespace trapezium_area_l238_238773

theorem trapezium_area 
    (PQRS : Type)
    (P Q R S : PQRS)
    (angle_PQR : ∠PQR = 90)
    (angle_QRS_gt_90 : ∠QRS > 90)
    (diag_SQ : SQ = 24)
    (bisector_S : is_bisector SQ ∠S)
    (dist_R_to_QS : dist R (line QS) = 5)
    : area PQRS = (27420 / 169) :=
sorry

end trapezium_area_l238_238773


namespace probability_at_least_one_six_l238_238218

open ProbabilityTheory

/-- When two fair dice are rolled simultaneously, given that the numbers on both dice are different, 
the probability that at least one die shows a 6 is 1/3. -/
theorem probability_at_least_one_six (dice : Finset (ℕ × ℕ)) :
  (∀ x ∈ dice, x.1 ≠ x.2) →
  (6 ∈ Finset.range 7) → 
  prob ((λ x : ℕ × ℕ, x.1 ≠ x.2) ∧ (λ x : ℕ × ℕ, x.1 = 6 ∨ x.2 = 6)) dice = 1 / 3 :=
by
  -- convert question conditions and correct answer
  sorry

end probability_at_least_one_six_l238_238218


namespace evaluate_expression_at_two_l238_238930

theorem evaluate_expression_at_two : (2 * (2:ℝ)^2 - 3 * 2 + 4) = 6 := by
  sorry

end evaluate_expression_at_two_l238_238930


namespace maximum_rabbits_condition_l238_238106

-- Define the conditions and constraints
variables {N : ℕ}
variables (total_rabbits long_ears jump_far : ℕ)
variables (at_least_three_with_both : Prop)

-- State the conditions with exact values and assumptions
def conditions := 
  total_rabbits = N ∧
  long_ears = 13 ∧
  jump_far = 17 ∧
  at_least_three_with_both = (∃ a b c : ℕ, a >= 3 ∧ b = (long_ears - a) ∧ c = (jump_far - a))

-- State the theorem to be proved
theorem maximum_rabbits_condition :
  ∀ {N : ℕ}, conditions N long_ears jump_far at_least_three_with_both → N ≤ 27 :=
by sorry

end maximum_rabbits_condition_l238_238106


namespace minimize_quadratic_l238_238198

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end minimize_quadratic_l238_238198


namespace symmetric_center_of_shifted_sine_l238_238531

theorem symmetric_center_of_shifted_sine :
  ∀ x : ℝ, y = 2 * sin (2 * x) →
  shift_left (y) (π / 12) =
  2 * sin (2 * (x - π / 12)) →
  symmetric_center (shift_left (2 * sin (2 * x)) (π / 12)) = (-π / 12, 0) :=
sorry

end symmetric_center_of_shifted_sine_l238_238531


namespace tan_4356_equals_l238_238999

noncomputable def tan_periodic (x : ℝ) : ℝ := Real.tan x

theorem tan_4356_equals :
  tan_periodic 4356 = 0.7265 :=
by
  -- Condition: tangent function has a period of 360 degrees or 2π in radians.
  have h1 : 4356 % 360 = 36 := sorry,
  -- Thus, tan(4356°) = tan(36°)
  have h2 : tan_periodic 4356 = tan_periodic 36 := by
    rw [tan_periodic, Real.tan_periodic, h1],
  -- Use trigonometric tables or calculation for tan(36°)
  exact sorry

end tan_4356_equals_l238_238999


namespace find_x3_l238_238545

-- Define the problem conditions
noncomputable def f (x : ℝ) : ℝ := Real.log x
def x1 : ℝ := 1
def x2 : ℝ := 10
def y1 : ℝ := f x1
def y2 : ℝ := f x2

-- Point C trisects the line segment AB
def yc : ℝ := (2/3) * y1 + (1/3) * y2

-- Given condition for the vertical line through C intersecting the curve at E(x3, y3)
def y3 : ℝ := yc

-- Statement to prove
theorem find_x3 : ∃ x3 : ℝ, f x3 = y3 ∧ x3 = 10^(1/3) :=
by {
  -- Given conditions
  have h_y1 : y1 = Real.log 1 := rfl,
  have h_y2 : y2 = Real.log 10 := rfl,
  have h_y3 : y3 = (2/3)*y1 + (1/3)*y2 := rfl,
  have h_x1_pos : 0 < x1 := by norm_num,
  have h_x2_pos : x1 < x2 := by norm_num,

  -- Simplify expressions
  rw [h_y1, Real.log_one] at h_y3,
  rw [h_y2, mul_zero, add_zero] at h_y3,

  -- Calculate y3
  have hyc : y3 = (1/3)*Real.log 10 := by simp [h_y3, mul_one],

  -- Prove existence of x3 such that f x3 = y3
  use 10^(1/3),
  split; sorry
}

end find_x3_l238_238545


namespace jerry_total_logs_l238_238450

def logs_from_trees (p m w : Nat) : Nat :=
  80 * p + 60 * m + 100 * w

theorem jerry_total_logs :
  logs_from_trees 8 3 4 = 1220 :=
by
  -- Proof here
  sorry

end jerry_total_logs_l238_238450


namespace value_of_expression_l238_238734

theorem value_of_expression (x y z : ℝ) (hz : z ≠ 0) 
    (h1 : 2 * x - 3 * y - z = 0) 
    (h2 : x + 3 * y - 14 * z = 0) : 
    (x^2 + 3 * x * y) / (y^2 + z^2) = 7 := 
by 
  sorry

end value_of_expression_l238_238734


namespace third_quadrant_trig_identity_l238_238736

theorem third_quadrant_trig_identity (α : ℝ) (h : (π + 2 * Int.pi * k : ℝ) < α ∧ α < (3 * π / 2 + 2 * Int.pi * k : ℝ)) :
  let y := (abs (sin (α / 2)) / sin (α / 2) + abs (cos (α / 2)) / cos (α / 2)) in
  y = 0 :=
by
  sorry

end third_quadrant_trig_identity_l238_238736


namespace chuck_accessible_area_l238_238998

noncomputable def accessible_area (leash_length : ℝ) (total_angle : ℝ) (restricted_area : ℝ) : ℝ :=
  (total_angle / 360 * π * (leash_length ^ 2)) - restricted_area

theorem chuck_accessible_area : accessible_area 5 270 (π / 4) = 18.5 * π := by
  sorry

end chuck_accessible_area_l238_238998


namespace compare_sizes_negative_fractions_l238_238633

-- Definitions based on the conditions
def abs (x : ℝ) : ℝ := if x < 0 then -x else x

lemma compare_fractions : abs (-3/4) = 3/4 ∧ abs (-3/5) = 3/5 ∧ 3/4 > 3/5 :=
by 
  have h1 : abs (-3/4) = 3/4, from if_pos (by norm_num),
  have h2 : abs (-3/5) = 3/5, from if_pos (by norm_num),
  have h3 : 3/4 > 3/5, from by norm_num
  exact ⟨h1, h2, h3⟩

theorem compare_sizes_negative_fractions : -3/4 < -3/5 :=
by 
  have h : compare_fractions := compare_fractions,
  rw [abs, abs] at h,
  have h_pos : 3/4 > 3/5 := h.2.2,
  exact neg_lt_neg_iff.mpr h_pos

end compare_sizes_negative_fractions_l238_238633


namespace minimize_f_at_3_l238_238212

-- Define the quadratic function f(x) = 3x^2 - 18x + 7
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

-- The theorem stating that f(x) attains its minimum when x = 3
theorem minimize_f_at_3 : ∀ x : ℝ, f(x) ≥ f(3) := 
by 
  sorry

end minimize_f_at_3_l238_238212


namespace max_m_l238_238416

noncomputable def f (x a : ℝ) : ℝ := 2 ^ |x + a|

theorem max_m (a m : ℝ) (H1 : ∀ x, f (3 + x) a = f (3 - x) a) 
(H2 : ∀ x y, x ≤ y → y ≤ m → f x a ≥ f y a) : 
  m = 3 :=
by
  sorry

end max_m_l238_238416


namespace coeff_sum_eq_minus_243_l238_238411

theorem coeff_sum_eq_minus_243 (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x y : ℝ, (x - 2 * y) ^ 5 = a * (x + 2 * y) ^ 5 + a₁ * (x + 2 * y)^4 * y + a₂ * (x + 2 * y)^3 * y^2 
             + a₃ * (x + 2 * y)^2 * y^3 + a₄ * (x + 2 * y) * y^4 + a₅ * y^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = -243 :=
by
  intros h
  sorry

end coeff_sum_eq_minus_243_l238_238411


namespace sum_of_circle_areas_l238_238601

theorem sum_of_circle_areas (a b c: ℝ)
  (h1: a + b = 6)
  (h2: b + c = 8)
  (h3: a + c = 10) :
  π * a^2 + π * b^2 + π * c^2 = 56 * π := 
by
  sorry

end sum_of_circle_areas_l238_238601


namespace even_number_of_mixed_1x2_rectangles_l238_238571

theorem even_number_of_mixed_1x2_rectangles
  (m n : ℕ)
  (is_tiled : ∀ (x y : ℕ), (x < m) ∧ (y < n) → (exists (t : ℕ), (t = 2 ∧ (∀ i j, (i < 2) ∧ (j < 2) → (x + i < m ∧ y + j < n))) ∨ (t = 3 ∧ (∀ i j, (i < 1) ∧ (j < 3) → (x + i < m ∧ y + j < n)) ∨ (t = 3 ∧ (∀ i j, (i < 3) ∧ (j < 1) → (x + i < m ∧ y + j < n))))))
  : Even (number_of_mixed_1x2_rectangles m n is_tiled) := sorry

end even_number_of_mixed_1x2_rectangles_l238_238571


namespace maximum_numbers_up_to_235_l238_238829

def max_remaining_numbers : ℕ := 118

theorem maximum_numbers_up_to_235 (numbers : set ℕ) (h₁ : ∀ n ∈ numbers, n ≤ 235)
  (h₂ : ∀ a b ∈ numbers, a ≠ b → ¬ (a - b).abs ∣ a) :
  numbers.card ≤ max_remaining_numbers :=
sorry

end maximum_numbers_up_to_235_l238_238829


namespace john_average_speed_l238_238779

theorem john_average_speed :
  let total_distance := 345
  let first_day_morning_time := (4 + 10 / 60)
  let first_day_afternoon_time := 2
  let second_day_time := 5
  let total_time := first_day_morning_time + first_day_afternoon_time + second_day_time
  (total_distance / total_time) ≈ 30.97 := by
{
  let total_distance := 345
  let first_day_morning_time := 4 + 10 / 60
  let first_day_afternoon_time := 2
  let second_day_time := 5
  let total_time := first_day_morning_time + first_day_afternoon_time + second_day_time
  let avg_speed := total_distance / total_time
  show avg_speed ≈ 30.97,
  sorry
}

end john_average_speed_l238_238779


namespace gcd_547_323_l238_238551

theorem gcd_547_323 : Nat.gcd 547 323 = 1 := 
by
  sorry

end gcd_547_323_l238_238551


namespace output_when_input_is_3_l238_238215

theorem output_when_input_is_3 : 
  ∀ x y : ℤ, x = 3 ∧ y = 3 * x^2 - 5 * x → (x, y) = (3, 12) :=
by
  intros x y h,
  cases h,
  sorry

end output_when_input_is_3_l238_238215


namespace imaginary_part_of_z_l238_238678

-- Definition based on the problem condition
def z_condition (z : ℂ) : Prop := z * (2 + complex.i) = 1

-- Statement of the proof problem
theorem imaginary_part_of_z (z : ℂ) (h : z_condition z) : complex.im z = -1/5 :=
sorry

end imaginary_part_of_z_l238_238678


namespace maximum_rabbits_l238_238089

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l238_238089


namespace minimize_f_at_3_l238_238209

-- Define the quadratic function f(x) = 3x^2 - 18x + 7
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

-- The theorem stating that f(x) attains its minimum when x = 3
theorem minimize_f_at_3 : ∀ x : ℝ, f(x) ≥ f(3) := 
by 
  sorry

end minimize_f_at_3_l238_238209


namespace solve_for_x_l238_238404

theorem solve_for_x (x : ℤ) (h : 3 * x = 2 * x + 6) : x = 6 := by
  sorry

end solve_for_x_l238_238404


namespace midpoint_passing_l238_238433

theorem midpoint_passing (ABC A' B' D P Q : Type) [acute_triangle ABC]
  (A'B' : line) (Lambda_A' : line) (BD : line) (AD : line)
  (ha : altitude ABC A') (hb : altitude ABC B')
  (arc_condition : D ∈ circumcircle_arc_ (ACB ABC))
  (BD_meets_Lambda_A'_at_P : intersect BD Lambda_A' == P)
  (AD_meets_BB'_at_Q : intersect AD (altitude_line ABC B') == Q)
  (segments_line : line.passes_through A'B' (midpoint P Q)) :
  line.passes_through A'B' (midpoint P Q) :=
sorry

end midpoint_passing_l238_238433


namespace poor_horse_seventh_day_good_horse_arrives_qi_ninth_day_horses_meet_ninth_day_good_horse_travelled_when_meet_l238_238441

noncomputable section

-- We start by defining the sequences for the good horse and poor horse
def good_horse_distance (n : ℕ) : ℕ := 103 + 13 * (n - 1)
def poor_horse_distance (n : ℕ) : ℕ := 97 - (0.5 : ℚ) * (n - 1)

-- Sum of first n terms of the given arithmetic sequence
def good_horse_total_distance (n : ℕ) : ℕ := n * 103 + (7 * (n - 1) * 13) / 2

-- Given the conditions, we need to prove:
theorem poor_horse_seventh_day : poor_horse_distance 7 = 94 :=
by
  rw [poor_horse_distance]
  norm_num

theorem good_horse_arrives_qi_ninth_day : (∑ i in range 9, good_horse_distance i) = 1125 :=
sorry

theorem horses_meet_ninth_day :
  (∑ i in range 9, good_horse_distance i) + (∑ i in range 9, poor_horse_distance i) = 1125 * 2 :=
sorry

theorem good_horse_travelled_when_meet : (∑ i in range 9, good_horse_distance i) = 1395 :=
sorry

end poor_horse_seventh_day_good_horse_arrives_qi_ninth_day_horses_meet_ninth_day_good_horse_travelled_when_meet_l238_238441


namespace trapezoid_planar_l238_238244

-- Definitions and conditions based on the problem statement
variable {α β : Type} [plane : affine_space α]
variable {Triangle : set α}
def non_collinear_points (pts : set α) : Prop := ¬∃ (l : line α), pts ⊆ l

-- The trapezoid has one pair of parallel sides and lies in a plane
def is_trapezoid (quad : set α) : Prop :=
  ∃ a b c d : α, quad = {a, b, c, d} ∧
  plane.line_parallel (affine_space.mk_line a b) (affine_space.mk_line c d) ∧ 
  non_collinear_points {a, b, c, d}

-- Given a quadrilateral (four points), it can be planar or spatial
def planar_or_spatial (quad : set α) : Prop :=
  ∃ a b c d : α, quad = {a, b, c, d} ∧ (non_collinear_points {a, b, c, d})

-- Lean statement to prove the problem
theorem trapezoid_planar (quad : set α) :
  is_trapezoid quad → planar_or_spatial quad := 
sorry

end trapezoid_planar_l238_238244


namespace max_remained_numbers_l238_238851

theorem max_remained_numbers (S : Finset ℕ) (hSubset : S ⊆ Finset.range 236)
  (hCondition : ∀ a b c ∈ S, a ≠ b → a ≠ c → b ≠ c → ¬(b - a ∣ c)) : S.card ≤ 118 := 
sorry

end max_remained_numbers_l238_238851


namespace equilateral_triangle_l238_238474

-- Definitions and conditions
variables {A B C P D E F : Type}
variables [triangle A B C] [point_inside_triangle P A B C]
variables [meet_opposite_sides AP BP CP D E F A B C]
variables [bicentric_quadrilateral PDCE PEAF PFBD]

-- Theorem statement
theorem equilateral_triangle (h₁ : triangle A B C) 
                             (h₂: point_inside_triangle P A B C)
                             (h₃: meet_opposite_sides AP BP CP D E F A B C)
                             (h₄: bicentric_quadrilateral PDCE PEAF PFBD) :
  equilateral A B C :=
sorry

end equilateral_triangle_l238_238474


namespace maximum_numbers_up_to_235_l238_238825

def max_remaining_numbers : ℕ := 118

theorem maximum_numbers_up_to_235 (numbers : set ℕ) (h₁ : ∀ n ∈ numbers, n ≤ 235)
  (h₂ : ∀ a b ∈ numbers, a ≠ b → ¬ (a - b).abs ∣ a) :
  numbers.card ≤ max_remaining_numbers :=
sorry

end maximum_numbers_up_to_235_l238_238825


namespace length_of_box_l238_238523

theorem length_of_box (v : ℝ) (w : ℝ) (h : ℝ) (l : ℝ) (conversion_factor : ℝ) (v_gallons : ℝ)
  (h_inch : ℝ) (conversion_inches_feet : ℝ) :
  v_gallons / conversion_factor = v → 
  h_inch / conversion_inches_feet = h →
  v = l * w * h →
  w = 25 →
  v_gallons = 4687.5 →
  conversion_factor = 7.5 →
  h_inch = 6 →
  conversion_inches_feet = 12 →
  l = 50 :=
by
  sorry

end length_of_box_l238_238523


namespace simplify_fraction_l238_238042

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l238_238042


namespace problem_axiom_example_l238_238936

axiom axiom_three : ∀ (A B C : Type) [plane A] [plane B] [plane C] (h : ¬ collinear A B C), ∃! p : plane A, B, C ⊆ p

def is_axiom (s : Type → Prop) : Prop := sorry

theorem problem_axiom_example :
  is_axiom (λ A B C => ∀ (h : ¬ collinear A B C), ∃! p : plane A, B, C ⊆ p) :=
by
  sorry

end problem_axiom_example_l238_238936


namespace n_exists_smallest_l238_238467

open Int

theorem n_exists_smallest
  (a b n : ℤ)
  (h₁ : a ≡ 23 [MOD 60])
  (h₂ : b ≡ 95 [MOD 60])
  (range_n : 150 ≤ n ∧ n ≤ 191)
  (h₃ : (a - b) ≡ n [MOD 60]) :
  n = 168 :=
sorry

end n_exists_smallest_l238_238467


namespace trapezoid_segment_ratio_l238_238273

theorem trapezoid_segment_ratio (s l : ℝ) (h₁ : 3 * s + l = 1) (h₂ : 2 * l + 6 * s = 2) :
  l = 2 * s :=
by
  sorry

end trapezoid_segment_ratio_l238_238273


namespace find_correct_equation_l238_238962

variables (x : ℝ)

def cost_per_piece_A := 7800 / (1.5 * x)
def cost_per_piece_B := 6400 / x

theorem find_correct_equation (h1 : 7800 / (1.5 * x) + 30 = 6400 / x) :
  cost_per_piece_A x + 30 = cost_per_piece_B x :=
by
  sorry

end find_correct_equation_l238_238962


namespace avg_weight_section_B_is_80_l238_238148

def avg_weight_section_b 
    (num_students_A : ℕ) (num_students_B : ℕ) 
    (avg_weight_A : ℕ) (avg_weight_total : ℕ) : ℕ :=
    ∃ (W_B : ℕ), 
    num_students_A = 50 ∧ 
    num_students_B = 50 ∧ 
    avg_weight_A = 60 ∧ 
    avg_weight_total = 70 ∧ 
    100 * avg_weight_total = (num_students_A * avg_weight_A) + (num_students_B * W_B)

theorem avg_weight_section_B_is_80 : avg_weight_section_b 50 50 60 70 :=
begin
  use 80,
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  exact rfl,
  norm_num,
end

end avg_weight_section_B_is_80_l238_238148


namespace minimize_quadratic_function_l238_238180

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end minimize_quadratic_function_l238_238180


namespace minimize_f_at_3_l238_238208

-- Define the quadratic function f(x) = 3x^2 - 18x + 7
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

-- The theorem stating that f(x) attains its minimum when x = 3
theorem minimize_f_at_3 : ∀ x : ℝ, f(x) ≥ f(3) := 
by 
  sorry

end minimize_f_at_3_l238_238208


namespace vector_line_equation_l238_238141

open Real

noncomputable def vector_projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let numer := (u.1 * v.1 + u.2 * v.2)
  let denom := (v.1 * v.1 + v.2 * v.2)
  (numer * v.1 / denom, numer * v.2 / denom)

theorem vector_line_equation (x y : ℝ) :
  vector_projection (x, y) (3, 4) = (-3, -4) → 
  y = -3 / 4 * x - 25 / 4 :=
  sorry

end vector_line_equation_l238_238141


namespace max_min_values_l238_238719

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem max_min_values (a : ℝ) : 
  (∀ x ∈ set.Icc (0:ℝ) a, f x ≤ 3 ∧ 
  (∃ x', x' ∈ set.Icc (0:ℝ) a ∧ f x' = 3)) ∧ 
  (∃ x'', x'' ∈ set.Icc (0:ℝ) a ∧ f x'' = 2) ↔ 
  1 ≤ a ∧ a ≤ 2 := 
by
  sorry

end max_min_values_l238_238719


namespace minimize_f_l238_238188

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l238_238188


namespace max_rabbits_l238_238076

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l238_238076


namespace percentage_error_in_area_l238_238616

theorem percentage_error_in_area {s : ℝ} (H : s > 0) :
  let s' := 1.04 * s
      A := s^2
      A' := (1.04 * s)^2
      E := (A' - A)
  in (E / A) * 100 = 8.16 := 
by
  let s' := 1.04 * s
  let A := s^2
  let A' := (1.04 * s)^2
  let E := A' - A
  have h1 : E = 0.0816 * s^2 := sorry
  have h2 : E / A = 0.0816 := sorry
  rw[h1, h2]
  norm_num
  sorry

end percentage_error_in_area_l238_238616


namespace prob_Dali_prints_in_row_l238_238800

theorem prob_Dali_prints_in_row:
  (∃ (n m : ℕ), n = 12 ∧ m = 4 ∧
  let total_pieces := n
  let dali_prints := m
  let other_pieces := total_pieces - dali_prints
  let total_arrangements := nat.factorial total_pieces
  let favorable_arrangements := nat.factorial other_pieces * nat.factorial dali_prints
  let probability := favorable_arrangements / total_arrangements
  probability = 1 / 55) :=
sorry

end prob_Dali_prints_in_row_l238_238800


namespace min_k_to_group_shirts_l238_238431

theorem min_k_to_group_shirts (W P : ℕ) (hW : W = 21) (hP : P = 21) : 
  ∃ k : ℕ, k = 10 ∧ (∀ (order : list bool), (count true order = W) 
  ∧ (count false order = P) → ∃ w_removed p_removed, w_removed = k 
  ∧ p_removed = k 
  ∧ let remaining := remove_n_from_list order true w_removed, 
    remaining' := remove_n_from_list remaining false p_removed in 
    grouped true remaining' 
  ∧ grouped false remaining') :=
sorry

-- Auxiliary definitions that might be necessary for the Lean 4 formalization
def count (b : bool) : list bool → ℕ
| [] := 0
| (x :: xs) := (if x = b then 1 else 0) + count b xs

def remove_n_from_list : list bool → bool → ℕ → list bool
| l _ 0 := l
| [] _ _ := []
| (x :: xs) b (n + 1) := if x = b 
                          then remove_n_from_list xs b n 
                          else x :: remove_n_from_list xs b (n + 1)

def grouped (b : bool) : list bool → Prop 
| [] := true
| (x :: xs) := if x = b 
                 then (match xs with 
                       | (y :: ys) := if y = b then grouped b xs else ys.first ≠ b
                       | [] := true
                       end)
                 else grouped b xs

end min_k_to_group_shirts_l238_238431


namespace find_a_n_find_T_n_find_lambda_q_and_c_n_l238_238687

section MathSequence

variable (a : ℕ → ℕ) (S : ℕ → ℕ) (b c : ℕ → ℕ) (λ q : ℝ)
variable (n : ℕ)

/-- Given a sequence {a_n} of positive terms such that the sum of the first n terms is S_n,
    where S_n = 1/4 a_n^2 + 1/2 a_n -/
axiom positive_terms : ∀ (n : ℕ), S n = 1 / 4 * (a n)^2 + 1 / 2 * (a n)

/-- (1) The sequence {a_n} such that a_1 = 2 and for n ≥ 2, a_n = 2n -/
theorem find_a_n : a 1 = 2 ∧ ∀ n, 2 ≤ n → a n = 2 * n := 
sorry

/-- (2) Define the sequence {b_n} as follows:
    b_n =   a_n, if n is odd
            b_{n/2}, if n is even -/
def b : ℕ → ℕ :=
  λ n, if n % 2 = 1 then a n else b (n / 2)

/-- Define c_n as follows: c_n = b_{2^n + 4} and let T_n be the sum of the first n terms of {c_n} -/
def c : ℕ → ℕ :=
  λ n, b (2^n + 4)

def T : ℕ → ℕ :=
  λ n, (finset.range n).sum (λ i, c (i + 1))

/-- The sum of the first n terms T_n is given by:
    T_n =   6, if n = 1
            8, if n = 2
            2^n + 2n, if n ≥ 3 -/
theorem find_T_n : T 1 = 6 ∧ T 2 = 8 ∧ ∀ n, 3 ≤ n → T n = 2^n + 2 * n := 
sorry

/-- (3) Given b_n = λ q^{a_n} + λ, and c_n = 3 + n + sum_{i=1}^{n} b_i, 
    prove that the sequence {c_n} forms a geometric progression
    for λ = -1 and q = sqrt(3 / 4). Find the general term c_n. -/
theorem find_lambda_q_and_c_n (λ : ℝ) (q : ℝ) :
  λ = -1 ∧ q = real.sqrt (3 / 4) →
  ∀ n, c n = 4 * (3 / 4)^(n + 1) :=
sorry

end MathSequence

end find_a_n_find_T_n_find_lambda_q_and_c_n_l238_238687


namespace sum_of_c_n_l238_238370

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℕ := 2^n

def c_n (n : ℕ) : ℕ :=
  if odd n then a_n n else b_n n

def S_2n (n : ℕ) : ℕ :=
  ∑ i in range (2 * n), c_n (i + 1)

theorem sum_of_c_n (n : ℕ) :
  S_2n n = 2 * n^2 - n + (4^(n+1) - 4) / 3 :=
sorry

end sum_of_c_n_l238_238370


namespace marble_221_is_green_l238_238270

def marble_sequence_color (n : ℕ) : String :=
  let cycle_length := 15
  let red_count := 6
  let green_start := red_count + 1
  let green_end := red_count + 5
  let position := n % cycle_length
  if position ≠ 0 then
    let cycle_position := position
    if cycle_position <= red_count then "red"
    else if cycle_position <= green_end then "green"
    else "blue"
  else "blue"

theorem marble_221_is_green : marble_sequence_color 221 = "green" :=
by
  -- proof to be filled in
  sorry

end marble_221_is_green_l238_238270


namespace abs_diff_a1_b1_is_2_l238_238133

-- Given conditions and definitions
def a1 : ℕ := 61
def b1 : ℕ := 59
def a_vals : List ℕ := [61, 19, 11]
def b_vals : List ℕ := [59, 20, 10]

-- Given that the expression yields 2013 with the factorials
def exp := (List.product (a_vals.map Nat.factorial)) / (List.product (b_vals.map Nat.factorial))

-- Problem statement: Prove that |a1 - b1| = 2.
theorem abs_diff_a1_b1_is_2 : abs (a1 - b1) = 2 := by
  sorry

end abs_diff_a1_b1_is_2_l238_238133


namespace limit_cosine_expression_l238_238626

def small_angle_approx_sin5x (x : ℝ) : Prop := abs (sin (5 * x) - 5 * x) < ε
def small_angle_approx_sin2x (x : ℝ) : Prop := abs (sin (2 * x) - 2 * x) < ε
def small_angle_approx_cos2x (x : ℝ) : Prop := abs ((1 - cos (2 * x)) - (2 * x^2) / 2) < ε
def cos_diff_identity (A B : ℝ) : Prop := abs (cos A - cos B + 2 * sin ((A + B) / 2) * sin ((A - B) / 2)) < ε

theorem limit_cosine_expression :
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs x < δ →
    small_angle_approx_sin5x x ∧
    small_angle_approx_sin2x x ∧
    small_angle_approx_cos2x x ∧
    cos_diff_identity (7 * x) (3 * x) →
    abs ((1 - cos (2 * x)) / (cos (7 * x) - cos (3 * x)) + 1/10) < ε) := 
sorry

end limit_cosine_expression_l238_238626


namespace initial_amount_saved_l238_238312

noncomputable section

def cost_of_couch : ℝ := 750
def cost_of_table : ℝ := 100
def cost_of_lamp : ℝ := 50
def amount_still_owed : ℝ := 400

def total_cost : ℝ := cost_of_couch + cost_of_table + cost_of_lamp

theorem initial_amount_saved (initial_amount : ℝ) :
  initial_amount = total_cost - amount_still_owed ↔ initial_amount = 500 :=
by
  -- the proof is omitted
  sorry

end initial_amount_saved_l238_238312


namespace pyramid_volume_l238_238002

def is_rectangle (ABC : Type) (A B C D : ABC → ABC) : Prop := sorry

theorem pyramid_volume 
  (ABC : Type) [metric_space ABC] 
  (A B C D : ABC) (M : ABC)
  (is_rect : is_rectangle ABC A B C D)
  (DM_perpendicular_to_ABC : ∀ P ∈ {A, B, C, D}, ∀ Q ∈ {M}, dist DM_perpendicular_to_ABC P Q = dist DM_perpendicular_to_ABC B Q)
  (DM_int : ∃ (d : ℕ), d = dist M D)
  (MA_even : ∃ (a : ℕ), a * 2 = dist M A ∧ a * 2 + 2 = dist M B ∧ a * 2 + 4 = dist M C) :
  volume (pyramid M A B C D) = 8 * real.sqrt 3 := 
sorry

end pyramid_volume_l238_238002


namespace pq_intersects_ab_at_midpoint_l238_238013

theorem pq_intersects_ab_at_midpoint
  (A B C X Y P Q M : Type)
  (h1 : A ≠ B ∧ AC < BC)
  (h2 : ∀ {a}, a ∈ T},
  (h3 : is_perpendicular_bisector X AB)
  (h4 : is_projection_of_onto X AC P)
  (h5 : is_projection_of_onto Y BC Q)
  (h6 : is_perpendicular_bisector Y AC)
  (h7 : M = midpoint A B) :
  line PQ ∩ line AB = {M} :=
by
  sorry

end pq_intersects_ab_at_midpoint_l238_238013


namespace hockey_players_l238_238153

theorem hockey_players (n : ℕ) (h1 : n < 30) (h2 : n % 2 = 0) (h3 : n % 4 = 0) (h4 : n % 7 = 0) :
  (n / 4 = 7) :=
by
  sorry

end hockey_players_l238_238153


namespace distance_between_parallel_lines_l238_238516

theorem distance_between_parallel_lines :
  ∀ (k : ℝ), (∃ (d : ℝ), d = real.dist_lines (kx + 6y + 2 = 0) (4x - 2y + 2 = 0) ∧ 
    d = 4 * real.sqrt 5 / 15) :=
begin
  sorry
end

end distance_between_parallel_lines_l238_238516


namespace max_value_fraction_l238_238475

theorem max_value_fraction {a b : ℕ} (ha : a ∈ {2, 3, 4, 5, 6, 7, 8}) (hb : b ∈ {2, 3, 4, 5, 6, 7, 8}) :
  ( ∃ ab_val : ℚ, ab_val = a / (10 * b + a) + b / (10 * a + b) ∧ ab_val ≤ 1/4 ) :=
sorry

end max_value_fraction_l238_238475


namespace rowing_time_l238_238262

-- Defining the given conditions
def V_m : ℝ := 8     -- Speed of the man in still water (kmph)
def V_r : ℝ := 1.2   -- Speed of the river (kmph)
def D_total : ℝ := 7.82 -- Total distance traveled by the man (km)

-- The effective speeds upstream and downstream
def V_up := V_m - V_r
def V_down := V_m + V_r

-- The distance to the place he rows to
def D := D_total / 2

-- The time taken to row upstream
def T_up := D / V_up

-- The time taken to row downstream
def T_down := D / V_down

-- Total time taken to row to the place and back
def T_total := T_up + T_down

-- The main proof problem
theorem rowing_time : T_total = 1 := 
by
  sorry

end rowing_time_l238_238262


namespace probability_long_jump_probability_long_jump_and_100_meters_l238_238063

def events : List String := ["long jump", "100 meters", "200 meters", "400 meters"]

-- Question 1: Probability of choosing "long jump"
theorem probability_long_jump : 
  (1 : ℚ) / List.length events = 1 / 4 := 
by
  simp [events]
  norm_num

-- Question 2: Probability of choosing both "long jump" and "100 meters"
def combinations (lst : List String) : List (String × String) :=
  (lst.product lst).filter (λ p, p.fst ≠ p.snd)

def favorable_outcomes : List (String × String) :=
  [("long jump", "100 meters"), ("100 meters", "long jump")]

theorem probability_long_jump_and_100_meters :
  (favorable_outcomes.length : ℚ) / (combinations events).length = 1 / 6 :=
by
  simp [favorable_outcomes, combinations, events]
  norm_num

#check probability_long_jump
#check probability_long_jump_and_100_meters

end probability_long_jump_probability_long_jump_and_100_meters_l238_238063


namespace locus_circle_if_a_greater_s_squared_l238_238357

theorem locus_circle_if_a_greater_s_squared 
  (s a : ℝ) 
  (triangle : set (ℝ × ℝ)) 
  (is_equilateral_triangle : ∀ ⟨x, y⟩ ∈ triangle, ∃ (v1 v2 v3 : ℝ × ℝ), 
    v1 = (0, 0) ∧ v2 = (s, 0) ∧ v3 = (s / 2, (Real.sqrt 3) * s / 2))
  (P : ℝ × ℝ) 
  (distance_sum : (P.fst - 0)^2 + (P.snd - 0)^2 + (P.fst - s)^2 + (P.snd - 0)^2 + (P.fst - s / 2)^2 + (P.snd - (Real.sqrt 3) * s / 2)^2 = a)
  (h : a > s^2) 
  : ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ Q, Q = P ↔ (Q.fst - center.fst)^2 + (Q.snd - center.snd)^2 = radius^2 := 
sorry

end locus_circle_if_a_greater_s_squared_l238_238357


namespace lucas_1500th_day_is_sunday_l238_238486

def days_in_week : ℕ := 7

def start_day : ℕ := 5  -- 0: Monday, 1: Tuesday, ..., 5: Friday

def nth_day_of_life (n : ℕ) : ℕ :=
  (n - 1 + start_day) % days_in_week

theorem lucas_1500th_day_is_sunday : nth_day_of_life 1500 = 0 :=
by
  sorry

end lucas_1500th_day_is_sunday_l238_238486


namespace sin_alpha_eq_sqrt3_div_2_l238_238361

theorem sin_alpha_eq_sqrt3_div_2 {α : ℝ} 
  (h1 : sin α = (√3 / 2)) 
  (h2 : 0 < α) 
  (h3 : α < 2 * π) : 
  α = π / 3 ∨ α = 2 * π / 3 :=
by
  sorry

end sin_alpha_eq_sqrt3_div_2_l238_238361


namespace max_remained_numbers_l238_238850

theorem max_remained_numbers (S : Finset ℕ) (hSubset : S ⊆ Finset.range 236)
  (hCondition : ∀ a b c ∈ S, a ≠ b → a ≠ c → b ≠ c → ¬(b - a ∣ c)) : S.card ≤ 118 := 
sorry

end max_remained_numbers_l238_238850


namespace max_value_of_vector_expression_l238_238788

open_locale big_operators

variables {V : Type*} [inner_product_space ℝ V]

theorem max_value_of_vector_expression (a b c : V) 
  (h₁ : ∥a∥ = 3) (h₂ : ∥b∥ = 2) (h₃ : ∥c∥ = 4) :
  (∥a - 3 • b∥ ^ 2 + ∥b - 3 • c∥ ^ 2 + ∥c - 3 • a∥ ^ 2) ≤ 428 :=
sorry

end max_value_of_vector_expression_l238_238788


namespace magnitude_of_conjugate_l238_238374

-- Define the imaginary unit as a complex number
def i : ℂ := complex.I

-- The given condition: 1 - i = (2 + 4 * i) / z
axiom condition : ∃ z : ℂ, 1 - i = (2 + 4 * i) / z

-- The statement to prove
theorem magnitude_of_conjugate (z : ℂ) (h : 1 - i = (2 + 4 * i) / z) : |conj z| = real.sqrt 10 :=
sorry

end magnitude_of_conjugate_l238_238374


namespace box_surface_area_l238_238145

theorem box_surface_area (x y z : ℝ) 
  (h1 : 4 * (x + y + z) = 140) 
  (h2 : sqrt (x^2 + y^2 + z^2) = 21) : 
  2 * (x * y + x * z + y * z) = 784 := 
by
  -- Placeholder to ensure the code is buildable
  sorry

end box_surface_area_l238_238145


namespace percent_freshmen_psychology_majors_l238_238285

-- Define the total number of students in our context
def total_students : ℕ := 100

-- Define what 80% of total students being freshmen means
def freshmen (total : ℕ) : ℕ := 8 * total / 10

-- Define what 60% of freshmen being in the school of liberal arts means
def freshmen_in_liberal_arts (total : ℕ) : ℕ := 6 * freshmen total / 10

-- Define what 50% of freshmen in the school of liberal arts being psychology majors means
def freshmen_psychology_majors (total : ℕ) : ℕ := 5 * freshmen_in_liberal_arts total / 10

theorem percent_freshmen_psychology_majors :
  (freshmen_psychology_majors total_students : ℝ) / total_students * 100 = 24 :=
by
  sorry

end percent_freshmen_psychology_majors_l238_238285


namespace last_two_digits_of_1976_pow_100_l238_238521

theorem last_two_digits_of_1976_pow_100 :
  (1976 ^ 100) % 100 = 76 :=
by
  sorry

end last_two_digits_of_1976_pow_100_l238_238521


namespace angle_equality_l238_238440

-- Given: an acute triangle ABC with points D, E, P, R, Q, S as described in the problem
theorem angle_equality
  (A B C D E P R Q S : Type)
  [euclidean_geometry A B C]
  (h1 : is_acute_triangle A B C)
  (h2 : tangent_to_circumscribed_circle A B C D E)
  (h3 : intersection_of_lines AE BC P)
  (h4 : intersection_of_lines BD AC R)
  (h5 : midpoint_of_segment A P Q)
  (h6 : midpoint_of_segment B R S) :
  ∠ABQ = ∠BAS :=
sorry  -- Proof omitted

end angle_equality_l238_238440


namespace haohau_age_l238_238423

-- Definitions based on conditions
def current_year : ℕ := 2015
def experienced_leap_years : List ℕ := [2012, 2008]
def birth_year_multiple_of_9 (year : ℕ) : Prop := year % 9 = 0

-- Mathematically equivalent Lean statement
theorem haohau_age (birth_year : ℕ) 
    (h1 : current_year = 2015)
    (h2 : experienced_leap_years = [2012, 2008])
    (h3 : birth_year_multiple_of_9 birth_year)
    (h4 : births_year ∈ filter (λ y, y % 9 = 0) (range 2004 (current_year + 1))) :

    birth_year = 2007 → (2016 - birth_year) = 9 := 
by 
  intros h_birth_year 
  simp [h_birth_year] 
  norm_num  -- simplifying the arithmetic
  sorry

end haohau_age_l238_238423


namespace sum_arithmetic_series_l238_238667

theorem sum_arithmetic_series (n : ℕ) (h1 : 1 ≤ n) : 
  ∑ k in finset.range (2 * n - 1), (n + k) = (2 * n - 1) * 2 / 2 :=
by
  sorry

end sum_arithmetic_series_l238_238667


namespace base_12_decimal_count_l238_238307

theorem base_12_decimal_count :
  let numbers := {n : ℕ | n < 1200 ∧ (n.toDigits 12).all (λ d, d ≤ 9)} in
  numbers.card = 90 :=
by
  sorry

end base_12_decimal_count_l238_238307


namespace tangent_line_at_origin_l238_238007

-- Define the function f(x) = x^3 + ax with an extremum at x = 1
def f (x a : ℝ) : ℝ := x^3 + a * x

-- Define the condition for a local extremum at x = 1: f'(1) = 0
def extremum_condition (a : ℝ) : Prop := (3 * 1^2 + a = 0)

-- Define the derivative of f at x = 0
def derivative_at_origin (a : ℝ) : ℝ := 3 * 0^2 + a

-- Define the value of function at x = 0
def value_at_origin (a : ℝ) : ℝ := f 0 a

-- The main theorem to prove
theorem tangent_line_at_origin (a : ℝ) (ha : extremum_condition a) :
    (value_at_origin a = 0) ∧ (derivative_at_origin a = -3) → ∀ x, (3 * x + (f x a - f 0 a) / (x - 0) = 0) := by
  sorry

end tangent_line_at_origin_l238_238007


namespace Vasya_finish_book_in_three_days_l238_238162

variable (P : ℕ)

def pages_read_day1 : ℕ := P / 2
def pages_remaining_day1 : ℕ := P - pages_read_day1
def pages_read_day2 : ℕ := pages_remaining_day1 / 3
def pages_remaining_day2 : ℕ := pages_remaining_day1 - pages_read_day2
def pages_read_day3 : ℕ := (pages_read_day1 + pages_read_day2) / 2

theorem Vasya_finish_book_in_three_days (P : ℕ) :
  pages_read_day1 P + pages_read_day2 P + pages_read_day3 P = P :=
by
  sorry

end Vasya_finish_book_in_three_days_l238_238162


namespace max_remaining_numbers_l238_238816

theorem max_remaining_numbers : 
  ∃ (S ⊆ {n | 1 ≤ n ∧ n ≤ 235}), (∀ x ∈ S, ∀ y ∈ S, x ≠ y → ¬(x ∣ (y - x))) ∧ card S = 118 :=
by
  sorry

end max_remaining_numbers_l238_238816


namespace yogurt_cost_l238_238861

-- Define the conditions given in the problem
def total_cost_ice_cream : ℕ := 20 * 6
def spent_difference : ℕ := 118

theorem yogurt_cost (y : ℕ) 
  (h1 : total_cost_ice_cream = 2 * y + spent_difference) : 
  y = 1 :=
  sorry

end yogurt_cost_l238_238861


namespace method_1_max_sections_method_2_max_sections_method_3_max_sections_num_pipes_method_2_method_3_alternative_method_1_and_3_l238_238938

-- Define the conditions of the problem
def pipe_length : ℝ := 6
def section_0_8m : ℝ := 0.8
def section_2_5m : ℝ := 2.5
def num_0_8m_needed : ℕ := 100
def num_2_5m_needed : ℕ := 32

-- Method definitions
def method_1_num_sections : ℕ := (pipe_length / section_0_8m).floor
def method_2_num_sections (remainder: ℝ) : ℕ := (remainder / section_0_8m).floor
def method_3_num_sections : ℕ := method_2_num_sections (pipe_length - 2 * section_2_5m)

-- Prove Method 1 calculation
theorem method_1_max_sections : method_1_num_sections = 7 := 
by sorry

-- Prove Method 2 calculation
theorem method_2_max_sections : method_2_num_sections (pipe_length - section_2_5m) = 4 := 
by sorry

-- Prove Method 3 calculation
theorem method_3_max_sections : method_3_num_sections = 1 := 
by sorry

-- Define the system of equations
def system_equations (x y : ℕ) :=
  (x + 2*y = num_2_5m_needed) ∧ (4*x + y = num_0_8m_needed)

-- Prove the number of pipes needed for Methods 2 and 3
theorem num_pipes_method_2_method_3 : ∃ x y : ℕ, system_equations x y ∧ x = 24 ∧ y = 4 :=
by sorry

-- Alternative method combination
def alternative_system_equations (m n : ℕ) :=
  (7*m + n = num_0_8m_needed) ∧ (2*n = num_2_5m_needed)

-- Prove alternative methods 1 and 3 combination
theorem alternative_method_1_and_3 : ∃ m n : ℕ, alternative_system_equations m n ∧ m = 12 ∧ n = 16 :=
by sorry

end method_1_max_sections_method_2_max_sections_method_3_max_sections_num_pipes_method_2_method_3_alternative_method_1_and_3_l238_238938


namespace possible_integer_roots_l238_238068

theorem possible_integer_roots :
  ∀ (a b c d e : ℤ), ∃ m : ℕ, m ∈ {0, 1, 2, 3, 5} ∧ ∀ x : ℤ, x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 → multiplicity x = m := 
sorry

end possible_integer_roots_l238_238068


namespace distinguishable_arrangements_l238_238401

-- Define number of each type of tiles
def brown_tiles := 2
def purple_tiles := 1
def green_tiles := 3
def yellow_tiles := 2
def total_tiles := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem distinguishable_arrangements :
  (Nat.factorial total_tiles) / 
  ((Nat.factorial green_tiles) * 
   (Nat.factorial brown_tiles) * 
   (Nat.factorial yellow_tiles) * 
   (Nat.factorial purple_tiles)) = 1680 := by
  sorry

end distinguishable_arrangements_l238_238401


namespace green_peaches_in_each_basket_l238_238648

theorem green_peaches_in_each_basket (G : ℕ) 
  (h1 : ∀ B : ℕ, B = 15) 
  (h2 : ∀ R : ℕ, R = 19) 
  (h3 : ∀ P : ℕ, P = 345) 
  (h_eq : 345 = 15 * (19 + G)) : 
  G = 4 := by
  sorry

end green_peaches_in_each_basket_l238_238648


namespace parabola_focus_directrix_distance_l238_238386

theorem parabola_focus_directrix_distance (p : ℝ) (hp : 0 < p) (focus : ℝ × ℝ) :
  (focus.1^2 + focus.2^2 = 4) ∧ (focus = (p/2, 0)) → p = 4 :=
by
  intro h
  cases h with hc hf
  have hp2 : p/2 = 2 :=
    by
      -- Detail omitted for brevity
      sorry
  have hpeq : p = 4 := by
    -- Detail omitted for brevity
    sorry
  exact hpeq

end parabola_focus_directrix_distance_l238_238386


namespace limit_expression_l238_238327

theorem limit_expression : 
  (Real.seqLimit (fun n => (12 * n + 5) / (Real.cbrt (27 * n^3 + 6 * n^2 + 8))) (4 : ℝ)) :=
by sorry

end limit_expression_l238_238327


namespace solve_for_y_l238_238017

theorem solve_for_y (y : ℝ)
  (h1 : 9 * y^2 + 8 * y - 1 = 0)
  (h2 : 27 * y^2 + 44 * y - 7 = 0) : 
  y = 1 / 9 :=
sorry

end solve_for_y_l238_238017


namespace not_net_of_cuboid_l238_238781

noncomputable def cuboid_closed_path (c : Type) (f : c → c) :=
∀ (x1 x2 : c), ∃ (y : c), f x1 = y ∧ f x2 = y

theorem not_net_of_cuboid (c : Type) [Nonempty c] [DecidableEq c] (net : c → Set c) (f : c → c) :
  cuboid_closed_path c f → ¬ (∀ x, net x = {x}) :=
by
  sorry

end not_net_of_cuboid_l238_238781


namespace tiffany_total_bags_l238_238917

-- Define the initial and additional bags correctly
def bags_on_monday : ℕ := 10
def bags_next_day : ℕ := 3
def bags_day_after : ℕ := 7

-- Define the total bags calculation
def total_bags (initial : ℕ) (next : ℕ) (after : ℕ) : ℕ :=
  initial + next + after

-- Prove that the total bags collected is 20
theorem tiffany_total_bags : total_bags bags_on_monday bags_next_day bags_day_after = 20 :=
by
  sorry

end tiffany_total_bags_l238_238917


namespace monotone_decreasing_interval_l238_238377

noncomputable def f (x : ℝ) : ℝ := -2 * real.sin (2 * x + π / 4)

theorem monotone_decreasing_interval :
  f(π / 8) = -2 →
  ∃ a b : ℝ, [a, b] = [π / 8, 5 * π / 8] ∧
    ∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ b → f(y) ≤ f(x) :=
begin
  intros h,
  use [π / 8, 5 * π / 8],
  split,
  refl,
  intros x y hx hy hxy,
  sorry  -- Proof to be completed
end

end monotone_decreasing_interval_l238_238377


namespace add_to_fraction_l238_238932

theorem add_to_fraction (x : ℕ) :
  (3 + x) / (11 + x) = 5 / 9 ↔ x = 7 :=
by
  sorry

end add_to_fraction_l238_238932


namespace carpet_area_l238_238855

/-- A rectangular floor with a length of 15 feet and a width of 12 feet needs 20 square yards of carpet to cover it. -/
theorem carpet_area (length_feet : ℕ) (width_feet : ℕ) (feet_per_yard : ℕ) (length_yards : ℕ) (width_yards : ℕ) (area_sq_yards : ℕ) :
  length_feet = 15 ∧
  width_feet = 12 ∧
  feet_per_yard = 3 ∧
  length_yards = length_feet / feet_per_yard ∧
  width_yards = width_feet / feet_per_yard ∧
  area_sq_yards = length_yards * width_yards → 
  area_sq_yards = 20 :=
by
  sorry

end carpet_area_l238_238855


namespace functions_even_odd_properties_l238_238721

noncomputable def f1 (x : ℝ) : ℝ := log (1 - x^2) / (abs (x^2 - 2) - 2)

noncomputable def f2 (x : ℝ) : ℝ := (x - 1) * sqrt ((x + 1) / (x - 1))

noncomputable def f3 (a : ℝ) (h : a > 0 ∧ a ≠ 1) (x : ℝ) : ℝ := log a (x + sqrt (x^2 + 1))

noncomputable def f4 (x : ℝ) : ℝ := x * (1 / (2^x - 1) + 1 / 2)

theorem functions_even_odd_properties
  (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  (∀ x, f1 x = f1 (-x)) ∧
  (¬ (∀ x, f2 x = f2 (-x)) ∧ ¬ (∀ x, f2 x = -f2 (-x))) ∧
  (∀ x, f3 a h x = -f3 a h (-x)) ∧
  (∀ x, f4 x = f4 (-x)) :=
sorry

end functions_even_odd_properties_l238_238721


namespace rectangle_other_side_length_l238_238898

variable (a : ℝ)

theorem rectangle_other_side_length
  (P : ℝ) (hP : P = 8)
  (l : ℝ) (hl : l = -a - 2) :
  ∃ w, w = 6 + a :=
by
  have eq : P = 2 * (l + w) := sorry  -- From perimeter formula
  use 6 + a
  sorry  -- Solve the equation and prove it

end rectangle_other_side_length_l238_238898


namespace alberto_spent_2457_l238_238278

-- Define the expenses by Samara on each item
def oil_expense : ℕ := 25
def tires_expense : ℕ := 467
def detailing_expense : ℕ := 79

-- Define the additional amount Alberto spent more than Samara
def additional_amount : ℕ := 1886

-- Total amount spent by Samara
def samara_total_expense : ℕ := oil_expense + tires_expense + detailing_expense

-- The amount spent by Alberto
def alberto_expense := samara_total_expense + additional_amount

-- Theorem stating the amount spent by Alberto
theorem alberto_spent_2457 :
  alberto_expense = 2457 :=
by {
  -- Include the actual proof here if necessary
  sorry
}

end alberto_spent_2457_l238_238278


namespace max_rabbits_with_long_ears_and_jumping_far_l238_238096

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l238_238096


namespace areas_ratio_9_to_1_l238_238772

-- Defining the entities and conditions involved
variables (A B C D S E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace S] [MetricSpace E]

-- AB and CD are parallel sides
variable (AB_parallel_CD : ∃ l1 l2 : Line, l1.parallel l2 ∧ l1.contains A ∧ l1.contains B ∧ l2.contains C ∧ l2.contains D)

-- AB is twice the length of CD
variable (AB_twice_CD : dist A B = 2 * dist C D)

-- S is the intersection of diagonals AC and BD
variable (S_intersection_diagonals : ∃ AC BD : Line, AC.contains A ∧ AC.contains C ∧ BD.contains B ∧ BD.contains D ∧ AC.intersect BD = S)

-- Triangles ABS and CDS are equilateral
variable (triangle_ABS_equilateral : equilateral_triangle A B S)
variable (triangle_CDS_equilateral : equilateral_triangle C D S)

-- Point E lies on segment BS such that angle ACE is 30 degrees
variable (E_on_BS_angle_ACE : ∃ ace : Angle, ace.contains A ∧ ace.contains C ∧ ace.contains E ∧ ace.measure = 30)

-- The main theorem to prove the ratio of areas
noncomputable def ratio_areas_ABCD_EBC : Prop :=
  let area_ABCD := area_of_quadrilateral A B C D in
  let area_EBC := area_of_triangle E B C in
  area_ABCD / area_EBC = 9

-- Statement for Lean
theorem areas_ratio_9_to_1
  (hAB_parallel_CD : AB_parallel_CD)
  (hAB_twice_CD : AB_twice_CD)
  (hS_intersection_diagonals : S_intersection_diagonals)
  (htriangle_ABS_eq : triangle_ABS_equilateral)
  (htriangle_CDS_eq : triangle_CDS_equilateral)
  (hE_on_BS_angle_ACE : E_on_BS_angle_ACE) :
  ratio_areas_ABCD_EBC A B C D S E := sorry

end areas_ratio_9_to_1_l238_238772


namespace quadratic_condition_l238_238879

theorem quadratic_condition (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * x + 3 = 0) → a ≠ 0 :=
by 
  intro h
  -- Proof will be here
  sorry

end quadratic_condition_l238_238879


namespace mother_age_five_times_xiaohua_six_years_ago_l238_238223

theorem mother_age_five_times_xiaohua_six_years_ago :
  ∀ (Xiaohua_age mother_age : ℕ), Xiaohua_age = 12 → mother_age = 36 →
  ∃ (years_ago : ℕ), years_ago = 6 ∧ (mother_age - years_ago) = 5 * (Xiaohua_age - years_ago) :=
by
  intros Xiaohua_age mother_age h_xiaohua_age h_mother_age
  use 6
  split
  · refl
  · rw [h_xiaohua_age, h_mother_age]
    sorry

end mother_age_five_times_xiaohua_six_years_ago_l238_238223


namespace max_rabbits_with_traits_l238_238120

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l238_238120


namespace part1_part2_l238_238713

noncomputable def f (x a : ℝ) : ℝ := x^2 - a * x - a * Real.log x

theorem part1 (a : ℝ) :
  (∃ x : ℝ, f x a = x^2 - a * x - a * Real.log x ∧ (∀ y : ℝ, (2 * x - a - a / x = 0) → y = x)) →
  a = 1 :=
sorry

theorem part2 (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ici Real.exp(1) → f x a ≥ 0) →
  a ≤ Real.exp(2) / (Real.exp(1) + 1) :=
sorry

end part1_part2_l238_238713


namespace simplify_fraction_l238_238053

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l238_238053


namespace solve_for_x_l238_238336

theorem solve_for_x (x : ℚ) :
  (x^2 - 4*x + 3) / (x^2 - 7*x + 6) = (x^2 - 3*x - 10) / (x^2 - 2*x - 15) →
  x = -3 / 4 :=
by
  intro h
  sorry

end solve_for_x_l238_238336


namespace maximum_rabbits_l238_238083

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l238_238083


namespace sara_likes_numbers_divisible_by_8_l238_238027

theorem sara_likes_numbers_divisible_by_8 :
  (∃ n : ℕ, (n % 10) ∈ {0, 2, 4, 6, 8} ∧ n % 8 = 0) → 
  (∀ n : ℕ, n % 8 = 0 → (n % 10) ∈ {0, 2, 4, 6, 8} ∧ {0, 2, 4, 6, 8}.card = 5) :=
sorry

end sara_likes_numbers_divisible_by_8_l238_238027


namespace parallel_line_plane_l238_238414

-- Define vectors
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Dot product definition
def dotProduct (u v : Vector3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

-- Options given
def optionA : Vector3D × Vector3D := (⟨1, 0, 0⟩, ⟨-2, 0, 0⟩)
def optionB : Vector3D × Vector3D := (⟨1, 3, 5⟩, ⟨1, 0, 1⟩)
def optionC : Vector3D × Vector3D := (⟨0, 2, 1⟩, ⟨-1, 0, -1⟩)
def optionD : Vector3D × Vector3D := (⟨1, -1, 3⟩, ⟨0, 3, 1⟩)

-- Main theorem
theorem parallel_line_plane :
  (dotProduct (optionA.fst) (optionA.snd) ≠ 0) ∧
  (dotProduct (optionB.fst) (optionB.snd) ≠ 0) ∧
  (dotProduct (optionC.fst) (optionC.snd) ≠ 0) ∧
  (dotProduct (optionD.fst) (optionD.snd) = 0) :=
by
  -- Using sorry to skip the proof
  sorry

end parallel_line_plane_l238_238414


namespace largest_interesting_number_l238_238167

def is_interesting (n : ℕ) : Prop :=
  ∀ (digits : List ℕ) (h : digits.reverse.mk_nat = n),
    ∀ (i : ℕ) (h1 : i > 0) (h2 : i < digits.length - 1),
      digits.nth_le i h2 < (digits.nth_le (i - 1) sorry + digits.nth_le (i + 1) sorry) / 2

theorem largest_interesting_number :
  ∃ n : ℕ, is_interesting n ∧ n = 96433469 :=
by
  existsi 96433469
  split
  { sorry } -- Proof that 96433469 is interesting goes here
  { refl } -- Trivially true since we are claiming the number itself

end largest_interesting_number_l238_238167


namespace find_value_of_A_l238_238395

theorem find_value_of_A (x : ℝ) (h₁ : x - 3 * (x - 2) ≥ 2) (h₂ : 4 * x - 2 < 5 * x - 1) (h₃ : x ≠ 1) (h₄ : x ≠ -1) (h₅ : x ≠ 0) (hx : x = 2) :
  let A := (3 * x / (x - 1) - x / (x + 1)) / (x / (x^2 - 1))
  A = 8 :=
by
  -- Proof will be filled in
  sorry

end find_value_of_A_l238_238395


namespace maximum_numbers_no_divisible_difference_l238_238819

theorem maximum_numbers_no_divisible_difference :
  ∃ (S : Finset ℕ), S ⊆ Finset.range 236 ∧ 
  (∀ a b c ∈ S, a ≠ b → b ≠ c → c ≠ a → ¬ (a - b = 0) ∨ ¬ (c ∣ (a - b))) ∧ S.card ≤ 118 :=
by
  sorry

end maximum_numbers_no_divisible_difference_l238_238819


namespace asymptotes_of_hyperbola_l238_238518

-- Define the standard form of the hyperbola equation given in the problem.
def hyperbola_eq (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 3) = 1

-- Define the asymptote equations for the given hyperbola.
def asymptote_eq (x y : ℝ) : Prop := (√3 * x + 2 * y = 0) ∨ (√3 * x - 2 * y = 0)

-- State the theorem that the asymptotes of the given hyperbola are as described.
theorem asymptotes_of_hyperbola (x y : ℝ) : hyperbola_eq x y → asymptote_eq x y :=
by
  sorry

end asymptotes_of_hyperbola_l238_238518


namespace solve_differential_eq_l238_238506

noncomputable theory

-- Define the given differential equation as a condition
def differential_eq (y : ℝ → ℝ) (y' : ℝ → ℝ) (x : ℝ) : Prop :=
  y' x * real.cos x + y x * real.sin x = 1

-- State the theorem which needs to be proved
theorem solve_differential_eq (C : ℝ) (y : ℝ → ℝ) (y' : ℝ → ℝ) :
  (∀ x, differential_eq y y' x) → (∀ x, y x = real.sin x + C * real.cos x) :=
sorry

end solve_differential_eq_l238_238506


namespace max_visible_cubes_from_point_l238_238305

theorem max_visible_cubes_from_point (n : ℕ) (h : n = 12) :
  let total_cubes := n^3
  let face_cube_count := n * n
  let edge_count := n
  let visible_face_count := 3 * face_cube_count
  let double_counted_edges := 3 * (edge_count - 1)
  let corner_cube_count := 1
  visible_face_count - double_counted_edges + corner_cube_count = 400 := by
  sorry

end max_visible_cubes_from_point_l238_238305


namespace shadow_boundary_eqn_l238_238596

noncomputable def boundary_of_shadow (x : ℝ) : ℝ := x^2 / 10 - 1

theorem shadow_boundary_eqn (radius : ℝ) (center : ℝ × ℝ × ℝ) (light_source : ℝ × ℝ × ℝ) (x y: ℝ) :
  radius = 2 →
  center = (0, 0, 2) →
  light_source = (0, -2, 3) →
  y = boundary_of_shadow x :=
by
  intros hradius hcenter hlight
  sorry

end shadow_boundary_eqn_l238_238596


namespace perfect_squares_with_property_l238_238653

open Nat

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k : ℕ, p.Prime ∧ k > 0 ∧ n = p^k

def satisfies_property (n : ℕ) : Prop :=
  ∀ a : ℕ, a ∣ n → a ≥ 15 → is_prime_power (a + 15)

theorem perfect_squares_with_property :
  {n | satisfies_property n ∧ ∃ k : ℕ, n = k^2} = {1, 4, 9, 16, 49, 64, 196} :=
by
  sorry

end perfect_squares_with_property_l238_238653


namespace limit_of_expression_l238_238234

noncomputable def limit_expression (a x : ℝ) := (2 - x / a) ^ tan (π * x / (2 * a))

theorem limit_of_expression (a : ℝ) (h : 0 < a) :
  Filter.Tendsto (λ x, limit_expression a x) (nhds a) (nhds (Real.exp (2 / π))) :=
sorry

end limit_of_expression_l238_238234


namespace polynomial_solutions_l238_238661

-- Define the type of the polynomials and statement of the problem
def P1 (x : ℝ) : ℝ := x
def P2 (x : ℝ) : ℝ := x^2 + 1
def P3 (x : ℝ) : ℝ := x^4 + 2*x^2 + 2

theorem polynomial_solutions :
  (∀ x : ℝ, P1 (x^2 + 1) = P1 x^2 + 1) ∧
  (∀ x : ℝ, P2 (x^2 + 1) = P2 x^2 + 1) ∧
  (∀ x : ℝ, P3 (x^2 + 1) = P3 x^2 + 1) :=
by
  -- Proof will go here
  sorry

end polynomial_solutions_l238_238661


namespace total_cost_is_correct_l238_238541

def area1 : ℝ := 17.56 * 10000 -- in square meters
def area2 : ℝ := 25.92 * 10000 -- in square meters
def area3 : ℝ := 11.76 * 10000 -- in square meters
def cost_per_meter1 : ℝ := 3   -- in Rs.
def cost_per_meter2 : ℝ := 4   -- in Rs.
def cost_per_meter3 : ℝ := 5   -- in Rs.

noncomputable def total_cost_of_fencing (a1 a2 a3 cpm1 cpm2 cpm3 : ℝ) : ℝ :=
  let r1 := Real.sqrt (a1 / Real.pi)
  let r2 := Real.sqrt (a2 / Real.pi)
  let r3 := Real.sqrt (a3 / Real.pi)
  let c1 := 2 * Real.pi * r1
  let c2 := 2 * Real.pi * r2
  let c3 := 2 * Real.pi * r3
  (c1 * cpm1) + (c2 * cpm2) + (c3 * cpm3)

theorem total_cost_is_correct :
  total_cost_of_fencing area1 area2 area3 cost_per_meter1 cost_per_meter2 cost_per_meter3 ≈ 17752.8 :=
sorry

end total_cost_is_correct_l238_238541


namespace farm_owns_60_more_horses_than_cows_l238_238499

-- Let x be the number of cows initially
-- The number of horses initially is 4x
-- After selling 15 horses and buying 15 cows, the ratio of horses to cows becomes 7:3

theorem farm_owns_60_more_horses_than_cows (x : ℕ) (h_pos : 0 < x)
  (h_ratio : (4 * x - 15) / (x + 15) = 7 / 3) :
  (4 * x - 15) - (x + 15) = 60 :=
by
  sorry

end farm_owns_60_more_horses_than_cows_l238_238499


namespace number_of_distinct_real_numbers_l238_238009

-- Define the function g.
def g (x : ℝ) : ℝ := x^2 - 3 * x

-- The main statement which asks to prove that there are 8 distinct real numbers d such that g(g(g(g(d)))) = 2.
theorem number_of_distinct_real_numbers (d : ℝ) :
  {d | g (g (g (g d))) = 2}.to_finset.card = 8 :=
sorry

end number_of_distinct_real_numbers_l238_238009


namespace fourth_root_is_four_l238_238916

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^4 - 8 * x^3 - 7 * x^2 + 9 * x + 11

-- Conditions that must be true for the given problem
@[simp] def f_neg1_zero : f (-1) = 0 := by sorry
@[simp] def f_2_zero : f (2) = 0 := by sorry
@[simp] def f_neg3_zero : f (-3) = 0 := by sorry

-- The theorem stating the fourth root
theorem fourth_root_is_four (root4 : ℝ) (H : f root4 = 0) : root4 = 4 := by sorry

end fourth_root_is_four_l238_238916


namespace legally_drive_after_hours_l238_238874

theorem legally_drive_after_hours (n : ℕ) :
  (∀ t ≥ n, 0.8 * (0.5 : ℝ) ^ t ≤ 0.2) ↔ n = 2 :=
by
  sorry

end legally_drive_after_hours_l238_238874


namespace accuracy_of_rounded_number_l238_238610

-- Define the given conditions
def given_number : ℝ := 150.38 * 10^6

-- The place value we are checking
def place_value := 100

-- The accuracy check function
def is_accurate_to_place (num : ℝ) (place : ℕ) : Prop :=
  let shifted_num := num / place
  let rounded_num := (shifted_num + 0.5).floor
  rounded_num * place = num

-- The proof problem
theorem accuracy_of_rounded_number : is_accurate_to_place given_number place_value := 
sorry

end accuracy_of_rounded_number_l238_238610


namespace has_no_extreme_values_l238_238718

noncomputable def y (x : ℝ) := x - Real.log(1 + x^2)

lemma no_extreme_values : ∀ x : ℝ, (deriv y x) ≥ 0 := 
by 
  intros x 
  calc 
    deriv y x = 1 - (2*x)/(1 + x^2) : sorry -- this corresponds to the given derivative
    deriv y x = (x-1)^2 / (1 + x^2) : sorry -- this simplifies the derivative

theorem has_no_extreme_values : ∀ x : ℝ, ¬(∃ x0 : ℝ, deriv y x0 = 0) := 
by 
  intros x 
  simp [no_extreme_values x] -- since the derivative is always non-negative, there are no extreme values
  sorry 

end has_no_extreme_values_l238_238718


namespace proof1_proof2_proof3_l238_238259

-- Define the function and conditions
axiom f : ℝ → ℝ
axiom h1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y) - 2
axiom h2 : ∀ x : ℝ, x > 0 → f(x) > 2

-- Proof 1: f(0) = 2
theorem proof1 : f (0) = 2 := 
sorry

-- Proof 2: f(x) is strictly increasing
theorem proof2 : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2) := 
sorry

-- Proof 3: Solve the inequality
theorem proof3 : ∀ t : ℝ, f(2 * t^2 - t - 3) - 2 < 0 → -1 < t ∧ t < 3 / 2 := 
sorry

end proof1_proof2_proof3_l238_238259


namespace running_percent_correct_l238_238257

def cricketer_runs_total : ℕ := 183
def boundaries_count : ℕ := 14
def boundary_runs : ℕ := boundaries_count * 4
def sixes_count : ℕ := 3
def sixes_runs : ℕ := sixes_count * 6
def no_balls_count : ℕ := 2
def no_balls_runs : ℕ := no_balls_count * 1
def wides_count : ℕ := 5
def wides_runs : ℕ := wides_count * 1
def byes_count : ℕ := 3
def byes_runs : ℕ := byes_count * 1

def total_boundary_six_runs : ℕ := boundary_runs + sixes_runs
def total_extras_runs : ℕ := no_balls_runs + wides_runs + byes_runs
def total_boundary_sixes_extras_runs : ℕ := total_boundary_six_runs + total_extras_runs
def runs_by_running : ℕ := cricketer_runs_total - total_boundary_sixes_extras_runs

def percent_runs_by_running : Float := (runs_by_running.toFloat / cricketer_runs_total.toFloat) * 100

theorem running_percent_correct :
  abs (percent_runs_by_running - 54.10) < 0.01 :=
sorry

end running_percent_correct_l238_238257


namespace billy_sleep_total_l238_238747

theorem billy_sleep_total :
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  day1 + day2 + day3 + day4 = 30 :=
by
  -- Definitions
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  -- Assertion
  have h : day1 + day2 + day3 + day4 = 30 := sorry
  exact h

end billy_sleep_total_l238_238747


namespace walnut_trees_initially_in_park_l238_238539

def initial_trees_in_park (final_trees planted_trees : ℕ) : ℕ :=
  final_trees - planted_trees

theorem walnut_trees_initially_in_park (final_trees planted_trees initial_trees : ℕ) 
  (h1 : final_trees = 55) 
  (h2 : planted_trees = 33)
  (h3 : initial_trees = initial_trees_in_park final_trees planted_trees) :
  initial_trees = 22 :=
by
  rw [initial_trees_in_park, h1, h2]
  simp
  exact h3
  sorry

end walnut_trees_initially_in_park_l238_238539


namespace esports_gender_relationship_probability_at_least_one_male_selected_expected_value_liking_esports_l238_238062

theorem esports_gender_relationship :
  ∀ (a b c d n : ℕ), 
    a = 120 → b = 80 → c = 100 → d = 100 → n = 400 → 
    ((n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))) > 3.841 :=
by sorry

theorem probability_at_least_one_male_selected :
  ∀ (total_dislike male_dislike female_dislike selected total_selected : ℕ),
    total_dislike = 9 → male_dislike = 4 → female_dislike = 5 → selected = 3 → total_selected = 9 →
    let p_all_females := (Nat.choose female_dislike selected) / (Nat.choose total_dislike selected) in
    1 - p_all_females = 37 / 42 :=
by sorry

theorem expected_value_liking_esports :
  ∀ (total_students total_like_esports sample_size : ℕ),
    total_students = 400 → total_like_esports = 220 → sample_size = 10 →
    let p := total_like_esports / total_students in
    (sample_size : ℝ) * p = 11 / 2 :=
by sorry

end esports_gender_relationship_probability_at_least_one_male_selected_expected_value_liking_esports_l238_238062


namespace triangle_CEF_equilateral_l238_238875

theorem triangle_CEF_equilateral
  (A B C D B1 C1 D1 F E : Point)
  (h1 : arc_division_eq_3 A B C D)
  (h2 : rotated_by_pi_over_3 A B C D B1 C1 D1)
  (h3 : F = intersection (line_through A B1) (line_through D C1))
  (h4 : on_angle_bisector E (angle_bisector B1 B A) ∧ (dist B D = dist D E)) :
  is_equilateral_triangle (Triangle.mk C E F) :=
by sorry

-- Definitions of Point, arc_division_eq_3, rotated_by_pi_over_3, intersection, line_through,
-- angle_bisector, points_on bisector, Triangle.mk, is_equilateral_triangle may be abstractions
-- or sufficiently simple definitions in the Mathlib library or the user can define them as needed.

end triangle_CEF_equilateral_l238_238875


namespace height_comparison_l238_238976

variable {a b c h_a h_b h_c : ℝ}
variable (triangle_ABC : {a : ℝ} × {b : ℝ} × {c : ℝ})
variable (suspension_A : triangle_ABC.1 < triangle_ABC.2 ∧ triangle_ABC.2 < triangle_ABC.3)
variable (height_A : ℝ)
variable (height_B : ℝ)
variable (height_C : ℝ)
variable (median_A : ℝ)
variable (h_A : height_A ≥ median_A)
variable (median_BC : median_A = (height_B + height_C) / 2)
variable (hypothesis : height_A^2 + (height_B + height_C)^2 = triangle_ABC.3^2 )

theorem height_comparison : height_A > height_B > height_C := 
  sorry

end height_comparison_l238_238976


namespace area_between_concentric_circles_l238_238160

theorem area_between_concentric_circles
  (C : Point)
  (A D B : Point)
  (AC BD : ℝ)
  (chord_length : ℝ)
  (outer_radius : ℝ)
  (tangent_inner_circle : ∃ B C D, ∠ ACD = 90 ∧ BD = chord_length)
  (center : A = D)
  (conditions : AC = outer_radius ∧ BD = chord_length / 2 ∧ 2 * BD = chord_length)
  : ∃ annular_area : ℝ, annular_area = 100 * real.pi := 
begin
  sorry
end

end area_between_concentric_circles_l238_238160


namespace eggs_in_each_basket_l238_238158

theorem eggs_in_each_basket (n : ℕ) (h₁ : 5 ≤ n) (h₂ : n ∣ 30) (h₃ : n ∣ 42) : n = 6 :=
sorry

end eggs_in_each_basket_l238_238158


namespace problem_2013_factorial_l238_238135

theorem problem_2013_factorial (a_1 a_2 : ℕ) (b_1 b_2 : ℕ) (m n : ℕ) 
  (h1 : a_1! * a_2! = b_1! * b_2! * 2013)
  (h2 : a_1 ≥ a_2)
  (h3 : b_1 ≥ b_2)
  (h4 : ∀ x y, (x + y < a_1 + b_1) → ¬(x! * a_2! = y! * b_2! * 2013)) :
  |a_1 - b_1| = 2 :=
sorry

end problem_2013_factorial_l238_238135


namespace negation_of_odd_cube_l238_238895

theorem negation_of_odd_cube :
  ¬ (∀ n : ℤ, n % 2 = 1 → n^3 % 2 = 1) ↔ ∃ n : ℤ, n % 2 = 1 ∧ n^3 % 2 = 0 :=
sorry

end negation_of_odd_cube_l238_238895


namespace not_in_range_l238_238645

noncomputable def g (x c: ℝ) : ℝ := x^2 + c * x + 5

theorem not_in_range (c : ℝ) (hc : -2 * Real.sqrt 2 < c ∧ c < 2 * Real.sqrt 2) :
  ∀ x : ℝ, g x c ≠ 3 :=
by
  intros
  sorry

end not_in_range_l238_238645


namespace least_number_125_l238_238170

theorem least_number_125 (n : ℕ) (h₁ : n = 125) : ∃ k : ℕ, n = 12 * k + 5 := by
  exists 10
  sorry

end least_number_125_l238_238170


namespace largest_interesting_number_l238_238166

def is_interesting (n : ℕ) : Prop :=
  ∀ (digits : List ℕ) (h : digits.reverse.mk_nat = n),
    ∀ (i : ℕ) (h1 : i > 0) (h2 : i < digits.length - 1),
      digits.nth_le i h2 < (digits.nth_le (i - 1) sorry + digits.nth_le (i + 1) sorry) / 2

theorem largest_interesting_number :
  ∃ n : ℕ, is_interesting n ∧ n = 96433469 :=
by
  existsi 96433469
  split
  { sorry } -- Proof that 96433469 is interesting goes here
  { refl } -- Trivially true since we are claiming the number itself

end largest_interesting_number_l238_238166


namespace rectangle_midpoint_y_coordinate_l238_238908

theorem rectangle_midpoint_y_coordinate :
  ∀ (y : ℝ), (∃ l : ℝ → ℝ, ∀ x : ℝ, l x = 0.2 * x) → 
  (∃ y' : ℝ, y' = 2 * 1) →
  ((1, 0), (9, 0), (1, y), (9, y)) ∧ (0, 0) ∧ l = (0.2 * x) → y = 2 :=
by
  sorry

end rectangle_midpoint_y_coordinate_l238_238908


namespace power_mean_inequality_l238_238015

theorem power_mean_inequality (a b : ℝ) (n : ℕ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hn : 0 < n) :
  (a^n + b^n) / 2 ≥ ((a + b) / 2)^n := 
by
  sorry

end power_mean_inequality_l238_238015


namespace taxi_fare_fraction_l238_238583

theorem taxi_fare_fraction
  (cost_first_part : ℝ)
  (cost_per_additional_part : ℝ)
  (total_cost : ℝ)
  (total_distance : ℝ)
  (additional_distance : ℝ)
  (num_additional_parts : ℝ)
  (length_per_part : ℝ):
  cost_first_part = 2.10 →
  cost_per_additional_part = 0.40 →
  total_cost = 17.70 →
  total_distance = 8 →
  additional_distance = total_distance - 1 →
  total_cost = cost_first_part + cost_per_additional_part * num_additional_parts →
  num_additional_parts = additional_distance / length_per_part →
  length_per_part = 7 / 39
  sorry

end taxi_fare_fraction_l238_238583


namespace total_number_of_games_l238_238230

theorem total_number_of_games (n : ℕ) (h : n = 10) : (nat.choose n 2) = 45 := by
  have h_choose : (nat.choose 10 2) = 45 := sorry
  rw h at h_choose
  exact h_choose

end total_number_of_games_l238_238230


namespace divide_rope_into_parts_l238_238646

theorem divide_rope_into_parts:
  (∀ rope_length : ℝ, rope_length = 5 -> ∀ parts : ℕ, parts = 4 -> (∀ i : ℕ, i < parts -> ((rope_length / parts) = (5 / 4)))) :=
by sorry

end divide_rope_into_parts_l238_238646


namespace most_probable_sellable_samples_l238_238131

/-- Prove that the most probable number k of sellable samples out of 24,
given each has a 0.6 probability of being sellable, is either 14 or 15. -/
theorem most_probable_sellable_samples (n : ℕ) (p : ℝ) (q : ℝ) (k₀ k₁ : ℕ) 
  (h₁ : n = 24) (h₂ : p = 0.6) (h₃ : q = 1 - p)
  (h₄ : 24 * p - q < k₀) (h₅ : k₀ < 24 * p + p) 
  (h₆ : k₀ = 14) (h₇ : k₁ = 15) :
  (k₀ = 14 ∨ k₀ = 15) :=
  sorry

end most_probable_sellable_samples_l238_238131


namespace least_lambda_l238_238352

theorem least_lambda 
  (n : ℕ) (hn : 0 < n)
  (x : Finₓ (n) → ℝ)
  (hx : ∀ i, 0 < x i ∧ x i < (π / 2))
  (htan_prod : (∏ i in Finₓ.range n, Real.tan (x i)) = 2^(n / 2)) :
  ∃ λ > 0,
  (∑ i in Finₓ.range n, Real.cos (x i)) ≤ 
  if n = 1 ∨ n = 2 then (n / Real.sqrt 3) else (n - 1) :=
sorry

end least_lambda_l238_238352


namespace sum_consecutive_odds_seventh_power_l238_238295

theorem sum_consecutive_odds_seventh_power :
  ∃ (n : ℕ), (∑ i in (finset.range 1000).map (λ i, 2 * n - 999 + (2 * i : ℕ)), (2 * n - 999 + (2 * i : ℕ))) = 10^7 :=
by sorry

end sum_consecutive_odds_seventh_power_l238_238295


namespace time_for_second_half_l238_238941

-- Given conditions as Lean definitions
def distance_total := 40
def distance_half := distance_total / 2
def time_first_half (v : ℝ) := distance_half / v
def speed_after_injury (v : ℝ) := v / 2
def time_second_half (v : ℝ) := distance_half / speed_after_injury(v)
def additional_time := 5

-- The problem statement translated into a Lean 4 theorem
theorem time_for_second_half (v : ℝ) (h : time_second_half(v) = time_first_half(v) + additional_time) :
  time_second_half(v) = 10 := by
  sorry

end time_for_second_half_l238_238941


namespace crackers_per_friend_l238_238490

theorem crackers_per_friend
  (total_crackers : ℕ)
  (friends : ℕ)
  (remaining_crackers : ℕ)
  (total_crackers = 15)
  (friends = 5)
  (remaining_crackers = 10)
  : total_crackers - remaining_crackers = friends * 1 := 
by
  sorry

end crackers_per_friend_l238_238490


namespace max_remaining_numbers_l238_238844

theorem max_remaining_numbers : 
  ∃ s : Finset ℕ, s ⊆ (Finset.range 236) ∧ (∀ x y ∈ s, x ≠ y → ¬ (x - y).abs ∣ x) ∧ s.card = 118 := 
by
  sorry

end max_remaining_numbers_l238_238844


namespace same_sign_iff_product_positive_different_sign_iff_product_negative_l238_238729

variable (a b : ℝ)

theorem same_sign_iff_product_positive :
  ((a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0)) ↔ (a * b > 0) :=
sorry

theorem different_sign_iff_product_negative :
  ((a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)) ↔ (a * b < 0) :=
sorry

end same_sign_iff_product_positive_different_sign_iff_product_negative_l238_238729


namespace sister_conic_sections_l238_238163

theorem sister_conic_sections 
    (b : ℝ) (hb : 0 < b ∧ b < 2)
    (e1 e2 : ℝ) 
    (he : e1 * e2 = real.sqrt 15 / 4) 
    (M N : ℝ × ℝ) : 
    (∀ (x y : ℝ), (x * x / 4 + y * y / (b * b) = 1 → 
      (∃ (C2 : ℝ × ℝ → Prop), (∀ (x y : ℝ), C2 (x, y) ↔ (x * x / 4 - y * y = 1)))) ∧ 
    ((kAM kBN : ℝ) → ((kAM / kBN = -1 / 3) ∧ ∀ w : ℝ, (w = kAM^2 + ⅔ * kBN → 
      ((-3 / 4 < w ∧ w < -11 / 36) ∨ (13 / 36 < w ∧ w < 5 / 4))))) :=
by 
  intros x y hxy
  existsi (λ (xy : ℝ × ℝ), xy.1 * xy.1 / 4 - xy.2 * xy.2 = 1)
  split
  { intro h
    split; intro heq
    { exact eq_of_heq heq }
    { use [real.sqrt 3, -real.sqrt 5],
      split; linarith } }
  { intros kAM kBN hw
    split; reflexivity
    intro w
    exact sorry }

end sister_conic_sections_l238_238163


namespace mean_and_variance_l238_238286

def scores_A : List ℝ := [8, 9, 14, 15, 15, 16, 21, 22]
def scores_B : List ℝ := [7, 8, 13, 15, 15, 17, 22, 23]

noncomputable def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)
noncomputable def variance (l : List ℝ) : ℝ := mean (l.map (λ x => (x - (mean l)) ^ 2))

theorem mean_and_variance :
  (mean scores_A = mean scores_B) ∧ (variance scores_A < variance scores_B) :=
by
  sorry

end mean_and_variance_l238_238286


namespace max_rabbits_l238_238078

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l238_238078


namespace maximum_rabbits_condition_l238_238108

-- Define the conditions and constraints
variables {N : ℕ}
variables (total_rabbits long_ears jump_far : ℕ)
variables (at_least_three_with_both : Prop)

-- State the conditions with exact values and assumptions
def conditions := 
  total_rabbits = N ∧
  long_ears = 13 ∧
  jump_far = 17 ∧
  at_least_three_with_both = (∃ a b c : ℕ, a >= 3 ∧ b = (long_ears - a) ∧ c = (jump_far - a))

-- State the theorem to be proved
theorem maximum_rabbits_condition :
  ∀ {N : ℕ}, conditions N long_ears jump_far at_least_three_with_both → N ≤ 27 :=
by sorry

end maximum_rabbits_condition_l238_238108


namespace celebrity_matching_probability_l238_238589

noncomputable def probability_correct_match : ℚ := 1 / 24

theorem celebrity_matching_probability :
  let n := 4 in
  let total_arrangements := factorial n in
  let correct_arrangements := 1 in
  (correct_arrangements / total_arrangements : ℚ) = probability_correct_match :=
by
  sorry

end celebrity_matching_probability_l238_238589


namespace find_unknown_numbers_l238_238967

-- Define the known car numbers
def car1 : ℕ := 119
def car2 : ℕ := 179

-- Define the conditions for the unknown car numbers
def contains_digit_3 (n : ℕ) : Prop :=
  ∃ m, n = m * 10^2 + 3 * 10 + n % 10 ∨ n = m * 10^2 + (n % 100 / 10) * 10 + 3 ∨ n = 3 * 10^2 + n % 100

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- The main problem statement encapsulating all given conditions and asking for proof
theorem find_unknown_numbers (x y : ℕ) (hx : is_three_digit x) (hy : is_three_digit y)
  (contains3x : contains_digit_3 x) (contains3y : contains_digit_3 y)
  (h₁ : x * y + car1 * car2 = 105080) : 
  (x = 337 ∧ y = 363) ∨ (x = 363 ∧ y = 337) :=
begin
  sorry
end

end find_unknown_numbers_l238_238967


namespace letter_arrangements_l238_238730

theorem letter_arrangements :
  (∑ j in Finset.range 5, ∑ m in Finset.range 6, 
  Nat.choose 5 j * Nat.choose 5 m * Nat.multichoose 5 [4 - j, 6 - 5 + j - m, m]) 
  = -- Required expression
  sorry

end letter_arrangements_l238_238730


namespace well_depth_l238_238965

theorem well_depth (diameter volume : ℝ) (h_diam : diameter = 2) (h_vol : volume = 31.41592653589793) :
  ∃ depth : ℝ, depth = 10 :=
by
  -- Define the radius
  let radius := diameter / 2

  -- Calculate the volume using the cylinder volume formula
  have volume_def : volume = Real.pi * radius^2 * 10, from sorry

  -- Prove that depth = 10
  use 10
  exact sorry

end well_depth_l238_238965


namespace maximum_numbers_no_divisible_difference_l238_238824

theorem maximum_numbers_no_divisible_difference :
  ∃ (S : Finset ℕ), S ⊆ Finset.range 236 ∧ 
  (∀ a b c ∈ S, a ≠ b → b ≠ c → c ≠ a → ¬ (a - b = 0) ∨ ¬ (c ∣ (a - b))) ∧ S.card ≤ 118 :=
by
  sorry

end maximum_numbers_no_divisible_difference_l238_238824


namespace max_remaining_numbers_l238_238835

/-- 
The board initially has numbers 1, 2, 3, ..., 235.
Among the remaining numbers, no number is divisible by the difference of any two others.
Prove that the maximum number of numbers that could remain on the board is 118.
-/
theorem max_remaining_numbers : 
  ∃ S : set ℕ, (∀ a ∈ S, 1 ≤ a ∧ a ≤ 235) ∧ (∀ a b ∈ S, a ≠ b → ¬ ∃ d, d ∣ (a - b)) ∧ 
  ∃ T : set ℕ, S ⊆ T ∧ T ⊆ finset.range 236 ∧ T.card = 118 := 
sorry

end max_remaining_numbers_l238_238835


namespace chocolate_chip_cookies_l238_238964

theorem chocolate_chip_cookies (chocolate_chips_per_recipe : ℕ) (num_recipes : ℕ) (total_chocolate_chips : ℕ) 
  (h1 : chocolate_chips_per_recipe = 2) 
  (h2 : num_recipes = 23) 
  (h3 : total_chocolate_chips = chocolate_chips_per_recipe * num_recipes) : 
  total_chocolate_chips = 46 :=
by
  rw [h1, h2] at h3
  exact h3

-- sorry

end chocolate_chip_cookies_l238_238964


namespace distance_squared_to_center_l238_238254

theorem distance_squared_to_center (B : ℝ × ℝ) (A C : ℝ × ℝ) :
  let O := (2, 3)
  ∧ (A.1 - 2)^2 + (A.2 - 3 + 8)^2 = 64
  ∧ (C.1 + 3 - 2)^2 + (C.2 - 3)^2 = 64
  ∧ (B.1, B.2) = B 
  ∧ (A.1, A.2) = (B.1, B.2 + 8)
  ∧ (C.1, C.2) = (B.1 + 3, B.2)
  ∧ (A.2 - C.2)^2 + (A.1 - C.1)^2 = 64
  → (B.1 - 2)^2 + (B.2 - 3)^2 = 41 := by
  sorry

end distance_squared_to_center_l238_238254


namespace probability_draws_exactly_five_l238_238685

/-- Condition: A bag contains cards labeled with numbers 1, 2, and 3. -/
def cards : Finset ℕ := {1, 2, 3}

/-- Condition: Calculating number of possible outcomes when drawing 5 times with replacement -/
def total_outcomes : ℕ := cards.card ^ 5

/-- Condition: Calculating number of successful outcomes where all three cards are drawn by the 5th draw -/
def successful_outcomes : ℕ := 
  (Finset.card (Finset.powersetLen 2 cards) * (2^4 - 2))

/-- Probability of successful outcomes over total possible outcomes -/
def probability : ℚ := successful_outcomes / total_outcomes

/-- The proof problem: The probability of stopping exactly after 5 draws is 14/81 -/
theorem probability_draws_exactly_five : probability = 14 / 81 := by
  sorry

end probability_draws_exactly_five_l238_238685


namespace simplify_fraction_l238_238056

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l238_238056


namespace collinear_A1_B1_C1_D1_l238_238266

noncomputable theory

open EuclideanGeometry

variable {P : Type} [EuclideanGeometry P]
variable (A B C D O A1 B1 C1 D1 : P)

-- Conditions
def circumscribed_quadrilateral_around_circle (A B C D O : P) : Prop :=
  ∃ M H N K : P, 
  tangent_to_circle M A D O ∧
  tangent_to_circle H A B O ∧
  tangent_to_circle N B C O ∧
  tangent_to_circle K C D O

def altitudes_in_triangle_AOB (A O B A1 B1 : P) : Prop :=
  altitude_from A O B A1 ∧ altitude_from B O A B1

def altitudes_in_triangle_COD (C O D C1 D1 : P) : Prop :=
  altitude_from C O D C1 ∧ altitude_from D O C D1

-- Prove that A1, B1, C1 and D1 are collinear
theorem collinear_A1_B1_C1_D1
  (h1 : circumscribed_quadrilateral_around_circle A B C D O)
  (h2 : altitudes_in_triangle_AOB A O B A1 B1) 
  (h3 : altitudes_in_triangle_COD C O D C1 D1) :
  collinear {A1, B1, C1, D1} :=
sorry

end collinear_A1_B1_C1_D1_l238_238266


namespace find_HM_is_correct_l238_238232

/-- Define the points and sides of the triangle --/
structure Triangle :=
  (A B C : ℝ)
  (AB AC BC : ℝ)
  (AB_pos : AB > 0)
  (AC_pos : AC > 0)
  (BC_pos : BC > 0)
  (valid_triangle : AB + AC > BC ∧ AB + BC > AC ∧ AC + BC > AB)

noncomputable def Length_HM (T : Triangle) (H : ℝ) (M : ℝ) : ℝ :=
  if (H > 0 ∧ H < T.AB) ∧ (M > 0 ∧ M < T.BC) then (1 / 6) * T.BC else 0

noncomputable def find_HM (T : Triangle) (H : ℝ) (M : ℝ) : ℝ := T.AC * (1 / 5) / T.AB

theorem find_HM_is_correct (T : Triangle)
  (HT_similar : ∃ H M : ℝ, (H : ℝ) > 0 ∧ (H : ℝ) < T.AB ∧ (M : ℝ) > 0 ∧ (M : ℝ) < T.BC ∧ 
                              (Triangle.AB / Triangle.AC) = (H / Triangle.BC) ) :
  find_HM T H M = 7 / 3 :=
by
  sorry

end find_HM_is_correct_l238_238232


namespace circumference_difference_is_correct_l238_238911

noncomputable def area_A : ℝ := 198.4
noncomputable def area_B : ℝ := 251.1
noncomputable def pi_approx : ℝ := 3.1

noncomputable def radius (A : ℝ) : ℝ := real.sqrt (A / pi_approx)
noncomputable def circumference (r : ℝ) : ℝ := 2 * pi_approx * r

noncomputable def radius_A : ℝ := radius area_A
noncomputable def radius_B : ℝ := radius area_B
noncomputable def circumference_A : ℝ := circumference radius_A
noncomputable def circumference_B : ℝ := circumference radius_B
noncomputable def difference_in_circumference : ℝ := circumference_B - circumference_A

theorem circumference_difference_is_correct :
  difference_in_circumference = 6.2 :=
by
  sorry

end circumference_difference_is_correct_l238_238911


namespace chess_tournament_l238_238247

theorem chess_tournament (n : ℕ) (h1 : 10 * 9 * n / 2 = 90) : n = 2 :=
by
  sorry

end chess_tournament_l238_238247


namespace arithmetic_sequence_a10_gt_0_l238_238355

variable {α : Type*} [LinearOrderedField α]

-- Definitions of the conditions
def arithmetic_sequence (a : ℕ → α) := ∀ n1 n2, a n1 - a n2 = (n1 - n2) * (a 1 - a 0)
def a9_lt_0 (a : ℕ → α) := a 9 < 0
def a1_add_a18_gt_0 (a : ℕ → α) := a 1 + a 18 > 0

-- The proof statement
theorem arithmetic_sequence_a10_gt_0 
  (a : ℕ → α) 
  (h_arith : arithmetic_sequence a) 
  (h_a9 : a9_lt_0 a) 
  (h_a1_a18 : a1_add_a18_gt_0 a) : 
  a 10 > 0 := 
sorry

end arithmetic_sequence_a10_gt_0_l238_238355


namespace coefficient_ten_is_29_l238_238664

noncomputable def coeff_ten : polynomial ℚ :=
  (polynomial.C (1:ℚ) + polynomial.X + polynomial.X⁻¹)^3 *
  (polynomial.C (1:ℚ) + polynomial.X^(2:ℤ) + polynomial.X⁻²)^3 *
  (polynomial.C (1:ℚ) + polynomial.X^(5:ℤ) + polynomial.X⁻⁵)

theorem coefficient_ten_is_29 :
  polynomial.coeff coeff_ten 10 = 29 :=
  sorry

end coefficient_ten_is_29_l238_238664


namespace carl_marbles_l238_238993

theorem carl_marbles (initial : ℕ) (lost_frac : ℚ) (additional : ℕ) (gift : ℕ) (lost : ℕ)
  (initial = 12) 
  (lost_frac = 1 / 2)
  (additional = 10)
  (gift = 25)
  (lost = initial * lost_frac) :
  ((initial - lost) + additional + gift = 41) :=
sorry

end carl_marbles_l238_238993


namespace max_non_managers_l238_238570

theorem max_non_managers (n_mngrs n_non_mngrs : ℕ) (hmngrs : n_mngrs = 8) 
                (h_ratio : (5 : ℚ) / 24 < (n_mngrs : ℚ) / n_non_mngrs) :
                n_non_mngrs ≤ 38 :=
by {
  sorry
}

end max_non_managers_l238_238570


namespace solve_equation_roots_l238_238309

theorem solve_equation_roots :
  ∀ x : ℝ, (3*x^2 + 1)/(x - 2) - (3*x + 8)/4 + (5 - 9*x)/(x - 2) + 2 = 0 ↔ x ≈ 3.29 ∨ x ≈ -0.40 :=
by
  sorry

end solve_equation_roots_l238_238309


namespace toms_age_ratio_l238_238922

variable (T N : ℕ)

def toms_age_condition : Prop :=
  T = 3 * (T - 4 * N) + N

theorem toms_age_ratio (h : toms_age_condition T N) : T / N = 11 / 2 :=
by sorry

end toms_age_ratio_l238_238922


namespace total_built_up_area_l238_238889

theorem total_built_up_area
    (A1 A2 A3 A4 : ℕ)
    (hA1 : A1 = 480)
    (hA2 : A2 = 560)
    (hA3 : A3 = 200)
    (hA4 : A4 = 440)
    (total_plot_area : ℕ)
    (hplots : total_plot_area = 4 * (480 + 560 + 200 + 440) / 4)
    : 800 = total_plot_area - (A1 + A2 + A3 + A4) :=
by
  -- This is where the solution will be filled in
  sorry

end total_built_up_area_l238_238889


namespace max_rabbits_l238_238112

theorem max_rabbits (N : ℕ) (h1 : ∀ k, k = N → k = 27 → true)
    (h2 : ∀ n_l : ℕ, n_l = 13 → n_l <= N)
    (h3 : ∀ n_j : ℕ, n_j = 17 → n_j <= N)
    (h4 : ∀ n_both : ℕ, n_both >= 3 → true) :
  N <= 27 :=
begin
  sorry
end

end max_rabbits_l238_238112


namespace max_min_p_l238_238332

-- Define the probability space and events
variable {Ω : Type} [ProbabilitySpace Ω]
variable (P : Event Ω → ℝ)
variable (A₁ A₂ A₃ A₄ : Event Ω)

-- State the problem: define p(A1, A2, A3, A4)
def p (A₁ A₂ A₃ A₄ : Event Ω) : ℝ :=
  P(A₁ ∩ A₂ ∩ A₃ ∩ A₄) - P(A₁) * P(A₂) * P(A₃) * P(A₄)

-- Theorem statement with the maximum and minimum values
theorem max_min_p :
  max (p A₁ A₂ A₃ A₄) = 0.4725 ∧ min (p A₁ A₂ A₃ A₄) = -0.3164 := sorry

end max_min_p_l238_238332


namespace max_x_value_l238_238011

theorem max_x_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : xy + xz + yz = 8) : 
  x ≤ 7 / 3 :=
sorry

end max_x_value_l238_238011


namespace audrey_paint_time_l238_238978

theorem audrey_paint_time :
  (∃ A : ℝ, (1 / 5 + 1 / A = 11 / 30) → A = 6) :=
begin
  sorry
end

end audrey_paint_time_l238_238978


namespace mysterious_jungle_l238_238765

inductive Species
| Parrot : Species
| Crow : Species

open Species

def Alan := Species
def Bob := Species
def Carol := Species
def David := Species
def Emma := Species

def Alan_statement (alan : Alan) (emma : Emma) : Prop :=
  (alan = Parrot ∧ emma = Crow) ∨ (alan = Crow ∧ emma = Parrot)

def Bob_statement (david : David) : Prop :=
  (Bob = Parrot → david = Crow) ∧ (Bob = Crow → david ≠ Crow)

def Carol_statement : Prop :=
  (Carol = Parrot → Bob = Crow) ∧ (Carol = Crow → Bob ≠ Crow)

def David_statement : Prop :=
  (David = Parrot → ∃ a b c : Species, a ≠ Crow ∧ b ≠ Crow ∧ c ≠ Crow ∧ a ≠ David ∧ b ≠ David ∧ c ≠ David) ∧ (David = Crow → ¬ exists A : fin 5→ Species, (A 0 ≠ Crow ∧ A 1 ≠ Crow ∧ A 2 ≠ Crow ∧ A 3 ≠ Crow ∧ A 4 ≠ Crow))

def Emma_statement : Prop :=
  (Emma = Parrot → Carol = Crow) ∧ (Emma = Crow → Carol ≠ Crow)

theorem mysterious_jungle :
  ∃ (Alan Bob Carol David Emma : Species),
    Alan_statement Alan Emma ∧
    Bob_statement David ∧
    Carol_statement ∧
    David_statement ∧
    Emma_statement ∧
    (Alan + Bob + Carol + David + Emma = 3) :=
sorry

end mysterious_jungle_l238_238765


namespace sum_fraction_inequality_l238_238524

theorem sum_fraction_inequality {x : Fin 5 → ℝ} (hx : ∀ i, 0 ≤ x i) 
  (h_sum : ∑ i, 1 / (1 + x i) = 1) : ∑ i, x i / (4 + (x i)^2) ≤ 1 :=
by
  sorry

end sum_fraction_inequality_l238_238524


namespace reptile_house_animal_multiple_l238_238064

theorem reptile_house_animal_multiple (R F x : ℕ) (hR : R = 16) (hF : F = 7) (hCond : R = x * F - 5) : x = 3 := by
  sorry

end reptile_house_animal_multiple_l238_238064


namespace rectangle_length_from_circle_perimeter_l238_238555

-- Definitions based on conditions
def radius (P : ℝ) : ℝ := P / (2 * 4.14)

def circle_perimeter_to_length (P : ℝ) : ℝ :=
  let r := radius P
  3.14 * r

-- Theorem based on the question and correct answer
theorem rectangle_length_from_circle_perimeter (P : ℝ) (h : P = 20.7) :
  circle_perimeter_to_length P = 7.85 :=
by
  -- proof goes here
  sorry

end rectangle_length_from_circle_perimeter_l238_238555


namespace goose_eggs_calculation_l238_238229

theorem goose_eggs_calculation (E : ℝ) (hatch_fraction : ℝ) (survived_first_month_fraction : ℝ) 
(survived_first_year_fraction : ℝ) (survived_first_year : ℝ) (no_more_than_one_per_egg : Prop) 
(h_hatch : hatch_fraction = 1/3) 
(h_month_survival : survived_first_month_fraction = 3/4)
(h_year_survival : survived_first_year_fraction = 2/5)
(h_survived120 : survived_first_year = 120)
(h_no_more_than_one : no_more_than_one_per_egg) :
  E = 1200 :=
by
  -- Convert the information from conditions to formulate the equation
  sorry


end goose_eggs_calculation_l238_238229


namespace find_point_P_no_perpendicular_lines_l238_238360

-- Definition of the foci of the ellipse
def F1 : ℝ × ℝ := (-real.sqrt 3, 0)
def F2 : ℝ × ℝ := (real.sqrt 3, 0)

-- Equation of the ellipse and circle
def on_ellipse (P : ℝ × ℝ) : Prop := (P.1 ^ 2) / 4 + P.2 ^ 2 = 1
def on_circle (P : ℝ × ℝ) : Prop := P.1 ^ 2 + P.2 ^ 2 = 1 / 4

-- Given conditions
def in_first_quadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 > 0
def dot_product (P : ℝ × ℝ) : ℝ := 
  let PF1 := (-real.sqrt 3 - P.1, - P.2)
  let PF2 := (real.sqrt 3 - P.1, - P.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2

-- Problem 1: Finding point P
theorem find_point_P (P : ℝ × ℝ) (h1 : on_ellipse P) (h2 : dot_product P = -5 / 4) (h3 : in_first_quadrant P) :
  P = (1, real.sqrt 3 / 2) := 
sorry

-- Problem 2: No such line l exists
theorem no_perpendicular_lines (l : ℝ → ℝ)
  (tangent_line : ∀ (P : ℝ × ℝ), (on_circle P) → l P.1 = P.2)
  (A B : ℝ × ℝ) (hA : on_ellipse A) (hB : on_ellipse B) :
  ¬(let O := (0,0) in (O.1 * A.1 + O.2 * A.2) = 0 ∧ (O.1 * B.1 + O.2 * B.2) = 0) :=
sorry

end find_point_P_no_perpendicular_lines_l238_238360


namespace max_remained_numbers_l238_238849

theorem max_remained_numbers (S : Finset ℕ) (hSubset : S ⊆ Finset.range 236)
  (hCondition : ∀ a b c ∈ S, a ≠ b → a ≠ c → b ≠ c → ¬(b - a ∣ c)) : S.card ≤ 118 := 
sorry

end max_remained_numbers_l238_238849


namespace max_rabbits_l238_238115

theorem max_rabbits (N : ℕ) (h1 : ∀ k, k = N → k = 27 → true)
    (h2 : ∀ n_l : ℕ, n_l = 13 → n_l <= N)
    (h3 : ∀ n_j : ℕ, n_j = 17 → n_j <= N)
    (h4 : ∀ n_both : ℕ, n_both >= 3 → true) :
  N <= 27 :=
begin
  sorry
end

end max_rabbits_l238_238115


namespace max_S_n_value_arithmetic_sequence_l238_238354

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

axiom arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

axiom sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = ∑ i in finset.range n, a i

axiom S3_eq_S10 (S : ℕ → ℝ) : S 3 = S 10

axiom a1_positive (a : ℕ → ℝ) : 0 < a 1

theorem max_S_n_value_arithmetic_sequence 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
  (arith_seq : arithmetic_sequence a d) 
  (sum_terms : sum_of_first_n_terms S a) 
  (S3_S10_equal : S 3 = S 10) 
  (a1_pos : 0 < a 1) :
  a 7 = 0 → S 6 = S 7 :=
sorry

end max_S_n_value_arithmetic_sequence_l238_238354


namespace price_of_lemonade_l238_238279

def costOfIngredients : ℝ := 20
def numberOfCups : ℕ := 50
def desiredProfit : ℝ := 80

theorem price_of_lemonade (price_per_cup : ℝ) :
  (costOfIngredients + desiredProfit) / numberOfCups = price_per_cup → price_per_cup = 2 :=
by
  sorry

end price_of_lemonade_l238_238279


namespace power_function_through_point_l238_238388

noncomputable def f (x k α : ℝ) : ℝ := k * x ^ α

theorem power_function_through_point (k α : ℝ) (h : f (1/2) k α = Real.sqrt 2) : 
  k + α = 1/2 := 
by 
  sorry

end power_function_through_point_l238_238388


namespace clock_correct_time_fraction_l238_238580

/-- A 12-hour digital clock problem:
A 12-hour digital clock displays the hour and minute of a day.
Whenever it is supposed to display a '1' or a '2', it mistakenly displays a '9'.
The fraction of the day during which the clock shows the correct time is 7/24.
-/
theorem clock_correct_time_fraction : (7 : ℚ) / 24 = 7 / 24 :=
by sorry

end clock_correct_time_fraction_l238_238580


namespace min_value_at_3_l238_238201

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end min_value_at_3_l238_238201


namespace hyperbola_eccentricity_correct_l238_238464

noncomputable def hyperbolaEccentricity (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (F1 F2 P : EuclideanSpace ℝ (Fin 2))
  (O : EuclideanSpace ℝ (Fin 2))
  (P_on_hyperbola : (P.1 * P.1) / (a * a) - (P.2 * P.2) / (b * b) = 1)
  (condition1 : (P + F2) ∙ (F2 - P) = 0)
  (condition2 : dist P F1 = sqrt 3 * dist P F2) : 
  ℝ := 
  sqrt 3 + 1

theorem hyperbola_eccentricity_correct {a b : ℝ} (a_pos : a > 0) (b_pos : b > 0)
  (F1 F2 P : EuclideanSpace ℝ (Fin 2))
  (O : EuclideanSpace ℝ (Fin 2))
  (P_on_hyperbola : (P.1 * P.1) / (a * a) - (P.2 * P.2) / (b * b) = 1)
  (condition1 : (P + F2) ∙ (F2 - P) = 0)
  (condition2 : dist P F1 = sqrt 3 * dist P F2) : 
  hyperbolaEccentricity a b a_pos b_pos F1 F2 P O P_on_hyperbola condition1 condition2 = sqrt 3 + 1 := 
sorry

end hyperbola_eccentricity_correct_l238_238464


namespace gamesNextMonth_l238_238449

def gamesThisMonth : ℕ := 11
def gamesLastMonth : ℕ := 17
def totalPlannedGames : ℕ := 44

theorem gamesNextMonth :
  (totalPlannedGames - (gamesThisMonth + gamesLastMonth) = 16) :=
by
  unfold totalPlannedGames
  unfold gamesThisMonth
  unfold gamesLastMonth
  sorry

end gamesNextMonth_l238_238449


namespace number_with_specific_places_l238_238263

theorem number_with_specific_places :
  ∃ (n : Real), 
    (n / 10 % 10 = 6) ∧ -- tens place
    (n / 1 % 10 = 0) ∧  -- ones place
    (n * 10 % 10 = 0) ∧  -- tenths place
    (n * 100 % 10 = 6) →  -- hundredths place
    n = 60.06 :=
by
  sorry

end number_with_specific_places_l238_238263


namespace ten_digit_word_partition_impossible_l238_238663

noncomputable def hamming_distance (a b : String) : ℕ :=
  (List.zipWith (≠) a.toList b.toList).count id

theorem ten_digit_word_partition_impossible :
  ¬ ∃ (G1 G2 : Set String), (∀ w1 w2 ∈ G1, hamming_distance w1 w2 ≥ 3) ∧ 
                            (∀ w1 w2 ∈ G2, hamming_distance w1 w2 ≥ 3) ∧ 
                            ∀ w ∈ (Set.univ : Set String), w ∈ G1 ∨ w ∈ G2 :=
by
  sorry

end ten_digit_word_partition_impossible_l238_238663


namespace total_balloons_l238_238913

theorem total_balloons (T : ℕ) 
    (h1 : T / 4 = 100)
    : T = 400 := 
by
  sorry

end total_balloons_l238_238913


namespace problem_ankbhattacharya_l238_238460

theorem problem_ankbhattacharya (a₁ a₂ a₃ a₄ a₅ : ℝ)
  (h1 : a₁ * a₂ + a₂ * a₃ + a₃ * a₄ + a₄ * a₅ + a₅ * a₁ = 20)
  (h2 : a₁ * a₃ + a₂ * a₄ + a₃ * a₅ + a₄ * a₁ + a₅ * a₂ = 22) :
  ∃ (m n : ℤ), (m > 0) ∧ (n > 0) ∧ (100 * m + n = 2105) :=
begin
  sorry
end

end problem_ankbhattacharya_l238_238460


namespace max_remaining_numbers_l238_238811

theorem max_remaining_numbers : 
  ∃ (S ⊆ {n | 1 ≤ n ∧ n ≤ 235}), (∀ x ∈ S, ∀ y ∈ S, x ≠ y → ¬(x ∣ (y - x))) ∧ card S = 118 :=
by
  sorry

end max_remaining_numbers_l238_238811


namespace problem1_problem2_l238_238631

-- Part 1
theorem problem1 : | - (1 / 2) | + (-1) ^ 2019 + 2 ^ (-1) - (pi - 3) ^ 0 = -1 := 
by 
  sorry

-- Part 2
theorem problem2 : -(-2) + (pi - 3.14) ^ 0 + 27 + (-1 / 3) ^ (-1) = 27 :=
by
  sorry

end problem1_problem2_l238_238631


namespace no_pos_int_squares_l238_238632

open Nat

theorem no_pos_int_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬(∃ k m : ℕ, k ^ 2 = a ^ 2 + b ∧ m ^ 2 = b ^ 2 + a) :=
sorry

end no_pos_int_squares_l238_238632


namespace simplify_expression_1_simplify_expression_2_l238_238869

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) : 4 * (a + b) + 2 * (a + b) - (a + b) = 5 * a + 5 * b :=
  sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) : (3 * m / 2) - (5 * m / 2 - 1) + 3 * (4 - m) = -4 * m + 13 :=
  sorry

end simplify_expression_1_simplify_expression_2_l238_238869


namespace mean_equality_l238_238130

theorem mean_equality (z : ℚ) (h1 : (8 + 15 + 24) / 3 = 47 / 3) (h2 : (18 + z) / 2 = 47 / 3) : z = 40 / 3 :=
by
  sorry

end mean_equality_l238_238130


namespace locus_of_points_of_tangency_is_circle_excluding_AB_l238_238692

-- Define the setup and conditions
variables {l : Type*} {A B : l} -- A and B are points on line l

-- Define the property of tangency for circles at points A and B
def circle_tangent_at (C : Type*) (l : Type*) (P : l) : Prop := sorry -- Circle C tangent to line l at point P

-- Let's state the main goal
theorem locus_of_points_of_tangency_is_circle_excluding_AB (C D : Type*) (M : l) :
  circle_tangent_at C l A → circle_tangent_at D l B → (M ∈ circle_excluding A B) := sorry

end locus_of_points_of_tangency_is_circle_excluding_AB_l238_238692


namespace simplify_fraction_l238_238054

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l238_238054


namespace circles_internally_tangent_l238_238902

theorem circles_internally_tangent :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 6 * x + 4 * y + 12 = 0) ∧ (x^2 + y^2 - 14 * x - 2 * y + 14 = 0) →
  ∃ (C1 C2 : ℝ × ℝ) (r1 r2 : ℝ),
  C1 = (3, -2) ∧ r1 = 1 ∧
  C2 = (7, 1) ∧ r2 = 6 ∧
  dist C1 C2 = r2 - r1 :=
by
  sorry

end circles_internally_tangent_l238_238902


namespace simplify_fraction_l238_238047

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 :=
by
  sorry

end simplify_fraction_l238_238047


namespace negation_of_p_l238_238389

-- Defining the proposition 'p'
def p : Prop := ∃ x : ℝ, x^3 > x

-- Stating the theorem
theorem negation_of_p : ¬p ↔ ∀ x : ℝ, x^3 ≤ x :=
by
  sorry

end negation_of_p_l238_238389


namespace jerry_total_logs_l238_238452

-- Given conditions
def pine_logs_per_tree := 80
def maple_logs_per_tree := 60
def walnut_logs_per_tree := 100

def pine_trees_cut := 8
def maple_trees_cut := 3
def walnut_trees_cut := 4

-- Formulate the problem
theorem jerry_total_logs : 
  pine_logs_per_tree * pine_trees_cut + 
  maple_logs_per_tree * maple_trees_cut + 
  walnut_logs_per_tree * walnut_trees_cut = 1220 := 
by 
  -- Placeholder for the actual proof
  sorry

end jerry_total_logs_l238_238452


namespace f_neg_x_l238_238673

noncomputable def f : ℝ → ℝ
| x := if x > 0 then 2^x + 1 else -2^(-x) - 1

lemma odd_function_f : ∀ x : ℝ, f(-x) = -f(x) :=
by {
  intros x,
  unfold f,
  split_ifs with h1 h2,
  { -- Case x > 0
    rw [neg_pos, neg_neg_pos h1],
    exact rfl, },
  { -- Case -x > 0
    rw [neg_neg],
    exact rfl, },
  { -- Case x = 0
    exfalso,
    linarith [h1, h2], }
}

theorem f_neg_x : ∀ x ∈ (set.Iio 0 : set ℝ), f(x) = -2^(-x) - 1 :=
begin
  intros x hx,
  have h_pos : -x ∈ (set.Ioi (0:ℝ)), from set.mem_Iio.mp hx,
  rw [odd_function_f x, f],
  split_ifs with h_neg neg_pos,
  { simp at h_neg, contradiction, },
  { },
  { simp at neg_pos, contradiction, },
end

end f_neg_x_l238_238673


namespace max_rabbits_with_traits_l238_238123

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l238_238123


namespace orthogonal_vectors_angle_l238_238754

open Real EuclideanSpace

variables {α : Type*} [inner_product_space ℝ α]

/-- If non-zero vectors α and β satisfy |α + β| = |α - β|, then the angle between them is 90°. -/
theorem orthogonal_vectors_angle {α β : α} (hα : α ≠ 0) (hβ : β ≠ 0) (h_eq : ∥α + β∥ = ∥α - β∥) :
  inner α β = 0 :=
sorry

end orthogonal_vectors_angle_l238_238754


namespace total_vowels_written_l238_238990

-- Define the vowels and the condition
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def num_vowels : Nat := vowels.length
def times_written : Nat := 2

-- Assert the total number of vowels written
theorem total_vowels_written : (num_vowels * times_written) = 10 := by
  sorry

end total_vowels_written_l238_238990


namespace carl_marbles_l238_238996

-- Define initial conditions
def initial_marbles : ℕ := 12
def lost_marbles : ℕ := initial_marbles / 2
def remaining_marbles : ℕ := initial_marbles - lost_marbles
def additional_marbles : ℕ := 10
def new_marbles_from_mother : ℕ := 25

-- Define the final number of marbles Carl will put back in the jar
def total_marbles_put_back : ℕ := remaining_marbles + additional_marbles + new_marbles_from_mother

-- Statement to be proven
theorem carl_marbles : total_marbles_put_back = 41 :=
by
  sorry

end carl_marbles_l238_238996


namespace area_of_triangle_ABC_l238_238924

def point : Type := ℝ × ℝ

def A : point := (2, 1)
def B : point := (1, 4)
def on_line (C : point) : Prop := C.1 + C.2 = 9
def area_triangle (A B C : point) : ℝ := 0.5 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - B.1 * A.2 - C.1 * B.2 - A.1 * C.2)

theorem area_of_triangle_ABC :
  ∃ C : point, on_line C ∧ area_triangle A B C = 2 :=
sorry

end area_of_triangle_ABC_l238_238924


namespace smallest_positive_s_for_g_eq_zero_l238_238476

def g (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x + 2 * Real.tan x + 5 * Real.cot x

theorem smallest_positive_s_for_g_eq_zero :
  ∃ s > 0, g s = 0 ∧ ⌊s⌋ = 3 :=
by
  sorry

end smallest_positive_s_for_g_eq_zero_l238_238476


namespace triangle_proof_l238_238619

-- Let A, B, C, D be points in a plane
variables {A B C D : Type} [AddCommGroup A]

-- Let vectors AB, AC, AD be elements of this group
variables (AB AC AD BD DC BC : A)

-- Let λ be a real number greater than 0
variable (λ : ℝ)
hypothesis (hλ : λ > 0)

-- Given condition
hypothesis (h : BD = λ • DC)

-- Prove that AD = (1/(1+λ)) • AB + (λ/(1+λ)) • AC
theorem triangle_proof (hBD : BD = λ • DC) :
  AD = (1 / (1 + λ)) • AB + (λ / (1 + λ)) • AC := 
sorry

end triangle_proof_l238_238619


namespace initial_weights_of_apples_l238_238287

variables {A B : ℕ}

theorem initial_weights_of_apples (h₁ : A + B = 75) (h₂ : A - 5 = (B + 5) + 7) :
  A = 46 ∧ B = 29 :=
by
  sorry

end initial_weights_of_apples_l238_238287


namespace total_results_count_l238_238536

theorem total_results_count (N : ℕ) (S : ℕ) 
  (h1 : S = 50 * N) 
  (h2 : (12 * 14) + (12 * 17) = 372)
  (h3 : S = 372 + 878) : N = 25 := 
by 
  sorry

end total_results_count_l238_238536


namespace average_temp_tues_to_fri_l238_238509

theorem average_temp_tues_to_fri (T W Th : ℕ) 
  (h1: (42 + T + W + Th) / 4 = 48) 
  (mon: 42 = 42) 
  (fri: 10 = 10) :
  (T + W + Th + 10) / 4 = 40 := by
  sorry

end average_temp_tues_to_fri_l238_238509


namespace translation_complex_l238_238599

theorem translation_complex (z w : ℂ) : 
  (∀ z, z + (4 + 7i - 1 - 3i) = 2 - i + (4 + 7i - 1 - 3i)) → (2 - i + (4 + 7i - 1 - 3i) = 5 + 3i) :=
by
  sorry

end translation_complex_l238_238599


namespace problem_proof_l238_238701

noncomputable def problem (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y ≥ 0 ∧ x ^ 2019 + y = 1) → (x + y ^ 2019 > 1 - 1 / 300)

theorem problem_proof (x y : ℝ) : problem x y :=
by
  intros h
  sorry

end problem_proof_l238_238701


namespace find_1989th_term_l238_238573

noncomputable def sequence_term (k : ℕ) : ℚ :=
  let ⟨n, m⟩ := 
    if k = 0 then (1, 1) 
    else 
      let g := nat.find (λ g, k ≤ g * (g + 1) / 2) in
      let i := k - (g * (g - 1) / 2) in
      (g + 1 - i, i) 
  in 
  m / n

theorem find_1989th_term : 
  sequence_term 1989 = 7 / 9 :=
sorry

end find_1989th_term_l238_238573


namespace sum_of_alan_and_bob_l238_238666

theorem sum_of_alan_and_bob (ages : Fin 4 → ℕ) (h1 : multiset.of_fn ages = {3, 8, 12, 14})
  (alan carl dan : Fin 4)
  (h2 : ages alan < ages carl)
  (h3 : ∃ d : Fin 4, d ≠ dan ∧ (ages alan + ages d) % 5 = 0)
  (h4 : (ages carl + ages dan) % 5 = 0)
  (h5 : ∃ bob : Fin 4, bob ≠ carl ∧ bob ≠ alan ∧ bob ≠ dan) :
  (ages alan + ages (classical.some h5)) = 17 :=
by
  sorry

end sum_of_alan_and_bob_l238_238666


namespace max_rabbits_l238_238079

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l238_238079


namespace max_followers_l238_238853

def island_inhabitants : Type := {n // n = 2018}

constant knights : ℕ
constant liars : ℕ
constant followers : ℕ

axiom total_residents : 2018 = knights + liars + followers
axiom yes_responses : 1009 = followers + (if knights > liars then knights else liars)
axiom no_responses : 1009 = liars + followers  -- This assumes balanced no responses due to conditional nature in problem

theorem max_followers (K L F: ℕ) (h1: K + L + F = 2018) (h2: 1009 = F + if K > L then K else L) (h3: 1009 = L + F) : F ≤ 1009 :=
sorry

end max_followers_l238_238853


namespace termite_ridden_not_collapsing_l238_238803

theorem termite_ridden_not_collapsing
  (total_termite_ridden : ℚ)
  (termite_ridden_collapsing : ℚ) :
  total_termite_ridden = 1/3 →
  termite_ridden_collapsing = 5/8 →
  total_termite_ridden - (total_termite_ridden * termite_ridden_collapsing) = 1/8 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end termite_ridden_not_collapsing_l238_238803


namespace max_remaining_numbers_l238_238807

def numbers (n : ℕ) : set ℕ := {k | 1 ≤ k ∧ k ≤ n}

def valid_subset (s : set ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → (a - b) ≠ 0 → ¬ (a - b) ∣ c

theorem max_remaining_numbers : ∃ s : set ℕ, s ⊆ numbers 235 ∧ valid_subset s ∧ card s = 118 := 
sorry

end max_remaining_numbers_l238_238807


namespace average_stamps_per_day_l238_238487

noncomputable def calculate_average_stamps : ℚ :=
  let day1 := 12
  let day2 := day1 + 10
  let day3 := day2 + 10 + 5  -- adjusted for miscount
  let day4 := day2 + 20
  let day5 := day4 + 10
  let day6 := day5 + 10
  let day7 := day6 + 10
  let total_stamps := day1 + day2 + day3 + day4 + day5 + day6 + day7 in
  total_stamps / 7

theorem average_stamps_per_day :
  calculate_average_stamps = 42.714 :=
by
  sorry

end average_stamps_per_day_l238_238487


namespace max_remaining_numbers_l238_238837

/-- 
The board initially has numbers 1, 2, 3, ..., 235.
Among the remaining numbers, no number is divisible by the difference of any two others.
Prove that the maximum number of numbers that could remain on the board is 118.
-/
theorem max_remaining_numbers : 
  ∃ S : set ℕ, (∀ a ∈ S, 1 ≤ a ∧ a ≤ 235) ∧ (∀ a b ∈ S, a ≠ b → ¬ ∃ d, d ∣ (a - b)) ∧ 
  ∃ T : set ℕ, S ⊆ T ∧ T ⊆ finset.range 236 ∧ T.card = 118 := 
sorry

end max_remaining_numbers_l238_238837


namespace soy_milk_calculation_l238_238422

variable (total_milk : ℝ) (regular_milk : ℝ)

def soy_milk_drunk (total_milk regular_milk : ℝ) : ℝ := total_milk - regular_milk

theorem soy_milk_calculation 
  (h_total : total_milk = 0.6) 
  (h_regular : regular_milk = 0.5) : 
  soy_milk_drunk total_milk regular_milk = 0.1 := 
by
  rw [h_total, h_regular]
  norm_num

end soy_milk_calculation_l238_238422


namespace terminating_decimal_l238_238324

-- Define the given fraction
def frac : ℚ := 21 / 160

-- Define the decimal representation
def dec : ℚ := 13125 / 100000

-- State the theorem to be proved
theorem terminating_decimal : frac = dec := by
  sorry

end terminating_decimal_l238_238324


namespace distance_center_to_point_eq_sqrt85_l238_238169

-- Define the circle and its center
def circle_center : ℝ × ℝ := (3, -2)

-- Define the given point
def given_point : ℝ × ℝ := (-3, 5)

-- Prove the distance between the center of the circle and the given point is sqrt(85)
theorem distance_center_to_point_eq_sqrt85 :
  let c := circle_center in
  let p := given_point in
  (c.fst - p.fst)^2 + (c.snd - p.snd)^2 = 85 :=
by 
  sorry

end distance_center_to_point_eq_sqrt85_l238_238169


namespace hockey_championship_max_k_volleyball_championship_max_k_l238_238534

theorem hockey_championship_max_k : ∃ (k : ℕ), 0 < k ∧ k ≤ 20 ∧ k = 18 :=
by
  -- proof goes here
  sorry

theorem volleyball_championship_max_k : ∃ (k : ℕ), 0 < k ∧ k ≤ 20 ∧ k = 15 :=
by
  -- proof goes here
  sorry

end hockey_championship_max_k_volleyball_championship_max_k_l238_238534


namespace find_erased_integer_l238_238981

theorem find_erased_integer (s : list ℕ)
  (h1 : s = [6, 12, 1, 3, 11, 10, 8, 15, 13, 9, 7, 4, 14, 5, 2]) :
  ∃ n, n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] ∧ n ∉ s := by
  use 16
  sorry

end find_erased_integer_l238_238981


namespace range_of_b_l238_238712

noncomputable def f (x a b : ℝ) : ℝ :=
  x + a / x + b

theorem range_of_b (b : ℝ) :
  (∀ (a x : ℝ), (1/2 ≤ a ∧ a ≤ 2) ∧ (1/4 ≤ x ∧ x ≤ 1) → f x a b ≤ 10) →
  b ≤ 7 / 4 :=
by
  sorry

end range_of_b_l238_238712


namespace arithmetic_progression_l238_238557

-- Define the general formula for the nth term of an arithmetic progression
def nth_term (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the conditions given in the problem
def condition1 (a1 d : ℤ) : Prop := nth_term a1 d 13 = 3 * nth_term a1 d 3
def condition2 (a1 d : ℤ) : Prop := nth_term a1 d 18 = 2 * nth_term a1 d 7 + 8

-- The main proof problem statement
theorem arithmetic_progression (a1 d : ℤ) (h1 : condition1 a1 d) (h2 : condition2 a1 d) : a1 = 12 ∧ d = 4 :=
by
  sorry

end arithmetic_progression_l238_238557


namespace probability_at_least_two_heads_l238_238796

theorem probability_at_least_two_heads (n: ℕ) (p: ℚ) 
  (h_n: n = 4) (h_p: p = 1 / 2) : 
  let prob_two_heads := (finset.card (finset.filter (λ k, k = 2) (finset.range (n + 1)))) * (p^2) * ((1 - p)^(n - 2))
  let prob_three_heads := (finset.card (finset.filter (λ k, k = 3) (finset.range (n + 1)))) * (p^3) * ((1 - p)^(n - 3))
  let prob_four_heads := (finset.card (finset.filter (λ k, k = 4) (finset.range (n + 1)))) * (p^4) * ((1 - p)^(n - 4))
  (prob_two_heads + prob_three_heads + prob_four_heads) = 11 / 16 := 
by {
  sorry
}

end probability_at_least_two_heads_l238_238796


namespace geometric_sequence_12th_term_l238_238125

theorem geometric_sequence_12th_term 
  (a_4 a_8 : ℕ) (h4 : a_4 = 2) (h8 : a_8 = 162) :
  ∃ a_12 : ℕ, a_12 = 13122 :=
by
  sorry

end geometric_sequence_12th_term_l238_238125


namespace modular_inverse_property_l238_238316

def gcd (a : ℕ) (b : ℕ) : ℕ :=
if a = 0 then b else gcd (b % a) a

theorem modular_inverse_property (a b c d : ℕ) (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h2 : a < 9 ∧ b < 9 ∧ c < 9 ∧ d < 9)
    (h3 : gcd a 9 = 1 ∧ gcd b 9 = 1 ∧ gcd c 9 = 1 ∧ gcd d 9 = 1) :
    ((a * b * c + a * b * d + a * c * d + b * c * d) * (nat.inv_mod (a * b * c * d) 9)) % 9 = 6 :=
sorry

end modular_inverse_property_l238_238316


namespace max_rabbits_l238_238081

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l238_238081


namespace AD_perpendicular_to_BS_l238_238793

open Point Triangle Angle Circle Line Segment

variables {A B C D E F G S : Point}
variables (ABC : Triangle)
variables (k : Circle)
variables (h1 : ABC.angle_bisector A B C D)
variables (h2 : k.passes_through D)
variables (h3 : k.tangent_to_line AC E)
variables (h4 : k.tangent_to_line AB F)
variables (h5 : k.second_intersection_with BC G)
variables (h6 : Line.through E G S)
variables (h7 : Line.through D F S)
variables (h8 : ABC.side_length B C > ABC.side_length C A)

theorem AD_perpendicular_to_BS : Line.perpendicular (Line.through A D) (Line.through B S) :=
by
  sorry

end AD_perpendicular_to_BS_l238_238793


namespace max_rooks_placement_l238_238553

-- Define the chessboard size
def board_size : ℕ := 8

-- Define the condition that a rook attacks all squares on its file and rank
def attacks (r1 c1 r2 c2 : ℕ) : Prop :=
  r1 = r2 ∨ c1 = c2

-- Define the condition that each rook must attack no more than one other rook
def valid_placement (placements : list (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ) (hi hj : i < placements.length ∧ j < placements.length), 
    i ≠ j → (attacks (placements.nthLe i hi).1 (placements.nthLe i hi).2 
                      (placements.nthLe j hj).1 (placements.nthLe j hj).2 →
           ∀ (k : ℕ) (hk : k < placements.length), 
             k ≠ i ∧ k ≠ j → 
             ¬ attacks (placements.nthLe k hk).1 (placements.nthLe k hk).2 
                       (placements.nthLe i hi).1 (placements.nthLe i hi).2 
             ∧ ¬ attacks (placements.nthLe k hk).1 (placements.nthLe k hk).2 
                       (placements.nthLe j hj).1 (placements.nthLe j hj).2)

-- Define the main theorem stating the maximum number of rooks
theorem max_rooks_placement : 
  ∃ (placements : list (ℕ × ℕ)), 
    placements.length = 10 ∧ valid_placement placements :=
sorry

end max_rooks_placement_l238_238553


namespace actual_amount_paid_l238_238556

def price : ℝ := 2250
def initial_payment : ℝ := 250
def monthly_payment : ℝ := 100
def monthly_interest_rate : ℝ := 0.01
def total_months : ℕ := 20
def interest_first_term : ℝ := 20
def interest_common_difference : ℝ := 1

theorem actual_amount_paid :
  let total_interest := (interest_first_term + (interest_first_term + (total_months - 1) * interest_common_difference)) * total_months / 2 in
  let total_paid := price + total_interest in
  total_paid = 2460 := 
by
  sorry

end actual_amount_paid_l238_238556


namespace minimize_quadratic_l238_238199

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end minimize_quadratic_l238_238199


namespace similarity_of_triangles_l238_238986

universe u

variables {α : Type u} [EuclideanGeometry α] 
variables (A B C D E M N O : α)

-- Definitions for equilateral triangles and centroid
def is_equilateral_triangle (A B C : α) : Prop :=
  ∀ (P Q R : α), (dist A B = dist B C) ∧ (dist B C = dist C A)

def is_centroid (O A B C : α) : Prop :=
  ∃ (G : α), G = centroid A B C ∧ dist O G = 2/3 * dist A G

-- Midpoints definition
def is_midpoint (M X Y : α) : Prop := dist X M = dist M Y

-- Statement that we are proving
theorem similarity_of_triangles (hABC : is_equilateral_triangle A B C)
  (hCDE : is_equilateral_triangle C D E) 
  (hM : is_midpoint M B D) 
  (hN : is_midpoint N A E) 
  (hO : is_centroid O A B C) :
  similar O M E O N D := sorry

end similarity_of_triangles_l238_238986


namespace initial_volume_of_mixture_l238_238429

variable (V : ℕ)
variable (h1 : (0.10 * V) + 14 = 0.25 * (V + 14))

theorem initial_volume_of_mixture : V = 70 :=
by
  sorry

end initial_volume_of_mixture_l238_238429


namespace q_minus_r_l238_238318

noncomputable def problem (x : ℝ) : Prop :=
  (5 * x - 15) / (x^2 + x - 20) = x + 3

def q_and_r (q r : ℝ) : Prop :=
  q ≠ r ∧ problem q ∧ problem r ∧ q > r

theorem q_minus_r (q r : ℝ) (h : q_and_r q r) : q - r = 2 :=
  sorry

end q_minus_r_l238_238318


namespace additional_times_ridden_by_Matt_l238_238491

def Maurice_rides_before_visit := 8

def Total_Matt_rides (H M : ℕ) := 8 + M = 3 * H

theorem additional_times_ridden_by_Matt (H : ℕ) (hH : H = Maurice_rides_before_visit) :
  ∃ M : ℕ, Total_Matt_rides H M ∧ M = 16 :=
by
  use 16
  split
  · rw [hH]
    apply Eq.trans
    exact Total_Matt_rides
    rw [← Nat.add_sub_assoc, Nat.add_sub_assoc]
  · sorry


end additional_times_ridden_by_Matt_l238_238491


namespace geometric_sequence_l238_238069

variable {α : Type*} [LinearOrderedField α]

-- Define the geometric sequence
def geom_seq (a₁ r : α) (n : ℕ) : α := a₁ * r^(n-1)

theorem geometric_sequence :
  ∀ (a₁ : α), a₁ > 0 → geom_seq a₁ 2 3 * geom_seq a₁ 2 11 = 16 → geom_seq a₁ 2 5 = 1 :=
by
  intros a₁ h_pos h_eq
  sorry

end geometric_sequence_l238_238069


namespace sequence_contains_one_or_three_sequence_contains_three_iff_sequence_contains_one_iff_l238_238260

def f (n : ℕ) : ℕ :=
if n % 2 = 0 then n / 2 else n + 3

def a_seq (a0 : ℕ) : ℕ → ℕ
| 0     := a0
| (n+1) := f (a_seq n)

theorem sequence_contains_one_or_three (m : ℕ) (hm : m > 0) :
  ∃ n, a_seq m n = 1 ∨ a_seq m n = 3 :=
sorry

theorem sequence_contains_three_iff (m : ℕ) (hm : m > 0) :
  (∃ n, a_seq m n = 3) ↔ 3 ∣ m :=
sorry

theorem sequence_contains_one_iff (m : ℕ) (hm : m > 0) :
  (∃ n, a_seq m n = 1) ↔ ¬ (3 ∣ m) :=
sorry

end sequence_contains_one_or_three_sequence_contains_three_iff_sequence_contains_one_iff_l238_238260


namespace polina_happy_arrangements_l238_238033

theorem polina_happy_arrangements:
  ∃ ways : ℕ, ways = 24 ∧
  (∃ chains stones pendants : list (string × string × string), 
    (∀ pair1 pair2 ∈ chains, pair1 ≠ pair2) ∧
    (∃ pair1 ∈ chains, pair1.1 = "iron" ∧ pair1.2 = "sun") ∧
    (∀ i j ∈ chains, i ≠ j → i.1 ≠ j.1) ∧
    (∀ i j ∈ stones, i ≠ j → i.2 ≠ j.2) ∧
    (∀ i j ∈ pendants, i ≠ j → i.3 ≠ j.3) ∧
    next_to (some_chain_with_sun.1) ((some_chain_with_sun.1 ≠ "gold" ∧ "gold") ∪ ("silver"))
  ) :=
sorry

end polina_happy_arrangements_l238_238033


namespace relationship_between_a_and_b_l238_238760

open Polynomial

noncomputable def quadratic_eq_with_common_root (a b t : ℚ) : Prop :=
  (X^2 + C a * X + C b = 0) ∧ (X^2 + C b * X + C a = 0) ∧ (∃ x : ℚ, x = t)

theorem relationship_between_a_and_b
  (a b : ℚ) (h_diff : a ≠ b) (h_common_root : ∃ t : ℚ,
    (eval t (X^2 + C a * X + C b) = 0) ∧
    (eval t (X^2 + C b * X + C a) = 0)) :
  a + b + 1 = 0 :=
sorry

end relationship_between_a_and_b_l238_238760


namespace original_price_is_correct_l238_238969

-- Given conditions as Lean definitions
def reduced_price : ℝ := 2468
def reduction_amount : ℝ := 161.46

-- To find the original price including the sales tax
def original_price_including_tax (P : ℝ) : Prop :=
  P - reduction_amount = reduced_price

-- The proof statement to show the price is 2629.46
theorem original_price_is_correct : original_price_including_tax 2629.46 :=
by
  sorry

end original_price_is_correct_l238_238969


namespace average_test_score_45_percent_l238_238409

theorem average_test_score_45_percent (x : ℝ) 
  (h1 : 0.45 * x + 0.50 * 78 + 0.05 * 60 = 84.75) : 
  x = 95 :=
by sorry

end average_test_score_45_percent_l238_238409


namespace find_y_solution_l238_238758

noncomputable def series_sum (y : ℝ) : ℝ :=
1 + 3 * y + 5 * y^2 + 7 * y^3 + ∑' n : ℕ, ((2 * n + 1) * y^n)

theorem find_y_solution : 
  ∃ y : ℝ, (series_sum y = 16) ∧ (y = (33 - Real.sqrt 129) / 32) :=
begin
  use (33 - Real.sqrt 129) / 32,
  split,
  {
    -- The statement that series_sum ((33 - sqrt 129) / 32) equals 16 should be proved here.
    sorry
  },
  refl,
end

end find_y_solution_l238_238758


namespace exists_translations_of_parabola_l238_238447

noncomputable def translated_parabola : Prop :=
  ∃ (h k : ℝ), (∀ x : ℝ, y = -(x - h)^2 + k) ∧ k = h^2 ∧
  ∀ A : ℝ, A = 1/2 * 2 * h * k ∧ (A = 1 → h = 1 ∨ h = -1)

theorem exists_translations_of_parabola :
  translated_parabola :=
sorry

end exists_translations_of_parabola_l238_238447


namespace diagonal_sum_inequality_l238_238006

variables (P P' : Type) 
variables [convex_quadrilateral P] [convex_quadrilateral P']
variables [inside P' P]

// Definitions of diagonal sums
noncomputable def sum_diagonals (Q : Type) [convex_quadrilateral Q] : ℝ := 
  (diagonal_length Q 1) + (diagonal_length Q 2)

variables (d : ℝ) (d' : ℝ)
variables (h1 : sum_diagonals P = d) 
variables (h2 : sum_diagonals P' = d')

theorem diagonal_sum_inequality : d' < 2 * d := 
  sorry

end diagonal_sum_inequality_l238_238006


namespace find_k_l238_238741

theorem find_k (k : ℕ) : (1 / 3)^32 * (1 / 125)^k = 1 / 27^32 → k = 0 :=
by {
  sorry
}

end find_k_l238_238741


namespace proj_of_3u_l238_238789

noncomputable def proj (z u : ℝ × ℝ) : ℝ × ℝ :=
let norm_sq := z.1^2 + z.2^2 in
let dot_product := u.1 * z.1 + u.2 * z.2 in
((dot_product / norm_sq) * z.1, (dot_product / norm_sq) * z.2)

theorem proj_of_3u (u z : ℝ × ℝ) (hu : proj z u = (-1, 4)) : 
  proj z (3 * u.1, 3 * u.2) = (-3, 12) :=
by sorry

end proj_of_3u_l238_238789


namespace maximum_numbers_no_divisible_difference_l238_238820

theorem maximum_numbers_no_divisible_difference :
  ∃ (S : Finset ℕ), S ⊆ Finset.range 236 ∧ 
  (∀ a b c ∈ S, a ≠ b → b ≠ c → c ≠ a → ¬ (a - b = 0) ∨ ¬ (c ∣ (a - b))) ∧ S.card ≤ 118 :=
by
  sorry

end maximum_numbers_no_divisible_difference_l238_238820


namespace sum_of_possible_values_of_x_l238_238018

noncomputable def f (x : ℝ) : ℝ :=
  if x < 3 then 6 * x + 8 else 3 * x - 9

theorem sum_of_possible_values_of_x :
  (∀ x : ℝ, f(x) = 3 → (x = -5/6 ∨ x = 4)) ∧ 
  (∀ x1 x2 : ℝ, f(x1) = 3 → f(x2) = 3 → x1 + x2 = 19/6) :=
by
  sorry

end sum_of_possible_values_of_x_l238_238018


namespace angle_PQM_is_17_degrees_l238_238032

open EuclideanGeometry

theorem angle_PQM_is_17_degrees (A B C M P Q : Point)
  (h1: IsRightTriangle A B C)
  (h2: midpoint M A C)
  (h3: angle A B C = 90°)
  (h4: angle B C A = 90° - 17°)
  (h5: AP = PM)
  (h6: CQ = QM) :
  angle P Q M = 17° := 
sorry

end angle_PQM_is_17_degrees_l238_238032


namespace max_remaining_numbers_l238_238814

theorem max_remaining_numbers : 
  ∃ (S ⊆ {n | 1 ≤ n ∧ n ≤ 235}), (∀ x ∈ S, ∀ y ∈ S, x ≠ y → ¬(x ∣ (y - x))) ∧ card S = 118 :=
by
  sorry

end max_remaining_numbers_l238_238814


namespace minimize_f_at_3_l238_238207

-- Define the quadratic function f(x) = 3x^2 - 18x + 7
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

-- The theorem stating that f(x) attains its minimum when x = 3
theorem minimize_f_at_3 : ∀ x : ℝ, f(x) ≥ f(3) := 
by 
  sorry

end minimize_f_at_3_l238_238207


namespace max_rabbits_with_long_ears_and_jumping_far_l238_238090

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l238_238090


namespace initial_walnut_trees_l238_238537

theorem initial_walnut_trees (total_trees_after_planting : ℕ) (trees_planted_today : ℕ) (initial_trees : ℕ) : 
  (total_trees_after_planting = 55) → (trees_planted_today = 33) → (initial_trees + trees_planted_today = total_trees_after_planting) → (initial_trees = 22) :=
by
  sorry

end initial_walnut_trees_l238_238537


namespace min_value_is_3_l238_238636

theorem min_value_is_3 (a b : ℝ) (h1 : a > b / 2) (h2 : 2 * a > b) : (2 * a + b) / a ≥ 3 :=
sorry

end min_value_is_3_l238_238636


namespace sum_of_1000_consecutive_odd_is_seventh_power_l238_238297

theorem sum_of_1000_consecutive_odd_is_seventh_power :
  ∃ n : ℕ, let S := (list.range 1000).map (λ k, 2*k + 1) in (S.sum = n^7) :=
sorry

end sum_of_1000_consecutive_odd_is_seventh_power_l238_238297


namespace arithmetic_expression_eval_l238_238630

theorem arithmetic_expression_eval : 
  (1000 * 0.09999) / 10 * 999 = 998001 := 
by 
  sorry

end arithmetic_expression_eval_l238_238630


namespace region_area_is_24_l238_238876

theorem region_area_is_24 :
  (∀ x y: ℝ, (x - y + 5) * (x + y) ≥ 0 → (0 ≤ x ∧ x ≤ 3) → true) →
  let region := {p : ℝ × ℝ | (p.1 - p.2 + 5) * (p.1 + p.2) ≥ 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ 3} in
  let area := 24 in
  ∫∫ (region x) 1.dμ = area :=
sorry -- Proof is skipped as instructed

end region_area_is_24_l238_238876


namespace maximum_numbers_up_to_235_l238_238826

def max_remaining_numbers : ℕ := 118

theorem maximum_numbers_up_to_235 (numbers : set ℕ) (h₁ : ∀ n ∈ numbers, n ≤ 235)
  (h₂ : ∀ a b ∈ numbers, a ≠ b → ¬ (a - b).abs ∣ a) :
  numbers.card ≤ max_remaining_numbers :=
sorry

end maximum_numbers_up_to_235_l238_238826


namespace painters_work_l238_238448

theorem painters_work (w1 w2 : ℕ) (d1 d2 : ℚ) (C : ℚ) (h1 : w1 * d1 = C) (h2 : w2 * d2 = C) (p : w1 = 5) (t : d1 = 1.6) (a : w2 = 4) : d2 = 2 := 
by
  sorry

end painters_work_l238_238448


namespace solution_set_condition_l238_238143

-- The assumptions based on the given conditions
variables (a b : ℝ)

noncomputable def inequality_system_solution_set (x : ℝ) : Prop :=
  (x + 2 * a > 4) ∧ (2 * x - b < 5)

theorem solution_set_condition (a b : ℝ) :
  (∀ x : ℝ, inequality_system_solution_set a b x ↔ 0 < x ∧ x < 2) →
  (a + b) ^ 2023 = 1 :=
by
  intro h
  sorry

end solution_set_condition_l238_238143


namespace total_cans_from_recycling_l238_238058

noncomputable def recycleCans (n : ℕ) : ℕ :=
  if n < 6 then 0 else n / 6 + recycleCans (n / 6 + n % 6)

theorem total_cans_from_recycling:
  recycleCans 486 = 96 :=
by
  sorry

end total_cans_from_recycling_l238_238058


namespace cost_per_game_l238_238284

theorem cost_per_game 
  (x : ℝ)
  (shoe_rent : ℝ := 0.50)
  (total_money : ℝ := 12.80)
  (games : ℕ := 7)
  (h1 : total_money - shoe_rent = 12.30)
  (h2 : 7 * x = 12.30) :
  x = 1.76 := 
sorry

end cost_per_game_l238_238284


namespace geometric_sequence_sum_l238_238704

open Real

variable {a a5 a3 a4 S4 q : ℝ}

theorem geometric_sequence_sum (h1 : q < 1)
                             (h2 : a + a5 = 20)
                             (h3 : a3 * a5 = 64) :
                             S4 = 120 := by
  sorry

end geometric_sequence_sum_l238_238704


namespace range_of_m_l238_238338

def f (x : ℝ) : ℝ := log (abs (x + 3) - abs (x - 7))

theorem range_of_m (m : ℝ) : (∃ x : ℝ, f x > m) ↔ m < 1 := 
by
  sorry

end range_of_m_l238_238338


namespace fred_packs_of_football_cards_l238_238342

-- Stating the conditions 
def cost_per_pack_football := 2.73
def cost_pack_pokemon := 4.01
def cost_deck_baseball := 8.95
def total_spent := 18.42

-- Formalizing the question and expected answer in Lean
theorem fred_packs_of_football_cards (F : ℝ) 
  (h : cost_per_pack_football * F + cost_pack_pokemon + cost_deck_baseball = total_spent):
  F = 2 := by
  sorry -- Proof is skipped

end fred_packs_of_football_cards_l238_238342


namespace sum_of_coordinates_of_other_endpoint_l238_238587

-- Define the endpoint and the scaled midpoint conditions
def endpoint1 := (10, -5)
def scaledMidpoint := (12, -18)

-- The question is to determine the sum of the coordinates of the other endpoint
theorem sum_of_coordinates_of_other_endpoint :
  let midpoint := (fst scaledMidpoint / 2, snd scaledMidpoint / 2) in
  let other_endpoint_x := 2 * midpoint.1 - endpoint1.1 in
  let other_endpoint_y := 2 * midpoint.2 - endpoint1.2 in
  other_endpoint_x + other_endpoint_y = -11 :=
by
  sorry  -- proof is omitted

end sum_of_coordinates_of_other_endpoint_l238_238587


namespace maximize_grazing_area_l238_238987

noncomputable def side_length : ℝ := 12
noncomputable def rope_length : ℝ := 4
def stake_distance : ℝ := 3

-- Coordinates of the stakes around the square pond
def coordinates (point : String) : ℝ × ℝ :=
  match point with
  | "A" => (0, 0)
  | "B" => (3, 0)
  | "C" => (3, 3)
  | "D" => (0, 3)
  | _   => (0, 0) -- for any other point, default to the origin
  
theorem maximize_grazing_area : 
  let grazing_area (point : String) : ℝ :=
    match point with
    | "A" => (1/2 * Real.pi * rope_length^2) + (1/4 * Real.pi * (rope_length - stake_distance)^2)
    | "B" => (3/4 * Real.pi * rope_length^2)
    | "C" => (1/2 * Real.pi * rope_length^2) + (1/4 * Real.pi * (rope_length - stake_distance)^2)
    | "D" => (1/2 * Real.pi * rope_length^2)
    | _   => 0  -- for any other point, default to 0
  in grazing_area "B" = 12 * Real.pi
:= sorry

end maximize_grazing_area_l238_238987


namespace walnut_trees_initially_in_park_l238_238540

def initial_trees_in_park (final_trees planted_trees : ℕ) : ℕ :=
  final_trees - planted_trees

theorem walnut_trees_initially_in_park (final_trees planted_trees initial_trees : ℕ) 
  (h1 : final_trees = 55) 
  (h2 : planted_trees = 33)
  (h3 : initial_trees = initial_trees_in_park final_trees planted_trees) :
  initial_trees = 22 :=
by
  rw [initial_trees_in_park, h1, h2]
  simp
  exact h3
  sorry

end walnut_trees_initially_in_park_l238_238540


namespace zero_fraction_when_x_is_pm_5_l238_238075

theorem zero_fraction_when_x_is_pm_5 (x : ℝ) : 
  ((x = 5 ∨ x = -5) → (x^2 - 25 = 0 ∧ 4 * x^2 - 2 * x ≠ 0) → (x^2 - 25) / (4 * x^2 - 2 * x) = 0) :=
by
  intros hx hcond
  cases hx
  case left =>
    rw [hx]
    simp [*,/* simplifying expressions */]
    sorry  -- The proof step to simplify and verify the fraction goes to zero.
  case right =>
    rw [hx]
    simp [*,/* simplifying expressions */]
    sorry  -- The proof step to simplify and verify the fraction goes to zero.

end zero_fraction_when_x_is_pm_5_l238_238075


namespace eventually_zero_after_iterations_l238_238943

theorem eventually_zero_after_iterations (A B C D : ℕ) :
  ∃ n, (∃ k, A = A * 2^k) ∧ (∃ k, B = B * 2^k) ∧ (∃ k, C = C * 2^k) ∧ (∃ k, D = D * 2^k) → 
  ∃ m, (iterate (λ p : ℕ × ℕ × ℕ × ℕ, let (A, B, C, D) := p in (abs (A - B), abs (B - C), abs (C - D), abs (D - A))) m (A, B, C, D) = (0, 0, 0, 0)) :=
by sorry

end eventually_zero_after_iterations_l238_238943


namespace rationalize_denominator_l238_238037

theorem rationalize_denominator (a : ℝ) : (a : ℝ) > 0 → 
  (45 : ℝ) / (Real.cbrt 45) = 45 ^ (2 / 3) := 
by sorry

end rationalize_denominator_l238_238037


namespace tan_alpha_eq_neg_12_div_5_l238_238345

theorem tan_alpha_eq_neg_12_div_5 (α : ℝ) (h1 : sin α + cos α = 7 / 13) (h2 : α ∈ Ioo (-π) 0) :
  tan α = -12 / 5 :=
sorry

end tan_alpha_eq_neg_12_div_5_l238_238345


namespace cos_neg_angle_identity_l238_238634

theorem cos_neg_angle_identity : 
  let angle1 := -1830
  let angle2 := angle1 + 6 * 360
  let standard_angle := 30
  cos angle1 = cos standard_angle :=
by
  -- Using the given conditions
  have h1 : angle2 = 330 := by sorry
  have h2 : cos angle2 = cos 330 := by sorry
  have h3 : 330 = 360 - standard_angle := by sorry
  have h4 : cos 330 = cos (360 - standard_angle) := by sorry
  have h5 : cos (360 - standard_angle) = cos standard_angle := by sorry
  have h6 : cos (30) = real.sqrt 3 / 2 := by sorry
  exact eq.trans (eq.trans (eq.trans (eq.trans (eq.trans (cos_eq_cos_of_eq_sub_angle h1) h2) h3) h4) h5) h6

end cos_neg_angle_identity_l238_238634


namespace trigonometric_identity_in_triangle_l238_238445

theorem trigonometric_identity_in_triangle
  (A B C : ℝ)
  (triangle_ABC : A + B + C = 180)
  (right_angle : C = 90) :
  (cos A + sin A = cos B + sin B) ↔ (cos A + sin A = cos B + sin B ∧ A + B = 90) := 
sorry

end trigonometric_identity_in_triangle_l238_238445


namespace divide_segment_exactly_l238_238777

variable (A B : Point) (AB : ℝ) (r : ℝ)
-- length of the segment AB
def segment_length : ℝ := AB
-- condition for intersection at midpoint
def arc_condition : Prop := (2 * r) = (AB * Real.sqrt 2)

theorem divide_segment_exactly :
  arc_condition A B AB r → r = AB * (Real.sqrt 2) / 2 :=
by
  sorry

end divide_segment_exactly_l238_238777


namespace find_frac_a_b_c_l238_238973

theorem find_frac_a_b_c (a b c : ℝ) (h1 : a = 2 * b) (h2 : a^2 + b^2 = c^2) : (a + b) / c = (3 * Real.sqrt 5) / 5 :=
by
  sorry

end find_frac_a_b_c_l238_238973


namespace c2_equals_19_l238_238350

-- Definitions based on provided conditions
def symmetric_sequence (seq : List ℕ) : Prop :=
  ∀ n, n > 0 ∧ n ≤ seq.length / 2 → seq[n - 1] = seq[seq.length - n]

def arithmetic_sequence (seq : List ℕ) (a d : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < seq.length → seq[i] = a + i * d

-- Specific sequence and properties from the problem
def c : List ℕ := [sorry]  -- The full sequence is not provided, assuming it is given.

axiom given_sequence_params :
  symmetric_sequence c ∧
  arithmetic_sequence (c.drop 10) 1 2 ∧ -- c.drop 10 drops the first 10 elements
  c.length = 21

-- The statement to be proved
theorem c2_equals_19 : c[1] = 19 :=
by
  sorry

end c2_equals_19_l238_238350


namespace binom_eight_three_l238_238303

theorem binom_eight_three : Nat.choose 8 3 = 56 := by
  sorry

end binom_eight_three_l238_238303


namespace max_rabbits_with_long_ears_and_jumping_far_l238_238093

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l238_238093


namespace nancy_games_l238_238801

theorem nancy_games (games_last_month games_next_month total_games games_this_month : ℕ) 
  (h_last_month : games_last_month = 8) 
  (h_next_month : games_next_month = 7) 
  (h_total : total_games = 24) 
  (h_equation : total_games = games_last_month + games_this_month + games_next_month) : 
  games_this_month = 9 :=
by {
  rw [h_last_month, h_next_month, h_total] at h_equation,
  linarith,
}

end nancy_games_l238_238801


namespace mul_powers_same_base_l238_238628

theorem mul_powers_same_base (x : ℝ) : (x ^ 8) * (x ^ 2) = x ^ 10 :=
by
  exact sorry

end mul_powers_same_base_l238_238628


namespace largest_interesting_number_l238_238165

def interesting (n : ℕ) : Prop :=
  ∀ i, 1 < i ∧ i < Nat.digits n ∧ i < List.length (Nat.digits n)
  → (Nat.digits n).nth i < ((Nat.digits n).nth (i - 1) + (Nat.digits n).nth (i + 1)) / 2

theorem largest_interesting_number :
  ∃ (n : ℕ), interesting n ∧ ∀ (m : ℕ), interesting m → m ≤ n :=
  ∃ (n : ℕ), interesting n ∧ (n = 96433469)
  sorry

end largest_interesting_number_l238_238165


namespace product_of_factors_l238_238289

theorem product_of_factors : (2.1 * (53.2 - 0.2) = 111.3) := by
  sorry

end product_of_factors_l238_238289


namespace percentage_of_smaller_number_l238_238514

theorem percentage_of_smaller_number :
  let large_num := 2475
  let small_num := 825
  let difference := large_num - small_num
  7.5 / 100 * large_num = 22.5 / 100 * small_num :=
by
  let large_num := 2475
  let small_num := 825
  let difference := large_num - small_num
  have h_diff : difference = 1650 := by rfl -- Given condition
  have h_calc : 7.5 / 100 * large_num = 22.5 / 100 * small_num := by sorry -- Computation
  exact h_calc

end percentage_of_smaller_number_l238_238514


namespace billy_sleep_total_l238_238750

def billy_sleep : Prop :=
  let first_night := 6
  let second_night := first_night + 2
  let third_night := second_night / 2
  let fourth_night := third_night * 3
  first_night + second_night + third_night + fourth_night = 30

theorem billy_sleep_total : billy_sleep := by
  sorry

end billy_sleep_total_l238_238750


namespace increase_in_radius_l238_238320

theorem increase_in_radius (D1 D2 : ℝ) (r1 : ℝ) (N1 N2 : ℝ) (r2 : ℝ) :
  D1 = 450 →
  D2 = 440 →
  r1 = 15 →
  (N1 * 2 * Real.pi * r1 = 450) →
  (N2 * 2 * Real.pi * r2 = 440) →
  N1 / N2 = 450 / 440 →
  r2 = 15 * (450 / 440) →
  r2 - r1 = 0.34 :=
begin
  sorry
end

end increase_in_radius_l238_238320


namespace cos_7x_eq_cos_5x_has_7_solutions_l238_238136

theorem cos_7x_eq_cos_5x_has_7_solutions : 
  let solutions := {x | 0 ≤ x ∧ x ≤ π ∧ cos (7 * x) = cos (5 * x)}
  finset.card (solutions.to_finset) = 7 :=
by 
  sorry

end cos_7x_eq_cos_5x_has_7_solutions_l238_238136


namespace childrenNumber_l238_238762

-- Definitions following directly from the conditions
def totalCookies : ℕ := 120
def cookiesPerChild : ℕ := 20
def fractionEatenByAdults : ℚ := 1 / 3

-- The number of children is what we want to determine
def numberOfChildren (total : ℕ) (perChild : ℕ) (fractionEaten : ℚ) : ℕ :=
  let eatenByAdults := (fractionEaten * total).toNat
  let remainingCookies := total - eatenByAdults
  remainingCookies / perChild

-- Proof stating that under the given conditions, the number of children is 4
theorem childrenNumber : numberOfChildren totalCookies cookiesPerChild fractionEatenByAdults = 4 := 
  by
  sorry

end childrenNumber_l238_238762


namespace car_owners_without_bikes_l238_238764

noncomputable def adults_who_own_cars_and_bikes (total_adults car_owners bike_owners : ℕ) : ℕ :=
  car_owners + bike_owners - total_adults

theorem car_owners_without_bikes (car_owners bike_owners total_adults adults_who_own_both : ℕ)
  (h1 : total_adults = 400) (h2 : car_owners = 370) (h3 : bike_owners = 30) (h4 : adults_who_own_both = 0) :
  car_owners - adults_who_own_both = 370 :=
by {
  rw [h1, h2, h3, h4],
  simp,
  sorry
}

end car_owners_without_bikes_l238_238764


namespace line_distance_eq_l238_238073

-- Define the line equation and the condition for the distance from that line
def line_eq (x y : ℝ) := x - y - 1 = 0

def is_line_at_distance_2 (m : ℝ) : Prop :=
  |-(1 + m)| / Real.sqrt(2) = 2

theorem line_distance_eq (m : ℝ) :
  is_line_at_distance_2 m → m = 2 * Real.sqrt(2) - 1 ∨ m = -(2 * Real.sqrt(2) + 1) :=
by
  sorry

end line_distance_eq_l238_238073


namespace solution_set_inequality_l238_238005

def floor (x : ℝ) : ℤ := ⌊x⌋

theorem solution_set_inequality (x : ℝ) :
  ∃ (a b : ℝ), (a = -2) ∧ (b = 6) ∧ (∀ (y : ℝ), (floor y * floor y - 3 * floor y - 10 ≤ 0) ↔ (y ∈ set.Ico a b)) :=
by
  use [-2, 6]
  sorry

end solution_set_inequality_l238_238005


namespace container_liquids_l238_238585

theorem container_liquids :
  (∀ c : Container, c ≠ Cup → liquid_in c ≠ water ∧ liquid_in c ≠ milk) →
  (∀ c1 c2 c3 : Container, (liquid_in c2 = lemonade ∧ between c1 c2 c3) → (c1 = jug ∧ c3 = bank ∨ c1 = bank ∧ c3 = jug)) →
  (∀ c : Container, c = Jar → liquid_in c ≠ lemonade ∧ liquid_in c ≠ water) →
  (∀ c1 c2 c3 : Container, (c1 = glass ∧ c2 = jar ∧ c3 = cup ∨ c1 = glass ∧ c2 = cup ∧ c3 = jar) ∧ milk ∈ [liquid_in c1, liquid_in c2, liquid_in c3]) →
  liquid_in cup = lemonade ∧ liquid_in glass = water ∧ liquid_in jug = milk ∧ liquid_in bank = kvass :=
sorry

end container_liquids_l238_238585


namespace greatest_multiple_3_5_7_less_than_1000_l238_238926

theorem greatest_multiple_3_5_7_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ 
           ∀ m : ℕ, m < 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≤ n :=
begin
  use 945,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  intros m hm,
  cases hm with hm0 hm1,
  cases hm1 with hm2 hm3,
  cases hm3 with hm4 hm5,
  have hz : m = 105 * (m / 105) + m % 105 := Nat.div_add_mod m 105,
  have hw : m % 105 < 105 := Nat.mod_lt m (by norm_num),
  have hv : m % 105 % 3 = 0,
  { exact Nat.mod_mod _ 3 },
  have hx : m % 105 % 5 = 0,
  { exact Nat.mod_mod _ 5 },
  have hy : m % 105 % 7 = 0,
  { exact Nat.mod_mod _ 7 },
  have hz : m % 105 = 0 := by
  { apply Nat.eq_zero_of_dvd_of_lt,
    { apply and.intro,
      { intro,
        { apply Nat.mod_lt,
          { by norm_num } } }
      { exact Nat.mod_eq_zero_of_dvd,
        { apply and.intro,
          { intro,
            { apply Nat.mod_lt,
              { by norm_num } } },
          { apply Nat.mod_dvd_of_dvd,
            { exact Nat.mod_eq_of_lt (by norm_num) } } } },
      {},
    { exact Nat.mul_le_mul_left 105 (by norm_num) },
    { exact Nat.le_of_dvd (by norm_num) (Nat.dvd_of_mod_eq_zero hz) } },
  exact Nat.lt_of_le_of_lt (Nat.mul_le_mul_left 105 (Nat.div_le_of_le_mul hw)) (by norm_num),
end

end greatest_multiple_3_5_7_less_than_1000_l238_238926


namespace max_remaining_numbers_l238_238842

theorem max_remaining_numbers : 
  ∃ s : Finset ℕ, s ⊆ (Finset.range 236) ∧ (∀ x y ∈ s, x ≠ y → ¬ (x - y).abs ∣ x) ∧ s.card = 118 := 
by
  sorry

end max_remaining_numbers_l238_238842


namespace distribute_rabbits_l238_238502

theorem distribute_rabbits :
  let rabbits := ["Peter", "Pauline", "Flopsie", "Mopsie", "Cotton-tail", "Topsy"]
  let parent := {"Peter", "Pauline"}
  let child := {"Flopsie", "Mopsie", "Cotton-tail", "Topsy"}
  let stores := (fin 5)
  (number of ways to distribute the rabbits to stores under the condition
  that no store has both a parent and a child) = 380 :=
sorry

end distribute_rabbits_l238_238502


namespace maximum_numbers_no_divisible_difference_l238_238818

theorem maximum_numbers_no_divisible_difference :
  ∃ (S : Finset ℕ), S ⊆ Finset.range 236 ∧ 
  (∀ a b c ∈ S, a ≠ b → b ≠ c → c ≠ a → ¬ (a - b = 0) ∨ ¬ (c ∣ (a - b))) ∧ S.card ≤ 118 :=
by
  sorry

end maximum_numbers_no_divisible_difference_l238_238818


namespace cost_of_child_office_visit_l238_238554

theorem cost_of_child_office_visit 
  (num_adult_per_hour: ℕ) (num_child_per_hour: ℕ) 
  (cost_adult: ℕ) (hours_per_day: ℕ) (total_income: ℕ) 
  (adult_patients: ℕ) (child_patients: ℕ) 
  (A_income: ℕ) (equation:  ℕ → ℕ):
  num_adult_per_hour = 4 →
  num_child_per_hour = 3 →
  cost_adult = 50 →
  hours_per_day = 8 →
  total_income = 2200 →
  adult_patients = (num_adult_per_hour * hours_per_day) →
  child_patients = (num_child_per_hour * hours_per_day) →
  A_income = (adult_patients * cost_adult) →
  equation A_income child_patients = total_income →
  25 = 25 :=
by
  intros,
  sorry

end cost_of_child_office_visit_l238_238554


namespace find_f6_l238_238568

-- conditions
def f : ℤ → ℤ 
| 4 := 14
| n := f (n - 1) - n

-- statement
theorem find_f6 : f 6 = 3 := by
  sorry

end find_f6_l238_238568


namespace hyperbola_eccentricity_eq_2_l238_238522

-- Definitions and conditions
def hyperbola_eq (a b : ℝ) : Prop := ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1)

def is_focus (F : ℝ × ℝ) (a b : ℝ) : Prop := F = (-(sqrt (a^2 + b^2)), 0)

def is_equilateral_triangle (A B F : ℝ × ℝ) (a b : ℝ) : Prop :=
  let AB := dist A B in
  let AF := dist A F in
  let BF := dist B F in
  A ≠ B ∧ A ≠ F ∧ B ≠ F ∧ AB = AF ∧ AF = BF ∧ AB = 2 * b

-- Main theorem statement
theorem hyperbola_eccentricity_eq_2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (F : ℝ × ℝ)
  (h_focus : is_focus F a b) (A B : ℝ × ℝ)
  (h_triangle : is_equilateral_triangle A B F a b) : 
  let c := sqrt (a^2 + b^2) in
  let e := c / a in
  e = 2 :=
begin
  sorry
end

end hyperbola_eccentricity_eq_2_l238_238522


namespace abc_range_l238_238717

def f : ℝ → ℝ :=
λ x, if x > 0 then abs (Real.log x) else 2 * x + 6

theorem abc_range (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq : f a = f b ∧ f b = f c) : -3 < a ∧ a ≤ 0 ∧ 0 < b ∧ b < 1 ∧ 1 < c ∧ c ≤ 10^6 → 
  ∃ abc : ℝ, abc = a * b * c ∧ abc ∈ Set.Ioc (-3) 0 :=
by
  sorry

end abc_range_l238_238717


namespace max_rabbits_with_traits_l238_238124

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l238_238124


namespace train_platform_length_l238_238271

theorem train_platform_length:
  ∀ (train_length speed_kmph time_seconds : ℕ), 
  train_length = 750 → 
  speed_kmph = 180 → 
  time_seconds = 15 → 
  let speed_mps := (speed_kmph * 1000) / 3600 in
  let distance := speed_mps * time_seconds in
  distance = train_length → 
  platform_length = 0 :=
by
  intros train_length speed_kmph time_seconds h_train_length h_speed_kmph h_time_seconds
  let speed_mps := (speed_kmph * 1000) / 3600
  let distance := speed_mps * time_seconds
  suffices h_distance : distance = train_length
  sorry

end train_platform_length_l238_238271


namespace correctAnswerIsC_l238_238933

-- Define the condition that certain elements can form a set.
def canFormSet (x : Type) : Prop := ∃ s : set x, True

-- Define specific examples mentioned in the problem.
def verySmallPositiveNumbers : Type := { n : ℝ // 0 < n }
def famousMathematicians : Type := { x : ℕ // True } -- Just a placeholder, actual set is unclear.
def countriesIn2014IncheonAsianGames : Type := { c : String // True } -- Placeholder, could be a list of countries.
def approximateValuesOfSqrt3 : Type := { r : ℝ // True } -- Placeholder, approximate values are ill-defined.

-- Now we state the theorem with given conditions.
theorem correctAnswerIsC :
  ¬ canFormSet verySmallPositiveNumbers ∧
  ¬ canFormSet famousMathematicians ∧
  ¬ canFormSet approximateValuesOfSqrt3 ∧
  canFormSet countriesIn2014IncheonAsianGames :=
begin
  sorry, -- We use sorry to skip the proof.
end

end correctAnswerIsC_l238_238933


namespace problem_2013_factorial_l238_238134

theorem problem_2013_factorial (a_1 a_2 : ℕ) (b_1 b_2 : ℕ) (m n : ℕ) 
  (h1 : a_1! * a_2! = b_1! * b_2! * 2013)
  (h2 : a_1 ≥ a_2)
  (h3 : b_1 ≥ b_2)
  (h4 : ∀ x y, (x + y < a_1 + b_1) → ¬(x! * a_2! = y! * b_2! * 2013)) :
  |a_1 - b_1| = 2 :=
sorry

end problem_2013_factorial_l238_238134


namespace find_k_of_orthogonal_l238_238726

def a : ℝ × ℝ := (12, k) 
def b : ℝ × ℝ := (1 - k, 14)
def orthogonal (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0

theorem find_k_of_orthogonal (k : ℝ) (h : orthogonal a b) : k = -6 :=
  sorry

end find_k_of_orthogonal_l238_238726


namespace max_remaining_numbers_l238_238810

def numbers (n : ℕ) : set ℕ := {k | 1 ≤ k ∧ k ≤ n}

def valid_subset (s : set ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → (a - b) ≠ 0 → ¬ (a - b) ∣ c

theorem max_remaining_numbers : ∃ s : set ℕ, s ⊆ numbers 235 ∧ valid_subset s ∧ card s = 118 := 
sorry

end max_remaining_numbers_l238_238810


namespace projection_of_a_in_direction_of_b_l238_238392

open InnerProductSpace EuclideanSpace

variables (a b : EuclideanSpace ℝ) (θ : ℝ)

def dot_product_condition : Prop := (inner a (a + b)) = 5
def norm_a : Prop := ∥a∥ = 2
def norm_b : Prop := ∥b∥ = 1

theorem projection_of_a_in_direction_of_b :
  dot_product_condition a b →
  norm_a a →
  norm_b b →
  (inner a b) / (∥b∥ * ∥b∥) = 1 :=
by
  intros h1 h2 h3
  sorry

end projection_of_a_in_direction_of_b_l238_238392


namespace perfect_square_of_seq_l238_238140

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ ∀ n ≥ 3, a n = 7 * a (n - 1) - a (n - 2)

theorem perfect_square_of_seq (a : ℕ → ℤ) (h : seq a) (n : ℕ) (hn : 0 < n) :
  ∃ k : ℤ, k * k = a n + 2 + a (n + 1) :=
sorry

end perfect_square_of_seq_l238_238140


namespace digits_count_concatenated_l238_238896

-- Define the conditions for the digit count of 2^n and 5^n
def digits_count_2n (n p : ℕ) : Prop := 10^(p-1) ≤ 2^n ∧ 2^n < 10^p
def digits_count_5n (n q : ℕ) : Prop := 10^(q-1) ≤ 5^n ∧ 5^n < 10^q

-- The main theorem to prove the number of digits when 2^n and 5^n are concatenated
theorem digits_count_concatenated (n p q : ℕ) 
  (h1 : digits_count_2n n p) 
  (h2 : digits_count_5n n q): 
  p + q = n + 1 := by 
  sorry

end digits_count_concatenated_l238_238896


namespace constant_term_of_expansion_is_80_l238_238643

-- Define the problem parameters and expressions
def binomial_expansion_constant_term: Int :=
  let c3 := (-2)^3 * Nat.choose 6 3       -- Term for r = 3
  let c4 := (-2)^4 * Nat.choose 6 4       -- Term for r = 4
  c3 + c4

-- The theorem statement asserting the constant term
theorem constant_term_of_expansion_is_80 : binomial_expansion_constant_term = 80 := by 
  sorry

end constant_term_of_expansion_is_80_l238_238643


namespace median_salary_is_correct_l238_238624

-- Define a structure for position and salary data
structure EmployeePosition :=
  (title : String)
  (count : Nat)
  (salary : Int)

-- Given data
def employeePositions : List EmployeePosition :=
  [{title := "CEO", count := 1, salary := 150000},
   {title := "General Manager", count := 4, salary := 95000},
   {title := "Manager", count := 12, salary := 80000},
   {title := "Assistant Manager", count := 8, salary := 55000},
   {title := "Clerk", count := 40, salary := 25000}]

noncomputable def totalEmployees : Nat :=
  employeePositions.foldl (fun acc pos => acc + pos.count) 0

noncomputable def medianSalary (positions : List EmployeePosition) : Int :=
  let sortedEmployees := positions.flatMap (fun pos => List.replicate pos.count pos.salary)
  let sortedSalaries := sortedEmployees.sort (<=)
  sortedSalaries.get! (sortedSalaries.length / 2)

theorem median_salary_is_correct : medianSalary employeePositions = 25000 := by
  sorry

end median_salary_is_correct_l238_238624


namespace hyperbola_center_l238_238655

theorem hyperbola_center 
  (x y : ℝ)
  (h : 9 * x ^ 2 + 54 * x - 16 * y ^ 2 - 128 * y - 200 = 0) : 
  (x = -3) ∧ (y = -4) := 
sorry

end hyperbola_center_l238_238655


namespace fruit_count_l238_238025

theorem fruit_count :
  let limes_mike : ℝ := 32.5
  let limes_alyssa : ℝ := 8.25
  let limes_jenny_picked : ℝ := 10.8
  let limes_jenny_ate := limes_jenny_picked / 2
  let limes_jenny := limes_jenny_picked - limes_jenny_ate
  let plums_tom : ℝ := 14.5
  let plums_tom_ate : ℝ := 2.5
  let X := (limes_mike - limes_alyssa) + limes_jenny
  let Y := plums_tom - plums_tom_ate
  X = 29.65 ∧ Y = 12 :=
by {
  sorry
}

end fruit_count_l238_238025


namespace find_f_2017_l238_238714

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then real.log 2 (3 - x) else f (x - 1) - f (x - 2)

theorem find_f_2017 : f 2017 = real.log 2 3 - 2 :=
sorry

end find_f_2017_l238_238714


namespace simplify_fraction_l238_238323

theorem simplify_fraction (x y z : ℝ) (h : x + 2 * y + z ≠ 0) :
  (x^2 + y^2 - 4 * z^2 + 2 * x * y) / (x^2 + 4 * y^2 - z^2 + 2 * x * z) = (x + y - 2 * z) / (x + z - 2 * y) :=
by
  sorry

end simplify_fraction_l238_238323


namespace remainder_division_l238_238335

-- Define the polynomials involved
def dividend : polynomial ℚ := polynomial.X^4 + polynomial.X^2 + 1
def divisor : polynomial ℚ := polynomial.X^2 - 2 * polynomial.X + 4
def remainder : polynomial ℚ := -6 * polynomial.X - 3

-- State the theorem: remainder when dividend is divided by divisor
theorem remainder_division : polynomial.mod_by_monic dividend (polynomial.monic_of_monic_of_degree_eq_zero divisor sorry) = remainder := 
sorry

end remainder_division_l238_238335


namespace maximum_snowballs_constructible_l238_238031

def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem maximum_snowballs_constructible
  (patrick_radius : ℝ) (anderson_radius : ℝ)
  (patrick_snowball_volume anderson_snowball_volume : ℝ)
  (r_patrick : patrick_radius = 10)
  (r_anderson : anderson_radius = 4)
  (v_patrick : patrick_snowball_volume = volume_of_sphere patrick_radius)
  (v_anderson : anderson_snowball_volume = volume_of_sphere anderson_radius) :
  (patrick_snowball_volume / anderson_snowball_volume).floor = 15 := by
  sorry

end maximum_snowballs_constructible_l238_238031


namespace incenter_incircle_of_TRIangle_AST_is_inside_in_TRriangle_ABC_incircle_l238_238127

/-- Problem conditions:
1. The incircle of triangle ABC is tangent to sides BC, CA, and AB at points D, E, and F respectively.
2. The reflection of F with respect to B is T, and the reflection of E with respect to C is S.
To prove: The incenter of triangle AST is inside or on the incircle of triangle ABC.
-/
theorem incenter_incircle_of_TRIangle_AST_is_inside_in_TRriangle_ABC_incircle
  (A B C D E F T S : Type)
  [incircle_tangency : incircle_tangent_to_sides A B C D E F]
  [reflection_F_B : reflect F B T]
  [reflection_E_C : reflect E C S] :
  is_incenter_inside_incircle_of_AST
  (triangle_incenter A S T) (incircle A B C) :=
sorry

end incenter_incircle_of_TRIangle_AST_is_inside_in_TRriangle_ABC_incircle_l238_238127


namespace radek_result_l238_238237

theorem radek_result (a b : ℤ) : a = 6 ∧ b = -12 :=
by {
  
  have h : a = 2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10,
    calc 
      2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10
        = -1 + 4 - 5 + 6 - 7 + 8 - 9 + 10 : by ring
        = 3 - 5 + 6 - 7 + 8 - 9 + 10 : by ring
        = -2 + 6 - 7 + 8 - 9 + 10 : by ring
        = 4 - 7 + 8 - 9 + 10 : by ring
        = -3 + 8 - 9 + 10 : by ring
        = 5 - 9 + 10 : by ring
        = -4 + 10 : by ring
        = 6 : by ring,
  exact ⟨h, by {have : b = a - 18, rw h, exact dec_trivial}⟩ }

end radek_result_l238_238237


namespace kayla_scores_on_eighth_level_l238_238780

def fibonacci : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

def kayla_score : ℕ → ℤ
| 0 := 2
| 1 := 1
| n := if n % 2 = 0 then kayla_score (n-1) + fibonacci n
      else kayla_score (n-1) - fibonacci n

theorem kayla_scores_on_eighth_level : kayla_score 7 = -7 :=
by
  -- Proof goes here
  sorry

end kayla_scores_on_eighth_level_l238_238780


namespace containers_needed_l238_238489

def milk : ℚ := 15
def chicken_stock : ℚ := 3 * milk
def pureed_vegetables : ℚ := 5
def other_ingredients : ℚ := 4
def total_soup : ℚ := milk + chicken_stock + pureed_vegetables + other_ingredients
def container_capacity : ℚ := 2.5

theorem containers_needed : Nat := Nat.ceil (total_soup / container_capacity) = 28 := by
  sorry

end containers_needed_l238_238489


namespace problem_1_1_problem_1_2_problem_2_problem_3_l238_238504

-- Definitions for problems based on conditions
def pow_mul_rule (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by sorry

-- Problem 1.1
theorem problem_1_1 : (-1/2)^4 * (-1/2)^6 = (-1/2)^(10) :=
calc
  (-1/2)^4 * (-1/2)^6 = (-1/2)^(4 + 6)   : pow_mul_rule (-1/2) 4 6
...                  = (-1/2)^10         : by sorry

-- Problem 1.2
theorem problem_1_2 : 3^2 * (-3)^3 = -243 :=
calc
  3^2 * (-3)^3 = (-3)^2 * (-3)^3          : by sorry
...           = (-3)^(2 + 3)             : pow_mul_rule (-3) 2 3
...           = (-3)^5                   : by sorry
...           = -243                     : by sorry

-- Problem 2
theorem problem_2 : 2^3 + 2^3 + 2^3 + 2^3 = 2^5 :=
calc
  2^3 + 2^3 + 2^3 + 2^3 = 4 * 2^3        : by sorry
...                     = 2^2 * 2^3      : by sorry
...                     = 2^(2 + 3)      : pow_mul_rule 2 2 3
...                     = 2^5            : by sorry

-- Problem 3
theorem problem_3 (x y : ℝ) (p : ℕ) : (x-y)^2 * (x-y)^p * (x-y)^5 = (x-y)^2023 → p = 2016 :=
assume h1 : (x - y)^2 * (x - y)^p * (x - y)^5 = (x - y)^2023,
calc
  2 + p + 5 = 2023                      : by sorry
...       p = 2023 - 2 - 5              : by sorry
...       p = 2016                      : by sorry

end problem_1_1_problem_1_2_problem_2_problem_3_l238_238504


namespace fk_at_one_F_inequality_l238_238684

noncomputable def f₀ (n : ℕ) (x : ℝ) : ℝ := x^n

noncomputable def f (k n : ℕ) (x : ℝ) : ℝ :=
match k with
| 0     => f₀ n x
| (k+1) => (derivative (f k n)) x / (f k n 1)

noncomputable def F (n : ℕ) (x : ℝ) : ℝ :=
∑ k in finset.range (n + 1), nat.choose n k * (f k n (x^2))

theorem fk_at_one (n k : ℕ) (h₁ : k ≤ n) : f k n 1 = n - k + 1 := sorry

theorem F_inequality (n : ℕ) (x₁ x₂ : ℝ) (h₁ : -1 ≤ x₁) (h₂ : x₁ ≤ 1) (h₃ : -1 ≤ x₂) (h₄ : x₂ ≤ 1) :
  |F n x₁ - F n x₂| ≤ 2^(n-1)*(n+2) - n - 1 := sorry

end fk_at_one_F_inequality_l238_238684


namespace max_remaining_numbers_l238_238809

def numbers (n : ℕ) : set ℕ := {k | 1 ≤ k ∧ k ≤ n}

def valid_subset (s : set ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → (a - b) ≠ 0 → ¬ (a - b) ∣ c

theorem max_remaining_numbers : ∃ s : set ℕ, s ⊆ numbers 235 ∧ valid_subset s ∧ card s = 118 := 
sorry

end max_remaining_numbers_l238_238809


namespace minimize_quadratic_l238_238194

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_l238_238194


namespace minimize_quadratic_l238_238197

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end minimize_quadratic_l238_238197


namespace abs_diff_a1_b1_is_2_l238_238132

-- Given conditions and definitions
def a1 : ℕ := 61
def b1 : ℕ := 59
def a_vals : List ℕ := [61, 19, 11]
def b_vals : List ℕ := [59, 20, 10]

-- Given that the expression yields 2013 with the factorials
def exp := (List.product (a_vals.map Nat.factorial)) / (List.product (b_vals.map Nat.factorial))

-- Problem statement: Prove that |a1 - b1| = 2.
theorem abs_diff_a1_b1_is_2 : abs (a1 - b1) = 2 := by
  sorry

end abs_diff_a1_b1_is_2_l238_238132


namespace min_value_at_3_l238_238202

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end min_value_at_3_l238_238202


namespace solution_to_equation_l238_238330

-- Define the given conditions
noncomputable def given_equation (x : ℝ) : Prop := 
  real.cbrt (3 - x) + real.sqrt (x - 1) = 2

-- Prove that x = 2 satisfies the given conditions
theorem solution_to_equation : given_equation 2 :=
  sorry -- proof omitted

end solution_to_equation_l238_238330


namespace most_significant_k_l238_238139

theorem most_significant_k (k : ℝ) (h : sqrt (k^2 - 32) = sqrt 77) : 
  k = sqrt 109 ∨ k = -sqrt 109 :=
by
  sorry

end most_significant_k_l238_238139


namespace simplify_expression_correct_l238_238051

def simplify_expression : ℚ :=
  (5^5 + 5^3) / (5^4 - 5^2)

theorem simplify_expression_correct : simplify_expression = 65 / 12 :=
  sorry

end simplify_expression_correct_l238_238051


namespace intersect_skew_lines_infinite_l238_238759

theorem intersect_skew_lines_infinite (a b c : Line) (h_skew : ∀(u v : Line), (u ≠ v) → (u ≠ v) → (u ∩ v = ∅)) :
  ∃ (l : Line), (∀ (p : Point), p ∈ l → (p ∈ a ∧ p ∈ b ∧ p ∈ c)) ∧ ∃ (n : ℕ), n = ∞ :=
sorry

end intersect_skew_lines_infinite_l238_238759


namespace minimize_quadratic_l238_238195

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end minimize_quadratic_l238_238195


namespace minimize_y_l238_238791

variable (a b x : ℝ)

def y := (x - a)^2 + (x - b)^2

theorem minimize_y : ∃ x : ℝ, (∀ (x' : ℝ), y x a b ≤ y x' a b) ∧ x = (a + b) / 2 := by
  sorry

end minimize_y_l238_238791


namespace common_chord_length_l238_238594

noncomputable def radius_ratio := 4 / 3
noncomputable def segment_OQ := 7
noncomputable def segment_ON := 5
noncomputable def segment_QN := 2

theorem common_chord_length :
∃ (R : ℝ), (4 * R)^2 - 2^2 = (3 * R)^2 - 5^2 ∧
  2 * real.sqrt (3^2 * R^2 - 5^2) = 2 * real.sqrt 23 := by
  sorry

end common_chord_length_l238_238594


namespace workCompletionTime_l238_238944

noncomputable def workRateA (W : ℝ) : ℝ := W / 30
noncomputable def workRateB (W : ℝ) : ℝ := W / 30
noncomputable def workRateC (W : ℝ) : ℝ := W / 40

theorem workCompletionTime (W : ℝ) : ∃ x : ℝ, 
  (workRateA W + workRateB W + workRateC W) * (x - 4) + 
  (workRateA W + workRateB W) * 4 = W ∧ x ≈ 15 :=
sorry

end workCompletionTime_l238_238944


namespace value_range_f_l238_238533

noncomputable def f (x : ℝ) : ℝ := Real.logb 3 (x^2 - 2*x + 10)

theorem value_range_f :
  (∀ x : ℝ, x^2 - 2*x + 10 ≥ 9) ->
  (∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 2) :=
by
  sorry

end value_range_f_l238_238533


namespace mrs_coppers_class_l238_238909

theorem mrs_coppers_class (J_initial J_left n_absent : ℕ) (hJ_initial : J_initial = 100)
    (hJ_left : J_left = 34) (hneaten_per_child : ∀ (x : ℕ), 3 * x - (J_initial - J_left) = 0) 
    (h_absent : n_absent = 2) : ∃ (y : ℕ), y = 24 :=
by
  have h_eaten_total : J_initial - J_left = 66 := by simp [hJ_initial, hJ_left]
  obtain ⟨x, hx⟩ : ∃ x, 3 * x = 66 := by
    { use 22, rw mul_comm, exact eq.trans (mul_comm 3 22.symm) h_eaten_total }
  have h_total_children : x + n_absent = 24 := by
    simp [hx, h_absent]
  exact ⟨24, h_total_children⟩

end mrs_coppers_class_l238_238909


namespace max_rabbits_with_long_ears_and_jumping_far_l238_238094

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l238_238094


namespace orangeade_ratio_l238_238028

theorem orangeade_ratio (O W : ℝ) (price1 price2 : ℝ) (revenue1 revenue2 : ℝ)
  (h1 : price1 = 0.30) (h2 : price2 = 0.20)
  (h3 : revenue1 = revenue2)
  (glasses1 glasses2 : ℝ)
  (V : ℝ) :
  glasses1 = (O + W) / V → glasses2 = (O + 2 * W) / V →
  revenue1 = glasses1 * price1 → revenue2 = glasses2 * price2 →
  (O + W) * price1 = (O + 2 * W) * price2 → O / W = 1 :=
by sorry

end orangeade_ratio_l238_238028


namespace smallest_pencils_l238_238214

theorem smallest_pencils (P : ℕ) :
  (P > 2) ∧
  (P % 5 = 2) ∧
  (P % 9 = 2) ∧
  (P % 11 = 2) →
  P = 497 := by
  sorry

end smallest_pencils_l238_238214


namespace add_and_simplify_fractions_l238_238866

theorem add_and_simplify_fractions :
  (1 / 462) + (23 / 42) = 127 / 231 :=
by
  sorry

end add_and_simplify_fractions_l238_238866


namespace billy_sleep_total_l238_238745

theorem billy_sleep_total :
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  day1 + day2 + day3 + day4 = 30 :=
by
  -- Definitions
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  -- Assertion
  have h : day1 + day2 + day3 + day4 = 30 := sorry
  exact h

end billy_sleep_total_l238_238745


namespace one_root_eq_condition_l238_238340

-- We need to define the conditions and prove the equivalence
theorem one_root_eq_condition (p : ℝ) :
  (∀ (x : ℝ), (x + 1 = Real.sqrt (p * x)) → x ∈ Set.Icc (-1) 0 ∪ {1}) ↔ (p ∈ Set.Icc (-∞) (0 : ℝ) ∪ {4}) :=
by
  sorry

end one_root_eq_condition_l238_238340


namespace tripletE_sum_not_eq_2_l238_238937

def tripletA := (3/4, 1/2, 3/4)
def tripletB := (1.2, -0.4, 1.2)
def tripletC := (3/5, 7/10, 7/10)
def tripletD := (3.3, -1.6, 0.3)
def tripletE := (6/5, 1/5, 2/5)

noncomputable def sum_triplet (t : Rat × Rat × Rat) : Rat :=
  t.1 + t.2.1 + t.2.2

theorem tripletE_sum_not_eq_2 : sum_triplet tripletE ≠ 2 := by
  sorry

end tripletE_sum_not_eq_2_l238_238937


namespace sum_of_possible_students_l238_238974

theorem sum_of_possible_students :
  (∑ s in Finset.filter (λ s, s ∈ (Icc 160 210) ∧ (s - 1) % 8 = 0) (Finset.range 211)) = 1295 :=
by
  sorry

end sum_of_possible_students_l238_238974


namespace solution_set_of_f_inequality_l238_238983

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotonic_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop := ∀ x y, x ∈ I → y ∈ I → x < y → f y ≤ f x

theorem solution_set_of_f_inequality
  (h_even : is_even f)
  (h_monotonic : is_monotonic_decreasing f Set.Ioi(0))
  (h_f1 : f 1 = 0) :
  {x : ℝ | f x > 0} = Set.Ioo (-1) 1 \ {0} := 
sorry

end solution_set_of_f_inequality_l238_238983


namespace maximum_value_N_27_l238_238097

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end maximum_value_N_27_l238_238097


namespace combination_problem_binomial_coefficient_l238_238243

-- Problem (1)
theorem combination_problem (n : ℕ) (h : (n - 5) * (n - 6) - 1 = 89) : nat.choose n 2 = 105 :=
sorry

-- Problem (2)
theorem binomial_coefficient (x : ℝ) :
  (x - real.sqrt 3) ^ 10 = ∑ k in finset.range 11, (nat.choose 10 k) * x^(10 - k) * (-real.sqrt 3)^k
  ∧ (((nat.choose 10 4) : ℝ) * ((real.sqrt 3) ^ 4) = 1890) :=
sorry

end combination_problem_binomial_coefficient_l238_238243


namespace largest_number_in_set_l238_238892

theorem largest_number_in_set :
  let S := {0.01, 0.2, 0.03, 0.02, 0.1}
  in max' S (by decide) = 0.2 :=
by
  let S := {0.01, 0.2, 0.03, 0.02, 0.1}
  have h_nonempty : S.nonempty := by decide
  show max' S h_nonempty = 0.2
  sorry

end largest_number_in_set_l238_238892


namespace cost_graph_representation_l238_238396

-- Define the cost function C
def C (n : ℕ) : ℕ :=
  20 * n

-- Define the condition on n and the nature of the graph
theorem cost_graph_representation : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 15 →
  ∃ S : set (ℕ × ℕ), S = { (n, C n) | (1 ≤ n ∧ n ≤ 15) } ∧
  ∀ (a b : ℕ), (a, C a) ∈ S → (b, C b) ∈ S → a = b → C a = C b :=
sorry  -- Proof is not provided

end cost_graph_representation_l238_238396


namespace find_a_b_l238_238339

theorem find_a_b (a b : ℝ) (h1 : b - a = -7) (h2 : 64 * (a + b) = 20736) :
  a = 165.5 ∧ b = 158.5 :=
by
  sorry

end find_a_b_l238_238339


namespace room_length_l238_238128

theorem room_length (width : ℝ) (total_cost : ℝ) (cost_per_sq_meter : ℝ) (length : ℝ) : 
  width = 3.75 ∧ total_cost = 14437.5 ∧ cost_per_sq_meter = 700 → length = 5.5 :=
by
  sorry

end room_length_l238_238128


namespace simplify_fraction_l238_238041

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l238_238041


namespace athlete_possible_scores_l238_238280

theorem athlete_possible_scores (n_attempts : ℕ) (n_2pointers : ℕ) (n_3pointers : ℕ) :
  n_attempts = 8 → 
  (∀ n_2pointers n_3pointers, n_2pointers + n_3pointers = 8 →
   (16 ≤ n_3pointers * 3 + n_2pointers * 2 ∧ n_3pointers * 3 + n_2pointers * 2 ≤ 24)) →
  ∃ unique_totals : finset ℕ, (unique_totals = finset.range' 16 9) :=
by 
  sorry

end athlete_possible_scores_l238_238280


namespace cistern_emptying_time_l238_238255

theorem cistern_emptying_time (R L : ℝ) (h1 : R * 8 = 1) (h2 : (R - L) * 10 = 1) : 1 / L = 40 :=
by
  -- proof omitted
  sorry

end cistern_emptying_time_l238_238255


namespace simplify_fraction_l238_238044

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l238_238044


namespace compare_BL_BG_l238_238238

noncomputable section

variables {A B C : Type*} [euclidean_space A] [euclidean_space B] [euclidean_space C]

-- Given conditions
structure Triangle (α : Type*) [euclidean_space α] :=
  (A B C : α)
  (angle_A : ∡ A B C = 90°)
  (AB : dist A B = 1)
  (BC : dist B C = 2)

-- Definition of angle bisector BL and centroid G
variable {T : Triangle ℝ}

theorem compare_BL_BG (hT : Triangle ℝ) :
  let B' := hT.B,
      C' := hT.C,
      BL := bisector_length_measure hT.A hT.B hT.C,
      G := centroid hT.A hT.B hT.C
  in dist B' G > BL := sorry

end compare_BL_BG_l238_238238


namespace exists_n_such_that_perimeter_decreases_or_right_triangle_l238_238463

-- Assuming a basic definition of a non-degenerate triangle in a geometric context
structure Triangle :=
  (A B C : Point)
  (non_degenerate : A ≠ B ∧ B ≠ C ∧ C ≠ A)

-- Function declaration for the orthocenter transformation
noncomputable def orthic_triangle_transformation (T : Triangle) : Triangle := sorry

-- Function declaration to extract the perimeter of a triangle
noncomputable def perimeter (T : Triangle) : ℝ := sorry

theorem exists_n_such_that_perimeter_decreases_or_right_triangle (ABC : Triangle)
  (H_A : Point) (H_B : Point) (H_C : Point) (h : Triangle → Triangle) :
  ∃ n : ℕ, let T_n := (λ n, nat.iterate h n ABC) in
    (T_n n).is_right_triangle ∨ (perimeter (T_n n) < perimeter ABC) := 
sorry

end exists_n_such_that_perimeter_decreases_or_right_triangle_l238_238463


namespace tan_add_pi_over_4_sin_over_expression_l238_238699

variable (α : ℝ)

theorem tan_add_pi_over_4 (h : Real.tan α = 2) : 
  Real.tan (α + π / 4) = -3 := 
  sorry

theorem sin_over_expression (h : Real.tan α = 2) : 
  (Real.sin (2 * α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1) = 1 := 
  sorry

end tan_add_pi_over_4_sin_over_expression_l238_238699


namespace simplify_fraction_l238_238046

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 :=
by
  sorry

end simplify_fraction_l238_238046


namespace perpendicular_NV_KL_l238_238784

open EuclideanGeometry

theorem perpendicular_NV_KL
  (A B K L T N U V : Point)
  (hK : OnCircle K (semicircle A B))
  (hL : OnCircle L (semicircle A B))
  (hT : intersection (line_through A K) (line_through A L) T)
  (hN : LiesOnSegment N A B ∧ perpendicular (line_through T N) (line_through A B))
  (hU : intersection (perpendicular_bisector A B) (line_through K L) U)
  (hV : LiesOnSegment V K L ∧ ∠ (line_through U A) (line_through U V) = ∠ (line_through U B) (line_through U V)) :
  perpendicular (line_through N V) (line_through K L) := sorry

end perpendicular_NV_KL_l238_238784


namespace part1_part2_l238_238703

theorem part1 (a : ℝ) (h : a ≠ 0) (h_sum_coeff : (∑ k in finset.range n.succ, (nat.choose n k) * (a^2)^k) = 16) : n = 4 :=
sorry

theorem part2 (a : ℝ) (h : a ≠ 0) (h_max_coeff : (nat.choose 4 2) * a^4 = 54) : a = sqrt 3 ∨ a = -sqrt 3 :=
sorry

end part1_part2_l238_238703


namespace white_washing_cost_l238_238515

theorem white_washing_cost
    (length width height : ℝ)
    (door_width door_height window_width window_height : ℝ)
    (num_doors num_windows : ℝ)
    (paint_cost : ℝ)
    (extra_paint_fraction : ℝ)
    (perimeter := 2 * (length + width))
    (door_area := num_doors * (door_width * door_height))
    (window_area := num_windows * (window_width * window_height))
    (wall_area := perimeter * height)
    (paint_area := wall_area - door_area - window_area)
    (total_area := paint_area * (1 + extra_paint_fraction))
    : total_area * paint_cost = 6652.8 :=
by sorry

end white_washing_cost_l238_238515


namespace right_triangle_properties_l238_238766

theorem right_triangle_properties (a b c h : ℝ)
  (ha: a = 5) (hb: b = 12) (h_right_angle: a^2 + b^2 = c^2)
  (h_area: (1/2) * a * b = (1/2) * c * h) :
  c = 13 ∧ h = 60 / 13 :=
by
  sorry

end right_triangle_properties_l238_238766


namespace train_passes_bridge_in_expected_time_l238_238228

def train_length : ℕ := 360
def speed_kmph : ℕ := 45
def bridge_length : ℕ := 140

def speed_mps : ℚ := (speed_kmph * 1000) / 3600
def total_distance : ℕ := train_length + bridge_length
def time_to_pass : ℚ := total_distance / speed_mps

theorem train_passes_bridge_in_expected_time : time_to_pass = 40 := by
  sorry

end train_passes_bridge_in_expected_time_l238_238228


namespace sqrt_123400_l238_238671

theorem sqrt_123400 (h1: Real.sqrt 12.34 = 3.512) : Real.sqrt 123400 = 351.2 :=
by 
  sorry

end sqrt_123400_l238_238671


namespace num_distinct_necklace_arrangements_l238_238768

theorem num_distinct_necklace_arrangements (n : ℕ) (hn : n = 8) :
  (nat.factorial n) / (n * 2) = 2520 :=
by
  rw hn
  sorry

end num_distinct_necklace_arrangements_l238_238768


namespace max_C1_ranking_l238_238605

theorem max_C1_ranking (n m : ℕ) (ranked_by_judges : (fin n) → (fin m) → ℕ) : 
  (∀ i : fin n, ∀ j k : fin m, |ranked_by_judges i j - ranked_by_judges i k| ≤ 3) →
  (∃ C : (fin n) → ℕ, 
    ∀ i : fin n, C i = ∑ j : fin m, ranked_by_judges i j ∧ 
    (∀ k : fin n,  C k = 24)) := sorry

end max_C1_ranking_l238_238605


namespace limit_of_f_is_zero_l238_238326

noncomputable def f (x y : ℝ) : ℝ := x^2 * y / (x^2 + y^2)

theorem limit_of_f_is_zero : 
  tendsto (λ p : ℝ × ℝ, f p.1 p.2) (𝓝 (0, 0)) (𝓝 0) :=
sorry

end limit_of_f_is_zero_l238_238326


namespace lambda1_plus_lambda2_half_l238_238012

variables {A B C D E : Type} [add_comm_group A] [vector_space ℝ A]
variables (AB AC BC DB BE DE : A)
variables (λ1 λ2 : ℝ)

def AD_eq_half_AB (AB AD : A) : Prop := AD = (1/2) • AB
def BE_eq_two_thirds_BC (BC BE : A) : Prop := BE = (2/3) • BC
def DE_eq_linear_combination (DE AB AC : A) (λ1 λ2 : ℝ) : Prop := DE = λ1 • AB + λ2 • AC

theorem lambda1_plus_lambda2_half
  (h1 : AD_eq_half_AB AB AD)
  (h2 : BE_eq_two_thirds_BC BC BE)
  (h3 : DE_eq_linear_combination DE AB AC λ1 λ2) :
  λ1 + λ2 = 1/2 :=
  sorry

end lambda1_plus_lambda2_half_l238_238012


namespace max_rabbits_l238_238082

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l238_238082


namespace cube_volume_surface_area_l238_238176

theorem cube_volume_surface_area (x : ℝ) (s : ℝ)
  (h1 : s^3 = 3 * x)
  (h2 : 6 * s^2 = 6 * x) :
  x = 3 :=
by sorry

end cube_volume_surface_area_l238_238176


namespace option_a_is_linear_l238_238561

def is_linear_function (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), m ≠ 0 ∧ (∀ x, f x = m * x + b)

theorem option_a_is_linear : is_linear_function (λ x, -8 * x) :=
sorry

end option_a_is_linear_l238_238561


namespace train_speed_l238_238567

-- Conditions
def train_length := 240  -- Length of the train in meters (found from conditions and equations)
def stationary_train_length := 300 -- Length of the stationary train in meters
def time_to_pass_pole := 12 -- Time to pass the pole in seconds
def time_to_pass_stationary_train := 27 -- Time to pass the stationary train in seconds

-- Prove that the speed is 20 meters per second
theorem train_speed :
  let L := 240 in  -- Given solution, length of the train
  let V1 := L / time_to_pass_pole in
  let V2 := (L + stationary_train_length) / time_to_pass_stationary_train in
  V1 = V2 → V1 = 20 := 
by
  intros,
  have h : time_to_pass_pole = 12 := rfl, -- Given condition
  have h' : time_to_pass_stationary_train = 27 := rfl, -- Given condition
  calc
  V1 = train_length / time_to_pass_pole : rfl
  ... = 240 / 12 : rfl
  ... = 20 : rfl
  sorry -- Step to conclude V1 = V2 and hence V1 = 20

end train_speed_l238_238567


namespace find_original_number_l238_238413

theorem find_original_number (x : ℕ) (h1 : 10 * x + 9 + 2 * x = 633) : x = 52 :=
by
  sorry

end find_original_number_l238_238413


namespace min_pie_pieces_l238_238886

theorem min_pie_pieces (p : ℕ) : 
  (∀ (k : ℕ), (k = 5 ∨ k = 7) → ∃ (m : ℕ), p = k * m ∨ p = m * k) → p = 11 := 
sorry

end min_pie_pieces_l238_238886


namespace max_handshakes_l238_238954

theorem max_handshakes (n : ℕ) (h : n = 25) : ∃ k : ℕ, k = 300 :=
by
  use (n * (n - 1)) / 2
  have h_eq : n = 25 := h
  rw h_eq
  sorry

end max_handshakes_l238_238954


namespace altitude_inequality_l238_238461

-- Definitions of altitudes and inradius
variables {ℝ} (Δ : Type*) -- The type of our triangle
variables (a b c h_a h_b h_c r s : ℝ) [noncomputable]

-- Given triangle with altitudes and inradius
variables (ABC : Δ) 

-- Conditions
hypothesis h_a_def : h_a = 2 * r * s / a
hypothesis h_b_def : h_b = 2 * r * s / b
hypothesis h_c_def : h_c = 2 * r * s / c

-- Required inequality
theorem altitude_inequality 
  (h_a_def : h_a = 2 * r * s / a)
  (h_b_def : h_b = 2 * r * s / b)
  (h_c_def : h_c = 2 * r * s / c)
  : h_a + 4 * h_b + 9 * h_c > 36 * r :=
sorry

end altitude_inequality_l238_238461


namespace parabola_focus_l238_238529

theorem parabola_focus (h : ∀ x y : ℝ, y ^ 2 = -12 * x → True) : (-3, 0) = (-3, 0) :=
  sorry

end parabola_focus_l238_238529


namespace det_set_size_l238_238466

open Matrix

-- Define the problem conditions
def M_2021 : set (Matrix (Fin 2021) (Fin 2021) ℤ) :=
  {A | ∀ i : Fin 2021, (∑ j, if A i j = 1 then 1 else 0) ≤ 2 ∧ 
                       ∀ i j, A i j = 0 ∨ A i j = 1}

-- Define the target set of determinants
def det_set : set ℤ := {det A | A ∈ M_2021}

-- The target theorem to prove the size of det_set
theorem det_set_size : det_set.to_finset.card = 1349 := sorry

end det_set_size_l238_238466


namespace problem_non_trivial_solutions_l238_238311

theorem problem_non_trivial_solutions (a b c : ℝ) :
  (∀ a b c : ℝ, sqrt (a^2 + b^2 + c^2) = 0 → a = 0 ∧ b = 0 ∧ c = 0) ∧ 
  (∃ a b c : ℝ, sqrt (a^2 + b^2 + c^2) = c ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)) ∧ 
  (∀ a b c : ℝ, sqrt (a^2 + b^2 + c^2) = a + b + c → a = 0 ∧ b = 0 ∧ c = 0) ∧ 
  (∀ a b c : ℝ, sqrt (a^2 + b^2 + c^2) = abc → a = 0 ∧ b = 0 ∧ c = 0) → 
  ∀ a b c : ℝ, sqrt (a^2 + b^2 + c^2) = c ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) := sorry

end problem_non_trivial_solutions_l238_238311


namespace polynomial_property_proof_l238_238482

noncomputable def polynomial_real_roots_property (n : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ x : ℝ, x ≥ (vector.maximum (vector.of_fn b)) → 
    let f := (λ x, x ^ n + ∑ i in finset.range n, a i * x ^ (n - 1 - i)) in
    f (x + 1) ≥ 2 * n^2 / (∑ i in finset.range n, 1 / (x - b i))
  
theorem polynomial_property_proof (n : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ) (h : n ≥ 2) (h2 : ∀ i < n, a i = real.coeff (polynomial.from_roots n b)) : 
  polynomial_real_roots_property n a b :=
sorry

end polynomial_property_proof_l238_238482


namespace maximum_rabbits_l238_238084

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l238_238084


namespace add_and_simplify_fractions_l238_238865

theorem add_and_simplify_fractions :
  (1 / 462) + (23 / 42) = 127 / 231 :=
by
  sorry

end add_and_simplify_fractions_l238_238865


namespace sum_consecutive_odds_seventh_power_l238_238294

theorem sum_consecutive_odds_seventh_power :
  ∃ (n : ℕ), (∑ i in (finset.range 1000).map (λ i, 2 * n - 999 + (2 * i : ℕ)), (2 * n - 999 + (2 * i : ℕ))) = 10^7 :=
by sorry

end sum_consecutive_odds_seventh_power_l238_238294


namespace find_value_of_a_l238_238715

section
variable (a : ℝ)
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x + 2 else -x^2

theorem find_value_of_a (h : f (f a) = 2) : a = real.sqrt 2 :=
by
  sorry
end

end find_value_of_a_l238_238715


namespace meeting_time_l238_238493

-- Define the conditions as given in the problem
def track_length : ℕ := 400
def speed_mona : ℕ := 18
def speed_sona : ℕ := 36
def speed_ravi : ℕ := 24
def speed_nina : ℕ := 48

-- Conversion of speed from km/h to m/min
def speed_mona_m_min : ℚ := (speed_mona * 1000 : ℚ) / 60
def speed_sona_m_min : ℚ := (speed_sona * 1000 : ℚ) / 60
def speed_ravi_m_min : ℚ := (speed_ravi * 1000 : ℚ) / 60
def speed_nina_m_min : ℚ := (speed_nina * 1000 : ℚ) / 60

-- Time to complete one lap in minutes
def time_mona_min : ℚ := track_length / speed_mona_m_min
def time_sona_min : ℚ := track_length / speed_sona_m_min
def time_ravi_min : ℚ := track_length / speed_ravi_m_min
def time_nina_min : ℚ := track_length / speed_nina_m_min

-- Time to complete one lap in seconds
def time_mona_sec : ℚ := time_mona_min * 60
def time_sona_sec : ℚ := time_sona_min * 60
def time_ravi_sec : ℚ := time_ravi_min * 60
def time_nina_sec : ℚ := time_nina_min * 60

-- Convert these times to natural numbers (if possible)
def time_mona_nat : ℕ := (time_mona_sec).ceil.to_nat
def time_sona_nat : ℕ := (time_sona_sec).ceil.to_nat
def time_ravi_nat : ℕ := (time_ravi_sec).ceil.to_nat
def time_nina_nat : ℕ := (time_nina_sec).ceil.to_nat

-- Question proving statement
theorem meeting_time : nat.lcm (nat.lcm time_mona_nat time_sona_nat) (nat.lcm time_ravi_nat time_nina_nat) / 60 = 4 := 
sorry

end meeting_time_l238_238493


namespace sphere_surface_area_of_cube_l238_238147

-- Problem statement
theorem sphere_surface_area_of_cube (a : ℝ) (h : a = 2) : 
  let R := (a * Real.sqrt 3) / 2 in
  let S := 4 * Real.pi * R^2 in
  S = 12 * Real.pi :=
by
  have := h
  subst this
  let R := (2 * Real.sqrt 3) / 2
  let S := 4 * Real.pi * R^2
  show S = 12 * Real.pi
  sorry

end sphere_surface_area_of_cube_l238_238147


namespace carl_marbles_l238_238995

-- Define initial conditions
def initial_marbles : ℕ := 12
def lost_marbles : ℕ := initial_marbles / 2
def remaining_marbles : ℕ := initial_marbles - lost_marbles
def additional_marbles : ℕ := 10
def new_marbles_from_mother : ℕ := 25

-- Define the final number of marbles Carl will put back in the jar
def total_marbles_put_back : ℕ := remaining_marbles + additional_marbles + new_marbles_from_mother

-- Statement to be proven
theorem carl_marbles : total_marbles_put_back = 41 :=
by
  sorry

end carl_marbles_l238_238995


namespace maximum_rabbits_condition_l238_238105

-- Define the conditions and constraints
variables {N : ℕ}
variables (total_rabbits long_ears jump_far : ℕ)
variables (at_least_three_with_both : Prop)

-- State the conditions with exact values and assumptions
def conditions := 
  total_rabbits = N ∧
  long_ears = 13 ∧
  jump_far = 17 ∧
  at_least_three_with_both = (∃ a b c : ℕ, a >= 3 ∧ b = (long_ears - a) ∧ c = (jump_far - a))

-- State the theorem to be proved
theorem maximum_rabbits_condition :
  ∀ {N : ℕ}, conditions N long_ears jump_far at_least_three_with_both → N ≤ 27 :=
by sorry

end maximum_rabbits_condition_l238_238105


namespace simplify_trig_identity_l238_238040

theorem simplify_trig_identity (x : ℝ) :
  tan x + 3 * tan (3 * x) + 9 * tan (9 * x) + 27 * cot (27 * x) = cot x :=
by sorry

end simplify_trig_identity_l238_238040


namespace store_incur_loss_of_one_percent_l238_238595

theorem store_incur_loss_of_one_percent
    (a b x : ℝ)
    (h1 : x = a * 1.1)
    (h2 : x = b * 0.9)
    : (2 * x - (a + b)) / (a + b) = -0.01 :=
by
  -- Proof goes here
  sorry

end store_incur_loss_of_one_percent_l238_238595


namespace equation_of_perpendicular_line_l238_238367

theorem equation_of_perpendicular_line
  (m : ℝ)
  (l1 : affine_affine_subspace ℝ ℝ 2 := {
    carrier := {x | x + (1 + m) * y + m - 2 = 0},
    direction := (0, 0)} )
  (l2 : affine_affine_subspace ℝ ℝ 2 := {
    carrier := {x | m * x + 2 * y + 8 = 0},
    direction := (0, 0)} )
  (A : affine_affine_subspace ℝ ℝ 2 := {
    carrier := {3, 2},
    direction := (0, 0)} )
  (h : l1 ∥ l2)
  : ∃ (l : affine_affine_subspace ℝ ℝ 2), 
    (l.carrier = {x | 2 * x - y - 4 = 0 }) :=
sorry

end equation_of_perpendicular_line_l238_238367


namespace eccentricity_of_ellipse_eq_half_l238_238375

variables {a b c d1 d2 : ℝ}
noncomputable def eccentricity_ellipse (a b c : ℝ) (h1 : a > b)
  (h2 : b > 0) (h3 : 2 * c = a) : ℝ := c / a

theorem eccentricity_of_ellipse_eq_half :
  ∀ (a b c d1 d2 : ℝ),
  (a > b) → (b > 0) → (d1 + d2 = 2 * a) →
  (d1 + d2 = 4 * c) → 2 * c = a →
  eccentricity_ellipse a b c (a > b) (b > 0) (2 * c = a) = 1 / 2 :=
  by
    intros
    rw [eccentricity_ellipse]
    sorry

end eccentricity_of_ellipse_eq_half_l238_238375


namespace remaining_calories_proof_l238_238038

def volume_of_rectangular_block (length width height : ℝ) : ℝ :=
  length * width * height

def volume_of_cube (side : ℝ) : ℝ :=
  side * side * side

def remaining_volume (initial_volume eaten_volume : ℝ) : ℝ :=
  initial_volume - eaten_volume

def remaining_calories (remaining_volume calorie_density : ℝ) : ℝ :=
  remaining_volume * calorie_density

theorem remaining_calories_proof :
  let calorie_density := 110
  let original_length := 4
  let original_width := 8
  let original_height := 2
  let cube_side := 2
  let original_volume := volume_of_rectangular_block original_length original_width original_height
  let eaten_volume := volume_of_cube cube_side
  let remaining_vol := remaining_volume original_volume eaten_volume
  let resulting_calories := remaining_calories remaining_vol calorie_density
  resulting_calories = 6160 := by
  repeat { sorry }

end remaining_calories_proof_l238_238038


namespace smallest_n_for_divisibility_l238_238175

theorem smallest_n_for_divisibility : ∃ n: ℕ, (n > 0) ∧ (n^2 % 24 = 0) ∧ (n^3 % 864 = 0) ∧ ∀ m : ℕ, 
  (m > 0) ∧ (m^2 % 24 = 0) ∧ (m^3 % 864 = 0) → (12 ≤ m) :=
begin
  sorry
end

end smallest_n_for_divisibility_l238_238175


namespace sarah_score_l238_238860

theorem sarah_score (j g s : ℕ) 
  (h1 : g = 2 * j) 
  (h2 : s = g + 50) 
  (h3 : (s + g + j) / 3 = 110) : 
  s = 162 := 
by 
  sorry

end sarah_score_l238_238860


namespace abs_neg_2023_l238_238575

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l238_238575


namespace average_salary_company_l238_238226

-- Define the conditions
def num_managers : Nat := 15
def num_associates : Nat := 75
def avg_salary_managers : ℤ := 90000
def avg_salary_associates : ℤ := 30000

-- Define the goal to prove
theorem average_salary_company : 
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) / (num_managers + num_associates) = 40000 := by
  sorry

end average_salary_company_l238_238226


namespace simplify_fraction_l238_238048

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 :=
by
  sorry

end simplify_fraction_l238_238048


namespace initial_group_size_l238_238066

theorem initial_group_size
  (n : ℕ) (W : ℝ)
  (h_avg_increase : ∀ W n, ((W + 12) / n) = (W / n + 3))
  (h_new_person_weight : 82 = 70 + 12) : n = 4 :=
by
  sorry

end initial_group_size_l238_238066


namespace trigonometric_identity_solution_l238_238225

theorem trigonometric_identity_solution (x : ℝ) (k : ℤ) :
  (5.45 * (Real.tan x) ^ 4 + (Real.cot x) ^ 4 = (82 / 9) * ((Real.tan x) * (Real.tan (2 * x)) + 1) * Real.cos (2 * x)) →
  (x = (π / 6) * (3 * k + 1) ∨ x = (π / 6) * (3 * k - 1)) :=
by
  -- Proof omitted
  sorry

end trigonometric_identity_solution_l238_238225


namespace add_fractions_l238_238651

-- Define the two fractions
def frac1 := 7 / 8
def frac2 := 9 / 12

-- The problem: addition of the two fractions and expressing in simplest form
theorem add_fractions : frac1 + frac2 = (13 : ℚ) / 8 := 
by 
  sorry

end add_fractions_l238_238651


namespace determine_angle_l238_238728

variable (a b : ℝ^3)
variable (θ : ℝ)

def magnitude_equal_two : Prop := (‖a‖ = 2) ∧ (‖b‖ = 2)

def dot_product_condition : Prop := a • (b - a) = -6

def angle_between_vectors : Prop := θ = 2 * Real.pi / 3

theorem determine_angle (h1 : magnitude_equal_two a b) (h2 : dot_product_condition a b) : angle_between_vectors a b θ := 
sorry

end determine_angle_l238_238728


namespace plane_seven_color_no_two_points_one_unit_apart_l238_238264

noncomputable def hex_tiling_coloring (a : ℝ) : Prop :=
  ∀ p₁ p₂ : ℝ × ℝ, ∀ color : ℕ, 
    (p₁ ≠ p₂ ∧ dist p₁ p₂ = 1) → 
    (tiling_coloring_function p₁ = color) → 
    (tiling_coloring_function p₂ ≠ color)

theorem plane_seven_color_no_two_points_one_unit_apart : 
  ∃ (a : ℝ), (1 / (Real.sqrt 7) < a ∧ a < 1 / 2) ∧ hex_tiling_coloring a := 
by 
  sorry

end plane_seven_color_no_two_points_one_unit_apart_l238_238264


namespace sum_powers_divisible_by_17_l238_238858

theorem sum_powers_divisible_by_17 : 
  (∑ k in Finset.range 16, (k + 1) ^ 1999) % 17 = 0 := 
by 
  sorry

end sum_powers_divisible_by_17_l238_238858


namespace max_value_frac_l238_238366

theorem max_value_frac (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 5) :
  ∃ z, z = (x + y) / x ∧ z ≤ 2 / 3 := by
  sorry

end max_value_frac_l238_238366


namespace exists_raft_shape_l238_238606

theorem exists_raft_shape (river : Type) (chip_path : river → bool) : 
  ∃ raft_shape : set river, 
    (∀ point ∈ raft_shape, chip_path point = true) ∧ 
    (∀ edge_point, edge_point ∈ raft_shape → touches_bank edge_point) := 
sorry

end exists_raft_shape_l238_238606


namespace rectangular_coordinates_of_spherical_coordinates_l238_238592

noncomputable def rectangular_coordinates (ρ θ φ : ℝ) : ℝ × ℝ × ℝ := 
  let x' := ρ * sin (φ - π/2) * cos θ
  let y' := ρ * sin (φ - π/2) * sin θ
  let z' := ρ * cos (φ - π/2)
  (x', y', z')

theorem rectangular_coordinates_of_spherical_coordinates : 
  rectangular_coordinates 7 (atan2 3 2) (acos (-6 / 7)) = (12 / sqrt 13, 18 / sqrt 13, sqrt 13) :=
by
  simp [rectangular_coordinates]
  sorry

end rectangular_coordinates_of_spherical_coordinates_l238_238592


namespace proof_problem_l238_238391

namespace ProofProblem

open Real

def line1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, 2 * x - y + a = 0
def line2 : ℝ → ℝ → Prop := λ x y, -4 * x + 2 * y + 1 = 0
def line3 : ℝ → ℝ → Prop := λ x y, x + y - 1 = 0

-- Condition: distance between line1 and line2 is (7 / 10) * sqrt 5
def dist_between_lines (a : ℝ) : Prop :=
  (| -2 * a - 1 | / sqrt (16 + 4) = (7 / 10) * sqrt 5)

-- Condition for existence of point P
def exists_point_P (a : ℝ) (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧
  -- Distance from P to line1 is half the distance from P to line2
  (| 2 * m - n + a | / sqrt 5 = (1 / 2) * (| -4 * m + 2 * n + 1 | / sqrt 20)) ∧
  -- Ratio of distance from P to line1 and line3 is sqrt 2 : sqrt 5
  (| 2 * m - n + a | / sqrt 5 / (| m + n - 1 | / sqrt 2) = sqrt 2 / sqrt 5)

-- Main theorem statement for proof problem
theorem proof_problem :
  ∃ a : ℝ, dist_between_lines a ∧
           a = 3 ∧
           ∃ m n : ℝ, exists_point_P a m n ∧
                      m = 1 / 9 ∧ n = 37 / 18 :=
begin
  minimize sorry
end

end ProofProblem

end proof_problem_l238_238391


namespace area_of_EFCD_l238_238443

noncomputable def area_of_quadrilateral (AB CD altitude: ℝ) :=
  let sum_bases_half := (AB + CD) / 2
  let small_altitude := altitude / 2
  small_altitude * (sum_bases_half + CD) / 2

theorem area_of_EFCD
  (AB CD altitude : ℝ)
  (AB_len : AB = 10)
  (CD_len : CD = 24)
  (altitude_len : altitude = 15)
  : area_of_quadrilateral AB CD altitude = 153.75 :=
by
  rw [AB_len, CD_len, altitude_len]
  simp [area_of_quadrilateral]
  sorry

end area_of_EFCD_l238_238443


namespace solve_quadratic_eq_l238_238353

theorem solve_quadratic_eq (a : ℝ) (x : ℝ) 
  (h : a ∈ ({-1, 1, a^2} : Set ℝ)) : 
  (x^2 - (1 - a) * x - 2 = 0) → (x = 2 ∨ x = -1) := by
  sorry

end solve_quadratic_eq_l238_238353


namespace decreasing_interval_gx_l238_238021

noncomputable def f : ℝ → ℝ := sorry
def g (x : ℝ) : ℝ := x^2 * f (x - 1)

theorem decreasing_interval_gx : ∀ x, 0 < x ∧ x < 1 → ∀ y, 0 < y ∧ y < x → g(x) > g(y) :=
sorry

end decreasing_interval_gx_l238_238021


namespace max_remaining_numbers_l238_238805

def numbers (n : ℕ) : set ℕ := {k | 1 ≤ k ∧ k ≤ n}

def valid_subset (s : set ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → (a - b) ≠ 0 → ¬ (a - b) ∣ c

theorem max_remaining_numbers : ∃ s : set ℕ, s ⊆ numbers 235 ∧ valid_subset s ∧ card s = 118 := 
sorry

end max_remaining_numbers_l238_238805


namespace largest_coprime_set_l238_238635

def is_coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def largest_integer_n (N : ℕ) : Prop :=
  ∃ (S : Finset ℕ), S.card = N ∧ (∀ a ∈ S, a ≤ 17) ∧ (∀ a b ∈ S, a ≠ b → is_coprime a b)

theorem largest_coprime_set : largest_integer_n 8 :=
sorry

end largest_coprime_set_l238_238635


namespace certain_event_l238_238560

def event_A : Prop := ∃ t : ℤ, t = t -- Placeholder for uncertain event (heavy rain tomorrow)

def event_B : Prop := ∃ t : ℤ, t = t -- Placeholder for uncertain event (football match on TV)

def event_D : Prop := ∃ t : ℤ, t = t -- Placeholder for uncertain event (Xiaoming scored 80 points)

def event_C : Prop :=
  let num_people : ℕ := 368
  let days_in_year : ℕ := 365
  ∃ (f : fin num_people → fin days_in_year), ∀ i j, i ≠ j → f i = f j

theorem certain_event : event_C :=
  by {
    sorry
  }

end certain_event_l238_238560


namespace cot_150_eq_neg_sqrt3_l238_238650

theorem cot_150_eq_neg_sqrt3 :
  ∀ (x : ℝ), (cot x = 1 / tan x) →
  (tan (180 - x) = -tan x) →
  (tan 30 = 1 / real.sqrt 3) →
  cot 150 = -real.sqrt 3 :=
by
  intros x h_cot h_tan_sub h_tan_30
  sorry

end cot_150_eq_neg_sqrt3_l238_238650


namespace max_sum_of_arithmetic_sequence_l238_238674

theorem max_sum_of_arithmetic_sequence (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ a b c : ℝ, x < a ∧ a < b ∧ b < c ∧ c < y ∧
  ((x + y) / 2 = b) ∧ (c = (b + y) / 2) ∧ b + c + y = (3 / 4) * (x + 3 * y) ∧
  3 * (x + 3 * y) / 4 ≤ 3 * (2 * √10) / 2 :=
sorry

end max_sum_of_arithmetic_sequence_l238_238674


namespace line_intersects_ellipse_l238_238588

noncomputable def possible_slopes : set ℝ :=
  {m : ℝ | -sqrt (9 / 5) ≤ m ∧ m ≤ sqrt (9 / 5)}

theorem line_intersects_ellipse (m : ℝ) :
  let y := λ x, m * x + 7
  let ellipse := ∀ x y, 4 * x^2 + 25 * y^2 = 100
  (∃ x y, ellipse x (y x)) ↔ m ∈ possible_slopes :=
by
  sorry

end line_intersects_ellipse_l238_238588


namespace johns_total_expenditure_is_1170_l238_238778

noncomputable def total_expenditure 
    (cost_A : ℕ → ℚ)
    (cost_B : ℕ → ℚ)
    (NP_A : ℕ)
    (NP_B : ℕ)
    (discount_A : ℕ → ℚ)
    (discount_B : ℕ → ℚ)
    (treats_A_days1_10 : ℕ)
    (treats_A_days11_20 : ℕ)
    (treats_A_days21_30 : ℕ)
    (treats_B_days1_10 : ℕ)
    (treats_B_days11_20 : ℕ)
    (treats_B_days21_30 : ℕ)
    : ℚ :=
  let total_A := (treats_A_days1_10 + treats_A_days11_20 + treats_A_days21_30)
  let total_B := (treats_B_days1_10 + treats_B_days11_20 + treats_B_days21_30)
  have NP_requirement : (NP_A * total_A + NP_B * total_B) ≥ 40 := sorry
  let base_cost_A := cost_A total_A * total_A
  let base_cost_B := cost_B total_B * total_B
  let actual_cost_A := base_cost_A * (1 - discount_A total_A)
  let actual_cost_B := base_cost_B * (1 - discount_B total_B)
  actual_cost_A + actual_cost_B

theorem johns_total_expenditure_is_1170 : total_expenditure 
  (λ n, 0.10)  -- cost per Type A treat
  (λ n, 0.15)  -- cost per Type B treat
  1            -- NP per Type A treat
  2            -- NP per Type B treat
  (λ n, if n ≥ 50 then 0.10 else 0)  -- 10% discount for 50+ Type A treats
  (λ n, if n ≥ 30 then 0.20 else 0)  -- 20% discount for 30+ Type B treats
  30          -- Type A treats per day for days 1-10
  20          -- Type A treats per day for days 11-20
  0           -- Type A treats per day for days 21-30
  0           -- Type B treats per day for days 1-10
  20          -- Type B treats per day for days 11-20
  40          -- Type B treats per day for days 21-30
  = 11.70 := 
sorry

end johns_total_expenditure_is_1170_l238_238778


namespace binom_eight_three_l238_238304

theorem binom_eight_three : Nat.choose 8 3 = 56 := by
  sorry

end binom_eight_three_l238_238304


namespace thirty_percent_greater_l238_238569

theorem thirty_percent_greater (x : ℝ) (h : x = 1.3 * 88) : x = 114.4 :=
sorry

end thirty_percent_greater_l238_238569


namespace derivative_of_y_l238_238658

-- Define the function y
def y (x : ℝ) : ℝ := - (1 / (3 * (Real.sin x)^3)) - (1 / (Real.sin x)) + (1 / 2) * Real.log ((1 + Real.sin x) / (1 - Real.sin x))

-- Statement to prove the derivative of y
theorem derivative_of_y (x : ℝ) : deriv y x = 1 / (Real.cos x * (Real.sin x)^4) := by
  sorry

end derivative_of_y_l238_238658


namespace maximum_rabbits_condition_l238_238110

-- Define the conditions and constraints
variables {N : ℕ}
variables (total_rabbits long_ears jump_far : ℕ)
variables (at_least_three_with_both : Prop)

-- State the conditions with exact values and assumptions
def conditions := 
  total_rabbits = N ∧
  long_ears = 13 ∧
  jump_far = 17 ∧
  at_least_three_with_both = (∃ a b c : ℕ, a >= 3 ∧ b = (long_ears - a) ∧ c = (jump_far - a))

-- State the theorem to be proved
theorem maximum_rabbits_condition :
  ∀ {N : ℕ}, conditions N long_ears jump_far at_least_three_with_both → N ≤ 27 :=
by sorry

end maximum_rabbits_condition_l238_238110


namespace maximum_rabbits_l238_238088

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l238_238088


namespace sine_five_l238_238549

noncomputable def sine_value (x : ℝ) : ℝ :=
  Real.sin (5 * x)

theorem sine_five : sine_value 1 = -0.959 := 
  by
  sorry

end sine_five_l238_238549


namespace triangle_area_l238_238129

variable (x y : ℝ)

def line_eq : Prop := x - 2*y - 3 = 0

def circle_eq : Prop := (x - 2) ^ 2 + (y + 3) ^ 2 = 9

theorem triangle_area {x y : ℝ} (E F C : Point ℝ) : 
  (line_eq x y) ∧ (circle_eq x y) → 
  triangle_area E F C = 2 * Real.sqrt 5 := by
  sorry

end triangle_area_l238_238129


namespace trigonometric_identity_l238_238705

theorem trigonometric_identity (t : ℝ) (ht : t > 0) :
  let P := (-4*t, 3*t)
  let α := real.angle.of_pt P
  2 * real.angle.sin α + real.angle.cos α = 2 / 5 :=
sorry

end trigonometric_identity_l238_238705


namespace imaginary_part_of_z_l238_238677

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + complex.i) = 1) : complex.im z = -1/5 := 
sorry

end imaginary_part_of_z_l238_238677


namespace avg_starting_with_d_l238_238039

-- Define c and d as positive integers
variables (c d : ℤ) (hc : c > 0) (hd : d > 0)

-- Define d as the average of the seven consecutive integers starting with c
def avg_starting_with_c (c : ℤ) : ℤ := (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7

-- Define the condition that d is the average of the seven consecutive integers starting with c
axiom d_is_avg_starting_with_c : d = avg_starting_with_c c

-- Prove that the average of the seven consecutive integers starting with d equals c + 6
theorem avg_starting_with_d (c d : ℤ) (hc : c > 0) (hd : d > 0) (h : d = avg_starting_with_c c) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 6 := by
  sorry

end avg_starting_with_d_l238_238039


namespace sum_of_1000_consecutive_odd_is_seventh_power_l238_238296

theorem sum_of_1000_consecutive_odd_is_seventh_power :
  ∃ n : ℕ, let S := (list.range 1000).map (λ k, 2*k + 1) in (S.sum = n^7) :=
sorry

end sum_of_1000_consecutive_odd_is_seventh_power_l238_238296


namespace production_growth_rate_eq_l238_238662

theorem production_growth_rate_eq 
  (x : ℝ)
  (H : 100 + 100 * (1 + x) + 100 * (1 + x) ^ 2 = 364) : 
  100 + 100 * (1 + x) + 100 * (1 + x) ^ 2 = 364 :=
by {
  sorry
}

end production_growth_rate_eq_l238_238662


namespace exists_n_for_sum_to_9999_l238_238310

theorem exists_n_for_sum_to_9999 : 
  ∃ n : ℕ, (n > 2013) ∧ ∃ (f : (ℕ → ℤ) → ℤ), f (λ k, if k.mod 2 = 0 then (k * k : ℤ) else - (k * k : ℤ)) = 9999 := 
sorry

end exists_n_for_sum_to_9999_l238_238310


namespace geom_seq_sum_4n_l238_238144

-- Assume we have a geometric sequence with positive terms and common ratio q
variables (a : ℕ → ℝ) (q : ℝ) (n : ℕ)

-- The sum of the first n terms of the geometric sequence is S_n
noncomputable def S_n : ℝ := a 0 * (1 - q^n) / (1 - q)

-- Given conditions
axiom h1 : S_n a q n = 2
axiom h2 : S_n a q (3 * n) = 14

-- We need to prove that S_{4n} = 30
theorem geom_seq_sum_4n : S_n a q (4 * n) = 30 :=
by
  sorry

end geom_seq_sum_4n_l238_238144


namespace log_condition_necessary_not_sufficient_l238_238468

noncomputable def base_of_natural_logarithm := Real.exp 1

variable (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : 0 < b) (h4 : b ≠ 1)

theorem log_condition_necessary_not_sufficient (h : 0 < a ∧ a < b ∧ b < 1) :
  (Real.log 2 / Real.log a > Real.log base_of_natural_logarithm / Real.log b) :=
sorry

end log_condition_necessary_not_sufficient_l238_238468


namespace billy_sleep_total_l238_238746

theorem billy_sleep_total :
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  day1 + day2 + day3 + day4 = 30 :=
by
  -- Definitions
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  -- Assertion
  have h : day1 + day2 + day3 + day4 = 30 := sorry
  exact h

end billy_sleep_total_l238_238746


namespace maximum_rabbits_l238_238085

theorem maximum_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : ∀ n ≤ N, 3 ≤ 13 + 17 - N) : 
  N ≤ 27 :=
by {
  sorry
}

end maximum_rabbits_l238_238085


namespace functional_eq_solution_l238_238328

/-- Define the function f: ℝ → ℝ
    1. The set {f(x) / x | x ∈ ℝ \ {0}} is finite.
    2. f(x - 1 - f(x)) = f(x) - 1 - x for all x.
    We need to show f(x) = x ∨ f(x) = -x -/

theorem functional_eq_solution (f : ℝ → ℝ)
  (h1 : ∃ n : ℕ, ∃ cs : fin n → ℝ, ∀ x ≠ 0, ∃ i : fin n, f(x) = (cs i) * x)
  (h2 : ∀ x, f(x - 1 - f(x)) = f(x) - 1 - x) :
  ∀ x, f(x) = x ∨ f(x) = -x :=
by
  sorry

end functional_eq_solution_l238_238328


namespace harry_bus_time_l238_238400

variable (x : ℝ) -- Define x as a real number representing the time Harry has been on the bus

-- Define the rest of the bus journey time
def rest_of_journey : ℝ := 25

-- Define the total bus journey time
def total_bus_journey : ℝ := x + rest_of_journey

-- Define the walking time
def walking_time : ℝ := (1/2) * total_bus_journey

-- Define the total travel time
def total_travel_time : ℝ := x + rest_of_journey + walking_time

-- The problem statement in Lean
theorem harry_bus_time : total_travel_time x = 60 → x = 15 :=
by
  -- Sorry is used to skip the proof
  sorry

end harry_bus_time_l238_238400


namespace median_AD_equation_altitude_BH_equation_l238_238694

open Set

-- Define the coordinates of the vertices of the triangle
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (5, 1)
def C : ℝ × ℝ := (-1, -1)

-- Define the equations of the lines
def median_AD : LinearMap ℝ ℝ := LinearMap.mk₂ 3 1 (-6)
def altitude_BH : LinearMap ℝ ℝ := LinearMap.mk₂ 1 2 (-7)

-- Prove the equations given the coordinates
theorem median_AD_equation :
  (∃ m : LinearMap ℝ ℝ, m = median_AD) :=
sorry

theorem altitude_BH_equation :
  (∃ m : LinearMap ℝ ℝ, m = altitude_BH) :=
sorry

end median_AD_equation_altitude_BH_equation_l238_238694


namespace distance_traveled_l238_238248

variable (v d : ℝ)
axiom cond1 : d = v * 9
axiom cond2 : d = (v + 20) * 6

theorem distance_traveled : d = 360 := by
  conv_lhs { rw [cond2] }
  conv_rhs { rw [cond1] }
  simp at *
  sorry

end distance_traveled_l238_238248


namespace minimize_f_l238_238183

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l238_238183


namespace distance_between_intersections_l238_238897

-- Define the parabola and circle equations
def parabola (y : ℝ) : ℝ := (y^2 / 12)
def circle (x y : ℝ) : ℝ := (x^2 + y^2 - 4*x - 6*y)

-- Define the proof problem
theorem distance_between_intersections :
  ∃ P Q : ℝ × ℝ, P ≠ Q ∧ 
              P.2^2 = 12 * P.1 ∧ 
              circle P.1 P.2 = 0 ∧
              Q.2^2 = 12 * Q.1 ∧ 
              circle Q.1 Q.2 = 0 ∧
              dist P Q = 3 * real.sqrt 13 :=
sorry

end distance_between_intersections_l238_238897


namespace minimize_f_at_3_l238_238211

-- Define the quadratic function f(x) = 3x^2 - 18x + 7
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

-- The theorem stating that f(x) attains its minimum when x = 3
theorem minimize_f_at_3 : ∀ x : ℝ, f(x) ≥ f(3) := 
by 
  sorry

end minimize_f_at_3_l238_238211


namespace sin_2alpha_l238_238672

-- Define the vectors and their dot product in Lean
def a (α : ℝ) : ℝ × ℝ := (1, Real.cos α)
def b (α : ℝ) : ℝ × ℝ := (Real.sin α, 1)

-- Define the orthogonality condition
def orthogonal (α : ℝ) : Prop := (a α).fst * (b α).fst + (a α).snd * (b α).snd = 0

-- Define the final theorem to be proven
theorem sin_2alpha (α : ℝ) (h : orthogonal α) : Real.sin (2 * α) = -1 := 
by
  sorry

end sin_2alpha_l238_238672


namespace minimize_f_l238_238185

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l238_238185


namespace products_in_polynomial_expansion_pow3_two_digit_numbers_count_sum_of_products_four_digit_numbers_l238_238498

-- Part (a)
theorem products_in_polynomial_expansion_pow3 : 
  ∀ (a b c d : ℕ), (a = 1) → (b = 2) → (c = 3) → (d = 4) → ((a + b + c + d) ^ 3) = 64 :=
sorry

-- Part (b)
theorem two_digit_numbers_count :
  ∀ (digits : Finset ℕ), (digits = {1, 2, 3, 4}) → (digits.card * digits.card) = 16 :=
sorry

-- Part (c)
theorem sum_of_products_four_digit_numbers :
  ∀ (digits : Finset ℕ), (digits = {1, 2, 3, 4}) → ((digits.sum) ^ 4) = 10000 :=
sorry

end products_in_polynomial_expansion_pow3_two_digit_numbers_count_sum_of_products_four_digit_numbers_l238_238498


namespace transformations_return_to_origin_triangl_U_l238_238600

noncomputable def triangle_U_transformations_return_to_origin : Nat :=
  let transformations := [
    (λ p : ℤ × ℤ, (-p.2, p.1)),  -- rotation 90°
    (λ p : ℤ × ℤ, (-p.1, -p.2)), -- rotation 180°
    (λ p : ℤ × ℤ, (p.2, -p.1)),  -- rotation 270°
    (λ p : ℤ × ℤ, (p.2, p.1)),   -- reflection y=x
    (λ p : ℤ × ℤ, (-p.2, -p.1))  -- reflection y=-x
  ]

/-- The number of sequences of three transformations that return a triangle with vertices at 
(0, 0), (6, 0), and (0, 2) to its original position is 14. -/
theorem transformations_return_to_origin_triangl_U :
  (count_equivalent_transformations (0, 0) (6, 0) (0, 2) transformations 3) = 14 := sorry

end transformations_return_to_origin_triangl_U_l238_238600


namespace area_of_region_l238_238786

def region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + y + floor x + floor y ≤ 5

theorem area_of_region :
  (∫∫ (x y : ℝ) in region, 1) = 9 / 2 :=
by
  sorry

end area_of_region_l238_238786


namespace sand_exchange_impossible_to_achieve_l238_238425

-- Let G and P be the initial weights of gold and platinum sand, respectively
def initial_G : ℕ := 1 -- 1 kg
def initial_P : ℕ := 1 -- 1 kg

-- Initial values for g and p
def initial_g : ℕ := 1001
def initial_p : ℕ := 1001

-- Daily reduction of either g or p
axiom decrease_g_or_p (g p : ℕ) : g > 1 ∨ p > 1 → (g = g - 1 ∨ p = p - 1) ∧ (g ≥ 1) ∧ (p ≥ 1)

-- Final condition: after 2000 days, g and p both equal to 1
axiom final_g_p_after_2000_days : ∀ (g p : ℕ), (g = initial_g - 2000) ∧ (p = initial_p - 2000) → g = 1 ∧ p = 1

-- State of the system, defined as S = G * p + P * g
def S (G P g p : ℕ) : ℕ := G * p + P * g

-- Prove that after 2000 days, the banker cannot have at least 2 kg of each type of sand
theorem sand_exchange_impossible_to_achieve (G P g p : ℕ) (h : G = initial_G) (h1 : P = initial_P) 
  (h2 : g = initial_g) (h3 : p = initial_p) : 
  ∀ (d : ℕ), (d = 2000) → (g = 1) ∧ (p = 1) 
    → (S G P g p < 4) :=
by
  sorry

end sand_exchange_impossible_to_achieve_l238_238425


namespace simplify_fraction_l238_238045

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 :=
by
  sorry

end simplify_fraction_l238_238045


namespace imaginary_part_of_z_l238_238680

-- Definition based on the problem condition
def z_condition (z : ℂ) : Prop := z * (2 + complex.i) = 1

-- Statement of the proof problem
theorem imaginary_part_of_z (z : ℂ) (h : z_condition z) : complex.im z = -1/5 :=
sorry

end imaginary_part_of_z_l238_238680


namespace min_value_at_3_l238_238203

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end min_value_at_3_l238_238203


namespace simplify_fraction_l238_238055

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l238_238055


namespace minimize_f_l238_238187

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l238_238187


namespace union_when_m_equals_4_subset_implies_m_range_l238_238483

-- Define the sets and conditions
def set_A := { x : ℝ | -2 ≤ x ∧ x ≤ 5 }
def set_B (m : ℝ) := { x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Problem 1: When m = 4, find the union of A and B
theorem union_when_m_equals_4 : ∀ x, x ∈ set_A ∪ set_B 4 ↔ -2 ≤ x ∧ x ≤ 7 :=
by sorry

-- Problem 2: If B ⊆ A, find the range of the real number m
theorem subset_implies_m_range (m : ℝ) : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ m ≤ 3 :=
by sorry

end union_when_m_equals_4_subset_implies_m_range_l238_238483


namespace max_remaining_numbers_l238_238834

/-- 
The board initially has numbers 1, 2, 3, ..., 235.
Among the remaining numbers, no number is divisible by the difference of any two others.
Prove that the maximum number of numbers that could remain on the board is 118.
-/
theorem max_remaining_numbers : 
  ∃ S : set ℕ, (∀ a ∈ S, 1 ≤ a ∧ a ≤ 235) ∧ (∀ a b ∈ S, a ≠ b → ¬ ∃ d, d ∣ (a - b)) ∧ 
  ∃ T : set ℕ, S ⊆ T ∧ T ⊆ finset.range 236 ∧ T.card = 118 := 
sorry

end max_remaining_numbers_l238_238834


namespace option_C_is_only_linear_system_l238_238935

def is_linear_equation (eqn : String) : Prop := sorry

def is_system_of_two_linear_equations (system : List String) : Prop :=
  system.length = 2 ∧ system.forall is_linear_equation

theorem option_C_is_only_linear_system :
  let optionA := ["x + y = 1", "1 / x + 1 / y = 8"]
  let optionB := ["x + y = 5", "y + z = 7"]
  let optionC := ["x = 1", "3x - 2y = 6"]
  let optionD := ["x - y = xy", "x - y = 1"]
  is_system_of_two_linear_equations optionC ∧
  ¬(is_system_of_two_linear_equations optionA) ∧
  ¬(is_system_of_two_linear_equations optionB) ∧
  ¬(is_system_of_two_linear_equations optionD) :=
by
  sorry

end option_C_is_only_linear_system_l238_238935


namespace reduced_price_per_kg_of_oil_l238_238268

/-- The reduced price per kg of oil is approximately Rs. 48 -
given a 30% reduction in price and the ability to buy 5 kgs more
for Rs. 800. -/
theorem reduced_price_per_kg_of_oil
  (P R : ℝ)
  (h1 : R = 0.70 * P)
  (h2 : 800 / R = (800 / P) + 5) : 
  R = 48 :=
sorry

end reduced_price_per_kg_of_oil_l238_238268


namespace ellipse_equation_and_circle_correct_l238_238480

noncomputable def ellipse (x y : ℝ) (a b : ℝ) := (x^2) / (a^2) + (y^2) / (b^2) = 1

theorem ellipse_equation_and_circle_correct :
  (ellipse 2 (Real.sqrt 2) a b) ∧ (ellipse (Real.sqrt 6) 1 a b) 
  → (a = Real.sqrt 8) ∧ (b = 2) 
  ∧ (
      ∃ r : ℝ, r^2 = 8 / 3 ∧
      ∀ k : ℝ, ∀ m : ℝ,
        let t : ℝ := 8 * k^2 - m^2 + 4 in
        t > 0 → 
        (3 * m^2 - 8 * k^2 - 8 = 0) 
        ∧
        (x y : ℝ, ellipse x y 2 (Real.sqrt 8)) → 
        k^2 = (3 * m^2 - 8) / 8 
        ∧ m^2 > 2 ∧ 3 * m^2 ≥ 8 
        ∧ r = Real.sqrt (8 / 3) 
    )
by {
  sorry
}

end ellipse_equation_and_circle_correct_l238_238480


namespace f_divisible_by_4_f_not_divisible_by_5_l238_238465

def f (n : ℤ) : ℤ := 4 * (n^2 + 2*n + 2)

theorem f_divisible_by_4 (n : ℤ) : 4 ∣ f n := by
  unfold f
  apply dvd_mul_right

theorem f_not_divisible_by_5 (n : ℤ) : ¬(5 ∣ f n) := by
  unfold f
  intro h
  have : 5 ∣ 4 * (n^2 + 2*n + 2) := h
  have : 5 ∣ (n^2 + 2*n + 2) :=
  begin
    apply (dvd_of_mul_dvd_left (by norm_num)).mp,
    exact this,
  end,
  have h_mod : (n^2 + 2*n + 2) % 5 = 3 := sorry,
  rw [int.dvd_iff_mod_eq_zero] at this,
  linarith,
  sorry

end f_divisible_by_4_f_not_divisible_by_5_l238_238465


namespace ratio_of_segments_sum_one_l238_238604

theorem ratio_of_segments_sum_one
  (A B C D E : Type)
  [is_equilateral_triangle A B C]
  (D_on_AB : D ∈ line_segment A B)
  (E_on_AC : E ∈ line_segment A C)
  (DE_touches_incircle : touches_incircle D E A B C) :
  (line_segment_ratio A D D B + line_segment_ratio A E E C = 1) :=
sorry

end ratio_of_segments_sum_one_l238_238604


namespace average_mark_of_excluded_students_l238_238065

noncomputable def average_mark_excluded (A : ℝ) (N : ℕ) (R : ℝ) (excluded_count : ℕ) (remaining_count : ℕ) : ℝ :=
  ((N : ℝ) * A - (remaining_count : ℝ) * R) / (excluded_count : ℝ)

theorem average_mark_of_excluded_students : 
  average_mark_excluded 70 10 90 5 5 = 50 := 
by 
  sorry

end average_mark_of_excluded_students_l238_238065


namespace ab_sum_l238_238681

theorem ab_sum (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 7) (h3 : |a - b| = b - a) : a + b = 10 ∨ a + b = 4 :=
by
  sorry

end ab_sum_l238_238681


namespace maximize_area_of_CDFE_l238_238240

theorem maximize_area_of_CDFE :
  let rectangle_length := 2
  let rectangle_width := 1
  let AE := x
  let AF := y
  let area_CDFE (x y : ℝ) : ℝ := 
    1/2 * (2 - x) * y + 1/2 * (1 - y) * x
  in x = 1/2 ∧ y = 1/2 → 
     area_CDFE x y = 5/8 :=
by
  sorry

end maximize_area_of_CDFE_l238_238240


namespace customers_left_l238_238977

theorem customers_left (initial_customers remaining_customers : ℕ) 
  (h_initial : initial_customers = 14) 
  (h_remaining : remaining_customers = 9) :
  initial_customers - remaining_customers = 5 :=
by
  rw [h_initial, h_remaining]
  norm_num
  sorry -- Placeholder to ensure the theorem can compile successfully

end customers_left_l238_238977


namespace probability_B_given_A_is_one_sixth_l238_238161

noncomputable def event_A {Ω : Type} (outcome : Ω) : Prop :=
  -- Dice A shows greater than 4 (5 or 6)
  (outcome = 5) ∨ (outcome = 6)

noncomputable def event_B {Ω : Type} (outcome_A : Ω) (outcome_B : Ω) : Prop :=
  -- Sum of dice A and B equals 7
  outcome_A + outcome_B = 7

noncomputable def probability_B_given_A {Ω : Type} (outcome_A : Ω) (outcome_B : Ω) : ℚ :=
  if event_A outcome_A then
    (if event_B outcome_A outcome_B then 1 else 0) / 12
  else
    0

theorem probability_B_given_A_is_one_sixth {Ω : Type} (outcome_A outcome_B : Ω) :
  probability_B_given_A outcome_A outcome_B = 1 / 6 :=
begin
  sorry
end

end probability_B_given_A_is_one_sixth_l238_238161


namespace dihedral_angle_BDE_BDC_l238_238442

/-- 
  Given a triangular prism S-ABC,
  1. SA is perpendicular to the plane ABC,
  2. AB is perpendicular to BC,
  3. E is the midpoint of SC,
  4. D is a point on AC,
  5. DE is perpendicular to SC,
  6. SA equals AB,
  7. SB equals BC.

  Prove that the dihedral angle between the faces BDE and BDC using BD as the edge is 30 degrees.
-/
-- Defining the given conditions
variables (S A B C D E : Type)
variables [OrderedSemiring S] [MetricSpace S] [normedAddCommGroup V] [normedSpace ℝ V]

-- Essential conditions for the proof
def is_perpendicular_to_plane (v : V) (p : AffineSubspace ℝ V) : Prop := sorry
def midpoint (a b : Point) : Point := sorry
def point_on (p : Point) (line : AffineSubspace ℝ V) : Prop := sorry
def same_length (a b : Point) (c d : Point) : Prop := sorry

-- Assumptions from the problem
axiom SA_perp_plane_ABC : is_perpendicular_to_plane (S - A) (affineSpan ℝ {A, B, C})
axiom AB_perp_BC : ∀ A B C : Point, is_perpendicular (A - B) (B - C)
axiom E_is_midpoint_of_SC : E = midpoint S C
axiom D_is_point_on_AC : point_on D (line ℝ A C)
axiom DE_perp_SC : is_perpendicular (D - E) (S - C)
axiom SA_eq_AB : same_length S A A B
axiom SB_eq_BC : same_length S B B C

-- The goal to prove the dihedral angle is 30 degrees
theorem dihedral_angle_BDE_BDC : angle_between_faces (affineSpan ℝ {B, D, E}) 
  (affineSpan ℝ {B, D, C}) = 30 := 
sorry

end dihedral_angle_BDE_BDC_l238_238442


namespace middle_group_frequency_l238_238428

theorem middle_group_frequency (capacity : ℕ) (n_rectangles : ℕ) (A_mid A_other : ℝ) 
  (h_capacity : capacity = 300)
  (h_rectangles : n_rectangles = 9)
  (h_areas : A_mid = 1 / 5 * A_other)
  (h_total_area : A_mid + A_other = 1) : 
  capacity * A_mid = 50 := by
  sorry

end middle_group_frequency_l238_238428


namespace range_of_a_l238_238417

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = log (a * x^2 - 2 * x + 1))
    (h2 : set.range f = set.univ) : a ∈ set.Icc 0 1 := sorry

end range_of_a_l238_238417


namespace find_number_l238_238250

theorem find_number (x : ℝ) (h : 0.3 * x + 0.1 * 0.5 = 0.29) : x = 0.8 :=
by
  sorry

end find_number_l238_238250


namespace infinitely_many_n_with_property_l238_238036

theorem infinitely_many_n_with_property :
  ∃ᶠ n : ℕ in at_top, ∀ p : ℕ, prime p → p ∣ n^2 + 3 → 
  ∃ m : ℕ, p ∣ m^2 + 3 ∧ m^2 < n := 
by
  sorry

end infinitely_many_n_with_property_l238_238036


namespace number_of_distinguishable_large_triangles_eq_84_l238_238912

open Function

def number_of_distinguishable_large_triangles (C : Finset Color) (c₀ : Color) (h: c₀ ∈ C) : ℕ :=
  let C' := C.erase c₀ in
  let all_same_color := C'.card in
  let two_same_one_diff := C'.card * (C'.card - 1) in
  let all_different_colors := Nat.choose C'.card 3 in
  all_same_color + two_same_one_diff + all_different_colors

theorem number_of_distinguishable_large_triangles_eq_84 (C : Finset Color) (c₀ : Color) (h: c₀ ∈ C) (hC : C.card = 8) : 
  number_of_distinguishable_large_triangles C c₀ h = 84 := by
  sorry

end number_of_distinguishable_large_triangles_eq_84_l238_238912


namespace vegan_meals_count_l238_238798

theorem vegan_meals_count 
  (total_clients : ℕ) (kosher_clients : ℕ) 
  (both_vegan_kosher_clients : ℕ) 
  (neither_kosher_vegan_clients : ℕ) 
  (total_clients = 30) 
  (kosher_clients = 8) 
  (both_vegan_kosher_clients = 3) 
  (neither_kosher_vegan_clients = 18)
  : ∃ V, V = 10 := by 
  sorry

end vegan_meals_count_l238_238798


namespace limit_example_l238_238991

theorem limit_example :
  (tendsto (λ x : ℝ, (exp (2 * x) - exp (-5 * x)) / (2 * sin x - tan x)) (nhds 0) (nhds 7)) :=
  sorry

end limit_example_l238_238991


namespace base_angle_is_15_l238_238985

-- Define an isosceles triangle with one angle being 150°
inductive Triangle
| mk (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : Triangle

def is_isosceles (T : Triangle) : Prop :=
  match T with
  | Triangle.mk a b c ha hb hc => a = b ∨ b = c ∨ a = c

def has_angle (T : Triangle) (angle : ℝ) : Prop :=
  match T with
  | Triangle.mk a b c ha hb hc => angle = a ∨ angle = b ∨ angle = c

-- Given conditions
axiom iso_triangle : Triangle.mk 150 15 15 sorry sorry sorry
axiom is_iso : is_isosceles iso_triangle
axiom has_150_degrees : has_angle iso_triangle 150

-- The theorem statement
theorem base_angle_is_15 : ∃ (α : ℝ), α = 15 ∧ 
  (match iso_triangle with
  | Triangle.mk a b c ha hb hc => (a = α ∨ b = α ∨ c = α) ∧ (a + b + c = 180)) :=
by
  sorry

end base_angle_is_15_l238_238985


namespace arithmetic_progression_nth_term_l238_238942

theorem arithmetic_progression_nth_term (u1 d : ℕ) (n : ℕ) : 
  (arity: ℕ) → (sum x between 0 and arity - 1) = u1 + (n - 1) * d :=
sorry

end arithmetic_progression_nth_term_l238_238942


namespace angle_between_vectors_magnitude_of_sum_l238_238727

noncomputable def vector_a : ℝ × ℝ := (Real.sqrt 3, 1)

def vector_dot (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

axiom a_dot_b : ℝ
axiom condition1 : vector_dot vector_a (2 • vector_b) = -12
axiom condition2 : vector_dot vector_a vector_b = 2

theorem angle_between_vectors (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h1 : a = σ) (h2: vector_dot a (2 • b) = -12) (h3: ℝ) :
  ∃θ ∈ [0,π], θ = π / 3 :=
sorry

theorem magnitude_of_sum (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = vector_a) (hb : vector_dot a b = 2) (hb1 : vector_dot a (2 • vector_b) = -12) :
  |2 • a + b| = 2 • sqrt(7) :=
sorry

end angle_between_vectors_magnitude_of_sum_l238_238727


namespace lake_circumference_ratio_l238_238253

theorem lake_circumference_ratio 
    (D C : ℝ) 
    (hD : D = 100) 
    (hC : C = 314) : 
    C / D = 3.14 := 
sorry

end lake_circumference_ratio_l238_238253


namespace complex_division_l238_238242

theorem complex_division  :
  (4 - 2 * complex.i) / (1 + complex.i) = 1 - 3 * complex.i :=
by
  sorry

end complex_division_l238_238242


namespace incorrect_statement_l238_238609

noncomputable def first_line_of_defense := "Skin and mucous membranes"
noncomputable def second_line_of_defense := "Antimicrobial substances and phagocytic cells in body fluids"
noncomputable def third_line_of_defense := "Immune organs and immune cells"
noncomputable def non_specific_immunity := "First and second line of defense"
noncomputable def specific_immunity := "Third line of defense"
noncomputable def d_statement := "The defensive actions performed by the three lines of defense in the human body are called non-specific immunity"

theorem incorrect_statement : d_statement ≠ specific_immunity ∧ d_statement ≠ non_specific_immunity := by
  sorry

end incorrect_statement_l238_238609


namespace temperature_difference_correct_l238_238939

def refrigerator_temp : ℝ := 3
def freezer_temp : ℝ := -10
def temperature_difference : ℝ := refrigerator_temp - freezer_temp

theorem temperature_difference_correct : temperature_difference = 13 := 
by
  sorry

end temperature_difference_correct_l238_238939


namespace delores_remaining_money_l238_238642

variable (delores_money : ℕ := 450)
variable (computer_price : ℕ := 1000)
variable (computer_discount : ℝ := 0.30)
variable (printer_price : ℕ := 100)
variable (printer_tax_rate : ℝ := 0.15)
variable (table_price_euros : ℕ := 200)
variable (exchange_rate : ℝ := 1.2)

def computer_sale_price : ℝ := computer_price * (1 - computer_discount)
def printer_total_cost : ℝ := printer_price * (1 + printer_tax_rate)
def table_cost_dollars : ℝ := table_price_euros * exchange_rate
def total_cost : ℝ := computer_sale_price + printer_total_cost + table_cost_dollars
def remaining_money : ℝ := delores_money - total_cost

theorem delores_remaining_money : remaining_money = -605 := by
  sorry

end delores_remaining_money_l238_238642


namespace min_value_at_3_l238_238204

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end min_value_at_3_l238_238204


namespace max_remaining_numbers_l238_238808

def numbers (n : ℕ) : set ℕ := {k | 1 ≤ k ∧ k ≤ n}

def valid_subset (s : set ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → (a - b) ≠ 0 → ¬ (a - b) ∣ c

theorem max_remaining_numbers : ∃ s : set ℕ, s ⊆ numbers 235 ∧ valid_subset s ∧ card s = 118 := 
sorry

end max_remaining_numbers_l238_238808


namespace positive_difference_in_x_coordinates_l238_238343

theorem positive_difference_in_x_coordinates (
  -- Conditions for line l
  pt_l1 : (ℝ × ℝ) := (0, 10), 
  pt_l2 : (ℝ × ℝ) := (4, 0),
  
  -- Conditions for line m
  pt_m1 : (ℝ × ℝ) := (0, 3), 
  pt_m2 : (ℝ × ℝ) := (9, 0),
  
  -- Coordinates when reaching y = 20
  x_l : ℝ := -4,
  x_m : ℝ := -51
) : (abs (x_l - x_m) = 47) :=
sorry

end positive_difference_in_x_coordinates_l238_238343


namespace distinct_real_roots_infinite_representation_l238_238863

theorem distinct_real_roots_infinite_representation :
  ∃ α β : ℝ, α ≠ β ∧ α > 1 ∧ β > 1 ∧
  ∀ n : ℕ, ∃ (r s : ℕ), n = ⌊r * α⌋ ∧ n = ⌊s * β⌋ ∧ 
  (∃ (m : ℕ), ∀ m : ℕ, ∃ r s : ℕ, m = ⌊r * α⌋ ∧ m = ⌊s * β⌋) :=
begin
  sorry
end

end distinct_real_roots_infinite_representation_l238_238863


namespace a5_eq_neg3_l238_238439

-- Define arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sequence with given conditions
def a (n : ℕ) : ℤ :=
  if n = 2 then -5
  else if n = 8 then 1
  else sorry  -- Placeholder for other values

axiom a3_eq_neg5 : a 2 = -5
axiom a9_eq_1 : a 8 = 1
axiom a_is_arithmetic : is_arithmetic_sequence a

-- Statement to prove
theorem a5_eq_neg3 : a 4 = -3 :=
by
  sorry

end a5_eq_neg3_l238_238439


namespace age_problem_l238_238763

variable (x : ℕ)
variable (Jacob_age son_age : ℕ)

-- Define the current ages of Jacob and his son
def Jacob := 40
def son := 10

-- Condition 1: In x years, Jacob's age will be three times that of his son's age.
def condition1 : Prop := (Jacob + x) = 3 * (son + x)

-- Condition 2: The same number of years ago, Jacob's age was seven times that of his son's age.
def condition2 : Prop := (Jacob - x) = 7 * (son - x)

theorem age_problem : condition1 x Jacob son ∧ condition2 x Jacob son → x = 5 :=
by
  sorry

end age_problem_l238_238763


namespace foci_of_ellipse_l238_238513

theorem foci_of_ellipse (m n : ℝ) (h : m < n ∧ n < 0) :
  ∃ f : ℝ, ∀ p ∈ ({ (0, f), (0, -f) } : set (ℝ × ℝ)),
  (∀ x y : ℝ, m * x^2 + n * y^2 + m * n = 0 ↔ (y^2 / -m + x^2 / -n) = 1) ∧ f = sqrt (n - m) :=
by sorry

end foci_of_ellipse_l238_238513


namespace equilateral_triangle_circle_tangent_l238_238016

theorem equilateral_triangle_circle_tangent
  (A B C D M N P Q : Point)
  (h_equilateral : equilateral_triangle A B C)
  (hD_on_BC : segment_contains B C D)
  (circle_touch_D : ∃ γ, circle γ ∧ touches_at γ B C D)
  (hM_on_AB : segment_contains A B M)
  (hN_on_AB : segment_contains A B N)
  (hP_on_AC : segment_contains A C P)
  (hQ_on_AC : segment_contains A C Q)
  (circle_intersect_AB_AC : ∃ γ, circle γ ∧ intersects_at γ A B M N ∧ intersects_at γ A C P Q)
  :  dist_between B D + dist_between A M + dist_between A N = dist_between C D + dist_between A P + dist_between A Q :=
sorry

end equilateral_triangle_circle_tangent_l238_238016


namespace largest_percent_error_l238_238455
noncomputable def max_percent_error (d : ℝ) (d_err : ℝ) (r_err : ℝ) : ℝ :=
  let d_min := d - d * d_err
  let d_max := d + d * d_err
  let r := d / 2
  let r_min := r - r * r_err
  let r_max := r + r * r_err
  let area_actual := Real.pi * r^2
  let area_d_min := Real.pi * (d_min / 2)^2
  let area_d_max := Real.pi * (d_max / 2)^2
  let area_r_min := Real.pi * r_min^2
  let area_r_max := Real.pi * r_max^2
  let error_d_min := (area_actual - area_d_min) / area_actual * 100
  let error_d_max := (area_d_max - area_actual) / area_actual * 100
  let error_r_min := (area_actual - area_r_min) / area_actual * 100
  let error_r_max := (area_r_max - area_actual) / area_actual * 100
  max (max error_d_min error_d_max) (max error_r_min error_r_max)

theorem largest_percent_error 
  (d : ℝ) (d_err : ℝ) (r_err : ℝ) 
  (h_d : d = 30) (h_d_err : d_err = 0.15) (h_r_err : r_err = 0.10) : 
  max_percent_error d d_err r_err = 31.57 := by
  sorry

end largest_percent_error_l238_238455


namespace medial_triangle_AB_AC_BC_l238_238010

theorem medial_triangle_AB_AC_BC
  (l m n : ℝ)
  (A B C : Type)
  (midpoint_BC := (l, 0, 0))
  (midpoint_AC := (0, m, 0))
  (midpoint_AB := (0, 0, n)) :
  (AB^2 + AC^2 + BC^2) / (l^2 + m^2 + n^2) = 8 :=
by
  sorry

end medial_triangle_AB_AC_BC_l238_238010


namespace sum_ineq_l238_238479

theorem sum_ineq (a b : ℕ → ℝ) :
  let S := (finset.range 2020).sum 
  (λ m, (finset.range 2020).sum 
    (λ n, a (m+1) * b (n+1) / (real.sqrt (m+1) + real.sqrt (n+1)) ^ 2)) 
  in S ≤ 2 * real.sqrt ((finset.range 2020).sum (λ m, a (m+1)^2)) *
           real.sqrt ((finset.range 2020).sum (λ n, b (n+1)^2)) :=
sorry

end sum_ineq_l238_238479


namespace compute_abs_3m_minus_2n_l238_238761

theorem compute_abs_3m_minus_2n {m n : ℤ}
  (h1 : ∀ x, x = 3 → 3 * x ^ 3 - m * x + n = 0)
  (h2 : ∀ x, x = -4 → 3 * x ^ 3 - m * x + n = 0) :
  |3 * m - 2 * n| = 33 :=
by
  intro x
  sorry

end compute_abs_3m_minus_2n_l238_238761


namespace part_I_part_II_l238_238437

noncomputable def curve_M (theta : ℝ) : ℝ := 4 * Real.cos theta

noncomputable def line_l (t m alpha : ℝ) : ℝ × ℝ :=
  let x := m + t * Real.cos alpha
  let y := t * Real.sin alpha
  (x, y)

theorem part_I (varphi : ℝ) :
  let OB := curve_M (varphi + π / 4)
  let OC := curve_M (varphi - π / 4)
  let OA := curve_M varphi
  OB + OC = Real.sqrt 2 * OA := by
  sorry

theorem part_II (m alpha : ℝ) :
  let varphi := π / 12
  let B := (1, Real.sqrt 3)
  let C := (3, -Real.sqrt 3)
  exists t1 t2, line_l t1 m alpha = B ∧ line_l t2 m alpha = C :=
  have hα : alpha = 2 * π / 3 := by sorry
  have hm : m = 2 := by sorry
  sorry

end part_I_part_II_l238_238437


namespace C_must_be_2_l238_238308

-- Define the given digits and their sum conditions
variables (A B C D : ℤ)

-- The sum of known digits for the first number
def sum1_known_digits := 7 + 4 + 5 + 2

-- The sum of known digits for the second number
def sum2_known_digits := 3 + 2 + 6 + 5

-- The first number must be divisible by 3
def divisible_by_3 (n : ℤ) : Prop := n % 3 = 0

-- Conditions for the divisibility by 3 of both numbers
def conditions := divisible_by_3 (sum1_known_digits + A + B + D) ∧ 
                  divisible_by_3 (sum2_known_digits + A + B + C)

-- The statement of the theorem
theorem C_must_be_2 (A B D : ℤ) (h : conditions A B 2 D) : C = 2 :=
  sorry

end C_must_be_2_l238_238308


namespace amount_paid_out_l238_238510

theorem amount_paid_out 
  (amount : ℕ) 
  (h1 : amount % 50 = 0) 
  (h2 : ∃ (n : ℕ), n ≥ 15 ∧ amount = n * 5000 ∨ amount = n * 1000)
  (h3 : ∃ (n : ℕ), n ≥ 35 ∧ amount = n * 1000) : 
  amount = 29950 :=
by 
  sorry

end amount_paid_out_l238_238510


namespace max_value_f_l238_238893

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x - 2 * cos x

theorem max_value_f : ∃ (x : ℝ), f x = 4 :=
sorry

end max_value_f_l238_238893


namespace maximum_value_N_27_l238_238098

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end maximum_value_N_27_l238_238098


namespace number_of_purchasing_schemes_l238_238427

def total_cost (a : Nat) (b : Nat) : Nat := 8 * a + 10 * b

def valid_schemes : List (Nat × Nat) :=
  [(4, 4), (4, 5), (4, 6), (4, 7),
   (5, 4), (5, 5), (5, 6),
   (6, 4), (6, 5),
   (7, 4)]

theorem number_of_purchasing_schemes : valid_schemes.length = 9 := sorry

end number_of_purchasing_schemes_l238_238427


namespace percentage_increase_l238_238904

theorem percentage_increase (original_price new_price : ℝ) (h₀ : original_price = 300) (h₁ : new_price = 420) :
  ((new_price - original_price) / original_price) * 100 = 40 :=
by
  -- Insert the proof here
  sorry

end percentage_increase_l238_238904


namespace new_concentration_of_solution_l238_238963

theorem new_concentration_of_solution 
  (Q : ℚ) 
  (initial_concentration : ℚ := 0.4) 
  (new_concentration : ℚ := 0.25) 
  (replacement_fraction : ℚ := 1/3) 
  (new_solution_concentration : ℚ := 0.35) :
  (initial_concentration * (1 - replacement_fraction) + new_concentration * replacement_fraction)
  = new_solution_concentration := 
by 
  sorry

end new_concentration_of_solution_l238_238963


namespace coefficient_x4_l238_238317

-- Define the given expression
def expr := 2 * (x ^ 3 - 2 * x ^ 4 + x ^ 2) + 4 * (x ^ 2 + 3 * x ^ 4 - x ^ 3 + 2 * x ^ 5) - 6 * (2 + x - 5 * x ^ 4 + 2 * x ^ 3)

-- Define a statement that proves the coefficient of x^4 is 38
theorem coefficient_x4 : coeff (expr) 4 = 38 :=
by
  sorry

end coefficient_x4_l238_238317


namespace financing_amount_correct_l238_238398

-- Define the conditions
def monthly_payment : ℕ := 150
def years : ℕ := 5
def months_per_year : ℕ := 12

-- Define the total financed amount
def total_financed : ℕ := monthly_payment * years * months_per_year

-- The statement that we need to prove
theorem financing_amount_correct : total_financed = 9000 := 
by
  sorry

end financing_amount_correct_l238_238398


namespace tangent_line_parallel_range_of_m_l238_238381

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x

-- Prove that a = 1 given the properties of the derivative and tangent line
theorem tangent_line_parallel (a : ℝ) :
  ∀ (x : ℝ), Derivative (λ x => f x a) x = 0 ->
            (∀ (y : ℝ), Derivative (λ y => Real.exp y) x = -1/2) ->
            a = 1 :=
sorry

-- Prove the range of m such that f(x) + m = 2x - x^2 has exactly two distinct real roots in [1/2, 2]
theorem range_of_m (m : ℝ) :
  (∀ (x : ℝ), 1 / x - 3 + 2 * x = 0) ->
  (2 * x - 1) * (x - 1) / x ->
  (∀ (y : ℝ), y ∈ (Set.Icc (1/2 : ℝ) 2) -> f y 1 + m = 2 * y - y ^ 2) ->
  (Real.log 2 + 5 / 4 ≤ m ∧ m < 2) :=
sorry


end tangent_line_parallel_range_of_m_l238_238381


namespace pie_division_min_pieces_l238_238885

-- Define the problem as a Lean statement
theorem pie_division_min_pieces : ∃ n : ℕ, (∀ m ∈ {5, 7}, n % m = 0) ∧ n = 11 :=
by
  use 11
  split
  -- Prove for 5
  { intro m
    intro hm
    cases hm
    -- m = 5
    { exact Nat.mod_eq_zero_of_dvd (Nat.dvd_trans (Nat.dvd_refl 11) (Nat.dvd_of_mem_divisors hm)) }
    -- m = 7
    { exact Nat.mod_eq_zero_of_dvd (Nat.dvd_trans (Nat.dvd_refl 11) (Nat.dvd_of_mem_divisors hm)) }
    -- Impossible, there are only 5 and 7
    contradiction }
  -- Prove n = 11
  exact rfl

end pie_division_min_pieces_l238_238885


namespace cloth_total_l238_238527

-- Definitions of the conditions
variable (a₁ aₙ : ℕ)  -- a₁ is the first term, aₙ is the last term
variable (n : ℕ)  -- n is the total number of terms
variable (d : ℕ)  -- d is the common difference

-- Given conditions
def first_term := a₁ = 5
def last_term := aₙ = 1
def total_days := n = 30
def arithmetic_seq := d = (a₁ - aₙ) / (n - 1)

-- The sum of an arithmetic sequence
def sum_arithmetic_sequence := (n * (a₁ + aₙ)) / 2

-- The proof goal
theorem cloth_total (h1 : first_term) (h2 : last_term) (h3 : total_days) (h4 : arithmetic_seq):
  sum_arithmetic_sequence = 90 :=
by
  sorry

end cloth_total_l238_238527


namespace minimize_f_at_3_l238_238210

-- Define the quadratic function f(x) = 3x^2 - 18x + 7
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

-- The theorem stating that f(x) attains its minimum when x = 3
theorem minimize_f_at_3 : ∀ x : ℝ, f(x) ≥ f(3) := 
by 
  sorry

end minimize_f_at_3_l238_238210


namespace length_of_train_is_100_l238_238272

-- Conditions
def speed_kph : ℝ := 90
def time_to_cross_pole_s : ℝ := 4

-- Conversion factor from km/hr to m/s
def kph_to_mps (speed_kph : ℝ) : ℝ := (speed_kph * 1000) / 3600

-- Question (Length of the train)
def length_of_train_km (speed_kph : ℝ) (time_s : ℝ) : ℝ :=
  kph_to_mps(speed_kph) * time_s

-- Proof statement
theorem length_of_train_is_100 :
  length_of_train_km speed_kph time_to_cross_pole_s = 100 := by
  sorry

end length_of_train_is_100_l238_238272


namespace exp_decreasing_iff_a_in_interval_l238_238074

theorem exp_decreasing_iff_a_in_interval (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2 - a)^x > (2 - a)^y) ↔ 1 < a ∧ a < 2 :=
by 
  sorry

end exp_decreasing_iff_a_in_interval_l238_238074


namespace mario_hibiscus_l238_238488

def hibiscus_flowers (F : ℕ) : Prop :=
  let F2 := 2 * F
  let F3 := 4 * F2
  F + F2 + F3 = 22 → F = 2

theorem mario_hibiscus (F : ℕ) : hibiscus_flowers F :=
  sorry

end mario_hibiscus_l238_238488


namespace find_pairs_l238_238652

def path_exists (n k : ℕ) (M : ℕ → ℕ → ℤ) : Prop :=
∃ path : List (ℕ × ℕ), 
(path.head = (0, 0)) ∧ (path.last = (k - 1, k - 1)) ∧
∀ i (p p' : ℕ × ℕ), ((p, p') ∈ (List.zip path (List.tail path))) →
(p.1 = p'.1 ∨ p.2 = p'.2) ∧ ((path.map (λ (ij : ℕ × ℕ), M ij.1 ij.2)).sum % n = 0)

theorem find_pairs (n k : ℕ) (M : ℕ → ℕ → ℤ) :
  (∀ (M : ℕ → ℕ → ℤ), path_exists n k M) ↔ n ≤ k :=
sorry

-- Explanation:
-- path_exists defines the existence of a path from the left to the right edges of the array with required properties.
-- find_pairs states that there exists such a path for all k x k arrays if and only if n ≤ k.


end find_pairs_l238_238652


namespace probability_all_four_same_flips_l238_238923

theorem probability_all_four_same_flips : 
  let prob := ∑ n in (Set.Icc 1 (Top)), (1 / 2) ^ (4 * n) 
  in prob = 1 / 15 :=
by 
  -- Definitions and setup
  sorry

end probability_all_four_same_flips_l238_238923


namespace digits_concatenation_l238_238525

noncomputable def num_digits (n : ℕ) : ℕ :=
  (Real.log10 n).floor + 1

theorem digits_concatenation :
  let n1 := 2 ^ 2021
  let n2 := 5 ^ 2021
  num_digits n1 + num_digits n2 = 2022 := by
  sorry

end digits_concatenation_l238_238525


namespace brahmagupta_quadrilateral_diagonals_l238_238637

theorem brahmagupta_quadrilateral_diagonals (a b c d p q : ℝ) :
  p^2 + q^2 = a^2 + b^2 + c^2 + d^2 ∧ 2 * p * q = a^2 + c^2 - b^2 - d^2 :=
begin
  sorry
end

end brahmagupta_quadrilateral_diagonals_l238_238637


namespace num_correct_props_geometric_sequence_l238_238371

-- Define what it means to be a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Original Proposition P
def Prop_P (a : ℕ → ℝ) :=
  a 1 < a 2 ∧ a 2 < a 3 → ∀ n : ℕ, a n < a (n + 1)

-- Converse of Proposition P
def Conv_Prop_P (a : ℕ → ℝ) :=
  ( ∀ n : ℕ, a n < a (n + 1) ) → a 1 < a 2 ∧ a 2 < a 3

-- Inverse of Proposition P
def Inv_Prop_P (a : ℕ → ℝ) :=
  ¬(a 1 < a 2 ∧ a 2 < a 3) → ¬( ∀ n : ℕ, a n < a (n + 1) )

-- Contrapositive of Proposition P
def Contra_Prop_P (a : ℕ → ℝ) :=
  ¬( ∀ n : ℕ, a n < a (n + 1) ) → ¬(a 1 < a 2 ∧ a 2 < a 3)

-- Main theorem to be proved
theorem num_correct_props_geometric_sequence (a : ℕ → ℝ) :
  is_geometric_sequence a → 
  Prop_P a ∧ Conv_Prop_P a ∧ Inv_Prop_P a ∧ Contra_Prop_P a := by
  sorry

end num_correct_props_geometric_sequence_l238_238371


namespace midpoint_sum_l238_238794

-- Define the initial conditions
def midpoint_f (a b : ℕ) := (3 + a) / 2 = 5 ∧ (10 + b) / 2 = 7

-- Statement to prove
theorem midpoint_sum (a b : ℕ) (h : midpoint_f a b) : a + b = 11 :=
by
  cases h with h₁ h₂
  sorry

end midpoint_sum_l238_238794


namespace period_tan2x_cot2x_l238_238928

theorem period_tan2x_cot2x : 
  (∃ p, ∀ x, tan(2 * (x + p)) + cot(2 * (x + p)) = tan(2 * x) + cot(2 * x)) → p = π / 2 :=
sorry

end period_tan2x_cot2x_l238_238928


namespace triangle_equilateral_l238_238458

-- Define the conditions in the problem
variables {A B C D E F : Type} [triangle : triangle A B C]
variables (D_on_BC : lies_on D B C) (E_on_CA : lies_on E C A) (F_on_AB : lies_on F A B)
variables (AD_median : is_median A D) (BE_bisector : is_bisector B E) (CF_altitude : is_altitude C F)
variables (angleFDE_C : angle F D E = angle C) (angleDEF_A : angle D E F = angle A) (angleEFD_B : angle E F D = angle B)

-- The theorem to prove that ABC is equilateral
theorem triangle_equilateral (h_acute : acute_triangle A B C) :
  equilateral A B C :=
sorry

end triangle_equilateral_l238_238458


namespace class_teacher_age_l238_238948

theorem class_teacher_age (n_students : ℕ) (avg_age_students : ℝ) 
    (leaving_student_age : ℝ) (new_avg_with_teacher : ℝ) 
    (teacher_age : ℝ) (H1 : n_students = 45) (H2 : avg_age_students = 14) 
    (H3 : leaving_student_age = 15) (H4 : new_avg_with_teacher = 14.66)
    (H5 : teacher_age = 44.7) : 
    let total_age_students := n_students * avg_age_students in
    let remaining_students := n_students - 1 in
    let total_age_remaining := total_age_students - leaving_student_age in
    let total_age_with_teacher := total_age_remaining + teacher_age in
    total_age_with_teacher / n_students = new_avg_with_teacher :=
by
  rw [H1, H2, H3, H5]
  let total_age_students := 45 * 14
  let remaining_students := 45 - 1
  let total_age_remaining := total_age_students - 15
  let total_age_with_teacher := total_age_remaining + 44.7
  have : total_age_with_teacher / 45 = 14.66 := by sorry
  exact this

end class_teacher_age_l238_238948


namespace maximum_numbers_up_to_235_l238_238827

def max_remaining_numbers : ℕ := 118

theorem maximum_numbers_up_to_235 (numbers : set ℕ) (h₁ : ∀ n ∈ numbers, n ≤ 235)
  (h₂ : ∀ a b ∈ numbers, a ≠ b → ¬ (a - b).abs ∣ a) :
  numbers.card ≤ max_remaining_numbers :=
sorry

end maximum_numbers_up_to_235_l238_238827


namespace original_number_is_115_l238_238931

-- Define the original number N, the least number to be subtracted (given), and the divisor
variable (N : ℤ) (k : ℤ)

-- State the condition based on the problem's requirements
def least_number_condition := ∃ k : ℤ, N - 28 = 87 * k

-- State the proof problem: Given the condition, prove the original number
theorem original_number_is_115 (h : least_number_condition N) : N = 115 := 
by
  sorry

end original_number_is_115_l238_238931


namespace jane_rejected_percentage_l238_238783

theorem jane_rejected_percentage (P : ℕ) (John_rejected : ℤ) (Jane_inspected_rejected : ℤ) :
  John_rejected = 7 * P ∧
  Jane_inspected_rejected = 5 * P ∧
  (John_rejected + Jane_inspected_rejected) = 75 * P → 
  Jane_inspected_rejected = P  :=
by sorry

end jane_rejected_percentage_l238_238783


namespace fourth_vertex_of_square_l238_238914

def A : ℂ := 2 - 3 * Complex.I
def B : ℂ := 3 + 2 * Complex.I
def C : ℂ := -3 + 2 * Complex.I

theorem fourth_vertex_of_square : ∃ D : ℂ, 
  (D - B) = (B - A) * Complex.I ∧ 
  (D - C) = (C - A) * Complex.I ∧ 
  (D = -3 + 8 * Complex.I) :=
sorry

end fourth_vertex_of_square_l238_238914


namespace distribute_fruits_l238_238434

theorem distribute_fruits (n m k : ℕ) (h_n : n = 3) (h_m : m = 6) (h_k : k = 1) :
  ((3 ^ n) * (Finset.card ((Finset.Icc 0 m).subsetsOfCard 2).attach)) = 756 :=
by
  sorry

end distribute_fruits_l238_238434


namespace problem_1_problem_2_l238_238059

def f (x : ℝ) : ℝ := 9^x / (9^x + 3)

theorem problem_1 (a : ℝ) (h : 0 < a ∧ a < 1) : f(a) + f(1 - a) = 1 := 
sorry

theorem problem_2 : (∑ k in Finset.range 999 | k > 0, f(k / 1000)) = 999 / 2 :=
sorry

end problem_1_problem_2_l238_238059


namespace minimize_quadratic_l238_238196

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end minimize_quadratic_l238_238196


namespace ellipse_foci_distance_is_correct_l238_238915

-- Given points
def point1 := (1, 6)
def point2 := (4, -3)
def point3 := (11, 6)

-- Definition of the ellipse and calculation
def ellipse_focus_distance (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let c := ((p1.1 + p3.1) / 2, (p1.2 + p3.2) / 2) in  -- Midpoint as the center
  let major_axis := ( (p1.1 - p3.1)^2 + (p1.2 - p3.2)^2 )^.5 / 2 in  -- Semi-major axis
  let minor_axis := ( (p2.1 - p2.1)^2 + (2 * (p2.2 - c.2))^2 )^.5 / 2 in  -- Semi-minor axis
  2 * ((minor_axis^2 - major_axis^2 )^.5)  -- Distance between foci

-- Lean theorem statement to prove the problem
theorem ellipse_foci_distance_is_correct :
  ellipse_focus_distance point1 point2 point3 = 4 * real.sqrt 14 := by
  sorry

end ellipse_foci_distance_is_correct_l238_238915


namespace percentage_error_in_area_l238_238615

theorem percentage_error_in_area {s : ℝ} (H : s > 0) :
  let s' := 1.04 * s
      A := s^2
      A' := (1.04 * s)^2
      E := (A' - A)
  in (E / A) * 100 = 8.16 := 
by
  let s' := 1.04 * s
  let A := s^2
  let A' := (1.04 * s)^2
  let E := A' - A
  have h1 : E = 0.0816 * s^2 := sorry
  have h2 : E / A = 0.0816 := sorry
  rw[h1, h2]
  norm_num
  sorry

end percentage_error_in_area_l238_238615


namespace binomial_coeff_8_3_l238_238302

theorem binomial_coeff_8_3 : nat.choose 8 3 = 56 := by
  sorry

end binomial_coeff_8_3_l238_238302


namespace train_passes_man_in_21_seconds_l238_238227

-- Define the parameters
def train_length : ℝ := 350 -- meters
def train_speed : ℝ := 68   -- km/h
def man_speed : ℝ := 8      -- km/h

-- Convert speeds from km/h to m/s
def kmph_to_mps (speed : ℝ) : ℝ := speed * 1000 / 3600

-- Calculate relative speed
def relative_speed (train_speed man_speed : ℝ) : ℝ :=
  kmph_to_mps train_speed - kmph_to_mps man_speed

-- Calculate the time taken for the train to pass the man
def pass_time (distance speed : ℝ) : ℝ :=
  distance / speed

-- Main theorem statement
theorem train_passes_man_in_21_seconds :
  pass_time train_length (relative_speed train_speed man_speed) = 21 := by
  -- Insert proof here
  sorry

end train_passes_man_in_21_seconds_l238_238227


namespace multiply_expression_l238_238799

theorem multiply_expression (x : ℝ) : 
  (x^4 + 49 * x^2 + 2401) * (x^2 - 49) = x^6 - 117649 :=
by
  sorry

end multiply_expression_l238_238799


namespace concyclic_points_condition_l238_238691

-- Assume square ABCD, points S and P on AB and BC, respectively
noncomputable def concyclic_condition 
  (A B C D S P Q R : Type) 
  [Square ABCD S P Q R] 
  (AS : ℝ) 
  (CP : ℝ) : Prop :=
  (BS : ℝ) (BP : ℝ) → BS * BP = 2 * AS * CP

-- Theorem setup
theorem concyclic_points_condition 
  (A B C D S P Q R : Type)
  [Square ABCD S P Q R]
  (AS CP : ℝ) :
  ∃ BS BP : ℝ, BS * BP = 2 * AS * CP :=
sorry

end concyclic_points_condition_l238_238691


namespace complex_quadrant_l238_238512

theorem complex_quadrant (z : ℂ) (h : z * (2 - Complex.i) = 2 + Complex.i) :
  let conj_z := Complex.conj z in
  ∃ x y : ℝ, conj_z = x + y * Complex.i ∧ x > 0 ∧ y < 0 :=
by
  sorry

end complex_quadrant_l238_238512


namespace correct_calculation_A_l238_238559

theorem correct_calculation_A: (∃ x : ℝ, x^3 = -8 ∧ x = -2) :=
begin
  use -2,
  split,
  { -- Prove (-2)^3 = -8
    calc (-2) ^ 3 = (-2) * (-2) * (-2) : by refl
            ...  = 4 * (-2)          : by rw mul_neg_of_pos_of_neg; refl
            ...  = -8                : by norm_num,
  },
  { -- Prove the solution x = -2 is the same
    refl,
  }
end

end correct_calculation_A_l238_238559


namespace quadratic_trinomial_expression_c_l238_238221

def is_quadratic_trinomial (expr : Expr) : Prop :=
  expr = (5 - x - y^2)

theorem quadratic_trinomial_expression_c :
  is_quadratic_trinomial (5 - x - y^2) := 
sorry

end quadratic_trinomial_expression_c_l238_238221


namespace sufficient_but_not_necessary_l238_238951

theorem sufficient_but_not_necessary (x : ℝ) : (x > 1 → π^x > 1) ∧ (¬(π^x > 1 ↔ x > 1)) := 
by
  sorry

end sufficient_but_not_necessary_l238_238951


namespace max_remaining_numbers_l238_238843

theorem max_remaining_numbers : 
  ∃ s : Finset ℕ, s ⊆ (Finset.range 236) ∧ (∀ x y ∈ s, x ≠ y → ¬ (x - y).abs ∣ x) ∧ s.card = 118 := 
by
  sorry

end max_remaining_numbers_l238_238843


namespace hire_Zhang_Ying_l238_238584

-- Definition of conditions
def importance_ratios : (ℕ × ℕ × ℕ) := (6, 3, 1) -- The importance ratios

def Wang_Li_scores : (ℕ × ℕ × ℕ) := (14, 16, 18) -- Wang Li's scores
def Zhang_Ying_scores : (ℕ × ℕ × ℕ) := (18, 16, 12) -- Zhang Ying's scores

-- Calculations of weighted scores according to given ratios
def weighted_score (scores : (ℕ × ℕ × ℕ)) (ratios : (ℕ × ℕ × ℕ)) : ℕ :=
  scores.1 * ratios.1 + scores.2 * ratios.2 + scores.3 * ratios.3

-- The proof problem statement
theorem hire_Zhang_Ying :
  weighted_score Zhang_Ying_scores importance_ratios > weighted_score Wang_Li_scores importance_ratios :=
sorry

end hire_Zhang_Ying_l238_238584


namespace largest_angle_of_convex_pentagon_l238_238256

theorem largest_angle_of_convex_pentagon :
  ∀ (x : ℝ), (x + 2) + (2 * x + 3) + (3 * x + 4) + (4 * x + 5) + (5 * x + 6) = 540 →
  5 * (104 / 3 : ℝ) + 6 = 538 / 3 := 
by
  intro x
  intro h
  sorry

end largest_angle_of_convex_pentagon_l238_238256


namespace line_intersects_circle_l238_238369

theorem line_intersects_circle {a b : ℝ} (h : a^2 + b^2 > 1) :
  ∀ (x y : ℝ), x^2 + y^2 = 1 → ax + by = 1 → (∃ p : ℝ, p^2 = 1) := sorry

end line_intersects_circle_l238_238369


namespace max_remaining_numbers_l238_238806

def numbers (n : ℕ) : set ℕ := {k | 1 ≤ k ∧ k ≤ n}

def valid_subset (s : set ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≠ b → (a - b) ≠ 0 → ¬ (a - b) ∣ c

theorem max_remaining_numbers : ∃ s : set ℕ, s ⊆ numbers 235 ∧ valid_subset s ∧ card s = 118 := 
sorry

end max_remaining_numbers_l238_238806


namespace insulation_problem_l238_238435

noncomputable def annual_energy_cost (m : ℝ) (x : ℝ) : ℝ := (3 * m) / (4 * x + 5)

noncomputable def total_cost (m : ℝ) (x : ℝ) : ℝ := 40 * annual_energy_cost m x + 80 * x

theorem insulation_problem :
  ∃ (m : ℝ) (x_min : ℝ) (S_min : ℝ),
    (∃ P : ℝ, 90 = P ∧ P = annual_energy_cost 15 0) ∧
    m = 15 ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 8 → total_cost 15 x = 1800 / (4 * x + 5) + 8 * x) ∧
    x_min = 6.25 ∧
    S_min = 110 ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 8 → total_cost 15 x ≥ 110)
  := by clear

end insulation_problem_l238_238435


namespace field_division_l238_238261

theorem field_division
  (total_area : ℕ)
  (part_area : ℕ)
  (diff : ℕ → ℕ)
  (X : ℕ)
  (h_total : total_area = 900)
  (h_part : part_area = 405)
  (h_diff : diff (total_area - part_area - part_area) = (1 / 5 : ℚ) * X)
  : X = 450 := 
sorry

end field_division_l238_238261


namespace abc_minus_def_l238_238945

-- Defining the function f
def f (xyz : Nat) : Nat :=
  let x := xyz / 100
  let y := (xyz / 10) % 10
  let z := xyz % 10
  5^x * 2^y * 3^z

-- Main proof statement
theorem abc_minus_def (abc def : Nat) (h : f(abc) = 3 * f(def)) : abc - def = 1 := 
by
  sorry

end abc_minus_def_l238_238945


namespace natasha_money_l238_238496

theorem natasha_money :
  ∃ (N C : ℝ), 
    (N = 6 * C) ∧ -- Natasha has 6 times as much money as Cosima
    (N + 2*C + C) * (7/5) - (N + 2*C + C) = 36 ∧ -- profit condition
    N = 60 := 
by
  use [60, 10]
  split
  { rw [6 * 10], norm_num }
  split
  { have h1 : 7 / 5 = 1.4 := by norm_num,
    have h2 : 9 * 10 = 90 := by norm_num,
    rw [h1, h2],
    norm_num, }
  norm_num

end natasha_money_l238_238496


namespace coefficient_of_x3y2z2_in_polynomial_l238_238550

theorem coefficient_of_x3y2z2_in_polynomial : 
  (x y z : ℂ) → 
  coefficient (x^3 * y^2 * z^2) ((x + y)^5 * (z + (1/z))^4) = 40 := 
by sorry

end coefficient_of_x3y2z2_in_polynomial_l238_238550


namespace maximum_numbers_up_to_235_l238_238831

def max_remaining_numbers : ℕ := 118

theorem maximum_numbers_up_to_235 (numbers : set ℕ) (h₁ : ∀ n ∈ numbers, n ≤ 235)
  (h₂ : ∀ a b ∈ numbers, a ≠ b → ¬ (a - b).abs ∣ a) :
  numbers.card ≤ max_remaining_numbers :=
sorry

end maximum_numbers_up_to_235_l238_238831


namespace add_and_simplify_fractions_l238_238867

theorem add_and_simplify_fractions :
  (1 / 462) + (23 / 42) = 127 / 231 :=
by
  sorry

end add_and_simplify_fractions_l238_238867


namespace carl_marbles_l238_238994

theorem carl_marbles (initial : ℕ) (lost_frac : ℚ) (additional : ℕ) (gift : ℕ) (lost : ℕ)
  (initial = 12) 
  (lost_frac = 1 / 2)
  (additional = 10)
  (gift = 25)
  (lost = initial * lost_frac) :
  ((initial - lost) + additional + gift = 41) :=
sorry

end carl_marbles_l238_238994


namespace sum_series_eq_one_quarter_l238_238629

noncomputable def series_term (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))

theorem sum_series_eq_one_quarter : 
  (∑' n, series_term (n + 1)) = 1 / 4 :=
by
  sorry

end sum_series_eq_one_quarter_l238_238629


namespace percentage_error_in_square_area_l238_238614

theorem percentage_error_in_square_area:
  ∀ (S : ℝ), let S' := 1.04 * S in 
             let A  := S * S in 
             let A' := S' * S' in
             (A' - A) / A * 100 = 8.16 :=
by
  intro S
  let S' := 1.04 * S
  let A := S * S
  let A' := S' * S'
  have h1: A' = 1.04 * 1.04 * S * S := sorry
  have h2: (A' - A) / A * 100 = ((1.04 * 1.04 - 1) * S * S) / (S * S) * 100 := sorry
  have h3: ... := sorry
  exact sorry

end percentage_error_in_square_area_l238_238614


namespace side_length_of_equilateral_triangle_with_radius_3_l238_238899

noncomputable def equilateral_triangle_side_length (radius : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  circumference / 3

theorem side_length_of_equilateral_triangle_with_radius_3 :
  equilateral_triangle_side_length 3 ≈ 6.28 :=
by
  have h : 2 * Real.pi ≈ 6.28 := by norm_num1
  show equilateral_triangle_side_length 3 ≈ 6.28
  calc
    equilateral_triangle_side_length 3
        = (2 * Real.pi * 3) / 3          : rfl
    ... = 2 * Real.pi                  : by ring
    ... ≈ 6.28                        : h

end side_length_of_equilateral_triangle_with_radius_3_l238_238899


namespace jerry_total_logs_l238_238451

def logs_from_trees (p m w : Nat) : Nat :=
  80 * p + 60 * m + 100 * w

theorem jerry_total_logs :
  logs_from_trees 8 3 4 = 1220 :=
by
  -- Proof here
  sorry

end jerry_total_logs_l238_238451


namespace B2_is_midpoint_A2C2_l238_238239

-- Definitions for points and conditions
variables (A B C : Point)
variables (A1 B1 C1 : Point)
variables (A0 C0 : Point)
variables (A2 B2 C2 : Point)

-- Conditions:
-- 1. Triangle ABC
-- 2. A1, B1, C1 are the cevians intersecting at one point
-- 3. A0 and C0 are the midpoints of the sides BC and AB respectively
-- 4. B1C1, B1A1, and B1B intersect the line A0C0 at points C2, A2, and B2 respectively

-- Definition of midpoints
def midpoint (X Y : Point) : Point := (X + Y) / 2

-- Theorem: B2 is the midpoint of A2 and C2
theorem B2_is_midpoint_A2C2 
  (H1 : concurrent [A1, B1, C1])
  (H2 : midpoint B C = A0)
  (H3 : midpoint A B = C0)
  (H4 : intersect (line_through B1 C1) (line_through A0 C0) = C2)
  (H5 : intersect (line_through B1 A1) (line_through A0 C0) = A2)
  (H6 : intersect (line_through B1 B) (line_through A0 C0) = B2) :
  B2 = midpoint A2 C2 :=
sorry

end B2_is_midpoint_A2C2_l238_238239


namespace probability_qualified_and_not_qualified_l238_238348

noncomputable def total_ways : ℕ := nat.choose 5 2

noncomputable def favorable_ways : ℕ := nat.choose 3 1 * nat.choose 2 1

theorem probability_qualified_and_not_qualified :
  (favorable_ways : ℚ) / total_ways = 3 / 5 := by
  sorry

end probability_qualified_and_not_qualified_l238_238348


namespace union_of_sets_l238_238019

def A := { x : ℝ | -1 ≤ x ∧ x ≤ 5 }
def B := { x : ℝ | 3 < x ∧ x < 9 }

theorem union_of_sets : (A ∪ B) = { x : ℝ | -1 ≤ x ∧ x < 9 } :=
by
  sorry

end union_of_sets_l238_238019


namespace compare_neg_rational_decimal_l238_238299

theorem compare_neg_rational_decimal : 
  -3 / 4 > -0.8 := 
by 
  sorry

end compare_neg_rational_decimal_l238_238299


namespace circle_coloring_l238_238862

-- Define the problem statement
theorem circle_coloring (circles : ℕ) (non_overlapping : Prop) :
  (∀ (radii_table : set ℕ), ∃ (coloring : circles → ℕ), 
  (∀ (c1 c2 : circles), touching c1 c2 → coloring c1 ≠ coloring c2)) :=
by
  sorry

end circle_coloring_l238_238862


namespace triangle_AC_length_l238_238419

noncomputable theory

-- We define the basic structure and conditions given in the problem.
def Triangle (A B C : Type) := 
∃ tan_B : ℝ, tan_B = Real.sqrt 3 ∧
∃ AB : ℝ, AB = 3 ∧
∃ area : ℝ, area = 3 * Real.sqrt 3 / 2

-- Define the statement of the problem as a Lean 4 theorem.
theorem triangle_AC_length {A B C : Type} (h : Triangle A B C) : 
  ∃ AC : ℝ, AC = Real.sqrt 7 :=
by
  rcases h with ⟨tan_B, h1, AB, h2, area, h3⟩
  use Real.sqrt 7
  sorry

-- Justication for definitions used:
-- tan_B, AB, and area are used as per the conditions in a).
-- The statement with AC = Real.sqrt 7.

end triangle_AC_length_l238_238419


namespace max_remaining_numbers_l238_238815

theorem max_remaining_numbers : 
  ∃ (S ⊆ {n | 1 ≤ n ∧ n ≤ 235}), (∀ x ∈ S, ∀ y ∈ S, x ≠ y → ¬(x ∣ (y - x))) ∧ card S = 118 :=
by
  sorry

end max_remaining_numbers_l238_238815


namespace find_side_length_of_square_l238_238690

noncomputable def square_side_length (a : ℝ) : Prop :=
  let h₁ := ∀ (B D : ℝ), B = 1 ∧ D = 2
  let h₂ := 30 * Real.pi / 180 -- converting degrees to radians
  by sorry

theorem find_side_length_of_square (a : ℝ) (h₁: ∀ (B D : ℝ), B = 1 ∧ D = 2) (h₂ : (30 : ℝ) * Real.pi / 180) : 
  a = 2 * Real.sqrt 5 := 
by 
  sorry

end find_side_length_of_square_l238_238690


namespace maximum_value_N_27_l238_238102

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end maximum_value_N_27_l238_238102


namespace max_remaining_numbers_l238_238813

theorem max_remaining_numbers : 
  ∃ (S ⊆ {n | 1 ≤ n ∧ n ≤ 235}), (∀ x ∈ S, ∀ y ∈ S, x ≠ y → ¬(x ∣ (y - x))) ∧ card S = 118 :=
by
  sorry

end max_remaining_numbers_l238_238813


namespace units_digit_a_128_1_l238_238949

def a (i j : ℕ) : ℕ :=
  if i = 1 then j^j else a (i-1) j + a (i-1) (j+1)

theorem units_digit_a_128_1 :
  (a 128 1) % 10 = 4 := 
sorry

end units_digit_a_128_1_l238_238949


namespace max_remaining_numbers_l238_238839

theorem max_remaining_numbers : 
  ∃ s : Finset ℕ, s ⊆ (Finset.range 236) ∧ (∀ x y ∈ s, x ≠ y → ¬ (x - y).abs ∣ x) ∧ s.card = 118 := 
by
  sorry

end max_remaining_numbers_l238_238839


namespace area_of_octagon_l238_238020

open Real

def midpoint (x y : Point) : Point := (x + y) / 2

def circumcircle_radius : ℝ := 10
def perimeter_square : ℝ := 40
def side_square := perimeter_square / 4  -- 10
def area_octagon : ℝ := 200

-- Given conditions
variables
  (A B C D : Point)
  (circumcircle_center : Point)
  (circumcircle_radius : ℝ)
  (perimeter_square : ℝ)

-- Given conditions expressed
axiom h1 : dist circumcircle_center A = circumcircle_radius
axiom h2 : dist circumcircle_center B = circumcircle_radius
axiom h3 : dist circumcircle_center C = circumcircle_radius
axiom h4 : dist circumcircle_center D = circumcircle_radius
axiom h5 : perimeter_square = 40
axiom h6 : circumcircle_radius = 10

-- Define points A', B', C', D' on the circumcircle that are constructed by perpendicular bisectors.
def A' : Point := sorry  -- You'd need to define it properly
def B' : Point := sorry
def C' : Point := sorry
def D' : Point := sorry

-- Proof statement
theorem area_of_octagon (h1 : dist circumcircle_center A = circumcircle_radius)
                        (h2 : dist circumcircle_center B = circumcircle_radius)
                        (h3 : dist circumcircle_center C = circumcircle_radius)
                        (h4 : dist circumcircle_center D = circumcircle_radius)
                        (h5 : perimeter_square = 40)
                        (h6 : circumcircle_radius = 10) :
  area (polygon [A, A', B, B', C, C', D, D']) = area_octagon :=
sorry

end area_of_octagon_l238_238020


namespace min_f_on_neg_l238_238738

noncomputable def phi : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f (a b : ℝ) (phi g : ℝ → ℝ) : ℝ → ℝ := λ x, a * phi x + b * g x + 2

axiom odd_phi (x : ℝ) : phi (-x) = -phi x
axiom odd_g (x : ℝ) : g (-x) = -g x
axiom max_f_on_pos (a b : ℝ) (h : ∀ x > 0, f a b phi g x ≤ 5) : ∀ x > 0, f a b phi g x ≤ 5

theorem min_f_on_neg (a b : ℝ) (h₁ : ∀ x, phi (-x) = -phi x) (h₂ : ∀ x, g (-x) = -g x)
  (h₃ : ∀ x > 0, f a b phi g x ≤ 5) :
  ∃ x < 0, f a b phi g x = -1 := sorry

end min_f_on_neg_l238_238738


namespace am_gm_inequality_l238_238224

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem am_gm_inequality : (a / b) + (b / c) + (c / a) ≥ 3 := by
  sorry

end am_gm_inequality_l238_238224


namespace inscribed_sphere_radius_l238_238970

theorem inscribed_sphere_radius (a : ℝ)
  (ha : a = 1) : 
  ∃ r : ℝ, r = (Real.sqrt 6 - Real.sqrt 2) / 4 := 
by
  use (Real.sqrt 6 - Real.sqrt 2) / 4
  sorry

end inscribed_sphere_radius_l238_238970


namespace find_ordered_pair_l238_238659

theorem find_ordered_pair (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
    (real.sqrt (2 + real.sqrt (45 + 20 * real.sqrt 5)) = real.sqrt a + real.sqrt b) ↔ (a = 2 ∧ b = 5) :=
by
  sorry

end find_ordered_pair_l238_238659


namespace correct_operation_l238_238563

theorem correct_operation :
  ¬ ( (-3 : ℤ) * x ^ 2 * y ) ^ 3 = -9 * (x ^ 6) * y ^ 3 ∧
  ¬ (a + b) * (a + b) = (a ^ 2 + b ^ 2) ∧
  (4 * x ^ 3 * y ^ 2) * (x ^ 2 * y ^ 3) = (4 * x ^ 5 * y ^ 5) ∧
  ¬ ((-a) + b) * (a - b) = (a ^ 2 - b ^ 2) :=
by
  sorry

end correct_operation_l238_238563


namespace quadrilateral_SUVR_area_l238_238542

structure Triangle :=
(P Q R : Point)
(side1 : ℝ)
(side2 : ℝ)
(area : ℝ)

def segment_ratio (a b : Point) (c d : ℝ) : Prop := dist a b = c * d

noncomputable def area_of_quadrilateral (P Q R S T U V : Point) : ℝ :=
  let ΔPQR := Triangle.mk P Q R 60 15 180
  let PS := (1/(3:ℝ)) * ΔPQR.side1
  let PT := (1/(3:ℝ)) * ΔPQR.side2
  let ΔPST := 0.5 * PS * PT * (2/(5:ℝ))
  let ratio : ℝ := 0.2 -- calculated area scaling factor
  let ΔPVQ := 21.42
  let ΔPSTU := ΔPST * ratio
  ΔPQR.area - (ΔPST + ΔPVQ - ΔPSTU)

theorem quadrilateral_SUVR_area : area_of_quadrilateral P Q R S T U V = 141.44 := 
by
  sorry

end quadrilateral_SUVR_area_l238_238542


namespace simplify_expression_l238_238868

theorem simplify_expression (x : ℝ) : 3 * x + 5 * x ^ 2 + 2 - (9 - 4 * x - 5 * x ^ 2) = 10 * x ^ 2 + 7 * x - 7 :=
by
  sorry

end simplify_expression_l238_238868


namespace clay_boys_proof_l238_238782

variable (total_students : ℕ)
variable (total_boys : ℕ)
variable (total_girls : ℕ)
variable (jonas_students : ℕ)
variable (clay_students : ℕ)
variable (birch_students : ℕ)
variable (jonas_boys : ℕ)
variable (birch_girls : ℕ)

noncomputable def boys_from_clay (total_students total_boys total_girls jonas_students clay_students birch_students jonas_boys birch_girls : ℕ) : ℕ :=
  let birch_boys := birch_students - birch_girls
  let clay_boys := total_boys - (jonas_boys + birch_boys)
  clay_boys

theorem clay_boys_proof (h1 : total_students = 180) (h2 : total_boys = 94) 
    (h3 : total_girls = 86) (h4 : jonas_students = 60) 
    (h5 : clay_students = 80) (h6 : birch_students = 40) 
    (h7 : jonas_boys = 30) (h8 : birch_girls = 24) : 
  boys_from_clay total_students total_boys total_girls jonas_students clay_students birch_students jonas_boys birch_girls = 48 := 
by 
  simp [boys_from_clay] 
  sorry

end clay_boys_proof_l238_238782


namespace three_digit_numbers_count_l238_238731

theorem three_digit_numbers_count : 
  ∃ (count : ℕ), count = 3 ∧ 
  ∀ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
             (n / 100 = 9) ∧ 
             (∃ a b c, n = 100 * a + 10 * b + c ∧ a + b + c = 27) ∧ 
             (n % 2 = 0) → count = 3 :=
sorry

end three_digit_numbers_count_l238_238731


namespace rationalize_denominator_l238_238503

theorem rationalize_denominator :
  \sqrt{\frac{5}{2 + \sqrt{2}}} = \frac{\sqrt{10}}{2} :=
by sorry

end rationalize_denominator_l238_238503


namespace sum_of_angles_is_60_degrees_l238_238638

-- Definition of the geometrical setup
variables {point : Type*} [metric_space point] (A B C D E F : point)

-- Assume A, B, C, D form a parallelogram
def parallelogram (A B C D : point) : Prop :=
  dist A B = dist C D ∧ dist A D = dist B C

-- Assume equilateral triangles are constructed outward on AB and BC to points E and F respectively
def equilateral_triangle (X Y Z : point) : Prop :=
  dist X Y = dist Y Z ∧ dist Y Z = dist Z X

-- Coordinate for two outward equilateral triangles on AB and BC
def setup (A B C D E F : point) : Prop :=
  parallelogram A B C D ∧
  equilateral_triangle A B E ∧
  equilateral_triangle B C F

-- Angle measure between three points
def angle (A B C : point) : ℝ := sorry -- placeholder for angle measure definition

-- Theorem that the sum of angles CED and AFD is 60°
theorem sum_of_angles_is_60_degrees (A B C D E F : point) 
  (h : setup A B C D E F) :
  angle C E D + angle A F D = 60 := sorry

end sum_of_angles_is_60_degrees_l238_238638


namespace inverse_function_l238_238891

theorem inverse_function :
  (∃ f : ℝ → ℝ, ∀ x > 1, f (1 + log (x - 1)) = x) ∧
  (∃ g : ℝ → ℝ, ∀ y : ℝ, g y = exp (y - 1) + 1 ∧ ∀ x > 1, g (1 + log (x - 1)) = x) :=
by sorry

end inverse_function_l238_238891


namespace min_value_problem_l238_238670

noncomputable def minimum_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (collinear : ∃ k : ℝ, (a - 1, -1) = k * (-b - 1, 2)) : ℝ :=
\begin
  if collinear then sorry else 9
end

theorem min_value_problem :
  ∀ (a b : ℝ),
    a > 0 →
    b > 0 →
    ∃ k : ℝ, (a - 1, -1) = k * (-b - 1, 2) →
    minimum_value a b = 9 :=
begin
  intros a b ha hb hcol,
  unfold minimum_value,
  rw if_pos hcol,
  sorry
end

end min_value_problem_l238_238670


namespace eccentricity_range_of_hyperbola_l238_238528

noncomputable def hyperbola_eccentricity_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : set ℝ :=
  {e : ℝ | 1 < e ∧ e ≤ 3}

theorem eccentricity_range_of_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ e : ℝ, (e = (Real.sqrt (a^2 + b^2)) / a) → e ∈ hyperbola_eccentricity_range a b ha hb :=
sorry

end eccentricity_range_of_hyperbola_l238_238528


namespace max_remaining_numbers_l238_238845

theorem max_remaining_numbers : 
  ∃ s : Finset ℕ, s ⊆ (Finset.range 236) ∧ (∀ x y ∈ s, x ≠ y → ¬ (x - y).abs ∣ x) ∧ s.card = 118 := 
by
  sorry

end max_remaining_numbers_l238_238845


namespace num_valid_grids_eq_3_l238_238298

def valid_grid (M : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  let nums := Finset.range 9
  let values := Finset.image M (Finset.univ : Finset (Fin 3 × Fin 3))
  values = nums ∧
  (∀ i, (M i 0 + M i 2 - M i 1 = M 0 0 + M 0 2 - M 0 1)) ∧
  (∀ j, (M 0 j + M 2 j - M 1 j = M 0 0 + M 2 0 - M 1 0)) ∧
  (M 0 0 + M 1 1 - M 2 2 = M 0 0 + M 0 2 - M 0 1) ∧
  (M 0 2 + M 1 1 - M 2 0 = M 0 0 + M 0 2 - M 0 1)

theorem num_valid_grids_eq_3 : 
  (Finset.filter valid_grid (Finset.univ : Finset (Matrix (Fin 3) (Fin 3) ℕ))).card = 3 :=
sorry

end num_valid_grids_eq_3_l238_238298


namespace typing_time_l238_238919

theorem typing_time (typing_speed : ℕ) (words_per_page : ℕ) (number_of_pages : ℕ) (h1 : typing_speed = 90) (h2 : words_per_page = 450) (h3 : number_of_pages = 10) : 
  (words_per_page / typing_speed) * number_of_pages = 50 := 
by
  sorry

end typing_time_l238_238919


namespace heads_count_l238_238578

theorem heads_count (H T : ℕ) (h1 : H + T = 128) (h2 : H = T + 12) : H = 70 := by
  sorry

end heads_count_l238_238578


namespace simplify_expression_correct_l238_238052

def simplify_expression : ℚ :=
  (5^5 + 5^3) / (5^4 - 5^2)

theorem simplify_expression_correct : simplify_expression = 65 / 12 :=
  sorry

end simplify_expression_correct_l238_238052


namespace distance_between_foci_of_hyperbola_l238_238888

theorem distance_between_foci_of_hyperbola :
  ∀ (x y : ℝ), (x * y = 4) → 
  ∃ d : ℝ, d = 4 * real.sqrt 2 :=
by
  intros x y h
  use 4 * real.sqrt 2
  sorry

end distance_between_foci_of_hyperbola_l238_238888


namespace max_rabbits_with_long_ears_and_jumping_far_l238_238095

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l238_238095


namespace Mike_additional_money_needed_proof_l238_238026

-- Definitions of conditions
def phone_cost : ℝ := 1300
def smartwatch_cost : ℝ := 500
def phone_discount : ℝ := 0.10
def smartwatch_discount : ℝ := 0.15
def sales_tax : ℝ := 0.07
def mike_has_percentage : ℝ := 0.40

-- Definitions of intermediate calculations
def discounted_phone_cost : ℝ := phone_cost * (1 - phone_discount)
def discounted_smartwatch_cost : ℝ := smartwatch_cost * (1 - smartwatch_discount)
def total_cost_before_tax : ℝ := discounted_phone_cost + discounted_smartwatch_cost
def total_tax : ℝ := total_cost_before_tax * sales_tax
def total_cost_after_tax : ℝ := total_cost_before_tax + total_tax
def mike_has_amount : ℝ := total_cost_after_tax * mike_has_percentage
def additional_money_needed : ℝ := total_cost_after_tax - mike_has_amount

-- Theorem statement
theorem Mike_additional_money_needed_proof :
  additional_money_needed = 1023.99 :=
by sorry

end Mike_additional_money_needed_proof_l238_238026


namespace sin_product_identity_l238_238856

theorem sin_product_identity (θ : ℝ) (n : ℕ) :
  sin (n * θ) = 2^(n-1) * ∏ k in Finset.range n, sin (θ + k * π / n) :=
sorry

end sin_product_identity_l238_238856


namespace area_of_sector_correct_l238_238947

-- Define the problem parameters
def radius : ℝ := 15
def angle : ℝ := 42
def pi : ℝ := Real.pi

-- Define the formula for the area of a sector
def areaOfSector (r θ : ℝ) : ℝ :=
  (θ / 360) * pi * r^2

-- Prove that the area of the sector with the given parameters is approximately 82.4749 square meters
theorem area_of_sector_correct :
  areaOfSector radius angle ≈ 82.4749 :=
by
  sorry

end area_of_sector_correct_l238_238947


namespace white_balls_count_l238_238960

theorem white_balls_count (total_balls green_balls yellow_balls red_balls purple_balls : ℕ) (prob_neither_red_nor_purple : ℝ) : 
  total_balls = 60 → 
  green_balls = 18 → 
  yellow_balls = 5 → 
  red_balls = 6 → 
  purple_balls = 9 → 
  prob_neither_red_nor_purple = 0.75 → 
  ∃ (white_balls : ℕ), white_balls = 22 :=
by
  intros
  let neither_red_nor_purple := total_balls * prob_neither_red_nor_purple
  let known_balls := green_balls + yellow_balls
  let white_balls := neither_red_nor_purple - known_balls
  use white_balls
  sorry

end white_balls_count_l238_238960


namespace minimize_quadratic_l238_238191

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_l238_238191


namespace maximum_distinct_integer_roots_l238_238321

-- Definitions based on the conditions
def polynomial_game (P : polynomial ℤ) : Prop :=
  ∀ (coeffs : fin 100 → ℤ)
    (coeffs_nonzero : ∀ i, coeffs i ≠ 0),
    ∃ roots : list ℤ,
      ∀ root, root ∈ roots →
        polynomial.eval root P = 0

theorem maximum_distinct_integer_roots :
  ∀ (P : polynomial ℤ), polynomial_game P →
    ∃ max_roots : ℕ, max_roots = 2 :=
by 
  sorry

end maximum_distinct_integer_roots_l238_238321


namespace average_mpg_highway_l238_238281

variable (mpg_city : ℝ) (H mpg : ℝ) (gallons : ℝ) (max_distance : ℝ)

noncomputable def SUV_fuel_efficiency : Prop :=
  mpg_city  = 7.6 ∧
  gallons = 20 ∧
  max_distance = 244 ∧
  H * gallons = max_distance

theorem average_mpg_highway (h1 : mpg_city = 7.6) (h2 : gallons = 20) (h3 : max_distance = 244) :
  SUV_fuel_efficiency mpg_city H gallons max_distance → H = 12.2 :=
by
  intros h
  cases h
  sorry

end average_mpg_highway_l238_238281


namespace thermal_shirts_price_reduction_l238_238251

theorem thermal_shirts_price_reduction :
  ∃ x : ℝ, 
      (∀ sales profit : ℝ, sales = 20 → profit = 40 → 
       ∀ k : ℝ, sales' = sales + 2 * k → profit' = profit - k →
       (profit' * sales' = 1200 → k = 20)) := 
begin
  sorry
end

end thermal_shirts_price_reduction_l238_238251


namespace bus_weight_conversion_l238_238152

noncomputable def round_to_nearest (x : ℚ) : ℤ := Int.floor (x + 0.5)

theorem bus_weight_conversion (kg_to_pound : ℚ) (bus_weight_kg : ℚ) 
  (h : kg_to_pound = 0.4536) (h_bus : bus_weight_kg = 350) : 
  round_to_nearest (bus_weight_kg / kg_to_pound) = 772 := by
  sorry

end bus_weight_conversion_l238_238152


namespace division_multiplication_eval_l238_238322

theorem division_multiplication_eval : (18 / (5 + 2 - 3)) * 4 = 18 := 
by
  sorry

end division_multiplication_eval_l238_238322


namespace assembly_line_arrangements_l238_238982

-- Definitions for tasks
inductive Task
| AddAxles : Task
| AddWheels : Task
| InstallWindshield : Task
| InstallInstrumentPanel : Task
| InstallSteeringWheel : Task
| InstallInteriorSeating : Task

open Task

-- Condition that AddAxles must be done before AddWheels
def axle_before_wheels (seq : List Task) : Prop :=
  List.indexOf AddAxles seq < List.indexOf AddWheels seq

-- Question reduces to counting permutations under conditions
def valid_assemblies : List (List Task) :=
  List.permutations [AddAxles, AddWheels, InstallWindshield, InstallInstrumentPanel, InstallSteeringWheel, InstallInteriorSeating].filter axle_before_wheels

-- The number of valid ways to arrange the tasks
def number_of_valid_assemblies : Nat :=
  valid_assemblies.length

-- Proof Statement
theorem assembly_line_arrangements : number_of_valid_assemblies = 120 :=
by
  sorry

end assembly_line_arrangements_l238_238982


namespace maximum_value_N_27_l238_238100

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end maximum_value_N_27_l238_238100


namespace k_value_if_function_not_in_first_quadrant_l238_238753

theorem k_value_if_function_not_in_first_quadrant : 
  ∀ k : ℝ, (∀ x : ℝ, x > 0 → (k - 2) * x ^ (|k|) + k ≤ 0) → k = -1 :=
by
  sorry

end k_value_if_function_not_in_first_quadrant_l238_238753


namespace polar_to_rectangular_coords_l238_238757

-- Define the polar coordinates of the point M
def polar_coords : ℝ × ℝ := (2, π / 6)

-- Define the expected rectangular coordinates
def rectangular_coords : ℝ × ℝ := (√3, 1)

-- The theorem stating the problem
theorem polar_to_rectangular_coords :
  ∃ M : ℝ × ℝ, M = polar_coords →
  (M.fst * Real.cos M.snd, M.fst * Real.sin M.snd) = rectangular_coords :=
by
  sorry

end polar_to_rectangular_coords_l238_238757


namespace range_of_g_l238_238618

noncomputable def g (x : ℝ) : ℝ := (Real.cos x)^4 + (Real.sin x)^2

theorem range_of_g : Set.Icc (3 / 4) 1 = Set.range g :=
by
  sorry

end range_of_g_l238_238618


namespace limit_cosine_expression_l238_238625

def small_angle_approx_sin5x (x : ℝ) : Prop := abs (sin (5 * x) - 5 * x) < ε
def small_angle_approx_sin2x (x : ℝ) : Prop := abs (sin (2 * x) - 2 * x) < ε
def small_angle_approx_cos2x (x : ℝ) : Prop := abs ((1 - cos (2 * x)) - (2 * x^2) / 2) < ε
def cos_diff_identity (A B : ℝ) : Prop := abs (cos A - cos B + 2 * sin ((A + B) / 2) * sin ((A - B) / 2)) < ε

theorem limit_cosine_expression :
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs x < δ →
    small_angle_approx_sin5x x ∧
    small_angle_approx_sin2x x ∧
    small_angle_approx_cos2x x ∧
    cos_diff_identity (7 * x) (3 * x) →
    abs ((1 - cos (2 * x)) / (cos (7 * x) - cos (3 * x)) + 1/10) < ε) := 
sorry

end limit_cosine_expression_l238_238625


namespace jerry_total_logs_l238_238453

-- Given conditions
def pine_logs_per_tree := 80
def maple_logs_per_tree := 60
def walnut_logs_per_tree := 100

def pine_trees_cut := 8
def maple_trees_cut := 3
def walnut_trees_cut := 4

-- Formulate the problem
theorem jerry_total_logs : 
  pine_logs_per_tree * pine_trees_cut + 
  maple_logs_per_tree * maple_trees_cut + 
  walnut_logs_per_tree * walnut_trees_cut = 1220 := 
by 
  -- Placeholder for the actual proof
  sorry

end jerry_total_logs_l238_238453


namespace adiabatic_compression_work_l238_238709

noncomputable def adiabatic_work (p1 V1 V2 k : ℝ) (h₁ : k > 1) (h₂ : V1 > 0) (h₃ : V2 > 0) : ℝ :=
  (p1 * V1) / (k - 1) * (1 - (V1 / V2)^(k - 1))

theorem adiabatic_compression_work (p1 V1 V2 k W : ℝ) (h₁ : k > 1) (h₂ : V1 > 0) (h₃ : V2 > 0) :
  W = adiabatic_work p1 V1 V2 k h₁ h₂ h₃ :=
sorry

end adiabatic_compression_work_l238_238709


namespace geometric_sequence_arithmetic_progression_l238_238372

open Nat

/--
Given a geometric sequence \( \{a_n\} \) where \( a_1 = 1 \) and the sequence terms
\( 4a_1 \), \( 2a_2 \), \( a_3 \) form an arithmetic progression, prove that
the common ratio \( q = 2 \) and the sum of the first four terms \( S_4 = 15 \).
-/
theorem geometric_sequence_arithmetic_progression (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h₀ : a 1 = 1)
    (h₁ : ∀ n, S n = (1 - q^n) / (1 - q)) 
    (h₂ : ∀ k n, a (k + n) = a k * q ^ n) 
    (h₃ : 4 * a 1 + a 3 = 4 * a 2) :
  q = 2 ∧ S 4 = 15 := 
sorry

end geometric_sequence_arithmetic_progression_l238_238372


namespace ronald_laundry_frequency_l238_238859

variable (Tim_laundry_frequency Ronald_laundry_frequency : ℕ)

theorem ronald_laundry_frequency :
  (Tim_laundry_frequency = 9) →
  (18 % Ronald_laundry_frequency = 0) →
  (18 % Tim_laundry_frequency = 0) →
  (Ronald_laundry_frequency ≠ 1) →
  (Ronald_laundry_frequency ≠ 18) →
  (Ronald_laundry_frequency ≠ 9) →
  (Ronald_laundry_frequency = 3) :=
by
  intros hTim hRonaldMultiple hTimMultiple hNot1 hNot18 hNot9
  sorry

end ronald_laundry_frequency_l238_238859


namespace problem_l238_238696

open Complex

-- Definitions used in conditions:
def p (a : ℚ) := a = -1

def q (a : ℚ) := 
  let z : ℚ := (1 + complex.i) / (1 + a * complex.i)
  z.re = 0 ∧ z.im ≠ 0

-- The proposition to be proved:
theorem problem (a : ℚ) : p(a) ↔ q(a) :=
by
  sorry -- The proof goes here

end problem_l238_238696


namespace longest_side_of_triangle_l238_238602

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2).sqrt

theorem longest_side_of_triangle :
  let A := (2, 2)
  let B := (5, 6)
  let C := (6, 2)
  let dAB := distance A B
  let dAC := distance A C
  let dBC := distance B C
  max (max dAB dAC) dBC = 5 :=
by {
  let A := (2, 2)
  let B := (5, 6)
  let C := (6, 2)
  let dAB := distance A B
  let dAC := distance A C
  let dBC := distance B C
  sorry
}

end longest_side_of_triangle_l238_238602


namespace twelve_tone_equal_temperament_ratio_l238_238565

theorem twelve_tone_equal_temperament_ratio (a : ℕ → ℝ) (n : ℕ)
  (h1 : ∃ q : ℝ, q > 0 ∧ (∀ k : ℕ, a (k + 1) = a k * q))
  (h2 : a 13 = 2 * a 1) :
  a 8 / a 2 = real.sqrt 2 :=
by
  sorry

end twelve_tone_equal_temperament_ratio_l238_238565


namespace polynomial_abs_value_at_neg_one_l238_238477

theorem polynomial_abs_value_at_neg_one:
  ∃ g : Polynomial ℝ, 
  (∀ x ∈ ({0, 1, 2, 4, 5, 6} : Set ℝ), |g.eval x| = 15) → 
  |g.eval (-1)| = 75 :=
by
  sorry

end polynomial_abs_value_at_neg_one_l238_238477


namespace simplify_fraction_l238_238043

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l238_238043


namespace minimize_f_l238_238186

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l238_238186


namespace jason_current_money_l238_238456

/-- Definition of initial amounts and earnings -/
def fred_initial : ℕ := 49
def jason_initial : ℕ := 3
def fred_current : ℕ := 112
def jason_earned : ℕ := 60

/-- The main theorem -/
theorem jason_current_money : jason_initial + jason_earned = 63 := 
by
  -- proof omitted for this example
  sorry

end jason_current_money_l238_238456


namespace linear_function_expression_l238_238368

def linear_function_through_point_and_parallel (f : ℝ → ℝ) : Prop :=
  (f 0 = 5) ∧ (∀ x, f x = x + 5)

theorem linear_function_expression :
  ∃ f : ℝ → ℝ, (f 0 = 5) ∧ (∀ x, f x = x + 5) :=
sorry

end linear_function_expression_l238_238368


namespace even_factors_of_n_l238_238792

theorem even_factors_of_n {n : ℕ} (h : n = 2^5 * 3^3 * 7) : 
  (∃ k : ℕ, (k > 0) ∧ (k <= 40)) :=
begin
  -- Proof will be added here later
  sorry,
end

end even_factors_of_n_l238_238792


namespace pyramid_volume_l238_238306

-- Define the length of the sides
def side_length : ℝ := 2

-- Define the height of the pyramid
def height : ℝ := Real.sqrt 2

-- Define the base area of the pyramid
def base_area : ℝ := side_length * side_length

-- The volume of a pyramid formula
-- Volume = 1/3 * Base Area * Height
def volume_pyramid (base_area height : ℝ) : ℝ := 
  (1 / 3) * base_area * height

-- The theorem we need to prove
theorem pyramid_volume : volume_pyramid base_area height = (4 * Real.sqrt 2) / 3 :=
by
  sorry

end pyramid_volume_l238_238306


namespace maximum_numbers_no_divisible_difference_l238_238823

theorem maximum_numbers_no_divisible_difference :
  ∃ (S : Finset ℕ), S ⊆ Finset.range 236 ∧ 
  (∀ a b c ∈ S, a ≠ b → b ≠ c → c ≠ a → ¬ (a - b = 0) ∨ ¬ (c ∣ (a - b))) ∧ S.card ≤ 118 :=
by
  sorry

end maximum_numbers_no_divisible_difference_l238_238823


namespace integer_roots_b_values_l238_238329

theorem integer_roots_b_values (b : ℤ) :
  (∃ (x : ℤ), x^4 + 4*x^3 + 4*x^2 + b*x + 12 = 0) ↔ b ∈ { -38, -21, -2, 10, 13, 34 } :=
by
  sorry

end integer_roots_b_values_l238_238329


namespace tangent_lengths_equal_trajectory_l238_238415

theorem tangent_lengths_equal_trajectory :
  (∀ x y : ℝ, ((x + 1)^2 + (y + 1)^2 - 4 = (x - 3)^2 + (y - 2)^2 - 1) ↔ (4 * x + 3 * y = -1)) :=
by
  intros x y
  constructor
  sorry

end tangent_lengths_equal_trajectory_l238_238415


namespace minimize_f_l238_238184

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l238_238184


namespace simplify_expression_l238_238505

theorem simplify_expression (a : ℝ) (h : a = -2) : 
  (1 - a / (a + 1)) / (1 / (1 - a ^ 2)) = 1 / 3 :=
by
  subst h
  sorry

end simplify_expression_l238_238505


namespace baron_munchausen_claim_is_false_l238_238623

theorem baron_munchausen_claim_is_false (P : Type) [polygon P] (O : point_inside P) : 
  ¬(∀ l : line, passes_through O l → divides_into_three_polygons P l) := 
sorry

end baron_munchausen_claim_is_false_l238_238623


namespace increasing_range_of_a_l238_238686

noncomputable def f (x : ℝ) (a : ℝ) := 
  if x ≤ 1 then -x^2 + 4*a*x 
  else (2*a + 3)*x - 4*a + 5

theorem increasing_range_of_a :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a ≤ f x₂ a) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
sorry

end increasing_range_of_a_l238_238686


namespace sum_8_smallest_n_T_div_5_eq_188_l238_238337

/-- Definition of T_n as given in the conditions. -/
def T (n : ℕ) : ℕ := (n - 1) * n * (n + 1) * (3 * n + 2) / 24

/-- The sum of the 8 smallest values of n such that T_n is divisible by 5 equals 188. -/
theorem sum_8_smallest_n_T_div_5_eq_188 : ∑ i in (Finset.filter (λ n => T n % 5 = 0) (Finset.range (50))) (some_inner_function i subset) : ℕ := 188 :=
sorry

end sum_8_smallest_n_T_div_5_eq_188_l238_238337


namespace max_rabbits_l238_238114

theorem max_rabbits (N : ℕ) (h1 : ∀ k, k = N → k = 27 → true)
    (h2 : ∀ n_l : ℕ, n_l = 13 → n_l <= N)
    (h3 : ∀ n_j : ℕ, n_j = 17 → n_j <= N)
    (h4 : ∀ n_both : ℕ, n_both >= 3 → true) :
  N <= 27 :=
begin
  sorry
end

end max_rabbits_l238_238114


namespace factors_of_72_multiples_of_6_l238_238732

theorem factors_of_72_multiples_of_6 :
  {d : ℕ | d ∣ 72 ∧ d ≥ 1 ∧ ∃ k : ℕ, d = 6 * k}.card = 6 := by
sorry

end factors_of_72_multiples_of_6_l238_238732


namespace repeating_decimal_product_l238_238333

theorem repeating_decimal_product (x : ℚ) (h : x = 4 / 9) : x * 9 = 4 := 
by
  sorry

end repeating_decimal_product_l238_238333


namespace max_rabbits_with_traits_l238_238119

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l238_238119


namespace amber_josh_departure_time_l238_238607

def latest_departure_time (flight_time : ℕ) (check_in_time : ℕ) (drive_time : ℕ) (parking_time : ℕ) :=
  flight_time - check_in_time - drive_time - parking_time

theorem amber_josh_departure_time :
  latest_departure_time 20 2 (45 / 60) (15 / 60) = 17 :=
by
  -- Placeholder for actual proof
  sorry

end amber_josh_departure_time_l238_238607


namespace grandma_age_l238_238397

theorem grandma_age  :
  let months_to_years := 50 / 12 in
  let remaining_months := 50 % 12 in
  let age_in_years := 60 + months_to_years in
  let total_days := (remaining_months * 30) + (40 * 7) + 30 in
  let additional_years := total_days / 365 in
  (age_in_years + additional_years) = 65 := 
begin
  sorry
end

end grandma_age_l238_238397


namespace equation_of_ellipse_dp_fixed_point_l238_238702

-- Definition of the Ellipse
def ellipse (x y : ℝ) : Prop :=
  (1 / 4) * x^2 + (1 / 9) * y^2 = 1

-- Conditions
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (real.sqrt 3, -3/2)

-- Fixed Point Determination
def fixed_point := (0, 9 * real.sqrt 5 / 5)

-- Proof Statement
theorem equation_of_ellipse (x y : ℝ) :
  (0, 0) = (0, 0) ∧
  A = (-2, 0) ∧
  B = (real.sqrt 3, -3 / 2) →
  ellipse x y ↔ (1 / 4) * x^2 + (1 / 9) * y^2 = 1 :=
by sorry

theorem dp_fixed_point (x y : ℝ) :
  (0, 0) = (0, 0) ∧
  A = (-2, 0) ∧
  B = (real.sqrt 3, -3 / 2) →
  ∀ (F : ℝ × ℝ) (C D P : ℝ × ℝ),
    (∃ k : ℝ, P = (x, 9 * real.sqrt 5 / 5) →
    (y - 9 * real.sqrt 5 / 5 = k * (x - fst (0, 0)))) ↔
    P = (0, 7 * real.sqrt 5 / 5) :=
by sorry

end equation_of_ellipse_dp_fixed_point_l238_238702


namespace sum_of_coefficients_l238_238344

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (∀ x : ℝ, (x^3 - 1) * (x + 1)^7 = a_0 + a_1 * (x + 3) + 
           a_2 * (x + 3)^2 + a_3 * (x + 3)^3 + a_4 * (x + 3)^4 + 
           a_5 * (x + 3)^5 + a_6 * (x + 3)^6 + a_7 * (x + 3)^7 + 
           a_8 * (x + 3)^8 + a_9 * (x + 3)^9 + a_10 * (x + 3)^10) →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 9 := 
by
  -- proof steps skipped
  sorry

end sum_of_coefficients_l238_238344


namespace sqrt_meaningful_range_l238_238755

theorem sqrt_meaningful_range (x : ℝ) : 
  (x + 4) ≥ 0 ↔ x ≥ -4 :=
by sorry

end sqrt_meaningful_range_l238_238755


namespace max_distance_l238_238393

variables (a b c : ℝ^3)

-- Conditions
axiom unit_vectors : ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥c∥ = 1
axiom given_condition : ∥a - c∥^2 + ∥b - c∥^2 = 3

-- Proof statement
theorem max_distance : ∥a - b∥ ≤ (sqrt 15) / 2 :=
by
  sorry

end max_distance_l238_238393


namespace handshake_problem_l238_238952

noncomputable def number_of_handshakes (n : ℕ) : ℕ :=
  n.choose 2

theorem handshake_problem : number_of_handshakes 25 = 300 := 
  by
  sorry

end handshake_problem_l238_238952


namespace gary_initial_stickers_l238_238668

theorem gary_initial_stickers (S : ℕ) :
  (2 / 3 * S) * (3 / 4) = 36 → S = 72 := 
by
  intro h
  have : (1 / 2) * S = 36,
  { rw [mul_assoc, (mul_comm (3 / 4) (2 / 3)), ← mul_assoc, mul_assoc (2 / 3) (3 / 4) _, mul_inv_cancel', one_mul] at h,
    exact h, 
    norm_num },
  field_simp at this,
  linarith

-- Skip the proof for now
sorry

end gary_initial_stickers_l238_238668


namespace minimize_quadratic_function_l238_238181

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end minimize_quadratic_function_l238_238181


namespace negation_correct_l238_238934

theorem negation_correct (x : ℝ) : -(3 * x - 2) = -3 * x + 2 := 
by sorry

end negation_correct_l238_238934


namespace simplify_expression_l238_238627

variable (a : ℝ)

theorem simplify_expression : 
  (a^2 / (a^(1/2) * a^(2/3))) = a^(5/6) :=
by
  sorry

end simplify_expression_l238_238627


namespace integral_equation_of_diff_eq_l238_238665

theorem integral_equation_of_diff_eq :
  ∀ (φ : ℝ → ℝ) (y : ℝ → ℝ),
  (∀ x, y 0 = 1 ∧ (y' 0 = 0) ∧ (y'' + x * y' + y = 0))
  →
  (∀ x, φ x = -1 - ∫ t in 0..x, (2 * x - t) * φ t ∂t) :=
by sorry

end integral_equation_of_diff_eq_l238_238665


namespace general_formula_sum_sequence_min_value_S_l238_238241

variable {α : Type} [LinearOrderedField α]

def a1 : α := -11
def d : α := 2
def a (n : ℕ) : α := 2 * n - 13
def S (n : ℕ) : α := n^2 - 12 * n

theorem general_formula (n : ℕ) : a n = 2 * n - 13 :=
by sorry

theorem sum_sequence (n : ℕ) : S n = n^2 - 12 * n :=
by sorry

theorem min_value_S : ∃ n, S n = -36 :=
by 
  use 6
  show S 6 = -36
  sorry

end general_formula_sum_sequence_min_value_S_l238_238241


namespace max_rabbits_with_traits_l238_238121

open Set

theorem max_rabbits_with_traits (N : ℕ) (long_ears jump_far : ℕ → Prop)
  (total : ∀ x, long_ears x → jump_far x → x < N)
  (h1 : ∀ x, long_ears x → x < 13)
  (h2 : ∀ x, jump_far x → x < 17)
  (h3 : ∃ x, long_ears x ∧ jump_far x) :
  N ≤ 27 :=
by
  -- Adding the conditions as hypotheses
  sorry

end max_rabbits_with_traits_l238_238121


namespace find_initial_amount_l238_238656

-- defining conditions
def compound_interest (A P : ℝ) (r : ℝ) (n t : ℕ) : ℝ :=
  A - P

-- main theorem to prove the principal amount
theorem find_initial_amount 
  (A P : ℝ) (r : ℝ)
  (n t : ℕ)
  (h_P : A = P * (1 + r / n)^t)
  (compound_interest_eq : A - P = 1785.98)
  (r_eq : r = 0.20)
  (n_eq : n = 1)
  (t_eq : t = 5) :
  P = 1200 :=
by
  sorry

end find_initial_amount_l238_238656


namespace intersection_of_torus_and_plane_is_two_circles_l238_238061

noncomputable theory

def torus (a b : ℝ) (h : a > b) := {p : ℝ × ℝ × ℝ | let x := p.1, y := p.2, z := p.3 in
  ((Real.sqrt (x^2 + y^2)) - a)^2 + z^2 = b^2}

def plane_through_origin := {p : ℝ × ℝ × ℝ | p.3 * Real.sqrt ((p.1^2 + p.2^2) / (a^2 - b^2)) = p.3}

theorem intersection_of_torus_and_plane_is_two_circles (a b : ℝ) (h : a > b)
  (T : set (ℝ × ℝ × ℝ)) (P : set (ℝ × ℝ × ℝ)) (H₁ : T = torus a b h) (H₂ : P = plane_through_origin) :
  ∃ c₁ c₂ : set (ℝ × ℝ × ℝ), P ∩ T = c₁ ∪ c₂ ∧ is_circle c₁ ∧ is_circle c₂ :=
sorry

end intersection_of_torus_and_plane_is_two_circles_l238_238061


namespace number_of_distinct_triangles_l238_238688

-- Define the points in the 'T' shape
def points : Finset (Nat × Nat) := {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (2, 1), (2, 2)}

-- Define the function to count number of combinations
def combinations (n k : Nat) : Nat := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the property of collinear points
def collinear_horizontal : Finset (Finset (Nat × Nat)) := (Finset.powerset points).filter (λ s, s.card = 3 ∧ (∀ (x y ∈ s), x.2 = y.2))
def collinear_vertical : Finset (Finset (Nat × Nat)) := (Finset.powerset points).filter (λ s, s.card = 3 ∧ (∀ (x y ∈ s), x.1 = y.1))

-- Define the theorem to prove the number of distinct triangles
theorem number_of_distinct_triangles : ∃ n, n = (combinations 7 3) - (combinations 5 3 + combinations 3 3) ∧ n = 24 :=
by
  have h1 : combinations 7 3 = 35 := by sorry
  have h2 : combinations 5 3 = 10 := by sorry
  have h3 : combinations 3 3 = 1 := by sorry
  exists 24
  have h4 : 24 = 35 - (10 + 1) := by sorry
  exact ⟨h1, h2, h3, h4⟩

end number_of_distinct_triangles_l238_238688


namespace avg_transformation_l238_238689

theorem avg_transformation
  (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 2) :
  ((3 * x₁ + 1) + (3 * x₂ + 1) + (3 * x₃ + 1) + (3 * x₄ + 1) + (3 * x₅ + 1)) / 5 = 7 :=
by
  sorry

end avg_transformation_l238_238689


namespace sequence_sum_floor_value_l238_238004

-- Define the sequence using recurrence relation
def a : ℕ → ℝ
| 0     := 1/2
| (n+1) := a n ^ 2 + 3 * a n + 1

-- Define the floor function with the greatest integer condition
def greatest_integer_le (x : ℝ) : ℤ := int.floor x

theorem sequence_sum_floor_value :
  greatest_integer_le (∑ k in Finset.range 2017, a (k + 1) / (a (k + 1) + 2)) = 2015 :=
by sorry

end sequence_sum_floor_value_l238_238004


namespace max_remaining_numbers_l238_238833

/-- 
The board initially has numbers 1, 2, 3, ..., 235.
Among the remaining numbers, no number is divisible by the difference of any two others.
Prove that the maximum number of numbers that could remain on the board is 118.
-/
theorem max_remaining_numbers : 
  ∃ S : set ℕ, (∀ a ∈ S, 1 ≤ a ∧ a ≤ 235) ∧ (∀ a b ∈ S, a ≠ b → ¬ ∃ d, d ∣ (a - b)) ∧ 
  ∃ T : set ℕ, S ⊆ T ∧ T ⊆ finset.range 236 ∧ T.card = 118 := 
sorry

end max_remaining_numbers_l238_238833


namespace min_value_at_3_l238_238205

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end min_value_at_3_l238_238205


namespace no_infinite_subdivision_exists_l238_238593

theorem no_infinite_subdivision_exists : ¬ ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (∀ n : ℕ,
    ∃ (ai bi : ℝ), ai > bi ∧ bi > 0 ∧ ai * bi = a * b ∧
    (ai / bi = a / b ∨ bi / ai = a / b)) :=
sorry

end no_infinite_subdivision_exists_l238_238593


namespace minimize_quadratic_l238_238190

theorem minimize_quadratic : 
  ∃ x : ℝ, (∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_l238_238190


namespace range_of_slope_k_constant_AM_AN_find_equation_of_line_l238_238351

open Real

noncomputable def circleC (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 1

def line_through (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  B.2 = k * (B.1 - A.1) + A.2

def intersects (l C : ℝ × ℝ → Prop) (P Q : ℝ × ℝ) : Prop :=
  l (P.1, P.2) ∧ C (P.1, P.2) ∧ l (Q.1, Q.2) ∧ C (Q.1, Q.2)

theorem range_of_slope_k (A : ℝ × ℝ) (k : ℝ) (l : ℝ × ℝ → Prop) :
  (∃ (M N : ℝ × ℝ), intersects l circleC M N) 
  → (4 - sqrt 7) / 3 < k ∧ k < (4 + sqrt 7) / 3 := 
sorry

theorem constant_AM_AN (A M N : ℝ × ℝ) (k x1 x2 y1 y2 : ℝ) (l : ℝ × ℝ → Prop) : 
  M = (x1, y1) ∧ N = (x2, y2) 
  → (x1 + x2) = (4 * (k + 1)) / (k^2 + 1)
  → (x1 * x2) = 7 / (k^2 + 1)
  → \Vec.dot ((x1, y1 - 1), (x2, y2 - 1)) = 7 := 
sorry

theorem find_equation_of_line (O A M N l : ℝ × ℝ → Prop) (k : ℝ) :
  (∃ (M N : ℝ × ℝ), intersects l circleC M N) 
  → \Vec.dot ((O.1, O.2), (M.1, M.2)) = 12 
  → k = 1 
  → line_through A (0, 1) 1 :=
sorry

end range_of_slope_k_constant_AM_AN_find_equation_of_line_l238_238351


namespace difference_in_perimeters_of_rectangles_l238_238071

theorem difference_in_perimeters_of_rectangles 
  (l h : ℝ) (hl : l ≥ 0) (hh : h ≥ 0) :
  let length_outer := 7
  let height_outer := 5
  let perimeter_outer := 2 * (length_outer + height_outer)
  let perimeter_inner := 2 * (l + h)
  let difference := perimeter_outer - perimeter_inner
  difference = 24 :=
by
  let length_outer := 7
  let height_outer := 5
  let perimeter_outer := 2 * (length_outer + height_outer)
  let perimeter_inner := 2 * (l + h)
  let difference := perimeter_outer - perimeter_inner
  sorry

end difference_in_perimeters_of_rectangles_l238_238071


namespace tangent_line_correct_l238_238880

-- Defining the function y = x / (2x - 1)
def func (x : ℝ) : ℝ := x / (2 * x - 1)

-- Point at which the tangent line is considered
def point_x : ℝ := 1
def point_y : ℝ := 1

-- The slope of the tangent line to y = x / (2x - 1) at x = 1
def derivative_at_x (x : ℝ) : ℝ := -1 / ((2 * x - 1) ^ 2)
def slope_at_point : ℝ := derivative_at_x 1

-- The equation of the tangent line in point-slope form
def tangent_line_eq (x y : ℝ) : Prop := y - point_y = slope_at_point * (x - point_x)

-- The Lean statement to prove
theorem tangent_line_correct (x y : ℝ) (h : func point_x = point_y) :
  tangent_line_eq x y ↔ x + y - 2 = 0 :=
by
  sorry

end tangent_line_correct_l238_238880


namespace circle_square_area_difference_l238_238252

def radius := 3
def circle_area := real.pi * radius^2
def square_side := 2 * radius
def square_area := square_side^2
def area_difference := (square_area - circle_area : ℝ)

theorem circle_square_area_difference :
  area_difference = 36 - 9 * real.pi :=
by
  sorry

end circle_square_area_difference_l238_238252


namespace limit_at_a_l238_238235

noncomputable def limit_expression (x a : ℝ) : ℝ :=
  (2 - x / a) ^ Real.tan (π * x / (2 * a))

theorem limit_at_a (a : ℝ) (h : a ≠ 0) :
  filter.tendsto (λ x, limit_expression x a) (nhds a) (nhds (Real.exp (2 / π))) :=
sorry

end limit_at_a_l238_238235


namespace triplet_E_does_not_sum_to_zero_l238_238564

def triplet_A : (ℚ × ℚ × ℚ) := (1/3, 2/3, -1)
def triplet_B : (ℚ × ℚ × ℚ) := (1.5, -1.5, 0)
def triplet_C : (ℚ × ℚ × ℚ) := (0.2, -0.7, 0.5)
def triplet_D : (ℚ × ℚ × ℚ) := (-1.2, 1.1, 0.1)
def triplet_E : (ℚ × ℚ × ℚ) := (4/5, 2/5, -7/5)

theorem triplet_E_does_not_sum_to_zero :
  (triplet_E.1 + triplet_E.2 + triplet_E.3 ≠ 0) :=
by 
  sorry

end triplet_E_does_not_sum_to_zero_l238_238564


namespace range_dot_product_l238_238394

noncomputable theory
open Real

def a (x : ℝ) : ℝ × ℝ := (sin x + cos x, 1)
def b (x : ℝ) : ℝ × ℝ := (1, sin x * cos x)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem range_dot_product :
  ∀ x ∈ Icc 0 (π / 2), 1 ≤ dot_product (a x) (b x) ∧ dot_product (a x) (b x) ≤ sqrt 2 + 1 / 2 :=
by
  sorry

end range_dot_product_l238_238394


namespace correct_conclusions_l238_238711

-- Define the various conditions given
def condition1 : Prop := ∃ x : ℝ, y : ℝ, y = (x-1)/(2*x+1) ∧ (x, y) = (-1/2, -1/2)

def condition2 : Prop := ∀ (x : ℝ) (k : ℝ), (0 < x ∧ x < 1) → (x - 1/x + k ≠ 0) → k ≥ 2

def condition3 (a b : ℝ) : Prop := (2*a - 3*b + 1 < 0) → 3*b - 2*a > 1

def condition4 : Prop := ∀ φ : ℝ, (φ > 0) → (∃ k : ℤ, φ = -1/2*k*π - 5*π/12) → φ = π/12 

-- The main theorem statement
theorem correct_conclusions : 
  ¬ condition1 ∧ 
  ¬ condition2 ∧ 
  (∀ a b : ℝ, condition3 a b) ∧ 
  condition4 :=
sorry

end correct_conclusions_l238_238711


namespace find_length_of_AC_in_triangle_ABC_l238_238444

noncomputable def length_AC_in_triangle_ABC
  (AB BC : ℝ) (angle_B : ℝ) (h_AB : AB = 1) (h_BC : BC = 2) (h_angle_B : angle_B = Real.pi / 3) :
  ℝ :=
  let cos_B := Real.cos (Real.pi / 3)
  let AC_squared := AB^2 + BC^2 - 2 * AB * BC * cos_B
  Real.sqrt AC_squared

theorem find_length_of_AC_in_triangle_ABC :
  ∃ AC : ℝ, ∀ (AB BC : ℝ) (angle_B : ℝ) (h_AB : AB = 1) (h_BC : BC = 2) (h_angle_B : angle_B = Real.pi / 3),
    length_AC_in_triangle_ABC AB BC angle_B h_AB h_BC h_angle_B = Real.sqrt 3 :=
by sorry

end find_length_of_AC_in_triangle_ABC_l238_238444


namespace cylinder_volume_division_l238_238940

theorem cylinder_volume_division :
  let r_large := 3
  let h_large := 8
  let r_small := 2
  let h_small := 5
  let V_large := π * r_large ^ 2 * h_large
  let V_small := π * r_small ^ 2 * h_small
  (V_large / V_small).floor = 3 :=
by
  sorry

end cylinder_volume_division_l238_238940


namespace magnitude_difference_l238_238725

open Real

variables {a b : ℝ^3}

def magnitude (v : ℝ^3) := sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def angle_between_vectors (u v : ℝ^3) := 
  acos ((u.1 * v.1 + u.2 * v.2 + u.3 * v.3) / (magnitude u * magnitude v))

theorem magnitude_difference
  (h_angle : angle_between_vectors a b = 2 * π / 3)   -- 120 degrees in radians
  (ha : magnitude a = 1)
  (hb : magnitude b = 3) :
  magnitude (a - b) = sqrt 13 :=
by
  sorry

end magnitude_difference_l238_238725


namespace find_theta_plus_3phi_l238_238362

variables (θ φ : ℝ)

-- The conditions
variables (h1 : 0 < θ ∧ θ < π / 2) (h2 : 0 < φ ∧ φ < π / 2)
variables (h3 : Real.tan θ = 1 / 3) (h4 : Real.sin φ = 3 / 5)

theorem find_theta_plus_3phi :
  θ + 3 * φ = π - Real.arctan (199 / 93) :=
sorry

end find_theta_plus_3phi_l238_238362


namespace smallest_number_of_cubes_l238_238566

theorem smallest_number_of_cubes (length width depth : ℕ) (h_length : length = 36) (h_width : width = 45) (h_depth : depth = 18):
  ∃ n : ℕ, n = 40 :=
by
  use 40
  sorry

end smallest_number_of_cubes_l238_238566


namespace subset_condition_divides_twice_l238_238873

theorem subset_condition_divides_twice:
  ∀ (S : Set ℕ), (S ⊆ Finset.range 2013.to_set) → (1000 ≤ S.card) → 
  ∃ a b ∈ S, a ≠ b ∧ b ∣ 2 * a :=
begin
  sorry
end

end subset_condition_divides_twice_l238_238873


namespace common_chord_length_l238_238544

theorem common_chord_length (r d : ℝ) (h_r : r = 12) (h_d : d = 16) (h_overlap : d < 2 * r) :
  let L := 2 * real.sqrt (r^2 - (d/2)^2) in
  L = 8 * real.sqrt(5) :=
by
  -- Definitions used in conditions
  let r := 12
  let d := 16
  have h1 : r = 12 := by assumption
  have h2 : d = 16 := by assumption
  have h3 : d < 2 * r := by linarith
  
  -- Skip the actual proof
  sorry

end common_chord_length_l238_238544


namespace problem_statement_l238_238787

variables {α : ℂ} {b : ℕ → ℝ} {n : ℕ}

theorem problem_statement
  (h_nonreal_root : α^4 = 1 ∧ (α ≠ 1 ∧ α ≠ -1 ∧ α ≠ 1)) 
  (h_real_numbers : ∀ k, 1 ≤ k ∧ k ≤ n → b k ∈ ℝ)
  (h_given_condition : (∑ k in finset.range n, 1 / (b k + α)) = 3 + 4 * complex.I) :
  (∑ k in finset.range n, (3 * b k - 2) / (b k ^ 2 - b k + 1)) = 6 :=
by sorry

end problem_statement_l238_238787


namespace max_ints_less_than_neg5_l238_238906

theorem max_ints_less_than_neg5 (a b c d e : ℤ) (h : a + b + c + d + e = 20) :
  (∃ k ≤ 4, ∀ (x ∈ {a, b, c, d, e}), x < -5 → k = {a, b, c, d, e}.count (λ x, x < -5)) :=
by
  sorry

end max_ints_less_than_neg5_l238_238906


namespace max_rabbits_l238_238116

theorem max_rabbits (N : ℕ) (h1 : ∀ k, k = N → k = 27 → true)
    (h2 : ∀ n_l : ℕ, n_l = 13 → n_l <= N)
    (h3 : ∀ n_j : ℕ, n_j = 17 → n_j <= N)
    (h4 : ∀ n_both : ℕ, n_both >= 3 → true) :
  N <= 27 :=
begin
  sorry
end

end max_rabbits_l238_238116


namespace simplify_expression_correct_l238_238049

def simplify_expression : ℚ :=
  (5^5 + 5^3) / (5^4 - 5^2)

theorem simplify_expression_correct : simplify_expression = 65 / 12 :=
  sorry

end simplify_expression_correct_l238_238049


namespace seating_arrangements_l238_238150

theorem seating_arrangements (n : ℕ) (h : n = 5) :
  (∃ (κακά : Π i, fin n), (∀ i, κακά i = i → 1 ≤ i ≤ 2)) →
  (fintype (fin n)) → fintype.card (fin n) = (120 - 10 - 1 : ℕ) :=
by
  have : n = 5 := h
  sorry

end seating_arrangements_l238_238150


namespace maximum_numbers_up_to_235_l238_238830

def max_remaining_numbers : ℕ := 118

theorem maximum_numbers_up_to_235 (numbers : set ℕ) (h₁ : ∀ n ∈ numbers, n ≤ 235)
  (h₂ : ∀ a b ∈ numbers, a ≠ b → ¬ (a - b).abs ∣ a) :
  numbers.card ≤ max_remaining_numbers :=
sorry

end maximum_numbers_up_to_235_l238_238830


namespace percentage_of_hexagon_area_is_closest_to_17_l238_238265

noncomputable def tiling_area_hexagon_percentage : Real :=
  let total_area := 2 * 3
  let square_area := 1 * 1 
  let squares_count := 5 -- Adjusted count from 8 to fit total area properly
  let square_total_area := squares_count * square_area
  let hexagon_area := total_area - square_total_area
  let percentage := (hexagon_area / total_area) * 100
  percentage

theorem percentage_of_hexagon_area_is_closest_to_17 :
  abs (tiling_area_hexagon_percentage - 17) < 1 :=
sorry

end percentage_of_hexagon_area_is_closest_to_17_l238_238265


namespace number_of_functions_satisfying_condition_l238_238383

def condition_f (n : ℕ) (f : Fin 10 → Fin 5) :=
  ∀ k : Fin 9, |(f k) - (f ⟨k.1 + 1, by simp [k.2]⟩)| ≥ 3

theorem number_of_functions_satisfying_condition :
  ∃ (f : Fin 10 → Fin 5), condition_f 10 f ∧ (number_of_satisfying_functions f = 288) :=
sorry

end number_of_functions_satisfying_condition_l238_238383


namespace total_visitors_l238_238277

variable (V : ℕ)

-- Define the conditions
def forty_percent_fell_ill (total : ℕ) : Prop := 0.4 * total = (total - 300)
def three_hundred_not_fell_ill (total : ℕ) : Prop := 0.6 * total = 300

-- Prove that the total number of visitors is 500
theorem total_visitors (V : ℕ) (h1 : three_hundred_not_fell_ill V) : V = 500 :=
sorry

end total_visitors_l238_238277


namespace angle_of_inclination_theorem_l238_238520

variable (R r : ℝ)
variable (h : ℝ := R)

-- Given conditions
variable (h_eq_R : h = R)
variable (P_hex_eq_P_tri : (6 * 2 * (r * Real.sqrt 3) / 3) = (3 * 2 * (R * Real.sqrt 3) / 3))

-- Statement to be proved
theorem angle_of_inclination_theorem (φ : ℝ) :
  P_hex_eq_P_tri → h_eq_R → φ = Real.arctan 4 := by
  sorry

end angle_of_inclination_theorem_l238_238520


namespace geometric_series_modulo_l238_238334

theorem geometric_series_modulo :
  let S := (∑ i in Finset.range 500, 3^i) in
  S % 500 = 440 :=
by
  -- let S be the sum of the geometric series
  let S := (∑ i in Finset.range 500, 3^i)
  -- simplify the sum using the geometric series formula
  have hS : S = (3^500 - 1) / 2 := sorry
  -- need to find the value of (3^500 - 1) / 2 % 500
  have h500 : (3^500 - 1) / 2 % 500 = 440 := sorry
  -- combine results
  exact h500
  sorry

end geometric_series_modulo_l238_238334


namespace mean_and_median_of_points_l238_238622

theorem mean_and_median_of_points :
  let points := [50, 57, 49, 57, 32, 46, 65, 28, 92]
  let mean := (476 : ℝ) / 9
  let median := 50
  (mean ≈ 52.89) ∧ (median = 50) :=
by
  let points := [50, 57, 49, 57, 32, 46, 65, 28, 92]
  let mean := (476 : ℝ) / 9
  let median := 50
  have h_mean : mean ≈ 52.89 := sorry
  have h_median : median = 50 := sorry
  exact ⟨h_mean, h_median⟩

end mean_and_median_of_points_l238_238622


namespace parametric_to_standard_l238_238639

theorem parametric_to_standard (θ : ℝ) (x y : ℝ)
  (h1 : x = 1 + 2 * Real.cos θ)
  (h2 : y = 2 * Real.sin θ) :
  (x - 1)^2 + y^2 = 4 := 
sorry

end parametric_to_standard_l238_238639


namespace inverse_f_3_eq_neg1_l238_238484

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 2

def inverse_f_domain : set ℝ :=
  {x | -2 ≤ x ∧ x < 0}

theorem inverse_f_3_eq_neg1 :
  function.injective f →
  (∀ y ∈ (f '' inverse_f_domain), ∃! x ∈ inverse_f_domain, f x = y) →
  f⁻¹' {3} = {-1} :=
by
  intros h_injective h_surjective
  have h : f (-1) = 3 :=
    by simp [f]
  simp [set.image, inverse_f_domain, ← set.singleton_eq_singleton_iff] at h_surjective
  rw set.inv_image at h_surjective
  sorry

end inverse_f_3_eq_neg1_l238_238484


namespace next_in_sequence_is_65_by_19_l238_238802

section
  open Int

  -- Definitions for numerators
  def numerator_sequence : ℕ → ℤ
  | 0 => -3
  | 1 => 5
  | 2 => -9
  | 3 => 17
  | 4 => -33
  | (n + 5) => numerator_sequence n * (-2) + 1

  -- Definitions for denominators
  def denominator_sequence : ℕ → ℕ
  | 0 => 4
  | 1 => 7
  | 2 => 10
  | 3 => 13
  | 4 => 16
  | (n + 5) => denominator_sequence n + 3

  -- Next term in the sequence
  def next_term (n : ℕ) : ℚ :=
    (numerator_sequence (n + 5) : ℚ) / (denominator_sequence (n + 5) : ℚ)

  -- Theorem stating the next number in the sequence
  theorem next_in_sequence_is_65_by_19 :
    next_term 0 = 65 / 19 :=
  by
    unfold next_term
    simp [numerator_sequence, denominator_sequence]
    sorry
end

end next_in_sequence_is_65_by_19_l238_238802


namespace arccos_one_eq_zero_l238_238300

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  -- the proof will go here
  sorry

end arccos_one_eq_zero_l238_238300


namespace right_triangle_area_l238_238708

theorem right_triangle_area (A B C : ℝ) (hA : A = 64) (hB : B = 49) (hC : C = 225) :
  let a := Real.sqrt A
  let b := Real.sqrt B
  let c := Real.sqrt C
  ∃ (area : ℝ), area = (1 / 2) * a * b ∧ area = 28 :=
by
  sorry

end right_triangle_area_l238_238708
