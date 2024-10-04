import Mathlib

namespace wallet_amount_l123_123138

-- Definitions of given conditions
def num_toys := 28
def cost_per_toy := 10
def num_teddy_bears := 20
def cost_per_teddy_bear := 15

-- Calculation of total costs
def total_cost_of_toys := num_toys * cost_per_toy
def total_cost_of_teddy_bears := num_teddy_bears * cost_per_teddy_bear

-- Total amount of money in Louise's wallet
def total_cost := total_cost_of_toys + total_cost_of_teddy_bears

-- Proof that the total cost is $580
theorem wallet_amount : total_cost = 580 :=
by
  -- Skipping the proof for now
  sorry

end wallet_amount_l123_123138


namespace sum_of_eight_numbers_l123_123902

variable (avg : ℝ) (n : ℕ)

-- Given condition
def average_eq_of_eight_numbers : avg = 5.5 := rfl
def number_of_items_eq_eight : n = 8 := rfl

-- Theorem to prove
theorem sum_of_eight_numbers (h1 : average_eq_of_eight_numbers avg)
                             (h2 : number_of_items_eq_eight n) :
  avg * n = 44 :=
by
  -- Proof will be inserted here
  sorry

end sum_of_eight_numbers_l123_123902


namespace smallest_n_floor_sum_exceeds_2000_l123_123128

theorem smallest_n_floor_sum_exceeds_2000 :
  let floor_div := λ (k : ℕ), k / 15 in
  let sum_floor := λ (n : ℕ), (Finset.range (n + 1)).sum (λ k, floor_div k) in
  let condition := ∀ m : ℕ, (sum_floor m > 2000) → m ≥ 252 in
  condition 252 :=
by
  sorry

end smallest_n_floor_sum_exceeds_2000_l123_123128


namespace additional_slow_workers_needed_l123_123482

-- Definitions based on conditions
def production_per_worker_fast (m : ℕ) (n : ℕ) (a : ℕ) : ℚ := m / (n * a)
def production_per_worker_slow (m : ℕ) (n : ℕ) (b : ℕ) : ℚ := m / (n * b)

def required_daily_production (p : ℕ) (q : ℕ) : ℚ := p / q

def contribution_fast_workers (m : ℕ) (n : ℕ) (a : ℕ) (c : ℕ) : ℚ :=
  (m * c) / (n * a)

def remaining_production (p : ℕ) (q : ℕ) (m : ℕ) (n : ℕ) (a : ℕ) (c : ℕ) : ℚ :=
  (p / q) - ((m * c) / (n * a))

def required_slow_workers (p : ℕ) (q : ℕ) (m : ℕ) (n : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) : ℚ :=
  ((p * n * a - q * m * c) * b) / (q * m * a)

theorem additional_slow_workers_needed (m n a b p q c : ℕ) :
  required_slow_workers p q m n a b c = ((p * n * a - q * m * c) * b) / (q * m * a) := by
  sorry

end additional_slow_workers_needed_l123_123482


namespace expected_value_eight_sided_die_l123_123358

noncomputable def expected_value_above_four : ℝ :=
  let outcomes := [5, 6, 7, 8]
  let probabilities := [1 / 4, 1 / 4, 1 / 4, 1 / 4]
  let base_expected_value := (outcomes.zip probabilities).sumBy (λ p, p.1 * p.2)
  base_expected_value * (1 / 2)

theorem expected_value_eight_sided_die :
  expected_value_above_four = 3.25 := by
  sorry

end expected_value_eight_sided_die_l123_123358


namespace find_angles_of_triangle_ABC_l123_123080

open EuclideanGeometry

-- Definitions for acute-angled triangle, circumcenter, intersected points and given angles
variable (A B C D E O : Point) (α β γ : ℝ)
variable [is_acute_angle_triangle : α + β + γ = 180 ∧ α < 90 ∧ β < 90 ∧ γ < 90]
variable [circumcenter_triangle_O : IsCircumcenter O A B C]
variable [intersect_points : Intersect BO AC D ∧ Intersect CO AB E]
variable [angle_definitions : ∠ BDE = 50 ∧ ∠ CED = 30]

-- Proof statement to find angles of triangle ABC
theorem find_angles_of_triangle_ABC :
    ∠ BAC = 50 ∧ ∠ ABC = 70 ∧ ∠ BCA = 60 :=
by
  sorry

end find_angles_of_triangle_ABC_l123_123080


namespace number_of_terms_in_arithmetic_sequence_l123_123475

noncomputable def arithmetic_sequence_terms (a d n : ℕ) : Prop :=
  let sum_first_three := 3 * a + 3 * d = 34
  let sum_last_three := 3 * a + 3 * (n - 1) * d = 146
  let sum_all := n * (2 * a + (n - 1) * d) / 2 = 390
  (sum_first_three ∧ sum_last_three ∧ sum_all) → n = 13

theorem number_of_terms_in_arithmetic_sequence (a d n : ℕ) : arithmetic_sequence_terms a d n → n = 13 := 
by
  sorry

end number_of_terms_in_arithmetic_sequence_l123_123475


namespace ArcherInGoldenArmorProof_l123_123929

-- Definitions of the problem
variables (soldiers : Nat) (archers soldiersInGolden : Nat) 
variables (soldiersInBlack archersInGolden archersInBlack swordsmenInGolden swordsmenInBlack : Nat)
variables (truthfulSwordsmenInBlack lyingArchersInBlack lyingSwordsmenInGold truthfulArchersInGold : Nat)
variables (yesToGold yesToArcher yesToMonday : Nat)

-- Given conditions
def ProblemStatement : Prop :=
  soldiers = 55 ∧
  yesToGold = 44 ∧
  yesToArcher = 33 ∧
  yesToMonday = 22 ∧
  soldiers = archers + (soldiers - archers) ∧
  soldiers = soldiersInGolden + soldiersInBlack ∧
  archers = archersInGolden + archersInBlack ∧
  soldiersInGolden = archersInGolden + swordsmenInGolden ∧
  soldiersInBlack = archersInBlack + swordsmenInBlack ∧
  truthfulSwordsmenInBlack = swordsmenInBlack ∧
  lyingArchersInBlack = archersInBlack ∧
  lyingSwordsmenInGold = swordsmenInGolden ∧
  truthfulArchersInGold = archersInGolden ∧
  yesToGold = truthfulArchersInGold + (swordsmenInGolden + archersInBlack) ∧
  yesToArcher = truthfulArchersInGold + lyingSwordsmenInGold

-- Conclusion
def Conclusion : Prop :=
  archersInGolden = 22

-- Proof statement
theorem ArcherInGoldenArmorProof : ProblemStatement → Conclusion :=
by
  sorry

end ArcherInGoldenArmorProof_l123_123929


namespace greatest_possible_points_l123_123923

def total_games (n_teams : ℕ) : ℕ :=
  (n_teams * (n_teams - 1)) / 2 * 2

def total_points (n_games : ℕ) : ℕ :=
  3 * n_games

def top_teams_points (points : ℕ) : Prop :=
  (∃ A B C D : ℕ, A = B ∧ B = C ∧ C = D ∧  A = points) ∧ A + B + C + D = 4 * points

theorem greatest_possible_points :
  ∀ (n_teams : ℕ) (points : ℕ), 
  n_teams = 8 → total_games n_teams = 56 → total_points 56 = 168 →
  (∃ top_points : ℕ, top_teams_points top_points) →
  top_points = 42 :=
by
  intros n_teams points ht hg hp htp
  sorry

end greatest_possible_points_l123_123923


namespace expected_sides_rectangle_expected_sides_polygon_l123_123627

-- Part (a)
theorem expected_sides_rectangle (k : ℕ) (h : k > 0) : (4 + 4 * k) / (k + 1) → 4 :=
by sorry

-- Part (b)
theorem expected_sides_polygon (n k : ℕ) (h : n > 2) (h_k : k ≥ 0) : (n + 4 * k) / (k + 1) = (n + 4 * k) / (k + 1) :=
by sorry

end expected_sides_rectangle_expected_sides_polygon_l123_123627


namespace general_formula_compare_magnitudes_l123_123406

open Nat

-- Definition of the sequence sum S_n
def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in range (n+1), a (i+1)

-- Condition: 2a_n = S_n + n for n in ℕ^*
def condition (a : ℕ → ℕ) (n : ℕ) : Prop :=
  2 * a n = sequence_sum a n + n

-- General formula of the terms in the sequence a_n
theorem general_formula (a : ℕ → ℕ) (h : ∀ n, condition a (n+1)) (n : ℕ) : a (n + 1) = 2^(n + 1) - 1 :=
  sorry

-- Comparison of magnitudes S_n and f(n) where f(n) = n^2
def f (n : ℕ) : ℕ := n * n

theorem compare_magnitudes (a : ℕ → ℕ) (h1 : ∀ n, condition a (n+1)) (h2: ∀ n, a (n + 1) = 2^(n + 1) - 1) (n : ℕ) (hn : 3 ≤ n) :
  sequence_sum a n > f n :=
  sorry

end general_formula_compare_magnitudes_l123_123406


namespace sum_of_integers_l123_123584

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 10) (h2 : x * y = 80) (hx_pos : 0 < x) (hy_pos : 0 < y) : x + y = 20 := by
  sorry

end sum_of_integers_l123_123584


namespace transform_identity_l123_123616

theorem transform_identity (a b c d : ℝ) : 
  (a^2 + b^2) * (c^2 + d^2) = (a * c + b * d)^2 + (a * d - b * c)^2 := 
sorry

end transform_identity_l123_123616


namespace archers_in_golden_armor_l123_123956

theorem archers_in_golden_armor (total_soldiers archers_swordsmen total_affirmations armor_affirmations archer_affirmations monday_affirmations : ℕ)
  (h1: total_soldiers = 55)
  (h2: armor_affirmations = 44)
  (h3: archer_affirmations = 33)
  (h4: monday_affirmations = 22)
  (h5: ∑ x in ({armor_affirmations, archer_affirmations, monday_affirmations} : finset ℕ), x = 99) 
  : ∃ (archers_in_golden : ℕ), archers_in_golden = 22 := by
  -- Definitions and theorems will go here
  sorry

end archers_in_golden_armor_l123_123956


namespace prove_there_exists_sum_abc_l123_123580

noncomputable def problem_statement (M : Set ℕ) (A B C : Set ℕ) :=
  (∀ (a ∈ A) (b ∈ B) (c ∈ C), a + b = c ∨ a + c = b ∨ b + c = a) ∧
  A ∪ B ∪ C = M ∧
  A ∩ B = ∅ ∧
  A ∩ C = ∅ ∧
  B ∩ C = ∅ ∧
  ∃ (A B C : Set ℕ), 
    A.card = 672 ∧ 
    B.card = 672 ∧ 
    C.card = 672 

theorem prove_there_exists_sum_abc : 
  ∀ {M : Set ℕ},
  M = { x | 1 ≤ x ∧ x ≤ 2016 } → 
  ∃ (A B C : Set ℕ), (∀ (a ∈ A) (b ∈ B) (c ∈ C), a + b = c ∨ a + c = b ∨ b + c = a) ∧
  A ∪ B ∪ C = M ∧
  A ∩ B = ∅ ∧
  A ∩ C = ∅ ∧
  B ∩ C = ∅ ∧
  (A.card = 672 ∧ B.card = 672 ∧ C.card = 672) :=
by
  sorry

end prove_there_exists_sum_abc_l123_123580


namespace y_squared_range_l123_123465

theorem y_squared_range (y : ℝ) (h : (∛(y + 12) - ∛(y - 12) = 4)) : 105 < y^2 ∧ y^2 < 115 := 
sorry

end y_squared_range_l123_123465


namespace vector_magnitude_at_theta_max_l123_123864

variables {a b : EuclideanSpace ℝ (fin 2)}

-- Given conditions
def not_collinear : Prop := ¬ (∃ k : ℝ, b = k • a)
def mag_a : Prop := ‖a‖ = 1
def dot_a_b : Prop := ⬝ a = b = 1
def theta_max : Prop := ∃θ : ℝ, ∀ θ', θ ≥ θ'

-- Statement to prove
theorem vector_magnitude_at_theta_max
  (h1 : not_collinear) (h2 : mag_a) (h3 : dot_a_b) (h4 : theta_max) :
  ‖a - b‖ = sqrt 3 :=
sorry

end vector_magnitude_at_theta_max_l123_123864


namespace calculate_mod_l123_123457

theorem calculate_mod
  (x : ℤ)
  (h : 4 * x + 9 ≡ 3 [ZMOD 19]) :
  3 * x + 8 ≡ 13 [ZMOD 19] :=
sorry

end calculate_mod_l123_123457


namespace true_statements_count_l123_123557

theorem true_statements_count {x y a b : ℝ} (hx : x > 0) (hy : y > 0) (ha : a > 0) (hb : b > 0) (hxa : x < a) (hyb : y < b) 
: (ite (x + y < a + b) 1 0) + (ite (x - y < a - b) 1 0) + (ite (xy < ab) 1 0) + (ite ((x / y < a / b) → (x / y < a / b)) 1 0) = 2 :=
sorry

end true_statements_count_l123_123557


namespace derek_walk_time_difference_l123_123992

theorem derek_walk_time_difference :
  ∃ x : ℕ, 12 * x - 9 * x = 60 ∧ (12 - 9) * x = 60 :=
by {
  use 20,
  split,
  {
    calc
      12 * 20 - 9 * 20 = 240 - 180 := by ring
                    ... = 60       := by norm_num,
  },
  {
    calc
      (12 - 9) * 20 = 3 * 20 := by norm_num
                    ... = 60 := by norm_num,
  },
}

end derek_walk_time_difference_l123_123992


namespace geometric_series_sum_l123_123369

theorem geometric_series_sum :
  let a := -3
  let r := -2
  let n := 9
  let term := a * r^(n-1)
  let Sn := (a * (r^n - 1)) / (r - 1)
  term = -768 → Sn = 514 := by
  intros a r n term Sn h_term
  sorry

end geometric_series_sum_l123_123369


namespace sum_cosines_eq_zero_l123_123658

theorem sum_cosines_eq_zero (n : ℕ) (h : n > 1) :
  ∑ i in Finset.range n, Real.cos (2 * i * Real.pi / n) = 0 :=
by
  sorry

end sum_cosines_eq_zero_l123_123658


namespace sum_in_base_8_l123_123549

-- Definitions and the theorem statement
theorem sum_in_base_8 :
  ∀ (c : ℕ), (c + 3) * (c + 7) * (c + 9) = 4 * c^3 + 3 * c^2 + 7 * c + 5 →
  let t := (c + 3) + (c + 7) + (c + 9) in 
  t = 43 → nat_to_base 8 (3 * c + 19) = "53" :=
by
  intros c h_product h_sum t_def
  have h_c : c = 8
  { sorry }
  rw [t_def, h_c]
  calc
    43 = nat_to_base 8 43 : rfl
    ... = "53"           : sorry -- base conversion

end sum_in_base_8_l123_123549


namespace complex_number_solution_l123_123429

open Complex

noncomputable def z : ℂ := 1 - I
def i : ℂ := I
def z_conjugate : ℂ := conj z

theorem complex_number_solution :
  (z_conjugate / (1 - i)) = i ∧ z = 1 - I :=
by
  sorry

end complex_number_solution_l123_123429


namespace smallest_four_digit_multiple_of_18_l123_123763

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n = 1008 ∧ (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ 
                                ∀ m : ℕ, ((1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0)) → 1008 ≤ m :=
by
  sorry

end smallest_four_digit_multiple_of_18_l123_123763


namespace angle_AC_BD_either_36_or_90_l123_123609

noncomputable def angle_between_AC_and_BD (A B C D : ℝ × ℝ × ℝ) (h_AB : dist A B = dist B C) (h_BC : dist B C = dist C D) (angle_ABC : ∠ A B C = 36) (angle_BCD : ∠ B C D = 36) (angle_CDA : ∠ C D A = 36) : ℝ :=
sorry

theorem angle_AC_BD_either_36_or_90 (A B C D : ℝ × ℝ × ℝ) (h_AB : dist A B = dist B C) (h_BC : dist B C = dist C D) (angle_ABC : ∠ A B C = 36) (angle_BCD : ∠ B C D = 36) (angle_CDA : ∠ C D A = 36) :
  angle_between_AC_and_BD A B C D h_AB h_BC angle_ABC angle_BCD angle_CDA = 36 ∨
  angle_between_AC_and_BD A B C D h_AB h_BC angle_ABC angle_BCD angle_CDA = 90 :=
sorry

end angle_AC_BD_either_36_or_90_l123_123609


namespace smallest_sum_of_primes_using_digits_1_to_7_once_l123_123030

theorem smallest_sum_of_primes_using_digits_1_to_7_once : 
  ∃ (p1 p2 p3 : ℕ), prime p1 ∧ prime p2 ∧ prime p3 ∧ 
  (∀ d ∈ finset.range 1 8, d ∈ digits 10 p1 ∨ d ∈ digits 10 p2 ∨ d ∈ digits 10 p3) ∧ 
  (∑ i in {p1,p2,p3}, id i = 263) ∧ 
  ∀ (q1 q2 q3 : ℕ), prime q1 → prime q2 → prime q3 → 
  (∀ d ∈ finset.range 1 8, d ∈ digits 10 q1 ∨ d ∈ digits 10 q2 ∨ d ∈ digits 10 q3) → 
    ∑ i in {q1,q2,q3}, id i ≥ 263 :=
by
  sorry

end smallest_sum_of_primes_using_digits_1_to_7_once_l123_123030


namespace max_triangle_area_l123_123019

noncomputable def ω := 2

def f (x : Real) : Real := 2 * sin (ω * x)
def g (x : Real) : Real := 2 * sin (2 * x - (2 * Real.pi) / 3)

def a : Real := 5
def A : Real := Real.pi / 3

def triangle_area_max (a : Real) (A : Real) : Real :=
  (1 / 2) * a^2 * (sin A)

theorem max_triangle_area :
  triangle_area_max a A = (25 * Real.sqrt 3) / 4
:= by
  sorry

end max_triangle_area_l123_123019


namespace a_b_total_money_l123_123302

variable (A B : ℝ)

theorem a_b_total_money (h1 : (4 / 15) * A = (2 / 5) * 484) (h2 : B = 484) : A + B = 1210 := by
  sorry

end a_b_total_money_l123_123302


namespace binomial_sum_remainder_l123_123727

theorem binomial_sum_remainder : (∑ k in Finset.range 34 \ {0}, Nat.choose 33 k) % 9 = 7 := sorry

end binomial_sum_remainder_l123_123727


namespace I1I2B1B2_cyclic_l123_123091

-- Definitions for the given conditions
variables {A B C A1 B1 C1 I1 I2 B2 : Point}

-- Assume the given conditions as hypotheses
axiom triangle_ABC : Triangle A B C
axiom altitude_AA1 : IsAltitude A A1 B C
axiom altitude_BB1 : IsAltitude B B1 A C
axiom altitude_CC1 : IsAltitude C C1 A B
axiom incenter_I1 : IsIncenter I1 (Triangle A C1 B1)
axiom incenter_I2 : IsIncenter I2 (Triangle C A1 B1)
axiom incircle_touches_AC_at_B2 : TouchesIncircleAt (Triangle A B C) A C B2

-- The statement to be proved
theorem I1I2B1B2_cyclic : CyclicQuadrilateral I1 I2 B1 B2 :=
by
  sorry

end I1I2B1B2_cyclic_l123_123091


namespace cos_squared_sum_sin_squared_sum_l123_123986

theorem cos_squared_sum (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.cos (A / 2) ^ 2 + Real.cos (B / 2) ^ 2 + Real.cos (C / 2) ^ 2 =
  2 * (1 + Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2)) :=
sorry

theorem sin_squared_sum (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (A / 2) ^ 2 + Real.sin (B / 2) ^ 2 + Real.sin (C / 2) ^ 2 =
  1 - 2 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) :=
sorry

end cos_squared_sum_sin_squared_sum_l123_123986


namespace circle_equation_tangent_to_line_and_through_intersections_l123_123867

theorem circle_equation_tangent_to_line_and_through_intersections
  (C1 C2 : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, C1 x y ↔ x^2 + y^2 = 4) ∧
  (∀ x y : ℝ, C2 x y ↔ x^2 + y^2 - 2 * x - 4 * y + 4 = 0) ∧
  (∀ x y : ℝ, l x y ↔ x + 2 * y = 0) →
  (∀ x y : ℝ, (x^2 + y^2 - x - 2 * y = 0) ↔
    ((C1 x y ∧ C2 x y) ∧ l x y = 0)) :=
by
  intros h
  sorry

end circle_equation_tangent_to_line_and_through_intersections_l123_123867


namespace champagne_equality_impossible_l123_123187

theorem champagne_equality_impossible :
  ∀ (n : ℕ) (glasses : Fin n → ℚ)
    (h1 : n = 2018)
    (h2 : ∃! i : Fin n, glasses i = 2 ∧ ∀ j : Fin n, i ≠ j → glasses j = 1),
    ¬(∃ m : ℚ, ∀ i : Fin n, glasses i = m) :=
by
  intro n glasses h1 h2
  cases h1
  rcases h2 with ⟨i, ⟨hi2, hi1⟩⟩
  have h_total : ∑ i, glasses i = 2019 := sorry
  intro h_eq
  cases h_eq with m h_glasses
  have h_m : m * 2018 = 2019 := sorry
  have h_m_not_int : ¬ ∃ k : ℤ, k = 2019 / 2018 := sorry
  contradiction

end champagne_equality_impossible_l123_123187


namespace no_obtuse_triangle_probability_l123_123799

noncomputable def probability_no_obtuse_triangle : ℝ :=
 let θ := 1/2 in
 let prob_A2_A3_given_A0A1 := (3/8) * (3/8) in
 θ * prob_A2_A3_given_A0A1

theorem no_obtuse_triangle_probability :
  probability_no_obtuse_triangle = 9/128 :=
by
  sorry

end no_obtuse_triangle_probability_l123_123799


namespace largest_m_divisibility_property_l123_123439

theorem largest_m_divisibility_property :
  ∃ m : ℕ, m = 499 ∧ ∀ S : set ℕ, S ⊆ { n | n ∈ finset.range 1000 + 1} → finset.card S = 501 →
    ∃ x y ∈ S, x ≠ y ∧ (x ∣ y ∨ y ∣ x) :=
by sorry

end largest_m_divisibility_property_l123_123439


namespace point_of_intersection_of_asymptotes_l123_123745

def f (x : ℝ) : ℝ := (x^2 - 6 * x + 8) / (x^2 - 6 * x + 9)

theorem point_of_intersection_of_asymptotes : ∃ x y : ℝ, x = 3 ∧ y = 1 ∧ 
  ∀ (ε > 0), ∃ δ > 0, ∀ (x' : ℝ), 0 < abs (x' - x) < δ → abs (f x' - y) < ε :=
by 
  sorry

end point_of_intersection_of_asymptotes_l123_123745


namespace sum_primes_between_20_and_40_l123_123648

open Nat

def primesBetween20And40 : List Nat := [23, 29, 31, 37]

theorem sum_primes_between_20_and_40 :
  (primesBetween20And40.sum = 120) :=
by
  sorry

end sum_primes_between_20_and_40_l123_123648


namespace expected_sides_general_expected_sides_rectangle_l123_123624

-- General Problem
theorem expected_sides_general (n k : ℕ) : 
  (∀ n k : ℕ, n ≥ 3 → k ≥ 0 → (1:ℝ) / ((k + 1:ℝ)) * (n + 4 * k:ℝ) ≤ (n + 4 * k) / (k + 1)) := 
begin
  sorry
end

-- Specific Problem for Rectangle
theorem expected_sides_rectangle (k : ℕ) :
  (∀ k : ℕ, k ≥ 0 → 4 = (4 + 4 * k) / (k + 1)) := 
begin
  sorry
end

end expected_sides_general_expected_sides_rectangle_l123_123624


namespace coffee_shop_tables_l123_123326

def base_seven_to_base_ten (n : Nat) : Nat :=
  let d2 := n / 49
  let r2 := n % 49
  let d1 := r2 / 7
  let r1 := r2 % 7
  d2 * 49 + d1 * 7 + r1

theorem coffee_shop_tables :
  let chairs_base7 := 321
  let people_per_table := 3
  base_seven_to_base_ten(chairs_base7) / people_per_table = 54 :=
by
  sorry

end coffee_shop_tables_l123_123326


namespace roberto_outfits_l123_123175

-- Roberto's wardrobe constraints
def num_trousers : ℕ := 5
def num_shirts : ℕ := 6
def num_jackets : ℕ := 4
def num_shoes : ℕ := 3
def restricted_jacket_shoes : ℕ := 2

-- The total number of valid outfits
def total_outfits_with_constraint : ℕ := 330

-- Proving the equivalent of the problem statement
theorem roberto_outfits :
  (num_trousers * num_shirts * (num_jackets - 1) * num_shoes) + (num_trousers * num_shirts * 1 * restricted_jacket_shoes) = total_outfits_with_constraint :=
by
  sorry

end roberto_outfits_l123_123175


namespace symmetric_xy_plane_M_symmetric_z_axis_M_l123_123088

-- Define point M
def M : ℝ × ℝ × ℝ := (1, -2, 3)

-- Proposition : Symmetric point with respect to the xy-plane
def symmetric_xy_plane (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, p.2, -p.3)

-- Proposition : Symmetric point with respect to the z-axis
def symmetric_z_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (-p.1, -p.2, p.3)

-- Proof statements
theorem symmetric_xy_plane_M : symmetric_xy_plane M = (1, -2, -3) := by
  sorry

theorem symmetric_z_axis_M : symmetric_z_axis M = (-1, 2, 3) := by
  sorry

end symmetric_xy_plane_M_symmetric_z_axis_M_l123_123088


namespace sum_mod_1_to_20_l123_123282

theorem sum_mod_1_to_20 :
  (∑ i in finset.range 21, i) % 9 = 3 :=
by
  sorry

end sum_mod_1_to_20_l123_123282


namespace probability_same_plane_l123_123818

-- Define the number of vertices in a cube
def num_vertices : ℕ := 8

-- Define the number of vertices to be selected
def selection : ℕ := 4

-- Define the total number of ways to select 4 vertices out of 8
def total_ways : ℕ := Nat.choose num_vertices selection

-- Define the number of favorable ways to have 4 vertices lie in the same plane
def favorable_ways : ℕ := 12

-- Define the probability that the 4 selected vertices lie in the same plane
def probability : ℚ := favorable_ways / total_ways

-- The statement we need to prove
theorem probability_same_plane : probability = 6 / 35 := by
  sorry

end probability_same_plane_l123_123818


namespace no_obtuse_triangle_l123_123789

-- Conditions
def points_on_circle_uniformly_at_random (n : ℕ) : Prop :=
  ∀ i < n, ∀ j < n, i ≠ j → ∃ θ_ij : ℝ, 0 ≤ θ_ij ∧ θ_ij ≤ π

-- Theorem statement
theorem no_obtuse_triangle (hn : points_on_circle_uniformly_at_random 4) :
  let p := \left(\frac{1}{2}\right)^6 in
  p = \frac{1}{64} :=
sorry

end no_obtuse_triangle_l123_123789


namespace population_increase_rate_4_percent_l123_123209

-- Define the conditions
def present_population : ℝ := 1240
def population_after_1_year : ℝ := 1289.6

-- Define the formula for population increase rate
def population_increase_rate (present_population population_after_1_year : ℝ) : ℝ :=
  ((population_after_1_year - present_population) / present_population) * 100

-- Theorem stating the population increase rate is 4%
theorem population_increase_rate_4_percent :
  population_increase_rate present_population population_after_1_year = 4 :=
by
  sorry

end population_increase_rate_4_percent_l123_123209


namespace joan_games_attended_l123_123104
-- Mathematical definitions based on the provided conditions

def total_games_played : ℕ := 864
def games_missed_by_Joan : ℕ := 469

-- Theorem statement
theorem joan_games_attended : total_games_played - games_missed_by_Joan = 395 :=
by
  -- Proof omitted
  sorry

end joan_games_attended_l123_123104


namespace ArcherInGoldenArmorProof_l123_123926

-- Definitions of the problem
variables (soldiers : Nat) (archers soldiersInGolden : Nat) 
variables (soldiersInBlack archersInGolden archersInBlack swordsmenInGolden swordsmenInBlack : Nat)
variables (truthfulSwordsmenInBlack lyingArchersInBlack lyingSwordsmenInGold truthfulArchersInGold : Nat)
variables (yesToGold yesToArcher yesToMonday : Nat)

-- Given conditions
def ProblemStatement : Prop :=
  soldiers = 55 ∧
  yesToGold = 44 ∧
  yesToArcher = 33 ∧
  yesToMonday = 22 ∧
  soldiers = archers + (soldiers - archers) ∧
  soldiers = soldiersInGolden + soldiersInBlack ∧
  archers = archersInGolden + archersInBlack ∧
  soldiersInGolden = archersInGolden + swordsmenInGolden ∧
  soldiersInBlack = archersInBlack + swordsmenInBlack ∧
  truthfulSwordsmenInBlack = swordsmenInBlack ∧
  lyingArchersInBlack = archersInBlack ∧
  lyingSwordsmenInGold = swordsmenInGolden ∧
  truthfulArchersInGold = archersInGolden ∧
  yesToGold = truthfulArchersInGold + (swordsmenInGolden + archersInBlack) ∧
  yesToArcher = truthfulArchersInGold + lyingSwordsmenInGold

-- Conclusion
def Conclusion : Prop :=
  archersInGolden = 22

-- Proof statement
theorem ArcherInGoldenArmorProof : ProblemStatement → Conclusion :=
by
  sorry

end ArcherInGoldenArmorProof_l123_123926


namespace archers_in_golden_l123_123938

variables (soldiers archers swordsmen wearing_golden wearing_black : ℕ)
variables (truth_swordsmen_black lie_swordsmen_golden lie_archers_black truth_archers_golden : ℕ)

-- Given conditions
variables (cond1 : archers + swordsmen = 55)
variables (cond2 : wearing_golden + wearing_black = 55)
variables (cond3 : truth_swordsmen_black + lie_swordsmen_golden + lie_archers_black + truth_archers_golden = 55)
variables (cond4 : wearing_golden = 44)
variables (cond5 : archers = 33)
variables (cond6 : truth_swordsmen_black + lie_archers_black = 22)

-- Define the mathematic equivalent proof problem
theorem archers_in_golden : archers = 22 :=
by
  sorry

end archers_in_golden_l123_123938


namespace smallest_k_proof_l123_123148

noncomputable def smallest_k : ℕ :=
  45

theorem smallest_k_proof :
  ∀ (A : Finset ℕ), A.card = smallest_k →
    (∃ x y ∈ A, x ≠ y ∧ |Real.sqrt x - Real.sqrt y| < 1) :=
by
  sorry

end smallest_k_proof_l123_123148


namespace number_of_goats_l123_123608

variable (chickens : ℕ) (piglets : ℕ) (sick_animals : ℕ) (total_sick : ℕ)

-- Conditions
def chickens := 26
def piglets := 40
def sick_animals := (chickens + piglets + goats) / 2
def total_sick := 50

-- Proving the number of goats given the conditions
theorem number_of_goats (goats : ℕ) : goats = 34 :=
by
  have total_animals := total_sick * 2
  have total_chickens_piglets := chickens + piglets
  have goats := total_animals - total_chickens_piglets
  sorry

end number_of_goats_l123_123608


namespace avg_people_moving_to_florida_per_hour_l123_123980

theorem avg_people_moving_to_florida_per_hour (people : ℕ) (days : ℕ) (hours_per_day : ℕ) 
  (h1 : people = 3000) (h2 : days = 5) (h3 : hours_per_day = 24) : 
  people / (days * hours_per_day) = 25 := by
  sorry

end avg_people_moving_to_florida_per_hour_l123_123980


namespace repeated_process_pure_alcohol_l123_123161

theorem repeated_process_pure_alcohol : 
  ∃ n : ℕ, n ≥ 4 ∧ ∀ m < 4, 2 * (1 / 2 : ℝ)^(m : ℝ) ≥ 0.2 := by
  sorry

end repeated_process_pure_alcohol_l123_123161


namespace claps_per_second_l123_123477

theorem claps_per_second (claps_per_minute : Nat) (seconds_per_minute : Nat) (h : claps_per_minute = 1020) (hs : seconds_per_minute = 60) : 
  claps_per_minute / seconds_per_minute = 17 := by
  rw [h, hs]
  sorry

end claps_per_second_l123_123477


namespace minimum_distance_BC_and_coordinates_C_l123_123078

-- Definitions as per the conditions in a)
variable (A B C : ℝ × ℝ)

-- Given points
def A_coord : ℝ × ℝ := (-3, 2)
def B_coord : ℝ × ℝ := (3, 4)

-- The condition AC is parallel to the x-axis implies the y-coordinate of C equals the y-coordinate of A.
def AC_parallel : Prop := C.snd = A_coord.snd

-- Define the distance function for points in 2D space
def distance (p1 p2 : ℝ × ℝ) : ℝ := abs (p2.1 - p1.1) + abs (p2.2 - p1.2)

-- The statement that we need to prove
theorem minimum_distance_BC_and_coordinates_C (hA : A = A_coord) (hB : B = B_coord) (hC_parallel : AC_parallel) :
  distance B C = 2 ∧ C = (3, 2) :=
by
  sorry

end minimum_distance_BC_and_coordinates_C_l123_123078


namespace no_valid_coloring_l123_123989

open Nat

-- Define the color type
inductive Color
| blue
| red
| green

-- Define the coloring function
def color : ℕ → Color := sorry

-- Define the properties of the coloring function
def valid_coloring (color : ℕ → Color) : Prop :=
  ∀ (m n : ℕ), m > 1 → n > 1 → color m ≠ color n → 
    color (m * n) ≠ color m ∧ color (m * n) ≠ color n

-- Theorem: It is not possible to color all natural numbers greater than 1 as described
theorem no_valid_coloring : ¬ ∃ (color : ℕ → Color), valid_coloring color :=
by
  sorry

end no_valid_coloring_l123_123989


namespace smallest_k_sqrt_diff_l123_123152

-- Define a predicate for the condition that the square root difference is less than 1
def sqrt_diff_less_than_one (a b : ℕ) : Prop :=
  |real.sqrt a - real.sqrt b| < 1

-- Define the main theorem that encapsulates the problem statement
theorem smallest_k_sqrt_diff (cards : Finset ℕ) (h : cards = Finset.range 2016) : 
  ∃ k : ℕ, k = 45 ∧ ∀ s : Finset ℕ, s.card = k →
    ∃ (x y ∈ s), x ≠ y ∧ sqrt_diff_less_than_one x y :=
begin
  sorry
end

end smallest_k_sqrt_diff_l123_123152


namespace smallest_four_digit_multiple_of_18_l123_123760

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n = 1008 ∧ (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ 
                                ∀ m : ℕ, ((1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0)) → 1008 ≤ m :=
by
  sorry

end smallest_four_digit_multiple_of_18_l123_123760


namespace quilt_patch_cost_is_correct_l123_123513

noncomputable def quilt_area : ℕ := 16 * 20

def patch_area : ℕ := 4

def first_10_patch_cost : ℕ := 10

def discount_patch_cost : ℕ := 5

def total_patches (quilt_area patch_area : ℕ) : ℕ := quilt_area / patch_area

def cost_for_first_10 (first_10_patch_cost : ℕ) : ℕ := 10 * first_10_patch_cost

def cost_for_discounted (total_patches first_10_patch_cost discount_patch_cost : ℕ) : ℕ :=
  (total_patches - 10) * discount_patch_cost

def total_cost (cost_for_first_10 cost_for_discounted : ℕ) : ℕ :=
  cost_for_first_10 + cost_for_discounted

theorem quilt_patch_cost_is_correct :
  total_cost (cost_for_first_10 first_10_patch_cost)
             (cost_for_discounted (total_patches quilt_area patch_area) first_10_patch_cost discount_patch_cost) = 450 :=
by
  sorry

end quilt_patch_cost_is_correct_l123_123513


namespace four_points_no_obtuse_triangle_l123_123774

noncomputable def probability_no_obtuse_triangle : ℝ :=
1 / 64

theorem four_points_no_obtuse_triangle (A B C D : circle) :
  (∀ (P Q : circle) (PQ_angle : ℝ), PQ_angle < π/2) → 
  probability_no_obtuse_triangle = 1 / 64 :=
sorry

end four_points_no_obtuse_triangle_l123_123774


namespace triple_supplementary_angle_l123_123277

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l123_123277


namespace lateral_edge_length_l123_123089

theorem lateral_edge_length (a c : ℝ) :
  (∃ (S A B C : ℝ) (AS BS CS : ℝ), AS = BS ∧ BS = CS ∧
    dihedral_angle S A S C = 180 ∧
    AS = √((a ^ 2 + c ^ 2) / 4)) :=
sorry

end lateral_edge_length_l123_123089


namespace probability_no_obtuse_triangle_is_9_over_64_l123_123813

noncomputable def probability_no_obtuse_triangle (A0 A1 A2 A3 : ℝ) : ℝ :=
  -- Define the probabilistic model according to the problem conditions
  -- Assuming A0, A1, A2, and A3 are positions of points on the circle parametrized by angles in radians
  let θ := real.angle.lower_half (A1 - A0) in
  let prob_A2 := (π - θ) / (2 * π) in
  let prob_A3 := (π - θ) / (2 * π) in
  (1 / 2) * (prob_A2 * prob_A3)

theorem probability_no_obtuse_triangle_is_9_over_64 :
  probability_no_obtuse_triangle A0 A1 A2 A3 = 9 / 64 :=
by sorry

end probability_no_obtuse_triangle_is_9_over_64_l123_123813


namespace point_of_intersection_of_asymptotes_l123_123746

def f (x : ℝ) : ℝ := (x^2 - 6 * x + 8) / (x^2 - 6 * x + 9)

theorem point_of_intersection_of_asymptotes : ∃ x y : ℝ, x = 3 ∧ y = 1 ∧ 
  ∀ (ε > 0), ∃ δ > 0, ∀ (x' : ℝ), 0 < abs (x' - x) < δ → abs (f x' - y) < ε :=
by 
  sorry

end point_of_intersection_of_asymptotes_l123_123746


namespace total_length_S_l123_123119

open Real

def S : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ abs (abs (abs x - 2) - 1) + abs (abs (abs y - 2) - 1) = 1}

theorem total_length_S : ∑ (p ∈ S), (euclidean_metric.dist p p) = 64 * sqrt 2 :=
by
  sorry

end total_length_S_l123_123119


namespace max_gcd_bn_bnp1_l123_123529

def b_n (n : ℕ) : ℤ := (7 ^ n - 4) / 3
def b_n_plus_1 (n : ℕ) : ℤ := (7 ^ (n + 1) - 4) / 3

theorem max_gcd_bn_bnp1 (n : ℕ) : ∃ d_max : ℕ, (∀ d : ℕ, (gcd (b_n n) (b_n_plus_1 n) ≤ d) → d ≤ d_max) ∧ d_max = 3 :=
sorry

end max_gcd_bn_bnp1_l123_123529


namespace no_obtuse_triangle_probability_eq_l123_123784

noncomputable def probability_no_obtuse_triangle : ℝ :=
  let num_points := 4
  -- Condition (1): Four points are chosen uniformly at random on a circle.
  -- Condition (2): An obtuse angle occurs if the minor arc exceeds π/2.
  9 / 64

theorem no_obtuse_triangle_probability_eq :
  let num_points := 4
  ∀ (points : Fin num_points → ℝ), 
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∠ (points i, points j, points k) < π / 2) →
  probability_no_obtuse_triangle = 9 / 64 :=
by
  sorry

end no_obtuse_triangle_probability_eq_l123_123784


namespace awards_distribution_proof_l123_123578

def awards_distribution_ways := 
  ∃ (n : ℕ), 
    n = 300 ∧ 
    ∃ (awards : Finset ℕ) (students : Finset ℕ), 
      awards.card = 6 ∧ students.card = 3 ∧ 
      ∀ (f : ℕ → ℕ), 
        (∀ a ∈ awards, ∃ s ∈ students, f a = s) ∧ 
        (∀ s ∈ students, (Finset.filter (λ a, f a = s) awards).card > 0)

theorem awards_distribution_proof : awards_distribution_ways :=
  sorry

end awards_distribution_proof_l123_123578


namespace sum_of_coordinates_point_on_h_l123_123472

noncomputable def g : ℝ → ℝ := sorry
noncomputable def h (x : ℝ) : ℝ := (g(x) - 2)^2

theorem sum_of_coordinates_point_on_h :
  g 4 = 8 →
  h 4 = (g 4 - 2)^2 →
  4 + h 4 = 40 :=
by
  intros h1 h2
  rw [h2]
  rw [h1]
  simp
  norm_num
  rfl

end sum_of_coordinates_point_on_h_l123_123472


namespace product_of_integers_l123_123590

theorem product_of_integers (a b : ℤ) (h1 : Int.gcd a b = 12) (h2 : Int.lcm a b = 60) : a * b = 720 :=
sorry

end product_of_integers_l123_123590


namespace smallest_four_digit_multiple_of_18_l123_123761

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n = 1008 ∧ (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ 
                                ∀ m : ℕ, ((1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0)) → 1008 ≤ m :=
by
  sorry

end smallest_four_digit_multiple_of_18_l123_123761


namespace arithmetic_sequence_sum_l123_123975

variable {a : ℕ → ℕ}
variable [ArithmeticSequence a]

theorem arithmetic_sequence_sum :
  a 4 + a 6 + a 2010 + a 2012 = 8 → (∑ i in range (2015 + 1), a i) = 4030 :=
by
  sorry

end arithmetic_sequence_sum_l123_123975


namespace sum_of_odd_integers_from_15_to_45_l123_123632

-- Define the arithmetic series of odd integers from 15 to 45
def arithmetic_series : List ℤ := List.range' 15 31 |> List.filter (λ n, n % 2 = 1)

-- Define the sum of the series
def sum_arithmetic_series : ℤ :=
  arithmetic_series.sum

-- Theorem stating the sum of the odd integers from 15 to 45 is 480
theorem sum_of_odd_integers_from_15_to_45 :
  sum_arithmetic_series = 480 :=
by
  -- This proof is left as an exercise
  sorry

end sum_of_odd_integers_from_15_to_45_l123_123632


namespace vec_c_is_linear_comb_of_a_b_l123_123872

structure Vec2 :=
  (x : ℝ)
  (y : ℝ)

def a := Vec2.mk 1 2
def b := Vec2.mk (-2) 3
def c := Vec2.mk 4 1

theorem vec_c_is_linear_comb_of_a_b : c = Vec2.mk (2 * a.x - b.x) (2 * a.y - b.y) :=
  by
    sorry

end vec_c_is_linear_comb_of_a_b_l123_123872


namespace sum_of_odd_integers_from_15_to_45_l123_123631

-- Define the arithmetic series of odd integers from 15 to 45
def arithmetic_series : List ℤ := List.range' 15 31 |> List.filter (λ n, n % 2 = 1)

-- Define the sum of the series
def sum_arithmetic_series : ℤ :=
  arithmetic_series.sum

-- Theorem stating the sum of the odd integers from 15 to 45 is 480
theorem sum_of_odd_integers_from_15_to_45 :
  sum_arithmetic_series = 480 :=
by
  -- This proof is left as an exercise
  sorry

end sum_of_odd_integers_from_15_to_45_l123_123631


namespace smallest_positive_four_digit_multiple_of_18_l123_123764

-- Define the predicates for conditions
def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def multiple_of_18 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 18 * k

-- Define the main theorem
theorem smallest_positive_four_digit_multiple_of_18 : 
  ∃ n : ℕ, four_digit_number n ∧ multiple_of_18 n ∧ ∀ m : ℕ, four_digit_number m ∧ multiple_of_18 m → n ≤ m :=
begin
  use 1008,
  split,
  { -- proof that 1008 is a four-digit number
    split,
    { linarith, },
    { linarith, }
  },

  split,
  { -- proof that 1008 is a multiple of 18
    use 56,
    norm_num,
  },

  { -- proof that 1008 is the smallest such number
    intros m h1 h2,
    have h3 := Nat.le_of_lt,
    sorry, -- Detailed proof would go here
  }
end

end smallest_positive_four_digit_multiple_of_18_l123_123764


namespace shortest_altitude_of_right_triangle_l123_123219

-- Define the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Given conditions about the triangle
def right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the area of the triangle
def area (a b : ℕ) : ℝ := (1/2) * a * b

-- Define the altitude
noncomputable def altitude (area : ℝ) (c : ℕ) : ℝ := (2 * area) / c

-- Proving the length of the shortest altitude
theorem shortest_altitude_of_right_triangle 
  (h : ℝ) 
  (ha : a = 9) 
  (hb : b = 12) 
  (hc : c = 15) 
  (rt : right_triangle a b c) : 
  altitude (area a b) c = 7.2 :=
sorry

end shortest_altitude_of_right_triangle_l123_123219


namespace smallest_k_proof_l123_123150

noncomputable def smallest_k : ℕ :=
  45

theorem smallest_k_proof :
  ∀ (A : Finset ℕ), A.card = smallest_k →
    (∃ x y ∈ A, x ≠ y ∧ |Real.sqrt x - Real.sqrt y| < 1) :=
by
  sorry

end smallest_k_proof_l123_123150


namespace not_possible_coloring_l123_123988

def color : Nat → Option ℕ := sorry

def all_colors_used (f : Nat → Option ℕ) : Prop := 
  (∃ n, f n = some 0) ∧ (∃ n, f n = some 1) ∧ (∃ n, f n = some 2)

def valid_coloring (f : Nat → Option ℕ) : Prop :=
  ∀ (a b : Nat), 1 < a → 1 < b → f a ≠ f b → f (a * b) ≠ f a ∧ f (a * b) ≠ f b

theorem not_possible_coloring : ¬ (∃ f : Nat → Option ℕ, all_colors_used f ∧ valid_coloring f) := 
sorry

end not_possible_coloring_l123_123988


namespace find_a_for_tangent_l123_123854

theorem find_a_for_tangent (a : ℝ) (h : y = 3 * x - log (x + a)) (tangent_eq: tangent_at (0, 0) y = 2 * x) : a = 1 :=
sorry

end find_a_for_tangent_l123_123854


namespace find_S12_l123_123082

variable {a : Nat → Int} -- representing the arithmetic sequence {a_n}
variable {S : Nat → Int} -- representing the sums of the first n terms, S_n

-- Condition: a_1 = -9
axiom a1_def : a 1 = -9

-- Condition: (S_n / n) forms an arithmetic sequence
axiom arithmetic_s : ∃ d : Int, ∀ n : Nat, S n / n = -9 + (n - 1) * d

-- Condition: 2 = S9 / 9 - S7 / 7
axiom condition : S 9 / 9 - S 7 / 7 = 2

-- We want to prove: S_12 = 36
theorem find_S12 : S 12 = 36 := 
sorry

end find_S12_l123_123082


namespace no_obtuse_triangle_l123_123793

-- Conditions
def points_on_circle_uniformly_at_random (n : ℕ) : Prop :=
  ∀ i < n, ∀ j < n, i ≠ j → ∃ θ_ij : ℝ, 0 ≤ θ_ij ∧ θ_ij ≤ π

-- Theorem statement
theorem no_obtuse_triangle (hn : points_on_circle_uniformly_at_random 4) :
  let p := \left(\frac{1}{2}\right)^6 in
  p = \frac{1}{64} :=
sorry

end no_obtuse_triangle_l123_123793


namespace total_amount_shared_l123_123355

theorem total_amount_shared (A B C : ℕ) : 
  let ratio_A := 2
  let ratio_B := 3
  let ratio_C := 5 in
  let Amanda_portion := 30 in
  let multiplier := Amanda_portion / ratio_A in
  A = ratio_A * multiplier ∧ 
  B = ratio_B * multiplier ∧ 
  C = ratio_C * multiplier ∧
  A + B + C = 150 :=
by {
  sorry
}

end total_amount_shared_l123_123355


namespace fraction_tips_l123_123351

theorem fraction_tips {S : ℝ} (H1 : S > 0) (H2 : tips = (7 / 3 : ℝ) * S) (H3 : bonuses = (2 / 5 : ℝ) * S) :
  (tips / (S + tips + bonuses)) = (5 / 8 : ℝ) :=
by
  sorry

end fraction_tips_l123_123351


namespace angle_triple_supplement_l123_123271

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l123_123271


namespace expected_sides_rectangle_expected_sides_polygon_l123_123625

-- Part (a)
theorem expected_sides_rectangle (k : ℕ) (h : k > 0) : (4 + 4 * k) / (k + 1) → 4 :=
by sorry

-- Part (b)
theorem expected_sides_polygon (n k : ℕ) (h : n > 2) (h_k : k ≥ 0) : (n + 4 * k) / (k + 1) = (n + 4 * k) / (k + 1) :=
by sorry

end expected_sides_rectangle_expected_sides_polygon_l123_123625


namespace tf_questions_count_l123_123696

-- Define the conditions of the problem
def valid_tf_combinations (n : ℕ) : ℕ := 2^n - 2
def mc_combinations : ℕ := 4 * 4
def total_combinations (n : ℕ) : ℕ := valid_tf_combinations(n) * mc_combinations

-- The main statement to prove
theorem tf_questions_count : ∃ n : ℕ, total_combinations(n) = 96 ∧ n = 3 :=
by
  sorry

end tf_questions_count_l123_123696


namespace problem_I_problem_II_l123_123476

-- Define triangle and conditions
structure Triangle :=
  (A B C : ℝ)  -- Angles A, B, C in radians
  (a b c : ℝ)  -- Sides a, b, c opposite to angles A, B, C respectively
  (condition_2acosC_minus_c_eq_2b : 2 * a * Real.cos C - c = 2 * b)

-- Prove the magnitude of angle A
theorem problem_I (T : Triangle) : T.A = 2 * Real.pi / 3 := sorry

-- Define extended conditions for problem II
structure TriangleWithBisector extends Triangle :=
  (c_value : T.c = Real.sqrt 2)
  (BD_value : Real.sqrt 3)

-- Prove the value of side a given additional conditions
theorem problem_II (TWB : TriangleWithBisector) : TWB.to_Triangle.a = Real.sqrt 2 := sorry

end problem_I_problem_II_l123_123476


namespace max_students_equal_distribution_l123_123607

-- Define the number of pens and pencils
def pens : ℕ := 1008
def pencils : ℕ := 928

-- Define the problem statement which asks for the GCD of the given numbers
theorem max_students_equal_distribution : Nat.gcd pens pencils = 16 :=
by 
  -- Lean's gcd computation can be used to confirm the result
  sorry

end max_students_equal_distribution_l123_123607


namespace book_length_l123_123905

variable (length width perimeter : ℕ)

theorem book_length
  (h1 : perimeter = 100)
  (h2 : width = 20)
  (h3 : perimeter = 2 * (length + width)) :
  length = 30 :=
by sorry

end book_length_l123_123905


namespace smallest_k_proof_l123_123147

noncomputable def smallest_k : ℕ :=
  45

theorem smallest_k_proof :
  ∀ (A : Finset ℕ), A.card = smallest_k →
    (∃ x y ∈ A, x ≠ y ∧ |Real.sqrt x - Real.sqrt y| < 1) :=
by
  sorry

end smallest_k_proof_l123_123147


namespace point_of_intersection_of_asymptotes_l123_123744

def f (x : ℝ) : ℝ := (x^2 - 6 * x + 8) / (x^2 - 6 * x + 9)

theorem point_of_intersection_of_asymptotes : ∃ x y : ℝ, x = 3 ∧ y = 1 ∧ 
  ∀ (ε > 0), ∃ δ > 0, ∀ (x' : ℝ), 0 < abs (x' - x) < δ → abs (f x' - y) < ε :=
by 
  sorry

end point_of_intersection_of_asymptotes_l123_123744


namespace avg_people_moving_to_florida_per_hour_l123_123979

theorem avg_people_moving_to_florida_per_hour (people : ℕ) (days : ℕ) (hours_per_day : ℕ) 
  (h1 : people = 3000) (h2 : days = 5) (h3 : hours_per_day = 24) : 
  people / (days * hours_per_day) = 25 := by
  sorry

end avg_people_moving_to_florida_per_hour_l123_123979


namespace angle_triple_supplement_l123_123270

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l123_123270


namespace alcohol_percentage_new_mixture_l123_123322

theorem alcohol_percentage_new_mixture (initial_volume new_volume alcohol_initial : ℝ)
  (h1 : initial_volume = 15)
  (h2 : alcohol_initial = 0.20 * initial_volume)
  (h3 : new_volume = initial_volume + 5) :
  (alcohol_initial / new_volume) * 100 = 15 := by
  sorry

end alcohol_percentage_new_mixture_l123_123322


namespace angle_triple_supplement_l123_123257

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123257


namespace sum_div_9_remainder_l123_123286

theorem sum_div_9_remainder :
  ∑ i in Finset.range 21, i % 9 = 4 :=
  sorry

end sum_div_9_remainder_l123_123286


namespace carriages_people_equation_l123_123081

theorem carriages_people_equation (x : ℕ) :
  3 * (x - 2) = 2 * x + 9 :=
sorry

end carriages_people_equation_l123_123081


namespace perpendicular_lines_m_l123_123446

/-- Given two lines l1: (m-2)x + 3y + 2m = 0 and l2: x + my + 6 = 0,
    if l1 is perpendicular to l2, then m = 1/2.-/
theorem perpendicular_lines_m (m : ℝ) :
  let l1 := (m - 2) * x + 3 * y + 2 * m = 0
  let l2 := x + m * y + 6 = 0
  (∀ (x y : ℝ), l1 x y ∧ l2 x y → false) → m = 1 / 2 :=
by { sorry }

end perpendicular_lines_m_l123_123446


namespace banana_bunches_l123_123555

theorem banana_bunches
    (total_bananas : ℕ)
    (bunches_with_7 : ℕ)
    (bananas_per_bunch_with_7 : ℕ)
    (bunches_with_8 : ℕ)
    (bananas_per_bunch_with_8 : ℕ) :
    total_bananas = (bunches_with_7 * bananas_per_bunch_with_7) + (bunches_with_8 * bananas_per_bunch_with_8) →
    total_bananas = 83 →
    bunches_with_7 = 5 →
    bananas_per_bunch_with_7 = 7 →
    bananas_per_bunch_with_8 = 8 →
    bunches_with_8 = 6 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end banana_bunches_l123_123555


namespace prime_pair_divisibility_l123_123387

theorem prime_pair_divisibility (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (p ∣ (5^q + 1)) ∧ (q ∣ (5^p + 1)) ↔ (p, q) ∈ {(2, 2), (2, 13), (3, 3), (3, 7), (13, 2), (7, 3)} :=
by
  sorry

end prime_pair_divisibility_l123_123387


namespace sum_primes_between_20_and_40_l123_123637

theorem sum_primes_between_20_and_40 :
  23 + 29 + 31 + 37 = 120 :=
by
  sorry

end sum_primes_between_20_and_40_l123_123637


namespace geometric_progression_common_ratio_l123_123910

theorem geometric_progression_common_ratio (y r : ℝ) (h : (40 + y)^2 = (10 + y) * (90 + y)) :
  r = (40 + y) / (10 + y) → r = (90 + y) / (40 + y) → r = 5 / 3 :=
by
  sorry

end geometric_progression_common_ratio_l123_123910


namespace insert_threes_divisible_by_19_l123_123167

theorem insert_threes_divisible_by_19 (k : ℕ) : 
  (∃ n : ℕ, n = 12000 + 8 -> (∃ r : ℕ, (n + r) = 19 * b)) :=
begin
  let base_number := 12008,
  ∃ m : ℕ, insert_threes(base_number, k) = m,
  use 19,
  let n := Iterate_to_Insert5(base_number, 5, s),
  simp,
  sorry
end

end insert_threes_divisible_by_19_l123_123167


namespace inheritance_amount_l123_123111

theorem inheritance_amount (x : ℝ) (hx1 : 0.25 * x + 0.1 * x = 15000) : x = 42857 := 
by
  -- Proof omitted
  sorry

end inheritance_amount_l123_123111


namespace number_of_ordered_pairs_l123_123377

theorem number_of_ordered_pairs :
  { n : ℕ | ∃ m : ℕ, m ≥ n ∧ m^2 - n^2 = 180 }.to_finset.card = 3 :=
  sorry

end number_of_ordered_pairs_l123_123377


namespace range_f_on_interval_l123_123211

noncomputable def f (x : ℝ) : ℝ := -2 * Real.sin (x - Real.pi / 3)

theorem range_f_on_interval : set.range (λ x : {x : ℝ // 0 ≤ x ∧ x ≤ Real.pi}, f (x : ℝ)) = set.Icc (-2) (Real.sqrt 3) :=
by
  sorry

end range_f_on_interval_l123_123211


namespace checkerboard_black_squares_l123_123711

-- Define the checkerboard and the conditions
def is_red (x y : ℕ) : Prop := (x + y) % 2 = 0

-- Define the size of the checkerboard
def checkerboard_size : ℕ := 32

-- Define the function to count the number of black squares
def count_black_squares (size : ℕ) : ℕ :=
  let total_squares := size * size in
  let red_squares := (total_squares + 1) / 2 in
  total_squares - red_squares

-- Prove that the number of black squares in a 32x32 checkerboard is 511
theorem checkerboard_black_squares : count_black_squares checkerboard_size = 511 :=
by {
  -- Add proof steps here later
  sorry
}

end checkerboard_black_squares_l123_123711


namespace insert_threes_divisible_by_19_l123_123168

theorem insert_threes_divisible_by_19 (k : ℕ) : 
  (∃ n : ℕ, n = 12000 + 8 -> (∃ r : ℕ, (n + r) = 19 * b)) :=
begin
  let base_number := 12008,
  ∃ m : ℕ, insert_threes(base_number, k) = m,
  use 19,
  let n := Iterate_to_Insert5(base_number, 5, s),
  simp,
  sorry
end

end insert_threes_divisible_by_19_l123_123168


namespace four_points_no_obtuse_triangle_l123_123779

noncomputable def probability_no_obtuse_triangle : ℝ :=
1 / 64

theorem four_points_no_obtuse_triangle (A B C D : circle) :
  (∀ (P Q : circle) (PQ_angle : ℝ), PQ_angle < π/2) → 
  probability_no_obtuse_triangle = 1 / 64 :=
sorry

end four_points_no_obtuse_triangle_l123_123779


namespace angle_triple_supplement_l123_123265

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123265


namespace smallest_k_sqrt_diff_l123_123154

-- Define a predicate for the condition that the square root difference is less than 1
def sqrt_diff_less_than_one (a b : ℕ) : Prop :=
  |real.sqrt a - real.sqrt b| < 1

-- Define the main theorem that encapsulates the problem statement
theorem smallest_k_sqrt_diff (cards : Finset ℕ) (h : cards = Finset.range 2016) : 
  ∃ k : ℕ, k = 45 ∧ ∀ s : Finset ℕ, s.card = k →
    ∃ (x y ∈ s), x ≠ y ∧ sqrt_diff_less_than_one x y :=
begin
  sorry
end

end smallest_k_sqrt_diff_l123_123154


namespace find_a5_in_factorial_base_l123_123595

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem find_a5_in_factorial_base (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ)
  (h873 : 873 = a₁ + a₂ * factorial 2 + a₃ * factorial 3 + a₄ * factorial 4 + a₅ * factorial 5 + a₆ * factorial 6)
  (h_range : ∀ k, k ∈ {1, 2, 3, 4, 5, 6} → 0 ≤ a₆ ∧ a₆ ≤ 6 ∧
                   0 ≤ a₅ ∧ a₅ ≤ 5 ∧
                   0 ≤ a₄ ∧ a₄ ≤ 4 ∧
                   0 ≤ a₃ ∧ a₃ ≤ 3 ∧
                   0 ≤ a₂ ∧ a₂ ≤ 2 ∧
                   0 ≤ a₁ ∧ a₁ ≤ 1) : a₅ = 1 := 
sorry

end find_a5_in_factorial_base_l123_123595


namespace sum_primes_between_20_and_40_l123_123649

open Nat

def primesBetween20And40 : List Nat := [23, 29, 31, 37]

theorem sum_primes_between_20_and_40 :
  (primesBetween20And40.sum = 120) :=
by
  sorry

end sum_primes_between_20_and_40_l123_123649


namespace function_increasing_interval_l123_123723

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem function_increasing_interval :
  ∀ x : ℝ, x > 0 → deriv f x > 0 := 
sorry

end function_increasing_interval_l123_123723


namespace largest_class_students_l123_123071

theorem largest_class_students (n1 n2 n3 n4 n5 : ℕ) (h1 : n1 = x) (h2 : n2 = x - 2) (h3 : n3 = x - 4) (h4 : n4 = x - 6) (h5 : n5 = x - 8) (h_sum : n1 + n2 + n3 + n4 + n5 = 140) : x = 32 :=
by {
  sorry
}

end largest_class_students_l123_123071


namespace angle_triple_supplement_l123_123264

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123264


namespace smallest_positive_period_of_f_min_value_of_f_on_interval_l123_123432

noncomputable def f (x : ℝ) : ℝ := sin(π/2 - x) * sin x - sqrt 3 * (sin x)^2

theorem smallest_positive_period_of_f :
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f(x + T) = f x) ∧ (∀ T' : ℝ, (∀ x : ℝ, f(x + T') = f x) → T ≤ T') :=
sorry

theorem min_value_of_f_on_interval :
  ∃ m : ℝ, (∀ x ∈ set.Icc (0:ℝ) (π/4), f x ≥ m) ∧ (∃ x ∈ set.Icc (0:ℝ) (π/4), f x = m) ∧ m = (1 - sqrt 3) / 2 :=
sorry

end smallest_positive_period_of_f_min_value_of_f_on_interval_l123_123432


namespace angle_triple_supplement_l123_123262

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123262


namespace train_truck_load_l123_123617

variables (x y : ℕ)

def transport_equations (x y : ℕ) : Prop :=
  (2 * x + 5 * y = 120) ∧ (8 * x + 10 * y = 440)

def tonnage (x y : ℕ) : ℕ :=
  5 * x + 8 * y

theorem train_truck_load
  (x y : ℕ)
  (h : transport_equations x y) :
  tonnage x y = 282 :=
sorry

end train_truck_load_l123_123617


namespace angle_triple_supplementary_l123_123247

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l123_123247


namespace acute_triangle_problem_l123_123924

noncomputable def triangle_A : Type := sorry -- add definition for acute triangle where ∠A corresponds to sinA value
noncomputable def triangle_A_area : triangle_A -> ℝ := sorry -- function to get the area of the triangle
noncomputable def side_length_a : triangle_A -> ℝ := sorry -- function to get side length a
noncomputable def side_length_b : triangle_A -> ℝ := sorry -- function to get side length b
noncomputable def side_length_c : triangle_A -> ℝ := sorry -- function to get side length c
noncomputable def angle_sin_value_a : triangle_A -> ℝ := sorry -- function to get sinA

/-- 
Prove that for an acute triangle ABC with given conditions:
1. $\sin(A) = \frac{2\sqrt{2}}{3}$
2. $a = 2$
3. Area of $\triangle ABC = \sqrt{2}$

Then:
1. $\tan^{2} \frac{B+C}{2} + \sin^{2} \frac{A}{2} = \frac{7}{3}$
2. $b = \sqrt{3}$
-/

theorem acute_triangle_problem (t : triangle_A) (sin_a : angle_sin_value_a t = (2 * real.sqrt 2) / 3) 
  (area : triangle_A_area t = real.sqrt 2) (a_eq : side_length_a t = 2) : 
  (tan_squared_half_angle_sum_BC : real.tan (angle_sum_BC_half t / 2)^2 + real.sin (angle_A_half t / 2)^2 = 7/3) ∧
  (b_eq : side_length_b t = real.sqrt 3) :=
sorry

end acute_triangle_problem_l123_123924


namespace sum_of_a_b_l123_123085

def symmetric_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

theorem sum_of_a_b (a b : ℝ) (h : symmetric_x_axis (3, a) (b, 4)) : a + b = -1 :=
by
  sorry

end sum_of_a_b_l123_123085


namespace math_problem_l123_123651

theorem math_problem 
  (num := 1 * 2 * 3 * 4 * 5 * 6 * 7)
  (den := 1 + 2 + 3 + 4 + 5 + 6 + 7) :
  (num / den) = 180 :=
by
  sorry

end math_problem_l123_123651


namespace sum_abs_a_leq_threshold_l123_123833

theorem sum_abs_a_leq_threshold (n : ℕ) (a : Fin n → ℂ)
  (h : ∀ I : Finset (Fin n), I.Nonempty → ∥(∏ j in I, (1 + a j)) - 1∥ ≤ 1/2) :
  ∑ j, ∥a j∥ ≤ 1/2 + Real.log 2 + Real.pi/3 := sorry

end sum_abs_a_leq_threshold_l123_123833


namespace ArcherInGoldenArmorProof_l123_123928

-- Definitions of the problem
variables (soldiers : Nat) (archers soldiersInGolden : Nat) 
variables (soldiersInBlack archersInGolden archersInBlack swordsmenInGolden swordsmenInBlack : Nat)
variables (truthfulSwordsmenInBlack lyingArchersInBlack lyingSwordsmenInGold truthfulArchersInGold : Nat)
variables (yesToGold yesToArcher yesToMonday : Nat)

-- Given conditions
def ProblemStatement : Prop :=
  soldiers = 55 ∧
  yesToGold = 44 ∧
  yesToArcher = 33 ∧
  yesToMonday = 22 ∧
  soldiers = archers + (soldiers - archers) ∧
  soldiers = soldiersInGolden + soldiersInBlack ∧
  archers = archersInGolden + archersInBlack ∧
  soldiersInGolden = archersInGolden + swordsmenInGolden ∧
  soldiersInBlack = archersInBlack + swordsmenInBlack ∧
  truthfulSwordsmenInBlack = swordsmenInBlack ∧
  lyingArchersInBlack = archersInBlack ∧
  lyingSwordsmenInGold = swordsmenInGolden ∧
  truthfulArchersInGold = archersInGolden ∧
  yesToGold = truthfulArchersInGold + (swordsmenInGolden + archersInBlack) ∧
  yesToArcher = truthfulArchersInGold + lyingSwordsmenInGold

-- Conclusion
def Conclusion : Prop :=
  archersInGolden = 22

-- Proof statement
theorem ArcherInGoldenArmorProof : ProblemStatement → Conclusion :=
by
  sorry

end ArcherInGoldenArmorProof_l123_123928


namespace det_A2_minus_3A_l123_123113

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  !![2, 4;
    3, 2]

theorem det_A2_minus_3A : det (A * A - (3 : ℤ) • A) = 88 := by
  sorry

end det_A2_minus_3A_l123_123113


namespace smallest_n_for_integer_S_n_l123_123544

def b := 8
def S_n (n : ℕ) := n * b^(n - 1) * (1 + 1/2 + 1/3 + 1/4 + 1/5 + 1/6 + 1/7)

theorem smallest_n_for_integer_S_n : ∃ n : ℕ, S_n n ∈ ℤ ∧ ∀ m : ℕ, m < n → S_n m ∉ ℤ :=
begin
  use 105,
  -- Proof outline:
  -- 1. Define b = 8
  -- 2. Define S_n as given
  -- 3. Show that S_105 is an integer and for any m < 105, S_m is not an integer
  sorry
end

end smallest_n_for_integer_S_n_l123_123544


namespace spatial_relationship_l123_123036

open Plane Line

variable {α β γ : Plane} {l m : Line}

theorem spatial_relationship 
  (h1 : m ⊆ α)
  (h2 : m ⟂ γ)
  (h3 : l = β ⊓ γ)
  (h4 : l ∥ α) :
  α ⟂ γ ∧ l ⟂ m :=
sorry

end spatial_relationship_l123_123036


namespace impossible_event_l123_123652

noncomputable def EventA := ∃ (ω : ℕ), ω = 0 ∨ ω = 1
noncomputable def EventB := ∃ (t : ℤ), t >= 0
noncomputable def Bag := {b : String // b = "White"}
noncomputable def EventC := ∀ (x : Bag), x.val ≠ "Red"
noncomputable def EventD := ∀ (a b : ℤ), (a > 0 ∧ b < 0) → a > b

theorem impossible_event:
  (EventA ∧ EventB ∧ EventD) →
  EventC :=
by
  sorry

end impossible_event_l123_123652


namespace threes_inserted_divisible_by_19_l123_123165

theorem threes_inserted_divisible_by_19 (n : ℕ) : 
  let num := 120 * 10^(n+1) + 3 * ((10^n - 1) / 9) * 10 + 8 
  in num % 19 = 0 :=
by
  sorry

end threes_inserted_divisible_by_19_l123_123165


namespace total_repairs_cost_eq_l123_123159

-- Assume the initial cost of the scooter is represented by a real number C.
variable (C : ℝ)

-- Given conditions
def spent_on_first_repair := 0.05 * C
def spent_on_second_repair := 0.10 * C
def spent_on_third_repair := 0.07 * C

-- Total repairs expenditure
def total_repairs := spent_on_first_repair C + spent_on_second_repair C + spent_on_third_repair C

-- Selling price and profit
def selling_price := 1.25 * C
def profit := 1500
def profit_calc := selling_price C - (C + total_repairs C)

-- Statement to be proved: The total repairs is equal to $11,000.
theorem total_repairs_cost_eq : total_repairs 50000 = 11000 := by
  sorry

end total_repairs_cost_eq_l123_123159


namespace number_of_archers_in_golden_armor_l123_123973

-- Define the problem context
structure Soldier where
  is_archer : Bool
  wears_golden_armor : Bool

def truth_teller (s : Soldier) (is_black_armor : Bool) : Bool :=
  if s.is_archer then s.wears_golden_armor = is_black_armor
  else s.wears_golden_armor ≠ is_black_armor

def response (s : Soldier) (question : String) (is_black_armor : Bool) : Bool :=
  match question with
  | "Are you wearing golden armor?" => if truth_teller s is_black_armor then s.wears_golden_armor else ¬s.wears_golden_armor
  | "Are you an archer?" => if truth_teller s is_black_armor then s.is_archer else ¬s.is_archer
  | "Is today Monday?" => if truth_teller s is_black_armor then True else False -- An assumption that today not being Monday means False
  | _ => False

-- Problem condition setup
def total_soldiers : Nat := 55
def golden_armor_yes : Nat := 44
def archer_yes : Nat := 33
def monday_yes : Nat := 22

-- Define the main theorem
theorem number_of_archers_in_golden_armor :
  ∃ l : List Soldier,
    l.length = total_soldiers ∧
    l.countp (λ s => response s "Are you wearing golden armor?" True) = golden_armor_yes ∧
    l.countp (λ s => response s "Are you an archer?" True) = archer_yes ∧
    l.countp (λ s => response s "Is today Monday?" True) = monday_yes ∧
    l.countp (λ s => s.is_archer ∧ s.wears_golden_armor) = 22 :=
sorry

end number_of_archers_in_golden_armor_l123_123973


namespace rational_terms_binomial_expansion_l123_123083

theorem rational_terms_binomial_expansion (x : ℚ) :
  let T1 := x^4,
      T5 := (35 / 8) * x,
      T9 := 1 / (256 * x^2) in
  ∀ T, (T = T1 ∨ T = T5 ∨ T = T9) →
  ∃ n : ℕ, n = 8 ∧
    (T = (√x + 1 / (2 * x^(1/4)))^n) → -- Expanding binomials implicitly
  sorry

end rational_terms_binomial_expansion_l123_123083


namespace quilt_patch_cost_is_correct_l123_123514

noncomputable def quilt_area : ℕ := 16 * 20

def patch_area : ℕ := 4

def first_10_patch_cost : ℕ := 10

def discount_patch_cost : ℕ := 5

def total_patches (quilt_area patch_area : ℕ) : ℕ := quilt_area / patch_area

def cost_for_first_10 (first_10_patch_cost : ℕ) : ℕ := 10 * first_10_patch_cost

def cost_for_discounted (total_patches first_10_patch_cost discount_patch_cost : ℕ) : ℕ :=
  (total_patches - 10) * discount_patch_cost

def total_cost (cost_for_first_10 cost_for_discounted : ℕ) : ℕ :=
  cost_for_first_10 + cost_for_discounted

theorem quilt_patch_cost_is_correct :
  total_cost (cost_for_first_10 first_10_patch_cost)
             (cost_for_discounted (total_patches quilt_area patch_area) first_10_patch_cost discount_patch_cost) = 450 :=
by
  sorry

end quilt_patch_cost_is_correct_l123_123514


namespace point_value_of_other_questions_is_4_l123_123300

theorem point_value_of_other_questions_is_4
  (total_points : ℕ)
  (total_questions : ℕ)
  (points_from_2_point_questions : ℕ)
  (other_questions : ℕ)
  (points_each_2_point_question : ℕ)
  (points_from_2_point_questions_calc : ℕ)
  (remaining_points : ℕ)
  (point_value_of_other_type : ℕ)
  : total_points = 100 →
    total_questions = 40 →
    points_each_2_point_question = 2 →
    other_questions = 10 →
    points_from_2_point_questions = 30 →
    points_from_2_point_questions_calc = points_each_2_point_question * points_from_2_point_questions →
    remaining_points = total_points - points_from_2_point_questions_calc →
    remaining_points = other_questions * point_value_of_other_type →
    point_value_of_other_type = 4 := by
  sorry

end point_value_of_other_questions_is_4_l123_123300


namespace rate_downstream_l123_123684

-- Define the man's rate in still water
def rate_still_water : ℝ := 24.5

-- Define the rate of the current
def rate_current : ℝ := 7.5

-- Define the man's rate upstream (unused in the proof but given in the problem)
def rate_upstream : ℝ := 17.0

-- Prove that the man's rate when rowing downstream is as stated given the conditions
theorem rate_downstream : rate_still_water + rate_current = 32 := by
  simp [rate_still_water, rate_current]
  norm_num

end rate_downstream_l123_123684


namespace perfect_square_iff_l123_123725

theorem perfect_square_iff (x y z : ℕ) (hx : x ≥ y) (hy : y ≥ z) (hz : z > 0) :
  ∃ k : ℕ, 4^x + 4^y + 4^z = k^2 ↔ ∃ b : ℕ, b > 0 ∧ x = 2 * b - 1 + z ∧ y = b + z :=
by
  sorry

end perfect_square_iff_l123_123725


namespace length_of_segment_KN_l123_123566

variable (R : ℝ) -- Radius of the circle
variable (A B X K N : EuclideanGeometry.Point) -- Points involved
variable [EuclideanGeometry IsEuclidean n] -- Natural number n indicating the dimension (usually 2)
variable (H1 : EuclideanGeometry.IsDiameter AB R) -- AB is a diameter of the circle with radius R
variable (H2 : EuclideanGeometry.OnCircle K R ∧ EuclideanGeometry.OnCircle N R) -- K and N lie on the same circle with radius R
variable (H3 : EuclideanGeometry.InSameHalfPlane K N AB) -- K and N are in the same half-plane wrt AB
variable (H4 : EuclideanGeometry.Angle K X A = 60 ∧ EuclideanGeometry.Angle N X B = 60) -- Given angles

theorem length_of_segment_KN : EuclideanGeometry.Distance K N = R := 
by 
  sorry

end length_of_segment_KN_l123_123566


namespace find_x_l123_123604

noncomputable def x_half_y (x y : ℚ) : Prop := x = (1 / 2) * y
noncomputable def y_third_z (y z : ℚ) : Prop := y = (1 / 3) * z

theorem find_x (x y z : ℚ) (h₁ : x_half_y x y) (h₂ : y_third_z y z) (h₃ : z = 100) :
  x = 16 + (2 / 3 : ℚ) :=
by
  sorry

end find_x_l123_123604


namespace sin_double_angle_l123_123046

theorem sin_double_angle (θ : ℝ) (h : sin θ + cos θ = 1 / 5) : sin (2 * θ) = -24 / 25 :=
by
  sorry

end sin_double_angle_l123_123046


namespace super_air_has_hamiltonian_cycle_l123_123917

def graph := Type

variables (G : graph) (V : finset ℕ) (E : V → V → Prop) [fintype V]

def degree (v : V) : ℕ := (finset.filter (λ u, E v u) V).card 

def concur_air_flight (E : V → V → Prop) (u v : V) : Prop :=
degree u + degree v ≥ 100

def exists_hamiltonian_cycle (E : V → V → Prop) : Prop :=
∃ (cycle : list V), (list.nodup cycle) ∧ (list.length cycle = fintype.card V) ∧ ∀ (i : ℕ), (E (cycle.nth_le i sorry) (cycle.nth_le ((i + 1) % (list.length cycle)) sorry))

theorem super_air_has_hamiltonian_cycle
  (h_concur : ∃ (H : graph) (V' : finset ℕ) (E' : V' → V' → Prop) [fintype V'], 
  (∃ (cycle : list V'), (list.nodup cycle) ∧ (list.length cycle = fintype.card V') ∧ ∀ (i : ℕ), concur_air_flight E' (cycle.nth_le i sorry) (cycle.nth_le ((i + 1) % (list.length cycle)) sorry)))
  : ∃ (cycle : list V), (list.nodup cycle) ∧ (list.length cycle = fintype.card V) ∧ ∀ (i : ℕ), (E (cycle.nth_le i sorry) (cycle.nth_le ((i + 1) % (list.length cycle)) sorry)) :=
sorry

end super_air_has_hamiltonian_cycle_l123_123917


namespace digit_at_2015th_is_8_l123_123581

noncomputable def even_numbers : ℕ → ℕ
| 0 := 2
| (n + 1) := even_numbers n + 2

noncomputable def digit_list : ℕ → List ℕ
| n := (even_numbers n).toString.toList.map (λ c, c.toNat - '0'.toNat)

noncomputable def concatenated_digits : ℕ → List ℕ
| 0 := []
| (n + 1) := concatenated_digits n ++ digit_list n

noncomputable def nth_digit (n : ℕ) : ℕ :=
(concatenated_digits n).getNth! (⟨2014, sorry⟩ : Fin (n + 1))

theorem digit_at_2015th_is_8 : nth_digit 2015 = 8 := sorry

end digit_at_2015th_is_8_l123_123581


namespace sum_of_incircle_radii_eq_CH_l123_123983

open Real

variables {A B C H : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace H]

-- Conditions of the problem
variable {triangle_ABC : Triangle}
variable (angle_ACB_is_right : ∠ A C B = 90)
variable (H_is_foot : ProjectiveGeometry.orthocenter A B C = H)

-- Define the inradius function for a right-angled triangle
noncomputable def inradius (a b c : ℝ) : ℝ := (a + b - c) / 2

-- Assume the lengths AC, BC, AB, AH, BH, CH
variables (AC BC AB AH BH CH : ℝ)
variables (hAC : triangle_ABC.side A C = AC)
variables (hBC : triangle_ABC.side B C = BC)
variables (hAB : triangle_ABC.side A B = AB)
variables (hAH : segment A H .length = AH)
variables (hBH : segment B H .length = BH)
variables (hCH : segment C H .length = CH)

theorem sum_of_incircle_radii_eq_CH :
  inradius AC BC AB + inradius BH CH BC + inradius AH CH AC = CH :=
sorry

end sum_of_incircle_radii_eq_CH_l123_123983


namespace roots_squared_sum_l123_123713

theorem roots_squared_sum (p q r : ℂ) (h : ∀ x : ℂ, 3 * x ^ 3 - 3 * x ^ 2 + 6 * x - 9 = 0 → x = p ∨ x = q ∨ x = r) :
  p^2 + q^2 + r^2 = -3 :=
by
  sorry

end roots_squared_sum_l123_123713


namespace archers_in_golden_armor_l123_123953

theorem archers_in_golden_armor (total_soldiers archers_swordsmen total_affirmations armor_affirmations archer_affirmations monday_affirmations : ℕ)
  (h1: total_soldiers = 55)
  (h2: armor_affirmations = 44)
  (h3: archer_affirmations = 33)
  (h4: monday_affirmations = 22)
  (h5: ∑ x in ({armor_affirmations, archer_affirmations, monday_affirmations} : finset ℕ), x = 99) 
  : ∃ (archers_in_golden : ℕ), archers_in_golden = 22 := by
  -- Definitions and theorems will go here
  sorry

end archers_in_golden_armor_l123_123953


namespace solution_proof_l123_123521

open EuclideanGeometry

noncomputable def problem_statement : Prop :=
  ∀ (A B C D E F : Point)
    (Γ ω : Circle)
    (I : Point),
    (triangle A B C) ∧
    (circumcircle A B C = Γ) ∧
    (incircle A B C = ω) ∧
    (ω.touches BC D) ∧
    (ω.touches CA E) ∧
    (ω.touches AB F) ∧
    (I = ω.center) →
    inverse_circle Γ I.ω.radius = nine_point_circle_triangle D E F

-- The 'sorry' here indicates that the proof is intentionally left out.
theorem solution_proof : problem_statement :=
by
  sorry

end solution_proof_l123_123521


namespace quilt_patch_cost_is_correct_l123_123512

noncomputable def quilt_area : ℕ := 16 * 20

def patch_area : ℕ := 4

def first_10_patch_cost : ℕ := 10

def discount_patch_cost : ℕ := 5

def total_patches (quilt_area patch_area : ℕ) : ℕ := quilt_area / patch_area

def cost_for_first_10 (first_10_patch_cost : ℕ) : ℕ := 10 * first_10_patch_cost

def cost_for_discounted (total_patches first_10_patch_cost discount_patch_cost : ℕ) : ℕ :=
  (total_patches - 10) * discount_patch_cost

def total_cost (cost_for_first_10 cost_for_discounted : ℕ) : ℕ :=
  cost_for_first_10 + cost_for_discounted

theorem quilt_patch_cost_is_correct :
  total_cost (cost_for_first_10 first_10_patch_cost)
             (cost_for_discounted (total_patches quilt_area patch_area) first_10_patch_cost discount_patch_cost) = 450 :=
by
  sorry

end quilt_patch_cost_is_correct_l123_123512


namespace cosine_inequality_l123_123163

-- Define the angles and the condition
variable (α β γ : ℝ)
variable (h : α + β + γ = π)

-- The target inequality to prove
theorem cosine_inequality : 
  cos (2 * α) + cos (2 * β) - cos (2 * γ) ≤ 3 / 2 := by
  sorry

end cosine_inequality_l123_123163


namespace vertical_lines_count_l123_123485

theorem vertical_lines_count (n : ℕ) 
  (h_intersections : (18 * n * (n - 1)) = 756) : 
  n = 7 :=
by 
  sorry

end vertical_lines_count_l123_123485


namespace count_special_n_l123_123198

-- Define the function F with the given conditions
def F : ℕ → ℕ
| 4*n := F (2*n) + F n
| 4*n + 2 := F (4*n) + 1
| 2*n + 1 := F (2*n) + 1
| 0 := 0  -- Add a base case for zero

-- Theorem statement translating the problem into Lean
theorem count_special_n (m : ℕ) : 
  ∃ count : ℕ, (∀ n < 2^m, F(4 * n) = F(3 * n) → count += 1) ∧ 
               count = F(2^(m+1)) := 
  sorry

end count_special_n_l123_123198


namespace m_range_l123_123545

noncomputable def f : ℝ → ℝ := sorry

lemma f_symmetry (x : ℝ) : f(2 + x) = f(-x) := sorry

lemma f_decreasing_on_interval (x : ℝ) (h : 1 ≤ x) : f x ≥ f (x + 1) := sorry

theorem m_range (m : ℝ) (h : f (1 - m) < f m) : m > 1/2 := sorry

end m_range_l123_123545


namespace intersection_points_l123_123058

noncomputable def f (x : ℝ) : ℝ :=
  if x % 2 ∈ (-1 : ℝ) .. 1 then |x % 2| else |(x % 2) - 2|

noncomputable def log4 (x : ℝ) : ℝ := real.log x / real.log 4

theorem intersection_points :
  ∃ n : ℕ, n = 6 ∧ 
    set.finite (set_of (λ x : ℝ, f x = log4 (abs x))).to_finset :=
  sorry

end intersection_points_l123_123058


namespace height_of_pyramid_l123_123680

theorem height_of_pyramid :
  let edge_cube := 6
  let edge_base_square_pyramid := 10
  let cube_volume := edge_cube ^ 3
  let sphere_volume := cube_volume
  let pyramid_volume := 2 * sphere_volume
  let base_area_square_pyramid := edge_base_square_pyramid ^ 2
  let height_pyramid := 12.96
  pyramid_volume = (1 / 3) * base_area_square_pyramid * height_pyramid :=
by
  sorry

end height_of_pyramid_l123_123680


namespace archers_in_golden_armor_l123_123952

theorem archers_in_golden_armor (total_soldiers archers_swordsmen total_affirmations armor_affirmations archer_affirmations monday_affirmations : ℕ)
  (h1: total_soldiers = 55)
  (h2: armor_affirmations = 44)
  (h3: archer_affirmations = 33)
  (h4: monday_affirmations = 22)
  (h5: ∑ x in ({armor_affirmations, archer_affirmations, monday_affirmations} : finset ℕ), x = 99) 
  : ∃ (archers_in_golden : ℕ), archers_in_golden = 22 := by
  -- Definitions and theorems will go here
  sorry

end archers_in_golden_armor_l123_123952


namespace elmer_more_than_penelope_l123_123158

-- Definitions based on the conditions
def penelope_food_daily : ℝ := 20
def greta_food_daily : ℝ := penelope_food_daily / 10
def milton_food_daily : ℝ := greta_food_daily / 100
def elmer_food_daily : ℝ := milton_food_daily * 4000

-- Theorem stating the required proof
theorem elmer_more_than_penelope : elmer_food_daily - penelope_food_daily = 60 :=
by
  sorry

end elmer_more_than_penelope_l123_123158


namespace number_of_archers_in_golden_armor_l123_123969

-- Define the problem context
structure Soldier where
  is_archer : Bool
  wears_golden_armor : Bool

def truth_teller (s : Soldier) (is_black_armor : Bool) : Bool :=
  if s.is_archer then s.wears_golden_armor = is_black_armor
  else s.wears_golden_armor ≠ is_black_armor

def response (s : Soldier) (question : String) (is_black_armor : Bool) : Bool :=
  match question with
  | "Are you wearing golden armor?" => if truth_teller s is_black_armor then s.wears_golden_armor else ¬s.wears_golden_armor
  | "Are you an archer?" => if truth_teller s is_black_armor then s.is_archer else ¬s.is_archer
  | "Is today Monday?" => if truth_teller s is_black_armor then True else False -- An assumption that today not being Monday means False
  | _ => False

-- Problem condition setup
def total_soldiers : Nat := 55
def golden_armor_yes : Nat := 44
def archer_yes : Nat := 33
def monday_yes : Nat := 22

-- Define the main theorem
theorem number_of_archers_in_golden_armor :
  ∃ l : List Soldier,
    l.length = total_soldiers ∧
    l.countp (λ s => response s "Are you wearing golden armor?" True) = golden_armor_yes ∧
    l.countp (λ s => response s "Are you an archer?" True) = archer_yes ∧
    l.countp (λ s => response s "Is today Monday?" True) = monday_yes ∧
    l.countp (λ s => s.is_archer ∧ s.wears_golden_armor) = 22 :=
sorry

end number_of_archers_in_golden_armor_l123_123969


namespace determinant_of_rotation_75_degrees_l123_123133

variable (θ : ℝ) (Q : Matrix (Fin 2) (Fin 2) ℝ)
def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

noncomputable def determinant_of_rotation_matrix (θ : ℝ) : ℝ :=
  Matrix.det (rotation_matrix θ)

theorem determinant_of_rotation_75_degrees :
  determinant_of_rotation_matrix 75 = 1 := by
  sorry

end determinant_of_rotation_75_degrees_l123_123133


namespace time_for_b_l123_123467

theorem time_for_b (A B C : ℚ) (H1 : A + B + C = 1/4) (H2 : A = 1/12) (H3 : C = 1/18) : B = 1/9 :=
by {
  sorry
}

end time_for_b_l123_123467


namespace maximum_value_trig_expression_l123_123739

theorem maximum_value_trig_expression :
  ∃ θ : ℝ, 0 < θ ∧ θ < π ∧ ∀ x, (0 < x ∧ x < π) → 
  (sin(x / 2)^2 * (1 + cos(x)^2) ≤ 2) := sorry

end maximum_value_trig_expression_l123_123739


namespace cos_B_eq_one_third_side_b_length_l123_123985

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {area : ℝ}

theorem cos_B_eq_one_third : 
  let m := (Real.cos A, -Real.sin A)
  let n := (Real.cos C, Real.sin C)
  (m.1 * n.1 + m.2 * n.2 = -(1 / 3)) ->
  Real.cos B = 1 / 3 :=
sorry

theorem side_b_length (h_cos_B : Real.cos B = 1 / 3)
                      (h_sin_B : Real.sin B = 2 * Real.sqrt 2 / 3)
                      (h_c : c = 2)
                      (h_area : area = 2 * Real.sqrt 2) :
  let a := 3 in
  b = Real.sqrt (a * a + c * c - 2 * a * c * (1 / 3)) :=
sorry

end cos_B_eq_one_third_side_b_length_l123_123985


namespace math_problem_l123_123999

noncomputable def exists_g (f : ℝ × ℝ → ℝ) :=
  (∀ x y z : ℝ, f (x, y) + f (y, z) + f (z, x) = 0) →
  ∃ g : ℝ → ℝ, ∀ x y : ℝ, f (x, y) = g x - g y

theorem math_problem (f : ℝ × ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x, y) + f (y, z) + f (z, x) = 0) :
  ∃ g : ℝ → ℝ, ∀ x y : ℝ, f (x, y) = g x - g y :=
exists_g f h

end math_problem_l123_123999


namespace smallest_four_digit_multiple_of_18_l123_123762

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n = 1008 ∧ (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ 
                                ∀ m : ℕ, ((1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0)) → 1008 ≤ m :=
by
  sorry

end smallest_four_digit_multiple_of_18_l123_123762


namespace threes_inserted_divisible_by_19_l123_123166

theorem threes_inserted_divisible_by_19 (n : ℕ) : 
  let num := 120 * 10^(n+1) + 3 * ((10^n - 1) / 9) * 10 + 8 
  in num % 19 = 0 :=
by
  sorry

end threes_inserted_divisible_by_19_l123_123166


namespace smallest_possible_value_of_b_l123_123203

theorem smallest_possible_value_of_b (a b x : ℕ) (h_pos_x : 0 < x)
  (h_gcd : Nat.gcd a b = x + 7)
  (h_lcm : Nat.lcm a b = x * (x + 7))
  (h_a : a = 56)
  (h_x : x = 21) :
  b = 294 := by
  sorry

end smallest_possible_value_of_b_l123_123203


namespace speed_of_first_train_l123_123597

-- Definitions of the conditions
def ratio_speed (speed1 speed2 : ℝ) := speed1 / speed2 = 7 / 8
def speed_of_second_train := 400 / 4

-- The theorem we want to prove
theorem speed_of_first_train (speed2 := speed_of_second_train) (h : ratio_speed S1 speed2) :
  S1 = 87.5 :=
by 
  sorry

end speed_of_first_train_l123_123597


namespace fraction_of_red_knights_magical_l123_123204

def total_knights : ℕ := 28
def red_fraction : ℚ := 3 / 7
def magical_fraction : ℚ := 1 / 4
def red_magical_to_blue_magical_ratio : ℚ := 3

theorem fraction_of_red_knights_magical :
  let red_knights := red_fraction * total_knights
  let blue_knights := total_knights - red_knights
  let total_magical := magical_fraction * total_knights
  let red_magical_fraction := 21 / 52
  let blue_magical_fraction := red_magical_fraction / red_magical_to_blue_magical_ratio
  red_knights * red_magical_fraction + blue_knights * blue_magical_fraction = total_magical :=
by
  sorry

end fraction_of_red_knights_magical_l123_123204


namespace sum_digits_n_plus_one_l123_123120

/-- 
Let S(n) be the sum of the digits of a positive integer n.
Given S(n) = 29, prove that the possible values of S(n + 1) are 3, 12, or 30.
-/
theorem sum_digits_n_plus_one (S : ℕ → ℕ) (n : ℕ) (h : S n = 29) :
  S (n + 1) = 3 ∨ S (n + 1) = 12 ∨ S (n + 1) = 30 := 
sorry

end sum_digits_n_plus_one_l123_123120


namespace sum_of_weighted_inequality_l123_123130

theorem sum_of_weighted_inequality (n : ℕ) (a : Fin n → ℝ) 
    (h_sum_zero : (∑ i, a i) = 0)
    (h_sum_abs_one : (∑ i, |a i|) = 1) : 
    |∑ i, (i + 1) * a i| ≤ (n - 1) / 2 := 
begin
  sorry
end

end sum_of_weighted_inequality_l123_123130


namespace volume_of_pyramid_ABCDG_l123_123118

-- Define the cube with edge length 2
def cube_edge_length := 2

-- Define the base area of square ABCD
def base_area_ABCD := cube_edge_length ^ 2

-- Define the height of the pyramid ABCDG
def height_pyramid_ABCDG := cube_edge_length

-- Define the volume formula for a pyramid
def volume_pyramid (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- Proof statement: the volume of pyramid ABCDG is 8 / 3 cubic units
theorem volume_of_pyramid_ABCDG : volume_pyramid base_area_ABCD height_pyramid_ABCDG = 8 / 3 :=
by
  sorry

end volume_of_pyramid_ABCDG_l123_123118


namespace dealer_sold_bmws_l123_123330

theorem dealer_sold_bmws (total_cars mercedes_pct toyota_pct nissan_pct ford_pct : ℝ)
  (h1 : total_cars = 300)
  (h2 : mercedes_pct = 0.1)
  (h3 : toyota_pct = 0.2)
  (h4 : nissan_pct = 0.3)
  (h5 : ford_pct = 0.15) :
  let bmws_sold := total_cars * (1 - (mercedes_pct + toyota_pct + nissan_pct + ford_pct)) in
  bmws_sold = 75 :=
by
  -- Proof goes here
  sorry

end dealer_sold_bmws_l123_123330


namespace length_AM_l123_123499

-- Define the relevant problem parameters and assumptions
variables (A B C M : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
variables (AB AC BC BM MC AM : ℝ)
variables (perimeter_ABC perimeter_ABM : ℝ)

-- Given problem conditions
def is_isosceles (AB AC : ℝ) := AB = AC
def midpoint (BC BM MC : ℝ) := BM = MC
def perimeter_TRIANGLE_ABC (AB AC BM : ℝ) := AB + 2 * BM + AC
def perimeter_TRIANGLE_ABM (AB BM AM : ℝ) := AB + BM + AM

-- Given values in the problem
def values := 
  (perimeter_TRIANGLE_ABC AB AC BM = 64) ∧
  (perimeter_TRIANGLE_ABM AB BM AM = 40)

-- Final assertion to prove
theorem length_AM 
  (h1 : is_isosceles AB AC) 
  (h2 : midpoint BC BM MC) 
  (h3 : perimeter_TRIANGLE_ABC AB AC BM = 64) 
  (h4 : perimeter_TRIANGLE_ABM AB BM AM = 40) : 
  AM = 8 := 
begin 
  sorry 
end

end length_AM_l123_123499


namespace smallest_k_sqrt_diff_l123_123153

-- Define a predicate for the condition that the square root difference is less than 1
def sqrt_diff_less_than_one (a b : ℕ) : Prop :=
  |real.sqrt a - real.sqrt b| < 1

-- Define the main theorem that encapsulates the problem statement
theorem smallest_k_sqrt_diff (cards : Finset ℕ) (h : cards = Finset.range 2016) : 
  ∃ k : ℕ, k = 45 ∧ ∀ s : Finset ℕ, s.card = k →
    ∃ (x y ∈ s), x ≠ y ∧ sqrt_diff_less_than_one x y :=
begin
  sorry
end

end smallest_k_sqrt_diff_l123_123153


namespace quilt_patch_cost_l123_123511

-- Definitions of the conditions
def length : ℕ := 16
def width : ℕ := 20
def patch_area : ℕ := 4
def cost_first_10 : ℕ := 10
def cost_after_10 : ℕ := 5
def num_first_patches : ℕ := 10

-- Define the calculations based on the problem conditions
def quilt_area : ℕ := length * width
def total_patches : ℕ := quilt_area / patch_area
def cost_first : ℕ := num_first_patches * cost_first_10
def remaining_patches : ℕ := total_patches - num_first_patches
def cost_remaining : ℕ := remaining_patches * cost_after_10
def total_cost : ℕ := cost_first + cost_remaining

-- Statement of the proof problem
theorem quilt_patch_cost : total_cost = 450 := by
  -- Placeholder for the proof
  sorry

end quilt_patch_cost_l123_123511


namespace elvie_age_l123_123221

variable (E : ℕ) (A : ℕ)

theorem elvie_age (hA : A = 11) (h : E + A + (E * A) = 131) : E = 10 :=
by
  sorry

end elvie_age_l123_123221


namespace common_difference_is_zero_l123_123543

-- Definitions from the conditions
variables {a b c : ℝ}
variable (h₀ : 0 < a) -- a is positive
variable (h₁ : 0 < b) -- b is positive
variable (h₂ : 0 < c) -- c is positive
variable (h₃ : ¬ (a = b)) -- distinct values, a ≠ b
variable (h₄ : ¬ (b = c)) -- distinct values, b ≠ c
variable (h₅ : ¬ (a = c)) -- distinct values, a ≠ c
variable (h₆ : b = a^3) -- b = a^3
variable (h₇ : c = a^9) -- c = a^9

-- The goal statement: proving the common difference is 0
theorem common_difference_is_zero :
  log b a = log c b ∧ log c b = log a c ↔ log b a = 1 / 3 ∧ log c b = 1 / 3 :=
by
  sorry

end common_difference_is_zero_l123_123543


namespace arithmetic_sequence_sum_l123_123428

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h1 : a 2 + a 4 = 4) (h2 : a 3 + a 5 = 8) : 
  (∑ i in Finset.range 8, a (i + 1)) = 40 :=
sorry

end arithmetic_sequence_sum_l123_123428


namespace harkamal_purchased_mangoes_l123_123447

theorem harkamal_purchased_mangoes :
  ∃ (m : ℕ), m = 9 ∧ (8 * 70 + m * 60 = 1100) :=
by {
  use 9,
  split,
  { refl, },
  { calc
      8 * 70 + 9 * 60
      = 560 + 9 * 60 : by refl
      ... = 560 + 540 : by refl
      ... = 1100 : by refl,
  },
  sorry
}

end harkamal_purchased_mangoes_l123_123447


namespace find_integer_value_l123_123862

def g (x : ℝ) : ℝ := (1 / 5) * x^2 - x - 4

theorem find_integer_value :
  ∃ x : ℤ, g (g (g (x))) = -4 ∧ x = -5 :=
sorry

end find_integer_value_l123_123862


namespace jills_uncles_medicine_last_time_l123_123515

theorem jills_uncles_medicine_last_time :
  let pills := 90
  let third_of_pill_days := 3
  let days_per_full_pill := 9
  let days_per_month := 30
  let total_days := pills * days_per_full_pill
  let total_months := total_days / days_per_month
  total_months = 27 :=
by {
  sorry
}

end jills_uncles_medicine_last_time_l123_123515


namespace minimum_weights_unique_weights_l123_123837

-- Proof problem 1
theorem minimum_weights (n : ℕ) (h : n > 0) : 
  f n = Int.ceil (Real.logb 3 (2 * n + 1)) :=
sorry

-- Proof problem 2
theorem unique_weights (n : ℕ) (m : ℕ) (h : n = (3^m - 1) / 2) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ m → a i = 3^(i - 1)) :=
sorry

end minimum_weights_unique_weights_l123_123837


namespace number_of_elephants_after_three_years_l123_123478

noncomputable def initial_number_of_giraffes := 5
noncomputable def initial_number_of_penguins := 2 * initial_number_of_giraffes
noncomputable def initial_number_of_elephants := 0
noncomputable def initial_number_of_lions := initial_number_of_giraffes + initial_number_of_elephants
noncomputable def initial_number_of_bears := 0

noncomputable def initial_budget := 10000
noncomputable def giraffe_cost := 1000
noncomputable def penguin_cost := 500
noncomputable def elephant_cost := 1200
noncomputable def lion_cost := 1100
noncomputable def bear_cost := 1300

def total_initial_cost := initial_number_of_giraffes * giraffe_cost + initial_number_of_penguins * penguin_cost + initial_number_of_elephants * elephant_cost + initial_number_of_lions * lion_cost + initial_number_of_bears * bear_cost

theorem number_of_elephants_after_three_years : initial_number_of_elephants = 0 := by
  have h_budget : total_initial_cost = initial_budget, by sorry
  have h_growth_effect : initial_number_of_elephants = 0, from sorry
  sorry

end number_of_elephants_after_three_years_l123_123478


namespace angle_triple_supplement_l123_123253

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123253


namespace parity_and_monotonicity_range_of_m_l123_123017

noncomputable def f (a x : ℝ) := (a / (a - 1)) * (2^x - 2^(-x))

theorem parity_and_monotonicity (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (f a (-x) = -f a x) ∧ 
  ((0 < a ∧ a < 1) → ∀ x1 x2, x1 < x2 → f a x1 > f a x2) ∧
  (a > 1 → ∀ x1 x2, x1 < x2 → f a x1 < f a x2) :=
by sorry

theorem range_of_m (a m : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x ∈ Ioo (-1 : ℝ) 1, f a (m - 1) + f a m < 0) :
  ((0 < a ∧ a < 1) → 1 / 2 < m ∧ m < 1) ∧ 
  (a > 1 → 0 < m ∧ m < 1 / 2) :=
by sorry

end parity_and_monotonicity_range_of_m_l123_123017


namespace find_additional_amount_l123_123677

-- Definitions for the given conditions
def expected_earnings_per_share := 0.80
def actual_earnings_per_share := 1.10
def total_dividend_paid := 260
def number_of_shares := 500

-- Derived definitions based on the conditions
def expected_dividend_per_share := (1 / 2) * expected_earnings_per_share
def additional_earnings_per_share := actual_earnings_per_share - expected_earnings_per_share
def total_dividend_per_share := total_dividend_paid / number_of_shares

-- The core condition to prove
def additional_amount_per_share_for_each_additional_0_10_earnings 
  (expected_dividend_per_share : ℝ)
  (additional_earnings_per_share : ℝ)
  (total_dividend_per_share : ℝ) :=
  ∃ (x : ℝ), 
  x = 0.04 ∧ 
  expected_dividend_per_share + (additional_earnings_per_share / 0.10) * x = total_dividend_per_share

theorem find_additional_amount :
  additional_amount_per_share_for_each_additional_0_10_earnings
    expected_dividend_per_share 
    additional_earnings_per_share 
    total_dividend_per_share :=
by
  sorry

end find_additional_amount_l123_123677


namespace total_students_l123_123230

-- Definition of the conditions given in the problem
def num5 : ℕ := 12
def num6 : ℕ := 6 * num5

-- The theorem representing the mathematically equivalent proof problem
theorem total_students : num5 + num6 = 84 :=
by
  sorry

end total_students_l123_123230


namespace probability_no_obtuse_triangle_is_9_over_64_l123_123814

noncomputable def probability_no_obtuse_triangle (A0 A1 A2 A3 : ℝ) : ℝ :=
  -- Define the probabilistic model according to the problem conditions
  -- Assuming A0, A1, A2, and A3 are positions of points on the circle parametrized by angles in radians
  let θ := real.angle.lower_half (A1 - A0) in
  let prob_A2 := (π - θ) / (2 * π) in
  let prob_A3 := (π - θ) / (2 * π) in
  (1 / 2) * (prob_A2 * prob_A3)

theorem probability_no_obtuse_triangle_is_9_over_64 :
  probability_no_obtuse_triangle A0 A1 A2 A3 = 9 / 64 :=
by sorry

end probability_no_obtuse_triangle_is_9_over_64_l123_123814


namespace constant_term_in_expansion_is_neg_220_l123_123976

-- Define expansion and its properties
noncomputable def expansion := (∛x - 1/x)^12

theorem constant_term_in_expansion_is_neg_220 
  (x : ℝ) :
  (sqrt[3] x - 1/x)^12 = expansion → 
  -- Conditions from original problem
  (2^12 = 4096) ↔ 
  -- Conclusion
  (constant_term expansion = -220) :=
by
  sorry

end constant_term_in_expansion_is_neg_220_l123_123976


namespace survey_households_selected_l123_123068

theorem survey_households_selected 
    (total_households : ℕ) 
    (middle_income_families : ℕ) 
    (low_income_families : ℕ) 
    (high_income_selected : ℕ)
    (total_high_income_families : ℕ)
    (total_selected_households : ℕ) 
    (H1 : total_households = 480)
    (H2 : middle_income_families = 200)
    (H3 : low_income_families = 160)
    (H4 : high_income_selected = 6)
    (H5 : total_high_income_families = total_households - (middle_income_families + low_income_families))
    (H6 : total_selected_households * total_high_income_families = high_income_selected * total_households) :
    total_selected_households = 24 :=
by
  -- The actual proof will go here:
  sorry

end survey_households_selected_l123_123068


namespace find_x_pow_4095_minus_recip_l123_123419

variables (x : ℝ) (h : x - 1/x = real.sqrt 2)

theorem find_x_pow_4095_minus_recip (x : ℝ) (h : x - 1/x = real.sqrt 2) : 
  x ^ 4095 - 1 / (x ^ 4095) = 20 * real.sqrt 2 :=
sorry

end find_x_pow_4095_minus_recip_l123_123419


namespace unique_root_a_b_values_l123_123032

theorem unique_root_a_b_values {a b : ℝ} (h1 : ∀ x, x^2 + a * x + b = 0 ↔ x = 1) : a = -2 ∧ b = 1 := by
  sorry

end unique_root_a_b_values_l123_123032


namespace angle_bisector_of_BQC_l123_123536

open EuclideanGeometry

variables {A B C D X P Q : Point}
variables [ConvexQuadrilateral A B C D]
variables [Intersection X (Diagonals AC BD)]
variables [Intersection P (Lines AB CD)]
variables [Intersection Q (Lines PX AD)]

theorem angle_bisector_of_BQC
  (h1 : ∠ ABX = 90°)
  (h2 : ∠ XCD = 90°) :
  IsAngleBisector QP (∠ BQC) := 
sorry

end angle_bisector_of_BQC_l123_123536


namespace find_modulus_of_z_l123_123127

noncomputable def z : ℂ := sorry
noncomputable def w : ℂ := sorry

theorem find_modulus_of_z (hz1 : complex.abs (2 * z - w) = 20) 
                          (hz2 : complex.abs (z + 2 * w) = 10) 
                          (hz3 : complex.abs (z + w) = 5) : 
                          complex.abs z = 0 := 
by
  sorry

end find_modulus_of_z_l123_123127


namespace regular_n_gon_center_inside_circle_l123_123663

-- Define a regular n-gon
structure RegularNGon (n : ℕ) :=
(center : ℝ × ℝ)
(vertices : Fin n → (ℝ × ℝ))

-- Define the condition to be able to roll and reflect the n-gon over any of its sides
def canReflectSymmetrically (n : ℕ) (g : RegularNGon n) : Prop := sorry

-- Definition of a circle with a given center and radius
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

-- Define the problem for determining if reflection can bring the center of n-gon inside any circle
def canCenterBeInsideCircle (n : ℕ) (g : RegularNGon n) (c : Circle) : Prop :=
  ∃ (f : ℝ × ℝ → ℝ × ℝ), -- Some function representing the reflections
    canReflectSymmetrically n g ∧ f g.center = c.center

-- State the main theorem determining for which n-gons the assertion is true
theorem regular_n_gon_center_inside_circle (n : ℕ) 
  (h : n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 6) : 
  ∀ (g : RegularNGon n) (c : Circle), canCenterBeInsideCircle n g c :=
sorry

end regular_n_gon_center_inside_circle_l123_123663


namespace f_values_and_range_l123_123426

variable {f : ℝ → ℝ}
variable {x : ℝ}
variable {y : ℝ}

axiom fx_increasing : ∀ x y, 0 < x → x < y → f(x) < f(y)
axiom f_additive : ∀ x y, 0 < x → 0 < y → f(x * y) = f(x) + f(y)
axiom f_at_2 : f(2) = 1

theorem f_values_and_range :
  f(4) = 2 ∧ f(1 / 2) = -1 ∧ (∀ x, f(2 * x) - f(x - 3) > 2 → 3 < x ∧ x < 6) :=
by
  sorry

end f_values_and_range_l123_123426


namespace square_field_area_l123_123336

-- Define relevant conditions
def walking_speed := 6 * 1000 / 3600  -- walking speed in m/s
def crossing_time := 9  -- time in seconds
def diagonal_length := walking_speed * crossing_time  -- diagonal length in meters

-- The length s of the side of the square satisfies: diagonal_length^2 = 2 * s^2
-- Therefore, s^2 = (diagonal_length^2) / 2

-- Define the theorem to prove the area is 112.5 m^2
theorem square_field_area : 
  (diagonal_length^2 / 2) = 112.5 :=
by
  -- We are only providing the statement, so we use sorry to denote the proof is omitted
  sorry

end square_field_area_l123_123336


namespace problem_sum_150_consecutive_integers_l123_123653

theorem problem_sum_150_consecutive_integers : 
  ∃ k : ℕ, 150 * k + 11325 = 5310375 :=
sorry

end problem_sum_150_consecutive_integers_l123_123653


namespace no_obtuse_triangle_probability_l123_123795

noncomputable def probability_no_obtuse_triangle : ℝ :=
 let θ := 1/2 in
 let prob_A2_A3_given_A0A1 := (3/8) * (3/8) in
 θ * prob_A2_A3_given_A0A1

theorem no_obtuse_triangle_probability :
  probability_no_obtuse_triangle = 9/128 :=
by
  sorry

end no_obtuse_triangle_probability_l123_123795


namespace probability_same_plane_l123_123819

-- Define the number of vertices in a cube
def num_vertices : ℕ := 8

-- Define the number of vertices to be selected
def selection : ℕ := 4

-- Define the total number of ways to select 4 vertices out of 8
def total_ways : ℕ := Nat.choose num_vertices selection

-- Define the number of favorable ways to have 4 vertices lie in the same plane
def favorable_ways : ℕ := 12

-- Define the probability that the 4 selected vertices lie in the same plane
def probability : ℚ := favorable_ways / total_ways

-- The statement we need to prove
theorem probability_same_plane : probability = 6 / 35 := by
  sorry

end probability_same_plane_l123_123819


namespace sum_of_lambdas_l123_123855

-- Define the parameters for the ellipse and points involved in the problem
variables (a b : ℝ)
variable (c : ℝ) -- x-coordinate of focus F
variable (y : ℝ) -- y-coordinate of point P on the y-axis

-- Define conditions and the mathematical proof statement
theorem sum_of_lambdas (xM xN : ℝ) (k : ℝ) (λ1 λ2 : ℝ) :
  (b^2 * xM^2 + a^2 * k^2 * (xM - c)^2 = a^2 * b^2) →
  (b^2 * xN^2 + a^2 * k^2 * (xN - c)^2 = a^2 * b^2) →
  (λ1 = xM / (c - xM)) →
  (λ2 = xN / (c - xN)) →
  λ1 + λ2 = - (2 * a^2 / b^2) :=
sorry

end sum_of_lambdas_l123_123855


namespace gcd_g10_g13_l123_123540

-- Define the polynomial function g
def g (x : ℤ) : ℤ := x^3 - 3 * x^2 + x + 2050

-- State the theorem to prove that gcd(g(10), g(13)) is 1
theorem gcd_g10_g13 : Int.gcd (g 10) (g 13) = 1 := by
  sorry

end gcd_g10_g13_l123_123540


namespace polar_to_cartesian_l123_123029

theorem polar_to_cartesian (θ : ℝ) (ρ : ℝ) (x y : ℝ) :
  (ρ = 2 * Real.sin θ + 4 * Real.cos θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x - 8)^2 + (y - 2)^2 = 68 :=
by
  intros hρ hx hy
  -- Proof steps would go here
  sorry

end polar_to_cartesian_l123_123029


namespace slope_of_tangent_line_at_half_l123_123830

noncomputable def f (x : ℝ) : ℝ := x^3 - 2
def x : ℝ := 1 / 2

theorem slope_of_tangent_line_at_half :
  (deriv f x) = 3 / 4 :=
by
  sorry

end slope_of_tangent_line_at_half_l123_123830


namespace product_gcd_lcm_4000_l123_123710

-- Definitions of gcd and lcm for the given numbers
def gcd_40_100 := Nat.gcd 40 100
def lcm_40_100 := Nat.lcm 40 100

-- Problem: Prove that the product of the gcd and lcm of 40 and 100 equals 4000
theorem product_gcd_lcm_4000 : gcd_40_100 * lcm_40_100 = 4000 := by
  sorry

end product_gcd_lcm_4000_l123_123710


namespace inverse_proportion_passes_first_and_third_quadrants_l123_123028

theorem inverse_proportion_passes_first_and_third_quadrants (m : ℝ) :
  ((∀ x : ℝ, x ≠ 0 → (x > 0 → (m - 3) / x > 0) ∧ (x < 0 → (m - 3) / x < 0)) → m = 5) := 
by 
  sorry

end inverse_proportion_passes_first_and_third_quadrants_l123_123028


namespace union_complement_A_B_l123_123033

noncomputable def A : set ℝ := {x | x^2 - 2*x - 3 > 0}
noncomputable def B : set ℝ := {x | (real.log (x - 2)) ≤ 0}
def complement_A : set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem union_complement_A_B :
  (complement_A ∪ B) = {x | -1 ≤ x ∧ x ≤ 3} :=
sorry

end union_complement_A_B_l123_123033


namespace piggy_bank_donation_l123_123359

theorem piggy_bank_donation (total_earnings : ℕ) (cost_of_ingredients : ℕ) 
  (total_donation_homeless_shelter : ℕ) : 
  (total_earnings = 400) → (cost_of_ingredients = 100) → (total_donation_homeless_shelter = 160) → 
  (total_donation_homeless_shelter - (total_earnings - cost_of_ingredients) / 2 = 10) :=
by
  intros h1 h2 h3
  sorry

end piggy_bank_donation_l123_123359


namespace remainder_of_series_div_9_l123_123287

def sum (n : Nat) : Nat := n * (n + 1) / 2

theorem remainder_of_series_div_9 : (sum 20) % 9 = 3 :=
by
  -- The proof will go here
  sorry

end remainder_of_series_div_9_l123_123287


namespace finish_work_in_20_days_l123_123574

-- Define the individual work rates of Ravi, Prakash, and Seema.
def Ravi_work_rate : ℚ := 1 / 50
def Prakash_work_rate : ℚ := 1 / 75
def Seema_work_rate : ℚ := 1 / 60

-- Define the combined work rate.
def combined_work_rate : ℚ := Ravi_work_rate + Prakash_work_rate + Seema_work_rate

-- Conclusion: They can finish the work together in 20 days.
theorem finish_work_in_20_days : (1 / combined_work_rate) = 20 :=
by
  have h1 : combined_work_rate = 1 / 20 := by sorry
  rw [←h1, one_div_one_div]
  rfl

end finish_work_in_20_days_l123_123574


namespace quadratic_transformation_l123_123596

theorem quadratic_transformation (a b c : ℝ) (h : a * x^2 + b * x + c = 5 * (x + 2)^2 - 7) :
  ∃ (n m g : ℝ), 2 * a * x^2 + 2 * b * x + 2 * c = n * (x - g)^2 + m ∧ g = -2 :=
by
  sorry

end quadratic_transformation_l123_123596


namespace triangle_side_length_l123_123350

noncomputable def sin_deg (θ : ℝ) : ℝ := Real.sin (θ * Real.pi / 180)

theorem triangle_side_length (a b c : ℝ) (angleA angleB angleC : ℝ)
    (hA : angleA = 53) (hB : angleB = 93) (hC : angleC = 34)
    (h1 : sin_deg angleC = 0.5592) (h2 : sin_deg angleB = 0.9986) (h3 : c = 7) :
    a = 3.92 :=
by
  have h_sinA := sin_deg angleA
  have h_sinC := sin_deg angleC
  have h_law_of_sines := (7 * h_sinC) / h2
  sorry

end triangle_side_length_l123_123350


namespace angle_triple_supplement_l123_123255

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123255


namespace angle_triple_supplement_l123_123261

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123261


namespace dice_probability_l123_123816

open_locale big_operators

noncomputable def probability_of_no_one : ℚ :=
let individual_probability := (5 : ℚ) / 6 in
individual_probability ^ 4

theorem dice_probability (a b c d : ℕ) (h_a : 1 ≤ a ∧ a ≤ 6) (h_b : 1 ≤ b ∧ b ≤ 6) (h_c : 1 ≤ c ∧ c ≤ 6) (h_d : 1 ≤ d ∧ d ≤ 6) :
  (∏ x in {a, b, c, d}, if x = 1 then 0 else 1) = 1 ↔ probability_of_no_one = 625 / 1296 :=
by sorry

end dice_probability_l123_123816


namespace minimum_a_value_for_no_solution_l123_123906

theorem minimum_a_value_for_no_solution : 
  ∀ (a : ℝ), 
  (¬ ∃ x ∈ set.Ioo 0 (1 / 2), (2 - a) * (x - 1) - 2 * real.log x = 0) ↔ a ≥ 2 - 4 * real.log 2 :=
by sorry

end minimum_a_value_for_no_solution_l123_123906


namespace problem1_problem2_l123_123841

-- Define the eccentricity and the ellipse passing through point M based on the conditions
def eccentricity := 1 / 2
def M := (1 : ℝ, 3 / 2 : ℝ)
def C_eq (x y : ℝ) := x^2 / 4 + y^2 / 3 = 1

-- Define the point where the line passes
def P := (2 : ℝ, 1 : ℝ)

-- Define the correct answer from the solution
def ellipse_equation := ∀ (x y : ℝ), C_eq x y = (x^2 / 4 + y^2 / 3 = 1)
def line_equation := ∃ k : ℝ, k = 1 / 2 ∧ ∀ (x : ℝ), x = 2 → k * (x - 2) + 1 = (1 / 2) * x

-- Problem 1: Prove the equation of the ellipse
theorem problem1 : ellipse_equation := 
by sorry

-- Problem 2: Prove the existence and equation of the line
theorem problem2 : 
(∃ (k : ℝ), k = 1 / 2 ∧ ∀ (x : ℝ), (k * (x - 2) + 1) * x = (1 / 2) * x) :=
by sorry

end problem1_problem2_l123_123841


namespace range_of_even_function_l123_123721

-- Definitions and conditions
variable (a : ℝ) (f : ℝ → ℝ)
#check Real
#check log
#check even

-- f is an even function defined on the interval [-2a + 3, a]
noncomputable def is_even_function (f : ℝ → ℝ) (a : ℝ) : Prop :=
    ∀ x ∈ [-2a + 3, a], f x = f (-x)

-- For 0 ≤ x ≤ a, f(x) = log_a(2x + 3)
noncomputable def f_def (a : ℝ) (x : ℝ) : Prop :=
    0 ≤ x ∧ x ≤ a → f x = Real.log a (2 * x + 3)

-- The main theorem statement
theorem range_of_even_function :
  ((-2 * a + 3) + a = 0) →
  (is_even_function f a) →
  (∀ x, f_def a x) →
  (∃ y, y = f x ∧ y ∈ [1, 2]) :=
by
    sorry

end range_of_even_function_l123_123721


namespace total_money_john_makes_l123_123996

-- Define the hourly rate for the first 12 hours
def hourly_rate_first_period := 5000

-- Define the total hours for the first period
def hours_first_period := 12

-- Define the percentage increase for the second period
def percentage_increase := 0.20

-- Define the additional money generated per hour for the second period
def extra_hourly_rate := hourly_rate_first_period * percentage_increase

-- Define the hourly rate for the second period
def hourly_rate_second_period := hourly_rate_first_period + extra_hourly_rate

-- Define the total hours for the second period
def hours_second_period := 14

-- Define the total money made by John
def total_money_made := 
  (hours_first_period * hourly_rate_first_period) + 
  (hours_second_period * hourly_rate_second_period)

-- The theorem to prove
theorem total_money_john_makes : total_money_made = 144000 := by
  sorry

end total_money_john_makes_l123_123996


namespace sum_primes_between_20_and_40_l123_123647

open Nat

def primesBetween20And40 : List Nat := [23, 29, 31, 37]

theorem sum_primes_between_20_and_40 :
  (primesBetween20And40.sum = 120) :=
by
  sorry

end sum_primes_between_20_and_40_l123_123647


namespace range_of_a_l123_123421

variable {x a : ℝ}

def condition_p : Prop := 0 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 1

def condition_q : Prop := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a (h : condition_p → condition_q) : 0 ≤ a ∧ a ≤ 1/2 := sorry

end range_of_a_l123_123421


namespace find_percentage_l123_123678

theorem find_percentage (P : ℕ) (h: (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 18) : P = 15 :=
sorry

end find_percentage_l123_123678


namespace eval_expression_at_values_l123_123381

theorem eval_expression_at_values : 
  ∀ x y : ℕ, x = 3 ∧ y = 4 → 
  5 * (x^(y+1)) + 6 * (y^(x+1)) + 2 * x * y = 2775 :=
by
  intros x y hxy
  cases hxy
  sorry

end eval_expression_at_values_l123_123381


namespace factorable_polynomials_l123_123700

def polynomial_1 (x y : ℝ) : ℝ := -x^2 - y^2
def polynomial_2 (a : ℝ) : ℝ := 1 - (a - 1)^2
def polynomial_3 (x y : ℝ) : ℝ := 2x^2y^3 - 8xy^2
def polynomial_4 (m n : ℝ) : ℝ := m^2 - 6 * m * n + 9 * n^2

theorem factorable_polynomials (x y a m n : ℝ) :
  (∃ b c : ℝ, polynomial_2 a = b^2 - c^2) ∧
  (∃ p q : ℝ, polynomial_4 m n = (p^2 + q)^2) :=
sorry

end factorable_polynomials_l123_123700


namespace archers_in_golden_armor_l123_123955

theorem archers_in_golden_armor (total_soldiers archers_swordsmen total_affirmations armor_affirmations archer_affirmations monday_affirmations : ℕ)
  (h1: total_soldiers = 55)
  (h2: armor_affirmations = 44)
  (h3: archer_affirmations = 33)
  (h4: monday_affirmations = 22)
  (h5: ∑ x in ({armor_affirmations, archer_affirmations, monday_affirmations} : finset ℕ), x = 99) 
  : ∃ (archers_in_golden : ℕ), archers_in_golden = 22 := by
  -- Definitions and theorems will go here
  sorry

end archers_in_golden_armor_l123_123955


namespace period_and_extreme_values_axis_of_symmetry_center_of_symmetry_l123_123858

section
variables (x : ℝ) (k : ℤ)

def f (x : ℝ) : ℝ := (sin x + cos x) ^ 2 + cos (2 * x)

theorem period_and_extreme_values :
  (∀ x, f (x + π) = f x) ∧
  (∀ x, x ∈ set.range f → x ≤ sqrt 2 + 1) ∧
  (forall x, x ∈ set.range f → x ≥ 1 - sqrt 2) := by sorry

theorem axis_of_symmetry :
  ∀ k : ℤ, ∀ x : ℝ, (2 * x + π / 4 = k * π + π / 2) → x = (k * π / 2) + (π / 8) := by sorry

theorem center_of_symmetry :
  ∀ k : ℤ, ∀ x : ℝ, (2 * x + π / 4 = k * π) → x = (k * π / 2) - (π / 8) ∧ f x = 1 := by sorry

end

end period_and_extreme_values_axis_of_symmetry_center_of_symmetry_l123_123858


namespace system_soln_l123_123445

theorem system_soln (a1 b1 a2 b2 : ℚ)
  (h1 : a1 * 3 + b1 * 6 = 21)
  (h2 : a2 * 3 + b2 * 6 = 12) :
  (3 = 3 ∧ -3 = -3) ∧ (a1 * (2 * 3 + -3) + b1 * (3 - -3) = 21) ∧ (a2 * (2 * 3 + -3) + b2 * (3 - -3) = 12) :=
by
  sorry

end system_soln_l123_123445


namespace sample_transformation_sd_l123_123008

-- Definitions for the conditions
variables {x : Type*} [has_pow ℝ ℝ] [metric_space ℝ]

def variance (s : ℝ) : ℝ := s^2
def standard_deviation (v : ℝ) : ℝ := real.sqrt v

-- Given condition
def sample_variance : ℝ := 3

-- The target standard deviation we want to prove
def expected_standard_deviation : ℝ := 2 * real.sqrt 3

-- The theorem to prove
theorem sample_transformation_sd :
  standard_deviation (4 * sample_variance) = expected_standard_deviation :=
by sorry

end sample_transformation_sd_l123_123008


namespace f_sum_to_zero_l123_123420

-- Definitions for the given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

variable (f : ℝ → ℝ)

axiom odd_f : is_odd_function f
axiom even_f_shift : is_even_function (λ x, f (x + 1))
axiom f_in_interval : ∀ x, 2 < x ∧ x < 4 → f x = |x - 3|

-- Statement of the proof problem
theorem f_sum_to_zero : f 1 + f 2 + f 3 + f 4 = 0 :=
sorry

end f_sum_to_zero_l123_123420


namespace determinant_is_zero_l123_123129

-- Define polynomials over the complex numbers (since roots of polynomials over reals can be non-real)
noncomputable theory
open Complex

variables {p q r : ℂ}
variables {a b c d : ℂ}

-- Conditions: a, b, c, and d are roots of the polynomial x^4 + px^2 + qx + r = 0
def poly (x : ℂ) : ℂ := x^4 + p * x^2 + q * x + r

-- Define the condition that a, b, c, d are roots of the polynomial
def conditions := poly a = 0 ∧ poly b = 0 ∧ poly c = 0 ∧ poly d = 0

-- Define the matrix whose determinant we need to compute
def defined_matrix : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![a, b, c, d],
    ![b, c, d, a],
    ![c, d, a, b],
    ![d, a, b, c]]

-- State the main theorem
theorem determinant_is_zero (h : conditions) : defined_matrix.det = 0 :=
sorry

end determinant_is_zero_l123_123129


namespace student_scores_four_marks_per_correct_answer_l123_123073

theorem student_scores_four_marks_per_correct_answer:
  (∃ x : ℤ, 
    let total_questions := 80 in
    let correct_answers := 40 in
    let wrong_answers := total_questions - correct_answers in
    let marks_lost_per_wrong_answer := 1 in
    let total_marks := 120 in
    let total_marks_secured := correct_answers * x - wrong_answers * marks_lost_per_wrong_answer in
    total_marks_secured = total_marks ∧ x = 4) := 
by
  sorry

end student_scores_four_marks_per_correct_answer_l123_123073


namespace Linda_original_savings_l123_123550

variable (TV_cost : ℝ := 200) -- TV cost
variable (savings : ℝ) -- Linda's original savings

-- Prices, Discounts, Taxes
variable (sofa_price : ℝ := 600)
variable (sofa_discount : ℝ := 0.20)
variable (sofa_tax : ℝ := 0.05)

variable (dining_table_price : ℝ := 400)
variable (dining_table_discount : ℝ := 0.15)
variable (dining_table_tax : ℝ := 0.06)

variable (chair_set_price : ℝ := 300)
variable (chair_set_discount : ℝ := 0.25)
variable (chair_set_tax : ℝ := 0.04)

variable (coffee_table_price : ℝ := 100)
variable (coffee_table_discount : ℝ := 0.10)
variable (coffee_table_tax : ℝ := 0.03)

variable (service_charge_rate : ℝ := 0.02) -- Service charge rate

noncomputable def discounted_price_with_tax (price discount tax : ℝ) : ℝ :=
  let discounted_price := price * (1 - discount)
  let taxed_price := discounted_price * (1 + tax)
  taxed_price

noncomputable def total_furniture_cost : ℝ :=
  let sofa_cost := discounted_price_with_tax sofa_price sofa_discount sofa_tax
  let dining_table_cost := discounted_price_with_tax dining_table_price dining_table_discount dining_table_tax
  let chair_set_cost := discounted_price_with_tax chair_set_price chair_set_discount chair_set_tax
  let coffee_table_cost := discounted_price_with_tax coffee_table_price coffee_table_discount coffee_table_tax
  let combined_cost := sofa_cost + dining_table_cost + chair_set_cost + coffee_table_cost
  combined_cost * (1 + service_charge_rate)

theorem Linda_original_savings : savings = 4 * TV_cost ∧ savings / 4 * 3 = total_furniture_cost :=
by
  sorry -- Proof skipped

end Linda_original_savings_l123_123550


namespace solve_question_l123_123533

noncomputable def question_statement : Prop :=
  ∀ z : ℂ, (z^2 + complex.abs z ^ 2 = 5 - 7 * complex.I) → (complex.abs z ^ 2 = 7.4)

theorem solve_question : question_statement :=
by
  sorry

end solve_question_l123_123533


namespace sum_of_primes_between_20_and_40_l123_123643

theorem sum_of_primes_between_20_and_40 : 
  (23 + 29 + 31 + 37) = 120 := 
by
  -- Proof goes here
sorry

end sum_of_primes_between_20_and_40_l123_123643


namespace line_equation_passing_through_point_with_equal_intercepts_l123_123333

theorem line_equation_passing_through_point_with_equal_intercepts :
  ∃ a, a ≠ 0 ∧ (∀ x y, (x + y = a) → (x = -3 ∧ y = -2 → a = -5)) ∧ 
       (∀ x y, x + y + 5 = 0):
  sorry

end line_equation_passing_through_point_with_equal_intercepts_l123_123333


namespace average_apples_per_guest_l123_123103

def servings_per_pie : ℕ := 8
def pies : ℕ := 3
def servings := pies * servings_per_pie
def apples_per_serving : ℝ := 1.5
def total_apples := servings * apples_per_serving
def guests : ℕ := 12
def apples_per_guest := total_apples / guests

theorem average_apples_per_guest : apples_per_guest = 3 := by
  sorry

end average_apples_per_guest_l123_123103


namespace general_term_sum_of_terms_l123_123409

variable {a_n : ℕ → ℤ} -- assuming the general term can be expressed as an integer-valued function over natural numbers

-- Variables representing conditions
variable (a1 a2 a3 a4 : ℤ)
variable (d : ℤ)
variable (n : ℕ)

-- Condition 1: Arithmetic sequence with common difference 2
axiom common_difference (n : ℕ) : a_n (n + 1) = a_n n + 2

-- Condition 2: All terms positive
axiom all_terms_positive (n : ℕ) : a_n n > 0

-- Condition 3: Known equation a_2 * a_4 = 4 * a_3 + 1
axiom known_relation : a_n 2 * a_n 4 = 4 * a_n 3 + 1

-- Question 1: Find the general term a_n = 2n - 1
theorem general_term (n : ℕ) : a_n n = 2 * n - 1 := by
    sorry

-- Question 2: Sum of terms a_1 + a_3 + a_9 + ... + a_{3^n}
theorem sum_of_terms (n : ℕ) : ∑ k in (Finset.range (n + 1)), a_n (3 ^ k) = 3 ^ (n + 1) - n - 2 := by
    sorry

end general_term_sum_of_terms_l123_123409


namespace g_is_increasing_on_0_pi_div_2_l123_123430

def f (x : ℝ) : ℝ :=
  if x < 0 then
    π * Real.cos x
  else
    f (x - π)

def g (x : ℝ) : ℝ :=
  Real.sin (2 * x - f (2 * π / 3))

-- Now we state that [0, π / 2] is an interval where g is increasing.
theorem g_is_increasing_on_0_pi_div_2 :
  ∀ x1 x2, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 2 → g x1 ≤ g x2 :=
sorry

end g_is_increasing_on_0_pi_div_2_l123_123430


namespace proof_problem_l123_123838

/-- Given conditions for the sequences and sets --/
noncomputable def S (n : ℕ) : ℤ := n^2 + 2*n
noncomputable def a (n : ℕ) : ℤ := 2 * n + 1 
noncomputable def k (n : ℕ) : ℤ := 2 * n + 2
noncomputable def b (n : ℕ) : ℤ := 4 * (2 * n + 1) * 4^n
noncomputable def T (n : ℕ) : ℤ := (6 * n + 1) * 4^(n + 2) / 9 - 16 / 9
def A : set ℤ := {x | ∃ (n : ℕ), x = k n}
def B : set ℤ := {x | ∃ (n : ℕ), x = 2 * a n}
def c (n : ℕ) : ℤ := 12 * n - 6

/-- Proving that given the conditions, these sequences and sets satisfy the following properties --/
theorem proof_problem :
  (∀ n, S n = n^2 + 2*n) ∧
  (∀ n, a n = 2*n + 1) ∧
  (∀ n, b n = 4 * (2*n + 1) * 4^n) ∧
  (∀ n, T n = (6*n + 1) * 4^(n+2) / 9 - 16 / 9) ∧
  (A = B) ∧
  (∀ n, c n = 12 * n - 6) :=
sorry

end proof_problem_l123_123838


namespace johns_boxes_l123_123995

def percentage_fewer (x y : ℕ) (p : ℝ) : ℕ :=
  x - (p * x).toNat

theorem johns_boxes (John Jules Joseph Stan : ℕ) (h1 : Stan = 100)
  (h2 : Joseph = percentage_fewer Stan 0.8)
  (h3 : Jules = Joseph + 5)
  (h4 : John = 30) : John = 30 :=
by
  rw [h4]
  trivial

end johns_boxes_l123_123995


namespace rational_numbers_equal_l123_123606

theorem rational_numbers_equal {a : ℕ → ℚ} (n : ℕ) (h : ∀ S : finset (fin (2 * n + 1)), S.card = 2 * n → ∃ S1 S2 : finset (fin (2 * n + 1)), S1.card = n ∧ S2.card = n ∧ (S1 ∪ S2 = S) ∧ (S1 ≠ S2) ∧ (S1.sum a = S2.sum a)) :
  ∀ i j : fin (2 * n + 1), a i = a j :=
sorry

end rational_numbers_equal_l123_123606


namespace sum_primes_between_20_and_40_l123_123638

theorem sum_primes_between_20_and_40 :
  23 + 29 + 31 + 37 = 120 :=
by
  sorry

end sum_primes_between_20_and_40_l123_123638


namespace three_digit_integers_product_36_l123_123884

theorem three_digit_integers_product_36 : 
  ∃ (num_digits : ℕ), num_digits = 21 ∧ 
    ∀ (a b c : ℕ), 
      (1 ≤ a ∧ a ≤ 9) ∧ 
      (1 ≤ b ∧ b ≤ 9) ∧ 
      (1 ≤ c ∧ c ≤ 9) ∧ 
      (a * b * c = 36) → 
      num_digits = 21 :=
sorry

end three_digit_integers_product_36_l123_123884


namespace probability_diff_colors_l123_123479

theorem probability_diff_colors (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) (total_outcomes : ℕ) (favorable_outcomes : ℕ) :
  total_balls = 4 ∧ white_balls = 3 ∧ black_balls = 1 ∧ drawn_balls = 2 ∧ 
  total_outcomes = Nat.choose 4 2 ∧ favorable_outcomes = Nat.choose 3 1 * Nat.choose 1 1
  → favorable_outcomes / total_outcomes = 1 / 2 :=
by
  sorry

end probability_diff_colors_l123_123479


namespace inequality_of_variables_l123_123459

variable (a b c : ℝ)

theorem inequality_of_variables (h1 : sqrt (1 + 2 * a) = 1.01) 
                                (h2 : exp b = 1.01) 
                                (h3 : 1 / (1 - c) = 1.01) : 
  a > b ∧ b > c := 
by
  let a_value : ℝ := ((1.01 : ℝ) ^ 2 - 1)/2
  let b_value : ℝ := real.log 1.01
  let c_value : ℝ := 1 - 1 / (1.01 : ℝ)
  
  -- Placeholder for steps to compare values
  sorry

end inequality_of_variables_l123_123459


namespace minimal_positive_period_f_max_value_g_on_interval_l123_123021

noncomputable def f (x: ℝ) : ℝ :=
  sin ((π * x / 4) - (π / 6)) - 2 * cos (π * x / 8)^2 + 1

noncomputable def g (x: ℝ) : ℝ :=
  f (2 - x)

theorem minimal_positive_period_f : minimal_period f = 8 := 
  sorry

theorem max_value_g_on_interval : ∃ x ∈ set.Icc (0 : ℝ) (4 / 3), g x = sqrt 3 / 2 := 
  sorry

end minimal_positive_period_f_max_value_g_on_interval_l123_123021


namespace archers_in_golden_l123_123936

variables (soldiers archers swordsmen wearing_golden wearing_black : ℕ)
variables (truth_swordsmen_black lie_swordsmen_golden lie_archers_black truth_archers_golden : ℕ)

-- Given conditions
variables (cond1 : archers + swordsmen = 55)
variables (cond2 : wearing_golden + wearing_black = 55)
variables (cond3 : truth_swordsmen_black + lie_swordsmen_golden + lie_archers_black + truth_archers_golden = 55)
variables (cond4 : wearing_golden = 44)
variables (cond5 : archers = 33)
variables (cond6 : truth_swordsmen_black + lie_archers_black = 22)

-- Define the mathematic equivalent proof problem
theorem archers_in_golden : archers = 22 :=
by
  sorry

end archers_in_golden_l123_123936


namespace two_distinct_zeros_l123_123057

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 3 * abs (Real.cos x) - Real.cos x + m

theorem two_distinct_zeros (m : ℝ) : 
  (∃ x₁ x₂ ∈ (0 : ℝ) < 2 * Real.pi, x₁ ≠ x₂ ∧ f x₁ m = 0 ∧ f x₂ m = 0) ↔ 
  (-4 < m ∧ m ≤ -2) ∨ (m = 0) := 
sorry

end two_distinct_zeros_l123_123057


namespace trigonometric_inequality_l123_123023

-- Let \( f(x) \) be defined as \( cos \, x \)
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- Given a, b, c are the sides of triangle ∆ABC opposite to angles A, B, C respectively
variables {a b c A B C : ℝ}

-- Condition: \( 3a^2 + 3b^2 - c^2 = 4ab \)
variable (h : 3 * a^2 + 3 * b^2 - c^2 = 4 * a * b)

-- Goal: Prove that \( f(\cos A) \leq f(\sin B) \)
theorem trigonometric_inequality (h1 : A + B + C = π) (h2 : a^2 + b^2 - 2 * a * b * Real.cos C = c^2) : 
  f (Real.cos A) ≤ f (Real.sin B) :=
by
  sorry

end trigonometric_inequality_l123_123023


namespace trigonometric_expression_value_l123_123398

theorem trigonometric_expression_value 
  (h1 : cos (π / 2 + x) = 4 / 5) 
  (h2 : x ∈ Ioo (-π / 2) 0) :
  (sin (2 * x) - 2 * (sin x)^2) / (1 + tan x) = -168 / 25 := 
by
  -- Proof omitted; corresponds to the steps mentioned above
  sorry

end trigonometric_expression_value_l123_123398


namespace maximum_value_l123_123123

noncomputable def max_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  a ^ 2 + 3 * b ^ 2

theorem maximum_value
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b) :
  ∃ x : ℝ, 2 * (a - x) * (x + real.sqrt (x ^ 2 + 3 * b ^ 2)) = a ^ 2 + 3 * b ^ 2 :=
sorry

end maximum_value_l123_123123


namespace sum_not_divisible_by_5_l123_123568

theorem sum_not_divisible_by_5 (n : ℕ) (hn : n > 0) : ¬ 5 ∣ (∑ k in Finset.range (n + 1), 2 ^ (3 * k) * Nat.choose (2 * n + 1) (2 * k + 1)) :=
by
  sorry

end sum_not_divisible_by_5_l123_123568


namespace sum_first_100_terms_l123_123414

noncomputable def seq : ℕ → ℕ
| 0       := 1
| 1       := 2
| (n + 2) := seq n + 1 + (-1)^n

def partial_sum (n : ℕ) : ℕ :=
  (List.range n).map seq |> List.sum

theorem sum_first_100_terms :
  partial_sum 100 = 2600 :=
by
  sorry

end sum_first_100_terms_l123_123414


namespace find_a_values_l123_123865

-- Define the set A and the conditions
def A (a : ℝ) : Set ℝ := { x | a * x^2 + 2 * x + a = 0 }

-- Define the main theorem statement
theorem find_a_values (a : ℝ) :
  (A a).card = 1 → a ∈ {0, 1, -1} :=
sorry

end find_a_values_l123_123865


namespace find_m_plus_b_l123_123589

-- Define the reflection property based on given conditions
def reflected_about_line (p q : Point) (m b : ℝ) : Prop :=
  let mid := ((p.1 + q.1) / 2, (p.2 + q.2) / 2) in
  p.2 = m * p.1 + b ∧ q.2 = m * q.1 + b ∧ (mid.2 = m * mid.1 + b)

-- Define the main theorem
theorem find_m_plus_b (m b : ℝ) :
  reflected_about_line (2, 3) (6, 7) m b →
  m + b = 8 :=
by
  sorry

end find_m_plus_b_l123_123589


namespace number_of_pupils_wrong_entry_l123_123340

theorem number_of_pupils_wrong_entry 
  (n : ℕ) (A : ℝ) 
  (h_wrong_entry : ∀ m, (m = 85 → n * (A + 1 / 2) = n * A + 52))
  (h_increase : ∀ m, (m = 33 → n * (A + 1 / 2) = n * A + 52)) 
  : n = 104 := 
sorry

end number_of_pupils_wrong_entry_l123_123340


namespace cost_of_drapes_l123_123112

theorem cost_of_drapes (D: ℝ) (h1 : 3 * 40 = 120) (h2 : D * 3 + 120 = 300) : D = 60 :=
  sorry

end cost_of_drapes_l123_123112


namespace pascal_triangle_row_sum_ratio_l123_123452

theorem pascal_triangle_row_sum_ratio : 
  let S₁ := ∑ k in Finset.range 101, Nat.choose 100 k,
      S₂ := ∑ k in Finset.range 102, Nat.choose 101 k
  in S₂ / S₁ = 2 := 
by
  let S₁ := ∑ k in Finset.range 101, Nat.choose 100 k
  let S₂ := ∑ k in Finset.range 102, Nat.choose 101 k
  have hS₁ : S₁ = 2^100 := Nat.sum_range_choose (by norm_num)
  have hS₂ : S₂ = 2^101 := Nat.sum_range_choose (by norm_num)
  calc
    S₂ / S₁ = 2^101 / 2^100 : by rw [hS₁, hS₂]
          ... = 2 : by norm_num

end pascal_triangle_row_sum_ratio_l123_123452


namespace number_of_archers_in_golden_armor_l123_123966

-- Define the problem context
structure Soldier where
  is_archer : Bool
  wears_golden_armor : Bool

def truth_teller (s : Soldier) (is_black_armor : Bool) : Bool :=
  if s.is_archer then s.wears_golden_armor = is_black_armor
  else s.wears_golden_armor ≠ is_black_armor

def response (s : Soldier) (question : String) (is_black_armor : Bool) : Bool :=
  match question with
  | "Are you wearing golden armor?" => if truth_teller s is_black_armor then s.wears_golden_armor else ¬s.wears_golden_armor
  | "Are you an archer?" => if truth_teller s is_black_armor then s.is_archer else ¬s.is_archer
  | "Is today Monday?" => if truth_teller s is_black_armor then True else False -- An assumption that today not being Monday means False
  | _ => False

-- Problem condition setup
def total_soldiers : Nat := 55
def golden_armor_yes : Nat := 44
def archer_yes : Nat := 33
def monday_yes : Nat := 22

-- Define the main theorem
theorem number_of_archers_in_golden_armor :
  ∃ l : List Soldier,
    l.length = total_soldiers ∧
    l.countp (λ s => response s "Are you wearing golden armor?" True) = golden_armor_yes ∧
    l.countp (λ s => response s "Are you an archer?" True) = archer_yes ∧
    l.countp (λ s => response s "Is today Monday?" True) = monday_yes ∧
    l.countp (λ s => s.is_archer ∧ s.wears_golden_armor) = 22 :=
sorry

end number_of_archers_in_golden_armor_l123_123966


namespace maximize_daily_profit_l123_123331

noncomputable def daily_profit : ℝ → ℝ → ℝ
| x, c => if h : 0 < x ∧ x ≤ c then (3 * (9 * x - 2 * x^2)) / (2 * (6 - x)) else 0

theorem maximize_daily_profit (c : ℝ) (x : ℝ) (h1 : 0 < c) (h2 : c < 6) :
  (y = daily_profit x c) ∧
  (if 0 < c ∧ c < 3 then x = c else if 3 ≤ c ∧ c < 6 then x = 3 else False) :=
by
  sorry

end maximize_daily_profit_l123_123331


namespace sum_binomial_2k_eq_2_2n_l123_123662

open scoped BigOperators

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sum_binomial_2k_eq_2_2n (n : ℕ) :
  ∑ k in Finset.range (n + 1), 2^k * binomial_coeff (2*n - k) n = 2^(2*n) := 
by
  sorry

end sum_binomial_2k_eq_2_2n_l123_123662


namespace minimum_pumps_required_l123_123315

theorem minimum_pumps_required
  (V_tank : ℝ)
  (percent_fill : ℝ)
  (pump_capacity : ℝ)
  (efficiency : ℝ) :
  V_tank = 1000 →
  percent_fill = 0.85 →
  pump_capacity = 150 →
  efficiency = 0.75 →
  let V_fill := percent_fill * V_tank in
  let effective_capacity := efficiency * pump_capacity in
  let num_pumps := V_fill / effective_capacity in
  ∃ (n : ℕ), n = ⌈num_pumps⌉ ∧ n = 8 :=
by
  intros h_V_tank h_percent_fill h_pump_capacity h_efficiency
  have V_fill_def : V_fill = 0.85 * 1000 := by rw [h_V_tank, h_percent_fill]
  have effective_capacity_def : effective_capacity = 0.75 * 150 := by rw [h_pump_capacity, h_efficiency]
  have num_pumps_def : num_pumps = (0.85 * 1000) / (0.75 * 150) :=
    by rw [V_fill_def, effective_capacity_def]
  have num_pumps_val : num_pumps = 850 / 112.5 :=
    by norm_num at num_pumps_def
  have rounded_val : ⌈num_pumps⌉ = 8 :=
    by norm_num [num_pumps_val]
  use 8
  split
  · exact rounded_val
  · exact rounded_val

end minimum_pumps_required_l123_123315


namespace ArcherInGoldenArmorProof_l123_123931

-- Definitions of the problem
variables (soldiers : Nat) (archers soldiersInGolden : Nat) 
variables (soldiersInBlack archersInGolden archersInBlack swordsmenInGolden swordsmenInBlack : Nat)
variables (truthfulSwordsmenInBlack lyingArchersInBlack lyingSwordsmenInGold truthfulArchersInGold : Nat)
variables (yesToGold yesToArcher yesToMonday : Nat)

-- Given conditions
def ProblemStatement : Prop :=
  soldiers = 55 ∧
  yesToGold = 44 ∧
  yesToArcher = 33 ∧
  yesToMonday = 22 ∧
  soldiers = archers + (soldiers - archers) ∧
  soldiers = soldiersInGolden + soldiersInBlack ∧
  archers = archersInGolden + archersInBlack ∧
  soldiersInGolden = archersInGolden + swordsmenInGolden ∧
  soldiersInBlack = archersInBlack + swordsmenInBlack ∧
  truthfulSwordsmenInBlack = swordsmenInBlack ∧
  lyingArchersInBlack = archersInBlack ∧
  lyingSwordsmenInGold = swordsmenInGolden ∧
  truthfulArchersInGold = archersInGolden ∧
  yesToGold = truthfulArchersInGold + (swordsmenInGolden + archersInBlack) ∧
  yesToArcher = truthfulArchersInGold + lyingSwordsmenInGold

-- Conclusion
def Conclusion : Prop :=
  archersInGolden = 22

-- Proof statement
theorem ArcherInGoldenArmorProof : ProblemStatement → Conclusion :=
by
  sorry

end ArcherInGoldenArmorProof_l123_123931


namespace distance_between_rotated_line_and_given_line_l123_123423

noncomputable def intersection_point
  (A B C : ℝ) (y_val : ℝ) : ℝ × ℝ :=
let x_val := (C - B * y_val) / A in (x_val, y_val)

noncomputable def line_slope
  (A B : ℝ) : ℝ :=
-A / B

noncomputable def rotation_new_slope
  (m : ℝ) (θ : ℝ) : ℝ :=
let t := Real.tan θ in (m + t) / (1 - m * t)

def line_eq_through_point
  (m : ℝ) (P : ℝ × ℝ) : ℝ × ℝ × ℝ :=
(A, B, C) where
  A := -m
  B := 1
  C := -P.snd + m * P.fst

noncomputable def distance_between_parallel_lines
  (A B C1 C2 : ℝ) : ℝ :=
(abs (C2 - C1)) / sqrt (A * A + B * B)

theorem distance_between_rotated_line_and_given_line :
  let l := (3, -1, -6) in
  let l_rot := rotation_new_slope (line_slope l.1 l.2) (π / 4) in
  let P := intersection_point l.1 l.2 l.3 0 in
  let l1 := line_eq_through_point l_rot P in
  let other_line := (4, 2, 1) in
  distance_between_parallel_lines l1.1 l1.2 l1.3 other_line.3 = (9 * Real.sqrt 5) / 10 :=
by
  let l := (3, -1, -6)
  let l_rot := rotation_new_slope (line_slope l.1 l.2) (π / 4)
  let P := intersection_point l.1 l.2 l.3 0
  let l1 := line_eq_through_point l_rot P
  let other_line := (4, 2, 1)
  have h : distance_between_parallel_lines l1.1 l1.2 l1.3 other_line.3 = (9 * Real.sqrt 5) / 10, 
  from sorry
  exact h

end distance_between_rotated_line_and_given_line_l123_123423


namespace four_points_no_obtuse_triangle_l123_123776

noncomputable def probability_no_obtuse_triangle : ℝ :=
1 / 64

theorem four_points_no_obtuse_triangle (A B C D : circle) :
  (∀ (P Q : circle) (PQ_angle : ℝ), PQ_angle < π/2) → 
  probability_no_obtuse_triangle = 1 / 64 :=
sorry

end four_points_no_obtuse_triangle_l123_123776


namespace length_CD_in_quadrilateral_l123_123173

/-- 
Quadrilateral ABCD inscribed in a circle with radius 1, where diagonal AC
is a diameter of the circle, diagonal BD equals side AB, the diagonals intersect at P, 
and the length of PC is 2/5. Prove the length of side CD is 2√2/5.
-/
theorem length_CD_in_quadrilateral 
  (O : Point)
  (A B C D P : Point)
  (circle : Circle O 1)
  (diameter : Line O A ≠ Line O C)
  (AC_diameter : Line A C)
  (intersecting_point : A.x = P.x ∨ B.x = P.x ∨ C.x = P.x ∨ D.x = P.x)
  (diag_AB_eq_BD : dist A B = dist B D)
  (PC : dist P C = 2 / 5)
  (A_x : A.x = -1)
  (C_x : C.x = 1)
  (circle_equality : ∀ x : Point, (dist x O = 1) ↔ x ∈ circle) :
  dist C D = 2 * sqrt 2 / 5 := by
sorry

end length_CD_in_quadrilateral_l123_123173


namespace ratio_of_selling_price_to_production_cost_l123_123105

/-- Define the necessary variables and assumptions -/
variables (production_capacity daily_demand : ℕ)
          (production_cost selling_price : ℝ)
          (weekly_loss : ℝ)
          (days_per_week : ℕ)
          (profit_per_tire : ℝ)
          (ratio : ℝ)

-- Given assumptions
def conditions : Prop := 
  production_capacity = 1000 ∧
  daily_demand = 1200 ∧
  production_cost = 250 ∧
  weekly_loss = 175000 ∧
  days_per_week = 7 ∧
  profit_per_tire = 125 ∧
  selling_price = ratio * production_cost ∧
  weekly_loss = (daily_demand - production_capacity) * profit_per_tire * days_per_week

-- Proof to show the ratio of the selling price to the production cost
theorem ratio_of_selling_price_to_production_cost (h : conditions) : ratio = 1.5 :=
sorry

end ratio_of_selling_price_to_production_cost_l123_123105


namespace smallest_int_x_l123_123630

theorem smallest_int_x (x : ℤ) (h : 2 * x + 5 < 3 * x - 10) : x = 16 :=
sorry

end smallest_int_x_l123_123630


namespace ratio_of_speeds_l123_123319

theorem ratio_of_speeds (v_A v_B : ℝ) :
  (∀ t < : ℝ,
      (t = 3 → t = 12)
        → ((3 * v_A = abs (-600 + 3 * v_B)) ∧ (12 * v_A = abs (-600 + 12 * v_B)))
  → (v_A / v_B = 2 / 3) :=
sorry

end ratio_of_speeds_l123_123319


namespace f_2022_eq_0_l123_123425

-- Given function f defined on the reals
def f : ℝ → ℝ := sorry

-- Given conditions
axiom f_even : ∀ (x : ℝ), f(2 * x + 1) = f(-2 * x + 1)
axiom f_odd : ∀ (x : ℝ), f(x + 2) = -f(-x + 2)

-- Prove that f(2022) = 0
theorem f_2022_eq_0 : f 2022 = 0 := sorry

end f_2022_eq_0_l123_123425


namespace angle_triple_supplement_l123_123266

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123266


namespace shopping_mall_loss_l123_123679

theorem shopping_mall_loss :
  ∀ (a b : ℝ),
  a * (1.2 ^ 2) = 23.04 →
  b * (0.8 ^ 2) = 23.04 →
  a + b = 52 →
  2 * 23.04 = 46.08 →
  52 > 46.08 →
  (52 - 46.08 = 5.92) :=
begin
  intros a b ha hb hab current_revenue loss,
  exact sorry,
end

end shopping_mall_loss_l123_123679


namespace uranus_appears_7_minutes_after_6AM_l123_123610

def mars_last_seen := 0 * 60 + 10 -- 12:10 AM in minutes after midnight
def jupiter_after_mars := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def uranus_appearance := mars_last_seen + jupiter_after_mars + uranus_after_jupiter

theorem uranus_appears_7_minutes_after_6AM : uranus_appearance - (6 * 60) = 7 := by
  sorry

end uranus_appears_7_minutes_after_6AM_l123_123610


namespace five_digit_numbers_with_two_identical_digits_l123_123448

theorem five_digit_numbers_with_two_identical_digits : 
  ∃ n : ℕ, 
    n = 4032 ∧ 
    (∀ k : ℕ, 10000 ≤ k ∧ k < 100000 → k % 10000 / 1000 = 2 → 
    (∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
    (x ≠ 2 ∧ y ≠ 2 ∧ z ≠ 2 → 
    (k = 22000 + x * 1000 + y * 100 + z * 10 ∨ 
     k = 20000 + x * 10000 + 2000 + y * 100 + z * 10 ∨ 
     k = 20000 + x * 10000 + y * 1000 + 200 + z * 10 ∨ 
     k = 20000 + x * 10000 + y * 1000 + z * 100 + 20) ∨ 
    (x = k / 10000 ∧ y = (k % 10000) / 1000 ∧ z = (k % 1000) / 100 ∧
    k = 20000 + x * 1000 + x * 100 + y * 10 + z ∨ 
    k = 20000 + x * 1000 + y * 100 + x * 10 + z ∨ 
    k = 20000 + x * 1000 + y * 100 + z * 10 + x ∨ 
    k = 20000 + y * 1000 + y * 100 + x * 10 + x)))) :=
begin
  sorry
end

end five_digit_numbers_with_two_identical_digits_l123_123448


namespace minimum_shift_value_l123_123908

noncomputable def f : ℝ → ℝ := λ x, Real.sin (1 / 2 * x)
noncomputable def g : ℝ → ℝ := λ x, Real.cos (1 / 2 * x)

theorem minimum_shift_value (ϕ : ℝ) (hϕ : ϕ > 0) (h : ∀ x, f (x + ϕ) = g x) : ϕ = π := by
  have h1 : ∀ x, Real.sin (1 / 2 * (x + ϕ)) = Real.cos (1 / 2 * x), from h
  sorry

end minimum_shift_value_l123_123908


namespace angle_triple_supplementary_l123_123252

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l123_123252


namespace ticket_distribution_count_l123_123728

theorem ticket_distribution_count :
  ∃ (count : ℕ), count = 96 ∧
    (∃ (distributions : Finset (list (Fin 4))),
      ∀ (d : list (Fin 4)), d ∈ distributions →
        (length d = 5 ∧
         (∀ (p : Fin 4), (countp (λ x, x = p) d) ≥ 1) ∧
         (∀ (p : Fin 4), (countp (λ x, x = p) d) ≤ 2) ∧
         (∀ (p : Fin 4), (countp (λ x, x = p) d = 2 → ∃ (i j : ℕ), (abs (i - j) = 1))))) :=
begin
  sorry -- Proof goes here
end

end ticket_distribution_count_l123_123728


namespace ArcherInGoldenArmorProof_l123_123933

-- Definitions of the problem
variables (soldiers : Nat) (archers soldiersInGolden : Nat) 
variables (soldiersInBlack archersInGolden archersInBlack swordsmenInGolden swordsmenInBlack : Nat)
variables (truthfulSwordsmenInBlack lyingArchersInBlack lyingSwordsmenInGold truthfulArchersInGold : Nat)
variables (yesToGold yesToArcher yesToMonday : Nat)

-- Given conditions
def ProblemStatement : Prop :=
  soldiers = 55 ∧
  yesToGold = 44 ∧
  yesToArcher = 33 ∧
  yesToMonday = 22 ∧
  soldiers = archers + (soldiers - archers) ∧
  soldiers = soldiersInGolden + soldiersInBlack ∧
  archers = archersInGolden + archersInBlack ∧
  soldiersInGolden = archersInGolden + swordsmenInGolden ∧
  soldiersInBlack = archersInBlack + swordsmenInBlack ∧
  truthfulSwordsmenInBlack = swordsmenInBlack ∧
  lyingArchersInBlack = archersInBlack ∧
  lyingSwordsmenInGold = swordsmenInGolden ∧
  truthfulArchersInGold = archersInGolden ∧
  yesToGold = truthfulArchersInGold + (swordsmenInGolden + archersInBlack) ∧
  yesToArcher = truthfulArchersInGold + lyingSwordsmenInGold

-- Conclusion
def Conclusion : Prop :=
  archersInGolden = 22

-- Proof statement
theorem ArcherInGoldenArmorProof : ProblemStatement → Conclusion :=
by
  sorry

end ArcherInGoldenArmorProof_l123_123933


namespace min_value_x2_plus_y2_l123_123900

theorem min_value_x2_plus_y2 :
  ∀ x y : ℝ, (x + 5)^2 + (y - 12)^2 = 196 → x^2 + y^2 ≥ 1 :=
by
  intros x y h
  sorry

end min_value_x2_plus_y2_l123_123900


namespace find_n_from_binomial_condition_l123_123898

theorem find_n_from_binomial_condition (n : ℕ) (h : Nat.choose n 3 = 7 * Nat.choose n 1) : n = 43 :=
by
  -- The proof steps would be filled in here
  sorry

end find_n_from_binomial_condition_l123_123898


namespace quadratic_has_real_roots_l123_123376

open Real

theorem quadratic_has_real_roots (k : ℝ) (h : k ≠ 0) :
    ∃ x : ℝ, x^2 + k * x + k^2 - 1 = 0 ↔
    -2 / sqrt 3 ≤ k ∧ k ≤ 2 / sqrt 3 :=
by
  sorry

end quadratic_has_real_roots_l123_123376


namespace difference_face_local_value_8_l123_123309

theorem difference_face_local_value_8 :
  let numeral := 96348621
  let digit := 8
  let face_value := digit
  let position := 3  -- 0-indexed place for thousands
  let local_value := digit * 10^position
  local_value - face_value = 7992 :=
by
  let numeral := 96348621
  let digit := 8
  let face_value := digit
  let position := 3
  let local_value := digit * 10^position
  show local_value - face_value = 7992
  sorry

end difference_face_local_value_8_l123_123309


namespace sum_simplest_form_probability_eq_7068_l123_123332

/-- A jar has 15 red candies and 20 blue candies. Terry picks three candies at random,
    then Mary picks three of the remaining candies at random.
    Given that the probability that they get the same color combination (all reds or all blues, irrespective of order),
    find this probability in the simplest form. The sum of the numerator and denominator in simplest form is: 7068. -/
noncomputable def problem_statement : Nat :=
  let total_candies := 15 + 20;
  let terry_red_prob := (15 * 14 * 13) / (total_candies * (total_candies - 1) * (total_candies - 2));
  let mary_red_prob := (12 * 11 * 10) / ((total_candies - 3) * (total_candies - 4) * (total_candies - 5));
  let both_red := terry_red_prob * mary_red_prob;

  let terry_blue_prob := (20 * 19 * 18) / (total_candies * (total_candies - 1) * (total_candies - 2));
  let mary_blue_prob := (17 * 16 * 15) / ((total_candies - 3) * (total_candies - 4) * (total_candies - 5));
  let both_blue := terry_blue_prob * mary_blue_prob;

  let total_probability := both_red + both_blue;
  let simplest := 243 / 6825; -- This should be simplified form
  243 + 6825 -- Sum of numerator and denominator

theorem sum_simplest_form_probability_eq_7068 : problem_statement = 7068 :=
by sorry

end sum_simplest_form_probability_eq_7068_l123_123332


namespace parallel_lines_not_equal_l123_123239

-- Definitions based on the conditions
structure line (P : Type) [plane P] :=
  (is_infinitely_extending : Prop)

def parallel (L1 L2 : line P) [plane P] :=
  (L1 ≠ L2) ∧ ∀ (p : P), ¬ (L1 ∈ p ∧ L2 ∈ p)

-- The Lean statement for the problem
theorem parallel_lines_not_equal {P : Type} [plane P] (L1 L2 : line P) :
  parallel L1 L2 → ¬(L1 = L2) :=
by
  intro h
  -- Given that lines extend infinitely and cannot be measured
  have h_inf : L1.is_infinitely_extending ∧ L2.is_infinitely_extending := sorry
  -- From the definition of parallel lines
  have h_parallel : ∀ (p : P), ¬ (L1 ∈ p ∧ L2 ∈ p) := h.2
  -- Assume lines L1 and L2 were equal, which leads to a contradiction
  contradiction
  -- Conclude the lines L1 and L2 are not equal
  sorry

end parallel_lines_not_equal_l123_123239


namespace proof_1_proof_2_l123_123092

noncomputable def triangle_setup (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = π ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 = b^2 + c^2 - 2 * b * c * cos A ∧
  b^2 = a^2 + c^2 - 2 * a * c * cos B ∧
  c^2 = a^2 + b^2 - 2 * a * b * cos C

theorem proof_1 {A B C a b c : ℝ} (h : triangle_setup A B C a b c) (h1 : a * cos B - b * cos A = (1/2) * c) : (tan A / tan B = 3) :=
by
  -- proof steps needed
  sorry

theorem proof_2 {A B C a b c : ℝ} (h : triangle_setup A B C a b c) (h1 : a * cos B - b * cos A = (1/2) * c) : (∀ x, x ∈ {y : ℝ | ∃(tan B = y), (tan (A - B) ≤ (sqrt 3) / 3})) :=
by
  -- proof steps needed
  sorry

end proof_1_proof_2_l123_123092


namespace complement_union_l123_123722

def R := Set ℝ

def A : Set ℝ := {x | x ≥ 1}

def B : Set ℝ := {y | ∃ x, x ≥ 1 ∧ y = Real.exp x}

theorem complement_union (R : Set ℝ) (A : Set ℝ) (B : Set ℝ) :
  (A ∪ B)ᶜ = {x | x < 1} := by
  sorry

end complement_union_l123_123722


namespace candy_distribution_l123_123345

theorem candy_distribution (a b c : ℕ) (w1 w2 w3 : ℤ) (total_weight : ℤ) :
  total_weight = 185 ∧ w1 = 16 ∧ w2 = 17 ∧ w3 = 21 ∧
  (total_weight = a * w1 + b * w2 + c * w3) ->
  (∃ a b c : ℕ, a = 5 ∧ b = 0 ∧ c = 5) :=
by
  intros h,
  sorry

end candy_distribution_l123_123345


namespace theater_ticket_difference_l123_123347

theorem theater_ticket_difference
  (O B : ℕ)
  (h1 : O + B = 355)
  (h2 : 12 * O + 8 * B = 3320) :
  B - O = 115 :=
sorry

end theater_ticket_difference_l123_123347


namespace pinwheel_area_l123_123389

theorem pinwheel_area (A B C D : ℝ × ℝ)
  (hA : A = (0, 6))
  (hB : B = (6, 0))
  (hC : C = (0, 0))
  (hD : D = (6, 6))
  (center : ℝ × ℝ)
  (h_center : center = (3, 3)) :
  let triangle_area (a b c : ℝ × ℝ) :=
    1 / 2 * | a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2) | in
  let area_one_triangle := triangle_area center A B in
  let total_area := 4 * area_one_triangle in
  total_area = 36 :=
by
  have h1: triangle_area (3, 3) (0, 6) (6, 0) = 9, from sorry,
  have h2: 4 * 9 = 36, by norm_num,
  show 4 * (triangle_area (3, 3) (0, 6) (6, 0)) = 36, from by rw [h1, h2]

end pinwheel_area_l123_123389


namespace archers_in_golden_armor_count_l123_123948

theorem archers_in_golden_armor_count : 
  ∃ (archers_golden : ℕ), 
    let total_soldiers := 55 in
    let golden_armor_affirmative := 44 in
    let archer_affirmative := 33 in
    let monday_affirmative := 22 in
    let number_archers_golden := archers_golden in
    (number_archers_golden = 22) ∧ 
    ∃ (archer : ℕ) (swordsman : ℕ) (golden : ℕ) (black : ℕ),
      archer + swordsman = total_soldiers ∧
      golden + black = total_soldiers ∧
      golden + black = total_soldiers ∧
      (swordsman * golden + archer * golden = golden_armor_affirmative) ∧
      (swordsman * archer + swordsman * golden = archer_affirmative) ∧
      ((!monday_affirmative ∧ swordsman * golden) + (!monday_affirmative ∧ archer * black) = monday_affirmative)
:= sorry

end archers_in_golden_armor_count_l123_123948


namespace range_of_a_l123_123024

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x - a > 0) ↔ a ∈ set.Ioo (-4) 0 :=
begin
  sorry
end

end range_of_a_l123_123024


namespace circumcircle_bisects_BP_l123_123497

variable (A B C P Q M : Type)
variables [acute_triangle : ∀ A B C : Type, Prop] 
variables [circumcircle : ∀ A B C : Type, Prop]
variable (AB_gt_BC : ∀ A B C : Type, Prop)
variables [midpoints : ∀ A B C P Q : Type, Prop]
variables [perpendicular : ∀ Q M AB : Type, Prop]

theorem circumcircle_bisects_BP :
  acute_triangle A B C →
  circumcircle A B C →
  AB_gt_BC A B C →
  midpoints A C P Q →
  perpendicular Q M AB →
  ∃ S, midpoint S B P ∧ concyclic B M C S :=
  sorry

end circumcircle_bisects_BP_l123_123497


namespace point_of_tangency_l123_123346

theorem point_of_tangency : 
    ∃ (m n : ℝ), 
    (∀ x : ℝ, x ≠ 0 → n = 1 / m ∧ (-1 / m^2) = (n - 2) / (m - 0)) ∧ 
    m = 1 ∧ n = 1 :=
by
  sorry

end point_of_tangency_l123_123346


namespace expected_value_area_std_dev_area_cm_l123_123670

-- Data definitions based on conditions
def X : ℝ := 2
def Y : ℝ := 1
def Var_X : ℝ := (0.003)^2
def Var_Y : ℝ := (0.002)^2

-- Expected value definition
def E_X : ℝ := 2
def E_Y : ℝ := 1

-- Problem statement: Expected value of the area
theorem expected_value_area : (E_X * E_Y) = 2 := by 
  sorry

-- Problem statement: Standard deviation of the area
theorem std_dev_area_cm : 
  (sqrt ((E_X ^ 2) * Var_Y + (E_Y ^ 2) * Var_X + Var_X * Var_Y)) * 100 = 50 := by
  sorry

end expected_value_area_std_dev_area_cm_l123_123670


namespace weeks_project_lasts_l123_123042

-- Definition of the conditions
def meal_cost : ℤ := 4
def people : ℤ := 4
def days_per_week : ℤ := 5
def total_spent : ℤ := 1280
def weekly_cost : ℤ := meal_cost * people * days_per_week

-- Problem statement: prove that the number of weeks the project will last equals 16 weeks.
theorem weeks_project_lasts : total_spent / weekly_cost = 16 := by 
  sorry

end weeks_project_lasts_l123_123042


namespace max_projection_x_axis_l123_123503

theorem max_projection_x_axis (A P : ℝ × ℝ) (x y : ℝ)
  (hA : A = (x, y))
  (hEllipse : x^2 / 16 + y^2 / 4 = 1)
  (λ : ℝ)
  (hOP : P = (λ * x, λ * y))
  (hOA_OP : x * (λ * x) + y * (λ * y) = 6) :
  ∃ M : ℝ, M = √3 ∧ ∀ P', P' = (m, _) → m ≤ M :=
sorry

end max_projection_x_axis_l123_123503


namespace spiral_grid_last_column_sum_l123_123554

-- Define properties of the grid and the spiral fill
def in_spiral_grid (n: ℕ) (pos: ℕ × ℕ) : ℕ :=
  sorry -- A function that would return the number at position pos in an n x n spiral grid

-- Numerical grid parameters
def grid_size := 15
def center_row := 8
def center_col := 8

-- Positions in the last column
def bottom_right : ℕ × ℕ := (grid_size, grid_size)
def top_right : ℕ × ℕ := (1, grid_size)

-- The theorem statement
theorem spiral_grid_last_column_sum :
  in_spiral_grid grid_size bottom_right + in_spiral_grid grid_size top_right = 436 :=
  sorry

end spiral_grid_last_column_sum_l123_123554


namespace bc_product_l123_123228

theorem bc_product (b c : ℤ) : (∀ r : ℝ, r^2 - r - 2 = 0 → r^4 - b * r - c = 0) → b * c = 30 :=
by
  sorry

end bc_product_l123_123228


namespace total_spent_correct_l123_123177

def shorts : ℝ := 13.99
def shirt : ℝ := 12.14
def jacket : ℝ := 7.43
def total_spent : ℝ := 33.56

theorem total_spent_correct : shorts + shirt + jacket = total_spent :=
by
  sorry

end total_spent_correct_l123_123177


namespace distance_between_midpoints_l123_123365

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_midpoints :
  let m1 := midpoint (1, -1) (3, 2)
  let m2 := midpoint (4, -1) (7, 2)
  distance m1 m2 = 3.5 :=
by
  let m1 := midpoint (1, -1) (3, 2)
  let m2 := midpoint (4, -1) (7, 2)
  have h_mid1 : m1 = (2, 0.5) := by sorry
  have h_mid2 : m2 = (5.5, 0.5) := by sorry
  show distance m1 m2 = 3.5 from by
    rw [h_mid1, h_mid2]
    sorry

end distance_between_midpoints_l123_123365


namespace angle_triple_supplementary_l123_123250

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l123_123250


namespace arithmetic_sequence_properties_geometric_sequence_properties_l123_123410

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ :=
  2 * n - 1

-- Define the sum of the first n terms of {a_n}
def S (n : ℕ) : ℕ :=
  n ^ 2

-- Prove the nth term and the sum of the first n terms of {a_n}
theorem arithmetic_sequence_properties (n : ℕ) :
  a n = 2 * n - 1 ∧ S n = n ^ 2 :=
by sorry

-- Define the geometric sequence {b_n}
def b (n : ℕ) : ℕ :=
  2 ^ (2 * n - 1)

-- Define the sum of the first n terms of {b_n}
def T (n : ℕ) : ℕ :=
  (2 ^ n * (4 ^ n - 1)) / 3

-- Prove the nth term and the sum of the first n terms of {b_n}
theorem geometric_sequence_properties (n : ℕ) (a4 S4 : ℕ) (q : ℕ)
  (h_a4 : a4 = a 4)
  (h_S4 : S4 = S 4)
  (h_q : q ^ 2 - (a4 + 1) * q + S4 = 0) :
  b n = 2 ^ (2 * n - 1) ∧ T n = (2 ^ n * (4 ^ n - 1)) / 3 :=
by sorry

end arithmetic_sequence_properties_geometric_sequence_properties_l123_123410


namespace four_points_no_obtuse_triangle_l123_123780

noncomputable def probability_no_obtuse_triangle : ℝ :=
1 / 64

theorem four_points_no_obtuse_triangle (A B C D : circle) :
  (∀ (P Q : circle) (PQ_angle : ℝ), PQ_angle < π/2) → 
  probability_no_obtuse_triangle = 1 / 64 :=
sorry

end four_points_no_obtuse_triangle_l123_123780


namespace quilt_patch_cost_l123_123509

-- Definitions of the conditions
def length : ℕ := 16
def width : ℕ := 20
def patch_area : ℕ := 4
def cost_first_10 : ℕ := 10
def cost_after_10 : ℕ := 5
def num_first_patches : ℕ := 10

-- Define the calculations based on the problem conditions
def quilt_area : ℕ := length * width
def total_patches : ℕ := quilt_area / patch_area
def cost_first : ℕ := num_first_patches * cost_first_10
def remaining_patches : ℕ := total_patches - num_first_patches
def cost_remaining : ℕ := remaining_patches * cost_after_10
def total_cost : ℕ := cost_first + cost_remaining

-- Statement of the proof problem
theorem quilt_patch_cost : total_cost = 450 := by
  -- Placeholder for the proof
  sorry

end quilt_patch_cost_l123_123509


namespace triple_supplementary_angle_l123_123278

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l123_123278


namespace f_monotonically_decreasing_intervals_f_α_value_l123_123433

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := sin (π/4 + x) * sin (π/4 - x)

-- Proof that f(x) is monotonically decreasing in the given intervals
theorem f_monotonically_decreasing_intervals :
  ∀ (k : ℤ), ∀ (x : ℝ), k * π ≤ x ∧ x ≤ k * π + π/2 → f' x < 0 := 
by sorry

-- Definition and condition for α 
def α (α : ℝ) : Prop := 0 < α ∧ α < π/2 ∧ sin (α - π/4) = 1/2

-- Proof for the value of f(α) given α is an acute angle satisfying the condition
theorem f_α_value (α : ℝ) (hα : α α) : f α = - √3 / 2 :=
by sorry

end f_monotonically_decreasing_intervals_f_α_value_l123_123433


namespace determine_winner_games_l123_123142

theorem determine_winner_games (T : ℕ) (hT : T = 20) : 
  (games_played_to_determine_winner T = 19) := 
by
  sorry

def games_played_to_determine_winner (teams : ℕ) : ℕ :=
  teams - 1

end determine_winner_games_l123_123142


namespace num_archers_golden_armor_proof_l123_123961
noncomputable section

structure Soldier :=
  (is_archer : Bool)
  (is_golden : Bool)
  (tells_truth : Bool)

def count_soldiers (soldiers : List Soldier) : Nat :=
  soldiers.length

def count_truthful_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => s.tells_truth)).count q

def count_lying_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => ¬s.tells_truth)).count q

def num_archers_golden_armor (soldiers : List Soldier) : Nat :=
  (soldiers.filter (λ s => s.is_archer ∧ s.is_golden)).length

theorem num_archers_golden_armor_proof (soldiers : List Soldier)
  (h1 : count_soldiers soldiers = 55)
  (h2 : count_truthful_responses soldiers (λ s => s.is_golden) + 
        count_lying_responses soldiers (λ s => s.is_golden) = 44)
  (h3 : count_truthful_responses soldiers (λ s => s.is_archer) + 
        count_lying_responses soldiers (λ s => s.is_archer) = 33)
  (h4 : count_truthful_responses soldiers (λ s => true) + 
        count_lying_responses soldiers (λ s => false) = 22) :
  num_archers_golden_armor soldiers = 22 := by
  sorry

end num_archers_golden_armor_proof_l123_123961


namespace probability_no_obtuse_triangle_is_9_over_64_l123_123811

noncomputable def probability_no_obtuse_triangle (A0 A1 A2 A3 : ℝ) : ℝ :=
  -- Define the probabilistic model according to the problem conditions
  -- Assuming A0, A1, A2, and A3 are positions of points on the circle parametrized by angles in radians
  let θ := real.angle.lower_half (A1 - A0) in
  let prob_A2 := (π - θ) / (2 * π) in
  let prob_A3 := (π - θ) / (2 * π) in
  (1 / 2) * (prob_A2 * prob_A3)

theorem probability_no_obtuse_triangle_is_9_over_64 :
  probability_no_obtuse_triangle A0 A1 A2 A3 = 9 / 64 :=
by sorry

end probability_no_obtuse_triangle_is_9_over_64_l123_123811


namespace archers_in_golden_l123_123940

variables (soldiers archers swordsmen wearing_golden wearing_black : ℕ)
variables (truth_swordsmen_black lie_swordsmen_golden lie_archers_black truth_archers_golden : ℕ)

-- Given conditions
variables (cond1 : archers + swordsmen = 55)
variables (cond2 : wearing_golden + wearing_black = 55)
variables (cond3 : truth_swordsmen_black + lie_swordsmen_golden + lie_archers_black + truth_archers_golden = 55)
variables (cond4 : wearing_golden = 44)
variables (cond5 : archers = 33)
variables (cond6 : truth_swordsmen_black + lie_archers_black = 22)

-- Define the mathematic equivalent proof problem
theorem archers_in_golden : archers = 22 :=
by
  sorry

end archers_in_golden_l123_123940


namespace intersection_eq_2_l123_123866

def M : Set ℝ := {x : ℝ | -2 < 2 * x - 1 ∧ 2 * x - 1 < 5}
def N : Set ℕ := {x : ℕ | -1 < x ∧ x < 8}

theorem intersection_eq_2 : (M ∩ N : Set ℝ) = {2} := sorry

end intersection_eq_2_l123_123866


namespace inequality_a_b_l123_123537

theorem inequality_a_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ sqrt (a * b) + sqrt ((a^2 + b^2) / 2) :=
sorry

end inequality_a_b_l123_123537


namespace positive_integers_count_l123_123451

theorem positive_integers_count :
  let condition1 (n : ℕ) := (150 * n) ^ 40 > n ^ 80
  let condition2 (n : ℕ) := n ^ 80 > 3 ^ 160
  ∃ f : Finite {n : ℕ | condition1 n ∧ condition2 n}, (f.support.card = 140) :=
begin
  sorry
end

end positive_integers_count_l123_123451


namespace ratio_of_vacations_to_classes_l123_123874

-- Translating the conditions into definitions
def GrantVacations := Nat
def KelvinClasses : Nat := 90
def TotalVacationsAndClasses (V : GrantVacations) : Prop := V + KelvinClasses = 450

-- Statement of the problem as a Lean theorem
theorem ratio_of_vacations_to_classes (V : GrantVacations) (h : TotalVacationsAndClasses V) : V / KelvinClasses = 4 :=
by
  sorry

end ratio_of_vacations_to_classes_l123_123874


namespace count_ways_to_ensure_sum_at_least_6_l123_123708

-- Defining the list of capacities and the target sum.
def capacities : List ℕ := [1, 1, 2, 2, 3, 4]

-- Function to calculate the number of ways to choose 3 cables with sum at least 6.
def count_ways (lst : List ℕ) (k : ℕ) (n : ℕ) : ℕ :=
  (lst.combinations k).count (λ x => x.sum ≥ n)

theorem count_ways_to_ensure_sum_at_least_6 :
  count_ways capacities 3 6 = 15 := 
sorry


end count_ways_to_ensure_sum_at_least_6_l123_123708


namespace number_of_dolls_is_18_l123_123223

def total_toys : ℕ := 24
def fraction_action_figures : ℚ := 1 / 4
def number_action_figures : ℕ := (fraction_action_figures * total_toys).to_nat
def number_dolls : ℕ := total_toys - number_action_figures

theorem number_of_dolls_is_18 :
  number_dolls = 18 :=
by
  sorry

end number_of_dolls_is_18_l123_123223


namespace three_digit_integers_with_product_36_l123_123878

/--
There are 21 distinct 3-digit integers such that the product of their digits equals 36, and each digit is between 1 and 9.
-/
theorem three_digit_integers_with_product_36 : 
  ∃ n : ℕ, digit_product_count 36 3 n ∧ n = 21 :=
sorry

end three_digit_integers_with_product_36_l123_123878


namespace probability_pair_product_multiple_of_12_l123_123064

noncomputable def pair_product_multiple_of_12 (a b : ℕ) : Prop :=
  ∃ k, a * b = 12 * k

theorem probability_pair_product_multiple_of_12 :
  ∃ k : ℚ, k = 2 / 3 ∧ 
           (let S := {4, 6, 8, 9};
                all_pairs := [(4, 6), (4, 8), (4, 9), (6, 8), (6, 9), (8, 9)];
                pairs_with_mult_12 := [(4, 6), (4, 9), (6, 8), (8, 9)];
                total_pairs := (1 / 2) * 4 * (4 - 1) in
            ((length (filter (λ pair, pair_product_multiple_of_12 (fst pair) (snd pair)) all_pairs) : ℚ) / total_pairs) = k) :=
begin
  sorry
end

end probability_pair_product_multiple_of_12_l123_123064


namespace fewer_than_2n_three_halves_pairs_l123_123832

theorem fewer_than_2n_three_halves_pairs (n : ℕ) (points : fin n → ℝ × ℝ) :
  (∑ i : fin n, (finset.filter (λ j : fin n, dist (points i) (points j) = 1) finset.univ).card < 2 * n^(3/2 : ℝ)) := by
sorry

end fewer_than_2n_three_halves_pairs_l123_123832


namespace cookie_distribution_l123_123655

theorem cookie_distribution : 
  ∀ (n c T : ℕ), n = 6 → c = 4 → T = n * c → T = 24 :=
by 
  intros n c T h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end cookie_distribution_l123_123655


namespace earnings_per_hour_l123_123657

-- Define the conditions
def widgetsProduced : Nat := 750
def hoursWorked : Nat := 40
def totalEarnings : ℝ := 620
def earningsPerWidget : ℝ := 0.16

-- Define the proof goal
theorem earnings_per_hour :
  ∃ H : ℝ, (hoursWorked * H + widgetsProduced * earningsPerWidget = totalEarnings) ∧ H = 12.5 :=
by
  sorry

end earnings_per_hour_l123_123657


namespace problem_example_l123_123079

noncomputable section

open Real

def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (C.1 - A.1) = k * (B.1 - A.1) ∧ (C.2 - A.2) = k * (B.2 - A.2)

def lambda_condition (λ : ℝ) : Prop :=
  (λ = 1 - 1 / 3)

def g (m : ℝ) : ℝ :=
  if m ≤ 0 then 1
  else if m < 1 then 1 - m^2
  else 2 - 2 * m

theorem problem_example (λ : ℝ) (A B : ℝ × ℝ) (x m : ℝ) (f : ℝ → ℝ) :
  A = (1, cos x) → 
  B = (1 + cos x, cos x) →
  x ∈ Icc 0 (π / 2) →
  ∃ λ, λ_condition λ ∧ λ = 2 / 3 →
  f x = A.1 * (1/3 * A.1 + λ * B.1) + A.2 * (1/3 * A.2 + λ * B.2) - (2 * m + 1 / 3) * (abs (B.1 - A.1)) →
  (∀ x, f x = cos x ^ 2 - 2 * m * cos x + 1) →
  ∃ m, ∀ m, max (g m) = 1 :=
by sorry

end problem_example_l123_123079


namespace volume_ratio_l123_123690

-- Definition of the volumes
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
def volume_hemisphere (r : ℝ) : ℝ := (1 / 2) * volume_sphere (3 * r)

-- Problem statement: Prove the ratio of the volume of the sphere to the volume of the hemisphere is 1 / 13.5
theorem volume_ratio (r : ℝ) : (volume_sphere r) / (volume_hemisphere r) = 1 / 13.5 := 
by
  sorry

end volume_ratio_l123_123690


namespace range_of_a_l123_123896

noncomputable theory

-- Definitions derived from conditions:
def condition1 (x y : ℝ) : Prop := x > 0 ∧ y > 0
def condition2 (x y : ℝ) : Prop := x + y + 8 = x * y
def condition3 (x y a : ℝ) : Prop := (x + y)^2 - a * (x + y) + 1 ≥ 0

-- The mathematically equivalent proof problem:
theorem range_of_a (x y a : ℝ) (hx : condition1 x y) (hy : condition2 x y) (hz : condition3 x y a) : a ≤ 65 / 8 :=
sorry

end range_of_a_l123_123896


namespace probability_no_obtuse_triangle_l123_123805

namespace CirclePoints

noncomputable def no_obtuse_triangle_probability : ℝ := 
  let p := 1/64 in
    p

theorem probability_no_obtuse_triangle (X : ℕ → ℝ) (hcirc : ∀ n, 0 ≤ X n ∧ X n < 2 * π) (hpoints : (∀ n m, n ≠ m → X n ≠ X m)) :
  no_obtuse_triangle_probability = 1/64 :=
sorry

end CirclePoints

end probability_no_obtuse_triangle_l123_123805


namespace largest_m_for_divisibility_property_l123_123442

theorem largest_m_for_divisibility_property : ∃ m, (∀ (S : Finset ℕ), S.card = 1000 - m → ∃ a b ∈ S, a ∣ b ∨ b ∣ a) ∧ m = 499 :=
by
  sorry

end largest_m_for_divisibility_property_l123_123442


namespace smallest_four_digit_multiple_of_18_l123_123756

-- Define the concept of a four-digit number
def four_digit (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

-- Define the concept of a multiple of 18
def multiple_of_18 (N : ℕ) : Prop := ∃ k : ℕ, N = 18 * k

-- Define the combined condition for N being a four-digit multiple of 18
def four_digit_multiple_of_18 (N : ℕ) : Prop := four_digit N ∧ multiple_of_18 N

-- State that 1008 is the smallest such number
theorem smallest_four_digit_multiple_of_18 : ∀ N : ℕ, four_digit_multiple_of_18 N → 1008 ≤ N := 
by
  intros N H
  sorry

end smallest_four_digit_multiple_of_18_l123_123756


namespace find_f_m_eq_neg_one_l123_123435

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^(2 - m)

theorem find_f_m_eq_neg_one (m : ℝ)
  (h1 : ∀ x : ℝ, f x m = - f (-x) m) (h2 : m^2 - m = 3 + m) :
  f m m = -1 :=
by
  sorry

end find_f_m_eq_neg_one_l123_123435


namespace repeating_decimal_to_fraction_l123_123385

theorem repeating_decimal_to_fraction : (6 + 81 / 99) = 75 / 11 := 
by 
  sorry

end repeating_decimal_to_fraction_l123_123385


namespace man_l123_123303

-- Define the conditions as variables
variables (V_with_current V_current V_man V_against : ℝ)

-- Define the given conditions using hypotheses
def conditions := V_with_current = 15 ∧ V_current = 2.8 ∧ V_with_current = V_man + V_current

-- Lean statement of the equivalent proof problem
theorem man's_speed_against_current
  (h : conditions V_with_current V_current V_man V_against) :
  V_against = V_man - V_current → V_against = 9.4 :=
by
  cases h with h1 h2 h3
  cases h2 with h4 h5
  sorry

end man_l123_123303


namespace angle_triple_supplement_l123_123272

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l123_123272


namespace three_digit_isosceles_count_l123_123541

theorem three_digit_isosceles_count : 
  let count := (Nat.choose 9 2) * 2 - 20 + 9 in
  count = 165 :=
by
  let valid_triplets := 9 + (2 * (Nat.choose 9 2 - 10)) -- Calculations are being performed initially and adjusted as per conditions
  have : valid_triplets = 165 := sorry -- Specified exact checks and calculations
  exact this

end three_digit_isosceles_count_l123_123541


namespace expected_sides_general_expected_sides_rectangle_l123_123622

-- General Problem
theorem expected_sides_general (n k : ℕ) : 
  (∀ n k : ℕ, n ≥ 3 → k ≥ 0 → (1:ℝ) / ((k + 1:ℝ)) * (n + 4 * k:ℝ) ≤ (n + 4 * k) / (k + 1)) := 
begin
  sorry
end

-- Specific Problem for Rectangle
theorem expected_sides_rectangle (k : ℕ) :
  (∀ k : ℕ, k ≥ 0 → 4 = (4 + 4 * k) / (k + 1)) := 
begin
  sorry
end

end expected_sides_general_expected_sides_rectangle_l123_123622


namespace value_of_f_x0_l123_123851

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) := ∀ ⦃x y : ℝ⦄, x ∈ s → y ∈ s → x < y → f x < f y

noncomputable def f (x : ℝ) := 2^x - log (1/2) x

theorem value_of_f_x0 {a x_0 : ℝ} (h1 : 0 < x_0) (h2 : x_0 < a) 
  (h3 : is_increasing_on f (Set.Ioi 0)) 
  (h4 : f a = 0) : 
  f x_0 < 0 :=
sorry

end value_of_f_x0_l123_123851


namespace best_regression_effect_l123_123484

theorem best_regression_effect (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.36)
  (h2 : R2_2 = 0.95)
  (h3 : R2_3 = 0.74)
  (h4 : R2_4 = 0.81):
  max (max (max R2_1 R2_2) R2_3) R2_4 = 0.95 := by
  sorry

end best_regression_effect_l123_123484


namespace angle_triple_supplement_l123_123263

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123263


namespace inequality_for_positive_nums_l123_123164

theorem inequality_for_positive_nums 
    (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
    a^2 / b + c^2 / d ≥ (a + c)^2 / (b + d) :=
by
  sorry

end inequality_for_positive_nums_l123_123164


namespace find_segment_length_l123_123402

-- Define the standard equation of the hyperbola and its left vertex.
def hyperbola_focus : ℝ × ℝ := (-3, 0)

-- Define the standard equation of the parabola with its focus as given by the hyperbola's vertex.
def parabola_eq (y x: ℝ) : Prop := y^2 = -12 * x

-- Define the line equation passing through point (0, 2) with slope 1.
def line_eq (y x: ℝ) : Prop := y = x + 2

-- Define the segment length AB obtained from intersection points.
def segment_length (x1 x2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + ((x1 + 2) - (x2 + 2))^2)

theorem find_segment_length :
    ∃ (x1 x2: ℝ), parabola_eq (x1 + 2) x1 ∧ parabola_eq (x2 + 2) x2 ∧ segment_length x1 x2 = 4 * real.sqrt 30 :=
by
  sorry

end find_segment_length_l123_123402


namespace P_neg1_le_ξ_le_0_l123_123853

noncomputable def normal_dist (mu sigma : ℝ) : Prop := sorry

variable (σ : ℝ)
variable (ξ : ℝ → Prop)

axiom ξ_normal : normal_dist 0 σ^2
axiom P_ξ_gte_1 : ℙ {ω | ξ ω ⟨1⟩} = 0.4 

theorem P_neg1_le_ξ_le_0 : ℙ {ω | -1 ≤ ξ ω ∧ ξ ω ≤ 0} = 0.1 := 
by
  exact sorry

end P_neg1_le_ξ_le_0_l123_123853


namespace multiples_of_6_and_8_l123_123044

open Nat

theorem multiples_of_6_and_8 (n m k : ℕ) (h₁ : n = 33) (h₂ : m = 25) (h₃ : k = 8) :
  (n - k) + (m - k) = 42 :=
by
  sorry

end multiples_of_6_and_8_l123_123044


namespace algebraic_expression_value_l123_123051

-- Define the given condition
def condition (a b : ℝ) : Prop := a + b - 2 = 0

-- State the theorem to prove the algebraic expression value
theorem algebraic_expression_value (a b : ℝ) (h : condition a b) : a^2 - b^2 + 4 * b = 4 := by
  sorry

end algebraic_expression_value_l123_123051


namespace sigma_algebra_generated_by_F_and_C_l123_123527

variable {Ω : Type} -- The underlying set

def is_sigma_algebra (F : set (set Ω)) : Prop :=
  ∀ (A B : set Ω), A ∈ F ∧ B ∈ F → (A ∩ B ∈ F ∧ A ∪ B ∈ F ∧ -A ∈ F) ∧ (⋃₀ F ∈ F ∧ ⋂₀ F ∈ F)

variable (F : set (set Ω)) (C : set Ω)

theorem sigma_algebra_generated_by_F_and_C (h_F : is_sigma_algebra F) (h_notin : C ∉ F) :
  let S := {A ∩ C ∪ B ∩ -C | A B ∈ F}
  in (is_sigma_algebra (sigma_algebra F ∪ {C})) →
      sigma_algebra F ∪ {C} = S := sorry

end sigma_algebra_generated_by_F_and_C_l123_123527


namespace tangerine_count_l123_123233

-- Definitions based directly on the conditions
def initial_oranges : ℕ := 5
def remaining_oranges : ℕ := initial_oranges - 2
def remaining_tangerines (T : ℕ) : ℕ := T - 10
def condition1 (T : ℕ) : Prop := remaining_tangerines T = remaining_oranges + 4

-- Theorem to prove the number of tangerines in the bag
theorem tangerine_count (T : ℕ) (h : condition1 T) : T = 17 :=
by
  sorry

end tangerine_count_l123_123233


namespace probability_no_obtuse_triangle_l123_123804

namespace CirclePoints

noncomputable def no_obtuse_triangle_probability : ℝ := 
  let p := 1/64 in
    p

theorem probability_no_obtuse_triangle (X : ℕ → ℝ) (hcirc : ∀ n, 0 ≤ X n ∧ X n < 2 * π) (hpoints : (∀ n m, n ≠ m → X n ≠ X m)) :
  no_obtuse_triangle_probability = 1/64 :=
sorry

end CirclePoints

end probability_no_obtuse_triangle_l123_123804


namespace point_of_intersection_of_asymptotes_l123_123747

noncomputable def asymptotes_intersection_point : ℝ × ℝ := (3, 1)

theorem point_of_intersection_of_asymptotes (x y : ℝ) (h : y = (x ^ 2 - 6 * x + 8) / (x ^ 2 - 6 * x + 9)) (hx : x = 3) (hy : y = 1) :
  (x, y) = (3, 1) :=
by
  rw [hx, hy]
  simp
  exact asymptotes_intersection_point
  sorry -- Proof steps are not provided, only statement is required.

end point_of_intersection_of_asymptotes_l123_123747


namespace circle_standard_equation_l123_123399

theorem circle_standard_equation (a : ℝ) (h₀ : (x - a)^2 + y^2 = 25)
  (h₁ : ∃(y : ℝ), (0, y) ∈ set_of (λ p, (p.1 - a)^2 + p.2^2 = 25) ∧ abs y = 4) :
  (a = 3 ∨ a = -3) → ( ∃ x y, (x - 3)^2 + y^2 = 25) ∨ ( ∃ x y, (x + 3)^2 + y^2 = 25) :=
by
  sorry

end circle_standard_equation_l123_123399


namespace simple_interest_rate_l123_123895

-- Main statement
theorem simple_interest_rate (P : ℕ) : 
  let T := 10 in
  let SI := P in
  let finalAmount := 2 * P in
  let simpleInterest := (P * 10 * R) / 100 in
  finalAmount = P + simpleInterest →
  R = 10 := 
by
  sorry

end simple_interest_rate_l123_123895


namespace time_to_clear_each_other_from_meeting_l123_123669

-- Define the problem conditions
def length_car1 : ℝ := 120 -- meters
def length_car2 : ℝ := 280 -- meters
def speed_car1 : ℝ := 42 * 1000 / 3600 -- converting km/h to m/s
def speed_car2 : ℝ := 30 * 1000 / 3600 -- converting km/h to m/s

-- State the theorem that needs to be proven
theorem time_to_clear_each_other_from_meeting : 
  let total_distance := length_car1 + length_car2 in
  let relative_speed := speed_car1 + speed_car2 in
  (total_distance / relative_speed) = 20 :=
by 
  -- Insert proof here
  sorry

end time_to_clear_each_other_from_meeting_l123_123669


namespace minimal_polynomial_degree_l123_123686

def is_root (p : Polynomial ℚ) (x : ℝ) : Prop := Polynomial.aeval x p = 0

theorem minimal_polynomial_degree :
  ∃ p : Polynomial ℚ, 
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → is_root p (n + real.sqrt (n + 1)) ∧ is_root p (n - real.sqrt (n + 1))) ∧ 
    p.degree = 191 := 
sorry

end minimal_polynomial_degree_l123_123686


namespace ratio_paid_back_to_initial_debt_l123_123039

def initial_debt : ℕ := 40
def still_owed : ℕ := 30
def paid_back (initial_debt still_owed : ℕ) : ℕ := initial_debt - still_owed

theorem ratio_paid_back_to_initial_debt
  (initial_debt still_owed : ℕ) :
  (paid_back initial_debt still_owed : ℚ) / initial_debt = 1 / 4 :=
by 
  sorry

end ratio_paid_back_to_initial_debt_l123_123039


namespace probability_truth_or_lie_l123_123666

axiom probability_truth : ℝ := 0.30
axiom probability_lie : ℝ := 0.20
axiom probability_both : ℝ := 0.10

theorem probability_truth_or_lie :
  probability_truth + probability_lie - probability_both = 0.40 :=
by sorry

end probability_truth_or_lie_l123_123666


namespace ellipse_equation_incircle_area_max_l123_123009

open Real
open Function

noncomputable def foci_of_ellipse (F1 F2: ℝ × ℝ) (P Q: ℝ × ℝ) (PQ_length: ℝ) :=
F1 = (-1, 0) ∧ F2 = (1, 0) ∧ PQ_length = 3 ∧
∃ (a b: ℝ), a > b ∧ b > 0 ∧
  let eqn := (x, y): ∀(x y: ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 in
  let PQ_conditions := (y_1: ℝ), (x, 0): ∀(x y: ℝ), (y = 0 ∧ x = m*y + 1 ∧ (x, y) ∈ eqn.1) in
  (a^2 - b^2 = 1 ∧ 2 * b^2 / a = 3 ∧ a = 2 ∧ b = sqrt(3)) ∧ eqn = (x, y): (x^2 / 4) + (y^2 / 3) = 1

noncomputable def incircle_max_area (F1 F2 M N: ℝ × ℝ) :=
F1 = (-1, 0) ∧ F2 = (1, 0) ∧
∃ (R: ℝ), ∃ (line_equation: ℝ → ℝ),
∃ (area_of_incircle: ℝ),
  line_equation = λ (y: ℝ), y + 1 ∧
  area_of_incircle = (9 / 16) * π ∧
  let line_eq := line_equation.1 ((y_1, y_2): ℝ), (y_1 > 0 ∧ y_2 < 0) ∧ 
  let slope_check := line_eq != 0 in
line_eq = ((y_1: ℝ), (y_2: ℝ)), slope_check = true

theorem ellipse_equation :
  ∃ (F1 F2: ℝ × ℝ) (P Q: ℝ × ℝ) (PQ_length: ℝ),
  foci_of_ellipse F1 F2 P Q PQ_length :=
begin
  sorry
end

theorem incircle_area_max : 
  ∃ (F1 F2 M N: ℝ × ℝ),
  incircle_max_area F1 F2 M N :=
begin
  sorry
end

end ellipse_equation_incircle_area_max_l123_123009


namespace x_cubed_mod_25_l123_123894

noncomputable def remainder_of_x_cubed_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [MOD 25]) (h2 : 4 * x ≡ 20 [MOD 25]) : ℤ :=
  x^3 % 25

theorem x_cubed_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [MOD 25]) (h2 : 4 * x ≡ 20 [MOD 25]) : remainder_of_x_cubed_mod_25 x h1 h2 = 8 :=
sorry

end x_cubed_mod_25_l123_123894


namespace probability_at_least_two_women_is_correct_l123_123897

def group := {men := 9, women := 6}
def total_people := group.men + group.women
def selected_people := 4

noncomputable def probability_at_least_two_women : ℚ :=
  1 - (nat.choose group.men selected_people + group.women * nat.choose group.men (selected_people - 1)) / 
  nat.choose total_people selected_people

theorem probability_at_least_two_women_is_correct :
  probability_at_least_two_women = 7 / 13 := 
  sorry

end probability_at_least_two_women_is_correct_l123_123897


namespace trig_expression_value_l123_123826

theorem trig_expression_value (α : ℝ) (h : Real.tan (Real.pi + α) = 2) : 
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / (Real.sin (Real.pi + α) - Real.cos (Real.pi - α)) = 3 := 
by
  sorry

end trig_expression_value_l123_123826


namespace volleyball_tournament_l123_123072

noncomputable theory

variable (n : ℕ)
-- Conditions
def teams_played (n : ℕ) : Prop := n > 1 ∧ ∃ points: Finset ℕ, points = finset.range n ∧
                  ∀ k : ℕ, k < n → k ∈ points

-- Theorem Statement
theorem volleyball_tournament (n : ℕ) (h : teams_played n) :
  ∃ (penultimate_points : ℕ), penultimate_points = n - 2 ∧ 
  (∃ winner_points : ℕ, winner_points = n - 1 ∧ ∀ team : ℕ, team ≠ winner_points → team ≠ n - 2 → points team ≠ n - 1) :=
begin
  sorry
end

end volleyball_tournament_l123_123072


namespace f_neg_l123_123417

variable (f : ℝ → ℝ)

-- Given condition that f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- The form of f for x ≥ 0
def f_pos (x : ℝ) (h : 0 ≤ x) : f x = -x^2 + 2 * x := sorry

-- Objective to prove f(x) for x < 0
theorem f_neg {x : ℝ} (h : x < 0) (hf_odd : odd_function f) (hf_pos : ∀ x, 0 ≤ x → f x = -x^2 + 2 * x) : f x = x^2 + 2 * x := 
by 
  sorry

end f_neg_l123_123417


namespace triple_supplementary_angle_l123_123275

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l123_123275


namespace sum_of_odd_integers_15_to_45_l123_123633

theorem sum_of_odd_integers_15_to_45 : (∑ i in finset.range (16), (15 + 2 * i)) = 480 := 
sorry

end sum_of_odd_integers_15_to_45_l123_123633


namespace binom_divisible_by_prime_l123_123569

theorem binom_divisible_by_prime {p k : ℕ} (hp : Nat.Prime p) (hk1 : 1 ≤ k) (hk2 : k ≤ p - 1) : Nat.choose p k % p = 0 := 
  sorry

end binom_divisible_by_prime_l123_123569


namespace line_BB_l123_123494

variables {P : Type} [Plane P] -- Replace with the required plane geometry type if available

-- Given conditions
variables (A B C A₁ C₁ A' C' B' O : P)
variable (Ω : Circle P) -- Circumcircle
variable [circumcircle Ω A B C] -- Triangle ABC has circumcircle Ω

variables [is_acute_angled_triangle A B C]
variables [is_altitude A A₁ B C]
variables [is_altitude C C₁ A B]
variables [is_on A' (intersections (line_through A₁ C₁) Ω)]
variables [is_on C' (intersections (line_through A₁ C₁) Ω)]
variables [tangent_to Ω A']
variables [tangent_to Ω C']
variables [intersection_point (tangents_to Ω A' C') = B']

-- Statement to prove
theorem line_BB'_passes_through_center_of_circumcircle :
  line_through B B' = line_through O :=
sorry

end line_BB_l123_123494


namespace problem_I2_3_problem_I2_4_l123_123893

-- Problem I2.3: Given X and conditions, prove the units digit of X, R, is 3.
theorem problem_I2_3 (Q : ℕ) (X : ℕ) (R : ℕ) (h1 : X = Nat.sqrt ((100 * 102 * 103 * 105) + (Q - 3)))
(h2: R = X % 10) : R = 3 := sorry

-- Problem I2_4: Given R, find the sum of the last three digits of 2012^3 and prove S is 17.
theorem problem_I2_4 (R : ℕ) (S : ℕ) (h1 : ∀ x : ℝ, Nat.floor ((ℕ.log 2012 x) * x ^ (ℕ.log 2012 x)) = Nat.floor (x ^ R))
(h2 : S = 7 + 2 + 8) : S = 17 := sorry

end problem_I2_3_problem_I2_4_l123_123893


namespace golden_apples_first_six_months_l123_123703

-- Use appropriate namespaces
namespace ApolloProblem

-- Define the given conditions
def total_cost : ℕ := 54
def months_in_half_year : ℕ := 6

-- Prove that the number of golden apples charged for the first six months is 18
theorem golden_apples_first_six_months (X : ℕ) 
  (h1 : 6 * X + 6 * (2 * X) = total_cost) : 
  6 * X = 18 := 
sorry

end ApolloProblem

end golden_apples_first_six_months_l123_123703


namespace sum_of_pqrstu_l123_123530

theorem sum_of_pqrstu (p q r s t : ℤ) (h : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = -72) 
  (hpqrs : p ≠ q) (hnpr : p ≠ r) (hnps : p ≠ s) (hnpt : p ≠ t) (hnqr : q ≠ r) 
  (hnqs : q ≠ s) (hnqt : q ≠ t) (hnrs : r ≠ s) (hnrt : r ≠ t) (hnst : s ≠ t) : 
  p + q + r + s + t = 25 := 
by
  sorry

end sum_of_pqrstu_l123_123530


namespace number_of_archers_in_golden_armor_l123_123971

-- Define the problem context
structure Soldier where
  is_archer : Bool
  wears_golden_armor : Bool

def truth_teller (s : Soldier) (is_black_armor : Bool) : Bool :=
  if s.is_archer then s.wears_golden_armor = is_black_armor
  else s.wears_golden_armor ≠ is_black_armor

def response (s : Soldier) (question : String) (is_black_armor : Bool) : Bool :=
  match question with
  | "Are you wearing golden armor?" => if truth_teller s is_black_armor then s.wears_golden_armor else ¬s.wears_golden_armor
  | "Are you an archer?" => if truth_teller s is_black_armor then s.is_archer else ¬s.is_archer
  | "Is today Monday?" => if truth_teller s is_black_armor then True else False -- An assumption that today not being Monday means False
  | _ => False

-- Problem condition setup
def total_soldiers : Nat := 55
def golden_armor_yes : Nat := 44
def archer_yes : Nat := 33
def monday_yes : Nat := 22

-- Define the main theorem
theorem number_of_archers_in_golden_armor :
  ∃ l : List Soldier,
    l.length = total_soldiers ∧
    l.countp (λ s => response s "Are you wearing golden armor?" True) = golden_armor_yes ∧
    l.countp (λ s => response s "Are you an archer?" True) = archer_yes ∧
    l.countp (λ s => response s "Is today Monday?" True) = monday_yes ∧
    l.countp (λ s => s.is_archer ∧ s.wears_golden_armor) = 22 :=
sorry

end number_of_archers_in_golden_armor_l123_123971


namespace find_k_l123_123076

noncomputable def curve_C (x y : ℝ) : Prop :=
  x^2 + (y^2 / 4) = 1

noncomputable def line_eq (k x y : ℝ) : Prop :=
  y = k * x + 1

theorem find_k (k : ℝ) :
  (∃ A B : ℝ × ℝ, (curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧ line_eq k A.1 A.2 ∧ line_eq k B.1 B.2 ∧ 
   (A.1 * B.1 + A.2 * B.2 = 0))) ↔ (k = 1/2 ∨ k = -1/2) :=
sorry

end find_k_l123_123076


namespace find_A_l123_123293

theorem find_A (A B : ℕ) (h1: 3 + 6 * (100 + 10 * A + B) = 691) (h2 : 100 ≤ 6 * (100 + 10 * A + B) ∧ 6 * (100 + 10 * A + B) < 1000) : 
A = 8 :=
sorry

end find_A_l123_123293


namespace maximum_value_of_f_in_interval_l123_123593

noncomputable def f (x : ℝ) := (Real.sin x)^2 + (Real.sqrt 3) * Real.cos x - (3 / 4)

theorem maximum_value_of_f_in_interval : 
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 1 := 
  sorry

end maximum_value_of_f_in_interval_l123_123593


namespace fraction_shaded_l123_123598

theorem fraction_shaded :
  ∃ (P Q R S : Type) (six_equal_squares : P × Q × R × S) (shading_pattern : (six_equal_squares → Prop)),
  (shading_pattern five_out_of_twelve_squares) → 
  (frac_shaded PQRS = 5 / 12) :=
begin
  sorry
end

end fraction_shaded_l123_123598


namespace three_digit_integers_with_product_36_l123_123881

-- Definition of the problem conditions
def is_three_digit_integer (n : Nat) : Prop :=
  100 ≤ n ∧ n < 1000

def digit_product_is_36 (n : Nat) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  1 ≤ d1 ∧ d1 ≤ 9 ∧ 1 ≤ d2 ∧ d2 ≤ 9 ∧ 1 ≤ d3 ∧ d3 ≤ 9 ∧ (d1 * d2 * d3 = 36)

-- The statement of the proof
theorem three_digit_integers_with_product_36 :
  {n : Nat | is_three_digit_integer n ∧ digit_product_is_36 n}.toFinset.card = 21 := 
by
  sorry

end three_digit_integers_with_product_36_l123_123881


namespace find_a_l123_123474

theorem find_a (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) (a : ℝ) (h1 : ∀ n, S_n n = 3^(n+1) + a)
  (h2 : ∀ n, a_n (n+1) = S_n (n+1) - S_n n)
  (h3 : ∀ n m k, a_n m * a_n k = (a_n n)^2 → n = m + k) : 
  a = -3 := 
sorry

end find_a_l123_123474


namespace crayons_difference_l123_123157

theorem crayons_difference (crayons_given_away : ℤ) (crayons_lost : ℤ) (h1 : crayons_given_away = 571) (h2 : crayons_lost = 161) : 
  crayons_given_away - crayons_lost = 410 := 
by
  rw [h1, h2]
  norm_num

end crayons_difference_l123_123157


namespace uranus_appearance_minutes_after_6AM_l123_123612

-- Definitions of the given times and intervals
def mars_last_seen : Int := 0 -- 12:10 AM in minutes after midnight
def jupiter_after_mars : Int := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter : Int := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def reference_time : Int := 6 * 60 -- 6:00 AM in minutes after midnight

-- Statement of the problem
theorem uranus_appearance_minutes_after_6AM :
  let jupiter_first_appearance := mars_last_seen + jupiter_after_mars
  let uranus_first_appearance := jupiter_first_appearance + uranus_after_jupiter
  (uranus_first_appearance - reference_time) = 7 := by
  sorry

end uranus_appearance_minutes_after_6AM_l123_123612


namespace number_of_tables_l123_123481

theorem number_of_tables (c t : ℕ) (h1 : c = 8 * t) (h2 : 4 * c + 3 * t = 759) : t = 22 := by
  sorry

end number_of_tables_l123_123481


namespace kostyas_table_prime_l123_123516

theorem kostyas_table_prime (n : ℕ) (h₁ : n > 3) 
    (h₂ : ¬ ∃ r s : ℕ, r ≥ 3 ∧ s ≥ 3 ∧ n = r * s - (r + s)) : 
    Prime (n + 1) := 
sorry

end kostyas_table_prime_l123_123516


namespace find_number_that_satisfies_condition_l123_123702

theorem find_number_that_satisfies_condition : ∃ x : ℝ, x / 3 + 12 = 20 ∧ x = 24 :=
by
  sorry

end find_number_that_satisfies_condition_l123_123702


namespace Nina_can_buy_8_widgets_at_reduced_cost_l123_123556

def money_Nina_has : ℕ := 48
def widgets_she_can_buy_initially : ℕ := 6
def reduction_per_widget : ℕ := 2

theorem Nina_can_buy_8_widgets_at_reduced_cost :
  let initial_cost_per_widget := money_Nina_has / widgets_she_can_buy_initially
  let reduced_cost_per_widget := initial_cost_per_widget - reduction_per_widget
  money_Nina_has / reduced_cost_per_widget = 8 :=
by
  sorry

end Nina_can_buy_8_widgets_at_reduced_cost_l123_123556


namespace product_positions_8_2_100_100_l123_123501

def num_at_position : ℕ → ℕ → ℤ
| 0, _ => 0
| n, k => 
  let remainder := k % 3
  if remainder = 1 then 1 
  else if remainder = 2 then 2
  else -3

theorem product_positions_8_2_100_100 : 
  num_at_position 8 2 * num_at_position 100 100 = -3 :=
by
  unfold num_at_position
  -- unfold necessary definition steps
  sorry

end product_positions_8_2_100_100_l123_123501


namespace pauline_total_cost_l123_123564

def taco_shells_cost : ℝ := 5
def bell_peppers_cost_each : ℝ := 1.5
def bell_peppers_quantity : ℕ := 4
def meat_cost_per_pound : ℝ := 3
def meat_quantity : ℕ := 2
def tomatoes_cost_each : ℝ := 0.75
def tomatoes_quantity : ℕ := 3
def shredded_cheese_cost : ℝ := 4
def tortillas_cost : ℝ := 2.5
def salsa_cost : ℝ := 3.25

def bell_peppers_total_cost : ℝ := bell_peppers_cost_each * bell_peppers_quantity
def meat_total_cost : ℝ := meat_cost_per_pound * meat_quantity
def tomatoes_total_cost : ℝ := tomatoes_cost_each * tomatoes_quantity

def total_cost : ℝ :=
  taco_shells_cost + bell_peppers_total_cost + meat_total_cost +
  tomatoes_total_cost + shredded_cheese_cost + tortillas_cost + salsa_cost

theorem pauline_total_cost : total_cost = 29 := by
  unfold total_cost bell_peppers_total_cost meat_total_cost tomatoes_total_cost
  -- Performing calculations to verify the total cost
  have h : 5 + 6 + 6 + 2.25 + 4 + 2.5 + 3.25 = 29 := sorry -- Verifying the arithmetic manually.
  rw [h]
  rfl

end pauline_total_cost_l123_123564


namespace first_term_formula_correct_l123_123316

theorem first_term_formula_correct
  (S n d a : ℝ) 
  (h_sum_formula : S = (n / 2) * (2 * a + (n - 1) * d)) :
  a = (S / n) + (n - 1) * (d / 2) := 
sorry

end first_term_formula_correct_l123_123316


namespace PQ_eq_QR_iff_bisectors_meet_on_AC_l123_123114

open EuclideanGeometry

variables {A B C D P Q R : Point}

-- Conditions: convex quadrilateral ABCD, P, Q, and R are feet of the perpendiculars from D to BC, CA, and AB
def isConvexQuadrilateral (A B C D : Point) : Prop := ConvexQuadrilateral A B C D
def isFootOfPerpendicular (P D : Point) (l : Line) : Prop := FootOfPerpendicular D l P

-- Question: Prove PQ = QR if and only if the bisectors of angles ABC and ADC meet on segment AC
theorem PQ_eq_QR_iff_bisectors_meet_on_AC
  (h1 : isConvexQuadrilateral A B C D)
  (h2 : isFootOfPerpendicular P D (Line.fromPoints B C))
  (h3 : isFootOfPerpendicular Q D (Line.fromPoints C A))
  (h4 : isFootOfPerpendicular R D (Line.fromPoints A B)) :
  Segment.length (Segment.mk P Q) = Segment.length (Segment.mk Q R) ↔
  ∃ E : Point, IsSegment AC E ∧ IsAngleBisector (Line.fromPoints A C) (Angle.mk A B C) E ∧ IsAngleBisector (Line.fromPoints A C) (Angle.mk A D C) E :=
sorry

end PQ_eq_QR_iff_bisectors_meet_on_AC_l123_123114


namespace sum_div_9_remainder_l123_123285

theorem sum_div_9_remainder :
  ∑ i in Finset.range 21, i % 9 = 4 :=
  sorry

end sum_div_9_remainder_l123_123285


namespace ratio_of_age_differences_l123_123176

variable (R J K : ℕ)

-- conditions
axiom h1 : R = J + 6
axiom h2 : R + 2 = 2 * (J + 2)
axiom h3 : (R + 2) * (K + 2) = 108

-- statement to prove
theorem ratio_of_age_differences : (R - J) = 2 * (R - K) := 
sorry

end ratio_of_age_differences_l123_123176


namespace part1_part2_part3_l123_123548

variable (a b c : ℝ)

-- Define the function f
def f (x : ℝ) : ℝ := (1 / 3) * x^3 - (a / 2) * x^2 + b * x + c

-- 1. Prove that b = 0 and c = 1 given f(x) and the conditions
theorem part1 (h1 : f 0 = 1) (h2 : f' 0 = 0) : b = 0 ∧ c = 1 :=
by
  sorry

-- Define the function f'
def f' (x : ℝ) : ℝ := x^2 - a * x + b

-- 2. Prove the monotonicity intervals of f given a > 0
theorem part2 (ha : 0 < a) : 
  ((∀ x, x < 0 → f' x > 0) ∧ 
   (∀ x, 0 < x ∧ x < a → f' x < 0) ∧ 
   (∀ x, a < x → f' x > 0)) :=
by
  sorry

-- Define the function g and its derivative g'
def g (x : ℝ) : ℝ := f x + 2 * x
def g' (x : ℝ) : ℝ := f' x + 2

-- 3. Prove the range of a given that g has a decreasing interval in (-2, -1)
theorem part3 (h3 : ∃ x : ℝ, -2 < x ∧ x < -1 ∧ g' x ≤ 0) : a ≤ -2 * Real.sqrt 2 :=
by
  sorry

end part1_part2_part3_l123_123548


namespace square_in_rectangle_percentage_l123_123344

theorem square_in_rectangle_percentage (s : ℝ) :
  let w := 3 * s 
  let l := 4.5 * s 
  let A_square := s^2 
  let A_rectangle := l * w 
  (A_square / A_rectangle) * 100 = 7.41 :=
by
  let w := 3 * s
  let l := 4.5 * s
  let A_square := s^2
  let A_rectangle := 4.5 * s * 3 * s
  have h1 : A_square = s^2 := rfl
  have h2 : A_rectangle = 13.5 * s^2 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end square_in_rectangle_percentage_l123_123344


namespace parabola_problem_l123_123526

theorem parabola_problem :
  let P := λ x : ℝ, 4 * x^2,
      V1 := (0 : ℝ, 0 : ℝ),
      A := (a : ℝ, P a),
      B := (b : ℝ, P b),
      M := (a + b) / 2,
      midpoint_coords := (M, 2 * ((a + b)^2 + 1 / 32)),
      Q := λ x : ℝ, 8 * x^2 + 1 / 16,
      V2 := (0 : ℝ, 1 / 16),
      F2 := (0 : ℝ, 17 / 256),
      F1 := (0 : ℝ, 1 / 4),
      distance_y := λ (p1 p2 : ℝ) , abs ((p1 : ℝ) - (p2 : ℝ)),
      ratio := distance_y F1 F2 / distance_y 0 (1 / 16)
  in ∃ a b : ℝ, a * b = -1 / 64 ∧ 
       (∀ x : ℝ, Q x = midpoint_coords) ∧ 
       V1 = (0, 0) ∧ 
       V2 = (0, 1 / 16) ∧ 
       F2 = (0, 17 / 256) ∧ 
       ratio = 1 / 16 := sorry

end parabola_problem_l123_123526


namespace sequence_sum_2017_l123_123027

def sequence (n : ℕ) : ℤ :=
  n * Int.cos (n * Real.pi / 2)

def sum_sequence (n : ℕ) : ℤ :=
  ∑ i in Finset.range (n + 1), sequence i

theorem sequence_sum_2017 :
  sum_sequence 2017 = 1008 := sorry

end sequence_sum_2017_l123_123027


namespace archers_in_golden_l123_123937

variables (soldiers archers swordsmen wearing_golden wearing_black : ℕ)
variables (truth_swordsmen_black lie_swordsmen_golden lie_archers_black truth_archers_golden : ℕ)

-- Given conditions
variables (cond1 : archers + swordsmen = 55)
variables (cond2 : wearing_golden + wearing_black = 55)
variables (cond3 : truth_swordsmen_black + lie_swordsmen_golden + lie_archers_black + truth_archers_golden = 55)
variables (cond4 : wearing_golden = 44)
variables (cond5 : archers = 33)
variables (cond6 : truth_swordsmen_black + lie_archers_black = 22)

-- Define the mathematic equivalent proof problem
theorem archers_in_golden : archers = 22 :=
by
  sorry

end archers_in_golden_l123_123937


namespace coeff_x17_x18_l123_123736

theorem coeff_x17_x18 :
  let polynomial := (1 + X^5 + X^7)^20 in
  (coeff polynomial 17 = 3420) ∧ (coeff polynomial 18 = 0) :=
by {
  sorry
}

end coeff_x17_x18_l123_123736


namespace num_pairs_solution_l123_123205

theorem num_pairs_solution : 
  (∃ (m n : ℕ), (m > 0 ∧ n > 0 ∧ 4 / m + 2 / n = 1) ∧ 
  (m, n) = (5, 10) ∨ (m, n) = (6, 6) ∨ (m, n) = (8, 4) ∨ (m, n) = (12, 3)).pairwise_iff.count = 4 :=
sorry

end num_pairs_solution_l123_123205


namespace point_of_intersection_of_asymptotes_l123_123749

noncomputable def asymptotes_intersection_point : ℝ × ℝ := (3, 1)

theorem point_of_intersection_of_asymptotes (x y : ℝ) (h : y = (x ^ 2 - 6 * x + 8) / (x ^ 2 - 6 * x + 9)) (hx : x = 3) (hy : y = 1) :
  (x, y) = (3, 1) :=
by
  rw [hx, hy]
  simp
  exact asymptotes_intersection_point
  sorry -- Proof steps are not provided, only statement is required.

end point_of_intersection_of_asymptotes_l123_123749


namespace shortest_altitude_l123_123214

/-!
  Prove that the shortest altitude of a right triangle with sides 9, 12, and 15 is 7.2.
-/

theorem shortest_altitude (a b c : ℕ) (h : a^2 + b^2 = c^2) (ha : a = 9) (hb : b = 12) (hc : c = 15) :
  7.2 ≤ a ∧ 7.2 ≤ b ∧ 7.2 ≤ (2 * (a * b) / c) := 
sorry

end shortest_altitude_l123_123214


namespace exists_epsilon_inequality_l123_123538

theorem exists_epsilon_inequality (n : ℕ) (h : n ≥ 2) (a : Fin n → ℝ) :
  ∃ (ε : Fin n → ℤ), (∀ i, ε i = 1 ∨ ε i = -1) ∧
  (let sum_a := ∑ i, a i in
   let sum_ea := ∑ i, (ε i) * a i in
   sum_a^2 + sum_ea^2 ≤ (n + 1) * ∑ i, (a i)^2) :=
sorry

end exists_epsilon_inequality_l123_123538


namespace mean_transformation_l123_123408

variable {x1 x2 x3 : ℝ}
variable (s : ℝ)
variable (h_var : s^2 = (1 / 3) * (x1^2 + x2^2 + x3^2 - 12))

theorem mean_transformation :
  (x1 + 1 + x2 + 1 + x3 + 1) / 3 = 3 :=
by
  sorry

end mean_transformation_l123_123408


namespace junghyeon_stickers_l123_123192

def total_stickers : ℕ := 25
def junghyeon_sticker_count (yejin_stickers : ℕ) : ℕ := 2 * yejin_stickers + 1

theorem junghyeon_stickers (yejin_stickers : ℕ) (h : yejin_stickers + junghyeon_sticker_count yejin_stickers = total_stickers) : 
  junghyeon_sticker_count yejin_stickers = 17 :=
  by
  sorry

end junghyeon_stickers_l123_123192


namespace distance_planes_l123_123378

-- Define the planes with their given equations
def Plane1 : (ℝ × ℝ × ℝ) → Prop := λ (p : ℝ × ℝ × ℝ), 2 * p.1 - 4 * p.2 + 7 * p.3 = 10
def Plane2 : (ℝ × ℝ × ℝ) → Prop := λ (p : ℝ × ℝ × ℝ), 4 * p.1 - 8 * p.2 + 14 * p.3 = 22

-- Define the distance function and calculate the distance between the planes
noncomputable def distance_between_planes : ℝ :=
  let n := (2:ℝ, -4, 7) in
  let d1 := (10:ℝ) in
  let d2 := 22 / 2 in -- Because the plane's equation simplifies to 2x - 4y + 7z = 11
  let numerator := abs (d2 - d1) in
  let denominator := real.sqrt ((2^2) + ((-4)^2) + (7^2)) in
  numerator / denominator

-- Prove the distance between Plane1 and Plane2 is 1 / sqrt(69)
theorem distance_planes : distance_between_planes = 1 / real.sqrt 69 := by
  sorry

end distance_planes_l123_123378


namespace sum_of_first_12_terms_l123_123496

theorem sum_of_first_12_terms (a : ℕ → ℕ) (d : ℤ) (h₁ : a 1 = 3) (h₂ : a 10 = 3 * a 3) :
  let S₁₂ := (12 / 2) * (2 * a 1 + (12 - 1) * d)
  in S₁₂ = 168 :=
sorry

end sum_of_first_12_terms_l123_123496


namespace field_area_restriction_l123_123750

theorem field_area_restriction (S : ℚ) (b : ℤ) (a : ℚ) (x y : ℚ) 
  (h1 : 10 * 300 * S ≤ 10000)
  (h2 : 2 * a = - b)
  (h3 : abs (6 * y) + 3 ≥ 3)
  (h4 : 2 * abs (2 * x) - abs b ≤ 9)
  (h5 : b ∈ [-4, -3, -2, -1, 0, 1, 2, 3, 4])
: S ≤ 10 / 3 := sorry

end field_area_restriction_l123_123750


namespace ways_to_reach_5_5_l123_123338

def moves_to_destination : ℕ → ℕ → ℕ
| 0, 0     => 1
| 0, j+1   => moves_to_destination 0 j
| i+1, 0   => moves_to_destination i 0
| i+1, j+1 => moves_to_destination i (j+1) + moves_to_destination (i+1) j + moves_to_destination i j

theorem ways_to_reach_5_5 : moves_to_destination 5 5 = 1573 := by
  sorry

end ways_to_reach_5_5_l123_123338


namespace find_bounded_area_l123_123991

theorem find_bounded_area (f : ℝ → ℝ) (hf_cont : continuous f) (hf_mono : monotone f)
  (hf_0 : f 0 = 0) (hf_1 : f 1 = 1) : 
  let A1 := 4 * ∫ u in 0..1, f u
  let A2 := ∫ x in 1..5, 4 * f x
  let A_triangle := (5 * 5) / 2
  A1 + A2 - A_triangle = 7.5 :=
sorry

end find_bounded_area_l123_123991


namespace quadratic_inequality_solution_empty_l123_123356

theorem quadratic_inequality_solution_empty :
    (∀ x : ℝ, -x^2 + x + 1 ≤ 0 ∨ 2 * x^2 - 3 * x + 4 < 0 ∨ x^2 + 3 * x + 10 > 0 ∨ -x^2 - 4 * x + 3 > 0) → 
    ¬ (∃ x : ℝ, 2 * x^2 - 3 * x + 4 < 0) :=
begin
  sorry
end

end quadratic_inequality_solution_empty_l123_123356


namespace intersection_on_incircle_l123_123087

variables {A B C : Type}
variables (AB: line_segment A B)
variables [right_angle (triangle A B C) at B]
variables (I: point → Type)
variables [incenter I (triangle A B C)]
variables (M: midpoint of AB)
variables [tangent_to (circumcircle (triangle A B C)) at C]
variables (S: point of line passing (tangent_to (circumcircle (triangle A B C)) at C) through (intersection_of (tangent from A to I) and (tangent from B to I)))
variables (H: orthocenter of (triangle A B C))
variables (P: point passing some intersections)

theorem intersection_on_incircle :
  intersects (tangent_to (circumcircle (triangle A B C)) at C) (line_through S M) (incircle (triangle A B C)) :=
sorry

end intersection_on_incircle_l123_123087


namespace solve_inequality_prove_conditional_l123_123436

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + abs (x - 2)

-- Solve the inequality f(x) ≤ 2|x|
theorem solve_inequality : {x : ℝ | f x ≤ 2 * abs x} = set.Icc 1 2 := sorry

-- Prove if f(x) ≥ a^2 + 4b^2 + 5c^2 - 1/4 for any x in ℝ, then ac + 4bc ≤ 1
theorem prove_conditional (a b c : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 + 4*b^2 + 5*c^2 - 1/4) → (a*c + 4*b*c ≤ 1) := sorry

end solve_inequality_prove_conditional_l123_123436


namespace A_eq_B_l123_123074

variables (α : Type) (Q : α → Prop)
variables (A B C : α → Prop)

-- Conditions
-- 1. For the questions where both B and C answered "yes", A also answered "yes".
axiom h1 : ∀ q, B q ∧ C q → A q
-- 2. For the questions where A answered "yes", B also answered "yes".
axiom h2 : ∀ q, A q → B q
-- 3. For the questions where B answered "yes", at least one of A and C answered "yes".
axiom h3 : ∀ q, B q → (A q ∨ C q)

-- Prove that A and B gave the same answer to all questions
theorem A_eq_B : ∀ q, A q ↔ B q :=
sorry

end A_eq_B_l123_123074


namespace product_of_positive_integral_values_of_n_l123_123391

theorem product_of_positive_integral_values_of_n :
  (∃ n : ℕ, 0 < n ∧ ∃ q : ℕ, Prime q ∧ n^2 - 41 * n + 420 = q) →
  ∏ {n : ℕ} (H : 0 < n ∧ ∃ q : ℕ, Prime q ∧ n^2 - 41 * n + 420 = q), n = 418 :=
begin
  sorry
end

end product_of_positive_integral_values_of_n_l123_123391


namespace smallest_positive_four_digit_multiple_of_18_l123_123766

-- Define the predicates for conditions
def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def multiple_of_18 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 18 * k

-- Define the main theorem
theorem smallest_positive_four_digit_multiple_of_18 : 
  ∃ n : ℕ, four_digit_number n ∧ multiple_of_18 n ∧ ∀ m : ℕ, four_digit_number m ∧ multiple_of_18 m → n ≤ m :=
begin
  use 1008,
  split,
  { -- proof that 1008 is a four-digit number
    split,
    { linarith, },
    { linarith, }
  },

  split,
  { -- proof that 1008 is a multiple of 18
    use 56,
    norm_num,
  },

  { -- proof that 1008 is the smallest such number
    intros m h1 h2,
    have h3 := Nat.le_of_lt,
    sorry, -- Detailed proof would go here
  }
end

end smallest_positive_four_digit_multiple_of_18_l123_123766


namespace systematic_sampling_l123_123914

axiom car (n : ℕ) : Prop

def is_car_selected (n : ℕ) : Prop :=
  n % 10 = 6

def sampling_method : Prop :=
  ∀ (n : ℕ), car n → n % 10 = 6

theorem systematic_sampling :
  sampling_method → ( ∀ (n : ℕ), is_car_selected n → car n ) → true := 
begin
  sorry
end

end systematic_sampling_l123_123914


namespace sum_of_a_equals_five_l123_123025

theorem sum_of_a_equals_five
  (f : ℕ → ℕ → ℕ)  -- Represents the function f defined by Table 1
  (a : ℕ → ℕ)  -- Represents the occurrences a₀, a₁, ..., a₄
  (h1 : a 0 + a 1 + a 2 + a 3 + a 4 = 5)  -- Condition 1
  (h2 : 0 * a 0 + 1 * a 1 + 2 * a 2 + 3 * a 3 + 4 * a 4 = 5)  -- Condition 2
  : a 0 + a 1 + a 2 + a 3 = 5 :=
sorry

end sum_of_a_equals_five_l123_123025


namespace unique_solution_l123_123388

theorem unique_solution (k n : ℕ) (hk : k > 0) (hn : n > 0) (h : (7^k - 3^n) ∣ (k^4 + n^2)) : (k = 2 ∧ n = 4) :=
by
  sorry

end unique_solution_l123_123388


namespace sum_primes_between_20_and_40_l123_123650

open Nat

def primesBetween20And40 : List Nat := [23, 29, 31, 37]

theorem sum_primes_between_20_and_40 :
  (primesBetween20And40.sum = 120) :=
by
  sorry

end sum_primes_between_20_and_40_l123_123650


namespace area_of_rectangle_is_432_l123_123591

/-- Define the width of the rectangle --/
def width : ℕ := 12

/-- Define the length of the rectangle, which is three times the width --/
def length : ℕ := 3 * width

/-- The area of the rectangle is length multiplied by width --/
def area : ℕ := length * width

/-- Proof problem: the area of the rectangle is 432 square meters --/
theorem area_of_rectangle_is_432 :
  area = 432 :=
sorry

end area_of_rectangle_is_432_l123_123591


namespace range_of_a_if_tangents_coincide_l123_123859

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then x^2 + x + a else -1 / x

theorem range_of_a_if_tangents_coincide :
  (∃ (x1 x2 : ℝ), x1 < 0 ∧ 0 < x2 ∧ f' x1 = f' x2) →
  ∃ (a : ℝ), -2 < a ∧ a < 1 / 4 :=
by
  sorry

end range_of_a_if_tangents_coincide_l123_123859


namespace average_score_of_male_students_l123_123100

theorem average_score_of_male_students
  (female_students : ℕ) (male_students : ℕ) (female_avg_score : ℕ) (class_avg_score : ℕ)
  (h_female_students : female_students = 20)
  (h_male_students : male_students = 30)
  (h_female_avg_score : female_avg_score = 75)
  (h_class_avg_score : class_avg_score = 72) :
  (30 * (((class_avg_score * (female_students + male_students)) - (female_avg_score * female_students)) / male_students) = 70) :=
by
  -- Sorry for the proof
  sorry

end average_score_of_male_students_l123_123100


namespace find_b_l123_123531

noncomputable def p (x : ℝ) : ℝ := 3 * x - 8
noncomputable def q (x : ℝ) (b : ℝ) : ℝ := 4 * x - b

theorem find_b (b : ℝ) : p (q 3 b) = 10 → b = 6 :=
by
  unfold p q
  intro h
  sorry

end find_b_l123_123531


namespace one_angle_greater_135_l123_123003

noncomputable def angles_sum_not_form_triangle (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : Prop :=
  ∀ (A B C : ℝ), 
   (A < a + b ∧ A < a + c ∧ A < b + c) →
  (B < a + b ∧ B < a + c ∧ B < b + c) →
  (C < a + b ∧ C < a + c ∧ C < b + c) →
  ∃ α β γ, α > 135 ∧ β < 60 ∧ γ < 60 ∧ α + β + γ = 180

theorem one_angle_greater_135 {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : angles_sum_not_form_triangle a b c ha hb hc) :
  ∃ α β γ, α > 135 ∧ α + β + γ = 180 :=
sorry

end one_angle_greater_135_l123_123003


namespace archers_in_golden_l123_123941

variables (soldiers archers swordsmen wearing_golden wearing_black : ℕ)
variables (truth_swordsmen_black lie_swordsmen_golden lie_archers_black truth_archers_golden : ℕ)

-- Given conditions
variables (cond1 : archers + swordsmen = 55)
variables (cond2 : wearing_golden + wearing_black = 55)
variables (cond3 : truth_swordsmen_black + lie_swordsmen_golden + lie_archers_black + truth_archers_golden = 55)
variables (cond4 : wearing_golden = 44)
variables (cond5 : archers = 33)
variables (cond6 : truth_swordsmen_black + lie_archers_black = 22)

-- Define the mathematic equivalent proof problem
theorem archers_in_golden : archers = 22 :=
by
  sorry

end archers_in_golden_l123_123941


namespace students_catching_up_on_homework_l123_123915

-- Definitions for the given conditions
def total_students := 120
def silent_reading_students := (2/5 : ℚ) * total_students
def board_games_students := (3/10 : ℚ) * total_students
def group_discussions_students := (1/8 : ℚ) * total_students
def other_activities_students := silent_reading_students + board_games_students + group_discussions_students
def catching_up_homework_students := total_students - other_activities_students

-- Statement of the proof problem
theorem students_catching_up_on_homework : catching_up_homework_students = 21 := by
  sorry

end students_catching_up_on_homework_l123_123915


namespace no_valid_coloring_l123_123990

open Nat

-- Define the color type
inductive Color
| blue
| red
| green

-- Define the coloring function
def color : ℕ → Color := sorry

-- Define the properties of the coloring function
def valid_coloring (color : ℕ → Color) : Prop :=
  ∀ (m n : ℕ), m > 1 → n > 1 → color m ≠ color n → 
    color (m * n) ≠ color m ∧ color (m * n) ≠ color n

-- Theorem: It is not possible to color all natural numbers greater than 1 as described
theorem no_valid_coloring : ¬ ∃ (color : ℕ → Color), valid_coloring color :=
by
  sorry

end no_valid_coloring_l123_123990


namespace sum_mod_1_to_20_l123_123281

theorem sum_mod_1_to_20 :
  (∑ i in finset.range 21, i) % 9 = 3 :=
by
  sorry

end sum_mod_1_to_20_l123_123281


namespace lateral_surface_area_of_cone_l123_123056

theorem lateral_surface_area_of_cone (r h : ℝ) (hr : r = sqrt 2) (hh : h = 2) :
    ∃ (A : ℝ), A = 2 * sqrt 3 * π := 
begin
    sorry,
end

end lateral_surface_area_of_cone_l123_123056


namespace gopi_annual_salary_l123_123037

-- Define the conditions
def servant_receives (annual_salary : ℝ) (turban_price : ℝ) (months_worked : ℝ) : ℝ :=
  (months_worked / 12) * annual_salary + turban_price

def correct_annual_salary (annual_salary : ℝ) : Prop :=
  servant_receives annual_salary 90 9 = 135

-- State the theorem we aim to prove
theorem gopi_annual_salary : correct_annual_salary 60 := by
  unfold correct_annual_salary
  unfold servant_receives
  sorry

end gopi_annual_salary_l123_123037


namespace min_dot_product_l123_123848

-- Definitions based on given conditions
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)
def P (x : ℝ) (hx : 0 < x) : ℝ × ℝ := (x, 9 / x)

-- Define the dot product between vectors OA and OP
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove that the minimum value of the dot product of OA and OP is 6
theorem min_dot_product : ∀ x (hx : 0 < x), dot_product (1,  1) (x, 9 / x) ≥ 6 :=
by {
  intros x hx,
  sorry -- proof omitted as per instructions
}

end min_dot_product_l123_123848


namespace expected_sides_rectangle_expected_sides_polygon_l123_123626

-- Part (a)
theorem expected_sides_rectangle (k : ℕ) (h : k > 0) : (4 + 4 * k) / (k + 1) → 4 :=
by sorry

-- Part (b)
theorem expected_sides_polygon (n k : ℕ) (h : n > 2) (h_k : k ≥ 0) : (n + 4 * k) / (k + 1) = (n + 4 * k) / (k + 1) :=
by sorry

end expected_sides_rectangle_expected_sides_polygon_l123_123626


namespace f_is_p_plus_2_l123_123576

def f (α p : ℝ) : ℝ :=
(p * (Real.cos α)^3 - (Real.cos (3 * α))) / (Real.cos α) + 
(p * (Real.sin α)^3 + (Real.sin (3 * α))) / (Real.sin α)

theorem f_is_p_plus_2 (α p : ℝ) (h : p = p) : f α p = p + 2 :=
sorry

end f_is_p_plus_2_l123_123576


namespace four_digit_palindromes_odd_digit_palindromes_l123_123493

noncomputable def count_palindromes (digits : ℕ) : ℕ :=
  if digits = 1 then 9
  else if digits % 2 = 0 then count_palindromes (digits - 1)
  else 9 * 10 ^ (digits / 2)

theorem four_digit_palindromes : count_palindromes 4 = 90 :=
by sorry

theorem odd_digit_palindromes (n : ℕ) (hn : n > 0) : count_palindromes (2 * n + 1) = 9 * 10^n :=
by sorry

end four_digit_palindromes_odd_digit_palindromes_l123_123493


namespace graduation_photo_l123_123227

theorem graduation_photo (students : Finset (fin 5))
  (A B C : Fin 5)
  (h_distinct : ∀ i j : fin 5, i ∈ students → j ∈ students → i ≠ j → i ≠ j)
  (h_A_not_next_to_B : ∀ permutation : List (fin 5), permutation.perm_of_list ((A, B), (B, A)))
  (h_BC_together : ∀ permutation : List (fin 5), permutation.perm_of_list ((B, C), (C, B))) :
  (number_of_arrangements students A B C) = 36 :=
sorry

end graduation_photo_l123_123227


namespace sum_of_primes_between_20_and_40_l123_123646

theorem sum_of_primes_between_20_and_40 : 
  (23 + 29 + 31 + 37) = 120 := 
by
  -- Proof goes here
sorry

end sum_of_primes_between_20_and_40_l123_123646


namespace sandy_gave_puppies_l123_123178

theorem sandy_gave_puppies 
  (original_puppies : ℕ) 
  (puppies_with_spots : ℕ) 
  (puppies_left : ℕ) 
  (h1 : original_puppies = 8) 
  (h2 : puppies_with_spots = 3) 
  (h3 : puppies_left = 4) : 
  original_puppies - puppies_left = 4 := 
by {
  -- This is a placeholder for the proof.
  sorry
}

end sandy_gave_puppies_l123_123178


namespace probability_no_obtuse_triangle_is_9_over_64_l123_123812

noncomputable def probability_no_obtuse_triangle (A0 A1 A2 A3 : ℝ) : ℝ :=
  -- Define the probabilistic model according to the problem conditions
  -- Assuming A0, A1, A2, and A3 are positions of points on the circle parametrized by angles in radians
  let θ := real.angle.lower_half (A1 - A0) in
  let prob_A2 := (π - θ) / (2 * π) in
  let prob_A3 := (π - θ) / (2 * π) in
  (1 / 2) * (prob_A2 * prob_A3)

theorem probability_no_obtuse_triangle_is_9_over_64 :
  probability_no_obtuse_triangle A0 A1 A2 A3 = 9 / 64 :=
by sorry

end probability_no_obtuse_triangle_is_9_over_64_l123_123812


namespace at_least_one_black_ball_drawn_l123_123925

theorem at_least_one_black_ball_drawn 
  (num_black_balls : ℕ) 
  (num_white_balls : ℕ) 
  (total_balls_drawn : ℕ) 
  (num_black_balls = 4) 
  (num_white_balls = 2) 
  (total_balls_drawn = 3) :
  ∃ (k : ℕ), k ≥ 1 ∧ k ≤ total_balls_drawn :=
sorry

end at_least_one_black_ball_drawn_l123_123925


namespace angle_triple_supplement_l123_123256

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123256


namespace convex_polygon_diagonal_length_l123_123886

noncomputable def sameLengthDiagonalsConvexPolygon (n : ℕ) : Prop :=
  ∀ (P : ConvexPolygon n), (∀ (d₁ d₂ : Diagonal P), length d₁ = length d₂)

theorem convex_polygon_diagonal_length (n : ℕ) :
  (ConvexPolygon n → sameLengthDiagonalsConvexPolygon n) → (n = 4 ∨ n = 5) :=
by
  sorry

end convex_polygon_diagonal_length_l123_123886


namespace quilt_patch_cost_l123_123508

-- conditions
def quilt_length : ℕ := 16
def quilt_width : ℕ := 20
def patch_area : ℕ := 4
def first_ten_patch_cost : ℕ := 10
def subsequent_patch_cost : ℕ := 5

theorem quilt_patch_cost :
  let total_quilt_area := quilt_length * quilt_width,
      total_patches := total_quilt_area / patch_area,
      cost_first_ten := 10 * first_ten_patch_cost,
      remaining_patches := total_patches - 10,
      cost_remaining := remaining_patches * subsequent_patch_cost,
      total_cost := cost_first_ten + cost_remaining
  in total_cost = 450 :=
by
  sorry

end quilt_patch_cost_l123_123508


namespace contradiction_example_l123_123162

theorem contradiction_example (x : ℝ) (h : x^2 - 1 = 0) : x = -1 ∨ x = 1 :=
by
  sorry

end contradiction_example_l123_123162


namespace angle_B_value_l123_123352

-- Define the triangle ABC and points X and Y with the given conditions.
variable {A B C X Y : Type}
variable [triangle : Triangle A B C]
variable (AX_bisects_A : Bisects (angle A) (segment AX) B C)
variable (BY_bisects_B : Bisects (angle B) (segment BY) C A)
variable (angle_A : angle A = 60)
variable (AB_BX_AY_YB_eq : segment AB + segment BX = segment AY + segment YB)

-- State the theorem
theorem angle_B_value : ∀ (A B C : Type) (X : on BC) (Y : on CA),
  (Bisects (angle A) (segment AX) B C) →
  (Bisects (angle B) (segment BY) C A) →
  (angle A = 60) →
  (segment AB + segment BX = segment AY + segment YB) →
  angle B = 80 :=
by
  sorry

end angle_B_value_l123_123352


namespace sum_of_A_l123_123922

def A (i j : ℕ) : ℕ :=
  if h : 1 ≤ i ∧ i ≤ 10 ∧ 1 ≤ j ∧ j ≤ 10 then
    if j = 10 then 90 + i else 0
  else 0

theorem sum_of_A : (∑ i in Finset.range 10, A (i + 1) 10) = 955 := 
by {
  sorry
}

end sum_of_A_l123_123922


namespace smallest_number_of_pluses_l123_123383

-- Conditions from the problem statement
def equation_constant : List Int := [2, 0, 1, 5, 2, 0, 1, 5, 2, 0, 1, 5]

/-- Each asterisk in the equation should be replaced with either + or - -/
def is_correct_replacement (lst : List Int) : Bool :=
  lst.sum = 0 

-- Our main theorem
theorem smallest_number_of_pluses (k : Nat) (h : k = 2) :
  (∃ (replacements : List Int), 
    is_correct_replacement replacements ∧ 
    replacements.length = equation_constant.length ∧ 
    (replacements.count (λ x => x > 0)) = k) := 
by
  sorry

end smallest_number_of_pluses_l123_123383


namespace angle_triple_supplement_l123_123258

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123258


namespace f_is_odd_f_is_decreasing_range_of_x_l123_123547

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn (x y : ℝ) : f(x + y) = f(x) + f(y)
axiom f_neg_when_positive (x : ℝ) (h : 0 < x) : f(x) < 0
axiom f_at_one : f(1) = -2

-- Question 1: Prove that f is odd
theorem f_is_odd (x : ℝ) : f(-x) = -f(x) :=
sorry

-- Question 2: Prove that f is decreasing on ℝ
theorem f_is_decreasing (x₁ x₂ : ℝ) (h : x₁ < x₂) : f(x₁) > f(x₂) :=
sorry

-- Question 3: Find the range of x for which 2 ≤ |f(x)| ≤ 6 holds
theorem range_of_x (x : ℝ) : 2 ≤ |f(x)| ∧ |f(x)| ≤ 6 ↔ (-3 ≤ x ∧ x ≤ -1) ∨ (1 ≤ x ∧ x ≤ 3) :=
sorry

end f_is_odd_f_is_decreasing_range_of_x_l123_123547


namespace archers_in_golden_l123_123934

variables (soldiers archers swordsmen wearing_golden wearing_black : ℕ)
variables (truth_swordsmen_black lie_swordsmen_golden lie_archers_black truth_archers_golden : ℕ)

-- Given conditions
variables (cond1 : archers + swordsmen = 55)
variables (cond2 : wearing_golden + wearing_black = 55)
variables (cond3 : truth_swordsmen_black + lie_swordsmen_golden + lie_archers_black + truth_archers_golden = 55)
variables (cond4 : wearing_golden = 44)
variables (cond5 : archers = 33)
variables (cond6 : truth_swordsmen_black + lie_archers_black = 22)

-- Define the mathematic equivalent proof problem
theorem archers_in_golden : archers = 22 :=
by
  sorry

end archers_in_golden_l123_123934


namespace train_speed_l123_123304

/-
Problem Statement:
Prove that the speed of a train is 26.67 meters per second given:
  1. The length of the train is 320 meters.
  2. The time taken to cross the telegraph post is 12 seconds.
-/

theorem train_speed (distance time : ℝ) (h1 : distance = 320) (h2 : time = 12) :
  (distance / time) = 26.67 :=
by
  rw [h1, h2]
  norm_num
  sorry

end train_speed_l123_123304


namespace part_I_solution_part_II_solution_l123_123022

def f (x : ℝ) : ℝ := |2 * x + 2| - |x - 2|

theorem part_I_solution (x : ℝ) : f(x) > 2 ↔ x < -6 ∨ x > 2/3 := sorry

theorem part_II_solution (t : ℝ) : (∀ x : ℝ, f(x) ≥ t^2 - 7/2*t) ↔ (3/2 ≤ t ∧ t ≤ 2) := sorry

end part_I_solution_part_II_solution_l123_123022


namespace f_f_of_six_l123_123846

noncomputable def f (x : ℝ) (t : ℝ) : ℝ :=
  if x ≥ 3 then log 3 (x + t) else 3 ^ x

theorem f_f_of_six (t : ℝ) (h : log 3 (3 + t) = 0) : f (f 6 t) t = 4 := by 
  sorry

end f_f_of_six_l123_123846


namespace plane_SAB_perpendicular_to_base_ABC_l123_123362

open EuclideanGeometry

def Plane (p1 p2 p3 : Point) : Set Point :=
  {q : Point | ∃ (a b c : ℝ), a * p1 + b * p2 + c * p3 = q}

theorem plane_SAB_perpendicular_to_base_ABC {A B C D S : Point}
  (convex_quadrilateral : ConvexQuadrilateral A B C D)
  (h1 : dist B C * dist A D = dist B D * dist A C)
  (h2 : angle A D S = angle B D S)
  (h3 : angle A C S = angle B C S) :
  ∀ {p1 p2 p3 : Point}, conv_plane (Plane p1 p2 p3) ∧
  ∀ {base_plane : Set Point}, base_plane = Plane A B C D ∧
  ∃ line_point, line_point ∈ Plane S A B ∧ line_point ∈ base_plane →
  Plane S A B ⟂ Plane A B C D :=
sorry

end plane_SAB_perpendicular_to_base_ABC_l123_123362


namespace moles_of_nacl_formed_l123_123740

noncomputable def reaction (nh4cl: ℕ) (naoh: ℕ) : ℕ :=
  if nh4cl = naoh then nh4cl else min nh4cl naoh

theorem moles_of_nacl_formed (nh4cl: ℕ) (naoh: ℕ) (h_nh4cl: nh4cl = 2) (h_naoh: naoh = 2) :
  reaction nh4cl naoh = 2 :=
by
  rw [h_nh4cl, h_naoh]
  sorry

end moles_of_nacl_formed_l123_123740


namespace ArcherInGoldenArmorProof_l123_123930

-- Definitions of the problem
variables (soldiers : Nat) (archers soldiersInGolden : Nat) 
variables (soldiersInBlack archersInGolden archersInBlack swordsmenInGolden swordsmenInBlack : Nat)
variables (truthfulSwordsmenInBlack lyingArchersInBlack lyingSwordsmenInGold truthfulArchersInGold : Nat)
variables (yesToGold yesToArcher yesToMonday : Nat)

-- Given conditions
def ProblemStatement : Prop :=
  soldiers = 55 ∧
  yesToGold = 44 ∧
  yesToArcher = 33 ∧
  yesToMonday = 22 ∧
  soldiers = archers + (soldiers - archers) ∧
  soldiers = soldiersInGolden + soldiersInBlack ∧
  archers = archersInGolden + archersInBlack ∧
  soldiersInGolden = archersInGolden + swordsmenInGolden ∧
  soldiersInBlack = archersInBlack + swordsmenInBlack ∧
  truthfulSwordsmenInBlack = swordsmenInBlack ∧
  lyingArchersInBlack = archersInBlack ∧
  lyingSwordsmenInGold = swordsmenInGolden ∧
  truthfulArchersInGold = archersInGolden ∧
  yesToGold = truthfulArchersInGold + (swordsmenInGolden + archersInBlack) ∧
  yesToArcher = truthfulArchersInGold + lyingSwordsmenInGold

-- Conclusion
def Conclusion : Prop :=
  archersInGolden = 22

-- Proof statement
theorem ArcherInGoldenArmorProof : ProblemStatement → Conclusion :=
by
  sorry

end ArcherInGoldenArmorProof_l123_123930


namespace function_extreme_values_l123_123208

-- Conditions about M, N, and the function.
variables (a b : ℝ) (f : ℝ → ℝ)
def on_graph (a b : ℝ) : Prop := b = 1 / a
def symmetrical_about_y (a b : ℝ) : ℝ × ℝ := (-a, b)
def on_line_x_minus_y_plus_3 (p : ℝ × ℝ) : Prop := p.1 - p.2 + 3 = 0
def function_f (a b : ℝ) (x : ℝ) : ℝ := (a * b) * x^2 + (a + b) * x - 1

-- Lean statement to prove the properties of f(x) on the interval [-2, 2)
theorem function_extreme_values :
  on_graph a b →
  on_line_x_minus_y_plus_3 (symmetrical_about_y a b) →
  ab = 1 → a + b = 3 →
  let f := function_f a b in 
  ∀ x ∈ set.Ico (-2 : ℝ) 2, 
    -13/4 ≤ f x ∧ f x < 9 → ∃ t, ∃ m, f t = m ∧ m = -13/4 :=
by
  intros h1 h2 h3 h4 f x hx
  sorry

end function_extreme_values_l123_123208


namespace triangle_ABC_area_l123_123502

-- Define the conditions
variables (s : ℝ) (AM BC : ℝ)
axiom h1 : s^2 = 64
axiom h2 : s = 8
axiom h3 : AM = 8
axiom h4 : BC = 4

-- Prove that the area of triangle ABC is 16 cm^2
theorem triangle_ABC_area : (1 / 2) * BC * AM = 16 := by
  -- Leveraging the known values
  have h5 : BC = 4 := h4
  have h6 : AM = 8 := h3
  sorry  -- Proof steps go here

end triangle_ABC_area_l123_123502


namespace large_rectangle_perimeter_l123_123769

-- Definitions for conditions
def rectangle_area (l b : ℝ) := l * b
def is_large_rectangle_perimeter (l b perimeter : ℝ) := perimeter = 2 * (l + b)

-- Statement of the theorem
theorem large_rectangle_perimeter :
  ∃ (l b : ℝ), rectangle_area l b = 8 ∧ 
               (∀ l_rect b_rect: ℝ, is_large_rectangle_perimeter l_rect b_rect 32) :=
by
  sorry

end large_rectangle_perimeter_l123_123769


namespace max_friendship_pairs_l123_123920

-- Define conditions
def group_size : ℕ := 2020
def mutual_friendship (A B : ℕ) : Prop := true -- Mutual friendship is considered here as a concept

-- Define the property that no two people share a friend
def no_shared_friends (A B C : ℕ) (hAB : mutual_friendship A B) (hAC : mutual_friendship A C) : false := sorry
def no_shared_friends' (A B : ℕ) (hAB : mutual_friendship A B) (hAA' : mutual_friendship A A') (A' ≠ B) : false := sorry

-- Define the maximum number of friendship pairs 
theorem max_friendship_pairs:
  ∀ (G : finset ℕ), (G.card = group_size) -> 
  (∀ (A B : ℕ) (h : A ∈ G) (h' : B ∈ G), mutual_friendship A B -> no_shared_friends A B) -> 
  (∀ (A B C : ℕ) (hAB : mutual_friendship A B) (hAC : mutual_friendship A C), no_shared_friends' A B C hAB hAC) ->
  ∃ F : finset (finset ℕ), (∀ (f ∈ F), f.card = 2) ∧ F.card = 1010 := sorry

end max_friendship_pairs_l123_123920


namespace sebastian_students_count_l123_123575

theorem sebastian_students_count (n : ℕ) 
  (h1 : Sebastian is the 70th best student)
  (h2 : Sebastian is the 70th worst student) :
  n = 139 := by
  sorry

end sebastian_students_count_l123_123575


namespace number_of_height_groups_l123_123207

theorem number_of_height_groups
  (max_height : ℕ) (min_height : ℕ) (class_width : ℕ)
  (h_max : max_height = 186)
  (h_min : min_height = 167)
  (h_class_width : class_width = 3) :
  (max_height - min_height + class_width - 1) / class_width = 7 := by
  sorry

end number_of_height_groups_l123_123207


namespace smallest_four_digit_multiple_of_18_l123_123755

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 18 ∣ n ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ 18 ∣ m → n ≤ m := by
  use 1008
  sorry

end smallest_four_digit_multiple_of_18_l123_123755


namespace needed_irises_l123_123212

noncomputable def ratio_irises_roses := 3 / 7
noncomputable def initial_roses := 42
noncomputable def added_roses := 35
noncomputable def total_roses := initial_roses + added_roses

theorem needed_irises (ratio_irises_roses : ℚ) (total_roses : ℕ) : 
    (ratio_irises_roses * total_roses) = 33 :=
by
  have calculation : total_roses = initial_roses + added_roses := rfl
  have units_of_roses : ℕ := total_roses / 7
  have total_irises : ℕ := 3 * units_of_roses
  have result : total_irises = 33 := rfl 
  exact result

end needed_irises_l123_123212


namespace probability_cube_vertices_in_plane_l123_123820

open Finset

noncomputable def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_cube_vertices_in_plane : 
  let total_ways := choose 8 4 in
  let favorable_ways := 12 in
  0 < total_ways →  -- Ensure total_ways is non-zero to avoid division by zero
  let P := (favorable_ways : ℝ) / (total_ways : ℝ) in
  P = 6 / 35 :=
by 
  sorry

end probability_cube_vertices_in_plane_l123_123820


namespace no_obtuse_triangle_probability_eq_l123_123782

noncomputable def probability_no_obtuse_triangle : ℝ :=
  let num_points := 4
  -- Condition (1): Four points are chosen uniformly at random on a circle.
  -- Condition (2): An obtuse angle occurs if the minor arc exceeds π/2.
  9 / 64

theorem no_obtuse_triangle_probability_eq :
  let num_points := 4
  ∀ (points : Fin num_points → ℝ), 
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∠ (points i, points j, points k) < π / 2) →
  probability_no_obtuse_triangle = 9 / 64 :=
by
  sorry

end no_obtuse_triangle_probability_eq_l123_123782


namespace wall_bricks_count_l123_123328

theorem wall_bricks_count :
  ∃ x : ℕ, 
  let rate1 := x / 8,
      rate2 := x / 12,
      combined_rate := rate1 + rate2 - 15 in
  6 * combined_rate = x ∧ x = 360 :=
by
  sorry

end wall_bricks_count_l123_123328


namespace count_ways_5_balls_into_4_boxes_l123_123454

-- Statement of the problem:
theorem count_ways_5_balls_into_4_boxes : ∀ (dist_balls boxes : ℕ), 
  dist_balls = 5 → boxes = 4 → 
  (number_of_ways_to_put_balls_in_boxes dist_balls boxes = 61) := 
by
  intros dist_balls boxes h1 h2
  sorry

end count_ways_5_balls_into_4_boxes_l123_123454


namespace exist_five_nums_with_conditions_l123_123229

theorem exist_five_nums_with_conditions :
  ∃ (S : Set ℤ), S.card = 5 ∧ 
    (∃ (a b c : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a < b ∧ b < c ∧ a * b * c = 8) ∧
    (∃ (x y z : ℤ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x < y ∧ y < z ∧ x * y * z = 27) :=
by
  sorry

end exist_five_nums_with_conditions_l123_123229


namespace investment_A_l123_123698

noncomputable def investment_by_A (profit_B : ℝ) (investment_B : ℝ) (investment_C : ℝ) (profit_diff_AC : ℝ) : ℝ :=
  (prof_diff_AC * 10000 * (1 + investment_C/investment_B)/profit_B) / (1 + (investment_C / investment_B)/(investment_B / 10000))

theorem investment_A (profit_B : ℝ) (investment_B : ℝ) (investment_C : ℝ) (profit_diff_AC : ℝ) :
  investment_B = 10000 →
  investment_C = 12000 →
  profit_B = 2500 →
  profit_diff_AC = 999.9999999999998 →
  investment_by_A profit_B investment_B investment_C profit_diff_AC = 16000 :=
by
  intros hB hC hP hD
  rw [hB, hC, hP, hD]
  sorry

end investment_A_l123_123698


namespace find_n_l123_123318

theorem find_n
  (n : ℕ)
  (h1 : 2287 % n = r)
  (h2 : 2028 % n = r)
  (h3 : 1806 % n = r)
  (h_r_non_zero : r ≠ 0) : 
  n = 37 :=
by
  sorry

end find_n_l123_123318


namespace average_computation_l123_123424

variable {a b c X Y Z : ℝ}

theorem average_computation 
  (h1 : a + b + c = 15)
  (h2 : X + Y + Z = 21) :
  ((2 * a + 3 * X) + (2 * b + 3 * Y) + (2 * c + 3 * Z)) / 3 = 31 :=
by
  sorry

end average_computation_l123_123424


namespace hexagon_inscribes_circle_l123_123699

theorem hexagon_inscribes_circle
  (A B C D E F : Point)
  (h1 : dist A B = dist B C)
  (h2 : dist B C = dist C D)
  (h3 : dist C D = dist D E)
  (h4 : dist D E = dist E F)
  (h5 : dist E F = dist F A)
  (h6 : dist A D = dist B E)
  (h7 : dist B E = dist C F)
  (convex : is_convex_hexagon A B C D E F)
  : inscribed_circle A B C D E F :=
by 
  sorry

end hexagon_inscribes_circle_l123_123699


namespace archers_in_golden_armor_count_l123_123949

theorem archers_in_golden_armor_count : 
  ∃ (archers_golden : ℕ), 
    let total_soldiers := 55 in
    let golden_armor_affirmative := 44 in
    let archer_affirmative := 33 in
    let monday_affirmative := 22 in
    let number_archers_golden := archers_golden in
    (number_archers_golden = 22) ∧ 
    ∃ (archer : ℕ) (swordsman : ℕ) (golden : ℕ) (black : ℕ),
      archer + swordsman = total_soldiers ∧
      golden + black = total_soldiers ∧
      golden + black = total_soldiers ∧
      (swordsman * golden + archer * golden = golden_armor_affirmative) ∧
      (swordsman * archer + swordsman * golden = archer_affirmative) ∧
      ((!monday_affirmative ∧ swordsman * golden) + (!monday_affirmative ∧ archer * black) = monday_affirmative)
:= sorry

end archers_in_golden_armor_count_l123_123949


namespace stamps_sum_to_n_l123_123656

noncomputable def selectStamps : Prop :=
  ∀ (n : ℕ) (k : ℕ), n > 0 → 
                      ∃ stamps : List ℕ, 
                      stamps.length = k ∧ 
                      n ≤ stamps.sum ∧ stamps.sum < 2 * k → 
                      ∃ (subset : List ℕ), 
                      subset.sum = n

theorem stamps_sum_to_n : selectStamps := sorry

end stamps_sum_to_n_l123_123656


namespace circumscribed_circle_radius_l123_123899

theorem circumscribed_circle_radius (b c : ℝ) (A : ℝ) (R : ℝ)
  (h1 : b = 6) (h2 : c = 2) (h3 : A = π / 3) :
  R = (2 * Real.sqrt 21) / 3 :=
by
  sorry

end circumscribed_circle_radius_l123_123899


namespace line_curve_intersects_at_P_l123_123504

-- Define the parametric equations of the line
def line_x_eq (t : ℝ) (α : ℝ) : ℝ := t * Real.cos α
def line_y_eq (t : ℝ) (α : ℝ) : ℝ := t * Real.sin α

-- Define the parametric equations of the curve
def curve_x_eq (β : ℝ) : ℝ := 4 + 2 * Real.cos β
def curve_y_eq (β : ℝ) : ℝ := 2 * Real.sin β

-- Define the polar coordinates of P
def polar_coordinates_P (ρ θ : ℝ) : Prop :=
  θ = Real.pi / 6 ∧ ρ = 2 * Real.sqrt 3

-- Theorem statement
theorem line_curve_intersects_at_P (α β t : ℝ) (P : ℝ × ℝ) :
  (line_x_eq t α = curve_x_eq β ∧ line_y_eq t α = curve_y_eq β) →
  (polar_coordinates_P P.1 P.2) :=
by sorry

end line_curve_intersects_at_P_l123_123504


namespace average_percentage_decrease_is_correct_l123_123155

def old_revenue_A := 69.0
def new_revenue_A := 48.0
def old_revenue_B := 120.0
def new_revenue_B := 100.0
def old_revenue_C := 172.0
def new_revenue_C := 150.0

def percentage_decrease (old_revenue new_revenue : Float) : Float := 
  ((old_revenue - new_revenue) / old_revenue) * 100

def percentage_decrease_A := percentage_decrease old_revenue_A new_revenue_A
def percentage_decrease_B := percentage_decrease old_revenue_B new_revenue_B
def percentage_decrease_C := percentage_decrease old_revenue_C new_revenue_C

def average_percentage_decrease : Float := 
  (percentage_decrease_A + percentage_decrease_B + percentage_decrease_C) / 3

theorem average_percentage_decrease_is_correct :
  abs (average_percentage_decrease - 19.96) < 0.01 := by sorry

end average_percentage_decrease_is_correct_l123_123155


namespace suma_time_l123_123667

/--
Given that Renu can complete a work in 8 days, and together with Suma they can complete the same work in 3 days,
prove that Suma alone can complete the work in 4.8 days.
-/
theorem suma_time (W : ℝ) (h₁ : W / 8) (h₂ : W / 3) : (24 / 5 : ℝ) = 4.8 := by
  sorry

end suma_time_l123_123667


namespace alpha_beta_working_together_time_l123_123614

theorem alpha_beta_working_together_time
  (A B C : ℝ)
  (h : ℝ)
  (hA : A = B + 5)
  (work_together_A : A > 0)
  (work_together_B : B > 0)
  (work_together_C : C > 0)
  (combined_work : 1/A + 1/B + 1/C = 1/(A - 6))
  (combined_work2 : 1/A + 1/B + 1/C = 1/(B - 1))
  (time_gamma : 1/A + 1/B + 1/C = 2/C) :
  h = 4/3 :=
sorry

end alpha_beta_working_together_time_l123_123614


namespace num_archers_golden_armor_proof_l123_123960
noncomputable section

structure Soldier :=
  (is_archer : Bool)
  (is_golden : Bool)
  (tells_truth : Bool)

def count_soldiers (soldiers : List Soldier) : Nat :=
  soldiers.length

def count_truthful_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => s.tells_truth)).count q

def count_lying_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => ¬s.tells_truth)).count q

def num_archers_golden_armor (soldiers : List Soldier) : Nat :=
  (soldiers.filter (λ s => s.is_archer ∧ s.is_golden)).length

theorem num_archers_golden_armor_proof (soldiers : List Soldier)
  (h1 : count_soldiers soldiers = 55)
  (h2 : count_truthful_responses soldiers (λ s => s.is_golden) + 
        count_lying_responses soldiers (λ s => s.is_golden) = 44)
  (h3 : count_truthful_responses soldiers (λ s => s.is_archer) + 
        count_lying_responses soldiers (λ s => s.is_archer) = 33)
  (h4 : count_truthful_responses soldiers (λ s => true) + 
        count_lying_responses soldiers (λ s => false) = 22) :
  num_archers_golden_armor soldiers = 22 := by
  sorry

end num_archers_golden_armor_proof_l123_123960


namespace orange_juice_amount_l123_123731

theorem orange_juice_amount (strawberries yogurt total_ingredients : ℝ) 
  (h_strawberries : strawberries = 0.2) 
  (h_yogurt : yogurt = 0.1) 
  (h_total_ingredients : total_ingredients = 0.5) : 
  total_ingredients - (strawberries + yogurt) = 0.2 :=
by
  rw [h_strawberries, h_yogurt, h_total_ingredients]
  exact rfl

end orange_juice_amount_l123_123731


namespace circles_intersection_line_equation_l123_123412

noncomputable def circle1_intersection_line_equation (x y : ℝ) :=
  (x - 2)^2 + (y - 1)^2 = 4 ∧ x^2 + (y - 2)^2 = 9 → x + 2y - 1 = 0

theorem circles_intersection_line_equation (x y : ℝ) :
  (x - 2)^2 + (y - 1)^2 = 4 → x^2 + (y - 2)^2 = 9 → x + 2y - 1 = 0 :=
sorry

end circles_intersection_line_equation_l123_123412


namespace number_of_dolls_l123_123225

theorem number_of_dolls (total_toys : ℕ) (fraction_action_figures : ℚ) 
  (remaining_fraction_action_figures : fraction_action_figures = 1 / 4) 
  (remaining_fraction_dolls : 1 - fraction_action_figures = 3 / 4) 
  (total_toys_eq : total_toys = 24) : 
  (total_toys - total_toys * fraction_action_figures) = 18 := 
by 
  sorry

end number_of_dolls_l123_123225


namespace jonah_total_lemonade_l123_123729

theorem jonah_total_lemonade : 
  0.25 + 0.4166666666666667 + 0.25 + 0.5833333333333334 = 1.5 :=
by
  sorry

end jonah_total_lemonade_l123_123729


namespace domain_of_f_l123_123199

noncomputable def f : ℝ → ℝ := sorry

theorem domain_of_f :
  (∀ x : ℝ, x ≠ 0 → f x + f (1 / x) = x ^ 2) →
  ∀ x : ℝ, x ≠ 0 :=
by
  intros h x hx
  have h1 := h x hx
  sorry

end domain_of_f_l123_123199


namespace archers_in_golden_armor_l123_123957

theorem archers_in_golden_armor (total_soldiers archers_swordsmen total_affirmations armor_affirmations archer_affirmations monday_affirmations : ℕ)
  (h1: total_soldiers = 55)
  (h2: armor_affirmations = 44)
  (h3: archer_affirmations = 33)
  (h4: monday_affirmations = 22)
  (h5: ∑ x in ({armor_affirmations, archer_affirmations, monday_affirmations} : finset ℕ), x = 99) 
  : ∃ (archers_in_golden : ℕ), archers_in_golden = 22 := by
  -- Definitions and theorems will go here
  sorry

end archers_in_golden_armor_l123_123957


namespace midpoint_AB_is_correct_l123_123974

/--
In the Cartesian coordinate system, given points A (-1, 2) and B (3, 0), prove that the coordinates of the midpoint of segment AB are (1, 1).
-/
theorem midpoint_AB_is_correct :
  let A := (-1, 2)
  let B := (3, 0)
  (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1 := 
by {
  let A := (-1, 2)
  let B := (3, 0)
  sorry -- this part is omitted as no proof is needed
}

end midpoint_AB_is_correct_l123_123974


namespace angle_triple_supplement_l123_123273

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l123_123273


namespace find_eccentricity_of_hyperbola_l123_123863

noncomputable def parabola : set (ℝ × ℝ) := {p | ∃ y x, p = (x, y) ∧ y^2 = 4 * x}
noncomputable def hyperbola (m : ℝ) : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ x^2 / m - y^2 = 1}

noncomputable def focus_of_parabola : (ℝ × ℝ) := (1, 0)
noncomputable def directrix_of_parabola : set (ℝ × ℝ) := {p | p.fst = -1}

noncomputable def angle_AFB_is_60_degrees 
(A B F : ℝ × ℝ) : Prop := (A ≠ B) ∧ (∡ A F B = 60)

def eccentricity_of_hyperbola (m : ℝ) := 
sqrt (1 + 1/m)

theorem find_eccentricity_of_hyperbola {A B : ℝ × ℝ} {m : ℝ}
  (hA : A ∈ hyperbola m) 
  (hB : B ∈ hyperbola m) 
  (hF : focus_of_parabola = (1, 0))
  (hD : directrix_of_parabola ⊆ hyperbola m)
  (hAngle : angle_AFB_is_60_degrees A B focus_of_parabola):
  eccentricity_of_hyperbola m = sqrt 30 / 3 := 
sorry

end find_eccentricity_of_hyperbola_l123_123863


namespace value_of_expression_l123_123061

theorem value_of_expression (a b c : ℝ) (h : a * (-2)^5 + b * (-2)^3 + c * (-2) - 5 = 7) :
  a * 2^5 + b * 2^3 + c * 2 - 5 = -17 :=
by sorry

end value_of_expression_l123_123061


namespace difference_in_combined_area_l123_123455

-- Define the dimensions of the two rectangular sheets of paper
def paper1_length : ℝ := 11
def paper1_width : ℝ := 17
def paper2_length : ℝ := 8.5
def paper2_width : ℝ := 11

-- Define the areas of one side of each sheet
def area1 : ℝ := paper1_length * paper1_width -- 187
def area2 : ℝ := paper2_length * paper2_width -- 93.5

-- Define the combined areas of front and back of each sheet
def combined_area1 : ℝ := 2 * area1 -- 374
def combined_area2 : ℝ := 2 * area2 -- 187

-- Prove that the difference in combined area is 187
theorem difference_in_combined_area : combined_area1 - combined_area2 = 187 :=
by 
  -- Using the definitions above to simplify the goal
  sorry

end difference_in_combined_area_l123_123455


namespace fair_decision_l123_123067

def fair_selection (b c : ℕ) : Prop :=
  (b - c)^2 = b + c

theorem fair_decision (b c : ℕ) : fair_selection b c := by
  sorry

end fair_decision_l123_123067


namespace probability_red_ball_is_correct_l123_123717

noncomputable def probability_red_ball : ℚ :=
  let prob_A := 1 / 3
  let prob_B := 1 / 3
  let prob_C := 1 / 3
  let prob_red_A := 3 / 10
  let prob_red_B := 7 / 10
  let prob_red_C := 5 / 11
  (prob_A * prob_red_A) + (prob_B * prob_red_B) + (prob_C * prob_red_C)

theorem probability_red_ball_is_correct : probability_red_ball = 16 / 33 := 
by
  sorry

end probability_red_ball_is_correct_l123_123717


namespace pascal_triangle_sum_first_30_rows_l123_123885

theorem pascal_triangle_sum_first_30_rows : (∑ n in Finset.range 30, (n + 1)) = 465 := 
by
  sorry

end pascal_triangle_sum_first_30_rows_l123_123885


namespace expected_sides_general_expected_sides_rectangle_l123_123623

-- General Problem
theorem expected_sides_general (n k : ℕ) : 
  (∀ n k : ℕ, n ≥ 3 → k ≥ 0 → (1:ℝ) / ((k + 1:ℝ)) * (n + 4 * k:ℝ) ≤ (n + 4 * k) / (k + 1)) := 
begin
  sorry
end

-- Specific Problem for Rectangle
theorem expected_sides_rectangle (k : ℕ) :
  (∀ k : ℕ, k ≥ 0 → 4 = (4 + 4 * k) / (k + 1)) := 
begin
  sorry
end

end expected_sides_general_expected_sides_rectangle_l123_123623


namespace inequality_holds_for_positive_reals_equality_condition_l123_123180

theorem inequality_holds_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  4 * (a^3 + b^3 + c^3 + 3) ≥ 3 * (a + 1) * (b + 1) * (c + 1) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (4 * (a^3 + b^3 + c^3 + 3) = 3 * (a + 1) * (b + 1) * (c + 1)) ↔ (a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end inequality_holds_for_positive_reals_equality_condition_l123_123180


namespace square_perimeter_is_256_l123_123668

def area_rectangle (length width : ℝ) : ℝ := length * width

noncomputable def area_square (A : ℝ) : ℝ := 2 * A

noncomputable def side_length_square (A : ℝ) : ℝ := real.sqrt A

noncomputable def perimeter_square (side_length : ℝ) : ℝ := 4 * side_length

theorem square_perimeter_is_256 :
  area_rectangle 32 64 = 2048 ∧ area_square 2048 = 4096 ∧
  side_length_square 4096 = 64 ∧ perimeter_square 64 = 256 :=
by
  sorry

end square_perimeter_is_256_l123_123668


namespace alpha_plus_beta_l123_123849

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

axiom α_obtuse : π / 2 < α ∧ α < π
axiom β_obtuse : π / 2 < β ∧ β < π

axiom cos_α : Real.cos α = -2 * Real.sqrt 5 / 5
axiom sin_β : Real.sin β = Real.sqrt 10 / 10

theorem alpha_plus_beta : α + β = 7 * π / 4 :=
by {
  sorry
}

end alpha_plus_beta_l123_123849


namespace probability_no_obtuse_triangle_l123_123802

namespace CirclePoints

noncomputable def no_obtuse_triangle_probability : ℝ := 
  let p := 1/64 in
    p

theorem probability_no_obtuse_triangle (X : ℕ → ℝ) (hcirc : ∀ n, 0 ≤ X n ∧ X n < 2 * π) (hpoints : (∀ n m, n ≠ m → X n ≠ X m)) :
  no_obtuse_triangle_probability = 1/64 :=
sorry

end CirclePoints

end probability_no_obtuse_triangle_l123_123802


namespace constant_term_of_binomial_expansion_l123_123772

noncomputable def binomial_constant_term : ℚ :=
  let T := ((x : ℚ) / 2 - 1 / x^(1 / 3)) ^ 12 in
  ((-(1 / 8) * ↑(nat.choose 12 9))) -- This represents the binomial coefficient and the corresponding simplifications

theorem constant_term_of_binomial_expansion :
  ∀ (x : ℚ), binomial_constant_term = -55 / 2 :=
 by
  let x := 2
  rfl

end constant_term_of_binomial_expansion_l123_123772


namespace num_archers_golden_armor_proof_l123_123964
noncomputable section

structure Soldier :=
  (is_archer : Bool)
  (is_golden : Bool)
  (tells_truth : Bool)

def count_soldiers (soldiers : List Soldier) : Nat :=
  soldiers.length

def count_truthful_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => s.tells_truth)).count q

def count_lying_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => ¬s.tells_truth)).count q

def num_archers_golden_armor (soldiers : List Soldier) : Nat :=
  (soldiers.filter (λ s => s.is_archer ∧ s.is_golden)).length

theorem num_archers_golden_armor_proof (soldiers : List Soldier)
  (h1 : count_soldiers soldiers = 55)
  (h2 : count_truthful_responses soldiers (λ s => s.is_golden) + 
        count_lying_responses soldiers (λ s => s.is_golden) = 44)
  (h3 : count_truthful_responses soldiers (λ s => s.is_archer) + 
        count_lying_responses soldiers (λ s => s.is_archer) = 33)
  (h4 : count_truthful_responses soldiers (λ s => true) + 
        count_lying_responses soldiers (λ s => false) = 22) :
  num_archers_golden_armor soldiers = 22 := by
  sorry

end num_archers_golden_armor_proof_l123_123964


namespace ratio_ad_l123_123473

-- Definitions of conditions
variables {a b c d : ℝ}
def ratio_ab : Prop := a / b = 5 / 3
def ratio_bc : Prop := b / c = 1 / 5
def ratio_cd : Prop := c / d = 3 / 2

-- The theorem to prove
theorem ratio_ad (h₁ : ratio_ab) (h₂ : ratio_bc) (h₃ : ratio_cd) : a / d = 1 / 2 :=
by
  sorry

end ratio_ad_l123_123473


namespace arithmetic_sequence_sum_l123_123495

noncomputable def sum_of_first_30_terms (a : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range 30, a i

theorem arithmetic_sequence_sum (a : ℕ → ℝ)
  (h1 : a 1 + a 2 + a 3 = 3)
  (h2 : a 28 + a 29 + a 30 = 165) :
  sum_of_first_30_terms a = 840 :=
sorry

end arithmetic_sequence_sum_l123_123495


namespace smallest_four_digit_multiple_of_18_l123_123757

-- Define the concept of a four-digit number
def four_digit (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

-- Define the concept of a multiple of 18
def multiple_of_18 (N : ℕ) : Prop := ∃ k : ℕ, N = 18 * k

-- Define the combined condition for N being a four-digit multiple of 18
def four_digit_multiple_of_18 (N : ℕ) : Prop := four_digit N ∧ multiple_of_18 N

-- State that 1008 is the smallest such number
theorem smallest_four_digit_multiple_of_18 : ∀ N : ℕ, four_digit_multiple_of_18 N → 1008 ≤ N := 
by
  intros N H
  sorry

end smallest_four_digit_multiple_of_18_l123_123757


namespace option_a_option_b_option_d_l123_123844

open MeasureTheory

variables {Ω : Type*} [MeasurableSpace Ω]
variables {P : Measure Ω} [ProbabilityMeasure P]
variables {A B : Set Ω} [MeasurableSet A] [MeasurableSet B]

theorem option_a (hB_subset_A : B ⊆ A) (hPA : P A = 0.4) (hPB : P B = 0.2) :
  P (A ∪ B) = 0.4 ∧ P (A ∩ B) = 0.2 := sorry

theorem option_b (h_mutually_exclusive : P (A ∩ B) = 0) (hPA : P A = 0.4) (hPB : P B = 0.2) :
  P (A ∪ B) = 0.6 := sorry

theorem option_d (h_independent : indep_sets (measurable_set A) (measurable_set B) P)
  (hPA : P A = 0.4) (hPB : P B = 0.2) :
  P (A ∩ B) = 0.08 ∧ P (A ∪ B) = 0.52 := sorry

end option_a_option_b_option_d_l123_123844


namespace arithmetic_sequence_theorem_l123_123427

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (n * (a 1 + a n)) / 2

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_theorem (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : arithmetic_sequence a)
  (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h_a1_pos : a 1 > 0)
  (h_condition : -1 < a 7 / a 6 ∧ a 7 / a 6 < 0) :
  (∃ d, d < 0) ∧ (∀ n, S n > 0 → n ≤ 12) :=
sorry

end arithmetic_sequence_theorem_l123_123427


namespace archers_in_golden_armor_count_l123_123946

theorem archers_in_golden_armor_count : 
  ∃ (archers_golden : ℕ), 
    let total_soldiers := 55 in
    let golden_armor_affirmative := 44 in
    let archer_affirmative := 33 in
    let monday_affirmative := 22 in
    let number_archers_golden := archers_golden in
    (number_archers_golden = 22) ∧ 
    ∃ (archer : ℕ) (swordsman : ℕ) (golden : ℕ) (black : ℕ),
      archer + swordsman = total_soldiers ∧
      golden + black = total_soldiers ∧
      golden + black = total_soldiers ∧
      (swordsman * golden + archer * golden = golden_armor_affirmative) ∧
      (swordsman * archer + swordsman * golden = archer_affirmative) ∧
      ((!monday_affirmative ∧ swordsman * golden) + (!monday_affirmative ∧ archer * black) = monday_affirmative)
:= sorry

end archers_in_golden_armor_count_l123_123946


namespace triple_supplementary_angle_l123_123280

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l123_123280


namespace archers_in_golden_armor_l123_123950

theorem archers_in_golden_armor (total_soldiers archers_swordsmen total_affirmations armor_affirmations archer_affirmations monday_affirmations : ℕ)
  (h1: total_soldiers = 55)
  (h2: armor_affirmations = 44)
  (h3: archer_affirmations = 33)
  (h4: monday_affirmations = 22)
  (h5: ∑ x in ({armor_affirmations, archer_affirmations, monday_affirmations} : finset ℕ), x = 99) 
  : ∃ (archers_in_golden : ℕ), archers_in_golden = 22 := by
  -- Definitions and theorems will go here
  sorry

end archers_in_golden_armor_l123_123950


namespace reduced_travel_time_l123_123349

-- Definition of conditions as given in part a)
def initial_speed := 48 -- km/h
def initial_time := 50/60 -- hours (50 minutes)
def required_speed := 60 -- km/h
def reduced_time := 40/60 -- hours (40 minutes)

-- Problem statement
theorem reduced_travel_time :
  ∃ t2, (initial_speed * initial_time = required_speed * t2) ∧ (t2 = reduced_time) :=
by
  sorry

end reduced_travel_time_l123_123349


namespace min_sum_of_permutations_l123_123001

open BigOperators

def perms_of_six := [(1 : ℕ), 2, 3, 4, 5, 6]

def is_permutation (l : List ℕ) : Prop :=
  l ~ perms_of_six

theorem min_sum_of_permutations :
  ∀ (a b c : Fin 6 → ℕ),
  (is_permutation (List.ofFn a)) →
  (is_permutation (List.ofFn b)) →
  (is_permutation (List.ofFn c)) →
  ∑ i, (a i) * (b i) * (c i) = 162 :=
by
  sorry

end min_sum_of_permutations_l123_123001


namespace cube_vertices_probability_l123_123823

theorem cube_vertices_probability (totalVertices : ℕ) (selectedVertices : ℕ) 
   (totalCombinations : ℕ) (favorableOutcomes : ℕ) : 
   totalVertices = 8 ∧ selectedVertices = 4 ∧ totalCombinations = 70 ∧ favorableOutcomes = 12 → 
   (favorableOutcomes : ℚ) / totalCombinations = 6 / 35 := by
   sorry

end cube_vertices_probability_l123_123823


namespace conservation_of_mass_l123_123379

def molecular_weight_C := 12.01
def molecular_weight_H := 1.008
def molecular_weight_O := 16.00
def molecular_weight_Na := 22.99

def molecular_weight_C9H8O4 := (9 * molecular_weight_C) + (8 * molecular_weight_H) + (4 * molecular_weight_O)
def molecular_weight_NaOH := molecular_weight_Na + molecular_weight_O + molecular_weight_H
def molecular_weight_C7H6O3 := (7 * molecular_weight_C) + (6 * molecular_weight_H) + (3 * molecular_weight_O)
def molecular_weight_CH3COONa := (2 * molecular_weight_C) + (3 * molecular_weight_H) + (2 * molecular_weight_O) + molecular_weight_Na

theorem conservation_of_mass :
  (molecular_weight_C9H8O4 + molecular_weight_NaOH) = (molecular_weight_C7H6O3 + molecular_weight_CH3COONa) := by
  sorry

end conservation_of_mass_l123_123379


namespace proof_problem_l123_123077

-- Define the parametric equations for curve C
def parametric_curve (α : ℝ) : ℝ × ℝ :=
  (sqrt 2 * (sin α - cos α), (sqrt 2 / 2) * (sin α + cos α))

-- Define the polar equation for line l in Cartesian coordinates
def line_l (x y m : ℝ) : Prop := 
  x + 2 * y + m = 0

-- Define the Cartesian equation of curve C
def cartesian_curve (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Top-level theorem to prove equivalence of conditions to answers
theorem proof_problem (α x y : ℝ) (m : ℝ) 
  (h₁ : parametric_curve α = (x, y))
  (h₂ : cartesian_curve x y)
  (h₃ : ∃ (d : ℝ), d = (4 * sqrt 10) / 5)
  (h₄ : ∃ (m_pos : ℝ), m_pos = 2 * sqrt 2)
  (h₅ : ∃ (m_neg : ℝ), m_neg = -6 * sqrt 2) :
  (cartesian_curve x y) ∧ (line_l x y m) ∧ (m = 2 * sqrt 2 ∨ m = -6 * sqrt 2) :=
by { sorry }

end proof_problem_l123_123077


namespace exists_point_with_equilateral_projection_l123_123172

theorem exists_point_with_equilateral_projection (A B C : Type) [acute_triang ABC] :
  ∃ (Q : point_in_triangle ABC), let M := foot_of_perpendicular Q A B,
                                     N := foot_of_perpendicular Q B C,
                                     P := foot_of_perpendicular Q C A in
  equilateral_triangle M N P :=
sorry

end exists_point_with_equilateral_projection_l123_123172


namespace ratio_of_areas_l123_123343

noncomputable def perimeter := ℝ
def side_square (P : perimeter) := P / 4
def side_pentagon (P : perimeter) := P / 5
def circ_radius_square (P : perimeter) := (P * Real.sqrt 2) / 8
def circ_area_square (P : perimeter) : ℝ := Real.pi * (circ_radius_square P) ^ 2
def circ_radius_pentagon (P : perimeter) := P / (10 * Real.sin (Real.pi / 5))
def circ_area_pentagon (P : perimeter) : ℝ := Real.pi * (circ_radius_pentagon P) ^ 2

theorem ratio_of_areas (P : perimeter) : 
  (circ_area_square P) / (circ_area_pentagon P) = (500 - 100 * Real.sqrt 5) / 256 := sorry

end ratio_of_areas_l123_123343


namespace domain_width_of_composed_function_l123_123890

theorem domain_width_of_composed_function
  (h : ℝ → ℝ) (dom_h : ∀ x, x ∈ set.Icc (-8 : ℝ) 8 → x ∈ domain h) :
  (∀ x, x ∈ set.Icc (-24 : ℝ) 24 → h (x / 3) ∈ domain h) →
  (set.Icc (-24 : ℝ) 24).width = 48 :=
by
  sorry

end domain_width_of_composed_function_l123_123890


namespace infinite_primes_mod3_eq2_l123_123571

theorem infinite_primes_mod3_eq2 : 
  (∀ n : ℕ, ∃ p : ℕ, prime p ∧ p ≡ 2 [MOD 3]) := 
sorry

end infinite_primes_mod3_eq2_l123_123571


namespace no_obtuse_triangle_l123_123794

-- Conditions
def points_on_circle_uniformly_at_random (n : ℕ) : Prop :=
  ∀ i < n, ∀ j < n, i ≠ j → ∃ θ_ij : ℝ, 0 ≤ θ_ij ∧ θ_ij ≤ π

-- Theorem statement
theorem no_obtuse_triangle (hn : points_on_circle_uniformly_at_random 4) :
  let p := \left(\frac{1}{2}\right)^6 in
  p = \frac{1}{64} :=
sorry

end no_obtuse_triangle_l123_123794


namespace volumes_equal_l123_123010

theorem volumes_equal 
  (AB BC CA DA DB DC EF FG GE HE HF HG : ℝ)
  (h_AB : AB = 13) (h_BC : BC = 5) (h_CA : CA = 12)
  (h_DA : DA = 13) (h_DB : DB = 6) (h_DC : DC = 5)
  (h_EF : EF = 13) (h_FG : FG = 13) (h_GE : GE = 8)
  (h_HE : HE = 5) (h_HF : HF = 12) (h_HG : HG = 5) : 
  volume_of_tetrahedron AB BC CA DA DB DC = volume_of_tetrahedron EF FG GE HE HF HG :=
sorry

end volumes_equal_l123_123010


namespace solve_a_perpendicular_l123_123909

theorem solve_a_perpendicular (a : ℝ) : 
  ((2 * a + 5) * (2 - a) + (a - 2) * (a + 3) = 0) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end solve_a_perpendicular_l123_123909


namespace uranus_appears_7_minutes_after_6AM_l123_123611

def mars_last_seen := 0 * 60 + 10 -- 12:10 AM in minutes after midnight
def jupiter_after_mars := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def uranus_appearance := mars_last_seen + jupiter_after_mars + uranus_after_jupiter

theorem uranus_appears_7_minutes_after_6AM : uranus_appearance - (6 * 60) = 7 := by
  sorry

end uranus_appears_7_minutes_after_6AM_l123_123611


namespace length_BC_calculation_l123_123354

noncomputable def length_BC_of_triangle_area_is_50 (a : ℝ) (length_BC : ℝ) : Prop :=
  let B := (1 - a, (1 - a)^2)
  let C := (1 + a, (1 + a)^2)
  let base := (C.fst - B.fst)
  let height := (1 + a)^2 - 1
  (abs (a * (a^2 + 2*a)) = 50) → 
  length_BC = 2 * a

theorem length_BC_calculation :
  ∃ a length_BC, length_BC_of_triangle_area_is_50 a length_BC ∧ length_BC = 5.8 :=
by
  let a := 2.924  -- Approximate solution of the equation a·(a^2 + 2a) = 50
  let length_BC := 5.848
  use a, length_BC
  sorry

end length_BC_calculation_l123_123354


namespace total_cost_l123_123363

variable (a b : ℝ)

def tomato_cost (a : ℝ) := 30 * a
def cabbage_cost (b : ℝ) := 50 * b

theorem total_cost (a b : ℝ) : 
  tomato_cost a + cabbage_cost b = 30 * a + 50 * b := 
by 
  unfold tomato_cost cabbage_cost
  sorry

end total_cost_l123_123363


namespace no_such_function_f_l123_123307

noncomputable def p : ℝ → ℝ := sorry
noncomputable def q : ℝ → ℝ := sorry
noncomputable def h : ℝ → ℝ := sorry

axiom n : ℕ
axiom m : ℕ

axiom p_decreasing : ∀ (x y: ℝ), x < y → p y < p x
axiom functional_eq : ∀ x : ℝ, p (q (n * x + m : ℕ) + h x) = n * (q (p x) + h x) + m

theorem no_such_function_f : ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (q (p x) + h x) = f x ^ 2 + 1 := sorry

end no_such_function_f_l123_123307


namespace only_one_true_proposition_l123_123567

noncomputable def proposition1 (a b : Vector ℝ) : Prop :=
  (a.dot b < 0 → angle a b = π / 2)

noncomputable def proposition2 (α β : ℝ) : Prop :=
  (cos α * cos β = 1 → sin (α + β) = 0)

noncomputable def proposition3 (x : ℝ) : Prop :=
  isMinimal (λ x, sqrt (x^2 + 9) + 1 / sqrt (x^2 + 9)) 2

noncomputable def proposition4 (a : ℝ) : Prop :=
  (a > 0) →
  (¬ (∀ x, log (a * x^2 + 2 * a + 3) ∈ set.univ) ∧
   (∀ x ∈ set.Ioi 1, differentiable_on ℝ (λ x, x + a / x) → monotonic (λ x, x + a / x))) →
  a ≤ 1

theorem only_one_true_proposition :
  (proposition1 ∨ proposition2 ∨ proposition3 ∨ proposition4) ∧
  ((proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4) ∨
   (¬proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ ¬proposition4) ∨
   (¬proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ ¬proposition4) ∨
   (¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4))
  := sorry

end only_one_true_proposition_l123_123567


namespace intersection_of_asymptotes_l123_123742

theorem intersection_of_asymptotes :
  let f := λ x : ℝ, (x^2 - 6*x + 8) / (x^2 - 6*x + 9)
  ∃ x y : ℝ, (x = 3 ∧ y = 1) ∧ (∀ ε > 0, ∀ δ > 0, (∀ x' : ℝ, (0 < |x' - x| ∧ |x' - x| < δ) → |f x' - y| > ε))
:= sorry

end intersection_of_asymptotes_l123_123742


namespace solve_for_p_l123_123048

theorem solve_for_p (p : ℕ) : 16^6 = 4^p → p = 12 := by
  sorry

end solve_for_p_l123_123048


namespace total_groups_l123_123069

-- Define the problem conditions
def boys : ℕ := 9
def girls : ℕ := 12

-- Calculate the required combinations
def C (n k: ℕ) : ℕ := n.choose k
def groups_with_two_boys_one_girl : ℕ := C boys 2 * C girls 1
def groups_with_two_girls_one_boy : ℕ := C girls 2 * C boys 1

-- Statement of the theorem to prove
theorem total_groups : groups_with_two_boys_one_girl + groups_with_two_girls_one_boy = 1026 := 
by sorry

end total_groups_l123_123069


namespace bear_brothers_sum_l123_123673

def unique_chars (chars : List Char) : Prop :=
  chars.nodup

def valid_digit (ch : Char) (d : ℤ) : Prop :=
  d ≥ 0 ∧ d ≤ 9

def bear_digit_sum (big second : ℤ) : Prop :=
  big + second < 9

def bear_digit_compare (big second : ℤ) : Prop :=
  big > second

def is_three_digit (n : ℤ) : Prop :=
  n ≥ 100 ∧ n < 1000

def bear_product (x y : ℤ) : ℤ :=
  x * y

def sum_valid_bear_products : ℤ :=
  504 + 182

theorem bear_brothers_sum :
  ∃ (熊大 熊二 : ℤ),
  unique_chars ['熊', '大', '二'] ∧
  valid_digit '大' 熊大 ∧
  valid_digit '二' 熊二 ∧
  bear_digit_sum 熊大 熊二 ∧
  bear_digit_compare 熊大 熊二 ∧
  is_three_digit (bear_product 熊大 熊二) ∧
  sum_valid_bear_products = 686 :=
by
  sorry

end bear_brothers_sum_l123_123673


namespace amount_of_salmon_sold_first_week_l123_123232

-- Define the conditions
def fish_sold_in_two_weeks (x : ℝ) := x + 3 * x = 200

-- Define the theorem we want to prove
theorem amount_of_salmon_sold_first_week (x : ℝ) (h : fish_sold_in_two_weeks x) : x = 50 :=
by
  sorry

end amount_of_salmon_sold_first_week_l123_123232


namespace eleven_pow_four_l123_123298

theorem eleven_pow_four : 11 ^ 4 = 14641 := 
by sorry

end eleven_pow_four_l123_123298


namespace no_integer_solutions_l123_123191

theorem no_integer_solutions (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (hq : Nat.Prime (2*p + 1)) :
  ∀ (x y z : ℤ), x^p + 2 * y^p + 5 * z^p = 0 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end no_integer_solutions_l123_123191


namespace three_digit_integers_with_product_36_l123_123876

/--
There are 21 distinct 3-digit integers such that the product of their digits equals 36, and each digit is between 1 and 9.
-/
theorem three_digit_integers_with_product_36 : 
  ∃ n : ℕ, digit_product_count 36 3 n ∧ n = 21 :=
sorry

end three_digit_integers_with_product_36_l123_123876


namespace find_prime_p_l123_123099

open Nat

theorem find_prime_p (n p : ℕ) (hp_prime : Prime p)
  (h1 : p ∣ (3 * n - 1))
  (h2 : p ∣ (n - 10)) : p = 29 := 
sorry

end find_prime_p_l123_123099


namespace archers_in_golden_armor_count_l123_123943

theorem archers_in_golden_armor_count : 
  ∃ (archers_golden : ℕ), 
    let total_soldiers := 55 in
    let golden_armor_affirmative := 44 in
    let archer_affirmative := 33 in
    let monday_affirmative := 22 in
    let number_archers_golden := archers_golden in
    (number_archers_golden = 22) ∧ 
    ∃ (archer : ℕ) (swordsman : ℕ) (golden : ℕ) (black : ℕ),
      archer + swordsman = total_soldiers ∧
      golden + black = total_soldiers ∧
      golden + black = total_soldiers ∧
      (swordsman * golden + archer * golden = golden_armor_affirmative) ∧
      (swordsman * archer + swordsman * golden = archer_affirmative) ∧
      ((!monday_affirmative ∧ swordsman * golden) + (!monday_affirmative ∧ archer * black) = monday_affirmative)
:= sorry

end archers_in_golden_armor_count_l123_123943


namespace find_p_value_l123_123000

theorem find_p_value (x y : ℝ) 
  (h : abs (x - 1/2) + real.sqrt (y^2 - 1) = 0) : abs x + abs y = 3/2 := 
sorry

end find_p_value_l123_123000


namespace no_obtuse_triangle_probability_l123_123801

noncomputable def probability_no_obtuse_triangle : ℝ :=
 let θ := 1/2 in
 let prob_A2_A3_given_A0A1 := (3/8) * (3/8) in
 θ * prob_A2_A3_given_A0A1

theorem no_obtuse_triangle_probability :
  probability_no_obtuse_triangle = 9/128 :=
by
  sorry

end no_obtuse_triangle_probability_l123_123801


namespace shuttle_speed_conversion_l123_123342

theorem shuttle_speed_conversion
  (speed_kmph : ℕ)
  (seconds_hour : ℕ)
  (condition_speed : speed_kmph = 14400)
  (condition_seconds : seconds_hour = 3600) :
  speed_kmph / seconds_hour = 4 :=
by
  rw [condition_speed, condition_seconds]
  exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

end shuttle_speed_conversion_l123_123342


namespace diameter_of_inner_circles_l123_123618

theorem diameter_of_inner_circles (d E_diam : ℝ) (r_ratio : ℝ) (h_inside : True) (h_non_overlap : True) 
  (h_equal : True) (h_ratio : (π * 12^2 - 2 * (π * (d / 2)^2)) / (2 * (π * (d / 2)^2)) = r_ratio) :
  d = 4 * real.sqrt 2 :=
by
  have h_diam_E : E_diam = 24 := by sorry
  have h_radius_E : 2 * 12 = 24 := by sorry
  have h_area_E : π * 12^2 = 144 * π := by sorry
  have h : 144 * π - π * (d^2 / 2) = 4 * (π * (d^2 / 2)) := by sorry
  have h_combined : 144 * π = 9 * π * d^2 / 2 := by sorry
  have h_squared : d^2 = 32 := by sorry
  have h_final : d = 4 * real.sqrt 2 := by sorry
  exact h_final

end diameter_of_inner_circles_l123_123618


namespace number_of_distinct_parabolas_passing_through_0_1_l123_123588

theorem number_of_distinct_parabolas_passing_through_0_1 :
  let A := {n | -5 ≤ n ∧ n ≤ 5 ∧ n ∈ Set.Univ  ℤ}
  let downward_opening (a : ℤ) := a < 0
  let distinct (a b c : ℤ) := a ≠ b ∧ b ≠ c ∧ a ≠ c
  let values (n : ℤ) := n ∈ A
  let pass_through_0_1 (c : ℤ) := c = -1
  ∃ (f : ℤ → ℤ → ℤ → ℕ),
    (∀ a b c, downward_opening a → distinct a b c → values a → values b → values c → pass_through_0_1 c → f a b c = 1) →
      (∑ a b c in A, if (downward_opening a ∧ distinct a b c ∧ values a ∧ values b ∧ values c ∧ pass_through_0_1 c) then f a b c else 0) = 36 :=
by
  let A := {n | -5 ≤ n ∧ n ≤ 5 ∧ n ∈ Set.Univ ℤ}
  let downward_opening (a : ℤ) := a < 0
  let distinct (a b c : ℤ) := a ≠ b ∧ b ≠ c ∧ a ≠ c
  let values (n : ℤ) := n ∈ A
  let pass_through_0_1 (c : ℤ) := c = -1
  sorry

end number_of_distinct_parabolas_passing_through_0_1_l123_123588


namespace intersection_of_asymptotes_l123_123741

theorem intersection_of_asymptotes :
  let f := λ x : ℝ, (x^2 - 6*x + 8) / (x^2 - 6*x + 9)
  ∃ x y : ℝ, (x = 3 ∧ y = 1) ∧ (∀ ε > 0, ∀ δ > 0, (∀ x' : ℝ, (0 < |x' - x| ∧ |x' - x| < δ) → |f x' - y| > ε))
:= sorry

end intersection_of_asymptotes_l123_123741


namespace circle_problem_l123_123098

theorem circle_problem
  (E F : ℝ × ℝ)
  (G C1_center : ℝ × ℝ)
  (P A B M N : ℝ × ℝ)
  (hE : E = (-2, 0))
  (hF : F = (-4, 2))
  (hA : A = (-6, 0))
  (hB : B = (-2, 0))
  (hG : G = (-2, -4))
  (h_c1_center : 2 * C1_center.1 - C1_center.2 + 8 = 0)
  (hP_not_A_B : P ≠ A ∧ P ≠ B)
  (C ON_PA : ∃ k : ℝ, M = (0, k*P.2 / (P.1 + 6)))
  (C ON_PB : ∃ k : ℝ, N = (0, k*P.2 / (P.1 + 2))) :
  ∃ Center : ℝ × ℝ, 
  (∀ x y : ℝ, (x + 4)^2 + y^2 = 4) ∧ 
  (∀ k : ℝ, 3*k + 4*-4 + 22 = 0 ∨ k = -2) ∧ 
  (∃ Point : ℝ × ℝ, Point = (-2*√3, 0)) :=
sorry

end circle_problem_l123_123098


namespace no_representation_of_form_eight_k_plus_3_or_5_l123_123572

theorem no_representation_of_form_eight_k_plus_3_or_5 (k : ℤ) :
  ∀ x y : ℤ, (8 * k + 3 ≠ x^2 - 2 * y^2) ∧ (8 * k + 5 ≠ x^2 - 2 * y^2) :=
by sorry

end no_representation_of_form_eight_k_plus_3_or_5_l123_123572


namespace profit_percentage_correct_l123_123692

def cost_price_A : ℝ := 60
def cost_price_B : ℝ := 45
def cost_price_C : ℝ := 30

def selling_price_A : ℝ := 75
def selling_price_B : ℝ := 54
def selling_price_C : ℝ := 39

def total_cost_price : ℝ := cost_price_A + cost_price_B + cost_price_C
def total_selling_price : ℝ := selling_price_A + selling_price_B + selling_price_C
def total_profit : ℝ := total_selling_price - total_cost_price
def profit_percentage : ℝ := (total_profit / total_cost_price) * 100

theorem profit_percentage_correct : profit_percentage = 24.44 := by
  sorry

end profit_percentage_correct_l123_123692


namespace ab_value_l123_123466

/-- 
  Given the conditions:
  - a - b = 10
  - a^2 + b^2 = 210
  Prove that ab = 55.
-/
theorem ab_value (a b : ℝ) (h1 : a - b = 10) (h2 : a^2 + b^2 = 210) : a * b = 55 :=
by
  sorry

end ab_value_l123_123466


namespace three_digit_integers_with_product_36_l123_123879

-- Definition of the problem conditions
def is_three_digit_integer (n : Nat) : Prop :=
  100 ≤ n ∧ n < 1000

def digit_product_is_36 (n : Nat) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  1 ≤ d1 ∧ d1 ≤ 9 ∧ 1 ≤ d2 ∧ d2 ≤ 9 ∧ 1 ≤ d3 ∧ d3 ≤ 9 ∧ (d1 * d2 * d3 = 36)

-- The statement of the proof
theorem three_digit_integers_with_product_36 :
  {n : Nat | is_three_digit_integer n ∧ digit_product_is_36 n}.toFinset.card = 21 := 
by
  sorry

end three_digit_integers_with_product_36_l123_123879


namespace number_of_archers_in_golden_armor_l123_123967

-- Define the problem context
structure Soldier where
  is_archer : Bool
  wears_golden_armor : Bool

def truth_teller (s : Soldier) (is_black_armor : Bool) : Bool :=
  if s.is_archer then s.wears_golden_armor = is_black_armor
  else s.wears_golden_armor ≠ is_black_armor

def response (s : Soldier) (question : String) (is_black_armor : Bool) : Bool :=
  match question with
  | "Are you wearing golden armor?" => if truth_teller s is_black_armor then s.wears_golden_armor else ¬s.wears_golden_armor
  | "Are you an archer?" => if truth_teller s is_black_armor then s.is_archer else ¬s.is_archer
  | "Is today Monday?" => if truth_teller s is_black_armor then True else False -- An assumption that today not being Monday means False
  | _ => False

-- Problem condition setup
def total_soldiers : Nat := 55
def golden_armor_yes : Nat := 44
def archer_yes : Nat := 33
def monday_yes : Nat := 22

-- Define the main theorem
theorem number_of_archers_in_golden_armor :
  ∃ l : List Soldier,
    l.length = total_soldiers ∧
    l.countp (λ s => response s "Are you wearing golden armor?" True) = golden_armor_yes ∧
    l.countp (λ s => response s "Are you an archer?" True) = archer_yes ∧
    l.countp (λ s => response s "Is today Monday?" True) = monday_yes ∧
    l.countp (λ s => s.is_archer ∧ s.wears_golden_armor) = 22 :=
sorry

end number_of_archers_in_golden_armor_l123_123967


namespace mowing_time_l123_123553

theorem mowing_time (length width: ℝ) (swath_width_overlap_rate: ℝ)
                    (walking_speed: ℝ) (ft_per_inch: ℝ)
                    (length_eq: length = 100)
                    (width_eq: width = 120)
                    (swath_eq: swath_width_overlap_rate = 24)
                    (walking_eq: walking_speed = 4500)
                    (conversion_eq: ft_per_inch = 1/12) :
                    (length / walking_speed) * (width / (swath_width_overlap_rate * ft_per_inch)) = 1.33 :=
by
    rw [length_eq, width_eq, swath_eq, walking_eq, conversion_eq]
    exact sorry

end mowing_time_l123_123553


namespace min_sum_l123_123411

theorem min_sum (n : ℕ) (hn : n ≥ 3) (a b : Fin 2n → ℝ)
  (ha_nonneg : ∀ i, 0 ≤ a i)
  (hb_nonneg : ∀ i, 0 ≤ b i)
  (h_sum : (∑ i, a i) = (∑ i, b i) ∧ (0 < (∑ i, a i)))
  (h_cond : ∀ i : Fin 2n, a i * a ((i + 2) % 2n) ≥ b i + b ((i + 1) % 2n)) :
  (∑ i, a i) = if n = 3 then 12 else if n ≥ 4 then 16 else sorry :=
sorry

end min_sum_l123_123411


namespace extreme_value_and_inequality_l123_123434

theorem extreme_value_and_inequality
  (f : ℝ → ℝ)
  (a c : ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_extreme : f 1 = -2)
  (h_f_def : ∀ x : ℝ, f x = a * x^3 + c * x)
  (h_a_c : a = 1 ∧ c = -3) :
  (∀ x : ℝ, x < -1 → deriv f x > 0) ∧
  (∀ x : ℝ, -1 < x ∧ x < 1 → deriv f x < 0) ∧
  (∀ x : ℝ, 1 < x → deriv f x > 0) ∧
  f (-1) = 2 ∧
  (∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 → |f x₁ - f x₂| < 4) :=
by sorry

end extreme_value_and_inequality_l123_123434


namespace number_of_archers_in_golden_armor_l123_123970

-- Define the problem context
structure Soldier where
  is_archer : Bool
  wears_golden_armor : Bool

def truth_teller (s : Soldier) (is_black_armor : Bool) : Bool :=
  if s.is_archer then s.wears_golden_armor = is_black_armor
  else s.wears_golden_armor ≠ is_black_armor

def response (s : Soldier) (question : String) (is_black_armor : Bool) : Bool :=
  match question with
  | "Are you wearing golden armor?" => if truth_teller s is_black_armor then s.wears_golden_armor else ¬s.wears_golden_armor
  | "Are you an archer?" => if truth_teller s is_black_armor then s.is_archer else ¬s.is_archer
  | "Is today Monday?" => if truth_teller s is_black_armor then True else False -- An assumption that today not being Monday means False
  | _ => False

-- Problem condition setup
def total_soldiers : Nat := 55
def golden_armor_yes : Nat := 44
def archer_yes : Nat := 33
def monday_yes : Nat := 22

-- Define the main theorem
theorem number_of_archers_in_golden_armor :
  ∃ l : List Soldier,
    l.length = total_soldiers ∧
    l.countp (λ s => response s "Are you wearing golden armor?" True) = golden_armor_yes ∧
    l.countp (λ s => response s "Are you an archer?" True) = archer_yes ∧
    l.countp (λ s => response s "Is today Monday?" True) = monday_yes ∧
    l.countp (λ s => s.is_archer ∧ s.wears_golden_armor) = 22 :=
sorry

end number_of_archers_in_golden_armor_l123_123970


namespace friends_same_group_probability_l123_123193

noncomputable def probability_same_group (n : ℕ) (groups : ℕ) : ℚ :=
  1 / groups * 1 / groups

theorem friends_same_group_probability :
  ∀ (students groups : ℕ), 
  students = 900 → groups = 4 →
  (probability_same_group students groups = 1 / 16) :=
by
  intros students groups h_students h_groups
  rw [probability_same_group]
  have h1 : students = 900 := h_students
  have h2 : groups = 4 := h_groups
  simp [h1, h2]
  norm_num
  sorry

end friends_same_group_probability_l123_123193


namespace max_valid_subset_cardinality_l123_123396

-- Define the problem conditions
def consecutive_natural_numbers : List Nat := List.range (2014 + 1)

def no_five_times_relation (s : Set Nat) : Prop :=
  ∀ a b ∈ s, ¬ (b = 5 * a)

def max_valid_subset_size (n : Nat) (s : Set Nat) : Prop :=
  no_five_times_relation s ∧ s.card = n

-- State the equivalent proof problem
theorem max_valid_subset_cardinality
  (s : Set Nat) (h : s ⊆ consecutive_natural_numbers) :
  ∃ n, max_valid_subset_size n s ∧ n = 1679 :=
sorry

end max_valid_subset_cardinality_l123_123396


namespace find_value_of_expression_l123_123831

theorem find_value_of_expression (m n : ℝ) 
  (h1 : m^2 + 2 * m * n = 3) 
  (h2 : m * n + n^2 = 4) : 
  m^2 + 3 * m * n + n^2 = 7 := 
by
  sorry

end find_value_of_expression_l123_123831


namespace problem_l123_123847

-- Define the conditions
structure Point where
  x : ℝ
  y : ℝ

def on_circle (D : Point) : Prop := 
  D.x^2 + D.y^2 = 12

def E (D : Point) : Point := 
  {x := D.x, y := 0}

def P (D : Point) (P : Point) : Prop := 
  P.x = (sqrt 3 / 3) * D.x ∧ P.y = (1 / 2) * D.y

def curve_C (P : Point) : Prop := 
  (P.x^2 / 4) + (P.y^2 / 3) = 1

-- Define the tangency and quadrilateral area conditions
def line_l (k m : ℝ) (P : Point) : Prop := 
  P.y = k * P.x + m

def tangency_condition (k m : ℝ) : Prop := 
  ∃ x y, line_l k m ⟨x, y⟩ ∧ curve_C ⟨x, y⟩ ∧ 
  (3 + 4 * k^2) * x^2 + 8 * k * m * x + 4 * m^2 - 12 = 0

def quadrilateral_area (A1 A2 M N : Point) : ℝ :=
  let d1 := abs ((-2) * k + m) / sqrt (1 + k^2) in
  let d2 := abs (2 * k + m) / sqrt (1 + k^2) in
  let mn2 := 16 - ( 
    ( (2 * k - m)^2 / (1 + k^2) + (2 * k + m)^2 / (1 + k^2) - 
    (2 * abs (4 * k^2 - m^2)) / (1 + k^2))) in
  1 / 2 * sqrt ((6 + 16 * k^2) / (1 + k^2) * (16 / (1 + k^2))) >>= 
  λ _ => 4 * sqrt (3 + 4 * k^2) / ((1 + k^2)^2) 

-- Lean statement for the proof problem
theorem problem (D : Point) (P : Point) (k m : ℝ) :
  on_circle D →
  E D →
  P D P →
  curve_C P ∧ (
  ∃ A1 A2 M N : Point, 
    line_l k m P ∧ tangency_condition k m ∧ 
    quadrilateral_area A1 A2 M N = 4 * sqrt 3 ∧ k = 0
  ) := 
by
  intros hD hE hP
  sorry

end problem_l123_123847


namespace integration_of_polynomial_l123_123468

theorem integration_of_polynomial :
  (∫ x in 0..2, 3*x^2 + 1) = 10 :=
by sorry

end integration_of_polynomial_l123_123468


namespace champagne_equality_impossible_l123_123188

theorem champagne_equality_impossible :
  ∀ (n : ℕ) (glasses : Fin n → ℚ)
    (h1 : n = 2018)
    (h2 : ∃! i : Fin n, glasses i = 2 ∧ ∀ j : Fin n, i ≠ j → glasses j = 1),
    ¬(∃ m : ℚ, ∀ i : Fin n, glasses i = m) :=
by
  intro n glasses h1 h2
  cases h1
  rcases h2 with ⟨i, ⟨hi2, hi1⟩⟩
  have h_total : ∑ i, glasses i = 2019 := sorry
  intro h_eq
  cases h_eq with m h_glasses
  have h_m : m * 2018 = 2019 := sorry
  have h_m_not_int : ¬ ∃ k : ℤ, k = 2019 / 2018 := sorry
  contradiction

end champagne_equality_impossible_l123_123188


namespace polygonal_chain_length_l123_123095

theorem polygonal_chain_length (n : ℕ) : 
  ∃ p : list (ℝ × ℝ), (∀ q ∈ p, 0 ≤ q.1 ∧ q.1 ≤ 1 ∧ 0 ≤ q.2 ∧ q.2 ≤ 1) ∧ 
  list.length p = n^2 ∧ 
  sum (list.map (λ i, dist (p.nth_le i sorry) (p.nth_le (i + 1) sorry)) (list.range (n^2 - 1))) ≤ 2 * n :=
sorry

end polygonal_chain_length_l123_123095


namespace probability_no_obtuse_triangle_l123_123808

namespace CirclePoints

noncomputable def no_obtuse_triangle_probability : ℝ := 
  let p := 1/64 in
    p

theorem probability_no_obtuse_triangle (X : ℕ → ℝ) (hcirc : ∀ n, 0 ≤ X n ∧ X n < 2 * π) (hpoints : (∀ n m, n ≠ m → X n ≠ X m)) :
  no_obtuse_triangle_probability = 1/64 :=
sorry

end CirclePoints

end probability_no_obtuse_triangle_l123_123808


namespace baking_completion_time_l123_123323

theorem baking_completion_time (start_time : ℕ) (partial_bake_time : ℕ) (fraction_baked : ℕ) :
  start_time = 9 → partial_bake_time = 3 → fraction_baked = 4 →
  (start_time + (partial_bake_time * fraction_baked)) = 21 :=
by
  intros h_start h_partial h_fraction
  sorry

end baking_completion_time_l123_123323


namespace solution_set_of_inequality_l123_123601

theorem solution_set_of_inequality (x : ℝ) : x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := 
by
  sorry

end solution_set_of_inequality_l123_123601


namespace lawnmower_value_depreciation_l123_123364

theorem lawnmower_value_depreciation (C : ℝ) (r1 r2 : ℝ) :
  C = 100 → r1 = 0.25 → r2 = 0.20 → 
  let V1 := C * (1 - r1) in
  let V2 := V1 * (1 - r2) in
  V2 = 60 :=
begin
  intros C_eq r1_eq r2_eq,
  rw [C_eq, r1_eq, r2_eq],
  let V1 := 100 * (1 - 0.25),
  let V2 := V1 * (1 - 0.20),
  have V1_eq: V1 = 75, by norm_num,
  have V2_eq: V2 = 60, by norm_num,
  assumption,
end

end lawnmower_value_depreciation_l123_123364


namespace ArcherInGoldenArmorProof_l123_123932

-- Definitions of the problem
variables (soldiers : Nat) (archers soldiersInGolden : Nat) 
variables (soldiersInBlack archersInGolden archersInBlack swordsmenInGolden swordsmenInBlack : Nat)
variables (truthfulSwordsmenInBlack lyingArchersInBlack lyingSwordsmenInGold truthfulArchersInGold : Nat)
variables (yesToGold yesToArcher yesToMonday : Nat)

-- Given conditions
def ProblemStatement : Prop :=
  soldiers = 55 ∧
  yesToGold = 44 ∧
  yesToArcher = 33 ∧
  yesToMonday = 22 ∧
  soldiers = archers + (soldiers - archers) ∧
  soldiers = soldiersInGolden + soldiersInBlack ∧
  archers = archersInGolden + archersInBlack ∧
  soldiersInGolden = archersInGolden + swordsmenInGolden ∧
  soldiersInBlack = archersInBlack + swordsmenInBlack ∧
  truthfulSwordsmenInBlack = swordsmenInBlack ∧
  lyingArchersInBlack = archersInBlack ∧
  lyingSwordsmenInGold = swordsmenInGolden ∧
  truthfulArchersInGold = archersInGolden ∧
  yesToGold = truthfulArchersInGold + (swordsmenInGolden + archersInBlack) ∧
  yesToArcher = truthfulArchersInGold + lyingSwordsmenInGold

-- Conclusion
def Conclusion : Prop :=
  archersInGolden = 22

-- Proof statement
theorem ArcherInGoldenArmorProof : ProblemStatement → Conclusion :=
by
  sorry

end ArcherInGoldenArmorProof_l123_123932


namespace angle_C_is_3pi_over_4_l123_123065

theorem angle_C_is_3pi_over_4 (A B C : ℝ) (a b c : ℝ) (h_tri : 0 < B ∧ B < π ∧ 0 < C ∧ C < π) 
  (h_eq : b * Real.cos C + c * Real.sin B = 0) : C = 3 * π / 4 :=
by
  sorry

end angle_C_is_3pi_over_4_l123_123065


namespace ratio_of_triangle_areas_l123_123505

theorem ratio_of_triangle_areas 
  (X Y Z P : Type)
  [triangle_XYZ : triangle X Y Z]
  (hXY : XY = 21)
  (hXZ : XZ = 28)
  (hYZ : YZ = 25)
  (hXP_angle_bisector : is_angle_bisector X P Y Z) :
    area_ratio (triangle X Y P) (triangle X Z P) = 3 / 4 := 
sorry

end ratio_of_triangle_areas_l123_123505


namespace no_obtuse_triangle_probability_eq_l123_123785

noncomputable def probability_no_obtuse_triangle : ℝ :=
  let num_points := 4
  -- Condition (1): Four points are chosen uniformly at random on a circle.
  -- Condition (2): An obtuse angle occurs if the minor arc exceeds π/2.
  9 / 64

theorem no_obtuse_triangle_probability_eq :
  let num_points := 4
  ∀ (points : Fin num_points → ℝ), 
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∠ (points i, points j, points k) < π / 2) →
  probability_no_obtuse_triangle = 9 / 64 :=
by
  sorry

end no_obtuse_triangle_probability_eq_l123_123785


namespace angle_bisector_intersection_area_l123_123034

theorem angle_bisector_intersection_area (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let s := (a + b + c) / 2 in 
  let T := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  let t := 2 * a * b * c * T / ((a + b) * (b + c) * (c + a)) in
  t = 2 * a * b * c * Real.sqrt (s * (s - a) * (s - b) * (s - c)) / ((a + b) * (b + c) * (c + a)) :=
by
  sorry

end angle_bisector_intersection_area_l123_123034


namespace area_sin_cos_eq_l123_123194

noncomputable def area_between_curves : ℝ :=
  ∫ x in 0..(π / 4), (cos x - sin x) +
  ∫ x in (π / 4)..(5 * π / 4), (sin x - cos x) +
  ∫ x in (5 * π / 4)..(2 * π), (cos x - sin x)

theorem area_sin_cos_eq : area_between_curves = 2 * sqrt 2 :=
by
  -- Proof is omitted
  sorry

end area_sin_cos_eq_l123_123194


namespace time_to_pass_platform_l123_123301

-- Definitions of given conditions
def length_of_train := 1200 -- meters
def time_to_cross_tree := 120 -- seconds
def length_of_platform := 500 -- meters

-- Derived speed of the train
def speed_of_train := length_of_train / time_to_cross_tree -- meters per second

-- The goal is to determine the time to pass the platform
theorem time_to_pass_platform : 
  let total_distance := length_of_train + length_of_platform in
  let time := total_distance / speed_of_train in
  time = 170 := 
by
  -- Proof goes here
  sorry

end time_to_pass_platform_l123_123301


namespace quilt_patch_cost_l123_123506

-- conditions
def quilt_length : ℕ := 16
def quilt_width : ℕ := 20
def patch_area : ℕ := 4
def first_ten_patch_cost : ℕ := 10
def subsequent_patch_cost : ℕ := 5

theorem quilt_patch_cost :
  let total_quilt_area := quilt_length * quilt_width,
      total_patches := total_quilt_area / patch_area,
      cost_first_ten := 10 * first_ten_patch_cost,
      remaining_patches := total_patches - 10,
      cost_remaining := remaining_patches * subsequent_patch_cost,
      total_cost := cost_first_ten + cost_remaining
  in total_cost = 450 :=
by
  sorry

end quilt_patch_cost_l123_123506


namespace no_line_intersects_all_sides_of_triangle_l123_123570

theorem no_line_intersects_all_sides_of_triangle (A B C : Point) (l : Line) :
  ¬ (∃ (P1 P2 P3 : Point), P1 ∈ l ∧ P2 ∈ l ∧ P3 ∈ l ∧ P1 ∈ line_through A B ∧ 
  P2 ∈ line_through B C ∧ P3 ∈ line_through A C) :=
begin
  sorry
end

end no_line_intersects_all_sides_of_triangle_l123_123570


namespace chess_prize_orders_l123_123913

theorem chess_prize_orders : 
  let players := {1, 2, 3} -- set of players
  let matches := 2 -- number of outcomes per match
  ∃ n : ℕ, n = 2 * 2 ∧ n = 4 :=
by
  sorry

end chess_prize_orders_l123_123913


namespace not_possible_coloring_l123_123987

def color : Nat → Option ℕ := sorry

def all_colors_used (f : Nat → Option ℕ) : Prop := 
  (∃ n, f n = some 0) ∧ (∃ n, f n = some 1) ∧ (∃ n, f n = some 2)

def valid_coloring (f : Nat → Option ℕ) : Prop :=
  ∀ (a b : Nat), 1 < a → 1 < b → f a ≠ f b → f (a * b) ≠ f a ∧ f (a * b) ≠ f b

theorem not_possible_coloring : ¬ (∃ f : Nat → Option ℕ, all_colors_used f ∧ valid_coloring f) := 
sorry

end not_possible_coloring_l123_123987


namespace sum_of_eight_numbers_l123_123903

theorem sum_of_eight_numbers (nums : List ℝ) (h_len : nums.length = 8) (h_avg : (nums.sum / 8) = 5.5) : nums.sum = 44 :=
by
  sorry

end sum_of_eight_numbers_l123_123903


namespace fair_attendance_percent_l123_123206

variable (A : ℝ)

theorem fair_attendance_percent (A_pos : A > 0) : 
  let projected_attendance := 1.25 * A;
      actual_attendance := 0.80 * A in
  (actual_attendance / projected_attendance) * 100 = 64 :=
by
  -- This part will contain the actual proof. For now, we include sorry to bypass the proof.
  sorry

end fair_attendance_percent_l123_123206


namespace abc_divisible_by_6_l123_123117

theorem abc_divisible_by_6 (a b c : ℤ) (h : 18 ∣ (a^3 + b^3 + c^3)) : 6 ∣ (a * b * c) :=
by
  sorry

end abc_divisible_by_6_l123_123117


namespace smallest_k_sqrt_diff_l123_123151

-- Define a predicate for the condition that the square root difference is less than 1
def sqrt_diff_less_than_one (a b : ℕ) : Prop :=
  |real.sqrt a - real.sqrt b| < 1

-- Define the main theorem that encapsulates the problem statement
theorem smallest_k_sqrt_diff (cards : Finset ℕ) (h : cards = Finset.range 2016) : 
  ∃ k : ℕ, k = 45 ∧ ∀ s : Finset ℕ, s.card = k →
    ∃ (x y ∈ s), x ≠ y ∧ sqrt_diff_less_than_one x y :=
begin
  sorry
end

end smallest_k_sqrt_diff_l123_123151


namespace archers_in_golden_l123_123935

variables (soldiers archers swordsmen wearing_golden wearing_black : ℕ)
variables (truth_swordsmen_black lie_swordsmen_golden lie_archers_black truth_archers_golden : ℕ)

-- Given conditions
variables (cond1 : archers + swordsmen = 55)
variables (cond2 : wearing_golden + wearing_black = 55)
variables (cond3 : truth_swordsmen_black + lie_swordsmen_golden + lie_archers_black + truth_archers_golden = 55)
variables (cond4 : wearing_golden = 44)
variables (cond5 : archers = 33)
variables (cond6 : truth_swordsmen_black + lie_archers_black = 22)

-- Define the mathematic equivalent proof problem
theorem archers_in_golden : archers = 22 :=
by
  sorry

end archers_in_golden_l123_123935


namespace statement_C_l123_123297

theorem statement_C (x : ℝ) (h : x^2 < 4) : x < 2 := 
sorry

end statement_C_l123_123297


namespace common_point_condition_l123_123843

noncomputable def circle := sorry -- placeholder definition
variables (k k₁ k₂ k₃: circle)
variables (K₁ K₂ K₃: (euclidean_space ℝ (fin 2)) -- Centers of the circles

-- Given conditions
axiom centers_on_k : K₁ ∈ k ∧ K₂ ∈ k ∧ K₃ ∈ k
axiom intersection_points_on_k : 
  ∃ M₁₂ M₂₃ M₃₁ : euclidean_space ℝ (fin 2),
  (M₁₂ ∈ k) ∧ (M₂₃ ∈ k) ∧ (M₃₁ ∈ k) ∧
  (M₁₂ ∈ k₁) ∧ (M₁₂ ∈ k₂) ∧
  (M₂₃ ∈ k₂) ∧ (M₂₃ ∈ k₃) ∧
  (M₃₁ ∈ k₃) ∧ (M₃₁ ∈ k₁)

-- The theorem to prove
theorem common_point_condition: ∃ P : euclidean_space ℝ (fin 2),
  (P ∈ k₁) ∧ (P ∈ k₂) ∧ (P ∈ k₃) ↔
  (P ∈ orthocenter (triangle K₁ K₂ K₃)) ∨ 
  (∃ Q : euclidean_space ℝ (fin 2), Q ∈ k ∧ Q ∈ k₁ ∧ Q ∈ k₂ ∧ Q ∈ k₃) := 
sorry

end common_point_condition_l123_123843


namespace hyperbola_eccentricity_l123_123594

theorem hyperbola_eccentricity (m : ℝ) (h : 0 < m) :
  ∃ e, e = Real.sqrt (1 + m) ∧ e > Real.sqrt 2 → m > 1 :=
by
  sorry

end hyperbola_eccentricity_l123_123594


namespace fraction_simplification_l123_123245

theorem fraction_simplification :
  (3^2 * 3^(-4)) / (2^3 * 2^(-5)) = (4 / 9) := by
  sorry

end fraction_simplification_l123_123245


namespace product_of_p_q_l123_123047

noncomputable def p : ℝ := -2
noncomputable def q : ℝ := 2

theorem product_of_p_q (h : (1 - complex.i : ℂ) * (1 + complex.i) = 2) : p * q = -4 := 
by 
  -- proof omitted
  sorry

end product_of_p_q_l123_123047


namespace smallest_positive_four_digit_multiple_of_18_l123_123767

-- Define the predicates for conditions
def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def multiple_of_18 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 18 * k

-- Define the main theorem
theorem smallest_positive_four_digit_multiple_of_18 : 
  ∃ n : ℕ, four_digit_number n ∧ multiple_of_18 n ∧ ∀ m : ℕ, four_digit_number m ∧ multiple_of_18 m → n ≤ m :=
begin
  use 1008,
  split,
  { -- proof that 1008 is a four-digit number
    split,
    { linarith, },
    { linarith, }
  },

  split,
  { -- proof that 1008 is a multiple of 18
    use 56,
    norm_num,
  },

  { -- proof that 1008 is the smallest such number
    intros m h1 h2,
    have h3 := Nat.le_of_lt,
    sorry, -- Detailed proof would go here
  }
end

end smallest_positive_four_digit_multiple_of_18_l123_123767


namespace quotient_of_division_l123_123292

theorem quotient_of_division (dividend divisor remainder quotient : ℕ) 
  (h1 : dividend = 15968) 
  (h2 : divisor = 179) 
  (h3 : remainder = 37) 
  (h4 : quotient = 89) : 
  dividend = (divisor * quotient) + remainder := by 
  -- Given conditions
  have h1 : 15968 = (179 * 89) + 37, from sorry,
  -- Hence proved
  exact h1

end quotient_of_division_l123_123292


namespace red_balls_number_l123_123491

namespace BallDrawing

variable (x : ℕ) -- define x as the number of red balls

noncomputable def total_balls : ℕ := x + 4
noncomputable def yellow_ball_probability : ℚ := 4 / total_balls x

theorem red_balls_number : yellow_ball_probability x = 0.2 → x = 16 :=
by
  unfold yellow_ball_probability
  sorry

end BallDrawing

end red_balls_number_l123_123491


namespace goods_train_length_l123_123685

def speed_mans_train : ℝ := 100 -- Speed of man's train in kmph
def speed_goods_train : ℝ := 12 -- Speed of goods train in kmph
def time_to_pass : ℝ := 9 -- Time taken for freight train to pass man in seconds

noncomputable def relative_speed : ℝ := (speed_mans_train + speed_goods_train) * (1000 / 3600)
noncomputable def length_of_goods_train : ℝ := relative_speed * time_to_pass

theorem goods_train_length :
  length_of_goods_train ≈ 280 := sorry

end goods_train_length_l123_123685


namespace example_1_example_2_l123_123827

noncomputable def f (x : ℝ) := 3^x

theorem example_1 (x y : ℝ) : f(x) * f(y) = f(x + y) :=
by 
  -- proof goes here
  sorry

theorem example_2 (x y : ℝ) : f(x) / f(y) = f(x - y) :=
by 
  -- proof goes here
  sorry

end example_1_example_2_l123_123827


namespace intersection_A_B_l123_123444

def setA : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.log(2 - x^2) / Real.log 2}
def setB : Set ℝ := {x : ℝ | x^2 - x - 2 ≤ 0}

theorem intersection_A_B :
  (setA ∩ setB) = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := 
by
  -- Proof omitted
  sorry

end intersection_A_B_l123_123444


namespace smallest_four_digit_multiple_of_18_l123_123752

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 18 ∣ n ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ 18 ∣ m → n ≤ m := by
  use 1008
  sorry

end smallest_four_digit_multiple_of_18_l123_123752


namespace angle_triple_supplement_l123_123268

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l123_123268


namespace travel_time_and_speed_l123_123875

theorem travel_time_and_speed :
  (total_time : ℝ) = 5.5 →
  (bus_whole_journey : ℝ) = 1 →
  (bus_half_journey : ℝ) = bus_whole_journey / 2 →
  (walk_half_journey : ℝ) = total_time - bus_half_journey →
  (walk_whole_journey : ℝ) = 2 * walk_half_journey →
  (bus_speed_factor : ℝ) = walk_whole_journey / bus_whole_journey →
  walk_whole_journey = 10 ∧ bus_speed_factor = 10 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end travel_time_and_speed_l123_123875


namespace archers_in_golden_l123_123939

variables (soldiers archers swordsmen wearing_golden wearing_black : ℕ)
variables (truth_swordsmen_black lie_swordsmen_golden lie_archers_black truth_archers_golden : ℕ)

-- Given conditions
variables (cond1 : archers + swordsmen = 55)
variables (cond2 : wearing_golden + wearing_black = 55)
variables (cond3 : truth_swordsmen_black + lie_swordsmen_golden + lie_archers_black + truth_archers_golden = 55)
variables (cond4 : wearing_golden = 44)
variables (cond5 : archers = 33)
variables (cond6 : truth_swordsmen_black + lie_archers_black = 22)

-- Define the mathematic equivalent proof problem
theorem archers_in_golden : archers = 22 :=
by
  sorry

end archers_in_golden_l123_123939


namespace lambda_value_l123_123771

noncomputable def omega : ℂ := sorry
noncomputable def lambda := sorry

axiom omega_norm_eq_3 : abs omega = 3
axiom w_lam_equilateral : ∃ (λ : ℝ), λ > 1 ∧ 
  (∃ (θ : ℝ), (ω = 3 * complex.exp (θ * complex.I)) ∧ 
   (λ - 1)^2 = 12)

theorem lambda_value : lambda = 1 + 2 * real.sqrt 3 := by
  have h1 : ∃ (λ : ℝ), λ > 1 ∧ λ - 1 = 2 * real.sqrt 3 :=
    by sorry
  sorry

end lambda_value_l123_123771


namespace base_5_to_base_7_l123_123373

theorem base_5_to_base_7 :
  ∀ (n : ℕ), base_5_to_dec 412 = dec_to_base_7 n → n = 212 :=
by
  sorry

-- Auxiliary definitions
def base_5_to_dec (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := n / 100
  d0 * 5^0 + d1 * 5^1 + d2 * 5^2

def dec_to_base_7 (n : ℕ) : ℕ :=
  let d0 := n % 7
  let d1 := (n / 7) % 7
  let d2 := n / 49
  d2 * 100 + d1 * 10 + d0

end base_5_to_base_7_l123_123373


namespace town_population_original_l123_123348

noncomputable def original_population (n : ℕ) : Prop :=
  let increased_population := n + 1500
  let decreased_population := (85 / 100 : ℚ) * increased_population
  decreased_population = n + 1455

theorem town_population_original : ∃ n : ℕ, original_population n ∧ n = 1200 :=
by
  sorry

end town_population_original_l123_123348


namespace mod_mult_congruence_l123_123189

theorem mod_mult_congruence (n : ℤ) (h1 : 215 ≡ 65 [ZMOD 75])
  (h2 : 789 ≡ 39 [ZMOD 75]) (h3 : 215 * 789 ≡ n [ZMOD 75]) (hn : 0 ≤ n ∧ n < 75) :
  n = 60 :=
by
  sorry

end mod_mult_congruence_l123_123189


namespace probability_acute_angle_AMB_l123_123490

noncomputable def square_side_length := 2

structure Square :=
(A B C D : ℝ × ℝ)
(side_length : ℝ)
(valid_square : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)

structure Point :=
(x y : ℝ)

def inside_square (P : Point) (S : Square) : Prop :=
S.A.1 ≤ P.x ∧ P.x ≤ S.C.1 ∧ S.A.2 ≤ P.y ∧ P.y ≤ S.C.2

def acute_angle_condition (P : Point) (S : Square) (θ : ℝ) : Prop :=
θ < π / 2

theorem probability_acute_angle_AMB (S : Square) (M : Point) (h_in_square : inside_square M S) :
  (1 - π / 8) = sorry :=
sorry

end probability_acute_angle_AMB_l123_123490


namespace range_h_l123_123715

def h (t : ℝ) : ℝ := (t^2 + (5 / 4) * t) / (t^2 + 2)

theorem range_h : set.range h = set.Icc 0 (128 / 103) :=
sorry

end range_h_l123_123715


namespace max_happy_monkeys_l123_123337

/-- Conditions: Given 20 pears, 30 bananas, 40 peaches, and 50 tangerines -/
def n_pears : ℕ := 20
def n_bananas : ℕ := 30
def n_peaches : ℕ := 40
def n_tangerines : ℕ := 50
def total_fruits_except_tangerines := n_pears + n_bananas + n_peaches

/-- A happy monkey consumes three different fruits including one tangerine -/
def happy_monkey (n : ℕ) := n * 3

/-- Theorem: Maximum number of happy monkeys -/
theorem max_happy_monkeys : total_fruits_except_tangerines / 2 = 45 :=
by
  have := total_fruits_except_tangerines
  calc
    20 + 30 + 40 = 90 := by simp
    90 / 2 = 45 := by norm_num
  sorry

end max_happy_monkeys_l123_123337


namespace trigonometric_identity_l123_123397

open Real

theorem trigonometric_identity (α β : ℝ) (h : 2 * cos (2 * α + β) - 3 * cos β = 0) :
  tan α * tan (α + β) = -1 / 5 := 
by {
  sorry
}

end trigonometric_identity_l123_123397


namespace three_pairwise_parallel_not_coplanar_lines_divide_space_l123_123035

-- Define the types and conditions
variable (a b c : Line)
variable (pairwise_parallel : Parallel a b ∧ Parallel b c ∧ Parallel c a)
variable (not_coplanar : ¬Coplanar a b c)

-- Statement of the theorem
theorem three_pairwise_parallel_not_coplanar_lines_divide_space (a b c : Line)
  (pairwise_parallel : Parallel a b ∧ Parallel b c ∧ Parallel c a)
  (not_coplanar : ¬Coplanar a b c) :
  determines_planes a b c 3 ∧ divides_space a b c 7 :=
  sorry

end three_pairwise_parallel_not_coplanar_lines_divide_space_l123_123035


namespace no_obtuse_triangle_probability_l123_123798

noncomputable def probability_no_obtuse_triangle : ℝ :=
 let θ := 1/2 in
 let prob_A2_A3_given_A0A1 := (3/8) * (3/8) in
 θ * prob_A2_A3_given_A0A1

theorem no_obtuse_triangle_probability :
  probability_no_obtuse_triangle = 9/128 :=
by
  sorry

end no_obtuse_triangle_probability_l123_123798


namespace number_of_special_matrices_l123_123534

-- Definitions from the given conditions
variables {F : Type*} [field F] (p : ℕ) [fintype F] [fintype (fin p)]
def matrix_2x2 := matrix (fin 2) (fin 2) F
def trace_eq_one (A : matrix_2x2 F) : Prop := A 0 0 + A 1 1 = 1
def det_eq_zero (A : matrix_2x2 F) : Prop := A 0 0 * A 1 1 - A 0 1 * A 1 0 = 0
def S : set (matrix_2x2 F) := {A | trace_eq_one A ∧ det_eq_zero A}

-- The proof statement
theorem number_of_special_matrices : S.card = p^2 + p :=
sorry

end number_of_special_matrices_l123_123534


namespace marble_ratio_correct_l123_123706

-- Necessary given conditions
variables (x : ℕ) (Ben_initial John_initial : ℕ) (John_post Ben_post : ℕ)
variables (h1 : Ben_initial = 18)
variables (h2 : John_initial = 17)
variables (h3 : Ben_post = Ben_initial - x)
variables (h4 : John_post = John_initial + x)
variables (h5 : John_post = Ben_post + 17)

-- Define the ratio of the number of marbles Ben gave to John to the number of marbles Ben had initially
def marble_ratio := (x : ℕ) / Ben_initial

-- The theorem we want to prove
theorem marble_ratio_correct (h1 : Ben_initial = 18) (h2 : John_initial = 17) (h3 : Ben_post = Ben_initial - x)
(h4 : John_post = John_initial + x) (h5 : John_post = Ben_post + 17) : marble_ratio x Ben_initial = 1/2 := by 
  sorry

end marble_ratio_correct_l123_123706


namespace number_of_dolls_l123_123224

theorem number_of_dolls (total_toys : ℕ) (fraction_action_figures : ℚ) 
  (remaining_fraction_action_figures : fraction_action_figures = 1 / 4) 
  (remaining_fraction_dolls : 1 - fraction_action_figures = 3 / 4) 
  (total_toys_eq : total_toys = 24) : 
  (total_toys - total_toys * fraction_action_figures) = 18 := 
by 
  sorry

end number_of_dolls_l123_123224


namespace no_obtuse_triangle_probability_eq_l123_123781

noncomputable def probability_no_obtuse_triangle : ℝ :=
  let num_points := 4
  -- Condition (1): Four points are chosen uniformly at random on a circle.
  -- Condition (2): An obtuse angle occurs if the minor arc exceeds π/2.
  9 / 64

theorem no_obtuse_triangle_probability_eq :
  let num_points := 4
  ∀ (points : Fin num_points → ℝ), 
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∠ (points i, points j, points k) < π / 2) →
  probability_no_obtuse_triangle = 9 / 64 :=
by
  sorry

end no_obtuse_triangle_probability_eq_l123_123781


namespace flour_per_new_base_is_one_fifth_l123_123374

def total_flour : ℚ := 40 * (1 / 8)

def flour_per_new_base (p : ℚ) (total_flour : ℚ) : ℚ := total_flour / p

theorem flour_per_new_base_is_one_fifth :
  flour_per_new_base 25 total_flour = 1 / 5 :=
by
  sorry

end flour_per_new_base_is_one_fifth_l123_123374


namespace loaves_count_l123_123694

theorem loaves_count 
  (init_loaves : ℕ)
  (sold_percent : ℕ) 
  (bulk_purchase : ℕ)
  (bulk_discount_percent : ℕ)
  (evening_purchase : ℕ)
  (evening_discount_percent : ℕ)
  (final_loaves : ℕ)
  (h1 : init_loaves = 2355)
  (h2 : sold_percent = 30)
  (h3 : bulk_purchase = 750)
  (h4 : bulk_discount_percent = 20)
  (h5 : evening_purchase = 489)
  (h6 : evening_discount_percent = 15)
  (h7 : final_loaves = 2888) :
  let mid_morning_sold := init_loaves * sold_percent / 100
  let loaves_after_sale := init_loaves - mid_morning_sold
  let bulk_discount_loaves := bulk_purchase * bulk_discount_percent / 100
  let loaves_after_bulk_purchase := loaves_after_sale + bulk_purchase
  let evening_discount_loaves := evening_purchase * evening_discount_percent / 100
  let loaves_after_evening_purchase := loaves_after_bulk_purchase + evening_purchase
  loaves_after_evening_purchase = final_loaves :=
by
  sorry

end loaves_count_l123_123694


namespace print_shop_cost_comparisons_l123_123392

theorem print_shop_cost_comparisons :
  let cost_X := 1.25 * 60,
      cost_Y := 2.75 * 60,
      cost_Z := 3.00 * 60 * 0.9, -- After 10% discount
      cost_W := 2.00 * 60 - 5
  in cost_X = 75 ∧
     cost_Y - cost_X = 90 ∧
     cost_Z - cost_X = 87 ∧
     cost_W - cost_X = 40 :=
by {
  -- Definitions
  let cost_X := 1.25 * 60
  let cost_Y := 2.75 * 60
  let cost_Z := 3.00 * 60 * 0.9 -- After 10% discount
  let cost_W := 2.00 * 60 - 5
  
  -- Proofs
  have h1 : cost_X = 75, by norm_num [cost_X]
  have h2 : cost_Y - cost_X = 90, by norm_num [cost_Y, cost_X]
  have h3 : cost_Z - cost_X = 87, by norm_num [cost_Z, cost_X]
  have h4 : cost_W - cost_X = 40, by norm_num [cost_W, cost_X]
  
  exact ⟨h1, h2, h3, h4⟩
}

end print_shop_cost_comparisons_l123_123392


namespace smallest_four_digit_multiple_of_18_l123_123753

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 18 ∣ n ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ 18 ∣ m → n ≤ m := by
  use 1008
  sorry

end smallest_four_digit_multiple_of_18_l123_123753


namespace sum_of_primes_20_to_40_l123_123640

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ (2 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime |>.sum

theorem sum_of_primes_20_to_40 : sum_of_primes_between 20 40 = 120 := by
  sorry

end sum_of_primes_20_to_40_l123_123640


namespace shopkeeper_loss_l123_123689

-- Definition of the problem conditions
def percentage_loss (X : ℝ) : ℝ :=
  let intended_sale_price := 1.10 * X
  let remaining_goods_value := 0.40 * X
  let sold_goods_value := remaining_goods_value * 1.10
  let loss := X - sold_goods_value
  (loss / X) * 100

theorem shopkeeper_loss (X : ℝ) (h_pos: X > 0) : 
  percentage_loss X = 56 := 
by
  -- Proof to be filled in later
  sorry

end shopkeeper_loss_l123_123689


namespace derivative_of_f_eval_deriv_at_pi_over_6_l123_123829

noncomputable def f (x : Real) : Real := (Real.sin x) ^ 4 + (Real.cos x) ^ 4

theorem derivative_of_f : ∀ x, deriv f x = -Real.sin (4 * x) :=
by
  intro x
  sorry

theorem eval_deriv_at_pi_over_6 : deriv f (Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  rw [derivative_of_f]
  sorry

end derivative_of_f_eval_deriv_at_pi_over_6_l123_123829


namespace impossibility_of_forming_equation_l123_123654

theorem impossibility_of_forming_equation :
  ¬ ∃ (a b c d e f g : ℤ) (ops : List (ℤ → ℤ → ℤ)) (eq_op : ℤ → ℤ → Prop),
    let expr := [a, b, c, d, e, f, g].zip_with ops (λ x y op, op x y)
    eq_op (list.sum expr) _ := sorry

end impossibility_of_forming_equation_l123_123654


namespace no_obtuse_triangle_l123_123790

-- Conditions
def points_on_circle_uniformly_at_random (n : ℕ) : Prop :=
  ∀ i < n, ∀ j < n, i ≠ j → ∃ θ_ij : ℝ, 0 ≤ θ_ij ∧ θ_ij ≤ π

-- Theorem statement
theorem no_obtuse_triangle (hn : points_on_circle_uniformly_at_random 4) :
  let p := \left(\frac{1}{2}\right)^6 in
  p = \frac{1}{64} :=
sorry

end no_obtuse_triangle_l123_123790


namespace seeds_per_plant_l123_123993

theorem seeds_per_plant :
  let trees := 2
  let plants_per_tree := 20
  let total_plants := trees * plants_per_tree
  let planted_trees := 24
  let planting_fraction := 0.60
  exists S : ℝ, planting_fraction * (total_plants * S) = planted_trees ∧ S = 1 :=
by
  sorry

end seeds_per_plant_l123_123993


namespace min_groups_required_l123_123676

-- Define the conditions
def total_children : ℕ := 30
def max_children_per_group : ℕ := 12
def largest_divisor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ d ≤ max_children_per_group

-- Define the property that we are interested in: the minimum number of groups required
def min_num_groups (total : ℕ) (group_size : ℕ) : ℕ := total / group_size

-- Prove the minimum number of groups is 3 given the conditions
theorem min_groups_required : ∃ d, largest_divisor total_children d ∧ min_num_groups total_children d = 3 :=
sorry

end min_groups_required_l123_123676


namespace find_f_2017_l123_123834

noncomputable def f : ℝ → ℝ :=
sorry

axiom cond1 : ∀ x : ℝ, f (1 + x) + f (1 - x) = 0
axiom cond2 : ∀ x : ℝ, f (-x) = f x
axiom cond3 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = 2^x - 1

theorem find_f_2017 : f 2017 = 1 :=
by
  sorry

end find_f_2017_l123_123834


namespace problem_a_b_c_d_l123_123116

open Real

/-- The main theorem to be proved -/
theorem problem_a_b_c_d
  (a b c d : ℝ)
  (hab : 0 < a) (hcd : 0 < c) (hab' : 0 < b) (hcd' : 0 < d)
  (h1 : a > c) (h2 : b < d)
  (h3 : a + sqrt b ≥ c + sqrt d)
  (h4 : sqrt a + b ≤ sqrt c + d) :
  a + b + c + d > 1 :=
by
  sorry

end problem_a_b_c_d_l123_123116


namespace smallest_positive_period_monotonically_decreasing_interval_max_min_values_in_interval_l123_123871

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x + 2 * cos x ^ 2

-- Prove the smallest positive period of f(x) is π
theorem smallest_positive_period {x : ℝ} : ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = π := sorry

-- Prove the monotonically decreasing interval of f(x)
theorem monotonically_decreasing_interval {k : ℤ} :
  is_monotonically_decreasing_interval (λ x, [(k * π + π / 6), (2 * π / 3 + k * π)]) f := sorry

-- Prove the maximum and minimum values of f(x) in [0, π/2]
theorem max_min_values_in_interval : 
  ∃ x_max x_min, max_value_in_interval 0 (π / 2) f = 3 ∧ min_value_in_interval 0 (π / 2) f = 0 := sorry

end smallest_positive_period_monotonically_decreasing_interval_max_min_values_in_interval_l123_123871


namespace John_large_bottles_count_l123_123106

/-- Given the conditions:
  - price per large bottle: $1.89
  - price per small bottle: $1.38
  - number of small bottles: 750
  - approximate average price paid per bottle: $1.7057
  Prove that the number of large bottles purchased is approximately 1323.
-/
theorem John_large_bottles_count :
  ∃ (L : ℤ), (1.89 * L + 1.38 * 750) / (L + 750) = 1.7057 ∧ L ≈ 1323 :=
by 
  use 1323
  sorry

end John_large_bottles_count_l123_123106


namespace king_removed_no_empty_seat_l123_123108

/-- If Lady Guinivera calls the King off the table before midnight, and given the initial conditions:
  1. There are 100 seats around the table.
  2. Each person has a glass with either white or red wine.
  3. At least one glass with red wine and one glass with white wine.
  4. At midnight:
     - White wine glasses are moved to the left neighbor.
     - Red wine glasses are moved to the right neighbor.
  Prove that removing the King does not leave any seat without a glass of wine after midnight. -/
theorem king_removed_no_empty_seat :
  ∀ seats : ℕ, seats = 100 →
  ∃ (glasses : ℕ → bool), -- bool:: tt represents red wine, ff represents white wine
  (∃ i, i < 100 ∧ glasses i = tt) ∧ (∃ j, j < 100 ∧ glasses j = ff) →
  (∀ i : ℕ, i < 100 → 
    (glasses i = tt ∨ glasses i = ff) →
    glasses ((i + if glasses i = tt then 1 else (100 - 1)) % 100) = glasses i) →
  (∀ i : ℕ, i < 99 → glasses ((i + 1) % 99) = glasses ((i % 99) + 1)) →
  ∀ i : ℕ, i < 99 + 1 → glasses i ≠ none :=
begin
  sorry
end

end king_removed_no_empty_seat_l123_123108


namespace smallest_non_factor_product_of_factors_of_60_l123_123619

theorem smallest_non_factor_product_of_factors_of_60 :
  ∃ x y : ℕ, x ≠ y ∧ x ∣ 60 ∧ y ∣ 60 ∧ ¬ (x * y ∣ 60) ∧ ∀ x' y' : ℕ, x' ≠ y' → x' ∣ 60 → y' ∣ 60 → ¬(x' * y' ∣ 60) → x * y ≤ x' * y' := 
sorry

end smallest_non_factor_product_of_factors_of_60_l123_123619


namespace legs_heads_difference_l123_123483

variables (D C L H : ℕ)

theorem legs_heads_difference
    (hC : C = 18)
    (hL : L = 2 * D + 4 * C)
    (hH : H = D + C) :
    L - 2 * H = 36 :=
by
  have h1 : C = 18 := hC
  have h2 : L = 2 * D + 4 * C := hL
  have h3 : H = D + C := hH
  sorry

end legs_heads_difference_l123_123483


namespace triangles_in_divided_square_l123_123321

theorem triangles_in_divided_square (V E F : ℕ) 
  (hV : V = 24) 
  (h1 : 3 * F + 1 = 2 * E) 
  (h2 : V - E + F = 2) : F = 43 ∧ (F - 1 = 42) := 
by 
  have hF : F = 43 := sorry
  have hTriangles : F - 1 = 42 := sorry
  exact ⟨hF, hTriangles⟩

end triangles_in_divided_square_l123_123321


namespace P_lt_Q_l123_123888

variable (a : ℝ) (ha : 0 ≤ a)
def P : ℝ := Real.sqrt a + Real.sqrt (a + 7)
def Q : ℝ := Real.sqrt (a + 3) + Real.sqrt (a + 4)

theorem P_lt_Q : P a < Q a :=
by
  sorry

end P_lt_Q_l123_123888


namespace shortest_altitude_of_right_triangle_l123_123218

-- Define the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Given conditions about the triangle
def right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the area of the triangle
def area (a b : ℕ) : ℝ := (1/2) * a * b

-- Define the altitude
noncomputable def altitude (area : ℝ) (c : ℕ) : ℝ := (2 * area) / c

-- Proving the length of the shortest altitude
theorem shortest_altitude_of_right_triangle 
  (h : ℝ) 
  (ha : a = 9) 
  (hb : b = 12) 
  (hc : c = 15) 
  (rt : right_triangle a b c) : 
  altitude (area a b) c = 7.2 :=
sorry

end shortest_altitude_of_right_triangle_l123_123218


namespace equation_has_three_solutions_l123_123887

theorem equation_has_three_solutions :
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x, x ∈ s ↔ x^2 * (x - 1) * (x - 2) = 0 := 
by
  sorry

end equation_has_three_solutions_l123_123887


namespace intersection_of_asymptotes_l123_123743

theorem intersection_of_asymptotes :
  let f := λ x : ℝ, (x^2 - 6*x + 8) / (x^2 - 6*x + 9)
  ∃ x y : ℝ, (x = 3 ∧ y = 1) ∧ (∀ ε > 0, ∀ δ > 0, (∀ x' : ℝ, (0 < |x' - x| ∧ |x' - x| < δ) → |f x' - y| > ε))
:= sorry

end intersection_of_asymptotes_l123_123743


namespace math_problem_l123_123012

def prop1 (x : ℝ) : Prop := 
  if x > 1 then x > 2 else True

def prop2 (α : ℝ) : Prop := 
  ¬ (sin α = 1/2) → α ≠ π/6

def prop3 (x y : ℝ) : Prop := 
  ¬ (x = 0 ∧ y = 0) → ¬ (x * y = 0)

def prop4_exists (x_0 : ℝ) : Prop :=
  x_0^2 - x_0 + 1 ≤ 0

def prop4 : Prop := 
  ¬ ∃ x_0 : ℝ, prop4_exists x_0

theorem math_problem : 
  (¬ prop1 1) ∧ prop2 (π/6) ∧ (¬ prop3 0 1) ∧ prop4 :=
by
  sorry

end math_problem_l123_123012


namespace greatest_possible_n_l123_123054

theorem greatest_possible_n (n : ℤ) (h1 : 102 * n^2 ≤ 8100) : n ≤ 8 :=
sorry

end greatest_possible_n_l123_123054


namespace cube_vertices_probability_l123_123825

theorem cube_vertices_probability (totalVertices : ℕ) (selectedVertices : ℕ) 
   (totalCombinations : ℕ) (favorableOutcomes : ℕ) : 
   totalVertices = 8 ∧ selectedVertices = 4 ∧ totalCombinations = 70 ∧ favorableOutcomes = 12 → 
   (favorableOutcomes : ℚ) / totalCombinations = 6 / 35 := by
   sorry

end cube_vertices_probability_l123_123825


namespace ratio_of_roses_l123_123360

theorem ratio_of_roses (total_flowers tulips carnations roses : ℕ) 
  (h1 : total_flowers = 40) 
  (h2 : tulips = 10) 
  (h3 : carnations = 14) 
  (h4 : roses = total_flowers - (tulips + carnations)) :
  roses / total_flowers = 2 / 5 :=
by
  sorry

end ratio_of_roses_l123_123360


namespace game_lives_l123_123918

theorem game_lives (initial_lives first_level_lives second_level_lives : ℕ) :
  initial_lives = 2 → first_level_lives = 6 → second_level_lives = 11 →
  initial_lives + first_level_lives + second_level_lives = 19 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end game_lives_l123_123918


namespace triangle_area_percentage_of_pentagon_l123_123339

-- Condition 1: Let the leg length of the isosceles right triangle be 2.
def leg_length : ℝ := 2

-- Condition 2: The hypotenuse of the triangle
def hypotenuse_length : ℝ := leg_length * Real.sqrt 2

-- Condition 3: The dimensions of the rectangle are hypotenuse length and twice the leg length
def rect_width : ℝ := hypotenuse_length
def rect_height : ℝ := 2 * leg_length

-- Area calculations
def area_triangle : ℝ := (1 / 2) * leg_length * leg_length
def area_rectangle : ℝ := rect_width * rect_height
def area_pentagon : ℝ := area_triangle + area_rectangle

-- Wanted percentage
def desired_percentage : ℝ := 19.31

theorem triangle_area_percentage_of_pentagon :
  (area_triangle / area_pentagon) * 100 = desired_percentage := by
  sorry

end triangle_area_percentage_of_pentagon_l123_123339


namespace no_obtuse_triangle_probability_eq_l123_123783

noncomputable def probability_no_obtuse_triangle : ℝ :=
  let num_points := 4
  -- Condition (1): Four points are chosen uniformly at random on a circle.
  -- Condition (2): An obtuse angle occurs if the minor arc exceeds π/2.
  9 / 64

theorem no_obtuse_triangle_probability_eq :
  let num_points := 4
  ∀ (points : Fin num_points → ℝ), 
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∠ (points i, points j, points k) < π / 2) →
  probability_no_obtuse_triangle = 9 / 64 :=
by
  sorry

end no_obtuse_triangle_probability_eq_l123_123783


namespace compute_r_value_l123_123126

theorem compute_r_value :
  ∀ (r t n : ℕ),
  r = 4^t - t^2 →
  t = 3^n + n^2 →
  n = 3 →
  r = 2^72 - 1296 :=
by
  intros r t n hr ht hn
  rw [hn] at ht
  simp at ht
  rw [ht] at hr
  have ht36 : t = 36 := by
    have : 3^3 = 27 := rfl
    have : 3^2 = 9 := rfl
    simp only [nat.add_sub_assoc this] at ht
  rw [ht36] at hr
  norm_num at hr
  exact hr

end compute_r_value_l123_123126


namespace number_of_archers_in_golden_armor_l123_123968

-- Define the problem context
structure Soldier where
  is_archer : Bool
  wears_golden_armor : Bool

def truth_teller (s : Soldier) (is_black_armor : Bool) : Bool :=
  if s.is_archer then s.wears_golden_armor = is_black_armor
  else s.wears_golden_armor ≠ is_black_armor

def response (s : Soldier) (question : String) (is_black_armor : Bool) : Bool :=
  match question with
  | "Are you wearing golden armor?" => if truth_teller s is_black_armor then s.wears_golden_armor else ¬s.wears_golden_armor
  | "Are you an archer?" => if truth_teller s is_black_armor then s.is_archer else ¬s.is_archer
  | "Is today Monday?" => if truth_teller s is_black_armor then True else False -- An assumption that today not being Monday means False
  | _ => False

-- Problem condition setup
def total_soldiers : Nat := 55
def golden_armor_yes : Nat := 44
def archer_yes : Nat := 33
def monday_yes : Nat := 22

-- Define the main theorem
theorem number_of_archers_in_golden_armor :
  ∃ l : List Soldier,
    l.length = total_soldiers ∧
    l.countp (λ s => response s "Are you wearing golden armor?" True) = golden_armor_yes ∧
    l.countp (λ s => response s "Are you an archer?" True) = archer_yes ∧
    l.countp (λ s => response s "Is today Monday?" True) = monday_yes ∧
    l.countp (λ s => s.is_archer ∧ s.wears_golden_armor) = 22 :=
sorry

end number_of_archers_in_golden_armor_l123_123968


namespace probability_no_obtuse_triangle_is_9_over_64_l123_123809

noncomputable def probability_no_obtuse_triangle (A0 A1 A2 A3 : ℝ) : ℝ :=
  -- Define the probabilistic model according to the problem conditions
  -- Assuming A0, A1, A2, and A3 are positions of points on the circle parametrized by angles in radians
  let θ := real.angle.lower_half (A1 - A0) in
  let prob_A2 := (π - θ) / (2 * π) in
  let prob_A3 := (π - θ) / (2 * π) in
  (1 / 2) * (prob_A2 * prob_A3)

theorem probability_no_obtuse_triangle_is_9_over_64 :
  probability_no_obtuse_triangle A0 A1 A2 A3 = 9 / 64 :=
by sorry

end probability_no_obtuse_triangle_is_9_over_64_l123_123809


namespace triple_supplementary_angle_l123_123276

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l123_123276


namespace probability_no_obtuse_triangle_l123_123803

namespace CirclePoints

noncomputable def no_obtuse_triangle_probability : ℝ := 
  let p := 1/64 in
    p

theorem probability_no_obtuse_triangle (X : ℕ → ℝ) (hcirc : ∀ n, 0 ≤ X n ∧ X n < 2 * π) (hpoints : (∀ n m, n ≠ m → X n ≠ X m)) :
  no_obtuse_triangle_probability = 1/64 :=
sorry

end CirclePoints

end probability_no_obtuse_triangle_l123_123803


namespace cube_vertices_probability_l123_123824

theorem cube_vertices_probability (totalVertices : ℕ) (selectedVertices : ℕ) 
   (totalCombinations : ℕ) (favorableOutcomes : ℕ) : 
   totalVertices = 8 ∧ selectedVertices = 4 ∧ totalCombinations = 70 ∧ favorableOutcomes = 12 → 
   (favorableOutcomes : ℚ) / totalCombinations = 6 / 35 := by
   sorry

end cube_vertices_probability_l123_123824


namespace largest_m_for_divisibility_property_l123_123441

theorem largest_m_for_divisibility_property : ∃ m, (∀ (S : Finset ℕ), S.card = 1000 - m → ∃ a b ∈ S, a ∣ b ∨ b ∣ a) ∧ m = 499 :=
by
  sorry

end largest_m_for_divisibility_property_l123_123441


namespace point_of_intersection_of_asymptotes_l123_123748

noncomputable def asymptotes_intersection_point : ℝ × ℝ := (3, 1)

theorem point_of_intersection_of_asymptotes (x y : ℝ) (h : y = (x ^ 2 - 6 * x + 8) / (x ^ 2 - 6 * x + 9)) (hx : x = 3) (hy : y = 1) :
  (x, y) = (3, 1) :=
by
  rw [hx, hy]
  simp
  exact asymptotes_intersection_point
  sorry -- Proof steps are not provided, only statement is required.

end point_of_intersection_of_asymptotes_l123_123748


namespace a_2n_is_perfect_square_l123_123539

-- Definitions based on the conditions
def a : ℕ → ℕ
| 0     := 0
| 1     := 1
| 2     := 1
| 3     := 2
| 4     := 4
| (n+1) := if n > 4 then a n + a (n-2) + a (n-3) else 0

theorem a_2n_is_perfect_square (n : ℕ) (h : n ≥ 3) : ∃ k : ℕ, a (2 * n) = k * k :=
sorry

end a_2n_is_perfect_square_l123_123539


namespace area_triangle_ABD_twice_OABC_l123_123386

-- Definitions of points O, A, B, and C
def O := (0 : ℝ, 0 : ℝ)
def A := (3 : ℝ, 0 : ℝ)
def B := (0 : ℝ, 3 : ℝ)
def C := (3 : ℝ, 3 : ℝ)

-- Definition of the target point D
def D := (-9 : ℝ, 0 : ℝ)

-- Definition of the area of a triangle given coordinates
def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (Q.2 - P.2) * (R.1 - P.1))

-- The main theorem we want to prove
theorem area_triangle_ABD_twice_OABC :
  area_triangle A B D = 2 * (3 * 3) := by
  sorry

end area_triangle_ABD_twice_OABC_l123_123386


namespace village_household_count_l123_123480

theorem village_household_count
  (H : ℕ)
  (water_per_household_per_month : ℕ := 20)
  (total_water : ℕ := 2000)
  (duration_months : ℕ := 10)
  (total_consumption_condition : water_per_household_per_month * H * duration_months = total_water) :
  H = 10 :=
by
  sorry

end village_household_count_l123_123480


namespace industrial_lubricants_percentage_l123_123325

noncomputable def percentage_microphotonics : ℕ := 14
noncomputable def percentage_home_electronics : ℕ := 19
noncomputable def percentage_food_additives : ℕ := 10
noncomputable def percentage_gmo : ℕ := 24
noncomputable def total_percentage : ℕ := 100
noncomputable def percentage_basic_astrophysics : ℕ := 25

theorem industrial_lubricants_percentage :
  total_percentage - (percentage_microphotonics + percentage_home_electronics + 
  percentage_food_additives + percentage_gmo + percentage_basic_astrophysics) = 8 := 
sorry

end industrial_lubricants_percentage_l123_123325


namespace num_positive_integers_n_l123_123672

theorem num_positive_integers_n (n : ℕ) : 
  (∃ n, ( ∃ k : ℕ, n = 2015 * k^2 ∧ ∃ m, m^2 = 2015 * n) ∧ 
          (∃ k : ℕ, n = 2015 * k^2 ∧  ∃ l : ℕ, 2 * 2015 * k^2 = l * (1 + k^2)))
  →
  n = 5 := sorry

end num_positive_integers_n_l123_123672


namespace quilt_patch_cost_l123_123507

-- conditions
def quilt_length : ℕ := 16
def quilt_width : ℕ := 20
def patch_area : ℕ := 4
def first_ten_patch_cost : ℕ := 10
def subsequent_patch_cost : ℕ := 5

theorem quilt_patch_cost :
  let total_quilt_area := quilt_length * quilt_width,
      total_patches := total_quilt_area / patch_area,
      cost_first_ten := 10 * first_ten_patch_cost,
      remaining_patches := total_patches - 10,
      cost_remaining := remaining_patches * subsequent_patch_cost,
      total_cost := cost_first_ten + cost_remaining
  in total_cost = 450 :=
by
  sorry

end quilt_patch_cost_l123_123507


namespace archers_in_golden_armor_count_l123_123944

theorem archers_in_golden_armor_count : 
  ∃ (archers_golden : ℕ), 
    let total_soldiers := 55 in
    let golden_armor_affirmative := 44 in
    let archer_affirmative := 33 in
    let monday_affirmative := 22 in
    let number_archers_golden := archers_golden in
    (number_archers_golden = 22) ∧ 
    ∃ (archer : ℕ) (swordsman : ℕ) (golden : ℕ) (black : ℕ),
      archer + swordsman = total_soldiers ∧
      golden + black = total_soldiers ∧
      golden + black = total_soldiers ∧
      (swordsman * golden + archer * golden = golden_armor_affirmative) ∧
      (swordsman * archer + swordsman * golden = archer_affirmative) ∧
      ((!monday_affirmative ∧ swordsman * golden) + (!monday_affirmative ∧ archer * black) = monday_affirmative)
:= sorry

end archers_in_golden_armor_count_l123_123944


namespace sum_of_primes_20_to_40_l123_123639

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ (2 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime |>.sum

theorem sum_of_primes_20_to_40 : sum_of_primes_between 20 40 = 120 := by
  sorry

end sum_of_primes_20_to_40_l123_123639


namespace smallest_four_digit_multiple_of_18_l123_123759

-- Define the concept of a four-digit number
def four_digit (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

-- Define the concept of a multiple of 18
def multiple_of_18 (N : ℕ) : Prop := ∃ k : ℕ, N = 18 * k

-- Define the combined condition for N being a four-digit multiple of 18
def four_digit_multiple_of_18 (N : ℕ) : Prop := four_digit N ∧ multiple_of_18 N

-- State that 1008 is the smallest such number
theorem smallest_four_digit_multiple_of_18 : ∀ N : ℕ, four_digit_multiple_of_18 N → 1008 ≤ N := 
by
  intros N H
  sorry

end smallest_four_digit_multiple_of_18_l123_123759


namespace ceil_sqrt_224_eq_15_l123_123732

theorem ceil_sqrt_224_eq_15 : Real.ceil (Real.sqrt 224) = 15 :=
by
  sorry

end ceil_sqrt_224_eq_15_l123_123732


namespace same_oxidation_state_HNO3_N2O5_l123_123094

def oxidation_state_HNO3 (H O: Int) : Int := 1 + 1 + (3 * (-2))
def oxidation_state_N2O5 (H O: Int) : Int := (2 * 1) + (5 * (-2))
def oxidation_state_substances_equal : Prop :=
  oxidation_state_HNO3 1 (-2) = oxidation_state_N2O5 1 (-2)

theorem same_oxidation_state_HNO3_N2O5 : oxidation_state_substances_equal :=
  by
  sorry

end same_oxidation_state_HNO3_N2O5_l123_123094


namespace problem_statement_l123_123701

theorem problem_statement (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) :=
  sorry

end problem_statement_l123_123701


namespace sum_S10_correct_l123_123404

noncomputable def sequence_a : ℕ → ℕ
| 0 := 0
| (n + 1) := if n % 2 = 0 then (sequence_a n) + (n / 2 + 1)
             else (sequence_a n) + (n / 2 + 1)

def sequence_b (n : ℕ) : ℚ :=
(2 * n + 1)^2 / (sequence_a (2 * n + 1))

def S_total (n : ℕ) : ℚ :=
∑ i in Finset.range n, sequence_b (i + 1)

theorem sum_S10_correct :
  S_total 10 = 450 / 11 :=
sorry

end sum_S10_correct_l123_123404


namespace max_value_of_f_product_of_zeros_l123_123020

variables {a b x x1 x2 : ℝ}

-- Define the function
def f (x : ℝ) : ℝ := Real.log x - a * x + b

-- Conditions
-- f(x) has two distinct zeros x1 and x2
axiom h1 : f(x1) = 0
axiom h2 : f(x2) = 0
axiom h3 : x1 ≠ x2
axiom h4 : 0 < a

-- Proof of 1. Finding the maximum value of f(x)
theorem max_value_of_f :
  ∃ (x : ℝ), (∀ y, f y ≤ f x) ∧ f x = -Real.log a - 1 + b :=
sorry

-- Proof of 2. Proving x1 * x2 < 1/a^2
theorem product_of_zeros :
  x1 * x2 < 1 / a^2 :=
sorry

end max_value_of_f_product_of_zeros_l123_123020


namespace matrix_addition_correct_l123_123751

def matrixA : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then 4 else -2
  else
    if j = 0 then -3 else 5

def matrixB : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then -6 else 0
  else
    if j = 0 then 7 else -8

def resultMatrix : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then -2 else -2
  else
    if j = 0 then 4 else -3

theorem matrix_addition_correct :
  matrixA + matrixB = resultMatrix :=
by
  sorry

end matrix_addition_correct_l123_123751


namespace max_value_f_on_interval_l123_123860

def f (x : ℝ) : ℝ := 2 / (x - 1) + 1

theorem max_value_f_on_interval : ∃ x ∈ set.Icc 2 6, ∀ y ∈ set.Icc 2 6, f y ≤ f x ∧ f x = 3 :=
by
  exists 2
  simp [f]
  split
  sorry
  sorry

end max_value_f_on_interval_l123_123860


namespace parallelogram_ratio_of_sides_l123_123403

theorem parallelogram_ratio_of_sides :
  ∀ (a b d1 d2 : ℝ),
  (cos (60 * real.pi / 180) = 1 / 2) ∧
  (cos (120 * real.pi / 180) = -1 / 2) ∧
  d1^2 = a^2 + b^2 - a * b ∧
  d2^2 = a^2 + b^2 + a * b ∧
  d1^2 / d2^2 = 1 / 3 →
  a = b :=
begin
  sorry
end

end parallelogram_ratio_of_sides_l123_123403


namespace probability_no_obtuse_triangle_l123_123807

namespace CirclePoints

noncomputable def no_obtuse_triangle_probability : ℝ := 
  let p := 1/64 in
    p

theorem probability_no_obtuse_triangle (X : ℕ → ℝ) (hcirc : ∀ n, 0 ≤ X n ∧ X n < 2 * π) (hpoints : (∀ n m, n ≠ m → X n ≠ X m)) :
  no_obtuse_triangle_probability = 1/64 :=
sorry

end CirclePoints

end probability_no_obtuse_triangle_l123_123807


namespace solution_set_f_ge_1_l123_123016

def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 3 else (1 / 3) ^ x - 2

theorem solution_set_f_ge_1 : 
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≤ -1 ∨ x ≥ 3} :=
by
  sorry

end solution_set_f_ge_1_l123_123016


namespace percent_of_x_is_y_l123_123665

variable {x y : ℝ}

theorem percent_of_x_is_y (h : 0.5 * (x - y) = 0.2 * (x + y)) : y / x ≈ 0.4286 :=
by
  sorry

end percent_of_x_is_y_l123_123665


namespace calc_result_l123_123393

-- Define the operation and conditions
def my_op (a b c : ℝ) : ℝ :=
  3 * (a - b - c)^2

theorem calc_result (x y z : ℝ) : 
  my_op ((x - y - z)^2) ((y - x - z)^2) ((z - x - y)^2) = 0 :=
by
  sorry

end calc_result_l123_123393


namespace number_of_cats_l123_123226

def number_of_dogs : ℕ := 43
def number_of_fish : ℕ := 72
def total_pets : ℕ := 149

theorem number_of_cats : total_pets - (number_of_dogs + number_of_fish) = 34 := 
by
  sorry

end number_of_cats_l123_123226


namespace factor_difference_of_squares_example_l123_123296

theorem factor_difference_of_squares_example :
    (m : ℝ) → (m ^ 2 - 4 = (m + 2) * (m - 2)) :=
by
    intro m
    sorry

end factor_difference_of_squares_example_l123_123296


namespace plane_contains_points_l123_123738

-- Define the points
def point_a : EuclideanSpace ℝ (Fin 3) := ![-2, 3, -1]
def point_b : EuclideanSpace ℝ (Fin 3) := ![2, 3, 1]
def point_c : EuclideanSpace ℝ (Fin 3) := ![4, 1, 0]

-- Define the vectors obtained from subtracting point_a from point_b and point_c
def vector_ab := point_b - point_a
def vector_ac := point_c - point_a

-- Define the cross product vector to find the normal to the plane
def cross_product := EuclideanSpace.cross_product vector_ab vector_ac

-- Define the normalized vector based on the condition A > 0
def normal_vector := if cross_product.1 > 0 then cross_product else -cross_product

-- Define the plane equation using the normal vector and compute D
noncomputable def plane_equation (x y z : ℝ) : ℝ :=
  let A := normal_vector.1
  let B := normal_vector.2
  let C := normal_vector.3
  let D := -(A * point_a.1 + B * point_a.2 + C * point_a.3)
  A * x + B * y + C * z + D

-- Prove that the plane equation equals 0 for the given points
theorem plane_contains_points :
  ∀ (x y z : ℝ),
    (x = -2 ∧ y = 3 ∧ z = -1) ∨
    (x = 2 ∧ y = 3 ∧ z = 1) ∨
    (x = 4 ∧ y = 1 ∧ z = 0) →
    plane_equation x y z = 0 := by
  intros x y z h
  cases h
  case or.inl h1 =>
    rw [h1.1, h1.2, h1.3]
    sorry
  case or.inr h2 =>
    cases h2
    case or.inl h21 =>
      rw [h21.1, h21.2, h21.3]
      sorry
    case or.inr h22 =>
      rw [h22.1, h22.2, h22.3]
      sorry

end plane_contains_points_l123_123738


namespace johns_mean_score_l123_123107

theorem johns_mean_score : 
  let scores := [82, 88, 94, 90] in 
  (List.sum scores) / (scores.length : ℝ) = 88.5 :=
by
  let scores := [82, 88, 94, 90]
  sorry

end johns_mean_score_l123_123107


namespace angle_F_is_54_l123_123179

theorem angle_F_is_54
  (B A F G C : Type)
  (intersect : BF ∩ AG = {C})
  (h1 : AB = BC)
  (h2 : BC = CF)
  (h3 : CF = FG)
  (h4 : ∠A = 3 * ∠B) :
  ∠F = 54 :=
  sorry

end angle_F_is_54_l123_123179


namespace angle_triple_supplementary_l123_123249

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l123_123249


namespace trig_quadrant_l123_123050

theorem trig_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  ∃ k : ℤ, α = (2 * k + 1) * π + α / 2 :=
sorry

end trig_quadrant_l123_123050


namespace square_has_most_axes_of_symmetry_l123_123357

def Shape : Type := { shape : String // shape ∈ {"Square", "Equilateral Triangle", "Isosceles Triangle", "Isosceles Trapezoid"} }

def axes_of_symmetry (s : Shape) : ℕ :=
  match s.val with
  | "Square" => 4
  | "Equilateral Triangle" => 3
  | "Isosceles Triangle" => 1
  | "Isosceles Trapezoid" => 1
  | _ => 0

theorem square_has_most_axes_of_symmetry :
    ∀ (s : Shape), axes_of_symmetry s ≤ axes_of_symmetry { val := "Square", property := sorry } :=
by
  sorry

end square_has_most_axes_of_symmetry_l123_123357


namespace matrix_B_pow_100_eq_I_l123_123998

def matrix_B := Matrix.of ![![0, 1, 0], ![-1, 0, 0], ![0, 0, 1]]
def identity_matrix := Matrix.of ![![1, 0, 0], ![0, 1, 0], ![0, 0, 1]]

theorem matrix_B_pow_100_eq_I : matrix_B^100 = identity_matrix :=
by {
  sorry
}

end matrix_B_pow_100_eq_I_l123_123998


namespace constant_term_binomial_expansion_l123_123586

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| 0, k => if k = 0 then 1 else 0
| n+1, 0 => 1
| n+1, k+1 => binom n k + binom n (k+1)

-- The general term of the binomial expansion
def gen_term (n r : ℕ) (x : ℤ): ℤ := binom n r * (1/3 : ℚ)^r * x^(3 - 3 * r / 2)

-- Problem statement: Prove that the constant term in the given expansion is 5/3
theorem constant_term_binomial_expansion : 
  ∃ (n : ℕ) (r : ℕ) (x : ℤ), n = 6 ∧ r = 2 ∧ gen_term n r x = (5/3 : ℚ) :=
by
  sorry

end constant_term_binomial_expansion_l123_123586


namespace geometric_seq_product_l123_123919

theorem geometric_seq_product
  (a : ℕ → ℝ) (hpos : ∀ n, 0 < a n)
  (hgeo : ∃ r, ∀ n, a(n + 1) = r * a n)
  (hlog : log 2 (a 2 * a 3 * a 5 * a 7 * a 8) = 5) :
  a 1 * a 9 = 4 :=
sorry

end geometric_seq_product_l123_123919


namespace original_peaches_l123_123141

theorem original_peaches (picked: ℕ) (current: ℕ) (initial: ℕ) : 
  picked = 52 → 
  current = 86 → 
  initial = current - picked → 
  initial = 34 := 
by intros h1 h2 h3
   subst h1
   subst h2
   subst h3
   simp

end original_peaches_l123_123141


namespace num_archers_golden_armor_proof_l123_123963
noncomputable section

structure Soldier :=
  (is_archer : Bool)
  (is_golden : Bool)
  (tells_truth : Bool)

def count_soldiers (soldiers : List Soldier) : Nat :=
  soldiers.length

def count_truthful_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => s.tells_truth)).count q

def count_lying_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => ¬s.tells_truth)).count q

def num_archers_golden_armor (soldiers : List Soldier) : Nat :=
  (soldiers.filter (λ s => s.is_archer ∧ s.is_golden)).length

theorem num_archers_golden_armor_proof (soldiers : List Soldier)
  (h1 : count_soldiers soldiers = 55)
  (h2 : count_truthful_responses soldiers (λ s => s.is_golden) + 
        count_lying_responses soldiers (λ s => s.is_golden) = 44)
  (h3 : count_truthful_responses soldiers (λ s => s.is_archer) + 
        count_lying_responses soldiers (λ s => s.is_archer) = 33)
  (h4 : count_truthful_responses soldiers (λ s => true) + 
        count_lying_responses soldiers (λ s => false) = 22) :
  num_archers_golden_armor soldiers = 22 := by
  sorry

end num_archers_golden_armor_proof_l123_123963


namespace proof_problem_l123_123422

-- Define the conditions
variable (A B P : Point)
variable (C l : Line)
variable (M N : Point)

-- Define Point A is on circle F
def is_on_circle (A : Point) : Prop :=
  (A.x - 4)^2 + A.y^2 = 16

-- Define Point B coordinates
def point_B : Prop :=
  B.x = -4 ∧ B.y = 0

-- Define P is the midpoint of AB
def midpoint (A B P : Point) : Prop :=
  P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

-- Define the line l passing through point (-1,3) and the intercept distance condition
def line_l_through_point : Prop :=
  l.contains ⟨-1, 3⟩ ∧ l.intercept_distance_to_origin = 1

-- Define intercept distance for the specific conditions
def intercept_distance_to_origin (l : Line) : ℝ :=
  abs (k * 0 - 1 * 0 + 3) / sqrt (1^2 + k^2) = 1

-- Define the problem in Lean statement
theorem proof_problem :
  (∃ A : Point, is_on_circle A) →
  point_B B →
  midpoint A B P →
  (∃ (C : Circle), ∀ P, midpoint A B P → C.equation = P) ∧
  (l.contains ⟨-1, 3⟩ ∧ line_l_through_point M N |MN| = 2.sqrt 3) →
  C.equation = "x^2 + y^2 = 4" ∧ (l.equation = "4x + 3y - 5 = 0" ∨ l.equation = "x = -1") :=
sorry

end proof_problem_l123_123422


namespace ratio_of_sums_l123_123523

def N : ℕ := 36 * 72 * 50 * 81

def sum_of_odd_divisors (n : ℕ) : ℕ :=
  n.divisors.filter (λ d => d % 2 = 1).sum

def sum_of_even_divisors (n : ℕ) : ℕ :=
  n.divisors.filter (λ d => d % 2 = 0).sum

theorem ratio_of_sums : (sum_of_odd_divisors N : ℚ) / (sum_of_even_divisors N) = 1 / 126 := by
  sorry

end ratio_of_sums_l123_123523


namespace integral_point_distance_l123_123559

theorem integral_point_distance (m n : ℤ) :
  let d := abs ((m : ℝ) - (n : ℝ) + 1 / real.sqrt 2) / real.sqrt 2
  in d ≥ (real.sqrt 2 - 1) / 2 :=
by
  sorry

end integral_point_distance_l123_123559


namespace sum_of_x_and_y_l123_123052

theorem sum_of_x_and_y (x y : ℤ) (h1 : 3 + x = 5) (h2 : -3 + y = 5) : x + y = 10 :=
by
  sorry

end sum_of_x_and_y_l123_123052


namespace man_walking_rate_is_12_l123_123335

theorem man_walking_rate_is_12 (M : ℝ) (woman_speed : ℝ) (time_waiting : ℝ) (catch_up_time : ℝ) 
  (woman_speed_eq : woman_speed = 12) (time_waiting_eq : time_waiting = 1 / 6) 
  (catch_up_time_eq : catch_up_time = 1 / 6): 
  (M * catch_up_time = woman_speed * time_waiting) → M = 12 := by
  intro h
  rw [woman_speed_eq, time_waiting_eq, catch_up_time_eq] at h
  sorry

end man_walking_rate_is_12_l123_123335


namespace total_intersections_l123_123552

def north_south_streets : ℕ := 10
def east_west_streets : ℕ := 10

theorem total_intersections :
  (north_south_streets * east_west_streets = 100) :=
by
  sorry

end total_intersections_l123_123552


namespace no_obtuse_triangle_probability_l123_123797

noncomputable def probability_no_obtuse_triangle : ℝ :=
 let θ := 1/2 in
 let prob_A2_A3_given_A0A1 := (3/8) * (3/8) in
 θ * prob_A2_A3_given_A0A1

theorem no_obtuse_triangle_probability :
  probability_no_obtuse_triangle = 9/128 :=
by
  sorry

end no_obtuse_triangle_probability_l123_123797


namespace min_integral_value_l123_123519

def f (p q : ℕ) (x : ℝ) : ℝ := x ^ p * (Real.sin x) ^ q * (Real.cos x) ^ (4 - p - q)

theorem min_integral_value :
  (∫ x in 0..(Real.pi / 2), f 4 0 x) = Real.pi ^ 5 / 160 :=
  sorry

end min_integral_value_l123_123519


namespace value_of_expression_l123_123291

theorem value_of_expression :
  3 - (-3)⁻² = 26 / 9 := 
by 
  norm_num   -- simplifies simple arithmetic expressions
  sorry      -- placeholder for the complete proof if needed

end value_of_expression_l123_123291


namespace modulus_of_z_l123_123134

theorem modulus_of_z (z : ℂ) (h : (1 - complex.i) * z = 1 + complex.i) : complex.abs z = 1 :=
by
  sorry

end modulus_of_z_l123_123134


namespace exists_symmetric_tangent_min_length_l123_123053

noncomputable def circle_symmetric_tangent_min_length (a b : ℝ) : Prop :=
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 3 = 0) →
  (∀ a b : ℝ, 2 * a * x + b * y + 6 = 0) →
  (a - 2 * b - 6 = 0) →
  let d : ℝ := Real.sqrt(2 * a^2 - 8 * a + 26)
  let tangent_len : ℝ := Real.sqrt(d^2 - 2)
  tangent_len = 4

theorem exists_symmetric_tangent_min_length :
  ∃ (a b : ℝ), circle_symmetric_tangent_min_length a b :=
by
  sorry

end exists_symmetric_tangent_min_length_l123_123053


namespace rahul_batting_average_before_match_l123_123573

open Nat

theorem rahul_batting_average_before_match (R : ℕ) (A : ℕ) :
  (R + 69 = 6 * 54) ∧ (A = R / 5) → (A = 51) :=
by
  sorry

end rahul_batting_average_before_match_l123_123573


namespace length_of_LN_l123_123486

theorem length_of_LN (LM N : ℝ) (hN : sin N = 5/13) (hLM : LM = 10) : 
  ∃ LN : ℝ, LN = 26 := 
by
sorry

end length_of_LN_l123_123486


namespace archers_in_golden_armor_l123_123954

theorem archers_in_golden_armor (total_soldiers archers_swordsmen total_affirmations armor_affirmations archer_affirmations monday_affirmations : ℕ)
  (h1: total_soldiers = 55)
  (h2: armor_affirmations = 44)
  (h3: archer_affirmations = 33)
  (h4: monday_affirmations = 22)
  (h5: ∑ x in ({armor_affirmations, archer_affirmations, monday_affirmations} : finset ℕ), x = 99) 
  : ∃ (archers_in_golden : ℕ), archers_in_golden = 22 := by
  -- Definitions and theorems will go here
  sorry

end archers_in_golden_armor_l123_123954


namespace sum_of_projections_l123_123984

-- Define the given lengths of the triangle sides
def a : ℝ := 5
def b : ℝ := 7
def c : ℝ := 6

-- Define the centroid projections to the sides
noncomputable def GP : ℝ := 2 * real.sqrt 6 / 3
noncomputable def GQ : ℝ := 4 * real.sqrt 6 / 7
noncomputable def GR : ℝ := 4 * real.sqrt 6 / 5

-- Prove the sum of GP, GQ, and GR
theorem sum_of_projections : GP + GQ + GR = (122 * real.sqrt 6) / 105 :=
by
  sorry

end sum_of_projections_l123_123984


namespace probability_cube_vertices_in_plane_l123_123822

open Finset

noncomputable def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_cube_vertices_in_plane : 
  let total_ways := choose 8 4 in
  let favorable_ways := 12 in
  0 < total_ways →  -- Ensure total_ways is non-zero to avoid division by zero
  let P := (favorable_ways : ℝ) / (total_ways : ℝ) in
  P = 6 / 35 :=
by 
  sorry

end probability_cube_vertices_in_plane_l123_123822


namespace sequence_term_a10_finite_sequence_values_a_l123_123405

theorem sequence_term_a10 (a : ℝ) (h1 : a = 1) : 
  let a_seq : ℕ → ℝ := 
    λ n, Nat.recOn n a
      (λ n a_n, if n % 2 = 0 then a_n + 2 else a_n / 2) 
  in a_seq 10 = 63 / 16 :=
by
  sorry

theorem finite_sequence_values_a (a : ℝ) :
  (∀ n : ℕ, ∃ m : ℕ, ∀ k : ℕ, k > m → a_seq k = a_seq k + n) ↔ a = 2 :=
by
  let a_seq : ℕ → ℝ := 
      λ n, Nat.recOn n a
      (λ n a_n, if n % 2 = 0 then a_n + 2 else a_n / 2) 
  sorry

end sequence_term_a10_finite_sequence_values_a_l123_123405


namespace four_points_no_obtuse_triangle_l123_123775

noncomputable def probability_no_obtuse_triangle : ℝ :=
1 / 64

theorem four_points_no_obtuse_triangle (A B C D : circle) :
  (∀ (P Q : circle) (PQ_angle : ℝ), PQ_angle < π/2) → 
  probability_no_obtuse_triangle = 1 / 64 :=
sorry

end four_points_no_obtuse_triangle_l123_123775


namespace nine_distinct_numbers_product_l123_123160

variable (a b c d e f g h i : ℕ)

theorem nine_distinct_numbers_product (ha : a = 12) (hb : b = 9) (hc : c = 2)
                                      (hd : d = 1) (he : e = 6) (hf : f = 36)
                                      (hg : g = 18) (hh : h = 4) (hi : i = 3) :
  (a * b * c = 216) ∧ (d * e * f = 216) ∧ (g * h * i = 216) ∧
  (a * d * g = 216) ∧ (b * e * h = 216) ∧ (c * f * i = 216) ∧
  (a * e * i = 216) ∧ (c * e * g = 216) :=
by
  sorry

end nine_distinct_numbers_product_l123_123160


namespace three_digit_integers_with_product_36_l123_123877

/--
There are 21 distinct 3-digit integers such that the product of their digits equals 36, and each digit is between 1 and 9.
-/
theorem three_digit_integers_with_product_36 : 
  ∃ n : ℕ, digit_product_count 36 3 n ∧ n = 21 :=
sorry

end three_digit_integers_with_product_36_l123_123877


namespace range_of_a_l123_123438

theorem range_of_a (a : ℝ) : (∀ x : ℤ, x > 2 * a - 3 ∧ 2 * (x : ℝ) ≥ 3 * ((x : ℝ) - 2) + 5) ↔ (1 / 2 ≤ a ∧ a < 1) :=
sorry

end range_of_a_l123_123438


namespace num_archers_golden_armor_proof_l123_123959
noncomputable section

structure Soldier :=
  (is_archer : Bool)
  (is_golden : Bool)
  (tells_truth : Bool)

def count_soldiers (soldiers : List Soldier) : Nat :=
  soldiers.length

def count_truthful_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => s.tells_truth)).count q

def count_lying_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => ¬s.tells_truth)).count q

def num_archers_golden_armor (soldiers : List Soldier) : Nat :=
  (soldiers.filter (λ s => s.is_archer ∧ s.is_golden)).length

theorem num_archers_golden_armor_proof (soldiers : List Soldier)
  (h1 : count_soldiers soldiers = 55)
  (h2 : count_truthful_responses soldiers (λ s => s.is_golden) + 
        count_lying_responses soldiers (λ s => s.is_golden) = 44)
  (h3 : count_truthful_responses soldiers (λ s => s.is_archer) + 
        count_lying_responses soldiers (λ s => s.is_archer) = 33)
  (h4 : count_truthful_responses soldiers (λ s => true) + 
        count_lying_responses soldiers (λ s => false) = 22) :
  num_archers_golden_armor soldiers = 22 := by
  sorry

end num_archers_golden_armor_proof_l123_123959


namespace f_n_cases_f_n_eq_n_solutions_l123_123835

-- Define the function f(n) with given conditions
def f (n : ℕ) : ℕ :=
if n > 2000 then n - 12
else f (f (n + 16))

theorem f_n_cases (n : ℕ) : 
    (f n = if n > 2000 then n - 12 else 1988 + (n % 4)) :=
sorry

theorem f_n_eq_n_solutions (n : ℕ) : 
  (f n = n) ↔ (n = 1989 ∨ n = 1990 ∨ n = 1991 ∨ n = 1992) :=
sorry

end f_n_cases_f_n_eq_n_solutions_l123_123835


namespace probability_center_inside_tetrahedron_l123_123395

open Real

/-- The probability that the center of the sphere lies inside the tetrahedron formed by four randomly chosen points on the surface of the sphere is 1/8. -/
theorem probability_center_inside_tetrahedron : 
  let S := set.univ : set (ℝ^3),
      P1, P2, P3, P4 : S,
      O : ℝ^3
  in probability (center_inside_tetrahedron P1 P2 P3 P4) = 1/8 := 
sorry

end probability_center_inside_tetrahedron_l123_123395


namespace cos_arithmetic_sequence_l123_123007

-- Problem statement translation to Lean
theorem cos_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (a1 a5 a9 : ℝ)
    (h_arith_seq : ∀ n, a (n + 1) = a n + d)
    (h_eq_a1 : a 1 = a1)
    (h_eq_a5 : a 5 = a5)
    (h_eq_a9 : a 9 = a9)
    (h_sum : a1 + a5 + a9 = π) :
    (cos (a 2 + a 8) = -1/2) :=
begin
    -- proof goes here
    sorry
end

end cos_arithmetic_sequence_l123_123007


namespace cost_per_charge_l123_123040

theorem cost_per_charge
  (charges : ℕ) (budget left : ℝ) (cost_per_charge : ℝ)
  (charges_eq : charges = 4)
  (budget_eq : budget = 20)
  (left_eq : left = 6) :
  cost_per_charge = (budget - left) / charges :=
by
  apply sorry

end cost_per_charge_l123_123040


namespace no_solution_to_inequalities_l123_123171

theorem no_solution_to_inequalities :
  ∀ (x y z t : ℝ), 
    ¬ (|x| > |y - z + t| ∧
       |y| > |x - z + t| ∧
       |z| > |x - y + t| ∧
       |t| > |x - y + z|) :=
by
  intro x y z t
  sorry

end no_solution_to_inequalities_l123_123171


namespace Rectangle_Circle_Problem_l123_123518

-- Defining the setup for the mathematical problem
variables {A B C D P Q R : Type*}

-- Assume ABCD is a rectangle
variable [AB : is_rectangle A B C D]

-- Assume D is the center of the circle with radius DA
variable [DCircle : is_circle_center D (distance D A)]

-- Assume the circle intersects the extension of AD at P
variable [DP : on_circle_extension A D P (distance D A)]

-- Assume line PC cuts the circle at Q and the extension of AB at R
variable [PCQ : on_line_and_circle P C Q]
variable [PCR : on_line_extension_and_intersection P C R A B]

-- The goal is to prove QB = BR
theorem Rectangle_Circle_Problem : distance Q B = distance B R := 
sorry

end Rectangle_Circle_Problem_l123_123518


namespace AB1_eq_CA1_l123_123982

variable (a b c d : ℝ)

-- 1. Conditions
def condition1 : Prop := a^2 + b^2 + d^2 - a*c + d^2 = c^2
def condition2 : Prop := -c*(c-a) + d^2 = 0

-- 2. Distances computation
def distance_AB1 : ℝ := sqrt (a^2 + b^2 + d^2)
def distance_CA1 : ℝ := sqrt (c^2 + d^2)

-- Proof statement
theorem AB1_eq_CA1 (h1 : condition1 a b c d) (h2 : condition2 a b c d) : distance_AB1 a b d = distance_CA1 c d := 
by 
  -- Expand distances from the correct answer
  have eq1 : c^2 = a^2 + b^2 := sorry
  have eq2 : a^2 + b^2 + d^2 = c^2 + d^2 := by
    rw [eq1]
  show sqrt (a^2 + b^2 + d^2) = sqrt (c^2 + d^2)
  rw [eq2]
  -- thus, the conclusion
  exact sorry

end AB1_eq_CA1_l123_123982


namespace no_obtuse_triangle_l123_123792

-- Conditions
def points_on_circle_uniformly_at_random (n : ℕ) : Prop :=
  ∀ i < n, ∀ j < n, i ≠ j → ∃ θ_ij : ℝ, 0 ≤ θ_ij ∧ θ_ij ≤ π

-- Theorem statement
theorem no_obtuse_triangle (hn : points_on_circle_uniformly_at_random 4) :
  let p := \left(\frac{1}{2}\right)^6 in
  p = \frac{1}{64} :=
sorry

end no_obtuse_triangle_l123_123792


namespace overall_gain_percent_l123_123469

variables (C_A S_A C_B S_B : ℝ)

def cost_price_A (n : ℝ) : ℝ := n * C_A
def selling_price_A (n : ℝ) : ℝ := n * S_A

def cost_price_B (n : ℝ) : ℝ := n * C_B
def selling_price_B (n : ℝ) : ℝ := n * S_B

theorem overall_gain_percent :
  (selling_price_A 25 = cost_price_A 50) →
  (selling_price_B 30 = cost_price_B 60) →
  ((S_A - C_A) / C_A * 100 = 100) ∧ ((S_B - C_B) / C_B * 100 = 100) :=
by
  sorry

end overall_gain_percent_l123_123469


namespace average_people_per_hour_l123_123978

-- Define the conditions
def people_moving : ℕ := 3000
def days : ℕ := 5
def hours_per_day : ℕ := 24
def total_hours : ℕ := days * hours_per_day

-- State the problem
theorem average_people_per_hour :
  people_moving / total_hours = 25 :=
by
  -- Proof goes here
  sorry

end average_people_per_hour_l123_123978


namespace recurring_decimal_to_rational_l123_123310

theorem recurring_decimal_to_rational : 
  (0.125125125 : ℝ) = 125 / 999 :=
sorry

end recurring_decimal_to_rational_l123_123310


namespace car_speed_l123_123659

theorem car_speed (v : ℝ) (h : (1/v) * 3600 = ((1/48) * 3600) + 15) : v = 40 := 
by 
  sorry

end car_speed_l123_123659


namespace number_of_zuminglish_advanced_words_8_letters_mod_100_l123_123066

def zuminglish_advanced_words (n : ℕ) : ℕ × ℕ × ℕ :=
  let rec helper (d e f count : ℕ) : ℕ × ℕ × ℕ :=
    if count = n then (d, e, f)
    else let d_next := 2 * d + 2 * f
         let e_next := d
         let f_next := e
         in helper d_next e_next f_next (count + 1)
  helper 2 1 2 2

theorem number_of_zuminglish_advanced_words_8_letters_mod_100 :
  let (d8, e8, f8) := zuminglish_advanced_words 8 in
  (d8 + e8 + f8) % 100 = 24 :=
by
  let (d8, e8, f8) := zuminglish_advanced_words 8
  have h : (d8 + e8 + f8) % 100 = 24 := sorry
  exact h

end number_of_zuminglish_advanced_words_8_letters_mod_100_l123_123066


namespace probability_no_obtuse_triangle_is_9_over_64_l123_123810

noncomputable def probability_no_obtuse_triangle (A0 A1 A2 A3 : ℝ) : ℝ :=
  -- Define the probabilistic model according to the problem conditions
  -- Assuming A0, A1, A2, and A3 are positions of points on the circle parametrized by angles in radians
  let θ := real.angle.lower_half (A1 - A0) in
  let prob_A2 := (π - θ) / (2 * π) in
  let prob_A3 := (π - θ) / (2 * π) in
  (1 / 2) * (prob_A2 * prob_A3)

theorem probability_no_obtuse_triangle_is_9_over_64 :
  probability_no_obtuse_triangle A0 A1 A2 A3 = 9 / 64 :=
by sorry

end probability_no_obtuse_triangle_is_9_over_64_l123_123810


namespace train_stop_time_l123_123384

theorem train_stop_time (speed_no_stops speed_with_stops : ℕ) (time_per_hour : ℕ) (stoppage_time_per_hour : ℕ) :
  speed_no_stops = 45 →
  speed_with_stops = 30 →
  time_per_hour = 60 →
  stoppage_time_per_hour = 20 :=
by
  intros h1 h2 h3
  sorry

end train_stop_time_l123_123384


namespace games_needed_for_winner_l123_123145

theorem games_needed_for_winner (n : ℕ) (hn : n = 20) : 
    let games := (n - 1) in
    games = 19 :=
by {
    rw hn,
    have h : n - 1 = 20 - 1 := by rw hn,
    exact h,
}

end games_needed_for_winner_l123_123145


namespace product_of_all_possible_values_for_b_l123_123592

noncomputable def product_of_b (b : ℝ) : ℝ :=
  if (3*b - 5)^2 + (b + 3)^2 = (3*Real.sqrt 5)^2 then b else 0

theorem product_of_all_possible_values_for_b : 
  (∑ b in {b : ℝ | (3*b - 5)^2 + (b + 3)^2 = (3*Real.sqrt 5)^2}, b) = -11/10 := by
  sorry

end product_of_all_possible_values_for_b_l123_123592


namespace four_points_no_obtuse_triangle_l123_123777

noncomputable def probability_no_obtuse_triangle : ℝ :=
1 / 64

theorem four_points_no_obtuse_triangle (A B C D : circle) :
  (∀ (P Q : circle) (PQ_angle : ℝ), PQ_angle < π/2) → 
  probability_no_obtuse_triangle = 1 / 64 :=
sorry

end four_points_no_obtuse_triangle_l123_123777


namespace simplify_expression_l123_123182

theorem simplify_expression :
  8 * (15 / 4) * (-45 / 50) = - (12 / 25) :=
by
  sorry

end simplify_expression_l123_123182


namespace S4_value_l123_123070

noncomputable theory

def geometric_series (a r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

def S (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem S4_value
  (a r : ℝ)
  (h_pos : ∀ n, geometric_series a r n > 0)
  (h1 : S a r 2 = 7)
  (h2 : S a r 6 = 91) :
  S a r 4 = 14 :=
sorry

end S4_value_l123_123070


namespace geometric_sequence_find_b_value_l123_123002

variable {a : ℕ → ℝ}

variable {S : ℕ → ℝ}

noncomputable def k : ℝ := sorry  -- value of k is to be determined

axiom positive_terms (n : ℕ) : a n > 0
axiom sum_of_terms (n : ℕ) : S n = ∑ i in finset.range n, a i
axiom points_on_graph (n : ℕ) : S n = k * a n + b
axiom a1_a6 : a 1 + a 6 = 66
axiom a2_a5 : a 2 * a 5 = 128

theorem geometric_sequence :
  ∃ q > 0, ∀ n, a (n + 1) = q * a n :=
sorry

theorem find_b_value {b : ℝ} :
  a 1 + a 6 = 66 → a 2 * a 5 = 128 → k = 2 → ∃ b, S 1 = k * a 1 + b :=
sorry

end geometric_sequence_find_b_value_l123_123002


namespace determine_winner_games_l123_123143

theorem determine_winner_games (T : ℕ) (hT : T = 20) : 
  (games_played_to_determine_winner T = 19) := 
by
  sorry

def games_played_to_determine_winner (teams : ℕ) : ℕ :=
  teams - 1

end determine_winner_games_l123_123143


namespace common_tangent_segment_length_l123_123840

variables {C : Point} {O1 O2 : Point} {r R : ℝ}
-- Declare the existence of the two circles centered at O1 and O2 with radii r and R touching the angle
variables (circle1 : Circle O1 r) (circle2 : Circle O2 R)
-- Define the condition that the circles do not touch each other
variables (CirclesDoNotTouch : dist O1 O2 > (R + r))

theorem common_tangent_segment_length (PQ : ℝ) :
  PQ > 2 * real.sqrt (R * r) :=
sorry

end common_tangent_segment_length_l123_123840


namespace peanut_price_is_correct_l123_123697

noncomputable def price_per_pound_of_peanuts : ℝ := 
  let total_weight := 100
  let mixed_price_per_pound := 2.5
  let cashew_weight := 60
  let cashew_price_per_pound := 4
  let peanut_weight := total_weight - cashew_weight
  let total_revenue := total_weight * mixed_price_per_pound
  let cashew_cost := cashew_weight * cashew_price_per_pound
  let peanut_cost := total_revenue - cashew_cost
  peanut_cost / peanut_weight

theorem peanut_price_is_correct :
  price_per_pound_of_peanuts = 0.25 := 
by sorry

end peanut_price_is_correct_l123_123697


namespace min_translation_is_pi_div_6_l123_123013

def f (x : Real) : Real := Real.sin (2 * x + Real.pi / 3)

theorem min_translation_is_pi_div_6 (φ : Real) (hφ : φ > 0) 
  (odd_f : ∀ x, f (x - φ) = -f (-(x - φ))) 
  : φ = Real.pi / 6 :=
by
  sorry

end min_translation_is_pi_div_6_l123_123013


namespace arithmetic_mean_l123_123407

theorem arithmetic_mean (n : ℕ) (h : 1 < n) :
  let set := (1 + 1/n) :: List.replicate n 1 in
  (List.sum set : ℝ) / (n + 1) = 1 + 1/(n * (n + 1)) :=
by
  sorry

end arithmetic_mean_l123_123407


namespace numerical_value_expression_l123_123461

theorem numerical_value_expression (x y z : ℚ) (h1 : x - 4 * y - 2 * z = 0) (h2 : 3 * x + 2 * y - z = 0) (h3 : z ≠ 0) : 
  (x^2 - 5 * x * y) / (2 * y^2 + z^2) = 164 / 147 :=
by sorry

end numerical_value_expression_l123_123461


namespace element_with_62_07_percent_mass_l123_123737

noncomputable def atomic_mass (e : String) : Option ℝ :=
  if e = "C" then some 12.01
  else if e = "H" then some 1.008
  else if e = "O" then some 16.00
  else none

noncomputable def molar_mass_compound : ℝ :=
  (3 * 12.01) + (6 * 1.008) + (1 * 16.00)

def mass_percentage (element : String) (mass : ℝ) : ℝ :=
  (mass / molar_mass_compound) * 100

theorem element_with_62_07_percent_mass (element : String) :
  mass_percentage element (3 * 12.01) = 62.07 → element = "C" :=
by
  intro h
  sorry

end element_with_62_07_percent_mass_l123_123737


namespace Bernardo_has_higher_probability_l123_123707

-- Defining the sets from which Bernardo and Silvia pick their numbers
def BernardoSet := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def SilviaSet := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Defining the problem statement, which is the probability comparison as per the given conditions
noncomputable def probability_Bernardo_larger : ℚ := 13 / 20

-- The Lean 4 theorem statement for the problem
theorem Bernardo_has_higher_probability :
  ∃ (prob : ℚ), prob = probability_Bernardo_larger :=
begin
  use probability_Bernardo_larger,
  sorry
end

end Bernardo_has_higher_probability_l123_123707


namespace total_cookies_baked_l123_123041

def cookies_baked_yesterday : ℕ := 435
def cookies_baked_today : ℕ := 139

theorem total_cookies_baked : cookies_baked_yesterday + cookies_baked_today = 574 := by
  sorry

end total_cookies_baked_l123_123041


namespace all_div_by_6_upto_88_l123_123401

def numbers_divisible_by_6_upto_88 := [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]

theorem all_div_by_6_upto_88 :
  ∀ n : ℕ, 1 < n ∧ n ≤ 88 ∧ n % 6 = 0 → n ∈ numbers_divisible_by_6_upto_88 :=
by
  intro n
  intro h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2
  sorry

end all_div_by_6_upto_88_l123_123401


namespace correct_sample_size_l123_123353

axiom Population : Type := { x : ℕ // x < 1001 }
axiom Sample : Type := { y : ℕ // y < 101 }

def population_size (s : Population) : ℕ := 1000

def sample_size (s : Sample) : ℕ := 100

theorem correct_sample_size : sample_size (s : Sample) = 100 :=
by
  sorry

end correct_sample_size_l123_123353


namespace estimate_students_scoring_above_120_l123_123916

noncomputable def classSize : ℕ := 60
noncomputable def meanScore : ℝ := 110
noncomputable def stdDevScore : ℝ := 10
noncomputable def probWithinRange : ℝ := 0.35

theorem estimate_students_scoring_above_120 (students : ℕ) (mean score stddev : ℝ) (P : ℝ) :
  students = classSize →
  mean = meanScore →
  stddev = stdDevScore →
  P = probWithinRange →
  let probAbove120 := (1 - P * 2) / 2 in
  let estimatedStudents := probAbove120 * students in
  estimatedStudents = 9 :=
by
  intros students_eq mean_eq stddev_eq P_eq
  unfold probAbove120 estimatedStudents
  sorry

end estimate_students_scoring_above_120_l123_123916


namespace matrix_problem_l123_123060

variable {n : Type} [Field n] [AddGroup n]
variable (B : Matrix n n n)
variable I : Matrix n n n

-- Assumption: The matrix B has an inverse
variable [Invertible B]

-- Assumption: (B - 3 * I)(B - 5 * I) = 0
axiom h : (B - 3 • I) ⬝ (B - 5 • I) = 0

-- Target: Prove B + 10 * B⁻¹ = 8 * I
theorem matrix_problem : B + 10 • ⅟ B = 8 • I := 
by sorry

end matrix_problem_l123_123060


namespace translated_function_correct_l123_123236

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * (x - Real.pi / 2) + Real.pi / 3)

noncomputable def h (x : ℝ) : ℝ := g x - 2

theorem translated_function_correct :
  ∀ x : ℝ, h x = Real.sin (2 * x - 2 * Real.pi / 3) - 2 :=
begin
  intro x,
  sorry
end

end translated_function_correct_l123_123236


namespace constant_term_value_l123_123891

variable (y : ℝ)

def constant_term_in_expansion (y : ℝ) (n : ℕ) : ℝ :=
  -- The function calculating the constant term of (y + 2/y)^n.
  sorry -- this would be the detailed expansion expression

theorem constant_term_value :
  let n := 3 * (∫ x in (-real.pi / 2)..(real.pi / 2), real.sin x + real.cos x) in
  constant_term_in_expansion y (nat.floor n) = 160 := by
  sorry

end constant_term_value_l123_123891


namespace range_fx_in_interval_l123_123200

-- Define the function f(x) as given in the problem
noncomputable def f (x : ℝ) : ℝ := sin (x / 2) * cos (x / 2) + cos (x / 2) ^ 2

-- Define the interval for x
def interval (x : ℝ) : Prop := 0 < x ∧ x < pi / 2

-- State the range of f(x) on the given interval
theorem range_fx_in_interval :
  (∀ x : ℝ, interval x → (1 / 2 : ℝ) < f x ∧ f x ≤ (1 + Real.sqrt 2) / 2) :=
by
  sorry

end range_fx_in_interval_l123_123200


namespace smallest_four_digit_multiple_of_18_l123_123754

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 18 ∣ n ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ 18 ∣ m → n ≤ m := by
  use 1008
  sorry

end smallest_four_digit_multiple_of_18_l123_123754


namespace archers_in_golden_armor_count_l123_123942

theorem archers_in_golden_armor_count : 
  ∃ (archers_golden : ℕ), 
    let total_soldiers := 55 in
    let golden_armor_affirmative := 44 in
    let archer_affirmative := 33 in
    let monday_affirmative := 22 in
    let number_archers_golden := archers_golden in
    (number_archers_golden = 22) ∧ 
    ∃ (archer : ℕ) (swordsman : ℕ) (golden : ℕ) (black : ℕ),
      archer + swordsman = total_soldiers ∧
      golden + black = total_soldiers ∧
      golden + black = total_soldiers ∧
      (swordsman * golden + archer * golden = golden_armor_affirmative) ∧
      (swordsman * archer + swordsman * golden = archer_affirmative) ∧
      ((!monday_affirmative ∧ swordsman * golden) + (!monday_affirmative ∧ archer * black) = monday_affirmative)
:= sorry

end archers_in_golden_armor_count_l123_123942


namespace B_oxen_count_l123_123306

/- 
  A puts 10 oxen for 7 months.
  B puts some oxen for 5 months.
  C puts 15 oxen for 3 months.
  The rent of the pasture is Rs. 175.
  C should pay Rs. 45 as his share of rent.
  We need to prove that B put 12 oxen for grazing.
-/

def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

def A_ox_months := oxen_months 10 7
def C_ox_months := oxen_months 15 3

def total_rent : ℕ := 175
def C_rent_share : ℕ := 45

theorem B_oxen_count (x : ℕ) : 
  (C_rent_share : ℝ) / total_rent = (C_ox_months : ℝ) / (A_ox_months + 5 * x + C_ox_months) →
  x = 12 := 
by
  sorry

end B_oxen_count_l123_123306


namespace archers_in_golden_armor_l123_123951

theorem archers_in_golden_armor (total_soldiers archers_swordsmen total_affirmations armor_affirmations archer_affirmations monday_affirmations : ℕ)
  (h1: total_soldiers = 55)
  (h2: armor_affirmations = 44)
  (h3: archer_affirmations = 33)
  (h4: monday_affirmations = 22)
  (h5: ∑ x in ({armor_affirmations, archer_affirmations, monday_affirmations} : finset ℕ), x = 99) 
  : ∃ (archers_in_golden : ℕ), archers_in_golden = 22 := by
  -- Definitions and theorems will go here
  sorry

end archers_in_golden_armor_l123_123951


namespace perfect_squares_diff_count_of_odd_perfect_squares_lt_25000_l123_123174

theorem perfect_squares_diff (a : ℕ) (h : a^2 < 25000) :
  (∃ b : ℕ, (b + 1)^2 - b^2 = a^2) ↔ (a % 2 = 1) :=
sorry

theorem count_of_odd_perfect_squares_lt_25000 :
  {a : ℕ | a^2 < 25000 ∧ a % 2 = 1}.card = 79 :=
sorry

end perfect_squares_diff_count_of_odd_perfect_squares_lt_25000_l123_123174


namespace length_MN_l123_123437

noncomputable def hyperbola := { p : ℝ × ℝ | p.1^2 / 3 - p.2^2 = 1 }

def origin : ℝ × ℝ := (0, 0)
def focus : ℝ × ℝ := (2, 0)

def line_through_focus (m x : ℝ) : ℝ := m * (x - 2)

def asymptote1 : ℝ → ℝ := λ x, (sqrt 3 / 3) * x
def asymptote2 : ℝ → ℝ := λ x, (- sqrt 3 / 3) * x

theorem length_MN (M N O : ℝ × ℝ) :
  hyperbola ∈ { p : ℝ × ℝ | p.1 = (3/2) ∧ p.2 = - (sqrt 3 / 2) } →
  hyperbola ∈ { p : ℝ × ℝ | p.1 = 3 ∧ p.2 = sqrt 3 } →
  (triangle OMN) is_right_angle →
  dist M N = 3 :=
sorry

end length_MN_l123_123437


namespace centrally_symmetric_impl_congruent_l123_123240

-- Definitions from the conditions:
def centrally_symmetric {α : Type} [metric_space α] (p : α) (S T : set α) : Prop :=
  ∃ (O : α), ∀ (x ∈ S), ∃ (y ∈ T), dist (rotate_180 O x) y = 0

-- Proving the question in terms of Lean statement
theorem centrally_symmetric_impl_congruent {α : Type} [metric_space α] {S T : set α} {p : α} :
  centrally_symmetric p S T → congruent S T := 
sorry

end centrally_symmetric_impl_congruent_l123_123240


namespace blue_marbles_difference_l123_123237

theorem blue_marbles_difference  (a b : ℚ) 
  (h1 : 3 * a + 2 * b = 80)
  (h2 : 2 * a = b) :
  (7 * a - 3 * b) = 80 / 7 := by
  sorry

end blue_marbles_difference_l123_123237


namespace num_handshakes_l123_123361

-- Define the conditions as constants
constant num_twins : ℕ := 24
constant num_triplets : ℕ := 24
constant twin_handshake_twins : ℕ := 22
constant twin_handshake_triplets : ℕ := 16
constant triplet_handshake_triplets : ℕ := 21
constant triplet_handshake_twins : ℕ := 8

theorem num_handshakes : 
  let handshake_twins := (num_twins * twin_handshake_twins) / 2,
      handshake_triplets := (num_triplets * triplet_handshake_triplets) / 2,
      handshake_cross := num_twins * twin_handshake_triplets in
  handshake_twins + handshake_triplets + handshake_cross = 900 := by
  -- Proof goes here
  sorry

end num_handshakes_l123_123361


namespace no_obtuse_triangle_l123_123791

-- Conditions
def points_on_circle_uniformly_at_random (n : ℕ) : Prop :=
  ∀ i < n, ∀ j < n, i ≠ j → ∃ θ_ij : ℝ, 0 ≤ θ_ij ∧ θ_ij ≤ π

-- Theorem statement
theorem no_obtuse_triangle (hn : points_on_circle_uniformly_at_random 4) :
  let p := \left(\frac{1}{2}\right)^6 in
  p = \frac{1}{64} :=
sorry

end no_obtuse_triangle_l123_123791


namespace g_is_odd_l123_123097

def g (x : ℝ) : ℝ := (3^x - 2) / (3^x + 2)

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  intros x
  sorry

end g_is_odd_l123_123097


namespace triangle_formation_probability_l123_123921

def ak (k : ℕ) : ℝ := 2 * Real.sin (k * Real.pi / 15)

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def valid_triangles : ℕ :=
  (Finset.univ : Finset (Fin 8)).sum (λ x,
    (Finset.univ : Finset (Fin 8)).sum (λ y,
      (Finset.univ : Finset (Fin 8)).filter (λ z,
        x < y ∧ y < z ∧
          is_triangle (ak x) (ak y) (ak z)
      ).card
    )
  )

noncomputable def total_combinations : ℕ := ( (Finset.univ : Finset (Fin 8)).card * (Finset.univ : Finset (Fin 8)).card * (Finset.univ : Finset (Fin 8)).card - valid_triangles) / 6

theorem triangle_formation_probability : valid_triangles / 105 = 345 / 455 :=
by sorry

end triangle_formation_probability_l123_123921


namespace abc_ineq_l123_123181

theorem abc_ineq (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 0) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
by 
  sorry

end abc_ineq_l123_123181


namespace clear_time_correct_l123_123313

def length_train1 : ℝ := 131       -- Length of Train 1 in meters
def length_train2 : ℝ := 165       -- Length of Train 2 in meters
def speed_train1_kmh : ℝ := 80     -- Speed of Train 1 in km/h
def speed_train2_kmh : ℝ := 65     -- Speed of Train 2 in km/h
def total_length := length_train1 + length_train2  -- Total distance to be covered

-- Conversion factor from km/h to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

-- Relative speed in m/s
def relative_speed_ms := kmh_to_ms (speed_train1_kmh + speed_train2_kmh)

-- Time to be clear of each other
def clear_time : ℝ := total_length / relative_speed_ms

theorem clear_time_correct : clear_time ≈ 7.35 :=
by
  sorry

end clear_time_correct_l123_123313


namespace probability_cube_vertices_in_plane_l123_123821

open Finset

noncomputable def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_cube_vertices_in_plane : 
  let total_ways := choose 8 4 in
  let favorable_ways := 12 in
  0 < total_ways →  -- Ensure total_ways is non-zero to avoid division by zero
  let P := (favorable_ways : ℝ) / (total_ways : ℝ) in
  P = 6 / 35 :=
by 
  sorry

end probability_cube_vertices_in_plane_l123_123821


namespace remainder_of_series_div_9_l123_123288

def sum (n : Nat) : Nat := n * (n + 1) / 2

theorem remainder_of_series_div_9 : (sum 20) % 9 = 3 :=
by
  -- The proof will go here
  sorry

end remainder_of_series_div_9_l123_123288


namespace expected_sides_rectangle_expected_sides_polygon_l123_123628

-- Part (a)
theorem expected_sides_rectangle (k : ℕ) (h : k > 0) : (4 + 4 * k) / (k + 1) → 4 :=
by sorry

-- Part (b)
theorem expected_sides_polygon (n k : ℕ) (h : n > 2) (h_k : k ≥ 0) : (n + 4 * k) / (k + 1) = (n + 4 * k) / (k + 1) :=
by sorry

end expected_sides_rectangle_expected_sides_polygon_l123_123628


namespace average_people_per_hour_l123_123977

-- Define the conditions
def people_moving : ℕ := 3000
def days : ℕ := 5
def hours_per_day : ℕ := 24
def total_hours : ℕ := days * hours_per_day

-- State the problem
theorem average_people_per_hour :
  people_moving / total_hours = 25 :=
by
  -- Proof goes here
  sorry

end average_people_per_hour_l123_123977


namespace sum_of_primes_20_to_40_l123_123641

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ (2 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime |>.sum

theorem sum_of_primes_20_to_40 : sum_of_primes_between 20 40 = 120 := by
  sorry

end sum_of_primes_20_to_40_l123_123641


namespace point_P_in_fourth_quadrant_l123_123565

-- Define the problem and the conditions
theorem point_P_in_fourth_quadrant :
  let P : ℝ × ℝ := (Real.tan 549, Real.cos 549) in
  P.1 > 0 ∧ P.2 < 0 →
  P ∈ {q | q.1 > 0 ∧ q.2 < 0} :=
by
  sorry -- Proof placeholder

end point_P_in_fourth_quadrant_l123_123565


namespace sqrt_fraction_l123_123629

theorem sqrt_fraction {a b c : ℝ}
  (h1 : a = Real.sqrt 27)
  (h2 : b = Real.sqrt 243)
  (h3 : c = Real.sqrt 48) :
  (a + b) / c = 3 := by
  sorry

end sqrt_fraction_l123_123629


namespace angle_triple_supplementary_l123_123251

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l123_123251


namespace impossible_equal_distribution_l123_123185

def total_glasses : ℕ := 2018
def initial_contents (n : ℕ) : ℕ :=
  if n = 0 then 2 else 1

def khryusha_step (A B : ℚ) : ℚ × ℚ :=
  let avg := (A + B) / 2 in (avg, avg)

theorem impossible_equal_distribution :
  ¬ ∃ (final_contents : fin total_glasses → ℚ),
    (∀ i, initial_contents i ∈ {1, 2}) ∧
    (∑ i : fin total_glasses, initial_contents i = 2019) ∧
    (∀ step : ℕ, ∀ (i j : fin total_glasses),
      khryusha_step (initial_contents i) (initial_contents j) = (final_contents i, final_contents j)) ∧
    (∀ i : fin total_glasses, final_contents i = 2019 / 2018) := sorry

end impossible_equal_distribution_l123_123185


namespace find_sum_l123_123661

-- Definitions of simple interest calculations
def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

-- Problem conditions
def time := 3 : ℝ
def rate (R : ℝ) := R : ℝ
def SI₁(P R : ℝ) : ℝ := simple_interest P R time
def SI₂(P R : ℝ) : ℝ := simple_interest P (R + 2) time

-- Equivalent proof problem
theorem find_sum (R : ℝ) (h : SI₂ P R - SI₁ P R = 360) : P = 6000 :=
by sorry

end find_sum_l123_123661


namespace wind_velocity_proof_l123_123210

noncomputable def find_velocity (k : ℝ) (A₁ P₁ V₁ A₂ P₂ : ℝ) : ℝ :=
  let k₁ := P₁ / (A₁ * V₁^2)
  if k = k₁ then Real.sqrt (P₂ / (k * A₂)) else 0

theorem wind_velocity_proof :
  let A₁ := 12
  let P₁ := 4.8
  let V₁ := 20
  let A₂ := 48
  let P₂ := 60
  let expected_velocity := 35
  let k := 0.001
  find_velocity k A₁ P₁ V₁ A₂ P₂ = expected_velocity := 
by
  unfold find_velocity
  norm_num [A₁, P₁, V₁, A₂, P₂, expected_velocity, k]
  apply congr_arg Real.sqrt
  norm_num
  sorry

end wind_velocity_proof_l123_123210


namespace triangle_area_formula_l123_123308

def base := 18
def height := 6
def expected_area := 54

theorem triangle_area_formula (b h : ℕ) : (1 / 2 : ℝ) * (b * h) = 54 := by
  sorry

end triangle_area_formula_l123_123308


namespace minimal_value_of_roots_is_12_l123_123124

noncomputable def g (x : ℝ) : ℝ := x^4 - 10 * x^3 + 35 * x^2 - 50 * x + 24

def are_roots (x1 x2 x3 x4 : ℝ) : Prop := 
  g x1 = 0 ∧ g x2 = 0 ∧ g x3 = 0 ∧ g x4 = 0

def minimal_value (w1 w2 w3 w4 : ℝ) : ℝ :=
  min (|w1 * w2 - w3 * w4|) 
    (min (|w1 * w3 - w2 * w4|) 
      (min (|w1 * w4 - w2 * w3|) 
        (min (|w2 * w3 - w1 * w4|) 
          (min (|w2 * w4 - w1 * w3|) 
            |w3 * w4 - w1 * w2|))))

theorem minimal_value_of_roots_is_12 : 
  ∃ (w1 w2 w3 w4 : ℝ), are_roots w1 w2 w3 w4 ∧ minimal_value w1 w2 w3 w4 = 12 :=
by
  sorry

end minimal_value_of_roots_is_12_l123_123124


namespace theta_value_l123_123137

noncomputable def a (θ : ℝ) : ℝ × ℝ := (3/2, Real.sin θ)
noncomputable def b (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 1/3)

theorem theta_value (θ : ℝ) (hθ : 0 < θ ∧ θ < Real.pi / 2) 
  (haparallel : a θ ≠ (λ k:ℝ, (k * (a θ).1, k * (a θ).2)) (b θ).1) : 
  θ = Real.pi / 4 :=
sorry

end theta_value_l123_123137


namespace smallest_A_is_144_l123_123231

noncomputable def smallest_A (B : ℕ) := B * 28 + 4

theorem smallest_A_is_144 :
  ∃ (B : ℕ), smallest_A B = 144 ∧ ∀ (B' : ℕ), B' * 28 + 4 < 144 → false :=
by
  sorry

end smallest_A_is_144_l123_123231


namespace games_needed_for_winner_l123_123144

theorem games_needed_for_winner (n : ℕ) (hn : n = 20) : 
    let games := (n - 1) in
    games = 19 :=
by {
    rw hn,
    have h : n - 1 = 20 - 1 := by rw hn,
    exact h,
}

end games_needed_for_winner_l123_123144


namespace sequence_bound_l123_123443

def sequence (a : ℕ → ℝ) : Prop :=
  a 0 = 1 ∧ a 1 = 2 ∧ ∀ n, n ≥ 1 → a (n + 1) = a n + 1 / a n

theorem sequence_bound (a : ℕ → ℝ) (h : sequence a) : 63 < a 2004 ∧ a 2004 < 78 :=
  sorry

end sequence_bound_l123_123443


namespace kylie_daisies_l123_123997

theorem kylie_daisies :
  ∀ (initial_daisies sister_daisies final_daisies daisies_given_to_mother total_daisies : ℕ),
    initial_daisies = 5 →
    sister_daisies = 9 →
    final_daisies = 7 →
    total_daisies = initial_daisies + sister_daisies →
    daisies_given_to_mother = total_daisies - final_daisies →
    daisies_given_to_mother * 2 = total_daisies :=
by
  intros initial_daisies sister_daisies final_daisies daisies_given_to_mother total_daisies h1 h2 h3 h4 h5
  sorry

end kylie_daisies_l123_123997


namespace ArcherInGoldenArmorProof_l123_123927

-- Definitions of the problem
variables (soldiers : Nat) (archers soldiersInGolden : Nat) 
variables (soldiersInBlack archersInGolden archersInBlack swordsmenInGolden swordsmenInBlack : Nat)
variables (truthfulSwordsmenInBlack lyingArchersInBlack lyingSwordsmenInGold truthfulArchersInGold : Nat)
variables (yesToGold yesToArcher yesToMonday : Nat)

-- Given conditions
def ProblemStatement : Prop :=
  soldiers = 55 ∧
  yesToGold = 44 ∧
  yesToArcher = 33 ∧
  yesToMonday = 22 ∧
  soldiers = archers + (soldiers - archers) ∧
  soldiers = soldiersInGolden + soldiersInBlack ∧
  archers = archersInGolden + archersInBlack ∧
  soldiersInGolden = archersInGolden + swordsmenInGolden ∧
  soldiersInBlack = archersInBlack + swordsmenInBlack ∧
  truthfulSwordsmenInBlack = swordsmenInBlack ∧
  lyingArchersInBlack = archersInBlack ∧
  lyingSwordsmenInGold = swordsmenInGolden ∧
  truthfulArchersInGold = archersInGolden ∧
  yesToGold = truthfulArchersInGold + (swordsmenInGolden + archersInBlack) ∧
  yesToArcher = truthfulArchersInGold + lyingSwordsmenInGold

-- Conclusion
def Conclusion : Prop :=
  archersInGolden = 22

-- Proof statement
theorem ArcherInGoldenArmorProof : ProblemStatement → Conclusion :=
by
  sorry

end ArcherInGoldenArmorProof_l123_123927


namespace intersecting_lines_l123_123238

theorem intersecting_lines {c d : ℝ} 
  (h₁ : 12 = 2 * 4 + c) 
  (h₂ : 12 = -4 + d) : 
  c + d = 20 := 
sorry

end intersecting_lines_l123_123238


namespace shortest_altitude_l123_123216

/-!
  Prove that the shortest altitude of a right triangle with sides 9, 12, and 15 is 7.2.
-/

theorem shortest_altitude (a b c : ℕ) (h : a^2 + b^2 = c^2) (ha : a = 9) (hb : b = 12) (hc : c = 15) :
  7.2 ≤ a ∧ 7.2 ≤ b ∧ 7.2 ≤ (2 * (a * b) / c) := 
sorry

end shortest_altitude_l123_123216


namespace angle_triple_supplement_l123_123269

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l123_123269


namespace bug_probability_l123_123324

/-- Define the structure of a cube and the bug's moves. -/
structure Cube :=
(vertices : Fin 8)
(start : Vertex)
(move : Fin 8 → Vertex → Vertex)

def probability_valid_path (cube : Cube) : ℝ :=
let total_paths := 3^8 in
let hamiltonian_cycles := 12 in
hamiltonian_cycles / total_paths

theorem bug_probability (cube : Cube) :
  cube.start = cube.move 8 cube.start →
  (\Sum i in Finset.univ, ∃ hv : Vertex, cube.move i hv ⟹ cube.move (8 - i) hv)  →
  probability_valid_path cube = 4 / 2187 := by
  sorry

end bug_probability_l123_123324


namespace solution_set_sequence_l123_123026

theorem solution_set_sequence {a_n : ℕ → ℝ} (h : ∀ n, a_n = 2 * n) : 
  ∀ n, n ∈ set.Ioo 0 1 -> a_n = 2 * n := 
by
  sorry

end solution_set_sequence_l123_123026


namespace cone_prism_volume_ratio_l123_123341

theorem cone_prism_volume_ratio {w h : ℝ} (h_pos : h > 0) (w_pos : w > 0) :
  let r := w in
  let V_cone := (1 / 3) * π * r^2 * h in
  let V_prism := (2 * w * w * h) in
  (V_cone / V_prism) = (π / 6) :=
by
  -- variable definitions
  let r := w
  let V_cone := (1 / 3) * π * r^2 * h
  let V_prism := 2 * w^2 * h
  -- final equality to prove
  have : V_cone / V_prism = (π / 6) := sorry
  exact this

end cone_prism_volume_ratio_l123_123341


namespace relation_between_a_b_c_l123_123460

noncomputable def a : ℝ := (Real.log (3 / Real.pi))^2
noncomputable def b : ℝ := (Real.log 2)^2
noncomputable def c : ℝ := Real.exp (2 / Real.exp 1 * Real.log 2)

theorem relation_between_a_b_c : a < b ∧ b < c := by
  have h0 : 0 < Real.log (3 / Real.pi) := sorry
  have h1 : Real.log (3 / Real.pi) < Real.log 2 := sorry
  have h2 : a = (Real.log (3 / Real.pi))^2 := sorry
  have h3 : b = (Real.log 2)^2 := sorry
  have h4 : 0 < Real.log 2 := sorry
  have h5 : 2 < Real.exp 1 := sorry
  have h6 : c = Real.exp (2 / Real.exp 1 * Real.log 2) := sorry
  have h7 : 2 / Real.exp 1 < 1 := sorry
  have h8 : c > 1 := sorry
  have h9 : 0 < b < 1 := sorry
  have h10 : a < b := by
    rw [h2, h3]
    exact Real.pow_lt_pow_of_lt_left h0 h1 (Real.log (3 / Real.pi)).nonneg
  have h11 : b < c := by
    rw [h3, h6]
    exact sorry

  exact ⟨h10, h11⟩

end relation_between_a_b_c_l123_123460


namespace value_ratio_l123_123981

def digit_place_value (num : Int) (place : Int) : Float :=
  num * (10 : Float) ^ place

constant digit_9_in_74982_1035_place_value : Float :=
  digit_place_value 9 2 -- 9 in the hundreds place

constant digit_3_in_74982_1035_place_value : Float :=
  digit_place_value 3 (-3) -- 3 in the thousandths place

theorem value_ratio (d9 d3 : Float) :
  d9 = digit_place_value 9 2 → d3 = digit_place_value 3 (-3) →
  d9 / d3 = 100000 := by
  sorry

end value_ratio_l123_123981


namespace midpoint_bisector_property_l123_123716

-- Definition of our problem in Lean
def Midpoints (cube : Type) := Set (cube × cube)

noncomputable def cube_edge_midpoints (cube : Type) : Midpoints cube := sorry

theorem midpoint_bisector_property (cube : Type) (M : Midpoints cube) 
    (hM : M = cube_edge_midpoints cube) 
    (P Q : M) : ∃ R S ∈ M, ∀ x y : M, (x ≠ P ∧ x ≠ Q ∧ y ≠ P ∧ y ≠ Q) → 
    (∀ x y : M, (x, y) ∈ Plane.PerpendicularBisector P Q → (R ∈ x ∧ S ∈ y)) :=
sorry

end midpoint_bisector_property_l123_123716


namespace num_archers_golden_armor_proof_l123_123965
noncomputable section

structure Soldier :=
  (is_archer : Bool)
  (is_golden : Bool)
  (tells_truth : Bool)

def count_soldiers (soldiers : List Soldier) : Nat :=
  soldiers.length

def count_truthful_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => s.tells_truth)).count q

def count_lying_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => ¬s.tells_truth)).count q

def num_archers_golden_armor (soldiers : List Soldier) : Nat :=
  (soldiers.filter (λ s => s.is_archer ∧ s.is_golden)).length

theorem num_archers_golden_armor_proof (soldiers : List Soldier)
  (h1 : count_soldiers soldiers = 55)
  (h2 : count_truthful_responses soldiers (λ s => s.is_golden) + 
        count_lying_responses soldiers (λ s => s.is_golden) = 44)
  (h3 : count_truthful_responses soldiers (λ s => s.is_archer) + 
        count_lying_responses soldiers (λ s => s.is_archer) = 33)
  (h4 : count_truthful_responses soldiers (λ s => true) + 
        count_lying_responses soldiers (λ s => false) = 22) :
  num_archers_golden_armor soldiers = 22 := by
  sorry

end num_archers_golden_armor_proof_l123_123965


namespace sum_valid_two_digit_integers_l123_123290

theorem sum_valid_two_digit_integers : 
  let valid_two_digit_integer (n : ℕ) : Prop :=
    ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 10 * a + b ∧ (a - b) ∣ n ∧ (a * b) ∣ n 
  in (∑ n in finset.filter valid_two_digit_integer (finset.range 100)) = 36 :=
by
  sorry

end sum_valid_two_digit_integers_l123_123290


namespace num_mappings_Q_to_P_l123_123063

-- Define the sets P and Q
noncomputable def P : Type := sorry
noncomputable def Q : Type := {a, b, c}

-- Define the condition
axiom mappings_from_P_to_Q : card ({f : P → Q | true}) = 81

-- Prove the number of mappings from Q to P
theorem num_mappings_Q_to_P : card ({g : Q → P | true}) = 64 := by
  sorry

end num_mappings_Q_to_P_l123_123063


namespace spent_on_music_l123_123456

variable (total_allowance : ℝ) (fraction_music : ℝ)

-- Assuming the conditions
def conditions : Prop :=
  total_allowance = 50 ∧ fraction_music = 3 / 10

-- The proof problem
theorem spent_on_music (h : conditions total_allowance fraction_music) : 
  total_allowance * fraction_music = 15 := by
  cases h with
  | intro h_total h_fraction =>
  sorry

end spent_on_music_l123_123456


namespace angle_triple_supplement_l123_123260

theorem angle_triple_supplement {x : ℝ} (h1 : ∀ y : ℝ, y + (180 - y) = 180) (h2 : x = 3 * (180 - x)) :
  x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123260


namespace compute_dot_product_l123_123121

noncomputable def u : ℝ^3 := sorry -- Define u here
noncomputable def v : ℝ^3 := sorry -- Define v here
noncomputable def z : ℝ^3 := 3 • u - 2 • v
noncomputable def w : ℝ^3 := u × v + 2 • u

axiom orthogonal_unit_vectors (u v : ℝ^3) : (u.dot v = 0) ∧ (u.norm = 1) ∧ (v.norm = 1)
axiom cross_product_relation (w u : ℝ^3) : w × u = 2 • v + z

theorem compute_dot_product : u.dot (v × w) = 1 :=
  by
    have h1 : (u.dot v = 0) := (orthogonal_unit_vectors u v).1
    have h2 : (u.norm = 1) := (orthogonal_unit_vectors u v).2.1
    have h3 : (v.norm = 1) := (orthogonal_unit_vectors u v).2.2
    have h4 : w × u = 2 • v + z := cross_product_relation w u
    sorry

end compute_dot_product_l123_123121


namespace binomial_sum_solution_l123_123367

theorem binomial_sum_solution :
  (1 / 2 ^ 1992) * ∑ n in Finset.range (997), (-3) ^ n * Nat.choose 1992 (2 * n) = -1 / 2 :=
by
  sorry

end binomial_sum_solution_l123_123367


namespace tablets_selection_l123_123675

theorem tablets_selection (medA medB : ℕ) (h_medA : medA = 10) (h_medB : medB = 16) :
  ∃ n, (n = 18 ∧ (∀ k, k ≥ n → k - medB ≥ 2 ∧ k - medA ≥ 2)) :=
by
  use 18
  split
  { refl }
  { intros k hk
    split
    { sorry } -- We acknowledge that these details need to be filled in, but this completes our statement
    { sorry }
  }

end tablets_selection_l123_123675


namespace smallest_four_digit_solution_l123_123390

theorem smallest_four_digit_solution :
  ∃ (x : ℕ), 
    1000 ≤ x ∧ x < 10000 ∧ 
    (x % 3 = 1) ∧ 
    (x % 4 = 3) ∧ 
    (x % 13 = 10) ∧ 
    (x % 7 = 3) ∧ 
    x = 1329 := 
begin
  sorry
end

end smallest_four_digit_solution_l123_123390


namespace isosceles_triangle_side_length_l123_123688

/-- A regular hexagon with a side length of 2 forms three isosceles triangles, 
    each with a vertex at the center and base as one of the hexagon's sides.
    Given that the sum of the areas of these triangles equals half the hexagon's area,
    the length of one of the two congruent sides of each isosceles triangle is 2. -/
theorem isosceles_triangle_side_length (s : ℝ) (a_hexagon : ℝ)
    (h_side_length : s = 2)
    (h_area_condition : 3 * (1 / 2) * s * (s / 2) = (1 / 2) * a_hexagon): 
    s = 2 := 
sorry

end isosceles_triangle_side_length_l123_123688


namespace sum_div_9_remainder_l123_123284

theorem sum_div_9_remainder :
  ∑ i in Finset.range 21, i % 9 = 4 :=
  sorry

end sum_div_9_remainder_l123_123284


namespace negative_integer_solution_l123_123602

theorem negative_integer_solution (M : ℤ) (h1 : 2 * M^2 + M = 12) (h2 : M < 0) : M = -4 :=
sorry

end negative_integer_solution_l123_123602


namespace sum_of_eight_numbers_l123_123904

theorem sum_of_eight_numbers (nums : List ℝ) (h_len : nums.length = 8) (h_avg : (nums.sum / 8) = 5.5) : nums.sum = 44 :=
by
  sorry

end sum_of_eight_numbers_l123_123904


namespace train_cross_time_l123_123314

open Real

def cross_time (v1 v2 l1 l2 : ℝ) : ℝ :=
  let relative_speed := (v1 + v2) * (1 / 3600)
  let total_length := l1 + l2
  total_length / relative_speed

theorem train_cross_time : 
  cross_time 90 90 1.10 0.9 = 40 :=
by
  sorry

end train_cross_time_l123_123314


namespace probability_of_event_B_l123_123681

def fair_dice := { n : ℕ // 1 ≤ n ∧ n ≤ 8 }

def event_B (x y : fair_dice) : Prop := x.val = y.val + 2

def total_outcomes : ℕ := 64

def favorable_outcomes : ℕ := 6

theorem probability_of_event_B : (favorable_outcomes : ℚ) / total_outcomes = 3/32 := by
  have h1 : (64 : ℚ) = 8 * 8 := by norm_num
  have h2 : (6 : ℚ) / 64 = 3 / 32 := by norm_num
  sorry

end probability_of_event_B_l123_123681


namespace value_of_x_plus_y_l123_123049

variable {x y : ℝ}

theorem value_of_x_plus_y (h1 : 1 / x + 1 / y = 1) (h2 : 1 / x - 1 / y = 9) : x + y = -1 / 20 := 
sorry

end value_of_x_plus_y_l123_123049


namespace find_p_over_q_at_neg1_l123_123587

noncomputable def p (x : ℝ) : ℝ := (-27 / 8) * x
noncomputable def q (x : ℝ) : ℝ := (x + 5) * (x - 1)

theorem find_p_over_q_at_neg1 : p (-1) / q (-1) = 27 / 64 := by
  -- Skipping the proof
  sorry

end find_p_over_q_at_neg1_l123_123587


namespace stream_speed_fraction_l123_123334

theorem stream_speed_fraction (B S : ℝ) (h1 : B = 3 * S) 
  (h2 : (1 / (B - S)) = 2 * (1 / (B + S))) : (S / B) = 1 / 3 :=
sorry

end stream_speed_fraction_l123_123334


namespace reaction_rate_comparison_l123_123317

theorem reaction_rate_comparison
  (v : ℝ) -- volume of the containers
  (t : ℝ) -- time intervals
  (m_CO2 : ℝ) -- mass of CO2
  (m_H2S : ℝ) -- mass of H2S
  (M_CO2 : ℝ) -- molar mass of CO2
  (M_H2S : ℝ) -- molar mass of H2S
  (same_volume : ∀ x y : ℝ, v x = v y)
  (same_time : ∀ x y : ℝ, t x = t y)
  (mass_CO2 : m_CO2 = 23)
  (mass_H2S : m_H2S = 20)
  (molar_mass_CO2 : M_CO2 = 44)
  (molar_mass_H2S : M_H2S = 34) :
  (m_H2S / M_H2S) > (m_CO2 / M_CO2) :=
  by
    -- proof goes here, which is not required in the problem statement.
    sorry

end reaction_rate_comparison_l123_123317


namespace hyperbola_directrix_l123_123852

-- Assume the definition of focus of the parabola and hyperbola properties.
def parabola_focus (p : ℝ) := (p, 0)

noncomputable def hyperbola_focus_right (a c : ℝ) (h : a > 0) : Prop :=
  c = real.sqrt (a^2 + 3)

theorem hyperbola_directrix (a : ℝ) (h : a > 0) (h_focus : hyperbola_focus_right a 2 h) :
  a^2 = 1 →
  c = 2 →
  x = 1 / 2 :=
begin
  sorry
end

end hyperbola_directrix_l123_123852


namespace last_n_digits_same_l123_123599

def sequence_a : ℕ → ℕ
| 0     := 5
| (n+1) := (sequence_a n) * (sequence_a n)

theorem last_n_digits_same (n : ℕ) : 10^n ∣ (sequence_a (n+1) - sequence_a n) :=
by sorry

end last_n_digits_same_l123_123599


namespace parametric_curve_intersects_itself_l123_123372

-- Given parametric equations
def param_x (t : ℝ) : ℝ := t^2 + 3
def param_y (t : ℝ) : ℝ := t^3 - 6 * t + 4

-- Existential statement for self-intersection
theorem parametric_curve_intersects_itself :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ param_x t1 = param_x t2 ∧ param_y t1 = param_y t2 ∧ param_x t1 = 9 ∧ param_y t1 = 4 :=
sorry

end parametric_curve_intersects_itself_l123_123372


namespace angle_triple_supplement_l123_123267

theorem angle_triple_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by sorry

end angle_triple_supplement_l123_123267


namespace prove_f_prime_eq_three_halves_l123_123463

noncomputable def problem_statement (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x : ℝ, differentiable_at ℝ f x) ∧
  (∀ Δx : ℝ, Δx ≠ 0 → limit (λ Δx, (f (a + 2 * Δx) - f a) / (3 * Δx)) (𝓝 0) 1)

theorem prove_f_prime_eq_three_halves (f : ℝ → ℝ) (a : ℝ) :
  problem_statement f a → deriv f a = 3 / 2 := 
sorry

end prove_f_prime_eq_three_halves_l123_123463


namespace smallest_k_proof_l123_123149

noncomputable def smallest_k : ℕ :=
  45

theorem smallest_k_proof :
  ∀ (A : Finset ℕ), A.card = smallest_k →
    (∃ x y ∈ A, x ≠ y ∧ |Real.sqrt x - Real.sqrt y| < 1) :=
by
  sorry

end smallest_k_proof_l123_123149


namespace log_base2_gt_zero_l123_123031

theorem log_base2_gt_zero : ¬ (∃ x : ℝ, log 2 (3 ^ x + 1) ≤ 0) :=
by
  sorry

end log_base2_gt_zero_l123_123031


namespace angle_inequality_l123_123090

theorem angle_inequality (V A B C D : Type) 
  [trihedral_angle V A B C] 
  [angle_bisector VD (dihedral_angle BVC)] 
  (h1 : \angle AVD < \frac{\pi}{2}) 
  (h2 : \angle AVD = \frac{\pi}{2}) 
  (h3 : \angle AVD > \frac{\pi}{2}) : 
  \frac{1}{2}(\angle AVB + \angle AVC) ≤ \angle AVD :=
sorry

end angle_inequality_l123_123090


namespace isosceles_triangle_l123_123115

-- Definitions and conditions
variables {A B C G : Type}
variables [T : Triangle A B C] -- A, B, C are points forming the triangle ABC
variables [Centroid G T]       -- G is the centroid of triangle ABC
variables {a b c : ℝ}          -- lengths of sides BC, CA, and AB respectively

-- Given conditions
def AB_GC_eq_AC_GB (AB GC AC GB : ℝ) : Prop :=
  AB + GC = AC + GB

-- The theorem to prove
theorem isosceles_triangle (hG : Centroid G T) (a b c : ℝ)
  (h : AB_GC_eq_AC_GB (AB := c) (GC := sqrt (2*a^2 + 2*b^2 - c^2) / 3)
                       (AC := b) (GB := sqrt (2*a^2 + 2*c^2 - b^2) / 3)) :
  b = c :=
sorry

end isosceles_triangle_l123_123115


namespace inequality_1_system_of_inequalities_l123_123184

-- Statement for inequality (1)
theorem inequality_1 (x : ℝ) : 2 - x ≥ (x - 1) / 3 - 1 → x ≤ 2.5 := 
sorry

-- Statement for system of inequalities (2)
theorem system_of_inequalities (x : ℝ) : 
  (5 * x + 1 < 3 * (x - 1)) ∧ ((x + 8) / 5 < (2 * x - 5) / 3 - 1) → false := 
sorry

end inequality_1_system_of_inequalities_l123_123184


namespace sphere_radius_l123_123327

-- Define the radius and height of the cone
def radius_cone : Real := 2
def height_cone : Real := 6

-- Define the volume formulas for the cone and the sphere
def volume_cone (r h : Real) := (1 / 3) * Math.pi * r^2 * h
def volume_sphere (r : Real) := (4 / 3) * Math.pi * r^3

-- Define the target volume of the sphere
def target_volume_sphere (r_cone h_cone : Real) :=
  2 * volume_cone r_cone h_cone

-- The statement we aim to prove
theorem sphere_radius :
  ∃ r_sphere : Real, volume_sphere r_sphere = target_volume_sphere radius_cone height_cone ∧ 
  r_sphere = 2 * Real.cbrt 3 :=
sorry

end sphere_radius_l123_123327


namespace find_price_first_day_l123_123693

-- Define the variables and conditions
def price_first_day (x y : ℤ) : Prop :=
  let rev_day1 := x * y in
  let rev_day2 := (x - 1) * (y + 100) in
  let rev_day3 := (x + 2) * (y - 100) in
  rev_day1 = rev_day2 ∧ rev_day1 = rev_day3 ∧ 
  (100 * x - y = 100) ∧ (100 * x - 2 * y = 200)

-- The goal
theorem find_price_first_day (x y : ℤ) : price_first_day x y → x = 4 := 
by
  sorry

end find_price_first_day_l123_123693


namespace dolls_total_correct_l123_123102

def Jazmin_dolls : Nat := 1209
def Geraldine_dolls : Nat := 2186
def total_dolls : Nat := Jazmin_dolls + Geraldine_dolls

theorem dolls_total_correct : total_dolls = 3395 := by
  sorry

end dolls_total_correct_l123_123102


namespace shortest_altitude_of_right_triangle_l123_123217

-- Define the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Given conditions about the triangle
def right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the area of the triangle
def area (a b : ℕ) : ℝ := (1/2) * a * b

-- Define the altitude
noncomputable def altitude (area : ℝ) (c : ℕ) : ℝ := (2 * area) / c

-- Proving the length of the shortest altitude
theorem shortest_altitude_of_right_triangle 
  (h : ℝ) 
  (ha : a = 9) 
  (hb : b = 12) 
  (hc : c = 15) 
  (rt : right_triangle a b c) : 
  altitude (area a b) c = 7.2 :=
sorry

end shortest_altitude_of_right_triangle_l123_123217


namespace production_difference_l123_123719

theorem production_difference (w t : ℕ) (h1 : w = 3 * t) :
  (w * t) - ((w + 6) * (t - 3)) = 3 * t + 18 :=
by
  sorry

end production_difference_l123_123719


namespace exists_reflection_inside_quadrilateral_l123_123170

variable {P : Type} [metric_space P]

structure Quadrilateral (P : Type) := 
  (A B C D : P)
  (convex : convex {A, B, C, D} P)

def midpoint {P : Type} [metric_space P] (x y : P) : P :=
  (x + y) / 2

def reflection_over_midpoint {P : Type} [metric_space P] (A B : P) : P :=
  B + (B - midpoint A B)

theorem exists_reflection_inside_quadrilateral 
  (Q : Quadrilateral) : 
  ∃ V ∈ {Q.A, Q.B, Q.C, Q.D}, reflection_over_midpoint V (midpoint Q.A Q.B) ∈ convex_hull ℝ {Q.A, Q.B, Q.C, Q.D} :=
sorry

end exists_reflection_inside_quadrilateral_l123_123170


namespace exists_synchronous_set_of_size_2021_l123_123243

def is_synchronous (S : Set ℕ) : Prop :=
  ∀ a b ∈ S, multiset of (nat.digits 10 (a^2)) = multiset of (nat.digits 10 (b^2))

theorem exists_synchronous_set_of_size_2021 :
  ∃ S : Set ℕ, is_synchronous S ∧ S.card = 2021 :=
begin
  let S := {10^2021 + 10^k | k in finset.range 2021},
  use S,
  split,
  { intros a ha b hb,
    rcases ha with ⟨ka, hka, rfl⟩,
    rcases hb with ⟨kb, hkb, rfl⟩,
    have ha_squared := ((10^2021 + 10^ka) ^ 2).digits 10,
    have hb_squared := ((10^2021 + 10^kb) ^ 2).digits 10,
    sorry },
  { exact finset.card_range _ },
end

end exists_synchronous_set_of_size_2021_l123_123243


namespace range_of_PA_l123_123086

theorem range_of_PA
  (A B P : Type)
  (distance : A → B → ℝ)
  (hAB : distance A B = 2)
  (hPA_PB : ∀ P, distance P A + distance P B = 6) :
  ∀ P, 2 ≤ distance P A ∧ distance P A ≤ 4 :=
by
  sorry

end range_of_PA_l123_123086


namespace water_height_volume_proof_l123_123605

noncomputable theory

-- Define the problem conditions (parameters of the tank and water percentage)
def tank_radius : ℝ := 20
def tank_height : ℝ := 120
def water_percentage : ℝ := 0.3

-- Define the correct height form and values
def a : ℝ := 60
def b : ℝ := 6

-- Volume of the cone with given radius and height
def tank_volume (r h : ℝ) : ℝ := (1 / 3) * π * r ^ 2 * h

-- Volume of the water in the tank
def water_volume (V : ℝ) : ℝ := water_percentage * V

-- General form of height for the water given the full tank's volume
def water_height (h : ℝ) (x : ℝ) : ℝ := h * x

-- Proof statement (goal)
theorem water_height_volume_proof :
  let V := tank_volume tank_radius tank_height,
      VW := water_volume V,
      x := VW / V,
      height := water_height tank_height (x^(1 / 3))
  in height = a * (b / 10)^(1 / 3) ∧ (a + b = 66) :=
by
  -- Skipping the proof, as requested
  sorry

end water_height_volume_proof_l123_123605


namespace triple_supplementary_angle_l123_123279

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l123_123279


namespace projection_of_a_on_b_l123_123870

variables (a b : Vector) 
variable ha_length : ‖a‖ = 3
variable hb_length : ‖b‖ = 2 * sqrt 3
variable ha_perp : a.dot (a - b) = 0

theorem projection_of_a_on_b : proj b a = (3 / 4) • b := by
  sorry

end projection_of_a_on_b_l123_123870


namespace average_salary_all_workers_l123_123195

-- Define the given conditions as constants
def num_technicians : ℕ := 7
def avg_salary_technicians : ℕ := 12000

def num_workers_total : ℕ := 21
def num_workers_remaining := num_workers_total - num_technicians
def avg_salary_remaining_workers : ℕ := 6000

-- Define the statement we need to prove
theorem average_salary_all_workers :
  let total_salary_technicians := num_technicians * avg_salary_technicians
  let total_salary_remaining_workers := num_workers_remaining * avg_salary_remaining_workers
  let total_salary_all_workers := total_salary_technicians + total_salary_remaining_workers
  let avg_salary_all_workers := total_salary_all_workers / num_workers_total
  avg_salary_all_workers = 8000 :=
by
  sorry

end average_salary_all_workers_l123_123195


namespace points_eq_l123_123720

-- Definition of the operation 
def star (a b : ℝ) : ℝ := a^2 * b + a * b^2

-- The property we want to prove
theorem points_eq : {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1} =
    {p : ℝ × ℝ | p.1 = 0} ∪ {p : ℝ × ℝ | p.2 = 0} ∪ {p : ℝ × ℝ | p.1 + p.2 = 0} :=
by
  sorry

end points_eq_l123_123720


namespace no_obtuse_triangle_probability_l123_123796

noncomputable def probability_no_obtuse_triangle : ℝ :=
 let θ := 1/2 in
 let prob_A2_A3_given_A0A1 := (3/8) * (3/8) in
 θ * prob_A2_A3_given_A0A1

theorem no_obtuse_triangle_probability :
  probability_no_obtuse_triangle = 9/128 :=
by
  sorry

end no_obtuse_triangle_probability_l123_123796


namespace determine_B_l123_123122

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem determine_B (B : ℕ) (h1 : 0 ≤ B ∧ B ≤ 9) :
  (462720B % 2 = 0) ∧ (digit_sum (462720B) % 3 = 0) ∧ (462720B % 4 = 0) ∧
  (462720B % 5 = 0) ∧ (462720B % 6 = 0) ∧ (462720B % 8 = 0) ∧
  (digit_sum (462720B) % 9 = 0) ↔ B = 6 := 
sorry

end determine_B_l123_123122


namespace find_a1_l123_123370

theorem find_a1 :
  ∃ a1 : ℝ, 
  (∃ a2 : ℝ, a2 = 2 ∧ (∀ n : ℕ, ∀ a (h1 : a n = a), a (n + 1) = 1 / (1 - a))) →
  a1 = 1 / 2 :=
by
  sorry

end find_a1_l123_123370


namespace a_21_eq_1024_l123_123197

-- Define sequences a_n and b_n and their properties
def a_n : ℕ → ℝ
def b_n : ℕ → ℝ 

-- Given conditions
axiom a_1_eq : a_n 1 = 1
axiom b_geom : ∀ n : ℕ, b_n (n+1) = b_n n * b_n n -- Geometric sequence property
axiom seq_relation : ∀ n : ℕ, b_n n = a_n (n + 1) / a_n n
axiom b_10_11 : b_n 10 * b_n 11 = 2

-- The goal to prove
theorem a_21_eq_1024 : a_n 21 = 1024 := by
  sorry

end a_21_eq_1024_l123_123197


namespace find_EG_length_l123_123487

-- Let O be the center of the semicircle and the midpoint of EF
variables (E F G H O V P S : Point)
variables (k : ℝ)
variables (m : Line)
variables [Condition_1 : Cofactor.Rectangle EFGH_Semicircle EF]
variables [Condition_2 : Meets_Line m EFGH_Semicircle_Paths_S]
variables [Condition_3 : Line_Division_Ratio m 1 3]
variables [Condition_4 : EV_Length E V 108]
variables [Condition_5 : EP_Length E P 162]
variables [Condition_6 : VF_Length V F 216]

theorem find_EG_length : ∃ x y : ℝ, EG = 243 * Real.sqrt 2 ∧ Int_add x y = 245 :=
by
  sorry

end find_EG_length_l123_123487


namespace completing_square_eq_sum_l123_123038

theorem completing_square_eq_sum :
  ∃ (a b c : ℤ), a > 0 ∧ (∀ (x : ℝ), 36 * x^2 - 60 * x + 25 = (a * x + b)^2 - c) ∧ a + b + c = 26 :=
by
  sorry

end completing_square_eq_sum_l123_123038


namespace domain_of_g_l123_123368

theorem domain_of_g (x : ℝ) : 
    ∃ (g : ℝ → ℝ),
    g = (λ x, 1 / ⌊x^2 - 8 * x + 18⌋)
    ∧ (∀ x : ℝ, g x ∈ ℝ) := 
begin
  sorry
end

end domain_of_g_l123_123368


namespace function_passes_fixed_point_l123_123857

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1)

theorem function_passes_fixed_point (a : ℝ) (h : a > 0) (h' : a ≠ 1) : f(a, 1) = 1 :=
by
  unfold f
  sorry

end function_passes_fixed_point_l123_123857


namespace three_digit_repeated_digits_percentage_l123_123892

noncomputable def percentage_repeated_digits : ℝ :=
  let total_numbers := 900
  let non_repeated := 9 * 9 * 8
  let repeated := total_numbers - non_repeated
  (repeated / total_numbers) * 100

theorem three_digit_repeated_digits_percentage :
  percentage_repeated_digits = 28.0 := by
  sorry

end three_digit_repeated_digits_percentage_l123_123892


namespace evaluate_absolute_value_l123_123382

theorem evaluate_absolute_value (π : ℝ) (h : π < 5.5) : |5.5 - π| = 5.5 - π :=
by
  sorry

end evaluate_absolute_value_l123_123382


namespace cos_alpha_value_l123_123415

def f (α : Real) : Real :=
  (sin((Real.pi / 2) - α) * cos(-α) * tan(Real.pi + α)) / cos(Real.pi - α)

theorem cos_alpha_value (α : Real) (hα : α > Real.pi ∧ α < 3 * Real.pi / 2)
  (h : f α = 2 * Real.sqrt 5 / 5) : cos α = -Real.sqrt 5 / 5 :=
by
  sorry

end cos_alpha_value_l123_123415


namespace Marty_combinations_l123_123551

def unique_combinations (colors techniques : ℕ) : ℕ :=
  colors * techniques

theorem Marty_combinations :
  unique_combinations 6 5 = 30 := by
  sorry

end Marty_combinations_l123_123551


namespace x_y_sum_cube_proof_l123_123532

noncomputable def x_y_sum_cube (x y : ℝ) : ℝ := x^3 + y^3

theorem x_y_sum_cube_proof (x y : ℝ) (hx : 1 < x) (hy : 1 < y)
  (h_eq : (Real.log x / Real.log 2)^3 + (Real.log y / Real.log 3)^3 = 3 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x_y_sum_cube x y = 307 :=
sorry

end x_y_sum_cube_proof_l123_123532


namespace minimize_abs_diff_l123_123014

-- Define the function f(x) = 2^x + log_2 x
def f (x : ℝ) : ℝ := 2^x + Real.log x / Real.log 2

-- Define the sequence a_n = 0.1n
def a (n : ℕ) : ℝ := 0.1 * n

-- Define the condition to minimize
def cond (n : ℕ) : ℝ := |f (a n) - 2005|

-- State the main theorem
theorem minimize_abs_diff : ∃ n : ℕ, cond n = 110 := sorry

end minimize_abs_diff_l123_123014


namespace expected_sides_general_expected_sides_rectangle_l123_123621

-- General Problem
theorem expected_sides_general (n k : ℕ) : 
  (∀ n k : ℕ, n ≥ 3 → k ≥ 0 → (1:ℝ) / ((k + 1:ℝ)) * (n + 4 * k:ℝ) ≤ (n + 4 * k) / (k + 1)) := 
begin
  sorry
end

-- Specific Problem for Rectangle
theorem expected_sides_rectangle (k : ℕ) :
  (∀ k : ℕ, k ≥ 0 → 4 = (4 + 4 * k) / (k + 1)) := 
begin
  sorry
end

end expected_sides_general_expected_sides_rectangle_l123_123621


namespace calculate_a_plus_b_l123_123462

theorem calculate_a_plus_b (a b : ℝ) (h1 : 3 = a + b / 2) (h2 : 2 = a + b / 4) : a + b = 5 :=
by
  sorry

end calculate_a_plus_b_l123_123462


namespace find_cos_A_and_side_c_l123_123912

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (triangle : a = 3) (side_b : b = 2 * Real.sqrt 6) (angle_relation : B = 2 * A)

theorem find_cos_A_and_side_c (h₁ : a = 3) (h₂ : b = 2 * Real.sqrt 6) (h₃ : B = 2 * A) :
  (cos A = Real.sqrt 6 / 3) ∧ (c = 5) :=
sorry

end find_cos_A_and_side_c_l123_123912


namespace comparison_of_a_b_c_l123_123889

theorem comparison_of_a_b_c (a b c : ℝ) (h_a : a = 0.3^2) (h_b : b = log 2 0.3) (h_c : c = 2^0.3) : 
  b < a ∧ a < c := by
  sorry

end comparison_of_a_b_c_l123_123889


namespace num_both_volunteers_l123_123558

noncomputable def num_parents : Nat := 84
noncomputable def supervise_volunteers : Nat := 25
noncomputable def refreshment_volunteers : Nat := 42
noncomputable def n_neither : ℕ := 42 / 1.5

theorem num_both_volunteers :
  ∃ B : ℕ, supervise_volunteers + refreshment_volunteers - B + n_neither = num_parents ∧ B = 11 :=
by 
  -- We already know the values from the problem, we are formalizing the proof problem here
  use 11
  split
  case left => simp [num_parents, supervise_volunteers, refreshment_volunteers, n_neither]; norm_num
  case right => rfl

end num_both_volunteers_l123_123558


namespace power_div_eq_l123_123244

theorem power_div_eq (a : ℕ) (h : 36 = 6^2) : (6^12 / 36^5) = 36 := by
  sorry

end power_div_eq_l123_123244


namespace solve_wives_l123_123101

noncomputable def name_wives : Prop :=
  ∃ (brown_wife green_wife white_wife smith_wife : String),
  (brown_wife = "Dorothy" ∧ green_wife = "Carol" ∧ white_wife = "Betty" ∧ smith_wife = "Anna") ∧
  let anna_drink := 2 in
  let betty_drink := 3 in
  let carol_drink := 4 in
  let dorothy_drink := 5 in
  let brown_drink := dorothy_drink in
  let green_drink := 2 * carol_drink in
  let white_drink := 3 * betty_drink in
  let smith_drink := 4 * anna_drink in
  dorothy_drink + carol_drink + betty_drink + anna_drink +
  brown_drink + green_drink + white_drink + smith_drink = 44

theorem solve_wives : name_wives :=
sorry

end solve_wives_l123_123101


namespace tank_length_l123_123695

theorem tank_length (W D : ℝ) (cost_per_sq_m total_cost : ℝ) (L : ℝ):
  W = 12 →
  D = 6 →
  cost_per_sq_m = 0.70 →
  total_cost = 520.8 →
  total_cost = cost_per_sq_m * ((2 * (W * D)) + (2 * (L * D)) + (L * W)) →
  L = 25 :=
by
  intros hW hD hCostPerSqM hTotalCost hEquation
  sorry

end tank_length_l123_123695


namespace min_value_of_a_squared_plus_b_squared_l123_123836

-- Problem definition and condition
def is_on_circle (a b : ℝ) : Prop :=
  (a^2 + b^2 - 2*a + 4*b - 20) = 0

-- Theorem statement
theorem min_value_of_a_squared_plus_b_squared (a b : ℝ) (h : is_on_circle a b) :
  a^2 + b^2 = 30 - 10 * Real.sqrt 5 :=
sorry

end min_value_of_a_squared_plus_b_squared_l123_123836


namespace digit_one_more_frequent_l123_123320

def sum_digits (n : ℕ) : ℕ :=
  n.toString.toList.foldl (λ acc c, acc + (c.toNat - '0'.toNat)) 0

def repeated_digit_sum (n : ℕ) : ℕ :=
  let rec sum_digits_recursive (n : ℕ) : ℕ :=
    if n < 10 then n else sum_digits_recursive (sum_digits n)
  sum_digits_recursive n

theorem digit_one_more_frequent (N : ℕ) :
  (count (repeated_digit_sum <$> (List.range (N + 1))) (Eq 1)) =
  (count (repeated_digit_sum <$> (List.range (N + 1))) (Eq 2)) + 1 :=
  sorry

end digit_one_more_frequent_l123_123320


namespace num_solutions_eq_4_l123_123043

-- Define the equation as a proposition
def equation (x y : ℤ) : Prop := x^4 + y^2 = 2 * y + x^2

-- State the theorem to prove the number of solutions
theorem num_solutions_eq_4 : 
  {p : ℤ × ℤ // equation p.1 p.2 }.to_finset.card = 4 :=
by sorry

end num_solutions_eq_4_l123_123043


namespace intercept_sum_abs_eq_three_l123_123856

theorem intercept_sum_abs_eq_three :
  let l_eq := λ x y : ℝ, 2 * x - 5 * y + 10 = 0 in
  let a := (-10 : ℝ) / 2 in -- x-intercept: solve 2x - 5(0) + 10 = 0 -> x = -5
  let b := 10 / (-5) in -- y-intercept: solve 2(0) - 5y + 10 = 0 -> y = 2
  |a + b| = 3 :=
by
  -- computation done here to match the value of |a + b| 
  sorry

end intercept_sum_abs_eq_three_l123_123856


namespace angle_bisector_is_base_l123_123196

theorem angle_bisector_is_base {A B C K : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space K] 
  (h_iso : dist A B = dist A C) (h_bisect : ∀ P, angle P B C = angle P K B / 2) 
  (h_divides : dist B K = dist K C) : dist B C = dist B K :=
sorry

end angle_bisector_is_base_l123_123196


namespace probability_neither_mix_l123_123660

variable (total_buyers cake_buyers muffin_buyers both_buyers : ℕ)
variable (h_total : total_buyers = 100)
variable (h_cake : cake_buyers = 50)
variable (h_muffin : muffin_buyers = 40)
variable (h_both : both_buyers = 15)

theorem probability_neither_mix : 
  let neither_mix := total_buyers - (cake_buyers + muffin_buyers - both_buyers) in
  let probability := (neither_mix : ℝ) / (total_buyers : ℝ) in
  probability = 0.25 := by
sorry

end probability_neither_mix_l123_123660


namespace minimum_squares_one_mark_minimum_squares_three_marks_l123_123839

noncomputable def min_marked_squares (n : ℕ) (k : ℕ) := 
  inf {m : ℕ | ∀ (i j : ℕ), i + 2 < n → j + 2 < n → 
    ∃ (x y : ℕ), x ≤ i + 2  ∧ y ≤ j + 2 ∧ x + 2 ≥ i ∧ y + 2 ≥ j ∧ m (x, y)}

theorem minimum_squares_one_mark : min_marked_squares 8 1 = 4 := 
by { sorry }

theorem minimum_squares_three_marks : min_marked_squares 8 3 = 16 := 
by { sorry }

end minimum_squares_one_mark_minimum_squares_three_marks_l123_123839


namespace fifth_equation_in_sequence_l123_123011

theorem fifth_equation_in_sequence :
  (1^3 + 2^3 = 3^2) ∧
  (1^3 + 2^3 + 3^3 = 6^2) ∧
  (1^3 + 2^3 + 3^3 + 4^3 = 10^2) →
  1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 :=
by
  intro h,
  sorry

end fifth_equation_in_sequence_l123_123011


namespace no_obtuse_triangle_probability_l123_123800

noncomputable def probability_no_obtuse_triangle : ℝ :=
 let θ := 1/2 in
 let prob_A2_A3_given_A0A1 := (3/8) * (3/8) in
 θ * prob_A2_A3_given_A0A1

theorem no_obtuse_triangle_probability :
  probability_no_obtuse_triangle = 9/128 :=
by
  sorry

end no_obtuse_triangle_probability_l123_123800


namespace DQ_bisector_of_ABC_is_parallelogram_l123_123517

section
variables {A B C D M N Q : Type*}
variables [parallelogram A B C D] -- Assumption that ABCD is a parallelogram
variables (M : point_on_segment A B) -- M is on the segment AB
variables (N : point_on_segment B C) -- N is on the segment BC
variables {AM CN : ℝ} (h_AM_CN: AM = CN ∧ AM ≠ 0) -- AM = CN ≠ 0
variables (Q : intersection_point (line_through A N) (line_through C M)) -- Q is the intersection of lines AN and CM

-- The property we want to prove: DQ bisects angle ADC
noncomputable
def angle_bisector_DQ : Prop :=
  is_angle_bisector (line_through D Q) (angle A D C)

theorem DQ_bisector_of_ABC_is_parallelogram :
  angle_bisector_DQ A B C D M N Q AM CN h_AM_CN :=
sorry
end

end DQ_bisector_of_ABC_is_parallelogram_l123_123517


namespace matrix_problem_l123_123059

variable {n : Type} [Field n] [AddGroup n]
variable (B : Matrix n n n)
variable I : Matrix n n n

-- Assumption: The matrix B has an inverse
variable [Invertible B]

-- Assumption: (B - 3 * I)(B - 5 * I) = 0
axiom h : (B - 3 • I) ⬝ (B - 5 • I) = 0

-- Target: Prove B + 10 * B⁻¹ = 8 * I
theorem matrix_problem : B + 10 • ⅟ B = 8 • I := 
by sorry

end matrix_problem_l123_123059


namespace normal_transversals_pass_through_orthocenter_l123_123242

-- Given an orthocentric tetrahedron A1A2A3A4
variables {A1 A2 A3 A4 : Type*}

-- Assume A1, A2, A3 and A4 are points in a 3D space forming an orthocentric tetrahedron.
noncomputable def orthocentric_tetrahedron (A1 A2 A3 A4 : Type*) :=
  ∃ (B1 B2 B3 B4 : Type*), parallelepiped_with_equal_edges_and_rhombic_faces A1 A2 A3 A4 B1 B2 B3 B4 ∧
  (A1A2 ⊥ A3A4) ∧ (A1A3 ⊥ A2A4) ∧ (A1A4 ⊥ A2A3)

-- Prove that the normal transversals of the opposite edges pass through the orthocenter
theorem normal_transversals_pass_through_orthocenter (A1 A2 A3 A4 : Type*)
  (h_tetra : orthocentric_tetrahedron A1 A2 A3 A4) :
  ∃ H : Type*, orthocenter A1 A2 A3 A4 H ∧
  ∀ (edge : Type*), is_edge A1 A2 A3 A4 edge → normal_transversal edge ⊤ H :=
sorry

end normal_transversals_pass_through_orthocenter_l123_123242


namespace largest_m_divisibility_property_l123_123440

theorem largest_m_divisibility_property :
  ∃ m : ℕ, m = 499 ∧ ∀ S : set ℕ, S ⊆ { n | n ∈ finset.range 1000 + 1} → finset.card S = 501 →
    ∃ x y ∈ S, x ≠ y ∧ (x ∣ y ∨ y ∣ x) :=
by sorry

end largest_m_divisibility_property_l123_123440


namespace part_one_part_two_l123_123861

def f (x : ℝ) := |x + 2|

theorem part_one (x : ℝ) : 2 * f x < 4 - |x - 1| ↔ -7 / 3 < x ∧ x < -1 := sorry

theorem part_two (m n : ℝ) (x a : ℝ) (h : m > 0) (h : n > 0) (h : m + n = 1) :
  (|x - a| - f x ≤ 1/m + 1/n) ↔ (-6 ≤ a ∧ a ≤ 2) := sorry

end part_one_part_two_l123_123861


namespace probability_gather_info_both_workshops_l123_123489

theorem probability_gather_info_both_workshops :
  ∃ (p : ℚ), p = 56 / 62 :=
by
  sorry

end probability_gather_info_both_workshops_l123_123489


namespace no_obtuse_triangle_probability_eq_l123_123786

noncomputable def probability_no_obtuse_triangle : ℝ :=
  let num_points := 4
  -- Condition (1): Four points are chosen uniformly at random on a circle.
  -- Condition (2): An obtuse angle occurs if the minor arc exceeds π/2.
  9 / 64

theorem no_obtuse_triangle_probability_eq :
  let num_points := 4
  ∀ (points : Fin num_points → ℝ), 
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∠ (points i, points j, points k) < π / 2) →
  probability_no_obtuse_triangle = 9 / 64 :=
by
  sorry

end no_obtuse_triangle_probability_eq_l123_123786


namespace no_obtuse_triangle_l123_123788

-- Conditions
def points_on_circle_uniformly_at_random (n : ℕ) : Prop :=
  ∀ i < n, ∀ j < n, i ≠ j → ∃ θ_ij : ℝ, 0 ≤ θ_ij ∧ θ_ij ≤ π

-- Theorem statement
theorem no_obtuse_triangle (hn : points_on_circle_uniformly_at_random 4) :
  let p := \left(\frac{1}{2}\right)^6 in
  p = \frac{1}{64} :=
sorry

end no_obtuse_triangle_l123_123788


namespace rectangle_c_on_farthest_right_l123_123768

def Rectangle := (Red : ℕ) (Blue : ℕ) (Green : ℕ) (Yellow : ℕ)

def A : Rectangle := (2, 1, 5, 8)
def B : Rectangle := (1, 0, 4, 7)
def C : Rectangle := (3, 6, 2, 9)
def D : Rectangle := (6, 4, 3, 5)
def E : Rectangle := (8, 3, 6, 0)

def row_of_rectangles := [A, B, C, D, E]

noncomputable def unique_yellow_values (rects : List Rectangle) : List ℕ :=
List.filter (λ y, List.count y (rects.map (λ r, r.Yellow)) = 1) (rects.map (λ r, r.Yellow))

theorem rectangle_c_on_farthest_right :
  row_of_rectangles.getLast! = C :=
by
  -- The proof steps would go here...
  sorry

end rectangle_c_on_farthest_right_l123_123768


namespace sample_size_l123_123213

-- Definitions from Conditions
def ratio_young := 7
def ratio_middle_aged := 5
def ratio_elderly := 3
def young_employees_sample := 14

-- Target statement to prove
theorem sample_size (total_sample_size : ℕ) :
  (ratio_young : ℚ) / (ratio_young + ratio_middle_aged + ratio_elderly) * total_sample_size = young_employees_sample → 
  total_sample_size = 30 :=
by
  intro h
  have : total_sample_size = 14 * (ratio_young + ratio_middle_aged + ratio_elderly) / ratio_young := by 
  { rw [h, show ratio_young / (ratio_young + ratio_middle_aged + ratio_elderly) * total_sample_size = young_employees_sample, from h] }
  sorry -- Proof to be filled in

end sample_size_l123_123213


namespace f_neg_a_l123_123471

variable (x a : ℝ)
def f (x : ℝ) : ℝ := x^2 * sin x + 1

theorem f_neg_a (h : f a = 11) : f (-a) = -9 :=
by
  -- proof skipped
  sorry

end f_neg_a_l123_123471


namespace four_points_no_obtuse_triangle_l123_123778

noncomputable def probability_no_obtuse_triangle : ℝ :=
1 / 64

theorem four_points_no_obtuse_triangle (A B C D : circle) :
  (∀ (P Q : circle) (PQ_angle : ℝ), PQ_angle < π/2) → 
  probability_no_obtuse_triangle = 1 / 64 :=
sorry

end four_points_no_obtuse_triangle_l123_123778


namespace zero_point_neg_x₀_l123_123190

-- Define odd function property
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define zero point condition for the function
def is_zero_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = Real.exp x₀

-- The main theorem to be proved
theorem zero_point_neg_x₀ (f : ℝ → ℝ) (x₀ : ℝ)
  (h_odd : is_odd_function f)
  (h_zero : is_zero_point f x₀) :
  f (-x₀) * Real.exp x₀ + 1 = 0 :=
sorry

end zero_point_neg_x₀_l123_123190


namespace youngest_child_age_l123_123220

theorem youngest_child_age (x : ℝ) (h : x + (x + 1) + (x + 2) + (x + 3) = 12) : x = 1.5 :=
by sorry

end youngest_child_age_l123_123220


namespace perimeter_of_figure_l123_123682

-- Given conditions
def side_length : Nat := 2
def num_horizontal_segments : Nat := 16
def num_vertical_segments : Nat := 10

-- Define a function to calculate the perimeter based on the given conditions
def calculate_perimeter (side_length : Nat) (num_horizontal_segments : Nat) (num_vertical_segments : Nat) : Nat :=
  (num_horizontal_segments * side_length) + (num_vertical_segments * side_length)

-- Statement to be proved
theorem perimeter_of_figure : calculate_perimeter side_length num_horizontal_segments num_vertical_segments = 52 :=
by
  -- The proof would go here
  sorry

end perimeter_of_figure_l123_123682


namespace tips_fraction_of_income_correct_l123_123305

variable {S : ℝ} (T : ℝ)

def tips_fraction_of_income (T S : ℝ) (hT : T = (3 / 4) * S) : Prop :=
  T / (S + T) = 3 / 7

theorem tips_fraction_of_income_correct (S : ℝ) (hS : S ≠ 0) (hT : T = (3 / 4) * S) : 
  tips_fraction_of_income T S hT :=
begin
  sorry
end

end tips_fraction_of_income_correct_l123_123305


namespace distance_relationship_l123_123004

noncomputable def plane_parallel (α β : Type) : Prop := sorry
noncomputable def line_in_plane (m : Type) (α : Type) : Prop := sorry
noncomputable def point_on_line (A : Type) (m : Type) : Prop := sorry
noncomputable def distance (A B : Type) : ℝ := sorry
noncomputable def distance_point_to_line (A : Type) (n : Type) : ℝ := sorry
noncomputable def distance_between_lines (m n : Type) : ℝ := sorry

variables (α β m n A B : Type)
variables (a b c : ℝ)

axiom plane_parallel_condition : plane_parallel α β
axiom line_m_in_alpha : line_in_plane m α
axiom line_n_in_beta : line_in_plane n β
axiom point_A_on_m : point_on_line A m
axiom point_B_on_n : point_on_line B n
axiom distance_a : a = distance A B
axiom distance_b : b = distance_point_to_line A n
axiom distance_c : c = distance_between_lines m n

theorem distance_relationship : c ≤ b ∧ b ≤ a := by
  sorry

end distance_relationship_l123_123004


namespace total_dots_not_visible_l123_123773

theorem total_dots_not_visible :
  let total_dots := 4 * 21
  let visible_sum := 1 + 2 + 3 + 3 + 4 + 5 + 5 + 6
  total_dots - visible_sum = 55 :=
by
  sorry

end total_dots_not_visible_l123_123773


namespace angle_OQP_is_right_angle_l123_123525

open EuclideanGeometry

variables {Ω : Type*} [metric_space Ω] [normed_group Ω] [normed_space ℝ Ω] [inner_product_space ℝ Ω]

def circle (O : Ω) (r : ℝ) : set Ω := {P | dist P O = r}

def cyclic_quad (A B C D : Ω) : Prop :=
∃ O : Ω, ∃ r : ℝ, circle O r = {P | ∃ (θ : ℝ), unit_vector (rotate θ (landmark_basis O r)) ∈ {A, B, C, D}}

def intersection_of_diagonals (P A C B D : Ω) : Prop :=
∃ A' C' B' D' : Ω, segment A C ∩ segment B D = {P}

noncomputable def circumcircle_intersection (P Q A B C D : Ω) : Prop :=
∃ Ω₁ Ω₂ : set Ω, Ω₁ = circumcircle A B P ∧ Ω₂ = circumcircle C D P ∧ Q ∈ Ω₁ ∩ Ω₂

theorem angle_OQP_is_right_angle (O A B C D P Q : Ω) 
    (H1 : circle O 1 ⊆ {A, B, C, D})
    (H2 : cyclic_quad A B C D)
    (H3 : intersection_of_diagonals P A C B D)
    (H4 : circumcircle_intersection P Q A B C D) :
      angle O Q P = π / 2 :=
by
  sorry

end angle_OQP_is_right_angle_l123_123525


namespace company_x_total_employees_l123_123084

-- Definitions for conditions
def initial_percentage : ℝ := 0.60
def Q2_hiring_males : ℕ := 30
def Q2_new_percentage : ℝ := 0.57
def Q3_hiring_females : ℕ := 50
def Q3_new_percentage : ℝ := 0.62
def Q4_hiring_males : ℕ := 40
def Q4_hiring_females : ℕ := 10
def Q4_new_percentage : ℝ := 0.58

-- Statement of the proof problem
theorem company_x_total_employees :
  ∃ (E : ℕ) (F : ℕ), 
    (F = initial_percentage * E ∧
     F = Q2_new_percentage * (E + Q2_hiring_males) ∧
     F + Q3_hiring_females = Q3_new_percentage * (E + Q2_hiring_males + Q3_hiring_females) ∧
     F + Q3_hiring_females + Q4_hiring_females = Q4_new_percentage * (E + Q2_hiring_males + Q3_hiring_females + Q4_hiring_males + Q4_hiring_females)) →
    E + Q2_hiring_males + Q3_hiring_females + Q4_hiring_males + Q4_hiring_females = 700 :=
sorry

end company_x_total_employees_l123_123084


namespace sum_inequality_l123_123135

theorem sum_inequality (n : ℕ) (h_n : n ≥ 2) (x : ℕ → ℝ) 
  (hx : ∀ k, k ≤ n → 0 ≤ x k ∧ x k ≤ 1) :
  ∑ k in Finset.range n, ∑ j in Finset.range k, k * x k * x j ≤ 
  (n - 1) / 3 * ∑ k in Finset.range n, k * x k :=
sorry

end sum_inequality_l123_123135


namespace isosceles_of_equal_medians_l123_123169

theorem isosceles_of_equal_medians {A B C D E M : Type} [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] [inhabited M] 
  (triangle_ABC : A) (med_AD : D) (med_BE : E) (centroid_M : M)
  (AD_eq_BE : med_AD = med_BE) : 
  is_isosceles_triangle A B C :=
sorry

end isosceles_of_equal_medians_l123_123169


namespace isosceles_triangle_ef_length_l123_123492

theorem isosceles_triangle_ef_length :
  ∀ (D E F J : Type) [EuclideanGeometry] (DE DF EF : ℝ),
    is_isosceles_triangle D E F ∧
    distance D E = 6 ∧
    distance D F = 6 ∧
    altitude_from_vertex D F E J ∧
    2 * distance J E = distance J F →
      distance E F = 6 * real.sqrt 2 :=
by sorry

end isosceles_triangle_ef_length_l123_123492


namespace tangent_line_equation_at_x_eq_1_l123_123585

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_equation_at_x_eq_1 :
  tangent_line_eq_to_curve f 1 = Eq {x - y - 1 = 0} :=
sorry

end tangent_line_equation_at_x_eq_1_l123_123585


namespace plastic_bag_estimation_l123_123234

theorem plastic_bag_estimation (a b c d e f : ℕ) (class_size : ℕ) (h1 : a = 33) 
  (h2 : b = 25) (h3 : c = 28) (h4 : d = 26) (h5 : e = 25) (h6 : f = 31) (h_class_size : class_size = 45) :
  let count := a + b + c + d + e + f
  let average := count / 6
  average * class_size = 1260 := by
{ 
  sorry 
}

end plastic_bag_estimation_l123_123234


namespace sandwich_cost_90_cents_l123_123994

def sandwich_cost (bread_cost ham_cost cheese_cost : ℕ) : ℕ :=
  2 * bread_cost + ham_cost + cheese_cost

theorem sandwich_cost_90_cents :
  sandwich_cost 15 25 35 = 90 :=
by
  -- Proof goes here
  sorry

end sandwich_cost_90_cents_l123_123994


namespace num_bricks_required_l123_123329

/-- A courtyard is 28 meters long and 13 meters wide and is to be paved
with bricks of dimensions 22 cm by 12 cm. Prove that 13788 bricks are required. -/
theorem num_bricks_required 
  (courtyard_length : ℝ := 28) 
  (courtyard_width : ℝ := 13) 
  (brick_length_cm : ℝ := 22) 
  (brick_width_cm : ℝ := 12)
  (one_meter_in_cm : ℝ := 100) 
  (area_courtyard : ℝ := courtyard_length * courtyard_width)
  (brick_length : ℝ := brick_length_cm / one_meter_in_cm)
  (brick_width : ℝ := brick_width_cm / one_meter_in_cm)
  (area_brick : ℝ := brick_length * brick_width)
  (num_bricks : ℕ := ((area_courtyard / area_brick).ceil).to_nat) :
  num_bricks = 13788 := 
by
  sorry

end num_bricks_required_l123_123329


namespace zero_sum_bound_l123_123431

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x ^ 2 * log x - a

theorem zero_sum_bound (a x1 x2 : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) (h_order : x1 < x2) :
  1 < x1 + x2 ∧ x1 + x2 < 2 / sqrt exp 1 :=
sorry

end zero_sum_bound_l123_123431


namespace sqrt_expression_domain_l123_123470

theorem sqrt_expression_domain (x : ℝ) : (∃ y : ℝ, y = sqrt x - 2) ↔ 0 ≤ x :=
by 
  sorry

end sqrt_expression_domain_l123_123470


namespace repeating_decimal_sum_l123_123294

theorem repeating_decimal_sum (x : ℝ) (h : x = 0.35353535) : 
  let num := 35
  let denom := 99
  ((num / Real.gcd num denom) + (denom / Real.gcd num denom)) = 134 :=
by
  sorry

end repeating_decimal_sum_l123_123294


namespace geometric_sequence_ratio_l123_123873

theorem geometric_sequence_ratio (a b c q : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ b + c - a = x * q ∧ c + a - b = x * q^2 ∧ a + b - c = x * q^3 ∧ a + b + c = x) →
  q^3 + q^2 + q = 1 :=
by
  sorry

end geometric_sequence_ratio_l123_123873


namespace garden_area_increase_l123_123687

theorem garden_area_increase (l w : ℝ) (h_lw : l = 80) (h_ww : w = 20)
  (h_perimeter : 2 * (l + w) = 200) :
  let original_area := l * w in
  let original_perimeter := 2 * (l + w) in
  let square_side := original_perimeter / 4 in
  let square_area := square_side ^ 2 in
  let square_area_increase := square_area - original_area in
  let new_width := original_perimeter / 6 in
  let new_length := 2 * new_width in
  let new_rect_area := new_length * new_width in
  let new_rect_area_increase := new_rect_area - original_area in
  square_area_increase = 900 ∧ (new_rect_area - original_area ≈ 622.22) ∧ (square_area_increase > new_rect_area_increase) :=
by
  sorry

end garden_area_increase_l123_123687


namespace Z_in_fourth_quadrant_l123_123498

noncomputable def Z : ℂ := (2 / (3 - (complex.I : ℂ))) + (complex.I : ℂ) ^ 3

theorem Z_in_fourth_quadrant : (0 < Z.re) ∧ (Z.im < 0) := by
  -- Definitions expressed in conditions
  have h₁ : Z = (3/5 : ℂ) - (4/5 : ℂ) * complex.I := by sorry
  
  -- Use h₁ to show the real part is positive and imaginary part is negative
  sorry

end Z_in_fourth_quadrant_l123_123498


namespace solve_fraction_l123_123062

theorem solve_fraction (x : ℝ) (h₁ : x^2 - 1 = 0) (h₂ : (x - 2) * (x + 1) ≠ 0) : x = 1 := 
sorry

end solve_fraction_l123_123062


namespace range_of_a_l123_123845

theorem range_of_a (a : ℝ) (P M : set ℝ) (hP : P = {x | x^2 ≤ 1}) (hM : M = {a}) (hPM : P ∪ M = P) :
  a ∈ set.Icc (-1:ℝ) 1 := 
sorry

end range_of_a_l123_123845


namespace sum_primes_between_20_and_40_l123_123635

theorem sum_primes_between_20_and_40 :
  23 + 29 + 31 + 37 = 120 :=
by
  sorry

end sum_primes_between_20_and_40_l123_123635


namespace distinct_tower_heights_l123_123183

def num_distinct_heights (num_bricks : ℕ) (height1 height2 height3 : ℕ) : ℕ :=
let min_height := num_bricks * height1 in
let max_height := num_bricks * height3 in
let step_size := gcd (height2 - height1) (height3 - height1) in
(max_height - min_height) / step_size + 1

theorem distinct_tower_heights :
  num_distinct_heights 62 3 11 17 = 435 := by
  sorry

end distinct_tower_heights_l123_123183


namespace ratio_a_to_c_l123_123528

variables {a b c d : ℚ}

theorem ratio_a_to_c
  (h1 : a / b = 5 / 2)
  (h2 : c / d = 4 / 1)
  (h3 : d / b = 3 / 10) :
  a / c = 25 / 12 :=
sorry

end ratio_a_to_c_l123_123528


namespace sum_of_eight_numbers_l123_123901

variable (avg : ℝ) (n : ℕ)

-- Given condition
def average_eq_of_eight_numbers : avg = 5.5 := rfl
def number_of_items_eq_eight : n = 8 := rfl

-- Theorem to prove
theorem sum_of_eight_numbers (h1 : average_eq_of_eight_numbers avg)
                             (h2 : number_of_items_eq_eight n) :
  avg * n = 44 :=
by
  -- Proof will be inserted here
  sorry

end sum_of_eight_numbers_l123_123901


namespace number_of_archers_in_golden_armor_l123_123972

-- Define the problem context
structure Soldier where
  is_archer : Bool
  wears_golden_armor : Bool

def truth_teller (s : Soldier) (is_black_armor : Bool) : Bool :=
  if s.is_archer then s.wears_golden_armor = is_black_armor
  else s.wears_golden_armor ≠ is_black_armor

def response (s : Soldier) (question : String) (is_black_armor : Bool) : Bool :=
  match question with
  | "Are you wearing golden armor?" => if truth_teller s is_black_armor then s.wears_golden_armor else ¬s.wears_golden_armor
  | "Are you an archer?" => if truth_teller s is_black_armor then s.is_archer else ¬s.is_archer
  | "Is today Monday?" => if truth_teller s is_black_armor then True else False -- An assumption that today not being Monday means False
  | _ => False

-- Problem condition setup
def total_soldiers : Nat := 55
def golden_armor_yes : Nat := 44
def archer_yes : Nat := 33
def monday_yes : Nat := 22

-- Define the main theorem
theorem number_of_archers_in_golden_armor :
  ∃ l : List Soldier,
    l.length = total_soldiers ∧
    l.countp (λ s => response s "Are you wearing golden armor?" True) = golden_armor_yes ∧
    l.countp (λ s => response s "Are you an archer?" True) = archer_yes ∧
    l.countp (λ s => response s "Is today Monday?" True) = monday_yes ∧
    l.countp (λ s => s.is_archer ∧ s.wears_golden_armor) = 22 :=
sorry

end number_of_archers_in_golden_armor_l123_123972


namespace sum_mod_1_to_20_l123_123283

theorem sum_mod_1_to_20 :
  (∑ i in finset.range 21, i) % 9 = 3 :=
by
  sorry

end sum_mod_1_to_20_l123_123283


namespace larry_jogs_each_day_l123_123110

theorem larry_jogs_each_day
  (days_first_week : ℕ)
  (days_second_week : ℕ)
  (total_hours_two_weeks : ℕ)
  (total_days := days_first_week + days_second_week)
  (total_minutes := total_hours_two_weeks * 60):
  days_first_week = 3 →
  days_second_week = 5 →
  total_hours_two_weeks = 4 →
  total_minutes / total_days = 30 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end larry_jogs_each_day_l123_123110


namespace ordered_systems_of_pos_rationals_l123_123724

def is_integer (q : ℚ) : Prop := ∃ n : ℤ, q = n

theorem ordered_systems_of_pos_rationals (x y z : ℚ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxy : is_integer (x + 1 / y))
  (hyz : is_integer (y + 1 / z))
  (hzx : is_integer (z + 1 / x)) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 1 ∧ y = 1/2 ∧ z = 2) ∨
  (x = 1/2 ∧ y = 2 ∧ z = 1) ∨
  (x = 2 ∧ y = 1 ∧ z = 1/2) ∨
  (x = 1/2 ∧ y = 2/3 ∧ z = 3) ∨
  (x = 3 ∧ y = 1/2 ∧ z = 2/3) ∨
  (x = 2/3 ∧ y = 3 ∧ z = 1/2) ∨
  (x = 1/3 ∧ y = 3/2 ∧ z = 2) ∨
  (x = 2 ∧ y = 1/3 ∧ z = 3/2) ∨
  (x = 3/2 ∧ y = 2 ∧ z = 1/3) :=
sorry

end ordered_systems_of_pos_rationals_l123_123724


namespace angle_triple_supplementary_l123_123246

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l123_123246


namespace find_derivative_at_1_l123_123005

-- Define the function f(x) = x^α
noncomputable def f (x : ℝ) (α : ℝ) := x^α

-- Define the conditions
lemma condition1 (α : ℝ) : f 2 α = 8 :=
begin
  rw [f, pow_eq_pow], -- Replace definition of f and use that 2^α = 8
  norm_num, -- Simplify the numerical part
end

-- Statement of the problem: Prove that f'(1) = 3, given the conditions
theorem find_derivative_at_1 :
  ∃ α : ℝ, (∃ h: f 2 α = 8, deriv (λ x, f x α) 1 = 3) :=
begin
  use 3, -- Candidate α
  use (condition1 3).eq_symm, -- Use the condition that 2^3 = 8
  by {
    -- Compute the derivative
    have f_deriv : deriv (λ x, f x 3) = λ x, 3 * x ^ (3 - 1) := by sorry,
    rw f_deriv,
    norm_num,
  }
end

end find_derivative_at_1_l123_123005


namespace number_of_dolls_is_18_l123_123222

def total_toys : ℕ := 24
def fraction_action_figures : ℚ := 1 / 4
def number_action_figures : ℕ := (fraction_action_figures * total_toys).to_nat
def number_dolls : ℕ := total_toys - number_action_figures

theorem number_of_dolls_is_18 :
  number_dolls = 18 :=
by
  sorry

end number_of_dolls_is_18_l123_123222


namespace constant_term_in_expansion_l123_123418

noncomputable def integral_value : ℝ :=
  ∫ x in 0..2, 3 * x^2

theorem constant_term_in_expansion :
  integral_value = 8 ∧
  let n := integral_value in
  (x - (1 / (2 * x)))^n → (0 * (x - (1 / (2 * x)))^n)  = 35 / 8 :=
by
  sorry

end constant_term_in_expansion_l123_123418


namespace angle_triple_supplementary_l123_123248

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l123_123248


namespace daphne_two_visits_in_365_days_l123_123718

def visits_in_days (d1 d2 : ℕ) (days : ℕ) : ℕ :=
  days / Nat.lcm d1 d2

theorem daphne_two_visits_in_365_days :
  let days := 365
  let lcm_all := Nat.lcm 4 (Nat.lcm 6 (Nat.lcm 8 10))
  (visits_in_days 4 6 lcm_all + 
   visits_in_days 4 8 lcm_all + 
   visits_in_days 4 10 lcm_all + 
   visits_in_days 6 8 lcm_all + 
   visits_in_days 6 10 lcm_all + 
   visits_in_days 8 10 lcm_all) * 
   (days / lcm_all) = 129 :=
by
  sorry

end daphne_two_visits_in_365_days_l123_123718


namespace solution_of_fractional_equation_l123_123907

theorem solution_of_fractional_equation :
  (∃ x, x ≠ 3 ∧ (x / (x - 3) - 2 = (m - 1) / (x - 3))) → m = 4 := by
  sorry

end solution_of_fractional_equation_l123_123907


namespace probability_no_obtuse_triangle_is_9_over_64_l123_123815

noncomputable def probability_no_obtuse_triangle (A0 A1 A2 A3 : ℝ) : ℝ :=
  -- Define the probabilistic model according to the problem conditions
  -- Assuming A0, A1, A2, and A3 are positions of points on the circle parametrized by angles in radians
  let θ := real.angle.lower_half (A1 - A0) in
  let prob_A2 := (π - θ) / (2 * π) in
  let prob_A3 := (π - θ) / (2 * π) in
  (1 / 2) * (prob_A2 * prob_A3)

theorem probability_no_obtuse_triangle_is_9_over_64 :
  probability_no_obtuse_triangle A0 A1 A2 A3 = 9 / 64 :=
by sorry

end probability_no_obtuse_triangle_is_9_over_64_l123_123815


namespace projection_of_b_on_a_l123_123850

noncomputable section

variables {V : Type*} [inner_product_space ℝ V]

-- Define the unit vectors c and d
variables (c d a b : V)
variables (h1 : ∥c∥ = 1) (h2 : ∥d∥ = 1)
variables (angle_cd : real.angle (c, d) = real.pi / 3)

-- Define vectors a and b
def vec_a := c + 3 • d
def vec_b := 2 • c

-- Projection calculation
theorem projection_of_b_on_a :
  vector_projection vec_a vec_b = (5 / real.sqrt 13) • vec_a :=
sorry

end projection_of_b_on_a_l123_123850


namespace range_of_m_l123_123015

noncomputable def f (x m : ℝ) := 2^|2 * x - m|

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, 2 ≤ x → x ≤ y → f x m ≤ f y m) → m ≤ 4 := by
  sorry

end range_of_m_l123_123015


namespace cookies_ratio_l123_123139

theorem cookies_ratio (total_cookies sells_mr_stone brock_buys left_cookies katy_buys : ℕ)
  (h1 : total_cookies = 5 * 12)
  (h2 : sells_mr_stone = 2 * 12)
  (h3 : brock_buys = 7)
  (h4 : left_cookies = 15)
  (h5 : total_cookies - sells_mr_stone - brock_buys - left_cookies = katy_buys) :
  katy_buys / brock_buys = 2 :=
by sorry

end cookies_ratio_l123_123139


namespace locus_of_midpoints_of_KL_l123_123714

theorem locus_of_midpoints_of_KL (K L M : Point) (A B C D : Point) (sq : square A B C D)
  (K_on_AB : K ∈ line_segment A B) (L_on_BC : L ∈ line_segment B C) (M_on_CD : M ∈ line_segment C D)
  (equilateral_KLM : equilateral_triangle K L M) :
  ∃ line : Line, (line_parallel line (line_segment_mem AD)) ∧ (line_passes_through_midpoint_of_BE line) :=
sorry

end locus_of_midpoints_of_KL_l123_123714


namespace graph_does_not_pass_through_quadrant_III_l123_123202

-- Define the linear function
def linear_function (x : ℝ) : ℝ := - (2 / 3) * x + 3

-- Define the conditions for the quadrants
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0
def in_quadrant_III (x y : ℝ) : Prop := x < 0 ∧ y < 0
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- The theorem that the graph does not pass through Quadrant III
theorem graph_does_not_pass_through_quadrant_III :
  ∀ x : ℝ, ¬ in_quadrant_III x (linear_function x) :=
by
  sorry

end graph_does_not_pass_through_quadrant_III_l123_123202


namespace not_true_n_gt_24_l123_123125

theorem not_true_n_gt_24 (n : ℕ) (h : 1/3 + 1/4 + 1/6 + 1/n = 1) : n ≤ 24 := 
by
  -- Placeholder for the proof
  sorry

end not_true_n_gt_24_l123_123125


namespace bank_exceeds_1600cents_in_9_days_after_Sunday_l123_123109

theorem bank_exceeds_1600cents_in_9_days_after_Sunday
  (a : ℕ)
  (r : ℕ)
  (initial_deposit : ℕ)
  (days_after_sunday : ℕ)
  (geometric_series : ℕ -> ℕ)
  (sum_geometric_series : ℕ -> ℕ)
  (geo_series_definition : ∀(n : ℕ), geometric_series n = 5 * 2^n)
  (sum_geo_series_definition : ∀(n : ℕ), sum_geometric_series n = 5 * (2^n - 1))
  (exceeds_condition : ∀(n : ℕ), sum_geometric_series n > 1600 -> n >= 9) :
  days_after_sunday = 9 → a = 5 → r = 2 → initial_deposit = 5 → days_after_sunday = 9 → geometric_series 1 = 10 → sum_geometric_series 9 > 1600 :=
by sorry

end bank_exceeds_1600cents_in_9_days_after_Sunday_l123_123109


namespace find_m_l123_123524

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m * x = 0}

def C_UA : Set ℕ := {1, 2}

theorem find_m (m : ℝ) (hA : A m = {0, 3}) (hCUA : U \ A m = C_UA) : m = -3 := 
  sorry

end find_m_l123_123524


namespace fg_of_2_eq_225_l123_123416

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem fg_of_2_eq_225 : f (g 2) = 225 := by
  sorry

end fg_of_2_eq_225_l123_123416


namespace quilt_patch_cost_l123_123510

-- Definitions of the conditions
def length : ℕ := 16
def width : ℕ := 20
def patch_area : ℕ := 4
def cost_first_10 : ℕ := 10
def cost_after_10 : ℕ := 5
def num_first_patches : ℕ := 10

-- Define the calculations based on the problem conditions
def quilt_area : ℕ := length * width
def total_patches : ℕ := quilt_area / patch_area
def cost_first : ℕ := num_first_patches * cost_first_10
def remaining_patches : ℕ := total_patches - num_first_patches
def cost_remaining : ℕ := remaining_patches * cost_after_10
def total_cost : ℕ := cost_first + cost_remaining

-- Statement of the proof problem
theorem quilt_patch_cost : total_cost = 450 := by
  -- Placeholder for the proof
  sorry

end quilt_patch_cost_l123_123510


namespace find_b_l123_123201

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem find_b
  (b : ℝ)
  (p1 p2 : ℝ × ℝ)
  (midpoint_eq : midpoint p1 p2 = (4, 7))
  (line_passing : ∀ x y, (x, y) = (4, 7) → x + y = b) :
  b = 11 :=
sorry

end find_b_l123_123201


namespace faye_books_l123_123734

theorem faye_books (initial_books given_away final_books books_bought: ℕ) 
  (h1 : initial_books = 34) 
  (h2 : given_away = 3) 
  (h3 : final_books = 79) 
  (h4 : final_books = initial_books - given_away + books_bought) : 
  books_bought = 48 := 
by 
  sorry

end faye_books_l123_123734


namespace archers_in_golden_armor_count_l123_123947

theorem archers_in_golden_armor_count : 
  ∃ (archers_golden : ℕ), 
    let total_soldiers := 55 in
    let golden_armor_affirmative := 44 in
    let archer_affirmative := 33 in
    let monday_affirmative := 22 in
    let number_archers_golden := archers_golden in
    (number_archers_golden = 22) ∧ 
    ∃ (archer : ℕ) (swordsman : ℕ) (golden : ℕ) (black : ℕ),
      archer + swordsman = total_soldiers ∧
      golden + black = total_soldiers ∧
      golden + black = total_soldiers ∧
      (swordsman * golden + archer * golden = golden_armor_affirmative) ∧
      (swordsman * archer + swordsman * golden = archer_affirmative) ∧
      ((!monday_affirmative ∧ swordsman * golden) + (!monday_affirmative ∧ archer * black) = monday_affirmative)
:= sorry

end archers_in_golden_armor_count_l123_123947


namespace closest_point_on_parabola_to_line_l123_123735

theorem closest_point_on_parabola_to_line :
  ∃ (x y : ℝ), (x^2 = (1 / 4) * y) ∧ y = 1 ∧ y = 4 * x ^ 2 ∧ x = 1 / 2 :=
by
  use 1 / 2, 1
  split
  { 
    have : (1 / 2)^2 = 1 / 4 := by norm_num,
    rw this, norm_num }
  split
  { norm_num }
  split
  { field_simp, ring }
  { norm_num }
          

end closest_point_on_parabola_to_line_l123_123735


namespace cyclic_quadrilateral_BXCY_l123_123535

open EuclideanGeometry

noncomputable def midpoint (a b : Point) : Point := (1/2 : ℝ) • a + (1/2 : ℝ) • b

noncomputable def is_reflection (P Q R : Point) : Prop := midpoint P Q = R

theorem cyclic_quadrilateral_BXCY
  {A B C O D E M X Y : Point}
  (h_triangle : ¬ colinear A B C)
  (h_O_circumcenter : is_circumcenter O A B C)
  (h_AB_neq_AC : A ≠ B ∧ A ≠ C)
  (h_D_on_angle_bisector : is_angle_bisector_of A D B C)
  (h_M_midpoint_BC : midpoint B C = M)
  (h_E_reflection : is_reflection D E M)
  (h_X_on_AO : on_line_through X A O)
  (h_Y_on_AD : on_line_through Y A D)
  (h_X_perpendicular_BC : is_perpendicular X D B C)
  (h_Y_perpendicular_BC : is_perpendicular Y E B C) :
  cyclic_quad B X C Y :=
sorry

end cyclic_quadrilateral_BXCY_l123_123535


namespace average_marks_math_chem_l123_123312

variables (M P C : ℕ)

theorem average_marks_math_chem :
  (M + P = 20) → (C = P + 20) → (M + C) / 2 = 20 := 
by
  sorry

end average_marks_math_chem_l123_123312


namespace odd_function_at_zero_l123_123828

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_at_zero (f : ℝ → ℝ) (h : is_odd_function f) : f 0 = 0 :=
by
  -- assume the definitions but leave the proof steps and focus on the final conclusion
  sorry

end odd_function_at_zero_l123_123828


namespace rectangle_ratio_of_pentomino_l123_123726

-- Conditions
def is_pentomino_shape (s : Set Point) : Prop := sorry
def is_circumscribed_rectangle (r : Rectangle) (s : Set Point) : Prop := sorry

-- Math proof problem
theorem rectangle_ratio_of_pentomino (r : Rectangle) (s : Set Point) (h1 : is_pentomino_shape s) (h2 : is_circumscribed_rectangle r s) :
  r.length / r.width = 2 :=
sorry

end rectangle_ratio_of_pentomino_l123_123726


namespace michael_remaining_money_l123_123140

variables (m b n : ℝ) (h1 : (1 : ℝ) / 3 * m = 1 / 2 * n * b) (h2 : 5 = m / 15)

theorem michael_remaining_money : m - (2 / 3 * m + m / 15) = 4 / 15 * m :=
by
  have hb1 : 2 / 3 * m = (2 * m) / 3 := by ring
  have hb2 : m / 15 = (1 * m) / 15 := by ring
  rw [hb1, hb2]
  sorry

end michael_remaining_money_l123_123140


namespace three_digit_integers_product_36_l123_123883

theorem three_digit_integers_product_36 : 
  ∃ (num_digits : ℕ), num_digits = 21 ∧ 
    ∀ (a b c : ℕ), 
      (1 ≤ a ∧ a ≤ 9) ∧ 
      (1 ≤ b ∧ b ≤ 9) ∧ 
      (1 ≤ c ∧ c ≤ 9) ∧ 
      (a * b * c = 36) → 
      num_digits = 21 :=
sorry

end three_digit_integers_product_36_l123_123883


namespace range_of_m_for_union_range_of_m_for_intersection_l123_123413

open Set

variable {A B : Set ℝ}
variable {m : ℝ}

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def set_B (m : ℝ) : Set ℝ := {x | m - 4 ≤ x ∧ x ≤ 3m + 2}

theorem range_of_m_for_union (h : set_A ∪ set_B m = set_B m) : 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

theorem range_of_m_for_intersection (h : set_A ∩ set_B m = set_B m) : m < -3 :=
by
  sorry

end range_of_m_for_union_range_of_m_for_intersection_l123_123413


namespace min_distance_PQ_l123_123546

-- Define the parameters for point P on the given function
def onGraphP (P : ℝ × ℝ) : Prop :=
  ∃ x : ℝ, P = (x, -real.sqrt (4 - (x - 1)^2))

-- Define the condition for point Q with parameter a
def Q (a : ℝ) : ℝ × ℝ :=
  (2 * a, a - 3)

-- Define the condition that Q lies on the line x - 2y - 6 = 0
def onLineQ (a : ℝ) : Prop :=
  let Q_coord := Q a in
  Q_coord.1 - 2 * Q_coord.2 - 6 = 0

-- Define the distance from the center (1,0) to the line x - 2y - 6 = 0
def distance_from_center_to_line : ℝ :=
  abs (1 * 1 + 0 * (-2) - 6) / real.sqrt (1^2 + (-2)^2)

-- State the theorem regarding the minimum distance |PQ|
theorem min_distance_PQ : ∃ P Q a, onGraphP P ∧ onLineQ a ∧ (∃ PQ_min : ℝ, PQ_min = real.sqrt 5 - 2):
  sorry

end min_distance_PQ_l123_123546


namespace digit_sum_square_l123_123131

theorem digit_sum_square (n : ℕ) (hn : 0 < n) :
  let A := (4 * (10 ^ (2 * n) - 1)) / 9
  let B := (8 * (10 ^ n - 1)) / 9
  ∃ k : ℕ, A + 2 * B + 4 = k ^ 2 := 
by
  sorry

end digit_sum_square_l123_123131


namespace initial_water_amount_l123_123683

def daily_evaporation_rate : ℝ := 0.04
def total_days : ℕ := 10
def percentage_evaporation : ℝ := 1.6 / 100
def total_evaporation : ℝ := daily_evaporation_rate * total_days

theorem initial_water_amount (W : ℝ) (h1 : daily_evaporation_rate * total_days = total_evaporation)
  (h2 : total_evaporation = percentage_evaporation * W) : W = 25 :=
by
  -- Proof omitted
  sorry

end initial_water_amount_l123_123683


namespace find_AF_l123_123075

-- Define the given conditions
structure Rectangle (A B C D F : Type) :=
(AB BC : ℝ)
(CBF_angle : ℝ)

variables {A B C D F : Type} [Rectangle A B C D F]

-- Hypotheses derived from the problem conditions
axiom h1 : Rectangle AB 18 BC 12 CBF_angle (30 : ℝ)

-- The objective statement to be proved
theorem find_AF : ∃ (AF : ℝ), AF = real.sqrt (336 - 96 * real.sqrt 3) :=
by sorry

end find_AF_l123_123075


namespace tetrahedron_colorings_l123_123380

-- Define the problem conditions
def tetrahedron_faces : ℕ := 4
def colors : List String := ["red", "white", "blue", "yellow"]

-- The theorem statement
theorem tetrahedron_colorings :
  ∃ n : ℕ, n = 35 ∧ ∀ (c : List String), c.length = tetrahedron_faces → c ⊆ colors →
  (true) := -- Placeholder (you can replace this condition with the appropriate condition)
by
  -- The proof is omitted with 'sorry' as instructed
  sorry

end tetrahedron_colorings_l123_123380


namespace probability_no_obtuse_triangle_l123_123806

namespace CirclePoints

noncomputable def no_obtuse_triangle_probability : ℝ := 
  let p := 1/64 in
    p

theorem probability_no_obtuse_triangle (X : ℕ → ℝ) (hcirc : ∀ n, 0 ≤ X n ∧ X n < 2 * π) (hpoints : (∀ n m, n ≠ m → X n ≠ X m)) :
  no_obtuse_triangle_probability = 1/64 :=
sorry

end CirclePoints

end probability_no_obtuse_triangle_l123_123806


namespace domain_of_f_f_is_odd_l123_123018

def f (x : ℝ) : ℝ := (1 / (3 ^ x - 1)) + 1 / 2

theorem domain_of_f :
  ∀ x : ℝ, x ≠ 0 ↔ (f x ∈ Set.univ) :=
by sorry

theorem f_is_odd :
  ∀ x : ℝ, f (-x) = -f x :=
by sorry

end domain_of_f_f_is_odd_l123_123018


namespace triangle_sum_OA1_AA1_equals_1_l123_123096

theorem triangle_sum_OA1_AA1_equals_1
  (A B C O A1 B1 C1 : Point)
  (hA1 : collinear A O A1) (hB1 : collinear B O B1) (hC1 : collinear C O C1)
  (ints_A : intersects A O A1 (side BC)) (ints_B : intersects B O B1 (side AC)) (ints_C : intersects C O C1 (side AB))
  : (dist O A1 / dist A A1) + (dist O B1 / dist B B1) + (dist O C1 / dist C C1) = 1 := sorry

end triangle_sum_OA1_AA1_equals_1_l123_123096


namespace find_angle_AMB_l123_123093

-- Define the given angles
def angle_A : ℝ := 45
def angle_B : ℝ := 15

-- Define the configuration of the triangle and the point M
noncomputable def angle_ABC (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :=
  ∃ (M : Type) [metric_space M], 
        M = point_on_extension A C ∧ CM = 2 * AC

-- Main theorem to prove
theorem find_angle_AMB {A B C : Type} [metric_space A] [metric_space B] [metric_space C] 
  (h₁ : \(\angle A = 45^\circ\))
  (h₂: \(\angle B = 15^\circ\))
  (h₃ : \(\angle C = 180^\circ - \(\angle A + \(\angle B\))) 
  (h₄: point_on_extension)
  (h₅: CM = 2 * AC) :
  \(\angle AMB = 75^\circ\) :=
 sorry

end find_angle_AMB_l123_123093


namespace num_archers_golden_armor_proof_l123_123962
noncomputable section

structure Soldier :=
  (is_archer : Bool)
  (is_golden : Bool)
  (tells_truth : Bool)

def count_soldiers (soldiers : List Soldier) : Nat :=
  soldiers.length

def count_truthful_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => s.tells_truth)).count q

def count_lying_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => ¬s.tells_truth)).count q

def num_archers_golden_armor (soldiers : List Soldier) : Nat :=
  (soldiers.filter (λ s => s.is_archer ∧ s.is_golden)).length

theorem num_archers_golden_armor_proof (soldiers : List Soldier)
  (h1 : count_soldiers soldiers = 55)
  (h2 : count_truthful_responses soldiers (λ s => s.is_golden) + 
        count_lying_responses soldiers (λ s => s.is_golden) = 44)
  (h3 : count_truthful_responses soldiers (λ s => s.is_archer) + 
        count_lying_responses soldiers (λ s => s.is_archer) = 33)
  (h4 : count_truthful_responses soldiers (λ s => true) + 
        count_lying_responses soldiers (λ s => false) = 22) :
  num_archers_golden_armor soldiers = 22 := by
  sorry

end num_archers_golden_armor_proof_l123_123962


namespace hot_dog_remainder_l123_123464

theorem hot_dog_remainder : 35252983 % 6 = 1 :=
by
  sorry

end hot_dog_remainder_l123_123464


namespace exists_n_for_binomial_congruence_l123_123520

theorem exists_n_for_binomial_congruence 
  (p : ℕ) (a k : ℕ) (prime_p : Nat.Prime p) 
  (positive_a : a > 0) (positive_k : k > 0)
  (h1 : p^a < k) (h2 : k < 2 * p^a) : 
  ∃ n : ℕ, n < p^(2 * a) ∧ (Nat.choose n k) % p^a = n % p^a ∧ n % p^a = k % p^a :=
by
  sorry

end exists_n_for_binomial_congruence_l123_123520


namespace original_expenditure_beginning_month_l123_123705

theorem original_expenditure_beginning_month (A E : ℝ)
  (h1 : E = 35 * A)
  (h2 : E + 84 = 42 * (A - 1))
  (h3 : E + 124 = 37 * (A + 1))
  (h4 : E + 154 = 40 * (A + 1)) :
  E = 630 := 
sorry

end original_expenditure_beginning_month_l123_123705


namespace encyclopedia_total_pages_l123_123562

theorem encyclopedia_total_pages (chapters : ℕ) (pages_per_chapter : ℕ) (h1 : chapters = 12) (h2 : pages_per_chapter = 782) : chapters * pages_per_chapter = 9384 :=
by
  rw [h1, h2]
  norm_num
  exact rfl

end encyclopedia_total_pages_l123_123562


namespace ratio_of_levels_beaten_l123_123730

theorem ratio_of_levels_beaten (total_levels beaten_levels not_beaten_levels : ℕ) 
  (h_total : total_levels = 32) (h_beaten : beaten_levels = 24)
  (h_not_beaten : not_beaten_levels = total_levels - beaten_levels) :
  beaten_levels / (total_levels - beaten_levels) = 3 :=
by 
  rw [h_total, h_beaten, h_not_beaten]
  norm_num
  sorry

end ratio_of_levels_beaten_l123_123730


namespace birds_cannot_all_end_up_on_same_tree_l123_123156

theorem birds_cannot_all_end_up_on_same_tree : 
  (∀ (n : Nat) (h : n = 6), 
    ∃ (initial_positions : Fin n → Nat), 
      (∀ i : Fin n, initial_positions i = 1) ∧ 
      (∀ t : Nat, 
         ∃ (bird_positions : Fin n → Fin n → Nat), 
           (∀ i : Fin n, bird_positions 0 i = initial_positions i) ∧ 
           (∀ t' : Nat, 
              (∀ i : Fin n, bird_positions (t'+1) i = 
                bird_positions t' ((i + 1) % n) + bird_positions t' ((i - 1 + n) % n))) ∧ 
           (∀ i, ∃ t, bird_positions t i = n))
  → False := 
begin
  sorry
end

end birds_cannot_all_end_up_on_same_tree_l123_123156


namespace exists_point_distance_greater_l123_123560

noncomputable theory
open Classical
open Real

theorem exists_point_distance_greater 
  (P : Finset (ℝ × ℝ)) (L : Finset (ℝ × ℝ × ℝ)): 
  ∃ A : ℝ × ℝ, 
    ∀ p ∈ P, ∀ l ∈ L, 
      dist A p > distance_to_line A l :=
by
  let O : ℝ × ℝ := (0, 0) -- Arbitrary initial point
  let r : ℝ := max_dist O P -- Selecting radius containing all points and lines
  -- Construction of required point A with required distance property
  sorry

-- Helper function to compute the maximum distance from O to points in P
def max_dist (O : ℝ × ℝ) (P : Finset(ℝ × ℝ)) : ℝ :=
  P.fold (λ acc p, max acc (dist O p)) 0

-- Distance from a point to a line (assuming the line is given in the form Ax + By + C = 0)
def distance_to_line (A : ℝ × ℝ) (l : ℝ × ℝ × ℝ) : ℝ :=
  let '(x, y) := A in
  let '(a, b, c) := l in
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

end exists_point_distance_greater_l123_123560


namespace nigel_has_more_money_than_twice_initial_l123_123146

noncomputable def nigel_money_more_than_twice_original (initial_win : ℕ) (mother_gift : ℕ) (final_give_away : ℕ) (final_amount : ℕ) 
  (twice_initial : ℕ) (more_than_twice_initial : ℕ) (unknown_initial_give : ℕ) : Prop :=
initial_win = 45 ∧
mother_gift = 80 ∧
final_give_away = 25 ∧
final_amount = initial_win + mother_gift - final_give_away ∧
twice_initial = 2 * initial_win ∧
more_than_twice_initial = final_amount - twice_initial ∧
more_than_twice_initial = 10

theorem nigel_has_more_money_than_twice_initial : 
  ∃ initial_win mother_gift final_give_away final_amount twice_initial more_than_twice_initial unknown_initial_give,
  nigel_money_more_than_twice_original initial_win mother_gift final_give_away final_amount twice_initial more_than_twice_initial unknown_initial_give :=
begin
  use 45,
  use 80,
  use 25,
  use 100,
  use 90,
  use 10,
  use 0, -- This value is not relevant in final conditional expressions.
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { refl },
end

end nigel_has_more_money_than_twice_initial_l123_123146


namespace obtuse_triangle_count_l123_123600

def is_obtuse_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 < c^2

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem obtuse_triangle_count : 
  let k_values := {k : ℕ | satisfies_triangle_inequality 11 15 k ∧ is_obtuse_triangle 11 15 k ∨ is_obtuse_triangle 11 k 15 ∧ satisfies_triangle_inequality 11 15 k} in
  (k_values.filter (λ k, let k := k in k > 0)).card = 13 :=
by sorry

end obtuse_triangle_count_l123_123600


namespace dot_product_value_l123_123869

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Conditionally introduce the conditions
variables (hb : ∥b∥ = 4) (hp : inner_product_space.proj b a = 1/2)

theorem dot_product_value : inner_product_space.inner a b = 2 :=
by
  sorry

end dot_product_value_l123_123869


namespace digit_count_log3_log3_log3_l123_123458

noncomputable def log3 : ℝ → ℝ := log 3

theorem digit_count_log3_log3_log3 (x : ℝ) (h : log3 (log3 (log3 x)) = 3) : 
  ∃ d : ℕ, d = 13 ∧ (x = 3^(3^27) ∧ nat.floor (log 10 x / log 10 3 + 1) = d) :=
by
  sorry

end digit_count_log3_log3_log3_l123_123458


namespace count_two_digit_even_congruent_to_1_mod_4_l123_123453

theorem count_two_digit_even_congruent_to_1_mod_4 : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, n % 4 = 1 ∧ 10 ≤ n ∧ n ≤ 98 ∧ n % 2 = 0) ∧ S.card = 23 := 
sorry

end count_two_digit_even_congruent_to_1_mod_4_l123_123453


namespace fourth_root_of_46656000_l123_123712

noncomputable def sixty : ℕ := 60
noncomputable def three_point_six : ℝ := 3.6
noncomputable def num : ℝ := (sixty : ℝ) * three_point_six
noncomputable def sixty_fourth : ℕ := sixty ^ 4
noncomputable def three_point_six_fourth : ℝ := (three_point_six ^ 4).toReal
noncomputable def num_fourth : ℝ := (46656000 : ℝ)

#eval sixty_fourth  -- 12960000
#eval three_point_six_fourth  -- 1679616.0
#eval num  -- 216.0

theorem fourth_root_of_46656000 : real.sqrt (real.sqrt num_fourth) = num :=
by sorry

end fourth_root_of_46656000_l123_123712


namespace triple_supplementary_angle_l123_123274

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end triple_supplementary_angle_l123_123274


namespace hyper_prism_in_box_probability_l123_123770

theorem hyper_prism_in_box_probability :
  ∀ (a b : Finset ℕ),
    a.card = 5 → b.card = 5 →
    a ∪ b = Finset.range 501 →
    (∀ (x ∈ a), ∃ (y ∈ b), y > x) →
    (∀ (y ∈ b), ∃ (x ∈ a), y > x) →
    ∀ x ∈ Finset.range 501, x ∈ a ∨ x ∈ b :=
by
  intros a b h_card_a h_card_b h_union h_a_b h_b_a x h_x
  sorry

end hyper_prism_in_box_probability_l123_123770


namespace num_archers_golden_armor_proof_l123_123958
noncomputable section

structure Soldier :=
  (is_archer : Bool)
  (is_golden : Bool)
  (tells_truth : Bool)

def count_soldiers (soldiers : List Soldier) : Nat :=
  soldiers.length

def count_truthful_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => s.tells_truth)).count q

def count_lying_responses (soldiers : List Soldier) (q : Soldier → Bool) : Nat :=
  (soldiers.filter (λ s => ¬s.tells_truth)).count q

def num_archers_golden_armor (soldiers : List Soldier) : Nat :=
  (soldiers.filter (λ s => s.is_archer ∧ s.is_golden)).length

theorem num_archers_golden_armor_proof (soldiers : List Soldier)
  (h1 : count_soldiers soldiers = 55)
  (h2 : count_truthful_responses soldiers (λ s => s.is_golden) + 
        count_lying_responses soldiers (λ s => s.is_golden) = 44)
  (h3 : count_truthful_responses soldiers (λ s => s.is_archer) + 
        count_lying_responses soldiers (λ s => s.is_archer) = 33)
  (h4 : count_truthful_responses soldiers (λ s => true) + 
        count_lying_responses soldiers (λ s => false) = 22) :
  num_archers_golden_armor soldiers = 22 := by
  sorry

end num_archers_golden_armor_proof_l123_123958


namespace count_five_digit_with_4_or_5_l123_123450

def num_five_digit_integers_with_4_or_5 : ℕ := 61328

theorem count_five_digit_with_4_or_5 :
  let total_digits := 90000
  let without_4_5 :=
    let digit_set := {0, 1, 2, 3, 6, 7, 8, 9}
    7 * 8 * 8 * 8 * 8
  total_digits - without_4_5 = num_five_digit_integers_with_4_or_5 := by sorry

end count_five_digit_with_4_or_5_l123_123450


namespace smallest_four_digit_multiple_of_18_l123_123758

-- Define the concept of a four-digit number
def four_digit (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

-- Define the concept of a multiple of 18
def multiple_of_18 (N : ℕ) : Prop := ∃ k : ℕ, N = 18 * k

-- Define the combined condition for N being a four-digit multiple of 18
def four_digit_multiple_of_18 (N : ℕ) : Prop := four_digit N ∧ multiple_of_18 N

-- State that 1008 is the smallest such number
theorem smallest_four_digit_multiple_of_18 : ∀ N : ℕ, four_digit_multiple_of_18 N → 1008 ≤ N := 
by
  intros N H
  sorry

end smallest_four_digit_multiple_of_18_l123_123758


namespace total_cost_l123_123704

theorem total_cost (cost_sandwich cost_soda cost_cookie : ℕ)
    (num_sandwich num_soda num_cookie : ℕ) 
    (h1 : cost_sandwich = 4) 
    (h2 : cost_soda = 3) 
    (h3 : cost_cookie = 2) 
    (h4 : num_sandwich = 4) 
    (h5 : num_soda = 6) 
    (h6 : num_cookie = 7):
    cost_sandwich * num_sandwich + cost_soda * num_soda + cost_cookie * num_cookie = 48 :=
by
  sorry

end total_cost_l123_123704


namespace area_of_egg_shaped_curve_l123_123371

noncomputable def radius : ℝ := 1
noncomputable def center_o : (ℝ × ℝ) := (0, 0)
noncomputable def arc_ae_center : (ℝ × ℝ) := (radius * √2, radius * √2)
noncomputable def arc_cf_center : (ℝ × ℝ) := (-radius * √2, radius * √2)
noncomputable def arc_ef_center : (ℝ × ℝ) := (0, radius * √2)

def is_egg_shaped_curve (o a e c f d : (ℝ × ℝ)) : Prop :=
  o = center_o ∧ a.1 = radius ∧ a.2 = 0 ∧ 
  c = arc_ae_center ∧ 
  e.1 = -radius / √2∧ e.2 = radius + radius / 2 ∧ 
  d.1 = radius ∧ d.2 = radius * √2 ∧ 
  f.1 = radius ∧ f.2 = radius

theorem area_of_egg_shaped_curve (o a e c f d : (ℝ × ℝ)) (h : is_egg_shaped_curve o a e c f d) :
  let area := (3 - sqrt 2) * Real.pi - 1
  in area = (3 - sqrt 2) * Real.pi - 1 :=
by
  sorry

end area_of_egg_shaped_curve_l123_123371


namespace sum_of_primes_20_to_40_l123_123642

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ (2 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime |>.sum

theorem sum_of_primes_20_to_40 : sum_of_primes_between 20 40 = 120 := by
  sorry

end sum_of_primes_20_to_40_l123_123642


namespace uranus_appearance_minutes_after_6AM_l123_123613

-- Definitions of the given times and intervals
def mars_last_seen : Int := 0 -- 12:10 AM in minutes after midnight
def jupiter_after_mars : Int := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter : Int := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def reference_time : Int := 6 * 60 -- 6:00 AM in minutes after midnight

-- Statement of the problem
theorem uranus_appearance_minutes_after_6AM :
  let jupiter_first_appearance := mars_last_seen + jupiter_after_mars
  let uranus_first_appearance := jupiter_first_appearance + uranus_after_jupiter
  (uranus_first_appearance - reference_time) = 7 := by
  sorry

end uranus_appearance_minutes_after_6AM_l123_123613


namespace puzzle_island_words_l123_123561

theorem puzzle_island_words (A : Type) (KobishExtended : FinType A) (h : Fintype.card A = 24) :
  ∑ k in finset.range 5, (((@finset.univ (fin k) (fin.fintype _)).card - (@finset.univ (fin k) (fin.fintype (⟨A, ⟨⟩⟩ \ {X}))).card)) = 53640 := 
by sorry

end puzzle_island_words_l123_123561


namespace slope_of_asymptotes_of_hyperbola_l123_123006

noncomputable def hyperbola_slope_asymptotes (m : ℝ) : Prop :=
  let a := 4 in
  let b_squared := 5 * m - 1 in
  let b := real.sqrt b_squared in
  b_squared = 9 → (m^2 + 12 = 16) → (slope_asymptotes : set ℝ) = {3 / 4, -3 / 4}

theorem slope_of_asymptotes_of_hyperbola (m : ℝ) :
  (m^2 + 12 = 16) ∧ (5 * m - 1 = 9) → hyperbola_slope_asymptotes m :=
by
  intros h
  cases h
  sorry

end slope_of_asymptotes_of_hyperbola_l123_123006


namespace derivative_of_sqrt_l123_123583

-- Definition of the function y
def y (x : ℝ) : ℝ := sqrt x

-- Statement of the theorem
theorem derivative_of_sqrt (x : ℝ) (h : x > 0) : deriv y x = 1 / (2 * sqrt x) :=
by
  sorry

end derivative_of_sqrt_l123_123583


namespace cyclic_quadrilateral_cosine_product_l123_123674

theorem cyclic_quadrilateral_cosine_product :
  ∀ (A B C D : Type*)
  [InnerProductSpace ℝ A]
  [InnerProductSpace ℝ B]
  [InnerProductSpace ℝ C]
  [InnerProductSpace ℝ D]
  (AB BC CD AD : ℝ)
  (h1: AB = 3)
  (h2: BC = 3)
  (h3: CD = 3)
  (h4: AD = 2)
  (h_inscribed: Metric.isCyclic (Set.insert A (Set.insert B (Set.insert C (Set.insert D ∅))))
    := (1 - Real.cos (angle A B C)) * (1 - Real.cos (angle B C D)) = 2 := sorry

end cyclic_quadrilateral_cosine_product_l123_123674


namespace sum_of_odd_integers_15_to_45_l123_123634

theorem sum_of_odd_integers_15_to_45 : (∑ i in finset.range (16), (15 + 2 * i)) = 480 := 
sorry

end sum_of_odd_integers_15_to_45_l123_123634


namespace bart_interest_after_10_years_l123_123582

theorem bart_interest_after_10_years :
  let P := 5000  -- Initial principal
  let r := 0.03  -- Interest rate
  let n := 10     -- Number of years
  let A := P * (1 + r)^n  -- Amount after 10 years
  A - P = 1720 := by
  let P := 5000
  let r := 0.03
  let n := 10
  let A := P * (1 + r)^n
  have h1 : A = P * (1 + r)^n := rfl
  have h2 : A - P = (P * (1 + r)^n) - P := rfl
  let A_val := P * (1 + r)^n
  change A_val at h1
  norm_num at h1
  assumption

end bart_interest_after_10_years_l123_123582


namespace sin_double_angle_inequality_l123_123911

/-- Given angles A, B, C of a triangle such that 0 ≤ A ≤ B ≤ C < π/2, 
prove that sin(2 * A) ≥ sin(2 * B) ≥ sin(2 * C). -/
theorem sin_double_angle_inequality (A B C : ℝ) 
  (h₁ : A ≤ B) (h₂ : B ≤ C) (h₃ : C < π / 2) (h₄ : 0 ≤ A) :
  real.sin (2 * A) ≥ real.sin (2 * B) ∧ real.sin (2 * B) ≥ real.sin (2 * C) := by
  sorry

end sin_double_angle_inequality_l123_123911


namespace proposition1_proposition2_main_theorem_l123_123400

-- Definition of circle
def circle (θ : ℝ) : set (ℝ × ℝ) :=
  { p | (p.1 + real.cos θ) ^ 2 + (p.2 - real.sin θ) ^ 2 = 1 }

-- Definition of line
def line (k : ℝ) : set (ℝ × ℝ) :=
  { p | p.2 = k * p.1 }

-- Proposition 1: Circle and line always intersect
theorem proposition1 (k θ : ℝ) : 
  ∃ (p : ℝ × ℝ), p ∈ circle θ ∧ p ∈ line k :=
sorry

-- Proposition 2: There exists a theta such that the line is tangent to the circle
theorem proposition2 (k : ℝ) : 
  ∃ θ : ℝ, ∃ (p : ℝ × ℝ), p ∈ circle θ ∧ p ∈ line k ∧
    (∃ (t : ℝ), ∀ q ∈ circle θ, ((q.1 - p ∘ fst)^ 2 + (q.2 - p ∘ snd) ^ 2) ≥ t ^ 2) :=
sorry

-- Main theorem combining the correct propositions ① and ②
theorem main_theorem (k θ : ℝ) :
  (∃ (p : ℝ × ℝ), p ∈ circle θ ∧ p ∈ line k) ∧
  (∃ θ : ℝ, ∃ (p : ℝ × ℝ), p ∈ circle θ ∧ p ∈ line k ∧
    (∃ (t : ℝ), ∀ q ∈ circle θ, ((q.1 - p ∘ fst) ^ 2 + (q.2 - p ∘ snd) ^ 2) ≥ t ^ 2)) :=
⟨proposition1 k θ, proposition2 k⟩

end proposition1_proposition2_main_theorem_l123_123400


namespace vertical_asymptote_condition_l123_123394

theorem vertical_asymptote_condition (k : ℝ) :
  (∀ x : ℝ, g x = (x^2 + 3 * x + k) / ((x - 7) * (x + 4)) ∧
    (x - 7) * (x + 4) = 0 → (x ∈ {7, -4})) →
  (∀ p : ℝ, (p = 7 ∨ p = -4) → (p^2 + 3 * p + k = 0) → (p = 7 ∧ k = -70) ∨ (p = -4 ∧ k = -4)) :=
by
  intro h1 h2
  sorry

noncomputable def g (x k : ℝ) := (x^2 + 3*x + k) / ((x - 7) * (x + 4))

end vertical_asymptote_condition_l123_123394


namespace complement_of_M_in_U_is_14_l123_123136

def U : Set ℕ := {x | x < 5 ∧ x > 0}

def M : Set ℕ := {x | x^2 - 5 * x + 6 = 0}

theorem complement_of_M_in_U_is_14 : 
  {x | x ∈ U ∧ x ∉ M} = {1, 4} :=
by
  sorry

end complement_of_M_in_U_is_14_l123_123136


namespace smallest_positive_four_digit_multiple_of_18_l123_123765

-- Define the predicates for conditions
def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def multiple_of_18 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 18 * k

-- Define the main theorem
theorem smallest_positive_four_digit_multiple_of_18 : 
  ∃ n : ℕ, four_digit_number n ∧ multiple_of_18 n ∧ ∀ m : ℕ, four_digit_number m ∧ multiple_of_18 m → n ≤ m :=
begin
  use 1008,
  split,
  { -- proof that 1008 is a four-digit number
    split,
    { linarith, },
    { linarith, }
  },

  split,
  { -- proof that 1008 is a multiple of 18
    use 56,
    norm_num,
  },

  { -- proof that 1008 is the smallest such number
    intros m h1 h2,
    have h3 := Nat.le_of_lt,
    sorry, -- Detailed proof would go here
  }
end

end smallest_positive_four_digit_multiple_of_18_l123_123765


namespace three_digit_integers_product_36_l123_123882

theorem three_digit_integers_product_36 : 
  ∃ (num_digits : ℕ), num_digits = 21 ∧ 
    ∀ (a b c : ℕ), 
      (1 ≤ a ∧ a ≤ 9) ∧ 
      (1 ≤ b ∧ b ≤ 9) ∧ 
      (1 ≤ c ∧ c ≤ 9) ∧ 
      (a * b * c = 36) → 
      num_digits = 21 :=
sorry

end three_digit_integers_product_36_l123_123882


namespace probability_same_plane_l123_123817

-- Define the number of vertices in a cube
def num_vertices : ℕ := 8

-- Define the number of vertices to be selected
def selection : ℕ := 4

-- Define the total number of ways to select 4 vertices out of 8
def total_ways : ℕ := Nat.choose num_vertices selection

-- Define the number of favorable ways to have 4 vertices lie in the same plane
def favorable_ways : ℕ := 12

-- Define the probability that the 4 selected vertices lie in the same plane
def probability : ℚ := favorable_ways / total_ways

-- The statement we need to prove
theorem probability_same_plane : probability = 6 / 35 := by
  sorry

end probability_same_plane_l123_123817


namespace archers_in_golden_armor_count_l123_123945

theorem archers_in_golden_armor_count : 
  ∃ (archers_golden : ℕ), 
    let total_soldiers := 55 in
    let golden_armor_affirmative := 44 in
    let archer_affirmative := 33 in
    let monday_affirmative := 22 in
    let number_archers_golden := archers_golden in
    (number_archers_golden = 22) ∧ 
    ∃ (archer : ℕ) (swordsman : ℕ) (golden : ℕ) (black : ℕ),
      archer + swordsman = total_soldiers ∧
      golden + black = total_soldiers ∧
      golden + black = total_soldiers ∧
      (swordsman * golden + archer * golden = golden_armor_affirmative) ∧
      (swordsman * archer + swordsman * golden = archer_affirmative) ∧
      ((!monday_affirmative ∧ swordsman * golden) + (!monday_affirmative ∧ archer * black) = monday_affirmative)
:= sorry

end archers_in_golden_armor_count_l123_123945


namespace sum_primes_between_20_and_40_l123_123636

theorem sum_primes_between_20_and_40 :
  23 + 29 + 31 + 37 = 120 :=
by
  sorry

end sum_primes_between_20_and_40_l123_123636


namespace solve_congruence_13n_mod47_l123_123579

theorem solve_congruence_13n_mod47 (n : ℤ) (h : 13 * n ≡ 8 [MOD 47]) : n ≡ 29 [MOD 47] := by
  sorry

end solve_congruence_13n_mod47_l123_123579


namespace ellipse_equation_minimum_triangle_area_PD_EQ_PE_l123_123842

noncomputable def ellipse := {x y : ℝ // (x^2 / 4) + y^2 = 1}
noncomputable def hyperbola := {x y : ℝ // (x^2 / (1 : ℝ)) - (y^2 / (2 + 1)) = 1}

theorem ellipse_equation (a b : ℝ) (h1 : a^2 = 4) (h2 : b^2 = 1) (h3 : e = sqrt 3 / 2) :
  (x : ℝ) (y : ℝ) : (x^2 / 4) + y^2 = 1 := sorry

theorem minimum_triangle_area (k m : ℝ) (h1 : k < 0) (h2 : m^2 = 4 * k^2 + 1) :
  ∃ (A : ℝ), A = 2 := sorry

theorem PD_EQ_PE (x₀ y₀ y₁ : ℝ) (A := (-2, 0)) (B := (2, 0)) (D := (x₀, y₀)) (E := (x₀, 0)) 
(h1 : ∃ y₁ : ℝ, y₁ = 2 * y₀ / (x₀ + 2))
(C := (2, y₁)) (AD_parallel_OC : ∀ y₁ : ℝ, y₁ * (x₀ + 2) = 2 * y₀) 
(P := (x₀, y₀ / 2)) : 
    ∥P - D∥ = ∥P - E∥ := 
sorry

end ellipse_equation_minimum_triangle_area_PD_EQ_PE_l123_123842


namespace find_angle_between_vectors_l123_123868

variables (a b : ℝ → ℝ) [nonzero a] [nonzero b] -- defining vector functions

def orthogonal (a b : ℝ → ℝ) : Prop := 
  (a · (2 * a + b)) = 0 -- dot product equality for orthogonality

def norm_squared (v : ℝ → ℝ) : ℝ := 
  (v · v) -- norm squared is the dot product with itself

def magnitude_relation (a b : ℝ → ℝ) (h : norm_squared b = 16 * norm_squared a) :=
  h -- magnitude condition inferred from norms

def angle_between (a b : ℝ → ℝ) : ℝ :=
  let cos_theta := (-1 / 2) in
  real.acos cos_theta -- angle θ such that cos θ = -1/2

theorem find_angle_between_vectors
  (a b : ℝ → ℝ)
  [nonzero a]
  [nonzero b]
  (h1 : orthogonal a b)
  (h2 : |b| = 4*|a|): 
  angle_between a b = 2 * real.pi / 3 :=
begin
  sorry -- proof
end

end find_angle_between_vectors_l123_123868


namespace number_of_boys_l123_123311

def school_problem (x y : ℕ) : Prop :=
  (x + y = 400) ∧ (y = (x / 100) * 400)

theorem number_of_boys (x y : ℕ) (h : school_problem x y) : x = 80 :=
by
  sorry

end number_of_boys_l123_123311


namespace sum_of_primes_between_20_and_40_l123_123644

theorem sum_of_primes_between_20_and_40 : 
  (23 + 29 + 31 + 37) = 120 := 
by
  -- Proof goes here
sorry

end sum_of_primes_between_20_and_40_l123_123644


namespace probability_to_pass_the_test_l123_123488

noncomputable def probability_passes_test : ℝ :=
  (finset.card (@set.to_finset {s : finset (fin 3) // s.card = 2})) * 
    (0.6 ^ 2) * (0.4) + 
  (finset.card (@set.to_finset {s : finset (fin 3) // s.card = 3})) * (0.6 ^ 3)

theorem probability_to_pass_the_test : 
  probability_passes_test = 0.648 :=
by 
  sorry

end probability_to_pass_the_test_l123_123488


namespace rectangular_prism_height_eq_17_l123_123299

-- Defining the lengths of the edges of the cubes and rectangular prism
def side_length_cube1 := 10
def edges_cube := 12
def length_rect_prism := 8
def width_rect_prism := 5

-- The total length of the wire used for each shape must be equal
def wire_length_cube1 := edges_cube * side_length_cube1
def wire_length_rect_prism (h : ℕ) := 4 * length_rect_prism + 4 * width_rect_prism + 4 * h

theorem rectangular_prism_height_eq_17 (h : ℕ) :
  wire_length_cube1 = wire_length_rect_prism h → h = 17 := 
by
  -- The proof goes here
  sorry

end rectangular_prism_height_eq_17_l123_123299


namespace sweeties_remainder_l123_123733

theorem sweeties_remainder (m k : ℤ) (h : m = 12 * k + 11) :
  (4 * m) % 12 = 8 :=
by
  -- The proof steps will go here
  sorry

end sweeties_remainder_l123_123733


namespace angle_triple_supplement_l123_123259

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123259


namespace all_roots_of_P_are_roots_of_R_R_has_no_multiple_roots_l123_123671

open Polynomial

variables {R : Type*} [CommRing R]

-- Define the polynomial P(x)
variable (P : R[X])

-- Define the gcd of P(x) and its derivative P'(x)
noncomputable def Q := gcd P P.derivative

-- Define the quotient polynomial R(x)
noncomputable def R := P / Q

-- Prove that all roots of P are roots of R
theorem all_roots_of_P_are_roots_of_R :
  ∀ (r : R), is_root P r → is_root (R P) r :=
sorry

-- Prove that R does not have multiple roots
theorem R_has_no_multiple_roots :
  ∀ (r : R) (k : ℕ), multiplicity r (R P) = k → k ≤ 1 :=
sorry

end all_roots_of_P_are_roots_of_R_R_has_no_multiple_roots_l123_123671


namespace div_by_13_l123_123620

theorem div_by_13 (n : ℕ) (h : 0 < n) : 13 ∣ (4^(2*n - 1) + 3^(n + 1)) :=
by 
  sorry

end div_by_13_l123_123620


namespace remainder_of_series_div_9_l123_123289

def sum (n : Nat) : Nat := n * (n + 1) / 2

theorem remainder_of_series_div_9 : (sum 20) % 9 = 3 :=
by
  -- The proof will go here
  sorry

end remainder_of_series_div_9_l123_123289


namespace three_planes_divide_space_into_possible_parts_l123_123615

-- Define the possible scenarios as conditions
inductive PlaneDivisionScenario
| pairwise_parallel
| pairwise_coplanar_intersecting
| pairwise_intersecting_with_three_lines
| two_planes_intersecting_third_intersecting_both

-- Define the proof statement
theorem three_planes_divide_space_into_possible_parts (scenario : PlaneDivisionScenario) : 
  ∃ parts : ℕ, parts ∈ {4, 6, 7, 8} :=
by
  cases scenario
  case PlaneDivisionScenario.pairwise_parallel =>
    use 4
  case PlaneDivisionScenario.pairwise_coplanar_intersecting =>
    use 6
  case PlaneDivisionScenario.pairwise_intersecting_with_three_lines =>
    use 7
  case PlaneDivisionScenario.two_planes_intersecting_third_intersecting_both =>
    use 8
  sorry

end three_planes_divide_space_into_possible_parts_l123_123615


namespace circle_center_radius_sum_l123_123522

theorem circle_center_radius_sum:
  let D := { x // ∃ y, x^2 + 20 * x + y^2 + 18 * y = -36 } in
  let p := -10 in
  let q := -9 in
  let s := Real.sqrt 145 in
  p + q + s = -19 + Real.sqrt 145 :=
by
  sorry

end circle_center_radius_sum_l123_123522


namespace football_throw_incorrect_statement_l123_123235

theorem football_throw_incorrect_statement :
  ∀ (l v_max g : ℝ), 
  (∀ t θ : ℝ, 0 ≤ θ ∧ θ ≤ (2 * Math.pi) → t = (2 * v_max * Math.sin(θ)) / g) →
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ (2 * Math.pi) → (v_max^2 * Math.sin(2 * θ)) / g ≥ l) →
  (v_max < Real.sqrt (g * l) ∨ v_max >= Real.sqrt (g * l)) →
  (∀ t θ : ℝ, (l increases) → (v_max held fixed) → max t does not necessarily increase) :=
begin
  sorry
end

end football_throw_incorrect_statement_l123_123235


namespace evalExpression_l123_123366
noncomputable def calcExpression : Real :=
  (-1) ^ 2022 + (27 : Real) ^ (1 / 3) + (4 : Real).sqrt + | (3 : Real).sqrt - 2 |

theorem evalExpression : calcExpression = 8 - (3 : Real).sqrt := 
  by
  sorry

end evalExpression_l123_123366


namespace final_value_l123_123375

noncomputable def double_factorial (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2^((n/2) : ℕ) * factorial (n/2)
  else if n > 1 then n * double_factorial (n-2)
  else 1

noncomputable def sum_expression : ℚ :=
  (∑ i in Finset.range 1011, (Nat.choose (2*i) i) / 2^(2*i))

theorem final_value : (let a := 1019; let b := 1 in 
     (a * b / 10 : ℚ) = 101.9) :=
sorry

end final_value_l123_123375


namespace sum_of_ratios_at_least_one_l123_123691

theorem sum_of_ratios_at_least_one (s : ℝ) (a b : ℕ → ℝ) (n : ℕ)
    (h1 : s > 0)
    (h2 : ∀ i, i < n → a i ≤ b i)
    (h3 : ∑ i in finset.range n, a i * b i = s^2) :
    ∑ i in finset.range n, a i / b i ≥ 1 :=
sorry

end sum_of_ratios_at_least_one_l123_123691


namespace paint_one_third_of_square_l123_123563

theorem paint_one_third_of_square :
  @nat.choose 18 6 = 18564 :=
by sorry

end paint_one_third_of_square_l123_123563


namespace no_obtuse_triangle_probability_eq_l123_123787

noncomputable def probability_no_obtuse_triangle : ℝ :=
  let num_points := 4
  -- Condition (1): Four points are chosen uniformly at random on a circle.
  -- Condition (2): An obtuse angle occurs if the minor arc exceeds π/2.
  9 / 64

theorem no_obtuse_triangle_probability_eq :
  let num_points := 4
  ∀ (points : Fin num_points → ℝ), 
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∠ (points i, points j, points k) < π / 2) →
  probability_no_obtuse_triangle = 9 / 64 :=
by
  sorry

end no_obtuse_triangle_probability_eq_l123_123787


namespace power_of_H_with_respect_to_Γ_l123_123132

-- conditions
variables {α β γ : ℝ} (A B C H : Point) -- Points and angles
variable {R : ℝ} -- Radius of circumcircle

-- assuming necessary geometric properties
-- Triangle ABC
axiom triangle_ABC : Triangle A B C

-- Orthocenter H of triangle ABC
axiom orthocenter_H : Orthocenter triangle_ABC H

-- Circumcircle Γ of triangle ABC
axiom circumcircle_Γ : Circumcircle triangle_ABC

-- circumradius R of triangle's circumcircle
axiom circumradius_R : Circumradius circumcircle_Γ = R

-- Angles α, β, γ of the triangle
axiom angle_α : angle A = α
axiom angle_β : angle B = β
axiom angle_γ : angle C = γ

-- The power of H with respect to Γ
def power_of_H (H : Point) (Γ : Circumcircle) : ℝ :=
  (euclidean_dist (H, center(Γ)))^2 - (radius(Γ))^2

-- The theorem statement
theorem power_of_H_with_respect_to_Γ (H : Point) (Γ : Circumcircle) (R : ℝ) (α β γ : ℝ):
  power_of_H H Γ = 8 * R^2 * (cos α) * (cos β) * (cos γ) := 
sorry

end power_of_H_with_respect_to_Γ_l123_123132


namespace sum_of_primes_between_20_and_40_l123_123645

theorem sum_of_primes_between_20_and_40 : 
  (23 + 29 + 31 + 37) = 120 := 
by
  -- Proof goes here
sorry

end sum_of_primes_between_20_and_40_l123_123645


namespace coefficient_x2_in_expansion_l123_123500

theorem coefficient_x2_in_expansion : 
  let expr := (2 * (x : ℚ)^3 - x⁻¹)^6 in
  let general_term (r : ℕ) := (nat.choose 6 r) * (2 * (x : ℚ)^3)^(6 - r) * (-x⁻¹)^r in
  ∃ (r : ℕ), (18 - 4 * r = 2) ∧ general_term r = (60 : ℚ * x^2) :=
by
  sorry

end coefficient_x2_in_expansion_l123_123500


namespace array_count_l123_123449

-- Define a 5x5 matrix with entries in ℤ (integers)
def Array5x5 := Matrix (Fin 5) (Fin 5) ℤ

-- Define the necessary conditions
def isValidArray (A : Array5x5) : Prop :=
  (∀ i : Fin 5, ∑ j, A i j = 0) ∧
  (∀ j : Fin 5, ∑ i, A i j = 0) ∧
  (∀ i : Fin 5, (∑ j, if A i j = 1 then 1 else 0) = 3) ∧
  (∀ j : Fin 5, (∑ i, if A i j = 1 then 1 else 0) = 3)

-- The statement to prove
theorem array_count : ∃ (f : Fin 5 → Fin 5 → ℤ), isValidArray f ∧ 
  (∃ n : ℕ, n = 310 ∧ (f : Fin 5 → Fin 5 → ℤ) ≠ f2 → n = 310) :=
sorry

end array_count_l123_123449


namespace range_of_x_l123_123045

theorem range_of_x (x : ℝ) : ¬ (x ∈ set.Icc 2 5 ∨ x < 1 ∨ x > 4) → x ∈ set.Ico 1 2 :=
by
  intro h
  sorry

end range_of_x_l123_123045


namespace onion_pieces_per_student_l123_123603

theorem onion_pieces_per_student (total_pizzas : ℕ) (slices_per_pizza : ℕ)
  (cheese_pieces_leftover : ℕ) (onion_pieces_leftover : ℕ) (students : ℕ) (cheese_per_student : ℕ)
  (h1 : total_pizzas = 6) (h2 : slices_per_pizza = 18) (h3 : cheese_pieces_leftover = 8) (h4 : onion_pieces_leftover = 4)
  (h5 : students = 32) (h6 : cheese_per_student = 2) :
  ((total_pizzas * slices_per_pizza) - cheese_pieces_leftover - onion_pieces_leftover - (students * cheese_per_student)) / students = 1 := 
by
  sorry

end onion_pieces_per_student_l123_123603


namespace percentage_liquid_x_in_mixture_l123_123664

-- Definitions of the conditions
def liquid_x_in_solution_a_percent := 0.008
def liquid_x_in_solution_b_percent := 0.018
def weight_solution_a := 300
def weight_solution_b := 700

-- The theorem to prove the percentage of liquid X in the mixture
theorem percentage_liquid_x_in_mixture :
  let total_liquid_x := (liquid_x_in_solution_a_percent * weight_solution_a) +
                        (liquid_x_in_solution_b_percent * weight_solution_b)
  let total_weight := weight_solution_a + weight_solution_b
  total_liquid_x / total_weight * 100 = 1.5 :=
by
  sorry

end percentage_liquid_x_in_mixture_l123_123664


namespace three_digit_integers_with_product_36_l123_123880

-- Definition of the problem conditions
def is_three_digit_integer (n : Nat) : Prop :=
  100 ≤ n ∧ n < 1000

def digit_product_is_36 (n : Nat) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  1 ≤ d1 ∧ d1 ≤ 9 ∧ 1 ≤ d2 ∧ d2 ≤ 9 ∧ 1 ≤ d3 ∧ d3 ≤ 9 ∧ (d1 * d2 * d3 = 36)

-- The statement of the proof
theorem three_digit_integers_with_product_36 :
  {n : Nat | is_three_digit_integer n ∧ digit_product_is_36 n}.toFinset.card = 21 := 
by
  sorry

end three_digit_integers_with_product_36_l123_123880


namespace shortest_altitude_l123_123215

/-!
  Prove that the shortest altitude of a right triangle with sides 9, 12, and 15 is 7.2.
-/

theorem shortest_altitude (a b c : ℕ) (h : a^2 + b^2 = c^2) (ha : a = 9) (hb : b = 12) (hc : c = 15) :
  7.2 ≤ a ∧ 7.2 ≤ b ∧ 7.2 ≤ (2 * (a * b) / c) := 
sorry

end shortest_altitude_l123_123215


namespace angle_triple_supplement_l123_123254

theorem angle_triple_supplement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end angle_triple_supplement_l123_123254


namespace impossible_equal_distribution_l123_123186

def total_glasses : ℕ := 2018
def initial_contents (n : ℕ) : ℕ :=
  if n = 0 then 2 else 1

def khryusha_step (A B : ℚ) : ℚ × ℚ :=
  let avg := (A + B) / 2 in (avg, avg)

theorem impossible_equal_distribution :
  ¬ ∃ (final_contents : fin total_glasses → ℚ),
    (∀ i, initial_contents i ∈ {1, 2}) ∧
    (∑ i : fin total_glasses, initial_contents i = 2019) ∧
    (∀ step : ℕ, ∀ (i j : fin total_glasses),
      khryusha_step (initial_contents i) (initial_contents j) = (final_contents i, final_contents j)) ∧
    (∀ i : fin total_glasses, final_contents i = 2019 / 2018) := sorry

end impossible_equal_distribution_l123_123186


namespace find_a_l123_123055

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then
  2
else
  -x^3 + 6*x^2 - 9*x + 2 - a

def is_twin_point_pair (f : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
A = (-B.1, -B.2) ∧ f(A.1) = A.2 ∧ f(B.1) = B.2

def twin_point_pairs {f : ℝ → ℝ} (l : list (ℝ × ℝ)) : Prop :=
∀ (A B : ℝ × ℝ), A ∈ l → B ∈ l → is_twin_point_pair f A B ↔ A = B ∨ ∃ C, C = (B.1, B.2) ∧ C ∈ l

theorem find_a :
  (∀a : ℝ, twin_point_pairs [(0, f 0 a), (1, f 1 a)] → a = 0) :=
sorry

end find_a_l123_123055


namespace factorization_eq_l123_123295

theorem factorization_eq :
  ∀ (a : ℝ), a^2 + 4 * a - 21 = (a - 3) * (a + 7) := by
  intro a
  sorry

end factorization_eq_l123_123295


namespace sum_inequality_l123_123542

variable (n : ℕ)

theorem sum_inequality (h : 0 < n) : 
  ∑ i in Finset.range n + 1, (1 / (i^2 + i) ^ (3/4 : ℝ)) > 2 - (2 / (Real.sqrt (n + 1))) :=
sorry

end sum_inequality_l123_123542


namespace at_least_one_shooter_hits_target_l123_123241

-- Definition stating the probability of the first shooter hitting the target
def prob_A1 : ℝ := 0.7

-- Definition stating the probability of the second shooter hitting the target
def prob_A2 : ℝ := 0.8

-- The event that at least one shooter hits the target
def prob_at_least_one_hit : ℝ := prob_A1 + prob_A2 - (prob_A1 * prob_A2)

-- Prove that the probability that at least one shooter hits the target is 0.94
theorem at_least_one_shooter_hits_target : prob_at_least_one_hit = 0.94 :=
by
  sorry

end at_least_one_shooter_hits_target_l123_123241


namespace exponent_rule_example_l123_123709

theorem exponent_rule_example : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  sorry

end exponent_rule_example_l123_123709


namespace simplify_polynomial_l123_123577

theorem simplify_polynomial (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 5) - (x^6 + 2 * x^5 + x^4 - x^3 + 7) = 
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - 2 :=
by
  sorry

end simplify_polynomial_l123_123577
