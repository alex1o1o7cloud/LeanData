import Mathlib

namespace three_digit_divisible_by_14_and_6_l133_133138

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def divisible_by (a b : ℕ) : Prop := b % a = 0

theorem three_digit_divisible_by_14_and_6 : (card (set_of (λ n, is_three_digit n ∧ divisible_by 42 n)) = 21) :=
sorry

end three_digit_divisible_by_14_and_6_l133_133138


namespace product_of_two_numbers_l133_133650

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 70) (h2 : x - y = 10) : x * y = 1200 :=
by
  sorry

end product_of_two_numbers_l133_133650


namespace smallest_cost_l133_133782

def gift1_choc := 3
def gift1_caramel := 15
def price1 := 350

def gift2_choc := 20
def gift2_caramel := 5
def price2 := 500

def equal_candies (m n : ℕ) : Prop :=
  gift1_choc * m + gift2_choc * n = gift1_caramel * m + gift2_caramel * n

def total_cost (m n : ℕ) : ℕ :=
  price1 * m + price2 * n

theorem smallest_cost :
  ∃ m n : ℕ, equal_candies m n ∧ total_cost m n = 3750 :=
by {
  sorry
}

end smallest_cost_l133_133782


namespace prob_17_33_mod_7_l133_133324

theorem prob_17_33_mod_7 : (17^33) % 7 = 6 := by
  have h1 : 17 % 7 = 3 := rfl  -- 17 ≡ 3 (mod 7)
  have h2 : 17^33 % 7 = 3^33 % 7 := by 
    rw [← Nat.ModEq.pow_mod _ _ 33, h1]
    exact Nat.ModEq.symm (Nat.ModEq.pow _ h1 _)
  have h3 : 3^6 % 7 = 1 := by
    -- sequence of simplifications as shown in the solution
    calc
      3^1 % 7 = 3   := rfl
      3^2 % 7 = 2   := rfl
      3^3 % 7 = 6   := rfl
      3^4 % 7 = 4   := rfl
      3^5 % 7 = 5   := rfl
      3^6 % 7 = 1   := rfl

  have h4 : 3^33 % 7 = (3^6)^5 * 3^3 % 7 := by rw [← Nat.pow_add, Nat.mul_mod]

  have h5 : (3^6)^5 % 7 = 1^5 % 7 := by rw [h3, one_pow]

  have h6 : 1^5 % 7 = 1 := by rfl

  have h7 : 3^33 % 7 = (1 * 3^3) % 7 := by rw [h5, h4, h6]

  have h8 : 3^3 % 7 = 6 := rfl

  exact calc
    17^33 % 7 = 3^33 % 7       := h2
    ...      = (1 * 3^3) % 7  := by rw [h7, h8]
    ...      = 6 % 7         := rfl
    ...      = 6            := rfl

end prob_17_33_mod_7_l133_133324


namespace line_AB_fixed_point_l133_133217

-- Define the vector space over real numbers.
variables {V : Type*} [inner_product_space ℝ V] [complete_space V]

-- Define the points O, A, and B in the vector space V.
variables (O A B : V)

-- Define the scalars p, q, c as real numbers.
variables (p q c : ℝ)

-- Define the distances (OA and OB) as positive real numbers.
variables (OA OB : ℝ) (hOA : 0 < OA) (hOB : 0 < OB)

-- Conditions stating that A and B move along fixed rays from O.
-- OA = OA would be the magnitude of vector (O -> A), similarly OB.
axiom ray_condition : OA ≠ 0 ∧ OB ≠ 0 ∧ 
  (∃ (λ μ : ℝ), λ ≠ 0 ∧ μ ≠ 0 ∧ A = O + λ • (A - O) ∧ B = O + μ • (B - O))

-- Given condition that the quantity remains constant.
axiom constant_condition : p / OA + q / OB = c

-- Prove the line AB passes through a fixed point.
theorem line_AB_fixed_point : ∃ F : V, ∀ (A B : V), 
  (ray_condition O A B OA OB) → 
  (constant_condition p q c O A B OA OB) →
  ∃ t : ℝ, (F = (t • A + (1 - t) • B)) :=
begin
  sorry
end

end line_AB_fixed_point_l133_133217


namespace exponentiation_notation_l133_133545

theorem exponentiation_notation (a : ℕ) (n : ℕ) : 
  (nat.rec 1 (fun _ ih => a * ih) n) = a^n := 
begin
  by sorry
end

end exponentiation_notation_l133_133545


namespace inclination_angle_line_l133_133888

-- Define the conditions
variable (a b c : ℝ) (α : ℝ)
hypothesis h1 : sin α + cos α = 0
hypothesis h2 : ∀ (x y : ℝ), a * x + b * y + c = 0 → Function.monotoneOn (sin α + cos α) {z | z ≠ 0 }

-- Target statement
theorem inclination_angle_line (a b c α : ℝ) (h1 : sin α + cos α = 0) 
  : a - b = 0 :=
sorry

end inclination_angle_line_l133_133888


namespace tan_3theta_l133_133513

theorem tan_3theta (θ : ℝ) (h : Real.tan θ = 3 / 4) : Real.tan (3 * θ) = -12.5 :=
sorry

end tan_3theta_l133_133513


namespace cosine_sixth_power_l133_133305

theorem cosine_sixth_power : 
  (∃ b1 b2 b3 b4 b5 b6 : ℝ, ∀ θ : ℝ,
    cos θ ^ 6 = 
      b1 * cos θ + 
      b2 * cos (2 * θ) + 
      b3 * cos (3 * θ) + 
      b4 * cos (4 * θ) + 
      b5 * cos (5 * θ) + 
      b6 * cos (6 * θ)) →
  ∑ (i : Fin 6), (ite (i = 0) 0 (ite (i = 1) (15 / 32) (ite (i = 2) 0 (ite (i = 3) (3 / 16) (ite (i = 4) 0 (1 / 32))))) ^ 2 = 131 / 512 :=
by
  intro h,
  rcases h with ⟨b1, b2, b3, b4, b5, b6, hcos⟩,
  have hb1 : b1 = 0, sorry,
  have hb2 : b2 = 15 / 32, sorry,
  have hb3 : b3 = 0, sorry,
  have hb4 : b4 = 3 / 16, sorry,
  have hb5 : b5 = 0, sorry,
  have hb6 : b6 = 1 / 32, sorry,
  simp [hb1, hb2, hb3, hb4, hb5, hb6],
  norm_num,
  rfl

end cosine_sixth_power_l133_133305


namespace initial_number_of_chickens_l133_133157

theorem initial_number_of_chickens
    (initial_horses : ℕ)
    (initial_sheep : ℕ)
    (initial_chickens : ℕ)
    (male_animals : ℕ)
    (goats_gifted : ℕ)
    (half_animals_male : bool)
    (total_male_animals : ℕ)
    (initial_animals_divided_by_2 : bool)
    (new_total_animals : ℕ)
    (half_new_total_animals_is_male : bool)
    : initial_horses = 100 →
      initial_sheep = 29 →
      goats_gifted = 37 →
      total_male_animals = 53 →
      half_new_total_animals_is_male →
      initial_animals_divided_by_2 →
      (initial_horses + initial_sheep + initial_chickens) / 2 + goats_gifted = new_total_animals →
      new_total_animals = total_male_animals * 2 →
      half_animals_male →
      initial_chickens = 9 := 
by
    intros h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈ h₉
    -- Proof steps would go here
    sorry

end initial_number_of_chickens_l133_133157


namespace sphere_reflections_l133_133503

-- Define points and reflections in Lean
variables {A B C D P Q Q_A Q_B Q_C Q_D P_A P_B P_C P_D : Type}

/--
Given a tetrahedron A B C D, let P be an arbitrary point in space.
Let P_D, P_C, P_B, P_A be the reflections of P on the planes of faces ABC, ABD, ACD, and BCD respectively.
Let Q be the center of the sphere passing through P_D, P_C, P_B, P_A.
Let Q_D, Q_C, Q_B, Q_A be the reflections of Q on the same planes.
Prove that the sphere passing through Q_D, Q_C, Q_B, Q_A has its center at P and has the same radius
as the sphere passing through P_D, P_C, P_B, P_A.
-/
theorem sphere_reflections
  (tetrahedron : A B C D)
  (pointP : P)
  (reflection_PD : P_D)
  (reflection_PC : P_C)
  (reflection_PB : P_B)
  (reflection_PA : P_A)
  (sphere_center_Q : Q)
  (reflection_QD : Q_D)
  (reflection_QC : Q_C)
  (reflection_QB : Q_B)
  (reflection_QA : Q_A)
  (h1 : reflection_PD = reflection_P_on_plane_ABC pointP A B C)
  (h2 : reflection_PC = reflection_P_on_plane_ABD pointP A B D)
  (h3 : reflection_PB = reflection_P_on_plane_ACD pointP A C D)
  (h4 : reflection_PA = reflection_P_on_plane_BCD pointP B C D)
  (h5 : sphere_center_Q = center_of_sphere P_D P_C P_B P_A)
  (h6 : reflection_QD = reflection_Q_on_plane_ABC sphere_center_Q A B C)
  (h7 : reflection_QC = reflection_Q_on_plane_ABD sphere_center_Q A B D)
  (h8 : reflection_QB = reflection_Q_on_plane_ACD sphere_center_Q A C D)
  (h9 : reflection_QA = reflection_Q_on_plane_BCD sphere_center_Q B C D) :
  (center_of_sphere Q_D Q_C Q_B Q_A = pointP) ∧
  (radius_of_sphere Q_D Q_C Q_B Q_A = radius_of_sphere P_D P_C P_B P_A) :=
sorry

end sphere_reflections_l133_133503


namespace bowling_competition_sequences_l133_133159

theorem bowling_competition_sequences :
  let matches := 5 -- There are 5 matches in total (6 vs 5, winner vs 4, ..., winner vs 1)
  let outcomes_per_match := 2
  (outcomes_per_match ^ matches) = 32 := by
  let matches := 5
  let outcomes_per_match := 2
  show outcomes_per_match ^ matches = 32
  sorry

end bowling_competition_sequences_l133_133159


namespace two_extensions_result_five_operations_result_l133_133130

-- Given two positive numbers and a defined extension operation c = ab + a + b.
def extend (a b : ℕ) : ℕ := a * b + a + b

-- 1. Proving the result for two extensions starting with 1 and 2.
theorem two_extensions_result : extend (max 1 2) (extend 1 2) = 17 :=
by
  sorry

-- 2. Proving the extension formula after five operations.
theorem five_operations_result (p q : ℕ) (hpq : p > q) : 
  let c1 := extend p q,
      c2 := extend (max p c1) (min p c1),
      c3 := extend (max c2 c1) (min c2 c1),
      c4 := extend (max c3 c2) (min c3 c2),
      c5 := extend (max c4 c3) (min c4 c3)
    in c5 = (q+1)^8 * (p+1)^5 - 1 ∧ (8 + 5) = 13 :=
by
  sorry

end two_extensions_result_five_operations_result_l133_133130


namespace women_lawyers_percentage_l133_133749

-- Define the conditions of the problem
variable {T : ℝ} (h1 : 0.80 * T = 0.80 * T)                          -- Placeholder for group size, not necessarily used directly
variable (h2 : 0.32 = 0.80 * L)                                       -- Given condition of the problem: probability of selecting a woman lawyer

-- Define the theorem to be proven
theorem women_lawyers_percentage (h2 : 0.32 = 0.80 * L) : L = 0.4 :=
by
  sorry

end women_lawyers_percentage_l133_133749


namespace average_shift_l133_133021

variable (a b c : ℝ)

-- Given condition: The average of the data \(a\), \(b\), \(c\) is 5.
def average_is_five := (a + b + c) / 3 = 5

-- Define the statement to prove: The average of the data \(a-2\), \(b-2\), \(c-2\) is 3.
theorem average_shift (h : average_is_five a b c) : ((a - 2) + (b - 2) + (c - 2)) / 3 = 3 :=
by
  sorry

end average_shift_l133_133021


namespace number_of_valid_integers_l133_133421

theorem number_of_valid_integers (n : ℕ) (h : n ≥ 1) :
  let F := λ (n : ℕ), 2^(n + 1) - 2 * n - 2 
  F n = ∑ k in Finset.range(n + 1), if k > 0 then (n - k + 1) * 2^(k - 1) - 1 else 0 :=
by
  sorry

end number_of_valid_integers_l133_133421


namespace log2_q_for_tournament_l133_133431

noncomputable def tournament_game_log2 (num_teams : ℕ) : ℕ :=
  let num_games := num_teams * (num_teams - 1) / 2
  let factorial_powers_2 := ∑ n in Finset.range (num_teams + 1), num_teams / 2^n
  num_games - factorial_powers_2

theorem log2_q_for_tournament :
  tournament_game_log2 50 = 1178 :=
by
  sorry

end log2_q_for_tournament_l133_133431


namespace angle_between_diagonal_and_base_l133_133374

theorem angle_between_diagonal_and_base 
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) :
  ∃ θ : ℝ, θ = Real.arctan (Real.sin (α / 2)) :=
sorry

end angle_between_diagonal_and_base_l133_133374


namespace arithmetic_sequence_difference_l133_133322

theorem arithmetic_sequence_difference :
  ∀ (a d : ℕ), a = 3 → d = 5 → ((a + 104 * d) - (a + 99 * d) = 25) :=
by
  intros a d ha hd
  rw [ha, hd]
  rfl
  sorry

end arithmetic_sequence_difference_l133_133322


namespace given_triangle_conditions_l133_133553

open Real

theorem given_triangle_conditions
  (A B C a b c : ℝ)
  (h1 : a * cos C = c * (2 * cos B - cos A) + 2 * b * cos C)
  (h2 : a^2 * sin((A + B) / 2) = (1 / 2) * a * b * sin C) :
  C = π / 3 :=
by
  sorry

end given_triangle_conditions_l133_133553


namespace concyclic_A_N_F_P_l133_133983

-- Definitions of points and given conditions
variables {A B C M N P D E F : Type}
variables [triangle : EuclideanGeometry.Triangle A B C]
variables [acute_scalene_triangle : EuclideanGeometry.AcuteScaleneTriangle A B C]
variables [midpoint_M : EuclideanGeometry.Midpoint M B C]
variables [midpoint_N : EuclideanGeometry.Midpoint N C A]
variables [midpoint_P : EuclideanGeometry.Midpoint P A B]
variables [perp_bisector_D : EuclideanGeometry.PerpendicularBisector D A B (EuclideanGeometry.Ray A M)]
variables [perp_bisector_E : EuclideanGeometry.PerpendicularBisector E A C (EuclideanGeometry.Ray A M)]
variables [intersection_F : EuclideanGeometry.Intersection F (EuclideanGeometry.Line B D) (EuclideanGeometry.Line C E)]
variables [inside_triangle : EuclideanGeometry.PointInsideTriangle F A B C]

-- Theorem stating that A, N, F, and P are concyclic
theorem concyclic_A_N_F_P :
  EuclideanGeometry.ConcyclicPoints A N F P :=
begin
  sorry
end

end concyclic_A_N_F_P_l133_133983


namespace expressions_positive_l133_133283

-- Definitions based on given conditions
def A := 2.5
def B := -0.8
def C := -2.2
def D := 1.1
def E := -3.1

-- The Lean statement to prove the necessary expressions are positive numbers.

theorem expressions_positive :
  (B + C) / E = 0.97 ∧
  B * D - A * C = 4.62 ∧
  C / (A * B) = 1.1 :=
by
  -- Assuming given conditions and steps to prove the theorem.
  sorry

end expressions_positive_l133_133283


namespace find_m_l133_133063

noncomputable def ellipse : set (ℝ × ℝ) := { p | (p.1)^2 / 3 + (p.2)^2 = 1 }
def f1 : ℝ × ℝ := (- real.sqrt 2, 0)
def f2 : ℝ × ℝ := (real.sqrt 2, 0)

def line (m : ℝ) : set (ℝ × ℝ) := { p | p.2 = p.1 + m }

theorem find_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line m ∧ B ∈ line m ∧
   2 * (abs ((f1.1 * (A.2 - B.2) + A.1 * (B.2 - f1.2) + B.1 * (f1.2 - A.2)) / 2))
 = abs ((f2.1 * (A.2 - B.2) + A.1 * (B.2 - f2.2) + B.1 * (f2.2 - A.2)) / 2)) →
  m = - real.sqrt 2 / 3 :=
sorry

end find_m_l133_133063


namespace range_of_k_l133_133102

noncomputable def operation (a b : ℝ) : ℝ := Real.sqrt (a * b) + a + b

theorem range_of_k (k : ℝ) (h : operation 1 (k^2) < 3) : -1 < k ∧ k < 1 :=
by
  sorry

end range_of_k_l133_133102


namespace lcm_ac_is_420_l133_133634

theorem lcm_ac_is_420 (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 21) :
    Nat.lcm a c = 420 :=
sorry

end lcm_ac_is_420_l133_133634


namespace time_left_for_exercises_l133_133181

theorem time_left_for_exercises (total_minutes : ℕ) (piano_minutes : ℕ) (writing_minutes : ℕ) (reading_minutes : ℕ) : 
  total_minutes = 120 ∧ piano_minutes = 30 ∧ writing_minutes = 25 ∧ reading_minutes = 38 → 
  total_minutes - (piano_minutes + writing_minutes + reading_minutes) = 27 :=
by
  intro h
  cases h with h_total h
  cases h with h_piano h
  cases h with h_writing h_reading
  rw [h_total, h_piano, h_writing, h_reading]
  exactly rfl

end time_left_for_exercises_l133_133181


namespace axis_symmetry_shifted_graph_l133_133520

open Real

theorem axis_symmetry_shifted_graph :
  ∀ k : ℤ, ∃ x : ℝ, (y = 2 * sin (2 * x)) ∧
  y = 2 * sin (2 * (x + π / 12)) ↔
  x = k * π / 2 + π / 6 :=
sorry

end axis_symmetry_shifted_graph_l133_133520


namespace fraction_problem_l133_133795

theorem fraction_problem (a b c d e: ℚ) (val: ℚ) (h_a: a = 1/4) (h_b: b = 1/3) 
  (h_c: c = 1/6) (h_d: d = 1/8) (h_val: val = 72) :
  (a * b * c * val + d) = 9 / 8 :=
by {
  sorry
}

end fraction_problem_l133_133795


namespace ellipse_intersection_area_condition_l133_133029

theorem ellipse_intersection_area_condition :
  let ellipse := ∀ (x y : ℝ), (x^2 / 3 + y^2 = 1)
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let line m := ∀ (x y : ℝ), (y = x + m)
  ∀ (A B : ℝ × ℝ),
    ((A.1^2 / 3 + A.2^2 = 1) ∧ (y = A.1 + m)) ∧
    ((B.1^2 / 3 + B.2^2 = 1) ∧ (y = B.1 + m)) ∧
    (area (F1, A, B) = 2 * area (F2, A, B)) →
    m = -Real.sqrt 2 / 3 := by sorry

end ellipse_intersection_area_condition_l133_133029


namespace binomial_distribution_probability_l133_133894

open Real

-- Define the binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  nat.choose n k

-- Define the probability mass function for binomial distribution
def binomial_pmf (n k : ℕ) (p : ℝ) : ℝ :=
  binomial_coeff n k * p^k * (1 - p)^(n - k)

-- The given problem's conditions
def n : ℕ := 6
def k : ℕ := 3
def p : ℝ := 1 / 2

-- Now, we state the theorem to prove the value of P(X = 3)
theorem binomial_distribution_probability :
  binomial_pmf n k p = 5 / 16 :=
by
  sorry

end binomial_distribution_probability_l133_133894


namespace value_of_ff_of_one_ninth_l133_133490

def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 3
  else 2^x

theorem value_of_ff_of_one_ninth : (f (f (1 / 9))) = 1 / 4 :=
by
  sorry

end value_of_ff_of_one_ninth_l133_133490


namespace probability_of_drawing_all_blue_marbles_l133_133758

-- Define the problem conditions and question
def total_marbles := 7
def blue_marbles := 4
def yellow_marbles := 3

-- Definition of the probability that all 4 blue marbles are drawn before all 3 yellow marbles
def desired_probability := 3 / 7

theorem probability_of_drawing_all_blue_marbles :
  let total_arrangements := nat.choose 7 3
      favorable_arrangements := nat.choose 6 2
  in (favorable_arrangements / total_arrangements) = desired_probability :=
by
  sorry

end probability_of_drawing_all_blue_marbles_l133_133758


namespace ratio_Nikki_to_Michael_l133_133184

theorem ratio_Nikki_to_Michael
  (M Joyce Nikki Ryn : ℕ)
  (h1 : Joyce = M + 2)
  (h2 : Nikki = 30)
  (h3 : Ryn = (4 / 5) * Nikki)
  (h4 : M + Joyce + Nikki + Ryn = 76) :
  Nikki / M = 3 :=
by {
  sorry
}

end ratio_Nikki_to_Michael_l133_133184


namespace maria_wins_with_probability_half_l133_133993

theorem maria_wins_with_probability_half :
  let p := probability_of_more_heads(2023, "Maria", "Bilyana"),
      q := probability_of_same_heads(2023, "Maria", "Bilyana") in
  (2 * p + q = 1) → (1 / 2) * p + (1 / 2) * (p + q) = (1 / 2) :=
by
  sorry

-- Auxiliary definitions
noncomputable def probability_of_more_heads (n : ℕ) (player1 player2 : String) : ℝ := sorry
noncomputable def probability_of_same_heads (n : ℕ) (player1 player2 : String) : ℝ := sorry

end maria_wins_with_probability_half_l133_133993


namespace percentage_carbonated_water_in_mixture_l133_133768

-- Define the constants based on the conditions
def first_solution_CW := 0.80
def second_solution_CW := 0.55
def fraction_first_sol := 0.6799999999999997

-- Define the total volume and carbonated water volume calculation
def total_mixture_volume (V : ℝ) : ℝ := V
def carbonated_water_volume (V : ℝ) : ℝ :=
  first_solution_CW * (fraction_first_sol * V) + 
  second_solution_CW * ((1 - fraction_first_sol) * V)

-- Define the main theorem
theorem percentage_carbonated_water_in_mixture (V : ℝ) :
  (carbonated_water_volume V / total_mixture_volume V) * 100 = 71.99999999999999 :=
by
  sorry

end percentage_carbonated_water_in_mixture_l133_133768


namespace tradesman_overall_percentage_gain_l133_133381

def defraud (percent: ℝ) (amount: ℝ) : ℝ :=
    amount + (amount * percent / 100)

def item_gain (buy_defraud: ℝ) (sell_defraud: ℝ) (base_amount: ℝ): ℝ :=
    let buy_amount := defraud buy_defraud base_amount
    let sell_amount := defraud sell_defraud base_amount
    sell_amount - buy_amount

theorem tradesman_overall_percentage_gain :
  let amount_spent := 100 in
  let gain_A := item_gain (-30) 30 amount_spent in
  let gain_B := item_gain (-20) 10 amount_spent in
  let gain_C := item_gain (-10) 20 amount_spent in
  let total_gain := gain_A + gain_B + gain_C in
  let total_outlay := 3 * amount_spent in
  (total_gain / total_outlay) * 100 = 20 :=
by
  sorry

end tradesman_overall_percentage_gain_l133_133381


namespace ceil_sqrt_product_l133_133818

noncomputable def ceil_sqrt_3 : ℕ := ⌈Real.sqrt 3⌉₊
noncomputable def ceil_sqrt_12 : ℕ := ⌈Real.sqrt 12⌉₊
noncomputable def ceil_sqrt_120 : ℕ := ⌈Real.sqrt 120⌉₊

theorem ceil_sqrt_product :
  ceil_sqrt_3 * ceil_sqrt_12 * ceil_sqrt_120 = 88 :=
by
  sorry

end ceil_sqrt_product_l133_133818


namespace angle_between_skew_lines_l133_133879

/-- Angle formed by skew lines m and n given the dihedral angle conditions --/
theorem angle_between_skew_lines (α β l m n : Type)
  [is_line α] [is_line β] [is_line l]
  [is_line m] [is_line n]
  (h1 : is_dihedral_angle α l β 60)
  (h2 : perpendicular m α)
  (h3 : perpendicular n β)
  (h4 : skew_lines m n) : angle m n = 60 := 
sorry

end angle_between_skew_lines_l133_133879


namespace wall_paint_area_l133_133383

theorem wall_paint_area
  (A₁ : ℕ) (A₂ : ℕ) (A₃ : ℕ) (A₄ : ℕ)
  (H₁ : A₁ = 32)
  (H₂ : A₂ = 48)
  (H₃ : A₃ = 32)
  (H₄ : A₄ = 48) :
  A₁ + A₂ + A₃ + A₄ = 160 :=
by
  sorry

end wall_paint_area_l133_133383


namespace roots_count_in_interval_l133_133277

noncomputable def f : ℝ → ℝ := sorry

theorem roots_count_in_interval :
  (∀ x : ℝ, f(2 + x) = f(2 - x)) ∧
  (∀ x : ℝ, f(7 + x) = f(7 - x)) ∧
  (f 0 = 0) →
  (finset.card {(x : ℝ) | f x = 0 ∧ -1000 ≤ x ∧ x ≤ 1000}.to_finset = 401) :=
by {
  -- skipping the proof, assume the theorem holds
  sorry
}

end roots_count_in_interval_l133_133277


namespace mixture_proportion_exists_l133_133120

-- Define the ratios and densities of the liquids
variables (k : ℝ) (ρ1 ρ2 ρ3 : ℝ) (m1 m2 m3 : ℝ)
variables (x y : ℝ)

-- Given conditions
def density_ratio : Prop := 
  ρ1 = 6 * k ∧ ρ2 = 3 * k ∧ ρ3 = 2 * k

def mass_condition : Prop := 
  m2 / m1 ≤ 2 / 7

-- Must prove that a solution exists where the resultant density is the arithmetic mean
def mixture_density : Prop := 
  (m1 + m2 + m3) / ((m1 / ρ1) + (m2 / ρ2) + (m3 / ρ3)) = (ρ1 + ρ2 + ρ3) / 3

-- Statement (No proof provided)
theorem mixture_proportion_exists (k : ℝ) (m1 m2 m3 : ℝ) (ρ1 ρ2 ρ3 : ℝ) (x y : ℝ) :
  density_ratio k ρ1 ρ2 ρ3 →
  mass_condition m1 m2 →
  mixture_density m1 m2 m3 k ρ1 ρ2 ρ3 :=
sorry

end mixture_proportion_exists_l133_133120


namespace divisors_of_2700_l133_133829

def prime_factors_2700 : ℕ := 2^2 * 3^3 * 5^2

def number_of_positive_divisors (n : ℕ) (a b c : ℕ) : ℕ :=
  (a + 1) * (b + 1) * (c + 1)

theorem divisors_of_2700 : number_of_positive_divisors 2700 2 3 2 = 36 := by
  sorry

end divisors_of_2700_l133_133829


namespace find_a_l133_133896

theorem find_a (a : ℝ) (A : Set ℝ) (B : Set ℝ)
  (hA : A = {a^2, a + 1, 3})
  (hB : B = {a - 3, 2a - 1, a^2 + 1})
  (hIntersect : A ∩ B = {3}) : a = 6 ∨ a = Real.sqrt 2 ∨ a = - Real.sqrt 2 :=
by
  sorry

end find_a_l133_133896


namespace ellipse_area_condition_l133_133078

open Real

theorem ellipse_area_condition (m : ℝ) :
  (∀ x y : ℝ, ((x^2)/3 + y^2 = 1) → (y = x + m) → 
  let F₁ := (-sqrt 2, 0)
      F₂ := (sqrt 2, 0)
  in distance F₁ (x, y) = 2 * distance F₂ (x, y)) → 
  m = -sqrt 2 / 3 :=
by
  sorry

end ellipse_area_condition_l133_133078


namespace arithmetic_sequence_term_number_l133_133646

theorem arithmetic_sequence_term_number :
  ∀ (a : ℕ → ℤ) (n : ℕ),
    (a 1 = 1) →
    (∀ m, a (m + 1) = a m + 3) →
    (a n = 2014) →
    n = 672 :=
by
  -- conditions
  intro a n h1 h2 h3
  -- proof skipped
  sorry

end arithmetic_sequence_term_number_l133_133646


namespace gcd_of_all_three_digit_palindromes_is_one_l133_133691

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define a function to calculate the gcd of a list of numbers
def gcd_list (l : List ℕ) : ℕ :=
  l.foldr Nat.gcd 0

-- The main theorem that needs to be proven
theorem gcd_of_all_three_digit_palindromes_is_one :
  gcd_list (List.filter is_palindrome {n | 100 ≤ n ∧ n ≤ 999}.toList) = 1 :=
by
  sorry

end gcd_of_all_three_digit_palindromes_is_one_l133_133691


namespace quadratic_minimum_eq_one_l133_133117

variable (p q : ℝ)

theorem quadratic_minimum_eq_one (hq : q = 1 + p^2 / 18) : 
  ∃ x : ℝ, 3 * x^2 + p * x + q = 1 :=
by
  sorry

end quadratic_minimum_eq_one_l133_133117


namespace angle_between_PQ_and_CD_is_90_l133_133245

-- Define cyclic quadrilaterals, midpoints, perpendiculars, and intersection geometry
variable {A B C D P E F Q : Type}
variable [MetricGeometry : EuclideanGeometry ℝ]

-- Conditions corresponding to the original problem
axiom cyclic_quadrilateral (cyclic : CyclicQuad A B C D)
axiom diagonals_intersect_at_P (intersect : IntersectAt AC BD P)
axiom APB_is_obtuse (obtuse_angle : ObtuseAngle (∠ A P B))
axiom E_F_midpoints (midp_E : IsMidpoint E A D) (midp_F : IsMidpoint F B C)
axiom perpendiculars_intersect_at_Q (perpend_E : Perpendicular E A C Q) (perpend_F : Perpendicular F B D Q)

-- The theorem to prove the angle between PQ and CD is 90 degrees
theorem angle_between_PQ_and_CD_is_90 (angle_90 : ∠ P Q C D = 90°) : 
  cyclic_quadrilateral cyclic →
  diagonals_intersect_at_P intersect →
  APB_is_obtuse obtuse_angle →
  E_F_midpoints midp_E midp_F →
  perpendiculars_intersect_at_Q perpend_E perpend_F →
  angle_90 :=
by sorry

end angle_between_PQ_and_CD_is_90_l133_133245


namespace no_base_b_square_of_integer_l133_133097

theorem no_base_b_square_of_integer (b : ℕ) : ¬(∃ n : ℕ, n^2 = b^2 + 3 * b + 1) → b < 4 ∨ b > 8 := by
  sorry

end no_base_b_square_of_integer_l133_133097


namespace elisa_lap_time_improvement_l133_133428

theorem elisa_lap_time_improvement :
  (let initial_laps := 15
       initial_time := 30
       improved_laps := 20
       improved_time := 36
       initial_lap_time := initial_time / initial_laps
       improved_lap_time := improved_time / improved_laps
   in initial_lap_time - improved_lap_time) = 0.2 :=
by
  sorry

end elisa_lap_time_improvement_l133_133428


namespace dot_product_equivalent_l133_133131

theorem dot_product_equivalent (a b : ℝ × ℝ × ℝ) (h₁ : a = (3, 5, -4)) (h₂ : b = (2, 1, 8)) :
  (a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2) = -21 :=
by
  rw [h₁, h₂]
  simp
  norm_num
  sorry

end dot_product_equivalent_l133_133131


namespace gcd_three_digit_palindromes_l133_133681

theorem gcd_three_digit_palindromes : ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → Nat.gcd (102 * a + 10 * b) 1 = 1 :=
by
  intros a b h
  sorry

end gcd_three_digit_palindromes_l133_133681


namespace period_of_f_axis_of_symmetry_value_of_f_at_B_l133_133524

-- Define the trigonometric function f
def f (x : ℝ) : ℝ := 2 * sin x * cos (π / 2 - x) - sqrt 3 * sin (π + x) * cos x + sin (π / 2 + x) * cos x

-- Define the given conditions in the problem
variables (a b c : ℝ)
variable (A B C : ℝ)
variable (h1 : c * sin A = sqrt 3 * a * cos C)
variable (h2 : (a - c) * (a + c) = b * (b - c))

-- Prove the period of the function f is π
theorem period_of_f : is_periodic f π :=
sorry

-- Prove the equation of the axis of symmetry of f
theorem axis_of_symmetry : ∃ k : ℤ, ∀ x : ℝ, x = π / 3 + k * π / 2 :=
sorry

-- Prove that the value of f(B) is 5/2
theorem value_of_f_at_B (h3 : B = π - A - C) : f B = 5 / 2 :=
sorry

end period_of_f_axis_of_symmetry_value_of_f_at_B_l133_133524


namespace sum_of_solutions_eq_35_over_3_l133_133156

theorem sum_of_solutions_eq_35_over_3 (a b : ℝ) 
  (h1 : 2 * a + b = 14) (h2 : a + 2 * b = 21) : 
  a + b = 35 / 3 := 
by
  sorry

end sum_of_solutions_eq_35_over_3_l133_133156


namespace part1_part2_l133_133099

-- Define the initial conditions and the given inequality.
def condition1 (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def condition2 (m : ℝ) (x : ℝ) : Prop := x = (1/2)^(m - 1) ∧ 1 < m ∧ m < 2

-- Definitions of the correct ranges
def range_x (x : ℝ) : Prop := 1/2 < x ∧ x < 3/4
def range_a (a : ℝ) : Prop := 1/3 ≤ a ∧ a ≤ 1/2

-- Mathematical equivalent proof problem
theorem part1 {x : ℝ} (h1 : condition1 x (1/4)) (h2 : ∃ (m : ℝ), condition2 m x) : range_x x :=
sorry

theorem part2 {a : ℝ} (h : ∀ x : ℝ, (1/2 < x ∧ x < 1) → condition1 x a) : range_a a :=
sorry

end part1_part2_l133_133099


namespace sqrt_product_of_powers_eq_l133_133407

theorem sqrt_product_of_powers_eq :
  ∃ (x y z : ℕ), prime x ∧ prime y ∧ prime z ∧ x = 2 ∧ y = 3 ∧ z = 5 ∧
  sqrt (x^4 * y^6 * z^2) = 540 := by
  use 2, 3, 5
  show prime 2, from prime_two
  show prime 3, from prime_three
  show prime 5, from prime_five
  show 2 = 2, from rfl
  show 3 = 3, from rfl
  show 5 = 5, from rfl
  sorry

end sqrt_product_of_powers_eq_l133_133407


namespace sum_xy_eq_l133_133880

theorem sum_xy_eq (n : ℕ) (xs ys : Fin n → ℕ) (hpos : ∀ i, xs i > 0 ∧ ys i > 0)
  (hsol : ∀ i, (xs i) * (ys i) = 6 * (xs i + ys i)) : 
  (Finset.univ.sum (λ i, xs i + ys i)) = 290 :=
by
  sorry

end sum_xy_eq_l133_133880


namespace ellipse_equation_l133_133479

-- Define the conditions and problem statement
variable (F1 F2 : ℝ × ℝ) (a b : ℝ) (AB : ℝ × ℝ)

-- Ellipse definition and conditions:
ab_conditions : (a > b) ∧ (b > 0)
ellipse_eq : (AB.1 / a) ^ 2 + (AB.2 / b) ^ 2 = 1

-- Given that AB passes through F2
passes_through_F2 : AB = F2

-- Given the constant perimeter of triangle F1AB is 16
perimeter_const : |(dist F1 AB)| + |(dist F1 F2)| + |(dist F2 AB)| = 16

-- Arithmetic sequence condition
arithmetic_sequence : 2 * |(dist F1 F2)| = |(dist F1 AB)| + |(dist F2 AB)|

-- Equivalent proof problem in Lean
theorem ellipse_equation (cond1 : ab_conditions)
                         (cond2 : ellipse_eq)
                         (cond3 : passes_through_F2)
                         (cond4 : perimeter_const)
                         (cond5 : arithmetic_sequence) :
  (a = 4) ∧ (b^2 = 12) := by
  sorry

end ellipse_equation_l133_133479


namespace find_some_value_l133_133732

theorem find_some_value (m n : ℝ) (some_value : ℝ) 
  (h₁ : m = n / 2 - 2 / 5)
  (h₂ : m + 2 = (n + some_value) / 2 - 2 / 5) :
  some_value = 4 := 
sorry

end find_some_value_l133_133732


namespace measure_of_angle_B_l133_133635

noncomputable def angle_opposite_side (a b c : ℝ) (h : (c^2)/(a+b) + (a^2)/(b+c) = b) : ℝ :=
  if h : (c^2)/(a+b) + (a^2)/(b+c) = b then 60 else 0

theorem measure_of_angle_B {a b c : ℝ} (h : (c^2)/(a+b) + (a^2)/(b+c) = b) : 
  angle_opposite_side a b c h = 60 :=
by
  sorry

end measure_of_angle_B_l133_133635


namespace right_triangle_area_l133_133822

theorem right_triangle_area (a b c : ℝ) (m_b : ℝ) (h1 : a = 10) (h2 : m_b = 13)
  (h3 : a^2 + b^2 = c^2)
  (h4 : m_b^2 = (2*a^2 + 2*c^2 - b^2) / 4) : 
  let area := (1/2) * a * b in
  area = 10 * Real.sqrt 69 := 
by 
  sorry

end right_triangle_area_l133_133822


namespace hyperbola_eccentricity_l133_133825

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (h_asymptotes : a / b = 4 / 3) 
  (eccentricity : ℝ := sqrt (1 + b^2 / a^2)) :
  eccentricity = 5 / 4 :=
sorry

end hyperbola_eccentricity_l133_133825


namespace proof_sampling_problem_l133_133666

def sampling_problem : Prop :=
  let n_total := 600
  let n_sampled := 50
  let first_num := 3
  let range_C := (496, 600)
  let sampling_rate := n_total / n_sampled
  let term (n : Nat) := first_num + sampling_rate * (n - 1)
  ∀ n : Nat, (43 ≤ n ∧ n ≤ 50 → 496 ≤ term n ∧ term n ≤ 600) → (card {n : Nat | 43 ≤ n ∧ n ≤ 50} = 8)

theorem proof_sampling_problem : sampling_problem := by
  sorry

end proof_sampling_problem_l133_133666


namespace Simson_line_bisects_PH_l133_133472

variables {A B C H P : Type}
variables [Geometry A] [Geometry B] [Geometry C] [Geometry H] [Geometry P]

-- Let ABC be a triangle with orthocenter H, and let P be a point on the circumcircle of triangle ABC.
def triangle_ABC (A B C : Point) : Triangle := by sorry
def orthocenter_H_of_triangle_ABC (A B C H : Point) (t : Triangle A B C) : Prop := by sorry
def P_on_circumcircle_of_triangle_ABC (A B C P : Point) (t : Triangle A B C) : Prop := by sorry

-- Define the Simson line of P with respect to triangle ABC.
def Simson_line_of_P (A B C P : Point) : Line := by sorry

-- Define the bisection of the segment PH by a line.
def bisects_PH (H P : Point) (l : Line) : Prop := by sorry

-- Theorem statement in Lean
theorem Simson_line_bisects_PH (A B C H P : Point)
    (t : Triangle A B C)
    (orthocenterH : orthocenter_H_of_triangle_ABC A B C H t)
    (onCircumcircle : P_on_circumcircle_of_triangle_ABC A B C P t) :
  bisects_PH H P (Simson_line_of_P A B C P) :=
by sorry

end Simson_line_bisects_PH_l133_133472


namespace mathematicians_correctness_l133_133271

theorem mathematicians_correctness :
  ∃ (scenario1_w1 s_w1 : ℕ) (scenario1_w2 s_w2 : ℕ) (scenario2_w1 s2_w1 : ℕ) (scenario2_w2 s2_w2 : ℕ),
    scenario1_w1 = 4 ∧ s_w1 = 7 ∧ scenario1_w2 = 3 ∧ s_w2 = 5 ∧
    scenario2_w1 = 8 ∧ s2_w1 = 14 ∧ scenario2_w2 = 3 ∧ s2_w2 = 5 ∧
    let total_white1 := scenario1_w1 + scenario1_w2,
        total_choco1 := s_w1 + s_w2,
        prob1 := (total_white1 : ℚ) / total_choco1,
        total_white2 := scenario2_w1 + scenario2_w2,
        total_choco2 := s2_w1 + s2_w2,
        prob2 := (total_white2 : ℚ) / total_choco2,
        prob_box1 := 4 / 7,
        prob_box2 := 3 / 5 in
    (prob1 = 7 / 12 ∧ prob2 = 11 / 19 ∧
    (19 / 35 < prob_box1 ∧ prob_box1 < prob_box2) ∧
    (prob_box1 ≠ 19 / 35 ∧ prob_box1 ≠ 3 / 5)) :=
sorry

end mathematicians_correctness_l133_133271


namespace value_of_expression_l133_133147

theorem value_of_expression {x y : ℝ} (h1 : x = 3) (h2 : y = 4) :
  (x^5 + 2*y^3) / 8 = 46.375 :=
by {
  -- Given conditions
  rw [h1, h2],
  -- Simplify expression
  calc (3:ℝ)^5 + 2*(4:ℝ)^3 = 243 + 2*64 : by norm_num
                        ... = 243 + 128 : by norm_num
                        ... = 371 : by norm_num
                        ... = 46.375 * 8 : by norm_num
                        ... = 46.375 : by simp,
  sorry
}

end value_of_expression_l133_133147


namespace average_score_girls_l133_133932

theorem average_score_girls (num_boys num_girls : ℕ) (avg_boys avg_class : ℕ) : 
  num_boys = 12 → 
  num_girls = 4 → 
  avg_boys = 84 → 
  avg_class = 86 → 
  ∃ avg_girls : ℕ, avg_girls = 92 :=
by
  intros h1 h2 h3 h4
  sorry

end average_score_girls_l133_133932


namespace line_k_x_intercept_l133_133988

theorem line_k_x_intercept :
  ∀ (x y : ℝ), 3 * x - 5 * y + 40 = 0 ∧ 
  ∃ m' b', (m' = 4) ∧ (b' = 20 - 4 * 20) ∧ 
  (y = m' * x + b') →
  ∃ x_inter, (y = 0) → (x_inter = 15) := 
by
  sorry

end line_k_x_intercept_l133_133988


namespace product_of_two_numbers_l133_133651

theorem product_of_two_numbers (a b : ℝ) (h1 : a + b = 70) (h2 : a - b = 10) : a * b = 1200 := 
by
  sorry

end product_of_two_numbers_l133_133651


namespace glove_selection_l133_133012

theorem glove_selection :
  let n := 6                -- Number of pairs
  let k := 4                -- Number of selected gloves
  let m := 1                -- Number of matching pairs
  let total_ways := n * 10 * 8 / 2  -- Calculation based on solution steps
  total_ways = 240 := by
  sorry

end glove_selection_l133_133012


namespace words_fully_lit_probability_l133_133788

/-- Definition stating the display states of the words "I", "love", "Gaoyou" --/
inductive LightState
  | allOff
  | loveOn
  | loveAndIOff
  | loveAndGaoyouOff
  | allLit

/-- Probability of the words being fully lit given the lighting conditions. --/
def words_lit_probability : ℚ :=
  1 / 3

theorem words_fully_lit_probability
  (initial_state : LightState)
  (∀s : LightState, s = LightState.loveOn → s = LightState.allLit ∨ s = LightState.loveAndIOff ∨ s = LightState.loveAndGaoyouOff) :
  initial_state = LightState.allLit → words_lit_probability = 1 / 3 :=
by
  sorry

end words_fully_lit_probability_l133_133788


namespace greatest_common_factor_of_three_digit_palindromes_l133_133669

def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b

def gcf (a b : ℕ) : ℕ := 
  if a = 0 then b else gcf (b % a) a

theorem greatest_common_factor_of_three_digit_palindromes : 
  ∃ g, (∀ n, is_palindrome n → g ∣ n) ∧ (∀ d, (∀ n, is_palindrome n → d ∣ n) → d ∣ g) :=
by
  use 101
  sorry

end greatest_common_factor_of_three_digit_palindromes_l133_133669


namespace length_of_train_is_correct_l133_133727

-- Define the conditions given in the problem
def speed_km_per_hr : ℝ := 60
def time_seconds : ℝ := 5

-- Convert speed to meters per second
def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)

-- Define the length of the train
def length_of_train : ℝ := speed_m_per_s * time_seconds

-- The theorem we want to prove
theorem length_of_train_is_correct : length_of_train = 83.35 :=
sorry

end length_of_train_is_correct_l133_133727


namespace calories_intake_l133_133183

/-- 
Given:
1. John burns 2300 calories a day.
2. John needs to burn 4000 calories to lose 1 pound.
3. John will take 80 days to lose 10 pounds.
Prove:
John eats 1800 calories a day.
-/
theorem calories_intake (daily_burn : ℕ) (cal_per_pound : ℕ) (days : ℕ) (pounds : ℕ)
  (daily_burn_eq : daily_burn = 2300)
  (cal_per_pound_eq : cal_per_pound = 4000)
  (days_eq : days = 80)
  (pounds_eq : pounds = 10) :
  let daily_deficit := (pounds * cal_per_pound) / days in
  (daily_burn - daily_deficit) = 1800 :=
by
  sorry

end calories_intake_l133_133183


namespace students_last_year_l133_133547

theorem students_last_year (students_this_year : ℝ) (increase_percent : ℝ) (last_year_students : ℝ) 
  (h1 : students_this_year = 960) 
  (h2 : increase_percent = 0.20) 
  (h3 : students_this_year = last_year_students * (1 + increase_percent)) : 
  last_year_students = 800 :=
by 
  sorry

end students_last_year_l133_133547


namespace no_rational_points_on_sqrt3_circle_l133_133784

theorem no_rational_points_on_sqrt3_circle (x y : ℚ) : x^2 + y^2 ≠ 3 :=
sorry

end no_rational_points_on_sqrt3_circle_l133_133784


namespace determine_hyperbola_equation_l133_133095

-- Defining the conditions
def hyperbola_equation (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (∃ c : ℝ, c = 6 ∧ c^2 = a^2 + b^2) ∧ (∃ k : ℝ, k = sqrt 3 ∧ (b = k * a))

-- Stating the theorem
theorem determine_hyperbola_equation : 
  ∀ (a b : ℝ), hyperbola_equation (a b) → ∃ (h : Prop), h = (a^2 = 9 ∧ b^2 = 27 ∧ (a > 0 ∧ b > 0 ∧ (h = (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)))) :=
by
  sorry

end determine_hyperbola_equation_l133_133095


namespace find_x_l133_133913

theorem find_x (x : ℝ) : (4^(x + 2) = 68 + 4^x) -> x = Real.logb 4 (68 / 15) :=
by
  sorry

end find_x_l133_133913


namespace rancher_problem_l133_133763

theorem rancher_problem (s c : ℕ) (h : 30 * s + 35 * c = 1500) : (s = 1 ∧ c = 42) ∨ (s = 36 ∧ c = 12) := 
by
  sorry

end rancher_problem_l133_133763


namespace toy_cars_in_third_box_l133_133566

theorem toy_cars_in_third_box (total_cars first_box second_box : ℕ) (H1 : total_cars = 71) 
    (H2 : first_box = 21) (H3 : second_box = 31) : total_cars - (first_box + second_box) = 19 :=
by
  sorry

end toy_cars_in_third_box_l133_133566


namespace sum_of_angles_l133_133944

theorem sum_of_angles (A B C x y : ℝ) 
  (hA : A = 34) 
  (hB : B = 80) 
  (hC : C = 30)
  (pentagon_angles_sum : A + B + (360 - x) + 90 + (120 - y) = 540) : 
  x + y = 144 :=
by
  sorry

end sum_of_angles_l133_133944


namespace least_common_multiple_of_wang_numbers_l133_133165

noncomputable def wang_numbers (n : ℕ) : List ℕ :=
  -- A function that returns the wang numbers in the set from 1 to n
  sorry

noncomputable def LCM (list : List ℕ) : ℕ :=
  -- A function that computes the least common multiple of a list of natural numbers
  sorry

theorem least_common_multiple_of_wang_numbers :
  LCM (wang_numbers 100) = 10080 :=
sorry

end least_common_multiple_of_wang_numbers_l133_133165


namespace angle_cosine_value_l133_133617

noncomputable def vector_cosine_angle (a b : ℝ) [inner_product_space ℝ (euclidean_space ℝ)] (angle_ab : real.angle) :
  real.cos(get_angle ((a + b) : euclidean_space ℝ) ((a - b) : euclidean_space ℝ)) = - sqrt 21 / 7 :=
  sorry

-- Definitions for conditions
def a_norm_one := ∥(1 : euclidean_space ℝ)∥ = 1
def b_norm_two := ∥(2 : euclidean_space ℝ)∥ = 2
def angle_ab_is_pi_over_3 := angle_ab = real.angle.pi / 3

-- Lean Statement
theorem angle_cosine_value (a b : ℝ) [inner_product_space ℝ (euclidean_space ℝ)]:
  a_norm_one → b_norm_two → angle_ab_is_pi_over_3 →
  real.cos(get_angle ((a + b) : euclidean_space ℝ) ((a - b) : euclidean_space ℝ)) = - sqrt 21 / 7 := 
  by
  intros h1 h2 h3
  rw [a_norm_one, b_norm_two, angle_ab_is_pi_over_3]
  exact vector_cosine_angle a b (real.angle.pi / 3)

end angle_cosine_value_l133_133617


namespace constant_chromosome_number_l133_133222

theorem constant_chromosome_number (rabbits : Type) 
  (sex_reproduction : rabbits → Prop)
  (maintain_chromosome_number : Prop)
  (meiosis : Prop)
  (fertilization : Prop) : 
  (meiosis ∧ fertilization) ↔ maintain_chromosome_number :=
sorry

end constant_chromosome_number_l133_133222


namespace gcf_of_three_digit_palindromes_is_one_l133_133685

-- Define a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define the greatest common factor (gcd) function
def gcd_of_all_palindromes : ℕ :=
  (Finset.range 999).filter is_palindrome |>.list.foldr gcd 0

-- State the theorem
theorem gcf_of_three_digit_palindromes_is_one :
  gcd_of_all_palindromes = 1 :=
sorry

end gcf_of_three_digit_palindromes_is_one_l133_133685


namespace circle_tangent_to_x_axis_at_origin_l133_133149

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 + Dx + Ey + F = 0)
  (h_tangent : ∃ x, x^2 + (0 : ℝ)^2 + Dx + E * 0 + F = 0 ∧ ∃ r : ℝ, ∀ x y, x^2 + (y - r)^2 = r^2) :
  D = 0 ∧ E ≠ 0 ∧ F ≠ 0 :=
by
  sorry

end circle_tangent_to_x_axis_at_origin_l133_133149


namespace g_coefficients_l133_133975

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 + 4*x + 2

def hasThreeDistinctRoots (p : Polynomial ℝ) : Prop :=
  ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ (p.eval r1 = 0) ∧ (p.eval r2 = 0) ∧ (p.eval r3 = 0)

noncomputable def g (x : ℝ) : ℝ := x^3 - x^2 + 72*x + 8

theorem g_coefficients :
  (polynomial.coeff g 2 = -1) ∧ (polynomial.coeff g 1 = 72) ∧ (polynomial.coeff g 0 = 8) :=
by
  sorry

end g_coefficients_l133_133975


namespace mathematicians_correct_l133_133257

noncomputable def scenario1 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 4 ∧ total1 = 7 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario2 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 8 ∧ total1 = 14 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario3 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  (19 / 35 < 4 / 7) ∧ (4 / 7 < 3 / 5)

noncomputable def probability (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : ℝ :=
  (whites1 + whites2) / (total1 + total2)

theorem mathematicians_correct :
  let whites1_s1 := 4 in
  let total1_s1 := 7 in
  let whites2_s1 := 3 in
  let total2_s1 := 5 in
  let whites1_s2 := 8 in
  let total1_s2 := 14 in
  let whites2_s2 := 3 in
  let total2_s2 := 5 in
  scenario1 whites1_s1 total1_s1 whites2_s1 total2_s1 →
  scenario2 whites1_s2 total1_s2 whites2_s2 total2_s2 →
  scenario3 whites1_s1 total1_s1 whites2_s2 total2_s2 →
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 ≤ 3 / 5 ∨
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 = 4 / 7 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 ≤ 3 / 5 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 = 4 / 7 :=
begin
  intros,
  sorry

end mathematicians_correct_l133_133257


namespace distance_P_to_xOy_plane_l133_133552

noncomputable def distance_to_plane (P : ℝ × ℝ × ℝ) : ℝ :=
| P.2.2 |

theorem distance_P_to_xOy_plane :
  distance_to_plane (1, -2, 3) = 3 :=
sorry

end distance_P_to_xOy_plane_l133_133552


namespace unit_cost_calculation_l133_133353

theorem unit_cost_calculation : 
  ∀ (total_cost : ℕ) (ounces : ℕ), total_cost = 84 → ounces = 12 → (total_cost / ounces = 7) :=
by
  intros total_cost ounces h1 h2
  sorry

end unit_cost_calculation_l133_133353


namespace val_total_value_is_38_80_l133_133667

def initial_nickels : Nat := 20
def initial_dimes (nickels : Nat) : Nat := 3 * nickels
def initial_quarters (dimes : Nat) : Nat := 2 * dimes

def value_in_usd (nickels dimes quarters : Nat) : Real :=
  (nickels * 0.05) + (dimes * 0.10) + (quarters * 0.25)

def new_nickels (initial_nickels : Nat) : Nat := 2 * initial_nickels
def canadian_nickels (new_nickels : Nat) : Nat := new_nickels / 2
def us_nickels (new_nickels : Nat) : Nat := new_nickels / 2

def value_of_canadian_nickels_in_usd (canadian_nickels : Nat) (exchange_rate : Real) : Real :=
  (canadian_nickels * 0.05) * exchange_rate

noncomputable def total_value_of_money : Real :=
  let initial_dimes := initial_dimes initial_nickels
  let initial_quarters := initial_quarters initial_dimes
  let initial_value := value_in_usd initial_nickels initial_dimes initial_quarters
  let newly_found_nickels := new_nickels initial_nickels
  let canadian_nickels := canadian_nickels newly_found_nickels
  let us_nickels := us_nickels newly_found_nickels
  let value_canadian := value_of_canadian_nickels_in_usd canadian_nickels 0.8
  let value_us := us_nickels * 0.05
  initial_value + value_canadian + value_us

theorem val_total_value_is_38_80 : total_value_of_money = 38.80 := by sorry

end val_total_value_is_38_80_l133_133667


namespace mathematicians_correctness_l133_133274

theorem mathematicians_correctness :
  ∃ (scenario1_w1 s_w1 : ℕ) (scenario1_w2 s_w2 : ℕ) (scenario2_w1 s2_w1 : ℕ) (scenario2_w2 s2_w2 : ℕ),
    scenario1_w1 = 4 ∧ s_w1 = 7 ∧ scenario1_w2 = 3 ∧ s_w2 = 5 ∧
    scenario2_w1 = 8 ∧ s2_w1 = 14 ∧ scenario2_w2 = 3 ∧ s2_w2 = 5 ∧
    let total_white1 := scenario1_w1 + scenario1_w2,
        total_choco1 := s_w1 + s_w2,
        prob1 := (total_white1 : ℚ) / total_choco1,
        total_white2 := scenario2_w1 + scenario2_w2,
        total_choco2 := s2_w1 + s2_w2,
        prob2 := (total_white2 : ℚ) / total_choco2,
        prob_box1 := 4 / 7,
        prob_box2 := 3 / 5 in
    (prob1 = 7 / 12 ∧ prob2 = 11 / 19 ∧
    (19 / 35 < prob_box1 ∧ prob_box1 < prob_box2) ∧
    (prob_box1 ≠ 19 / 35 ∧ prob_box1 ≠ 3 / 5)) :=
sorry

end mathematicians_correctness_l133_133274


namespace ellipse_area_condition_l133_133072

open Real

theorem ellipse_area_condition (m : ℝ) :
  (∀ x y : ℝ, ((x^2)/3 + y^2 = 1) → (y = x + m) → 
  let F₁ := (-sqrt 2, 0)
      F₂ := (sqrt 2, 0)
  in distance F₁ (x, y) = 2 * distance F₂ (x, y)) → 
  m = -sqrt 2 / 3 :=
by
  sorry

end ellipse_area_condition_l133_133072


namespace modified_monotonous_count_l133_133800

def is_modified_monotonous (n : ℕ) : Prop :=
  -- Definition that determines if a number is modified-monotonous
  -- Must include digit '5', and digits must form a strictly increasing or decreasing sequence
  sorry 

def count_modified_monotonous (n : ℕ) : ℕ :=
  2 * (8 * (2^8) + 2^8) + 1 -- Formula for counting modified-monotonous numbers including '5'

theorem modified_monotonous_count : count_modified_monotonous 5 = 4609 := 
  by 
    sorry

end modified_monotonous_count_l133_133800


namespace ellipse_intersection_area_condition_l133_133028

theorem ellipse_intersection_area_condition :
  let ellipse := ∀ (x y : ℝ), (x^2 / 3 + y^2 = 1)
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let line m := ∀ (x y : ℝ), (y = x + m)
  ∀ (A B : ℝ × ℝ),
    ((A.1^2 / 3 + A.2^2 = 1) ∧ (y = A.1 + m)) ∧
    ((B.1^2 / 3 + B.2^2 = 1) ∧ (y = B.1 + m)) ∧
    (area (F1, A, B) = 2 * area (F2, A, B)) →
    m = -Real.sqrt 2 / 3 := by sorry

end ellipse_intersection_area_condition_l133_133028


namespace count_parallel_or_perpendicular_pairs_l133_133415

-- Definitions of lines from the conditions
def line1 : ℝ → ℝ := λ x => 3 * x + 4
def line2 : ℝ → ℝ := λ x => (12 * x + 20) / 4
def line3 : ℝ → ℝ := λ x => (6 * x - 1) / 2
def line4 : ℝ → ℝ := λ x => (2 * x - 6) / 3
def line5 : ℝ → ℝ := λ x => (-15 * x + 5) / 5

-- Statement of the theorem
theorem count_parallel_or_perpendicular_pairs :
  ∃ (n : ℕ), n = 3 :=
sorry

end count_parallel_or_perpendicular_pairs_l133_133415


namespace find_x_l133_133643

noncomputable def radius : ℝ := 6
noncomputable def height : ℝ := 4
noncomputable def volume (R H : ℝ) : ℝ := π * R^2 * H

theorem find_x :
  ∃ x : ℝ,
    volume (radius + x) height = volume radius (height + 2 * x) ↔ x = 6 :=
by
  sorry

end find_x_l133_133643


namespace find_m_l133_133067

noncomputable def ellipse : set (ℝ × ℝ) := { p | (p.1)^2 / 3 + (p.2)^2 = 1 }
def f1 : ℝ × ℝ := (- real.sqrt 2, 0)
def f2 : ℝ × ℝ := (real.sqrt 2, 0)

def line (m : ℝ) : set (ℝ × ℝ) := { p | p.2 = p.1 + m }

theorem find_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line m ∧ B ∈ line m ∧
   2 * (abs ((f1.1 * (A.2 - B.2) + A.1 * (B.2 - f1.2) + B.1 * (f1.2 - A.2)) / 2))
 = abs ((f2.1 * (A.2 - B.2) + A.1 * (B.2 - f2.2) + B.1 * (f2.2 - A.2)) / 2)) →
  m = - real.sqrt 2 / 3 :=
sorry

end find_m_l133_133067


namespace remaining_score_is_80_l133_133006

def scores : List ℕ := [85, 95, 75, 65]
def average_score := 80
def total_students := 5

theorem remaining_score_is_80 (x : ℕ) :
  (list.sum scores + x) / total_students = average_score → x = 80 :=
by
  sorry

end remaining_score_is_80_l133_133006


namespace bus_ride_duration_l133_133991

theorem bus_ride_duration (total_hours : ℕ) (train_hours : ℕ) (walk_minutes : ℕ) (wait_factor : ℕ) 
    (h_total : total_hours = 8)
    (h_train : train_hours = 6)
    (h_walk : walk_minutes = 15)
    (h_wait : wait_factor = 2) : 
    let total_minutes := total_hours * 60
    let train_minutes := train_hours * 60
    let wait_minutes := wait_factor * walk_minutes
    let travel_minutes := total_minutes - train_minutes
    let bus_ride_minutes := travel_minutes - walk_minutes - wait_minutes
    bus_ride_minutes = 75 :=
by
  sorry

end bus_ride_duration_l133_133991


namespace max_unmarried_women_under_30_l133_133933

-- Define total number of people
def total_people : ℕ := 150

-- Define ratios
def women_ratio : ℚ := 3/5
def under_30_ratio_among_women : ℚ := 2/5
def married_ratio_among_total : ℚ := 1/2
def retired_or_unemployed_ratio_among_total : ℚ := 1/4

-- Calculate numbers
def number_of_women := (women_ratio * total_people).natAbs
def number_of_women_under_30 := (under_30_ratio_among_women * number_of_women).natAbs
def number_of_married_people := (married_ratio_among_total * total_people).natAbs
def number_of_retired_or_unemployed_people := (retired_or_unemployed_ratio_among_total * total_people).natAbs

-- Define the theorem to prove
theorem max_unmarried_women_under_30 : number_of_women_under_30 = 36 := by
  sorry

end max_unmarried_women_under_30_l133_133933


namespace cot_inverse_sum_l133_133838

theorem cot_inverse_sum : 
  cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 97 / 40 :=
by
  sorry

end cot_inverse_sum_l133_133838


namespace perry_income_l133_133281

def tax (I : ℝ) : ℝ :=
if I ≤ 5000 then 0.08 * I
else 0.08 * 5000 + 0.10 * (I - 5000)

theorem perry_income : ∃ I : ℝ, tax I = 950 ∧ I = 10500 := by
  -- proof skipped
  sorry

end perry_income_l133_133281


namespace greatest_possible_sum_l133_133767

variable {a b c d e : ℕ}
variable {x y z w u v : ℕ}

theorem greatest_possible_sum :
  (∃ (s : Finset ℕ), 
      s.card = 5 ∧ 
      let pairwise_sums := 
        (∑ (s₁, s₂) in (Finset.pairs s), s₁ + s₂) in 
      pairwise_sums = {203, 350, 298, 245, x, y, z, w, u, v}) → 
  x + y + z + w + u + v ≤ 4384 :=
by
  sorry

end greatest_possible_sum_l133_133767


namespace gcf_of_all_three_digit_palindromes_l133_133716

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ is_palindrome n

def gcf_of_palindromes : ℕ :=
  101

theorem gcf_of_all_three_digit_palindromes : 
  ∀ n, is_three_digit_palindrome n → 101 ∣ n := by
    sorry

end gcf_of_all_three_digit_palindromes_l133_133716


namespace triangle_count_l133_133137

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_not_equilateral (a b c : ℕ) : Prop :=
  a ≠ b ∨ b ≠ c ∨ a ≠ c

def is_not_isosceles (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_not_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 ≠ c^2

def valid_noncongruent_triangles : ℕ :=
  (Finset.range 20).card {a b c ∈ Finset.Ico 1 20 | is_valid_triangle a b c ∧ (a + b + c < 20) 
                         ∧ is_not_equilateral a b c ∧ is_not_isosceles a b c ∧ is_not_right_triangle a b c}

theorem triangle_count : valid_noncongruent_triangles = 15 := by
  sorry -- Proof is to be provided

end triangle_count_l133_133137


namespace find_angle_A_find_area_triangle_l133_133868

-- Define the problem conditions and theorems
theorem find_angle_A {a b c A B C : ℝ} 
  (h_eq_lengths : (b - c)^2 = a^2 - b * c)
  (h_triangle : Triangle a b c A B C) :
  A = π / 3 :=
by
  sorry

theorem find_area_triangle {a b c A B C : ℝ} 
  (h_a : a = 3)
  (h_sin : sin C = 2 * sin B)
  (h_angle_A : A = π / 3)
  (h_cos_rule : cos A = (b^2 + c^2 - a^2) / (2 * b * c))
  (h_triangle : Triangle a b c A B C) :
  (1 / 2) * b * c * sin A = 3 * sqrt 3 / 2 :=
by
  sorry

end find_angle_A_find_area_triangle_l133_133868


namespace measure_angle_RPQ_l133_133941

theorem measure_angle_RPQ (P R S Q : Type) 
[IsLineSegment R S] [IsPointOnLineSegment P R S] [IsLineSegment Q P] [IsBisector Q P (angle S Q R)]
[IsDistanceEqual (dist P Q) (dist P R)] 
(angle_RSQ : ℝ) (angle_RPQ : ℝ) 
(h1 : angle S Q R = angle_RSQ) (h2 : angle P Q R = angle_RSQ / 2)
(h3 : angle P R Q = angle P Q R) (h4 : angle P R Q + angle P Q R + angle_RPQ = 180)
(h5 : angle_RPQ = 4 * angle_RSQ) :
  angle_RPQ = 144 := 
  sorry


end measure_angle_RPQ_l133_133941


namespace trig_proof_find_area_l133_133196

variables {A B C : ℝ} {a b c : ℝ}

-- Problem (1)
theorem trig_proof (h : a * cos^2 (C / 2) + c * cos^2 (A / 2) = (3 / 2) * b) : 
  sin A + sin C = 2 * sin B :=
sorry

-- Problem (2)
noncomputable def triangle_area (b : ℝ) (d : ℝ) (A : ℝ) (c : ℝ) :=
  1 / 2 * b * c * sin A

theorem find_area (h1 : b = 2) (h2 : a * cos^2 (C / 2) + c * cos^2 (A / 2) = (3 / 2) * b)
  (h3 : ∀ A B C : ℝ, sin A + sin C = 2 * sin B)
  (h4 : ∃ x y : ℝ, x * y * cos A = 3) :
  triangle_area b h4.some.sqrt 3
sorry

end trig_proof_find_area_l133_133196


namespace planes_determined_by_four_points_l133_133483

-- Define the problem context
variables (A B C D : Point) -- Four points in space

-- Define the non-collinearity condition
def are_not_collinear (a b c : Point) : Prop := 
  ¬ ∃ (α β : ℝ), α ≠ 0 ∧ β ≠ 0 ∧ (c = α • a + β • b)

-- Define the main result to prove: The number of planes determined by four points
theorem planes_determined_by_four_points (h1 : ¬ are_not_collinear A B C)
                                         (h2 : ¬ are_not_collinear A B D)
                                         (h3 : ¬ are_not_collinear A C D)
                                         (h4 : ¬ are_not_collinear B C D) :
  ∃ (n : ℕ), n = 1 ∨ n = 4 :=
by
  sorry -- Proof to be filled later

end planes_determined_by_four_points_l133_133483


namespace number_of_moles_of_H2O_l133_133441

def reaction_stoichiometry (n_NaOH m_Cl2 : ℕ) : ℕ :=
  1  -- Moles of H2O produced according to the balanced equation with the given reactants

theorem number_of_moles_of_H2O 
  (n_NaOH : ℕ) (m_Cl2 : ℕ) 
  (h_NaOH : n_NaOH = 2) 
  (h_Cl2 : m_Cl2 = 1) :
  reaction_stoichiometry n_NaOH m_Cl2 = 1 :=
by
  rw [h_NaOH, h_Cl2]
  -- Would typically follow with the proof using the conditions and stoichiometric relation
  sorry  -- Proof step omitted

end number_of_moles_of_H2O_l133_133441


namespace cot_sum_eq_cot_sum_example_l133_133835

theorem cot_sum_eq (a b c d : ℝ) :
  cot (arccot a + arccot b + arccot c + arccot d) = (a * b * c * d - (a * b + b * c + c * d) + 1) / (a + b + c + d)
:= sorry

theorem cot_sum_example :
  cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 1021 / 420
:= sorry

end cot_sum_eq_cot_sum_example_l133_133835


namespace monthly_avg_growth_rate_25_max_avg_tourists_next_10_days_l133_133725

-- Definition of given conditions regarding tourists count in February and April
def tourists_in_february : ℕ := 16000
def tourists_in_april : ℕ := 25000

-- Theorem 1: Monthly average growth rate of tourists from February to April is 25%.
theorem monthly_avg_growth_rate_25 :
  (tourists_in_april : ℝ) = tourists_in_february * (1 + 0.25)^2 :=
sorry

-- Definition of given conditions for tourists count from May 1st to May 21st
def tourists_may_1_to_21 : ℕ := 21250
def max_total_tourists_may : ℕ := 31250 -- Expressed in thousands as 31.25 in millions

-- Theorem 2: Maximum average number of tourists per day in the next 10 days of May.
theorem max_avg_tourists_next_10_days :
  ∀ (a : ℝ), tourists_may_1_to_21 + 10 * a ≤ max_total_tourists_may →
  a ≤ 10000 :=
sorry

end monthly_avg_growth_rate_25_max_avg_tourists_next_10_days_l133_133725


namespace range_of_m_l133_133101

theorem range_of_m 
  (h : ∃ a b : ℤ, a < b ∧ ∀ x ∈ Set.Ioo (Real (m - Real.sqrt (m^2 - 4)) / 2) (Real (m + Real.sqrt (m^2 - 4)) / 2), x = a ∨ x = b) : 
  m ∈ Set.Icc (-8 / 5 : ℚ) (-2 / 3 : ℚ) ∨ m ∈ Set.Ioo (8 / 3 : ℚ) (18 / 5 : ℚ) := sorry

end range_of_m_l133_133101


namespace distance_B_to_line_AC_l133_133949

theorem distance_B_to_line_AC :
  let AB := (1, 1, 2)
  let AC := (2, 1, 1)
  (∃ d : ℝ, d = real.sqrt (66) / 6) :=
by
  sorry

end distance_B_to_line_AC_l133_133949


namespace find_inverse_value_l133_133146

theorem find_inverse_value (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x^3 + 6) : f⁻¹ 69 = real.cbrt 21 := by
  sorry

end find_inverse_value_l133_133146


namespace sum_of_real_solutions_eq_l133_133514

theorem sum_of_real_solutions_eq (a : ℝ) (ha : a > 1) :
  let f := λ x : ℝ, sqrt (a - sqrt (a + x)) - x in
  ∃ x : ℝ, f x = 0 ∧ x = (sqrt (4 * a - 3) - 1) / 2 :=
sorry

end sum_of_real_solutions_eq_l133_133514


namespace TP_bisects__l133_133662

open EuclideanGeometry

variables {Point : Type} [EuclideanGeometry.Space Point]

-- Consider the necessary points and circles
variables (O1 O2 T P A B : Point) (r1 r2 : ℝ)
variable h_circles_tangent : tangent_at_point (circle O1 r1) (circle O2 r2) T
variable h_AB_chord : chord (circle O1 r1) A B
variable h_AB_tangent : tangent_at_point (circle O2 r2) A

theorem TP_bisects_∠ATB :
  bisects_angle (line_through T P) ∠ATB :=
sorry

end TP_bisects__l133_133662


namespace train_length_correct_l133_133728

noncomputable def speed_kmph : ℝ := 60
noncomputable def time_sec : ℝ := 6

-- Conversion factor from km/hr to m/s
noncomputable def conversion_factor := (1000 : ℝ) / 3600

-- Speed in m/s
noncomputable def speed_mps := speed_kmph * conversion_factor

-- Length of the train
noncomputable def train_length := speed_mps * time_sec

theorem train_length_correct :
  train_length = 100.02 :=
by
  sorry

end train_length_correct_l133_133728


namespace inequality_not_always_true_l133_133133

theorem inequality_not_always_true (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z ≠ 0) : ¬(∀ (z : ℝ), xz > yz^2) :=
sorry

end inequality_not_always_true_l133_133133


namespace ellipse_intersection_area_condition_l133_133024

theorem ellipse_intersection_area_condition :
  let ellipse := ∀ (x y : ℝ), (x^2 / 3 + y^2 = 1)
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let line m := ∀ (x y : ℝ), (y = x + m)
  ∀ (A B : ℝ × ℝ),
    ((A.1^2 / 3 + A.2^2 = 1) ∧ (y = A.1 + m)) ∧
    ((B.1^2 / 3 + B.2^2 = 1) ∧ (y = B.1 + m)) ∧
    (area (F1, A, B) = 2 * area (F2, A, B)) →
    m = -Real.sqrt 2 / 3 := by sorry

end ellipse_intersection_area_condition_l133_133024


namespace ellipse_area_condition_l133_133076

open Real

theorem ellipse_area_condition (m : ℝ) :
  (∀ x y : ℝ, ((x^2)/3 + y^2 = 1) → (y = x + m) → 
  let F₁ := (-sqrt 2, 0)
      F₂ := (sqrt 2, 0)
  in distance F₁ (x, y) = 2 * distance F₂ (x, y)) → 
  m = -sqrt 2 / 3 :=
by
  sorry

end ellipse_area_condition_l133_133076


namespace num_correct_propositions_l133_133197

-- Definitions:
variables {Line Plane : Type*}
variables (m n : Line) (α β γ : Plane)

-- Assumptions corresponding to propositions
def prop_1 (h₁: m ⊆ α) (h₂: α ∥ β) : m ∥ β := sorry
def prop_2 (h₁: m ⊆ α) (h₂: n ⊆ α) (h₃: m ∥ β) (h₄: n ∥ β) : α ∥ β := sorry
def prop_3 (h₁: m ⊥ α) (h₂: m ⊥ β) (h₃: n ⊥ α) : n ⊥ β := sorry
def prop_4 (h₁: α ⊥ γ) (h₂: β ⊥ γ) (h₃: m ⊥ α) : m ⊥ β := sorry

-- Main statement:
theorem num_correct_propositions : 
  let correct_props := (prop_1 m n α β γ, prop_2 m n α β γ, prop_3 m n α β γ, prop_4 m n α β γ) in
  (correct_props.count (λ p, p = true)) = 2 := 
sorry

end num_correct_propositions_l133_133197


namespace max_value_of_expression_l133_133963

theorem max_value_of_expression (a b c d : ℝ) 
  (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb2 : b ≤ 2) (hc : 0 ≤ c) (hc1 : c ≤ 1) (hd : 0 ≤ d) (hd1 : d ≤ 1) :
  ∃ M, (M = 1 ∧ ∀ a b c d, 0 ≤ a → a ≤ 1 → 0 ≤ b → b ≤ 2 → 0 ≤ c → c ≤ 1 → 0 ≤ d → d ≤ 1 → 
  sqrt (sqrt (a * b * c * d)) + sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ≤ M) :=
sorry

end max_value_of_expression_l133_133963


namespace find_a_l133_133096

theorem find_a (a : ℝ) (f g : ℝ → ℝ) (h1 : ∀ x, f x = Real.log x) (h2 : ∀ x, g x = -f x) (h3 : g a = 1) : a = Real.exp (-1) :=
by
  sorry

end find_a_l133_133096


namespace distance_D_to_plane_ABC_is_2sqrt29_29_l133_133853

noncomputable def vector := (ℚ × ℚ × ℚ)

def dist_point_to_plane (A B C D : vector): ℚ :=
  let u := (B.1 - A.1, B.2 - A.2, B.3 - A.3) in
  let v := (C.1 - A.1, C.2 - A.2, C.3 - A.3) in
  let normal := (
    u.2 * v.3 - u.3 * v.2, 
    u.3 * v.1 - u.1 * v.3, 
    u.1 * v.2 - u.2 * v.1
  ) in
  let n_len := (normal.1 ^ 2 + normal.2 ^ 2 + normal.3 ^ 2).sqrt in
  let w := (D.1 - A.1, D.2 - A.2, D.3 - A.3) in
  let dist := (normal.1 * w.1 + normal.2 * w.2 + normal.3 * w.3).abs / n_len in
  dist

theorem distance_D_to_plane_ABC_is_2sqrt29_29 : 
  dist_point_to_plane (0,0,2) (0,2,1) (2,1,0) (2,0,1) = 2 * real.sqrt 29 / 29 := 
sorry

end distance_D_to_plane_ABC_is_2sqrt29_29_l133_133853


namespace value_of_AB_value_of_sin_2A_minus_pi_over_4_l133_133523

-- Definitions of the given conditions for triangle ABC
variables {A B C : Type} [is_triangle A B C]
variables (BC AC : ℝ) (sinC sinA : ℝ)

-- Conditions
axiom cond1 : BC = Real.sqrt 5
axiom cond2 : AC = 3
axiom cond3 : sinC = 2 * sinA

-- Questions 1: Prove the value of AB
theorem value_of_AB : AB = 2 * Real.sqrt 5 :=
by
  -- proof steps would go here, using sin rule and given conditions
  sorry

-- Questions 2: Prove the value of sin(2A - π / 4)
theorem value_of_sin_2A_minus_pi_over_4 : sin (2 * A - Real.pi / 4) = Real.sqrt 2 / 10 :=
by
  -- proof steps would go here, using trigonometric identities and given conditions
  sorry

end value_of_AB_value_of_sin_2A_minus_pi_over_4_l133_133523


namespace find_eccentricity_l133_133636

variables {a b x_N x_M : ℝ}
variable {e : ℝ}

-- Conditions
def line_passes_through_N (x_N : ℝ) (x_M : ℝ) : Prop :=
x_N ≠ 0 ∧ x_N = 4 * x_M

def hyperbola (x y a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def midpoint_x_M (x_M : ℝ) : Prop :=
∃ (x1 x2 y1 y2 : ℝ), (x1 + x2) / 2 = x_M

-- Proof Problem
theorem find_eccentricity
  (hN : line_passes_through_N x_N x_M)
  (hC : hyperbola x_N 0 a b)
  (hM : midpoint_x_M x_M) :
  e = 2 :=
sorry

end find_eccentricity_l133_133636


namespace weight_of_new_person_l133_133619

theorem weight_of_new_person (avg_increase : ℝ) (original_weight : ℝ) (num_persons : ℕ) (new_weight : ℝ) 
  (h1 : avg_increase = 3.8)
  (h2 : original_weight = 75)
  (h3 : num_persons = 15)
  (h4 : num_persons * avg_increase = new_weight - original_weight)
  : new_weight = 132 :=
by
  have h5 : 15 * 3.8 = new_weight - 75, from sorry,
  calc
    new_weight = 75 + 57 : by sorry
            ... = 132    : by sorry

end weight_of_new_person_l133_133619


namespace trig_evaluation_trig_identity_value_l133_133348

-- Problem 1: Prove the trigonometric evaluation
theorem trig_evaluation :
  (Real.cos (9 * Real.pi / 4)) + (Real.tan (-Real.pi / 4)) + (Real.sin (21 * Real.pi)) = (Real.sqrt 2 / 2) - 1 :=
by
  sorry

-- Problem 2: Prove the value given the trigonometric identity
theorem trig_identity_value (θ : ℝ) (h : Real.sin θ = 2 * Real.cos θ) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (2 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 8 / 7 :=
by
  sorry

end trig_evaluation_trig_identity_value_l133_133348


namespace four_digit_integers_count_l133_133905

theorem four_digit_integers_count :
  ∃ n : ℕ, n = 42 ∧
    ∀ x : ℕ, (1000 ≤ x ∧ x < 10000) →
    (∀ i j, i ≠ j → (x.digits 10).nth i ≠ (x.digits 10).nth j) →
    (x / 1000 ≠ 0) →
    (x % 5 = 0) →
    (List.maximum (x.digits 10) = some 6) →
    (n = x) := sorry

end four_digit_integers_count_l133_133905


namespace smallest_sum_of_squares_l133_133624

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 91) : x^2 + y^2 ≥ 109 :=
sorry

end smallest_sum_of_squares_l133_133624


namespace problem_sum_f_200_l133_133009

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def f (n : ℕ) : ℕ :=
  if is_square n then 0 else ⌊1 / (Real.fract (Real.sqrt n))⌋

def sum_f (m : ℕ) : ℕ :=
  (Finset.range m).sum f

theorem problem_sum_f_200 : sum_f 200 = 629 := 
  sorry

end problem_sum_f_200_l133_133009


namespace population_net_increase_l133_133539

def birth_rate : ℕ := 8
def birth_time : ℕ := 2
def death_rate : ℕ := 6
def death_time : ℕ := 2
def seconds_per_minute : ℕ := 60
def minutes_per_hour : ℕ := 60
def hours_per_day : ℕ := 24

theorem population_net_increase :
  (birth_rate / birth_time - death_rate / death_time) * (seconds_per_minute * minutes_per_hour * hours_per_day) = 86400 :=
by
  sorry

end population_net_increase_l133_133539


namespace total_photos_l133_133530

def n_pages1 : ℕ := 12
def photos_per_page1 : ℕ := 2
def n_pages2 : ℕ := 9
def photos_per_page2 : ℕ := 3

theorem total_photos : n_pages1 * photos_per_page1 + n_pages2 * photos_per_page2 = 51 := 
by 
  sorry

end total_photos_l133_133530


namespace heavier_athlete_l133_133487

noncomputable def findHeavierAthlete (w : List ℤ) (p : List ℤ) (i1 i2 : ℤ) (i3 : ℤ) : ℤ :=
  if h: w.length = 4 ∧ p.length = 5 ∧ 
         p = [99, 113, 125, 130, 144] ∧ 
         i1 ∉ p ∧ i2 ∉ p
  then 66
  else 0

theorem heavier_athlete (A B C D : ℤ) 
  (h1 : A, B, C, D are all integers) 
  (h2 : ∀ x ∈ [A, B, C, D], x ∈ [99, 113, 125, 130, 144]) 
  (h3 : ∀ x ∈ [99, 113, 125, 130, 144], x ∈ [A+B, A+C, A+D, B+C, B+D, C+D]) 
  (h4 : A and B did not weigh together) : 
  findHeavierAthlete [A, B, C, D] [99, 113, 125, 130, 144] A B = 66 :=
by
  sorry

end heavier_athlete_l133_133487


namespace sum_of_distances_is_31_l133_133971

noncomputable def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem sum_of_distances_is_31 : 
  let focus := (0, 1 / 4 : ℝ)
  let p1 := (-3, 9 : ℝ)
  let p2 := (1, 1 : ℝ)
  let p3 := (4, 16 : ℝ)
  let p4 := (-2, 4 : ℝ)
  euclidean_distance focus p1 +
  euclidean_distance focus p2 +
  euclidean_distance focus p3 +
  euclidean_distance focus p4 = 31 :=
by
  have h1 : euclidean_distance focus p1 = real.sqrt 85.5625, sorry
  have h2 : euclidean_distance focus p2 = real.sqrt 1.5625, sorry
  have h3 : euclidean_distance focus p3 = real.sqrt 264.0625, sorry
  have h4 : euclidean_distance focus p4 = real.sqrt 18.0625, sorry
  calc
    euclidean_distance focus p1 +
    euclidean_distance focus p2 +
    euclidean_distance focus p3 +
    euclidean_distance focus p4
      = real.sqrt 85.5625 + real.sqrt 1.5625 + real.sqrt 264.0625 + real.sqrt 18.0625 : by rw [h1, h2, h3, h4]
  ... = 9.25 + 1.25 + 16.25 + 4.25 : by sorry -- Numerical values for square roots
  ... = 31 : by norm_num

end sum_of_distances_is_31_l133_133971


namespace central_angle_of_sector_l133_133618

theorem central_angle_of_sector {r l : ℝ} 
  (h1 : 2 * r + l = 4) 
  (h2 : (1 / 2) * l * r = 1) : 
  l / r = 2 :=
by 
  sorry

end central_angle_of_sector_l133_133618


namespace a_drew_four_l133_133845

def players := {A, B, C, D, E}
def card_numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def scores : players → ℕ
| A => 11
| B => 4
| C => 7
| D => 16
| E => 17

def cards (p : players) : Set ℕ := sorry

theorem a_drew_four : 4 ∈ (cards A) :=
by
  sorry

end a_drew_four_l133_133845


namespace arithmetic_sequence_sum_l133_133940

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 2 + a 12 = 32) : a 3 + a 11 = 32 :=
sorry

end arithmetic_sequence_sum_l133_133940


namespace hexagon_side_length_l133_133236

theorem hexagon_side_length 
  (area_rectangle: ℝ) 
  (side_length_hexagon: ℝ) 
  (h1: area_rectangle = 12 * 15)
  (h2: (3 * real.sqrt 3 / 2) * side_length_hexagon^2 = area_rectangle) : 
  side_length_hexagon = 8.3 :=
by
  have area_hexagon := area_rectangle,
  rw h1 at area_hexagon,
  rw h2 at area_hexagon,
  sorry

end hexagon_side_length_l133_133236


namespace cot_inverse_sum_l133_133839

theorem cot_inverse_sum : 
  cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 97 / 40 :=
by
  sorry

end cot_inverse_sum_l133_133839


namespace relationship_x_y_l133_133088

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (OA OB OP PA AB : V) (x y λ : ℝ)

-- Conditions
axiom non_zero_OA : OA ≠ 0
axiom non_zero_OB : OB ≠ 0
axiom not_collinear : ∃ k : ℝ, OA ≠ k • OB
axiom two_OP_eq : 2 • OP = x • OA + y • OB
axiom PA_eq_lambda_AB : PA = λ • AB
axiom AB_def : AB = OB - OA

-- Problem statement
theorem relationship_x_y : x + y = 2 :=
sorry

end relationship_x_y_l133_133088


namespace hexagon_area_l133_133215

variable (S : ℝ) (htriangle : ∀ A B C : ℝ × ℝ, ∃ (triangle : Triangle) (acute : IsAcute triangle), Area triangle = S)

theorem hexagon_area (S : ℝ) (htriangle : ∀ A B C : ℝ × ℝ, ∃ (triangle : Triangle) (acute : IsAcute triangle), Area triangle = S) :
  ∃ (hexagon : Shape), Area hexagon = S / 2 := 
sorry

end hexagon_area_l133_133215


namespace greatest_common_factor_of_three_digit_palindromes_l133_133671

def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b

def gcf (a b : ℕ) : ℕ := 
  if a = 0 then b else gcf (b % a) a

theorem greatest_common_factor_of_three_digit_palindromes : 
  ∃ g, (∀ n, is_palindrome n → g ∣ n) ∧ (∀ d, (∀ n, is_palindrome n → d ∣ n) → d ∣ g) :=
by
  use 101
  sorry

end greatest_common_factor_of_three_digit_palindromes_l133_133671


namespace square_area_l133_133937

theorem square_area (s AG EG : ℝ) (ABCD : set (ℝ × ℝ))
                    (E : (ℝ × ℝ)) (F : (ℝ × ℝ)) (G : (ℝ × ℝ)) :
  (∀ A B C D : (ℝ × ℝ),
      distinct [A, B, C, D] ∧
      (A.1 = B.1 ∧ B.2 = C.2 ∧ C.1 = D.1 ∧ D.2 = A.2) ∧
      (A.1 + s, A.2) = B ∧
      (B.1, B.2 + s) = C ∧
      (C.1 - s, C.2) = D ∧
      (D.1, D.2 - s) = A) →
  E.1 >= A.1 ∧ E.1 <= B.1 ∧ F.2 >= B.2 ∧ F.2 <= C.2 ∧
  (G.1 = A.1 + 8 ∧ G.2 = A.2 + 10) ∧
  dist E G = 10 ∧ dist A G = 8 →
  s = 18 :=
by
  basis using sorry

end square_area_l133_133937


namespace fisherman_catch_l133_133361

theorem fisherman_catch (M : ℝ) (h1 : 0 < M) :
  (let largest := 0.35 * M in
   let remaining := 0.65 * M in
   let smallest := (5 / 13) * remaining in
   let cooked := M - (largest + smallest) in
   let n : ℕ := 10 in
   cooked = 0.4 * M) :=
sorry

end fisherman_catch_l133_133361


namespace master_wang_resting_on_sunday_again_l133_133211

theorem master_wang_resting_on_sunday_again (n : ℕ) 
  (works_days := 8) 
  (rest_days := 2) 
  (week_days := 7) 
  (cycle_days := works_days + rest_days) 
  (initial_rest_saturday_sunday : Prop) : 
  (initial_rest_saturday_sunday → ∃ n : ℕ, (week_days * n) % cycle_days = rest_days) → 
  (∃ n : ℕ, n = 7) :=
by
  sorry

end master_wang_resting_on_sunday_again_l133_133211


namespace Da_Yan_sequence_20th_term_l133_133615

noncomputable def Da_Yan_sequence_term (n: ℕ) : ℕ :=
  if n % 2 = 0 then
    (n^2) / 2
  else
    (n^2 - 1) / 2

theorem Da_Yan_sequence_20th_term : Da_Yan_sequence_term 20 = 200 :=
by
  sorry

end Da_Yan_sequence_20th_term_l133_133615


namespace quadratic_roots_l133_133011

theorem quadratic_roots (b c : ℝ) (h : b^2 - 4 * c > 0) :
  ∃ x : ℝ, (x^2 + b * x + c = 0) ∧ (x = (-b + real.sqrt (b^2 - 4 * c)) / 2 ∨ x = (-b - real.sqrt (b^2 - 4 * c)) / 2) :=
by
  sorry

end quadratic_roots_l133_133011


namespace count_digits_l133_133453

theorem count_digits (X : ℕ) 
  (h1 : 0 ≤ X) 
  (h2 : X ≤ 9) : 
  (count (fun X => (6 + X) % 3 = 0) (list.range' 0 10) = 4) :=
sorry

end count_digits_l133_133453


namespace bar_charts_cannot_show_increase_or_decrease_l133_133395

theorem bar_charts_cannot_show_increase_or_decrease (Q : Type) (bar_chart_represents_amount : Q -> Prop) :
  ∀ (q : Q), bar_chart_represents_amount q → ¬ (bar_chart_represents_amount q → bar_charts_show_increase_or_decrease q) :=
by
  intros
  assume h
  sorry

end bar_charts_cannot_show_increase_or_decrease_l133_133395


namespace grant_made_79_dollars_l133_133903

def sell_price (original_price : ℕ) (discount : ℕ) : ℕ := original_price - (original_price * discount / 100)

theorem grant_made_79_dollars :
  let baseball_cards := 25
  let baseball_bat := 10
  let baseball_glove := sell_price 30 20
  let baseball_cleats := 2 * 10
  in baseball_cards + baseball_bat + baseball_glove + baseball_cleats = 79 :=
by
  let baseball_cards := 25
  let baseball_bat := 10
  let baseball_glove := sell_price 30 20
  let baseball_cleats := 2 * 10
  show baseball_cards + baseball_bat + baseball_glove + baseball_cleats = 79
  sorry

end grant_made_79_dollars_l133_133903


namespace natasha_average_speed_climbing_l133_133590

theorem natasha_average_speed_climbing (D : ℝ) (h1 : 4 + 2 = 6) (h2 : (2 * D) / 6 = 1.5) : D / 4 = 1.125 :=
by
  have h3 : 2 * D = 6 * 1.5 := by sorry
  have h4 : D = 4.5 := by sorry
  show D / 4 = 1.125 from
    calc
      D / 4 = 4.5 / 4 : by rw [h4]
      ... = 1.125   : by norm_num

end natasha_average_speed_climbing_l133_133590


namespace fraction_identity_l133_133014

theorem fraction_identity (x y z : ℤ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) :
  (x + y) / (3 * y - 2 * z) = 5 :=
by
  sorry

end fraction_identity_l133_133014


namespace xy_product_l133_133593

theorem xy_product (x y : ℝ) 
  (h1 : 2^x = 16^(y + 1)) 
  (h2 : 27^y = 3^(x - 2)) : x * y = 8 := 
by 
  sorry

end xy_product_l133_133593


namespace max_number_of_9_letter_palindromes_l133_133438

theorem max_number_of_9_letter_palindromes : 26^5 = 11881376 :=
by sorry

end max_number_of_9_letter_palindromes_l133_133438


namespace arithmetic_seq_necessary_not_sufficient_l133_133207

noncomputable def arithmetic_sequence (a b c : ℝ) : Prop :=
  a + c = 2 * b

noncomputable def proposition_B (a b c : ℝ) : Prop :=
  b ≠ 0 ∧ (a / b) + (c / b) = 2

theorem arithmetic_seq_necessary_not_sufficient (a b c : ℝ) :
  (arithmetic_sequence a b c → proposition_B a b c) ∧ 
  (∃ a' b' c', arithmetic_sequence a' b' c' ∧ ¬ proposition_B a' b' c') := by
  sorry

end arithmetic_seq_necessary_not_sufficient_l133_133207


namespace draw_probability_l133_133328

-- Define probabilities
def p_jian_wins : ℝ := 0.4
def p_gu_not_wins : ℝ := 0.6

-- Define the probability of the game ending in a draw
def p_draw : ℝ := p_gu_not_wins - p_jian_wins

-- State the theorem to be proved
theorem draw_probability : p_draw = 0.2 :=
by
  sorry

end draw_probability_l133_133328


namespace natalie_bushes_for_zucchinis_l133_133816

theorem natalie_bushes_for_zucchinis :
  (bushes containers zucchinis : ℕ) 
  (yields_containers : containers = 10 * bushes) 
  (trade_ratio : 2 * containers = 3 * zucchinis)
  (target_zucchinis : zucchinis = 72) :
  ∃ bush_count : ℕ, bush_count = 15 :=
by
  sorry

end natalie_bushes_for_zucchinis_l133_133816


namespace minimum_A_l133_133895

noncomputable def minA : ℝ := (1 + Real.sqrt 2) / 2

theorem minimum_A (x y z w : ℝ) (A : ℝ) 
    (h : xy + 2 * yz + zw ≤ A * (x^2 + y^2 + z^2 + w^2)) :
    A ≥ minA := 
sorry

end minimum_A_l133_133895


namespace carla_brush_length_l133_133801

theorem carla_brush_length 
  (C : ℝ)
  (Carmen_brush_inches : ℝ)
  (Carmen_brush_cm : ℝ)
  (h1 : Carmen_brush_cm = 45)
  (h2 : Carmen_brush_inches = 1.5 * C)
  (conversion_factor : ℝ)
  (h3 : conversion_factor = 2.54)
  (h4 : Carmen_brush_inches = Carmen_brush_cm / conversion_factor) : 
  C ≈ 11.81 :=
by
  sorry

end carla_brush_length_l133_133801


namespace gcd_of_all_three_digit_palindromes_is_one_l133_133696

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define a function to calculate the gcd of a list of numbers
def gcd_list (l : List ℕ) : ℕ :=
  l.foldr Nat.gcd 0

-- The main theorem that needs to be proven
theorem gcd_of_all_three_digit_palindromes_is_one :
  gcd_list (List.filter is_palindrome {n | 100 ≤ n ∧ n ≤ 999}.toList) = 1 :=
by
  sorry

end gcd_of_all_three_digit_palindromes_is_one_l133_133696


namespace garageHasWheels_l133_133363

-- Define the conditions
def bikeWheelsPerBike : Nat := 2
def bikesInGarage : Nat := 10

-- State the theorem to be proved
theorem garageHasWheels : bikesInGarage * bikeWheelsPerBike = 20 := by
  sorry

end garageHasWheels_l133_133363


namespace correlation_non_deterministic_relationship_l133_133329

theorem correlation_non_deterministic_relationship
  (independent_var_fixed : Prop)
  (dependent_var_random : Prop)
  (correlation_def : Prop)
  (correlation_randomness : Prop) :
  (correlation_def → non_deterministic) :=
by
  sorry

end correlation_non_deterministic_relationship_l133_133329


namespace find_ck_l133_133289

-- Definitions based on the conditions
def arithmetic_sequence (d : ℕ) (n : ℕ) : ℕ :=
  1 + (n - 1) * d

def geometric_sequence (r : ℕ) (n : ℕ) : ℕ :=
  r^(n - 1)

def combined_sequence (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_sequence d n + geometric_sequence r n

-- Given conditions
variable {d r k : ℕ}
variable (hd : combined_sequence d r (k-1) = 250)
variable (hk : combined_sequence d r (k+1) = 1250)

-- The theorem statement
theorem find_ck : combined_sequence d r k = 502 :=
  sorry

end find_ck_l133_133289


namespace greatest_number_of_plants_in_one_row_l133_133140

open Nat

/-- The GCD of the number of sunflower, corn, and tomato plants -/
def gcd_plants : ℕ :=
  gcd (gcd 45 81) 63

/-- Prove that the greatest number of plants we can put in one row is 9. -/
theorem greatest_number_of_plants_in_one_row :
  gcd_plants = 9 := by
  sorry

end greatest_number_of_plants_in_one_row_l133_133140


namespace sum_of_cubes_l133_133457

theorem sum_of_cubes
  (a b c : ℝ)
  (h₁ : a + b + c = 7)
  (h₂ : ab + ac + bc = 9)
  (h₃ : a * b * c = -18) :
  a^3 + b^3 + c^3 = 100 := by
  sorry

end sum_of_cubes_l133_133457


namespace geometric_sequence_common_ratio_arithmetic_sequence_a1_l133_133474

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_a : ∀ n, a n = a 1 * 2^n) 
  (h_S4 : S 4 = a 5 - a 1) : 
  (2 = 2) := by 
  sorry

theorem arithmetic_sequence_a1 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (T : ℕ → ℝ)
  (h_b : ∀ n, b n = log 2 (a (n + 1))) 
  (h_t4 : T 4 = 2 * b 5) : 
  a 1 = 1 := by 
  sorry

end geometric_sequence_common_ratio_arithmetic_sequence_a1_l133_133474


namespace find_angle_B_find_perimeter_l133_133928

theorem find_angle_B
  (a b c : ℝ)
  (A B C : ℝ)
  (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0)
  (h₃ : A + B + C = π)
  (h₄ : ∃ (a b c : ℝ), a = b * sin A / sin B ∧ b = b ∧ c = b * sin C / sin B)
  (h₅ : sin B * (a * cos B + b * cos A) = sqrt 3 * c * cos B) :
  B = π / 3 := 
sorry

theorem find_perimeter 
  (a b c : ℝ)
  (A B C : ℝ)
  (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0)
  (h₃ : b = 2 * sqrt 3)
  (h₄ : (1/2) * a * c * sin B = 2 * sqrt 3)
  (h₅ : B = π / 3)
  (h₆ : a^2 + c^2 - 2 * a * c * cos B = b^2) :
  a + b + c = 6 + 2 * sqrt 3 :=
sorry

end find_angle_B_find_perimeter_l133_133928


namespace train_a_distance_at_meeting_l133_133735

noncomputable def train_a_speed : ℝ := 75 / 3
noncomputable def train_b_speed : ℝ := 75 / 2
noncomputable def relative_speed : ℝ := train_a_speed + train_b_speed
noncomputable def time_until_meet : ℝ := 75 / relative_speed
noncomputable def distance_traveled_by_train_a : ℝ := train_a_speed * time_until_meet

theorem train_a_distance_at_meeting : distance_traveled_by_train_a = 30 := by
  sorry

end train_a_distance_at_meeting_l133_133735


namespace solution_set_l133_133204

variables {f : ℝ → ℝ}

-- Condition 1: f(x) is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f(-x) = - (f x)

-- Condition 2: f is defined on the domain [-5, 5]
def in_domain (f : ℝ → ℝ) (x : ℝ) : Prop :=
-5 ≤ x ∧ x ≤ 5

-- prove the solution set to f(x) < 0
theorem solution_set (f_odd : odd_function f)
  (graph_condition : ∀ x, (0 ≤ x ∧ x ≤ 5) → (f x < 0 ↔ (2 < x ∧ x ≤ 5))) :
∀ x, in_domain f x → (f x < 0 ↔ ((-2 < x ∧ x < 0) ∨ (2 < x ∧ x ≤ 5))) :=
by
  sorry

end solution_set_l133_133204


namespace triangle_is_isosceles_l133_133554

open EuclideanGeometry

/-- In triangle \( \triangle ABC \), the bisector of \( \angle BAC \) intersects side \( BC \) at point \( A_0 \). 
    Points \( B_0 \) and \( C_0 \) are similarly defined. Let \( A_1 \) be the reflection of point \( A_0 \) 
    in the midpoint of side \( BC \). Points \( B_1 \) and \( C_1 \) are similarly defined.
    Let the circumcircle of \( \triangle A_1 B_1 C_1 \) intersect line \( BC \) again at point \( X \) 
    (if the circle is tangent to \( BC \), then \( X \) coincides with \( A_1 \), similarly for other cases). 
    Let the circumcircle intersect line \( CA \) again at point \( Y \) and line \( AB \) again at point \( Z \).
    If points \( X \), \( Y \), and \( Z \) are the perpendicular projections of some point \( P \) onto lines \( BC \), 
    \( CA \), and \( AB \) respectively, then triangle \( \triangle ABC \) is isosceles.
-/
theorem triangle_is_isosceles (A B C A₀ B₀ C₀ A₁ B₁ C₁ X Y Z P : Point)
  (hA₀ : is_angle_bisector A B C A₀)
  (hB₀ : is_angle_bisector B C A B₀)
  (hC₀ : is_angle_bisector C A B C₀)
  (hA₁ : is_reflection A₀ (midpoint B C) A₁)
  (hB₁ : is_reflection B₀ (midpoint C A) B₁)
  (hC₁ : is_reflection C₀ (midpoint A B) C₁)
  (hCircumcircle : intersect_circumcircle A₁ B₁ C₁)
  (hX : intersection_line_circumcircle X (line B C) A₁)
  (hY : intersection_line_circumcircle Y (line C A) B₁)
  (hZ : intersection_line_circumcircle Z (line A B) C₁)
  (hP_proj_X : perpendicular_projection P (line B C) X)
  (hP_proj_Y : perpendicular_projection P (line C A) Y)
  (hP_proj_Z : perpendicular_projection P (line A B) Z) :
  is_isosceles_triangle A B C := sorry

end triangle_is_isosceles_l133_133554


namespace range_of_a_l133_133910

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x < a → x ^ 2 > 1 ∧ ¬(x ^ 2 > 1 → x < a)) : a ≤ -1 :=
sorry

end range_of_a_l133_133910


namespace ellipse_condition_l133_133290

-- Define fixed points A and B
variables {A B : ℝ × ℝ}

-- Define the Euclidean distance
def distance (P Q : ℝ × ℝ) : ℝ := ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2).sqrt

-- Define the condition for point P
def condition (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  distance P A + distance P B = 2 * distance A B

-- The main statement
theorem ellipse_condition :
  ∀ (P : ℝ × ℝ), condition P A B ↔ 
    ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    ∃ (E : set (ℝ × ℝ)), (∀ Q ∈ E, distance Q A + distance Q B = a) ∧
    E = { Q | distance Q A + distance Q B = a } :=
sorry

end ellipse_condition_l133_133290


namespace spinner_final_direction_is_west_l133_133555

def initial_direction : string := "north"

def rotate (initial : string) (clockwise1 : Rational) (counterclockwise : Rational) (clockwise2 : Rational) : string :=
  let total_clockwise := (clockwise1.num * clockwise1.denom⁻¹) + (clockwise2.num * clockwise2.denom⁻¹)
  let total_counterclockwise := counterclockwise.num * counterclockwise.denom⁻¹
  let net_clockwise_quarters := (total_clockwise * 4) - (total_counterclockwise * 4)
  let final_quarters := (net_clockwise_quarters % 4)
  match final_quarters with
  | 0 => "north"
  | 1 => "east"
  | 2 => "south"
  | 3 => "west"
  | _ => "north"  -- should not hit this case

theorem spinner_final_direction_is_west :
  rotate initial_direction (7/2) (-21/4) (1/2) = "west" :=
  sorry

end spinner_final_direction_is_west_l133_133555


namespace children_l133_133225

theorem children's_book_pages (P : ℝ)
  (h1 : P > 0)
  (c1 : ∃ P_rem, P_rem = P - (0.2 * P))
  (c2 : ∃ P_today, P_today = (0.35 * (P - (0.2 * P))))
  (c3 : ∃ Pages_left, Pages_left = (P - (0.2 * P) - (0.35 * (P - (0.2 * P)))) ∧ Pages_left = 130) :
  P = 250 := by
  sorry

end children_l133_133225


namespace positive_difference_solutions_l133_133443

theorem positive_difference_solutions :
  let f (x : ℝ) := 9 - x^2 / 4
  let x1 := 12
  let x2 := -12
  |x1 - x2| = 24 :=
by
  let f (x : ℝ) := 9 - x^2 / 4
  have h : f 12 = -27 := sorry
  have h' : f (-12) = -27 := sorry
  have solution_correctness : ∀ x, ∃ (y : ℝ), y = x ∨ y = -x :=
    sorry
  let x1 := 12
  let x2 := -12
  have solutions := solution_correctness 12
  have diff := |x1 - x2|
  show diff = 24
  sorry

end positive_difference_solutions_l133_133443


namespace sum_of_solutions_eq_230_over_3_l133_133325

theorem sum_of_solutions_eq_230_over_3 :
  (∑ x in {y | y = |2*y - |50 - 2*y|| ∧ (50 - 2*y ≥ 0 ∨ 50 - 2*y < 0)}.to_finset, x) = 230 / 3 :=
by
  sorry

end sum_of_solutions_eq_230_over_3_l133_133325


namespace cos_theta_max_l133_133327

theorem cos_theta_max (θ : ℝ) (hθ : ∀ x, (sin x + 2 * cos x) ≤ (sin θ + 2 * cos θ)) : 
  cos θ = 2 * real.sqrt 5 / 5 :=
sorry

end cos_theta_max_l133_133327


namespace number_of_pairs_x_y_leq_2_pow_r_plus_1_l133_133569

theorem number_of_pairs_x_y_leq_2_pow_r_plus_1
  (r : ℕ) (p : Fin r → ℕ) (n : Fin r → ℕ)
  (hp : ∀ i j : Fin r, i ≠ j → Prime (p i) ∧ p i ≠ p j) :
  ∃ (max_pairs : ℕ), max_pairs ≤ 2^(r + 1) ∧ ∀ (x y : ℤ),
     (x ^ 3 + y ^ 3 = ∏ i, (p i) ^ (n i) → (x, y) ∈ finset.powerset.univ.filter (λ t, t.card ≤ max_pairs)) :=
begin
  sorry
end

end number_of_pairs_x_y_leq_2_pow_r_plus_1_l133_133569


namespace number_of_integer_solutions_l133_133153

theorem number_of_integer_solutions :
  ∃ (a : ℤ), (-1 - 2 * (a : ℝ)) < 0 ∧ (2 * (a : ℝ) - 4) < 0 ∧ set.finite {a : ℤ | (-1 - 2 * (a : ℝ)) < 0 ∧ (2 * (a : ℝ) - 4) < 0} ∧ set.count {a : ℤ | (-1 - 2 * (a : ℝ)) < 0 ∧ (2 * (a : ℝ) - 4) < 0} = 1 :=
by sorry

end number_of_integer_solutions_l133_133153


namespace value_of_m_l133_133243

theorem value_of_m (m : ℤ) : (let poly := (x - m) * (x + 7) in coeff poly 0) = 14 → m = -2 :=
by
  -- Skipping the proof as per instructions
  sorry

end value_of_m_l133_133243


namespace xy_sum_l133_133830

theorem xy_sum (x y : ℝ) (h1 : x^3 - 6 * x^2 + 15 * x = 12) (h2 : y^3 - 6 * y^2 + 15 * y = 16) : x + y = 4 := 
sorry

end xy_sum_l133_133830


namespace circle_and_tangent_lines_l133_133465

-- Define the problem conditions
def passes_through (a b r : ℝ) : Prop :=
  (a - (-2))^2 + (b - 2)^2 = r^2 ∧
  (a - (-5))^2 + (b - 5)^2 = r^2

def lies_on_line (a b : ℝ) : Prop :=
  a + b + 3 = 0

-- Define the standard equation of the circle
def is_circle_eq (a b r : ℝ) : Prop := ∀ x y : ℝ, 
  (x - a)^2 + (y - b)^2 = r^2 ↔ (x + 5)^2 + (y - 2)^2 = 9

-- Define the tangent lines
def is_tangent_lines (x y k : ℝ) : Prop :=
  (k = (20 / 21) ∨ x = -2) → (20 * x - 21 * y + 229 = 0 ∨ x = -2)

-- The theorem statement in Lean 4
theorem circle_and_tangent_lines (a b r : ℝ) (x y k : ℝ) :
  passes_through a b r →
  lies_on_line a b →
  is_circle_eq a b r →
  is_tangent_lines x y k :=
by {
  sorry
}

end circle_and_tangent_lines_l133_133465


namespace relationship_between_PR_PS_BC_l133_133346

-- Define the geometric entities and relations
variables (A B C D P R S Q : Type) [rect : rectangle A B C D]

-- Assume the following conditions
variable (hP_on_AB : point_on_line P AB)
variable (hPR_perp_AC : perpendicular PR AC)
variable (hPS_perp_BD : perpendicular PS BD)
variable (hPQ_parallel_BD : parallel PQ BD)
variable (hPQ_perp_BC : perpendicular PQ BC)

-- Here we'll state the theorem that needs to be proven:
theorem relationship_between_PR_PS_BC :
  PR + PS = BC := sorry

end relationship_between_PR_PS_BC_l133_133346


namespace hamiltonian_path_exists_l133_133134

-- Define the cities as nodes
inductive City
| SaintPetersburg
| Tver
| Yaroslavl
| NizhnyNovgorod
| Moscow
| Kazan
open City

-- Define the edges as pairs of connected cities
def edges : List (City × City) :=
  [(SaintPetersburg, Tver),
   (Yaroslavl, NizhnyNovgorod),
   (Moscow, Kazan),
   (NizhnyNovgorod, Kazan),
   (Moscow, Tver),
   (Moscow, NizhnyNovgorod)]

-- Define the degree of each city
def degree (c : City) : Nat :=
  edges.count (λ e, e.1 = c || e.2 = c)

-- Define the Hamiltonian path
def HamiltonianPath (path : List City) : Prop :=
  path.length = 6 ∧
  (∀ e ∈ edges, (∃ p.1 = e.1 ∧ p.2 = e.2) ∨ (∀ p.1 = e.2 ∧ p.2 = e.1))

-- The main theorem to prove
theorem hamiltonian_path_exists :
  ∃ (start : City), (start = SaintPetersburg ∨ start = Yaroslavl) ∧ (∃ path, HamiltonianPath path ∧ path.head = start) :=
  sorry

end hamiltonian_path_exists_l133_133134


namespace fraction_of_cream_in_cupA_l133_133179

   noncomputable def cupA_initial : ℚ := 8 -- ounces of coffee in Cup A
   noncomputable def cupB_initial : ℚ := 6 -- ounces of cream in Cup B
   noncomputable def cupC_initial : ℚ := 4 -- ounces of sugar in Cup C

   theorem fraction_of_cream_in_cupA :
     let cupA_after_step1 := cupA_initial - (1/3 * cupA_initial),
         cupB_after_step1 := cupB_initial + (1/3 * cupA_initial),
         cupA_after_step2 := cupA_after_step1 + (1/2 * cupB_after_step1),
         cupA_after_step3 := cupA_after_step2 - (1/4 * cupA_after_step2),
         cupC_after_step3 := cupC_initial + (1/4 * cupA_after_step2),
         cupA_after_step4 := cupA_after_step3 + (1/3 * cupC_after_step3),
         cream_in_cupA := (1/2 * (1/3 * cupB_after_step1))
     in (cream_in_cupA / cupA_after_step4) = 3/10 := sorry
   
end fraction_of_cream_in_cupA_l133_133179


namespace bus_ride_time_l133_133989

def walking_time : ℕ := 15
def waiting_time : ℕ := 2 * walking_time
def train_ride_time : ℕ := 360
def total_trip_time : ℕ := 8 * 60

theorem bus_ride_time : 
  (total_trip_time - (walking_time + waiting_time + train_ride_time)) = 75 := by
  sorry

end bus_ride_time_l133_133989


namespace license_plate_count_is_correct_l133_133135

/-- Define the number of consonants in the English alphabet --/
def num_consonants : Nat := 20

/-- Define the number of possibilities for 'A' --/
def num_A : Nat := 1

/-- Define the number of even digits --/
def num_even_digits : Nat := 5

/-- Define the total number of valid four-character license plates --/
def total_license_plate_count : Nat :=
  num_consonants * num_A * num_consonants * num_even_digits

/-- Theorem stating that the total number of license plates is 2000 --/
theorem license_plate_count_is_correct : 
  total_license_plate_count = 2000 :=
  by
    -- The proof is omitted
    sorry

end license_plate_count_is_correct_l133_133135


namespace simplify_and_evaluate_l133_133596

variable (x : ℝ)

theorem simplify_and_evaluate (hx : x = 2) : 
    (let expr := (x^2 + x) / (x^2 - 2*x + 1) / ((2 / (x - 1)) - (1 / x)) in expr) = 4 :=
by
  -- Substitute x = 2 into the expression
  rw hx
  sorry

end simplify_and_evaluate_l133_133596


namespace rhombus_area_l133_133244

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 5) (h_d2 : d2 = 8) : 
  (1 / 2) * d1 * d2 = 20 :=
by
  sorry

end rhombus_area_l133_133244


namespace farmer_boxes_last_week_l133_133186

theorem farmer_boxes_last_week 
    (pomelos_last_week : ℕ)
    (boxes_this_week : ℕ)
    (dozens_this_week : ℕ)
    (dozens_to_pomelos : ℕ → ℕ := λd, 12 * d)
    (number_of_boxes_last_week : ℕ := 6)
    (number_of_pomelos_per_box_this_week : ℕ := dozens_to_pomelos dozens_this_week / boxes_this_week) :
    pomelos_last_week = 240 ∧ boxes_this_week = 20 ∧ dozens_this_week = 60 → 
    number_of_boxes_last_week = pomelos_last_week / number_of_pomelos_per_box_this_week :=
by
  sorry

end farmer_boxes_last_week_l133_133186


namespace triangle_cot_b_cot_c_l133_133172

theorem triangle_cot_b_cot_c (A B C E Q : Point)
  (h_triangle : is_triangle A B C)
  (h_median : is_median A E)
  (h_angle : ∠AEQ = 60) :
  |cot B - cot C| = 2 :=
sorry

end triangle_cot_b_cot_c_l133_133172


namespace find_m_for_area_of_triangles_l133_133060

noncomputable def ellipse_foci := (-real.sqrt 2, 0, real.sqrt 2, 0)

theorem find_m_for_area_of_triangles : 
  (∃ (m : ℝ), 
    let F1 := (-real.sqrt 2, 0) in
    let F2 := (real.sqrt 2, 0) in
    let A B : ℝ × ℝ in
    let ellipse : ∀ x y : ℝ, (x^2 / 3) + y^2 = 1 := sorry in
    let line := λ x y : ℝ, y = x + m in
    line_intersects_ellipse : ∀ A B : ℝ × ℝ, disjoint ((A, B) ∪ ellipse) := sorry in
    (area_of_triangle(F1, A, B) = 2 * area_of_triangle(F2, A, B)) → m = -real.sqrt 2 / 3) :=
sorry

end find_m_for_area_of_triangles_l133_133060


namespace minimum_value_correct_l133_133609

noncomputable def minimum_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_eq : x^2 + y^2 + z^2 = 1) : ℝ :=
  (z + 1)^2 / (2 * x * y * z)

theorem minimum_value_correct {x y z : ℝ}
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : x^2 + y^2 + z^2 = 1) :
  minimum_value x y z h_pos h_eq = 3 + 2 * Real.sqrt 2 :=
sorry

end minimum_value_correct_l133_133609


namespace probability_product_is_cube_l133_133771

theorem probability_product_is_cube (p q : ℕ) (hpq_coprime : Nat.gcd p q = 1) 
  (hpq_pos : p > 0 ∧ q > 0) 
  (hprob : (p : ℚ) / (q : ℚ) = 53 / 324) : p + q = 377 :=
begin
  sorry,
end

end probability_product_is_cube_l133_133771


namespace p_divisible_by_5_l133_133318

noncomputable theory

open Nat

def q_infinity_cubed (q : ℚ) : ℕ → ℚ := 
λ k => (-1)^k * (2*k+1) * q^(k*(k+1)/2)

def q_infinity_quart (q : ℚ) (n : ℕ) : Prop :=
  ∑ k in range (n + 1), ((-1)^k * (2*k+1) * q^(k*(k+1)/2)) = 0

def mod_equiv1 (q : ℚ) : Prop :=
  (1 - q^5)/(1 - q)^5 ≡ 1 [MOD 5]

theorem p_divisible_by_5 (n : ℕ) : 
  ((∀ n : ℕ, q_infinity_cubed q ∑ k in range (n+1), k) ∨ q_infinity_quart q n) ∧
  mod_equiv1 q ∧ 
  p (5 * n + 4) ∣ 5 := sorry

end p_divisible_by_5_l133_133318


namespace total_photos_newspaper_l133_133538

theorem total_photos_newspaper (pages1 pages2 photos_per_page1 photos_per_page2 : ℕ)
  (h1 : pages1 = 12) (h2 : photos_per_page1 = 2)
  (h3 : pages2 = 9) (h4 : photos_per_page2 = 3) :
  (pages1 * photos_per_page1) + (pages2 * photos_per_page2) = 51 :=
by
  sorry

end total_photos_newspaper_l133_133538


namespace line_segment_angle_subtend_eq_l133_133664

-- Definitions & Assumptions
variables {A B C D X : Type} [Parallelogram A B C D] (inside : Inside X A B C D)
hypothesis  (H : ∡ A X B + ∡ C X D = 180)

-- The Theorem Statement
theorem line_segment_angle_subtend_eq :
  ∡ O A X = ∡ O C X :=
by sorry

end line_segment_angle_subtend_eq_l133_133664


namespace rate_of_escalator_l133_133390

-- Definitions based on the conditions
def length_of_escalator : ℕ := 210
def walking_speed : ℕ := 2
def time_to_cover_length : ℕ := 15

-- Main theorem to prove the rate of the escalator
theorem rate_of_escalator : 
  ∃ (v : ℕ), (v + walking_speed) * time_to_cover_length = length_of_escalator ∧ v = 12 :=
by
  use 12
  split
  {
    rw [Nat.mul_comm, Nat.add_comm],
    have h1 : 2 + 12 = 14 := rfl,
    have h2 : 15 * 14 = 210 := rfl,
    rw [← h2, h1]
  }
  {
    rfl
  }

end rate_of_escalator_l133_133390


namespace range_of_positive_integers_in_list_l133_133731

theorem range_of_positive_integers_in_list :
  ∀ (d : List Int), d = List.range' (-4) 12 → 
  (d.filter (λ x => x > 0)).length ≥ 1 →
  (d.filter (λ x => x > 0)).maximum - (d.filter (λ x => x > 0)).minimum = 6 :=
by intros d h₁ h₂; sorry

end range_of_positive_integers_in_list_l133_133731


namespace find_m_value_l133_133053

-- Define the ellipse equation and its foci
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define the foci of the ellipse C
def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the line equation
def line (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the condition for the areas of triangles F1AB and F2AB
def triangle_area_condition (F1 F2 : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := sorry in -- Coordinates of intersection points need to be defined
  let B := sorry in -- Coordinates of intersection points need to be defined
  sorry /- The actual calculation of the areas would require more complex integration and geometry.
           Here we just outline the condition -/

-- The proof statement
theorem find_m_value :
  ∃ m : ℝ, 
    ∀ x y : ℝ,
      ellipse x y →
      line m x y →
      triangle_area_condition F1 F2 m →
      m = -Real.sqrt 2 / 3 :=
  sorry

end find_m_value_l133_133053


namespace arithmetic_sequence_sum_eq_40_l133_133326

theorem arithmetic_sequence_sum_eq_40 
  (x y : ℕ) 
  (h_seq : ∃ d : ℕ, [2, 6, 10, x, y, 26].succ_nth (fun n => 2 + n*d) d) : 
  x + y = 40 :=
sorry

end arithmetic_sequence_sum_eq_40_l133_133326


namespace inversion_circles_alignment_and_equal_radii_l133_133461

theorem inversion_circles_alignment_and_equal_radii 
  (k1 k2 k3 : Circle) : 
  ∃ (k : Circle), 
    (∀ (i j : ℕ), (i = 1 ∨ i = 2 ∨ i = 3) → 
                 (j = 1 ∨ j = 2 ∨ j = 3) → 
                 i ≠ j → 
                 (inversion_circle_image k1 k ≃ Circle) ∧ 
                 (inversion_circle_image k2 k ≃ Circle) ∧ 
                 (inversion_circle_image k3 k ≃ Circle) ∧
                 (are_centers_collinear (map_inv_circles k k1 k2 k3)) ∧
                 (radii_eq (map_inv_circle k k1) (map_inv_circle k k2)) := 
sorry

end inversion_circles_alignment_and_equal_radii_l133_133461


namespace smallest_degree_R_l133_133287

-- Definitions and assumptions based on the problem conditions
variables {R : Type} [CommRing R] [IsDomain R]

-- Given polynomial P
def P (x : R) : Polynomial R := sorry

-- Polynomial P has degree 10^5 and leading coefficient 1
axiom degree_P : degree (P (Polynomial.X : Polynomial R)) = 100000
axiom leading_coeff_P : leadingCoeff (P Polynomial.X) = 1

-- Define R(x) as given in the problem
def R (x : R) : Polynomial R := (P (Polynomial.X ^ 1000 + 1)) - (P (Polynomial.X) ^ 1000)

-- The theorem to be proved
theorem smallest_degree_R : degree (R Polynomial.X) = 99 * 10^6 :=
sorry

end smallest_degree_R_l133_133287


namespace find_number_of_women_in_first_group_l133_133755

variables (W : ℕ)

-- Conditions
def women_coloring_rate := 10
def total_cloth_colored_in_3_days := 180
def women_in_first_group := total_cloth_colored_in_3_days / 3

theorem find_number_of_women_in_first_group
  (h1 : 5 * women_coloring_rate * 4 = 200)
  (h2 : W * women_coloring_rate = women_in_first_group) :
  W = 6 :=
by
  sorry

end find_number_of_women_in_first_group_l133_133755


namespace liquid_mixture_ratio_l133_133127

theorem liquid_mixture_ratio (m1 m2 m3 : ℝ) (ρ1 ρ2 ρ3 : ℝ) (k : ℝ)
  (hρ1 : ρ1 = 6 * k) (hρ2 : ρ2 = 3 * k) (hρ3 : ρ3 = 2 * k)
  (h_condition : m1 ≥ 3.5 * m2)
  (h_arith_mean : (m1 + m2 + m3) / (m1 / ρ1 + m2 / ρ2 + m3 / ρ3) = (ρ1 + ρ2 + ρ3) / 3) :
    ∃ x y : ℝ, x ≤ 2/7 ∧ (4 * x + 15 * y = 7) := sorry

end liquid_mixture_ratio_l133_133127


namespace triangulation_graph_one_stroke_l133_133592

def is_triangulation_graph_drawable_in_one_stroke (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 3 * k

theorem triangulation_graph_one_stroke (n : ℕ) :
  (∃ G : SimpleGraph, G.isEulerian ∧ G.isTriangulationGraph n) ↔ is_triangulation_graph_drawable_in_one_stroke n := by
  sorry

end triangulation_graph_one_stroke_l133_133592


namespace find_original_shirt_price_l133_133564

noncomputable def original_shirt_price (S pants_orig_price jacket_orig_price total_paid : ℝ) :=
  let discounted_shirt := S * 0.5625
  let discounted_pants := pants_orig_price * 0.70
  let discounted_jacket := jacket_orig_price * 0.64
  let total_before_loyalty := discounted_shirt + discounted_pants + discounted_jacket
  let total_after_loyalty := total_before_loyalty * 0.90
  let total_after_tax := total_after_loyalty * 1.15
  total_after_tax = total_paid

theorem find_original_shirt_price : 
  original_shirt_price S 50 75 150 → S = 110.07 :=
by
  intro h
  sorry

end find_original_shirt_price_l133_133564


namespace largest_power_of_3_factorial_sum_l133_133451

theorem largest_power_of_3_factorial_sum :
  ∀ n : ℕ, (∀ (k ≤ n), k = 22) → n = 22 := sorry

end largest_power_of_3_factorial_sum_l133_133451


namespace gcf_of_all_three_digit_palindromes_l133_133717

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ is_palindrome n

def gcf_of_palindromes : ℕ :=
  101

theorem gcf_of_all_three_digit_palindromes : 
  ∀ n, is_three_digit_palindrome n → 101 ∣ n := by
    sorry

end gcf_of_all_three_digit_palindromes_l133_133717


namespace simplified_radical_example_l133_133775

def is_simplified_quadratic_radical (r : ℝ) : Prop :=
  ∀a b : ℤ, r = (a : ℝ) * real.sqrt (b : ℝ) → a = 1

theorem simplified_radical_example :
  is_simplified_quadratic_radical (real.sqrt 6) ∧
  ¬is_simplified_quadratic_radical (real.sqrt 12) ∧
  ¬is_simplified_quadratic_radical (real.sqrt 20) ∧
  ¬is_simplified_quadratic_radical (real.sqrt 32) :=
by
  sorry

end simplified_radical_example_l133_133775


namespace find_m_value_l133_133052

-- Define the ellipse equation and its foci
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define the foci of the ellipse C
def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the line equation
def line (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the condition for the areas of triangles F1AB and F2AB
def triangle_area_condition (F1 F2 : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := sorry in -- Coordinates of intersection points need to be defined
  let B := sorry in -- Coordinates of intersection points need to be defined
  sorry /- The actual calculation of the areas would require more complex integration and geometry.
           Here we just outline the condition -/

-- The proof statement
theorem find_m_value :
  ∃ m : ℝ, 
    ∀ x y : ℝ,
      ellipse x y →
      line m x y →
      triangle_area_condition F1 F2 m →
      m = -Real.sqrt 2 / 3 :=
  sorry

end find_m_value_l133_133052


namespace equation_of_line_l_l133_133746

theorem equation_of_line_l :
  (∃ l : ℝ → ℝ → Prop, 
     (∀ x y, l x y ↔ (x - y + 3) = 0)
     ∧ (∀ x y, l x y → x^2 + (y - 3)^2 = 4)
     ∧ (∀ x y, l x y → x + y + 1 = 0)) :=
sorry

end equation_of_line_l_l133_133746


namespace last_two_digits_of_9_pow_2008_l133_133633

theorem last_two_digits_of_9_pow_2008 : (9 ^ 2008) % 100 = 21 := 
by
  sorry

end last_two_digits_of_9_pow_2008_l133_133633


namespace Cade_remaining_marbles_l133_133794

def initial_marbles := 87
def given_marbles := 8
def remaining_marbles := initial_marbles - given_marbles

theorem Cade_remaining_marbles : remaining_marbles = 79 := by
  sorry

end Cade_remaining_marbles_l133_133794


namespace max_projection_area_is_1_l133_133661

theorem max_projection_area_is_1 (A B C D : Point)
  (h₁ : is_isosceles_right_triangle A B C 2)
  (h₂ : is_isosceles_right_triangle A C D 2)
  (h₃ : dihedral_angle_eq A B C D (degrees 60))
  (h₄ : rotates_around_edge A C) : 
  max_projection_area A B C D A C = 1 :=
sorry

end max_projection_area_is_1_l133_133661


namespace triangle_isosceles_extended_altitudes_l133_133965

theorem triangle_isosceles_extended_altitudes
  (ABC : Triangle)
  (k : ℝ)
  (hpos : k > 0)
  (hAA' : AA' = k * BC)
  (hBB' : BB' = k * AC)
  (hCC' : CC' = k * AB) :
  k = 1 → (is_isosceles (A'B' C') ∧ A'B' = B'C') := 
sorry

end triangle_isosceles_extended_altitudes_l133_133965


namespace find_exponent_of_power_function_l133_133833

theorem find_exponent_of_power_function (a : ℝ) :
  (∀ x : ℝ, (differentiable (λ x, x^a)) ∧ ((∂ (λ x, x ^ a) / ∂ x) 1 = -4)) → a = -4 :=
sorry

end find_exponent_of_power_function_l133_133833


namespace choir_member_count_l133_133284

theorem choir_member_count (n : ℕ) : 
  (n ≡ 4 [MOD 7]) ∧ 
  (n ≡ 8 [MOD 6]) ∧ 
  (50 ≤ n ∧ n ≤ 200) 
  ↔ 
  (n = 60 ∨ n = 102 ∨ n = 144 ∨ n = 186) := 
by 
  sorry

end choir_member_count_l133_133284


namespace find_m_for_area_of_triangles_l133_133056

noncomputable def ellipse_foci := (-real.sqrt 2, 0, real.sqrt 2, 0)

theorem find_m_for_area_of_triangles : 
  (∃ (m : ℝ), 
    let F1 := (-real.sqrt 2, 0) in
    let F2 := (real.sqrt 2, 0) in
    let A B : ℝ × ℝ in
    let ellipse : ∀ x y : ℝ, (x^2 / 3) + y^2 = 1 := sorry in
    let line := λ x y : ℝ, y = x + m in
    line_intersects_ellipse : ∀ A B : ℝ × ℝ, disjoint ((A, B) ∪ ellipse) := sorry in
    (area_of_triangle(F1, A, B) = 2 * area_of_triangle(F2, A, B)) → m = -real.sqrt 2 / 3) :=
sorry

end find_m_for_area_of_triangles_l133_133056


namespace area_triangle_FQH_l133_133943

open Set

structure Point where
  x : ℝ
  y : ℝ

def Rectangle (A B C D : Point) : Prop :=
  A.x = B.x ∧ C.x = D.x ∧ A.y = D.y ∧ B.y = C.y

def IsMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def AreaTrapezoid (A B C D : Point) : ℝ :=
  0.5 * (B.x - A.x + D.x - C.x) * (A.y - C.y)

def AreaTriangle (A B C : Point) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

variables (E P R H F Q G : Point)

-- Conditions
axiom h1 : Rectangle E F G H
axiom h2 : E.y - P.y = 8
axiom h3 : R.y - H.y = 8
axiom h4 : F.x - E.x = 16
axiom h5 : AreaTrapezoid P R H G = 160

-- Target to prove
theorem area_triangle_FQH : AreaTriangle F Q H = 80 :=
sorry

end area_triangle_FQH_l133_133943


namespace domain_f_m_eq_7_range_m_if_fx_ge2_is_R_l133_133109

section

variable {x : ℝ}

def f (x : ℝ) (m : ℝ) : ℝ := log 2 ((|x + 1| + |x - 2|) - m)

-- Part 1: Prove the domain of f when m = 7 is (-∞, -3) ∪ (4, ∞)
theorem domain_f_m_eq_7 : 
  (∀ x : ℝ, f x 7 ∈ (Mem   (-∞ : Set ℝ) -3) ∪ (Mem (4 : Set ℝ) ∞)) := sorry

-- Part 2: Prove that if the solution set of f(x) ≥ 2 is ℝ, then the range of m is (-∞, -1]
theorem range_m_if_fx_ge2_is_R : 
  (∀ x : ℝ, (f x m) ≥ 2 ↔ x ∈ (Set.Univ : Set ℝ)) → (m ≤ -1)
:= sorry

end

end domain_f_m_eq_7_range_m_if_fx_ge2_is_R_l133_133109


namespace trapezium_second_side_length_l133_133823

-- Define the problem in Lean
variables (a h A b : ℝ)

-- Define the conditions
def conditions : Prop :=
  a = 20 ∧ h = 25 ∧ A = 475

-- Prove the length of the second parallel side
theorem trapezium_second_side_length (h_cond : conditions a h A) : b = 18 :=
by
  sorry

end trapezium_second_side_length_l133_133823


namespace exists_sum_divisible_by_11_l133_133570

open Set

theorem exists_sum_divisible_by_11 :
  ∃ (B : Set ℕ), (B ⊆ {n | 1 ≤ n ∧ n ≤ 100}) ∧ (B.card = 53) ∧ ∃ (x y ∈ B), x ≠ y ∧ 11 ∣ (x + y) :=
begin
  sorry
end

end exists_sum_divisible_by_11_l133_133570


namespace cleaning_time_together_l133_133736

theorem cleaning_time_together (t : ℝ) (h_t : 3 = t / 3) (h_john_time : 6 = 6) : 
  (5 / (1 / 6 + 1 / 9)) = 3.6 :=
by
  sorry

end cleaning_time_together_l133_133736


namespace faces_of_tetrahedron_cannot_change_places_l133_133378

theorem faces_of_tetrahedron_cannot_change_places (T : Type) [regular_tetrahedron T]
  (is_rolled : ∀ (p q : T), edge_flip p q) 
  (returns_to_original_position : ∃ (flips : set (T → T)), ∀ (t : T), t ∈ flips -> t = t_original) :
  ∀ (faces : set (face T)), ¬ switch_places faces (∑ edge_flip faces) returns_to_original_position :=
begin
  sorry
end

end faces_of_tetrahedron_cannot_change_places_l133_133378


namespace yi_jianlian_shots_l133_133404

theorem yi_jianlian_shots (x y : ℕ) 
  (h1 : x + y = 16 - 3) 
  (h2 : 2 * x + y = 28 - 3 * 3) : 
  x = 6 ∧ y = 7 := 
by 
  sorry

end yi_jianlian_shots_l133_133404


namespace extremum_point_of_quadratic_l133_133629

theorem extremum_point_of_quadratic (x : ℝ) : (∀ x : ℝ, (x^2 + 1)) → 0 = 0 :=
by
  sorry

end extremum_point_of_quadratic_l133_133629


namespace find_expression_for_f_l133_133106

-- Define the given function using the condition
def given_function (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f (2*x + 1) = x^2 - 2*x

-- Define the target function
def target_function (f : ℝ → ℝ) : Prop :=
  f = (λ x : ℝ, x^2 / 4 - 3*x / 2 + 5 / 4)

-- State the theorem
theorem find_expression_for_f :
  ∃ (f : ℝ → ℝ), given_function f ∧ target_function f :=
by
  existsi (λ x : ℝ, x^2 / 4 - 3*x / 2 + 5 / 4)
  simp [given_function, target_function]
  sorry

end find_expression_for_f_l133_133106


namespace mass_proportion_l133_133123

namespace DensityMixture

variables (k m_1 m_2 m_3 : ℝ)
def rho_1 := 6 * k
def rho_2 := 3 * k
def rho_3 := 2 * k
def arithmetic_mean := (rho_1 k + rho_2 k + rho_3 k) / 3
def density_mixture := (m_1 + m_2 + m_3) / 
    (m_1 / rho_1 k + m_2 / rho_2 k + m_3 / rho_3 k)
def mass_ratio_condition := m_1 / m_2 ≥ 3.5

theorem mass_proportion 
  (k_pos : 0 < k)
  (mass_cond : mass_ratio_condition k m_1 m_2) :
  ∃ (x y : ℝ), (4 * x + 15 * y = 7) ∧ (density_mixture k m_1 m_2 m_3 = arithmetic_mean k) ∧ mass_cond := 
sorry

end DensityMixture

end mass_proportion_l133_133123


namespace gcf_of_all_three_digit_palindromes_l133_133713

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ is_palindrome n

def gcf_of_palindromes : ℕ :=
  101

theorem gcf_of_all_three_digit_palindromes : 
  ∀ n, is_three_digit_palindrome n → 101 ∣ n := by
    sorry

end gcf_of_all_three_digit_palindromes_l133_133713


namespace part1_part2_part3_l133_133108

noncomputable def f (a x : ℝ) := (x + 1 - a) / (a - x)

-- The first part to prove f(x) + f(2a - x) + 2 = 0
theorem part1 (a x : ℝ) (h1 : x ≠ a) : f(a, x) + f(a, 2 * a - x) + 2 = 0 :=
sorry

-- The second part to calculate the center of symmetry
theorem part2 (a b : ℝ) (h1 : f(a, 3) + f(a, 6 - 3) - 2 * b = 0) : a + b = -4 / 7 :=
sorry

-- The third part to find the range of f(x) in the specified domain
theorem part3 (a x : ℝ) (h1 : x ∈ set.Icc (a + 1 / 2) (a + 1)) : f(a, x) ∈ set.Icc (-3) (-2) :=
sorry

end part1_part2_part3_l133_133108


namespace polynomial_satisfies_condition_l133_133582

/--
Let f be a real-valued function such that for any real numbers x and y,
the inequality f(x*y) + f(y - x) ≥ f(y + x) holds.

1. Provide a non-constant polynomial f that satisfies this condition.
2. Prove that for any real number x, f(x) ≥ 0.
-/
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f(x * y) + f(y - x) ≥ f(y + x)

theorem polynomial_satisfies_condition (f : ℝ → ℝ) (hf : satisfies_condition f) :
  (∃ (a b c : ℝ), a ≠ 0 ∧ f = λ x, a * x^2 + b * x + c) ∧ (∀ x : ℝ, f x ≥ 0) :=
sorry

end polynomial_satisfies_condition_l133_133582


namespace distribute_tourists_l133_133158

-- Define the number of ways k tourists can distribute among n cinemas
def num_ways (n k : ℕ) : ℕ := n^k

-- Theorem stating the number of distribution ways
theorem distribute_tourists (n k : ℕ) : num_ways n k = n^k :=
by sorry

end distribute_tourists_l133_133158


namespace clothing_store_gross_profit_l133_133360

theorem clothing_store_gross_profit :
  ∃ S : ℝ, S = 81 + 0.25 * S ∧
  ∃ new_price : ℝ,
    new_price = S - 0.20 * S ∧
    ∃ profit : ℝ,
      profit = new_price - 81 ∧
      profit = 5.40 :=
by
  sorry

end clothing_store_gross_profit_l133_133360


namespace solve_system_l133_133230

theorem solve_system (a b c x y z : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (eq1 : a * y + b * x = c)
  (eq2 : c * x + a * z = b)
  (eq3 : b * z + c * y = a) :
  x = (b^2 + c^2 - a^2) / (2 * b * c) ∧ 
  y = (a^2 + c^2 - b^2) / (2 * a * c) ∧ 
  z = (a^2 + b^2 - c^2) / (2 * a * b) :=
sorry

end solve_system_l133_133230


namespace power_function_evaluation_l133_133279

theorem power_function_evaluation (n : ℝ) (f : ℝ → ℝ) (h1 : ∃ n, ∀ x, f x = x^n)
  (h2 : f 2 = real.sqrt 2) : f 9 = 3 :=
sorry

end power_function_evaluation_l133_133279


namespace debts_equal_in_25_days_l133_133418

-- Define the initial debts and the interest rates
def Darren_initial_debt : ℝ := 200
def Darren_interest_rate : ℝ := 0.08
def Fergie_initial_debt : ℝ := 300
def Fergie_interest_rate : ℝ := 0.04

-- Define the debts as a function of days passed t
def Darren_debt (t : ℝ) : ℝ := Darren_initial_debt * (1 + Darren_interest_rate * t)
def Fergie_debt (t : ℝ) : ℝ := Fergie_initial_debt * (1 + Fergie_interest_rate * t)

-- Prove that Darren and Fergie will owe the same amount in 25 days
theorem debts_equal_in_25_days : ∃ t, Darren_debt t = Fergie_debt t ∧ t = 25 := by
  sorry

end debts_equal_in_25_days_l133_133418


namespace maximizing_p_f_expected_total_cost_inspection_decision_l133_133357

open Classical

noncomputable def f (p : ℝ) : ℝ := 
  let C10_2 := (10.choose 2 : ℝ)
  C10_2 * p^2 * (1 - p)^8

noncomputable def f_prime (p : ℝ) : ℝ :=
  let C10_2 := (10.choose 2 : ℝ)
  C10_2 * (2 * p * (1 - p)^8 - 8 * p^2 * (1 - p)^7)

theorem maximizing_p_f (p : ℝ) (hp : 0 < p ∧ p < 1):
  ∃ p0, p0 = 0.2 ∧ f_prime p0 = 0 :=
sorry

noncomputable def E_X (a : ℕ) : ℕ :=
  15 + 14 * a

theorem expected_total_cost (a : ℕ) : 
  E_X a = 15 + 14 * a :=
sorry

theorem inspection_decision (a : ℕ) (h_a : a ≥ 8) :
  ∀ cost : ℕ, cost > (15 + 14 * a) → cost = 120 :=
sorry

end maximizing_p_f_expected_total_cost_inspection_decision_l133_133357


namespace mathematicians_correctness_l133_133273

theorem mathematicians_correctness :
  ∃ (scenario1_w1 s_w1 : ℕ) (scenario1_w2 s_w2 : ℕ) (scenario2_w1 s2_w1 : ℕ) (scenario2_w2 s2_w2 : ℕ),
    scenario1_w1 = 4 ∧ s_w1 = 7 ∧ scenario1_w2 = 3 ∧ s_w2 = 5 ∧
    scenario2_w1 = 8 ∧ s2_w1 = 14 ∧ scenario2_w2 = 3 ∧ s2_w2 = 5 ∧
    let total_white1 := scenario1_w1 + scenario1_w2,
        total_choco1 := s_w1 + s_w2,
        prob1 := (total_white1 : ℚ) / total_choco1,
        total_white2 := scenario2_w1 + scenario2_w2,
        total_choco2 := s2_w1 + s2_w2,
        prob2 := (total_white2 : ℚ) / total_choco2,
        prob_box1 := 4 / 7,
        prob_box2 := 3 / 5 in
    (prob1 = 7 / 12 ∧ prob2 = 11 / 19 ∧
    (19 / 35 < prob_box1 ∧ prob_box1 < prob_box2) ∧
    (prob_box1 ≠ 19 / 35 ∧ prob_box1 ≠ 3 / 5)) :=
sorry

end mathematicians_correctness_l133_133273


namespace find_x0_l133_133016

def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 1 + Real.log x

theorem find_x0 (x0 : ℝ) (h : f' x0 = 2) : x0 = Real.exp 1 := by
  sorry

end find_x0_l133_133016


namespace range_of_function_simplify_expression_l133_133747

-- Problem 1
theorem range_of_function:
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → 
  (∃ y, y = 2 * x + 4 * sqrt (2 - x) ∧ y ∈ set.Icc 4 6) :=
sorry

-- Problem 2
theorem simplify_expression:
  ∀ (θ : ℝ), θ = 40 * real.pi / 180 →
  (sqrt (1 - 2 * sin θ * cos θ) / (cos θ - sqrt (1 - cos θ ^ 2)) = 1) :=
sorry

end range_of_function_simplify_expression_l133_133747


namespace star_value_example_l133_133857

def star (a b c : ℤ) : ℤ := (a + b + c) ^ 2

theorem star_value_example : star 3 (-5) 2 = 0 :=
by
  sorry

end star_value_example_l133_133857


namespace midpoint_product_l133_133192

theorem midpoint_product (x y : ℝ) (h1 : (4 : ℝ) = (x + 10) / 2) (h2 : (-2 : ℝ) = (-6 + y) / 2) : x * y = -4 := by
  sorry

end midpoint_product_l133_133192


namespace find_m_l133_133064

noncomputable def ellipse : set (ℝ × ℝ) := { p | (p.1)^2 / 3 + (p.2)^2 = 1 }
def f1 : ℝ × ℝ := (- real.sqrt 2, 0)
def f2 : ℝ × ℝ := (real.sqrt 2, 0)

def line (m : ℝ) : set (ℝ × ℝ) := { p | p.2 = p.1 + m }

theorem find_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line m ∧ B ∈ line m ∧
   2 * (abs ((f1.1 * (A.2 - B.2) + A.1 * (B.2 - f1.2) + B.1 * (f1.2 - A.2)) / 2))
 = abs ((f2.1 * (A.2 - B.2) + A.1 * (B.2 - f2.2) + B.1 * (f2.2 - A.2)) / 2)) →
  m = - real.sqrt 2 / 3 :=
sorry

end find_m_l133_133064


namespace max_value_when_a_is_1_value_of_a_when_max_is_2_l133_133492

noncomputable def f (a x : ℝ) : ℝ := -x^2 + 2 * a * x + 1 - a

theorem max_value_when_a_is_1
  (x : ℝ) (a : ℝ)
  (h : a = 1) :
  ∃ x, f a x = 1 := 
by {
  let fa := f a x,
  rw h at fa,
  use 1,
  exact sorry
}

theorem value_of_a_when_max_is_2
  (a : ℝ)
  (H : ∀ x ∈ set.Icc (0 : ℝ) (1 : ℝ), f a x ≤ 2) :
  a = -1 ∨ a = 2 :=
sorry

end max_value_when_a_is_1_value_of_a_when_max_is_2_l133_133492


namespace parallel_lines_unique_plane_l133_133220

theorem parallel_lines_unique_plane (a b : Line) (h_par : ∀ (P : Point) (Q : Point), P ∈ a → Q ∈ b → (a = b) → false) :
  ∃! (α : Plane), ∀ (P : Point), (P ∈ a) → (P ∈ α) ∧ ∀ (Q : Point), (Q ∈ b) → (Q ∈ α) :=
begin
  sorry
end

end parallel_lines_unique_plane_l133_133220


namespace students_taking_both_music_and_art_l133_133753

theorem students_taking_both_music_and_art (total_students : ℕ) (students_music : ℕ) (students_art : ℕ) (students_neither : ℕ) : 
  total_students = 500 → 
  students_music = 20 → 
  students_art = 20 → 
  students_neither = 470 → 
  (students_music + students_art - (total_students - students_neither) = 10) :=
begin
  intros h_total h_music h_art h_neither,
  calc
    students_music + students_art - (total_students - students_neither)
      = 20 + 20 - (500 - 470) : by rw [h_total, h_music, h_art, h_neither]
  ... = 20 + 20 - 30        : by rfl
  ... = 40 - 30             : by rfl
  ... = 10                  : by rfl,
end

end students_taking_both_music_and_art_l133_133753


namespace no_years_between_2000_and_3000_l133_133792

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def two_digit_prime_palindrome : ℕ := 11

theorem no_years_between_2000_and_3000 :
  (∀ y : ℕ, (2000 ≤ y ∧ y < 3000) ∧ is_palindrome y ∧ 
            (∃ p1 p2 : ℕ, is_prime p1 ∧ p1 < 100 ∧ is_palindrome p1 ∧ 
                          is_prime p2 ∧ 1000 ≤ p2 ∧ p2 < 10000 ∧ is_palindrome p2 ∧ 
                          y = p1 * p2) → false) :=
begin
  sorry
end

end no_years_between_2000_and_3000_l133_133792


namespace smallest_possible_value_of_M_l133_133655

theorem smallest_possible_value_of_M :
  ∃ (N M : ℕ), N > 0 ∧ M > 0 ∧ 
               ∃ (r_6 r_36 r_216 r_M : ℕ), 
               r_6 < 6 ∧ 
               r_6 < r_36 ∧ r_36 < 36 ∧ 
               r_36 < r_216 ∧ r_216 < 216 ∧ 
               r_216 < r_M ∧ 
               r_36 = (r_6 * r) ∧ 
               r_216 = (r_6 * r^2) ∧ 
               r_M = (r_6 * r^3) ∧ 
               Nat.mod N 6 = r_6 ∧ 
               Nat.mod N 36 = r_36 ∧ 
               Nat.mod N 216 = r_216 ∧ 
               Nat.mod N M = r_M ∧ 
               M = 2001 :=
sorry

end smallest_possible_value_of_M_l133_133655


namespace not_square_l133_133584

open Int

theorem not_square (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬ ∃ k : ℤ, (a^2 : ℤ) + ⌈(4 * a^2 : ℤ) / b⌉ = k^2 :=
by
  sorry

end not_square_l133_133584


namespace sqrt_of_product_eq_540_l133_133411

theorem sqrt_of_product_eq_540 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := 
by 
  sorry 

end sqrt_of_product_eq_540_l133_133411


namespace max_vector_sum_eq_3_l133_133164

noncomputable def A : ℝ × ℝ := (Real.sqrt 3, 1)
def unit_circle (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 1
def max_vector_sum (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 + B.1) ^ 2 + (A.2 + B.2) ^ 2)

theorem max_vector_sum_eq_3 (B : ℝ × ℝ) (hB : unit_circle B) :
  max_vector_sum A B ≤ 3 :=
sorry

end max_vector_sum_eq_3_l133_133164


namespace divide_large_triangles_l133_133917

def large_triangle (t : Triangle) : Prop :=
  t.a > 1 ∧ t.b > 1 ∧ t.c > 1

def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

noncomputable def side_length (P Q : Point) : ℝ :=
  sorry

def example_triangle : Triangle :=
  sorry -- Define specific triangle later

theorem divide_large_triangles :
  (∀ T : Triangle, equilateral_triangle T.p1 T.p2 T.p3 → side_length T.p1 T.p2 = 5 →
  ∃ ts : list Triangle, ts.length ≥ 100 ∧ ∀ t ∈ ts, large_triangle t ∧ 
  (∀ t₁ t₂ ∈ ts, t₁ ≠ t₂ → ∃ v : (Vertex | Edge), share_comm v t₁ t₂)) ∧
  (∀ T : Triangle, equilateral_triangle T.p1 T.p2 T.p3 → side_length T.p1 T.p2 = 3 →
  ∃ ts : list Triangle, ts.length ≥ 100 ∧ ∀ t ∈ ts, large_triangle t ∧ 
  (∀ t₁ t₂ ∈ ts, t₁ ≠ t₂ → ∃ v : (Vertex | Edge), share_comm v t₁ t₂))) :=
sorry

end divide_large_triangles_l133_133917


namespace problem_1_problem_2_l133_133478

-- Definitions for set A and B when a = 3 for (1)
def A : Set ℝ := { x | -x^2 + 5 * x - 4 > 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 ≤ 0 }

-- Theorem for (1)
theorem problem_1 : A ∪ (Bᶜ) = Set.univ := sorry

-- Function to describe B based on a for (2)
def B_a (a : ℝ) : Set ℝ := { x | x^2 - (a + 2) * x + 2 * a ≤ 0 }
def A_set : Set ℝ := { x | -x^2 + 5 * x - 4 > 0 }

-- Theorem for (2)
theorem problem_2 (a : ℝ) : (1 < a ∧ a < 4) → (A_set ∩ B_a a ≠ ∅ ∧ B_a a ⊆ A_set ∧ B_a a ≠ A_set) := sorry

end problem_1_problem_2_l133_133478


namespace reflect_parabola_x_axis_l133_133549

theorem reflect_parabola_x_axis (x : ℝ) (a b c : ℝ) :
  (∀ y : ℝ, y = x^2 + x - 2 → -y = x^2 + x - 2) →
  (∀ y : ℝ, -y = x^2 + x - 2 → y = -x^2 - x + 2) :=
by
  intros h₁ h₂
  intro y
  sorry

end reflect_parabola_x_axis_l133_133549


namespace Jenna_reading_goal_l133_133177

def num_days_not_reading : ℕ := 4
def pages_on_23rd : ℕ := 100
def pages_per_other_day : ℕ := 20
def total_days_in_september : ℕ := 30

theorem Jenna_reading_goal :
  let total_days_reading := total_days_in_september - num_days_not_reading in
  let other_days_reading := total_days_reading - 1 in
  let total_pages := other_days_reading * pages_per_other_day + pages_on_23rd in
  total_pages = 600 :=
by
  sorry

end Jenna_reading_goal_l133_133177


namespace find_m_l133_133034

variables (a b : ℝ) (x y m : ℝ) (A B F1 F2 : ℝ × ℝ)

-- Given conditions
def ellipse := (x^2 / 3) + y^2 = 1
def line := y = x + m
def foci_F1 := F1 = (-sqrt 2, 0)
def foci_F2 := F2 = (sqrt 2, 0)
def area_F1_AB := λ A B : ℝ × ℝ, 2 * |det(F1, A, B)| = |det(F2, A, B)|

-- The proof problem statement
theorem find_m : 
  ellipse ∧ 
  line ∧ 
  foci_F1 ∧ 
  foci_F2 ∧ 
  (area_F1_AB A B) →
  m = -sqrt(2) / 3 := sorry

end find_m_l133_133034


namespace volume_tetrahedrons_equal_l133_133166

open_locale classical

variables {A B C D A_i B_1 C_1 D_1 : Type*}
variables [is_tetrahedron A B C D] -- assuming a predicate is_tetrahedron exists
variables [is_point_on_face A_i B C D] -- assuming a predicate is_point_on_face exists
variables [parallel_plane_through A A_i B_1 C_1 D_1 B C D] -- assuming a predicate parallel_plane_through exists

theorem volume_tetrahedrons_equal :
  volume (tetrahedron A_1 B_1 C_1 D_1) = volume (tetrahedron A B C D) :=
sorry

end volume_tetrahedrons_equal_l133_133166


namespace area_of_square_l133_133808

def P : ℝ × ℝ := (1, 1)
def Q : ℝ × ℝ := (-4, 2)
def R : ℝ × ℝ := (-3, 7)
def S : ℝ × ℝ := (2, 6)

theorem area_of_square :
  let dist (A B : ℝ × ℝ) := real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  in dist P Q * dist P Q = 26 :=
sorry

end area_of_square_l133_133808


namespace greatest_common_factor_of_three_digit_palindromes_l133_133675

def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b

def gcf (a b : ℕ) : ℕ := 
  if a = 0 then b else gcf (b % a) a

theorem greatest_common_factor_of_three_digit_palindromes : 
  ∃ g, (∀ n, is_palindrome n → g ∣ n) ∧ (∀ d, (∀ n, is_palindrome n → d ∣ n) → d ∣ g) :=
by
  use 101
  sorry

end greatest_common_factor_of_three_digit_palindromes_l133_133675


namespace triangle_angle_zero_l133_133935

theorem triangle_angle_zero (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 4 * a * b) : 
  ∃ C : ℝ, C = 0 ∧ ∃ angle : ℝ, angle = degrees_to_radians(C) ∧ cos angle = 1 :=
by
  sorry

end triangle_angle_zero_l133_133935


namespace girls_population_after_four_years_l133_133541

theorem girls_population_after_four_years :
  let total_students := 300 / 0.40 in
  let initial_girls := 0.60 * total_students in
  let year1_girls := initial_girls - (0.05 * initial_girls) in
  let year2_girls := year1_girls + (0.03 * year1_girls) in
  let year3_girls := year2_girls - (0.04 * year2_girls) in
  let year4_girls := year3_girls + (0.02 * year3_girls) in
  year4_girls = 431 :=
by
  sorry

end girls_population_after_four_years_l133_133541


namespace intersection_point_of_lines_l133_133155

variable {x y a b : ℝ}

-- conditions from part a)
def system_satisfied := (2 * (-1) + 3 = b) ∧ (-1 - 3 = a)
def lines_intersection := (-2 * (-1) + b = 3) ∧ ((-1) - a = 3)

-- theorem statement
theorem intersection_point_of_lines (h1 : system_satisfied) : lines_intersection :=
by {
  -- proof would go here
  sorry
}

end intersection_point_of_lines_l133_133155


namespace shanghai_taipei_distance_l133_133098

noncomputable def earth_radius : ℝ := 6371

structure Location where
  longitude : ℝ
  latitude : ℝ

def Shanghai : Location := { longitude := 121, latitude := 31 }
def Taipei : Location := { longitude := 121, latitude := 25 }

def spherical_distance (A B : Location) (radius : ℝ) : ℝ :=
  let lat_diff := (A.latitude - B.latitude).abs
  (lat_diff / 360) * 2 * Real.pi * radius

theorem shanghai_taipei_distance :
  spherical_distance Shanghai Taipei earth_radius ≈ 667 := sorry

end shanghai_taipei_distance_l133_133098


namespace orthocenter_lies_on_XY_l133_133205

variables {A B C M N W X Y H : Type*}
variables [acute_triangle : is_acute ∠A ∠B ∠C]
variables [feet_M : is_foot M B A]
variables [feet_N : is_foot N C A]
variables [point_W : on_segment_W W B C]
variables [circle_omega1 : is_circumcircle_omega1 (circumcircle B W N)]
variables [circle_omega2 : is_circumcircle_omega2 (circumcircle C W M)]
variables [diametric_X : is_diametric_opposite X W circle_omega1]
variables [diametric_Y : is_diametric_opposite Y W circle_omega2]

theorem orthocenter_lies_on_XY (ABC : triangle A B C) (H : orthocenter A B C)
  (M : foot B A) (N : foot C A) (W : point on BC)
  (ω1 : circumcircle B W N) (ω2 : circumcircle C W M)
  (X : diametric_opposite W ω1) (Y : diametric_opposite W ω2) :
  lies_on H X Y :=
sorry

end orthocenter_lies_on_XY_l133_133205


namespace kathy_sunny_days_prob_l133_133653

noncomputable def prob_exactly_two_sunny_days : ℚ :=
  let num_ways := (nat.choose 5 2) in -- ways to choose 2 sunny days out of 5
  let prob_sunny := (1/4 : ℚ) in -- probability of one sunny day
  let prob_rainy := (3/4 : ℚ) in -- probability of one rainy day
  let prob_specific_seq := prob_sunny^2 * prob_rainy^3 in -- probability of a specific sequence of 2 sunny and 3 rainy days
  num_ways * prob_specific_seq -- total probability for exactly 2 sunny days

theorem kathy_sunny_days_prob : prob_exactly_two_sunny_days = 135 / 512 :=
  sorry

end kathy_sunny_days_prob_l133_133653


namespace base8_arithmetic_l133_133832

/-- Define base 8 digits -/
structure Digit8 :=
(val : Nat)
(property : val < 8)

def base8_to_nat (digits : List Digit8) : Nat :=
  digits.foldr (λ d acc, d.val + 8 * acc) 0

def nat_to_base8 (n : Nat) : List Digit8 :=
  if n = 0 then [⟨0, by decide⟩]
  else
    let rec digits_aux (n : Nat) : List Digit8 :=
      if n = 0 then [] else ⟨n % 8, by apply Nat.mod_lt; decide⟩ :: digits_aux (n / 8)
    digits_aux n

def add_base8 (a b : List Digit8) : List Digit8 :=
  nat_to_base8 (base8_to_nat a + base8_to_nat b)

def sub_base8 (a b : List Digit8) : List Digit8 :=
  nat_to_base8 (base8_to_nat a - base8_to_nat b)

theorem base8_arithmetic : sub_base8 (add_base8 [⟨5, by decide⟩, ⟨2, by decide⟩, ⟨3, by decide⟩, ⟨4, by decide⟩] [⟨2, by decide⟩, ⟨3, by decide⟩, ⟨5, by decide⟩])
                                [⟨1, by decide⟩, ⟨2, by decide⟩, ⟨7, by decide⟩]
                                = [⟨5, by decide⟩, ⟨3, by decide⟩, ⟨4, by decide⟩, ⟨4, by decide⟩] := by
  sorry

end base8_arithmetic_l133_133832


namespace steven_shirts_l133_133608

theorem steven_shirts : 
  (∀ (S A B : ℕ), S = 4 * A ∧ A = 6 * B ∧ B = 3 → S = 72) := 
by
  intro S A B
  intro h
  cases h with h1 h2
  cases h2 with hA hB
  rw [hB, hA]
  sorry

end steven_shirts_l133_133608


namespace find_m_for_area_of_triangles_l133_133058

noncomputable def ellipse_foci := (-real.sqrt 2, 0, real.sqrt 2, 0)

theorem find_m_for_area_of_triangles : 
  (∃ (m : ℝ), 
    let F1 := (-real.sqrt 2, 0) in
    let F2 := (real.sqrt 2, 0) in
    let A B : ℝ × ℝ in
    let ellipse : ∀ x y : ℝ, (x^2 / 3) + y^2 = 1 := sorry in
    let line := λ x y : ℝ, y = x + m in
    line_intersects_ellipse : ∀ A B : ℝ × ℝ, disjoint ((A, B) ∪ ellipse) := sorry in
    (area_of_triangle(F1, A, B) = 2 * area_of_triangle(F2, A, B)) → m = -real.sqrt 2 / 3) :=
sorry

end find_m_for_area_of_triangles_l133_133058


namespace number_of_non_subsets_l133_133986

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}
def unionAB := A ∪ B

theorem number_of_non_subsets (A B : Set ℕ) (unionAB = {1, 2, 3}) :
  ∃ P : Set ℕ, P ⊆ unionAB ∧ ¬ P ⊆_unionAB :=
sorry

end number_of_non_subsets_l133_133986


namespace find_inverse_sum_l133_133115

variable (a b x y : ℝ)

def line_eq := x - y + 1 = 0
def hyperbola_eq (a b : ℝ) := x^2 / a + y^2 / b = 1
def intersection_condition : a * b < 0 := sorry
def perpendicular_OP_OQ (x1 y1 x2 y2 : ℝ) : x1 * x2 + y1 * y2 = 0 := sorry

theorem find_inverse_sum (a b : ℝ) (h1 : line_eq x y) (h2 : hyperbola_eq a b) (h3 : intersection_condition) (h4 : perpendicular_OP_OQ x1 y1 x2 y2) :
  (1 / a + 1 / b) = 2 := sorry

end find_inverse_sum_l133_133115


namespace find_a_l133_133898

noncomputable def A := { x : ℝ | 1 ≤ x ∧ x ≤ 4 }

def M (a : ℝ) := { x : ℝ | (x - a) * (x - 2) ≤ 0 }

theorem find_a (a: ℝ) : M(a) ⊆ A → 1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end find_a_l133_133898


namespace triangle_area_l133_133371

theorem triangle_area (h : ℝ) (hypotenuse : h = 12) (angle : ∃θ : ℝ, θ = 30 ∧ θ = 30) :
  ∃ (A : ℝ), A = 18 * Real.sqrt 3 :=
by
  sorry

end triangle_area_l133_133371


namespace find_m_l133_133037

variables (a b : ℝ) (x y m : ℝ) (A B F1 F2 : ℝ × ℝ)

-- Given conditions
def ellipse := (x^2 / 3) + y^2 = 1
def line := y = x + m
def foci_F1 := F1 = (-sqrt 2, 0)
def foci_F2 := F2 = (sqrt 2, 0)
def area_F1_AB := λ A B : ℝ × ℝ, 2 * |det(F1, A, B)| = |det(F2, A, B)|

-- The proof problem statement
theorem find_m : 
  ellipse ∧ 
  line ∧ 
  foci_F1 ∧ 
  foci_F2 ∧ 
  (area_F1_AB A B) →
  m = -sqrt(2) / 3 := sorry

end find_m_l133_133037


namespace RS_length_l133_133291

-- Define the vertices of the tetrahedron
variables (P Q R S : Type) (dist : P → P → ℝ)

-- Given lengths of edges
axiom edge1 : dist P Q = 35
axiom edge2 : dist P R = 12 
axiom edge3 : dist P S = 24 
axiom edge4 : dist Q R = 31 
axiom edge5 : dist Q S = 17 
axiom edge6 : dist R S = 6

-- Triangle inequalities
axiom tri1 : dist P Q + dist P R > dist Q R
axiom tri2 : dist P Q + dist P S > dist Q S
axiom tri3 : dist P R + dist P S > dist R S
axiom tri4 : dist Q R + dist Q S > dist R S
axiom tri5 : dist Q R + dist P R > dist P Q
axiom tri6 : dist Q S + dist P S > dist P Q

-- Theorem to prove that length of edge RS is 6
theorem RS_length : dist R S = 6 :=
by
  -- Use the given edges and triangle inequalities in the proof
  sorry

end RS_length_l133_133291


namespace find_m_l133_133068

noncomputable def ellipse : set (ℝ × ℝ) := { p | (p.1)^2 / 3 + (p.2)^2 = 1 }
def f1 : ℝ × ℝ := (- real.sqrt 2, 0)
def f2 : ℝ × ℝ := (real.sqrt 2, 0)

def line (m : ℝ) : set (ℝ × ℝ) := { p | p.2 = p.1 + m }

theorem find_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line m ∧ B ∈ line m ∧
   2 * (abs ((f1.1 * (A.2 - B.2) + A.1 * (B.2 - f1.2) + B.1 * (f1.2 - A.2)) / 2))
 = abs ((f2.1 * (A.2 - B.2) + A.1 * (B.2 - f2.2) + B.1 * (f2.2 - A.2)) / 2)) →
  m = - real.sqrt 2 / 3 :=
sorry

end find_m_l133_133068


namespace max_F_value_no_min_F_value_l133_133015

def f (x : ℝ) : ℝ := 3 - 2 * |x|
def g (x : ℝ) : ℝ := x^2 - 2 * x

def F (x : ℝ) : ℝ :=
  if f x ≥ g x then g x else f x

theorem max_F_value : ∃ x : ℝ, ∀ y : ℝ, F y ≤ F x ∧ F x = 7 - 2 * real.sqrt 7 :=
sorry

theorem no_min_F_value : ∀ M : ℝ, ∃ x : ℝ, F x < M :=
sorry

end max_F_value_no_min_F_value_l133_133015


namespace pirates_gold_coins_l133_133846

theorem pirates_gold_coins (S a b c d e : ℕ) (h1 : a = S / 3) (h2 : b = S / 4) (h3 : c = S / 5) (h4 : d = S / 6) (h5 : e = 90) :
  S = 1800 :=
by
  -- Definitions and assumptions would go here
  sorry

end pirates_gold_coins_l133_133846


namespace logical_inconsistency_in_dihedral_angle_def_l133_133001

-- Define the given incorrect definition
def incorrect_dihedral_angle_def : String :=
  "A dihedral angle is an angle formed by two half-planes originating from one straight line."

-- Define the correct definition
def correct_dihedral_angle_def : String :=
  "A dihedral angle is a spatial figure consisting of two half-planes that share a common edge."

-- Define the logical inconsistency
theorem logical_inconsistency_in_dihedral_angle_def :
  incorrect_dihedral_angle_def ≠ correct_dihedral_angle_def := by
  sorry

end logical_inconsistency_in_dihedral_angle_def_l133_133001


namespace election_votes_l133_133656

theorem election_votes (T V : ℕ) 
    (hT : 8 * T = 11 * 20000) 
    (h_total_votes : T = 2500 + V + 20000) :
    V = 5000 :=
by
    sorry

end election_votes_l133_133656


namespace parabola_equation_slope_constant_l133_133939

-- Definitions
def F : ℝ × ℝ := ⟨1, 0⟩
def l := {x : ℝ × ℝ | x.1 = -3}
def l' := {x : ℝ × ℝ | x.1 = -1}
def E := {P : ℝ × ℝ | (P.2)^2 = 4 * P.1}
def A (t : ℝ) := (1, t)
def is_on_E (P : ℝ × ℝ) := (P.2)^2 = 4 * P.1
def slope (P Q : ℝ × ℝ) := (Q.2 - P.2) / (Q.1 - P.1)
def is_neg_recip (m1 m2 : ℝ) := m1 * m2 = -1

-- Theorems
theorem parabola_equation : ∀ P : ℝ × ℝ, P ∈ E ↔ is_on_E P := by sorry

theorem slope_constant (t : ℝ) (ht : t > 0) (C D : ℝ × ℝ) (hC : C ≠ A t) (hD : D ≠ A t)
    (hC_E : C ∈ E) (hD_E : D ∈ E)
    (h_neg_recip : is_neg_recip (slope (A t) C) (slope (A t) D)) :
    slope C D = -1 := by sorry

end parabola_equation_slope_constant_l133_133939


namespace find_m_l133_133033

variables (a b : ℝ) (x y m : ℝ) (A B F1 F2 : ℝ × ℝ)

-- Given conditions
def ellipse := (x^2 / 3) + y^2 = 1
def line := y = x + m
def foci_F1 := F1 = (-sqrt 2, 0)
def foci_F2 := F2 = (sqrt 2, 0)
def area_F1_AB := λ A B : ℝ × ℝ, 2 * |det(F1, A, B)| = |det(F2, A, B)|

-- The proof problem statement
theorem find_m : 
  ellipse ∧ 
  line ∧ 
  foci_F1 ∧ 
  foci_F2 ∧ 
  (area_F1_AB A B) →
  m = -sqrt(2) / 3 := sorry

end find_m_l133_133033


namespace cn_less_than_bn_l133_133575

/-- 
Define the sum of the digits of a natural number k
-/
def S (k : ℕ) : ℕ := sorry -- We define S to be the sum of the digits function

/-- 
Define the function f as n - S(n)
-/
def f (n : ℕ) : ℕ := n - S(n)

/-- 
The k-th iteration of the function f
-/
def f_iter : ℕ → ℕ → ℕ
| 0, n := n
| (k+1), n := f (f_iter k n)

theorem cn_less_than_bn (n k : ℕ) (d : ℕ) (b0 c0 : ℕ) 
  (hk : k = 10 ^ d) 
  (h_bound : 10 ^ d > 20 * d * (n + 1)) 
  (hb0 : b0 = 10 ^ k - 1) 
  (hc0 : c0 = 10 ^ k - 1) : 
  f_iter n c0 < f_iter n b0 := 
sorry
-- The proof part is skipped as per the problem statement.

end cn_less_than_bn_l133_133575


namespace students_remaining_at_end_l133_133742

theorem students_remaining_at_end (initial students_left students_arrived : ℕ)
  (h_initial : initial = 33)
  (h_students_left : students_left = 18)
  (h_students_arrived : students_arrived = 14) :
  initial - students_left + students_arrived = 29 :=
by
  rw [h_initial, h_students_left, h_students_arrived]
  sorry

end students_remaining_at_end_l133_133742


namespace trig_expression_equality_l133_133856

theorem trig_expression_equality (α : ℝ) (h : Real.tan α = 1 / 2) : (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -4 :=
by
  sorry

end trig_expression_equality_l133_133856


namespace part1_part2_i_part2_ii_l133_133112

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * Real.log x + 1 / Real.exp x

theorem part1 (k : ℝ) (h : ¬ MonotoneOn (f k) (Set.Icc 2 3)) :
  3 / Real.exp 3 < k ∧ k < 2 / Real.exp 2 :=
sorry

variables {x1 x2 : ℝ}
variable (k : ℝ)
variable (h0 : 0 < x1)
variable (h1 : x1 < x2)
variable (h2 : k = x1 / Real.exp x1 ∧ k = x2 / Real.exp x2)

theorem part2_i :
  e / Real.exp x2 - e / Real.exp x1 > -Real.log (x2 / x1) ∧ -Real.log (x2 / x1) > 1 - x2 / x1 :=
sorry

theorem part2_ii : |f k x1 - f k x2| < 1 :=
sorry

end part1_part2_i_part2_ii_l133_133112


namespace constant_polynomial_of_divides_l133_133571

-- Definitions for the conditions
variables {P : Polynomial ℤ} (a : Fin 2019 → ℕ)

-- The main theorem statement
theorem constant_polynomial_of_divides (h : ∀ n > 0, (P.eval n) ∣ (∑ i, (a i : ℤ)^n)) : P.is_constant :=
begin
  sorry
end

end constant_polynomial_of_divides_l133_133571


namespace volume_of_rotated_solid_l133_133647

theorem volume_of_rotated_solid (unit_cylinder_r1 h1 r2 h2 : ℝ) :
  unit_cylinder_r1 = 6 → h1 = 1 → r2 = 3 → h2 = 4 → 
  (π * unit_cylinder_r1^2 * h1 + π * r2^2 * h2) = 72 * π :=
by 
-- We place the arguments and sorry for skipping the proof
  sorry

end volume_of_rotated_solid_l133_133647


namespace jack_apples_final_count_l133_133559

theorem jack_apples_final_count :
  ∀ (initial_apples : ℕ) (percent_sold_to_jill percent_sold_to_june : ℚ)
    (teacher_apples bad_apples : ℕ),
  initial_apples = 150 →
  percent_sold_to_jill = 0.30 →
  percent_sold_to_june = 0.20 →
  teacher_apples = 1 →
  bad_apples = 5 →
  let apples_after_jill := initial_apples - (initial_apples * percent_sold_to_jill).to_nat in
  let apples_after_june := apples_after_jill - (apples_after_jill * percent_sold_to_june).to_nat in
  let final_apples := apples_after_june - teacher_apples - bad_apples in
  final_apples = 78 :=
by
  intros initial_apples percent_sold_to_jill percent_sold_to_june teacher_apples bad_apples
  intros h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  let apples_after_jill := 150 - (150 * 0.30).to_nat
  let apples_after_june := apples_after_jill - (apples_after_jill * 0.20).to_nat
  let final_apples := apples_after_june - 1 - 5
  show final_apples = 78, from
  by 
    rw [(150 * 0.30).to_nat, (apples_after_jill * 0.20).to_nat]
    rw [natural_num_literals]
    exact (show final_apples = 78, from rfl)
  done.

end jack_apples_final_count_l133_133559


namespace gcf_of_three_digit_palindromes_is_one_l133_133684

-- Define a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define the greatest common factor (gcd) function
def gcd_of_all_palindromes : ℕ :=
  (Finset.range 999).filter is_palindrome |>.list.foldr gcd 0

-- State the theorem
theorem gcf_of_three_digit_palindromes_is_one :
  gcd_of_all_palindromes = 1 :=
sorry

end gcf_of_three_digit_palindromes_is_one_l133_133684


namespace area_ratio_ellipse_l133_133081

noncomputable def ellipse := { x // (x.1^2 / 3 + x.2^2 = 1) }
def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))
def intersects (line : ℝ × ℝ) (C : ℝ × ℝ → Prop) := ∃ (x y : ℝ), (C (x, y)) ∧ (y = x + line.2)

theorem area_ratio_ellipse 
  (a b : ℝ) (h_ellipse : a^2 = 3 ∧ b^2 = 1) (m : ℝ) 
  (h_foci : foci a b = ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))) 
  (h_condition : ∀ (x : ℝ), intersects (x, m) (λ pt, pt.1^2 / 3 + pt.2^2 = 1) →
     |((x - real.sqrt (a^2 - b^2)) + m)| = 2|((x + real.sqrt (a^2 - b^2)) + m)|) :
  m = -real.sqrt(2) / 3 := sorry

end area_ratio_ellipse_l133_133081


namespace find_circle_center_l133_133892

-- Define the polar circle in Lean
def polar_circle (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- State the theorem to convert polar to Cartesian and find the center
theorem find_circle_center :
  ∀ ρ θ : ℝ, (polar_circle ρ θ) → ∃ x y : ℝ, 
    (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) ∧ ((x-1)^2 + y^2 = 1) :=
begin
  sorry
end

end find_circle_center_l133_133892


namespace number_of_isosceles_triangle_l133_133350

theorem number_of_isosceles_triangle (a : ℝ) (h₁ : 1 < a) :
  (a > real.sqrt 3 → ∃ t : ℕ, t = 3 ∧
    (∀ x y, (x, y) ∈ { (x, y) : ℝ × ℝ | (x^2 / a^2 + y^2 = 1) ∧ y = k*x + 1 ∧ y = -(1/k)*x + 1}) ∧
      exists v1 v2 v3, isosceles_right_triangle (0, 1) v1 v2 v3) ∧
  (a ≤ real.sqrt 3 → ∃ t : ℕ, t = 1 ∧
    (∀ x y, (x, y) ∈ { (x, y) : ℝ × ℝ | (x^2 / a^2 + y^2 = 1) ∧ y = k*x + 1 ∧ y = -(1/k)*x + 1}) ∧
      exists v1 v2 v3, isosceles_right_triangle (0, 1) v1 v2 v3)) :=
by sorry

end number_of_isosceles_triangle_l133_133350


namespace intersection_of_sets_l133_133091

def SetA : Set ℝ := {x | 0 < x ∧ x < 3}
def SetB : Set ℝ := {x | x > 2}
def SetC : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_of_sets :
  SetA ∩ SetB = SetC :=
by
  sorry

end intersection_of_sets_l133_133091


namespace area_under_curve_l133_133342

theorem area_under_curve :
  (∫ x in 0..2, x^2 * sqrt (4 - x^2)) = π :=
sorry

end area_under_curve_l133_133342


namespace axis_of_symmetry_equation_l133_133111

noncomputable def f (x : ℝ) : ℝ := sin x - cos x

noncomputable def g (x : ℝ) : ℝ := sqrt 2 * sin (x / 2 - 5 * π / 12)

theorem axis_of_symmetry_equation (k : ℤ) : 
  ∃ x : ℝ, g x = g (x + 2kπ) ↔ x = 2kπ + 11 * π / 6 :=
sorry

end axis_of_symmetry_equation_l133_133111


namespace find_a_values_l133_133495

open Real

theorem find_a_values (a : ℝ) (h : a ≥ 0) :
  let y := λ x : ℝ, x^3 - 3*x,
      interval := Icc a (a + 1),
      max_val := Sup (y '' interval),
      min_val := Inf (y '' interval) in
  max_val - min_val = 2 ↔ (a = 0 ∨ a = sqrt3 - 1) :=
by
  sorry

end find_a_values_l133_133495


namespace parallel_vectors_k_l133_133132

theorem parallel_vectors_k (k : ℝ) : (∃ (u v : ℝ), u • (k, 1) = v • (k + 1, -2)) → k = -1/3 :=
by
  sorry

end parallel_vectors_k_l133_133132


namespace complement_U_A_A_intersection_B_complement_U_A_intersection_B_complement_U_A_intersection_B_B_l133_133588

open Set

variable U : Set ℝ := @univ ℝ
variable A : Set ℝ := {x | -2 < x ∧ x < 3}
variable B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

theorem complement_U_A : compl A = {x : ℝ | x ≥ 3 ∨ x ≤ -2} := sorry

theorem A_intersection_B : A ∩ B = {x : ℝ | -2 < x ∧ x < 3} := sorry

theorem complement_U_A_intersection_B : compl (A ∩ B) = {x : ℝ | x ≥ 3 ∨ x ≤ -2} := sorry

theorem complement_U_A_intersection_B_B : compl A ∩ B = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ x = 3} := sorry

end complement_U_A_A_intersection_B_complement_U_A_intersection_B_complement_U_A_intersection_B_B_l133_133588


namespace max_value_t_min_value_y_l133_133886

-- 1. Prove that the maximum value of t given |2x+5| + |2x-1| - t ≥ 0 is s = 6.
theorem max_value_t (t : ℝ) (x : ℝ) :
  (abs (2*x + 5) + abs (2*x - 1) - t ≥ 0) → (t ≤ 6) :=
by sorry

-- 2. Given s = 6 and 4a + 5b = s, prove that the minimum value of y = 1/(a+2b) + 4/(3a+3b) is y = 3/2.
theorem min_value_y (a b : ℝ) (s : ℝ) :
  s = 6 → (4*a + 5*b = s) → (a > 0) → (b > 0) → 
  (1/(a + 2*b) + 4/(3*a + 3*b) ≥ 3/2) :=
by sorry

end max_value_t_min_value_y_l133_133886


namespace polygon_implies_convex_l133_133976

noncomputable def convex_polygon_exists (n : ℕ) (l : Fin n → ℝ) : Prop :=
  ∀ i : Fin n, 
    ∑ j in Finset.univ.filter (≠ i), l j > l i

theorem polygon_implies_convex (n : ℕ) (l : Fin n → ℝ) 
  (h : (∀ i : Fin n, l i > 0) ∧ convex_polygon_exists n l) : 
  ∃ P : Polygon, is_convex_polygon_with_sides P l :=
sorry

end polygon_implies_convex_l133_133976


namespace sum_ratio_15_l133_133900

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (n : ℕ)

-- The sum of the first n terms of the sequences
def sum_a (n : ℕ) := S n
def sum_b (n : ℕ) := T n

-- The ratio condition
def ratio_condition := ∀ n, a n * (n + 1) = b n * (3 * n + 21)

theorem sum_ratio_15
  (ha : sum_a 15 = 15 * a 8)
  (hb : sum_b 15 = 15 * b 8)
  (h_ratio : ratio_condition a b) :
  sum_a 15 / sum_b 15 = 5 :=
sorry

end sum_ratio_15_l133_133900


namespace mathematician_correctness_l133_133251

theorem mathematician_correctness (box1_white1 box1_total1 box2_white1 box2_total1 : ℕ)
                                  (prob1 prob2 : ℚ) :
  (box1_white1 = 4 ∧ box1_total1 = 7 ∧ box2_white1 = 3 ∧ box2_total1 = 5 ∧ prob1 = (4:ℚ) / 7 ∧ prob2 = (3:ℚ) / 5) →
  (box1_white2 = 8 ∧ box1_total2 = 14 ∧ box2_white2 = 3 ∧ box2_total2 = 5 ∧ prob1 = (8:ℚ) / 14 ∧ prob2 = (3:ℚ) / 5) →
  (prob1 < (7:ℚ) / 12 ∧ prob2 > (11:ℚ)/19) →
  ¬((19:ℚ)/35 > (4:ℚ)/7 ∧ (19:ℚ)/35 < (3:ℚ)/5) :=
by
  sorry

end mathematician_correctness_l133_133251


namespace value_of_sum_ratio_l133_133925

theorem value_of_sum_ratio (w x y: ℝ) (hx: w / x = 1 / 3) (hy: w / y = 2 / 3) : (x + y) / y = 3 :=
sorry

end value_of_sum_ratio_l133_133925


namespace find_m_l133_133069

noncomputable def ellipse : set (ℝ × ℝ) := { p | (p.1)^2 / 3 + (p.2)^2 = 1 }
def f1 : ℝ × ℝ := (- real.sqrt 2, 0)
def f2 : ℝ × ℝ := (real.sqrt 2, 0)

def line (m : ℝ) : set (ℝ × ℝ) := { p | p.2 = p.1 + m }

theorem find_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line m ∧ B ∈ line m ∧
   2 * (abs ((f1.1 * (A.2 - B.2) + A.1 * (B.2 - f1.2) + B.1 * (f1.2 - A.2)) / 2))
 = abs ((f2.1 * (A.2 - B.2) + A.1 * (B.2 - f2.2) + B.1 * (f2.2 - A.2)) / 2)) →
  m = - real.sqrt 2 / 3 :=
sorry

end find_m_l133_133069


namespace total_photos_newspaper_l133_133534

-- Define the conditions
def num_pages1 := 12
def photos_per_page1 := 2
def num_pages2 := 9
def photos_per_page2 := 3

-- Define the total number of photos
def total_photos := (num_pages1 * photos_per_page1) + (num_pages2 * photos_per_page2)

-- Prove that the total number of photos is 51
theorem total_photos_newspaper : total_photos = 51 := by
  -- We will fill the proof later
  sorry

end total_photos_newspaper_l133_133534


namespace fourth_root_closest_option_l133_133276

def N := 2001200120012001

theorem fourth_root_closest_option :
  Real.fourth_root N ≈ 6700 := 
sorry

end fourth_root_closest_option_l133_133276


namespace gcd_three_digit_palindromes_l133_133706

theorem gcd_three_digit_palindromes : 
  GCD (set.image (λ (p : ℕ × ℕ), 101 * p.1 + 10 * p.2) 
    ({a | a ≠ 0 ∧ a < 10} × {b | b < 10})) = 1 := 
by
  sorry

end gcd_three_digit_palindromes_l133_133706


namespace mathematician_correctness_l133_133254

theorem mathematician_correctness (box1_white1 box1_total1 box2_white1 box2_total1 : ℕ)
                                  (prob1 prob2 : ℚ) :
  (box1_white1 = 4 ∧ box1_total1 = 7 ∧ box2_white1 = 3 ∧ box2_total1 = 5 ∧ prob1 = (4:ℚ) / 7 ∧ prob2 = (3:ℚ) / 5) →
  (box1_white2 = 8 ∧ box1_total2 = 14 ∧ box2_white2 = 3 ∧ box2_total2 = 5 ∧ prob1 = (8:ℚ) / 14 ∧ prob2 = (3:ℚ) / 5) →
  (prob1 < (7:ℚ) / 12 ∧ prob2 > (11:ℚ)/19) →
  ¬((19:ℚ)/35 > (4:ℚ)/7 ∧ (19:ℚ)/35 < (3:ℚ)/5) :=
by
  sorry

end mathematician_correctness_l133_133254


namespace root_of_arithmetic_sequence_quadratic_l133_133644

theorem root_of_arithmetic_sequence_quadratic (p q r x k : ℝ) 
  (h_arith_seq : q = p - k ∧ r = p - 2 * k) 
  (h_order : p ≥ q ∧ q ≥ r ∧ r ≥ 0) 
  (h_quad_root : (p * x^2 + q * x + r = 0)) 
  (h_discriminant : (p - k)^2 - 4 * p * (p - 2 * k) = 0) 
  : x = 1 - sqrt 6 / 2 := 
sorry

end root_of_arithmetic_sequence_quadratic_l133_133644


namespace original_cost_of_car_l133_133223

-- Conditions
variables (C : ℝ)
variables (spent_on_repairs : ℝ := 8000)
variables (selling_price : ℝ := 68400)
variables (profit_percent : ℝ := 54.054054054054056)

-- Statement to be proved
theorem original_cost_of_car :
  C + spent_on_repairs = selling_price - (profit_percent / 100) * C :=
sorry

end original_cost_of_car_l133_133223


namespace closest_fraction_to_medals_won_l133_133394

theorem closest_fraction_to_medals_won (medals_won total_medals : ℕ) 
    (assuming_won : medals_won = 38)
    (assuming_total : total_medals = 150)
    (target_fraction := (medals_won : ℚ) / total_medals) 
    (choices := [1/3, 1/4, 1/5, 1/6, 1/7] : list ℚ) :
  ∃ (closest : ℚ), closest ∈ choices ∧ abs (target_fraction - closest) = minimum (abs (target_fraction - c) for c in choices) ∧ closest = 1/4 :=
begin
  sorry
end

end closest_fraction_to_medals_won_l133_133394


namespace S_is_three_rays_l133_133194

structure Point := (x : ℝ) (y : ℝ)

def S : set Point := 
  { p | 
    (p.x = 3 ∧ p.y ≤ 7) ∨
    (p.y = 7 ∧ p.x ≤ 3) ∨
    (p.y = p.x + 4 ∧ p.x ≤ 3)
  }

theorem S_is_three_rays : S = 
  { p | 
    (p.x = 3 ∧ p.y ≤ 7) ∨
    (p.y = 7 ∧ p.x ≤ 3) ∨
    (p.y = p.x + 4 ∧ p.x ≤ 3)
  } := 
  sorry

end S_is_three_rays_l133_133194


namespace counties_received_rain_on_tuesday_l133_133160

variables (P : Set → ℝ) (A B : Set)

-- Given conditions
axiom PA : P A = 0.7
axiom PAB : P (A ∩ B) = 0.4
axiom neitherA_norB : P (Set.univ \ (A ∪ B)) = 0.2

-- Required to prove
theorem counties_received_rain_on_tuesday :
  P B = 0.5 :=
by
  -- Proof omitted
  sorry

end counties_received_rain_on_tuesday_l133_133160


namespace friends_recycle_15_pounds_l133_133508

-- Define the given conditions in Lean 4
variable (pounds_per_point : ℝ := 3) -- Condition: For every 3 pounds they earned 1 point.
variable (gwen_pounds : ℝ := 5) -- Condition: Gwen recycled 5 pounds.
variable (total_points : ℕ := 6) -- Condition: They earned 6 points.

-- Define the target: How many pounds did Gwen's friends recycle?
def friends_pounds : ℝ := (total_points - ⌊gwen_pounds / pounds_per_point⌋) * pounds_per_point

-- Lean 4 statement for the proof problem
theorem friends_recycle_15_pounds : friends_pounds pounds_per_point gwen_pounds total_points = 15 := by
  sorry

end friends_recycle_15_pounds_l133_133508


namespace max_a_value_l133_133477

theorem max_a_value 
  (x y a : ℝ) 
  (h1 : x - y ≤ 0) 
  (h2 : x + y - 5 ≥ 0) 
  (h3 : y - 3 ≤ 0) : 
  a ≤ 25 / 13 → ∀ x y, (a * (x^2 + y^2) ≤ (x + y)^2) := 
begin
  sorry
end

end max_a_value_l133_133477


namespace find_m_for_area_of_triangles_l133_133062

noncomputable def ellipse_foci := (-real.sqrt 2, 0, real.sqrt 2, 0)

theorem find_m_for_area_of_triangles : 
  (∃ (m : ℝ), 
    let F1 := (-real.sqrt 2, 0) in
    let F2 := (real.sqrt 2, 0) in
    let A B : ℝ × ℝ in
    let ellipse : ∀ x y : ℝ, (x^2 / 3) + y^2 = 1 := sorry in
    let line := λ x y : ℝ, y = x + m in
    line_intersects_ellipse : ∀ A B : ℝ × ℝ, disjoint ((A, B) ∪ ellipse) := sorry in
    (area_of_triangle(F1, A, B) = 2 * area_of_triangle(F2, A, B)) → m = -real.sqrt 2 / 3) :=
sorry

end find_m_for_area_of_triangles_l133_133062


namespace largest_integer_prime_l133_133321

open Int

noncomputable def quadratic_expression (x : ℤ) : ℤ := |8 * x^2 - 62 * x + 21|

def is_prime_quadratic (x : ℤ) : Prop := nat.prime (int.nat_abs (quadratic_expression x))

theorem largest_integer_prime :
  ∃ x : ℤ, is_prime_quadratic x ∧ (∀ y : ℤ, is_prime_quadratic y → x ≥ y) ∧ x = 4 :=
begin
  existsi 4,
  split,
  { sorry },
  split,
  { intros y hy,
    sorry },
  { refl }
end

end largest_integer_prime_l133_133321


namespace triangle_median_length_l133_133952

variable (XY XZ XM YZ : ℝ)

theorem triangle_median_length :
  XY = 6 →
  XZ = 8 →
  XM = 5 →
  YZ = 10 := by
  sorry

end triangle_median_length_l133_133952


namespace total_people_served_l133_133591

variable (total_people : ℕ)
variable (people_not_buy_coffee : ℕ := 10)

theorem total_people_served (H : (2 / 5 : ℚ) * total_people = people_not_buy_coffee) : total_people = 25 := 
by
  sorry

end total_people_served_l133_133591


namespace total_photos_newspaper_l133_133536

theorem total_photos_newspaper (pages1 pages2 photos_per_page1 photos_per_page2 : ℕ)
  (h1 : pages1 = 12) (h2 : photos_per_page1 = 2)
  (h3 : pages2 = 9) (h4 : photos_per_page2 = 3) :
  (pages1 * photos_per_page1) + (pages2 * photos_per_page2) = 51 :=
by
  sorry

end total_photos_newspaper_l133_133536


namespace relationship_between_a_and_b_l133_133090

variable (a b : ℝ)

def in_interval (x : ℝ) := 0 < x ∧ x < 1

theorem relationship_between_a_and_b 
  (ha : in_interval a)
  (hb : in_interval b)
  (h : (1 - a) * b > 1 / 4) : a < b :=
sorry

end relationship_between_a_and_b_l133_133090


namespace quadratic_root_reciprocal_l133_133469

theorem quadratic_root_reciprocal :
  (∀ (a b : ℝ), (a + b = 6) ∧ (a * b = -5) → (1 / a + 1 / b = -6 / 5)) :=
by
  intros a b h
  cases h with hab hprod
  rw [div_add_div_same, hab, hprod, add_neg_div]
  sorry

end quadratic_root_reciprocal_l133_133469


namespace cos_theta_minus_phi_eq_l133_133144

theorem cos_theta_minus_phi_eq
  (θ φ : ℝ)
  (h1 : complex.exp (complex.I * θ) = (4/5 : ℂ) + (3/5 : ℂ) * complex.I)
  (h2 : complex.exp (complex.I * φ) = (5/13 : ℂ) + (12/13 : ℂ) * complex.I) :
  real.cos (θ - φ) = -16/65 :=
by
  sorry

end cos_theta_minus_phi_eq_l133_133144


namespace mathematicians_correctness_l133_133264

theorem mathematicians_correctness:
  ∀ (w1₁ w1₂ t1₁ t1₂ w2₁ w2₂ t2₁ t2₂: ℕ),
  (w1₁ = 4) ∧ (t1₁ = 7) ∧ (w1₂ = 3) ∧ (t1₂ = 5) →
  (w2₁ = 8) ∧ (t2₁ = 14) ∧ (w2₂ = 3) ∧ (t2₂ = 5) →
  (4/7 : ℚ) < 3/5 →
  (w1₁ + w1₂) / (t1₁ + t1₂) = 7 / 12 ∨ (w2₁ + w2₂) / (t2₁ + t2₂) = 11 / 19 →
  ¬ (w1₁ + w1₂) / (t1₁ + t1₂) < 19 / 35 ∧ (w2₁ + w2₂) / (t2₁ + t2₂) < 19 / 35 :=
by
  sorry

end mathematicians_correctness_l133_133264


namespace number_of_monsters_l133_133610

theorem number_of_monsters
    (M S : ℕ)
    (h1 : 4 * M + 3 = S)
    (h2 : 5 * M = S - 6) :
  M = 9 :=
sorry

end number_of_monsters_l133_133610


namespace precise_approximate_classification_l133_133748

def data_points : List String := ["Xiao Ming bought 5 books today",
                                  "The war in Afghanistan cost the United States $1 billion per month in 2002",
                                  "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion",
                                  "The human brain has 10,000,000,000 cells",
                                  "Xiao Hong scored 92 points on this test",
                                  "The Earth has more than 1.5 trillion tons of coal reserves"]

def is_precise (data : String) : Bool :=
  match data with
  | "Xiao Ming bought 5 books today" => true
  | "The war in Afghanistan cost the United States $1 billion per month in 2002" => true
  | "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion" => true
  | "Xiao Hong scored 92 points on this test" => true
  | _ => false

def is_approximate (data : String) : Bool :=
  match data with
  | "The human brain has 10,000,000,000 cells" => true
  | "The Earth has more than 1.5 trillion tons of coal reserves" => true
  | _ => false

theorem precise_approximate_classification :
  (data_points.filter is_precise = ["Xiao Ming bought 5 books today",
                                    "The war in Afghanistan cost the United States $1 billion per month in 2002",
                                    "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion",
                                    "Xiao Hong scored 92 points on this test"]) ∧
  (data_points.filter is_approximate = ["The human brain has 10,000,000,000 cells",
                                        "The Earth has more than 1.5 trillion tons of coal reserves"]) :=
by sorry

end precise_approximate_classification_l133_133748


namespace pentadecagon_diagonals_l133_133401

theorem pentadecagon_diagonals :
  let n := 15
  in (n * (n - 3)) / 2 = 90 :=
by
  let n := 15
  have h : (n * (n - 3)) / 2 = (15 * (15 - 3)) / 2 := by rfl
  have h1 : (15 * (15 - 3)) / 2 = (15 * 12) / 2 := by rfl
  have h2 : (15 * 12) / 2 = 180 / 2 := by rfl
  have h3 : 180 / 2 = 90 := by rfl
  show (n * (n - 3)) / 2 = 90 from h3

end pentadecagon_diagonals_l133_133401


namespace find_t_0_l133_133285

noncomputable theory

open_locale classical

-- Define the motion equation
def motion_equation (t : ℝ) : ℝ := 7 * t ^ 2 - 13 * t + 8

-- Define the derivative of the motion equation
def motion_derivative (t : ℝ) : ℝ := 14 * t - 13

-- Define the statement that the instantaneous velocity at t_0 is 1
def instantaneous_velocity_at (t_0 : ℝ) : Prop := motion_derivative t_0 = 1

theorem find_t_0 :
  ∃ (t_0 : ℝ), instantaneous_velocity_at t_0 :=
begin
  use 1,
  unfold instantaneous_velocity_at motion_derivative,
  norm_num,
end

end find_t_0_l133_133285


namespace solve_inequality_minimum_value_of_function_l133_133613

def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 4)

theorem solve_inequality (x : ℝ) : f(x) > 2 ↔ x < -7 ∨ x > 5 / 3 :=
sorry

theorem minimum_value_of_function : ∃ x : ℝ, ∀ y : ℝ, f(x) ≤ f(y) ∧ f(x) = -9 / 2 :=
sorry

end solve_inequality_minimum_value_of_function_l133_133613


namespace find_m_l133_133065

noncomputable def ellipse : set (ℝ × ℝ) := { p | (p.1)^2 / 3 + (p.2)^2 = 1 }
def f1 : ℝ × ℝ := (- real.sqrt 2, 0)
def f2 : ℝ × ℝ := (real.sqrt 2, 0)

def line (m : ℝ) : set (ℝ × ℝ) := { p | p.2 = p.1 + m }

theorem find_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line m ∧ B ∈ line m ∧
   2 * (abs ((f1.1 * (A.2 - B.2) + A.1 * (B.2 - f1.2) + B.1 * (f1.2 - A.2)) / 2))
 = abs ((f2.1 * (A.2 - B.2) + A.1 * (B.2 - f2.2) + B.1 * (f2.2 - A.2)) / 2)) →
  m = - real.sqrt 2 / 3 :=
sorry

end find_m_l133_133065


namespace math_problem_l133_133973

variables {a b c d e : ℤ}

theorem math_problem 
(h1 : a - b + c - e = 7)
(h2 : b - c + d + e = 9)
(h3 : c - d + a - e = 5)
(h4 : d - a + b + e = 1)
: a + b + c + d + e = 11 := 
by 
  sorry

end math_problem_l133_133973


namespace circumcircle_diameter_l133_133951

theorem circumcircle_diameter {a b : ℝ} {cosθ : ℝ} 
  (ha : a = 2) 
  (hb : b = 3) 
  (hcosθ : cosθ = 1 / 3) : 
  diameter_of_circumcircle a b cosθ = 9 * Real.sqrt 2 / 4 := 
sorry

end circumcircle_diameter_l133_133951


namespace probability_factor_120_lt_8_l133_133323

theorem probability_factor_120_lt_8 :
  let n := 120
  let total_factors := 16
  let favorable_factors := 6
  (6 / 16 : ℚ) = 3 / 8 :=
by 
  sorry

end probability_factor_120_lt_8_l133_133323


namespace mixture_proportion_exists_l133_133119

-- Define the ratios and densities of the liquids
variables (k : ℝ) (ρ1 ρ2 ρ3 : ℝ) (m1 m2 m3 : ℝ)
variables (x y : ℝ)

-- Given conditions
def density_ratio : Prop := 
  ρ1 = 6 * k ∧ ρ2 = 3 * k ∧ ρ3 = 2 * k

def mass_condition : Prop := 
  m2 / m1 ≤ 2 / 7

-- Must prove that a solution exists where the resultant density is the arithmetic mean
def mixture_density : Prop := 
  (m1 + m2 + m3) / ((m1 / ρ1) + (m2 / ρ2) + (m3 / ρ3)) = (ρ1 + ρ2 + ρ3) / 3

-- Statement (No proof provided)
theorem mixture_proportion_exists (k : ℝ) (m1 m2 m3 : ℝ) (ρ1 ρ2 ρ3 : ℝ) (x y : ℝ) :
  density_ratio k ρ1 ρ2 ρ3 →
  mass_condition m1 m2 →
  mixture_density m1 m2 m3 k ρ1 ρ2 ρ3 :=
sorry

end mixture_proportion_exists_l133_133119


namespace product_of_two_numbers_l133_133652

theorem product_of_two_numbers (a b : ℝ) (h1 : a + b = 70) (h2 : a - b = 10) : a * b = 1200 := 
by
  sorry

end product_of_two_numbers_l133_133652


namespace perfect_score_is_21_l133_133915

-- Define the number of games played
def num_games : ℕ := 3

-- Define the points per game
def points_per_game : ℕ := 7

-- Define the perfect score based on the conditions
def perfect_score : ℕ := num_games * points_per_game

-- The theorem to prove the perfect score is 21 points
theorem perfect_score_is_21 : perfect_score = 21 := by
  calc
    perfect_score = num_games * points_per_game : rfl
    ... = 3 * 7 : rfl
    ... = 21 : by norm_num

end perfect_score_is_21_l133_133915


namespace sum_of_intersection_points_l133_133882

noncomputable def f (x : ℝ) : ℝ := sorry

theorem sum_of_intersection_points (m : ℕ) 
  (inter_points : Fin m → (ℝ × ℝ))
  (h_inter : ∀ (i : Fin m), 
    ∃ x : ℝ, (inter_points i).1 = x ∧ (inter_points i).2 = f x ∧ f (-x) = 2 - f x) 
  (h_eq : ∀ (i : Fin m), (inter_points i).2 = (1 + 1 / (inter_points i).1) ∨ (inter_points i).2 = f (inter_points i).1) :
  ∑ i, (inter_points i).1 + (inter_points i).2 = m := 
sorry

end sum_of_intersection_points_l133_133882


namespace arithmetic_sequence_common_difference_l133_133475

theorem arithmetic_sequence_common_difference (a : Nat → Int)
  (h1 : a 1 = 2) 
  (h3 : a 3 = 8)
  (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1))  -- General form for an arithmetic sequence given two terms
  : a 2 - a 1 = 3 :=
by
  -- The main steps of the proof will follow from the arithmetic progression properties
  sorry

end arithmetic_sequence_common_difference_l133_133475


namespace smallest_a1_value_l133_133206

noncomputable def a_seq (n : ℕ) : ℝ :=
if n = 0 then 29 / 98 else if n > 0 then 15 * a_seq (n - 1) - 2 * n else 0

theorem smallest_a1_value :
  (∃ f : ℕ → ℝ, (∀ n > 0, f n = 15 * f (n - 1) - 2 * n) ∧ (∀ n, f n > 0) ∧ (f 1 = 29 / 98)) :=
sorry

end smallest_a1_value_l133_133206


namespace min_cost_l133_133772

/-- Define the problem parameters and constraints. -/
def bus_capacity_A := 36
def bus_capacity_B := 60
def cost_A := 1600
def cost_B := 2400
def total_buses := 21
def max_difference := 7
def total_passengers := 900

/-- Define the cost function for the given number of model A and model B buses. -/
def cost (x y : ℕ) : ℕ := cost_A * x + cost_B * y

/-- Prove that the minimum rental cost is 36800 yuan under the given constraints. -/
theorem min_cost :
  ∃ (x y : ℕ), 
    36 * x + 60 * y ≥ 900 ∧
    x + y ≤ 21 ∧
    y - x ≤ 7 ∧
    x ≥ 0 ∧
    y ≥ 0 ∧
    cost x y = 36800 :=
by
  use 5, 12
  simp [bus_capacity_A, bus_capacity_B, cost_A, cost_B, cost, total_buses, max_difference, total_passengers]
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; sorry -- This will be filled in with the results of cost calculation.

end min_cost_l133_133772


namespace two_digit_number_representation_l133_133773

def tens_digit := ℕ
def units_digit := ℕ

theorem two_digit_number_representation (b a : ℕ) : 
  (∀ (b a : ℕ), 10 * b + a = 10 * b + a) := sorry

end two_digit_number_representation_l133_133773


namespace probability_two_green_balls_l133_133334

theorem probability_two_green_balls (r y g: ℕ) (htotal: r = 3 ∧ y = 5 ∧ g = 4):
    let total_balls := r + y + g in total_balls = 12 →
    let ways_total := Nat.choose total_balls 3 in ways_total = 220 →

    let ways_two_green := Nat.choose g 2 * Nat.choose (r + y) 1 in ways_two_green = 48 →

    (ways_two_green / ways_total : ℚ) = 12 / 55 :=
begin
  sorry,
end

end probability_two_green_balls_l133_133334


namespace bus_ride_duration_l133_133992

theorem bus_ride_duration (total_hours : ℕ) (train_hours : ℕ) (walk_minutes : ℕ) (wait_factor : ℕ) 
    (h_total : total_hours = 8)
    (h_train : train_hours = 6)
    (h_walk : walk_minutes = 15)
    (h_wait : wait_factor = 2) : 
    let total_minutes := total_hours * 60
    let train_minutes := train_hours * 60
    let wait_minutes := wait_factor * walk_minutes
    let travel_minutes := total_minutes - train_minutes
    let bus_ride_minutes := travel_minutes - walk_minutes - wait_minutes
    bus_ride_minutes = 75 :=
by
  sorry

end bus_ride_duration_l133_133992


namespace liquid_mixture_ratio_l133_133125

theorem liquid_mixture_ratio (m1 m2 m3 : ℝ) (ρ1 ρ2 ρ3 : ℝ) (k : ℝ)
  (hρ1 : ρ1 = 6 * k) (hρ2 : ρ2 = 3 * k) (hρ3 : ρ3 = 2 * k)
  (h_condition : m1 ≥ 3.5 * m2)
  (h_arith_mean : (m1 + m2 + m3) / (m1 / ρ1 + m2 / ρ2 + m3 / ρ3) = (ρ1 + ρ2 + ρ3) / 3) :
    ∃ x y : ℝ, x ≤ 2/7 ∧ (4 * x + 15 * y = 7) := sorry

end liquid_mixture_ratio_l133_133125


namespace gcd_three_digit_palindromes_l133_133679

theorem gcd_three_digit_palindromes : ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → Nat.gcd (102 * a + 10 * b) 1 = 1 :=
by
  intros a b h
  sorry

end gcd_three_digit_palindromes_l133_133679


namespace sin_beta_plus_five_pi_over_four_l133_133481

theorem sin_beta_plus_five_pi_over_four (α β : ℝ)
  (h1 : sin (α - β) * cos α - cos (α - β) * sin α = 3 / 5)
  (h2 : π < β ∧ β < 3 * π / 2) :
  sin (β + 5 * π / 4) = (7 * Real.sqrt 2) / 10 := 
by
  sorry

end sin_beta_plus_five_pi_over_four_l133_133481


namespace total_toothpicks_grid_area_l133_133660

open Nat

-- Definitions
def grid_length : Nat := 30
def grid_width : Nat := 50

-- Prove the total number of toothpicks
theorem total_toothpicks : (31 * grid_width + 51 * grid_length) = 3080 := by
  sorry

-- Prove the area enclosed by the grid
theorem grid_area : (grid_length * grid_width) = 1500 := by
  sorry

end total_toothpicks_grid_area_l133_133660


namespace martha_points_calculation_l133_133994

theorem martha_points_calculation :
  let beef_cost := 3 * 11
  let beef_discount := 0.10 * beef_cost
  let total_beef_cost := beef_cost - beef_discount

  let fv_cost := 8 * 4
  let fv_discount := 0.05 * fv_cost
  let total_fv_cost := fv_cost - fv_discount

  let spices_cost := 2 * 6

  let other_groceries_cost := 37 - 3

  let total_cost := total_beef_cost + total_fv_cost + spices_cost + other_groceries_cost

  let spending_points := (total_cost / 10).floor * 50

  let bonus_points_over_100 := if total_cost > 100 then 250 else 0

  let loyalty_points := 100
  
  spending_points + bonus_points_over_100 + loyalty_points = 850 := by
    sorry

end martha_points_calculation_l133_133994


namespace cot_sum_eq_cot_sum_example_l133_133837

theorem cot_sum_eq (a b c d : ℝ) :
  cot (arccot a + arccot b + arccot c + arccot d) = (a * b * c * d - (a * b + b * c + c * d) + 1) / (a + b + c + d)
:= sorry

theorem cot_sum_example :
  cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 1021 / 420
:= sorry

end cot_sum_eq_cot_sum_example_l133_133837


namespace gcf_of_three_digit_palindromes_is_one_l133_133688

-- Define a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define the greatest common factor (gcd) function
def gcd_of_all_palindromes : ℕ :=
  (Finset.range 999).filter is_palindrome |>.list.foldr gcd 0

-- State the theorem
theorem gcf_of_three_digit_palindromes_is_one :
  gcd_of_all_palindromes = 1 :=
sorry

end gcf_of_three_digit_palindromes_is_one_l133_133688


namespace ellipse_intersection_area_condition_l133_133025

theorem ellipse_intersection_area_condition :
  let ellipse := ∀ (x y : ℝ), (x^2 / 3 + y^2 = 1)
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let line m := ∀ (x y : ℝ), (y = x + m)
  ∀ (A B : ℝ × ℝ),
    ((A.1^2 / 3 + A.2^2 = 1) ∧ (y = A.1 + m)) ∧
    ((B.1^2 / 3 + B.2^2 = 1) ∧ (y = B.1 + m)) ∧
    (area (F1, A, B) = 2 * area (F2, A, B)) →
    m = -Real.sqrt 2 / 3 := by sorry

end ellipse_intersection_area_condition_l133_133025


namespace greatest_common_factor_of_three_digit_palindromes_l133_133673

def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b

def gcf (a b : ℕ) : ℕ := 
  if a = 0 then b else gcf (b % a) a

theorem greatest_common_factor_of_three_digit_palindromes : 
  ∃ g, (∀ n, is_palindrome n → g ∣ n) ∧ (∀ d, (∀ n, is_palindrome n → d ∣ n) → d ∣ g) :=
by
  use 101
  sorry

end greatest_common_factor_of_three_digit_palindromes_l133_133673


namespace line_intersects_circle_and_not_pass_center_l133_133424

noncomputable def distance_from_center_to_line (a b c x0 y0 : ℝ) : ℝ :=
  (| a * x0 + b * y0 + c |) / Real.sqrt (a * a + b * b)

theorem line_intersects_circle_and_not_pass_center :
  ∀ (x y : ℝ),
    3 * x - 4 * y - 9 = 0 →
    x^2 + y^2 = 4 →
    let d := distance_from_center_to_line 3 (-4) (-9) 0 0 in
    d < 2 ∧ (3 * 0 - 4 * 0 - 9 ≠ 0) :=
by
  intro x y h_line h_circle d
  sorry

end line_intersects_circle_and_not_pass_center_l133_133424


namespace determinant_positive_l133_133188

variable {n : Type*} [Fintype n] [DecidableEq n]
variable (A : Matrix n n ℝ)

theorem determinant_positive (h : A + A.transpose = 1) : det A > 0 := sorry

end determinant_positive_l133_133188


namespace degree_of_fluctuation_is_variance_l133_133387

-- Definitions based on the conditions in part a)
def Mean (s : Set ℝ) : ℝ := (s.Sum id) / (card s)
def Mode (s : Set ℝ) : Set ℝ := {x ∈ s | ∀ y ∈ s, (count x s) ≥ (count y s)}
def Variance (s : Set ℝ) : ℝ := (s.Sum (λ x, (x - Mean s) ^ 2)) / (card s)
def Frequency (x : ℝ) (s : Set ℝ) : ℕ := count x s

-- The proof problem statement
theorem degree_of_fluctuation_is_variance (s : Set ℝ) :
  (∃ (measure : Set ℝ → ℝ), measure s = Variance s) :=
by sorry

end degree_of_fluctuation_is_variance_l133_133387


namespace min_colors_n_gun_l133_133468

def min_colors (n : ℕ) : ℕ := n * (n + 2)

theorem min_colors_n_gun (n : ℕ) (hn : 0 < n) :
  ∃ (k : ℕ), k = n * (n + 2) ∧
  (∀ (color : ℕ → ℕ → ℕ), (∃ (c : ℕ), ∀ (i j : ℕ), color i j < c) →
   (∀ (x y : ℕ), ∀ (dx dy : ℤ),
    (0 ≤ dx ∧ dx < n ∧ 0 ≤ dy ∧ dy < n) →
    ∃ (xi xj yi yj : ℕ), 
      xi = x + dx ∧ xj = x + dy ∧
      yi = y + dx ∧ yj = y + dy ∧
      color xi xj ≠ color yi yj))
:=
begin
  use min_colors n,
  split,
  { refl, },
  { sorry, }
end

end min_colors_n_gun_l133_133468


namespace smallest_side_length_of_integer_squares_l133_133376

theorem smallest_side_length_of_integer_squares (n : ℕ) 
  (h1 : ∃ (squares : ℕ → ℕ), (∀ i < 15, squares i > 0) ∧ 
          (∑ i in Finset.range 12, squared i = 12) ∧
          (∃ (remaining : list ℕ), remaining.length = 3 ∧ 
                                  (∀ r ∈ remaining, r > 0) ∧ 
                                  (∑ r in remaining, r = n^2 - 12))) :
  n ≥ 5 :=
begin
  sorry
end

end smallest_side_length_of_integer_squares_l133_133376


namespace cyclists_meet_at_starting_point_l133_133734

-- Define the conditions: speeds of cyclists and the circumference of the circle
def speed_cyclist1 : ℝ := 7
def speed_cyclist2 : ℝ := 8
def circumference : ℝ := 300

-- Define the total speed by summing individual speeds
def relative_speed : ℝ := speed_cyclist1 + speed_cyclist2

-- Define the time required to meet at the starting point
def meeting_time : ℝ := 20

-- The theorem statement which states that given the conditions, the cyclists will meet after 20 seconds
theorem cyclists_meet_at_starting_point :
  meeting_time = circumference / relative_speed :=
sorry

end cyclists_meet_at_starting_point_l133_133734


namespace mass_proportion_l133_133124

namespace DensityMixture

variables (k m_1 m_2 m_3 : ℝ)
def rho_1 := 6 * k
def rho_2 := 3 * k
def rho_3 := 2 * k
def arithmetic_mean := (rho_1 k + rho_2 k + rho_3 k) / 3
def density_mixture := (m_1 + m_2 + m_3) / 
    (m_1 / rho_1 k + m_2 / rho_2 k + m_3 / rho_3 k)
def mass_ratio_condition := m_1 / m_2 ≥ 3.5

theorem mass_proportion 
  (k_pos : 0 < k)
  (mass_cond : mass_ratio_condition k m_1 m_2) :
  ∃ (x y : ℝ), (4 * x + 15 * y = 7) ∧ (density_mixture k m_1 m_2 m_3 = arithmetic_mean k) ∧ mass_cond := 
sorry

end DensityMixture

end mass_proportion_l133_133124


namespace tank_capacity_is_353_l133_133380

-- Definitions for the given conditions
def initial_fraction := 1 / 10
def final_fraction := 2 / 3
def added_gallons := 200
def total_capacity := 353

-- The proof statement
theorem tank_capacity_is_353 :
  ∀ (x : ℕ), 
    initial_fraction * x + added_gallons = final_fraction * x →
    x = total_capacity :=
by
  fix x
  assume h : initial_fraction * x + added_gallons = final_fraction * x
  sorry

end tank_capacity_is_353_l133_133380


namespace probability_of_rerolling_two_dice_l133_133562

theorem probability_of_rerolling_two_dice :
  let probability := 5/72 in
  ∃ (a b c : ℕ), (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) ∧ 
                 (a + b + c = 9) ∧ (a ≤ 4) ∧ (a + b ≥ 9) ∧ b ∈ [5, 6] ∧ 
                 c ∈ [5, 6] ∧ 
                 (∃ n, n = 15) ∧ 
                 ∃ total_outcomes, total_outcomes = 216 ∧ 
                 probability = (n / total_outcomes) := sorry

end probability_of_rerolling_two_dice_l133_133562


namespace dot_product_uv_32_5_l133_133506

noncomputable def angle := 60 * Real.pi / 180 -- converting 60 degrees to radians
def norm_u : ℝ := 5
def norm_v : ℝ := 13

def dot_product (u v : ℝ) (theta : ℝ) : ℝ := u * v * Real.cos theta

theorem dot_product_uv_32_5 : dot_product norm_u norm_v angle = 32.5 :=
by
  -- the proof will go here
  sorry

end dot_product_uv_32_5_l133_133506


namespace find_m_value_l133_133050

-- Define the ellipse equation and its foci
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define the foci of the ellipse C
def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the line equation
def line (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the condition for the areas of triangles F1AB and F2AB
def triangle_area_condition (F1 F2 : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := sorry in -- Coordinates of intersection points need to be defined
  let B := sorry in -- Coordinates of intersection points need to be defined
  sorry /- The actual calculation of the areas would require more complex integration and geometry.
           Here we just outline the condition -/

-- The proof statement
theorem find_m_value :
  ∃ m : ℝ, 
    ∀ x y : ℝ,
      ellipse x y →
      line m x y →
      triangle_area_condition F1 F2 m →
      m = -Real.sqrt 2 / 3 :=
  sorry

end find_m_value_l133_133050


namespace possible_card_numbers_l133_133214

theorem possible_card_numbers
  (a b c d : ℕ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : c ≤ d)
  (h4 : ∀ x y, (
    (exists (p q : {p // p ∈ {a, b, c, d} ∧ q ∈ {a,b,c,d} ∧ p ≠ q}), (p + q < 9) ↔ (1/3 : ℚ)) ∧
    (exists (r s : {r // r ∈ {a, b, c, d} ∧ s ∈ {a, b, c, d} ∧ r ≠ s}), (r + s = 9) ↔ (1/3 : ℚ)) ∧
    (exists (t u : {t // t ∈ {a, b, c, d} ∧ u ∈ {a, b, c, d} ∧ t ≠ u}), (t + u > 9) ↔ (1/3 : ℚ))
  )) :
  (a = 1 ∧ b = 2 ∧ c = 7 ∧ d = 8) ∨
  (a = 1 ∧ b = 3 ∧ c = 6 ∧ d = 8) ∨
  (a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 8) ∨
  (a = 2 ∧ b = 3 ∧ c = 6 ∧ d = 7) ∨
  (a = 2 ∧ b = 4 ∧ c = 5 ∧ d = 7) ∨
  (a = 3 ∧ b = 4 ∧ c = 5 ∧ d = 6) :=
sorry

end possible_card_numbers_l133_133214


namespace area_of_region_l133_133398

theorem area_of_region :
  let region := { p : ℝ × ℝ | abs (p.1 - 50) + abs (p.2) = abs (p.1 / 5) } in
  let area := (1/2) * 20.83 * 20 in
  region.nonempty → ∃ a : ℝ, a = 208.3 ∧ ∀ x ∈ region, x = area :=
sorry

end area_of_region_l133_133398


namespace ellipse_area_condition_l133_133073

open Real

theorem ellipse_area_condition (m : ℝ) :
  (∀ x y : ℝ, ((x^2)/3 + y^2 = 1) → (y = x + m) → 
  let F₁ := (-sqrt 2, 0)
      F₂ := (sqrt 2, 0)
  in distance F₁ (x, y) = 2 * distance F₂ (x, y)) → 
  m = -sqrt 2 / 3 :=
by
  sorry

end ellipse_area_condition_l133_133073


namespace largest_coefficient_term_l133_133004

theorem largest_coefficient_term :
  let bin_expansion := (x^2 - (1/x))^11 in
  ∃ (n : ℕ), n = 6 ∧ term_with_largest_coefficient bin_expansion (n + 1) :=
by
  let x := sorry -- x is a placeholder for the variable in the binomial expression
  exact sorry

end largest_coefficient_term_l133_133004


namespace valid_probabilities_and_invalid_probability_l133_133267

theorem valid_probabilities_and_invalid_probability :
  (let first_box_1 := (4, 7)
       second_box_1 := (3, 5)
       combined_prob_1 := (first_box_1.1 + second_box_1.1) / (first_box_1.2 + second_box_1.2),
       first_box_2 := (8, 14)
       second_box_2 := (3, 5)
       combined_prob_2 := (first_box_2.1 + second_box_2.1) / (first_box_2.2 + second_box_2.2),
       prob_1 := first_box_1.1 / first_box_1.2,
       prob_2 := second_box_2.1 / second_box_2.2
     in (combined_prob_1 = 7 / 12 ∧ combined_prob_2 = 11 / 19) ∧ (19 / 35 < prob_1 ∧ prob_1 < prob_2) → False) :=
by
  sorry

end valid_probabilities_and_invalid_probability_l133_133267


namespace product_lcm_gcd_12_15_l133_133446

theorem product_lcm_gcd_12_15 : 
  let a := 12
  let b := 15
  let gcd := Nat.gcd a b
  let lcm := Nat.lcm a b
  in gcd * lcm = 180 :=
by 
  let a := 12
  let b := 15
  let gcd := Nat.gcd a b
  let lcm := Nat.lcm a b
  show gcd * lcm = 180, from sorry

end product_lcm_gcd_12_15_l133_133446


namespace cot_inverse_sum_l133_133840

theorem cot_inverse_sum : 
  cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 97 / 40 :=
by
  sorry

end cot_inverse_sum_l133_133840


namespace find_c_l133_133154

theorem find_c (c : ℝ) (h : ∀ x, x = (-13 + real.sqrt 19) / 4 ∨ x = (-13 - real.sqrt 19) / 4 → (∀ x, 2 * x^2 + 13 * x + c = 0)) : c = 18.75 :=
sorry

end find_c_l133_133154


namespace mathematicians_correctness_l133_133262

theorem mathematicians_correctness:
  ∀ (w1₁ w1₂ t1₁ t1₂ w2₁ w2₂ t2₁ t2₂: ℕ),
  (w1₁ = 4) ∧ (t1₁ = 7) ∧ (w1₂ = 3) ∧ (t1₂ = 5) →
  (w2₁ = 8) ∧ (t2₁ = 14) ∧ (w2₂ = 3) ∧ (t2₂ = 5) →
  (4/7 : ℚ) < 3/5 →
  (w1₁ + w1₂) / (t1₁ + t1₂) = 7 / 12 ∨ (w2₁ + w2₂) / (t2₁ + t2₂) = 11 / 19 →
  ¬ (w1₁ + w1₂) / (t1₁ + t1₂) < 19 / 35 ∧ (w2₁ + w2₂) / (t2₁ + t2₂) < 19 / 35 :=
by
  sorry

end mathematicians_correctness_l133_133262


namespace concentric_circles_l133_133391

-- Given a cyclic quadrilateral ABCD
variables {A B C D A1 B1 C1 D1 A2 B2 C2 D2 A3 B3 C3 D3 : Type}
variable (h_cyclic : ∀ (P : Type), P ∈ {A, B, C, D} → P ∈ Circle)

-- Definitions of the reflected points forming three cyclic quadrilaterals
noncomputable def reflections_ABC (A B C D : Type) :=
  ∃ (A1 B1 C1 D1 : Type), ((A1 = reflect A B C) ∧ (B1 = reflect B A C) ∧ 
  (C1 = reflect C B A) ∧ (D1 = reflect D B C))

noncomputable def reflections_ABD (A B C D : Type) :=
  ∃ (A2 B2 C2 D2 : Type), ((A2 = reflect A B D) ∧ (B2 = reflect B A D) ∧ 
  (C2 = reflect C B D) ∧ (D2 = reflect D B A))

noncomputable def reflections_ACD (A B C D : Type) :=
  ∃ (A3 B3 C3 D3 : Type), ((A3 = reflect A C D) ∧ (B3 = reflect B A D) ∧
  (C3 = reflect C A D) ∧ (D3 = reflect D A C))

-- Conclude that the three circles are concentric
theorem concentric_circles (A B C D : Type) (h_cyclic : ∀ (P : Type), P ∈ {A, B, C, D} → P ∈ Circle)
  (R1 : reflections_ABC A B C D) (R2 : reflections_ABD A B C D) (R3 : reflections_ACD A B C D) :
  concentric R1 R2 R3 :=
sorry

end concentric_circles_l133_133391


namespace rectangle_length_l133_133919

theorem rectangle_length (sq_side_len rect_width : ℕ) (sq_area : ℕ) (rect_len : ℕ) 
    (h1 : sq_side_len = 6) 
    (h2 : rect_width = 4) 
    (h3 : sq_area = sq_side_len * sq_side_len) 
    (h4 : sq_area = rect_width * rect_len) :
    rect_len = 9 := 
by 
  sorry

end rectangle_length_l133_133919


namespace fifth_almost_perfect_is_32_l133_133916

def is_almost_perfect (n : ℕ) : Prop :=
  (∑ i in finset.filter (λ d, d ∣ n ∧ d ≠ n) (finset.range (n + 1)), i) = n - 1

theorem fifth_almost_perfect_is_32 : ∃ n : ℕ, is_almost_perfect (2^n) ∧ finset.nth (finset.filter is_almost_perfect finset.range) 5 = 32 :=
sorry

end fifth_almost_perfect_is_32_l133_133916


namespace total_students_in_class_l133_133778

theorem total_students_in_class (A M B : ℕ) (hA : A = 35) (hM : M = 32) (hB : B = 19)
    (h : ∀ x, x ∉ A ∨ x ∉ M → false) :
  A + M - B = 48 :=
by
  rw [hA, hM, hB]
  exact rfl

end total_students_in_class_l133_133778


namespace alia_study_minutes_l133_133385

theorem alia_study_minutes (
  study_first_7_days : ℕ,
  study_next_4_days : ℕ,
  total_days : ℕ,
  daily_average_minutes : ℕ
) (h1 : study_first_7_days = 7 * 30)
  (h2 : study_next_4_days = 4 * 45)
  (h3 : total_days = 12)
  (h4 : daily_average_minutes = 40) :
  (480 - (study_first_7_days + study_next_4_days)) = 90 :=
by
  sorry

end alia_study_minutes_l133_133385


namespace line_charactersitic_l133_133776

theorem line_charactersitic :
  let line_eq := λ x, -x - 3 in
  (∃ (p : ℝ × ℝ), p = (3, 0) ∧ line_eq p.1 = 0) → False :=
by
  let line_eq := λ x, -x - 3
  intro h
  obtain ⟨p, hp₁, hp₂⟩ := h
  rw hp₁ at hp₂
  sorry

end line_charactersitic_l133_133776


namespace find_x_plus_y_l133_133019

theorem find_x_plus_y (x y : ℝ) 
    (h1 : 27^(x - 2) = 3^(3 * y + 6))
    (h2 : 8^x = 2^(2 * y + 14)) :
    x + y = 8 :=
sorry

end find_x_plus_y_l133_133019


namespace gcd_three_digit_palindromes_l133_133676

theorem gcd_three_digit_palindromes : ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → Nat.gcd (102 * a + 10 * b) 1 = 1 :=
by
  intros a b h
  sorry

end gcd_three_digit_palindromes_l133_133676


namespace sum_digits_next_l133_133968

-- Given the sum of the digits function S(n)
def S (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Defining the properties based on conditions
theorem sum_digits_next (n : ℕ) (h : S n = 876) : S (n + 1) = 877 :=
begin
  sorry -- Proof goes here
end

end sum_digits_next_l133_133968


namespace pedal_circle_intersection_l133_133739

noncomputable def pedal_circle (P: Point) (A B C: Point) : Circle := sorry 

theorem pedal_circle_intersection
  (A B C D : Point) 
  {ABC_parallelogram : A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ parallelogram A B C D} :
  let O := intersection_of_diagonals A C B D in
  pedal_circle D A B C passes_through O :=
sorry

end pedal_circle_intersection_l133_133739


namespace recipe_calls_for_nine_cups_of_flour_l133_133995

def cups_of_flour (x : ℕ) := 
  ∃ cups_added_sugar : ℕ, 
    cups_added_sugar = (6 - 4) ∧ 
    x = cups_added_sugar + 7

theorem recipe_calls_for_nine_cups_of_flour : cups_of_flour 9 :=
by
  sorry

end recipe_calls_for_nine_cups_of_flour_l133_133995


namespace problem1_problem2_l133_133799

-- Problem 1 Statement
theorem problem1 (a : ℝ) (h : a ≠ 1) : (a^2 / (a - 1) - a - 1) = (1 / (a - 1)) :=
by
  sorry

-- Problem 2 Statement
theorem problem2 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ -y) : 
  (2 * x * y / (x^2 - y^2)) / ((1 / (x - y)) + (1 / (x + y))) = y :=
by
  sorry

end problem1_problem2_l133_133799


namespace trigonometric_identity_l133_133347

theorem trigonometric_identity :
  Real.tan (70 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) * (Real.sqrt 3 * Real.tan (20 * Real.pi / 180) - 1) = -1 :=
by
  sorry

end trigonometric_identity_l133_133347


namespace gcd_three_digit_palindromes_l133_133709

theorem gcd_three_digit_palindromes : 
  GCD (set.image (λ (p : ℕ × ℕ), 101 * p.1 + 10 * p.2) 
    ({a | a ≠ 0 ∧ a < 10} × {b | b < 10})) = 1 := 
by
  sorry

end gcd_three_digit_palindromes_l133_133709


namespace sin_four_theta_l133_133908

theorem sin_four_theta 
  (θ : ℂ) 
  (h : complex.exp (complex.I * θ) = (2 + complex.I * real.sqrt 5) / 3) : 
  real.sin (4 * θ.im) = -((8 * real.sqrt 5) / 81) :=
sorry

end sin_four_theta_l133_133908


namespace range_of_m_l133_133855

theorem range_of_m
  (α β : ℝ)
  (m : ℝ)
  (h₁ : α ∈ set.Icc (-real.pi / 2) (real.pi / 2))
  (h₂ : β ∈ set.Icc (-real.pi / 2) (real.pi / 2))
  (h₃ : α + β < 0)
  (h₄ : real.sin α = 1 - m)
  (h₅ : real.sin β = 1 - m^2) :
  m ∈ set.Ioo 1 (real.sqrt 2) ∪ set.Icc 1 (real.sqrt 2) :=
sorry

end range_of_m_l133_133855


namespace count_primes_eq_three_l133_133454

def satisfies_congruences (p m n k : ℤ) : Prop :=
  (m + n + k) % p = 0 ∧
  (m * n + m * k + n * k) % p = 1 ∧
  (m * n * k) % p = 2

def count_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11, 13]
  primes.count (λ p, ∃ m n k : ℤ, satisfies_congruences p m n k)

theorem count_primes_eq_three : count_primes = 3 := by
  sorry

end count_primes_eq_three_l133_133454


namespace curve_is_circle_l133_133248

theorem curve_is_circle (ρ θ : ℝ) (h : ρ = 5 * Real.sin θ) : 
  ∃ (C : ℝ × ℝ) (r : ℝ), ∀ (x y : ℝ),
  (x, y) = (ρ * Real.cos θ, ρ * Real.sin θ) → 
  (x - C.1) ^ 2 + (y - C.2) ^ 2 = r ^ 2 :=
by
  existsi (0, 5 / 2), 5 / 2
  sorry

end curve_is_circle_l133_133248


namespace dentist_filling_cost_l133_133319

variable (F : ℝ)
variable (total_bill : ℝ := 5 * F)
variable (cleaning_cost : ℝ := 70)
variable (extraction_cost : ℝ := 290)
variable (two_fillings_cost : ℝ := 2 * F)

theorem dentist_filling_cost :
  total_bill = cleaning_cost + two_fillings_cost + extraction_cost → 
  F = 120 :=
by
  intros h
  sorry

end dentist_filling_cost_l133_133319


namespace science_club_members_neither_l133_133999

theorem science_club_members_neither {S B C : ℕ} (total : S = 60) (bio : B = 40) (chem : C = 35) (both : ℕ := 25) :
    S - ((B - both) + (C - both) + both) = 10 :=
by
  sorry

end science_club_members_neither_l133_133999


namespace smallest_n_l133_133576

open Nat

noncomputable def floor_sum_condition (n : ℕ) : ℕ :=
  (0 to n).sum (λ k => k / 15)

theorem smallest_n (n : ℕ) (h : floor_sum_condition n > 2011) : n = 253 :=
  sorry

end smallest_n_l133_133576


namespace monthly_avg_growth_rate_25_max_avg_tourists_next_10_days_l133_133724

-- Definition of given conditions regarding tourists count in February and April
def tourists_in_february : ℕ := 16000
def tourists_in_april : ℕ := 25000

-- Theorem 1: Monthly average growth rate of tourists from February to April is 25%.
theorem monthly_avg_growth_rate_25 :
  (tourists_in_april : ℝ) = tourists_in_february * (1 + 0.25)^2 :=
sorry

-- Definition of given conditions for tourists count from May 1st to May 21st
def tourists_may_1_to_21 : ℕ := 21250
def max_total_tourists_may : ℕ := 31250 -- Expressed in thousands as 31.25 in millions

-- Theorem 2: Maximum average number of tourists per day in the next 10 days of May.
theorem max_avg_tourists_next_10_days :
  ∀ (a : ℝ), tourists_may_1_to_21 + 10 * a ≤ max_total_tourists_may →
  a ≤ 10000 :=
sorry

end monthly_avg_growth_rate_25_max_avg_tourists_next_10_days_l133_133724


namespace area_ratio_ellipse_l133_133083

noncomputable def ellipse := { x // (x.1^2 / 3 + x.2^2 = 1) }
def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))
def intersects (line : ℝ × ℝ) (C : ℝ × ℝ → Prop) := ∃ (x y : ℝ), (C (x, y)) ∧ (y = x + line.2)

theorem area_ratio_ellipse 
  (a b : ℝ) (h_ellipse : a^2 = 3 ∧ b^2 = 1) (m : ℝ) 
  (h_foci : foci a b = ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))) 
  (h_condition : ∀ (x : ℝ), intersects (x, m) (λ pt, pt.1^2 / 3 + pt.2^2 = 1) →
     |((x - real.sqrt (a^2 - b^2)) + m)| = 2|((x + real.sqrt (a^2 - b^2)) + m)|) :
  m = -real.sqrt(2) / 3 := sorry

end area_ratio_ellipse_l133_133083


namespace one_inch_cubes_with_red_paint_at_least_2_faces_l133_133292

-- Define the original 3-inch cube painted red on all faces
def original_cube_faces_painted : ℕ := 6
def original_cube_side_length : ℕ := 3

-- Define the conditions of the problem
def num_corners_with_3_faces_painted (cube_side: ℕ) : ℕ := 8
def num_edges_with_2_faces_painted (cube_side: ℕ) : ℕ := 12

-- Define the number of small cubes with at least two red faces
def num_small_cubes_with_red_paint_at_least_2_faces (side: ℕ) : ℕ :=
  num_corners_with_3_faces_painted(side) + num_edges_with_2_faces_painted(side)

-- Define the theorem to be proved
theorem one_inch_cubes_with_red_paint_at_least_2_faces :
  num_small_cubes_with_red_paint_at_least_2_faces(original_cube_side_length) = 20 :=
by
  sorry

end one_inch_cubes_with_red_paint_at_least_2_faces_l133_133292


namespace angle_is_90_degrees_l133_133577

noncomputable def a : ℝ^3 := ![3, -1, -4]
noncomputable def b : ℝ^3 := ![sqrt 3, 5, -2]
noncomputable def c : ℝ^3 := ![7, -3, 6]

noncomputable def angle_between (u v : ℝ^3) :=
  let dot_product := u ⬝ v
  let norm_u := u.norm
  let norm_v := v.norm
  real.arccos (dot_product / (norm_u * norm_v))

noncomputable def vector_calculation : ℝ^3 :=
  let dot_ac := a ⬝ c
  let dot_ab := a ⬝ b
  (dot_ac • b) - (dot_ab • c)

theorem angle_is_90_degrees : angle_between a vector_calculation = π / 2 := by
  sorry

end angle_is_90_degrees_l133_133577


namespace range_of_x_l133_133950

theorem range_of_x (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
by {
  sorry
}

end range_of_x_l133_133950


namespace find_b_l133_133504

-- Defining the input vectors and the conditions as specified
def a : ℝ × ℝ := (1, -2)

-- Defining the condition for parallel vectors
def is_parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2 

-- Distance function
def norm (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Defining the condition involving norms
def condition (b : ℝ × ℝ) : Prop := 
  norm (1 + b.1, -2 + b.2) < norm a

-- The theorem to prove
theorem find_b : 
  ∃ b : ℝ × ℝ, is_parallel a b ∧ condition b ∧ b = (-1, 2) := 
  by 
  -- Completing the proof is not required
  sorry

end find_b_l133_133504


namespace correct_propositions_l133_133623

theorem correct_propositions :
  (∀ (P : Type) (a b c d : P), parallelogram a b c d →
    (∃ e f, intersection (diagonal a c) (diagonal b d) e ∧ line_through e f → 
      area (triangle a b e) = area (triangle c d e))) ∧
  (∀ O A B C : Point, circle O →
    central_angle O A B = 70° → 
    ¬(point_on_circle O C ∧ ¬(C = A ∨ C = B) →
      (inscribed_angle A C B = 35° ∨ inscribed_angle A C B = 145°))) ∧
  (¬∀ (n : ℕ), regular_polygon n → centrally_symmetric n) ∧
  (∀ (T : Type) a b c : Triangle, midline T a b = (side_length b c) / 2 → right_triangle T) :=
sorry

end correct_propositions_l133_133623


namespace count_valid_pairs_l133_133187

open Finset
open Function

def is_valid_pair (A B : Finset ℕ) :=
  A ∪ B = {1,2,3,4,5,6,7,8,9,10,11,12} ∧
  A ∩ B = ∅ ∧
  A.card ∉ A ∧
  B.card ∉ B

theorem count_valid_pairs : ∃ N, N = 4094 ∧
  ∃ (A B : Finset ℕ), is_valid_pair A B := 
begin
  have hN : ∃ N, N = 4094 := ⟨4094, rfl⟩,
  sorry
end

end count_valid_pairs_l133_133187


namespace total_combined_monthly_earnings_experienced_sailors_l133_133373

theorem total_combined_monthly_earnings_experienced_sailors :
  ∀ (total_sailors : ℕ) (inexperienced_sailors : ℕ) (hourly_wage_inexperienced : ℕ) 
    (work_hours_per_week : ℕ) (weeks_per_month : ℕ),
  total_sailors = 17 →
  inexperienced_sailors = 5 →
  hourly_wage_inexperienced = 10 →
  work_hours_per_week = 60 →
  weeks_per_month = 4 →
  let experienced_sailors := total_sailors - inexperienced_sailors in
  let hourly_wage_experienced := hourly_wage_inexperienced + (hourly_wage_inexperienced / 5) in
  let weekly_earnings_experienced := hourly_wage_experienced * work_hours_per_week in
  let total_weekly_earnings_experienced := experienced_sailors * weekly_earnings_experienced in
  let total_monthly_earnings_experienced := total_weekly_earnings_experienced * weeks_per_month in
  total_monthly_earnings_experienced = 34560 :=
by
  intros total_sailors inexperienced_sailors hourly_wage_inexperienced work_hours_per_week weeks_per_month
  assume h1 : total_sailors = 17,
         h2 : inexperienced_sailors = 5,
         h3 : hourly_wage_inexperienced = 10,
         h4 : work_hours_per_week = 60,
         h5 : weeks_per_month = 4
  let experienced_sailors := total_sailors - inexperienced_sailors
  let hourly_wage_experienced := hourly_wage_inexperienced + (hourly_wage_inexperienced / 5)
  let weekly_earnings_experienced := hourly_wage_experienced * work_hours_per_week
  let total_weekly_earnings_experienced := experienced_sailors * weekly_earnings_experienced
  let total_monthly_earnings_experienced := total_weekly_earnings_experienced * weeks_per_month
  show total_monthly_earnings_experienced = 34560 from sorry

end total_combined_monthly_earnings_experienced_sailors_l133_133373


namespace cannot_meet_on_street_l133_133648

noncomputable def same_speed (car1 car2 : ℕ → ℝ) (t : ℕ) : Prop :=
  car1 t = car2 t

noncomputable def equilateral_triangles (t : ℕ) : Prop :=
  ∀ car, car t = car (t - 1) + 1  -- Simplification of moves on an equilateral triangle grid

noncomputable def turn120 (t : ℕ) : Prop :=
  ∀ car, car t = car (t - 1) + 1  ∨ car t = car (t - 1) - 1 -- Simplified turn by 120 degrees at intersections

noncomputable def move_conditions (car1 car2 : ℕ → ℝ) : Prop :=
  same_speed car1 car2 ∧ equilateral_triangles 0 ∧ turn120 0 

theorem cannot_meet_on_street (car1 car2 : ℕ → ℝ) (t : ℕ) (h : move_conditions car1 car2) :
  (∀ t, car1 t ≠ car2 t) :=
sorry

end cannot_meet_on_street_l133_133648


namespace mathematician_correctness_l133_133250

theorem mathematician_correctness (box1_white1 box1_total1 box2_white1 box2_total1 : ℕ)
                                  (prob1 prob2 : ℚ) :
  (box1_white1 = 4 ∧ box1_total1 = 7 ∧ box2_white1 = 3 ∧ box2_total1 = 5 ∧ prob1 = (4:ℚ) / 7 ∧ prob2 = (3:ℚ) / 5) →
  (box1_white2 = 8 ∧ box1_total2 = 14 ∧ box2_white2 = 3 ∧ box2_total2 = 5 ∧ prob1 = (8:ℚ) / 14 ∧ prob2 = (3:ℚ) / 5) →
  (prob1 < (7:ℚ) / 12 ∧ prob2 > (11:ℚ)/19) →
  ¬((19:ℚ)/35 > (4:ℚ)/7 ∧ (19:ℚ)/35 < (3:ℚ)/5) :=
by
  sorry

end mathematician_correctness_l133_133250


namespace math_problem_l133_133359

noncomputable def circle_equation (D E F : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0

def passes_through (x1 y1 x2 y2 : ℝ) (D E F : ℝ) : Prop :=
  (x1^2 + y1^2 + D * x1 + E * y1 + F = 0) ∧
  (x2^2 + y2^2 + D * x2 + E * y2 + F = 0)

def on_line (x y : ℝ) : Prop :=
  2 * x - y - 4 = 0

def chord_length (D E : ℝ) (c_dist : ℝ) : ℝ :=
  2 * real.sqrt (c_dist^2 - (D / 2)^2 - (E / 2)^2 + 4)

open Real

theorem math_problem :
  passes_through 0 2 2 0 (-8) (-8) 12 ∧ on_line 4 4 ∧ circle_equation (-8) (-8) 12 ∧ chord_length (-8) (-8) 4 = 4 :=
by
  sorry

end math_problem_l133_133359


namespace find_scalars_eq_zero_l133_133480

theorem find_scalars_eq_zero (a b : ℝ × ℝ) (λ μ : ℝ)
  (h₀ : a = (1, 3))
  (h₁ : b = (1, -2))
  (h₂ : λ * a.1 + μ * b.1 = 0)
  (h₃ : λ * a.2 + μ * b.2 = 0) : λ = 0 ∧ μ = 0 :=
by
  -- proof steps would go here
  sorry

end find_scalars_eq_zero_l133_133480


namespace gcd_of_all_three_digit_palindromes_is_one_l133_133692

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define a function to calculate the gcd of a list of numbers
def gcd_list (l : List ℕ) : ℕ :=
  l.foldr Nat.gcd 0

-- The main theorem that needs to be proven
theorem gcd_of_all_three_digit_palindromes_is_one :
  gcd_list (List.filter is_palindrome {n | 100 ≤ n ∧ n ≤ 999}.toList) = 1 :=
by
  sorry

end gcd_of_all_three_digit_palindromes_is_one_l133_133692


namespace gcd_three_digit_palindromes_l133_133708

theorem gcd_three_digit_palindromes : 
  GCD (set.image (λ (p : ℕ × ℕ), 101 * p.1 + 10 * p.2) 
    ({a | a ≠ 0 ∧ a < 10} × {b | b < 10})) = 1 := 
by
  sorry

end gcd_three_digit_palindromes_l133_133708


namespace handrail_length_approx_l133_133769

-- Define the given conditions
def turn_degrees : ℝ := 225
def rise_feet : ℝ := 12
def radius_feet : ℝ := 4

-- Formalize the proof statement
theorem handrail_length_approx :
  let arc_length := (turn_degrees / 360) * (2 * π * radius_feet),
      handrail_length := Real.sqrt (rise_feet^2 + arc_length^2)
  in Real.floor (handrail_length * 10) / 10 = 19.8 :=
by
  sorry

end handrail_length_approx_l133_133769


namespace total_books_l133_133640

-- Define the sequence conditions
def arithmetic_sequence (a0 d n : ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ k : ℕ, k < n → a k = a0 + k * d

-- Specify the problem conditions
def problem_conditions : Prop :=
  ∃ (a : ℕ → ℤ) (n : ℕ),
  arithmetic_sequence 35 (-3) n a ∧
  a (n - 1) = 1

-- Prove the total number of books given the conditions
theorem total_books (a : ℕ → ℤ) (n : ℕ) (h₁ : arithmetic_sequence 35 (-3) n a) (h₂ : a (n - 1) = 1) : 
 ∑i in finset.range n, a i = 222 :=
by {
  sorry
}

end total_books_l133_133640


namespace gcd_three_digit_palindromes_l133_133700

open Nat

theorem gcd_three_digit_palindromes :
  (∀ a b : ℕ, a ≠ 0 → a < 10 → b < 10 → True) ∧
  let S := {n | ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b} in
  S.Gcd = 1 := by
  sorry

end gcd_three_digit_palindromes_l133_133700


namespace points_line_slope_intercept_l133_133628

theorem points_line_slope_intercept (m b : ℝ) : 
  (∀ x : ℝ, y = m * x + b) 
  (line_through (1, 2) (4, 11)) 
  (m + b = 2) := sorry

end points_line_slope_intercept_l133_133628


namespace greatest_common_factor_of_three_digit_palindromes_l133_133674

def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b

def gcf (a b : ℕ) : ℕ := 
  if a = 0 then b else gcf (b % a) a

theorem greatest_common_factor_of_three_digit_palindromes : 
  ∃ g, (∀ n, is_palindrome n → g ∣ n) ∧ (∀ d, (∀ n, is_palindrome n → d ∣ n) → d ∣ g) :=
by
  use 101
  sorry

end greatest_common_factor_of_three_digit_palindromes_l133_133674


namespace max_number_of_triangles_l133_133777

-- Define the permissible lengths
def lengths : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define a function that checks if three given lengths can form a triangle
def forms_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define a set of all possible triangles formed by the given lengths
def all_triangles : Set (ℕ × ℕ × ℕ) :=
  { (a, b, c) | a ∈ lengths ∧ b ∈ lengths ∧ c ∈ lengths ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ forms_triangle a b c }

-- Define the problem statement
theorem max_number_of_triangles : ∃ n, n = 7 ∧ card all_triangles = n :=
by
  sorry

end max_number_of_triangles_l133_133777


namespace hexagon_indistinguishable_under_rotation_l133_133139

noncomputable def hexagon_colorings : Nat :=
  let G := CyclicGroup 6
  let fixed_points : Finset Nat := {64, 2, 4, 8, 4, 2}
  (1 / 6 * (64 + 2 + 4 + 8 + 4 + 2))

theorem hexagon_indistinguishable_under_rotation :
  hexagon_colorings = 14 :=
by
  have h : 1 / 6 * (64 + 2 + 4 + 8 + 4 + 2) = 14 :=
    sorry
  exact h

end hexagon_indistinguishable_under_rotation_l133_133139


namespace initial_persons_count_l133_133620

open Real

def average_weight_increase (n : ℕ) (increase_per_person : ℝ) : ℝ :=
  increase_per_person * n

def weight_difference (new_weight old_weight : ℝ) : ℝ :=
  new_weight - old_weight

theorem initial_persons_count :
  ∀ (n : ℕ),
  average_weight_increase n 2.5 = weight_difference 95 75 → n = 8 :=
by
  intro n h
  sorry

end initial_persons_count_l133_133620


namespace find_m_l133_133066

noncomputable def ellipse : set (ℝ × ℝ) := { p | (p.1)^2 / 3 + (p.2)^2 = 1 }
def f1 : ℝ × ℝ := (- real.sqrt 2, 0)
def f2 : ℝ × ℝ := (real.sqrt 2, 0)

def line (m : ℝ) : set (ℝ × ℝ) := { p | p.2 = p.1 + m }

theorem find_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line m ∧ B ∈ line m ∧
   2 * (abs ((f1.1 * (A.2 - B.2) + A.1 * (B.2 - f1.2) + B.1 * (f1.2 - A.2)) / 2))
 = abs ((f2.1 * (A.2 - B.2) + A.1 * (B.2 - f2.2) + B.1 * (f2.2 - A.2)) / 2)) →
  m = - real.sqrt 2 / 3 :=
sorry

end find_m_l133_133066


namespace determine_slope_l3_l133_133210

structure Point :=
  (x : ℚ)
  (y : ℚ)

def Line (p : Point) (m : ℚ) : Point → Prop :=
  λ q, q.y = m * (q.x - p.x) + p.y

def LineEquation1 (p : Point) : Prop := 
  5 * p.x + 4 * p.y = 2

def LineEquation2 (p : Point) : Prop := 
  p.y = 2

def triangle_area (A B C : Point) : ℚ :=
  abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2)

theorem determine_slope_l3 :
  ∃ m : ℚ, 
  let l1 := {p : Point // LineEquation1 p} in
  let l2 := {p : Point // LineEquation2 p} in
  let A := ⟨-2, -3⟩ in
  let B := ⟨-6/5, 2⟩ in
  (∃ C : Point, 
    C ∈ {p // LineEquation2 p} ∧ 
    Line A m C ∧ 
    triangle_area A B C = 4) ∧ 
  m = 25 / 12 :=
begin
  have A : Point := { x := -2, y := -3 },
  have B : Point := { x := -6/5, y := 2 },
  sorry
end

end determine_slope_l3_l133_133210


namespace find_m_l133_133044

def ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + y^2 = 1

def focus1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def focus2 : ℝ × ℝ := (Real.sqrt 2, 0)

def line (x y m : ℝ) : Prop :=
  y = x + m

def intersects (m : ℝ) (x y : ℝ) : Prop :=
  ellipse x y ∧ line x y m

def area_relation (m : ℝ) : Prop :=
  let x := -m -- This might be simplified as it's used to calculate distances
  let dist1 := Real.abs (focus1.fst + m)
  let dist2 := Real.abs (focus2.fst + m)
  dist1 = 2 * dist2

theorem find_m :
  area_relation (-Real.sqrt 2 / 3) :=
by
  sorry

end find_m_l133_133044


namespace students_liking_both_l133_133930

theorem students_liking_both (total_students : ℕ) (students_apple_pie : ℕ) (students_chocolate_cake : ℕ) (students_neither : ℕ) :
  total_students = 40 →
  students_apple_pie = 18 →
  students_chocolate_cake = 15 →
  students_neither = 12 →
  let students_at_least_one := total_students - students_neither in
  students_at_least_one = 28 →
  let students_both := students_apple_pie + students_chocolate_cake - students_at_least_one in
  students_both = 5 :=
by
  intros ht ha hc hn h1
  simp only [] at *
  rw [ht, ha, hc, hn] at h1
  rw ← h1
  refl

end students_liking_both_l133_133930


namespace sum_difference_l133_133399

theorem sum_difference : 
  let even_sum := (1500 / 2) * (2 + (2 + (1500 - 1) * 2)),
      multiple_of_3_sum := (1500 / 2) * (3 + (3 + (1500 - 1) * 3))
  in multiple_of_3_sum - even_sum = 1125750 :=
by
  sorry

end sum_difference_l133_133399


namespace find_m_l133_133032

variables (a b : ℝ) (x y m : ℝ) (A B F1 F2 : ℝ × ℝ)

-- Given conditions
def ellipse := (x^2 / 3) + y^2 = 1
def line := y = x + m
def foci_F1 := F1 = (-sqrt 2, 0)
def foci_F2 := F2 = (sqrt 2, 0)
def area_F1_AB := λ A B : ℝ × ℝ, 2 * |det(F1, A, B)| = |det(F2, A, B)|

-- The proof problem statement
theorem find_m : 
  ellipse ∧ 
  line ∧ 
  foci_F1 ∧ 
  foci_F2 ∧ 
  (area_F1_AB A B) →
  m = -sqrt(2) / 3 := sorry

end find_m_l133_133032


namespace quadratic_has_real_roots_for_all_K_l133_133807

theorem quadratic_has_real_roots_for_all_K (K : ℝ) : 
  let a := K^3,
      b := -(4 * K^3 + 1),
      c := 3 * K^3,
      Δ := b^2 - 4 * a * c in
  Δ >= 0 :=
by
  let a := K^3
  let b := -(4 * K^3 + 1)
  let c := 3 * K^3
  let Δ := b^2 - 4 * a * c
  have h : Δ = 4 * K^6 + 8 * K^3 + 1
  sorry

end quadratic_has_real_roots_for_all_K_l133_133807


namespace find_m_l133_133045

def ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + y^2 = 1

def focus1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def focus2 : ℝ × ℝ := (Real.sqrt 2, 0)

def line (x y m : ℝ) : Prop :=
  y = x + m

def intersects (m : ℝ) (x y : ℝ) : Prop :=
  ellipse x y ∧ line x y m

def area_relation (m : ℝ) : Prop :=
  let x := -m -- This might be simplified as it's used to calculate distances
  let dist1 := Real.abs (focus1.fst + m)
  let dist2 := Real.abs (focus2.fst + m)
  dist1 = 2 * dist2

theorem find_m :
  area_relation (-Real.sqrt 2 / 3) :=
by
  sorry

end find_m_l133_133045


namespace range_of_a_l133_133107

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then -x^2 - 1 else Real.log (x + 1)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x ≤ a * x) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l133_133107


namespace average_loss_per_loot_box_l133_133565

theorem average_loss_per_loot_box
  (cost_per_loot_box : ℝ := 5)
  (value_standard_item : ℝ := 3.5)
  (probability_rare_item_A : ℝ := 0.05)
  (value_rare_item_A : ℝ := 10)
  (probability_rare_item_B : ℝ := 0.03)
  (value_rare_item_B : ℝ := 15)
  (probability_rare_item_C : ℝ := 0.02)
  (value_rare_item_C : ℝ := 20) 
  : (cost_per_loot_box 
      - (0.90 * value_standard_item 
      + probability_rare_item_A * value_rare_item_A 
      + probability_rare_item_B * value_rare_item_B 
      + probability_rare_item_C * value_rare_item_C)) = 0.50 := by 
  sorry

end average_loss_per_loot_box_l133_133565


namespace prime_power_value_l133_133876

theorem prime_power_value (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h1 : Nat.Prime (7 * p + q)) (h2 : Nat.Prime (p * q + 11)) : 
  p ^ q = 8 ∨ p ^ q = 9 := 
sorry

end prime_power_value_l133_133876


namespace find_a_l133_133924

noncomputable def f (a x : ℝ) : ℝ := Real.log (a * x + 1)

theorem find_a {a : ℝ} (h : (deriv (f a) 0 = 1)) : a = 1 :=
by
  -- Proof goes here
  sorry

end find_a_l133_133924


namespace number_of_teachers_l133_133296

theorem number_of_teachers
  (students : ℕ) (lessons_per_student_per_day : ℕ) (lessons_per_teacher_per_day : ℕ) (students_per_class : ℕ)
  (h1 : students = 1200)
  (h2 : lessons_per_student_per_day = 5)
  (h3 : lessons_per_teacher_per_day = 4)
  (h4 : students_per_class = 30) :
  ∃ teachers : ℕ, teachers = 50 :=
by
  have total_lessons : ℕ := lessons_per_student_per_day * students
  have classes : ℕ := total_lessons / students_per_class
  have teachers : ℕ := classes / lessons_per_teacher_per_day
  use teachers
  sorry

end number_of_teachers_l133_133296


namespace quadrilateral_midpoint_intersection_l133_133762

-- Definitions of points and properties
variables {A B C D : Type} [add_comm_group A] [add_comm_group B] [add_comm_group C] [add_comm_group D]

-- Definitions for O and midpoints using given conditions such as intersection with circle and midpoint properties
variables (O : A) (midX midY : A) 
variables (pA pB pC pD : A) 

-- Assuming circle circumscription property
def circle_circumscribed (O : A) (A B C D : A) : Prop :=
  -- property of a circle circumscribed around A, B, C, D
  sorry

-- Assuming quadrilateral is not a parallelogram
def not_parallelogram (A B C D : A) : Prop :=
  -- property that quadrilateral ABCD is not a parallelogram
  sorry

-- The proof statement for the given condition
theorem quadrilateral_midpoint_intersection
  (h_circumscribed : circle_circumscribed O pA pB pC pD)
  (h_not_parallelogram : not_parallelogram pA pB pC pD)
  (h_midpoints : midX = (pA + pB) / 2 ∧ midY = (pC + pD) / 2) :
  (midX = midY ↔ OA * OC = OB * OD) :=
  sorry

end quadrilateral_midpoint_intersection_l133_133762


namespace graph_shift_l133_133631

def g (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2) ^ 2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0

theorem graph_shift (x : ℝ) : g (x - 3) = if 0 ≤ x - 3 ∧ x - 3 ≤ 3 then g (x - 3) else 0 :=
  sorry

end graph_shift_l133_133631


namespace exists_line_parallel_to_gamma_l133_133505

variable {α β γ : Plane}

-- Definitions of necessary conditions
def is_perpendicular_to (P Q : Plane) : Prop := ∀ l ∈ P, ∀ m ∈ Q, l ⊥ m
def intersects_but_not_perpendicular (P Q : Plane) : Prop := (∃ l ∈ P, l ∈ Q) ∧ ¬is_perpendicular_to P Q
def is_parallel_to (l : Line) (P : Plane) : Prop := ∃ m ∈ P, l ∥ m

-- Given planes α, β, γ with the specified conditions
def problem_conditions (α β γ : Plane) : Prop :=
  is_perpendicular_to β γ ∧ intersects_but_not_perpendicular α γ

-- The proof goal
theorem exists_line_parallel_to_gamma (α β γ : Plane) (h : problem_conditions α β γ) :
  ∃ a ∈ α, is_parallel_to a γ := sorry

end exists_line_parallel_to_gamma_l133_133505


namespace sum_inequality_l133_133234

variable {n : ℕ}
variables {a b : Fin n → ℝ}

-- Conditions
def cond1 (h : ∀ i j : Fin n, (i ≤ j) → a i ≥ a j) := ∀ i j : Fin n, (i ≤ j) → a i ≥ a j
def cond2 (h : ∀ i : Fin n, 0 < a i) := ∀ i : Fin n, 0 < a i
def cond3 (h : b 0 ≥ a 0) := b 0 ≥ a 0
def cond4 (h : ∀ k : Fin n, (∏ i in (Finset.range k.succ).to_finset, b i) ≥ (∏ i in (Finset.range k.succ).to_finset, a i)) := ∀ k : Fin n, (∏ i in (Finset.range k.succ).to_finset, b i) ≥ (∏ i in (Finset.range k.succ).to_finset, a i)

-- The inequality to be proved
theorem sum_inequality (h₁ : cond1 a) (h₂ : cond2 a)
  (h₃ : cond3 b) (h₄ : cond4 a b) : 
  (∑ i, b i) ≥ (∑ i, a i) :=
sorry

end sum_inequality_l133_133234


namespace elevator_total_period_l133_133627

theorem elevator_total_period (rate : ℕ := (4 : ℕ)) 
                            (stories : ℕ := 11) 
                            (stop_time : ℕ := 2) 
                            (total_trips : ℝ := 34.285714285714285) : 
                            (total_time_in_hours : ℝ) :=
by {
  let intervals_per_trip := 20,
  let travel_time := (intervals_per_trip / rate : ℝ),
  let total_time_per_trip := travel_time + stop_time := (7 : ℝ),
  let total_time := total_trips * total_time_per_trip,
  let total_time_in_hours := total_time / 60,
  have proof : total_time_in_hours = 4 := rfl,
  exact proof,
}

end elevator_total_period_l133_133627


namespace steven_has_72_shirts_l133_133605

def brian_shirts : ℕ := 3
def andrew_shirts (brian : ℕ) : ℕ := 6 * brian
def steven_shirts (andrew : ℕ) : ℕ := 4 * andrew

theorem steven_has_72_shirts : steven_shirts (andrew_shirts brian_shirts) = 72 := 
by 
  -- We add "sorry" here to indicate that the proof is omitted
  sorry

end steven_has_72_shirts_l133_133605


namespace no_solution_sin_sum_eq_sin_product_l133_133435

theorem no_solution_sin_sum_eq_sin_product :
  ∀ x y : ℝ, (0 < x ∧ x < π / 2) ∧ (0 < y ∧ y < π / 2) → ¬ (sin x + sin y = sin (x * y)) := 
by
  intros x y h
  sorry

end no_solution_sin_sum_eq_sin_product_l133_133435


namespace pufferfish_count_l133_133299

theorem pufferfish_count (s p : ℕ) (h1 : s = 5 * p) (h2 : s + p = 90) : p = 15 :=
by
  sorry

end pufferfish_count_l133_133299


namespace constant_rate_of_train_B_l133_133308

-- Defining the given conditions
def distance_between_stations : ℝ := 350
def speed_of_train_A : ℝ := 40
def distance_when_trains_pass : ℝ := 200

-- Defining what we know
def time_for_train_A_to_pass : ℝ := distance_when_trains_pass / speed_of_train_A
def distance_travelled_by_train_B : ℝ := distance_between_stations - distance_when_trains_pass

-- The theorem we want to prove
theorem constant_rate_of_train_B : 
  time_for_train_A_to_pass = 5 ∧ distance_travelled_by_train_B = 150 → distance_travelled_by_train_B / time_for_train_A_to_pass = 30 := by
  sorry

end constant_rate_of_train_B_l133_133308


namespace minimum_value_f_l133_133861

noncomputable def f (a θ : Real) : Real :=
  Real.sin θ ^ 3 + 4 / (3 * a * (Real.sin θ) ^ 2 - a ^ 3)

theorem minimum_value_f (a θ : Real) 
  (h1 : 0 < a) 
  (h2 : a < Real.sqrt 3 * Real.sin θ) 
  (h3 : π/4 ≤ θ ∧ θ ≤ 5*π/6) : 
  ∃ a θ, ∀ a θ, f a θ ≥ 3 :=
sorry

end minimum_value_f_l133_133861


namespace monotone_decreasing_f_l133_133105

def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≥ 1 then (2*a - 1)*x + a
else log a x

theorem monotone_decreasing_f (a : ℝ) : (∀ x y, 0 < x ∧ x < y ∧ x < 1 ∧ y ≤ 1 → f a y ≤ f a x) ∧ 
  (∀ x y, x ≥ 1 ∧ y > x → f a y ≤ f a x) ∧ 
  (f a 1 ≤ f a 1) → a ∈ set.Icc 0 (1/3) :=
by
  sorry

end monotone_decreasing_f_l133_133105


namespace derivative_at_neg_one_l133_133909

variable (a b : ℝ)

-- Define the function f(x)
def f (x : ℝ) : ℝ := a * x^4 + b * x^2 + 6

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Given condition f'(1) = 2
axiom h : f' a b 1 = 2

-- Statement to prove f'(-1) = -2
theorem derivative_at_neg_one : f' a b (-1) = -2 :=
by 
  sorry

end derivative_at_neg_one_l133_133909


namespace number_of_correct_propositions_is_0_l133_133129

open Set

variables {m n : Line} {α β : Plane}

-- Given conditions
def condition1 : Prop := m ⊥ α
def condition2 : Prop := n ⊆ β

-- Propositions to check
def proposition1 : Prop := (α ∥ β) → (m ⊥ n)
def proposition2 : Prop := (m ⊥ n) → (α ∥ β)
def proposition3 : Prop := (m ∥ n) → (α ⊥ β)
def proposition4 : Prop := (α ⊥ β) → (m ∥ n)

-- The main statement to prove
theorem number_of_correct_propositions_is_0 :
  condition1 → condition2 →
  (¬ proposition1 ∧ ¬ proposition2 ∧ ¬ proposition3 ∧ ¬ proposition4) :=
by
  intros h1 h2
  split
  . intro h
    have : False := sorry
    contradiction
  split
  . intro h
    have : False := sorry
    contradiction
  split
  . intro h
    have : False := sorry
    contradiction
  . intro h
    have : False := sorry
    contradiction

end number_of_correct_propositions_is_0_l133_133129


namespace rectangle_perimeter_ratio_l133_133770

noncomputable def paper_folded_rectangle (side_length : ℝ) : (ℝ × ℝ) :=
  (side_length / 2, side_length)

noncomputable def cut_folded_paper (rectangle : (ℝ × ℝ)) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let (height, width) := rectangle
  in let small_width := width / 3
     let large_width := small_width * 2
     ((height, small_width), (height, large_width))

noncomputable def perimeter (rectangle: (ℝ × ℝ)) : ℝ :=
  let (height, width) := rectangle
  in 2 * (height + width)

theorem rectangle_perimeter_ratio
  (side_length : ℝ) (h : side_length = 10) :
  let folded_rectangle := paper_folded_rectangle side_length,
      rectangles := cut_folded_paper folded_rectangle in
  (perimeter rectangles.1) / (perimeter rectangles.2) = 5 / 7 :=
by
  sorry

end rectangle_perimeter_ratio_l133_133770


namespace sock_pair_count_l133_133907

theorem sock_pair_count (white_socks brown_socks blue_socks : ℕ) 
  (white_socks = 4) (brown_socks = 4) (blue_socks = 2) : 
  (nat.choose white_socks 2) + 
  (nat.choose brown_socks 2) + 
  (nat.choose blue_socks 2) = 13 := 
  by 
    sorry

end sock_pair_count_l133_133907


namespace total_distance_is_1095_l133_133568

noncomputable def totalDistanceCovered : ℕ :=
  let running_first_3_months := 3 * 3 * 10
  let running_next_3_months := 3 * 3 * 20
  let running_last_6_months := 3 * 6 * 30
  let total_running := running_first_3_months + running_next_3_months + running_last_6_months

  let swimming_first_6_months := 3 * 6 * 5
  let total_swimming := swimming_first_6_months

  let total_hiking := 13 * 15

  total_running + total_swimming + total_hiking

theorem total_distance_is_1095 : totalDistanceCovered = 1095 := by
  sorry

end total_distance_is_1095_l133_133568


namespace willy_crayons_difference_l133_133723

def willy : Int := 5092
def lucy : Int := 3971
def jake : Int := 2435

theorem willy_crayons_difference : willy - (lucy + jake) = -1314 := by
  sorry

end willy_crayons_difference_l133_133723


namespace domain_log_floor_l133_133195

def floor (x : ℝ) : ℤ := Int.floor x

theorem domain_log_floor (x : ℝ) : 
  1 ≤ x ↔ ∃ y : ℝ, y = Real.log (floor x) := 
by
  sorry

end domain_log_floor_l133_133195


namespace ellipse_area_condition_l133_133071

open Real

theorem ellipse_area_condition (m : ℝ) :
  (∀ x y : ℝ, ((x^2)/3 + y^2 = 1) → (y = x + m) → 
  let F₁ := (-sqrt 2, 0)
      F₂ := (sqrt 2, 0)
  in distance F₁ (x, y) = 2 * distance F₂ (x, y)) → 
  m = -sqrt 2 / 3 :=
by
  sorry

end ellipse_area_condition_l133_133071


namespace first_term_of_geometric_series_l133_133781

theorem first_term_of_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) :
  r = 1 / 4 → S = 20 → S = a / (1 - r) → a = 15 :=
by
  intro hr hS hsum
  sorry

end first_term_of_geometric_series_l133_133781


namespace area_ratio_ellipse_l133_133082

noncomputable def ellipse := { x // (x.1^2 / 3 + x.2^2 = 1) }
def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))
def intersects (line : ℝ × ℝ) (C : ℝ × ℝ → Prop) := ∃ (x y : ℝ), (C (x, y)) ∧ (y = x + line.2)

theorem area_ratio_ellipse 
  (a b : ℝ) (h_ellipse : a^2 = 3 ∧ b^2 = 1) (m : ℝ) 
  (h_foci : foci a b = ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))) 
  (h_condition : ∀ (x : ℝ), intersects (x, m) (λ pt, pt.1^2 / 3 + pt.2^2 = 1) →
     |((x - real.sqrt (a^2 - b^2)) + m)| = 2|((x + real.sqrt (a^2 - b^2)) + m)|) :
  m = -real.sqrt(2) / 3 := sorry

end area_ratio_ellipse_l133_133082


namespace sqrt_expression_l133_133409

theorem sqrt_expression :
  (Real.sqrt (2 ^ 4 * 3 ^ 6 * 5 ^ 2)) = 540 := sorry

end sqrt_expression_l133_133409


namespace find_c_plus_d_l133_133455

-- Define the piecewise function f
def f (x : ℝ) (c d : ℝ) : ℝ :=
if x < 3 then c * x + d else 7 - 2 * x

-- Define the property f(f(x)) = x
def property (f : ℝ → ℝ) : Prop :=
∀ x, f (f x) = x

-- Statement to show c + d = 3 given the conditions
theorem find_c_plus_d (c d : ℝ) (h : property (λ x, f x c d)) : c + d = 3 :=
sorry

end find_c_plus_d_l133_133455


namespace combination_value_l133_133456

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

noncomputable def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem combination_value {n : ℕ} (h : n * (n - 1) = 90) : combination (n + 2) n = 66 :=
begin
  sorry
end

end combination_value_l133_133456


namespace stock_yield_correct_l133_133377

-- Define the stock price and the quoted yield
def StockPrice : ℝ := 225
def QuotedYield : ℝ := 0.08

-- Define the dividend payment based on the quoted yield and stock price
def DividendPayment : ℝ := (QuotedYield * StockPrice)

-- Define the function to calculate the yield from dividend payment and stock price
def calculateYield (dividend : ℝ) (price : ℝ) : ℝ := (dividend / price) * 100

-- The main theorem we need to prove: the stock's yield matches the quoted yield
theorem stock_yield_correct : calculateYield DividendPayment StockPrice = QuotedYield * 100 := by
  sorry

end stock_yield_correct_l133_133377


namespace polly_breakfast_minutes_l133_133219
open Nat

theorem polly_breakfast_minutes (B : ℕ) 
  (lunch_minutes : ℕ)
  (dinner_4_days_minutes : ℕ)
  (dinner_3_days_minutes : ℕ)
  (total_minutes : ℕ)
  (h1 : lunch_minutes = 5 * 7)
  (h2 : dinner_4_days_minutes = 10 * 4)
  (h3 : dinner_3_days_minutes = 30 * 3)
  (h4 : total_minutes = 305) 
  (h5 : 7 * B + lunch_minutes + dinner_4_days_minutes + dinner_3_days_minutes = total_minutes) :
  B = 20 :=
by
  -- proof omitted
  sorry

end polly_breakfast_minutes_l133_133219


namespace proof_problem_l133_133860

noncomputable def theorem_problem : Prop :=
  ∀ x y : ℝ, (|x + y + 1| + sqrt (2 * x - y) = 0) → (x - y = 1 / 3)

theorem proof_problem : theorem_problem := 
  by
    sorry

end proof_problem_l133_133860


namespace solution_unique_l133_133962

theorem solution_unique (n : ℕ) (x : ℝ) (h : ∑ k in finset.range (n + 1), (k + 1) * x^(k + 1) / (1 + x^((k + 1) * 2)) = n * (n + 1) / 4) : x = 1 :=
sorry

end solution_unique_l133_133962


namespace sum_of_subset_divisible_by_n_l133_133766

def exists_subset_sum_divisible_by_n (n : ℕ) (h_n : n ≥ 3) (S : Finset ℕ) (h_card : S.card = n-1) : Prop :=
  ∃ T : Finset ℕ, T ⊆ S ∧ T.nonempty ∧ (T.sum id) % n = 0

theorem sum_of_subset_divisible_by_n
  (n : ℕ)
  (h_n : n ≥ 3)
  (S : Finset ℕ)
  (h_card : S.card = n-1)
  (h_diff : ∃ x y ∈ S, x ≠ y ∧ ¬ ( (x - y) % n = 0 )) :
  exists_subset_sum_divisible_by_n n h_n S h_card :=
sorry

end sum_of_subset_divisible_by_n_l133_133766


namespace infinite_x_not_congruent_l133_133190

theorem infinite_x_not_congruent (k : ℕ) (m : ℕ → ℕ) (a : ℕ → ℕ)
  (h_m1 : 2 ≤ m 0)
  (h_m : ∀ i, i < k - 1 → 2 * m i ≤ m (i + 1)) :
  ∃ᶠ x in Filter.at_top, ∀ i < k, x % m i ≠ a i :=
sorry

end infinite_x_not_congruent_l133_133190


namespace condition1_num_registration_methods_condition2_num_registration_methods_condition3_num_registration_methods_l133_133227

-- Definitions corresponding to each condition
def numMethods_participates_in_one_event (students events : ℕ) : ℕ :=
  events ^ students

def numMethods_event_limit_one_person (students events : ℕ) : ℕ :=
  students * (students - 1) * (students - 2)

def numMethods_person_limit_in_events (students events : ℕ) : ℕ :=
  students ^ events

-- Theorems to be proved
theorem condition1_num_registration_methods : 
  numMethods_participates_in_one_event 6 3 = 729 :=
by
  sorry

theorem condition2_num_registration_methods : 
  numMethods_event_limit_one_person 6 3 = 120 :=
by
  sorry

theorem condition3_num_registration_methods : 
  numMethods_person_limit_in_events 6 3 = 216 :=
by
  sorry

end condition1_num_registration_methods_condition2_num_registration_methods_condition3_num_registration_methods_l133_133227


namespace symmetric_points_distance_l133_133022

theorem symmetric_points_distance (A B C : ℝ × ℝ)
    (hAB : dist A B < 1) (hBC : dist B C < 1) (hAC : dist A C < 1):
    ∃ P Q : ℝ × ℝ, (P ∈ (generate_points {A, B, C} ∧ Q ∈ generate_points {A, B, C})) ∧ dist P Q > 1 :=
sorry

/-- Defining the method to generate symmetric points relative to a line segment -/
noncomputable def generate_points : set (ℝ × ℝ) → set (ℝ × ℝ)
| points := points ∪ {sym_point | ∃ (X Y : ℝ × ℝ) ∈ points, sym_point = symmetric_point X Y}

-- Placeholder for the symmetric point calculation
noncomputable def symmetric_point (X Y : ℝ × ℝ) : ℝ × ℝ :=
sorry

end symmetric_points_distance_l133_133022


namespace D_72_l133_133966

def D (n : ℕ) : ℕ :=
  -- Definition of D(n) should be provided here
  sorry

theorem D_72 : D 72 = 121 :=
  sorry

end D_72_l133_133966


namespace num_possible_1st_2nd_3rd_outcomes_l133_133382

-- Definition of participants
inductive Participant
| Abe | Bobby | Charles | Devin | Edwin | Fiona

-- Predicate that checks if a participant is in the top three.
def is_in_top_three (p : Participant) (top_three : List Participant) : Prop :=
  p ∈ top_three

-- The proof problem statement
theorem num_possible_1st_2nd_3rd_outcomes : 
  ∀ top_three : List Participant,
    (∀ p, is_in_top_three p top_three → p ≠ Participant.Fiona) →
    top_three.length = 3 →
    (finset.univ.filter (λ (r : list Participant), r.length = 3 ∧ ∀ p, p ∈ r → p ≠ Participant.Fiona)).card = 60 :=
begin
  sorry
end

end num_possible_1st_2nd_3rd_outcomes_l133_133382


namespace binary_1101001_to_decimal_l133_133417

theorem binary_1101001_to_decimal :
  let b := "1101001" 
  let n := b.foldr (λ c acc, 2 * acc + (if c = '1' then 1 else 0)) 0
  n = 105 ∧ n % 2 = 1 :=
by
  let b := "1101001" 
  let n := b.foldr (λ c acc, 2 * acc + (if c = '1' then 1 else 0)) 0
  have h₀ : n = 105, sorry
  have h₁ : n % 2 = 1, sorry
  exact ⟨h₀, h₁⟩

end binary_1101001_to_decimal_l133_133417


namespace moles_of_NH4NO3_combined_is_2_l133_133442

noncomputable def moles_of_NH4NO3_combined (n: ℕ): Prop :=
  ∀ (NH4NO3 NaOH: ℕ), NaOH = 2 → NH4NO3 + NaOH → H2O = 2 → NH4NO3 = n

theorem moles_of_NH4NO3_combined_is_2 : moles_of_NH4NO3_combined 2 :=
by {
  sorry
}

end moles_of_NH4NO3_combined_is_2_l133_133442


namespace endomorphisms_of_Z2_are_linear_functions_l133_133002

namespace GroupEndomorphism

-- Definition of an endomorphism: a homomorphism from Z² to itself
def is_endomorphism (f : ℤ × ℤ → ℤ × ℤ) : Prop :=
  ∀ a b : ℤ × ℤ, f (a + b) = f a + f b

-- Definition of the specific form of endomorphisms for Z²
def specific_endomorphism_form (u v : ℤ × ℤ) (φ : ℤ × ℤ) : ℤ × ℤ :=
  (φ.1 * u.1 + φ.2 * v.1, φ.1 * u.2 + φ.2 * v.2)

-- Main theorem:
theorem endomorphisms_of_Z2_are_linear_functions :
  ∀ φ : ℤ × ℤ → ℤ × ℤ, is_endomorphism φ →
  ∃ u v : ℤ × ℤ, φ = specific_endomorphism_form u v := by
  sorry

end GroupEndomorphism

end endomorphisms_of_Z2_are_linear_functions_l133_133002


namespace ellipse_intersection_area_condition_l133_133030

theorem ellipse_intersection_area_condition :
  let ellipse := ∀ (x y : ℝ), (x^2 / 3 + y^2 = 1)
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let line m := ∀ (x y : ℝ), (y = x + m)
  ∀ (A B : ℝ × ℝ),
    ((A.1^2 / 3 + A.2^2 = 1) ∧ (y = A.1 + m)) ∧
    ((B.1^2 / 3 + B.2^2 = 1) ∧ (y = B.1 + m)) ∧
    (area (F1, A, B) = 2 * area (F2, A, B)) →
    m = -Real.sqrt 2 / 3 := by sorry

end ellipse_intersection_area_condition_l133_133030


namespace tangent_line_equations_triangle_area_l133_133877

noncomputable def problem_conditions :
  Prop := ∃ (x y : ℝ), (x^2 + y^2 - 4*x - 6*y + 12 = 0) ∧ (x = 3 ∧ y = 5)

theorem tangent_line_equations (x y : ℝ) (hx : x^2 + y^2 - 4*x - 6*y + 12 = 0) :
  (∃ k : ℝ, y = k*x + 5 - 3*k ∧ (abs (-k + 2) / sqrt (k^2 + 1) = 1)) → 
  (∃ k : ℝ, k = 3 / 4 ∨ x = 3) :=
sorry

theorem triangle_area (x y : ℝ) (hx : x^2 + y^2 - 4*x - 6*y + 12 = 0) :
  S = 1/2 :=
sorry

end tangent_line_equations_triangle_area_l133_133877


namespace mixture_proportion_exists_l133_133121

-- Define the ratios and densities of the liquids
variables (k : ℝ) (ρ1 ρ2 ρ3 : ℝ) (m1 m2 m3 : ℝ)
variables (x y : ℝ)

-- Given conditions
def density_ratio : Prop := 
  ρ1 = 6 * k ∧ ρ2 = 3 * k ∧ ρ3 = 2 * k

def mass_condition : Prop := 
  m2 / m1 ≤ 2 / 7

-- Must prove that a solution exists where the resultant density is the arithmetic mean
def mixture_density : Prop := 
  (m1 + m2 + m3) / ((m1 / ρ1) + (m2 / ρ2) + (m3 / ρ3)) = (ρ1 + ρ2 + ρ3) / 3

-- Statement (No proof provided)
theorem mixture_proportion_exists (k : ℝ) (m1 m2 m3 : ℝ) (ρ1 ρ2 ρ3 : ℝ) (x y : ℝ) :
  density_ratio k ρ1 ρ2 ρ3 →
  mass_condition m1 m2 →
  mixture_density m1 m2 m3 k ρ1 ρ2 ρ3 :=
sorry

end mixture_proportion_exists_l133_133121


namespace option_B_is_orthogonal_l133_133810

def vector_a : ℝ × ℝ × ℝ := (1, -1, 1)
def vector_b : ℝ × ℝ × ℝ := (-1, 0, 1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem option_B_is_orthogonal :
  dot_product vector_a vector_b = 0 :=
by sorry

end option_B_is_orthogonal_l133_133810


namespace prob_task1_and_not_task2_l133_133341

def prob_task1_completed : ℚ := 5 / 8
def prob_task2_completed : ℚ := 3 / 5

theorem prob_task1_and_not_task2 : 
  ((prob_task1_completed) * (1 - prob_task2_completed)) = 1 / 4 := 
by 
  sorry

end prob_task1_and_not_task2_l133_133341


namespace stocks_closed_higher_today_l133_133333

theorem stocks_closed_higher_today (yesterday_today_difference : ∀ i : ℕ, i < 1980 → ℕ)
  (higher_price_greater : 0.20):
  (∃ high low : ℕ, high = 1080 ∧ low + 1.20 * low = 1980) := by
  let low := 900
  let high := 1.20 * low
  have h : high = 1080 := by
    calc
      high = 1.20 * low : rfl
         ... = 1.20 * 900 : rfl
         ... = 1080 : by norm_num
  exact ⟨1080, 900, h, sorry⟩

end stocks_closed_higher_today_l133_133333


namespace cost_of_gravelling_l133_133337

theorem cost_of_gravelling (length width pathWidth : ℝ) (costPerSqMeter : ℝ)
                           (h_length : length = 110)
                           (h_width : width = 65)
                           (h_pathWidth : pathWidth = 2.5)
                           (h_costPerSqMeter : costPerSqMeter = 0.4) :
  let totalLength := length + 2 * pathWidth
      totalWidth := width + 2 * pathWidth
      areaTotal := totalLength * totalWidth
      areaGrassy := length * width
      areaPath := areaTotal - areaGrassy
  in areaPath * costPerSqMeter = 360 := 
by 
  sorry

end cost_of_gravelling_l133_133337


namespace num_3_digit_multiples_l133_133510

def is_3_digit (n : Nat) : Prop := 100 ≤ n ∧ n ≤ 999
def multiple_of (k n : Nat) : Prop := ∃ m : Nat, n = m * k

theorem num_3_digit_multiples (count_35_not_70 : Nat) (h : count_35_not_70 = 13) :
  let count_multiples_35 := (980 / 35) - (105 / 35) + 1
  let count_multiples_70 := (980 / 70) - (140 / 70) + 1
  count_multiples_35 - count_multiples_70 = count_35_not_70 := sorry

end num_3_digit_multiples_l133_133510


namespace trig_identity_proof_l133_133092

theorem trig_identity_proof (α β : ℝ) (h : cos α ^ 2 * sin β ^ 2 + sin α ^ 2 * cos β ^ 2 = cos α * sin α * cos β * sin β) :
  (sin β ^ 2 * cos α ^ 2 / sin α ^ 2) + (cos β ^ 2 * sin α ^ 2 / cos α ^ 2) = 1 :=
by
  sorry

end trig_identity_proof_l133_133092


namespace weather_on_tenth_day_cloudy_l133_133315

/-- 
Vasya solved problems for 10 days - at least one problem each day.
Every day (except the first), if the weather was cloudy, he solved one more problem than the previous day,
and if it was sunny, he solved one less problem.
In the first 9 days, Vasya solved 13 problems. 
Prove that the weather was cloudy on the tenth day.
-/
theorem weather_on_tenth_day_cloudy
  (problems : Fin 10 → ℕ)
  (weather : Fin 10 → Bool)
  (hpos : ∀ i, 1 ≤ problems i)
  (hinit : ∑ i in Finset.range 9, problems i = 13)
  (hweather : ∀ i, 0 < i → 
                (¬weather i → problems i = problems (i - 1) + 1) ∧ 
                (weather i → problems i = problems (i - 1) - 1)) :
  ¬weather 9 := 
sorry

end weather_on_tenth_day_cloudy_l133_133315


namespace isosceles_right_triangle_exists_l133_133786

theorem isosceles_right_triangle_exists
  {A B C D E M : Type*}
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited M]
  (is_isosceles_right_triangle_ABC : is_isosceles_right_triangle A B C)
  (is_isosceles_right_triangle_ADE : is_isosceles_right_triangle A D E)
  (fixed_ABC : fixed_triangle A B C)
  (rotated_ADE : rotated_around A D E)
  (M_on_EC : M ∈ line_segment E C)
  : ∃ M : Type*, is_isosceles_right_triangle B M D := sorry

end isosceles_right_triangle_exists_l133_133786


namespace partial_sum_sequence_b_theorem_l133_133499

noncomputable def partial_sum_sequence_a (n : ℕ) : ℕ :=
2 ^ n - 1

def sequence_a (n : ℕ) : ℕ :=
if n = 0 then 0 else 2 ^ (n - 1)

def b_sequence_base : ℕ := 3

def sequence_b : ℕ → ℕ
| 0       := b_sequence_base
| (k + 1) := sequence_a k + sequence_b k

noncomputable def partial_sum_sequence_b (n : ℕ) : ℕ :=
∑ i in Finset.range n, sequence_b i

theorem partial_sum_sequence_b_theorem (n : ℕ) :
  partial_sum_sequence_b n = 2^n + 2 * n - 1 :=
sorry

end partial_sum_sequence_b_theorem_l133_133499


namespace find_n_l133_133744

theorem find_n (n : ℕ) (h : 256 = 2 ^ 8) : (256 ^ (1 / 2 : ℝ) = 2 ^ n) → n = 4 :=
by
  sorry

end find_n_l133_133744


namespace max_volume_l133_133364

variable (x y z : ℝ) (V : ℝ)
variable (k : ℝ)

-- Define the constraint
def constraint := x + 2 * y + 3 * z = 180

-- Define the volume
def volume := x * y * z

-- The goal is to show that under the constraint, the maximum possible volume is 36000 cubic cm.
theorem max_volume :
  (∀ (x y z : ℝ) (h : constraint x y z), volume x y z ≤ 36000) :=
  sorry

end max_volume_l133_133364


namespace minimum_concerts_l133_133393

theorem minimum_concerts (total_musicians : ℕ) (h : total_musicians = 6) :
  ∃ (min_concerts : ℕ), min_concerts = 4 :=
by
  use 4
  sorry

end minimum_concerts_l133_133393


namespace area_ratio_ellipse_l133_133086

noncomputable def ellipse := { x // (x.1^2 / 3 + x.2^2 = 1) }
def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))
def intersects (line : ℝ × ℝ) (C : ℝ × ℝ → Prop) := ∃ (x y : ℝ), (C (x, y)) ∧ (y = x + line.2)

theorem area_ratio_ellipse 
  (a b : ℝ) (h_ellipse : a^2 = 3 ∧ b^2 = 1) (m : ℝ) 
  (h_foci : foci a b = ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))) 
  (h_condition : ∀ (x : ℝ), intersects (x, m) (λ pt, pt.1^2 / 3 + pt.2^2 = 1) →
     |((x - real.sqrt (a^2 - b^2)) + m)| = 2|((x + real.sqrt (a^2 - b^2)) + m)|) :
  m = -real.sqrt(2) / 3 := sorry

end area_ratio_ellipse_l133_133086


namespace max_value_of_f_range_of_λ_summation_inequality_l133_133491

noncomputable def f (x : ℝ) : ℝ := (1 - x) * Real.exp x - 1

noncomputable def g (x : ℝ) (λ : ℝ) : ℝ := Real.exp x + λ * Real.log (1 - x) - 1

theorem max_value_of_f : ∀ x, f x ≤ 0 ∧ ∃ x, f x = 0 :=
by
  sorry

theorem range_of_λ (x : ℝ) : 
  x ≥ 0 → (∀ x, g x λ ≤ 0) → λ ≥ 1 :=
by
  sorry

theorem summation_inequality (n : ℕ) : 
  (∑ k in Finset.range (n + 1), 1 / Real.exp (n + k + 1)) < 
  n + Real.log 2 :=
by
  sorry

end max_value_of_f_range_of_λ_summation_inequality_l133_133491


namespace part1_part2_part3_l133_133867

-- Conditions
def A : Set ℝ := { x : ℝ | 2 < x ∧ x < 6 }
def B (m : ℝ) : Set ℝ := { x : ℝ | m + 1 < x ∧ x < 2 * m }

-- Proof statements
theorem part1 : A ∪ B 2 = { x : ℝ | 2 < x ∧ x < 6 } := by
  sorry

theorem part2 (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) → m ≤ 3 := by
  sorry

theorem part3 (m : ℝ) : (∃ x, x ∈ B m) ∧ (∀ x, x ∉ A ∩ B m) → m ≥ 5 := by
  sorry

end part1_part2_part3_l133_133867


namespace probability_Jenny_Jack_in_picture_l133_133178

theorem probability_Jenny_Jack_in_picture :
  ∀ (t : ℕ), t = 900 →
  (∃ (prob : ℚ), prob = 23 / 60) :=
by
  -- Define constants
  let jenny_lap_time : ℕ := 75
  let jack_lap_time : ℕ := 70

  -- Define the time in seconds from the start
  let total_seconds : ℕ := 900

  -- Restate given conditions
  have h1 : total_seconds = 900 := by rfl
  
  -- Assume the calculated probabilities
  let jenny_in_picture_seconds : ℚ := 25
  let jack_in_picture_seconds : ℚ := 23.33

  -- Calculate the overlap
  let overlap : ℚ := min jenny_in_picture_seconds jack_in_picture_seconds

  -- Define the total time window
  let total_time_window : ℕ := 60

  -- Calculate the probability
  let probability : ℚ := overlap / total_time_window

  -- State the conclusion
  have h2 : probability = 23 / 60 := sorry

  -- Conclude the theorem
  exact ⟨23 / 60, h2⟩

end probability_Jenny_Jack_in_picture_l133_133178


namespace sophie_total_spend_l133_133602

def total_cost_with_discount_and_tax : ℝ :=
  let cupcakes_price := 5 * 2
  let doughnuts_price := 6 * 1
  let apple_pie_price := 4 * 2
  let cookies_price := 15 * 0.60
  let chocolate_bars_price := 8 * 1.50
  let soda_price := 12 * 1.20
  let gum_price := 3 * 0.80
  let chips_price := 10 * 1.10
  let total_before_discount := cupcakes_price + doughnuts_price + apple_pie_price + cookies_price + chocolate_bars_price + soda_price + gum_price + chips_price
  let discount := 0.10 * total_before_discount
  let subtotal_after_discount := total_before_discount - discount
  let sales_tax := 0.06 * subtotal_after_discount
  let total_cost := subtotal_after_discount + sales_tax
  total_cost

theorem sophie_total_spend :
  total_cost_with_discount_and_tax = 69.45 :=
sorry

end sophie_total_spend_l133_133602


namespace triangle_dot_product_is_two_l133_133871

noncomputable def triangle_dot_product
    (AB AC : ℝ) (angle_A : ℝ) -- these represent |AB|, |AC|, and ∠A respectively
    (area : ℝ)
    (h0 : area = sqrt 3) 
    (h1 : angle_A = π / 3)
    : ℝ :=
    let product := (2 * area) / (sin angle_A) in
    product * (cos angle_A)

theorem triangle_dot_product_is_two
    (AB AC : ℝ) (angle_A : ℝ)
    (area : ℝ)
    (h0 : area = sqrt 3) 
    (h1 : angle_A = π / 3) :
    triangle_dot_product AB AC angle_A area h0 h1 = 2 :=
by
  sorry

end triangle_dot_product_is_two_l133_133871


namespace repeating_decimal_sum_l133_133639

/-
The sum of the numerator and the denominator of the fraction representing 3.71717171... in its lowest terms is 467.
-/
theorem repeating_decimal_sum (y : ℚ) (h : y = 3 + 71/990) : y.num + y.denom = 467 :=
by {
  have : y = 368 / 99,
  { calc
      y = 3 + 71 / 990 : by rw h
      ... = 3 + 71 / 99 / 10 : by norm_num
      ... = (3 * 10 + 71 / 99 * 10) / 10 : by rw add_div
      ... = (30 + 7.171717171717171817 * 10) / 10 : by norm_num
      ... = 368 / 99 : sorry},
  sorry }

#print repeating_decimal_sum

end repeating_decimal_sum_l133_133639


namespace find_n_value_l133_133278

theorem find_n_value (n : ℕ) (a : ℕ → ℝ) (S_n : ℕ → ℝ) :
  (∀ n, a n = 1 / (sqrt (n + 1) + sqrt n)) →
  S_n n = 9 →
  n = 99 :=
by
  intro ha hS
  sorry

end find_n_value_l133_133278


namespace f_2017_eq_one_l133_133859

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x - β)

-- Given conditions
variables {a b α β : ℝ}
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ α ≠ 0 ∧ β ≠ 0)
variable (h_f2016 : f 2016 a α b β = -1)

-- The goal
theorem f_2017_eq_one : f 2017 a α b β = 1 :=
sorry

end f_2017_eq_one_l133_133859


namespace complete_the_square_l133_133720

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 6 * x - 10 = 0

-- State the proof problem
theorem complete_the_square (x : ℝ) (h : quadratic_eq x) : (x - 3)^2 = 19 :=
by 
  -- Skip the proof using sorry
  sorry

end complete_the_square_l133_133720


namespace proof_inequality_l133_133463

noncomputable def inequality_proof (α : ℝ) (a b : ℝ) (m : ℕ) : Prop :=
  (0 < α) → (α < Real.pi / 2) →
  (m ≥ 1) →
  (0 < a) → (0 < b) →
  (a / (Real.cos α)^m + b / (Real.sin α)^m ≥ (a^((2:ℝ)/(m+2)) + b^((2:ℝ)/(m+2)))^((m+2)/2))

-- Statement of the proof problem
theorem proof_inequality (α : ℝ) (a b : ℝ) (m : ℕ) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : 1 ≤ m) (h4 : 0 < a) (h5 : 0 < b) : 
  a / (Real.cos α)^m + b / (Real.sin α)^m ≥ 
    (a^((2:ℝ)/(m+2)) + b^((2:ℝ)/(m+2)))^((m+2)/2) :=
by
  sorry

end proof_inequality_l133_133463


namespace effective_rate_of_interest_l133_133247

-- Defining the necessary constants and the formula for EAR
def nominal_interest_rate : ℝ := 0.12
def compounding_periods : ℕ := 2
def time_years : ℕ := 1

def effective_annual_rate (i : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  (1 + i / n) ^ (n * t) - 1

-- The proof statement
theorem effective_rate_of_interest :
  (effective_annual_rate nominal_interest_rate compounding_periods time_years) * 100 = 12.36 :=
by
  sorry

end effective_rate_of_interest_l133_133247


namespace even_function_property_l133_133873

def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := f(x) + (2:ℝ)^x
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem even_function_property (hf: even_function f) (h1 : g (Real.log 7 / Real.log 2) = 3) :
    g (Real.log (1/7) / Real.log 2) = -(27/7 : ℝ) :=
sorry

end even_function_property_l133_133873


namespace steven_has_72_shirts_l133_133603

def brian_shirts : ℕ := 3
def andrew_shirts (brian : ℕ) : ℕ := 6 * brian
def steven_shirts (andrew : ℕ) : ℕ := 4 * andrew

theorem steven_has_72_shirts : steven_shirts (andrew_shirts brian_shirts) = 72 := 
by 
  -- We add "sorry" here to indicate that the proof is omitted
  sorry

end steven_has_72_shirts_l133_133603


namespace tenth_root_of_unity_identity_l133_133982

noncomputable def z : ℂ := complex.cos (3 * real.pi / 5) + complex.sin (3 * real.pi / 5) * complex.I

theorem tenth_root_of_unity_identity :
  (z / (1 + z^2)) + (z^3 / (1 + z^6)) + (z^5 / (1 + z^10)) = (z + z^3 - 1/2) / 3 :=
by {
  -- Proof is omitted
  sorry
}

end tenth_root_of_unity_identity_l133_133982


namespace time_sum_l133_133558

def currentTime : Time := ⟨15, 15, 20⟩
def additionalTime : Duration := ⟨198 * 3600 + 47 * 60 + 36⟩ -- Convert hours and minutes to seconds

theorem time_sum : 
  let ⟨A, B, C⟩ := addTime currentTime additionalTime 
  in A + B + C = 68 := sorry

end time_sum_l133_133558


namespace sum_of_10_least_n_Sn_div_3_l133_133008

def S_n (n : ℕ) : ℕ := (n * (n - 1) * (n + 1) * (3 * n + 2)) / 24

theorem sum_of_10_least_n_Sn_div_3 : 
  ∑ i in (Finset.range 40).filter (λ n, S_n n % 3 = 0) | (take 10) = 197 :=
by
  sorry

end sum_of_10_least_n_Sn_div_3_l133_133008


namespace mathematicians_correctness_l133_133260

theorem mathematicians_correctness:
  ∀ (w1₁ w1₂ t1₁ t1₂ w2₁ w2₂ t2₁ t2₂: ℕ),
  (w1₁ = 4) ∧ (t1₁ = 7) ∧ (w1₂ = 3) ∧ (t1₂ = 5) →
  (w2₁ = 8) ∧ (t2₁ = 14) ∧ (w2₂ = 3) ∧ (t2₂ = 5) →
  (4/7 : ℚ) < 3/5 →
  (w1₁ + w1₂) / (t1₁ + t1₂) = 7 / 12 ∨ (w2₁ + w2₂) / (t2₁ + t2₂) = 11 / 19 →
  ¬ (w1₁ + w1₂) / (t1₁ + t1₂) < 19 / 35 ∧ (w2₁ + w2₂) / (t2₁ + t2₂) < 19 / 35 :=
by
  sorry

end mathematicians_correctness_l133_133260


namespace eccentricity_of_ellipse_l133_133103

theorem eccentricity_of_ellipse (a c : ℝ) (h1 : 2 * c = a) : (c / a) = (1 / 2) :=
by
  -- This is where we would write the proof, but we're using sorry to skip the proof steps.
  sorry

end eccentricity_of_ellipse_l133_133103


namespace count_integers_satisfying_inequality_l133_133906

theorem count_integers_satisfying_inequality :
  {m : ℤ | m ≠ 0 ∧ (1 : ℚ) / |m| ≥ 1 / 5}.to_finset.card = 10 :=
by
  sorry

end count_integers_satisfying_inequality_l133_133906


namespace convex_over_real_l133_133000

def f (x : ℝ) : ℝ := x^4 - 2 * x^3 + 36 * x^2 - x + 7

theorem convex_over_real : ∀ x : ℝ, 0 ≤ (12 * x^2 - 12 * x + 72) :=
by sorry

end convex_over_real_l133_133000


namespace composite_function_evaluation_l133_133113

variable (f g : ℝ → ℝ)
variable (x : ℝ)

-- Definitions of the functions
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x + 2

theorem composite_function_evaluation : f (g 3) = 25 :=
by
  -- Sorry is used to skip the proof, ensuring that the statement can be checked as valid.
  sorry

end composite_function_evaluation_l133_133113


namespace fiftyth_day_of_N_minus_1_is_tuesday_l133_133176

theorem fiftyth_day_of_N_minus_1_is_tuesday :
  ∀ (N : ℕ),
  (day_of_week N 250 = friday) →
  (day_of_week (N + 1) 150 = friday) →
  (is_non_leap_year (N + 1)) →
  (day_of_week (N - 1) 50 = tuesday) :=
by
  sorry

end fiftyth_day_of_N_minus_1_is_tuesday_l133_133176


namespace hypotenuse_length_l133_133372

theorem hypotenuse_length (x y h : ℝ)
  (hx : (1 / 3) * π * y * x^2 = 1620 * π)
  (hy : (1 / 3) * π * x * y^2 = 3240 * π) :
  h = Real.sqrt 507 :=
by
  sorry

end hypotenuse_length_l133_133372


namespace roots_ratio_sum_eq_six_l133_133911

theorem roots_ratio_sum_eq_six (x1 x2 : ℝ) (h1 : 2 * x1^2 - 4 * x1 + 1 = 0) (h2 : 2 * x2^2 - 4 * x2 + 1 = 0) :
  (x1 / x2) + (x2 / x1) = 6 :=
sorry

end roots_ratio_sum_eq_six_l133_133911


namespace tangent_parallel_l133_133824

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel (P₀ : ℝ × ℝ) :
  (∃ x : ℝ, (P₀ = (x, f x) ∧ deriv f x = 4)) 
  ↔ (P₀ = (1, 0) ∨ P₀ = (-1, -4)) :=
by 
  sorry

end tangent_parallel_l133_133824


namespace four_digit_integers_count_l133_133904

def is_valid_digit_set (digits : List ℕ) : Prop :=
  digits.length = 4 ∧ digits.nodup ∧ 6 ∈ digits ∧ (∀ d ∈ digits, d < 10)

def satisfies_conditions (n : ℕ) : Prop :=
  let digits := n.digits 10
  1000 ≤ n ∧ n < 10000 ∧
  is_valid_digit_set digits ∧
  digits.head ≠ 0 ∧
  n % 4 = 0 ∧
  digits.maximum = some 6

theorem four_digit_integers_count : ∃ count : ℕ, count = 50 ∧
  (count = ∑ n in finset.range 10000, if satisfies_conditions n then 1 else 0) := 
sorry

end four_digit_integers_count_l133_133904


namespace min_value_of_n_l133_133199

theorem min_value_of_n 
  (n : ℕ) 
  (x : ℕ → ℚ) 
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → |x i| < 1)
  (h2 : ∑ i in Finset.range n, |x i| = 19 + |∑ i in Finset.range n, x i|) 
  : n = 20 := 
sorry

end min_value_of_n_l133_133199


namespace find_f_values_l133_133985

noncomputable def f : ℕ → ℕ := sorry

axiom condition1 : ∀ (a b : ℕ), a ≠ b → (a * f a + b * f b > a * f b + b * f a)
axiom condition2 : ∀ (n : ℕ), f (f n) = 3 * n

theorem find_f_values : f 1 + f 6 + f 28 = 66 := 
by
  sorry

end find_f_values_l133_133985


namespace symmetric_graphs_about_y_eq_x_imply_inverse_l133_133874

def f (x : ℝ) : ℝ := (1 / 2) ^ x
def g (x : ℝ) : ℝ := log (1/2) x

theorem symmetric_graphs_about_y_eq_x_imply_inverse :
  (∀ x : ℝ, f (g x) = x) ∧ (∀ y : ℝ, g (f y) = y) ->
  f (-2) = 4 :=
by
  intro h
  sorry

end symmetric_graphs_about_y_eq_x_imply_inverse_l133_133874


namespace ellipse_intersection_area_condition_l133_133027

theorem ellipse_intersection_area_condition :
  let ellipse := ∀ (x y : ℝ), (x^2 / 3 + y^2 = 1)
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let line m := ∀ (x y : ℝ), (y = x + m)
  ∀ (A B : ℝ × ℝ),
    ((A.1^2 / 3 + A.2^2 = 1) ∧ (y = A.1 + m)) ∧
    ((B.1^2 / 3 + B.2^2 = 1) ∧ (y = B.1 + m)) ∧
    (area (F1, A, B) = 2 * area (F2, A, B)) →
    m = -Real.sqrt 2 / 3 := by sorry

end ellipse_intersection_area_condition_l133_133027


namespace FibonacciCoeffSymmetry_FibonacciCoeffFormula_l133_133344

-- Define Fibonacci coefficients
def fib (n : ℕ) : ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fib (n + 1) + fib n

def FibonacciCoefficient (n k : ℕ) : ℚ :=
  if k = 0 then 1
  else (finset.range k).prod (λ i => fib (n - i)) / (finset.range k).prod (λ i => fib (k - i))

-- Problem (a): Symmetry of Fibonacci Coefficients
theorem FibonacciCoeffSymmetry (n k : ℕ) (hnk : k ≤ n) :
  FibonacciCoefficient n k = FibonacciCoefficient n (n - k) :=
sorry

-- Problem (b): Formula relating Fibonacci coefficients
theorem FibonacciCoeffFormula (n k : ℕ) (hn : 0 < n) (hk : 0 < k) :
  FibonacciCoefficient n k = fib (n - k + 1) * FibonacciCoefficient (n - 1) (k - 1) + fib (k - 1) * FibonacciCoefficient (n - 1) k :=
sorry

end FibonacciCoeffSymmetry_FibonacciCoeffFormula_l133_133344


namespace least_k_l133_133010

def a_n (n : ℕ) (h : n ≥ 2) : ℝ := (real.cbrt (n^3 + n^2 - n - 1)) / n

theorem least_k (k : ℕ) (h : 106) : (∏ n in finset.Icc 2 106, a_n n (by linarith [finset.mem_Icc.mp _])) > 3 := sorry

end least_k_l133_133010


namespace find_m_l133_133036

variables (a b : ℝ) (x y m : ℝ) (A B F1 F2 : ℝ × ℝ)

-- Given conditions
def ellipse := (x^2 / 3) + y^2 = 1
def line := y = x + m
def foci_F1 := F1 = (-sqrt 2, 0)
def foci_F2 := F2 = (sqrt 2, 0)
def area_F1_AB := λ A B : ℝ × ℝ, 2 * |det(F1, A, B)| = |det(F2, A, B)|

-- The proof problem statement
theorem find_m : 
  ellipse ∧ 
  line ∧ 
  foci_F1 ∧ 
  foci_F2 ∧ 
  (area_F1_AB A B) →
  m = -sqrt(2) / 3 := sorry

end find_m_l133_133036


namespace travelers_cross_river_l133_133307

variables (traveler1 traveler2 traveler3 : ℕ)  -- weights of travelers
variable (raft_capacity : ℕ)  -- maximum carrying capacity of the raft

-- Given conditions
def conditions :=
  traveler1 = 3 ∧ traveler2 = 3 ∧ traveler3 = 5 ∧ raft_capacity = 7

-- Prove that the travelers can all cross the river successfully
theorem travelers_cross_river :
  conditions traveler1 traveler2 traveler3 raft_capacity →
  (traveler1 + traveler2 ≤ raft_capacity) ∧
  (traveler1 ≤ raft_capacity) ∧
  (traveler3 ≤ raft_capacity) ∧
  (traveler1 + traveler2 ≤ raft_capacity) →
  true :=
by
  intros h_conditions h_validity
  sorry

end travelers_cross_river_l133_133307


namespace triangle_equilateral_from_midpoint_circles_l133_133317

theorem triangle_equilateral_from_midpoint_circles (a b c : ℝ)
  (h1 : ∃ E F G : ℝ → ℝ, ∀ x, (|E x| = a/4 ∨ |F x| = b/4 ∨ |G x| = c/4))
  (h2 : (|a/2| ≤ a/4 + b/4) ∧ (|b/2| ≤ b/4 + c/4) ∧ (|c/2| ≤ c/4 + a/4)) :
  a = b ∧ b = c :=
sorry

end triangle_equilateral_from_midpoint_circles_l133_133317


namespace line_AB_fixed_point_l133_133218

-- Define the vector space over real numbers.
variables {V : Type*} [inner_product_space ℝ V] [complete_space V]

-- Define the points O, A, and B in the vector space V.
variables (O A B : V)

-- Define the scalars p, q, c as real numbers.
variables (p q c : ℝ)

-- Define the distances (OA and OB) as positive real numbers.
variables (OA OB : ℝ) (hOA : 0 < OA) (hOB : 0 < OB)

-- Conditions stating that A and B move along fixed rays from O.
-- OA = OA would be the magnitude of vector (O -> A), similarly OB.
axiom ray_condition : OA ≠ 0 ∧ OB ≠ 0 ∧ 
  (∃ (λ μ : ℝ), λ ≠ 0 ∧ μ ≠ 0 ∧ A = O + λ • (A - O) ∧ B = O + μ • (B - O))

-- Given condition that the quantity remains constant.
axiom constant_condition : p / OA + q / OB = c

-- Prove the line AB passes through a fixed point.
theorem line_AB_fixed_point : ∃ F : V, ∀ (A B : V), 
  (ray_condition O A B OA OB) → 
  (constant_condition p q c O A B OA OB) →
  ∃ t : ℝ, (F = (t • A + (1 - t) • B)) :=
begin
  sorry
end

end line_AB_fixed_point_l133_133218


namespace number_of_female_athletes_l133_133757

theorem number_of_female_athletes (male_athletes female_athletes coaches sample_size : ℕ) 
    (total_individuals : ℕ := male_athletes + female_athletes + coaches)
    (probability_of_selection : ℚ := sample_size / total_individuals) :
  male_athletes = 112 ∧ female_athletes = 84 ∧ coaches = 28 ∧ sample_size = 32 →
  female_athletes * probability_of_selection = 12 := 
by 
  intros h
  cases h with h₁ h₂
  cases h₂ with h₃ h₄
  cases h₄ with h₅ h₆
  rw [h₁, h₃, h₅, h₆]
  have h_probability := calc
    32 / (112 + 84 + 28) = 32 / 224 : by norm_num
    ... = 1 / 7 : by norm_num
  rw [h_probability]
  norm_num
  sorry

end number_of_female_athletes_l133_133757


namespace find_m_value_l133_133048

-- Define the ellipse equation and its foci
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define the foci of the ellipse C
def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the line equation
def line (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the condition for the areas of triangles F1AB and F2AB
def triangle_area_condition (F1 F2 : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := sorry in -- Coordinates of intersection points need to be defined
  let B := sorry in -- Coordinates of intersection points need to be defined
  sorry /- The actual calculation of the areas would require more complex integration and geometry.
           Here we just outline the condition -/

-- The proof statement
theorem find_m_value :
  ∃ m : ℝ, 
    ∀ x y : ℝ,
      ellipse x y →
      line m x y →
      triangle_area_condition F1 F2 m →
      m = -Real.sqrt 2 / 3 :=
  sorry

end find_m_value_l133_133048


namespace num_ways_from_a_to_b_l133_133761

-- Define the structure of the grid, movements and the cells a and b
variable (Grid : Type) [Nonempty Grid] [Finite Grid]

-- Define cells
variable (a b : Grid)

-- Define movement rules
axiom move_up : Grid → Grid
axiom move_right : Grid → Grid
axiom move_down : Grid → Grid

-- Define conditions: movement one cell at a time and avoiding blocked cells
axiom valid_move : (g : Grid) → (move_up g ≠ g) ∧ (move_right g ≠ g) ∧ (move_down g ≠ g)

-- Define the proof problem
theorem num_ways_from_a_to_b (a b : Grid) (h_moves : (move_up a = move_up b) ∨ (move_right a = move_right b)) : 
  ∃ (n : ℕ), n = 16 :=
by
  sorry

end num_ways_from_a_to_b_l133_133761


namespace gcd_three_digit_palindromes_l133_133677

theorem gcd_three_digit_palindromes : ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → Nat.gcd (102 * a + 10 * b) 1 = 1 :=
by
  intros a b h
  sorry

end gcd_three_digit_palindromes_l133_133677


namespace cot_arccot_identity_cot_addition_formula_problem_solution_l133_133841

noncomputable def cot (x : ℝ) : ℝ := 1 / tan x
noncomputable def arccot (x : ℝ) : ℝ := arctan (1 / x)

theorem cot_arccot_identity (a : ℝ) : cot (arccot a) = a := sorry

theorem cot_addition_formula (a b : ℝ) : 
  cot (arccot a + arccot b) = (a * b - 1) / (a + b) := sorry

theorem problem_solution : 
  cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 971 / 400 := sorry

end cot_arccot_identity_cot_addition_formula_problem_solution_l133_133841


namespace mathematicians_correctness_l133_133272

theorem mathematicians_correctness :
  ∃ (scenario1_w1 s_w1 : ℕ) (scenario1_w2 s_w2 : ℕ) (scenario2_w1 s2_w1 : ℕ) (scenario2_w2 s2_w2 : ℕ),
    scenario1_w1 = 4 ∧ s_w1 = 7 ∧ scenario1_w2 = 3 ∧ s_w2 = 5 ∧
    scenario2_w1 = 8 ∧ s2_w1 = 14 ∧ scenario2_w2 = 3 ∧ s2_w2 = 5 ∧
    let total_white1 := scenario1_w1 + scenario1_w2,
        total_choco1 := s_w1 + s_w2,
        prob1 := (total_white1 : ℚ) / total_choco1,
        total_white2 := scenario2_w1 + scenario2_w2,
        total_choco2 := s2_w1 + s2_w2,
        prob2 := (total_white2 : ℚ) / total_choco2,
        prob_box1 := 4 / 7,
        prob_box2 := 3 / 5 in
    (prob1 = 7 / 12 ∧ prob2 = 11 / 19 ∧
    (19 / 35 < prob_box1 ∧ prob_box1 < prob_box2) ∧
    (prob_box1 ≠ 19 / 35 ∧ prob_box1 ≠ 3 / 5)) :=
sorry

end mathematicians_correctness_l133_133272


namespace inscribed_square_side_length_l133_133556

noncomputable def height_of_equilateral_triangle (a : ℝ) : ℝ :=
  a * Real.sqrt 3 / 2

theorem inscribed_square_side_length (a x : ℝ) (h : height_of_equilateral_triangle a) :
  ∃ x : ℝ, x = a * (2 - Real.sqrt 3) :=
by
  sorry

end inscribed_square_side_length_l133_133556


namespace rick_has_eaten_servings_l133_133297

theorem rick_has_eaten_servings (calories_per_serving block_servings remaining_calories total_calories servings_eaten : ℝ) 
  (h1 : calories_per_serving = 110) 
  (h2 : block_servings = 16) 
  (h3 : remaining_calories = 1210) 
  (h4 : total_calories = block_servings * calories_per_serving)
  (h5 : servings_eaten = (total_calories - remaining_calories) / calories_per_serving) :
  servings_eaten = 5 :=
by 
  sorry

end rick_has_eaten_servings_l133_133297


namespace range_of_y_l133_133425

theorem range_of_y :
  ∀ (b : Fin 20 → ℕ),
    (∀ i, (b i) = 0 ∨ (b i) = 3) →
    0 ≤ (∑ i, (b i) / 4^(1 + i)) ∧ 
    (∑ i, (b i) / 4^(1 + i) < 1/4 ∨ 3/4 ≤ ∑ i, (b i) / 4^(1 + i) ∧ ∑ i, (b i) / 4^(1 + i) ≤ 1) :=
by
  sorry

end range_of_y_l133_133425


namespace polynomial_product_linear_term_zero_const_six_l133_133922

theorem polynomial_product_linear_term_zero_const_six (a b : ℝ)
  (h1 : (a + 2 * b = 0)) 
  (h2 : b = 6) : (a + b = -6) :=
by
  sorry

end polynomial_product_linear_term_zero_const_six_l133_133922


namespace find_f2_l133_133961

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f (f x) = (x ^ 2 - x) / 2 * f x + 2 - x

theorem find_f2 : f 2 = 2 :=
by
  sorry

end find_f2_l133_133961


namespace steven_shirts_l133_133607

theorem steven_shirts : 
  (∀ (S A B : ℕ), S = 4 * A ∧ A = 6 * B ∧ B = 3 → S = 72) := 
by
  intro S A B
  intro h
  cases h with h1 h2
  cases h2 with hA hB
  rw [hB, hA]
  sorry

end steven_shirts_l133_133607


namespace angle_between_vectors_l133_133902

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Defining the conditions given in the problem
def condition1 := ‖a‖ = 2
def condition2 := ‖b‖ = 3
def condition3 := (a - b) ⬝ a = 1

-- Defining the problem statement
theorem angle_between_vectors :
  condition1 a →
  condition2 b →
  condition3 a b →
  ∃ θ : ℝ, θ = real.arccos ((1 : ℝ) / (‖a‖ * ‖b‖)) ∧ θ = real.pi / 3 :=
by sorry

end angle_between_vectors_l133_133902


namespace combined_profit_is_14000_l133_133852

-- Define constants and conditions
def center1_daily_packages : ℕ := 10000
def daily_profit_per_package : ℝ := 0.05
def center2_multiplier : ℕ := 3
def days_per_week : ℕ := 7

-- Define the profit for the first center
def center1_daily_profit : ℝ := center1_daily_packages * daily_profit_per_package

-- Define the packages processed by the second center
def center2_daily_packages : ℕ := center1_daily_packages * center2_multiplier

-- Define the profit for the second center
def center2_daily_profit : ℝ := center2_daily_packages * daily_profit_per_package

-- Define the combined daily profit
def combined_daily_profit : ℝ := center1_daily_profit + center2_daily_profit

-- Define the combined weekly profit
def combined_weekly_profit : ℝ := combined_daily_profit * days_per_week

-- Prove that the combined weekly profit is $14,000
theorem combined_profit_is_14000 : combined_weekly_profit = 14000 := by
  -- You can replace sorry with the steps to solve the proof.
  sorry

end combined_profit_is_14000_l133_133852


namespace triangle_angles_l133_133175

theorem triangle_angles (A B C P Q : Type) 
  (h1 : ∠BAC = 60)
  (h2 : AP bisects ∠BAC ∧ AP intersects BC at P)
  (h3 : BQ bisects ∠ABC ∧ BQ intersects CA at Q)
  (h4 : AB + BP = AQ + BQ) :
  (∠BAC = 60) ∧ (∠ABC = 80) ∧ (∠ACB = 40) := 
by
  sorry

end triangle_angles_l133_133175


namespace find_angle_of_inclination_l133_133488

noncomputable def curve_polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * Real.cos (θ - Real.pi / 3) - 1 = 0

noncomputable def line_parametric_eq (t α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, sqrt 3 + t * Real.sin α)

noncomputable def curve_cartesian_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x - 2 * sqrt 3 * y - 1 = 0

theorem find_angle_of_inclination (α : ℝ) :
  (∀ (ρ θ : ℝ), curve_polar_equation ρ θ → 
   ∀ t, curve_cartesian_equation (fst (line_parametric_eq t α)) (snd (line_parametric_eq t α)) →
   ∃ (t1 t2 : ℝ), t1 + t2 = 2 * Real.cos α ∧ t1 * t2 = -4 ∧ 
   Real.abs (t1 - t2) = 3 * sqrt 2) →
   α = Real.pi / 4 ∨ α = 3 * Real.pi / 4 :=
sorry

end find_angle_of_inclination_l133_133488


namespace distance_from_origin_l133_133918

theorem distance_from_origin (m : ℝ) (h : m + 1 = 0) : Real.dist (m - 2, m + 1) (0, 0) = 3 :=
by 
  sorry

end distance_from_origin_l133_133918


namespace milk_rate_proof_l133_133657

theorem milk_rate_proof
  (initial_milk : ℕ := 30000)
  (time_pumped_out : ℕ := 4)
  (rate_pumped_out : ℕ := 2880)
  (time_adding_milk : ℕ := 7)
  (final_milk : ℕ := 28980) :
  ((final_milk - (initial_milk - time_pumped_out * rate_pumped_out)) / time_adding_milk = 1500) :=
by {
  sorry
}

end milk_rate_proof_l133_133657


namespace function_symmetry_at_point_l133_133110

theorem function_symmetry_at_point :
  ∀(x : ℝ), (\sin (2 * (x - π/12) - π/6) = -\sin (2 * (x + π/12) - π/6)) :=
λ x, sorry

end function_symmetry_at_point_l133_133110


namespace gcd_three_digit_palindromes_l133_133682

theorem gcd_three_digit_palindromes : ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → Nat.gcd (102 * a + 10 * b) 1 = 1 :=
by
  intros a b h
  sorry

end gcd_three_digit_palindromes_l133_133682


namespace minimum_distance_is_sqrt_2_l133_133439

noncomputable def minimum_distance_between_lines (m : ℝ) : ℝ :=
  (|m^2 + 4|) / (2 * real.sqrt 2)

theorem minimum_distance_is_sqrt_2 :
  ∃ m : ℝ, minimum_distance_between_lines m = real.sqrt 2 :=
begin
  use 0,
  have h : minimum_distance_between_lines 0 = real.sqrt 2,
  { unfold minimum_distance_between_lines,
    simp,
    norm_num },
  exact h,
end

end minimum_distance_is_sqrt_2_l133_133439


namespace helium_balloon_buoyancy_l133_133738

variable (m m₁ Mₐ M_b : ℝ)
variable (h₁ : m₁ = 10)
variable (h₂ : Mₐ = 4)
variable (h₃ : M_b = 29)

theorem helium_balloon_buoyancy :
  m = (m₁ * Mₐ) / (M_b - Mₐ) :=
by
  sorry

end helium_balloon_buoyancy_l133_133738


namespace problem_proof_l133_133611

theorem problem_proof (x : ℝ) (h : (27 * 9^x) / 4^x = 3^x / 8^x) : 2^(-(1 + Real.log2 3) * x) = 216 := by
  sorry

end problem_proof_l133_133611


namespace gcd_of_all_three_digit_palindromes_is_one_l133_133690

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define a function to calculate the gcd of a list of numbers
def gcd_list (l : List ℕ) : ℕ :=
  l.foldr Nat.gcd 0

-- The main theorem that needs to be proven
theorem gcd_of_all_three_digit_palindromes_is_one :
  gcd_list (List.filter is_palindrome {n | 100 ≤ n ∧ n ≤ 999}.toList) = 1 :=
by
  sorry

end gcd_of_all_three_digit_palindromes_is_one_l133_133690


namespace gcd_of_all_three_digit_palindromes_is_one_l133_133693

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define a function to calculate the gcd of a list of numbers
def gcd_list (l : List ℕ) : ℕ :=
  l.foldr Nat.gcd 0

-- The main theorem that needs to be proven
theorem gcd_of_all_three_digit_palindromes_is_one :
  gcd_list (List.filter is_palindrome {n | 100 ≤ n ∧ n ≤ 999}.toList) = 1 :=
by
  sorry

end gcd_of_all_three_digit_palindromes_is_one_l133_133693


namespace pufferfish_count_l133_133300

theorem pufferfish_count (s p : ℕ) (h1 : s = 5 * p) (h2 : s + p = 90) : p = 15 :=
by
  sorry

end pufferfish_count_l133_133300


namespace john_newspapers_l133_133182

theorem john_newspapers (N : ℕ) (selling_price buying_price total_cost total_revenue : ℝ) 
  (h1 : selling_price = 2)
  (h2 : buying_price = 0.25 * selling_price)
  (h3 : total_cost = N * buying_price)
  (h4 : total_revenue = 0.8 * N * selling_price)
  (h5 : total_revenue - total_cost = 550) :
  N = 500 := 
by 
  -- actual proof here
  sorry

end john_newspapers_l133_133182


namespace intercept_sum_eq_30_l133_133743

/-- Defining variables and conditions --/
def m : ℕ := 29

/-- The equation is satisfied on modulo m graph paper if the sum of intercepts x₀ and y₀ equals 30 -/
theorem intercept_sum_eq_30 :
  ∃ (x₀ y₀ : ℤ), (0 ≤ x₀ ∧ x₀ < m) ∧ (0 ≤ y₀ ∧ y₀ < m) ∧
  (4 * x₀ ≡ -1 [MOD m]) ∧ (5 * y₀ ≡ -1 [MOD m]) ∧ (x₀ + y₀ = 30) :=
begin
  sorry
end

end intercept_sum_eq_30_l133_133743


namespace area_ratio_ellipse_l133_133084

noncomputable def ellipse := { x // (x.1^2 / 3 + x.2^2 = 1) }
def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))
def intersects (line : ℝ × ℝ) (C : ℝ × ℝ → Prop) := ∃ (x y : ℝ), (C (x, y)) ∧ (y = x + line.2)

theorem area_ratio_ellipse 
  (a b : ℝ) (h_ellipse : a^2 = 3 ∧ b^2 = 1) (m : ℝ) 
  (h_foci : foci a b = ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))) 
  (h_condition : ∀ (x : ℝ), intersects (x, m) (λ pt, pt.1^2 / 3 + pt.2^2 = 1) →
     |((x - real.sqrt (a^2 - b^2)) + m)| = 2|((x + real.sqrt (a^2 - b^2)) + m)|) :
  m = -real.sqrt(2) / 3 := sorry

end area_ratio_ellipse_l133_133084


namespace mathematicians_correct_l133_133255

noncomputable def scenario1 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 4 ∧ total1 = 7 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario2 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 8 ∧ total1 = 14 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario3 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  (19 / 35 < 4 / 7) ∧ (4 / 7 < 3 / 5)

noncomputable def probability (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : ℝ :=
  (whites1 + whites2) / (total1 + total2)

theorem mathematicians_correct :
  let whites1_s1 := 4 in
  let total1_s1 := 7 in
  let whites2_s1 := 3 in
  let total2_s1 := 5 in
  let whites1_s2 := 8 in
  let total1_s2 := 14 in
  let whites2_s2 := 3 in
  let total2_s2 := 5 in
  scenario1 whites1_s1 total1_s1 whites2_s1 total2_s1 →
  scenario2 whites1_s2 total1_s2 whites2_s2 total2_s2 →
  scenario3 whites1_s1 total1_s1 whites2_s2 total2_s2 →
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 ≤ 3 / 5 ∨
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 = 4 / 7 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 ≤ 3 / 5 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 = 4 / 7 :=
begin
  intros,
  sorry

end mathematicians_correct_l133_133255


namespace sum_first_ten_terms_arithmetic_sequence_zero_l133_133972

theorem sum_first_ten_terms_arithmetic_sequence_zero
  {a : ℕ → ℝ} (d : ℝ) (h₁ : d ≠ 0) (h₂ : (a 4)^2 + (a 5)^2 = (a 6)^2 + (a 7)^2) :
  (∑ i in finset.range 10, a i) = 0 := 
sorry

end sum_first_ten_terms_arithmetic_sequence_zero_l133_133972


namespace marathon_time_l133_133815

def total_time_marathon : ℝ := 
  let pace_flat := 10 / 1 -- 10 miles per hour on flat
  let time_segment_11_15 := 5 / (0.9 * pace_flat)
  let time_segment_16_19 := 4 / (0.85 * pace_flat)
  let time_segment_20_23 := 4 / (0.95 * pace_flat)
  let time_segment_24_26 := 3 / (0.98 * pace_flat)
  1 + (time_segment_11_15 + time_segment_16_19 + time_segment_20_23 + time_segment_24_26)

theorem marathon_time : total_time_marathon = 2.7534 := 
by
  sorry

end marathon_time_l133_133815


namespace fencing_required_l133_133336

theorem fencing_required
  (L : ℝ) (A : ℝ) (h_L : L = 20) (h_A : A = 400) : 
  (2 * (A / L) + L) = 60 :=
by
  sorry

end fencing_required_l133_133336


namespace relationship_between_a_b_c_l133_133858

def a : ℝ := Real.sqrt 5
def b : ℝ := 2
def c : ℝ := Real.sqrt 3

theorem relationship_between_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_between_a_b_c_l133_133858


namespace sinB_law_of_sines_l133_133168

variable (A B C : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Assuming a triangle with sides and angles as described
variable (a b : ℝ) (sinA sinB : ℝ)
variable (h₁ : a = 3) (h₂ : b = 5) (h₃ : sinA = 1 / 3)

theorem sinB_law_of_sines : sinB = 5 / 9 :=
by
  -- Placeholder for the proof
  sorry

end sinB_law_of_sines_l133_133168


namespace sphere_min_pressure_diameter_l133_133375

noncomputable def min_pressure_sphere_diameter (alpha : ℝ) (rho : ℝ) : ℝ :=
  2 * real.cbrt (3 * alpha / (2 * real.pi * rho^2))

theorem sphere_min_pressure_diameter : 
  let alpha := 30 -- Surface tension in dyn/cm
  let rho := 0.8 -- Density in g/cm^3
  min_pressure_sphere_diameter alpha rho = 14 :=
by
  sorry

end sphere_min_pressure_diameter_l133_133375


namespace semi_circle_area_correct_l133_133718

-- Define the diameter
def diameter : ℝ := 10

-- Define the radius
def radius : ℝ := diameter / 2

-- Define the area of a full circle
def full_circle_area : ℝ := Real.pi * radius^2

-- Define the area of the semi-circle
def semi_circle_area : ℝ := full_circle_area / 2

-- Theorem stating the area of the semi-circle is 12.5 * pi square meters
theorem semi_circle_area_correct : semi_circle_area = 12.5 * Real.pi :=
  by
    -- Proof steps here
    sorry

end semi_circle_area_correct_l133_133718


namespace increasing_interval_l133_133811

def f (x : ℝ) : ℝ := real.cbrt (x^2)

theorem increasing_interval : ∀ x, 0 < x → ∀ y, x ≤ y → f x ≤ f y :=
by
  intros x hx y hxy
  sorry

end increasing_interval_l133_133811


namespace sqrt_of_product_eq_540_l133_133412

theorem sqrt_of_product_eq_540 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := 
by 
  sorry 

end sqrt_of_product_eq_540_l133_133412


namespace total_photos_l133_133531

def n_pages1 : ℕ := 12
def photos_per_page1 : ℕ := 2
def n_pages2 : ℕ := 9
def photos_per_page2 : ℕ := 3

theorem total_photos : n_pages1 * photos_per_page1 + n_pages2 * photos_per_page2 = 51 := 
by 
  sorry

end total_photos_l133_133531


namespace family_reunion_people_l133_133805

theorem family_reunion_people (pasta_per_person : ℚ) (total_pasta : ℚ) (recipe_people : ℚ) : 
  pasta_per_person = 2 / 7 ∧ total_pasta = 10 -> recipe_people = 35 :=
by
  sorry

end family_reunion_people_l133_133805


namespace complement_N_subset_M_l133_133899

-- Definitions for the sets M and N
def M : Set ℝ := {x | x * (x - 3) < 0}
def N : Set ℝ := {x | x < 1 ∨ x ≥ 3}

-- Complement of N in ℝ
def complement_N : Set ℝ := {x | ¬(x < 1 ∨ x ≥ 3)}

-- The theorem stating that complement_N is a subset of M
theorem complement_N_subset_M : complement_N ⊆ M :=
by
  sorry

end complement_N_subset_M_l133_133899


namespace find_m_l133_133041

def ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + y^2 = 1

def focus1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def focus2 : ℝ × ℝ := (Real.sqrt 2, 0)

def line (x y m : ℝ) : Prop :=
  y = x + m

def intersects (m : ℝ) (x y : ℝ) : Prop :=
  ellipse x y ∧ line x y m

def area_relation (m : ℝ) : Prop :=
  let x := -m -- This might be simplified as it's used to calculate distances
  let dist1 := Real.abs (focus1.fst + m)
  let dist2 := Real.abs (focus2.fst + m)
  dist1 = 2 * dist2

theorem find_m :
  area_relation (-Real.sqrt 2 / 3) :=
by
  sorry

end find_m_l133_133041


namespace area_ratio_ellipse_l133_133079

noncomputable def ellipse := { x // (x.1^2 / 3 + x.2^2 = 1) }
def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))
def intersects (line : ℝ × ℝ) (C : ℝ × ℝ → Prop) := ∃ (x y : ℝ), (C (x, y)) ∧ (y = x + line.2)

theorem area_ratio_ellipse 
  (a b : ℝ) (h_ellipse : a^2 = 3 ∧ b^2 = 1) (m : ℝ) 
  (h_foci : foci a b = ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))) 
  (h_condition : ∀ (x : ℝ), intersects (x, m) (λ pt, pt.1^2 / 3 + pt.2^2 = 1) →
     |((x - real.sqrt (a^2 - b^2)) + m)| = 2|((x + real.sqrt (a^2 - b^2)) + m)|) :
  m = -real.sqrt(2) / 3 := sorry

end area_ratio_ellipse_l133_133079


namespace mathematician_correctness_l133_133252

theorem mathematician_correctness (box1_white1 box1_total1 box2_white1 box2_total1 : ℕ)
                                  (prob1 prob2 : ℚ) :
  (box1_white1 = 4 ∧ box1_total1 = 7 ∧ box2_white1 = 3 ∧ box2_total1 = 5 ∧ prob1 = (4:ℚ) / 7 ∧ prob2 = (3:ℚ) / 5) →
  (box1_white2 = 8 ∧ box1_total2 = 14 ∧ box2_white2 = 3 ∧ box2_total2 = 5 ∧ prob1 = (8:ℚ) / 14 ∧ prob2 = (3:ℚ) / 5) →
  (prob1 < (7:ℚ) / 12 ∧ prob2 > (11:ℚ)/19) →
  ¬((19:ℚ)/35 > (4:ℚ)/7 ∧ (19:ℚ)/35 < (3:ℚ)/5) :=
by
  sorry

end mathematician_correctness_l133_133252


namespace trig_inequality_l133_133142

noncomputable def acute (x : ℝ) := 0 < x ∧ x < π / 2

theorem trig_inequality (α β : ℝ) 
  (hα : acute α) (hβ : acute β) : 
  cos α * sin (2 * α) * sin (2 * β) ≤ (4 * real.sqrt 3) / 9 :=
sorry

end trig_inequality_l133_133142


namespace intersection_sets_l133_133501

open Set

variable {R : Type*} [LinearOrder R]

def A (x : R) : Prop := x + 1 > 0
def B (x : R) : Prop := x * (x - 1) < 0

theorem intersection_sets (x : R) :
  (A x ∧ B x) ↔ (0 < x ∧ x < 1) :=
sorry

end intersection_sets_l133_133501


namespace two_false_propositions_l133_133812

theorem two_false_propositions (a : ℝ) :
  (¬((a > -3) → (a > -6))) ∧ (¬((a > -6) → (a > -3))) → (¬(¬(a > -3) → ¬(a > -6))) :=
by
  sorry

end two_false_propositions_l133_133812


namespace find_area_triangle_ADC_l133_133167

structure Triangle (α : Type) := 
  (A B C D : α)
  (AB BC AC : ℝ)
  (angle_ABC : ℝ)
  (angle_bisector_AD : Prop)

-- Define the specific triangle with the given conditions
def triangle_ABC : Triangle Point := {
  A := 0,
  B := 1,
  C := 2,
  D := 3,
  AB := 90,
  BC := 56,  -- this is found from solution steps
  AC := 106, -- this is found from solution steps
  angle_ABC := 90,
  angle_bisector_AD := true  -- AD is an angle bisector
}

noncomputable def area_triangle_ADC (t : Triangle Point) : ℝ :=
  (1 / 2) * t.AB * (212 / 7)

theorem find_area_triangle_ADC (t : Triangle Point) :
  t = triangle_ABC → area_triangle_ADC t = 1363 :=
by
  intro h
  subst h
  sorry

end find_area_triangle_ADC_l133_133167


namespace range_of_f_l133_133489

noncomputable def f : ℝ → ℝ :=
λ x, if x > 1 then Real.log x + 2 else Real.exp x - 2

theorem range_of_f :
  (Set.range f) = Set.Ioo (-2) (Real.exp 1 - 2) ∪ Set.Ioi 2 := by
sorry

end range_of_f_l133_133489


namespace average_of_first_and_last_l133_133641

theorem average_of_first_and_last (numbers : List ℤ) (largest smallest median : ℤ) 
  (h1 : largest ∈ numbers)
  (h2 : smallest ∈ numbers)
  (hm : median ∈ numbers)
  (h3 : largest > median)
  (h4 : median > smallest)
  (h5 : ∀ first last, sum_of_positions_first_last numbers largest smallest median first last)
  (h6 : numbers.length = 6)
  :
  (∀ first last, condition_4 first last → average_of_first_last first last = 11.5) := 
sorry

end average_of_first_and_last_l133_133641


namespace pure_imaginary_complex_l133_133093

theorem pure_imaginary_complex (a : ℝ) : 
  let i := complex.I in (1 + a * i) * (2 + i) ∈ set_of (λ z : ℂ, z.re = 0) → a = 2 :=
by
  sorry

end pure_imaginary_complex_l133_133093


namespace upper_limit_range_set_W_l133_133224

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the set W to be the set of prime numbers between 10 and n
def set_W (n : ℕ) : Set ℕ := { p | p > 10 ∧ p < n ∧ is_prime p }

-- Define the range of a set
def range_of_set (s : Set ℕ) : ℕ := s.sup' (⟨11, by simp [is_prime]⟩) - s.inf' (⟨11, by simp [is_prime]⟩)

-- Mathematical statement we want to prove
theorem upper_limit_range_set_W (n : ℕ) (h : range_of_set (set_W n) = 12) : n = 23 := by
  sorry

end upper_limit_range_set_W_l133_133224


namespace all_points_X_lie_on_one_circle_l133_133368

open Real

-- Definitions of the problem
def quadrilateral_inscribed_in_circle (A B C D : Point) (Ω : Circle) : Prop :=
  Ω.contains A ∧ Ω.contains B ∧ Ω.contains C ∧ Ω.contains D

def tangent_circles_with_chords (A B C D X : Point) (Ω₁ Ω₂ Ω : Circle) : Prop :=
  (Ω₁.tangent_to Ω₂) ∧ (Ω₁.contains A) ∧ (Ω₁.contains B) ∧ (Ω₂.contains C) ∧ (Ω₂.contains D) ∧ (tangent_point Ω₁ Ω₂ = X)

def points_X_on_circle (A B C D X O : Point) (r : ℝ) : Prop :=
  (O = radical_center A B C D) ∧ (sqrt (O.distance_to A * O.distance_to B) = r) ∧ ((circle centered at O with radius r).contains X)

-- The statement to be proved
theorem all_points_X_lie_on_one_circle (A B C D X O : Point) (Ω Ω₁ Ω₂ : Circle) (r : ℝ) :
  quadrilateral_inscribed_in_circle A B C D Ω →
  tangent_circles_with_chords A B C D X Ω₁ Ω₂ Ω →
  points_X_on_circle A B C D X O r :=
by
  sorry

end all_points_X_lie_on_one_circle_l133_133368


namespace sqrt_of_product_eq_540_l133_133413

theorem sqrt_of_product_eq_540 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := 
by 
  sorry 

end sqrt_of_product_eq_540_l133_133413


namespace find_m_l133_133046

def ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + y^2 = 1

def focus1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def focus2 : ℝ × ℝ := (Real.sqrt 2, 0)

def line (x y m : ℝ) : Prop :=
  y = x + m

def intersects (m : ℝ) (x y : ℝ) : Prop :=
  ellipse x y ∧ line x y m

def area_relation (m : ℝ) : Prop :=
  let x := -m -- This might be simplified as it's used to calculate distances
  let dist1 := Real.abs (focus1.fst + m)
  let dist2 := Real.abs (focus2.fst + m)
  dist1 = 2 * dist2

theorem find_m :
  area_relation (-Real.sqrt 2 / 3) :=
by
  sorry

end find_m_l133_133046


namespace exists_two_people_known_or_unknown_l133_133936

theorem exists_two_people_known_or_unknown (n : ℕ) (h : n ≥ 4) : 
  ∃ A B : Fin n, A ≠ B ∧ 
  (∃ T : Finset (Fin n), T.card = (n - 2) ∧ (∀ C ∈ T, (∃ k : ℤ, k ≥ 0 ∧ k ≤ ((n / 2) : ℤ) - 1 ∧
  ( (∀ D ∈ T, knows C D) ∨ (∀ D ∈ T, ¬ knows C D))))) := 
sorry

end exists_two_people_known_or_unknown_l133_133936


namespace trig_sum_sin_cos_l133_133741

noncomputable def T_k (k : ℕ) (α : List ℝ) : ℝ :=
  (List.pmap (λ (k : Fin k) hx, (α.nth k.1).tan) (Finset.powersetLen k α.toFinset).val)).sum

theorem trig_sum_sin_cos (n : ℕ)
  (α : Fin n → ℝ) :
  (sin (Finset.univ.sum (λ i, α i)) = 
   (Finset.univ.prod (λ i, cos (α i))) * (List.sum (List.pmap (λ (k : ℕ) (hk : k % 2 = 1), T_k k (List.ofFn α)) (List.range (n + 1))))) ∧
  (cos (Finset.univ.sum (λ i, α i)) =
   (Finset.univ.prod (λ i, cos (α i))) * (1 - List.sum (List.pmap (λ (k : ℕ) (hk : k % 2 = 0), T_k k (List.ofFn α)) (List.range (n + 1))))) :=
by 
sorry

end trig_sum_sin_cos_l133_133741


namespace sqrt_expression_l133_133410

theorem sqrt_expression :
  (Real.sqrt (2 ^ 4 * 3 ^ 6 * 5 ^ 2)) = 540 := sorry

end sqrt_expression_l133_133410


namespace sqrt_product_of_powers_eq_l133_133405

theorem sqrt_product_of_powers_eq :
  ∃ (x y z : ℕ), prime x ∧ prime y ∧ prime z ∧ x = 2 ∧ y = 3 ∧ z = 5 ∧
  sqrt (x^4 * y^6 * z^2) = 540 := by
  use 2, 3, 5
  show prime 2, from prime_two
  show prime 3, from prime_three
  show prime 5, from prime_five
  show 2 = 2, from rfl
  show 3 = 3, from rfl
  show 5 = 5, from rfl
  sorry

end sqrt_product_of_powers_eq_l133_133405


namespace sequence_periodic_l133_133420

noncomputable def sequence : ℕ → ℝ
| 0       := 0
| 1       := real.sqrt 2
| (n + 1) := (real.sqrt 3 * sequence n - 1) / (sequence n + real.sqrt 3)

theorem sequence_periodic : ∀ n : ℕ, sequence (n + 6) = sequence n := 
sorry

end sequence_periodic_l133_133420


namespace find_hyperbola_eq_no_fixed_point_M_l133_133496

variables {a b m x y : ℝ}
variables (m_pos: m > 0)
variables (e : ℝ) (he : e = 2)

def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def point_P (x y m : ℝ) : Prop := x = 0 ∧ y = m
def slope_1 (A B : ℝ × ℝ) : Prop := (B.2 - A.2) / (B.1 - A.1) = 1
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ l, slope_1 A B ∧ l = λ x, x - m ∧ hyperbola A.1 A.2 a b ∧ hyperbola B.1 B.2 a b
def vector_relation (A P B : ℝ × ℝ) : Prop := dist A P = 3 * dist P B
def dot_product_3 (O A B : ℝ × ℝ) : Prop := dot_product O A * dot_product O B = 3

theorem find_hyperbola_eq (a b : ℝ) (hyp : hyperbola x y a b) (he : e = 2) :
  hyperbola x y a (sqrt 3 * a) :=
sorry

theorem no_fixed_point_M (Q F M : ℝ × ℝ) (FM_2QMF : angle Q F M = 2 * angle Q M F) :
  ¬(∃ M, M.1 < 0 ∧ FM_2QMF) :=
sorry

end find_hyperbola_eq_no_fixed_point_M_l133_133496


namespace probability_of_C_l133_133362

theorem probability_of_C (P_A P_B P_D : ℚ) (h₁ : P_A = 1/4) (h₂ : P_B = 1/3) (h₃ : P_D = 1/6) :
  let P_C := 1 - (P_A + P_B + P_D) in P_C = 1/4 :=
by
  -- Introduction to proof environment, further proof can be done here.
  sorry

end probability_of_C_l133_133362


namespace log_expression_equality_l133_133745

theorem log_expression_equality : log 2 9 * log 3 4 + 2 ^ log 2 3 = 7 := 
by 
  -- Begin proof sketch:
  -- 1. Express log 2 9 using the change of base formula
  -- 2. Express log 3 4 using the change of base formula
  -- 3. Simplify the product of the two expressions
  -- 4. Simplify 2 ^ log 2 3 to 3
  -- 5. Add the simplified expressions to get 7
  sorry

end log_expression_equality_l133_133745


namespace solution_set_log_inequality_l133_133630

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem solution_set_log_inequality (a : ℝ) (h_max : ∀ x ∈ set.Icc 0 a, f x ≤ 3) (h_min : ∀ x ∈ set.Icc 0 a, -1 ≤ f x) :
  { x : ℝ | log a (x - 1) ≤ 0 } = set.Icc 1 2 :=
by
  sorry

end solution_set_log_inequality_l133_133630


namespace normal_probability_l133_133208

noncomputable def ξ : MeasureTheory.ProbabilityMeasure ℝ :=
  MeasureTheory.ProbabilityMeasure.ofReal (probabilityMeasureOfNormal 1 σ²)

theorem normal_probability ξ : 
  P(ξ < 1) = 1/2 → P(ξ > 2) = p → P(0 < ξ < 1) = 1/2 - p :=
by
  intros h1 h2
  sorry

end normal_probability_l133_133208


namespace inverse_function_proof_l133_133891

-- Define the function f(x) = 1 + a^x, where a is defined by the point (3, 9) lying on the graph of f
def f (a : ℝ) (x : ℝ) : ℝ := 1 + a^x

-- Define the condition that the point (3, 9) lies on the graph of f
def point_on_graph (a : ℝ) : Prop := f a 3 = 9

-- Since the point (3, 9) lies on the graph, we solve for a to get a = 2
def a_val := 2

-- Define the function with the found value of a
def f_with_a (x : ℝ) : ℝ := 1 + a_val^x

-- Define the inverse function we want to prove: f⁻¹(x) = log₂(x - 1)
def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2 - 1

-- State that f_with_a and f_inv are inverses of each other
theorem inverse_function_proof : ∀ x : ℝ, x > 1 → f_inv (f_with_a x) = x ∧ f_with_a (f_inv x) = x := sorry

end inverse_function_proof_l133_133891


namespace find_t_closest_l133_133834

noncomputable def vector_v (t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 5 * t, -2 + 7 * t, -4 - 3 * t)

def vector_a : ℝ × ℝ × ℝ :=
  (5, 2, 7)

def direction_vector : ℝ × ℝ × ℝ :=
  (5, 7, -3)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_t_closest :
  ∃ t : ℝ, t = 5 / 83 ∧ dot_product ((vector_v t).1 - vector_a.1, (vector_v t).2 - vector_a.2, (vector_v t).3 - vector_a.3) direction_vector = 0 :=
begin
  existsi 5 / 83,
  split,
  { refl },
  { sorry }
end

end find_t_closest_l133_133834


namespace find_beta_plus_two_alpha_is_pi_l133_133464

noncomputable def find_beta_plus_two_alpha (α β : ℝ) : ℝ :=
if h : α ∈ (0 : ℝ, Real.pi) ∧ β ∈ (0 : ℝ, Real.pi) ∧ (Real.cos α + Real.cos β - Real.cos (α + β) = 3 / 2) then
  2 * α + β
else
  0

theorem find_beta_plus_two_alpha_is_pi (α β : ℝ) (hα : α ∈ (0 : ℝ, Real.pi))
  (hβ : β ∈ (0 : ℝ, Real.pi)) (hcos : Real.cos α + Real.cos β - Real.cos (α + β) = 3 / 2) :
  find_beta_plus_two_alpha α β = Real.pi :=
by
  rw [find_beta_plus_two_alpha, dif_pos]
  swap
  use ⟨hα, hβ, hcos⟩
  sorry

end find_beta_plus_two_alpha_is_pi_l133_133464


namespace calculate_total_fine_l133_133754

-- Define the initial fine
def initial_fine : ℝ := 0.07

-- Define the daily increment based on the day
def daily_increment (d : ℕ) : ℝ :=
  if d > 1 then 0.05 + 0.05 * (d - 2) else 0

-- Define the total fine recursively with interest rate and daily increment
noncomputable def total_fine (d : ℕ) (r : ℝ) : ℝ :=
  if d = 1 then initial_fine
  else min ((total_fine (d - 1) r) + (daily_increment d)) (2 * (total_fine (d - 1) r)) * (1 + r)

-- Prove the total fine on day d
theorem calculate_total_fine (d : ℕ) (r : ℝ) : ℝ :=
  total_fine d r

end calculate_total_fine_l133_133754


namespace function_shift_monotonicity_g_monotonically_decreasing_on_I_l133_133309

def f (x : ℝ) : ℝ := sin (2 * x) - (sqrt 3) * cos (2 * x)

def g (y : ℝ) : ℝ := 2 * sin (2 * y + (2 * π / 3))

theorem function_shift_monotonicity :
  ∀ x, g(x) = f(x + π / 2) :=
by sorry

theorem g_monotonically_decreasing_on_I :
  ∀ x ∈ Icc (-π / 12) (5 * π / 12), g' x < 0 :=
by sorry

end function_shift_monotonicity_g_monotonically_decreasing_on_I_l133_133309


namespace period_of_cosine_function_l133_133423

theorem period_of_cosine_function : 
  let y (x : ℝ) := cos ((3 * x / 4) + (π / 6)) 
  (∃ T : ℝ, T = 8 * π / 3 ∧ ∀ x, y (x + T) = y x) :=
begin
  sorry,
end

end period_of_cosine_function_l133_133423


namespace find_5_minus_c_l133_133141

theorem find_5_minus_c (c d : ℤ) (h₁ : 5 + c = 6 - d) (h₂ : 3 + d = 8 + c) : 5 - c = 7 := by
  sorry

end find_5_minus_c_l133_133141


namespace total_votes_l133_133384

theorem total_votes (votes_veggies : ℕ) (votes_meat : ℕ) (H1 : votes_veggies = 337) (H2 : votes_meat = 335) : votes_veggies + votes_meat = 672 :=
by
  sorry

end total_votes_l133_133384


namespace problem_solution_l133_133466

def f (x : ℕ) : ℝ := sorry

axiom f_add_eq_mul (p q : ℕ) : f (p + q) = f p * f q
axiom f_one_eq_three : f 1 = 3

theorem problem_solution :
  (f 1 ^ 2 + f 2) / f 1 + 
  (f 2 ^ 2 + f 4) / f 3 + 
  (f 3 ^ 2 + f 6) / f 5 + 
  (f 4 ^ 2 + f 8) / f 7 = 24 := 
by
  sorry

end problem_solution_l133_133466


namespace exists_constants_C1_C2_l133_133848

def s (m : ℕ) : ℕ :=
  -- denote the sum of the decimal digits of m
  sorry

def k_stable (S : set ℕ) (k : ℕ) : Prop :=
  ∀ X ⊆ S, X.nonempty → s (∑ x in X, x) = k

def f (n : ℕ) : ℕ :=
  -- the minimal k for which there exists a k-stable set with n integers
  sorry

theorem exists_constants_C1_C2 (n : ℕ) (h : n ≥ 2) :
  ∃ C1 C2 : ℝ, 0 < C1 ∧ C1 < C2 ∧ C1 * (log 10 n) ≤ f n ∧ f n ≤ C2 * (log 10 n) :=
sorry

end exists_constants_C1_C2_l133_133848


namespace no_real_solutions_l133_133422

-- Define the equation
def equation (x : ℝ) : Prop :=
  (2 * x ^ 2 - 6 * x + 5) ^ 2 + 1 = -|x|

-- Declare the theorem which states there are no real solutions to the given equation
theorem no_real_solutions : ∀ x : ℝ, ¬ equation x :=
by
  intro x
  sorry

end no_real_solutions_l133_133422


namespace num_distinct_solutions_l133_133600

theorem num_distinct_solutions : ∃ (x : ℝ), |x^2 - |2 * x - 3|| = 5 ∧ 
  (Σ' (x : ℝ), x = 1 + Real.sqrt 3 ∨ x = 2) = 2 :=
by
  sorry

end num_distinct_solutions_l133_133600


namespace problem1_correct_problem2_correct_l133_133403

noncomputable def problem1 := 5 + (-6) + 3 - 8 - (-4)
noncomputable def problem2 := -2^2 - 3 * (-1)^3 - (-1) / (-1 / 2)^2

theorem problem1_correct : problem1 = -2 := by
  rw [problem1]
  sorry

theorem problem2_correct : problem2 = 3 := by
  rw [problem2]
  sorry

end problem1_correct_problem2_correct_l133_133403


namespace height_at_D_l133_133388

-- Definitions and assumptions based on the problem conditions
def A : ℝ × ℝ × ℝ := (0, 0, 15)
def B : ℝ × ℝ × ℝ := (5, 0, 12)
def C : ℝ × ℝ × ℝ := (2.5, 2.5 * Real.sqrt 3, 11)
def D : ℝ × ℝ := (0, 5 * Real.sqrt 3)

-- Lean statement to prove the height of pillar at D
theorem height_at_D : ∃ z : ℝ, D = (0, 5 * Real.sqrt 3) ∧ z = 18 :=
by
  sorry

end height_at_D_l133_133388


namespace compute_CP_l133_133189

variables (A B C X P : Type) [ABC : triangle A B C] [X_on_AB : on_side X A B] 
variables (BXC_eq : angle B X C = 60) (P_on_CX : on_segment P C X)
variables (BP_perpendicular_AC : perpendicular B P A C) (AB_eq : dist A B = 6) 
variables (AC_eq : dist A C = 7) (BP_eq : dist B P = 4)

theorem compute_CP : dist C P = sqrt 38 - 3 :=
sorry

end compute_CP_l133_133189


namespace tire_circumference_l133_133339

variable (rpm : ℕ) (car_speed_kmh : ℕ) (circumference : ℝ)

-- Define the conditions
def conditions : Prop :=
  rpm = 400 ∧ car_speed_kmh = 24

-- Define the statement to prove
theorem tire_circumference (h : conditions rpm car_speed_kmh) : circumference = 1 :=
sorry

end tire_circumference_l133_133339


namespace angle_R_parallel_lines_l133_133589

theorem angle_R_parallel_lines
  (p q : Line)
  (h_parallel : p ∥ q)
  (angle_P : ℝ)
  (h_angle_P : angle_P = 100)
  (angle_Q : ℝ)
  (h_angle_Q : angle_Q = 130) :
  ∃ angle_R : ℝ, angle_R = 130 :=
by 
  sorry

end angle_R_parallel_lines_l133_133589


namespace expansion_properties_l133_133484

noncomputable def rational_terms_expansion (n : ℕ) : list (ℝ × ℝ) :=
  let T := λ r : ℕ, (1 / 2) ^ r * real.binom n r * (λ x : ℝ, x ^ ((16 - 3 * r) / 4)) in
  if n = 8 then
    [(4, T 0 1.0), (1, T 4 (35 / 8)), (-2, T 8 (1 / 256))]
  else
    []

noncomputable def max_coeff_terms_expansion (n r : ℕ) (x : ℝ) : ℝ :=
  (1 / 2) ^ r * real.binom n r * x ^ ((16 - 3 * r) / 4)

theorem expansion_properties (x : ℝ) :
  let n := 8 in
  rational_terms_expansion n = [(4, x^4), (1, (35/8)*x), (-2, (1/256)*x^(-2))] ∧
  max_coeff_terms_expansion n 2 (x ^ (5/2)) = 7 * x ^ (5/2) ∧
  max_coeff_terms_expansion n 3 (x ^ (7/4)) = 7 * x ^ (7/4) := by
  sorry

end expansion_properties_l133_133484


namespace ac_eq_af_l133_133969

theorem ac_eq_af
  (Γ₁ Γ₂ : Circle)
  (A D : Point)
  (h₁ : A ∈ Γ₁)
  (h₂ : D ∈ Γ₁)
  (h₃ : A ∈ Γ₂)
  (h₄ : D ∈ Γ₂)
  (B : Point)
  (h₅ : is_tangent Γ₁ (Line.through A B) A)
  (h₆ : B ∈ Γ₂)
  (C : Point)
  (h₇ : is_tangent Γ₂ (Line.through A C) A)
  (h₈ : C ∈ Γ₁)
  (E : Point)
  (h₉ : E ∈ Line.ray A B)
  (h₁₀ : distance B E = distance A B)
  (Ω : Circle)
  (h₁₁ : A ∈ Ω)
  (h₁₂ : D ∈ Ω)
  (h₁₃ : E ∈ Ω)
  (h₁₄ : F : Point)
  (h₁₅ : F ∈ Line.segment A C)
  (h₁₆ : second_inter (Line.segment A C) Ω F) :
  distance A C = distance A F :=
begin
  sorry
end

end ac_eq_af_l133_133969


namespace calculation_correct_l133_133974

def f (x : ℚ) := (2 * x^2 + 6 * x + 9) / (x^2 + 3 * x + 5)
def g (x : ℚ) := 2 * x + 1

theorem calculation_correct : f (g 2) + g (f 2) = 308 / 45 := by
  sorry

end calculation_correct_l133_133974


namespace time_left_for_exercises_l133_133180

theorem time_left_for_exercises (total_minutes : ℕ) (piano_minutes : ℕ) (writing_minutes : ℕ) (reading_minutes : ℕ) : 
  total_minutes = 120 ∧ piano_minutes = 30 ∧ writing_minutes = 25 ∧ reading_minutes = 38 → 
  total_minutes - (piano_minutes + writing_minutes + reading_minutes) = 27 :=
by
  intro h
  cases h with h_total h
  cases h with h_piano h
  cases h with h_writing h_reading
  rw [h_total, h_piano, h_writing, h_reading]
  exactly rfl

end time_left_for_exercises_l133_133180


namespace scheduling_lectures_l133_133759

theorem scheduling_lectures :
  ∃ (n : ℕ), n = 2520 ∧ 
    ∀ (lecturers : list string), lecturers.length = 7 → 
    (∃ (j s b : string), j ∈ lecturers ∧ s ∈ lecturers ∧ b ∈ lecturers ∧ 
      j ≠ s ∧ j ≠ b ∧ s ≠ b ∧
      (lecturers.index_of j < lecturers.index_of s) ∧ 
      (lecturers.index_of j < lecturers.index_of b)) :=
sorry

end scheduling_lectures_l133_133759


namespace people_in_group_l133_133237

theorem people_in_group (n : ℕ) 
  (h1 : ∀ (new_weight old_weight : ℕ), old_weight = 70 → new_weight = 110 → (70 * n + (new_weight - old_weight) = 70 * n + 4 * n)) :
  n = 10 :=
sorry

end people_in_group_l133_133237


namespace exists_non_intersecting_segments_l133_133980

def S : Type := {p : ℝ × ℝ // ∃ (n : ℕ), p ∈ set.range (λ i, (i : ℝ, i : ℝ)) ∧ (2 * n).succ = 2 * i}
variables (n : ℕ) (points : set S)
variables [decidable_pred (λ p : S, ∃ (r : ℝ) (b : ℝ), r != b)]
variables (red_points : fin n → S) (blue_points : fin n → S)

-- Conditions: 
-- 1. points consists of 2n points
axiom no_three_collinear : ∀ (p1 p2 p3 : S), p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬ collinear {p1, p2, p3}
axiom red_and_blue_partition : points = {p // ∃ (i : fin n), p = red_points i ∨ p = blue_points i}.to_set

theorem exists_non_intersecting_segments : 
  ∃ (pairing : fin n → (S × S)), 
  (∀ i, ∃ j, pairing i = (red_points i, blue_points j)) ∧
  ∀ i j, (let (a1, b1) := pairing i in let (a2, b2) := pairing j in a1 ≠ a2 ∧ b1 ≠ b2 → ¬ intersect ⟨a1, b1⟩ ⟨a2, b2⟩) :=
sorry

end exists_non_intersecting_segments_l133_133980


namespace max_beauty_value_is_17_l133_133414

noncomputable def max_beauty_value : ℕ := 17

/-- 
Given 15 circles (vertices) and 20 segments (edges) with each circle containing a number from {0, 1, 2},
define the beauty value of a configuration as the number of segments such that the absolute difference
between the numbers in the connected circles is 1.
-/
def beauty_value (f : Fin 15 → Fin 3) (edges : Fin 20 → (Fin 15 × Fin 15)) : ℕ :=
  (Fin 20).univ.count (λ e => |f (edges e).1 - f (edges e).2| = 1)

theorem max_beauty_value_is_17 (f : Fin 15 → Fin 3) (edges : Fin 20 → (Fin 15 × Fin 15)) :
  ∃! N, (∀ g : Fin 15 → Fin 3, beauty_value g edges ≤ max_beauty_value) ∧ ∃ h : Fin 15 → Fin 3, beauty_value h edges = max_beauty_value ∧
  N = (Fin 15 → Fin 3).univ.count (λ g => beauty_value g edges = max_beauty_value) :=
sorry

end max_beauty_value_is_17_l133_133414


namespace initial_pipes_count_l133_133601

theorem initial_pipes_count (n r : ℝ) 
  (h1 : n * r = 1 / 12) 
  (h2 : (n + 10) * r = 1 / 4) : 
  n = 5 := 
by 
  sorry

end initial_pipes_count_l133_133601


namespace integral1_integral2_integral3_integral4_l133_133826

-- Proof problem 1
theorem integral1 :
  ∫ (x : ℝ) in set.univ, 1 / (3 * x^2 - 6 * x + 5) = (sqrt 6 / 6) * arctan ((sqrt 6 / 2) * (x - 1)) + C :=
by sorry

-- Proof problem 2
theorem integral2 :
  ∫ (x : ℝ) in set.univ, (x + 3) / (x^2 + 4 * x - 1) = 
    (1 / 2) * log (abs (x^2 + 4 * x - 1)) + (1 / (2 * sqrt 5)) * log (abs ((x + 2 - sqrt 5) / (x + 2 + sqrt 5))) + C :=
by sorry

-- Proof problem 3
theorem integral3 :
  ∫ (x : ℝ) in set.univ, (x^3 - 2 * x^2 + x - 1) / (3 + 2 * x - x^2) = 
    ∫ (x : ℝ) in set.univ, (-x + (4 * x / (3 + 2 * x - x^2))) :=
by sorry

-- Proof problem 4
theorem integral4 :
  ∫ (x : ℝ) in set.univ, (2 * x - 1) / (x^2 - 2 * x) = log (abs (x^2 - 2 * x)) + (1 / 2) * log (abs ((x - 2) / x)) + C :=
by sorry

end integral1_integral2_integral3_integral4_l133_133826


namespace four_digit_numbers_with_conditions_l133_133136

-- Conditions
def is_prime_digit (d : ℕ) : Prop := d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7
def is_even_digit (d : ℕ) : Prop := d % 2 = 0
def are_all_digits_different (d1 d2 d3 d4 : ℕ) : Prop := d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

-- Main statement
theorem four_digit_numbers_with_conditions :
  (∑ d1 in {d | is_prime_digit d}, 
   ∑ d2 in {d | is_even_digit d},
   ∑ d3 in {d | d ≠ d1 ∧ d ≠ d2},
   ∑ d4 in {d | d ≠ d1 ∧ d ≠ d2 ∧ d ≠ d3},
   1) = 1064 :=
by sorry

end four_digit_numbers_with_conditions_l133_133136


namespace find_y_l133_133161

variable (A B C D X Y Z : Type) [IsPoint A] [IsPoint B] [IsPoint C]
  [IsPoint D] [IsPoint X] [IsPoint Y] [IsPoint Z]

variable (angle_AXB angle_BXD angle_CYD angle_YXZ angle_XYZ y : ℝ)

-- Define angles
axiom angle_AXB_def : angle_AXB = 70
axiom angle_BXD_def : angle_BXD = 80
axiom angle_CYD_def : angle_CYD = 140

-- Define angle YXZ and angle XYZ based on given conditions and geometry
axiom angle_YXZ_def : angle_YXZ = 180 - angle_AXB - angle_BXD
axiom angle_XYZ_def : angle_XYZ = 180 - angle_CYD

-- Define angle sum property of triangle
axiom triangle_sum_angles : ∀ angle_YXZ angle_XYZ y, angle_YXZ + angle_XYZ + y = 180

theorem find_y (h1: angle_AXB = 70) (h2: angle_BXD = 80) (h3: angle_CYD = 140)
  (h4: angle_YXZ = 30) (h5: angle_XYZ = 40) : y = 110 :=
by
  rw [←angle_YXZ_def, ←angle_AXB_def, ←angle_BXD_def] at h4
  rw [←angle_XYZ_def, ←angle_CYD_def] at h5
  exact triangle_sum_angles y 110 sorry

end find_y_l133_133161


namespace domain_of_f_l133_133809

noncomputable def f (x : ℝ) : ℝ := (x^2 - 16) / (x - 8)

theorem domain_of_f :
  ∀ x : ℝ, x ∈ (set.Iio 8 ∪ set.Ioi 8) :=
begin
  intro x,
  split,
  { intro h,
    simp [f],
    split_ifs,
    { rcases h with h_lt|h_gt,
      exact h_lt,
      exact h_gt } },
  { cases h,
    { exact or.inl (h.1 h.2) },
    { exact or.inr (h.1 h.2) } }
end

end domain_of_f_l133_133809


namespace mathematicians_correctness_l133_133270

theorem mathematicians_correctness :
  ∃ (scenario1_w1 s_w1 : ℕ) (scenario1_w2 s_w2 : ℕ) (scenario2_w1 s2_w1 : ℕ) (scenario2_w2 s2_w2 : ℕ),
    scenario1_w1 = 4 ∧ s_w1 = 7 ∧ scenario1_w2 = 3 ∧ s_w2 = 5 ∧
    scenario2_w1 = 8 ∧ s2_w1 = 14 ∧ scenario2_w2 = 3 ∧ s2_w2 = 5 ∧
    let total_white1 := scenario1_w1 + scenario1_w2,
        total_choco1 := s_w1 + s_w2,
        prob1 := (total_white1 : ℚ) / total_choco1,
        total_white2 := scenario2_w1 + scenario2_w2,
        total_choco2 := s2_w1 + s2_w2,
        prob2 := (total_white2 : ℚ) / total_choco2,
        prob_box1 := 4 / 7,
        prob_box2 := 3 / 5 in
    (prob1 = 7 / 12 ∧ prob2 = 11 / 19 ∧
    (19 / 35 < prob_box1 ∧ prob_box1 < prob_box2) ∧
    (prob_box1 ≠ 19 / 35 ∧ prob_box1 ≠ 3 / 5)) :=
sorry

end mathematicians_correctness_l133_133270


namespace not_right_triangle_l133_133517

variables {A B C : Type} [Field A] [Field B] [Field C]

-- Angle sum in a triangle is 180°
axiom angle_sum (α β γ : Real) : α + β + γ = 180

-- Defining the conditions
def cond1 (a b c : Real) : Prop := b^2 = (a + c) * (a - c)
def cond2 (a b c : Real) : Prop := a / 1 = b / Real.sqrt 3 ∧ b / Real.sqrt 3 = c / 2
def cond3 (α β γ : Real) : Prop := γ = α - β
def cond4 (α β γ : Real) : Prop := (α / 3 = β / 4 ∧ β / 4 = γ / 5)

-- The proof problem statement
theorem not_right_triangle {a b c : Real} {α β γ : Real} :
  ¬ ∃ (cond1 a b c) ∨ ∃ (cond2 a b c) ∨ ∃ (cond3 α β γ) ∨ ∃ (cond4 α β γ), 
  α + β + γ = 180 ∧ γ < 90 :=
sorry

end not_right_triangle_l133_133517


namespace sum_simplification_l133_133332

theorem sum_simplification (n : ℕ) :
  (∑ k in Finset.range n, 1 / ((k + 2) * (k)!)) = 1 - (1 / ((n + 1)!)) := sorry

end sum_simplification_l133_133332


namespace acceleration_at_90_seconds_l133_133280

theorem acceleration_at_90_seconds
    (v_60 v_120 : ℝ)
    (hv60 : v_60 = 20)
    (hv120 : v_120 = 30) :
    let Δv := v_120 - v_60 in
    let Δt := 120 - 60 in
    let a := Δv / Δt in
    a ≈ 0.33 :=
by
  let Δv := v_120 - v_60
  let Δt := 120 - 60
  let a := Δv / Δt
  have h : a = 10 / 60 := by sorry  -- this skips the detailed proof steps
  have approx : 10 / 60 ≈ 0.33 := by sorry  -- this approximates to 0.33
  exact approx

end acceleration_at_90_seconds_l133_133280


namespace gcf_of_three_digit_palindromes_is_one_l133_133689

-- Define a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define the greatest common factor (gcd) function
def gcd_of_all_palindromes : ℕ :=
  (Finset.range 999).filter is_palindrome |>.list.foldr gcd 0

-- State the theorem
theorem gcf_of_three_digit_palindromes_is_one :
  gcd_of_all_palindromes = 1 :=
sorry

end gcf_of_three_digit_palindromes_is_one_l133_133689


namespace trig_identity_1_trig_identity_2_l133_133349

-- statement for part (1)
theorem trig_identity_1 : 
  sin (25 * Real.pi / 6) + cos (25 * Real.pi / 3) + tan (-25 * Real.pi / 4) = 0 := 
by sorry

-- statement for part (2)
theorem trig_identity_2 (α : ℝ) : 
  (sin (5 * Real.pi - α) * cos (α + 3 * Real.pi / 2) * cos (Real.pi + α)) / 
  (sin (α - 3 * Real.pi / 2) * cos (α + Real.pi / 2) * tan (α - 3 * Real.pi)) = 
  cos α := 
by sorry

end trig_identity_1_trig_identity_2_l133_133349


namespace gcf_of_all_three_digit_palindromes_l133_133714

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ is_palindrome n

def gcf_of_palindromes : ℕ :=
  101

theorem gcf_of_all_three_digit_palindromes : 
  ∀ n, is_three_digit_palindrome n → 101 ∣ n := by
    sorry

end gcf_of_all_three_digit_palindromes_l133_133714


namespace number_of_terms_in_arithmetic_sequence_l133_133509

theorem number_of_terms_in_arithmetic_sequence :
  ∀ (a d l : ℤ), a = -48 → d = 6 → l = 78 → (l - a) / d + 1 = 22 :=
by
  intros a d l ha hd hl
  rw [ha, hd, hl]
  norm_num
  sorry

end number_of_terms_in_arithmetic_sequence_l133_133509


namespace gcf_of_three_digit_palindromes_is_one_l133_133683

-- Define a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define the greatest common factor (gcd) function
def gcd_of_all_palindromes : ℕ :=
  (Finset.range 999).filter is_palindrome |>.list.foldr gcd 0

-- State the theorem
theorem gcf_of_three_digit_palindromes_is_one :
  gcd_of_all_palindromes = 1 :=
sorry

end gcf_of_three_digit_palindromes_is_one_l133_133683


namespace inequality_xyz_l133_133959

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x * y / z) + (y * z / x) + (z * x / y) > 2 * ((x ^ 3 + y ^ 3 + z ^ 3) ^ (1 / 3)) :=
by
  sorry

end inequality_xyz_l133_133959


namespace inequality_for_positive_reals_l133_133089

variable {a b c : ℝ}
variable {k : ℕ}

theorem inequality_for_positive_reals 
  (hab : a > 0) 
  (hbc : b > 0) 
  (hac : c > 0) 
  (hprod : a * b * c = 1) 
  (hk : k ≥ 2) 
  : (a ^ k) / (a + b) + (b ^ k) / (b + c) + (c ^ k) / (c + a) ≥ 3 / 2 := 
sorry

end inequality_for_positive_reals_l133_133089


namespace optimal_floor_optimal_floor_achieved_at_three_l133_133760

theorem optimal_floor : ∀ (n : ℕ), n > 0 → (n + 9 / n : ℝ) ≥ 6 := sorry

theorem optimal_floor_achieved_at_three : ∃ n : ℕ, (n > 0 ∧ (n + 9 / n : ℝ) = 6) := sorry

end optimal_floor_optimal_floor_achieved_at_three_l133_133760


namespace exists_a_l133_133557

noncomputable def func (a : ℝ) (x : ℝ) : ℝ := sin x ^ 2 + a * cos x + a

theorem exists_a : ∃ a : ℝ, ∀ x ∈ set.Icc 0 real.pi, func a x ≤ 1 ∧ (∃ y ∈ set.Icc 0 real.pi, func a y = 1) :=
by
  let a := 1
  existsi a
  split
  intro x hx
  sorry
  existsi (0 : ℝ)
  split
  simp
  sorry

end exists_a_l133_133557


namespace truncated_cube_volume_l133_133756

theorem truncated_cube_volume (a : ℝ) : 
  let x := a / (2 + Real.sqrt 2)
  let removed_volume := 8 * (1/6) * x^3 
  let total_volume := a^3
  let final_volume := total_volume - removed_volume
  (20 - 14 * Real.sqrt 2) = (284 - 198 * Real.sqrt 2)
  final_volume = (7 / 3) * a^3 * (Real.sqrt 2 - 1) :=
by {
  sorry
}

end truncated_cube_volume_l133_133756


namespace gcf_of_three_digit_palindromes_is_one_l133_133687

-- Define a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define the greatest common factor (gcd) function
def gcd_of_all_palindromes : ℕ :=
  (Finset.range 999).filter is_palindrome |>.list.foldr gcd 0

-- State the theorem
theorem gcf_of_three_digit_palindromes_is_one :
  gcd_of_all_palindromes = 1 :=
sorry

end gcf_of_three_digit_palindromes_is_one_l133_133687


namespace find_angle_C_find_side_a_l133_133926

theorem find_angle_C (c b : ℝ) (C B : ℝ) 
  (h1 : c = 2) 
  (h2 : (c, sqrt 3 * b) = (cos C, sin B)) 
  (h3 : sqrt 3 * b * cos C - c * sin B = 0) : 
  C = π / 3 := 
sorry

theorem find_side_a (A B a b : ℝ)
  (h1 : (sin (A + B), sin (2 * A), sin (B - A)) ∈ Set.Icc (0 : ℝ) (1 : ℝ))
  (h2 : 2 * sin (2 * A) = sin (A + B) + sin (B - A))
  (h3 : 2 * a = b ∨ cos A = 0)
  (h4 : c = 2) :
  a = (4 * sqrt 3)/3 ∨ a = (2 * sqrt 3)/3 := 
sorry

end find_angle_C_find_side_a_l133_133926


namespace sequence_harmonic_mean_harmonic_mean_symmetric_l133_133343

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, a k ≠ 0 → a (k-1) ≠ 0 → a (k+1) ≠ 0 → 
  (a k)⁻¹ = ((a (k-1))⁻¹ + (a (k+1))⁻¹) / 2

noncomputable def nth_term (a1 a2 : ℝ) (n : ℕ) : ℝ :=
  a1 * a2 / ((n-1) * a1 - (n-2) * a2)

theorem sequence_harmonic_mean {a1 a2 : ℝ} (n : ℕ) (h1 : a1 ≠ 0) (h2 : a2 ≠ 0)
  (a : ℕ → ℝ) (h_seq : sequence a) :
  a n = nth_term a1 a2 n :=
sorry

theorem harmonic_mean_symmetric {a1 a2 : ℝ} (n i : ℕ) (h1 : a1 ≠ 0) (h2 : a2 ≠ 0)
  (a : ℕ → ℝ) (h_seq : sequence a) :
  (a (n-i))⁻¹ = (2 * (a n)⁻¹ - (a (n+i))⁻¹) :=
sorry

end sequence_harmonic_mean_harmonic_mean_symmetric_l133_133343


namespace global_chess_tournament_total_games_global_chess_tournament_player_wins_l133_133528

theorem global_chess_tournament_total_games (num_players : ℕ) (h200 : num_players = 200) :
  (num_players * (num_players - 1)) / 2 = 19900 := by
  sorry

theorem global_chess_tournament_player_wins (num_players losses : ℕ) 
  (h200 : num_players = 200) (h30 : losses = 30) :
  (num_players - 1) - losses = 169 := by
  sorry

end global_chess_tournament_total_games_global_chess_tournament_player_wins_l133_133528


namespace julia_ink_containers_l133_133567

-- Definitions based on conditions
def total_posters : Nat := 60
def posters_remaining : Nat := 45
def lost_containers : Nat := 1

-- Required to be proven statement
theorem julia_ink_containers : 
  (total_posters - posters_remaining) = 15 → 
  posters_remaining / 15 = 3 := 
by 
  sorry

end julia_ink_containers_l133_133567


namespace find_k_l133_133901

-- Definition of vectors a and b
def vec_a (k : ℝ) : ℝ × ℝ := (-1, k)
def vec_b : ℝ × ℝ := (3, 1)

-- Definition of dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Property of perpendicular vectors (dot product is zero)
def is_perpendicular (u v : ℝ × ℝ) : Prop := dot_product u v = 0

-- Problem statement
theorem find_k (k : ℝ) :
  is_perpendicular (vec_a k) (vec_a k) →
  (k = -2 ∨ k = 1) :=
sorry

end find_k_l133_133901


namespace pyramid_height_l133_133370

noncomputable def perimeter : ℝ := 40
noncomputable def apex_distance : ℝ := 15
noncomputable def square_side : ℝ := perimeter / 4
noncomputable def half_diagonal : ℝ := (Math.sqrt (square_side ^ 2 + square_side ^ 2)) / 2

theorem pyramid_height :
  let pyramid_height := Math.sqrt (apex_distance ^ 2 - half_diagonal ^ 2)
  pyramid_height = 5 * Math.sqrt 7 := by
    sorry

end pyramid_height_l133_133370


namespace gcd_of_all_three_digit_palindromes_is_one_l133_133695

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define a function to calculate the gcd of a list of numbers
def gcd_list (l : List ℕ) : ℕ :=
  l.foldr Nat.gcd 0

-- The main theorem that needs to be proven
theorem gcd_of_all_three_digit_palindromes_is_one :
  gcd_list (List.filter is_palindrome {n | 100 ≤ n ∧ n ≤ 999}.toList) = 1 :=
by
  sorry

end gcd_of_all_three_digit_palindromes_is_one_l133_133695


namespace find_m_l133_133070

noncomputable def ellipse : set (ℝ × ℝ) := { p | (p.1)^2 / 3 + (p.2)^2 = 1 }
def f1 : ℝ × ℝ := (- real.sqrt 2, 0)
def f2 : ℝ × ℝ := (real.sqrt 2, 0)

def line (m : ℝ) : set (ℝ × ℝ) := { p | p.2 = p.1 + m }

theorem find_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line m ∧ B ∈ line m ∧
   2 * (abs ((f1.1 * (A.2 - B.2) + A.1 * (B.2 - f1.2) + B.1 * (f1.2 - A.2)) / 2))
 = abs ((f2.1 * (A.2 - B.2) + A.1 * (B.2 - f2.2) + B.1 * (f2.2 - A.2)) / 2)) →
  m = - real.sqrt 2 / 3 :=
sorry

end find_m_l133_133070


namespace general_formula_l133_133551

-- Define the conditions of the sequence
def a : ℕ → ℤ
| 1     := 5
| (n+1) := a n + 3

-- State the theorem that needs to be proved
theorem general_formula (n : ℕ) : a n = 3 * n + 2 :=
sorry

end general_formula_l133_133551


namespace simplest_quadratic_radical_l133_133331

-- Define the given conditions
def option_A := Real.sqrt 8 = 2 * Real.sqrt 2
def option_B := Real.sqrt (1 / 3) = Real.sqrt 3 / 3
def option_C := Real.sqrt 6 = Real.sqrt 6
def option_D := Real.sqrt 0.1 = Real.sqrt 10 / 10

-- Theorem statement: \( \sqrt{6} \) (option C) is the simplest quadratic radical
theorem simplest_quadratic_radical :
  (√6 = √6) ∧ 
  (√8 ≠ √{simplest}) ∧ 
  (√(1/3) ≠ √{simplest}) ∧ 
  (√0.1 ≠ √{simplest}) :=
sorry

end simplest_quadratic_radical_l133_133331


namespace two_triangles_with_complementary_angles_are_right_triangles_l133_133665

-- Definitions based on the conditions:
def are_complementary (α β : ℝ) : Prop := α + β = 90

def is_right_triangle (T : Type) [triangle T] : Prop :=
  ∃ (A B C : T), (angle A B C = 90) ∨ (angle A C B = 90) ∨ (angle B A C = 90)

-- The final theorem to state the proof problem:
theorem two_triangles_with_complementary_angles_are_right_triangles 
  (T1 T2 : Type) [triangle T1] [triangle T2] 
  (α₁ β₁ α₂ β₂ : ℝ) 
  (h1 : are_complementary α₁ β₁)
  (h2 : are_complementary α₂ β₂)
  (h_sum1 : α₁ + β₁ + γ₁ = 180)
  (h_sum2 : α₂ + β₂ + γ₂ = 180) :
  is_right_triangle T1 ∧ is_right_triangle T2 :=
sorry

end two_triangles_with_complementary_angles_are_right_triangles_l133_133665


namespace jim_reads_less_hours_l133_133563

-- Conditions
def initial_speed : ℕ := 40 -- pages per hour
def initial_pages_per_week : ℕ := 600 -- pages
def speed_increase_factor : ℚ := 1.5
def new_pages_per_week : ℕ := 660 -- pages

-- Calculations based on conditions
def initial_hours_per_week : ℚ := initial_pages_per_week / initial_speed
def new_speed : ℚ := initial_speed * speed_increase_factor
def new_hours_per_week : ℚ := new_pages_per_week / new_speed

-- Theorem Statement
theorem jim_reads_less_hours :
  initial_hours_per_week - new_hours_per_week = 4 :=
  sorry

end jim_reads_less_hours_l133_133563


namespace line_intersects_circle_exists_tangent_line_to_circle_l133_133878

def circle (q : ℝ) : set (ℝ × ℝ) :=
{ p | (p.1 + real.cos q) ^ 2 + (p.2 - real.sin q) ^ 2 = 1 }

def line (k : ℝ) : set (ℝ × ℝ) :=
{ p | p.2 = k * p.1 }

theorem line_intersects_circle (k q : ℝ) : 
  ∃ x y : ℝ, (x, y) ∈ circle q ∧ (x, y) ∈ line k :=
sorry

theorem exists_tangent_line_to_circle (k : ℝ) : 
  ∃ q : ℝ, ∃ x y : ℝ, (x, y) ∈ circle q ∧ (x, y) ∈ line k ∧ 
  ∃ t : ℝ, (x, y) = (t, k * t) ∧ ∀ ε > 0, ¬(∃ u v : ℝ, 
  u ≠ t ∧ (u, v) ∈ circle q ∧ (u, v) ∈ line k ∧ u ∈ set.Ioo (t - ε) (t + ε)) :=
sorry

end line_intersects_circle_exists_tangent_line_to_circle_l133_133878


namespace base_seven_to_ten_l133_133668

theorem base_seven_to_ten :
  (6 * 7^4 + 5 * 7^3 + 2 * 7^2 + 3 * 7^1 + 4 * 7^0) = 16244 :=
by sorry

end base_seven_to_ten_l133_133668


namespace sum_of_squares_first_50_even_integers_l133_133003

theorem sum_of_squares_first_50_even_integers:
  ∑ i in finset.range 51, (2 * i)^2 = 171700 :=
by
  sorry

end sum_of_squares_first_50_even_integers_l133_133003


namespace gcf_of_three_digit_palindromes_is_one_l133_133686

-- Define a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define the greatest common factor (gcd) function
def gcd_of_all_palindromes : ℕ :=
  (Finset.range 999).filter is_palindrome |>.list.foldr gcd 0

-- State the theorem
theorem gcf_of_three_digit_palindromes_is_one :
  gcd_of_all_palindromes = 1 :=
sorry

end gcf_of_three_digit_palindromes_is_one_l133_133686


namespace quadratic_solution_unique_l133_133612

noncomputable def solve_quad_eq (a : ℝ) (h : a ≠ 0) (h_uniq : (36^2 - 4 * a * 12) = 0) : ℝ :=
-2 / 3

theorem quadratic_solution_unique (a : ℝ) (h : a ≠ 0) (h_uniq : (36^2 - 4 * a * 12) = 0) :
  (∃! x : ℝ, a * x^2 + 36 * x + 12 = 0) ∧ (solve_quad_eq a h h_uniq) = -2 / 3 :=
by
  sorry

end quadratic_solution_unique_l133_133612


namespace total_photos_l133_133532

def n_pages1 : ℕ := 12
def photos_per_page1 : ℕ := 2
def n_pages2 : ℕ := 9
def photos_per_page2 : ℕ := 3

theorem total_photos : n_pages1 * photos_per_page1 + n_pages2 * photos_per_page2 = 51 := 
by 
  sorry

end total_photos_l133_133532


namespace probability_r25_to_r35_l133_133235

theorem probability_r25_to_r35 (n : ℕ) (r : Fin n → ℕ) (h : n = 50) 
  (distinct : ∀ i j : Fin n, i ≠ j → r i ≠ r j) : 1 + 1260 = 1261 :=
by
  sorry

end probability_r25_to_r35_l133_133235


namespace fish_population_estimate_l133_133751

/-- A biologist tags 60 fish and releases them on May 1. On September 1, a sample of 70 fish includes 
3 tagged fish. 25% of tagged fish from May 1 are presumed dead or migrated by September 1. 
40% of fish caught in September were not in the lake on May 1. 
Prove the estimated total number of fish in the lake on May 1 is 840. -/
theorem fish_population_estimate :
  ∃ (N : ℕ), 
    let tagged_fish_may := 60 in
    let september_sample := 70 in
    let tagged_fish_september := 3 in
    let fish_may_present_september := 0.60 * september_sample in
    N = (tagged_fish_may * fish_may_present_september) / tagged_fish_september ∧ N = 840 :=
begin
  sorry
end

end fish_population_estimate_l133_133751


namespace downstream_speed_l133_133355

noncomputable def V_b : ℝ := 7
noncomputable def V_up : ℝ := 4
noncomputable def V_s : ℝ := V_b - V_up

theorem downstream_speed :
  V_b + V_s = 10 := sorry

end downstream_speed_l133_133355


namespace gcf_of_all_three_digit_palindromes_l133_133715

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ is_palindrome n

def gcf_of_palindromes : ℕ :=
  101

theorem gcf_of_all_three_digit_palindromes : 
  ∀ n, is_three_digit_palindrome n → 101 ∣ n := by
    sorry

end gcf_of_all_three_digit_palindromes_l133_133715


namespace triangle_area_ABC_is_32_l133_133437

-- Define the vertices of the triangle
variables (A B C : Point)
variables (lengthAB lengthAC : ℝ)
variables (angleBAC : ℝ)
variables (is_isosceles : triangle_is_isosceles A B C)
variables (angleBAC_is_45_degrees : angleBAC = (π / 4))
variables (lengthAB_eq_8 : lengthAB = 8)
variables (lengthAC_eq_8 : lengthAC = 8)

-- State the theorem to prove
theorem triangle_area_ABC_is_32 :
  area_of_triangle A B C = 32 :=
sorry

end triangle_area_ABC_is_32_l133_133437


namespace unique_valid_arrangement_l133_133213

-- Define the peg colors
inductive Color
| Black
| White
| Red
| Green
| Blue
| Yellow

-- Define the board and peg placement conditions
structure PegBoard := 
  (rows: Fin 6)   -- 6 rows
  (cols: Fin 6)   -- 6 columns

-- Definitions of the constraints
def pegs : List (Color × Nat) := [(Color.Black, 6), (Color.White, 5), (Color.Red, 4), (Color.Green, 3), (Color.Blue, 2), (Color.Yellow, 1)]

-- Function to check placement constraints
def validPlacement (placements: List (Color × PegBoard)) : Prop :=
  ∀ c r1 r2, c ∈ [Color.Black, Color.White, Color.Red, Color.Green, Color.Blue, Color.Yellow] →
  r1 ≠ r2 → ∃b1 b2, (c, b1) ∈ placements ∧ (c, b2) ∈ placements ∧ 
  (b1.rows ≠ b2.rows ∧ b1.cols ≠ b2.cols ∧ abs (b1.rows - b2.rows) ≠ abs(b1.cols - b2.cols))

-- The theorem stating there is exactly one valid arrangement
theorem unique_valid_arrangement : ∃! (placements : List (Color × PegBoard)), validPlacement placements :=
sorry

end unique_valid_arrangement_l133_133213


namespace find_polynomials_l133_133191

noncomputable def P := polynomial ℤ

def satisfies_conditions (P : P) (p : ℕ) (h_prime : nat.prime p) : Prop :=
  (∀ x : ℕ, x > 0 → P.eval (↑x) > x) ∧
  (∀ m : ℕ, m > 0 → ∃ l : ℕ, l ≥ 0 ∧ m ∣ (nat.iterate P.eval (↑l) p))

theorem find_polynomials (P : P) (p : ℕ) (h_prime : nat.prime p) :
  satisfies_conditions P p h_prime ↔ P = polynomial.C 1 + polynomial.X ∨ P = polynomial.C p + polynomial.X :=
sorry

end find_polynomials_l133_133191


namespace prime_solution_l133_133228

theorem prime_solution (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) 
  (h1 : 2 * a - b + 7 * c = 1826) (h2 : 3 * a + 5 * b + 7 * c = 2007) :
  a = 7 ∧ b = 29 ∧ c = 263 :=
by
  sorry

end prime_solution_l133_133228


namespace gcd_three_digit_palindromes_l133_133707

theorem gcd_three_digit_palindromes : 
  GCD (set.image (λ (p : ℕ × ℕ), 101 * p.1 + 10 * p.2) 
    ({a | a ≠ 0 ∧ a < 10} × {b | b < 10})) = 1 := 
by
  sorry

end gcd_three_digit_palindromes_l133_133707


namespace gcd_three_digit_palindromes_l133_133703

open Nat

theorem gcd_three_digit_palindromes :
  (∀ a b : ℕ, a ≠ 0 → a < 10 → b < 10 → True) ∧
  let S := {n | ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b} in
  S.Gcd = 1 := by
  sorry

end gcd_three_digit_palindromes_l133_133703


namespace greatest_common_factor_of_three_digit_palindromes_l133_133670

def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b

def gcf (a b : ℕ) : ℕ := 
  if a = 0 then b else gcf (b % a) a

theorem greatest_common_factor_of_three_digit_palindromes : 
  ∃ g, (∀ n, is_palindrome n → g ∣ n) ∧ (∀ d, (∀ n, is_palindrome n → d ∣ n) → d ∣ g) :=
by
  use 101
  sorry

end greatest_common_factor_of_three_digit_palindromes_l133_133670


namespace exist_valid_circle_l133_133087

-- Define a structure for a point in the Euclidean plane
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define a predicate for collinearity of three points
def collinear (A B C : Point) : Prop :=
(A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y) = 0)

-- Define a predicate for a circle passing through three points and containing no other points inside
def valid_circle (P : Point → Prop) (A B C : Point) : Prop :=
(P A ∧ P B ∧ P C) ∧ ∀ D, P D → D = A ∨ D = B ∨ D = C ∨ ¬ ∃ k : ℝ, (A.x - k) * (A.x - k) + (A.y - k) * (A.y - k) < (D.x - k) * (D.x - k) + (D.y - k) * (D.y - k)

-- The main theorem
theorem exist_valid_circle (P : Point → Prop) :
  (∃ A B C, ¬ collinear A B C ∧ P A ∧ P B ∧ P C) →
  ∃ A B C, ¬ collinear A B C ∧ valid_circle P A B C :=
begin
  intros h,
  sorry -- the proof is omitted
end

end exist_valid_circle_l133_133087


namespace shirts_for_21_profit_l133_133997

variable (S P : ℝ)

-- Conditions
def condition1 := S = 21
def condition2 := 2 * P = 4 * S
def condition3 := 7 * S + 3 * P = 175

-- Question
theorem shirts_for_21_profit (h1 : condition1) (h2 : condition2) (h3 : condition3) : 21 / S = 1 :=
by sorry

end shirts_for_21_profit_l133_133997


namespace log_identity_l133_133512

theorem log_identity (m y : ℝ) (h : log m y * log 3 m = 4) : y = 81 :=
sorry

end log_identity_l133_133512


namespace gcd_three_digit_palindromes_l133_133680

theorem gcd_three_digit_palindromes : ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → Nat.gcd (102 * a + 10 * b) 1 = 1 :=
by
  intros a b h
  sorry

end gcd_three_digit_palindromes_l133_133680


namespace range_of_k_l133_133485

theorem range_of_k (k : ℝ) : (∀ x : ℝ, 2 * k * x^2 + k * x + 1 / 2 ≥ 0) → k ∈ Set.Ioc 0 4 := 
by 
  sorry

end range_of_k_l133_133485


namespace ellipse_area_condition_l133_133074

open Real

theorem ellipse_area_condition (m : ℝ) :
  (∀ x y : ℝ, ((x^2)/3 + y^2 = 1) → (y = x + m) → 
  let F₁ := (-sqrt 2, 0)
      F₂ := (sqrt 2, 0)
  in distance F₁ (x, y) = 2 * distance F₂ (x, y)) → 
  m = -sqrt 2 / 3 :=
by
  sorry

end ellipse_area_condition_l133_133074


namespace constant_value_P_l133_133148

theorem constant_value_P (x : ℝ) (h1 : 1 / 10 < x) (h2 : x < 1 / 2) :
  let P := |1 - 2 * x| + |1 - 3 * x| + |1 - 4 * x| + |1 - 5 * x| + 
           |1 - 6 * x| + |1 - 7 * x| + |1 - 8 * x| + |1 - 9 * x| + |1 - 10 * x|
  in P = 3 :=
sorry

end constant_value_P_l133_133148


namespace ellipse_intersection_area_condition_l133_133026

theorem ellipse_intersection_area_condition :
  let ellipse := ∀ (x y : ℝ), (x^2 / 3 + y^2 = 1)
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let line m := ∀ (x y : ℝ), (y = x + m)
  ∀ (A B : ℝ × ℝ),
    ((A.1^2 / 3 + A.2^2 = 1) ∧ (y = A.1 + m)) ∧
    ((B.1^2 / 3 + B.2^2 = 1) ∧ (y = B.1 + m)) ∧
    (area (F1, A, B) = 2 * area (F2, A, B)) →
    m = -Real.sqrt 2 / 3 := by sorry

end ellipse_intersection_area_condition_l133_133026


namespace besector_ac_l133_133240

-- Definition of the problem in Lean language
theorem besector_ac (ABC : Triangle) (A B C : Point) (A1 C1 I P : Point)
  (h_angle_B : ∠ B = 60°)
  (h_AA1 : is_angle_bisector A A1)
  (h_CC1 : is_angle_bisector C C1)
  (h_incenter_I : incenter I ABC)
  (h_circumcircles_intersect_P : intersect_circumcircle (triangle_circumcircle ABC) (triangle_circumcircle A1IC1) = P) :
  bisect AC (line PI) := 
sorry

end besector_ac_l133_133240


namespace total_photos_newspaper_l133_133535

-- Define the conditions
def num_pages1 := 12
def photos_per_page1 := 2
def num_pages2 := 9
def photos_per_page2 := 3

-- Define the total number of photos
def total_photos := (num_pages1 * photos_per_page1) + (num_pages2 * photos_per_page2)

-- Prove that the total number of photos is 51
theorem total_photos_newspaper : total_photos = 51 := by
  -- We will fill the proof later
  sorry

end total_photos_newspaper_l133_133535


namespace find_m_for_area_of_triangles_l133_133055

noncomputable def ellipse_foci := (-real.sqrt 2, 0, real.sqrt 2, 0)

theorem find_m_for_area_of_triangles : 
  (∃ (m : ℝ), 
    let F1 := (-real.sqrt 2, 0) in
    let F2 := (real.sqrt 2, 0) in
    let A B : ℝ × ℝ in
    let ellipse : ∀ x y : ℝ, (x^2 / 3) + y^2 = 1 := sorry in
    let line := λ x y : ℝ, y = x + m in
    line_intersects_ellipse : ∀ A B : ℝ × ℝ, disjoint ((A, B) ∪ ellipse) := sorry in
    (area_of_triangle(F1, A, B) = 2 * area_of_triangle(F2, A, B)) → m = -real.sqrt 2 / 3) :=
sorry

end find_m_for_area_of_triangles_l133_133055


namespace hours_of_use_per_charge_l133_133185

theorem hours_of_use_per_charge
  (c h u : ℕ)
  (h_c : c = 10)
  (h_fraction : h = 6)
  (h_use : 6 * u = 12) :
  u = 2 :=
sorry

end hours_of_use_per_charge_l133_133185


namespace function_translation_and_symmetry_l133_133632

theorem function_translation_and_symmetry (f : ℝ → ℝ) (h : ∀ x, f(x - 1) = Real.exp x) : ∀ x, f x = Real.exp (x + 1) :=
by
  sorry

end function_translation_and_symmetry_l133_133632


namespace right_triangle_sets_l133_133544

theorem right_triangle_sets :
  (1.5^2 + 2^2 = 2.5^2) ∧
  (2 * sqrt 2^2 = 2^2) ∧
  (12^2 + 16^2 = 20^2) ∧
  (0.5^2 + 1.2^2 = 1.3^2) :=
by {
  sorry
}

end right_triangle_sets_l133_133544


namespace difference_between_number_and_its_3_5_l133_133440

theorem difference_between_number_and_its_3_5 (x : ℕ) (h : x = 155) :
  x - (3 / 5 : ℚ) * x = 62 := by
  sorry

end difference_between_number_and_its_3_5_l133_133440


namespace max_S_value_l133_133887

-- Define the values of a and b from the inequality
def a := 2
def b := 6

-- Define the conditions for m, n, and S
variables (m n : ℝ)

-- Given conditions
axiom mn_cond : m * n = a / b
axiom m_n_range : m ∈ Ioo (-1 : ℝ) 1 ∧ n ∈ Ioo (-1 : ℝ) 1

-- Define the function S based on m and n
def S (m n : ℝ) := a / (m^2 - 1) + b / (3 * (n^2 - 1))

-- The problem is to prove that the maximum value of S is -6
theorem max_S_value : ∀ (m n : ℝ), m_n_range -> mn_cond -> S m n ≤ -6 :=
by
  sorry

end max_S_value_l133_133887


namespace rotational_homothety_commute_iff_centers_coincide_l133_133202

-- Define rotational homothety and its properties
structure RotationalHomothety (P : Type*) :=
(center : P)
(apply : P → P)
(is_homothety : ∀ p, apply (apply p) = apply p)

variables {P : Type*} [TopologicalSpace P] (H1 H2 : RotationalHomothety P)

-- Prove the equivalence statement
theorem rotational_homothety_commute_iff_centers_coincide :
  (H1.center = H2.center) ↔ (H1.apply ∘ H2.apply = H2.apply ∘ H1.apply) :=
sorry

end rotational_homothety_commute_iff_centers_coincide_l133_133202


namespace find_m_for_area_of_triangles_l133_133061

noncomputable def ellipse_foci := (-real.sqrt 2, 0, real.sqrt 2, 0)

theorem find_m_for_area_of_triangles : 
  (∃ (m : ℝ), 
    let F1 := (-real.sqrt 2, 0) in
    let F2 := (real.sqrt 2, 0) in
    let A B : ℝ × ℝ in
    let ellipse : ∀ x y : ℝ, (x^2 / 3) + y^2 = 1 := sorry in
    let line := λ x y : ℝ, y = x + m in
    line_intersects_ellipse : ∀ A B : ℝ × ℝ, disjoint ((A, B) ∪ ellipse) := sorry in
    (area_of_triangle(F1, A, B) = 2 * area_of_triangle(F2, A, B)) → m = -real.sqrt 2 / 3) :=
sorry

end find_m_for_area_of_triangles_l133_133061


namespace parabola_tangent_area_ratio_l133_133574

structure Point (ℝ : Type) :=
  (x : ℝ) (y : ℝ)

-- Define the parabola and its tangency property
def parabola (p : ℝ) (pt : Point ℝ) : Prop :=
  pt.y^2 = 4 * p * pt.x

def tangent_to_parabola (p : ℝ) (P M : Point ℝ) : Prop :=
  M.y * P.y = 2 * (P.x + M.x)

-- Define the triangle area based on points
def triangle_area (P M N : Point ℝ) : ℝ :=
  0.5 * abs ((M.x - P.x) * (N.y - P.y) - (N.x - P.x) * (M.y - P.y))

-- Define the arc ratio division problem
theorem parabola_tangent_area_ratio (P M N : Point ℝ) (p : ℝ) 
(hP : P = Point.mk (-8) 2) 
(hParabola : parabola p M)
(hParabolaN : parabola p N)
(hTangentM : tangent_to_parabola p P M)
(hTangentN : tangent_to_parabola p P N) :
  area_ratio (triangle_area P M N) 1 2 :=
sorry

end parabola_tangent_area_ratio_l133_133574


namespace number_of_good_convex_m_gons_number_of_bad_convex_m_gons_l133_133586

def is_good_diagonal (d : ℕ) (n : ℕ) : Prop := 
  (d % 2 = 0) ∧ (n % 2 = 0)

def is_bad_diagonal (d : ℕ) (n : ℕ) : Prop := 
  (d % 2 = 1) ∧ (n % 2 = 1)

theorem number_of_good_convex_m_gons (m : ℕ) (h1 : 4 < m) (h2 : m ≤ 1008) : 
  ∀ (n : ℕ), n = 2016 → 
  2 * nat.choose 1008 m = (number_of_good_convex_m_gons m n) :=
sorry

theorem number_of_bad_convex_m_gons (m : ℕ) (h1 : 4 < m) (h2 : m ≤ 1008) (h3 : m % 2 = 0) :
  ∀ (n : ℕ), n = 2016 → 
  (2016 * nat.choose ((2016 - m) / 2 - 1) (m - 1)) / m = (number_of_bad_convex_m_gons m n) :=
sorry

end number_of_good_convex_m_gons_number_of_bad_convex_m_gons_l133_133586


namespace mass_proportion_l133_133122

namespace DensityMixture

variables (k m_1 m_2 m_3 : ℝ)
def rho_1 := 6 * k
def rho_2 := 3 * k
def rho_3 := 2 * k
def arithmetic_mean := (rho_1 k + rho_2 k + rho_3 k) / 3
def density_mixture := (m_1 + m_2 + m_3) / 
    (m_1 / rho_1 k + m_2 / rho_2 k + m_3 / rho_3 k)
def mass_ratio_condition := m_1 / m_2 ≥ 3.5

theorem mass_proportion 
  (k_pos : 0 < k)
  (mass_cond : mass_ratio_condition k m_1 m_2) :
  ∃ (x y : ℝ), (4 * x + 15 * y = 7) ∧ (density_mixture k m_1 m_2 m_3 = arithmetic_mean k) ∧ mass_cond := 
sorry

end DensityMixture

end mass_proportion_l133_133122


namespace find_m_value_l133_133051

-- Define the ellipse equation and its foci
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define the foci of the ellipse C
def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the line equation
def line (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the condition for the areas of triangles F1AB and F2AB
def triangle_area_condition (F1 F2 : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := sorry in -- Coordinates of intersection points need to be defined
  let B := sorry in -- Coordinates of intersection points need to be defined
  sorry /- The actual calculation of the areas would require more complex integration and geometry.
           Here we just outline the condition -/

-- The proof statement
theorem find_m_value :
  ∃ m : ℝ, 
    ∀ x y : ℝ,
      ellipse x y →
      line m x y →
      triangle_area_condition F1 F2 m →
      m = -Real.sqrt 2 / 3 :=
  sorry

end find_m_value_l133_133051


namespace max_points_on_four_coplanar_circles_l133_133851

noncomputable def max_points_on_circles (num_circles : ℕ) (max_intersections : ℕ) : ℕ :=
num_circles * max_intersections

theorem max_points_on_four_coplanar_circles :
  max_points_on_circles 4 2 = 8 := 
sorry

end max_points_on_four_coplanar_circles_l133_133851


namespace determine_m_value_l133_133814

-- Define the condition that the roots of the quadratic are given
def quadratic_equation_has_given_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, (8 * x^2 + 4 * x + m = 0) → (x = (-2 + (Complex.I * Real.sqrt 88)) / 8) ∨ (x = (-2 - (Complex.I * Real.sqrt 88)) / 8)

-- The main statement to be proven
theorem determine_m_value (m : ℝ) (h : quadratic_equation_has_given_roots m) : m = 13 / 4 :=
sorry

end determine_m_value_l133_133814


namespace infinite_product_expectation_not_equal_l133_133007

noncomputable theory -- Because we are dealing with infinite products and expectations

open MeasureTheory -- Necessary for dealing with measure-theoretic constructs (e.g., expectations)

-- Assuming xi is a sequence of random variables with xi : ℕ → ℝ

variable (ξ : ℕ → MeasureTheory.ProbabilitySpace ℝ) -- ξ represents a sequence of random variables
variable (h_indep : ∀ n, MeasureTheory.IndependentSequence (λ i, ξ i)) -- Independence of the variables
variable (h_integrable : ∀ n, MeasureTheory.Integrable (ξ n)) -- Integrability of the variables

-- The statement we want to prove
theorem infinite_product_expectation_not_equal :
  MeasureTheory.condexp (MeasureTheory.Prod (ξ 1)) (MeasureTheory.measurable_prod) != ∏ n, MeasureTheory.condexp (ξ n) (MeasureTheory.measurable_prod) :=
sorry -- Proof goes here

end infinite_product_expectation_not_equal_l133_133007


namespace satisfy_conditions_l133_133094

theorem satisfy_conditions
  (sin cos : ℝ → ℝ)
  (k m n : ℕ)
  (t : ℝ)
  (h1 : (1 + sin t) * (1 + cos t) = 5 / 4)
  (h2 : (1 - sin t) * (1 - cos t) = (m : ℝ) / (n : ℝ) - real.sqrt k)
  (h3 : nat.coprime m n) :
  k + m + n = 27 :=
sorry

end satisfy_conditions_l133_133094


namespace find_difference_eq_895_l133_133981

theorem find_difference_eq_895 :
  let m := Nat.find (λ k => 100 ≤ 13 * k + 7) 
  let n := Nat.find (λ l => 1000 ≤ 13 * l + 7) 
  13 * m + 7 = 111 ∧ 13 * n + 7 = 1006 → n - m = 77 - 8 := 895 :=
by
  sorry

end find_difference_eq_895_l133_133981


namespace female_officers_count_l133_133340

theorem female_officers_count (total_officers_on_duty : ℕ) 
  (percent_female_on_duty : ℝ) 
  (female_officers_on_duty : ℕ) 
  (half_of_total_on_duty_is_female : total_officers_on_duty / 2 = female_officers_on_duty) 
  (percent_condition : percent_female_on_duty * (total_officers_on_duty / 2) = female_officers_on_duty) :
  total_officers_on_duty = 250 :=
by
  sorry

end female_officers_count_l133_133340


namespace ratio_of_area_PQRS_to_circle_l133_133221

theorem ratio_of_area_PQRS_to_circle 
  {PQRS : Type} {circle : Type} [Geometry PQRS] [Inscribed PQRS circle]
  (PR_diameter : Diameter circle (segment PR))
  (angle_PQR : Angle PQR = 45)
  (angle_QPR : Angle QPR = 60) :
  let r := radius circle in
    area PQRS / area circle = (2 + sqrt 3) / (2 * π) :=
by sorry

end ratio_of_area_PQRS_to_circle_l133_133221


namespace fraction_value_l133_133005

theorem fraction_value :
  (1^4 + 2009^4 + 2010^4) / (1^2 + 2009^2 + 2010^2) = 4038091 := by
  sorry

end fraction_value_l133_133005


namespace gcd_eq_55_l133_133400

theorem gcd_eq_55 : Nat.gcd 5280 12155 = 55 := sorry

end gcd_eq_55_l133_133400


namespace border_material_needed_l133_133242

noncomputable def radius (π : ℚ) (A : ℚ) : ℚ := real.sqrt (A * 7 / 22)

noncomputable def circumference (r : ℚ) (π : ℚ) : ℚ := 2 * π * r

noncomputable def required_border (circ : ℚ) : ℚ := circ + 5

theorem border_material_needed (π : ℚ) (A : ℚ) 
  (hπ : π = 22 / 7) (hA : A = 176) : 
  required_border (circumference (radius π A) π) = 
  88 * real.sqrt 14 / 7 + 5 :=
by
  sorry

end border_material_needed_l133_133242


namespace angle_FIG_constant_l133_133960

theorem angle_FIG_constant 
  (ABC : Type)
  [inhabited ABC]
  (A B C I D E F G : ABC)
  (r : ABC → Prop)
  (Hisosceles : AB = AC)
  (Hline_r : r I)
  (Htouches : r D ∧ r E ∧ touches D AB ∧ touches E AC)
  (HBFCG : BF = CE ∧ CG = BD) :
  ∀ (r' : ABC → Prop), angle (F I G) r = angle (F I G) r' :=
sorry

end angle_FIG_constant_l133_133960


namespace cot_arccot_identity_cot_addition_formula_problem_solution_l133_133843

noncomputable def cot (x : ℝ) : ℝ := 1 / tan x
noncomputable def arccot (x : ℝ) : ℝ := arctan (1 / x)

theorem cot_arccot_identity (a : ℝ) : cot (arccot a) = a := sorry

theorem cot_addition_formula (a b : ℝ) : 
  cot (arccot a + arccot b) = (a * b - 1) / (a + b) := sorry

theorem problem_solution : 
  cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 971 / 400 := sorry

end cot_arccot_identity_cot_addition_formula_problem_solution_l133_133843


namespace gcd_three_digit_palindromes_l133_133710

theorem gcd_three_digit_palindromes : 
  GCD (set.image (λ (p : ℕ × ℕ), 101 * p.1 + 10 * p.2) 
    ({a | a ≠ 0 ∧ a < 10} × {b | b < 10})) = 1 := 
by
  sorry

end gcd_three_digit_palindromes_l133_133710


namespace sum_of_areas_is_858_l133_133599

def first_six_odd_squares : List ℕ := [1^2, 3^2, 5^2, 7^2, 9^2, 11^2]

def rectangle_area (width length : ℕ) : ℕ := width * length

def sum_of_areas : ℕ := (first_six_odd_squares.map (rectangle_area 3)).sum

theorem sum_of_areas_is_858 : sum_of_areas = 858 := 
by
  -- Our aim is to show that sum_of_areas is 858
  -- The proof will be developed here
  sorry

end sum_of_areas_is_858_l133_133599


namespace power_equation_l133_133018

theorem power_equation (x a : ℝ) (h : x^(-a) = 3) : x^(2 * a) = 1 / 9 :=
sorry

end power_equation_l133_133018


namespace min_value_of_expression_l133_133984

noncomputable def minimum_value_expression : ℝ :=
  let f (a b : ℝ) := a^4 + b^4 + 16 / (a^2 + b^2)^2
  4

theorem min_value_of_expression (a b : ℝ) (h : 0 < a ∧ 0 < b) : 
  let f := a^4 + b^4 + 16 / (a^2 + b^2)^2
  ∃ c : ℝ, f = c ∧ c = 4 :=
sorry

end min_value_of_expression_l133_133984


namespace solution_correct_l133_133813

-- Define the conditions
def condition1 (x : ℝ) : Prop := 2 ≤ |x - 3| ∧ |x - 3| ≤ 5
def condition2 (x : ℝ) : Prop := (x - 3) ^ 2 ≤ 16

-- Define the solution set
def solution_set (x : ℝ) : Prop := (-1 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 7)

-- Prove that the solution set is correct given the conditions
theorem solution_correct (x : ℝ) : condition1 x ∧ condition2 x ↔ solution_set x :=
by
  sorry

end solution_correct_l133_133813


namespace even_function_l133_133953

def f (x : ℝ) : ℝ := log (x^2 + sqrt (1 + x^4))

theorem even_function : ∀ x : ℝ, f (-x) = f x :=
by
  intros x
  sorry

end even_function_l133_133953


namespace ellipse_area_condition_l133_133075

open Real

theorem ellipse_area_condition (m : ℝ) :
  (∀ x y : ℝ, ((x^2)/3 + y^2 = 1) → (y = x + m) → 
  let F₁ := (-sqrt 2, 0)
      F₂ := (sqrt 2, 0)
  in distance F₁ (x, y) = 2 * distance F₂ (x, y)) → 
  m = -sqrt 2 / 3 :=
by
  sorry

end ellipse_area_condition_l133_133075


namespace gcd_three_digit_palindromes_l133_133697

open Nat

theorem gcd_three_digit_palindromes :
  (∀ a b : ℕ, a ≠ 0 → a < 10 → b < 10 → True) ∧
  let S := {n | ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b} in
  S.Gcd = 1 := by
  sorry

end gcd_three_digit_palindromes_l133_133697


namespace total_age_proof_l133_133527

variable (K : ℕ) -- Kaydence's age
variable (T : ℕ) -- Total age of people in the gathering

def Kaydence_father_age : ℕ := 60
def Kaydence_mother_age : ℕ := Kaydence_father_age - 2
def Kaydence_brother_age : ℕ := Kaydence_father_age / 2
def Kaydence_sister_age : ℕ := 40
def elder_cousin_age : ℕ := Kaydence_brother_age + 2 * Kaydence_sister_age
def younger_cousin_age : ℕ := elder_cousin_age / 2 + 3
def grandmother_age : ℕ := 3 * Kaydence_mother_age - 5

theorem total_age_proof (K : ℕ) : T = 525 + K :=
by 
  sorry

end total_age_proof_l133_133527


namespace value_of_x_cube_plus_inv_cube_l133_133460

theorem value_of_x_cube_plus_inv_cube (x : ℝ) (hx : x < 0) (h : x - x⁻¹ = -1) : x^3 + x⁻³ = -2 :=
sorry

end value_of_x_cube_plus_inv_cube_l133_133460


namespace phi_value_for_unique_symmetry_center_l133_133494

theorem phi_value_for_unique_symmetry_center :
  ∃ (φ : ℝ), (0 < φ ∧ φ < π / 2) ∧
  (φ = π / 12 ∨ φ = π / 6 ∨ φ = π / 3 ∨ φ = 5 * π / 12) ∧
  ((∃ x : ℝ, 2 * x + φ = π ∧ π / 6 < x ∧ x < π / 3) ↔ φ = 5 * π / 12) :=
  sorry

end phi_value_for_unique_symmetry_center_l133_133494


namespace max_food_per_guest_l133_133787

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ) 
                           (h_total_food : total_food = 411) 
                           (h_min_guests : min_guests = 165) : 
  total_food / min_guests = 2.49 :=
by 
  sorry

end max_food_per_guest_l133_133787


namespace find_m_value_l133_133054

-- Define the ellipse equation and its foci
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define the foci of the ellipse C
def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the line equation
def line (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the condition for the areas of triangles F1AB and F2AB
def triangle_area_condition (F1 F2 : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := sorry in -- Coordinates of intersection points need to be defined
  let B := sorry in -- Coordinates of intersection points need to be defined
  sorry /- The actual calculation of the areas would require more complex integration and geometry.
           Here we just outline the condition -/

-- The proof statement
theorem find_m_value :
  ∃ m : ℝ, 
    ∀ x y : ℝ,
      ellipse x y →
      line m x y →
      triangle_area_condition F1 F2 m →
      m = -Real.sqrt 2 / 3 :=
  sorry

end find_m_value_l133_133054


namespace smallest_term_of_bn_div_an_is_four_l133_133471

theorem smallest_term_of_bn_div_an_is_four
  (a : ℕ → ℚ)
  (b : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a n * a (n + 1) = 2 * S n)
  (h3 : b 1 = 16)
  (h4 : ∀ n, b (n + 1) - b n = 2 * n) :
  ∃ n : ℕ, ∀ m : ℕ, (m ≠ 4 → b m / a m > b 4 / a 4) ∧ (n = 4) := sorry

end smallest_term_of_bn_div_an_is_four_l133_133471


namespace solve_a_l133_133114

theorem solve_a (a x : ℝ) : (a^2 - 3)*x^2 + 5*x - 2 > 0 ↔ x ∈ set.Ioo (1/2) 2 → a = 1 ∨ a = -1 :=
by
  sorry

end solve_a_l133_133114


namespace tom_read_books_l133_133659

theorem tom_read_books :
  let books_may := 2
  let books_june := 6
  let books_july := 10
  books_may + books_june + books_july = 18 := by
  sorry

end tom_read_books_l133_133659


namespace valid_probabilities_and_invalid_probability_l133_133265

theorem valid_probabilities_and_invalid_probability :
  (let first_box_1 := (4, 7)
       second_box_1 := (3, 5)
       combined_prob_1 := (first_box_1.1 + second_box_1.1) / (first_box_1.2 + second_box_1.2),
       first_box_2 := (8, 14)
       second_box_2 := (3, 5)
       combined_prob_2 := (first_box_2.1 + second_box_2.1) / (first_box_2.2 + second_box_2.2),
       prob_1 := first_box_1.1 / first_box_1.2,
       prob_2 := second_box_2.1 / second_box_2.2
     in (combined_prob_1 = 7 / 12 ∧ combined_prob_2 = 11 / 19) ∧ (19 / 35 < prob_1 ∧ prob_1 < prob_2) → False) :=
by
  sorry

end valid_probabilities_and_invalid_probability_l133_133265


namespace unique_solution_l133_133821

theorem unique_solution (n m : ℕ) (hn : 0 < n) (hm : 0 < m) : 
  n^2 = m^4 + m^3 + m^2 + m + 1 ↔ (n, m) = (11, 3) :=
by sorry

end unique_solution_l133_133821


namespace students_like_both_basketball_and_cricket_l133_133931

open Set

theorem students_like_both_basketball_and_cricket 
  (A B : Set α) (s : Finset α)
  (h1 : ∥A ∪ B∥ = 9)
  (h2 : ∥A∥ = 7)
  (h3 : ∥B∥ = 5) :
  ∥A ∩ B∥ = 3 :=
by -- proof
  sorry

end students_like_both_basketball_and_cricket_l133_133931


namespace steven_shirts_l133_133606

theorem steven_shirts : 
  (∀ (S A B : ℕ), S = 4 * A ∧ A = 6 * B ∧ B = 3 → S = 72) := 
by
  intro S A B
  intro h
  cases h with h1 h2
  cases h2 with hA hB
  rw [hB, hA]
  sorry

end steven_shirts_l133_133606


namespace find_m_l133_133038

variables (a b : ℝ) (x y m : ℝ) (A B F1 F2 : ℝ × ℝ)

-- Given conditions
def ellipse := (x^2 / 3) + y^2 = 1
def line := y = x + m
def foci_F1 := F1 = (-sqrt 2, 0)
def foci_F2 := F2 = (sqrt 2, 0)
def area_F1_AB := λ A B : ℝ × ℝ, 2 * |det(F1, A, B)| = |det(F2, A, B)|

-- The proof problem statement
theorem find_m : 
  ellipse ∧ 
  line ∧ 
  foci_F1 ∧ 
  foci_F2 ∧ 
  (area_F1_AB A B) →
  m = -sqrt(2) / 3 := sorry

end find_m_l133_133038


namespace sum_equals_92_l133_133796

noncomputable def sum_cos_csc : ℝ :=
  ∑ x in finset.Icc 3 45, 2 * real.cos x * real.cos 2 * (1 + real.csc (x - 2) * real.csc (x + 2))

theorem sum_equals_92 : sum_cos_csc = 92 :=
sorry

end sum_equals_92_l133_133796


namespace max_perfect_squares_l133_133740

/-- Let (a₁, a₂, ..., a₁₀₀) be a permutation of (1, 2, ..., 100).
    Prove that the maximum number of perfect squares that can exist among the
    numbers Sₙ = a₁ + a₂ + ... + aₙ for n = 1, 2, ..., 100 is 60. -/
theorem max_perfect_squares (a : Fin 100 → Nat)
  (h1 : ∀ i : Fin 100, 1 ≤ a i ∧ a i ≤ 100)
  (h2 : ∀ i j : Fin 100, i ≠ j → a i ≠ a j) :
  ∃ seq : Fin 100 → Nat, (∀ i, seq i = (Finset.range (i+1)).sum a) ∧
    (Finset.univ.filter (λ i, (seq i).isSquare)).card = 60 :=
begin
  sorry,
end

end max_perfect_squares_l133_133740


namespace possible_values_of_d_l133_133282

/-- A quadratic polynomial with leading coefficient 1, coefficients forming an arithmetic progression
with common difference d, and roots differing by d, restricts the possible values of d. -/
theorem possible_values_of_d
  (f : Polynomial ℝ)
  (d : ℝ)
  (h1 : f.leadingCoeff = 1)
  (h2 : ∃ a b c, (f.coeff 0 = c ∧ f.coeff 1 = b ∧ f.coeff 2 = a) ∧ 
                 (a = 1 ∧ (b = 1 + d) ∨ (b = 1 - d) ∨ (b = 1 + 2d) ∧ 
                 ∃ x1 x2, f.roots = [x1, x2] ∧ x1 - x2 = d)) :
  (d = -1 ∨ d = -1/2) :=
by
  sorry

end possible_values_of_d_l133_133282


namespace at_least_two_students_same_correct_answers_l133_133934

theorem at_least_two_students_same_correct_answers 
  (n : ℕ) (students : ℕ) (total_correct : ℕ) (hs : students = 10) (ht : total_correct = 42) :
  ∃ a b : ℕ, a ≠ b ∧ a ≤ 42 ∧ b ≤ 42 ∧ ∃ k : ℕ → ℕ, function.injective k ∧ sum k students = total_correct :=
by
  sorry

end at_least_two_students_same_correct_answers_l133_133934


namespace time_for_embankments_l133_133658

theorem time_for_embankments (rate : ℚ) (t1 t2 : ℕ) (w1 w2 : ℕ)
    (h1 : w1 = 75) (h2 : w2 = 60) (h3 : t1 = 4)
    (h4 : rate = 1 / (w1 * t1 : ℚ)) 
    (h5 : t2 = 1 / (w2 * rate)) : 
    t1 + t2 = 9 :=
sorry

end time_for_embankments_l133_133658


namespace distance_B_to_line_AC_l133_133948

theorem distance_B_to_line_AC :
  let AB := (1, 1, 2)
  let AC := (2, 1, 1)
  (∃ d : ℝ, d = real.sqrt (66) / 6) :=
by
  sorry

end distance_B_to_line_AC_l133_133948


namespace solve_for_x_l133_133367

theorem solve_for_x (x : ℝ) (h_pos : 0 < x) (h : (x / 100) * (x ^ 2) = 9) : x = 10 * (3 ^ (1 / 3)) :=
by
  sorry

end solve_for_x_l133_133367


namespace leos_time_is_1230_l133_133295

theorem leos_time_is_1230
  (theo_watch_slow: Int)
  (theo_watch_fast_belief: Int)
  (leo_watch_fast: Int)
  (leo_watch_slow_belief: Int)
  (theo_thinks_time: Int):
  theo_watch_slow = 10 ∧
  theo_watch_fast_belief = 5 ∧
  leo_watch_fast = 5 ∧
  leo_watch_slow_belief = 10 ∧
  theo_thinks_time = 720
  → leo_thinks_time = 750 :=
by
  sorry

end leos_time_is_1230_l133_133295


namespace gcd_lcm_product_l133_133445

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 12) (h2 : b = 15) :
  Nat.gcd a b * Nat.lcm a b = 180 := by
  rw [h1, h2]
  have gcd_val : Nat.gcd 12 15 = 3 := by sorry
  have lcm_val : Nat.lcm 12 15 = 60 := by sorry
  rw [gcd_val, lcm_val]
  norm_num

end gcd_lcm_product_l133_133445


namespace ratio_of_cost_puppy_to_parakeet_l133_133312

-- Definitions of costs
def cost_parakeet : ℕ := 10
def cost_kitten (P : ℕ) : ℕ := 2 * P
def total_cost (P K D : ℕ) : ℕ := 3 * P + 2 * K + 2 * D

-- Given Problem Conditions
axiom cost_parakeet_is_10 : cost_parakeet = 10
axiom cost_kitten_is_twice_parakeet : ∀ P, cost_kitten P = 2 * P
axiom total_cost_is_130 : total_cost cost_parakeet (cost_kitten cost_parakeet) (D) = 130

-- Theorem to prove
theorem ratio_of_cost_puppy_to_parakeet (D : ℕ) : D = 30 → cost_parakeet = 10 → (D / cost_parakeet) = 3 :=
by
  intro hD hP
  rw [hD, hP]
  norm_num
  sorry

end ratio_of_cost_puppy_to_parakeet_l133_133312


namespace range_of_m_l133_133151

-- Definitions based on conditions
def inequality_holds (m : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → (m * Real.exp x / x ≥ 6 - 4 * x)

-- The statement to be proved
theorem range_of_m (m : ℝ) : inequality_holds m → m ≥ 2 * Real.exp (-(1 / 2)) :=
by
  sorry

end range_of_m_l133_133151


namespace coat_price_reduction_l133_133733

theorem coat_price_reduction (original_price reduction_amount : ℝ) 
  (h1 : original_price = 500) 
  (h2 : reduction_amount = 200) : 
  (reduction_amount / original_price) * 100 = 40 := 
by
  rw [h1, h2]
  norm_num
  sorry

end coat_price_reduction_l133_133733


namespace next_natural_number_adjacent_l133_133920

open Nat

theorem next_natural_number_adjacent (a : ℕ) (n : ℕ) (h : n = a^2) : n + 1 = a^2 + 1 := by
  rw [h]
  rfl

end next_natural_number_adjacent_l133_133920


namespace rem_x_y_mul_2_l133_133797

-- Define the remainder function for real numbers
def real_rem (x y : ℝ) : ℝ := x - y * (floor (x / y))

-- Define the numbers x and y
def x : ℝ := 5 / 7
def y : ℝ := 3 / 4

-- State the theorem to be proven
theorem rem_x_y_mul_2 : real_rem x y * 2 = 10 / 7 :=
by
  sorry

end rem_x_y_mul_2_l133_133797


namespace select_eight_genuine_dinars_l133_133226

theorem select_eight_genuine_dinars (coins : Fin 11 → ℝ) :
  (∃ (fake_coin : Option (Fin 11)), 
    ((∀ i j : Fin 11, i ≠ j → coins i = coins j) ∨
    (∀ (genuine_coins impostor_coins : Finset (Fin 11)), 
      genuine_coins ∪ impostor_coins = Finset.univ →
      impostor_coins.card = 1 →
      (∃ difference : ℝ, ∀ i ∈ genuine_coins, coins i = difference) ∧
      (∃ i ∈ impostor_coins, coins i ≠ difference)))) →
  (∃ (selected_coins : Finset (Fin 11)), selected_coins.card = 8 ∧
   (∀ i j : Fin 11, i ∈ selected_coins → j ∈ selected_coins → coins i = coins j)) :=
sorry

end select_eight_genuine_dinars_l133_133226


namespace ellipse_area_condition_l133_133077

open Real

theorem ellipse_area_condition (m : ℝ) :
  (∀ x y : ℝ, ((x^2)/3 + y^2 = 1) → (y = x + m) → 
  let F₁ := (-sqrt 2, 0)
      F₂ := (sqrt 2, 0)
  in distance F₁ (x, y) = 2 * distance F₂ (x, y)) → 
  m = -sqrt 2 / 3 :=
by
  sorry

end ellipse_area_condition_l133_133077


namespace minimum_cubes_required_l133_133369

def cube_snaps_visible (n : Nat) : Prop := 
  ∀ (cubes : Fin n → Fin 6 → Bool),
    (∀ i, (cubes i 0 ∧ cubes i 1) ∨ ¬(cubes i 0 ∨ cubes i 1)) → 
    ∃ i j, (i ≠ j) ∧ 
            (cubes i 0 ↔ ¬ cubes j 0) ∧ 
            (cubes i 1 ↔ ¬ cubes j 1)

theorem minimum_cubes_required : 
  ∃ n, cube_snaps_visible n ∧ n = 4 := 
  by sorry

end minimum_cubes_required_l133_133369


namespace coefficient_of_monomial_l133_133621

theorem coefficient_of_monomial (a b : ℝ) : ∃ c : ℝ, -5 * Real.pi * a^2 * b = c * a^2 * b ∧ c = -5 * Real.pi := by
  use -5 * Real.pi
  exact ⟨rfl, rfl⟩

end coefficient_of_monomial_l133_133621


namespace triangle_properties_l133_133865

theorem triangle_properties 
  (a b c : ℝ) (A B C : ℝ)
  (h_tri : A + B + C = π)
  (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (h_sides : b^2 = a^2 + c^2 - sqrt(3) * a * c) :
  (B = π / 6) ∧ 
  (∃ U V : ℝ, U = cos A + sin C ∧ (sqrt(3) / 2 < U ∧ U < 3 / 2)) :=
by
  sorry  

end triangle_properties_l133_133865


namespace brownie_cost_l133_133790

theorem brownie_cost (total_money : ℕ) (num_pans : ℕ) (pieces_per_pan : ℕ) 
    (total_money = 32) (num_pans = 2) (pieces_per_pan = 8) : 
    (total_money / (num_pans * pieces_per_pan) = 2) := by
  sorry

end brownie_cost_l133_133790


namespace andy_position_after_3030_turns_l133_133783

def start_position : (ℕ × ℕ) := (30, -30)

def initial_direction : ℕ := 0 -- North

def move_length (n : ℕ) : ℕ := n + 1

def turn_right (d : ℕ) : ℕ := (d + 1) % 4

def new_position (pos : ℕ × ℕ) (d : ℕ) (len : ℕ) : ℕ × ℕ :=
  match d with
  | 0 => (pos.fst, pos.snd + len) -- North
  | 1 => (pos.fst + len, pos.snd) -- East
  | 2 => (pos.fst, pos.snd - len) -- South
  | 3 => (pos.fst - len, pos.snd) -- West
  | _ => pos -- This default case should never be reached

theorem andy_position_after_3030_turns :
  let final_pos := (list.range 3030).foldl
    (λ pos turn_num, let
      (cur_pos, cur_dir) := pos
    in (new_position cur_pos cur_dir (move_length turn_num), turn_right cur_dir))
    (start_position, initial_direction)
  in final_pos.fst = (4573, -1546) :=
show final_pos.fst.snd = (4573, -1546), from sorry

end andy_position_after_3030_turns_l133_133783


namespace exists_integer_n_l133_133967

theorem exists_integer_n (P : ℤ[X]) (q : ℤ) : ∃ n, q ∣ (Finset.range (q^2).nat_abs).sum (λ i, P.eval (i+1)) := 
begin
  sorry
end

end exists_integer_n_l133_133967


namespace number_of_sets_B_satisfying_union_l133_133572

theorem number_of_sets_B_satisfying_union (A : Set ℕ) (hA : A = {1, 2}) :
  {B : Set ℕ // A ∪ B = {1, 2, 3}}.card = 4 := 
sorry

end number_of_sets_B_satisfying_union_l133_133572


namespace brownie_cost_l133_133789

theorem brownie_cost (total_money : ℕ) (num_pans : ℕ) (pieces_per_pan : ℕ) 
    (total_money = 32) (num_pans = 2) (pieces_per_pan = 8) : 
    (total_money / (num_pans * pieces_per_pan) = 2) := by
  sorry

end brownie_cost_l133_133789


namespace total_photos_newspaper_l133_133537

theorem total_photos_newspaper (pages1 pages2 photos_per_page1 photos_per_page2 : ℕ)
  (h1 : pages1 = 12) (h2 : photos_per_page1 = 2)
  (h3 : pages2 = 9) (h4 : photos_per_page2 = 3) :
  (pages1 * photos_per_page1) + (pages2 * photos_per_page2) = 51 :=
by
  sorry

end total_photos_newspaper_l133_133537


namespace triangle_sine_ratio_l133_133163

-- Define points A and C
def A : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (1, 0)

-- Define the condition of point B being on the ellipse
def isOnEllipse (B : ℝ × ℝ) : Prop :=
  (B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1

-- Define the sin law ratio we need to prove
noncomputable def sin_ratio (sin_A sin_C sin_B : ℝ) : ℝ := 
  (sin_A + sin_C) / sin_B

-- Prove the required sine ratio condition
theorem triangle_sine_ratio (B : ℝ × ℝ) (sin_A sin_C sin_B : ℝ)
  (hB : isOnEllipse B) (hA : sin_A = 0) (hC : sin_C = 0) (hB_nonzero : sin_B ≠ 0) :
  sin_ratio sin_A sin_C sin_B = 2 :=
by
  -- Skipping proof
  sorry

end triangle_sine_ratio_l133_133163


namespace complement_union_correct_l133_133587

open Set

variable U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
variable S : Set ℕ := {1, 3, 5}
variable T : Set ℕ := {3, 6}

theorem complement_union_correct :
  (U \ (S ∪ T) = {2, 4, 7, 8}) :=
by {
  -- proof goes here
  sorry
}

end complement_union_correct_l133_133587


namespace cot_sum_eq_cot_sum_example_l133_133836

theorem cot_sum_eq (a b c d : ℝ) :
  cot (arccot a + arccot b + arccot c + arccot d) = (a * b * c * d - (a * b + b * c + c * d) + 1) / (a + b + c + d)
:= sorry

theorem cot_sum_example :
  cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 1021 / 420
:= sorry

end cot_sum_eq_cot_sum_example_l133_133836


namespace median_of_data_set_l133_133637

theorem median_of_data_set : 
  median ({5, 7, 5, 8, 6, 13, 5} : multiset ℕ) = 6 := 
by
  sorry

end median_of_data_set_l133_133637


namespace area_ratio_ellipse_l133_133080

noncomputable def ellipse := { x // (x.1^2 / 3 + x.2^2 = 1) }
def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))
def intersects (line : ℝ × ℝ) (C : ℝ × ℝ → Prop) := ∃ (x y : ℝ), (C (x, y)) ∧ (y = x + line.2)

theorem area_ratio_ellipse 
  (a b : ℝ) (h_ellipse : a^2 = 3 ∧ b^2 = 1) (m : ℝ) 
  (h_foci : foci a b = ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))) 
  (h_condition : ∀ (x : ℝ), intersects (x, m) (λ pt, pt.1^2 / 3 + pt.2^2 = 1) →
     |((x - real.sqrt (a^2 - b^2)) + m)| = 2|((x + real.sqrt (a^2 - b^2)) + m)|) :
  m = -real.sqrt(2) / 3 := sorry

end area_ratio_ellipse_l133_133080


namespace length_of_AB_l133_133365

noncomputable def parabola_focus : ℝ × ℝ := (0, 1)

noncomputable def slope_of_line : ℝ := Real.tan (Real.pi / 6)

-- Equation of the line in point-slope form
noncomputable def line_eq (x : ℝ) : ℝ :=
  (slope_of_line * x) + 1

-- Intersection points of the line with the parabola y = (1/4)x^2
noncomputable def parabola_eq (x : ℝ) : ℝ :=
  (1/4) * x ^ 2

theorem length_of_AB :
  ∃ A B : ℝ × ℝ, 
    (A.2 = parabola_eq A.1) ∧
    (B.2 = parabola_eq B.1) ∧ 
    (A.2 = line_eq A.1) ∧
    (B.2 = line_eq B.1) ∧
    ((((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) ^ (1 / 2)) = 16 / 3) :=
by
  sorry

end length_of_AB_l133_133365


namespace sum_of_adjacent_to_nine_l133_133642

theorem sum_of_adjacent_to_nine (arrangement : List ℕ) (h : arrangement.perm ([2, 3, 4, 6, 8, 9, 12, 18, 24, 27, 36, 54, 72, 108, 216])) :
  (∀ i, i < arrangement.length → gcd (arrangement.get i) (arrangement.get ((i + 1) % arrangement.length)) > 1) →
  ∃ i, arrangement.get i = 9 ∧ 
  (arrangement.get ((i - 1 + arrangement.length) % arrangement.length) + arrangement.get ((i + 1) % arrangement.length) = 30) :=
begin
  sorry
end

end sum_of_adjacent_to_nine_l133_133642


namespace limit_of_second_derivative_l133_133145

variables {ℝ : Type} [LinearOrderedField ℝ]

theorem limit_of_second_derivative (f : ℝ → ℝ) (x0 : ℝ)
  (h : ℝ → Prop) (Hf : ∀ h, h → True)
  (f''_x0 : f'' x0 = -3) : 
  lim (λ h, (f (x0 + h) - f (x0 - 3 * h)) / h) 0 = -12 :=
begin
  sorry
end

end limit_of_second_derivative_l133_133145


namespace number_of_semesters_l133_133561

-- Define the given conditions
def units_per_semester : ℕ := 20
def cost_per_unit : ℕ := 50
def total_cost : ℕ := 2000

-- Define the cost per semester using the conditions
def cost_per_semester := units_per_semester * cost_per_unit

-- Prove the number of semesters is 2 given the conditions
theorem number_of_semesters : total_cost / cost_per_semester = 2 := by
  -- Add a placeholder "sorry" to skip the actual proof
  sorry

end number_of_semesters_l133_133561


namespace roots_ratio_sum_l133_133578

theorem roots_ratio_sum (a b m : ℝ) 
  (m1 m2 : ℝ)
  (h_roots : a ≠ b ∧ b ≠ 0 ∧ m ≠ 0 ∧ a ≠ 0 ∧ 
    ∀ x : ℝ, m * (x^2 - 3 * x) + 2 * x + 7 = 0 → (x = a ∨ x = b)) 
  (h_ratio : (a / b) + (b / a) = 7 / 3)
  (h_m1_m2_eq : ((3 * m - 2) ^ 2) / (7 * m) - 2 = 7 / 3)
  (h_m_vieta : (3 * m - 2) ^ 2 - 27 * m * (91 / 3) = 0) :
  (m1 + m2 = 127 / 27) ∧ (m1 * m2 = 4 / 9) →
  ((m1 / m2) + (m2 / m1) = 47.78) :=
sorry

end roots_ratio_sum_l133_133578


namespace prove_least_value_of_n_l133_133849

-- Defining the statements as per the problem conditions and question
def every_color_used_infinitely (colors : set ℕ) : Prop :=
  ∀ c ∈ colors, ∃ (points : ℕ → ℝ × ℝ), function.injective points ∧ ∀ n, color (points n) = c

def exists_line_with_two_colors (colors : set ℕ) : Prop :=
  ∃ (line : ℝ × ℝ → Prop) (c1 c2 : ℕ), c1 ∈ colors ∧ c2 ∈ colors ∧
  ∀ pt : ℝ × ℝ, line pt → color pt = c1 ∨ color pt = c2

def concyclic_distinct_colors (points : set (ℝ × ℝ)) (n : ℕ) : Prop :=
  ∃ pts ∈ points, set.card pts = 4 ∧
  ∃ C : set (ℝ × ℝ), set.finite C ∧ 
  set.subset pts C ∧ ∀ p q : (ℝ × ℝ), p ∈ pts → q ∈ pts → 
  p ≠ q ∧ color p ≠ color q

def least_value_of_n (colors : set ℕ) : Prop :=
  ∃ n, n ∈ colors ∧ 
  concyclic_distinct_colors points n ∧
  ∀ m < n, ¬ concyclic_distinct_colors points m

theorem prove_least_value_of_n : ∃ n, least_value_of_n n = 5 :=
sorry

end prove_least_value_of_n_l133_133849


namespace intersection_eq_l133_133897

def M : Set Real := {x | x^2 < 3 * x}
def N : Set Real := {x | Real.log x < 0}

theorem intersection_eq : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_eq_l133_133897


namespace airplane_total_luggage_weight_l133_133654

def num_people := 6
def bags_per_person := 5
def weight_per_bag := 50
def additional_bags := 90

def total_weight_people := num_people * bags_per_person * weight_per_bag
def total_weight_additional_bags := additional_bags * weight_per_bag

def total_luggage_weight := total_weight_people + total_weight_additional_bags

theorem airplane_total_luggage_weight : total_luggage_weight = 6000 :=
by
  sorry

end airplane_total_luggage_weight_l133_133654


namespace number_of_topless_cubical_box_configurations_l133_133233

theorem number_of_topless_cubical_box_configurations :
  let L_shaped_figure := {L : Type} -- Representing an L-shaped figure.
  let additional_squares := {A B C D E F G H I J : Type} -- Ten possible additional squares.
  let valid_squares := {A, B, C, D, G, H, I, J}
  ∃ n : ℕ, n = 8 ∧ (∀ s ∈ valid_squares, 
       let full_figure := L_shaped_figure ∪ {s} 
       ∃ can_be_folded : bool, can_be_folded := true) :=
  sorry -- Proof to be provided

end number_of_topless_cubical_box_configurations_l133_133233


namespace absolute_sum_example_l133_133013

theorem absolute_sum_example :
  let x := -1;
      z := 3.7;
      w := 9.3 in
  |z - x| + |w - x| = 15 := by
  sorry

end absolute_sum_example_l133_133013


namespace mathematicians_correctness_l133_133261

theorem mathematicians_correctness:
  ∀ (w1₁ w1₂ t1₁ t1₂ w2₁ w2₂ t2₁ t2₂: ℕ),
  (w1₁ = 4) ∧ (t1₁ = 7) ∧ (w1₂ = 3) ∧ (t1₂ = 5) →
  (w2₁ = 8) ∧ (t2₁ = 14) ∧ (w2₂ = 3) ∧ (t2₂ = 5) →
  (4/7 : ℚ) < 3/5 →
  (w1₁ + w1₂) / (t1₁ + t1₂) = 7 / 12 ∨ (w2₁ + w2₂) / (t2₁ + t2₂) = 11 / 19 →
  ¬ (w1₁ + w1₂) / (t1₁ + t1₂) < 19 / 35 ∧ (w2₁ + w2₂) / (t2₁ + t2₂) < 19 / 35 :=
by
  sorry

end mathematicians_correctness_l133_133261


namespace find_m_value_l133_133047

-- Define the ellipse equation and its foci
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define the foci of the ellipse C
def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the line equation
def line (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the condition for the areas of triangles F1AB and F2AB
def triangle_area_condition (F1 F2 : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := sorry in -- Coordinates of intersection points need to be defined
  let B := sorry in -- Coordinates of intersection points need to be defined
  sorry /- The actual calculation of the areas would require more complex integration and geometry.
           Here we just outline the condition -/

-- The proof statement
theorem find_m_value :
  ∃ m : ℝ, 
    ∀ x y : ℝ,
      ellipse x y →
      line m x y →
      triangle_area_condition F1 F2 m →
      m = -Real.sqrt 2 / 3 :=
  sorry

end find_m_value_l133_133047


namespace tangent_line_eqn_triangle_area_l133_133987

open Real

def cubic_function (x : ℝ) := x^3 - 2*x

def tangent_point : ℝ × ℝ := (1, -1)

theorem tangent_line_eqn:
  ∀ (x y : ℝ) (tangent_line : ℝ → ℝ), 
    (∃ (m : ℝ), m = (deriv cubic_function) 1 ∧ tangent_line = (fun x => m * (x - 1) + (-1))) → 
    (∀ x y, tangent_line x = y ↔ x - y - 2 = 0) :=
  sorry

theorem triangle_area:
  ∀ (tangent_line : ℝ → ℝ),
    (∃ (m : ℝ), m = (deriv cubic_function) 1 ∧ tangent_line = (fun x => m * (x - 1) + (-1))) →
    let intercepts := (0, tangent_line 0) in
    let area := (1/2 * abs intercepts.1 * abs intercepts.2) in
    intercepts.1 * intercepts.2 = 4 := 
  sorry

end tangent_line_eqn_triangle_area_l133_133987


namespace correct_calculation_of_exponentiation_l133_133721

theorem correct_calculation_of_exponentiation (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end correct_calculation_of_exponentiation_l133_133721


namespace ordered_quadruples_div_100_l133_133581

theorem ordered_quadruples_div_100 :
  let n := (Finset.range 50).succ.powersetLen 3.card * 49 / 6 in
  n = 18424 ∧ n / 100 = 184.24 :=
by
  -- Define the even integer quadruples condition
  let even_quadruple (x1 x2 x3 x4 : ℕ) := ∃ y1 y2 y3 y4 : ℕ, x1 = 2 * y1 ∧ x2 = 2 * y2 ∧ x3 = 2 * y3 ∧ x4 = 2 * y4 ∧ y1 > 0 ∧ y2 > 0 ∧ y3 > 0 ∧ y4 > 0

  -- Define the summation condition
  let sum_condition (x1 x2 x3 x4 : ℕ) : Prop := x1 + x2 + x3 + x4 = 100

  -- Calculate the correct number of quadruples
  have n_def : n = (Finset.range 50).succ.powersetLen 3.card * 49 / 6 := rfl

  -- The number of quadruples
  have n_val : n = 18424 := sorry

  -- Calculate n divided by 100
  have n_div_100 : n / 100 = 184.24 := sorry

  exact n_val ∧ n_div_100

end ordered_quadruples_div_100_l133_133581


namespace compute_division_l133_133803

theorem compute_division (a : ℕ) : (19^12 / 19^5 = 893871739) := by
  let result := 19^12 / 19^5
  have result_eq : result = 19^7 := by
    calc
      19^12 / 19^5 = 19^(12 - 5) : by rw [Nat.div_pow_sub 19 12 5]
      ... = 19^7   : by norm_num
  have final_result : 19^7 = 893871739 := by norm_num
  rw ← result_eq
  exact final_result

end compute_division_l133_133803


namespace valid_probabilities_and_invalid_probability_l133_133269

theorem valid_probabilities_and_invalid_probability :
  (let first_box_1 := (4, 7)
       second_box_1 := (3, 5)
       combined_prob_1 := (first_box_1.1 + second_box_1.1) / (first_box_1.2 + second_box_1.2),
       first_box_2 := (8, 14)
       second_box_2 := (3, 5)
       combined_prob_2 := (first_box_2.1 + second_box_2.1) / (first_box_2.2 + second_box_2.2),
       prob_1 := first_box_1.1 / first_box_1.2,
       prob_2 := second_box_2.1 / second_box_2.2
     in (combined_prob_1 = 7 / 12 ∧ combined_prob_2 = 11 / 19) ∧ (19 / 35 < prob_1 ∧ prob_1 < prob_2) → False) :=
by
  sorry

end valid_probabilities_and_invalid_probability_l133_133269


namespace sum_even_if_product_odd_l133_133923

theorem sum_even_if_product_odd (a b : ℤ) (h : (a * b) % 2 = 1) : (a + b) % 2 = 0 := 
by
  sorry

end sum_even_if_product_odd_l133_133923


namespace g_is_self_inverse_l133_133791

def f (x : ℝ) : ℝ := (x - 4) / (x - 3)

def g (a : ℝ) (x : ℝ) : ℝ := f (x + a)

theorem g_is_self_inverse (a : ℝ) (x : ℝ) (h₁ : g a x = (g a)⁻¹ x) (h₂ : symmetric y = x - 3) :
  a = -3 := 
sorry

end g_is_self_inverse_l133_133791


namespace area_CF1F2_correct_l133_133875

-- Define the conditions for the problem
section

variables {F1 F2 C : ℝ × ℝ} {b : ℝ}
def ellipse_foci := (F1.1 - F2.1)^2 + (F1.2 - F2.2)^2 = 4 * (3^2 - b^2)

-- 1. The line intersects the ellipse at points A and B
variables {x1 x2 y1 y2 : ℝ}
def line_eq (x y : ℝ) : Prop := x + 3 * y = 7
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / b^2) = 1
def intersection_A : Prop := line_eq x1 y1 ∧ ellipse_eq x1 y1
def intersection_B : Prop := line_eq x2 y2 ∧ ellipse_eq x2 y2

-- 2. The given condition 0 < b < 3
axiom b_cond : 0 < b ∧ b < 3

-- 3. The midpoint of segment AB is (1, 2)
def midpoint_condition : Prop := (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = 2

-- 4. Area of the triangle should be 2√3
def triangle_area_CF1F2 : ℝ := 1/2 * 2 * (F1.2 - F2.2).abs
def correct_area : ℝ := 2 * real.sqrt 3

theorem area_CF1F2_correct 
    (ellipse_foci : ellipse_foci)
    (intersection_A : intersection_A)
    (intersection_B : intersection_B)
    (midpoint_cond : midpoint_condition)
    : triangle_area_CF1F2 = correct_area :=
by sorry

end

end area_CF1F2_correct_l133_133875


namespace simplify_expression_l133_133203

variable (a b c d : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (h : a + b + c = d)

theorem simplify_expression :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 0 := by
  sorry

end simplify_expression_l133_133203


namespace cos_b4_b6_l133_133486

theorem cos_b4_b6 (a b : ℕ → ℝ) (d : ℝ) 
  (ha_geom : ∀ n, a (n + 1) / a n = a 1)
  (hb_arith : ∀ n, b (n + 1) = b n + d)
  (ha_prod : a 1 * a 5 * a 9 = -8)
  (hb_sum : b 2 + b 5 + b 8 = 6 * Real.pi) : 
  Real.cos ((b 4 + b 6) / (1 - a 3 * a 7)) = -1 / 2 :=
sorry

end cos_b4_b6_l133_133486


namespace equilateral_triangle_side_length_l133_133216

noncomputable def side_length (a : ℝ) := if a = 0 then 0 else (a : ℝ) * (3 : ℝ) / 2

theorem equilateral_triangle_side_length
  (a : ℝ)
  (h1 : a ≠ 0)
  (A := (a, - (1 / 3) * a^2))
  (B := (-a, - (1 / 3) * a^2))
  (Habo : (A.1 - 0)^2 + (A.2 - 0)^2 = (B.1 - 0)^2 + (B.2 - 0)^2) :
  ∃ s : ℝ, s = 9 / 2 :=
by
  sorry

end equilateral_triangle_side_length_l133_133216


namespace parallel_lines_cond_l133_133622

theorem parallel_lines_cond (a c : ℝ) :
    (∀ (x y : ℝ), (a * x - 2 * y - 1 = 0) ↔ (6 * x - 4 * y + c = 0)) → 
        (a = 3 ∧ ∃ (c : ℝ), c ≠ -2) ∨ (a = 3 ∧ c = -2) := 
sorry

end parallel_lines_cond_l133_133622


namespace hyperbola_standard_eq_proof_l133_133519

noncomputable def real_axis_length := 6
noncomputable def asymptote_slope := 3 / 2

def hyperbola_standard_eq (a b : ℝ) :=
  ∀ x y : ℝ, (y^2 / a^2 - x^2 / b^2 = 1)

theorem hyperbola_standard_eq_proof (a b : ℝ) 
  (h_a : 2 * a = real_axis_length)
  (h_b : a / b = asymptote_slope) :
  hyperbola_standard_eq 3 2 := 
by
  sorry

end hyperbola_standard_eq_proof_l133_133519


namespace mathematicians_correct_l133_133259

noncomputable def scenario1 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 4 ∧ total1 = 7 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario2 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 8 ∧ total1 = 14 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario3 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  (19 / 35 < 4 / 7) ∧ (4 / 7 < 3 / 5)

noncomputable def probability (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : ℝ :=
  (whites1 + whites2) / (total1 + total2)

theorem mathematicians_correct :
  let whites1_s1 := 4 in
  let total1_s1 := 7 in
  let whites2_s1 := 3 in
  let total2_s1 := 5 in
  let whites1_s2 := 8 in
  let total1_s2 := 14 in
  let whites2_s2 := 3 in
  let total2_s2 := 5 in
  scenario1 whites1_s1 total1_s1 whites2_s1 total2_s1 →
  scenario2 whites1_s2 total1_s2 whites2_s2 total2_s2 →
  scenario3 whites1_s1 total1_s1 whites2_s2 total2_s2 →
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 ≤ 3 / 5 ∨
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 = 4 / 7 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 ≤ 3 / 5 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 = 4 / 7 :=
begin
  intros,
  sorry

end mathematicians_correct_l133_133259


namespace sqrt_product_of_powers_eq_l133_133406

theorem sqrt_product_of_powers_eq :
  ∃ (x y z : ℕ), prime x ∧ prime y ∧ prime z ∧ x = 2 ∧ y = 3 ∧ z = 5 ∧
  sqrt (x^4 * y^6 * z^2) = 540 := by
  use 2, 3, 5
  show prime 2, from prime_two
  show prime 3, from prime_three
  show prime 5, from prime_five
  show 2 = 2, from rfl
  show 3 = 3, from rfl
  show 5 = 5, from rfl
  sorry

end sqrt_product_of_powers_eq_l133_133406


namespace line_slope_and_intersection_l133_133448

theorem line_slope_and_intersection:
  (∀ x y : ℝ, x^2 + x / 4 + y / 5 = 1 → ∀ m : ℝ, m = -5 / 4) ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 → ¬ (x^2 + x / 4 + y / 5 = 1)) :=
by
  sorry

end line_slope_and_intersection_l133_133448


namespace largest_term_in_expansion_l133_133104

theorem largest_term_in_expansion 
  (x : ℝ) (n : ℕ) 
  (h : (2 : ℝ)^(2 * n) - (2 : ℝ)^n = 240)
  (h4 : n = 4) : 
  ∃ (k : ℕ), k > 0 ∧ 
  (binomial 4 k * (sqrt x ^ k) * (1 / cbrt x ^ (4 - k))) = 6 * cbrt x :=
by 
  sorry

end largest_term_in_expansion_l133_133104


namespace problem_1_correct_problem_2_correct_l133_133473

-- Definitions for points and slopes
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 4, y := 0 }
def B : Point := { x := 8, y := 10 }
def C : Point := { x := 0, y := 6 }

def slope (p1 p2: Point) : ℝ := (p2.y - p1.y) / (p2.x - p1.x)

-- Problem 1: Equation of the line passing through A and parallel to BC
def line_parallel_to_BC : Prop :=
  let k := slope B C
  let k_parallel := k
  let eq : Point → ℝ → ℝ → Prop := λ p m b, p.y = m * p.x + b
  let b := A.y - k_parallel * A.x
  eq A k_parallel b ∧ (eq = λ p m b, p.y = (1 / 2:ℝ) * p.x + (-4:ℝ))

-- Problem 2: Equation of the line containing the altitude on edge AC
def line_altitude_on_AC : Prop :=
  let k := slope A C
  let k_altitude := -(1 / k)
  let eq : Point → ℝ → ℝ → Prop := λ p m b, p.y = m * p.x + b
  let b := A.y - k_altitude * A.x
  eq A k_altitude b ∧ (eq = λ p m b, p.y = (2 / 3:ℝ) * p.x + (-8 / 3:ℝ))

-- Statements asserting the correctness of the proofs
theorem problem_1_correct : line_parallel_to_BC := sorry
theorem problem_2_correct : line_altitude_on_AC := sorry

end problem_1_correct_problem_2_correct_l133_133473


namespace problem_I_problem_II_l133_133866

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2
noncomputable def h (x : ℝ) : ℝ := f x - x + 1
noncomputable def phi (m x : ℝ) : ℝ := m * g x + x * f x
noncomputable def t (x : ℝ) : ℝ := (-1 - Real.log x) / x

theorem problem_I :
  ∃ (x : ℝ), x > 0 ∧ h x = 0 :=
sorry

theorem problem_II (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x2 < x1) :
  ∃ m : ℝ, m ≤ -1/2 ∧ (forall x1 x2, mg x2 - mg x1 > x1 f x1 - x2 f x2) :=
sorry

end problem_I_problem_II_l133_133866


namespace jihye_marbles_l133_133956

theorem jihye_marbles (Y : ℕ) (h1 : Y + (Y + 11) = 85) : Y + 11 = 48 := by
  sorry

end jihye_marbles_l133_133956


namespace students_in_class_l133_133427

theorem students_in_class
 (S : ℕ)
 (erasers_original pencils_original erasers_left pencils_left : ℕ)
 (h_erasers : erasers_original = 49)
 (h_pencils : pencils_original = 66)
 (h_erasers_left : erasers_left = 4)
 (h_pencils_left : pencils_left = 6)
 (erasers_to_divide : ℕ := erasers_original - erasers_left)
 (pencils_to_divide : ℕ := pencils_original - pencils_left)
 (h_erasers_to_divide : erasers_to_divide = 45)
 (h_pencils_to_divide : pencils_to_divide = 60)
 (h_divide_erasers : 45 % S = 0) 
 (h_divide_pencils : 60 % S = 0):
 S = 15 := 
sorry

end students_in_class_l133_133427


namespace proof_problem_l133_133493

def f (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x + 1

theorem proof_problem (p q r : ℝ) (h : ∀ x : ℝ, p * f x + q * f (x + r) = 2018) (h_pq : p = 1009) (h_q : q = 1009) (h_r : r = Real.pi) : p * Real.cos r + q = 0 :=
by
  sorry

end proof_problem_l133_133493


namespace log_neq_x_minus_one_l133_133286

theorem log_neq_x_minus_one (x : ℝ) (h₁ : 0 < x) : Real.log x ≠ x - 1 :=
sorry

end log_neq_x_minus_one_l133_133286


namespace alex_original_seat_was_six_l133_133598

-- Define the problem setup and conditions
structure TheaterFriends :=
  (num_seats : ℕ) 
  (initial_positions : Fin num_seats → String) -- map seat numbers to friends

-- Conditions given in the problem
def friends_initial := {
  num_seats := 6,
  initial_positions := λ n, 
    if n = 0 then "Alex" 
    else if n = 1 then "Bob" 
    else if n = 2 then "Carol"
    else if n = 3 then "Dave"
    else if n = 4 then "Eve"
    else "Faye"
}

def bob_move_right (pos : Fin 6) : Fin 6 := if pos = 1 then 2 else pos
def carol_move_left (pos : Fin 6) : Fin 6 := if pos = 2 then 0 else pos
def dave_eve_switch (pos : Fin 6) : Fin 6 := if pos = 3 then 4 else if pos = 4 then 3 else pos
def faye_move_right (pos : Fin 6) : Fin 6 := if pos = 5 then 5 else if pos = 4 then 5 else pos

def final_positions (init : Fin 6 → String) : Fin 6 → String :=
  λ pos, 
    let pos := bob_move_right pos in
    let pos := carol_move_left pos in
    let pos := dave_eve_switch pos in
    let pos := faye_move_right pos in
    init pos

-- Prove that Alex's original seat was 6
theorem alex_original_seat_was_six : friends_initial.initial_positions (6 - 1) = "Alex" 
  := by 
    sorry

end alex_original_seat_was_six_l133_133598


namespace geometric_series_sum_l133_133550

-- Defining the geometric sequence terms and constants
def a1 : ℝ := 8
def q : ℝ := 1 / 2
def an : ℝ := 1 / 2

-- The formula for the sum of the first n terms in a geometric sequence
def Sn : ℝ := (a1 - an * q) / (1 - q)

-- The theorem statement
theorem geometric_series_sum :
  Sn = 31 / 2 := 
sorry

end geometric_series_sum_l133_133550


namespace ellipse_intersection_area_condition_l133_133023

theorem ellipse_intersection_area_condition :
  let ellipse := ∀ (x y : ℝ), (x^2 / 3 + y^2 = 1)
  let F1 := (-Real.sqrt 2, 0)
  let F2 := (Real.sqrt 2, 0)
  let line m := ∀ (x y : ℝ), (y = x + m)
  ∀ (A B : ℝ × ℝ),
    ((A.1^2 / 3 + A.2^2 = 1) ∧ (y = A.1 + m)) ∧
    ((B.1^2 / 3 + B.2^2 = 1) ∧ (y = B.1 + m)) ∧
    (area (F1, A, B) = 2 * area (F2, A, B)) →
    m = -Real.sqrt 2 / 3 := by sorry

end ellipse_intersection_area_condition_l133_133023


namespace part_I_part_II_l133_133169

section triangle_problem

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Condition for part (I)
def condition_I (a c C : ℝ) := 
  c = Real.sqrt 6 ∧
  C = 2 * Real.pi / 3 ∧
  a = Real.sqrt 2 

def result_I (b : ℝ) := 
  b = Real.sqrt 2

theorem part_I : condition_I a c C → result_I b :=
by sorry

-- Condition for part (II)
def condition_II (a b c A B C : ℝ) := 
  c = Real.sqrt 6 ∧
  C = 2 * Real.pi / 3 ∧
  sin B = 2 * sin A

def result_II (area : ℝ) := 
  area = 3 * Real.sqrt 3 / 7

theorem part_II : condition_II a b c A B C → result_II ((1/2) * a * b * sin C) :=
by sorry

end triangle_problem

end part_I_part_II_l133_133169


namespace potatoes_fraction_l133_133750

theorem potatoes_fraction (w : ℝ) (x : ℝ) (h_weight : w = 36) (h_fraction : w / x = 36) : x = 1 :=
by
  sorry

end potatoes_fraction_l133_133750


namespace find_third_root_l133_133311

variables (a b : ℝ)

def poly (x : ℝ) : ℝ := a * x^3 + (a + 3 * b) * x^2 + (b - 4 * a) * x + (10 - a)

def root1 := -3
def root2 := 4

axiom root1_cond : poly a b root1 = 0
axiom root2_cond : poly a b root2 = 0

theorem find_third_root (a b : ℝ) (h1 : poly a b root1 = 0) (h2 : poly a b root2 = 0) : 
  ∃ r3 : ℝ, r3 = -1/2 :=
sorry

end find_third_root_l133_133311


namespace negation_abs_lt_one_l133_133638

theorem negation_abs_lt_one (x : ℝ) : (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by
  sorry

end negation_abs_lt_one_l133_133638


namespace gcd_three_digit_palindromes_l133_133704

theorem gcd_three_digit_palindromes : 
  GCD (set.image (λ (p : ℕ × ℕ), 101 * p.1 + 10 * p.2) 
    ({a | a ≠ 0 ∧ a < 10} × {b | b < 10})) = 1 := 
by
  sorry

end gcd_three_digit_palindromes_l133_133704


namespace find_a2_plus_b2_l133_133459

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a * b = -1) (h2 : a - b = 2) : a^2 + b^2 = 2 := 
by
  sorry

end find_a2_plus_b2_l133_133459


namespace eighth_graders_only_math_is_39_l133_133626

noncomputable def eighth_graders_only_math 
  (total_students : ℕ) 
  (math_students : ℕ) 
  (foreign_language_students : ℕ) 
  (both_classes_students : ℕ) : ℕ :=
  math_students - both_classes_students

theorem eighth_graders_only_math_is_39 
  (total_students : ℕ := 93) 
  (math_students : ℕ := 70)
  (foreign_language_students : ℕ := 54) 
  (both_classes_students_calc : ℕ := math_students + foreign_language_students - total_students) 
  : eighth_graders_only_math total_students math_students foreign_language_students both_classes_students_calc = 39 :=
by
  have total_students_correct : total_students = 93 := rfl
  have math_students_correct : math_students = 70 := rfl
  have foreign_language_students_correct : foreign_language_students = 54 := rfl
  have both_classes_students_correct : both_classes_students_calc = 31 := by
    rw [math_students_correct, foreign_language_students_correct, total_students_correct]
    norm_num
  rw [both_classes_students_correct]
  norm_num
  sorry

end eighth_graders_only_math_is_39_l133_133626


namespace greatest_common_factor_of_three_digit_palindromes_l133_133672

def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b

def gcf (a b : ℕ) : ℕ := 
  if a = 0 then b else gcf (b % a) a

theorem greatest_common_factor_of_three_digit_palindromes : 
  ∃ g, (∀ n, is_palindrome n → g ∣ n) ∧ (∀ d, (∀ n, is_palindrome n → d ∣ n) → d ∣ g) :=
by
  use 101
  sorry

end greatest_common_factor_of_three_digit_palindromes_l133_133672


namespace area_ratio_ellipse_l133_133085

noncomputable def ellipse := { x // (x.1^2 / 3 + x.2^2 = 1) }
def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))
def intersects (line : ℝ × ℝ) (C : ℝ × ℝ → Prop) := ∃ (x y : ℝ), (C (x, y)) ∧ (y = x + line.2)

theorem area_ratio_ellipse 
  (a b : ℝ) (h_ellipse : a^2 = 3 ∧ b^2 = 1) (m : ℝ) 
  (h_foci : foci a b = ((-real.sqrt (a^2 - b^2), 0), (real.sqrt (a^2 - b^2), 0))) 
  (h_condition : ∀ (x : ℝ), intersects (x, m) (λ pt, pt.1^2 / 3 + pt.2^2 = 1) →
     |((x - real.sqrt (a^2 - b^2)) + m)| = 2|((x + real.sqrt (a^2 - b^2)) + m)|) :
  m = -real.sqrt(2) / 3 := sorry

end area_ratio_ellipse_l133_133085


namespace find_f_pi_l133_133432

noncomputable def f (x : ℝ) : ℝ := sorry

-- Given conditions
axiom f_condition : ∀ x : ℝ, f x + 2 * f (π / 2 - x) = Real.sin x

theorem find_f_pi : f π = -2 / 3 := by
  sorry

end find_f_pi_l133_133432


namespace gcd_three_digit_palindromes_l133_133702

open Nat

theorem gcd_three_digit_palindromes :
  (∀ a b : ℕ, a ≠ 0 → a < 10 → b < 10 → True) ∧
  let S := {n | ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b} in
  S.Gcd = 1 := by
  sorry

end gcd_three_digit_palindromes_l133_133702


namespace tan_sum_l133_133143

theorem tan_sum (x y : ℝ) (h1 : Real.sin x + Real.sin y = 85 / 65) (h2 : Real.cos x + Real.cos y = 60 / 65) :
  Real.tan x + Real.tan y = 17 / 12 :=
sorry

end tan_sum_l133_133143


namespace calculation_is_correct_l133_133798

theorem calculation_is_correct : -1^6 + 8 / (-2)^2 - abs (-4 * 3) = -9 := by
  sorry

end calculation_is_correct_l133_133798


namespace college_girls_count_l133_133526

theorem college_girls_count (B G : ℕ) (h1 : B / G = 8 / 5) (h2 : B + G = 546) : G = 210 :=
by
  sorry

end college_girls_count_l133_133526


namespace minimum_value_of_y_exists_l133_133416

theorem minimum_value_of_y_exists :
  ∃ (y : ℝ), (∀ (x : ℝ), (y + x) = (y - x)^2 + 3 * (y - x) + 3) ∧ y = -1/2 :=
by sorry

end minimum_value_of_y_exists_l133_133416


namespace angle_APK_eq_angle_LBC_l133_133543

variables (A B C D K L P : Type*) [ConvexQuadrilateral A B C D] [OnLine B D K] [OnLine K C L]
variables [SimTri BAD BKL] [OnLine CD P] [Parallel DP DL]

theorem angle_APK_eq_angle_LBC :
  ∠APK = ∠LBC :=
sorry

end angle_APK_eq_angle_LBC_l133_133543


namespace bella_steps_704_l133_133396
noncomputable def distance_in_feet := 2 * 5280
def bella_speed (b : ℕ) := b
def ella_speed (b : ℕ) := 5 * b
def combined_speed (b : ℕ) := bella_speed b + ella_speed b
def meeting_time (b : ℕ) := distance_in_feet / combined_speed b
def bella_distance (b : ℕ) := bella_speed b * meeting_time b
def bella_step_length := 2.5
def steps_bella_takes (b : ℕ) := bella_distance b / bella_step_length

-- The final proof statement:
theorem bella_steps_704 (b : ℕ) (b_pos : b > 0) : steps_bella_takes b = 704 :=
by sorry

end bella_steps_704_l133_133396


namespace roots_cubic_sum_cubes_l133_133579

theorem roots_cubic_sum_cubes (a b c : ℝ) 
    (h1 : 6 * a^3 - 803 * a + 1606 = 0)
    (h2 : 6 * b^3 - 803 * b + 1606 = 0)
    (h3 : 6 * c^3 - 803 * c + 1606 = 0) :
    (a + b)^3 + (b + c)^3 + (c + a)^3 = 803 := 
by
  sorry

end roots_cubic_sum_cubes_l133_133579


namespace simplify_f_eval_f_at_neg_25_div_4_pi_l133_133017

def f (α : ℝ) : ℝ := 
  (sin (2 * π - α) * cos (π + α) * cos (π / 2 + α) * cos (11 * π / 2 - α)) / 
  (2 * sin (3 * π + α) * sin (-π - α) * sin (9 * π / 2 + α))

theorem simplify_f (α : ℝ) : f α = (1 / 2) * tan α :=
sorry

theorem eval_f_at_neg_25_div_4_pi : f (-25 / 4 * π) = -sqrt 2 / 4 :=
sorry

end simplify_f_eval_f_at_neg_25_div_4_pi_l133_133017


namespace zero_vector_expressions_l133_133597

-- Definitions of vector operations
variables {V : Type*} [add_comm_group V]

-- Given conditions for the vector expressions
def expr1 (A B C : V) : V := A + B + C
def expr2 (A C B D : V) : V := A - C + B - D
def expr3 (A D : V) : V := A - D + (D - A)
def expr4 (N Q P M : V) : V := N + Q + M - P

-- Mathematically equivalent proof problem
theorem zero_vector_expressions (A B C D N Q P M : V) :
  (expr1 A B C = 0 ∧ expr2 A C B D = 0 ∧ expr3 A D = 0 ∧ expr4 N Q P M = 0) →
  4 = 4 :=
by
  intros h
  sorry

end zero_vector_expressions_l133_133597


namespace functional_eq_solution_l133_133863

theorem functional_eq_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x^2 + y ≠ 0 → f(x^2 + y) = f(x)^2 + f(x * y) / f(x)) : ∀ x ≠ 0, f x = x :=
by
  sorry

end functional_eq_solution_l133_133863


namespace find_m_value_l133_133049

-- Define the ellipse equation and its foci
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define the foci of the ellipse C
def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the line equation
def line (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the condition for the areas of triangles F1AB and F2AB
def triangle_area_condition (F1 F2 : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := sorry in -- Coordinates of intersection points need to be defined
  let B := sorry in -- Coordinates of intersection points need to be defined
  sorry /- The actual calculation of the areas would require more complex integration and geometry.
           Here we just outline the condition -/

-- The proof statement
theorem find_m_value :
  ∃ m : ℝ, 
    ∀ x y : ℝ,
      ellipse x y →
      line m x y →
      triangle_area_condition F1 F2 m →
      m = -Real.sqrt 2 / 3 :=
  sorry

end find_m_value_l133_133049


namespace san_antonio_bound_bus_passes_austin_bound_buses_l133_133793

-- Define the structure of the problem, including the conditions:

/-- A bus schedule where buses from Austin to San Antonio leave every hour on the hour,
 and buses from San Antonio to Austin leave every hour at quarter past the hour.
 The trip from one city to the other takes 4 hours.
 Given a San Antonio-bound bus, we want to prove it passes 4 Austin-bound buses in transit. -/
def bus_schedule (time_in_hours : ℕ → Bool) : Prop :=
  ∀ (sa_to_austin_start : ℕ) (austin_to_sa_start : ℕ),
    (sa_to_austin_start % 4 = 1) ∧ (austin_to_sa_start % 4 = 0) ∧
    time_in_hours = 4 ∧
    time_in_hours / 4 = 1

/-- Proof of the problem -/
theorem san_antonio_bound_bus_passes_austin_bound_buses :
  bus_schedule time_in_hours →  -- Assumes the defined condition of the schedule
  ∃ n, n = 4 :=                 -- Proves that the number of passes is 4
by
  sorry

end san_antonio_bound_bus_passes_austin_bound_buses_l133_133793


namespace min_bottles_needed_l133_133351

theorem min_bottles_needed (num_people : ℕ) (exchange_rate : ℕ) (bottles_needed_per_person : ℕ) (total_bottles_purchased : ℕ):
  num_people = 27 → exchange_rate = 3 → bottles_needed_per_person = 1 → total_bottles_purchased = 18 → 
  ∀ n, n = num_people → (n / bottles_needed_per_person) = 27 ∧ (num_people * 2 / 3) = 18 :=
by
  intros
  sorry

end min_bottles_needed_l133_133351


namespace fraction_green_tin_l133_133726

variable (C : ℕ) -- Let the total number of cookies be C.

-- Definition of conditions
def two_thirds_cookies_blue_or_green := (2/3 : ℚ) * C
def one_quarter_cookies_blue := (1/4 : ℚ) * C
def remaining_cookies_red := (1/3 : ℚ) * C

-- Fraction of cookies in the green tin
def cookies_green := two_thirds_cookies_blue_or_green - one_quarter_cookies_blue
def total_cookies_blue_green := one_quarter_cookies_blue + cookies_green

-- Prove the statement
theorem fraction_green_tin : 
  (cookies_green / total_cookies_blue_green = (5/8 : ℚ)) := 
sorry

end fraction_green_tin_l133_133726


namespace even_function_translation_l133_133883

theorem even_function_translation (a : ℝ) (h : 0 < a ∧ a < π / 2) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = 3 * Real.sin (2 * x - π / 3)) : 
  (∀ x, f(x + a) = f(-x - a)) → a = 5 * π / 12 :=
by
  intros h₁ h₂ h₃
  sorry

end even_function_translation_l133_133883


namespace find_number_l133_133352

theorem find_number (x : ℝ) (h : 0.50 * x = 0.30 * 50 + 13) : x = 56 :=
by
  sorry

end find_number_l133_133352


namespace domain_of_y_l133_133921

theorem domain_of_y (f : ℝ → ℝ) (x : ℝ) 
  (Hf : ∀ t, -3 / 2 ≤ t ∧ t ≤ -1 → f t = f t) :
  (∃ y, y = (f (1 / x)) / (sqrt (x + 1))) → (-1 < x ∧ x ≤ - 1 / 2) :=
sorry

end domain_of_y_l133_133921


namespace other_number_in_product_l133_133912

theorem other_number_in_product (w : ℕ) (n : ℕ) (hw_pos : 0 < w) (n_factor : Nat.lcm (2^5) (Nat.gcd  864 w) = 2^5 * 3^3) (h_w : w = 144) : n = 6 :=
by
  -- proof would go here
  sorry

end other_number_in_product_l133_133912


namespace find_first_offset_l133_133436

theorem find_first_offset (x : ℝ) : 
  let area := 180
  let diagonal := 24
  let offset2 := 6
  (area = (diagonal * (x + offset2)) / 2) -> x = 9 :=
sorry

end find_first_offset_l133_133436


namespace digit_a_solution_l133_133320

theorem digit_a_solution :
  ∃ a : ℕ, a000 + a998 + a999 = 22997 → a = 7 :=
sorry

end digit_a_solution_l133_133320


namespace german_math_olympiad_problem_2003_l133_133914

theorem german_math_olympiad_problem_2003 (n : ℕ) (a : ℕ → ℕ) (h₀ : 0 < n)
  (h₁ : ∀ n, Nat.find (λ m, n ∣ m.factorial) = a n)
  (h₂ : a n = (2 * n) / 3) :
  n = 9 :=
by
  have h₃ : 9 ∣ 6.factorial := by norm_num
  use 9, 6
  sorry

end german_math_olympiad_problem_2003_l133_133914


namespace gcd_three_digit_palindromes_l133_133701

open Nat

theorem gcd_three_digit_palindromes :
  (∀ a b : ℕ, a ≠ 0 → a < 10 → b < 10 → True) ∧
  let S := {n | ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b} in
  S.Gcd = 1 := by
  sorry

end gcd_three_digit_palindromes_l133_133701


namespace david_marks_in_english_l133_133419

theorem david_marks_in_english (marks_math: ℕ) (marks_physics: ℕ) (marks_chemistry: ℕ) (marks_biology: ℕ) (average_marks: ℕ) (num_subjects: ℕ) (total_marks: ℕ) : 
  marks_math = 35 →
  marks_physics = 42 →
  marks_chemistry = 57 →
  marks_biology = 55 →
  average_marks = 45 →
  num_subjects = 5 →
  total_marks = 45 * 5 →
  ∑ (marks_math + marks_physics + marks_chemistry + marks_biology + english_marks) = total_marks →
  english_marks = 36 :=
by
  intros
  sorry

end david_marks_in_english_l133_133419


namespace hyperbola_asymptote_angle_l133_133847

theorem hyperbola_asymptote_angle
  (a b : ℝ) 
  (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → a > b)
  (h_angle : ∀ θ : ℝ, θ = 60 → θ = real.arccos((a^2 - b^2) / (a^2 + b^2))) :
  a / b = real.sqrt 3 := sorry

end hyperbola_asymptote_angle_l133_133847


namespace gcd_of_all_three_digit_palindromes_is_one_l133_133694

-- Define what it means to be a three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

-- Define a function to calculate the gcd of a list of numbers
def gcd_list (l : List ℕ) : ℕ :=
  l.foldr Nat.gcd 0

-- The main theorem that needs to be proven
theorem gcd_of_all_three_digit_palindromes_is_one :
  gcd_list (List.filter is_palindrome {n | 100 ≤ n ∧ n ≤ 999}.toList) = 1 :=
by
  sorry

end gcd_of_all_three_digit_palindromes_is_one_l133_133694


namespace curve_and_line_properties_l133_133945

noncomputable def general_equation_of_curve (x y α : ℝ) : Prop :=
  x = 2 + 3 * Real.cos α ∧ y = 3 * Real.sin α

noncomputable def polar_to_cartesian (ρ θ : ℝ) : (ℝ × ℝ) :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem curve_and_line_properties 
  (x y : ℝ)
  (α : ℝ)
  (t₁ t₂ : ℝ)
  (P : ℝ × ℝ := (0, -1))
  (A B : ℝ × ℝ)
  (ρ θ : ℝ) :
  general_equation_of_curve x y α →
  polar_to_cartesian ρ θ = (x, y) →
  (2 * x - y - 1 = 0) →
  (x = 2 + 3 * Real.cos α ∧ y = 3 * Real.sin α) →
  A ≠ B →
  (P = (0, -1)) →
  (A.1, A.2, B.1, B.2) ∈ ({t // t^2 - (8*5^.5)/5*t - 4=0}) →
  (1 / Real.abs (Real.sqrt (A.1^2 + A.2^2)) + 1 / Real.abs (Real.sqrt (B.1^2 + B.2^2)) = 3*Real.sqrt 5 / 5) :=
begin
  sorry
end

end curve_and_line_properties_l133_133945


namespace measure_angle_B_sum_of_sides_a_c_l133_133927

-- Define that in triangle ABC, angle B can be computed given the conditions
theorem measure_angle_B (A B C : ℝ) (a b c : ℝ) (h1 : b = 4) 
  (h2 : b * cos A = (2 * c + a) * cos (π - B)) : B = 2 * π / 3 := sorry

-- Define that in triangle ABC, given specific conditions, we can compute a + c
theorem sum_of_sides_a_c (A B C : ℝ) (a b c : ℝ) 
  (h1 : b = 4) 
  (h2 : b * cos A = (2 * c + a) * cos (π - B))
  (h3 : (1 / 2) * a * c * sin B = sqrt 3) : a + c = 2 * sqrt 5 := sorry

end measure_angle_B_sum_of_sides_a_c_l133_133927


namespace merck_hourly_rate_l133_133958

-- Define the relevant data from the problem
def hours_donaldsons : ℕ := 7
def hours_merck : ℕ := 6
def hours_hille : ℕ := 3
def total_earnings : ℕ := 273

-- Define the total hours based on the conditions
def total_hours : ℕ := hours_donaldsons + hours_merck + hours_hille

-- Define what we want to prove:
def hourly_rate := total_earnings / total_hours

theorem merck_hourly_rate : hourly_rate = 273 / (7 + 6 + 3) := by
  sorry

end merck_hourly_rate_l133_133958


namespace calculate_f_neg3_l133_133806

noncomputable def f : ℝ → ℝ := sorry -- definition of the function f

axiom functional_equation (x y : ℝ) : f(x + y) = f(x) + f(y) + 2 * x * y
axiom f_one : f(1) = 2

theorem calculate_f_neg3 : f(-3) = 6 := by
  sorry

end calculate_f_neg3_l133_133806


namespace pyramid_apex_angle_l133_133238

theorem pyramid_apex_angle (A B C D E O : Type) 
  (square_base : Π (P Q : Type), Prop) 
  (isosceles_triangle : Π (R S T : Type), Prop)
  (AEB_angle : Π (X Y Z : Type), Prop) 
  (angle_AOB : ℝ)
  (angle_AEB : ℝ)
  (square_base_conditions : square_base A B ∧ square_base B C ∧ square_base C D ∧ square_base D A)
  (isosceles_triangle_conditions : isosceles_triangle A E B ∧ isosceles_triangle B E C ∧ isosceles_triangle C E D ∧ isosceles_triangle D E A)
  (center : O)
  (diagonals_intersect_at_right_angle : angle_AOB = 90)
  (measured_angle_at_apex : angle_AEB = 100) :
False :=
sorry

end pyramid_apex_angle_l133_133238


namespace max_y_coordinate_is_1_l133_133827

theorem max_y_coordinate_is_1 :
  ∃ θ : ℝ, (r = sin (3 * θ) → y = r * sin θ) → (∃ θ_0, sin θ_0 = 1 / 2 ∧ y = 1) :=
sorry

end max_y_coordinate_is_1_l133_133827


namespace trapezium_area_l133_133338

theorem trapezium_area (a b d : ℕ) (h₁ : a = 28) (h₂ : b = 18) (h₃ : d = 15) :
  (a + b) * d / 2 = 345 := by
{
  sorry
}

end trapezium_area_l133_133338


namespace product_floor_ceil_evaluation_l133_133819

theorem product_floor_ceil_evaluation : 
  (List.prod (List.map (λ n, (Int.floor (-n - 0.5) * Int.ceil (n + 0.5))) [5, 4, 3, 2, 1, 0])) = -518400 := 
by
  sorry

end product_floor_ceil_evaluation_l133_133819


namespace gcf_of_all_three_digit_palindromes_l133_133712

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ is_palindrome n

def gcf_of_palindromes : ℕ :=
  101

theorem gcf_of_all_three_digit_palindromes : 
  ∀ n, is_three_digit_palindrome n → 101 ∣ n := by
    sorry

end gcf_of_all_three_digit_palindromes_l133_133712


namespace valid_probabilities_and_invalid_probability_l133_133266

theorem valid_probabilities_and_invalid_probability :
  (let first_box_1 := (4, 7)
       second_box_1 := (3, 5)
       combined_prob_1 := (first_box_1.1 + second_box_1.1) / (first_box_1.2 + second_box_1.2),
       first_box_2 := (8, 14)
       second_box_2 := (3, 5)
       combined_prob_2 := (first_box_2.1 + second_box_2.1) / (first_box_2.2 + second_box_2.2),
       prob_1 := first_box_1.1 / first_box_1.2,
       prob_2 := second_box_2.1 / second_box_2.2
     in (combined_prob_1 = 7 / 12 ∧ combined_prob_2 = 11 / 19) ∧ (19 / 35 < prob_1 ∧ prob_1 < prob_2) → False) :=
by
  sorry

end valid_probabilities_and_invalid_probability_l133_133266


namespace circle_radius_equation_l133_133885

theorem circle_radius_equation (x y : ℝ) :
  x^2 + y^2 - 4*x - 2*y - 5 = 0 → ∃ r, r = sqrt 10 :=
by
  sorry

end circle_radius_equation_l133_133885


namespace gcd_three_digit_palindromes_l133_133678

theorem gcd_three_digit_palindromes : ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → Nat.gcd (102 * a + 10 * b) 1 = 1 :=
by
  intros a b h
  sorry

end gcd_three_digit_palindromes_l133_133678


namespace complex_number_solution_l133_133518

open Complex

theorem complex_number_solution (z : ℂ) (h1 : (z - 3 * I) / (z + I) ∈ set_of (λ x, x < 0))
 h2 : (z - 3) / (z + 1) ∈ set_of (λ x, Im x = (0 : ℝ)) :
 z = sqrt 3 * I ∨ z = - sqrt 3 * I :=
begin
  sorry
end

end complex_number_solution_l133_133518


namespace cost_of_article_l133_133515

theorem cost_of_article (C G1 G2 : ℝ) (h1 : G1 = 380 - C) (h2 : G2 = 450 - C) (h3 : G2 = 1.10 * G1) : 
  C = 320 :=
by
  sorry

end cost_of_article_l133_133515


namespace race_course_length_l133_133752

theorem race_course_length (v : ℝ) (d : ℝ) (h1 : 4 * (d - 69) = d) : d = 92 :=
by
  sorry

end race_course_length_l133_133752


namespace gcf_of_all_three_digit_palindromes_l133_133711

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ is_palindrome n

def gcf_of_palindromes : ℕ :=
  101

theorem gcf_of_all_three_digit_palindromes : 
  ∀ n, is_three_digit_palindrome n → 101 ∣ n := by
    sorry

end gcf_of_all_three_digit_palindromes_l133_133711


namespace jakes_digging_time_l133_133560

theorem jakes_digging_time
  (J : ℕ)
  (Paul_work_rate : ℚ := 1/24)
  (Hari_work_rate : ℚ := 1/48)
  (Combined_work_rate : ℚ := 1/8)
  (Combined_work_eq : 1 / J + Paul_work_rate + Hari_work_rate = Combined_work_rate) :
  J = 16 := sorry

end jakes_digging_time_l133_133560


namespace lucas_income_36000_l133_133294

variable (q I : ℝ)

-- Conditions as Lean 4 definitions
def tax_below_30000 : ℝ := 0.01 * q * 30000
def tax_above_30000 (I : ℝ) : ℝ := 0.01 * (q + 3) * (I - 30000)
def total_tax (I : ℝ) : ℝ := tax_below_30000 q + tax_above_30000 q I
def total_tax_condition (I : ℝ) : Prop := total_tax q I = 0.01 * (q + 0.5) * I

theorem lucas_income_36000 (h : total_tax_condition q I) : I = 36000 := by
  sorry

end lucas_income_36000_l133_133294


namespace tangent_line_at_point_is_correct_l133_133249

-- Define the curve as a function
def curve (x : ℝ) : ℝ := Real.sqrt (1 - x)

-- Define the point where we calculate the tangent
def point : ℝ × ℝ := (3/4, 1/2)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 4 * x + 4 * y - 5 = 0

-- State the problem
theorem tangent_line_at_point_is_correct :
  tangent_line (point.1) (curve (point.1)) :=
  sorry

end tangent_line_at_point_is_correct_l133_133249


namespace correct_optionD_l133_133722

def operationA (a : ℝ) : Prop := a^3 + 3 * a^3 = 5 * a^6
def operationB (a : ℝ) : Prop := 7 * a^2 * a^3 = 7 * a^6
def operationC (a : ℝ) : Prop := (-2 * a^3)^2 = 4 * a^5
def operationD (a : ℝ) : Prop := a^8 / a^2 = a^6

theorem correct_optionD (a : ℝ) : ¬ operationA a ∧ ¬ operationB a ∧ ¬ operationC a ∧ operationD a :=
by
  unfold operationA operationB operationC operationD
  sorry

end correct_optionD_l133_133722


namespace inequality_proof_l133_133595

-- Define the main theorem with the conditions
theorem inequality_proof 
  (a b c d : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) ∧
  ((a = b ∧ b = c ∧ c = d) ↔ (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a)) := 
sorry

end inequality_proof_l133_133595


namespace midsegment_length_is_10_l133_133542

-- Define the problem setup
structure IsoscelesTrapezoid where
  height : ℝ
  diagonals_perpendicular : Prop

-- Assume we have an isosceles trapezoid with height 10 and perpendicular diagonals
def example_trapezoid : IsoscelesTrapezoid := {
  height := 10,
  diagonals_perpendicular := true
}

-- Prove that the midsegment of the isosceles trapezoid is 10
theorem midsegment_length_is_10 (T : IsoscelesTrapezoid) (hT : T = example_trapezoid) : T.height = 10 → 10 = 10 :=
by
  intros h_height
  rw [h_height]
  exact eq.refl 10
  -- Other required properties and detailed steps are skipped
  -- Add additional justification if necessary
  sorry

end midsegment_length_is_10_l133_133542


namespace find_coordinates_M_l133_133128

-- Define point structures and mathematical statements
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def vec (p1 p2 : Point) : Point := 
  ⟨p2.x - p1.x, p2.y - p1.y, p2.z - p1.z⟩

def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def on_line (p M: Point) (λ : ℝ) : Prop :=
  M = ⟨λ * p.x, λ * p.y, p.z⟩

def perpendicular (v1 v2 : Point) : Prop :=
  dot_product v1 v2 = 0

noncomputable def find_point_M : Point :=
  ⟨-1/2, 1/2, 1⟩

theorem find_coordinates_M:
  ∃ (M : Point), 
    on_line (Point.mk (-1) 1 1) M (1/2) ∧ 
    perpendicular (vec (Point.mk 1 2 (-3)) M) (vec (Point.mk 0 0 1) (Point.mk (-1) 1 1)) :=
begin
  use find_point_M,
  unfold on_line perpendicular vec dot_product find_point_M Point.mk,
  split,
  { simp [find_point_M] },
  {
    have : vec (Point.mk 1 2 (-3)) (find_point_M) = Point.mk (-3/2) (-3/2) 4,
    { simp [find_point_M, vec, Point.mk] },
    simp [*, dot_product, Point.mk],
  }
end

end find_coordinates_M_l133_133128


namespace functional_equation_solution_l133_133434

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x + y) + x * y = f(x) * f(y)) →
  (f = (λ x, 1 - x) ∨ f = (λ x, x + 1)) :=
by
  sorry

end functional_equation_solution_l133_133434


namespace simplify_expression_l133_133580

variable {a b c k : ℝ}
variable (a_nonzero : a ≠ 0) 
variable (b_nonzero : b ≠ 0)
variable (c_nonzero : c ≠ 0)
variable (k_nonzero : k ≠ 0)

def x : ℝ := k * (b / c) + (c / b)
def y : ℝ := k * (a / c) + (c / a)
def z : ℝ := k * (a / b) + (b / a)

theorem simplify_expression : 
  (x a b c k)^2 + (y a b c k)^2 + (z a b c k)^2 - (x a b c k) * (y a b c k) * (z a b c k) = 0 :=
by
  sorry

end simplify_expression_l133_133580


namespace always_monochromatic_rectangle_l133_133938

-- Define the region D
def D : set (ℕ × ℕ) := {p | 1 ≤ p.1 ∧ p.1 ≤ 3 ∧ 1 ≤ p.2 ∧ p.2 ≤ 7}

-- Define the possible colors
inductive Color
| Red
| Blue

open Color

-- Color each point in D
def color (p : ℕ × ℕ) : Color := sorry -- Arbitrary coloring function

theorem always_monochromatic_rectangle :
  ∀ (coloring : (ℕ × ℕ) → Color), 
  ∃ (p1 p2 p3 p4 : ℕ × ℕ),
    p1 ∈ D ∧ p2 ∈ D ∧ p3 ∈ D ∧ p4 ∈ D ∧
    p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.2 = p3.2 ∧ p2.2 = p4.2 ∧
    color p1 = color p2 ∧ color p2 = color p3 ∧ color p3 = color p4 :=
begin
  sorry -- Proof goes here
end

end always_monochromatic_rectangle_l133_133938


namespace sequence_formula_l133_133118

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧
  (∀ n ≥ 2, (∑ k in Finset.range n, a (k + 1)) = n ^ 2 * a n) ∧
  (∀ n ≥ 2, a n = (n ^ 2 * a n - (n - 1) ^ 2 * a (n - 1)))

theorem sequence_formula (a : ℕ → ℝ) (h : sequence a) :
  ∀ n, a n = 2 / (n * (n + 1)) :=
sorry

end sequence_formula_l133_133118


namespace x_intercept_of_line_l133_133379

open Real

def line_through_points : Real × Real → Real × Real → (Real → Real) :=
  λ p1 p2, λ x, ((p2.2 - p1.2) / (p2.1 - p1.1)) * (x - p1.1) + p1.2

theorem x_intercept_of_line :
  let p1 := (1 : Real, 3 : Real)
  let p2 := (5 : Real, -1 : Real)
  let line := line_through_points p1 p2
  ∃ x0 : Real, line x0 = 0 ∧ x0 = 4 :=
by
  let p1 := (1 : Real, 3 : Real)
  let p2 := (5 : Real, -1 : Real)
  let line := line_through_points p1 p2
  use 4
  sorry

end x_intercept_of_line_l133_133379


namespace speaker_is_male_nurse_l133_133529

-- Defining the variables for the conditions
variables (a b c d : ℕ)
variables (a + b ≥ c + d) (d > a) (a > b) (c ≥ 1) (a + b + c + d + 1 = 13)

-- Main theorem statement
theorem speaker_is_male_nurse (a b c d : ℕ)
  (h1 : a + b ≥ c + d)
  (h2 : d > a)
  (h3 : a > b)
  (h4 : c ≥ 1)
  (h5 : a + b + c + d + 1 = 13) :
  -- Conclusion: The speaker is a male nurse
  (a = 4) ∧ (b = 3) ∧ (d = 5) ∧ (c = 1)
:= sorry -- proof is omitted

end speaker_is_male_nurse_l133_133529


namespace triangle_AC_length_l133_133173

theorem triangle_AC_length (A B C : Type) [euclidean.metric_space A] [euclidean.metric_space B] (AB : Real) (B C : Real) :
  AB = 2 → 
  angle B = π / 3 → 
  angle C = π / 4 → 
  ∃ AC : Real, AC = sqrt 6 := 
by
  sorry

end triangle_AC_length_l133_133173


namespace solution_to_equation_l133_133293

theorem solution_to_equation :
  ∃ x ∈ set.Ioo 0 real.pi, 2 * real.cos (x - real.pi / 4) = 1 ∧ x = 7 * real.pi / 12 :=
by
  sorry

end solution_to_equation_l133_133293


namespace Seokjin_tangerines_per_day_l133_133594

theorem Seokjin_tangerines_per_day 
  (T_initial : ℕ) (D : ℕ) (T_remaining : ℕ) 
  (h1 : T_initial = 29) 
  (h2 : D = 8) 
  (h3 : T_remaining = 5) : 
  (T_initial - T_remaining) / D = 3 := 
by
  sorry

end Seokjin_tangerines_per_day_l133_133594


namespace sqrt_expression_l133_133408

theorem sqrt_expression :
  (Real.sqrt (2 ^ 4 * 3 ^ 6 * 5 ^ 2)) = 540 := sorry

end sqrt_expression_l133_133408


namespace steven_has_72_shirts_l133_133604

def brian_shirts : ℕ := 3
def andrew_shirts (brian : ℕ) : ℕ := 6 * brian
def steven_shirts (andrew : ℕ) : ℕ := 4 * andrew

theorem steven_has_72_shirts : steven_shirts (andrew_shirts brian_shirts) = 72 := 
by 
  -- We add "sorry" here to indicate that the proof is omitted
  sorry

end steven_has_72_shirts_l133_133604


namespace total_photos_newspaper_l133_133533

-- Define the conditions
def num_pages1 := 12
def photos_per_page1 := 2
def num_pages2 := 9
def photos_per_page2 := 3

-- Define the total number of photos
def total_photos := (num_pages1 * photos_per_page1) + (num_pages2 * photos_per_page2)

-- Prove that the total number of photos is 51
theorem total_photos_newspaper : total_photos = 51 := by
  -- We will fill the proof later
  sorry

end total_photos_newspaper_l133_133533


namespace numbered_triangles_bound_l133_133389

-- Definitions of the conditions
def equilateral_triangle_divided (n : ℕ) := n^2

def numbered_triangles_condition (n m : ℕ) := 
  ∀ x, x ≤ m → (x ∈ (range (n^2)) ∧ ∀ y, y = x + 1 → adjacent x y)

-- Main theorem statement
theorem numbered_triangles_bound (n m : ℕ) 
  (condition : numbered_triangles_condition n m) :
  m ≤ n^2 - n + 1 :=
sorry

end numbered_triangles_bound_l133_133389


namespace find_a_2b_l133_133430

theorem find_a_2b 
  (a b : ℤ) 
  (h1 : a * b = -150) 
  (h2 : a + b = -23) : 
  a + 2 * b = -55 :=
sorry

end find_a_2b_l133_133430


namespace gcd_three_digit_palindromes_l133_133698

open Nat

theorem gcd_three_digit_palindromes :
  (∀ a b : ℕ, a ≠ 0 → a < 10 → b < 10 → True) ∧
  let S := {n | ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b} in
  S.Gcd = 1 := by
  sorry

end gcd_three_digit_palindromes_l133_133698


namespace mathematicians_correct_l133_133258

noncomputable def scenario1 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 4 ∧ total1 = 7 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario2 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 8 ∧ total1 = 14 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario3 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  (19 / 35 < 4 / 7) ∧ (4 / 7 < 3 / 5)

noncomputable def probability (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : ℝ :=
  (whites1 + whites2) / (total1 + total2)

theorem mathematicians_correct :
  let whites1_s1 := 4 in
  let total1_s1 := 7 in
  let whites2_s1 := 3 in
  let total2_s1 := 5 in
  let whites1_s2 := 8 in
  let total1_s2 := 14 in
  let whites2_s2 := 3 in
  let total2_s2 := 5 in
  scenario1 whites1_s1 total1_s1 whites2_s1 total2_s1 →
  scenario2 whites1_s2 total1_s2 whites2_s2 total2_s2 →
  scenario3 whites1_s1 total1_s1 whites2_s2 total2_s2 →
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 ≤ 3 / 5 ∨
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 = 4 / 7 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 ≤ 3 / 5 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 = 4 / 7 :=
begin
  intros,
  sorry

end mathematicians_correct_l133_133258


namespace distance_from_point_B_to_line_AC_l133_133946

noncomputable def distance_from_point_to_line (a b : ℝ×ℝ×ℝ) : ℝ :=
let ax := a.1 in let ay := a.2 in let az := a.3 in
let bx := b.1 in let by := b.2 in let bz := b.3 in
let u := (bx, by, bz) in
let a := (ax, ay, az) in
let magnitude_u := Real.sqrt (bx^2 + by^2 + bz^2) in
let unit_u := (bx / magnitude_u, by / magnitude_u, bz / magnitude_u) in
let a_dot_u := (ax * unit_u.1 + ay * unit_u.2 + az * unit_u.3) in
let a_square := (ax^2 + ay^2 + az^2) in
let distance := Real.sqrt (a_square - a_dot_u^2) in
distance

theorem distance_from_point_B_to_line_AC :
  distance_from_point_to_line (1, 1, 2) (2, 1, 1) = Real.sqrt 66 / 6 :=
by sorry

end distance_from_point_B_to_line_AC_l133_133946


namespace total_rent_l133_133212

variables {R₅₀ R₆₀ : ℕ}

theorem total_rent (h1: 50 * R₅₀ + 60 * R₆₀ = 400)
                   (h2: 50 * (R₅₀ + 10) + 60 * (R₆₀ - 10) = 0.75 * (50 * R₅₀ + 60 * R₆₀)) :
  50 * R₅₀ + 60 * R₆₀ = 400 :=
by sorry

end total_rent_l133_133212


namespace odd_three_digit_numbers_count_l133_133850

open Finset Nat

def digits := {1, 2, 3, 4, 5}

theorem odd_three_digit_numbers_count : 
  ∃ count : ℕ, 
    count = 36 ∧ 
    ∀ (n : ℕ), 
      (n ∈ digits ∧ n % 2 = 1) →
      (∃ (units : ℕ) (tens : ℕ) (hundreds : ℕ), 
        units ∈ digits ∧ tens ∈ digits ∧ hundreds ∈ digits ∧ 
        units ≠ tens ∧ tens ≠ hundreds ∧ hundreds ≠ units ∧ 
        n = units + 10 * tens + 100 * hundreds) -> 
      count = 36 :=
sorry

end odd_three_digit_numbers_count_l133_133850


namespace hyperbola_eccentricity_l133_133870

variables (a b : ℝ) (P F1 F2 : ℝ × ℝ)
variables (c : ℝ) (h₁ : a > 0) (h₂ : b > 0)

-- P is on the right branch of the hyperbola
def is_on_hyperbola (P : ℝ × ℝ) : Prop :=
  (P.fst^2 / a^2) - (P.snd^2 / b^2) = 1

-- F1, F2 are the foci of the hyperbola
def F1_pos : ℝ := -c
def F2_pos : ℝ := c

-- Given |PF1 + PF2| = 2c
def vector_sum_eq_2c (P : ℝ × ℝ) : Prop :=
  (Real.sqrt ((P.fst - F1_pos)^2 + P.snd^2) + Real.sqrt ((P.fst - F2_pos)^2 + P.snd^2)) = 2 * c

-- Given area of triangle PF1F2 is ac
def triangle_area_eq_ac (P : ℝ × ℝ) : Prop :=
  0.5 * (Real.sqrt ((P.fst - F1_pos)^2 + P.snd^2) * (Real.sqrt ((P.fst - F2_pos)^2 + P.snd^2)) * Real.sin (π/2)) = a * c

-- Prove that eccentricity of the hyperbola is (sqrt(5) + 1) / 2
theorem hyperbola_eccentricity : 
  is_on_hyperbola P →
  vector_sum_eq_2c P →
  triangle_area_eq_ac P →
  ((a^2 + b^2)^0.5 / a) = (Real.sqrt 5 + 1) / 2 :=
sorry

end hyperbola_eccentricity_l133_133870


namespace product_of_two_numbers_l133_133649

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 70) (h2 : x - y = 10) : x * y = 1200 :=
by
  sorry

end product_of_two_numbers_l133_133649


namespace count_7digit_multiple_11_l133_133314

-- Define the digits and their sum
def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7]

-- Define the sum of digits
def sum_digits : ℕ := digits.sum

-- Define the properties of 7-digit number
def is_multiple_of_11 (n : ℕ) : Prop :=
  let digits_list := n.digits 10
  let (odd_sum, even_sum) := digits_list.foldl (λ (acc : ℕ × ℕ) (p : ℕ × ℕ), if p.1 % 2 = 0 then (acc.1 + p.2, acc.2) else (acc.1, acc.2 + p.2)) (0, 0)
  (odd_sum - even_sum) % 11 = 0

-- Prove how many such numbers can be formed
theorem count_7digit_multiple_11 : 
  sum_digits = 28 →
  ∃ count : ℕ, count = 576 ∧
    (∀ n : ℕ, n ∈ List.range (10^7 - 10^6) → is_multiple_of_11 n → ∃ (perm : List ℕ) (h : perm.perm digits), n = perm.foldl (λ acc d, acc * 10 + d) 0) :=
by
  sorry

end count_7digit_multiple_11_l133_133314


namespace circle_center_radius_sum_l133_133193

theorem circle_center_radius_sum :
  ∃ (c d s : ℝ), c = -6 ∧ d = -7 ∧ s = Real.sqrt 13 ∧
  (x^2 + 14 * y + 72 = -y^2 - 12 * x → c + d + s = -13 + Real.sqrt 13) :=
sorry

end circle_center_radius_sum_l133_133193


namespace mathematician_correctness_l133_133253

theorem mathematician_correctness (box1_white1 box1_total1 box2_white1 box2_total1 : ℕ)
                                  (prob1 prob2 : ℚ) :
  (box1_white1 = 4 ∧ box1_total1 = 7 ∧ box2_white1 = 3 ∧ box2_total1 = 5 ∧ prob1 = (4:ℚ) / 7 ∧ prob2 = (3:ℚ) / 5) →
  (box1_white2 = 8 ∧ box1_total2 = 14 ∧ box2_white2 = 3 ∧ box2_total2 = 5 ∧ prob1 = (8:ℚ) / 14 ∧ prob2 = (3:ℚ) / 5) →
  (prob1 < (7:ℚ) / 12 ∧ prob2 > (11:ℚ)/19) →
  ¬((19:ℚ)/35 > (4:ℚ)/7 ∧ (19:ℚ)/35 < (3:ℚ)/5) :=
by
  sorry

end mathematician_correctness_l133_133253


namespace isosceles_triangle_AC_three_times_CE_l133_133162

variable {A B C D M E : Type}
variable [hACBC : A = B] -- Given ∠AC = ∠BC
variable (hD : D) -- D is the foot of the altitude through C
variable (hM : midpoint M C D) -- M is the midpoint of CD
variable (hBM : line B M intersects line A C at E) -- BM intersects AC at E

theorem isosceles_triangle_AC_three_times_CE (A B C D M E : Type)
  (hACBC : A = B) (hD : D) (hM : midpoint M C D) (hBM : line B M intersects line A C at E) :
  A = 3 * E :=
  sorry

end isosceles_triangle_AC_three_times_CE_l133_133162


namespace longest_path_when_Q_equidistant_l133_133246

theorem longest_path_when_Q_equidistant :
  ∀ (O A B C D Q : Point) (r : ℝ),
    Circle.center O ∧
    (∃ (d : ℝ), diameter (LineSegment A B) = 12 ∧ distance A B = d * 2) ∧
    (∃ (d1 d2 : ℝ), distance A C = 3 ∧ distance B D = 3) ∧ 
    (∃ (q : Point), on_circle q O) ∧
    Circle.radius (Circle.mk O Q) = 1 →
    is_longest_path (LineSegment.mk C Q ⟶ LineSegment.mk Q D) when equidistant C Q D :=
sorry

end longest_path_when_Q_equidistant_l133_133246


namespace intersection_of_sets_l133_133500

variable A : Set ℤ := Set.of_list [-1, 2, 4]
variable B : Set ℤ := Set.of_list [-1, 0, 2]

theorem intersection_of_sets : A ∩ B = Set.of_list [-1, 2] := by
    sorry

end intersection_of_sets_l133_133500


namespace problem_1_problem_2_problem_3_l133_133645

def seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n ^ 2 - a n + 1

def initial (a : ℕ → ℕ) : Prop :=
  a 1 = 2

theorem problem_1 (a : ℕ → ℕ) (h_seq : seq a) (h_initial : initial a) : 
  ∀ n : ℕ, a (n + 2) > a n :=
by
  sorry

theorem problem_2 (a : ℕ → ℕ) (h_seq : seq a) (h_initial : initial a) :
  ∀ n : ℕ, n ≥ 2 → (2 ^ 2 ^ (n - 1) < a (n + 1) - 1 ∧ a (n + 1) - 1 < 2 ^ 2 ^ n) :=
by
  sorry

theorem problem_3 (a : ℕ → ℕ) (h_seq : seq a) (h_initial : initial a) :
  let S (n : ℕ) := ∑ i in finset.range (n + 1), (1 / a i)
  in filter.tendsto S filter.at_top (nhds 1) :=
by
  sorry

end problem_1_problem_2_problem_3_l133_133645


namespace roots_sum_of_squares_l133_133200

theorem roots_sum_of_squares {p q r : ℝ} 
  (h₁ : ∀ x : ℝ, (x - p) * (x - q) * (x - r) = x^3 - 24 * x^2 + 50 * x - 35) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 1052 :=
by
  have h_sum : p + q + r = 24 := by sorry
  have h_product : p * q + q * r + r * p = 50 := by sorry
  sorry

end roots_sum_of_squares_l133_133200


namespace base_three_to_base_ten_l133_133239

theorem base_three_to_base_ten : 
  let n := 1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 1 * 3^1 + 2 * 3^0
  in n = 140 :=
by
  let n := 1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 1 * 3^1 + 2 * 3^0
  show n = 140
  sorry

end base_three_to_base_ten_l133_133239


namespace inverse_square_variation_l133_133737

theorem inverse_square_variation (k : ℝ) (y x : ℝ) (h1: x = k / y^2) (h2: 0.25 = k / 36) : 
  x = 1 :=
by
  -- Here, you would provide further Lean code to complete the proof
  -- using the given hypothesis h1 and h2, along with some computation.
  sorry

end inverse_square_variation_l133_133737


namespace monotonic_intervals_0_lt_a_lt_1_monotonic_intervals_a_eq_1_monotonic_intervals_1_lt_a_minimum_value_a_neg_1_l133_133884

open Real

def f (a x : ℝ) : ℝ := a * log x + (x^2 / 2) - (a + 1) * x

theorem monotonic_intervals_0_lt_a_lt_1 {a x : ℝ} (h_a : 0 < a) (h_a_le_1 : a < 1) :
    (∀ {x}, 0 < x ∧ x < a → (f a x)' > 0) ∧ 
    (∀ {x}, a < x ∧ x < 1 → (f a x)' < 0) ∧ 
    (∀ {x}, 1 < x → (f a x)' > 0) := sorry

theorem monotonic_intervals_a_eq_1 {x : ℝ} :
    0 < x → (f 1 x)' ≥ 0 := sorry

theorem monotonic_intervals_1_lt_a {a x : ℝ} (h_a : 1 < a) :
    (∀ {x}, 0 < x ∧ x < 1 → (f a x)' > 0) ∧ 
    (∀ {x}, 1 < x ∧ x < a → (f a x)' < 0) ∧ 
    (∀ {x}, a < x → (f a x)' > 0) := sorry

theorem minimum_value_a_neg_1 :
    (∀ {x}, 0 < x → f (-1) x ≥ 1/2) := sorry


end monotonic_intervals_0_lt_a_lt_1_monotonic_intervals_a_eq_1_monotonic_intervals_1_lt_a_minimum_value_a_neg_1_l133_133884


namespace min_length_AB_l133_133893

noncomputable def C1_Equation : (ℝ × ℝ) → Prop :=
  λ p, (p.1 ^ 2 + (p.2 - 1) ^ 2 = 1)

noncomputable def C2_Equation : (ℝ × ℝ) → Prop :=
  λ p, (p.1 - 2 * p.2 - 3 = 0)

theorem min_length_AB :
  let d := λ p1 p2, real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)
  in ∃ A B : ℝ × ℝ, C1_Equation A ∧ C2_Equation B ∧ d A B = real.sqrt 5 - 1 :=
sorry

end min_length_AB_l133_133893


namespace right_angle_vertex_polar_coordinates_l133_133497

theorem right_angle_vertex_polar_coordinates
  {A B : ℝ × ℝ} 
  (hA : A = (2, π / 4))
  (hB : B = (2, 5 * π / 4))
  (isosceles_right : ∃ C : ℝ × ℝ, is_isosceles_right_triangle A B C) :
  (∃ C : ℝ × ℝ, C = (2, 3 * π / 4) ∨ C = (2, 7 * π / 4)) :=
sorry

def is_isosceles_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let (a₁, a₂) := A in
  let (b₁, b₂) := B in
  let (c₁, c₂) := C in
  ((c₁ - a₁) ^ 2 + (c₂ - a₂) ^ 2 = (c₁ - b₁) ^ 2 + (c₂ - b₂) ^ 2) ∧ 
  ((c₁ - a₁) * (c₁ - b₁) + (c₂ - a₂) * (c₂ - b₂) = 0)

end right_angle_vertex_polar_coordinates_l133_133497


namespace slope_of_line_intersecting_two_lines_l133_133467

theorem slope_of_line_intersecting_two_lines (k : ℝ) :
  (∃ A B : ℝ × ℝ, 
    A.2 = 1 ∧ 
    (A.2 + 1) = k * (A.1 - 1) ∧ 
    B.1 - B.2 = 7 ∧ 
    (B.2 + 1) = k * (B.1 - 1) ∧ 
    (A.1 + B.1) / 2 = 1 ∧ 
    (A.2 + B.2 + 1) / 2 = -1) → 
  k = -2/3 :=
begin
  sorry
end

end slope_of_line_intersecting_two_lines_l133_133467


namespace number_of_even_positive_factors_l133_133462

theorem number_of_even_positive_factors : 
  let n := 2^4 * 3^3 * 7 in
  -- Define a function that computes the number of even factors of n
  (∑ (a : Fin 5) in {1, 2, 3, 4}.toFinset, ∑ (b : Fin 4) in Finset.range 4, ∑ (c : Fin 2) in Finset.range 2, 1).val = 32 := 
by 
  sorry

end number_of_even_positive_factors_l133_133462


namespace find_a_perpendicular_lines_l133_133152

theorem find_a_perpendicular_lines 
  (a : ℤ)
  (l1 : ∀ x y : ℤ, a * x + 4 * y + 7 = 0)
  (l2 : ∀ x y : ℤ, 2 * x - 3 * y - 1 = 0) : 
  (∃ a : ℤ, a = 6) :=
by sorry

end find_a_perpendicular_lines_l133_133152


namespace sum_of_solutions_is_zero_l133_133548

theorem sum_of_solutions_is_zero :
  ∀ (x y : ℝ), y = 9 → x^2 + y^2 = 169 → x = sqrt 88 ∨ x = -sqrt 88 → (sqrt 88 + -sqrt 88) = 0 :=
by sorry

end sum_of_solutions_is_zero_l133_133548


namespace log_base_5_of_cube_root_25_eq_l133_133820

noncomputable def log_base_5_of_cube_root_25 : Real :=
  Real.log 5 (Real.cbrt (25 : Real))

theorem log_base_5_of_cube_root_25_eq : log_base_5_of_cube_root_25 = 2 / 3 := by 
  sorry

end log_base_5_of_cube_root_25_eq_l133_133820


namespace g_n_formula_l133_133232

-- Definitions derived from the conditions
def f (x : ℝ) : ℝ := Real.log (1 + x)
def f_prime (x : ℝ) : ℝ := 1 / (1 + x)
def g (x : ℝ) : ℝ := x * f_prime x

def g_n : ℕ+ → ℝ → ℝ
| ⟨1, _⟩, x => g x
| ⟨n + 1, hn⟩, x => g (g_n ⟨n, Nat.succ_pos' n⟩ x)

-- Induction hypothesis and proof goal
theorem g_n_formula (n : ℕ+) (x : ℝ) (hx : 0 ≤ x) : g_n n x = x / (1 + (n : ℕ) * x) :=
by
  sorry

end g_n_formula_l133_133232


namespace find_m_l133_133031

variables (a b : ℝ) (x y m : ℝ) (A B F1 F2 : ℝ × ℝ)

-- Given conditions
def ellipse := (x^2 / 3) + y^2 = 1
def line := y = x + m
def foci_F1 := F1 = (-sqrt 2, 0)
def foci_F2 := F2 = (sqrt 2, 0)
def area_F1_AB := λ A B : ℝ × ℝ, 2 * |det(F1, A, B)| = |det(F2, A, B)|

-- The proof problem statement
theorem find_m : 
  ellipse ∧ 
  line ∧ 
  foci_F1 ∧ 
  foci_F2 ∧ 
  (area_F1_AB A B) →
  m = -sqrt(2) / 3 := sorry

end find_m_l133_133031


namespace exponential_inequality_solution_l133_133831

theorem exponential_inequality_solution (x : ℝ) : 3^x < 1/27 ↔ x < -3 := sorry

end exponential_inequality_solution_l133_133831


namespace solve_problem_l133_133516

theorem solve_problem :
  ∃ a b c d e f : ℤ,
  (208208 = 8^5 * a + 8^4 * b + 8^3 * c + 8^2 * d + 8 * e + f) ∧
  (0 ≤ a ∧ a ≤ 7) ∧ (0 ≤ b ∧ b ≤ 7) ∧ (0 ≤ c ∧ c ≤ 7) ∧
  (0 ≤ d ∧ d ≤ 7) ∧ (0 ≤ e ∧ e ≤ 7) ∧ (0 ≤ f ∧ f ≤ 7) ∧
  (a * b * c + d * e * f = 72) :=
by
  sorry

end solve_problem_l133_133516


namespace distance_from_point_B_to_line_AC_l133_133947

noncomputable def distance_from_point_to_line (a b : ℝ×ℝ×ℝ) : ℝ :=
let ax := a.1 in let ay := a.2 in let az := a.3 in
let bx := b.1 in let by := b.2 in let bz := b.3 in
let u := (bx, by, bz) in
let a := (ax, ay, az) in
let magnitude_u := Real.sqrt (bx^2 + by^2 + bz^2) in
let unit_u := (bx / magnitude_u, by / magnitude_u, bz / magnitude_u) in
let a_dot_u := (ax * unit_u.1 + ay * unit_u.2 + az * unit_u.3) in
let a_square := (ax^2 + ay^2 + az^2) in
let distance := Real.sqrt (a_square - a_dot_u^2) in
distance

theorem distance_from_point_B_to_line_AC :
  distance_from_point_to_line (1, 1, 2) (2, 1, 1) = Real.sqrt 66 / 6 :=
by sorry

end distance_from_point_B_to_line_AC_l133_133947


namespace triangle_ratio_proof_l133_133392

noncomputable def triangle_ratio (a b c : ℝ) := (a + b)^2 + (b + c)^2 + (c + a)^2 / (2 * (a + b + c)^2)


theorem triangle_ratio_proof 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (S_triangle_ABC S_PQRSTU : ℝ) 
  (h4 : S_PQRSTU / S_triangle_ABC = ((a + b)^2 + (b + c)^2 + (c + a)^2) / (2 * (a + b + c)^2)) 
  : triangle_ratio a b c = (h4) / (S_triangle_ABC) :=
by
  sorry

end triangle_ratio_proof_l133_133392


namespace hexagon_points_fourth_layer_l133_133785

theorem hexagon_points_fourth_layer :
  ∃ (h : ℕ → ℕ), h 1 = 1 ∧ (∀ n ≥ 2, h n = h (n - 1) + 6 * (n - 1)) ∧ h 4 = 37 :=
by
  sorry

end hexagon_points_fourth_layer_l133_133785


namespace odd_product_probability_range_l133_133663

theorem odd_product_probability_range :
  ∃ p : ℝ, ∀ n : ℕ, (n = 1000) →
    let k := n / 2 in
    let p1 := k / n in
    let p2 := (k - 1) / (n - 1) in
    p = p1 * p2 ∧ (1 / 6 : ℝ) < p ∧ p < (1 / 3 : ℝ) :=
  sorry

end odd_product_probability_range_l133_133663


namespace jane_total_drawing_paper_l133_133954

theorem jane_total_drawing_paper (brown_sheets : ℕ) (yellow_sheets : ℕ) 
    (h1 : brown_sheets = 28) (h2 : yellow_sheets = 27) : 
    brown_sheets + yellow_sheets = 55 := 
by
    sorry

end jane_total_drawing_paper_l133_133954


namespace bank_deposit_l133_133366

theorem bank_deposit (income : ℝ) (provident_fund_portion insurance_premium_portion domestic_needs_portion : ℝ)
  (h_income : income = 200)
  (h_provident : provident_fund_portion = 1 / 16)
  (h_insurance : insurance_premium_portion = 1 / 15)
  (h_domestic : domestic_needs_portion = 5 / 7) :
  let provident_fund := provident_fund_portion * income in
  let remaining_after_provident := income - provident_fund in
  let insurance_premium := insurance_premium_portion * remaining_after_provident in
  let remaining_after_insurance := remaining_after_provident - insurance_premium in
  let domestic_spending := domestic_needs_portion * remaining_after_insurance in
  let remaining_income := remaining_after_insurance - domestic_spending in
  remaining_income = 50 :=
by
  sorry

end bank_deposit_l133_133366


namespace quadratic_function_properties_l133_133872

-- We define the primary conditions
def axis_of_symmetry (f : ℝ → ℝ) (x_sym : ℝ) : Prop := 
  ∀ x, f x = f (2 * x_sym - x)

def minimum_value (f : ℝ → ℝ) (y_min : ℝ) (x_min : ℝ) : Prop := 
  ∀ x, f x_min ≤ f x

def passes_through (f : ℝ → ℝ) (pt : ℝ × ℝ) : Prop := 
  f pt.1 = pt.2

-- We need to prove that a quadratic function exists with the given properties and find intersections
theorem quadratic_function_properties :
  ∃ f : ℝ → ℝ,
    axis_of_symmetry f (-1) ∧
    minimum_value f (-4) (-1) ∧
    passes_through f (-2, 5) ∧
    (∀ y : ℝ, f 0 = y → y = 5) ∧
    (∀ x : ℝ, f x = 0 → (x = -5/3 ∨ x = -1/3)) :=
sorry

end quadratic_function_properties_l133_133872


namespace minimal_period_36_l133_133585

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then real.sqrt 2
  else if n = 1 then 2
  else sequence (n-1) * (sequence (n-2))^2

-- Minimal period p is given by: 
def minimal_period (p : ℕ) : Prop :=
  ∀ m ≥ p, ∃ k : ℕ, sequence (m + k) % 2014 = sequence k % 2014

theorem minimal_period_36 : minimal_period 36 :=
sorry

end minimal_period_36_l133_133585


namespace triangle_area_result_l133_133429

/-- Given conditions -/

variables (A B C D E F G : Point)
variables (l1 l2 : Line)
variables (r : ℝ)
variables {RA : ℝ} {AB AC AE AD : ℝ}

def is_equilateral (A B C : Point) : Prop := 
  ∀ a b c, a = dist A B ∧ b = dist A C ∧ c = dist B C ∧ a = b ∧ b = c

def on_circle (A B C : Point) (r : ℝ) : Prop :=
  ∀ a b c, dist A O = r ∧ dist B O = r ∧ dist C O = r

def on_line (A B D : Point) : Prop :=
  ∃ l, A ∈ l ∧ B ∈ l ∧ D ∈ l

def distances : Prop := 
  A = 0 ∧ B = dist A B ∧ C = dist A C ∧ D = dist A ()
  D = 15 ∧ E = 14 

def angles : Prop :=
  ∀ a b c, a = 60 ∧ b = 60 ∧ c = 60  

def parallel_lines (l1 l2 : Line) (α β : ang) : Prop :=
  ∀ β,  (l1 ∥ AE . CR l2 //AD ) ♡ ∧ β =60

def intersection (D E : Point) : Prop :=
  ∃ F, ∃ l1 ∥ l2, l1 ∩ l2 = F ∧ 
  ∃ G, F = collinear A ∧ intersection (G, circle)


/-- The required proof -/

theorem triangle_area_result :
  ∃ (p q r : ℕ), p + q + r = 6154 := 
begin

  let ABC : Triangle := sides (3√3) dist (.ABC)

  O,r=>3 ∧ had 150

  have AF : l1 ∥ l2 => .F
 
  let =  help := by AE∥AD

  have parallel_lines:by simplified 

  let EAF : ⅍ 250√           /

 sorry

end

end triangle_area_result_l133_133429


namespace tank_capacity_l133_133335

variable (C : ℝ)

def leak_rate (C : ℝ) : ℝ := C / 3   -- litres per hour
def inlet_rate : ℝ := 6 * 60         -- litres per hour
def net_emptying_rate (C : ℝ) : ℝ := C / 12  -- litres per hour

theorem tank_capacity : leak_rate C - inlet_rate = net_emptying_rate C → C = 864 := 
by
  sorry

end tank_capacity_l133_133335


namespace min_value_reciprocal_sum_l133_133482

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : log 4 (1/a + 1/b) = log 2 (sqrt (1/(a * b)))) : (1/a + 1/b) = 4 := 
sorry

end min_value_reciprocal_sum_l133_133482


namespace area_triangle_OKT_l133_133358

-- Conditions
variables (A B C O M N K T : Type) [circle ω] [triangle ΔABC]
variable (a : ℝ)
variables [tangent_to_sides BA BC ω M N]
variables [parallel_through_point M BC intersect_ray BO K]
variables [point_on_ray MN T]
variable (angle_condition : ∠ MTK = 1/2 * ∠ ABC)
variable [tangent_line KT ω]

-- Theorem Statement
theorem area_triangle_OKT (BM : a = |BM|):
  area (triangle O K T) = (a^2) / 2 :=
sorry

end area_triangle_OKT_l133_133358


namespace arithmetic_sequence_75th_term_diff_l133_133779

theorem arithmetic_sequence_75th_term_diff :
  ∃ (L G : ℚ), 
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 150 → 20 ≤ 80 + (k - 75) * d ∧ 80 + (k - 75) * d ≤ 120) ∧
    (sum (λ k, 80 + (k - 75) * d) (finset.range 150) = 12000) ∧
    (80 - 75 * (40 / 149) = L) ∧
    (80 + 75 * (40 / 149) = G) ∧
    (G - L = 6000 / 149) :=
sorry

end arithmetic_sequence_75th_term_diff_l133_133779


namespace liquid_mixture_ratio_l133_133126

theorem liquid_mixture_ratio (m1 m2 m3 : ℝ) (ρ1 ρ2 ρ3 : ℝ) (k : ℝ)
  (hρ1 : ρ1 = 6 * k) (hρ2 : ρ2 = 3 * k) (hρ3 : ρ3 = 2 * k)
  (h_condition : m1 ≥ 3.5 * m2)
  (h_arith_mean : (m1 + m2 + m3) / (m1 / ρ1 + m2 / ρ2 + m3 / ρ3) = (ρ1 + ρ2 + ρ3) / 3) :
    ∃ x y : ℝ, x ≤ 2/7 ∧ (4 * x + 15 * y = 7) := sorry

end liquid_mixture_ratio_l133_133126


namespace angle_at_shared_vertex_l133_133780

noncomputable def vertexAngle (n : ℕ) : ℝ := (n - 2) * 180 / n

theorem angle_at_shared_vertex 
  (A B C D E F G : Point)
  (p : Pentagon A D E F G)
  (t : EquilateralTriangle A B C)
  (shared_on_circle : Inscribed t p)
  (shared_vertex : ∃ A : Point, Vertex A t ∧ Vertex A p) :
  ∠ BAC + ∠ BAG / 2 = 114 := 
sorry

end angle_at_shared_vertex_l133_133780


namespace find_m_for_area_of_triangles_l133_133059

noncomputable def ellipse_foci := (-real.sqrt 2, 0, real.sqrt 2, 0)

theorem find_m_for_area_of_triangles : 
  (∃ (m : ℝ), 
    let F1 := (-real.sqrt 2, 0) in
    let F2 := (real.sqrt 2, 0) in
    let A B : ℝ × ℝ in
    let ellipse : ∀ x y : ℝ, (x^2 / 3) + y^2 = 1 := sorry in
    let line := λ x y : ℝ, y = x + m in
    line_intersects_ellipse : ∀ A B : ℝ × ℝ, disjoint ((A, B) ∪ ellipse) := sorry in
    (area_of_triangle(F1, A, B) = 2 * area_of_triangle(F2, A, B)) → m = -real.sqrt 2 / 3) :=
sorry

end find_m_for_area_of_triangles_l133_133059


namespace monotonic_intervals_max_min_values_l133_133881

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) * Real.sin x

theorem monotonic_intervals :
  (∀ k ∈ ℤ, (∀ x, 2 * k * π - 3 * π / 4 < x ∧ x < 2 * k * π + π / 4 -> f' x > 0)) ∧
  (∀ k ∈ ℤ, (∀ x, 2 * k * π + π / 4 < x ∧ x < 2 * k * π + 5 * π / 4 -> f' x < 0)) :=
sorry

theorem max_min_values :
  let x₁ := -π
  let x₂ := -3 * π / 4
  let x₃ := π / 4
  let x₄ := π
  f x₃ = (Real.sqrt 2 / 2) * Real.exp (-π / 4) ∧
  f x₂ = -(Real.sqrt 2 / 2) * Real.exp (3 * π / 4) ∧
  f x₁ = 0 ∧
  f x₄ = 0 :=
sorry

end monotonic_intervals_max_min_values_l133_133881


namespace find_m_for_area_of_triangles_l133_133057

noncomputable def ellipse_foci := (-real.sqrt 2, 0, real.sqrt 2, 0)

theorem find_m_for_area_of_triangles : 
  (∃ (m : ℝ), 
    let F1 := (-real.sqrt 2, 0) in
    let F2 := (real.sqrt 2, 0) in
    let A B : ℝ × ℝ in
    let ellipse : ∀ x y : ℝ, (x^2 / 3) + y^2 = 1 := sorry in
    let line := λ x y : ℝ, y = x + m in
    line_intersects_ellipse : ∀ A B : ℝ × ℝ, disjoint ((A, B) ∪ ellipse) := sorry in
    (area_of_triangle(F1, A, B) = 2 * area_of_triangle(F2, A, B)) → m = -real.sqrt 2 / 3) :=
sorry

end find_m_for_area_of_triangles_l133_133057


namespace number_of_valid_propositions_l133_133854

-- Definitions for planes and lines, with notations for perpendicular (⊥) and parallel (||) relationships
variable {α β : Type*} [plane α] [plane β]
variable {l : Type*} [line l]

axiom perp (l : Type*) (p : plane α) : Prop
axiom parallel (l : Type*) (p : plane β) : Prop
axiom plane_perp (p1 p2 : plane α) : Prop
axiom not_contained (l : Type*) (p : plane α) : Prop

-- Conditions
def l_perp_α := perp l α
def l_parallel_β := parallel l β
def α_perp_β := plane_perp α β
def l_not_in_α := not_contained l α
def l_not_in_β := not_contained l β

theorem number_of_valid_propositions :
  (l_perp_α ∧ l_parallel_β ∧ (α_perp_β → False) → False) ∧
  (l_perp_α ∧ α_perp_β ∧ (l_parallel_β → False) → False) ∧
  (l_parallel_β ∧ α_perp_β ∧ (l_perp_α → False) → True) →
  2 = ∑ (λ x, if x then 1 else 0) [
    (l_perp_α ∧ l_parallel_β → α_perp_β),
    (l_perp_α ∧ α_perp_β → l_parallel_β),
    (l_parallel_β ∧ α_perp_β → l_perp_α)] := 
sorry

end number_of_valid_propositions_l133_133854


namespace intersection_is_singleton_zero_l133_133502

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {-2, 0}

-- Define the theorem to be proved
theorem intersection_is_singleton_zero : M ∩ N = {0} :=
by
  -- Proof is provided by the steps above but not needed here
  sorry

end intersection_is_singleton_zero_l133_133502


namespace coltons_share_ratio_l133_133802

-- Definitions based on conditions
def burger_length_in_inches : ℕ := 12
def coltons_share_in_inches : ℕ := 6

-- Theorem statement
theorem coltons_share_ratio (h : coltons_share_in_inches > 0) : coltons_share_in_inches / burger_length_in_inches = 1 / 2 :=
by
  have h1 : burger_length_in_inches = 12 := rfl
  have h2 : coltons_share_in_inches = 6 := rfl
  have h3 : coltons_share_in_inches / burger_length_in_inches = 1 / 2 := by
    rw [h1, h2]
    norm_num
  exact h3

end coltons_share_ratio_l133_133802


namespace find_t_l133_133458

-- Conditions
def a : ℝ × ℝ := (-1, 3)
def b (t : ℝ) : ℝ × ℝ := (1, t)
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Theorem
theorem find_t (t : ℝ) (h : perpendicular (a.1 - 2 * b t.1, a.2 - 2 * b t.2) a) : t = 2 :=
  sorry

end find_t_l133_133458


namespace remainder_of_x_squared_div_20_l133_133511

theorem remainder_of_x_squared_div_20
  (x : ℤ)
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 4 * x ≡ 12 [ZMOD 20]) :
  (x * x) % 20 = 4 :=
sorry

end remainder_of_x_squared_div_20_l133_133511


namespace cot_arccot_identity_cot_addition_formula_problem_solution_l133_133842

noncomputable def cot (x : ℝ) : ℝ := 1 / tan x
noncomputable def arccot (x : ℝ) : ℝ := arctan (1 / x)

theorem cot_arccot_identity (a : ℝ) : cot (arccot a) = a := sorry

theorem cot_addition_formula (a b : ℝ) : 
  cot (arccot a + arccot b) = (a * b - 1) / (a + b) := sorry

theorem problem_solution : 
  cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 971 / 400 := sorry

end cot_arccot_identity_cot_addition_formula_problem_solution_l133_133842


namespace measles_cases_1995_l133_133525

-- Definitions based on the conditions
def initial_cases_1970 : ℕ := 300000
def final_cases_2000 : ℕ := 200
def cases_1990 : ℕ := 1000
def decrease_rate : ℕ := 14950 -- Annual linear decrease from 1970-1990
def a : ℤ := -8 -- Coefficient for the quadratic phase

-- Function modeling the number of cases in the quadratic phase (1990-2000)
def measles_cases (x : ℕ) : ℤ := a * (x - 1990)^2 + cases_1990

-- The statement we want to prove
theorem measles_cases_1995 : measles_cases 1995 = 800 := by
  sorry

end measles_cases_1995_l133_133525


namespace proof_l133_133998

-- Definition of the logical statements
def all_essays_correct (maria : Type) : Prop := sorry
def passed_course (maria : Type) : Prop := sorry

-- Condition provided in the problem
axiom condition : ∀ (maria : Type), all_essays_correct maria → passed_course maria

-- We need to prove this
theorem proof (maria : Type) : ¬ (passed_course maria) → ¬ (all_essays_correct maria) :=
by sorry

end proof_l133_133998


namespace number_of_siblings_l133_133817

-- Definitions for the given conditions
def total_height : ℕ := 330
def sibling1_height : ℕ := 66
def sibling2_height : ℕ := 66
def sibling3_height : ℕ := 60
def last_sibling_height : ℕ := 70  -- Derived from the solution steps
def eliza_height : ℕ := last_sibling_height - 2

-- The final question to validate
theorem number_of_siblings (h : 2 * sibling1_height + sibling3_height + last_sibling_height + eliza_height = total_height) :
  4 = 4 :=
by {
  -- Condition h states that the total height is satisfied
  -- Therefore, it directly justifies our claim without further computation here.
  sorry
}

end number_of_siblings_l133_133817


namespace resultant_profit_percentage_l133_133765

theorem resultant_profit_percentage (p : ℝ) (h1 : p > 0) :
  let pa := p * (1 + 0.3) in
  let pb := pa * (1 - 0.2) in
  ((pb - p) / p) * 100 = 4 :=
by
  sorry

end resultant_profit_percentage_l133_133765


namespace find_digits_for_divisibility_l133_133116

theorem find_digits_for_divisibility (d1 d2 : ℕ) (h1 : d1 < 10) (h2 : d2 < 10) :
  (32 * 10^7 + d1 * 10^6 + 35717 * 10 + d2) % 72 = 0 →
  d1 = 2 ∧ d2 = 6 :=
by
  sorry

end find_digits_for_divisibility_l133_133116


namespace distance_P_to_y_axis_l133_133476

-- Define the Point structure
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Condition: Point P with coordinates (-3, 5)
def P : Point := ⟨-3, 5⟩

-- Definition of distance from a point to the y-axis
def distance_to_y_axis (p : Point) : ℝ :=
  abs p.x

-- Proof problem statement
theorem distance_P_to_y_axis : distance_to_y_axis P = 3 := 
  sorry

end distance_P_to_y_axis_l133_133476


namespace simplify_expression_l133_133614

variable (b : ℤ)

theorem simplify_expression :
  (3 * b + 6 - 6 * b) / 3 = -b + 2 :=
sorry

end simplify_expression_l133_133614


namespace josanna_next_test_score_l133_133957

theorem josanna_next_test_score :
  let scores := [75, 85, 65, 95, 70]
  let current_sum := scores.sum
  let current_average := current_sum / scores.length
  let desired_average := current_average + 10
  let new_test_count := scores.length + 1
  let desired_sum := desired_average * new_test_count
  let required_score := desired_sum - current_sum
  required_score = 138 :=
by
  sorry

end josanna_next_test_score_l133_133957


namespace quadratic_real_roots_range_l133_133470

theorem quadratic_real_roots_range (m : ℝ) : (∃ x y : ℝ, x ≠ y ∧ mx^2 + 2*x + 1 = 0 ∧ yx^2 + 2*y + 1 = 0) → m ≤ 1 ∧ m ≠ 0 :=
by 
sorry

end quadratic_real_roots_range_l133_133470


namespace cannot_form_optionE_l133_133330

-- Define the 4x4 tile
structure Tile4x4 :=
(matrix : Fin 4 → Fin 4 → Bool) -- Boolean to represent black or white

-- Define the condition of alternating rows and columns
def alternating_pattern (tile : Tile4x4) : Prop :=
  (∀ i, tile.matrix i 0 ≠ tile.matrix i 1 ∧
         tile.matrix i 2 ≠ tile.matrix i 3) ∧
  (∀ j, tile.matrix 0 j ≠ tile.matrix 1 j ∧
         tile.matrix 2 j ≠ tile.matrix 3 j)

-- Example tiles for options A, B, C, D, E
def optionA : Tile4x4 := sorry
def optionB : Tile4x4 := sorry
def optionC : Tile4x4 := sorry
def optionD : Tile4x4 := sorry
def optionE : Tile4x4 := sorry

-- Given pieces that can form a 4x4 alternating tile
axiom given_piece1 : Tile4x4
axiom given_piece2 : Tile4x4

-- Combining given pieces to form a 4x4 tile
def combine_pieces (p1 p2 : Tile4x4) : Tile4x4 := sorry -- Combination logic here

-- Proposition stating the problem
theorem cannot_form_optionE :
  (∀ tile, tile = optionA ∨ tile = optionB ∨ tile = optionC ∨ tile = optionD ∨ tile = optionE →
    (tile = optionA ∨ tile = optionB ∨ tile = optionC ∨ tile = optionD → alternating_pattern tile) ∧
    tile = optionE → ¬alternating_pattern tile) :=
sorry

end cannot_form_optionE_l133_133330


namespace point_I_l133_133979

def incenter (A B C I : Point) : Prop :=
  ∃ r, (ball I r).circumscribes ⟨A, B, C⟩

def interior_point (P : Point) (ABC : Triangle) : Prop :=
  ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ P = x • A + y • B + z • C

def angle_eq_condition (A B C P : Point) : Prop :=
  ∠ P B A + ∠ P C A = ∠ P B C + ∠ P C B

theorem point_I (A B C I P : Point) (h_incenter : incenter A B C I) (h_interior_point : interior_point P ⟨A, B, C⟩) (h_angle_condition : angle_eq_condition A B C P) :
  dist A P ≥ dist A I ∧ (dist A P = dist A I ↔ P = I) := 
sorry

end point_I_l133_133979


namespace a_n_general_formula_b_n_general_formula_exist_mn_arithmetic_seq_l133_133100

-- Define the arithmetic sequence a_n with common difference
def arithmetic_sequence (a_1 d: ℕ) (n: ℕ) : ℕ := a_1 + (n - 1) * d

-- Define the condition a_2 * a_3 = 15
def condition_a2_a3 (a_1 d: ℕ) : Prop := (arithmetic_sequence a_1 d 2) * (arithmetic_sequence a_1 d 3) = 15

-- Define the sum of the first 4 terms S_4 = 16
def sum_first4 (a_1 d: ℕ) : Prop := (arithmetic_sequence a_1 d 1) + (arithmetic_sequence a_1 d 2) + (arithmetic_sequence a_1 d 3) + (arithmetic_sequence a_1 d 4) = 16

-- Define the sequence a_n
def a_n (n: ℕ) : ℕ := 2 * n - 1

-- Define the conditions for sequence b_n
def b_sequence_condition (a_1 b_1: ℕ) (n: ℕ) : ℕ := b_1 + (1/((a_n (n )) * (a_n (n+1))))

-- Prove a_n = 2n - 1
theorem a_n_general_formula : ∀n: ℕ, n > 0 →
  ∃a_1 d: ℕ, d > 0 ∧ condition_a2_a3 a_1 d ∧ sum_first4 a_1 d ∧ a_n n = 2 * n - 1 := sorry

-- Prove b_n = (3n-2)/(2n-1)
theorem b_n_general_formula : ∀n: ℕ, n > 0 →
  ∃b_1, b_1 = a_n 1 ∧ b_sequence_condition (a_n 1) b_1 n ∧ b_n n = (3 * n - 2) / (2 * n - 1) := sorry

-- Prove the existence of m and n
theorem exist_mn_arithmetic_seq : ∃m n: ℕ, m > 0 ∧ n > 0 ∧ m ≠ n ∧ 
  let b2 := (4 / 3), bm := (3 / 2 - 1 / (4 * m - 2)), bn := (3 / 2 - 1 / (4 * n - 2)) in
  b2 + bn = 2 * bm := sorry

end a_n_general_formula_b_n_general_formula_exist_mn_arithmetic_seq_l133_133100


namespace product_lcm_gcd_12_15_l133_133447

theorem product_lcm_gcd_12_15 : 
  let a := 12
  let b := 15
  let gcd := Nat.gcd a b
  let lcm := Nat.lcm a b
  in gcd * lcm = 180 :=
by 
  let a := 12
  let b := 15
  let gcd := Nat.gcd a b
  let lcm := Nat.lcm a b
  show gcd * lcm = 180, from sorry

end product_lcm_gcd_12_15_l133_133447


namespace incorrect_option_B_l133_133540

-- Define the conditions as given in the problem
variables (x : ℝ) (distance_A distance_B : ℝ)
def speed_A : ℝ := 7
def speed_B : ℝ := 6.5
def initial_head_start : ℝ := 5

-- Define the total distance run by each person
def distance_A := speed_A * x
def distance_B := initial_head_start + speed_B * x

-- Prove that the equation corresponding to option B is incorrect
theorem incorrect_option_B : ¬(7 * x + 5 = 6.5 * x) :=
by 
  intro h,
  have h1 : 7 * x = 6.5 * x + 5 := 
    calc
      7 * x = 6.5 * x + 5 : sorry, -- derived from condition
  linarith [h, h1]


end incorrect_option_B_l133_133540


namespace find_m_l133_133040

def ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + y^2 = 1

def focus1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def focus2 : ℝ × ℝ := (Real.sqrt 2, 0)

def line (x y m : ℝ) : Prop :=
  y = x + m

def intersects (m : ℝ) (x y : ℝ) : Prop :=
  ellipse x y ∧ line x y m

def area_relation (m : ℝ) : Prop :=
  let x := -m -- This might be simplified as it's used to calculate distances
  let dist1 := Real.abs (focus1.fst + m)
  let dist2 := Real.abs (focus2.fst + m)
  dist1 = 2 * dist2

theorem find_m :
  area_relation (-Real.sqrt 2 / 3) :=
by
  sorry

end find_m_l133_133040


namespace valid_probabilities_and_invalid_probability_l133_133268

theorem valid_probabilities_and_invalid_probability :
  (let first_box_1 := (4, 7)
       second_box_1 := (3, 5)
       combined_prob_1 := (first_box_1.1 + second_box_1.1) / (first_box_1.2 + second_box_1.2),
       first_box_2 := (8, 14)
       second_box_2 := (3, 5)
       combined_prob_2 := (first_box_2.1 + second_box_2.1) / (first_box_2.2 + second_box_2.2),
       prob_1 := first_box_1.1 / first_box_1.2,
       prob_2 := second_box_2.1 / second_box_2.2
     in (combined_prob_1 = 7 / 12 ∧ combined_prob_2 = 11 / 19) ∧ (19 / 35 < prob_1 ∧ prob_1 < prob_2) → False) :=
by
  sorry

end valid_probabilities_and_invalid_probability_l133_133268


namespace rth_term_l133_133452

-- Given arithmetic progression sum formula
def Sn (n : ℕ) : ℕ := 3 * n^2 + 4 * n + 5

-- Prove that the r-th term of the sequence is 6r + 1
theorem rth_term (r : ℕ) : (Sn r) - (Sn (r - 1)) = 6 * r + 1 :=
by
  sorry

end rth_term_l133_133452


namespace even_product_probability_l133_133310

theorem even_product_probability :
  let S := {1, 2, 3, 4, 5}
  let total_ways := Nat.choose 5 2
  let odd_ways := Nat.choose 3 2
  let even_product_ways := total_ways - odd_ways
  (even_product_ways : ℚ) / total_ways = 7 / 10 :=
by
  let S := {1, 2, 3, 4, 5}
  let total_ways := Nat.choose 5 2
  let odd_ways := Nat.choose 3 2
  let even_product_ways := total_ways - odd_ways
  have h1 : (even_product_ways : ℚ) = 7 := sorry
  have h2 : (total_ways : ℚ) = 10 := sorry
  rw [h1, h2]
  norm_num
  done

end even_product_probability_l133_133310


namespace find_m_l133_133043

def ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + y^2 = 1

def focus1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def focus2 : ℝ × ℝ := (Real.sqrt 2, 0)

def line (x y m : ℝ) : Prop :=
  y = x + m

def intersects (m : ℝ) (x y : ℝ) : Prop :=
  ellipse x y ∧ line x y m

def area_relation (m : ℝ) : Prop :=
  let x := -m -- This might be simplified as it's used to calculate distances
  let dist1 := Real.abs (focus1.fst + m)
  let dist2 := Real.abs (focus2.fst + m)
  dist1 = 2 * dist2

theorem find_m :
  area_relation (-Real.sqrt 2 / 3) :=
by
  sorry

end find_m_l133_133043


namespace geometry_problem_l133_133174

-- Define a triangle ABC, and the incenter P.
variables (A B C P D E : Type) [Triangle A B C]

-- BP is extended to intersect the circumcircle at D
variable (circumcircle_intersects_at_D : Line (B, P) ∩ Circumcircle (A, B, C) = D)

-- AC is extended to intersect the circumcircle again at E
variable (circumcircle_intersects_at_E : Line (A, C) ∩ Circumcircle (A, B, C) = E)

-- Points B, P, D, E are concyclic
variable (B_P_D_E_is_concyclic : Concyclic B P D E)

-- Stating the theorem to be proven
theorem geometry_problem : segment_length D E = segment_length C E := 
by sorry

end geometry_problem_l133_133174


namespace flywheel_stops_rotating_at_8_seconds_l133_133275

-- Define the angular displacement function
def angular_displacement (t : ℝ) : ℝ := 8 * t - 0.5 * t^2

-- Define the angular velocity function as the derivative of angular displacement
def angular_velocity (t : ℝ) : ℝ := derivative angular_displacement t

theorem flywheel_stops_rotating_at_8_seconds :
  ∃ t : ℝ, angular_velocity t = 0 ∧ t = 8 :=
by 
  sorry

end flywheel_stops_rotating_at_8_seconds_l133_133275


namespace distance_from_center_to_CD_l133_133616

variables (A B C D : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

-- Define the rhombus and its properties
structure Rhombus (A B C D : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] :=
  (angle_A : ℝ)
  (projection_AB_on_AD : ℝ)
  (center_distance_to_side_CD : ℝ)

-- The condition that the angle A is 45 degrees and the projection is 12
def rhombus_with_conditions :=
  Rhombus.mk 45 12 6

-- The theorem to prove the distance from center to side CD
theorem distance_from_center_to_CD (r : Rhombus ℝ ℝ ℝ ℝ) :
  r.center_distance_to_side_CD = 6 := 
  sorry

end distance_from_center_to_CD_l133_133616


namespace functional_eq_solution_l133_133433

theorem functional_eq_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (f (x) ^ 2 + f (y)) = x * f (x) + y) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := sorry

end functional_eq_solution_l133_133433


namespace tan_product_pi_over_6_3_2_undefined_l133_133804

noncomputable def tan_pi_over_6 : ℝ := Real.tan (Real.pi / 6)
noncomputable def tan_pi_over_3 : ℝ := Real.tan (Real.pi / 3)
noncomputable def tan_pi_over_2 : ℝ := Real.tan (Real.pi / 2)

theorem tan_product_pi_over_6_3_2_undefined :
  ∃ (x y : ℝ), Real.tan (Real.pi / 6) = x ∧ Real.tan (Real.pi / 3) = y ∧ Real.tan (Real.pi / 2) = 0 :=
by
  sorry

end tan_product_pi_over_6_3_2_undefined_l133_133804


namespace fraction_product_equals_l133_133402

def frac1 := 7 / 4
def frac2 := 8 / 14
def frac3 := 9 / 6
def frac4 := 10 / 25
def frac5 := 28 / 21
def frac6 := 15 / 45
def frac7 := 32 / 16
def frac8 := 50 / 100

theorem fraction_product_equals : 
  (frac1 * frac2 * frac3 * frac4 * frac5 * frac6 * frac7 * frac8) = (4 / 5) := 
by
  sorry

end fraction_product_equals_l133_133402


namespace find_m_l133_133521

noncomputable def power_function_increasing_condition (m : ℝ) : Prop :=
  (∀ x y : ℝ, 0 < x → x < y → (m^2 - m - 1)*x^m < (m^2 - m - 1)*y^m)

theorem find_m :
  ∃ m : ℝ, (power_function_increasing_condition m) ∧ (m^2 - m - 1 = 1) ∧ (m > 0) ∧ (m = 2) :=
begin
  sorry
end

end find_m_l133_133521


namespace weather_on_tenth_day_cloudy_l133_133316

/-- 
Vasya solved problems for 10 days - at least one problem each day.
Every day (except the first), if the weather was cloudy, he solved one more problem than the previous day,
and if it was sunny, he solved one less problem.
In the first 9 days, Vasya solved 13 problems. 
Prove that the weather was cloudy on the tenth day.
-/
theorem weather_on_tenth_day_cloudy
  (problems : Fin 10 → ℕ)
  (weather : Fin 10 → Bool)
  (hpos : ∀ i, 1 ≤ problems i)
  (hinit : ∑ i in Finset.range 9, problems i = 13)
  (hweather : ∀ i, 0 < i → 
                (¬weather i → problems i = problems (i - 1) + 1) ∧ 
                (weather i → problems i = problems (i - 1) - 1)) :
  ¬weather 9 := 
sorry

end weather_on_tenth_day_cloudy_l133_133316


namespace complex_number_modulus_l133_133150

noncomputable def modulus (z : ℂ) : ℝ := complex.abs z

theorem complex_number_modulus (z : ℂ) (i : ℂ) (h : i * i = -1) (h_eq : z * i^2018 = 3 + 4*complex.I) : modulus z = 5 :=
by sorry

end complex_number_modulus_l133_133150


namespace team_OT_matches_l133_133304

variable (T x M: Nat)

-- Condition: Team C played T matches in the first week.
def team_C_matches_T : Nat := T

-- Condition: Team C played x matches in the first week.
def team_C_matches_x : Nat := x

-- Condition: Team O played M matches in the first week.
def team_O_matches_M : Nat := M

-- Condition: Team C has not played against Team A.
axiom C_not_played_A : ¬ (team_C_matches_T = team_C_matches_x)

-- Condition: Team B has not played against a specified team (interpreted).
axiom B_not_played_specified : ∀ x, ¬ (team_C_matches_x = x)

-- The proof for the number of matches played by team \(\overrightarrow{OT}\).
theorem team_OT_matches : T = 4 := 
    sorry

end team_OT_matches_l133_133304


namespace vector_magnitude_addition_l133_133869

theorem vector_magnitude_addition
  (a b : EuclideanSpace ℝ (Fin 2))
  (h1 : ‖a‖ = 3)
  (h2 : ‖b‖ = 4)
  (h3 : real.angle.vsub_sub_eq 120) :
  ‖a + b‖ = real.sqrt 13 := by
sorry

end vector_magnitude_addition_l133_133869


namespace playlist_duration_l133_133231

def run_duration : ℕ := 90
def song_3min_count : ℕ := 10
def song_4min_count : ℕ := 12
def song_6min_count : ℕ := 15
def max_songs_per_category : ℕ := 7
def favorite_band_song_duration : ℕ := 4
def favorite_band_songs_minimum : ℕ := 3

theorem playlist_duration (total_run_time : ℕ) (s3_count s4_count s6_count max_per_cat : ℕ) 
    (fav_song_duration fav_min_count : ℕ) : 
    s3_count = 10 → s4_count = 12 → s6_count = 15 → total_run_time = 90 →
    max_per_cat = 7 → fav_song_duration = 4 → fav_min_count = 3 →
    ((min s3_count max_per_cat) * 3 + (min s4_count max_per_cat) * 4 + (min s6_count max_per_cat) * 6) ≥ total_run_time
:= 
begin
    intros h1 h2 h3 h4 h5 h6 h7,
    have h_s3_time : ℕ := min s3_count max_per_cat * 3,
    have h_s4_time : ℕ := (min s4_count max_per_cat - fav_min_count) * fav_song_duration + fav_min_count * fav_song_duration,
    have h_s6_time : ℕ := min s6_count max_per_cat * 6,
    have h_total_time : ℕ := h_s3_time + h_s4_time + h_s6_time,
    calc
        h_total_time = h_s3_time + h_s4_time + h_s6_time : by sorry
               ...  = 21 + 28 + 42 : by { rw [←h1, ←h2, ←h3, ←h5, ←h6, ←h7], sorry }
               ...  = 91 : by sorry
               ...  ≥ total_run_time : by { rw h4, exact nat.le_refl 91 }
end

end playlist_duration_l133_133231


namespace probability_irrational_number_l133_133298

open Real

def cards : List ℝ := [22/7, sqrt 6, -0.5, Real.pi, 0]

theorem probability_irrational_number :
  (∃ P : ℙ, P.event_set = (λ x, x ∈ {sqrt 6, Real.pi}).to_finset) →
  P.probability (λ x, x ∈ {sqrt 6, Real.pi}.to_finset) = 2 / 5 := by
  sorry

end probability_irrational_number_l133_133298


namespace homework_prob_l133_133303

theorem homework_prob (
  students_A : ℕ := 20
  students_B : ℕ := 80
  students_C : ℕ := 50
  students_D : ℕ := 100
  forget_A : ℚ := 0.2
  forget_B : ℚ := 0.15
  forget_C : ℚ := 0.25
  forget_D : ℚ := 0.1) :
  let total_students := students_A + students_B + students_C + students_D in
  let total_forgot := (students_A * forget_A) + (students_B * forget_B) + (students_C * forget_C) + (students_D * forget_D) in
  let percentage_forgot := (total_forgot / total_students) * 100 in
  percentage_forgot = 15.6 ∧ (forget_D = 0.1) ∧ (forget_C = 0.25) := sorry

end homework_prob_l133_133303


namespace sum_to_64_is_negative_311_l133_133426

def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

def sign_changes (n : ℕ) : ℤ := 
  if ∃ k, triangular k = n then -1 else 1

def series : ℕ → ℤ
| 0 => 0
| (n + 1) => series n + sign_changes (n + 1) * (n + 1)

theorem sum_to_64_is_negative_311 : series 64 = -311 := 
  sorry

end sum_to_64_is_negative_311_l133_133426


namespace cubic_roots_expression_l133_133970

noncomputable def polynomial : Polynomial ℂ :=
  Polynomial.X^3 - 3 * Polynomial.X - 2

theorem cubic_roots_expression (α β γ : ℂ)
  (h1 : (Polynomial.X - Polynomial.C α) * 
        (Polynomial.X - Polynomial.C β) * 
        (Polynomial.X - Polynomial.C γ) = polynomial) :
  α * (β - γ)^2 + β * (γ - α)^2 + γ * (α - β)^2 = -18 :=
by
  sorry

end cubic_roots_expression_l133_133970


namespace find_AB2_plus_AC2_plus_BC2_l133_133573

-- Define the parameters and given conditions
variables {A B C G : Point}
variable (sides : Triangle)
variable (area_hypotenuse_triangle : ℝ)
variable (AB AC BC : ℝ) -- These represent the side lengths of triangle ABC

-- Assume G is the centroid of triangle ABC
def centroid (ABC : Triangle) : Point := sorry

variables (GA GB GC sq_AB sq_AC sq_BC : ℝ)

-- Conditions
axiom cond_1 : G = centroid sides
axiom cond_2 : GA^2 + GB^2 + GC^2 = 48
axiom cond_3 : area sides = 2 * 6 -- The area of the triangle with sides 3, 4, 5 is 6

-- Define the target statement to prove
theorem find_AB2_plus_AC2_plus_BC2 : AB^2 + AC^2 + BC^2 = 216 :=
-- Some steps depend on advanced theorems and geometry which will be encapsulated in 'sorry'
sorry

end find_AB2_plus_AC2_plus_BC2_l133_133573


namespace central_angle_relation_l133_133522

theorem central_angle_relation
  (R L : ℝ)
  (α : ℝ)
  (r l β : ℝ)
  (h1 : r = 0.5 * R)
  (h2 : l = 1.5 * L)
  (h3 : L = R * α)
  (h4 : l = r * β) : 
  β = 3 * α :=
by
  sorry

end central_angle_relation_l133_133522


namespace min_xyz_value_l133_133978

theorem min_xyz_value (x y z : ℝ) (h1 : x + y + z = 1) (h2 : z = 2 * y) (h3 : y ≤ (1 / 3)) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (∀ a b c : ℝ, (a + b + c = 1) → (c = 2 * b) → (b ≤ (1 / 3)) → 0 < a → 0 < b → 0 < c → (a * b * c) ≥ (x * y * z) → (a * b * c) = (8 / 243)) :=
by sorry

end min_xyz_value_l133_133978


namespace sum_of_reciprocals_of_first_2003_triangular_numbers_l133_133583

def triangular_number (n : ℕ) : ℚ :=
  n * (n + 1) / 2

def sum_reciprocals (N : ℕ) : ℚ :=
  ∑ n in Finset.range N, 1 / triangular_number (n + 1)

theorem sum_of_reciprocals_of_first_2003_triangular_numbers :
  sum_reciprocals 2003 = 2003 / 1002 :=
by
  sorry

end sum_of_reciprocals_of_first_2003_triangular_numbers_l133_133583


namespace variance_scaled_data_l133_133498

-- Define the original variance condition
def original_variance (x : list ℝ) : Prop :=
  variance x = 3

-- Define the new data set by scaling the original data
def scaled_data (x : list ℝ) : list ℝ :=
  x.map (λ xi, 2 * xi)

-- The main statement asserting the variance of the scaled data
theorem variance_scaled_data (x : list ℝ) (h : original_variance x) :
  variance (scaled_data x) = 12 :=
by
  sorry

end variance_scaled_data_l133_133498


namespace gcd_lcm_product_l133_133444

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 12) (h2 : b = 15) :
  Nat.gcd a b * Nat.lcm a b = 180 := by
  rw [h1, h2]
  have gcd_val : Nat.gcd 12 15 = 3 := by sorry
  have lcm_val : Nat.lcm 12 15 = 60 := by sorry
  rw [gcd_val, lcm_val]
  norm_num

end gcd_lcm_product_l133_133444


namespace find_P_l133_133171

variables (A B C D E P : Type)
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup E] 
variables [VectorSpace ℝ A] [VectorSpace ℝ B] [VectorSpace ℝ C] [VectorSpace ℝ D] [VectorSpace ℝ E]

def is_in_triangle (D : A) (B C : A) : Prop :=
  ∃ b c : ℝ, b + c = 1 ∧ D = b • B + c • C ∧ b/c = 2/3

def is_extended_ratio (E : A) (A C : A) : Prop :=
  ∃ a e : ℝ, a + e = 1 ∧ e < 0 ∧ E = a • A + e • C ∧ a/abs e = 4/1

def intersection (P E D : A) (B A : A) : P = intersection_of BE AD

theorem find_P (A B C : A) (D E P : A)
  (hD : is_in_triangle D B C) 
  (hE : is_extended_ratio E A C) 
  (hP : intersection P E D B A) : 
  P = (11/41) • A + (21/41) • B + (9/41) • C :=
sorry

end find_P_l133_133171


namespace cost_of_selected_items_l133_133996

-- Definitions for the conditions
def classes := 12
def price_folder := 3.50
def price_notebook := 3
def price_binder := 5
def price_pencil := 1
def price_eraser := 0.75
def price_highlighter := 3.25
def price_marker := 3.50
def price_sticky_notes := 2.50
def price_calculator := 10.50
def price_sketchbook := 4.50
def price_paints := 18
def price_color_pencils := 7
def total_spent := 210

-- Proving the combined cost of paints, color pencils, calculator, and sketchbook
theorem cost_of_selected_items :
  price_paints + price_color_pencils + price_calculator + price_sketchbook = 40 := by
  sorry

end cost_of_selected_items_l133_133996


namespace find_m_l133_133039

def ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + y^2 = 1

def focus1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def focus2 : ℝ × ℝ := (Real.sqrt 2, 0)

def line (x y m : ℝ) : Prop :=
  y = x + m

def intersects (m : ℝ) (x y : ℝ) : Prop :=
  ellipse x y ∧ line x y m

def area_relation (m : ℝ) : Prop :=
  let x := -m -- This might be simplified as it's used to calculate distances
  let dist1 := Real.abs (focus1.fst + m)
  let dist2 := Real.abs (focus2.fst + m)
  dist1 = 2 * dist2

theorem find_m :
  area_relation (-Real.sqrt 2 / 3) :=
by
  sorry

end find_m_l133_133039


namespace unit_digit_S2016_l133_133209

def sequence (a : ℕ → ℕ) : Prop :=
  (a 1 = 6) ∧ ∀ n, a (n+1) = ((5 * a n) / 4 + (3 * (a n ^ 2 - 2).sqrt()) / 4).to_nat

def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (finset.range n).sum (λi, a (i+1))

theorem unit_digit_S2016 (a : ℕ → ℕ) (h : sequence a) :
  (S a 2016) % 10 = 1 :=
sorry

end unit_digit_S2016_l133_133209


namespace original_triangle_area_l133_133625

theorem original_triangle_area (A_new : ℝ) (scale_factor : ℝ) (A_original : ℝ) 
  (h1: scale_factor = 5) (h2: A_new = 200) (h3: A_new = scale_factor^2 * A_original) : 
  A_original = 8 :=
by
  sorry

end original_triangle_area_l133_133625


namespace solve_equation_l133_133229

theorem solve_equation :
  (∀ x : ℝ, x ≠ 2 / 3 → (7 * x + 3) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) → (x = 1 / 3) ∨ (x = -3)) :=
by
  sorry

end solve_equation_l133_133229


namespace radius_of_inscribed_circle_l133_133864

open Real

/-- Triangle vertices -/
def B : (ℝ × ℝ) := (0, 0)
def A : (ℝ × ℝ) := (1, 0)
def C : (ℝ × ℝ) := (1, 2)

/-- Distance calculations -/
noncomputable def dist (p q : ℝ × ℝ) := sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def AB := dist A B
def BC := dist B C
def AC := dist A C

/-- Semi-perimeter calculation -/
noncomputable def semi_perimeter := (AB + BC + AC) / 2

/-- Area calculation using vertices of the triangle -/
noncomputable def area := 1 / 2 * (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - B.1 * A.2 - C.1 * B.2 - A.1 * C.2)

/-- Radius of the inscribed circle -/
noncomputable def radius_inscribed_circle := area / semi_perimeter

/-- The theorem to be verified: the radius of the inscribed circle -/
theorem radius_of_inscribed_circle:
  radius_inscribed_circle = (3 - sqrt 5) / 2 :=
by
  sorry

end radius_of_inscribed_circle_l133_133864


namespace find_m_l133_133035

variables (a b : ℝ) (x y m : ℝ) (A B F1 F2 : ℝ × ℝ)

-- Given conditions
def ellipse := (x^2 / 3) + y^2 = 1
def line := y = x + m
def foci_F1 := F1 = (-sqrt 2, 0)
def foci_F2 := F2 = (sqrt 2, 0)
def area_F1_AB := λ A B : ℝ × ℝ, 2 * |det(F1, A, B)| = |det(F2, A, B)|

-- The proof problem statement
theorem find_m : 
  ellipse ∧ 
  line ∧ 
  foci_F1 ∧ 
  foci_F2 ∧ 
  (area_F1_AB A B) →
  m = -sqrt(2) / 3 := sorry

end find_m_l133_133035


namespace new_volume_l133_133288

noncomputable def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h

theorem new_volume (V_orig : ℝ)
  (h r : ℝ)
  (hV : V_orig = cylinder_volume r h)
  (hVOrig : V_orig = 16) : 
  cylinder_volume (2 * r) (h / 2) = 32 :=
by
  sorry

end new_volume_l133_133288


namespace intersection_minimum_distance_l133_133890

theorem intersection_minimum_distance (α t1 t2 : ℝ) (hα : 0 < α ∧ α < π)
  (l_eqns : ∀ t : ℝ, 1 + t * Real.cos α = 4 / (Real.sin² α) ∧ t * Real.sin α * t * Real.sin α = 4 * (1 + t * Real.cos α)) :
  (|t1 - t2| = 4 ↔ α = π / 2) :=
begin
  sorry
end

end intersection_minimum_distance_l133_133890


namespace remainder_when_a_divided_by_11_l133_133198

theorem remainder_when_a_divided_by_11 (n : ℕ) (hn : 0 < n) :
  (∃ a, (a ≡ (5 ^ (2 * n) + 6)⁻¹ [MOD 11]) ∧ (if even n then a ≡ 8 [MOD 11] else a ≡ 9 [MOD 11])) :=
by
  sorry

end remainder_when_a_divided_by_11_l133_133198


namespace mathematicians_correct_l133_133256

noncomputable def scenario1 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 4 ∧ total1 = 7 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario2 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  whites1 = 8 ∧ total1 = 14 ∧ whites2 = 3 ∧ total2 = 5

noncomputable def scenario3 (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : Prop :=
  (19 / 35 < 4 / 7) ∧ (4 / 7 < 3 / 5)

noncomputable def probability (whites1 : ℕ) (total1 : ℕ) (whites2 : ℕ) (total2 : ℕ) : ℝ :=
  (whites1 + whites2) / (total1 + total2)

theorem mathematicians_correct :
  let whites1_s1 := 4 in
  let total1_s1 := 7 in
  let whites2_s1 := 3 in
  let total2_s1 := 5 in
  let whites1_s2 := 8 in
  let total1_s2 := 14 in
  let whites2_s2 := 3 in
  let total2_s2 := 5 in
  scenario1 whites1_s1 total1_s1 whites2_s1 total2_s1 →
  scenario2 whites1_s2 total1_s2 whites2_s2 total2_s2 →
  scenario3 whites1_s1 total1_s1 whites2_s2 total2_s2 →
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 ≤ 3 / 5 ∨
  probability whites1_s1 total1_s1 whites2_s1 total2_s1 = 4 / 7 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 ≤ 3 / 5 ∨
  probability whites1_s2 total1_s2 whites2_s2 total2_s2 = 4 / 7 :=
begin
  intros,
  sorry

end mathematicians_correct_l133_133256


namespace min_value_expression_l133_133977

open Real

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (min_value (λ (x y z : ℝ), (x / y + y / z + z / x + x / z)) = 4) :=
begin
  sorry
end

end min_value_expression_l133_133977


namespace second_cannibal_wins_l133_133356

/-- Define a data structure for the position on the chessboard -/
structure Position where
  x : Nat
  y : Nat
  deriving Inhabited, DecidableEq

/-- Check if two positions are adjacent in a legal move (vertical or horizontal) -/
def isAdjacent (p1 p2 : Position) : Bool :=
  (p1.x = p2.x ∧ (p1.y = p2.y + 1 ∨ p1.y = p2.y - 1)) ∨
  (p1.y = p2.y ∧ (p1.x = p2.x + 1 ∨ p1.x = p2.x - 1))

/-- Define the initial positions of the cannibals -/
def initialPositionFirstCannibal : Position := ⟨1, 1⟩
def initialPositionSecondCannibal : Position := ⟨8, 8⟩

/-- Define a move function for a cannibal (a valid move should keep it on the board) -/
def move (p : Position) (direction : String) : Position :=
  match direction with
  | "up"     => if p.y < 8 then ⟨p.x, p.y + 1⟩ else p
  | "down"   => if p.y > 1 then ⟨p.x, p.y - 1⟩ else p
  | "left"   => if p.x > 1 then ⟨p.x - 1, p.y⟩ else p
  | "right"  => if p.x < 8 then ⟨p.x + 1, p.y⟩ else p
  | _        => p

/-- Predicate determining if a cannibal can eat the other by moving to its position -/
def canEat (p1 p2 : Position) : Bool :=
  p1 = p2

/-- 
  Prove that the second cannibal will eat the first cannibal with the correct strategy. 
  We formalize the fact that with correct play, starting from the initial positions, 
  the second cannibal (initially at ⟨8, 8⟩) can always force a win.
-/
theorem second_cannibal_wins :
  ∀ (p1 p2 : Position), 
  p1 = initialPositionFirstCannibal →
  p2 = initialPositionSecondCannibal →
  (∃ strategy : (Position → String), ∀ positionFirstCannibal : Position, canEat (move p2 (strategy p2)) positionFirstCannibal) :=
by
  sorry

end second_cannibal_wins_l133_133356


namespace problem_l133_133450

theorem problem (f : ℝ → ℝ) (hf : Differentiable ℝ f) (hf' : f' 1 = 0) (hcond : ∀ x, (x - 1) * (Deriv f x) > 0) :
  f 0 + f 2 > 2 * f 1 :=
sorry

end problem_l133_133450


namespace distance_k_l133_133449

theorem distance_k (a b c d e k : ℝ) (h : {2, 4, 5, 7, 8, k, 13, 15, 17, 19} = {abs (a - b), abs (a - c), abs (a - d), abs (a - e), abs (b - c), abs (b - d), abs (b - e), abs (c - d), abs (c - e), abs (d - e)}) : k = 12 :=
sorry

end distance_k_l133_133449


namespace min_value_5x_5y_l133_133201

theorem min_value_5x_5y (x y : ℝ) (h : x + y = 4) : ∃ m : ℝ, m = 20 ∧ (∀ z : ℝ, z = 5*x + 5*y → z ≥ m) := 
by
  existsi 20
  split
  . refl
  . intro z h_z
    rw [←h_z, ←h]
    norm_num -- This line ensures that the arithmetic simplification happens, proving that 5 * 4 = 20.
    sorry -- Verification of the minimum under the given constraint will follow with a full proof.

end min_value_5x_5y_l133_133201


namespace president_vice_president_combinations_l133_133546

theorem president_vice_president_combinations (n : ℕ) (hn : n ≥ 2) : nat.choose n 2 + nat.choose n 1 = 42 :=
by {
  assume h: n = 7,
  ring_nf,
  sorry
}

end president_vice_president_combinations_l133_133546


namespace cos_alpha_plus_pi_over_3_range_of_f_A_in_triangle_l133_133507

noncomputable def f (x : ℝ) : ℝ :=
  (⟨sqrt 2 * cos (x / 4), 2 * cos (x / 4)⟩ : ℝ × ℝ) • (⟨sqrt 2 * cos (x / 4), sqrt 3 * sin (x / 4)⟩ : ℝ × ℝ)

theorem cos_alpha_plus_pi_over_3 (α : ℝ) (h : f α = 2) :
  cos (α + π / 3) = 1 / 2 := by sorry

theorem range_of_f_A_in_triangle (A B C a b c : ℝ) (h1 : (2 * a - b) * cos C = c * cos B)
  (h2 : A + B + C = π) -- This is a triangle condition, that the sum of angles = π
  (h3 : 0 < A) (h4 : A < π) : 
  2 < f A ∧ f A < 3 := by sorry

end cos_alpha_plus_pi_over_3_range_of_f_A_in_triangle_l133_133507


namespace option_c_is_incorrect_l133_133764

/-- Define the temperature data -/
def temps : List Int := [-20, -10, 0, 10, 20, 30]

/-- Define the speed of sound data corresponding to the temperatures -/
def speeds : List Int := [318, 324, 330, 336, 342, 348]

/-- The speed of sound at 10 degrees Celsius -/
def speed_at_10 : Int := 336

/-- The incorrect claim in option C -/
def incorrect_claim : Prop := (speed_at_10 * 4 ≠ 1334)

/-- Prove that the claim in option C is incorrect -/
theorem option_c_is_incorrect : incorrect_claim :=
by {
  sorry
}

end option_c_is_incorrect_l133_133764


namespace cricket_run_rate_l133_133730

theorem cricket_run_rate (first_10_overs_rate : ℝ) (target_runs : ℕ) :
  first_10_overs_rate = 3.2 →
  target_runs = 282 →
  ((target_runs - (first_10_overs_rate * 10))/40) = 6.25 :=
begin
  intros h1 h2,
  sorry
end

end cricket_run_rate_l133_133730


namespace mathematicians_correctness_l133_133263

theorem mathematicians_correctness:
  ∀ (w1₁ w1₂ t1₁ t1₂ w2₁ w2₂ t2₁ t2₂: ℕ),
  (w1₁ = 4) ∧ (t1₁ = 7) ∧ (w1₂ = 3) ∧ (t1₂ = 5) →
  (w2₁ = 8) ∧ (t2₁ = 14) ∧ (w2₂ = 3) ∧ (t2₂ = 5) →
  (4/7 : ℚ) < 3/5 →
  (w1₁ + w1₂) / (t1₁ + t1₂) = 7 / 12 ∨ (w2₁ + w2₂) / (t2₁ + t2₂) = 11 / 19 →
  ¬ (w1₁ + w1₂) / (t1₁ + t1₂) < 19 / 35 ∧ (w2₁ + w2₂) / (t2₁ + t2₂) < 19 / 35 :=
by
  sorry

end mathematicians_correctness_l133_133263


namespace line_ellipse_common_point_l133_133889

theorem line_ellipse_common_point (a m : ℝ) :
  (a + 1) * 3 + (3 * a - 1) * 1 - (6 * a + 2) = 0 →
  m ≠ 16 →
  (9 / 16 : ℝ) + (1 / m) ≤ 1 →
  m ∈ set.Ico (16 / 7) 16 ∪ set.Ioi 16 :=
sorry

end line_ellipse_common_point_l133_133889


namespace remaining_watermelons_l133_133955

-- Define the given conditions
def initial_watermelons : ℕ := 35
def watermelons_eaten : ℕ := 27

-- Define the question as a theorem
theorem remaining_watermelons : 
  initial_watermelons - watermelons_eaten = 8 :=
by
  sorry

end remaining_watermelons_l133_133955


namespace smallest_positive_x_for_palindrome_l133_133719

def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr 
  s = s.reverse

theorem smallest_positive_x_for_palindrome :
  ∃ x > 0, is_palindrome (x + 7234) ∧ ∀ y > 0, y < x → ¬ is_palindrome (y + 7234) ∧ x = 213 :=
by {
  sorry
}

end smallest_positive_x_for_palindrome_l133_133719


namespace problems_per_page_l133_133345

theorem problems_per_page (total_problems finished_problems pages_left problems_per_page : ℕ)
  (h1 : total_problems = 40)
  (h2 : finished_problems = 26)
  (h3 : pages_left = 2)
  (h4 : total_problems - finished_problems = pages_left * problems_per_page) :
  problems_per_page = 7 :=
by
  sorry

end problems_per_page_l133_133345


namespace aquarium_pufferfish_problem_l133_133301

/-- Define the problem constants and equations -/
theorem aquarium_pufferfish_problem :
  ∃ (P S : ℕ), S = 5 * P ∧ S + P = 90 ∧ P = 15 :=
by
  sorry

end aquarium_pufferfish_problem_l133_133301


namespace max_sides_of_smaller_polygons_in_heptagon_l133_133774

theorem max_sides_of_smaller_polygons_in_heptagon :
  ∃ m : ℕ, (∀ P, P ∈ convex_polygons_formed_by_diagonals_of_heptagon → sides P ≤ m) ∧ 
             m = 7 :=
by
  sorry

-- Definitions go below:
def heptagon : Type := { x : Type // cardinal.mk x = 7 }
def convex_polygons_formed_by_diagonals_of_heptagon : set (set (set heptagon)) :=
  { P | is_convex P ∧ ∃ D, is_diagonal D ∧ forms_non_overlapping_polygons P D }

end max_sides_of_smaller_polygons_in_heptagon_l133_133774


namespace length_of_second_train_l133_133354

theorem length_of_second_train 
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (time_to_cross : ℝ)
  (relative_speed : ℝ)
  (total_distance_covered : ℝ)
  (L : ℝ)
  (h1 : length_first_train = 210)
  (h2 : speed_first_train = 120 * 1000 / 3600)
  (h3 : speed_second_train = 80 * 1000 / 3600)
  (h4 : time_to_cross = 9)
  (h5 : relative_speed = (120 * 1000 / 3600) + (80 * 1000 / 3600))
  (h6 : total_distance_covered = relative_speed * time_to_cross)
  (h7 : total_distance_covered = length_first_train + L) : 
  L = 289.95 :=
by {
  sorry
}

end length_of_second_train_l133_133354


namespace product_in_M_l133_133862

def M : Set ℤ := {x | ∃ (a b : ℤ), x = a^2 - b^2}

theorem product_in_M (p q : ℤ) (hp : p ∈ M) (hq : q ∈ M) : p * q ∈ M :=
by
  sorry

end product_in_M_l133_133862


namespace amelia_painted_faces_l133_133386

def faces_of_cuboid : ℕ := 6
def number_of_cuboids : ℕ := 6

theorem amelia_painted_faces : faces_of_cuboid * number_of_cuboids = 36 :=
by {
  sorry
}

end amelia_painted_faces_l133_133386


namespace university_average_age_l133_133729

theorem university_average_age
  (n : ℕ)
  (avg_age_arts : ℕ := 21)
  (avg_age_tech : ℕ := 18)
  (num_arts_classes : ℕ := 8)
  (num_tech_classes : ℕ := 5) :
  ((avg_age_arts * num_arts_classes * n) + (avg_age_tech * num_tech_classes * n)) / 
  ((num_arts_classes * n) + (num_tech_classes * n)) = 19.85 := by
  sorry

end university_average_age_l133_133729


namespace light_ray_reflection_l133_133306

def reflected_vector (v : ℝ × ℝ × ℝ) (plane : ℕ) : ℝ × ℝ × ℝ :=
  match plane with
  | 0 => (v.1, v.2, -v.3)
  | 1 => (v.1, -v.2, v.3)
  | 2 => (-v.1, v.2, v.3)
  | _ => v

theorem light_ray_reflection (a b c : ℝ) :
  let v := (a, b, c)
  let v1 := reflected_vector v 0
  let v2 := reflected_vector v1 1
  let v3 := reflected_vector v2 2
  v3 = (-a, -b, -c) :=
by 
  let v := (a, b, c)
  let v1 := reflected_vector v 0
  let v2 := reflected_vector v1 1
  let v3 := reflected_vector v2 2
  show v3 = (-a, -b, -c)
  sorry

end light_ray_reflection_l133_133306


namespace seats_in_16th_row_l133_133929

def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem seats_in_16th_row : arithmetic_sequence 5 2 16 = 35 := by
  sorry

end seats_in_16th_row_l133_133929


namespace min_value_expression_l133_133828

theorem min_value_expression : ∃ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 6 * x + 4 * y + 5 = 2 := 
sorry

end min_value_expression_l133_133828


namespace problem1_problem2_problem3_l133_133397

-- Problem 1 Lean Statement
theorem problem1 : sqrt 25 + cbrt (-27) - sqrt (1 / 9) = 5 / 3 :=
by sorry

-- Problem 2 Lean Statement
theorem problem2 : abs (sqrt 3 - 2) + sqrt 9 + cbrt (-64) = 1 - sqrt 3 :=
by sorry

-- Problem 3 Lean Statement
theorem problem3 (x y : ℝ) : 
  (2 * x - y = 5) ∧ (3 * x + 4 * y = 2) → (x = 2 ∧ y = -1) :=
by sorry

end problem1_problem2_problem3_l133_133397


namespace tangency_points_diameter_l133_133241

theorem tangency_points_diameter 
  {k1 k2 k3 : Circle}
  (h1 : Tangent k1 k2)
  (h2 : Tangent k2 k3)
  (h3 : Tangent k3 k1) :
  ∃ (A B : Point), A ∈ k3 ∧ B ∈ k3 ∧ Diameter k3 A B := 
sorry

end tangency_points_diameter_l133_133241


namespace intersecting_lines_a_b_sum_zero_l133_133844

theorem intersecting_lines_a_b_sum_zero
    (a b : ℝ)
    (h₁ : ∀ z : ℝ × ℝ, z = (3, -3) → z.1 = (1 / 3) * z.2 + a)
    (h₂ : ∀ z : ℝ × ℝ, z = (3, -3) → z.2 = (1 / 3) * z.1 + b)
    :
    a + b = 0 := by
  sorry

end intersecting_lines_a_b_sum_zero_l133_133844


namespace find_m_l133_133042

def ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + y^2 = 1

def focus1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def focus2 : ℝ × ℝ := (Real.sqrt 2, 0)

def line (x y m : ℝ) : Prop :=
  y = x + m

def intersects (m : ℝ) (x y : ℝ) : Prop :=
  ellipse x y ∧ line x y m

def area_relation (m : ℝ) : Prop :=
  let x := -m -- This might be simplified as it's used to calculate distances
  let dist1 := Real.abs (focus1.fst + m)
  let dist2 := Real.abs (focus2.fst + m)
  dist1 = 2 * dist2

theorem find_m :
  area_relation (-Real.sqrt 2 / 3) :=
by
  sorry

end find_m_l133_133042


namespace gcd_three_digit_palindromes_l133_133699

open Nat

theorem gcd_three_digit_palindromes :
  (∀ a b : ℕ, a ≠ 0 → a < 10 → b < 10 → True) ∧
  let S := {n | ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b} in
  S.Gcd = 1 := by
  sorry

end gcd_three_digit_palindromes_l133_133699


namespace gcd_three_digit_palindromes_l133_133705

theorem gcd_three_digit_palindromes : 
  GCD (set.image (λ (p : ℕ × ℕ), 101 * p.1 + 10 * p.2) 
    ({a | a ≠ 0 ∧ a < 10} × {b | b < 10})) = 1 := 
by
  sorry

end gcd_three_digit_palindromes_l133_133705


namespace matrix_solution_l133_133964

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 1], ![5, 3]]

def B : Matrix (Fin 2) (Fin 1) ℤ := ![![4], ![11]]

def X : Matrix (Fin 2) (Fin 1) ℤ := ![![1], ![2]]

theorem matrix_solution : ∃ X : Matrix (Fin 2) (Fin 1) ℤ, (A ⬝ X) = B ∧ X = ![![1], ![2]] :=
  by
  use X
  split
  sorry
  rfl

end matrix_solution_l133_133964


namespace aquarium_pufferfish_problem_l133_133302

/-- Define the problem constants and equations -/
theorem aquarium_pufferfish_problem :
  ∃ (P S : ℕ), S = 5 * P ∧ S + P = 90 ∧ P = 15 :=
by
  sorry

end aquarium_pufferfish_problem_l133_133302


namespace v4_at_2_eq_3_l133_133313

noncomputable def horner_poly (x : ℝ) : ℝ :=
    (((((x - 2) * x + 0) * x + 0) * x + 3) * x - 4) * x + 0

def v4 (x : ℝ) : ℝ :=
    (((((x - 2) * x + 0) * x + 0) * x + 3) * x + 0)

theorem v4_at_2_eq_3 : v4 2 = 3 :=
by
  have h : v4 2 = (((((2 - 2) * 2 + 0) * 2 + 0) * 2 + 3) * 2 + 0) := rfl
  rw h
  norm_num
  sorry

end v4_at_2_eq_3_l133_133313


namespace bus_ride_time_l133_133990

def walking_time : ℕ := 15
def waiting_time : ℕ := 2 * walking_time
def train_ride_time : ℕ := 360
def total_trip_time : ℕ := 8 * 60

theorem bus_ride_time : 
  (total_trip_time - (walking_time + waiting_time + train_ride_time)) = 75 := by
  sorry

end bus_ride_time_l133_133990


namespace conditions_equivalent_l133_133020

open Finset

variable {n : ℕ} (D : Finset ℕ) (f : ℕ → ℤ)

-- Declare that 'n' is a positive integer
axiom positive_n : 0 < n

-- Declare that 'D' is the set of positive divisors of 'n'
axiom divisors_D : ∀ d ∈ D, d ∣ n ∧ 0 < d

-- Define conditions 'a' and 'b'
def condition_a : Prop :=
  ∀ m ∈ D, n ∣ ∑ d in D.filter (λ d, d ∣ m), f d * Nat.choose (n / d) (m / d)

def condition_b : Prop :=
  ∀ k ∈ D, k ∣ ∑ d in D.filter (λ d, d ∣ k), f d

-- Prove that conditions 'a' and 'b' are equivalent
theorem conditions_equivalent : condition_a D f ↔ condition_b D f :=
sorry

end conditions_equivalent_l133_133020


namespace sum_of_diameters_l133_133942

section CircleDiameters

variables (D d QV ST : ℝ)
variables (P Q R S T U O V : Type) -- Placeholder types for points in the diagram
variables [Geometry Q R S T U O] -- Assuming a geometry context

-- Assume the conditions given in the problem
variables (c1 : D - d = 9)
variables (c2 : QV = 9)
variables (c3 : ST = 5)

-- The theorem stating the conclusion
theorem sum_of_diameters : D + d = 91 :=
sorry

end CircleDiameters

end sum_of_diameters_l133_133942


namespace triangle_sum_range_l133_133170

variables {A B C : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C]

noncomputable def angle_C (A B C : ℝ) : ℝ := 60

noncomputable def AB : ℝ := 2

theorem triangle_sum_range (AC BC : ℝ) (h1 : angle_C A B C = 60)
  (h2 : AB = 2) :
  2 < AC + BC ∧ AC + BC ≤ 4 :=
sorry

end triangle_sum_range_l133_133170
