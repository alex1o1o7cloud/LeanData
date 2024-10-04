import Mathlib

namespace cos_gamma_prime_l787_787355

theorem cos_gamma_prime : 
  ∀ (α' β' γ' : ℝ), 
  ∃ (cα cβ cγ : ℝ), 
    cα = (2 / 5) ∧
    cβ = (1 / 4) ∧ 
    (cα ^ 2 + cβ ^ 2 + cγ ^ 2 = 1) → 
    cγ = (sqrt 311 / 20) :=
by
  sorry

end cos_gamma_prime_l787_787355


namespace find_natural_numbers_l787_787954

theorem find_natural_numbers (a b : ℕ) (p : ℕ) (hp : Nat.Prime p)
  (h : a^3 - b^3 = 633 * p) : a = 16 ∧ b = 13 :=
by
  sorry

end find_natural_numbers_l787_787954


namespace pascal_triangle_row_10_sum_l787_787699

theorem pascal_triangle_row_10_sum :
  (∑ k in Finset.range (11), Nat.choose 10 k) = 1024 :=
by 
sorry

end pascal_triangle_row_10_sum_l787_787699


namespace trigonometric_solution_l787_787787

theorem trigonometric_solution
    (x y z : ℝ)
    (h₀ : cos x ≠ 0)
    (h₁ : cos y ≠ 0)
    (n k m : ℤ)
    (hx : x = n * π)
    (hy : y = k * π)
    (hz : z = π / 2 + 2 * m * π) :
    (\(cos x)\^2 + 1 / (cos x)\^2)\^3 + (\(cos y)\^2 + 1 / (cos y)\^2)\^3 = 16 * sin z := by
  sorry

end trigonometric_solution_l787_787787


namespace a_6_equals_l787_787834

noncomputable def S (n : ℕ) : ℕ := sorry

def a : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => 3 * S (n+1)

theorem a_6_equals : a 6 = 3 * 4^4 := by
  sorry

end a_6_equals_l787_787834


namespace range_of_a_l787_787626

variable {a : ℝ}
def f (x : ℝ) : ℝ := x * Real.log x + 2 + 1 / a
def g (x : ℝ) : ℝ := -x^2 - (-5) * x - 4  -- Given b is solved to be -5 from x = 5/2

theorem range_of_a (h1 : ∀ x1 ∈ Set.Icc (Real.exp (-1)) 1, ∃ x2 < 3, f x1 = g x2) : a < 0 := 
  sorry

end range_of_a_l787_787626


namespace number_of_people_today_l787_787515

/-- Define the initial conditions and the problem statement to prove
there are 28 people in the group today. -/
theorem number_of_people_today (P M : ℕ) 
  (h₁ : 550 = P * M) 
  (h₂ : 550 = (P + 5) * (M - 3)) : 
  P = 28 :=
begin
  sorry
end

end number_of_people_today_l787_787515


namespace prime_implication_divisibility_pairs_l787_787831

-- Definition of the sequence (x_n)
def sequence_x : ℕ → ℕ
| 0       := 2
| 1       := 1
| (n + 2) := sequence_x (n + 1) + sequence_x n

-- Part (a)
theorem prime_implication (n : ℕ) (hn : n ≥ 1) (hx_n : Nat.Prime (sequence_x n)) :
  Nat.Prime n ∨ ¬ ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ p ≠ 2 :=
sorry

-- Part (b)
theorem divisibility_pairs (m n : ℕ) (h : sequence_x m ∣ sequence_x n) :
  (m = 0 ∧ ∃ k : ℕ, n = 3 * k) ∨
  (m = 1 ∧ ∃ k : ℕ, n = k) ∨
  (m > 1 ∧ ∃ k : ℕ, n = (2 * k + 1) * m) :=
sorry

end prime_implication_divisibility_pairs_l787_787831


namespace find_function_expression_l787_787978

theorem find_function_expression (f : ℝ → ℝ) : 
  (∀ x : ℝ, f (x - 1) = x^2 - 3 * x) → 
  (∀ x : ℝ, f x = x^2 - x - 2) :=
by
  sorry

end find_function_expression_l787_787978


namespace find_lambda_l787_787662

theorem find_lambda
  (λ : ℝ)
  (a : ℝ × ℝ := (λ, 1))
  (b : ℝ × ℝ := (λ + 2, 1))
  (h : (2 * λ + 2)^2 + 4 = 4) :
  λ = -1 :=
sorry

end find_lambda_l787_787662


namespace expression_not_equal_l787_787481

theorem expression_not_equal :
  let e1 := 250 * 12
  let e2 := 25 * 4 + 30
  let e3 := 25 * 40 * 3
  let product := 25 * 120
  e2 ≠ product :=
by
  let e1 := 250 * 12
  let e2 := 25 * 4 + 30
  let e3 := 25 * 40 * 3
  let product := 25 * 120
  sorry

end expression_not_equal_l787_787481


namespace smallest_m_correct_l787_787752

def smallest_m (k : ℕ) (h : k > 1) : ℕ :=
if h₀ : Even k then
  let n := k / 2 in (n.factorial * n.factorial) + 1
else
  let m := (k - 1) / 2 in let n := (k + 1) / 2 in (m.factorial * n.factorial) + 1

theorem smallest_m_correct (k : ℕ) (h : k > 1) : ∃ f : ℤ[X], 
  (f.coeff_range.all (λ c, c ∈ ℤ)) ∧ (∃ α : ℤ, f.eval α = 1) ∧ 
  (∃ m : ℤ, f.eval_range.count_eq m k) ∧ m = smallest_m k h := sorry

end smallest_m_correct_l787_787752


namespace integral_sin_pi_half_to_three_pi_half_l787_787593

theorem integral_sin_pi_half_to_three_pi_half :
  ∫ x in (Set.Icc (Real.pi / 2) (3 * Real.pi / 2)), Real.sin x = 0 :=
by
  sorry

end integral_sin_pi_half_to_three_pi_half_l787_787593


namespace conditional_probability_l787_787867

theorem conditional_probability (P : set → ℝ) (A B : set) (hPB : P B = 0.3) (hPAB : P (A ∩ B) = 0.2) :
  P (A ∩ B) / P B = 2 / 3 :=
by sorry

end conditional_probability_l787_787867


namespace no_three_ones_on_top_l787_787386

theorem no_three_ones_on_top (D1 D2 D3 D4 : ℕ → ℕ) (h1 : ∀ x, D1 x + D1 (7 - x) = 7)
  (h2 : ∀ x, D2 x + D2 (7 - x) = 7) (h3 : ∀ x, D3 x + D3 (7 - x) = 7)
  (h4 : ∀ x, D4 x + D4 (7 - x) = 7) 
  (hD1 : D1 1 = 1 ∧ D1 2 = 2 ∧ D1 4 = 4 ∧ D1 5 = 5 ∧ D1 3 = 3)
  (hD2 : D2 1 = 1 ∧ D2 2 = 2 ∧ D2 4 = 4 ∧ D2 5 = 5 ∧ D2 3 = 3)
  (hD3 : D3 1 = 1 ∧ D3 2 = 2 ∧ D3 4 = 4 ∧ D3 5 = 5 ∧ D3 3 = 3)
  (hD4 : D4 1 = 1 ∧ D4 2 = 2 ∧ D4 4 = 4 ∧ D4 5 = 5 ∧ D4 3 = 3) :
  ¬ (∃ (D : matrix (fin 2) (fin 2) (ℕ → ℕ)),
        (D 0 0 = D1 ∧ D 0 1 = D2 ∧ D 1 0 = D3 ∧ D 1 1 = D4) ∧
        (D1 1 = 1 ∧ D2 1 = 1 ∧ D3 1 = 1 ∧ D4 1 ≠ 1) ∧
        (∀ i j, D i j (D i j) = D i j)) := sorry

end no_three_ones_on_top_l787_787386


namespace problem_statement_l787_787644

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x)

lemma f_odd : ∀ x : ℝ, f (-x) = -f x :=
begin
  intro x,
  unfold f,
  rw [Real.log_neg_eq_log, Real.log_inv],
end

lemma f_periodic : ∀ x : ℝ, f (x + 3) = f x :=
begin
  intro x,
  unfold f,
  ring_nf,
  rw Real.log_inv_eq_neg_log,
  congr' 1,
end

theorem problem_statement : f 2010 + f 2011 = 1 :=
by {
  unfold f,
  have H : f 2010 = f (2010 % 3), sorry,
  have H0 : 2010 % 3 = 0, sorry,
  have H1 : 2011 % 3 = 1, sorry,
  rw [H, H0],
  rw [H, H1],
  dsimp [f],
  sorry
}

end problem_statement_l787_787644


namespace xyz_value_l787_787635

theorem xyz_value (x y z : ℚ)
  (h1 : x + y + z = 1)
  (h2 : x + y - z = 2)
  (h3 : x - y - z = 3) :
  x * y * z = 1/2 :=
by
  sorry

end xyz_value_l787_787635


namespace skateboard_speed_l787_787797

noncomputable def skateboard_speed_mph (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * (3600 / 5280)

theorem skateboard_speed : skateboard_speed_mph 330 45 = 10.75 := 
  by {
    rw [skateboard_speed_mph],
    norm_num,
    sorry
  }

end skateboard_speed_l787_787797


namespace right_triangle_is_stable_l787_787132

def shape := Type

structure Parallelogram : shape
structure RightTriangle : shape
structure Rectangle : shape
structure Square : shape

def is_stable (s : shape) : Prop :=
  s = RightTriangle

theorem right_triangle_is_stable :
  is_stable RightTriangle :=
by
  sorry

end right_triangle_is_stable_l787_787132


namespace ellipse_with_given_parameters_l787_787812

noncomputable def ellipse_equation : ℝ → ℝ → ℝ
  | x, y => x^2 / 2 + y^2

theorem ellipse_with_given_parameters (a b : ℝ) (c : ℝ)
  (h1 : foci_on_x_axis) 
  (h2 : minor_axis_length b 2) 
  (h3 : eccentricity (c / a) (sqrt 2 / 2))
  (h4 : a^2 = b^2 + c^2) :
  ellipse_equation x y = 1 := 
  sorry

end ellipse_with_given_parameters_l787_787812


namespace value_of_x_l787_787972

def g (x : ℝ) : ℝ := (3 * x^2 - 7 * x - 8) / (x^2 - 5 * x + 3)

theorem value_of_x (x : ℝ) (h : g(x) = 3) (hx₁ : x ≠ 1) (hx₂ : x ≠ 3) : x = 17 / 8 :=
sorry

end value_of_x_l787_787972


namespace climbing_stairs_l787_787896

noncomputable def total_methods_to_climb_stairs : ℕ :=
  (Nat.choose 8 5) + (Nat.choose 8 6) + (Nat.choose 8 7) + 1

theorem climbing_stairs (n : ℕ := 9) (min_steps : ℕ := 6) (max_steps : ℕ := 9)
  (H1 : min_steps ≤ n)
  (H2 : n ≤ max_steps)
  : total_methods_to_climb_stairs = 93 := by
  sorry

end climbing_stairs_l787_787896


namespace find_a_l787_787619

noncomputable def modulus (z : ℂ) : ℝ := complex.abs z

theorem find_a (a : ℝ) (h_a_neg : a < 0) :
  let z : ℂ := (3 * a * complex.I) / (1 - 2 * complex.I)
  modulus z = real.sqrt 5 → a = - (5/3) :=
by
  intro z h_modulus
  sorry

end find_a_l787_787619


namespace hexagonal_prism_cross_section_area_hexagonal_prism_cross_section_area_hexagonal_prism_cross_section_area_l787_787916

theorem hexagonal_prism_cross_section_area (a α : ℝ) (α_range1 : 0 < α ∧ α < π / 6) :
  let S := 3 * a^2 * sqrt 3 / (2 * cos α) in
  S = cross_section_area a α := sorry

theorem hexagonal_prism_cross_section_area (a α : ℝ) (α_range2 : π / 6 ≤ α ∧ α < arctan (2 / sqrt 3)) :
  let S := a^2 / (6 * cos α) * (18 * cot α - 3 * sqrt 3 - 2 * sqrt 3 * (cot α)^2) in
  S = cross_section_area a α := sorry

theorem hexagonal_prism_cross_section_area (a α : ℝ) (α_range3 : arctan (2 / sqrt 3) ≤ α ∧ α < π / 2) :
  let S := a^2 / (sqrt 3 * sin α) * (sqrt 3 + cot α) in
  S = cross_section_area a α := sorry

end hexagonal_prism_cross_section_area_hexagonal_prism_cross_section_area_hexagonal_prism_cross_section_area_l787_787916


namespace negation_of_universal_prop_eq_exists_l787_787824

theorem negation_of_universal_prop_eq_exists :
  ¬(∀ x : ℝ, 2^x ≤ 0) ↔ ∃ x : ℝ, 2^x > 0 :=
by 
  sorry

end negation_of_universal_prop_eq_exists_l787_787824


namespace lowest_score_within_2_std_devs_l787_787808

theorem lowest_score_within_2_std_devs (mean std_dev : ℝ) (h_mean : mean = 60) (h_std_dev : std_dev = 10) :
  let lowest_score := mean - 2 * std_dev 
  in lowest_score = 40 :=
by {
  rw [h_mean, h_std_dev],
  let lowest_score := 60 - 2 * 10,
  -- sorry to indicate the gap where proof would be
  sorry,
}

end lowest_score_within_2_std_devs_l787_787808


namespace line_AB_fixed_tangent_circle_l787_787351

-- Define the problem
variables {Ω Γ : Set ℝ} {P A B: ℝ}

-- Circles Ω and Γ with Ω inside Γ, and point P on Γ
-- Lines PA and PB are tangents from P to Ω
axiom interior_of (Ω Γ : Set ℝ) : Ω ⊆ Γ
axiom point_on_circumference (P : ℝ) : P ∈ Γ
axiom tangents (A B P : ℝ) : ∀ (A B : ℝ), A ≠ P ∧ B ≠ P ∧ line P A ∈ tangent_to Ω ∧ line P B ∈ tangent_to Ω

-- Prove that the line AB is tangent to a fixed circle when P varies on Γ
theorem line_AB_fixed_tangent_circle :
  (∀ P ∈ Γ, ∃ C : Set ℝ, (tangent_line (line A B) C)) :=
sorry

end line_AB_fixed_tangent_circle_l787_787351


namespace max_area_triangle_l787_787718

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

noncomputable def line_eq (x y : ℝ) : Prop := 2 * Real.sqrt 2 * x - y - 1 = 0

theorem max_area_triangle (x1 y1 x2 y2 xp yp : ℝ) (h1 : circle_eq x1 y1) (h2 : circle_eq x2 y2) (h3 : circle_eq xp yp)
  (h4 : line_eq x1 y1) (h5 : line_eq x2 y2) (h6 : (xp, yp) ≠ (x1, y1)) (h7 : (xp, yp) ≠ (x2, y2)) :
  ∃ S : ℝ, S = 10 * Real.sqrt 5 / 9 :=
by
  sorry

end max_area_triangle_l787_787718


namespace original_cost_of_meal_l787_787187

-- Definitions for conditions
def meal_cost (initial_cost : ℝ) : ℝ :=
  initial_cost + 0.085 * initial_cost + 0.18 * initial_cost

-- The theorem we aim to prove
theorem original_cost_of_meal (total_cost : ℝ) (h : total_cost = 35.70) :
  ∃ initial_cost : ℝ, initial_cost = 28.23 ∧ meal_cost initial_cost = total_cost :=
by
  use 28.23
  rw [meal_cost, h]
  sorry

end original_cost_of_meal_l787_787187


namespace tangents_in_circumcircle_triang_BOC_l787_787993

variables {A B C O : Point}
variables (hABC : ∀ {A B C : Point}, IsAcuteTriangle A B C)
variables (hO : ∀ {A B C : Point}, IsCircumcenter O A B C)
variables (ω : Circle)
variables (h1 : TangentsAt ω A AO)
variables (h2 : Tangents ω BC)
variables {B C O : Point}

theorem tangents_in_circumcircle_triang_BOC 
  (hABC : AcuteTriangle ABC)
  (hO : Circumcenter O ABC)
  (ω : Circle.tangentAt A AO)
  (ω : Circle.tangentLine BC)
  : ∃ ω, Tangents ω (Circumcircle BOC) := 
sorry

end tangents_in_circumcircle_triang_BOC_l787_787993


namespace lengths_of_DF_l787_787012

noncomputable def quadrilateral := sorry -- Placeholder for quadrilateral definition

noncomputable def ∠ := sorry -- Placeholder for angle definition
noncomputable def length := sorry -- Placeholder for length definition
noncomputable def perpendicular := sorry -- Placeholder for perpendicular definition

theorem lengths_of_DF (A B C D E F: quadrilateral)
  (h1 : ∠ A B C = 90)
  (h2 : ∠ C D A = 90)
  (h3 : length (B, C) = 7)
  (h4 : on_segment E B D)
  (h5 : on_segment F B D)
  (h6 : perpendicular (A, E) (B, D))
  (h7 : perpendicular (C, F) (B, D))
  (h8 : length (B, E) = 3) :
  ∃ (DF_smallest DF_largest : ℝ), DF_smallest * DF_largest = 9 :=
sorry

end lengths_of_DF_l787_787012


namespace smallest_integer_value_l787_787475

theorem smallest_integer_value (x : ℤ) (h1 : 7 - 3 * x > 22) (h2 : x < 5) : x ≤ -6 := by
  have h : x < -5 := by
    have : -3 * x > 15 := by
      linarith
    have : x < -5 := by
      linarith
    exact this
  have : -6 < x -> false := by
    intro h_neg
    have : -6 + 1 = -5 := by
      norm_num
    have : x < -5 := h
    linarith
  exact Int.le_of_lt_add_one this
  sorry

end smallest_integer_value_l787_787475


namespace total_candies_l787_787052

variable (Adam James Rubert : Nat)
variable (Adam_has_candies : Adam = 6)
variable (James_has_candies : James = 3 * Adam)
variable (Rubert_has_candies : Rubert = 4 * James)

theorem total_candies : Adam + James + Rubert = 96 :=
by
  sorry

end total_candies_l787_787052


namespace children_being_catered_l787_787486

-- Define the total meal units available
def meal_units_for_adults : ℕ := 70
def meal_units_for_children : ℕ := 90
def meals_eaten_by_adults : ℕ := 14
def remaining_meal_units : ℕ := meal_units_for_adults - meals_eaten_by_adults

theorem children_being_catered :
  (remaining_meal_units * meal_units_for_children) / meal_units_for_adults = 72 := by
{
  sorry
}

end children_being_catered_l787_787486


namespace evaluate_expression_l787_787951

theorem evaluate_expression (k : ℤ) : 
  3 ^ (-k) - 3 ^ (-(k + 2)) + 3 ^ (-(k + 1)) - 3 ^ (-(k - 1)) = (-16 * 3 ^ (-k)) / 9 :=
by sorry

end evaluate_expression_l787_787951


namespace range_of_k_for_distinct_real_roots_l787_787971

theorem range_of_k_for_distinct_real_roots (k : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (k - 1) * x1^2 - 2 * x1 + 1 = 0 ∧ (k - 1) * x2^2 - 2 * x2 + 1 = 0) →
    k < 2 ∧ k ≠ 1 :=
by
  sorry

end range_of_k_for_distinct_real_roots_l787_787971


namespace identity_proof_l787_787689

theorem identity_proof (x y : ℝ) (h1 : x + y = 5 / 11) (h2 : x - y = 1 / 55) : x^2 - y^2 = 1 / 121 :=
by 
  sorry

end identity_proof_l787_787689


namespace average_is_51_l787_787414

-- Define the problem conditions
variables (S L : ℕ) 

-- 5 positive integers with the necessary conditions
def is_valid_set (a b c d e : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  max a (max b (max c (max d e))) - min a (min b (min c (min d e))) = 10 ∧
  max a (max b (max c (max d e))) ≤ 53

-- Define the set of integers
def integer_set : set (ℕ × ℕ × ℕ × ℕ × ℕ) :=
  { (a, b, c, d, e) | is_valid_set a b c d e }

-- Calculate average
noncomputable def average (a b c d e : ℕ) : ℚ :=
  (a + b + c + d + e : ℚ) / 5

-- Statement of the problem
theorem average_is_51 :
  ∃ (a b c d e : ℕ), (a, b, c, d, e) ∈ integer_set ∧ average a b c d e = 51 := 
by
  sorry


end average_is_51_l787_787414


namespace longest_side_length_l787_787085

-- Define the sides of the triangle
def side_a : ℕ := 9
def side_b (x : ℕ) : ℕ := 2 * x + 3
def side_c (x : ℕ) : ℕ := 3 * x - 2

-- Define the perimeter condition
def perimeter_condition (x : ℕ) : Prop := side_a + side_b x + side_c x = 45

-- Main theorem statement: Length of the longest side is 19
theorem longest_side_length (x : ℕ) (h : perimeter_condition x) : side_b x = 19 ∨ side_c x = 19 :=
sorry

end longest_side_length_l787_787085


namespace missing_fraction_l787_787761

-- Define the initial conditions
variables (x : ℝ)
def coins_lost := (1 / 3) * x
def coins_spent := (1 / 4) * x
def coins_found := (5 / 6) * coins_lost
def remaining_lost := coins_lost - coins_found

-- The main theorem statement
theorem missing_fraction : (coins_spent + remaining_lost) / x = 11 / 36 :=
by
    -- To skip the proof for now
    sorry

end missing_fraction_l787_787761


namespace find_x_l787_787963

theorem find_x (x : ℝ) (h_pos : 0 < x) (h_eq : x * floor x = 48) : x = 8.0 :=
sorry

end find_x_l787_787963


namespace distance_to_y_axis_of_3_neg5_l787_787332

theorem distance_to_y_axis_of_3_neg5 : 
  ∀ (x y : ℝ), (x, y) = (3, -5) → abs x = 3 :=
by
  intros x y h
  rw [Prod.mk.inj_iff] at h
  cases h
  rw [h_left]
  exact abs_of_pos (by norm_num [h_left])

end distance_to_y_axis_of_3_neg5_l787_787332


namespace total_marbles_is_260_l787_787215

-- Define the number of marbles in each jar based on the conditions.
def first_jar : Nat := 80
def second_jar : Nat := 2 * first_jar
def third_jar : Nat := (1 / 4 : ℚ) * first_jar

-- Prove that the total number of marbles is 260.
theorem total_marbles_is_260 : first_jar + second_jar + third_jar = 260 := by
  sorry

end total_marbles_is_260_l787_787215


namespace max_result_l787_787183

-- Define the expressions as Lean definitions
def expr1 : Int := 2 + (-2)
def expr2 : Int := 2 - (-2)
def expr3 : Int := 2 * (-2)
def expr4 : Int := 2 / (-2)

-- State the theorem
theorem max_result : 
  (expr2 = 4) ∧ (expr2 > expr1) ∧ (expr2 > expr3) ∧ (expr2 > expr4) :=
by
  sorry

end max_result_l787_787183


namespace train_departure_time_l787_787835

-- Conditions
def arrival_time : ℕ := 1000  -- Representing 10:00 as 1000 (in minutes since midnight)
def travel_time : ℕ := 15  -- 15 minutes

-- Definition of time subtraction
def time_sub (arrival : ℕ) (travel : ℕ) : ℕ :=
arrival - travel

-- Proof that the train left at 9:45
theorem train_departure_time : time_sub arrival_time travel_time = 945 := by
  sorry

end train_departure_time_l787_787835


namespace first_degree_function_solution_l787_787633

-- We are given that f(x) is a first-degree function, i.e., f(x) = a * x + b

def is_first_degree (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x : ℝ, f(x) = a * x + b

-- Given conditions
variables {f : ℝ → ℝ} (hf : is_first_degree f)
variables (h1 : 2 * f 2 - 3 * f 1 = 5)
variables (h2 : 2 * f 0 - f (-1) = 1)

-- Goal: f(x) = 3x - 2
theorem first_degree_function_solution :
  is_first_degree f ∧ (2 * f 2 - 3 * f 1 = 5) ∧ (2 * f 0 - f (-1) = 1) → ∀ x : ℝ, f(x) = 3 * x - 2 := 
sorry

end first_degree_function_solution_l787_787633


namespace range_of_function_l787_787096

theorem range_of_function : 
  set.range (λ x : ℝ, 2 * (Real.sin x) ^ 2 * (Real.cos x)) = 
    (set.Icc (-(4 / 9) * Real.sqrt 3) ((4 / 9) * Real.sqrt 3)) :=
sorry

end range_of_function_l787_787096


namespace wechat_group_red_envelopes_l787_787314

theorem wechat_group_red_envelopes :
  let members := ['A', 'B', 'C', 'D', 'E'] in
  let red_envelopes := [2, 2, 3, 3] in
  let grab_one_each (persons : List Char) (envelopes : List ℕ) : Prop :=
    (persons.length > 0) → (persons.length ≤ envelopes.length) → (envelopes.length > 0) → Prop in
  let conditions := (members.length = 5) ∧ (red_envelopes.length = 4) ∧
                    (all_persons_grab (members, red_envelopes)) in
  ∃ (count : ℕ), conditions ∧ count = 18 :=
sorry

end wechat_group_red_envelopes_l787_787314


namespace surface_area_L_shaped_l787_787952

theorem surface_area_L_shaped {n m : ℕ} (h1 : n = 10) (h2 : m = 5) :
  let base_layer := n in
  let second_layer := m in
  let total_surface_area := 45 in
  total_surface_area = 45 :=
by
  -- Define base layer properties
  have base_visible_top := 6,
  have base_visible_front := 10,
  have base_visible_back := 10,
  have base_visible_ends := 2,

  -- Define second layer properties
  have second_visible_top := 5,
  have second_visible_front := 5,
  have second_visible_back := 5,
  have second_visible_ends := 2,
  
  -- Calculate total surface area
  have total_visible_faces := base_visible_top + base_visible_front + base_visible_back + base_visible_ends +
                              second_visible_top + second_visible_front + second_visible_back + second_visible_ends,

  -- Prove the total surface area is correct
  show total_visible_faces = 45,
  sorry

end surface_area_L_shaped_l787_787952


namespace beans_in_jar_l787_787799

theorem beans_in_jar (B : ℕ) 
  (h1 : B / 4 = number_of_red_beans)
  (h2 : number_of_red_beans = B / 4)
  (h3 : number_of_white_beans = (B * 3 / 4) / 3)
  (h4 : number_of_white_beans = B / 4)
  (h5 : number_of_remaining_beans_after_white = B / 2)
  (h6 : 143 = B / 4):
  B = 572 :=
by
  sorry

end beans_in_jar_l787_787799


namespace sum_of_squares_of_consecutive_integers_divisible_by_5_l787_787937

theorem sum_of_squares_of_consecutive_integers_divisible_by_5 (n : ℤ) :
  (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) % 5 = 0 :=
by
  sorry

end sum_of_squares_of_consecutive_integers_divisible_by_5_l787_787937


namespace symmetric_point_x_axis_l787_787076

/-- Given point P has coordinates (-3, 2) and point P' is symmetric to P with respect to the x-axis. 
    We prove that the coordinates of point P' are (-3, -2). -/
theorem symmetric_point_x_axis :
  ∃ P' : ℝ × ℝ, let P := (-3 : ℝ, 2 : ℝ) in P' = (-3, -2) ∧ P'.1 = P.1 ∧ P'.2 = -P.2 :=
by
  sorry

end symmetric_point_x_axis_l787_787076


namespace problem_A_eq_B_l787_787731

theorem problem_A_eq_B
  (a : ℕ) (a_pos : 0 < a) (not_square : ∀ (m : ℕ), a ≠ m * m) :
  let A := {k : ℕ | ∃ x y : ℤ, x > ⌊real.sqrt (a : ℝ)⌋ ∧ k = (x^2 - a) / (x^2 - y^2)} in
  let B := {k : ℕ | ∃ x y : ℤ, 0 ≤ x ∧ x < ⌊real.sqrt (a : ℝ)⌋ ∧ k = (x^2 - a) / (x^2 - y^2)} in
  A = B := by
  sorry

end problem_A_eq_B_l787_787731


namespace tylers_age_l787_787846

theorem tylers_age (B T : ℕ) 
  (h1 : T = B - 3) 
  (h2 : T + B = 11) : 
  T = 4 :=
sorry

end tylers_age_l787_787846


namespace find_n_l787_787656

noncomputable def a : ℝ := Real.log 2020 / Real.log 2019
noncomputable def b : ℝ := Real.log 2019 / Real.log 2020

def f (x : ℝ) : ℝ := a + x - b^x

theorem find_n : ∃ (n : ℤ), n = -1 ∧ ∃ (x0 : ℝ), x0 ∈ (n : ℝ), x0 ∈ Ioo n (n + 1) ∧ f x0 = 0 :=
by
  sorry

end find_n_l787_787656


namespace train_length_in_meters_l787_787142

theorem train_length_in_meters (speed_kmph : ℝ) (time_seconds : ℝ) 
  (h1 : speed_kmph = 18) (h2 : time_seconds = 5) : 
  let speed_mps := (speed_kmph * 1000) / 3600 in
  let length_of_train := speed_mps * time_seconds in
  length_of_train = 25 :=
by
  unfold speed_mps
  unfold length_of_train
  calc
    (18 * 1000 / 3600) * 5 = (18000 / 3600) * 5 : by sorry
                         ... = 5 * 5 : by sorry
                         ... = 25 : by sorry

end train_length_in_meters_l787_787142


namespace lunas_phone_bill_percentage_l787_787370

variables (H F P : ℝ)

theorem lunas_phone_bill_percentage :
  F = 0.60 * H ∧ H + F = 240 ∧ H + F + P = 249 →
  (P / F) * 100 = 10 :=
by
  intros
  sorry

end lunas_phone_bill_percentage_l787_787370


namespace apples_distribution_l787_787872

theorem apples_distribution (total_apples : ℕ) (rotten_apples : ℕ) (boxes : ℕ) (remaining_apples : ℕ) (apples_per_box : ℕ) :
  total_apples = 40 →
  rotten_apples = 4 →
  boxes = 4 →
  remaining_apples = total_apples - rotten_apples →
  apples_per_box = remaining_apples / boxes →
  apples_per_box = 9 :=
by
  intros
  sorry

end apples_distribution_l787_787872


namespace reduced_cost_per_meter_l787_787521

theorem reduced_cost_per_meter (original_cost total_cost new_length original_length : ℝ) :
  original_cost = total_cost / original_length →
  new_length = original_length + 4 →
  total_cost = total_cost →
  original_cost - (total_cost / new_length) = 1 :=
by sorry

end reduced_cost_per_meter_l787_787521


namespace temperature_on_tuesday_l787_787806

theorem temperature_on_tuesday (T W Th : ℝ) 
  (h1 : (T + W + Th) / 3 = 52) 
  (h2 : (W + Th + 53) / 3 = 54) : 
  T = 47 := 
by
  -- We declare the conditions.
  have eq1 : T + W + Th = 156 := by linarith [h1]
  have eq2 : W + Th + 53 = 162 := by linarith [h2]
  
  -- We then derive the equation by eliminating variables.
  have sub_eq : eq2 - eq1 
  have T_eq : T = 47 := by linarith [sub_eq]
  
  exact T_eq
  sorry

end temperature_on_tuesday_l787_787806


namespace new_person_weight_l787_787415

def average_weight_increase (w1 w2 w_new : ℝ) (increase : ℝ) : Prop :=
  (w1 + w2) / 2 + increase = (w1 + w_new) / 2

theorem new_person_weight (w1 w2 increase w_replace w_new : ℝ)
  (h1 : w_replace = 65)
  (h2 : increase = 4.5)
  (h3 : average_weight_increase w1 w_replace w_new increase) :
  w_new = 74 :=
by
  unfold average_weight_increase at h3
  simp only [h1, h2] at h3
  linarith

end new_person_weight_l787_787415


namespace sphere_surface_area_l787_787546

theorem sphere_surface_area
  (a b c : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) :
  ∃ (A : ℝ), A = 14 * real.pi := 
by
  sorry

end sphere_surface_area_l787_787546


namespace range_of_a_l787_787009

noncomputable def condition_p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
noncomputable def condition_q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) :
  (¬(∀ x, condition_p x)) → (¬(∀ x, condition_q x a)) → 
  (∀ x, condition_p x ↔ condition_q x a) → (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end range_of_a_l787_787009


namespace zero_point_in_interval_l787_787836

noncomputable def f (x a : ℝ) := 2^x - 2/x - a

theorem zero_point_in_interval (a : ℝ) : 
  (∃ x, 1 < x ∧ x < 2 ∧ f x a = 0) → 0 < a ∧ a < 3 :=
by
  sorry

end zero_point_in_interval_l787_787836


namespace certain_event_count_l787_787548

-- Define the events based on the conditions
def event1 : Prop := ¬(pureWaterFreezesAt20CelsiusUnderStandardAtmosphericPressure)
def event2 : Prop := ¬(scoresOverMaximumIn100PointExam)
def event3 : Prop := coinTossLandsBottomSideUp
def event4 : Prop := rectAreaLengthTimesWidth (a b : ℝ)

-- Event definitions (mock definitions to match conditions)
def pureWaterFreezesAt20CelsiusUnderStandardAtmosphericPressure : Prop := False
def scoresOverMaximumIn100PointExam : Prop := False
def coinTossLandsBottomSideUp : Prop := True -- Random event, tagged True for simplicity
def rectAreaLengthTimesWidth (a b : ℝ) : Prop := True

-- Statement to prove exactly 1 certain event
theorem certain_event_count : (∃! e, e = event4) ∧ ¬event1 ∧ ¬event2 ∧ ¬event3 ∧ event4 :=
by {
  sorry
}

end certain_event_count_l787_787548


namespace range_of_3_plus_cos_l787_787854

theorem range_of_3_plus_cos :
  (∀ x : ℝ, 2 ≤ 3 + cos x ∧ 3 + cos x ≤ 4) :=
by
  sorry

end range_of_3_plus_cos_l787_787854


namespace range_cosA_minus_sinC_l787_787706

noncomputable def acute_triangle := 
  {A B C : ℝ // 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧
   A + B + C = π}

noncomputable def sides_correspond_to_angles (a b c A B C : ℝ) := 
  a = 2 * b * Real.sin A

theorem range_cosA_minus_sinC 
  {A B C : ℝ} {a b c : ℝ} 
  (h₁ : acute_triangle A B C)
  (h₂ : sides_correspond_to_angles a b c A B C) : 
  ∃ (x y : ℝ), x = - (Real.sqrt 3) / 2 ∧ y = 1 / 2 ∧ 
  ∀ z, z = Real.cos A - Real.sin C → x < z ∧ z < y :=
sorry

end range_cosA_minus_sinC_l787_787706


namespace height_of_son_l787_787513

theorem height_of_son (
  height_lamp_post : ℝ := 6,
  height_father : ℝ := 1.8,
  distance_father_to_post : ℝ := 2.1,
  distance_son_to_father : ℝ := 0.45
) : ∃ h : ℝ, h ≈ 1.41 ∧ (
  let s := distance_father_to_post - distance_son_to_father in
  let ratio_father := height_father / s in
  let ratio_son := h / s in
  ratio_father = ratio_son
) := by
  sorry

end height_of_son_l787_787513


namespace euclid_1976_part_a_problem_4_l787_787066

theorem euclid_1976_part_a_problem_4
  (p q y1 y2 : ℝ)
  (h1 : y1 = p * 1^2 + q * 1 + 5)
  (h2 : y2 = p * (-1)^2 + q * (-1) + 5)
  (h3 : y1 + y2 = 14) :
  p = 2 :=
by
  sorry

end euclid_1976_part_a_problem_4_l787_787066


namespace cube_beautiful_faces_l787_787339

theorem cube_beautiful_faces (numbers : Finset ℕ) (isBeautifulFace : (ℕ × ℕ × ℕ × ℕ) → Prop) :
  (∀ (a b c d : ℕ), isBeautifulFace (a, b, c, d) ↔ a = b + c + d ∨ b = a + c + d ∨ c = a + b + d ∨ d = a + b + c) →
  6 ∈ numbers →
  numbers = {1, 2, 3, 4, 5, 6, 7, 8} →
  ∃ (adj1 adj2 adj3 : ℕ), {adj1, adj2, adj3} ∈ {{2, 3, 5}, {3, 5, 7}, {2, 3, 7}} :=
by sorry

end cube_beautiful_faces_l787_787339


namespace correct_value_of_wrongly_read_number_l787_787074

theorem correct_value_of_wrongly_read_number 
  (avg_wrong : ℝ) (n : ℕ) (wrong_value : ℝ) (avg_correct : ℝ) :
  avg_wrong = 5 →
  n = 10 →
  wrong_value = 26 →
  avg_correct = 6 →
  let sum_wrong := avg_wrong * n
  let correct_sum := avg_correct * n
  let difference := correct_sum - sum_wrong
  let correct_value := wrong_value + difference
  correct_value = 36 :=
by
  intros h_avg_wrong h_n h_wrong_value h_avg_correct
  let sum_wrong := avg_wrong * n
  let correct_sum := avg_correct * n
  let difference := correct_sum - sum_wrong
  let correct_value := wrong_value + difference
  sorry

end correct_value_of_wrongly_read_number_l787_787074


namespace lcm_36_125_l787_787959

-- Define the prime factorizations
def factorization_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
def factorization_125 : List (ℕ × ℕ) := [(5, 3)]

-- Least common multiple definition
noncomputable def my_lcm (a b : ℕ) : ℕ :=
  a * b / (Nat.gcd a b)

-- Theorem to prove
theorem lcm_36_125 : my_lcm 36 125 = 4500 :=
by
  sorry

end lcm_36_125_l787_787959


namespace sum_of_g_54_l787_787361

def f (x : ℝ) : ℝ := 4 * x^2 + 6
def g (y : ℝ) : ℝ := y^2 - y + 2

theorem sum_of_g_54 : g(54) = 28 := sorry

end sum_of_g_54_l787_787361


namespace final_population_correct_l787_787157

noncomputable def initialPopulation : ℕ := 300000
noncomputable def immigration : ℕ := 50000
noncomputable def emigration : ℕ := 30000

noncomputable def populationAfterImmigration : ℕ := initialPopulation + immigration
noncomputable def populationAfterEmigration : ℕ := populationAfterImmigration - emigration

noncomputable def pregnancies : ℕ := populationAfterEmigration / 8
noncomputable def twinPregnancies : ℕ := pregnancies / 4
noncomputable def singlePregnancies : ℕ := pregnancies - twinPregnancies

noncomputable def totalBirths : ℕ := twinPregnancies * 2 + singlePregnancies
noncomputable def finalPopulation : ℕ := populationAfterEmigration + totalBirths

theorem final_population_correct : finalPopulation = 370000 :=
by
  sorry

end final_population_correct_l787_787157


namespace evaluate_expression_l787_787931

theorem evaluate_expression : (-7)^3 / 7^2 - 4^4 + 5^2 = -238 := by
  sorry

end evaluate_expression_l787_787931


namespace minimum_positive_period_of_f_l787_787822

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem minimum_positive_period_of_f : 
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧ 
  ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ Real.pi := 
sorry

end minimum_positive_period_of_f_l787_787822


namespace trigonometric_solution_l787_787789

theorem trigonometric_solution
    (x y z : ℝ)
    (h₀ : cos x ≠ 0)
    (h₁ : cos y ≠ 0)
    (n k m : ℤ)
    (hx : x = n * π)
    (hy : y = k * π)
    (hz : z = π / 2 + 2 * m * π) :
    (\(cos x)\^2 + 1 / (cos x)\^2)\^3 + (\(cos y)\^2 + 1 / (cos y)\^2)\^3 = 16 * sin z := by
  sorry

end trigonometric_solution_l787_787789


namespace proof_problem_l787_787391

noncomputable def alpha : ℝ := sorry
noncomputable def beta : ℝ := sorry

def problem_conditions : Prop :=
  0 ≤ alpha ∧ alpha ≤ π / 2 ∧
  0 ≤ beta ∧ beta ≤ π / 2 ∧
  Real.tan alpha = 5 ∧
  Real.cot beta = 2 / 3

theorem proof_problem (h : problem_conditions) : alpha + beta = 3 * π / 4 := sorry

end proof_problem_l787_787391


namespace fixed_point_range_l787_787968

theorem fixed_point_range (a : ℝ) : (∃ x : ℝ, x = x^2 + x + a) → a ≤ 0 :=
sorry

end fixed_point_range_l787_787968


namespace point_line_distance_l787_787659

theorem point_line_distance (a : ℝ) (h : a > 0) :
    dist (a, 2) {p | p.1 - p.2 + 3 = 0} = 1 → a = Real.sqrt 2 - 1 :=
by
  intro hDist
  have hDistFormula : abs (a - 2 + 3) / Real.sqrt (1^2 + (-1)^2) = 1,
    from hDist
  simp [pow_two, add_eq_zero_iff, mul_self_sqrt (show 0 ≤ 2 from by norm_num)] at hDistFormula
  have hAbs : abs (a + 1) = Real.sqrt 2, from (eq_pos_of_mul_eq_one hDistFormula)
  rw abs_eq_iff at hAbs
  cases hAbs 
  case inl h1 =>
    linarith
  case inr h2 =>
    linarith

end point_line_distance_l787_787659


namespace train_speed_is_correct_l787_787913

-- Let us define the conditions in Lean
def train_length : ℝ := 300
def bridge_length : ℝ := 300
def crossing_time : ℝ := 45

-- The total distance to be covered
def total_distance : ℝ := train_length + bridge_length

-- The speed of the train computed
def train_speed : ℝ := total_distance / crossing_time

-- The theorem to prove that the speed of the train is 13.33 m/s
theorem train_speed_is_correct : train_speed = 13.33 :=
sorry

end train_speed_is_correct_l787_787913


namespace fraction_meaningful_l787_787099

-- Define the condition for the fraction being meaningful
def denominator_not_zero (x : ℝ) : Prop := x + 1 ≠ 0

-- Define the statement to be proved
theorem fraction_meaningful (x : ℝ) : denominator_not_zero x ↔ x ≠ -1 :=
by
  sorry

end fraction_meaningful_l787_787099


namespace cost_of_siding_l787_787054

-- Definitions as per the conditions
def side_wall_area : Nat := 8 * 6
def total_side_walls_area : Nat := 2 * side_wall_area
def roof_face_area : Nat := 8 * 5
def total_roof_faces_area : Nat := 2 * roof_face_area
def total_area_to_cover : Nat := total_side_walls_area + total_roof_faces_area
def siding_section_area : Nat := 10 * 15
def cost_per_section : Nat := 35

-- The problem statement as a Lean theorem
theorem cost_of_siding : 
  let num_sections := Nat.ceil_div total_area_to_cover siding_section_area in
  let total_cost := num_sections * cost_per_section in 
  total_cost = 70 :=
by
  sorry

end cost_of_siding_l787_787054


namespace pounds_over_weight_limit_l787_787343

-- Definitions based on conditions

def bookcase_max_weight : ℝ := 80

def weight_hardcover_book : ℝ := 0.5
def number_hardcover_books : ℕ := 70
def total_weight_hardcover_books : ℝ := number_hardcover_books * weight_hardcover_book

def weight_textbook : ℝ := 2
def number_textbooks : ℕ := 30
def total_weight_textbooks : ℝ := number_textbooks * weight_textbook

def weight_knick_knack : ℝ := 6
def number_knick_knacks : ℕ := 3
def total_weight_knick_knacks : ℝ := number_knick_knacks * weight_knick_knack

def total_weight_items : ℝ := total_weight_hardcover_books + total_weight_textbooks + total_weight_knick_knacks

theorem pounds_over_weight_limit : total_weight_items - bookcase_max_weight = 33 := by
  sorry

end pounds_over_weight_limit_l787_787343


namespace find_general_term_l787_787609

variable (a : ℕ → ℚ)

-- Define the sequence
def seq_def (an_rec : (ℕ → ℚ) → ℕ → ℚ) (a : ℕ → ℚ) :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = an_rec a n

-- The recurrence relation
def an_rec (a : ℕ → ℚ) (n : ℕ) : ℚ := (2 * a n) / (2 + a n)

-- The general term to be proven
def general_term (a : ℕ → ℚ) (n : ℕ) : ℚ := 2 / (n + 1)

theorem find_general_term :
  seq_def an_rec a →
  ∀ n : ℕ, n > 0 → a n = general_term a n := 
sorry

end find_general_term_l787_787609


namespace molecular_weight_of_compound_l787_787601

-- Define the atomic weights for Hydrogen, Chlorine, and Oxygen
def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cl : ℝ := 35.45
def atomic_weight_O : ℝ := 15.999

-- Define the molecular weight of the compound
def molecular_weight (H_weight : ℝ) (Cl_weight : ℝ) (O_weight : ℝ) : ℝ :=
  H_weight + Cl_weight + 2 * O_weight

-- The proof problem statement
theorem molecular_weight_of_compound :
  molecular_weight atomic_weight_H atomic_weight_Cl atomic_weight_O = 68.456 :=
sorry

end molecular_weight_of_compound_l787_787601


namespace complex_conversion_l787_787207

noncomputable def rectangular_form (z : ℂ) : ℂ := z

theorem complex_conversion :
  rectangular_form (complex.sqrt 2 * complex.exp (13 * real.pi * complex.I / 4)) = 1 + complex.I :=
by
  sorry

end complex_conversion_l787_787207


namespace total_albums_l787_787030

-- Definitions based on given conditions
def adele_albums : ℕ := 30
def bridget_albums : ℕ := adele_albums - 15
def katrina_albums : ℕ := 6 * bridget_albums
def miriam_albums : ℕ := 5 * katrina_albums

-- The final statement to be proved
theorem total_albums : adele_albums + bridget_albums + katrina_albums + miriam_albums = 585 :=
by
  sorry

end total_albums_l787_787030


namespace closest_to_fraction_l787_787591

theorem closest_to_fraction (options : List ℝ) (h1 : options = [2000, 1500, 200, 2500, 3000]) :
  ∃ closest : ℝ, closest ∈ options ∧ closest = 2000 :=
by
  sorry

end closest_to_fraction_l787_787591


namespace find_length_of_AB_l787_787774

theorem find_length_of_AB (x y : ℝ) (AP PB AQ QB PQ AB : ℝ) 
  (h1 : AP = 3 * x) 
  (h2 : PB = 4 * x) 
  (h3 : AQ = 4 * y) 
  (h4 : QB = 5 * y)
  (h5 : PQ = 5) 
  (h6 : AP + PB = AB)
  (h7 : AQ + QB = AB)
  (h8 : PQ = AQ - AP)
  (h9 : 7 * x = 9 * y) : 
  AB = 315 := 
by
  sorry

end find_length_of_AB_l787_787774


namespace cos_product_l787_787251

theorem cos_product (x y : ℝ) (h1 : sin x + sin y = 0.6) (h2 : cos x + cos y = 0.8) : 
  cos x * cos y = -11 / 100 := 
by 
  sorry

end cos_product_l787_787251


namespace cauchy_schwarz_inequality_for_sequences_cauchy_schwarz_inequality_equality_condition_l787_787016

theorem cauchy_schwarz_inequality_for_sequences 
  (n : ℕ) 
  (hn : 0 < n) 
  (e : Fin n → ℝ) 
  (f : Fin n → ℝ) 
  (hf : ∀ i, 0 < f i) : 
  (∑ i, (e i)^2 / (f i)) ≥ (∑ i, e i)^2 / (∑ i, f i) :=
sorry

-- Condition for equality
theorem cauchy_schwarz_inequality_equality_condition 
  (n : ℕ) 
  (hn : 0 < n) 
  (e : Fin n → ℝ) 
  (f : Fin n → ℝ) 
  (hf : ∀ i, 0 < f i) : 
  ((∑ i, (e i)^2 / (f i)) = (∑ i, e i)^2 / (∑ i, f i)) ↔ 
  ∃ (λ : ℝ), ∀ i, e i = λ * f i :=
sorry

end cauchy_schwarz_inequality_for_sequences_cauchy_schwarz_inequality_equality_condition_l787_787016


namespace range_of_k_l787_787578

theorem range_of_k (H : ∀ (n : ℕ), H_n = (b_1 + 2 * b_2 + ∙∙∙ + 2^(n - 1) * b_n) / n)
  (H_avg : ∀ (n : ℕ), H n = 2^(n + 1))
  (Sn_le_S5 : ∀ (n : ℕ), S_n ≤ S_5) : 
  (7 / 3 : ℝ) ≤ k ∧ k ≤ (12 / 5 : ℝ) :=
begin
  sorry
end

end range_of_k_l787_787578


namespace travel_speed_l787_787898

theorem travel_speed (distance : ℕ) (time : ℕ) (h_distance : distance = 160) (h_time : time = 8) :
  ∃ speed : ℕ, speed = distance / time ∧ speed = 20 :=
by
  sorry

end travel_speed_l787_787898


namespace distance_l1_l2_l787_787290

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  |C2 - C1| / Real.sqrt (A^2 + B^2)

theorem distance_l1_l2 :
  distance_between_parallel_lines 3 4 (-3) 2 = 1 :=
by
  -- Add the conditions needed to assert the theorem
  let l1 := (3, 4, -3) -- definition of line l1
  let l2 := (3, 4, 2)  -- definition of line l2
  -- Calculate the distance using the given formula
  let d := distance_between_parallel_lines 3 4 (-3) 2
  -- Assert the result
  show d = 1
  sorry

end distance_l1_l2_l787_787290


namespace fish_too_small_l787_787194

theorem fish_too_small
    (ben_fish : ℕ) (judy_fish : ℕ) (billy_fish : ℕ) (jim_fish : ℕ) (susie_fish : ℕ)
    (total_filets : ℕ) (filets_per_fish : ℕ) :
    ben_fish = 4 →
    judy_fish = 1 →
    billy_fish = 3 →
    jim_fish = 2 →
    susie_fish = 5 →
    total_filets = 24 →
    filets_per_fish = 2 →
    (ben_fish + judy_fish + billy_fish + jim_fish + susie_fish) - (total_filets / filets_per_fish) = 3 := 
by 
  intros
  sorry

end fish_too_small_l787_787194


namespace chess_club_team_selection_l787_787416

theorem chess_club_team_selection :
  (Nat.choose 8 5) * (Nat.choose 10 3) = 6720 :=
by
  sorry

end chess_club_team_selection_l787_787416


namespace intersection_of_lines_on_circle_l787_787263

open Function Real Set

noncomputable theory

variables {A B C K I X H O F : Type*}
variables [geometry A B C K I X H O F]

-- Definitions based on given conditions
def circumcircle (ABC : Set A B C K I X H O F) := sorry -- circumcircle definition placeholder
def orthocenter (ABC : Set A B C K I X H O F) := sorry -- orthocenter definition placeholder
def incenter (ABC : Set A B C K I X H O F) := sorry -- incenter definition placeholder
def midpoint (arc : Set A B C K I X H O F) := sorry -- midpoint of an arc placeholder
def arc_contains (H F : Set A B C K I X H O F) := sorry -- arc containment condition placeholder

-- Given conditions
variables (circumcircleABC : circumcircle A B C)
variables (orthocenterH : orthocenter A B C)
variables (incenterI : incenter A B C)
variables (midpointF : midpoint B C)
variables (arcContainsCondition : arc_contains H F)
variables (AXH_eq_AFH : ∀ X, ∠(A X H) = ∠(A F H))

-- Question: Prove that the intersection of lines AO and KI lies on the circumcircle
theorem intersection_of_lines_on_circle (AO KI : Type*)
  (intersection_pt : intersection AO KI)
  (on_circ : lies_on_circ intersection_pt circumcircleABC) : True :=
sorry

end intersection_of_lines_on_circle_l787_787263


namespace volume_midpoint_trajectory_l787_787624

-- Definitions for the tetrahedron
structure Tetrahedron (A B C O : Point) :=
(perpendicular_OA_OB : A.x = 0 ∧ A.y = 0 ∧ A.z != 0)
(perpendicular_OB_OC : B.x = 0 ∧ B.z = 0 ∧ B.y != 0)
(perpendicular_OA_OC : C.y = 0 ∧ C.z = 0 ∧ C.x != 0)
(OA_length : dist O A = 6)
(OB_length : dist O B = 6)
(OC_length : dist O C = 6)

-- Definitions for the moving line segment MN
structure MoveSegment (M N O : Point) (MN_length : ℝ) :=
(M_on_OA : M ∈ line_seg O A)
(N_in_BCO : N ∈ triangle B C O)
(MN_length2 : MN_length = 2)

-- The volume of the geometric body formed by the trajectory of the midpoint R
theorem volume_midpoint_trajectory 
  (A B C O M N : Point) 
  (tetra : Tetrahedron A B C O)
  (mn_seg : MoveSegment M N O 2) 
  (R : Point) 
  (midpoint_R : R = midpoint M N) :
  volume_trajectory_midpoint_R R O (Sphere O 1) (Tetrahedron A B C O) = 36 - pi / 6 ∨ π / 6 := sorry

end volume_midpoint_trajectory_l787_787624


namespace second_term_of_arithmetic_sequence_l787_787694

-- Define the statement of the problem
theorem second_term_of_arithmetic_sequence 
  (a d : ℝ) 
  (h : a + (a + 2 * d) = 10) : 
  a + d = 5 := 
by 
  sorry

end second_term_of_arithmetic_sequence_l787_787694


namespace find_parabola_and_line_eq_l787_787272

-- Definitions and conditions
def parabola_vertex_at_origin := (0, 0)
def parabola_focus_at := (1 / 2, 0)
def circle_eq (x y : ℝ) : Prop := (x - 3 / 2)^2 + (y - 8)^2 = 49
def distance_eq (a b c d : ℝ) : Prop := abs (a - b) = abs (c - d)

-- The Lean statement
theorem find_parabola_and_line_eq
    (A B F : ℝ × ℝ) (l : ℝ → ℝ × ℝ) :
    F = parabola_focus_at →
    (∃ A B : ℝ × ℝ, parabola_vertex_at_origin.1 = 0 ∧ parabola_focus_at.1 = 1 / 2 ∧ A ≠ B) →
    (∃ D E : ℝ × ℝ, circle_eq D.1 D.2 ∧ circle_eq E.1 E.2 ∧ D = E ∧ distance_eq A.1 B.1 D.1 E.1) →
    (∀ t, l t = (t^2 + 1 / 2, t)) →
    (∀ x y, (x, y) ∈ (λ t, parabola_focus_at t)) →
    (∃ l : ℝ → ℝ, (∀ x y, l x y = 2 * x - 4 * y - 1)) :=
by {
  -- Proof steps can be filled in here
  sorry
}

end find_parabola_and_line_eq_l787_787272


namespace victor_score_l787_787467

-- Definitions based on the conditions
def max_marks : ℕ := 300
def percentage : ℕ := 80

-- Statement to be proved
theorem victor_score : (percentage * max_marks) / 100 = 240 := by
  sorry

end victor_score_l787_787467


namespace cone_generatrix_length_theorem_l787_787303

noncomputable def cone_generatrix_length 
  (diameter : ℝ) 
  (unfolded_side_area : ℝ) 
  (h_diameter : diameter = 6)
  (h_area : unfolded_side_area = 18 * Real.pi) : 
  ℝ :=
6

theorem cone_generatrix_length_theorem 
  (diameter : ℝ) 
  (unfolded_side_area : ℝ) 
  (h_diameter : diameter = 6)
  (h_area : unfolded_side_area = 18 * Real.pi) :
  cone_generatrix_length diameter unfolded_side_area h_diameter h_area = 6 :=
sorry

end cone_generatrix_length_theorem_l787_787303


namespace quadratic_solution_l787_787064

theorem quadratic_solution (x : ℝ) : x^2 - 5 * x - 6 = 0 ↔ (x = 6 ∨ x = -1) :=
by
  sorry

end quadratic_solution_l787_787064


namespace unique_T_n_from_S_2019_l787_787238

-- Define the geometric sequence, sums of terms, and sums of sums
def geometric_sequence (g : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, g (n + 1) = g n * r

def S_n (g : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finRange (n + 1), g i

def T_n (g : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finRange (n + 1), S_n g i

-- Define the known value
def S_2019_val (g : ℕ → ℝ) (val : ℝ) : Prop :=
S_n g 2019 = val

-- The main statement
theorem unique_T_n_from_S_2019 (g : ℕ → ℝ) (r : ℝ) (val : ℝ) :
  geometric_sequence g r →
  S_2019_val g val →
  ∃ n : ℕ, T_n g n = T_n g 2019 :=
by
  sorry

end unique_T_n_from_S_2019_l787_787238


namespace march_1_is_monday_l787_787688

theorem march_1_is_monday (March_8_is_Monday : nat → bool) :
  (March_8_is_Monday 8 = true) → (March_8_is_Monday 1 = true) :=
by
  sorry

end march_1_is_monday_l787_787688


namespace exponential_fraction_l787_787477

theorem exponential_fraction :
  (2^2014 + 2^2012) / (2^2014 - 2^2012) = 5 / 3 := 
by
  sorry

end exponential_fraction_l787_787477


namespace chess_match_schedule_count_l787_787934

-- The conditions
variable {School : Type*} [Fintype School] (players_school : Fintype.card School = 4)
variable {Players : Type*} [Fintype Players] (players : Fintype.card Players = 8)

open Fintype

noncomputable def num_ways_to_schedule_chess_match (School : Type*) [Fintype School] (players_school : Fintype.card School = 4) : ℕ :=
  factorial 8

-- The theorem
theorem chess_match_schedule_count : 
  num_ways_to_schedule_chess_match School 4 = 40320 := by
  sorry

end chess_match_schedule_count_l787_787934


namespace stones_symmetric_exchange_exists_l787_787445
open Finset

theorem stones_symmetric_exchange_exists (colors : Fin 13 → Bool) :
  ∃ i j : Fin 13, i ≠ j ∧
  let colors' := (fun k => if k = i then colors j else if k = j then colors i else colors k) in
  ∃ sym_ax : (Fin 13), 
  ∀ k, colors' k = colors' (sym_ax - k) :=
sorry

end stones_symmetric_exchange_exists_l787_787445


namespace max_chord_length_l787_787279

noncomputable def ellipse : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), (p = (x, y) ∧ (x^2 / 4 + y^2 = 1))}

def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x : ℝ), p = (x, k * x + 1)}

theorem max_chord_length :
  ∃ k : ℝ, ∀ (P Q : ℝ × ℝ),
    P ∈ ellipse ∧ Q ∈ ellipse ∧ P ∈ line k ∧ Q ∈ line k →
    dist P Q ≤ 4 * sqrt 3 / 3 ∧ 
    (∃ P Q : ℝ × ℝ, P ∈ ellipse ∧ Q ∈ ellipse ∧ P ∈ line k ∧ Q ∈ line k ∧ dist P Q = 4 * sqrt 3 / 3) :=
sorry

end max_chord_length_l787_787279


namespace parallel_vectors_eq_l787_787292

theorem parallel_vectors_eq (x y : ℝ) (h : ∃ k : ℝ, (x, 4, 1) = k • (-2, y, -1)) :
  x = 2 ∧ y = -4 :=
by {
  cases h with k hk,
  ext,
  -- split the vector components
  let ⟨hx, hy, hz⟩ := hk,
  -- solve for each component
  sorry
}

end parallel_vectors_eq_l787_787292


namespace price_of_kid_ticket_l787_787784

theorem price_of_kid_ticket (k a : ℤ) (hk : k = 6) (ha : a = 2)
  (price_kid price_adult : ℤ)
  (hprice_adult : price_adult = 2 * price_kid)
  (hcost_total : 6 * price_kid + 2 * price_adult = 50) :
  price_kid = 5 :=
by
  sorry

end price_of_kid_ticket_l787_787784


namespace iso_triangle_angles_straight_line_l787_787849

theorem iso_triangle_angles_straight_line 
  (A_1 B_1 C_1 A_2 B_2 C_2 : Type)
  (e : Type)
  (isosceles : A_1B_1 = A_1C_1)
  (equilaterals : (∃ (d1 : _) (d2 : _) (d3 : _), 
      is_equilateral_triangle A_1 B_1 d1 ∧ 
      is_equilateral_triangle B_1 C_1 d2 ∧ 
      is_equilateral_triangle C_1 A_1 d3 ∧ 
      vertices_on_line A_2 B_2 C_2 e))
  : angles_of_isosceles_triangle A_1 B_1 C_1 = ([21.8, 79.1] : List ℝ) :=
by
  -- proof goes here
  sorry

end iso_triangle_angles_straight_line_l787_787849


namespace max_value_PF1_PF2_l787_787423

theorem max_value_PF1_PF2 (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (a : ℝ) :
  (P.x^2 / 8 + P.y^2 = 1) →
  (F1 = (2 * sqrt 2, 0)) →
  (F2 = (-2 * sqrt 2, 0)) →
  a = 2 * sqrt 2 →
  |dist P F1 + dist P F2| = 4 * sqrt 2 →
  (∀ P, dist P F1 * dist P F2 ≤ 8) :=
by
  sorry

end max_value_PF1_PF2_l787_787423


namespace hyperbola_eccentricity_l787_787268

-- Define the conditions
variables (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
def hyperbola (x y : ℝ) := (x^2) / (a^2) - (y^2) / (b^2) = 1

-- Statement to prove the eccentricity
theorem hyperbola_eccentricity (hx : 1) (hy : a) :
  -- Condition of the asymptote passing through the given point
  (2 * b) / a = 2 * sqrt 3 →
  -- Prove the eccentricity is 2
  √(1 + 3) = 2 :=
by
  sorry

end hyperbola_eccentricity_l787_787268


namespace length_AE_l787_787721

theorem length_AE (A B C D E : Type) 
  (AB AC AD AE : ℝ) 
  (angle_BAC : ℝ)
  (h1 : AB = 4.5) 
  (h2 : AC = 5) 
  (h3 : angle_BAC = 30) 
  (h4 : AD = 1.5) 
  (h5 : AD / AB = AE / AC) : 
  AE = 1.6667 := 
sorry

end length_AE_l787_787721


namespace expected_matches_variance_matches_l787_787512

variable (N : ℕ)

/-- Expected number of matching pairs in a deck of N cards as described in the problem description. -/
theorem expected_matches : 
  let I := λ k, (1 : ℕ) in
  let S := ∑ k in Finset.range N, I k in
  (1 : ℝ) = 1 :=
by
  sorry

/-- Variance of the number of matching pairs in a deck of N cards as described in the problem description. -/
theorem variance_matches : 
  let I := λ k, (1 : ℕ) in
  let S := ∑ k in Finset.range N, I k in
  let variance := (𝔼[S^2] - (𝔼[S])^2 : ℝ) in
  variance = 1 :=
by
  sorry

end expected_matches_variance_matches_l787_787512


namespace weekly_sales_correct_target_profit_correct_max_profit_correct_l787_787453

-- Definitions from conditions
def cost_price : ℕ := 80
def init_selling_price : ℕ := 90
def init_sales_weekly : ℕ := 600
def price_increase_effect : ℕ := 10
def max_selling_price : ℕ := 110

-- Problem 1
def weekly_sales_function (x : ℕ) : ℕ := -10 * x + 1500

-- Problem 2
def target_profit : ℕ := 8250
def selling_price_for_target_profit (x : ℕ) : Prop :=
    let sales := weekly_sales_function x in
    (x - cost_price) * sales = target_profit

-- Problem 3
def profit_function (x : ℕ) : ℕ :=
    (x - cost_price) * weekly_sales_function x

-- Statement for maximized profit given max_selling_price constraint
def max_profit_price : ℕ := 110
def max_profit_value : ℕ := 12000

-- Equivalent proof statements
theorem weekly_sales_correct :
  ∀ x : ℕ, weekly_sales_function x = -10 * x + 1500 :=
sorry

theorem target_profit_correct :
  selling_price_for_target_profit 95 :=
sorry

theorem max_profit_correct :
  (profit_function max_profit_price = max_profit_value) ∧ 
  (∀ (x : ℕ), x ≤ max_selling_price → profit_function x ≤ max_profit_value) :=
sorry

end weekly_sales_correct_target_profit_correct_max_profit_correct_l787_787453


namespace lattice_points_count_l787_787669

theorem lattice_points_count (x y : ℤ) :
  (x^2 - y^2 = 77) → ∃ l : finset (ℤ × ℤ), l.card = 4 ∧ ∀ (a b : ℤ), ((a, b) ∈ l ↔ a^2 - b^2 = 77) :=
by
  -- We provide the existence of the finset and the proper cardinality (4)
  -- Sorry denotes the omission of the detailed steps
  sorry

end lattice_points_count_l787_787669


namespace max_area_of_rotating_lines_l787_787716

structure Point :=
(x : ℝ)
(y : ℝ)

def slope (p1 p2 : Point) : ℝ :=
(p2.y - p1.y) / (p2.x - p1.x)

def y_intercept (p : Point) (m : ℝ) : ℝ :=
p.y - m * p.x

def line (m : ℝ) (b : ℝ) : ℝ → ℝ :=
λ x, m * x + b

def vertical_line (x : ℝ) : ℝ → Point :=
λ y, ⟨x, y⟩

noncomputable def rotate (angle : ℝ) (p : Point) : Point :=
sorry -- Implementation of rotation around specific point is beyond this example.

noncomputable def intersection (ℓ₁ ℓ₂ : ℝ → ℝ) : Point :=
sorry -- Intersection calculation is beyond this example.

def area_of_triangle (p1 p2 p3 : Point) : ℝ :=
(1 / 2) * | p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y) |

theorem max_area_of_rotating_lines : 
  let A := Point.mk 0 0,
      B := Point.mk 8 0,
      C := Point.mk 15 0,
      ℓA := line 2 0,
      ℓB := vertical_line 8,
      ℓC := line (-2) (y_intercept (Point.mk 15 0) (-2))
  in ∃ time : ℝ, ∃ p1 p2 p3 : Point, 
      p1 = intersection ℓB ℓC ∧
      p2 = intersection ℓA ℓC ∧
      p3 = intersection ℓA ℓB ∧
      area_of_triangle p1 p2 p3 = 0.5 := 
sorry

end max_area_of_rotating_lines_l787_787716


namespace power_identity_l787_787295

theorem power_identity (x : ℝ) (h : (2^x - 4^x) + (2^(-x) - 4^(-x)) = 3) :
  (8^x + 3 * 2^x) + (8^(-x) + 3 * 2^(-x)) = -1 :=
by
  -- Placeholder for the proof
  sorry

end power_identity_l787_787295


namespace correct_operation_l787_787856

theorem correct_operation (a : ℝ) : 
  (a^3 * a^4 ≠ a^12) ∧
  (a^5 / a⁻³ ≠ a^2) ∧
  ((3 * a^4)^2 ≠ 6 * a^8) ∧
  ((-a)^5 * a = -a^6) :=
by
  -- insert detailed proof for each part here
  sorry

end correct_operation_l787_787856


namespace general_term_sequence_l787_787580

def mean_reciprocal (n : ℕ) (a : ℕ → ℕ) : ℚ := 
  n / (∑ i in finset.range n, a (i+1) : ℚ)

theorem general_term_sequence (a : ℕ → ℕ) :
  (∀ n, mean_reciprocal n a = 1 / (2 * n - 1)) →
  (∀ n, a n = 4 * n - 3) :=
sorry

end general_term_sequence_l787_787580


namespace hyperbola_eccentricity_l787_787424

theorem hyperbola_eccentricity (p : ℝ) (hp : p > 0) 
  (hfocus : -sqrt(3 + p^2 / 16) = -p / 2) :
  let a := sqrt(3 + p^2 / 16)
  let c := p / 2 
  ∃ e : ℝ, e = c / a ∧ e = 2 * sqrt 3 / 3 :=
by
  let a := sqrt (3 + p^2 / 16)
  let c := p / 2
  use c / a
  have h := by sorry
  rw [<-h]
  exact sorry

end hyperbola_eccentricity_l787_787424


namespace min_a4_in_arithmetic_sequence_l787_787710

noncomputable def arithmetic_sequence_min_a4 (a1 d : ℝ) 
(S4 : ℝ := 4 * a1 + 6 * d)
(S5 : ℝ := 5 * a1 + 10 * d)
(a4 : ℝ := a1 + 3 * d) : Prop :=
  S4 ≤ 4 ∧ S5 ≥ 15 → a4 = 7

theorem min_a4_in_arithmetic_sequence (a1 d : ℝ) (h1 : 4 * a1 + 6 * d ≤ 4) 
(h2 : 5 * a1 + 10 * d ≥ 15) : 
arithmetic_sequence_min_a4 a1 d := 
by {
  sorry -- Proof is omitted
}

end min_a4_in_arithmetic_sequence_l787_787710


namespace different_signs_abs_value_larger_l787_787684

variable {a b : ℝ}

theorem different_signs_abs_value_larger (h1 : a + b < 0) (h2 : ab < 0) : 
  (a > 0 ∧ b < 0 ∧ |a| < |b|) ∨ (a < 0 ∧ b > 0 ∧ |b| < |a|) :=
sorry

end different_signs_abs_value_larger_l787_787684


namespace elise_initial_money_l787_787949

theorem elise_initial_money :
  ∃ (X : ℤ), X + 13 - 2 - 18 = 1 ∧ X = 8 :=
by
  sorry

end elise_initial_money_l787_787949


namespace range_of_function_y_l787_787429

noncomputable def function_y (x : ℝ) := (sin x - cos x) / (2 - sin x * cos x)

theorem range_of_function_y : 
  set.range (function_y) = set.Icc (-2 * real.sqrt 2 / 5) (2 * real.sqrt 2 / 5) := 
by 
  sorry

end range_of_function_y_l787_787429


namespace decreasing_interval_of_logarithm_derived_function_l787_787189

theorem decreasing_interval_of_logarithm_derived_function :
  ∀ (x : ℝ), 1 < x → ∃ (f : ℝ → ℝ), (f x = x / (x - 1)) ∧ (∀ (h : x ≠ 1), deriv f x < 0) :=
by
  sorry

end decreasing_interval_of_logarithm_derived_function_l787_787189


namespace jelly_sold_l787_787894

theorem jelly_sold (G S R P : ℕ) (h1 : G = 2 * S) (h2 : R = 2 * P) (h3 : R = G / 3) (h4 : P = 6) : S = 18 := by
  sorry

end jelly_sold_l787_787894


namespace prove_f_cos_eq_l787_787685

variable (f : ℝ → ℝ)

theorem prove_f_cos_eq :
  (∀ x : ℝ, f (Real.sin x) = 3 - Real.cos (2 * x)) →
  (∀ x : ℝ, f (Real.cos x) = 3 + Real.cos (2 * x)) :=
by
  sorry

end prove_f_cos_eq_l787_787685


namespace triangle_dot_product_l787_787627

-- Define basic geometry constructs and conditions
def Point := ℝ × ℝ

-- Triangle ABC with given properties
variables (A B C D : Point)
variables (angle_BAC : ℝ) (AB AC BD DC : ℝ)

-- Conditions 
def is_triangle (A B C : Point) : Prop :=
  let (Ax, Ay) := A in
  let (Bx, By) := B in
  let (Cx, Cy) := C in
  let AB := real.sqrt ((Bx - Ax)^2 + (By - Ay)^2) in
  let AC := real.sqrt ((Cx - Ax)^2 + (Cy - Ay)^2) in
  AB = 2 ∧ AC = 1 ∧ angle_BAC = 2 * real.pi / 3

def point_D_on_BC (A B C D : Point) (BD DC : ℝ) : Prop :=
  let (Bx, By) := B in
  let (Cx, Cy) := C in
  let (Dx, Dy) := D in
  BD * 2 = DC ∧
    Dx = (2 * Cx + Bx) / 3 ∧
    Dy = (2 * Cy + By) / 3

-- Dot product of vectors
def dot_product (P Q R : Point) : ℝ :=
  let (Px, Py) := P in
  let (Qx, Qy) := Q in
  let (Rx, Ry) := R in
  (Qx - Px) * (Rx - Qx) + (Qy - Py) * (Ry - Qy)

-- Formal Proof Problem Statement
theorem triangle_dot_product :
  is_triangle A B C →
  point_D_on_BC A B C D BD DC →
  dot_product A D B C = -1 / 3 :=
by
  sorry

end triangle_dot_product_l787_787627


namespace find_sin_cos_value_l787_787631

theorem find_sin_cos_value (x : ℝ) (h₁ : cos x + 3 * sin x = 2) : 3 * sin x - cos x = 4 :=
sorry

end find_sin_cos_value_l787_787631


namespace set_remains_unchanged_paired_numbers_l787_787988

noncomputable def problem_set : Finset ℝ := 
  {a_1, a_2, ..., a_100}

theorem set_remains_unchanged (S : ℝ) (hS : S = problem_set.sum id) :
  ∀ a_i ∈ problem_set, (problem_set.erase a_i).insert (S - a_i) = problem_set :=
sorry

theorem paired_numbers (a_i : ℝ) (a_i_in_set : a_i ∈ problem_set) :
  ∃ a_j, a_j ≠ a_i ∧ a_j = -a_i :=
sorry

end set_remains_unchanged_paired_numbers_l787_787988


namespace perfect_squares_divisible_by_4_l787_787672

theorem perfect_squares_divisible_by_4 :
  {n : ℕ | (∃ k : ℕ, n = k * k) ∧ n < 100 ∧ n % 4 = 0}.to_finset.card = 4 :=
by sorry

end perfect_squares_divisible_by_4_l787_787672


namespace min_cubes_required_l787_787487

def volume_of_box (L W H : ℕ) : ℕ := L * W * H
def volume_of_cube (v_cube : ℕ) : ℕ := v_cube
def minimum_number_of_cubes (V_box V_cube : ℕ) : ℕ := V_box / V_cube

theorem min_cubes_required :
  minimum_number_of_cubes (volume_of_box 12 16 6) (volume_of_cube 3) = 384 :=
by sorry

end min_cubes_required_l787_787487


namespace marbles_total_is_260_l787_787217

/-- Define the number of marbles in each jar. -/
def jar1 : ℕ := 80
def jar2 : ℕ := 2 * jar1
def jar3 : ℕ := jar1 / 4

/-- The total number of marbles Courtney has. -/
def total_marbles : ℕ := jar1 + jar2 + jar3

/-- Proving that the total number of marbles is 260. -/
theorem marbles_total_is_260 : total_marbles = 260 := 
by
  sorry

end marbles_total_is_260_l787_787217


namespace find_unit_prices_minimize_total_cost_l787_787155

def unit_prices_ (x y : ℕ) :=
  x + 2 * y = 40 ∧ 2 * x + 3 * y = 70
  
theorem find_unit_prices (x y: ℕ) (h: unit_prices_ x y): x = 20 ∧ y = 10 := 
  sorry

def total_cost (m: ℕ) := 20 * m + 10 * (60 - m)

theorem minimize_total_cost (m : ℕ) (h1 : 60 ≥ m) (h2 : m ≥ 20) : 
  total_cost m = 800 → m = 20 :=
  sorry

end find_unit_prices_minimize_total_cost_l787_787155


namespace solve_trig_identity_l787_787795

theorem solve_trig_identity (x y z : ℝ) (h1 : cos x ≠ 0) (h2 : cos y ≠ 0)
  (h_eq : (cos x ^ 2 + 1 / (cos x ^ 2)) ^ 3 + (cos y ^ 2 + 1 / (cos y ^ 2)) ^ 3 = 16 * sin z) :
  ∃ (n k m : ℤ), x = n * π ∧ y = k * π ∧ z = π / 2 + 2 * m * π :=
by
  sorry

end solve_trig_identity_l787_787795


namespace non_shaded_rectangle_perimeter_l787_787071

-- Definitions based on conditions
def larger_rectangle_area : ℕ := 150
def shaded_rectangle_area : ℕ := 110
def larger_rectangle_dimensions : (ℕ × ℕ) := (15, 10)
def all_angles_right : Prop := true  -- This is a given condition but not used directly in calculations.

-- Perimeter of the non-shaded region
theorem non_shaded_rectangle_perimeter :
  let non_shaded_rectangle_area := larger_rectangle_area - shaded_rectangle_area in
  non_shaded_rectangle_area = 40 →
  ∃ (length width : ℕ), length * width = non_shaded_rectangle_area ∧
  let perimeter := 2 * (length + width) in
  perimeter = 26 :=
by
  sorry

end non_shaded_rectangle_perimeter_l787_787071


namespace mod_congruent_integers_l787_787676

theorem mod_congruent_integers (n : ℕ) : (∃ k : ℕ, n = 13 * k + 7 ∧ 7 + 13 * k < 2000) ↔ 
  (7 + 13 * 153 < 2000 ∧ ∀ m : ℕ, (0 ≤ m ∧ m ≤ 153) → n = 13 * m + 7) := by
  sorry

end mod_congruent_integers_l787_787676


namespace toothpicks_at_20th_stage_l787_787446

theorem toothpicks_at_20th_stage (n : ℕ) (a d : ℕ) (h_a : a = 3) (h_d : d = 3) :
  T n = a + d * (n - 1) → T 20 = 60 :=
by
  intro formula
  rw [h_a, h_d]
  sorry

end toothpicks_at_20th_stage_l787_787446


namespace arcsin_double_angle_identity_l787_787018

open Real

theorem arcsin_double_angle_identity (x θ : ℝ) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) (h₃ : arcsin x = θ) (h₄ : -π / 2 ≤ θ) (h₅ : θ ≤ -π / 4) :
    arcsin (2 * x * sqrt (1 - x^2)) = -(π + 2 * θ) := by
  sorry

end arcsin_double_angle_identity_l787_787018


namespace mean_of_solutions_l787_787584

noncomputable theory

def polynomial_mean : ℚ :=
  let solutions := {x : ℂ | x^3 - 2 * x^2 - 8 * x = 0} in
  (0 + 4 + -2 : ℂ) / 3

theorem mean_of_solutions :
  polynomial_mean = 2 / 3 :=
sorry

end mean_of_solutions_l787_787584


namespace black_white_ratio_l787_787964

-- Required constants for the radii and coloring
def radii := [1, 3, 5, 7, 9]
def colors := ["black", "white", "black", "white", "black"]

-- Areas calculation uses the formula π * r^2
def area (r : ℕ) : ℝ := Real.pi * (r ^ 2)

-- Function to calculate the area of rings between successive circles
def ring_area (outer : ℕ) (inner : ℕ) : ℝ := area outer - area inner

-- Defining the colors of the concentric circles
def color_state : ℕ → string
| 0 => "black"
| 1 => "white"
| 2 => "black"
| 3 => "white"
| 4 => "black"
| _ => "none"

-- Calculate total area of black and white regions
def total_black_area : ℝ :=
  area 1 + ring_area 5 3 + ring_area 9 7

def total_white_area : ℝ :=
  ring_area 3 1 + ring_area 7 5

-- Main theorem statement
theorem black_white_ratio :
  total_black_area / total_white_area = 49 / 32 :=
by
  sorry

end black_white_ratio_l787_787964


namespace sangeun_initial_money_l787_787778

theorem sangeun_initial_money :
  ∃ (X : ℝ), 
  ((X / 2 - 2000) / 2 - 2000 = 0) ∧ 
  X = 12000 :=
by sorry

end sangeun_initial_money_l787_787778


namespace num_good_bulbs_eq_6_l787_787158

def num_good_bulbs (total_bulbs : ℕ)  (selected_bulbs : ℕ)  (prob_lighted : ℝ) : ℕ :=
  let total_combinations := Nat.choose total_bulbs selected_bulbs
  let required_combinations := Nat.choose 4 3
  (total_combinations * (1 - prob_lighted) / required_combinations).toNat

theorem num_good_bulbs_eq_6 : num_good_bulbs 10 3 0.9666666666666667 = 6 := 
  by 
  sorry

end num_good_bulbs_eq_6_l787_787158


namespace expectation_value_l787_787170

noncomputable def C (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

def P_zero : ℚ := (C 5 2) / (C 7 2)
def P_one : ℚ := (C 5 1 * C 2 1) / (C 7 2)
def P_two : ℚ := (C 2 2) / (C 7 2)

def xi_expectation : ℚ := 0 * P_zero + 1 * P_one + 2 * P_two

theorem expectation_value : xi_expectation = 4 / 7 := 
by {
  sorry
}

end expectation_value_l787_787170


namespace coeff_x2_expansion_l787_787640

def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

theorem coeff_x2_expansion (n : ℕ) (h : 9 * binomial_coefficient n 2 = 54) : n = 4 :=
sorry

end coeff_x2_expansion_l787_787640


namespace min_value_P_l787_787825

noncomputable def is_permutation_of (s t : List ℕ) : Prop :=
s ~ t

theorem min_value_P :
  ∀ (p q r : List ℕ),
  p.length = 3 → q.length = 3 → r.length = 3 →
  is_permutation_of (p ++ q ++ r) [1, 2, 3, 4, 5, 6, 7, 8, 9] →
  ∃ (P : ℕ), P = p.product * q.product * r.product + p.product * q.product * r.product + p.product * q.product * r.product ∧ P = 214 :=
sorry

end min_value_P_l787_787825


namespace ratio_of_length_to_width_l787_787820

-- Definitions of conditions
def width := 5
def area := 75

-- Theorem statement proving the ratio is 3
theorem ratio_of_length_to_width {l : ℕ} (h1 : l * width = area) : l / width = 3 :=
by sorry

end ratio_of_length_to_width_l787_787820


namespace proof_problem_l787_787914

def add_and_process (a b c : ℝ) : ℝ :=
  let sum := a + b
  let rounded := (Real.floor (sum * 100) / 100)
  rounded - c

theorem proof_problem : add_and_process 123.4567 98.764 0.02 = 222.20 := 
by 
  sorry

end proof_problem_l787_787914


namespace triangle_is_isosceles_l787_787992

variable (a b c : ℝ)
variable (h : a^2 - b * c = a * (b - c))

theorem triangle_is_isosceles (a b c : ℝ) (h : a^2 - b * c = a * (b - c)) : a = b ∨ b = c ∨ c = a := by
  sorry

end triangle_is_isosceles_l787_787992


namespace arithmetic_sequence_general_term_l787_787994

theorem arithmetic_sequence_general_term
  (d : ℕ) (a : ℕ → ℕ)
  (ha4 : a 4 = 14)
  (hd : d = 3) :
  ∃ a₁, ∀ n, a n = a₁ + (n - 1) * d := by
  sorry

end arithmetic_sequence_general_term_l787_787994


namespace part_I_part_II_l787_787647

open Real

noncomputable def A := {x : ℝ | -1 < x ∧ x ≤ 5}
noncomputable def B (a : ℝ) := {x : ℝ | x < -2 ∨ x > 4 | x^2 - 2*x + a > 0}
noncomputable def B_complement (a : ℝ) := {x : ℝ | x^2 - 2*x + a ≤ 0}

-- (I) When a = -8, prove A ∩ B = {x : ℝ | 4 < x ∧ x ≤ 5}
theorem part_I : B (-8) = {x : ℝ | x < -2 ∨ x > 4} →
  A ∩ {x : ℝ | x < -2 ∨ x > 4} = {x : ℝ | 4 < x ∧ x ≤ 5} :=
by
  intros hB
  sorry

-- (II) If A ∩ B_complement = {x : ℝ | -1 < x ∧ x ≤ 3}, prove a = -3
theorem part_II : (A ∩ B_complement a = {x : ℝ | -1 < x ∧ x ≤ 3}) → a = -3 :=
by
  intros hA_complement_B
  sorry

end part_I_part_II_l787_787647


namespace horner_eval_example_l787_787847

noncomputable def horner_eval (c : List ℕ) (x : ℕ) : ℕ :=
c.foldr (λ a acc, a + x * acc) 0

theorem horner_eval_example : horner_eval [1, 1, 1, 1, 0, 1] 3 = 283 :=
by
  -- We need to evaluate the polynomial using Horner's method
  -- The given polynomial f(x) = x^5 + x^3 + x^2 + x + 1 in Horner's form is represented by the coefficients [1, 1, 1, 1, 0, 1]
  -- horner_eval [1, 1, 1, 1, 0, 1] 3 = 283
  sorry

end horner_eval_example_l787_787847


namespace log2_five_gt_log2_three_l787_787567

-- Given conditions:
def log2_is_strictly_increasing (x₁ x₂ : ℝ) (h : x₁ < x₂) : log 2 x₁ < log 2 x₂ := sorry

-- Let's define the specific values:
theorem log2_five_gt_log2_three : log 2 5 > log 2 3 :=
by
  -- By using the given condition that log2 is strictly increasing:
  have h₁ : 5 > 3 := by linarith
  exact log2_is_strictly_increasing 3 5 h₁

end log2_five_gt_log2_three_l787_787567


namespace more_girls_than_boys_l787_787101

def ratio_boys_girls (B G : ℕ) : Prop := B = (3/5 : ℚ) * G

def total_students (B G : ℕ) : Prop := B + G = 16

theorem more_girls_than_boys (B G : ℕ) (h1 : ratio_boys_girls B G) (h2 : total_students B G) : G - B = 4 :=
by
  sorry

end more_girls_than_boys_l787_787101


namespace simplify_fractions_l787_787402

theorem simplify_fractions : 5 * (21 / 6) * (18 / -63) = -5 := by
  sorry

end simplify_fractions_l787_787402


namespace empty_seats_l787_787873

theorem empty_seats (total_seats : ℕ) (people_watching : ℕ) (h_total_seats : total_seats = 750) (h_people_watching : people_watching = 532) : 
  total_seats - people_watching = 218 :=
by
  sorry

end empty_seats_l787_787873


namespace selected_numbers_product_l787_787613

theorem selected_numbers_product :
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 32 ∧ 1 ≤ y ∧ y ≤ 32 ∧ (528 - x - y = x * y) ∧ (x * y = 484) := 
begin
  -- solution omitted
  sorry
end

end selected_numbers_product_l787_787613


namespace group_with_odd_digit_sums_has_more_members_l787_787545

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def is_odd_sum_of_digits (n : ℕ) : Prop :=
  sum_of_digits n % 2 = 1

def count_numbers_with_property (n : ℕ) (P : ℕ → Prop) : ℕ :=
  (finset.range (n + 1)).filter P |>.card

theorem group_with_odd_digit_sums_has_more_members :
  count_numbers_with_property 1000000 is_odd_sum_of_digits >
  1000000 / 2 :=
by
  sorry

end group_with_odd_digit_sums_has_more_members_l787_787545


namespace whale_sixth_hour_consumption_l787_787181

theorem whale_sixth_hour_consumption : 
  ∃ (x : ℕ), 
    (x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) + (x + 24) + (x + 28) + (x + 32) = 450) 
    ∧ (x + 20 = 54) := 
by {
  sorry,
}

end whale_sixth_hour_consumption_l787_787181


namespace all_rows_unique_l787_787942

-- Define the generation rules according to the problem statement.
def generate_row (a : ℕ) (n : ℕ) : List ℕ :=
  if n = 0 then
    [a]
  else
    let prev_row := generate_row a (n-1)
    (prev_row.map (λ k => k^2)) ++ (prev_row.map (λ k => k + 1))

-- Statement that all rows of the triangle table have unique elements.
theorem all_rows_unique (a : ℕ) (n : ℕ) : ∀ row ≤ n, nodup (generate_row a row) := by
  sorry

end all_rows_unique_l787_787942


namespace closest_numbers_to_1994_l787_787433

def sequence_nth (n : ℕ) : ℕ :=
  -- definition of the nth term in the sequence according to the formation rule
  sorry

theorem closest_numbers_to_1994 : 
  let idx := 1994 in 
  let term_before := 1993 in
  let term_after := 1995 in
  sequence_nth idx = term_before ∨ sequence_nth idx = term_after :=
  sorry

end closest_numbers_to_1994_l787_787433


namespace find_a_l787_787816

theorem find_a (b c : ℤ) 
  (vertex_condition : ∀ (x : ℝ), x = -1 → (ax^2 + b*x + c) = -2)
  (point_condition : ∀ (x : ℝ), x = 0 → (a*x^2 + b*x + c) = -1) :
  ∃ (a : ℤ), a = 1 :=
by
  sorry

end find_a_l787_787816


namespace multiples_of_7_count_l787_787670

def is_multiple_of_7 (n : ℕ) : Prop := n % 7 = 0

def in_range (n : ℕ) : Prop := 50 ≤ n ∧ n ≤ 150

theorem multiples_of_7_count : (finset.card (finset.filter (λ n, is_multiple_of_7 n ∧ in_range n) (finset.range 151))) = 14 :=
by
  sorry

end multiples_of_7_count_l787_787670


namespace projection_ratio_correct_l787_787417

-- Definitions and conditions directly from problem statement
variables {a b c α β γ : ℝ}
def cubic (x : ℝ) := x^3 + a * x^2 + b * x + c

-- Intersection points with the x-axis
def intersects_at_x_axis (f : ℝ → ℝ) : Prop :=
  ∃ α β γ : ℝ, f α = 0 ∧ f β = 0 ∧ f γ = 0

-- Tangents from points
def tangents_intersect_at_p_and_q (f : ℝ → ℝ) (α β : ℝ) : Prop :=
  ∃ P : ℝ, P ≠ α ∧ ∃ Q : ℝ, Q ≠ β

-- Projection ratio calculation
def projection_ratio (α β γ : ℝ) : ℝ :=
  let AB := β - α
  let PQ := (α - β) / 2
  AB / PQ

theorem projection_ratio_correct (f : ℝ → ℝ) {α β γ : ℝ}
  (h1 : f = cubic)
  (h2 : intersects_at_x_axis f)
  (h3 : tangents_intersect_at_p_and_q f α β) :
  projection_ratio α β γ = -2 :=
sorry

end projection_ratio_correct_l787_787417


namespace function_domain_length_correct_l787_787230

noncomputable def function_domain_length : ℕ :=
  let p : ℕ := 240 
  let q : ℕ := 1
  p + q

theorem function_domain_length_correct : function_domain_length = 241 := by
  sorry

end function_domain_length_correct_l787_787230


namespace matrix_power_10_l787_787936

theorem matrix_power_10 :
  let A := (Matrix.of ![![1, 1], ![1, 1]]) in
  A ^ 10 = (Matrix.of ![![512, 512], ![512, 512]]) :=
by
  let A := (Matrix.of ![![1, 1], ![1, 1]])
  sorry

end matrix_power_10_l787_787936


namespace sum_of_first_100_terms_b_sequence_l787_787255

noncomputable def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b_sequence (n : ℕ) : ℝ := (-1 : ℝ)^(n-1) * ((4 * n) / ((2 * n - 1) * (2 * n + 1)))

theorem sum_of_first_100_terms_b_sequence : 
  ∑ k in Finset.range 100, b_sequence (k + 1) = 200 / 201 :=
sorry

end sum_of_first_100_terms_b_sequence_l787_787255


namespace triplet_D_forms_right_angle_triangle_l787_787922

-- Definitions for sets of side lengths
def setA : ℕ × ℕ × ℕ := (1, 3, 4)
def setB : ℕ × ℕ × ℕ := (2, 3, 4)
def setC : ℕ × ℕ × ℕ := (1, 1, sqrt 3)
def setD : ℕ × ℕ × ℕ := (5, 12, 13)

-- Definition of Pythagorean theorem condition
def is_right_angle_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Theorem statement
theorem triplet_D_forms_right_angle_triangle : is_right_angle_triangle 5 12 13 :=
sorry

end triplet_D_forms_right_angle_triangle_l787_787922


namespace concyclic_AIKE_l787_787190

universe u
variables {α : Type u}

def cyclic (A B C D : α) [metric_space α] :=
  dist A B * dist C D + dist B C * dist D A + dist C A * dist B D = dist A C * dist B D + dist A B * dist D C + dist B C * dist A D

variables (A B C D P E F I J K : α) [metric_space α]

-- Conditions
axiom inscribed_quadrilateral : cyclic A B C D  -- ABCD is inscribed in a circle
axiom intersection_diagonals : ∃ P, ∀ Q, Q ∈ [A, C] ∩ [B, D] ↔ Q = P  -- Diagonals intersect at P
axiom circumcircle_APD : (∃ O', (E ∈ circ O' A P D ∧ E ∈ seg A B))  -- Circumcircle of ΔAPD intersects AB at E
axiom circumcircle_BPC : (∃ O', (F ∈ circ O' B P C ∧ F ∈ seg A B))  -- Circumcircle of ΔBPC intersects AB at F
axiom incenter_ADE : incenter A D E I  -- I is the incenter of ΔADE
axiom incenter_BCF : incenter B C F J  -- J is the incenter of ΔBCF
axiom intersection_IJ_AC : K ∈ [I, J] ∩ [A, C]  -- IJ intersects AC at K

-- Proof to be done
theorem concyclic_AIKE : cyclic A I K E :=
by sorry

end concyclic_AIKE_l787_787190


namespace exists_2005_distinct_numbers_l787_787224

theorem exists_2005_distinct_numbers :
  ∃ (S : Finset ℕ), S.card = 2005 ∧ 
  ∀ x ∈ S, (∑ y in (S.erase x), y) % x = 0 :=
sorry

end exists_2005_distinct_numbers_l787_787224


namespace at_least_one_false_l787_787634

theorem at_least_one_false (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
  by
  sorry

end at_least_one_false_l787_787634


namespace part1_part2_part3_l787_787161

noncomputable def prob_of_first_class_day : ℚ := 27 / 64

theorem part1 (prob_first_class : ℚ) (prob_second_class : ℚ) :
  prob_first_class = 1 / 2 →
  prob_second_class = 4 / 10 →
  3 * (prob_first_class^2) * (1 - prob_first_class)^2 = prob_of_first_class_day :=
by 
  intros,
  have prob_not_first_class := 1 - prob_first_class,
  have binomial_coefficient := 3,
  have calculation := binomial_coefficient * (prob_first_class ^ 2) * (prob_not_first_class ^ 2),
  exact calculation

theorem part2 (prob_first_class : ℚ) :
  prob_first_class = 1 / 2 →
  prob_first_class * prob_first_class / ((prob_first_class * prob_first_class) + 
  2 * prob_first_class * (1 - prob_first_class)) = 1 / 3 :=
by 
  intros,
  have prob_not_first_class := 1 - prob_first_class,
  have cond_prob := prob_first_class * prob_first_class / (prob_first_class * prob_first_class + 2 * prob_first_class * prob_not_first_class),
  exact cond_prob

noncomputable def expected_daily_profit : ℚ := 12200

theorem part3 (prob_first_class : ℚ) (prob_second_class : ℚ) :
  prob_first_class = 1 / 2 →
  prob_second_class = 4 / 10 →
  let profit_values := [16000, 14000, 5000, 12000, 3000, -6000] in
  let probabilities := [
    prob_first_class ^ 2, 
    2 * (prob_first_class * prob_second_class), 
    2 * (prob_first_class * (1 - prob_first_class - prob_second_class)), 
    prob_second_class^2,
    2 * (prob_second_class * (1 - prob_first_class - prob_second_class)),
    (1 - prob_first_class - prob_second_class)^2
  ] in
  (list.zip_with (λ x y, x * y) probabilities profit_values).sum = expected_daily_profit :=
by 
  intros,
  let profit_values := [16000, 14000, 5000, 12000, 3000, -6000],
  let probabilities := [
    prob_first_class ^ 2, 
    2 * (prob_first_class * prob_second_class), 
    2 * (prob_first_class * (1 - prob_first_class - prob_second_class)), 
    prob_second_class^2,
    2 * (prob_second_class * (1 - prob_first_class - prob_second_class)),
    (1 - prob_first_class - prob_second_class)^2
  ],
  have expected_value := list.zip_with (λ x y, x * y) probabilities profit_values.sum,
  exact expected_value

end part1_part2_part3_l787_787161


namespace infinite_triples_l787_787059

theorem infinite_triples : ∃ᶠ (a b c : ℕ), 2 * a ^ 2 + 3 * b ^ 2 - 5 * c ^ 2 = 1997 := sorry

end infinite_triples_l787_787059


namespace four_digit_number_correct_l787_787598

theorem four_digit_number_correct :
  ∃ (grid : Fin 3 → Fin 2 → ℕ), 
    (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 6) ∧
    (∀ i, ∀ (j₁ j₂ : Fin 2), j₁ ≠ j₂ → grid i j₁ ≠ grid i j₂) ∧
    (∀ j, ∀ (i₁ i₂ : Fin 3), i₁ ≠ i₂ → grid i₁ j ≠ grid i₂ j) ∧
    (∀ i j, ∑ k in finset.univ.filter (λ k, (i, j) ∈ small_circle_coords), grid (i + k.1) (j + k.2) = (some composite number)) ∧
    (∀ (i j : Fin 3 × Fin 2), (i, j) ∉ small_circle_coords → is_prime (grid i.1 i.2 + grid j.1 j.2)) ∧
    (fourdigit_number = 4123) := sorry

end four_digit_number_correct_l787_787598


namespace max_value_f_l787_787966

noncomputable def f (x : ℝ) := x^6 / (x^12 + 3*x^9 - 6*x^6 + 12*x^3 + 27)

theorem max_value_f : ∃ x_max : ℝ, f x_max = 1 / (12 * real.sqrt 3) := 
sorry

end max_value_f_l787_787966


namespace integer_pairs_l787_787599

theorem integer_pairs (m n : ℤ) : (m + n = m * n - 1) ↔ 
            ((m, n) = (2, 3) ∨ (m, n) = (3, 2) ∨ (m, n) = (0, -1) ∨ (m, n) = (-1, 0)) :=
begin
  sorry
end

end integer_pairs_l787_787599


namespace locus_is_hyperbola_perpendicular_points_inverse_sum_constant_l787_787288

-- Define the circles and parameters
def circleA (x y : ℝ) := (x + 4)^2 + y^2 = 25
def circleB (x y : ℝ) := (x - 4)^2 + y^2 = 1

-- Define the hyperbola
def locus_of_center (x y : ℝ) := (x^2 / 4) - (y^2 / 12) = 1 ∧ x > 0

-- Prove the first statement
theorem locus_is_hyperbola :
  (∀ (x y : ℝ), (∃ (Mx My : ℝ), circleA Mx My ∧ circleB Mx My ∧ tangent_to_both_circles Mx My x y) → locus_of_center x y) :=
sorry

-- Define the perpendicular condition and the second statement
def is_perpendicular (P Q : ℝ × ℝ) (O : ℝ × ℝ) := (P.1 - O.1) * (Q.1 - O.1) = 0 ∨ (P.2 - O.2) * (Q.2 - O.2) = 0

theorem perpendicular_points_inverse_sum_constant (k : ℝ) (h : k ≠ 0) :
  (∀ (P Q : ℝ × ℝ), ∃ (x y : ℝ), locus_of_center x y ∧ ((y = k * x) ∨ (y = -1/k * x)) ∧ is_perpendicular P Q (0, 0) →
  (1 / (P.1^2 + P.2^2) + 1 / (Q.1^2 + Q.2^2) = 1 / 6)) :=
sorry

end locus_is_hyperbola_perpendicular_points_inverse_sum_constant_l787_787288


namespace service_center_milepost_l787_787814

theorem service_center_milepost :
  let mp4 := 50
  let mp12 := 190
  let service_center := mp4 + (mp12 - mp4) / 2
  service_center = 120 :=
by
  let mp4 := 50
  let mp12 := 190
  let service_center := mp4 + (mp12 - mp4) / 2
  sorry

end service_center_milepost_l787_787814


namespace probability_difference_3_or_greater_l787_787456

noncomputable def set_of_numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def total_combinations : ℕ := (finset.image (λ (p : finset ℕ), p) (finset.powerset (finset.univ ∩ set_of_numbers))).card / 2

def pairs_with_difference_less_than_3 (S : finset ℕ) : finset (finset ℕ) :=
finset.filter (λ (p : finset ℕ), p.card = 2 ∧ abs ((p.min' sorry) - (p.max' sorry)) < 3) S

def successful_pairs (S : finset ℕ) : finset (finset ℕ) :=
finset.filter (λ (p : finset ℕ), p.card = 2 ∧ abs ((p.min' sorry) - (p.max' sorry)) ≥ 3) S
   
def probability (S : finset ℕ) : ℚ :=
(successful_pairs S).card.to_rat / total_combinations

theorem probability_difference_3_or_greater :
  probability set_of_numbers = 7/12 :=
sorry

end probability_difference_3_or_greater_l787_787456


namespace calculate_current_l787_787650

noncomputable def V1 : ℂ := 2 + I
noncomputable def V2 : ℂ := -1 + 4 * I
noncomputable def Z : ℂ := 2 + 2 * I
noncomputable def V : ℂ := V1 + V2

theorem calculate_current : (V / Z) = -1 + I := by
  -- Further proof would go here
  sorry

end calculate_current_l787_787650


namespace pipes_fill_tank_in_8_hours_l787_787535

theorem pipes_fill_tank_in_8_hours (A B C : ℝ) (hA : A = 1 / 56) (hB : B = 2 * A) (hC : C = 2 * B) :
  1 / (A + B + C) = 8 :=
by
  sorry

end pipes_fill_tank_in_8_hours_l787_787535


namespace square_side_length_in_right_triangle_l787_787048

theorem square_side_length_in_right_triangle :
  ∀ (DE EF DF : ℝ) (s : ℝ),
  DE = 5 ∧ EF = 12 ∧ DF = 13 ∧
  (∃ P Q S R, 
    P ∈ DF ∧ Q ∈ DF ∧ 
    S ∈ DE ∧ 
    R ∈ EF ∧ 
    square PQRS ∧ 
    ℓ PQ = s) →
  s = 780 / 169 :=
by {
  sorry
}

end square_side_length_in_right_triangle_l787_787048


namespace complex_number_quadrant_l787_787149

theorem complex_number_quadrant :
  let i := Complex.I in
  let z := (1 + i) / i in
  Complex.re z = 1 ∧ Complex.im z = -1 → 
  (Complex.re z > 0 ∧ Complex.im z < 0) :=
by 
  let i := Complex.I
  let z := (1 + i) / i
  show (Complex.re z = 1 ∧ Complex.im z = -1 → (Complex.re z > 0 ∧ Complex.im z < 0))
  sorry

end complex_number_quadrant_l787_787149


namespace l1_equation_correct_l2_equation_correct_l787_787257

noncomputable def circle_eq : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 - 2*x + 4*y - 4 = 0

def P : ℝ × ℝ := (2, 0)

def line_l1_valid (l1_eq : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, (l1_eq = (λ x y, k*x - y - 2*k = 0) ∨ l1_eq = (λ x y, x = 2)) ∧
  (∃ center : ℝ × ℝ, center = (1, -2) ∧ (√((center.1 - 2)^2 + center.2^2) = 1))

def length_of_chord (length : ℝ) : Prop :=
  length = 4*√2

def line_l2_valid (l2_eq : ℝ → ℝ → Prop) : Prop :=
  ∃ b : ℝ, (l2_eq = (λ x y, x - y - b = 0) ∧ (l2_eq = (λ x y, x - y - 4 = 0) ∨ l2_eq = (λ x y, x - y + 1 = 0)))

theorem l1_equation_correct : 
  ∀ l1_eq : ℝ → ℝ → Prop, (line_l1_valid l1_eq) → (∃ y, l1_eq 2 y) → (l1_eq = (λ x y, x = 2) ∨ l1_eq = (λ x y, 3 * x - 4 * y - 6 = 0)) := 
by sorry

theorem l2_equation_correct : 
  ∀ l2_eq : ℝ → ℝ → Prop, (line_l2_valid l2_eq) → ((∃ x y : ℝ, l2_eq x y) ∧ (∃ x y : ℝ, circle_eq x y)) → (l2_eq = (λ x y, x - y - 4 = 0) ∨ l2_eq = (λ x y, x - y + 1 = 0)) := 
by sorry

end l1_equation_correct_l2_equation_correct_l787_787257


namespace sixth_grade_percentage_combined_l787_787437

def maplewood_percentages := [10, 20, 15, 15, 10, 15, 15]
def brookside_percentages := [16, 14, 13, 12, 12, 18, 15]

def maplewood_students := 150
def brookside_students := 180

def sixth_grade_maplewood := maplewood_students * (maplewood_percentages.get! 6) / 100
def sixth_grade_brookside := brookside_students * (brookside_percentages.get! 6) / 100

def total_students := maplewood_students + brookside_students
def total_sixth_graders := sixth_grade_maplewood + sixth_grade_brookside

def sixth_grade_percentage := total_sixth_graders / total_students * 100

theorem sixth_grade_percentage_combined : sixth_grade_percentage = 15 := by 
  sorry

end sixth_grade_percentage_combined_l787_787437


namespace distribute_people_l787_787325

theorem distribute_people :
  ∃ (M W : ℕ) (group1 group2 group3 : list (ℕ × ℕ)) (M1 M2 M3 : ℕ) (W1 W2 W3 : ℕ),
    M = 4 ∧ W = 5 ∧
    group1.length = 2 ∧ group2.length = 3 ∧ group3.length = 3 ∧
    ∀ group, group ∈ [group1, group2, group3] → ∃ m w, (m, w) ∈ group ∧ w ≠ 0 ∧ m ≠ 0 ∧
    (((M1, W1) :: group1) ++ group2 ++ group3).length = 9 ∧
    M1 + M2 + M3 = 4 ∧ W1 + W2 + W3 = 5 ∧
    (((((M1, W1) :: group1) ++ group2 ++ group3).length = 9 ∧
    (((M, W) :: list_of_groups)).length = 3 ∧
    (((M1 * W1) + (M2 * W2) + (M3 * W3)) = 360).
sorry

end distribute_people_l787_787325


namespace exponent_equation_l787_787679

theorem exponent_equation (x : ℝ) (h : 7^(6 * x) = 2401) : 7^(6 * x - 4) = 1 := by
  sorry

end exponent_equation_l787_787679


namespace specific_n_values_l787_787581

theorem specific_n_values (n : ℕ) : 
  ∃ m : ℕ, 
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → m % k = 0) ∧ 
    (m % (n + 1) ≠ 0) ∧ 
    (m % (n + 2) ≠ 0) ∧ 
    (m % (n + 3) ≠ 0) ↔ n = 1 ∨ n = 2 ∨ n = 6 := 
by
  sorry

end specific_n_values_l787_787581


namespace sales_function_profit_8250_max_profit_l787_787455

-- Conditions
def cost_price := 80
def min_price := 90
def max_price := 110

def initial_sales := 600
def decrease_per_yuan := 10

-- Problem 1
theorem sales_function (x : ℕ) : 
  x >= min_price -> 
  y = 600 - 10 * (x - 90) -> 
  y = -10 * x + 1500 := sorry

-- Problem 2
theorem profit_8250 (x : ℕ) :
  x >= min_price -> 
  (x - 80) * (600 - 10 * (x - 90)) = 8250 -> 
  x = 95 := sorry

-- Problem 3
theorem max_profit (x : ℕ) :
  x >= min_price ->
  x <= max_price ->
  (x - 80) * (600 - 10 * (x - 90)) = -10*(110 - 80)^2 + 12250 -> 
  ( (x ≤ 110) 
    -> ∃ p : ℕ, p = -10*(110 - x)^2 + 12250
    -> p ≤ 12000 ) := sorry

end sales_function_profit_8250_max_profit_l787_787455


namespace count_integers_congruent_to_7_mod_13_l787_787674

theorem count_integers_congruent_to_7_mod_13:
  (∃ S : Finset ℕ, S.card = 154 ∧ ∀ n ∈ S, n < 2000 ∧ n % 13 = 7) :=
by
  sorry

end count_integers_congruent_to_7_mod_13_l787_787674


namespace domain_of_function_l787_787078

-- Define the function
def f (x : ℝ) : ℝ := real.sqrt (2^x - 4)

-- State the theorem to prove the domain of the function
theorem domain_of_function : {x : ℝ | 2^x - 4 ≥ 0} = set.Ici 2 :=
by
  sorry

end domain_of_function_l787_787078


namespace final_sum_is_653_l787_787967

-- Define the sequence a_n as per the given conditions
def a_n (n : ℕ) : ℕ :=
  if n % 240 = 0 then 15
  else if n % 144 = 0 then 16
  else if n % 306 = 0 then 17
  else 0

-- Define the sum of the sequence up to 2999
def sum_sequence : ℕ :=
  ∑ n in Finset.range 3000, a_n n

-- The final statement we want to prove
theorem final_sum_is_653 : sum_sequence = 653 := by
  sorry

end final_sum_is_653_l787_787967


namespace value_of_d_is_two_l787_787421

theorem value_of_d_is_two (d : ℝ) :
  (∀ (x y : ℝ), 2 * x - y = d → ∀ (p q : ℝ), (p, q) = (2, 2) → True) →
  (-2, 4) ≠ (6, 0) → ∀ (m n p q : ℝ), (m, n) = (-2, 4) → (p, q) = (6, 0) →
  ∃ (x y: ℝ), (x, y) = ((-2 + 6) / 2, (4 + 0) / 2) ∧ 2 * x - y = d :=
begin
  intros h_segment_neq h_segment_eq,
  existsi (2 : ℝ),
  existsi (2 : ℝ),
  split,
  { exact ⟨rfl, rfl⟩ },
  { sorry }
end

end value_of_d_is_two_l787_787421


namespace distance_sum_identity_l787_787260

noncomputable def squared_distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem distance_sum_identity
  (a b c x y : ℝ)
  (A B C P G : ℝ × ℝ)
  (hA : A = (a, b))
  (hB : B = (-c, 0))
  (hC : C = (c, 0))
  (hG : G = (a / 3, b / 3))
  (hP : P = (x, y))
  (hG_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) :
  squared_distance A P + squared_distance B P + squared_distance C P =
  squared_distance A G + squared_distance B G + squared_distance C G + 3 * squared_distance G P :=
by sorry

end distance_sum_identity_l787_787260


namespace circumcenter_lies_on_reflection_line_l787_787019

variables {A B C D U : Type} [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D]
variable [circumcenterQuaddilateral : ∀ P Q R S : Type, ConvexQuadrilateral P Q R S → Point] 

-- Define the cyclic quadrilateral, and perpendicular diagonals conditions
variables (h_cyclic: CyclicQuadrilateral A B C D)
          (h_perpendicular_diagonals: Perpendicular (diagonal A C) (diagonal B D))
          (h_circumcenter: U = circumcenter A B C D)

-- Reflect the diagonal AC across the angle bisector of ∠BAD
noncomputable theory
def reflect_diagonal (line : Line) (angle_bisector : Line) : Line := sorry

noncomputable def g : Line := reflect_diagonal (diagonal A C) (angle_bisector ∠BAD)

-- Main theorem stating that U lies on the line g
theorem circumcenter_lies_on_reflection_line :
  U ∈ g :=
sorry

end circumcenter_lies_on_reflection_line_l787_787019


namespace triangle_area_l787_787696

theorem triangle_area (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) : 
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 := 
by 
  sorry

end triangle_area_l787_787696


namespace find_a_extreme_values_l787_787651

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + 4 * Real.log (x + 1)
noncomputable def f' (x a : ℝ) : ℝ := 2 * (x - a) + 4 / (x + 1)

-- Given conditions
theorem find_a (a : ℝ) :
  f' 1 a = 0 ↔ a = 2 :=
by
  sorry

theorem extreme_values :
  ∃ x : ℝ, -1 < x ∧ f (0 : ℝ) 2 = 4 ∨ f (1 : ℝ) 2 = 1 + 4 * Real.log 2 :=
by
  sorry

end find_a_extreme_values_l787_787651


namespace age_of_B_l787_787490

variable (a b : ℕ)

-- Conditions
def condition1 := a + 10 = 2 * (b - 10)
def condition2 := a = b + 5

-- The proof goal
theorem age_of_B (h1 : condition1 a b) (h2 : condition2 a b) : b = 35 := by
  sorry

end age_of_B_l787_787490


namespace length_of_platform_correct_l787_787500

-- Given data
variables (length_of_train : ℝ) (time_to_cross_platform : ℝ) (time_to_cross_pole : ℝ)
#check length_of_train -- ℝ
#check time_to_cross_platform -- ℝ
#check time_to_cross_pole -- ℝ

-- Definitions and assumptions based on given conditions
def length_of_train := 300  -- meters
def time_to_cross_platform := 39  -- seconds
def time_to_cross_pole := 18  -- seconds

-- Define the length of the platform
noncomputable def length_of_platform (v : ℝ) : ℝ := (v * time_to_cross_platform) - length_of_train

-- Calculate the speed of the train
noncomputable def speed_of_train : ℝ := length_of_train / time_to_cross_pole

-- The proof problem statement
theorem length_of_platform_correct : length_of_platform speed_of_train = 350.13 := 
by 
    sorry -- Proof not required

end length_of_platform_correct_l787_787500


namespace tan_2alpha_values_l787_787245

theorem tan_2alpha_values (α : ℝ) (h : 2 * sin (2 * α) = 1 + cos (2 * α)) :
  tan (2 * α) = 4 / 3 ∨ tan (2 * α) = 0 :=
by
  sorry

end tan_2alpha_values_l787_787245


namespace three_nabla_four_l787_787682

noncomputable def modified_operation (a b : ℝ) : ℝ :=
  (a + b^2) / (1 + a * b^2)

theorem three_nabla_four : modified_operation 3 4 = 19 / 49 := 
  by 
  sorry

end three_nabla_four_l787_787682


namespace geometric_series_sum_l787_787202

theorem geometric_series_sum :
  let a := (1 : ℝ) / 2
  let r := (1 : ℝ) / 2
  let n := 5
  S = a * (1 - r^n) / (1 - r) → S = 31 / 32 :=
by
  intros a r n S h
  have h1 : a = 1 / 2 := rfl
  have h2 : r = 1 / 2 := rfl
  have h3 : n = 5 := rfl
  rw [h1, h2, h3] at h
  sorry

end geometric_series_sum_l787_787202


namespace simplify_and_evaluate_l787_787061

theorem simplify_and_evaluate :
  ∀ (x y : ℚ), x = 2 ∧ y = -1 / 2 →
  x * (x - 4 * y) + (2 * x + y) * (2 * x - y) - (2 * x - y) ^ 2 = 7 / 2 :=
by
  intros x y h
  cases h with hx hy
  rw [hx, hy]
  dsimp
  -- simplify steps would go here
  sorry

end simplify_and_evaluate_l787_787061


namespace largest_angle_in_pentagon_l787_787818

-- Define the conditions as variables and assumptions in Lean
variable (x : ℝ) -- Middle angle in degrees

-- Conditions
axiom sum_of_pentagon : (x - 3) + (x - 2) + (x - 1) + x + (x + 1) = 540

-- Goal
theorem largest_angle_in_pentagon : x = 109 → (x + 1) = 110 :=
by
  intro hx
  have h : (x - 3) + (x - 2) + (x - 1) + x + (x + 1) = 5x - 5 := by linarith
  rw [sum_of_pentagon, h] at hx
  linarith

end largest_angle_in_pentagon_l787_787818


namespace hyperbola_eccentricity_l787_787958

theorem hyperbola_eccentricity (a b : ℝ) (h : a = b) : 
  let c := Real.sqrt (2 * a^2)
  in (c / a) = Real.sqrt 2 := by
  sorry

end hyperbola_eccentricity_l787_787958


namespace most_stable_performance_l787_787612

-- Define the variances for each player
def variance_A : ℝ := 0.66
def variance_B : ℝ := 0.52
def variance_C : ℝ := 0.58
def variance_D : ℝ := 0.62

-- State the theorem
theorem most_stable_performance : variance_B < variance_C ∧ variance_C < variance_D ∧ variance_D < variance_A :=
by
  -- Since we are tasked to write only the statement, the proof part is skipped.
  sorry

end most_stable_performance_l787_787612


namespace community_service_selection_l787_787510

theorem community_service_selection {boys girls A B : ℕ} 
    (total_students : boys + girls = 10) 
    (boys_count : boys = 6) 
    (girls_count : girls = 4) 
    (participation_condition : A ∈ {1, 2} ↔ B ∈ {1, 2}) 
    (min_two_girls : girls ≥ 2) : 
    ∃ n, n = 85 := 
sorry

end community_service_selection_l787_787510


namespace tens_digit_of_9_pow_1503_is_2_l787_787123

def last_two_digits_cycle : List (Nat × Nat) := 
  [(1, 9), (2, 81), (3, 29), (4, 61), (5, 49), (6, 41), 
   (7, 69), (8, 21), (9, 89), (10, 1)]

theorem tens_digit_of_9_pow_1503_is_2 : 
  let n := 1503
  let last_two_digits := last_two_digits_cycle.get! ((n - 1) % 10)
  (last_two_digits / 10) % 10 = 2 :=
by 
  let n := 1503
  have last_two_digits := List.get! last_two_digits_cycle ((n - 1) % 10)
  have expected := (29 : Nat)
  exact Eq.refl ((expected / 10) % 10)

end tens_digit_of_9_pow_1503_is_2_l787_787123


namespace exist_BD_max_perimeter_l787_787252

-- Given conditions
def is_convex_quadrilateral (A B C D : Point) : Prop := 
  convex_quadrilateral A B C D

def angle_A_eq (A B C D : Point) : Prop :=
  angle A B C = π / 3

def BC_eq_1 (B C : Point) : Prop := 
  distance B C = 1

def DC_eq_2 (D C : Point) : Prop := 
  distance D C = 2

def area_ABDC_eq (A B C D : Point) : Prop := 
  area A B D C = sqrt 3 / 2

-- Proof problem statements
theorem exist_BD (A B C D : Point) 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : angle_A_eq A B C D)
  (h3 : BC_eq_1 B C)
  (h4 : DC_eq_2 D C)
  (h5 : area_ABDC_eq A B C D) : 
  distance B D = sqrt 3 ∨ distance B D = sqrt 7 :=
sorry

theorem max_perimeter (A B C D : Point)
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : angle_A_eq A B C D)
  (h3 : BC_eq_1 B C)
  (h4 : DC_eq_2 D C)
  (h5 : area_ABDC_eq A B C D) :
  max_perimeter (quadrilateral A B C D) = 2 * sqrt 7 + 3 :=
sorry

end exist_BD_max_perimeter_l787_787252


namespace total_albums_l787_787038

theorem total_albums (Adele Bridget Katrina Miriam : ℕ) 
  (h₁ : Adele = 30)
  (h₂ : Bridget = Adele - 15)
  (h₃ : Katrina = 6 * Bridget)
  (h₄ : Miriam = 5 * Katrina) : Adele + Bridget + Katrina + Miriam = 585 := 
by
  sorry

end total_albums_l787_787038


namespace bobbo_minimum_speed_increase_l787_787195

theorem bobbo_minimum_speed_increase
  (initial_speed: ℝ)
  (river_width : ℝ)
  (current_speed : ℝ)
  (waterfall_distance : ℝ)
  (midpoint_distance : ℝ)
  (required_increase: ℝ) :
  initial_speed = 2 ∧ river_width = 100 ∧ current_speed = 5 ∧ waterfall_distance = 175 ∧ midpoint_distance = 50 ∧ required_increase = 3 → 
  (required_increase = (50 / (50 / current_speed)) - initial_speed) := 
by
  sorry

end bobbo_minimum_speed_increase_l787_787195


namespace correct_sum_pq_l787_787729

-- Given:
def y_polynomial : Polynomial ℤ := y^8 + 64 * y^4 + 36
def p (y : ℤ) : Polynomial ℤ := Polynomial.monic (y^4 + 8 * y^2 + 6)
def q (y : ℤ) : Polynomial ℤ := Polynomial.monic (y^4 - 8 * y^2 + 6)

-- Prove:
theorem correct_sum_pq : (p 1) + (q 1) = 14 := by
  sorry

end correct_sum_pq_l787_787729


namespace sum_of_midpoint_is_six_l787_787557

-- Define the endpoints
def endpoint1 : (ℝ × ℝ) := (10, -3)
def endpoint2 : (ℝ × ℝ) := (-4, 9)

-- Define the midpoint
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the sum of the coordinates of the midpoint
def sum_of_midpoint_coordinates (M : ℝ × ℝ) : ℝ :=
  M.1 + M.2

-- The main theorem
theorem sum_of_midpoint_is_six : sum_of_midpoint_coordinates (midpoint endpoint1 endpoint2) = 6 := 
by 
  -- This is where the proof would go 
  sorry

end sum_of_midpoint_is_six_l787_787557


namespace medial_triangle_ratio_l787_787007

theorem medial_triangle_ratio (p q r : ℝ)
  (A B C : ℝ × ℝ × ℝ)
  (mid_BC : (p, 2, 0) = ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2))
  (mid_AC : (0, q, 2) = ((A.1 + C.1) / 2, (A.2 + C.2) / 2, (A.3 + C.3) / 2))
  (mid_AB : (0, 0, r) = ((B.1 + C.1) / 2, (B.2 + C.2) / 2, (B.3 + C.3) / 2))
  : (AB^2 + AC^2 + BC^2) / (p^2 + q^2 + r^2) = 8 :=
sorry

end medial_triangle_ratio_l787_787007


namespace positive_integers_divide_n_plus_7_l787_787850

theorem positive_integers_divide_n_plus_7 (n : ℕ) (hn_pos : 0 < n) : n ∣ n + 7 ↔ n = 1 ∨ n = 7 :=
by 
  sorry

end positive_integers_divide_n_plus_7_l787_787850


namespace abs_neg_three_l787_787411

theorem abs_neg_three : abs (-3) = 3 :=
sorry

end abs_neg_three_l787_787411


namespace find_b_l787_787830

noncomputable def P (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_b (a b c : ℝ) (h1: (-(a / 3) = -c)) (h2 : (-(a / 3) = 1 + a + b + c)) (h3: c = 2) : b = -11 :=
by
  sorry

end find_b_l787_787830


namespace nth_derivative_divisible_l787_787013

theorem nth_derivative_divisible (P : Polynomial ℤ) (n : ℕ) : 
  ∀ k : ℕ, k-th_coefficient (P.derivative^[n]) % n.factorial = 0 :=
sorry

end nth_derivative_divisible_l787_787013


namespace larger_root_exceeds_smaller_l787_787139

theorem larger_root_exceeds_smaller :
  let a := 2 
  let b := 5 
  let c := -12 
  let discriminant := b ^ 2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let root1 := (-b + sqrt_discriminant) / (2 * a)
  let root2 := (-b - sqrt_discriminant) / (2 * a)
  root1 > root2 ∧ root1 - root2 = 5.5 :=
by
  let a := 2
  let b := 5
  let c := -12
  let discriminant := b ^ 2 - 4 * a * c
  have h_discriminant : discriminant = 121 := by sorry
  have h_sqrt_discriminant : sqrt discriminant = 11 := by sorry
  let root1 := (-b + sqrt discriminant) / (2 * a)
  let root2 := (-b - sqrt discriminant) / (2 * a)
  have h_root1 : root1 = 1.5 := by sorry
  have h_root2 : root2 = -4 := by sorry
  have h1 : root1 > root2 := by linarith
  have h2 : root1 - root2 = 5.5 := by linarith
  exact ⟨h1, h2⟩

end larger_root_exceeds_smaller_l787_787139


namespace simplify_and_evaluate_expression_l787_787782

theorem simplify_and_evaluate_expression (x y : ℚ) (h_x : x = -2) (h_y : y = 1/2) :
  (x + 2 * y)^2 - (x + y) * (x - y) = -11/4 := by
  sorry

end simplify_and_evaluate_expression_l787_787782


namespace difference_in_ages_l787_787105

/-- Definitions: --/
def sum_of_ages (B J : ℕ) := B + J = 70
def jennis_age (J : ℕ) := J = 19

/-- Theorem: --/
theorem difference_in_ages : ∀ (B J : ℕ), sum_of_ages B J → jennis_age J → B - J = 32 :=
by
  intros B J hsum hJ
  rw [jennis_age] at hJ
  rw [sum_of_ages] at hsum
  sorry

end difference_in_ages_l787_787105


namespace vertex_x_of_parabola_l787_787080

theorem vertex_x_of_parabola 
  (a b c : ℝ)
  (h₁ : (λ x : ℝ, a * x^2 + b * x + c) (-2) = 8)
  (h₂ : (λ x : ℝ, a * x^2 + b * x + c) 4 = 8)
  (h₃ : (λ x : ℝ, a * x^2 + b * x + c) 7 = 15) :
  ∃ x_vertex, x_vertex = 1 :=
by
  sorry

end vertex_x_of_parabola_l787_787080


namespace Sheila_attendance_probability_l787_787399

-- Definitions as per given conditions
def P_rain := 0.5
def P_sunny := 0.3
def P_cloudy := 0.2
def P_Sheila_goes_given_rain := 0.3
def P_Sheila_goes_given_sunny := 0.7
def P_Sheila_goes_given_cloudy := 0.5

-- Define the probability calculation
def P_Sheila_attends := 
  (P_rain * P_Sheila_goes_given_rain) + 
  (P_sunny * P_Sheila_goes_given_sunny) + 
  (P_cloudy * P_Sheila_goes_given_cloudy)

-- Final theorem statement
theorem Sheila_attendance_probability : P_Sheila_attends = 0.46 := by
  sorry

end Sheila_attendance_probability_l787_787399


namespace no_integer_roots_l787_787046

theorem no_integer_roots (x : ℤ) : ¬ (x^2 + 2^2018 * x + 2^2019 = 0) :=
sorry

end no_integer_roots_l787_787046


namespace total_trees_l787_787925

-- Definitions based on the conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3

-- Theorem stating the total number of apple trees planted by Ava and Lily
theorem total_trees : ava_trees + lily_trees = 15 := by
  -- We skip the proof for now
  sorry

end total_trees_l787_787925


namespace notebooks_difference_l787_787760

noncomputable def price_more_than_dime (p : ℝ) : Prop := p > 0.10
noncomputable def payment_equation (nL nN : ℕ) (p : ℝ) : Prop :=
  (nL * p = 2.10 ∧ nN * p = 2.80)

theorem notebooks_difference (nL nN : ℕ) (p : ℝ) (h1 : price_more_than_dime p) (h2 : payment_equation nL nN p) :
  nN - nL = 2 :=
by sorry

end notebooks_difference_l787_787760


namespace no_triangles_with_geometric_progression_angles_l787_787671

theorem no_triangles_with_geometric_progression_angles :
  ¬ ∃ (a r : ℕ), a ≥ 10 ∧ (a + a * r + a * r^2 = 180) ∧ (a ≠ a * r) ∧ (a ≠ a * r^2) ∧ (a * r ≠ a * r^2) :=
sorry

end no_triangles_with_geometric_progression_angles_l787_787671


namespace kamal_marks_in_english_l787_787348

theorem kamal_marks_in_english :
  ∀ (E Math Physics Chemistry Biology Average : ℕ), 
    Math = 65 → 
    Physics = 82 → 
    Chemistry = 67 → 
    Biology = 85 → 
    Average = 79 → 
    (Math + Physics + Chemistry + Biology + E) / 5 = Average → 
    E = 96 :=
by
  intros E Math Physics Chemistry Biology Average
  intros hMath hPhysics hChemistry hBiology hAverage hTotal
  sorry

end kamal_marks_in_english_l787_787348


namespace frogs_need_new_pond_l787_787884

theorem frogs_need_new_pond
  (num_frogs : ℕ) 
  (num_tadpoles : ℕ) 
  (num_survivor_tadpoles : ℕ) 
  (pond_capacity : ℕ) 
  (hc1 : num_frogs = 5)
  (hc2 : num_tadpoles = 3 * num_frogs)
  (hc3 : num_survivor_tadpoles = (2 * num_tadpoles) / 3)
  (hc4 : pond_capacity = 8):
  ((num_frogs + num_survivor_tadpoles) - pond_capacity) = 7 :=
by sorry

end frogs_need_new_pond_l787_787884


namespace evaluate_P_at_8_l787_787756

noncomputable def P : Polynomial ℝ :=
  12 * (X - 2) * (X - 3) * (X - 4) * (X - 6)^2 * (X - 7)^2 * (X - 1)

theorem evaluate_P_at_8 :
  P.eval 8 = 483840 :=
by 
  -- The proof will go here if it's necessary
  sorry

end evaluate_P_at_8_l787_787756


namespace probability_of_diff_ge_3_l787_787460

noncomputable def probability_diff_ge_3 : ℚ :=
  let elements := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let total_pairs := (elements.to_finset.card.choose 2 : ℕ)
  let valid_pairs := (total_pairs - 15 : ℕ) -- 15 pairs with a difference of less than 3
  valid_pairs / total_pairs

theorem probability_of_diff_ge_3 :
  probability_diff_ge_3 = 7 / 12 := by
  sorry

end probability_of_diff_ge_3_l787_787460


namespace min_value_ineq_l787_787745

noncomputable def min_value (xs : Fin 50 → ℝ) : ℝ :=
  ∑ i, xs i / (1 - (xs i)^2)

theorem min_value_ineq (xs : Fin 50 → ℝ) (hpos : ∀ i, 0 < xs i) 
  (hsum : ∑ i, (xs i)^2 = 2) : 
  min_value xs ≥ 3 * Real.sqrt 3 :=
by 
  sorry

end min_value_ineq_l787_787745


namespace find_EA_l787_787698

variables (R E D A C M : Point)
variables (angle_DRE angle_RED : ℝ)
variables (RD : ℝ)

axiom angle_DRE_eq_60 : angle_DRE = 60
axiom angle_RED_eq_60 : angle_RED = 60
axiom RD_eq_2 : RD = 2
axiom M_midpoint_RD : M = midpoint R D
axiom RC_perp_ED : Perpendicular (line R C) (line E D)
axiom EA_3AD : ∃ AD, AD = distance E D ∧ EA = 3 * AD

theorem find_EA : EA = 6 :=
begin
  -- proof goes here
  sorry
end

end find_EA_l787_787698


namespace train_length_calculation_l787_787179

def speed_kmph := 60 -- speed in kilometers per hour
def time_seconds := 15 -- time in seconds
def platform_length := 130.02 -- length of the platform in meters

def kmph_to_mps (kmph: ℝ) : ℝ := (kmph * 1000) / 3600

theorem train_length_calculation :
  let speed_mps := kmph_to_mps speed_kmph in
  let distance_covered := speed_mps * time_seconds in
  let train_length := distance_covered - platform_length in
  train_length = 119.98 :=
by
  sorry

end train_length_calculation_l787_787179


namespace monochromatic_triangle_in_K6_l787_787703

theorem monochromatic_triangle_in_K6 :
  ∀ (color : Fin 6 → Fin 6 → Prop),
  (∀ (a b : Fin 6), a ≠ b → (color a b ↔ color b a)) →
  (∃ (x y z : Fin 6), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (color x y = color y z ∧ color y z = color z x)) :=
by
  sorry

end monochromatic_triangle_in_K6_l787_787703


namespace probability_of_three_correct_deliveries_l787_787965

-- Define a combination function
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the factorial function
def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the problem with conditions and derive the required probability
theorem probability_of_three_correct_deliveries :
  (combination 5 3) / (factorial 5) = 1 / 12 := by
  sorry

end probability_of_three_correct_deliveries_l787_787965


namespace find_m_plus_n_l787_787205

    noncomputable def parallelogram_slope : ℤ × ℤ :=
    let vertices := [(12, 50), (12, 120), (30, 160), (30, 90)] in
    let slope := (5 : ℚ) in -- 5 = 5/1
    let m := 5 in
    let n := 1 in
    (m, n)

    theorem find_m_plus_n : let (m, n) := parallelogram_slope in m + n = 6 :=
    by
    rw [parallelogram_slope]
    -- from the definition of parallelogram_slope
    show (5, 1).fst + (5, 1).snd = 6
    rfl
    
end find_m_plus_n_l787_787205


namespace trig_expression_value_l787_787297

theorem trig_expression_value (θ : ℝ) (h : Real.tan θ = 2) : 
  (2 * Real.cos θ) / (Real.sin (Real.pi / 2 + θ) + Real.sin (Real.pi + θ)) = -2 := 
by 
  sorry

end trig_expression_value_l787_787297


namespace solve_equation_l787_787796

theorem solve_equation : ∃ x : ℝ, (x^3 - ⌊x⌋ = 3) := 
sorry

end solve_equation_l787_787796


namespace find_general_equation_of_line_l787_787832

variables {x y k b : ℝ}

-- Conditions: slope of the line is -2 and sum of its intercepts is 12.
def slope_of_line (l : ℝ → ℝ → Prop) : Prop := ∃ b, ∀ x y, l x y ↔ y = -2 * x + b
def sum_of_intercepts (l : ℝ → ℝ → Prop) : Prop := ∃ b, b + (b / 2) = 12

-- Question: What is the general equation of the line?
noncomputable def general_equation (l : ℝ → ℝ → Prop) : Prop :=
  slope_of_line l ∧ sum_of_intercepts l → ∀ x y, l x y ↔ 2 * x + y - 8 = 0

-- The theorem we need to prove
theorem find_general_equation_of_line (l : ℝ → ℝ → Prop) : general_equation l :=
sorry

end find_general_equation_of_line_l787_787832


namespace largest_beautiful_less_100_l787_787899

def is_beautiful (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : Fin k → ℕ), k > 1 ∧ (∀ i j, i ≤ j → a i ≥ a j) ∧ n = (∑ i, a i) ∧ n = (∏ i, a i) ∧ ∀ (b : Fin k → ℕ), (n = (∑ i, b i) ∧ n = (∏ i, b i)) → (∀ i, a i = b i)

theorem largest_beautiful_less_100 : ∀ m : ℕ, (m < 100 → is_beautiful m) → m ≤ 95 :=
by
  sorry

end largest_beautiful_less_100_l787_787899


namespace average_payment_is_correct_l787_787875

theorem average_payment_is_correct:
  let first_payment := 500
  let first_count := 25
  let remaining_payment := first_payment + 100
  let remaining_count := 52 - first_count
  let total_first := first_count * first_payment
  let total_remaining := remaining_count * remaining_payment
  let total_payment := total_first + total_remaining
  let number_of_payments := first_count + remaining_count 
  average_payment (total_payment / number_of_payments : ℝ) = 551.92 :=
by
  sorry

end average_payment_is_correct_l787_787875


namespace over_limit_weight_l787_787340

variable (hc_books : ℕ) (hc_weight : ℕ → ℝ)
variable (tex_books : ℕ) (tex_weight : ℕ → ℝ)
variable (knick_knacks : ℕ) (knick_weight : ℕ → ℝ)
variable (weight_limit : ℝ)

axiom hc_books_value : hc_books = 70
axiom hc_weight_value : hc_weight hc_books = 0.5
axiom tex_books_value : tex_books = 30
axiom tex_weight_value : tex_weight tex_books = 2
axiom knick_knacks_value : knick_knacks = 3
axiom knick_weight_value : knick_weight knick_knacks = 6
axiom weight_limit_value : weight_limit = 80

theorem over_limit_weight : 
  (hc_books * hc_weight hc_books + tex_books * tex_weight tex_books + knick_knacks * knick_weight knick_knacks) - weight_limit = 33 := by
  sorry

end over_limit_weight_l787_787340


namespace ten_digit_numbers_divisible_by_72_l787_787113

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

-- Define the given number
def given_number : ℕ := 20192020

-- Helper function to calculate the sum of digits of a number
def digit_sum (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Helper function to form a 10-digit number by prepending and appending digits
def form_number (prepend append : ℕ) : ℕ := (prepend * 10^9) + (given_number * 10) + append

-- The main problem statement to prove
theorem ten_digit_numbers_divisible_by_72 (prepend1 prepend2 append1 append2 : ℕ) :
  form_number 2 0 = 2201920200 ∧ form_number 3 8 = 3201920208 ∧
  (is_divisible_by (form_number prepend1 append1) 72 ∧ is_divisible_by (form_number prepend2 append2) 72) :=
by
  -- Explicitly state the append and prepend
  assume (prepend1 = 2) (append1 = 0) (prepend2 = 3) (append2 = 8),
  -- State that the numbers formed are as expected
  have h1 : form_number 2 0 = 2201920200, from rfl,
  have h2 : form_number 3 8 = 3201920208, from rfl,
  -- State the divisibility checks
  have h3 : is_divisible_by 2201920200 72, sorry,
  have h4 : is_divisible_by 3201920208 72, sorry,
  exact ⟨h1, h2, ⟨h3, h4⟩⟩

end ten_digit_numbers_divisible_by_72_l787_787113


namespace Rachel_brought_25_cookies_l787_787765

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Total_cookies : ℕ := 60

theorem Rachel_brought_25_cookies : (Total_cookies - (Mona_cookies + Jasmine_cookies) = 25) :=
by
  sorry

end Rachel_brought_25_cookies_l787_787765


namespace frogs_needing_new_pond_l787_787888

theorem frogs_needing_new_pond : 
    (initial_frogs tadpoles_survival_fraction pond_capacity : ℕ) 
    (h1 : initial_frogs = 5) 
    (h2 : tadpoles_survival_fraction = 2 / 3) 
    (h3 : pond_capacity = 8) 
    : (initial_frogs + tadpoles_survival_fraction * 3 * initial_frogs - pond_capacity = 7) :=
by
    sorry

end frogs_needing_new_pond_l787_787888


namespace probability_area_greater_than_half_circumference_l787_787533

theorem probability_area_greater_than_half_circumference :
  (∑ r in {1, 2, 3, 4, 5, 6}.toFinset.filter (λ r, r > 1), (1/6 : ℝ)) = 5/6 :=
by
  -- here goes the proof
  sorry

end probability_area_greater_than_half_circumference_l787_787533


namespace contrapositive_sine_l787_787484

/-- Prove that the contrapositive of "If \(x = y\), then \(\sin x = \sin y\)" is a true proposition. --/
theorem contrapositive_sine (x y : ℝ) : ¬ (sin x = sin y) → ¬ (x = y) :=
by
  sorry

end contrapositive_sine_l787_787484


namespace composite_sequence_exists_l787_787998

theorem composite_sequence_exists (a n : ℕ) (h_a : a ≥ 2) (h_n : n > 0) : 
  ∃ k > 0, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (¬ Nat.prime (a^k + i)) :=
by
  sorry

end composite_sequence_exists_l787_787998


namespace find_ordered_pair_l787_787798

theorem find_ordered_pair {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : ∀ x, x^2 + a * x + b = 0 ↔ x = -2 * a ∨ x = b)
  (h2 : b = -2 * -2 * a) : (a, b) = (-1/2, -1/2) :=
by
  sorry

end find_ordered_pair_l787_787798


namespace exist_2005_numbers_l787_787223

theorem exist_2005_numbers :
  ∃ (a : ℕ → ℕ), (∀ n, n < 2005 → a n ∈ ℕ) ∧ (∀ i j, i < j → a i ≠ a j) ∧
  (∀ i, i < 2005 → ((Finset.range 2005).erase i).sum (λ j, a j) % a i = 0) :=
sorry

end exist_2005_numbers_l787_787223


namespace sin_2x_is_4_div_5_l787_787616

variables {x : ℝ}

def a := (Real.cos x, -2)
def b := (Real.sin x, 1)
def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

theorem sin_2x_is_4_div_5 (hx : parallel a b) : Real.sin (2 * x) = 4 / 5 := by
  sorry

end sin_2x_is_4_div_5_l787_787616


namespace proof_problem_l787_787239

-- Define the operation
def star (a b : ℝ) : ℝ := (a - b) ^ 2

-- The proof problem as a Lean statement
theorem proof_problem (x y : ℝ) : star ((x - y) ^ 2 + 1) ((y - x) ^ 2 + 1) = 0 := by
  sorry

end proof_problem_l787_787239


namespace Danny_soda_left_l787_787575

theorem Danny_soda_left (bottles : ℕ) (drink_percent : ℝ) (gift_percent : ℝ) : 
  bottles = 3 → drink_percent = 0.9 → gift_percent = 0.7 → 
  let remaining1 := 1 - drink_percent in
  let remaining2 := 1 - gift_percent in
  (remaining1 + 2 * remaining2) = 0.7 := 
by
  intros h_bottles h_drink h_gift
  rw [h_bottles, h_drink, h_gift]
  let lhs := (1 - 0.9) + 2 * (1 - 0.7)
  have h : lhs = 0.7 := by 
    calc lhs = 0.1 + 2 * 0.3 : by rfl
        ... = 0.1 + 0.6 : by norm_num
        ... = 0.7 : by norm_num
  exact h

end Danny_soda_left_l787_787575


namespace quadrilateral_with_four_right_angles_is_plane_l787_787550

def quadrilateral (sides : ℕ) (angles : list ℕ) : Prop :=
  sides = 4 ∧ angles.length = 4 ∧ angles.sum = 360

def is_plane_figure (shape : Prop) : Prop :=
  shape

theorem quadrilateral_with_four_right_angles_is_plane :
  ∀ (angles : list ℕ), (quadrilateral 4 angles) ∧ 
  (∀ (a : ℕ), a ∈ angles → a = 90) → is_plane_figure (quadrilateral 4 angles) :=
by
  intros angles h
  sorry

end quadrilateral_with_four_right_angles_is_plane_l787_787550


namespace average_speed_jeffrey_l787_787346
-- Import the necessary Lean library.

-- Initial conditions in the problem, restated as Lean definitions.
def distance_jog (d : ℝ) : Prop := d = 3
def speed_jog (s : ℝ) : Prop := s = 4
def distance_walk (d : ℝ) : Prop := d = 4
def speed_walk (s : ℝ) : Prop := s = 3

-- Target statement to prove using Lean.
theorem average_speed_jeffrey :
  ∀ (dj sj dw sw : ℝ), distance_jog dj → speed_jog sj → distance_walk dw → speed_walk sw →
    (dj + dw) / ((dj / sj) + (dw / sw)) = 3.36 := 
  by
    intros dj sj dw sw hj hs hw hw
    sorry

end average_speed_jeffrey_l787_787346


namespace perimeter_square_l787_787073

noncomputable def side_length (s : ℝ) :=
  s^2 = s * real.sqrt 2

theorem perimeter_square (s : ℝ) (h : side_length s) : 
  4 * s = 4 * real.sqrt 2 :=
by
  sorry

end perimeter_square_l787_787073


namespace find_a_for_square_of_binomial_l787_787294

-- Primary Problem Statement
theorem find_a_for_square_of_binomial :
  ∃ a : ℚ, (∀ x : ℚ, a * x^2 - 25 * x + 9 = (-(25 / 6) * x + 3)^2) :=
begin
  use (625 / 36),
  sorry
end

end find_a_for_square_of_binomial_l787_787294


namespace proof_ellipse_eccentricity_l787_787996

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h : a > b ∧ b > 0) : ℝ :=
  if (a - b) = 0 then sorry else
  let c := Math.sqrt (a^2 - b^2) in
  c / a

theorem proof_ellipse_eccentricity {a b : ℝ} (h : a > b ∧ b > 0)
  (ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (radius : ℝ) (vertices : ℝ) (focus : ℝ)
  (symmetric_point : ℝ → ℝ → ℝ → ℝ)
  (circle_radius : ℝ) : eccentricity_of_ellipse a b h = 1 / 2 :=
sorry

end proof_ellipse_eccentricity_l787_787996


namespace train_length_l787_787537

theorem train_length (v_train : ℝ) (v_man : ℝ) (time : ℝ) (length : ℝ) 
    (hv_train : v_train = 60) (hv_man : v_man = 6) (htime : time = 11.999040076793857) 
    (hlength : length = 220) : 
    v_train + v_man = 66 → 
    66 * (5 / 18) = 18.33333333333333 → 
    18.33333333333333 * time = length :=
begin
  sorry
end

end train_length_l787_787537


namespace base10_to_base7_of_804_l787_787469

def base7 (n : ℕ) : ℕ :=
  let d3 := n / 343
  let r3 := n % 343
  let d2 := r3 / 49
  let r2 := r3 % 49
  let d1 := r2 / 7
  let d0 := r2 % 7
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

theorem base10_to_base7_of_804 :
  base7 804 = 2226 :=
by
  -- Proof to be filled in.
  sorry

end base10_to_base7_of_804_l787_787469


namespace sector_area_correct_l787_787072

-- Definition of the radius and the central angle
def r : ℝ := 1
def alpha : ℝ := 2 * Real.pi / 3

-- Definition of the area formula for a sector
def sector_area (r alpha : ℝ) : ℝ := (1 / 2) * alpha * r ^ 2

-- Statement of the theorem to prove
theorem sector_area_correct : sector_area r alpha = Real.pi / 3 := by
  -- we replace the proof with "sorry" as per instructions
  sorry

end sector_area_correct_l787_787072


namespace translation_preserves_parallel_and_equal_length_l787_787543

theorem translation_preserves_parallel_and_equal_length
    (A B C D : ℝ)
    (after_translation : (C - A) = (D - B))
    (connecting_parallel : C - A = D - B) :
    (C - A = D - B) ∧ (C - A = D - B) :=
by
  sorry

end translation_preserves_parallel_and_equal_length_l787_787543


namespace problem1_problem2_l787_787246

variable (α : Real) -- Define α as a real variable

-- Define the conditions
def condition1 : Prop := sin α = -2 * sqrt 5 / 5
def condition2 : Prop := tan α < 0

-- Problem 1: Prove tan α = -2 given the conditions
theorem problem1 (h1 : condition1 α) (h2 : condition2 α) : tan α = -2 := 
sorry

-- Problem 2: Prove the given trigonometric expression equals -5 under the same conditions
theorem problem2 (h1 : condition1 α) (h2 : condition2 α) : 
  (2 * sin (α + π) + cos (2 * π - α)) / (cos (α - π / 2) - sin (3 * π / 2 + α)) = -5 := 
sorry

end problem1_problem2_l787_787246


namespace track_length_is_correct_l787_787779

noncomputable def length_of_track (v : ℝ) : ℝ := 
  let speed_uphill := v / 4
  let min_distance := 4
  let max_distance := 13
  let distance_difference := max_distance - min_distance
  
  -- Prove that the total length of the track is 2 * distance_difference
  2 * distance_difference

theorem track_length_is_correct (v : ℝ) : length_of_track v = 24 := by
  let speed_uphill := v / 4
  let min_distance := 4
  let max_distance := 13
  let distance_difference := max_distance - min_distance

  -- There is no actual need to prove intermediate steps if we are posing the theorem statement
  -- since we only need the final length based on the given relational data
  -- Hence we assert the conclusion directly
  show 2 * distance_difference = 24
  sorry

end track_length_is_correct_l787_787779


namespace coeff_x9_in_expansion_l787_787470

-- Define the binomial theorem and the expansion
theorem coeff_x9_in_expansion :
  let a := 3
  let b := -5 * x ^ 3
  let n := 4
  ∑ (k : ℕ) in finset.range (n + 1), (nat.choose n k) * (a^(n - k)) * (b ^ k) = (3 - 5 * (x ^ 3)) ^ 4 →
  -- The coefficient term for x^9
  ((nat.choose 4 3) * (3 ^ (4 - 3)) * ((-5) ^ 3)) = -1500 :=
by
  sorry

end coeff_x9_in_expansion_l787_787470


namespace g_is_odd_l787_787724

def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 2)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_is_odd_l787_787724


namespace sum_of_intervals_l787_787607

noncomputable def floor_function (x : ℝ) : ℤ := int.floor x

noncomputable def f (x : ℝ) : ℝ :=
  floor_function x * (2015 ^ (x - floor_function x) - 1)

theorem sum_of_intervals :
  ∑ i in (finset.range 2014), (0 : ℝ) = 0 :=
by sorry

end sum_of_intervals_l787_787607


namespace Courtney_total_marbles_l787_787210

theorem Courtney_total_marbles (first_jar second_jar third_jar : ℕ) 
  (h1 : first_jar = 80)
  (h2 : second_jar = 2 * first_jar)
  (h3 : third_jar = first_jar / 4) :
  first_jar + second_jar + third_jar = 260 := 
by
  sorry

end Courtney_total_marbles_l787_787210


namespace functions_identical_l787_787483

theorem functions_identical :
  (∀ x : ℝ, abs x = sqrt (x^2)) ∧ (∀ x : ℝ, ∀ t : ℝ, x^2 - 2*x - 1 = t^2 - 2*t - 1) :=
by
  sorry

end functions_identical_l787_787483


namespace coefficient_x_squared_l787_787639

theorem coefficient_x_squared (n : ℕ) (h : (1 + 3 * X)^n = expand $ coeff (x^2)) : n = 4 := by
  sorry

end coefficient_x_squared_l787_787639


namespace solve_for_z_l787_787786

theorem solve_for_z (z : ℂ) (h : 4 + (2 * complex.I)*z = 3 - (5 * complex.I)*z) : 
  z = complex.I / 7 := 
sorry

end solve_for_z_l787_787786


namespace probability_same_color_opposite_foot_is_correct_l787_787764

noncomputable def probability_same_color_opposite_foot : ℚ :=
  let total_shoes := 32
  let total_combinations := 32.choose 2
  let black_probability := (16 * 8) / (total_shoes * (total_shoes - 1))
  let brown_probability := (8 * 4) / (total_shoes * (total_shoes - 1))
  let grey_probability := (6 * 3) / (total_shoes * (total_shoes - 1))
  let white_probability := (2 * 1) / (total_shoes * (total_shoes - 1))
  (black_probability + brown_probability + grey_probability + white_probability)

theorem probability_same_color_opposite_foot_is_correct :
  probability_same_color_opposite_foot = (45 / 248) := by
  sorry

end probability_same_color_opposite_foot_is_correct_l787_787764


namespace determine_number_of_shelves_l787_787439

-- Define the total distance Karen bikes round trip
def total_distance : ℕ := 3200

-- Define the number of books per shelf
def books_per_shelf : ℕ := 400

-- Calculate the one-way distance from Karen's home to the library
def one_way_distance (total_distance : ℕ) : ℕ := total_distance / 2

-- Define the total number of books, which is the same as the one-way distance
def total_books (one_way_distance : ℕ) : ℕ := one_way_distance

-- Calculate the number of shelves
def number_of_shelves (total_books : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books / books_per_shelf

theorem determine_number_of_shelves :
  number_of_shelves (total_books (one_way_distance total_distance)) books_per_shelf = 4 :=
by 
  -- the proof would go here
  sorry

end determine_number_of_shelves_l787_787439


namespace difference_divisible_by_10_l787_787393

theorem difference_divisible_by_10 : (43 ^ 43 - 17 ^ 17) % 10 = 0 := by
  sorry

end difference_divisible_by_10_l787_787393


namespace alfred_maize_storing_years_l787_787544

theorem alfred_maize_storing_years :
  ∃ y : ℕ, (12 * y - 5 + 8 = 27) ∧ y = 2 :=
by
  use 2
  split
  · linarith
  · rfl

end alfred_maize_storing_years_l787_787544


namespace frogs_need_new_pond_l787_787885

theorem frogs_need_new_pond
  (num_frogs : ℕ) 
  (num_tadpoles : ℕ) 
  (num_survivor_tadpoles : ℕ) 
  (pond_capacity : ℕ) 
  (hc1 : num_frogs = 5)
  (hc2 : num_tadpoles = 3 * num_frogs)
  (hc3 : num_survivor_tadpoles = (2 * num_tadpoles) / 3)
  (hc4 : pond_capacity = 8):
  ((num_frogs + num_survivor_tadpoles) - pond_capacity) = 7 :=
by sorry

end frogs_need_new_pond_l787_787885


namespace smallest_n_divisible_by_2022_l787_787800

theorem smallest_n_divisible_by_2022 (n : ℕ) (h1 : n > 1) (h2 : (n^7 - 1) % 2022 = 0) : n = 79 :=
sorry

end smallest_n_divisible_by_2022_l787_787800


namespace train_problem_l787_787055

theorem train_problem (Sat M S C : ℕ) 
  (h_boarding_day : true)
  (h_arrival_day : true)
  (h_date_matches_car_on_monday : M = C)
  (h_seat_less_than_car : S < C)
  (h_sat_date_greater_than_car : Sat > C) :
  C = 2 ∧ S = 1 :=
by sorry

end train_problem_l787_787055


namespace price_of_other_frisbees_l787_787908

theorem price_of_other_frisbees 
  (P : ℝ) 
  (x : ℝ)
  (h1 : x + (64 - x) = 64)
  (h2 : P * x + 4 * (64 - x) = 196)
  (h3 : 64 - x ≥ 4) 
  : P = 3 :=
sorry

end price_of_other_frisbees_l787_787908


namespace maximize_perimeter_l787_787770

open Real

namespace TrianglePerimeterMaximization

/--
For a fixed chord AB of a circle and a point P on the arc AB,
the perimeter of triangle ABP is maximized when P is the midpoint
of the arc AB.
-/
theorem maximize_perimeter (A B P : Point) (r : ℝ) (h_circle : Circle A r) (h_on_arc : OnArc P A B) (h_midpoint : Midpoint P A B) :
  maximize_perimeter_triangle A B P = maximize_perimeter_triangle A B midpoint(A, B) :=
sorry

end TrianglePerimeterMaximization

end maximize_perimeter_l787_787770


namespace max_terms_arithmetic_sequence_fixed_sum_l787_787751

theorem max_terms_arithmetic_sequence_fixed_sum (a : ℕ → ℤ) (n : ℕ) 
  (h_arith_seq : ∀ i, a (i + 1) - a i = a 2 - a 1)
  (h_sum_condition : ∀ j : ℤ, j ∈ {0, 1, 2, 3} → (finset.range n).sum (λ i, |a i + j|) = 2028) :
  n ≤ 52 :=
sorry

end max_terms_arithmetic_sequence_fixed_sum_l787_787751


namespace trapezoid_longer_side_length_l787_787172

theorem trapezoid_longer_side_length (x : ℝ) :
  let side := 1 in
  let trapezoid_area := 1 / 3 in
  let height := 1 / 2 in
  let sum_of_parallel_sides := x + 1 / 2 in
  side * side = 1 →
  (2 * trapezoid_area + trapezoid_area = side * side) →
  (trapezoid_area = 1 / 2 * sum_of_parallel_sides * height) →
  x = 5 / 6 :=
by
  sorry

end trapezoid_longer_side_length_l787_787172


namespace parallel_vectors_2x_plus_y_l787_787247

noncomputable def vec_a (x : ℝ) : ℝ × ℝ × ℝ := (2, 1, x)
noncomputable def vec_b (y : ℝ) : ℝ × ℝ × ℝ := (4, y, -1)

def are_parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2, k * b.3)

theorem parallel_vectors_2x_plus_y (x y : ℝ) : are_parallel (vec_a x) (vec_b y) → 2 * x + y = 1 :=
by
  sorry

end parallel_vectors_2x_plus_y_l787_787247


namespace perpendicular_lines_parallel_l787_787240

noncomputable def line := Type
noncomputable def plane := Type

variables (m n : line) (α : plane)

def parallel (l1 l2 : line) : Prop := sorry -- Definition of parallel lines
def perpendicular (l : line) (α : plane) : Prop := sorry -- Definition of perpendicular line to a plane

theorem perpendicular_lines_parallel (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end perpendicular_lines_parallel_l787_787240


namespace expected_visible_people_l787_787148

theorem expected_visible_people (n : ℕ) (h : n > 0) : 
  (∑ i in Finset.range n, 1 / (i + 1 : ℝ)) = 1 + 1 / 2 + 1 / 3 + ... + 1 / n :=
sorry

end expected_visible_people_l787_787148


namespace can_transport_l787_787704

noncomputable def max_load_possible : ℝ :=
  28 / 5

theorem can_transport (W : Type) [linear_ordered_field W] (P : W → W) (pt : set W) :
  (∀ x ∈ pt, 0 < x ∧ x ≤ 1) →
  (∃ f g : W → ℕ, 
   ∀ x ∈ pt, 
    let truck1_load := (f x) * 3,
        truck2_load := (g x) * 4 in
    (truck1_load + truck2_load) / pt.card ≥ max_load_possible) :=
begin
  sorry
end

end can_transport_l787_787704


namespace triangle_inequality_l787_787976

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a / (b + c) + b / (c + a) + c / (a + b) ≤ 2 :=
sorry

end triangle_inequality_l787_787976


namespace math_proof_l787_787681

noncomputable def proof_problem : Prop :=
  ∃ x : ℝ, log 6 (4 * x) = 2 ∧ log x 27 = 3 / 2

theorem math_proof (h : ∃ x : ℝ, log 6 (4 * x) = 2) : ∃ x : ℝ, log x 27 = 3 / 2 := 
by
  sorry

end math_proof_l787_787681


namespace total_pages_read_is_785_l787_787373

-- Definitions based on the conditions in the problem
def pages_read_first_five_days : ℕ := 5 * 52
def pages_read_next_five_days : ℕ := 5 * 63
def pages_read_last_three_days : ℕ := 3 * 70

-- The main statement to prove
theorem total_pages_read_is_785 :
  pages_read_first_five_days + pages_read_next_five_days + pages_read_last_three_days = 785 :=
by
  sorry

end total_pages_read_is_785_l787_787373


namespace max_k_sqrt_expr_l787_787471

theorem max_k_sqrt_expr :
  ∃ k, (∀ x, 3 ≤ x ∧ x ≤ 6 → sqrt (x - 3) + sqrt (6 - x) ≥ k) ∧ k = sqrt 6 :=
sorry

end max_k_sqrt_expr_l787_787471


namespace Courtney_total_marbles_l787_787212

theorem Courtney_total_marbles (first_jar second_jar third_jar : ℕ) 
  (h1 : first_jar = 80)
  (h2 : second_jar = 2 * first_jar)
  (h3 : third_jar = first_jar / 4) :
  first_jar + second_jar + third_jar = 260 := 
by
  sorry

end Courtney_total_marbles_l787_787212


namespace calculator_prices_and_relations_l787_787910

theorem calculator_prices_and_relations :
  ∃ (x y : ℝ), 
  (2 * x + 3 * y = 156) ∧
  (3 * x + y = 122) ∧
  (x = 30) ∧
  (y = 32) ∧
  (∀ (z : ℝ), y1 z = 24 * z) ∧
  (∀ (z : ℝ), y2 z = 28 * z) :=
sorry

end calculator_prices_and_relations_l787_787910


namespace find_yellow_flowers_l787_787763

def flower_counts (Y P G : ℕ) : Prop :=
  P = 1.8 * Y ∧ G = 0.25 * (Y + P) ∧ Y + P + G = 35

theorem find_yellow_flowers {Y P G : ℕ} 
  (h : flower_counts Y P G) : Y = 10 :=
by sorry

end find_yellow_flowers_l787_787763


namespace ratio_of_shaded_and_white_area_l787_787125

-- Let us define the setup of the problem
def is_midpoint (a b c : Point) : Prop :=
  midpoint a b = c

def is_shaded_area_half_white_area {α : Type} [metric_space α] 
  (A B C D E F G H I J K L M N O P : α) : Prop :=
  is_midpoint A B C ∧ is_midpoint B C D ∧ 
  is_midpoint E F G ∧ is_midpoint F G H ∧ 
  is_midpoint I J K ∧ is_midpoint J K L ∧ 
  is_midpoint M N O ∧ is_midpoint N O P ∧
  -- Shaded triangles count versus white triangles count
  (shaded_triangles_count = 5) ∧ (white_triangles_count = 3)

theorem ratio_of_shaded_and_white_area {α : Type} [metric_space α] 
  (A B C D E F G H I J K L M N O P : α) 
  (h : is_shaded_area_half_white_area A B C D E F G H I J K L M N O P) :
  (shaded_area / white_area) = (5 / 3) :=
sorry

end ratio_of_shaded_and_white_area_l787_787125


namespace tricycles_count_l787_787321

theorem tricycles_count (b t : ℕ) 
  (hyp1 : b + t = 10)
  (hyp2 : 2 * b + 3 * t = 26) : 
  t = 6 := 
by 
  sorry

end tricycles_count_l787_787321


namespace compute_final_avg_l787_787068

def avg2 (a b : ℝ) : ℝ := (a + b) / 2
def avg4 (a b c d : ℝ) : ℝ := (a + b + c + d) / 4

theorem compute_final_avg : avg4 (avg2 2 4) (avg2 1 3) (avg2 0 2) (avg2 1 1) = (35 / 16) :=
by
  sorry

end compute_final_avg_l787_787068


namespace stock_percentage_correct_l787_787326

-- Definitions based on conditions
def investment_amount : ℝ := 4590
def price_per_unit : ℝ := 102
def desired_income : ℝ := 900

-- The problem statement translated to Lean
theorem stock_percentage_correct :
  (desired_income / (investment_amount / price_per_unit) / price_per_unit) * 100 ≈ 19.61 :=
by
  sorry

end stock_percentage_correct_l787_787326


namespace small_bottles_sold_percentage_l787_787907

theorem small_bottles_sold_percentage
  (small_bottles : ℕ) (big_bottles : ℕ) (percent_sold_big_bottles : ℝ)
  (remaining_bottles : ℕ) (percent_sold_small_bottles : ℝ) :
  small_bottles = 6000 ∧
  big_bottles = 14000 ∧
  percent_sold_big_bottles = 0.23 ∧
  remaining_bottles = 15580 ∧ 
  percent_sold_small_bottles / 100 * 6000 + 0.23 * 14000 + remaining_bottles = small_bottles + big_bottles →
  percent_sold_small_bottles = 37 := 
by
  intros
  exact sorry

end small_bottles_sold_percentage_l787_787907


namespace geometric_sequence_arithmetic_progression_ratio_l787_787985

theorem geometric_sequence_arithmetic_progression_ratio :
  ∀ {a : ℕ → ℝ} {q : ℝ},
  (∀ n, a(n + 1) = q * a(n)) → -- condition that sequence is geometric
  (a(1) > 0) → -- condition that the terms are positive
  (2 * (1 / 2 * a(3)) = a(1) + 2 * a(2)) → -- condition that a_1, (1/2)a_3, and 2a_2 form an arithmetic sequence
  (q = 1 + Real.sqrt 2) → -- solved value for the common ratio q
  (a(9) + a(10)) / (a(7) + a(8)) = 3 + 2 * Real.sqrt 2 := -- final result for the ratio (a_9 + a_10) / (a_7 + a_8)
by 
  intros a q h_geom_pos h_pos h_arith h_q
  sorry

end geometric_sequence_arithmetic_progression_ratio_l787_787985


namespace parallel_DE_FG_l787_787364

/-- Given an acute-angled triangle ABC with circumcircle Γ and points D and E on AB and AC such that AD = AE. The perpendicular bisectors of BD and CE intersect the small arcs ⟦AB⟧ and ⟦AC⟧ at points F and G respectively. Prove that DE is parallel to FG. -/
theorem parallel_DE_FG {A B C D E F G : Point} (h_triangle_ABC : acute_triangle A B C)
  (h_circumcircle : circumcircle Γ A B C) (h_D_on_AB : on_segment D A B) 
  (h_E_on_AC : on_segment E A C) (h_AD_AE : AD = AE)
  (h_F_on_perpbis_AB : on_perpbisector_segment F B D) (h_F_on_arc : on_arc_minor F A B)
  (h_G_on_perpbis_AC : on_perpbisector_segment G C E) (h_G_on_arc : on_arc_minor G A C) :
  parallel (line_through D E) (line_through F G) :=
sorry

end parallel_DE_FG_l787_787364


namespace maximum_c_l787_787693

theorem maximum_c (a b c : ℝ) (h1 : 2^a + 2^b = 2^(a + b))
  (h2 : 2^a + 2^b + 2^c = 2^(a + b + c)) :
  c ≤ 2 - Real.log 3 / Real.log 2 := sorry

end maximum_c_l787_787693


namespace monotonicity_of_g_range_of_m_l787_787741

noncomputable def f (x : ℝ) : ℝ := x * real.exp x

-- Problem 1
theorem monotonicity_of_g (a : ℝ) (h : a > 0) :
  let g (x : ℝ) := f x - (1/2) * a * (x + 1)^2 in
  if a = 1/real.exp 1 then
    ∀ x : ℝ, 0 ≤ (x + 1) * (real.exp x - a)
  else if 0 < a ∧ a < 1/real.exp 1 then
    ∀ x : ℝ, (x < real.log a ∨ x > -1) → (x + 1) * (real.exp x - a) ≥ 0 ∧
             (x ≠ real.log a ∨ x ≠ -1) → (x + 1) * (real.exp x - a) < 0
  else if a > 1/real.exp 1 then
    ∀ x : ℝ, (x < -1 ∨ x > real.log a) → (x + 1) * (real.exp x - a) ≥ 0 ∧
             (x ≠ -1 ∨ x ≠ real.log a) → (x + 1) * (real.exp x - a) < 0
  else
    false :=
sorry

-- Problem 2
theorem range_of_m (m : ℝ) :
  (∃ x0 : ℝ, (f x0 - x0 * real.exp x0 = m)) ↔ -5 / real.exp (-2)^2 < m ∧ m < 0 :=
sorry

end monotonicity_of_g_range_of_m_l787_787741


namespace range_subtraction_l787_787655

theorem range_subtraction {a b : ℝ} (h_range : set.image (fun x => 2 * real.cos x) (set.Icc (real.pi / 3) (4 * real.pi / 3)) = set.Icc a b) :
  b - a = 3 :=
by
  sorry

end range_subtraction_l787_787655


namespace gcf_450_144_l787_787124

theorem gcf_450_144 : Nat.gcd 450 144 = 18 := by
  sorry

end gcf_450_144_l787_787124


namespace smallest_delicious_integer_l787_787398

theorem smallest_delicious_integer :
  ∃ B : ℤ, (∃ n : ℤ, (B ≥ n ∧ B < n + 100 ∧ (list.range 100).sum + 100 * n = 5050)) ∧ ∀ B' : ℤ, (∃ n' : ℤ, (B' ≥ n' ∧ B' < n' + 100 ∧ (list.range 100).sum + 100 * n' = 5050)) → B ≤ B' :=
begin
  sorry
end

end smallest_delicious_integer_l787_787398


namespace max_correct_answers_l787_787176

theorem max_correct_answers (c w b : ℕ) (h1 : c + w + b = 30) (h2 : 4 * c - w = 85) : c ≤ 23 :=
  sorry

end max_correct_answers_l787_787176


namespace expected_sticky_corn_in_sample_l787_787876

/-- Our initial conditions -/
def sweet_corn := 42
def black_corn := 7
def sticky_corn := 56
def high_oil_corn := 35
def total_sampled_corn := 40

/-- The main theorem we want to prove -/
theorem expected_sticky_corn_in_sample :
  let total_corn := sweet_corn + black_corn + sticky_corn + high_oil_corn in
  (total_sampled_corn * sticky_corn) / total_corn = 16 :=
by
  sorry

end expected_sticky_corn_in_sample_l787_787876


namespace incorrect_statement_c_l787_787221

open Real

theorem incorrect_statement_c (p q: ℝ) : ¬(∀ x: ℝ, (x * abs x + p * x + q = 0 ↔ p^2 - 4 * q ≥ 0)) :=
sorry

end incorrect_statement_c_l787_787221


namespace magnitude_AB_is_correct_l787_787628

open Real

def point (ℝ : Type) := ℝ × ℝ

def A : point ℝ := (3, 2)
def B : point ℝ := (-1, 6)

def vector_magnitude (p1 p2 : point ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem magnitude_AB_is_correct : vector_magnitude A B = 4 * sqrt 2 := by 
  sorry

end magnitude_AB_is_correct_l787_787628


namespace arrangements_two_girls_next_l787_787107

theorem arrangements_two_girls_next (boys girls : ℕ) (exactly_two_girls_next : Prop) : 
  boys = 4 → girls = 3 → exactly_two_girls_next → ∃ n, n = 2880 :=
by
  intro hBoys hGirls hCondition
  use 2880
  sorry

end arrangements_two_girls_next_l787_787107


namespace trigonometric_solution_l787_787788

theorem trigonometric_solution
    (x y z : ℝ)
    (h₀ : cos x ≠ 0)
    (h₁ : cos y ≠ 0)
    (n k m : ℤ)
    (hx : x = n * π)
    (hy : y = k * π)
    (hz : z = π / 2 + 2 * m * π) :
    (\(cos x)\^2 + 1 / (cos x)\^2)\^3 + (\(cos y)\^2 + 1 / (cos y)\^2)\^3 = 16 * sin z := by
  sorry

end trigonometric_solution_l787_787788


namespace inequality_relationship_l787_787974

theorem inequality_relationship (m : ℝ) (a b : ℝ)
  (h1 : 9^m = 10)
  (h2 : a = 10^m - 11) 
  (h3 : b = 8^m - 9) :
  a > 0 ∧ 0 > b := 
sorry

end inequality_relationship_l787_787974


namespace total_trees_l787_787924

-- Definitions based on the conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3

-- Theorem stating the total number of apple trees planted by Ava and Lily
theorem total_trees : ava_trees + lily_trees = 15 := by
  -- We skip the proof for now
  sorry

end total_trees_l787_787924


namespace line_equation_passing_through_point_l787_787104

noncomputable def equation_of_line (m : ℝ) (A : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x1, y1) := A in
  (3, -1, 5) -- since the line equation will be 3x - y - 5 = 0

theorem line_equation_passing_through_point
  (m : ℝ) (A : ℝ × ℝ)
  (hm : m = 3)
  (hA : A = (1, -2)) :
  equation_of_line m A = (3, -1, -5) :=
sorry

end line_equation_passing_through_point_l787_787104


namespace cos_sum_alpha_beta_l787_787757

theorem cos_sum_alpha_beta :
  ∀ (α β : ℝ),
    (0 < α ∧ α < π / 3) →
    (π / 6 < β ∧ β < π / 2) →
    5 * sqrt 3 * sin α + 5 * cos α = 8 →
    sqrt 2 * sin β + sqrt 6 * cos β = 2 →
    cos (α + β) = - (sqrt 2) / 10 :=
by
  intros α β h1 h2 h3 h4
  sorry

end cos_sum_alpha_beta_l787_787757


namespace people_arrangement_l787_787108

theorem people_arrangement (A B C D E : Type) :
  let people : List (A ⊕ B ⊕ C ⊕ D ⊕ E) := [head, tail, filled, filled, filled]
  ∃ (A : people) (B : people), -- A occupies head or tail position, B can't occupy head or tail
  ↑(people.removeAllWithIndex [head, tail].count) = 3 →
  (∃ remaining_people : people, remaining_people.length == 3) →
  2 * 3 * factorial 3 = 36 :=
sorry

end people_arrangement_l787_787108


namespace enclosed_area_different_book_selections_sqrt_relation_negative_extremum_l787_787868

theorem enclosed_area : 
  (2 * ∫ x in (0 : Real)..(Real.pi / 3), Real.sin x) = 1 := 
sorry

theorem different_book_selections (books : Finset ℕ) (h : books.card = 5) :
  (∑ (m₁ m₂ x₁ x₂ : Fin₅), m₁ ≠ m₂ ∨ x₁ ≠ x₂) = 90 :=
sorry

theorem sqrt_relation (a b : ℝ) (hb : b ≠ 0) (h : Real.sqrt (6 + a / b) = 6 * Real.sqrt (a / b)) :
  a + b = 41 :=
sorry

theorem negative_extremum (a : ℝ) (h : a < 0) :
  (∃ x < 0, Real.exp x + 2 * a = 0) ↔ -1 / 2 < a :=
sorry

end enclosed_area_different_book_selections_sqrt_relation_negative_extremum_l787_787868


namespace animal_jump_distances_l787_787422

theorem animal_jump_distances (gh_jump_without_obstacle gh_obstacle frog_jump_difference frog_obstacle kangaroo_jump_multiplier kangaroo_obstacle : ℕ) :
  gh_jump_without_obstacle = 25 →
  gh_obstacle = 5 →
  frog_jump_difference = 15 →
  frog_obstacle = 10 →
  kangaroo_jump_multiplier = 2 →
  kangaroo_obstacle = 15 →
  let gh_total := gh_jump_without_obstacle + gh_obstacle,
      frog_jump_without_obstacle := gh_jump_without_obstacle + frog_jump_difference,
      frog_total := frog_jump_without_obstacle + frog_obstacle,
      kangaroo_jump_without_obstacle := frog_jump_without_obstacle * kangaroo_jump_multiplier,
      kangaroo_total := kangaroo_jump_without_obstacle + kangaroo_obstacle,
      total_distance := gh_total + frog_total + kangaroo_total
  in total_distance = 175 := by
  intros; dsimp;
  sorry

end animal_jump_distances_l787_787422


namespace obtuse_angle_tetrahedron_limit_l787_787058

theorem obtuse_angle_tetrahedron_limit {h k : ℝ} (h_pos : 0 < h) (k_pos : 0 < k) :
  ∀ (P Q R : EuclideanSpace ℝ (Fin 3)),
  (P = ⟨2, 0, 0⟩) ∧ (Q = ⟨h, h * Real.sqrt 3, 0⟩) ∧ (R = ⟨k, k / Real.sqrt 3, Real.sqrt (8 / 3)⟩) →
  ∃ (angle : ℝ), angle > 90 ∧ angle ≤ 120 :=
begin
  sorry
end

end obtuse_angle_tetrahedron_limit_l787_787058


namespace measure_of_area_l787_787354

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 3 then x^2
else if 3 < x ∧ x ≤ 7 then 10 - x
else if 7 < x ∧ x ≤ 10 then x - 7
else 0

def integral_f_a_b (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
∫ x in a..b, f x

theorem measure_of_area :
  let M := integral_f_a_b 0 3 (λ x, x^2) +
           integral_f_a_b 3 7 (λ x, 10 - x) +
           integral_f_a_b 7 10 (λ x, x - 7)
  in M = 43.5 :=
by
  sorry

end measure_of_area_l787_787354


namespace grandfather_grandson_ages_l787_787057

theorem grandfather_grandson_ages :
  ∃ (x y a b : ℕ), 
    70 < x ∧ 
    x < 80 ∧ 
    x - a = 10 * (y - a) ∧ 
    x + b = 8 * (y + b) ∧ 
    x = 71 ∧ 
    y = 8 :=
by
  sorry

end grandfather_grandson_ages_l787_787057


namespace convert_1920_deg_to_rad_l787_787573

theorem convert_1920_deg_to_rad :
  let degrees_to_radians := λ (d : ℝ), (π / 180) * d in
  degrees_to_radians 1920 = 32 * π / 3 :=
by
  let degrees_to_radians := λ (d : ℝ), (π / 180) * d
  sorry

end convert_1920_deg_to_rad_l787_787573


namespace angle_proof_l787_787776

variable (A B C : Type) [AddGroup A] [AddGroup B]
variables (AC BC AB : A) (angle_A angle_B : B)

-- Definitions based on conditions
def condition (AC BC AB : A) : Prop := 
  AC / BC = (AB + BC) / AC

-- The theorem to prove
theorem angle_proof (h : condition AC BC AB) : angle_B = 2 * angle_A :=
  sorry

end angle_proof_l787_787776


namespace part_one_part_two_l787_787997

noncomputable def ellipse_eq := (x y a : ℝ) (ha : a > 0) : Prop := x^2 / a^2 + y^2 / 3 = 1
def focus_eq := (x y : ℝ) : Prop := x = -1 ∧ y = 0
def vertice_A := (xa : ℝ) : xa = -2
def vertice_B := (xb : ℝ) : xb = 2
def line_f_eq := (x y m : ℝ) : Prop := y = m * (x + 1)

theorem part_one (a x1 x2 y1 y2 : ℝ) (ha : a = 2) (m : ℝ) 
  (h_ellipse : ellipse_eq x1 y1 a ∧ ellipse_eq x2 y2 a)
  (h_line : line_f_eq x1 y1 1 ∧ line_f_eq x2 y2 1) :
  (abs (x1 - x2) * sqrt (1 + 1) = 24/7) := by
sorry

theorem part_two (a x1 x2 y1 y2 k : ℝ) (h1 : k ≠ 0) (ha : a = 2)
    (h_ellipse : ellipse_eq x1 y1 a ∧ ellipse_eq x2 y2 a)
    (h_line : line_f_eq x1 y1 k ∧ line_f_eq x2 y2 k) (ha_neg: k ≠0):
    max (2 * abs (k * ((x1 + 1) + (x2 + 1))) / (3 + 4 * k^2)) = sqrt 3 := by 
sorry

end part_one_part_two_l787_787997


namespace volleyball_team_starters_selection_l787_787771

theorem volleyball_team_starters_selection : 
  ∃ (n : ℕ), n = nat.choose 16 6 ∧ n = 8008 :=
by
  use (nat.choose 16 6)
  split
  · rfl
  · norm_num
  sorry

end volleyball_team_starters_selection_l787_787771


namespace problem1_problem2_l787_787560

-- Problem 1: Prove that 2023 * 2023 - 2024 * 2022 = 1
theorem problem1 : 2023 * 2023 - 2024 * 2022 = 1 := 
by 
  sorry

-- Problem 2: Prove that (-4 * x * y^3) * (1/2 * x * y) + (-3 * x * y^2)^2 = 7 * x^2 * y^4
theorem problem2 (x y : ℝ) : (-4 * x * y^3) * ((1/2) * x * y) + (-3 * x * y^2)^2 = 7 * x^2 * y^4 := 
by 
  sorry

end problem1_problem2_l787_787560


namespace triangle_properties_l787_787169

-- conditions
def hypotenuse : ℕ := 13
def leg₁ : ℕ := 5

-- Pythagorean theorem for right triangles
def leg₂ : ℕ := (hypotenuse^2 - leg₁^2).natAbs.sqrt

-- The area and perimeter definitions
def area : ℕ := (leg₁ * leg₂) / 2
def perimeter : ℕ := leg₁ + leg₂ + hypotenuse

-- The statement to be proven
theorem triangle_properties :
  (leg₂ = 12) ∧ (area = 30) ∧ (perimeter = 30) :=
by
  sorry

end triangle_properties_l787_787169


namespace coin_orientation_inverted_when_reach_R_l787_787152

theorem coin_orientation_inverted_when_reach_R (P Q R : Point) (d : ℝ) (coin_orientation : ℝ):
  dist P Q = d → dist Q R = d → 
  rotate coin_orientation 270 = 0 → -- initial rotation brings F to a certain mark
  rotate (rotate coin_orientation 270) 270 = 180 := -- rotating 270 degrees twice results in total 180 inversion
sorry

end coin_orientation_inverted_when_reach_R_l787_787152


namespace area_of_triangle_MAB_l787_787714

noncomputable def triangle_area (A B M : ℝ × ℝ) : ℝ :=
  0.5 * ((B.1 - A.1) * (M.2 - A.2) - (M.1 - A.1) * (B.2 - A.2))

theorem area_of_triangle_MAB :
  let C1 (p : ℝ × ℝ) := p.1^2 - p.2^2 = 2
  let C2 (p : ℝ × ℝ) := ∃ θ, p.1 = 2 + 2 * Real.cos θ ∧ p.2 = 2 * Real.sin θ
  let M := (3.0, 0.0)
  let A := (2, 2 * Real.sin (Real.pi / 6))
  let B := (2 * Real.sqrt 3, 2 * Real.sin (Real.pi / 6))
  triangle_area A B M = (3 * Real.sqrt 3 - 3) / 2 :=
by
  sorry

end area_of_triangle_MAB_l787_787714


namespace book_length_l787_787403

theorem book_length (P : ℕ) (h1 : 2323 = (P - 2323) + 90) : P = 4556 :=
by
  sorry

end book_length_l787_787403


namespace complex_quadrant_l787_787857

theorem complex_quadrant (z : ℂ) (h : z * (2 - I) = 2 + I) : 0 < z.re ∧ 0 < z.im := 
sorry

end complex_quadrant_l787_787857


namespace range_of_k_l787_787330

theorem range_of_k {k : ℝ} :
  (∃ (x y : ℝ), x^2 + y^2 + 8 * x + 15 = 0 ∧ y = k * x ∧ (∀ r : ℝ, r > 0 ∧ 
  (∃ z w : ℝ, (z - x) ^ 2 + (w - y) ^ 2 = r ^ 2 ∧ (z + 4) ^ 2 + w ^ 2 = (r + 1) ^ 2))) ↔ 
  -4/3 ≤ k ∧ k ≤ real.sqrt 3 / 3 :=
begin
  sorry
end

end range_of_k_l787_787330


namespace quadruple_nested_function_l787_787141

def a (k : ℕ) : ℕ := (k + 1) ^ 2

theorem quadruple_nested_function (k : ℕ) (h : k = 1) : a (a (a (a (k)))) = 458329 :=
by
  rw [h]
  sorry

end quadruple_nested_function_l787_787141


namespace train_speed_l787_787538

theorem train_speed (length_m : ℝ) (time_s : ℝ) (h_length : length_m = 133.33333333333334) (h_time : time_s = 8) : 
  let length_km := length_m / 1000
  let time_hr := time_s / 3600
  length_km / time_hr = 60 :=
by
  sorry

end train_speed_l787_787538


namespace count_sequences_returns_Q_l787_787736

-- Defining the transformations
abbrev Point := ℤ × ℤ
def rotation_90 (p : Point) : Point := (-p.2, p.1)
def rotation_180 (p : Point) : Point := (-p.1, -p.2)
def rotation_270 (p : Point) : Point := (p.2, -p.1)
def reflection_x (p : Point) : Point := (p.1, -p.2)
def reflection_y (p : Point) : Point := (-p.1, p.2)
def reflection_y_eq_x (p : Point) : Point := (p.2, p.1)

-- Defining the vertices of the quadrilateral Q
def Q_vertices : List Point := [(0, 0), (6, 0), (6, 4), (0, 4)]

-- Apply a sequence of transformations
def apply_transformations (seq : List (Point → Point)) (pts : List Point) : List Point :=
  List.map (λ p => List.foldl (flip id) p seq) pts

-- Determine if the quadrilateral returns to the original position
def Q_returns_to_origin (seq : List (Point → Point)) : Bool :=
  apply_transformations seq Q_vertices == Q_vertices

-- Generate all sequences of three transformations
def transform_fns : List (Point → Point) := [
  rotation_90, rotation_180, rotation_270,
  reflection_x, reflection_y, reflection_y_eq_x]

def all_transform_sequences : List (List (Point → Point)) :=
  List.replicateM 3 transform_fns

-- Count the valid sequences
def count_valid_sequences : Nat :=
  List.length (List.filter Q_returns_to_origin all_transform_sequences)

theorem count_sequences_returns_Q : count_valid_sequences = 18 := sorry

end count_sequences_returns_Q_l787_787736


namespace common_tangent_line_sum_of_zeros_l787_787742

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) (a : ℝ) := x^2 + a
noncomputable def F (x : ℝ) (a : ℝ) := f x - g x a

theorem common_tangent_line (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₁ ≠ x₂ ∧
   (f x₁ - (1 / x₁) * x₁ = g x₁ a - (2 * x₁) * x₁)) →
  a ∈ Ici (-(1 + Real.log 2) / 2) := 
sorry

theorem sum_of_zeros (a x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : F x₁ a = 0) (h₃ : F x₂ a = 0) : 
  x₁ + x₂ > Real.sqrt 2 :=
sorry

end common_tangent_line_sum_of_zeros_l787_787742


namespace determine_juniors_score_l787_787175

noncomputable def juniors_score (juniors seniors freshmen : ℕ) 
(average_class_score average_seniors_score average_freshmen_score : ℝ) : ℝ :=
let 
  total_students := juniors + seniors + freshmen in
let 
  total_class_score := total_students * average_class_score in
let 
  total_seniors_score := seniors * average_seniors_score in
let 
  total_freshmen_score := freshmen * average_freshmen_score in
(total_class_score - total_seniors_score - total_freshmen_score) / juniors

theorem determine_juniors_score :
  juniors_score 30 50 20 80 85 70 = 78 :=
sorry

end determine_juniors_score_l787_787175


namespace geometric_sequence_ratio_l787_787289

theorem geometric_sequence_ratio 
  (a_n b_n : ℕ → ℝ) 
  (S_n T_n : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, S_n n = a_n n * (1 - (1/2)^n)) 
  (h2 : ∀ n : ℕ, T_n n = b_n n * (1 - (1/3)^n))
  (h3 : ∀ n, n > 0 → (S_n n) / (T_n n) = (3^n + 1) / 4) : 
  (a_n 3) / (b_n 3) = 9 :=
by
  sorry

end geometric_sequence_ratio_l787_787289


namespace problem1_problem2_problem3_problem4_l787_787203

theorem problem1 : (-4.7 : ℝ) + 0.9 = -3.8 := by
  sorry

theorem problem2 : (- (1 / 2) : ℝ) - (-(1 / 3)) = -(1 / 6) := by
  sorry

theorem problem3 : (- (10 / 9) : ℝ) * (- (6 / 10)) = (2 / 3) := by
  sorry

theorem problem4 : (0 : ℝ) * (-5) = 0 := by
  sorry

end problem1_problem2_problem3_problem4_l787_787203


namespace prove_length_QA_l787_787261

-- Define the main points and the circle
structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

structure Circle where
  center : Point
  radius : ℝ

-- Definitions from the problem
axiom Q : Point
axiom O : Circle
axiom A B C : Point
axiom h1 : distance Q O.center = O.radius  -- Q is a tangent to O
axiom h2 : midpoint A B = C  -- C is the midpoint of A and B
axiom h3 : distance A B = 6  -- AB = 6

-- We need to prove that length of QA is 6
theorem prove_length_QA : distance Q A = 36 := by
  sorry

end prove_length_QA_l787_787261


namespace identify_quadratic_equation_l787_787130

def is_quadratic_equation (eq : String) : Prop :=
  match eq with
  | "B" => true
  | _   => false

theorem identify_quadratic_equation :
  (is_quadratic_equation "A" = false) ∧ 
  (is_quadratic_equation "B" = true) ∧ 
  (is_quadratic_equation "C" = false) ∧ 
  (is_quadratic_equation "D" = false) :=
by
  split; sorry

end identify_quadratic_equation_l787_787130


namespace value_of_m_l787_787094

variable {x y m : ℝ}

-- Polynomial definition
def polynomial (m : ℝ) : ℝ := 1/5 * x^2 * y^(abs m) - (m + 1) * y + 1/7

-- Conditions for the polynomial to be a cubic binomial in terms of x and y
def isCubicBinomial (m : ℝ) : Prop :=
  abs m = 1 ∧ (-(m + 1)) = 0

-- Lean statement for the proof problem
theorem value_of_m (h : isCubicBinomial m) : m = -1 :=
  by sorry

end value_of_m_l787_787094


namespace number_of_nonempty_subsets_l787_787426

def S : Set ℕ := {1, 2, 3}

theorem number_of_nonempty_subsets : (Set.powerset S).card - 1 = 7 := by
  sorry

end number_of_nonempty_subsets_l787_787426


namespace geometric_series_sum_l787_787201

theorem geometric_series_sum :
  let a := (1 : ℝ) / 2
  let r := (1 : ℝ) / 2
  let n := 5
  S = a * (1 - r^n) / (1 - r) → S = 31 / 32 :=
by
  intros a r n S h
  have h1 : a = 1 / 2 := rfl
  have h2 : r = 1 / 2 := rfl
  have h3 : n = 5 := rfl
  rw [h1, h2, h3] at h
  sorry

end geometric_series_sum_l787_787201


namespace locus_of_tangency_points_l787_787973

open_locale big_operators

-- We define the problem
variables (A B C D : ℝ) -- Points on a line
variables (k1 k2 : circle ℝ) -- Circles passing through specified points
noncomputable def circle_tangency_locus (A B C D : ℝ) : set (euclidean_space ℝ) :=
{ E | ∃ (k1 k2 : circle ℝ), (A ∈ k1 ∧ B ∈ k1 ∧ C ∈ k2 ∧ D ∈ k2 ∧ k1 ≠ k2 ∧ E ∈ k1 ∧ E ∈ k2) }

-- The theorem we want to prove
theorem locus_of_tangency_points (A B C D : ℝ) :
    let f := {x : ℝ | x = A ∨ x = B ∨ x = C ∨ x = D} in
    ∀ E ∈ (circle_tangency_locus A B C D), (distance E ∈ (circle k, {A, B, C, D})) → 
    E ∉ f :=
sorry

end locus_of_tangency_points_l787_787973


namespace class_books_transfer_l787_787566

theorem class_books_transfer :
  ∀ (A B n : ℕ), 
    A = 200 → B = 200 → 
    (B + n = 3/2 * (A - n)) →
    n = 40 :=
by sorry

end class_books_transfer_l787_787566


namespace oranges_sold_l787_787442

def bags : ℕ := 10
def oranges_per_bag : ℕ := 30
def rotten_oranges : ℕ := 50
def oranges_for_juice : ℕ := 30

theorem oranges_sold : (bags * oranges_per_bag) - rotten_oranges - oranges_for_juice = 220 := by
  sorry

end oranges_sold_l787_787442


namespace joe_money_fraction_l787_787727

theorem joe_money_fraction :
  ∃ f : ℝ,
    (200 : ℝ) = 160 + (200 - 160) ∧
    160 - 160 * f - 20 = 40 + 160 * f + 20 ∧
    f = 1 / 4 :=
by
  -- The proof should go here.
  sorry

end joe_money_fraction_l787_787727


namespace parallel_planes_of_perpendicular_lines_l787_787001

-- Definitions of planes and lines
variable (Plane Line : Type)
variable (α β γ : Plane)
variable (m n : Line)

-- Relations between planes and lines
variable (perpendicular : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Conditions for the proof
variable (m_perp_α : perpendicular α m)
variable (n_perp_β : perpendicular β n)
variable (m_par_n : line_parallel m n)

-- Statement of the theorem
theorem parallel_planes_of_perpendicular_lines :
  parallel α β :=
sorry

end parallel_planes_of_perpendicular_lines_l787_787001


namespace arc_length_parametric_l787_787863

-- Define the parametric equations and their interval
def x (t : ℝ) := 4 * (Real.cos t) ^ 3
def y (t : ℝ) := 4 * (Real.sin t) ^ 3

theorem arc_length_parametric :
  (∫ t in (Real.pi / 6 : ℝ)..(Real.pi / 4), √((-(12 * (Real.cos t) ^ 2 * Real.sin t)) ^ 2 + (12 * (Real.sin t) ^ 2 * Real.cos t) ^ 2)) = 3 / 2 :=
sorry

end arc_length_parametric_l787_787863


namespace opposite_of_neg_sqrt_two_l787_787827

theorem opposite_of_neg_sqrt_two : -(-Real.sqrt 2) = Real.sqrt 2 := 
by {
  sorry
}

end opposite_of_neg_sqrt_two_l787_787827


namespace triangle_area_l787_787122

-- Defining the vertices of the triangle
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (-2, -7)
def C : ℝ × ℝ := (8, -2)

-- Proof statement for the area of the triangle
theorem triangle_area (A B C : ℝ × ℝ) : 
  A = (-2, 3) → 
  B = (-2, -7) → 
  C = (8, -2) → 
  (1 / 2) * 10 * 10 = 50 :=
by
  intros hA hB hC
  rw [hA, hB, hC]
  sorry

end triangle_area_l787_787122


namespace largest_n_integer_expression_l787_787852

theorem largest_n_integer_expression (n : ℕ) (h : (∃ k : ℚ, k = √(7 + 2*√n) / (2*√7 - √n))) : 
  n ≤ 343 :=
sorry

end largest_n_integer_expression_l787_787852


namespace angle_between_lateral_face_and_base_of_pyramid_l787_787821

theorem angle_between_lateral_face_and_base_of_pyramid
  (θ : ℝ)
  (H_cond : ∀ H : ℝ, angle_between_lateral_edge_and_base H = planar_angle_at_apex H) :
  θ = Real.arctan (Real.sqrt (1 + Real.sqrt 5)) :=
sorry

end angle_between_lateral_face_and_base_of_pyramid_l787_787821


namespace marbles_total_is_260_l787_787218

/-- Define the number of marbles in each jar. -/
def jar1 : ℕ := 80
def jar2 : ℕ := 2 * jar1
def jar3 : ℕ := jar1 / 4

/-- The total number of marbles Courtney has. -/
def total_marbles : ℕ := jar1 + jar2 + jar3

/-- Proving that the total number of marbles is 260. -/
theorem marbles_total_is_260 : total_marbles = 260 := 
by
  sorry

end marbles_total_is_260_l787_787218


namespace scaling_transformation_correct_l787_787114

-- Define the initial and target graph equations
def initialEquation (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1
def targetEquation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the scaling transformation
def scalingTransformation (x y : ℝ) : ℝ × ℝ := (1/2 * x, (Real.sqrt 3) / 3 * y)

-- State the theorem that the scaling transformation is correct
theorem scaling_transformation_correct (x y : ℝ) (h₀ : initialEquation x y) :
    targetEquation (scalingTransformation x y).fst (scalingTransformation x y).snd :=
sorry

end scaling_transformation_correct_l787_787114


namespace jordan_run_time_for_7_miles_l787_787347

-- Defining the conditions and question
theorem jordan_run_time_for_7_miles :
  -- Conditions
  (steve_time_to_run_5_miles: ℝ) (h1: steve_time_to_run_5_miles = 30) -- It took Steve 30 minutes to run 5 miles
  (jordan_time_to_run_4_miles: ℝ) (h2: jordan_time_to_run_4_miles = 0.5 * steve_time_to_run_5_miles) -- Jordan ran 4 miles in half the time it took Steve to run 5 miles
  (speed_increase: ℝ) (h3: speed_increase = 0.1) -- Jordan’s speed increases by 10% after the first half of his distance
  
  -- Question and correct answer
  {t : ℝ // t = 25.568} := sorry -- The time it takes Jordan to run 7 miles is 25.568 minutes

end jordan_run_time_for_7_miles_l787_787347


namespace line_through_parabola_midpoint_l787_787280

noncomputable def parabola := { points : ℝ × ℝ | ∃ x y, points = (x, y) ∧ y^2 = 4 * x }

theorem line_through_parabola_midpoint (A B : ℝ × ℝ) (x1 y1 x2 y2 : ℝ)
  (hA : A = (x1, y1)) (hB : B = (x2, y2))
  (HA_on_parabola : A ∈ parabola) (HB_on_parabola : B ∈ parabola)
  (midpoint_condition : (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = 2) :
  ∃ m b, (∀ x y, y = m * x + b ↔ y = x) :=
begin
  sorry
end

end line_through_parabola_midpoint_l787_787280


namespace over_limit_weight_l787_787341

variable (hc_books : ℕ) (hc_weight : ℕ → ℝ)
variable (tex_books : ℕ) (tex_weight : ℕ → ℝ)
variable (knick_knacks : ℕ) (knick_weight : ℕ → ℝ)
variable (weight_limit : ℝ)

axiom hc_books_value : hc_books = 70
axiom hc_weight_value : hc_weight hc_books = 0.5
axiom tex_books_value : tex_books = 30
axiom tex_weight_value : tex_weight tex_books = 2
axiom knick_knacks_value : knick_knacks = 3
axiom knick_weight_value : knick_weight knick_knacks = 6
axiom weight_limit_value : weight_limit = 80

theorem over_limit_weight : 
  (hc_books * hc_weight hc_books + tex_books * tex_weight tex_books + knick_knacks * knick_weight knick_knacks) - weight_limit = 33 := by
  sorry

end over_limit_weight_l787_787341


namespace father_age_is_30_l787_787488

theorem father_age_is_30 {M F : ℝ} 
  (h1 : M = (2 / 5) * F) 
  (h2 : M + 6 = (1 / 2) * (F + 6)) :
  F = 30 :=
sorry

end father_age_is_30_l787_787488


namespace eval_polynomial_eq_5_l787_787594

noncomputable def eval_polynomial_at_root : ℝ :=
let x := (3 + 3 * Real.sqrt 5) / 2 in
if h : x^2 - 3*x - 9 = 0 then x^3 - 3*x^2 - 9*x + 5 else 0

theorem eval_polynomial_eq_5 :
  eval_polynomial_at_root = 5 :=
by
  sorry

end eval_polynomial_eq_5_l787_787594


namespace locus_of_midpoint_l787_787878

theorem locus_of_midpoint
  (x y : ℝ)
  (h : ∃ (A : ℝ × ℝ), A = (2*x, 2*y) ∧ (A.1)^2 + (A.2)^2 - 8*A.1 = 0) :
  x^2 + y^2 - 4*x = 0 :=
by
  sorry

end locus_of_midpoint_l787_787878


namespace toothpicks_in_20th_stage_l787_787448

theorem toothpicks_in_20th_stage :
  (3 + 3 * (20 - 1) = 60) :=
by
  sorry

end toothpicks_in_20th_stage_l787_787448


namespace quadratic_square_binomial_l787_787586

theorem quadratic_square_binomial (c : ℝ) :
  (∃ a : ℝ, (λ x : ℝ, x^2 + 116 * x + c = (x + a)^2)) → c = 3364 :=
begin
  intros h,
  cases h with a ha,
  rw [mul_assoc, add_assoc, add_assoc] at ha,
  have : 2 * a = 116, sorry, -- skipped detailed calculation
  have : a = 58, sorry,
  rw this at ha,
  have : c = 58^2, sorry,
  norm_num at this,
  exact this,
end

end quadratic_square_binomial_l787_787586


namespace intersection_is_expected_l787_787734

open Set

def setA : Set ℝ := { x | (x + 1) / (x - 2) ≤ 0 }
def setB : Set ℝ := { x | x^2 - 4 * x + 3 ≤ 0 }
def expectedIntersection : Set ℝ := { x | 1 ≤ x ∧ x < 2 }

theorem intersection_is_expected :
  (setA ∩ setB) = expectedIntersection := by
  sorry

end intersection_is_expected_l787_787734


namespace part_a_part_b_part_c_l787_787840

-- Define the conditions for the meetings
structure MeetingSetup (A B C : Type) where
  meetings : (A → B → C → Prop)
  unique_pair : ∀ a b c d e f, meetings a b c ∧ meetings d e f → (a = d ∧ b = e ∧ c = f)

-- Part (a): If the meeting is possible, then the number of mathematicians in the groups is equal
theorem part_a (A B C : Type) (setup : MeetingSetup A B C) :
  ∃n, ∃ (f : Fin n → A), ∃ (g : Fin n → B), ∃ (h : Fin n → C), 
  (∀ i j k, setup.meetings (f i) (g j) (h k)) ∧
  (∀ i j, setup.unique_pair (f i) (g j) (h j)) := sorry

-- Part (b): If there are 3 mathematicians in each group, the arrangement is possible
def fin_3 := Fin 3

theorem part_b :
  ∃ (A B C : Type) (f : fin_3 → A) (g : fin_3 → B) (h : fin_3 → C), 
  ∀ i j k, (f i, g j, h k) ∈ {p : A × B × C | ∃ m:MeetingSetup A B C, m.meetings (f i) (g j) (h k)} := sorry

-- Part (c): If the number of mathematicians in the groups are equal, the arrangement is possible
theorem part_c (n : ℕ) :
  ∃ (A B C : Type) (f : Fin n → A) (g : Fin n → B) (h : Fin n → C),
  ∀ i j k, (f i, g j, h k) ∈ {p : A × B × C | ∃ m:MeetingSetup A B C, m.meetings (f i) (g j) (h k)} := sorry

end part_a_part_b_part_c_l787_787840


namespace proof_problem_statement_l787_787999

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1) ^ 2 + y ^ 2 = 20

-- Define point B on the x-axis
variable l : ℝ
def point_B : ℝ × ℝ := (l, 0)

-- Define point A moving on the circle C
def point_A (x y : ℝ) : Prop := circle_C x y

-- Define point M
def point_M : ℝ × ℝ := (0, 1 / 5)

-- Define the parabola C2
def parabola_C2 (x y : ℝ) : Prop := y = x ^ 2

-- Define point N moving on parabola C2
def point_N (x y : ℝ) : Prop := parabola_C2 x y 

noncomputable def trajectory_C1 : Prop := ∀ (x y : ℝ), 
  (point_A x y → 
   ∀ P_x P_y, 
   let P := (P_x, P_y) in
   -- condition P lies on ellipse:
   (x - P_x)^2 + (y - P_y)^2 = 4 → 
   (P_x^2 / 5 + P_y^2 / 4 = 1))

noncomputable def max_area_triangle_MPQ : ℝ := sqrt 130 / 5

-- Statement combining the proof of trajectory being an ellipse and maximum area calculation
theorem proof_problem_statement (x y P_x P_y Q_x Q_y : ℝ) (P : ℝ × ℝ := (P_x, P_y)) (Q : ℝ × ℝ := (Q_x, Q_y)) :
  (point_A x y → 
   (∀ (N_x N_y : ℝ), point_N N_x N_y → (
   (x - P_x)^2 + (y - P_y)^2 = 4 → 
   ((P_x^2 / 5 + P_y^2 / 4 = 1) ∧ 
   max_area_triangle_MPQ = sqrt 130 / 5))) :=
sorry

end proof_problem_statement_l787_787999


namespace entropy_decomposition_entropy_independence_entropy_monotonicity_l787_787754

noncomputable def H_xi_eta (k l : ℕ) (xi_vals : fin k → Type*)
  (eta_vals : fin l → Type*) (Pxi : fin k → ℝ) (P_eta_given_xi : Π (i : fin k) (j : fin l), ℝ): ℝ :=
∑ i in finset.univ, Pxi i * (-∑ j in finset.univ, P_eta_given_xi i j * real.log2 (P_eta_given_xi i j))

noncomputable def H_xi (k : ℕ) (xi_vals : fin k → Type*) (Pxi : fin k → ℝ) : ℝ :=
-∑ i in finset.univ, Pxi i * real.log2 (Pxi i)

noncomputable def H_xi_eta_joint (k l : ℕ) (P_joint : (fin k × fin l) → ℝ) : ℝ :=
-∑ p in finset.univ, P_joint p * real.log2 (P_joint p)

theorem entropy_decomposition (k l : ℕ) (xi_vals : fin k → Type*) (eta_vals : fin l → Type*)
  (Pxi : fin k → ℝ) (P_eta_given_xi : Π (i : fin k) (j : fin l), ℝ) (P_joint : (fin k × fin l) → ℝ) :
  H_xi_eta_joint k l P_joint = H_xi k xi_vals Pxi + H_xi_eta k l xi_vals eta_vals Pxi P_eta_given_xi := sorry

theorem entropy_independence (k l : ℕ) (xi_vals : fin k → Type*)
  (eta_vals : fin l → Type*) (Pxi : fin k → ℝ) (P_eta : fin l → ℝ) (P_joint : (fin k × fin l) → ℝ)
  (h_ind : ∀ (i : fin k) (j : fin l), P_joint (i, j) = Pxi i * P_eta j) :
  H_xi_eta_joint k l P_joint = H_xi k xi_vals Pxi + (-∑ j in finset.univ, P_eta j * real.log2 (P_eta j)) := sorry

theorem entropy_monotonicity (k l : ℕ) (xi_vals : fin k → Type*)
  (eta_vals : fin l → Type*) (Pxi : fin k → ℝ) (P_eta_given_xi : Π (i : fin k) (j : fin l), ℝ) 
  (P_eta : Π (j : fin l), ℝ) :
  0 ≤ H_xi_eta k l xi_vals eta_vals Pxi P_eta_given_xi ∧ H_xi_eta k l xi_vals eta_vals Pxi P_eta_given_xi ≤ -∑ j in finset.univ, P_eta j * real.log2 (P_eta j) := sorry

end entropy_decomposition_entropy_independence_entropy_monotonicity_l787_787754


namespace computer_additions_l787_787159

theorem computer_additions (additions_per_second : ℕ) (hours : ℕ) (pause_minutes_per_hour : ℕ) : 
  additions_per_second = 15000 → hours = 12 → pause_minutes_per_hour = 10 →
  let active_minutes_per_hour := 60 - pause_minutes_per_hour in
  let active_seconds_per_hour := active_minutes_per_hour * 60 in
  let additions_per_hour := additions_per_second * active_seconds_per_hour in
  let total_additions := additions_per_hour * hours in
  total_additions = 540000000 :=
begin
  intros h_additions_per_second h_hours h_pause_minutes_per_hour,
  unfold active_minutes_per_hour active_seconds_per_hour additions_per_hour total_additions,
  rw [h_additions_per_second, h_hours, h_pause_minutes_per_hour],
  norm_num,
end

end computer_additions_l787_787159


namespace trapezoid_diagonals_l787_787392

variables (A B C D : Type) [OrderedField ℝ]

structure Trapezoid (A B C D : Type) := 
(base1 : Segment A B)
(base2 : Segment C D)
(diagonal1 : Segment A C)
(diagonal2 : Segment B D)

variables {trapezoid : Trapezoid A B C D}
variables (angle_DAB angle_CBA : ℝ)

-- The condition: base angles are not equal and angle_DAB < angle_CBA
axiom angles_condition : angle_DAB < angle_CBA

theorem trapezoid_diagonals (h : angles_condition) : 
  trapezoid.diagonal1.length > trapezoid.diagonal2.length := 
sorry

end trapezoid_diagonals_l787_787392


namespace trigonometric_identity_proof_l787_787617

theorem trigonometric_identity_proof (alpha : Real)
(h1 : Real.tan (alpha + π / 4) = 1 / 2)
(h2 : -π / 2 < alpha ∧ alpha < 0) :
  (2 * Real.sin alpha ^ 2 + Real.sin (2 * alpha)) / Real.cos (alpha - π / 4) = - (2 * Real.sqrt 5) / 5 :=
by
  sorry

end trigonometric_identity_proof_l787_787617


namespace greatest_five_digit_number_sum_l787_787735

theorem greatest_five_digit_number_sum (M : ℕ) (h1 : M < 100000) (h2 : M > 9999) 
  (digits : List ℕ) (h3 : M.digits = digits) 
  (h4 : digits.prod = 90) 
  (h5 : ∀ N : ℕ, N < 100000 → N > 9999 → N.digits.prod = 90 → N ≤ M) : 
  digits.sum = 18 :=
sorry

end greatest_five_digit_number_sum_l787_787735


namespace length_of_goods_train_l787_787860

theorem length_of_goods_train 
  (speed_kmh : ℕ) 
  (platform_length_m : ℕ) 
  (cross_time_s : ℕ) :
  speed_kmh = 72 → platform_length_m = 280 → cross_time_s = 26 → 
  ∃ train_length_m : ℕ, train_length_m = 240 :=
by
  intros h1 h2 h3
  sorry

end length_of_goods_train_l787_787860


namespace Oliver_monster_club_cards_l787_787377

theorem Oliver_monster_club_cards (BG AB MC : ℕ) 
  (h1 : BG = 48) 
  (h2 : BG = 3 * AB) 
  (h3 : MC = 2 * AB) : 
  MC = 32 :=
by sorry

end Oliver_monster_club_cards_l787_787377


namespace beetle_speed_l787_787138

-- Define the parameters for the problem.
def ant_distance : ℝ := 500 -- in meters
def ant_time : ℝ := 60 -- in minutes
def beetle_distance_share : ℝ := 0.85 -- beetle walks 15% less distance

-- Convert ant's time from minutes to hours.
def ant_time_in_hours : ℝ := ant_time / 60

-- Calculate the distance the beetle walks.
def beetle_distance : ℝ := beetle_distance_share * ant_distance

-- Convert beetle's distance from meters to kilometers.
def beetle_distance_in_km : ℝ := beetle_distance / 1000

-- Define the expected speed of the beetle in km/h.
def expected_beetle_speed : ℝ := 0.425

-- Theorem: Proving the beetle's speed in km/h given the conditions.
theorem beetle_speed :
  (beetle_distance_in_km / ant_time_in_hours) = expected_beetle_speed := by
  -- The proof is omitted here.
  sorry

end beetle_speed_l787_787138


namespace max_value_quadratic_function_l787_787329

open Real

theorem max_value_quadratic_function (r : ℝ) (x₀ y₀ : ℝ) (P_tangent : (2 / x₀) * x - y₀ = 0) 
  (circle_tangent : (x₀ - 3) * (x - 3) + y₀ * y = r^2) :
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ), f x = 1 / 2 * x * (3 - x)) ∧ 
  (∀ (x : ℝ), f x ≤ 9 / 8) :=
by
  sorry

end max_value_quadratic_function_l787_787329


namespace find_P_l787_787637

-- Define the points A and P
def Point : Type := ℝ × ℝ × ℝ

def distance (A P : Point) : ℝ :=
  Real.sqrt ((P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2 + (P.3 - A.3) ^ 2)

-- Given conditions
def A : Point := (5,2,-6)
def on_x_axis (P : Point) : Prop := P.2 = 0 ∧ P.3 = 0

-- The statement to prove: possible coordinates for P
theorem find_P (P : Point) (hP : on_x_axis P) (hDist : distance A P = 7) : 
  P = (8,0,0) ∨ P = (2,0,0) :=
by 
  sorry

end find_P_l787_787637


namespace regular_18_gon_sum_l787_787526

-- Define what it means to be a regular 18-gon
def regular_18_gon (n : ℕ) : Prop :=
  n = 18

-- Define the number of lines of symmetry for a regular 18-gon
def lines_of_symmetry (n : ℕ) (hs : regular_18_gon n) : ℕ :=
  n

-- Define the smallest positive angle of rotational symmetry for a regular 18-gon
def smallest_angle_of_rotational_symmetry (n : ℕ) (hs : regular_18_gon n) : ℕ :=
  360 / n

-- Prove that the sum of lines of symmetry and the smallest positive angle for a regular 18-gon equals 38
theorem regular_18_gon_sum :
  let n := 18 in
  regular_18_gon n →
  lines_of_symmetry n (by sorry) + smallest_angle_of_rotational_symmetry n (by sorry) = 38 :=
by sorry

end regular_18_gon_sum_l787_787526


namespace coin_game_winning_strategy_l787_787106

theorem coin_game_winning_strategy :
  (∃ n : ℕ, n = 2005) → 
  ∀ (P1_strategy : ℕ → ℕ) (P2_strategy : ℕ → ℕ), 
    -- Initial conditions
    (∀ k, P1_strategy k % 2 = 1 ∧ P1_strategy k ∈ {1, 3, ..., 99}) →
    (∀ k, P2_strategy k % 2 = 0 ∧ P2_strategy k ∈ {2, 4, ..., 100}) → 
    (∃ n' ≤ 2005, (n - 85) % 101 = 1) → 
    True := 
  sorry

end coin_game_winning_strategy_l787_787106


namespace optimal_playground_dimensions_and_area_l787_787562

theorem optimal_playground_dimensions_and_area:
  ∃ (l w : ℝ), 2 * l + 2 * w = 380 ∧ l ≥ 100 ∧ w ≥ 60 ∧ l * w = 9000 :=
by
  sorry

end optimal_playground_dimensions_and_area_l787_787562


namespace max_cosine_value_l787_787738

theorem max_cosine_value (a b : ℝ) (h : Real.cos (a - b) = Real.cos a + Real.cos b) : 1 ≥ Real.cos a :=
sorry

end max_cosine_value_l787_787738


namespace vertices_set_lie_outside_circle_l787_787618

variable {k : Type} [metric_space k]
variables {O P S M : k}
variables (c : circle k) (P : point k)

def reflection (O P : point k) : point k :=
  sorry -- Define reflection of P over O

def midpoint (P S : point k) : point k :=
  sorry -- Define midpoint of points P and S

noncomputable def isosceles_right_triangle (P Q R : point k) : Prop :=
  sorry -- Define isosceles right triangle property

theorem vertices_set_lie_outside_circle (k : circle k) (P : point k) :
  ∃ (Q : set (point k)), 
    (∀ R : point k, (R ∈ semicircle k ∧ isosceles_right_triangle P Q R) → Q ∉ interior k) ∧
    (∀ Q : point k, Q ∉ (semicircle k ∪ {P, midpoint P (reflection O P)})) :=
sorry

end vertices_set_lie_outside_circle_l787_787618


namespace solve_fraction_equation_l787_787063

theorem solve_fraction_equation (x : ℚ) (h : (x + 11) / (x - 4) = (x - 3) / (x + 6)) : x = -9 / 4 :=
begin
  sorry
end

end solve_fraction_equation_l787_787063


namespace length_AB_3_l787_787713

noncomputable def pointA (t : ℝ) : ℝ × ℝ := (-t, sqrt(3) * t)
noncomputable def pointA_polar (θ : ℝ) : ℝ := 4 * sin θ

noncomputable def pointB (t : ℝ) : ℝ × ℝ := (-t, sqrt(3) * t)
noncomputable def pointB_polar (θ : ℝ) : ℝ := 2 * sin θ

theorem length_AB_3:
  let θ := 2 * Real.pi / 3 in
  let ρ1 := pointA_polar θ in
  let ρ2 := pointB_polar θ in
  abs (ρ1 - ρ2) = Real.sqrt 3 :=
by
  let θ := 2 * Real.pi / 3
  let ρ1 := pointA_polar θ
  let ρ2 := pointB_polar θ
  sorry

end length_AB_3_l787_787713


namespace olympiad_problem_l787_787911

variable (a b c d : ℕ)
variable (N : ℕ := a + b + c + d)

theorem olympiad_problem
  (h1 : (a + d) / (N:ℚ) = 0.5)
  (h2 : (b + d) / (N:ℚ) = 0.6)
  (h3 : (c + d) / (N:ℚ) = 0.7)
  : (d : ℚ) / N * 100 = 40 := by
  sorry

end olympiad_problem_l787_787911


namespace two_colorable_regions_l787_787427

-- Define a type for regions of the plane
inductive Region
| insideCircle : ℕ → Region -- Regions defined inside a single circle
| betweenLines : ℕ → ℕ → Region -- Regions defined between two lines

def regions_colored_properly (regions : Finset Region) (coloring : Region → bool) :=
  ∀ (r1 r2 : Region), r1 ≠ r2 ∧ (shares_boundary r1 r2) → (coloring r1 ≠ coloring r2)

axiom shares_boundary : Region → Region → Prop

theorem two_colorable_regions : 
  ∀ (regions : Finset Region), 
  ∃ (coloring : Region → bool), regions_colored_properly regions coloring := 
by
  sorry

end two_colorable_regions_l787_787427


namespace part_I_part_II_l787_787284

variable (x : ℝ)

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Define complement of B in real numbers
def neg_RB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Part I: Statement for a = -2
theorem part_I (a : ℝ) (h : a = -2) : A a ∩ neg_RB = {x | -1 ≤ x ∧ x ≤ 1} := by
  sorry

-- Part II: Statement for A ∪ B = B
theorem part_II (a : ℝ) (h : ∀ x, A a x -> B x) : a < -4 ∨ a > 5 := by
  sorry

end part_I_part_II_l787_787284


namespace sin_300_eq_neg_sqrt3_div_2_l787_787569

-- Defining the problem statement as a Lean theorem
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l787_787569


namespace double_sum_equals_three_halves_l787_787558

noncomputable def double_sum_expression : ℝ :=
  ∑' j : ℕ, ∑' k : ℕ, 3^(- 2 * k - j - (k + j)^2)

theorem double_sum_equals_three_halves :
  double_sum_expression = 3 / 2 :=
sorry

end double_sum_equals_three_halves_l787_787558


namespace prism_volume_l787_787527

def Prism := {l w h : ℝ // l * w = 15 ∧ w * h = 20 ∧ l * h = 24 ∧ h = real.sqrt 4}

theorem prism_volume : ∀ (p : Prism), (p.val.l * p.val.w * p.val.h) = 30 :=
by
  intro p
  let ⟨l, w, h, h_lw, h_wh, h_lh, h_h⟩ := p
  sorry

end prism_volume_l787_787527


namespace only_threes_remain_l787_787443

theorem only_threes_remain :
  ∃ k : ℕ, k ≥ 47 ∧ (∀ n1 n2 n3 : ℕ, (n1 mod 2 = 0 ∧ n2 mod 2 = 0 ∧ n3 mod 2 ≠ 0) 
  → (n1 ≤ 16 ∧ n2 ≤ 20 ∧ n3 ≤ 31) → n3 = k) :=
sorry

end only_threes_remain_l787_787443


namespace find_angle_CBO_l787_787705

theorem find_angle_CBO (BAO_eq_CAO : ∀ A B C O : Type, angle A B O = angle C A O)
                        (CBO_eq_ABO : ∀ A B C O : Type, angle C B O = angle A B O)
                        (ACO_eq_BCO : ∀ A B C O : Type, angle A C O = angle B C O)
                        (AOC_110 : ∀ A O C : Type, angle A O C = 110) :
                        angle C B O = 20 :=
by
  sorry

end find_angle_CBO_l787_787705


namespace albums_total_l787_787035

noncomputable def miriam_albums (katrina_albums : ℕ) := 5 * katrina_albums
noncomputable def katrina_albums (bridget_albums : ℕ) := 6 * bridget_albums
noncomputable def bridget_albums (adele_albums : ℕ) := adele_albums - 15
noncomputable def total_albums (adele_albums : ℕ) (bridget_albums : ℕ) (katrina_albums : ℕ) (miriam_albums : ℕ) := 
  adele_albums + bridget_albums + katrina_albums + miriam_albums

theorem albums_total (adele_has_30 : adele_albums = 30) : 
  total_albums 30 (bridget_albums 30) (katrina_albums (bridget_albums 30)) (miriam_albums (katrina_albums (bridget_albums 30))) = 585 := 
by 
  -- In Lean, replace the term 'sorry' below with the necessary steps to finish the proof
  sorry

end albums_total_l787_787035


namespace count_heavy_tailed_permutations_l787_787520

open Finset
open BigOperators

def is_heavy_tailed (s : List ℕ) : Prop :=
  s.take 3.sum < s.drop 3.sum

noncomputable def heavy_tailed_permutations :=
  (univ.permutations ({1, 2, 3, 4, 5, 6} : Finset ℕ).toList).count (λ s, is_heavy_tailed s)

theorem count_heavy_tailed_permutations :
  heavy_tailed_permutations = 540 :=
sorry

end count_heavy_tailed_permutations_l787_787520


namespace cos_gamma_prime_l787_787358

theorem cos_gamma_prime (x' y' z' : ℝ) (hx : 0 < x') (hy : 0 < y') (hz : 0 < z') 
  (cos_alpha' : cos ⁡(atan2 y' x') = 2 / 5) 
  (cos_beta' : cos ⁡(atan2 z' (x'² + y'²) ^ (1 / 2)) = 1 / 4) :
  (cos ⁡(atan2 y' (sqrt (x'^2 + y'^2 + z'^2))) = sqrt 311 / 20) :=
by
  sorry

end cos_gamma_prime_l787_787358


namespace relative_positions_l787_787305

-- Definitions for the conditions
def quadratic_function (c : ℝ) : ℝ → ℝ := λ x, x^2 - 2 * x + c

structure Point :=
(x : ℝ)
(y : ℝ)

variables (c y1 y2 y3 : ℝ)
variables (A : Point)
variables (B : Point)
variables (C : Point)

hypothesis A_def : A = Point.mk (-1) y1
hypothesis B_def : B = Point.mk (2) y2
hypothesis C_def : C = Point.mk (-3) y3
hypothesis A_on_curve : A.y = quadratic_function c A.x
hypothesis B_on_curve : B.y = quadratic_function c B.x
hypothesis C_on_curve : C.y = quadratic_function c C.x

theorem relative_positions :
  y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relative_positions_l787_787305


namespace triangle_side_b_eq_1_l787_787990

theorem triangle_side_b_eq_1 {a b c : ℝ} {B C : ℝ} 
  (ha : a = sqrt 3) 
  (hB : sin B = 1 / 2) 
  (hC : C = π / 6) 
  : b = 1 :=
by
  sorry

end triangle_side_b_eq_1_l787_787990


namespace strawberry_jelly_sales_l787_787890

def jelly_sales (grape strawberry raspberry plum : ℕ) : Prop :=
  grape = 2 * strawberry ∧
  raspberry = 2 * plum ∧
  raspberry = grape / 3 ∧
  plum = 6

theorem strawberry_jelly_sales {grape strawberry raspberry plum : ℕ}
    (h : jelly_sales grape strawberry raspberry plum) : 
    strawberry = 18 :=
by
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2
  sorry

end strawberry_jelly_sales_l787_787890


namespace unique_k_n_m_solution_l787_787582

-- Problem statement
theorem unique_k_n_m_solution :
  ∃ (k : ℕ) (n : ℕ) (m : ℕ), k = 1 ∧ n = 2 ∧ m = 3 ∧ 3^k + 5^k = n^m ∧
  ∀ (k₀ : ℕ) (n₀ : ℕ) (m₀ : ℕ), (3^k₀ + 5^k₀ = n₀^m₀ ∧ m₀ ≥ 2) → (k₀ = 1 ∧ n₀ = 2 ∧ m₀ = 3) :=
by
  sorry

end unique_k_n_m_solution_l787_787582


namespace sheep_human_ratio_l787_787768

noncomputable def ratio_sheep_to_humans (H S C : ℕ) : ℕ × ℕ :=
(H, S) := (100 / 2, 75 - 100 / 2) in
(S / Nat.gcd S H, H / Nat.gcd S H)

theorem sheep_human_ratio :
  ∀ H S C : ℕ, C = 2 * H → C = 100 → S + H = 75 → ratio_sheep_to_humans H S C = (1, 2) :=
by
  intros
  dsimp [ratio_sheep_to_humans]
  sorry

end sheep_human_ratio_l787_787768


namespace remaining_card_proof_l787_787838

-- Define the set of cards
def cards : Finset ℕ := {2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the statements by A, B, C, and D
def A_statement (x y : ℕ) : Prop :=
  (x = y + 1 ∨ y = x + 1) ∧ Nat.coprime x y

def B_statement (x y : ℕ) : Prop :=
  ¬Nat.coprime x y ∧ ¬(x % y = 0 ∨ y % x = 0)

def C_statement (x y : ℕ) : Prop :=
  (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 7) ∧ (y ≠ 2 ∧ y ≠ 3 ∧ y ≠ 5 ∧ y ≠ 7) ∧ Nat.coprime x y

def D_statement (x y : ℕ) : Prop :=
  (x % y = 0 ∨ y % x = 0) ∧ ¬Nat.coprime x y

-- The final goal statement to prove
theorem remaining_card_proof :
  ∃ (cards_A cards_B cards_C cards_D : Finset ℕ), 
    cards_A.card = 2 ∧ 
    cards_B.card = 2 ∧ 
    cards_C.card = 2 ∧ 
    cards_D.card = 2 ∧ 
    cards_A ∪ cards_B ∪ cards_C ∪ cards_D = cards.erase 7 ∧
    ∃ (a1 a2 : ℕ), a1 ∈ cards_A ∧ a2 ∈ cards_A ∧ A_statement a1 a2 ∧
    ∃ (b1 b2 : ℕ), b1 ∈ cards_B ∧ b2 ∈ cards_B ∧ B_statement b1 b2 ∧
    ∃ (c1 c2 : ℕ), c1 ∈ cards_C ∧ c2 ∈ cards_C ∧ C_statement c1 c2 ∧
    ∃ (d1 d2 : ℕ), d1 ∈ cards_D ∧ d2 ∈ cards_D ∧ D_statement d1 d2 :=
sorry

end remaining_card_proof_l787_787838


namespace min_radius_l787_787359

def Point := (ℝ × ℝ)
def Hexagon := {p : Point | --(((p definition inside the boundary of a hexagon with side length 1))--; }
def Coloring := Point → Fin 3

def valid_coloring (hex: Hexagon) (r: ℝ) (c: Coloring): Prop :=
  ∀ p1 p2 : Point, p1 ∈ hex → p2 ∈ hex → c p1 = c p2 → dist p1 p2 < r

theorem min_radius (hex: Hexagon) (c: Coloring) : 
  (∃ r, valid_coloring hex r c) → (∃ r = 3/2, valid_coloring hex r c) := by
  sorry

end min_radius_l787_787359


namespace least_value_of_quadratic_l787_787819

noncomputable def minimum_value (a b c : ℝ) (h : a > 0) : ℝ :=
(a : ℝ) (b : ℝ) (c : ℝ) (h : a > 0), let f (x : ℝ) := a * x^2 + (b + 5) * x + c in
let critical_x := - (b + 5) / (2 * a) in
f critical_x

theorem least_value_of_quadratic {a b c : ℝ} (h : a > 0) :
  minimum_value a b c h = (-b^2 - 10*b - 25 + 4*a*c) / (4*a) := by
soviert sorry

end least_value_of_quadratic_l787_787819


namespace news_spread_l787_787177

theorem news_spread {G : SimpleGraph (Fin 1000)} (connectedG : G.IsConnected) :
  ∃ (S : Finset (Fin 1000)), S.card = 90 ∧ ∀ v, ∃ u ∈ S, (G.path_length u v ≤ 10) :=
by
  sorry

end news_spread_l787_787177


namespace monotonic_intervals_inequality_holds_l787_787657

open Real

-- Definitions of the functions mentioned in the problem
def f (a x : ℝ) : ℝ := 1 + log x - a * x^2
def g (x : ℝ) : ℝ := (2 / exp 2) * (exp x / x^2)
def k (x : ℝ) : ℝ := log x / x

-- Proof statement for monotonic intervals
theorem monotonic_intervals (a x : ℝ) (hx : 0 < x) : 
  (a > 0 → 
    (∀ y, 0 < y ∧ y < (sqrt (2 * a) / (2 * a)) → (f a y) > 0) ∧ 
    (∀ z, (sqrt (2 * a) / (2 * a)) < z ∧ z < ∞ → (f a z) < 0)) ∧
  (a ≤ 0 → (∀ x, 0 < x → (f a x) > 0)) := 
sorry

-- Proof statement for the inequality involving the given functions
theorem inequality_holds (a x : ℝ) (hx : 0 < x) :
  x * (f a x) < (2 / exp 2) * (exp x) + x - a * x^3 :=
sorry

end monotonic_intervals_inequality_holds_l787_787657


namespace population_Lake_Bright_l787_787842

-- Definition of total population
def T := 80000

-- Definition of population of Gordonia
def G := (1 / 2) * T

-- Definition of population of Toadon
def Td := (60 / 100) * G

-- Proof that the population of Lake Bright is 16000
theorem population_Lake_Bright : T - (G + Td) = 16000 :=
by {
    -- Leaving the proof as sorry
    sorry
}

end population_Lake_Bright_l787_787842


namespace students_taking_neither_l787_787767

variable (total_students math_students physics_students both_students : ℕ)
variable (h1 : total_students = 80)
variable (h2 : math_students = 50)
variable (h3 : physics_students = 40)
variable (h4 : both_students = 25)

theorem students_taking_neither (h1 : total_students = 80)
    (h2 : math_students = 50)
    (h3 : physics_students = 40)
    (h4 : both_students = 25) :
    total_students - (math_students - both_students + physics_students - both_students + both_students) = 15 :=
by
    sorry

end students_taking_neither_l787_787767


namespace chessboard_selection_l787_787707

theorem chessboard_selection :
  let board : List (List Nat) := List.replicate 8 (List.replicate 8 0),
      rows_selected : Fin 8 → Fin 8 → Bool := sorry, -- Function to indicate if a cell is selected in a certain row
      cols_selected : Fin 8 → Fin 8 → Bool := sorry in
  -- Condition 1: All black squares are selected
  (∀ (r c : Fin 8), (r + c) % 2 = 1 → rows_selected r c = true) →
  -- Condition 2: Each row has exactly 7 squares selected
  (∀ r : Fin 8, (List.filter (λ c, rows_selected r c = true) (List.range 8)).length = 7) →
  -- Condition 3: Each column has exactly 7 squares selected
  (∀ c : Fin 8, (List.filter (λ r, cols_selected r c = true) (List.range 8)).length = 7) →
  -- Conclusion: Total selection ways
  (board.length * List.length (board.head?)).choose 56 = 576 :=
sorry

end chessboard_selection_l787_787707


namespace length_PR_is_20_l787_787335

-- Define the square side length
variable (s : ℝ)

-- Definitions involving areas of triangles
def total_area_cutoff (x y : ℝ) : ℝ := (x^2 + y^2)

-- Condition provided in the problem
variable condition1 : total_area_cutoff x y = 200

-- Define the length of PR based on the problem statement
def length_PR (x y : ℝ) : ℝ := real.sqrt (2 * total_area_cutoff x y)

-- Define the main theorem statement claiming PR is 20
theorem length_PR_is_20 (x y : ℝ) (h : total_area_cutoff x y = 200) : length_PR x y = 20 :=
by
  sorry

end length_PR_is_20_l787_787335


namespace largest_digit_change_l787_787805

theorem largest_digit_change (d : ℕ) : 
  (641 + 852 + 973 = 2456) → 
  (d ∈ {4, 5, 6, 7, 8}) →
  (641 + 852 + 963 = 2456) → 
  d = 7 := 
by
  intros h1 h2 h3
  sorry

end largest_digit_change_l787_787805


namespace parabola_focus_to_line_distance_l787_787077

-- Define the parabola and the line
def parabola_focus : ℝ × ℝ := (2, 0)
def line (x y : ℝ) : Prop := sqrt 3 * x - y = 0

-- Define the distance from a point to a line in ℝ²
def distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / sqrt (a^2 + b^2)

-- State the problem in Lean
theorem parabola_focus_to_line_distance : distance_point_to_line parabola_focus (sqrt 3) (-1) 0 = sqrt 3 :=
by
  -- Skipping the proof
  sorry

end parabola_focus_to_line_distance_l787_787077


namespace average_infection_rate_l787_787897

theorem average_infection_rate (x : ℕ) : 
  1 + x + x * (1 + x) = 81 :=
sorry

end average_infection_rate_l787_787897


namespace andrew_vacation_days_l787_787552

-- Andrew's working days and vacation accrual rate
def days_worked : ℕ := 300
def vacation_rate : Nat := 10
def vacation_days_earned : ℕ := days_worked / vacation_rate

-- Days off in March and September
def days_off_march : ℕ := 5
def days_off_september : ℕ := 2 * days_off_march
def total_days_off : ℕ := days_off_march + days_off_september

-- Remaining vacation days calculation
def remaining_vacation_days : ℕ := vacation_days_earned - total_days_off

-- Problem statement to prove
theorem andrew_vacation_days : remaining_vacation_days = 15 :=
by
  -- Substitute the known values and perform the calculation
  unfold remaining_vacation_days vacation_days_earned total_days_off vacation_rate days_off_march days_off_september days_worked
  norm_num
  sorry

end andrew_vacation_days_l787_787552


namespace arithmetic_expression_evaluation_l787_787932

theorem arithmetic_expression_evaluation :
  (1 / 6 * -6 / (-1 / 6) * 6) = 36 :=
by {
  sorry
}

end arithmetic_expression_evaluation_l787_787932


namespace sasha_stickers_l787_787397

variables (m n : ℕ) (t : ℝ)

-- Conditions
def conditions : Prop :=
  m < n ∧ -- Fewer coins than stickers
  m ≥ 1 ∧ -- At least one coin
  n ≥ 1 ∧ -- At least one sticker
  t > 1 ∧ -- t is greater than 1
  m * t + n = 100 ∧ -- Coin increase condition
  m + n * t = 101 -- Sticker increase condition

-- Theorem stating that the number of stickers must be 34 or 66
theorem sasha_stickers : conditions m n t → n = 34 ∨ n = 66 :=
sorry

end sasha_stickers_l787_787397


namespace total_apple_trees_l787_787926

-- Definitions and conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3
def total_trees : ℕ := ava_trees + lily_trees

-- Statement to be proved
theorem total_apple_trees :
  total_trees = 15 := by
  sorry

end total_apple_trees_l787_787926


namespace probability_adjacent_l787_787137

theorem probability_adjacent (a b c d e : Type) : 
  let total_arrangements := factorial 5 in
  let favorable_arrangements := factorial 4 * factorial 2 in
  (favorable_arrangements / total_arrangements : ℚ) = (2 / 5 : ℚ) := 
by 
  sorry

end probability_adjacent_l787_787137


namespace Danny_soda_left_l787_787574

theorem Danny_soda_left (bottles : ℕ) (drink_percent : ℝ) (gift_percent : ℝ) : 
  bottles = 3 → drink_percent = 0.9 → gift_percent = 0.7 → 
  let remaining1 := 1 - drink_percent in
  let remaining2 := 1 - gift_percent in
  (remaining1 + 2 * remaining2) = 0.7 := 
by
  intros h_bottles h_drink h_gift
  rw [h_bottles, h_drink, h_gift]
  let lhs := (1 - 0.9) + 2 * (1 - 0.7)
  have h : lhs = 0.7 := by 
    calc lhs = 0.1 + 2 * 0.3 : by rfl
        ... = 0.1 + 0.6 : by norm_num
        ... = 0.7 : by norm_num
  exact h

end Danny_soda_left_l787_787574


namespace area_of_G1G2G3_l787_787000

theorem area_of_G1G2G3 (A B C P G1 G2 G3 : Point) 
    (h_centroid1 : is_centroid(A, B, C, P)) 
    (h_centroid2 : is_centroid(P, B, C, G1)) 
    (h_centroid3 : is_centroid(P, C, A, G2)) 
    (h_centroid4 : is_centroid(P, A, B, G3)) 
    (h_area_ABC : triangle_area(A, B, C) = 24)
    : triangle_area(G1, G2, G3) = 8 / 3 :=
begin
    sorry
end

end area_of_G1G2G3_l787_787000


namespace log_sum_value_l787_787855

theorem log_sum_value :
  let log_base (b x : ℝ) := real.log x / real.log b in
  let log_identity (a b x : ℝ) := a * log_base b x = log_base b (x^a) in
  let log_add (b x y : ℝ) := log_base b x + log_base b y = log_base b (x * y) in
  let log_sub (b x y : ℝ) := log_base b x - log_base b y = log_base b (x / y) in
  log_base 10 16 + 3 * log_base 5 25 + 4 * log_base 10 2 + log_base 10 64 - log_base 10 8 = 9.311 := 
by 
  -- Proof would go here
  sorry

end log_sum_value_l787_787855


namespace trig_solve_identity_l787_787791

theorem trig_solve_identity {x y z : ℝ} (hx : cos x ≠ 0) (hy : cos y ≠ 0) :
  (cos x ^ 2 + 1 / cos x ^ 2) ^ 3 + (cos y ^ 2 + 1 / cos y ^ 2) ^ 3 = 16 * sin z ↔
  (∃ n k m : ℤ, x = n * Real.pi ∧ y = k * Real.pi ∧ z = Real.pi / 2 + 2 * m * Real.pi) :=
by
  sorry

end trig_solve_identity_l787_787791


namespace total_trip_duration_proof_l787_787451

-- Naming all components
def driving_time : ℝ := 5
def first_jam_duration (pre_first_jam_drive : ℝ) : ℝ := 1.5 * pre_first_jam_drive
def second_jam_duration (between_first_and_second_drive : ℝ) : ℝ := 2 * between_first_and_second_drive
def third_jam_duration (between_second_and_third_drive : ℝ) : ℝ := 3 * between_second_and_third_drive
def pit_stop_duration : ℝ := 0.5
def pit_stops : ℕ := 2
def initial_drive : ℝ := 1
def second_drive : ℝ := 1.5

-- Additional drive time calculation
def remaining_drive : ℝ := driving_time - initial_drive - second_drive

-- Total duration calculation
def total_duration (initial_drive : ℝ) (second_drive : ℝ) (remaining_drive : ℝ) (first_jam_duration : ℝ) 
(second_jam_duration : ℝ) (third_jam_duration : ℝ) (pit_stop_duration : ℝ) (pit_stops : ℕ) : ℝ :=
  driving_time + first_jam_duration + second_jam_duration + third_jam_duration + (pit_stop_duration * pit_stops)

theorem total_trip_duration_proof :
  total_duration initial_drive second_drive remaining_drive (first_jam_duration initial_drive)
                  (second_jam_duration second_drive) (third_jam_duration remaining_drive) pit_stop_duration pit_stops 
  = 18 :=
by
  -- Proof steps would go here
  sorry

end total_trip_duration_proof_l787_787451


namespace sum_of_x_values_l787_787961

open Real

noncomputable def cos3 (θ : ℝ) := (cos θ) ^ 3

theorem sum_of_x_values :
  (∑ x in ({ x : ℝ | 0 < x ∧ x < 180 ∧ cos3 (2 * x) + cos3 (4 * x) = 8 * cos3 (3 * x) * cos3 x }).to_finset, x) = 540 :=
sorry

end sum_of_x_values_l787_787961


namespace pounds_over_weight_limit_l787_787342

-- Definitions based on conditions

def bookcase_max_weight : ℝ := 80

def weight_hardcover_book : ℝ := 0.5
def number_hardcover_books : ℕ := 70
def total_weight_hardcover_books : ℝ := number_hardcover_books * weight_hardcover_book

def weight_textbook : ℝ := 2
def number_textbooks : ℕ := 30
def total_weight_textbooks : ℝ := number_textbooks * weight_textbook

def weight_knick_knack : ℝ := 6
def number_knick_knacks : ℕ := 3
def total_weight_knick_knacks : ℝ := number_knick_knacks * weight_knick_knack

def total_weight_items : ℝ := total_weight_hardcover_books + total_weight_textbooks + total_weight_knick_knacks

theorem pounds_over_weight_limit : total_weight_items - bookcase_max_weight = 33 := by
  sorry

end pounds_over_weight_limit_l787_787342


namespace closed_set_example_l787_787622

def isClosed (A : Set ℤ) : Prop :=
  ∀ a b ∈ A, a + b ∈ A ∧ a - b ∈ A

theorem closed_set_example :
  isClosed {n : ℤ | ∃ k : ℤ, n = 3 * k} :=
by
  sorry

end closed_set_example_l787_787622


namespace craig_apples_total_l787_787219

-- Defining the conditions
def initial_apples_craig : ℝ := 20.0
def apples_from_eugene : ℝ := 7.0

-- Defining the total number of apples Craig will have
noncomputable def total_apples_craig : ℝ := initial_apples_craig + apples_from_eugene

-- The theorem stating that Craig will have 27.0 apples.
theorem craig_apples_total : total_apples_craig = 27.0 := by
  -- Proof here
  sorry

end craig_apples_total_l787_787219


namespace trisector_length_of_right_triangle_l787_787168

theorem trisector_length_of_right_triangle (DE EF : ℝ) (h1 : DE = 5) (h2 : EF = 12) :
  let DF := Real.sqrt (DE^2 + EF^2)
  let x := 60 / (12 * Real.sqrt 3 + 5)
  DF = 13 → 2 * x = (1440 * Real.sqrt 3 - 600) / 407 :=
by
  -- Definitions
  let DF := Real.sqrt (DE^2 + EF^2)
  let x := 60 / (12 * Real.sqrt 3 + 5)
  -- Calculate hypotenuse
  have DF_eq : DF = Real.sqrt (DE^2 + EF^2) := rfl
  -- Conditions from problem statement
  have h1 : DE = 5 := by sorry
  have h2 : EF = 12 := by sorry
  -- Given and prove statements
  have hypotenuse : DF = 13 := by sorry
  show 2 * x = (1440 * Real.sqrt 3 - 600) / 407 from sorry

end trisector_length_of_right_triangle_l787_787168


namespace trig_solve_identity_l787_787790

theorem trig_solve_identity {x y z : ℝ} (hx : cos x ≠ 0) (hy : cos y ≠ 0) :
  (cos x ^ 2 + 1 / cos x ^ 2) ^ 3 + (cos y ^ 2 + 1 / cos y ^ 2) ^ 3 = 16 * sin z ↔
  (∃ n k m : ℤ, x = n * Real.pi ∧ y = k * Real.pi ∧ z = Real.pi / 2 + 2 * m * Real.pi) :=
by
  sorry

end trig_solve_identity_l787_787790


namespace intersection_M_N_l787_787022

def M : Set ℝ := { x | x < 2017 }
def N : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } := 
by 
  sorry

end intersection_M_N_l787_787022


namespace comprehensive_satisfaction_equality_l787_787542

theorem comprehensive_satisfaction_equality {m_A m_B : ℝ} :
  (m_A = (3 / 5) * m_B) →
  (h_A = sqrt (m_A / (m_A + 12) * m_B / (m_B + 5))) →
  (h_B = sqrt (m_A / (m_A + 3) * m_B / (m_B + 20))) →
  h_A = h_B :=
by sorry

end comprehensive_satisfaction_equality_l787_787542


namespace smallest_d_value_l787_787166

noncomputable def smallest_d : ℝ :=
  let d := real.sqrt 3625 / 24 in
  if 5 + d > 0 then d else - d

theorem smallest_d_value (d : ℝ) (H : real.sqrt ((5 * real.sqrt 5)^2 + (d + 5)^2) = 5 * d) : d = smallest_d :=
by sorry

end smallest_d_value_l787_787166


namespace fourth_grade_students_l787_787324

theorem fourth_grade_students (initial_students left_students new_students final_students : ℕ) 
    (h1 : initial_students = 33) 
    (h2 : left_students = 18) 
    (h3 : new_students = 14) 
    (h4 : final_students = initial_students - left_students + new_students) :
    final_students = 29 := 
by 
    sorry

end fourth_grade_students_l787_787324


namespace roots_inequality_l787_787390

theorem roots_inequality (p : ℝ) (hp : p ≠ 0) (x1 x2 : ℝ) 
  (hroots : ∀ (r : ℝ), r = x1 ∨ r = x2 ↔ r^2 + p * r - 1/(2 * p^2) = 0) :
  x1^4 + x2^4 ≥ 2 + real.sqrt 2 := 
sorry

end roots_inequality_l787_787390


namespace cos_gamma_prime_l787_787356

theorem cos_gamma_prime : 
  ∀ (α' β' γ' : ℝ), 
  ∃ (cα cβ cγ : ℝ), 
    cα = (2 / 5) ∧
    cβ = (1 / 4) ∧ 
    (cα ^ 2 + cβ ^ 2 + cγ ^ 2 = 1) → 
    cγ = (sqrt 311 / 20) :=
by
  sorry

end cos_gamma_prime_l787_787356


namespace area_quadrilateral_cmkd_l787_787387

-- Definition of the problem in Lean 4
theorem area_quadrilateral_cmkd
  (parallelogram : Type)
  (A B C D M K : parallelogram)
  (area_ABCD : ℝ)
  (H1 : ∃ (BM_MC_ratio : ℝ), BM_MC_ratio = 1 / 2)
  (H2 : (segment AM) ∩ (diagonal BD) = K)
  (H3 : area_ABCD = 1) :
  area (quadrilateral C M K D) = 11 / 12 :=
sorry

end area_quadrilateral_cmkd_l787_787387


namespace total_albums_l787_787032

-- Definitions based on given conditions
def adele_albums : ℕ := 30
def bridget_albums : ℕ := adele_albums - 15
def katrina_albums : ℕ := 6 * bridget_albums
def miriam_albums : ℕ := 5 * katrina_albums

-- The final statement to be proved
theorem total_albums : adele_albums + bridget_albums + katrina_albums + miriam_albums = 585 :=
by
  sorry

end total_albums_l787_787032


namespace Jason_reroll_exactly_two_dice_probability_l787_787344

noncomputable def probability_reroll_two_dice : ℚ :=
  let favorable_outcomes := 5 * 3 + 1 * 3 + 5 * 3
  let total_possibilities := 6^3
  favorable_outcomes / total_possibilities

theorem Jason_reroll_exactly_two_dice_probability : probability_reroll_two_dice = 5 / 9 := 
  sorry

end Jason_reroll_exactly_two_dice_probability_l787_787344


namespace radius_of_circle_l787_787093

noncomputable def center : Type := ℝ
noncomputable def radius (h : center) : ℝ := 
  if h = -5 then 5 * Real.sqrt 2 else 0

theorem radius_of_circle : radius (-5) = 5 * Real.sqrt 2 := 
by sorry

end radius_of_circle_l787_787093


namespace paintable_integer_sum_correct_l787_787666

def is_paintable (h t u : ℕ) : Prop :=
  (h ≥ 3 ∧ t ≥ 3 ∧ u ≥ 3) ∧ 
  (∀ n : ℕ, ∃ k1 k2 k3 : ℕ, n = k1 * h ∨ n = k2 * t + 1 ∨ n = k3 * u + 2) ∧
  (∀ m n : ℕ, m ≠ n → ∃ k1 k2 : ℕ, m ≠ k1 * h ∧ n ≠ k2 * t + 1 ∧ m ≠ k1 * u + 2)

noncomputable def paintable_integers_sum : ℕ :=
  ∑ (h t u : ℕ) in {h | is_paintable h t u}, (100 * h + 10 * t + u)

theorem paintable_integer_sum_correct : paintable_integers_sum = 757 := by
  sorry

end paintable_integer_sum_correct_l787_787666


namespace negation_of_universal_proposition_l787_787823

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
sorry

end negation_of_universal_proposition_l787_787823


namespace sara_draws_l787_787499

theorem sara_draws (n : ℕ) (k : ℕ) (h_n : n = 15) (h_k : k = 4) : (n ^ k) = 50625 :=
by
  rw [h_n, h_k]
  norm_num
  exact rfl

end sara_draws_l787_787499


namespace partition_impossible_l787_787561

theorem partition_impossible :
  ¬ ∃ (groups : list (list ℕ)),
    (∀ g ∈ groups, g.length = 3 ∧ ∃ a b c, g = [a, b, c] ∧ a = b + c ∨ b = a + c ∨ c = a + b) ∧
    (∃ (l : list ℕ), l = list.range' 1 33 ∧ list.join groups = l) ∧
    (list.length groups = 11) :=
begin
  sorry
end

end partition_impossible_l787_787561


namespace find_principal_l787_787862

theorem find_principal (r : ℝ) (t : ℝ) (A : ℝ) (n : ℕ) (h_r : r = 0.11) (h_t : t = 2.4) (h_A : A = 1120) (h_n : n = 1) : 
    ∃ P : ℝ, P = A / (1 + r / n) ^ (n * t) ∧ P ≈ 863.84 :=
by
  have r_eq : r = 0.11 := h_r
  have t_eq : t = 2.4 := h_t
  have A_eq : A = 1120 := h_A
  have n_eq : n = 1 := h_n
  let P := A / (1 + r / n) ^ (n * t)
  use P
  split
  · rw [r_eq, t_eq, A_eq, n_eq]
    have : (1 + 0.11 / 1) = 1.11 := by norm_num
    specialize pow_eq 1.11 2.4
    norm_num at this
    sorry
  · sorry

end find_principal_l787_787862


namespace double_root_divisors_l787_787522

theorem double_root_divisors (b3 b2 b1 s : ℤ) (h : 0 = (s^2) • (x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 50)) : 
  s = -5 ∨ s = -1 ∨ s = 1 ∨ s = 5 :=
by
  sorry

end double_root_divisors_l787_787522


namespace exists_t_for_line_l787_787919

noncomputable def parametric_equation_line_proof : Prop :=
  ∃ t : ℝ, (3 + t) + (1 - t) = 4

theorem exists_t_for_line : 
  parametric_equation_line_proof ∧ ∀ x y : ℝ, x + y = 4 → x + y - 2 = 2 :=
by
  constructor
  sorry
  intros x y h
  calc
    x + y - 2 = 4 - 2 := by rw [h]
            ... = 2     := by norm_num


end exists_t_for_line_l787_787919


namespace radius_of_larger_circle_l787_787241

open Real

theorem radius_of_larger_circle (r : ℝ) (R : ℝ) (h1 : r = 2)
  (h2 : ∀ (C1 C2 : ℝ × ℝ), (sq_dist C1 C2 = (2 * r) ^ 2))
  (h3 : R = r * (1 + 1 * sqrt 2)) : R = 2 * (1 + sqrt 2) := 
by
  rw [h3, h1]
  ring_nf
  rw [sqrt_two_mul_sqrt_two]
  simp
  sorry

end radius_of_larger_circle_l787_787241


namespace quotient_equivalence_l787_787478

variable (N H J : ℝ)

theorem quotient_equivalence
  (h1 : N / H = 1.2)
  (h2 : H / J = 5 / 6) :
  N / J = 1 := by
  sorry

end quotient_equivalence_l787_787478


namespace min_packs_for_soda_l787_787785

theorem min_packs_for_soda (max_packs : ℕ) (packs : List ℕ) : 
  let num_cans := 95
  let max_each_pack := 4
  let pack_8 := packs.count 8 
  let pack_15 := packs.count 15
  let pack_18 := packs.count 18
  pack_8 ≤ max_each_pack ∧ pack_15 ≤ max_each_pack ∧ pack_18 ≤ max_each_pack ∧ 
  pack_8 * 8 + pack_15 * 15 + pack_18 * 18 = num_cans ∧ 
  pack_8 + pack_15 + pack_18 = max_packs → max_packs = 6 :=
sorry

end min_packs_for_soda_l787_787785


namespace cos_of_angle_through_point_l787_787270

variable (α : RealAngle) (P : Point2D)
variable (hx : P.x = -6) (hy : P.y = 8)

theorem cos_of_angle_through_point : cos α = -3 / 5 :=
sorry

end cos_of_angle_through_point_l787_787270


namespace squirrel_rise_per_circuit_l787_787532

theorem squirrel_rise_per_circuit
    (post_height : ℕ)
    (post_circumference : ℕ)
    (squirrel_travel : ℕ)
    (H1 : post_height = 16)
    (H2 : post_circumference = 3)
    (H3 : squirrel_travel = 12) :
    post_height / (squirrel_travel / post_circumference) = 4 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end squirrel_rise_per_circuit_l787_787532


namespace greatest_power_of_2_factor_of_expression_l787_787851

theorem greatest_power_of_2_factor_of_expression :
  ∀ (a b : ℤ), 15 = 3 * 5 → 6 = 2 * 3 → (a = 15) → (b = 6) →
  ∃ k : ℕ, 2^k ∣ a^702 - b^351 ∧ ∀ m : ℕ, 2^m ∣ a^702 - b^351 → m ≤ 351 :=
begin
  sorry
end

end greatest_power_of_2_factor_of_expression_l787_787851


namespace remainder_of_82460_div_8_l787_787473

theorem remainder_of_82460_div_8 :
  82460 % 8 = 4 :=
sorry

end remainder_of_82460_div_8_l787_787473


namespace radius_of_larger_circle_theorem_l787_787243

noncomputable def radius_of_larger_circle (r : ℝ) : ℝ := 2 * (1 + real.sqrt 2)

theorem radius_of_larger_circle_theorem :
  ∀ (r : ℝ), (∀ C₁ C₂ C₃ C₄ : { c : ℝ × ℝ // (c.1 ^ 2 + c.2 ^ 2) = r ^ 2 },
                  ∀ L : { c : ℝ × ℝ // (c.1 ^ 2 + c.2 ^ 2) = (radius_of_larger_circle r) ^ 2 },
                      dist (C₁ : ℝ × ℝ) (C₂ : ℝ × ℝ) = 2 * r ∧
                      dist (C₂ : ℝ × ℝ) (C₃ : ℝ × ℝ) = 2 * r ∧
                      dist (C₃ : ℝ × ℝ) (C₄ : ℝ × ℝ) = 2 * r ∧
                      dist (C₄ : ℝ × ℝ) (C₁ : ℝ × ℝ) = 2 * r ∧
                      dist (C₁ : ℝ × ℝ) (L : ℝ × ℝ) = radius_of_larger_circle r - r ∧
                      dist (C₂ : ℝ × ℝ) (L : ℝ × ℝ) = radius_of_larger_circle r - r ∧
                      dist (C₃ : ℝ × ℝ) (L : ℝ × ℝ) = radius_of_larger_circle r - r ∧
                      dist (C₄ : ℝ × ℝ) (L : ℝ × ℝ) = radius_of_larger_circle r - r)
      → radius_of_larger_circle r = 2 * (1 + real.sqrt 2) :=
by 
  sorry

end radius_of_larger_circle_theorem_l787_787243


namespace parking_lot_total_spaces_l787_787519

theorem parking_lot_total_spaces (ratio_fs_cc : ℕ) (ratio_cc_fs : ℕ) (fs_spaces : ℕ) (total_spaces : ℕ) 
  (h1 : ratio_fs_cc = 11) (h2 : ratio_cc_fs = 4) (h3 : fs_spaces = 330) :
  total_spaces = 450 :=
by
  sorry

end parking_lot_total_spaces_l787_787519


namespace find_parabola_parameters_l787_787658

noncomputable def parabola_properties (p m : ℝ) (hp : p > 0) (A : ℝ × ℝ) (hA : A = (m, 4)) : Prop :=
  let focus := (0, p / 2) in
  let directrix := -p / 2 in
  let distance_to_focus := real.sqrt ((m - 0)^2 + (4 - (p / 2))^2) in
  let distance_to_directrix := abs (4 - directrix) in
  (distance_to_focus = 17 / 4) ∧ (distance_to_focus = distance_to_directrix)

theorem find_parabola_parameters (m : ℝ) (p : ℝ) (hp : p > 0) (A : ℝ × ℝ) (hA : A = (m, 4))
  (h_dist : parabola_properties p m hp A hA) :
  p = 1 / 2 ∧ (m = 2 ∨ m = -2) :=
sorry

end find_parabola_parameters_l787_787658


namespace rational_square_root_l787_787614

theorem rational_square_root {x y : ℚ} 
  (h : (x^2 + y^2 - 2) * (x + y)^2 + (xy + 1)^2 = 0) : 
  ∃ r : ℚ, r * r = 1 + x * y := 
sorry

end rational_square_root_l787_787614


namespace domain_of_f_l787_787229

noncomputable def f (x : ℝ) : ℝ := real.sqrt (-8 * x^2 + 10 * x + 3)

def domain_f : set ℝ := {x | -8 * x^2 + 10 * x + 3 ≥ 0}

theorem domain_of_f :
  domain_f = set.Icc (-(1:ℝ)/4) (3/4) :=
by
  sorry

end domain_of_f_l787_787229


namespace probability_diff_geq_3_l787_787464

open Finset
open BigOperators

-- Define the set of numbers
def number_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the condition for pairs with positive difference ≥ 3
def valid_pair (a b : ℕ) : Prop := a ≠ b ∧ abs (a - b) ≥ 3

-- Calculate the probability as a fraction
theorem probability_diff_geq_3 : 
  let total_pairs := (number_set.product number_set).filter (λ ab, ab.1 < ab.2) in
  let valid_pairs := total_pairs.filter (λ ab, valid_pair ab.1 ab.2) in
  valid_pairs.card / total_pairs.card = 7 / 12 :=
by
  sorry

end probability_diff_geq_3_l787_787464


namespace line_parallel_not_coincident_l787_787309

theorem line_parallel_not_coincident (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ k : ℝ, (∀ x y : ℝ, a * x + 2 * y + 6 = k * (x + (a - 1) * y + (a^2 - 1))) →
  a = -1 :=
by
  sorry

end line_parallel_not_coincident_l787_787309


namespace parallel_lines_not_coincident_l787_787307

theorem parallel_lines_not_coincident (a : ℝ) :
  (∀ x y : ℝ, ax + 2 * y + 6 = 0) ∧
  (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ (b : ℝ), ∀ x y : ℝ, ax + 2 * y + b = 0 ∧ x + (a - 1) * y + b = 0 →
  a = -1 :=
by
  sorry

end parallel_lines_not_coincident_l787_787307


namespace problem_statement_l787_787749

variables {A B C M N L : Type} [EuclideanGeometry A]
variables {a b c : ℝ}

-- Let M, N be midpoints of AB, AC respectively
def midpoint (P Q : A) (M : A) : Prop := dist P M = dist Q M ∧ dist P M + dist M Q = dist P Q

-- Condition given in the problem
variables (h1 : midpoint A B M) (h2 : midpoint A C N) (h3: dist C M / dist A C = (Real.sqrt 3) / 2)

-- The theorem we need to prove
theorem problem_statement (B N A : Type) [EuclideanGeometry A] (h1 : midpoint A B M) (h2 : midpoint A C N)
  (h3: dist C M / dist A C = Real.sqrt 3 / 2) :
  dist B N / dist A B = Real.sqrt 3 / 2 :=
by
  sorry

end problem_statement_l787_787749


namespace time_to_pass_tree_l787_787180

-- Define the conditions given in the problem
def train_length : ℕ := 1200
def platform_length : ℕ := 700
def time_to_pass_platform : ℕ := 190

-- Calculate the total distance covered while passing the platform
def distance_passed_platform : ℕ := train_length + platform_length

-- The main theorem we need to prove
theorem time_to_pass_tree : (distance_passed_platform / time_to_pass_platform) * train_length = 120 := 
by
  sorry

end time_to_pass_tree_l787_787180


namespace man_work_rate_l787_787518

theorem man_work_rate (W : ℝ) : 
  let M := W / 4 in
  M = W / (W / 4) :=
by 
  -- Assuming the combined work rate of man and son is W/3 per day
  let combined_work_rate := W / 3

  -- Assuming the son's work rate is W/12 per day
  let son_work_rate := W / 12

  -- The equation for combined work rate: M + son_work_rate = combined_work_rate
  have h : M + son_work_rate = combined_work_rate, from sorry

  -- Substituting son's work rate into the equation
  have h_sub : M + W / 12 = W / 3, from sorry

  -- Solving for M
  sorry

end man_work_rate_l787_787518


namespace find_Sn_sum_sequence_19_l787_787363

variables (a_n : ℕ → ℕ) (S_n : ℕ → ℤ) (n : ℕ)

def a_7 : Prop := a_n 7 = 5
def S_5 : Prop := S_n 5 = -55
def b_n (n : ℕ) : ℤ := S_n n / n

theorem find_Sn (hn : ∀ n, S_n n = 2 * n^2 - 21 * n) : ∀ n, S_n n = 2 * n^2 - 21 * n :=
by sorry

theorem sum_sequence_19 (h_S : S_5 ∧ a_7) (hn : ∀ n, S_n n = 2 * n^2 - 21 * n) : 
  (∑ i in finset.range 19, (1 : ℤ) / (b_n i * b_n (i + 1))) = -1 / 19 :=
by sorry

end find_Sn_sum_sequence_19_l787_787363


namespace units_digit_k_squared_plus_2_k_l787_787743

noncomputable def k : ℕ := 2009^2 + 2^2009 - 3

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_k_squared_plus_2_k : units_digit (k^2 + 2^k) = 1 := by
  sorry

end units_digit_k_squared_plus_2_k_l787_787743


namespace lcm_210_913_eq_2310_l787_787083

theorem lcm_210_913_eq_2310 : Nat.lcm 210 913 = 2310 := by
  sorry

end lcm_210_913_eq_2310_l787_787083


namespace surface_area_of_glued_cubes_l787_787120

noncomputable def calculate_surface_area (large_cube_edge_length : ℕ) : ℕ :=
sorry

theorem surface_area_of_glued_cubes :
  calculate_surface_area 4 = 136 :=
sorry

end surface_area_of_glued_cubes_l787_787120


namespace probability_of_desired_event_l787_787299

-- The problem condition: A fair six-sided die
def fair_six_sided_die : finset ℕ := {1, 2, 3, 4, 5, 6}

-- Roll the die five times
def roll_die_five_times : finset (fin 6 → ℕ) :=
  finset.pi_finset (fin 5) (λ _, fair_six_sided_die)

-- Event: Rolling a 1 at least twice
def at_least_two_ones (rolls : fin 5 → ℕ) : Prop :=
  (finset.filter (λ i, rolls i = 1) finset.univ).card ≥ 2

-- Event: Rolling a 2 at least twice
def at_least_two_twos (rolls : fin 5 → ℕ) : Prop :=
  (finset.filter (λ i, rolls i = 2) finset.univ).card ≥ 2

-- Event: Rolling one other number
def exactly_one_other_number (rolls : fin 5 → ℕ) : Prop :=
  (finset.filter (λ i, rolls i ≠ 1 ∧ rolls i ≠ 2) finset.univ).card = 1

-- Complete event: Rolling the number 1 at least twice and the number 2 at least twice
def desired_event (rolls : fin 5 → ℕ) : Prop :=
  at_least_two_ones rolls ∧ at_least_two_twos rolls ∧ exactly_one_other_number rolls

-- The measure for uniform probability space
noncomputable def uniform_measure : measure (fin 5 → ℕ) :=
  measure.pi 5 (λ _, measure_uniform (set.to_finite {1, 2, 3, 4, 5, 6}))

-- The statement of the probability
theorem probability_of_desired_event :
  (uniform_measure {rolls : fin 5 → ℕ | desired_event rolls} / uniform_measure {rolls : fin 5 → ℕ | true}) = 5 / 324 :=
by 
  sorry

end probability_of_desired_event_l787_787299


namespace mode_unique_value_l787_787623

theorem mode_unique_value (x : ℕ) (h : ∀ y ∈ {3, x, 6, 5, 4}, y = 4 → y ≠ 4 → false) : x = 4 := by
  sorry

end mode_unique_value_l787_787623


namespace number_of_red_candies_is_4_l787_787837

-- Define the parameters as given in the conditions
def number_of_green_candies : ℕ := 5
def number_of_blue_candies : ℕ := 3
def likelihood_of_blue_candy : ℚ := 25 / 100

-- Define the total number of candies
def total_number_of_candies (number_of_red_candies : ℕ) : ℕ :=
  number_of_green_candies + number_of_blue_candies + number_of_red_candies

-- Define the proof statement
theorem number_of_red_candies_is_4 (R : ℕ) :
  (3 / total_number_of_candies R = 25 / 100) → R = 4 :=
sorry

end number_of_red_candies_is_4_l787_787837


namespace find_y_l787_787237

theorem find_y (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : y = 5 :=
by
  sorry

end find_y_l787_787237


namespace triangle_shape_l787_787310

open Float

theorem triangle_shape (A B C : ℝ) (a b c : ℝ) (h1 : sin A / sin B = 5 / 11) (h2 : sin B / sin C = 11 / 13) (h3 : ∀ (t : ℝ), t ≠ 0 → a = 5 * t ∧ b = 11 * t ∧ c = 13 * t) : A + B + C = π → cos C < 0 :=
sorry

end triangle_shape_l787_787310


namespace matrix_problem_l787_787352

variables {n : ℕ}
variables {α : ℂ} {A : Matrix (Fin n) (Fin n) ℂ}

theorem matrix_problem 
  (h1 : α ≠ 0)
  (h2 : A ≠ 0)
  (h3 : A * A + (A.conj_transpose) * (A.conj_transpose) = α • (A * A.conj_transpose))
  (h4 : A.conj_transpose = Matrix.transpose (A.map Complex.conj)):
  α ∈ ℝ ∧ abs α ≤ 2 ∧ A * A.conj_transpose = A.conj_transpose * A :=
sorry

end matrix_problem_l787_787352


namespace total_candy_l787_787196

/-- Bobby ate 26 pieces of candy initially. -/
def initial_candy : ℕ := 26

/-- Bobby ate 17 more pieces of candy thereafter. -/
def more_candy : ℕ := 17

/-- Prove that the total number of pieces of candy Bobby ate is 43. -/
theorem total_candy : initial_candy + more_candy = 43 := by
  -- The total number of candies should be 26 + 17 which is 43
  sorry

end total_candy_l787_787196


namespace Oliver_monster_club_cards_l787_787378

theorem Oliver_monster_club_cards (BG AB MC : ℕ) 
  (h1 : BG = 48) 
  (h2 : BG = 3 * AB) 
  (h3 : MC = 2 * AB) : 
  MC = 32 :=
by sorry

end Oliver_monster_club_cards_l787_787378


namespace measure_of_B_range_of_a2_b2_l787_787773

noncomputable def triangle := {a b c A B C : ℝ}

def condition_1 (t : triangle) := t.a * sin t.B - sqrt 3 * t.b * cos t.B * cos t.C = sqrt 3 * t.c * cos t.B ^ 2
def condition_2 (t : triangle) := t.b * cos t.C + 0.5 * t.c = t.a
def condition_3 (t : triangle) := sqrt 3 * t.b * sin t.A / (1 + cos t.B) = t.a
def acute_triangle (t : triangle) := t.A < π / 2 ∧ t.B < π / 2 ∧ t.C < π / 2

theorem measure_of_B (t : triangle) :
  (condition_1 t ∨ condition_2 t ∨ condition_3 t) →
  t.B = π / 3 :=
sorry

theorem range_of_a2_b2 (t : triangle) (hc : t.c = 1) (hacute : acute_triangle t) :
  (condition_1 t ∨ condition_2 t ∨ condition_3 t) →
  1 < t.a ^ 2 + t.b ^ 2 ∧ t.a ^ 2 + t.b ^ 2 < 7 :=
sorry

end measure_of_B_range_of_a2_b2_l787_787773


namespace MX_XC_ratio_l787_787075

theorem MX_XC_ratio (D K M B P : Point)
  (A B C D : Point)
  (parallelogram_ABCD : is_parallelogram A B C D)
  (DK_eq_KM : distance D K = distance K M)
  (BP_eq_quarter_BM : distance B P = 0.25 * distance B M)
  (X_intersection : ∃ X, on_line X M C ∧ on_plane X A K P) :
  ratio (distance M X) (distance X C) = 3 / 4 :=
sorry

end MX_XC_ratio_l787_787075


namespace highest_temperature_l787_787665

theorem highest_temperature (lowest_temp : ℝ) (max_temp_diff : ℝ) :
  lowest_temp = 18 → max_temp_diff = 4 → lowest_temp + max_temp_diff = 22 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end highest_temperature_l787_787665


namespace distance_center_to_plane_l787_787989

-- Given definitions
variables {A B C O : Type}
variables {r : ℝ} (h_r : r = 13)
variables {AB BC AC : ℝ} (h_AB : AB = 6) (h_BC : BC = 8) (h_AC : AC = 10)

-- Proof statement 
theorem distance_center_to_plane {d : ℝ} (hd : d = 12) : 
  ∃ O A B C : Type, (r = 13) ∧ (AB = 6) ∧ (BC = 8) ∧ (AC = 10) ∧ (distance O (plane A B C) = 12) :=
sorry

end distance_center_to_plane_l787_787989


namespace incorrect_eq_count_is_3_l787_787337

variables (x : ℚ) (e1 e2 e3 e4 : ℚ) -- Define variables for equations

def eq1_lhs := (3 - 2 * x) / 3 - (x - 2) / 2 = 1
def eq1_transformed := 2 * (3 - 2 * x) - 3 * (x - 2) = 6

def eq2_lhs := 3 * x + 8 = -4 * x - 7
def eq2_transformed := 3 * x + 4 * x = 7 - 8

def eq3_lhs := 7 * (3 - x) - 5 * (x - 3) = 8
def eq3_transformed := 21 - 7 * x - 5 * x + 15 = 8

def eq4_lhs := (3 / 7) * x = 7 / 3
def eq4_transformed := x = 49 / 9

-- Define the number of incorrect transformations
noncomputable def number_of_incorrect_eqs :=
  let incorrect_eq1 := (eq1_transformed ≠ 2 * (3 - 2 * x) - 3 * (x - 2) = 6 * 1)
      incorrect_eq2 := (eq2_transformed ≠ 3 * x + 4 * x = -15)
      incorrect_eq3 := (eq3_transformed = 21 - 7 * x - 5 * x + 15 = 8)
      incorrect_eq4 := (eq4_transformed = x = 49 / 9)
  in 
  if incorrect_eq1 then 1 else 0 + if incorrect_eq2 then 1 else 0 + if ¬incorrect_eq3 then 0 else 0 + if ¬incorrect_eq4 then 0 else 0

theorem incorrect_eq_count_is_3 : number_of_incorrect_eqs = 3 :=
by 
  sorry

end incorrect_eq_count_is_3_l787_787337


namespace geometric_sequence_square_sum_l787_787645

theorem geometric_sequence_square_sum (a : ℕ → ℝ) (n : ℕ) 
  (h1 : ∀ i, a (i + 1) = sqrt 2 * a i) 
  (h2 : a 0 = 4) : 
  (∑ i in Finset.range n, (a i)^2) = 2^(n + 4) - 16 :=
by
  sorry

end geometric_sequence_square_sum_l787_787645


namespace length_FX_correct_l787_787780

noncomputable def hexagon_length_FX (side_AB: ℝ) (AX_to_AB: ℝ) (side_length: ℝ): ℝ :=
let AB := side_length,
    AX := AX_to_AB * AB,
    PX := AX + AB,
    AP := AB / 2,
    PF := AP * Real.sqrt 3 in
Real.sqrt (PF^2 + PX^2)

theorem length_FX_correct : hexagon_length_FX 2 4 2 = 2 * Real.sqrt 21 := by
  sorry

end length_FX_correct_l787_787780


namespace sales_from_11_to_12_l787_787589

/-- During a promotional event, the sales from 9:00 AM to 10:00 AM amounted to 25,000 RMB. 
    The frequency distribution histogram indicates that the sales from 11:00 AM to 12:00 PM 
    were 4 times the sales from 9:00 AM to 10:00 AM. 
    Prove that the sales from 11:00 AM to 12:00 PM were 100,000 RMB.
-/
theorem sales_from_11_to_12 
  (sales_9_to_10 : ℕ) 
  (multiplier : ℕ) 
  (sales_9_to_10_value : sales_9_to_10 = 25000) 
  (multiplier_value : multiplier = 4) : 
  sales_9_to_10 * multiplier = 100000 := 
by
  rw [sales_9_to_10_value, multiplier_value]
  norm_num
  sorry

end sales_from_11_to_12_l787_787589


namespace find_x_l787_787291

noncomputable def collinear (u v : ℝ × ℝ) := ∃ k : ℝ, u = k • v

theorem find_x
  (a b : ℝ × ℝ)
  (hab : ¬ collinear a b)
  (x : ℝ) :
  let c := (x • a) + b,
      d := a + ((2 * x - 1) • b) in
  collinear c d ↔ (x = 1 ∨ x = -1 / 2) :=
by
sorry

end find_x_l787_787291


namespace number_of_refuels_needed_l787_787154

noncomputable def fuelTankCapacity : ℕ := 50
noncomputable def distanceShanghaiHarbin : ℕ := 2560
noncomputable def fuelConsumptionRate : ℕ := 8
noncomputable def safetyFuel : ℕ := 6

theorem number_of_refuels_needed
  (fuelTankCapacity : ℕ)
  (distanceShanghaiHarbin : ℕ)
  (fuelConsumptionRate : ℕ)
  (safetyFuel : ℕ) :
  (fuelTankCapacity = 50) →
  (distanceShanghaiHarbin = 2560) →
  (fuelConsumptionRate = 8) →
  (safetyFuel = 6) →
  ∃ n : ℕ, n = 4 := by
  sorry

end number_of_refuels_needed_l787_787154


namespace quadrilateral_is_trapezoid_l787_787712

theorem quadrilateral_is_trapezoid
  (AB CD : ℝ)
  (h_parallel : AB ∥ CD)
  (h_roots : ∃ m : ℝ, ∃ x : ℝ, x^2 - 3 * m * x + (2 * m^2 + m - 2) = 0 ∧ x ≠ AB ∧ x ≠ CD ∧ AB ≠ CD) :
  is_trapezoid AB CD :=
sorry

end quadrilateral_is_trapezoid_l787_787712


namespace points_concyclic_l787_787755

-- Let ABC be an acute triangle
variables {A B C : Type} [acute_triangle A B C]

-- Assume AC < BC
variable (h1: AC < BC)

-- A circle passing through A and B, intersects AC at A1 and BC at B1
variables {A₁ B₁ : Type}
variable (h2 : circle_through A B → intersects AC at A₁ → intersects BC at B₁)

-- Circles ∪ A₁B₁C and ∪ ABC intersect at points C and P
variables {P : Type}
variable (h3 : circles_intersect (A₁B₁C) (ABC) at C P)

-- Segments A₁B and AB₁ intersect at S
variable {S : Type}
variable (h4 : intersect_segments (A₁B) (AB₁) at S)

-- Let Q and R be the reflections of S with respect to CA and CB respectively
variables {Q R : Type}
variable (h5: reflection_of S with_respect_to CA at Q)
variable (h6: reflection_of S with_respect_to CB at R)

-- Prove that points P, Q, R and C are concyclic
theorem points_concyclic {A B C A₁ B₁ P S Q R : Type} 
  (h1 : AC < BC)
  (h2 : circle_through A B → intersects AC at A₁ → intersects BC at B₁)
  (h3 : circles_intersect (A₁B₁C) (ABC) at C P)
  (h4 : intersect_segments (A₁B) (AB₁) at S)
  (h5 : reflection_of S with_respect_to CA at Q)
  (h6 : reflection_of S with_respect_to CB at R) : 
  concyclic P Q R C :=
by sorry

end points_concyclic_l787_787755


namespace inscribed_circles_tangent_at_same_point_l787_787254

theorem inscribed_circles_tangent_at_same_point 
  (a b c d e : ℝ)
  (h_tangential : a + c = b + d) :
  let AM := (a + e - b) / 2
  let AN := (d + e - c) / 2
  in AM = AN := 
by
  sorry

end inscribed_circles_tangent_at_same_point_l787_787254


namespace jelly_sold_l787_787892

theorem jelly_sold (G S R P : ℕ) (h1 : G = 2 * S) (h2 : R = 2 * P) (h3 : R = G / 3) (h4 : P = 6) : S = 18 := by
  sorry

end jelly_sold_l787_787892


namespace inhabitable_land_fraction_l787_787301

theorem inhabitable_land_fraction (total_surface not_water_covered initially_inhabitable tech_advancement_viable : ℝ)
  (h1 : not_water_covered = 1 / 3 * total_surface)
  (h2 : initially_inhabitable = 1 / 3 * not_water_covered)
  (h3 : tech_advancement_viable = 1 / 2 * (not_water_covered - initially_inhabitable)) :
  (initially_inhabitable + tech_advancement_viable) / total_surface = 2 / 9 := 
sorry

end inhabitable_land_fraction_l787_787301


namespace rectangle_perimeter_eq_30sqrt10_l787_787900

theorem rectangle_perimeter_eq_30sqrt10 (A : ℝ) (l : ℝ) (w : ℝ) 
  (hA : A = 500) (hlw : l = 2 * w) (hArea : A = l * w) : 
  2 * (l + w) = 30 * Real.sqrt 10 :=
by
  sorry

end rectangle_perimeter_eq_30sqrt10_l787_787900


namespace meaningful_fraction_l787_787098

theorem meaningful_fraction (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) :=
by {
  sorry -- Proof goes here
}

end meaningful_fraction_l787_787098


namespace sum_of_first_5_terms_l787_787983

variables {a : ℕ → ℕ}
variable (q : ℝ)
variable (S_5 : ℝ)

axiom a2_a4_eq_a1 : a 2 * a 4 = a 1
axiom mean_a2_2a5_eq_5 : ((a 2) + 2 * (a 5)) / 2 = 5
def common_ratio (q : ℝ) := q
def first_term (q a : ℝ) := a / q
def sum_S5 (a1 q : ℝ) : ℝ := a1 * (1 - q^5) / (1 - q)

theorem sum_of_first_5_terms (q : ℝ) (a1 : ℝ)
  (h1 : a 2 * a 4 = a 1)
  (h2 : (a 2 + 2 * a 5) / 2 = 5) :
  S_5 = 31 :=
by
  sorry

end sum_of_first_5_terms_l787_787983


namespace area_of_region_l787_787082

variable (a : ℝ) (x y : ℝ)

theorem area_of_region (h1 : a > 0) :
  (λ (eq1 : (x + a * y)^3 = 8 * a^3)
     (eq2 : (a * x - y)^2 = 4 * a^2),
     let x1 := (2 * (1 + a)) / (a + 1/a),
         y1 := -(x1 / a) + 2,
         y2 := a * x1 - 2 * a,
         y3 := a * x1 + 2 * a in
     abs (a * x1 * y1 - a * x1 * y2) / 2 := 8 * a / (a^2 + 1)) sorry

end area_of_region_l787_787082


namespace tan_product_min_value_l787_787630

noncomputable def alpha : ℝ := sorry
noncomputable def beta : ℝ := sorry
noncomputable def gamma : ℝ := sorry

axiom acute_angles : (0 < alpha ∧ alpha < π / 2) ∧ (0 < beta ∧ beta < π / 2) ∧ (0 < gamma ∧ gamma < π / 2)
axiom cos_squared_sum : Real.cos(alpha) ^ 2 + Real.cos(beta) ^ 2 + Real.cos(gamma) ^ 2 = 1

theorem tan_product_min_value : Real.tan(alpha) * Real.tan(beta) * Real.tan(gamma) = 2 * Real.sqrt 2 := 
by 
  sorry

end tan_product_min_value_l787_787630


namespace G_G_G_G_G_1_eq_0_l787_787938

def G (x : ℝ) : ℝ := 2 * Real.sin (π * x / 2)

theorem G_G_G_G_G_1_eq_0 : G (G (G (G (G 1)))) = 0 :=
by
  sorry

end G_G_G_G_G_1_eq_0_l787_787938


namespace convert_sqrt2_exp_to_rectangular_l787_787208

noncomputable def convert_to_rectangular_form (z : ℂ) : ℂ :=
  z

def theta : ℝ := 13 * Real.pi / 4
def r : ℝ := Real.sqrt 2

theorem convert_sqrt2_exp_to_rectangular :
  convert_to_rectangular_form (r * Complex.exp (Complex.I * theta)) = (1 : ℂ) + (Complex.I) :=
by
  sorry

end convert_sqrt2_exp_to_rectangular_l787_787208


namespace root_of_linear_equation_l787_787144

theorem root_of_linear_equation (b c : ℝ) (hb : b ≠ 0) :
  ∃ x : ℝ, 0 * x^2 + b * x + c = 0 → x = -c / b :=
by
  -- The proof steps would typically go here
  sorry

end root_of_linear_equation_l787_787144


namespace angle_ADE_iff_ratios_l787_787311

theorem angle_ADE_iff_ratios {A B C D E : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] 
  (angle_A_eq_90 : ∑ (v : innerProductSpace), v = 90) 
  (AC_gt_AB : ∀ {v w : innerProductSpace}, v = AC → w = AB → v > w)
  (AB_AE_BD : ∀ {x y z : innerProductSpace}, x = AB → y = AE → z = BD → x = y ∧ y = z)
  (ADE_90_iff_ratios : ∀ {u v w x y z : innerProductSpace}, u = angle_ADE → v = 90 → 
    x = AB → y = AC → z = BC → u = v ↔ x : y : z = 3 : 4 : 5) :
  sorry

end angle_ADE_iff_ratios_l787_787311


namespace trihedral_angle_properties_l787_787047

-- Definitions for the problem's conditions
variables {α β γ : ℝ}
variables {A B C S : Type}
variables (angle_ASB angle_BSC angle_CSA : ℝ)

-- Given the conditions of the trihedral angle and the dihedral angles
theorem trihedral_angle_properties 
  (h1 : angle_ASB + angle_BSC + angle_CSA < 2 * Real.pi)
  (h2 : α + β + γ > Real.pi) : 
  true := 
by
  sorry

end trihedral_angle_properties_l787_787047


namespace polynomial_evaluation_l787_787596

noncomputable def evaluate_polynomial (x : ℝ) : ℝ :=
  x^3 - 3 * x^2 - 9 * x + 5

theorem polynomial_evaluation (x : ℝ) (h_pos : x > 0) (h_eq : x^2 - 3 * x - 9 = 0) :
  evaluate_polynomial x = 5 :=
by
  sorry

end polynomial_evaluation_l787_787596


namespace units_digit_of_x4_plus_x_inv4_l787_787690

theorem units_digit_of_x4_plus_x_inv4 (x : ℝ) (h : x^2 - 13 * x + 1 = 0) :
  (x^4 + x^(-4)) % 10 = 7 :=
sorry

end units_digit_of_x4_plus_x_inv4_l787_787690


namespace arith_seq_sum_first_110_l787_787995

variable {α : Type*} [OrderedRing α]

theorem arith_seq_sum_first_110 (a₁ d : α) :
  (10 * a₁ + 45 * d = 100) →
  (100 * a₁ + 4950 * d = 10) →
  (110 * a₁ + 5995 * d = -110) :=
by
  intros h1 h2
  sorry

end arith_seq_sum_first_110_l787_787995


namespace animal_population_l787_787316

theorem animal_population (p: ℝ) (initial: ℕ) (surviving: ℝ) (months: ℕ):
  (p = 1 / 10) -> (initial = 500) -> (surviving = 364.5) ->
  months = Int.floor (Real.log 0.729 / Real.log (9 / 10)) -> months = 3 :=
begin
  intros hp hi hs hm,
  rw [hp, hi, hs] at hm,
  -- previous steps go into the proof
  sorry
end

end animal_population_l787_787316


namespace probability_diff_geq_3_l787_787462

open Finset
open BigOperators

-- Define the set of numbers
def number_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the condition for pairs with positive difference ≥ 3
def valid_pair (a b : ℕ) : Prop := a ≠ b ∧ abs (a - b) ≥ 3

-- Calculate the probability as a fraction
theorem probability_diff_geq_3 : 
  let total_pairs := (number_set.product number_set).filter (λ ab, ab.1 < ab.2) in
  let valid_pairs := total_pairs.filter (λ ab, valid_pair ab.1 ab.2) in
  valid_pairs.card / total_pairs.card = 7 / 12 :=
by
  sorry

end probability_diff_geq_3_l787_787462


namespace sam_speed_is_3_mph_l787_787592

-- Eugene's speed definition
def eugene_speed : ℝ := 5

-- Carlos's speed definition
def carlos_speed : ℝ := (4/5) * eugene_speed

-- Sam's speed definition
def sam_speed : ℝ := (3/4) * carlos_speed

-- The statement to prove
theorem sam_speed_is_3_mph : sam_speed = 3 := by
  -- Skipping the proof details
  sorry

end sam_speed_is_3_mph_l787_787592


namespace inequality_am_gm_l787_787750

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / c + c / b) ≥ (4 * a / (a + b)) ∧ (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
by
  -- Proof steps
  sorry

end inequality_am_gm_l787_787750


namespace purple_tetrahedron_volume_l787_787160

theorem purple_tetrahedron_volume (side_length : ℝ) (purple_vertices : List (ℝ × ℝ × ℝ)) 
  (h_cube : ∀ v ∈ purple_vertices, v ∈ cube_vertices) (h_colors : alternately_colored cube_vertices) 
  (h_side : side_length = 8) : 
  ∃ V : ℝ, V = 170.67 :=
by
  let cube_volume := side_length ^ 3
  let base_area := 0.5 * side_length * side_length
  let height := side_length
  let smaller_tetra_volume := (1 / 3) * base_area * height
  let total_smaller_tetra_volume := 4 * smaller_tetra_volume
  let purple_tetrahedron_volume := cube_volume - total_smaller_tetra_volume
  use purple_tetrahedron_volume
  rw h_side
  norm_num
  sorry

end purple_tetrahedron_volume_l787_787160


namespace min_sum_of_squares_l787_787021

theorem min_sum_of_squares 
  (x_1 x_2 x_3 : ℝ)
  (h1: x_1 + 3 * x_2 + 4 * x_3 = 72)
  (h2: x_1 = 3 * x_2)
  (h3: 0 < x_1)
  (h4: 0 < x_2)
  (h5: 0 < x_3) : 
  x_1^2 + x_2^2 + x_3^2 = 347.04 := 
sorry

end min_sum_of_squares_l787_787021


namespace smallest_b_value_l787_787395

noncomputable def smallest_b (a b : ℝ) : ℝ :=
if a > 2 ∧ 2 < a ∧ a < b 
   ∧ (2 + a ≤ b) 
   ∧ ((1 / a) + (1 / b) ≤ 1 / 2) 
then b else 0

theorem smallest_b_value : ∀ (a b : ℝ), 
  (2 < a) → (a < b) → (2 + a ≤ b) → 
  ((1 / a) + (1 / b) ≤ 1 / 2) → 
  b = 3 + Real.sqrt 5 := sorry

end smallest_b_value_l787_787395


namespace mean_and_variance_of_binomial_distribution_probability_cumulative_score_l787_787726

-- Part (1)
theorem mean_and_variance_of_binomial_distribution :
  let X : ℕ → Probability := binomial 5 (2 / 3)
  mean X = 10 / 3 ∧ variance X = 10 / 9 :=
sorry

-- Part (2)
theorem probability_cumulative_score (n : ℕ) :
  let P : ℕ → ℚ := λ n, (3 / 5 - (4 / 15) * (- (2 / 3))^(n - 1))
  P n = (3 / 5 - (4 / 15) * (- (2 / 3))^(n - 1) :=
sorry

end mean_and_variance_of_binomial_distribution_probability_cumulative_score_l787_787726


namespace abs_sum_fraction_le_sum_abs_fraction_l787_787739

variable (a b : ℝ)

theorem abs_sum_fraction_le_sum_abs_fraction (a b : ℝ) :
  (|a + b| / (1 + |a + b|)) ≤ (|a| / (1 + |a|)) + (|b| / (1 + |b|)) :=
sorry

end abs_sum_fraction_le_sum_abs_fraction_l787_787739


namespace arithmetic_sequence_sum_cubes_l787_787438

theorem arithmetic_sequence_sum_cubes (y : ℤ) (n : ℕ) (h_n : n > 6)
  (h_sum : (∑ i in Finset.range (n + 1), (y + 4 * i)^3) = -5832) : n = 11 := 
sorry

end arithmetic_sequence_sum_cubes_l787_787438


namespace square_line_intersection_l787_787517

theorem square_line_intersection (A B C D E F: Type) [square : is_square A B C D]
  (h₁ : line_through A ∩ line CD = E)
  (h₂ : line_through A ∩ line BC = F) :
  (1 / (AE^2) + 1 / (AF^2) = 1 / (AB^2)) :=
sorry

end square_line_intersection_l787_787517


namespace find_fourth_vertex_l787_787839

open Complex

theorem find_fourth_vertex (A B C: ℂ) (hA: A = 2 + 3 * Complex.I) 
                            (hB: B = -3 + 2 * Complex.I) 
                            (hC: C = -2 - 3 * Complex.I) : 
                            ∃ D : ℂ, D = 2.5 + 0.5 * Complex.I :=
by 
  sorry

end find_fourth_vertex_l787_787839


namespace Jaylen_total_vegetables_l787_787345

def Jaylen_vegetables (J_bell_peppers J_green_beans J_carrots J_cucumbers : Nat) : Nat :=
  J_bell_peppers + J_green_beans + J_carrots + J_cucumbers

theorem Jaylen_total_vegetables :
  let Kristin_bell_peppers := 2
  let Kristin_green_beans := 20
  let Jaylen_bell_peppers := 2 * Kristin_bell_peppers
  let Jaylen_green_beans := (Kristin_green_beans / 2) - 3
  let Jaylen_carrots := 5
  let Jaylen_cucumbers := 2
  Jaylen_vegetables Jaylen_bell_peppers Jaylen_green_beans Jaylen_carrots Jaylen_cucumbers = 18 := 
by
  sorry

end Jaylen_total_vegetables_l787_787345


namespace plane_parallel_if_perpendicular_to_same_line_l787_787266

variables (Line Plane : Type)
variable  (parallel : Line → Plane → Prop)
variable  (perpendicular : Line → Plane → Prop)
variables (m n : Line) (α β γ : Plane)

theorem plane_parallel_if_perpendicular_to_same_line 
  (hmα : perpendicular m α) 
  (hmβ : perpendicular m β) 
  : α = β ∨ (∀ p q : Plane, p = q ∨ parallel p q) := 
sorry

end plane_parallel_if_perpendicular_to_same_line_l787_787266


namespace Sin_Sub_Cos_Negative_Interval_l787_787603

theorem Sin_Sub_Cos_Negative_Interval (x : ℝ) (h₀ : 0 < x) (h₁ : x < 2 * Real.pi) : 
  (sin x - cos x < 0) ↔ (0 < x ∧ x < (Real.pi / 4)) ∨ (5 * Real.pi / 4 < x ∧ x < 2 * Real.pi) :=
by
  sorry

end Sin_Sub_Cos_Negative_Interval_l787_787603


namespace total_pets_combined_l787_787804

def teddy_dogs : ℕ := 7
def teddy_cats : ℕ := 8
def ben_dogs : ℕ := teddy_dogs + 9
def dave_cats : ℕ := teddy_cats + 13
def dave_dogs : ℕ := teddy_dogs - 5

def teddy_pets : ℕ := teddy_dogs + teddy_cats
def ben_pets : ℕ := ben_dogs
def dave_pets : ℕ := dave_cats + dave_dogs

def total_pets : ℕ := teddy_pets + ben_pets + dave_pets

theorem total_pets_combined : total_pets = 54 :=
by
  -- proof goes here
  sorry

end total_pets_combined_l787_787804


namespace bisection_method_accuracy_l787_787636

noncomputable def bisection_method_halvings (f : ℝ → ℝ) (a b : ℝ) (ε : ℝ) : ℕ :=
  if h : continuous_on f (set.Icc a b) ∧ f a * f b < 0 ∧ 0 < ε then
    nat.ceil (real.logb 2 ((b - a) / ε))
  else
    0

theorem bisection_method_accuracy 
  {f : ℝ → ℝ} {a b : ℝ} (h_cont : continuous_on f (set.Icc a b)) (h_sign : f a * f b < 0) :
  bisection_method_halvings f 0 1 0.1 = 4 :=
by
  have h_ε : 0 < 0.1 := by norm_num
  have h_bounds : bisection_method_halvings f 0 1 0.1 = 
    nat.ceil (real.logb 2 ((1 - 0) / 0.1)) := rfl
  rw [h_bounds]
  have log_val : real.logb 2 10 ≈ 3.321928094887362 ≈ 4 :=
    by norm_num
  exactly log_val
  sorry

end bisection_method_accuracy_l787_787636


namespace combined_total_pets_l787_787802

structure People := 
  (dogs : ℕ)
  (cats : ℕ)

def Teddy : People := {dogs := 7, cats := 8}
def Ben : People := {dogs := 7 + 9, cats := 0}
def Dave : People := {dogs := 7 - 5, cats := 8 + 13}

def total_pets (p : People) : ℕ := p.dogs + p.cats

theorem combined_total_pets : 
  total_pets Teddy + total_pets Ben + total_pets Dave = 54 := by
  sorry

end combined_total_pets_l787_787802


namespace arithmetic_expression_evaluation_l787_787933

theorem arithmetic_expression_evaluation :
  (1 / 6 * -6 / (-1 / 6) * 6) = 36 :=
by {
  sorry
}

end arithmetic_expression_evaluation_l787_787933


namespace f_neg_l787_787304

-- Given conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = f(x)

def f_pos (x : ℝ) : ℝ :=
  x * (1 - x) 

-- Mathematical problem statement
theorem f_neg (f : ℝ → ℝ)
  (H_even : is_even_function f)
  (H_f_pos : ∀ x, 0 ≤ x → f(x) = f_pos x)
  (x : ℝ) (H_x_neg : x ≤ 0) :
  f(x) = -x * (1 + x) :=
sorry

end f_neg_l787_787304


namespace albums_total_l787_787033

noncomputable def miriam_albums (katrina_albums : ℕ) := 5 * katrina_albums
noncomputable def katrina_albums (bridget_albums : ℕ) := 6 * bridget_albums
noncomputable def bridget_albums (adele_albums : ℕ) := adele_albums - 15
noncomputable def total_albums (adele_albums : ℕ) (bridget_albums : ℕ) (katrina_albums : ℕ) (miriam_albums : ℕ) := 
  adele_albums + bridget_albums + katrina_albums + miriam_albums

theorem albums_total (adele_has_30 : adele_albums = 30) : 
  total_albums 30 (bridget_albums 30) (katrina_albums (bridget_albums 30)) (miriam_albums (katrina_albums (bridget_albums 30))) = 585 := 
by 
  -- In Lean, replace the term 'sorry' below with the necessary steps to finish the proof
  sorry

end albums_total_l787_787033


namespace find_people_who_own_only_cats_l787_787109

variable (C : ℕ)

theorem find_people_who_own_only_cats
  (ownsOnlyDogs : ℕ)
  (ownsCatsAndDogs : ℕ)
  (ownsCatsDogsSnakes : ℕ)
  (totalPetOwners : ℕ)
  (h1 : ownsOnlyDogs = 15)
  (h2 : ownsCatsAndDogs = 5)
  (h3 : ownsCatsDogsSnakes = 3)
  (h4 : totalPetOwners = 59) :
  C = 36 :=
by
  sorry

end find_people_who_own_only_cats_l787_787109


namespace school_pays_correct_amount_l787_787070

theorem school_pays_correct_amount :
  let price_A : ℕ := 100
  let price_B : ℕ := 120
  let price_C : ℕ := 140
  let count_A_kindergarten : ℕ := 2
  let count_B_elem : ℕ := 2 * count_A_kindergarten
  let count_C_high : ℕ := 3 * count_A_kindergarten
  let total_cost_before_discount := 
    (count_A_kindergarten * price_A) + 
    (count_B_elem * price_B) + 
    (count_C_high * price_C)
  let discount_rate := if total_cost_before_discount > 2000 then 0.15
                       else if total_cost_before_discount > 1000 then 0.10
                       else if total_cost_before_discount > 500 then 0.05
                       else 0.0
  let discount := discount_rate * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount
  total_cost_after_discount = 1368 :=
by
  sorry

end school_pays_correct_amount_l787_787070


namespace find_m_of_power_fn_and_increasing_l787_787815

theorem find_m_of_power_fn_and_increasing (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 5) * x^(m - 1) > 0) →
  m^2 - m - 5 = 1 →
  1 < m →
  m = 3 :=
sorry

end find_m_of_power_fn_and_increasing_l787_787815


namespace quadratic_real_roots_iff_range_of_a_l787_787869

theorem quadratic_real_roots_iff_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a = 0) ↔ a ≤ 1 :=
by
  sorry

end quadratic_real_roots_iff_range_of_a_l787_787869


namespace mustard_found_at_third_table_l787_787193

variable (a b T : ℝ)
def found_mustard_at_first_table := (a = 0.25)
def found_mustard_at_second_table := (b = 0.25)
def total_mustard_found := (T = 0.88)

theorem mustard_found_at_third_table
  (h1 : found_mustard_at_first_table a)
  (h2 : found_mustard_at_second_table b)
  (h3 : total_mustard_found T) :
  T - (a + b) = 0.38 := by
  sorry

end mustard_found_at_third_table_l787_787193


namespace integer_triangle_answer_l787_787711

def integer_triangle_condition :=
∀ a r : ℕ, (1 ≤ a ∧ a ≤ 19) → 
(a = 12) → (r = 3) → 
(r = 96 / (20 + a))

theorem integer_triangle_answer : 
  integer_triangle_condition := 
by
  sorry

end integer_triangle_answer_l787_787711


namespace odot_computation_l787_787092

noncomputable def op (a b : ℚ) : ℚ := 
  (a + b) / (1 + a * b)

theorem odot_computation : op 2 (op 3 (op 4 5)) = 7 / 8 := 
  by 
  sorry

end odot_computation_l787_787092


namespace find_m_l787_787163

def g : ℤ → ℤ :=
  λ n, if n % 2 = 1 then n + 5 else n / 2

theorem find_m (m : ℤ) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 15) : m = 55 :=
  sorry

end find_m_l787_787163


namespace power_mod_2000_l787_787468

theorem power_mod_2000 (n : ℕ) : 7 ^ n ≡ 1359 [MOD 2000] :=
by
  have h1 : 7^13 ≡ 707 [MOD 2000], from sorry,
  have h2 : 7^8 ≡ 801 [MOD 2000], from sorry,
  have h3 : 7^2023 = 7^(13 * 155 + 8), by sorry,
  have h4 : 7^(13 * 155 + 8) ≡ (707^155 * 801) [MOD 2000], by sorry,
  exact sorry

end power_mod_2000_l787_787468


namespace batsman_average_after_17th_inning_l787_787153

noncomputable def batsman_average_17th_inning 
  (A : ℝ)  -- average score before the 17th inning
  (new_runs : ℝ)  -- runs scored in the 17th inning
  (average_increase : ℝ)  -- increase in average
  (total_innings : ℕ)  -- total innings before the new average
  
  -- Condition:
  (h1 : 17 * (A + average_increase) = 16 * A + new_runs) : 
  (new_average : ℝ) := 
  A + average_increase
  
theorem batsman_average_after_17th_inning 
  (A : ℝ)
  (new_runs : ℝ = 88)
  (average_increase : ℝ = 3)
  (total_innings : ℕ = 16)
  (h1 : 17 * (A + average_increase) = 16 * A + new_runs) :
  batsman_average_17th_inning A new_runs average_increase total_innings h1 = 40 :=
sorry

end batsman_average_after_17th_inning_l787_787153


namespace days_to_exceed_50000_l787_787024

noncomputable def A : ℕ → ℝ := sorry
def k : ℝ := sorry
def t := 10
def lg2 := 0.3
def lg (x : ℝ) : ℝ := sorry -- assuming the definition of logarithm base 10

axiom user_count : ∀ t, A t = 500 * Real.exp (k * t)
axiom condition_after_10_days : A 10 = 2000
axiom log_property : lg 2 = lg2

theorem days_to_exceed_50000 : ∃ t : ℕ, t ≥ 34 ∧ 500 * Real.exp (k * t) > 50000 := 
sorry

end days_to_exceed_50000_l787_787024


namespace probability_art_second_course_l787_787171

variable (totalCourses : ℕ) (peCourses : ℕ) (artCourses : ℕ) (chooseCourses : ℕ)
variable (P_A : ℚ) (P_B_given_A : ℚ)

-- Define the conditions
def school : Prop := totalCourses = 6 ∧ peCourses = 4 ∧ artCourses = 2 ∧ chooseCourses = 2

-- Define the probability of choosing a physical education course first
def P_A_def : Prop := P_A = (4 : ℚ) / 6

-- Define the probability of choosing an art course second given a physical education course was chosen first
def P_B_given_A_def : Prop := P_B_given_A = (2 : ℚ) / 5

-- The final theorem
theorem probability_art_second_course
    (h_school : school totalCourses peCourses artCourses chooseCourses)
    (h_P_A : P_A_def peCourses totalCourses P_A)
    (h_P_B_given_A_def : P_B_given_A_def artCourses (totalCourses - 1) P_B_given_A)
    : P_B_given_A = (2 : ℚ) / 5 := sorry

end probability_art_second_course_l787_787171


namespace meaningful_fraction_l787_787097

theorem meaningful_fraction (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) :=
by {
  sorry -- Proof goes here
}

end meaningful_fraction_l787_787097


namespace range_of_a_l787_787277

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 + (2*a^2 + 2)*x - a^2 + 4*a - 7) / 
             (x^2 + (a^2 + 4*a - 5)*x - a^2 + 4*a - 7) < 0) →
  (∃ I : set (ℝ×ℝ), (∀ i ∈ I, ∃ xa xb, i = (xa, xb) ∧ xa < xb) ∧
                     (∀ i ∈ I, is_open (set.Ioo (i.1) (i.2))) ∧
                     (real.sum (λ i, (i.2 - i.1)) I ≥ 4)) →
  (a ≤ 1 ∨ a ≥ 3) :=
sorry

end range_of_a_l787_787277


namespace sum_of_digits_of_M_l787_787939

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_M :
  let M := Nat.floor (Real.sqrt (36^50 * 50^36)) in
  sum_of_digits M = 36 :=
by
  let M := Nat.floor (Real.sqrt (36^50 * 50^36))
  have : M = 6^50 * 10^36 := sorry
  sorry

end sum_of_digits_of_M_l787_787939


namespace equilateral_triangle_inscribed_l787_787186

theorem equilateral_triangle_inscribed
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ) 
  (h1 : A = (0, sqrt 3 / 3))
  (h2 : (A.1)² + 3 * (A.2)² = 3)
  (h3 : (B.1)² + 3 * (B.2)² = 3)
  (h4 : (C.1)² + 3 * (C.2)² = 3)
  (h5 : B.1 = -C.1)
  (h6 : B.2 = C.2) :
  let side_length_squared := (B.1 - C.1)² + (B.2 - C.2)²
  in side_length_squared = 39 / 25 :=
  sorry

end equilateral_triangle_inscribed_l787_787186


namespace y_intercept_of_line_is_neg1_l787_787817

-- Define the line equation as a predicate.
def line_eq (x y : ℝ) : Prop := 2 * x - y - 1 = 0

-- The theorem statement to be proved.
theorem y_intercept_of_line_is_neg1 : ∃ y : ℝ, line_eq 0 y ∧ y = -1 := 
by
  use -1
  simp [line_eq]
  sorry

end y_intercept_of_line_is_neg1_l787_787817


namespace living_room_area_l787_787871

-- Define the dimensions of the carpet
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_area : ℝ := carpet_length * carpet_width -- 36

-- Define the percent coverage of the carpet over the living room floor
def percent_coverage : ℝ := 0.2

-- Define the total area of the living room floor
def total_area : ℝ := carpet_area / percent_coverage -- 180

-- Prove that the total area of the living room floor is 180 square feet
theorem living_room_area : total_area = 180 :=
by
  have h1 : carpet_area = 36 := by sorry
  have h2 : percent_coverage = 0.2 := by sorry
  have h3 : total_area = carpet_area / percent_coverage := by sorry
  exact sorry

end living_room_area_l787_787871


namespace area_WXYZ_is_5200_l787_787328

noncomputable def area_of_rectangle_WXYZ : ℝ :=
  let W := (-4, 3) : ℝ × ℝ
  let X := (996, 203) : ℝ × ℝ
  let Y := (-2, -7) : ℝ × ℝ
  let WX := Real.sqrt ((996 - (-4)) ^ 2 + (203 - 3) ^ 2)
  let WY := Real.sqrt ((-2 - (-4)) ^ 2 + (-7 - 3) ^ 2)
  WX * WY

theorem area_WXYZ_is_5200 : area_of_rectangle_WXYZ = 5200 :=
by
  let W := (-4, 3) : ℝ × ℝ
  let X := (996, 203) : ℝ × ℝ
  let Y := (-2, -7) : ℝ × ℝ
  let WX := Real.sqrt ((996 - (-4)) ^ 2 + (203 - 3) ^ 2)
  let WY := Real.sqrt ((-2 - (-4)) ^ 2 + (-7 - 3) ^ 2)
  have h_WX : WX = 100 * Real.sqrt 104 := by sorry
  have h_WY : WY = 2 * Real.sqrt 26 := by sorry
  have h_area := h_WX.symm ▸ h_WY.symm ▸ rfl
  exact h_area.symm

end area_WXYZ_is_5200_l787_787328


namespace v_2010_is_0_l787_787419

-- Define the function g
def g (x : Nat) : Nat :=
  if x = 0 then 2
  else if x = 1 then 5
  else if x = 2 then 3
  else if x = 3 then 0
  else if x = 4 then 1
  else if x = 5 then 4
  else 0  -- assuming x is always within [0,5]

-- Define the sequence v recursively
def v : ℕ → ℕ
| 0       := 0
| (n + 1) := g (v n)

-- The theorem we will prove
theorem v_2010_is_0 : v 2010 = 0 :=
sorry

end v_2010_is_0_l787_787419


namespace radius_of_larger_circle_l787_787242

open Real

theorem radius_of_larger_circle (r : ℝ) (R : ℝ) (h1 : r = 2)
  (h2 : ∀ (C1 C2 : ℝ × ℝ), (sq_dist C1 C2 = (2 * r) ^ 2))
  (h3 : R = r * (1 + 1 * sqrt 2)) : R = 2 * (1 + sqrt 2) := 
by
  rw [h3, h1]
  ring_nf
  rw [sqrt_two_mul_sqrt_two]
  simp
  sorry

end radius_of_larger_circle_l787_787242


namespace tan_theta_at_maximum_l787_787962

theorem tan_theta_at_maximum
  (f : ℝ → ℝ)
  (h : ∀ x, f x = 3 * sin (x + π / 6))
  (h_max : ∀ θ, f θ = f (θ - 2 * π) → θ = π / 3 + 2 * k * π) :
  tan (π / 3) = sqrt 3 :=
by
  sorry

end tan_theta_at_maximum_l787_787962


namespace calculate_total_cost_l787_787129

def cost_of_parallel_sides (l1 l2 ppf : ℕ) : ℕ :=
  l1 * ppf + l2 * ppf

def cost_of_non_parallel_sides (l1 l2 ppf : ℕ) : ℕ :=
  l1 * ppf + l2 * ppf

def total_cost (p_l1 p_l2 np_l1 np_l2 ppf np_pf : ℕ) : ℕ :=
  cost_of_parallel_sides p_l1 p_l2 ppf + cost_of_non_parallel_sides np_l1 np_l2 np_pf

theorem calculate_total_cost :
  total_cost 25 37 20 24 48 60 = 5616 :=
by
  -- Assuming the conditions are correctly applied, the statement aims to validate that the calculated
  -- sum of the costs for the specified fence sides equal Rs 5616.
  sorry

end calculate_total_cost_l787_787129


namespace maximum_cardinality_l787_787530

def distinct_positive_integers_set (T : Set ℕ) : Prop :=
  ∀ x ∈ T, 0 < x ∧ ∀ y ∈ T, x ≠ y → x ≠ y

def arithmetic_mean_integer (T : Set ℕ) (x : ℕ) : Prop :=
  ((∑ y in T \ {x}, y) / (|T| - 1)) ∈ ℤ

def problem_conditions (T : Set ℕ) : Prop :=
  1 ∈ T ∧ 3003 ∈ T ∧ distinct_positive_integers_set T ∧
  ∀ x ∈ T, arithmetic_mean_integer T x

theorem maximum_cardinality (T : Set ℕ) (h : problem_conditions T) : |T| ≤ 43 := 
  sorry

end maximum_cardinality_l787_787530


namespace exists_n_ge_3_polynomials_l787_787587

theorem exists_n_ge_3_polynomials :
  ∃ (P Q T : Polynomial ℤ) (n : ℕ), n ≥ 3 ∧ 
  (P.coeffs.all (λ c, 0 < c)) ∧ 
  (Q.coeffs.all (λ c, 0 < c)) ∧
  (T.coeffs.all (λ c, 0 < c)) ∧
  (T = (Polynomial.x ^ 2 - 3 * Polynomial.x + 3) * P) ∧
  (T = (Polynomial.C (1 / 20:ℚ) * Polynomial.x ^ 2 - Polynomial.C (1 / 15:ℚ) * Polynomial.x + Polynomial.C (1 / 12:ℚ)) * Q) :=
sorry

end exists_n_ge_3_polynomials_l787_787587


namespace correspondence_is_function_l787_787010

variables {A B : Type} (hA : ∃ (a : A), true) (hB : ∃ (b : B), true)
variable (f : A → B)

theorem correspondence_is_function :
  (∀ x : A, ∃! y : B, f x = y) → (∃ (f : A → B), ∀ x : A, ∃! y : B, f x = y) :=
by
  intro h
  exists f
  exact h
  sorry

end correspondence_is_function_l787_787010


namespace exists_positive_integers_abcd_l787_787625

theorem exists_positive_integers_abcd (m : ℤ) : ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ (a * b - c * d = m) := by
  sorry

end exists_positive_integers_abcd_l787_787625


namespace chocolates_initial_count_l787_787946

theorem chocolates_initial_count (remaining_chocolates: ℕ) 
    (daily_percentage: ℝ) (days: ℕ) 
    (final_chocolates: ℝ) 
    (remaining_fraction_proof: remaining_fraction = 0.7) 
    (days_proof: days = 3) 
    (final_chocolates_proof: final_chocolates = 28): 
    (remaining_fraction^days * (initial_chocolates:ℝ) = final_chocolates) → 
    (initial_chocolates = 82) := 
by 
  sorry

end chocolates_initial_count_l787_787946


namespace samantha_owes_more_l787_787053

noncomputable def final_amount (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

def compounding_difference (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  final_amount P r 12 t - final_amount P r 2 t

theorem samantha_owes_more {P : ℝ} {r : ℝ} {t : ℝ} 
  (hP : P = 8000) (hr : r = 0.12) (ht : t = 3) :
  compounding_difference P r t = 58.08 :=
by
  rw [hP, hr, ht]
  sorry

end samantha_owes_more_l787_787053


namespace max_minibuses_l787_787717

structure CityMap (α : Type _) :=
(num_intersections : Nat)
(num_streets : Nat)
(is_graph : α)
(n_minibuses : Nat)
(closed_routes : α → Prop) -- Represents non-intersecting closed routes
(unique_street_usage : α → Prop) -- Only one minibus on each street

theorem max_minibuses (G : Type _) [CityMap G] :
  (G.num_streets = 35) →
  (∀ route ∈ G.closed_routes G.is_graph, G.unique_street_usage route) →
  ∃ N : Nat, G.n_minibuses = N ∧ N ≤ 35 ∧ (∀ (r : G), G.n_minibuses = 35 → G.closed_routes r) :=
by {
  sorry
}

end max_minibuses_l787_787717


namespace bodhi_yacht_animals_l787_787766

def total_animals (cows foxes zebras sheep : ℕ) : ℕ :=
  cows + foxes + zebras + sheep

theorem bodhi_yacht_animals :
  ∀ (cows foxes sheep : ℕ), foxes = 15 → cows = 20 → sheep = 20 → total_animals cows foxes (3 * foxes) sheep = 100 :=
by
  intros cows foxes sheep h1 h2 h3
  rw [h1, h2, h3]
  show total_animals 20 15 (3 * 15) 20 = 100
  sorry

end bodhi_yacht_animals_l787_787766


namespace harmonic_geometric_arithmetic_means_l787_787430

theorem harmonic_geometric_arithmetic_means {n : ℕ} (n_pos : n > 0) 
  (α : Fin n → ℝ) (α_pos : ∀ i, 0 < α i) :
  let h := n / ∑ i, 1 / α i
      g := (∏ i, α i) ^ (1/n : ℝ)
      a := ∑ i, α i / n
  in h ≤ g ∧ g ≤ a ∧ (h = g ∧ g = a ↔ ∀ i j, α i = α j) := by
  sorry

end harmonic_geometric_arithmetic_means_l787_787430


namespace min_abs_diff_l787_787683

theorem min_abs_diff (a b : ℕ) (h_cond : a > 0 ∧ b > 0 ∧ ab - 4 * a + 6 * b = 528) : |a - b| = 9 :=
sorry

end min_abs_diff_l787_787683


namespace Vasya_and_Petya_no_mistake_exists_l787_787848

def is_prime (n : ℕ) : Prop := ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem Vasya_and_Petya_no_mistake_exists :
  ∃ x : ℝ, (∃ p : ℕ, is_prime p ∧ 10 * x = p) ∧ 
           (∃ q : ℕ, is_prime q ∧ 15 * x = q) :=
sorry

end Vasya_and_Petya_no_mistake_exists_l787_787848


namespace coeff_x2_expansion_l787_787641

def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

theorem coeff_x2_expansion (n : ℕ) (h : 9 * binomial_coefficient n 2 = 54) : n = 4 :=
sorry

end coeff_x2_expansion_l787_787641


namespace mul_assoc_l787_787091

variables {R : Type} [NonUnitalNonAssocSemiring R] (mul : R → R → R)

-- Conditions
axiom zero_mul (a : R) : mul 0 a = a
axiom mul_assoc_comm (a b c : R) : mul a (mul b c) = mul c (mul b a)

-- Theorem to prove
theorem mul_assoc (a b c : R) : mul a (mul b c) = mul (mul a b) c :=
sorry

end mul_assoc_l787_787091


namespace fifth_number_in_21st_row_l787_787554

theorem fifth_number_in_21st_row : 
  let nth_odd_number (n : ℕ) := 2 * n - 1 
  let sum_first_n_rows (n : ℕ) := n * (n + (n - 1))
  nth_odd_number 405 = 809 := 
by
  sorry

end fifth_number_in_21st_row_l787_787554


namespace find_regular_polygon_interior_angles_l787_787227

theorem find_regular_polygon_interior_angles (n : ℕ) (θ : ℕ) :
  n ∣ 360 → n ≥ 3 → θ = 180 - 360 / n → θ ∈ {60, 90, 108, 120, 135, 140, 144, 150, 156, 160, 162, 165, 168, 170, 171, 172, 174, 175, 176, 177, 178, 179} := by
  sorry

end find_regular_polygon_interior_angles_l787_787227


namespace last_letter_151st_permutation_eq_O_l787_787410

theorem last_letter_151st_permutation_eq_O : 
  ∀ (s : List Char), 
    s = ['J', 'O', 'K', 'I', 'N', 'G'] → 
    (perm : Nat → List Char) → 
    (∀ n, perm n = getPermutationLexicographic s n) → 
    (151 ≤ factorial 6) → 
    (perm 151).last = 'O' :=
by
  intros s h
  sorry

end last_letter_151st_permutation_eq_O_l787_787410


namespace simplify_cos_diff_l787_787781

theorem simplify_cos_diff : 
  (cos (45 * Real.pi / 180) - cos (135 * Real.pi / 180)) = Real.sqrt 2 :=
by
  -- Provide the known values of cos 45° and cos 135°
  have h45 : cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := sorry
  have h135 : cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 := sorry
  -- Use these values to prove the main statement
  rw [h45, h135]
  calc 
    (Real.sqrt 2 / 2) - (- Real.sqrt 2 / 2)
        = (Real.sqrt 2 / 2) + (Real.sqrt 2 / 2) : by sorry
    ... = Real.sqrt 2 : by sorry

end simplify_cos_diff_l787_787781


namespace cost_of_souvenirs_in_usd_l787_787928

def spent_on_kc_and_br := 347
def spent_on_tshirts := spent_on_kc_and_br - 146
def discount_on_tshirts := 0.10 * spent_on_tshirts
def actual_paid_tshirts := spent_on_tshirts - discount_on_tshirts
def sales_tax_kc := 0.12 * spent_on_kc_and_br
def import_tax_br := 0.08 * spent_on_kc_and_br
def total_with_taxes := spent_on_kc_and_br + sales_tax_kc + import_tax_br
def total_spent_local_currency := actual_paid_tshirts + total_with_taxes
def total_spent_usd := total_spent_local_currency * 0.75

theorem cost_of_souvenirs_in_usd : total_spent_usd = 447.975 := by
  sorry

end cost_of_souvenirs_in_usd_l787_787928


namespace handshakes_count_l787_787192

-- Define the parameters
def teams : ℕ := 3
def players_per_team : ℕ := 7
def referees : ℕ := 3

-- Calculate handshakes among team members
def handshakes_among_teams :=
  let unique_handshakes_per_team := players_per_team * 2 * players_per_team / 2
  unique_handshakes_per_team * teams

-- Calculate handshakes between players and referees
def players_shake_hands_with_referees :=
  teams * players_per_team * referees

-- Calculate total handshakes
def total_handshakes :=
  handshakes_among_teams + players_shake_hands_with_referees

-- Proof statement
theorem handshakes_count : total_handshakes = 210 := by
  sorry

end handshakes_count_l787_787192


namespace yellow_marbles_problem_l787_787444

variable (Y B R : ℕ)

theorem yellow_marbles_problem
  (h1 : Y + B + R = 19)
  (h2 : B = (3 * R) / 4)
  (h3 : R = Y + 3) :
  Y = 5 :=
by
  sorry

end yellow_marbles_problem_l787_787444


namespace range_of_a_l787_787653

noncomputable def f (x a : ℝ) : ℝ := Real.logBase 0.5 (x^2 - a * x + 4 * a)

theorem range_of_a (a : ℝ) :
  (-2 < a ∧ a ≤ 4) ↔ Monotone (λ x, f x a) :=
  sorry

end range_of_a_l787_787653


namespace correct_options_l787_787278

theorem correct_options
  (m : ℝ) 
  (l : ℝ → ℝ) 
  (A : Prop)
  (C : Prop) : 
  (A ↔ ∀ (m = -3), l = (λ x, -(x + 5))) 
  ∧ (C ↔ ∀ (x : ℝ), l = (λ x, x - 1)) :=
by
  sorry

end correct_options_l787_787278


namespace farm_problem_l787_787769

theorem farm_problem
    (initial_cows : ℕ := 12)
    (initial_pigs : ℕ := 34)
    (remaining_animals : ℕ := 30)
    (C : ℕ)
    (P : ℕ)
    (h1 : P = 3 * C)
    (h2 : initial_cows - C + (initial_pigs - P) = remaining_animals) :
    C = 4 :=
by
  sorry

end farm_problem_l787_787769


namespace ratio_of_interest_l787_787435

noncomputable def principal1 : ℝ := 2800
noncomputable def rate1 : ℝ := 5
noncomputable def time1 : ℝ := 3
noncomputable def principal2 : ℝ := 4000
noncomputable def rate2 : ℝ := 10
noncomputable def time2 : ℝ := 2

noncomputable def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100
noncomputable def compound_interest (P R T : ℝ) : ℝ := P * ((1 + R / 100)^T - 1)

theorem ratio_of_interest :
  let SI := simple_interest principal1 rate1 time1 in
  let CI := compound_interest principal2 rate2 time2 in
  SI / CI = 1 / 2 :=
by
  sorry

end ratio_of_interest_l787_787435


namespace intersect_or_parallel_l787_787723

variables {A B C A1 B1 C1 A2 B2 C2 : Type*}

-- Statements representing the conditions
def is_altitude (A B C X : Type*) : Prop := sorry -- Placeholder for the definition of an altitude
def is_diameter (A1 A2 : Type*) : Prop := sorry -- Placeholder for the definition of a diameter of the nine-point circle

-- The formal math statement in Lean 4
theorem intersect_or_parallel 
  (h1 : is_altitude A B C A1)
  (h2 : is_altitude B C A B1)
  (h3 : is_altitude C A B C1)
  (h4 : is_diameter A1 A2)
  (h5 : is_diameter B1 B2)
  (h6 : is_diameter C1 C2) :
  ∃ P : Type*, (∃ Q : Type*, ∃ R : Type*, P ≠ Q ∧ Q ≠ R ∧ P ≠ R) ∨ 
  (∀ x : Type*, ∃ y : Type*, ∀ z : Type*, x = y ∨ y = z ∨ x = z) :=
sorry

end intersect_or_parallel_l787_787723


namespace exist_2005_numbers_l787_787222

theorem exist_2005_numbers :
  ∃ (a : ℕ → ℕ), (∀ n, n < 2005 → a n ∈ ℕ) ∧ (∀ i j, i < j → a i ≠ a j) ∧
  (∀ i, i < 2005 → ((Finset.range 2005).erase i).sum (λ j, a j) % a i = 0) :=
sorry

end exist_2005_numbers_l787_787222


namespace intersection_points_find_a_l787_787331

-- Definition of the parametric curve C
def curve_C (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, Real.sin θ)

-- Definition of the parametric line l with parameter t and parameter a
def line_l (t a : ℝ) : ℝ × ℝ := (a + 4 * t, 1 - t)

-- Problem 1
theorem intersection_points (θ t : ℝ) (a : ℝ) (h : a = -1) :
  let C := curve_C θ
  let l := line_l t a
  (C = (3, 0) ∨ C = (-21 / 25, 24 / 25)) ↔ 
  (∃ θ t : ℝ, (curve_C θ = (3, 0) ∧ line_l t (-1) = (3, 0)) ∨
              (curve_C θ = (-21 / 25, 24 / 25) ∧ line_l t (-1) = (-21 / 25, 24 / 25))) := by sorry

-- Problem 2
theorem find_a (θ : ℝ) (d : ℝ) (h_d : d = Real.sqrt 17) :
  let C := curve_C θ
  let distance (P : ℝ × ℝ) (a : ℝ) : ℝ :=
    abs (3 * Real.cos θ + 4 * Real.sin θ - a - 4) / Real.sqrt 17
  (∃ a : ℝ, a = -16 ∨ a = 8) ↔ 
  (∃ θ : ℝ, distance (curve_C θ) (-16) = d ∨ distance (curve_C θ) 8 = d) := by sorry

end intersection_points_find_a_l787_787331


namespace typists_in_initial_group_l787_787691

theorem typists_in_initial_group (letters : ℕ) (minutes : ℕ) (total_typists : ℕ) : (letters = 44) → (minutes = 20) → (total_typists = 30) → 
  (let one_typist_20 = letters * 3 in one_typist_20 = 198 / total_typists) →
  (letters / (198 / total_typists)) = 20 :=
by
  sorry

end typists_in_initial_group_l787_787691


namespace cylinder_lateral_area_cylinder_volume_cylinder_surface_area_cone_volume_l787_787930

-- Problem 1
theorem cylinder_lateral_area (C H : ℝ) (hC : C = 1.8) (hH : H = 1.5) :
  C * H = 2.7 := by sorry 

-- Problem 2
theorem cylinder_volume (D H : ℝ) (hD : D = 3) (hH : H = 8) :
  (3.14 * ((D * 10 / 2) ^ 2) * H) = 5652 :=
by sorry

-- Problem 3
theorem cylinder_surface_area (r h : ℝ) (hr : r = 6) (hh : h = 5) :
    (3.14 * r * 2 * h + 3.14 * r ^ 2 * 2) = 414.48 :=
by sorry

-- Problem 4
theorem cone_volume (B H : ℝ) (hB : B = 18.84) (hH : H = 6) :
  (1 / 3 * B * H) = 37.68 :=
by sorry

end cylinder_lateral_area_cylinder_volume_cylinder_surface_area_cone_volume_l787_787930


namespace birds_on_fence_total_l787_787150

variable (initial_birds : ℕ) (additional_birds : ℕ)

theorem birds_on_fence_total {initial_birds additional_birds : ℕ} (h1 : initial_birds = 4) (h2 : additional_birds = 6) :
    initial_birds + additional_birds = 10 :=
  by
  sorry

end birds_on_fence_total_l787_787150


namespace smallest_number_mod_l787_787233

theorem smallest_number_mod (x : ℕ) :
  (x % 2 = 1) → (x % 3 = 2) → x = 5 :=
by
  sorry

end smallest_number_mod_l787_787233


namespace solve_ode_with_initial_condition_l787_787234

-- Define the function y(x)
def y (x : ℝ) : ℝ := Real.sin x + 1

-- Define the differential equation condition
def diff_eq (x : ℝ) : Prop := deriv y x = Real.cos x

-- Define the initial condition
def initial_condition : Prop := y 0 = 1

-- The statement of the theorem
theorem solve_ode_with_initial_condition : (∀ x, diff_eq x) ∧ initial_condition := by
  sorry

end solve_ode_with_initial_condition_l787_787234


namespace limit_quot_proof_l787_787250

noncomputable def limit_quot (p q : ℝ) (h_p1 : p > 1) (h_p2 : p ≠ 2) (h_q1 : q > 1) (h_q2 : q ≠ 2) : ℝ :=
  let seq := (n : ℕ) → (p^(n+1) - q^n) / (p^(n+2) - 2*q^(n+1))
  Real.lim seq

theorem limit_quot_proof (p q : ℝ) (h_p1 : p > 1) (h_p2 : p ≠ 2) (h_q1 : q > 1) (h_q2 : q ≠ 2) :
  limit_quot p q h_p1 h_p2 h_q1 h_q2 = 1/p ∨ limit_quot p q h_p1 h_p2 h_q1 h_q2 = 1/(2*q) ∨ limit_quot p q h_p1 h_p2 h_q1 h_q2 = (p - 1)/(p^2 - 2*p) := by
  -- Proof goes here
  sorry

end limit_quot_proof_l787_787250


namespace complex_conversion_l787_787206

noncomputable def rectangular_form (z : ℂ) : ℂ := z

theorem complex_conversion :
  rectangular_form (complex.sqrt 2 * complex.exp (13 * real.pi * complex.I / 4)) = 1 + complex.I :=
by
  sorry

end complex_conversion_l787_787206


namespace ratio_of_pq_l787_787753

def is_pure_imaginary (z : Complex) : Prop :=
  z.re = 0

theorem ratio_of_pq (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (H : is_pure_imaginary ((Complex.ofReal 3 - Complex.ofReal 4 * Complex.I) * (Complex.ofReal p + Complex.ofReal q * Complex.I))) :
  p / q = -4 / 3 :=
by
  sorry

end ratio_of_pq_l787_787753


namespace perpendicular_diagonal_pairs_count_is_6_l787_787338

-- Definitions for vertices and diagonals in a regular triangular prism
variables (A B C A₁ B₁ C₁ : Type) [RegularTriangularPrism A B C A₁ B₁ C₁] 

-- The given condition that diagonal A B₁ is perpendicular to diagonal B C₁
axiom AB₁_perp_BC₁ : Perpendicular (diagonal A B₁) (diagonal B C₁)

-- Statement to prove the number of mutually perpendicular diagonal pairs including AB₁ ⊥ BC₁
theorem perpendicular_diagonal_pairs_count_is_6 : 
  ∃ (k : ℕ), k = 6 ∧ (number_of_perpendicular_pairs (diagonals A B C A₁ B₁ C₁)) = k := 
sorry

end perpendicular_diagonal_pairs_count_is_6_l787_787338


namespace b_n_eq_T_6_eq_d_n_eq_l787_787621

noncomputable def a (n : ℕ) := sorry
noncomputable def S (n : ℕ) := sorry

axiom a1 : a 1 = 1
axiom a2 : ∀ n : ℕ, S (n + 1) = 4 * a n + 2

noncomputable def b (n : ℕ) := a (n + 1) - 2 * a n

theorem b_n_eq (n : ℕ) : b n = 3 * 2^(n - 1) := sorry

noncomputable def c (n : ℕ) := 1 / (a (n + 1) - 2 * a n)
noncomputable def T : ℕ → ℝ
| 0 => 0
| n + 1 => T n + c n

theorem T_6_eq : T 6 = 21 / 32 := sorry

noncomputable def d (n : ℕ) := a n / 2^n

theorem d_n_eq (n : ℕ) : d n = 3 * n / 4 - 1 / 4 := sorry

end b_n_eq_T_6_eq_d_n_eq_l787_787621


namespace smallest_angle_between_a_c_l787_787002

noncomputable def a : EuclideanSpace ℝ (Fin 3) := ![2, 0, 0]
noncomputable def b : EuclideanSpace ℝ (Fin 3) := ![1, 0, 0]
noncomputable def c : EuclideanSpace ℝ (Fin 3) := ![1, 1, √5 - 1]

noncomputable def k : EuclideanSpace ℝ (Fin 3) := ![2, -1, 0]

theorem smallest_angle_between_a_c :
  ∥a∥ = 2 →
  ∥b∥ = 2 →
  ∥c∥ = 3 →
  (a × (a × c) + b = k) →
  (∀ θ: ℝ, θ = 30) :=
by
  intros h1 h2 h3 h4
  sorry

end smallest_angle_between_a_c_l787_787002


namespace maximum_area_equilateral_triangle_in_rectangle_l787_787434

theorem maximum_area_equilateral_triangle_in_rectangle
  (AB CD : ℝ) (hAB : AB = 12) (hAD : CD = 5) :
  ∃ (p q r : ℕ), q ≠ 1 ∧ (∃ q' : ℕ, q = q'^2 → False) ∧
  p * real.sqrt q - r = (15 : ℝ) * real.sqrt 3 - 10 ∧ 
  p + q + r = 28 :=
by
  use [15, 3, 10]
  split
  -- q is not divisible by any square of a prime number
  have hq : ∀ (q' : ℕ), q ≠ q'^2,
  { 
    intro q',
    -- sorry for simplification purpose
    sorry
  },
  exact ⟨by {intro z, cases p, sorry}, hq⟩,
  -- area calculation and sum check
  repeat {sorry}

end maximum_area_equilateral_triangle_in_rectangle_l787_787434


namespace geometric_progression_of_sequence_sum_of_sequence_terms_l787_787615

-- Definitions from the problem statement
variable {a_n : ℕ → ℝ}
def S (n : ℕ) : ℝ := 2 * a_n n + n^2 - 3 * n - 1

noncomputable def b (n : ℕ) : ℝ := a_n n * Real.cos (n * Real.pi)

-- Problem (1)
theorem geometric_progression_of_sequence (h₁ : ∀ n, S n = 2 * a_n n + n^2 - 3 * n - 1) :
  ∃ r : ℝ, ∀ n ≥ 1, a_n (n + 1) - 2 * (n + 1) = r * (a_n n - 2 * n) := sorry

-- Problem (2)
theorem sum_of_sequence_terms (h₁ : ∀ n, S n = 2 * a_n n + n^2 - 3 * n - 1) :
  ∀ n, ∑ k in Finset.range n, b k = if n % 2 = 1 then - (2 ^ (n - 1) / 3) - n - 5 / 3 
                                    else (1 / 3) * (2 ^ n - 1) + n := sorry

end geometric_progression_of_sequence_sum_of_sequence_terms_l787_787615


namespace num_prime_numbers_with_units_digit_7_num_prime_numbers_less_than_100_with_units_digit_7_l787_787089

def is_prime (n : ℕ) : Prop := Nat.Prime n
def ends_with_7 (n : ℕ) : Prop := n % 10 = 7

theorem num_prime_numbers_with_units_digit_7 (n : ℕ) (h1 : n < 100) (h2 : ends_with_7 n) : is_prime n :=
by sorry

theorem num_prime_numbers_less_than_100_with_units_digit_7 : 
  ∃ (l : List ℕ), (∀ x ∈ l, x < 100 ∧ ends_with_7 x ∧ is_prime x) ∧ l.length = 6 :=
by sorry

end num_prime_numbers_with_units_digit_7_num_prime_numbers_less_than_100_with_units_digit_7_l787_787089


namespace numberOfRealRoots_l787_787960

noncomputable def findNumberOfRealRoots : Prop :=
  ∃ x1 x2 x3 : ℝ,
    (log x1 / log 10) ^ 2 - ⌊log x1 / log 10⌋ - 2 = 0 ∧
    (log x2 / log 10) ^ 2 - ⌊log x2 / log 10⌋ - 2 = 0 ∧
    (log x3 / log 10) ^ 2 - ⌊log x3 / log 10⌋ - 2 = 0 ∧
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3

theorem numberOfRealRoots : findNumberOfRealRoots :=
sorry

end numberOfRealRoots_l787_787960


namespace dice_sum_not_20_l787_787762

/-- Given that Louise rolls four standard six-sided dice (with faces numbered from 1 to 6)
    and the product of the numbers on the upper faces is 216, prove that it is not possible
    for the sum of the upper faces to be 20. -/
theorem dice_sum_not_20 (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) 
                        (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) 
                        (product : a * b * c * d = 216) : a + b + c + d ≠ 20 := 
by sorry

end dice_sum_not_20_l787_787762


namespace percentage_non_defective_l787_787319

theorem percentage_non_defective :
  let total_units : ℝ := 100
  let M1_percentage : ℝ := 0.20
  let M2_percentage : ℝ := 0.25
  let M3_percentage : ℝ := 0.30
  let M4_percentage : ℝ := 0.15
  let M5_percentage : ℝ := 0.10
  let M1_defective_percentage : ℝ := 0.02
  let M2_defective_percentage : ℝ := 0.04
  let M3_defective_percentage : ℝ := 0.05
  let M4_defective_percentage : ℝ := 0.07
  let M5_defective_percentage : ℝ := 0.08

  let M1_total := total_units * M1_percentage
  let M2_total := total_units * M2_percentage
  let M3_total := total_units * M3_percentage
  let M4_total := total_units * M4_percentage
  let M5_total := total_units * M5_percentage

  let M1_defective := M1_total * M1_defective_percentage
  let M2_defective := M2_total * M2_defective_percentage
  let M3_defective := M3_total * M3_defective_percentage
  let M4_defective := M4_total * M4_defective_percentage
  let M5_defective := M5_total * M5_defective_percentage

  let total_defective := M1_defective + M2_defective + M3_defective + M4_defective + M5_defective
  let total_non_defective := total_units - total_defective
  let percentage_non_defective := (total_non_defective / total_units) * 100

  percentage_non_defective = 95.25 := by
  sorry

end percentage_non_defective_l787_787319


namespace solve_for_m_l787_787286

theorem solve_for_m (x y m : ℝ) 
  (h1 : 2 * x + y = 3 * m) 
  (h2 : x - 4 * y = -2 * m)
  (h3 : y + 2 * m = 1 + x) :
  m = 3 / 5 := 
by 
  sorry

end solve_for_m_l787_787286


namespace max_S_value_max_S_value_achievable_l787_787369

theorem max_S_value (x y z w : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  (x^2 * y + y^2 * z + z^2 * w + w^2 * x - x * y^2 - y * z^2 - z * w^2 - w * x^2) ≤ 8 / 27 :=
sorry

theorem max_S_value_achievable :
  ∃ (x y z w : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) ∧ 
  (x^2 * y + y^2 * z + z^2 * w + w^2 * x - x * y^2 - y * z^2 - z * w^2 - w * x^2) = 8 / 27 :=
sorry

end max_S_value_max_S_value_achievable_l787_787369


namespace max_attendance_days_l787_787947
-- Import Mathlib to bring in the entirety of the necessary library.

-- Define the conditions.
def unavailable_Anna := { "Mon", "Wed", "Sat" }
def unavailable_Bill := { "Tues", "Thurs", "Fri" }
def unavailable_Carl := { "Mon", "Tues", "Thurs", "Fri", "Sat" }
def unavailable_Dana := { "Mon", "Wed" }

-- List of all days.
def days := { "Mon", "Tues", "Wed", "Thurs", "Fri", "Sat"}

-- Definition of available days for all participants.
def available_day (day: String) : Prop :=
  (day ∉ unavailable_Anna) ∧
  (day ∉ unavailable_Bill) ∧
  (day ∉ unavailable_Carl) ∧
  (day ∉ unavailable_Dana)

-- Prove the days when the most people can attend.
theorem max_attendance_days :
  (available_day "Tues") ∨
  (available_day "Wed") ∨
  (available_day "Thurs") ∨
  (available_day "Fri") ∨
  (available_day "Sat") := by
  -- The proof is omitted for the purposes of this task.
  sorry

end max_attendance_days_l787_787947


namespace toothpicks_at_20th_stage_l787_787447

theorem toothpicks_at_20th_stage (n : ℕ) (a d : ℕ) (h_a : a = 3) (h_d : d = 3) :
  T n = a + d * (n - 1) → T 20 = 60 :=
by
  intro formula
  rw [h_a, h_d]
  sorry

end toothpicks_at_20th_stage_l787_787447


namespace range_nonvoid_interior_l787_787350

noncomputable def integral (a b : ℝ) (f : ℝ → ℝ) : ℝ := ∫ t in a..b, f t

theorem range_nonvoid_interior
  (f g : ℝ → ℝ) 
  (hf : ∀ x, f x > 0) 
  (hg : ∀ y, g y > 0) 
  (hfc : Continuous f)
  (hgc : Continuous g) 
  (E : set ℝ) 
  (hE : E ⊆ Icc 0 ⊤) 
  (hEm : (measure_theory.measure_space.volume E).toReal > 0) 
  :
  ∃ (a b : ℝ), 
    (∃ (x y ∈ E), a = integral 0 x f ∧ b = integral 0 y g) ∧ 
    ∃ ε > 0, ∀ (u v : ℝ), abs (u - a) < ε → abs (v - b) < ε → (∃ (x y ∈ E), u = integral 0 x f ∧ v = integral 0 y g) :=
sorry

end range_nonvoid_interior_l787_787350


namespace lychee_ratio_l787_787041

theorem lychee_ratio (total_lychees : ℕ) (sold_lychees : ℕ) (remaining_home : ℕ) (remaining_after_eat : ℕ) 
    (h1: total_lychees = 500) 
    (h2: sold_lychees = total_lychees / 2) 
    (h3: remaining_home = total_lychees - sold_lychees) 
    (h4: remaining_after_eat = 100)
    (h5: remaining_after_eat + (remaining_home - remaining_after_eat) = remaining_home) : 
    (remaining_home - remaining_after_eat) / remaining_home = 3 / 5 :=
by
    -- Proof is omitted
    sorry

end lychee_ratio_l787_787041


namespace eval_polynomial_eq_5_l787_787595

noncomputable def eval_polynomial_at_root : ℝ :=
let x := (3 + 3 * Real.sqrt 5) / 2 in
if h : x^2 - 3*x - 9 = 0 then x^3 - 3*x^2 - 9*x + 5 else 0

theorem eval_polynomial_eq_5 :
  eval_polynomial_at_root = 5 :=
by
  sorry

end eval_polynomial_eq_5_l787_787595


namespace train_speed_kmph_l787_787536

-- Definitions
def train_length : ℝ := 110 -- Length in meters
def bridge_length : ℝ := 170 -- Length in meters
def crossing_time : ℝ := 16.7986561075114 -- Time in seconds

-- Total distance
def total_distance : ℝ :=
  train_length + bridge_length

-- Speed in m/s
def speed_mps : ℝ :=
  total_distance / crossing_time

-- Speed in km/h
def speed_kmph : ℝ :=
  speed_mps * 3.6

-- Theorem
theorem train_speed_kmph : speed_kmph = 60 :=
by
  sorry

end train_speed_kmph_l787_787536


namespace strawberry_jelly_sales_l787_787891

def jelly_sales (grape strawberry raspberry plum : ℕ) : Prop :=
  grape = 2 * strawberry ∧
  raspberry = 2 * plum ∧
  raspberry = grape / 3 ∧
  plum = 6

theorem strawberry_jelly_sales {grape strawberry raspberry plum : ℕ}
    (h : jelly_sales grape strawberry raspberry plum) : 
    strawberry = 18 :=
by
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2
  sorry

end strawberry_jelly_sales_l787_787891


namespace probability_three_white_balls_l787_787503

def total_balls := 11
def white_balls := 5
def black_balls := 6
def balls_drawn := 5
def white_balls_drawn := 3
def black_balls_drawn := 2

theorem probability_three_white_balls :
  let total_outcomes := Nat.choose total_balls balls_drawn
  let favorable_outcomes := (Nat.choose white_balls white_balls_drawn) * (Nat.choose black_balls black_balls_drawn)
  (favorable_outcomes : ℚ) / total_outcomes = 25 / 77 :=
by
  sorry

end probability_three_white_balls_l787_787503


namespace count_permutations_correct_l787_787602

noncomputable def count_valid_permutations (n : ℕ) : ℕ :=
  if n % 2 = 1 then 1 else 0

theorem count_permutations_correct :
  ∀ n, count_valid_permutations n =
    (if ∃ p : Equiv.Perm (Fin n),
        ∃! i : Fin n, p (p i) ≥ i then 1 else 0) :=
by
  intros n
  sorry

end count_permutations_correct_l787_787602


namespace simplest_quadratic_radical_l787_787549

noncomputable def optionA := Real.sqrt 7
noncomputable def optionB := Real.sqrt 9
noncomputable def optionC := Real.sqrt 12
noncomputable def optionD := Real.sqrt (2 / 3)

theorem simplest_quadratic_radical :
  optionA = Real.sqrt 7 ∧
  optionB = Real.sqrt 9 ∧
  optionC = Real.sqrt 12 ∧
  optionD = Real.sqrt (2 / 3) ∧
  (optionB = 3 ∧ optionC = 2 * Real.sqrt 3 ∧ optionD = Real.sqrt 6 / 3) ∧
  (optionA < 3 ∧ optionA < 2 * Real.sqrt 3 ∧ optionA < Real.sqrt 6 / 3) :=
  by {
    sorry
  }

end simplest_quadratic_radical_l787_787549


namespace line_parallel_not_coincident_l787_787308

theorem line_parallel_not_coincident (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ k : ℝ, (∀ x y : ℝ, a * x + 2 * y + 6 = k * (x + (a - 1) * y + (a^2 - 1))) →
  a = -1 :=
by
  sorry

end line_parallel_not_coincident_l787_787308


namespace proof_problem_l787_787274

noncomputable def f (x k : ℝ) : ℝ :=
  (x - 1) * Real.exp x - k * x^2

theorem proof_problem (k : ℝ) (h1 : k > 1 / 2)
  (h2 : ∀ x : ℝ, x > 0 → f x k + (Real.log (2 * k))^2 + 2 * k * (Real.log (Real.exp / (2 * k))) > 0) :
  f (k - 1 + Real.log 2) k < f k k := sorry

end proof_problem_l787_787274


namespace non_prime_300th_term_l787_787121

open Nat

-- Definition to check prime numbers
def is_prime (n : ℕ) : Prop :=
  if n <= 1 then false else ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Definition of the sequence of non-primes by filtering primes out
def non_prime_seq : ℕ → ℕ :=
  Nat.recOn 1 (λ n acc, if is_prime acc then acc + 1 else acc)

-- The theorem we want to prove
theorem non_prime_300th_term : non_prime_seq 300 = 609 :=
  sorry

end non_prime_300th_term_l787_787121


namespace tangent_line_y_eq_x_cubed_at_origin_l787_787235

theorem tangent_line_y_eq_x_cubed_at_origin :
  let f := λ x : ℝ, x^3
  let f' := λ x : ℝ, 3 * x^2
  f'(0) = 0
→ ∀ x : ℝ, x = 0 → f x = f 0:
(by {
  sorry
}) 

end tangent_line_y_eq_x_cubed_at_origin_l787_787235


namespace card_game_final_amounts_l787_787182

theorem card_game_final_amounts 
  (T : ℕ) -- T is the total initial amount of money in the game
  (h_initial : T = 1080) -- Given T = 12 * 90 = 1080 from the solution
  (win_amount : ℕ) 
  (h_win : win_amount = 12) 
  (initial_ratio_A : ℚ) (initial_ratio_B : ℚ) (initial_ratio_C : ℚ)
  (final_ratio_A : ℚ) (final_ratio_B : ℚ) (final_ratio_C : ℚ)
  (h_initial_ratios: initial_ratio_A = 7 / 18 ∧ initial_ratio_B = 6 / 18 ∧ initial_ratio_C = 5 / 18)
  (h_final_ratios: final_ratio_A = 6 / 15 ∧ final_ratio_B = 5 / 15 ∧ final_ratio_C = 4 / 15) :
  (let final_A := final_ratio_A * T in
   let final_B := final_ratio_B * T in
   let final_C := final_ratio_C * T in
   final_A = 432 ∧ final_B = 360 ∧ final_C = 288) :=
  sorry

end card_game_final_amounts_l787_787182


namespace initial_sand_in_bucket_A_l787_787929

theorem initial_sand_in_bucket_A (C : ℝ) : 
  let bucketB_capacity := C / 2
  let sand_in_B := (3 / 8) * bucketB_capacity
  let after_pour := (7 / 16) * C
  let x := after_pour - sand_in_B
  x / C = 1 / 4 := by
  let bucketB_capacity := C / 2
  let sand_in_B := (3 / 8) * bucketB_capacity
  let after_pour := (7 / 16) * C
  let x := after_pour - sand_in_B
  show x / C = 1 / 4
  sorry

end initial_sand_in_bucket_A_l787_787929


namespace sufficient_but_not_necessary_condition_for_x_1_l787_787534

noncomputable def sufficient_but_not_necessary_condition (x : ℝ) : Prop :=
(x = 1 → (x = 1 ∨ x = 2)) ∧ ¬ ((x = 1 ∨ x = 2) → x = 1)

theorem sufficient_but_not_necessary_condition_for_x_1 :
  sufficient_but_not_necessary_condition 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_for_x_1_l787_787534


namespace count_integers_congruent_to_7_mod_13_l787_787673

theorem count_integers_congruent_to_7_mod_13:
  (∃ S : Finset ℕ, S.card = 154 ∧ ∀ n ∈ S, n < 2000 ∧ n % 13 = 7) :=
by
  sorry

end count_integers_congruent_to_7_mod_13_l787_787673


namespace determine_a_l787_787678

theorem determine_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x + 1 > 0) ↔ a = 2 := by
  sorry

end determine_a_l787_787678


namespace geometric_series_sum_l787_787200

theorem geometric_series_sum : 
  let a0 := (1 / 2 : ℝ)
  let r := (1 / 2 : ℝ) 
  let n := 5
  (∑ i in finset.range (n + 1), a0 * r ^ i) = 31 / 32 :=
by
  let a0 := (1 / 2 : ℝ)
  let r := (1 / 2 : ℝ) 
  let n := 5
  let finite_geo_sum : ℝ := ∑ i in finset.range (n + 1), a0 * r ^ i
  have h_finite_geo_sum : finite_geo_sum = a0 * (1 - r^n) / (1 - r) := sorry
  rw [h_finite_geo_sum]
  have h_geometric_sum : a0 * (1 - r^n) / (1 - r) = 31 / 32 := sorry
  exact h_geometric_sum

end geometric_series_sum_l787_787200


namespace three_digit_powers_of_two_count_l787_787677

theorem three_digit_powers_of_two_count : (finset.filter (λ n : ℕ, 100 ≤ 2^n ∧ 2^n < 1000) (finset.range 100)).card = 3 :=
by
  sorry

end three_digit_powers_of_two_count_l787_787677


namespace probability_a_leq_b_l787_787524

def setA := {1, 2, 3}
def setB := {2, 3, 4}

def P (S : Set (ℕ × ℕ)) : ℚ :=
  (S.card : ℚ) / ((setA.prod setB).card : ℚ)

theorem probability_a_leq_b : P({p | p.1 ∈ setA ∧ p.2 ∈ setB ∧ p.1 <= p.2}) = 8/9 :=
  by
  sorry

end probability_a_leq_b_l787_787524


namespace opposite_of_neg_sqrt_two_l787_787826

theorem opposite_of_neg_sqrt_two : -(-Real.sqrt 2) = Real.sqrt 2 := 
by {
  sorry
}

end opposite_of_neg_sqrt_two_l787_787826


namespace total_distance_collinear_centers_l787_787565

theorem total_distance_collinear_centers (r1 r2 r3 : ℝ) (d12 d13 d23 : ℝ) 
  (h1 : r1 = 6) 
  (h2 : r2 = 14) 
  (h3 : d12 = r1 + r2) 
  (h4 : d13 = r3 - r1) 
  (h5 : d23 = r3 - r2) :
  d13 = d12 + r1 := by
  -- proof follows here
  sorry

end total_distance_collinear_centers_l787_787565


namespace gcd_factorial_sub_one_l787_787017

theorem gcd_factorial_sub_one (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq : p > q) :
  Nat.gcd (p.factorial - 1) (q.factorial - 1) ≤ p ^ (5/3 : ℚ) := 
sorry

end gcd_factorial_sub_one_l787_787017


namespace oliver_cards_l787_787379

variable {MC AB BG : ℕ}

theorem oliver_cards : 
  (BG = 48) → 
  (BG = 3 * AB) → 
  (MC = 2 * AB) → 
  MC = 32 := 
by 
  intros h1 h2 h3
  sorry

end oliver_cards_l787_787379


namespace smallest_positive_value_is_expr1_l787_787585

noncomputable def sqrt2 : ℝ := Real.sqrt 2
noncomputable def sqrt8 : ℝ := Real.sqrt 8
noncomputable def sqrt18 : ℝ := Real.sqrt 18
noncomputable def sqrt17 : ℝ := Real.sqrt 17

def expr1 : ℝ := 12 - 4 * sqrt8
def expr2 : ℝ := 4 * sqrt8 - 12
def expr3 : ℝ := 15 - 3 * sqrt18
def expr4 : ℝ := 35 - 7 * sqrt17
def expr5 : ℝ := 7 * sqrt17 - 35

theorem smallest_positive_value_is_expr1 :
  expr1 < expr2 ∧ expr1 < expr3 ∧ expr1 < expr4 ∧ (0 < expr1) :=
by {
  sorry
}

end smallest_positive_value_is_expr1_l787_787585


namespace rational_powers_diff_is_integer_l787_787267

noncomputable def is_integer (r : ℚ) : Prop :=
  ∃ (n : ℤ), r = n

theorem rational_powers_diff_is_integer
    (a b : ℚ)
    (h_distinct : a ≠ b)
    (h_positive : 0 < a ∧ 0 < b)
    (h_infinite : ∀ n : ℕ, ∃ m > n, (a ^ m - b ^ m) ∈ ℤ) :
    is_integer a ∧ is_integer b :=
sorry -- proof required

end rational_powers_diff_is_integer_l787_787267


namespace prob_exactly_two_trains_on_time_is_0_398_l787_787588

-- Definitions and conditions
def eventA := true
def eventB := true
def eventC := true

def P_A : ℝ := 0.8
def P_B : ℝ := 0.7
def P_C : ℝ := 0.9

def P_not_A : ℝ := 1 - P_A
def P_not_B : ℝ := 1 - P_B
def P_not_C : ℝ := 1 - P_C

-- Question definition (to be proved)
def exact_two_on_time : ℝ :=
  P_A * P_B * P_not_C + P_A * P_not_B * P_C + P_not_A * P_B * P_C

-- Theorem statement
theorem prob_exactly_two_trains_on_time_is_0_398 :
  exact_two_on_time = 0.398 := sorry

end prob_exactly_two_trains_on_time_is_0_398_l787_787588


namespace cabin_price_correct_l787_787663

noncomputable def cabin_price 
  (cash : ℤ)
  (cypress_trees : ℤ) (pine_trees : ℤ) (maple_trees : ℤ)
  (price_cypress : ℤ) (price_pine : ℤ) (price_maple : ℤ)
  (remaining_cash : ℤ)
  (expected_price : ℤ) : Prop :=
   cash + (cypress_trees * price_cypress + pine_trees * price_pine + maple_trees * price_maple) - remaining_cash = expected_price

theorem cabin_price_correct :
  cabin_price 150 20 600 24 100 200 300 350 130000 :=
by
  sorry

end cabin_price_correct_l787_787663


namespace exists_triplet_with_gcd_conditions_l787_787730

-- Given the conditions as definitions in Lean.
variables (S : Set ℕ)
variable [Infinite S] -- S is an infinite set of positive integers.
variables {a b c d x y z : ℕ}
variable (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S)
variable (hdistinct : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d) 
variable (hgcd_neq : gcd a b ≠ gcd c d)

-- The formal proof statement.
theorem exists_triplet_with_gcd_conditions :
  ∃ (x y z : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ gcd x y = gcd y z ∧ gcd y z ≠ gcd z x :=
sorry

end exists_triplet_with_gcd_conditions_l787_787730


namespace ratio_of_areas_l787_787044

noncomputable def area (A B C D : ℝ) : ℝ := 0  -- Placeholder, exact area definition will require geometrical formalism.

variables (A B C D P Q R S : ℝ)

-- Define the conditions
variables (h1 : AB = BP) (h2 : BC = CQ) (h3 : CD = DR) (h4 : DA = AS)

-- Lean 4 statement for the proof problem
theorem ratio_of_areas : area A B C D / area P Q R S = 1/5 :=
sorry

end ratio_of_areas_l787_787044


namespace mike_trip_represented_by_graph_e_l787_787029

-- Definitions based on the conditions
def drives_from_home_to_coffee_shop_slowly (dist time: ℝ) : Prop :=
  -- Gradual rise in distance over time
  ∀ t ≤ time, 0 < dist

def stops_at_coffee_shop (duration: ℝ) : Prop :=
  duration = 30

def drives_to_highway_then_mall (dist time: ℝ) : Prop :=
  -- Moderate then steeper rise in distance over time
  ∃ t₁ t₂, t₁ < t₂ ∧ 0 < t₁ / t₂ ∧ t₁ < time ∧ t₂ < time ∧ dist/t₁ < dist/t₂

def shops_at_mall (duration: ℝ) : Prop :=
  duration = 45

def drives_back_home_with_stop (dist time: ℝ) : Prop :=
  -- Steep then moderate decline and stop at grocery store
  ∃ t₁ t₂ t₃, t₁ < t₂ ∧ t₂ < t₃ ∧ 
    dist/t₁ > dist/t₂ ∧ dist/t₂ > dist / t₃ ∧ 
    t₃ = 15

-- Problem statement in Lean
theorem mike_trip_represented_by_graph_e (dist time: ℝ) :
  drives_from_home_to_coffee_shop_slowly dist 30 →
  stops_at_coffee_shop 30 →
  drives_to_highway_then_mall dist 45 →
  shops_at_mall 45 →
  drives_back_home_with_stop dist 15 →
  (graph_represents_trip dist time = E) :=
begin
  sorry
end

end mike_trip_represented_by_graph_e_l787_787029


namespace revenue_fraction_l787_787877

variable (N D J : ℝ)
variable (h1 : J = 1 / 5 * N)
variable (h2 : D = 4.166666666666666 * (N + J) / 2)

theorem revenue_fraction (h1 : J = 1 / 5 * N) (h2 : D = 4.166666666666666 * (N + J) / 2) : N / D = 2 / 5 :=
by
  sorry

end revenue_fraction_l787_787877


namespace quadratic_problem_solution_l787_787067

noncomputable def solve_quadratic_problem (b c : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), 
    (x1 ≠ x2) ∧ 
    (x1^2 + b * x1 + c = 0) ∧ 
    (x2^2 + b * x2 + c = 0) ∧ 
    (abs (x1 - x2) = 1) ∧ 
    (abs (b - c) = 1) 

example : set (ℝ × ℝ) := 
  {(-1, 0), (5, 6), (1, 0), (3, 2)}

theorem quadratic_problem_solution :
  ∀ (b c : ℝ), solve_quadratic_problem b c ↔ (b, c) ∈ example :=
sorry

end quadratic_problem_solution_l787_787067


namespace count_valid_cs_l787_787608

def is_multiple_of_10 (n : ℕ) : Prop :=
  n % 10 = 0

def is_3_more_than_multiple_of_10 (n : ℕ) : Prop :=
  (n - 3) % 10 = 0

def has_solution (c : ℕ) : Prop :=
  (is_multiple_of_10 c ∨ is_3_more_than_multiple_of_10 c)

theorem count_valid_cs : 
  (finset.range 2001).filter has_solution).card = 402 :=
sorry

end count_valid_cs_l787_787608


namespace least_number_to_add_l787_787491

theorem least_number_to_add (n d : ℕ) (h : n = 1024) (h_d : d = 25) :
  ∃ x : ℕ, (n + x) % d = 0 ∧ x = 1 :=
by sorry

end least_number_to_add_l787_787491


namespace journeymen_percentage_after_layoff_l787_787376

noncomputable def total_employees : ℝ := 20210
noncomputable def fraction_journeymen : ℝ := 2 / 7
noncomputable def total_journeymen : ℝ := total_employees * fraction_journeymen
noncomputable def laid_off_journeymen : ℝ := total_journeymen / 2
noncomputable def remaining_journeymen : ℝ := total_journeymen / 2
noncomputable def remaining_employees : ℝ := total_employees - laid_off_journeymen
noncomputable def journeymen_percentage : ℝ := (remaining_journeymen / remaining_employees) * 100

theorem journeymen_percentage_after_layoff : journeymen_percentage = 16.62 := by
  sorry

end journeymen_percentage_after_layoff_l787_787376


namespace determinant_of_angles_l787_787011

theorem determinant_of_angles (A B C : ℝ) 
  (h_triangle : A + B + C = π) : 
  Matrix.det ![
    ![Real.sin A ^ 2, Real.cot A, Real.sin A], 
    ![Real.sin B ^ 2, Real.cot B, Real.sin B], 
    ![Real.sin C ^ 2, Real.cot C, Real.sin C]
  ] = 0 :=
by
  sorry

end determinant_of_angles_l787_787011


namespace number_of_convex_quadrilaterals_l787_787409

-- Each definition used in Lean 4 statement should directly appear in the conditions problem.

variable {n : ℕ} -- Definition of n in Lean

-- Conditions
def distinct_points_on_circle (n : ℕ) : Prop := n = 10

-- Question and correct answer
theorem number_of_convex_quadrilaterals (h : distinct_points_on_circle n) : 
    (n.choose 4) = 210 := by
  sorry

end number_of_convex_quadrilaterals_l787_787409


namespace train_cross_pole_time_l787_787539

-- Defining the given conditions
def speed_km_hr : ℕ := 54
def length_m : ℕ := 135

-- Conversion of speed from km/hr to m/s
def speed_m_s : ℤ := (54 * 1000) / 3600

-- Statement to be proved
theorem train_cross_pole_time : (length_m : ℤ) / speed_m_s = 9 := by
  sorry

end train_cross_pole_time_l787_787539


namespace largest_rectangle_in_circle_l787_787701

theorem largest_rectangle_in_circle {r : ℝ} (h : r = 6) : 
  ∃ A : ℝ, A = 72 := 
by 
  sorry

end largest_rectangle_in_circle_l787_787701


namespace truth_values_1_truth_values_2_l787_787133

variables (p q : Prop)

-- Definitions of the conditions
def p1 : Prop := 3.prime
def q1 : Prop := ∃ m, 3 = 2 * m
def p2 : Prop := ∃ x, x = -2 ∧ x^2 + x - 2 = 0
def q2 : Prop := ∃ x, x = 1 ∧ x^2 + x - 2 = 0

-- Problems translated to proof statements
theorem truth_values_1 : (p1 ∨ q1) = true ∧ (p1 ∧ q1) = false ∧ (¬p1) = false := sorry

theorem truth_values_2 : (p2 ∨ q2) = true ∧ (p2 ∧ q2) = true ∧ (¬p2) = false := sorry

end truth_values_1_truth_values_2_l787_787133


namespace Courtney_total_marbles_l787_787211

theorem Courtney_total_marbles (first_jar second_jar third_jar : ℕ) 
  (h1 : first_jar = 80)
  (h2 : second_jar = 2 * first_jar)
  (h3 : third_jar = first_jar / 4) :
  first_jar + second_jar + third_jar = 260 := 
by
  sorry

end Courtney_total_marbles_l787_787211


namespace sum_of_roots_of_quadratic_l787_787127

theorem sum_of_roots_of_quadratic :
  ∑ x in {x : ℝ | x^2 - 16 * x + 21 = 0}.to_finset = 16 :=
by
  sorry

end sum_of_roots_of_quadratic_l787_787127


namespace cone_volume_l787_787945

-- Define the volume function for a cone
def volume_of_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

-- problem: Diameter of the cone is 12 cm and height is 8 cm
def diameter : ℝ := 12
def height : ℝ := 8

-- Radius is half of diameter
def radius : ℝ := diameter / 2

-- Statement of the problem
theorem cone_volume : volume_of_cone radius height = 96 * π := by
  -- Place all necessary steps here 
  sorry

end cone_volume_l787_787945


namespace user_count_exceed_50000_l787_787025

noncomputable def A (t : ℝ) (k : ℝ) := 500 * Real.exp (k * t)

theorem user_count_exceed_50000 :
  (∃ k : ℝ, A 10 k = 2000) →
  (∀ t : ℝ, A t k > 50000) →
  ∃ t : ℝ, t >= 34 :=
by
  sorry

end user_count_exceed_50000_l787_787025


namespace circle_intersection_property_l787_787111

theorem circle_intersection_property
  (k1 k2 k3 : Circle)
  (O : Point)
  (A B C A' B' C' : Point)
  (h1 : k1 ∩ k2 = {O, C})
  (h2 : k1 ∩ k3 = {O, B})
  (h3 : k2 ∩ k3 = {O, A})
  (h4 : O ∈ Triangle A B C)
  (hAO : Line O A = k1 ∩ k2 = {O, A'})
  (hBO : Line O B = k2 ∩ k3 = {O, B'})
  (hCO : Line O C = k1 ∩ k2 = {O, C'}) :
  ∣AO∣ / ∣AA'∣ + ∣BO∣ / ∣BB'∣ + ∣CO∣ / ∣CC'∣ = 1 :=
sorry

end circle_intersection_property_l787_787111


namespace limit_computation_l787_787198

noncomputable def limit_example : Real :=
  lim (λ x:Real, (5^(2*x) - 2^(3*x))/(sin x + sin (x^2))) 0

theorem limit_computation :
  limit_example = log (25 / 8) := by
  sorry

end limit_computation_l787_787198


namespace sum_of_primes_le_square_l787_787748

open Nat

def prime (n : Nat) := ∃ k : Nat, k > 1 ∧ n = Nat.prime k

def sum_of_primes (n : Nat) : Nat :=
  ∑ i in range n, if h : prime (i + 1) then (i + 1) else 0

theorem sum_of_primes_le_square (n : Nat) :
  ∃ m : Nat, sum_of_primes n ≤ m ^ 2 ∧ m ^ 2 ≤ sum_of_primes (n + 1) := by
  sorry

end sum_of_primes_le_square_l787_787748


namespace total_fence_poles_needed_l787_787859

def number_of_poles_per_side := 27

theorem total_fence_poles_needed (n : ℕ) (h : n = number_of_poles_per_side) : 
  4 * n - 4 = 104 :=
by sorry

end total_fence_poles_needed_l787_787859


namespace coeff_x2_l787_787336

theorem coeff_x2 (n : ℕ) (hn : 4^n = 32 * 2^n) :
  let x := 5,
  (3^2 * (Nat.choose x 2) = 90) :=
by
  sorry

end coeff_x2_l787_787336


namespace fibonacci_diff_is_fibonacci_l787_787365

def fibonacci_seq : ℕ → ℕ 
| 0 := 1
| 1 := 1
| (n + 2) :=  fibonacci_seq (n + 1) + fibonacci_seq n

-- Sequence of products of two Fibonacci numbers, ordered increasingly
noncomputable def product_fib (n : ℕ) : ℕ := sorry

-- Sequence of differences of products of Fibonacci terms
def diff_prod_fib (n : ℕ) : ℕ :=
  product_fib (n + 1) - product_fib n

theorem fibonacci_diff_is_fibonacci :
  ∀ n : ℕ, ∃ k : ℕ, diff_prod_fib n = fibonacci_seq k :=
sorry

end fibonacci_diff_is_fibonacci_l787_787365


namespace pow_fraction_eq_l787_787476

theorem pow_fraction_eq : (4:ℕ) = 2^2 ∧ (8:ℕ) = 2^3 → (4^800 / 8^400 = 2^400) :=
by
  -- proof steps should go here, but they are omitted as per the instruction
  sorry

end pow_fraction_eq_l787_787476


namespace mod_congruent_integers_l787_787675

theorem mod_congruent_integers (n : ℕ) : (∃ k : ℕ, n = 13 * k + 7 ∧ 7 + 13 * k < 2000) ↔ 
  (7 + 13 * 153 < 2000 ∧ ∀ m : ℕ, (0 ≤ m ∧ m ≤ 153) → n = 13 * m + 7) := by
  sorry

end mod_congruent_integers_l787_787675


namespace perpendicular_chords_exist_l787_787498

theorem perpendicular_chords_exist (red_points : Fin 100 → ℝ) (arc_lengths : Fin 100 → ℕ) :
  ∀ i j k l ∈ Finset.univ, 
  (arc_lengths i).toNat ∈ Finset.range 1 101 → 
  (arc_lengths j).toNat ∈ Finset.range 1 101 → 
  ∃ (p q r s: Fin 100), 
  p ≠ q ∧ r ≠ s ∧ 
  (p ≠ r ∨ q ≠ s) ∧ 
  (is_perpendicular_chord p q r s red_points) :=
sorry

def is_perpendicular_chord (p q r s: Fin 100) (red_points : Fin 100 → ℝ) : Prop := 
  -- Placeholder definition; needs to be appropriately defined based on the circle setup
  -- Ensure two chords pq and rs are perpendicular
  ∠(red_points p) (red_points q) = 90 ∧ ∠(red_points r) (red_points s) = 90

end perpendicular_chords_exist_l787_787498


namespace part1_part2_l787_787977

-- Definitions based on the problem conditions
def f (x : ℝ) : ℝ := Real.exp x - 2 * Real.log x
def g (x : ℝ) : ℝ := Real.cos x - 1 + x^2 / 2

-- Conditions: x > 0
variable {x : ℝ} (hx : 0 < x)

-- Part (1): Prove that g(x) > 0 for x > 0
theorem part1 (hx : 0 < x) : g x > 0 :=
sorry

-- Part (2) condition: f(x) ≥ 2 * Real.log((x + 1) / x) + (x + 2) / (x + 1) - Real.cos (real.abs a * x)
def condition (a x : ℝ) : Prop :=
  f x ≥ 2 * Real.log((x + 1) / x) + (x + 2) / (x + 1) - Real.cos (real.abs a * x)

-- Part (2): Prove the range of a is [-1, 1]
theorem part2 (hx : 0 < x) (hcond : ∀ x > 0, condition a x) : a ∈ Set.Icc (-1) 1 :=
sorry

end part1_part2_l787_787977


namespace ISI_club_members_l787_787695

theorem ISI_club_members 
  (members : Type)
  (committee : Type)
  (is_member : members → committee → Prop)
  (h1 : ∀ (m : members), ∃! (c1 c2 : committee), c1 ≠ c2 ∧ is_member m c1 ∧ is_member m c2)
  (h2 : ∀ (c1 c2 : committee), c1 ≠ c2 → ∃! (m : members), is_member m c1 ∧ is_member m c2)
  (h3 : ∃ (c : fin 5), true)  -- There are 5 committees
  : fintype.card members = 10 := sorry

end ISI_club_members_l787_787695


namespace find_areas_l787_787167

theorem find_areas (a b : ℕ) (h1 : 104 * b = 143 * 40) (h2 : 25 * 66 = a * b) : a = 30 ∧ b = 55 := 
by {
    have h1' : b = 55 := by {
        calc b
            = (143 * 40) / 104 : by { sorry }
            = 55 : by { sorry },
    },
    have h2' : a = 30 := by {
        rw h1' at h2,
        calc a
            = (25 * 66) / 55 : by { sorry }
            = 30 : by { sorry },
    },
    exact ⟨h2', h1'⟩,
}

end find_areas_l787_787167


namespace evalTrigExpressionIsCorrect_l787_787226

noncomputable def evalTrigExpression : Prop :=
  (sin (30 * π / 180) * cos (24 * π / 180) + cos (150 * π / 180) * cos (90 * π / 180)) / 
  (sin (34 * π / 180) * cos (16 * π / 180) + cos (146 * π / 180) * cos (106 * π / 180)) =
  (cos (24 * π / 180)) / (2 * sin (18 * π / 180))

theorem evalTrigExpressionIsCorrect : evalTrigExpression := 
by
  sorry

end evalTrigExpressionIsCorrect_l787_787226


namespace automobile_travel_distance_l787_787185

theorem automobile_travel_distance
  (a r : ℝ) : 
  let feet_per_yard := 3
  let seconds_per_minute := 60
  let travel_feet := a / 4
  let travel_seconds := 2 * r
  let rate_yards_per_second := (travel_feet / travel_seconds) / feet_per_yard
  let total_seconds := 10 * seconds_per_minute
  let total_yards := rate_yards_per_second * total_seconds
  total_yards = 25 * a / r := by
  sorry

end automobile_travel_distance_l787_787185


namespace toggle_off_impossible_l787_787556

-- Definitions as per the problem conditions
def toggle (grid : Array (Array Bool)) (row col : Fin 5) : Array (Array Bool) :=
  let toggledRow (r : Array Bool) := r.modify row.toNat (not)
  let toggledGrid := grid.modify row.toNat toggledRow
  toggledGrid.modify_all (fun c => c.modify col.toNat (not))

def allOnesOffable (grid : Array (Array Bool)) : Prop :=
  ∃ n : Nat, ∃ ops : List (Fin 5 × Fin 5),
    n = ops.length ∧ (ops.foldl (λ g op => toggle g op.1 op.2) grid = Array.mkArray 5 (Array.mkArray 5 false))

-- Initial grid of all lights on
def initialGrid : Array (Array Bool) :=
  Array.mkArray 5 (Array.mkArray 5 true)

-- Formal statement of the proof problem
theorem toggle_off_impossible : ¬ allOnesOffable initialGrid :=
sorry

end toggle_off_impossible_l787_787556


namespace square_side_length_in_right_triangle_l787_787049

theorem square_side_length_in_right_triangle :
  ∀ (DE EF DF : ℝ) (s : ℝ),
  DE = 5 ∧ EF = 12 ∧ DF = 13 ∧
  (∃ P Q S R, 
    P ∈ DF ∧ Q ∈ DF ∧ 
    S ∈ DE ∧ 
    R ∈ EF ∧ 
    square PQRS ∧ 
    ℓ PQ = s) →
  s = 780 / 169 :=
by {
  sorry
}

end square_side_length_in_right_triangle_l787_787049


namespace max_value_of_f_range_of_k_l787_787654

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (ln x) / x - k / x

/-- Proof problem statement: part 1 -/
theorem max_value_of_f {x k : ℝ} 
  (h1 : f 1 k = ln 1)
  (h2 : (deriv (λ x, f x k)) 1 = 10) :
  let k := 9 in
  ∃ x, x > 0 ∧ f x k = (1 / exp 10) :=
sorry

/-- Proof problem statement: part 2 -/
theorem range_of_k {x : ℝ} 
  (h3 : x ≥ 1)
  (h4 : x^2 * (f x k) + 1 / (x + 1) ≥ 0)
  (h5 : k ≥ (1 / 2) * x^2 + (exp 2 - 2) * x - exp x - 7) :
  k ∈ (Icc (exp 2 - 9) (1 / 2)) :=
sorry

end max_value_of_f_range_of_k_l787_787654


namespace spatial_diagonals_count_l787_787881

-- Defining an instance of the polyhedron with the given properties
def polyhedron : Type := {
  vertices : ℕ,
  edges : ℕ,
  faces : ℕ,
  triangular_faces : ℕ,
  quadrilateral_faces : ℕ,
  spatial_diagonals : ℕ
}

-- Defining the given polyhedron P
def P : polyhedron := {
  vertices := 26,
  edges := 60,
  faces := 36,
  triangular_faces := 24,
  quadrilateral_faces := 12,
  spatial_diagonals := 241
}

-- Stating the main theorem
theorem spatial_diagonals_count (P : polyhedron) : 
  P.vertices = 26 → 
  P.edges = 60 → 
  P.faces = 36 → 
  P.triangular_faces = 24 → 
  P.quadrilateral_faces = 12 → 
  P.spatial_diagonals = 241 := 
  by
  sorry

end spatial_diagonals_count_l787_787881


namespace count_4_digit_multiples_of_5_is_9_l787_787293

noncomputable def count_4_digit_multiples_of_5 : Nat :=
  let digits := [2, 7, 4, 5]
  let last_digit := 5
  let remaining_digits := [2, 7, 4]
  let case_1 := 3
  let case_2 := 3 * 2
  case_1 + case_2

theorem count_4_digit_multiples_of_5_is_9 : count_4_digit_multiples_of_5 = 9 :=
by
  sorry

end count_4_digit_multiples_of_5_is_9_l787_787293


namespace car_always_leaves_boundary_l787_787042

def car_can_leave_grid (n : ℕ) : Prop :=
  ∀ (start : ℕ × ℕ), start = (n // 2, n // 2) →
  (∃ (steps : list (ℕ × ℕ)), steps.head = start ∧
    (∀ k < steps.length, steps.nth k + (0, 1) = some (steps.nth (k + 1)) ∨ steps.nth k + (1, 0) = some (steps.nth (k + 1))) ∧
    ∀ step ∈ steps, step.1 = 0 ∨ step.2 = 0)

theorem car_always_leaves_boundary (n : ℕ) (h : 0 < n) : car_can_leave_grid n :=
by
  -- Proof is beyond the scope of this transformation and hence skipped
  sorry

end car_always_leaves_boundary_l787_787042


namespace find_coordinates_of_D_l787_787259

theorem find_coordinates_of_D
  (A B C D : ℝ × ℝ)
  (hA : A = (-1, 2))
  (hB : B = (0, 0))
  (hC : C = (1, 7))
  (hParallelogram : ∃ u v, u * (B - A) + v * (C - D) = (0, 0) ∧ u * (C - D) + v * (B - A) = (0, 0)) :
  D = (0, 9) :=
sorry

end find_coordinates_of_D_l787_787259


namespace least_three_digit_digits_product_is_125_l787_787853

def digits_product_three_digit_least (x : Nat) : Prop :=
  100 ≤ x ∧ x < 1000 ∧ (let h := (x / 100); let t := (x / 10) % 10; let u := x % 10 in h * t * u = 10)

theorem least_three_digit_digits_product_is_125 : ∃ x, digits_product_three_digit_least x ∧ ∀ y, digits_product_three_digit_least y → x ≤ y :=
begin
    use 125,
    split,
    { -- Prove that 125 satisfies the condition
      unfold digits_product_three_digit_least,
      split,
      { exact (by norm_num : 100 ≤ 125), },
      split,
      { exact (by norm_num : 125 < 1000), },
      { unfold digits_product_three_digit_least._match_1,
        norm_num, },
    },
    { -- Prove that 125 is the least such number
      intros y h,
      unfold digits_product_three_digit_least at h,
      rcases h with ⟨hy1, hy2, hy3⟩,
      unfold digits_product_three_digit_least._match_1 at hy3,
      interval_cases ((y / 100), (y / 10) % 10, y % 10) using hy3,
      exact le_of_lt (by norm_num : 125 < _),
    },
end

end least_three_digit_digits_product_is_125_l787_787853


namespace monotonic_increasing_interval_l787_787088

open Real

def f (x : ℝ) : ℝ := x^2 * log x

theorem monotonic_increasing_interval :
  ∀ x, x > sqrt e / e → deriv f x > 0 :=
by
  sorry

end monotonic_increasing_interval_l787_787088


namespace K_time_correct_l787_787147

open Real

noncomputable def K_speed : ℝ := sorry
noncomputable def M_speed : ℝ := K_speed - 1 / 2
noncomputable def K_time : ℝ := 45 / K_speed
noncomputable def M_time : ℝ := 45 / M_speed

theorem K_time_correct (K_speed_correct : 45 / K_speed - 45 / M_speed = 1 / 2) : K_time = 45 / K_speed :=
by
  sorry

end K_time_correct_l787_787147


namespace trig_cosine_identity_l787_787136

theorem trig_cosine_identity (z : ℝ) (k l : ℤ) :
  (cos z * cos (2*z) * cos (4*z) * cos (8*z) = 1 / 16) →
  (sin z ≠ 0) →
  (∃ (k : ℤ), z = (2 * real.pi * k) / 15 ∧ (k % 15 ≠ 0) ∨ 
   ∃ (k : ℤ), z = (real.pi * (2 * k + 1)) / 17 ∧ (2 * k + 1) % 17 ≠ 8) :=
sorry

end trig_cosine_identity_l787_787136


namespace coloring_10x10_board_l787_787220

theorem coloring_10x10_board : 
  (∃ f : ℕ × ℕ → bool, 
    (∀ i j, i ≤ 9 → j ≤ 9 → 
      f i j == ff ∨ f i j == tt) ∧ 
    (∀ i j, i ≤ 8 → j ≤ 8 → 
      f i j ≠ f (i+1) j ∧ 
      f i j ≠ f i (j+1) ∧ 
      ∃ k l, k = i + 1 ∧ l = j + 1 ∧ f i j ≠ f k j ∧ f i j ≠ f i l ∧ f i j = f (k) (l))) ↔ true) := sorry

end coloring_10x10_board_l787_787220


namespace complex_number_solution_l787_787692

theorem complex_number_solution (z : ℂ) : (1 + 2 * complex.I) * z = -3 + 4 * complex.I → z = (3 / 5) + (12 / 5) * complex.I :=
by
  sorry

end complex_number_solution_l787_787692


namespace ants_no_collision_probability_l787_787948

-- Definitions
def cube_vertices : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7}

def adjacent (v : ℕ) : Finset ℕ :=
  match v with
  | 0 => {1, 3, 4}
  | 1 => {0, 2, 5}
  | 2 => {1, 3, 6}
  | 3 => {0, 2, 7}
  | 4 => {0, 5, 7}
  | 5 => {1, 4, 6}
  | 6 => {2, 5, 7}
  | 7 => {3, 4, 6}
  | _ => ∅

-- Hypothesis: Each ant moves independently to one of the three adjacent vertices.

-- Result to prove
def X : ℕ := sorry  -- The number of valid ways ants can move without collisions

theorem ants_no_collision_probability : 
  ∃ X, (X / (3 : ℕ)^8 = X / 6561) :=
  by
    sorry

end ants_no_collision_probability_l787_787948


namespace katie_finished_problems_l787_787349

theorem katie_finished_problems (total_problems : ℕ) (problems_left : ℕ) (h1 : total_problems = 9) (h2 : problems_left = 4) : 
  total_problems - problems_left = 5 :=
by
  rw [h1, h2]
  norm_num
  sorry

end katie_finished_problems_l787_787349


namespace option_C_is_quadratic_trinomial_l787_787131

def is_quadratic_trinomial (expr : Prop) : Prop := sorry

def option_A := (a^2 - 3 : ℝ)
def option_B := (3^2 + 3 + 1 : ℝ)
def option_C := (3^2 + a + a * b : ℝ)
def option_D := (x^2 + y^2 + x - y : ℝ)

theorem option_C_is_quadratic_trinomial : is_quadratic_trinomial option_C :=
by
  -- Insert proof here
  sorry

end option_C_is_quadratic_trinomial_l787_787131


namespace expression_closest_to_l787_787813

theorem expression_closest_to (A B C D E : ℝ) (h : 3.25 * 9.252 * (6.22 + 3.78) - 10 = A) :
  A ≈ 282.5 ∧ closest A [250, 300, 350, 400, 450] = 300 := 
by
  -- Using sorry to skip the proof part.
  sorry

end expression_closest_to_l787_787813


namespace count_number_of_ways_to_connect_l787_787747

noncomputable def ways_to_connect_graph (a : List ℕ) (k : ℕ) : ℕ :=
  a.foldl (λ acc x => acc * x) 1 * (a.foldl (λ acc x => acc + x) 0)^(k - 2)

theorem count_number_of_ways_to_connect (G : Type) [simple_graph G] 
  (components : List (Set G)) (k : ℕ)
  (h_component_sizes : List.map Set.card components = List.range_h (λ i, a_i) k) :
  ∃ (a : List ℕ), ways_to_connect_graph a k =
    ∏ i in (List.range k), a_n i * (List.foldr (· + ·) 0 (List.range_h (λ i, a_n i) k))^(k-2) :=
by
  sorry

end count_number_of_ways_to_connect_l787_787747


namespace cos_sum_eighth_power_l787_787568

theorem cos_sum_eighth_power :
  ∑ k in Finset.range 91, (Real.cos (k * Real.pi / 180))^8 = 455 / 32 :=
by
  sorry

end cos_sum_eighth_power_l787_787568


namespace number_of_correct_judgments_l787_787981

theorem number_of_correct_judgments (f : ℝ → ℝ)
  (h_monotone : ∀ x₁ x₂ : ℝ, (x₁ - x₂) * (f x₁ - f x₂) < 0)
  (a b c : ℝ) (h_abc : c < b ∧ b < a) 
  (h_fabc : f a * f b * f c < 0) :
  (∃ d : ℝ, d < c) ∨ (∃ d : ℝ, b < d ∧ d < a) → 
  (¬(∃ d : ℝ, d > a) ∧ ¬(∃ d : ℝ, c < d ∧ d < b)) :=
begin
  sorry
end

end number_of_correct_judgments_l787_787981


namespace sample_mean_equation_l787_787273

variables {ι : Type} [fintype ι]
variables (x : ι → ℝ) (n : ι → ℕ) (C : ℝ) (N : ℕ)
variables (u : ι → ℝ) (u_def : ∀ i, u i = x i - C)

noncomputable def sample_mean := (∑ i, n i * x i) / N

theorem sample_mean_equation (hN : ∑ i, n i = N) :
  sample_mean x n N = C + (∑ i, n i * u i) / N :=
sorry

end sample_mean_equation_l787_787273


namespace max_area_of_garden_l787_787514

theorem max_area_of_garden
  (w : ℕ) (l : ℕ)
  (h1 : l = 2 * w)
  (h2 : l + 2 * w = 480) : l * w = 28800 :=
sorry

end max_area_of_garden_l787_787514


namespace fraction_meaningful_l787_787100

-- Define the condition for the fraction being meaningful
def denominator_not_zero (x : ℝ) : Prop := x + 1 ≠ 0

-- Define the statement to be proved
theorem fraction_meaningful (x : ℝ) : denominator_not_zero x ↔ x ≠ -1 :=
by
  sorry

end fraction_meaningful_l787_787100


namespace quadratic_properties_l787_787810

-- Define the quadratic function
def quadratic_function : ℝ → ℝ :=
  λ x, (1/2) * (x + 4)^2 + 5

-- The direction of the opening of a quadratic function is determined by the leading coefficient
def opening_direction (a : ℝ) : Prop :=
  a > 0

-- Define the axis of symmetry for the quadratic function y = a(x-h)^2 + k
def axis_of_symmetry (h : ℝ) : ℝ × ℝ :=
  (-h, 0)

-- Define the vertex for the quadratic function y = a(x-h)^2 + k
def vertex (h k : ℝ) : ℝ × ℝ :=
  (h, k)
  
theorem quadratic_properties :
  (quadratic_function x = (1/2) * (x + 4)^2 + 5) →
  opening_direction (1/2) ∧
  axis_of_symmetry 4 = (-4, 0) ∧
  vertex (-4) 5 = (-4, 5) :=
by
  sorry

end quadratic_properties_l787_787810


namespace sum_of_solutions_eq_sqrt3_l787_787366

theorem sum_of_solutions_eq_sqrt3 :
  (∑ (x : ℝ) in {x | 0 < x ∧ x ^ 3 ^ real.sqrt 3 = real.sqrt 3 ^ 3 ^ x}, x) = real.sqrt 3 :=
by
  sorry

end sum_of_solutions_eq_sqrt3_l787_787366


namespace total_trip_length_l787_787396

theorem total_trip_length :
  ∀ (d : ℝ), 
    (∀ fuel_per_mile : ℝ, fuel_per_mile = 0.03 →
      ∀ battery_miles : ℝ, battery_miles = 50 →
      ∀ avg_miles_per_gallon : ℝ, avg_miles_per_gallon = 50 →
      (d / (fuel_per_mile * (d - battery_miles))) = avg_miles_per_gallon →
      d = 150) := 
by
  intros d fuel_per_mile fuel_per_mile_eq battery_miles battery_miles_eq avg_miles_per_gallon avg_miles_per_gallon_eq trip_condition
  sorry

end total_trip_length_l787_787396


namespace polygon_area_l787_787334

-- Definitions for the problem conditions
structure Polygon where
  sides: ℕ
  side_length: ℝ
  perimeter: ℝ
  perpendicular_sides: sides % 4 = 0

-- The specific polygon in the problem
def given_polygon : Polygon :=
{ sides := 32,
  side_length := 2,
  perimeter := 64,
  perpendicular_sides := by norm_num }

-- The theorem to prove
theorem polygon_area (p : Polygon) (h_sides : p.sides = 32)
  (h_side_length : p.side_length = 2) (h_perimeter : p.perimeter = 64) :
  p.perimeter = p.sides * p.side_length →
  (h_sides * h_side_length)^2 * 36 = 144 := sorry

end polygon_area_l787_787334


namespace Collatz_8th_term_1_unique_values_count_l787_787312

theorem Collatz_8th_term_1_unique_values_count :
    {n : ℕ // n > 0 ∧ collatz 8 n = 1}.card = 6 :=
begin
  sorry -- Proof would go here
end

end Collatz_8th_term_1_unique_values_count_l787_787312


namespace canoe_total_weight_calculation_canoe_maximum_weight_limit_l787_787772

def canoe_max_people : ℕ := 8
def people_with_pets_ratio : ℚ := 3 / 4
def adult_weight : ℚ := 150
def child_weight : ℚ := adult_weight / 2
def dog_weight : ℚ := adult_weight / 3
def cat1_weight : ℚ := adult_weight / 10
def cat2_weight : ℚ := adult_weight / 8

def canoe_capacity_with_pets : ℚ := people_with_pets_ratio * canoe_max_people

def total_weight_adults_and_children : ℚ := 4 * adult_weight + 2 * child_weight
def total_weight_pets : ℚ := dog_weight + cat1_weight + cat2_weight
def total_weight : ℚ := total_weight_adults_and_children + total_weight_pets

def max_weight_limit : ℚ := canoe_max_people * adult_weight

theorem canoe_total_weight_calculation :
  total_weight = 833 + 3 / 4 := by
  sorry

theorem canoe_maximum_weight_limit :
  max_weight_limit = 1200 := by
  sorry

end canoe_total_weight_calculation_canoe_maximum_weight_limit_l787_787772


namespace find_x_when_parallel_l787_787003

-- Given vectors
def a : ℝ × ℝ := (-2, 4)
def b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Conditional statement: parallel vectors
def parallel_vectors (u v : ℝ × ℝ) : Prop := 
  u.1 * v.2 = u.2 * v.1

-- Proof statement
theorem find_x_when_parallel (x : ℝ) (h : parallel_vectors a (b x)) : x = 1 := 
  sorry

end find_x_when_parallel_l787_787003


namespace equal_distances_in_acute_triangle_l787_787715

theorem equal_distances_in_acute_triangle
  (ABC : Triangle)
  (A B C : Point)
  (angle_BAC : ∠BAC = 60)
  (I O H : Point)
  (hI : I = incenter ABC)
  (hO : O = circumcenter ABC)
  (hH : H = orthocenter ABC) :
  dist O I = dist I H :=
by sorry

end equal_distances_in_acute_triangle_l787_787715


namespace marbles_total_is_260_l787_787216

/-- Define the number of marbles in each jar. -/
def jar1 : ℕ := 80
def jar2 : ℕ := 2 * jar1
def jar3 : ℕ := jar1 / 4

/-- The total number of marbles Courtney has. -/
def total_marbles : ℕ := jar1 + jar2 + jar3

/-- Proving that the total number of marbles is 260. -/
theorem marbles_total_is_260 : total_marbles = 260 := 
by
  sorry

end marbles_total_is_260_l787_787216


namespace maximize_greenhouse_planting_area_l787_787540

theorem maximize_greenhouse_planting_area
    (a b : ℝ)
    (h : a * b = 800)
    (planting_area : ℝ := (a - 4) * (b - 2)) :
  (a = 40 ∧ b = 20) ↔ planting_area = 648 :=
by
  sorry

end maximize_greenhouse_planting_area_l787_787540


namespace triangle_area_of_incircle_l787_787151

/-- Given a circle ω₁ with radius 1 and diameter AB, and another circle ω₂ with maximum radius such that it is tangent to AB and internally tangent to ω₁, if a point C is chosen such that ω₂ is the incircle of triangle ABC, then the area of triangle ABC is 3/2. -/
theorem triangle_area_of_incircle 
    (ω₁ ω₂ : Circle) 
    (radius_ω₁ : ω₁.radius = 1) 
    (diam_ω₁ : ω₁.diameter = segment A B) 
    (tangent_γ1 : ω₂.radius = 1/2) 
    (incircle_cond : IsIncircle ω₂ (triangle ABC)) : 
    area (triangle ABC) = 3/2 :=
sorry

end triangle_area_of_incircle_l787_787151


namespace arithmetic_sequence_fifth_term_l787_787440

theorem arithmetic_sequence_fifth_term :
  ∀ (a d : ℤ), (a + 19 * d = 15) → (a + 20 * d = 18) → (a + 4 * d = -30) :=
by
  intros a d h1 h2
  sorry

end arithmetic_sequence_fifth_term_l787_787440


namespace evaluate_f_f_neg4_l787_787276

def f (x : ℝ) : ℝ :=
  if x > 0 then real.sqrt x else 2 ^ (-x)

theorem evaluate_f_f_neg4 : f (f (-4)) = 4 := by
  sorry

end evaluate_f_f_neg4_l787_787276


namespace muffin_cost_relation_l787_787069

variable (m b : ℝ)

variable (S := 5 * m + 4 * b)
variable (C := 10 * m + 18 * b)

theorem muffin_cost_relation (h1 : C = 3 * S) : m = 1.2 * b :=
  sorry

end muffin_cost_relation_l787_787069


namespace salesman_l787_787902

/-- A salesman bought a case of 48 backpacks for $576. He sold 17 of them for $18 each, 
10 were sold to a department store for $25 each, and the remainder were sold for $22 each.
We need to prove that the salesman's profit is $442. -/
theorem salesman's_profit :
  let cost := 576 in
  let sold_first_17_price := 17 * 18 in
  let sold_next_10_price := 10 * 25 in
  let remaining_bags := 48 - 27 in
  let sold_remaining_price := remaining_bags * 22 in
  let total_sales := sold_first_17_price + sold_next_10_price + sold_remaining_price in
  let profit := total_sales - cost in
  profit = 442 :=
by
  sorry

end salesman_l787_787902


namespace perfect_square_trinomial_m_l787_787686

theorem perfect_square_trinomial_m (m : ℤ) : (∃ (a : ℤ), (x : ℝ) → x^2 + m * x + 9 = (x + a)^2) → (m = 6 ∨ m = -6) :=
sorry

end perfect_square_trinomial_m_l787_787686


namespace find_2a_minus_b_l787_787404

-- Define conditions
def f (x : ℝ) (a b : ℝ) := a * x + b
def g (x : ℝ) := -5 * x + 7
def h (x : ℝ) (a b : ℝ) := f (g x) a b
def h_inv (x : ℝ) := x - 9

-- Statement to prove
theorem find_2a_minus_b (a b : ℝ) 
(h_eq : ∀ x, h x a b = a * (-5 * x + 7) + b)
(h_inv_eq : ∀ x, h_inv x = x - 9)
(h_hinv_eq : ∀ x, h (h_inv x) a b = x) :
  2 * a - b = -54 / 5 := sorry

end find_2a_minus_b_l787_787404


namespace can_all_mushrooms_become_good_l787_787165

def is_bad (w : Nat) : Prop := w ≥ 10
def is_good (w : Nat) : Prop := w < 10

def mushrooms_initially_bad := 90
def mushrooms_initially_good := 10

def total_mushrooms := mushrooms_initially_bad + mushrooms_initially_good
def total_worms_initial := mushrooms_initially_bad * 10

theorem can_all_mushrooms_become_good :
  ∃ worms_distribution : Fin total_mushrooms → Nat,
  (∀ i : Fin total_mushrooms, is_good (worms_distribution i)) :=
sorry

end can_all_mushrooms_become_good_l787_787165


namespace area_of_triangle_ABC_l787_787374

variable (A B C D E G : Point)
variable (tri : Triangle A B C)
variable (med1 : Median tri D A)
variable (med2 : Median tri E B)
variable (angle_med1_med2 : Angle med1 med2 = 45)
variable (len_AD : Length med1 D A = 18)
variable (len_BE : Length med2 E B = 24)

theorem area_of_triangle_ABC :
  Area tri = 144 * sqrt 2 :=
sorry

end area_of_triangle_ABC_l787_787374


namespace find_g_minus_one_l787_787406

noncomputable def g (x : ℝ) : ℝ := -- Provide the expression for g here later 

theorem find_g_minus_one :
  ∀ (g : ℝ → ℝ),
  (∀ x, ∃ c d, g(x) = c * x + d) ∧
  (∀ x, g(x) = 3 * (g⁻¹ x) + 5) ∧
  g(0) = 3 →
  g(-1) = 3 - Real.sqrt 3 :=
by
  sorry

end find_g_minus_one_l787_787406


namespace distinct_configurations_20x16_grid_l787_787915

theorem distinct_configurations_20x16_grid 
  (rows : ℕ) (cols : ℕ) (total_switches : ℕ) 
  (initial_state : (ℕ × ℕ) → bool)
  (toggle_row : ℕ → ((ℕ × ℕ) → bool) → (ℕ × ℕ) → bool)
  (toggle_col : ℕ → ((ℕ × ℕ) → bool) → (ℕ × ℕ) → bool) :
  rows = 20 → cols = 16 → total_switches = 36 →
  (∀ state, ∀ r, r < rows → toggle_row r state = (λ (pos : ℕ × ℕ), if pos.1 = r then ¬ state pos else state pos)) →
  (∀ state, ∀ c, c < cols → toggle_col c state = (λ (pos : ℕ × ℕ), if pos.2 = c then ¬ state pos else state pos)) →
  (set.count (λ seq, (seq.count = total_switches ∧ ∀ pos : set (ℕ × ℕ), seq.inter (flip_row_toggle_row ∪ flip_col_toggle_col) state (rows, cols) = seq.state.init)) (2 ^ 35) :=
by 
  intros;
  sorry

end distinct_configurations_20x16_grid_l787_787915


namespace seniors_in_statistics_l787_787039

theorem seniors_in_statistics (total_students : ℕ) (half_stats : total_students / 2 = 60) (percent_seniors : 90 / 100 = 0.9) :
  let students_in_stats := total_students / 2
  let seniors_in_stats := 0.9 * students_in_stats in
  seniors_in_stats = 54 :=
by
  sorry

end seniors_in_statistics_l787_787039


namespace min_sum_of_first_12_terms_l787_787709

variable {a : ℕ → ℝ}

theorem min_sum_of_first_12_terms (h1 : ∀ n, 0 < a n) (h2 : a 4 * a 9 = 36) :
  ∑ i in finset.range 12, a (i + 1) ≥ 72 :=
by
  sorry

end min_sum_of_first_12_terms_l787_787709


namespace Kolya_can_form_triangles_l787_787110

theorem Kolya_can_form_triangles :
  ∃ (K1a K1b K1c K3a K3b K3c V1 V2 V3 : ℝ), 
  (K1a + K1b + K1c = 1) ∧
  (K3a + K3b + K3c = 1) ∧
  (V1 + V2 + V3 = 1) ∧
  (K1a = 0.5) ∧ (K1b = 0.25) ∧ (K1c = 0.25) ∧
  (K3a = 0.5) ∧ (K3b = 0.25) ∧ (K3c = 0.25) ∧
  (∀ (V1 V2 V3 : ℝ), V1 + V2 + V3 = 1 → 
  (
    (K1a + V1 > K3b ∧ K1a + K3b > V1 ∧ V1 + K3b > K1a) ∧ 
    (K1b + V2 > K3a ∧ K1b + K3a > V2 ∧ V2 + K3a > K1b) ∧ 
    (K1c + V3 > K3c ∧ K1c + K3c > V3 ∧ V3 + K3c > K1c)
  )) :=
sorry

end Kolya_can_form_triangles_l787_787110


namespace simplify_expr_l787_787970

variable {x y : ℝ}

theorem simplify_expr (hx : x ≠ 0) (hy : y ≠ 0) :
  ((x^3 + 1) / x) * ((y^3 + 1) / y) - ((x^3 - 1) / y) * ((y^3 - 1) / x) = 2 * x^2 + 2 * y^2 :=
by sorry

end simplify_expr_l787_787970


namespace count_non_self_intersecting_paths_l787_787388

theorem count_non_self_intersecting_paths : 
  let vertices := (A₁ A₂ A₃ A₄ A₅ A₆ A₇ A₈ A₉ A₁₀ A₁₁ A₁₂ : Fin 12)
  in ∃ (paths : Fin 2^10), paths = 1024 :=
begin
  sorry
end

end count_non_self_intersecting_paths_l787_787388


namespace max_abs_asin_b_l787_787975

theorem max_abs_asin_b (a b c : ℝ) (h : ∀ x : ℝ, |a * (Real.cos x)^2 + b * Real.sin x + c| ≤ 1) :
  ∃ M : ℝ, (∀ x : ℝ, |a * Real.sin x + b| ≤ M) ∧ M = 2 :=
sorry

end max_abs_asin_b_l787_787975


namespace total_albums_l787_787037

theorem total_albums (Adele Bridget Katrina Miriam : ℕ) 
  (h₁ : Adele = 30)
  (h₂ : Bridget = Adele - 15)
  (h₃ : Katrina = 6 * Bridget)
  (h₄ : Miriam = 5 * Katrina) : Adele + Bridget + Katrina + Miriam = 585 := 
by
  sorry

end total_albums_l787_787037


namespace g_value_l787_787368

theorem g_value {
  (g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, g (x * g y + 2 * x) = 2 * x * y + g x)
:
  let n := 2
  let s := 0
  n * s = 0 :=
by
  let n := 2
  let s := 0
  exact (n * s = 0)

end g_value_l787_787368


namespace length_of_pond_l787_787084

-- Define the problem conditions
variables (W L S : ℝ)
variables (h1 : L = 2 * W) (h2 : L = 24) 
variables (A_field A_pond : ℝ)
variables (h3 : A_pond = 1 / 8 * A_field)

-- State the theorem
theorem length_of_pond :
  A_field = L * W ∧ A_pond = S^2 ∧ A_pond = 1 / 8 * A_field ∧ L = 24 ∧ L = 2 * W → 
  S = 6 :=
by
  sorry

end length_of_pond_l787_787084


namespace soda_left_is_70_percent_l787_787577

def bottles_left_percentage (initial_bottles: ℕ) (drank_percentage: ℝ) (given_percentage: ℝ) : ℝ :=
  let drank_amount := 1 * drank_percentage
  let given_amount := 2 * given_percentage
  let remaining_bottles := initial_bottles - drank_amount - given_amount
  remaining_bottles * 100

theorem soda_left_is_70_percent :
  ∀ (initial_bottles: ℕ) (drank_percentage: ℝ) (given_percentage: ℝ),
    initial_bottles = 3 → drank_percentage = 0.9 → given_percentage = 0.7 →
    bottles_left_percentage initial_bottles drank_percentage given_percentage = 70 :=
begin
  intros initial_bottles drank_percentage given_percentage h1 h2 h3,
  simp [bottles_left_percentage, h1, h2, h3],
  norm_num,
end

end soda_left_is_70_percent_l787_787577


namespace convert_base7_to_base2_l787_787572

-- Definitions and conditions
def base7_to_decimal (n : ℕ) : ℕ :=
  2 * 7^1 + 5 * 7^0

def decimal_to_binary (n : ℕ) : ℕ :=
  -- Reversing the binary conversion steps
  -- 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 19
  1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

-- Proof problem
theorem convert_base7_to_base2 : decimal_to_binary (base7_to_decimal 25) = 10011 :=
by {
  sorry
}

end convert_base7_to_base2_l787_787572


namespace number_of_committees_correct_l787_787164

noncomputable def number_of_committees (teams members host_selection non_host_selection : ℕ) : ℕ :=
  have ways_to_choose_host := teams
  have ways_to_choose_four_from_seven := Nat.choose members host_selection
  have ways_to_choose_two_from_seven := Nat.choose members non_host_selection
  have total_non_host_combinations := ways_to_choose_two_from_seven ^ (teams - 1)
  ways_to_choose_host * ways_to_choose_four_from_seven * total_non_host_combinations

theorem number_of_committees_correct :
  number_of_committees 5 7 4 2 = 34134175 := by
  sorry

end number_of_committees_correct_l787_787164


namespace minor_premise_is_T₁_l787_787943

-- Definitions based on conditions
def R : Prop := "A rectangle is a parallelogram"
def T₁ : Prop := "A triangle is not a parallelogram"
def T₂ : Prop := "A triangle is not a rectangle"

-- Statement proving that the minor premise is T₁
theorem minor_premise_is_T₁ : T₁ :=
sorry

end minor_premise_is_T₁_l787_787943


namespace problem1_problem2_l787_787264

open Real

theorem problem1 (a θ : ℝ) (h : (λ x : ℝ, x^2 - a * x + a) = 0) :
  sin θ^3 + cos θ^3 = sqrt 2 - 2 := sorry

theorem problem2 (a θ : ℝ) (h : (λ x : ℝ, x^2 - a * x + a) = 0) :
  tan θ + 1 / tan θ = -1 - sqrt 2 := sorry

end problem1_problem2_l787_787264


namespace combined_average_marks_l787_787905

theorem combined_average_marks :
  let class_A_students := 30
  let class_B_students := 50
  let class_C_students := 40
  let class_D_students := 60
  let class_A_avg_marks := 40
  let class_B_avg_marks := 90
  let class_C_avg_marks := 70
  let class_D_avg_marks := 80
  let total_students := class_A_students + class_B_students + class_C_students + class_D_students
  let total_marks := (class_A_students * class_A_avg_marks) + 
                     (class_B_students * class_B_avg_marks) + 
                     (class_C_students * class_C_avg_marks) + 
                     (class_D_students * class_D_avg_marks)
  let combined_avg_marks := total_marks / total_students.toFloat
  combined_avg_marks = 73.89 := sorry

end combined_average_marks_l787_787905


namespace interval_increase_f_l787_787583

noncomputable def f (x : ℝ) : ℝ := Real.log (-(x^2) + 2 * x + 3) / Real.log 0.5

theorem interval_increase_f : 
  ∀ x ∈ Set.Ico (-1 : ℝ) 3, 
    Deriv f x > 0 → x ∈ Set.Ioo 1 3 :=
by
  sorry

end interval_increase_f_l787_787583


namespace length_n_l787_787015

def smallest_non_factor (n : ℕ) : ℕ :=
  if n % 2 ≠ 0 then 2 else
  if n % 3 ≠ 0 then 3 else
  if n % 5 ≠ 0 then 5 else
  7 -- Simplified; the actual function would need to explore more factors.

def length (n : ℕ) : ℕ :=
  if n % 2 ≠ 0 then 1 else
    let a := (Nat.find (λ a => n = 2^a * (2 * (n / 2^a) + 1))) in
    if (λ t, (1 < t ∧ t < 2^(a+1)) → n % t = 0) ∀ (t : ℕ) then 3 else
    2

theorem length_n (n : ℕ) (h : n ≥ 3): length n = 
  if n % 2 ≠ 0 then 1 else
  if let a := Nat.find (λ a => n = 2^a * (2 * (n / 2^a) + 1)) in
    (λ t, (1 < t ∧ t < 2^(a+1)) → n % t = 0) ∀ t 
    then 3 else 2 := sorry

end length_n_l787_787015


namespace sufficient_condition_not_necessary_condition_l787_787629

variable (λ : ℝ) (n : ℕ)

def sequence (λ : ℝ) (n : ℕ) : ℝ := n^2 - 2 * λ * n

theorem sufficient_condition (h : λ < 1) : ∀ n : ℕ, 
  sequence λ (n + 1) > sequence λ n := by {
  sorry
}

theorem not_necessary_condition (h₁ : ∀ n : ℕ, sequence λ (n + 1) > sequence λ n) : 
  ¬ (∀ λ : ℝ, ∀ n : ℕ, sequence λ (n + 1) > sequence λ n → λ < 1) := by {
  sorry
}

end sufficient_condition_not_necessary_condition_l787_787629


namespace combined_total_pets_l787_787801

structure People := 
  (dogs : ℕ)
  (cats : ℕ)

def Teddy : People := {dogs := 7, cats := 8}
def Ben : People := {dogs := 7 + 9, cats := 0}
def Dave : People := {dogs := 7 - 5, cats := 8 + 13}

def total_pets (p : People) : ℕ := p.dogs + p.cats

theorem combined_total_pets : 
  total_pets Teddy + total_pets Ben + total_pets Dave = 54 := by
  sorry

end combined_total_pets_l787_787801


namespace min_edges_bound_l787_787986

-- Define the given conditions for the graph G
variables {V : Type*} [fintype V] [decidable_eq V]
variables (G : simple_graph V) (n : ℕ)

-- Given conditions
variable (h1 : fintype.card V = n)
variable (h2 : n ≥ 4)
variable (h3 : ∀ v w : V, v ≠ w → ∃ u : V, G.adj u v ∧ G.adj u w)

-- Define the goal statement for the smallest possible number of edges
def min_edges (G : simple_graph V) := ∃ e, e = finset.card G.edge_set

theorem min_edges_bound (G : simple_graph V)
  (h1 : fintype.card V = n) (h2 : n ≥ 4) (h3 : ∀ v w : V, v ≠ w → ∃ u : V, G.adj u v ∧ G.adj u w) :
  min_edges G = (3 * (n-1)) / 2 :=
sorry

end min_edges_bound_l787_787986


namespace weekly_sales_correct_target_profit_correct_max_profit_correct_l787_787452

-- Definitions from conditions
def cost_price : ℕ := 80
def init_selling_price : ℕ := 90
def init_sales_weekly : ℕ := 600
def price_increase_effect : ℕ := 10
def max_selling_price : ℕ := 110

-- Problem 1
def weekly_sales_function (x : ℕ) : ℕ := -10 * x + 1500

-- Problem 2
def target_profit : ℕ := 8250
def selling_price_for_target_profit (x : ℕ) : Prop :=
    let sales := weekly_sales_function x in
    (x - cost_price) * sales = target_profit

-- Problem 3
def profit_function (x : ℕ) : ℕ :=
    (x - cost_price) * weekly_sales_function x

-- Statement for maximized profit given max_selling_price constraint
def max_profit_price : ℕ := 110
def max_profit_value : ℕ := 12000

-- Equivalent proof statements
theorem weekly_sales_correct :
  ∀ x : ℕ, weekly_sales_function x = -10 * x + 1500 :=
sorry

theorem target_profit_correct :
  selling_price_for_target_profit 95 :=
sorry

theorem max_profit_correct :
  (profit_function max_profit_price = max_profit_value) ∧ 
  (∀ (x : ℕ), x ≤ max_selling_price → profit_function x ≤ max_profit_value) :=
sorry

end weekly_sales_correct_target_profit_correct_max_profit_correct_l787_787452


namespace probability_of_diff_ge_3_l787_787459

noncomputable def probability_diff_ge_3 : ℚ :=
  let elements := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let total_pairs := (elements.to_finset.card.choose 2 : ℕ)
  let valid_pairs := (total_pairs - 15 : ℕ) -- 15 pairs with a difference of less than 3
  valid_pairs / total_pairs

theorem probability_of_diff_ge_3 :
  probability_diff_ge_3 = 7 / 12 := by
  sorry

end probability_of_diff_ge_3_l787_787459


namespace ratio_of_elements_l787_787935

theorem ratio_of_elements (total_weight : ℕ) (element_B_weight : ℕ) 
  (h_total : total_weight = 324) (h_B : element_B_weight = 270) :
  (total_weight - element_B_weight) / element_B_weight = 1 / 5 :=
by
  sorry

end ratio_of_elements_l787_787935


namespace toothpicks_in_20th_stage_l787_787449

theorem toothpicks_in_20th_stage :
  (3 + 3 * (20 - 1) = 60) :=
by
  sorry

end toothpicks_in_20th_stage_l787_787449


namespace greatest_positive_integer_N_l787_787600

def condition (x : Int) (y : Int) : Prop :=
  (x^2 - x * y) % 1111 ≠ 0

theorem greatest_positive_integer_N :
  ∃ N : Nat, (∀ (x : Fin N) (y : Fin N), x ≠ y → condition x y) ∧ N = 1000 :=
by
  sorry

end greatest_positive_integer_N_l787_787600


namespace exists_2005_distinct_numbers_l787_787225

theorem exists_2005_distinct_numbers :
  ∃ (S : Finset ℕ), S.card = 2005 ∧ 
  ∀ x ∈ S, (∑ y in (S.erase x), y) % x = 0 :=
sorry

end exists_2005_distinct_numbers_l787_787225


namespace solve_problem_l787_787385

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem solve_problem :
  is_prime 2017 :=
by
  have h1 : 2017 > 1 := by linarith
  have h2 : ∀ m : ℕ, m ∣ 2017 → m = 1 ∨ m = 2017 :=
    sorry
  exact ⟨h1, h2⟩

end solve_problem_l787_787385


namespace minimal_value_of_f_l787_787472

noncomputable def f (x y z : ℝ) : ℝ :=  Real.sqrt(2 * x + 1) + Real.sqrt(3 * y + 1) + Real.sqrt(4 * z + 1)

theorem minimal_value_of_f :
  ∃ (x y z : ℝ), x + y + z = 4 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  f x y z = Real.sqrt (61 / 27) + Real.sqrt (183 / 36) + Real.sqrt (976 / 108) :=
sorry

end minimal_value_of_f_l787_787472


namespace equilateral_triangle_side_length_l787_787720

theorem equilateral_triangle_side_length
  (A B C : Type)
  (AB : ℝ)
  (h1: ∠A = 60)
  (h2: ∠B = 60)
  (h3: AB = 10) 
  : BC = 10 := 
by
  sorry

end equilateral_triangle_side_length_l787_787720


namespace convex_quadrilateral_from_five_points_l787_787861

-- Definition that establishes the condition of no three points being collinear
structure NonCollinear5Points (A B C D E : Point) : Prop :=
(leftmost_exists : ∃ P : Point, ∀ Q : Point, (P ≠ Q) ∧ (Q ≠ P → ¬Collinear P Q))

theorem convex_quadrilateral_from_five_points (A B C D E : Point) 
    (hA : NonCollinear A B C D E) :
    ∃ P1 P2 P3 P4, IsConvexQuadrilateral P1 P2 P3 P4 := 
sorry

end convex_quadrilateral_from_five_points_l787_787861


namespace rect_prism_diag_l787_787525

theorem rect_prism_diag (l w h : ℝ) (h_l : l = 12) (h_w : w = 15) (h_h : h = 8) :
  (l^2 + w^2 + h^2 = 433) → (∀ d : ℝ, d = Real.sqrt (l^2 + w^2 + h^2) → d = Real.sqrt 433) :=
by
  intros h1 h2
  rw [h1, h2]
  rfl
  sorry

end rect_prism_diag_l787_787525


namespace area_of_efgh_is_144_l787_787553

-- Given an original square with side length 6 units
def original_square_side_length : ℝ := 6

-- Given a semicircle around each side of the square, each with a radius of half the side length of the original square
def semicircle_radius (side_length : ℝ) := side_length / 2

-- Define a square EFGH that is tangent to each of the four semicircles
noncomputable def efgh_side_length (original_square_side_length : ℝ) := original_square_side_length + 2 * semicircle_radius(original_square_side_length)

-- Define the area of square EFGH
noncomputable def efgh_area (side_length : ℝ) := side_length^2

-- Prove that the area of square EFGH is 144 square units
theorem area_of_efgh_is_144 : efgh_area (efgh_side_length original_square_side_length) = 144 :=
by
  -- This is a placeholder for the proof
  sorry

end area_of_efgh_is_144_l787_787553


namespace segment_AC_length_l787_787564

-- Define the circle with a given circumference and identify points
variables (C : Type*) [metric_space C] (O A B : C)
variables (r : ℝ) (c : real.pi * 2 * r = 18 * real.pi) (AB_diameter : dist A B = 2 * r)
variables (angle_CAB_60 : real.angle O A B = real.pi / 3)

-- Use the distance function in metric space to assert AC length
theorem segment_AC_length : dist A O = 9 :=
  sorry

end segment_AC_length_l787_787564


namespace zero_function_l787_787367

noncomputable def f : ℝ → ℝ := sorry -- Let it be a placeholder for now.

theorem zero_function (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b))
  (h_int : ∀ n : ℕ, ∫ x in a..b, (x : ℝ)^n * f x = 0) :
  ∀ x ∈ Set.Icc a b, f x = 0 :=
by
  sorry -- placeholder for the proof

end zero_function_l787_787367


namespace a10_contains_at_least_1000_nines_l787_787103

def sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 9 ∧ ∀ k, a (k + 1) = 3 * (a k)^4 + 4 * (a k)^3

def has_at_least_1000_nines (n : ℕ) : Prop :=
  let num_digits := n.to_digits 10
  num_digits.count 9 >= 1000

theorem a10_contains_at_least_1000_nines :
  ∃ a : ℕ → ℕ, sequence a ∧ has_at_least_1000_nines (a 10) :=
by
  sorry

end a10_contains_at_least_1000_nines_l787_787103


namespace variance_transformed_sample_l787_787271

variable {α : Type} [AddCommGroup α] [Module ℝ α] [MetricSpace α] [NormedGroup α]

noncomputable def variance (s : Finset α) : ℝ :=
  s.sum (λ x => (norm x) ^ 2) / s.card.toReal - (norm (s.sum (λ x => x) / s.card.toReal)) ^ 2

variables (a1 a2 a3 : ℝ) (a : ℝ)
hypothesis (var_a : variance ({a1, a2, a3} : Finset ℝ) = a) : ℝ

theorem variance_transformed_sample :
  variance ({3 * a1 + 1, 3 * a2 + 1, 3 * a3 + 1} : Finset ℝ) = 9 * a := by
  sorry

end variance_transformed_sample_l787_787271


namespace log2_real_coeff_sum_l787_787020

open Complex Nat

theorem log2_real_coeff_sum 
    (T : ℝ)
    (hT : T = (∑ k in Finset.range (2012), if even k then (↑((choose 2011 k : ℕ) * (I ^ k)) * (1 ^ k)).re else 0)) :
    log 2 T = 1005 :=
sorry

end log2_real_coeff_sum_l787_787020


namespace instantaneous_velocity_at_10_l787_787418
open Real  

def displacement (t : ℝ) : ℝ := 3 * t ^ 2 - 2 * t + 1

theorem instantaneous_velocity_at_10 : 
  let v (t : ℝ) := deriv displacement t in
  v 10 = 58 :=
by
  sorry

end instantaneous_velocity_at_10_l787_787418


namespace term_2_6_position_l787_787043

theorem term_2_6_position : 
  ∃ (seq : ℕ → ℚ), 
    (seq 23 = 2 / 6) ∧ 
    (∀ n, ∃ k, (n = (k * (k + 1)) / 2 ∧ k > 0 ∧ k <= n)) :=
by sorry

end term_2_6_position_l787_787043


namespace hockey_team_ties_l787_787668

theorem hockey_team_ties (W T : ℕ) (h1 : 2 * W + T = 60) (h2 : W = T + 12) : T = 12 :=
by
  sorry

end hockey_team_ties_l787_787668


namespace problem_statement_l787_787269

-- Define the function f(x) with the given conditions
def f (ω x : ℝ) : ℝ := cos (ω * x - ω * π / 6)

-- Define the function g(x) as cos(2x)
def g (x : ℝ) : ℝ := cos (2 * x)

-- Define the condition that the smallest positive period of f(x) is π
def smallest_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x ∧ T > 0 ∧ ∀ T', 0 < T' < T → ¬ (∀ x, f(x + T') = f x)

-- Define the correctness condition
def is_shift_right_by (f g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x = g (x - a)

-- The theorem to prove
theorem problem_statement (ω : ℝ) (hω : ω > 0) (hT : smallest_period (f ω) π) :
  ω = 2 ∧ is_shift_right_by (f 2) g (π / 6) :=
by
    sorry

end problem_statement_l787_787269


namespace min_value_of_a_squared_l787_787708

variables {A B C a b c : ℝ}

-- Given conditions
axiom acute_angled_triangle : A > 0 ∧ A < π / 2 ∧ B > 0 ∧ B < π / 2 ∧ C > 0 ∧ C < π / 2 ∧ (A + B + C = π)
axiom sides_opposite_angles : (a^2 + b^2 - c^2 = 2 * a * b * real.cos C) ∧ (b^2 + c^2 - a^2 = 2 * b * c * real.cos A) ∧ (c^2 + a^2 - b^2 = 2 * c * a * real.cos B)
axiom given_sine_relation : b^2 * real.sin C = 4 * real.sqrt 2 * real.sin B
axiom given_area : (1 / 2) * b * c * real.sin A = 8 / 3

-- Objective: Prove the minimum value of a^2
theorem min_value_of_a_squared (h_acute : acute_angled_triangle) (h_sides : sides_opposite_angles) (h_sine_rel : given_sine_relation) (h_area : given_area) :
  ∃ a, a^2 = 16 * real.sqrt 2 / 3 :=
sorry

end min_value_of_a_squared_l787_787708


namespace reciprocal_of_neg_two_thirds_l787_787431

-- Definition for finding the reciprocal
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The proof problem statement
theorem reciprocal_of_neg_two_thirds : reciprocal (-2 / 3) = -3 / 2 :=
sorry

end reciprocal_of_neg_two_thirds_l787_787431


namespace unpainted_unit_cubes_l787_787501

theorem unpainted_unit_cubes : 
  let total_cubes := 64 in
  let painted_per_face := 4 in
  let faces := 6 in
  let corners := 8 in
  let total_painted := painted_per_face * faces in
  let total_corner_adjustment := corners * 2 in
  let actual_painted := total_painted - total_corner_adjustment in
  total_cubes - actual_painted = 56 := 
by
  sorry

end unpainted_unit_cubes_l787_787501


namespace salesman_l787_787901

/-- A salesman bought a case of 48 backpacks for $576. He sold 17 of them for $18 each, 
10 were sold to a department store for $25 each, and the remainder were sold for $22 each.
We need to prove that the salesman's profit is $442. -/
theorem salesman's_profit :
  let cost := 576 in
  let sold_first_17_price := 17 * 18 in
  let sold_next_10_price := 10 * 25 in
  let remaining_bags := 48 - 27 in
  let sold_remaining_price := remaining_bags * 22 in
  let total_sales := sold_first_17_price + sold_next_10_price + sold_remaining_price in
  let profit := total_sales - cost in
  profit = 442 :=
by
  sorry

end salesman_l787_787901


namespace probability_no_rain_five_days_probability_drought_alert_approx_l787_787428

theorem probability_no_rain_five_days (p : ℚ) (h : p = 1/3) :
  (p ^ 5) = 1 / 243 :=
by
  -- Add assumptions and proceed
  sorry

theorem probability_drought_alert_approx (p : ℚ) (h : p = 1/3) :
  4 * (p ^ 2) = 4 / 9 :=
by
  -- Add assumptions and proceed
  sorry

end probability_no_rain_five_days_probability_drought_alert_approx_l787_787428


namespace seven_lines_angle_less_than_26_l787_787979

/-- 
Given 7 lines on a plane with none of them being parallel, 
prove that there exist two lines with an angle between them 
less than 26 degrees.
-/
theorem seven_lines_angle_less_than_26 :
  ∀ (L : Fin 7 → Line), 
  (∀ i j : Fin 7, i ≠ j → ¬Parallel (L i) (L j)) → 
  ∃ i j : Fin 7, i ≠ j ∧ angle (L i) (L j) < 26 :=
by
  sorry

end seven_lines_angle_less_than_26_l787_787979


namespace BD_value_l787_787737

noncomputable def triangle_area_given_hypotenuse (AC BD : ℝ) : Prop :=
  (1 / 2) * AC * BD = 200

theorem BD_value (AC BD : ℝ) (h₁: AC = 40) (h₂: triangle_area_given_hypotenuse AC BD) :
  BD = 10 :=
begin
  sorry
end

end BD_value_l787_787737


namespace a_and_b_together_time_eq_4_over_3_l787_787841

noncomputable def work_together_time (a b c h : ℝ) :=
  (1 / a) + (1 / b) + (1 / c) = (1 / (a - 6)) ∧
  (1 / a) + (1 / b) = 1 / h ∧
  (1 / (a - 6)) = (1 / (b - 1)) ∧
  (1 / (a - 6)) = 2 / c

theorem a_and_b_together_time_eq_4_over_3 (a b c h : ℝ) (h_wt : work_together_time a b c h) : 
  h = 4 / 3 :=
  sorry

end a_and_b_together_time_eq_4_over_3_l787_787841


namespace maximum_area_of_right_triangle_hypotenuse_6_l787_787313

-- Define the problem conditions in Lean 4
variables {a b : ℝ} (hypotenuse : a^2 + b^2 = 36)

-- Statement to be proved
theorem maximum_area_of_right_triangle_hypotenuse_6 : 
  ∃ (S : ℝ), S = 9 ∧ ∀ m n > 0, m^2 + n^2 = 36 → S = ½ * m * n := 
  sorry

end maximum_area_of_right_triangle_hypotenuse_6_l787_787313


namespace ned_diffuse_time_l787_787375

theorem ned_diffuse_time :
  let flights := 40
      time_per_flight := 13
      bomb_time := 58
      time_spent := 273
  in let total_time := flights * time_per_flight in
     let remaining_time := total_time - time_spent in
     let extra_time := time_spent - remaining_time in
     remaining_time + extra_time = 84 := by
  sorry

end ned_diffuse_time_l787_787375


namespace correct_propositions_l787_787920

-- Define the propositions as functions that return Prop.
def prop1 : Prop := ∀ (x : ℝ), x^2 ∈ ℝ
def prop2 : Prop := ∀ (x : ℝ), (x^2 ∈ ℝ) → (x ∈ ℝ)
def prop3 : Prop := ∀ (x1 x2 : ℂ) (y1 y2 : ℂ), (x1 + y1 * complex.I = x2 + y2 * complex.I) → (x1 = x2 ∧ y1 = y2)
def prop4 : Prop := ∀ (x1 x2 : ℂ) (y1 y2 : ℂ), (x1 = x2 ∧ y1 = y2) → (x1 + y1 * complex.I = x2 + y2 * complex.I)

-- The correct propositions are:
theorem correct_propositions : prop1 ∧ prop4 :=
by
  sorry

end correct_propositions_l787_787920


namespace system_of_eq_solutions_l787_787065

theorem system_of_eq_solutions :
  ∀ (x y z t : ℝ), 
  (x * y * z = x + y + z) →
  (y * z * t = y + z + t) →
  (z * t * x = z + t + x) →
  (t * x * y = t + x + y) →
  ((x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0) ∨ 
   (x = sqrt 3 ∧ y = sqrt 3 ∧ z = sqrt 3 ∧ t = sqrt 3) ∨
   (x = -sqrt 3 ∧ y = -sqrt 3 ∧ z = -sqrt 3 ∧ t = -sqrt 3)) :=
by sorry

end system_of_eq_solutions_l787_787065


namespace intersection_A_B_union_A_B_union_A_complement_B_l787_787283

noncomputable def A : Set ℝ := {x | x ≤ 5}
noncomputable def B : Set ℝ := {x | -3 < x ∧ x ≤ 8}

theorem intersection_A_B : A ∩ B = {x | -3 < x ∧ x ≤ 5} := sorry

theorem union_A_B : A ∪ B = {x | x ≤ 8} := sorry

theorem union_A_complement_B : A ∪ (set.univ \ B) = {x | x ≤ 5 ∨ x > 8} := sorry

end intersection_A_B_union_A_B_union_A_complement_B_l787_787283


namespace problem1_problem2_problem3_l787_787579
noncomputable section

-- Define the length of intervals
def interval_length (m n : ℝ) (h : n > m) : ℝ := n - m

-- (1) Given inequality ax^2 + 12x - 3 > 0 forms an interval of length 2√3, prove a = -3
theorem problem1 (a : ℝ) (h : 
  ∀ x : ℝ, ax^2 + 12*x - 3 > 0 ↔ (∃ x1 x2 : ℝ, x1 < x2 ∧ ax^2 + 12*x - 3 = 0 ∧ interval_length x1 x2 (by linarith [x1 < x2]) = 2 * Real.sqrt 3)) : 
  a = -3 := sorry

-- (2) Given inequality x^2 - 3x + (sin θ + cos θ) < 0 (θ ∈ (ℝ))
theorem problem2 (θ : ℝ) : ∀ x : ℝ, x^2 - 3*x + (Real.sin θ + Real.cos θ) < 0 ↔ 
    (∃ (m n : ℝ) (h : n > m), interval_length m n h ∈ Icc 1 ((2 : ℝ) * Real.sqrt 2 - 1)) := sorry

-- (3) Given system of inequalities has sum of interval lengths equal to 5, prove range of t
theorem problem3 (t : ℝ) (h : 
  (∃ (A B : set ℝ), 
     (∀ x, 7 / (x + 2) > 1 ↔ x ∈ A) ∧
     (∀ x, log 2 x + log 2 (t * x + 3 * t) < 3 ↔ x ∈ B) ∧
     ∀ (m n : ℝ) (hA : n > m), interval_length m n hA = 5)) : 
  t ∈ Ioc 0 (1 / 5) := sorry

end problem1_problem2_problem3_l787_787579


namespace greatest_number_of_groups_l787_787551

theorem greatest_number_of_groups (boys girls : ℕ) (h_boys : boys = 10) (h_girls : girls = 15) :
  nat.gcd boys girls = 5 := by
  sorry

end greatest_number_of_groups_l787_787551


namespace frogs_need_new_pond_l787_787883

theorem frogs_need_new_pond
  (num_frogs : ℕ) 
  (num_tadpoles : ℕ) 
  (num_survivor_tadpoles : ℕ) 
  (pond_capacity : ℕ) 
  (hc1 : num_frogs = 5)
  (hc2 : num_tadpoles = 3 * num_frogs)
  (hc3 : num_survivor_tadpoles = (2 * num_tadpoles) / 3)
  (hc4 : pond_capacity = 8):
  ((num_frogs + num_survivor_tadpoles) - pond_capacity) = 7 :=
by sorry

end frogs_need_new_pond_l787_787883


namespace number_of_music_students_l787_787506

-- Definitions based on given conditions
def total_students : ℕ := 500
def art_students : ℕ := 20
def both_music_and_art_students : ℕ := 10
def neither_music_nor_art_students : ℕ := 460

-- Proving the number of students taking music
theorem number_of_music_students : ∃ M : ℕ, M = 30 :=
by
  have total_union : ℕ := total_students - neither_music_nor_art_students
  have music_students : ℕ := total_union - (art_students - both_music_and_art_students)
  existsi music_students
  have : total_students = (music_students + art_students - both_music_and_art_students) + neither_music_nor_art_students
  simp [total_students, art_students, both_music_and_art_students, neither_music_nor_art_students] at *
  have : 500 = (music_students + 10) + 460
  exists 30
  sorry

end number_of_music_students_l787_787506


namespace parallel_line_slope_l787_787474

theorem parallel_line_slope (x y : ℝ) : (3 * x - 6 * y = 9) -> (∃ m : ℝ, m = 1/2) :=
by
  intro h
  use 1/2
  sorry

end parallel_line_slope_l787_787474


namespace sphere_property_l787_787702

-- Define the properties in a circle
def is_perpendicular (a b : ℝ → ℝ) : Prop := sorry -- This is a placeholder definition for perpendicularity

-- The given condition for a circle
def circle_condition (center : ℝ → ℝ) (midpoint : ℝ → ℝ) (chord : ℝ → ℝ) : Prop :=
  is_perpendicular (λ t, midpoint t - center t) (λ t, chord t)

-- Define the properties in a sphere
def is_perpendicular3d (line : ℝ → ℝ) (plane : ℝ → ℝ → Prop) : Prop := sorry -- Placeholder for 3d perpendicularity

-- The theorem we need to prove
theorem sphere_property
    (center_sphere : ℝ → ℝ)
    (center_circular_section : ℝ → ℝ)
    (plane_circular_section : ℝ → ℝ → Prop)
    (h_circle_prop : ∀ (center : ℝ → ℝ) (midpoint : ℝ → ℝ) (chord : ℝ → ℝ), circle_condition center midpoint chord) :
  is_perpendicular3d (λ t, center_circular_section t - center_sphere t) plane_circular_section :=
sorry

end sphere_property_l787_787702


namespace solve_fruit_supermarket_l787_787505

def sales_data := [(150, 3), (160, 4), (170, 6), (180, 5), (190, 1), (200, 1)]

def purchase_price := 6
def selling_price := 10
def return_price := 4

def expected_profit (purchase_amount : ℕ) : ℚ :=
  let profit := (purchase_amount * (selling_price - purchase_price) - purchase_amount * if purchase_amount > 170 then 2 else 1) in
  profit

theorem solve_fruit_supermarket :
  let avg_sales := 170 in
  let variance_sales := 170 in
  let purchase_170_profit := expected_profit 170 in
  let purchase_180_profit := expected_profit 180 in
  avg_sales = 170 ∧ variance_sales = 170 ∧ purchase_180_profit > purchase_170_profit :=
by
  sorry

end solve_fruit_supermarket_l787_787505


namespace find_a9_find_S10_l787_787360

variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)
variable (d : ℤ)

-- Conditions from the problem
def cond1 : Prop := a 1 + 2 * a 4 = a 6
def cond2 : Prop := S 3 = 3

-- Definitions of the sequence terms and sums based on typical arithmetic sequence properties
def sequence_def : Prop := ∀ n, a (n + 1) = a 1 + n * d
def sum_def : Prop := ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

-- Problem formulation
theorem find_a9 (h1 : cond1) (h2 : cond2) (h_seq : sequence_def) (h_sum : sum_def) :
  a 9 = 15 := sorry

theorem find_S10 (h1 : cond1) (h2 : cond2) (h_seq : sequence_def) (h_sum : sum_def) :
  S 10 = 80 := sorry

end find_a9_find_S10_l787_787360


namespace area_of_regular_inscribed_polygon_f3_properties_of_f_l787_787413

noncomputable def f (n : ℕ) : ℝ :=
  if h : n ≥ 3 then (n / 2) * Real.sin (2 * Real.pi / n) else 0

theorem area_of_regular_inscribed_polygon_f3 :
  f 3 = (3 * Real.sqrt 3) / 4 :=
by
  sorry

theorem properties_of_f (n : ℕ) (hn : n ≥ 3) :
  (f n = (n / 2) * Real.sin (2 * Real.pi / n)) ∧
  (f n < f (n + 1)) ∧ 
  (f n < f (2 * n) ∧ f (2 * n) ≤ 2 * f n) :=
by
  sorry

end area_of_regular_inscribed_polygon_f3_properties_of_f_l787_787413


namespace abs_neg_three_l787_787412

theorem abs_neg_three : abs (-3) = 3 :=
sorry

end abs_neg_three_l787_787412


namespace joggers_proof_l787_787372

noncomputable def joggers_problem : Prop :=
  ∃ (T : ℕ),
    let Christopher := 20 * T in
    let Martha := if (T >= 15) then T - 15 else 0 in
    let Alexander := T + 22 in
    let Natasha := 2 * (Martha + Alexander) in
    Christopher = 80 ∧ Christopher - Natasha = 28

theorem joggers_proof : joggers_problem :=
sorry

end joggers_proof_l787_787372


namespace f_2019_eq_2019_l787_787604

def f : ℝ → ℝ := sorry

axiom f_pos : ∀ x, x > 0 → f x > 0
axiom f_one : f 1 = 1
axiom f_eq : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2

theorem f_2019_eq_2019 : f 2019 = 2019 :=
by sorry

end f_2019_eq_2019_l787_787604


namespace probability_difference_3_or_greater_l787_787458

noncomputable def set_of_numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def total_combinations : ℕ := (finset.image (λ (p : finset ℕ), p) (finset.powerset (finset.univ ∩ set_of_numbers))).card / 2

def pairs_with_difference_less_than_3 (S : finset ℕ) : finset (finset ℕ) :=
finset.filter (λ (p : finset ℕ), p.card = 2 ∧ abs ((p.min' sorry) - (p.max' sorry)) < 3) S

def successful_pairs (S : finset ℕ) : finset (finset ℕ) :=
finset.filter (λ (p : finset ℕ), p.card = 2 ∧ abs ((p.min' sorry) - (p.max' sorry)) ≥ 3) S
   
def probability (S : finset ℕ) : ℚ :=
(successful_pairs S).card.to_rat / total_combinations

theorem probability_difference_3_or_greater :
  probability set_of_numbers = 7/12 :=
sorry

end probability_difference_3_or_greater_l787_787458


namespace problem_BD_length_l787_787382

variable (b : ℝ) (x : ℝ)
variable (h : b^2 - 3 ≥ 0)

theorem problem_BD_length :
  let A := (0, 0)
  let B := (sqrt (b^2 + 1), 0)
  let C := (0, 1)
  let D := (sqrt (b^2 + 1), 2)

  ∥B - D∥ = sqrt (b^2 - 3) := by
    sorry

end problem_BD_length_l787_787382


namespace sqrt_inequality_l787_787394

theorem sqrt_inequality :
  (sqrt 6 - sqrt 5 > 2 * sqrt 2 - sqrt 7) :=
by
  sorry

end sqrt_inequality_l787_787394


namespace limit_of_function_l787_787864

open Real

-- The problem statement converted into Lean 4 statement
theorem limit_of_function :
  tendsto (λ x : ℝ, (2 - exp (sin x)) ^ (cot (π * x))) (nhds 0) (nhds (exp (-1 / π))) :=
  sorry

end limit_of_function_l787_787864


namespace set_complement_union_l787_787733

def A : Set ℝ := {x | -2 < x ∧ x < 5}
def B : Set ℝ := {x | x ≤ 3 ∨ x ≥ 8}
def compl (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

theorem set_complement_union :
  compl A ∪ compl B = {x | x > 3 ∨ x ≤ -2} :=
by
  sorry

end set_complement_union_l787_787733


namespace polynomial_p_evaluation_l787_787362

noncomputable def p : ℝ → ℝ := sorry

theorem polynomial_p_evaluation : 
  (∀ n ∈ range 9, p (3^n) = 1 / 3^n) → p(0) = 19682 / 19683 :=
by
  intro h
  sorry

end polynomial_p_evaluation_l787_787362


namespace solve_equation_l787_787062

noncomputable def z (x : ℂ) : ℂ := x - 4

theorem solve_equation (x : ℂ) :
  (z x + 2)^4 + (z x - 2)^4 = -16 →
  x = 4 + 2 * complex.i * real.sqrt 3 ∨
  x = 4 - 2 * complex.i * real.sqrt 3 ∨
  x = 4 + complex.i * real.sqrt 2 ∨
  x = 4 - complex.i * real.sqrt 2 :=
by
  sorry

end solve_equation_l787_787062


namespace kyler_wins_zero_l787_787317

-- Definitions based on conditions provided
def peter_games_won : ℕ := 5
def peter_games_lost : ℕ := 3
def emma_games_won : ℕ := 4
def emma_games_lost : ℕ := 4
def kyler_games_lost : ℕ := 4

-- Number of games each player played
def peter_total_games : ℕ := peter_games_won + peter_games_lost
def emma_total_games : ℕ := emma_games_won + emma_games_lost
def kyler_total_games (k : ℕ) : ℕ := k + kyler_games_lost

-- Step 1: total number of games in the tournament
def total_games (k : ℕ) : ℕ := (peter_total_games + emma_total_games + kyler_total_games k) / 2

-- Step 2: Total games equation
def games_equation (k : ℕ) : Prop := 
  (peter_games_won + emma_games_won + k = total_games k)

-- The proof problem, we need to prove Kyler's wins
theorem kyler_wins_zero : games_equation 0 := by
  -- proof omitted
  sorry

end kyler_wins_zero_l787_787317


namespace functional_equation_zero_solution_l787_787953

theorem functional_equation_zero_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
sorry

end functional_equation_zero_solution_l787_787953


namespace max_sum_of_distances_le_a_b_c_d_l787_787327

theorem max_sum_of_distances_le_a_b_c_d
  (a b c d : ℝ)
  (h_ord : d ≤ c ∧ c ≤ b ∧ b ≤ a)
  (ABCD : Quadrilateral a b c d)
  (P : ABCD.interior)
  (A' B' C' D' : ABCD.sides_intersection_points P) :
  let s := (P.distance_to_side A') + (P.distance_to_side B') + (P.distance_to_side C') + (P.distance_to_side D')
  in s ≤ a + b + c + d :=
by
  sorry

end max_sum_of_distances_le_a_b_c_d_l787_787327


namespace inscribed_square_in_right_triangle_l787_787050

theorem inscribed_square_in_right_triangle
  (DE EF DF : ℝ) (h1 : DE = 5) (h2 : EF = 12) (h3 : DF = 13)
  (t : ℝ) (h4 : t = 780 / 169) :
  (∃ PQRS : ℝ, PQRS = t) :=
begin
  use t,
  exact h4,
  sorry,
end

end inscribed_square_in_right_triangle_l787_787050


namespace distance_between_parallel_tangents_l787_787940

theorem distance_between_parallel_tangents 
  (m : ℝ) 
  : let d := (abs(m * (m - 2))) / (2 * sqrt(1 + m^2)) 
    in d = (abs(m * (m - 2))) / (2 * sqrt(1 + m^2)) :=
by
  sorry

end distance_between_parallel_tangents_l787_787940


namespace find_a_find_A_l787_787697

-- Part (I)
theorem find_a (b c : ℝ) (A : ℝ) (hb : b = 2) (hc : c = 2 * Real.sqrt 3) (hA : A = 5 * Real.pi / 6) :
  ∃ a : ℝ, a = 2 * Real.sqrt 7 :=
by {
  sorry
}

-- Part (II)
theorem find_A (b c : ℝ) (C : ℝ) (hb : b = 2) (hc : c = 2 * Real.sqrt 3) (hC : C = Real.pi / 2 + A) :
  ∃ A : ℝ, A = Real.pi / 6 :=
by {
  sorry
}

end find_a_find_A_l787_787697


namespace volume_of_larger_part_of_pyramid_proof_l787_787866

noncomputable def volume_of_larger_part_of_pyramid (a b : ℝ) (inclined_angle : ℝ) (area_ratio : ℝ) : ℝ :=
let h_trapezoid := Real.sqrt ((2 * Real.sqrt 3)^2 - (Real.sqrt 3)^2 / 4)
let height_pyramid := (1 / 2) * h_trapezoid * Real.tan (inclined_angle)
let volume_total := (1 / 3) * (((a + b) / 2) * Real.sqrt ((a - b) ^ 2 + 4 * h_trapezoid ^ 2) * height_pyramid)
let volume_smaller := (1 / (5 + 7)) * 7 * volume_total
(volume_total - volume_smaller)

theorem volume_of_larger_part_of_pyramid_proof  :
  (volume_of_larger_part_of_pyramid 2 (Real.sqrt 3) (Real.pi / 6) (5 / 7) = 0.875) :=
by
sorry

end volume_of_larger_part_of_pyramid_proof_l787_787866


namespace jelly_sold_l787_787893

theorem jelly_sold (G S R P : ℕ) (h1 : G = 2 * S) (h2 : R = 2 * P) (h3 : R = G / 3) (h4 : P = 6) : S = 18 := by
  sorry

end jelly_sold_l787_787893


namespace eccentricity_of_ellipse_standard_equation_of_ellipse_l787_787112

noncomputable theory
open_locale classical

-- Definitions

def ellipse (a b : ℝ) : Prop :=
a > b ∧ b > 0 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

def point_M (b : ℝ) : ℝ × ℝ := (-2 * b, 0)

def are_perpendicular (ma mb : ℝ) : Prop :=
ma * mb = -1

def meet_at_point (A B M : ℝ × ℝ) : Prop :=
True  -- placeholder since actual intersection conditions are not provided

def area_quadrilateral_MAFB (area : ℝ) : Prop :=
area = 2 + sqrt 2

-- Statements to prove

theorem eccentricity_of_ellipse (a b : ℝ) (M : ℝ × ℝ) (MA_perp_MB : Prop) (e : ℝ)
  (h_ellipse : ellipse a b)
  (h_pointM : M = point_M b)
  (h_perp : MA_perp_MB = are_perpendicular 1 (-1))
  : e = sqrt 6 / 3 := sorry

theorem standard_equation_of_ellipse (a b : ℝ) 
  (h_area : area_quadrilateral_MAFB (2 + sqrt 2))
  (h_a_squared : a^2 = 3 * b^2) :
  (∀ x y : ℝ, x^2 / 6 + y^2 / 2 = 1) := sorry

end eccentricity_of_ellipse_standard_equation_of_ellipse_l787_787112


namespace print_shop_x_charges_l787_787969

theorem print_shop_x_charges (x : ℝ) (h1 : ∀ y : ℝ, y = 1.70) (h2 : 40 * x + 20 = 40 * 1.70) : x = 1.20 :=
by
  sorry

end print_shop_x_charges_l787_787969


namespace z_in_fourth_quadrant_l787_787265

noncomputable def z_position : Prop :=
  let i := complex.I in
  ∃ z : ℂ, z * (1 + 2 * i) ^ 2 = 3 + 4 * i ∧ ((z.re > 0) ∧ (z.im < 0))

theorem z_in_fourth_quadrant : z_position := sorry

end z_in_fourth_quadrant_l787_787265


namespace exterior_angle_regular_octagon_l787_787719

-- Definition and proof statement
theorem exterior_angle_regular_octagon :
  let n := 8 -- The number of sides of the polygon (octagon)
  let interior_angle_sum := 180 * (n - 2) 
  let interior_angle := interior_angle_sum / n
  let exterior_angle := 180 - interior_angle
  exterior_angle = 45 :=
by
  let n := 8
  let interior_angle_sum := 180 * (n - 2) 
  let interior_angle := interior_angle_sum / n
  let exterior_angle := 180 - interior_angle
  sorry

end exterior_angle_regular_octagon_l787_787719


namespace statement_B_statement_C_statement_A_is_false_statement_D_is_false_l787_787642

-- Define f(x) as an odd function and satisfy given conditions
variables {f : ℝ → ℝ}

-- Assume f is an odd function, f(x+2) = f(-x), and f(1) = 2
axiom odd_f : ∀ x, f(-x) = -f(x)
axiom f_property : ∀ x, f(x + 2) = f(-x)
axiom f_one : f(1) = 2

theorem statement_B : ∀ x, deriv f (x + 4) = deriv f x :=
by sorry

theorem statement_C : ∀ x, deriv f (-x) = deriv f x :=
by sorry

theorem statement_A_is_false : f 2023 ≠ 2 :=
by sorry

theorem statement_D_is_false : deriv f 1 ≠ 1 :=
by sorry

end statement_B_statement_C_statement_A_is_false_statement_D_is_false_l787_787642


namespace bushes_needed_for_octagon_perimeter_l787_787923

theorem bushes_needed_for_octagon_perimeter
  (side_length : ℝ) (spacing : ℝ)
  (octagonal : ∀ (s : ℝ), s = 8 → 8 * s = 64)
  (spacing_condition : ∀ (p : ℝ), p = 64 → p / spacing = 32) :
  spacing = 2 → side_length = 8 → (64 / 2 = 32) := 
by
  sorry

end bushes_needed_for_octagon_perimeter_l787_787923


namespace value_is_85_over_3_l787_787296

theorem value_is_85_over_3 (a b : ℚ)  (h1 : 3 * a + 6 * b = 48) (h2 : 8 * a + 4 * b = 84) : 2 * a + 3 * b = 85 / 3 := 
by {
  -- Proof will go here
  sorry
}

end value_is_85_over_3_l787_787296


namespace probability_of_drawing_ball_three_twice_sum_six_l787_787610

def balls : Finset ℕ := {1, 2, 3, 4}

-- Definitions for the problem
def draws : Type := list ℕ

def valid_draws (d : draws) : Prop :=
  d.length = 3 ∧ d.all (λ n, n ∈ balls)

def sum_is_six (d : draws) : Prop :=
  d.sum = 6

def count_ball_three (d : draws) : ℕ :=
  d.count (3)

def favorable (d : draws) : Prop :=
  count_ball_three d = 2

-- The main theorem we want to prove
theorem probability_of_drawing_ball_three_twice_sum_six : 
  (∑ d in (Finset.filter valid_draws (draws.ball_combinations 4 3)), if favorable d ∧ sum_is_six d then 1 else 0).to_real = 0 := 
  sorry

end probability_of_drawing_ball_three_twice_sum_six_l787_787610


namespace fallen_striped_tiles_l787_787450

-- Definition of checkerboard pattern and given conditions
def checkerboard_pattern : Prop := sorry -- Assume a definition for the pattern

-- Remaining tiles shown in the image (assumed given as a list or grid)
def remaining_tiles : list (list (option bool)) := sorry -- Option bool can signify striped (true) or plain (false) and missing (none)

-- Definition of the number of fallen striped tiles
def number_of_fallen_striped_tiles : ℕ := 15

-- Theorem to prove the correct number of fallen striped tiles
theorem fallen_striped_tiles {C : checkerboard_pattern} {R : remaining_tiles} :
  number_of_fallen_striped_tiles = 15 :=
sorry

end fallen_striped_tiles_l787_787450


namespace choose_pencils_diff_colors_l787_787590

noncomputable theory

-- Define the type for boxes and pencils
variable (Box : Type) (Pencil : Type) [decidable_eq Pencil]

-- Define the number of boxes and boxes themselves
def num_boxes : ℕ := 10
def boxes := fin num_boxes

-- Define a function that gives the number of pencils in each box
variable (num_pencils_in_box : boxes → ℕ)

-- Condition: Each of the ten boxes contains a different number of pencils
axiom unique_pencil_counts :
  ∀ (i j : boxes), i ≠ j → num_pencils_in_box i ≠ num_pencils_in_box j

-- Define a function that gives the pencils in each box
variable (pencils_in_box : boxes → fin (num_pencils_in_box))

-- Define a function that gives the color of each pencil
variable (color : Pencil → ℕ)

-- Condition: No two pencils in the same box are of the same color
axiom unique_colors_in_box :
  ∀ (i : boxes) (p1 p2 : Pencil), 
    p1 ≠ p2 → color p1 ≠ color p2

-- Question: Prove that one can choose one pencil from each box so that no two are of the same color
theorem choose_pencils_diff_colors :
  ∃ (chosen_pencils : boxes → Pencil),
  ∀ (i j : boxes), i ≠ j → color (chosen_pencils i) ≠ color (chosen_pencils j) := 
by
  sorry

end choose_pencils_diff_colors_l787_787590


namespace solve_trig_identity_l787_787793

theorem solve_trig_identity (x y z : ℝ) (h1 : cos x ≠ 0) (h2 : cos y ≠ 0)
  (h_eq : (cos x ^ 2 + 1 / (cos x ^ 2)) ^ 3 + (cos y ^ 2 + 1 / (cos y ^ 2)) ^ 3 = 16 * sin z) :
  ∃ (n k m : ℤ), x = n * π ∧ y = k * π ∧ z = π / 2 + 2 * m * π :=
by
  sorry

end solve_trig_identity_l787_787793


namespace salesman_profit_l787_787904

theorem salesman_profit :
  let total_backpacks := 48
  let total_cost := 576
  let swap_meet_sold := 17
  let swap_meet_price := 18
  let department_store_sold := 10
  let department_store_price := 25
  let remaining_price := 22
  let remaining_sold := total_backpacks - swap_meet_sold - department_store_sold
  let revenue_swap_meet := swap_meet_sold * swap_meet_price
  let revenue_department_store := department_store_sold * department_store_price
  let revenue_remaining := remaining_sold * remaining_price
  let total_revenue := revenue_swap_meet + revenue_department_store + revenue_remaining
  let profit := total_revenue - total_cost
  in profit = 442 := by
  sorry

end salesman_profit_l787_787904


namespace parallel_lines_not_coincident_l787_787306

theorem parallel_lines_not_coincident (a : ℝ) :
  (∀ x y : ℝ, ax + 2 * y + 6 = 0) ∧
  (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ (b : ℝ), ∀ x y : ℝ, ax + 2 * y + b = 0 ∧ x + (a - 1) * y + b = 0 →
  a = -1 :=
by
  sorry

end parallel_lines_not_coincident_l787_787306


namespace log_sum_geometric_sequence_l787_787984

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem log_sum_geometric_sequence (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : geometric_sequence a) (h_condition : a 5 * a 6 + a 4 * a 7 = 18) : 
  (∑ i in finset.range 10, real.log (a i) / real.log 3) = 10 := 
sorry

end log_sum_geometric_sequence_l787_787984


namespace range_of_fractional_expression_l787_787987

theorem range_of_fractional_expression :
  ∀ (M : Type) (A B C : ℝ × ℝ) (a x y : ℝ),
  (∃ m n p : ℝ, f(M) = (m, n, p) ∧ m = 1/2 ∧ x + y = 1/2 ∧ 1/x + 4/y = a)
  ∧ abs (A.1 - B.1) * abs (A.2 - B.2) * cos(30 * π / 180) = 2 * sqrt 3
  ∧ ∠ A B C = 30 * π / 180
  → 
  (∀ t : ℝ, (∃ a : ℝ, a ≥ 18) → t = (a^2 + 2) / a → t ∈ [163 / 9, ∞)) :=
sorry

end range_of_fractional_expression_l787_787987


namespace exponential_inequality_l787_787632

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^a * b^b ≥ (a * b)^((a + b) / 2) :=
sorry

end exponential_inequality_l787_787632


namespace important_set_of_symmetric_difference_l787_787333

-- Define important sets and strategic sets
structure Graph (V E : Type) :=
  (edges : E → V × V)

def important_set {V E : Type} [DecidableEq V] (G : Graph V E) (S : set E) : Prop :=
  ∀ (T : set E), T ⊆ S → 
    let V' := {v : V | ∃ (e : E) (h : e ∈ T), v ∈ G.edges e} in
    ∃ (U₁ U₂ : set V), U₁ ≠ ∅ ∧ U₂ ≠ ∅ ∧ U₁ ∩ U₂ = ∅ ∧ (V' \ U₁ \ U₂ = ∅)

def strategic_set {V E : Type} [DecidableEq V] (G : Graph V E) (S : set E) : Prop :=

theorem important_set_of_symmetric_difference 
  {V E : Type} [DecidableEq V] (G : Graph V E) 
  (S₁ S₂ : set E) 
  (h₁ : strategic_set G S₁) 
  (h₂ : strategic_set G S₂) 
  (hne : ¬ (S₁ = S₂)) : 
sorry

end important_set_of_symmetric_difference_l787_787333


namespace boxes_filled_per_week_l787_787191

-- Define the total number of hens and the percentages
def total_hens : ℕ := 270
def laying_percentage : ℝ := 0.90
def eight_am_percentage : ℝ := 0.40
def nine_am_percentage : ℝ := 0.50
def unusable_percentage : ℝ := 0.05
def eggs_per_box : ℕ := 7
def days_per_week : ℕ := 7

-- Define the total number of laying hens
def laying_hens : ℕ := (laying_percentage * total_hens).toNat

-- Define the number of laying hens at 8am and 9am
def laying_hens_eight : ℕ := (eight_am_percentage * laying_hens).toNat
def laying_hens_nine : ℕ := (nine_am_percentage * laying_hens).toNat

-- Define the total number of eggs collected each morning
def total_eggs : ℕ := laying_hens_eight + laying_hens_nine

-- Define the number of usable eggs after removing the unusable ones
def usable_eggs : ℕ := total_eggs - (unusable_percentage * total_eggs).toNat

-- Define the number of boxes filled per day
def boxes_per_day : ℕ := usable_eggs / eggs_per_box

-- Define the total number of boxes filled per week
def boxes_per_week : ℕ := boxes_per_day * days_per_week

theorem boxes_filled_per_week : boxes_per_week = 203 := 
by 
  have h1 : laying_hens = 243 := by sorry
  have h2 : laying_hens_eight = 97 := by sorry
  have h3 : laying_hens_nine = 121 := by sorry
  have h4 : total_eggs = 218 := by sorry
  have h5 : usable_eggs = 208 := by sorry
  have h6 : boxes_per_day = 29 := by sorry
  have h7 : boxes_per_week = 203 := by sorry
  exact h7

end boxes_filled_per_week_l787_787191


namespace part1_part2_part3_l787_787258

-- Definitions and conditions related to the given functions
def y1 (m : ℝ) (x : ℝ) : ℝ := -x^2 + (m + 2) * x - 2 * m + 1
def y2 (n : ℝ) (x : ℝ) : ℝ := (n + 2) * x - 2 * n - 3

-- Condition: Vertex of y1 when m = 0 lies on the graph of y2
def condition1 (P : ℝ × ℝ) (n : ℝ) : Prop := P = (1, 2) ∧ y2 n 1 = 2
-- Proof problem part 1: Find value of n
theorem part1 (n : ℝ) : condition1 (1, 2) n → n = -3 := sorry

-- Condition: All points P lie on a certain parabola as m varies
def condition2 (m : ℝ) : ℝ × ℝ := ((m + 2) / 2, (m^2 - 4 * m + 8) / 4)
-- Proof problem part 2: Points P of y1 lie on the parabola y = x^2 - 4x + 5
theorem part2 (a : ℝ) : (∃ m : ℝ, condition2 m = (a, a^2 - 4 * a + 5)) := sorry

-- Condition: For -1 < x < 2, it is always the case that y2 < y1
def condition3 (m n x : ℝ) : Prop := -1 < x ∧ x < 2 ∧ y2 n x < y1 m x
-- Proof problem part 3: Find range of values for m - n satisfying condition3
theorem part3 (m n : ℝ) : (∀ x : ℝ, condition3 m n x) → m - n ≤ 1 := sorry

end part1_part2_part3_l787_787258


namespace perfect_square_pairs_l787_787956

-- Definition of a perfect square
def is_perfect_square (k : ℕ) : Prop :=
∃ (n : ℕ), n * n = k

-- Main theorem statement
theorem perfect_square_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  is_perfect_square ((2^m - 1) * (2^n - 1)) ↔ (m = n) ∨ (m = 3 ∧ n = 6) ∨ (m = 6 ∧ n = 3) :=
sorry

end perfect_square_pairs_l787_787956


namespace roots_of_eq_l787_787432

theorem roots_of_eq (x : ℝ) : (x - 1) * (x - 2) = 0 ↔ (x = 1 ∨ x = 2) := by
  sorry

end roots_of_eq_l787_787432


namespace lines_concurrent_l787_787991

-- Define a triangle with points A, B, C and additional points M, N, R
variables {A B C M N R : Type}

-- Define angles as real values
variable {α β γ : ℝ}

-- Assume the given conditions
axiom angle_BAR_eq_alpha : ∠ BAR = α 
axiom angle_CAN_eq_alpha : ∠ CAN = α 
axiom angle_CBM_eq_beta : ∠ CBM = β 
axiom angle_ABR_eq_beta : ∠ ABR = β 
axiom angle_ACN_eq_gamma : ∠ ACN = γ 
axiom angle_BCM_eq_gamma : ∠ BCM = γ 

-- Prove that lines AM, BN, and CR are concurrent
theorem lines_concurrent : concurrent {AM, BN, CR} :=
by sorry

end lines_concurrent_l787_787991


namespace LineChart_characteristics_and_applications_l787_787027

-- Definitions related to question and conditions
def LineChart : Type := sorry
def represents_amount (lc : LineChart) : Prop := sorry
def reflects_increase_or_decrease (lc : LineChart) : Prop := sorry

-- Theorem related to the correct answer
theorem LineChart_characteristics_and_applications (lc : LineChart) :
  represents_amount lc ∧ reflects_increase_or_decrease lc :=
sorry

end LineChart_characteristics_and_applications_l787_787027


namespace translated_parabola_expression_l787_787117

theorem translated_parabola_expression (x : ℝ) :
  let y := x^2 in
  let y_left := (x + 1)^2 in
  let y_up_left := y_left + 3 in
  y_up_left = (x + 1)^2 + 3 :=
by
  sorry

end translated_parabola_expression_l787_787117


namespace six_digit_symmetrical_numbers_count_l787_787485

-- Definitions
def is_symmetrical (n : Nat) : Bool :=
  let s := toString n
  s == s.reverse

def six_digit_symmetrical_remain_symmetrical (n : Nat) : Prop :=
  is_symmetrical n ∧ (Nat.toDigits n).length = 6 ∧ is_symmetrical (n + 110)

-- Problem statement
theorem six_digit_symmetrical_numbers_count : { n : Nat // six_digit_symmetrical_remain_symmetrical n }.card = 81 :=
by
  sorry

end six_digit_symmetrical_numbers_count_l787_787485


namespace find_solution_l787_787955

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_first_p_squares (p : ℕ) : ℕ := p * (p + 1) * (2 * p + 1) / 6

theorem find_solution : ∃ (n p : ℕ), p.Prime ∧ sum_first_n n = 3 * sum_first_p_squares p ∧ (n, p) = (5, 2) := 
by
  sorry

end find_solution_l787_787955


namespace no_primitive_root_mod_n_l787_787744

theorem no_primitive_root_mod_n (n : ℕ) (h : ∃ (p q : ℕ), p ≠ q ∧ p.prime ∧ q.prime ∧ p % 2 = 1 ∧ q % 2 = 1 ∧ p ∣ n ∧ q ∣ n) : 
  ¬ ∃ a : ℤ, nat.coprime a n ∧ (∀ b : ℤ, nat.coprime b n → ∃ k : ℕ, b = a^k % n) :=
by sorry

end no_primitive_root_mod_n_l787_787744


namespace complement_intersection_l787_787287

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def M : Set ℕ := {1, 4}
noncomputable def N : Set ℕ := {2, 3}

theorem complement_intersection :
  ((U \ M) ∩ N) = {2, 3} :=
by
  sorry

end complement_intersection_l787_787287


namespace binomial_sum_zero_l787_787389

open BigOperators

theorem binomial_sum_zero {n m : ℕ} (h1 : 1 ≤ m) (h2 : m < n) :
  ∑ k in Finset.range (n + 1), (-1 : ℤ) ^ k * k ^ m * Nat.choose n k = 0 :=
by
  sorry

end binomial_sum_zero_l787_787389


namespace monotonic_decreasing_interval_of_function_l787_787087

noncomputable def domain := set.Icc (-1/2 : ℝ) (1/2 : ℝ)
noncomputable def original_function (x : ℝ) : ℝ := -sqrt (1 - 4 * x^2)

theorem monotonic_decreasing_interval_of_function :
  ∀ x ∈ domain, ∀ y ∈ domain, (x < y) → (original_function x > original_function y) :=
  sorry

end monotonic_decreasing_interval_of_function_l787_787087


namespace least_value_of_sum_l787_787140

theorem least_value_of_sum (x y z : ℤ) 
  (h_cond : (x - 10) * (y - 5) * (z - 2) = 1000) : x + y + z ≥ 56 :=
sorry

end least_value_of_sum_l787_787140


namespace minimal_cycles_l787_787906

open GraphTheory

-- Let's define a structure and the proof problem

structure Graph :=
  (vertices : ℕ)
  (edges : ℕ)
  (connected_components : ℕ)

def cyclomatic_number (G : Graph) : ℕ :=
  G.edges - G.vertices + G.connected_components

theorem minimal_cycles (G : Graph) (h_vertices : G.vertices = 2013) (h_edges : G.edges = 3013) (h_cc : G.connected_components = 1) :
  ∃ r, r = 1001 :=
by
  have h_cyclomatic : cyclomatic_number G = 1001 :=
    calc
      cyclomatic_number G = G.edges - G.vertices + G.connected_components : by rfl
      ... = 3013 - 2013 + 1 : by rw [h_edges, h_vertices, h_cc]
      ... = 1001 : by norm_num
  use 1001
  exact h_cyclomatic

end minimal_cycles_l787_787906


namespace salary_increase_percent_l787_787040

noncomputable def salary_increase (initial_salary : ℝ) : ℝ :=
  initial_salary * (1.12)^3

theorem salary_increase_percent (initial_salary : ℝ) :
  (salary_increase initial_salary - initial_salary) / initial_salary * 100 ≈ 41 :=
by 
  sorry

end salary_increase_percent_l787_787040


namespace user_count_exceed_50000_l787_787026

noncomputable def A (t : ℝ) (k : ℝ) := 500 * Real.exp (k * t)

theorem user_count_exceed_50000 :
  (∃ k : ℝ, A 10 k = 2000) →
  (∀ t : ℝ, A t k > 50000) →
  ∃ t : ℝ, t >= 34 :=
by
  sorry

end user_count_exceed_50000_l787_787026


namespace integral_solution_l787_787204

open Complex

noncomputable def integral_problem : ℂ :=
  ∫ z in {z : ℂ | abs z = 1 ∧ 0 ≤ arg z ∧ arg z ≤ π}, (z^2 + z * conj(z)) * dz

theorem integral_solution :
  integral_problem = -(8 / 3) := by
  -- proof skipped
  sorry

end integral_solution_l787_787204


namespace distance_from_P_to_A_l787_787302

theorem distance_from_P_to_A (z : ℝ) (P : ℝ × ℝ × ℝ) (hPO : P = (0, 0, z) ∧ abs z = 1) : 
  let d := dist P (1, 1, 1) in d = real.sqrt 2 ∨ d = real.sqrt 6 :=
sorry

end distance_from_P_to_A_l787_787302


namespace line_passes_through_quadrants_l787_787086

noncomputable def line_equation (x : ℝ) : ℝ := (1 / 5) * x + real.sqrt 2

theorem line_passes_through_quadrants : 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = line_equation x) ∧ 
  (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ y = line_equation x) ∧ 
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ y = line_equation x) :=
sorry

end line_passes_through_quadrants_l787_787086


namespace evaluate_expression_when_x_is_3_l787_787783

theorem evaluate_expression_when_x_is_3 :
  let x := 3 in
  (((x + 1) / (x - 2) + 1) / ((x^2 - 2*x) / (x^2 - 4*x + 4))) = 5 / 3 :=
by
  sorry

end evaluate_expression_when_x_is_3_l787_787783


namespace DeMorgansLaws_l787_787606

variable (U : Type) (A B : Set U)

theorem DeMorgansLaws :
  (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ ∧ (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ :=
by
  -- Statement of the theorems, proof is omitted
  sorry

end DeMorgansLaws_l787_787606


namespace cos_gamma_prime_l787_787357

theorem cos_gamma_prime (x' y' z' : ℝ) (hx : 0 < x') (hy : 0 < y') (hz : 0 < z') 
  (cos_alpha' : cos ⁡(atan2 y' x') = 2 / 5) 
  (cos_beta' : cos ⁡(atan2 z' (x'² + y'²) ^ (1 / 2)) = 1 / 4) :
  (cos ⁡(atan2 y' (sqrt (x'^2 + y'^2 + z'^2))) = sqrt 311 / 20) :=
by
  sorry

end cos_gamma_prime_l787_787357


namespace inscribed_square_in_right_triangle_l787_787051

theorem inscribed_square_in_right_triangle
  (DE EF DF : ℝ) (h1 : DE = 5) (h2 : EF = 12) (h3 : DF = 13)
  (t : ℝ) (h4 : t = 780 / 169) :
  (∃ PQRS : ℝ, PQRS = t) :=
begin
  use t,
  exact h4,
  sorry,
end

end inscribed_square_in_right_triangle_l787_787051


namespace polynomials_commute_l787_787006

noncomputable def f (a b c : ℝ) : ℝ → ℝ := λ x, a * x ^ 2 + b * x + c

def is_commutative_with (f p : ℝ → ℝ) : Prop := ∀ x, p (f x) = f (p x)

def p (x : ℝ) : ℝ := sorry
def q (x : ℝ) : ℝ := sorry

theorem polynomials_commute
  (a b c : ℝ) (ha : a ≠ 0)
  (p_comm : is_commutative_with (f a b c) p)
  (q_comm : is_commutative_with (f a b c) q)
  : ∀ x, p (q x) = q (p x) := by
  sorry

end polynomials_commute_l787_787006


namespace compute_f_six_l787_787732

def f (x : Int) : Int :=
  if x ≥ 0 then -x^2 - 1 else x + 10

theorem compute_f_six (x : Int) : f (f (f (f (f (f 1))))) = -35 :=
by
  sorry

end compute_f_six_l787_787732


namespace number_of_routes_from_M_to_P_is_4_l787_787323

def routes_from_M_to_P (routes: ℕ) : Prop :=
  let M_to_X := 1
  let M_to_Y := 1
  let X_to_Z := 1
  let X_to_W := 1
  let X_direct := 1
  let Y_to_Z := 1
  let W_direct := 1
  let Z_direct := 1

  -- Ways from Z to P
  let ways_Z_to_P := Z_direct

  -- Ways from W to P
  let ways_W_to_P := W_direct

  -- Ways from X to P
  let ways_X_to_P := ways_Z_to_P + ways_W_to_P + X_direct

  -- Ways from Y to P
  let ways_Y_to_P := Y_to_Z * ways_Z_to_P

  -- Ways from M to P
  let ways_M_to_P := M_to_X * ways_X_to_P + M_to_Y * ways_Y_to_P

  routes = ways_M_to_P

theorem number_of_routes_from_M_to_P_is_4 : routes_from_M_to_P 4 :=
by simp [routes_from_M_to_P, 
         let ways_Z_to_P := 1 in 
         let ways_W_to_P := 1 in 
         let ways_X_to_P := 1 + 1 + 1 in 
         let ways_Y_to_P := 1 * 1 in 
         let ways_M_to_P := 1 * 3 + 1 * 1 in
         ways_M_to_P]

end number_of_routes_from_M_to_P_is_4_l787_787323


namespace circumspheres_intersection_l787_787134

theorem circumspheres_intersection (A B C A' B' C' A'' B'' C'' X X' X'' O : Type)
  (tet1 : XABC)
  (tet2 : X'A'B'C')
  (tet3 : X"A"B"C")
  (collinear : collinear {X, X', X''})
  (meet_at_O : meets_at_O {A, B, C} {X} {O} ∧ meets_at_O {A', B', C'} {X'} {O} ∧ meets_at_O {A'', B'', C''} {X''} {O})
  (coincident_pts : O = O' ∧ O' = O'') : 
  ∃ intersection, 
    (intersection = sphere ∨ intersection = point O ∨ intersection = circle) := 
sorry

end circumspheres_intersection_l787_787134


namespace students_sharing_name_and_surname_l787_787318

theorem students_sharing_name_and_surname (n: ℕ) (counts: finset ℕ): 
  n = 33 → 
  (∀ c ∈ counts, c ∈ finset.range 11) → 
  counts.card = 11 → 
  ∃ (a b : fin n), a ≠ b ∧ 
    (∀ (group: fin 11), (a.val.count_sharing_name group = b.val.count_sharing_name group) ∧ 
      (a.val.count_sharing_surname group = b.val.count_sharing_surname group)) 
  → True
:= 
by 
  sorry

end students_sharing_name_and_surname_l787_787318


namespace find_g_60_l787_787420

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_60 (h1 : ∀ x y : ℝ, 0 < x → 0 < y → g (x * y) = g x / y)
                  (h2 : g 40 = 25) : g 60 = 50 / 3 :=
begin
  sorry
end

end find_g_60_l787_787420


namespace frogs_needing_new_pond_l787_787887

theorem frogs_needing_new_pond : 
    (initial_frogs tadpoles_survival_fraction pond_capacity : ℕ) 
    (h1 : initial_frogs = 5) 
    (h2 : tadpoles_survival_fraction = 2 / 3) 
    (h3 : pond_capacity = 8) 
    : (initial_frogs + tadpoles_survival_fraction * 3 * initial_frogs - pond_capacity = 7) :=
by
    sorry

end frogs_needing_new_pond_l787_787887


namespace day_after_march_20_2005_l787_787843

theorem day_after_march_20_2005 (n : ℕ) (hn : n = 2005^2) (start_day : string) (hs : start_day = "Sunday") : 
  let final_day := if (n % 7 = 0) then "Sunday"
                   else if (n % 7 = 1) then "Monday"
                   else if (n % 7 = 2) then "Tuesday"
                   else if (n % 7 = 3) then "Wednesday"
                   else if (n % 7 = 4) then "Thursday"
                   else if (n % 7 = 5) then "Friday"
                   else "Saturday"
  in final_day = "Saturday" := 
by
  sorry

end day_after_march_20_2005_l787_787843


namespace find_a_g_increasing_find_m_l787_787652

-- Condition 1: Given the function f(x) = 2^x, and f(a+2) = 8, prove a = 1.
theorem find_a (a : ℝ) (h : 2 ^ (a + 2) = 8) : a = 1 :=
sorry

-- Condition 2: Let the function g(x) = 1 - 2 / (1 + 2^x), prove g(x) is increasing.
theorem g_increasing : ∀ x y : ℝ, x < y → (1 - 2 / (1 + 2^x)) < (1 - 2 / (1 + 2^y)) :=
sorry

-- Condition 3: Given the function h(x) = m * exp(x) + exp(2 * x), and 
-- the minimum value of h(x) on the interval [0, ln 2] is 0, prove m = -1
theorem find_m (m : ℝ) (hmin : ∀ x ∈ set.Icc (0 : ℝ) (Real.log 2), m * (Real.exp x) + (Real.exp (2 * x)) ≥ 0)
  (h0 : ∃ x ∈ set.Icc (0 : ℝ) (Real.log 2), m * (Real.exp x) + (Real.exp (2 * x)) = 0) : m = -1 :=
sorry

end find_a_g_increasing_find_m_l787_787652


namespace solve_trig_identity_l787_787794

theorem solve_trig_identity (x y z : ℝ) (h1 : cos x ≠ 0) (h2 : cos y ≠ 0)
  (h_eq : (cos x ^ 2 + 1 / (cos x ^ 2)) ^ 3 + (cos y ^ 2 + 1 / (cos y ^ 2)) ^ 3 = 16 * sin z) :
  ∃ (n k m : ℤ), x = n * π ∧ y = k * π ∧ z = π / 2 + 2 * m * π :=
by
  sorry

end solve_trig_identity_l787_787794


namespace radius_of_larger_circle_theorem_l787_787244

noncomputable def radius_of_larger_circle (r : ℝ) : ℝ := 2 * (1 + real.sqrt 2)

theorem radius_of_larger_circle_theorem :
  ∀ (r : ℝ), (∀ C₁ C₂ C₃ C₄ : { c : ℝ × ℝ // (c.1 ^ 2 + c.2 ^ 2) = r ^ 2 },
                  ∀ L : { c : ℝ × ℝ // (c.1 ^ 2 + c.2 ^ 2) = (radius_of_larger_circle r) ^ 2 },
                      dist (C₁ : ℝ × ℝ) (C₂ : ℝ × ℝ) = 2 * r ∧
                      dist (C₂ : ℝ × ℝ) (C₃ : ℝ × ℝ) = 2 * r ∧
                      dist (C₃ : ℝ × ℝ) (C₄ : ℝ × ℝ) = 2 * r ∧
                      dist (C₄ : ℝ × ℝ) (C₁ : ℝ × ℝ) = 2 * r ∧
                      dist (C₁ : ℝ × ℝ) (L : ℝ × ℝ) = radius_of_larger_circle r - r ∧
                      dist (C₂ : ℝ × ℝ) (L : ℝ × ℝ) = radius_of_larger_circle r - r ∧
                      dist (C₃ : ℝ × ℝ) (L : ℝ × ℝ) = radius_of_larger_circle r - r ∧
                      dist (C₄ : ℝ × ℝ) (L : ℝ × ℝ) = radius_of_larger_circle r - r)
      → radius_of_larger_circle r = 2 * (1 + real.sqrt 2) :=
by 
  sorry

end radius_of_larger_circle_theorem_l787_787244


namespace coeff_conditions_l787_787758

noncomputable def polyWithRoots : Polynomial ℚ :=
  Polynomial.C 1 * (Polynomial.X - 0) * (Polynomial.X - 1) * (Polynomial.X + 1) * (Polynomial.X - 2) * (Polynomial.X + 2)

theorem coeff_conditions :
  ∃ (a b c d e : ℚ), 
    (Q(x) = x^5 + ax^4 + bx^3 + cx^2 + dx + e) ∧
    Q(x) = polyWithRoots ∧ 
    (a = 0) ∧ (b = 0) ∧ (e = 0) ∧ 
    (c ≠ 0) ∧ (d ≠ 0) :=
by
  sorry

end coeff_conditions_l787_787758


namespace pandas_arrangement_correct_l787_787028

def arrange_pandas_ways : ℕ := 1440

theorem pandas_arrangement_correct (pandas : Finset ℕ) (h₁ : pandas.card = 9)
  (h₂ : ∀ a b c : ℕ, a ∈ pandas → b ∈ pandas → c ∈ pandas → 
    (a < b ∧ b < c → a + b + c = (pandas.min' (by norm_num)) + (pandas.max' (by norm_num))) → 
    increment_height_order :
  pandas.erase (pandas.min' (by norm_num)).erase (pandas.max' (by norm_num)).card = 6 ) :
  arrange_pandas_ways = 2 * nat.factorial 6 := sorry

end pandas_arrangement_correct_l787_787028


namespace find_integer_n_in_range_l787_787228

theorem find_integer_n_in_range :
  ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 1997 ∧ (2^n + 2) % n = 0 ∧ n = 946 :=
begin
  use 946,
  split, exact le_of_eq (rfl : 946 = 946),
  split, exact le_of_eq (rfl : 946 = 946),
  split, exact nat.dvd_refl 946,
  refl,
end

end find_integer_n_in_range_l787_787228


namespace root_power_eq_l787_787649

theorem root_power_eq
    (a m : ℝ)
    (h_eq : ∀ x : ℝ, x^2 - (a^2 - 2) * x - 1 = 0 → x = -1 ∨ x = m) 
    (h_prod : -1 * m = -1) :
    a^m = ± real.sqrt 2 :=
by
  sorry

end root_power_eq_l787_787649


namespace maximum_obtuse_triangles_from_4_points_l787_787184

-- Define the statement such that the problem condition is accounted for.
theorem maximum_obtuse_triangles_from_4_points :
  ∀ (A B C D : ℝ × ℝ),
  (∃ (t1 t2 t3 t4: Triangle),
    (t1.vertices = {A, B, C} ∨ t1.vertices = {A, B, D} ∨ t1.vertices = {A, C, D} ∨ t1.vertices = {B, C, D}) ∧
    (t2.vertices = {A, B, C} ∨ t2.vertices = {A, B, D} ∨ t2.vertices = {A, C, D} ∨ t2.vertices = {B, C, D}) ∧
    (t3.vertices = {A, B, C} ∨ t3.vertices = {A, B, D} ∨ t3.vertices = {A, C, D} ∨ t3.vertices = {B, C, D}) ∧
    (t4.vertices = {A, B, C} ∨ t4.vertices = {A, B, D} ∨ t4.vertices = {A, C, D} ∨ t4.vertices = {B, C, D}) ∧
  count_obtuse_triangles t1 t2 t3 t4 = 3) :=
sorry

end maximum_obtuse_triangles_from_4_points_l787_787184


namespace coffee_bag_contains_10_5_ounces_l787_787371

-- Variables representing the given conditions
variables
  (cups_per_day : ℕ)
  (ounces_per_cup : ℝ)
  (bag_cost : ℝ)
  (milk_per_week_gallons : ℝ)
  (milk_cost_per_gallon : ℝ)
  (total_weekly_cost : ℝ)

-- Given values from the problem conditions
def cups_per_day := 2
def ounces_per_cup := 1.5
def bag_cost := 8
def milk_per_week_gallons := 0.5
def milk_cost_per_gallon := 4
def total_weekly_cost := 18

-- Theorem to prove the number of ounces in a bag of coffee is 10.5
theorem coffee_bag_contains_10_5_ounces :
  (bag_cost / ((total_weekly_cost - (milk_per_week_gallons * milk_cost_per_gallon)) / 
  (cups_per_day * ounces_per_cup * 7))) = 10.5 :=
begin
  -- skipping the proof
  sorry
end

end coffee_bag_contains_10_5_ounces_l787_787371


namespace number_of_x_eq_3_l787_787759

def S := {A, A1, A2, A3, A4, A5}

def op (x y : S) : S :=
  match (x, y) with
  | (A, A) => A
  | (A, A1) => A1
  | (A, A2) => A2
  | (A, A3) => A3
  | (A, A4) => A4
  | (A, A5) => A5
  | (A1, A1) => match 2 % 4 with
                | 0 => A
                | 1 => A1
                | 2 => A2
                | 3 => A3
  | (A1, A2) => match 3 % 4 with
                | 0 => A
                | 1 => A1
                | 2 => A2
                | 3 => A3
  | (A1, A3) => match 4 % 4 with
                | 0 => A
                | 1 => A1
                | 2 => A2
                | 3 => A3
  | (A1, A4) => match 5 % 4 with
                | 0 => A
                | 1 => A1
                | 2 => A2
                | 3 => A3
  | (A1, A5) => match 6 % 4 with
                | 0 => A
                | 1 => A1
                | 2 => A2
                | 3 => A3
  -- similarly for other combinations

theorem number_of_x_eq_3 : (# S.filter (λ x, (op (op x x) A2 = A))) = 3 := by
  sorry

end number_of_x_eq_3_l787_787759


namespace citrus_grove_total_orchards_l787_787509

theorem citrus_grove_total_orchards (lemons_orchards oranges_orchards grapefruits_orchards limes_orchards total_orchards : ℕ) 
  (h1 : lemons_orchards = 8) 
  (h2 : oranges_orchards = lemons_orchards / 2) 
  (h3 : grapefruits_orchards = 2) 
  (h4 : limes_orchards = grapefruits_orchards) 
  (h5 : total_orchards = lemons_orchards + oranges_orchards + grapefruits_orchards + limes_orchards) : 
  total_orchards = 16 :=
by 
  sorry

end citrus_grove_total_orchards_l787_787509


namespace area_of_square_l787_787143

theorem area_of_square (side_length : ℕ) (h : side_length = 8) : side_length * side_length = 64 :=
by
  have h' : side_length = 8 := h
  rw h'
  sorry  -- Proof goes here

end area_of_square_l787_787143


namespace min_m_even_function_l787_787844

theorem min_m_even_function :
  ∃ m > 0, ∀ x : ℝ, sin(3 * (x + m) + π / 4) = sin(3 * (-x - m) + π / 4) ∧ m = π / 12 :=
sorry

end min_m_even_function_l787_787844


namespace trig_solve_identity_l787_787792

theorem trig_solve_identity {x y z : ℝ} (hx : cos x ≠ 0) (hy : cos y ≠ 0) :
  (cos x ^ 2 + 1 / cos x ^ 2) ^ 3 + (cos y ^ 2 + 1 / cos y ^ 2) ^ 3 = 16 * sin z ↔
  (∃ n k m : ℤ, x = n * Real.pi ∧ y = k * Real.pi ∧ z = Real.pi / 2 + 2 * m * Real.pi) :=
by
  sorry

end trig_solve_identity_l787_787792


namespace train_passes_man_in_approx_18_seconds_l787_787489

-- Definitions for the conditions
def train_length : ℝ := 330 -- in meters
def train_speed : ℝ := 60   -- in km/hr
def man_speed : ℝ := 6      -- in km/hr

-- Relative speed calculation
def relative_speed_kmhr : ℝ := train_speed + man_speed 
def relative_speed_ms : ℝ := relative_speed_kmhr * 1000 / 3600

-- Time calculation
def time_to_pass := train_length / relative_speed_ms

-- The proof statement
theorem train_passes_man_in_approx_18_seconds :
  abs (time_to_pass - 18) < 1 :=
sorry

end train_passes_man_in_approx_18_seconds_l787_787489


namespace smallest_number_of_cubes_l787_787874

theorem smallest_number_of_cubes : 
    ∃ (n : ℕ), n = 56 ∧ 
    ∃ (side_length : ℕ), side_length > 0 ∧ 
    (35 % side_length = 0) ∧ (20 % side_length = 0) ∧ (10 % side_length = 0) ∧ 
    (35 / side_length) * (20 / side_length) * (10 / side_length) = n :=
begin
    sorry
end

end smallest_number_of_cubes_l787_787874


namespace correct_calculation_l787_787135

variable (n : ℕ)
variable (h1 : 63 + n = 70)

theorem correct_calculation : 36 * n = 252 :=
by
  -- Here we will need the Lean proof, which we skip using sorry
  sorry

end correct_calculation_l787_787135


namespace polyhedra_share_interior_point_l787_787570

theorem polyhedra_share_interior_point
  (A : Fin₉ →  ℝ × ℝ × ℝ) -- 9 vertices
  (P1 : Set (ℝ × ℝ × ℝ)) -- Initial polyhedron P1
  (P1_convex : Convex ℝ P1) -- P1 is convex
  (vertices_P1 : ∀ i : Fin₉, A i ∈ P1)  -- Vertices of P1 are in P1
  (P_i : Fin₉ → Set (ℝ × ℝ × ℝ)) -- Translated polyhedra P_i
  (P_i_translation : ∀ i : Fin₉, ∃ t : ℝ × ℝ × ℝ, P_i i = { p | ∃ q ∈ P1, p = (q.1 + t.1, q.2 + t.2, q.3 + t.3) })  -- P_i are translated versions of P1
  (vertices_P_i : ∀ i : Fin₉, vertices_P1 i = vertices_P1 0 + A i - A 0) -- Translation property
  (volume_properties : ∑ i, volume (P_i i) = 9 * volume P1) -- Volume property
  (P_containment : ∀ i, P_i i ⊆ P1) -- Containment property
: ∃ i j : Fin₉, i ≠ j ∧ ∃ x : ℝ × ℝ × ℝ, x ∈ interior (P_i i) ∧ x ∈ interior (P_i j) := 
sorry

end polyhedra_share_interior_point_l787_787570


namespace scientific_notation_of_300_million_l787_787833

theorem scientific_notation_of_300_million : 
  300000000 = 3 * 10^8 := 
by
  sorry

end scientific_notation_of_300_million_l787_787833


namespace projections_equal_l787_787384

noncomputable def point := ℝ × ℝ

noncomputable def midpoint (A B : point) : point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def is_diameter (O A C : point) : Prop := dist O A = dist O C

def perpendicular (A B C : point) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  AB.1 * BC.1 + AB.2 * BC.2 = 0

theorem projections_equal (A B C D O AA1 CC1 P : point)
  (h_circ : is_diameter O A C)
  (h_perp1 : perpendicular A AA1 B)
  (h_perp2 : perpendicular C CC1 B)
  (h_perp_O : perpendicular O P B)
  (h_mid_O : O = midpoint A C)
  (h_mid_P : P = midpoint B D) :
  dist A AA1 = dist C CC1 := sorry

end projections_equal_l787_787384


namespace shop_earnings_l787_787700

def cola_price := 3
def juice_price := 1.5
def water_price := 1
def sports_drink_price := 2.5
def cola_sold := 18
def juice_sold := 15
def water_sold := 30
def sports_drinks_sold_paid := 22

def total_earnings : ℝ :=
  cola_sold * cola_price +
  juice_sold * juice_price +
  water_sold * water_price +
  sports_drinks_sold_paid * sports_drink_price

theorem shop_earnings : total_earnings = 161.5 :=
  by
  -- Proof (steps omitted)
  sorry

end shop_earnings_l787_787700


namespace sandy_money_left_l787_787777

theorem sandy_money_left (total_money : ℝ) (spent_percentage : ℝ) (money_left : ℝ) : 
  total_money = 320 → spent_percentage = 0.30 → money_left = (total_money * (1 - spent_percentage)) → 
  money_left = 224 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end sandy_money_left_l787_787777


namespace option_D_correct_l787_787480

theorem option_D_correct (x : ℝ) : 2 * x^2 * (3 * x)^2 = 18 * x^4 :=
by sorry

end option_D_correct_l787_787480


namespace geometric_series_sum_l787_787199

theorem geometric_series_sum : 
  let a0 := (1 / 2 : ℝ)
  let r := (1 / 2 : ℝ) 
  let n := 5
  (∑ i in finset.range (n + 1), a0 * r ^ i) = 31 / 32 :=
by
  let a0 := (1 / 2 : ℝ)
  let r := (1 / 2 : ℝ) 
  let n := 5
  let finite_geo_sum : ℝ := ∑ i in finset.range (n + 1), a0 * r ^ i
  have h_finite_geo_sum : finite_geo_sum = a0 * (1 - r^n) / (1 - r) := sorry
  rw [h_finite_geo_sum]
  have h_geometric_sum : a0 * (1 - r^n) / (1 - r) = 31 / 32 := sorry
  exact h_geometric_sum

end geometric_series_sum_l787_787199


namespace convert_to_polar_l787_787941

theorem convert_to_polar (x y : ℝ) (hx : x = Real.sqrt 2) (hy : y = -Real.sqrt 2) :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2*Real.pi ∧ r = 2 ∧ θ = 7*Real.pi/4 := by
  use [2, 7*Real.pi/4]
  split
  · exact zero_lt_two
  split
  · exact zero_le (7*Real.pi/4)
  split
  · exact div_lt_self' seven_ne_zero three_ne_zero two_pos
  split
  · sorry
  · sorry

end convert_to_polar_l787_787941


namespace janice_bought_30_fifty_cent_items_l787_787725

theorem janice_bought_30_fifty_cent_items (x y z : ℕ) (h1 : x + y + z = 40) (h2 : 50 * x + 150 * y + 300 * z = 4500) : x = 30 :=
by
  sorry

end janice_bought_30_fifty_cent_items_l787_787725


namespace initial_number_of_men_l787_787079

theorem initial_number_of_men (x : ℕ) :
    (50 * x = 25 * (x + 20)) → x = 20 := 
by
  sorry

end initial_number_of_men_l787_787079


namespace red_area_percentage_l787_787405

-- Definitions based on conditions in a)
variable (s : ℝ) (r : ℝ)
def flag_area : ℝ := s ^ 2
def cross_red_area : ℝ := 0.25 * flag_area s

-- The formulation of the problem statement
theorem red_area_percentage (h : 4 * real.pi * r ^ 2 = cross_red_area s) :
  100 * (4 * real.pi * r ^ 2 / flag_area s) = 25 :=
by
  sorry

end red_area_percentage_l787_787405


namespace total_marbles_is_260_l787_787213

-- Define the number of marbles in each jar based on the conditions.
def first_jar : Nat := 80
def second_jar : Nat := 2 * first_jar
def third_jar : Nat := (1 / 4 : ℚ) * first_jar

-- Prove that the total number of marbles is 260.
theorem total_marbles_is_260 : first_jar + second_jar + third_jar = 260 := by
  sorry

end total_marbles_is_260_l787_787213


namespace sum_of_values_l787_787081

def p (x : ℝ) : ℝ := abs x - 2
def q (x : ℝ) : ℝ := -abs x

def evaluated_values := list.map (λ x, q (p x)) [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_values : list.sum evaluated_values = -10 := by sorry

end sum_of_values_l787_787081


namespace count_of_random_events_is_three_l787_787917

-- Define each condition as predicates
def is_random_event1 : Prop := ∃ ω, ω ∈ {outcome | outcome = 2} 
def is_random_event2 : Prop := false
def is_random_event3 : Prop := ∃ ω, ω ∈ {win, lose}
def is_random_event4 : Prop := ∃ ω, ω ∈ {boy, girl}
def is_random_event5 : Prop := false

-- Define the number of random events
def number_of_random_events (events : List Prop) : ℕ :=
  events.count (λ e, e = true)

-- Define the list of events
def events : List Prop := [is_random_event1, is_random_event2, is_random_event3, is_random_event4, is_random_event5]

-- Prove that the number of random events is 3
theorem count_of_random_events_is_three : number_of_random_events events = 3 := 
  sorry

end count_of_random_events_is_three_l787_787917


namespace total_apple_trees_l787_787927

-- Definitions and conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3
def total_trees : ℕ := ava_trees + lily_trees

-- Statement to be proved
theorem total_apple_trees :
  total_trees = 15 := by
  sorry

end total_apple_trees_l787_787927


namespace acid_solution_mixture_l787_787523

def final_acid_percentage (v1 v2 : ℝ) (c1 c2 : ℝ) (v_total : ℝ) : ℝ :=
  ((v1 * c1) + (v2 * c2)) / v_total * 100

theorem acid_solution_mixture :
  final_acid_percentage 420 210 0.6 0.3 630 = 50 := by
  sorry

end acid_solution_mixture_l787_787523


namespace percentage_of_items_sold_l787_787173

theorem percentage_of_items_sold (total_items price_per_item discount_rate debt creditors_balance remaining_balance : ℕ)
  (H1 : total_items = 2000)
  (H2 : price_per_item = 50)
  (H3 : discount_rate = 80)
  (H4 : debt = 15000)
  (H5 : remaining_balance = 3000) :
  (total_items * (price_per_item - (price_per_item * discount_rate / 100)) + remaining_balance = debt + remaining_balance) →
  (remaining_balance / (price_per_item - (price_per_item * discount_rate / 100)) / total_items * 100 = 90) :=
by
  sorry

end percentage_of_items_sold_l787_787173


namespace problem1_l787_787494

noncomputable def sqrt7_minus_1_pow_0 : ℝ := (Real.sqrt 7 - 1)^0
noncomputable def minus_half_pow_neg_2 : ℝ := (-1 / 2)^(-2 : ℤ)
noncomputable def sqrt3_tan_30 : ℝ := Real.sqrt 3 * Real.tan (Real.pi / 6)

theorem problem1 : sqrt7_minus_1_pow_0 - minus_half_pow_neg_2 + sqrt3_tan_30 = -2 := by
  sorry

end problem1_l787_787494


namespace project_choice_ways_l787_787493

-- Definitions of students and projects
inductive Student
| A | B | C | D

inductive Project
| CreativePottery | Rubbings | TieDyeing | WallHangings | PaperCutting

-- The main theorem statement
theorem project_choice_ways :
  let students := [Student.A, Student.B, Student.C, Student.D]
  let projects := [Project.CreativePottery, Project.Rubbings, Project.TieDyeing, Project.WallHangings, Project.PaperCutting]
  (number of ways for at least three different project choices for students) = 480 :=
sorry

end project_choice_ways_l787_787493


namespace soda_left_is_70_percent_l787_787576

def bottles_left_percentage (initial_bottles: ℕ) (drank_percentage: ℝ) (given_percentage: ℝ) : ℝ :=
  let drank_amount := 1 * drank_percentage
  let given_amount := 2 * given_percentage
  let remaining_bottles := initial_bottles - drank_amount - given_amount
  remaining_bottles * 100

theorem soda_left_is_70_percent :
  ∀ (initial_bottles: ℕ) (drank_percentage: ℝ) (given_percentage: ℝ),
    initial_bottles = 3 → drank_percentage = 0.9 → given_percentage = 0.7 →
    bottles_left_percentage initial_bottles drank_percentage given_percentage = 70 :=
begin
  intros initial_bottles drank_percentage given_percentage h1 h2 h3,
  simp [bottles_left_percentage, h1, h2, h3],
  norm_num,
end

end soda_left_is_70_percent_l787_787576


namespace proof_problem_l787_787425

def f (x a : ℝ) : ℝ := 1 / (2^x + 1) + a
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem proof_problem :
  (¬ ∃ a : ℝ, is_odd (λ x, f x a) ∧ a = 1/2) ∧
  (¬ (∀ Δ : Triangle, (Δ.A > Δ.B → Δ.sinA > Δ.sinB) → (Δ.A < Δ.B → Δ.sinA < Δ.sinB))) ∧
  ((∀ a b c : ℝ, (a ≠ 0) → (b ≠ 0) → (c ≠ 0) → (b^2 = a * c ↔ b = √(a * c))) ∧
  (∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0)) → 
  2 := sorry

end proof_problem_l787_787425


namespace total_wage_l787_787502

theorem total_wage (work_days_A work_days_B : ℕ) (wage_A : ℕ) (total_wage : ℕ) 
  (h1 : work_days_A = 10) 
  (h2 : work_days_B = 15) 
  (h3 : wage_A = 1980)
  (h4 : (wage_A / (wage_A / (total_wage * 3 / 5))) = 3)
  : total_wage = 3300 :=
sorry

end total_wage_l787_787502


namespace management_sampled_count_l787_787508

variable (total_employees salespeople management_personnel logistical_support staff_sample_size : ℕ)
variable (proportional_sampling : Prop)
variable (n_management_sampled : ℕ)

axiom h1 : total_employees = 160
axiom h2 : salespeople = 104
axiom h3 : management_personnel = 32
axiom h4 : logistical_support = 24
axiom h5 : proportional_sampling
axiom h6 : staff_sample_size = 20

theorem management_sampled_count : n_management_sampled = 4 :=
by
  -- The proof is omitted as per instructions
  sorry

end management_sampled_count_l787_787508


namespace cost_price_per_meter_l787_787912

def selling_price_for_85_meters : ℝ := 8925
def profit_per_meter : ℝ := 25
def number_of_meters : ℝ := 85

theorem cost_price_per_meter : (selling_price_for_85_meters - profit_per_meter * number_of_meters) / number_of_meters = 80 := by
  sorry

end cost_price_per_meter_l787_787912


namespace area_doubled_l787_787620

open EuclideanGeometry

theorem area_doubled (A B C D P A' B' C' D' : Point) 
  (hA' : Segment P A' ≃ Segment A B ∧ Parallel (Segment P A') (Segment A B))
  (hB' : Segment P B' ≃ Segment B C ∧ Parallel (Segment P B') (Segment B C))
  (hC' : Segment P C' ≃ Segment C D ∧ Parallel (Segment P C') (Segment C D))
  (hD' : Segment P D' ≃ Segment D A ∧ Parallel (Segment P D') (Segment D A)) :
  area (Quadrilateral A' B' C' D') = 2 * area (Quadrilateral A B C D) :=
sorry

end area_doubled_l787_787620


namespace choir_members_count_l787_787807

theorem choir_members_count (n : ℕ) 
  (h1 : 150 < n) 
  (h2 : n < 300) 
  (h3 : n % 6 = 1) 
  (h4 : n % 8 = 3) 
  (h5 : n % 9 = 2) : 
  n = 163 :=
sorry

end choir_members_count_l787_787807


namespace painting_prices_l787_787775

theorem painting_prices (P : ℝ) (h₀ : 55000 = 3.5 * P - 500) : 
  P = 15857.14 :=
by
  -- P represents the average price of the previous three paintings.
  -- Given the condition: 55000 = 3.5 * P - 500
  -- We need to prove: P = 15857.14
  sorry

end painting_prices_l787_787775


namespace triangle_similar_l787_787401

-- Definitions of the problem elements and conditions.
def angle_complementary (angle1 angle2 : ℝ) : Prop :=
  angle1 + angle2 = 90

def inscribable_quadrilateral (A B C D : Point) : Prop :=
  ∃ (circum : Circle), A ∈ circum ∧ B ∈ circum ∧ C ∈ circum ∧ D ∈ circum

def triangle_similarity (A B C D E F : Point) : Prop :=
  ∠FBE = ∠DCB ∧ ∠FEB = ∠BD ∧ ∠EFC = ∠BCD

-- Stating the theorem where we establish the similarity between the two triangles.
theorem triangle_similar (A B C D E F G: Point) 
  (h1 : ∠FBE = 90 - ∠FEB)
  (h2 : ∠DCE = 90 - ∠DCB)
  (h3 : ∠FEB = ∠BDC)
  (h4 : inscribable_quadrilateral B C D E)
  (h5 : ∠BEC = 90)
  : triangle_similarity B E F B C D :=
by
  sorry

end triangle_similar_l787_787401


namespace θ_plus_φ_eq_7_π_over_12_l787_787740

noncomputable def f (x θ φ : ℝ) : ℝ := cos (x + θ) + sqrt 2 * sin (x + φ)

theorem θ_plus_φ_eq_7_π_over_12 
  (θ φ : ℝ)
  (h1 : θ > 0 ∧ θ < π / 2)
  (h2 : φ > 0 ∧ φ < π / 2)
  (h3 : ∀ x, f x θ φ = f (-x) θ φ)
  (h4 : cos θ = (sqrt 6 / 3) * sin φ) :
  θ + φ = 7 * π / 12 :=
sorry

end θ_plus_φ_eq_7_π_over_12_l787_787740


namespace scientific_notation_of_384000_l787_787381

theorem scientific_notation_of_384000 : 384000 = 3.84 * 10^5 :=
by
  sorry

end scientific_notation_of_384000_l787_787381


namespace gondor_laptops_wednesday_l787_787664

/-- Gondor's phone repair earnings per unit -/
def phone_earning : ℕ := 10

/-- Gondor's laptop repair earnings per unit -/
def laptop_earning : ℕ := 20

/-- Number of phones repaired on Monday -/
def phones_monday : ℕ := 3

/-- Number of phones repaired on Tuesday -/
def phones_tuesday : ℕ := 5

/-- Number of laptops repaired on Thursday -/
def laptops_thursday : ℕ := 4

/-- Total earnings of Gondor -/
def total_earnings : ℕ := 200

/-- Number of laptops repaired on Wednesday, which we need to prove equals 2 -/
def laptops_wednesday : ℕ := 2

theorem gondor_laptops_wednesday : 
    (phones_monday * phone_earning + phones_tuesday * phone_earning + 
    laptops_thursday * laptop_earning + laptops_wednesday * laptop_earning = total_earnings) :=
by
    sorry

end gondor_laptops_wednesday_l787_787664


namespace candy_distribution_l787_787408

theorem candy_distribution (candies : Finset ℕ) (bags : Finset ℕ) (h : candies.card = 10) (h_bags : bags.card = 4) :
  (∃ f : candies → bags, (∀ b ∈ bags, ∃ c ∈ candies, f c = b) ∧ (f.cases_on _ _ _ _ _ _ _).card = 20643840) := sorry

end candy_distribution_l787_787408


namespace famous_figures_mathematicians_l787_787918

-- List of figures encoded as integers for simplicity
def Bill_Gates := 1
def Gauss := 2
def Liu_Xiang := 3
def Nobel := 4
def Chen_Jingrun := 5
def Chen_Xingshen := 6
def Gorky := 7
def Einstein := 8

-- Set of mathematicians encoded as a set of integers
def mathematicians : Set ℕ := {2, 5, 6}

-- Correct answer set
def correct_answer_set : Set ℕ := {2, 5, 6}

-- The statement to prove
theorem famous_figures_mathematicians:
  mathematicians = correct_answer_set :=
by sorry

end famous_figures_mathematicians_l787_787918


namespace total_pets_combined_l787_787803

def teddy_dogs : ℕ := 7
def teddy_cats : ℕ := 8
def ben_dogs : ℕ := teddy_dogs + 9
def dave_cats : ℕ := teddy_cats + 13
def dave_dogs : ℕ := teddy_dogs - 5

def teddy_pets : ℕ := teddy_dogs + teddy_cats
def ben_pets : ℕ := ben_dogs
def dave_pets : ℕ := dave_cats + dave_dogs

def total_pets : ℕ := teddy_pets + ben_pets + dave_pets

theorem total_pets_combined : total_pets = 54 :=
by
  -- proof goes here
  sorry

end total_pets_combined_l787_787803


namespace num_imaginary_complex_numbers_l787_787056

theorem num_imaginary_complex_numbers : 
  let S := {0, 1, 2, 3, 4, 5}
  in let count_imaginary := (S.filter (λ b, b ≠ 0)).card * 5
  in count_imaginary = 25 :=
by {
  sorry
}

end num_imaginary_complex_numbers_l787_787056


namespace oliver_cards_l787_787380

variable {MC AB BG : ℕ}

theorem oliver_cards : 
  (BG = 48) → 
  (BG = 3 * AB) → 
  (MC = 2 * AB) → 
  MC = 32 := 
by 
  intros h1 h2 h3
  sorry

end oliver_cards_l787_787380


namespace value_added_to_075_of_number_l787_787436

theorem value_added_to_075_of_number (N V : ℝ) (h1 : 0.75 * N + V = 8) (h2 : N = 8) : V = 2 := by
  sorry

end value_added_to_075_of_number_l787_787436


namespace order_of_logs_l787_787248

theorem order_of_logs (a b c : ℝ) (ha : a = Real.logBase 4 Real.e)
  (hb : b = Real.logBase 3 4) (hc : c = Real.logBase 4 5) : a < c ∧ c < b :=
by
  sorry

end order_of_logs_l787_787248


namespace license_plate_count_l787_787466

def rotokas_alphabet := {'A', 'E', 'G', 'I', 'K', 'O', 'P', 'R', 'S', 'T', 'U', 'V'}

def valid_license_plate (plate : List Char) : Bool :=
  plate.length = 6 ∧
  (plate.head? = some 'A' ∨ plate.head? = some 'E') ∧
  plate.ilast? = some 'R' ∧
  ('V' ∉ plate) ∧
  (plate.nodup)

theorem license_plate_count (plates : List (List Char)) :
  (∀ p ∈ plates, valid_license_plate p) →
  plates.length = 6048 :=
sorry

end license_plate_count_l787_787466


namespace raisin_cookie_difference_l787_787667

variable (baked_yesterday : ℕ)
variable (baked_today : ℕ)

theorem raisin_cookie_difference : baked_yesterday = 300 → baked_today = 280 → baked_yesterday - baked_today = 20 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end raisin_cookie_difference_l787_787667


namespace blue_stripe_area_l787_787882

def cylinder_diameter : ℝ := 20
def cylinder_height : ℝ := 60
def stripe_width : ℝ := 4
def stripe_revolutions : ℕ := 3

theorem blue_stripe_area : 
  let circumference := Real.pi * cylinder_diameter
  let stripe_length := stripe_revolutions * circumference
  let expected_area := stripe_width * stripe_length
  expected_area = 240 * Real.pi :=
by
  sorry

end blue_stripe_area_l787_787882


namespace probability_difference_3_or_greater_l787_787457

noncomputable def set_of_numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def total_combinations : ℕ := (finset.image (λ (p : finset ℕ), p) (finset.powerset (finset.univ ∩ set_of_numbers))).card / 2

def pairs_with_difference_less_than_3 (S : finset ℕ) : finset (finset ℕ) :=
finset.filter (λ (p : finset ℕ), p.card = 2 ∧ abs ((p.min' sorry) - (p.max' sorry)) < 3) S

def successful_pairs (S : finset ℕ) : finset (finset ℕ) :=
finset.filter (λ (p : finset ℕ), p.card = 2 ∧ abs ((p.min' sorry) - (p.max' sorry)) ≥ 3) S
   
def probability (S : finset ℕ) : ℚ :=
(successful_pairs S).card.to_rat / total_combinations

theorem probability_difference_3_or_greater :
  probability set_of_numbers = 7/12 :=
sorry

end probability_difference_3_or_greater_l787_787457


namespace intersection_S_T_l787_787285

noncomputable def S : set ℝ := {x | x^2 + 2 * x = 0}
noncomputable def T : set ℝ := {x | x^2 - 2 * x = 0}

theorem intersection_S_T :
  (S ∩ T) = {0} :=
sorry

end intersection_S_T_l787_787285


namespace a_general_form_l787_787249

noncomputable def F : ℕ → ℝ
| 0 := 0
| 1 := 1
| n + 2 := F n + F (n + 1)

noncomputable def arccosh (x : ℝ) : ℝ := Real.arccosh x -- Hyperbolic arccosine

constant a : ℕ → ℝ
axiom a_1 : a 1 = 97
axiom a_2 : a 2 = 97
axiom recurrence_relation : ∀ n > 1, a (n + 1) = a n * a (n - 1) + sqrt ((a n ^ 2 - 1) * (a (n - 1) ^ 2 - 1))

theorem a_general_form (n : ℕ) (hn : n ≥ 1) : 
  a n = Real.cosh (F n * arccosh 97) :=
sorry

end a_general_form_l787_787249


namespace sequence_decreasing_l787_787102

theorem sequence_decreasing : 
  ∀ (n : ℕ), n ≥ 1 → (1 / 2^(n - 1)) > (1 / 2^n) := 
by {
  sorry
}

end sequence_decreasing_l787_787102


namespace sum_of_positive_integers_n_l787_787126

theorem sum_of_positive_integers_n
  (n : ℕ) (h1: n > 0)
  (h2 : Nat.lcm n 100 = Nat.gcd n 100 + 300) :
  n = 350 :=
sorry

end sum_of_positive_integers_n_l787_787126


namespace total_albums_l787_787036

theorem total_albums (Adele Bridget Katrina Miriam : ℕ) 
  (h₁ : Adele = 30)
  (h₂ : Bridget = Adele - 15)
  (h₃ : Katrina = 6 * Bridget)
  (h₄ : Miriam = 5 * Katrina) : Adele + Bridget + Katrina + Miriam = 585 := 
by
  sorry

end total_albums_l787_787036


namespace sin_intersections_ratio_l787_787809

theorem sin_intersections_ratio :
  ∃ p q : ℕ, (p, q) = (1, 5) ∧ gcd p q = 1 ∧ p < q ∧ 
  (∀ x y : ℝ, sin x = sin 60 ∧ sin y = sin 60 → x ≠ y → y - x = 60 ∨ y - x = 300) := 
begin
  sorry
end

end sin_intersections_ratio_l787_787809


namespace cost_of_new_shoes_l787_787507

theorem cost_of_new_shoes 
    (R : ℝ) 
    (L_r : ℝ) 
    (L_n : ℝ) 
    (increase_percent : ℝ) 
    (H_R : R = 13.50) 
    (H_L_r : L_r = 1) 
    (H_L_n : L_n = 2) 
    (H_inc_percent : increase_percent = 0.1852) : 
    2 * (R * (1 + increase_percent) / L_n) = 32.0004 := 
by
    sorry

end cost_of_new_shoes_l787_787507


namespace curve_crossing_point_l787_787555

theorem curve_crossing_point :
  (∃ t : ℝ, (t^2 - 4 = 2) ∧ (t^3 - 6 * t + 4 = 4)) ∧
  (∃ t' : ℝ, t ≠ t' ∧ (t'^2 - 4 = 2) ∧ (t'^3 - 6 * t' + 4 = 4)) :=
sorry

end curve_crossing_point_l787_787555


namespace find_k_l787_787162

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n - 2

theorem find_k :
  ∃ k : ℤ, k % 2 = 1 ∧ f (f (f k)) = 35 ∧ k = 29 := 
sorry

end find_k_l787_787162


namespace polynomial_evaluation_l787_787597

noncomputable def evaluate_polynomial (x : ℝ) : ℝ :=
  x^3 - 3 * x^2 - 9 * x + 5

theorem polynomial_evaluation (x : ℝ) (h_pos : x > 0) (h_eq : x^2 - 3 * x - 9 = 0) :
  evaluate_polynomial x = 5 :=
by
  sorry

end polynomial_evaluation_l787_787597


namespace odd_prime_divisor_l787_787281

theorem odd_prime_divisor (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n < m)
  (h4 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 → 
    ∃ x_k : ℕ, x_k = (m + k) / (n + k) ∧ (m + k) % (n + k) = 0) :
  ∃ p : ℕ, prime p ∧ p % 2 = 1 ∧ p ∣ (∏ k in finset.range (n + 2), ((m + (k + 1)) / (n + (k + 1)))) - 1 :=
sorry

end odd_prime_divisor_l787_787281


namespace equilateral_triangle_l787_787400

theorem equilateral_triangle (x : ℝ) (h : 3 * tan x - 3 * tan(x / 2) - 2 * sqrt 3 = 0) : x = π / 3 :=
sorry

end equilateral_triangle_l787_787400


namespace range_of_distances_l787_787746

theorem range_of_distances (t θ : ℝ)
    (x_line y_line : ℝ := 1 - t)
    (y_line : ℝ := 8 - 2 * t)
    (x_curve y_curve : ℝ := 1 + sqrt 5 * cos θ)
    (y_curve : ℝ := -2 + sqrt 5 * sin θ) :
    set_of (λ d : ℝ, 
        ∃ P Q : ℝ × ℝ,
        P = (x_line, y_line) ∧ 
        Q = (x_curve, y_curve) ∧ 
        d = (dist P Q)
    ) = {d | d > sqrt 5} := 
sorry

end range_of_distances_l787_787746


namespace calculate_industrial_lubricants_percentage_l787_787879

-- Define the given conditions
def budget_percentages : List (String × ℕ) := [
  ("microphotonics", 14),
  ("home electronics", 24),
  ("food additives", 20),
  ("genetically modified microorganisms", 29)
]

-- Calculate the percentage for basic astrophysics based on the given degrees
def basic_astrophysics_percentage (degrees : ℕ) (total_degrees : ℕ) : ℕ :=
  (degrees * 100) / total_degrees

-- Calculate the total percentage already allocated
def total_allocated (percentages : List (String × ℕ)) : ℕ :=
  percentages.foldr (fun (_, p) acc => acc + p) 0

-- Calculate remaining percentage for industrial lubricants
def industrial_lubricants_percentage (allocated : ℕ) : ℕ :=
  100 - allocated

theorem calculate_industrial_lubricants_percentage :
  let basic_astrophysics := basic_astrophysics_percentage 18 360
  let total_allocated_budget := total_allocated budget_percentages + basic_astrophysics
  industrial_lubricants_percentage total_allocated_budget = 8 :=
by
  let basic_astrophysics := basic_astrophysics_percentage 18 360
  have basic_astrophysics_val : basic_astrophysics = 5 := by sorry
  let total_allocated_budget := total_allocated budget_percentages + basic_astrophysics
  have total_allocated_val : total_allocated_budget = 92 := by sorry
  show industrial_lubricants_percentage total_allocated_budget = 8
  simp [industrial_lubricants_percentage, total_allocated_val]
  exact eq.refl _

end calculate_industrial_lubricants_percentage_l787_787879


namespace custom_op_example_l787_787090

-- Define the custom operation ⊕
def custom_op (a b : ℝ) : ℝ := a - (a / b) + b

-- Statement to be proven
theorem custom_op_example : custom_op 9 3 = 9 := 
by
  sorry

end custom_op_example_l787_787090


namespace strawberry_jelly_sales_l787_787889

def jelly_sales (grape strawberry raspberry plum : ℕ) : Prop :=
  grape = 2 * strawberry ∧
  raspberry = 2 * plum ∧
  raspberry = grape / 3 ∧
  plum = 6

theorem strawberry_jelly_sales {grape strawberry raspberry plum : ℕ}
    (h : jelly_sales grape strawberry raspberry plum) : 
    strawberry = 18 :=
by
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2
  sorry

end strawberry_jelly_sales_l787_787889


namespace cone_height_l787_787880

theorem cone_height (V : ℝ) (π : ℝ) (r h : ℝ) (sqrt2 : ℝ) :
  V = 9720 * π →
  sqrt2 = Real.sqrt 2 →
  h = r * sqrt2 →
  V = (1/3) * π * r^2 * h →
  h = 38.7 :=
by
  intros
  sorry

end cone_height_l787_787880


namespace neg_p_l787_787282

variable {x : ℝ}

def p := ∀ x > 0, Real.sin x ≤ 1

theorem neg_p : ¬ p ↔ ∃ x > 0, Real.sin x > 1 :=
by
  sorry

end neg_p_l787_787282


namespace digit_in_decimal_expansion_l787_787479

theorem digit_in_decimal_expansion (n : ℕ) (h : n = 534) : 
  let seq := [3, 8, 4, 6, 1, 5] in
  seq[(n - 1) % seq.length] = 5 :=
by
  let seq := [3, 8, 4, 6, 1, 5]
  have h_seq_length : seq.length = 6 := rfl
  have h_mod : (n - 1) % seq.length = 5 := sorry
  rw [h_seq_length, h_mod]
  rfl

end digit_in_decimal_expansion_l787_787479


namespace geometric_progression_exists_l787_787957

theorem geometric_progression_exists :
  ∃ (b_1 b_2 b_3 b_4 q : ℚ), 
    b_1 - b_2 = 35 ∧ 
    b_3 - b_4 = 560 ∧ 
    b_2 = b_1 * q ∧ 
    b_3 = b_1 * q^2 ∧ 
    b_4 = b_1 * q^3 ∧ 
    ((b_1 = 7 ∧ q = -4 ∧ b_2 = -28 ∧ b_3 = 112 ∧ b_4 = -448) ∨ 
    (b_1 = -35/3 ∧ q = 4 ∧ b_2 = -140/3 ∧ b_3 = -560/3 ∧ b_4 = -2240/3)) :=
by
  sorry

end geometric_progression_exists_l787_787957


namespace heartsuit_xx_false_l787_787944

def heartsuit (x y : ℝ) : ℝ := |x - y|

theorem heartsuit_xx_false (x : ℝ) : heartsuit x x ≠ x :=
by sorry

end heartsuit_xx_false_l787_787944


namespace t_minus_s_l787_787529

/-
Conditions:
1. The school has 200 students.
2. The school has 6 teachers.
3. Each student attends one class.
4. Each teacher teaches one class.
5. The enrollments in the classes are 80, 40, 40, 20, 10, and 10.
6. Classes with 10 students each are special workshops that combine students from other classes.
-/
def total_students : ℕ := 200
def total_teachers : ℕ := 6
def class_sizes : List ℕ := [80, 40, 40, 20, 10, 10]

/-- 
Calculate t (average number of students per teacher):
Each of the six teachers teaches one class, so:
t = (sum of class sizes) / number of teachers
-/
noncomputable def t : ℝ := (class_sizes.map (λ x, x)).sum / (total_teachers : ℝ)

/-- 
Calculate s (average number of students per student):
s = Σ (class size[i] * probability of being in that class[i])
-/
noncomputable def s : ℝ := 
  class_sizes.map (λ x, (x : ℝ) * (x / total_students)).sum

theorem t_minus_s : t - s = -17.67 :=
by 
  -- Skipping the proof step
  sorry

end t_minus_s_l787_787529


namespace map_distance_to_real_distance_l787_787811

theorem map_distance_to_real_distance (d_map : ℝ) (scale : ℝ) (d_real : ℝ) 
    (h1 : d_map = 7.5) (h2 : scale = 8) : d_real = 60 :=
by
  sorry

end map_distance_to_real_distance_l787_787811


namespace quadruples_count_l787_787231

noncomputable def number_of_quadruples (p q r s : ℝ) : ℕ :=
  if (p^2 + q^2 + r^2 + s^2 = 9 ∧ (p + q + r + s) * (p^3 + q^3 + r^3 + s^3) = 81)
  then 1
  else 0

theorem quadruples_count :
  (∑ (x : ℕ) in (finset.map (function.embedding.prod_map (function.embedding.prod_map (function.embedding.prod_map finset.univ.embed finset.univ.embed) finset.univ.embed) finset.univ)).val.range, x) = 15 :=
sorry

end quadruples_count_l787_787231


namespace smallest_nat_k_l787_787145

theorem smallest_nat_k {x : Fin 100 → Fin 25 → ℝ}
  (h_row_sum : ∀ i : Fin 100, ∑ j, x i j ≤ 1)
  (h_rearranged : ∀ j : Fin 25, ∀ i₁ i₂ : Fin 100, i₁ ≤ i₂ → x' i₁ j ≥ x' i₂ j)
  (h_x'_def : ∀ j : Fin 25, ∀ i : Fin 100, ∃ p, x' i j = x p j)
  : ∃ k : ℕ, k = 97 ∧ ∀ i : Fin 4, i.val + 97 < 100 → ∑ j, x' ⟨i.val + 97, sorry⟩ j ≤ 1 :=
by
  sorry

end smallest_nat_k_l787_787145


namespace geometric_sequence_value_a3_l787_787253

-- Define the geometric sequence
def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

-- Conditions given in the problem
variable (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 2)
variable (h₂ : (geometric_sequence a₁ q 4) * (geometric_sequence a₁ q 6) = 4 * (geometric_sequence a₁ q 7) ^ 2)

-- The goal is to prove that a₃ = 1
theorem geometric_sequence_value_a3 : geometric_sequence a₁ q 3 = 1 :=
by
  sorry

end geometric_sequence_value_a3_l787_787253


namespace rectangles_form_regular_icosahedron_l787_787383

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem rectangles_form_regular_icosahedron (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : b > 0) :
  let ratio := b / a in
  ratio = golden_ratio →
  true := sorry

end rectangles_form_regular_icosahedron_l787_787383


namespace days_to_exceed_50000_l787_787023

noncomputable def A : ℕ → ℝ := sorry
def k : ℝ := sorry
def t := 10
def lg2 := 0.3
def lg (x : ℝ) : ℝ := sorry -- assuming the definition of logarithm base 10

axiom user_count : ∀ t, A t = 500 * Real.exp (k * t)
axiom condition_after_10_days : A 10 = 2000
axiom log_property : lg 2 = lg2

theorem days_to_exceed_50000 : ∃ t : ℕ, t ≥ 34 ∧ 500 * Real.exp (k * t) > 50000 := 
sorry

end days_to_exceed_50000_l787_787023


namespace transformed_data_mean_and_variance_l787_787646

variables {n : ℕ}
variables {x : ℕ → ℝ}

noncomputable def mean (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  (∑ i in finset.range n, x i) / n

noncomputable def variance (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  let m := mean x n in
  (∑ i in finset.range n, (x i - m) ^ 2) / n

theorem transformed_data_mean_and_variance (x : ℕ → ℝ) (n : ℕ) 
  (h_mean : mean x n = 6) (h_stddev : (variance x n).sqrt = 2) :
  mean (λ i, 2 * x i - 6) n = 6 ∧ variance (λ i, 2 * x i - 6) n = 16 := 
by
  sorry

end transformed_data_mean_and_variance_l787_787646


namespace average_cost_of_pencil_l787_787174

noncomputable def total_cost (price_pencils : ℕ) (shipping_cost : ℕ) : ℕ :=
  price_pencils + shipping_cost

noncomputable def average_cost_per_pencil (total_cost : ℕ) (number_of_pencils : ℕ) : ℕ :=
  (total_cost / number_of_pencils).to_nat

theorem average_cost_of_pencil :
  let price_pencils_dollars := 24.75
  let shipping_cost_dollars := 8.25
  let number_of_pencils := 150
  let price_pencils_cents := (price_pencils_dollars * 100).to_nat -- converting dollars to cents
  let shipping_cost_cents := (shipping_cost_dollars * 100).to_nat -- converting dollars to cents
  avg_cost == 22
  :=
begin
  let total_cost_cents := total_cost price_pencils_cents shipping_cost_cents,
  let avg_cost := average_cost_per_pencil total_cost_cents number_of_pencils,
  have : avg_cost = 22,
  sorry
end

end average_cost_of_pencil_l787_787174


namespace least_power_divisible_by_240_l787_787300

theorem least_power_divisible_by_240 (n : ℕ) (a : ℕ) (h_a : a = 60) (h : a^n % 240 = 0) : 
  n = 2 :=
by
  sorry

end least_power_divisible_by_240_l787_787300


namespace total_shaded_area_l787_787895

theorem total_shaded_area (S T : ℕ) 
  (h1 : 12 / S = 4)
  (h2 : S / T = 3) :
  (S * S) + 8 * (T * T) = 17 :=
by
  sorry

end total_shaded_area_l787_787895


namespace roots_of_polynomial_sum_of_reciprocals_l787_787008

-- We use a broader import to ensure access to Vieta's formulas and polynomial properties.
theorem roots_of_polynomial_sum_of_reciprocals :
  (∀ (p q r s : ℂ), polynomial.map coe (x^4 - 4*x^3 + 7*x^2 - 3*x + 2) = 0) →
  (∀ (p q r s : ℂ), ∃ (pq pr ps qr qs rs pqrs : ℂ),
  pq + pr + ps + qr + qs + rs = 7 ∧ pqrs = 2) →
  (∀ (p q r s : ℂ), (1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s) = 7 / 2)) :=
begin
  sorry
end

end roots_of_polynomial_sum_of_reciprocals_l787_787008


namespace product_of_two_numbers_l787_787845

theorem product_of_two_numbers (a b : ℝ)
  (h1 : a + b = 8 * (a - b))
  (h2 : a * b = 30 * (a - b)) :
  a * b = 400 / 7 :=
by
  sorry

end product_of_two_numbers_l787_787845


namespace root_expression_l787_787197

theorem root_expression {p q x1 x2 : ℝ}
  (h1 : x1^2 + p * x1 + q = 0)
  (h2 : x2^2 + p * x2 + q = 0) :
  (x1 / x2 + x2 / x1) = (p^2 - 2 * q) / q :=
by {
  sorry
}

end root_expression_l787_787197


namespace golf_tournament_percentage_increase_l787_787563

theorem golf_tournament_percentage_increase:
  let electricity_bill := 800
  let cell_phone_expenses := electricity_bill + 400
  let golf_tournament_cost := 1440
  (golf_tournament_cost - cell_phone_expenses) / cell_phone_expenses * 100 = 20 :=
by
  sorry

end golf_tournament_percentage_increase_l787_787563


namespace convert_sqrt2_exp_to_rectangular_l787_787209

noncomputable def convert_to_rectangular_form (z : ℂ) : ℂ :=
  z

def theta : ℝ := 13 * Real.pi / 4
def r : ℝ := Real.sqrt 2

theorem convert_sqrt2_exp_to_rectangular :
  convert_to_rectangular_form (r * Complex.exp (Complex.I * theta)) = (1 : ℂ) + (Complex.I) :=
by
  sorry

end convert_sqrt2_exp_to_rectangular_l787_787209


namespace at_least_four_solve_l787_787188

noncomputable def arthur_prob : ℚ := 1/4
noncomputable def bella_prob : ℚ := 3/10
noncomputable def xavier_prob : ℚ := 1/6
noncomputable def yvonne_prob : ℚ := 1/2
noncomputable def zelda_prob : ℚ := 5/8
noncomputable def not_zelda_prob : ℚ := 1 - zelda_prob

noncomputable def required_prob : ℚ := arthur_prob * yvonne_prob * bella_prob * xavier_prob * not_zelda_prob

theorem at_least_four_solve :
  required_prob = 1 / 426.666... := 
sorry

end at_least_four_solve_l787_787188


namespace even_function_values_l787_787496

theorem even_function_values (a : ℝ) 
  (h : ∀ x : ℝ, (λ x, (x + a) * 3^(x - 2 + a^2) - (x - a) * 3^(8 - x - 3 * a)) (-x) = 
                 (λ x, (x + a) * 3^(x - 2 + a^2) - (x - a) * 3^(8 - x - 3 * a)) (x))
  : a = 2 ∨ a = -5 := 
sorry

end even_function_values_l787_787496


namespace tangent_circles_ratios_l787_787118

theorem tangent_circles_ratios
  (C D : set Point)
  (P A E F M N : Point)
  (tangent_C_D : tangent C D P) 
  (A_on_C : A ∈ C)
  (AM_tangent_D : tangent AM D A M)
  (AN_tangent_D : tangent AN D A N)
  (AM_intersects_C_at_E : ∃ E ∈ C, A = E ∨ M = E)
  (AN_intersects_C_at_F : ∃ F ∈ C, A = F ∨ N = F) :
  (segment_ratio P E) / (segment_ratio P F) = (segment_ratio M E) / (segment_ratio N F) :=
by
  sorry

end tangent_circles_ratios_l787_787118


namespace odd_number_mod_pow_l787_787014

theorem odd_number_mod_pow {a n : ℤ} (h_a : Odd a) (h_n : 0 < n) : 
  a ^ (2^n) ≡ 1 [MOD 2^(n+2)] := 
by 
  sorry

end odd_number_mod_pow_l787_787014


namespace negation_proposition_p_l787_787045

theorem negation_proposition_p (x y : ℝ) : (¬ ((x - 1) ^ 2 + (y - 2) ^ 2 = 0) → (x ≠ 1 ∨ y ≠ 2)) :=
by
  sorry

end negation_proposition_p_l787_787045


namespace sum_of_valid_x_119_l787_787116

theorem sum_of_valid_x_119 :
  (∑ x in (Finset.filter (λ (x : ℕ), 10 ≤ x ∧ (360 / x) ≥ 12 ∧ 360 % x = 0) (finset.range 361)), x) = 119 :=
by
  sorry

end sum_of_valid_x_119_l787_787116


namespace original_number_is_842_l787_787146

theorem original_number_is_842 (x y z : ℕ) (h1 : x * z = y^2)
  (h2 : 100 * z + x = 100 * x + z - 594)
  (h3 : 10 * z + y = 10 * y + z - 18)
  (hx : x = 8) (hy : y = 4) (hz : z = 2) :
  100 * x + 10 * y + z = 842 :=
by
  sorry

end original_number_is_842_l787_787146


namespace salesman_profit_l787_787903

theorem salesman_profit :
  let total_backpacks := 48
  let total_cost := 576
  let swap_meet_sold := 17
  let swap_meet_price := 18
  let department_store_sold := 10
  let department_store_price := 25
  let remaining_price := 22
  let remaining_sold := total_backpacks - swap_meet_sold - department_store_sold
  let revenue_swap_meet := swap_meet_sold * swap_meet_price
  let revenue_department_store := department_store_sold * department_store_price
  let revenue_remaining := remaining_sold * remaining_price
  let total_revenue := revenue_swap_meet + revenue_department_store + revenue_remaining
  let profit := total_revenue - total_cost
  in profit = 442 := by
  sorry

end salesman_profit_l787_787903


namespace max_stickers_for_one_player_l787_787511

-- Define the constants and conditions
constant num_players : ℕ := 25
constant avg_stickers : ℕ := 4
constant min_stickers_per_player : ℕ := 1
def total_stickers : ℕ := num_players * avg_stickers
def min_stickers_for_24_players : ℕ := 24 * min_stickers_per_player

-- The theorem statement
theorem max_stickers_for_one_player : total_stickers - min_stickers_for_24_players = 76 := 
by 
  sorry

end max_stickers_for_one_player_l787_787511


namespace oranges_sold_l787_787441

def bags : ℕ := 10
def oranges_per_bag : ℕ := 30
def rotten_oranges : ℕ := 50
def oranges_for_juice : ℕ := 30

theorem oranges_sold : (bags * oranges_per_bag) - rotten_oranges - oranges_for_juice = 220 := by
  sorry

end oranges_sold_l787_787441


namespace probability_diff_geq_3_l787_787463

open Finset
open BigOperators

-- Define the set of numbers
def number_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the condition for pairs with positive difference ≥ 3
def valid_pair (a b : ℕ) : Prop := a ≠ b ∧ abs (a - b) ≥ 3

-- Calculate the probability as a fraction
theorem probability_diff_geq_3 : 
  let total_pairs := (number_set.product number_set).filter (λ ab, ab.1 < ab.2) in
  let valid_pairs := total_pairs.filter (λ ab, valid_pair ab.1 ab.2) in
  valid_pairs.card / total_pairs.card = 7 / 12 :=
by
  sorry

end probability_diff_geq_3_l787_787463


namespace exists_strategy_for_puzzle_l787_787119

-- Define the problem context
def exists_strategy_to_deduce_numbers (S : Set ℕ) (sum_to : ℕ) : Prop :=
  ∀ (A B : ℕ), A ∈ S → B ∈ S →
    ∃ (strategy : ℕ → ℕ → List ℕ), 
      (∀ (n ∈ strategy A B), n > 0) ∧
      (List.sum (strategy A B) = sum_to) ∧
      (∀ (A' B' : ℕ), A' ∈ S → B' ∈ S → A' ≠ A → B' ≠ B → strategy A B ≠ strategy A' B')

theorem exists_strategy_for_puzzle :
  exists_strategy_to_deduce_numbers (Set.range (λ n, n + 1) ∩ Set.Ico 1 251) 20 :=
sorry

end exists_strategy_for_puzzle_l787_787119


namespace complement_union_is_correct_l787_787660

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_is_correct : (U \ A) ∪ B = {0, 2, 4} := by
  sorry

end complement_union_is_correct_l787_787660


namespace f_f_neg3_eq_neg1_l787_787275

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (x + 2) else Real.log x

theorem f_f_neg3_eq_neg1 : f (f (-3)) = -1 :=
by
  sorry

end f_f_neg3_eq_neg1_l787_787275


namespace sales_function_profit_8250_max_profit_l787_787454

-- Conditions
def cost_price := 80
def min_price := 90
def max_price := 110

def initial_sales := 600
def decrease_per_yuan := 10

-- Problem 1
theorem sales_function (x : ℕ) : 
  x >= min_price -> 
  y = 600 - 10 * (x - 90) -> 
  y = -10 * x + 1500 := sorry

-- Problem 2
theorem profit_8250 (x : ℕ) :
  x >= min_price -> 
  (x - 80) * (600 - 10 * (x - 90)) = 8250 -> 
  x = 95 := sorry

-- Problem 3
theorem max_profit (x : ℕ) :
  x >= min_price ->
  x <= max_price ->
  (x - 80) * (600 - 10 * (x - 90)) = -10*(110 - 80)^2 + 12250 -> 
  ( (x ≤ 110) 
    -> ∃ p : ℕ, p = -10*(110 - x)^2 + 12250
    -> p ≤ 12000 ) := sorry

end sales_function_profit_8250_max_profit_l787_787454


namespace total_albums_l787_787031

-- Definitions based on given conditions
def adele_albums : ℕ := 30
def bridget_albums : ℕ := adele_albums - 15
def katrina_albums : ℕ := 6 * bridget_albums
def miriam_albums : ℕ := 5 * katrina_albums

-- The final statement to be proved
theorem total_albums : adele_albums + bridget_albums + katrina_albums + miriam_albums = 585 :=
by
  sorry

end total_albums_l787_787031


namespace probability_of_diff_ge_3_l787_787461

noncomputable def probability_diff_ge_3 : ℚ :=
  let elements := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let total_pairs := (elements.to_finset.card.choose 2 : ℕ)
  let valid_pairs := (total_pairs - 15 : ℕ) -- 15 pairs with a difference of less than 3
  valid_pairs / total_pairs

theorem probability_of_diff_ge_3 :
  probability_diff_ge_3 = 7 / 12 := by
  sorry

end probability_of_diff_ge_3_l787_787461


namespace sum_product_le_four_l787_787005

theorem sum_product_le_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 4) : 
  a * b + b * c + c * d + d * a ≤ 4 := 
sorry

end sum_product_le_four_l787_787005


namespace max_cos_sum_l787_787547

theorem max_cos_sum (A B C : ℝ) (h_sum : A + B + C = π) (h_nonneg : 0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C)
  (h_lt_pi : A < π ∧ B < π ∧ C < π) :
  ∃ (θ : ℝ), cos θ = 1 / real.sqrt 2 ∧ cos A + cos B * cos C = 1 / real.sqrt 2 :=
begin
  sorry
end

end max_cos_sum_l787_787547


namespace alpha_beta_square_eq_eight_l787_787680

open Real

theorem alpha_beta_square_eq_eight :
  ∃ α β : ℝ, 
  (∀ x : ℝ, x^2 - 2 * x - 1 = 0 ↔ x = α ∨ x = β) → 
  (α ≠ β) → 
  (α - β)^2 = 8 :=
sorry

end alpha_beta_square_eq_eight_l787_787680


namespace train_speed_proof_l787_787178

noncomputable def train_speed_kmph 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_seconds : ℕ) : ℝ :=
(train_length + platform_length) / time_seconds * 3.6

theorem train_speed_proof : 
  train_speed_kmph 120 380.04 25 ≈ 72.01 :=
by 
  sorry


end train_speed_proof_l787_787178


namespace problem1_problem2_l787_787497

-- Definition for the first mathematical expression
def expr1 : ℝ :=
  (9 / 4)^(1 / 2) - (-2009)^0 - (8 / 27)^(2 / 3) + (3 / 2)^(-2)

-- Definition for the second mathematical expression
def expr2 : ℝ :=
  log 625 / log 25 + log 0.001 / log 10 + log (sqrt (exp 1)) + 2^(-1 + log 3 / log 2)

-- Theorem stating the equivalence of expr1 and 1/2
theorem problem1 : expr1 = 1 / 2 := by
  sorry

-- Theorem stating the equivalence of expr2 and 1
theorem problem2 : expr2 = 1 := by
  sorry

end problem1_problem2_l787_787497


namespace collinear_points_k_value_l787_787829

theorem collinear_points_k_value : 
  (∀ k : ℝ, ∃ (a : ℝ) (b : ℝ), ∀ (x : ℝ) (y : ℝ),
    ((x, y) = (1, -2) ∨ (x, y) = (3, 2) ∨ (x, y) = (6, k / 3)) → y = a * x + b) → k = 24 :=
by
sorry

end collinear_points_k_value_l787_787829


namespace find_false_proposition_l787_787921

-- Definitions as per the conditions given
def sum_of_exterior_angles (n : ℕ) : ℕ := 360
def triangle_side_inequality (a b c : ℝ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)
def exists_unique_parallel (p : ℝ × ℝ) (L : set (ℝ × ℝ)) : Prop := 
  (∃! l : set (ℝ × ℝ), is_line l ∧ (∀ q ∈ l, q ≠ p → q ∈ L)) 
def exists_unique_perpendicular (p : ℝ × ℝ) (L : set (ℝ × ℝ)) : Prop := 
  (∃! l : set (ℝ × ℝ), is_line l ∧ (∃ q ∈ l, (q ≠ p → ⊥)))

-- Proposition C statement
def proposition_C_false (p : ℝ × ℝ) (L : set (ℝ × ℝ)) : Prop :=
  ¬ exists_unique_parallel p L

-- Problem statement
theorem find_false_proposition (p : ℝ × ℝ) (L : set (ℝ × ℝ)) 
  (h1 : sum_of_exterior_angles 5 = 360)
  (h2 : ∀ a b c : ℝ, triangle_side_inequality a b c)
  (h3 : ¬ exists_unique_parallel p L)
  (h4 : exists_unique_perpendicular p L) :
  proposition_C_false p L :=
by sorry

end find_false_proposition_l787_787921


namespace cube_vertex_adjacency_l787_787492

noncomputable def beautiful_face (a b c d : ℕ) : Prop :=
  a = b + c + d ∨ b = a + c + d ∨ c = a + b + d ∨ d = a + b + c

theorem cube_vertex_adjacency :
  ∀ (v1 v2 v3 v4 v5 v6 v7 v8 : ℕ), 
  v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ v1 ≠ v6 ∧ v1 ≠ v7 ∧ v1 ≠ v8 ∧
  v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ v2 ≠ v6 ∧ v2 ≠ v7 ∧ v2 ≠ v8 ∧
  v3 ≠ v4 ∧ v3 ≠ v5 ∧ v3 ≠ v6 ∧ v3 ≠ v7 ∧ v3 ≠ v8 ∧
  v4 ≠ v5 ∧ v4 ≠ v6 ∧ v4 ≠ v7 ∧ v4 ≠ v8 ∧
  v5 ≠ v6 ∧ v5 ≠ v7 ∧ v5 ≠ v8 ∧
  v6 ≠ v7 ∧ v6 ≠ v8 ∧
  v7 ≠ v8 ∧
  beautiful_face v1 v2 v3 v4 ∧ beautiful_face v5 v6 v7 v8 ∧
  beautiful_face v1 v3 v5 v7 ∧ beautiful_face v2 v4 v6 v8 ∧
  beautiful_face v1 v2 v5 v6 ∧ beautiful_face v3 v4 v7 v8 →
  (v6 = 6 → (v1 = 2 ∧ v2 = 3 ∧ v3 = 5) ∨ 
   (v1 = 3 ∧ v2 = 5 ∧ v3 = 7) ∨ 
   (v1 = 2 ∧ v2 = 3 ∧ v3 = 7)) :=
sorry

end cube_vertex_adjacency_l787_787492


namespace perfect_square_closest_to_950_l787_787128

theorem perfect_square_closest_to_950 : 
  ∃ n : ℕ, n * n = 961 ∧ 
            (∀ m : ℕ, (m * m ≠ 961) → 
              abs (950 - 961) < abs (950 - m * m)) :=
by
  sorry

end perfect_square_closest_to_950_l787_787128


namespace smallest_N_exists_l787_787320

theorem smallest_N_exists :
  ∃ (N : ℕ), (N ≥ 730) ∧
    (∀ (students : list (ℝ × ℝ × ℝ × ℝ)),
      students.length = N →
      (∃ (subset : list (ℝ × ℝ × ℝ × ℝ)),
        subset ⊆ students ∧ subset.length = 10 ∧
        (∃ (i j : Fin 4), i ≠ j ∧
          ∀ (seq : Fin 10 → ℕ),
            (∀ k : Fin 10, seq k < subset.length) →
            let subjects := λ (idx : Fin 10) => list.indexN idx {a,b,c,d} in
            (is_sorted (λ k, subjects (seq k))) ))) :=
sorry

end smallest_N_exists_l787_787320


namespace existence_of_special_set_l787_787256

-- Define the required conditions and statement
theorem existence_of_special_set (n : ℕ) (h : n ≥ 3) :
  ∃ S : Finset ℕ, S.card = n ∧
    ∀ (A B : Finset ℕ), A ≠ B ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ A ⊆ S ∧ B ⊆ S → 
    Nat.coprime 
      ((A.sum id) / B.card) ((A.sum id) - (A.card * (B.sum id) / B.card)) ∧ 
    ¬Nat.prime ((A.sum id) / B.card)
  :=
  sorry

end existence_of_special_set_l787_787256


namespace february_1_is_tuesday_l787_787687

def day_of_week := ℕ -- We will represent days of the week as natural numbers from 0 to 6 (starting from 0: Monday, and so forth).

-- Function to compute the day of the week of a given day in February, given the day of the week for February 23
def find_day_of_week (feb_23_wday : day_of_week) (d : ℕ) : day_of_week :=
  (feb_23_wday + (d - 23)) % 7

theorem february_1_is_tuesday (feb_23_wday : day_of_week) (H : feb_23_wday = 2) : find_day_of_week feb_23_wday 1 = 1 :=
by
  sorry

end february_1_is_tuesday_l787_787687


namespace keith_gave_away_p_l787_787728

theorem keith_gave_away_p (k_init : Nat) (m_init : Nat) (final_pears : Nat) (k_gave_away : Nat) (total_init: Nat := k_init + m_init) :
  k_init = 47 →
  m_init = 12 →
  final_pears = 13 →
  k_gave_away = total_init - final_pears →
  k_gave_away = 46 :=
by
  -- Insert proof here (skip using sorry)
  sorry

end keith_gave_away_p_l787_787728


namespace complex_exponential_sum_l787_787298

theorem complex_exponential_sum (α β : ℝ) (h : complex.exp (complex.I * α) + complex.exp (complex.I * β) = (1/3 : ℂ) + (2/5 : ℂ) * complex.I) :
  complex.exp (- complex.I * α) + complex.exp (- complex.I * β) = (1/3 : ℂ) - (2/5 : ℂ) * complex.I :=
by sorry

end complex_exponential_sum_l787_787298


namespace total_meters_built_l787_787528

/-- Define the length of the road -/
def road_length (L : ℕ) := L = 1000

/-- Define the average meters built per day -/
def average_meters_per_day (A : ℕ) := A = 120

/-- Define the number of days worked from July 29 to August 2 -/
def number_of_days_worked (D : ℕ) := D = 5

/-- The total meters built by the time they finished -/
theorem total_meters_built
  (L A D : ℕ)
  (h1 : road_length L)
  (h2 : average_meters_per_day A)
  (h3 : number_of_days_worked D)
  : L / D * A = 600 := by
  sorry

end total_meters_built_l787_787528


namespace vector_solution_l787_787648

variables (a x : Type) [AddGroup x] [Module ℝ x]

theorem vector_solution (a x : x) (h : 3 • (a + x) = x) : x = - 3 / 2 • a :=
by
  sorry

end vector_solution_l787_787648


namespace total_marbles_is_260_l787_787214

-- Define the number of marbles in each jar based on the conditions.
def first_jar : Nat := 80
def second_jar : Nat := 2 * first_jar
def third_jar : Nat := (1 / 4 : ℚ) * first_jar

-- Prove that the total number of marbles is 260.
theorem total_marbles_is_260 : first_jar + second_jar + third_jar = 260 := by
  sorry

end total_marbles_is_260_l787_787214


namespace derangements_count_four_l787_787611

-- Define what it means to be a derangement
def is_derangement (σ : perm (fin 4)) : Prop :=
  ∀ i, σ i ≠ i

-- Define the set of derangements of 4 elements
def derangements (n : ℕ) := {σ : perm (fin n) // is_derangement σ}

-- Declare the main problem statement
theorem derangements_count_four :
  (derangements 4).card = 9 := sorry

end derangements_count_four_l787_787611


namespace sum_of_two_dice_is_9_probability_l787_787465

theorem sum_of_two_dice_is_9_probability : 
  let outcomes := [(1,8), (2,7), (3,6), (4,5), (5,4), (6,3), (7,2), (8,1)];
      total_outcomes := 8 * 8;
      favorable_outcomes := 2 * outcomes.length
  in (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 4 := by
  sorry

end sum_of_two_dice_is_9_probability_l787_787465


namespace meet_conditions_l787_787982

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + |x| + 1

theorem meet_conditions :
  (∀ (x1 x2 : ℝ), x1 ∈ set.Ioi 0 → x2 ∈ set.Ioi 0 → x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0)
  ∧
  (∀ (x : ℝ), f x = f (-x)) :=
by
  sorry

end meet_conditions_l787_787982


namespace difference_in_profit_shares_l787_787541

variables (capital_A capital_B capital_C profit_B : ℕ)
variables (profit_A profit_C : ℕ)

-- Defining the capital investments
def A_investment : ℕ := capital_A
def B_investment : ℕ := capital_B
def C_investment : ℕ := capital_C

-- Given condition for investments
axiom A_investment_condition : A_investment = 8000
axiom B_investment_condition : B_investment = 10000
axiom C_investment_condition : C_investment = 12000

-- Given condition for B's profit share
axiom B_profit_condition : profit_B = 2500

-- Calculating the ratio and simplifying it as per given conditions/proportion
def investment_ratio_A : ℕ := 4
def investment_ratio_B : ℕ := 5
def investment_ratio_C : ℕ := 6

-- Defining the individual profit shares based on the ratio
def one_part_value : ℕ := profit_B / investment_ratio_B

def A_profit_share : ℕ := investment_ratio_A * one_part_value
def C_profit_share : ℕ := investment_ratio_C * one_part_value

-- Given condition for A and C's profit shares
axiom A_profit_condition : profit_A = A_profit_share
axiom C_profit_condition : profit_C = C_profit_share

-- The theorem to prove the difference
theorem difference_in_profit_shares : profit_C - profit_A = 1000 :=
by {
  -- Insert proof steps
  sorry
}

end difference_in_profit_shares_l787_787541


namespace quadratic_roots_identity_l787_787262

theorem quadratic_roots_identity (α β : ℝ) (hαβ : α^2 - 3*α - 4 = 0 ∧ β^2 - 3*β - 4 = 0) : 
  α^2 + α*β - 3*α = 0 := 
by 
  sorry

end quadratic_roots_identity_l787_787262


namespace limit_problem_l787_787950

theorem limit_problem (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (∃ L : ℝ, L = (if a > 1 then a else 1)) ∧ 
  (∀ ε > 0, ∃ N : ℝ, ∀ x > N, abs ((\frac{a^x - 1}{x * (a - 1)})^(1/x) - L) < ε) :=
by 
  sorry

end limit_problem_l787_787950


namespace total_emeralds_l787_787516

theorem total_emeralds {R D E : ℕ} (h1 : R = D + 15) (boxes : Finset ℕ)
  (h2 : boxes = {2, 4, 5, 7, 8, 13})
  (h3 : ∃ diamonds rubies emeralds : Finset ℕ,
    diamonds.card = 2 ∧
    rubies.card = 2 ∧
    emeralds.card = 2 ∧
    diamonds.sum = D ∧
    rubies.sum = R ∧
    emeralds.sum = E ∧
    diamonds ∪ rubies ∪ emeralds = boxes) :
  E = 12 :=
sorry

end total_emeralds_l787_787516


namespace stratified_sampling_2nd_year_students_l787_787156

theorem stratified_sampling_2nd_year_students
  (students_1st_year : ℕ) (students_2nd_year : ℕ) (students_3rd_year : ℕ) (total_sample_size : ℕ) :
  students_1st_year = 1000 ∧ students_2nd_year = 800 ∧ students_3rd_year = 700 ∧ total_sample_size = 100 →
  (students_2nd_year * total_sample_size / (students_1st_year + students_2nd_year + students_3rd_year) = 32) :=
by
  intro h
  sorry

end stratified_sampling_2nd_year_students_l787_787156


namespace misread_signs_in_front_of_6_terms_l787_787858

/-- Define the polynomial function --/
def poly (x : ℝ) : ℝ :=
  10 * x ^ 9 + 9 * x ^ 8 + 8 * x ^ 7 + 7 * x ^ 6 + 6 * x ^ 5 + 5 * x ^ 4 + 4 * x ^ 3 + 3 * x ^ 2 + 2 * x + 1

/-- Xiao Ming's mistaken result --/
def mistaken_result : ℝ := 7

/-- Correct value of the expression at x = -1 --/
def correct_value : ℝ := poly (-1)

/-- The difference due to misreading signs --/
def difference : ℝ := mistaken_result - correct_value

/-- Prove that Xiao Ming misread the signs in front of 6 terms --/
theorem misread_signs_in_front_of_6_terms :
  difference / 2 = 6 :=
by
  simp [difference, correct_value, poly]
  -- the proof steps would go here
  sorry

#eval poly (-1)  -- to validate the correct value
#eval mistaken_result - poly (-1)  -- to validate the difference

end misread_signs_in_front_of_6_terms_l787_787858


namespace trigonometric_identity_l787_787495

theorem trigonometric_identity (α : ℝ) :
  4.62 * (cos (2 * α))^4 - 6 * (cos (2 * α))^2 * (sin (2 * α))^2 + (sin (2 * α))^4 = cos (8 * α) :=
sorry

end trigonometric_identity_l787_787495


namespace number_multiplied_by_approx_l787_787504

variable (X : ℝ)

theorem number_multiplied_by_approx (h : (0.0048 * X) / (0.05 * 0.1 * 0.004) = 840) : X = 3.5 :=
by
  sorry

end number_multiplied_by_approx_l787_787504


namespace find_k_l787_787865

theorem find_k (k : ℕ) : (∃ n : ℕ, 2^k + 8*k + 5 = n^2) ↔ k = 2 := by
  sorry

end find_k_l787_787865


namespace calculate_shaded_area_l787_787909

-- Define the vertices of the square and triangle
def vertex_A := (0 : ℝ, 0 : ℝ)
def vertex_B := (0 : ℝ, 15 : ℝ)
def vertex_C := (15 : ℝ, 15 : ℝ)
def vertex_D := (15 : ℝ, 0 : ℝ)
def vertex_E := (30 : ℝ, 0 : ℝ)
def vertex_F := (15 : ℝ, 15 : ℝ)

-- Functions to find the intersection of BE and DF
def line_eq_B to E (x : ℝ) := -1/2 * x + 15
def line_eq_D to F (x : ℝ) := -1.5 * x + 37.5

def intersection_G : ℝ × ℝ := (15, 7.5)

-- Base and height for the necessary triangles
def base_DG := 15 : ℝ
def height_GE := 7.5 : ℝ
def area_DGE := 1/2 * base_DG * height_GE

def base_DE := 30 : ℝ
def height_BE := 15 : ℝ
def area_DBE := 1/2 * base_DE * height_BE

noncomputable def area_shaded_region : ℝ := area_DBE - area_DGE

theorem calculate_shaded_area : area_shaded_region = 168.75 :=
sorry

end calculate_shaded_area_l787_787909


namespace total_match_sequences_l787_787407

theorem total_match_sequences :
  let teams := 7
  let total_players := 14 in 
    (Nat.choose total_players teams) = 3432 := by
  sorry

end total_match_sequences_l787_787407


namespace min_value_MA_dot_MB_l787_787661

open real

theorem min_value_MA_dot_MB (OP A B : ℝ × ℝ) (hOP : OP = (2, 1)) (hOA : A = (1, 7)) (hOB : B = (5, 1)) :
  ∃ M : ℝ × ℝ, (∃ λ : ℝ, M = (2 * λ, λ)) ∧ (let MA := (1 - 2 * M.1, 7 - M.2),
                                            MB := (5 - 2 * M.1, 1 - M.2) in
                                          ∀λ, MA.1 * MB.1 + MA.2 * MB.2 ≥ -8) :=
by
  use (2 * 2, 2)
  use 2
  sorry

end min_value_MA_dot_MB_l787_787661


namespace trig_identity_proof_l787_787232

theorem trig_identity_proof : 
  (\sin (20 * real.pi / 180) * \cos (10 * real.pi / 180) + \cos (160 * real.pi / 180) * \cos (110 * real.pi / 180)) / 
  (\sin (24 * real.pi / 180) * \cos (6 * real.pi / 180) + \cos (156 * real.pi / 180) * \cos (96 * real.pi / 180)) =
  (1 - \sin (40 * real.pi / 180)) / (1 - \sin (48 * real.pi / 180)) :=
by
  sorry

end trig_identity_proof_l787_787232


namespace find_b_l787_787828

theorem find_b (b c x1 x2 : ℝ)
  (h_parabola_intersects_x_axis : (x1 ≠ x2) ∧ x1 * x2 = c ∧ x1 + x2 = -b ∧ x2 - x1 = 1)
  (h_parabola_intersects_y_axis : c ≠ 0)
  (h_length_ab : x2 - x1 = 1)
  (h_area_abc : (1 / 2) * (x2 - x1) * |c| = 1)
  : b = -3 :=
sorry

end find_b_l787_787828


namespace problem_I_problem_II_problem_III_l787_787980
 
noncomputable def f (t x : ℝ) : ℝ := x^3 - (3 * (t + 1) / 2) * x^2 + 3 * t * x + 1 

theorem problem_I (t : ℝ) (h : t > 0) :  (∀ x ∈ (Set.Icc 0 2), f t x = f t x → ∀ y, f' t y ≠ 0) → t = 1 
  sorry

theorem problem_II (t x₀ : ℝ) (h₀ : t > 0) (h₁ : x₀ ∈ Set.Ioc 0 2) : 
  (∃ x₀ ∈ (Set.Ioc 0 2), ∀ y ∈ (0..2), (f t x₀ = f t y) → (∀ y z, (y ∈ Set.Icc 0 2) = (z ∈ Set.Icc 0 2)) * f t x₀ = f t 2) → 
  (t ∈ Set.Icc (5/3) 2) 
  sorry

theorem problem_III (t x : ℝ) (h₀ : t > 0) : 
  (∀ x₀ ∈ Set.Ici 0, f t x₀ - x₀ * exp x₀ + 1 ≤ 1) → 
  t ∈ Set.Ioi 0 ∩ Set.Iic (1/3) 
  sorry

end problem_I_problem_II_problem_III_l787_787980


namespace frogs_needing_new_pond_l787_787886

theorem frogs_needing_new_pond : 
    (initial_frogs tadpoles_survival_fraction pond_capacity : ℕ) 
    (h1 : initial_frogs = 5) 
    (h2 : tadpoles_survival_fraction = 2 / 3) 
    (h3 : pond_capacity = 8) 
    : (initial_frogs + tadpoles_survival_fraction * 3 * initial_frogs - pond_capacity = 7) :=
by
    sorry

end frogs_needing_new_pond_l787_787886


namespace largest_prime_factor_of_95_and_133_is_19_l787_787482

theorem largest_prime_factor_of_95_and_133_is_19 :
  let largest_prime_55 := 11,
      largest_prime_63 := 7,
      largest_prime_95 := 19,
      largest_prime_133 := 19,
      largest_prime_143 := 13 in
    largest_prime_95 = 19 ∧ largest_prime_95 > largest_prime_55 ∧
    largest_prime_95 > largest_prime_63 ∧ largest_prime_95 > largest_prime_143 ∧
    largest_prime_133 = 19 ∧ largest_prime_133 > largest_prime_55 ∧
    largest_prime_133 > largest_prime_63 ∧ largest_prime_133 > largest_prime_143 :=
by
  sorry

end largest_prime_factor_of_95_and_133_is_19_l787_787482


namespace max_edges_cut_l787_787571

def volleyball_net := (10, 20)

def total_points := 431

def total_edges := 1230

def min_edges_connected (V : ℕ) := V - 1

theorem max_edges_cut :
    let V := total_points in
    let E := total_edges in
    V = 431 →
    E = 1230 →
    E - min_edges_connected V = 800 :=
begin
    intros V E hV hE,
    rw [hV, hE],
    sorry
end

end max_edges_cut_l787_787571


namespace track_length_l787_787115

theorem track_length (x : ℝ) (tom_dist1 jerry_dist1 : ℝ) (tom_dist2 jerry_dist2 : ℝ) (deg_gap : ℝ) :
  deg_gap = 120 ∧ 
  tom_dist1 = 120 ∧ 
  (tom_dist1 + jerry_dist1 = x * deg_gap / 360) ∧ 
  (jerry_dist1 + jerry_dist2 = x * deg_gap / 360 + 180) →
  x = 630 :=
by
  sorry

end track_length_l787_787115


namespace angle_D_in_quadrilateral_l787_787322

theorem angle_D_in_quadrilateral:
  ∀ (A B C D : ℝ), A = 70 ∧ B = 60 ∧ C = 40 ∧ (A + B + C + D = 360) → D = 190 :=
by
  intro A B C D
  intro h
  cases h with hA h_rest
  cases h_rest with hB h_rest2
  cases h_rest2 with hC h_sum
  rw [hA, hB, hC] at h_sum
  sorry

end angle_D_in_quadrilateral_l787_787322


namespace production_growth_l787_787095

variable (a : ℝ)  -- current production
variable (g : ℝ := 1.05)  -- annual growth rate (1 + 5%)
variable (x : ℕ)  -- number of years

-- This is the proposition that we want to prove.
theorem production_growth (hx : x > 0) : true :=
  ∃ y : ℝ, y = a * g ^ x

end production_growth_l787_787095


namespace power_neg_two_power_zero_l787_787559

theorem power_neg_two : 3^(-2) = 1 / 9 := 
by sorry

theorem power_zero :
  (Real.pi - 2)^0 = 1 := 
by sorry

end power_neg_two_power_zero_l787_787559


namespace find_a_max_value_l787_787643

noncomputable def f (x a : ℝ) : ℝ := x^3 - a * x^2 + 2

theorem find_a (a : ℝ) : (∃ c : ℝ, (c = 2 ∧ ∀ x : ℝ, f c a = 0)) → a = 3 :=
by 
  intro h
  cases h with c hc
  cases hc with hc1 hc2
  sorry

theorem max_value (a : ℝ) (hx : a = 3) : 
  ∃ M : ℝ, (∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f x a ≤ M) ∧ M = 2 :=
by 
  sorry

end find_a_max_value_l787_787643


namespace find_lambda_l787_787870

noncomputable
def triangleCircumcenter (A B C P : Point): Prop :=
  ∥P - A∥ = ∥P - B∥ ∧ ∥P - B∥ = ∥P - C∥

noncomputable 
def dot_product (v w : vector ℝ 2) : ℝ :=
  v.1 * w.1 + v.2 * w.2

variables {A B C P: Point} {PA PB PC : vector ℝ 2} {R: ℝ} {λ: ℝ}

-- Given conditions
axiom PA_eq : PA = P - A
axiom PB_eq : PB = P - B
axiom PC_eq : PC = P - C
axiom circumcenter : triangleCircumcenter A B C P
axiom angle_C : angle A P B = 120

-- The main statement to be proven
theorem find_lambda (h1 : PA + PB + λ•PC = 0) (h2 : ∥PA∥ = R) (h3 : ∥PB∥ = R) (h4 : ∥PC∥ = R) : λ = -1 := 
begin
  sorry
end

end find_lambda_l787_787870


namespace rectangle_diagonals_eq_sum_l787_787353

theorem rectangle_diagonals_eq_sum (ABCD : Rectangle) (P : Point) (hP : P ∈ side AB) :
  let PS := perpendicular_to_diagonal P BD,
      PR := perpendicular_to_diagonal P AC
  in PR + PS = 2 * BD :=
by
  sorry

end rectangle_diagonals_eq_sum_l787_787353


namespace sum_M_inter_N_eq_l787_787004

-- Let a be a natural number
variable (a : ℕ)

-- Define M as the set of all integers x that satisfy |x - a| < a + 1 / 2
def M : Set ℤ := {x | |x - a| < a + 1 / 2}

-- Define N as the set of all integers x that satisfy |x| < 2a
def N : Set ℤ := {x | |x| < 2 * a}

-- Prove that the sum of all integers in M ∩ N is a(2a - 1)
theorem sum_M_inter_N_eq : 
  ( ∑ x in (M a) ∩ (N a), x ) = a * (2 * a - 1) := by
  sorry

end sum_M_inter_N_eq_l787_787004


namespace trigonometric_identity_simplification_l787_787060

-- Define x, y, z as real numbers
variables (x y z : ℝ)

-- Statement of the theorem to be proved
theorem trigonometric_identity_simplification :
  sin (x + y) * cos z + cos (x + y) * sin z = sin (x + y + z) :=
sorry

end trigonometric_identity_simplification_l787_787060


namespace area_ratio_PQR_PQO_l787_787722

variables {P Q R A B O : Type}
variables [Triangle P Q R] (PA : Median P Q R A) (QB : AngleBisector Q B)
variables (h1 : PA ∩ QB = O) (h2 : 3 * PQ = 5 * QR)

theorem area_ratio_PQR_PQO : 
  (area (triangle P Q R)) / (area (triangle P Q O)) = 2.6 :=
sorry

end area_ratio_PQR_PQO_l787_787722


namespace albums_total_l787_787034

noncomputable def miriam_albums (katrina_albums : ℕ) := 5 * katrina_albums
noncomputable def katrina_albums (bridget_albums : ℕ) := 6 * bridget_albums
noncomputable def bridget_albums (adele_albums : ℕ) := adele_albums - 15
noncomputable def total_albums (adele_albums : ℕ) (bridget_albums : ℕ) (katrina_albums : ℕ) (miriam_albums : ℕ) := 
  adele_albums + bridget_albums + katrina_albums + miriam_albums

theorem albums_total (adele_has_30 : adele_albums = 30) : 
  total_albums 30 (bridget_albums 30) (katrina_albums (bridget_albums 30)) (miriam_albums (katrina_albums (bridget_albums 30))) = 585 := 
by 
  -- In Lean, replace the term 'sorry' below with the necessary steps to finish the proof
  sorry

end albums_total_l787_787034


namespace find_N_l787_787236

theorem find_N : 
  ∃ N : ℝ, 2 * (3.6 * 0.48 * 2.50) / (0.12 * N * 0.5) = 1600.0000000000002 ∧ N = 0.225 :=
begin
  use 0.225,
  split,
  { sorry },
  { refl }
end

end find_N_l787_787236


namespace coefficient_x_squared_l787_787638

theorem coefficient_x_squared (n : ℕ) (h : (1 + 3 * X)^n = expand $ coeff (x^2)) : n = 4 := by
  sorry

end coefficient_x_squared_l787_787638


namespace find_min_values_l787_787605

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 - 2 * x * y + 6 * y^2 - 14 * x - 6 * y + 72

theorem find_min_values :
  (∀x y : ℝ, f x y ≥ f (15 / 2) (1 / 2)) ∧ f (15 / 2) (1 / 2) = 22.5 :=
by
  sorry

end find_min_values_l787_787605


namespace cost_price_correct_l787_787531

-- Define selling price and profit percentage
def selling_price : ℝ := 1170
def profit_percentage : ℝ := 20 / 100

-- Define the cost price as per the conditions
def cost_price : ℝ := selling_price / (1 + profit_percentage)

-- Prove that the cost price is equal to $975
theorem cost_price_correct : cost_price = 975 := by
  -- Skipping the proof for now.
  sorry

end cost_price_correct_l787_787531


namespace probability_different_colors_l787_787315

-- Define the total number of blue and yellow chips
def blue_chips : ℕ := 5
def yellow_chips : ℕ := 7
def total_chips : ℕ := blue_chips + yellow_chips

-- Define the probability of drawing a blue chip and a yellow chip
def prob_blue : ℚ := blue_chips / total_chips
def prob_yellow : ℚ := yellow_chips / total_chips

-- Define the probability of drawing two chips of different colors
def prob_different_colors := 2 * (prob_blue * prob_yellow)

theorem probability_different_colors :
  prob_different_colors = (35 / 72) := by
  sorry

end probability_different_colors_l787_787315
