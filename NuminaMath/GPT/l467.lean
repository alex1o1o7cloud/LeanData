import Mathlib

namespace anya_lost_games_l467_467057

-- Define the girls playing table tennis
def girls : Type := {Anya Bella Valya Galya Dasha : girls}

-- Define the conditions
def games_played : girls → ℕ
| Anya => 4
| Bella => 6
| Valya => 7
| Galya => 10
| Dasha => 11

-- Total number of games played
def total_games : ℕ := 19
#eval total_games = (games_played Anya + games_played Bella + games_played Valya + games_played Galya + games_played Dasha) / 2

-- The main theorem to be proven
theorem anya_lost_games :
  ∀ i : ℕ, i ∈ {4, 8, 12, 16} ↔ Anya played and lost in game i
:=
by
  sorry

end anya_lost_games_l467_467057


namespace new_point_in_fourth_quadrant_l467_467243

-- Define the initial point P with coordinates (-3, 2)
def P : ℝ × ℝ := (-3, 2)

-- Define the move operation: 4 units to the right and 6 units down
def move (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 4, p.2 - 6)

-- Define the new point after the move operation
def P' : ℝ × ℝ := move P

-- Prove that the new point P' is in the fourth quadrant
theorem new_point_in_fourth_quadrant (x y : ℝ) (h : P' = (x, y)) : x > 0 ∧ y < 0 :=
by
  sorry

end new_point_in_fourth_quadrant_l467_467243


namespace smallest_x_exists_l467_467527

theorem smallest_x_exists {M : ℤ} (h : 2520 = 2^3 * 3^2 * 5 * 7) : 
  ∃ x : ℕ, 2520 * x = M^3 ∧ x = 3675 := 
by {
  sorry
}

end smallest_x_exists_l467_467527


namespace distinct_positive_factors_of_1320_l467_467201

theorem distinct_positive_factors_of_1320 : 
  ∀ n ≥ 0, prime_factorization 1320 = [(2, 3), (3, 1), (11, 1), (5, 1)] → factors_count 1320 = 32 :=
by
  intro n hn h
  sorry

end distinct_positive_factors_of_1320_l467_467201


namespace find_principal_amount_l467_467968

noncomputable def compound_interest_principal (A r t n : ℝ) : ℝ := 
  A / (1 + r/n)^(n*t)

theorem find_principal_amount :
  compound_interest_principal 1792 0.05 2.4 1 ≈ 1590.47 :=
sorry

end find_principal_amount_l467_467968


namespace product_of_roots_of_quadratic_equation_l467_467797

theorem product_of_roots_of_quadratic_equation :
  ∀ (x : ℝ), (x^2 + 14 * x + 48 = -4) → (-6) * (-8) = 48 :=
by
  sorry

end product_of_roots_of_quadratic_equation_l467_467797


namespace sum_of_coefficients_expansion_l467_467986

-- Define the integral a
noncomputable def a : ℝ := (1 / Real.pi) * ∫ x in -2..2, Real.sqrt (4 - x^2)

-- Define the statement to prove
theorem sum_of_coefficients_expansion :
  let b := ∑ i in Finset.range ((10 : ℕ) + 1), (nat.choose 10 i) * (Real.sqrt (Real.cbrt 1 ^ i * (a / Real.sqrt 1) ^ (10 - i)))
  b = 3 ^ 10 := sorry

end sum_of_coefficients_expansion_l467_467986


namespace sign_choice_bound_l467_467720

theorem sign_choice_bound (n : ℕ) (a : ℕ → ℝ) 
    (h1 : n ≥ 2)
    (h2 : ∀ i, 1 ≤ i ∧ i < n → 0 ≤ a i ∧ a i ≤ 2 * a (i - 1)) :
    ∃ (ε : ℕ → ℤ), (∀ i, ε i = 1 ∨ ε i = -1) ∧
    let S := ∑ i in finset.range n, ε i * a i in
    0 ≤ S ∧ S ≤ a 1 := 
sorry

end sign_choice_bound_l467_467720


namespace twenty_eighth_term_l467_467390

/-- Define the sequence where the term with denominator 2^n appears n times. -/
def sequence (n : ℕ) : ℚ :=
  if h : ∃ k : ℕ, ∑ i in finset.range k, i + 1 = n + 1 then
    1 / 2 ^ (finset.card (finset.range k)) 
  else
    0

theorem twenty_eighth_term :
  sequence 27 = 1 / 128 := by
sorry

end twenty_eighth_term_l467_467390


namespace translation_complex_l467_467866

theorem translation_complex (w : ℂ) (z1 z2 t1 t2 : ℂ) :
  z1 = 1 - 2 * complex.I →
  t1 = 4 + 3 * complex.I →
  z2 = 2 + 4 * complex.I →
  t2 = 5 + 9 * complex.I →
  t1 = z1 + w →
  t2 = z2 + w :=
begin
  sorry
end

end translation_complex_l467_467866


namespace C1_rect_eq_C2_elimination_l467_467264

section problem_I

variable (θ : ℝ)

def C1_polar (θ : ℝ) : ℝ := 24 / (4 * (Real.cos θ) + 3 * (Real.sin θ))
def C1_rect (x y : ℝ) : Prop := 4 * x + 3 * y - 24 = 0
def C2_parametric (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
def C2_ordinary (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem C1_rect_eq : 
  ∀ (ρ θ : ℝ), 
  C1_polar θ = ρ → 
  ∃ x y : ℝ, C1_rect x y ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ := 
sorry

theorem C2_elimination :
  ∃ x y : ℝ, ∀ θ : ℝ, C2_parametric θ = (x, y) → C2_ordinary x y :=
sorry

end problem_I

end C1_rect_eq_C2_elimination_l467_467264


namespace hourly_wage_l467_467860

theorem hourly_wage (reps : ℕ) (hours_per_day : ℕ) (days : ℕ) (total_payment : ℕ) :
  reps = 50 →
  hours_per_day = 8 →
  days = 5 →
  total_payment = 28000 →
  (total_payment / (reps * hours_per_day * days) : ℕ) = 14 :=
by
  intros h_reps h_hours_per_day h_days h_total_payment
  -- Now the proof steps can be added here
  sorry

end hourly_wage_l467_467860


namespace domain_f_f_at_neg4_f_at_two_thirds_l467_467116

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (x + 5) + 1 / (x - 2)

theorem domain_f :
  {x : ℝ | x >= -5 ∧ x ≠ 2} = {x : ℝ | x ≥ -5 ∧ x ≠ 2} :=
begin
  sorry
end

theorem f_at_neg4 : f (-4) = 5 / 6 :=
begin
  sorry
end

theorem f_at_two_thirds : f (2 / 3) = (sqrt 51 - 3) / 2 :=
begin
  sorry
end

end domain_f_f_at_neg4_f_at_two_thirds_l467_467116


namespace rotation_of_right_angled_triangle_l467_467483

-- Define the properties and conditions of the problem
structure RightAngledTriangle :=
  (a b c : ℝ)  -- sides of the triangle
  (h₁ : a^2 + b^2 = c^2)  -- condition for a right-angled triangle, c is the hypotenuse

-- Define the geometric body obtained by the rotation
inductive GeometricBody :=
| cone : GeometricBody
| frustum : GeometricBody
| cylinder : GeometricBody
| twoCones : GeometricBody  -- option C from the problem

-- Define the main theorem statement
theorem rotation_of_right_angled_triangle (T : RightAngledTriangle) :
  rotate_360_around_hypotenuse T = GeometricBody.twoCones :=
sorry

end rotation_of_right_angled_triangle_l467_467483


namespace sum_of_prime_factors_2310_l467_467436

theorem sum_of_prime_factors_2310 : 
  let prime_factors := {2, 3, 5, 7, 11};
      n := 2310;
  (∀ p ∈ prime_factors, nat.prime p) ∧ (2310 = 2 * 3 * 5 * 7 * 11) → 
  (∑ p in prime_factors, p) = 28 :=
by
sorry

end sum_of_prime_factors_2310_l467_467436


namespace sum_inequality_l467_467711

theorem sum_inequality (n : ℕ) (h_pos : 0 < n) :
  ∑ k in finset.range n, (k + 1) * real.sqrt (n + 1 - (k + 1)) 
    ≤ (n * (n + 1) * real.sqrt (2 * n + 1)) / (2 * real.sqrt 3) :=
sorry

end sum_inequality_l467_467711


namespace anya_lost_games_l467_467039

/-- 
Five girls (Anya, Bella, Valya, Galya, Dasha) played several games of table tennis. At any given time, two girls were playing,
and the other three were resting. The girl who lost the game went to rest, and her place was taken by the girl who had rested 
the most rounds. There are no ties in table tennis. Anya played 4 games, Bella played 6 games, Valya played 7 games, Galya 
played 10 games, and Dasha played 11 games.

We need to prove that the numbers of all games in which Anya lost are 4, 8, 12, and 16.
-/
theorem anya_lost_games :
  (∃ games : list ℕ,
    games.length = 19 ∧
    (∀ i, i ∈ games → i = 4 ∨ i = 8 ∨ i = 12 ∨ i = 16) ∧
    (∀ i, i ∉ games → true)) :=
by
  sorry

end anya_lost_games_l467_467039


namespace distinct_factors_1320_l467_467214

theorem distinct_factors_1320 : 
  ∃ n : ℕ, n = 24 ∧ 
  let a := 2 in 
  let b := 1 in 
  let c := 1 in 
  let d := 1 in 
  1320 = 2^a * 3^b * 5^c * 11^d :=
begin
  sorry,
end

end distinct_factors_1320_l467_467214


namespace unique_injective_f_solution_l467_467950

noncomputable def unique_injective_function (f : ℝ → ℝ) : Prop := 
  (∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y))

theorem unique_injective_f_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y))
  → (∀ x y : ℝ, f x = f y → x = y) -- injectivity condition
  → ∀ x : ℝ, f x = x :=
sorry

end unique_injective_f_solution_l467_467950


namespace minimum_value_of_expression_l467_467096

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : a > b) (h3 : ab = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 := 
sorry

end minimum_value_of_expression_l467_467096


namespace hyperbola_eccentricity_l467_467283

theorem hyperbola_eccentricity
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (F1 F2 P : EuclideanSpace ℝ (fin 2))
  (C : P ∈ {x : EuclideanSpace ℝ (fin 2) | (x.1^2 / a^2 - x.2^2 / b^2 = 1)})
  (PF1 : dist P F1)
  (PF2 : dist P F2)
  (PF1_plus_PF2_eq : PF1 + PF2 = 6 * a)
  (angle_PF1F2_eq : inner (P - F1) (P - F2) / (dist P F1 * dist P F2) = cos (π / 6)):
  (∃ e : ℝ, e = sqrt 3) ∧ (dist F1 F2 = 2 * sqrt 3 * a) :=
sorry

end hyperbola_eccentricity_l467_467283


namespace num_distinct_factors_1320_l467_467159

theorem num_distinct_factors_1320 : 
  ∃ n : ℕ, n = 1320 ∧ (finset.univ.filter (λ d, d > 0 ∧ 1320 % d = 0)).card = 32 := 
sorry

end num_distinct_factors_1320_l467_467159


namespace even_terms_in_expansion_l467_467647

theorem even_terms_in_expansion (m n : ℤ) (hm : 2 ∣ m) (hn : 2 ∣ n) : 
  (∀ k : ℕ, k ≤ 8 → (∃ r s : ℤ, (binomial 8 k) * (m ^ (8 - k)) * (n ^ k) = 2 * r * s)) :=
sorry

end even_terms_in_expansion_l467_467647


namespace binom_np_p_div_p4_l467_467712

theorem binom_np_p_div_p4 (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (h3 : 3 < p) (hn : n % p = 1) : p^4 ∣ Nat.choose (n * p) p - n := 
sorry

end binom_np_p_div_p4_l467_467712


namespace stratified_sampling_l467_467663

-- Define the constants from the problem
def total_students_first_year : ℕ := 30
def total_students_second_year : ℕ := 40
def students_drawn_first_year : ℕ := 9

-- Define the required constant using the given conditions
def students_drawn_second_year : ℕ := 12

-- State the theorem
theorem stratified_sampling:
  let students_drawn_second_year_computed := (students_drawn_first_year * total_students_second_year) / total_students_first_year in
  students_drawn_second_year_computed = students_drawn_second_year :=
by sorry

end stratified_sampling_l467_467663


namespace part_a_part_b_l467_467448

-- Part (a)
theorem part_a (f : ℝ → ℝ) (n : ℕ)
  (hf_diff : ∀ k ≤ n, Differentiable ℝ (iterated_deriv k f))
  (H : ∀ x y : ℝ, x ≠ y → 
    (f x - ∑ k in finset.range n, (x - y) ^ k / k! * (iterated_deriv k f y))
    / (x - y) ^ n = (iterated_deriv n f x + iterated_deriv n f y) / (n + 1)!): 
  ∃ p : polynomial ℝ, ∀ x, f x = p.eval x ∧ p.degree ≤ n := sorry

-- Part (b)
theorem part_b (f g : ℝ → ℝ) (φ : ℝ → ℝ)
  (H : ∀ x y : ℝ, x ≠ y → (f x - g y) / (x - y) = (φ x + φ y) / 2): 
  (∃ pf : polynomial ℝ, ∀ x, f x = pf.eval x ∧ pf.degree ≤ 2) ∧ ∀ x, g x = f x ∧ φ x = derivative f x := sorry

end part_a_part_b_l467_467448


namespace cosine_largest_angle_l467_467271

-- Given a triangle ABC where sin A : sin B : sin C = 2 : 3 : 4
def triangle_side_ratios (a b c : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = 2 * k ∧ b = 3 * k ∧ c = 4 * k

-- We need to prove that the cosine of the largest angle is -1/4
theorem cosine_largest_angle (a b c : ℝ) (h : triangle_side_ratios a b c) :
  (∃ C : ℝ, cos C = -1 / 4) :=
by sorry

end cosine_largest_angle_l467_467271


namespace length_of_platform_l467_467847

variable (L : ℕ)

theorem length_of_platform
  (train_length : ℕ)
  (time_cross_post : ℕ)
  (time_cross_platform : ℕ)
  (train_length_eq : train_length = 300)
  (time_cross_post_eq : time_cross_post = 18)
  (time_cross_platform_eq : time_cross_platform = 39)
  : L = 350 := sorry

end length_of_platform_l467_467847


namespace octal_subtraction_l467_467511

theorem octal_subtraction : (53 - 27 : ℕ) = 24 :=
by sorry

end octal_subtraction_l467_467511


namespace sqrt_m_minus_n_eq_3_l467_467393

theorem sqrt_m_minus_n_eq_3 :
  ∀ (m n x y : ℝ),
  (m * x - y = 3) →
  (3 * x + n * y = 14) →
  (x = 2) →
  (y = -1) →
  (sqrt (m - n) = 3 ∨ sqrt (m - n) = -3) :=
by
  intros m n x y h1 h2 hx hy
  sorry

end sqrt_m_minus_n_eq_3_l467_467393


namespace parity_of_f_monotonicity_of_f_9_l467_467614

-- Condition: f(x) = x + k / x with k ≠ 0
variable (k : ℝ) (hkn0 : k ≠ 0)
noncomputable def f (x : ℝ) : ℝ := x + k / x

-- 1. Prove the parity of the function is odd
theorem parity_of_f : ∀ x : ℝ, f k (-x) = -f k x := by
  sorry

-- Given condition: f(3) = 6, we derive k = 9
def k_9 : ℝ := 9
noncomputable def f_9 (x : ℝ) : ℝ := x + k_9 / x

-- 2. Prove the monotonicity of the function y = f(x) in the interval (-∞, -3]
theorem monotonicity_of_f_9 : ∀ (x1 x2 : ℝ), x1 < x2 → x1 ≤ -3 → x2 ≤ -3 → f_9 x1 < f_9 x2 := by
  sorry

end parity_of_f_monotonicity_of_f_9_l467_467614


namespace hypotenuse_length_l467_467481

theorem hypotenuse_length (a b : ℕ) (h : a = 9 ∧ b = 12) : ∃ c : ℕ, c = 15 ∧ a * a + b * b = c * c :=
by
  sorry

end hypotenuse_length_l467_467481


namespace volume_of_quadrilateral_pyramid_l467_467752

noncomputable def volume_pyramid (l : ℝ) : ℝ :=
  (l^3 * Real.sqrt 3) / 12

theorem volume_of_quadrilateral_pyramid (l : ℝ) (h₁ : l > 0) (h₂ : ∠60 = π / 3) :
  volume_pyramid l = (l^3 * Real.sqrt 3) / 12 :=
by
  sorry

end volume_of_quadrilateral_pyramid_l467_467752


namespace chalk_pieces_original_l467_467533

theorem chalk_pieces_original (
  siblings : ℕ := 3,
  friends : ℕ := 3,
  pieces_lost : ℕ := 2,
  pieces_added : ℕ := 12,
  pieces_needed_per_person : ℕ := 3
) : 
  ∃ (original_pieces : ℕ), 
    (original_pieces - pieces_lost + pieces_added) = 
    ((1 + siblings + friends) * pieces_needed_per_person) := 
  sorry

end chalk_pieces_original_l467_467533


namespace pages_in_book_l467_467398

theorem pages_in_book (total_digits : ℕ) (h_digits : total_digits = 972) : 
  ∃ n : ℕ, n = 360 ∧ (∑ k in range 10, 1) + (∑ k in range 10..100, 2) + (∑ k in range 100..n + 1, 3) = total_digits := 
by
  cases h_digits
  use 360
  split
  rfl
  sorry

end pages_in_book_l467_467398


namespace find_p_q_u_addition_l467_467288

noncomputable def maximum_length_diameter_intersection (MN A B C : Point) (d : ℝ) : Prop :=
  let NM := diameter MN
  A.isMidpoint NM /\ 
  B.isOnSemicircle NM /\ 
  C.isOnOtherSemicircle NM /\
  lengthSegment B M = 2/5 /\
  ∃ p q u : ℕ, u_prime : ∀ x : ℕ, prime x → ¬ (u % (x * x) = 0) ∧
  d = p - q * sqrt u ∧ p + q + u = 16

theorem find_p_q_u_addition (MN A B C : Point) (d : ℝ) :
  maximum_length_diameter_intersection MN A B C d :=
by
  sorry

end find_p_q_u_addition_l467_467288


namespace complex_third_quadrant_range_l467_467989

theorem complex_third_quadrant_range (m : ℝ) : 
  ((m + 4 < 0) ∧ (m - 2 < 0)) → m < -4 :=
by
  intro h
  cases h with h1 h2
  sorry

end complex_third_quadrant_range_l467_467989


namespace sum_lcm_eq_72_l467_467819

theorem sum_lcm_eq_72 (s : Finset ℕ) 
  (h1 : ∀ ν ∈ s, Nat.lcm ν 24 = 72) :
  s.sum id = 180 :=
by
  have h2 : ∀ ν, ν ∈ s → ∃ n, ν = 3 * n := 
    by sorry
  have h3 : ∀ n, ∃ ν, ν = 3 * n ∧ ν ∈ s := 
    by sorry
  sorry

end sum_lcm_eq_72_l467_467819


namespace anna_and_bob_play_together_l467_467316

-- Definitions based on the conditions
def total_players := 12
def matches_per_week := 2
def players_per_match := 6
def anna_and_bob := 2
def other_players := total_players - anna_and_bob
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Lean statement based on the equivalent proof problem
theorem anna_and_bob_play_together :
  combination other_players (players_per_match - anna_and_bob) = 210 := by
  -- To use Binomial Theorem in Lean
  -- The mathematical equivalent is C(10, 4) = 210
  sorry

end anna_and_bob_play_together_l467_467316


namespace find_MN_l467_467680

variables (O A B D M N C : Type) [Group O] [AddCommGroup A] [AddCommGroup B]
  [AddCommGroup D] [AddCommGroup M] [AddCommGroup N] [AddCommGroup C]
  (a b : A) (oa ob od om on oc : O)
  (h_a : \overrightarrow{OA} = a)
  (h_b : \overrightarrow{OB} = b)
  (h_BM : ∃ (BC : B), BM = ⅓ * BC)
  (h_CN : ∃ (CD : C), CN = ⅓ * CD)
  (h_C : ∃ (O : Type) (A B D : O) (h_C : C ∈ lines AB ∩ lines OD), true)
  (h_parallelogram : ∃ (A B O D : Type), parallelogram (A O B D))

theorem find_MN :
  ∃ (MN : M), MN = ½ * a - ⅙ * b :=
sorry

end find_MN_l467_467680


namespace length_of_each_cut_section_xiao_hong_age_l467_467507

theorem length_of_each_cut_section (x : ℝ) (h : 60 - 2 * x = 10) : x = 25 := sorry

theorem xiao_hong_age (y : ℝ) (h : 2 * y + 10 = 30) : y = 10 := sorry

end length_of_each_cut_section_xiao_hong_age_l467_467507


namespace regular_price_of_Pony_jeans_l467_467069

theorem regular_price_of_Pony_jeans 
(fox_price : ℝ)
(pony_discount_rate : ℝ)
(sum_discount_rate : ℝ) 
(total_savings : ℝ)
(pairs_fox : ℝ) 
(pairs_pony : ℝ) 
(h_fox_price : fox_price = 15)
(h_pony_discount_rate : pony_discount_rate = 13.999999999999993 / 100)
(h_sum_discount_rate : sum_discount_rate = 22 / 100)
(h_total_savings : total_savings = 8.64)
(h_pairs_fox : pairs_fox = 3)
(h_pairs_pony : pairs_pony = 2) :
let pony_price := 5.04 / 0.28 in
pony_price = 18 :=
by
  sorry

end regular_price_of_Pony_jeans_l467_467069


namespace evaluate_expression_l467_467464

theorem evaluate_expression :
  (116 * 2 - 116) - (116 * 2 + 104) / (3 ^ 2) + (104 * 3 - 104) - (104 * 3 + 94) / (4 ^ 2) ≈ 261.291 :=
by
  -- We need to compute each step taken in the mathematical evaluation
  have h1 : 116 * 2 - 116 = 116 := by norm_num,
  have h2 : 116 * 2 + 104 = 336 := by norm_num,
  have h3 : 104 * 3 - 104 = 208 := by norm_num,
  have h4 : 104 * 3 + 94 = 406 := by norm_num,

  -- Performing the necessary divisions
  have h5 : 336 / 9 ≈ 37.333 := by norm_num,
  have h6 : 406 / 16 ≈ 25.375 := by norm_num,

  -- Substituting back and solving the final arithmetic operations
  have h7 : 116 - 37.333 + 208 - 25.375 ≈ 78.666 + 182.625 := by norm_num,
  have h8 : 78.666 + 182.625 ≈ 261.291 := by norm_num,

  -- Conclude the proof by combining all steps
  show (116 * 2 - 116) - (116 * 2 + 104) / (3 ^ 2) + (104 * 3 - 104) - (104 * 3 + 94) / (4 ^ 2) ≈ 261.291
  from sorry

end evaluate_expression_l467_467464


namespace solve_cubic_root_eq_l467_467956

theorem solve_cubic_root_eq (x : ℝ) : (∃ x : ℝ, cbrt (5 - x) = -5 / 3) →
  x = 260 / 27 :=
by
  -- Add proof here
  sorry

end solve_cubic_root_eq_l467_467956


namespace distinct_four_digit_numbers_count_l467_467147

theorem distinct_four_digit_numbers_count (digits : Finset ℕ) (h : digits = {1, 2, 3, 4, 5}) :
  (∃ (numbers : Finset (ℕ × ℕ × ℕ × ℕ)), 
   (∀ (a b c d : ℕ), (a, b, c, d) ∈ numbers → a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
   numbers.card = 120) :=
begin
  sorry
end

end distinct_four_digit_numbers_count_l467_467147


namespace triangle_largest_angle_l467_467756

theorem triangle_largest_angle (y : ℝ) (h : 45 + 60 + y = 180) : y = 75 :=
by
  have h₁ : 45 + 60 = 105 := by norm_num
  have h₂ : y = 180 - 105 := by linarith [h, h₁]
  exact h₂

end triangle_largest_angle_l467_467756


namespace sqrt_11_plus_1_bounds_l467_467946

theorem sqrt_11_plus_1_bounds :
  4 < Real.sqrt 11 + 1 ∧ Real.sqrt 11 + 1 < 5 :=
by
  have h1 : Real.sqrt 9 = 3 := by norm_num
  have h2 : Real.sqrt 16 = 4 := by norm_num
  have h3 : 9 < 11 < 16 := by norm_num
  sorry

end sqrt_11_plus_1_bounds_l467_467946


namespace loss_percentage_first_book_l467_467223

-- Define constants for the problem
def total_cost : ℝ := 450
def C1 : ℝ := 262.5
def C2 : ℝ := total_cost - C1
def gain_percentage : ℝ := 19 / 100
def SP2 : ℝ := C2 * (1 + gain_percentage)
def SP1 : ℝ := SP2

-- Define the loss percentage formula
def loss_percentage (C : ℝ) (SP : ℝ) : ℝ := ((C - SP) / C) * 100

-- The theorem to prove
theorem loss_percentage_first_book :
  loss_percentage C1 SP1 = 15 := 
sorry

end loss_percentage_first_book_l467_467223


namespace intersection_complement_eq_l467_467286

open Set

noncomputable def U := ℝ
def A := {-3, -2, -1, 0, 1, 2}
def B := {x : ℝ | x >= 1}

theorem intersection_complement_eq : A ∩ (U \ B) = {-3, -2, -1, 0} := 
by sorry

end intersection_complement_eq_l467_467286


namespace perimeter_triangle_APR_l467_467427

theorem perimeter_triangle_APR (A B C P R Q : Point) (circle : Circle)
  (hAB : Tangent_from_to A B circle) (hAC : Tangent_from_to A C circle)
  (hP : Intersect_segment AB P) (hR : Intersect_segment AC R) 
  (hTang_Q : Tangent_circle Q circle)
  (hTouch_P : Touch_at Q circle B)
  (hTouch_R : Touch_at Q circle C)
  (hLen_AB : AB.length = 20) :
  Perimeter (Triangle A P R) = 40 :=
sorry

end perimeter_triangle_APR_l467_467427


namespace constant_term_in_expansion_term_with_max_binomial_coefficient_l467_467980

theorem constant_term_in_expansion
  (n : ℕ)
  (h : (comb n 4 * 2^4) / (comb n 2 * 2^2) = 56 / 3) :
  let C := binomial n 2 * (2^2) in C = 180 := 
begin
  sorry
end

theorem term_with_max_binomial_coefficient (x : ℝ) :
  x = 4 → let term := binomial 10 5 * 2^5 * 4^(-15 / 2) in term = (63 / 256) :=
begin
  intros hx,
  subst hx,
  sorry
end

end constant_term_in_expansion_term_with_max_binomial_coefficient_l467_467980


namespace find_integer_solutions_l467_467951

theorem find_integer_solutions (x : ℤ) :
  3 * |2 * x + 1| + 6 < 24 ↔ x ∈ {-3, -2, -1, 0, 1, 2} :=
by
  sorry

end find_integer_solutions_l467_467951


namespace count_1320_factors_l467_467166

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end count_1320_factors_l467_467166


namespace find_triplets_l467_467958

theorem find_triplets (a k m : ℕ) (hpos_a : 0 < a) (hpos_k : 0 < k) (hpos_m : 0 < m) (h_eq : k + a^k = m + 2 * a^m) :
  ∃ t : ℕ, 0 < t ∧ (a = 1 ∧ k = t + 1 ∧ m = t) :=
by
  sorry

end find_triplets_l467_467958


namespace problem1_l467_467463

def term1 : ℝ := (-1 / 3)⁻²
def term2 : ℤ := 4 * (-1)^2023
def term3 : ℤ := abs (-2^3)
def term4 : ℝ := (Real.pi - 5)^0

theorem problem1 : term1 + term2 - term3 + term4 = -2 :=
by
  sorry

end problem1_l467_467463


namespace find_general_term_find_sum_first_n_terms_l467_467081

noncomputable theory

open_locale big_operators

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (c : ℕ → ℝ) (T : ℕ → ℝ)

-- Given Condition for S_n
axiom S_n_def : ∀ n : ℕ, 0 < n → S n = 2 * a n - 2

-- Definition of c_n
def c_n (n : ℕ) : ℝ := (n + 1) * a n

-- Sum of first n terms of sequence {c_n}
def T_n (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), c i

-- Prove a_n = 2^n given S_n = 2 * a_n - 2
theorem find_general_term : ∀ n : ℕ, 0 < n → a n = 2^n := 
sorry

-- Prove T_n = n * 2^(n+1) given c_n = (n+1) * a_n
theorem find_sum_first_n_terms (n : ℕ) : T n = n * 2^(n+1) := 
sorry

end find_general_term_find_sum_first_n_terms_l467_467081


namespace find_f_log_a_sqrt2_minus_1_l467_467115

noncomputable def f (x : ℝ) : ℝ := (2 / (1 + 2^x)) + (1 / (1 + 4^x))

theorem find_f_log_a_sqrt2_minus_1 (a : ℝ) (h_a : a > 1) 
  (h_f : f (Real.log a (Real.sqrt 2 + 1)) = 1) : 
  f (Real.log a (Real.sqrt 2 - 1)) = 2 := by
  sorry

end find_f_log_a_sqrt2_minus_1_l467_467115


namespace anya_lost_games_correct_l467_467044

-- Definition of players and their game counts
def Players := {Anya, Bella, Valya, Galya, Dasha}
def games_played : Players → ℕ
| Anya   := 4
| Bella  := 6
| Valya  := 7
| Galya  := 10
| Dasha  := 11

-- Total games played by all players
def total_games_played : ℕ := Players.to_finset.sum games_played

-- Total number of games, considering each game involves two players
def total_number_of_games : ℕ := total_games_played / 2

-- The set of games in which Anya lost
def anya_lost_games : set ℕ := {4, 8, 12, 16}

theorem anya_lost_games_correct :
  ∀ i ∈ anya_lost_games, -- For each game in anya_lost_games
  anya_lost_games = {4, 8, 12, 16} := 
by
  sorry

end anya_lost_games_correct_l467_467044


namespace triangles_at_A_triangles_at_F_l467_467990

def points := ["A", "B", "C", "D", "E", "F"]

-- Problem 1: Number of triangles with vertex at A
theorem triangles_at_A : 
  ∃ (n : ℕ), n = 9 ∧ (∀ (t : list string), 
  t.length = 3 ∧ t.head = "A" → t.tail ⊆ points → t ≠ ["A", "A", "A"]) := 
sorry

-- Problem 2: Number of triangles with vertex at F
theorem triangles_at_F : 
  ∃ (n : ℕ), n = 9 ∧ (∀ (t : list string), 
  t.length = 3 ∧ t.head = "F" → t.tail ⊆ points → t ≠ ["F", "F", "F"]) := 
sorry

end triangles_at_A_triangles_at_F_l467_467990


namespace right_triangle_area_l467_467522

-- Definition of degrees and radians for angle, and trigonometric functions for computation
noncomputable def deg_to_rad (deg : Float) : Float :=
  deg * Float.pi / 180

noncomputable def cos_deg (deg : Float) : Float :=
  Float.cos (deg_to_rad deg)

noncomputable def tan_deg (deg : Float) : Float :=
  Float.tan (deg_to_rad deg)

-- Defining the problem
theorem right_triangle_area
  (alpha_deg : Float := 38 + 40 / 60) -- Angle α in degrees
  (fa : Float := 7.8) -- Angle bisector fa in cm
  : 1 / 2 * fa^2 * (cos_deg (alpha_deg / 2))^2 * tan_deg alpha_deg ≈ 21.67 := by
    sorry

end right_triangle_area_l467_467522


namespace anya_lost_games_l467_467051

-- Define the girls playing table tennis
def girls : Type := {Anya Bella Valya Galya Dasha : girls}

-- Define the conditions
def games_played : girls → ℕ
| Anya => 4
| Bella => 6
| Valya => 7
| Galya => 10
| Dasha => 11

-- Total number of games played
def total_games : ℕ := 19
#eval total_games = (games_played Anya + games_played Bella + games_played Valya + games_played Galya + games_played Dasha) / 2

-- The main theorem to be proven
theorem anya_lost_games :
  ∀ i : ℕ, i ∈ {4, 8, 12, 16} ↔ Anya played and lost in game i
:=
by
  sorry

end anya_lost_games_l467_467051


namespace probability_of_receiving_1_l467_467668

-- Define the probabilities and events
def P_A : ℝ := 0.5
def P_not_A : ℝ := 0.5
def P_B_given_A : ℝ := 0.9
def P_not_B_given_A : ℝ := 0.1
def P_B_given_not_A : ℝ := 0.05
def P_not_B_given_not_A : ℝ := 0.95

-- The main theorem that needs to be proved
theorem probability_of_receiving_1 : 
  (P_A * P_not_B_given_A + P_not_A * P_not_B_given_not_A) = 0.525 := by
  sorry

end probability_of_receiving_1_l467_467668


namespace no_solution_for_vectors_l467_467944

theorem no_solution_for_vectors {t s k : ℝ} :
  (∃ t s : ℝ, (1 + 6 * t = -1 + 3 * s) ∧ (3 + 1 * t = 4 + k * s)) ↔ k ≠ 0.5 :=
sorry

end no_solution_for_vectors_l467_467944


namespace sum_of_solutions_l467_467397

theorem sum_of_solutions : 
  let cond := λ x : ℝ, |x + 2| = 2 * |x - 2|
  finset.univ.filter cond |>.sum = 20/3 :=
begin
  sorry
end

end sum_of_solutions_l467_467397


namespace series_sum_l467_467922

theorem series_sum : 
  (∑ n : ℕ, (2^n) / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end series_sum_l467_467922


namespace cos_minus_sin2_alpha_l467_467979

noncomputable def alpha := ℝ
axiom tan_sub_pi_div_4_alpha (α : α) : Real.tan (Real.pi / 4 - α) = -1 / 2
axiom alpha_in_interval (α : α) : α > Real.pi ∧ α < 3 * Real.pi / 2

theorem cos_minus_sin2_alpha (α : α) (h1 : tan_sub_pi_div_4_alpha α) (h2 : alpha_in_interval α) : 
  Real.cos α - Real.sin (2 * α) = - (6 + Real.sqrt 10) / 10 :=
sorry

end cos_minus_sin2_alpha_l467_467979


namespace flag_blue_percentage_proof_l467_467517

noncomputable def flag_area_blue_percentage (s w : ℝ) (h : w * (4 * s - 3 * w) = 0.45 * s^2) : ℝ :=
  let area_triangle := (Real.sqrt 3 * w^2) / 16
  in (area_triangle / (s^2)) * 100

theorem flag_blue_percentage_proof (s w : ℝ) (h : w * (4 * s - 3 * w) = 0.45 * s^2) :
  flag_area_blue_percentage s w h = 1.08 :=
sorry

end flag_blue_percentage_proof_l467_467517


namespace parallel_conditions_l467_467306

-- Definitions of the lines
def l1 (m : ℝ) (x y : ℝ) : Prop := m * x + 3 * y - 6 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y + 2 = 0

-- Definition of parallel lines
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, l1 x y → l2 x y

-- Proof statement
theorem parallel_conditions (m : ℝ) :
  parallel (l1 m) (l2 m) ↔ (m = 1 ∨ m = -6) :=
by
  intros
  sorry

end parallel_conditions_l467_467306


namespace find_a_n_find_S_n_l467_467259

variable {a : ℕ → ℝ}

-- Conditions
axiom h_a2 : a 2 = 3
axiom h_a5 : a 5 = 81

-- Definitions from the conditions
def a_n := λ n : ℕ, 3^(n-1)
def b_n := λ n : ℕ, Real.log 3 (a_n n)
def S_n (n : ℕ) := (n * (n - 1)) / 2

-- Proof statements
theorem find_a_n : ∀ n : ℕ, a n = a_n n :=
by
  sorry

theorem find_S_n : ∀ n : ℕ, (∑ i in (Finset.range n).map Finset.N _ = n), b i = S_n n :=
by
  sorry

end find_a_n_find_S_n_l467_467259


namespace find_fractions_l467_467548

noncomputable def fractions_to_sum_86_111 : Prop :=
  ∃ (a b d₁ d₂ : ℕ), 0 < a ∧ 0 < b ∧ d₁ ≤ 100 ∧ d₂ ≤ 100 ∧
  Nat.gcd a d₁ = 1 ∧ Nat.gcd b d₂ = 1 ∧
  (a: ℚ) / d₁ + (b: ℚ) / d₂ = 86 / 111

theorem find_fractions : fractions_to_sum_86_111 :=
  sorry

end find_fractions_l467_467548


namespace value_of_f_neg1_plus_f_4_l467_467576

def f(x : ℝ) : ℝ :=
if x >= 2 then
  2 * x - 1
else
  -x^2 + 3 * x

theorem value_of_f_neg1_plus_f_4 : f (-1) + f 4 = 3 :=
sorry

end value_of_f_neg1_plus_f_4_l467_467576


namespace distinct_factors_1320_l467_467175

theorem distinct_factors_1320 :
  let p := 1320,
  prime_factors := [(2, 2), (3, 1), (5, 1), (11, 1)],
  num_factors := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in num_factors = 24 := by
  sorry

end distinct_factors_1320_l467_467175


namespace max_siskins_on_poles_l467_467882

-- Definitions based on problem conditions
def pole : Type := ℕ
def siskins (poles : pole) : Prop := poles ≤ 25
def adjacent (p₁ p₂ : pole) : Prop := (p₁ = p₂ + 1) ∨ (p₁ = p₂ - 1)

-- Given conditions
def conditions (p : pole → bool) : Prop :=
  ∀ p₁ p₂ : pole, p p₁ = true → p p₂ = true → adjacent p₁ p₂ → false

-- Main problem statement
theorem max_siskins_on_poles : ∃ p : pole → bool, (∀ i : pole, p i = true → siskins i) ∧ (conditions p) ∧ (∑ i in finset.range 25, if p i then 1 else 0 = 24) :=
begin
  sorry
end

end max_siskins_on_poles_l467_467882


namespace distinct_factors_1320_l467_467219

theorem distinct_factors_1320 : 
  ∃ n : ℕ, n = 24 ∧ 
  let a := 2 in 
  let b := 1 in 
  let c := 1 in 
  let d := 1 in 
  1320 = 2^a * 3^b * 5^c * 11^d :=
begin
  sorry,
end

end distinct_factors_1320_l467_467219


namespace count_1320_factors_l467_467174

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end count_1320_factors_l467_467174


namespace anya_lost_games_l467_467043

/-- 
Five girls (Anya, Bella, Valya, Galya, Dasha) played several games of table tennis. At any given time, two girls were playing,
and the other three were resting. The girl who lost the game went to rest, and her place was taken by the girl who had rested 
the most rounds. There are no ties in table tennis. Anya played 4 games, Bella played 6 games, Valya played 7 games, Galya 
played 10 games, and Dasha played 11 games.

We need to prove that the numbers of all games in which Anya lost are 4, 8, 12, and 16.
-/
theorem anya_lost_games :
  (∃ games : list ℕ,
    games.length = 19 ∧
    (∀ i, i ∈ games → i = 4 ∨ i = 8 ∨ i = 12 ∨ i = 16) ∧
    (∀ i, i ∉ games → true)) :=
by
  sorry

end anya_lost_games_l467_467043


namespace max_value_of_f_l467_467610

noncomputable def f (x : ℝ) : ℝ := (1 + real.sqrt 3 * real.tan x) * real.cos x

theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ set.Icc 0 (real.pi / 6) ∧ f x = real.sqrt 3 := 
sorry

end max_value_of_f_l467_467610


namespace car_y_start_time_minutes_l467_467514

theorem car_y_start_time_minutes :
  let avg_speed_X := 35 -- Car X's average speed in miles per hour
  let avg_speed_Y := 38 -- Car Y's average speed in miles per hour
  let distance_traveled_X := 490 -- Distance Car X traveled in miles
  let distance_till_stop := distance_traveled_X / avg_speed_X -- Total time Car X traveled after Car Y started, in hours
  let time_Y := distance_traveled_X / avg_speed_Y -- Time Car Y traveled until both cars stopped, in hours
  let time_X_before_Y := distance_till_stop - time_Y -- Time Car X traveled before Car Y started, in hours
  in time_X_before_Y * 60 = 66.318 := sorry

end car_y_start_time_minutes_l467_467514


namespace general_term_a_n_limit_of_sequence_l467_467618

noncomputable def a (n : ℕ) : ℕ := 
  if h : n ≥ 2 then 2 * n - 1 else 0

theorem general_term_a_n (n : ℕ) (h : n ≥ 2) : 
  a n = 2 * n - 1 := by
  simp [a, h]

theorem limit_of_sequence :
  (filter.at_top.tendsto (λ n : ℕ, (1 - (1 / (a n))) ^ n) (𝓝 (real.exp (-1 / 2)))) := by
-- proof omitted
  sorry

end general_term_a_n_limit_of_sequence_l467_467618


namespace divisible_by_4_divisible_by_8_divisible_by_16_l467_467709

variable (A B C D : ℕ)
variable (hB : B % 2 = 0)

theorem divisible_by_4 (h1 : (A + 2 * B) % 4 = 0) : 
  (1000 * D + 100 * C + 10 * B + A) % 4 = 0 :=
sorry

theorem divisible_by_8 (h2 : (A + 2 * B + 4 * C) % 8 = 0) :
  (1000 * D + 100 * C + 10 * B + A) % 8 = 0 :=
sorry

theorem divisible_by_16 (h3 : (A + 2 * B + 4 * C + 8 * D) % 16 = 0) :
  (1000 * D + 100 * C + 10 * B + A) % 16 = 0 :=
sorry

end divisible_by_4_divisible_by_8_divisible_by_16_l467_467709


namespace most_reasonable_sampling_l467_467784

-- Definitions
def significant_stage_differences : Prop := 
  ∃ vision_conditions : Type, ∃ stages : Type, ∃ (f : stages → vision_conditions), 
  f "primary_school" ≠ f "middle_school" ∧ f "middle_school" ≠ f "high_school"

def negligible_gender_differences : Prop := 
  ∀ (vision_conditions : Type) (genders : Type) (f : genders → vision_conditions), 
  f "boy" = f "girl"

def simple_random_sampling : Type := sorry  -- The type representing simple random sampling.

def stratified_sampling_by_gender : Type := sorry  -- The type representing stratified sampling by gender.

def stratified_sampling_by_stage : Type := sorry  -- The type representing stratified sampling by educational stage.

def systematic_sampling : Type := sorry  -- The type representing systematic sampling.

-- Prove the most reasonable sampling method
theorem most_reasonable_sampling :
  significant_stage_differences →
  negligible_gender_differences →
  (∀ method : Type, method = simple_random_sampling ∨
                    method = stratified_sampling_by_gender ∨
                    method = stratified_sampling_by_stage ∨
                    method = systematic_sampling) →
  stratified_sampling_by_stage = stratified_sampling_by_stage :=
by
  intros h1 h2 h3
  sorry

end most_reasonable_sampling_l467_467784


namespace anya_lost_games_l467_467036

def girl_count := 5
def anya_games := 4
def bella_games := 6
def valya_games := 7
def galya_games := 10
def dasha_games := 11
def total_games := 19

theorem anya_lost_games :
  (anya_games + bella_games + valya_games + galya_games + dasha_games) / 2 = total_games →
  ∀ games : list ℕ,
    games.length = total_games →
    (∀ g ∈ games, g > 0 ∧ g ≤ total_games) →
    (∀ g ∈ [anya_games, bella_games, valya_games, galya_games, dasha_games], g ≤ total_games) →
    ∀ a b : ℕ, a ≠ b →
    (anya_games, games.nth 3 = some 4) ∧
    (anya_games, games.nth 7 = some 8) ∧
    (anya_games, games.nth 11 = some 12) ∧
    (anya_games, games.nth 15 = some 16)
:= sorry

end anya_lost_games_l467_467036


namespace series_convergence_l467_467927

theorem series_convergence :
  ∑ n : ℕ, (2^n) / (3^(2^n) + 1) = 1 / 2 :=
by
  sorry

end series_convergence_l467_467927


namespace andrew_total_homeless_shelter_donation_l467_467499

-- Given constants and conditions
def bake_sale_total : ℕ := 400
def ingredients_cost : ℕ := 100
def piggy_bank_donation : ℕ := 10

-- Intermediate calculated values
def remaining_total : ℕ := bake_sale_total - ingredients_cost
def shelter_donation_from_bake_sale : ℕ := remaining_total / 2

-- Final goal statement
theorem andrew_total_homeless_shelter_donation :
  shelter_donation_from_bake_sale + piggy_bank_donation = 160 :=
by
  -- Proof to be provided.
  sorry

end andrew_total_homeless_shelter_donation_l467_467499


namespace Anya_loss_games_l467_467029

noncomputable def game_indices_Anya_lost (Anya Bella Valya Galya Dasha : ℕ) : Set ℕ :=
  {4, 8, 12, 16}

theorem Anya_loss_games
  (Anya Bella Valya Galya Dasha total_games : ℕ)
  (h1 : Anya = 4)
  (h2 : Bella = 6)
  (h3 : Valya = 7)
  (h4 : Galya = 10)
  (h5 : Dasha = 11)
  (h6 : total_games = (Anya + Bella + Valya + Galya + Dasha) / 2)
  (h7 : total_games = 19) :
  game_indices_Anya_lost Anya Bella Valya Galya Dasha = {4, 8, 12, 16} :=
  by sorry

end Anya_loss_games_l467_467029


namespace probability_function_increasing_interval_l467_467071

open Classical

theorem probability_function_increasing_interval :
  let domain_a := {0, 1, 2}
  let domain_b := {-1, 1, 3, 5}
  let total_cases := 12
  let favorable_cases := 5
  let probability := (favorable_cases : ℚ) / total_cases
  probability = 5 / 12 :=
by
  sorry

end probability_function_increasing_interval_l467_467071


namespace product_of_remaining_numbers_l467_467415

-- Definitions of the initial conditions
def initial_counts : List (ℕ × ℕ) := [(1, 2006), (2, 2007), (3, 2008), (4, 2009), (5, 2010)]

-- Definition of the type of operation
def operation (counts : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  sorry -- Define how an operation transforms the counts

-- Proposition that represents the problem statement
theorem product_of_remaining_numbers :
  ∃ (counts : List (ℕ × ℕ)),
  (counts = initial_counts) → -- Start with the initial counts
  (∀ op, counts = operation counts) → -- Apply a finite number of valid operations
  (counts.length = 2) → -- End up with exactly two different numbers on the blackboard
  (counts.nth 0).getOrElse (0, 0) * (counts.nth 1).getOrElse (0, 0) = 8 := -- Their product is 8
sorry

end product_of_remaining_numbers_l467_467415


namespace anya_game_losses_l467_467061

theorem anya_game_losses (games : ℕ → ℕ) (anya_games bella_games valya_games galya_games dasha_games : ℕ)
  (H_total : anya_games = 4 ∧ bella_games = 6 ∧ valya_games = 7 ∧ galya_games = 10 ∧ dasha_games = 11) :
  (games anya_games = 4 ∧ games bella_games = 6 ∧
   games valya_games = 7 ∧ games galya_games = 10 ∧
   games dasha_games = 11) →  [4, 8, 12, 16] := sorry

end anya_game_losses_l467_467061


namespace find_two_irreducible_fractions_l467_467553

theorem find_two_irreducible_fractions :
  ∃ (a b d1 d2 : ℕ), 
    (1 ≤ a) ∧ 
    (1 ≤ b) ∧ 
    (gcd a d1 = 1) ∧ 
    (gcd b d2 = 1) ∧ 
    (1 ≤ d1) ∧ 
    (d1 ≤ 100) ∧ 
    (1 ≤ d2) ∧ 
    (d2 ≤ 100) ∧ 
    (a / (d1 : ℚ) + b / (d2 : ℚ) = 86 / 111) := 
by {
  sorry
}

end find_two_irreducible_fractions_l467_467553


namespace series_convergence_l467_467929

theorem series_convergence :
  ∑ n : ℕ, (2^n) / (3^(2^n) + 1) = 1 / 2 :=
by
  sorry

end series_convergence_l467_467929


namespace rationalize_denominator_l467_467339

theorem rationalize_denominator (a b c : ℝ) (h1 : a = 7) (h2 : b = √98) (h3 : √98 = 7 * √2) :
  a / b * √2 = c ↔ c = √2 / 2 := by
  sorry

end rationalize_denominator_l467_467339


namespace concurrency_AM_DN_XY_l467_467503

/-- Given points A, B, C, D in sequence on a straight line,
two circles with diameters AC and BD intersecting at X and Y,
line XY intersecting BC at Z, and point P on line XY distinct from Z.
If the circle with diameter AC intersects line CP again at N,
prove that lines AM, DN, and XY are concurrent. -/
theorem concurrency_AM_DN_XY (A B C D X Y Z P N : Point)
    (h1 : distinct_points [A, B, C, D])
    (h2 : on_line [A, B, C, D])
    (h3 : on_circle X (diam_circle A C))
    (h4 : on_circle Y (diam_circle A C))
    (h5 : on_circle X (diam_circle B D))
    (h6 : on_circle Y (diam_circle B D))
    (h7 : collinear [X, Y, Z])
    (h8 : collinear [B, C, Z])
    (h9 : P ≠ Z ∧ collinear [X, Y, P])
    (h10 : on_circle N (diam_circle A C) ∧ collinear [C, P, N]) :
    concurrent_lines [line_through A M, line_through D N, line_through X Y] :=
sorry

end concurrency_AM_DN_XY_l467_467503


namespace max_siskins_on_poles_l467_467869

theorem max_siskins_on_poles (n : ℕ) (h : n = 25) :
  ∃ k : ℕ, k = 24 ∧ (∀ (poless: Fin n → ℕ) (siskins: Fin n → ℕ),
     (∀ i: Fin n, siskins i ≤ 1) 
     ∧ (∀ i: Fin n, (siskins i = 1 → (poless i = 0)))
     ∧ poless 0 = 0
     → ( ∀ j: Fin n, (j < n → siskins j + siskins (j+1) < 2)) 
     ∧ (k ≤ n)
     ∧ ( ∀ l: Fin n, ((l < k → siskins l = 1) →
       ((k ≤ l < n → siskins l = 0)))))
  sorry

end max_siskins_on_poles_l467_467869


namespace correct_propositions_l467_467377

open Real

noncomputable def represents_circle (t : ℝ) : Prop :=
  t = 5 / 2

noncomputable def represents_hyperbola (t : ℝ) : Prop :=
  (4 - t) * (t - 1) < 0

noncomputable def represents_ellipse_x_foci (t : ℝ) : Prop :=
  1 < t ∧ t < 5 / 2

theorem correct_propositions (t : ℝ) :
  ((∃ t, ¬represents_circle t) → False) →
  ((∃ t, represents_hyperbola t) → True) →
  ((∃ t, represents_ellipse_x_foci t) → True) →
  ({3, 4} : set ℕ) :=
begin
  intro h1,
  intro h2,
  intro h3,
  exact {3, 4},
end

end correct_propositions_l467_467377


namespace solve_cubic_root_eq_l467_467957

theorem solve_cubic_root_eq (x : ℝ) : (∃ x : ℝ, cbrt (5 - x) = -5 / 3) →
  x = 260 / 27 :=
by
  -- Add proof here
  sorry

end solve_cubic_root_eq_l467_467957


namespace Anya_loss_games_l467_467017

noncomputable def gamesPlayed : Nat := 19

constant Anya_games : Nat := 4
constant Bella_games : Nat := 6
constant Valya_games : Nat := 7
constant Galya_games : Nat := 10
constant Dasha_games : Nat := 11

constant gameResults : Fin 19 → Option String
constant lost (g : Nat) (p : String) : Prop

axiom Anya_game_indices : ∀ (i : Fin 4), lost (gameResults i.val) "Anya" → i.val = 4 * i + 4

theorem Anya_loss_games : list Nat :=
have Anya_lost_games : list Nat := [4, 8, 12, 16],
Anya_lost_games

end Anya_loss_games_l467_467017


namespace exists_triangle_l467_467778

universe u
variable {α : Type u}
variable [Fintype α] [DecidableEq α]

-- Define the structure for the statement: each person knows at least ⌊n/2⌋ people
def knows_at_least (P : α → α → Prop) (n : ℕ) [Fintype α] : Prop :=
  ∀ a, Fintype.card {b | P a b} ≥ n / 2

-- Define the condition: for any ⌊n/2⌋ people, either two of them know each other, 
-- or among the remaining people, two know each other
def condition2 (P : α → α → Prop) (h : ∀ a, Fintype.card {b | P a b} ≥ n / 2) [Finite α] (n : ℕ) :=
  ∀ S : Finset α, S.card = n / 2 → 
    (∃ a b, a ∈ S ∧ b ∈ S ∧ P a b) ∨
    (∃ a b, a ∉ S ∧ b ∉ S ∧ P a b)

-- The main theorem statement
theorem exists_triangle (α : Type u) [Finite α] [DecidableEq α] (P : α → α → Prop) (n : ℕ) (hn : 6 ≤ n) :
  knows_at_least P n →
  condition2 P knows_at_least (Fintype.card α) n →
  ∃ (x y z : α), P x y ∧ P y z ∧ P z x :=
sorry

end exists_triangle_l467_467778


namespace smallest_x_l467_467485

noncomputable def digitSum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_x (x a : ℕ) (h1 : a = 100 * x + 4950)
  (h2 : digitSum a = 50) :
  x = 99950 :=
by sorry

end smallest_x_l467_467485


namespace trees_total_count_l467_467416

theorem trees_total_count (D P : ℕ) 
  (h1 : D = 350 ∨ P = 350)
  (h2 : 300 * D + 225 * P = 217500) :
  D + P = 850 :=
by
  sorry

end trees_total_count_l467_467416


namespace oreo_milk_combinations_l467_467896

theorem oreo_milk_combinations :
  let oreo_flavors := 6
  let milk_flavors := 4
  let total_flavors := oreo_flavors + milk_flavors
  let alpha_combinations (n : Nat) := Nat.choose total_flavors n
  let beta_combinations (n : Nat) :=
    if n = 1 then oreo_flavors
    else if n = 2 then Nat.choose oreo_flavors 2 + oreo_flavors
    else if n = 3 then Nat.choose oreo_flavors 3 + oreo_flavors * (oreo_flavors - 1) + oreo_flavors
    else 0
  let total_combinations :=
    alpha_combinations 3
      + alpha_combinations 2 * beta_combinations 1
      + alpha_combinations 1 * beta_combinations 2
      + alpha_combinations 0 * beta_combinations 3
  total_combinations = 656 := by
  let oreo_flavors := 6
  let milk_flavors := 4
  let total_flavors := oreo_flavors + milk_flavors
  let alpha_combinations (n : Nat) := Nat.choose total_flavors n
  let beta_combinations (n : Nat) :=
    if n = 1 then oreo_flavors
    else if n = 2 then Nat.choose oreo_flavors 2 + oreo_flavors
    else if n = 3 then Nat.choose oreo_flavors 3 + oreo_flavors * (oreo_flavors - 1) + oreo_flavors
    else 0
  let total_combinations :=
    alpha_combinations 3
      + alpha_combinations 2 * beta_combinations 1
      + alpha_combinations 1 * beta_combinations 2
      + alpha_combinations 0 * beta_combinations 3
  total_combinations = 656 := sorry

end oreo_milk_combinations_l467_467896


namespace area_of_quadrilateral_is_10_l467_467256

theorem area_of_quadrilateral_is_10 (E : ℝ × ℝ) (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * x - 6 * y = 0) :
  ∃ (A B C D : ℝ × ℝ), 
    (longest_chord E circle_eq A C) ∧ (shortest_chord E circle_eq B D) ∧ area_of_quadrilateral A B C D = 10 :=
by {
  let E := (0,1),
  let circle_eq := λ x y, x^2 + y^2 - 2 * x - 6 * y = 0,
  let A := some_point_on_circle,
  let B := some_other_point_on_circle,
  let C := yet_another_point_on_circle,
  let D := still_another_point_on_circle,
  exact exists.intro (A, B, C, D) (and.intro (longest_chord_proof E A C circle_eq)
                                              (and.intro (shortest_chord_proof E B D circle_eq)
                                                         (area_calculation_proof A B C D)))
}

end area_of_quadrilateral_is_10_l467_467256


namespace tan_alpha_l467_467108

variable (α : Real)

axiom h₁ : α ∈ Set.Ioo (π/2) π
axiom h₂ : Real.sin α = Real.sqrt 3 / 3

theorem tan_alpha : Real.tan α = -Real.sqrt 2 / 2 := by
  sorry

end tan_alpha_l467_467108


namespace profit_percent_l467_467825

variable {P C : ℝ}

theorem profit_percent (h1: 2 / 3 * P = 0.82 * C) : ((P - C) / C) * 100 = 23 := by
  have h2 : C = (2 / 3 * P) / 0.82 := by sorry
  have h3 : (P - C) / C = (P - (2 / 3 * P) / 0.82) / ((2 / 3 * P) / 0.82) := by sorry
  have h4 : (P - (2 / 3 * P) / 0.82) / ((2 / 3 * P) / 0.82) = (0.82 * P - 2 / 3 * P) / (2 / 3 * P) := by sorry
  have h5 : (0.82 * P - 2 / 3 * P) / (2 / 3 * P) = 0.1533 := by sorry
  have h6 : 0.1533 * 100 = 23 := by sorry
  sorry

end profit_percent_l467_467825


namespace altitudes_order_l467_467387

variable {A a b c h_a h_b h_c : ℝ}

-- Conditions
axiom area_eq : A = (1/2) * a * h_a
axiom area_eq_b : A = (1/2) * b * h_b
axiom area_eq_c : A = (1/2) * c * h_c
axiom sides_order : a > b ∧ b > c

-- Conclusion
theorem altitudes_order : h_a < h_b ∧ h_b < h_c :=
by
  sorry

end altitudes_order_l467_467387


namespace equilateral_triangle_side_length_l467_467360

theorem equilateral_triangle_side_length (D E F Q : EuclideanGeometry.Point ℝ)
  (t : ℝ) 
  (h_equilateral : EuclideanGeometry.equilateral_triangle D E F)
  (h_DQ : EuclideanGeometry.dist D Q = 2)
  (h_EQ : EuclideanGeometry.dist E Q = 2 * Real.sqrt 2)
  (h_FQ : EuclideanGeometry.dist F Q = 3) :
  EuclideanGeometry.dist D E = Real.sqrt 19 :=
sorry

end equilateral_triangle_side_length_l467_467360


namespace false_propositions_count_l467_467623

def is_constant_seq {α : Type*} [AddCommGroup α] (a : ℕ → α) : Prop :=
∀ n m, a n = a m

def is_arithmetic_seq {α : Type*} [AddCommGroup α] (a : ℕ → α) : Prop :=
∃ d, ∀ n, a (n+1) = a n + d

def original_proposition {α : Type*} [AddCommGroup α] (a : ℕ → α) : Prop :=
is_constant_seq a → is_arithmetic_seq a

def inverse_proposition {α : Type*} [AddCommGroup α] (a : ℕ → α) : Prop :=
is_arithmetic_seq a → is_constant_seq a

def converse_proposition {α : Type*} [AddCommGroup α] (a : ℕ → α) : Prop :=
is_constant_seq a ← is_arithmetic_seq a

def contrapositive_proposition {α : Type*} [AddCommGroup α] (a : ℕ → α) : Prop :=
¬is_arithmetic_seq a → ¬is_constant_seq a

theorem false_propositions_count {α : Type*} [AddCommGroup α] (a : ℕ → α) (h : original_proposition a) :
(inverse_proposition a = false) ∧ (converse_proposition a = false) ∧ (contrapositive_proposition a = true) → 2 := by sorry

end false_propositions_count_l467_467623


namespace distinct_four_digit_count_l467_467156

theorem distinct_four_digit_count :
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  in count = 120 :=
by
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  show count = 120
  sorry

end distinct_four_digit_count_l467_467156


namespace value_of_x_l467_467401

variable (x y z : ℝ)

-- Conditions based on the problem statement
def condition1 := x = (1 / 3) * y
def condition2 := y = (1 / 4) * z
def condition3 := z = 96

-- The theorem to be proven
theorem value_of_x (cond1 : condition1) (cond2 : condition2) (cond3 : condition3) : x = 8 := 
by
  sorry

end value_of_x_l467_467401


namespace profit_percentage_on_cost_price_l467_467497

theorem profit_percentage_on_cost_price (
  (CP : ℝ) (SP : ℝ) (discount : ℝ) :
  CP = 51.50 →
  SP = 67.76 →
  discount = 0.05 →
  let LP := SP / (1 - discount) in
  let Profit := SP - CP in
  let Profit_Percentage := (Profit / CP) * 100 in
  Profit_Percentage ≈ 31.57
) : Prop := by
  assume h1 h2 h3,
  let LP := SP / (1 - discount),
  let Profit := SP - CP,
  let Profit_Percentage := (Profit / CP) * 100,
  show Profit_Percentage ≈ 31.57, from sorry

end profit_percentage_on_cost_price_l467_467497


namespace number_of_sets_M_l467_467221

-- Define the sets we are interested in.
def S1 : Set ℕ := {1, 2, 3}
def S2 : Set ℕ := {1, 2, 3, 4, 5, 6}

-- State the main theorem
theorem number_of_sets_M : ∃ (n : ℕ), n = 8 ∧ (∀ M : Set ℕ, S1 ⊆ M ∧ M ⊆ S2 → (∃ M_subsets : Finset (Set ℕ), M_subsets.card = n ∧ ∀ X ∈ M_subsets, S1 ⊆ X ∧ X ⊆ S2)) :=
by {
    use 8,
    sorry,  -- skipping the proof steps
}

end number_of_sets_M_l467_467221


namespace bus_speed_increase_l467_467394

theorem bus_speed_increase
  (S0 : ℕ) (D : ℕ) (hours : ℕ)
  (init_speed : S0 = 35)
  (total_distance : D = 552)
  (time_span : hours = 12) :
  ∃ x : ℕ, (S0 + (S0 + x) + (S0 + 2 * x) + ... + (S0 + 11 * x)) = D ∧ x = 2 :=
by
  sorry

end bus_speed_increase_l467_467394


namespace geometric_sequence_third_term_l467_467749

-- Define the problem statement in Lean 4
theorem geometric_sequence_third_term :
  ∃ r : ℝ, (a = 1024) ∧ (a_5 = 128) ∧ (a_5 = a * r^4) ∧ 
  (a_3 = a * r^2) ∧ (a_3 = 256) :=
sorry

end geometric_sequence_third_term_l467_467749


namespace stratified_sampling_example_l467_467777

theorem stratified_sampling_example :
  ∀ (total_students first_year second_year third_year sample_size : ℕ),
    total_students = 2700 →
    first_year = 900 →
    second_year = 1200 →
    third_year = 600 →
    sample_size = 135 →
    let first_year_sample := (first_year * sample_size) / total_students
    let second_year_sample := (second_year * sample_size) / total_students
    let third_year_sample := (third_year * sample_size) / total_students
    first_year_sample = 45 ∧ second_year_sample = 60 ∧ third_year_sample = 30 :=
by
  intros total_students first_year second_year third_year sample_size
  intros h_total_students h_first_year h_second_year h_third_year h_sample_size
  simp [h_total_students, h_first_year, h_second_year, h_third_year, h_sample_size]
  let first_year_sample := (first_year * sample_size) / total_students
  let second_year_sample := (second_year * sample_size) / total_students
  let third_year_sample := (third_year * sample_size) / total_students
  have h1 : first_year_sample = 45 := by
    rw [first_year, sample_size, total_students]
    norm_num
  have h2 : second_year_sample = 60 := by
    rw [second_year, sample_size, total_students]
    norm_num
  have h3 : third_year_sample = 30 := by
    rw [third_year, sample_size, total_students]
    norm_num
  exact ⟨h1, h2, h3⟩
  sorry

end stratified_sampling_example_l467_467777


namespace cells_at_end_of_12th_day_l467_467853

def initial_organisms : ℕ := 8
def initial_cells_per_organism : ℕ := 4
def total_initial_cells : ℕ := initial_organisms * initial_cells_per_organism
def division_period_days : ℕ := 3
def total_duration_days : ℕ := 12
def complete_periods : ℕ := total_duration_days / division_period_days
def common_ratio : ℕ := 3

theorem cells_at_end_of_12th_day :
  total_initial_cells * common_ratio^(complete_periods - 1) = 864 := by
  sorry

end cells_at_end_of_12th_day_l467_467853


namespace intersection_of_sets_l467_467844

open Set

theorem intersection_of_sets : 
  M = \{-1, 0, 1\} ∧ N = \{0, 1, 2\} → M ∩ N = \{0, 1\} :=
by
  sorry

end intersection_of_sets_l467_467844


namespace sum_is_8_l467_467388

def num1 : ℕ := 3030303030303030303030303030303030303030303030303030303030303030303030303030303030303030303
def num2 : ℕ := 505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505

def units_digit (n : ℕ) : ℕ := n % 10
def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10

noncomputable def product : ℕ := num1 * num2

def A : ℕ := thousands_digit product
def B : ℕ := units_digit product

theorem sum_is_8 : A + B = 8 := by
  sorry

end sum_is_8_l467_467388


namespace find_derivative_at_minus_third_l467_467577

noncomputable def f (f'_-1_3 : ℝ) (x : ℝ) : ℝ := x^2 + 2 * f'_-1_3 * x

theorem find_derivative_at_minus_third (f'_-1_3 : ℝ) (h : ∀ x, deriv (f f'_-1_3) x = 2 * x + 2 * f'_-1_3) : f'_-1_3 = 2 / 3 :=
by
  let x := -1/3
  have : deriv (f f'_-1_3) x = 0,
    from sorry,
  rw h x at this,
  sorry

end find_derivative_at_minus_third_l467_467577


namespace f_of_minus_one_l467_467114

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * sin x + b * x^3 + 5

theorem f_of_minus_one (a b : ℝ) (h : f a b 1 = 3) : f a b (-1) = 7 := 
sorry

end f_of_minus_one_l467_467114


namespace find_B_l467_467240

variable {A B C a b c : Real}

noncomputable def B_value (A B C a b c : Real) : Prop :=
  B = 2 * Real.pi / 3

theorem find_B 
  (h_triangle: a^2 + b^2 + c^2 = 2*a*b*Real.cos C)
  (h_cos_eq: (2 * a + c) * Real.cos B + b * Real.cos C = 0) : 
  B_value A B C a b c :=
by
  sorry

end find_B_l467_467240


namespace max_siskins_on_poles_l467_467892

-- Define the conditions
def total_poles : ℕ := 25

def adjacent (i j : ℕ) : Prop := (abs (i - j) = 1)

-- Define the main problem
theorem max_siskins_on_poles (h₁ : 0 < total_poles) 
  (h₂ : ∀ (i : ℕ), i ≥ 1 ∧ i ≤ total_poles → ∀ (j : ℕ), j ≥ 1 ∧ j ≤ total_poles ∧ adjacent i j 
    → ¬ (siskin_on i ∧ siskin_on j)) :
  ∃ (max_siskins : ℕ), max_siskins = 24 := 
begin
  sorry
end

end max_siskins_on_poles_l467_467892


namespace part1_min_value_part2_min_value_l467_467293

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem part1_min_value :
  ∃ (m : ℝ), m = 2 ∧ (∀ (x : ℝ), f x ≥ m) :=
sorry

theorem part2_min_value (a b : ℝ) (h : a^2 + b^2 = 2) :
  ∃ (y : ℝ), y = (1 / (a^2 + 1) + 4 / (b^2 + 1)) ∧ y = 9 / 4 :=
sorry

end part1_min_value_part2_min_value_l467_467293


namespace andrew_total_homeless_shelter_donation_l467_467500

-- Given constants and conditions
def bake_sale_total : ℕ := 400
def ingredients_cost : ℕ := 100
def piggy_bank_donation : ℕ := 10

-- Intermediate calculated values
def remaining_total : ℕ := bake_sale_total - ingredients_cost
def shelter_donation_from_bake_sale : ℕ := remaining_total / 2

-- Final goal statement
theorem andrew_total_homeless_shelter_donation :
  shelter_donation_from_bake_sale + piggy_bank_donation = 160 :=
by
  -- Proof to be provided.
  sorry

end andrew_total_homeless_shelter_donation_l467_467500


namespace present_worth_approx_l467_467472

noncomputable def amount_after_years (P : ℝ) : ℝ :=
  let A1 := P * (1 + 5 / 100)                      -- Amount after the first year.
  let A2 := A1 * (1 + 5 / 100)^2                   -- Amount after the second year.
  let A3 := A2 * (1 + 3 / 100)^4                   -- Amount after the third year.
  A3

noncomputable def banker's_gain (P : ℝ) : ℝ :=
  amount_after_years P - P

theorem present_worth_approx :
  ∃ P : ℝ, abs (P - 114.94) < 1 ∧ banker's_gain P = 36 :=
sorry

end present_worth_approx_l467_467472


namespace angle_A_is_pi_over_three_geometric_sum_reciprocal_l467_467684

section problem_one

variables {A B a b c : ℝ}

-- Conditions
def collinear (A B a b c: ℝ) : Prop := (cos A, cos B) = (λ t, (a, 2 * c - b)) t

theorem angle_A_is_pi_over_three (h: collinear A B a b c) : A = π / 3 := 
sorry

end problem_one

section problem_two

variables {a1 a4 : ℝ} (A: ℝ) (n : ℕ)

-- Conditions
def geometric_prog (a1 A a4 : ℝ) : Prop := 
a1 * cos A = 1 ∧ (a4 = 16) ∧ ∀ n : ℕ, let q := 2 in a1*(q ^ (n - 1)) = 2^((3 : ℕ) + 1)

def bn (a1 : ℝ) (n : ℕ) : ℝ :=
(log 2 (2^n)) * (log 2 (2^(n + 1)))

-- Sum of reciprocals of b_n
def Sn (n : ℕ) : ℝ := ∑ k in range (n+1), 1/(k*(k+1))

theorem geometric_sum_reciprocal (h: geometric_prog a1 A a4) : Sn n = n / (n + 1) :=
sorry

end problem_two

end angle_A_is_pi_over_three_geometric_sum_reciprocal_l467_467684


namespace distinct_positive_factors_of_1320_l467_467199

theorem distinct_positive_factors_of_1320 : 
  ∀ n ≥ 0, prime_factorization 1320 = [(2, 3), (3, 1), (11, 1), (5, 1)] → factors_count 1320 = 32 :=
by
  intro n hn h
  sorry

end distinct_positive_factors_of_1320_l467_467199


namespace domain_intersects_interval_l467_467235

def f (x : ℝ) : ℝ := real.log (real.sin (π * x) * real.sin (2 * π * x) * real.sin (3 * π * x) * real.sin (4 * π * x))

theorem domain_intersects_interval (n : ℕ) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → real.sin (π * x) * real.sin (2 * π * x) * real.sin (3 * π * x) * real.sin (4 * π * x) > 0) →
  n = 4 :=
sorry

end domain_intersects_interval_l467_467235


namespace geom_seq_common_ratio_l467_467077

theorem geom_seq_common_ratio (a₁ a₂ a₃ a₄ q : ℝ) 
  (h1 : a₁ + a₄ = 18)
  (h2 : a₂ * a₃ = 32)
  (h3 : a₂ = a₁ * q)
  (h4 : a₃ = a₁ * q^2)
  (h5 : a₄ = a₁ * q^3) : 
  q = 2 ∨ q = (1 / 2) :=
by {
  sorry
}

end geom_seq_common_ratio_l467_467077


namespace fraction_A_is_correct_l467_467828

-- Define the variables a and b
variables (a b : ℝ)

-- Define the fraction that we're proving is equal to the original fraction
-- Original fraction
def original_fraction : ℝ := (a - b) / (a + b)

-- Fraction from option A
def fraction_A : ℝ := (-a + b) / (-a - b)

-- The theorem statement to prove that the two fractions are equal
theorem fraction_A_is_correct : fraction_A a b = original_fraction a b :=
sorry

end fraction_A_is_correct_l467_467828


namespace triangle_equilateral_l467_467686

-- Defining the points and triangle properties based on given conditions
variables {A B C A1 B1 C1 : Type} [has_coe_to_sort A1] [has_coe_to_sort B1] [has_coe_to_sort C1]

-- Defining the triangle ABC as type with vertices A, B, C
structure triangle (A B C A1 B1 C1 : Type) :=
  (on_side_BC : A1 ∈ ℝ)
  (on_side_AC : B1 ∈ ℝ)
  (on_side_AB : C1 ∈ ℝ)
  (altitude : ∀ (A B C A1 : Type), A = (B + C + A1) / 2)
  (median : ∀ (B C B1 : Type), B1 = (C + B) / 2)
  (angle_bisector : ∀ (C C1 : Type), C1 = C / 2)
  (equilateral_triangle : ∀ (A1 B1 C1 : Type), A1 = B1 ∧ B1 = C1 ∧ C1 = A1)

-- The theorem to prove that triangle ABC is equilateral
theorem triangle_equilateral (A B C A1 B1 C1 : Type)
  [triangle A B C A1 B1 C1] :
  ∀ (A B C : Type), A = B ∧ B = C ∧ C = A :=
by sorry

end triangle_equilateral_l467_467686


namespace minimize_total_cost_l467_467601

open Real

def W (x : ℝ) : ℝ := 2 * x + 7200 / x

theorem minimize_total_cost : 
  ∀ x : ℝ, 
  50 ≤ x ∧ x ≤ 100 →
  (∀ y : ℝ, 50 ≤ y ∧ y ≤ 100 → W(x) ≤ W(y)) ↔ x = 60 :=
by sorry

end minimize_total_cost_l467_467601


namespace sin_double_angle_l467_467571

theorem sin_double_angle (α : ℝ) (h : real.tan α = -1 / 3) : real.sin (2 * α) = -3 / 5 :=
sorry

end sin_double_angle_l467_467571


namespace length_of_AC_l467_467446

variable (BC CD DE AB AE AC : ℝ)
variable (x : ℝ)

axiom BC_eq_3CD : BC = 3 * CD
axiom DE_eq_8 : DE = 8
axiom AB_eq_5 : AB = 5
axiom AE_eq_21 : AE = 21
axiom AE_def : AE = AB + BC + CD + DE

theorem length_of_AC : AC = 11 :=
by
  have h1 : AE = AB + BC + CD + DE := AE_def
  have h2 : AE = 21 := AE_eq_21
  have h3 : AB = 5 := AB_eq_5
  have h4 : DE = 8 := DE_eq_8
  have h5 : BC = 3 * CD := BC_eq_3CD
  -- Adding all conditions and calculation steps should follow here
  sorry

end length_of_AC_l467_467446


namespace sum_lcms_equals_l467_467807

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_lcms_equals :
  ( ∑ ν in (Finset.filter is_solution (Finset.range 100)), ν ) = 180 :=
sorry

end sum_lcms_equals_l467_467807


namespace smallest_m_plus_n_l467_467747

theorem smallest_m_plus_n : ∃ (m n : ℕ), m > 1 ∧ 
  (∃ (a b : ℝ), a = (1 : ℝ) / (m * n : ℝ) ∧ b = (m : ℝ) / (n : ℝ) ∧ b - a = (1 : ℝ) / 1007) ∧
  (∀ (k l : ℕ), k > 1 ∧ 
    (∃ (c d : ℝ), c = (1 : ℝ) / (k * l : ℝ) ∧ d = (k : ℝ) / (l : ℝ) ∧ d - c = (1 : ℝ) / 1007) → m + n ≤ k + l) ∧ 
  m + n = 19099 :=
sorry

end smallest_m_plus_n_l467_467747


namespace washing_machines_removed_l467_467251

theorem washing_machines_removed (crates boxes_per_crate washing_machines_per_box washing_machines_removed_per_box : ℕ) 
  (h_crates : crates = 10) (h_boxes_per_crate : boxes_per_crate = 6) 
  (h_washing_machines_per_box : washing_machines_per_box = 4) 
  (h_washing_machines_removed_per_box : washing_machines_removed_per_box = 1) :
  crates * boxes_per_crate * washing_machines_removed_per_box = 60 :=
by
  rw [h_crates, h_boxes_per_crate, h_washing_machines_removed_per_box]
  exact Nat.mul_assoc crates boxes_per_crate washing_machines_removed_per_box ▸
         Nat.mul_assoc 10 6 1 ▸ rfl


end washing_machines_removed_l467_467251


namespace sum_of_nus_is_45_l467_467802

noncomputable def sum_of_valid_nu : ℕ :=
  ∑ ν in {ν | ν > 0 ∧ Nat.lcm ν 24 = 72}.toFinset, ν

theorem sum_of_nus_is_45 : sum_of_valid_nu = 45 := by
  sorry

end sum_of_nus_is_45_l467_467802


namespace exists_irreducible_fractions_sum_to_86_over_111_l467_467558

theorem exists_irreducible_fractions_sum_to_86_over_111 :
  ∃ (a b p q : ℕ), (0 < a) ∧ (0 < b) ∧ (p ≤ 100) ∧ (q ≤ 100) ∧ (Nat.coprime a p) ∧ (Nat.coprime b q) ∧ (a * q + b * p = 86 * (p * q) / 111) :=
by
  sorry

end exists_irreducible_fractions_sum_to_86_over_111_l467_467558


namespace count_1320_factors_l467_467169

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end count_1320_factors_l467_467169


namespace point_in_fourth_quadrant_l467_467673

/-- A point in a Cartesian coordinate system -/
structure Point (α : Type) :=
(x : α)
(y : α)

/-- Given a point (4, -3) in the Cartesian plane, prove it lies in the fourth quadrant -/
theorem point_in_fourth_quadrant (P : Point Int) (hx : P.x = 4) (hy : P.y = -3) : 
  P.x > 0 ∧ P.y < 0 :=
by
  sorry

end point_in_fourth_quadrant_l467_467673


namespace class_mean_calculation_l467_467242

noncomputable def group1_students := 40
noncomputable def group1_avg_score := 68
noncomputable def group2_students := 15
noncomputable def group2_avg_score := 74
noncomputable def group3_students := 5
noncomputable def group3_avg_score := 88
noncomputable def total_students := 60

theorem class_mean_calculation :
  ((group1_students * group1_avg_score) + (group2_students * group2_avg_score) + (group3_students * group3_avg_score)) / total_students = 71.17 := 
by 
  sorry

end class_mean_calculation_l467_467242


namespace expression_value_l467_467456

theorem expression_value (x y z : ℕ) (hx : x = 1) (hy : y = 1) (hz : z = 3) : x^2 * y * z - x * y * z^2 = -6 :=
by
  -- Using the given conditions to solve for the expression's value.
  -- x = 1, y = 1, z = 3 are used to prove the given theorem.
  sorry

end expression_value_l467_467456


namespace min_value_abs_sum_l467_467383

theorem min_value_abs_sum (x: ℝ) : ∀ x, |x - 1| + |x - 3| ≥ 2 ∧ (∃ x, |x - 1| + |x - 3| = 2) := 
by 
  sorry

end min_value_abs_sum_l467_467383


namespace probability_of_receiving_one_l467_467666

noncomputable def probability_received_one : ℝ :=
let P_A := 0.5 in
let P_not_A := 0.5 in
let P_B_given_A := 0.9 in
let P_not_B_given_A := 0.1 in
let P_B_given_not_A := 0.05 in
let P_not_B_given_not_A := 0.95 in
let P_B := P_A * P_B_given_A + P_not_A * P_B_given_not_A in
1 - P_B

theorem probability_of_receiving_one :
  probability_received_one = 0.525 :=
by
  -- P_A = 0.5
  -- P_not_A = 0.5
  -- P_B_given_A = 0.9
  -- P_not_B_given_A = 0.1
  -- P_B_given_not_A = 0.05
  -- P_not_B_given_not_A = 0.95
  -- P_B = 0.5 * 0.9 + 0.5 * 0.05
  -- P_B = 0.45 + 0.025
  -- P_B = 0.475
  -- P_not_B = 1 - P_B
  -- P_not_B = 1 - 0.475
  -- P_not_B = 0.525
  sorry

end probability_of_receiving_one_l467_467666


namespace max_neg_p_l467_467842

theorem max_neg_p (p : ℤ) (h1 : p < 0) (h2 : ∃ k : ℤ, 2001 + p = k^2) : p ≤ -65 :=
by
  sorry

end max_neg_p_l467_467842


namespace shift_sin_function_l467_467421

theorem shift_sin_function {
  right_shift : ℝ := π / 6
} : 
  ∀ x : ℝ, sin(2 * (x - right_shift) + π / 3) = sin(2 * x) :=
begin
  sorry
end

end shift_sin_function_l467_467421


namespace value_of_x_l467_467404

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 :=
by
  sorry

end value_of_x_l467_467404


namespace vector_addition_scalar_multiplication_l467_467948

def u : ℝ × ℝ × ℝ := (3, -2, 5)
def v : ℝ × ℝ × ℝ := (-1, 6, -3)
def result : ℝ × ℝ × ℝ := (4, 8, 4)

theorem vector_addition_scalar_multiplication :
  2 • (u + v) = result :=
by
  sorry

end vector_addition_scalar_multiplication_l467_467948


namespace min_fencing_dims_l467_467309

theorem min_fencing_dims (x : ℕ) (h₁ : x * (x + 5) ≥ 600) (h₂ : x = 23) : 
  2 * (x + (x + 5)) = 102 := 
by
  -- Placeholder for the proof
  sorry

end min_fencing_dims_l467_467309


namespace distinct_factors_1320_l467_467179

theorem distinct_factors_1320 :
  let p := 1320,
  prime_factors := [(2, 2), (3, 1), (5, 1), (11, 1)],
  num_factors := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in num_factors = 24 := by
  sorry

end distinct_factors_1320_l467_467179


namespace cosine_identity_l467_467094

theorem cosine_identity (α : ℝ) (h1 : Real.cos (Float.pi * 75 / 180 + α) = 1 / 3) :
  Real.cos (Float.pi * 30 / 180 - 2 * α) = 7 / 9 :=
sorry

end cosine_identity_l467_467094


namespace prove_angle_cos_minus_sin_l467_467106

noncomputable def angle_cos_minus_sin : Prop :=
  let α_initial := (x : ℝ) (y : ℝ), y = 0 ∧ x ≥ 0  -- Initial side on the x-axis non-negative semiaxis
  let terminal_side := (x : ℝ) (y : ℝ), 4 * x - 3 * y = 0 ∧ x ≤ 0  -- Terminal side on the ray
  let α := (-3, -4)  -- Specific point on the ray
  let r := real.sqrt (α.1 ^ 2 + α.2 ^ 2)  -- Radius computation
  let cosα := α.1 / r
  let sinα := α.2 / r
  cosα - sinα = 1 / 5

theorem prove_angle_cos_minus_sin : angle_cos_minus_sin := by
  sorry

end prove_angle_cos_minus_sin_l467_467106


namespace value_of_x_l467_467399

variable (x y z : ℝ)

-- Conditions based on the problem statement
def condition1 := x = (1 / 3) * y
def condition2 := y = (1 / 4) * z
def condition3 := z = 96

-- The theorem to be proven
theorem value_of_x (cond1 : condition1) (cond2 : condition2) (cond3 : condition3) : x = 8 := 
by
  sorry

end value_of_x_l467_467399


namespace distinct_factors_1320_l467_467208

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end distinct_factors_1320_l467_467208


namespace benny_money_l467_467508

-- Conditions
def cost_per_apple (cost : ℕ) := cost = 4
def apples_needed (apples : ℕ) := apples = 5 * 18

-- The proof problem
theorem benny_money (cost : ℕ) (apples : ℕ) (total_money : ℕ) :
  cost_per_apple cost → apples_needed apples → total_money = apples * cost → total_money = 360 :=
by
  intros h_cost h_apples h_total
  rw [h_cost, h_apples] at h_total
  exact h_total

end benny_money_l467_467508


namespace average_price_of_goat_l467_467845

theorem average_price_of_goat (total_cost_goats_hens : ℕ) (num_goats num_hens : ℕ) (avg_price_hen : ℕ)
  (h1 : total_cost_goats_hens = 2500) (h2 : num_hens = 10) (h3 : avg_price_hen = 50) (h4 : num_goats = 5) :
  (total_cost_goats_hens - num_hens * avg_price_hen) / num_goats = 400 :=
sorry

end average_price_of_goat_l467_467845


namespace series_sum_l467_467924

theorem series_sum : 
  (∑ n : ℕ, (2^n) / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end series_sum_l467_467924


namespace key_lock_determination_l467_467780

theorem key_lock_determination :
  ∃ (f : Fin 6 → Finset (Fin 4)), 
    (∀ k, (f k).card = 2) ∧
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ (k : Fin 6), ∃ i j, i ≠ j ∧ f k = {i, j}) ∧
    (decreasing_tests_needed_to_determine_pairs 4 6 13) :=
sorry

end key_lock_determination_l467_467780


namespace MinkowskiSum_convex_l467_467329

noncomputable def MinkowskiSum (Φ1 Φ2 : set (ℝ × ℝ)) : set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ × ℝ), x ∈ Φ1 ∧ y ∈ Φ2 ∧ p = (x.1 + y.1, x.2 + y.2)}

def is_convex (S : set (ℝ × ℝ)) : Prop :=
  ∀ ⦃p₁ p₂ : ℝ × ℝ⦄, p₁ ∈ S → p₂ ∈ S → ∀ (t : ℝ), 0 ≤ t → t ≤ 1 →
  t • p₁ + (1 - t) • p₂ ∈ S

variables (Φ1 Φ2 : set (ℝ × ℝ))
hypothesis (h1 : is_convex Φ1)
hypothesis (h2 : is_convex Φ2)

theorem MinkowskiSum_convex : is_convex (MinkowskiSum Φ1 Φ2) :=
sorry

end MinkowskiSum_convex_l467_467329


namespace sum_of_nus_is_45_l467_467798

noncomputable def sum_of_valid_nu : ℕ :=
  ∑ ν in {ν | ν > 0 ∧ Nat.lcm ν 24 = 72}.toFinset, ν

theorem sum_of_nus_is_45 : sum_of_valid_nu = 45 := by
  sorry

end sum_of_nus_is_45_l467_467798


namespace odd_function_f_find_f_neg1_l467_467706

variable (a : ℝ)

def f : ℝ → ℝ := λ x, if x ≥ 0 then 2^x + x + a else -(2^(-x) + (-x) + a)

theorem odd_function_f (x : ℝ) : f (-x) = -f x :=
  by sorry

theorem find_f_neg1 (a : ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_f : ∀ x ≥ 0, f x = 2^x + x + a) :
  f (-1) = -2 :=
  by sorry

end odd_function_f_find_f_neg1_l467_467706


namespace part_one_part_two_l467_467113

def f (m x : ℝ) : ℝ := (x - 1) / (Real.log x - m * x^2)

theorem part_one (m : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → f m x > 1) → m = 0 :=
sorry

noncomputable def a : ℕ+ → ℝ
| ⟨1, h⟩ := Real.sqrt Real.exp 1
| ⟨n + 2, h⟩ := f 0 (a ⟨n + 1, nat.succ_pos n⟩)

theorem part_two (n : ℕ+) :
  2 ^ n * Real.log (a n) ≥ 1 :=
sorry

end part_one_part_two_l467_467113


namespace exists_A_eq_b_pow_n_l467_467299

theorem exists_A_eq_b_pow_n (n b : ℕ) (hnb : n > 1 ∧ b > 1)
    (hk : ∀ k : ℕ, k > 1 → ∃ a_k : ℤ, k ∣ (b : ℤ) - a_k ^ n) :
    ∃ A : ℤ, b = A ^ n :=
sorry

end exists_A_eq_b_pow_n_l467_467299


namespace polygon_area_l467_467265

theorem polygon_area (n : ℕ) (s : ℝ) (perimeter : ℝ) (area : ℝ) :
  n = 28 →
  perimeter = 56 →
  s = perimeter / n →
  (∀ (i : ℕ), i < n → scongruent (side_length i) s) →
  (∀ (i : ℕ), i < n → perpendicular (side i) (adjacent_side i)) →
  area = 25 * (s ^ 2) →
  area = 100 :=
by
  intros h_n h_perimeter h_side_length h_congruence h_perpendicular h_area_calc
  rw [h_n, h_perimeter, h_side_length] at *
  sorry

end polygon_area_l467_467265


namespace max_siskins_on_poles_l467_467891

-- Define the conditions
def total_poles : ℕ := 25

def adjacent (i j : ℕ) : Prop := (abs (i - j) = 1)

-- Define the main problem
theorem max_siskins_on_poles (h₁ : 0 < total_poles) 
  (h₂ : ∀ (i : ℕ), i ≥ 1 ∧ i ≤ total_poles → ∀ (j : ℕ), j ≥ 1 ∧ j ≤ total_poles ∧ adjacent i j 
    → ¬ (siskin_on i ∧ siskin_on j)) :
  ∃ (max_siskins : ℕ), max_siskins = 24 := 
begin
  sorry
end

end max_siskins_on_poles_l467_467891


namespace lattice_points_sum_l467_467721

def lattice_points (n : ℕ) : ℕ := 6 * n

noncomputable def sum_arithmetic_seq (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem lattice_points_sum :
  (1 / 2012 : ℝ) * (Finset.sum (Finset.range 1006) (λ k, lattice_points (2 * (k + 1)))) = 3021 :=
by
  let a := lattice_points 2
  let d := lattice_points 4 - lattice_points 2
  let n := 1006
  have seq_sum := sum_arithmetic_seq a d n
  sorry

end lattice_points_sum_l467_467721


namespace general_term_of_sequence_l467_467627

def S (n : ℕ) : ℕ := n^2 + 3 * n + 1

def a (n : ℕ) : ℕ := 
  if n = 1 then 5 
  else 2 * n + 2

theorem general_term_of_sequence (n : ℕ) : 
  a n = if n = 1 then 5 else (S n - S (n - 1)) := 
by 
  sorry

end general_term_of_sequence_l467_467627


namespace distinct_positive_factors_of_1320_l467_467195

theorem distinct_positive_factors_of_1320 : 
  ∀ n ≥ 0, prime_factorization 1320 = [(2, 3), (3, 1), (11, 1), (5, 1)] → factors_count 1320 = 32 :=
by
  intro n hn h
  sorry

end distinct_positive_factors_of_1320_l467_467195


namespace penalty_kicks_l467_467367

theorem penalty_kicks (total_players goalies : ℕ) (h_total : total_players = 25) (h_goalies : goalies = 4) : 
  let non_goalie_players := total_players - goalies in
  let penalty_kicks := non_goalie_players * goalies in
  penalty_kicks = 96 :=
by
  sorry

end penalty_kicks_l467_467367


namespace distinct_positive_factors_of_1320_l467_467194

theorem distinct_positive_factors_of_1320 : 
  ∀ n ≥ 0, prime_factorization 1320 = [(2, 3), (3, 1), (11, 1), (5, 1)] → factors_count 1320 = 32 :=
by
  intro n hn h
  sorry

end distinct_positive_factors_of_1320_l467_467194


namespace anya_game_losses_l467_467059

theorem anya_game_losses (games : ℕ → ℕ) (anya_games bella_games valya_games galya_games dasha_games : ℕ)
  (H_total : anya_games = 4 ∧ bella_games = 6 ∧ valya_games = 7 ∧ galya_games = 10 ∧ dasha_games = 11) :
  (games anya_games = 4 ∧ games bella_games = 6 ∧
   games valya_games = 7 ∧ games galya_games = 10 ∧
   games dasha_games = 11) →  [4, 8, 12, 16] := sorry

end anya_game_losses_l467_467059


namespace quadratic_function_range_l467_467589

noncomputable def quadratic_range (a b c : ℝ) : Prop :=
  (∃ (a b c : ℝ), a ≠ 0 ∧
    (∀ x : ℝ, a * x^2 + b * x + c) = (x = 1) ∧ (a * 1 + b * 1 + c = 0) ∧
    (∀ x : ℝ, a * x^2 + b * x + c) = (x = 0) ∧ (c = 1) ∧
    (a < 0) ∧ (2a + 2 > 0) ∧ (2a + 2 < 2) 
  ).toProp

theorem quadratic_function_range : ∃ (a b c : ℝ), quadratic_range a b c ∧ 0 < a - b + c ∧ a - b + c < 2 := sorry

end quadratic_function_range_l467_467589


namespace smallest_nonneg_integer_l467_467526

theorem smallest_nonneg_integer (n : ℕ) (h : 0 ≤ n ∧ n < 53) :
  50 * n ≡ 47 [MOD 53] → n = 2 :=
by
  sorry

end smallest_nonneg_integer_l467_467526


namespace tangent_at_point_l467_467617

theorem tangent_at_point (a b : ℝ) :
  (∀ x : ℝ, (x^3 - x^2 - a * x + b) = 2 * x + 1) →
  (a + b = -1) :=
by
  intro tangent_condition
  sorry

end tangent_at_point_l467_467617


namespace count_sequence_formula_l467_467385

-- We define the sequence transformation based on the given conditions.

noncomputable def initial_state : Set Int := {1}

def next_state (s : Set Int) : Set Int :=
  s.bind (λ a, {a - 1, a + 1}) \ {0}

def sequence (n : Nat) : Set Int :=
  Nat.rec initial_state (λ _ s, next_state s) n

-- Define the function that counts the number of elements in the sequence after n steps
def count_sequence (n : Nat) : Nat :=
  (sequence n).card

-- Statement of the problem in Lean: Prove the count of the sequence elements after n steps
theorem count_sequence_formula (n : ℕ) : count_sequence n = Nat.factorial (2 * n + 2) / (Nat.factorial (n + 1) * Nat.factorial (n + 1)) :=
  sorry

end count_sequence_formula_l467_467385


namespace ratio_M_AD_l467_467254

variable (A B C D M K N O : Type)
variable [square ABCD]
variable [is_midpoint K A B]
variable [is_point_on_side M A D]
variable [is_point_on_side N B C]
variable [is_intersection O (diagonal AC) (diagonal BD)]
variable [divides_into_equal_areas OK OM ON]

theorem ratio_M_AD : divides M AD = 5 : 1 :=
sorry

end ratio_M_AD_l467_467254


namespace problem_statement_l467_467609

noncomputable def f : ℝ → ℝ :=
λ x, if x > 1 then Real.log x / Real.log 3
     else if -1 < x ∧ x <= 1 then x^2
     else 3^x

theorem problem_statement :
  f (-f (Real.sqrt 3)) + f (f 0) + f (1 / f (-1)) = 5 / 4 := 
sorry

end problem_statement_l467_467609


namespace distinct_positive_factors_of_1320_l467_467200

theorem distinct_positive_factors_of_1320 : 
  ∀ n ≥ 0, prime_factorization 1320 = [(2, 3), (3, 1), (11, 1), (5, 1)] → factors_count 1320 = 32 :=
by
  intro n hn h
  sorry

end distinct_positive_factors_of_1320_l467_467200


namespace top_card_probability_spades_or_clubs_l467_467487

-- Definitions
def total_cards : ℕ := 52
def suits : ℕ := 4
def ranks : ℕ := 13
def spades_cards : ℕ := ranks
def clubs_cards : ℕ := ranks
def favorable_outcomes : ℕ := spades_cards + clubs_cards

-- Probability calculation statement
theorem top_card_probability_spades_or_clubs :
  (favorable_outcomes : ℚ) / (total_cards : ℚ) = 1 / 2 :=
  sorry

end top_card_probability_spades_or_clubs_l467_467487


namespace initial_books_count_l467_467864

-- Definitions of the given conditions
def shelves : ℕ := 9
def books_per_shelf : ℕ := 9
def books_remaining : ℕ := shelves * books_per_shelf
def books_sold : ℕ := 39

-- Statement of the proof problem
theorem initial_books_count : books_remaining + books_sold = 120 := 
by {
  sorry
}

end initial_books_count_l467_467864


namespace chalk_pieces_original_l467_467532

theorem chalk_pieces_original (
  siblings : ℕ := 3,
  friends : ℕ := 3,
  pieces_lost : ℕ := 2,
  pieces_added : ℕ := 12,
  pieces_needed_per_person : ℕ := 3
) : 
  ∃ (original_pieces : ℕ), 
    (original_pieces - pieces_lost + pieces_added) = 
    ((1 + siblings + friends) * pieces_needed_per_person) := 
  sorry

end chalk_pieces_original_l467_467532


namespace chantel_bracelets_giveaway_l467_467971

theorem chantel_bracelets_giveaway :
  ∀ (bracelets_per_day_1 : ℕ) (days_1 : ℕ) (giveaway_school : ℕ)
    (bracelets_per_day_2 : ℕ) (days_2 : ℕ) (final_count : ℕ),
    bracelets_per_day_1 = 2 → days_1 = 5 → giveaway_school = 3 →
    bracelets_per_day_2 = 3 → days_2 = 4 → final_count = 13 →
    let initial_count := days_1 * bracelets_per_day_1 in
    let after_school_giveaway := initial_count - giveaway_school in
    let additional_count := days_2 * bracelets_per_day_2 in
    let total_count_before_soccer := after_school_giveaway + additional_count in
    let soccer_giveaway := total_count_before_soccer - final_count in
    soccer_giveaway = 6 :=
by
  intros _ _ _ _ _ _
  intros hbp1 hdays1 hgs hbp2 hdays2 hfinal
  simp [hbp1, hdays1, hgs, hbp2, hdays2, hfinal]
  let initial_count := 5 * 2
  let after_school_giveaway := initial_count - 3
  let additional_count := 4 * 3
  let total_count_before_soccer := after_school_giveaway + additional_count
  let soccer_giveaway := total_count_before_soccer - 13
  have : soccer_giveaway = 6 := by
    sorry
  exact this

end chantel_bracelets_giveaway_l467_467971


namespace distance_from_center_to_line_l467_467123

noncomputable def line_parametric_eq : ℝ → ℝ × ℝ :=
  λ t, (t - 3, real.sqrt 3 * t)

noncomputable def circle_polar_eq_to_rect (θ ρ : ℝ) : Prop :=
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ ρ^2 - 4 * ρ * real.cos θ + 3 = 0

def line_standard_eq (x y : ℝ) : Prop :=
  real.sqrt 3 * x - y + 3 * real.sqrt 3 = 0

def circle_center : ℝ × ℝ := (2, 0)

def distance_point_to_line (px py a b c: ℝ) : ℝ :=
  abs (a * px + b * py + c) / real.sqrt (a^2 + b^2)

theorem distance_from_center_to_line : 
  distance_point_to_line (circle_center.1) (circle_center.2) (real.sqrt 3) (-1) (3 * real.sqrt 3) = 5 * real.sqrt 3 / 2 :=
by
  sorry

end distance_from_center_to_line_l467_467123


namespace sum_of_digits_of_second_smallest_multiple_l467_467284

theorem sum_of_digits_of_second_smallest_multiple :
  let M := (2 * Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))))
  Nat.digits 10 M = [8, 4, 0] → M = 840 → M.divisors = [1,2,3,4,5,6,7] → (Nat.digits 10 M).sum = 12 :=
by
  sorry

end sum_of_digits_of_second_smallest_multiple_l467_467284


namespace area_triangle_range_l467_467677

theorem area_triangle_range (x y : ℝ) (A B : ℝ × ℝ) (O : ℝ × ℝ)
  (h_ellipse : ∃ x y, x^2 + 4*y^2 = 8)
  (h_chord_length: dist A B = 5 / 2)
  (O_eq : O = (0, 0)) :
  ∃ S, S ∈ set.Icc (5 * real.sqrt 103 / 32) 2 :=
by
  sorry

end area_triangle_range_l467_467677


namespace triangle_probability_is_correct_l467_467321

-- Define the total number of figures
def total_figures : ℕ := 8

-- Define the number of triangles among the figures
def number_of_triangles : ℕ := 3

-- Define the probability function for choosing a triangle
def probability_of_triangle : ℚ := number_of_triangles / total_figures

-- The theorem to be proved
theorem triangle_probability_is_correct :
  probability_of_triangle = 3 / 8 := by
  sorry

end triangle_probability_is_correct_l467_467321


namespace bowling_average_proof_l467_467855

variable (initial_avg : ℚ) (wickets_before_last : ℕ) (wickets_last_match : ℕ) (avg_decrease : ℚ) (runs_before_last : ℚ) (new_avg : ℚ)

def bowling_average_problem (initial_avg = 12.4) (wickets_before_last = 175) (wickets_last_match = 8) (avg_decrease = 0.4) : Prop :=
  let runs_before_last := initial_avg * wickets_before_last
  let new_avg := initial_avg - avg_decrease
  let total_wickets := wickets_before_last + wickets_last_match
  let runs_after_last := new_avg * total_wickets
  let runs_last_match := runs_after_last - runs_before_last
  runs_last_match = 26

theorem bowling_average_proof : bowling_average_problem :=
by
  sorry

end bowling_average_proof_l467_467855


namespace no_vision_assistance_l467_467413

def students := 40
def percent_glasses := 0.25
def percent_contacts := 0.40

theorem no_vision_assistance :
  students * (1 - percent_glasses - percent_contacts) = 14 :=
by
  sorry

end no_vision_assistance_l467_467413


namespace special_prime_looking_count_is_478_l467_467520

def is_prime : ℕ → Prop := sorry -- Assume definition for prime
def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ m * k = n

def special_prime_looking (n : ℕ) : Prop :=
  is_composite n ∧ ¬(2 ∣ n) ∧ ¬(7 ∣ n) ∧ ¬(11 ∣ n)

noncomputable def count_special_prime_looking (n : ℕ) : ℕ :=
  (Finset.filter (λ x, special_prime_looking x) (Finset.range n).filter (λ x, x ≥ 1)).card

theorem special_prime_looking_count_is_478 :
  count_special_prime_looking 2000 = 478 :=
begin
  sorry
end

end special_prime_looking_count_is_478_l467_467520


namespace extreme_value_in_interval_l467_467654

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem extreme_value_in_interval (a : ℝ) (h : ∃ x ∈ Ioo a (a+1), ∃ c : ℝ, f c = f x) : a ∈ Ioo 0 1 := sorry

end extreme_value_in_interval_l467_467654


namespace Anya_loss_games_l467_467026

noncomputable def game_indices_Anya_lost (Anya Bella Valya Galya Dasha : ℕ) : Set ℕ :=
  {4, 8, 12, 16}

theorem Anya_loss_games
  (Anya Bella Valya Galya Dasha total_games : ℕ)
  (h1 : Anya = 4)
  (h2 : Bella = 6)
  (h3 : Valya = 7)
  (h4 : Galya = 10)
  (h5 : Dasha = 11)
  (h6 : total_games = (Anya + Bella + Valya + Galya + Dasha) / 2)
  (h7 : total_games = 19) :
  game_indices_Anya_lost Anya Bella Valya Galya Dasha = {4, 8, 12, 16} :=
  by sorry

end Anya_loss_games_l467_467026


namespace ellipse_standard_equation_l467_467085

theorem ellipse_standard_equation (a b c : ℝ) (h1 : 2 * a = 8) (h2 : c / a = 3 / 4) (h3 : b^2 = a^2 - c^2) :
  (x y : ℝ) →
  (x^2 / a^2 + y^2 / b^2 = 1 ∨ x^2 / b^2 + y^2 / a^2 = 1) :=
by
  sorry

end ellipse_standard_equation_l467_467085


namespace root_and_inequality_exists_l467_467103

noncomputable def polynomial (z: ℂ) (c: Fin (n + 1) → ℝ): ℂ :=
  z^n + (Fin (n + 1) → ℝ) 1 z^(n-1) + (Fin (n + 1) → ℝ) 2 z^(n-2) +
  ... + (Fin (n + 1) → ℝ) (n - 1) z + (Fin (n + 1) → ℝ) (n)

theorem root_and_inequality_exists (n : ℕ) (c: Fin (n + 1) → ℝ) 
(hp : |polynomial (complex.i) c | < 1) :
  ∃ a b : ℝ, polynomial (complex.ofReal a + complex.i * b) c = 0 ∧
             ((a^2 + b^2 + 1)^2 < 4 * b^2 + 1) :=
sorry

end root_and_inequality_exists_l467_467103


namespace income_distribution_l467_467858

theorem income_distribution 
  (I : ℝ) 
  (distributed_percent : ℝ) 
  (donated_percent : ℝ) 
  (remaining_amount : ℝ) 
  (distributed_amount : ℝ := distributed_percent * I / 100)
  (remaining_after_distribution : ℝ := I - distributed_amount)
  (donated_amount : ℝ := donated_percent * remaining_after_distribution / 100)
  (remaining_after_donation : ℝ := remaining_after_distribution - donated_amount)
  (deposited_amount : ℝ := remaining_after_donation - remaining_amount)
  (x : ℝ := deposited_amount * 100 / I) : 
  I = 1_200_000 → 
  distributed_percent = 60 → 
  donated_percent = 5 → 
  remaining_amount = 60_000 → 
  x = 33 := 
by 
  intros hI hD hD2 hR 
  rw [hI, hD, hD2, hR]
  sorry

end income_distribution_l467_467858


namespace distinct_four_digit_numbers_l467_467144

theorem distinct_four_digit_numbers : 
  {n : ℕ | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3, 4, 5} ∧ c ∈ {1, 2, 3, 4, 5} ∧ d ∈ {1, 2, 3, 4, 5}
  } = 120 :=
by
  sorry

end distinct_four_digit_numbers_l467_467144


namespace average_age_l467_467743

theorem average_age (avg_fifth_graders avg_parents avg_teachers : ℕ)
  (num_fifth_graders num_parents num_teachers : ℕ)
  (h1 : avg_fifth_graders = 10) (h2 : num_fifth_graders = 40)
  (h3 : avg_parents = 35) (h4 : num_parents = 60)
  (h5 : avg_teachers = 45) (h6 : num_teachers = 10) :
  (2950 / (num_fifth_graders + num_parents + num_teachers) : ℚ) = 26.81818181818182 := by
  -- Given conditions
  have fifth_graders_total : ℕ := num_fifth_graders * avg_fifth_graders,
  have parents_total : ℕ := num_parents * avg_parents,
  have teachers_total : ℕ := num_teachers * avg_teachers,

  -- Equalities as per the provided solution
  have total_age : ℕ := fifth_graders_total + parents_total + teachers_total,
  have total_individuals : ℕ := num_fifth_graders + num_parents + num_teachers,

  -- Calculate total age and verify equality to provided total age
  suffices : total_age = 2950, from sorry, -- to complete the proof

  -- Ensure average calculation holds true
  suffices : (total_age : ℚ) / total_individuals = 26.81818181818182, from sorry -- to complete the proof

end average_age_l467_467743


namespace num_factors_1320_l467_467188

theorem num_factors_1320 : 
  let n := 1320
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 11
  let e1 := 3
  let e2 := 1
  let e3 := 1
  let e4 := 1
  factors_count (p1^e1 * p2^e2 * p3^e3 * p4^e4) = 32 := 
by
  let prime_factors := [(p1, e1), (p2, e2), (p3, e3), (p4, e4)]
  sorry

end num_factors_1320_l467_467188


namespace acute_triangle_inequality_l467_467675

variable {A B C : ℝ} (h1 : 0 < A ∧ A < π / 2) (h2 : 0 < B ∧ B < π / 2) (h3 : 0 < C ∧ C < π / 2)
  (hABC : A + B + C = π)

theorem acute_triangle_inequality :
  cos A * cos B + cos B * cos C + cos C * cos A ≤ 1 / 2 + 2 * cos A * cos B * cos C :=
by
  sorry

end acute_triangle_inequality_l467_467675


namespace washing_machines_removed_l467_467250

theorem washing_machines_removed (crates boxes_per_crate washing_machines_per_box washing_machines_removed_per_box : ℕ) 
  (h_crates : crates = 10) (h_boxes_per_crate : boxes_per_crate = 6) 
  (h_washing_machines_per_box : washing_machines_per_box = 4) 
  (h_washing_machines_removed_per_box : washing_machines_removed_per_box = 1) :
  crates * boxes_per_crate * washing_machines_removed_per_box = 60 :=
by
  rw [h_crates, h_boxes_per_crate, h_washing_machines_removed_per_box]
  exact Nat.mul_assoc crates boxes_per_crate washing_machines_removed_per_box ▸
         Nat.mul_assoc 10 6 1 ▸ rfl


end washing_machines_removed_l467_467250


namespace monotone_decreasing_interval_l467_467966

theorem monotone_decreasing_interval :
  ∀ k : ℤ, ∀ x : ℝ, (k * π - π / 4 ≤ x ∧ x ≤ k * π + π / 8) →
  ∀ x1 x2 : ℝ, (k * π - π / 4 ≤ x1 ∧ x1 ≤ k * π + π / 8) →
  (k * π - π / 4 ≤ x2 ∧ x2 ≤ k * π + π / 8) →
  x1 ≤ x2 → (log (1 / 2) (sin x1 * cos x1 + cos x1 ^ 2)) ≥ (log (1 / 2) (sin x2 * cos x2 + cos x2 ^ 2)) :=
by
  intro k x hx x1 x2 hx1 hx2 hx1_le_x2
  sorry

end monotone_decreasing_interval_l467_467966


namespace series_sum_ln2_l467_467838

theorem series_sum_ln2 :
  (∑ n : ℕ, 1 / ((4 * n + 1) * (4 * n + 2) * (4 * n + 3))) = (1 / 4) * Real.log 2 :=
sorry

end series_sum_ln2_l467_467838


namespace star_comm_l467_467110

-- Definition of the binary operation star
def star (a b : ℝ) : ℝ := (a^b) / (b^a)

theorem star_comm (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : star a b = star b a := by
  sorry

end star_comm_l467_467110


namespace curve_points_satisfy_equation_l467_467238

theorem curve_points_satisfy_equation (C : Set (ℝ × ℝ)) (f : ℝ × ℝ → ℝ) :
  (∀ p : ℝ × ℝ, p ∈ C → f p = 0) → (∀ q : ℝ × ℝ, f q ≠ 0 → q ∉ C) :=
by
  intro h₁
  intro q
  intro h₂
  sorry

end curve_points_satisfy_equation_l467_467238


namespace anya_lost_games_l467_467052

-- Define the girls playing table tennis
def girls : Type := {Anya Bella Valya Galya Dasha : girls}

-- Define the conditions
def games_played : girls → ℕ
| Anya => 4
| Bella => 6
| Valya => 7
| Galya => 10
| Dasha => 11

-- Total number of games played
def total_games : ℕ := 19
#eval total_games = (games_played Anya + games_played Bella + games_played Valya + games_played Galya + games_played Dasha) / 2

-- The main theorem to be proven
theorem anya_lost_games :
  ∀ i : ℕ, i ∈ {4, 8, 12, 16} ↔ Anya played and lost in game i
:=
by
  sorry

end anya_lost_games_l467_467052


namespace persons_in_office_l467_467665

theorem persons_in_office
  (P : ℕ)
  (h1 : (P - (1/7 : ℚ)*P) = (6/7 : ℚ)*P)
  (h2 : (16.66666666666667/100 : ℚ) = 1/6) :
  P = 35 :=
sorry

end persons_in_office_l467_467665


namespace determine_f_l467_467462

open Real

def f (c : ℝ) (x : ℝ) : ℝ := c / x

theorem determine_f (f : ℝ → ℝ) 
  (h_pos : ∀ x, x > 0 → f x > 0)
  (h_equal_areas : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 →
    abs (x₁ * f x₂ - x₂ * f x₁) = (f x₁ + f x₂) * (x₂ - x₁)) :
  ∃ c > 0, ∀ x, x > 0 → f x = c / x :=
by
  sorry

end determine_f_l467_467462


namespace circle_square_area_ratio_l467_467486

theorem circle_square_area_ratio
  (r s : ℝ)
  (h1 : ∀ {s r : ℝ}, ∃ s r, r = s * sqrt 2 / 2)
  (h2 : s = r * sqrt 2) :
  (π * r ^ 2) / (s ^ 2) = π / 2 := by
  -- Proof goes here
  sorry

end circle_square_area_ratio_l467_467486


namespace log_identity_l467_467905

noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem log_identity : log 2 5 * log 3 2 * log 5 3 = 1 :=
by sorry

end log_identity_l467_467905


namespace fraction_at_x_eq_4571_div_39_l467_467515

def numerator (x : ℕ) : ℕ := x^6 - 16 * x^3 + x^2 + 64
def denominator (x : ℕ) : ℕ := x^3 - 8

theorem fraction_at_x_eq_4571_div_39 : numerator 5 / denominator 5 = 4571 / 39 :=
by
  sorry

end fraction_at_x_eq_4571_div_39_l467_467515


namespace anya_lost_games_l467_467042

/-- 
Five girls (Anya, Bella, Valya, Galya, Dasha) played several games of table tennis. At any given time, two girls were playing,
and the other three were resting. The girl who lost the game went to rest, and her place was taken by the girl who had rested 
the most rounds. There are no ties in table tennis. Anya played 4 games, Bella played 6 games, Valya played 7 games, Galya 
played 10 games, and Dasha played 11 games.

We need to prove that the numbers of all games in which Anya lost are 4, 8, 12, and 16.
-/
theorem anya_lost_games :
  (∃ games : list ℕ,
    games.length = 19 ∧
    (∀ i, i ∈ games → i = 4 ∨ i = 8 ∨ i = 12 ∨ i = 16) ∧
    (∀ i, i ∉ games → true)) :=
by
  sorry

end anya_lost_games_l467_467042


namespace certain_number_105_l467_467652

theorem certain_number_105 (a x : ℕ) (h0 : a = 105) (h1 : a^3 = x * 25 * 45 * 49) : x = 21 := by
  sorry

end certain_number_105_l467_467652


namespace distinct_four_digit_numbers_l467_467138

theorem distinct_four_digit_numbers : 
  let digits := {1, 2, 3, 4, 5} in (digits.card = 5) →
  ∃ count : ℕ, count = 5 * 4 * 3 * 2 ∧ count = 120 := 
by
  intros
  use 5 * 4 * 3 * 2
  split
  · refl
  · exact 120

end distinct_four_digit_numbers_l467_467138


namespace simplify_f_min_f_value_l467_467073

-- Definition of f(x) as given in the problem statement
def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 4 + 2 * (Real.cos x) ^ 4 + (Real.cos (2 * x))^2 - 3

-- Interval for x
def interval : Set ℝ := { x | Real.pi / 16 ≤ x ∧ x ≤ 3 * Real.pi / 16 }

-- Statement to prove: Simplification of f(x)
theorem simplify_f (x : ℝ) : f x = Real.cos (4 * x) - 1 :=
  sorry

-- Statement to prove: Minimum value of f(x) on the given interval
theorem min_f_value (hx : x ∈ interval) :
  Real.Inf (Set.image f interval) = -((Real.sqrt 2) + 2) / 2 ∧
  ∃ a ∈ interval, a = 3 * Real.pi / 16 ∧ f a = -((Real.sqrt 2) + 2) / 2 :=
  sorry

end simplify_f_min_f_value_l467_467073


namespace stuart_chords_l467_467359

-- Definitions and conditions based on the given problem
def concentric_circles (small large : Circle) : Prop :=
  small.center = large.center ∧ small.radius < large.radius

variables {C1 C2 : Circle} (P Q : Point)

-- Given conditions
def conditions (C1 C2 : Circle) (P Q : Point) : Prop :=
  concentric_circles C1 C2 ∧ tangent C1.line PQ ∧ angle P Q = 60

-- Statement of the proof problem:
theorem stuart_chords (C1 C2 : Circle) (P Q : Point) (h : conditions C1 C2 P Q) : 
  ∃ n : ℕ, n = 3 := sorry

end stuart_chords_l467_467359


namespace triangle_inequality_a_triangle_inequality_b_l467_467287

variable (α β γ : ℝ)

-- Assume α, β, γ are angles of a triangle
def is_triangle (α β γ : ℝ) := 
  α + β + γ = π ∧ α > 0 ∧ β > 0 ∧ γ > 0

theorem triangle_inequality_a (h : is_triangle α β γ) :
  (1 - Real.cos α) * (1 - Real.cos β) * (1 - Real.cos γ) ≥ 
  (Real.cos α) * (Real.cos β) * (Real.cos γ) := sorry

theorem triangle_inequality_b (h : is_triangle α β γ) :
  12 * (Real.cos α) * (Real.cos β) * (Real.cos γ) ≤ 
  2 * (Real.cos α) * (Real.cos β) + 2 * (Real.cos α) * (Real.cos γ) + 2 * (Real.cos β) * (Real.cos γ) ∧
  2 * (Real.cos α) * (Real.cos β) + 2 * (Real.cos α) * (Real.cos γ) + 2 * (Real.cos β) * (Real.cos γ) ≤
  (Real.cos α) + (Real.cos β) + (Real.cos γ) := sorry

end triangle_inequality_a_triangle_inequality_b_l467_467287


namespace series_convergence_l467_467928

theorem series_convergence :
  ∑ n : ℕ, (2^n) / (3^(2^n) + 1) = 1 / 2 :=
by
  sorry

end series_convergence_l467_467928


namespace anya_lost_games_l467_467038

/-- 
Five girls (Anya, Bella, Valya, Galya, Dasha) played several games of table tennis. At any given time, two girls were playing,
and the other three were resting. The girl who lost the game went to rest, and her place was taken by the girl who had rested 
the most rounds. There are no ties in table tennis. Anya played 4 games, Bella played 6 games, Valya played 7 games, Galya 
played 10 games, and Dasha played 11 games.

We need to prove that the numbers of all games in which Anya lost are 4, 8, 12, and 16.
-/
theorem anya_lost_games :
  (∃ games : list ℕ,
    games.length = 19 ∧
    (∀ i, i ∈ games → i = 4 ∨ i = 8 ∨ i = 12 ∨ i = 16) ∧
    (∀ i, i ∉ games → true)) :=
by
  sorry

end anya_lost_games_l467_467038


namespace king_plan_feasibility_l467_467476

-- Create a predicate for the feasibility of the king's plan
def feasible (n : ℕ) : Prop :=
  (n = 6 ∧ true) ∨ (n = 2004 ∧ false)

theorem king_plan_feasibility :
  ∀ n : ℕ, feasible n :=
by
  intro n
  sorry

end king_plan_feasibility_l467_467476


namespace prob_divisible_by_5_l467_467285

def S : Set ℕ := {n | ∃ j k m, j < k ∧ k < m ∧ m ≤ 29 ∧ n = 2^j + 2^k + 2^m}

theorem prob_divisible_by_5 (p q : ℕ) (hpq_coprime : Nat.coprime p q) (hprob : p / q = 50 / 4060) : p + q = 411 :=
by
  sorry

end prob_divisible_by_5_l467_467285


namespace num_factors_1320_l467_467186

theorem num_factors_1320 : 
  let n := 1320
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 11
  let e1 := 3
  let e2 := 1
  let e3 := 1
  let e4 := 1
  factors_count (p1^e1 * p2^e2 * p3^e3 * p4^e4) = 32 := 
by
  let prime_factors := [(p1, e1), (p2, e2), (p3, e3), (p4, e4)]
  sorry

end num_factors_1320_l467_467186


namespace shortest_broken_line_l467_467632

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  slope : ℝ
  intercept : ℝ

def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

def on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

def direction (p1 p2 : Point) (d : Point) : Prop :=
  (p2.y - p1.y) / (p2.x - p1.x) = d.y / d.x

theorem shortest_broken_line 
  (A B M N : Point) (line1 line2 : Line) (E F : Point)
  (h_parallel : parallel line1 line2)
  (h_on_line1_M : on_line M line1)
  (h_on_line2_N : on_line N line2)
  (h_opposite_sides : A.y * B.y < 0)
  (h_direction_MN : direction M N E F) :
  ∃ M N : Point, (on_line M line1) ∧ (on_line N line2) ∧
  (A.M.N.B with direction E F has the minimal length) := sorry

end shortest_broken_line_l467_467632


namespace max_siskins_on_poles_l467_467883

-- Definitions based on problem conditions
def pole : Type := ℕ
def siskins (poles : pole) : Prop := poles ≤ 25
def adjacent (p₁ p₂ : pole) : Prop := (p₁ = p₂ + 1) ∨ (p₁ = p₂ - 1)

-- Given conditions
def conditions (p : pole → bool) : Prop :=
  ∀ p₁ p₂ : pole, p p₁ = true → p p₂ = true → adjacent p₁ p₂ → false

-- Main problem statement
theorem max_siskins_on_poles : ∃ p : pole → bool, (∀ i : pole, p i = true → siskins i) ∧ (conditions p) ∧ (∑ i in finset.range 25, if p i then 1 else 0 = 24) :=
begin
  sorry
end

end max_siskins_on_poles_l467_467883


namespace exists_irreducible_fractions_sum_to_86_over_111_l467_467557

theorem exists_irreducible_fractions_sum_to_86_over_111 :
  ∃ (a b p q : ℕ), (0 < a) ∧ (0 < b) ∧ (p ≤ 100) ∧ (q ≤ 100) ∧ (Nat.coprime a p) ∧ (Nat.coprime b q) ∧ (a * q + b * p = 86 * (p * q) / 111) :=
by
  sorry

end exists_irreducible_fractions_sum_to_86_over_111_l467_467557


namespace number_of_terms_power_of_2_l467_467535

theorem number_of_terms_power_of_2 :
  (exists (seq : Fin 10 → ℕ), (∀ i, seq i = 2 ^ (10 * i)) ∧ (∑ i, seq i) = (2 ^ 97 + 1) / (2 ^ 5 + 1)) :=
sorry

end number_of_terms_power_of_2_l467_467535


namespace solve_quadratic_inequality_l467_467392

theorem solve_quadratic_inequality (x : ℝ) :
  (x^2 - 2*x - 3 < 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end solve_quadratic_inequality_l467_467392


namespace shorts_more_than_checkered_l467_467660

noncomputable def total_students : ℕ := 81

noncomputable def striped_shirts : ℕ := (2 * total_students) / 3

noncomputable def checkered_shirts : ℕ := total_students - striped_shirts

noncomputable def shorts : ℕ := striped_shirts - 8

theorem shorts_more_than_checkered :
  shorts - checkered_shirts = 19 :=
by
  sorry

end shorts_more_than_checkered_l467_467660


namespace cosine_of_angle_BHD_l467_467662

noncomputable def rectangular_solid_cos_angle_BHD (CD AB AD : ℝ) 
                                                  (angle_DHG angle_FHB : ℝ) 
                                                  (hCD : CD = 1) 
                                                  (hAB : AB = 2) 
                                                  (hAD : AD = 3) 
                                                  (hangle_DHG : angle_DHG = 30 * (Real.pi / 180)) 
                                                  (hangle_FHB : angle_FHB = 45 * (Real.pi / 180)) :
  ∃ (cos_BHD : ℝ), cos_BHD = (5 * Real.sqrt 2) / 12 :=
by
  -- Here would be the proof, but we use sorry to denote it's omitted.
  sorry

-- This theorem is effectively stating exactly what we want to prove.
theorem cosine_of_angle_BHD : rectangular_solid_cos_angle_BHD 1 2 3 (30 * (Real.pi / 180)) (45 * (Real.pi / 180))
  1 
  2 
  3 
  (30 * (Real.pi / 180)) 
  (45 * (Real.pi / 180)) :=
by
  -- Proof of this specific case is omitted for brevity.
  sorry

end cosine_of_angle_BHD_l467_467662


namespace smallest_six_digit_round_up_l467_467344

theorem smallest_six_digit_round_up :
  (∀ d ∈ [0, 1, 3, 5, 7, 9], d ∈ σ ∧ ∀ x ∈ σ, σ.perm [0, 1, 3, 5, 7, 9] → 
  (smallest_six_digit_number σ) = 103579) →
  round_to_nearest_thousand 103579 = 104000 :=
by
  sorry

end smallest_six_digit_round_up_l467_467344


namespace meso_tyler_time_to_type_40_pages_l467_467310

-- Define the typing speeds
def meso_speed : ℝ := 15 / 5 -- 3 pages per minute
def tyler_speed : ℝ := 15 / 3 -- 5 pages per minute
def combined_speed : ℝ := meso_speed + tyler_speed -- 8 pages per minute

-- Define the number of pages to type
def pages : ℝ := 40

-- Prove the time required to type the pages together
theorem meso_tyler_time_to_type_40_pages : 
  ∃ (t : ℝ), t = pages / combined_speed :=
by
  use 5 -- this is the correct answer
  sorry

end meso_tyler_time_to_type_40_pages_l467_467310


namespace equal_distances_l467_467897

-- Define the configuration of the circles and points
variables {O1 O2 A B C D P E F : Type}

axioms
  (h1 : ∀ (O1 O2 A B C D P E F : Type), 
    ∃ (O1 O2 : Type) (A B : set O1) (C D : set O2) (P : set (circle O1 O2)),
    intersect_circle(O1, O2) = {A, B} ∧ 
    external_angle_bisector_angle(O1, A, O2) = {C, D} ∧
    is_on_circle(B, circle C D, P) ∧ 
    intersection_point(C, line C P, O1) = E ∧ 
    intersection_point(D, line D P, O2) = F)

-- Prove that the distances from P to E and P to F are equal
theorem equal_distances : dist(P, E) = dist(P, F) :=
by sorry

end equal_distances_l467_467897


namespace alternating_sequence_converges_to_five_l467_467826

noncomputable def y : ℝ :=
  let rec y' : ℕ → ℝ := λ n, if n = 0 then 3 else 1 + 5 / y' (n - 1)
  in 3 + 5 / y' 1

theorem alternating_sequence_converges_to_five : y = 5 :=
  sorry

end alternating_sequence_converges_to_five_l467_467826


namespace numberOfIncreasingMatrices_l467_467376

noncomputable def countIncreasingMatrices : ℕ :=
  1036800

theorem numberOfIncreasingMatrices (M : matrix (fin 4) (fin 4) ℕ) :
  (∀ (i : fin 4) (j : fin 4), 1 ≤ M i j ∧ M i j ≤ 16) ∧
  (∀ (i1 i2 j : fin 4), i1 < i2 → M i1 j < M i2 j) ∧ 
  (∀ (i j1 j2 : fin 4), j1 < j2 → M i j1 < M i j2) →
  ∃ (n : ℕ), n = 1036800 :=
by sorry

end numberOfIncreasingMatrices_l467_467376


namespace tree_height_relationship_l467_467266

theorem tree_height_relationship (x : ℕ) : ∃ h : ℕ, h = 80 + 2 * x :=
by
  sorry

end tree_height_relationship_l467_467266


namespace total_splash_width_l467_467422

def pebbles : ℚ := 1/5
def rocks : ℚ := 2/5
def boulders : ℚ := 7/5
def mini_boulders : ℚ := 4/5
def large_pebbles : ℚ := 3/5

def num_pebbles : ℚ := 10
def num_rocks : ℚ := 5
def num_boulders : ℚ := 4
def num_mini_boulders : ℚ := 3
def num_large_pebbles : ℚ := 7

theorem total_splash_width : 
  num_pebbles * pebbles + 
  num_rocks * rocks + 
  num_boulders * boulders + 
  num_mini_boulders * mini_boulders + 
  num_large_pebbles * large_pebbles = 16.2 := by
  sorry

end total_splash_width_l467_467422


namespace area_of_shaded_region_l467_467107

theorem area_of_shaded_region
  (r_large : ℝ) (r_small : ℝ) (n_small : ℕ) (π : ℝ)
  (A_large : ℝ) (A_small : ℝ) (A_7_small : ℝ) (A_shaded : ℝ)
  (h1 : r_large = 20)
  (h2 : r_small = 10)
  (h3 : n_small = 7)
  (h4 : π = 3.14)
  (h5 : A_large = π * r_large^2)
  (h6 : A_small = π * r_small^2)
  (h7 : A_7_small = n_small * A_small)
  (h8 : A_shaded = A_large - A_7_small) :
  A_shaded = 942 :=
by
  sorry

end area_of_shaded_region_l467_467107


namespace triangle_inequality_l467_467298

-- Given conditions: a, b, c are the side lengths of a triangle.
variables (a b c : ℝ) (h1: a + b > c) (h2: b + c > a) (h3: c + a > b)

theorem triangle_inequality :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧ (a = b ∧ b = c → (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0)) :=
begin
  sorry
end

end triangle_inequality_l467_467298


namespace problem_1_problem_2_l467_467621

noncomputable def point_on_line (t : ℝ) : ℝ × ℝ :=
  (1 + (1/2) * t, (√3/2) * t)

def curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem problem_1 :
  ∀ (t1 t2 θ1 θ2 : ℝ),
    curve_C1 θ1 = point_on_line t1 →
    curve_C1 θ2 = point_on_line t2 →
    dist (curve_C1 θ1) (curve_C1 θ2) = 1 :=
by
  intros t1 t2 θ1 θ2 h1 h2
  sorry

def curve_C2 (θ : ℝ) : ℝ × ℝ :=
  (1/2 * Real.cos θ, (√3/2) * Real.sin θ)

def distance_to_line_l (p : ℝ × ℝ) : ℝ :=
  |(√3/2) * p.1 - (√3/2) * p.2 - √3 | / Real.sqrt 4

theorem problem_2 :
  ∀ (θ : ℝ),
    ∃ (d : ℝ),
      d = distance_to_line_l (curve_C2 θ) ∧
      d = √6 / 4 * (√2 - 1) :=
by
  intro θ
  use distance_to_line_l (curve_C2 θ)
  split
  · rfl
  · sorry

end problem_1_problem_2_l467_467621


namespace find_MN_length_l467_467590

/-- 
Given a rectangle ABCD. 
A circle intersects the side AB at points K and L. 
Find the length of segment MN if AK = 10, KL = 17, DN = 7. 
-/
theorem find_MN_length 
  (ABCD : Type)
  (circle_center : Point)
  (A B C D K L M N : Point)
  (rectangle_ABCD : is_rectangle ABCD A B C D)
  (circle_intersects_AB_at_K_L : intersects_circle circle_center A B K L)
  (AK : length A K = 10)
  (KL : length K L = 17)
  (DN : length D N = 7) : 
  length M N = 23 :=
sorry

end find_MN_length_l467_467590


namespace find_two_irreducible_fractions_l467_467552

theorem find_two_irreducible_fractions :
  ∃ (a b d1 d2 : ℕ), 
    (1 ≤ a) ∧ 
    (1 ≤ b) ∧ 
    (gcd a d1 = 1) ∧ 
    (gcd b d2 = 1) ∧ 
    (1 ≤ d1) ∧ 
    (d1 ≤ 100) ∧ 
    (1 ≤ d2) ∧ 
    (d2 ≤ 100) ∧ 
    (a / (d1 : ℚ) + b / (d2 : ℚ) = 86 / 111) := 
by {
  sorry
}

end find_two_irreducible_fractions_l467_467552


namespace summation_series_equals_half_l467_467939

theorem summation_series_equals_half :
  (\sum_{n=0}^{∞} 2^n / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end summation_series_equals_half_l467_467939


namespace F_domain_and_odd_minimum_value_of_M_l467_467619

noncomputable def f (x : ℝ) : ℝ := 2 - 3 * log x / log 2
noncomputable def g (x : ℝ) : ℝ := log x / log 2

noncomputable def F (x : ℝ) : ℝ := g ((1 - x) / (1 + x))
def domain_F : Set ℝ := { x | x > -1 ∧ x < 1}

theorem F_domain_and_odd :
  (∀ x, x ∈ domain_F ↔ (1 - x) / (1 + x) > 0) ∧
  (∀ x, x ∈ domain_F → F (-x) = -F(x)) :=
by
  sorry

noncomputable def M (x : ℝ) : ℝ := (f x + g x + abs (f x - g x)) / 2

theorem minimum_value_of_M : ∃ x, M x = 1 / 2 :=
by
  sorry

end F_domain_and_odd_minimum_value_of_M_l467_467619


namespace total_pizzas_made_l467_467635

theorem total_pizzas_made (hc1 : Heather made 4 * Craig made on day1)
                          (hc2 : Heather made on day2 = Craig made on day2 - 20)
                          (hc3 : Craig made on day1 = 40)
                          (hc4 : Craig made on day2 = Craig made on day1 + 60) :
   Heather made on day1 + Craig made on day1 + Heather made on day2 + Craig made on day2 = 380 := 
by
  sorry

end total_pizzas_made_l467_467635


namespace sum_of_nus_is_45_l467_467799

noncomputable def sum_of_valid_nu : ℕ :=
  ∑ ν in {ν | ν > 0 ∧ Nat.lcm ν 24 = 72}.toFinset, ν

theorem sum_of_nus_is_45 : sum_of_valid_nu = 45 := by
  sorry

end sum_of_nus_is_45_l467_467799


namespace determine_range_of_a_l467_467750

theorem determine_range_of_a (a : ℝ) : 
  (∀ x : ℝ, -1 ≤ sin x ∧ sin x ≤ 1 →
    (sin x = 1 → (sin x - a)^2 + 1 = 1) ∧
    (sin x = a → (\sin x - a)^2 + 1 = 1)) →
  -1 ≤ a ∧ a ≤ 0 :=
by
  sorry

end determine_range_of_a_l467_467750


namespace problem1_part1_problem1_part2_problem2_l467_467981

open Set

-- Definitions for sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def U : Set ℝ := univ

-- Part (1) of the problem
theorem problem1_part1 : A ∩ B = {x | 2 < x ∧ x ≤ 3} :=
sorry

theorem problem1_part2 : A ∪ (U \ B) = {x | x ≤ 3} :=
sorry

-- Definitions for set C
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Part (2) of the problem
theorem problem2 (a : ℝ) (h : C a ⊆ A) : 1 < a ∧ a ≤ 3 :=
sorry

end problem1_part1_problem1_part2_problem2_l467_467981


namespace stuart_segments_l467_467357

/-- Stuart draws a pair of concentric circles and chords of the large circle
    each tangent to the small circle.
    Given the angle ABC is 60 degrees, prove that he draws 3 segments
    before returning to his starting point. -/
theorem stuart_segments (angle_ABC : ℝ) (h : angle_ABC = 60) : 
  let n := 3 in n = 3 :=
sorry

end stuart_segments_l467_467357


namespace correct_option_l467_467827

-- Defining the conditions for each option
def optionA (m n : ℝ) : Prop := (m / n)^7 = m^7 * n^(1/7)
def optionB : Prop := (4)^(4/12) = (-3)^(1/3)
def optionC (x y : ℝ) : Prop := ((x^3 + y^3)^(1/4)) = (x + y)^(3/4)
def optionD : Prop := (9)^(1/6) = 3^(1/3)

-- Asserting that option D is correct
theorem correct_option : optionD :=
by
  sorry

end correct_option_l467_467827


namespace arithmetic_square_root_of_quarter_l467_467369

theorem arithmetic_square_root_of_quarter : (∃ x : ℝ, x^2 = 1 / 4) ∧ (nonneg (1 / 4)) → (frac_one_four : ℝ) := 
by 
  sorry

end arithmetic_square_root_of_quarter_l467_467369


namespace Fermatville_temperature_range_l467_467317

theorem Fermatville_temperature_range 
  (min_temp : ℤ) (max_temp : ℤ) 
  (h_min_temp : min_temp = -11) 
  (h_max_temp : max_temp = 14) :
  max_temp - min_temp = 25 :=
by {
  rw [h_min_temp, h_max_temp],
  norm_num,
}

end Fermatville_temperature_range_l467_467317


namespace max_value_of_a_squared_b_squared_c_squared_l467_467292

theorem max_value_of_a_squared_b_squared_c_squared
  (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_constraint : a + 2 * b + 3 * c = 1) : a^2 + b^2 + c^2 ≤ 1 :=
sorry

end max_value_of_a_squared_b_squared_c_squared_l467_467292


namespace maximal_points_two_manhattan_distances_l467_467839

noncomputable def point := (ℝ × ℝ)

def manhattan_distance (p q : point) : ℝ :=
  |p.1 - q.1| + |p.2 - q.2|

def has_two_distinct_manhattan_distances (s : set point) : Prop :=
  ∃ d1 d2 : ℝ, d1 ≠ d2 ∧ ∀ (p q ∈ s), p ≠ q → manhattan_distance p q = d1 ∨ manhattan_distance p q = d2

theorem maximal_points_two_manhattan_distances :
  ∀ s : set point, has_two_distinct_manhattan_distances s → set.finite s → card s ≤ 9 :=
sorry

end maximal_points_two_manhattan_distances_l467_467839


namespace num_factors_1320_l467_467185

theorem num_factors_1320 : 
  let n := 1320
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 11
  let e1 := 3
  let e2 := 1
  let e3 := 1
  let e4 := 1
  factors_count (p1^e1 * p2^e2 * p3^e3 * p4^e4) = 32 := 
by
  let prime_factors := [(p1, e1), (p2, e2), (p3, e3), (p4, e4)]
  sorry

end num_factors_1320_l467_467185


namespace opposite_signs_add_same_signs_sub_l467_467074

-- Definitions based on the conditions
variables {a b : ℤ}

-- 1. Case when a and b have opposite signs
theorem opposite_signs_add (h₁ : |a| = 4) (h₂ : |b| = 3) (h₃ : a * b < 0) :
  a + b = 1 ∨ a + b = -1 := 
sorry

-- 2. Case when a and b have the same sign
theorem same_signs_sub (h₁ : |a| = 4) (h₂ : |b| = 3) (h₃ : a * b > 0) :
  a - b = 1 ∨ a - b = -1 := 
sorry

end opposite_signs_add_same_signs_sub_l467_467074


namespace tree_height_equation_l467_467269

theorem tree_height_equation (x : ℕ) : ∀ h : ℕ, h = 80 + 2 * x := by
  sorry

end tree_height_equation_l467_467269


namespace arithmetical_solution_l467_467536

-- Definition of the problem using the digits
def digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the arithmetical expression
def check_equation (a b c d : Nat) : Prop :=
  (a + b = c * d ∧ c * d = 12) ∧ (a * 10 + b = 96 ∧ 96 / d = 12)

-- The theorem statement encapsulating our conditions and answer
theorem arithmetical_solution (a b c d : Nat) (ha : a ∈ digits) (hb : b ∈ digits) (hc : c ∈ digits) (hd : d ∈ digits) :
  check_equation 5 7 3 4 :=
begin
  -- We verify that our equation with specific values holds true
  sorry
end

end arithmetical_solution_l467_467536


namespace knights_and_liars_l467_467319

theorem knights_and_liars : 
  ∃ K L : ℕ, K + L = 11 ∧ (
    (abs (K - L) = 1 ∨ abs (K - L) = 2 ∨ abs (K - L) = 3 ∨ abs (K - L) = 4 ∨
     abs (K - L) = 5 ∨ abs (K - L) = 6 ∨ abs (K - L) = 7 ∨ abs (K - L) = 8 ∨
     abs (K - L) = 9 ∨ abs (K - L) = 10 ∨ abs (K - L) = 11) ∧
    ((K = 1) ∧ (L = 10) ∨ (K = 10 ∧ L = 1))
  ) := by
  sorry

end knights_and_liars_l467_467319


namespace new_ratio_A_B_l467_467849

/-- The can initially has liquids A and B in the ratio 7:5.
    9 liters of the mixture are drawn off.
    The can initially contains 21 liters of liquid A.
    After drawing off, the can is filled with liquid B.
    The new mixture ratio is calculated to be 7:9. --/
theorem new_ratio_A_B (initial_ratio : ℚ) (volume_A : ℚ) (drawn_off_volume : ℚ) (new_volume_B : ℚ) :
  initial_ratio = 7 / 5 ∧ volume_A = 21 ∧ drawn_off_volume = 9 ∧ new_volume_B = 9 →
  let total_initial_volume := 36 in
  let remaining_volume_A := 15.75 in
  let remaining_volume_B := 11.25 + new_volume_B in
  let new_ratio := remaining_volume_A / remaining_volume_B in
  new_ratio = 7 / 9 :=
sorry

end new_ratio_A_B_l467_467849


namespace unique_polynomial_P_l467_467967

noncomputable def polynomial_P (P : polynomial ℝ) : Prop :=
P ≠ 0 ∧ P.degree ≠ 0 ∧ (P.comp P = (polynomial.X^2 - polynomial.X + 1) * P)

theorem unique_polynomial_P (P : polynomial ℝ) :
  polynomial_P P → P = polynomial.X^2 - polynomial.X :=
by
  sorry

end unique_polynomial_P_l467_467967


namespace count_whole_numbers_between_4_and_18_l467_467977

theorem count_whole_numbers_between_4_and_18 :
  ∀ (x : ℕ), 4 < x ∧ x < 18 ↔ ∃ n : ℕ, n = 13 :=
by sorry

end count_whole_numbers_between_4_and_18_l467_467977


namespace solution_equiv_problems_l467_467952

-- Defining the conditions
variables {a b c x y z : ℤ}
def conditions (a b c x y z : ℤ) : Prop :=
  (a + b + c = x * y * z) ∧ (x + y + z = a * b * c) ∧ (a ≥ b ∧ b ≥ c ∧ c ≥ 1) ∧ (x ≥ y ∧ y ≥ z ∧ z ≥ 1)

-- Expected solutions set
def expected_solutions : set (ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) := {
  (3, 2, 1, 3, 2, 1),
  (6, 1, 1, 2, 2, 2),
  (7, 1, 1, 3, 3, 1),
  (8, 1, 1, 5, 2, 1),
  (2, 2, 2, 6, 1, 1),
  (3, 3, 1, 7, 1, 1),
  (5, 2, 1, 8, 1, 1)
}

-- Lean statement to prove the equivalence
theorem solution_equiv_problems (a b c x y z : ℤ) :
  conditions a b c x y z ↔ (a, b, c, x, y, z) ∈ expected_solutions :=
sorry

end solution_equiv_problems_l467_467952


namespace no_solution_system_l467_467458

theorem no_solution_system (a : ℝ) :
  (∀ (x : ℝ), (a ≠ 0 → (a^2 * x + 2 * a) / (a * x - 2 + a^2) < 0 ∨ ax + a ≤ 5/4)) ∧ 
  (a = 0 → ¬ ∃ (x : ℝ), (a^2 * x + 2 * a) / (a * x - 2 + a^2) ≥ 0 ∧ ax + a > 5/4) ↔ 
  a ∈ Set.Iic (-1/2) ∪ {0} :=
by sorry

end no_solution_system_l467_467458


namespace not_in_category_l467_467441

-- Define the type for the category of algorithms we are discussing
inductive AlgorithmCategory
| circle_area : AlgorithmCategory
| equation_of_line : AlgorithmCategory
| arithmetic_rules : AlgorithmCategory

-- Define if a given operation belongs to the AlgorithmCategory
def belongs_to_algorithm_category (op : AlgorithmCategory) : Prop :=
  op = AlgorithmCategory.circle_area ∨ op = AlgorithmCategory.equation_of_line ∨
  op = AlgorithmCategory.arithmetic_rules

-- Define the operations as per the problem conditions
def operation_circle_area := AlgorithmCategory.circle_area
def operation_possibility_24 := "Calculating the possibilities of reaching 24 with any 4 drawn playing cards"
def operation_equation_line := AlgorithmCategory.equation_of_line
def operation_arithmetic_rules := AlgorithmCategory.arithmetic_rules

-- State the theorem that option B is the correct answer
theorem not_in_category : ¬ belongs_to_algorithm_category (AlgorithmCategory.circle_area) ∧
                         (operation_possibility_24 = "Calculating the possibilities of reaching 24 with any 4 drawn playing cards") :=
begin
  sorry
end

end not_in_category_l467_467441


namespace relationship_between_A_and_B_l467_467302

theorem relationship_between_A_and_B (a : ℝ)
  (hA : ∀ x : ℝ, x^2 + 2 * a * x + 1 > 0)
  (hB : ∀ x : ℝ, x > 0 → ∀ y : ℝ, y > x → (log (y) (2 * a - 1)) < (log x (2 * a - 1))) :
  (-1 < a ∧ a < 1) → (1 / 2 < a ∧ a < 1) :=
by
  sorry

end relationship_between_A_and_B_l467_467302


namespace no_equal_angle_octagon_with_given_sides_l467_467330

theorem no_equal_angle_octagon_with_given_sides :
  ¬ ∃ (α : ℝ) (sides : Fin 8 → ℕ),
    (∀ i, sides i ∈ {1, 2, 3, 4, 5, 6, 7, 8}) ∧
    (∃ perm : Fin 8 → Fin 8, 
      let ordered_sides := fun i => sides (perm i)
      in 
      (ordered_sides 0 = 1 ∨ ordered_sides 1 = 1 ∨ ordered_sides 2 = 1 ∨ ordered_sides 3 = 1 ∨ ordered_sides 4 = 1 ∨ ordered_sides 5 = 1 ∨ ordered_sides 6 = 1 ∨ ordered_sides 7 = 1) ∧
      -- combine the side lengths to test the alignment
    
      (* The equation formation here would be complex, indicating the contradictions with the irrationality of √2*)
    ) :=
sorry

end no_equal_angle_octagon_with_given_sides_l467_467330


namespace anya_game_losses_l467_467063

theorem anya_game_losses (games : ℕ → ℕ) (anya_games bella_games valya_games galya_games dasha_games : ℕ)
  (H_total : anya_games = 4 ∧ bella_games = 6 ∧ valya_games = 7 ∧ galya_games = 10 ∧ dasha_games = 11) :
  (games anya_games = 4 ∧ games bella_games = 6 ∧
   games valya_games = 7 ∧ games galya_games = 10 ∧
   games dasha_games = 11) →  [4, 8, 12, 16] := sorry

end anya_game_losses_l467_467063


namespace words_per_minute_l467_467639

-- Define the conditions in Lean 4
def free_time_hours : ℕ := 8
def first_movie_time_hours : ℚ := 3.5
def second_movie_time_hours : ℚ := 1.5
def total_words_read : ℕ := 1800

-- Define the theorem to prove
theorem words_per_minute :
  let total_movie_time := first_movie_time_hours + second_movie_time_hours in
  let time_left_for_reading := free_time_hours - total_movie_time in
  let time_left_for_reading_minutes := time_left_for_reading * 60 in
  total_words_read / time_left_for_reading_minutes = 10 :=
by
  sorry

end words_per_minute_l467_467639


namespace unique_non_integer_l467_467759

theorem unique_non_integer :
  let x := sqrt 2 - 1 in
  let a := x - sqrt 2 in
  let b := x - 1 / x in
  let c := x + 1 / x in
  let d := x^2 + 2 * sqrt 2 in
  (¬is_int a ∧ is_int b ∧ is_int c ∧ is_int d) ∨
  (is_int a ∧ ¬is_int b ∧ is_int c ∧ is_int d) ∨
  (is_int a ∧ is_int b ∧ ¬is_int c ∧ is_int d) ∨
  (is_int a ∧ is_int b ∧ is_int c ∧ ¬is_int d) :=
by sorry

end unique_non_integer_l467_467759


namespace distinct_four_digit_numbers_count_l467_467132

theorem distinct_four_digit_numbers_count :
  (Finset.card (Finset.filter (λ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ (∀ d ∈ [x / 1000, (x / 100) % 10, (x / 10) % 10, x % 10], d ∈ [1, 2, 3, 4, 5]) ∧ (denote_distinct_digits (x)) ) (Finset.Ico 1000 10000))) = 120 := 
sorry

def denote_distinct_digits (x : ℕ) : Prop :=
  ∀ i j, (i ≠ j) → (x / (10 ^ i)) % 10 ≠ (x / (10 ^ j)) % 10

end distinct_four_digit_numbers_count_l467_467132


namespace distinct_factors_1320_l467_467206

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end distinct_factors_1320_l467_467206


namespace meeting_point_correct_travel_distance_correct_l467_467634

open Real

def coordinate := (Real, Real)

def midpoint (p1 p2 : coordinate) : coordinate :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def distance (p1 p2 : coordinate) : Real :=
  sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def harry_start : coordinate := (10, -3)
def sandy_start : coordinate := (2, 7)

def meeting_point : coordinate := midpoint harry_start sandy_start

def harry_distance := distance harry_start meeting_point
def sandy_distance := distance sandy_start meeting_point

theorem meeting_point_correct : meeting_point = (6, 2) :=
  sorry

theorem travel_distance_correct: harry_distance = sqrt 41 ∧ sandy_distance = sqrt 41 :=
  sorry

end meeting_point_correct_travel_distance_correct_l467_467634


namespace sum_lcm_eq_72_l467_467820

theorem sum_lcm_eq_72 (s : Finset ℕ) 
  (h1 : ∀ ν ∈ s, Nat.lcm ν 24 = 72) :
  s.sum id = 180 :=
by
  have h2 : ∀ ν, ν ∈ s → ∃ n, ν = 3 * n := 
    by sorry
  have h3 : ∀ n, ∃ ν, ν = 3 * n ∧ ν ∈ s := 
    by sorry
  sorry

end sum_lcm_eq_72_l467_467820


namespace num_distinct_factors_1320_l467_467160

theorem num_distinct_factors_1320 : 
  ∃ n : ℕ, n = 1320 ∧ (finset.univ.filter (λ d, d > 0 ∧ 1320 % d = 0)).card = 32 := 
sorry

end num_distinct_factors_1320_l467_467160


namespace find_four_a_plus_three_b_l467_467578

-- Define the function f
def f (x : ℝ) (a b : ℝ) := x^2 + a * x + b

-- Formal problem statement in Lean
theorem find_four_a_plus_three_b (a b : ℝ) 
  (h₁ : ∀ x : ℝ, x ∈ set.Icc (-1 : ℝ) 1 → |f x a b| ≤ 1 / 2) : 
  4 * a + 3 * b = -3 / 2 := 
sorry

end find_four_a_plus_three_b_l467_467578


namespace well_depth_is_2000_l467_467863

-- Given conditions
def total_time : ℝ := 10
def stone_law (t₁ : ℝ) : ℝ := 20 * t₁^2
def sound_velocity : ℝ := 1120

-- Statement to be proven
theorem well_depth_is_2000 :
  ∃ (d t₁ t₂ : ℝ), 
    d = stone_law t₁ ∧ t₂ = d / sound_velocity ∧ t₁ + t₂ = total_time :=
sorry

end well_depth_is_2000_l467_467863


namespace simplify_expression_l467_467350

-- Define the question and conditions
theorem simplify_expression (x y : ℝ) (h : |x + 1| + (2 * y - 4)^2 = 0) :
  (2*x^2*y - 3*x*y) - 2*(x^2*y - x*y + 1/2*x*y^2) + x*y = 4 :=
by
  -- proof steps if needed, but currently replaced with 'sorry' to indicate proof needed
  sorry

end simplify_expression_l467_467350


namespace no_more_than_four_intersection_points_l467_467735

noncomputable def conic1 (a b c d e f : ℝ) (x y : ℝ) : Prop := 
  a * x^2 + 2 * b * x * y + c * y^2 + 2 * d * x + 2 * e * y = f

noncomputable def conic2_param (P Q A : ℝ → ℝ) (t : ℝ) : ℝ × ℝ :=
  (P t / A t, Q t / A t)

theorem no_more_than_four_intersection_points (a b c d e f : ℝ)
  (P Q A : ℝ → ℝ) :
  (∃ t1 t2 t3 t4 t5,
    conic1 a b c d e f (P t1 / A t1) (Q t1 / A t1) ∧
    conic1 a b c d e f (P t2 / A t2) (Q t2 / A t2) ∧
    conic1 a b c d e f (P t3 / A t3) (Q t3 / A t3) ∧
    conic1 a b c d e f (P t4 / A t4) (Q t4 / A t4) ∧
    conic1 a b c d e f (P t5 / A t5) (Q t5 / A t5)) → false :=
sorry

end no_more_than_four_intersection_points_l467_467735


namespace find_b_given_conditions_l467_467076

constant b : ℝ

-- Define the circle equation
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 2

-- Define the line equation
def line (x y : ℝ) : Prop := y = 2 * x + b

-- Define the point (1, 2)
def center : ℝ × ℝ := (1, 2)

-- Define the conditions of the segment lengths and distance from the center to the line
def segment_length_y_axis : ℝ := 2
def segment_length_line : ℝ := 2
def distance_from_center_to_line : Prop :=
  abs ((2 * (center.fst) - (center.snd) + b) / sqrt 5) = 1

-- State the theorem
theorem find_b_given_conditions :
  (circle 0 1 ∧ circle 0 3 ∧
  (abs ((2 * (center.fst) - (center.snd) + b) / sqrt 5) =  1)) →
  (b = sqrt 5) ∨ (b = -sqrt 5) := sorry

end find_b_given_conditions_l467_467076


namespace eq_triangle_fold_theorem_l467_467477

noncomputable def equilateral_triangle_fold {A B C : Type} 
  (side_length : ℝ) (distance_from_B : ℝ) (fold_length_sq : ℝ) : Prop :=
  side_length = 15 ∧
  distance_from_B = 11 ∧
  fold_length_sq = 25800 / 361

theorem eq_triangle_fold_theorem : 
  equilateral_triangle_fold 15 11 (25800 / 361) := 
by 
  sorry

end eq_triangle_fold_theorem_l467_467477


namespace rationalize_denominator_l467_467340

theorem rationalize_denominator (a b c : ℝ) (h1 : a = 7) (h2 : b = √98) (h3 : √98 = 7 * √2) :
  a / b * √2 = c ↔ c = √2 / 2 := by
  sorry

end rationalize_denominator_l467_467340


namespace smallest_positive_period_and_max_value_and_intervals_l467_467613

noncomputable def f (x : ℝ) : ℝ := 
  let π := Real.pi in 
  (Real.cos (π / 3 + x)) * (Real.cos (π / 3 - x)) - (Real.sin x) * (Real.cos x) + 1 / 4

theorem smallest_positive_period_and_max_value_and_intervals :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∀ x : ℝ, f x ≤ f (k * Real.pi - Real.pi / 8) ∀ k : ℤ) ∧
  (f (k * Real.pi - Real.pi / 8) = Real.sqrt 2 / 2 ∀ k : ℤ) ∧
  (f (x) is monotonically decreasing in { x | k * Real.pi - Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 8 }, ∀ k : ℤ) :=
sorry

end smallest_positive_period_and_max_value_and_intervals_l467_467613


namespace log_z_m_eq_60_l467_467100

noncomputable def problem_statement (x y z m : ℝ) : Prop :=
  x > 1 ∧ y > 1 ∧ z > 1 ∧ m > 0 ∧ log x m = 24 ∧ log y m = 40 ∧ log (x * y * z) m = 12

theorem log_z_m_eq_60 (x y z m : ℝ) (h : problem_statement x y z m) : log z m = 60 :=
sorry

end log_z_m_eq_60_l467_467100


namespace find_fractions_l467_467550

noncomputable def fractions_to_sum_86_111 : Prop :=
  ∃ (a b d₁ d₂ : ℕ), 0 < a ∧ 0 < b ∧ d₁ ≤ 100 ∧ d₂ ≤ 100 ∧
  Nat.gcd a d₁ = 1 ∧ Nat.gcd b d₂ = 1 ∧
  (a: ℚ) / d₁ + (b: ℚ) / d₂ = 86 / 111

theorem find_fractions : fractions_to_sum_86_111 :=
  sorry

end find_fractions_l467_467550


namespace dots_in_next_square_l467_467783

def first_square_dots : ℕ := 1
def second_square_dots : ℕ := 9
def third_square_dots : ℕ := 25

theorem dots_in_next_square : ∀ n, (n = 4) → (∃ k, k = n → k^2 = 49) :=
by
  intro n
  intro hn
  use 7
  intros hk
  rw ←hk
  norm_num
  sorry

end dots_in_next_square_l467_467783


namespace slant_height_of_cone_l467_467772

theorem slant_height_of_cone
  (r : ℝ) (CSA : ℝ) (l : ℝ)
  (hr : r = 14)
  (hCSA : CSA = 1539.3804002589986) :
  CSA = Real.pi * r * l → l = 35 := 
sorry

end slant_height_of_cone_l467_467772


namespace acetone_mass_percentage_O_l467_467964

-- Definition of atomic masses
def atomic_mass_C := 12.01
def atomic_mass_H := 1.008
def atomic_mass_O := 16.00

-- Definition of the molar mass of acetone
def molar_mass_acetone := (3 * atomic_mass_C) + (6 * atomic_mass_H) + atomic_mass_O

-- Definition of mass percentage of oxygen in acetone
def mass_percentage_O_acetone := (atomic_mass_O / molar_mass_acetone) * 100

theorem acetone_mass_percentage_O : mass_percentage_O_acetone = 27.55 := by sorry

end acetone_mass_percentage_O_l467_467964


namespace count_1320_factors_l467_467170

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end count_1320_factors_l467_467170


namespace complex_conjugate_quadrant_l467_467542

/-- Given the complex number Z = (2 - I) / (1 + I), prove that the conjugate of Z, which is (1 / 2) + (3 / 2) * I,
    results in a point located in the first quadrant of the complex plane. -/
theorem complex_conjugate_quadrant :
  let Z : ℂ := (2 - I) / (1 + I),
      conj_Z := conj Z in
  0 < conj_Z.re ∧ 0 < conj_Z.im :=
by {
  sorry
}

end complex_conjugate_quadrant_l467_467542


namespace maximize_product_geometric_sequence_l467_467702

noncomputable def a_n (a1 q : ℝ) (n : ℕ) := a1 * q^(n-1)
noncomputable def S_n (a1 q : ℝ) (n : ℕ) := a1 * (1 - q^n) / (1 - q)

theorem maximize_product_geometric_sequence :
  ∃ n : ℕ, 
  (0 < n ∧ n ≤ 4 ∧ n = 4 ∨ n = 5) → 
  (∀ (a1 q : ℝ), 
  (a1 * q * a1 * q^3 = 16) → 
  (S_n a1 q 3 = 28) → 
  (∃ max_n : ℕ, ∀ k : ℕ, a1 * q^(k-1) = (a_n a1 q 1 * a_n a1 q 2 * ... * a_n a1 q n)) :=
begin
  sorry
end

end maximize_product_geometric_sequence_l467_467702


namespace distinct_four_digit_numbers_l467_467145

theorem distinct_four_digit_numbers : 
  {n : ℕ | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3, 4, 5} ∧ c ∈ {1, 2, 3, 4, 5} ∧ d ∈ {1, 2, 3, 4, 5}
  } = 120 :=
by
  sorry

end distinct_four_digit_numbers_l467_467145


namespace max_n_value_l467_467079

def sum_of_digits (n : Nat) : Nat :=
  n.digits.sum

-- Condition: all digits of n are distinct
def distinct_digits (n : Nat) : Prop :=
  let digits := n.digits
  digits.nodup

-- Condition: sum of digits of 3n equals 3 times the sum of digits of n
def sum_property (n : Nat) : Prop :=
  sum_of_digits (3 * n) = 3 * sum_of_digits n

-- The statement to prove
theorem max_n_value : ∃ (n : Nat), distinct_digits n ∧ sum_property n ∧ n = 3210 :=
sorry

end max_n_value_l467_467079


namespace students_enrolled_only_in_english_l467_467450

theorem students_enrolled_only_in_english (total_students enrolled_both_eng_ger enrolled_ger : ℕ)
  (H1 : total_students = 50)
  (H2 : enrolled_both_eng_ger = 12)
  (H3 : enrolled_ger = 22) :
  ∃ (E : ℕ), E = 50 - (enrolled_ger - enrolled_both_eng_ger) - enrolled_both_eng_ger := 
begin
  use 28,
  sorry -- This is where the proof would go
end

end students_enrolled_only_in_english_l467_467450


namespace calc_abc_squares_l467_467909

theorem calc_abc_squares :
  ∀ (a b c : ℝ),
  a^2 + 3 * b = 14 →
  b^2 + 5 * c = -13 →
  c^2 + 7 * a = -26 →
  a^2 + b^2 + c^2 = 20.75 :=
by
  intros a b c h1 h2 h3
  -- The proof is omitted; reasoning is provided in the solution.
  sorry

end calc_abc_squares_l467_467909


namespace max_siskins_on_poles_l467_467870

theorem max_siskins_on_poles (n : ℕ) (h : n = 25) :
  ∃ k : ℕ, k = 24 ∧ (∀ (poless: Fin n → ℕ) (siskins: Fin n → ℕ),
     (∀ i: Fin n, siskins i ≤ 1) 
     ∧ (∀ i: Fin n, (siskins i = 1 → (poless i = 0)))
     ∧ poless 0 = 0
     → ( ∀ j: Fin n, (j < n → siskins j + siskins (j+1) < 2)) 
     ∧ (k ≤ n)
     ∧ ( ∀ l: Fin n, ((l < k → siskins l = 1) →
       ((k ≤ l < n → siskins l = 0)))))
  sorry

end max_siskins_on_poles_l467_467870


namespace anya_lost_games_correct_l467_467047

-- Definition of players and their game counts
def Players := {Anya, Bella, Valya, Galya, Dasha}
def games_played : Players → ℕ
| Anya   := 4
| Bella  := 6
| Valya  := 7
| Galya  := 10
| Dasha  := 11

-- Total games played by all players
def total_games_played : ℕ := Players.to_finset.sum games_played

-- Total number of games, considering each game involves two players
def total_number_of_games : ℕ := total_games_played / 2

-- The set of games in which Anya lost
def anya_lost_games : set ℕ := {4, 8, 12, 16}

theorem anya_lost_games_correct :
  ∀ i ∈ anya_lost_games, -- For each game in anya_lost_games
  anya_lost_games = {4, 8, 12, 16} := 
by
  sorry

end anya_lost_games_correct_l467_467047


namespace find_n_value_l467_467898

theorem find_n_value :
  ∃ n : ℕ, (3 ≤ n ∧ ∑ k in (range (n - 2) + 3), 1 / ((k : ℕ) * (k + 1)) = 2014 / 6051) ∧ n = 2016 :=
begin
  sorry
end

end find_n_value_l467_467898


namespace anya_lost_games_l467_467035

def girl_count := 5
def anya_games := 4
def bella_games := 6
def valya_games := 7
def galya_games := 10
def dasha_games := 11
def total_games := 19

theorem anya_lost_games :
  (anya_games + bella_games + valya_games + galya_games + dasha_games) / 2 = total_games →
  ∀ games : list ℕ,
    games.length = total_games →
    (∀ g ∈ games, g > 0 ∧ g ≤ total_games) →
    (∀ g ∈ [anya_games, bella_games, valya_games, galya_games, dasha_games], g ≤ total_games) →
    ∀ a b : ℕ, a ≠ b →
    (anya_games, games.nth 3 = some 4) ∧
    (anya_games, games.nth 7 = some 8) ∧
    (anya_games, games.nth 11 = some 12) ∧
    (anya_games, games.nth 15 = some 16)
:= sorry

end anya_lost_games_l467_467035


namespace zack_vs_patrick_ratio_l467_467445

def george_travel : ℕ := 6
def joseph_travel : ℕ := george_travel / 2
def patrick_travel : ℕ := 3 * joseph_travel
def zack_travel : ℕ := 18

theorem zack_vs_patrick_ratio : zack_travel / patrick_travel = 2 :=
by
  have h1 : joseph_travel = george_travel / 2 := by rfl
  have h2 : patrick_travel = 3 * joseph_travel := by rfl
  have h3 : zack_travel = 18 := by rfl
  have ratio_eq : zack_travel / patrick_travel = 18 / (3 * (6 / 2)) := by
    rw [george_travel, h1, h2, h3]
  show 18 / 9 = 2,
  sorry

end zack_vs_patrick_ratio_l467_467445


namespace geometric_sequence_sum_condition_l467_467679

noncomputable def geometric_sequence {a_1 q : ℝ} (n : ℕ) : ℝ :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_sum_condition {a_1 q : ℝ} :
  (geometric_sequence 3 + geometric_sequence 5 = 20) → 
  (geometric_sequence 4 = 8) → 
  geometric_sequence 2 + geometric_sequence 6 = 34 :=
by
  sorry

end geometric_sequence_sum_condition_l467_467679


namespace point_on_line_l467_467690

theorem point_on_line (x_vals : List ℝ) (y_vals : List ℝ) (h_len : x_vals.length = 5) (h_y_len : y_vals.length = 5)
  (h_x_vals : x_vals = [15, 16, 18, 19, 22]) (h_y_vals : y_vals = [102, 98, 115, 115, 120])
  (a b : ℝ) (h_reg:eq : y = λ x, b * x + a) :
  (∃ (x_mean y_mean : ℝ), x_mean = (15 + 16 + 18 + 19 + 22) / 5 ∧ y_mean = (102 + 98 + 115 + 115 + 120) / 5 ∧ x_mean + 18 * y_mean = 110) := 
by 
  sorry

end point_on_line_l467_467690


namespace circumference_to_diameter_ratio_l467_467233

theorem circumference_to_diameter_ratio (C D : ℝ) (hC : C = 94.2) (hD : D = 30) :
  C / D = 3.14 :=
by
  rw [hC, hD]
  norm_num

end circumference_to_diameter_ratio_l467_467233


namespace wechat_payment_meaning_l467_467774

theorem wechat_payment_meaning (initial_balance after_receive_balance : ℝ)
  (recv_amount sent_amount : ℝ)
  (h1 : recv_amount = 200)
  (h2 : initial_balance + recv_amount = after_receive_balance)
  (h3 : after_receive_balance - sent_amount = initial_balance)
  : sent_amount = 200 :=
by
  -- starting the proof becomes irrelevant
  sorry

end wechat_payment_meaning_l467_467774


namespace dots_not_visible_l467_467067

-- Define the sum of numbers on a single die
def sum_die_faces : ℕ := 1 + 2 + 3 + 4 + 5 + 6

-- Define the sum of numbers on four dice
def total_dots_on_four_dice : ℕ := 4 * sum_die_faces

-- List the visible numbers
def visible_numbers : List ℕ := [1, 2, 2, 3, 3, 4, 5, 5, 6]

-- Calculate the sum of visible numbers
def sum_visible_numbers : ℕ := (visible_numbers.sum)

-- Define the math proof problem
theorem dots_not_visible : total_dots_on_four_dice - sum_visible_numbers = 53 := by
  sorry

end dots_not_visible_l467_467067


namespace sum_sequence_bound_l467_467993

noncomputable def sequence (a : ℝ) : ℕ → ℝ
| 0 := a
| (n + 1) := sequence n ^ 2 / (2  * (sequence n - 1))

theorem sum_sequence_bound (a : ℝ) (n : ℕ) (h_a : a > 2) :
  (∑ i in Finset.range n, sequence a i) < 2 * n + 2 * a - 6 + 4 / a :=
sorry

end sum_sequence_bound_l467_467993


namespace polynomial_remainder_l467_467710

-- Define the polynomials
noncomputable def p (z : ℂ) : ℂ := z ^ 2023 + 1
noncomputable def q (z : ℂ) : ℂ := z ^ 2 + z + 1
noncomputable def r (z : ℂ) : ℂ := z + 1

theorem polynomial_remainder :
  ∃ Q : ℂ → ℂ, p = q * Q + r :=
begin
  sorry
end

end polynomial_remainder_l467_467710


namespace distinct_factors_1320_l467_467207

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end distinct_factors_1320_l467_467207


namespace expression_simplifies_to_36_l467_467651

theorem expression_simplifies_to_36 (x : ℝ) : (x + 1)^2 + 2 * (x + 1) * (5 - x) + (5 - x)^2 = 36 :=
by
  sorry

end expression_simplifies_to_36_l467_467651


namespace solution_inequality_equivalence_l467_467015

-- Define the inequality to be proved
def inequality (x : ℝ) : Prop :=
  (x + 1 / 2) * (3 / 2 - x) ≥ 0

-- Define the set of solutions such that -1/2 ≤ x ≤ 3/2
def solution_set (x : ℝ) : Prop :=
  -1 / 2 ≤ x ∧ x ≤ 3 / 2

-- The statement to be proved: the solution set of the inequality is {x | -1/2 ≤ x ≤ 3/2}
theorem solution_inequality_equivalence :
  {x : ℝ | inequality x} = {x : ℝ | solution_set x} :=
by 
  sorry

end solution_inequality_equivalence_l467_467015


namespace cos_expression_range_l467_467102

theorem cos_expression_range (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hSum : A + B + C = Real.pi) :
  -25 / 16 < 3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) ∧ 3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) < 6 :=
sorry

end cos_expression_range_l467_467102


namespace number_must_be_prime_l467_467915

theorem number_must_be_prime (n : ℕ) (h : n ≥ 2) 
  (cond : ∀ (a : Fin n → ℤ), (∑ i, a i) % n ≠ 0 → ∃ i : Fin n, ∀ k ∈ Finset.range n, ∑ j in Finset.range (k + 1), a ((i + j) % n) % n ≠ 0) : n.prime :=
sorry

end number_must_be_prime_l467_467915


namespace tangent_lines_and_properties_l467_467066

open Real 

def parabola : set (ℝ × ℝ) := {p | ∃ x ∈ ℝ, p = (x, (x^2 + 16) / 8)}

def M : ℝ × ℝ := (3, 0)

theorem tangent_lines_and_properties : 
  ∃ f₁ f₂ : ℝ → ℝ, (∀ p ∈ parabola, M ∈ line_through ⟨p.y, f₁ p.x⟩) ∧ 
                     (∀ p ∈ parabola, M ∈ line_through ⟨p.y, f₂ p.x⟩) ∧
                     tangent_line f₁ = λ x, (-1 / 2) * x + 1.5 ∧ 
                     tangent_line f₂ = λ x, 2 * x - 6 ∧ 
                     angle_between_tangents f₁ f₂ = (π / 2) ∧ 
                     area_of_triangle (3, 0) (-2, 2.5) (8, 10) = 125 / 4 := 
sorry

def line_through (p q : ℝ × ℝ) : set (ℝ × ℝ) := { r | (r.fst - p.fst) * (q.snd - p.snd) = (q.fst - p.fst) * (r.snd - p.snd)}

def tangent_line (f: ℝ → ℝ) := λ (x: ℝ), f x

def angle_between_tangents (f₁ f₂: ℝ → ℝ) : ℝ := 
  let m1 := differentiable_at.slope f₁ in
  let m2 := differentiable_at.slope f₂ in
  real.arctan (abs ((m2 - m1) / (1 + m1 * m2)))

def area_of_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (p1.fst * (p2.snd - p3.snd) + p2.fst * (p3.snd - p1.snd) + p3.fst * (p1.snd - p2.snd))

end tangent_lines_and_properties_l467_467066


namespace odd_function_symmetry_l467_467382

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then x^2 else sorry

theorem odd_function_symmetry (x : ℝ) (k : ℕ) (h1 : ∀ y, f (-y) = -f y)
  (h2 : ∀ y, f y = f (2 - y)) (h3 : ∀ y, 0 < y ∧ y ≤ 1 → f y = y^2) :
  k = 45 / 4 → f k = -9 / 16 :=
by
  intros _
  sorry

end odd_function_symmetry_l467_467382


namespace isosceles_triangle_vertex_angle_measure_l467_467396

def isosceles_triangle_vertex_angle (a b h : ℝ) (φ : ℝ) : Prop :=
  2 * a = 3 * b * h

theorem isosceles_triangle_vertex_angle_measure
  (a b h : ℝ) (φ : ℝ)
  (h1 : isosceles_triangle_vertex_angle(a, b, h, φ))
  (acute : φ < 90) :
  φ = 138 :=
sorry

end isosceles_triangle_vertex_angle_measure_l467_467396


namespace general_term_formula_sum_of_sequence_l467_467703

-- Define the geometric sequence and the conditions
def a (n : ℕ) : ℕ := 2 ^ n

-- Main properties of the sequence
axiom a1 : a 1 = 2
axiom a3 : a 3 = a 2 + 4

-- Problem 1: Show the general term formula for {a_n}
theorem general_term_formula : ∀ n, a n = 2 ^ n := by
  sorry

-- Define the sum S_n for the sequence { (2n+1) a_n }
def Sn (n : ℕ) : ℕ := (∑ i in finset.range n, (2 * i + 1) * a i)

-- Problem 2: Calculate the sum of the first n terms Sn for the sequence { (2n+1) a_n }
theorem sum_of_sequence (n : ℕ) : Sn n = (2 * n - 1) * 2 ^ (n + 1) + 2 := by
  sorry

end general_term_formula_sum_of_sequence_l467_467703


namespace CD_value_l467_467289

open Lean

-- Define the elements in the problem
def Point := ℝ × ℝ

def distance (A B : Point) : ℝ := 
  ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2).sqrt

def right_triangle (A B C : Point) : Prop := 
  (A.2 = B.2 ∧ (A.1 ≠ B.1)) ∧ (B.1 = C.1 ∧ (B.2 ≠ C.2))

def diameter_circle_intersect (B C D : Point) : Prop :=
  (B.1 = C.1) ∧ (B.2 ≠ C.2) ∧ (B.1 = D.1)

noncomputable def AD := 2
noncomputable def BD := 3
noncomputable def CD := 4.5

-- Statement to be proven in Lean
theorem CD_value (A B C D : Point) 
  (h1 : right_triangle A B C) 
  (h2 : diameter_circle_intersect B C D)
  (h3 : distance A D = AD) 
  (h4 : distance B D = BD) : (distance C D = CD) := sorry

end CD_value_l467_467289


namespace percent_of_x_is_z_l467_467230

-- Defining the conditions as constants in the Lean environment
variables (x y z : ℝ)

-- Given conditions
def cond1 : Prop := 0.45 * z = 0.90 * y
def cond2 : Prop := y = 0.75 * x

-- The statement of the problem proving z = 1.5 * x under given conditions
theorem percent_of_x_is_z
  (h1 : cond1 z y)
  (h2 : cond2 y x) :
  z = 1.5 * x :=
sorry

end percent_of_x_is_z_l467_467230


namespace f_period_and_max_length_AC_l467_467633

-- Define the given vectors and the parallel condition
def vec_a (x : ℝ) := (1/2, (1/2) * sin x + (sqrt 3 / 2) * cos x)
def vec_b (x : ℝ) := (1, 2 * sin (x + π / 3))

-- Define the function f(x)
def f (x : ℝ) := 2 * sin (x + π / 3)

-- Given conditions
axiom parallel_vectors (x : ℝ) : vec_a x = vec_b x
axiom f_at_value (A : ℝ) : 0 < A ∧ A < π → f (2 * A - π / 6) = 1
axiom BC : ℝ := sqrt 7
axiom sin_B : sin B = sqrt 21 / 7

-- Prove that the minimum positive period of f(x) is 2π and its maximum value is 2
theorem f_period_and_max : 
  (∀ x : ℝ, f (x + 2 * π) = f x) ∧ (∀ x : ℝ, f x ≤ 2) := 
sorry

-- Given the additional conditions, find the length of AC
theorem length_AC (A B C : ℝ) :
  0 < A ∧ A < π ∧ f (2 * A - π / 6) = 1 ∧ 
  BC = sqrt 7 ∧ sin B = sqrt 21 / 7 →
  AC = 2 := 
sorry

end f_period_and_max_length_AC_l467_467633


namespace no_vision_assistance_l467_467412

def students := 40
def percent_glasses := 0.25
def percent_contacts := 0.40

theorem no_vision_assistance :
  students * (1 - percent_glasses - percent_contacts) = 14 :=
by
  sorry

end no_vision_assistance_l467_467412


namespace projection_exists_l467_467604

variable {x y : ℝ}
def vector_b := (x: ℝ, y: ℝ)
def vector_a := (2: ℝ, 0: ℝ)
def vector_c := (1: ℝ, 0: ℝ)

theorem projection_exists : 
  ∃ (b : ℝ × ℝ), let proj_a_b := ((2 * x) / 4) * vector_a in proj_a_b = vector_c :=
by
  let b := (x, y)
  let proj_a_b := ((2 * x) / 4) * (2, 0)
  let vector_c := (1, 0)
  use (1, y)
  sorry

end projection_exists_l467_467604


namespace chord_intersects_inner_circle_l467_467787

noncomputable def probability_chord_intersects_inner_circle
  (r1 r2 : ℝ) (h1 : r1 = 2) (h2 : r2 = 5) : ℝ :=
0.098

theorem chord_intersects_inner_circle :
  probability_chord_intersects_inner_circle 2 5 rfl rfl = 0.098 :=
sorry

end chord_intersects_inner_circle_l467_467787


namespace Anya_loss_games_l467_467019

noncomputable def gamesPlayed : Nat := 19

constant Anya_games : Nat := 4
constant Bella_games : Nat := 6
constant Valya_games : Nat := 7
constant Galya_games : Nat := 10
constant Dasha_games : Nat := 11

constant gameResults : Fin 19 → Option String
constant lost (g : Nat) (p : String) : Prop

axiom Anya_game_indices : ∀ (i : Fin 4), lost (gameResults i.val) "Anya" → i.val = 4 * i + 4

theorem Anya_loss_games : list Nat :=
have Anya_lost_games : list Nat := [4, 8, 12, 16],
Anya_lost_games

end Anya_loss_games_l467_467019


namespace distinct_four_digit_numbers_count_l467_467150

theorem distinct_four_digit_numbers_count (digits : Finset ℕ) (h : digits = {1, 2, 3, 4, 5}) :
  (∃ (numbers : Finset (ℕ × ℕ × ℕ × ℕ)), 
   (∀ (a b c d : ℕ), (a, b, c, d) ∈ numbers → a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
   numbers.card = 120) :=
begin
  sorry
end

end distinct_four_digit_numbers_count_l467_467150


namespace max_siskins_on_poles_l467_467890

-- Define the conditions
def total_poles : ℕ := 25

def adjacent (i j : ℕ) : Prop := (abs (i - j) = 1)

-- Define the main problem
theorem max_siskins_on_poles (h₁ : 0 < total_poles) 
  (h₂ : ∀ (i : ℕ), i ≥ 1 ∧ i ≤ total_poles → ∀ (j : ℕ), j ≥ 1 ∧ j ≤ total_poles ∧ adjacent i j 
    → ¬ (siskin_on i ∧ siskin_on j)) :
  ∃ (max_siskins : ℕ), max_siskins = 24 := 
begin
  sorry
end

end max_siskins_on_poles_l467_467890


namespace find_top_row_number_l467_467767

theorem find_top_row_number (x z : ℕ) (h1 : 8 = x * 2) (h2 : 16 = 2 * z)
  (h3 : 56 = 8 * 7) (h4 : 112 = 16 * 7) : x = 4 :=
by sorry

end find_top_row_number_l467_467767


namespace sequence_term_zero_l467_467282

theorem sequence_term_zero 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 2021 ^ 2021)
  (h2 : ∀ k ≥ 2, 0 ≤ a k ∧ a k < k)
  (h3 : ∀ k ≥ 1, (∑ i in Finset.range k, (-1) ^ (i + 1) * a (i + 1)) % k = 0) :
  a (2021 ^ 2022) = 0 :=
sorry -- Proof placeholder

end sequence_term_zero_l467_467282


namespace positive_integer_pairs_eq_49_l467_467220

/-- Number of positive integer pairs (x, y) such that xy / (x + y) = 1000 is 49 -/
theorem positive_integer_pairs_eq_49 : 
  {p : ℕ × ℕ // p.1 > 0 ∧ p.2 > 0 ∧ (p.1 * p.2) / (p.1 + p.2) = 1000}.card = 49 :=
sorry

end positive_integer_pairs_eq_49_l467_467220


namespace average_of_pqrs_l467_467232

variable (p q r s : ℝ)

theorem average_of_pqrs
  (h : (5 / 4) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 4 :=
by
  sorry

end average_of_pqrs_l467_467232


namespace angle_bisector_slope_of_lines_y_eq_x_and_y_eq_3x_l467_467368

theorem angle_bisector_slope_of_lines_y_eq_x_and_y_eq_3x (k : ℝ) :
  (∀ m1 m2 : ℝ, m1 = 1 ∧ m2 = 3 → k = (1 + Real.sqrt 5) / 2) :=
begin
  intros m1 m2 h,
  cases h with h_m1 h_m2,
  rw h_m1 at *,
  rw h_m2 at *,
  sorry,
end

end angle_bisector_slope_of_lines_y_eq_x_and_y_eq_3x_l467_467368


namespace rationalize_denominator_l467_467342

theorem rationalize_denominator (a b c : ℝ) (h1 : a = 7) (h2 : b = √98) (h3 : √98 = 7 * √2) :
  a / b * √2 = c ↔ c = √2 / 2 := by
  sorry

end rationalize_denominator_l467_467342


namespace point_in_fourth_quadrant_l467_467674

/-- A point in a Cartesian coordinate system -/
structure Point (α : Type) :=
(x : α)
(y : α)

/-- Given a point (4, -3) in the Cartesian plane, prove it lies in the fourth quadrant -/
theorem point_in_fourth_quadrant (P : Point Int) (hx : P.x = 4) (hy : P.y = -3) : 
  P.x > 0 ∧ P.y < 0 :=
by
  sorry

end point_in_fourth_quadrant_l467_467674


namespace integral_value_l467_467947

noncomputable def integral_expression : ℝ :=
  ∫ x in -1..1, (Real.exp (abs x) + Real.sqrt (1 - x^2))

theorem integral_value :
  integral_expression = 2 * (Real.exp(1) - 1) + (Real.pi / 2) :=
  sorry

end integral_value_l467_467947


namespace find_pos_real_nums_l467_467537

theorem find_pos_real_nums (x y z a b c : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z):
  (x + y + z = a + b + c) ∧ (4 * x * y * z = a^2 * x + b^2 * y + c^2 * z + a * b * c) →
  (a = y + z - x ∧ b = z + x - y ∧ c = x + y - z) :=
by
  sorry

end find_pos_real_nums_l467_467537


namespace y_coord_of_vertex_l467_467776

theorem y_coord_of_vertex (f : ℝ → ℝ) :
  (∃ f', f' = (λ x, f(x)^3 - f(x)) ∧ (∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c) ∧
   (∃ x1 x2 x3 : ℝ, f x1 = 0 ∧ f x2 = 1 ∧ f x3 = -1)) →
  ∃ y : ℝ, f (0 : ℝ) = y ∧ y = 0 :=
by
  sorry

end y_coord_of_vertex_l467_467776


namespace sum_of_positive_integers_lcm72_l467_467813

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_of_positive_integers_lcm72 : 
  ∑ ν in {ν | is_solution ν}.to_finset, ν = 180 :=
by 
  sorry

end sum_of_positive_integers_lcm72_l467_467813


namespace cube_root_solution_l467_467954

theorem cube_root_solution (x : ℝ) : (∃ x : ℝ, (∛(5 - x) = -5 / 3)) ↔ x = 260 / 27 :=
by
  sorry

end cube_root_solution_l467_467954


namespace diagonal_of_cube_surface_area_l467_467431

theorem diagonal_of_cube_surface_area (s : ℝ) (d : ℝ) (h : 6 * s^2 = 864) :
  d = s * real.sqrt 3 :=
by
  let s := real.sqrt (864 / 6)
  have : s = 12 := by calc
    s = real.sqrt (864 / 6)           : rfl
    ... = real.sqrt 144              : by norm_num
    ... = 12                         : by norm_num
  let d := s * real.sqrt 3
  show d = 12 * real.sqrt 3
  sorry

end diagonal_of_cube_surface_area_l467_467431


namespace washing_machines_removed_correct_l467_467249

-- Define the conditions
def crates : ℕ := 10
def boxes_per_crate : ℕ := 6
def washing_machines_per_box : ℕ := 4
def washing_machines_removed_per_box : ℕ := 1

-- Define the initial and final states
def initial_washing_machines_in_crate : ℕ := boxes_per_crate * washing_machines_per_box
def initial_washing_machines_in_container : ℕ := crates * initial_washing_machines_in_crate

def final_washing_machines_in_box : ℕ := washing_machines_per_box - washing_machines_removed_per_box
def final_washing_machines_in_crate : ℕ := boxes_per_crate * final_washing_machines_in_box
def final_washing_machines_in_container : ℕ := crates * final_washing_machines_in_crate

-- Number of washing machines removed
def washing_machines_removed : ℕ := initial_washing_machines_in_container - final_washing_machines_in_container

-- Theorem statement in Lean 4
theorem washing_machines_removed_correct : washing_machines_removed = 60 := by
  sorry

end washing_machines_removed_correct_l467_467249


namespace two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n_l467_467323

theorem two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n
  (n : ℕ) (h : 2 < n) : (2 * n - 1) ^ n + (2 * n) ^ n < (2 * n + 1) ^ n :=
sorry

end two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n_l467_467323


namespace correct_description_l467_467495

variable (A : Prop)
variable (B : Prop)
variable (C : Prop)
variable (D : Prop)
variable (answer : Prop)

-- Define the biological descriptions
def description_A := when isolating microorganisms that decompose organic matter from organic wastewater, the inoculated petri dishes must be cultured in a light incubator.

def description_B := in the process of wine fermentation, open the bottle cap every 12 hours or so to release CO₂.

def description_C := when observing cells under a microscope, if the cells are colorless and transparent, the aperture for light transmission can be adjusted to make the field of view darker.

def description_D := If the dilution spread plate method is used to count the number of viable E. coli bacteria, in addition to strict operation and repeated experiments, the dilution degree of the sample to be tested should also be ensured.

-- Problem statement
theorem correct_description : description_D := 
sorry

end correct_description_l467_467495


namespace max_cubes_fit_in_box_l467_467432

theorem max_cubes_fit_in_box :
  ∀ (h w l : ℕ) (cube_vol box_max_cubes : ℕ),
    h = 12 → w = 8 → l = 9 → cube_vol = 27 → 
    box_max_cubes = (h * w * l) / cube_vol → box_max_cubes = 32 :=
by
  intros h w l cube_vol box_max_cubes h_def w_def l_def cube_vol_def box_max_cubes_def
  sorry

end max_cubes_fit_in_box_l467_467432


namespace trig_identity_example_l467_467904

theorem trig_identity_example :
  cos (75 * Real.pi / 180) * cos (15 * Real.pi / 180) - sin (75 * Real.pi / 180) * sin (15 * Real.pi / 180) = 0 :=
by
  sorry

end trig_identity_example_l467_467904


namespace oxygen_mass_required_l467_467796

theorem oxygen_mass_required (n_H2O : ℕ) (molar_mass_O2 : ℝ) 
  (h_ratio : 2 * (n_H2O / 2) = n_H2O / 2 * 1) : n_H2O = 7 → molar_mass_O2 = 32 → 
  (7 / 2) * 32 = 112 :=
begin
  intros h_n_H2O h_molar_mass_O2,
  rw h_n_H2O,
  rw h_molar_mass_O2,
  norm_num
end

end oxygen_mass_required_l467_467796


namespace find_focus_l467_467899

-- Define the parabola with vertex at O and focus F
def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {P | ∃ t, P = (2 * p * t^2, 2 * p * t)}

-- Define the condition for point Q
def tangent_and_distance (p t : ℝ) : set (ℝ × ℝ) :=
  let P := (2 * p * t^2, 2 * p * t)
  let Q := (-2 * p * t^2, 0)
  abs (2 * p * t - (-2 * p * t)) = 2

-- Define sums of areas of circles C1 and C2
def circles_areas (p t : ℝ) : ℝ :=
  let θ := Real.arctan (1 / t)
  let r1 := 2 * p * t / (1 - Real.cos θ)
  let r2 := 2 * p * t / (1 + Real.cos θ)
  r1^2 + r2^2

-- Prove the minimization condition
theorem find_focus : ∀ {p t : ℝ},
  tangent_and_distance p t →
  ∃ F, F = (1 / Real.sqrt(3 - Real.sqrt 3), 0) ∧
  ∀ p t, circles_areas p t = circles_areas p t → 
  circles_areas p t ∧ sorry := sorry

end find_focus_l467_467899


namespace anya_lost_games_correct_l467_467045

-- Definition of players and their game counts
def Players := {Anya, Bella, Valya, Galya, Dasha}
def games_played : Players → ℕ
| Anya   := 4
| Bella  := 6
| Valya  := 7
| Galya  := 10
| Dasha  := 11

-- Total games played by all players
def total_games_played : ℕ := Players.to_finset.sum games_played

-- Total number of games, considering each game involves two players
def total_number_of_games : ℕ := total_games_played / 2

-- The set of games in which Anya lost
def anya_lost_games : set ℕ := {4, 8, 12, 16}

theorem anya_lost_games_correct :
  ∀ i ∈ anya_lost_games, -- For each game in anya_lost_games
  anya_lost_games = {4, 8, 12, 16} := 
by
  sorry

end anya_lost_games_correct_l467_467045


namespace max_value_f_in_interval_l467_467965

def f (x : ℝ) : ℝ := 6 * x / (1 + x^2)

theorem max_value_f_in_interval :
  ∃ x ∈ set.Icc 0 3, ∀ y ∈ set.Icc (0 : ℝ) 3, f y ≤ 3 ∧ f x = 3 :=
by
  sorry

end max_value_f_in_interval_l467_467965


namespace composition_computation_l467_467835

noncomputable def linear_function (p q : ℝ) : ℝ → ℝ := λ x, p * x + q

/--
Given 1000 linear functions of the form f_k(x) = p_k x + q_k for k = 1, 2, ..., 1000,
we can compute the value of their composition at a point x_0 in no more than 30 stages.
-/
theorem composition_computation (p : fin 1000 → ℝ) (q : fin 1000 → ℝ) (x₀ : ℝ) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = (λ x, foldr (λ k acc, linear_function (p k) (q k) acc) x (fin_range 1000)) x) ∧
  (∑' j in finset.range 1000, ∏ k in finset.range j, p k * q j ≤ 30) :=
sorry

end composition_computation_l467_467835


namespace valid_arrangements_unique_l467_467418

theorem valid_arrangements_unique :
  ∃! g : ℕ → ℕ → char, 
    (g 0 0 = 'A') ∧ (g 1 1 = 'B') ∧
    (∀ i j, i ≠ j → (g i j = 'A' ∨ g i j = 'B' ∨ g i j = 'C')) ∧
    (∀ i, MULTISET (g 0 i) = MULTISET (g 1 i) = MULTISET (g 2 i)) ∧
    (∀ j, MULTISET (g j 0) = MULTISET (g j 1) = MULTISET (g j 2)) :=
sorry

end valid_arrangements_unique_l467_467418


namespace solve_equation_l467_467738

theorem solve_equation : ∀ (x : ℝ), x ≠ 2 → -2 * x^2 = (4 * x + 2) / (x - 2) → x = 1 :=
by
  intros x hx h_eq
  sorry

end solve_equation_l467_467738


namespace constant_term_in_expansion_l467_467003

theorem constant_term_in_expansion {x : ℝ} (h : x ≠ 0) :
  let e := (1 / x - 1) * (sqrt x + 1) ^ 5 in
  constant_term e = 10 :=
by
  sorry

end constant_term_in_expansion_l467_467003


namespace range_of_a_l467_467087

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a ≤ x ∧ x ≤ a^2 + 1) → (x^2 - 3 * (a + 1) * x + 2 * (3 * a + 1) ≤ 0)) ↔ (1 ≤ a ∧ a ≤ 3 ∨ a = -1) :=
by
  sorry

end range_of_a_l467_467087


namespace abc_sum_equals_9_l467_467227

theorem abc_sum_equals_9 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a * b + c = 57) (h5 : b * c + a = 57) (h6 : a * c + b = 57) :
  a + b + c = 9 := 
sorry

end abc_sum_equals_9_l467_467227


namespace monotonic_intervals_range_of_m_three_tangent_lines_l467_467615

-- Problem 1
def f (x : ℝ) (a : ℝ) : ℝ := x^3 - 3 * a * x - 1

-- Monotonic intervals
theorem monotonic_intervals (a : ℝ) (h : a ≠ 0) :
  if a < 0 then
    (∀ x y : ℝ, x < y → f x a < f y a)
  else if a > 0 then
    (∀ x ∈ set.Icc (-real.sqrt a) (real.sqrt a), f' x a < 0) ∧
    (∀ x ∈ set.Ici (real.sqrt a) ∨ x ∈ set.Iic (-real.sqrt a), f' x a > 0)
  else
    false :=
sorry

-- Problem 2
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x = -1 ∧ f' x 1 = 0) →
  (∀ x : ℝ, (f x 1 - m) = 0) →
  -3 < m ∧ m < 1 :=
sorry

-- Problem 3
def h (x : ℝ) (a : ℝ) : ℝ := f x a + (3 * a - 1) * x + 1

theorem three_tangent_lines (a : ℝ) (h : a ≠ 0) : 
  (∃ t : ℝ, (λ t, 2 * t^3 - 6 * t^2 + 3) = 0) → 
  ∃ P : ℝ × ℝ, P.1 = 2 ∧ P.2 = 1 ∧ (λ t, 2 * t^3 - 6 * t^2 + 3) = 0 :=
sorry

end monotonic_intervals_range_of_m_three_tangent_lines_l467_467615


namespace tennis_balls_in_each_container_l467_467276

theorem tennis_balls_in_each_container (initial_balls : ℕ) (half_gone : ℕ) (remaining_balls : ℕ) (containers : ℕ) 
  (h1 : initial_balls = 100) 
  (h2 : half_gone = initial_balls / 2)
  (h3 : remaining_balls = initial_balls - half_gone)
  (h4 : containers = 5) :
  remaining_balls / containers = 10 := 
by
  sorry

end tennis_balls_in_each_container_l467_467276


namespace min_students_solving_most_l467_467408

theorem min_students_solving_most (students problems : Nat) 
    (total_students : students = 10) 
    (problems_per_student : Nat → Nat) 
    (problems_per_student_property : ∀ s, s < students → problems_per_student s = 3) 
    (common_problem : ∀ s1 s2, s1 < students → s2 < students → s1 ≠ s2 → ∃ p, p < problems ∧ (∃ (solves1 solves2 : Nat → Nat), (solves1 p = 1 ∧ solves2 p = 1) ∧ s1 < students ∧ s2 < students)): 
  ∃ min_students, min_students = 5 :=
by
  sorry

end min_students_solving_most_l467_467408


namespace cannot_determine_relationship_l467_467646

variable (x y : ℝ)

theorem cannot_determine_relationship (h : exp (-x) + log y < exp (-y) + log x) : 
  ¬(x > y ∧ y > 0) ∧ ¬(y > x ∧ x > 0) ∧ ¬(0 > x ∧ x > y) :=
by
  sorry

end cannot_determine_relationship_l467_467646


namespace parker_savings_l467_467505

-- Define the costs of individual items and meals
def burger_cost : ℝ := 5
def fries_cost : ℝ := 3
def drink_cost : ℝ := 3
def special_meal_cost : ℝ := 9.5
def kids_burger_cost : ℝ := 3
def kids_fries_cost : ℝ := 2
def kids_drink_cost : ℝ := 2
def kids_meal_cost : ℝ := 5

-- Define the number of meals Mr. Parker buys
def adult_meals : ℕ := 2
def kids_meals : ℕ := 2

-- Define the total cost of individual items for adults and children
def total_individual_cost_adults : ℝ :=
  adult_meals * (burger_cost + fries_cost + drink_cost)

def total_individual_cost_children : ℝ :=
  kids_meals * (kids_burger_cost + kids_fries_cost + kids_drink_cost)

-- Define the total cost of meal deals
def total_meals_cost : ℝ :=
  adult_meals * special_meal_cost + kids_meals * kids_meal_cost

-- Define the total cost of individual items for both adults and children
def total_individual_cost : ℝ :=
  total_individual_cost_adults + total_individual_cost_children

-- Define the savings
def savings : ℝ := total_individual_cost - total_meals_cost

theorem parker_savings : savings = 7 :=
by
  sorry

end parker_savings_l467_467505


namespace find_x_values_l467_467538

noncomputable def tan_inv := Real.arctan (Real.sqrt 3 / 2)

theorem find_x_values (x : ℝ) :
  (-Real.pi < x ∧ x ≤ Real.pi) ∧ (2 * Real.tan x - Real.sqrt 3 = 0) ↔
  (x = tan_inv ∨ x = tan_inv - Real.pi) :=
by
  sorry

end find_x_values_l467_467538


namespace distinct_factors_1320_l467_467210

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end distinct_factors_1320_l467_467210


namespace anya_lost_games_l467_467055

-- Define the girls playing table tennis
def girls : Type := {Anya Bella Valya Galya Dasha : girls}

-- Define the conditions
def games_played : girls → ℕ
| Anya => 4
| Bella => 6
| Valya => 7
| Galya => 10
| Dasha => 11

-- Total number of games played
def total_games : ℕ := 19
#eval total_games = (games_played Anya + games_played Bella + games_played Valya + games_played Galya + games_played Dasha) / 2

-- The main theorem to be proven
theorem anya_lost_games :
  ∀ i : ℕ, i ∈ {4, 8, 12, 16} ↔ Anya played and lost in game i
:=
by
  sorry

end anya_lost_games_l467_467055


namespace sum_lcm_eq_72_l467_467822

theorem sum_lcm_eq_72 (s : Finset ℕ) 
  (h1 : ∀ ν ∈ s, Nat.lcm ν 24 = 72) :
  s.sum id = 180 :=
by
  have h2 : ∀ ν, ν ∈ s → ∃ n, ν = 3 * n := 
    by sorry
  have h3 : ∀ n, ∃ ν, ν = 3 * n ∧ ν ∈ s := 
    by sorry
  sorry

end sum_lcm_eq_72_l467_467822


namespace math_problem_l467_467009

theorem math_problem
  (a b c d : ℕ)
  (h1 : a = 234)
  (h2 : b = 205)
  (h3 : c = 86400)
  (h4 : d = 300) :
  (a * b = 47970) ∧ (c / d = 288) :=
by
  sorry

end math_problem_l467_467009


namespace exists_x_with_minimal_period_gt_l467_467975

def recurrence_relation (x : ℝ) : ℕ → ℝ → ℝ
| 0, a₀   := a₀
| (n+1), aₙ := x - (1 / aₙ)

-- Definition of minimal period
def is_minimal_period (x : ℝ) (p : ℕ) : Prop :=
  (∀ a₀ : ℝ, periodic (recurrence_relation x p a₀) ∧
  (∀ q : ℕ, (0 < q ∧ q < p) → ∃ a₀ : ℝ, ¬ periodic (recurrence_relation x q a₀)))

-- Main theorem
theorem exists_x_with_minimal_period_gt (P : ℕ) (hP : 0 < P) : 
  ∃ x : ℝ, ∃ p : ℕ, p > P ∧ is_minimal_period x p :=
sorry

end exists_x_with_minimal_period_gt_l467_467975


namespace isosceles_triangle_perimeter_l467_467996

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : a ≠ b) :
  ∃ (c : ℕ), c = b ∧
              ((a + a > b ∧ a + b > a ∧ a + b > a ∧ 2 * a + b = c) ↔ c = 29) := 
begin
  sorry
end

end isosceles_triangle_perimeter_l467_467996


namespace distinct_factors_1320_l467_467216

theorem distinct_factors_1320 : 
  ∃ n : ℕ, n = 24 ∧ 
  let a := 2 in 
  let b := 1 in 
  let c := 1 in 
  let d := 1 in 
  1320 = 2^a * 3^b * 5^c * 11^d :=
begin
  sorry,
end

end distinct_factors_1320_l467_467216


namespace complex_expression_evaluation_l467_467746

-- Definition of the imaginary unit i with property i^2 = -1
def i : ℂ := Complex.I

-- Theorem stating that the given expression equals i
theorem complex_expression_evaluation : i * (1 - i) - 1 = i := by
  -- Proof omitted
  sorry

end complex_expression_evaluation_l467_467746


namespace num_distinct_factors_1320_l467_467163

theorem num_distinct_factors_1320 : 
  ∃ n : ℕ, n = 1320 ∧ (finset.univ.filter (λ d, d > 0 ∧ 1320 % d = 0)).card = 32 := 
sorry

end num_distinct_factors_1320_l467_467163


namespace domain_f_x_plus_2_l467_467616

-- Define the function f and its properties
variable (f : ℝ → ℝ)

-- Define the given condition: the domain of y = f(2x - 3) is [-2, 3]
def domain_f_2x_minus_3 : Set ℝ :=
  {x | -2 ≤ x ∧ x ≤ 3}

-- Express this condition formally
axiom domain_f_2x_minus_3_axiom :
  ∀ (x : ℝ), (x ∈ domain_f_2x_minus_3) → (2 * x - 3 ∈ Set.Icc (-7 : ℝ) 3)

-- Prove the desired result: the domain of y = f(x + 2) is [-9, 1]
theorem domain_f_x_plus_2 :
  ∀ (x : ℝ), (x ∈ Set.Icc (-9 : ℝ) 1) ↔ ((x + 2) ∈ Set.Icc (-7 : ℝ) 3) :=
sorry

end domain_f_x_plus_2_l467_467616


namespace distinct_positive_factors_of_1320_l467_467197

theorem distinct_positive_factors_of_1320 : 
  ∀ n ≥ 0, prime_factorization 1320 = [(2, 3), (3, 1), (11, 1), (5, 1)] → factors_count 1320 = 32 :=
by
  intro n hn h
  sorry

end distinct_positive_factors_of_1320_l467_467197


namespace find_phi_l467_467120

noncomputable def sine_shift_symm (φ : ℝ) : Prop :=
  let f := λ x : ℝ, Real.sin (3 * x + φ)
  0 < φ ∧ φ < Real.pi ∧
  (∀ x : ℝ, Real.sin (3 * (x - Real.pi / 12) + φ) = Real.sin (3 * x + φ - Real.pi / 4)) ∧
  (∀ x : ℝ, Real.sin (3 * x + φ - Real.pi / 4) = Real.sin (3 * x - (φ - Real.pi / 4))) →
  φ = 3 * Real.pi / 4

theorem find_phi : ∃ φ : ℝ, sine_shift_symm φ := sorry

end find_phi_l467_467120


namespace distinct_four_digit_numbers_count_l467_467148

theorem distinct_four_digit_numbers_count (digits : Finset ℕ) (h : digits = {1, 2, 3, 4, 5}) :
  (∃ (numbers : Finset (ℕ × ℕ × ℕ × ℕ)), 
   (∀ (a b c d : ℕ), (a, b, c, d) ∈ numbers → a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
   numbers.card = 120) :=
begin
  sorry
end

end distinct_four_digit_numbers_count_l467_467148


namespace tetrahedron_planes_intersect_at_sphere_center_l467_467859

-- A definition for the four planes and their intersection
theorem tetrahedron_planes_intersect_at_sphere_center 
  (T : Tetrahedron) 
  (centerCircumsphere : Point)
  (centerCircumcircle : ∀ (face : Face T), Point) : 
  (through_vertex (vertex : Vertex T) (face : OppositeFace vertex T) : Plane) := 
  ∀ (vertex : Vertex T),
  ∃! (intersection_point : Point),
  is_center_of_circumsphere (T) (intersection_point) :=
begin
  sorry
end

end tetrahedron_planes_intersect_at_sphere_center_l467_467859


namespace minimum_value_exp_function_l467_467992

theorem minimum_value_exp_function (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hab : a + b = a⁻¹ + b⁻¹) : 3^a + 81^b = 18 :=
sorry

end minimum_value_exp_function_l467_467992


namespace minimize_pollution_park_distance_l467_467498

noncomputable def pollution_index (x : ℝ) : ℝ :=
  (1 / x) + (4 / (30 - x))

theorem minimize_pollution_park_distance : ∃ x : ℝ, (0 < x ∧ x < 30) ∧ pollution_index x = 10 :=
by
  sorry

end minimize_pollution_park_distance_l467_467498


namespace decreasing_interval_of_y_l467_467384

def t (x : ℝ) : ℝ := -x^2 - 2*x + 8

noncomputable def y (x : ℝ) : ℝ := Real.log (t x)

theorem decreasing_interval_of_y :
  ∀ x : ℝ, -4 < x ∧ x < 2 → 
  differentiable_on ℝ y (Set.Ioo (-1 : ℝ) (2 : ℝ)) →
  ∀ x1 x2 : ℝ, -1 < x1 ∧ x2 < 2 ∧ x1 < x2 → 
  y x1 > y x2 := 
sorry

end decreasing_interval_of_y_l467_467384


namespace area_triangle_BOC_l467_467688

theorem area_triangle_BOC
    (A B C O K : Type)
    [HasDist A B] [HasDist A C] [HasDist B C] [HasDist A K] [HasDist K C]
    (h1 : dist A C = 14)
    (h2 : dist A B = 6)
    (h3 : midpoint O A C)
    (h4 : angle A K C = 90)
    (h5 : angle B A K = angle A C B) :
  area (triangle B O C) = 21 :=
sorry

end area_triangle_BOC_l467_467688


namespace S_formula_l467_467837

variable {n k : ℕ}

def S (n k : ℕ) : ℕ := 
  ∑ i in (range (k + 1)).filter (λ x, x % 2 = (1 : ℕ) % 2), (binom (k + 1) (i + 1) * S_i n)

theorem S_formula (n k : ℕ) : 
  S n k = (n + 1)^(k + 1) + n^(k + 1) - 1 ÷ 2 := 
sorry

end S_formula_l467_467837


namespace incorrect_statement_C_l467_467976

/-- 
  Prove that the function y = -1/2 * x + 3 does not intersect the y-axis at (6,0).
-/
theorem incorrect_statement_C 
: ∀ (x y : ℝ), y = -1/2 * x + 3 → (x, y) ≠ (6, 0) :=
by
  intros x y h
  sorry

end incorrect_statement_C_l467_467976


namespace find_unit_prices_minimal_cost_l467_467670

-- Definitions for part 1
def unitPrices (x y : ℕ) : Prop :=
  20 * x + 30 * y = 2920 ∧ x - y = 11 

-- Definitions for part 2
def costFunction (m : ℕ) : ℕ :=
  52 * m + 48 * (40 - m)

def additionalPurchase (m : ℕ) : Prop :=
  m ≥ 40 / 3

-- Statement for unit prices proof
theorem find_unit_prices (x y : ℕ) (h1 : 20 * x + 30 * y = 2920) (h2 : x - y = 11) : x = 65 ∧ y = 54 := 
  sorry

-- Statement for minimal cost proof
theorem minimal_cost (m : ℕ) (x y : ℕ) 
  (hx : 20 * x + 30 * y = 2920) 
  (hy : x - y = 11)
  (hx_65 : x = 65)
  (hy_54 : y = 54)
  (hm : m ≥ 40 / 3) : 
  costFunction m = 1976 ∧ m = 14 :=
  sorry

end find_unit_prices_minimal_cost_l467_467670


namespace number_of_correct_statements_l467_467111

def trigonometric_values_terminal_side_equal (α β : ℝ) (h : α = β) : sin α = sin β ∧ cos α = cos β := sorry

def sin_equal_in_triangle (A B : ℝ) (h : sin A = sin B) : A = B := sorry

def angle_measurement_independent_of_radius (A : ℝ) (r₁ r₂ : ℝ) (h : r₁ ≠ 0 ∧ r₂ ≠ 0) : (real.angle.toRadians A / r₁) = (real.angle.toRadians A / r₂) := sorry

def sin_equal_terminal_sides (α β : ℝ) (h : sin α = sin β) : real.angle.covers_same_sector α β ∨ real.angle.symmetric_about_y_axis α β := sorry

def cosine_negative_quadrant (θ : ℝ) (h : cos θ < 0) : (π / 2 < θ ∧ θ < π) ∨ (π < θ ∧ θ < (3 * π) / 2) := sorry

theorem number_of_correct_statements : (trigonometric_values_terminal_side_equal ∧ sin_equal_in_triangle ∧ angle_measurement_independent_of_radius) ∧ ¬sin_equal_terminal_sides ∧ ¬cosine_negative_quadrant → 3 = 3 :=
by
  sorry

end number_of_correct_statements_l467_467111


namespace value_of_x_l467_467400

variable (x y z : ℝ)

-- Conditions based on the problem statement
def condition1 := x = (1 / 3) * y
def condition2 := y = (1 / 4) * z
def condition3 := z = 96

-- The theorem to be proven
theorem value_of_x (cond1 : condition1) (cond2 : condition2) (cond3 : condition3) : x = 8 := 
by
  sorry

end value_of_x_l467_467400


namespace liz_fraction_of_total_money_l467_467131

theorem liz_fraction_of_total_money
  (hanna_money : ℝ) (joi_money : ℝ) (ken_money : ℝ)
  (liz_money : ℝ)
  (hanna_gave : hanna_money * (1 / 6) = liz_money)
  (joi_gave : joi_money * (1 / 5) = liz_money)
  (ken_gave : ken_money * (1 / 4) = liz_money) :
  let total_money := hanna_money + joi_money + ken_money
  in liz_money * 3 = (1 / 5) * total_money :=
by sorry

end liz_fraction_of_total_money_l467_467131


namespace kim_total_points_l467_467755

noncomputable def total_points 
    (easy_points : ℕ)
    (avg_points : ℕ)
    (hard_points : ℕ)
    (exp_points : ℕ)
    (easy_correct : ℕ)
    (avg_correct : ℕ)
    (hard_correct : ℕ)
    (exp_correct : ℕ)
    (bonus_points : ℕ)
    (bonus_count : ℕ) : ℕ :=
  easy_points * easy_correct + avg_points * avg_correct + hard_points * hard_correct + exp_points * exp_correct + bonus_points * bonus_count

theorem kim_total_points : 
  total_points 2 3 5 7 6 2 4 3 1 2 = 61 :=
by 
  simp [total_points]
  rfl

#eval kim_total_points

end kim_total_points_l467_467755


namespace max_siskins_on_poles_l467_467874

theorem max_siskins_on_poles : 
    (∀ (poles : List Nat), length poles = 25 → 
        ∀ (siskins_on_poles : List (Option Nat)), 
        length siskins_on_poles = 25 → 
        (∀ (i : Nat), (i < 25) → (∀ (j : Nat), (abs (j - i) = 1) → 
          (siskins_on_poles.nth i = some _) → (siskins_on_poles.nth j ≠ some _))) → 
        (∃ k, ∀ n, (n < 25) → 
            n + k ≤ 24 → 
            ∃ m, (siskins_on_poles.nth m = some m) ∧ 
              (m < 25) ∧ 
              (∀ l, (m ≠ l) → (siskins_on_poles.nth l ≠ some _)))) → true) := 
by sorry

end max_siskins_on_poles_l467_467874


namespace condition_p_neither_sufficient_nor_necessary_l467_467088

theorem condition_p_neither_sufficient_nor_necessary
  (x : ℝ) :
  (1/x ≤ 1 → x^2 - 2 * x ≥ 0) = false ∧ 
  (x^2 - 2 * x ≥ 0 → 1/x ≤ 1) = false := 
by 
  sorry

end condition_p_neither_sufficient_nor_necessary_l467_467088


namespace euler_form_example_l467_467519

-- Definitions used directly from the conditions
def euler_formula (x : ℝ) : ℂ := complex.exp (complex.I * x)

theorem euler_form_example : euler_formula (13 * real.pi / 2) = complex.I :=
by
  sorry

end euler_form_example_l467_467519


namespace midpoint_is_correct_l467_467523

-- Defining the points in 3D space.
def point1 : ℝ × ℝ × ℝ := (3, 7, 2)
def point2 : ℝ × ℝ × ℝ := (5, 1, 6)

-- The function to calculate the midpoint in 3D space.
def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

-- The statement we want to prove.
theorem midpoint_is_correct :
  midpoint point1 point2 = (4, 4, 4) :=
by
  sorry

end midpoint_is_correct_l467_467523


namespace can_capacity_l467_467449

/-- Given a can with a mixture of milk and water in the ratio 4:3, and adding 10 liters of milk
results in the can being full and changes the ratio to 5:2, prove that the capacity of the can is 30 liters. -/
theorem can_capacity (x : ℚ)
  (h1 : 4 * x + 3 * x + 10 = 30)
  (h2 : (4 * x + 10) / (3 * x) = 5 / 2) :
  4 * x + 3 * x + 10 = 30 := 
by sorry

end can_capacity_l467_467449


namespace area_of_triangle_integer_sides_l467_467625

def sequence_t : ℕ → ℝ
| 0     := 0
| 1     := 6
| (n+2) := 14 * sequence_t (n+1) - sequence_t n

theorem area_of_triangle_integer_sides (n : ℕ) (hn : n ≥ 1) : 
  ∃ (a : ℕ), 
    sequence_t n = (a : ℝ) / 4 * Real.sqrt (3 * (a : ℝ) ^ 2 - 12) ∧ 
    Int.has_repr (a - 1) ∧ Int.has_repr a ∧ Int.has_repr (a + 1) := 
sorry

end area_of_triangle_integer_sides_l467_467625


namespace sym_circle_equal_diagonals_l467_467594

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 6 * x = 0
def line_eq (x y : ℝ) : Prop := y = 2 * x + 1
def symmetric_circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9
def line_passing_eq (x y : ℝ) : Prop := y = x + 1

theorem sym_circle 
  (x y : ℝ)
  (circle_eq x y)
  (line_eq x y)
  : symmetric_circle_eq x y := 
sorry

theorem equal_diagonals 
  (x1 y1 x2 y2 : ℝ)
  (hx1 : x1 = x2 + 1)
  (hy1 : y1 = y2 + 1)
  (hx2 : symmetric_circle_eq x1 y1)
  (hy2 : symmetric_circle_eq x2 y2)
  (h_diag : x1 * x2 + y1 * y2 = 0)
  : line_passing_eq x1 y1 := 
sorry

end sym_circle_equal_diagonals_l467_467594


namespace bobs_password_probability_l467_467510

theorem bobs_password_probability :
  let even_digits := {0, 2, 4, 6, 8}
  let positive_digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let even_positive_digits := {2, 4, 6, 8}
  -- Probabilities
  let probability_first_digit_even := (5 : ℚ) / 10
  let probability_second_char_letter := (1 : ℚ)
  let probability_last_digit_even_positive := (4 : ℚ) / 9
  -- Total probability
  let total_probability := probability_first_digit_even * probability_second_char_letter * probability_last_digit_even_positive
  in total_probability = (2 : ℚ) / 9 :=
by
  sorry

end bobs_password_probability_l467_467510


namespace sin_alpha_minus_pi_over_6_l467_467093

theorem sin_alpha_minus_pi_over_6 (α : ℝ) 
  (h : cos (α - π / 3) - cos α = 1 / 3) : 
  sin (α - π / 6) = 1 / 3 :=
sorry

end sin_alpha_minus_pi_over_6_l467_467093


namespace anya_lost_games_l467_467040

/-- 
Five girls (Anya, Bella, Valya, Galya, Dasha) played several games of table tennis. At any given time, two girls were playing,
and the other three were resting. The girl who lost the game went to rest, and her place was taken by the girl who had rested 
the most rounds. There are no ties in table tennis. Anya played 4 games, Bella played 6 games, Valya played 7 games, Galya 
played 10 games, and Dasha played 11 games.

We need to prove that the numbers of all games in which Anya lost are 4, 8, 12, and 16.
-/
theorem anya_lost_games :
  (∃ games : list ℕ,
    games.length = 19 ∧
    (∀ i, i ∈ games → i = 4 ∨ i = 8 ∨ i = 12 ∨ i = 16) ∧
    (∀ i, i ∉ games → true)) :=
by
  sorry

end anya_lost_games_l467_467040


namespace find_a_l467_467575

-- Conditions
variables (a : ℝ)
def imaginary_unit := complex.I
def given_condition := (1 + a * imaginary_unit) * imaginary_unit = 3 + imaginary_unit

-- Theorem Statement
theorem find_a (h : given_condition) : a = -3 :=
sorry

end find_a_l467_467575


namespace quadratic_function_properties_l467_467080

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x

theorem quadratic_function_properties :
  (∀ x, f (-1 + x) = f (-1 - x)) ∧
  (∃ x₁ x₂, f x₁ = 0 ∧ f x₂ = 0 ∧ |x₁ - x₂| = 2) ∧
  (∀ k, ∀ x ∈ Set.Icc (-1) 2, g'(x, k) = g'(-1, k) ∨ g'(x, k) = g'(2, k) → k ≤ 0) :=
by
  sorry

def g (x k : ℝ) : ℝ := f x - k * x

noncomputable def g'(x k : ℝ) : ℝ := (x - (k - 2) / 2)^2 - (k - 2)^2 / 4

end quadratic_function_properties_l467_467080


namespace no_vision_assistance_l467_467410

theorem no_vision_assistance (total_students : ℕ) (wears_glasses : ℕ) (wears_contacts : ℕ)
    (h1 : total_students = 40)
    (h2 : wears_glasses = 0.25 * total_students) 
    (h3 : wears_contacts = 0.40 * total_students) :
    total_students - (wears_glasses + wears_contacts) = 14 := by
  sorry

end no_vision_assistance_l467_467410


namespace num_distinct_factors_1320_l467_467162

theorem num_distinct_factors_1320 : 
  ∃ n : ℕ, n = 1320 ∧ (finset.univ.filter (λ d, d > 0 ∧ 1320 % d = 0)).card = 32 := 
sorry

end num_distinct_factors_1320_l467_467162


namespace sum_of_positive_integers_nu_lcm_72_l467_467808

theorem sum_of_positive_integers_nu_lcm_72:
  let ν_values := { ν | Nat.lcm ν 24 = 72 }
  ∑ ν in ν_values, ν = 180 := by
  sorry

end sum_of_positive_integers_nu_lcm_72_l467_467808


namespace inequality_solution_l467_467943

theorem inequality_solution {x : ℝ} : 5 * x^2 + 7 * x > 3 ↔ x < -1 ∨ x > 3/5 := by
  sorry

end inequality_solution_l467_467943


namespace point_on_line_l467_467656

theorem point_on_line (x : ℝ) :
  (∃ x, (x, -6) ∈ line_through (0, 10) (-8, 0)) → x = -64 / 5 :=
by
  -- Definitions for the line_through function and point containment can be given here.
  sorry

end point_on_line_l467_467656


namespace solve_for_a_l467_467707

variable (a : ℝ)
def z1 : ℂ := (1 : ℂ) - (2 : ℂ) * Complex.i
def z2 : ℂ := (a : ℂ) + Complex.i

theorem solve_for_a 
  (H : (z1 * z2).re = 0) 
  (H_im : (z1 * z2).im ≠ 0) : 
  a = -2 := 
  sorry

end solve_for_a_l467_467707


namespace summation_series_equals_half_l467_467940

theorem summation_series_equals_half :
  (\sum_{n=0}^{∞} 2^n / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end summation_series_equals_half_l467_467940


namespace count_1320_factors_l467_467168

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end count_1320_factors_l467_467168


namespace triangular_pyramid_nonexistence_l467_467274

theorem triangular_pyramid_nonexistence
    (h : ℕ)
    (hb : ℕ)
    (P : ℕ)
    (h_eq : h = 60)
    (hb_eq : hb = 61)
    (P_eq : P = 62) :
    ¬ ∃ (a b c : ℝ), a + b + c = P ∧ 60^2 = 61^2 - (a^2 / 3) :=
by 
  sorry

end triangular_pyramid_nonexistence_l467_467274


namespace distinct_factors_1320_l467_467183

theorem distinct_factors_1320 :
  let p := 1320,
  prime_factors := [(2, 2), (3, 1), (5, 1), (11, 1)],
  num_factors := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in num_factors = 24 := by
  sorry

end distinct_factors_1320_l467_467183


namespace least_value_of_sum_l467_467234

theorem least_value_of_sum (a b : ℝ) (h1 : log 3 a + log 3 b ≥ 4) : a + b ≥ 18 := 
by
  sorry

end least_value_of_sum_l467_467234


namespace range_of_a_l467_467119

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≥ 1 then a^x else (3 - a) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 2 ≤ a ∧ a < 3 :=
by
  sorry

end range_of_a_l467_467119


namespace smallest_single_discount_l467_467252

noncomputable def discount1 : ℝ := (1 - 0.20) * (1 - 0.20)
noncomputable def discount2 : ℝ := (1 - 0.10) * (1 - 0.15)
noncomputable def discount3 : ℝ := (1 - 0.08) * (1 - 0.08) * (1 - 0.08)

theorem smallest_single_discount : ∃ n : ℕ, (1 - n / 100) < discount1 ∧ (1 - n / 100) < discount2 ∧ (1 - n / 100) < discount3 ∧ n = 37 := sorry

end smallest_single_discount_l467_467252


namespace find_two_fractions_sum_eq_86_over_111_l467_467564

theorem find_two_fractions_sum_eq_86_over_111 :
  ∃ (a1 a2 d1 d2 : ℕ), 
    (0 < d1 ∧ d1 ≤ 100) ∧ 
    (0 < d2 ∧ d2 ≤ 100) ∧ 
    (nat.gcd a1 d1 = 1) ∧ 
    (nat.gcd a2 d2 = 1) ∧ 
    (↑a1 / ↑d1 + ↑a2 / ↑d2 = 86 / 111) ∧
    (a1 = 2 ∧ d1 = 3) ∧ 
    (a2 = 4 ∧ d2 = 37) := 
by
  sorry

end find_two_fractions_sum_eq_86_over_111_l467_467564


namespace quadratic_roots_and_new_equation_l467_467275

theorem quadratic_roots_and_new_equation (p x_1 x_2 y_1 y_2 t : ℝ) 
  (h_eq : x_1^2 + p * x_1 + 1 = 0)
  (h_eq : x_2^2 + p * x_2 + 1 = 0)
  (h_cond_y1 : y_1 = x_1 * (1 - x_1)) 
  (h_cond_y2 : y_2 = x_2 * (1 - x_2)) :
  (∃ t : ℝ, t^2 + (p^2 + p - 2) * t + (p + 2) = 0) ∧ 
  (p ∈ set.Icc (-2.5) (-2) ∪ set.Icc (-1) (-1)) :=
sorry

end quadratic_roots_and_new_equation_l467_467275


namespace rationalize_denominator_l467_467335

theorem rationalize_denominator (sqrt_98 : Real) (h : sqrt_98 = 7 * Real.sqrt 2) : 
  7 / sqrt_98 = Real.sqrt 2 / 2 := 
by 
  sorry

end rationalize_denominator_l467_467335


namespace general_term_arithmetic_sequence_l467_467247

theorem general_term_arithmetic_sequence {a : ℕ → ℕ} (d : ℕ) (h_d : d ≠ 0)
  (h1 : a 3 + a 10 = 15)
  (h2 : (a 2 + d) * (a 2 + 10 * d) = (a 2 + 4 * d) * (a 2 + d))
  : ∀ n, a n = n + 1 :=
sorry

end general_term_arithmetic_sequence_l467_467247


namespace distinct_positive_factors_of_1320_l467_467196

theorem distinct_positive_factors_of_1320 : 
  ∀ n ≥ 0, prime_factorization 1320 = [(2, 3), (3, 1), (11, 1), (5, 1)] → factors_count 1320 = 32 :=
by
  intro n hn h
  sorry

end distinct_positive_factors_of_1320_l467_467196


namespace series_sum_l467_467923

theorem series_sum : 
  (∑ n : ℕ, (2^n) / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end series_sum_l467_467923


namespace log_base_3_81_sqrt_3_l467_467007

theorem log_base_3_81_sqrt_3 :
  log 3 (81 * real.sqrt 3) = 9 / 2 :=
by {
  have h1 : 81 = 3^4 := by norm_num,
  have h2 : real.sqrt 3 = 3^(1/2) := by { rw real.sqrt_eq_rpow, norm_num },
  rw [h1, h2, real.log_rpow, real.log_rpow, real.log_mul],
  norm_num,
  rw [mul_one]
}

end log_base_3_81_sqrt_3_l467_467007


namespace distinct_four_digit_numbers_l467_467143

theorem distinct_four_digit_numbers : 
  {n : ℕ | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3, 4, 5} ∧ c ∈ {1, 2, 3, 4, 5} ∧ d ∈ {1, 2, 3, 4, 5}
  } = 120 :=
by
  sorry

end distinct_four_digit_numbers_l467_467143


namespace complex_division_correct_l467_467984

variables (a b : ℝ) (i : ℂ) [is_imag_unit : i^2 = -1]

-- Given conditions
def complex_condition : Prop := a + b * i = i * (2 - i)

-- The goal is to prove that (b + a * i) / (a - b * i) = i
theorem complex_division_correct (h : complex_condition a b i) :
  (b + a * i) / (a - b * i) = i :=
sorry

end complex_division_correct_l467_467984


namespace anya_lost_games_l467_467037

/-- 
Five girls (Anya, Bella, Valya, Galya, Dasha) played several games of table tennis. At any given time, two girls were playing,
and the other three were resting. The girl who lost the game went to rest, and her place was taken by the girl who had rested 
the most rounds. There are no ties in table tennis. Anya played 4 games, Bella played 6 games, Valya played 7 games, Galya 
played 10 games, and Dasha played 11 games.

We need to prove that the numbers of all games in which Anya lost are 4, 8, 12, and 16.
-/
theorem anya_lost_games :
  (∃ games : list ℕ,
    games.length = 19 ∧
    (∀ i, i ∈ games → i = 4 ∨ i = 8 ∨ i = 12 ∨ i = 16) ∧
    (∀ i, i ∉ games → true)) :=
by
  sorry

end anya_lost_games_l467_467037


namespace thomas_spends_40000_in_a_decade_l467_467417

/-- 
Thomas spends 4k dollars every year on his car insurance.
One decade is 10 years.
-/
def spending_per_year : ℕ := 4000

def years_in_a_decade : ℕ := 10

/-- 
We need to prove that the total amount Thomas spends in a decade on car insurance equals $40,000.
-/
theorem thomas_spends_40000_in_a_decade : spending_per_year * years_in_a_decade = 40000 := by
  sorry

end thomas_spends_40000_in_a_decade_l467_467417


namespace pen_sales_average_l467_467832

theorem pen_sales_average (d : ℕ) (h1 : 96 + 44 * d > 0) (h2 : (96 + 44 * d) / (d + 1) = 48) : d = 12 :=
by
  sorry

end pen_sales_average_l467_467832


namespace value_of_a_l467_467581

theorem value_of_a (x y z a : ℤ) (k : ℤ) 
  (h1 : x = 4 * k) (h2 : y = 6 * k) (h3 : z = 10 * k) 
  (hy_eq : y^2 = 40 * a - 20) 
  (ha_int : ∃ m : ℤ, a = m) : a = 1 := 
  sorry

end value_of_a_l467_467581


namespace tan_sub_pi_over_4_l467_467648

variables (α : ℝ)
axiom tan_alpha : Real.tan α = 1 / 6

theorem tan_sub_pi_over_4 : Real.tan (α - Real.pi / 4) = -5 / 7 := by
  sorry

end tan_sub_pi_over_4_l467_467648


namespace not_B_l467_467912

def op (x y : ℝ) := (x - y) ^ 2

theorem not_B (x y : ℝ) : 2 * (op x y) ≠ op (2 * x) (2 * y) :=
by
  sorry

end not_B_l467_467912


namespace distinct_factors_1320_l467_467181

theorem distinct_factors_1320 :
  let p := 1320,
  prime_factors := [(2, 2), (3, 1), (5, 1), (11, 1)],
  num_factors := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in num_factors = 24 := by
  sorry

end distinct_factors_1320_l467_467181


namespace num_factors_1320_l467_467189

theorem num_factors_1320 : 
  let n := 1320
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 11
  let e1 := 3
  let e2 := 1
  let e3 := 1
  let e4 := 1
  factors_count (p1^e1 * p2^e2 * p3^e3 * p4^e4) = 32 := 
by
  let prime_factors := [(p1, e1), (p2, e2), (p3, e3), (p4, e4)]
  sorry

end num_factors_1320_l467_467189


namespace rationalize_denominator_l467_467331

theorem rationalize_denominator (a b c : ℝ) (h : b ≠ 0) (h0 : 98 = c * c) (h1 : 7 = c) :
  (7 / (Real.sqrt 98) = (Real.sqrt 2) / 2) :=
by
  sorry

end rationalize_denominator_l467_467331


namespace abs_inequality_solution_l467_467353

theorem abs_inequality_solution (x : ℝ) : 
  abs (2 * x - 1) < abs x + 1 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end abs_inequality_solution_l467_467353


namespace inverse_fourier_transform_l467_467962

noncomputable def F (p : ℝ) : ℂ :=
if 0 < p ∧ p < 1 then 1 else 0

noncomputable def f (x : ℝ) : ℂ :=
(1 / Real.sqrt (2 * Real.pi)) * ((1 - Complex.exp (-Complex.I * x)) / (Complex.I * x))

theorem inverse_fourier_transform :
  ∀ x, (f x) = (1 / (Real.sqrt (2 * Real.pi))) * ((1 - Complex.exp (-Complex.I * x)) / (Complex.I * x)) := by
  intros
  sorry

end inverse_fourier_transform_l467_467962


namespace probability_abs_le_1_88_l467_467718

open Probability

theorem probability_abs_le_1_88 {X : ℝ → ℝ} (hX : X ∼ stdNormal) (hP : P(X ≤ 1.88) = 0.97) :
  P(|X| ≤ 1.88) = 0.94 := 
sorry

end probability_abs_le_1_88_l467_467718


namespace maggie_bouncy_balls_l467_467724

theorem maggie_bouncy_balls (yellow_packs green_pack_given green_pack_bought : ℝ)
    (balls_per_pack : ℝ)
    (hy : yellow_packs = 8.0)
    (hg_given : green_pack_given = 4.0)
    (hg_bought : green_pack_bought = 4.0)
    (hbp : balls_per_pack = 10.0) :
    (yellow_packs * balls_per_pack + green_pack_bought * balls_per_pack - green_pack_given * balls_per_pack = 80.0) :=
by
  sorry

end maggie_bouncy_balls_l467_467724


namespace sum_of_k_equals_b_k_is_250500_l467_467290

noncomputable def a_seq : ℕ → ℝ
| 1     := 0.301
| k + 1 := if k % 2 = 0 
           then (("0." ++ (List.replicate (k + 3 - 2) '3' ++ 
                           ["01" ++ List.repeat '1' (k + 2 - (k + 3 - 2) - 2)]).asString).toReal)^(a_seq k)
           else (("0." ++ (List.replicate (k + 3 - 2) '3' ++ 
                           ["01" ++ List.repeat '0' (k + 2 - (k + 3 - 2) - 2)]).asString).toReal)^(a_seq k)

noncomputable def b_seq := List.sort (≥) (List.map a_seq (List.finRange 1001))

theorem sum_of_k_equals_b_k_is_250500 :
  (List.sum $ List.filter (λ k, a_seq k = b_seq k) (List.finRange 1001)) = 250500 :=
sorry

end sum_of_k_equals_b_k_is_250500_l467_467290


namespace parallel_planes_and_lines_l467_467714

variable {Point : Type*}
variable {Line : Type*} [incidence_space Point Line]
variable {Plane : Type*} [incidence_space Point Line Plane]

-- Let m and n be two different lines.
variables (m n : Line)
-- Let α and β be two different planes.
variables (α β : Plane)
-- With m ⊥ β and n ⊥ β
variables (h₁ : ∀ p : Point, p ∈ m → p ∈ β → perp (m : set Point) (β : set Point))
variables (h₂ : ∀ p : Point, p ∈ n → p ∈ β → perp (n : set Point) (β : set Point))

-- Then, "α ∥ β" is a necessary and sufficient condition for "m ∥ n" 
theorem parallel_planes_and_lines :
  (parallel α β) ↔ (parallel m n) :=
sorry

end parallel_planes_and_lines_l467_467714


namespace algorithm_result_l467_467592

theorem algorithm_result (a b c : ℕ) (h_a : a = 3) (h_b : b = 6) (h_c : c = 2) : 
  let m := if b < a then b else a in
  let m := if c < m then c else m in
  m = 2 :=
by
  have h1 : m = 3 := sorry -- Step involving the assumption h_a and initial m
  have h2 : b < a = false := sorry -- Comparison leading to the same m
  have h3 : m = 3 := sorry -- Validation after comparing b with m
  have h4 : c < m = true := sorry -- Comparison involving h_c
  have m := c := sorry -- Update of m after step 3
  exact show m = 2 from sorry -- Final result proof

end algorithm_result_l467_467592


namespace distinct_four_digit_numbers_count_l467_467149

theorem distinct_four_digit_numbers_count (digits : Finset ℕ) (h : digits = {1, 2, 3, 4, 5}) :
  (∃ (numbers : Finset (ℕ × ℕ × ℕ × ℕ)), 
   (∀ (a b c d : ℕ), (a, b, c, d) ∈ numbers → a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
   numbers.card = 120) :=
begin
  sorry
end

end distinct_four_digit_numbers_count_l467_467149


namespace cars_produced_total_l467_467469

theorem cars_produced_total :
  3884 + 2871 = 6755 :=
by
  sorry

end cars_produced_total_l467_467469


namespace compare_f_values_max_f_value_l467_467611

noncomputable def f (x : ℝ) : ℝ := -2 * Real.sin x - Real.cos (2 * x)

theorem compare_f_values :
  f (Real.pi / 4) > f (Real.pi / 6) :=
sorry

theorem max_f_value :
  ∃ x : ℝ, f x = 3 :=
sorry

end compare_f_values_max_f_value_l467_467611


namespace projection_of_point_A_on_xoz_l467_467600

noncomputable def A : ℝ × ℝ × ℝ := (3, 7, -4)

noncomputable def projection_xoz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, 0, p.3)

noncomputable def vector_square (v : ℝ × ℝ × ℝ) : ℝ :=
  v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2

theorem projection_of_point_A_on_xoz (A : ℝ × ℝ × ℝ) :
  vector_square (projection_xoz A) = 25 :=
by
  sorry

end projection_of_point_A_on_xoz_l467_467600


namespace max_siskins_on_poles_l467_467876

theorem max_siskins_on_poles : 
    (∀ (poles : List Nat), length poles = 25 → 
        ∀ (siskins_on_poles : List (Option Nat)), 
        length siskins_on_poles = 25 → 
        (∀ (i : Nat), (i < 25) → (∀ (j : Nat), (abs (j - i) = 1) → 
          (siskins_on_poles.nth i = some _) → (siskins_on_poles.nth j ≠ some _))) → 
        (∃ k, ∀ n, (n < 25) → 
            n + k ≤ 24 → 
            ∃ m, (siskins_on_poles.nth m = some m) ∧ 
              (m < 25) ∧ 
              (∀ l, (m ≠ l) → (siskins_on_poles.nth l ≠ some _)))) → true) := 
by sorry

end max_siskins_on_poles_l467_467876


namespace first_decimal_sqrt_l467_467524

theorem first_decimal_sqrt (n : ℕ) (h : 4 ≤ n) : 
  let d := (sqrt (n^2 + n + 1) % 1) * 10 in 
  (d < 6) ∧ (d ≥ 5) :=
by
  sorry

end first_decimal_sqrt_l467_467524


namespace sum_of_positive_integers_lcm72_l467_467815

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_of_positive_integers_lcm72 : 
  ∑ ν in {ν | is_solution ν}.to_finset, ν = 180 :=
by 
  sorry

end sum_of_positive_integers_lcm72_l467_467815


namespace general_term_formula_smallest_n_l467_467908

def a_n (n : ℕ) : ℕ :=
if n = 1 then 1 else 2 * n - 1

def S_n (n : ℕ) : ℕ := n * a_n n - n * (n - 1)

def T_n (n : ℕ) : ℚ :=
Finset.sum (Finset.range n) (λ k, 1 / ((a_n k.succ) * (a_n (k.succ.succ))))

theorem general_term_formula (n : ℕ) (hn : n ≥ 1) : a_n n = 2 * n - 1 :=
sorry

theorem smallest_n (T : ℚ → Prop) (hT : ∀ n, T (T_n n)) : ∃ (n : ℕ), T (T_n n) ∧ n = 12 :=
sorry

end general_term_formula_smallest_n_l467_467908


namespace solution_to_problem_l467_467361

theorem solution_to_problem (f : ℕ → ℕ) 
  (h1 : f 2 = 20)
  (h2 : ∀ n : ℕ, 0 < n → f (2 * n) + n * f 2 = f (2 * n + 2)) :
  f 10 = 220 :=
by
  sorry

end solution_to_problem_l467_467361


namespace value_of_x_l467_467405

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 := 
by
  sorry

end value_of_x_l467_467405


namespace percentage_boys_friends_l467_467696
-- Bringing in the necessary library

-- Definitions based on the conditions
def Julian_total_friends : ℕ := 80
def Julian_friend_percentage_boys : ℕ := 60
def Julian_friend_percentage_girls : ℕ := 40
def Boyd_total_friends : ℕ := 100

-- Calculations based on definitions
def Julian_boys_friends : ℕ := (Julian_friend_percentage_boys * Julian_total_friends) / 100
def Julian_girls_friends : ℕ := (Julian_friend_percentage_girls * Julian_total_friends) / 100
def Boyd_girls_friends : ℕ := 2 * Julian_girls_friends
def Boyd_boys_friends : ℕ := Boyd_total_friends - Boyd_girls_friends

-- The theorem to prove
theorem percentage_boys_friends (Julian_total_friends : ℕ := 80)
                                (Julian_friend_percentage_boys : ℕ := 60)
                                (Julian_friend_percentage_girls : ℕ := 40)
                                (Boyd_total_friends : ℕ := 100)
                                (Julian_boys_friends : ℕ := (Julian_friend_percentage_boys * Julian_total_friends) / 100)
                                (Julian_girls_friends : ℕ := (Julian_friend_percentage_girls * Julian_total_friends) / 100)
                                (Boyd_girls_friends : ℕ := 2 * Julian_girls_friends)
                                (Boyd_boys_friends : ℕ := Boyd_total_friends - Boyd_girls_friends) :
  ((Boyd_boys_friends * 100) / Boyd_total_friends) = 36 := by
sory

end percentage_boys_friends_l467_467696


namespace ratio_of_areas_l467_467687

theorem ratio_of_areas {ABC : Triangle} {R S T : Point}
  (hR_midpoint : is_midpoint R (segment BC))
  (hCS_ratio : CS = 3 * SA)
  (hAT_TB_ratio : AT/TB = p/q)
  (w : area (triangle CRS))
  (x : area (triangle RBT))
  (z : area (triangle ATS))
  (hx_squared_eq_wz : x^2 = w * z) :
  p/q = (sqrt 105 - 3) / 6 :=
sorry

end ratio_of_areas_l467_467687


namespace complex_solution_l467_467304

theorem complex_solution (z : ℂ) (h : (z - 2* complex.I) * (2 - complex.I) = 5) : z = 2 + 3 * complex.I :=
  sorry

end complex_solution_l467_467304


namespace girls_ran_9_miles_l467_467682

def boys_laps : ℕ := 34
def additional_laps : ℕ := 20
def lap_distance : ℚ := 1 / 6

def girls_laps : ℕ := boys_laps + additional_laps
def girls_miles : ℚ := girls_laps * lap_distance

theorem girls_ran_9_miles : girls_miles = 9 := by
  sorry

end girls_ran_9_miles_l467_467682


namespace triangle_area_l467_467241

theorem triangle_area (a b c A B C : ℝ)
  (h1 : a = b / sin C + c / sin B)
  (h2 : b = sqrt 2) : 
  (1 / 2) * a * b * sin C = 1 :=
by
  sorry

end triangle_area_l467_467241


namespace cube_root_solution_l467_467955

theorem cube_root_solution (x : ℝ) : (∃ x : ℝ, (∛(5 - x) = -5 / 3)) ↔ x = 260 / 27 :=
by
  sorry

end cube_root_solution_l467_467955


namespace additional_men_joined_l467_467466

theorem additional_men_joined (men_initial : ℕ) (days_initial : ℕ)
  (days_new : ℕ) (additional_men : ℕ) :
  men_initial = 600 →
  days_initial = 20 →
  days_new = 15 →
  (men_initial * days_initial) = ((men_initial + additional_men) * days_new) →
  additional_men = 200 := 
by
  intros h1 h2 h3 h4
  sorry

end additional_men_joined_l467_467466


namespace point_in_fourth_quadrant_l467_467671

def point : ℝ × ℝ := (4, -3)

def is_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l467_467671


namespace anya_lost_games_l467_467031

def girl_count := 5
def anya_games := 4
def bella_games := 6
def valya_games := 7
def galya_games := 10
def dasha_games := 11
def total_games := 19

theorem anya_lost_games :
  (anya_games + bella_games + valya_games + galya_games + dasha_games) / 2 = total_games →
  ∀ games : list ℕ,
    games.length = total_games →
    (∀ g ∈ games, g > 0 ∧ g ≤ total_games) →
    (∀ g ∈ [anya_games, bella_games, valya_games, galya_games, dasha_games], g ≤ total_games) →
    ∀ a b : ℕ, a ≠ b →
    (anya_games, games.nth 3 = some 4) ∧
    (anya_games, games.nth 7 = some 8) ∧
    (anya_games, games.nth 11 = some 12) ∧
    (anya_games, games.nth 15 = some 16)
:= sorry

end anya_lost_games_l467_467031


namespace max_siskins_on_poles_l467_467875

theorem max_siskins_on_poles : 
    (∀ (poles : List Nat), length poles = 25 → 
        ∀ (siskins_on_poles : List (Option Nat)), 
        length siskins_on_poles = 25 → 
        (∀ (i : Nat), (i < 25) → (∀ (j : Nat), (abs (j - i) = 1) → 
          (siskins_on_poles.nth i = some _) → (siskins_on_poles.nth j ≠ some _))) → 
        (∃ k, ∀ n, (n < 25) → 
            n + k ≤ 24 → 
            ∃ m, (siskins_on_poles.nth m = some m) ∧ 
              (m < 25) ∧ 
              (∀ l, (m ≠ l) → (siskins_on_poles.nth l ≠ some _)))) → true) := 
by sorry

end max_siskins_on_poles_l467_467875


namespace find_two_irreducible_fractions_l467_467554

theorem find_two_irreducible_fractions :
  ∃ (a b d1 d2 : ℕ), 
    (1 ≤ a) ∧ 
    (1 ≤ b) ∧ 
    (gcd a d1 = 1) ∧ 
    (gcd b d2 = 1) ∧ 
    (1 ≤ d1) ∧ 
    (d1 ≤ 100) ∧ 
    (1 ≤ d2) ∧ 
    (d2 ≤ 100) ∧ 
    (a / (d1 : ℚ) + b / (d2 : ℚ) = 86 / 111) := 
by {
  sorry
}

end find_two_irreducible_fractions_l467_467554


namespace distinct_positive_factors_of_1320_l467_467198

theorem distinct_positive_factors_of_1320 : 
  ∀ n ≥ 0, prime_factorization 1320 = [(2, 3), (3, 1), (11, 1), (5, 1)] → factors_count 1320 = 32 :=
by
  intro n hn h
  sorry

end distinct_positive_factors_of_1320_l467_467198


namespace move_line_down_eq_l467_467742

theorem move_line_down_eq (x y : ℝ) : (y = 2 * x) → (y - 3 = 2 * x - 3) :=
by
  sorry

end move_line_down_eq_l467_467742


namespace sum_A_otimes_B_eq_29970_l467_467999

def A : Set ℝ := { x | ∃ (k : ℕ), k ≤ 9 ∧ x = 2 * k }
def B : Set ℝ := {98, 99, 100}

def A_otimes_B : Set ℝ := { x | ∃ a b, a ∈ A ∧ b ∈ B ∧ x = a * b + a + b }

theorem sum_A_otimes_B_eq_29970 : (∑ x in A_otimes_B, x) = 29970 := by
  sorry

end sum_A_otimes_B_eq_29970_l467_467999


namespace exists_irreducible_fractions_sum_to_86_over_111_l467_467556

theorem exists_irreducible_fractions_sum_to_86_over_111 :
  ∃ (a b p q : ℕ), (0 < a) ∧ (0 < b) ∧ (p ≤ 100) ∧ (q ≤ 100) ∧ (Nat.coprime a p) ∧ (Nat.coprime b q) ∧ (a * q + b * p = 86 * (p * q) / 111) :=
by
  sorry

end exists_irreducible_fractions_sum_to_86_over_111_l467_467556


namespace range_of_a_l467_467612

-- Definitions and conditions
def f (x : ℝ) (a : ℝ) : ℝ := -x^2 + 2 * a * |x - 1|
variable (a : ℝ)
variable (x : ℝ)
variable (h_a_pos : a > 0)

-- 1. Monotonic intervals and maximum value when a = 2
def f_a2_monotonic_intervals : set (set ℝ) :=
  {{x : ℝ | x < -2} ∪ {x : ℝ | x > 1} ∪ {x : ℝ | -2 < x ∧ x < 1} ∪ {x : ℝ | x > 2}}

#check f_a2_monotonic_intervals

def f_a2_max_value : ℝ := 8

-- 2. Range of a for |f(x)| ≤ 2 for x ∈ [-2, 1.5]
theorem range_of_a (h_abs_fx_le_2: ∀ x ∈ set.Icc (-2 : ℝ) (3 / 2), |f x a| ≤ 2) :
  1 / 3 ≤ a ∧ a ≤ 17 / 4 := by
  sorry

end range_of_a_l467_467612


namespace room_square_footage_ratio_l467_467775

-- Definitions of variables based on the conditions
variables (M : ℝ) -- Square footage of Martha's room
variables (J : ℝ) (S : ℝ) (O : ℝ) -- Square footage of Jenny's, Sam's, and Olivia's rooms

-- Equation conditions based on the given problem
def conditions : Prop :=
  J = M + 120 ∧
  S = M - 60 ∧
  O = M + 30 ∧
  M + J + S + O = 1200

-- Target ratio
def target_ratio : ℝ := 41 / 39

-- Theorem proving the required ratio
theorem room_square_footage_ratio (h : conditions M J S O) :
  (J + S) / (M + O) = target_ratio :=
  sorry

end room_square_footage_ratio_l467_467775


namespace sum_of_positive_integers_lcm72_l467_467814

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_of_positive_integers_lcm72 : 
  ∑ ν in {ν | is_solution ν}.to_finset, ν = 180 :=
by 
  sorry

end sum_of_positive_integers_lcm72_l467_467814


namespace anya_game_losses_l467_467060

theorem anya_game_losses (games : ℕ → ℕ) (anya_games bella_games valya_games galya_games dasha_games : ℕ)
  (H_total : anya_games = 4 ∧ bella_games = 6 ∧ valya_games = 7 ∧ galya_games = 10 ∧ dasha_games = 11) :
  (games anya_games = 4 ∧ games bella_games = 6 ∧
   games valya_games = 7 ∧ games galya_games = 10 ∧
   games dasha_games = 11) →  [4, 8, 12, 16] := sorry

end anya_game_losses_l467_467060


namespace cube_surface_area_l467_467454

theorem cube_surface_area (V : ℝ) (s : ℝ) (A : ℝ) (h1 : V = s^3) (h2 : V = 1000) : A = 6 * s^2 → A = 600 :=
by
  intro h3
  have hs : s = 10 := by  sorry  -- here we usually prove s is 10 from V = s^3 and V = 1000
  rw [hs, pow_two] at h3
  exact h3 -- now it follows A = 6 * (10 ^ 2) = 600 

end cube_surface_area_l467_467454


namespace regular_polygons_from_cube_intersection_l467_467437

noncomputable def cube : Type := sorry  -- Define a 3D cube type
noncomputable def plane : Type := sorry  -- Define a plane type

-- Define what it means for a polygon to be regular (equilateral and equiangular)
def is_regular_polygon (polygon : Type) : Prop := sorry

-- Define a function that describes the intersection of a plane with a cube,
-- resulting in a polygon
noncomputable def intersection (c : cube) (p : plane) : Type := sorry

-- Define predicates for the specific regular polygons: triangle, quadrilateral, and hexagon
def is_triangle (polygon : Type) : Prop := sorry
def is_quadrilateral (polygon : Type) : Prop := sorry
def is_hexagon (polygon : Type) : Prop := sorry

-- Ensure these predicates imply regular polygons
axiom triangle_is_regular : ∀ (t : Type), is_triangle t → is_regular_polygon t
axiom quadrilateral_is_regular : ∀ (q : Type), is_quadrilateral q → is_regular_polygon q
axiom hexagon_is_regular : ∀ (h : Type), is_hexagon h → is_regular_polygon h

-- The main theorem statement
theorem regular_polygons_from_cube_intersection (c : cube) (p : plane) :
  is_regular_polygon (intersection c p) →
  is_triangle (intersection c p) ∨ is_quadrilateral (intersection c p) ∨ is_hexagon (intersection c p) :=
sorry

end regular_polygons_from_cube_intersection_l467_467437


namespace sum_lcm_eq_72_l467_467821

theorem sum_lcm_eq_72 (s : Finset ℕ) 
  (h1 : ∀ ν ∈ s, Nat.lcm ν 24 = 72) :
  s.sum id = 180 :=
by
  have h2 : ∀ ν, ν ∈ s → ∃ n, ν = 3 * n := 
    by sorry
  have h3 : ∀ n, ∃ ν, ν = 3 * n ∧ ν ∈ s := 
    by sorry
  sorry

end sum_lcm_eq_72_l467_467821


namespace complex_number_equality_l467_467985

theorem complex_number_equality (a b : ℝ) (i : ℂ) (h : i = complex.I) (h1 : (a + i) * (1 + i) = b * i) : 
  a + b * i = 1 + 2 * i := by
  sorry

end complex_number_equality_l467_467985


namespace Anya_loss_games_l467_467018

noncomputable def gamesPlayed : Nat := 19

constant Anya_games : Nat := 4
constant Bella_games : Nat := 6
constant Valya_games : Nat := 7
constant Galya_games : Nat := 10
constant Dasha_games : Nat := 11

constant gameResults : Fin 19 → Option String
constant lost (g : Nat) (p : String) : Prop

axiom Anya_game_indices : ∀ (i : Fin 4), lost (gameResults i.val) "Anya" → i.val = 4 * i + 4

theorem Anya_loss_games : list Nat :=
have Anya_lost_games : list Nat := [4, 8, 12, 16],
Anya_lost_games

end Anya_loss_games_l467_467018


namespace sum_of_alternating_sums_for_n_8_l467_467065

noncomputable def alternating_sum (s : Finset ℕ) : ℤ :=
  s.toList.sort (· > ·).alternatingSum

theorem sum_of_alternating_sums_for_n_8 :
  let S := Finset.range 8
  Finset.univ.image (λ s => alternating_sum s).sum = 1024 := sorry

end sum_of_alternating_sums_for_n_8_l467_467065


namespace distinct_four_digit_numbers_count_l467_467136

theorem distinct_four_digit_numbers_count :
  (Finset.card (Finset.filter (λ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ (∀ d ∈ [x / 1000, (x / 100) % 10, (x / 10) % 10, x % 10], d ∈ [1, 2, 3, 4, 5]) ∧ (denote_distinct_digits (x)) ) (Finset.Ico 1000 10000))) = 120 := 
sorry

def denote_distinct_digits (x : ℕ) : Prop :=
  ∀ i j, (i ≠ j) → (x / (10 ^ i)) % 10 ≠ (x / (10 ^ j)) % 10

end distinct_four_digit_numbers_count_l467_467136


namespace min_distance_midpoint_to_origin_l467_467127

/-- Given two points P1 (x1, y1) and P2 (x2, y2) that lie on the lines x - y - 5 = 0 and x - y - 15 = 0 respectively,
prove that the minimum distance from the midpoint P of P1P2 to the origin is 5√2. -/
theorem min_distance_midpoint_to_origin (x1 y1 x2 y2 : ℝ) (h1 : x1 - y1 = 5) (h2 : x2 - y2 = 15) : 
  let xP := (x1 + x2) / 2,
      yP := (y1 + y2) / 2,
      dist := (10 : ℝ) / (Real.sqrt 2) in
  dist = 5 * Real.sqrt 2 :=
by
  let xP := (x1 + x2) / 2
  let yP := (y1 + y2) / 2
  let midpoint_eq := xP - yP = 10
  let dist := ∣0 - 0 - 10∣ / Real.sqrt 2
  exact sorry

end min_distance_midpoint_to_origin_l467_467127


namespace max_siskins_on_poles_l467_467881

-- Definitions based on problem conditions
def pole : Type := ℕ
def siskins (poles : pole) : Prop := poles ≤ 25
def adjacent (p₁ p₂ : pole) : Prop := (p₁ = p₂ + 1) ∨ (p₁ = p₂ - 1)

-- Given conditions
def conditions (p : pole → bool) : Prop :=
  ∀ p₁ p₂ : pole, p p₁ = true → p p₂ = true → adjacent p₁ p₂ → false

-- Main problem statement
theorem max_siskins_on_poles : ∃ p : pole → bool, (∀ i : pole, p i = true → siskins i) ∧ (conditions p) ∧ (∑ i in finset.range 25, if p i then 1 else 0 = 24) :=
begin
  sorry
end

end max_siskins_on_poles_l467_467881


namespace tan_alpha_solution_expression_solution_l467_467988
noncomputable section

variables (α : ℝ)
open Real

-- Conditions
def condition1 := sin α + cos α = -sqrt 10 / 5
def condition2 := α ∈ Ioo (-π / 2) (π / 2)

-- Question 1
theorem tan_alpha_solution (h1 : condition1 α) (h2 : condition2 α) : tan α = -3 := sorry

-- Question 2
theorem expression_solution (h1 : condition1 α) (h2 : condition2 α) :
  2 * sin α ^ 2 + sin α * cos α - 1 = 1 / 2 := sorry

end tan_alpha_solution_expression_solution_l467_467988


namespace rationalize_denominator_l467_467334

theorem rationalize_denominator (a b c : ℝ) (h : b ≠ 0) (h0 : 98 = c * c) (h1 : 7 = c) :
  (7 / (Real.sqrt 98) = (Real.sqrt 2) / 2) :=
by
  sorry

end rationalize_denominator_l467_467334


namespace number_of_elements_in_P_union_Q_l467_467303

open Set

def P : Set ℝ := {x | x^2 - 3 * x + 2 = 0}

def Q : Set ℝ := {x | ∃ m ∈ P, x = 2 * m}

theorem number_of_elements_in_P_union_Q : (P ∪ Q).to_finset.card = 3 := by
  sorry

end number_of_elements_in_P_union_Q_l467_467303


namespace complex_magnitude_solution_l467_467584

noncomputable def complex_magnitude (z : ℂ) : ℝ := complex.abs z

theorem complex_magnitude_solution (z : ℂ) (h : complex.abs (z - (1 + 2 * complex.I)) = 0) : complex.abs z = real.sqrt 5 := 
sorry

end complex_magnitude_solution_l467_467584


namespace max_siskins_on_poles_l467_467871

theorem max_siskins_on_poles (n : ℕ) (h : n = 25) :
  ∃ k : ℕ, k = 24 ∧ (∀ (poless: Fin n → ℕ) (siskins: Fin n → ℕ),
     (∀ i: Fin n, siskins i ≤ 1) 
     ∧ (∀ i: Fin n, (siskins i = 1 → (poless i = 0)))
     ∧ poless 0 = 0
     → ( ∀ j: Fin n, (j < n → siskins j + siskins (j+1) < 2)) 
     ∧ (k ≤ n)
     ∧ ( ∀ l: Fin n, ((l < k → siskins l = 1) →
       ((k ≤ l < n → siskins l = 0)))))
  sorry

end max_siskins_on_poles_l467_467871


namespace average_age_increase_l467_467439

theorem average_age_increase
  (n : ℕ)
  (A : ℝ)
  (w : ℝ)
  (h1 : (n + 1) * (A + w) = n * A + 39)
  (h2 : (n + 1) * (A - 1) = n * A + 15)
  (hw : w = 7) :
  w = 7 := 
by
  sorry

end average_age_increase_l467_467439


namespace white_surface_area_fraction_l467_467850

theorem white_surface_area_fraction :
  let larger_cube_edge := 4
  let smaller_cube_edge := 1
  let total_smaller_cubes := 64
  let white_cubes := 48
  let black_cubes := 16
  let total_faces := 6
  let black_cubes_per_face := 2
  let surface_area := total_faces * larger_cube_edge^2
  let black_faces_exposed := total_faces * black_cubes_per_face
  let white_faces_exposed := surface_area - black_faces_exposed
  (white_faces_exposed / surface_area) = (7 / 8) :=
by
  let larger_cube_edge := 4
  let smaller_cube_edge := 1
  let total_smaller_cubes := 64
  let white_cubes := 48
  let black_cubes := 16
  let total_faces := 6
  let black_cubes_per_face := 2
  let surface_area := total_faces * larger_cube_edge^2
  let black_faces_exposed := total_faces * black_cubes_per_face
  let white_faces_exposed := surface_area - black_faces_exposed
  have h_white_fraction : (white_faces_exposed / surface_area) = (7 / 8) := sorry
  exact h_white_fraction

end white_surface_area_fraction_l467_467850


namespace equation_has_two_distinct_real_roots_l467_467607

open Real

theorem equation_has_two_distinct_real_roots (m : ℝ) :
  (∃ (x1 x2 : ℝ), 0 < x1 ∧ x1 < 16 ∧ 0 < x2 ∧ x2 < 16 ∧ x1 ≠ x2 ∧ exp (m * x1) = x1^2 ∧ exp (m * x2) = x2^2) ↔
  (log 2 / 2 < m ∧ m < 2 / exp 1) :=
by sorry

end equation_has_two_distinct_real_roots_l467_467607


namespace num_students_above_120_l467_467245

noncomputable def class_size : ℤ := 60
noncomputable def mean_score : ℝ := 110
noncomputable def std_score : ℝ := sorry  -- We do not know σ explicitly
noncomputable def probability_100_to_110 : ℝ := 0.35

def normal_distribution (x : ℝ) : Prop :=
  sorry -- placeholder for the actual normal distribution formula N(110, σ^2)

theorem num_students_above_120 :
  ∃ (students_above_120 : ℤ),
  (class_size = 60) ∧
  (∀ score, normal_distribution score → (100 ≤ score ∧ score ≤ 110) → probability_100_to_110 = 0.35) →
  students_above_120 = 9 :=
sorry

end num_students_above_120_l467_467245


namespace count_functions_with_period_pi_l467_467894

-- Conditions
def f1 (x : ℝ) : ℝ := sin (|x|)
def f2 (x : ℝ) : ℝ := |sin x|
def f3 (x : ℝ) : ℝ := sin (2 * x + 2 * Real.pi / 3)
def f4 (x : ℝ) : ℝ := tan (2 * x + 2 * Real.pi / 3)

-- The periodicity properties of the functions as given
axiom f1_not_periodic : ¬ ∃ T > 0, ∀ x, f1 (x + T) = f1 x
axiom f2_periodic : ∃ T > 0, T = Real.pi ∧ ∀ x, f2(x + T) = f2 x
axiom f3_periodic : ∃ T > 0, T = Real.pi ∧ ∀ x, f3(x + T) = f3 x
axiom f4_periodic : ∃ T > 0, T = Real.pi / 2 ∧ ∀ x, f4(x + T) = f4 x

-- Prove that there are exactly 2 functions with a period of π.
theorem count_functions_with_period_pi :
  (finset.filter (λ f, ∃ T > 0, T = Real.pi ∧ ∀ x, f (x + T) = f x)
    (finset.of_list [f1, f2, f3, f4])).card = 2 := by
  sorry

end count_functions_with_period_pi_l467_467894


namespace strictly_increasing_sequence_implies_lambda_gt_neg3_l467_467626

variable (λ : ℝ)

def a_n (n : ℕ) : ℝ := n^2 + λ * n + 1

theorem strictly_increasing_sequence_implies_lambda_gt_neg3 :
  (∀ n : ℕ, n > 0 → a_n λ (n + 1) > a_n λ n) → λ > -3 :=
sorry

end strictly_increasing_sequence_implies_lambda_gt_neg3_l467_467626


namespace max_siskins_on_poles_l467_467886

/-- 
Given 25 poles in a row, where only one siskin can occupy any given pole, 
and if a siskin lands on a pole, a siskin sitting on an adjacent pole will immediately fly away, 
the maximum number of siskins that can simultaneously be on the poles is 24.
-/
theorem max_siskins_on_poles : 
  ∀ (poles : Fin 25 → Bool), 
  (∀ i, poles i = false ∨ ∃ j, (abs (i - j) = 1 ∧ poles j = false))
  → ∃ n, (0 < n ∧ n ≤ 25 ∧ max_siskins poles n = 24) :=
by
  sorry

end max_siskins_on_poles_l467_467886


namespace percentage_cleared_land_l467_467851

-- Given conditions
def total_land : ℝ := 4999.999999999999
def grapes_percentage : ℝ := 0.10
def potatoes_percentage : ℝ := 0.80
def tomato_land : ℝ := 450

-- Unknown to be determined by conditions
def total_cleared_land : ℝ := 4500

-- Required to prove the percentage cleared land
theorem percentage_cleared_land (total_land grapes_percentage potatoes_percentage tomato_land : ℝ) : 
  let C := total_cleared_land in
  let cleared_percentage := (C / total_land) * 100 in
  cleared_percentage ≈ 90 :=
by
  have total_cleared_eq : C = tomato_land / (1 - grapes_percentage - potatoes_percentage) := sorry
  have : (total_cleared_land / total_land) * 100 = 90 := by sorry
  assumption

end percentage_cleared_land_l467_467851


namespace residue_of_alternating_sum_modulo_2024_l467_467297

def alternating_sum_sequence (n : ℕ) : ℤ :=
  ∑ i in (Finset.range n), if i % 2 = 0 then (i + 1 : ℤ) else -(i + 1 : ℤ)

theorem residue_of_alternating_sum_modulo_2024 : alternating_sum_sequence 2022 % 2024 = 1023 := 
by 
  sorry

end residue_of_alternating_sum_modulo_2024_l467_467297


namespace eccentricity_of_hyperbola_l467_467620

variables {a b : ℝ} 

def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

def eccentricity (c a : ℝ) : ℝ := c / a

theorem eccentricity_of_hyperbola
  (a b : ℝ)
  (h_a : a > 0)
  (h_b : b > 0)
  (AB BC : ℝ)
  (h1 :  AB = 2 * b^2 / a)
  (h2 :  BC = 2 * real.sqrt (a^2 + b^2))
  (h3 :  2 * AB = 3 * BC) :
  eccentricity (real.sqrt (a^2 + b^2)) a = 2 := sorry

end eccentricity_of_hyperbola_l467_467620


namespace imaginary_part_of_z_l467_467305

def z : ℂ := (2 + complex.i) / (1 + complex.i)^2

theorem imaginary_part_of_z : complex.im z = -1 := 
by 
sorry

end imaginary_part_of_z_l467_467305


namespace program_computes_3_pow_55_l467_467380

theorem program_computes_3_pow_55 :
  let S := (List.range 10).foldl (λ acc i, acc * 3 ^ (i + 1)) 1
  in S = 3^55 :=
by
  sorry

end program_computes_3_pow_55_l467_467380


namespace value_of_x_l467_467407

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 := 
by
  sorry

end value_of_x_l467_467407


namespace difference_x_max_min_l467_467831

theorem difference_x_max_min (x : ℝ) (w : ℝ) (s : set ℝ) (hS : s = {4, 314, 710, x}) (hW : w = 12) : 
  abs ((max (s \ {x}) + w) - (min (s \ {x}) - w)) = 682 :=
by 
  have h1 : max (s \ {x}) ≤ 710 := by sorry
  have h2 : min (s \ {x}) = 4 := by sorry
  have x_max := min (s \ {x}) + w
  have x_min := max (s \ {x}) - w
  show abs (x_max - x_min) = 682
  sorry

end difference_x_max_min_l467_467831


namespace b_ranges_acutetriangle_l467_467715

theorem b_ranges_acutetriangle (A B C: ℝ) (a b c: ℝ) (hA_ac: 0 < A ∧ A < π / 3) (hB: B = 2 * A) (hC: C = π - 3 * A) (hABC_acute: 0 < C ∧ C < 2 * π / 3) (ha: a = 1) : 1 < b ∧ b < 2 :=
by
  have hsin := (Real.sin_double_angle A)
  have hcos := Real.cos_double_angle
  sorry

end b_ranges_acutetriangle_l467_467715


namespace simplify_expression_l467_467349

-- Define the question and conditions
theorem simplify_expression (x y : ℝ) (h : |x + 1| + (2 * y - 4)^2 = 0) :
  (2*x^2*y - 3*x*y) - 2*(x^2*y - x*y + 1/2*x*y^2) + x*y = 4 :=
by
  -- proof steps if needed, but currently replaced with 'sorry' to indicate proof needed
  sorry

end simplify_expression_l467_467349


namespace find_range_of_set_w_l467_467453

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def set_w : set ℕ := {x | is_prime x ∧ 10 < x ∧ x < 25}

theorem find_range_of_set_w :
  ∃ (min_w max_w : ℕ), min_w ∈ set_w ∧ max_w ∈ set_w ∧ (∀ y ∈ set_w, min_w ≤ y ∧ y ≤ max_w) ∧ (max_w - min_w = 12) :=
sorry

end find_range_of_set_w_l467_467453


namespace I_pos_l467_467092

theorem I_pos (n : ℕ) :
  let I := (n+1)^2 + n - (nat.floor (real.sqrt ((n+1)^2 + n + 1)))^2 in
  I = n := 
sorry

end I_pos_l467_467092


namespace range_of_f_on_0_to_3_l467_467771

-- Let f be a function defined as x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 2 * x

-- We are to prove the range of f on the interval [0, 3] is exactly [-1, 3]
theorem range_of_f_on_0_to_3 : ∀ y, (∃ x ∈ set.Icc (0:ℝ) 3, f x = y) ↔ y ∈ set.Icc (-1:ℝ) 3 :=
by sorry

end range_of_f_on_0_to_3_l467_467771


namespace unique_non_integer_l467_467758

theorem unique_non_integer :
  let x := sqrt 2 - 1 in
  let a := x - sqrt 2 in
  let b := x - 1 / x in
  let c := x + 1 / x in
  let d := x^2 + 2 * sqrt 2 in
  (¬is_int a ∧ is_int b ∧ is_int c ∧ is_int d) ∨
  (is_int a ∧ ¬is_int b ∧ is_int c ∧ is_int d) ∨
  (is_int a ∧ is_int b ∧ ¬is_int c ∧ is_int d) ∨
  (is_int a ∧ is_int b ∧ is_int c ∧ ¬is_int d) :=
by sorry

end unique_non_integer_l467_467758


namespace proof_problem_l467_467326

theorem proof_problem (x y z : ℝ) (h₁ : x ≠ y) 
  (h₂ : (x^2 - y*z) / (x * (1 - y*z)) = (y^2 - x*z) / (y * (1 - x*z))) :
  x + y + z = 1/x + 1/y + 1/z :=
sorry

end proof_problem_l467_467326


namespace max_siskins_on_poles_l467_467887

/-- 
Given 25 poles in a row, where only one siskin can occupy any given pole, 
and if a siskin lands on a pole, a siskin sitting on an adjacent pole will immediately fly away, 
the maximum number of siskins that can simultaneously be on the poles is 24.
-/
theorem max_siskins_on_poles : 
  ∀ (poles : Fin 25 → Bool), 
  (∀ i, poles i = false ∨ ∃ j, (abs (i - j) = 1 ∧ poles j = false))
  → ∃ n, (0 < n ∧ n ≤ 25 ∧ max_siskins poles n = 24) :=
by
  sorry

end max_siskins_on_poles_l467_467887


namespace tiling_2002_gon_with_rhombuses_l467_467261

theorem tiling_2002_gon_with_rhombuses : ∀ n : ℕ, n = 1001 → (n * (n - 1) / 2) = 500500 :=
by sorry

end tiling_2002_gon_with_rhombuses_l467_467261


namespace distinct_factors_1320_l467_467217

theorem distinct_factors_1320 : 
  ∃ n : ℕ, n = 24 ∧ 
  let a := 2 in 
  let b := 1 in 
  let c := 1 in 
  let d := 1 in 
  1320 = 2^a * 3^b * 5^c * 11^d :=
begin
  sorry,
end

end distinct_factors_1320_l467_467217


namespace anya_lost_games_correct_l467_467050

-- Definition of players and their game counts
def Players := {Anya, Bella, Valya, Galya, Dasha}
def games_played : Players → ℕ
| Anya   := 4
| Bella  := 6
| Valya  := 7
| Galya  := 10
| Dasha  := 11

-- Total games played by all players
def total_games_played : ℕ := Players.to_finset.sum games_played

-- Total number of games, considering each game involves two players
def total_number_of_games : ℕ := total_games_played / 2

-- The set of games in which Anya lost
def anya_lost_games : set ℕ := {4, 8, 12, 16}

theorem anya_lost_games_correct :
  ∀ i ∈ anya_lost_games, -- For each game in anya_lost_games
  anya_lost_games = {4, 8, 12, 16} := 
by
  sorry

end anya_lost_games_correct_l467_467050


namespace infinite_solutions_x2_y2_z2_x3_y3_z3_l467_467731

-- Define the parametric forms
def param_x (k : ℤ) := k * (2 * k^2 + 1)
def param_y (k : ℤ) := 2 * k^2 + 1
def param_z (k : ℤ) := -k * (2 * k^2 + 1)

-- Prove the equation
theorem infinite_solutions_x2_y2_z2_x3_y3_z3 :
  ∀ k : ℤ, param_x k ^ 2 + param_y k ^ 2 + param_z k ^ 2 = param_x k ^ 3 + param_y k ^ 3 + param_z k ^ 3 :=
by
  intros k
  -- Calculation needs to be proved here, we place a placeholder for now
  sorry

end infinite_solutions_x2_y2_z2_x3_y3_z3_l467_467731


namespace max_siskins_on_poles_l467_467872

theorem max_siskins_on_poles (n : ℕ) (h : n = 25) :
  ∃ k : ℕ, k = 24 ∧ (∀ (poless: Fin n → ℕ) (siskins: Fin n → ℕ),
     (∀ i: Fin n, siskins i ≤ 1) 
     ∧ (∀ i: Fin n, (siskins i = 1 → (poless i = 0)))
     ∧ poless 0 = 0
     → ( ∀ j: Fin n, (j < n → siskins j + siskins (j+1) < 2)) 
     ∧ (k ≤ n)
     ∧ ( ∀ l: Fin n, ((l < k → siskins l = 1) →
       ((k ≤ l < n → siskins l = 0)))))
  sorry

end max_siskins_on_poles_l467_467872


namespace _l467_467836

/-- Define the problem statement and conditions -/
noncomputable def hexagon_perimeter : ℝ :=
  let PU := 10;
  let QS := 8;
  let QX := (8^2 - 10^2/2)^0.5; -- Pythagorean theorem solution for QX
  let PX := 10 + QX;
  4 * PX

example : hexagon_perimeter = 40 + 16 * 2.sqrt := by
  sorry

example : |hexagon_perimeter - 63| < 1 := by
  sorry

end _l467_467836


namespace percentage_increase_correct_l467_467751

-- Define the highest and lowest scores as given conditions.
def highest_score : ℕ := 92
def lowest_score : ℕ := 65

-- State that the percentage increase calculation will result in 41.54%
theorem percentage_increase_correct :
  ((highest_score - lowest_score) * 100) / lowest_score = 4154 / 100 :=
by sorry

end percentage_increase_correct_l467_467751


namespace tan_eq_sqrt3_iff_l467_467983

theorem tan_eq_sqrt3_iff (x : ℝ) : 
  (∃ k : ℤ, x = (π / 3) + k * π) ↔ tan x = sqrt 3 :=
by sorry

end tan_eq_sqrt3_iff_l467_467983


namespace max_siskins_on_poles_l467_467888

/-- 
Given 25 poles in a row, where only one siskin can occupy any given pole, 
and if a siskin lands on a pole, a siskin sitting on an adjacent pole will immediately fly away, 
the maximum number of siskins that can simultaneously be on the poles is 24.
-/
theorem max_siskins_on_poles : 
  ∀ (poles : Fin 25 → Bool), 
  (∀ i, poles i = false ∨ ∃ j, (abs (i - j) = 1 ∧ poles j = false))
  → ∃ n, (0 < n ∧ n ≤ 25 ∧ max_siskins poles n = 24) :=
by
  sorry

end max_siskins_on_poles_l467_467888


namespace negation_sin_leq_one_l467_467728

theorem negation_sin_leq_one :
  ¬ (∀ x : ℝ, sin x ≤ 1) ↔ ∃ x : ℝ, sin x > 1 :=
by
  sorry

end negation_sin_leq_one_l467_467728


namespace angle_Z_measure_l467_467512

/-- Given the measures of angles X, Y, and W inside a larger triangle
    including angle Z, angle Z is calculated using the sum of interior
    angles of triangles. -/

theorem angle_Z_measure
  (angle_X angle_Y angle_W : ℝ)
  (hX : angle_X = 34)
  (hY : angle_Y = 53)
  (hW : angle_W = 43) :
  ∃ (angle_Z : ℝ), angle_Z = 130 :=
by
  sorry

end angle_Z_measure_l467_467512


namespace speed_of_A_is_3_l467_467606

theorem speed_of_A_is_3:
  (∃ x : ℝ, 3 * x + 3 * (x + 2) = 24) → x = 3 :=
by
  sorry

end speed_of_A_is_3_l467_467606


namespace total_weight_of_sections_l467_467781

theorem total_weight_of_sections :
  let doll_length := 5
  let doll_weight := 29 / 8
  let tree_length := 4
  let tree_weight := 2.8
  let section_length := 2
  let doll_weight_per_meter := doll_weight / doll_length
  let tree_weight_per_meter := tree_weight / tree_length
  let doll_section_weight := doll_weight_per_meter * section_length
  let tree_section_weight := tree_weight_per_meter * section_length
  doll_section_weight + tree_section_weight = 57 / 20 :=
sorry

end total_weight_of_sections_l467_467781


namespace distinct_positive_factors_of_1320_l467_467193

theorem distinct_positive_factors_of_1320 : 
  ∀ n ≥ 0, prime_factorization 1320 = [(2, 3), (3, 1), (11, 1), (5, 1)] → factors_count 1320 = 32 :=
by
  intro n hn h
  sorry

end distinct_positive_factors_of_1320_l467_467193


namespace Anya_loss_games_l467_467016

noncomputable def gamesPlayed : Nat := 19

constant Anya_games : Nat := 4
constant Bella_games : Nat := 6
constant Valya_games : Nat := 7
constant Galya_games : Nat := 10
constant Dasha_games : Nat := 11

constant gameResults : Fin 19 → Option String
constant lost (g : Nat) (p : String) : Prop

axiom Anya_game_indices : ∀ (i : Fin 4), lost (gameResults i.val) "Anya" → i.val = 4 * i + 4

theorem Anya_loss_games : list Nat :=
have Anya_lost_games : list Nat := [4, 8, 12, 16],
Anya_lost_games

end Anya_loss_games_l467_467016


namespace num_nice_numbers_below_2018_l467_467010

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def coin_value (n : ℕ) : ℕ :=
  if n ≤ 6 then factorial n else 0

def is_nice (k : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ), 
  1 ≤ a ∧ a ≤ 1 ∧
  1 ≤ b ∧ b ≤ 2 ∧
  1 ≤ c ∧ c ≤ 3 ∧
  1 ≤ d ∧ d ≤ 4 ∧
  1 ≤ e ∧ e ≤ 5 ∧
  1 ≤ f ∧ f ≤ 6 ∧
  k = a * coin_value 1 + b * coin_value 2 + c * coin_value 3 +
      d * coin_value 4 + e * coin_value 5 + f * coin_value 6

def count_nice_numbers (limit : ℕ) : ℕ :=
  (Finset.range limit).filter is_nice |>.card

theorem num_nice_numbers_below_2018 : count_nice_numbers 2018 = 210 := 
by 
  sorry

end num_nice_numbers_below_2018_l467_467010


namespace hyperbola_eccentricity_calculation_l467_467078

noncomputable def hyperbola_eccentricity (E : String) : ℝ := 
  if E = "x-axis" then 5 / 4
  else if E = "y-axis" then 5 / 3
  else 0

theorem hyperbola_eccentricity_calculation : 
  ∀ E, (hyperbola_eccentricity E = 5 / 4 ∨ hyperbola_eccentricity E = 5 / 3) :=
by
  intro E
  simp [hyperbola_eccentricity]
  cases E
  case str s =>
    cases s
    case "x-axis" => left; rfl
    case "y-axis" => right; rfl
  case _ => sorry

end hyperbola_eccentricity_calculation_l467_467078


namespace distinct_factors_1320_l467_467176

theorem distinct_factors_1320 :
  let p := 1320,
  prime_factors := [(2, 2), (3, 1), (5, 1), (11, 1)],
  num_factors := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in num_factors = 24 := by
  sorry

end distinct_factors_1320_l467_467176


namespace midpoint_equidistant_from_PQ_l467_467423

-- Definitions for problem conditions
variables {A B C M N K P Q : Point}
variables {Omega : Circle}
variables [Triangle ABC] (AB_gt_BC : Length AB > Length BC) (Omega_inscribed_ABC : Omega ∈ circumcircle ABC)
variables (AM_eq_CN : Length AM = Length CN) (MN_inter_AC_at_K : Line MN ∩ Line AC = K)
variables (P_is_incenter : P = incenter (triangle AMK)) (Q_is_excenter : Q = excenter (triangle CNK) CN)

-- The theorem to prove
theorem midpoint_equidistant_from_PQ :
  let S := midpoint_arc ABC Omega in
  dist S P = dist S Q :=
sorry

end midpoint_equidistant_from_PQ_l467_467423


namespace maximize_expression_l467_467840

theorem maximize_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 2) :
  (∃ P : ℝ, P = (sqrt (b^2 + c^2)) / (3 - a) + (sqrt (c^2 + a^2)) / (3 - b) + a + b - 2022 * c) →
  P ≤ 3 :=
by
  sorry

end maximize_expression_l467_467840


namespace remainder_of_a55_l467_467704

def concatenate_integers (n : ℕ) : ℕ :=
  -- Function to concatenate integers from 1 to n into a single number.
  -- This is a placeholder, actual implementation may vary.
  sorry

theorem remainder_of_a55 (n : ℕ) (hn : n = 55) :
  concatenate_integers n % 55 = 0 := by
  -- Proof is omitted, provided as a guideline.
  sorry

end remainder_of_a55_l467_467704


namespace sum_lcm_eq_72_l467_467818

theorem sum_lcm_eq_72 (s : Finset ℕ) 
  (h1 : ∀ ν ∈ s, Nat.lcm ν 24 = 72) :
  s.sum id = 180 :=
by
  have h2 : ∀ ν, ν ∈ s → ∃ n, ν = 3 * n := 
    by sorry
  have h3 : ∀ n, ∃ ν, ν = 3 * n ∧ ν ∈ s := 
    by sorry
  sorry

end sum_lcm_eq_72_l467_467818


namespace series_sum_l467_467933

theorem series_sum :
  ∑' n : ℕ, (2 ^ n) / (3 ^ (2 ^ n) + 1) = 1 / 2 :=
sorry

end series_sum_l467_467933


namespace movie_tickets_ratio_l467_467757

theorem movie_tickets_ratio (R H : ℕ) (hR : R = 25) (hH : H = 93) : 
  (H / R : ℚ) = 93 / 25 :=
by
  sorry

end movie_tickets_ratio_l467_467757


namespace count_three_digit_distinct_under_800_l467_467642

-- Definitions
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 800
def distinct_digits (n : ℕ) : Prop := (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) 

-- Theorem
theorem count_three_digit_distinct_under_800 : ∃ k : ℕ, k = 504 ∧ ∀ n : ℕ, is_three_digit n → distinct_digits n → n < 800 :=
by 
  exists 504
  sorry

end count_three_digit_distinct_under_800_l467_467642


namespace max_value_l467_467630

def a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
def b : ℝ × ℝ := (Real.sqrt 3, -1)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem max_value : ∀ θ : ℝ, magnitude (2 • a θ - b) ≤ 4 :=
by {
  sorry
}

end max_value_l467_467630


namespace employed_males_population_percentage_l467_467270

-- Define the conditions of the problem
variables (P : Type) (population : ℝ) (employed_population : ℝ) (employed_females : ℝ)

-- Assume total population is 100
def total_population : ℝ := 100

-- 70 percent of the population are employed
def employed_population_percentage : ℝ := total_population * 0.70

-- 70 percent of the employed people are females
def employed_females_percentage : ℝ := employed_population_percentage * 0.70

-- 21 percent of the population are employed males
def employed_males_percentage : ℝ := 21

-- Main statement to be proven
theorem employed_males_population_percentage :
  employed_males_percentage = ((employed_population_percentage - employed_females_percentage) / total_population) * 100 :=
sorry

end employed_males_population_percentage_l467_467270


namespace infinite_solutions_xyz_l467_467732

theorem infinite_solutions_xyz (k : ℤ) : 
  let x := k * (2 * k^2 + 1),
      y := 2 * k^2 + 1,
      z := -k * (2 * k^2 + 1)
  in x ^ 2 + y ^ 2 + z ^ 2 = x ^ 3 + y ^ 3 + z ^ 3 := 
by sorry

end infinite_solutions_xyz_l467_467732


namespace solve_fx_eq_negative_three_l467_467112

def f (x : ℝ) : ℝ :=
  if x > 0 then -3^x else 1 - x^2

theorem solve_fx_eq_negative_three :
  (f 1 = -3) ∧ (f (-2) = -3) ∧ (∀ x, f x = -3 → (x = 1 ∨ x = -2)) := 
by 
  intros 
  sorry

end solve_fx_eq_negative_three_l467_467112


namespace distance_between_parallel_lines_l467_467902

-- Definitions and conditions
def line1 : LinearEquation := ⟨3, 4, -12⟩
def line2 (a : ℝ) : LinearEquation := ⟨a, 8, 11⟩
def are_parallel (l1 l2 : LinearEquation) : Prop :=
  l1.A * l2.B = l2.A * l1.B

-- Theorem statement
theorem distance_between_parallel_lines (a : ℝ) (h : a = 6) :
  are_parallel line1 (line2 a) →
  distance_between_lines line1 (line2 a) = 7 / 2 :=
by
  assume h_parallel
  have h_a : a = 6 := h
  sorry

end distance_between_parallel_lines_l467_467902


namespace range_of_a_l467_467105

def point := (ℝ × ℝ)

noncomputable def distance_to_line (P : point) (a b c : ℝ) : ℝ :=
  (abs (a * fst P + b * snd P + c)) / (sqrt (a^2 + b^2))

theorem range_of_a (a : ℝ) :
  let P : point := (4, a)
  let L : ℝ × ℝ × ℝ := (4, -3, -1)
  distance_to_line P 4 (-3) (-1) ≤ 3 → 0 ≤ a ∧ a ≤ 10 :=
by
  assume h
  sorry

end range_of_a_l467_467105


namespace arg_z_range_l467_467585

noncomputable def arg_range (z : ℂ) : Prop :=
  ∃ θ : ℝ, |θ| = real.pi / 6 ∧ θ = complex.arg ((z + 1) / (z + 2))

theorem arg_z_range {z : ℂ} (h : arg_range z) :
  real.arg z ∈ set.Icc (5 * real.pi / 6 - real.arcsin (√3 / 3)) (7 * real.pi / 6 + real.arcsin (√3 / 3)) :=
sorry

end arg_z_range_l467_467585


namespace number_of_pounds_of_vegetables_l467_467425

-- Defining the conditions
def beef_cost_per_pound : ℕ := 6  -- Beef costs $6 per pound
def vegetable_cost_per_pound : ℕ := 2  -- Vegetables cost $2 per pound
def beef_pounds : ℕ := 4  -- Troy buys 4 pounds of beef
def total_cost : ℕ := 36  -- The total cost of everything is $36

-- Prove the number of pounds of vegetables Troy buys is 6
theorem number_of_pounds_of_vegetables (V : ℕ) :
  beef_cost_per_pound * beef_pounds + vegetable_cost_per_pound * V = total_cost → V = 6 :=
by
  sorry  -- Proof to be filled in later

end number_of_pounds_of_vegetables_l467_467425


namespace equal_sets_l467_467442

def M : Set ℝ := {x | x^2 + 16 = 0}
def N : Set ℝ := {x | x^2 + 6 = 0}

theorem equal_sets : M = N := by
  sorry

end equal_sets_l467_467442


namespace goose_eggs_count_l467_467451

theorem goose_eggs_count (E : ℝ) (h1 : 1 / 4 * E = (1 / 4) * E)
  (h2 : 4 / 5 * (1 / 4) * E = (4 / 5) * (1 / 4) * E)
  (h3 : 3 / 5 * (4 / 5) * (1 / 4) * E = 120)
  (h4 : 120 = 120)
  : E = 800 :=
by
  sorry

end goose_eggs_count_l467_467451


namespace infinite_solutions_x2_y2_z2_x3_y3_z3_l467_467730

-- Define the parametric forms
def param_x (k : ℤ) := k * (2 * k^2 + 1)
def param_y (k : ℤ) := 2 * k^2 + 1
def param_z (k : ℤ) := -k * (2 * k^2 + 1)

-- Prove the equation
theorem infinite_solutions_x2_y2_z2_x3_y3_z3 :
  ∀ k : ℤ, param_x k ^ 2 + param_y k ^ 2 + param_z k ^ 2 = param_x k ^ 3 + param_y k ^ 3 + param_z k ^ 3 :=
by
  intros k
  -- Calculation needs to be proved here, we place a placeholder for now
  sorry

end infinite_solutions_x2_y2_z2_x3_y3_z3_l467_467730


namespace find_two_fractions_sum_eq_86_over_111_l467_467562

theorem find_two_fractions_sum_eq_86_over_111 :
  ∃ (a1 a2 d1 d2 : ℕ), 
    (0 < d1 ∧ d1 ≤ 100) ∧ 
    (0 < d2 ∧ d2 ≤ 100) ∧ 
    (nat.gcd a1 d1 = 1) ∧ 
    (nat.gcd a2 d2 = 1) ∧ 
    (↑a1 / ↑d1 + ↑a2 / ↑d2 = 86 / 111) ∧
    (a1 = 2 ∧ d1 = 3) ∧ 
    (a2 = 4 ∧ d2 = 37) := 
by
  sorry

end find_two_fractions_sum_eq_86_over_111_l467_467562


namespace meso_tyler_time_to_type_40_pages_l467_467311

-- Define the typing speeds
def meso_speed : ℝ := 15 / 5 -- 3 pages per minute
def tyler_speed : ℝ := 15 / 3 -- 5 pages per minute
def combined_speed : ℝ := meso_speed + tyler_speed -- 8 pages per minute

-- Define the number of pages to type
def pages : ℝ := 40

-- Prove the time required to type the pages together
theorem meso_tyler_time_to_type_40_pages : 
  ∃ (t : ℝ), t = pages / combined_speed :=
by
  use 5 -- this is the correct answer
  sorry

end meso_tyler_time_to_type_40_pages_l467_467311


namespace acute_angle_range_l467_467122

def vec_a (x : ℝ) : ℝ × ℝ := (1, x)
def vec_b (x : ℝ) : ℝ × ℝ := (2 * x + 3, -x)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def is_acute (x : ℝ) : Prop := dot_product (vec_a x) (vec_b x) > 0

theorem acute_angle_range:
  {x : ℝ // is_acute x} = {x : ℝ // -1 < x ∧ x < 0} ∪ {x : ℝ // 0 < x ∧ x < 3} :=
sorry

end acute_angle_range_l467_467122


namespace sum_lcms_equals_l467_467804

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_lcms_equals :
  ( ∑ ν in (Finset.filter is_solution (Finset.range 100)), ν ) = 180 :=
sorry

end sum_lcms_equals_l467_467804


namespace subset_singleton_zero_l467_467628

-- Variable declarations
variable {P : Set ℝ}

-- Definition of the set P
def P := {x : ℝ | x > -1}

-- Theorem statement
theorem subset_singleton_zero : {0} ⊆ P :=
by
  -- We leave the proof as a placeholder
  sorry

end subset_singleton_zero_l467_467628


namespace geometric_progression_general_formula_l467_467518

variable (a : ℕ → ℕ)

axiom a_1 : a 1 = 1
axiom a_rec : ∀ n, a (n + 1) = 2 * a n + 1

theorem geometric_progression : ∀ n, (a (n + 1) + 1) / (a n + 1) = 2 := by sorry

theorem general_formula : a = λ n, 2^n - 1 := by sorry

end geometric_progression_general_formula_l467_467518


namespace problem1_y_in_terms_of_x_problem2_area_problem3_volume_l467_467281

-- Definitions of the curve
def curve_x (t : ℝ) := Real.sin t
def curve_y (t : ℝ) := Real.sin (2 * t)

-- Problem 1: Express y in terms of x
theorem problem1_y_in_terms_of_x (x : ℝ) (h₀ : 0 ≤ t ∧ t ≤ Real.pi / 2) :
  ∃ t : ℝ, x = Real.sin t ∧ y = 2 * x * Real.sqrt (1 - x^2) := by
  sorry

-- Problem 2: Find the area enclosed by the x-axis and the curve C
theorem problem2_area :
  ∫ x in 0..1, 2 * x * Real.sqrt (1 - x^2) = 2 / 3 := by
  sorry

-- Problem 3: Find the volume of the solid generated by the rotation of D about the y-axis
theorem problem3_volume :
  2 * Real.pi * ∫ x in 0..1, x * 2 * x * Real.sqrt (1 - x^2) = 8 * Real.pi / 15 := by
  sorry

end problem1_y_in_terms_of_x_problem2_area_problem3_volume_l467_467281


namespace jones_trip_times_equal_l467_467695

theorem jones_trip_times_equal (v : ℝ) (hv : v > 0) :
  let t1 := 100 / v,
      t3 := 100 / v
  in t3 = t1 :=
by
  intros
  sorry

end jones_trip_times_equal_l467_467695


namespace find_divisor_l467_467963

theorem find_divisor (d : ℕ) (n : ℕ) (least : ℕ)
  (h1 : least = 2)
  (h2 : n = 433124)
  (h3 : ∀ d : ℕ, (d ∣ (n + least)) → d = 2) :
  d = 2 := 
sorry

end find_divisor_l467_467963


namespace find_minimum_angle_l467_467960

noncomputable def minimum_angle: ℝ :=
  840 * (π / 180)

theorem find_minimum_angle (A : ℝ) : 
  (∃ A, A ∈ {240, 480, 720, 840} ∧ ∀ B : ℝ, cos(B / 2) + sqrt(3) * sin(B / 2) ≥ cos(A / 2) + sqrt(3) * sin(A / 2)) ↔ (A = minimum_angle) :=
by
  sorry

end find_minimum_angle_l467_467960


namespace relationship_y1_y2_y3_l467_467843

noncomputable def parabola_value (x m : ℝ) : ℝ := -x^2 - 4 * x + m

variable (m y1 y2 y3 : ℝ)

def point_A_on_parabola : Prop := y1 = parabola_value (-3) m
def point_B_on_parabola : Prop := y2 = parabola_value (-2) m
def point_C_on_parabola : Prop := y3 = parabola_value 1 m


theorem relationship_y1_y2_y3 (hA : point_A_on_parabola y1 m)
                              (hB : point_B_on_parabola y2 m)
                              (hC : point_C_on_parabola y3 m) :
  y2 > y1 ∧ y1 > y3 := 
  sorry

end relationship_y1_y2_y3_l467_467843


namespace anya_lost_games_l467_467034

def girl_count := 5
def anya_games := 4
def bella_games := 6
def valya_games := 7
def galya_games := 10
def dasha_games := 11
def total_games := 19

theorem anya_lost_games :
  (anya_games + bella_games + valya_games + galya_games + dasha_games) / 2 = total_games →
  ∀ games : list ℕ,
    games.length = total_games →
    (∀ g ∈ games, g > 0 ∧ g ≤ total_games) →
    (∀ g ∈ [anya_games, bella_games, valya_games, galya_games, dasha_games], g ≤ total_games) →
    ∀ a b : ℕ, a ≠ b →
    (anya_games, games.nth 3 = some 4) ∧
    (anya_games, games.nth 7 = some 8) ∧
    (anya_games, games.nth 11 = some 12) ∧
    (anya_games, games.nth 15 = some 16)
:= sorry

end anya_lost_games_l467_467034


namespace dot_product_result_l467_467129

def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (2, m)
def c : ℝ × ℝ := (7, 1)

def are_parallel (a b : ℝ × ℝ) : Prop := 
  a.1 * b.2 = a.2 * b.1

def dot_product (u v : ℝ × ℝ) : ℝ := 
  u.1 * v.1 + u.2 * v.2

theorem dot_product_result : 
  ∀ m : ℝ, are_parallel a (b m) → dot_product (b m) c = 10 := 
by
  sorry

end dot_product_result_l467_467129


namespace S_m_expression_l467_467994

noncomputable def arithmetic_seq (a b n : ℕ) : ℕ :=
  a + (n - 1) * b

noncomputable def geometric_seq (b a n : ℕ) : ℕ :=
  b * a^(n - 1)

theorem S_m_expression (a b : ℕ) (h_a_positive : 0 < a) (h_b_positive : 0 < b)
 (hab1 : arithmetic_seq a b 1 < geometric_seq b a 1)
 (hab2 : geometric_seq b a 1 < arithmetic_seq a b 2)
 (hab3 : arithmetic_seq a b 2 < geometric_seq b a 2)
 (hab4 : geometric_seq b a 2 < arithmetic_seq a b 3):
  ∃ m n : ℕ, 
    m > 0 ∧ n > 0 ∧ 
    arithmetic_seq a b (m + 1) = geometric_seq b a n ∧
    let S_m := (b * ((m - 1) * a^m - m * a^(m - 1) + 1)) / (a - 1)^2 in
    S_m = (b * ((m - 1) * a^m - m * a^(m - 1) + 1)) / (a - 1)^2 :=
sorry

end S_m_expression_l467_467994


namespace length_diagonal_AC_possibilities_l467_467568

noncomputable def number_of_possible_lengths (AB BC CD DA : ℝ) (x : ℝ) : ℕ :=
  if (AB = 9 ∧ BC = 11 ∧ CD = 15 ∧ DA = 13) ∧
     (2 < x ∧ x < 20) 
  then fintype.card (set.Icc 3 19) else 0

theorem length_diagonal_AC_possibilities :
  number_of_possible_lengths 9 11 15 13 x = 17 :=
by sorry

end length_diagonal_AC_possibilities_l467_467568


namespace value_of_M_in_equation_l467_467824

theorem value_of_M_in_equation :
  ∀ {M : ℕ}, (32 = 2^5) ∧ (8 = 2^3) → (32^3 * 8^4 = 2^M) → M = 27 :=
by
  intros M h1 h2
  sorry

end value_of_M_in_equation_l467_467824


namespace MrYadavAnnualSavings_l467_467314

variable (S : ℝ)
variable (consumables_fraction : ℝ := 0.60)
variable (clothes_transport_fraction : ℝ := 0.50)
variable (clothes_transport_amount : ℕ := 2052)

theorem MrYadavAnnualSavings :
  (0.50 * (0.40 * S) = clothes_transport_amount) →
  ∃ (annual_savings : ℝ), annual_savings = 24624 :=
by {
  intro h,
  have hS : S = clothes_transport_amount / (0.50 * 0.40), sorry,
  let monthly_savings := 0.20 * S,
  let annual_savings := 12 * monthly_savings,
  use annual_savings,
  rw [monthly_savings, hS],
  field_simp [clothes_transport_amount],
  ring_nf,
  exact sorry
}

end MrYadavAnnualSavings_l467_467314


namespace surface_area_ratio_l467_467474

theorem surface_area_ratio (a : ℝ) (h1 : a > 0) :
  let b := a * (Real.sqrt 2 / 4) in
  let S1 := 2 * a^2 * Real.sqrt 3 in
  let S2 := 24 * b^2 in
  S1 / S2 = 2 * (Real.sqrt 3) / 3 := 
by
  let b := a * (Real.sqrt 2 / 4)
  let S1 := 2 * a^2 * Real.sqrt 3
  let S2 := 24 * b^2
  sorry

end surface_area_ratio_l467_467474


namespace a_n_proof_b_n_expression_c_n_bound_l467_467307

open BigOperators

-- Define the given vectors and conditions
def vector_a (x : ℝ) : ℝ × ℝ := (x, 2)
def vector_b (x n : ℝ) : ℝ × ℝ := (x + n, 2 * x - 1)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def y (x n : ℝ) : ℝ := dot_product (vector_a x) (vector_b x n)

noncomputable def a_n (n : ℕ+) : ℝ := 
  let sum_min_max := y 0 n + y 1 n in
  sum_min_max

-- Given summation condition
def summation_condition (n : ℕ+) : ℝ := 
  ∑ k in Finset.range n, (9 / 10 : ℝ)^k

-- Define b_n according to the solution
noncomputable def b_n : ℕ+ → ℝ
| ⟨1, _⟩ := 1
| ⟨n+1, hn⟩ := - (1 / 10) * (9 / 10)^(n-1)

-- Define c_n according to the solution
def c_n (n : ℕ+) : ℝ := -a_n n * b_n n

-- Prove the mathematical statements
theorem a_n_proof (n : ℕ+) : a_n n = n + 1 := by
  sorry

theorem b_n_expression (n : ℕ+) : b_n n = 
  if n = 1 then 1 else - (1 / 10) * (9 / 10)^(n-2) := by
  sorry

theorem c_n_bound (n : ℕ+) : ∃ k : ℕ+, ∀ m : ℕ+, c_n m ≤ c_n k := by
  let k := if n = 1 then 8 else 9
  use k
  sorry

end a_n_proof_b_n_expression_c_n_bound_l467_467307


namespace ab_value_l467_467098

theorem ab_value (a b : ℝ) (i : ℂ) (h : i = complex.I ∧ (2 - i) * (a - b * i) = (-8 - i) * i): a * b = 42 :=
by
  cases h with hi hc
  have hi_expr := hc
  sorry -- The proof steps are omitted as directed.

end ab_value_l467_467098


namespace arithmetic_sequence_minimization_l467_467681

theorem arithmetic_sequence_minimization (a b : ℕ) (h_range : 1 ≤ a ∧ b ≤ 17) (h_seq : a + b = 18) (h_min : ∀ x y, (1 ≤ x ∧ y ≤ 17 ∧ x + y = 18) → (1 / x + 25 / y) ≥ (1 / a + 25 / b)) : ∃ n : ℕ, n = 9 :=
by
  -- We'd usually follow by proving the conditions and defining the sequence correctly.
  -- Definitions and steps leading to finding n = 9 will be elaborated here.
  -- This placeholder is to satisfy the requirement only.
  sorry

end arithmetic_sequence_minimization_l467_467681


namespace problem1_problem2_l467_467082

/-- Problem 1: If c = 0, prove that a_(n+1)^2 > a_n^2. -/
theorem problem1 (a_n a_(n+1) S_n : ℕ → ℝ) (h_nonzero : ∀ n, a_n n ≠ 0) 
    (h_condition : ∀ n, sqrt (S_n n) = 2 * a_n n) : 
    ∀ n, (a_n (n+1))^2 > (a_n n)^2 :=
by
  sorry

/-- Problem 2: Determine if there exists a constant c such that {a_n} forms an arithmetic sequence 
and find all possible values of c if it exists. -/
theorem problem2 (a_n S_n : ℕ → ℝ) (h_nonzero : ∀ n, a_n n ≠ 0) 
    (h_arithmetic : ∃ d, ∀ n, a_n n = a_n 1 + n * d) : 
    ∃ c, (∀ n, sqrt (S_n n) = 2 * a_n n + c) ∧ c = 1/8 :=
by
  sorry

end problem1_problem2_l467_467082


namespace exists_irreducible_fractions_sum_to_86_over_111_l467_467559

theorem exists_irreducible_fractions_sum_to_86_over_111 :
  ∃ (a b p q : ℕ), (0 < a) ∧ (0 < b) ∧ (p ≤ 100) ∧ (q ≤ 100) ∧ (Nat.coprime a p) ∧ (Nat.coprime b q) ∧ (a * q + b * p = 86 * (p * q) / 111) :=
by
  sorry

end exists_irreducible_fractions_sum_to_86_over_111_l467_467559


namespace solve_eq_l467_467521

theorem solve_eq (a b : ℕ) : a * a = b * (b + 7) ↔ (a, b) = (0, 0) ∨ (a, b) = (12, 9) :=
by
  sorry

end solve_eq_l467_467521


namespace series_sum_l467_467931

theorem series_sum :
  ∑' n : ℕ, (2 ^ n) / (3 ^ (2 ^ n) + 1) = 1 / 2 :=
sorry

end series_sum_l467_467931


namespace num_factors_1320_l467_467184

theorem num_factors_1320 : 
  let n := 1320
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 11
  let e1 := 3
  let e2 := 1
  let e3 := 1
  let e4 := 1
  factors_count (p1^e1 * p2^e2 * p3^e3 * p4^e4) = 32 := 
by
  let prime_factors := [(p1, e1), (p2, e2), (p3, e3), (p4, e4)]
  sorry

end num_factors_1320_l467_467184


namespace no_valid_middle_number_l467_467395

theorem no_valid_middle_number
    (x : ℤ)
    (h1 : (x % 2 = 1))
    (h2 : 3 * x + 12 = x^2 + 20) :
    false :=
by
    sorry

end no_valid_middle_number_l467_467395


namespace anya_game_losses_l467_467058

theorem anya_game_losses (games : ℕ → ℕ) (anya_games bella_games valya_games galya_games dasha_games : ℕ)
  (H_total : anya_games = 4 ∧ bella_games = 6 ∧ valya_games = 7 ∧ galya_games = 10 ∧ dasha_games = 11) :
  (games anya_games = 4 ∧ games bella_games = 6 ∧
   games valya_games = 7 ∧ games galya_games = 10 ∧
   games dasha_games = 11) →  [4, 8, 12, 16] := sorry

end anya_game_losses_l467_467058


namespace anya_game_losses_l467_467064

theorem anya_game_losses (games : ℕ → ℕ) (anya_games bella_games valya_games galya_games dasha_games : ℕ)
  (H_total : anya_games = 4 ∧ bella_games = 6 ∧ valya_games = 7 ∧ galya_games = 10 ∧ dasha_games = 11) :
  (games anya_games = 4 ∧ games bella_games = 6 ∧
   games valya_games = 7 ∧ games galya_games = 10 ∧
   games dasha_games = 11) →  [4, 8, 12, 16] := sorry

end anya_game_losses_l467_467064


namespace average_decrease_l467_467744

theorem average_decrease (observations : Fin 6 → ℝ) (new_observation : ℝ) :
  (∑ i, observations i) / 6 = 14 →
  new_observation = 7 →
  ∑ i, observations i / 6 - (∑ i, observations i + new_observation) / 7 = 1 := by
sorry

end average_decrease_l467_467744


namespace probability_of_receiving_1_l467_467669

-- Define the probabilities and events
def P_A : ℝ := 0.5
def P_not_A : ℝ := 0.5
def P_B_given_A : ℝ := 0.9
def P_not_B_given_A : ℝ := 0.1
def P_B_given_not_A : ℝ := 0.05
def P_not_B_given_not_A : ℝ := 0.95

-- The main theorem that needs to be proved
theorem probability_of_receiving_1 : 
  (P_A * P_not_B_given_A + P_not_A * P_not_B_given_not_A) = 0.525 := by
  sorry

end probability_of_receiving_1_l467_467669


namespace series_sum_half_l467_467920

theorem series_sum_half :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1/2 := 
sorry

end series_sum_half_l467_467920


namespace max_siskins_on_poles_l467_467889

-- Define the conditions
def total_poles : ℕ := 25

def adjacent (i j : ℕ) : Prop := (abs (i - j) = 1)

-- Define the main problem
theorem max_siskins_on_poles (h₁ : 0 < total_poles) 
  (h₂ : ∀ (i : ℕ), i ≥ 1 ∧ i ≤ total_poles → ∀ (j : ℕ), j ≥ 1 ∧ j ≤ total_poles ∧ adjacent i j 
    → ¬ (siskin_on i ∧ siskin_on j)) :
  ∃ (max_siskins : ℕ), max_siskins = 24 := 
begin
  sorry
end

end max_siskins_on_poles_l467_467889


namespace complex_number_equality_l467_467372

theorem complex_number_equality :
  let z := (1 - complex.i)^(2 * complex.i)
  (1 - complex.i)^2 = -2 * complex.i →
  z = 2 :=
by
  intros
  sorry

end complex_number_equality_l467_467372


namespace distinct_factors_1320_l467_467182

theorem distinct_factors_1320 :
  let p := 1320,
  prime_factors := [(2, 2), (3, 1), (5, 1), (11, 1)],
  num_factors := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in num_factors = 24 := by
  sorry

end distinct_factors_1320_l467_467182


namespace smallest_possible_perimeter_l467_467482

open Real

theorem smallest_possible_perimeter
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a^2 + b^2 = 2016) :
  a + b + 2^3 * 3 * sqrt 14 = 48 + 2^3 * 3 * sqrt 14 :=
sorry

end smallest_possible_perimeter_l467_467482


namespace find_age_of_b_l467_467447

-- Definitions for the conditions
def is_two_years_older (a b : ℕ) : Prop := a = b + 2
def is_twice_as_old (b c : ℕ) : Prop := b = 2 * c
def total_age (a b c : ℕ) : Prop := a + b + c = 12

-- Proof statement
theorem find_age_of_b (a b c : ℕ) 
  (h1 : is_two_years_older a b) 
  (h2 : is_twice_as_old b c) 
  (h3 : total_age a b c) : 
  b = 4 := 
by 
  sorry

end find_age_of_b_l467_467447


namespace find_two_fractions_sum_eq_86_over_111_l467_467563

theorem find_two_fractions_sum_eq_86_over_111 :
  ∃ (a1 a2 d1 d2 : ℕ), 
    (0 < d1 ∧ d1 ≤ 100) ∧ 
    (0 < d2 ∧ d2 ≤ 100) ∧ 
    (nat.gcd a1 d1 = 1) ∧ 
    (nat.gcd a2 d2 = 1) ∧ 
    (↑a1 / ↑d1 + ↑a2 / ↑d2 = 86 / 111) ∧
    (a1 = 2 ∧ d1 = 3) ∧ 
    (a2 = 4 ∧ d2 = 37) := 
by
  sorry

end find_two_fractions_sum_eq_86_over_111_l467_467563


namespace parallel_lines_condition_l467_467998

noncomputable theory
open_locale classical

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, ax - 2y + 2 = 0 ↔ x + (a - 3)y + 1 = 0) ↔ a = 1 :=
sorry

end parallel_lines_condition_l467_467998


namespace if_2_3_4_then_1_if_1_3_4_then_2_l467_467841

variables {Plane Line : Type} 
variables (α β : Plane) (m n : Line)

-- assuming the perpendicular relationships as predicates
variable (perp : Plane → Plane → Prop) -- perpendicularity between planes
variable (perp' : Line → Line → Prop) -- perpendicularity between lines
variable (perp'' : Line → Plane → Prop) -- perpendicularity between line and plane

theorem if_2_3_4_then_1 :
  perp α β → perp'' m β → perp'' n α → perp' m n :=
by
  sorry

theorem if_1_3_4_then_2 :
  perp' m n → perp'' m β → perp'' n α → perp α β :=
by
  sorry

end if_2_3_4_then_1_if_1_3_4_then_2_l467_467841


namespace magnitude_of_b_l467_467128

noncomputable def vector_magnitude (x y : ℝ) : ℝ :=
  real.sqrt (x^2 + y^2)

theorem magnitude_of_b
  (y : ℝ)
  (a1 a2 : ℝ)
  (b1 b2 : ℝ)
  (h₀ : a1 = 1)
  (h₁ : a2 = 2)
  (h₂ : b1 = -2)
  (h₃ : b2 = y)
  (h₄ : (1 * -2) + (2 * y) = 0) : 
  vector_magnitude (-2) y = real.sqrt 5 := 
sorry

end magnitude_of_b_l467_467128


namespace find_x_l467_467765

-- Definitions of the conditions
def a (x : ℝ) := x - real.sqrt 2
def b (x : ℝ) := x - 1 / x
def c (x : ℝ) := x + 1 / x
def d (x : ℝ) := x^2 + 2 * real.sqrt 2

-- Theorem statement
theorem find_x :
  ∃ x : ℝ, (∃! n : ℤ, a x ∉ ℤ) ∧ a x = x - real.sqrt 2 ∧ b x = x - 1 / x ∧ c x = x + 1 / x ∧ d x = x^2 + 2 * real.sqrt 2 ∧ x = real.sqrt 2 - 1 :=
sorry

end find_x_l467_467765


namespace anya_lost_games_l467_467056

-- Define the girls playing table tennis
def girls : Type := {Anya Bella Valya Galya Dasha : girls}

-- Define the conditions
def games_played : girls → ℕ
| Anya => 4
| Bella => 6
| Valya => 7
| Galya => 10
| Dasha => 11

-- Total number of games played
def total_games : ℕ := 19
#eval total_games = (games_played Anya + games_played Bella + games_played Valya + games_played Galya + games_played Dasha) / 2

-- The main theorem to be proven
theorem anya_lost_games :
  ∀ i : ℕ, i ∈ {4, 8, 12, 16} ↔ Anya played and lost in game i
:=
by
  sorry

end anya_lost_games_l467_467056


namespace phi_value_l467_467655

theorem phi_value (phi : ℝ) :
  (∀ x, f x = (λ x, Real.sin (2 * x + phi)) x) ∧
  |phi| < Real.pi / 2 ∧
  (∀ x, (λ x, Real.sin (2 * (x + Real.pi / 6) + phi)) (-x) = -(λ x, Real.sin (2 * (x + Real.pi / 6) + phi)) x) →
  phi = -Real.pi / 3 :=
by
  sorry

end phi_value_l467_467655


namespace distinct_four_digit_numbers_count_l467_467151

theorem distinct_four_digit_numbers_count (digits : Finset ℕ) (h : digits = {1, 2, 3, 4, 5}) :
  (∃ (numbers : Finset (ℕ × ℕ × ℕ × ℕ)), 
   (∀ (a b c d : ℕ), (a, b, c, d) ∈ numbers → a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
   numbers.card = 120) :=
begin
  sorry
end

end distinct_four_digit_numbers_count_l467_467151


namespace product_mod_7_l467_467903

-- Declare the sequence terms and their cycle properties
def seq_20_terms := [-2, 8, -18, 28, -38, 48, -58, 68, -78, 88, -98, 108, -118, 128, -138, 148, -158, 168, -178, 188]
def mod_7_rems := λ x: Int, x % 7

-- Statement of the proof problem
theorem product_mod_7 : 
  mod_7_rems (List.foldl (*) 1 seq_20_terms) = 2 := by
  sorry

end product_mod_7_l467_467903


namespace num_distinct_factors_1320_l467_467165

theorem num_distinct_factors_1320 : 
  ∃ n : ℕ, n = 1320 ∧ (finset.univ.filter (λ d, d > 0 ∧ 1320 % d = 0)).card = 32 := 
sorry

end num_distinct_factors_1320_l467_467165


namespace positive_difference_of_R_coords_l467_467785

theorem positive_difference_of_R_coords :
    ∀ (xR yR : ℝ),
    ∃ (k : ℝ),
    (∀ (A B C R S : ℝ × ℝ), 
    A = (-1, 6) ∧ B = (1, 2) ∧ C = (7, 2) ∧ 
    R = (k, -0.5 * k + 5.5) ∧ S = (k, 2) ∧
    (0.5 * |7 - k| * |0.5 * k - 3.5| = 8)) → 
    |xR - yR| = 1 :=
by
  sorry

end positive_difference_of_R_coords_l467_467785


namespace stuart_segments_l467_467356

/-- Stuart draws a pair of concentric circles and chords of the large circle
    each tangent to the small circle.
    Given the angle ABC is 60 degrees, prove that he draws 3 segments
    before returning to his starting point. -/
theorem stuart_segments (angle_ABC : ℝ) (h : angle_ABC = 60) : 
  let n := 3 in n = 3 :=
sorry

end stuart_segments_l467_467356


namespace min_cost_rapid_advance_l467_467366

def rapidAdvanceMinCost (people : Nat) : Nat :=
  if people = 73 then 685 else sorry

-- Ensures we are working with the given conditions and can extend this later for more cases
variable (people cost : Nat)

-- Define the parameters for the two models
def sevenPersonRideCost := 65
def fivePersonRideCost := 50

-- Define that the tour group is exactly 73 people
def tourGroup := 73

-- Now, writing the proof statement
theorem min_cost_rapid_advance :
  (∀ rides_7 rides_5 : Nat, 7 * rides_7 + 5 * rides_5 = tourGroup → 
    rides_7 * sevenPersonRideCost + rides_5 * fivePersonRideCost ≥ cost) →
  cost = 685 :=
begin
  sorry
end

end min_cost_rapid_advance_l467_467366


namespace number_of_ways_to_form_groups_l467_467363

theorem number_of_ways_to_form_groups :
  let dogs := 10
  let group1 := 3
  let group2 := 5
  let group3 := 2
  (ℕ.choose 8 2) * (ℕ.choose 6 4) = 420 := by
{
  sorry
}

end number_of_ways_to_form_groups_l467_467363


namespace unique_n_divides_2_pow_n_minus_1_l467_467953

theorem unique_n_divides_2_pow_n_minus_1 (n : ℕ) (h : n ∣ 2^n - 1) : n = 1 :=
sorry

end unique_n_divides_2_pow_n_minus_1_l467_467953


namespace probability_negative_product_l467_467788

theorem probability_negative_product :
  let s := {-5, -8, 7, 4, -2, 1, 9} in
  ∃ p : ℚ, p = 4 / 7 ∧ 
  (∑ (x : ℤ) in s, ∑ (y : ℤ) in s, if x ≠ y ∧ x * y < 0 then 1 else 0) / 
  (∑ (x : ℤ) in s, ∑ (y : ℤ) in s, if x ≠ y then 1 else 0) = p := 
by
  -- Variables
  let s := {-5, -8, 7, 4, -2, 1, 9}
  -- Calculate total number of pairs
  have total_pairs := 7 * 6 / 2
  -- Calculate favorable pairs
  have favorable_pairs := 3 * 4
  -- Calculate probability
  have prob := favorable_pairs / total_pairs
  existsi (4 / 7 : ℚ)
  split
  · refl
  · field_simp
    norm_num
    sorry

end probability_negative_product_l467_467788


namespace unique_x_satisfying_conditions_l467_467761

theorem unique_x_satisfying_conditions :
  ∃ x : ℝ, x = Real.sqrt 2 - 1 ∧
  ((¬ ∃ (a b c d : ℤ), x - Real.sqrt 2 ∉ {a, b, c, d}) ∧
  (∃ (a b c : ℤ), ((x - Real.sqrt 2 = a ∨ x - 1/x = b ∨ x + 1/x = c) ∧ 
  (x^2 + 2 * Real.sqrt 2).isInteger))) :=
by sorry

end unique_x_satisfying_conditions_l467_467761


namespace find_fg_and_m_range_l467_467097

-- Definitions given in the problem
def is_even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_odd_function (g : ℝ → ℝ) := ∀ x, g x = -g (-x)

theorem find_fg_and_m_range 
  (f g : ℝ → ℝ)
  (h_even_f : is_even_function f)
  (h_odd_g : is_odd_function g)
  (h_eq : ∀ x, f x - g x = 2 ^ (1 - x))
  (m : ℝ) :
  (f = λ x, 2^x + 2 ^ (-x)) ∧ 
  (g = λ x, 2^x - 2 ^ (-x)) ∧ 
  (mf x = (g x) ^ 2 + 2 * m + 9 → m ≥ 10) :=
sorry

end find_fg_and_m_range_l467_467097


namespace Simson_line_parallel_to_OQ_l467_467727

-- Definitions of all the geometrical entities and given conditions
variables (A B C P Q O : Type) [circle : Circle O]
variables (α β γ : Real)
variables 
  (angles : 
    angle_between_vecs O P O A = α ∧
    angle_between_vecs O P O B = β ∧
    angle_between_vecs O P O C = γ ∧
    angle_between_vecs O P O Q = (α + β + γ) / 2)

theorem Simson_line_parallel_to_OQ (A B C P Q O : Type) [Circle O]
  (α β γ : Real)
  (angles : 
    angle_between_vecs O P O A = α ∧
    angle_between_vecs O P O B = β ∧
    angle_between_vecs O P O C = γ ∧
    angle_between_vecs O P O Q = (α + β + γ) / 2) :
  Simson_line P A B C ∥ OQ :=
by
  -- The detailed proof is omitted
  sorry

end Simson_line_parallel_to_OQ_l467_467727


namespace max_roots_eqn_l467_467379

variable {R : Type*} [LinearOrderedField R] -- R is a linearly ordered field, i.e., a type for reals
variable (a b c : R) -- define a, b, c as real numbers
variable (x : R) -- define x as a real number

theorem max_roots_eqn (h : a ≠ 0) :
  ∃ n ≤ 8, ∀ z : ℂ, Polynomial.eval z (a • Polynomial.C x^2 + b • Polynomial.C (|x|) + Polynomial.C c) = 0 → z = x :=
sorry

end max_roots_eqn_l467_467379


namespace pairs_of_integers_sum_product_quotient_l467_467657

theorem pairs_of_integers_sum_product_quotient (x y : ℤ) (h : x + y + x - y + x * y + x / y = 100) : 
  finset.card {p : ℤ × ℤ | let (x, y) := p in x + y + x - y + x * y + x / y = 100} = 3 := 
sorry

end pairs_of_integers_sum_product_quotient_l467_467657


namespace no_nonnegative_integral_solutions_l467_467941

theorem no_nonnegative_integral_solutions :
  ¬ ∃ (x y : ℕ), (x^4 * y^4 - 14 * x^2 * y^2 + 49 = 0) ∧ (x + y = 10) :=
by
  sorry

end no_nonnegative_integral_solutions_l467_467941


namespace max_profit_price_range_for_minimum_profit_l467_467471

noncomputable def functional_relationship (x : ℝ) : ℝ :=
-10 * x^2 + 2000 * x - 84000

theorem max_profit :
  ∃ x, (∀ x₀, x₀ ≠ x → functional_relationship x₀ < functional_relationship x) ∧
  functional_relationship x = 16000 := 
sorry

theorem price_range_for_minimum_profit :
  ∀ (x : ℝ), 
  -10 * (x - 100)^2 + 16000 - 1750 ≥ 12000 → 
  85 ≤ x ∧ x ≤ 115 :=
sorry

end max_profit_price_range_for_minimum_profit_l467_467471


namespace theater_ticket_cost_l467_467490

theorem theater_ticket_cost
  (num_persons : ℕ) 
  (num_children : ℕ) 
  (num_adults : ℕ)
  (children_ticket_cost : ℕ)
  (total_receipts_cents : ℕ)
  (A : ℕ) :
  num_persons = 280 →
  num_children = 80 →
  children_ticket_cost = 25 →
  total_receipts_cents = 14000 →
  num_adults = num_persons - num_children →
  200 * A + (num_children * children_ticket_cost) = total_receipts_cents →
  A = 60 :=
by
  intros h_num_persons h_num_children h_children_ticket_cost h_total_receipts_cents h_num_adults h_eqn
  sorry

end theater_ticket_cost_l467_467490


namespace series_sum_l467_467925

theorem series_sum : 
  (∑ n : ℕ, (2^n) / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end series_sum_l467_467925


namespace distinct_factors_1320_l467_467213

theorem distinct_factors_1320 : 
  ∃ n : ℕ, n = 24 ∧ 
  let a := 2 in 
  let b := 1 in 
  let c := 1 in 
  let d := 1 in 
  1320 = 2^a * 3^b * 5^c * 11^d :=
begin
  sorry,
end

end distinct_factors_1320_l467_467213


namespace series_convergence_l467_467930

theorem series_convergence :
  ∑ n : ℕ, (2^n) / (3^(2^n) + 1) = 1 / 2 :=
by
  sorry

end series_convergence_l467_467930


namespace matrix_product_result_l467_467907

theorem matrix_product_result :
  let A := ![![3, 1], ![4, -2]],
      B := ![![5, -3], ![2, 4]],
      C := A ⬝ B
  in C = ![![17, -5], ![16, -20]] := 
by
  sorry

end matrix_product_result_l467_467907


namespace value_of_x_l467_467403

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 :=
by
  sorry

end value_of_x_l467_467403


namespace solve_inequality_l467_467351

theorem solve_inequality (x : ℝ) : (|2 * x - 1| < |x| + 1) ↔ (0 < x ∧ x < 2) :=
by
  sorry

end solve_inequality_l467_467351


namespace typing_time_together_l467_467313

def meso_typing_rate : ℕ := 3 -- pages per minute
def tyler_typing_rate : ℕ := 5 -- pages per minute
def pages_to_type : ℕ := 40 -- pages

theorem typing_time_together :
  (meso_typing_rate + tyler_typing_rate) * 5 = pages_to_type :=
by
  sorry

end typing_time_together_l467_467313


namespace sum_of_positive_integers_nu_lcm_72_l467_467811

theorem sum_of_positive_integers_nu_lcm_72:
  let ν_values := { ν | Nat.lcm ν 24 = 72 }
  ∑ ν in ν_values, ν = 180 := by
  sorry

end sum_of_positive_integers_nu_lcm_72_l467_467811


namespace relationship_depends_on_a_l467_467595

theorem relationship_depends_on_a (a : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) :
  (log a 2 + log a 10 = log a 20 ∧ log a 20 = log a 2 + log a 10) ∧
  (log a 36 = 2 * log a 6 ∧ 2 * log a 6 = log a 36) ∧
  (log a 36 < log a 20 ↔ 0 < a ∧ a < 1) ∧
  (log a 36 > log a 20 ↔ a > 1) :=
sorry

end relationship_depends_on_a_l467_467595


namespace proof_alpha_plus_beta_l467_467226

-- Given conditions as definitions
def alphaRange : Set ℝ := set.Icc (π / 4) π
def betaRange : Set ℝ := set.Icc π (3 * π / 2)
constant α β : ℝ
axiom sin_2alpha_eq_sqrt5_div_5 : Real.sin (2 * α) = Real.sqrt 5 / 5
axiom sin_beta_minus_alpha_eq_sqrt10_div_10 : Real.sin (β - α) = Real.sqrt 10 / 10
axiom alpha_in_range : α ∈ alphaRange
axiom beta_in_range : β ∈ betaRange

-- The proof statement
theorem proof_alpha_plus_beta : α + β = 7 * π / 4 := by
  sorry

end proof_alpha_plus_beta_l467_467226


namespace distinct_factors_1320_l467_467204

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end distinct_factors_1320_l467_467204


namespace distinct_four_digit_numbers_l467_467142

theorem distinct_four_digit_numbers : 
  {n : ℕ | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3, 4, 5} ∧ c ∈ {1, 2, 3, 4, 5} ∧ d ∈ {1, 2, 3, 4, 5}
  } = 120 :=
by
  sorry

end distinct_four_digit_numbers_l467_467142


namespace coefficient_of_x2_l467_467795

def f (x : ℝ) := 3 * x^3 - 4 * x^2 - 2 * x - 3
def g (x : ℝ) := 2 * x^2 + 3 * x - 4

theorem coefficient_of_x2 : 
  (f(x) * g(x)).coeff 2 = 4 := 
sorry

end coefficient_of_x2_l467_467795


namespace candies_total_l467_467308

-- Defining the given conditions
def LindaCandies : ℕ := 34
def ChloeCandies : ℕ := 28
def TotalCandies : ℕ := LindaCandies + ChloeCandies

-- Proving the total number of candies
theorem candies_total : TotalCandies = 62 :=
  by
    sorry

end candies_total_l467_467308


namespace compare_abc_l467_467645

theorem compare_abc :
  let a := Real.log 17
  let b := 3
  let c := Real.exp (Real.sqrt 2)
  a < b ∧ b < c :=
by
  sorry

end compare_abc_l467_467645


namespace distinct_factors_1320_l467_467211

theorem distinct_factors_1320 : 
  ∃ n : ℕ, n = 24 ∧ 
  let a := 2 in 
  let b := 1 in 
  let c := 1 in 
  let d := 1 in 
  1320 = 2^a * 3^b * 5^c * 11^d :=
begin
  sorry,
end

end distinct_factors_1320_l467_467211


namespace john_shirts_left_l467_467694

theorem john_shirts_left (initial_shirts : ℕ) 
  (additional_designer_shirts: ℕ)
  (free_shirts: ℕ)
  (donation_percentage: ℝ)
  (initial_shirts_eq: initial_shirts = 62)
  (additional_designer_shirts_eq: additional_designer_shirts = 4)
  (free_shirts_eq: free_shirts = 1)
  (donation_percentage_eq: donation_percentage = 35/100) :
  let total_shirts_before_donation := initial_shirts + additional_designer_shirts + free_shirts,
  to_nat ((donation_percentage * total_shirts_before_donation : ℚ) : ℝ |>.floor) = 23 →
  (total_shirts_before_donation - 23 = 44) :=
by 
  assume h,
  sorry

end john_shirts_left_l467_467694


namespace derivative_of_f_is_correct_l467_467374

-- Define the function y = x^2 * sin(x)
def f (x : ℝ) : ℝ := x^2 * sin x

-- Define the expected derivative
def f_prime_expected (x : ℝ) : ℝ := x^2 * cos x + 2 * x * sin x

-- Statement: the derivative of f is equal to the expected derivative
theorem derivative_of_f_is_correct : ∀ x : ℝ, deriv f x = f_prime_expected x :=
by
  sorry

end derivative_of_f_is_correct_l467_467374


namespace unique_x_satisfying_conditions_l467_467762

theorem unique_x_satisfying_conditions :
  ∃ x : ℝ, x = Real.sqrt 2 - 1 ∧
  ((¬ ∃ (a b c d : ℤ), x - Real.sqrt 2 ∉ {a, b, c, d}) ∧
  (∃ (a b c : ℤ), ((x - Real.sqrt 2 = a ∨ x - 1/x = b ∨ x + 1/x = c) ∧ 
  (x^2 + 2 * Real.sqrt 2).isInteger))) :=
by sorry

end unique_x_satisfying_conditions_l467_467762


namespace equation_of_ellipse_area_of_triangle_l467_467593

def ellipse_major_axis_length := 6
def ellipse_eccentricity := 1 / 3

theorem equation_of_ellipse (a b : ℝ) (h_a : a > b) (h_ellipse : ∀ x y, x^2 / (a^2) + y^2 / (b^2) = 1) :
  a = 3 ∧ b = 2 * Real.sqrt 2 ∧ h_ellipse x y = (x^2 / 9 + y^2 / 8 = 1) := sorry

theorem area_of_triangle (F1 F2 P : ℝ × ℝ) (h_F1 : F1 = (-1, 0)) (h_F2 : F2 = (1, 0))
  (h_P : P ∈ {x | ∃ y, ellipse_equation x y}) (angle_PF1F2 : ∠(P, F1, F2) = π / 3) :
  area (triangle P F1 F2) = 8 * Real.sqrt 3 / 5 := sorry

end equation_of_ellipse_area_of_triangle_l467_467593


namespace circumcircles_intersect_at_S_l467_467126

structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A B C : Point)

def symmetric_about (O : Point) (P Q : Point) : Prop :=
  ∃ R : Point, R.x = 2 * O.x - P.x ∧ R.y = 2 * O.y - P.y ∧ Q = R

variables {A B C A1 B1 C1 O S : Point}
variables {T1 T2 : Triangle}

-- Define the problem statement
def triangles_are_symmetric (A B C A1 B1 C1 O : Point) : Prop :=
  symmetric_about O A A1 ∧ symmetric_about O B B1 ∧ symmetric_about O C C1

-- Main theorem statement
theorem circumcircles_intersect_at_S
  (triangle1 triangle2 : Triangle)
  (h1 : triangles_are_symmetric A B C A1 B1 C1 O)
  (h2 : triangle1 = ⟨A, B, C⟩)
  (h3 : triangle2 = ⟨A1, B1, C1⟩) :
  ∃ S : Point, S ∈ (circumcircle ⟨A, B, C⟩) ∧ 
               S ∈ (circumcircle ⟨A1, B, C1⟩) ∧ 
               S ∈ (circumcircle ⟨A1, B1, C⟩) ∧ 
               S ∈ (circumcircle ⟨A, B1, C1⟩) :=
by
  sorry

end circumcircles_intersect_at_S_l467_467126


namespace necessary_but_not_sufficient_condition_l467_467089

variables {Point Line Plane : Type} 

-- Definitions for the problem conditions
def is_subset_of (a : Line) (α : Plane) : Prop := sorry
def parallel_plane (a : Line) (β : Plane) : Prop := sorry
def parallel_lines (a b : Line) : Prop := sorry
def parallel_planes (α β : Plane) : Prop := sorry

-- The statement of the problem
theorem necessary_but_not_sufficient_condition (a b : Line) (α β : Plane) 
  (h1 : is_subset_of a α) (h2 : is_subset_of b β) :
  (parallel_plane a β ∧ parallel_plane b α) ↔ 
  (¬ parallel_planes α β ∧ sorry) :=
sorry

end necessary_but_not_sufficient_condition_l467_467089


namespace num_distinct_factors_1320_l467_467161

theorem num_distinct_factors_1320 : 
  ∃ n : ℕ, n = 1320 ∧ (finset.univ.filter (λ d, d > 0 ∧ 1320 % d = 0)).card = 32 := 
sorry

end num_distinct_factors_1320_l467_467161


namespace limit_of_n_b_n_l467_467000

noncomputable def L (x : ℝ) : ℝ := x - x^2 / 2

noncomputable def b_n (n : ℕ) : ℝ := 
  let L_iter := λ m x, (nat.recOn m x (λ _ acc, L acc))
  L_iter n (20 / n)

theorem limit_of_n_b_n : ∀ ε > 0, ∃ N, ∀ n > N, abs (n * b_n n - 40 / 19) < ε :=
sorry

end limit_of_n_b_n_l467_467000


namespace distinct_factors_1320_l467_467212

theorem distinct_factors_1320 : 
  ∃ n : ℕ, n = 24 ∧ 
  let a := 2 in 
  let b := 1 in 
  let c := 1 in 
  let d := 1 in 
  1320 = 2^a * 3^b * 5^c * 11^d :=
begin
  sorry,
end

end distinct_factors_1320_l467_467212


namespace triangle_area_inequality_l467_467279

variables {A B C D E F : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (ABC : Triangle A B C) (D : Point A B) (E : Point A C) (F : Point D E)
variables (x y z : ℝ)

noncomputable def area (t : Triangle A B C) : ℝ := sorry  -- Placeholder for area function

theorem triangle_area_inequality 
  (hD : x = AD / AB) 
  (hE : y = AE / AC) 
  (hF : z = DF / DE) : 
  (area (Triangle B D F) = (1 - x) * y * (area ABC)) ∧ 
  (area (Triangle C E F) = x * (1 - y) * (1 - z) * (area ABC)) ∧ 
  (real.cbrt (area (Triangle B D F)) + real.cbrt (area (Triangle C E F)) ≤ real.cbrt (area ABC)) :=
begin
  sorry
end

end triangle_area_inequality_l467_467279


namespace sum_lcms_equals_l467_467806

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_lcms_equals :
  ( ∑ ν in (Finset.filter is_solution (Finset.range 100)), ν ) = 180 :=
sorry

end sum_lcms_equals_l467_467806


namespace anya_game_losses_l467_467062

theorem anya_game_losses (games : ℕ → ℕ) (anya_games bella_games valya_games galya_games dasha_games : ℕ)
  (H_total : anya_games = 4 ∧ bella_games = 6 ∧ valya_games = 7 ∧ galya_games = 10 ∧ dasha_games = 11) :
  (games anya_games = 4 ∧ games bella_games = 6 ∧
   games valya_games = 7 ∧ games galya_games = 10 ∧
   games dasha_games = 11) →  [4, 8, 12, 16] := sorry

end anya_game_losses_l467_467062


namespace three_solutions_l467_467459

theorem three_solutions (a : ℝ) :
  (a = 0 ∨ a = 5 ∨ a = 9) ↔
  (∃ x1 x2 x3 : ℝ, 
    (x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ∧ 
    ∀ x, (sqrt (x - 1) * (abs (x^2 - 10 * x + 16) - a) = 0) → 
      (a * x^2 - 7 * x^2 - 10 * a * x + 70 * x + 21 * a - 147 = 0)) :=
begin
  sorry 
end

end three_solutions_l467_467459


namespace equation_of_line_l467_467090

-- Given
def P : ℝ × ℝ := (2, 4)

-- To Prove
theorem equation_of_line
  (passes_through_P : ∀ (a b : ℝ), a * (P.1) + b * (P.2) = 0)
  (intercepts_condition : ∀ (a b : ℝ), (a ≠ 0) ∧ (b/a = -1)) :
  ∃ (a b c : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ 
                 (a * P.1 + b * P.2 + c = 0) ∧ 
                 ((a * x + b * y + c = 0) → (2x - y = 0 ∨ x - y + 2 = 0)) :=
by sorry

end equation_of_line_l467_467090


namespace find_fractions_l467_467549

noncomputable def fractions_to_sum_86_111 : Prop :=
  ∃ (a b d₁ d₂ : ℕ), 0 < a ∧ 0 < b ∧ d₁ ≤ 100 ∧ d₂ ≤ 100 ∧
  Nat.gcd a d₁ = 1 ∧ Nat.gcd b d₂ = 1 ∧
  (a: ℚ) / d₁ + (b: ℚ) / d₂ = 86 / 111

theorem find_fractions : fractions_to_sum_86_111 :=
  sorry

end find_fractions_l467_467549


namespace range_of_lambda_l467_467599

noncomputable def f (x : ℝ) : ℝ := sorry
variables (x₁ x₂ λ : ℝ) (h_monotonic : ∀ a b, a < b → f a < f b ∨ f a > f b)
variables (h1 : x₁ < x₂) (h2 : λ ≠ -1)
def α : ℝ := (x₁ + λ * x₂) / (1 + λ)
def β : ℝ := (x₂ + λ * x₁) / (1 + λ)
variable (h3 : |f x₁ - f x₂| < |f α - f β|)

theorem range_of_lambda : λ < 0 :=
sorry

end range_of_lambda_l467_467599


namespace exist_triangle_with_given_altitudes_l467_467911

variables {R : Type*} [ordered_ring R]
variables {XY : line R} {B₁ C₁ : point R}

/-- there exists a triangle such that one side lies on a given line, 
with the given points being the feet of the altitudes to the other two sides -/
theorem exist_triangle_with_given_altitudes (XY : line R) (B₁ C₁ : point R) :
  ∃ (B C : point R), (B ∈ XY) ∧ (C ∈ XY) ∧
                     (is_perpendicular B₁ B C) ∧ (is_perpendicular C₁ C B) :=
sorry

end exist_triangle_with_given_altitudes_l467_467911


namespace majority_of_votes_l467_467664

def total_votes : ℕ := 4500
def percentage_winning : ℕ := 60

theorem majority_of_votes (total_votes : ℕ) (percentage_winning : ℕ) : ℕ :=
  let winning_votes := (percentage_winning * total_votes) / 100
  let other_votes := ((100 - percentage_winning) * total_votes) / 100
  winning_votes - other_votes

example : majority_of_votes total_votes percentage_winning = 900 := 
by {
  unfold majority_of_votes,
  simp,
  sorry
}

end majority_of_votes_l467_467664


namespace lower_denomination_cost_l467_467494

-- Conditions
def total_stamps : ℕ := 20
def total_cost_cents : ℕ := 706
def high_denomination_stamps : ℕ := 18
def high_denomination_cost : ℕ := 37
def low_denomination_stamps : ℕ := total_stamps - high_denomination_stamps

-- Theorem proving the cost of the lower denomination stamp.
theorem lower_denomination_cost :
  ∃ (x : ℕ), (high_denomination_stamps * high_denomination_cost) + (low_denomination_stamps * x) = total_cost_cents
  ∧ x = 20 :=
by
  use 20
  sorry

end lower_denomination_cost_l467_467494


namespace parallelepiped_properties_l467_467320

-- Define the structure of the parallelepiped and the given lengths
structure Parallelepiped :=
  (A B C D A1 B1 C1 D1 : Point)

-- Midpoint function required for defining center of the sphere
def midpoint (p1 p2 : Point) : Point := 
  {x := (p1.x + p2.x) / 2, 
   y := (p1.y + p2.y) / 2, 
   z := (p1.z + p2.z) / 2}

-- Given distances and properties
axiom BM_CM : ∀ (M C B : Point), distance B M = 1 ∧ distance C M = 3
axiom sphere_touches_planes : ∀ (C1 M : Point) (planes : List Plane), Sphere (midpoint C1 M) (distance C1 M / 2) touches planes -- Assumes the planes list represents the planes of the parallelepiped

-- Definitions of the properties to be proved
def length_of_AA1 (A A1 B C C1 M : Point) : ℝ := distance A A1
def radius_of_sphere (C1 M : Point) : ℝ := distance C1 M / 2
def volume_of_parallelepiped (BC BB1 AA1 : ℝ) : ℝ := BC * BB1 * AA1

-- The theorem statement
theorem parallelepiped_properties (p : Parallelepiped) (M : Point) :
  BM_CM M p.C p.B →
  sphere_touches_planes p.C1 M [plane1, plane2, plane3, plane4] →
  length_of_AA1 p.A p.A1 p.B p.C p.C1 M = 5 ∧
  radius_of_sphere p.C1 M = 2 ∧
  volume_of_parallelepiped (distance p.B p.C) (distance p.B1 p.B) 5 = 32 :=
begin 
  sorry
end

end parallelepiped_properties_l467_467320


namespace series_sum_l467_467921

theorem series_sum : 
  (∑ n : ℕ, (2^n) / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end series_sum_l467_467921


namespace a₅_equals_10_n_equals_100_l467_467591

def sequence_member (s t : ℕ) : ℕ := 2^s + 2^t

def is_valid_pair (s t : ℕ) : Prop := 0 ≤ s ∧ s < t ∧ t ∈ ℤ

noncomputable def sequence_a (n : ℕ) : ℕ :=
  list.nth_le (list.sort (≤) [sequence_member s t | s t : ℕ, is_valid_pair s t]) n sorry

theorem a₅_equals_10 : sequence_a 5 = 10 := sorry

theorem n_equals_100 (n : ℕ) (h : sequence_a n = 16640) : n = 100 := sorry

end a₅_equals_10_n_equals_100_l467_467591


namespace find_angle_ABC_l467_467272

variable (ABC : Type)
variable [IsTriangle ABC]
variables (A B C K : Point ABC)
variable (AB AC BK : Segment ABC)
variables (angle_BK_AB : Angle ABC)

-- Conditions
def is_median : Prop := midpoint K A C
def BK_half_AB : Prop := length BK = length AB / 2
def angle_BK_32 : Prop := measure angle_BK_AB = 32

-- Conclusion
def ABC_106 : Prop := measure angle A B C = 106

theorem find_angle_ABC : 
  is_median ABC A B C K →
  BK_half_AB ABC A B K AB AC BK →
  angle_BK_32 ABC A B K AB AC BK angle_BK_AB →
  ABC_106 ABC A B C :=
by
  intros h1 h2 h3
  sorry

end find_angle_ABC_l467_467272


namespace count_1320_factors_l467_467173

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end count_1320_factors_l467_467173


namespace conic_section_propositions_l467_467910

theorem conic_section_propositions:
  (M N : ℝ) (P : ℝ × ℝ) :
  ((M = (-2, 0)) ∧ (N = (2, 0)) ∧ (|P - M| + |P - N| = 3) → false) ∧
  (∀ a b : ℝ, is_hyperbola_with_focus_asymptote_distance (a, b)) ∧
  (foci_hyperbola_diff_foci_ellipse (16, 9)) ∧
  (∀ m : ℝ, m > 2 → roots_eccentricities (quadratic_eq_root m)) :=
sorry

/-- Definitions for supporting components -/

def is_hyperbola_with_focus_asymptote_distance (a b : ℝ) : Prop := sorry

def foci_hyperbola_diff_foci_ellipse (a b : ℝ) : Prop := sorry

def quadratic_eq_root (m : ℝ) : ℝ × ℝ := sorry

def roots_eccentricities (x y : ℝ) : Prop := sorry

end conic_section_propositions_l467_467910


namespace number_of_elements_of_E_l467_467699

def partitionable (n : ℕ) : Prop :=
  (3 < n) ∧ (n < 100) ∧ ((n * (n + 1) / 2) % 3 = 0)

noncomputable def E : Finset ℕ :=
  Finset.filter partitionable (Finset.range 100)

theorem number_of_elements_of_E : E.card = 64 :=
by
  sorry

end number_of_elements_of_E_l467_467699


namespace sum_lcms_equals_l467_467803

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_lcms_equals :
  ( ∑ ν in (Finset.filter is_solution (Finset.range 100)), ν ) = 180 :=
sorry

end sum_lcms_equals_l467_467803


namespace solve_x_l467_467914

def otimes (a b : ℝ) : ℝ := a - 3 * b

theorem solve_x : ∃ x : ℝ, otimes x 1 + otimes 2 x = 1 ∧ x = -1 :=
by
  use -1
  rw [otimes, otimes]
  sorry

end solve_x_l467_467914


namespace sort_rows_then_cols_remain_row_sorted_sort_cols_then_rows_is_different_l467_467829
-- Ensure everything necessary is imported

-- Part (a): Sorting rows first then columns keeps the row order
theorem sort_rows_then_cols_remain_row_sorted
  (A : Matrix ℕ ℕ ℝ)
  (h_row_sorted : ∀ i, Sorted (≤) (A i)) :
  let B := sortColumns (sortRows A)
  ∀ i, Sorted (≤) (B i) := by sorry

-- Part (b): Sorting columns first then rows results in different table
theorem sort_cols_then_rows_is_different
  (A : Matrix ℕ ℕ ℝ)
  (h_unsorted : ∃ i, ¬Sorted (≤) (A i) ∨ ∃ j, ¬Sorted (≤) (Aᵀ j)) :
  let B1 := sortRows (sortColumns A)
  let B2 := sortColumns (sortRows A)
  B1 ≠ B2 := by sorry

end sort_rows_then_cols_remain_row_sorted_sort_cols_then_rows_is_different_l467_467829


namespace num_factors_1320_l467_467187

theorem num_factors_1320 : 
  let n := 1320
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 11
  let e1 := 3
  let e2 := 1
  let e3 := 1
  let e4 := 1
  factors_count (p1^e1 * p2^e2 * p3^e3 * p4^e4) = 32 := 
by
  let prime_factors := [(p1, e1), (p2, e2), (p3, e3), (p4, e4)]
  sorry

end num_factors_1320_l467_467187


namespace min_operations_to_11_different_pieces_l467_467689

theorem min_operations_to_11_different_pieces (initial_pieces : ℕ) (H : initial_pieces = 111) :
  ∃ (ops : ℕ), ops = 2 ∧ ∃ (masses : Finset ℕ), masses.card = 11 ∧ ∀ (m ∈ masses) (n ∈ masses), m ≠ n :=
by
  sorry

end min_operations_to_11_different_pieces_l467_467689


namespace trains_cross_time_l467_467791

noncomputable def timeToCross (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let speed1_mps := speed1 * (5 / 18)
  let speed2_mps := speed2 * (5 / 18)
  let relative_speed := speed1_mps + speed2_mps
  let total_length := length1 + length2
  total_length / relative_speed

theorem trains_cross_time
  (length1 length2 : ℝ)
  (speed1 speed2 : ℝ)
  (h_length1 : length1 = 250)
  (h_length2 : length2 = 250)
  (h_speed1 : speed1 = 90)
  (h_speed2 : speed2 = 110) :
  timeToCross length1 length2 speed1 speed2 = 9 := 
by sorry

end trains_cross_time_l467_467791


namespace anya_lost_games_correct_l467_467049

-- Definition of players and their game counts
def Players := {Anya, Bella, Valya, Galya, Dasha}
def games_played : Players → ℕ
| Anya   := 4
| Bella  := 6
| Valya  := 7
| Galya  := 10
| Dasha  := 11

-- Total games played by all players
def total_games_played : ℕ := Players.to_finset.sum games_played

-- Total number of games, considering each game involves two players
def total_number_of_games : ℕ := total_games_played / 2

-- The set of games in which Anya lost
def anya_lost_games : set ℕ := {4, 8, 12, 16}

theorem anya_lost_games_correct :
  ∀ i ∈ anya_lost_games, -- For each game in anya_lost_games
  anya_lost_games = {4, 8, 12, 16} := 
by
  sorry

end anya_lost_games_correct_l467_467049


namespace median_inequality_l467_467529

variables (A B C M : Type) [triangle ABC]
variables (AB BC AC : ℝ) (BM : ℝ)

-- Definitions and conditions
-- \(BM\) is the median from vertex \(B\) to side \(AC\)
def isMedian : Prop := BM > (AB + BC - AC) / 2

-- The proof statement
theorem median_inequality (ABC_is_a_triangle: ∃ (A B C : ℝ), A + B + C = 180) : 
  isMedian :=
by 
  -- unfinished proof, hence "sorry"
  sorry

end median_inequality_l467_467529


namespace num_factors_1320_l467_467190

theorem num_factors_1320 : 
  let n := 1320
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 11
  let e1 := 3
  let e2 := 1
  let e3 := 1
  let e4 := 1
  factors_count (p1^e1 * p2^e2 * p3^e3 * p4^e4) = 32 := 
by
  let prime_factors := [(p1, e1), (p2, e2), (p3, e3), (p4, e4)]
  sorry

end num_factors_1320_l467_467190


namespace series_convergence_l467_467926

theorem series_convergence :
  ∑ n : ℕ, (2^n) / (3^(2^n) + 1) = 1 / 2 :=
by
  sorry

end series_convergence_l467_467926


namespace double_exceeds_one_fifth_by_nine_l467_467830

theorem double_exceeds_one_fifth_by_nine (x : ℝ) (h : 2 * x = (1 / 5) * x + 9) : x^2 = 25 :=
sorry

end double_exceeds_one_fifth_by_nine_l467_467830


namespace total_boys_l467_467723

noncomputable def total_students := 60

theorem total_boys (x : ℕ) (condition1 : x = total_students) (condition2 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ x) (condition3 : ∀ k, 0 < k ∧ k ≤ x) (condition4 : ∀ n : ℕ, boy_at (10 + 30 * n) = boy_at 40 ∧ even n = true) :
  x = 60 ∧ (x / 2) = 30 := 
by
  sorry

end total_boys_l467_467723


namespace single_elimination_games_needed_l467_467488

theorem single_elimination_games_needed (teams : ℕ) (h : teams = 19) : 
∃ games, games = 18 ∧ (∀ (teams_left : ℕ), teams_left = teams - 1 → games = teams - 1) :=
by
  -- define the necessary parameters and properties here 
  sorry

end single_elimination_games_needed_l467_467488


namespace general_term_T_n_formula_l467_467124

/-- Definition of the sequence {a_n} satisfying the recurrence relation -/
def seq_a : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * seq_a n + 1

/-- General term formula for the sequence {a_n} -/
theorem general_term (n : ℕ) : seq_a n = 2^n - 1 :=
sorry

/-- Definition of {c_n} -/
def c_n (n : ℕ) : ℚ :=
2^n / (seq_a n * seq_a (n + 1))

/-- Definition of {T_n} -/
def T_n (n : ℕ) : ℚ :=
∑ i in range (n + 1), c_n i

/-- Formula for {T_n} -/
theorem T_n_formula (n : ℕ) : T_n n = (2^(n + 1) - 2) / (2^(n + 1) - 1) :=
sorry

end general_term_T_n_formula_l467_467124


namespace count_1320_factors_l467_467171

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end count_1320_factors_l467_467171


namespace count_cosine_equals_neg_048_l467_467222

theorem count_cosine_equals_neg_048 : 
  ∃ (n : ℕ), 
  n = 2 ∧
  ∀ (x : ℝ), 
  0 ≤ x ∧ x < 360 → 
  (cos x = -0.48) → 
  n = 2 :=
by
  sorry

end count_cosine_equals_neg_048_l467_467222


namespace problem_expression_eval_l467_467748

theorem problem_expression_eval : (1 + 2 + 3) * (1 + 1/2 + 1/3) = 11 := by
  sorry

end problem_expression_eval_l467_467748


namespace arc_length_is_27_over_4_l467_467457

noncomputable def calc_arc_length : ℝ :=
  let x (t : ℝ) := 6 * (Real.cos t) ^ 3
  let y (t : ℝ) := 6 * (Real.sin t) ^ 3
  let dx_dt (t : ℝ) := -18 * (Real.cos t) ^ 2 * (Real.sin t)
  let dy_dt (t : ℝ) := 18 * (Real.sin t) ^ 2 * (Real.cos t)
  let integrand (t : ℝ) := Real.sqrt ((dx_dt t) ^ 2 + (dy_dt t) ^ 2)
  Real.integrate (0) (π / 3) integrand sorry

theorem arc_length_is_27_over_4 : calc_arc_length = 27 / 4 :=
sorry

end arc_length_is_27_over_4_l467_467457


namespace solve_system_l467_467355

theorem solve_system (X Y : ℝ) : 
  (X + (X + 2 * Y) / (X^2 + Y^2) = 2 ∧ Y + (2 * X - Y) / (X^2 + Y^2) = 0) ↔ (X = 0 ∧ Y = 1) ∨ (X = 2 ∧ Y = -1) :=
by
  sorry

end solve_system_l467_467355


namespace probability_of_receiving_one_l467_467667

noncomputable def probability_received_one : ℝ :=
let P_A := 0.5 in
let P_not_A := 0.5 in
let P_B_given_A := 0.9 in
let P_not_B_given_A := 0.1 in
let P_B_given_not_A := 0.05 in
let P_not_B_given_not_A := 0.95 in
let P_B := P_A * P_B_given_A + P_not_A * P_B_given_not_A in
1 - P_B

theorem probability_of_receiving_one :
  probability_received_one = 0.525 :=
by
  -- P_A = 0.5
  -- P_not_A = 0.5
  -- P_B_given_A = 0.9
  -- P_not_B_given_A = 0.1
  -- P_B_given_not_A = 0.05
  -- P_not_B_given_not_A = 0.95
  -- P_B = 0.5 * 0.9 + 0.5 * 0.05
  -- P_B = 0.45 + 0.025
  -- P_B = 0.475
  -- P_not_B = 1 - P_B
  -- P_not_B = 1 - 0.475
  -- P_not_B = 0.525
  sorry

end probability_of_receiving_one_l467_467667


namespace shortest_distance_is_1_l467_467391

-- Definition of the parametric curves
def C1 (θ : ℝ) : ℝ × ℝ × ℝ := (x1 θ, y1 θ, z1 θ)
def C2 (t : ℝ) : ℝ × ℝ × ℝ := (x2 t, y2 t, z2 t)

-- Euclidean distance between points on C1 and C2
def distance (θ t : ℝ) : ℝ :=
  real.sqrt ((fst (C1 θ) - fst (C2 t))^2 + (snd (C1 θ) - snd (C2 t))^2 + ((C1 θ).2 - (C2 t).2)^2)

-- The actual mathematical statement to prove
theorem shortest_distance_is_1 :
  ∃ θ t, distance θ t = 1 :=
sorry

end shortest_distance_is_1_l467_467391


namespace kaleb_chocolates_l467_467697

noncomputable def remaining_chocolates (initial_boxes : ℕ) (given_away_brother : ℕ) 
  (given_away_friend : ℕ) (given_away_classmates : ℕ) (pieces_per_box : ℕ) 
  (percent_eaten : ℝ) : ℕ :=
  let remaining_boxes := initial_boxes - (given_away_brother + given_away_friend + given_away_classmates) in
  let remaining_pieces := remaining_boxes * pieces_per_box in
  let eaten_pieces := floor (percent_eaten * remaining_pieces) in
  remaining_pieces - eaten_pieces

theorem kaleb_chocolates : remaining_chocolates 14 5 2 3 6 0.1 = 22 :=
by 
  sorry

end kaleb_chocolates_l467_467697


namespace convex_1990_gon_exists_l467_467004

theorem convex_1990_gon_exists :
  ∃ (P : List ℝ), P.Perms  (List.range' 1 1990) ∧ (∀ (i : Fin 1990), angle (P.nth_le i sorry) (P.nth_le (i + 1) % 1990 sorry) = 994 / 995 * π) :=
sorry

end convex_1990_gon_exists_l467_467004


namespace midpoint_condition_l467_467768

theorem midpoint_condition (c : ℝ) :
  (∃ A B : ℝ × ℝ,
    A ∈ { p : ℝ × ℝ | p.2 = p.1^2 - 2 * p.1 - 3 } ∧
    B ∈ { p : ℝ × ℝ | p.2 = p.1^2 - 2 * p.1 - 3 } ∧
    A ∈ { p : ℝ × ℝ | p.2 = -p.1^2 + 4 * p.1 + c } ∧
    B ∈ { p : ℝ × ℝ | p.2 = -p.1^2 + 4 * p.1 + c } ∧
    ((A.1 + B.1) / 2 + (A.2 + B.2) / 2) = 2017
  ) ↔
  c = 4031 := sorry

end midpoint_condition_l467_467768


namespace Anya_loss_games_l467_467022

noncomputable def gamesPlayed : Nat := 19

constant Anya_games : Nat := 4
constant Bella_games : Nat := 6
constant Valya_games : Nat := 7
constant Galya_games : Nat := 10
constant Dasha_games : Nat := 11

constant gameResults : Fin 19 → Option String
constant lost (g : Nat) (p : String) : Prop

axiom Anya_game_indices : ∀ (i : Fin 4), lost (gameResults i.val) "Anya" → i.val = 4 * i + 4

theorem Anya_loss_games : list Nat :=
have Anya_lost_games : list Nat := [4, 8, 12, 16],
Anya_lost_games

end Anya_loss_games_l467_467022


namespace range_of_a_fixed_point_l467_467580

open Function

def f (x a : ℝ) := x^3 - a * x

theorem range_of_a (a : ℝ) (h1 : 0 < a) : 0 < a ∧ a ≤ 3 ↔ ∀ x ≥ 1, 3 * x^2 - a > 0 :=
sorry

theorem fixed_point (a x0 : ℝ) (h_a : 0 < a) (h_b : a ≤ 3)
  (h1 : x0 ≥ 1) (h2 : f x0 a ≥ 1) (h3 : f (f x0 a) a = x0) (strict_incr : ∀ x y, x ≥ 1 → y ≥ 1 → x < y → f x a < f y a) :
  f x0 a = x0 :=
sorry

end range_of_a_fixed_point_l467_467580


namespace unique_non_integer_l467_467760

theorem unique_non_integer :
  let x := sqrt 2 - 1 in
  let a := x - sqrt 2 in
  let b := x - 1 / x in
  let c := x + 1 / x in
  let d := x^2 + 2 * sqrt 2 in
  (¬is_int a ∧ is_int b ∧ is_int c ∧ is_int d) ∨
  (is_int a ∧ ¬is_int b ∧ is_int c ∧ is_int d) ∨
  (is_int a ∧ is_int b ∧ ¬is_int c ∧ is_int d) ∨
  (is_int a ∧ is_int b ∧ is_int c ∧ ¬is_int d) :=
by sorry

end unique_non_integer_l467_467760


namespace distinct_factors_1320_l467_467209

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end distinct_factors_1320_l467_467209


namespace num_factors_1320_l467_467192

theorem num_factors_1320 : 
  let n := 1320
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 11
  let e1 := 3
  let e2 := 1
  let e3 := 1
  let e4 := 1
  factors_count (p1^e1 * p2^e2 * p3^e3 * p4^e4) = 32 := 
by
  let prime_factors := [(p1, e1), (p2, e2), (p3, e3), (p4, e4)]
  sorry

end num_factors_1320_l467_467192


namespace common_area_of_inscribed_equilateral_triangles_l467_467789

theorem common_area_of_inscribed_equilateral_triangles {r : ℝ} :
  let area_common := (λ r : ℝ, (r^2 * Real.sqrt 3) / 2) in
  ∃ (t₁ t₂ : Triangle), inscribed_in_circle t₁ r ∧ inscribed_in_circle t₂ r ∧
  equilateral t₁ ∧ equilateral t₂ ∧ 
  area (intersection t₁ t₂) ≥ area_common r :=
sorry

end common_area_of_inscribed_equilateral_triangles_l467_467789


namespace ratio_of_friends_l467_467346

theorem ratio_of_friends (friends_in_classes friends_in_clubs : ℕ) (thread_per_keychain total_thread : ℕ) 
  (h1 : thread_per_keychain = 12) (h2 : friends_in_classes = 6) (h3 : total_thread = 108)
  (keychains_total : total_thread / thread_per_keychain = 9) 
  (keychains_clubs : (total_thread / thread_per_keychain) - friends_in_classes = friends_in_clubs) :
  friends_in_clubs / friends_in_classes = 1 / 2 :=
by
  sorry

end ratio_of_friends_l467_467346


namespace magnitude_vector_difference_l467_467598

variables (e1 e2 : EuclideanSpace ℝ (Fin 2))

-- Conditions: unit vectors and angle between them is 45 degrees
def are_unit_vectors (v : EuclideanSpace ℝ (Fin 2)) := ∥v∥ = 1
def angle_45_degrees (v w : EuclideanSpace ℝ (Fin 2)) := real.inner_product_space.angle v w = real.pi / 4

-- Theorem statement
theorem magnitude_vector_difference :
  are_unit_vectors e1 ∧ are_unit_vectors e2 ∧ angle_45_degrees e1 e2 →
  ∥e1 - (real.sqrt 2) • e2∥ = 1 :=
by
  sorry

end magnitude_vector_difference_l467_467598


namespace exists_convex_1990_gon_with_equal_angles_and_specific_side_lengths_l467_467734

theorem exists_convex_1990_gon_with_equal_angles_and_specific_side_lengths :
  ∃ (P : polygon) (sides : fin 1990 → ℝ),
    (∀ i : fin 1990, angle_at_vertex P i = (1990 - 2) * 180 / 1990) ∧
    (∃ (perm : fin 1990 → fin 1990), 
      ∀ i : fin 1990, sides i = (perm i + 1)^2) ∧
    convex P ∧
    list.to_finset (list.of_fn sides) = 
    list.to_finset (list.of_fn (λ i : fin 1990, (i + 1)^2)) :=
sorry

end exists_convex_1990_gon_with_equal_angles_and_specific_side_lengths_l467_467734


namespace summation_series_equals_half_l467_467938

theorem summation_series_equals_half :
  (\sum_{n=0}^{∞} 2^n / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end summation_series_equals_half_l467_467938


namespace find_x_l467_467766

-- Definitions of the conditions
def a (x : ℝ) := x - real.sqrt 2
def b (x : ℝ) := x - 1 / x
def c (x : ℝ) := x + 1 / x
def d (x : ℝ) := x^2 + 2 * real.sqrt 2

-- Theorem statement
theorem find_x :
  ∃ x : ℝ, (∃! n : ℤ, a x ∉ ℤ) ∧ a x = x - real.sqrt 2 ∧ b x = x - 1 / x ∧ c x = x + 1 / x ∧ d x = x^2 + 2 * real.sqrt 2 ∧ x = real.sqrt 2 - 1 :=
sorry

end find_x_l467_467766


namespace num_factors_1320_l467_467191

theorem num_factors_1320 : 
  let n := 1320
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 11
  let e1 := 3
  let e2 := 1
  let e3 := 1
  let e4 := 1
  factors_count (p1^e1 * p2^e2 * p3^e3 * p4^e4) = 32 := 
by
  let prime_factors := [(p1, e1), (p2, e2), (p3, e3), (p4, e4)]
  sorry

end num_factors_1320_l467_467191


namespace find_two_fractions_sum_eq_86_over_111_l467_467561

theorem find_two_fractions_sum_eq_86_over_111 :
  ∃ (a1 a2 d1 d2 : ℕ), 
    (0 < d1 ∧ d1 ≤ 100) ∧ 
    (0 < d2 ∧ d2 ≤ 100) ∧ 
    (nat.gcd a1 d1 = 1) ∧ 
    (nat.gcd a2 d2 = 1) ∧ 
    (↑a1 / ↑d1 + ↑a2 / ↑d2 = 86 / 111) ∧
    (a1 = 2 ∧ d1 = 3) ∧ 
    (a2 = 4 ∧ d2 = 37) := 
by
  sorry

end find_two_fractions_sum_eq_86_over_111_l467_467561


namespace anya_lost_games_l467_467054

-- Define the girls playing table tennis
def girls : Type := {Anya Bella Valya Galya Dasha : girls}

-- Define the conditions
def games_played : girls → ℕ
| Anya => 4
| Bella => 6
| Valya => 7
| Galya => 10
| Dasha => 11

-- Total number of games played
def total_games : ℕ := 19
#eval total_games = (games_played Anya + games_played Bella + games_played Valya + games_played Galya + games_played Dasha) / 2

-- The main theorem to be proven
theorem anya_lost_games :
  ∀ i : ℕ, i ∈ {4, 8, 12, 16} ↔ Anya played and lost in game i
:=
by
  sorry

end anya_lost_games_l467_467054


namespace closest_whole_number_to_shaded_area_l467_467070

theorem closest_whole_number_to_shaded_area
  (π : ℝ)
  (rectangle_area : ℝ)
  (circle1_area : ℝ)
  (circle2_area : ℝ)
  (shaded_area : ℝ)
  (closest_integer : ℤ) :
  rectangle_area = 4 * 3 →
  circle1_area = π * (1 : ℝ) ^ 2 →
  circle2_area = π * (0.5 : ℝ) ^ 2 →
  shaded_area = rectangle_area - circle1_area - circle2_area →
  π = 3.14 →
  closest_integer = Int.toNat ⌊shaded_area⌉ →
  closest_integer = 8 :=
sorry

end closest_whole_number_to_shaded_area_l467_467070


namespace probability_deviation_l467_467770

def approx_prob : Prop :=
  let n : ℕ := 900
  let p : ℝ := 0.5
  let ε : ℝ := 0.02
  let φ (x : ℝ) : ℝ := (1 /2) * (1 + real.erf (x / real.sqrt 2))
  2 * φ (ε * real.sqrt (n / (p * (1 - p)))) ≈ 0.7698

theorem probability_deviation :
  approx_prob := by
  sorry

end probability_deviation_l467_467770


namespace no_solution_for_m_l467_467570

theorem no_solution_for_m (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (m : ℕ) (h3 : (ab)^2015 = (a^2 + b^2)^m) : false := 
sorry

end no_solution_for_m_l467_467570


namespace ratio_limit_l467_467258

noncomputable def area_trapezoid := sorry
noncomputable def area_triangle := sorry

theorem ratio_limit
  (O : Point)
  (r : ℝ)
  (AB XY : Chord)
  (θ : ℝ)
  (hAB_horizontal : horizontal AB)
  (hXY_inclined : inclined XY θ)
  (hO_collinear_PQ_cards : collinear O P Q R)
  (mid_AB_P : midpoint AB P)
  (S : ℝ := area_trapezoid AB XY)
  (T : ℝ := area_triangle OPY) :
  (filter.tendsto (λ (OP : ℝ), S / T) (𝓝 (r * cos θ)) (𝓝 (1/2 * (1 + tan θ)))) :=
begin
  sorry
end

end ratio_limit_l467_467258


namespace rationalize_denominator_l467_467341

theorem rationalize_denominator (a b c : ℝ) (h1 : a = 7) (h2 : b = √98) (h3 : √98 = 7 * √2) :
  a / b * √2 = c ↔ c = √2 / 2 := by
  sorry

end rationalize_denominator_l467_467341


namespace monotonically_increasing_intervals_gx_expression_and_range_l467_467117

-- Definitions for Part I
def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

-- Definitions for Part II
noncomputable def h (x : ℝ) : ℝ := 1 + 2 * Real.sin (x + π / 6)
noncomputable def g (x : ℝ) : ℝ := 1 + 2 * Real.sin (x - π / 6)
def interval_partition (k : ℤ) : Set ℝ := {x : ℝ | k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6}

-- Theorem Statements
theorem monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x ∈ interval_partition k, monotone (λ x, f x) :=
sorry 

theorem gx_expression_and_range :
  (∀ x ∈ (Set.Icc 0 π), g x = 1 + 2 * Real.sin (x - π / 6)) ∧
  Set.range (λ x, g x) = Set.Icc 0 3 :=
sorry

end monotonically_increasing_intervals_gx_expression_and_range_l467_467117


namespace count_numbers_with_digit_zero_l467_467641

theorem count_numbers_with_digit_zero :
  let count := (λ n, (to_digits 10 n).contains 0) in
  (finset.range 2501).count count = 591 := 
sorry

end count_numbers_with_digit_zero_l467_467641


namespace max_siskins_on_poles_l467_467873

theorem max_siskins_on_poles (n : ℕ) (h : n = 25) :
  ∃ k : ℕ, k = 24 ∧ (∀ (poless: Fin n → ℕ) (siskins: Fin n → ℕ),
     (∀ i: Fin n, siskins i ≤ 1) 
     ∧ (∀ i: Fin n, (siskins i = 1 → (poless i = 0)))
     ∧ poless 0 = 0
     → ( ∀ j: Fin n, (j < n → siskins j + siskins (j+1) < 2)) 
     ∧ (k ≤ n)
     ∧ ( ∀ l: Fin n, ((l < k → siskins l = 1) →
       ((k ≤ l < n → siskins l = 0)))))
  sorry

end max_siskins_on_poles_l467_467873


namespace circle_tangent_y_axis_l467_467587

noncomputable def parabola_focus (p : ℝ) (hp : 0 < p) : ℝ × ℝ :=
  (p / 2, 0)

noncomputable def circle_center (x₁ y₁ p : ℝ) : ℝ × ℝ :=
  ((2 * x₁ + p) / 4, y₁ / 2)

theorem circle_tangent_y_axis (p : ℝ) (hp : 0 < p) :
  ∀ (x₁ y₁ : ℝ), (x₁, y₁) ∈ {P : ℝ × ℝ | P.2^2 = 2 * p * P.1} →
  let c := circle_center x₁ y₁ p in
  let r := (2 * x₁ + p) / 4 in
  c.1 = r :=
begin
  intros x₁ y₁ hP c r,
  simp [circle_center, r],
  sorry
end

end circle_tangent_y_axis_l467_467587


namespace pigeons_in_house_l467_467263

variable (x F c : ℝ)

theorem pigeons_in_house 
  (H1 : F = (x - 75) * 20 * c)
  (H2 : F = (x + 100) * 15 * c) :
  x = 600 := by
  sorry

end pigeons_in_house_l467_467263


namespace range_of_g_l467_467543

noncomputable def g (x : ℝ) : ℝ := 
  let π2 := Real.pi * Real.pi in
  (19 * π2) / 48 + (π2 / 18) * (x - 3 / 2) ^ 2

theorem range_of_g :
  ∃ y ∈ Set.Ici ((19 * Real.pi * Real.pi) / 48), ∀ x, g x = y :=
sorry

end range_of_g_l467_467543


namespace intersection_M_N_correct_l467_467629

-- Definitions of sets M and N
def M : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def N : Set ℤ := {x : ℤ | -1 ≤ x ∧ x ≤ 2}

-- Correct Answer
def correct_answer : Set ℤ := {-1, 0}

-- Proof Statement
theorem intersection_M_N_correct : (M ∩ N : Set ℤ) = correct_answer := by
  sorry

end intersection_M_N_correct_l467_467629


namespace cos_transform_l467_467420

theorem cos_transform : 
  (∀ x, cos (2 * x + π) = cos (x + π / 2)) :=
by
  sorry

end cos_transform_l467_467420


namespace distinct_four_digit_numbers_count_l467_467133

theorem distinct_four_digit_numbers_count :
  (Finset.card (Finset.filter (λ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ (∀ d ∈ [x / 1000, (x / 100) % 10, (x / 10) % 10, x % 10], d ∈ [1, 2, 3, 4, 5]) ∧ (denote_distinct_digits (x)) ) (Finset.Ico 1000 10000))) = 120 := 
sorry

def denote_distinct_digits (x : ℕ) : Prop :=
  ∀ i j, (i ≠ j) → (x / (10 ^ i)) % 10 ≠ (x / (10 ^ j)) % 10

end distinct_four_digit_numbers_count_l467_467133


namespace parallel_lines_distance_l467_467528

theorem parallel_lines_distance (c : ℝ) :
  let L1 := λ (x y : ℝ), 3 * x - 2 * y - 1 = 0 in
  let L2 := λ (x y : ℝ), 3 * x - 2 * y + c = 0 in
  let distance := (λ (A B c1 c2 : ℝ), |c1 - c2| / (sqrt (A^2 + B^2))) in
  distance 3 (-2) (-1) c = 2 * sqrt 13 / 13 →
  c = 1 ∨ c = -3 :=
begin
  intros L1 L2 distance h,
  sorry
end

end parallel_lines_distance_l467_467528


namespace sum_of_seq_l467_467125

def seq_a (n : ℕ) : ℕ → ℕ
| 0 => 2
| m + 1 => (n + 1) * (seq_a m / m + 2)

def seq_b (n : ℕ) : ℕ :=
2 ^ (2 * seq_a n)

def seq_sum (n : ℕ) : ℕ :=
∑ k in finset.range (n + 1), seq_b k

theorem sum_of_seq (n : ℕ) : seq_sum n = 4 * (4^n - 1) / 3 :=
sorry

end sum_of_seq_l467_467125


namespace arithmetic_sequence_general_term_sum_sequence_result_l467_467995

-- Given definitions
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ a1 d, (a 1 = a1) ∧ (∀ n, a (n+1) = a n + d) ∧ (a 2 = 5) ∧ (5 * a1 + (5 * 4 * d) / 2 = 35)

def sum_of_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def sum_sequence (S : ℕ → ℤ) (T : ℕ → ℚ) : Prop :=
  ∀ n, T n = ∑ i in Finset.range n, 1 / (S (i + 1) - (i + 1))

-- Proof goals
theorem arithmetic_sequence_general_term :
  ∃ a : ℕ → ℤ, arithmetic_seq a ∧ ∀ n, a n = 2 * n + 1 :=
begin
  -- The proof would go here.
  sorry
end

theorem sum_sequence_result :
  ∃ S T, 
    (∃ a, arithmetic_seq a ∧ sum_of_terms a S) ∧
    sum_sequence S T ∧
    ∀ n, T n = n / (n + 1) :=
begin
  -- The proof would go here.
  sorry
end

end arithmetic_sequence_general_term_sum_sequence_result_l467_467995


namespace find_product_of_translated_roots_l467_467295

noncomputable def poly_roots_condition (x : ℂ) : Prop :=
  x^3 - 18 * x^2 + 20 * x - 8 = 0

-- Define the main theorem
theorem find_product_of_translated_roots (u v w : ℂ) (hu : poly_roots_condition u) (hv : poly_roots_condition v) (hw : poly_roots_condition w) :
  (2 + u) * (2 + v) * (2 + w) = 128 :=
by
  -- Use Vieta's formulas to convert provided conditions
  have H1 : u + v + w = 18 := by sorry
  have H2 : u * v + v * w + w * u = 20 := by sorry
  have H3 : u * v * w = 8 := by sorry
  -- Substitute these into the expanded form
  calc
    (2 + u) * (2 + v) * (2 + w) = 8 + 4 * (u + v + w) + 2 * (u * v + v * w + w * u) + (u * v * w) : by sorry
    ... = 8 + 4 * 18 + 2 * 20 + 8 : by { rw [H1, H2, H3] }
    ... = 8 + 72 + 40 + 8 : by sorry
    ... = 128 : by norm_num

end find_product_of_translated_roots_l467_467295


namespace distinct_factors_1320_l467_467203

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end distinct_factors_1320_l467_467203


namespace arithmetic_mean_first_term_of_innovated_degree_2_l467_467997

-- Given conditions
variables {n : ℕ} (A : Fin n → ℝ) 

-- Define max function over finite sets in Lean
def max_initial_segments (A : Fin n → ℝ) (k : Fin n) : ℝ :=
  Fin.fold (λ i acc, max acc (A i)) (A 0) (Fin.init k)

-- Define B tuple
def B (A : Fin n → ℝ) : Fin n → ℝ
  | ⟨k, _⟩ := max_initial_segments A k

-- Define innovated degree
def innovated_degree (A : Fin n → ℝ) : ℕ :=
  (Finset.image (λ k, B A k) Finset.univ).card

-- Arithmetic mean of first term of permutations whose innovated degrees equal 2
noncomputable def mean_of_first_term_with_degree_2 (n : ℕ) : ℚ :=
  n - (n - 1 : ℚ) / (HarmonicNumber (n - 1 : ℚ))

-- Theorem to prove the statement
theorem arithmetic_mean_first_term_of_innovated_degree_2 :
  mean_of_first_term_with_degree_2 n = n - (n - 1 : ℚ) / (HarmonicNumber (n - 1 : ℚ)) :=
sorry

end arithmetic_mean_first_term_of_innovated_degree_2_l467_467997


namespace num_ways_105_as_diff_of_squares_not_possible_106_as_diff_of_squares_l467_467465

-- Part (a) equivalent Lean statement:
theorem num_ways_105_as_diff_of_squares :
  {n : ℕ // ∃ x y : ℕ, x^2 - y^2 = 105} = {4} := 
sorry

-- Part (b) equivalent Lean statement:
theorem not_possible_106_as_diff_of_squares (x y : ℕ) :
  x^2 - y^2 = 106 → false :=
sorry

end num_ways_105_as_diff_of_squares_not_possible_106_as_diff_of_squares_l467_467465


namespace minimum_value_of_xy_l467_467099

theorem minimum_value_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = x * y) :
  x * y ≥ 8 :=
sorry

end minimum_value_of_xy_l467_467099


namespace exists_quadratic_satisfying_conditions_l467_467530

theorem exists_quadratic_satisfying_conditions :
  ∃ (a b c : ℝ), 
  (a - b + c = 0) ∧
  (∀ x : ℝ, x ≤ a * x^2 + b * x + c ∧ a * x^2 + b * x + c ≤ (1 + x^2) / 2) ∧ 
  (a = 1/4 ∧ b = 1/2 ∧ c = 1/4) :=
  sorry

end exists_quadratic_satisfying_conditions_l467_467530


namespace problem_statement_l467_467685

noncomputable def triangle_property (A B C P: Type) [is_triangle A B C] : Prop :=
  ∃ (α β: ℝ), 
    is_right_angle A C B ∧ 
    β > 45 ∧ 
    dist A B = 5 ∧ 
    dist C P = 2 ∧
    ∠ A P C = 2 * ∠ A C P

noncomputable def ratio_ap_bp (A B C P: Type) [is_triangle A B C] : ℝ :=
  (dist C B / 2) + (real.sqrt(1 + 4.5^2) / dist C B / 2)

theorem problem_statement :
  ∃ (p q r: ℕ),
    triangle_property ℝ ℝ ℝ ℝ →
      let ratio := 5 + 2 * real.sqrt(5.1875)
      ∧ (ratio = (p + q * real.sqrt r))
       ∧ (p + q + r = 51882) :=
sorry

end problem_statement_l467_467685


namespace coeff_x2_sum_l467_467002

-- Definition: Binomial coefficient
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Theorem: Coefficient of x^2 in the sum of expansions
theorem coeff_x2_sum : (∑ k in finset.range 8, binom (k + 2) 2) = 120 :=
by
  sorry

end coeff_x2_sum_l467_467002


namespace computer_price_increase_l467_467237

/-- If the price of a certain computer increased 30 percent from d dollars, 
    and 2d equals 580, what is the new price of the computer after the increase? -/
theorem computer_price_increase (d : ℝ) (hd : 2 * d = 580) (hinc : 0.30 * d) : 
  d + 0.30 * d = 377 :=
by
  sorry

end computer_price_increase_l467_467237


namespace max_min_M_l467_467579

noncomputable def M (x y : ℝ) : ℝ :=
  abs (x + y) + abs (y + 1) + abs (2 * y - x - 4)

theorem max_min_M (x y : ℝ) (hx : abs x ≤ 1) (hy : abs y ≤ 1) :
  3 ≤ M x y ∧ M x y ≤ 7 :=
sorry

end max_min_M_l467_467579


namespace quadratic_function_minimum_l467_467624

noncomputable def quadratic_function := 
  λ (a b : ℝ) (x : ℝ), a * x^2 + b * x + (b^2 / (2 * a))

theorem quadratic_function_minimum (a b : ℝ) (ha : a > 0) :
  ∃ x₀ y₀, y₀ = quadratic_function a b x₀ ∧ 
            (∀ x, quadratic_function a b x ≥ y₀) :=
begin
  sorry
end

end quadratic_function_minimum_l467_467624


namespace find_number_l467_467739

-- Define the conditions
variables (y : ℝ) (Some_number : ℝ) (x : ℝ)

-- State the given equation
def equation := 19 * (x + y) + Some_number = 19 * (-x + y) - 21

-- State the proposition to prove
theorem find_number (h : equation 1 y Some_number) : Some_number = -59 :=
sorry

end find_number_l467_467739


namespace arithmetic_sequence_sum_l467_467255

theorem arithmetic_sequence_sum (a_n : ℕ → ℝ) (h1 : a_n 1 + a_n 2 + a_n 3 + a_n 4 = 30) 
                               (h2 : a_n 1 + a_n 4 = a_n 2 + a_n 3) :
  a_n 2 + a_n 3 = 15 := 
by 
  sorry

end arithmetic_sequence_sum_l467_467255


namespace min_M_inequality_l467_467969

noncomputable def M_min : ℝ := 9 * Real.sqrt 2 / 32

theorem min_M_inequality :
  ∀ (a b c : ℝ),
    abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2))
    ≤ M_min * (a^2 + b^2 + c^2)^2 :=
by
  sorry

end min_M_inequality_l467_467969


namespace equal_segments_l467_467280

variables {A B C D E F G : Type*}
variables [triangle A B C] [midpoint D B C] [altitude E D C] 
variables [circumcircle F A B D] [intersection G DE AF]

theorem equal_segments (h1 : |AB| = |AC|) (h2 : |D| = midpoint(BC)) (h3 : |E| = foot_of_d_perpendicular(AC)) 
  (h4 : |F| = intersection_of_circumcircle_and_BE(B, ABD)) (h5 : |G| = intersection(DE, AF)) :
  |DG| = |GE| := 
sorry

end equal_segments_l467_467280


namespace percent_increase_l467_467895

theorem percent_increase (s1 : ℝ) (factor : ℝ) (s4 : ℝ) (p_increase : ℝ) :
  s1 = 3 → factor = 1.75 →
  let s2 := s1 * factor in
  let s3 := s2 * factor in
  let s4 := s3 * factor in
  let p1 := 3 * s1 in
  let p4 := 3 * s4 in
  p4 = 3 * 16.078125 →
  p_increase = ((p4 - p1) / p1) * 100 →
  p_increase = 435.9 :=
by
  intros h1 h2 h3 h4 h5;
  sorry

end percent_increase_l467_467895


namespace anya_lost_games_correct_l467_467048

-- Definition of players and their game counts
def Players := {Anya, Bella, Valya, Galya, Dasha}
def games_played : Players → ℕ
| Anya   := 4
| Bella  := 6
| Valya  := 7
| Galya  := 10
| Dasha  := 11

-- Total games played by all players
def total_games_played : ℕ := Players.to_finset.sum games_played

-- Total number of games, considering each game involves two players
def total_number_of_games : ℕ := total_games_played / 2

-- The set of games in which Anya lost
def anya_lost_games : set ℕ := {4, 8, 12, 16}

theorem anya_lost_games_correct :
  ∀ i ∈ anya_lost_games, -- For each game in anya_lost_games
  anya_lost_games = {4, 8, 12, 16} := 
by
  sorry

end anya_lost_games_correct_l467_467048


namespace Ivan_defeats_Koschei_l467_467640

-- Definitions of the springs and conditions based on the problem
section

variable (S: ℕ → Prop)  -- S(n) means the water from spring n
variable (deadly: ℕ → Prop)  -- deadly(n) if water from nth spring is deadly

-- Conditions
axiom accessibility (n: ℕ): (1 ≤ n ∧ n ≤ 9 → ∀ i: ℕ, S i)
axiom koschei_access: S 10
axiom lethality (n: ℕ): (S n → deadly n)
axiom neutralize (i j: ℕ): (1 ≤ i ∧ i < j ∧ j ≤ 9 → ∃ k: ℕ, S k ∧ k > j → ¬deadly i)

-- Statement to prove
theorem Ivan_defeats_Koschei:
  ∃ i: ℕ, (1 ≤ i ∧ i ≤ 9) → (S 10 → ¬deadly i) ∧ (S 0 ∧ (S 10 → deadly 0)) :=
sorry

end

end Ivan_defeats_Koschei_l467_467640


namespace sum_of_max_and_min_on_interval_l467_467773

def f (x : ℝ) : ℝ := (1 / 2) ^ x

theorem sum_of_max_and_min_on_interval : 
  let a := 0
  let b := 1
  let I := set.Icc a b
  (∀ x ∈ I, f(x) ≤ 1) →
  (∀ x ∈ I, f(x) ≥ 1 / 2) →
  (f(a) + f(b) = 3 / 2) :=
by
  sorry

end sum_of_max_and_min_on_interval_l467_467773


namespace summation_series_equals_half_l467_467936

theorem summation_series_equals_half :
  (\sum_{n=0}^{∞} 2^n / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end summation_series_equals_half_l467_467936


namespace exists_irreducible_fractions_sum_to_86_over_111_l467_467560

theorem exists_irreducible_fractions_sum_to_86_over_111 :
  ∃ (a b p q : ℕ), (0 < a) ∧ (0 < b) ∧ (p ≤ 100) ∧ (q ≤ 100) ∧ (Nat.coprime a p) ∧ (Nat.coprime b q) ∧ (a * q + b * p = 86 * (p * q) / 111) :=
by
  sorry

end exists_irreducible_fractions_sum_to_86_over_111_l467_467560


namespace max_siskins_on_poles_l467_467884

/-- 
Given 25 poles in a row, where only one siskin can occupy any given pole, 
and if a siskin lands on a pole, a siskin sitting on an adjacent pole will immediately fly away, 
the maximum number of siskins that can simultaneously be on the poles is 24.
-/
theorem max_siskins_on_poles : 
  ∀ (poles : Fin 25 → Bool), 
  (∀ i, poles i = false ∨ ∃ j, (abs (i - j) = 1 ∧ poles j = false))
  → ∃ n, (0 < n ∧ n ≤ 25 ∧ max_siskins poles n = 24) :=
by
  sorry

end max_siskins_on_poles_l467_467884


namespace projectile_first_hits_30m_l467_467378

noncomputable def height (t : ℝ) : ℝ := -4.9 * t^2 + 26.4 * t

theorem projectile_first_hits_30m (t₀ : ℝ) (h₀ : height t₀ = 30) : t₀ = 1.633 :=
by
  sorry

end projectile_first_hits_30m_l467_467378


namespace andrew_donates_160_to_homeless_shelter_l467_467501

/-- Andrew's bake sale earnings -/
def totalEarnings : ℕ := 400

/-- Amount kept by Andrew for ingredients -/
def ingredientsCost : ℕ := 100

/-- Amount Andrew donates from his own piggy bank -/
def piggyBankDonation : ℕ := 10

/-- The total amount Andrew donates to the homeless shelter -/
def totalDonationToHomelessShelter : ℕ :=
  let remaining := totalEarnings - ingredientsCost
  let halfDonation := remaining / 2
  halfDonation + piggyBankDonation

theorem andrew_donates_160_to_homeless_shelter : totalDonationToHomelessShelter = 160 := by
  sorry

end andrew_donates_160_to_homeless_shelter_l467_467501


namespace explicit_formula_and_unique_zero_l467_467075

noncomputable def f (a b x : ℝ) : ℝ :=
  a * x ^ 2 - (b + 1) * x * Math.log x - b

theorem explicit_formula_and_unique_zero (a b : ℝ) (h1 : 2 * e * a = 2 * (b + 1)) (h2 : -a * e ^ 2 + e * b + e - b = 0) :
  f 1 e x = x^2 - (e + 1) * x * Math.log x - e ∧ (0 ≤ x ∧ x ≤ e^4 → f 1 e x = 0 → x = e ^ 4) := sorry

end explicit_formula_and_unique_zero_l467_467075


namespace perpendicular_A1C1_AC_l467_467301

variables {A B C P A1 C1 : Type*} [metric_space A] [metric_space B] [metric_space C]
variables [metric_space P] [metric_space A1] [metric_space C1]

open_locale classical
noncomputable theory

def cyclic (A B C P : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space P] : Prop :=
  ∃ (circle : circle A B C), P ∈ circle

def tangent (b : Type*) [line b] (circle : circle) (B : Type*) [metric_space B] : Prop :=
  ∃ (tangent_point : B), tangent_point ∈ b ∧ tangent_point ∈ circle ∧ is_tangent b circle tangent_point

def perpendicular (L1 L2 : line) : Prop :=
  ∃ (P : Type*) [metric_space P], P ∈ L1 ∧ P ∈ L2 ∧ ∠ (L1, P, L2) = π / 2

axiom perpendiculars_dropped_to_lines
  (A B C P A1 C1 : Type*) [metric_space A] [metric_space B] [metric_space C]
  [metric_space P] [metric_space A1] [metric_space C1] :
  ∃ (line_AB line_BC : line),
    A ∈ line_AB ∧ B ∈ line_AB ∧ B ∈ line_BC ∧ C ∈ line_BC ∧
    (∃ (line_b : line), tangent line_b (circle B) B ∧ P ∈ line_b) ∧
    ∃ (PA1 PC1 : line),
      perpendicular P A1 line_AB ∧ perpendicular P C1 line_BC

theorem perpendicular_A1C1_AC
  (A B C P A1 C1 : Type*) [metric_space A] [metric_space B] [metric_space C]
  [metric_space P] [metric_space A1] [metric_space C1]
  (h1 : cyclic A B C P)
  (h2 : tangent P (circle B) B)
  (h3 : perpendiculars_dropped_to_lines A B C P A1 C1) :
  perpendicular A1 C1 AC :=
sorry

end perpendicular_A1C1_AC_l467_467301


namespace total_games_played_l467_467741

-- Define the conditions
variables (games_won : ℕ) (game_fraction_needed : ℚ) (games_left : ℕ) (games_needed_to_win : ℕ)

-- Assume values based on the conditions
def conditions := 
  games_won = 12 ∧
  game_fraction_needed = 2 / 3 ∧
  games_left = 10 ∧
  games_needed_to_win = 8

-- Calculate the total games by setting up the equation and solving
def total_games_needed (game_fraction_needed : ℚ) (total_wins : ℕ) : ℕ :=
  ((total_wins : ℚ) / game_fraction_needed).toNat

-- Prove the answer based on the conditions
theorem total_games_played (games_won : ℕ) (game_fraction_needed : ℚ) (games_left : ℕ) (games_needed_to_win : ℕ) :
  conditions games_won game_fraction_needed games_left games_needed_to_win →
  let total_wins := games_won + games_needed_to_win in
  let total_games := total_games_needed game_fraction_needed total_wins in
  (total_games - games_left) = 20 :=
begin
  intro h,
  rcases h with ⟨h1, h2, h3, h4⟩,
  simp *,
  sorry
end

end total_games_played_l467_467741


namespace rose_apples_l467_467343

variable x : ℕ -- let x denote the number of apples

-- Given conditions as assumptions
def num_friends := 3
def apples_per_friend := 3

theorem rose_apples : x = num_friends * apples_per_friend :=
by simp [num_friends, apples_per_friend]; exact sorry

end rose_apples_l467_467343


namespace anya_lost_games_l467_467030

def girl_count := 5
def anya_games := 4
def bella_games := 6
def valya_games := 7
def galya_games := 10
def dasha_games := 11
def total_games := 19

theorem anya_lost_games :
  (anya_games + bella_games + valya_games + galya_games + dasha_games) / 2 = total_games →
  ∀ games : list ℕ,
    games.length = total_games →
    (∀ g ∈ games, g > 0 ∧ g ≤ total_games) →
    (∀ g ∈ [anya_games, bella_games, valya_games, galya_games, dasha_games], g ≤ total_games) →
    ∀ a b : ℕ, a ≠ b →
    (anya_games, games.nth 3 = some 4) ∧
    (anya_games, games.nth 7 = some 8) ∧
    (anya_games, games.nth 11 = some 12) ∧
    (anya_games, games.nth 15 = some 16)
:= sorry

end anya_lost_games_l467_467030


namespace solution_triples_l467_467516

noncomputable def validate_triples (x y z : ℝ) : Prop :=
  (x * y + z = 40) ∧ (x * z + y = 51) ∧ (x + y + z = 19)

theorem solution_triples :
  {p : ℝ × ℝ × ℝ | validate_triples p.1 p.2 p.3} =
  {⟨12, 3, 4⟩, ⟨6, 5.4, 7.6⟩} :=
by { sorry }

end solution_triples_l467_467516


namespace negation_of_universal_proposition_l467_467973

theorem negation_of_universal_proposition (x : ℝ) :
  ¬ (∀ m : ℝ, 0 ≤ m ∧ m ≤ 1 → x + 1 / x ≥ 2^m) ↔ ∃ m : ℝ, (0 ≤ m ∧ m ≤ 1) ∧ (x + 1 / x < 2^m) := by
  sorry

end negation_of_universal_proposition_l467_467973


namespace Anya_loss_games_l467_467027

noncomputable def game_indices_Anya_lost (Anya Bella Valya Galya Dasha : ℕ) : Set ℕ :=
  {4, 8, 12, 16}

theorem Anya_loss_games
  (Anya Bella Valya Galya Dasha total_games : ℕ)
  (h1 : Anya = 4)
  (h2 : Bella = 6)
  (h3 : Valya = 7)
  (h4 : Galya = 10)
  (h5 : Dasha = 11)
  (h6 : total_games = (Anya + Bella + Valya + Galya + Dasha) / 2)
  (h7 : total_games = 19) :
  game_indices_Anya_lost Anya Bella Valya Galya Dasha = {4, 8, 12, 16} :=
  by sorry

end Anya_loss_games_l467_467027


namespace zeros_of_derivative_interior_l467_467588

noncomputable def polynomial
def is_jordan_curve (L : set ℂ) : Prop :=
  is_compact L ∧ is_connected L ∧ ∀ z ∈ L, ∃ U, is_open U ∧ z ∈ U ∧ ∀ y ∈ U ∩ L, y ≠ z

theorem zeros_of_derivative_interior (P : polynomial ℂ) 
  (L : set ℂ) (hL : L = {z : ℂ | ∥P.eval z∥ = 1} ∧ is_jordan_curve L) :
  ∀ z₀ ∈ L, P.derivative.eval z₀ = 0 → ∃ ε > 0, ∀ y, dist z₀ y < ε → (∥P.eval y∥ < 1) :=
sorry

end zeros_of_derivative_interior_l467_467588


namespace find_two_irreducible_fractions_l467_467551

theorem find_two_irreducible_fractions :
  ∃ (a b d1 d2 : ℕ), 
    (1 ≤ a) ∧ 
    (1 ≤ b) ∧ 
    (gcd a d1 = 1) ∧ 
    (gcd b d2 = 1) ∧ 
    (1 ≤ d1) ∧ 
    (d1 ≤ 100) ∧ 
    (1 ≤ d2) ∧ 
    (d2 ≤ 100) ∧ 
    (a / (d1 : ℚ) + b / (d2 : ℚ) = 86 / 111) := 
by {
  sorry
}

end find_two_irreducible_fractions_l467_467551


namespace bob_number_of_cats_l467_467901

noncomputable def total_dogs : ℕ := 7
noncomputable def food_percent_cat : ℝ := 0.03125

theorem bob_number_of_cats (D C T : ℝ) (x : ℕ)
  (h1: ∀ i j : ℕ, i < total_dogs → j < total_dogs → D = D)
  (h2: ∀ i j : ℕ, i < x → j < x → C = C)
  (h3: x * C = D)
  (h4: C = food_percent_cat * T)
  (h5: T = total_dogs * D) : x ≈ 5 :=
sorry

end bob_number_of_cats_l467_467901


namespace sum_of_leading_digits_l467_467700

def N : ℕ := 777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777

def leading_digit (n : ℝ) : ℕ :=
  match n.to_digits with
  | c::_ => c.digit_to_nat
  | _ => 0  -- this case should not occur for positive n

noncomputable def f (r : ℕ) : ℕ :=
  leading_digit (N^(1 / r : ℝ))

theorem sum_of_leading_digits :
  f 2 + f 3 + f 4 + f 5 + f 6 = 8 := by
  sorry

end sum_of_leading_digits_l467_467700


namespace solve_inequality_l467_467352

theorem solve_inequality (x : ℝ) : (|2 * x - 1| < |x| + 1) ↔ (0 < x ∧ x < 2) :=
by
  sorry

end solve_inequality_l467_467352


namespace value_of_x_l467_467402

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 :=
by
  sorry

end value_of_x_l467_467402


namespace rationalize_denominator_l467_467332

theorem rationalize_denominator (a b c : ℝ) (h : b ≠ 0) (h0 : 98 = c * c) (h1 : 7 = c) :
  (7 / (Real.sqrt 98) = (Real.sqrt 2) / 2) :=
by
  sorry

end rationalize_denominator_l467_467332


namespace factorial_expression_l467_467906

theorem factorial_expression : (8.factorial + 9.factorial) / 7.factorial = 80 := 
sorry

end factorial_expression_l467_467906


namespace find_even_and_decreasing_function_l467_467525

-- Define the functions
def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := 2 ^ x
def f3 (x : ℝ) : ℝ := x ^ 2
def f4 (x : ℝ) : ℝ := -x ^ 2

-- Define the properties of even and decreasing on [0, +∞)
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

-- The proof problem
theorem find_even_and_decreasing_function : 
  (is_even f1 ∧ is_decreasing_on_nonneg f1) ∨
  (is_even f2 ∧ is_decreasing_on_nonneg f2) ∨
  (is_even f3 ∧ is_decreasing_on_nonneg f3) ∨
  (is_even f4 ∧ is_decreasing_on_nonneg f4) :=
  (is_even f4 ∧ is_decreasing_on_nonneg f4) ∧ 
  ¬ (is_even f1 ∧ is_decreasing_on_nonneg f1) ∧
  ¬ (is_even f2 ∧ is_decreasing_on_nonneg f2) ∧
  ¬ (is_even f3 ∧ is_decreasing_on_nonneg f3) := by 
  sorry

end find_even_and_decreasing_function_l467_467525


namespace find_dividend_l467_467246

theorem find_dividend
  (R : ℕ)
  (Q : ℕ)
  (D : ℕ)
  (hR : R = 6)
  (hD_eq_5Q : D = 5 * Q)
  (hD_eq_3R_plus_2 : D = 3 * R + 2) :
  D * Q + R = 86 :=
by
  sorry

end find_dividend_l467_467246


namespace opposite_of_neg_two_l467_467386

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 :=
by simp [eq_comm]

end opposite_of_neg_two_l467_467386


namespace series_sum_half_l467_467916

theorem series_sum_half :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1/2 := 
sorry

end series_sum_half_l467_467916


namespace marble_probability_l467_467848

-- Define the conditions of the problem
def n := 60
def multiples (k : ℕ) : Finset ℕ := (Finset.range n).filter (λ x => x % k = 0)

-- Define the specifics of the proof problem
theorem marble_probability :
  let multiples_of_4 := multiples 4
  let multiples_of_6 := multiples 6
  let multiples_of_12 := multiples 12
  multiples_of_4.card + multiples_of_6.card - multiples_of_12.card = 20 →
  (20 : ℚ) / 60 = 1 / 3 :=
by
  intros h
  sorry

end marble_probability_l467_467848


namespace distinct_four_digit_count_l467_467153

theorem distinct_four_digit_count :
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  in count = 120 :=
by
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  show count = 120
  sorry

end distinct_four_digit_count_l467_467153


namespace sum_reciprocal_bound_l467_467698

noncomputable def recursive_seq (a : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 1
  else if n = 1 then a
  else ((recursive_seq a (n-1))^2 / (recursive_seq a (n-2))^2 - 2) * (recursive_seq a (n-1))
    
theorem sum_reciprocal_bound (a : ℝ) (k : ℕ) (h_a: a > 2) (h_k: k > 0) :
  (finset.range (k+1)).sum (λ i => 1 / recursive_seq a i) < 1 / 2 * (2 + a - real.sqrt (a^2 - 4)) :=
sorry

end sum_reciprocal_bound_l467_467698


namespace cost_of_notebook_and_pen_minimum_neutral_pens_l467_467470

-- Defining the cost equations for notebooks and pens
def cost_eq1 (x y : ℝ) := 4 * x + 3 * y = 38
def cost_eq2 (x y : ℝ) := x + 6 * y = 20

-- Prove the cost of one notebook and one neutral pen
theorem cost_of_notebook_and_pen : ∃ x y : ℝ, cost_eq1 x y ∧ cost_eq2 x y ∧ x = 8 ∧ y = 2 :=
by {
  existsi (8 : ℝ),
  existsi (2 : ℝ),
  simp [cost_eq1, cost_eq2],
  split, { norm_num },
  split, { norm_num },
  split;
  rfl,
}

-- Defining the constraints for the total cost
def total_cost_constraint (m : ℕ) := (8 * (60 - m) + 2 * m : ℝ) ≤ 330

-- Prove the minimum number of neutral pens
theorem minimum_neutral_pens : ∃ m : ℕ, total_cost_constraint m ∧ ∀ n : ℕ, total_cost_constraint n → m ≤ n :=
by {
  existsi 25,
  simp [total_cost_constraint],
  split,
  { norm_num },
  { intros n hn,
    suffices : n ≥ 25, { exact this },
    exact_mod_cast (le_of_nat_le_nat : 25 ≤ n),
    cases nat.le.dest (le_of_nat_le_nat : 25 ≤ n) with k hk,
    rw [hk, nat.cast_add, add_mul, ←nat.cast_add, nat.cast_le, ←nat.cast_min, nat.le_add_iff_nonneg_right],
    exact_mod_cast nat.zero_le k, }
}

end cost_of_notebook_and_pen_minimum_neutral_pens_l467_467470


namespace num_distinct_factors_1320_l467_467158

theorem num_distinct_factors_1320 : 
  ∃ n : ℕ, n = 1320 ∧ (finset.univ.filter (λ d, d > 0 ∧ 1320 % d = 0)).card = 32 := 
sorry

end num_distinct_factors_1320_l467_467158


namespace part1_part2_l467_467596

-- Definitions for Part (1)
def setA : Set ℝ := { x : ℝ | x^2 + 3 * x - 4 ≥ 0 }
def setB : Set ℝ := { x : ℝ | (x - 2) / x ≤ 0 }
def setC (a : ℝ) : Set ℝ := { x : ℝ | 2 * a < x ∧ x < 1 + a }

-- Theorem for Part (1)
theorem part1 (a : ℝ) : setC a ⊆ (setA ∩ setB) → a ∈ Ici (1/2:ℝ) :=
by
  sorry

-- Definitions for Part (2)
def setD (m : ℝ) : Set ℝ := { x : ℝ | x^2 - (2 * m + 1/2) * x + m * (m + 1/2) ≤ 0 }
def intersectionAB : Set ℝ := { x : ℝ | x ∈ setA ∧ x ∈ setB }

-- Theorem for Part (2)
theorem part2 (m : ℝ) : (∀ x, x ∈ intersectionAB → x ∈ setD m) ∧ ¬ (∀ x, x ∈ setD m → x ∈ intersectionAB) → m ∈ Icc (1:ℝ) (3/2:ℝ) :=
by
  sorry

end part1_part2_l467_467596


namespace point_on_curve_iff_F_eq_zero_l467_467653

variable (F : ℝ → ℝ → ℝ)
variable (a b : ℝ)

theorem point_on_curve_iff_F_eq_zero :
  (F a b = 0) ↔ (∃ P : ℝ × ℝ, P = (a, b) ∧ F P.1 P.2 = 0) :=
by
  sorry

end point_on_curve_iff_F_eq_zero_l467_467653


namespace find_unique_p_l467_467949

theorem find_unique_p (p : ℝ) (h1 : p ≠ 0) : (∀ x : ℝ, p * x^2 - 10 * x + 2 = 0 → p = 12.5) :=
by sorry

end find_unique_p_l467_467949


namespace bryan_travel_ratio_l467_467691

theorem bryan_travel_ratio
  (walk_time : ℕ)
  (bus_time : ℕ)
  (evening_walk_time : ℕ)
  (total_travel_hours : ℕ)
  (days_per_year : ℕ)
  (minutes_per_hour : ℕ)
  (minutes_total : ℕ)
  (daily_travel_time : ℕ) :
  walk_time = 5 →
  bus_time = 20 →
  evening_walk_time = 5 →
  total_travel_hours = 365 →
  days_per_year = 365 →
  minutes_per_hour = 60 →
  minutes_total = total_travel_hours * minutes_per_hour →
  daily_travel_time = (walk_time + bus_time + evening_walk_time) * 2 →
  (minutes_total / daily_travel_time = days_per_year) →
  (walk_time + bus_time + evening_walk_time) = daily_travel_time / 2 →
  (walk_time + bus_time + evening_walk_time) = daily_travel_time / 2 :=
by
  intros
  sorry

end bryan_travel_ratio_l467_467691


namespace sum_of_nus_is_45_l467_467801

noncomputable def sum_of_valid_nu : ℕ :=
  ∑ ν in {ν | ν > 0 ∧ Nat.lcm ν 24 = 72}.toFinset, ν

theorem sum_of_nus_is_45 : sum_of_valid_nu = 45 := by
  sorry

end sum_of_nus_is_45_l467_467801


namespace find_angle_between_vectors_l467_467101

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  Real.arccos ((a ⬝ b) / (‖a‖ * ‖b‖))

theorem find_angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3))
  (ha : ‖a‖ = 5) (hb : ‖b‖ = 4) (hab : a ⬝ b = -10) :
  angle_between_vectors a b = 2 * Real.pi / 3 :=
sorry

end find_angle_between_vectors_l467_467101


namespace range_of_a_l467_467650

theorem range_of_a (a : ℝ) (h_pos : 0 < a) : 
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π/4 ∧ cos x + real.sqrt a * sin x = real.sqrt a) ↔ 
  1 ≤ a ∧ a ≤ 3 + 2 * real.sqrt 2 :=
sorry

end range_of_a_l467_467650


namespace count_1320_factors_l467_467172

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end count_1320_factors_l467_467172


namespace sum_of_even_binomials_l467_467228

theorem sum_of_even_binomials (n : ℕ) (h : n % 2 = 0) (h_pos : n > 0) : 
  ∑ k in range (n/2 + 1), (binom n (2*k)) = 2^(n-1) :=
by
  sorry

end sum_of_even_binomials_l467_467228


namespace area_of_triangle_DEF_l467_467480

noncomputable def area_triangle_DEF 
  (Q : Point) (u1 u2 u3 : Triangle) 
  (area_u1 : ℝ) (area_u2 : ℝ) (area_u3 : ℝ) : ℝ :=
if h1 : area_u1 = 9 ∧ area_u2 = 16 ∧ area_u3 = 36 then 169 else 0

theorem area_of_triangle_DEF 
  (Q : Point) (D E F : Point) 
  (u1 u2 u3 : Triangle) 
  (h1 : u1.area = 9) 
  (h2 : u2.area = 16) 
  (h3 : u3.area = 36) 
  (h4 : lines_through_point_Q_parallel_to_sides DEF Q u1 u2 u3) : 
  area_triangle_DEF Q u1 u2 u3 u1.area u2.area u3.area = 169 :=
by
  unfold area_triangle_DEF
  rw [h1, h2, h3]
  rfl
  sorry

end area_of_triangle_DEF_l467_467480


namespace probability_longer_piece_at_least_x_times_shorter_l467_467479

-- Define the conditions
def is_shorter_piece (C : ℝ) : Prop :=
  C ≤ 0.5 ∨ (1 - C) ≤ 0.5

def is_longer_piece_at_least_x_times_shorter (C : ℝ) (x : ℝ) : Prop :=
  (1 - C) >= x * C ∨ C >= x * (1 - C)

-- Prove the probability statement
theorem probability_longer_piece_at_least_x_times_shorter (x : ℝ) (hx : x > 0) :
  Prob (λ C : ℝ, 0 ≤ C ∧ C ≤ 1 ∧ is_shorter_piece C ∧ is_longer_piece_at_least_x_times_shorter C x) 
  = (2 / (x + 1)) := 
by 
  sorry

end probability_longer_piece_at_least_x_times_shorter_l467_467479


namespace anya_lost_games_correct_l467_467046

-- Definition of players and their game counts
def Players := {Anya, Bella, Valya, Galya, Dasha}
def games_played : Players → ℕ
| Anya   := 4
| Bella  := 6
| Valya  := 7
| Galya  := 10
| Dasha  := 11

-- Total games played by all players
def total_games_played : ℕ := Players.to_finset.sum games_played

-- Total number of games, considering each game involves two players
def total_number_of_games : ℕ := total_games_played / 2

-- The set of games in which Anya lost
def anya_lost_games : set ℕ := {4, 8, 12, 16}

theorem anya_lost_games_correct :
  ∀ i ∈ anya_lost_games, -- For each game in anya_lost_games
  anya_lost_games = {4, 8, 12, 16} := 
by
  sorry

end anya_lost_games_correct_l467_467046


namespace complex_eq_l467_467104

-- Define the complex number z
def z : ℂ := 3 + 4 * complex.I

-- Define the given condition
def condition : Prop := z + complex.abs(z) * complex.I = 3 + 9 * complex.I

-- Prove that z = 3 + 4i given the condition
theorem complex_eq :
  condition → z = 3 + 4 * complex.I :=
by
  intro h
  sorry

end complex_eq_l467_467104


namespace pancake_cost_l467_467371

theorem pancake_cost :
  ∃ P : ℝ, 
    (let pancake_revenue := 60 * P in
     let bacon_revenue := 90 * 2 in
     let total_revenue := pancake_revenue + bacon_revenue in
     total_revenue = 420) ∧ P = 4 :=
by
  sorry

end pancake_cost_l467_467371


namespace max_siskins_on_poles_l467_467885

/-- 
Given 25 poles in a row, where only one siskin can occupy any given pole, 
and if a siskin lands on a pole, a siskin sitting on an adjacent pole will immediately fly away, 
the maximum number of siskins that can simultaneously be on the poles is 24.
-/
theorem max_siskins_on_poles : 
  ∀ (poles : Fin 25 → Bool), 
  (∀ i, poles i = false ∨ ∃ j, (abs (i - j) = 1 ∧ poles j = false))
  → ∃ n, (0 < n ∧ n ≤ 25 ∧ max_siskins poles n = 24) :=
by
  sorry

end max_siskins_on_poles_l467_467885


namespace distinct_four_digit_numbers_l467_467139

theorem distinct_four_digit_numbers : 
  let digits := {1, 2, 3, 4, 5} in (digits.card = 5) →
  ∃ count : ℕ, count = 5 * 4 * 3 * 2 ∧ count = 120 := 
by
  intros
  use 5 * 4 * 3 * 2
  split
  · refl
  · exact 120

end distinct_four_digit_numbers_l467_467139


namespace count_1320_factors_l467_467167

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end count_1320_factors_l467_467167


namespace area_of_given_triangle_is_32_l467_467013

noncomputable def area_of_triangle : ℕ :=
  let A := (-8, 0)
  let B := (0, 8)
  let C := (0, 0)
  1 / 2 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℤ).natAbs

theorem area_of_given_triangle_is_32 : area_of_triangle = 32 := 
  sorry

end area_of_given_triangle_is_32_l467_467013


namespace find_angle_A_find_b_c_l467_467631
open Real

-- Part I: Proving angle A
theorem find_angle_A (A B C : ℝ) (a b c : ℝ) (h₁ : (a + b + c) * (b + c - a) = 3 * b * c) :
  A = π / 3 :=
by sorry

-- Part II: Proving values of b and c given a=2 and area of triangle ABC is √3
theorem find_b_c (A B C : ℝ) (a b c : ℝ) (h₁ : a = 2) (h₂ : (1 / 2) * b * c * (sin (π / 3)) = sqrt 3) :
  b = 2 ∧ c = 2 :=
by sorry

end find_angle_A_find_b_c_l467_467631


namespace part1_part2_l467_467602

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 - (4 / (2 * a^x + a))

theorem part1 (h₁ : ∀ x, f a x = -f a (-x)) (h₂ : a > 0) (h₃ : a ≠ 1) : a = 2 :=
  sorry

theorem part2 (h₁ : a = 2) (x : ℝ) (hx : 0 < x ∧ x ≤ 1) (t : ℝ) :
  t * (f a x) ≥ 2^x - 2 ↔ t ≥ 0 :=
  sorry

end part1_part2_l467_467602


namespace increasing_function_probability_l467_467262

-- Define the function f(x) = x^2 + 2ax - 1
def f (a x : ℝ) := x^2 + 2 * a * x - 1

-- The derivative of f(x) with respect to x
def f_prime (a x : ℝ) := 2 * x + 2 * a

-- Define the interval in which a is randomly chosen
def a_interval : set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the sub-interval where f'(x) ≥ 0 on [1, +∞)
def increasing_sub_interval : set ℝ := {a | -1 ≤ a ∧ a ≤ 2}

-- Define the probability calculation
noncomputable def probability := 
  (↑(∫ x in increasing_sub_interval, 1)) / 
  (↑(∫ x in a_interval, 1))

-- assert the equivalent proof problem statement
theorem increasing_function_probability : 
  probability = 3 / 4 := 
sorry

end increasing_function_probability_l467_467262


namespace equivalent_pencils_total_l467_467414

noncomputable def total_equivalent_pencils : ℝ :=
  let initial_pencils_1 := 41.5
  let initial_pencils_2 := 25.2
  let initial_pens_3 := 13.6
  let added_pencils_1 := 30.7
  let added_pencils_2 := 18.5
  let added_pens_2 := 8.4
  let removed_pencils_1 := 5.3
  let removed_pencils_2 := 7.1
  let removed_pens_3 := 3.8
  let pencils_in_1 := initial_pencils_1 + added_pencils_1 - removed_pencils_1
  let pencils_in_2 := initial_pencils_2 + added_pencils_2 - removed_pencils_2
  let pens_in_3 := initial_pens_3 + added_pens_2 - removed_pens_3
  let pencils_from_pens_in_3 := pens_in_3 * 2
  pencils_in_1 + pencils_in_2 + pencils_from_pens_in_3

theorem equivalent_pencils_total : total_equivalent_pencils = 139.9 :=
by
  have h1 : 41.5 + 30.7 - 5.3 = 66.9 := by norm_num
  have h2 : 25.2 + 18.5 - 7.1 = 36.6 := by norm_num
  have h3 : 13.6 + 8.4 - 3.8 = 18.2 := by norm_num
  have h4 : 18.2 * 2 = 36.4 := by norm_num
  show total_equivalent_pencils = 66.9 + 36.6 + 36.4 from
    calc
      total_equivalent_pencils = 66.9 + 36.6 + 36.4 : by simp [h1, h2, h3, h4]
      ... = 139.9 : by norm_num
  sorry

end equivalent_pencils_total_l467_467414


namespace parallel_vectors_l467_467130

theorem parallel_vectors (m : ℝ) :
  let a : (ℝ × ℝ × ℝ) := (2, -1, 2)
  let b : (ℝ × ℝ × ℝ) := (-4, 2, m)
  (∀ k : ℝ, a = (k * -4, k * 2, k * m)) →
  m = -4 :=
by
  sorry

end parallel_vectors_l467_467130


namespace exists_sequence_a_l467_467972

def c (n : ℕ) : ℕ := 2017 ^ n

axiom f : ℕ → ℝ

axiom condition_1 : ∀ m n : ℕ, f (m + n) ≤ 2017 * f m * f (n + 325)

axiom condition_2 : ∀ n : ℕ, 0 < f (c (n + 1)) ∧ f (c (n + 1)) < (f (c n)) ^ 2017

theorem exists_sequence_a :
  ∃ (a : ℕ → ℕ), ∀ n k : ℕ, a k < n → f n ^ c k < f (c k) ^ n := sorry

end exists_sequence_a_l467_467972


namespace value_of_a3_plus_a5_l467_467084

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a : ℕ → α) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

theorem value_of_a3_plus_a5 (a : ℕ → α) (S : ℕ → α)
  (h_sequence : arithmetic_sequence a)
  (h_S7 : S 7 = 14)
  (h_sum_formula : ∀ n, S n = n * (a 1 + a n) / 2) :
  a 3 + a 5 = 4 :=
by
  sorry

end value_of_a3_plus_a5_l467_467084


namespace distinct_factors_1320_l467_467180

theorem distinct_factors_1320 :
  let p := 1320,
  prime_factors := [(2, 2), (3, 1), (5, 1), (11, 1)],
  num_factors := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in num_factors = 24 := by
  sorry

end distinct_factors_1320_l467_467180


namespace positive_integer_conditions_l467_467229

theorem positive_integer_conditions (p : ℕ) (hp : p > 0) : 
  (∃ k : ℕ, k > 0 ∧ 4 * p + 28 = k * (3 * p - 7)) ↔ (p = 6 ∨ p = 28) :=
by
  sorry

end positive_integer_conditions_l467_467229


namespace find_a7_l467_467260

noncomputable def geometric_sequence (r : ℝ) (a₁ : ℝ) (n : ℕ) :=
  a₁ * r^(n - 1)

theorem find_a7 (r : ℝ) (h : r ≠ 0):
  let a₁ := -16 in
  let a₄ := 8 in
  a₇ := geometric_sequence r a₁ 7 in
  (geometric_sequence r a₁ 1 = a₁) → 
  (geometric_sequence r a₁ 4 = a₄) → a₇ = -4 :=
by
  intros
  sorry

end find_a7_l467_467260


namespace quadrilateral_area_proof_l467_467794

noncomputable def area_of_quadrilateral : ℝ :=
  let circle (x y : ℝ) := x^2 + y^2 = 16 in
  let ellipse (x y : ℝ) := (x-3)^2 + 4*y^2 = 36 in
  14

theorem quadrilateral_area_proof :
  (∀ (x y : ℝ), circle x y → ellipse x y → area_of_quadrilateral = 14) :=
by
  intros x y h_circle h_ellipse
  sorry

end quadrilateral_area_proof_l467_467794


namespace compound_interest_rate_l467_467959

/-- Given the conditions for compound interest, we want to prove that the 
    annual interest rate is approximately 0.0396 --/
theorem compound_interest_rate 
  (P : ℝ) (t : ℝ) (n : ℕ) (I : ℝ) (r : ℝ) (A : ℝ) 
  (hP : P = 3000)
  (ht : t = 1.5)
  (hn : n = 2)
  (hI : I = 181.78817648189806)
  (hA : A = 3181.78817648189806)
  (hA_eq : A = P * (1 + r / n) ^ (n * t)) :
  r ≈ 0.0396 := sorry

end compound_interest_rate_l467_467959


namespace max_profit_l467_467867

variables (x y : ℕ)

def steel_constraint := 10 * x + 70 * y ≤ 700
def non_ferrous_constraint := 23 * x + 40 * y ≤ 642
def non_negativity := x ≥ 0 ∧ y ≥ 0
def profit := 80 * x + 100 * y

theorem max_profit (h₁ : steel_constraint x y)
                   (h₂ : non_ferrous_constraint x y)
                   (h₃ : non_negativity x y):
  profit x y = 2180 := 
sorry

end max_profit_l467_467867


namespace simplify_trig_identity_l467_467348

theorem simplify_trig_identity (x : ℝ) (h : ∀ θ : ℝ, Real.cot θ - 2 * Real.cot (2 * θ) = Real.tan θ) 
  : Real.tan x + 4 * Real.tan (2 * x) + 8 * Real.tan (4 * x) + 16 * Real.cot (16 * x) = Real.cot x := 
by 
  sorry

end simplify_trig_identity_l467_467348


namespace number_of_three_digit_multiples_of_9_with_odd_digits_l467_467643

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

def consists_only_of_odd_digits (n : ℕ) : Prop :=
  (∀ d ∈ (n.digits 10), d % 2 = 1)

theorem number_of_three_digit_multiples_of_9_with_odd_digits :
  ∃ t, t = 11 ∧
  (∀ n, is_three_digit_number n ∧ is_multiple_of_9 n ∧ consists_only_of_odd_digits n) → 1 ≤ t ∧ t ≤ 11 :=
sorry

end number_of_three_digit_multiples_of_9_with_odd_digits_l467_467643


namespace Chekalinsky_guarantees_win_l467_467461

theorem Chekalinsky_guarantees_win :
  (∀ (initial_config : Fin 8192),
   (∀ (situation : Fin 8192) (player : Bool),
    (player = false) → (situation ≠ initial_config)) →
   (∃ (move_strategy : Fin 13),
    (∀ (current_config : Fin 8192),
     (Chekalinsky_strategy move_strategy current_config)))) :=
sorry

end Chekalinsky_guarantees_win_l467_467461


namespace distinct_factors_1320_l467_467177

theorem distinct_factors_1320 :
  let p := 1320,
  prime_factors := [(2, 2), (3, 1), (5, 1), (11, 1)],
  num_factors := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in num_factors = 24 := by
  sorry

end distinct_factors_1320_l467_467177


namespace compound_interest_rate_l467_467492

theorem compound_interest_rate :
  ∃ r : ℝ, (1000 * (1 + r)^3 = 1331.0000000000005) ∧ r = 0.1 :=
by
  sorry

end compound_interest_rate_l467_467492


namespace area_A_l467_467322

variables {A B C A' B' C' : Type} [real_inner_product_space ℝ (A)] [real_inner_product_space ℝ (B)] [real_inner_product_space ℝ (C)]
variables (S : ℝ) (a b c : ℝ) [decidable_eq C] (AC BC AB : ℝ) (AC' BC' AB' CA AB_CA' : ℝ) [decidable_eq A'] (S_triangle_ABC S_triangle_A'B'C' : ℝ)

theorem area_A'B'C'_le_one_quarter_area_ABC
  (h1 : AC' / AB = a)
  (h2 : BC' / CA = b)
  (h3 : AB_CA' / AC = c)
  (h4 : AC' * BC' * AB_CA' = 2 * S)
  (h5 : AC' + BC' + AB_CA' = 1 - a - b - c + 1 * AC' * AB' * AB_CA' * BC' * S - AC * AB' * BC * S (equiv_trans h4 h5))
  (hA'A' : intersection S_triangle_ABC A' B' C' = is_point)
  : S_triangle_A'B'C' <= (1 / 4) * S_triangle_ABC :=
by sorry

end area_A_l467_467322


namespace find_two_irreducible_fractions_l467_467555

theorem find_two_irreducible_fractions :
  ∃ (a b d1 d2 : ℕ), 
    (1 ≤ a) ∧ 
    (1 ≤ b) ∧ 
    (gcd a d1 = 1) ∧ 
    (gcd b d2 = 1) ∧ 
    (1 ≤ d1) ∧ 
    (d1 ≤ 100) ∧ 
    (1 ≤ d2) ∧ 
    (d2 ≤ 100) ∧ 
    (a / (d1 : ℚ) + b / (d2 : ℚ) = 86 / 111) := 
by {
  sorry
}

end find_two_irreducible_fractions_l467_467555


namespace parabola_slope_l467_467622

theorem parabola_slope (p k : ℝ) (h1 : p > 0)
  (h_focus_distance : (p / 2) * (3^(1/2)) / (3 + 1^(1/2))^(1/2) = 3^(1/2))
  (h_AF_FB : exists A B : ℝ × ℝ, (A.1 = 2 - p / 2 ∧ 2 * (B.1 - 2) = 2)
    ∧ (A.2 = p - p / 2 ∧ A.2 = -2 * B.2)) :
  abs k = 2 * (2^(1/2)) :=
sorry

end parabola_slope_l467_467622


namespace part_a_part_b_l467_467324

-- Part (a)
theorem part_a (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m > n) : 
  (1 + 1 / (m:ℝ))^m > (1 + 1 / (n:ℝ))^n :=
by sorry

-- Part (b)
theorem part_b (m n : ℕ) (hm : 0 < m) (hn : 1 < n) (h : m > n) : 
  (1 + 1 / (m:ℝ))^(m + 1) < (1 + 1 / (n:ℝ))^(n + 1) :=
by sorry

end part_a_part_b_l467_467324


namespace remaining_angles_l467_467779

theorem remaining_angles (α β γ : ℝ) (hα : α = 110) (hβ : β = 60) (hγ : γ = 80) :
  ∃ (x y z : ℝ), x = 30 ∧ y = 50 ∧ z = 30 :=
by
  use [α - γ, α - β, β - (α - γ)]
  split
  · rw [hα, hγ]
    norm_num
  split
  · rw [hα, hβ]
    norm_num
  · rw [hβ, hα, hγ]
    norm_num
    sorry

end remaining_angles_l467_467779


namespace g_range_l467_467544

noncomputable def g (x : ℝ) : ℝ := x / (x^2 - 3*x + 2)

theorem g_range :
  set.range g = set.Iio (-3 - real.sqrt 8) ∪ set.Ici (-3 + real.sqrt 8) := sorry

end g_range_l467_467544


namespace find_boxes_l467_467693

variable (John Jules Joseph Stan : ℕ)

-- Conditions
axiom h1 : John = 30
axiom h2 : John = 6 * Jules / 5 -- Equivalent to John having 20% more boxes than Jules
axiom h3 : Jules = Joseph + 5
axiom h4 : Joseph = Stan / 5 -- Equivalent to Joseph having 80% fewer boxes than Stan

-- Theorem to prove
theorem find_boxes (h1 : John = 30) (h2 : John = 6 * Jules / 5) (h3 : Jules = Joseph + 5) (h4 : Joseph = Stan / 5) : Stan = 100 :=
sorry

end find_boxes_l467_467693


namespace distinct_factors_1320_l467_467215

theorem distinct_factors_1320 : 
  ∃ n : ℕ, n = 24 ∧ 
  let a := 2 in 
  let b := 1 in 
  let c := 1 in 
  let d := 1 in 
  1320 = 2^a * 3^b * 5^c * 11^d :=
begin
  sorry,
end

end distinct_factors_1320_l467_467215


namespace counterexample_conjecture_l467_467782

theorem counterexample_conjecture 
    (odd_gt_5 : ℕ → Prop) 
    (is_prime : ℕ → Prop) 
    (conjecture : ∀ n, odd_gt_5 n → ∃ p1 p2 p3, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ n = p1 + p2 + p3) : 
    ∃ n, odd_gt_5 n ∧ ¬ (∃ p1 p2 p3, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ n = p1 + p2 + p3) :=
sorry

end counterexample_conjecture_l467_467782


namespace range_of_a_l467_467118

def f (x : ℝ) : ℝ := 2 * Real.sin x - 3 * x

theorem range_of_a (a : ℝ) :
  (∀ m : ℝ, m ∈ Set.Icc (-2 : ℝ) 2 → f (m * a - 3) + f (a^2) > 0) ↔ (-1 < a ∧ a < 1) := by
  sorry

end range_of_a_l467_467118


namespace example_problem_l467_467496

open Real

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop := 
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f x < f y

theorem example_problem :
  (is_even (λ x : ℝ, 1 - x^2)) ∧ (is_increasing_on (λ x : ℝ, 1 - x^2) {x | x < 0}) :=
by
  sorry

end example_problem_l467_467496


namespace range_of_a_l467_467347

-- Conditions as mathematical definitions
def prob_A := 1 / 2
def prob_B (a : ℝ) := a
def prob_C (a : ℝ) := a
def ξ (a : ℝ) := 
  -- Definition of the random variable ξ that counts the number of hits
  let p_0 := (1 - 1/2) * (1 - a) * (1 - a),
      p_1 := 1/2 * (1 - a^2),
      p_2 := 1/2 * (2 * a - a^2),
      p_3 := 1/2 * a^2
  in (p_0, p_1, p_2, p_3)

-- Main theorem statement
theorem range_of_a (a : ℝ) (h : 0 < a ∧ a < 1) :
  let (p_0, p_1, p_2, p_3) := ξ a in
  p_1 > p_0 ∧ p_1 > p_2 ∧ p_1 > p_3 → 0 < a ∧ a ≤ 1/2 :=
by {
  sorry
}

end range_of_a_l467_467347


namespace rationalize_denominator_l467_467336

theorem rationalize_denominator (sqrt_98 : Real) (h : sqrt_98 = 7 * Real.sqrt 2) : 
  7 / sqrt_98 = Real.sqrt 2 / 2 := 
by 
  sorry

end rationalize_denominator_l467_467336


namespace ratio_children_to_adults_l467_467504

variable (f m c : ℕ)

-- Conditions
def average_age_female (f : ℕ) := 35
def average_age_male (m : ℕ) := 30
def average_age_child (c : ℕ) := 10
def overall_average_age (f m c : ℕ) := 25

-- Total age sums based on given conditions
def total_age_sum_female (f : ℕ) := 35 * f
def total_age_sum_male (m : ℕ) := 30 * m
def total_age_sum_child (c : ℕ) := 10 * c

-- Total sum and average conditions
def total_age_sum (f m c : ℕ) := total_age_sum_female f + total_age_sum_male m + total_age_sum_child c
def total_members (f m c : ℕ) := f + m + c

theorem ratio_children_to_adults (f m c : ℕ) (h : (total_age_sum f m c) / (total_members f m c) = 25) :
  (c : ℚ) / (f + m) = 2 / 3 := sorry

end ratio_children_to_adults_l467_467504


namespace ratio_shorter_to_longer_l467_467467

-- Define the total length and the length of the shorter piece
def total_length : ℕ := 90
def shorter_length : ℕ := 20

-- Define the length of the longer piece
def longer_length : ℕ := total_length - shorter_length

-- Define the ratio of shorter piece to longer piece
def ratio := shorter_length / longer_length

-- The target statement to prove
theorem ratio_shorter_to_longer : ratio = 2 / 7 := by
  sorry

end ratio_shorter_to_longer_l467_467467


namespace sum_of_positive_integers_nu_lcm_72_l467_467809

theorem sum_of_positive_integers_nu_lcm_72:
  let ν_values := { ν | Nat.lcm ν 24 = 72 }
  ∑ ν in ν_values, ν = 180 := by
  sorry

end sum_of_positive_integers_nu_lcm_72_l467_467809


namespace describe_S_l467_467701

def S : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | (p.2 ≤ 11 ∧ p.1 = 2) ∨ (p.1 ≤ 2 ∧ p.2 = 11) ∨ (p.1 ≥ 2 ∧ p.2 = p.1 + 9) }

theorem describe_S :
  S = { p : ℝ × ℝ | (p.2 ≤ 11 ∧ p.1 = 2) ∨ (p.1 ≤ 2 ∧ p.2 = 11) ∨ (p.1 ≥ 2 ∧ p.2 = p.1 + 9) } := 
by
  -- proof is omitted
  sorry

end describe_S_l467_467701


namespace soul_inequality_phi_inequality_iff_t_one_l467_467705

noncomputable def e : ℝ := Real.exp 1

theorem soul_inequality (x : ℝ) : e^x ≥ x + 1 ↔ x = 0 :=
by sorry

theorem phi_inequality_iff_t_one (x t : ℝ) : (∀ x, e^x - t*x - 1 ≥ 0) ↔ t = 1 :=
by sorry

end soul_inequality_phi_inequality_iff_t_one_l467_467705


namespace distinct_factors_1320_l467_467218

theorem distinct_factors_1320 : 
  ∃ n : ℕ, n = 24 ∧ 
  let a := 2 in 
  let b := 1 in 
  let c := 1 in 
  let d := 1 in 
  1320 = 2^a * 3^b * 5^c * 11^d :=
begin
  sorry,
end

end distinct_factors_1320_l467_467218


namespace fraction_of_power_l467_467708

noncomputable def m : ℕ := 32^500

theorem fraction_of_power (h : m = 2^2500) : m / 8 = 2^2497 :=
by
  have hm : m = 2^2500 := h
  sorry

end fraction_of_power_l467_467708


namespace series_sum_half_l467_467918

theorem series_sum_half :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1/2 := 
sorry

end series_sum_half_l467_467918


namespace interval_between_glows_l467_467754

-- Defining start and end times in seconds
def start_time_in_seconds : ℕ := (1 * 3600) + (57 * 60) + 58
def end_time_in_seconds : ℕ := (3 * 3600) + (20 * 60) + 47

-- Defining the total duration in seconds
def total_duration : ℕ := end_time_in_seconds - start_time_in_seconds

-- Given number of glows
def number_of_glows : ℝ := 276.05555555555554

-- Statement to prove the interval in seconds between each glow
theorem interval_between_glows : total_duration / number_of_glows ≈ 18 := by
  sorry

end interval_between_glows_l467_467754


namespace percent_increase_second_half_century_l467_467678

variable (P : ℝ) -- Initial population
variable (x : ℝ) -- Percentage increase in the second half of the century

noncomputable def population_first_half_century := 3 * P
noncomputable def population_end_century := P + 11 * P

theorem percent_increase_second_half_century :
  3 * P + (x / 100) * (3 * P) = 12 * P → x = 300 :=
by
  intro h
  sorry

end percent_increase_second_half_century_l467_467678


namespace andrew_donates_160_to_homeless_shelter_l467_467502

/-- Andrew's bake sale earnings -/
def totalEarnings : ℕ := 400

/-- Amount kept by Andrew for ingredients -/
def ingredientsCost : ℕ := 100

/-- Amount Andrew donates from his own piggy bank -/
def piggyBankDonation : ℕ := 10

/-- The total amount Andrew donates to the homeless shelter -/
def totalDonationToHomelessShelter : ℕ :=
  let remaining := totalEarnings - ingredientsCost
  let halfDonation := remaining / 2
  halfDonation + piggyBankDonation

theorem andrew_donates_160_to_homeless_shelter : totalDonationToHomelessShelter = 160 := by
  sorry

end andrew_donates_160_to_homeless_shelter_l467_467502


namespace bowls_per_minute_l467_467945

def ounces_per_bowl : ℕ := 10
def gallons_of_soup : ℕ := 6
def serving_time_minutes : ℕ := 15
def ounces_per_gallon : ℕ := 128

theorem bowls_per_minute :
  (gallons_of_soup * ounces_per_gallon / servings_time_minutes) / ounces_per_bowl = 5 :=
by
  sorry

end bowls_per_minute_l467_467945


namespace min_value_x_plus_y_l467_467991

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 19 / x + 98 / y = 1) : 
  x + y ≥ 117 + 14 * Real.sqrt 38 :=
  sorry

end min_value_x_plus_y_l467_467991


namespace num_distinct_factors_1320_l467_467157

theorem num_distinct_factors_1320 : 
  ∃ n : ℕ, n = 1320 ∧ (finset.univ.filter (λ d, d > 0 ∧ 1320 % d = 0)).card = 32 := 
sorry

end num_distinct_factors_1320_l467_467157


namespace rationalize_denominator_l467_467337

theorem rationalize_denominator (sqrt_98 : Real) (h : sqrt_98 = 7 * Real.sqrt 2) : 
  7 / sqrt_98 = Real.sqrt 2 / 2 := 
by 
  sorry

end rationalize_denominator_l467_467337


namespace feb_1_day_of_week_l467_467231

theorem feb_1_day_of_week (leap_year : Prop) (feb_29_is_monday : Prop) :
  leap_year → feb_29_is_monday → (day_of_week 1 2 leap_year) = monday := 
sorry

end feb_1_day_of_week_l467_467231


namespace union_of_sets_l467_467722

def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {2, 3, 6}

theorem union_of_sets : A ∪ B = {1, 2, 3, 5, 6} :=
by sorry

end union_of_sets_l467_467722


namespace perpendicular_bisector_c_l467_467381

/-- Given a line segment from (2, 4) to (6, 8) and a line x - y = c that is the perpendicular bisector of this segment, we show that c = -2. -/
theorem perpendicular_bisector_c (c : ℝ) :
  (∀ (x y : ℝ), (x - y = c) → (x, y) = (4, 6)) →
  c = -2 :=
by
  -- This statement translates the conditions and the conclusion directly
  intros h
  have midpoint_on_line := h 4 6
  apply midpoint_on_line
  -- Skip the detailed proof.
  sorry

end perpendicular_bisector_c_l467_467381


namespace runner_injury_point_l467_467484

theorem runner_injury_point
  (v d : ℝ)
  (h1 : 2 * (40 - d) / v = d / v + 11)
  (h2 : 2 * (40 - d) / v = 22) :
  d = 20 := 
by
  sorry

end runner_injury_point_l467_467484


namespace math_problem_l467_467291

theorem math_problem (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : a + b + c = 0) :
  (a^2 * b^2 / ((a^2 - b * c) * (b^2 - a * c)) +
  a^2 * c^2 / ((a^2 - b * c) * (c^2 - a * b)) +
  b^2 * c^2 / ((b^2 - a * c) * (c^2 - a * b))) = 1 :=
by
  sorry

end math_problem_l467_467291


namespace ben_paints_240_square_feet_l467_467493

theorem ben_paints_240_square_feet
  (total_area : ℕ)
  (allen_share_ratio ben_share_ratio : ℕ)
  (ratio : allen_share_ratio = 2 ∧ ben_share_ratio = 6)
  (total_area_condition : total_area = 320) :
  (ben_share_ratio / (allen_share_ratio + ben_share_ratio)) * total_area = 240 :=
by
  -- Given the condition that the ratio of Allen's work to Ben's work is 2:6, total_area is 320
  have h : ben_share_ratio / (allen_share_ratio + ben_share_ratio) * total_area = 240,
  sorry

end ben_paints_240_square_feet_l467_467493


namespace sum_of_positive_integers_lcm72_l467_467816

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_of_positive_integers_lcm72 : 
  ∑ ν in {ν | is_solution ν}.to_finset, ν = 180 :=
by 
  sorry

end sum_of_positive_integers_lcm72_l467_467816


namespace sum_x_y_z_l467_467389
open Real

theorem sum_x_y_z (a b : ℝ) (h1 : a / b = 98 / 63) (x y z : ℕ) (h2 : (sqrt a) / (sqrt b) = (x * sqrt y) / z) : x + y + z = 18 := 
by
  sorry

end sum_x_y_z_l467_467389


namespace ants_on_dodecahedron_l467_467005

-- Define the structure of the regular dodecahedron, since it's crucial for the problem (not fully fleshed out for simplicity)
structure Dodecahedron where 
  vertices : Fin 20

-- Define the ants' problem formally
theorem ants_on_dodecahedron :
  ∃ (P : prop), P = prob_no_two_ants_same_vertex (Dodecahedron) = 60 / 6561 :=  
sorry

end ants_on_dodecahedron_l467_467005


namespace distance_between_planes_l467_467539

-- Define the plane equations
def plane1 (x y z : ℝ) : Prop := 3 * x + y - z + 3 = 0
def plane2 (x y z : ℝ) : Prop := 6 * x + 2 * y - 2 * z + 7 = 0

-- Define a point on the first plane
def point_on_plane1 : ℝ × ℝ × ℝ := (0, 0, 3)

-- Compute the distance between a point and a plane
def point_to_plane_distance (p : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := p
  abs (6 * x + 2 * y - 2 * z + 7) / real.sqrt (6^2 + 2^2 + (-2)^2)

-- Prove the distance between the planes
theorem distance_between_planes : point_to_plane_distance point_on_plane1 = 1 / (2 * real.sqrt 11) := sorry

end distance_between_planes_l467_467539


namespace large_number_exponent_l467_467438

theorem large_number_exponent (h : 10000 = 10 ^ 4) : 10000 ^ 50 * 10 ^ 5 = 10 ^ 205 := 
by
  sorry

end large_number_exponent_l467_467438


namespace math_problem_proof_l467_467978

-- Definitions from the problem conditions
def original_set : Finset ℕ := Finset.range 500
def draw_a := Finset.nth_le (Finset.insert 0 original_set) 0 sorry  -- placeholder for drawing
def remaining_set := original_set.erase draw_a
def draw_b := Finset.nth_le (Finset.insert 0 remaining_set) 0 sorry  -- placeholder for drawing

def hyperbrick_dimensions := {a1 : ℕ // a1 ∈ original_set}
def hyperbox_dimensions := {b1 : ℕ // b1 ∈ remaining_set}

-- The main theorem statement where we need to prove the sum of numerator and denominator.
theorem math_problem_proof : let p := (16 : ℚ) / 70 in (p.num + p.den) = 43 :=
by
  sorry  -- Proof to be filled in later.

end math_problem_proof_l467_467978


namespace anya_lost_games_l467_467032

def girl_count := 5
def anya_games := 4
def bella_games := 6
def valya_games := 7
def galya_games := 10
def dasha_games := 11
def total_games := 19

theorem anya_lost_games :
  (anya_games + bella_games + valya_games + galya_games + dasha_games) / 2 = total_games →
  ∀ games : list ℕ,
    games.length = total_games →
    (∀ g ∈ games, g > 0 ∧ g ≤ total_games) →
    (∀ g ∈ [anya_games, bella_games, valya_games, galya_games, dasha_games], g ≤ total_games) →
    ∀ a b : ℕ, a ≠ b →
    (anya_games, games.nth 3 = some 4) ∧
    (anya_games, games.nth 7 = some 8) ∧
    (anya_games, games.nth 11 = some 12) ∧
    (anya_games, games.nth 15 = some 16)
:= sorry

end anya_lost_games_l467_467032


namespace other_number_l467_467834

theorem other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 2310) (h_hcf : Nat.gcd A B = 30) (h_A : A = 770) : B = 90 :=
by
  -- The proof is omitted here.
  sorry

end other_number_l467_467834


namespace smallest_x_mod_equation_l467_467014

theorem smallest_x_mod_equation : ∃ x : ℕ, 42 * x + 10 ≡ 5 [MOD 15] ∧ ∀ y : ℕ, 42 * y + 10 ≡ 5 [MOD 15] → x ≤ y :=
by
sorry

end smallest_x_mod_equation_l467_467014


namespace al_original_portion_l467_467491

variables (a b c d : ℝ)

theorem al_original_portion :
  a + b + c + d = 1200 →
  a - 150 + 2 * b + 2 * c + 3 * d = 1800 →
  a = 450 :=
by
  intros h1 h2
  sorry

end al_original_portion_l467_467491


namespace Anya_loss_games_l467_467023

noncomputable def game_indices_Anya_lost (Anya Bella Valya Galya Dasha : ℕ) : Set ℕ :=
  {4, 8, 12, 16}

theorem Anya_loss_games
  (Anya Bella Valya Galya Dasha total_games : ℕ)
  (h1 : Anya = 4)
  (h2 : Bella = 6)
  (h3 : Valya = 7)
  (h4 : Galya = 10)
  (h5 : Dasha = 11)
  (h6 : total_games = (Anya + Bella + Valya + Galya + Dasha) / 2)
  (h7 : total_games = 19) :
  game_indices_Anya_lost Anya Bella Valya Galya Dasha = {4, 8, 12, 16} :=
  by sorry

end Anya_loss_games_l467_467023


namespace problem_statement_l467_467716

noncomputable def f (a b c x : ℝ) : ℝ := a^x + b^x - c^x

theorem problem_statement (a b c : ℝ) (h1 : c > a) (h2 : a > 0) (h3 : c > b) (h4 : b > 0) (h5 : a + b > c):
  (∀ x : ℝ, x < 1 -> f a b c x > 0) ∧
  (∃ x : ℝ, x > 0 ∧ (¬ triangle_sides a^x b^x c^x)) ∧
  (is_obtuse a b c -> ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f a b c x = 0) ∧
  count_true [
    (∀ x : ℝ, x < 1 -> f a b c x > 0),
    (∃ x : ℝ, x > 0 ∧ (¬ triangle_sides a^x b^x c^x)),
    (is_obtuse a b c -> ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f a b c x = 0)
  ] = 3 :=
sorry

/- Auxiliary Definitions -/
def triangle_sides (x y z : ℝ) : Prop := x + y > z ∧ x + z > y ∧ y + z > x

def is_obtuse (a b c : ℝ) : Prop := a^2 + b^2 - c^2 < 0

def count_true (props : List Prop) : ℕ := props.filter id |>.length

end problem_statement_l467_467716


namespace tangential_quadrilateral_of_parallelogram_l467_467857

theorem tangential_quadrilateral_of_parallelogram 
  (A B C D O : Point)
  (a b c d : ℝ)
  (h_parallelogram : Parallelogram A B C D)
  (O_midpoint : Midpoint O A C ∧ Midpoint O B D)
  (circles_tangent : CirclesExternallyTangent A a B b C c D d)
  (h_condition : a + c = b + d) :
  TangentialQuadrilateral (ExternalTangentsQuadrilateral A B C D) :=
by sorry

end tangential_quadrilateral_of_parallelogram_l467_467857


namespace Anya_loss_games_l467_467025

noncomputable def game_indices_Anya_lost (Anya Bella Valya Galya Dasha : ℕ) : Set ℕ :=
  {4, 8, 12, 16}

theorem Anya_loss_games
  (Anya Bella Valya Galya Dasha total_games : ℕ)
  (h1 : Anya = 4)
  (h2 : Bella = 6)
  (h3 : Valya = 7)
  (h4 : Galya = 10)
  (h5 : Dasha = 11)
  (h6 : total_games = (Anya + Bella + Valya + Galya + Dasha) / 2)
  (h7 : total_games = 19) :
  game_indices_Anya_lost Anya Bella Valya Galya Dasha = {4, 8, 12, 16} :=
  by sorry

end Anya_loss_games_l467_467025


namespace angle_QPR_40_degrees_l467_467424

-- Define the isosceles triangle property
def is_isosceles (a b c : ℝ) (angle_ABC : ℝ) : Prop :=
  a = b ∧ angle_ABC / 2 = (180 - angle_ABC) / 2

-- Define a theorem with the conditions and conclusion
theorem angle_QPR_40_degrees : 
  ∀ (P Q R S : Type) 
  (PQ QR PR RS : ℝ) 
  (anglePQR anglePRS : ℝ),
  -- Conditions
  PQ = QR -> 
  PR = RS -> 
  anglePQR = 50 -> 
  anglePRS = 130 -> 

  -- Conclusion
  ∃ angleQPR : ℝ, angleQPR = 40 := 
begin
  intros,
  sorry
end

end angle_QPR_40_degrees_l467_467424


namespace sum_of_positive_integers_nu_lcm_72_l467_467812

theorem sum_of_positive_integers_nu_lcm_72:
  let ν_values := { ν | Nat.lcm ν 24 = 72 }
  ∑ ν in ν_values, ν = 180 := by
  sorry

end sum_of_positive_integers_nu_lcm_72_l467_467812


namespace series_sum_half_l467_467917

theorem series_sum_half :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1/2 := 
sorry

end series_sum_half_l467_467917


namespace find_x_in_equation_l467_467970

theorem find_x_in_equation :
  ∀ (x : ℝ), 3^2 * 9^(2*x + 1) / 27^(x + 1) = 81 → x = 3 := 
by
  intros x h
  sorry

end find_x_in_equation_l467_467970


namespace sum_of_positive_integers_lcm72_l467_467817

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_of_positive_integers_lcm72 : 
  ∑ ν in {ν | is_solution ν}.to_finset, ν = 180 :=
by 
  sorry

end sum_of_positive_integers_lcm72_l467_467817


namespace Anya_loss_games_l467_467021

noncomputable def gamesPlayed : Nat := 19

constant Anya_games : Nat := 4
constant Bella_games : Nat := 6
constant Valya_games : Nat := 7
constant Galya_games : Nat := 10
constant Dasha_games : Nat := 11

constant gameResults : Fin 19 → Option String
constant lost (g : Nat) (p : String) : Prop

axiom Anya_game_indices : ∀ (i : Fin 4), lost (gameResults i.val) "Anya" → i.val = 4 * i + 4

theorem Anya_loss_games : list Nat :=
have Anya_lost_games : list Nat := [4, 8, 12, 16],
Anya_lost_games

end Anya_loss_games_l467_467021


namespace anya_lost_games_l467_467033

def girl_count := 5
def anya_games := 4
def bella_games := 6
def valya_games := 7
def galya_games := 10
def dasha_games := 11
def total_games := 19

theorem anya_lost_games :
  (anya_games + bella_games + valya_games + galya_games + dasha_games) / 2 = total_games →
  ∀ games : list ℕ,
    games.length = total_games →
    (∀ g ∈ games, g > 0 ∧ g ≤ total_games) →
    (∀ g ∈ [anya_games, bella_games, valya_games, galya_games, dasha_games], g ≤ total_games) →
    ∀ a b : ℕ, a ≠ b →
    (anya_games, games.nth 3 = some 4) ∧
    (anya_games, games.nth 7 = some 8) ∧
    (anya_games, games.nth 11 = some 12) ∧
    (anya_games, games.nth 15 = some 16)
:= sorry

end anya_lost_games_l467_467033


namespace four_rounds_each_four_coins_l467_467489

-- Define the initial condition: four players with 4 coins each
def initialCoins : List ℕ := [4, 4, 4, 4]

-- Define the dynamics of a round
def round (coins : List ℕ) : List ℕ → List ℕ
| [g, r, _, _] := coins -- Assuming g is the index of the player with the green ball, and r is the index of the player with the red ball. Update coins accordingly.

-- Main theorem statement: The probability that each player has 4 coins at the end of the fourth round.
theorem four_rounds_each_four_coins :
  let probability := 5 / 192 
  (probability_end := sorry) -- Assuming we have a function or method to calculate this probability
  sum (probability_end initialCoins 4) = probability := sorry

end four_rounds_each_four_coins_l467_467489


namespace nested_sum_binomial_l467_467325

theorem nested_sum_binomial {n k : ℕ} (hn : 0 < n) (hk : 0 < k) :
  (∑ i1 in Finset.range (n + 1), 
  ∑ i2 in Finset.range (i1 + 1), 
  ∑ i3 in Finset.range (i2 + 1), 
  ... -- continue this pattern for all k sums
  ∑ ik in Finset.range (i_{k-1} + 1), 
  1) = Nat.choose (n + k - 1) k := sorry

end nested_sum_binomial_l467_467325


namespace series_sum_l467_467932

theorem series_sum :
  ∑' n : ℕ, (2 ^ n) / (3 ^ (2 ^ n) + 1) = 1 / 2 :=
sorry

end series_sum_l467_467932


namespace sales_tax_percentage_l467_467473

/--
A couple spent $184.80 in total while dining out and paid this amount using a credit card.
The total amount included a 20 percent tip which was paid on top of the price which already 
included a sales tax on top of the price of the food. The actual price of the food before tax 
and tip was $140. Prove that the percentage of the sales tax was 10%.
-/
theorem sales_tax_percentage
  (total : ℝ)
  (food_price : ℝ)
  (tip_rate : ℝ)
  (tax_rate : ℝ)
  (H : total = food_price * (1 + tax_rate / 100) * (1 + tip_rate)) :
  tax_rate = 10 :=
by
  let total := 184.80
  let food_price := 140
  let tip_rate := 0.20
  have H : total = food_price * (1 + 10 / 100) * (1 + 0.20),
  {
    sorry
  }
  exact eq_of_heq (cast H)

end sales_tax_percentage_l467_467473


namespace graph_of_equation_l467_467440

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := by
  sorry

end graph_of_equation_l467_467440


namespace total_pizzas_two_days_l467_467638

-- Declare variables representing the number of pizzas made by Craig and Heather
variables (c1 c2 h1 h2 : ℕ)

-- Given conditions as definitions
def condition_1 : Prop := h1 = 4 * c1
def condition_2 : Prop := h2 = c2 - 20
def condition_3 : Prop := c1 = 40
def condition_4 : Prop := c2 = c1 + 60

-- Prove that the total number of pizzas made in two days equals 380
theorem total_pizzas_two_days (c1 c2 h1 h2 : ℕ)
  (h_cond1 : condition_1 c1 h1) (h_cond2 : condition_2 h2 c2)
  (h_cond3 : condition_3 c1) (h_cond4 : condition_4 c1 c2) :
  (h1 + c1) + (h2 + c2) = 380 :=
by
  -- As we don't need to provide the proof here, just insert sorry
  sorry

end total_pizzas_two_days_l467_467638


namespace max_siskins_on_poles_l467_467893

-- Define the conditions
def total_poles : ℕ := 25

def adjacent (i j : ℕ) : Prop := (abs (i - j) = 1)

-- Define the main problem
theorem max_siskins_on_poles (h₁ : 0 < total_poles) 
  (h₂ : ∀ (i : ℕ), i ≥ 1 ∧ i ≤ total_poles → ∀ (j : ℕ), j ≥ 1 ∧ j ≤ total_poles ∧ adjacent i j 
    → ¬ (siskin_on i ∧ siskin_on j)) :
  ∃ (max_siskins : ℕ), max_siskins = 24 := 
begin
  sorry
end

end max_siskins_on_poles_l467_467893


namespace minimum_balls_to_ensure_8_same_color_l467_467658

theorem minimum_balls_to_ensure_8_same_color :
  let red := 15
  let green := 12
  let blue := 10
  let yellow := 7
  let white := 6
  let total := red + green + blue + yellow + white
  in total = 50 → (∀ n : ℕ, (n < 35 → (∃ r g b y w : ℕ, r ≤ red ∧ g ≤ green ∧ b ≤ blue ∧ y ≤ yellow ∧ w ≤ white ∧ r + g + b + y + w = n ∧ r < 8 ∧ g < 8 ∧ b < 8 ∧ y < 8 ∧ w < 8))
   ∧ (∀ n : ℕ, n = 35 → ∃ r g b y w : ℕ, r + g + b + y + w = n ∧ (r ≥ 8 ∨ g ≥ 8 ∨ b ≥ 8 ∨ y ≥ 8 ∨ w ≥ 8))) :=
begin
  intro red,
  intro green,
  intro blue,
  intro yellow,
  intro white,
  intro total,
  intro htotal,
  split,
  {
    intros n hn,
    -- sorry (proof steps would go here)
    sorry,
  },
  {
    intros n hn,
    -- sorry (proof steps would go here)
    sorry,
  }
end

end minimum_balls_to_ensure_8_same_color_l467_467658


namespace parallelogram_area_l467_467961

noncomputable def area_of_parallelogram (a b : ℝ) (θ : ℝ) : ℝ :=
  a * b * Real.sin θ

theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 25) (h2 : b = 15) (h3 : θ = 40 * Real.pi / 180) :
  area_of_parallelogram a b θ ≈ 240.45 :=
  by
  sorry

end parallelogram_area_l467_467961


namespace find_an_bn_find_Tn_l467_467676

namespace ArithmeticGeometricSequences

-- Definitions and conditions
def a1 : ℕ := 3
def b1 : ℕ := 1

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def q (b2 : ℕ) : ℚ := (S 2) / b2

axiom b2_S2_condition (b2 : ℕ) : b2 + S 2 = 12

-- Assertions
noncomputable def an (n : ℕ) : ℕ := 3 * n
noncomputable def bn (n : ℕ) : ℕ := 3^(n-1)

noncomputable def Sn (n : ℕ) : ℚ := (3 / 2 : ℚ) * ((bn n)^2 + bn n)

noncomputable def Tn (n : ℕ) : ℚ := (3 / 2) * ((∑ i in finset.range n, bn (i + 1)) + (∑ i in finset.range n, (bn (i + 1))^2))

noncomputable def Tn_closed_form (n : ℕ) : ℚ :=
  (3^(2 * n + 1) / 16 : ℚ) + (3^(n + 1) / 4) - (15 / 16)

-- Proof goals
theorem find_an_bn : ∀ n : ℕ, an n = 3 * n ∧ bn n = 3^(n-1) := 
by
  intro n
  sorry

theorem find_Tn : ∀ n : ℕ, Tn n = Tn_closed_form n :=
by
  intro n
  sorry

end ArithmeticGeometricSequences

end find_an_bn_find_Tn_l467_467676


namespace problem1_problem2_problem3_problem4_l467_467513

theorem problem1 : (-8) + 10 - 2 + (-1) = -1 := 
by
  sorry

theorem problem2 : 12 - 7 * (-4) + 8 / (-2) = 36 := 
by 
  sorry

theorem problem3 : ( (1/2) + (1/3) - (1/6) ) / (-1/18) = -12 := 
by 
  sorry

theorem problem4 : - 1 ^ 4 - (1 + 0.5) * (1/3) * (-4) ^ 2 = -33 / 32 := 
by 
  sorry


end problem1_problem2_problem3_problem4_l467_467513


namespace equation_solution_l467_467737

def solve_equation (x : ℝ) : Prop :=
  ((3 * x + 6) / (x^2 + 5 * x - 6) = (4 - x) / (x - 2)) ↔ 
  x = -3 ∨ x = (1 + Real.sqrt 17) / 2 ∨ x = (1 - Real.sqrt 17) / 2

theorem equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -6) (h3 : x ≠ 2) : solve_equation x :=
by
  sorry

end equation_solution_l467_467737


namespace distance_to_focus_l467_467753

noncomputable def ellipse_foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let c := real.sqrt (a^2 - b^2) 
  ((-c, 0), (c, 0))

theorem distance_to_focus 
  (a b : ℝ) 
  (h : a > b) 
  {P : ℝ × ℝ} 
  (hP : (P.fst / a)^2 + (P.snd / b)^2 = 1) 
  (hPf1 : P.fst = -real.sqrt (a^2 - b^2)) : 
  -- Given the conditions for the ellipse and point P
  let F2 := (real.sqrt (a^2 - b^2), 0)
  in |P.1 - F2.fst| = 7 / 2 :=
sorry

end distance_to_focus_l467_467753


namespace pipe_p_fills_cistern_in_12_minutes_l467_467790

theorem pipe_p_fills_cistern_in_12_minutes :
  (∃ (t : ℝ), 
    ∀ (q_fill_rate p_fill_rate : ℝ), 
      q_fill_rate = 1 / 15 ∧ 
      t > 0 ∧ 
      (4 * (1 / t + q_fill_rate) + 6 * q_fill_rate = 1) → t = 12) :=
sorry

end pipe_p_fills_cistern_in_12_minutes_l467_467790


namespace distinct_factors_1320_l467_467205

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end distinct_factors_1320_l467_467205


namespace circle_radius_l467_467683

/-- In triangle ABC, AB = AC = 144, and BC = 80. Circle P has radius 24 and is tangent to AC and BC. 
Circle Q is externally tangent to P and is tangent to both AB and BC. Assume no point of circle Q lies outside of triangle ABC. 
Then the radius r of circle Q can be expressed as r = 64 - 12 * sqrt(21) and the quantity m + nk can be computed as 316. -/
theorem circle_radius 
  (AB AC : ℝ) (BC : ℝ) 
  (radiusP : ℝ) (tangent_AC : AC → radiusP) (tangent_BC : BC → radiusP)
  (externally_tangent : ∀ P Q : ℝ, P = Q → Q ∈ P ∧ Q ∉ triangle ABC) :
  ∃ r : ℝ (m n k : ℕ), 
  (r = 64 - 12 * Real.sqrt 21) ∧ (m + n * k = 316) := by
  sorry

end circle_radius_l467_467683


namespace pipe_flow_rate_l467_467846

-- Define the constants for the given problem.
def tank_capacity : ℕ := 1000
def initial_water_volume : ℕ := tank_capacity / 2
def drain_rate1 : ℚ := 1000 / 4  -- 250 liters per minute
def drain_rate2 : ℚ := 1000 / 6  -- 166.67 liters per minute
def total_draining_rate : ℚ := drain_rate1 + drain_rate2
def filling_time : ℕ := 6
def volume_added : ℚ := initial_water_volume
def effective_filling_rate : ℚ := volume_added / filling_time

-- Hypotheses in noncomputable context for exact rational numbers and proof statement with sorry.
noncomputable def flow_rate : ℚ := effective_filling_rate + total_draining_rate
#eval flow_rate -- should output 500/1

theorem pipe_flow_rate (F : ℚ) (h : F = flow_rate) : F = 500 := by
  rw [h]
  norm_num
  sorry

end pipe_flow_rate_l467_467846


namespace log3_of_7_eq_ab_l467_467982

noncomputable def log3_of_2_eq_a (a : ℝ) : Prop := Real.log 2 / Real.log 3 = a
noncomputable def log2_of_7_eq_b (b : ℝ) : Prop := Real.log 7 / Real.log 2 = b

theorem log3_of_7_eq_ab (a b : ℝ) (h1 : log3_of_2_eq_a a) (h2 : log2_of_7_eq_b b) :
  Real.log 7 / Real.log 3 = a * b :=
sorry

end log3_of_7_eq_ab_l467_467982


namespace total_surface_area_of_tower_l467_467006

def volumes : List ℕ := [1, 27, 125, 343, 512, 729, 1000, 1331]

theorem total_surface_area_of_tower : 
  (let side_lengths := volumes.map (λ v, Int.toNat (Real.sqrt (Real.sqrt (Real.ofNat v)))) in
   let surface_areas := side_lengths.map (λ s, 6 * s^2) in
   surface_areas.zipWith (λ sa s, sa - s^2) side_lengths.tail).sum + 5 = 2250 :=
  sorry

end total_surface_area_of_tower_l467_467006


namespace find_m_l467_467541

noncomputable def P : ℝ := 
  (Finset.range 50).prod (λ n, (2 * n + 1) / (2 * n + 2))

theorem find_m : ∃ m : ℤ, 10^m < P ∧ P < 10^(m+1) ∧ m = -2 := by
  sorry

end find_m_l467_467541


namespace sin_phi_value_l467_467713

variables {u v w : ℝ^3}
variables {φ : ℝ}

-- Lets state our conditions
axiom nonzero_vectors : u ≠ 0 ∧ v ≠ 0 ∧ w ≠ 0
axiom not_parallel : (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * u ≠ b * v ∧ b * v ≠ c * w)
axiom given_condition : (u × v) × w = (1/4) * ∥v∥ * ∥w∥ • u
axiom angle_between: φ = real.arccos (inner v w / (∥v∥ * ∥w∥))

theorem sin_phi_value : real.sin φ = √15 / 4 :=
by
  -- Proof to be filled here
  sorry

end sin_phi_value_l467_467713


namespace infinite_solutions_xyz_l467_467733

theorem infinite_solutions_xyz (k : ℤ) : 
  let x := k * (2 * k^2 + 1),
      y := 2 * k^2 + 1,
      z := -k * (2 * k^2 + 1)
  in x ^ 2 + y ^ 2 + z ^ 2 = x ^ 3 + y ^ 3 + z ^ 3 := 
by sorry

end infinite_solutions_xyz_l467_467733


namespace product_simplification_l467_467566

noncomputable def a (n : ℕ) : ℚ := ((n + 1)^3 - 1) / (n * (n^3 - 1))

theorem product_simplification :
  ∏ n in finset.range 95 \ finset.range 4, a (n + 5) = 10101 / (99!) := by
sorry

end product_simplification_l467_467566


namespace teresa_science_marks_l467_467365

-- Definitions for the conditions
def music_marks : ℕ := 80
def social_studies_marks : ℕ := 85
def physics_marks : ℕ := music_marks / 2
def total_marks : ℕ := 275

-- Statement to prove
theorem teresa_science_marks : ∃ S : ℕ, 
  S + music_marks + social_studies_marks + physics_marks = total_marks ∧ S = 70 :=
sorry

end teresa_science_marks_l467_467365


namespace tangent_AO_circumcircle_AMP_l467_467083

variables {A B C O T M P D E : Type*}
variables [AddCommGroup A] [AffineSpace Points A]
variables [AddCommGroup B] [AffineSpace Points B]
variables [AddCommGroup C] [AffineSpace Points C]
variables [AddCommGroup O] [AffineSpace Points O]
variables [AddCommGroup T] [AffineSpace Points T]
variables [AddCommGroup M] [AffineSpace Points M]
variables [AddCommGroup P] [AffineSpace Points P]
variables [AddCommGroup D] [AffineSpace Points D]
variables [AddCommGroup E] [AffineSpace Points E]

variables {A B C O T M P D E : ℝ}
variables {triangle_ABC : Triangle A B C}
variables {circumcircle_O : Circle O}
variables {midpoint_MA : Midpoint M A T}
variables {P_condition : P ∈ Triangle A B C ∧ PB ⊥ PC}
variables {D_E_perpendicular: Perpendicular P A D E}
variables {BD_BP : BD = BP}
variables {CE_CP : CE = CP}
variables {AO_bisects_DE : Bisects AO D E}

theorem tangent_AO_circumcircle_AMP :
  Tangent AO (Circumcircle A M P) :=
sorry

end tangent_AO_circumcircle_AMP_l467_467083


namespace angle_relationship_l467_467086

theorem angle_relationship (α β : ℝ) (h1 : 0 < α)
                          (h2 : α < 2 * β)
                          (h3 : 2 * β ≤ π / 2)
                          (h4 : 2 * cos (α + β) * cos β = -1 + 2 * sin (α + β) * sin β) :
  α + 2 * β = 2 * π / 3 :=
by
    sorry

end angle_relationship_l467_467086


namespace tree_height_equation_l467_467268

theorem tree_height_equation (x : ℕ) : ∀ h : ℕ, h = 80 + 2 * x := by
  sorry

end tree_height_equation_l467_467268


namespace ratio_areas_ABC_ACD_l467_467253

variables {ABC ACD : Type*} [Triangle ABC] [Triangle ACD]
variables (a b c x : ℝ) (α : ℝ)
variables {r R : ℝ} (h1 : r/R = 3/4)
variables (AB BC AC CD : ℝ) (h_ABBC : AB = BC) (h_ACCD : AC = CD)
variables (angle_ACB angle_ACD : ℝ) (h_angles : angle_ACB = angle_ACD)

def ratio_of_areas (T1 T2 : Type*) [Triangle T1] [Triangle T2] : ℝ := 
  (area T1) / (area T2)

theorem ratio_areas_ABC_ACD :
  ratio_of_areas ABC ACD = 9 / 14 :=
begin
  sorry
end

end ratio_areas_ABC_ACD_l467_467253


namespace find_a_l467_467572

theorem find_a (a : ℝ) (h : (1 + a * complex.I) * complex.I = 3 + complex.I) : a = -3 :=
sorry

end find_a_l467_467572


namespace summation_series_equals_half_l467_467937

theorem summation_series_equals_half :
  (\sum_{n=0}^{∞} 2^n / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end summation_series_equals_half_l467_467937


namespace marigolds_sold_second_day_l467_467726

theorem marigolds_sold_second_day (x : ℕ) (h1 : 14 ≤ x)
  (h2 : 2 * x + 14 + x = 89) : x = 25 :=
by
  sorry

end marigolds_sold_second_day_l467_467726


namespace sum_of_prime_f_values_zero_l467_467567

def f (n : ℕ) : ℕ := n^4 - 380 * n^2 + 841

theorem sum_of_prime_f_values_zero :
  (∑ n in (range 100), if Prime (f n) then f n else 0) = 0 := 
  sorry

end sum_of_prime_f_values_zero_l467_467567


namespace total_pizzas_made_l467_467636

theorem total_pizzas_made (hc1 : Heather made 4 * Craig made on day1)
                          (hc2 : Heather made on day2 = Craig made on day2 - 20)
                          (hc3 : Craig made on day1 = 40)
                          (hc4 : Craig made on day2 = Craig made on day1 + 60) :
   Heather made on day1 + Craig made on day1 + Heather made on day2 + Craig made on day2 = 380 := 
by
  sorry

end total_pizzas_made_l467_467636


namespace domain_of_f_l467_467375

def f (x : ℝ) : ℝ := (3 * x^2) / (sqrt (1 - x)) + log (3 * x + 1)

theorem domain_of_f : ∀ x : ℝ, (x ∈ {y : ℝ | -1/3 < y ∧ y < 1} ↔ (0 < 1 - x ∧ 0 < 3 * x + 1)) := 
by
  sorry

end domain_of_f_l467_467375


namespace total_muffins_correct_l467_467649

-- Define the conditions
def boys_count := 3
def muffins_per_boy := 12
def girls_count := 2
def muffins_per_girl := 20

-- Define the question and answer
def total_muffins_for_sale : Nat :=
  boys_count * muffins_per_boy + girls_count * muffins_per_girl

theorem total_muffins_correct :
  total_muffins_for_sale = 76 := by
  sorry

end total_muffins_correct_l467_467649


namespace find_x_l467_467764

-- Definitions of the conditions
def a (x : ℝ) := x - real.sqrt 2
def b (x : ℝ) := x - 1 / x
def c (x : ℝ) := x + 1 / x
def d (x : ℝ) := x^2 + 2 * real.sqrt 2

-- Theorem statement
theorem find_x :
  ∃ x : ℝ, (∃! n : ℤ, a x ∉ ℤ) ∧ a x = x - real.sqrt 2 ∧ b x = x - 1 / x ∧ c x = x + 1 / x ∧ d x = x^2 + 2 * real.sqrt 2 ∧ x = real.sqrt 2 - 1 :=
sorry

end find_x_l467_467764


namespace find_fractions_l467_467546

noncomputable def fractions_to_sum_86_111 : Prop :=
  ∃ (a b d₁ d₂ : ℕ), 0 < a ∧ 0 < b ∧ d₁ ≤ 100 ∧ d₂ ≤ 100 ∧
  Nat.gcd a d₁ = 1 ∧ Nat.gcd b d₂ = 1 ∧
  (a: ℚ) / d₁ + (b: ℚ) / d₂ = 86 / 111

theorem find_fractions : fractions_to_sum_86_111 :=
  sorry

end find_fractions_l467_467546


namespace distinct_factors_1320_l467_467178

theorem distinct_factors_1320 :
  let p := 1320,
  prime_factors := [(2, 2), (3, 1), (5, 1), (11, 1)],
  num_factors := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in num_factors = 24 := by
  sorry

end distinct_factors_1320_l467_467178


namespace solve_for_a_l467_467583

open Complex

-- Given conditions and question in Lean 4
theorem solve_for_a (a : ℝ) (h1 : (complex.conj ((a - I) / (2 + I))) = 3 * I - (5 * I) / (2 - I)) : a = 3 := 
begin
  sorry -- proof omitted
end

end solve_for_a_l467_467583


namespace max_prime_factors_of_c_l467_467740

theorem max_prime_factors_of_c {c d : ℕ}
  (hc_pos : c > 0) (hd_pos : d > 0)
  (hgcd : (Nat.gcd c d).prime_factors.card = 8)
  (hlcm : (Nat.lcm c d).prime_factors.card = 36)
  (hc_less_than_hd : c.prime_factors.card < d.prime_factors.card) :
  c.prime_factors.card ≤ 22 := 
sorry

end max_prime_factors_of_c_l467_467740


namespace distinct_four_digit_numbers_l467_467140

theorem distinct_four_digit_numbers : 
  let digits := {1, 2, 3, 4, 5} in (digits.card = 5) →
  ∃ count : ℕ, count = 5 * 4 * 3 * 2 ∧ count = 120 := 
by
  intros
  use 5 * 4 * 3 * 2
  split
  · refl
  · exact 120

end distinct_four_digit_numbers_l467_467140


namespace color_rational_points_l467_467257

def v2 (x : ℚ) : ℤ := sorry  -- Assume v2 is defined correctly, similar to 2-adic valuation

theorem color_rational_points (n : ℕ) (h : 0 < n) : 
  ∃ (color : ℚ × ℚ → Fin n), 
  (∀ (P Q : ℚ × ℚ), P ≠ Q → ∀ j : Fin n, ∃ R ∈ closedSegment ℚ P Q, color R = j) :=
sorry

end color_rational_points_l467_467257


namespace tangent_lines_count_l467_467318

noncomputable theory

open Real

def num_tangent_lines (r1 r2 : ℝ) (no_overlap : r1 + r2 > dist (p1 : ℝ × ℝ) (p2 : ℝ × ℝ)) : ℝ :=
if r1 = r2 ∧ p1 = p2 then 0
else if dist p1 p2 = r1 + r2 then 3
else 4

theorem tangent_lines_count {r1 r2 : ℝ} {p1 p2 : ℝ × ℝ}
  (h1 : r1 = 5 ∧ r2 = 8)
  (no_overlap : r1 + r2 > dist p1 p2) :
  (num_tangent_lines r1 r2 no_overlap).card = 3 :=
begin
  sorry
end

end tangent_lines_count_l467_467318


namespace sin_shifted_angle_l467_467605

noncomputable def r : ℝ := real.sqrt ((-5:ℝ) ^ 2 + (-12:ℝ) ^ 2)

noncomputable def cos_alpha : ℝ := (-5:ℝ) / r

theorem sin_shifted_angle (α : ℝ) (h₁ : (cos_alpha = (-5:ℝ) / r)) :
  real.sin ((3 * real.pi / 2) + α) = 5 / 13 :=
by sorry

end sin_shifted_angle_l467_467605


namespace find_horizontal_length_l467_467769

variable (v h : ℝ)

-- Conditions
def is_horizontal_length_of_rectangle_perimeter_54_and_vertical_plus_3 (v h : ℝ) : Prop :=
  2 * h + 2 * v = 54 ∧ h = v + 3

-- The proof we aim to show
theorem find_horizontal_length (v h : ℝ) :
  is_horizontal_length_of_rectangle_perimeter_54_and_vertical_plus_3 v h → h = 15 :=
by
  sorry

end find_horizontal_length_l467_467769


namespace p_work_alone_time_l467_467833

variable (Wp Wq : ℝ)
variable (x : ℝ)

-- Conditions
axiom h1 : Wp = 1.5 * Wq
axiom h2 : (1 / x) + (Wq / Wp) * (1 / x) = 1 / 15

-- Proof of the question (p alone can complete the work in x days)
theorem p_work_alone_time : x = 25 :=
by
  -- Add your proof here
  sorry

end p_work_alone_time_l467_467833


namespace sally_initial_cards_l467_467345

theorem sally_initial_cards (X : ℕ) (h1 : X + 41 + 20 = 88) : X = 27 :=
by
  -- Proof goes here
  sorry

end sally_initial_cards_l467_467345


namespace distinct_four_digit_numbers_l467_467146

theorem distinct_four_digit_numbers : 
  {n : ℕ | ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3, 4, 5} ∧ c ∈ {1, 2, 3, 4, 5} ∧ d ∈ {1, 2, 3, 4, 5}
  } = 120 :=
by
  sorry

end distinct_four_digit_numbers_l467_467146


namespace lying_flat_points_relation_l467_467793

noncomputable def g (x : ℝ) := Real.exp x - x
noncomputable def h (x : ℝ) := Real.log x
noncomputable def ϕ (x : ℝ) := 2023 * x + 2023

noncomputable def g_lying_flat_point : ℝ := 1 -- a = 1
noncomputable def h_lying_flat_point : ℝ := sorry -- b in (1, e)
noncomputable def ϕ_lying_flat_point : ℝ := 0 -- c = 0

theorem lying_flat_points_relation (a b c : ℝ) 
  (hg : g_lying_flat_point = a)
  (hh : h_lying_flat_point = b)
  (hϕ : ϕ_lying_flat_point = c) : 
  b > a ∧ a > c :=
by
  rw [hg, hh, hϕ]
  split
  . -- Proof that b > a
    sorry
  . -- Proof that a > c
    exact zero_lt_one

end lying_flat_points_relation_l467_467793


namespace find_fractions_l467_467547

noncomputable def fractions_to_sum_86_111 : Prop :=
  ∃ (a b d₁ d₂ : ℕ), 0 < a ∧ 0 < b ∧ d₁ ≤ 100 ∧ d₂ ≤ 100 ∧
  Nat.gcd a d₁ = 1 ∧ Nat.gcd b d₂ = 1 ∧
  (a: ℚ) / d₁ + (b: ℚ) / d₂ = 86 / 111

theorem find_fractions : fractions_to_sum_86_111 :=
  sorry

end find_fractions_l467_467547


namespace sum_of_unit_fractions_l467_467443

theorem sum_of_unit_fractions : (1 / 2) + (1 / 3) + (1 / 7) + (1 / 42) = 1 := 
by 
  sorry

end sum_of_unit_fractions_l467_467443


namespace trains_crossing_time_l467_467428

-- Definitions based on given conditions
noncomputable def length_A : ℝ := 2500
noncomputable def time_A : ℝ := 50
noncomputable def length_B : ℝ := 3500
noncomputable def speed_factor : ℝ := 1.2

-- Speed computations
noncomputable def speed_A : ℝ := length_A / time_A
noncomputable def speed_B : ℝ := speed_A * speed_factor

-- Relative speed when moving in opposite directions
noncomputable def relative_speed : ℝ := speed_A + speed_B

-- Total distance covered when crossing each other
noncomputable def total_distance : ℝ := length_A + length_B

-- Time taken to cross each other
noncomputable def time_to_cross : ℝ := total_distance / relative_speed

-- Proof statement: Time taken is approximately 54.55 seconds
theorem trains_crossing_time :
  |time_to_cross - 54.55| < 0.01 := by
  sorry

end trains_crossing_time_l467_467428


namespace find_f_at_1_l467_467608

theorem find_f_at_1 :
  (∀ x : ℝ, f (2 * x - 1) = 4 * x^2) → f 1 = 4 :=
by
  intro h
  -- Sorry is used here as a placeholder for the proof
  sorry

end find_f_at_1_l467_467608


namespace shadow_problem_l467_467862

theorem shadow_problem 
  (base_side : ℝ := 2) 
  (pyramid_height : ℝ) 
  (shadow_area : ℝ := 36)
  (y : ℝ := (2 * (Float.sqrt 10 + 1)) / 9) :
  y = pyramid_height → 
  let integer_part := (800 * y).toInt in
  integer_part = 828 :=
by
  intro h
  sorry

end shadow_problem_l467_467862


namespace existence_of_triangle_l467_467987

theorem existence_of_triangle (n : ℕ) (hn : n ≥ 3) (points : fin n → ℝ × ℝ) :
  ∃ A B C : fin n, A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (∀ (P : fin n), P ≠ A ∧ P ≠ B → ¬ collinear (points A) (points B) (points P)) ∧
    (∀ (P : fin n), P ≠ A ∧ P ≠ C → ¬ collinear (points A) (points C) (points P)) ∧
    (∀ (P : fin n), P ≠ B ∧ P ≠ C → ¬ collinear (points B) (points C) (points P)) :=
begin
  sorry
end

-- Helper function to define collinearity of three points
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  let (x3, y3) := p3 in
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

end existence_of_triangle_l467_467987


namespace distinct_four_digit_numbers_l467_467137

theorem distinct_four_digit_numbers : 
  let digits := {1, 2, 3, 4, 5} in (digits.card = 5) →
  ∃ count : ℕ, count = 5 * 4 * 3 * 2 ∧ count = 120 := 
by
  intros
  use 5 * 4 * 3 * 2
  split
  · refl
  · exact 120

end distinct_four_digit_numbers_l467_467137


namespace anya_lost_games_l467_467053

-- Define the girls playing table tennis
def girls : Type := {Anya Bella Valya Galya Dasha : girls}

-- Define the conditions
def games_played : girls → ℕ
| Anya => 4
| Bella => 6
| Valya => 7
| Galya => 10
| Dasha => 11

-- Total number of games played
def total_games : ℕ := 19
#eval total_games = (games_played Anya + games_played Bella + games_played Valya + games_played Galya + games_played Dasha) / 2

-- The main theorem to be proven
theorem anya_lost_games :
  ∀ i : ℕ, i ∈ {4, 8, 12, 16} ↔ Anya played and lost in game i
:=
by
  sorry

end anya_lost_games_l467_467053


namespace distinct_four_digit_numbers_count_l467_467135

theorem distinct_four_digit_numbers_count :
  (Finset.card (Finset.filter (λ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ (∀ d ∈ [x / 1000, (x / 100) % 10, (x / 10) % 10, x % 10], d ∈ [1, 2, 3, 4, 5]) ∧ (denote_distinct_digits (x)) ) (Finset.Ico 1000 10000))) = 120 := 
sorry

def denote_distinct_digits (x : ℕ) : Prop :=
  ∀ i j, (i ≠ j) → (x / (10 ^ i)) % 10 ≠ (x / (10 ^ j)) % 10

end distinct_four_digit_numbers_count_l467_467135


namespace Daniela_is_12_years_old_l467_467900

noncomputable def auntClaraAge : Nat := 60

noncomputable def evelinaAge : Nat := auntClaraAge / 3

noncomputable def fidelAge : Nat := evelinaAge - 6

noncomputable def caitlinAge : Nat := fidelAge / 2

noncomputable def danielaAge : Nat := evelinaAge - 8

theorem Daniela_is_12_years_old (h_auntClaraAge : auntClaraAge = 60)
                                (h_evelinaAge : evelinaAge = 60 / 3)
                                (h_fidelAge : fidelAge = (60 / 3) - 6)
                                (h_caitlinAge : caitlinAge = ((60 / 3) - 6) / 2)
                                (h_danielaAge : danielaAge = (60 / 3) - 8) :
  danielaAge = 12 := 
  sorry

end Daniela_is_12_years_old_l467_467900


namespace cake_box_height_proof_l467_467506

-- Define the dimensions of the carton
def carton_length := 25
def carton_width := 42
def carton_height := 60

-- Define the dimensions of the cake box (base only, height to be determined)
def cake_box_base_length := 8
def cake_box_base_width := 7

-- Define the maximum number of cake boxes that can fit in the carton
def max_cake_boxes := 210

-- Define the height of the cake box (the value we need to prove)
def cake_box_height := 5

-- Calculate number of cake boxes that fit along each dimension
def boxes_along_length := carton_length / cake_box_base_length
def boxes_along_width := carton_width / cake_box_base_width
def boxes_in_base_area := boxes_along_length * boxes_along_width

-- Calculate the number of layers that fit within the carton height
def layers := max_cake_boxes / boxes_in_base_area

theorem cake_box_height_proof :
  (carton_height / layers).to_nat = cake_box_height := by
  sorry

end cake_box_height_proof_l467_467506


namespace stuart_chords_l467_467358

-- Definitions and conditions based on the given problem
def concentric_circles (small large : Circle) : Prop :=
  small.center = large.center ∧ small.radius < large.radius

variables {C1 C2 : Circle} (P Q : Point)

-- Given conditions
def conditions (C1 C2 : Circle) (P Q : Point) : Prop :=
  concentric_circles C1 C2 ∧ tangent C1.line PQ ∧ angle P Q = 60

-- Statement of the proof problem:
theorem stuart_chords (C1 C2 : Circle) (P Q : Point) (h : conditions C1 C2 P Q) : 
  ∃ n : ℕ, n = 3 := sorry

end stuart_chords_l467_467358


namespace solve_quadratic_l467_467736

theorem solve_quadratic (x : ℝ) (h₁ : x > 0) (h₂ : 3 * x^2 + 7 * x - 20 = 0) : x = 5 / 3 :=
sorry

end solve_quadratic_l467_467736


namespace sum_and_product_of_divisors_l467_467444

theorem sum_and_product_of_divisors (p : ℕ) (hp : Nat.Prime p) :
  let a := p ^ 100
  let Sigma_d := ∑ i in Finset.range 101, p ^ i
  let Pi_d := ∏ i in Finset.range 101, p ^ i
  Sigma_d = (p ^ 101 - 1) / (p - 1) ∧ Pi_d = a ^ 50 := by
  sorry

end sum_and_product_of_divisors_l467_467444


namespace distinct_four_digit_numbers_l467_467141

theorem distinct_four_digit_numbers : 
  let digits := {1, 2, 3, 4, 5} in (digits.card = 5) →
  ∃ count : ℕ, count = 5 * 4 * 3 * 2 ∧ count = 120 := 
by
  intros
  use 5 * 4 * 3 * 2
  split
  · refl
  · exact 120

end distinct_four_digit_numbers_l467_467141


namespace max_siskins_on_poles_l467_467879

-- Definitions based on problem conditions
def pole : Type := ℕ
def siskins (poles : pole) : Prop := poles ≤ 25
def adjacent (p₁ p₂ : pole) : Prop := (p₁ = p₂ + 1) ∨ (p₁ = p₂ - 1)

-- Given conditions
def conditions (p : pole → bool) : Prop :=
  ∀ p₁ p₂ : pole, p p₁ = true → p p₂ = true → adjacent p₁ p₂ → false

-- Main problem statement
theorem max_siskins_on_poles : ∃ p : pole → bool, (∀ i : pole, p i = true → siskins i) ∧ (conditions p) ∧ (∑ i in finset.range 25, if p i then 1 else 0 = 24) :=
begin
  sorry
end

end max_siskins_on_poles_l467_467879


namespace simplification_of_expression_l467_467974

theorem simplification_of_expression (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (k+a+b+c)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (ab+bc+ca)⁻¹ * ((ab)⁻² + (bc)⁻² + (ca)⁻²) =
   a⁻³ * b⁻³ * c⁻³ * (k+a+b+c)⁻¹ := by
  sorry

end simplification_of_expression_l467_467974


namespace tree_height_relationship_l467_467267

theorem tree_height_relationship (x : ℕ) : ∃ h : ℕ, h = 80 + 2 * x :=
by
  sorry

end tree_height_relationship_l467_467267


namespace parallel_lines_slope_eq_l467_467236

theorem parallel_lines_slope_eq (a : ℝ) :
  (a ≠ -1) ∧ (∀ x y : ℝ, ax + y - 1 = 0 → 2x + (a-1)y + 2 = 0) → a = 2 :=
by
  intro h
  sorry

end parallel_lines_slope_eq_l467_467236


namespace domain_f_l467_467540

noncomputable def f (x : ℝ) : ℝ := real.log (3 - 4 * real.sin x ^ 2)

theorem domain_f (x : ℝ) (k : ℤ) :
  (3 - 4 * real.sin x ^ 2 > 0) ↔ 
  ∀ k : ℤ, ((2 * k * Real.pi - Real.pi / 3 < x) ∧ (x < 2 * k * Real.pi + Real.pi / 3)) ∨
  ((2 * k * Real.pi + 2 * Real.pi / 3 < x) ∧ (x < 2 * k * Real.pi + 4 * Real.pi / 3)) :=
by { sorry }

end domain_f_l467_467540


namespace powers_of_two_solution_l467_467942

noncomputable def log : ℝ → ℝ → ℝ := real.log_base

theorem powers_of_two_solution :
    (∀ x : ℝ, (∃ k : ℕ, x = 2 ^ k) → (log 2 x * log x 6 = log 2 6) ↔ (x = 2 ∨ x = 4)) :=
begin
  sorry
end

end powers_of_two_solution_l467_467942


namespace washing_machines_removed_correct_l467_467248

-- Define the conditions
def crates : ℕ := 10
def boxes_per_crate : ℕ := 6
def washing_machines_per_box : ℕ := 4
def washing_machines_removed_per_box : ℕ := 1

-- Define the initial and final states
def initial_washing_machines_in_crate : ℕ := boxes_per_crate * washing_machines_per_box
def initial_washing_machines_in_container : ℕ := crates * initial_washing_machines_in_crate

def final_washing_machines_in_box : ℕ := washing_machines_per_box - washing_machines_removed_per_box
def final_washing_machines_in_crate : ℕ := boxes_per_crate * final_washing_machines_in_box
def final_washing_machines_in_container : ℕ := crates * final_washing_machines_in_crate

-- Number of washing machines removed
def washing_machines_removed : ℕ := initial_washing_machines_in_container - final_washing_machines_in_container

-- Theorem statement in Lean 4
theorem washing_machines_removed_correct : washing_machines_removed = 60 := by
  sorry

end washing_machines_removed_correct_l467_467248


namespace Mrs_Siroka_expected_11_guests_l467_467315

def sandwiches_and_guests (sandwiches guests : ℕ) : Prop :=
  ((sandwiches % 2 = 1) ∧ (25 / 2 = guests / (if sandwiches = 25 then 2 else if sandwiches = 35 then 3 else 4))
   ∧ (25 / 3 > guests) ∧ (35 / 3 > guests) ∧ (35 / 4 > guests) 
   ∧ (guests ∈ (9, 10, 11)))

theorem Mrs_Siroka_expected_11_guests : sandwiches_and_guests 11 := by
  sorry

end Mrs_Siroka_expected_11_guests_l467_467315


namespace billy_soda_distribution_l467_467509

theorem billy_soda_distribution (sisters : ℕ) (brothers : ℕ) (total_sodas : ℕ) (total_siblings : ℕ)
  (h1 : total_sodas = 12)
  (h2 : sisters = 2)
  (h3 : brothers = 2 * sisters)
  (h4 : total_siblings = sisters + brothers) :
  total_sodas / total_siblings = 2 :=
by
  sorry

end billy_soda_distribution_l467_467509


namespace problem_statement_l467_467294

open Polynomial

noncomputable def q : Polynomial ℤ := ∑ i in (finset.range 2011), (x : Polynomial ℤ) ^ i

noncomputable def divisor : Polynomial ℤ := x^5 + x^4 + 2 * x^3 + 3 * x^2 + x + 1

noncomputable def s : Polynomial ℤ := q % divisor

theorem problem_statement :
  |eval 2010 s| % 1000 = 111 :=
sorry

end problem_statement_l467_467294


namespace compute_f_at_2012_l467_467296

noncomputable def B := { x : ℚ | x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 2 }

noncomputable def h (x : ℚ) : ℚ := 2 - (1 / x)

noncomputable def f (x : B) : ℝ := sorry  -- As a placeholder since the definition isn't given directly

-- Main theorem
theorem compute_f_at_2012 : 
  (∀ x : B, f x + f ⟨h x, sorry⟩ = Real.log (abs (2 * (x : ℚ)))) →
  f ⟨2012, sorry⟩ = Real.log ((4024 : ℚ) / (4023 : ℚ)) :=
sorry

end compute_f_at_2012_l467_467296


namespace distinct_four_digit_numbers_count_l467_467134

theorem distinct_four_digit_numbers_count :
  (Finset.card (Finset.filter (λ (x : ℕ), 1000 ≤ x ∧ x < 10000 ∧ (∀ d ∈ [x / 1000, (x / 100) % 10, (x / 10) % 10, x % 10], d ∈ [1, 2, 3, 4, 5]) ∧ (denote_distinct_digits (x)) ) (Finset.Ico 1000 10000))) = 120 := 
sorry

def denote_distinct_digits (x : ℕ) : Prop :=
  ∀ i j, (i ≠ j) → (x / (10 ^ i)) % 10 ≠ (x / (10 ^ j)) % 10

end distinct_four_digit_numbers_count_l467_467134


namespace polar_coordinates_l467_467109

theorem polar_coordinates (x y z : ℝ) (h₁ : x = 1) (h₂ : y = sqrt 3) (h₃ : z = 3) :
  ∃ (ρ θ : ℝ), (ρ, θ, z) = (2, π / 3, 3) ∧ x = ρ * cos θ ∧ y = ρ * sin θ :=
by
  use 2, π / 3
  split
  sorry
  sorry

end polar_coordinates_l467_467109


namespace find_a_l467_467574

-- Conditions
variables (a : ℝ)
def imaginary_unit := complex.I
def given_condition := (1 + a * imaginary_unit) * imaginary_unit = 3 + imaginary_unit

-- Theorem Statement
theorem find_a (h : given_condition) : a = -3 :=
sorry

end find_a_l467_467574


namespace usual_time_is_36_l467_467455

-- Definition: let S be the usual speed of the worker (not directly relevant to the final proof)
noncomputable def S : ℝ := sorry

-- Definition: let T be the usual time taken by the worker
noncomputable def T : ℝ := sorry

-- Condition: The worker's speed is (3/4) of her normal speed, resulting in a time (T + 12)
axiom speed_delay_condition : (3 / 4) * S * (T + 12) = S * T

-- Theorem: Prove that the usual time T taken to cover the distance is 36 minutes
theorem usual_time_is_36 : T = 36 := by
  -- Formally stating our proof based on given conditions
  sorry

end usual_time_is_36_l467_467455


namespace series_sum_l467_467935

theorem series_sum :
  ∑' n : ℕ, (2 ^ n) / (3 ^ (2 ^ n) + 1) = 1 / 2 :=
sorry

end series_sum_l467_467935


namespace Anya_loss_games_l467_467020

noncomputable def gamesPlayed : Nat := 19

constant Anya_games : Nat := 4
constant Bella_games : Nat := 6
constant Valya_games : Nat := 7
constant Galya_games : Nat := 10
constant Dasha_games : Nat := 11

constant gameResults : Fin 19 → Option String
constant lost (g : Nat) (p : String) : Prop

axiom Anya_game_indices : ∀ (i : Fin 4), lost (gameResults i.val) "Anya" → i.val = 4 * i + 4

theorem Anya_loss_games : list Nat :=
have Anya_lost_games : list Nat := [4, 8, 12, 16],
Anya_lost_games

end Anya_loss_games_l467_467020


namespace smallest_positive_period_of_f_max_min_of_f_on_interval_l467_467121

noncomputable def f (x : ℝ) : ℝ :=
  sin x * cos x + sqrt 3 * cos (π - x) * cos x

theorem smallest_positive_period_of_f : ∃ T > 0, T = π ∧ (∀ x, f (x + T) = f x) := 
by
  use π
  sorry

theorem max_min_of_f_on_interval :
  (∀ x ∈ set.Icc (0 : ℝ) (π / 2), f x ≤ (1 - sqrt 3 / 2)) ∧
  (∀ x ∈ set.Icc (0 : ℝ) (π / 2), (-sqrt 3) ≤ f x) :=
by
  sorry

end smallest_positive_period_of_f_max_min_of_f_on_interval_l467_467121


namespace tan_sum_l467_467460

open Real

theorem tan_sum 
  (α β γ θ φ : ℝ)
  (h1 : tan θ = (sin α * cos γ - sin β * sin γ) / (cos α * cos γ - cos β * sin γ))
  (h2 : tan φ = (sin α * sin γ - sin β * cos γ) / (cos α * sin γ - cos β * cos γ)) : 
  tan (θ + φ) = tan (α + β) :=
by
  sorry

end tan_sum_l467_467460


namespace min_value_neg_infty_0_l467_467603

noncomputable section

open Function

-- Definitions of odd functions
def is_odd (f: ℝ → ℝ): Prop := ∀ x: ℝ, f(-x) = -f(x)

-- Given conditions
variables {f g: ℝ → ℝ} {a b: ℝ}
hypothesis (hf_odd: is_odd f)
hypothesis (hg_odd: is_odd g)
hypothesis (h_max: ∀ x > 0, a * f x + b * g x + 2 ≤ 5)

-- Proving the required result
theorem min_value_neg_infty_0: (∀ x < 0, a * f x + b * g x + 2 ≥ -1) :=
by
  sorry

end min_value_neg_infty_0_l467_467603


namespace ferris_wheel_cost_l467_467692

variable {tickets_total : ℕ}
variable {tickets_roller_coaster : ℕ := 4}
variable {tickets_bumper_cars : ℕ := 4}
variable {tickets_initial : ℕ := 5}
variable {tickets_to_buy : ℕ := 8}
variable {tickets_ferries_wheel : ℕ}

theorem ferris_wheel_cost:
  tickets_ferries_wheel = 5 :=
  by
  sorry

end ferris_wheel_cost_l467_467692


namespace num_distinct_factors_1320_l467_467164

theorem num_distinct_factors_1320 : 
  ∃ n : ℕ, n = 1320 ∧ (finset.univ.filter (λ d, d > 0 ∧ 1320 % d = 0)).card = 32 := 
sorry

end num_distinct_factors_1320_l467_467164


namespace basketball_team_starting_lineups_l467_467468

theorem basketball_team_starting_lineups (n : ℕ) (h_n : n = 12) :
  let point_guard_choices := nat.choose n 1,
  remaining_player_choices := nat.choose (n - 1) 5 in
  point_guard_choices * remaining_player_choices = 5544 :=
by
  sorry

end basketball_team_starting_lineups_l467_467468


namespace relationship_between_roses_and_total_flowers_l467_467244

variables (C V T R F : ℝ)
noncomputable def F_eq_64_42376521116678_percent_of_C := 
  C = 0.6442376521116678 * F

def V_eq_one_third_of_C := 
  V = (1 / 3) * C

def T_eq_one_ninth_of_C := 
  T = (1 / 9) * C

def F_eq_C_plus_V_plus_T_plus_R := 
  F = C + V + T + R

theorem relationship_between_roses_and_total_flowers (C V T R F : ℝ) 
    (h1 : C = 0.6442376521116678 * F)
    (h2 : V = 1 / 3 * C)
    (h3 : T = 1 / 9 * C)
    (h4 : F = C + V + T + R) :
    R = F - 13 / 9 * C := 
  by sorry

end relationship_between_roses_and_total_flowers_l467_467244


namespace distinct_factors_1320_l467_467202

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end distinct_factors_1320_l467_467202


namespace determinant_is_one_l467_467534

noncomputable def evalDet : ℝ :=
  Matrix.det ![
    #[Real.cos (α + β), Real.sin (α + β), -Real.sin α],
    #[-Real.sin β,        Real.cos β,          0 ],
    #[Real.sin α * Real.cos β, Real.sin α * Real.sin β, Real.cos α]
  ]

theorem determinant_is_one (α β : ℝ) : evalDet = 1 := by
  sorry

end determinant_is_one_l467_467534


namespace sequence_non_positive_l467_467719

theorem sequence_non_positive
  (a : ℕ → ℝ) (n : ℕ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h : ∀ k, 1 ≤ k → k ≤ n - 1 → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) :
  ∀ k, k ≤ n → a k ≤ 0 := 
sorry

end sequence_non_positive_l467_467719


namespace find_a_l467_467573

theorem find_a (a : ℝ) (h : (1 + a * complex.I) * complex.I = 3 + complex.I) : a = -3 :=
sorry

end find_a_l467_467573


namespace toy_car_difference_l467_467725

noncomputable def uncle_cars : ℕ := by sorry

noncomputable def auntie_cars : ℕ := 6

theorem toy_car_difference :
  let U := uncle_cars in
  let initial_cars := 150 in
  let grandpa_cars := 2 * U in
  let dad_cars := 10 in
  let mum_cars := 15 in
  initial_cars + grandpa_cars + U + dad_cars + mum_cars + auntie_cars = 196 →
  auntie_cars - U = 1 :=
by {
  sorry
}

end toy_car_difference_l467_467725


namespace no_more_than_five_planes_land_l467_467659

def distinct_distances_between_each_pair (cities : List ℕ) : Prop :=
  ∀ i j, i ≠ j → cities.nth i ≠ cities.nth j

def nearest_neighbor_landings (cities : List ℕ) (plane_landing : ℕ → ℕ) :=
  ∀ i, plane_landing i = argmin (λ j, if i ≠ j then distance (cities.nth i) (cities.nth j) else ∞)

theorem no_more_than_five_planes_land
  (cities : List ℕ)
  (h1 : distinct_distances_between_each_pair cities)
  (h2 : ∀ i, nearest_neighbor_landings cities i) :
  ∀ p, (∃ A1 A2 A3 A4 A5 A6, ∀ Ai, plane_landing Ai = p) → False :=
begin
  sorry
end

end no_more_than_five_planes_land_l467_467659


namespace max_knights_5x5_l467_467433

-- Define the size of the board
def board_size : Nat := 5

-- Define the condition that each knight must attack exactly two others
def knight_attacks_two (placement : List (Fin board_size × Fin board_size)) : Prop :=
  ∀ (pos : Fin board_size × Fin board_size), pos ∈ placement → 
    (∃! (other1 : Fin board_size × Fin board_size), 
      other1 ∈ placement ∧ knight_move pos other1) ∧
    (∃! (other2 : Fin board_size × Fin board_size), 
      other2 ∈ placement ∧ knight_move pos other2 ∧ other2 ≠ other1)

-- Define the knight's move
def knight_move (a b : Fin board_size × Fin board_size) : Prop :=
  let dx := abs (a.1.val - b.1.val) in
  let dy := abs (a.2.val - b.2.val) in
  (dx = 2 ∧ dy = 1) ∨ (dx = 1 ∧ dy = 2)

-- Statement of the problem
theorem max_knights_5x5 :
  ∃ (placement : List (Fin board_size × Fin board_size)), 
    length placement = 9 ∧ knight_attacks_two placement :=
sorry

end max_knights_5x5_l467_467433


namespace value_of_x_l467_467406

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 96) : x = 8 := 
by
  sorry

end value_of_x_l467_467406


namespace sum_of_solutions_sqrt_eq_8_l467_467823

theorem sum_of_solutions_sqrt_eq_8 : ∀ x, (sqrt ((x + 5)^2) = 8) → (x = 3 ∨ x = -13) → 3 + (-13) = -10 :=
by
  intros
  sorry

end sum_of_solutions_sqrt_eq_8_l467_467823


namespace sum_of_positive_integers_nu_lcm_72_l467_467810

theorem sum_of_positive_integers_nu_lcm_72:
  let ν_values := { ν | Nat.lcm ν 24 = 72 }
  ∑ ν in ν_values, ν = 180 := by
  sorry

end sum_of_positive_integers_nu_lcm_72_l467_467810


namespace triangle_area_double_tangent_area_l467_467328

theorem triangle_area_double_tangent_area
  (t_A t_B t_C : ℝ) :
  let A := (t_A^2, 2 * t_A),
      B := (t_B^2, 2 * t_B),
      C := (t_C^2, 2 * t_C),
      P := (t_B * t_C, t_B + t_C),
      Q := (t_C * t_A, t_C + t_A),
      R := (t_A * t_B, t_A + t_B) in
  let area_triangle (p1 p2 p3 : ℝ × ℝ) :=
    0.5 * abs (
      p1.1 * (p2.2 - p3.2) +
      p2.1 * (p3.2 - p1.2) +
      p3.1 * (p1.2 - p2.2)) in
  area_triangle A B C = 2 * area_triangle P Q R :=
sorry

end triangle_area_double_tangent_area_l467_467328


namespace directrix_equation_l467_467586

theorem directrix_equation (p m : ℝ) (h1 : p > 0)
(h2 : ∀ A B F : ℝ × ℝ, ∃ m > 0, dist A F = 3 * dist B F ∧ line_through F A ∧ line_through F B)
(h3 : ∃ C : ℝ × ℝ, C.1 = 0 ∧ C.2 = 0)
(h4 : ∃ A A1 : ℝ × ℝ, dist A A1 = dist A C ∧ right_angle A A1 C)
(h5 : let A A1 C F : ℝ × ℝ in area_quadrilateral A A1 C F = 12 * Real.sqrt 3)
: directrix l x = -Real.sqrt 2 := sorry

end directrix_equation_l467_467586


namespace cylinder_surface_area_l467_467475

theorem cylinder_surface_area (h : ℝ) (c : ℝ) (r : ℝ) 
  (h_eq : h = 2) (c_eq : c = 2 * Real.pi) (circumference_formula : c = 2 * Real.pi * r) : 
  2 * (Real.pi * r^2) + (2 * Real.pi * r * h) = 6 * Real.pi := 
by
  sorry

end cylinder_surface_area_l467_467475


namespace evaluate_f_at_5_l467_467429

def f (x : ℕ) : ℕ := x^5 + 2 * x^4 + 3 * x^3 + 4 * x^2 + 5 * x + 6

theorem evaluate_f_at_5 : f 5 = 4881 :=
by
-- proof
sorry

end evaluate_f_at_5_l467_467429


namespace point_in_fourth_quadrant_l467_467672

def point : ℝ × ℝ := (4, -3)

def is_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l467_467672


namespace sum_lcms_equals_l467_467805

def is_solution (ν : ℕ) : Prop := Nat.lcm ν 24 = 72

theorem sum_lcms_equals :
  ( ∑ ν in (Finset.filter is_solution (Finset.range 100)), ν ) = 180 :=
sorry

end sum_lcms_equals_l467_467805


namespace find_y_parallel_l467_467225

-- Definitions
def a : ℝ × ℝ := (2, 3)
def b (y : ℝ) : ℝ × ℝ := (4, -1 + y)

-- Parallel condition implies proportional components
def parallel_vectors (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- The proof problem
theorem find_y_parallel : ∀ y : ℝ, parallel_vectors a (b y) → y = 7 :=
by
  sorry

end find_y_parallel_l467_467225


namespace max_siskins_on_poles_l467_467877

theorem max_siskins_on_poles : 
    (∀ (poles : List Nat), length poles = 25 → 
        ∀ (siskins_on_poles : List (Option Nat)), 
        length siskins_on_poles = 25 → 
        (∀ (i : Nat), (i < 25) → (∀ (j : Nat), (abs (j - i) = 1) → 
          (siskins_on_poles.nth i = some _) → (siskins_on_poles.nth j ≠ some _))) → 
        (∃ k, ∀ n, (n < 25) → 
            n + k ≤ 24 → 
            ∃ m, (siskins_on_poles.nth m = some m) ∧ 
              (m < 25) ∧ 
              (∀ l, (m ≠ l) → (siskins_on_poles.nth l ≠ some _)))) → true) := 
by sorry

end max_siskins_on_poles_l467_467877


namespace series_sum_l467_467934

theorem series_sum :
  ∑' n : ℕ, (2 ^ n) / (3 ^ (2 ^ n) + 1) = 1 / 2 :=
sorry

end series_sum_l467_467934


namespace perimeter_of_regular_nonagon_l467_467435

def regular_nonagon_side_length := 3
def number_of_sides := 9

theorem perimeter_of_regular_nonagon (h1 : number_of_sides = 9) (h2 : regular_nonagon_side_length = 3) :
  9 * 3 = 27 :=
by
  sorry

end perimeter_of_regular_nonagon_l467_467435


namespace find_two_fractions_sum_eq_86_over_111_l467_467565

theorem find_two_fractions_sum_eq_86_over_111 :
  ∃ (a1 a2 d1 d2 : ℕ), 
    (0 < d1 ∧ d1 ≤ 100) ∧ 
    (0 < d2 ∧ d2 ≤ 100) ∧ 
    (nat.gcd a1 d1 = 1) ∧ 
    (nat.gcd a2 d2 = 1) ∧ 
    (↑a1 / ↑d1 + ↑a2 / ↑d2 = 86 / 111) ∧
    (a1 = 2 ∧ d1 = 3) ∧ 
    (a2 = 4 ∧ d2 = 37) := 
by
  sorry

end find_two_fractions_sum_eq_86_over_111_l467_467565


namespace find_m_l467_467478

-- Define the given conditions
variables {x1 x2 m : ℝ}
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Points A and B on the parabola
def A : ℝ × ℝ := (x1, parabola x1)
def B : ℝ × ℝ := (x2, parabola x2)

-- Condition: Points are symmetric about the line y = x + m
def symmetric_about := (A : ℝ × ℝ) (B : ℝ × ℝ) (m : ℝ) : Prop :=
  let x_mid := (A.1 + B.1) / 2 in
  let y_mid := (A.2 + B.2) / 2 in
  y_mid = x_mid + m

-- x1 * x2 = -1/2
def product_condition := x1 * x2 = -1 / 2

-- Prove the value of m
theorem find_m (h_symm : symmetric_about A B m) (h_prod : product_condition) : m = 3 / 2 :=
sorry

end find_m_l467_467478


namespace sum_of_nus_is_45_l467_467800

noncomputable def sum_of_valid_nu : ℕ :=
  ∑ ν in {ν | ν > 0 ∧ Nat.lcm ν 24 = 72}.toFinset, ν

theorem sum_of_nus_is_45 : sum_of_valid_nu = 45 := by
  sorry

end sum_of_nus_is_45_l467_467800


namespace total_matchsticks_l467_467011

theorem total_matchsticks (x y z : ℕ) (hx : x = 10) (hy : y = 50) (hz : z = 600) : x * y * z = 300000 :=
by
  rw [hx, hy, hz]
  norm_num

end total_matchsticks_l467_467011


namespace fixed_line_and_orthocenter_l467_467364

theorem fixed_line_and_orthocenter
    {a b u c : ℝ}
    {θ1 θ2 θ3 θ4 : ℝ}
    (h_ellipse : a > 0 ∧ b > 0)
    (h_u : u ≠ 0)
    (h_A : ∃ (θ : ℝ), A = (a * cos θ, b * sin θ))
    (h_B : ∃ (θ : ℝ), B = (a * cos θ, b * sin θ))
    (h_C : ∃ (θ : ℝ), C = (a * cos θ, b * sin θ))
    (h_D : ∃ (θ : ℝ), D = (a * cos θ, b * sin θ))
    (h_K : K = (u, 0))
    (h_K_AC : ∃ (A C : ℝ × ℝ), collinear A K C)
    (h_K_BD : ∃ (B D : ℝ × ℝ), collinear B K D)
    (h_M : M = line_intersection (line_through A B) (line_through D C))
    (h_N : N = line_intersection (line_through A D) (line_through B C)) :
    M.1 = a^2 / u ∧ N.1 = a^2 / u ∧ orthocenter_triangle KMN = (c^2 / u, 0) :=
by
  sorry

end fixed_line_and_orthocenter_l467_467364


namespace range_of_a_l467_467913

/--
Let f be a function defined on the interval [-1, 1] that is increasing and odd.
If f(-a+1) + f(4a-5) > 0, then the range of the real number a is (4/3, 3/2].
-/
theorem range_of_a
  (f : ℝ → ℝ)
  (h_dom : ∀ x, -1 ≤ x ∧ x ≤ 1 → f x = f x)  -- domain condition
  (h_incr : ∀ x y, x < y → f x < f y)          -- increasing condition
  (h_odd : ∀ x, f (-x) = - f x)                -- odd function condition
  (a : ℝ)
  (h_ineq : f (-a + 1) + f (4 * a - 5) > 0) :
  4 / 3 < a ∧ a ≤ 3 / 2 :=
sorry

end range_of_a_l467_467913


namespace distance_AE_BF_l467_467661

def vec3 := (ℝ × ℝ × ℝ)

def distance_between_skew_lines 
(AB AD AA1 : ℝ)
(A B E F : vec3)
(h_coord_B : B = (AB, 0, 0))
(h_coord_A1 : A = (0, 0, 0))
(h_coord_E : E = (30, 0, 15)) 
(h_coord_F : F = (60, 15, 15)) 
: ℝ := sorry

theorem distance_AE_BF :
∀ (A B E F : vec3),
B = (60, 0, 0) ∧
A = (0, 0, 0) ∧ 
E = (30, 0, 15) ∧ 
F = (60, 15, 15) ∧ 
distance_between_skew_lines 60 30 15 A B E F = 20 :=
begin
  -- Proof to be filled
  sorry
end

end distance_AE_BF_l467_467661


namespace determine_expression_for_f_find_t_value_l467_467362

-- Defining conditions as variables in Lean 4
variable (t : ℝ) (t_ne_zero : t ≠ 0)
def f (x : ℝ) := (x - (t + 2) / 2) ^ 2 - t ^ 2 / 4

-- Problem (1)
theorem determine_expression_for_f :
    f t_ne_zero t = (x : ℝ) -> (x - (t + 2) / 2) ^ 2 - t ^ 2 / 4 := 
sorry

-- Problem (2)
theorem find_t_value :
    (∀ x ∈ Set.Icc (-1 : ℝ) (1 / 2), f t_ne_zero t x ≥ -5) →
    (∃ t : ℝ, t = -9 / 2) :=
sorry

end determine_expression_for_f_find_t_value_l467_467362


namespace unique_x_satisfying_conditions_l467_467763

theorem unique_x_satisfying_conditions :
  ∃ x : ℝ, x = Real.sqrt 2 - 1 ∧
  ((¬ ∃ (a b c d : ℤ), x - Real.sqrt 2 ∉ {a, b, c, d}) ∧
  (∃ (a b c : ℤ), ((x - Real.sqrt 2 = a ∨ x - 1/x = b ∨ x + 1/x = c) ∧ 
  (x^2 + 2 * Real.sqrt 2).isInteger))) :=
by sorry

end unique_x_satisfying_conditions_l467_467763


namespace find_value_l467_467224

noncomputable def alpha := sorry -- Definition of α as roots of the polynomial
noncomputable def beta := sorry  -- Definition of β as roots of the polynomial
noncomputable def gamma := sorry -- Definition of γ as roots of the polynomial

axiom root_eqn (a b c : ℂ) : a^3 - a - 1 = 0 ∧ b^3 - b - 1 = 0 ∧ c^3 - c - 1 = 0

theorem find_value :
  (∀ a b c : ℂ, 
    root_eqn a b c ∧ 
    a + b + c = 0 ∧ 
    a * b + b * c + c * a = -1 ∧ 
    a * b * c = 1 
  ) → 
  (\frac{1 + α}{1 - α} + \frac{1 + β}{1 - β} + \frac{1 + γ}{1 - γ}) = -7 := 
by
  sorry

end find_value_l467_467224


namespace determine_k_l467_467854

theorem determine_k (k : ℚ) (h_collinear : ∃ (f : ℚ → ℚ), 
  f 0 = 3 ∧ f 7 = k ∧ f 21 = 2) : k = 8 / 3 :=
by
  sorry

end determine_k_l467_467854


namespace rationalize_denominator_l467_467333

theorem rationalize_denominator (a b c : ℝ) (h : b ≠ 0) (h0 : 98 = c * c) (h1 : 7 = c) :
  (7 / (Real.sqrt 98) = (Real.sqrt 2) / 2) :=
by
  sorry

end rationalize_denominator_l467_467333


namespace typing_time_together_l467_467312

def meso_typing_rate : ℕ := 3 -- pages per minute
def tyler_typing_rate : ℕ := 5 -- pages per minute
def pages_to_type : ℕ := 40 -- pages

theorem typing_time_together :
  (meso_typing_rate + tyler_typing_rate) * 5 = pages_to_type :=
by
  sorry

end typing_time_together_l467_467312


namespace distinct_four_digit_count_l467_467155

theorem distinct_four_digit_count :
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  in count = 120 :=
by
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  show count = 120
  sorry

end distinct_four_digit_count_l467_467155


namespace most_noteworthy_figure_is_mode_l467_467531

-- Define the types of possible statistics
inductive Statistic
| Median
| Mean
| Mode
| WeightedMean

-- Define a structure for survey data (details abstracted)
structure SurveyData where
  -- fields abstracted for this problem

-- Define the concept of the most noteworthy figure
def most_noteworthy_figure (data : SurveyData) : Statistic :=
  Statistic.Mode

-- Theorem to prove the most noteworthy figure in a survey's data is the mode
theorem most_noteworthy_figure_is_mode (data : SurveyData) :
  most_noteworthy_figure data = Statistic.Mode :=
by
  sorry

end most_noteworthy_figure_is_mode_l467_467531


namespace Anya_loss_games_l467_467024

noncomputable def game_indices_Anya_lost (Anya Bella Valya Galya Dasha : ℕ) : Set ℕ :=
  {4, 8, 12, 16}

theorem Anya_loss_games
  (Anya Bella Valya Galya Dasha total_games : ℕ)
  (h1 : Anya = 4)
  (h2 : Bella = 6)
  (h3 : Valya = 7)
  (h4 : Galya = 10)
  (h5 : Dasha = 11)
  (h6 : total_games = (Anya + Bella + Valya + Galya + Dasha) / 2)
  (h7 : total_games = 19) :
  game_indices_Anya_lost Anya Bella Valya Galya Dasha = {4, 8, 12, 16} :=
  by sorry

end Anya_loss_games_l467_467024


namespace max_distance_on_ellipse_l467_467300

def ellipse_eq (x y : ℝ) : Prop := x^2 + 4 * y^2 = 36

def left_focus : (ℝ × ℝ) := (-3 * real.sqrt 3, 0)

def max_distance_PF : ℝ := 6 + 3 * real.sqrt 3

theorem max_distance_on_ellipse :
  ∀ (P : ℝ × ℝ), ellipse_eq P.1 P.2 → dist P left_focus ≤ max_distance_PF :=
sorry

end max_distance_on_ellipse_l467_467300


namespace remainders_are_distinct_l467_467409

theorem remainders_are_distinct (a : ℕ → ℕ) (H1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → a i ≠ a (i % 100 + 1))
  (H2 : ∃ r1 r2 : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → a i % a (i % 100 + 1) = r1 ∨ a i % a (i % 100 + 1) = r2) :
  ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ 100 → (a (i % 100 + 1) % a i) ≠ (a (j % 100 + 1) % a j) :=
by
  sorry

end remainders_are_distinct_l467_467409


namespace math_problem_solution_l467_467095

noncomputable def prove_math_problem (θ : Real) : Prop :=
  cos (Real.pi - θ) > 0 →
  cos (Real.pi / 2 + θ) * (1 - 2 * cos (θ / 2) ^ 2) < 0 →
  (sin θ / abs (sin θ) + abs (cos θ) / cos θ + tan θ / abs (tan θ)) = -1

-- The theorem statement
theorem math_problem_solution (θ : Real) : prove_math_problem θ :=
by
  -- proof steps would go here if we were to include the proof
  sorry

end math_problem_solution_l467_467095


namespace solution_l467_467582

-- Definitions for perpendicular and parallel relations
def perpendicular (a b : Type) : Prop := sorry -- Abstraction for perpendicularity
def parallel (a b : Type) : Prop := sorry -- Abstraction for parallelism

-- Here we define x, y, z as variables
variables {x y : Type} {z : Type}

-- Conditions for Case 2
def case2_lines_plane (x y : Type) (z : Type) := 
  (perpendicular x z) ∧ (perpendicular y z) → (parallel x y)

-- Conditions for Case 3
def case3_planes_line (x y : Type) (z : Type) := 
  (perpendicular x z) ∧ (perpendicular y z) → (parallel x y)

-- Theorem statement combining both cases
theorem solution : case2_lines_plane x y z ∧ case3_planes_line x y z := 
sorry

end solution_l467_467582


namespace sum_of_reciprocals_of_divisors_eq_two_l467_467327

open Nat

theorem sum_of_reciprocals_of_divisors_eq_two
  {N : ℕ} (h : ∑ d in (divisors N), d = 2 * N) :
  ∑ d in (divisors N), (1 : ℚ) / d = 2 := 
sorry

end sum_of_reciprocals_of_divisors_eq_two_l467_467327


namespace anya_lost_games_l467_467041

/-- 
Five girls (Anya, Bella, Valya, Galya, Dasha) played several games of table tennis. At any given time, two girls were playing,
and the other three were resting. The girl who lost the game went to rest, and her place was taken by the girl who had rested 
the most rounds. There are no ties in table tennis. Anya played 4 games, Bella played 6 games, Valya played 7 games, Galya 
played 10 games, and Dasha played 11 games.

We need to prove that the numbers of all games in which Anya lost are 4, 8, 12, and 16.
-/
theorem anya_lost_games :
  (∃ games : list ℕ,
    games.length = 19 ∧
    (∀ i, i ∈ games → i = 4 ∨ i = 8 ∨ i = 12 ∨ i = 16) ∧
    (∀ i, i ∉ games → true)) :=
by
  sorry

end anya_lost_games_l467_467041


namespace larry_keith_ratio_l467_467278

/--
Keith scored 3 points.
Larry scored some marks.
Danny scored 5 more marks than Larry.
The total amount of marks scored by the three students is 26.
Prove that the ratio of Larry's score to Keith's score is \( 3:1 \).
--/
theorem larry_keith_ratio (L : ℕ) 
  (h1 : ∃ L, L + (L + 5) + 3 = 26) :
  L / 3 = 3 :=
by
  cases h1 with L hL,
  have h2 : 2 * L + 8 = 26, by linarith,
  have h3 : 2*L = 18, by linarith,
  have hL1 : L = 9, by linarith,
  have hL2 : 9 / 3 = 3, by norm_num,
  exact hL2


end larry_keith_ratio_l467_467278


namespace coefficient_x7_in_expansion_l467_467430

theorem coefficient_x7_in_expansion : 
  let x := x
  let k := 2
  let n := 9
  let y := -2
  (∑ i in Finset.range (n + 1), Binomial.natBinomial n i * x ^ (n - i) * y ^ i).coeff x 7 = 144 :=
by
  sorry

end coefficient_x7_in_expansion_l467_467430


namespace x_plus_y_equals_391_l467_467239

noncomputable def sum_integers : ℕ := (40 - 30 + 1) * (30 + 40) / 2
def num_even_integers : ℕ := (40 - 30) / 2 + 1
def x : ℕ := sum_integers
def y : ℕ := num_even_integers

theorem x_plus_y_equals_391 : x + y = 391 := by
  sorry

end x_plus_y_equals_391_l467_467239


namespace proof_question_l467_467786

-- Definitions of the given conditions
noncomputable def right_angled_triangle (A B C : Type) [MetricSpace A]
  (tr : Triangle A B C) : Prop :=
Triangle.isRight tr ∧ ∠ B = 90

noncomputable def circle_with_diameter (A B : Type) : Circle :=
Circle.diameter A B

noncomputable def circles_intersect (C1 C2 : Circle) (P B : Point) : Prop :=
Circle.intersectsAt C1 C2 P ∧ Circle.intersectsAt C1 C2 B

noncomputable def length_of_AB : Real := 5
noncomputable def length_of_BC : Real := 12
noncomputable def BP_length (x : Real) : Prop := x = BP

-- The proof goal
theorem proof_question (A B C P : Type) [MetricSpace A]
  (tr : Triangle A B C) (C1 C2 : Circle)
  (x BP : Real) [BP_length x]
  (cond : right_angled_triangle A B C tr)
  (circle_cond1 : circle_with_diameter A B = C1)
  (circle_cond2 : circle_with_diameter B C = C2)
  (intersect_cond : circles_intersect C1 C2 P B) :
  2400 / x = 520 := 
sorry

end proof_question_l467_467786


namespace arithmetic_sequence_n_l467_467861

noncomputable def sequence : ℕ → ℕ
| 0     := 0 -- To handle the fact that we start with x_1 = 3 (in practical terms, x_0 is not used)
| 1     := 3
| (n+2) := Int.floor (Real.sqrt 2 * sequence (n + 1))

theorem arithmetic_sequence_n :
  ∀ n : ℕ, n = 1 ∨ n = 3 ↔ sequence n + sequence (n + 2) = 2 * sequence (n + 1) := sorry

end arithmetic_sequence_n_l467_467861


namespace abs_inequality_solution_l467_467354

theorem abs_inequality_solution (x : ℝ) : 
  abs (2 * x - 1) < abs x + 1 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end abs_inequality_solution_l467_467354


namespace probability_prime_sum_dice_l467_467426

open Nat

-- Define the set of prime sums obtainable with two 8-sided dice.
def prime_sums : Finset ℕ := { 2, 3, 5, 7, 11, 13 }

-- Define the total number of outcomes when two 8-sided dice are tossed.
noncomputable def total_outcomes : ℕ := 8 * 8

-- Calculate the number of favorable outcomes where the sum of the dice is a prime number.
noncomputable def favorable_outcomes : ℕ :=
  (Finset.range 64).card (λ s, s ∈ prime_sums)

-- Compute the probability as the ratio of favorable outcomes to total outcomes.
noncomputable def prime_sum_probability : ℚ :=
  favorable_outcomes / total_outcomes

-- The theorem to prove the probability is 31/64.
theorem probability_prime_sum_dice : prime_sum_probability = 31 / 64 := by
  sorry

end probability_prime_sum_dice_l467_467426


namespace temperature_conversion_l467_467865

-- Define the problem condition
def celsius_to_fahrenheit (C : ℝ) : ℝ := (9 / 5) * C + 32

-- The proof problem statement in Lean 4
theorem temperature_conversion : celsius_to_fahrenheit 10 = 50 := 
by
  sorry

end temperature_conversion_l467_467865


namespace walking_time_difference_at_slower_speed_l467_467792

theorem walking_time_difference_at_slower_speed (T : ℕ) (v_s: ℚ) (h1: T = 32) (h2: v_s = 4/5) : 
  (T * (5/4) - T) = 8 :=
by
  sorry

end walking_time_difference_at_slower_speed_l467_467792


namespace no_vision_assistance_l467_467411

theorem no_vision_assistance (total_students : ℕ) (wears_glasses : ℕ) (wears_contacts : ℕ)
    (h1 : total_students = 40)
    (h2 : wears_glasses = 0.25 * total_students) 
    (h3 : wears_contacts = 0.40 * total_students) :
    total_students - (wears_glasses + wears_contacts) = 14 := by
  sorry

end no_vision_assistance_l467_467411


namespace y_squared_value_l467_467001

theorem y_squared_value (y : ℝ) (h_y_pos : 0 < y) (h_sin_arctan : Real.sin (Real.atan y) = 1 / (2 * y)) : 
    y^2 = (1 + Real.sqrt 17) / 8 :=
sorry

end y_squared_value_l467_467001


namespace distinct_four_digit_count_l467_467154

theorem distinct_four_digit_count :
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  in count = 120 :=
by
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  show count = 120
  sorry

end distinct_four_digit_count_l467_467154


namespace relationship_between_abc_l467_467072

noncomputable def a : ℝ := (1 / 3) ^ 3
noncomputable def b (x : ℝ) : ℝ := x ^ 3
noncomputable def c (x : ℝ) : ℝ := Real.log x

theorem relationship_between_abc (x : ℝ) (h : x > 2) : a < c x ∧ c x < b x := by
  have ha : a = (1/3) ^ 3 := rfl
  have hb : b x = x ^ 3 := rfl
  have hc : c x = Real.log x := rfl
  split
  { sorry }  -- Proof that a < c x
  { sorry }  -- Proof that c x < b x

end relationship_between_abc_l467_467072


namespace total_pizzas_two_days_l467_467637

-- Declare variables representing the number of pizzas made by Craig and Heather
variables (c1 c2 h1 h2 : ℕ)

-- Given conditions as definitions
def condition_1 : Prop := h1 = 4 * c1
def condition_2 : Prop := h2 = c2 - 20
def condition_3 : Prop := c1 = 40
def condition_4 : Prop := c2 = c1 + 60

-- Prove that the total number of pizzas made in two days equals 380
theorem total_pizzas_two_days (c1 c2 h1 h2 : ℕ)
  (h_cond1 : condition_1 c1 h1) (h_cond2 : condition_2 h2 c2)
  (h_cond3 : condition_3 c1) (h_cond4 : condition_4 c1 c2) :
  (h1 + c1) + (h2 + c2) = 380 :=
by
  -- As we don't need to provide the proof here, just insert sorry
  sorry

end total_pizzas_two_days_l467_467637


namespace math_club_problem_l467_467856

theorem math_club_problem :
  ((7^2 - 5^2) - 2)^3 = 10648 := by
  sorry

end math_club_problem_l467_467856


namespace min_a_proof_l467_467717

noncomputable def min_a : ℝ := 5 / 4

theorem min_a_proof (x1 x2 x3 x4 : ℝ) :
    ∃ (k1 k2 k3 k4 : ℤ), 
    ∑ i j in {i | 1 ≤ i < j ≤ 4}.to_finset, (([x1, x2, x3, x4].nth_le j sorry - (k1, k2, k3, k4).nth_le i sorry) -
      ([x1, x2, x3, x4].nth_le i sorry - (k1, k2, k3, k4).nth_le j sorry)) ^ 2 ≤ min_a :=
by
    sorry

end min_a_proof_l467_467717


namespace propA_sufficient_not_necessary_for_propB_l467_467273

-- Definitions and conditions
variables {a b c : ℝ} -- Declare side lengths a, b, c as real numbers
variables {A B C : ℝ} -- Declare angles A, B, C as real numbers

-- Proposition A and Proposition B definitions
def propA (a b c : ℝ) : Prop := b < (a + c) / 2
def propB (A B C : ℝ) : Prop := B < (A + C) / 2

-- Main theorem statement
theorem propA_sufficient_not_necessary_for_propB
  (triangle_ABC : a + b > c ∧ a + c > b ∧ b + c > a) -- Triangle inequality conditions
  (sine_law : ∃ k : ℝ, a = k * real.sin A ∧ b = k * real.sin B ∧ c = k * real.sin C) -- Law of sines
  (A_positive : 0 < A ∧ A < real.pi)
  (B_positive : 0 < B ∧ B < real.pi)
  (C_positive : 0 < C ∧ C < real.pi)
  : (propA a b c → propB A B C) ∧ (¬ propA a b c → propB A B C) :=
by
  sorry

end propA_sufficient_not_necessary_for_propB_l467_467273


namespace wage_difference_l467_467452

theorem wage_difference (P Q H: ℝ) (h1: P = 1.5 * Q) (h2: P * H = 300) (h3: Q * (H + 10) = 300) : P - Q = 5 :=
by
  sorry

end wage_difference_l467_467452


namespace minimum_shaded_cells_l467_467434

theorem minimum_shaded_cells (n : ℕ) (h1 : n = 35) (total_cells : ℕ) (h2 : total_cells = n * n) : 
  ∃ m : ℕ, m = 408 ∧ (∀ (x y : ℕ), (x < n) → (y < n) → (L_shape (x, y) → at_least_one_shaded (x, y) m)) := 
by
  sorry

-- Definition of an L-shape and the requirement that at least one cell is shaded
def L_shape (coordinates : ℕ × ℕ) : Prop := 
  ∃ (x y : ℕ), coordinates = (x, y) ∨ coordinates = (x + 1, y) ∨ coordinates = (x, y + 1) ∨ coordinates = (x + 1, y + 1)

def at_least_one_shaded (coordinates : ℕ × ℕ) (m : ℕ) : Prop :=
  ∃ (s : ℕ → ℕ → bool), (∀ i j, i < 35 → j < 35 → s i j = tt → (∃ k l, (k, l) ∈ {coordinates, (coordinates.1 + 1, coordinates.2), (coordinates.1, coordinates.2 + 1), (coordinates.1 + 1, coordinates.2 + 1)} ∧ s k l = tt)) → (λ s, count_shaded s ≤ m)

noncomputable def count_shaded (s : ℕ → ℕ → bool) : ℕ :=
  ∑ i j, if s i j = tt then 1 else 0

end minimum_shaded_cells_l467_467434


namespace minimize_average_cost_l467_467373

noncomputable def totalMaintenanceCost (x : ℕ) : ℕ := (x * (x + 1)) / 2

noncomputable def averageCost (x : ℕ) : ℝ := (500000 + 45000 * x + totalMaintenanceCost x) / x

theorem minimize_average_cost : (∀ x : ℕ, x > 0 → averageCost x ≥ averageCost 10) :=
begin
  sorry
end

end minimize_average_cost_l467_467373


namespace series_sum_half_l467_467919

theorem series_sum_half :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1/2 := 
sorry

end series_sum_half_l467_467919


namespace unique_intersection_value_l467_467569

theorem unique_intersection_value :
  (∀ (x y : ℝ), y = x^2 → y = 4 * x + k) → (k = -4) := 
by
  sorry

end unique_intersection_value_l467_467569


namespace average_increase_l467_467419

theorem average_increase (x : ℝ) (y : ℝ) (h : y = 0.245 * x + 0.321) : 
  ∀ x_increase : ℝ, x_increase = 1 → (0.245 * (x + x_increase) + 0.321) - (0.245 * x + 0.321) = 0.245 :=
by
  intro x_increase
  intro hx
  rw [hx]
  simp
  sorry

end average_increase_l467_467419


namespace unique_plants_in_ABC_l467_467068

-- Define the number of plants in each bed and the shared plants
def A : ℕ := 600
def B : ℕ := 500
def C : ℕ := 400
def AB : ℕ := 60
def AC : ℕ := 80
def BC : ℕ := 40
def ABC : ℕ := 20

-- Prove that the total number of unique plants in Beds A, B, and C is 1340
theorem unique_plants_in_ABC : (A + B + C - AB - AC - BC + ABC) = 1340 :=
by
  calc
    (A + B + C - AB - AC - BC + ABC)
      = 600 + 500 + 400 - 60 - 80 - 40 + 20 : by simp [A, B, C, AB, AC, BC, ABC]
  ... = 1500 - 180 + 20 : by norm_num
  ... = 1340 : by norm_num

end unique_plants_in_ABC_l467_467068


namespace sum_of_cubes_l467_467091

variable (a b c : ℝ)

theorem sum_of_cubes (h1 : a^2 + 3 * b = 2) (h2 : b^2 + 5 * c = 3) (h3 : c^2 + 7 * a = 6) :
  a^3 + b^3 + c^3 = -0.875 :=
by
  sorry

end sum_of_cubes_l467_467091


namespace total_pigs_indeterminate_l467_467745

noncomputable def average_weight := 15
def underweight_threshold := 16
def max_underweight_pigs := 4

theorem total_pigs_indeterminate :
  ∃ (P U : ℕ), U ≤ max_underweight_pigs ∧ (average_weight = 15) → P = P :=
sorry

end total_pigs_indeterminate_l467_467745


namespace rationalize_denominator_l467_467338

theorem rationalize_denominator (sqrt_98 : Real) (h : sqrt_98 = 7 * Real.sqrt 2) : 
  7 / sqrt_98 = Real.sqrt 2 / 2 := 
by 
  sorry

end rationalize_denominator_l467_467338


namespace distinct_four_digit_count_l467_467152

theorem distinct_four_digit_count :
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  in count = 120 :=
by
  let digits := {1, 2, 3, 4, 5}
  let count := (5 * 4 * 3 * 2)
  show count = 120
  sorry

end distinct_four_digit_count_l467_467152


namespace find_beta_l467_467597

theorem find_beta 
  (α β : ℝ) (h₀ : 0 < β ∧ β < α ∧ α < π / 2) 
  (h₁ : ∃ P : ℝ × ℝ, P = (1, 4 * (sqrt 3)) ∧ 
        (sin α = (4 * (sqrt 3)) / (real.sqrt (1 + (4 * (sqrt 3))^2))) ∧ 
        (cos α = 1 / (real.sqrt (1 + (4 * (sqrt 3))^2))))
  (h₂ : sin α * sin(π / 2 - β) + cos α * cos(π / 2 + β) = (3 * (sqrt 3)) / 14) 
  : β = π / 3 := sorry

end find_beta_l467_467597


namespace parabola_constant_term_l467_467370

theorem parabola_constant_term (a b c : ℝ) (h_vertex : (∀ y, x = a * (y - 1)^2 - 3) 
  (h_point : ∀ x y, point (-6, 3) ):
  c = -15/4 := 
begin
  sorry
end

end parabola_constant_term_l467_467370


namespace max_siskins_on_poles_l467_467880

-- Definitions based on problem conditions
def pole : Type := ℕ
def siskins (poles : pole) : Prop := poles ≤ 25
def adjacent (p₁ p₂ : pole) : Prop := (p₁ = p₂ + 1) ∨ (p₁ = p₂ - 1)

-- Given conditions
def conditions (p : pole → bool) : Prop :=
  ∀ p₁ p₂ : pole, p p₁ = true → p p₂ = true → adjacent p₁ p₂ → false

-- Main problem statement
theorem max_siskins_on_poles : ∃ p : pole → bool, (∀ i : pole, p i = true → siskins i) ∧ (conditions p) ∧ (∑ i in finset.range 25, if p i then 1 else 0 = 24) :=
begin
  sorry
end

end max_siskins_on_poles_l467_467880


namespace find_f_one_div_2007_l467_467852
-- We use all of Mathlib to ensure all necessary definitions and theorems are included.

noncomputable theory

-- Define the conditions of the function f.
def is_special_function (f : ℝ → ℝ) : Prop :=
  (f 0 = 0) ∧ 
  (∀ x : ℝ, f x + f (1 - x) = 1) ∧ 
  (∀ x : ℝ, f (x / 5) = (1 / 2) * f x) ∧ 
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → x₂ ≤ 1 → f x₁ ≤ f x₂)

-- State the theorem to be proved.
theorem find_f_one_div_2007 (f : ℝ → ℝ) (h : is_special_function f) : f (1 / 2007) = 1 / 32 :=
sorry

end find_f_one_div_2007_l467_467852


namespace total_apples_l467_467644

theorem total_apples (x : ℕ) : 
    (x - x / 5 - x / 12 - x / 8 - x / 20 - x / 4 - x / 7 - x / 30 - 4 * (x / 30) - 300 ≤ 50) -> 
    x = 3360 :=
by
    sorry

end total_apples_l467_467644


namespace initial_bottle_caps_l467_467277

theorem initial_bottle_caps (B : ℕ) 
    (truck_cost : ℕ) (car_cost : ℕ) (num_trucks : ℕ) (num_vehicles : ℕ) :
    truck_cost = 6 →
    car_cost = 5 →
    num_trucks = 10 →
    num_vehicles = 16 →
    B - (num_trucks * truck_cost) = (5 * 6 / 0.75 + 60) →
    B = 100 :=
by
    intros h_truck h_car h_trucks h_vehicles h_equation
    simp [h_truck, h_car, h_trucks] at h_equation
    have H : B = 100 := by sorry
    exact H

end initial_bottle_caps_l467_467277


namespace cot_of_sum_arccot_l467_467545

-- Definitions
def cot (x : ℝ) := 1 / tan x

-- Inverses for the cotangent
def arccot (x : ℝ) := atan (1 / x)

-- Main statement
theorem cot_of_sum_arccot
  : cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 1021 / 420 :=
by
  sorry

end cot_of_sum_arccot_l467_467545


namespace number_of_vertices_with_odd_degree_is_even_l467_467729

theorem number_of_vertices_with_odd_degree_is_even (G : Type) [Fintype G] [SimpleGraph G] :
  (∃ o : Finset G, ∀ v ∈ o, odd (G.degree v)) → even (Fintype.card {v : G // odd (G.degree v)}) :=
sorry

end number_of_vertices_with_odd_degree_is_even_l467_467729


namespace evaluate_expression_l467_467008

noncomputable def w := Complex.exp (2 * Real.pi * Complex.I / 11)

theorem evaluate_expression : (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) * (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) = 88573 := 
by 
  sorry

end evaluate_expression_l467_467008


namespace move_line_down_l467_467868

theorem move_line_down (k b : ℝ) (k_eq : k = 3) (b_eq : b = 0) (shift : ℝ) (shift_eq : shift = -2) :
  (λ x, k * x + (b + shift)) = (λ x, 3 * x - 2) :=
by
  sorry

end move_line_down_l467_467868


namespace Anya_loss_games_l467_467028

noncomputable def game_indices_Anya_lost (Anya Bella Valya Galya Dasha : ℕ) : Set ℕ :=
  {4, 8, 12, 16}

theorem Anya_loss_games
  (Anya Bella Valya Galya Dasha total_games : ℕ)
  (h1 : Anya = 4)
  (h2 : Bella = 6)
  (h3 : Valya = 7)
  (h4 : Galya = 10)
  (h5 : Dasha = 11)
  (h6 : total_games = (Anya + Bella + Valya + Galya + Dasha) / 2)
  (h7 : total_games = 19) :
  game_indices_Anya_lost Anya Bella Valya Galya Dasha = {4, 8, 12, 16} :=
  by sorry

end Anya_loss_games_l467_467028


namespace max_siskins_on_poles_l467_467878

theorem max_siskins_on_poles : 
    (∀ (poles : List Nat), length poles = 25 → 
        ∀ (siskins_on_poles : List (Option Nat)), 
        length siskins_on_poles = 25 → 
        (∀ (i : Nat), (i < 25) → (∀ (j : Nat), (abs (j - i) = 1) → 
          (siskins_on_poles.nth i = some _) → (siskins_on_poles.nth j ≠ some _))) → 
        (∃ k, ∀ n, (n < 25) → 
            n + k ≤ 24 → 
            ∃ m, (siskins_on_poles.nth m = some m) ∧ 
              (m < 25) ∧ 
              (∀ l, (m ≠ l) → (siskins_on_poles.nth l ≠ some _)))) → true) := 
by sorry

end max_siskins_on_poles_l467_467878


namespace max_value_f_when_a_neg4_num_roots_f_eq_0_no_satisfactory_a_l467_467012

-- 1. Maximum value of f(x) on the interval [1, e] when a = -4
theorem max_value_f_when_a_neg4 : 
  ∀ (x : ℝ), x ∈ Icc 1 real.e → (-4 * real.log x + x^2) ≤ (real.e^2 - 4) :=
sorry

-- 2. Number of roots of the equation f(x) = 0 in the interval [1, e]
theorem num_roots_f_eq_0 :
  ∀ (a : ℝ), 
    (if -2 ≤ a ∧ a < 0 ∨ -2 * real.e < a ∧ a < -2 then
      ∀ x, x ∈ Icc 1 real.e → ¬ (a * real.log x + x^2 = 0)
    else if a ≤ -2 * real.e ^ 2 ∨ a = -2 * real.e ∨ (-2 * real.e ^ 2 < a ∧ a < - e ^ 2) then
      ∃! x, x ∈ Icc 1 real.e ∧ (a * real.log x + x^2 = 0)
    else if - e ^ 2 ≤ a ∧ a < - 2 * real.e then
      ∃ x1 x2, x1 ∈ Icc 1 real.e ∧ x2 ∈ Icc 1 real.e ∧ x1 ≠ x2 ∧ (a * real.log x1 + x1^2 = 0) ∧ (a * real.log x2 + x2^2 = 0)
    else false) :=
sorry

-- 3. Inequality and range of 'a' when a > 0
theorem no_satisfactory_a :
  ∀ (a : ℝ), a > 0 → ¬ (∀ (x₁ x₂ : ℝ), 
    (x₁ ∈ Icc 1 real.e) → (x₂ ∈ Icc 1 real.e) → 
    ((abs ((a * real.log x₁ + x₁^2) - (a * real.log x₂ + x₂^2)) ≤ (abs ((1/x₁) - (1/x₂))))) :=
sorry

end max_value_f_when_a_neg4_num_roots_f_eq_0_no_satisfactory_a_l467_467012
