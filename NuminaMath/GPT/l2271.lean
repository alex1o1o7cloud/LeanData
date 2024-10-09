import Mathlib

namespace candle_height_problem_l2271_227190

-- Define the conditions given in the problem
def same_initial_height (height : ℝ := 1) := height = 1

def burn_rate_first_candle := 1 / 5

def burn_rate_second_candle := 1 / 4

def height_first_candle (t : ℝ) := 1 - (burn_rate_first_candle * t)

def height_second_candle (t : ℝ) := 1 - (burn_rate_second_candle * t)

-- Define the proof problem
theorem candle_height_problem : ∃ t : ℝ, height_first_candle t = 3 * height_second_candle t ∧ t = 40 / 11 :=
by
  sorry

end candle_height_problem_l2271_227190


namespace find_acute_angle_l2271_227124

noncomputable def vector_a (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, 2)
noncomputable def vector_b (α : ℝ) : ℝ × ℝ := (3, 4 * Real.sin α)
def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem find_acute_angle (α : ℝ) (h : are_parallel (vector_a α) (vector_b α)) (h_acute : 0 < α ∧ α < π / 2) : 
  α = π / 4 :=
by
  sorry

end find_acute_angle_l2271_227124


namespace part_a_part_b_part_c_l2271_227130

-- Part (a)
theorem part_a : 
  ∃ n : ℕ, n = 2023066 ∧ (∃ x y z : ℕ, x + y + z = 2013 ∧ x > 0 ∧ y > 0 ∧ z > 0) :=
sorry

-- Part (b)
theorem part_b : 
  ∃ n : ℕ, n = 1006 ∧ (∃ x y z : ℕ, x + y + z = 2013 ∧ x = y ∧ x > 0 ∧ y > 0 ∧ z > 0) :=
sorry

-- Part (c)
theorem part_c : 
  ∃ (x y z : ℕ), (x + y + z = 2013 ∧ (x * y * z = 671 * 671 * 671)) :=
sorry

end part_a_part_b_part_c_l2271_227130


namespace f_g_x_eq_l2271_227112

noncomputable def f (x : ℝ) : ℝ := (x * (x + 1)) / 3
noncomputable def g (x : ℝ) : ℝ := x + 3

theorem f_g_x_eq (x : ℝ) : f (g x) = (x^2 + 7*x + 12) / 3 := by
  sorry

end f_g_x_eq_l2271_227112


namespace count_dna_sequences_Rthea_l2271_227138

-- Definition of bases
inductive Base | H | M | N | T

-- Function to check whether two bases can be adjacent on the same strand
def can_be_adjacent (x y : Base) : Prop :=
  match x, y with
  | Base.H, Base.M => False
  | Base.M, Base.H => False
  | Base.N, Base.T => False
  | Base.T, Base.N => False
  | _, _ => True

-- Function to count the number of valid sequences
noncomputable def count_valid_sequences : Nat := 12 * 7^4

-- Theorem stating the expected count of valid sequences
theorem count_dna_sequences_Rthea : count_valid_sequences = 28812 := by
  sorry

end count_dna_sequences_Rthea_l2271_227138


namespace unique_positive_real_solution_l2271_227164

theorem unique_positive_real_solution (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h1 : x * y = z) (h2 : y * z = x) (h3 : z * x = y) : x = 1 ∧ y = 1 ∧ z = 1 :=
sorry

end unique_positive_real_solution_l2271_227164


namespace domain_of_f_l2271_227126

noncomputable def f (x k : ℝ) := (3 * x ^ 2 + 4 * x - 7) / (-7 * x ^ 2 + 4 * x + k)

theorem domain_of_f {x k : ℝ} (h : k < -4/7): ∀ x, -7 * x ^ 2 + 4 * x + k ≠ 0 :=
by 
  intro x
  sorry

end domain_of_f_l2271_227126


namespace y_at_40_l2271_227146

def y_at_x (x : ℤ) : ℤ :=
  3 * x + 4

theorem y_at_40 : y_at_x 40 = 124 :=
by {
  sorry
}

end y_at_40_l2271_227146


namespace acres_used_for_corn_l2271_227148

theorem acres_used_for_corn (total_land : ℕ) (ratio_beans : ℕ) (ratio_wheat : ℕ) (ratio_corn : ℕ)
  (total_ratio_parts : ℕ) (one_part_size : ℕ) :
  total_land = 1034 →
  ratio_beans = 5 →
  ratio_wheat = 2 →
  ratio_corn = 4 →
  total_ratio_parts = ratio_beans + ratio_wheat + ratio_corn →
  one_part_size = total_land / total_ratio_parts →
  ratio_corn * one_part_size = 376 :=
by
  intros
  sorry

end acres_used_for_corn_l2271_227148


namespace marcella_matching_pairs_l2271_227152

theorem marcella_matching_pairs (P : ℕ) (L : ℕ) (H : P = 20) (H1 : L = 9) : (P - L) / 2 = 11 :=
by
  -- definition of P and L are given by 20 and 9 respectively
  -- proof is omitted for the statement focus
  sorry

end marcella_matching_pairs_l2271_227152


namespace triangle_area_l2271_227132

theorem triangle_area {x y : ℝ} :

  (∀ a:ℝ, y = a ↔ a = x) ∧
  (∀ b:ℝ, y = -b ↔ b = x) ∧
  ( y = 10 )
  → 1 / 2 * abs (10 - (-10)) * 10 = 100 :=
by
  sorry

end triangle_area_l2271_227132


namespace sequence_general_term_l2271_227109

theorem sequence_general_term {a : ℕ → ℕ} (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 2) :
  ∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1 :=
by
  -- skip the proof with sorry
  sorry

end sequence_general_term_l2271_227109


namespace students_meet_time_l2271_227121

theorem students_meet_time :
  ∀ (distance rate1 rate2 : ℝ),
    distance = 350 ∧ rate1 = 1.6 ∧ rate2 = 1.9 →
    distance / (rate1 + rate2) = 100 := by
  sorry

end students_meet_time_l2271_227121


namespace base7_to_base10_l2271_227144

theorem base7_to_base10 : 
  let digit0 := 2
  let digit1 := 3
  let digit2 := 4
  let digit3 := 5
  let base := 7
  digit0 * base^0 + digit1 * base^1 + digit2 * base^2 + digit3 * base^3 = 1934 :=
by
  let digit0 := 2
  let digit1 := 3
  let digit2 := 4
  let digit3 := 5
  let base := 7
  sorry

end base7_to_base10_l2271_227144


namespace symmetrical_polynomial_l2271_227186

noncomputable def Q (x : ℝ) (f g h i j k : ℝ) : ℝ :=
  x^6 + f * x^5 + g * x^4 + h * x^3 + i * x^2 + j * x + k

theorem symmetrical_polynomial (f g h i j k : ℝ) :
  (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ Q 0 f g h i j k = 0 ∧
    Q x f g h i j k = x * (x - a) * (x + a) * (x - b) * (x + b) * (x - c) ∧
    Q x f g h i j k = Q (-x) f g h i j k) →
  f = 0 :=
by sorry

end symmetrical_polynomial_l2271_227186


namespace solve_for_x_l2271_227135

def star (a b : ℕ) := a * b + a + b

theorem solve_for_x : ∃ x : ℕ, star 3 x = 27 ∧ x = 6 :=
by {
  sorry
}

end solve_for_x_l2271_227135


namespace solve_quadratic_l2271_227136

theorem solve_quadratic (x : ℝ) (h : x^2 - 2 * x - 3 = 0) : x = 3 ∨ x = -1 := 
sorry

end solve_quadratic_l2271_227136


namespace max_f_max_ab_plus_bc_l2271_227169

def f (x : ℝ) := |x - 3| - 2 * |x + 1|

theorem max_f : ∃ (m : ℝ), m = 4 ∧ (∀ x : ℝ, f x ≤ m) := 
  sorry

theorem max_ab_plus_bc (a b c : ℝ) : a > 0 ∧ b > 0 → a^2 + 2 * b^2 + c^2 = 4 → (ab + bc) ≤ 2 :=
  sorry

end max_f_max_ab_plus_bc_l2271_227169


namespace grid_coloring_count_l2271_227142

/-- Let n be a positive integer with n ≥ 2. Each of the 2n vertices in a 2 × n grid need to be 
colored red (R), yellow (Y), or blue (B). The three vertices at the endpoints are already colored 
as shown in the problem description. For the remaining 2n-3 vertices, each vertex must be colored 
exactly one color, and adjacent vertices must be colored differently. We aim to show that the 
number of distinct ways to color the vertices is 3^(n-1). -/
theorem grid_coloring_count (n : ℕ) (hn : n ≥ 2) : 
  ∃ a_n b_n c_n : ℕ, 
    (a_n + b_n + c_n = 3^(n-1)) ∧ 
    (a_n = b_n) ∧ 
    (a_n = 2 * b_n + c_n) := 
by 
  sorry

end grid_coloring_count_l2271_227142


namespace smallest_value_of_3a_plus_1_l2271_227191

theorem smallest_value_of_3a_plus_1 (a : ℚ) (h : 8 * a^2 + 6 * a + 2 = 2) : 3 * a + 1 = -5/4 :=
by
  sorry

end smallest_value_of_3a_plus_1_l2271_227191


namespace total_students_l2271_227153

-- Define the conditions
def ratio_boys_to_girls (boys girls : ℕ) : Prop := boys = 3 * (girls / 2)
def boys_girls_difference (boys girls : ℕ) : Prop := boys = girls + 20

-- Define the property to be proved
theorem total_students (boys girls : ℕ) 
  (h1 : ratio_boys_to_girls boys girls)
  (h2 : boys_girls_difference boys girls) :
  boys + girls = 100 :=
sorry

end total_students_l2271_227153


namespace cows_total_l2271_227161

theorem cows_total {n : ℕ} :
  (n / 3) + (n / 6) + (n / 8) + (n / 24) + 15 = n ↔ n = 45 :=
by {
  sorry
}

end cows_total_l2271_227161


namespace minimum_a_plus_3b_l2271_227165

-- Define the conditions
variables (a b : ℝ)
axiom h_pos_a : a > 0
axiom h_pos_b : b > 0
axiom h_eq : a + 3 * b = 1 / a + 3 / b

-- State the theorem
theorem minimum_a_plus_3b (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : a + 3 * b = 1 / a + 3 / b) : 
  a + 3 * b ≥ 4 :=
sorry

end minimum_a_plus_3b_l2271_227165


namespace temperature_conversion_l2271_227151

theorem temperature_conversion (C F F_new C_new : ℚ) 
  (h_formula : C = (5/9) * (F - 32))
  (h_C : C = 30)
  (h_F_new : F_new = F + 15)
  (h_F : F = 86)
: C_new = (5/9) * (F_new - 32) ↔ C_new = 38.33 := 
by 
  sorry

end temperature_conversion_l2271_227151


namespace find_a_l2271_227181

theorem find_a (a : ℝ) (x : ℝ) (h₀ : a > 0) (h₁ : x > 0)
  (h₂ : a * Real.sqrt x = Real.log (Real.sqrt x))
  (h₃ : (a / (2 * Real.sqrt x)) = (1 / (2 * x))) : a = Real.exp (-1) :=
by
  sorry

end find_a_l2271_227181


namespace f_g_eq_g_f_iff_n_zero_l2271_227143

def f (x n : ℝ) : ℝ := x + n
def g (x q : ℝ) : ℝ := x^2 + q

theorem f_g_eq_g_f_iff_n_zero (x n q : ℝ) : (f (g x q) n = g (f x n) q) ↔ n = 0 := by 
  sorry

end f_g_eq_g_f_iff_n_zero_l2271_227143


namespace number_of_yellow_balls_l2271_227184

theorem number_of_yellow_balls (x : ℕ) :
  (4 : ℕ) / (4 + x) = 2 / 3 → x = 2 :=
by
  sorry

end number_of_yellow_balls_l2271_227184


namespace flower_beds_fraction_l2271_227177

open Real

noncomputable def parkArea (a b h : ℝ) := (a + b) / 2 * h
noncomputable def triangleArea (a : ℝ) := (1 / 2) * a ^ 2

theorem flower_beds_fraction 
  (a b h : ℝ) 
  (h_a: a = 15) 
  (h_b: b = 30) 
  (h_h: h = (b - a) / 2) :
  (2 * triangleArea h) / parkArea a b h = 1 / 4 := by 
  sorry

end flower_beds_fraction_l2271_227177


namespace banana_difference_l2271_227158

theorem banana_difference (d : ℕ) :
  (8 + (8 + d) + (8 + 2 * d) + (8 + 3 * d) + (8 + 4 * d) = 100) →
  d = 6 :=
by
  sorry

end banana_difference_l2271_227158


namespace find_monotonic_function_l2271_227185

-- Define Jensen's functional equation property
def jensens_eq (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (t : ℝ), 0 ≤ t ∧ t ≤ 1 → f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y

-- Define monotonicity property
def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- The main theorem stating the equivalence
theorem find_monotonic_function (f : ℝ → ℝ) (h₁ : jensens_eq f) (h₂ : monotonic f) : 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := 
sorry

end find_monotonic_function_l2271_227185


namespace greatest_x_inequality_l2271_227168

theorem greatest_x_inequality :
  ∃ x, -x^2 + 11 * x - 28 = 0 ∧ (∀ y, -y^2 + 11 * y - 28 ≥ 0 → y ≤ x) ∧ x = 7 :=
sorry

end greatest_x_inequality_l2271_227168


namespace bead_game_solution_l2271_227114

-- Define the main theorem, stating the solution is valid for r = (b + 1) / b
theorem bead_game_solution {r : ℚ} (h : r > 1) (b : ℕ) (hb : 1 ≤ b ∧ b ≤ 1010) :
  r = (b + 1) / b ∧ (∀ k : ℕ, k ≤ 2021 → True) := by
  sorry

end bead_game_solution_l2271_227114


namespace smallest_n_integer_l2271_227131

theorem smallest_n_integer (m n : ℕ) (s : ℝ) (h_m : m = (n + s)^4) (h_n_pos : 0 < n) (h_s_range : 0 < s ∧ s < 1 / 2000) : n = 8 := 
by
  sorry

end smallest_n_integer_l2271_227131


namespace cube_root_of_unity_identity_l2271_227166

theorem cube_root_of_unity_identity (ω : ℂ) (hω3: ω^3 = 1) (hω_ne_1 : ω ≠ 1) (hunit : ω^2 + ω + 1 = 0) :
  (1 - ω) * (1 - ω^2) * (1 - ω^4) * (1 - ω^8) = 9 :=
by
  sorry

end cube_root_of_unity_identity_l2271_227166


namespace max_product_of_sum_2000_l2271_227157

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l2271_227157


namespace sum_of_squares_gt_five_l2271_227196

theorem sum_of_squares_gt_five (a b c : ℝ) (h : a + b + c = 4) : a^2 + b^2 + c^2 > 5 :=
sorry

end sum_of_squares_gt_five_l2271_227196


namespace cups_of_flour_put_in_l2271_227128

-- Conditions
def recipeSugar : ℕ := 3
def recipeFlour : ℕ := 10
def neededMoreFlourThanSugar : ℕ := 5

-- Question: How many cups of flour did she put in?
-- Answer: 5 cups of flour
theorem cups_of_flour_put_in : (recipeSugar + neededMoreFlourThanSugar = recipeFlour) → recipeFlour - neededMoreFlourThanSugar = 5 := 
by
  intros h
  sorry

end cups_of_flour_put_in_l2271_227128


namespace cos_pi_minus_2alpha_l2271_227174

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (Real.pi - 2 * α) = -1 / 9 := 
by
  sorry

end cos_pi_minus_2alpha_l2271_227174


namespace find_p_and_q_l2271_227116

theorem find_p_and_q :
  (∀ p q: ℝ, (∃ x : ℝ, x^2 + p * x + q = 0 ∧ q * x^2 + p * x + 1 = 0) ∧ (-2) ^ 2 + p * (-2) + q = 0 ∧ p ≠ 0 ∧ q ≠ 0 → 
    (p, q) = (1, -2) ∨ (p, q) = (3, 2) ∨ (p, q) = (5/2, 1)) :=
sorry

end find_p_and_q_l2271_227116


namespace part1_l2271_227194

open Set

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

theorem part1 (U_eq : U = univ) 
  (A_eq : A = {x | (x - 5) / (x - 2) ≤ 0}) 
  (B_eq : B = {x | 1 < x ∧ x < 3}) :
  compl A ∩ compl B = {x | x ≤ 1 ∨ x > 5} := 
  sorry

end part1_l2271_227194


namespace f_bound_l2271_227171

theorem f_bound (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x + 1) - f x = 2 * x + 1) 
  (h2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) : ∀ x : ℝ, |f x| ≤ 2 + x^2 :=
by
  sorry

end f_bound_l2271_227171


namespace hollow_cylinder_surface_area_l2271_227147

theorem hollow_cylinder_surface_area (h : ℝ) (r_outer r_inner : ℝ) (h_eq : h = 12) (r_outer_eq : r_outer = 5) (r_inner_eq : r_inner = 2) :
  (2 * π * ((r_outer ^ 2 - r_inner ^ 2)) + 2 * π * r_outer * h + 2 * π * r_inner * h) = 210 * π :=
by
  rw [h_eq, r_outer_eq, r_inner_eq]
  sorry

end hollow_cylinder_surface_area_l2271_227147


namespace solve_fractional_equation_l2271_227183

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -5) : 
    (2 * x / (x - 1)) - 1 = 4 / (1 - x) → x = -5 := 
by
  sorry

end solve_fractional_equation_l2271_227183


namespace vector_t_solution_l2271_227125

theorem vector_t_solution (t : ℝ) :
  ∃ t, (∃ (AB AC BC : ℝ × ℝ), 
         AB = (t, 1) ∧ AC = (2, 2) ∧ BC = (2 - t, 1) ∧ 
         (AC.1 - AB.1) * AC.1 + (AC.2 - AB.2) * AC.2 = 0 ) → 
         t = 3 :=
by {
  sorry -- proof content omitted as per instructions
}

end vector_t_solution_l2271_227125


namespace sum_of_numbers_l2271_227119

theorem sum_of_numbers (x y : ℕ) (h1 : x * y = 9375) (h2 : y / x = 15) : x + y = 400 :=
by
  sorry

end sum_of_numbers_l2271_227119


namespace δ_can_be_arbitrarily_small_l2271_227176

-- Define δ(r) as the distance from the circle to the nearest point with integer coordinates.
def δ (r : ℝ) : ℝ := sorry -- exact definition would depend on the implementation details

-- The main theorem to be proven.
theorem δ_can_be_arbitrarily_small (ε : ℝ) (hε : ε > 0) : ∃ r : ℝ, r > 0 ∧ δ r < ε :=
sorry

end δ_can_be_arbitrarily_small_l2271_227176


namespace prime_number_property_l2271_227179

open Nat

-- Definition that p is prime
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Conjecture to prove: if p is a prime number and p^4 - 3p^2 + 9 is also a prime number, then p = 2.
theorem prime_number_property (p : ℕ) (h1 : is_prime p) (h2 : is_prime (p^4 - 3*p^2 + 9)) : p = 2 :=
sorry

end prime_number_property_l2271_227179


namespace Oliver_Battle_Gremlins_Card_Count_l2271_227163

theorem Oliver_Battle_Gremlins_Card_Count 
  (MonsterClubCards AlienBaseballCards BattleGremlinsCards : ℕ)
  (h1 : MonsterClubCards = 2 * AlienBaseballCards)
  (h2 : BattleGremlinsCards = 3 * AlienBaseballCards)
  (h3 : MonsterClubCards = 32) : 
  BattleGremlinsCards = 48 := by
  sorry

end Oliver_Battle_Gremlins_Card_Count_l2271_227163


namespace mary_score_unique_l2271_227123

theorem mary_score_unique (c w : ℕ) (s : ℕ) (h_score_formula : s = 35 + 4 * c - w)
  (h_limit : c + w ≤ 35) (h_greater_90 : s > 90) :
  (∀ s' > 90, s' ≠ s → ¬ ∃ c' w', s' = 35 + 4 * c' - w' ∧ c' + w' ≤ 35) → s = 91 :=
by
  sorry

end mary_score_unique_l2271_227123


namespace simplify_expression_l2271_227195

theorem simplify_expression (a b : ℝ) (h : a + b < 0) : 
  |a + b - 1| - |3 - (a + b)| = -2 :=
by 
  sorry

end simplify_expression_l2271_227195


namespace no_nat_solution_l2271_227104

theorem no_nat_solution (x y z : ℕ) : ¬ (x^3 + 2 * y^3 = 4 * z^3) :=
sorry

end no_nat_solution_l2271_227104


namespace gambler_win_percentage_l2271_227159

theorem gambler_win_percentage :
  ∀ (T W play_extra : ℕ) (P_win_extra P_week P_current P_required : ℚ),
    T = 40 →
    P_win_extra = 0.80 →
    play_extra = 40 →
    P_week = 0.60 →
    P_required = 48 →
    (W + P_win_extra * play_extra = P_required) →
    (P_current = (W : ℚ) / T * 100) →
    P_current = 40 :=
by
  intros T W play_extra P_win_extra P_week P_current P_required h1 h2 h3 h4 h5 h6 h7
  sorry

end gambler_win_percentage_l2271_227159


namespace find_number_l2271_227180

theorem find_number (x : ℝ) (h : 0.20 * x = 0.20 * 650 + 190) : x = 1600 :=
sorry

end find_number_l2271_227180


namespace polygon_sides_l2271_227154

-- Definitions of the conditions
def is_regular_polygon (n : ℕ) (int_angle ext_angle : ℝ) : Prop :=
  int_angle = 5 * ext_angle ∧ (int_angle + ext_angle = 180)

-- Main theorem statement
theorem polygon_sides (n : ℕ) (int_angle ext_angle : ℝ) :
  is_regular_polygon n int_angle ext_angle →
  (ext_angle = 360 / n) →
  n = 12 :=
sorry

end polygon_sides_l2271_227154


namespace total_revenue_l2271_227149

theorem total_revenue (C A : ℕ) (P_C P_A total_tickets adult_tickets revenue : ℕ)
  (hCC : C = 6) -- Children's ticket price
  (hAC : A = 9) -- Adult's ticket price
  (hTT : total_tickets = 225) -- Total tickets sold
  (hAT : adult_tickets = 175) -- Adult tickets sold
  (hTR : revenue = 1875) -- Total revenue
  : revenue = adult_tickets * A + (total_tickets - adult_tickets) * C := sorry

end total_revenue_l2271_227149


namespace find_a_l2271_227199

theorem find_a
  (x y a : ℝ)
  (h1 : x + y = 1)
  (h2 : 2 * x + y = 0)
  (h3 : a * x - 3 * y = 0) :
  a = -6 :=
sorry

end find_a_l2271_227199


namespace largest_multiple_of_7_negation_gt_neg150_l2271_227156

theorem largest_multiple_of_7_negation_gt_neg150 : 
  ∃ (k : ℤ), (k % 7 = 0 ∧ -k > -150 ∧ ∀ (m : ℤ), (m % 7 = 0 ∧ -m > -150 → m ≤ k)) :=
sorry

end largest_multiple_of_7_negation_gt_neg150_l2271_227156


namespace S_11_is_22_l2271_227111

-- Definitions and conditions
variable (a_1 d : ℤ) -- first term and common difference of the arithmetic sequence
noncomputable def S (n : ℤ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

-- The given condition
variable (h : S a_1 d 8 - S a_1 d 3 = 10)

-- The proof goal
theorem S_11_is_22 : S a_1 d 11 = 22 :=
by
  sorry

end S_11_is_22_l2271_227111


namespace value_of_x_in_equation_l2271_227170

theorem value_of_x_in_equation : 
  (∀ x : ℕ, 8 ^ 17 + 8 ^ 17 + 8 ^ 17 + 8 ^ 17 = 2 ^ x → x = 53) := 
by 
  sorry

end value_of_x_in_equation_l2271_227170


namespace exam_maximum_marks_l2271_227117

theorem exam_maximum_marks :
  (∃ M S E : ℕ, 
    (90 + 20 = 40 * M / 100) ∧ 
    (110 + 35 = 35 * S / 100) ∧ 
    (80 + 10 = 30 * E / 100) ∧ 
    M = 275 ∧ 
    S = 414 ∧ 
    E = 300) :=
by
  sorry

end exam_maximum_marks_l2271_227117


namespace negation_all_nonzero_l2271_227198

    theorem negation_all_nonzero (a b c : ℝ) : ¬ (¬ (a = 0 ∨ b = 0 ∨ c = 0)) → (a = 0 ∧ b = 0 ∧ c = 0) :=
    by
      sorry
    
end negation_all_nonzero_l2271_227198


namespace eqidistant_point_on_x_axis_l2271_227155

theorem eqidistant_point_on_x_axis (x : ℝ) : 
    (dist (x, 0) (-3, 0) = dist (x, 0) (2, 5)) → 
    x = 2 := by
  sorry

end eqidistant_point_on_x_axis_l2271_227155


namespace Dave_has_more_money_than_Derek_l2271_227100

def Derek_initial := 40
def Derek_expense1 := 14
def Derek_expense2 := 11
def Derek_expense3 := 5
def Derek_remaining := Derek_initial - Derek_expense1 - Derek_expense2 - Derek_expense3

def Dave_initial := 50
def Dave_expense := 7
def Dave_remaining := Dave_initial - Dave_expense

def money_difference := Dave_remaining - Derek_remaining

theorem Dave_has_more_money_than_Derek : money_difference = 33 := by sorry

end Dave_has_more_money_than_Derek_l2271_227100


namespace greatest_k_value_l2271_227172

-- Define a type for triangle and medians intersecting at centroid
structure Triangle :=
(medianA : ℝ)
(medianB : ℝ)
(medianC : ℝ)
(angleA : ℝ)
(angleB : ℝ)
(angleC : ℝ)
(centroid : ℝ)

-- Define a function to determine if the internal angles formed by medians 
-- are greater than 30 degrees
def angle_greater_than_30 (θ : ℝ) : Prop :=
  θ > 30

-- A proof statement that given a triangle and its medians dividing an angle
-- into six angles, the greatest possible number of these angles greater than 30° is 3.
theorem greatest_k_value (T : Triangle) : ∃ k : ℕ, k = 3 ∧ 
  (∀ θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ, 
    (angle_greater_than_30 θ₁ ∨ angle_greater_than_30 θ₂ ∨ angle_greater_than_30 θ₃ ∨ 
     angle_greater_than_30 θ₄ ∨ angle_greater_than_30 θ₅ ∨ angle_greater_than_30 θ₆) → 
    k = 3) := 
sorry

end greatest_k_value_l2271_227172


namespace divisor_of_1053_added_with_5_is_2_l2271_227118

theorem divisor_of_1053_added_with_5_is_2 :
  ∃ d : ℕ, d > 1 ∧ ∀ (x : ℝ), x = 5.000000000000043 → (1053 + x) % d = 0 → d = 2 :=
by
  sorry

end divisor_of_1053_added_with_5_is_2_l2271_227118


namespace equalize_marbles_condition_l2271_227173

variables (D : ℝ)
noncomputable def marble_distribution := 
    let C := 1.25 * D
    let B := 1.4375 * D
    let A := 1.725 * D
    let total := A + B + C + D
    let equal := total / 4
    let move_from_A := (A - equal) / A * 100
    let move_from_B := (B - equal) / B * 100
    let add_to_C := (equal - C) / C * 100
    let add_to_D := (equal - D) / D * 100
    (move_from_A, move_from_B, add_to_C, add_to_D)

theorem equalize_marbles_condition :
    marble_distribution D = (21.56, 5.87, 8.25, 35.31) := sorry

end equalize_marbles_condition_l2271_227173


namespace scientific_notation_of_384000_l2271_227167

theorem scientific_notation_of_384000 : 384000 = 3.84 * 10^5 :=
by
  sorry

end scientific_notation_of_384000_l2271_227167


namespace divisor_is_twelve_l2271_227133

theorem divisor_is_twelve (d : ℕ) (h : 64 = 5 * d + 4) : d = 12 := 
sorry

end divisor_is_twelve_l2271_227133


namespace division_remainder_l2271_227160

theorem division_remainder (q d D R : ℕ) (h_q : q = 40) (h_d : d = 72) (h_D : D = 2944) (h_div : D = d * q + R) : R = 64 :=
by sorry

end division_remainder_l2271_227160


namespace simplify_and_evaluate_expr_l2271_227193

theorem simplify_and_evaluate_expr (x : ℤ) (h : x = -2) : 
  (2 * x + 1) * (x - 2) - (2 - x) ^ 2 = -8 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_expr_l2271_227193


namespace find_length_DY_l2271_227101

noncomputable def length_DY : Real :=
    let AE := 2
    let AY := 4 * AE
    let DY  := Real.sqrt (66 + Real.sqrt 5)
    DY

theorem find_length_DY : length_DY = Real.sqrt (66 + Real.sqrt 5) := 
  by
    sorry

end find_length_DY_l2271_227101


namespace problem_statement_l2271_227107

def f (x : ℝ) (a : ℝ) : ℝ := -x^2 + 6*x + a^2 - 1

theorem problem_statement (a : ℝ) :
  f (Real.sqrt 2) a < f 4 a ∧ f 4 a < f 3 a :=
by
  sorry

end problem_statement_l2271_227107


namespace commodity_x_increase_rate_l2271_227182

variable (x_increase : ℕ) -- annual increase in cents of commodity X
variable (y_increase : ℕ := 20) -- annual increase in cents of commodity Y
variable (x_2001_price : ℤ := 420) -- price of commodity X in cents in 2001
variable (y_2001_price : ℤ := 440) -- price of commodity Y in cents in 2001
variable (year_difference : ℕ := 2010 - 2001) -- difference in years between 2010 and 2001
variable (x_y_diff_2010 : ℕ := 70) -- cents by which X is more expensive than Y in 2010

theorem commodity_x_increase_rate :
  x_increase * year_difference = (x_2001_price + x_increase * year_difference) - (y_2001_price + y_increase * year_difference) + x_y_diff_2010 := by
  sorry

end commodity_x_increase_rate_l2271_227182


namespace total_cost_of_repair_l2271_227188

noncomputable def cost_of_repair (tire_cost: ℝ) (num_tires: ℕ) (tax: ℝ) (city_fee: ℝ) (discount: ℝ) : ℝ :=
  let total_cost := (tire_cost * num_tires : ℝ)
  let total_tax := (tax * num_tires : ℝ)
  let total_city_fee := (city_fee * num_tires : ℝ)
  (total_cost + total_tax + total_city_fee - discount)

def car_A_tire_cost : ℝ := 7
def car_A_num_tires : ℕ := 3
def car_A_tax : ℝ := 0.5
def car_A_city_fee : ℝ := 2.5
def car_A_discount : ℝ := (car_A_tire_cost * car_A_num_tires) * 0.05

def car_B_tire_cost : ℝ := 8.5
def car_B_num_tires : ℕ := 2
def car_B_tax : ℝ := 0 -- no sales tax
def car_B_city_fee : ℝ := 2.5
def car_B_discount : ℝ := 0 -- expired coupon

theorem total_cost_of_repair : 
  cost_of_repair car_A_tire_cost car_A_num_tires car_A_tax car_A_city_fee car_A_discount + 
  cost_of_repair car_B_tire_cost car_B_num_tires car_B_tax car_B_city_fee car_B_discount = 50.95 :=
by
  sorry

end total_cost_of_repair_l2271_227188


namespace minimum_bats_examined_l2271_227137

theorem minimum_bats_examined 
  (bats : Type) 
  (R L : bats → Prop) 
  (total_bats : ℕ)
  (right_eye_bats : ∀ {b: bats}, R b → Fin 2)
  (left_eye_bats : ∀ {b: bats}, L b → Fin 3)
  (not_left_eye_bats: ∀ {b: bats}, ¬ L b → Fin 4)
  (not_right_eye_bats: ∀ {b: bats}, ¬ R b → Fin 5)
  : total_bats ≥ 7 := sorry

end minimum_bats_examined_l2271_227137


namespace find_a9_l2271_227115

variable (a : ℕ → ℤ)
variable (h1 : a 2 = -3)
variable (h2 : a 3 = -5)
variable (d : ℤ := a 3 - a 2)

theorem find_a9 : a 9 = -17 :=
by
  sorry

end find_a9_l2271_227115


namespace marc_journey_fraction_l2271_227162

-- Defining the problem based on identified conditions
def total_cycling_time (k : ℝ) : ℝ := 20 * k
def total_walking_time (k : ℝ) : ℝ := 60 * (1 - k)
def total_travel_time (k : ℝ) : ℝ := total_cycling_time k + total_walking_time k

theorem marc_journey_fraction:
  ∀ (k : ℝ), total_travel_time k = 52 → k = 1 / 5 :=
by
  sorry

end marc_journey_fraction_l2271_227162


namespace terminating_decimal_expansion_of_7_over_72_l2271_227129

theorem terminating_decimal_expansion_of_7_over_72 : (7 / 72) = 0.175 := 
sorry

end terminating_decimal_expansion_of_7_over_72_l2271_227129


namespace cost_per_pouch_is_20_l2271_227141

theorem cost_per_pouch_is_20 :
  let boxes := 10
  let pouches_per_box := 6
  let dollars := 12
  let cents_per_dollar := 100
  let total_pouches := boxes * pouches_per_box
  let total_cents := dollars * cents_per_dollar
  let cost_per_pouch := total_cents / total_pouches
  cost_per_pouch = 20 :=
by
  sorry

end cost_per_pouch_is_20_l2271_227141


namespace correct_operation_l2271_227113

-- Define the conditions as hypotheses
variable (a : ℝ)

-- A: \(a^2 \cdot a = a^3\)
def condition_A : Prop := a^2 * a = a^3

-- B: \((a^3)^3 = a^6\)
def condition_B : Prop := (a^3)^3 = a^6

-- C: \(a^3 + a^3 = a^5\)
def condition_C : Prop := a^3 + a^3 = a^5

-- D: \(a^6 \div a^2 = a^3\)
def condition_D : Prop := a^6 / a^2 = a^3

-- Proof that only condition A is correct:
theorem correct_operation : condition_A a ∧ ¬condition_B a ∧ ¬condition_C a ∧ ¬condition_D a :=
by
  sorry  -- Actual proofs would go here

end correct_operation_l2271_227113


namespace algebra_expression_evaluation_l2271_227108

theorem algebra_expression_evaluation (a : ℝ) (h : a^2 + 2 * a - 1 = 5) : -2 * a^2 - 4 * a + 5 = -7 :=
by
  sorry

end algebra_expression_evaluation_l2271_227108


namespace abs_eq_neg_self_iff_l2271_227127

theorem abs_eq_neg_self_iff (a : ℝ) : |a| = -a ↔ a ≤ 0 :=
by
  -- skipping proof with sorry
  sorry

end abs_eq_neg_self_iff_l2271_227127


namespace range_of_x_l2271_227103

noncomputable def y (x : ℝ) : ℝ := (x + 2) / (x - 1)

theorem range_of_x : ∀ x : ℝ, (y x ≠ 0) → x ≠ 1 := by
  intro x h
  sorry

end range_of_x_l2271_227103


namespace smallest_number_of_brownies_l2271_227145

noncomputable def total_brownies (m n : ℕ) : ℕ := m * n
def perimeter_brownies (m n : ℕ) : ℕ := 2 * m + 2 * n - 4
def interior_brownies (m n : ℕ) : ℕ := (m - 2) * (n - 2)

theorem smallest_number_of_brownies : 
  ∃ (m n : ℕ), 2 * interior_brownies m n = perimeter_brownies m n ∧ total_brownies m n = 36 :=
by
  sorry

end smallest_number_of_brownies_l2271_227145


namespace initial_hours_per_day_l2271_227106

/-- 
Given:
1. 18 men working a certain number of hours per day dig 30 meters deep.
2. To dig to a depth of 50 meters, working 6 hours per day, 22 extra men should be put to work (total of 40 men).

Prove:
The initial 18 men were working \(\frac{200}{9}\) hours per day.
-/
theorem initial_hours_per_day 
  (h : ℚ)
  (work_done_18_men : 18 * h * 30 = 40 * 6 * 50) :
  h = 200 / 9 :=
by
  sorry

end initial_hours_per_day_l2271_227106


namespace inscribed_sphere_volume_l2271_227192

theorem inscribed_sphere_volume
  (a : ℝ)
  (h_cube_surface_area : 6 * a^2 = 24) :
  (4 / 3) * Real.pi * (a / 2)^3 = (4 / 3) * Real.pi :=
by
  -- sorry to skip the actual proof
  sorry

end inscribed_sphere_volume_l2271_227192


namespace two_p_plus_q_l2271_227122

theorem two_p_plus_q (p q : ℚ) (h : p / q = 6 / 7) : 2 * p + q = 19 / 7 * q :=
by {
  sorry
}

end two_p_plus_q_l2271_227122


namespace find_a_b_l2271_227120

-- Conditions defining the solution sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 3 }
def B : Set ℝ := { x | -3 < x ∧ x < 2 }

-- The solution set of the inequality x^2 + ax + b < 0 is the intersection A∩B
def C : Set ℝ := A ∩ B

-- Proving that there exist values of a and b such that the solution set C corresponds to the inequality x^2 + ax + b < 0
theorem find_a_b : ∃ a b : ℝ, (∀ x : ℝ, C x ↔ x^2 + a*x + b < 0) ∧ a + b = -3 := 
by 
  sorry

end find_a_b_l2271_227120


namespace no_infinite_monochromatic_arithmetic_progression_l2271_227189

theorem no_infinite_monochromatic_arithmetic_progression : 
  ∃ (coloring : ℕ → ℕ), (∀ (q r : ℕ), ∃ (n1 n2 : ℕ), coloring (q * n1 + r) ≠ coloring (q * n2 + r)) := sorry

end no_infinite_monochromatic_arithmetic_progression_l2271_227189


namespace employee_pays_correct_amount_l2271_227102

theorem employee_pays_correct_amount
    (wholesale_cost : ℝ)
    (retail_markup : ℝ)
    (employee_discount : ℝ)
    (weekend_discount : ℝ)
    (sales_tax : ℝ)
    (final_price : ℝ) :
    wholesale_cost = 200 →
    retail_markup = 0.20 →
    employee_discount = 0.05 →
    weekend_discount = 0.10 →
    sales_tax = 0.08 →
    final_price = 221.62 :=
by
  intros h0 h1 h2 h3 h4
  sorry

end employee_pays_correct_amount_l2271_227102


namespace combined_rent_C_D_l2271_227197

theorem combined_rent_C_D :
  let rent_per_month_area_z := 100
  let rent_per_month_area_w := 120
  let months_c := 3
  let months_d := 6
  let rent_c := months_c * rent_per_month_area_z
  let rent_d := months_d * rent_per_month_area_w
  let combined_rent := rent_c + rent_d
  combined_rent = 1020 :=
by
  let rent_per_month_area_z := 100
  let rent_per_month_area_w := 120
  let months_c := 3
  let months_d := 6
  let rent_c := months_c * rent_per_month_area_z
  let rent_d := months_d * rent_per_month_area_w
  let combined_rent := rent_c + rent_d
  show combined_rent = 1020
  sorry

end combined_rent_C_D_l2271_227197


namespace power_function_passes_through_point_l2271_227110

theorem power_function_passes_through_point (a : ℝ) : (2 ^ a = Real.sqrt 2) → (a = 1 / 2) :=
  by
  intro h
  sorry

end power_function_passes_through_point_l2271_227110


namespace concert_attendance_difference_l2271_227178

/-- Define the number of people attending the first concert. -/
def first_concert_attendance : ℕ := 65899

/-- Define the number of people attending the second concert. -/
def second_concert_attendance : ℕ := 66018

/-- The proof statement that the difference in attendance between the second and first concert is 119. -/
theorem concert_attendance_difference :
  (second_concert_attendance - first_concert_attendance = 119) := by
  sorry

end concert_attendance_difference_l2271_227178


namespace student_A_more_stable_l2271_227139

-- Defining the variances of students A and B as constants
def S_A_sq : ℝ := 0.04
def S_B_sq : ℝ := 0.13

-- Statement of the theorem
theorem student_A_more_stable : S_A_sq < S_B_sq → true :=
by
  -- proof will go here
  sorry

end student_A_more_stable_l2271_227139


namespace man_speed_proof_l2271_227105

noncomputable def train_length : ℝ := 150 
noncomputable def crossing_time : ℝ := 6 
noncomputable def train_speed_kmph : ℝ := 84.99280057595394 
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

noncomputable def relative_speed_mps : ℝ := train_length / crossing_time
noncomputable def man_speed_mps : ℝ := relative_speed_mps - train_speed_mps
noncomputable def man_speed_kmph : ℝ := man_speed_mps * (3600 / 1000)

theorem man_speed_proof : man_speed_kmph = 5.007198224048459 := by 
  sorry

end man_speed_proof_l2271_227105


namespace b_minus_d_sq_value_l2271_227187

theorem b_minus_d_sq_value 
  (a b c d : ℝ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 3)
  (h3 : 2 * a - 3 * b + c + 4 * d = 17) :
  (b - d) ^ 2 = 25 :=
by
  sorry

end b_minus_d_sq_value_l2271_227187


namespace value_expression_l2271_227150

-- Definitions
variable (m n : ℝ)
def reciprocals (m n : ℝ) := m * n = 1

-- Theorem statement
theorem value_expression (m n : ℝ) (h : reciprocals m n) : m * n^2 - (n - 3) = 3 := by
  sorry

end value_expression_l2271_227150


namespace greatest_possible_difference_l2271_227140

def is_reverse (q r : ℕ) : Prop :=
  let q_tens := q / 10
  let q_units := q % 10
  let r_tens := r / 10
  let r_units := r % 10
  (q_tens = r_units) ∧ (q_units = r_tens)

theorem greatest_possible_difference (q r : ℕ) (hq1 : q ≥ 10) (hq2 : q < 100)
  (hr1 : r ≥ 10) (hr2 : r < 100) (hrev : is_reverse q r) (hpos_diff : q - r < 30) :
  q - r ≤ 27 :=
by
  sorry

end greatest_possible_difference_l2271_227140


namespace remainder_of_division_l2271_227175

theorem remainder_of_division (x : ℕ) (r : ℕ) :
  1584 - x = 1335 ∧ 1584 = 6 * x + r → r = 90 := by
  sorry

end remainder_of_division_l2271_227175


namespace non_deg_ellipse_projection_l2271_227134

theorem non_deg_ellipse_projection (m : ℝ) : 
  (3 * x^2 + 9 * y^2 - 12 * x + 18 * y + 6 * z = m → (m > -21)) := 
by
  sorry

end non_deg_ellipse_projection_l2271_227134
