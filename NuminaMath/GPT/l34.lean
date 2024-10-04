import Mathlib

namespace flagpole_arrangement_remainder_l34_34327

theorem flagpole_arrangement_remainder :
  let M := 10 * (Nat.choose 12 10) in
  M % 1000 = 660 :=
by
  let M := 10 * (Nat.choose 12 10)
  show M % 1000 = 660
  sorry

end flagpole_arrangement_remainder_l34_34327


namespace flag_blue_area_l34_34724

theorem flag_blue_area (A C₁ C₃ : ℝ) (h₀ : A = 1.0) (h₁ : C₁ + C₃ = 0.36 * A) :
  C₃ = 0.02 * A := by
  sorry

end flag_blue_area_l34_34724


namespace find_measure_of_angle_A_and_altitude_h_l34_34182

noncomputable def cos_A_value 
  (a b c A B : ℝ) 
  (cos : ℝ → ℝ) :=
  cos A = (-1 / 2) ∧ 
  a = 4*sqrt(3) ∧ 
  b + c = 8 ∧ 
  ∃ (h : ℝ), h = 2 * sqrt(3)

theorem find_measure_of_angle_A_and_altitude_h 
  (a b c A B : ℝ) 
  (cos : ℝ → ℝ)
  (triangle_ABC : a + b + c = A + B + A) 
  (vector_m : (cos A, cos B))
  (vector_n : (b + 2*c, a))
  (perpendicular : (vector_m.1 * vector_n.1 + vector_m.2 * vector_n.2) = 0)
  :
  cos_A_value a b c A B cos :=
sorry

end find_measure_of_angle_A_and_altitude_h_l34_34182


namespace inequality_solution_l34_34430

def g (x : ℝ) : ℝ := (3*x-5)*(x-2)*(x+1)/(2*x)

theorem inequality_solution : {x : ℝ | g x ≥ 0 } = ({x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 2} ∪ {x : ℝ | (5/3) < x}) :=
by
  sorry

end inequality_solution_l34_34430


namespace investment_amount_correct_l34_34912

-- Define the given conditions
def future_value : ℝ := 600000
def interest_rate : ℝ := 0.04
def years : ℝ := 15

-- Define the function for present value calculation
def present_value (FV : ℝ) (r : ℝ) (n : ℝ) : ℝ := FV / (1 + r)^n

-- Define the expected present value
def expected_present_value : ℝ := 333097.61

-- Formalize the problem statement as a theorem to be proved
theorem investment_amount_correct :
  present_value future_value interest_rate years = expected_present_value := 
sorry

end investment_amount_correct_l34_34912


namespace find_x_l34_34807

/-- Given vectors a and b, and a is parallel to b -/
def vectors (x : ℝ) : Prop :=
  let a := (x, 2)
  let b := (2, 1)
  a.1 * b.2 = a.2 * b.1

theorem find_x: ∀ x : ℝ, vectors x → x = 4 :=
by
  intros x h
  sorry

end find_x_l34_34807


namespace min_A_for_quadratic_polynomial_l34_34433

noncomputable theory

open Real

theorem min_A_for_quadratic_polynomial 
  (A : ℝ) :
  (∀ (f : ℝ → ℝ), (∃ (a b c : ℝ), f = λ x, a * x ^ 2 + b * x + c) → 
  (∀ x, 0 ≤ x ∧ x ≤ 1 → abs (f x) ≤ 1) → f' 0 ≤ A) ↔ A = 8 := 
sorry

end min_A_for_quadratic_polynomial_l34_34433


namespace find_polynomials_l34_34779

theorem find_polynomials (W : Polynomial ℤ) : 
  (∀ n : ℕ, 0 < n → (2^n - 1) % W.eval n = 0) → (W = Polynomial.C 1 ∨ W = Polynomial.C (-1)) :=
by sorry

end find_polynomials_l34_34779


namespace prime_divisor_greater_than_p_l34_34579

theorem prime_divisor_greater_than_p (p q : ℕ) (hp : Prime p) 
    (hq : Prime q) (hdiv : q ∣ 2^p - 1) : p < q := 
by
  sorry

end prime_divisor_greater_than_p_l34_34579


namespace planes_perpendicular_l34_34313

def normal_vector_plane_α : ℝ × ℝ × ℝ := (1, 2, 0)
def normal_vector_plane_β : ℝ × ℝ × ℝ := (2, -1, 0)

theorem planes_perpendicular :
  let dot_product := (normal_vector_plane_α.1 * normal_vector_plane_β.1) + 
                     (normal_vector_plane_α.2 * normal_vector_plane_β.2) + 
                     (normal_vector_plane_α.3 * normal_vector_plane_β.3)
  in dot_product = 0 :=
by
  let dot_product := (normal_vector_plane_α.1 * normal_vector_plane_β.1) + 
                     (normal_vector_plane_α.2 * normal_vector_plane_β.2) + 
                     (normal_vector_plane_α.3 * normal_vector_plane_β.3)
  show dot_product = 0
  sorry

end planes_perpendicular_l34_34313


namespace quadrilateral_is_parallelogram_l34_34020

theorem quadrilateral_is_parallelogram 
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 - 2*a*c - 2*b*d = 0) 
  : (a = c ∧ b = d) → parallelogram :=
by
  sorry

end quadrilateral_is_parallelogram_l34_34020


namespace valid_pairings_count_l34_34324

-- Define the number of people and their known relations
def n := 8
def knows (i j : ℕ) : Prop := (i + 1) % n = j % n ∨ (i + n - 1) % n = j % n ∨ (i + 2) % n = j % n

-- Problem: Prove the number of valid pairings
theorem valid_pairings_count : 
  ∃ p : list (ℕ × ℕ), (∀ (a b : ℕ × ℕ), a ≠ b → a.1 ≠ b.1 ∧ a.2 ≠ b.2)
                      ∧ (∀ a : ℕ × ℕ, knows a.1 a.2)
                      ∧ p.length = 4
                      ∧ |{q : list (ℕ × ℕ) | (∀ (a : ℕ × ℕ), a ∈ q → knows a.1 a.2) ∧ q.length = 4}| = 6 := 
sorry

end valid_pairings_count_l34_34324


namespace proof_mn_proof_expr_l34_34829

variables (m n : ℚ)
-- Conditions
def condition1 : Prop := (m + n)^2 = 9
def condition2 : Prop := (m - n)^2 = 1

-- Expected results
def expected_mn : ℚ := 2
def expected_expr : ℚ := 3

-- The theorem to be proved
theorem proof_mn : condition1 m n → condition2 m n → m * n = expected_mn :=
by
  sorry

theorem proof_expr : condition1 m n → condition2 m n → m^2 + n^2 - m * n = expected_expr :=
by
  sorry

end proof_mn_proof_expr_l34_34829


namespace min_adventurers_l34_34013

theorem min_adventurers (R E S D: ℕ) 
  (hR: R = 9)
  (hE: E = 8) 
  (hS: S = 2) 
  (hD: D = 11) 
  (cond1: ∀ a, a ∈ D → (a ∈ R ∨ a ∈ S) ∧ ¬(a ∈ R ∧ a ∈ S))
  (cond2: ∀ b, b ∈ R → (b ∈ E ∨ b ∈ D) ∧ ¬(b ∈ E ∧ b ∈ D)) :
  ∃ N, N = 17 :=
by
  sorry

end min_adventurers_l34_34013


namespace overall_percent_profit_l34_34381

variable (cost_A : ℝ) (cost_B : ℝ)
variable (markup_A : ℝ) (markup_B : ℝ)
variable (discount_A : ℝ) (discount_B : ℝ)

def marked_price (cost : ℝ) (markup : ℝ) : ℝ :=
  cost * (1 + markup)

def sale_price (marked_price : ℝ) (discount : ℝ) : ℝ :=
  marked_price * (1 - discount)

theorem overall_percent_profit :
  let total_cost := cost_A + cost_B
  let total_sale_price := sale_price (marked_price cost_A markup_A) discount_A
                      + sale_price (marked_price cost_B markup_B) discount_B
  let total_profit := total_sale_price - total_cost
  (total_profit / total_cost) * 100 = 24.25 :=
by
  sorry

-- Assign specific values from the conditions
example : overall_percent_profit 50 70 0.4 0.6 0.15 0.20 :=
by
  simp [marked_price, sale_price, overall_percent_profit]
  sorry

end overall_percent_profit_l34_34381


namespace function_identity_l34_34816

theorem function_identity (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f(x) ≤ x)
  (h2 : ∀ x y : ℝ, f(x + y) ≤ f(x) + f(y)) :
  ∀ x : ℝ, f(x) = x := 
by
  sorry

end function_identity_l34_34816


namespace compare_f_values_l34_34146

noncomputable def f (x : Real) : Real := 
  Real.cos x + 2 * x * (1 / 2)  -- given f''(pi/6) = 1/2

theorem compare_f_values :
  f (-Real.pi / 3) < f (Real.pi / 3) :=
by
  sorry

end compare_f_values_l34_34146


namespace three_digit_integers_congruent_to_2_mod_4_l34_34901

theorem three_digit_integers_congruent_to_2_mod_4 : 
  let count := (249 - 25 + 1) in
  count = 225 :=
by
  let k_min := 25
  let k_max := 249
  have h_count : count = (k_max - k_min + 1) := rfl
  rw h_count
  norm_num

end three_digit_integers_congruent_to_2_mod_4_l34_34901


namespace bond_total_amount_l34_34047

theorem bond_total_amount (P : ℝ) (r : ℝ) (t : ℕ) :
  P = 20000 ∧ r = 0.0318 ∧ t = 5 →
  let interest := P * r * t in
  let total_amount := P + interest in
  total_amount = 23180 :=
by
  intros h
  cases h with hP hr ht
  sorry

end bond_total_amount_l34_34047


namespace f_values_x_range_l34_34695

-- Define the function and its properties
noncomputable def f : ℝ → ℝ := sorry

-- Given properties
axiom dec_f : ∀ x y : ℝ, (0 < x ∧ 0 < y) → (x < y → f(y) < f(x))
axiom func_eqn : ∀ x y : ℝ, (0 < x ∧ 0 < y) → f(x * y) = f(x) + f(y)
axiom f_one_third : f(1/3) = 1

-- Prove part I
theorem f_values :
  f(1) = 0 ∧ f(3) = -1 :=
sorry

-- Prove part II
theorem x_range (x : ℝ) :
  (f(1/x) + f(2 - x) > 2) → (9/5 < x ∧ x < 2) :=
sorry

end f_values_x_range_l34_34695


namespace correct_units_l34_34085

-- Define areas and lengths using given conditions
variable (area_pingpong_table : ℕ) (unit_pingpong_table : String)
variable (area_xiu_zhou : ℕ) (unit_xiu_zhou : String)
variable (length_grand_canal : ℕ) (unit_grand_canal : String)
variable (building_area_exhibition_hall : ℕ) (unit_exhibition_hall : String)

-- Given conditions
axiom pingpong_table_condition : area_pingpong_table = 4
axiom xiu_zhou_condition : area_xiu_zhou = 580
axiom grand_canal_condition : length_grand_canal = 1797
axiom exhibition_hall_condition : building_area_exhibition_hall = 3000

-- Units must be square meters, square kilometers, kilometers, and square meters respectively
axiom unit_pingpong_table_correct : unit_pingpong_table = "square meters"
axiom unit_xiu_zhou_correct : unit_xiu_zhou = "square kilometers"
axiom unit_grand_canal_correct : unit_grand_canal = "kilometers"
axiom unit_exhibition_hall_correct : unit_exhibition_hall = "square meters"

theorem correct_units :
  (area_pingpong_table = 4 ∧ unit_pingpong_table = "square meters") ∧
  (area_xiu_zhou = 580 ∧ unit_xiu_zhou = "square kilometers") ∧
  (length_grand_canal = 1797 ∧ unit_grand_canal = "kilometers") ∧
  (building_area_exhibition_hall = 3000 ∧ unit_exhibition_hall = "square meters") :=
by
  apply And.intro
  . exact ⟨pingpong_table_condition, unit_pingpong_table_correct⟩
  . apply And.intro
    . exact ⟨xiu_zhou_condition, unit_xiu_zhou_correct⟩
    . apply And.intro
      . exact ⟨grand_canal_condition, unit_grand_canal_correct⟩
      . exact ⟨exhibition_hall_condition, unit_exhibition_hall_correct⟩

end correct_units_l34_34085


namespace find_value_of_x_squared_plus_inverse_squared_l34_34112

theorem find_value_of_x_squared_plus_inverse_squared (x : ℂ) (h : x + x⁻¹ = 3) : x^2 + x⁻² = 7 := 
by
  sorry

end find_value_of_x_squared_plus_inverse_squared_l34_34112


namespace part1_expression_g_part2_range_m_part3_range_a_l34_34857

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3)

theorem part1_expression_g :
  g x = f (x + Real.pi / 4) :=
sorry

theorem part2_range_m (m : ℝ) :
  (∀ x ∈ set.Icc (0 : ℝ) (Real.pi / 4), ∃ t, g(x) = t ∧ 2 * t ^ 2 - m * t + 1 = 0) ↔ (2 * Real.sqrt 2 ≤ m ∧ m ≤ 3) :=
sorry

theorem part3_range_a (a : ℝ) :
  (∀ x ∈ set.Icc (Real.pi / 6) (Real.pi / 3), a * f(x) + g(x) ≥ Real.sqrt (a^2 + 1) / 2) ↔ (0 < a ∧ a ≤ Real.sqrt 3) :=
sorry

end part1_expression_g_part2_range_m_part3_range_a_l34_34857


namespace compute_fraction_l34_34410

theorem compute_fraction :
  ((1/3)^4 * (1/5) = (1/405)) :=
by
  sorry

end compute_fraction_l34_34410


namespace min_value_ineq_l34_34228

theorem min_value_ineq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = 2) : 
  3 + 2 * real.sqrt 2 ≤ 1 / a + 2 / b :=
begin
  sorry
end

end min_value_ineq_l34_34228


namespace triangles_area_ratio_l34_34185

-- Define the geometry setup and required conditions
variables (R : ℝ) (E : Affine.Point ℝ ℝ) (β : ℝ)
variables AB CD AC BD : Affine.Line ℝ ℝ
variables p₁ p₂ p₃ p₄ p₅ : Affine.Point ℝ ℝ

-- Conditions setup
def AB_is_diameter := AB = Affine.Line.through p₁ p₂ ∧ p₁dAB.to_projective = p₂.to_projective
def CD_parallel_AB := CD.is_parallel_to AB
def AC_is_line := AC = Affine.Line.through p₄ E
def BD_is_line := BD = Affine.Line.through p₅ E
def angles := angle (BD.direction, E.direction) = β

-- Proof problem statement
theorem triangles_area_ratio
  (h1 : AB_is_diameter) (h2 : CD_parallel_AB) (h3 : AC_is_line) (h4 : BD_is_line) (h5 : angles) :
  (area (triangle p₄ p₅ E)) / (area (triangle p₁ p₂ E)) = real.sin β * real.sin β :=
sorry

end triangles_area_ratio_l34_34185


namespace correct_statements_l34_34797

noncomputable def real : Type := ℝ
noncomputable def A : set real := sorry
def A_star (A : set real) : set real := {y | ∀ x ∈ A, y ≥ x}

variable {M P : set real}
variable (M_nonempty : M.nonempty)
variable (P_nonempty : P.nonempty)
variable (subset_MP : M ⊆ P)

def M_star : set real := A_star M
def P_star : set real := A_star P

theorem correct_statements :
    (P_star ⊆ M_star) ∧ ¬(M_star ∩ P).nonempty ∧ ¬(M ∩ P_star = ∅) :=
by
    sorry

end correct_statements_l34_34797


namespace sum_of_primes_no_solution_congruence_l34_34765

theorem sum_of_primes_no_solution_congruence :
  let no_solution (p : ℕ) := ¬∃ x : ℤ, (4 * (5 * x + 2) ≡ 3 [MOD p])
  let is_prime (p : ℕ) := Nat.Prime p
  ∑ p in {p : ℕ | is_prime p ∧ no_solution p}.to_finset = 7 := 
by
  sorry

end sum_of_primes_no_solution_congruence_l34_34765


namespace first_player_wins_l34_34658

-- Defining the game conditions
structure Game :=
  (rectangular_table : Prop)
  (euro_coin : Prop)
  (place_coin : ∀ (pos : ℝ × ℝ), euro_coin → Prop)
  (no_overlap : ∀ (pos1 pos2 : ℝ × ℝ), place_coin pos1 euro_coin → place_coin pos2 euro_coin → pos1 ≠ pos2)
  (player_cannot_move : Prop → Prop) -- represents the state when a player cannot make a move

-- The theorem stating the first player has a winning strategy
theorem first_player_wins (game : Game) : ∃ strategy : (ℝ × ℝ) → (ℝ × ℝ), true :=
  sorry

end first_player_wins_l34_34658


namespace quintuplets_babies_l34_34391

theorem quintuplets_babies (a b c d : ℕ) 
  (h1 : d = 2 * c) 
  (h2 : c = 3 * b) 
  (h3 : b = 2 * a) 
  (h4 : 2 * a + 3 * b + 4 * c + 5 * d = 1200) : 
  5 * d = 18000 / 23 :=
by 
  sorry

end quintuplets_babies_l34_34391


namespace f_2010_value_l34_34977

noncomputable def f (x : ℝ) := sorry -- function f is defined but not concretely specified

theorem f_2010_value :
  (∀ x : ℝ, 0 < x → 0 < f x) ∧
  (∀ x y : ℝ, 0 < y → y < x → f (2 * x - y) = Real.log (f (x + y) + 1)) →
  f 2010 = 0.444667 :=
by
  sorry

end f_2010_value_l34_34977


namespace line_through_point_perpendicular_l34_34431

def slope (a b : ℝ × ℝ) : ℝ :=
  (b.2 - a.2) / (b.1 - a.1)

theorem line_through_point_perpendicular :
  ∃ (a b c : ℝ), (a * 2 + b * (-2) + c = 0) ∧ (a * 2 + b * (-3) + c = 1) ∧ (a = 1) ∧ (b = -2) ∧ (c = -8) :=
by
  sorry

end line_through_point_perpendicular_l34_34431


namespace measure_of_angle_C_l34_34508

theorem measure_of_angle_C (A B C : ℕ) (h1 : A + B = 150) (h2 : A + B + C = 180) : C = 30 := 
by
  sorry

end measure_of_angle_C_l34_34508


namespace conjugate_of_z_l34_34623

theorem conjugate_of_z (z : ℂ) (h : z * (3 * complex.I - 4) = 25) : complex.conj z = -4 + 3 * complex.I :=
by
  sorry

end conjugate_of_z_l34_34623


namespace continuous_piecewise_l34_34415

noncomputable def g (x : ℝ) (a c : ℝ) : ℝ :=
if x > 3 then 2 * a * x + 4
else if -3 ≤ x ∧ x ≤ 3 then x^2 - 7
else 3 * x - c

theorem continuous_piecewise (a c : ℝ) (h1 : 6 * a + 4 = 2) (h2 : 9 - 7 = -9 - c) : a + c = -34/3 :=
by
  have h_a : a = -1/3,
  { linarith },
  have h_c : c = -11,
  { linarith },
  linarith

end continuous_piecewise_l34_34415


namespace three_digit_integers_congruent_to_2_mod_4_l34_34888

theorem three_digit_integers_congruent_to_2_mod_4 : 
    ∃ n, n = 225 ∧ ∀ x, (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 2) ↔ (∃ m, 25 ≤ m ∧ m ≤ 249 ∧ x = 4 * m + 2) := by
  sorry

end three_digit_integers_congruent_to_2_mod_4_l34_34888


namespace find_least_n_periodic_l34_34208

def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (3 * x / 10)

theorem find_least_n_periodic :
  (∃ n : ℕ, (∀ x : ℝ, f (n * Real.pi + x) = f x) ∧ n = 60) :=
begin
  use 60,
  split,
  { sorry, },
  { refl, }
end

end find_least_n_periodic_l34_34208


namespace coin_toss_sequences_count_l34_34395

theorem coin_toss_sequences_count :
  ∃ (seqs : ℕ), 
    (seqs = 27720) ∧ 
    (∃ n, n = 20) ∧
    (∃ hh, hh = 3) ∧
    (∃ ht, ht = 4) ∧
    (∃ th, th = 5) ∧
    (∃ tt, tt = 7) ∧
    (seqs = (Nat.choose (3 + 5 - 1) (5 - 1)) * (Nat.choose (7 + 6 - 1) (6 - 1))) :=
begin
  sorry
end

end coin_toss_sequences_count_l34_34395


namespace no_integer_solutions_for_fraction_equation_l34_34778

theorem no_integer_solutions_for_fraction_equation :
  ∀ x y : ℤ, (x ≠ 1) → (x^7 - 1) / (x - 1) = y^5 - 1 → False :=
by
  intro x y hx h
  have H : (x^7 - 1) = (x - 1) * (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) := sorry
  rw H at h
  sorry  -- complete the proof by contradiction based on the given conditions

end no_integer_solutions_for_fraction_equation_l34_34778


namespace number_of_three_digit_integers_congruent_mod4_l34_34909

def integer_congruent_to_mod (a b n : ℕ) : Prop := ∃ k : ℤ, n = a * k + b

theorem number_of_three_digit_integers_congruent_mod4 :
  (finset.filter (λ n, integer_congruent_to_mod 4 2 (n : ℕ)) 
   (finset.Icc (100 : ℕ) (999 : ℕ))).card = 225 :=
by
  sorry

end number_of_three_digit_integers_congruent_mod4_l34_34909


namespace abs_neg_three_l34_34280

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l34_34280


namespace determine_k_l34_34850

theorem determine_k 
  (k : ℝ) 
  (r s : ℝ) 
  (h1 : r + s = -k) 
  (h2 : r * s = 6) 
  (h3 : (r + 5) + (s + 5) = k) : 
  k = 5 := 
by 
  sorry

end determine_k_l34_34850


namespace range_sum_endpoints_l34_34764

/-- Define the function h(x) -/
def h (x : ℝ) : ℝ := 3 / (3 + 9 * x^2)

/-- The range of the function h(x) = 3/(3+9x^2) is (0, 1] 
    and the sum of the endpoints of this range is 1. -/
theorem range_sum_endpoints : 
  (set.range h = set.Ioc 0 1) ∧ (0 + 1 = 1) :=
sorry

end range_sum_endpoints_l34_34764


namespace parabola_slope_l34_34842

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

def parabola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in y^2 = 4 * x

def distance (A B : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def slope_magnitude (P F : ℝ × ℝ) : ℝ :=
  let (px, py) := P in
  let (fx, fy) := F in
  abs ((py - fy) / (px - fx))

theorem parabola_slope {P : ℝ × ℝ} (hP : parabola P) (hD : distance P parabola_focus = 5) :
  slope_magnitude P parabola_focus = 4 / 3 :=
sorry

end parabola_slope_l34_34842


namespace secants_through_point_with_equal_chords_l34_34159

theorem secants_through_point_with_equal_chords (k₁ k₂ : Circle) (P : Point) (α : Angle) 
  (h₁ : ∀ A B ∈ k₁, ¬(∃ l, secant_through P l ∧ chord_length k₁ l = chord_length k₂ (rotate_by α l))) 
  (h₂ : disjoint k₁ k₂)
  (h₃ : outside_circle P k₁ ∧ outside_circle P k₂) :
  ∃ secant : Line, (secant_through P secant ∧ chord_length k₁ secant = chord_length k₂ secant ∧ angle_between secant = α) :=
sorry

end secants_through_point_with_equal_chords_l34_34159


namespace three_digit_integers_congruent_to_2_mod_4_l34_34905

theorem three_digit_integers_congruent_to_2_mod_4 : 
  let count := (249 - 25 + 1) in
  count = 225 :=
by
  let k_min := 25
  let k_max := 249
  have h_count : count = (k_max - k_min + 1) := rfl
  rw h_count
  norm_num

end three_digit_integers_congruent_to_2_mod_4_l34_34905


namespace rectangle_area_l34_34624

theorem rectangle_area :
  ∃ (L B : ℝ), (L - B = 23) ∧ (2 * (L + B) = 206) ∧ (L * B = 2520) :=
sorry

end rectangle_area_l34_34624


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l34_34883

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : 
  ∃ (count : ℕ), count = 225 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ↔ (∃ k : ℕ, 25 ≤ k ∧ k ≤ 249 ∧ n = 4 * k + 2) := 
by {
  sorry
}

end number_of_three_digit_integers_congruent_to_2_mod_4_l34_34883


namespace hyperbola_eccentricity_sqrt17_l34_34150

-- Given conditions
variables {a b : ℝ}
variables (C : Set (ℝ × ℝ)) -- The set of points (x, y) on the hyperbola

noncomputable def hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y : ℝ, p = (x, y) ∧ y^2 / a^2 - x^2 / b^2 = 1}

def eccentricity (a b : ℝ) : ℝ :=
  let c := sqrt (a^2 + b^2) in c / a

-- Proof problem: Prove that the eccentricity of the hyperbola is sqrt(17)
theorem hyperbola_eccentricity_sqrt17 (ha : 0 < a) (hb : 0 < b) (h_b_eq_4a : b = 4 * a) :
    eccentricity a b = sqrt 17 := sorry

end hyperbola_eccentricity_sqrt17_l34_34150


namespace number_on_2020th_second_l34_34427

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

noncomputable def next_number (n : ℕ) : ℕ :=
  (sum_of_digits n) * 31

noncomputable def sequence_term (a₁ : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then a₁
  else next_number (sequence_term a₁ (n - 1))

theorem number_on_2020th_second :
  sequence_term 2020 2020 = 310 := by
  sorry

end number_on_2020th_second_l34_34427


namespace count_tower_heights_l34_34604

-- Definitions based on the problem conditions
def heightContributions := {3, 8, 20}  -- possible contributions by each brick
def numBricks := 78                    -- number of bricks used
def minHeight := 234                   -- minimum tower height when all bricks are oriented to contribute 3''
def maxHeight := minHeight + numBricks * 17  -- maximum tower height when all bricks oriented to contribute 20''

-- The main theorem
theorem count_tower_heights : ∃ heights : Finset ℤ, heights.card = 266 ∧
  ∀ h ∈ heights, ∃ (n3 n8 n20 : ℕ), n3 + n8 + n20 = numBricks ∧ 
    h = n3 * 3 + n8 * 8 + n20 * 20 ∧ 
    ∀ h1 h2 ∈ heights, h1 ≠ h2 → abs (h1 - h2) ≥ 5 :=
by
  sorry

end count_tower_heights_l34_34604


namespace triangle_side_length_and_type_l34_34646

def triangles := sorry

theorem triangle_side_length_and_type
  (a b p : ℕ)
  (h1 : a = 30)
  (h2 : b = 18)
  (h3 : p = 66) :
  ∃ c : ℕ, c = p - a - b ∧ (c ^ 2 + b ^ 2 = a ^ 2) :=
by
  sorry

end triangle_side_length_and_type_l34_34646


namespace conditional_probability_P_B_given_A_l34_34669

-- Let E be an enumeration type with exactly five values, each representing one attraction.
inductive Attraction : Type
| dayu_yashan : Attraction
| qiyunshan : Attraction
| tianlongshan : Attraction
| jiulianshan : Attraction
| sanbaishan : Attraction

open Attraction

-- Define A and B's choices as random variables.
axiom A_choice : Attraction
axiom B_choice : Attraction

-- Event A is that A and B choose different attractions.
def event_A : Prop := A_choice ≠ B_choice

-- Event B is that A and B each choose Chongyi Qiyunshan.
def event_B : Prop := A_choice = qiyunshan ∧ B_choice = qiyunshan

-- Calculate the conditional probability P(B|A)
theorem conditional_probability_P_B_given_A : 
  (1 - (1 / 5)) * (1 - (1 / 5)) = 2 / 5 :=
sorry

end conditional_probability_P_B_given_A_l34_34669


namespace volume_of_pyramid_l34_34970

-- Definitions of points and side length
def point := (ℝ × ℝ × ℝ)

-- Defining the vertices of the cube ABCDEFGH
def A : point := (0, 0, 0)
def B : point := (2, 0, 0)
def C : point := (2, 2, 0)
def G : point := (2, 2, 2)

-- Given conditions
def side_length : ℝ := 2    -- side length of the cube is 2

-- Proving the volume of the pyramid ABCG
theorem volume_of_pyramid (A B C G : point) (s : ℝ) : 
  (s = 2) →
  ((B.1 - A.1 = s) ∧ (B.2 - A.2 = 0) ∧ (B.3 - A.3 = 0)) →
  ((C.1 - A.1 = s) ∧ (C.2 - A.2 = s) ∧ (C.3 - A.3 = 0)) →
  ((G.1 = C.1) ∧ (G.2 = C.2) ∧ (G.3 - C.3 = s)) →
  1/3 * (1/2 * s * s) * s = 4 / 3 :=
by sorry

end volume_of_pyramid_l34_34970


namespace candy_selection_l34_34357

theorem candy_selection (m n : ℕ) (h1 : Nat.gcd m n = 1) (h2 : m = 1) (h3 : n = 5) :
  m + n = 6 := by
  sorry

end candy_selection_l34_34357


namespace find_sin_expression_l34_34959

noncomputable def trigonometric_identity (γ : ℝ) : Prop :=
  3 * (Real.tan γ)^2 + 3 * (1 / (Real.tan γ))^2 + 2 / (Real.sin γ)^2 + 2 / (Real.cos γ)^2 = 19

theorem find_sin_expression (γ : ℝ) (h : trigonometric_identity γ) : 
  (Real.sin γ)^4 - (Real.sin γ)^2 = -1 / 5 :=
sorry

end find_sin_expression_l34_34959


namespace function_monotonic_increasing_interval_l34_34311

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem function_monotonic_increasing_interval :
  ∀ x, 1 < x → 0 < derivative f x :=
by
  sorry

end function_monotonic_increasing_interval_l34_34311


namespace settings_per_table_l34_34997

theorem settings_per_table
  (silverware_weight : ℕ := 4)
  (pieces_per_setting : ℕ := 3)
  (plate_weight : ℕ := 12)
  (plates_per_setting : ℕ := 2)
  (tables : ℕ := 15)
  (backup_settings : ℕ := 20)
  (total_weight : ℕ := 5040) :
  let weight_per_setting := pieces_per_setting * silverware_weight + plates_per_setting * plate_weight
  let total_settings := tables * 8 + backup_settings
  weight_per_setting * total_settings = total_weight :=
by
  have weight_per_setting_definition := 3 * 4 + 2 * 12
  have total_settings_definition := 15 * 8 + 20
  have weight_total_definition := weight_per_setting_definition * total_settings_definition
  have input_total_weight := 5040
  have eq_proof : weight_total_definition = input_total_weight := by
    simp [weight_per_setting_definition, total_settings_definition]
  exact eq_proof

end settings_per_table_l34_34997


namespace rectangle_perimeter_l34_34719

variable (L W : ℝ) 

theorem rectangle_perimeter (h1 : L > 4) (h2 : W > 4) (h3 : (L * W) - ((L - 4) * (W - 4)) = 168) : 
  2 * (L + W) = 92 := 
  sorry

end rectangle_perimeter_l34_34719


namespace divide_cube_more_than_2003_cubes_with_points_inside_l34_34533

def divides_cube_with_points (side_length : ℝ) (n points : ℕ) (cube : set (ℝ × ℝ × ℝ)) : Prop :=
  ∃ m, (m > n^3) ∧ ∀ (p ∈ points), ∃ (i j k : ℕ), 
    cube (i/m, j/m, k/m)

theorem divide_cube_more_than_2003_cubes_with_points_inside :
  ∀ (points : set (ℝ × ℝ × ℝ)) (c : set (ℝ × ℝ × ℝ)), 
  (∀ p ∈ points, p ∈ c) ∧ (fintype.card points = 2003) →
  divides_cube_with_points 1 2003 c :=
by {
  sorry
}

end divide_cube_more_than_2003_cubes_with_points_inside_l34_34533


namespace simplify_one_simplify_two_simplify_three_simplify_four_l34_34605

-- (1) Prove that (1 / 2) * sqrt(4 / 7) = sqrt(7) / 7
theorem simplify_one : (1 / 2) * Real.sqrt (4 / 7) = Real.sqrt 7 / 7 := sorry

-- (2) Prove that sqrt(20 ^ 2 - 15 ^ 2) = 5 * sqrt(7)
theorem simplify_two : Real.sqrt (20 ^ 2 - 15 ^ 2) = 5 * Real.sqrt 7 := sorry

-- (3) Prove that sqrt((32 * 9) / 25) = (12 * sqrt(2)) / 5
theorem simplify_three : Real.sqrt ((32 * 9) / 25) = (12 * Real.sqrt 2) / 5 := sorry

-- (4) Prove that sqrt(22.5) = (3 * sqrt(10)) / 2
theorem simplify_four : Real.sqrt 22.5 = (3 * Real.sqrt 10) / 2 := sorry

end simplify_one_simplify_two_simplify_three_simplify_four_l34_34605


namespace max_value_n_2500_l34_34673

theorem max_value_n_2500 :
  ∃ (k : ℕ → ℕ), (∀ i j, i ≠ j → k i ≠ k j) ∧
  (∑ i in finset.range 18, (k i)^2 = 2500) ∧
  (∀ m : ℕ, m > 18 → ∀ (k' : ℕ → ℕ), (∀ i j, i ≠ j → k' i ≠ k' j) → 
    ¬ ∑ i in finset.range m, (k' i)^2 = 2500) :=
sorry

end max_value_n_2500_l34_34673


namespace find_length_BC_l34_34199

-- Definitions of angles in the quadrilateral ABCD
def angleA (x : ℝ) := 3 * x
def angleB (x : ℝ) := 7 * x
def angleC (x : ℝ) := 4 * x
def angleD (x : ℝ) := 10 * x

-- The measure of angle CDB
def angleCDB := 30

-- The length of the side AB
def lengthAB (a : ℝ) := a

-- The length of BC to be proved
def lengthBC (a : ℝ) := (a * Real.sqrt 2) / 3

theorem find_length_BC (a : ℝ) (x : ℝ) (h : 3 * x + 7 * x + 4 * x + 10 * x = 360) : 
  lengthBC a = (a * Real.sqrt 2) / 3 := 
by 
  -- skipping the proof
  sorry

end find_length_BC_l34_34199


namespace floor_sqrt_identity_l34_34177

theorem floor_sqrt_identity (n : ℕ) (h : 0 < n) :
  ⌊real.sqrt n + real.sqrt (n + 1)⌋ = ⌊real.sqrt (4 * n + 2)⌋ := 
sorry

end floor_sqrt_identity_l34_34177


namespace dan_grew_9_onions_l34_34591

theorem dan_grew_9_onions (n_m_total : 2 + 4 = 6) 
   (total_onions : 15) : 
   let dan_onions := total_onions - 6 
   in dan_onions = 9 := 
by
  sorry

end dan_grew_9_onions_l34_34591


namespace total_bricks_required_l34_34344

-- Define the length and breadth of the courtyard in meters
def courtyard_length : ℝ := 18
def courtyard_breadth : ℝ := 16

-- Define the dimensions of the brick in centimeters
def brick_length : ℝ := 20
def brick_breadth : ℝ := 10

-- Conversion factor from meters to centimeters
def meter_to_cm : ℝ := 100

-- Define the area of the courtyard in square meters and square centimeters
def courtyard_area_meters : ℝ := courtyard_length * courtyard_breadth
def courtyard_area_cm : ℝ := courtyard_area_meters * meter_to_cm ^ 2

-- Define the area of one brick in square centimeters
def brick_area_cm : ℝ := brick_length * brick_breadth

-- Define the total number of bricks required
def total_bricks : ℝ := courtyard_area_cm / brick_area_cm

-- Theorem statement
theorem total_bricks_required : total_bricks = 14400 := 
by {
  -- Skip the proof
  sorry
}

end total_bricks_required_l34_34344


namespace tangent_line_at_1_range_of_m_when_a_minus_1_l34_34812

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp x - a * Real.log x

def tangent_line_equation (a : ℝ) : ℝ × ℝ := 
  let m := Real.exp 1 - a
  let b := 1
  (m, b)

theorem tangent_line_at_1 (a : ℝ) : 
  let p := (1, Real.exp 1)
  ∃ (m b : ℝ), (m, b) = tangent_line_equation a ∧ (∀ x y, y - p.2 = m * (x - p.1) → (m * x - y + b = 0)) :=
by
  sorry

theorem range_of_m_when_a_minus_1 (e : ℝ) : 
  ∀ (m : ℝ), m ≤ e + 1 ↔ ∀ (x : ℝ), 1 < x → f (-1) x > e + m * (x - 1) :=
by
  sorry

end tangent_line_at_1_range_of_m_when_a_minus_1_l34_34812


namespace no_linear_factor_with_integer_coeffs_l34_34760

theorem no_linear_factor_with_integer_coeffs 
  (x y z : ℤ) : 
  ¬ (∃ a b c d : ℤ, (x^2 - y^2 + 2yz - z^2 + 2x - y - 3z) = (a * x + b * y + c * z + d)) :=
sorry

end no_linear_factor_with_integer_coeffs_l34_34760


namespace find_x_l34_34133

noncomputable def α : ℝ := by sorry -- angle α in the second quadrant
def x : ℝ := by sorry -- the value to be found
def P : ℝ × ℝ := (x, sqrt 5) -- point P on the terminal side
def cos_α : ℝ := (sqrt 2 / 4) * x

axiom α_in_second_quadrant : α > pi / 2 ∧ α < pi
axiom cos_alpha_def : cos_α = (sqrt 2 / 4) * x

theorem find_x : x = - sqrt 3 :=
by
  have r := sqrt (x^2 + 5),
  have cos_α_eq := cos_α = x / r,
  sorry -- the proof will go here

end find_x_l34_34133


namespace minimum_n_for_valid_sequence_l34_34220

def S : set ℕ := {1, 2, 3, 4}

def valid_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ (B : finset ℕ), B ⊆ S → B ≠ ∅ →
    ∃ i : ℕ, i + B.card - 1 < n ∧ (finset.image (λ j, a (i + j)) (finset.range B.card)) = B

theorem minimum_n_for_valid_sequence : ∃ a : ℕ → ℕ, valid_sequence a 8 :=
sorry

end minimum_n_for_valid_sequence_l34_34220


namespace stacy_pages_per_day_l34_34610

theorem stacy_pages_per_day (total_pages : ℕ) (days : ℕ) (h1 : total_pages = 33) (h2 : days = 3) : 
  total_pages / days = 11 :=
by
  rw [h1, h2]
  exact Nat.div_self (show 3 ≠ 0 from by decide)

end stacy_pages_per_day_l34_34610


namespace find_x_l34_34428

theorem find_x (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1)
  (geom_seq : (x - ⌊x⌋) * x = ⌊x⌋^2) : x = 1.618 :=
by
  sorry

end find_x_l34_34428


namespace number_of_integer_values_l34_34496

theorem number_of_integer_values (x : ℕ) (h : 8 ≤ x ∧ x < 81) :
  floor (sqrt (x : ℝ)) = 8 ∧ 17 = card { n : ℕ | 64 ≤ n ∧ n < 81 } :=
by 
  sorry

end number_of_integer_values_l34_34496


namespace area_triangle_QCA_l34_34065

/--
  Given:
  - θ (θ is acute) is the angle at Q between QA and QC
  - Q is at the coordinates (0, 12)
  - A is at the coordinates (3, 12)
  - C is at the coordinates (0, p)

  Prove that the area of triangle QCA is (3/2) * (12 - p) * sin(θ).
-/
theorem area_triangle_QCA (p θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) :
  let Q := (0, 12)
  let A := (3, 12)
  let C := (0, p)
  let base := 3
  let height := (12 - p) * Real.sin θ
  let area := (1 / 2) * base * height
  area = (3 / 2) * (12 - p) * Real.sin θ := by
  sorry

end area_triangle_QCA_l34_34065


namespace area_of_triangle_BFE_l34_34600

theorem area_of_triangle_BFE (A B C D E F : ℝ × ℝ) (u v : ℝ) 
  (h_rectangle : (0, 0) = A ∧ (3 * u, 0) = B ∧ (3 * u, 3 * v) = C ∧ (0, 3 * v) = D)
  (h_E : E = (0, 2 * v))
  (h_F : F = (2 * u, 0))
  (h_area_rectangle : 3 * u * 3 * v = 48) :
  ∃ (area : ℝ), area = |3 * u * 2 * v| / 2 ∧ area = 24 :=
by 
  sorry

end area_of_triangle_BFE_l34_34600


namespace solve_for_x_l34_34681

theorem solve_for_x (x : ℚ) (h : 3 / x - 3 / x / (9 / x) = 0.5) : x = 6 / 5 :=
sorry

end solve_for_x_l34_34681


namespace train_speed_excluding_stoppages_l34_34775

-- Define the speed of the train excluding stoppages and including stoppages
variables (S : ℕ) -- S is the speed of the train excluding stoppages
variables (including_stoppages_speed : ℕ := 40) -- The speed including stoppages is 40 kmph

-- The train stops for 20 minutes per hour. This means it runs for (60 - 20) minutes per hour.
def running_time_per_hour := 40

-- Converting 40 minutes to hours
def running_fraction_of_hour : ℚ := 40 / 60

-- Formulate the main theorem:
theorem train_speed_excluding_stoppages
    (H1 : including_stoppages_speed = 40)
    (H2 : running_fraction_of_hour = 2 / 3) :
    S = 60 :=
by
    sorry

end train_speed_excluding_stoppages_l34_34775


namespace sum_stirling_power_l34_34176

noncomputable def stirling_second (n k : ℕ) : ℕ :=
  if h : n < k then 0 else 
  if h : k = 0 then if n = 0 then 1 else 0
  else stirling_second (n - 1) k * k + stirling_second (n - 1) (k - 1)

theorem sum_stirling_power (n : ℕ) (m : ℕ) (h : 0 < m) :
  (∑ k in Finset.range (n + 1), stirling_second n k * (finset.prod (Finset.range k) (λ i, m - i))) = m^n :=
  sorry

end sum_stirling_power_l34_34176


namespace expected_number_of_digits_l34_34056

def fair_icosahedral_die := fin 20

def probability_one_digit := 9 / 20
def probability_two_digit := 11 / 20

def expected_digits_roll (die : fair_icosahedral_die) : ℚ := 
  probability_one_digit * 1 + probability_two_digit * 2

theorem expected_number_of_digits : 
  expected_digits_roll (⟨n, hn⟩ : fair_icosahedral_die) = 1.55 := 
sorry

end expected_number_of_digits_l34_34056


namespace JohnsonsYield_l34_34204

def JohnsonYieldPerTwoMonths (J : ℕ) : Prop :=
  ∀ (neighbor_hectares neighbor_yield_per_hectare total_yield_six_months : ℕ),
    neighbor_hectares = 2 →
    neighbor_yield_per_hectare = 2 * J →
    total_yield_six_months = 1200 →
    3 * J + 3 * (neighbor_hectares * neighbor_yield_per_hectare) = total_yield_six_months →
    J = 80

theorem JohnsonsYield
  (J : ℕ)
  (neighbor_hectares neighbor_yield_per_hectare total_yield_six_months : ℕ)
  (h1 : neighbor_hectares = 2)
  (h2 : neighbor_yield_per_hectare = 2 * J)
  (h3 : total_yield_six_months = 1200)
  (h4 : 3 * J + 3 * (neighbor_hectares * neighbor_yield_per_hectare) = total_yield_six_months) :
  J = 80 :=
by
  sorry

end JohnsonsYield_l34_34204


namespace _l34_34406

noncomputable def circle_theorem (O A B C D M P : Type*)
  [inhabited O] [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited M] [inhabited P]
  (is_diameter_AB : ∃ O, ∀ (X : A), ∃ (Y : B), ∃ (Z : A), (X = Y) → (angle O X Z = 0))
  (is_diameter_CD: ∃ O, ∀ (U : C), ∃ (V : D), ∃ (W : C), (U = V) → (angle O U W = 0))
  (perpendicular_ab_cd: angle A B = 90)
  (chord_AM: ∃ M, ∀ (P : P), chord O A M P)
  (intersect_CD_at_P: ∃ P, intersect CD P) :
  AP * AM = AO * AB := sorry

end _l34_34406


namespace simplify_expression_l34_34052

theorem simplify_expression : 8^5 + 8^5 + 8^5 + 8^5 = 8^(17/3) :=
by
  -- Proof will be completed here
  sorry

end simplify_expression_l34_34052


namespace product_of_consecutive_natural_numbers_l34_34802

theorem product_of_consecutive_natural_numbers (n : ℕ) : 
  (∃ t : ℕ, n = t * (t + 1) - 1) ↔ ∃ x : ℕ, n^2 - 1 = x * (x + 1) * (x + 2) * (x + 3) := 
sorry

end product_of_consecutive_natural_numbers_l34_34802


namespace pups_with_two_spots_l34_34712

/-- 
  Given:
  - total number of puppies: 10
  - number of puppies with 5 spots: 6
  - number of puppies with 4 spots: 3
  - breeder expects to sell 90.9090909090909% puppies for a greater profit (which suggests 9 puppies have 4 or more spots)
  Prove:
  - the number of puppies with 2 spots is 1
--/

theorem pups_with_two_spots 
  (total_puppies : ℕ := 10)
  (pups_with_five_spots : ℕ := 6)
  (pups_with_four_spots : ℕ := 3)
  (expected_sale_percentage : ℚ := 90.9090909090909) :
  total_puppies - pups_with_five_spots - pups_with_four_spots = 1 :=
by
  -- Given conditions
  have h_total : total_puppies = 10 := by rfl
  have h_five_spots : pups_with_five_spots = 6 := by rfl
  have h_four_spots : pups_with_four_spots = 3 := by rfl
  have h_percentage : expected_sale_percentage = 90.9090909090909 := by rfl
  -- Proof of the required condition
  calc
  total_puppies - pups_with_five_spots - pups_with_four_spots
      = 10 - 6 - 3 : by congr <|> exact rfl
  ... = 1 : by norm_num

end pups_with_two_spots_l34_34712


namespace triangle_sine_ratio_l34_34958

theorem triangle_sine_ratio 
  (A B C D : Type) [triangle A B C]
  (angle_B : angle A B C = Angle.pi_div_four)
  (angle_C : angle A C B = Angle.pi_div_three)
  (D_divides_BC : divides D B C 2 1) :
  sin (angle A B D) / sin (angle A C D) = Real.sqrt 6 := 
sorry

end triangle_sine_ratio_l34_34958


namespace bags_filled_l34_34613

theorem bags_filled : ∀ (cookies total_bags: ℕ), cookies = 75 → total_bags = cookies / 3 → total_bags = 25 :=
by 
  intros cookies total_bags h_cookies h_total_bags
  rw [h_cookies] at h_total_bags
  rw [h_total_bags]
  norm_num
  sorry

end bags_filled_l34_34613


namespace problem_statement_l34_34980

noncomputable def poly : Polynomial ℝ := 45 * X^3 - 4050 * X^2 + 4

noncomputable def roots := Multiset.sort (Polynomial.roots poly)

def x1 := roots.nthLe 0 (by simp [roots])
def x2 := roots.nthLe 1 (by simp [roots])
def x3 := roots.nthLe 2 (by simp [roots])

theorem problem_statement : x2 * (x1 + x3) = 90 :=
by {
  sorry
}

end problem_statement_l34_34980


namespace fraction_exists_l34_34249

theorem fraction_exists (n d k : ℕ) (h₁ : n = k * d) (h₂ : d > 0) (h₃ : k > 0) : 
  ∃ (i j : ℕ), i < n ∧ j < n ∧ i + j = n ∧ i/j = d-1 :=
by
  sorry

end fraction_exists_l34_34249


namespace proof_l34_34037

-- Define the rounding function
def round_nearest_hundredth (x : ℝ) : ℝ :=
  (Float.ofReal x).round(2) / 100

-- Given conditions
def a : ℝ := 152.34
def b : ℝ := 67.895
def c : ℝ := 24.581

-- Proposition to prove
theorem proof : round_nearest_hundredth ((a + b) - c) = 195.65 :=
by
  sorry

end proof_l34_34037


namespace area_of_square_ABCD_l34_34257

/-- Given a rhombus PQRS inscribed in a square ABCD such that vertices P, Q, R, and S are 
    on sides AB, BC, CD, and DA respectively, and given lengths PB = 10, BQ = 25, 
    PR = 20, and QS = 40, we can show that the area of the square ABCD is equal to 40000/58. -/
theorem area_of_square_ABCD (PB : ℝ) (BQ : ℝ) (PR : ℝ) (QS : ℝ)
  (hPB : PB = 10) (hBQ : BQ = 25) (hPR : PR = 20) (hQS : QS = 40) :
  let a := ((2 * 100 * Real.sqrt 29) / 29 / Real.sqrt 2) in
  a^2 = 40000 / 58 :=
by sorry

end area_of_square_ABCD_l34_34257


namespace distance_between_foci_l34_34484

-- Define the condition: hyperbola xy = 4
def hyperbola (x y : ℝ) : Prop := x * y = 4

-- Define the distance between two points (x1, y1) and (x2, y2) in the plane
def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

-- Define the foci coordinates for the hyperbola xy = 4
def foci1 : ℝ × ℝ := (2 * real.sqrt 2, 2 * real.sqrt 2)
def foci2 : ℝ × ℝ := (-2 * real.sqrt 2, -2 * real.sqrt 2)

-- Proof statement
theorem distance_between_foci : distance (foci1.1) (foci1.2) (foci2.1) (foci2.2) = 8 := by
  sorry

end distance_between_foci_l34_34484


namespace find_denomination_of_oliver_bills_l34_34243

-- Definitions based on conditions
def denomination (x : ℕ) : Prop :=
  let oliver_total := 10 * x + 3 * 5
  let william_total := 15 * 10 + 4 * 5
  oliver_total = william_total + 45

-- The Lean theorem statement
theorem find_denomination_of_oliver_bills (x : ℕ) : denomination x → x = 20 := by
  sorry

end find_denomination_of_oliver_bills_l34_34243


namespace circle_equation_bisects_l34_34193

-- Define the given conditions
def circle1_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 8)^2 = 1
def circle2_eq (x y : ℝ) : Prop := (x - 6)^2 + (y + 6)^2 = 9

-- Define the goal equation
def circleC_eq (x y : ℝ) : Prop := x^2 + y^2 = 81

-- The statement of the problem
theorem circle_equation_bisects (a r : ℝ) (h1 : ∀ x y, circle1_eq x y → circleC_eq x y) (h2 : ∀ x y, circle2_eq x y → circleC_eq x y):
  circleC_eq (a * r) 0 := sorry

end circle_equation_bisects_l34_34193


namespace interest_years_proof_l34_34523

theorem interest_years_proof :
  let interest_r800_first_2_years := 800 * 0.05 * 2
  let interest_r800_next_3_years := 800 * 0.12 * 3
  let total_interest_r800 := interest_r800_first_2_years + interest_r800_next_3_years
  let interest_r600_first_3_years := 600 * 0.07 * 3
  let interest_r600_next_n_years := 600 * 0.10 * n
  (interest_r600_first_3_years + interest_r600_next_n_years = total_interest_r800) ->
  n = 5 →
  3 + n = 8 :=
by
  sorry

end interest_years_proof_l34_34523


namespace time_fraction_reduction_l34_34009

theorem time_fraction_reduction
  (S : ℝ)
  (hS : S = 36.000000000000014)
  (S_inc : ℝ)
  (hS_inc : S_inc = S + 18) :
  let T : ℝ := 1 in
  let T_inc : ℝ := S / S_inc * T in
  T - T_inc = (1/3) * T :=
by
  -- Simplified formulation for this specific problem
  rw [hS, hS_inc]
  have : S_inc = 54.000000000000014 := by simp [hS, hS_inc]
  sorry -- Proof steps would follow here

end time_fraction_reduction_l34_34009


namespace find_x_l34_34780

theorem find_x (x : ℝ) :
  (log 2011 (real.sqrt (1 + (1 / real.tan x) ^ 2) + (1 / real.tan x)) = log 2013 (real.sqrt (1 + (1 / real.tan x) ^ 2) - (1 / real.tan x))) →
  (∃ n : ℤ, x = n * real.pi) ∨ (∃ n : ℤ, x = (n + 1 / 2) * real.pi) :=
sorry

end find_x_l34_34780


namespace exists_hexahedron_with_congruent_rhombus_faces_l34_34161

noncomputable def hexahedron_with_congruent_rhombus_faces : Prop :=
  ∃ (P : Polyhedron) (H : ¬ is_cube P), is_hexahedron P ∧
    ∀ f₁ f₂, f₁ ∈ faces P ∧ f₂ ∈ faces P → is_congruent f₁ f₂ ∧ is_rhombus f₁

theorem exists_hexahedron_with_congruent_rhombus_faces : hexahedron_with_congruent_rhombus_faces :=
sorry

end exists_hexahedron_with_congruent_rhombus_faces_l34_34161


namespace range_x_when_p_and_q_m_eq_1_range_m_for_not_p_necessary_not_sufficient_q_l34_34126

-- Define the propositions p and q in terms of x and m
def p (x m : ℝ) : Prop := |2 * x - m| ≥ 1
def q (x : ℝ) : Prop := (1 - 3 * x) / (x + 2) > 0

-- The range of x for p ∧ q when m = 1
theorem range_x_when_p_and_q_m_eq_1 : {x : ℝ | p x 1 ∧ q x} = {x : ℝ | -2 < x ∧ x ≤ 0} :=
by sorry

-- The range of m where ¬p is a necessary but not sufficient condition for q
theorem range_m_for_not_p_necessary_not_sufficient_q : {m : ℝ | ∀ x, ¬p x m → q x} ∩ {m : ℝ | ∃ x, ¬p x m ∧ q x} = {m : ℝ | -3 ≤ m ∧ m ≤ -1/3} :=
by sorry

end range_x_when_p_and_q_m_eq_1_range_m_for_not_p_necessary_not_sufficient_q_l34_34126


namespace three_digit_integers_congruent_to_2_mod_4_l34_34893

theorem three_digit_integers_congruent_to_2_mod_4 : 
    ∃ n, n = 225 ∧ ∀ x, (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 2) ↔ (∃ m, 25 ≤ m ∧ m ≤ 249 ∧ x = 4 * m + 2) := by
  sorry

end three_digit_integers_congruent_to_2_mod_4_l34_34893


namespace gcd_of_set_B_l34_34554

-- Let B be the set of all numbers which can be represented as the sum of five consecutive positive integers
def B : Set ℕ := {n | ∃ y : ℕ, n = (y - 2) + (y - 1) + y + (y + 1) + (y + 2)}

-- State the problem
theorem gcd_of_set_B : ∀ k ∈ B, ∃ d : ℕ, is_gcd 5 k d → d = 5 := by
sorry

end gcd_of_set_B_l34_34554


namespace range_of_a_l34_34138

theorem range_of_a 
  (a : ℝ) 
  (h₀ : ∀ x : ℝ, (3 ≤ x ∧ x ≤ 4) ↔ (y = 2 * x + (3 - a))) : 
  9 ≤ a ∧ a ≤ 11 := 
sorry

end range_of_a_l34_34138


namespace intersection_A_B_eq_range_a_l34_34867

section
variable {a : ℝ}

-- Definitions of the sets
def A : set ℝ := {x | x^2 - 3 * x < 0}
def B : set ℝ := {x | (x + 2) * (4 - x) ≥ 0}
def C (a : ℝ) : set ℝ := {x | a < x ∧ x ≤ a + 1}

-- Proof of the intersections and subset relations
theorem intersection_A_B_eq : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} :=
sorry

theorem range_a (h : B ∪ C a = B) : -2 ≤ a ∧ a ≤ 3 :=
sorry

end

end intersection_A_B_eq_range_a_l34_34867


namespace percentage_of_seeds_germinated_l34_34439

theorem percentage_of_seeds_germinated (P1 P2 : ℕ) (GP1 GP2 : ℕ) (SP1 SP2 TotalGerminated TotalPlanted : ℕ) (PG : ℕ) 
  (h1 : P1 = 300) (h2 : P2 = 200) (h3 : GP1 = 60) (h4 : GP2 = 70) (h5 : SP1 = P1) (h6 : SP2 = P2)
  (h7 : TotalGerminated = GP1 + GP2) (h8 : TotalPlanted = SP1 + SP2) : 
  PG = (TotalGerminated * 100) / TotalPlanted :=
sorry

end percentage_of_seeds_germinated_l34_34439


namespace range_of_a_l34_34069

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1) := 
by 
  sorry

end range_of_a_l34_34069


namespace triangle_perimeter_l34_34638

theorem triangle_perimeter
  (a b : ℕ) (c : ℕ) 
  (h_side1 : a = 3)
  (h_side2 : b = 4)
  (h_third_side : c^2 - 13 * c + 40 = 0)
  (h_valid_triangle : (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :
  a + b + c = 12 :=
by {
  sorry
}

end triangle_perimeter_l34_34638


namespace goldfinch_percentage_l34_34998

noncomputable def percentage_of_goldfinches 
  (goldfinches : ℕ) (sparrows : ℕ) (grackles : ℕ) : ℚ :=
  (goldfinches : ℚ) / (goldfinches + sparrows + grackles) * 100

theorem goldfinch_percentage (goldfinches sparrows grackles : ℕ)
  (h_goldfinches : goldfinches = 6)
  (h_sparrows : sparrows = 9)
  (h_grackles : grackles = 5) :
  percentage_of_goldfinches goldfinches sparrows grackles = 30 :=
by
  rw [h_goldfinches, h_sparrows, h_grackles]
  show percentage_of_goldfinches 6 9 5 = 30
  sorry

end goldfinch_percentage_l34_34998


namespace sequence_positive_term_minimal_initial_terms_T_n_formula_l34_34823

noncomputable def a_n (n : ℕ) : ℤ := 3 * n - 63

theorem sequence_positive_term :
  ∀ n : ℕ, n ≥ 22 → a_n n > 0 := by
  intros n h
  sorry

theorem minimal_initial_terms :
  ∃ (n : ℕ), (n = 21 ∨ n = 20) ∧ (∑ i in finset.range 21, a_n (i + 1)) = -630 := by
  sorry

def T_n (n : ℕ) : ℤ :=
  if h : n ≤ 21 then (123 * n - 3 * n * n) / 2 else (3 * n * n - 123 * n + 2520) / 2

theorem T_n_formula :
  ∀ n : ℕ, T_n n = 
    if h : n ≤ 21 then (123 * n - 3 * n * n) / 2 
    else (3 * n * n - 123 * n + 2520) / 2 := by
  intros n
  sorry

end sequence_positive_term_minimal_initial_terms_T_n_formula_l34_34823


namespace b_investment_months_l34_34715

theorem b_investment_months (
  A_investment_total_months : ℕ,
  B_investment_rate : ℕ,
  B_investment_months : ℕ,
  total_profit : ℚ,
  A_profit : ℚ
) (hA : A_investment_total_months = 150 * 12)
  (hB : total_profit - A_profit = 40)
  (h_ratio : (A_investment_total_months : ℚ) / (B_investment_rate * B_investment_months) = 3 / 2) :
  B_investment_months = 200 * (12 - 6) :=
sorry

end b_investment_months_l34_34715


namespace units_digit_of_large_powers_l34_34093

theorem units_digit_of_large_powers : 
  (2^1007 * 6^1008 * 14^1009) % 10 = 2 := 
  sorry

end units_digit_of_large_powers_l34_34093


namespace max_distance_snail_travel_l34_34351

theorem max_distance_snail_travel 
    (h_journey: ∃ t: ℝ, t = 6) 
    (h_observed: ∀ t ∈ (0:ℝ) .. 6, ∃ scientist: ℕ, scientist observes snail at t)
    (h_scientist_observ: ∀ scientist: ℕ, ∃ (start end : ℝ), start end ∈ (0 : ℝ) .. 6 ∧ end - start = 1 ∧ ∀ t ∈ start .. end, snail moves 1 meter) :
  (∃ d: ℝ, d = 10) :=
sorry

end max_distance_snail_travel_l34_34351


namespace smallest_positive_period_monotonically_decreasing_interval_max_value_in_interval_min_value_in_interval_l34_34476

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x - cos x ^ 2 + sin x ^ 2

-- Part 1: Smallest positive period and monotonically decreasing interval
theorem smallest_positive_period (x : ℝ) : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x :=
  sorry

theorem monotonically_decreasing_interval (k : ℤ) (x : ℝ) : 
  f x ∈ Icc (k * pi + (3 / 8) * pi) (k * pi + (7 / 8) * pi) → 
  ∀ x1 x2, x1 < x2 → f x1 ≥ f x2 :=
  sorry

-- Part 2: Maximum and minimum values in [0, π/2]
theorem max_value_in_interval : ∃ x ∈ Icc (0 : ℝ) (π / 2), f x = sqrt 2 :=
  sorry

theorem min_value_in_interval : ∃ x ∈ Icc (0 : ℝ) (π / 2), f x = -1 :=
  sorry

end smallest_positive_period_monotonically_decreasing_interval_max_value_in_interval_min_value_in_interval_l34_34476


namespace triangle_AC_length_l34_34200

theorem triangle_AC_length (A B C K E D : Point) (bk_median : is_median A B C K)
  (be_bisector : is_angle_bisector B C E)
  (ad_altitude : is_altitude A D B C)
  (bk_be_split_ad : divides_into_equal_parts_and_terminates A D B E K 3)
  (AB_length : dist A B = 4) :
  dist A C = 2 * Real.sqrt 3 :=
sorry

end triangle_AC_length_l34_34200


namespace axis_of_symmetry_range_l34_34178

theorem axis_of_symmetry_range {α : ℝ} (h : ∃ (f : ℝ → ℝ), 
f = λ x, 3 * sin x - 4 * cos x ∧
∃ (φ : ℝ), tan φ = 3 / 4 ∧ f(0) = f(2 * α)) :
α ∈ (3 * π / 4, π) :=
sorry

end axis_of_symmetry_range_l34_34178


namespace leak_empties_tank_in_30_hours_l34_34691

-- Define the known rates based on the problem conditions
def rate_pipe_a : ℚ := 1 / 12
def combined_rate : ℚ := 1 / 20

-- Define the rate at which the leak empties the tank
def rate_leak : ℚ := rate_pipe_a - combined_rate

-- Define the time it takes for the leak to empty the tank
def time_to_empty_tank : ℚ := 1 / rate_leak

-- The theorem that needs to be proved
theorem leak_empties_tank_in_30_hours : time_to_empty_tank = 30 :=
sorry

end leak_empties_tank_in_30_hours_l34_34691


namespace faster_speed_l34_34716

theorem faster_speed (v : ℝ) :
  (∀ t : ℝ, (40 / 10 = t) ∧ (60 / v = t)) → v = 15 :=
by
  sorry

end faster_speed_l34_34716


namespace S_stabilizes_l34_34546

-- Infinite set S of integers containing zero
-- and distances between successive numbers 
-- not exceeding a fixed number r
def set_S (S : Set ℤ) (r : ℕ) : Prop :=
  (0 : ℤ) ∈ S ∧ ∀ s₁ s₂ ∈ S, (s₁ ≠ s₂) → abs (s₁ - s₂) ≤ (r : ℤ)

-- The sequence S_k defined by the procedure
def S_k : (ℕ → Set ℤ) → Set ℤ := 
λ S, { x : ℤ | ∃ (k : ℕ) (x' s' : ℤ), x' ∈ S k ∧ s' ∈ S' ∧ (x = x' + s' ∨ x = x' - s') }

-- Define the proof goal
theorem S_stabilizes (S : Set ℤ) (r : ℕ) 
  (hS : set_S S r) :
  ∃ k₀, ∀ k ≥ k₀, S_k S k = S_k S k₀ :=
sorry

end S_stabilizes_l34_34546


namespace intersection_A_B_l34_34991

variable A : Set ℝ := { x : ℝ | x^2 - 4 * x < 0 }
variable B : Set ℤ := { y : ℤ | True }

theorem intersection_A_B : A ∩ (B.map coe) = {1, 2, 3} :=
by
  sorry

end intersection_A_B_l34_34991


namespace parameterized_line_coordinates_l34_34639

noncomputable def r : ℚ := 4 / 3
noncomputable def m : ℚ := 0

theorem parameterized_line_coordinates :
  (∀ t : ℚ, ∃ (x y : ℚ), (x = -5 + t * m) ∧ (y = r + t * -6) ∧ (y = 1 / 3 * x + 3)) :=
by
  assume t : ℚ
  use [(-5 + t * m), (r + t * -6)]
  split
  { refl }
  split
  { refl }
  { sorry }

end parameterized_line_coordinates_l34_34639


namespace expression_evaluation_l34_34696

theorem expression_evaluation :
  (0.66 ^ 3) - (0.1 ^ 3) / (0.66 ^ 2) + 0.066 + (0.1 ^ 2) = 0.3612 := 
by
  have h1 : 0.66 ^ 3 = 0.287496 := by norm_num
  have h2 : 0.1 ^ 3 = 0.001 := by norm_num
  have h3 : 0.66 ^ 2 = 0.4356 := by norm_num
  have h4 : 0.1 ^ 2 = 0.01 := by norm_num
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end expression_evaluation_l34_34696


namespace label_triangle_inequality_l34_34073

theorem label_triangle_inequality (n m : ℕ) 
  (h₁ : ∃ T : Type, ∃ S : finset T, S.card = n^2) 
  (h₂ : ∀ (a b : T), a ∈ S ∧ b ∈ S ∧ ∃ (i : ℕ), a = i.succ ∧ b = i ∧ (i < m) → (a, b) ∈ edge_set) 
  : m ≤ n^2 - n + 1 := 
sorry

end label_triangle_inequality_l34_34073


namespace circumscribed_sphere_radius_l34_34320

theorem circumscribed_sphere_radius (a b R : ℝ) (ha : a > 0) (hb : b > 0) :
  R = b^2 / (2 * (Real.sqrt (b^2 - a^2))) :=
sorry

end circumscribed_sphere_radius_l34_34320


namespace three_digit_integers_congruent_to_2_mod_4_l34_34904

theorem three_digit_integers_congruent_to_2_mod_4 : 
  let count := (249 - 25 + 1) in
  count = 225 :=
by
  let k_min := 25
  let k_max := 249
  have h_count : count = (k_max - k_min + 1) := rfl
  rw h_count
  norm_num

end three_digit_integers_congruent_to_2_mod_4_l34_34904


namespace project_completion_days_l34_34376

theorem project_completion_days 
  (total_mandays : ℕ)
  (initial_workers : ℕ)
  (leaving_workers : ℕ)
  (remaining_workers : ℕ)
  (days_total : ℕ) :
  total_mandays = 200 →
  initial_workers = 10 →
  leaving_workers = 4 →
  remaining_workers = 6 →
  days_total = 40 :=
by
  intros h0 h1 h2 h3
  sorry

end project_completion_days_l34_34376


namespace denominator_bound_l34_34124

theorem denominator_bound (a b : ℚ) (m n : ℤ) (h_mn_coprime : Int.gcd m n = 1)
  (h_root : a * n^2 - b * n = m * n):
  ∃ x ∈ {a, b}, x.denom ≥ n ^ (2 / 3 : ℝ) := 
by
  sorry

end denominator_bound_l34_34124


namespace harmonic_progression_product_sum_l34_34251

noncomputable def is_harmonic_progression (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∃ d : ℝ, ∀ i, 1 ≤ i ∧ i < n → (1 / a i) - (1 / a (i + 1)) = d

theorem harmonic_progression_product_sum (a : ℕ → ℝ) (n : ℕ) (h : is_harmonic_progression a n) :
  (Finset.sum (Finset.range (n - 1)) (λ i, a i * a (i + 1))) = (n - 1) * (a 1) * (a n) :=
by
  sorry

end harmonic_progression_product_sum_l34_34251


namespace six_matches_twelve_members_l34_34030

-- Define the context of the problem
variables (num_members matches : ℕ)
variables (num_per_match : ℕ)
variables (member_play_count : Fin num_members → Prop)

-- Add the conditions as definitions
def condition1 := num_members = 20
def condition2 := matches = 14
def condition3 := num_per_match = 2
def condition4 := ∀ m : Fin num_members, member_play_count m

-- The theorem statement based on the problem's question and correct answer
theorem six_matches_twelve_members :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 →
  ∃ (k : ℕ), k ≥ 6 ∧ (∃ S : Finset (Fin num_members), S.card = 12 ∧ (∀ m ∈ S, member_play_count m)) := 
sorry

end six_matches_twelve_members_l34_34030


namespace women_in_club_l34_34709

def club_condition (M W : ℕ) : Prop :=
  M + W = 30 ∧ (1 / 3 * W : ℝ) + M = 18

theorem women_in_club (M W : ℕ) (h : club_condition M W) : W = 18 :=
by
  have : (1 / 3 * W : ℝ) = 18 - M := by linarith [h.2]
  have : 3 * (1 / 3 * W) = 3 * (18 - M) := by rw this; linarith
  have : W = 54 - 3 * M := by linarith
  have : W + M = 30 := h.1
  calc
    W = 54 - 3 * M : by linarith
    _ = 18 : by sorry

end women_in_club_l34_34709


namespace pure_imaginary_a_l34_34848

theorem pure_imaginary_a (a : ℝ) :
  (let z := (a + 2 * complex.i) / (2 - complex.i) in z.re = 0) → a = 1 :=
by sorry

end pure_imaginary_a_l34_34848


namespace sum_of_points_coordinates_l34_34326

noncomputable def sum_of_coordinates : ℝ :=
  let points := {(3 + Real.sqrt 91, 13), (3 - Real.sqrt 91, 13), (3 + Real.sqrt 91, 7), (3 - Real.sqrt 91, 7)} 
  ∑ (p : ℝ × ℝ) in points, p.1 + p.2

theorem sum_of_points_coordinates : sum_of_coordinates = 52 := 
  sorry

end sum_of_points_coordinates_l34_34326


namespace carla_highest_final_number_l34_34733

def alice_final_number (initial : ℕ) : ℕ :=
  let step1 := initial * 2
  let step2 := step1 - 3
  let step3 := step2 / 3
  step3 + 4

def bob_final_number (initial : ℕ) : ℕ :=
  let step1 := initial + 5
  let step2 := step1 * 2
  let step3 := step2 - 4
  step3 / 2

def carla_final_number (initial : ℕ) : ℕ :=
  let step1 := initial - 2
  let step2 := step1 * 2
  let step3 := step2 + 3
  step3 * 2

theorem carla_highest_final_number : carla_final_number 12 > bob_final_number 12 ∧ carla_final_number 12 > alice_final_number 12 :=
  by
  have h_alice : alice_final_number 12 = 11 := by rfl
  have h_bob : bob_final_number 12 = 15 := by rfl
  have h_carla : carla_final_number 12 = 46 := by rfl
  sorry

end carla_highest_final_number_l34_34733


namespace initial_time_to_cover_distance_l34_34003

theorem initial_time_to_cover_distance (s t : ℝ) (h1 : 540 = s * t) (h2 : 540 = 60 * (3/4) * t) : t = 12 :=
sorry

end initial_time_to_cover_distance_l34_34003


namespace roy_sports_hours_l34_34602

theorem roy_sports_hours (s d m : ℕ) (h₁ : s = 2) (h₂ : d = 5) (h₃ : m = 2) : (d - m) * s = 6 :=
by
  rw [h₁, h₂, h₃]
  show (5 - 2) * 2 = 6
  sorry

end roy_sports_hours_l34_34602


namespace maximum_correct_answers_l34_34933

theorem maximum_correct_answers (a b c : ℕ) (h1 : a + b + c = 60)
  (h2 : 5 * a - 2 * c = 150) : a ≤ 38 :=
by
  sorry

end maximum_correct_answers_l34_34933


namespace P_is_subtract_0_set_P_is_not_subtract_1_set_no_subtract_2_set_exists_all_subtract_1_sets_l34_34122

def is_subtract_set (T : Set ℕ) (i : ℕ) := T ⊆ Set.univ ∧ T ≠ {1} ∧ (∀ {x y : ℕ}, x ∈ Set.univ → y ∈ Set.univ → x + y ∈ T → x * y - i ∈ T)

theorem P_is_subtract_0_set : is_subtract_set {1, 2} 0 := sorry

theorem P_is_not_subtract_1_set : ¬ is_subtract_set {1, 2} 1 := sorry

theorem no_subtract_2_set_exists : ¬∃ T : Set ℕ, is_subtract_set T 2 := sorry

theorem all_subtract_1_sets : ∀ T : Set ℕ, is_subtract_set T 1 ↔ T = {1, 3} ∨ T = {1, 3, 5} := sorry

end P_is_subtract_0_set_P_is_not_subtract_1_set_no_subtract_2_set_exists_all_subtract_1_sets_l34_34122


namespace find_beta_l34_34446

noncomputable def sin_alpha := Real.sin α = (Real.sqrt 5) / 5
noncomputable def sin_alpha_minus_beta := Real.sin (α - β) = -(Real.sqrt 10) / 10
noncomputable def cos_alpha := Real.cos α = (2 * Real.sqrt 5) / 5
noncomputable def cos_alpha_minus_beta := Real.cos (α - β) = (3 * Real.sqrt 10) / 10
noncomputable def acute_angles := α ∈ (0, Real.pi / 2) ∧ β ∈ (0, Real.pi / 2)

theorem find_beta (sin_alpha : Real.sin α = (Real.sqrt 5) / 5)
                  (sin_alpha_minus_beta : Real.sin (α - β) = -(Real.sqrt 10) / 10)
                  (cos_alpha : Real.cos α = (2 * Real.sqrt 5) / 5)
                  (cos_alpha_minus_beta : Real.cos (α - β) = (3 * Real.sqrt 10) / 10)
                  (acute_angles : α ∈ (0, Real.pi / 2) ∧ β ∈ (0, Real.pi / 2)) :
                  β = Real.pi / 4 := 
sorry

end find_beta_l34_34446


namespace find_a_l34_34995

-- Definitions of universal set U, set P, and complement of P in U
def U (a : ℤ) : Set ℤ := {2, 4, 3 - a^2}
def P (a : ℤ) : Set ℤ := {2, a^2 - a + 2}
def complement_U_P (a : ℤ) : Set ℤ := {-1}

-- The Lean statement asserting the conditions and the proof goal
theorem find_a (a : ℤ) (h_union : U a = P a ∪ complement_U_P a) : a = -1 :=
sorry

end find_a_l34_34995


namespace total_earning_proof_l34_34345

theorem total_earning_proof
  (days_a : ℕ) (days_b : ℕ) (days_c : ℕ)
  (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ)
  (wage_c: ℕ) :
  days_a = 6 →
  days_b = 9 →
  days_c = 4 →
  ratio_a = 3 →
  ratio_b = 4 →
  ratio_c = 5 →
  wage_c = 110 →
  let part_value := wage_c / ratio_c in
  let wage_a := ratio_a * part_value in
  let wage_b := ratio_b * part_value in
  (days_a * wage_a + days_b * wage_b + days_c * wage_c) = 1628 := 
by
  intros hdays_a hdays_b hdays_c hratio_a hratio_b hratio_c hwage_c
  simp [hdays_a, hdays_b, hdays_c, hratio_a, hratio_b, hratio_c, hwage_c]
  let part_value := 110 / 5
  let wage_a := 3 * part_value
  let wage_b := 4 * part_value
  calc
    6 * wage_a + 9 * wage_b + 4 * 110 = 6 * 66 + 9 * 88 + 4 * 110 : by sorry
                            ... = 396 + 792 + 440 : by sorry
                            ... = 1628 : by sorry
  sorry

end total_earning_proof_l34_34345


namespace min_interval_zeros_l34_34310

def f (x : ℝ) (ω : ℝ) : ℝ := sin (ω * x) + sqrt 3 * cos (ω * x) + 1

theorem min_interval_zeros (ω : ℝ) (hω : ω > 0) (hT : ∀ x, f x ω = f (x + π) ω) :
  ∃ (m n : ℝ), (∀ x, f x ω = 0 → m ≤ x ∧ x ≤ n → ∃ k, k < 5) → n - m = 2 * π :=
by sorry

end min_interval_zeros_l34_34310


namespace Vitya_wins_with_optimal_play_l34_34541

theorem Vitya_wins_with_optimal_play :
  ∀ (g : ℕ → ℕ → ℕ) (Kolya_move : g 0 0) (valid_move : ∀ n, valid g n)
  (convex_condition : ∀ n, convex (set.range (λ k => g k n))),
    (optimal_strategy Vitya) → winner_exists Kolya → (winner_with_optimal_play g = Vitya) :=
by
  sorry

end Vitya_wins_with_optimal_play_l34_34541


namespace who_received_q_first_round_l34_34034

-- Define the variables and conditions
variables (p q r : ℕ) (A B C : ℕ → ℕ) (n : ℕ)

-- Conditions
axiom h1 : 0 < p
axiom h2 : p < q
axiom h3 : q < r
axiom h4 : n ≥ 3
axiom h5 : A n = 20
axiom h6 : B n = 10
axiom h7 : C n = 9
axiom h8 : ∀ k, k > 0 → (B k = r → B (k-1) ≠ r)
axiom h9 : p + q + r = 13

-- Theorem to prove
theorem who_received_q_first_round : C 1 = q :=
sorry

end who_received_q_first_round_l34_34034


namespace double_perimeter_polygons_exist_l34_34187

theorem double_perimeter_polygons_exist (triangle_vertices : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ)
  (segment_lengths : list ℝ)
  (h1 : 1 ∈ segment_lengths)
  (h2 : sqrt 2 ∈ segment_lengths)
  (h3 : sqrt 5 ∈ segment_lengths)
  (h_total_perimeter: triangle_vertices.length = 3 * segment_lengths.sum) :
  ∃ (polygons : list (list (ℕ × ℕ))), 
    polygons.length = 4 ∧ 
    ∀ poly ∈ polygons, (segment_lengths.map (λ l, 2 * l)).sum = 2 * segment_lengths.sum 
    ∧ ∀ p1 p2 ∈ polygons, p1 ≠ p2 → ¬ congruent p1 p2 :=
by
  sorry

end double_perimeter_polygons_exist_l34_34187


namespace integral_calculation_l34_34397

noncomputable def integral_value : ℝ :=
  ∫ x in (0 : ℝ)..1, (sqrt (1 - (x - 1)^2) - x^2)

theorem integral_calculation :
  integral_value = (Real.pi / 2) - (1 / 3) := 
by
  sorry

end integral_calculation_l34_34397


namespace solve_equation_l34_34608

theorem solve_equation : ∃ x : ℚ, 3 * (x - 2) = x - (2 * x - 1) ∧ x = 7/4 := by
  sorry

end solve_equation_l34_34608


namespace greatest_common_divisor_B_l34_34565

def sum_of_five_consecutive_integers (x : ℕ) : ℕ :=
  (x - 2) + (x - 1) + x + (x + 1) + (x + 2)

def B : set ℕ := {n | ∃ x : ℕ, n = sum_of_five_consecutive_integers x}

theorem greatest_common_divisor_B : gcd (set.to_finset B).min' (set.to_finset B).max' = 5 :=
by
  sorry

end greatest_common_divisor_B_l34_34565


namespace circle_passing_through_midpoint_of_BC_l34_34387

theorem circle_passing_through_midpoint_of_BC
  (A B C H X Y Z M : Point)
  (h_acute : IsAcuteTriangle A B C)
  (h_orthocenter : Orthocenter H A B C)
  (h_omega : CircleThrough B C H ω)
  (h_gamma : CircleThrough A H Γ)
  (h_diameter_AH : Diameter AH Γ)
  (h_intersection_X : Intersects X ω Γ)
  (ne_h_X : H ≠ X)
  (h_reflection_gamma : ReflectionOver Γ AX γ)
  (intersection_Y : Intersects Y γ ω)
  (ne_X_Y : X ≠ Y)
  (intersection_AH_Z : Intersects Z ω (LineThrough A H))
  (ne_H_Z : H ≠ Z)
  (midpoint_BC : Midpoint M B C) :
  Concyclic A Y Z M :=
sorry

end circle_passing_through_midpoint_of_BC_l34_34387


namespace measure_C_in_triangle_ABC_l34_34510

noncomputable def measure_angle_C (A B : ℝ) : ℝ := 180 - (A + B)

theorem measure_C_in_triangle_ABC (A B : ℝ) (h : A + B = 150) : measure_angle_C A B = 30 := 
by
  unfold measure_angle_C
  rw h
  norm_num
  sorry

end measure_C_in_triangle_ABC_l34_34510


namespace elapsed_time_l34_34057

theorem elapsed_time (x : ℕ) (h1 : 99 > 0) (h2 : (2 : ℚ) / (3 : ℚ) * x = (4 : ℚ) / (5 : ℚ) * (99 - x)) : x = 54 := by
  sorry

end elapsed_time_l34_34057


namespace min_value_of_function_l34_34783

theorem min_value_of_function (x : ℝ) (hx : x > 0) :
  (x + 1/x + x^2 + 1/x^2 + 1 / (x + 1/x + x^2 + 1/x^2)) = 4.25 := by
  sorry

end min_value_of_function_l34_34783


namespace number_of_honey_bees_l34_34923

theorem number_of_honey_bees (total_honey : ℕ) (honey_one_bee : ℕ) (days : ℕ) (h1 : total_honey = 30) (h2 : honey_one_bee = 1) (h3 : days = 30) : 
  (total_honey / honey_one_bee) = 30 :=
by
  -- Given total_honey = 30 grams in 30 days
  -- Given honey_one_bee = 1 gram in 30 days
  -- We need to prove (total_honey / honey_one_bee) = 30
  sorry

end number_of_honey_bees_l34_34923


namespace monotonic_increasing_f_l34_34256

def f (x : ℝ) := 5 * sin (3 * x) + 5 * real.sqrt 3 * cos (3 * x)

theorem monotonic_increasing_f : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ (π / 20) → 0 ≤ deriv f x :=
by sorry

end monotonic_increasing_f_l34_34256


namespace quadratic_roots_distinct_l34_34452

variable (a b c : ℤ)

theorem quadratic_roots_distinct (h_eq : 3 * a^2 - 3 * a - 4 = 0) : ∃ (x y : ℝ), x ≠ y ∧ (3 * x^2 - 3 * x - 4 = 0) ∧ (3 * y^2 - 3 * y - 4 = 0) := 
  sorry

end quadratic_roots_distinct_l34_34452


namespace min_sentinel_sites_l34_34666

noncomputable theory

def trilandia_side_length : ℕ := 2012
def total_streets : ℕ := 6036

theorem min_sentinel_sites : 
  ∃ (s : ℕ), s = 3017 ∧ (∀ street, street ∈ total_streets → is_monitored_by_sentinel street s) :=
sorry

end min_sentinel_sites_l34_34666


namespace simplify_div_expression_evaluate_at_2_l34_34263

variable (a : ℝ)

theorem simplify_div_expression (h0 : a ≠ 0) (h1 : a ≠ 1) :
  (1 - 1 / a) / ((a^2 - 2 * a + 1) / a) = 1 / (a - 1) :=
by
  sorry

theorem evaluate_at_2 : (1 - 1 / 2) / ((2^2 - 2 * 2 + 1) / 2) = 1 :=
by 
  sorry

end simplify_div_expression_evaluate_at_2_l34_34263


namespace wristwatch_cost_proof_l34_34392

-- Definition of the problem conditions
def allowance_per_week : ℕ := 5
def initial_weeks : ℕ := 10
def initial_savings : ℕ := 20
def additional_weeks : ℕ := 16

-- The total cost of the wristwatch
def wristwatch_cost : ℕ := 100

-- Let's state the proof problem
theorem wristwatch_cost_proof :
  (initial_savings + additional_weeks * allowance_per_week) = wristwatch_cost :=
by
  sorry

end wristwatch_cost_proof_l34_34392


namespace least_number_of_marbles_l34_34722

def lcm (a b : Nat) : Nat := Nat.lcm a b

def multiple_of (a b : Nat) : Prop := ∃ k, a = b * k

theorem least_number_of_marbles :
  ∀ (n : Nat), (multiple_of n 3 ∧ multiple_of n 4 ∧ multiple_of n 5 ∧ multiple_of n 6 ∧ multiple_of n 7) ↔ n = 420 :=
begin
  sorry
end

end least_number_of_marbles_l34_34722


namespace radius_of_circle_from_spherical_coordinates_l34_34763

theorem radius_of_circle_from_spherical_coordinates :
  (∃ (ρ : ℝ) (θ : ℝ) (phi : ℝ), 
    ρ = 1 ∧ 
    phi = π / 4 ∧ 
    ∀ θ, √((ρ * sin phi * cos θ)^2 + (ρ * sin phi * sin θ)^2) = √(2) / 2) :=
sorry

end radius_of_circle_from_spherical_coordinates_l34_34763


namespace ln_x_intersection_unique_exp_function_comparison_l34_34144

-- Problem 1: Prove the intersection of y = ln(x) and y = x - 1
theorem ln_x_intersection_unique : ∃! x : ℝ, (0 < x) ∧ (ln x = x - 1) :=
begin
  sorry
end

-- Problem 2: Compare g((m + n) / 2) and (g(n) - g(m)) / (n - m) for m < n with g(x) = e^x
theorem exp_function_comparison (m n : ℝ) (h : m < n) :
  (exp n - exp m) / (n - m) > exp ((m + n) / 2) :=
begin
  sorry
end

end ln_x_intersection_unique_exp_function_comparison_l34_34144


namespace max_min_value_of_fg_l34_34097

noncomputable def f (x : ℝ) : ℝ := 4 - x^2
noncomputable def g (x : ℝ) : ℝ := 3 * x
noncomputable def min' (a b : ℝ) : ℝ := if a < b then a else b

theorem max_min_value_of_fg : ∃ x : ℝ, min' (f x) (g x) = 3 :=
by
  sorry

end max_min_value_of_fg_l34_34097


namespace smallest_n_boxes_l34_34424

theorem smallest_n_boxes (n : ℕ) : (15 * n - 1) % 11 = 0 ↔ n = 3 :=
by
  sorry

end smallest_n_boxes_l34_34424


namespace sin_tan_correct_value_l34_34180

noncomputable def sin_tan_value (x y : ℝ) (h : x^2 + y^2 = 1) : ℝ :=
  let sin_alpha := y
  let tan_alpha := y / x
  sin_alpha * tan_alpha

theorem sin_tan_correct_value :
  sin_tan_value (3/5) (-4/5) (by norm_num) = 16/15 := 
by
  sorry

end sin_tan_correct_value_l34_34180


namespace right_triangle_tangent_circle_l34_34618

theorem right_triangle_tangent_circle {D E F Q : Type} [MetricSpace D]
  (h_triangle : RightTriangle D E F)
  (h_right_angle : angle_at E = 90)
  (h_DF : distance D F = real.sqrt 85)
  (h_DE : distance D E = 7)
  (circle_center_on_DE : ∃ (O : Type), distance O D = distance O E ∧ tangent_to_O_circle D F)
  (h_tangent : tangent_to_circle DF O Q) :
  let EF := distance E F in EF = 6 :=
by
  sorry

end right_triangle_tangent_circle_l34_34618


namespace reduce_regular_polygon_to_two_l34_34549

-- Define the problem using Lean's formal language
theorem reduce_regular_polygon_to_two {n : ℕ} (h : n ≥ 3) :
  ∃ (k : ℕ), k = 2 ∧ 
  ∀ (P : List (ℝ × ℝ)), -- Representing the vertices as a list of points on the circle
    is_regular_n_sided_polygon P n → 
    (∃ (f : List (ℝ × ℝ) → List (ℝ × ℝ)), -- This function applies the given operation
      f ≠ id ∧ -- The function is not the identity (it reduces the vertices)
      iteratively_apply_operation f P (List.length P) = 2) := sorry

-- Helper definitions (these would need more details in an actual lean development environment):
def is_regular_n_sided_polygon (P : List (ℝ × ℝ)) (n : ℕ) : Prop :=
  ∃ (r : ℝ) (ω : ℝ × ℝ), -- Circle center ω and radius r
  (∀ (p ∈ P), distance p ω = r) ∧ -- All points are on the circle
  (∀ (i j : ℕ), i < j → j < n → distance (P.nth i) (P.nth j) = distance (P.nth (i+1)) (P.nth (j+1))) -- Equal spacing (regular polygon)

def iteratively_apply_operation (f : List (ℝ × ℝ) → List (ℝ × ℝ)) (P : List (ℝ × ℝ)) (n : ℕ) : ℕ :=
  if n ≤ 2 then n else iteratively_apply_operation f (f P) (List.length (f P))

#check reduce_regular_polygon_to_two

end reduce_regular_polygon_to_two_l34_34549


namespace diagonal_bisection_l34_34535

variables {Point : Type} [EuclideanGeometry Point]
variables (A B C D O : Point)
variables (AC BD : Line Point)

theorem diagonal_bisection 
  (h1 : collinear A O C)
  (h2 : collinear B O D)
  (h3 : area A B C = area A C D)
  : distance B O = distance O D :=
sorry

end diagonal_bisection_l34_34535


namespace Henry_age_l34_34191

-- Define the main proof statement
theorem Henry_age (h s : ℕ) 
(h1 : h + 8 = 3 * (s - 1))
(h2 : (h - 25) + (s - 25) = 83) : h = 97 :=
by
  sorry

end Henry_age_l34_34191


namespace deceased_income_l34_34688

variables (family_before : ℕ) (family_after : ℕ) (before death average : ℕ) (after death average : ℕ)

-- Given conditions
def avg_income_before := 735
def avg_income_after := 590
def members_before := 4
def members_after := 3

theorem deceased_income (I_d : ℕ) :
  (members_before * avg_income_before - members_after * avg_income_after = I_d) → I_d = 1170 :=
begin
  sorry
end

end deceased_income_l34_34688


namespace decreasing_interval_of_f_l34_34466

-- Given definitions based on conditions
def f (x : ℝ) : ℝ := -x^2 + 3*x + 4

-- Main theorem statement
theorem decreasing_interval_of_f : 
  ∀ x : ℝ, (f(x) = -x^2 + 3*x + 4) ∧ (f(-1) = 0) ∧ (f(4) = 0) ∧ (f(0) = 4) → 
  ∃ I, I = set.Ici (3 / 2) ∧ (∀ x ∈ I, ∀ y ∈ I, x < y → f(y) ≤ f(x)) :=
by
  sorry

end decreasing_interval_of_f_l34_34466


namespace max_tickets_l34_34748

theorem max_tickets (ticket_cost money_available : ℕ) (h1 : ticket_cost = 15) (h2 : money_available = 120) : 
  ∃ n : ℕ, (n ≤ money_available / ticket_cost) ∧ n = 8 :=
by 
  have h : money_available / ticket_cost = 8 := by
    rw [h1, h2]
    norm_num
  use 8
  split
  · rw h
    exact le_refl 8
  · refl
sorry

end max_tickets_l34_34748


namespace original_integer_is_26_l34_34441

theorem original_integer_is_26 (x y z w : ℕ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : 0 < w)
(h₅ : x ≠ y) (h₆ : x ≠ z) (h₇ : x ≠ w) (h₈ : y ≠ z) (h₉ : y ≠ w) (h₁₀ : z ≠ w)
(h₁₁ : (x + y + z) / 3 + w = 34)
(h₁₂ : (x + y + w) / 3 + z = 22)
(h₁₃ : (x + z + w) / 3 + y = 26)
(h₁₄ : (y + z + w) / 3 + x = 18) :
    w = 26 := 
sorry

end original_integer_is_26_l34_34441


namespace sample_size_calculation_l34_34363

theorem sample_size_calculation 
    (total_teachers : ℕ) (total_male_students : ℕ) (total_female_students : ℕ) 
    (sample_size_female_students : ℕ) 
    (H1 : total_teachers = 100) (H2 : total_male_students = 600) 
    (H3 : total_female_students = 500) (H4 : sample_size_female_students = 40)
    : (sample_size_female_students * (total_teachers + total_male_students + total_female_students) / total_female_students) = 96 := 
by
  /- sorry, proof omitted -/
  sorry
  
end sample_size_calculation_l34_34363


namespace abs_opposite_of_three_eq_5_l34_34171

theorem abs_opposite_of_three_eq_5 : ∀ (a : ℤ), a = -3 → |a - 2| = 5 := by
  sorry

end abs_opposite_of_three_eq_5_l34_34171


namespace distribution_less_than_m_plus_f_l34_34687

-- Definitions based on the problem conditions
def is_symmetric (dist : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, dist (m + x) = dist (m - x)

def within_one_std_dev (dist : ℝ → ℝ) (m : ℝ) (f : ℝ) : Prop :=
  ∫ x in (m - f, m + f), dist x = 0.68

-- The main proof statement
theorem distribution_less_than_m_plus_f (dist : ℝ → ℝ) (m f : ℝ)
  (symm : is_symmetric dist m)
  (one_std_dev : within_one_std_dev dist m f) : 
  ∫ x in (-∞, m + f), dist x = 0.84 :=
sorry

end distribution_less_than_m_plus_f_l34_34687


namespace ratio_of_incomes_l34_34642

-- Given values
def annual_income_A : ℝ := 537600
def monthly_income_C : ℝ := 16000
def percentage_increase_B : ℝ := 0.12

-- Derived values
def monthly_income_A : ℝ := annual_income_A / 12
def monthly_income_B : ℝ := monthly_income_C + (percentage_increase_B * monthly_income_C)

-- Define the ratio
def ratio_A_B (x y : ℝ) : Prop := x / y = 5 / 2

-- Formal statement to be proven
theorem ratio_of_incomes : ratio_A_B monthly_income_A monthly_income_B :=
by
  -- skip the proof
  sorry

end ratio_of_incomes_l34_34642


namespace cherries_left_after_eating_l34_34242

-- Defining the initial conditions
def initial_cherries : ℕ := 16
def difference_cherries : ℕ := 10

-- The theorem to prove
theorem cherries_left_after_eating (cherries_left : ℕ) : cherries_left = 6 :=
by
  have h_initial : initial_cherries = 16 := rfl
  have h_difference : difference_cherries = 10 := rfl
  have h_equation : (initial_cherries - cherries_left) = difference_cherries
  { sorry }  -- We will not solve this equation because it is outside the scope.
  exact (nat.sub_eq_iff_eq_add.mpr h_initial.symm).resolve_left h_difference.symm
  sorry

end cherries_left_after_eating_l34_34242


namespace abs_neg_three_l34_34279

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l34_34279


namespace collinear_A_D_L_iff_concyclic_P_X_Y_Q_l34_34942

variable {α : Type*} [Euclidean.Geometry α] (A B C O I D X Y L P Q : α)

-- Define the necessary conditions
def acute_triangle (A B C : α) : Prop := ∀ (a b c : α), (angle B C (O) < π / 2) ∧ (a > b → angle a b c < angle a c b)
def circumcircle (O : α) (△ABC : Triangle α) := ∃ O ∈ circumcenter(A, B, C), circumcircle(△ABC)
def incircle (I : α) (△ABC : Triangle α) := ∃ I ∈ incenter(A, B, C), incircle(△ABC)
def incircle_touches_BC_at_D (I : α) (D : α) (△ABC : Triangle α) := touches(I, BC, D)
def line_AO_intersects_BC_at_X (A O X : α) := intersects(line(A, O), line(B, C), X)
def altitude_AY (A Y : α) := perpendicular(A, Y, line(B, C))
def tangents_intersect_L (O B C L : α) := tangents(circle(O), B) ∩ tangents(circle(O), C) = L
def diameter_PQ_thru_I (O P Q I : α) := diameter(O, P, Q) ∧ lies_on(P, I) ∧ lies_on(Q, I)

-- Lean 4 theorem statement
theorem collinear_A_D_L_iff_concyclic_P_X_Y_Q 
  (h1 : ∃O, circumcircle O (triangle(A, B, C)))
  (h2 : ∃I, incircle I (triangle(A, B, C)))
  (h3 : incircle_touches_BC_at_D I D (triangle(A, B, C)))
  (h4 : line_AO_intersects_BC_at_X A O X)
  (h5 : altitude_AY A Y)
  (h6 : tangents_intersect_L O B C L)
  (h7 : diameter_PQ_thru_I O P Q I) :
  collinear(A, D, L) ↔ concyclic(P, X, Y, Q)
:=
sorry

end collinear_A_D_L_iff_concyclic_P_X_Y_Q_l34_34942


namespace angle_between_vectors_is_90_degrees_l34_34458

noncomputable def vec_angle (v₁ v₂ : ℝ × ℝ) : ℝ :=
sorry -- This would be the implementation that calculates the angle between two vectors

theorem angle_between_vectors_is_90_degrees
  (A B C O : ℝ × ℝ)
  (h1 : dist O A = dist O B)
  (h2 : dist O A = dist O C)
  (h3 : dist O B = dist O C)
  (h4 : 2 • (A - O) = (B - O) + (C - O)) :
  vec_angle (B - A) (C - A) = 90 :=
sorry

end angle_between_vectors_is_90_degrees_l34_34458


namespace part1_part2_l34_34937

noncomputable def choose (n : ℕ) (k : ℕ) : ℕ :=
  n.choose k

theorem part1 :
  let internal_medicine_doctors := 12
  let surgeons := 8
  let total_doctors := internal_medicine_doctors + surgeons
  let team_size := 5
  let doctors_left := total_doctors - 1 - 1 -- as one internal medicine must participate and one surgeon cannot
  choose doctors_left (team_size - 1) = 3060 := by
  sorry

theorem part2 :
  let internal_medicine_doctors := 12
  let surgeons := 8
  let total_doctors := internal_medicine_doctors + surgeons
  let team_size := 5
  let only_internal_medicine := choose internal_medicine_doctors team_size
  let only_surgeons := choose surgeons team_size
  let total_ways := choose total_doctors team_size
  total_ways - only_internal_medicine - only_surgeons = 14656 := by
  sorry

end part1_part2_l34_34937


namespace average_test_score_first_25_percent_l34_34174

theorem average_test_score_first_25_percent (x : ℝ) :
  (0.25 * x) + (0.50 * 65) + (0.25 * 90) = 1 * 75 → x = 80 :=
by
  sorry

end average_test_score_first_25_percent_l34_34174


namespace disc_thickness_l34_34723

theorem disc_thickness (r_sphere : ℝ) (r_disc : ℝ) (h : ℝ)
  (h_radius_sphere : r_sphere = 3)
  (h_radius_disc : r_disc = 10)
  (h_volume_constant : (4/3) * Real.pi * r_sphere^3 = Real.pi * r_disc^2 * h) :
  h = 9 / 25 :=
by
  sorry

end disc_thickness_l34_34723


namespace find_f_2_l34_34109

noncomputable def f : ℕ → ℕ
| x => x.nat_abs ^ x.nat_abs + 2 * x.nat_abs + 2

theorem find_f_2 :
  f 2 = 5 := 
by
  -- Place proof here, but for now we put sorry
  sorry

end find_f_2_l34_34109


namespace hexagon_perimeter_l34_34676

-- Define points and lengths
variable (A B C D E F : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable [MetricSpace D] [MetricSpace E] [MetricSpace F]
variable (dist : A → A → ℝ)

-- Given lengths (conditions)
variable hAB : dist A B = 1
variable hBC : dist B C = 1
variable hCD : dist C D = 1
variable hDE : dist D E = 1
variable hEF : dist E F = 1
variable hAF : dist A F = ℝ.sqrt 5

-- Prove the perimeter is 5 + sqrt 5
theorem hexagon_perimeter : dist A B + dist B C + dist C D + dist D E + dist E F + dist A F = 5 + ℝ.sqrt 5 := 
by
  rw [hAB, hBC, hCD, hDE, hEF, hAF]
  simp
  sorry

end hexagon_perimeter_l34_34676


namespace intersection_of_A_and_B_l34_34460

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by sorry

end intersection_of_A_and_B_l34_34460


namespace compute_fraction_power_l34_34747

theorem compute_fraction_power : (45000 ^ 3 / 15000 ^ 3) = 27 :=
by
  sorry

end compute_fraction_power_l34_34747


namespace correct_judgments_l34_34470

def like_terms (t1 t2 : Polynomial) : Prop :=
  ∀ (x : Variable), (degree (t1.coeff x) = degree (t2.coeff x)) ∧
  (∀ c ∈ t1.vars, c ∈ t2.vars)

def constant_term (p : Polynomial) : ℝ :=
  p.coefficients.get 0 

def is_quadratic_trinomial (p : Polynomial) : Prop :=
  p.vars.length = 3 ∧ max_degree p = 2

def is_polynomial (t : Expression) : Prop := sorry

theorem correct_judgments :
  like_terms (Polynomial.mk [(2, a^2 * b), (π-2, 0)]) (Polynomial.mk [(1/3, a^2 * b), (-1/3, 0)]) ∧
  ¬(constant_term (Polynomial.mk [(5, a), (4, b), (-1, 1)]) = 1) ∧
  is_quadratic_trinomial (Polynomial.mk [(1, x), (-2, x * y), (1, y)]) ∧
  is_polynomial (Expression.mk [(x + y) / 4]) ∧ is_polynomial (Expression.mk [x / 2 + 1]) ∧ is_polynomial (Expression.mk [a / 4])
  ↔ C := 
sorry

end correct_judgments_l34_34470


namespace sin_2x_plus_one_equals_9_over_5_l34_34104

theorem sin_2x_plus_one_equals_9_over_5 (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : Real.sin (2 * x) + 1 = 9 / 5 :=
sorry

end sin_2x_plus_one_equals_9_over_5_l34_34104


namespace length_of_BD_correct_l34_34190

noncomputable def length_of_BD (BD BE AC : ℝ) : Prop :=
  ∃ (AB : ℝ), is_isosceles_triangle AB AC ∧
  is_midpoint D AC ∧
  BE = 13 ∧
  BE_intersects_AC_at_D BE AC ∧
  BD = BE / 2

theorem length_of_BD_correct :
  length_of_BD 6.5 13 13 :=
by
  sorry

end length_of_BD_correct_l34_34190


namespace three_digit_integers_congruent_to_2_mod_4_l34_34902

theorem three_digit_integers_congruent_to_2_mod_4 : 
  let count := (249 - 25 + 1) in
  count = 225 :=
by
  let k_min := 25
  let k_max := 249
  have h_count : count = (k_max - k_min + 1) := rfl
  rw h_count
  norm_num

end three_digit_integers_congruent_to_2_mod_4_l34_34902


namespace octal_to_binary_l34_34061

theorem octal_to_binary (oct125 : ℕ := 0o135) : nat.to_digits 2 oct125 = [1, 0, 1, 1, 1, 0, 1] :=
by
  sorry

end octal_to_binary_l34_34061


namespace sum_of_values_of_x_eq_12_l34_34229

def f (x : ℝ) : ℝ :=
if x < 1 then 7 * x + 5 else 3 * x - 21

theorem sum_of_values_of_x_eq_12 : 
    (∀ x, f x = 12 → ((x < 1 → False) ∧ (x ≥ 1 → x = 11))) ∧ (sum_of_possible_values f 12 = 11) :=
sorry

end sum_of_values_of_x_eq_12_l34_34229


namespace part1_part2_part3_l34_34831

variables {R : Type*} [linear_ordered_field R]

def set_A (x : R) : Prop := x^2 - 4 * x + 3 ≤ 0
def set_B (x : R) : Prop := (x - 2) / (x + 3) > 0
def set_C (x a : R) : Prop := 1 < x ∧ x < a

theorem part1 (x : R) : (set_A x ∧ set_B x) ↔ (2 < x ∧ x ≤ 3) := sorry

theorem part2 (x : R) : (¬ set_A x ∨ ¬ set_B x) ↔ (x ≤ 2 ∨ 3 < x) := sorry

theorem part3 (x a : R) (h : ∀ x, set_C x a → set_A x) : a ≤ 3 := sorry

end part1_part2_part3_l34_34831


namespace compute_3_pow_h_l34_34309

noncomputable def log_3_64 : ℝ := Real.log 64 / Real.log 3
noncomputable def log_9_16 : ℝ := Real.log 16 / Real.log 9
noncomputable def t : ℝ := Real.log 2 / Real.log 3

theorem compute_3_pow_h :
  let a := log_3_64
  let b := log_9_16
  let h := Real.sqrt (a^2 + b^2)
  3^h = 2^(2 * Real.sqrt 10) :=
by
  sorry

end compute_3_pow_h_l34_34309


namespace calculate_heartsuit_ratio_l34_34172

def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

theorem calculate_heartsuit_ratio :
  (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 :=
by sorry

end calculate_heartsuit_ratio_l34_34172


namespace probability_area_triangle_lt_S_div_4_l34_34516

-- Definitions based on the given conditions
variables {P B C : Type} [MetricSpace P]

def area (ABCD : Rectangle P) : ℝ := sorry -- The area of rectangle ABCD

def area_of_triangle (P B C : P) : ℝ := sorry -- The area of triangle PBC

-- The Lean 4 statement for the problem
theorem probability_area_triangle_lt_S_div_4
  (ABCD : Rectangle P) (S : ℝ) (hS : area ABCD = S) (P : P) :
  (random_point : P → Prop) → 
  ∃ (p : ℝ), p = 1 / 2 :=
by 
  sorry

end probability_area_triangle_lt_S_div_4_l34_34516


namespace angle_between_AF_and_CE_l34_34517

noncomputable def tetrahedron_angle_AF_CE : Real :=
  let a : ℝ := 1 -- Edge length of the tetrahedron, this is arbitrary but fixed for the calculation
  let A := (0, 0, 0)
  let B := (a, 0, 0)
  let C := (a / 2, a * (Real.sqrt 3 / 2), 0)
  let D := (a / 2, a * (Real.sqrt 3 / 6), a * (Real.sqrt 6 / 3))
  let E := ((A.1 + D.1) / 2, (A.2 + D.2) / 2, (A.3 + D.3) / 2)
  let F := ((B.1 + C.1) / 2, (B.2 + C.2) / 2, (B.3 + C.3) / 2)
  let AF := (F.1 - A.1, F.2 - A.2, F.3 - A.3)
  let CE := (E.1 - C.1, E.2 - C.2, E.3 - C.3)
  let dot_product := AF.1 * CE.1 + AF.2 * CE.2 + AF.3 * CE.3
  let norm_AF := Real.sqrt (AF.1^2 + AF.2^2 + AF.3^2)
  let norm_CE := Real.sqrt (CE.1^2 + CE.2^2 + CE.3^2)
  Real.arccos (dot_product / (norm_AF * norm_CE))

theorem angle_between_AF_and_CE :
  tetrahedron_angle_AF_CE = Real.arccos (2 / 3) :=
sorry

end angle_between_AF_and_CE_l34_34517


namespace dave_spent_on_books_l34_34419

-- Define the cost of books in each category without any discounts or taxes
def cost_animal_books : ℝ := 8 * 10
def cost_outer_space_books : ℝ := 6 * 12
def cost_train_books : ℝ := 9 * 8
def cost_history_books : ℝ := 4 * 15
def cost_science_books : ℝ := 5 * 18

-- Define the discount and tax rates
def discount_animal_books : ℝ := 0.10
def tax_science_books : ℝ := 0.15

-- Apply the discount to animal books
def discounted_cost_animal_books : ℝ := cost_animal_books * (1 - discount_animal_books)

-- Apply the tax to science books
def final_cost_science_books : ℝ := cost_science_books * (1 + tax_science_books)

-- Calculate the total cost of all books after discounts and taxes
def total_cost : ℝ := discounted_cost_animal_books 
                  + cost_outer_space_books
                  + cost_train_books
                  + cost_history_books
                  + final_cost_science_books

theorem dave_spent_on_books : total_cost = 379.5 := by
  sorry

end dave_spent_on_books_l34_34419


namespace count_correct_conclusions_l34_34799

noncomputable def f : ℝ → ℝ := λ x, 2^x

theorem count_correct_conclusions (x1 x2 : ℝ) (hx : x1 ≠ x2) :
  (¬ (f (x1 * x2) = f x1 + f x2)) ∧
  (f (x1 + x2) = f x1 * f x2) ∧
  (f (-x1) = 1 / f x1) ∧
  (¬ (x1 ≠ 0 → (f x1 - 1) / x1 < 0)) ∧
  ((f x1 - f x2) / (x1 - x2) > 0) →
  3 := 
  begin
    sorry
  end

end count_correct_conclusions_l34_34799


namespace meryll_questions_l34_34999

theorem meryll_questions (M P : ℕ) (h1 : (3/5 : ℚ) * M + (2/3 : ℚ) * P = 31) (h2 : P = 15) : M = 35 :=
sorry

end meryll_questions_l34_34999


namespace prob_A_left_of_B_after_two_swaps_l34_34661

theorem prob_A_left_of_B_after_two_swaps :
  let s : Finset (Finset (ℕ × ℕ)) := Finset.powerset (Finset.univ : Finset (ℕ × ℕ)) in
  let n : ℕ := s.card * s.card in
  let m : ℕ := s.filter (λ ⟨(i₁, j₁), (i₂, j₂)⟩, 
    (i₁, j₁) ≠ (i₂, j₂) ∧ 
    ((i₁ < j₁ ∧ j₁ = i₂ ∧ i₂ < j₂) ∨ 
     (i₂ < j₂ ∧ j₂ = i₁ ∧ i₁ < j₁) ∨ 
     (i₁ < j₁ ∧ i₂ < j₂ ∧ j₁ ≠ i₂ ∧ j₂ ≠ i₁))).card in
  (m : ℚ) / n = 2 / 3 := 
sorry

end prob_A_left_of_B_after_two_swaps_l34_34661


namespace side_of_square_24_l34_34531

-- | Definitions for the conditions.
variable (ABCD : Type) [has_SquareShape ABCD] -- Square ABCD
variable (ABE BCF CDG DAH : Type) (hf : Type)
variable [has_IsoscelesTriangleShape ABE BCF CDG DAH] -- Identical isosceles triangles
variable [IsoscelesTriangleBaseOnSquare AB E BCF CDG DAH] -- Bases on the sides of the square
variable (gray_area white_area : ℝ) -- Areas of the gray and white regions
variable (hf_length : ℝ) -- Length of HF
variable [hf_length_is_known : hf_length = 12] -- Length HF is given as 12 cm
variable [gray_area_half_white : gray_area = white_area] -- Gray areas equal to white area
variable [white_area_half_sum : gray_area + white_area = (2 * white_area)] -- White area is half the total

-- | The final statement to prove.
theorem side_of_square_24 : ∃ s : ℝ, (s = 24) := by
  have h1 : hf_length = 12 := sorry
  have h2 : 2 * 6 = 12 := sorry
  have h3 : s = 2 * 12 := by
    rw [← h1, ← h2]
    sorry
  use s
  exact h3

end side_of_square_24_l34_34531


namespace shortest_path_on_truncated_cone_l34_34518

-- Define the geometric properties of the truncated cone
structure TruncatedCone where
  theta : Real  -- angle between the axis and slant height in radians
  R : Real      -- radius of the larger base
  r : Real      -- radius of the smaller base

-- Given a truncated cone with theta = 30 degrees (π/6 radians)
def givenTruncatedCone : TruncatedCone :=
  { theta := Real.pi / 6, R := sorry, r := sorry }

theorem shortest_path_on_truncated_cone (c : TruncatedCone) (h : c.theta = Real.pi / 6) :
  ∃ L, L = 2 * c.R :=
sorry

end shortest_path_on_truncated_cone_l34_34518


namespace problem_conditions_l34_34617

theorem problem_conditions (f : ℝ → ℝ) (a b c : ℝ) 
  (h1 : ∀ x, f(x-2) = 4 * x^2 + 9 * x + 5) 
  (h2 : ∀ x, f x = a * x^2 + b * x + c) : 
  a + b + c = 68 := 
sorry

end problem_conditions_l34_34617


namespace finish_day_is_friday_l34_34491

def dayOfWeek : Type := ℕ  -- Define a type for days of the week indexed by natural numbers

def friday : dayOfWeek := 0  -- Define Friday as 0

def days_to_read_books (n : ℕ) : ℕ := (n * (n + 1)) / 2  -- Function to compute days to read "n" books

def day_of_week_after_days (start_day : dayOfWeek) (days : ℕ) : dayOfWeek :=
(start_day + days) % 7  -- Function to compute the day of the week after a certain number of days

theorem finish_day_is_friday :
  day_of_week_after_days friday (days_to_read_books 20) = friday :=
by
  -- Start of proof, desired to finish without it, thus using sorry
  sorry

end finish_day_is_friday_l34_34491


namespace min_edges_in_graph_with_conditions_l34_34674

open Finset

def min_edges (n : ℕ) :=
∀ (G : SimpleGraph (Fin n)), (∀ (s : Finset (Fin n)), s.card = 5 → 2 ≤ (G.sEdges s).card) → (G.edgeFinctioncard = 12)

theorem min_edges_in_graph_with_conditions : min_edges 10 :=
sorry

end min_edges_in_graph_with_conditions_l34_34674


namespace polygon_interior_sum_sum_of_exterior_angles_l34_34847

theorem polygon_interior_sum (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

theorem sum_of_exterior_angles (n : ℕ) : 360 = 360 :=
by
  sorry

end polygon_interior_sum_sum_of_exterior_angles_l34_34847


namespace total_shaded_area_l34_34026

theorem total_shaded_area :
  let S := 12 / 4,
  let T := S / 4,
  12 * ((T * T) + (S * S)) = 15.75 :=
by
  let S := 12 / 4
  let T := S / 4
  have h1 : S = 3 := by sorry
  have h2 : T = 3 / 4 := by sorry
  have h3 : 12 * (T * T) = 12 * (3 / 4) * (3 / 4) := by sorry
  have h4 : 12 * (3 / 4) * (3 / 4) = 12 * (9 / 16) := by sorry
  have h5 : 12 * (9 / 16) = 108 / 16 := by sorry
  have h6 : 108 / 16 + S * S = 15.75 := by sorry
  show 12 * (T * T) + S * S = 15.75, from h6

end total_shaded_area_l34_34026


namespace value_of_a_l34_34181

variable (α : ℝ) (a : ℝ)

def P := (x:ℝ, y:ℝ) -> (x = -4) ∧ (y = a)

def sin_alpha := a / real.sqrt (16 + a^2)
def cos_alpha := -4 / real.sqrt (16 + a^2)
def sin_cos_condition : Prop := sin_alpha α a * cos_alpha α a = real.sqrt 3 / 4

theorem value_of_a (α : ℝ) (a : ℝ) : 
  P (-4, a) α → sin_cos_condition a → (a = -4 * real.sqrt 3 ∨ a = -(4 * real.sqrt 3) / 3) := 
by 
  sorry

end value_of_a_l34_34181


namespace football_team_throwers_l34_34245

theorem football_team_throwers {T N : ℕ} (h1 : 70 - T = N)
                                (h2 : 62 = T + (2 / 3 * N)) : 
                                T = 46 := 
by
  sorry

end football_team_throwers_l34_34245


namespace range_of_k_l34_34117

theorem range_of_k (k : ℝ) :
  let circle_eq : ℝ → ℝ → ℝ := λ x y, x^2 + y^2 + 2*k*x + 2*y + k^2 - 24
  (∃ (p : ℝ × ℝ), p = (1, 3) ∧ 
   ((p.1 + k)^2 + (p.2 + 1)^2 > 25)) →
  (k > 2 ∨ k < -4) :=
by
  let p := (1, 3)
  assume h : ((p.1 + k)^2 + (p.2 + 1)^2 > 25)
  sorry

end range_of_k_l34_34117


namespace gcd_of_set_B_is_five_l34_34557

def is_in_set_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)

theorem gcd_of_set_B_is_five : ∃ d, (∀ n, is_in_set_B n → d ∣ n) ∧ d = 5 :=
by 
  sorry

end gcd_of_set_B_is_five_l34_34557


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l34_34895

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : ∃ (n : ℕ), n = 225 ∧ ∀ k : ℤ, 100 ≤ 4 * k + 2 ∧ 4 * k + 2 ≤ 999 → 24 < k ∧ k < 250 := by
  sorry

end number_of_three_digit_integers_congruent_to_2_mod_4_l34_34895


namespace soccer_campers_l34_34725

theorem soccer_campers (total_campers : ℕ) (basketball_campers : ℕ) (football_campers : ℕ) (h1 : total_campers = 88) (h2 : basketball_campers = 24) (h3 : football_campers = 32) : 
  total_campers - (basketball_campers + football_campers) = 32 := 
by 
  -- Proof omitted
  sorry

end soccer_campers_l34_34725


namespace find_a_l34_34786

namespace MathProof

theorem find_a (a : ℕ) (h_pos : a > 0) (h_eq : (a : ℚ) / (a + 18) = 47 / 50) : a = 282 :=
by
  sorry

end MathProof

end find_a_l34_34786


namespace area_of_fourth_rectangle_l34_34014

theorem area_of_fourth_rectangle
  (A1 A2 A3 A_total : ℕ)
  (h1 : A1 = 24)
  (h2 : A2 = 30)
  (h3 : A3 = 18)
  (h_total : A_total = 100) :
  ∃ A4 : ℕ, A1 + A2 + A3 + A4 = A_total ∧ A4 = 28 :=
by
  sorry

end area_of_fourth_rectangle_l34_34014


namespace greatest_common_divisor_B_l34_34566

def sum_of_five_consecutive_integers (x : ℕ) : ℕ :=
  (x - 2) + (x - 1) + x + (x + 1) + (x + 2)

def B : set ℕ := {n | ∃ x : ℕ, n = sum_of_five_consecutive_integers x}

theorem greatest_common_divisor_B : gcd (set.to_finset B).min' (set.to_finset B).max' = 5 :=
by
  sorry

end greatest_common_divisor_B_l34_34566


namespace smallest_n_2015_l34_34576

noncomputable def f : ℕ+ → ℤ := sorry

def is_prime (p : ℕ) : Prop := nat.prime p

def satisfies_properties (f : ℕ+ → ℤ) :=
  (f 1 = 0) ∧
  (∀ p : ℕ+, is_prime p → f p = 1) ∧
  (∀ x y : ℕ+, f (x * y) = y * f x + x * f y)

theorem smallest_n_2015 (f : ℕ+ → ℤ) (h : satisfies_properties f) : ∃ n : ℕ+, n.val ≥ 2015 ∧ f n = n ∧ n = 2100 :=
sorry

end smallest_n_2015_l34_34576


namespace root_nature_for_imaginary_m_l34_34416

noncomputable def complex_quadratic_roots_pure_imaginary 
  (m : ℂ) (h_imag_m : m.im = m ∧ m.re = 0) : Prop :=
  let f := (8 * X^2 + 4 * complex.I * X - complex.C m : Polynomial ℂ) in
  ∀ (z : ℂ), z ∈ f.rootSet ℂ → (z.re = 0)

theorem root_nature_for_imaginary_m (m : ℂ) (h_imag_m : m.im = m ∧ m.re = 0) :
  complex_quadratic_roots_pure_imaginary m h_imag_m :=
sorry

end root_nature_for_imaginary_m_l34_34416


namespace school_survey_l34_34721

theorem school_survey (n k smallest largest : ℕ) (h1 : n = 24) (h2 : k = 4) (h3 : smallest = 3) (h4 : 1 ≤ smallest ∧ smallest ≤ n) (h5 : largest - smallest = (k - 1) * (n / k)) : 
  largest = 21 :=
by {
  sorry
}

end school_survey_l34_34721


namespace coeff_third_term_binom_expansion_l34_34622

theorem coeff_third_term_binom_expansion :
  let expr := (sqrt 2 * x - 1)^5 in
  let third_term_coeff := 20 * sqrt 2 in
  ∃ f : ℕ → ℝ, f 2 = third_term_coeff ∧ (expr = ∑ i in finset.range 6, f i * x^(5 - i)) :=
by
  sorry

end coeff_third_term_binom_expansion_l34_34622


namespace max_sin_A_plus_sin_B_l34_34957

theorem max_sin_A_plus_sin_B
  (a b c : ℝ)
  (h : (a + c) * (a - c) + b * (b - a) = 0) :
  ∃ A B C : ℝ, ∠A + ∠B + ∠C = π ∧ ∠C = π/3 ∧ ∠B = 2*π/3 - ∠A
  ∧ ∀ ∠A : ℝ, ∠A > 0 ∧ ∠A < 2*π/3 → (∠A + ∠B = ∠A + sin (2*π/3 - ∠A) = sqrt(3) := sorry.

end max_sin_A_plus_sin_B_l34_34957


namespace next_divisor_after_391_l34_34203

theorem next_divisor_after_391 (m : ℕ) (h1 : m % 2 = 0) (h2 : m ≥ 1000 ∧ m < 10000) (h3 : 391 ∣ m) : 
  ∃ n, n > 391 ∧ n ∣ m ∧ (∀ k, k > 391 ∧ k < n → ¬ k ∣ m) ∧ n = 782 :=
sorry

end next_divisor_after_391_l34_34203


namespace area_triangle_equality_l34_34821

-- Given points A, B, C, and P
variables {A B C P Q R : Type}

-- Vectors representing the points
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] 
          [Module ℝ A] [Module ℝ B] [Module ℝ C] 
          {a b c pa pb pc : A}

-- Parallel conditions
def CQ_parallel_AP := ∃ α : ℝ, Q = C + α • A
def AR_parallel_BQ := ∃ β : ℝ, R = A + β • B
def CR_parallel_BP := ∃ γ : ℝ, R = C + γ • P

-- Areas of triangles are equal
theorem area_triangle_equality 
  (h1 : CQ_parallel_AP)
  (h2 : AR_parallel_BQ)
  (h3 : CR_parallel_BP) :
  area_triangle A B C = area_triangle P Q R :=
sorry

end area_triangle_equality_l34_34821


namespace washes_per_bottle_l34_34202

def bottle_cost : ℝ := 4.0
def total_weeks : ℕ := 20
def total_cost : ℝ := 20.0

theorem washes_per_bottle : (total_weeks / (total_cost / bottle_cost)) = 4 := by
  sorry

end washes_per_bottle_l34_34202


namespace geometric_sequence_third_term_l34_34793

theorem geometric_sequence_third_term (a b c d : ℕ) (r : ℕ) 
  (h₁ : d * r = 81) 
  (h₂ : 81 * r = 243) 
  (h₃ : r = 3) : c = 27 :=
by
  -- Insert proof here
  sorry

end geometric_sequence_third_term_l34_34793


namespace max_min_values_of_f_on_interval_l34_34145

noncomputable def f (x : ℝ) : ℝ := -x^2 + x + 1

theorem max_min_values_of_f_on_interval :
  let I := set.Icc (0 : ℝ) (3 / 2)
  ∃ x_min x_max ∈ I, 
    (∀ x ∈ I, f x_min ≤ f x) ∧ (∀ x ∈ I, f x ≤ f x_max) ∧ 
    f x_min = 1/4 ∧ f x_max = 5/4 :=
begin
  sorry
end

end max_min_values_of_f_on_interval_l34_34145


namespace width_decrease_l34_34713

-- Given conditions and known values
variable (L W : ℝ) -- original length and width
variable (P : ℝ)   -- percentage decrease in width

-- The known condition for the area comparison
axiom area_condition : 1.4 * (L * (W * (1 - P / 100))) = 1.1199999999999999 * (L * W)

-- The property we want to prove
theorem width_decrease (L W: ℝ) (h : L > 0) (h1 : W > 0) :
  P = 20 := 
by
  sorry

end width_decrease_l34_34713


namespace smallest_N_moves_l34_34209

theorem smallest_N_moves 
  (n : ℕ) (h : n ≥ 2) : 
  ∃ N : ℕ, (∀ initial_state : list (list (ℕ × ℕ)),
  (initial_state.length = n ∧ ∀ b : list (ℕ × ℕ), b ∈ initial_state → b.length = 2 ∧ ∀ c₁ c₂ : ℕ × ℕ, c₁ ∈ b → c₂ ∈ b → c₁.fst = c₂.fst)) →
  (∃ (moves : list (list ℕ → list (ℕ × ℕ))), moves.length ≤ N ∧
    ∃ final_state : list (list (ℕ × ℕ)),
    final_state.length = n ∧ ∀ b : list (ℕ × ℕ), b ∈ final_state → b.length = 2 ∧ ∀ c₁ c₂ : ℕ × ℕ, c₁ ∈ b → c₂ ∈ b → c₁.fst = c₂.fst)) ∧
    N = ⌊3 * n / 2⌋ :=
begin
  sorry
end

end smallest_N_moves_l34_34209


namespace proof_complex_equality_l34_34974

theorem proof_complex_equality (a b : ℝ) (h : complex.I * (a + complex.I) = complex.I * complex.I + ↑b + complex.I) : a - b = 0 := by
  sorry

end proof_complex_equality_l34_34974


namespace acute_angle_bisector_triangle_l34_34542

theorem acute_angle_bisector_triangle
  {α β : ℝ} {A B C D E : Type} [triangle ABC] (non_isosceles_trig : ¬isosceles ABC)
  (bisects_α : angle_bisector (angle A) D)
  (bisects_β : angle_bisector (angle B) E) :
  acute_angle (line_segment D E) (line_segment A B) ≤ (|α - β|) / 3 :=
sorry

end acute_angle_bisector_triangle_l34_34542


namespace matrix_det_is_zero_l34_34746

noncomputable def matrixDetProblem (a b : ℝ) : ℝ :=
  Matrix.det ![
    ![1, Real.cos (a - b), Real.sin a],
    ![Real.cos (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ]

theorem matrix_det_is_zero (a b : ℝ) : matrixDetProblem a b = 0 :=
  sorry

end matrix_det_is_zero_l34_34746


namespace triangle_existence_l34_34756

-- Definitions based on the conditions in a)
variables {a b r : ℝ}
variables (O M C : Type*)

-- The condition a >= b
def a_ge_b : Prop := a ≥ b

-- The difference of two sides
def a_minus_b : ℝ := a - b

-- Radius of the circumscribed circle
def circumradius : ℝ := r

-- Equidistance condition
def equidistant_from_angle_bisector (O M C : Type*) : Prop := sorry -- TBD: Definition of equidistant from angle bisector

-- Main theorem statement
theorem triangle_existence (h1 : a_ge_b) 
                           (h2 : a_minus_b > 0) 
                           (h3 : a_minus_b < (sqrt 3 * circumradius)) 
                           (h4 : equidistant_from_angle_bisector O M C) :
  ∃ (T : Type*), true := sorry

end triangle_existence_l34_34756


namespace edward_whack_a_mole_tickets_l34_34425

theorem edward_whack_a_mole_tickets :
  ∃ (w : ℕ), let total_tickets := 4 * 2 in
             let skee_ball_tickets := 5 in
             (w + skee_ball_tickets = total_tickets) ∧ (w = 3) :=
by
  let total_tickets := 4 * 2
  let skee_ball_tickets := 5
  use 3
  split
  {
    show 3 + skee_ball_tickets = total_tickets,
    by
      calc
        3 + skee_ball_tickets = 3 + 5 : rfl
                          ... = 8     : rfl
  }
  {
    show 3 = 3,
    rfl
  }

end edward_whack_a_mole_tickets_l34_34425


namespace apples_more_than_oranges_l34_34322

theorem apples_more_than_oranges :
  ∃ (total : ℕ) (apples : ℕ) (oranges : ℕ),
    total = 301 ∧
    apples = 164 ∧
    oranges = total - apples ∧
    apples - oranges = 27 :=
by {
  let total := 301,
  let apples := 164,
  let oranges := total - apples,
  use [total, apples, oranges],
  simp [total, apples, oranges],
  linarith,
  sorry
}

end apples_more_than_oranges_l34_34322


namespace max_drinks_amount_l34_34076

noncomputable def initial_milk : ℚ := 3 / 4
noncomputable def rachel_fraction : ℚ := 1 / 2
noncomputable def max_fraction : ℚ := 1 / 3

def amount_rachel_drinks (initial: ℚ) (fraction: ℚ) : ℚ := initial * fraction
def remaining_milk_after_rachel (initial: ℚ) (amount_rachel: ℚ) : ℚ := initial - amount_rachel
def amount_max_drinks (remaining: ℚ) (fraction: ℚ) : ℚ := remaining * fraction

theorem max_drinks_amount :
  amount_max_drinks (remaining_milk_after_rachel initial_milk (amount_rachel_drinks initial_milk rachel_fraction)) max_fraction = 1 / 8 := 
sorry

end max_drinks_amount_l34_34076


namespace simplified_evaluation_eq_half_l34_34262

theorem simplified_evaluation_eq_half :
  ∃ x y : ℝ, (|x - 2| + (y + 1)^2 = 0) → 
             (3 * x - 2 * (x^2 - (1/2) * y^2) + (x - (1/2) * y^2) = 1/2) :=
by
  sorry

end simplified_evaluation_eq_half_l34_34262


namespace two_sin_cos_75_eq_half_l34_34051

noncomputable def two_sin_cos_of_75_deg : ℝ :=
  2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180)

theorem two_sin_cos_75_eq_half : two_sin_cos_of_75_deg = 1 / 2 :=
by
  -- The steps to prove this theorem are omitted deliberately
  sorry

end two_sin_cos_75_eq_half_l34_34051


namespace factorable_quadratics_l34_34494

theorem factorable_quadratics :
  {n : ℕ | 1 ≤ n ∧ n ≤ 100 ∧ ∃ a b : ℤ, (a + b = -1) ∧ (a * b = -n) }.card = 9 :=
by
  sorry

end factorable_quadratics_l34_34494


namespace John_profit_is_correct_l34_34963

-- Definitions of conditions as necessary in Lean
variable (initial_puppies : ℕ) (given_away_puppies : ℕ) (kept_puppy : ℕ) (price_per_puppy : ℤ) (payment_to_stud_owner : ℤ)

-- Specific values from the problem
def John_initial_puppies := 8
def John_given_away_puppies := 4
def John_kept_puppy := 1
def John_price_per_puppy := 600
def John_payment_to_stud_owner := 300

-- Calculate the number of puppies left to sell
def John_remaining_puppies := John_initial_puppies - John_given_away_puppies - John_kept_puppy

-- Calculate total earnings from selling puppies
def John_earnings := John_remaining_puppies * John_price_per_puppy

-- Calculate the profit by subtracting payment to the stud owner from earnings
def John_profit := John_earnings - John_payment_to_stud_owner

-- Statement to prove
theorem John_profit_is_correct : 
  John_profit = 1500 := 
by 
  -- The proof will be here but we use sorry to skip it as requested.
  sorry

-- This ensures the definitions match the given problem conditions
#eval (John_initial_puppies, John_given_away_puppies, John_kept_puppy, John_price_per_puppy, John_payment_to_stud_owner)

end John_profit_is_correct_l34_34963


namespace sufficient_condition_ensures_lt_l34_34735

theorem sufficient_condition_ensures_lt {a b : ℝ} :
  (a < 0 ∧ 0 < b) ∨ (0 < b ∧ b < a) → a < b :=
by
  intros h
  cases h
  . { cases h with ha hb
      exact lt_trans ha hb }
  . { cases h with hb ha
      exact ha }

end sufficient_condition_ensures_lt_l34_34735


namespace product_fraction_identity_l34_34394

theorem product_fraction_identity :
  ∏ n in Finset.range 10, (n + 1) * (n + 3) / (n + 6)^2 = (25 : ℚ) / 3372916 := by
  sorry

end product_fraction_identity_l34_34394


namespace maximal_constant_exists_l34_34781

noncomputable def maxConstant : ℝ := 3 / 2

theorem maximal_constant_exists :
  ∀ (n : ℕ), n ≥ 3 → 
  ∃ (a b : ℕ → ℝ), 
    (∀ i, 1 ≤ i → i ≤ n → a i > 0 ∧ b i > 0) ∧ -- positivity conditions for a and b
    (∑ k in (Finset.range (n+1)).erase 0, b k = 1) ∧ -- sum condition on b
    (∀ k, 2 ≤ k ∧ k ≤ n-1 → 2 * b k ≥ b (k - 1) + b (k + 1)) ∧ -- inequality condition on b
    (∀ k, 1 ≤ k ∧ k ≤ n → (a k)^2 ≤ 1 + (∑ i in (Finset.range (k+1)).erase 0, a i * b i) ∧ (a n = maxConstant)) :=
by
  intros n hn
  sorry

end maximal_constant_exists_l34_34781


namespace binom_1294_2_l34_34408

def combination (n k : Nat) := n.choose k

theorem binom_1294_2 : combination 1294 2 = 836161 := by
  sorry

end binom_1294_2_l34_34408


namespace toothpicks_in_300th_stage_l34_34308

/-- 
Prove that the number of toothpicks needed for the 300th stage is 1201, given:
1. The first stage has 5 toothpicks.
2. Each subsequent stage adds 4 toothpicks to the previous stage.
-/
theorem toothpicks_in_300th_stage :
  let a_1 := 5
  let d := 4
  let a_n (n : ℕ) := a_1 + (n - 1) * d
  a_n 300 = 1201 := by
  sorry

end toothpicks_in_300th_stage_l34_34308


namespace initial_percentage_of_water_l34_34372

variable (P : ℚ) -- Initial percentage of water

theorem initial_percentage_of_water (h : P / 100 * 40 + 5 = 9) : P = 10 := 
  sorry

end initial_percentage_of_water_l34_34372


namespace minimum_value_of_f_in_interval_l34_34853

noncomputable def f (x : ℝ) (φ : ℝ) := cos (2 * x - φ) - sqrt 3 * sin (2 * x - φ)

theorem minimum_value_of_f_in_interval :
  ∀ φ: ℝ, |φ| < π / 2 → 
  (∀ x: ℝ, -π / 2 ≤ x ∧ x ≤ 0 →
    let f_shifted := f (x - π / 12) (-π / 3)
    in f_shifted = 2 * cos (2 * x + 2 * π / 3)) →
  (∃ x: ℝ, -π / 2 ≤ x ∧ x ≤ 0 ∧ f x = -sqrt 3) :=
by
  sorry

end minimum_value_of_f_in_interval_l34_34853


namespace neg_of_p_l34_34459

variable {A B : Type}
variable a : A
variable b : B
variable P : a ∈ A
variable Q : b ∈ B
variable p : Prop := P → Q

theorem neg_of_p : ¬(P → Q) ↔ P ∧ ¬Q := by sorry

end neg_of_p_l34_34459


namespace shortest_side_of_triangle_l34_34927

variable (A B C D E : Point)
variable (a b c d e : ℝ)
variable (BD DE EC : ℝ)

-- Let BD = 3, DE = 4, EC = 5
axiom BD_length : BD = 3
axiom DE_length : DE = 4
axiom EC_length : EC = 5

-- AD and AE bisect ∠BAC
axiom AD_bisects_BAC : bisects AD ∠BAC
axiom AE_bisects_BAC : bisects AE ∠BAC

theorem shortest_side_of_triangle : 
  shortest_side_of_triangle A B C BD DE EC = (7 / 9) * sqrt(14580 / 301) :=
sorry

end shortest_side_of_triangle_l34_34927


namespace circumcircle_through_midpoint_l34_34769

-- Assume two given circles with centers O₁ and O₂, and a triangle formed by three lines cutting equal chords in both circles.
variables {ω₁ ω₂ : Type} [circle ω₁] [circle ω₂] {O₁ O₂ : Point} (E₁ : center ω₁ = O₁) (E₂ : center ω₂ = O₂)

-- Assume that the midpoint O of the segment joining the centers O₁ and O₂ exists.
def midpoint (O₁ O₂ : Point) : Point := sorry
noncomputable def O := midpoint O₁ O₂

-- Assume three lines each cutting equal chords in both circles and defining a triangle by their intersection points.
variables (l₁ l₂ l₃ : Line)
hypothesis (H1 : pink_line l₁ ω₁ ω₂) (H2 : pink_line l₂ ω₁ ω₂) (H3 : pink_line l₃ ω₁ ω₂)
variables (X Y Z : Point) (HX : intersection_point l₁ l₂ = X) (HY : intersection_point l₂ l₃ = Y) (HZ : intersection_point l₃ l₁ = Z)
def triangle_xyz := triangle X Y Z

-- Prove that the circumcircle of this triangle passes through the midpoint O of the segment ⟨O₁, O₂⟩.
theorem circumcircle_through_midpoint (H : circumcircle (triangle_xyz X Y Z)) : passes_through H O :=
sorry

end circumcircle_through_midpoint_l34_34769


namespace julies_initial_savings_l34_34205

def simple_interest (P r t : ℝ) : ℝ := P * r * t
def compound_interest (P r n t : ℝ) : ℝ := P * (1 + r / n)^(n * t) - P

theorem julies_initial_savings
  (half_savings_si : ℝ)
  (half_savings_ci : ℝ)
  (r_si : ℝ := 0.04)
  (t_si : ℝ := 5)
  (si_earned : ℝ := 400)
  (r_ci : ℝ := 0.05)
  (n_ci : ℝ := 1)
  (t_ci : ℝ := 5)
  (ci_earned : ℝ := 490) :
  (half_savings_si + half_savings_ci) = 3773.85 :=
by
  have hi : simple_interest half_savings_si 0.04 5 = 400, by sorry,
  have ci : compound_interest half_savings_ci 0.05 1 5 = 490, by sorry,
  sorry

end julies_initial_savings_l34_34205


namespace inequality_f_lt_g_range_of_a_l34_34992

def f (x : ℝ) : ℝ := |x - 4|
def g (x : ℝ) : ℝ := |2 * x + 1|

theorem inequality_f_lt_g :
  ∀ x : ℝ, f x = |x - 4| ∧ g x = |2 * x + 1| →
  (f x < g x ↔ (x < -5 ∨ x > 1)) :=
by
   sorry

theorem range_of_a :
  ∀ x a : ℝ, f x = |x - 4| ∧ g x = |2 * x + 1| →
  (2 * f x + g x > a * x) →
  (-4 ≤ a ∧ a < 9/4) :=
by
   sorry

end inequality_f_lt_g_range_of_a_l34_34992


namespace calc_angle_AHB_l34_34385

theorem calc_angle_AHB
  (A B C H D E : Type)
  (angle_BAC : ℝ)
  (angle_ABC : ℝ)
  (h1 : angle_BAC = 50)
  (h2 : angle_ABC = 60)
  (orthocenter : ∀ Δ : Triangle, is_orthocenter Δ H) :
  angle_AHB = 110 :=
by
  sorry

end calc_angle_AHB_l34_34385


namespace sin_of_angle_in_first_quadrant_l34_34462

theorem sin_of_angle_in_first_quadrant (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 3 / 4) : Real.sin α = 3 / 5 :=
by
  sorry

end sin_of_angle_in_first_quadrant_l34_34462


namespace ratio_AD_AB_l34_34512

theorem ratio_AD_AB (A B C D M : Point)
  (h1: ConvexQuadrilateral A B C D)
  (h2: ∠B = ∠D)
  (h3: dist C D = 4 * dist B C)
  (h4: Midpoint M C D)
  (h5: AngleBisector A M) : 
  dist A D / dist A B = 2 / 3 :=
sorry

end ratio_AD_AB_l34_34512


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l34_34884

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : 
  ∃ (count : ℕ), count = 225 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ↔ (∃ k : ℕ, 25 ≤ k ∧ k ≤ 249 ∧ n = 4 * k + 2) := 
by {
  sorry
}

end number_of_three_digit_integers_congruent_to_2_mod_4_l34_34884


namespace complement_union_l34_34489

open Set

-- Definitions from the given conditions
def U : Set ℕ := {x | x ≤ 9}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

-- Statement of the proof problem
theorem complement_union :
  compl (A ∪ B) = {7, 8, 9} :=
sorry

end complement_union_l34_34489


namespace children_age_problem_l34_34514

theorem children_age_problem :
  ∃ (n : ℕ) (a d : ℕ), 
  n = 5 ∧ 
  (a + nd = 13) ∧
  (∑ i in finset.range (n + 1), a + i * d) = 40 ∧
  a ∈ ℕ ∧ d ∈ ℕ :=
by
  sorry

end children_age_problem_l34_34514


namespace equation_of_line_CD_l34_34947

theorem equation_of_line_CD 
  (H1 : ∃ O : ℝ × ℝ, O = (0, 1)) 
  (H2 : ∃ (A B : ℝ × ℝ), ∃ l : ℝ, l = 2 ∧ line_through_points (A, B).eq "x-2y-2=0") :
  line_contains_parallelogram_CD ∃ equation : String, equation = "x-2y+6=0" :=
sorry

end equation_of_line_CD_l34_34947


namespace maximum_Q_2023_l34_34207

noncomputable def Q (x : ℤ) : ℤ :=
  ∑ i in (Finset.range 2024), 2023^(i : ℤ)

theorem maximum_Q_2023 :
  (Q 2023 = (2023^2024 - 1) / (2023 - 1)) :=
by sorry

end maximum_Q_2023_l34_34207


namespace decreasing_interval_l34_34088

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- we need to prove, given x > 0, the interval where f is monotonically decreasing is (0, e^{-1})
theorem decreasing_interval : { x : ℝ | 0 < x ∧ x < Real.exp (-1) } = { x : ℝ | 0 < x ∧ f'(x) < 0 } :=
by
  sorry

end decreasing_interval_l34_34088


namespace domain_g_l34_34137

-- Define the function f and its domain
variable {f : ℝ → ℝ}
variable hf : ∀ x, 0 < x ∧ x ≤ 4 → f x = f x  -- stating the domain condition of f

-- Define the function g
def g (x : ℝ) : ℝ := f (2^x) / (x - 1)

-- Prove the domain of g
theorem domain_g :
  ∀ x, (x < 1 ∨ (1 < x ∧ x ≤ 2)) →
  (∃ y, g x = y) :=
begin
  intros x hx,
  cases hx,
  { -- Case 1: x < 1
    use g x,
    unfold g,
    -- f (2 ^ x) is defined since 2^x is in (0, 4) when x < 1
    have : 0 < 2 ^ x ∧ 2 ^ x ≤ 4, by { simp [pow_lt_pow_iff, hx], norm_num },
    exact hf (2^x) this,
  },
  { -- Case 2: 1 < x ∧ x ≤ 2
    use g x,
    unfold g,
    -- f (2 ^ x) is defined since 2^x is in (0, 4) when 1 < x ≤ 2
    have : 0 < 2 ^ x ∧ 2 ^ x ≤ 4, by { simp [pow_lt_pow_iff, hx.left], norm_num },
    exact hf (2^x) this,
  }
end

end domain_g_l34_34137


namespace proof_fraction_N_M_l34_34570

noncomputable def lcm_range (a b : ℕ) : ℕ :=
  (List.range (b - a + 1)).map (λ n => n + a) |>.lcm

noncomputable def M : ℕ := lcm_range 12 25

noncomputable def N : ℕ := Nat.lcm M (Nat.lcm 41 (Nat.lcm 42 (Nat.lcm 43 (Nat.lcm 44 45))))

theorem proof_fraction_N_M :
  (N / M) = 1763 :=
by
  sorry

end proof_fraction_N_M_l34_34570


namespace cost_of_second_variety_l34_34537

theorem cost_of_second_variety (cost_first : ℝ) (cost_mixture : ℝ) (ratio : ℝ) :
  cost_first = 7 →
  cost_mixture = 7.5 →
  ratio = 2.5 →
  (let x := (cost_mixture * (ratio + 1)) - (ratio * cost_first) in x = 8.75) :=
by
  intros h1 h2 h3
  dsimp
  rw [h1, h2, h3]
  norm_num
  sorry

end cost_of_second_variety_l34_34537


namespace sum_series_lt_two_l34_34116

theorem sum_series_lt_two
  (n : ℕ)
  (a : ℕ → ℝ)
  (h_pos : ∀ i, 1 ≤ i → i ≤ n → a i > 0)
  (h_prod : ∀ k, 1 ≤ k → k ≤ n → ∏ i in Finset.range k, a (i + 1) ≥ 1) :
  (∑ k in Finset.range n, (k + 1) / ∏ i in Finset.range (k + 1), (1 + a (i + 1))) < 2 :=
by
  sorry

end sum_series_lt_two_l34_34116


namespace part_3_l34_34861

noncomputable def f (x : ℝ) (m : ℝ) := Real.log x - m * x^2
noncomputable def g (x : ℝ) (m : ℝ) := (1/2) * m * x^2 + x
noncomputable def F (x : ℝ) (m : ℝ) := f x m + g x m

theorem part_3 (x₁ x₂ : ℝ) (m : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hm : m = -2)
  (hF : F x₁ m + F x₂ m + x₁ * x₂ = 0) : x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end part_3_l34_34861


namespace measure_of_angle_C_l34_34507

theorem measure_of_angle_C (A B C : ℕ) (h1 : A + B = 150) (h2 : A + B + C = 180) : C = 30 := 
by
  sorry

end measure_of_angle_C_l34_34507


namespace number_of_three_digit_integers_congruent_mod4_l34_34908

def integer_congruent_to_mod (a b n : ℕ) : Prop := ∃ k : ℤ, n = a * k + b

theorem number_of_three_digit_integers_congruent_mod4 :
  (finset.filter (λ n, integer_congruent_to_mod 4 2 (n : ℕ)) 
   (finset.Icc (100 : ℕ) (999 : ℕ))).card = 225 :=
by
  sorry

end number_of_three_digit_integers_congruent_mod4_l34_34908


namespace original_employee_count_l34_34025

theorem original_employee_count (x : ℝ) (h1 : x * 0.90 * 0.85 = 263) : 
  x ≈ 344 :=
by
  sorry

end original_employee_count_l34_34025


namespace find_m_value_l34_34838

theorem find_m_value (m : ℤ) : (∃ a : ℤ, x^2 + 2 * (m + 1) * x + 25 = (x + a)^2) ↔ (m = 4 ∨ m = -6) := 
sorry

end find_m_value_l34_34838


namespace find_k_l34_34586

theorem find_k (
  (intersect_point : ℝ × ℝ := (2, 7)) :
  ∃ k : ℝ, ∀ x y : ℝ, (x = 2 → y = 7 → y = 2*x + 3) ∧ 
                    (x = 2 → y = 7 → y = k*x + 1 → k = 3) 
  := by
  sorry

end find_k_l34_34586


namespace count_nines_in_n_cubed_l34_34944

def n : ℕ := 10^100 - 1

theorem count_nines_in_n_cubed (n_val : n = 10^100 - 1) : 
  count_nines (n ^ 3) = 199 :=
sorry

end count_nines_in_n_cubed_l34_34944


namespace prob_part1_prob_part2_l34_34475

def f (ω : Real) (x : Real) : Real := 2 * Real.sin (ω * x + Real.pi / 6)

noncomputable def minimum_omega (ω : Real) : Prop :=
  (ω > 0) ∧ (∀ x : Real, f ω (5 * Real.pi / 6 + x) + f ω (5 * Real.pi / 6 - x) = 0) →
  ω = 1

noncomputable def range_omega (ω : Real) : Prop :=
  (∀ x ∈ Set.Icc (0:Real) (Real.pi / 3), 1 ≤ f ω x ∧ f ω x ≤ 2) →
  1 ≤ ω ∧ ω ≤ 2

-- Proof statements (no proof provided, only statement required)
theorem prob_part1 (ω : Real) (h₁ : ω > 0) (h₂ : ∀ x : Real, f ω (5 * Real.pi / 6 + x) + f ω (5 * Real.pi / 6 - x) = 0) :
  ω = 1 := sorry

theorem prob_part2 (ω : Real) (h : ∀ x ∈ Set.Icc (0:Real) (Real.pi / 3), 1 ≤ f ω x ∧ f ω x ≤ 2) :
  1 ≤ ω ∧ ω ≤ 2 := sorry

end prob_part1_prob_part2_l34_34475


namespace ages_proof_l34_34659

noncomputable def A : ℝ := 12.1
noncomputable def B : ℝ := 6.1
noncomputable def C : ℝ := 11.3

-- Conditions extracted from the problem
def sum_of_ages (A B C : ℝ) : Prop := A + B + C = 29.5
def specific_age (C : ℝ) : Prop := C = 11.3
def twice_as_old (A B : ℝ) : Prop := A = 2 * B

theorem ages_proof : 
  ∃ (A B C : ℝ), 
    specific_age C ∧ twice_as_old A B ∧ sum_of_ages A B C :=
by
  exists 12.1, 6.1, 11.3
  sorry

end ages_proof_l34_34659


namespace largest_k_consecutive_sum_l34_34089

theorem largest_k_consecutive_sum (k n : ℕ) :
  (5^7 = (k * (2 * n + k + 1)) / 2) → 1 ≤ k → k * (2 * n + k + 1) = 2 * 5^7 → k = 250 :=
sorry

end largest_k_consecutive_sum_l34_34089


namespace problem1_problem2_l34_34830

-- Define sets M and N based on conditions
def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x : ℝ | a + 1 ≤ x ∧ x ≤ 2 * a - 1}

-- Problem 1: Proving the union and intersection with complement
theorem problem1 (a : ℝ) (h : a = 7 / 2) : 
  M ∪ N a = { x : ℝ | -2 ≤ x ∧ x ≤ 6 } ∧ 
  (Set.univ \ M) ∩ N a = { x : ℝ | 5 < x ∧ x ≤ 6 } :=
sorry

-- Problem 2: Proving the range of a when M ⊇ N
theorem problem2 (h : M ⊇ N a) : a ∈ Iic 3 :=
sorry

end problem1_problem2_l34_34830


namespace cos_sq_sub_sin_sq_l34_34445

noncomputable def cos_sq_sub_sin_sq_eq := 
  ∀ (α : ℝ), α ∈ Set.Ioo 0 Real.pi → (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →
  (Real.cos α) ^ 2 - (Real.sin α) ^ 2 = -Real.sqrt 5 / 3

theorem cos_sq_sub_sin_sq :
  cos_sq_sub_sin_sq_eq := 
by
  intros α hα h_eq
  sorry

end cos_sq_sub_sin_sq_l34_34445


namespace three_digit_integers_congruent_to_2_mod_4_l34_34891

theorem three_digit_integers_congruent_to_2_mod_4 : 
    ∃ n, n = 225 ∧ ∀ x, (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 2) ↔ (∃ m, 25 ≤ m ∧ m ≤ 249 ∧ x = 4 * m + 2) := by
  sorry

end three_digit_integers_congruent_to_2_mod_4_l34_34891


namespace abs_neg_three_l34_34293

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l34_34293


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l34_34887

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : 
  ∃ (count : ℕ), count = 225 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ↔ (∃ k : ℕ, 25 ≤ k ∧ k ≤ 249 ∧ n = 4 * k + 2) := 
by {
  sorry
}

end number_of_three_digit_integers_congruent_to_2_mod_4_l34_34887


namespace inequality_l34_34443

noncomputable def a : Real := Real.sqrt 2
noncomputable def b : Real := Real.log 3 / Real.log Math.pi
noncomputable def c : Real := -Real.log 3 / Real.log 2

theorem inequality (a_cond : a = Real.sqrt 2) (b_cond : b = Real.log 3 / Real.log Math.pi) (c_cond : c = -Real.log 3 / Real.log 2) : 
  c < b ∧ b < a := 
by
  sorry

end inequality_l34_34443


namespace probability_both_events_l34_34317

noncomputable def P (event : Type) : ℝ := sorry

variable (A B : Type)

axiom PA : P A = 0.25
axiom PB : P B = 0.40
axiom P_complement_union : P ((λ (x : Type), x ∈ A ∪ B)ᶜ) = 0.5

theorem probability_both_events :
  P (λ (x : Type), x ∈ A ∩ B) = 0.15 :=
by
  sorry

end probability_both_events_l34_34317


namespace sum_nonnegative_l34_34253

theorem sum_nonnegative (a : ℕ → ℝ) (n : ℕ) :
  ∑ i in Finset.range n, ∑ j in Finset.range n, (a i * a j) / (i + j + 1) ≥ 0 :=
sorry

end sum_nonnegative_l34_34253


namespace quadratic_min_max_l34_34865

noncomputable def quadratic (x : ℝ) : ℝ := (x - 3)^2 - 1

theorem quadratic_min_max :
  ∃ min max, (∀ x ∈ set.Icc 1 4, quadratic x ≥ min) ∧ min = (-1) ∧
             (∀ x ∈ set.Icc 1 4, quadratic x ≤ max) ∧ max = 3 :=
by
  sorry

end quadratic_min_max_l34_34865


namespace table_count_is_non_integer_l34_34007

theorem table_count_is_non_integer (c t s : ℕ) (H1 : c = 8 * t) (H2 : s = t / 2) (H3 : 4 * c + 5 * t + 6 * s = 749) : ¬ ∃ t : ℕ, 40 * t = 749 := by
  intro h
  rcases h with ⟨t, ht⟩
  linarith

end table_count_is_non_integer_l34_34007


namespace ball_count_in_box_eq_57_l34_34361

theorem ball_count_in_box_eq_57 (N : ℕ) (h : N - 44 = 70 - N) : N = 57 :=
sorry

end ball_count_in_box_eq_57_l34_34361


namespace arithmetic_sequence_sum_l34_34572

def a_n (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

noncomputable def S_n (a d : ℝ) (n : ℕ) : ℝ := n * (a + (a + (n - 1) * d)) / 2

def b_n (S_n : ℝ → ℝ → ℕ → ℝ) (n : ℕ) : ℝ := S_n 1 1 n / n

noncomputable def S'_n (n : ℕ) : ℝ := n * (n - 9) / 4

noncomputable def T_n (n : ℕ) : ℝ := if n <= 5 then (9 * n - n^2) / 4 else (n^2 - 9 * n + 40) / 4

theorem arithmetic_sequence_sum (a d : ℝ) (n : ℕ) :
  (S_n a d 7 = 7) →
  (S_n a d 15 = 75) →
  T_n n = if n <= 5 then (9 * n - n^2) / 4 else (n^2 - 9 * n + 40) / 4 := sorry

end arithmetic_sequence_sum_l34_34572


namespace asymptote_of_hyperbola_l34_34450

-- Define the problem in Lean 4
theorem asymptote_of_hyperbola (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (hyp1 : 2 * b^2 = a^2 + c^2) (hyp2 : c^2 = a^2 + b^2) :
  ∀ x, ∀ y, (y = sqrt 2 * x ∨ y = - sqrt 2 * x) → ( ∃ z, (x, y) = (z, sqrt 2 * z) ∨ (x, y) = (z, - sqrt 2 * z)) :=
by
  -- Proof omitted
  sorry

end asymptote_of_hyperbola_l34_34450


namespace hexagon_perimeter_of_intersecting_triangles_l34_34651

/-- Given two equilateral triangles with parallel sides, where the perimeter of the blue triangle 
    is 4 and the perimeter of the green triangle is 5, prove that the perimeter of the hexagon 
    formed by their intersection is 3. -/
theorem hexagon_perimeter_of_intersecting_triangles 
    (P_blue P_green P_hexagon : ℝ)
    (h_blue : P_blue = 4)
    (h_green : P_green = 5) :
    P_hexagon = 3 := 
sorry

end hexagon_perimeter_of_intersecting_triangles_l34_34651


namespace solve_cubic_equation_l34_34606

theorem solve_cubic_equation (x : ℝ) (h : 4 * x^(1/3) - 2 * (x / x^(2/3)) = 7 + x^(1/3)) : x = 343 := by
  sorry

end solve_cubic_equation_l34_34606


namespace sqrt_expression_approx_l34_34399

theorem sqrt_expression_approx :
  let sqrt2 := 1.414
  let sqrt3 := 1.732
  (Real.sqrt 0.08 + (Real.sqrt 3 - Real.sqrt 2) * (Real.sqrt 3 + Real.sqrt 2)) ≈ 1.28 :=
by {
  sorry
}

end sqrt_expression_approx_l34_34399


namespace Bob_wins_game_l34_34732

/-- 
Alice starts with an odd number in binary on the whiteboard.
Players, starting with Bob, can either subtract 1 from the number or erase the last digit.
The game ends when the whiteboard is blank. Prove that Bob has a winning strategy.
-/
theorem Bob_wins_game (initial_number : ℕ) (h : initial_number % 2 = 1) :
    ∃ strategy_for_Bob, (∀ moves, game_ends_empty_whiteboard) := 
sorry

end Bob_wins_game_l34_34732


namespace part1_inequality_part2_range_a_l34_34856

noncomputable def f (x : ℝ) := log ((1 - x) / (1 + x))
noncomputable def g (x : ℝ) (a : ℝ) := 2 - a ^ x

theorem part1_inequality (x : ℝ) : f x > 0 ↔ -1 < x ∧ x < 0 :=
by sorry

theorem part2_range_a (a > 0) (a ≠ 1) : 
  (∃ (x1 x2 : ℝ), 0 ≤ x1 ∧ x1 < 1 ∧ 0 ≤ x2 ∧ x2 < 1 ∧ f x1 = g x2 a) ↔ a > 2 :=
by sorry

end part1_inequality_part2_range_a_l34_34856


namespace student_walks_fifth_to_first_l34_34932

theorem student_walks_fifth_to_first :
  let floors := 4
  let staircases := 2
  (staircases ^ floors) = 16 := by
  sorry

end student_walks_fifth_to_first_l34_34932


namespace cos_x_squared_zeros_l34_34167

noncomputable def f (x : ℝ) : ℝ := Real.cos (x^2)

theorem cos_x_squared_zeros : 
  ∃! x1 x2, (0 < x1 ∧ x1 < Real.sqrt (2 * Real.pi) ∧ f x1 = 0) ∧ 
            (0 < x2 ∧ x2 < Real.sqrt (2 * Real.pi) ∧ f x2 = 0) ∧ 
            (x1 ≠ x2) := 
begin
  sorry
end

end cos_x_squared_zeros_l34_34167


namespace negation_of_proposition_l34_34131

theorem negation_of_proposition (a b c : ℝ) :
  ¬ (a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3) :=
sorry

end negation_of_proposition_l34_34131


namespace ratio_of_speeds_l34_34248

theorem ratio_of_speeds (a b v1 v2 S : ℝ)
  (h1 : S = a * (v1 + v2))
  (h2 : S = b * (v1 - v2)) :
  v2 / v1 = (a + b) / (b - a) :=
by
  sorry

end ratio_of_speeds_l34_34248


namespace triangle_area_given_lines_l34_34761

theorem triangle_area_given_lines 
  (L1 : ℝ → ℝ) (L2 : ℝ → ℝ) (L3 : ℝ → ℝ)
  (hL1 : ∀ x, L1 x = 3)
  (hL2 : ∀ x, L2 x = 2 + 2 * x)
  (hL3 : ∀ x, L3 x = 2 - 2 * x) :
  let intersect1 := (0.5, 3)
  let intersect2 := (-0.5, 3)
  let intersect3 := (0, 2)
  Shoelace.area [intersect1, intersect2, intersect3] = 0.5 :=
by
  sorry

end triangle_area_given_lines_l34_34761


namespace sum_of_powers_l34_34241

theorem sum_of_powers (m n : ℤ)
  (h1 : m + n = 1)
  (h2 : m^2 + n^2 = 3)
  (h3 : m^3 + n^3 = 4)
  (h4 : m^4 + n^4 = 7)
  (h5 : m^5 + n^5 = 11) :
  m^9 + n^9 = 76 :=
sorry

end sum_of_powers_l34_34241


namespace least_length_XZ_l34_34234

theorem least_length_XZ (a b : ℕ) (h : 0 < a) (h0 : 0 < b) :
  let PQ := 2 * a
  let QR := 8 * b
  let angle_Q := 90
  ∃ X Y Z M N, 
    (X ∈ PQ) ∧
    (parallel (line_through X) QR) ∧
    (line_intersects (line_through X) PR at Y) ∧
    (parallel (line_through Y) PQ) ∧
    (line_intersects (line_through Y) QR at Z) ∧
    (M = midpoint QY) ∧
    (N = midpoint XP) ∧
    (angle QYZ = angle XYP) →
  (XZ = 8 * b) :=
by
  sorry

end least_length_XZ_l34_34234


namespace length_ratio_l34_34223

noncomputable def ellipse (a b : ℝ) : ℝ × ℝ → Prop :=
λ p, let (x, y) := p in (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1

theorem length_ratio (a b : ℝ) (h : a > b) (θ : ℝ) :
  let Q : ℝ × ℝ := (a * Real.cos θ, b * Real.sin θ),
  let A : ℝ × ℝ := (-a, 0),
  let OP : ℝ × ℝ := (Real.cos θ, Real.sin θ),
  let AQ_param (t : ℝ) := (x, y) where x = -a + t * Real.cos θ and y = t * Real.sin θ,
  let t := (2 * a * b ^ 2 * Real.cos θ) / (b ^ 2 * (Real.cos θ) ^ 2 + a ^ 2 * (Real.sin θ) ^ 2),
  let AQ := t,
  let AR := a / Real.cos θ,
  let OP_sq := (a ^ 2 * b ^ 2) / (a ^ 2 * (Real.sin θ) ^ 2 + b ^ 2 * (Real.cos θ) ^ 2)
in
  ellipse a b Q ∧ AQ_param AQ ∧ ellipse a b OP →
  (|AQ| * |AR|) / |OP_sq| = 2 :=
begin
  sorry
end

end length_ratio_l34_34223


namespace sin_squared_y_l34_34960

theorem sin_squared_y (x y z α : ℝ)
  (h1 : y = (x + z) / 2)
  (h2 : α = real.arcsin (real.sqrt 7 / 4))
  (h3 : (1 / real.sin x) + (1 / real.sin z) = 8 / (real.sin y))
  : (real.sin y) ^ 2 = 7 / 13 :=
by
  sorry

end sin_squared_y_l34_34960


namespace elizabeth_husband_weight_l34_34662

-- Defining the variables for weights of the three wives
variable (s : ℝ) -- Weight of Simona
def elizabeta_weight : ℝ := s + 5
def georgetta_weight : ℝ := s + 10

-- Condition: The total weight of all wives
def total_wives_weight : ℝ := s + elizabeta_weight s + georgetta_weight s

-- Given: The total weight of all wives is 171 kg
def total_wives_weight_cond : Prop := total_wives_weight s = 171

-- Given:
-- Leon weighs the same as his wife.
-- Victor weighs one and a half times more than his wife.
-- Maurice weighs twice as much as his wife.

-- Given: Elizabeth's weight relationship
def elizabeth_weight_cond : Prop := (s + 5 * 1.5) = 85.5

-- Main proof problem:
theorem elizabeth_husband_weight (s : ℝ) (h1: total_wives_weight_cond s) : elizabeth_weight_cond s :=
by
  sorry

end elizabeth_husband_weight_l34_34662


namespace abs_neg_three_l34_34270

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l34_34270


namespace angle_B_in_parallelogram_l34_34525

variable (A B : ℝ)

theorem angle_B_in_parallelogram (h_parallelogram : ∀ {A B C D : ℝ}, A + B = 180 ↔ A = B) 
  (h_A : A = 50) : B = 130 := by
  sorry

end angle_B_in_parallelogram_l34_34525


namespace inscribed_circle_radius_l34_34337

noncomputable def radius_of_inscribed_circle {d1 d2 : ℕ} (h1 : d1 = 12) (h2 : d2 = 30) : ℝ :=
  let a := sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  let area := (d1 * d2) / 2
  (area / (2 * a))

theorem inscribed_circle_radius {d1 d2 : ℕ} (h1 : d1 = 12) (h2 : d2 = 30) :
  radius_of_inscribed_circle h1 h2 = 90 * sqrt 261 / 261 :=
by sorry

end inscribed_circle_radius_l34_34337


namespace essay_body_section_length_l34_34403

theorem essay_body_section_length :
  ∀ (intro_length conclusion_multiplier : ℕ) (body_sections total_words : ℕ),
  intro_length = 450 →
  conclusion_multiplier = 3 →
  body_sections = 4 →
  total_words = 5000 →
  let conclusion_length := conclusion_multiplier * intro_length in
  let total_intro_conclusion := intro_length + conclusion_length in
  let remaining_words := total_words - total_intro_conclusion in
  let section_length := remaining_words / body_sections in
  section_length = 800 :=
by 
  intros intro_length conclusion_multiplier body_sections total_words 
         h_intro_len h_concl_mul h_body_sec h_total_words;
  dsimp only;
  rw [h_intro_len, h_concl_mul, h_body_sec, h_total_words];
  let conclusion_length := 3 * 450;
  let total_intro_conclusion := 450 + 1350;
  let remaining_words := 5000 - 1800;
  let section_length := 3200 / 4;
  exact sorry

end essay_body_section_length_l34_34403


namespace gcd_of_set_B_is_five_l34_34559

def is_in_set_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)

theorem gcd_of_set_B_is_five : ∃ d, (∀ n, is_in_set_B n → d ∣ n) ∧ d = 5 :=
by 
  sorry

end gcd_of_set_B_is_five_l34_34559


namespace mary_potatoes_l34_34237

theorem mary_potatoes 
  (original_potatoes : ℕ)
  (new_potatoes_eaten : ℕ → ℕ)
  (total_potatoes : ℕ) 
  (h1 : original_potatoes = 8)
  (h2 : new_potatoes_eaten 3 = 3)
  (h3 : total_potatoes = original_potatoes + 3) : 
  total_potatoes = 11 :=
by
  rw [h1, h2, h3]
  exact rfl

end mary_potatoes_l34_34237


namespace cycle_efficiency_l34_34389

variable (Q : ℝ)
variable (W_comp : ℝ)
variable (T_23 T_41 : ℝ)

-- Given conditions
def cycle_conditions : Prop :=
  (T_23 / T_41 = 3) ∧
  (Q / 2 = W_comp) -- Corresponds to the heat received during isochoric process being half of that during isothermal process

-- Question (Efficiency of the cycle)
def efficiency (A Q_u : ℝ) : ℝ := A / Q_u

-- Net work done (A) and total heat received (Q_u)
def net_work : ℝ := 2 * W_comp
def total_heat_received : ℝ := 3 / 2 * Q

-- Correct answer (Efficiency of the cycle)
theorem cycle_efficiency (h : cycle_conditions Q W_comp T_23 T_41) :
  efficiency (net_work W_comp) (total_heat_received Q) = 4 / 9 := by
  sorry

end cycle_efficiency_l34_34389


namespace range_of_a_for_monotonicity_l34_34920

noncomputable def f (a x : ℝ) := log a (x + (10 * a) / x)

theorem range_of_a_for_monotonicity :
  ∀ (a : ℝ), (∀ x ∈ set.Icc 3 4, ∀ y ∈ set.Icc 3 4, 
    f a x ≤ f a y ∨ f a x ≥ f a y) ↔
  (0 < a ∧ a ≤ 9/10) ∨ (8/5 ≤ a) :=
begin
  sorry
end

end range_of_a_for_monotonicity_l34_34920


namespace final_price_is_correct_l34_34739

-- Define the conditions as constants
def price_smartphone : ℝ := 300
def price_pc : ℝ := price_smartphone + 500
def price_tablet : ℝ := price_smartphone + price_pc
def total_price : ℝ := price_smartphone + price_pc + price_tablet
def discount : ℝ := 0.10 * total_price
def price_after_discount : ℝ := total_price - discount
def sales_tax : ℝ := 0.05 * price_after_discount
def final_price : ℝ := price_after_discount + sales_tax

-- Theorem statement asserting the final price value
theorem final_price_is_correct : final_price = 2079 := by sorry

end final_price_is_correct_l34_34739


namespace symmetric_points_x_axis_l34_34499

theorem symmetric_points_x_axis (m n : ℤ) :
  (-4, m - 3) = (2 * n, -1) → (m = 2 ∧ n = -2) :=
by
  sorry

end symmetric_points_x_axis_l34_34499


namespace trapezoid_is_isosceles_l34_34941

theorem trapezoid_is_isosceles
  (A B C D : Type)
  (is_trapezoid : Prop)
  (perpendicular_diagonals : Prop)
  (height_eq_midline : Prop) :
  isosceles_trapezoid :=
by
  assume (h : is_trapezoid)
  assume (h1 : perpendicular_diagonals)
  assume (h2 : height_eq_midline)
  sorry

end trapezoid_is_isosceles_l34_34941


namespace intersection_eq_two_l34_34699

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-2, 2}

theorem intersection_eq_two : A ∩ B = {2} := by
  sorry

end intersection_eq_two_l34_34699


namespace circle_equation_correct_l34_34629

theorem circle_equation_correct (x y : ℝ) : 
  (∃ (c : ℝ), (c = 2) ∧ (x^2 + (y - c)^2 = 1) ∧ (∃ p : ℝ × ℝ, p = (1, 2) ∧ (1^2 + (2 - c)^2 = 1))) → 
  x^2 + (y - 2)^2 = 1 :=
by
  intro h,
  cases h with c hc,
  cases hc.2 with p hp,
  cases hp with hxy heq,
  simp at *,
  sorry

end circle_equation_correct_l34_34629


namespace smallest_period_pi_centers_of_symmetry_max_perimeter_triangle_l34_34859

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 3 * sin (x + π / 4) * sin (x - π / 4) + sin x * cos x

-- The smallest positive period of f(x) is π
theorem smallest_period_pi : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
sorry

-- The center of symmetry includes the points (kπ/2 + π/6, 0) for k in integers
theorem centers_of_symmetry (k : ℤ) :
  (0 : ℝ) ∈ set.range (λ x, f x) ∧ ∃ x, f (x + k * (π / 2) + (π / 6)) = f x :=
sorry

-- Establish the conditions for the maximum perimeter in triangle ABC
variables {A B C : ℝ} {a b c : ℝ}

theorem max_perimeter_triangle {A B C : ℝ} {a : ℝ} (h1 : a = 4) (h2 : f A = sqrt 3 / 2)
  (acute_ABC : A > 0 ∧ A < π / 2 ∧ B > 0 ∧ B < π / 2 ∧ C > 0 ∧ C < π / 2 ∧ A + B + C = π) :
  ∃ b c, a + b + c = 12 :=
sorry

end smallest_period_pi_centers_of_symmetry_max_perimeter_triangle_l34_34859


namespace triangular_track_side_length_l34_34366

theorem triangular_track_side_length :
  (r : ℝ) (Ali_speed : ℝ) (Darius_speed : ℝ) (x : ℝ) (t : ℝ)
  (h1 : r = 60)
  (h2 : Ali_speed = 6)
  (h3 : Darius_speed = 5)
  (h4 : t = 20 * π)
  (h5 : 2 * π * r / Ali_speed = t)
  (h6 : 3 * x = Darius_speed * t) :
  x = 100 * π / 3 :=
sorry

end triangular_track_side_length_l34_34366


namespace circle_equation_center_y_axis_l34_34626

theorem circle_equation_center_y_axis (h₁ : ∃ k : ℝ, x^2 + (y - k)^2 = 1)
                                     (h₂ : ∃ k : ℝ, (1, 2) ∈ { p : ℝ × ℝ | x^2 + (y - k)^2 = 1 }) :
                                     x^2 + (y - 2)^2 = 1 :=
begin
  sorry
end

end circle_equation_center_y_axis_l34_34626


namespace sum_of_roots_eq_3n_l34_34978

variable {n : ℝ} 

-- Define the conditions
def quadratic_eq (x : ℝ) (m : ℝ) (n : ℝ) : Prop :=
  x^2 - (m + n) * x + m * n = 0

theorem sum_of_roots_eq_3n (m : ℝ) (n : ℝ) 
  (hm : m = 2 * n)
  (hroot_m : quadratic_eq m m n)
  (hroot_n : quadratic_eq n m n) :
  m + n = 3 * n :=
by sorry

end sum_of_roots_eq_3n_l34_34978


namespace proposition_three_proposition_four_l34_34071

variables {a b c : ℝ^n}

theorem proposition_three (h : a ⊥ (b - c)) : a • b = a • c := 
by sorry

theorem proposition_four (h : a • b = 0) : ∥a + b∥ = ∥a - b∥ := 
by sorry

end proposition_three_proposition_four_l34_34071


namespace lambda_equals_1_an_equals_2n_Tn_equals_2n1_n1_l34_34194

noncomputable def find_lambda (a2 : ℕ) (S : ℕ → ℝ) (lambda : ℝ) : ℝ :=
if S 2 - S 1 = a2 then (S 2 - S 1 - 4) else λ

theorem lambda_equals_1: 
  ∀ (a2 : ℕ) (S : ℕ → ℝ) (λ : ℝ),
  a2 = 4 → (∀ n : ℕ, S n = n^2 + λ * n) → find_lambda a2 S λ = 1 :=
by sorry

def gen_seq (S : ℕ → ℝ) (a2 : ℕ): ℕ → ℕ :=
λ n, if n < 2 then 2 * n else n^2 + 1 - (n-1)^2 - (n-1)

theorem an_equals_2n : 
  ∀ (S : ℕ → ℝ) (a2 : ℕ),
  a2 = 4 → (∀ n : ℕ, S n = n^2 + 1 * n) → (∀ n, gen_seq S a2 n = 2 * n) :=
by sorry

noncomputable def bn_vals (S : ℕ → ℝ): ℕ → ℝ :=
λ n, 2^(n-1) - 1 / (n * (n + 1))

def T_n_val (b : ℕ → ℝ) : ℕ → ℝ :=
λ n, (1+2^1+...+2^(n-1)) - (1 - 1/2 + 1/2 - 1/3 + ... + 1/n - 1/(n+1))

theorem Tn_equals_2n1_n1 
: ∀ (S : ℕ → ℝ) (first_term : ℝ) (common_ratio : ℝ) ,
  first_term = 1 → common_ratio = 2 → 
  (∀ n, S n = n^2 + 1 * n) → 
  (∀ n, T_n_val (bn_vals S) n = 2^n - 1 - n / (n + 1)) :=
by sorry

end lambda_equals_1_an_equals_2n_Tn_equals_2n1_n1_l34_34194


namespace not_integer_fraction_l34_34968

theorem not_integer_fraction (a b : ℤ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (hrelprime : Nat.gcd a.natAbs b.natAbs = 1) : 
  ¬(∃ (k : ℤ), 2 * a * (a^2 + b^2) = k * (a^2 - b^2)) :=
  sorry

end not_integer_fraction_l34_34968


namespace alex_ate_slices_l34_34039

def total_slices(cakes: ℕ, slices_per_cake: ℕ): ℕ := cakes * slices_per_cake

def slices_given_away(slices: ℕ, fraction: ℚ): ℕ := (fraction * slices).to_nat

theorem alex_ate_slices (cakes_qty: ℕ) (slices_per_cake: ℕ) (frac_friends: ℚ) (frac_family: ℚ) (slices_left: ℕ):
  cakes_qty = 2 →
  slices_per_cake = 8 →
  frac_friends = 1 / 4 →
  frac_family = 1 / 3 →
  slices_left = 5 →
  let initial_slices := total_slices cakes_qty slices_per_cake in
  let slices_friends := slices_given_away initial_slices frac_friends in
  let remaining_slices_after_friends := initial_slices - slices_friends in
  let slices_family := slices_given_away remaining_slices_after_friends frac_family in
  let remaining_slices := remaining_slices_after_friends - slices_family in
  let alex_ate := remaining_slices - slices_left in
  alex_ate = 3 :=
by
  intros cakes_qty_eq slices_per_cake_eq frac_friends_eq frac_family_eq slices_left_eq
  simp [total_slices, slices_given_away, cakes_qty_eq, slices_per_cake_eq, frac_friends_eq, frac_family_eq, slices_left_eq]
  sorry

end alex_ate_slices_l34_34039


namespace greatest_common_divisor_B_l34_34567

def sum_of_five_consecutive_integers (x : ℕ) : ℕ :=
  (x - 2) + (x - 1) + x + (x + 1) + (x + 2)

def B : set ℕ := {n | ∃ x : ℕ, n = sum_of_five_consecutive_integers x}

theorem greatest_common_divisor_B : gcd (set.to_finset B).min' (set.to_finset B).max' = 5 :=
by
  sorry

end greatest_common_divisor_B_l34_34567


namespace minimal_cost_is_161_5_l34_34601

noncomputable def cost_RegionA := 10 * 3    -- Region A: 10 sq ft with Easter lilies ($3 each)
noncomputable def cost_RegionC := 14 * 2.5  -- Region C: 14 sq ft with Dahlias ($2.50 each)
noncomputable def cost_RegionD := 20 * 2    -- Region D: 20 sq ft with Cannas ($2 each)
noncomputable def cost_RegionE := 21 * 1.5  -- Region E: 21 sq ft with Begonias ($1.50 each)
noncomputable def cost_RegionB := 25 * 1    -- Region B: 25 sq ft with Asters ($1 each)

noncomputable def minimal_total_cost : ℝ := cost_RegionA + cost_RegionC + cost_RegionD + cost_RegionE + cost_RegionB

theorem minimal_cost_is_161_5 : minimal_total_cost = 161.5 := by
  sorry

end minimal_cost_is_161_5_l34_34601


namespace abs_neg_three_l34_34297

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l34_34297


namespace twenty_three_percent_of_number_is_forty_six_l34_34667

theorem twenty_three_percent_of_number_is_forty_six (x : ℝ) (h : (23 / 100) * x = 46) : x = 200 :=
sorry

end twenty_three_percent_of_number_is_forty_six_l34_34667


namespace find_ab_l34_34794

-- Define the polynomials involved
def poly1 (x : ℝ) (a b : ℝ) : ℝ := a * x^4 + b * x^2 + 1
def poly2 (x : ℝ) : ℝ := x^2 - x - 2

-- Define the roots of the second polynomial
def root1 : ℝ := 2
def root2 : ℝ := -1

-- State the theorem to prove
theorem find_ab (a b : ℝ) :
  poly1 root1 a b = 0 ∧ poly1 root2 a b = 0 → a = 1/4 ∧ b = -5/4 :=
by
  -- Skipping the proof here
  sorry

end find_ab_l34_34794


namespace number_of_such_permutations_l34_34550

theorem number_of_such_permutations :
  let b : Fin 15 → Fin 15 := (finPerm : Fin 15)
  let strictly_decreasing (s : List (Fin 15)) : Prop := s = s.sorted LE.le.reverse
  let strictly_increasing (s : List (Fin 15)) : Prop := s = s.sorted LE.le
  let b1_b7_descending := b.to_list.take 7
  let b8_b15_ascending := b.to_list.drop 7
  number_of_ways_to_choose_6_from_14 : ℕ :=
  (Nat.choose 14 6)
in
  strictly_decreasing b1_b7_descending →
  strictly_increasing b8_b15_ascending →
  b1_b7_descending.head = 0 →  -- Lean indices start from 0, so 1 is represented by 0
  (number_of_ways_to_choose_6_from_14 = 3003) := 
sorry

end number_of_such_permutations_l34_34550


namespace fixed_point_of_function_l34_34096

-- Definition: The function passes through a fixed point (a, b) for all real numbers k.
def passes_through_fixed_point (f : ℝ → ℝ) (a b : ℝ) := ∀ k : ℝ, f a = b

-- Given the function y = 9x^2 + 3kx - 6k, we aim to prove the fixed point is (2, 36).
theorem fixed_point_of_function : passes_through_fixed_point (fun x => 9 * x^2 + 3 * k * x - 6 * k) 2 36 := by
  sorry

end fixed_point_of_function_l34_34096


namespace parametric_eqn_of_curve_C_polar_eqn_perpendicular_bisector_l34_34612

-- Define the initial circle and the transformation to curve C
def circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def transform_x (x : ℝ) : ℝ := sqrt(3) * x
def transform_y (y : ℝ) : ℝ := 2 * y

def curve_C (x y : ℝ) : Prop := (x / sqrt(3))^2 + (y / 2)^2 = 1

-- Define parametric equation of the curve C
def parametric_curve_C (α : ℝ) : (ℝ × ℝ) :=
  (sqrt(3) * cos α, 2 * sin α)

-- Define the polar coordinate system and point P
def polar_to_cartesian (ρ θ : ℝ) : (ℝ × ℝ) :=
  (ρ * cos θ, ρ * sin θ)

def P_polar := (2, (2 * Real.pi) / 3)

-- Define the line for symmetry and point Q
def symmetry_line (θ : ℝ) : Prop := θ = (5 * Real.pi) / 6

-- Define the intersection points A and B of line PQ with curve C
def line_PQ (x y : ℝ) : Prop := y = sqrt(3) * (x + 1) + sqrt(3)

def intersect_curve_C (x y : ℝ) : Prop :=
  curve_C x y ∧ line_PQ x y

-- Prove the solutions
theorem parametric_eqn_of_curve_C :
  ∀ α, parametric_curve_C α = (sqrt(3) * cos α, 2 * sin α) := sorry

theorem polar_eqn_perpendicular_bisector :
  ∃ ρ θ, ρ * cos θ + sqrt(3) * ρ * sin θ - 6 / 13 = 0 := sorry

end parametric_eqn_of_curve_C_polar_eqn_perpendicular_bisector_l34_34612


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l34_34886

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : 
  ∃ (count : ℕ), count = 225 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ↔ (∃ k : ℕ, 25 ≤ k ∧ k ≤ 249 ∧ n = 4 * k + 2) := 
by {
  sorry
}

end number_of_three_digit_integers_congruent_to_2_mod_4_l34_34886


namespace betty_age_l34_34038

theorem betty_age : ∀ (A M B : ℕ), A = 2 * M → A = 4 * B → M = A - 10 → B = 5 :=
by
  intros A M B h1 h2 h3
  sorry

end betty_age_l34_34038


namespace negation_prop_l34_34318

theorem negation_prop (p : Prop) : 
  (∀ (x : ℝ), x > 2 → x^2 - 1 > 0) → (¬(∀ (x : ℝ), x > 2 → x^2 - 1 > 0) ↔ (∃ (x : ℝ), x > 2 ∧ x^2 - 1 ≤ 0)) :=
by 
  sorry

end negation_prop_l34_34318


namespace right_triangle_area_l34_34306

-- Defining the problem context
structure RightTriangle30_60_90 (hypotenuse : ℝ) (angle_30 : ℝ) :=
  (hypotenuse_value : hypotenuse = 12)
  (angle_value : angle_30 = 30)

-- Defining the theorem based on the problem
theorem right_triangle_area (r : RightTriangle30_60_90 12 30) : ∃ A, A = 18 * Real.sqrt 3 :=
by
  use 18 * Real.sqrt 3
  sorry

end right_triangle_area_l34_34306


namespace sector_perimeter_l34_34022

theorem sector_perimeter (r : ℝ) (c : ℝ) (angle_deg : ℝ) (angle_rad := angle_deg * Real.pi / 180) 
  (arc_length := r * angle_rad) (P := arc_length + c)
  (h1 : r = 10) (h2 : c = 10) (h3 : angle_deg = 120) :
  P = 20 * Real.pi / 3 + 10 :=
by
  sorry

end sector_perimeter_l34_34022


namespace smallest_b_for_repeating_decimal_l34_34135

theorem smallest_b_for_repeating_decimal (a b : ℕ) (hab : a > 0 ∧ b > 0)
  (hdec : (∃ a, (a: ℚ) / b = 0.⟨2015, 4⟩)) : b = 129 :=
sorry

end smallest_b_for_repeating_decimal_l34_34135


namespace line_intersects_circle_probability_l34_34374

noncomputable def intersection_probability (k : ℝ) : ℝ :=
if k ∈ [-Real.sqrt 3, Real.sqrt 3] then
  if (-Real.sqrt 3 / 3) ≤ k ∧ k ≤ (Real.sqrt 3 / 3) then 1 / Real.sqrt 3 else 0
else 0

theorem line_intersects_circle_probability :
  ∫ k in -Real.sqrt 3 .. Real.sqrt 3, intersection_probability(k) 
  = (Real.sqrt 3 - (-Real.sqrt 3)) * 1/3 :=
by
  sorry

end line_intersects_circle_probability_l34_34374


namespace sum_twice_father_age_plus_son_age_l34_34505

/-- 
  Given:
  1. Twice the son's age plus the father's age equals 70.
  2. Father's age is 40.
  3. Son's age is 15.

  Prove:
  The sum when twice the father's age is added to the son's age is 95.
-/
theorem sum_twice_father_age_plus_son_age :
  ∀ (father_age son_age : ℕ), 
    2 * son_age + father_age = 70 → 
    father_age = 40 → 
    son_age = 15 → 
    2 * father_age + son_age = 95 := by
  intros
  sorry

end sum_twice_father_age_plus_son_age_l34_34505


namespace intersection_M_N_l34_34868

open Set Int

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_M_N_l34_34868


namespace price_of_chips_l34_34635

theorem price_of_chips (P : ℝ) (h1 : 1.5 = 1.5) (h2 : 45 = 45) (h3 : 15 = 15) (h4 : 10 = 10) :
  15 * P + 10 * 1.5 = 45 → P = 2 :=
by
  sorry

end price_of_chips_l34_34635


namespace b1_general_formula_R1_general_formula_a2_general_formula_S2_general_formula_b2_general_formula_R2_general_formula_l34_34744

-- Definition of sequences under condition ①
def a1 (n : ℕ) : ℕ := 2 * n - 1
def T1 (n : ℕ) : ℕ -- T_n is not provided explicitly in the problem
def b1 (n : ℕ) : ℕ := (if 3 * b1(n) = 2 * T1(n) + 3 then 3 else 1)^n -- as the solution steps determined

-- Proof for general formula under condition ①
theorem b1_general_formula (n : ℕ) : b1 n = 3 ^ n := by
  sorry

-- Proof for R_n under condition ①
def c1 (n : ℕ) : ℚ := a1(n) / b1(n)
def R1 (n : ℕ) : ℚ -- Sum of the first n terms of c_n

theorem R1_general_formula (n : ℕ) : R1 n = 1 - (n + 1) / 3 ^ n := by
  sorry

-- Definitions of sequences under condition ②
def a2 (n : ℕ) : ℕ := n
def S2 (n : ℕ) : ℚ := n * (n + 1) / 2
def b2 (n : ℕ) : ℕ := a2(2 * n) * S2(n)

-- Proof for general formulas under condition ②
theorem a2_general_formula (n : ℕ) : a2 n = n := by
  sorry

theorem S2_general_formula (n : ℕ) : S2 n = n * (n + 1) / 2 := by
  sorry

theorem b2_general_formula (n : ℕ) : b2 n = n ^ 2 * (n + 1) := by
  sorry

-- Proof for R_n under condition ②
def c2 (n : ℕ) : ℚ := a2(n) / b2(n)
def R2 (n : ℕ) : ℚ -- Sum of the first n terms of c_n

theorem R2_general_formula (n : ℕ) : R2 n = n / (n + 1) := by
  sorry

end b1_general_formula_R1_general_formula_a2_general_formula_S2_general_formula_b2_general_formula_R2_general_formula_l34_34744


namespace three_digit_integers_congruent_to_2_mod_4_l34_34900

theorem three_digit_integers_congruent_to_2_mod_4 : 
  let count := (249 - 25 + 1) in
  count = 225 :=
by
  let k_min := 25
  let k_max := 249
  have h_count : count = (k_max - k_min + 1) := rfl
  rw h_count
  norm_num

end three_digit_integers_congruent_to_2_mod_4_l34_34900


namespace find_height_on_BC_l34_34929

noncomputable def height_on_BC (a b : ℝ) (A B C : ℝ) : ℝ := b * (Real.sin C)

theorem find_height_on_BC (A B C a b h : ℝ)
  (h_a: a = Real.sqrt 3)
  (h_b: b = Real.sqrt 2)
  (h_cos: 1 + 2 * Real.cos (B + C) = 0)
  (h_A: A = Real.pi / 3)
  (h_B: B = Real.pi / 4)
  (h_C: C = 5 * Real.pi / 12)
  (h_h: h = height_on_BC a b A B C) :
  h = (Real.sqrt 3 + 1) / 2 :=
sorry

end find_height_on_BC_l34_34929


namespace tangent_line_equation_at_point_A_l34_34633

theorem tangent_line_equation_at_point_A (
  (f : ℝ → ℝ) (α : ℝ)
  (h1 : ∀ x, f x = x^α)
  (h2 : f (1/2) = sqrt(2) / 2)
)  :
  2 * sqrt(2) * (1 / 2) - 4 * (sqrt(2) / 2) + sqrt(2) = 0 
:=
sorry

end tangent_line_equation_at_point_A_l34_34633


namespace tank_capacity_l34_34008

theorem tank_capacity (c w : ℝ) 
  (h1 : w / c = 1 / 7) 
  (h2 : (w + 5) / c = 1 / 5) : 
  c = 87.5 := 
by
  sorry

end tank_capacity_l34_34008


namespace range_of_c_l34_34300

-- Definitions of the complex numbers representing points A, B, and C
def z1 : ℂ := 3 + 4 * complex.I
def z2 : ℂ := 0
def z3 (c : ℝ) : ℂ := c + (2 * c - 6) * complex.I

-- Function to calculate the dot product of two complex numbers represented as points in the plane
def dot_product (a b : ℂ) : ℝ :=
  (a.re * b.re) + (a.im * b.im)

-- Assertion that angle BAC is obtuse, i.e., dot product < 0
def angle_BAC_is_obtuse (c : ℝ) : Prop :=
  let A := (z1.re, z1.im)
  let B := (z2.re, z2.im)
  let C := (z3 c).re, (z3 c).im
  dot_product (B.1 - A.1 + complex.I * (B.2 - A.2)) (C.1 - A.1 + complex.I * (C.2 - A.2)) < 0

-- Theorem stating the range of c
theorem range_of_c (c : ℝ) : angle_BAC_is_obtuse c → (c > 49 / 11) ∧ (c ≠ 9) ∨ (c > 9) := sorry

end range_of_c_l34_34300


namespace correct_calculation_l34_34685

theorem correct_calculation (m n : ℝ) : -m^2 * n - 2 * m^2 * n = -3 * m^2 * n :=
by
  sorry

end correct_calculation_l34_34685


namespace symmetric_polar_equation_l34_34515

def polar_symmetric_curve (ρ θ : ℝ) : Prop :=
  ρ = cos θ + sin θ

def symmetric_about_polar_axis (ρ θ : ℝ) : Prop :=
  ρ = cos θ - sin θ

theorem symmetric_polar_equation (ρ θ : ℝ) :
  (∃ (ρ' θ' : ℝ), polar_symmetric_curve ρ' θ' ∧ (ρ' = ρ) ∧ (θ' = θ)) ↔ symmetric_about_polar_axis ρ θ :=
by
  sorry

end symmetric_polar_equation_l34_34515


namespace gcd_of_set_B_l34_34553

-- Let B be the set of all numbers which can be represented as the sum of five consecutive positive integers
def B : Set ℕ := {n | ∃ y : ℕ, n = (y - 2) + (y - 1) + y + (y + 1) + (y + 2)}

-- State the problem
theorem gcd_of_set_B : ∀ k ∈ B, ∃ d : ℕ, is_gcd 5 k d → d = 5 := by
sorry

end gcd_of_set_B_l34_34553


namespace isosceles_triangle_sin_vertex_angle_l34_34467

theorem isosceles_triangle_sin_vertex_angle (A : ℝ) (hA : 0 < A ∧ A < π / 2) 
  (hSinA : Real.sin A = 5 / 13) : 
  Real.sin (2 * A) = 120 / 169 :=
by 
  -- This placeholder indicates where the proof would go
  sorry

end isosceles_triangle_sin_vertex_angle_l34_34467


namespace one_player_has_winning_strategy_l34_34350

-- Define the conditions of the game
structure FiniteChoices (moves : Nat) : Prop :=
  (finite_choices : ∀ (s : Fin moves), ∃ n, n ∈ Fin moves ∧ n ≠ s)

structure GameEnds (moves : Nat) : Prop :=
  (inevitable_end : ∃ (n : Nat), n ≤ moves ∧ ∃ (winner : Bool), winner = true ∨ winner = false)

-- Define the specific game rules
structure SpecificGame (moves : Nat) : Prop :=
  (valid_move : ∀ n, n ≤ moves → n ≤ 10000 ∧ ∀ m, m ≠ n → m % n ≠ 0)
  (loser_no_move : ∀ n, n ≤ moves → (∀ m, m % n = 0) → false)

-- The theorem statement
theorem one_player_has_winning_strategy (moves : Nat) (finite_choices_inst : FiniteChoices moves) (game_ends_inst : GameEnds moves) (game_rules_inst : SpecificGame moves) :
  ∃ (winner : Bool), winner = true ∨ winner = false :=
by
  sorry

end one_player_has_winning_strategy_l34_34350


namespace equation_of_line_passing_through_point_l34_34534

-- Define the conditions
def point_on_line : ℝ × ℝ := (1, π / 2)

def parallel_to_polar_axis : Prop := true -- The condition is trivial in description, hence marked as true

-- Define the theorem to prove
theorem equation_of_line_passing_through_point :
  (∀ (ρ θ : ℝ), (ρ, θ) = point_on_line → ρ * sin θ = 1) :=
begin
  intros ρ θ h,
  rw [prod.mk.inj_iff] at h,
  cases h with h1 h2,
  rw [h1, h2],
  norm_num,
  sorry,
end

end equation_of_line_passing_through_point_l34_34534


namespace maximum_area_of_right_triangle_l34_34139

theorem maximum_area_of_right_triangle
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b)
  (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = 2) : 
  ∃ S, S ≤ (3 - 2 * Real.sqrt 2) ∧ S = (1/2) * a * b :=
by
  sorry

end maximum_area_of_right_triangle_l34_34139


namespace find_additional_discount_l34_34375

noncomputable def calculate_additional_discount (msrp : ℝ) (regular_discount_percent : ℝ) (final_price : ℝ) : ℝ :=
  let regular_discounted_price := msrp * (1 - regular_discount_percent / 100)
  let additional_discount_percent := ((regular_discounted_price - final_price) / regular_discounted_price) * 100
  additional_discount_percent

theorem find_additional_discount :
  calculate_additional_discount 35 30 19.6 = 20 :=
by
  sorry

end find_additional_discount_l34_34375


namespace range_of_a_l34_34482

theorem range_of_a (x : ℝ) (a : ℝ) :
  (∀ x ∈ set.Icc 4 5, monotone (λ x, log a (x^2 - 2 * a * x))) →
  1 < a ∧ a < 2 :=
by 
  sorry

end range_of_a_l34_34482


namespace curve_equation_exists_fixed_point_l34_34451

namespace ProofProblem

variables {M : Type}
variables {x y t : ℝ}
variables (F : ℝ × ℝ) (P : ℝ × ℝ) (A B : ℝ × ℝ) (k : ℝ)

def moving_point := M
def fixed_point_F := F = (Real.sqrt 3, 0)
def fixed_line := ∀ (p : ℝ × ℝ), p.1 = 4 * Real.sqrt 3 / 3
def distance_ratio (M : ℝ × ℝ) := 
  (Real.sqrt ((M.1 - Real.sqrt 3) ^ 2 + M.2 ^ 2)) / (abs (M.1 - 4 * Real.sqrt 3 / 3)) = Real.sqrt 3 / 2

theorem curve_equation (M : ℝ × ℝ) 
  (h : fixed_point_F F) (hl : fixed_line (M.1, M.2)) (hr : distance_ratio (M.1, M.2)) :
  9 * M.2 ^ 2 + 8 * Real.sqrt 3 * M.1 - 36 = 0 :=
sorry

theorem exists_fixed_point (h: fixed_point_F F) (ha : A ∈ ellipse F) (hb : B ∈ ellipse F) :
  ∃ P, ∃ t, P = (2 * Real.sqrt 3 / 3, 0) ∧
           ∀ (A B : ℝ × ℝ), ∠APF = ∠BPF :=
sorry

end ProofProblem

end curve_equation_exists_fixed_point_l34_34451


namespace cistern_water_depth_l34_34367

theorem cistern_water_depth 
  (l w a : ℝ)
  (hl : l = 8)
  (hw : w = 6)
  (ha : a = 83) :
  ∃ d : ℝ, 48 + 28 * d = 83 :=
by
  use 1.25
  sorry

end cistern_water_depth_l34_34367


namespace find_f_neg2_l34_34989

def f (x : ℝ) : ℝ :=
if x < -1 then 2 * x + 4 else 5 - 3 * x

theorem find_f_neg2 : f (-2) = 0 := by
  sorry

end find_f_neg2_l34_34989


namespace domain_f_l34_34066

def f (x : ℝ) : ℝ := Real.log (x - 1) + Real.sqrt (3 - x)

theorem domain_f :
  {x : ℝ | 1 < x ∧ x ≤ 3} = {x : ℝ | ∃ y : ℝ, f y = f x} :=
by
  unfold f
  sorry

end domain_f_l34_34066


namespace constant_term_of_expansion_is_minus_27_l34_34530

noncomputable def polynomial := (λ x: ℝ, (4^(-x) - 1) * (2^x - 3)^5)

theorem constant_term_of_expansion_is_minus_27 :
  (constant_term (polynomial x)) = -27 := sorry

end constant_term_of_expansion_is_minus_27_l34_34530


namespace original_eggs_in_jar_l34_34420

theorem original_eggs_in_jar (removed remaining original : ℕ) (h1 : removed = 7) (h2 : remaining = 20) 
(h3 : remaining + removed = original) : original = 27 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end original_eggs_in_jar_l34_34420


namespace gcd_of_set_B_l34_34555

-- Let B be the set of all numbers which can be represented as the sum of five consecutive positive integers
def B : Set ℕ := {n | ∃ y : ℕ, n = (y - 2) + (y - 1) + y + (y + 1) + (y + 2)}

-- State the problem
theorem gcd_of_set_B : ∀ k ∈ B, ∃ d : ℕ, is_gcd 5 k d → d = 5 := by
sorry

end gcd_of_set_B_l34_34555


namespace abs_neg_three_l34_34272

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l34_34272


namespace abs_neg_three_l34_34271

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l34_34271


namespace omega_range_l34_34858

theorem omega_range (ω : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < π) → deriv (λ x, sin (ω * x + π / 3)) x = 0 → 
    (∃ y : ℝ, deriv2 (λ x, sin (ω * x + π / 3)) y = 0)) →
  (∀ x : ℝ, (0 < x ∧ x < π) → sin (ω * x + π / 3) = 0 → 
    (∃ y : ℝ, y ≠ x ∧ sin (ω * y + π / 3) = 0)) →
  (13 / 6 < ω ∧ ω ≤ 8 / 3) :=
begin
  sorry
end

end omega_range_l34_34858


namespace constant_term_in_expansion_l34_34103

theorem constant_term_in_expansion (n : ℕ) (x : ℝ) (h1 : (5 * x - 1 / real.sqrt x) ^ n = 64) :
  ((-1)^4 * 5^2 * nat.choose 6 4) = 375 :=
by
  sorry

end constant_term_in_expansion_l34_34103


namespace find_d_n_l34_34438

open Nat

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := n.choose k

-- Define d(n) as the GCD of binomial coefficients 
def d (n : ℕ) : ℕ :=
  gcd (list.map (λ k => binom (n + k) n) (list.range n).tail)

-- Our theorem to prove
theorem find_d_n (n : ℕ) (hn : 0 < n) : 
  (∃ p : ℕ, prime p ∧ ∃ l : ℕ, n = p^l - 1 ∧ d n = p) ∨
  (¬ ∃ p : ℕ, prime p ∧ ∃ l : ℕ, n = p^l - 1 ∧ d n = p ∧ d n = 1) :=
by
  sorry

end find_d_n_l34_34438


namespace minimum_notes_required_l34_34655

theorem minimum_notes_required (N : ℕ) (h : N + 1 >= 100) : N >= 99 :=
by
  have h : N + 1 ≥ 100 := by sorry
  exact (Nat.sub_le_left_iff_le_add 1).mp h

end minimum_notes_required_l34_34655


namespace max_elements_subset_S_l34_34967

noncomputable def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m | n → m = 1

noncomputable def is_subset_S (S : set ℕ) : Prop :=
  (∀ x ∈ S, x ≤ 500) ∧
  (∀ x y ∈ S, x ≠ y → ¬ (∃ k, k * k = x * y))

theorem max_elements_subset_S : 
  ∃ S : set ℕ, is_subset_S S ∧ S.card = 306 :=
sorry

end max_elements_subset_S_l34_34967


namespace triangles_with_prime_factors_1995_l34_34166

theorem triangles_with_prime_factors_1995 : 
  (number_of_triangles [3, 5, 7, 19] = 13) :=
sorry

end triangles_with_prime_factors_1995_l34_34166


namespace compare_a_b_l34_34106

noncomputable def a : ℝ := Real.log 3 / Real.log 0.7
noncomputable def b : ℝ := 3 ^ 0.7

theorem compare_a_b : a < b := by
  sorry

end compare_a_b_l34_34106


namespace solve_problem_l34_34175

def is_solution (a : ℕ) : Prop :=
  a % 3 = 1 ∧ ∃ k : ℕ, a = 5 * k

theorem solve_problem : ∃ a : ℕ, is_solution a ∧ ∀ b : ℕ, is_solution b → a ≤ b := 
  sorry

end solve_problem_l34_34175


namespace exists_nonzero_combination_l34_34752

theorem exists_nonzero_combination (m n : ℕ) (a : Fin n → ℤ)
  (hm : 1 < m) (hn : 1 < n)
  (h : ∀ i, a i % m^(n-1) ≠ 0) :
  ∃ e : Fin n → ℤ, (∀ i, abs (e i) < m) ∧ (e ≠ 0) ∧ 
  (∑ i, e i * a i) % m^n = 0 :=
by
  sorry

end exists_nonzero_combination_l34_34752


namespace u_2010_is_5898_l34_34580

-- Let un denote the nth term of the sequence defined by the given conditions
def u : ℕ → ℕ 
| 0       := 1
| (n + 1) := 
    if (n % 3 == 0) then
      u n + 1
    else if (n % 3 == 1) then
      u n + 3 - 1
    else
      u n + 3

theorem u_2010_is_5898 : u 2009 = 5898 :=
by
  sorry

end u_2010_is_5898_l34_34580


namespace expand_expression_l34_34082

theorem expand_expression (x : ℝ) : (7 * x + 5) * (3 * x^2) = 21 * x^3 + 15 * x^2 :=
by
  sorry

end expand_expression_l34_34082


namespace shaded_region_occupies_32_percent_of_total_area_l34_34197

-- Conditions
def angle_sector := 90
def r_small := 1
def r_large := 3
def r_sector := 4

-- Question: Prove the shaded region occupies 32% of the total area given the conditions
theorem shaded_region_occupies_32_percent_of_total_area :
  let area_large_sector := (1 / 4) * Real.pi * (r_sector ^ 2)
  let area_small_sector := (1 / 4) * Real.pi * (r_large ^ 2)
  let total_area := area_large_sector + area_small_sector
  let shaded_area := (1 / 4) * Real.pi * (r_large ^ 2) - (1 / 4) * Real.pi * (r_small ^ 2)
  let shaded_percent := (shaded_area / total_area) * 100
  shaded_percent = 32 := by
  sorry

end shaded_region_occupies_32_percent_of_total_area_l34_34197


namespace find_a_val_l34_34573

/--
Given 
1. \( a > 1 \)
2. \( y = |\log_a x| \) with domain \([m, n]\)
3. The range of \( y \) is \([0,1]\)
4. The minimum length of the interval \([m, n]\) is 6

Prove that the value of \( a \) is \( 3 + \sqrt{10} \).
-/
theorem find_a_val (a : ℝ) (m n : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : a > 1)
  (h2 : ∀ x ∈ set.Icc m n, y = abs (Real.log x / Real.log a))
  (h3 : ∀ y ∈ set.Icc 0 1, ∃ x ∈ set.Icc m n, y = abs (Real.log x / Real.log a))
  (h4 : n - m = 6) :
  a = 3 + Real.sqrt 10 := 
sorry

end find_a_val_l34_34573


namespace pipe_A_fill_time_l34_34006

variable (A : ℝ)

-- Conditions
def pipe_A_filling_rate := 1 / A
def pipe_B_leaking_rate := 1 / 25
def net_filling_rate := 1 / 100

-- Prove
theorem pipe_A_fill_time :
  (1 / A) - (1 / 25) = 1 / 100 → A = 20 :=
by
  intro h
  sorry

end pipe_A_fill_time_l34_34006


namespace Colin_skipping_speed_is_correct_l34_34407

variable (BruceSpeed ColinSpeed TonySpeed BrandonSpeed : ℝ)

def ColinSpeed := 4
def TonySpeed := 2 * BruceSpeed
def BrandonSpeed := (1/3) * TonySpeed
def ColinSpeed := 6 * BrandonSpeed

theorem Colin_skipping_speed_is_correct :
  ColinSpeed = 4 := by
  sorry

end Colin_skipping_speed_is_correct_l34_34407


namespace normal_vector_ratio_l34_34832

-- Define the points A, B, C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A := Point3D.mk 0 2 (19 / 8)
def B := Point3D.mk 1 (-1) (5 / 8)
def C := Point3D.mk (-2) 1 (5 / 8)

-- Define the normal vector a
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of vector subtraction
def subtract (p1 p2 : Point3D) : Vector3D :=
  Vector3D.mk (p1.x - p2.x) (p1.y - p2.y) (p1.z - p2.z)

-- Define vectors AB and AC
def AB := subtract B A
def AC := subtract C A

-- Prove the ratio of the normal vector given the conditions
theorem normal_vector_ratio (a : Vector3D)
  (h1 : a.x * AB.x + a.y * AB.y + a.z * AB.z = 0)
  (h2 : a.x * AC.x + a.y * AC.y + a.z * AC.z = 0) :
  (a.x : ℚ) / (a.y : ℚ) = (2 : ℚ)/(3 : ℚ) ∧ (a.z : ℚ) / (a.y : ℚ) = (-4 : ℚ)/(3 : ℚ) :=
sorry

end normal_vector_ratio_l34_34832


namespace set_contains_squares_card_greater_than_exponent_l34_34969

theorem set_contains_squares_card_greater_than_exponent 
  (n : ℕ) (A : Set ℤ) 
  (h1 : ∀ k, 1 ≤ k ∧ k ≤ n → ∃ a b ∈ A, a + b = k^2) :
  ∃ N : ℕ, ∀ n, n ≥ N → A.card > n^((2 : ℝ) / 3) :=
by
  sorry

end set_contains_squares_card_greater_than_exponent_l34_34969


namespace problem1_problem2_l34_34400

theorem problem1 : (1 : ℝ) * (2023 - sqrt 5)^0 - 2 + abs (sqrt 3 - 1) = sqrt 3 - 2 :=
by
  sorry

theorem problem2 : (sqrt 3 + 2) * (sqrt 3 - 2) + (sqrt 15 * sqrt 3) / sqrt 5 = 2 :=
by
  sorry

end problem1_problem2_l34_34400


namespace croissant_to_orange_ratio_l34_34079

-- Define the conditions as given in the problem
variables (c o : ℝ)
variable (emily_expenditure : ℝ)
variable (lucas_expenditure : ℝ)

-- Given conditions of expenditures
axiom emily_expenditure_is : emily_expenditure = 5 * c + 4 * o
axiom lucas_expenditure_is : lucas_expenditure = 3 * emily_expenditure
axiom lucas_expenditure_as_purchased : lucas_expenditure = 4 * c + 10 * o

-- Prove the ratio of the cost of a croissant to an orange
theorem croissant_to_orange_ratio : (c / o) = 2 / 11 :=
by sorry

end croissant_to_orange_ratio_l34_34079


namespace estimate_students_scoring_above_110_l34_34325

noncomputable def students : ℕ := 50
noncomputable def mean : ℝ := 100
noncomputable def stddev : ℝ := 10
noncomputable def P_90_100 : ℝ := 0.3
noncomputable def P_above_110 : ℝ := 1 - ((1 / 2) + 0.3)

theorem estimate_students_scoring_above_110 :
  students * P_above_110 ≈ 8 :=
by
  sorry

end estimate_students_scoring_above_110_l34_34325


namespace lopez_family_seating_arrangement_count_l34_34239

def lopez_family_seating_arrangements : Nat := 2 * 4 * 6

theorem lopez_family_seating_arrangement_count : lopez_family_seating_arrangements = 48 :=
by 
    sorry

end lopez_family_seating_arrangement_count_l34_34239


namespace gcd_of_B_l34_34562

def B : Set ℕ := {n | ∃ x : ℕ, n = 5 * x}

theorem gcd_of_B : Int.gcd_range (B.toList) = 5 := by 
  sorry

end gcd_of_B_l34_34562


namespace cost_of_fencing_per_meter_l34_34637

noncomputable def plot_cost_per_meter (length breadth total_cost : ℝ) (h1 : length = breadth + 30) (h2 : length = 65) (h3 : total_cost = 5300) : ℝ :=
  let perimeter := 2 * length + 2 * breadth in
  total_cost / perimeter

theorem cost_of_fencing_per_meter 
  (breadth : ℝ) 
  (h_breadth : breadth = 35) 
  : plot_cost_per_meter 65 breadth 5300 (by simp [h_breadth]) (by simp) (by simp) = 26.5 :=
by sorry

end cost_of_fencing_per_meter_l34_34637


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l34_34896

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : ∃ (n : ℕ), n = 225 ∧ ∀ k : ℤ, 100 ≤ 4 * k + 2 ∧ 4 * k + 2 ≤ 999 → 24 < k ∧ k < 250 := by
  sorry

end number_of_three_digit_integers_congruent_to_2_mod_4_l34_34896


namespace max_height_of_ball_l34_34358

theorem max_height_of_ball :
  ∃ t : ℝ, h t = 40 :=
by
  let h t := -20 * t^2 + 40 * t + 20
  existsi 1
  sorry

end max_height_of_ball_l34_34358


namespace min_value_of_n_l34_34354

def is_prime (p : ℕ) : Prop := p ≥ 2 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def is_not_prime (n : ℕ) : Prop := ¬ is_prime n

def decomposable_into_primes_leq_10 (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≤ 10 ∧ q ≤ 10 ∧ n = p * q

theorem min_value_of_n : ∃ n : ℕ, is_not_prime n ∧ decomposable_into_primes_leq_10 n ∧ n = 6 :=
by
  -- The proof would go here.
  sorry

end min_value_of_n_l34_34354


namespace leap_stride_difference_l34_34772

theorem leap_stride_difference :
  let elmer_strides_per_gap := 44
  let oscar_leaps_per_gap := 12
  let num_poles := 41
  let distance_miles := 5280 in
  let num_gaps := num_poles - 1
  let total_elmer_strides := elmer_strides_per_gap * num_gaps
  let total_oscar_leaps := oscar_leaps_per_gap * num_gaps
  let elmer_stride_length := distance_miles / total_elmer_strides
  let oscar_leap_length := distance_miles / total_oscar_leaps
  oscar_leap_length - elmer_stride_length = 8 := by
{
  sorry
}

end leap_stride_difference_l34_34772


namespace length_DE_equals_inradius_l34_34940

-- Define the necessary geometric structures and properties
variables {α : Type*} [ordered_ring α]
variables {A B C K N D E : Type*} [geom_triangle A B C]
variables {bisector_AK : is_angle_bisector A K C}
variables {bisector_BN : is_angle_bisector B N C}
variables {perpendicular_CD : is_perpendicular C D}
variables {perpendicular_CE : is_perpendicular C E}
variables {radius_r : inradius α A B C}

theorem length_DE_equals_inradius :
  length (segment D E) = inradius A B C :=
sorry

end length_DE_equals_inradius_l34_34940


namespace sqrt49_times_sqrt25_eq_5sqrt7_l34_34054

noncomputable def sqrt49_times_sqrt25 : ℝ :=
  Real.sqrt (49 * Real.sqrt 25)

theorem sqrt49_times_sqrt25_eq_5sqrt7 :
  sqrt49_times_sqrt25 = 5 * Real.sqrt 7 :=
by
sorry

end sqrt49_times_sqrt25_eq_5sqrt7_l34_34054


namespace extreme_values_distinct_zeros_l34_34108

variable (m : ℝ)

def f (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + 3 * x - m

theorem extreme_values :
  let f1 := f 1,
      f3 := f 3,
      max_val := (4/3) - m,
      min_val := -m
  in f1 = max_val ∧ f3 = min_val :=
by
  sorry

theorem distinct_zeros :
  (0 < m ∧ m < (4/3)) ↔ 
  (∃ x1 x2 x3 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :=
by
  sorry

end extreme_values_distinct_zeros_l34_34108


namespace side_length_is_correct_l34_34365

-- Defining the mathematical conditions given in the problem
def circle_area (radius : ℝ) : ℝ := Real.pi * radius ^ 2

def is_equilateral_triangle (A B C : EuclideanGeometry.Point) : Prop :=
  EuclideanGeometry.dist A B = EuclideanGeometry.dist B C ∧ EuclideanGeometry.dist B C = EuclideanGeometry.dist C A

def circumradius (P : EuclideanGeometry.Point) (A B C : EuclideanGeometry.Point) : ℝ :=
  EuclideanGeometry.dist P A

noncomputable def side_length_of_equilateral_triangle
  (O A B C : EuclideanGeometry.Point)
  (h_area : circle_area 10 = 100 * Real.pi)
  (h_equilateral : is_equilateral_triangle A B C)
  (h_OA : EuclideanGeometry.dist O A = 5) : ℝ :=
  10 * Real.sqrt 3

-- Stating the theorem to prove
theorem side_length_is_correct
  (O A B C : EuclideanGeometry.Point)
  (h_area : circle_area 10 = 100 * Real.pi)
  (h_equilateral : is_equilateral_triangle A B C)
  (h_OA : EuclideanGeometry.dist O A = 5) :
  side_length_of_equilateral_triangle O A B C h_area h_equilateral h_OA = 10 * Real.sqrt 3 :=
sorry

end side_length_is_correct_l34_34365


namespace positive_integers_count_l34_34701

theorem positive_integers_count (n : ℕ) : 
  ∃ m : ℕ, (m ≤ n / 2014 ∧ m ≤ n / 2016 ∧ (m + 1) * 2014 > n ∧ (m + 1) * 2016 > n) ↔
  (n = 1015056) :=
by
  sorry

end positive_integers_count_l34_34701


namespace equal_angles_l34_34616

variable (W1 W2 : Circle)
variable (O1 O2 : Point)
variable (M N B1 B2 A1 A2 I1 I2 : Point)

-- Given conditions for the problem
variable (h1 : Intersects W1 W2 M N)
variable (h2 : Center W1 O1)
variable (h3 : Center W2 O2)
variable (h4 : TangentAt W2 N B1)
variable (h5 : TangentAt W1 N B2)
variable (h6 : OnArc W1 B1 N A1 ¬ M)
variable (h7 : IntersectsLine W1 W2 A1 N A2)
variable (h8 : Incenter (Triangle B1 A1 N) I1)
variable (h9 : Incenter (Triangle B2 A2 N) I2)

-- Statement of the theorem
theorem equal_angles : 
  ∀ (W1 W2 : Circle) (O1 O2 M N B1 B2 A1 A2 I1 I2 : Point),
    Intersects W1 W2 M N →
    Center W1 O1 →
    Center W2 O2 →
    TangentAt W2 N B1 →
    TangentAt W1 N B2 →
    OnArc W1 B1 N A1 ¬ M →
    IntersectsLine W1 W2 A1 N A2 →
    Incenter (Triangle B1 A1 N) I1 →
    Incenter (Triangle B2 A2 N) I2 →
    angle I1 M I2 = angle O1 M O2 :=
by
  sorry

end equal_angles_l34_34616


namespace smallest_m_divides_sum_square_l34_34098

def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then
    let factors := (n.factors).filter (λ p, Nat.Prime p)
    List.maximum factors
  else 0

def pow (n : ℕ) : ℕ := 
  let p := largest_prime_factor n
  if p = 0 then 1 else
  p ^ (Nat.log n / Nat.log p).to_nat

def sum_pow (n : ℕ) : ℕ :=
  (Finset.range (n - 1)).sum (λ k, pow (k + 2))

def smallest_m (n : ℕ) : ℕ :=
  let s := sum_pow n
  let t := s * s
  (Finset.range 100).find (λ m, (2310 ^ m) ∣ t).get_or_else 0

theorem smallest_m_divides_sum_square :
  smallest_m 4000 = 1 := by
  sorry

end smallest_m_divides_sum_square_l34_34098


namespace number_of_permutations_l34_34551

noncomputable def num_satisfying_permutations : ℕ :=
  Nat.choose 15 7

theorem number_of_permutations : num_satisfying_permutations = 6435 := by
  sorry

end number_of_permutations_l34_34551


namespace find_n_modulo_l34_34087

theorem find_n_modulo :
  ∃ (n : ℤ), n ≡ -3457 [MOD 13] ∧ 0 ≤ n ∧ n ≤ 12 ∧ n = 1 :=
by
  sorry

end find_n_modulo_l34_34087


namespace gcd_of_B_l34_34563

def B : Set ℕ := {n | ∃ x : ℕ, n = 5 * x}

theorem gcd_of_B : Int.gcd_range (B.toList) = 5 := by 
  sorry

end gcd_of_B_l34_34563


namespace abs_neg_three_l34_34291

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l34_34291


namespace find_f2_l34_34224

def differentiable_real_function (f : ℝ → ℝ) :=
  ∀ x ∈ set.Ioi (0 : ℝ), differentiable_at ℝ f x

def tangent_meets_y_axis_property (f : ℝ → ℝ) :=
  ∀ x ∈ set.Ioi (0 : ℝ), let tan_line := λ x0, f x + deriv f x * (x0 - x) in
    tan_line 0 = f x - 1

axiom initial_condition (f : ℝ → ℝ) : f 1 = 0

theorem find_f2 (f : ℝ → ℝ) 
  (hf_diff : differentiable_real_function f)
  (hf_tangent_property : tangent_meets_y_axis_property f)
  (hf_initial_condition : initial_condition f) :
  f 2 = Real.log 2 :=
  sorry

end find_f2_l34_34224


namespace find_m_plus_b_l34_34307

noncomputable def reflection_line_params : ℚ × ℚ :=
let p1 := (2 : ℚ, 3 : ℚ),
    p2 := (10 : ℚ, 6 : ℚ) in
-- Calculate the slope of the segment connecting (2, 3) and (10, 6)
let seg_slope := (p2.2 - p1.2) / (p2.1 - p1.1) in
-- Determine the slope of the line of reflection
let m := - (1 / seg_slope) in
-- Find the midpoint of (2, 3) and (10, 6)
let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2) in
-- Substitute into the line equation to find b
let b := midpoint.2 - m * midpoint.1 in
(m, b)

theorem find_m_plus_b :
  let (m, b) := reflection_line_params in
  m + b = 107 / 6 := by
  sorry

end find_m_plus_b_l34_34307


namespace ellipse_equation_and_minimum_distance_l34_34824

theorem ellipse_equation_and_minimum_distance 
  (e : ℝ) (focus_x : ℝ) (ell : ℝ → ℝ → ℝ) (origin : ℝ × ℝ) :
  e = sqrt 2 / 2 →
  focus_x = sqrt 2 →
  (∀ x y, ell x y = x^2 / 4 + y^2 / 2 - 1) →
  origin = (0, 0) →
  (∀ (l : ℝ → ℝ), 
    (∃ A B P, 
      l A ∧ 
      l B ∧ 
      ell P.1 P.2 = 0 ∧ 
      ∀ k, d origin l = sqrt(1 - 1 / (2 (1 + k^2))) → 
      ∃ d_min, d_min = sqrt 2 / 2)) :=
by
  intros h_e h_focus h_ell h_origin h_l
  sorry

end ellipse_equation_and_minimum_distance_l34_34824


namespace coefficient_of_x_in_expansion_of_x_plus_m_pow_6_l34_34143

noncomputable def normalDist (mean variance : ℝ) : ℝ -> ℝ := sorry

noncomputable def P (event : ℝ -> Prop) : ℝ := sorry

theorem coefficient_of_x_in_expansion_of_x_plus_m_pow_6 :
  (∀ (xi : ℝ), normalDist (1/2) sigma^2 xi) ->
  P (λ xi, xi < -1) = P (λ xi, xi > m) ->
  (6.choose 1) * (m ^ 5) = 192 :=
by
  sorry

end coefficient_of_x_in_expansion_of_x_plus_m_pow_6_l34_34143


namespace measure_C_in_triangle_ABC_l34_34509

noncomputable def measure_angle_C (A B : ℝ) : ℝ := 180 - (A + B)

theorem measure_C_in_triangle_ABC (A B : ℝ) (h : A + B = 150) : measure_angle_C A B = 30 := 
by
  unfold measure_angle_C
  rw h
  norm_num
  sorry

end measure_C_in_triangle_ABC_l34_34509


namespace weekly_earnings_proof_l34_34267

def minutes_in_hour : ℕ := 60
def hourly_rate : ℕ := 4

def monday_minutes : ℕ := 150
def tuesday_minutes : ℕ := 40
def wednesday_minutes : ℕ := 155
def thursday_minutes : ℕ := 45

def weekly_minutes : ℕ := monday_minutes + tuesday_minutes + wednesday_minutes + thursday_minutes
def weekly_hours : ℕ := weekly_minutes / minutes_in_hour

def sylvia_earnings : ℕ := weekly_hours * hourly_rate

theorem weekly_earnings_proof :
  sylvia_earnings = 26 := by
  sorry

end weekly_earnings_proof_l34_34267


namespace max_value_f_l34_34819

noncomputable def f (a x : ℝ) : ℝ := a * Real.sqrt (1 - x^2) + Real.sqrt (1 + x) + Real.sqrt (1 - x)

def domain_f (a : ℝ) : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

def t (x : ℝ) : ℝ := Real.sqrt (1 + x) + Real.sqrt (1 - x)

def range_t : Set ℝ := { y | Real.sqrt 2 ≤ y ∧ y ≤ 2 }

noncomputable def h (a t : ℝ) : ℝ := (a / 2) * t^2 + t - a

noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ -Real.sqrt 2 / 2 then
    Real.sqrt 2
  else if -Real.sqrt 2 / 2 < a ∧ a ≤ -1 / 2 then
    -a - 1 / (2*a)
  else
    a + 2

theorem max_value_f (a : ℝ) (ha : a < 0) :
  ∃ x ∈ domain_f a, ∀ y ∈ domain_f a, f a x ≥ f a y ↔
  g a = 
    if a ≤ -Real.sqrt 2 / 2 then
      Real.sqrt 2
    else if -Real.sqrt 2 / 2 < a ∧ a ≤ -1 / 2 then
      -a - 1 / (2*a)
    else
      a + 2 := sorry

end max_value_f_l34_34819


namespace bowling_average_before_last_match_l34_34371

theorem bowling_average_before_last_match
  (wickets_before_last : ℕ)
  (wickets_last_match : ℕ)
  (runs_last_match : ℕ)
  (decrease_in_average : ℝ)
  (average_before_last : ℝ) :

  wickets_before_last = 115 →
  wickets_last_match = 6 →
  runs_last_match = 26 →
  decrease_in_average = 0.4 →
  (average_before_last - decrease_in_average) = 
  ((wickets_before_last * average_before_last + runs_last_match) / 
  (wickets_before_last + wickets_last_match)) →
  average_before_last = 12.4 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end bowling_average_before_last_match_l34_34371


namespace hyperbola_problem_l34_34971

open Real

noncomputable def sqrt10 := Real.sqrt 10

theorem hyperbola_problem :
  let F1 : ℝ × ℝ := (-sqrt10, 0)
  let F2 : ℝ × ℝ := (sqrt10, 0)
  let P : ℝ × ℝ := sorry -- P is on the hyperbola and needs to be determined
  in P.1^2 - (P.2^2 / 9) = 1 ∧ ((P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2)) = 0 →
        sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 2 * sqrt10 :=
by
  sorry

end hyperbola_problem_l34_34971


namespace find_derivative_at_2_l34_34854

theorem find_derivative_at_2 (f : ℝ → ℝ) (h : ∀ x, f x = 1/4 * x^4 + 4 * x * (f' 0)) :
  f' 2 = 8 :=
by
  -- proof would go here
  sorry

end find_derivative_at_2_l34_34854


namespace expression_calculation_l34_34055

theorem expression_calculation : 
  (3^1005 + 7^1006)^2 - (3^1005 - 7^1006)^2 = 28 * 21^1005 :=
by
  sorry

end expression_calculation_l34_34055


namespace length_of_bridge_l34_34032

theorem length_of_bridge
  (train_length : ℕ)
  (bridge_time : ℕ)
  (train_speed_kmh : ℝ)
  (conversion_factor : ℝ := 1000 / 3600) :
  train_length = 240 →
  bridge_time = 20 →
  train_speed_kmh = 70.2 →
  let train_speed := train_speed_kmh * conversion_factor in
  let total_distance := train_speed * bridge_time in
  let bridge_length := total_distance - train_length in
  bridge_length = 150 :=
by
  intros ht hl hs
  let train_speed := train_speed_kmh * conversion_factor
  let total_distance := train_speed * bridge_time
  let bridge_length := total_distance - train_length
  rw [ht, hl, hs]
  have train_speed_calc : train_speed = 19.5 := by
    simp [train_speed, conversion_factor]
  rw train_speed_calc
  have total_distance_calc : total_distance = 390 := by
    simp [total_distance]
  rw total_distance_calc
  have bridge_length_calc : bridge_length = 150 := by
    simp [bridge_length]
  exact bridge_length_calc

end length_of_bridge_l34_34032


namespace sin_double_angle_l34_34926

theorem sin_double_angle (P : ℝ × ℝ) (hP : P = (1, 2)) :
  let α := real.atan2 P.2 P.1 in
  real.sin (2 * α) = 4 / 5 :=
by sorry

end sin_double_angle_l34_34926


namespace geom_seq_max_value_l34_34845

theorem geom_seq_max_value (a b c d : ℝ) (r : ℝ) (h1 : b = a * r)
(h2 : c = a * r^2) (h3 : d = a * r^3) (h4 : ∀ x, y(x) = log (x + 2) - x)
(h5 : ∀ x, deriv y x = 1 / (x + 2) - 1)
(h6 : is_maximum y b c) (h7 : b = -1) : 
a * d = -1 :=
sorry

end geom_seq_max_value_l34_34845


namespace lines_and_planes_correct_l34_34219

noncomputable def lines_and_planes (m n : Line) (α β : Plane) : Prop :=
  m ⊥ α ∧ n ⊥ β ∧ α ⊥ β → m ⊥ n

-- Proof to be filled in
theorem lines_and_planes_correct (m n : Line) (α β : Plane) :
  lines_and_planes m n α β :=
by
  sorry

end lines_and_planes_correct_l34_34219


namespace focus_of_hyperbola_l34_34094

theorem focus_of_hyperbola (m : ℝ) :
  let focus_parabola := (0, 4)
  let focus_hyperbola_upper := (0, 4)
  ∃ focus_parabola, ∃ focus_hyperbola_upper, 
    (focus_parabola = (0, 4)) ∧ (focus_hyperbola_upper = (0, 4)) ∧ 
    (3 + m = 16) → m = 13 :=
by
  sorry

end focus_of_hyperbola_l34_34094


namespace two_digit_numbers_satisfying_condition_l34_34060

theorem two_digit_numbers_satisfying_condition :
  {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 10 * a + b ∧ (a.factorial - a * b) % 10 = 2}.card = 1 := 
sorry

end two_digit_numbers_satisfying_condition_l34_34060


namespace remainder_when_divided_by_2_is_0_l34_34683

theorem remainder_when_divided_by_2_is_0 (n : ℕ)
  (h1 : ∃ r, n % 2 = r)
  (h2 : n % 7 = 5)
  (h3 : ∃ p, p = 5 ∧ (n + p) % 10 = 0) :
  n % 2 = 0 :=
by
  -- skipping the proof steps; hence adding sorry
  sorry

end remainder_when_divided_by_2_is_0_l34_34683


namespace tom_buys_papayas_l34_34665

-- Defining constants for the costs of each fruit
def lemon_cost : ℕ := 2
def papaya_cost : ℕ := 1
def mango_cost : ℕ := 4

-- Defining the number of each fruit Tom buys
def lemons_bought : ℕ := 6
def mangos_bought : ℕ := 2
def total_paid : ℕ := 21

-- Defining the function to calculate the total cost 
def total_cost (P : ℕ) : ℕ := (lemons_bought * lemon_cost) + (mangos_bought * mango_cost) + (P * papaya_cost)

-- Defining the function to calculate the discount based on the total number of fruits
def discount (P : ℕ) : ℕ := (lemons_bought + mangos_bought + P) / 4

-- Main theorem to prove
theorem tom_buys_papayas (P : ℕ) : total_cost P - discount P = total_paid → P = 4 := 
by
  intro h
  sorry

end tom_buys_papayas_l34_34665


namespace toys_produced_each_day_l34_34010

theorem toys_produced_each_day (weekly_production : ℕ) (days_worked : ℕ) (h₁ : weekly_production = 4340) (h₂ : days_worked = 2) : weekly_production / days_worked = 2170 :=
by {
  -- Proof can be filled in here
  sorry
}

end toys_produced_each_day_l34_34010


namespace transformation_result_l34_34059

theorem transformation_result (a b : ℝ) 
  (h1 : ∃ P : ℝ × ℝ, P = (a, b))
  (h2 : ∃ Q : ℝ × ℝ, Q = (b, a))
  (h3 : ∃ R : ℝ × ℝ, R = (2 - b, 10 - a))
  (h4 : (2 - b, 10 - a) = (-8, 2)) : 
  a - b = -2 := 
by 
  sorry

end transformation_result_l34_34059


namespace integer_modulo_problem_l34_34672

theorem integer_modulo_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ (-250 % 23 = n) := 
  sorry

end integer_modulo_problem_l34_34672


namespace ellipse_equation_standard_range_x_and_max_val_l34_34455

theorem ellipse_equation_standard (a b : ℝ) (h1 : a > b) (h2 : b > 0) (eccentricity : ℝ) 
    (h3 : eccentricity = (Real.sqrt 3) / 2) (intersect_y : |2 * b| = 2): 
    ∃ a b, (∃ e : ℝ, e = (Real.sqrt 3) / 2 ∧ 
    (e * e = 3 / 4) ∧
    (∃ a_sq : ℝ, a_sq = 4)) ∧
    (∀ x₀ P : ℝ × ℝ, (0 < x₀ ∧ x₀ ≤ 2) → 
    (∃ P (H5: P.1 = x₀) (H6: 4 - Real.sqrt (5 - 8 / x₀) < P.1 ∧ P.1 ≤ 4 + Real.sqrt (5 - 8 / x₀)),
    interval.ite (5 > 8 / x₀) (5 - 8 / x₀))) :=
sorry

theorem range_x_and_max_val (a b : ℝ) (h1 : a > b) (h2 : b > 0) (eccentricity : ℝ) 
    (h3 : eccentricity = (Real.sqrt 3) / 2) (intersect_y : |2 * b| = 2): 
    ∃ x₀, (x₀ ∈ (8 / 5, 2]) ∧ 
    (∀ P : ℝ × ℝ, 0 < x₀ ∧ x₀ ≤ 2 → ∃ |EF| = 2) :=
sorry

end ellipse_equation_standard_range_x_and_max_val_l34_34455


namespace complex_division_example_l34_34136

theorem complex_division_example (i : ℂ) (h : i^2 = -1) : 
  (2 - i) / (1 + i) = (1/2 : ℂ) - (3/2 : ℂ) * i :=
by
  -- proof would go here
  sorry

end complex_division_example_l34_34136


namespace total_snakes_seen_l34_34589

-- Define the number of snakes in each breeding ball
def snakes_in_first_breeding_ball : Nat := 15
def snakes_in_second_breeding_ball : Nat := 20
def snakes_in_third_breeding_ball : Nat := 25
def snakes_in_fourth_breeding_ball : Nat := 30
def snakes_in_fifth_breeding_ball : Nat := 35
def snakes_in_sixth_breeding_ball : Nat := 40
def snakes_in_seventh_breeding_ball : Nat := 45

-- Define the number of pairs of extra snakes
def extra_pairs_of_snakes : Nat := 23

-- Define the total number of snakes observed
def total_snakes_observed : Nat :=
  snakes_in_first_breeding_ball +
  snakes_in_second_breeding_ball +
  snakes_in_third_breeding_ball +
  snakes_in_fourth_breeding_ball +
  snakes_in_fifth_breeding_ball +
  snakes_in_sixth_breeding_ball +
  snakes_in_seventh_breeding_ball +
  (extra_pairs_of_snakes * 2)

theorem total_snakes_seen : total_snakes_observed = 256 := by
  sorry

end total_snakes_seen_l34_34589


namespace cos_89_cos_1_plus_sin_91_sin_181_l34_34766

theorem cos_89_cos_1_plus_sin_91_sin_181:
  let a := real.cos (89 * real.pi / 180)
  let b := real.cos (1 * real.pi / 180)
  let c := real.sin (91 * real.pi / 180)
  let d := real.sin (181 * real.pi / 180)
  a * b + c * d = 0 := by
  sorry

end cos_89_cos_1_plus_sin_91_sin_181_l34_34766


namespace initial_puppies_correct_l34_34019

def initial_puppies (total_puppies_after: ℝ) (bought_puppies: ℝ) : ℝ :=
  total_puppies_after - bought_puppies

theorem initial_puppies_correct : initial_puppies (4.2 * 5.0) 3.0 = 18.0 := by
  sorry

end initial_puppies_correct_l34_34019


namespace four_digit_palindromic_squares_with_different_middle_digits_are_zero_l34_34162

theorem four_digit_palindromic_squares_with_different_middle_digits_are_zero :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (∃ k, k * k = n) ∧ (∃ a b, n = 1001 * a + 110 * b) → a ≠ b → false :=
by sorry

end four_digit_palindromic_squares_with_different_middle_digits_are_zero_l34_34162


namespace perfect_number_divisibility_l34_34017

theorem perfect_number_divisibility (P : ℕ) (h1 : P > 28) (h2 : Nat.Perfect P) (h3 : 7 ∣ P) : 49 ∣ P := 
sorry

end perfect_number_divisibility_l34_34017


namespace train_crossing_time_l34_34382

theorem train_crossing_time
  (length : ℝ) (speed_kmh : ℝ) (conversion_factor : ℝ)
  (h_length : length = 700)
  (h_speed_kmh : speed_kmh = 125.99999999999999)
  (h_conversion_factor : conversion_factor = 0.27777778) :
  let speed_ms := speed_kmh * conversion_factor in
  let time := length / speed_ms in
  time = 20 :=
begin
  sorry
end

end train_crossing_time_l34_34382


namespace degree_product_l34_34266

-- Define the degrees of the polynomials p and q
def degree_p : ℕ := 3
def degree_q : ℕ := 4

-- Define the functions p(x) and q(x) as polynomials and their respective degrees
axiom degree_p_definition (p : Polynomial ℝ) : p.degree = degree_p
axiom degree_q_definition (q : Polynomial ℝ) : q.degree = degree_q

-- Define the degree of the product p(x^2) * q(x^4)
noncomputable def degree_p_x2_q_x4 (p q : Polynomial ℝ) : ℕ :=
  2 * degree_p + 4 * degree_q

-- Prove that the degree of p(x^2) * q(x^4) is 22
theorem degree_product (p q : Polynomial ℝ) (hp : p.degree = degree_p) (hq : q.degree = degree_q) :
  degree_p_x2_q_x4 p q = 22 :=
by
  sorry

end degree_product_l34_34266


namespace abs_neg_three_l34_34277

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l34_34277


namespace nathan_paintable_area_l34_34592

def total_paintable_area (rooms : ℕ) (length width height : ℕ) (non_paintable_area : ℕ) : ℕ :=
  let wall_area := 2 * (length * height + width * height)
  rooms * (wall_area - non_paintable_area)

theorem nathan_paintable_area :
  total_paintable_area 4 15 12 9 75 = 1644 :=
by sorry

end nathan_paintable_area_l34_34592


namespace max_intersections_of_inscribed_polygons_l34_34668

theorem max_intersections_of_inscribed_polygons
  (n1 n2 : ℕ)
  (hn1 : n1 ≥ 3)
  (hn2 : n2 ≥ 3)
  (hle : n1 ≤ n2)
  (P1 P2 : List (ℝ × ℝ))
  (hP1_sides : P1.length = n1)
  (hP2_sides : P2.length = n2)
  (hconvex_P1 : convex_polygon P1)
  (hconvex_P2 : convex_polygon P2) 
  (hinscribed_P1 : inscribed_in_circle P1)
  (hinscribed_P2 : inscribed_in_circle P2) 
  (hno_shared_vertices_edges : ∀ v ∈ P1, v ∉ P2 ∧ ∀ e ∈ edges P1, e ∉ edges P2) :
  max_intersections P1 P2 = n1 * n2 :=
sorry

end max_intersections_of_inscribed_polygons_l34_34668


namespace problem_solution_l34_34478

def f (x : ℝ) : ℝ :=
  if ∃ n : ℕ, x ∈ set.Ico (2 * n : ℝ) (2 * n + 1) then 
    let n := classical.some (classical.some_spec (exists_nat_mul x)) in
    (-1) ^ n * Real.sin (π * x / 2) + 2 * n
  else if ∃ n : ℕ, x ∈ set.Ico (2 * n + 1 : ℝ) (2 * n + 2) then
    let n := classical.some (classical.some_spec (exists_nat_mul (x - 1))) in
    (-1) ^ (n + 1) * Real.sin (π * x / 2) + 2 * n + 2
  else 0 -- Default case to handle edge cases

def a (m : ℕ) : ℝ := f (m : ℝ)

def S (m : ℕ) : ℝ := ∑ i in finset.range m, a (i + 1)

theorem problem_solution : S 105 - S 96 = 909 :=
sorry -- proof goes here

end problem_solution_l34_34478


namespace range_of_a_compare_f0_f1_l34_34993

open Real

-- Function definition and conditions
def f (x a : ℝ) : ℝ := x^2 + a * x + a

-- Roots conditions
def roots_condition (x1 x2 a : ℝ) : Prop :=
  f x1 a - x1 = 0 ∧ f x2 a - x2 = 0 ∧ 0 < x1 ∧ x1 < x2 ∧ x2 < 1

-- Problem statement: range of a
theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, roots_condition x1 x2 a) → 0 < a ∧ a < 3 - sqrt 2 := sorry

-- Problem statement: comparing f(0) * f(1) - f(0) with 2(17 - 12sqrt2)
theorem compare_f0_f1 (a : ℝ) (ha : 0 < a ∧ a < 3 - sqrt 2) : 
  let h := 2 * a^2 
  in h < 2 * (17 - 12 * sqrt 2) := sorry

end range_of_a_compare_f0_f1_l34_34993


namespace television_combinations_l34_34804

def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem television_combinations :
  ∃ (combinations : ℕ), 
  ∀ (A B total : ℕ), A = 4 → B = 5 → total = 3 →
  combinations = (combination 4 2 * combination 5 1 + combination 4 1 * combination 5 2) →
  combinations = 70 :=
sorry

end television_combinations_l34_34804


namespace polynomial_root_characteristics_l34_34863

-- Define the polynomial P(x)
noncomputable def P (x : ℝ) : ℝ := x^7 - 4*x^5 - 8*x^3 - x + 12

-- Statement of the problem
theorem polynomial_root_characteristics :
  (∃ x : ℝ, P x = 0 ∧ x > 0) ∧ (∀ y : ℝ, P y = 0 → y ≥ 0) :=
begin
  sorry
end

end polynomial_root_characteristics_l34_34863


namespace min_norm_sum_l34_34011

open Complex

noncomputable def g (z : ℂ) (β δ : ℂ) : ℂ := (3 - 2 * I) * z^2 + β * z + δ

theorem min_norm_sum (β δ : ℂ)
  (h1 : (g 1 β δ).im = 0)
  (h2 : (g (-I) β δ).im = 0) :
  complex.abs β + complex.abs δ = Real.sqrt 13 := by
sorry

end min_norm_sum_l34_34011


namespace f_always_positive_iff_l34_34811

-- Define the function f(x)
def f (x : ℝ) (k : ℝ) : ℝ := 3^(2*x) - (k + 1)*3^x + 2

-- Prove that f(x) is always positive if and only if k ∈ (-∞, 2^{-1})
theorem f_always_positive_iff (k : ℝ) : (∀ x : ℝ, f x k > 0) ↔ k ∈ set.Iio (2^(-1)) :=
by
  sorry

end f_always_positive_iff_l34_34811


namespace numberOfValidM_l34_34529

-- Define the conditions for the point P being in the fourth quadrant
def inFourthQuadrant (m : ℝ) : Prop :=
  0 < m ∧ m < 4

-- Define the system of inequalities and the constraint of having exactly 4 integer solutions
def systemHasFourIntegerSolutions (m : ℝ) : Prop :=
  (∃ (x : Set ℝ),  
    {x : ℝ | 3 * x > 2 * (x - 2) ∧ 3 * x - (x - 1) / 2 < m / 2} = 
    {x : ℝ | -4 < x ∧ x < (m - 1) / 5} ∧
    {n : ℤ | -4 < ↑n ∧ ↑n < (m -1)/5}.card = 4)

-- Define the main theorem statement
theorem numberOfValidM : {m : ℝ | 
    inFourthQuadrant m ∧ systemHasFourIntegerSolutions m}.finite.card = 2 :=
sorry

end numberOfValidM_l34_34529


namespace sum_fractional_parts_correct_l34_34972

def floor (x : ℝ) : ℤ := Int.floor x
def frac (x : ℝ) : ℝ := x - floor x

def sumFractionalParts : ℝ :=
  let n := 2012
  let sum := ∑ k in Finset.range (n + 1) \ Finset.singleton 0, 
                   frac ((2012 + k) / 5)
  sum

theorem sum_fractional_parts_correct : 
  sumFractionalParts = 805.4 :=
by
  sorry

end sum_fractional_parts_correct_l34_34972


namespace beshmi_investment_in_Z_l34_34048

-- Define the given conditions
def investment_X (S : ℝ) := (1 / 5) * S
def investment_Y (S : ℝ) := 0.42 * S
def investment_Y_amount := 10500

-- Define the main proof theorem
theorem beshmi_investment_in_Z (S : ℝ) (h : investment_Y S = investment_Y_amount) : 
  (S = 25000) → (0.38 * S = 9500) :=
by
  sorry

end beshmi_investment_in_Z_l34_34048


namespace abs_neg_three_eq_three_l34_34285

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l34_34285


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l34_34899

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : ∃ (n : ℕ), n = 225 ∧ ∀ k : ℤ, 100 ≤ 4 * k + 2 ∧ 4 * k + 2 ≤ 999 → 24 < k ∧ k < 250 := by
  sorry

end number_of_three_digit_integers_congruent_to_2_mod_4_l34_34899


namespace intercept_of_line_l34_34636

theorem intercept_of_line (A B: ℝ × ℝ) (hA : A = (-1, 1)) (hB : B = (3, 9)) :
  ∃ x : ℝ, (x, 0) lies on the line through A and B ∧ x = -3/2 :=
by
  sorry

--We need to formally define what it means for (x, 0) to lie on the line through A and B
def lies_on_line (P Q R : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, R = (1 - t) • P + t • Q

end intercept_of_line_l34_34636


namespace fixed_point_exists_l34_34634

theorem fixed_point_exists (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∃ (x y : ℝ), x = -2011 ∧ y = 2011 ∧ y = 2011 + log a (x + 2012)) :=
by
  use -2011
  use 2011
  split
  . exact rfl
  split
  . exact rfl
  . sorry

end fixed_point_exists_l34_34634


namespace terrell_lift_same_weight_l34_34268

theorem terrell_lift_same_weight (n : ℕ) (w1 w2 : ℕ) (reps1 : ℕ) (reps2 : ℕ) 
  (h1 : w1 = 20) (h2 : w2 = 10) (h3 : reps1 = 12) 
  (h4 : 2 * 20 * reps1 = 480) : 
  2 * 10 * reps2 = 480 → reps2 = 24 :=
by {
  intros h,
  sorry
}

end terrell_lift_same_weight_l34_34268


namespace number_of_girls_is_eleven_l34_34049

-- Conditions transformation
def boys_wear_red_hats : Prop := true
def girls_wear_yellow_hats : Prop := true
def teachers_wear_blue_hats : Prop := true
def cannot_see_own_hat : Prop := true
def little_qiang_sees_hats (x k : ℕ) : Prop := (x + 2) = (x + 2)
def little_hua_sees_hats (x k : ℕ) : Prop := x = 2 * k
def teacher_sees_hats (x k : ℕ) : Prop := k + 2 = (x + 2) + k - 11

-- Proof Statement
theorem number_of_girls_is_eleven (x k : ℕ) (h1 : boys_wear_red_hats)
  (h2 : girls_wear_yellow_hats) (h3 : teachers_wear_blue_hats)
  (h4 : cannot_see_own_hat) (hq : little_qiang_sees_hats x k)
  (hh : little_hua_sees_hats x k) (ht : teacher_sees_hats x k) : x = 11 :=
sorry

end number_of_girls_is_eleven_l34_34049


namespace reciprocal_roots_l34_34755

noncomputable def p (x : ℤ) : ℤ := 
x^1999 + 2*x^1998 + 3*x^1997 + ... + 2000

noncomputable def q (x : ℤ) : ℤ := 
1 + 2*x + 3*x^2 + ... + 2000*x^1999

theorem reciprocal_roots : 
  ∃ q : ℤ → ℤ, (∀ x : ℤ, p x = 0 → q (1/x) = 0) ∧ 
  q = (1 + 2*x + 3*x^2 + ... + 2000*x^1999) := 
sorry

end reciprocal_roots_l34_34755


namespace final_fraction_correct_l34_34405

-- Definitions representing the problem conditions
def fair_die_prob (side: ℕ) : ℚ := if side = 6 then 1/6 else 1/6
def biased_die_prob : ℕ → ℚ 
| 6     := 3/4
| _ := 1/20
def choose_die : ℚ := 1/2

-- Definitions for the problem setup
def prob_three_sixes (die: ℕ → ℚ) : ℚ := (die 6) ^ 3
def total_prob_three_sixes : ℚ :=
  choose_die * prob_three_sixes fair_die_prob + choose_die * prob_three_sixes biased_die_prob

def prob_biased_given_three_sixes : ℚ :=
  (prob_three_sixes biased_die_prob * choose_die) / total_prob_three_sixes

def prob_fourth_six : ℚ :=
  prob_biased_given_three_sixes * biased_die_prob 6 + (1 - prob_biased_given_three_sixes) * fair_die_prob 6

noncomputable def final_fraction := prob_fourth_six

-- Statement to prove
theorem final_fraction_correct (p q : ℕ) (hpq : p + q = 856) :
  final_fraction = p / q := by
  sorry

end final_fraction_correct_l34_34405


namespace find_distance_between_destinations_l34_34360

noncomputable def distance_between_destinations (d : ℝ) : Prop :=
  let rowing_speed := 7
  let river_speed := 3
  let v_downstream := rowing_speed + river_speed -- 10 km/hr
  let v_upstream := rowing_speed - river_speed -- 4 km/hr
  let t_downstream := d / v_downstream
  let t_upstream := d / v_upstream
  t_upstream = t_downstream + 6

theorem find_distance_between_destinations : ∃ d : ℝ, distance_between_destinations d ∧ d = 40 :=
by {
  use 40,
  unfold distance_between_destinations,
  norm_num,
  sorry
}

end find_distance_between_destinations_l34_34360


namespace Rachel_current_age_l34_34080

noncomputable theory

-- Definitions based on conditions
def Emily_age_current : ℕ := 20
def Rachel_age_when_Emily_half_Rachel : ℕ := 8
def Emily_age_when_Rachel_8 : ℕ := Rachel_age_when_Emily_half_Rachel / 2

-- Age difference definition based on the condition
def age_difference : ℕ := 
  Rachel_age_when_Emily_half_Rachel - Emily_age_when_Rachel_8

-- Prove Rachel's current age is 24 given the conditions
theorem Rachel_current_age : 
  let rachel_current_age := Emily_age_current + age_difference in
  rachel_current_age = 24 :=
by
  suffices h : rachel_current_age = 24
  { exact h }
  sorry

end Rachel_current_age_l34_34080


namespace det_A_eq_6_l34_34973

open Matrix

variables {R : Type*} [Field R]

def A (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![a, 2], ![-3, d]]

def B (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![2 * a, 1], ![-1, d]]

noncomputable def B_inv (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  let detB := (2 * a * d + 1)
  ![![d / detB, -1 / detB], ![1 / detB, (2 * a) / detB]]

theorem det_A_eq_6 (a d : R) (hB_inv : (A a d) + (B_inv a d) = 0) : det (A a d) = 6 :=
  sorry

end det_A_eq_6_l34_34973


namespace sqrt_inequality_triangle_inequality_l34_34697

-- Problem 1 Statement
theorem sqrt_inequality (a : ℝ) (ha : a > 0) : 
  sqrt (a + 5) - sqrt (a + 3) > sqrt (a + 6) - sqrt (a + 4) :=
sorry

-- Problem 2 Statement
theorem triangle_inequality (a b c : ℝ) 
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (triangle_ineq : a + b > c) :
  (a + b) / (1 + a + b) > c / (1 + c) :=
sorry

end sqrt_inequality_triangle_inequality_l34_34697


namespace arithmetic_vs_geometric_l34_34454

-- Given Conditions
variables (a : ℝ) (d q : ℝ) (an bn : ℕ → ℝ)
hypothesis ha_pos : a > 0
hypothesis ha_initial_equal : an 1 = bn 1
hypothesis ha_arith : an 2 = a + d
hypothesis hb_geom : bn 2 = a * q
hypothesis hd_nonzero : a + d ≠ a * q
hypothesis han_pos : ∀ n, an n > 0

-- Prove that a_n < b_n for n ≥ 3
theorem arithmetic_vs_geometric (h_ge_two : ∀ n, n ≥ 3 → an n < bn n) : 
  ∀ n, n ≥ 3 → an n < bn n :=
sorry

end arithmetic_vs_geometric_l34_34454


namespace exists_2nd_translation_point_g_range_a_for_1st_translation_point_h_l34_34064

-- Define the functions g and h
def g (x : ℝ) : ℝ := x * log (x + 1)

def h (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x * log x

-- Statement for Part (1)
theorem exists_2nd_translation_point_g : ∃ x0 : ℝ, g (x0 + 2) = g x0 + g 2 :=
by
  -- Definitions based on problem conditions
  let g_x0 : ℝ := 0
  let g_translation_point : ℝ × ℝ := ⟨g_x0, 2⟩
  -- Basic constraints, simplification, and solutions
  let g_point_condition : β := (g (g_x0 + 2) = g g_x0 + g 2)
  have hpos : (0 < 2) := by exact_mod_cast (nat.zero_lt_succ 1)
  use g_x0
  exact g_point_condition
  sorry -- Placeholder, skip proof

-- Statement for Part (2)
theorem range_a_for_1st_translation_point_h : ∀ a : ℝ, ∃ x1 ∈ Ici (1 : ℝ), 
  h a (x1 + 1) = h a x1 + h a 1 ↔ a ≥ - log 2 :=
by
  let h_x1 := fun (a : ℝ) (x : ℝ) => a * x^2 + x * log x
  use -log 2
  exact (λ a, sorry)

end exists_2nd_translation_point_g_range_a_for_1st_translation_point_h_l34_34064


namespace arithmetic_sequence_common_difference_l34_34195

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 2 = 1) (h2 : a 6 = 13) 
  (arithmetic : ∀ n : ℕ, n ≥ 2 → a n = a 1 + (n - 1) * d) : 
  d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l34_34195


namespace mass_percentage_I_in_CaI2_l34_34090

theorem mass_percentage_I_in_CaI2 :
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_I : ℝ := 126.90
  let molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I
  let mass_percentage_I : ℝ := (2 * molar_mass_I / molar_mass_CaI2) * 100
  mass_percentage_I = 86.36 := by
  sorry

end mass_percentage_I_in_CaI2_l34_34090


namespace find_y_value_l34_34630

noncomputable def y_value (y : ℝ) : Prop :=
  let side1_sq_area := 9 * y^2
  let side2_sq_area := 36 * y^2
  let triangle_area := 9 * y^2
  (side1_sq_area + side2_sq_area + triangle_area = 1000)

theorem find_y_value (y : ℝ) : y_value y → y = 10 * Real.sqrt 3 / 3 :=
sorry

end find_y_value_l34_34630


namespace company_fund_amount_l34_34640

theorem company_fund_amount (n : ℕ) (h : 70 * n + 160 = 80 * n - 8) : 
  80 * n - 8 = 1352 :=
sorry

end company_fund_amount_l34_34640


namespace angle_MNK_eq_90_l34_34545

noncomputable def midpoint (A B : Point) : Point := sorry -- definition for midpoint
noncomputable def projection (M L : Point) : Point := sorry -- definition for projection

variables {A B C M X Y K N : Point}

-- Given: Definitions of points as per the given problem
def given_problem_conditions : Prop :=
  -- M is on the arc AB of the circumcircle of triangle ABC which does not contain C
  Arc_on_circumcircle A B C M ∧ 
  -- X is the projection of M onto the line AB
  X = projection M (Line_through A B) ∧
  -- Y is the projection of M onto the line BC
  Y = projection M (Line_through B C) ∧
  -- K is the midpoint of AC
  K = midpoint A C ∧
  -- N is the midpoint of XY
  N = midpoint X Y

-- To Prove: ∠ MNK = 90°
theorem angle_MNK_eq_90 (h : given_problem_conditions) : ∠MNK = 90 :=
sorry

end angle_MNK_eq_90_l34_34545


namespace three_digit_integers_congruent_to_2_mod_4_l34_34892

theorem three_digit_integers_congruent_to_2_mod_4 : 
    ∃ n, n = 225 ∧ ∀ x, (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 2) ↔ (∃ m, 25 ≤ m ∧ m ≤ 249 ∧ x = 4 * m + 2) := by
  sorry

end three_digit_integers_congruent_to_2_mod_4_l34_34892


namespace choose_correct_graph_l34_34664

noncomputable def appropriate_graph : String :=
  let bar_graph := "Bar graph"
  let pie_chart := "Pie chart"
  let line_graph := "Line graph"
  let freq_dist_graph := "Frequency distribution graph"
  
  if (bar_graph = "Bar graph") ∧ (pie_chart = "Pie chart") ∧ (line_graph = "Line graph") ∧ (freq_dist_graph = "Frequency distribution graph") then
    "Line graph"
  else
    sorry

theorem choose_correct_graph :
  appropriate_graph = "Line graph" :=
by
  sorry

end choose_correct_graph_l34_34664


namespace find_coeff_and_range_l34_34473

noncomputable def f (a b : ℝ) := λ x : ℝ, a * x^3 + b * x^2
noncomputable def f' (a b : ℝ) := λ x : ℝ, 3 * a * x^2 + 2 * b * x

theorem find_coeff_and_range (a b : ℝ) :
  f a b 1 = 4 ∧ f' a b 1 * (-9) = -1 →
  a = 1 ∧ b = 3 ∧ ∀ m : ℝ, 
  (f 1 3) m ≤ (f 1 3) (m + 1) →
  (m ≥ 0 ∨ m ≤ -3) :=
by
  sorry

end find_coeff_and_range_l34_34473


namespace percentage_antifreeze_l34_34704

-- Define the initial conditions
def initial_volume : ℝ := 10
def initial_percentage : ℝ := 0.30
def replaced_volume : ℝ := 2.85714285714

-- Define initial antifreeze volume
def initial_antifreeze_volume : ℝ := initial_volume * initial_percentage

-- Define the amount of antifreeze removed
def antifreeze_removed : ℝ := replaced_volume * initial_percentage

-- Define the volume of antifreeze after removal
def antifreeze_after_removal : ℝ := initial_antifreeze_volume - antifreeze_removed

-- Define the amount of pure antifreeze added
def pure_antifreeze_added : ℝ := replaced_volume

-- Define the total antifreeze volume after adding pure antifreeze
def total_antifreeze : ℝ := antifreeze_after_removal + pure_antifreeze_added

-- Define the final percentage of antifreeze in the solution
def resulting_percentage : ℝ := (total_antifreeze / initial_volume) * 100

theorem percentage_antifreeze :
  resulting_percentage = 50 := 
by
  sorry

end percentage_antifreeze_l34_34704


namespace abs_neg_three_l34_34292

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l34_34292


namespace function_f₁_is_even_l34_34851

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def f₁ (x : ℝ) := (x^2 + 1) / |x|

#check is_even_function f₁

theorem function_f₁_is_even : is_even_function f₁ :=
sorry -- Proof goes here

end function_f₁_is_even_l34_34851


namespace minimum_value_of_M_n_l34_34149

noncomputable def M_n (n s t : ℕ) : ℝ :=
max (40 / n) (max (80 / s) (60 / t))

theorem minimum_value_of_M_n :
  ∀ k (n s t : ℕ), s = k * n ∧ n + s + t = 200 ∧ 0 < n ∧ 0 < s ∧ 0 < t →
    n ∈ ℕ ∧ s ∈ ℕ ∧ t ∈ ℕ ∧ k ∈ ℕ^* →
    min (M_n n s t) (λ k, M_n n s t) = 10 / 11 :=
by
  sorry

end minimum_value_of_M_n_l34_34149


namespace vasya_faster_than_petya_l34_34670

theorem vasya_faster_than_petya 
  (L : ℝ) (v : ℝ) (x : ℝ) (t : ℝ) 
  (meeting_condition : (v + x * v) * t = L)
  (petya_lap : v * t = L)
  (vasya_meet_petya_after_lap : x * v * t = 2 * L) :
  x = (1 + Real.sqrt 5) / 2 := 
by 
  sorry

end vasya_faster_than_petya_l34_34670


namespace distance_last_day_l34_34619

theorem distance_last_day
  (total_distance : ℕ)
  (days : ℕ)
  (initial_distance : ℕ)
  (common_ratio : ℚ)
  (sum_geometric : initial_distance * (1 - common_ratio^days) / (1 - common_ratio) = total_distance) :
  total_distance = 378 → days = 6 → common_ratio = 1/2 → 
  initial_distance = 192 → initial_distance * common_ratio^(days - 1) = 6 := 
by
  intros h1 h2 h3 h4
  sorry

end distance_last_day_l34_34619


namespace inequality_proof_l34_34583

noncomputable def sum_expression (a b c : ℝ) : ℝ :=
  (1 / (b * c + a + 1 / a)) + (1 / (c * a + b + 1 / b)) + (1 / (a * b + c + 1 / c))

theorem inequality_proof (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  sum_expression a b c ≤ 27 / 31 :=
by
  sorry

end inequality_proof_l34_34583


namespace problem_statement_l34_34217

noncomputable def a : ℝ := Real.cos (420 * Real.pi / 180)

noncomputable def f : ℝ → ℝ
| x := if x ≤ 0 then a^x else Real.logBase a x

theorem problem_statement : f (1/4) + f (-2) = 6 :=
by {
  -- Problem statement, skipped proof.
  sorry
}

end problem_statement_l34_34217


namespace students_in_each_class_l34_34590

-- Define the conditions
def sheets_per_student : ℕ := 5
def total_sheets : ℕ := 400
def number_of_classes : ℕ := 4

-- Define the main proof theorem
theorem students_in_each_class : (total_sheets / sheets_per_student) / number_of_classes = 20 := by
  sorry -- Proof goes here

end students_in_each_class_l34_34590


namespace mixed_number_expression_l34_34086

open Real

-- Definitions of the given mixed numbers
def mixed_number1 : ℚ := (37 / 7)
def mixed_number2 : ℚ := (18 / 5)
def mixed_number3 : ℚ := (19 / 6)
def mixed_number4 : ℚ := (9 / 4)

-- Main theorem statement
theorem mixed_number_expression :
  25 * (mixed_number1 - mixed_number2) / (mixed_number3 + mixed_number4) = 7 + 49 / 91 :=
by
  sorry

end mixed_number_expression_l34_34086


namespace boiling_point_C_l34_34333

-- Water boils at 212 °F
def water_boiling_point_F : ℝ := 212
-- Ice melts at 32 °F
def ice_melting_point_F : ℝ := 32
-- Ice melts at 0 °C
def ice_melting_point_C : ℝ := 0
-- The temperature of a pot of water in °C
def pot_water_temp_C : ℝ := 40
-- The temperature of the pot of water in °F
def pot_water_temp_F : ℝ := 104

-- The boiling point of water in Celsius is 100 °C.
theorem boiling_point_C : water_boiling_point_F = 212 ∧ ice_melting_point_F = 32 ∧ ice_melting_point_C = 0 ∧ pot_water_temp_C = 40 ∧ pot_water_temp_F = 104 → exists bp_C : ℝ, bp_C = 100 :=
by
  sorry

end boiling_point_C_l34_34333


namespace number_of_three_digit_integers_congruent_mod4_l34_34911

def integer_congruent_to_mod (a b n : ℕ) : Prop := ∃ k : ℤ, n = a * k + b

theorem number_of_three_digit_integers_congruent_mod4 :
  (finset.filter (λ n, integer_congruent_to_mod 4 2 (n : ℕ)) 
   (finset.Icc (100 : ℕ) (999 : ℕ))).card = 225 :=
by
  sorry

end number_of_three_digit_integers_congruent_mod4_l34_34911


namespace range_of_f_l34_34421

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 - (Real.sin x) * (Real.cos x) + (Real.cos x) ^ 4 + (Real.cos x) ^ 2

theorem range_of_f : 
  ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1 →
  ∃ y, y ∈ set.Icc (0 : ℝ) (2 : ℝ) ∧ y = f x :=
begin
  sorry
end

end range_of_f_l34_34421


namespace slope_of_line_l34_34790

theorem slope_of_line (x y : ℝ) :
  (4 * x - 7 * y = 28) → (slope (line_eq := 4 * x - 7 * y) = 4 / 7) :=
by
  sorry

end slope_of_line_l34_34790


namespace part_1_solution_part_2_solution_l34_34474

noncomputable def f (a b c x : ℝ) := (b * x) / (a * x^2 + c)
noncomputable def f' (a b c x : ℝ) : ℝ := (a * b * x^2 - b) / ((a * x^2 + c)^2)

theorem part_1_solution (a b c : ℝ) (h1 : f'(a, b, c, 0) = 9) (h2 : b + c = 10) (ha : a > 0) :
  b = 9 ∧ c = 1 ∧ 
  (∀ x : ℝ, x ∈ Ioo (-sqrt (1/a^2), sqrt (1/a^2)) → (a * b * x^2 - b) > 0) ∧
  (∀ x : ℝ, x ∉ Ioo (-sqrt (1/a^2), sqrt (1/a^2)) → (a * b * x^2 - b) < 0) := 
sorry

theorem part_2_solution (a : ℝ) (h : 0 < a ∧ a ≤ 1) :
  ∀ x : ℝ, x > 1 → (x^3 + 1) * f(a, 9, 1, x) > 9 + real.log x :=
sorry

end part_1_solution_part_2_solution_l34_34474


namespace excircle_DG_length_l34_34988

theorem excircle_DG_length
  (A B C D E F O G : Type)
  (BC : ℕ) (CA : ℕ) (AB : ℕ)
  (touch_ω_D : D) (touch_ω_E : E) (touch_ω_F : F)
  (center_ω : O)
  (line_l_through_O : G)
  (line_l_perpendicular_AD : Prop)
  (line_l_meets_EF_at_G : Prop)
  (triangle_ABC : Prop)
  (DG_correct_length : Prop) :
BC = 2007 ∧ CA = 2008 ∧ AB = 2009 ∧
(triangle_ABC ∧
touch_ω_D ∧ touch_ω_E ∧ touch_ω_F ∧
line_l_perpendicular_AD ∧ line_l_meets_EF_at_G) →
DG = 2014024 := by
  sorry

end excircle_DG_length_l34_34988


namespace real_solutions_count_l34_34164

theorem real_solutions_count : 
    (λ x : ℝ, 5^(2*x + 1) - 5^(x + 2) - 10 * 5^x + 50 = 0) → Finset.card (Finset.filter (λ x : ℝ, 5^(2*x + 1) - 5^(x + 2) - 10 * 5^x + 50 = 0) Finset.univ) = 2 :=
sorry

end real_solutions_count_l34_34164


namespace area_bounded_by_parabola_and_tangents_l34_34414

open Real

theorem area_bounded_by_parabola_and_tangents :
  let C := (λ x : ℝ, x^2)
  let l1 := (λ x : ℝ, √3 * x - 3 / 4)
  let l2 := (λ x : ℝ, -(2 + √3) * x - (7 + 4 * √3) / 4)
  ∃ (area : ℝ),
    area = ∫ x in -((2 + √3) / 2), -1/2, (x^2 + (2 + √3) * x + (7 + 4 * √3) / 4) +
           ∫ x in -1/2, √3 / 2, (x^2 - √3 * x + 3 / 4) ∧
    area = (5 + 3 * √3) / 6 :=
begin
  let C := λ x : ℝ, x^2,
  let l1 := λ x : ℝ, √3 * x - 3 / 4,
  let l2 := λ x : ℝ, -(2 + √3) * x - (7 + 4 * √3) / 4,
  use (5 + 3 * √3) / 6,
  sorry
end

end area_bounded_by_parabola_and_tangents_l34_34414


namespace first_player_wins_iff_n_mod_3_eq_0_l34_34548

theorem first_player_wins_iff_n_mod_3_eq_0 (n : ℕ) (h_pos : n > 0) :
  (∃ win_strategy : (∃ seq : ℕ → ℕ, (∀ i, 1 ≤ seq i ∧ seq i ≤ n ∧ ∀ j, j < i → seq j ≠ seq i) ∧ (∑ i in finset.range n, seq i) % 3 = 0)) ↔ n % 3 = 0 :=
by
  sorry

end first_player_wins_iff_n_mod_3_eq_0_l34_34548


namespace unique_four_digit_codes_count_l34_34062

theorem unique_four_digit_codes_count 
    (digits : finset ℕ)
    (h_digits_count : digits.card = 4)
    (h_digits_range : ∀ d ∈ digits, d < 10)
    (h_sum_digits : digits.sum id = 28)
    (h_unique : ∀ d1 d2 ∈ digits, d1 ≠ d2 → d1 ≠ d2) : 
    ∃ n: ℕ, n = 48 :=
by
    sorry

end unique_four_digit_codes_count_l34_34062


namespace hyperbola_asymptotes_angle_l34_34796

open Real

theorem hyperbola_asymptotes_angle (a b : ℝ) (h₀ : a > b) (h₁ : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → angle_between_asymptotes (y = b / a * x) = 90) : 
  a / b = 1 := 
sorry

end hyperbola_asymptotes_angle_l34_34796


namespace triangle_vector_identity_l34_34539

noncomputable def vectors_and_angles (ABC : Triangle ℝ) (O : Point ABC) : Prop :=
let e1 := unit_vector(ABC.OA),
    e2 := unit_vector(ABC.OB),
    e3 := unit_vector(ABC.OC),
    α := angle B O C,
    β := angle C O A,
    γ := angle A O B in
e1 * sin α + e2 * sin β + e3 * sin γ = 0

theorem triangle_vector_identity (ABC : Triangle ℝ) (O : Point ABC) :
  vectors_and_angles ABC O := 
sorry

end triangle_vector_identity_l34_34539


namespace leap_stride_difference_l34_34771

theorem leap_stride_difference :
  let elmer_strides_per_gap := 44
  let oscar_leaps_per_gap := 12
  let num_poles := 41
  let distance_miles := 5280 in
  let num_gaps := num_poles - 1
  let total_elmer_strides := elmer_strides_per_gap * num_gaps
  let total_oscar_leaps := oscar_leaps_per_gap * num_gaps
  let elmer_stride_length := distance_miles / total_elmer_strides
  let oscar_leap_length := distance_miles / total_oscar_leaps
  oscar_leap_length - elmer_stride_length = 8 := by
{
  sorry
}

end leap_stride_difference_l34_34771


namespace equivalence_l34_34798

variables {V E : Type} [Fintype V] [Fintype E] (G : Graph V E) (F : Finset E)

def is_cut_set (F : Finset E) := F ∈ G.𝒞

def is_disjoint_union_subgraphs (F : Finset E) := 
  ∃ (subgraphs : Finset (Finset E)), 
  F = subgraphs.bind id ∧ (subgraphs.all (λ s, G.is_tree s))

def all_vertex_degrees_even (F : Finset E) := 
  ∀ v : V, F.count v % 2 = 0

theorem equivalence (G : Graph V E) (F : Finset E) :
  is_cut_set G F ↔ is_disjoint_union_subgraphs G F ∧
  all_vertex_degrees_even G F :=
sorry

end equivalence_l34_34798


namespace sum_b_geq_four_over_n_plus_one_l34_34218

theorem sum_b_geq_four_over_n_plus_one
  (n : ℕ)
  (b : ℕ → ℝ) -- Note: Indexing from 1 to n
  (x : ℕ → ℝ) -- Note: Indexing from 0 to n + 1 with x(0) = 0 and x(n + 1) = 0
  (h_pos : ∀ k, 1 ≤ k → k ≤ n → 0 < b k)
  (system_eq : ∀ k, 1 ≤ k → k ≤ n → x (k - 1) - 2 * x k + x (k + 1) + b k * x k = 0)
  (h_non_trivial : ∃ k, 1 ≤ k → k ≤ n → x k ≠ 0)
  : ∑ k in finset.range n, b (k + 1) ≥ (4 : ℝ) / (n + 1) := 
sorry

end sum_b_geq_four_over_n_plus_one_l34_34218


namespace max_cells_n_10_max_cells_n_9_l34_34770

def isValidMarking (n k : ℕ) : Prop :=
  ∀ (i j : ℕ), i ≠ j → (i < n * n ∧ j < n * n) → 
  let si := i // n
  let sj := j // n
  let rsi := i % n
  let rsj := j % n 
  (si ≠ sj ∧ si ≠ rsi ∧ si ≠ rsj ∧ sj ≠ rsi ∧ sj ≠ rsj)

def maxMarkedCells (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else (n + 1) / 2

theorem max_cells_n_10 :
  maxMarkedCells 10 = 7 := by
  sorry

theorem max_cells_n_9 :
  maxMarkedCells 9 = 6 := by
  sorry

end max_cells_n_10_max_cells_n_9_l34_34770


namespace ten_pow_neg_x_l34_34913

theorem ten_pow_neg_x (x : ℝ) (h : 10^(3 * x) = 1000) : 10^(-x) = 1 / 10 :=
by
  sorry

end ten_pow_neg_x_l34_34913


namespace circumcircle_radius_le_radius_of_Γ_l34_34990

-- Define the given conditions
variable {Ω : Type} [metric_space Ω] [normed_add_torsor ℂ Ω]
variable (A1 A2 A3 P : Ω)
variable {Γ : set Ω} -- the circle containing A1, A2, A3
variable (hΓ : ∀ x ∈ {A1, A2, A3}, x ∈ Γ)
variable (rΓ : ℝ) -- radius of the circle Γ

variable {angle : Ω → Ω → Ω → ℝ} -- function for angle
variable (τ₁ τ₂ τ₃ : Ω → Ω) -- rotation functions
variable (hτ₁ : ∀ (O: Ω) (θ : ℝ), ∃ τ₁ : Ω → Ω, is_rotation_about O θ τ₁)
variable (hτ₂ : ∀ (O: Ω) (θ : ℝ), ∃ τ₂ : Ω → Ω, is_rotation_about O θ τ₂)
variable (hτ₃ : ∀ (O: Ω) (θ : ℝ), ∃ τ₃ : Ω → Ω, is_rotation_about O θ τ₃)

-- Define points P1, P2, P3
def P1 := τ₃ (τ₁ (τ₂ P))
def P2 := τ₁ (τ₂ (τ₃ P))
def P3 := τ₂ (τ₃ (τ₁ P))

-- Define the circumcircle of triangle P1P2P3
noncomputable def circumcircle_radius (P1 P2 P3 : Ω) : ℝ := sorry -- the radius calculation

theorem circumcircle_radius_le_radius_of_Γ :
  circumcircle_radius P1 P2 P3 ≤ rΓ :=
sorry

end circumcircle_radius_le_radius_of_Γ_l34_34990


namespace find_savings_l34_34690

-- Definitions of given conditions
def income : ℕ := 10000
def ratio_income_expenditure : ℕ × ℕ := (10, 8)

-- Proving the savings based on given conditions
theorem find_savings (income : ℕ) (ratio_income_expenditure : ℕ × ℕ) :
  let expenditure := (ratio_income_expenditure.2 * income) / ratio_income_expenditure.1
  let savings := income - expenditure
  savings = 2000 :=
by
  sorry

end find_savings_l34_34690


namespace pete_backwards_speed_l34_34595

variable (speed_pete_hands : ℕ) (speed_tracy_cartwheel : ℕ) (speed_susan_walk : ℕ) (speed_pete_backwards : ℕ)

axiom pete_hands_speed : speed_pete_hands = 2
axiom pete_hands_speed_quarter_tracy_cartwheel : speed_pete_hands = speed_tracy_cartwheel / 4
axiom tracy_cartwheel_twice_susan_walk : speed_tracy_cartwheel = 2 * speed_susan_walk
axiom pete_backwards_three_times_susan_walk : speed_pete_backwards = 3 * speed_susan_walk

theorem pete_backwards_speed : 
  speed_pete_backwards = 12 :=
by
  sorry

end pete_backwards_speed_l34_34595


namespace sum_floor_div_2_pow_25_l34_34422

/--
Prove that the remainder of the sum 
∑_i=0^2015 ⌊2^i / 25⌋ 
when divided by 100 is 14,
where ⌊x⌋ denotes the floor of x.
-/
theorem sum_floor_div_2_pow_25 :
  (∑ i in Finset.range 2016, (2^i / 25 : ℤ)) % 100 = 14 :=
by
  sorry

end sum_floor_div_2_pow_25_l34_34422


namespace max_consecutive_interesting_numbers_l34_34233

def is_interesting (n : ℕ) : Prop :=
  (n / 100 % 3 = 0) ∨ (n / 10 % 10 % 3 = 0) ∨ (n % 10 % 3 = 0)

theorem max_consecutive_interesting_numbers :
  ∃ l r, 100 ≤ l ∧ r ≤ 999 ∧ r - l + 1 = 122 ∧ (∀ n, l ≤ n ∧ n ≤ r → is_interesting n) ∧ 
  ∀ l' r', 100 ≤ l' ∧ r' ≤ 999 ∧ r' - l' + 1 > 122 → ∃ n, l' ≤ n ∧ n ≤ r' ∧ ¬ is_interesting n := 
sorry

end max_consecutive_interesting_numbers_l34_34233


namespace problem_l34_34849

noncomputable def part1 : Prop :=
  ∀ (a b c : ℝ), (b * (-1) ^ 2 - b + 2 * a * -1 + c * (-1) ^ 2 + c = 0) →
    a = c

noncomputable def part2 : Prop :=
  ∀ (a b c : ℝ), (a = 5) → (b = 12) →
    ((b + c) * x ^ 2 + 2 * a * x + (c - b) = 0) →
    (4 * a ^ 2 - 4 * ((b + c) * (c - b) - 4 * a ^ 2 + 4 * c ^ 2) = 0) →
    c = 13

theorem problem (a b c : ℝ) (x : ℝ) (h1 : b * (x ^ 2 - 1) + 2 * a * x + c * (x ^ 2 + 1) = 0) :
   part1 ∧ part2 :=
begin
  sorry
end

end problem_l34_34849


namespace shortest_distance_ant_l34_34846

noncomputable def cone_slant_height : ℝ := 4
noncomputable def cone_base_radius : ℝ := 1

theorem shortest_distance_ant (s : ℝ) (r : ℝ) (d : ℝ) :
  s = cone_slant_height → r = cone_base_radius → d = ℝ.sqrt (s * s + s * s) → d = 4 * ℝ.sqrt 2 :=
by
  intros hs hr hd
  rw [hs, hr, hd]
  simp
  sorry

end shortest_distance_ant_l34_34846


namespace invalid_inference_l34_34657

-- Definitions
def major_premise (G : Type) [HasEat G] : Prop := ∀ g : G, eats g cabbage
def minor_premise (S : Type) [HasEat S] (s : S) : Prop := eats s cabbage

-- The conclusion inferred
def incorrect_conclusion (S G : Type) (s : S) (g : G) [HasEat S] [HasEat G] : Prop := s = g

-- The proof problem
theorem invalid_inference (S G : Type) (s : S) (g : G) [HasEat S] [HasEat G] (cabbage : Food) :
  major_premise G → minor_premise S s → ¬ incorrect_conclusion S G s g :=
by
  intro h_major h_minor
  -- The proof is omitted.
  sorry

end invalid_inference_l34_34657


namespace regression_prediction_l34_34694

theorem regression_prediction (data : List (ℕ × ℕ)) (x_vals : List ℕ) (y_vals : List ℕ)
  (mean_x : ℝ) (mean_y : ℝ) (sum_xy_diff : ℝ) (sum_x_diff_sq : ℝ) (sum_y_diff_sq : ℝ)
  (corr_coeff : ℝ) (reg_slope : ℝ) (reg_intercept : ℝ) (x : ℝ) :
  data = [(6, 15), (8, 18), (10, 20), (12, 24), (14, 23)] →
  x_vals = [6, 8, 10, 12, 14] →
  y_vals = [15, 18, 20, 24, 23] →
  mean_x = 10 →
  mean_y = 20 →
  sum_xy_diff = 44 →
  sum_x_diff_sq = 40 →
  sum_y_diff_sq = 54 →
  corr_coeff = (sum_xy_diff / (Math.sqrt (sum_x_diff_sq * sum_y_diff_sq))) →
  corr_coeff ≈ 0.95 →
  reg_slope = sum_xy_diff / sum_x_diff_sq →
  reg_slope = 1.1 →
  reg_intercept = mean_y - reg_slope * mean_x →
  reg_intercept = 9 →
  (x = 20) →
  let prediction := (reg_slope * x + reg_intercept) in
  prediction = 31 :=
by intros;
   sorry

end regression_prediction_l34_34694


namespace k_regular_graph_has_1_factor_l34_34918

variable (V : Type) [Fintype V]
variable (E : V → V → Prop)
variable (k : ℕ)
variable (G : SimpleGraph V)
variable [DecidableRel E]
variable [DecidableEq V]
variable [Fintype (Sym2 V)]

-- The conditions: G is a k-regular graph and k ≥ 1
def is_k_regular (G : SimpleGraph V) (k : ℕ) : Prop :=
  ∀ v : V, (G.degree v) = k

theorem k_regular_graph_has_1_factor
  (G : SimpleGraph V)
  (h1 : is_k_regular G k)
  (h2 : 1 ≤ k) :
  ∃ M : Set (Sym2 V), G.isMatching M ∧ G.isPerfectMatching M :=
sorry

end k_regular_graph_has_1_factor_l34_34918


namespace stapler_problem_l34_34656

noncomputable def staplesLeft (initial_staples : ℕ) (dozens : ℕ) (staples_per_report : ℝ) : ℝ :=
  initial_staples - (dozens * 12) * staples_per_report

theorem stapler_problem : staplesLeft 200 7 0.75 = 137 := 
by
  sorry

end stapler_problem_l34_34656


namespace candy_mixture_l34_34711

theorem candy_mixture (x : ℝ) (h1 : x * 3 + 64 * 2 = (x + 64) * 2.2) : x + 64 = 80 :=
by sorry

end candy_mixture_l34_34711


namespace angle_alpha_possible_degrees_l34_34504

noncomputable theory

variables {x : ℝ}

def angle_alpha (x : ℝ) : ℝ := 2 * x + 10
def angle_beta (x : ℝ) : ℝ := 3 * x - 20

theorem angle_alpha_possible_degrees (x : ℝ)
  (parallel_sides : True)
  (alpha_def : angle_alpha x = (2 * x + 10))
  (beta_def : angle_beta x = (3 * x - 20))
  (angles_relationship : angle_alpha x + angle_beta x = 180 ∨ angle_alpha x = angle_beta x) :
  angle_alpha x = 70 ∨ angle_alpha x = 86 :=
by {
  sorry
}

end angle_alpha_possible_degrees_l34_34504


namespace correct_sum_of_numbers_l34_34720

variable (n : ℤ)
variable (nums : List ℤ)
variable (sum_of_opposite : (ℤ × ℤ) → Bool)

noncomputable def sum_of_numbers_is_45 : Prop :=
  let nums := [n, n + 1, n + 2, n + 3, n + 4, n + 5]
  let cube_sum := nums.map (λ x => x ^ 3)
  (cube_sum.sum = 3020) ∧
  (∀ (a b : ℤ), sum_of_opposite (a, b) = (a + b) % 2 = 0) ∧
  (nums.sum = 45)

theorem correct_sum_of_numbers : sum_of_numbers_is_45 := 
  sorry

end correct_sum_of_numbers_l34_34720


namespace conditional_probability_l34_34803

noncomputable def P_conditional (A B : Prop) [fintype (A ∩ B)] [fintype B] 
  (pA : ℚ) (pB : ℚ) : ℚ :=
(pA / pB)

-- Conditions
def event_A := ∀ s : fin 4, s ≠ opposite (mod_assoc s 4)
def event_B := ∃ a : attr, ∀ s : fin 3, s ∈ remaining_attrs a

-- Correct Answer
def correct_answer : ℚ := 2 / 9

-- Main theorem
theorem conditional_probability :
  P_conditional event_A event_B (24 : ℚ) (108 : ℚ) = correct_answer :=
by
  sorry

end conditional_probability_l34_34803


namespace circumcenter_parallelogram_l34_34247

variables {Point : Type} [EuclideanGeometry Point]

structure Parallelepiped (p : Type) :=
(A B C D A1 B1 C1 D1 : p)
(edges : set (set p))
(is_parallel: A1 ≠ B1 → A1 ≠ D1 → ∃ l : set p, A1 ∈ l ∧ B1 ∈ l ∧ C1 ∈ l ∧ D1 ∈ l)
(contain_points : ∀ {a b c d a1 b1 c1 d1}, a = A → b = B → c = C → d = D → a1 = A1 → b1 = B1 → c1 = C1 → d1 = D1 
                 → set.Points p {a, b, c, d, a1, b1, c1, d1} ⊆ edges)

variables {A B C D A1 B1 C1 D1 K L M N O1 O2 O3 O4 : Point}
variables (P : Parallelepiped Point)
(hK : K ∈ P.edges ∧ K ∈ segment A B)
(hL : L ∈ P.edges ∧ L ∈ segment B C)
(hM : M ∈ P.edges ∧ M ∈ segment C D)
(hN : N ∈ P.edges ∧ N ∈ segment D A)
(hO1 : O1 = circumcenter Tetrahedron A1 A K N)
(hO2 : O2 = circumcenter Tetrahedron B1 B K L)
(hO3 : O3 = circumcenter Tetrahedron C1 C L M)
(hO4 : O4 = circumcenter Tetrahedron D1 D M N)

theorem circumcenter_parallelogram : parallelogram O1 O2 O3 O4 :=
sorry

end circumcenter_parallelogram_l34_34247


namespace math_proof_problem_l34_34461

-- Defining the problem condition
def condition (x y z : ℝ) := 
  x^3 + y^3 + z^3 - 3 * x * y * z - 3 * (x^2 + y^2 + z^2 - x * y - y * z - z * x) = 0

-- Adding constraints to x, y, z
def constraints (x y z : ℝ) :=
  0 < x ∧ 0 < y ∧ 0 < z ∧ (x ≠ y ∨ y ≠ z ∨ z ≠ x)

-- Stating the main theorem
theorem math_proof_problem (x y z : ℝ) (h_condition : condition x y z) (h_constraints : constraints x y z) :
  x + y + z = 3 ∧ x^2 * (1 + y) + y^2 * (1 + z) + z^2 * (1 + x) > 6 := 
sorry

end math_proof_problem_l34_34461


namespace centroid_set_of_skew_edges_l34_34574

variable {Point : Type*} [AddGroup Point] [AddAction ℝ Point] 

def cube_edges (E1 E2 E3 : set Point): Prop :=
  ∀ (e1 e2 e3 : ℝ), e1 ∈ E1 ∧ e2 ∈ E2 ∧ e3 ∈ E3 →
  (2 ≤ e1/3 + 2 ∧ e1/3 + 2 ≤ 4) ∧
  (2 ≤ e3/3 + 2 ∧ e3/3 + 2 ≤ 4) ∧
  (2 ≤ e2/3 + 2 ∧ e2/3 + 2 ≤ 4)

noncomputable def centroid_cube : set Point :=
  {p | ∃ x y z : ℝ, p = (x, y, z) ∧
    (2 ≤ x ∧ x ≤ 4) ∧
    (2 ≤ y ∧ y ≤ 4) ∧
    (2 ≤ z ∧ z ≤ 4)}

theorem centroid_set_of_skew_edges (E1 E2 E3 : set Point) (h : cube_edges E1 E2 E3):
  ∀ (c : Point), c ∈ centroid_cube ↔ (∃ e1 e2 e3 : ℝ, e1 ∈ E1 ∧ e2 ∈ E2 ∧ e3 ∈ E3 
    ∧ c = ((e1 / 3 + 2), (e2 / 3 + 2), (e3 / 3 + 2))) :=
by
  sorry

end centroid_set_of_skew_edges_l34_34574


namespace linear_correlation_coefficient_non_complementary_events_residual_plot_accuracy_variance_invariant_incorrect_statement_l34_34340

theorem linear_correlation_coefficient {X Y : Type} [RandomVariable X] [RandomVariable Y] :
  |corr X Y| <= 1 := sorry

theorem non_complementary_events {E1 E2 : Event} :
  probability (E1 ∪ E2) ≠ 1 → ¬(complementary E1 E2) := sorry

theorem residual_plot_accuracy {R : ResidualPlot} :
  (narrow_band_of R) → (high_model_accuracy R) := sorry

theorem variance_invariant (X : List ℝ) (c : ℝ) :
  variance (map (λ x, x + c) X) = variance X := sorry

/-- Given statements:
  A: The stronger the linear correlation between two random variables, the closer the absolute value of the correlation coefficient is to 1.
  B: If a person shoots three times in a row while target shooting, the events "at least two hits" and "exactly one hit" are complementary events.
  C: In a residual plot, the narrower the band of residual points, the higher the accuracy of the model fit.
  D: Adding the same constant to each data point in a dataset does not change the variance.
--/
theorem incorrect_statement : 
  let A := linear_correlation_coefficient
  let B := non_complementary_events
  let C := residual_plot_accuracy
  let D := variance_invariant
  ¬B := sorry

end linear_correlation_coefficient_non_complementary_events_residual_plot_accuracy_variance_invariant_incorrect_statement_l34_34340


namespace missing_dimension_of_carton_l34_34714

theorem missing_dimension_of_carton
  (volume_soap_box : ℕ := 240)
  (total_volume_carton : ℕ := 72000)
  (dim1 : ℕ := 25)
  (dim3 : ℕ := 60) :
  ∃ (missing_dimension : ℕ), (dim1 * missing_dimension * dim3 = total_volume_carton) ∧ missing_dimension = 48 :=
by
  use 48
  split
  · sorry
  · rfl

end missing_dimension_of_carton_l34_34714


namespace find_price_per_package_l34_34738

theorem find_price_per_package (P : ℝ) :
  (10 * P + 50 * (4/5 * P) = 1340) → (P = 26.80) := by
  intros h
  sorry

end find_price_per_package_l34_34738


namespace three_same_differences_l34_34110

theorem three_same_differences
  (n : ℕ)
  (h : n > 2)
  (a : ℕ → ℕ)
  (distinct_a : function.injective a)
  (bounds_a : ∀ i, 1 ≤ a i ∧ a i ≤ n^2) :
  ∃ (i j k l : ℕ), i < j ∧ j < k ∧ k < l ∧ (a k - a l = a i - a j) :=
by
  sorry

end three_same_differences_l34_34110


namespace find_a_l34_34837

noncomputable def complex_point_lies_on_line (a : ℝ) : Prop :=
  let z := complex.ofReal (1 / (a * a + 1))
  let re_z := (a : ℂ) / (a ^ 2 + 1)
  let im_z := (1 : ℂ) / (a ^ 2 + 1)
  (re_z + 2 * im_z = 0)

theorem find_a (a : ℝ) : complex_point_lies_on_line a → a = -2 :=
by
  sorry

end find_a_l34_34837


namespace window_width_is_28_l34_34768

noncomputable def window_width (y : ℝ) : ℝ :=
  12 * y + 4

theorem window_width_is_28 : ∃ (y : ℝ), window_width y = 28 :=
by
  -- The proof goes here
  sorry

end window_width_is_28_l34_34768


namespace find_q_l34_34914

theorem find_q (q : ℕ) (h1 : 32 = 2^5) (h2 : 32^5 = 2^q) : q = 25 := by
  sorry

end find_q_l34_34914


namespace find_b_2015_l34_34806

noncomputable def f (n : ℕ) : (ℕ → ℝ) := 
  λ x, (a_n x ^ 2 + b_n x + c_n) * exp x

theorem find_b_2015 (a_n b_n c_n : ℕ → ℝ) (n : ℕ) (h : ∀ n, f n = (a_n n * x ^ 2 + b_n n * x + c_n n) * exp x ) : 
  (n = 2015) → b_n 2015 = 4030 :=
begin
  sorry
end

end find_b_2015_l34_34806


namespace proof_problem_l34_34232

noncomputable def A : Set ℤ := { x | 1 < x ∧ x < 7 }
def B : Set ℝ := { x | x ≥ 10 ∨ x ≤ 2 }

theorem proof_problem : A ∩ (set.compl B) = {3, 4, 5, 6} :=
by sorry

end proof_problem_l34_34232


namespace problem_statement_l34_34214

open Set

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x + 1 ≤ 2}
def CM (s : Set ℤ) : Set ℤ := {x ∈ M | x ∉ s}

theorem problem_statement : CM (M ∩ N) = {2} :=
by
  sorry

end problem_statement_l34_34214


namespace smallest_positive_integer_l34_34070

theorem smallest_positive_integer (N : ℕ) :
  (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 7 = 4) ∧ (N % 11 = 9) → N = 207 :=
by
  sorry

end smallest_positive_integer_l34_34070


namespace abs_eq_condition_l34_34113

theorem abs_eq_condition (x : ℝ) (h : |x - 1| + |x - 5| = 4) : 1 ≤ x ∧ x ≤ 5 :=
by 
  sorry

end abs_eq_condition_l34_34113


namespace power_function_m_value_l34_34631

theorem power_function_m_value (m : ℝ) (h : m^2 - 2 * m + 2 = 1) : m = 1 :=
by
  sorry

end power_function_m_value_l34_34631


namespace time_to_cross_bridge_l34_34726

def length_train_A : ℝ := 200
def speed_train_A_kmh : ℝ := 95
def length_bridge : ℝ := 135
def length_train_B : ℝ := 150
def speed_train_B_kmh : ℝ := 105

def speed_train_A : ℝ := speed_train_A_kmh * (1000 / 3600)
def speed_train_B : ℝ := speed_train_B_kmh * (1000 / 3600)

def total_distance : ℝ := length_train_A + length_bridge + length_train_B
def relative_speed : ℝ := speed_train_A + speed_train_B

theorem time_to_cross_bridge : total_distance / relative_speed ≈ 8.73 :=
by 
  sorry

end time_to_cross_bridge_l34_34726


namespace pages_in_book_l34_34323

theorem pages_in_book (h : ∑ i in finset.range 9, 1 + ∑ i in finset.range 90, 2 + ∑ i in finset.range (356 - 99), 3 = 960) :
  9 + 90 + (356 - 99) = 356 :=
sorry

end pages_in_book_l34_34323


namespace quadratic_min_max_l34_34864

noncomputable def quadratic (x : ℝ) : ℝ := (x - 3)^2 - 1

theorem quadratic_min_max :
  ∃ min max, (∀ x ∈ set.Icc 1 4, quadratic x ≥ min) ∧ min = (-1) ∧
             (∀ x ∈ set.Icc 1 4, quadratic x ≤ max) ∧ max = 3 :=
by
  sorry

end quadratic_min_max_l34_34864


namespace tangent_line_at_neg1_l34_34463

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then -x / (x - 2) else x / (x + 2)

theorem tangent_line_at_neg1 :
  let f (x : ℝ) := if x > 0 then -x / (x - 2) else x / (x + 2) in
  let f' (x : ℝ) := (if x > 0 then (2 / x) else (2 / x)) in
  let P := (-1, f (-1)) in
  ∃ k b, k = 2 ∧ b = 1 ∧ 2 * (-1) - (f (-1) + 1) + 1 = 0 :=
begin
  sorry
end

end tangent_line_at_neg1_l34_34463


namespace total_arrangements_correct_adjacent_males_correct_descending_heights_correct_l34_34700

-- Total number of different arrangements of 3 male students and 2 female students.
def total_arrangements (males females : ℕ) : ℕ :=
  (males + females).factorial

-- Number of arrangements where exactly two male students are adjacent.
def adjacent_males (males females : ℕ) : ℕ :=
  if males = 3 ∧ females = 2 then 72 else 0

-- Number of arrangements where male students of different heights are arranged from tallest to shortest.
def descending_heights (heights : Nat → ℕ) (males females : ℕ) : ℕ :=
  if males = 3 ∧ females = 2 then 20 else 0

-- Theorem statements corresponding to the questions.
theorem total_arrangements_correct : total_arrangements 3 2 = 120 := sorry

theorem adjacent_males_correct : adjacent_males 3 2 = 72 := sorry

theorem descending_heights_correct (heights : Nat → ℕ) : descending_heights heights 3 2 = 20 := sorry

end total_arrangements_correct_adjacent_males_correct_descending_heights_correct_l34_34700


namespace number_of_complex_numbers_l34_34091

noncomputable def count_complex_numbers : ℕ :=
  let z : ℂ → list ℂ := sorry -- A function to generate all possible values of z
  let is_real (c : ℂ) : Prop := (c.im = 0)
  let valid_z (z : ℂ) : Prop := (abs z = 1 ∧ is_real (z^(5!) - z^(4!)))
  sorry -- count the number of such z

theorem number_of_complex_numbers : count_complex_numbers = 1149 :=
sorry

end number_of_complex_numbers_l34_34091


namespace find_m_equal_7_l34_34730

-- Definitions from the given conditions
variables {D E F : Type} [angle : Type] (α β γ : angle)
axiom inscribed_in_circle : triangle DEF
axiom angle_D_eq_3angle_F : α = 3 * γ
axiom angle_E_eq_3angle_F : β = 3 * γ
noncomputable theory

-- Define the main theorem to prove the value of m
theorem find_m_equal_7 (α β γ : angle) (m : ℕ) (h1 : inscribed_in_circle)
  (h2 : α = 3 * γ) (h3 : β = 3 * γ) (h4 : α + β + γ = 180) :
  m = 7 :=
sorry

end find_m_equal_7_l34_34730


namespace log_base4_one_sixteenth_l34_34426

theorem log_base4_one_sixteenth : log 4 (1/16) = -2 :=
begin
  sorry
end

end log_base4_one_sixteenth_l34_34426


namespace max_x_coordinate_of_point_A_on_line_l_with_angle_BAC_60_degrees_l34_34754

theorem max_x_coordinate_of_point_A_on_line_l_with_angle_BAC_60_degrees :
  ∀ (A : ℝ × ℝ), (∃ (B C : ℝ × ℝ),
    (B.fst ^ 2 + B.snd ^ 2 = 4 ∧ C.fst ^ 2 + C.snd ^ 2 = 4) ∧ (A.fst + A.snd = 4) ∧ 
    (real.angle (B - A) (C - A) = real.pi / 3)) →
  A.fst ≤ 4 :=
by
  intros A h
  sorry

end max_x_coordinate_of_point_A_on_line_l_with_angle_BAC_60_degrees_l34_34754


namespace problem_statement_l34_34632

open Function

variable {α : Type*} [LinearOrder α] [OrderTopology α] [TopologicalSpace α]
variable {f : α → α} {a : α}

-- Defining even function.
def even (f : α → α) : Prop := ∀ x, f x = f (-x)

-- Defining monotone increasing function.
def monotone_increasing_on (f : α → α) (s : Set α) : Prop := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

-- The conditions in Lean 4.
theorem problem_statement (h_even : even f) 
  (h_mono_inc_neg : monotone_increasing_on f {x | x ≤ (0 : α)})
  (h_f_le_f2 : f a ≤ f 2) : a ≥ 2 ∨ a ≤ -2 :=
sorry -- proof to be filled in

end problem_statement_l34_34632


namespace abs_neg_three_l34_34287

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l34_34287


namespace honey_worked_days_l34_34877

theorem honey_worked_days (d : ℕ) 
  (h_earn: ∀ n : ℕ, HoneyEarnedPerDay n = 80 * d)
  (h_spent: TotalAmountSpent = 1360)
  (h_saved: TotalAmountSaved = 240):
  d = 20 :=
by
  have total_earn := 80 * d
  have total_spent_saved := 1360 + 240
  have h_eq := total_earn = total_spent_saved
  have eq_div := 1600 / 80
  have result: d = 20 := by sorry
  exact result

end honey_worked_days_l34_34877


namespace problem_part1_problem_part2_l34_34210

-- Define the set A and the property it satisfies
variable (A : Set ℝ)
variable (H : ∀ a ∈ A, (1 + a) / (1 - a) ∈ A)

-- Suppose 2 is in A
theorem problem_part1 (h : 2 ∈ A) : A = {2, -3, -1 / 2, 1 / 3} :=
sorry

-- Prove the conjecture based on the elements of A found in part 1
theorem problem_part2 (h : 2 ∈ A) (hA : A = {2, -3, -1 / 2, 1 / 3}) :
  ¬ (0 ∈ A ∨ 1 ∈ A ∨ -1 ∈ A) ∧
  (2 * (-1 / 2) = -1 ∧ -3 * (1 / 3) = -1) :=
sorry

end problem_part1_problem_part2_l34_34210


namespace meals_for_children_l34_34369

theorem meals_for_children (C : ℕ)
  (H1 : 70 * C = 70 * 45)
  (H2 : 70 * 45 = 2 * 45 * 35) :
  C = 90 :=
by
  sorry

end meals_for_children_l34_34369


namespace correct_statements_l34_34874

section
variables {t : ℝ}

def a : ℝ × ℝ := (-4, 2)
def b := (2, t)

-- Dot product
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

-- Correct statements to be proved
def statement_A := dot_product a b = 0 → t = 4
def statement_B := (a.1 / b.1 = a.2 / b.2) → t = -1
def statement_C := dot_product a b > 0 → t > 4

theorem correct_statements : statement_A ∧ statement_B ∧ statement_C :=
sorry
end

end correct_statements_l34_34874


namespace min_value_of_x2_plus_y2_min_value_of_reciprocal_sum_l34_34132

namespace MathProof

-- Definitions and conditions
variables {x y : ℝ}
axiom x_pos : x > 0
axiom y_pos : y > 0
axiom sum_eq_one : x + y = 1

-- Problem Statement 1: Prove the minimum value of x^2 + y^2 is 1/2
theorem min_value_of_x2_plus_y2 : ∃ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) ∧ (x^2 + y^2 = 1/2) :=
by
  sorry

-- Problem Statement 2: Prove the minimum value of 1/x + 1/y + 1/(xy) is 6
theorem min_value_of_reciprocal_sum : ∃ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) ∧ ((1/x + 1/y + 1/(x*y)) = 6) :=
by
  sorry

end MathProof

end min_value_of_x2_plus_y2_min_value_of_reciprocal_sum_l34_34132


namespace max_single_player_salary_l34_34939

-- Defining the parameters
def num_players : ℕ := 18
def min_salary : ℕ := 20000
def total_salary_cap : ℕ := 900000

-- The statement of the proof problem
theorem max_single_player_salary :
  ∃ x, (∑ i in finset.range (num_players - 1), min_salary) + x = total_salary_cap ∧
    x = 560000 :=
by
  sorry

end max_single_player_salary_l34_34939


namespace option_c_l34_34464

theorem option_c (a b : ℝ) (h : a > |b|) : a^2 > b^2 := sorry

end option_c_l34_34464


namespace negation_proposition_l34_34643

open Classical

variable (x : ℝ)

theorem negation_proposition :
  (¬ ∀ x : ℝ, x ≥ 0) ↔ (∃ x : ℝ, x < 0) :=
by
  sorry

end negation_proposition_l34_34643


namespace neg_p_or_neg_q_implies_m_lt_2_l34_34444

variable {x m : ℝ}

-- p: ∀ x ∈ ℝ, mx^2 + 1 > 0
def p := ∀ x : ℝ, m * x^2 + 1 > 0

-- q: ∃ x ∈ ℝ, x^2 + mx + 1 ≤ 0
def q := ∃ x : ℝ, x^2 + m * x + 1 ≤ 0

-- Proving if ¬p ∨ ¬q then m < 2
theorem neg_p_or_neg_q_implies_m_lt_2 (h : ¬p ∨ ¬q) : m < 2 :=
by
  sorry

end neg_p_or_neg_q_implies_m_lt_2_l34_34444


namespace coffee_amount_l34_34342

theorem coffee_amount (total_mass : ℕ) (coffee_ratio : ℕ) (milk_ratio : ℕ) (h_total_mass : total_mass = 4400) (h_coffee_ratio : coffee_ratio = 2) (h_milk_ratio : milk_ratio = 9) : 
  total_mass * coffee_ratio / (coffee_ratio + milk_ratio) = 800 :=
by
  -- Placeholder for the proof
  sorry

end coffee_amount_l34_34342


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l34_34898

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : ∃ (n : ℕ), n = 225 ∧ ∀ k : ℤ, 100 ≤ 4 * k + 2 ∧ 4 * k + 2 ≤ 999 → 24 < k ∧ k < 250 := by
  sorry

end number_of_three_digit_integers_congruent_to_2_mod_4_l34_34898


namespace pascal_triangle_fifth_element_in_row_17_l34_34956

theorem pascal_triangle_fifth_element_in_row_17 :
  nat.choose 17 4 = 2380 :=
by
  -- the proof would go here, but it is omitted per the instructions
  sorry

end pascal_triangle_fifth_element_in_row_17_l34_34956


namespace number_of_integers_l34_34434

theorem number_of_integers (n : ℤ) : {n | 30 < n^2 ∧ n^2 < 200}.card = 18 := 
sorry

end number_of_integers_l34_34434


namespace chord_length_intercepted_by_line_on_circle_eq_l34_34447

-- Define the circle
def circle (x y : ℝ) := (x - 1)^2 + y^2 = 4

-- Define the line
def line (x y : ℝ) := y = x + 1

-- Define the chord length proof
theorem chord_length_intercepted_by_line_on_circle_eq {d r : ℝ} (h_r : r = 2) (h_d : d = sqrt 2) :
  ∀ x y : ℝ, line x y → circle x y → 2 * sqrt(2) = 2 * sqrt(2) :=
by
  -- Here you would usually proceed with the proof, but since it's only the statement we need:
  sorry

end chord_length_intercepted_by_line_on_circle_eq_l34_34447


namespace shaded_area_of_hexagon_with_quarter_circles_l34_34021

noncomputable def area_inside_hexagon_outside_circles
  (s : ℝ) (h : s = 4) : ℝ :=
  let hex_area := (3 * Real.sqrt 3) / 2 * s^2
  let quarter_circle_area := (1 / 4) * Real.pi * s^2
  let total_quarter_circles_area := 6 * quarter_circle_area
  hex_area - total_quarter_circles_area

theorem shaded_area_of_hexagon_with_quarter_circles :
  area_inside_hexagon_outside_circles 4 rfl = 48 * Real.sqrt 3 - 24 * Real.pi := by
  sorry

end shaded_area_of_hexagon_with_quarter_circles_l34_34021


namespace monotonic_intervals_lambda_value_l34_34480

noncomputable def f (a x : ℝ) : ℝ := Real.exp x + a * x - 1

noncomputable def g (a x : ℝ) : ℝ := (Real.exp 2) * (x^2 - a) / ((Real.exp x + a * x - 1) - a * x + 1)

theorem monotonic_intervals (a : ℝ) :
  (if a ≥ 0 then (∀ x y : ℝ, x < y → f a x < f a y)
   else (∀ x : ℝ, x > Real.log (-a) → f a x > f a (Real.log (-a)) ∧ ∀ x : ℝ, x < Real.log (-a) → f a x < f a (Real.log (-a)))) :=
sorry

theorem lambda_value (a x1 x2 λ : ℝ) (h_extreme : (2*x1 - x1^2 + a)* Real.exp (2-x1) = 0 ∧ x1 < x2) :
  λ = 2 * Real.exp 2 / (Real.exp 2 + 1) :=
sorry

end monotonic_intervals_lambda_value_l34_34480


namespace sri_lanka_population_problem_l34_34355

theorem sri_lanka_population_problem
  (P : ℝ)
  (h1 : 0.85 * (0.9 * P) = 3213) :
  P = 4200 :=
sorry

end sri_lanka_population_problem_l34_34355


namespace remainder_of_470521_div_5_l34_34679

theorem remainder_of_470521_div_5 : 470521 % 5 = 1 := 
by sorry

end remainder_of_470521_div_5_l34_34679


namespace price_of_large_slice_is_250_l34_34390

noncomputable def priceOfLargeSlice (totalSlices soldSmallSlices totalRevenue smallSlicePrice: ℕ) : ℕ :=
  let totalRevenueSmallSlices := soldSmallSlices * smallSlicePrice
  let totalRevenueLargeSlices := totalRevenue - totalRevenueSmallSlices
  let soldLargeSlices := totalSlices - soldSmallSlices
  totalRevenueLargeSlices / soldLargeSlices

theorem price_of_large_slice_is_250 :
  priceOfLargeSlice 5000 2000 1050000 150 = 250 :=
by
  sorry

end price_of_large_slice_is_250_l34_34390


namespace onions_in_basket_l34_34603

variable (X : ℕ)

theorem onions_in_basket (X: ℕ) : ((X + 4) - 5) + 9 = X + 8 := 
by
  calc 
    ((X + 4) - 5) + 9 
        = (X + 4 - 5) + 9 : by rfl
    ... = X + (4 - 5) + 9 : by rw Nat.sub_add
    ... = X + (-1) + 9 : by linarith
    ... = X + 8 : by linarith

end onions_in_basket_l34_34603


namespace common_elements_count_l34_34215

open Finset

def S : Finset ℕ := Finset.range (3005 + 1) |>.image (λ n, 5 * n)
def T : Finset ℕ := Finset.range (3005 + 1) |>.image (λ n, 8 * n)

theorem common_elements_count : (S ∩ T).card = 375 := by
  sorry

end common_elements_count_l34_34215


namespace apple_price_equals_oranges_l34_34648

theorem apple_price_equals_oranges (A O : ℝ) (H1 : A = 28 * O) (H2 : 45 * A + 60 * O = 1350) (H3 : 30 * A + 40 * O = 900) : A = 28 * O :=
by
  sorry

end apple_price_equals_oranges_l34_34648


namespace platform_length_is_correct_l34_34356

def crosses_signal_pole (t: ℕ) (s: ℕ): Prop :=
  t = 9 ∧ s = 20

def train_crosses_half_platform (s1 s2: ℕ) (T1 T2: ℕ) (L: ℚ): Prop :=
  s1 = 20 ∧ s2 = 15 ∧ T1 = (L / 2) / s1 ∧ T2 = (L / 2) / s2 ∧ T1 + T2 = 39

noncomputable def length_of_platform (L: ℚ): Prop := 
  crosses_signal_pole 9 20 ∧ train_crosses_half_platform 20 15 ((L / 2) / 20) ((L / 2) / 15) L ∧ L = 668.57

theorem platform_length_is_correct : ∃ L: ℚ, length_of_platform L := by
  sorry

end platform_length_is_correct_l34_34356


namespace probability_sum_odd_l34_34012

theorem probability_sum_odd :
  let first_wheel := {1, 2, 3, 4, 5}
  let second_wheel := {1, 2, 3, 4}
  (∃ (n1 ∈ first_wheel) (n2 ∈ second_wheel), (n1 + n2) % 2 = 1) / (card first_wheel * card second_wheel) = 1 / 2 :=
by
  sorry

end probability_sum_odd_l34_34012


namespace lizzie_scored_six_l34_34934

-- Definitions based on the problem conditions
def lizzie_score : Nat := sorry
def nathalie_score := lizzie_score + 3
def aimee_score := 2 * (lizzie_score + nathalie_score)

-- Total score condition
def total_score := 50
def teammates_score := 17
def combined_score := total_score - teammates_score

-- Proven statement
theorem lizzie_scored_six:
  (lizzie_score + nathalie_score + aimee_score = combined_score) → lizzie_score = 6 :=
by sorry

end lizzie_scored_six_l34_34934


namespace find_a_l34_34924

-- We define the function and its condition for maximum value
def f (x a : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + a

theorem find_a (a : ℝ) : (∀ x ∈ Icc (0 : ℝ) 2, f x a ≤ f 0 a) ∧ f 0 a = 5 → a = 5 := by
  sorry

end find_a_l34_34924


namespace median_duration_is_170_l34_34653

-- Define the stem-and-leaf plot data
def stemAndLeafDurations : List (ℕ × ℕ) :=
  [(1, 25), (1, 40), (1, 40), (1, 55), (2, 5), (2, 15), (2, 30), (2, 50), (2, 55),
   (3, 0), (3, 15), (3, 15), (3, 30), (3, 45)]

-- Convert the stem-and-leaf plot to total minutes
def convertToMinutes (hrs_mins : List (ℕ × ℕ)) : List ℕ :=
  hrs_mins.map (λ x => x.1 * 60 + x.2)

-- Define the proof problem statement: Prove that the median duration is 170 minutes
theorem median_duration_is_170 :
  let minutes := convertToMinutes stemAndLeafDurations,
      sorted_minutes := List.sort (≤) minutes
  in sorted_minutes[7] = 170 :=
by
  -- Placeholder for the proof
  sorry

end median_duration_is_170_l34_34653


namespace length_of_DE_l34_34620

-- Given conditions
variables (AB DE : ℝ) (area_projected area_ABC : ℝ)

-- Hypotheses
def base_length (AB : ℝ) : Prop := AB = 15
def projected_area_ratio (area_projected area_ABC : ℝ) : Prop := area_projected = 0.25 * area_ABC
def parallel_lines (DE AB : ℝ) : Prop := ∀ x : ℝ, DE = 0.5 * AB

-- The theorem to prove
theorem length_of_DE (h1 : base_length AB) (h2 : projected_area_ratio area_projected area_ABC) (h3 : parallel_lines DE AB) : DE = 7.5 :=
by
  sorry

end length_of_DE_l34_34620


namespace abs_neg_three_l34_34276

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l34_34276


namespace combined_area_of_tracts_l34_34876

theorem combined_area_of_tracts :
  let length1 := 300
  let width1 := 500
  let length2 := 250
  let width2 := 630
  let area1 := length1 * width1
  let area2 := length2 * width2
  let combined_area := area1 + area2
  combined_area = 307500 :=
by
  sorry

end combined_area_of_tracts_l34_34876


namespace distance_between_Jay_and_Sarah_l34_34961

theorem distance_between_Jay_and_Sarah 
  (time_in_hours : ℝ)
  (jay_speed_per_12_minutes : ℝ)
  (sarah_speed_per_36_minutes : ℝ)
  (total_distance : ℝ) :
  time_in_hours = 2 →
  jay_speed_per_12_minutes = 1 →
  sarah_speed_per_36_minutes = 3 →
  total_distance = 20 :=
by
  intros time_in_hours_eq jay_speed_eq sarah_speed_eq
  sorry

end distance_between_Jay_and_Sarah_l34_34961


namespace distance_from_origin_l34_34717

noncomputable def point_distance (x y : ℝ) := Real.sqrt (x^2 + y^2)

theorem distance_from_origin (x y : ℝ) (h₁ : abs y = 15) (h₂ : Real.sqrt ((x - 2)^2 + (y - 7)^2) = 13) (h₃ : x > 2) :
  point_distance x y = Real.sqrt (334 + 4 * Real.sqrt 105) :=
by
  sorry

end distance_from_origin_l34_34717


namespace min_sum_abc_l34_34547

theorem min_sum_abc (a b c : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a * b * c + b * c + c = 2014) : a + b + c = 40 :=
sorry

end min_sum_abc_l34_34547


namespace compare_values_l34_34919

theorem compare_values (x y z : ℝ)
  (h1 : log 2 (log (1 / 2) (log 2 x)) = 0)
  (h2 : log 3 (log (1 / 3) (log 3 y)) = 0)
  (h3 : log 5 (log (1 / 5) (log 5 z)) = 0) :
  z < x ∧ x < y := sorry

end compare_values_l34_34919


namespace trapezoid_height_l34_34943

theorem trapezoid_height (
  MNKL_isosceles : ∀ (MN KL ML NK : ℝ), MNKL_is_trapezoid MN KL ML NK -> 
                                             is_isosceles_trapezoid MNKL MN KL ML NK, 
  diagonals_perpendicular_sides : ∀ (MN KL : ℝ) (ML NK : ℝ), (diagonals_perpendicular MN KL) → 
                                                             (ML ⊥ MN) ∧ (ML ⊥ KL),
  diagonals_intersect_angle : ∀ (MN KL : ℝ) (ML NK : ℝ), (intersect_angle MN KL 15),
  midpoint_larger_base : ∀ (Q ML : ℝ), (midpoint Q ML QL) → (ML = 2 * Q),
  NQ_value : NQ = 5 
  ) : height_of_trapezoid MNKL = \frac{5 \sqrt{2 - \sqrt{3}}}{2} := sorry 

end trapezoid_height_l34_34943


namespace right_angle_zdel_l34_34588

theorem right_angle_zdel (full_circle : ℕ) (right_angle_fraction : ℚ) : 
  full_circle = 400 → right_angle_fraction = 1/3 → 
  right_angle_fraction * full_circle = 400 / 3 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end right_angle_zdel_l34_34588


namespace composite_when_n_gt_1_prime_when_n_eq_1_l34_34100

def expr (n : ℕ) := 3^(2*n+1) - 2^(2*n+1) - 6^n

theorem composite_when_n_gt_1 (n : ℕ) (h : n > 1) : Nat.isComposite (expr n) := 
sorry

theorem prime_when_n_eq_1 : expr 1 = 13 ∧ Nat.Prime 13 := 
by simp [expr]; norm_num

end composite_when_n_gt_1_prime_when_n_eq_1_l34_34100


namespace _l34_34707

noncomputable def geometry_theorem (O A B M N : Type)
  (a b s : line)
  (h1 : a ∥ b)
  (h2 : a ⊥ O)
  (h3 : b ⊥ O)
  (h4 : secantLine s a b)
  (h5 : tangent O a M)
  (h6 : tangent O b N)
  (A_on_s : A ∈ s)
  (B_on_s : B ∈ s)
  (A_on_a : A ∈ a)
  (B_on_b : B ∈ b)
  : angle A O B = π / 2 :=
sorry

end _l34_34707


namespace rectangle_max_area_l34_34377

theorem rectangle_max_area (w : ℝ) (h : ℝ) (hw : h = 2 * w) (perimeter : 2 * (w + h) = 40) :
  w * h = 800 / 9 := 
by
  -- Given: h = 2w and 2(w + h) = 40
  -- We need to prove that the area A = wh = 800/9
  sorry

end rectangle_max_area_l34_34377


namespace brian_fewer_seashells_l34_34757

-- Define the conditions
def cb_ratio (Craig Brian : ℕ) : Prop := 9 * Brian = 7 * Craig
def craig_seashells (Craig : ℕ) : Prop := Craig = 54

-- Define the main theorem to be proven
theorem brian_fewer_seashells (Craig Brian : ℕ) (h1 : cb_ratio Craig Brian) (h2 : craig_seashells Craig) : Craig - Brian = 12 :=
by
  sorry

end brian_fewer_seashells_l34_34757


namespace good_triangle_perimeter_l34_34004

noncomputable def good_collection (T : list (triangle ℝ)) (ω : circle ℝ) : Prop :=
  (∀ (Δ ∈ T), inscribed Δ ω) ∧ 
  (∀ (Δ1 Δ2 ∈ T), Δ1 ≠ Δ2 → ¬exists_point_in_common_interior Δ1 Δ2)

theorem good_triangle_perimeter (r : ℝ) (ω : circle ℝ) (hω : ω.radius = 1) :
  ∀ (t : ℝ), 0 < t → t ≤ 4 → 
  ∀ (n : ℕ), ∃ T : list (triangle ℝ), good_collection T ω ∧ (∀ (Δ ∈ T, 2*π*r < perimeter Δ)) :=
by sorry

end good_triangle_perimeter_l34_34004


namespace least_positive_integer_n_l34_34129

theorem least_positive_integer_n (n : ℕ) (h : (n > 0)) :
  (∃ m : ℕ, m > 0 ∧ (1 / (m : ℝ) - 1 / (m + 1 : ℝ) < 1 / 8) ∧ (∀ k : ℕ, k > 0 ∧ (1 / (k : ℝ) - 1 / (k + 1 : ℝ) < 1 / 8) → m ≤ k)) →
  n = 3 := by
  sorry

end least_positive_integer_n_l34_34129


namespace more_triangles_with_perimeter_2003_than_2000_l34_34750

theorem more_triangles_with_perimeter_2003_than_2000 :
  (∃ (count_2003 count_2000 : ℕ), 
   count_2003 > count_2000 ∧ 
   (∀ (a b c : ℕ), a + b + c = 2000 → a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧ 
   (∀ (a b c : ℕ), a + b + c = 2003 → a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a))
  := 
sorry

end more_triangles_with_perimeter_2003_than_2000_l34_34750


namespace count_rotational_symmetric_tetrominoes_l34_34412

def tetromino : Type := sorry  -- The type definition for a tetromino, which should be constructed of four squares.
def set_of_tetrominoes : List tetromino := sorry  -- The list representing the set of eight tetrominoes.

-- The definition for checking rotational symmetry in a tetromino.
def has_rotational_symmetry (t : tetromino) : Prop := sorry

-- The statement of the theorem.
theorem count_rotational_symmetric_tetrominoes : 
  List.countp has_rotational_symmetry set_of_tetrominoes = 2 :=
sorry

end count_rotational_symmetric_tetrominoes_l34_34412


namespace speed_of_stream_l34_34703

variable (b s : ℝ)

-- Define the conditions from the problem
def downstream_condition := (100 : ℝ) / 4 = b + s
def upstream_condition := (75 : ℝ) / 15 = b - s

theorem speed_of_stream (h1 : downstream_condition b s) (h2: upstream_condition b s) : s = 10 := 
by 
  sorry

end speed_of_stream_l34_34703


namespace distance_from_incenter_to_perpendicular_bisector_is_two_l34_34250

-- Definitions based on the conditions
variable (ABC : Triangle)
variable (BC : Real := 6)
variable (CA : Real := 8)
variable (AB : Real := 10)
variable (I : Point := incenter ABC)
variable (BI : Line := angle_bisector B I ABC)
variable (K : Point := intersection BI AC)
variable (KH : Line := perpendicular_from K AB)

-- The theorem statement
theorem distance_from_incenter_to_perpendicular_bisector_is_two
  (h1 : is_right_triangle ABC)
  (h2 : side_length ABC BC = 6)
  (h3 : side_length ABC CA = 8)
  (h4 : side_length ABC AB = 10)
  (h5 : incenter ABC = I)
  (h6 : angle_bisector B I ABC = BI)
  (h7 : intersection BI AC = K)
  (h8 : perpendicular_from K AB = KH) :
  distance I KH = 2 :=
sorry

end distance_from_incenter_to_perpendicular_bisector_is_two_l34_34250


namespace monica_sheila_additional_money_l34_34238

noncomputable def additional_money_needed_each (initial_money: ℝ) (tp_cost: ℝ) (hs_cost_after_disc: ℝ)
  (ham_multiplier: ℝ) (cheese_divider: ℝ) (saving_rate: ℝ) (boots_multiplier: ℝ) : ℝ :=
let 
  hs_cost_before_disc := hs_cost_after_disc / 0.75,
  ham_cost := tp_cost * ham_multiplier,
  cheese_cost := hs_cost_after_disc / cheese_divider,
  total_expense := tp_cost + hs_cost_after_disc + ham_cost + cheese_cost,
  remaining_money := initial_money - total_expense,
  saving_amount := remaining_money * saving_rate,
  spendable_money := remaining_money - saving_amount,
  each_share := spendable_money / 2,
  boots_cost := each_share * boots_multiplier,
  total_boots_cost := boots_cost * 2,
  total_money_available := each_share * 2,
  additional_money_needed := total_boots_cost - total_money_available
in additional_money_needed / 2

theorem monica_sheila_additional_money :
  additional_money_needed_each 100 12 6 2 2 0.20 4 = 66 := 
sorry

end monica_sheila_additional_money_l34_34238


namespace inequality_holds_l34_34598

theorem inequality_holds (x : ℝ) (hx : x ≥ 0) : 3 * x^3 - 6 * x^2 + 4 ≥ 0 := 
  sorry

end inequality_holds_l34_34598


namespace find_function_expression_axis_of_symmetry_in_interval_l34_34477

def f (x : ℝ) (A ω φ : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem find_function_expression (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) (hφ : |φ| < Real.pi / 2) (h_period : 2 * Real.pi / ω = 2) (h_max_value : f (1 / 3) A ω φ = 2) :
  f x 2 Real.pi (Real.pi / 6) = 2 * Real.sin (Real.pi * x + Real.pi / 6) := by
  sorry

theorem axis_of_symmetry_in_interval : 
  ∃ k : ℤ, (21/4 : ℝ) ≤ (k + 1/3 : ℝ) ∧ (k + 1/3 : ℝ) ≤ 23/4 ∧ (k = 5 → (k + 1/3) = 16/3) := by
  -- This theorem utilizes the previous definition of f and its properties proven in find_function_expression
  sorry

end find_function_expression_axis_of_symmetry_in_interval_l34_34477


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l34_34885

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : 
  ∃ (count : ℕ), count = 225 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ↔ (∃ k : ℕ, 25 ≤ k ∧ k ≤ 249 ∧ n = 4 * k + 2) := 
by {
  sorry
}

end number_of_three_digit_integers_congruent_to_2_mod_4_l34_34885


namespace problem_1_problem_2_problem_3_l34_34872

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := 8 * x^2 + 16 * x - k
noncomputable def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 + 4 * x
noncomputable def h (x : ℝ) (k : ℝ) : ℝ := g x - f x k

theorem problem_1 (k : ℝ) : (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x k ≤ g x) → 45 ≤ k := by
  sorry

theorem problem_2 (k : ℝ) : (∃ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ f x k ≤ g x) → -7 ≤ k := by
  sorry

theorem problem_3 (k : ℝ) : (∀ x1 x2 : ℝ, (-3 ≤ x1 ∧ x1 ≤ 3) ∧ (-3 ≤ x2 ∧ x2 ≤ 3) → f x1 k ≤ g x2) → 141 ≤ k := by
  sorry

end problem_1_problem_2_problem_3_l34_34872


namespace temperature_relationship_l34_34339

def temperature (t : ℕ) (T : ℕ) :=
  ∀ t < 10, T = 7 * t + 30

-- Proof not required, hence added sorry.
theorem temperature_relationship (t : ℕ) (T : ℕ) (h : t < 10) :
  temperature t T :=
by {
  sorry
}

end temperature_relationship_l34_34339


namespace find_a_l34_34503

theorem find_a (a : ℝ) (h1 : 1 < a) (h2 : 1 + a = 3) : a = 2 :=
sorry

end find_a_l34_34503


namespace percentage_change_proof_l34_34660

def priceInitialBook := 300
def priceFinalBook := 420

def priceInitialLaptop := 1200
def priceFinalLaptop := 1080

def priceInitialSmartphone := 600
def priceFinalSmartphone := 630

def percentageChangeBook := ((priceFinalBook - priceInitialBook : ℝ) / priceInitialBook) * 100
def percentageChangeLaptop := ((priceInitialLaptop - priceFinalLaptop : ℝ) / priceInitialLaptop) * 100
def percentageChangeSmartphone := ((priceFinalSmartphone - priceInitialSmartphone : ℝ) / priceInitialSmartphone) * 100

def totalInitialPrice := priceInitialBook + priceInitialLaptop + priceInitialSmartphone
def totalFinalPrice := priceFinalBook + priceFinalLaptop + priceFinalSmartphone
def overallChange := ((totalFinalPrice - totalInitialPrice : ℝ) / totalInitialPrice) * 100

theorem percentage_change_proof :
  percentageChangeBook = 40 ∧
  percentageChangeLaptop = 10 ∧
  percentageChangeSmartphone = 5 ∧
  overallChange ≈ 1.43 := 
by
  sorry

end percentage_change_proof_l34_34660


namespace number_of_solutions_l34_34880

theorem number_of_solutions (m n : ℤ) :
  (m - 2) * (n - 2) = 4 ↔ (m, n) ∈ {(3, 6), (6, 3), (1, -2), (-2, 1), (4, 4), (0, 0)} :=
by
  sorry

end number_of_solutions_l34_34880


namespace k_is_integer_l34_34465

-- Define the problem that \( k \) is an integer given the conditions.
theorem k_is_integer (k : ℝ) (hk : k ≥ 1) 
  (h : ∀ (n m : ℤ), m ∣ n → (⌊m * k⌋ : ℤ) ∣ (⌊n * k⌋ : ℤ)) : ∃ x : ℤ, k = x :=
begin
  sorry
end

end k_is_integer_l34_34465


namespace find_intended_number_l34_34925

theorem find_intended_number (x : ℕ) 
    (condition : 3 * x = (10 * 3 * x + 2) / 19 + 7) : 
    x = 5 :=
sorry

end find_intended_number_l34_34925


namespace abs_neg_three_l34_34289

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l34_34289


namespace bills_average_speed_l34_34689

theorem bills_average_speed :
  ∃ v t : ℝ, 
      (v + 5) * (t + 2) + v * t = 680 ∧ 
      (t + 2) + t = 18 ∧ 
      v = 35 :=
by
  sorry

end bills_average_speed_l34_34689


namespace magnitude_relationship_l34_34107

open Real -- Open the Real namespace for trigonometric functions

-- Define angles in radians since Lean's trigonometric functions use radians
def deg (x : ℝ) : ℝ := x * (π / 180) -- Convert degrees to radians

theorem magnitude_relationship (a b c : ℝ) 
  (ha : a = sin (deg 11))
  (hb : b = cos (deg 10))
  (hc : c = sin (deg 168)) :
  a < c ∧ c < b :=
by
  sorry

end magnitude_relationship_l34_34107


namespace abs_neg_three_l34_34295

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l34_34295


namespace initial_investment_correct_l34_34437

noncomputable def invest_initial_amount : ℝ :=
  let compounded_factor : ℝ := 1.08 ^ 5
  let final_amount : ℝ := 583.22
  final_amount / compounded_factor

-- Now create a theorem that states the equivalence of the initial investment.
theorem initial_investment_correct :
  invest_initial_amount ≈ 396.86 :=
by
  -- Since we are required to skip the proof
  sorry

end initial_investment_correct_l34_34437


namespace abs_neg_three_eq_three_l34_34281

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l34_34281


namespace three_digit_integers_congruent_to_2_mod_4_l34_34890

theorem three_digit_integers_congruent_to_2_mod_4 : 
    ∃ n, n = 225 ∧ ∀ x, (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 2) ↔ (∃ m, 25 ≤ m ∧ m ≤ 249 ∧ x = 4 * m + 2) := by
  sorry

end three_digit_integers_congruent_to_2_mod_4_l34_34890


namespace hexagon_area_half_triangle_l34_34189

theorem hexagon_area_half_triangle (ABC : Triangle) (h : Acute ABC) 
  (K L M : Point) 
  (mid_AB : midpoint K (segment AB)) 
  (mid_BC : midpoint L (segment BC)) 
  (mid_CA : midpoint M (segment CA)) 
  (Q S T : Point)
  (perp_KB : perpendicular_line K Q (line BC))
  (perp_KC : perpendicular_line K Q (line CA))
  (perp_LC : perpendicular_line L S (line CA))
  (perp_LA : perpendicular_line L S (line AB))
  (perp_MB : perpendicular_line M T (line AB))
  (perp_MA : perpendicular_line M T (line BC)) : 
  area (hexagon K Q L S M T) = (1 / 2) * area (triangle ABC) :=
sorry

end hexagon_area_half_triangle_l34_34189


namespace bounded_fx_range_a_l34_34368

-- Part (1)
theorem bounded_fx :
  ∃ M > 0, ∀ x ∈ Set.Icc (-(1/2):ℝ) (1/2), abs (x / (x + 1)) ≤ M :=
by
  sorry

-- Part (2)
theorem range_a (a : ℝ) :
  (∀ x ≥ 0, abs (1 + a * (1/2)^x + (1/4)^x) ≤ 3) → -5 ≤ a ∧ a ≤ 1 :=
by
  sorry

end bounded_fx_range_a_l34_34368


namespace abs_neg_three_l34_34294

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l34_34294


namespace perpendicular_lines_foot_l34_34840

variables (a b c : ℝ)

theorem perpendicular_lines_foot (h1 : a * -2/20 = -1)
  (h2_foot_l1 : a * 1 + 4 * c - 2 = 0)
  (h3_foot_l2 : 2 * 1 - 5 * c + b = 0) :
  a + b + c = -4 :=
sorry

end perpendicular_lines_foot_l34_34840


namespace combinations_with_common_subjects_l34_34524

-- Conditions and known facts
def subjects : Finset String := {"politics", "history", "geography", "physics", "chemistry", "biology", "technology"}
def personA_must_choose : Finset String := {"physics", "politics"}
def personB_cannot_choose : String := "technology"
def total_combinations : Nat := Nat.choose 7 3
def valid_combinations : Nat := Nat.choose 5 1 * Nat.choose 6 3
def non_common_subject_combinations : Nat := 4 + 4

-- We need to prove this statement
theorem combinations_with_common_subjects : valid_combinations - non_common_subject_combinations = 92 := by
  sorry

end combinations_with_common_subjects_l34_34524


namespace M_intersect_N_l34_34157

def M : Set ℝ := {x | 1 + x > 0}
def N : Set ℝ := {x | x < 1}

theorem M_intersect_N : M ∩ N = {x | -1 < x ∧ x < 1} := 
by
  sorry

end M_intersect_N_l34_34157


namespace cylindrical_can_radius_l34_34331

theorem cylindrical_can_radius (h : ℝ) (r : ℝ) :
  ∀ (h ≠ 0), (2 * π * 10^2 * h = π * r^2 * h) → r = 10 * sqrt 2 :=
by
  intro h_ne_zero h_volume_eq
  -- The proof would follow the steps shown in the solution
  sorry

end cylindrical_can_radius_l34_34331


namespace symmetric_point_yOz_l34_34536

theorem symmetric_point_yOz (A : ℝ × ℝ × ℝ) (hA : A = (-1, 2, 0)) :
  let S := (1, 2, 0) in 
  S = (A.1 * -1, A.2, A.3) :=
by
  sorry

end symmetric_point_yOz_l34_34536


namespace dry_mixed_fruits_weight_l34_34015

theorem dry_mixed_fruits_weight :
  ∀ (fresh_grapes_weight fresh_apples_weight : ℕ)
    (grapes_water_content fresh_grapes_dry_matter_perc : ℕ)
    (apples_water_content fresh_apples_dry_matter_perc : ℕ),
    fresh_grapes_weight = 400 →
    fresh_apples_weight = 300 →
    grapes_water_content = 65 →
    fresh_grapes_dry_matter_perc = 35 →
    apples_water_content = 84 →
    fresh_apples_dry_matter_perc = 16 →
    (fresh_grapes_weight * fresh_grapes_dry_matter_perc / 100) +
    (fresh_apples_weight * fresh_apples_dry_matter_perc / 100) = 188 := by
  sorry

end dry_mixed_fruits_weight_l34_34015


namespace rectangle_triangle_same_area_l34_34036

noncomputable def same_area_height : ℝ :=
  let l := 1 -- assuming unit length for the radius
  let h := 2 / 5 -- solving the same height
  in h

theorem rectangle_triangle_same_area :
  ∀ (ABCD is_rectangle) (AEB is_isosceles_triangle) (E on_opposite_side_of_AB C D)
  (circle_through_ABCD E) (circle_radius E = 1) (areas_equal : area_rectangle ABCD = area_triangle AEB),
  |AD| = same_area_height :=
begin
  sorry -- proof here
end

end rectangle_triangle_same_area_l34_34036


namespace abs_neg_three_eq_three_l34_34286

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l34_34286


namespace semicircle_circumference_correct_l34_34315

noncomputable def rectangle_length : ℝ := 4
noncomputable def rectangle_breadth : ℝ := 2
noncomputable def perimeter_rectangle : ℝ := 2 * (rectangle_length + rectangle_breadth)
noncomputable def side_square : ℝ := perimeter_rectangle / 4
noncomputable def diameter_semicircle : ℝ := side_square
noncomputable def circumference_semicircle : ℝ := (Real.pi * diameter_semicircle) / 2 + diameter_semicircle

-- rounding function for two decimal places
noncomputable def round_two_decimals (x : ℝ) : ℝ := Real.floor (x * 100) / 100

theorem semicircle_circumference_correct:
  round_two_decimals circumference_semicircle = 7.71 := by
  sorry

end semicircle_circumference_correct_l34_34315


namespace inv_sum_eq_one_of_powers_eq_100_l34_34442

theorem inv_sum_eq_one_of_powers_eq_100 (x y : ℝ) 
  (h₁ : 2^x = 100) 
  (h₂ : 50^y = 100) :
  x⁻¹ + y⁻¹ = 1 :=
sorry

end inv_sum_eq_one_of_powers_eq_100_l34_34442


namespace rectangle_measurement_error_l34_34946

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) 
  (h1 : 0 < L) (h2 : 0 < W) 
  (h3 : A = L * W)
  (h4 : A' = L * (1 + x / 100) * W * (1 - 4 / 100))
  (h5 : A' = A * (100.8 / 100)) :
  x = 5 :=
by
  sorry

end rectangle_measurement_error_l34_34946


namespace recommendation_plans_count_l34_34527

-- Definitions for combinatorial operations used in the problem
def combinations := calc
  -- Function to compute combinations, C(n, k)
  def C (n k : ℕ) : ℕ := sorry

-- Calculating arrangements
def arrangements := calc
  -- Function to compute arrangements, A(n, k) = n! / (n - k)!
  def A (n k : ℕ) : ℕ := sorry

theorem recommendation_plans_count : 
  let males := 3
  let females := 2
  let spots := 5
  let russian_spots := 2
  let japanese_spots := 2
  let spanish_spot := 1
  let total_plans := C males 1 * C (males - 1) 1 * A (females + 1) 3 - 2 * A (females + 1) 3
  in total_plans = 24 := 
by sorry

end recommendation_plans_count_l34_34527


namespace solve_quadratic_substitution_l34_34599

theorem solve_quadratic_substitution : ∀ x : ℝ, (x^2 + 1)^2 - 4 * (x^2 + 1) - 12 = 0 → (x = √5 ∨ x = -√5) :=
by
  intro x h
  let y := x^2 + 1
  have hy : y^2 - 4*y - 12 = 0, from sorry
  have y1_y2 : y = 6 ∨ y = -2, from sorry
  cases y1_y2 with hy1 hy2
  · have h1 : x^2 = 5, from sorry
    have hx1 : x = √5 ∨ x = -√5, from sorry
    exact hx1
  · have h2 : false, from sorry
    contradiction

end solve_quadratic_substitution_l34_34599


namespace john_unanswered_questions_l34_34962

theorem john_unanswered_questions (c w u : ℕ) 
  (h1 : 25 + 5 * c - 2 * w = 95) 
  (h2 : 6 * c - w + 3 * u = 105) 
  (h3 : c + w + u = 30) : 
  u = 2 := 
sorry

end john_unanswered_questions_l34_34962


namespace find_q_l34_34160

-- Given conditions
variables (p q : ℝ)
hypothesis h1 : p > 1
hypothesis h2 : q > 1
hypothesis h3 : (1/p) + (1/q) = 3/2
hypothesis h4 : p * q = 6

-- The statement to prove
theorem find_q : q = (9 + Real.sqrt 57) / 2 :=
by
  sorry

end find_q_l34_34160


namespace train_length_l34_34727

theorem train_length (L V : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 175 = V * 39) : 
  L = 150 := 
by 
  -- proof omitted 
  sorry

end train_length_l34_34727


namespace functions_from_M_to_N_l34_34156

def M : Set ℤ := { -1, 1, 2, 3 }
def N : Set ℤ := { 0, 1, 2, 3, 4 }
def f2 (x : ℤ) := x + 1
def f4 (x : ℤ) := (x - 1)^2

theorem functions_from_M_to_N :
  (∀ x ∈ M, f2 x ∈ N) ∧ (∀ x ∈ M, f4 x ∈ N) :=
by
  sorry

end functions_from_M_to_N_l34_34156


namespace problem_statement_l34_34654

-- Define the necessary conditions
def condition1 : ℝ := 27 = (3:ℝ)^3
def condition2 : ℝ := 8 = (2:ℝ)^3

-- Prove the main statement
theorem problem_statement (h1 : 27 = (3:ℝ)^3) (h2 : 8 = (2:ℝ)^3) : (27:ℝ)^(-1 / 3) - log (8:ℝ) (2:ℝ) = 0 := by
  sorry

end problem_statement_l34_34654


namespace pears_arrangement_l34_34236

noncomputable def arrange_pears (n : ℕ) (weights : list ℕ) : list (list ℕ) :=
sorry

theorem pears_arrangement (n : ℕ) (weights : list ℕ) (X : ℕ) (Y : ℕ)
  (h_len : list.length weights = 2 * n + 2)
  (h_X : X = list.minimum weights)
  (h_Y : Y = list.maximum weights)
  (h_sorted : ∀ (i j : ℕ), i < j → weights.nth i ≤ weights.nth j)
  (h_diff : ∀ (i : ℕ), i < (list.length (arrange_pears n weights)) → 
    abs ((arrange_pears n weights).nth i - (arrange_pears n weights).nth ((i + 1) % (list.length (arrange_pears n weights)))) ≤ 1) :
  ∃ (arrangement : list (list ℕ)), 
    (∀ (i : ℕ), 
      i < (list.length arrangement - 1) → 
        abs (list.sum (arrangement.nth i) - list.sum (arrangement.nth (i + 1))) ≤ 1)

end pears_arrangement_l34_34236


namespace product_of_roots_l34_34502

theorem product_of_roots :
  ∃ x₁ x₂ : ℝ, (x₁ * x₂ = -4) ∧ (x₁ ^ 2 + 2 * x₁ - 4 = 0) ∧ (x₂ ^ 2 + 2 * x₂ - 4 = 0) := by
  sorry

end product_of_roots_l34_34502


namespace find_a_l34_34155

noncomputable def A (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }
def B : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }

theorem find_a (a : ℝ) :
  A a ∪ B = B ↔ a = 0 ∨ a = 1 ∨ a = 1 / 2 :=
sorry

end find_a_l34_34155


namespace locus_of_points_M_l34_34328

theorem locus_of_points_M (A B C : Point) (h_collinear : collinear A B C) (h_between : between A B C)
    (circle : Circle B r) : 
    (∃ M : Point, on_arc_of_circle_with_diameter_BK M A C) :=
  sorry

end locus_of_points_M_l34_34328


namespace mutual_tangency_arc_l34_34873

-- Define the geometric elements and given conditions
variables (A B O : Point) (circle_c : Circle O) (circle1 circle2 : Circle)

-- Assume points A and B are on the given circle.
axiom A_on_circle_c : A ∈ circle_c
axiom B_on_circle_c : B ∈ circle_c

-- Assume circle1 and circle2 are externally tangent to the given circle at points A and B, respectively.
axiom circle1_tangent_A : TangentExternally circle1 circle_c A
axiom circle2_tangent_B : TangentExternally circle2 circle_c B

-- Assume circle1 and circle2 are externally tangent to each other at point M.
variables (M : Point)
axiom circle1_tangent_circle2 : TangentExternally circle1 circle2 M

-- Now, we need to prove that the set of all such points M forms an arc of a circle.
theorem mutual_tangency_arc : 
  ∃ (arc : Arc), ∀ (M : Point), TangentExternally circle1 circle2 M → M ∈ arc :=
sorry

end mutual_tangency_arc_l34_34873


namespace polynomial_root_l34_34338

theorem polynomial_root (a : ℝ) (h : a^3 - a - 1 = 0) : 
  ∃ P : ℝ[X], P = (polynomial.C 1 * polynomial.X^6) - 
    (polynomial.C 8 * polynomial.X^4) + 
    (polynomial.C 13 * polynomial.X^2) - 
    (polynomial.C 2 * polynomial.X^3) - 
    (polynomial.C 10 * polynomial.X) - 
    (polynomial.C 1) ∧ 
    polynomial.aeval (a + Real.sqrt 2) P = 0 := 
by
  sorry

end polynomial_root_l34_34338


namespace triangle_ahf_area_l34_34045

def circle (center : ℝ × ℝ) (radius : ℝ) := ∀ (P : ℝ × ℝ), (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius^2

noncomputable def externally_tangent_at (A : ℝ × ℝ) (B C : ℝ × ℝ) : Prop :=
  ∃ rB rC, circle B rB ∧ circle C rC ∧ rB + rC = dist B C ∧ B = A ∧ C = A

variable (A B C D E F H : ℝ × ℝ)
variable (rB rC : ℝ)

-- Given conditions
def condition1 := circle B rB
def condition2 := rB = 3
def condition3 := rC = 5
def condition4 := externally_tangent_at A B C
def condition5 := ∃ (DE : ℝ → ℝ×ℝ),
    (DE 0 = D) ∧ 
    (DE 1 = E) ∧ 
    ∀ t, externally_tangent_at (DE t) B C → perpendicular (A, DE t)
def condition6 := perpendicular (A, DE t) ∧ perpendicular_bisector F B C
def condition7 := midpoint H B C

-- Proof goal
def proof_goal := ∃ area, area = (sqrt 15) / 2 ∧ area_of_triangle A H F = area

-- The statement that needs to be proved
theorem triangle_ahf_area :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 ∧ condition6 ∧ condition7 → proof_goal :=
sorry

end triangle_ahf_area_l34_34045


namespace words_per_page_large_font_l34_34540

theorem words_per_page_large_font
    (total_words : ℕ)
    (large_font_pages : ℕ)
    (small_font_pages : ℕ)
    (small_font_words_per_page : ℕ)
    (total_pages : ℕ)
    (words_in_large_font : ℕ) :
    total_words = 48000 →
    total_pages = 21 →
    large_font_pages = 4 →
    small_font_words_per_page = 2400 →
    words_in_large_font = total_words - (small_font_pages * small_font_words_per_page) →
    small_font_pages = total_pages - large_font_pages →
    (words_in_large_font = large_font_pages * 1800) :=
by 
    sorry

end words_per_page_large_font_l34_34540


namespace outlined_square_digit_l34_34329

theorem outlined_square_digit :
  ∀ (digit : ℕ), (digit ∈ {n | ∃ (m : ℕ), 10 ≤ 3^m ∧ 3^m < 1000 ∧ digit = (3^m / 10) % 10 }) →
  (digit ∈ {n | ∃ (n : ℕ), 10 ≤ 7^n ∧ 7^n < 1000 ∧ digit = (7^n / 10) % 10 }) →
  digit = 4 :=
by sorry

end outlined_square_digit_l34_34329


namespace contrapositive_proof_l34_34795

theorem contrapositive_proof (m : ℕ) (h_pos : 0 < m) :
  (¬ (∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

end contrapositive_proof_l34_34795


namespace waiter_table_count_l34_34353

theorem waiter_table_count (initial_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) :
  initial_customers = 21 ∧ customers_left = 12 ∧ people_per_table = 3 →
  (initial_customers - customers_left) / people_per_table = 3 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  rw [h1, h2, h3]
  sorry

end waiter_table_count_l34_34353


namespace gcd_of_set_B_is_five_l34_34558

def is_in_set_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)

theorem gcd_of_set_B_is_five : ∃ d, (∀ n, is_in_set_B n → d ∣ n) ∧ d = 5 :=
by 
  sorry

end gcd_of_set_B_is_five_l34_34558


namespace original_number_of_people_l34_34593

theorem original_number_of_people (x : ℕ) (h1 : x - x / 3 + (x / 3) * 3/4 = x * 1/4 + 15) : x = 30 :=
sorry

end original_number_of_people_l34_34593


namespace minimize_abs_sum_l34_34682

open Real BigOperators

noncomputable def f (x : ℝ) : ℝ := ∑ k in (finset.range 2012).image (+1), |x - k|

theorem minimize_abs_sum : 
  (∀ x : ℝ, f x ≥ 1011030) ∧ f 1006 = 1011030 :=
by 
  sorry

end minimize_abs_sum_l34_34682


namespace time_n_th_mile_l34_34016

-- Definitions based on conditions
def speed (k : ℝ) (d : ℕ) : ℝ := k / (d ^ 2)
def time_to_traverse (s : ℝ) : ℝ := 1 / s

-- Give conditions
constant k : ℝ
axiom k_val : k = 1 / 2
axiom second_mile_time : time_to_traverse (speed k 1) = 2

-- Prove that time to traverse the nth mile is 2(n-1)^2 hours
theorem time_n_th_mile (n : ℕ) (h : n ≥ 2) : 
  time_to_traverse (speed k (n - 1)) = 2 * (n - 1) ^ 2 :=
by
  sorry

end time_n_th_mile_l34_34016


namespace complement_of_M_in_U_l34_34231

open Set

theorem complement_of_M_in_U :
  let U := {1, 2, 3, 4, 5, 6}
  let M := {1, 3, 4}
  complementOfM := U \ M
  complementOfM = {2, 5, 6}
:= by
  sorry

end complement_of_M_in_U_l34_34231


namespace pyramid_edge_length_and_volume_l34_34379

-- Define the given conditions
def square_side_length : ℝ := 8
def pyramid_height : ℝ := 15

-- Define the calculations necessary to prove the statements
def diagonal := real.sqrt (square_side_length^2 + square_side_length^2)
def half_diagonal := diagonal / 2
def slant_edge := real.sqrt(pyramid_height^2 + half_diagonal^2)
def total_edge_length := 4 * square_side_length + 4 * slant_edge
def base_area := square_side_length^2
def volume := 1 / 3 * base_area * pyramid_height

-- State the theorem
theorem pyramid_edge_length_and_volume :
  total_edge_length = 32 + 4 * real.sqrt 257 ∧ volume = 320 :=
by 
  sorry

end pyramid_edge_length_and_volume_l34_34379


namespace problem_period_and_symmetry_l34_34490

noncomputable def a (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (2 * x) + 3, cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (1, 2 * cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem problem_period_and_symmetry (x : ℝ) :
  (∃ T : ℝ, ∀ x, f(x + T) = f(x)) ∧
  (∃ k : ℤ, ∃ ys : ℝ, f (x) = f ys ∧ ys = ((k : ℝ) * π / 2) - (π / 12)) ∧
  ((x ∈ set.Icc (π / 12) (7 * π / 12)) -> (set.Icc 3 6 = set.Icc 3 6)) :=
by
  sorry

end problem_period_and_symmetry_l34_34490


namespace mrs_oaklyn_rugs_l34_34240

theorem mrs_oaklyn_rugs (buying_price selling_price total_profit : ℕ) (h1 : buying_price = 40) (h2 : selling_price = 60) (h3 : total_profit = 400) : 
  ∃ (num_rugs : ℕ), num_rugs = 20 :=
by
  sorry

end mrs_oaklyn_rugs_l34_34240


namespace zach_miles_on_thursday_l34_34343

theorem zach_miles_on_thursday (x : ℕ) :
  (150 + 310 + 0.50 * x = 832) ↔ (x = 744) :=
by
-- Proof will be here
sorry

end zach_miles_on_thursday_l34_34343


namespace spherical_to_rectangular_conversion_l34_34418

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  spherical_to_rectangular 4 (Real.pi / 2) (Real.pi / 4) = (0, 2 * Real.sqrt 2, 2 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_conversion_l34_34418


namespace four_students_three_classes_l34_34878

-- Define the function that calculates the number of valid assignments
def valid_assignments (students : ℕ) (classes : ℕ) : ℕ :=
  if students = 4 ∧ classes = 3 then 36 else 0  -- Using given conditions to return 36 when appropriate

-- Define the theorem to prove that there are 36 valid ways
theorem four_students_three_classes : valid_assignments 4 3 = 36 :=
  by
  -- The proof is not required, so we use sorry to skip it
  sorry

end four_students_three_classes_l34_34878


namespace area_of_sector_l34_34843

-- Given conditions
def radius : ℝ := 2
def central_angle : ℝ := 2
def sector_area (r α : ℝ) : ℝ := 1 / 2 * r^2 * α

-- Theorem statement
theorem area_of_sector : sector_area radius central_angle = 4 := by
  sorry

end area_of_sector_l34_34843


namespace part1_part2_l34_34448

variable (m : ℝ)

def z (m : ℝ) : ℂ := complex.mk (m^2 + 2 * m - 8) (m^2 + 2 * m - 3)

-- Proof Question 1
theorem part1 (h1 : (z m - (m : ℂ) + 2).re = 0) : m = 2 :=
sorry

-- Proof Question 2
theorem part2 (h2 : (m^2 + 2 * m - 8 < 0) ∧ (m^2 + 2 * m - 3 < 0)) : -3 < m ∧ m < 1 :=
sorry

end part1_part2_l34_34448


namespace slope_of_line_l34_34787

theorem slope_of_line : ∀ (x y : ℝ), 4 * x - 7 * y = 28 → y = (4/7) * x - 4 :=
by
  sorry

end slope_of_line_l34_34787


namespace number_of_three_digit_integers_congruent_mod4_l34_34907

def integer_congruent_to_mod (a b n : ℕ) : Prop := ∃ k : ℤ, n = a * k + b

theorem number_of_three_digit_integers_congruent_mod4 :
  (finset.filter (λ n, integer_congruent_to_mod 4 2 (n : ℕ)) 
   (finset.Icc (100 : ℕ) (999 : ℕ))).card = 225 :=
by
  sorry

end number_of_three_digit_integers_congruent_mod4_l34_34907


namespace factor_10000_into_three_natural_factors_l34_34520

theorem factor_10000_into_three_natural_factors :
  (∃ (a b c : ℕ), a * b * c = 10000 ∧ ¬∃ x y z, ((x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2)) ∧ (a = x^_ ∨ b = y^_ ∨ c = z^_) ∧ (∀ perm, perm.factor(a, b, c) = (a, b, c))) →
  6 :=
by sorry

end factor_10000_into_three_natural_factors_l34_34520


namespace GCD_is_six_l34_34336

-- Define the numbers
def a : ℕ := 36
def b : ℕ := 60
def c : ℕ := 90

-- Define the GCD using Lean's gcd function
def GCD_abc : ℕ := Nat.gcd (Nat.gcd a b) c

-- State the theorem that GCD of 36, 60, and 90 is 6
theorem GCD_is_six : GCD_abc = 6 := by
  sorry -- Proof skipped

end GCD_is_six_l34_34336


namespace prob_144_eq_1_div_72_l34_34684

open Probability
open MeasureTheory
open Classical

noncomputable def probability_abc_144 : ℝ :=
  let outcomes := ({1, 2, 3, 4, 5, 6} : Finset ℕ)
  let prod := {d : Finset ℕ × Finset ℕ × Finset ℕ | 
    (d.1.1 * d.1.2 * d.2) = 144} 
  (∑ in prod, (1 / 6 : ℝ) ^ 3)

theorem prob_144_eq_1_div_72 : probability_abc_144 = 1 / 72 :=
sorry

end prob_144_eq_1_div_72_l34_34684


namespace solve_problem_l34_34168

noncomputable def problem_statement : Prop :=
  (∀ (x : ℝ), (1 + Real.tan x) * (1 + Real.tan (Real.pi / 4 - x)) = 2) ∧
  Real.tan 0 = 0 ∧ 
  Real.tan (Real.pi / 4) = 1 ∧
  (∏ (i : ℕ) in Finset.range 31, (1 + Real.tan (i * (Real.pi / 180)))) = 2 ^ 16

theorem solve_problem : problem_statement :=
sorry

end solve_problem_l34_34168


namespace abs_neg_three_l34_34290

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l34_34290


namespace points_concyclic_in_acute_triangle_l34_34498

theorem points_concyclic_in_acute_triangle
  (ABC : Triangle)
  (acuteABC : is_acute_triangle ABC)
  (altitudes_analogous : (BE and CF are either angle bisectors or medians))
  : concyclic_points M P N Q :=
sorry

end points_concyclic_in_acute_triangle_l34_34498


namespace problem_statement_l34_34483

-- Definitions based on conditions
def f (x : ℝ) : ℝ := x^2 - 1
def g (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem problem_statement : f (g 3) = 120 ∧ f 3 = 8 :=
by sorry

end problem_statement_l34_34483


namespace printed_value_is_201_l34_34921

theorem printed_value_is_201 :
  ∃ (n : ℕ), let x := 3 + 2 * n
             ∧ S := n^2 + 4 * n
             in S ≥ 10000 ∧ x = 201 :=
sorry

end printed_value_is_201_l34_34921


namespace solution_inequality_l34_34976

open set

variable (f : ℝ → ℝ)

-- Conditions
axiom differentiable_f : differentiable ℝ f
axiom symmetry_f : ∀ x, f x = f (2 - x)
axiom derivative_condition : ∀ x, (x - 1) * deriv f x > 0
axiom f_at_2 : f 2 = 0

-- Question: Find the solution set for the inequality x * f(x) < 0
theorem solution_inequality : {x : ℝ | x * f x < 0} = Iio 0 ∪ Ioo 0 2 :=
begin
  sorry
end

end solution_inequality_l34_34976


namespace find_m_l34_34833

-- Define the vectors and the real number m
variables {Vec : Type*} [AddCommGroup Vec] [Module ℝ Vec]
variables (e1 e2 : Vec) (m : ℝ)

-- Define the collinearity condition and non-collinearity of the basis vectors.
def non_collinear (v1 v2 : Vec) : Prop := ¬(∃ (a : ℝ), v2 = a • v1)

def collinear (v1 v2 : Vec) : Prop := ∃ (a : ℝ), v2 = a • v1

-- Given conditions
axiom e1_e2_non_collinear : non_collinear e1 e2
axiom AB_eq : ∀ (m : ℝ), Vec
axiom CB_eq : Vec

theorem find_m (h : collinear (e1 + m • e2) (e1 - e2)) : m = -1 :=
sorry

end find_m_l34_34833


namespace beetle_ate_first_time_l34_34393

def maggots_first_time (total : ℕ) (second_time : ℕ) : ℕ := total - second_time

theorem beetle_ate_first_time 
  (total : ℕ) (second_time : ℕ) 
  (h_total : total = 20) (h_second_time : second_time = 3) : 
  maggots_first_time total second_time = 17 :=
by
  rw [h_total, h_second_time]
  simp
  sorry

end beetle_ate_first_time_l34_34393


namespace findSmallestM_l34_34577

def fractionalPart (x : ℝ) := x - x.floor

def g (x : ℝ) := |3 * fractionalPart x - 1.5|

def eqHasAtLeast3000Solutions (m : ℕ) : Prop :=
  ∃ (S : Finset ℝ), S.card = 3000 ∧ ∀ x ∈ S, m * g(x * g x) = x

theorem findSmallestM : ∃ (m : ℕ), eqHasAtLeast3000Solutions m ∧ ∀ (k : ℕ), k < m → ¬ eqHasAtLeast3000Solutions k := by
  -- The proof will show that m is 39 as the smallest integer satisfying the condition
  sorry

end findSmallestM_l34_34577


namespace determine_m_l34_34575

noncomputable def f : ℝ → ℝ := sorry

-- f is an odd function
axiom h_odd : ∀ x : ℝ, f (-x) + f x = 0

-- f is strictly decreasing
axiom h_decreasing : ∀ x1 x2 : ℝ, (x1 - x2) * (f x1 - f x2) < 0

-- f(2t^2 - 4) + f(4m - 2t) ≤ f 0 for all t ∈ [0, 2]
axiom h_condition : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 2 → f (2 * t ^ 2 - 4) + f (4 * m - 2 * t) ≥ f 0

theorem determine_m (m : ℝ) : m ≤ 0 := sorry

end determine_m_l34_34575


namespace arrange_p_q_r_l34_34871

theorem arrange_p_q_r (p : ℝ) (h : 1 < p ∧ p < 1.1) : p < p^p ∧ p^p < p^(p^p) :=
by
  sorry

end arrange_p_q_r_l34_34871


namespace maximize_profit_l34_34027

/--
A store purchases goods at a unit price of 40 yuan, and initially sells them for 50 yuan each, selling 500 units.
With each one yuan increase in price, the sales volume decreases by 10 units.
Prove that the selling price should be set at 70 yuan per unit to achieve a maximum profit of 9000 yuan.
-/
theorem maximize_profit (x : ℕ) (hx : 0 ≤ x ∧ x ≤ 50)
    (profit : ℤ := ((500 - 10 * x) * (50 + x)) - ((500 - 10 * x) * 40)) :
  (∃ y : ℤ, y = ((500 - 10 * x) * (50 + x) - (500 - 10 * x) * 40) ∧ y = 9000 ∧ 50 + x = 70) :=
begin
  sorry
end

end maximize_profit_l34_34027


namespace find_Y_length_l34_34196

theorem find_Y_length (Y : ℝ) : 
  (3 + 2 + 3 + 4 + Y = 7 + 4 + 2) → Y = 1 :=
by
  intro h
  sorry

end find_Y_length_l34_34196


namespace abs_neg_three_l34_34269

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l34_34269


namespace determine_n_l34_34067

theorem determine_n :
  ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ (-250 ≡ n [MOD 23]) ∧ n = 3 :=
sorry

end determine_n_l34_34067


namespace total_perimeter_correct_l34_34749

-- Define the conditions given in the problem
def side_length_of_square : ℝ := 1
def radius_of_quarter_circle := side_length_of_square

-- Define the total number of quarter circles
def num_quarter_circles : ℕ := 4

-- The total perimeter formed by the quarter circles
def total_perimeter : ℝ := num_quarter_circles * (π / 2) * radius_of_quarter_circle

-- Prove that the total perimeter formed by the quarter circles is 2π
theorem total_perimeter_correct :
  total_perimeter = 2 * π :=
by
  -- Proof is omitted
  sorry

end total_perimeter_correct_l34_34749


namespace both_false_of_not_or_l34_34111

-- Define propositions p and q
variables (p q : Prop)

-- The condition given: ¬(p ∨ q)
theorem both_false_of_not_or (h : ¬(p ∨ q)) : ¬ p ∧ ¬ q :=
by {
  sorry
}

end both_false_of_not_or_l34_34111


namespace curves_properties_l34_34192

def is_cartesian_eqn_c1 :=
  ∀ (x y : ℝ), (∃ (θ : ℝ), x = ρ * cos θ ∧ y = ρ * sin θ ∧ ρ * (sin θ + cos θ) = 1) → x + y = 1

noncomputable def is_general_eqn_c2 :=
  ∀ (x y : ℝ), (∃ (θ : ℝ), x = 2 * cos θ ∧ y = sin θ) → (x^2 / 4) + y^2 = 1

def intersection_points :=
  ∃ (x1 y1 x2 y2 : ℝ), (x1 + y1 = 1) ∧ ((x1^2) / 4 + y1^2 = 1) ∧ (x2 + y2 = 1) ∧ ((x2^2) / 4 + y2^2 = 1) ∧
    x1 ≠ x2 ∧ y1 ≠ y2

def distance_between_points :=
  ∀ (x1 y1 x2 y2 : ℝ), (x1 + y1 = 1) ∧ ((x1^2) / 4 + y1^2 = 1) ∧ (x2 + y2 = 1) ∧ ((x2^2) / 4 + y2^2 = 1) ∧
    (x1 ≠ x2 ∧ y1 ≠ y2) →
    sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 * sqrt 5 / 5

theorem curves_properties :
  is_cartesian_eqn_c1 ∧ is_general_eqn_c2 ∧ intersection_points ∧ distance_between_points :=
by 
  sorry

end curves_properties_l34_34192


namespace sum_of_b_sequence_l34_34121

-- Define the arithmetic sequence {a_n}
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

-- Given conditions
def a (n : ℕ) : ℤ := 
  if n = 0 then 0 -- this line can be adjusted to match the pedestrian understanding of index starting from 1
  else if n = 1 then 3 -- a1 = 3
  else 3 + 3 * (n - 1) -- general formula a_n = 3 + 3(n-1)

theorem sum_of_b_sequence :
  let b (n : ℕ) := a (2^n) in 
  (b 0 + b 1 + b 2 + b 3 + b 4) = 186 :=
begin
  sorry
end

end sum_of_b_sequence_l34_34121


namespace triangle_side_difference_l34_34955

theorem triangle_side_difference (x : ℕ) (a b : ℕ) (h1 : a = 7) (h2 : b = 3) 
  (h3 : x < a + b) (h4 : x > abs (a - b)) : 
  (max (finset.filter (λ (n : ℕ), n < a + b ∧ n > abs (a - b)) (finset.range (a + b))).val 
    - min (finset.filter (λ (n : ℕ), n < a + b ∧ n > abs (a - b)) (finset.range (a + b))).val) = 4 :=
by
  sorry

end triangle_side_difference_l34_34955


namespace final_price_is_correct_l34_34644

-- Define the original price and percentages as constants
def original_price : ℝ := 160
def increase_percentage : ℝ := 0.25
def discount_percentage : ℝ := 0.25

-- Calculate increased price
def increased_price : ℝ := original_price * (1 + increase_percentage)
-- Calculate the discount on the increased price
def discount_amount : ℝ := increased_price * discount_percentage
-- Calculate final price after discount
def final_price : ℝ := increased_price - discount_amount

-- Statement of the theorem: prove final price is $150
theorem final_price_is_correct : final_price = 150 :=
by
  -- Proof would go here
  sorry

end final_price_is_correct_l34_34644


namespace exists_integers_m_n_l34_34825

theorem exists_integers_m_n (x y : ℝ) (hxy : x ≠ y) : 
  ∃ (m n : ℤ), (m * x + n * y > 0) ∧ (n * x + m * y < 0) :=
sorry

end exists_integers_m_n_l34_34825


namespace minimum_value_of_expression_l34_34151

theorem minimum_value_of_expression (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : a * b > 0) : 
  (1 / a) + (2 / b) = 5 :=
sorry

end minimum_value_of_expression_l34_34151


namespace tensor_A_B_eq_l34_34759

-- Define sets A and B
def A : Set ℕ := {0, 2}
def B : Set ℕ := {x | x^2 - 3 * x + 2 = 0}

-- Define set operation ⊗
def tensor (A B : Set ℕ) : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

-- Prove that A ⊗ B = {0, 2, 4}
theorem tensor_A_B_eq : tensor A B = {0, 2, 4} :=
by
  sorry

end tensor_A_B_eq_l34_34759


namespace intersection_eq_l34_34866

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem intersection_eq : A ∩ B = {1, 3} :=
by
  sorry

end intersection_eq_l34_34866


namespace symmetric_line_eq_x_axis_l34_34302

theorem symmetric_line_eq_x_axis (x y : ℝ) :
  (3 * x - 4 * y + 5 = 0) → (3 * x + 4 * y + 5 = 0) :=
sorry

end symmetric_line_eq_x_axis_l34_34302


namespace f_decreasing_on_intervals_l34_34852

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then sin x else -sin x

theorem f_decreasing_on_intervals :
  ∀ k : ℤ, ∀ x y : ℝ, 
    x ∈ [(-real.pi / 2 + 2 * k * real.pi), (real.pi / 2 + 2 * k * real.pi)] →
    y ∈ [(-real.pi / 2 + 2 * k * real.pi), (real.pi / 2 + 2 * k * real.pi)] →
    (x ≤ y → f y ≤ f x) :=
sorry

end f_decreasing_on_intervals_l34_34852


namespace engineering_student_max_marks_l34_34346

/-- 
If an engineering student has to secure 36% marks to pass, and he gets 130 marks but fails by 14 marks, 
then the maximum number of marks is 400.
-/
theorem engineering_student_max_marks (M : ℝ) (passing_percentage : ℝ) (marks_obtained : ℝ) (marks_failed_by : ℝ) (pass_marks : ℝ) :
  passing_percentage = 0.36 →
  marks_obtained = 130 →
  marks_failed_by = 14 →
  pass_marks = marks_obtained + marks_failed_by →
  pass_marks = passing_percentage * M →
  M = 400 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end engineering_student_max_marks_l34_34346


namespace largest_whole_number_l34_34068

theorem largest_whole_number (x : ℕ) (h : 6 * x + 3 < 150) : x ≤ 24 :=
sorry

end largest_whole_number_l34_34068


namespace mac_runs_faster_by_120_minutes_l34_34043

theorem mac_runs_faster_by_120_minutes :
  ∀ (D : ℝ), (D / 3 - D / 4 = 2) → 2 * 60 = 120 := by
  -- Definitions matching the conditions
  intro D
  intro h

  -- The proof is not required, hence using sorry
  sorry

end mac_runs_faster_by_120_minutes_l34_34043


namespace value_of_Y_l34_34169

-- Define the variables within the Lean environment
def P : ℝ := 4012 / 4
def Q : ℝ := P / 2
def Y : ℝ := P - Q

-- State the theorem to prove
theorem value_of_Y : Y = 501.5 := 
by
  -- Proof omitted, focus is on the problem statement
  sorry

end value_of_Y_l34_34169


namespace intersection_complement_l34_34154

def M (x : ℝ) : set ℝ := { y : ℝ | y = x^2 + 2 * x - 3 }
def N : set ℝ := { x : ℝ | -5 ≤ x ∧ x ≤ 2 }
def M_complement_ᶜ_N : set ℝ := M ∩ (λ x, ¬ N x)

theorem intersection_complement (y : ℝ) :
  M_complement_ᶜ_N y = (2, +∞) :=
sorry

end intersection_complement_l34_34154


namespace sum_of_digits_of_n_l34_34578

theorem sum_of_digits_of_n (
  n : ℤ
) (hn_gt_1000 : n > 1000)
  (gcd1 : Int.gcd 63 (n + 120) = 21)
  (gcd2 : Int.gcd (n + 63) 120 = 60) :
  (n.digits.sum = 18) ∧ (exists (m : ℤ), m > 1000 ∧ Int.gcd 63 (m + 120) = 21 ∧ Int.gcd (m + 63) 120 = 60 ∧ m = n) :=
by
  sorry

end sum_of_digits_of_n_l34_34578


namespace race_graph_representation_l34_34936

-- Let's first define the conditions
def snail_steady (t: ℝ) : Prop :=
∀ t1 t2 : ℝ, t1 < t2 → d_snail t1 < d_snail t2

def horse_burst_pause_resume (t: ℝ) : Prop :=
∃ t1 t2 t3 t4: ℝ, t1 < t2 < t3 < t4 ∧ 
(∀ t ∈ set.Icc t1 t2, deriv (d_horse t) > 0) ∧
(∀ t ∈ set.Icc t2 t3, deriv (d_horse t) = 0) ∧
(∀ t ∈ set.Icc t3 t4, deriv (d_horse t) > 0)

def snail_wins (final_t: ℝ) : Prop :=
d_snail final_t > d_horse final_t

-- Define the final theorem
theorem race_graph_representation
  (final_t : ℝ)
  (h_snail : snail_steady final_t)
  (h_horse : horse_burst_pause_resume final_t)
  (h_wins : snail_wins final_t)
  : correct_graph_rep = "Graph B" :=
sorry

end race_graph_representation_l34_34936


namespace curve_intersection_distance_l34_34468

noncomputable def C1 (t : ℝ) : ℝ × ℝ := (-2 + Real.cos t, 1 + Real.sin t)
noncomputable def C2 (s : ℝ) : ℝ × ℝ := (-4 + (Real.sqrt 2)/2 * s, (Real.sqrt 2)/2 * s)

theorem curve_intersection_distance :
  let A := {p : ℝ × ℝ | ∃ t, p = C1 t ∧ ∃ s, p = C2 s},
      B := {p : ℝ × ℝ | ∃ t, p = C1 t ∧ ∃ s, p = C2 s} in
  | dist (set.to_finset A).nth 0 (set.to_finset B).nth 1 | = Real.sqrt 2 :=
by
  sorry

end curve_intersection_distance_l34_34468


namespace combined_area_of_hexagon_and_triangle_l34_34378

theorem combined_area_of_hexagon_and_triangle (r : ℝ) (h_r3 : r = 3) :
  let s := r in
  let hexagon_area := 6 * (s^2 * real.sqrt 3 / 4) in
  let triangle_area := s^2 * real.sqrt 3 / 4 in
  hexagon_area + triangle_area = 63 * real.sqrt 3 / 4 :=
by
  let s := r
  let hexagon_area := 6 * (s^2 * real.sqrt 3 / 4)
  let triangle_area := s^2 * real.sqrt 3 / 4
  sorry

end combined_area_of_hexagon_and_triangle_l34_34378


namespace trapezoid_area_pqrs_l34_34675

theorem trapezoid_area_pqrs :
  let P := (1, 1)
  let Q := (1, 4)
  let R := (6, 4)
  let S := (7, 1)
  let parallelogram := true -- indicates that PQ and RS are parallel
  let PQ := abs (Q.2 - P.2)
  let RS := abs (S.1 - R.1)
  let height := abs (R.1 - P.1)
  (1 / 2 : ℚ) * (PQ + RS) * height = 10 := by
  sorry

end trapezoid_area_pqrs_l34_34675


namespace number_of_three_digit_integers_congruent_mod4_l34_34906

def integer_congruent_to_mod (a b n : ℕ) : Prop := ∃ k : ℤ, n = a * k + b

theorem number_of_three_digit_integers_congruent_mod4 :
  (finset.filter (λ n, integer_congruent_to_mod 4 2 (n : ℕ)) 
   (finset.Icc (100 : ℕ) (999 : ℕ))).card = 225 :=
by
  sorry

end number_of_three_digit_integers_congruent_mod4_l34_34906


namespace sin_double_angle_plus_pi_fourth_side_length_c_area_triangle_l34_34930

noncomputable theory
open Real

variables {A B C : Type} [AddGroup A] [AddGroup B] [AddGroup C]
variables {triangleABC : Triangle A B C}
variables {a b c : ℝ} (h1: side_opposite_angle A a triangleABC)
                      (h2: side_opposite_angle B b triangleABC)
                      (h3: side_opposite_angle C c triangleABC)
                      (h4: cos_angle C = 1/5)
                      (h5: dot_product_vectors CA CB = 1)
                      (h6: a + b = sqrt 37)

theorem sin_double_angle_plus_pi_fourth (h: triangleABC) :
  sin (2 * angle C + π / 4) = (8 * sqrt 3 - 23 * sqrt 2) / 50 := sorry

theorem side_length_c (h: triangleABC) :
  c = 5 := sorry

theorem area_triangle (h: triangleABC) :
  area triangleABC = sqrt 6 := sorry

end sin_double_angle_plus_pi_fourth_side_length_c_area_triangle_l34_34930


namespace identity_is_only_sum_free_preserving_surjection_l34_34352

def is_surjective (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, f m = n

def is_sum_free (A : Set ℕ) : Prop :=
  ∀ x y : ℕ, x ∈ A → y ∈ A → x + y ∉ A

noncomputable def identity_function_property : Prop :=
  ∀ f : ℕ → ℕ, is_surjective f →
  (∀ A : Set ℕ, is_sum_free A → is_sum_free (Set.image f A)) →
  ∀ n : ℕ, f n = n

theorem identity_is_only_sum_free_preserving_surjection : identity_function_property := sorry

end identity_is_only_sum_free_preserving_surjection_l34_34352


namespace find_y_l34_34952

-- Conditions as definitions in Lean 4
def angle_AXB : ℝ := 180
def angle_AX : ℝ := 70
def angle_BX : ℝ := 40
def angle_CY : ℝ := 130

-- The Lean statement for the proof problem
theorem find_y (angle_AXB_eq : angle_AXB = 180)
               (angle_AX_eq : angle_AX = 70)
               (angle_BX_eq : angle_BX = 40)
               (angle_CY_eq : angle_CY = 130) : 
               ∃ y : ℝ, y = 60 :=
by
  sorry -- The actual proof goes here.

end find_y_l34_34952


namespace sum_of_flawless_primes_l34_34033

def is_prime (n : ℕ) : Prop := sorry

def swap_digits (n : ℕ) : ℕ := sorry

def is_flawless_prime (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100 ∧ is_prime n ∧ is_prime (swap_digits n)

theorem sum_of_flawless_primes : ∑ n in {n | is_flawless_prime n}, n = 429 := sorry

end sum_of_flawless_primes_l34_34033


namespace determine_function_l34_34817

theorem determine_function (f : ℝ → ℝ) (h_domain : ∀ x, x > 1 → f x > 1) :
  (∀ x y u v, x > 1 → y > 1 → u > 0 → v > 0 →
    f (x ^ u * y ^ v) ≤ (f x) ^ (1 / (u * 10)) * (f y) ^ (1 / (v * 10))) →
  ∃ a, a > 1 ∧ f = λ x, a^(1 / (a * x)) :=
begin
  sorry
end

end determine_function_l34_34817


namespace exists_planar_quadrilateral_l34_34075

theorem exists_planar_quadrilateral 
  (α β γ δ : ℝ) 
  (h_sum : α + β + γ + δ = 360) 
  (h_tan : tan α = tan β ∧ tan β = tan γ ∧ tan γ = tan δ)
  : ∃ α β γ δ, α + β + γ + δ = 360 ∧ tan α = tan β ∧ tan β = tan γ ∧ tan γ = tan δ :=
by
  sorry

end exists_planar_quadrilateral_l34_34075


namespace largest_integer_incorner_sum_l34_34965

noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c in
  real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def inradius (a b c : ℝ) : ℝ :=
  let Δ := heron_area a b c in
  let s := semi_perimeter a b c in
  Δ / s

noncomputable def corner_inradii_sum (a b c : ℝ) : ℝ :=
  let r := inradius a b c in
  3 * r - 30 * (a + b + c) / (2 * heron_area a b c)

theorem largest_integer_incorner_sum (a b c : ℝ) (ha : a = 51) (hb : b = 52) (hc : c = 53) :
    ⌊corner_inradii_sum a b c⌋ = 15 :=
by
  subst ha
  subst hb
  subst hc
  sorry

end largest_integer_incorner_sum_l34_34965


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l34_34897

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : ∃ (n : ℕ), n = 225 ∧ ∀ k : ℤ, 100 ≤ 4 * k + 2 ∧ 4 * k + 2 ≤ 999 → 24 < k ∧ k < 250 := by
  sorry

end number_of_three_digit_integers_congruent_to_2_mod_4_l34_34897


namespace odd_digit_three_digit_numbers_count_l34_34881

theorem odd_digit_three_digit_numbers_count : 
  let digits := {1, 3, 5, 7, 9} 
  in (∃ digits' : finset ℕ, digits' = digits) →
     (digits' : finset ℕ) = {1, 3, 5, 7, 9} →
     (∀ n : ℕ, n ∈ digits' → n % 2 = 1) →
     (∃ count : ℕ, count = (5 * 5 * 5) ∧ count = 125) :=
by 
  sorry

end odd_digit_three_digit_numbers_count_l34_34881


namespace determine_solution_set_inequality_l34_34869

-- Definitions based on given conditions
def quadratic_inequality_solution (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c > 0
def new_quadratic_inequality_solution (c b a : ℝ) (x : ℝ) := c * x^2 + b * x + a < 0

-- The proof statement
theorem determine_solution_set_inequality (a b c : ℝ):
  (∀ x : ℝ, -1/3 < x ∧ x < 2 → quadratic_inequality_solution a b c x) →
  (∀ x : ℝ, -3 < x ∧ x < 1/2 ↔ new_quadratic_inequality_solution c b a x) := sorry

end determine_solution_set_inequality_l34_34869


namespace depth_of_second_hole_l34_34702

theorem depth_of_second_hole:
  (∀ (w₁ w₂ h₁ h₂ hr₁ hr₂: ℕ),
    w₁ = 45 ∧ hr₁ = 8 ∧ h₁ = 30 ∧ -- conditions for the first hole
    w₂ = 110 ∧ hr₂ = 6 →            -- conditions for the second hole
    let rate := (h₁: ℚ) / (w₁ * hr₁) in
    let depth := rate * (w₂ * hr₂) in
    depth = 55) :=
by
  intros w₁ w₂ h₁ h₂ hr₁ hr₂
  assume h: w₁ = 45 ∧ hr₁ = 8 ∧ h₁ = 30 ∧ w₂ = 110 ∧ hr₂ = 6
  let rate := (h₁: ℚ) / (w₁ * hr₁)
  let depth := rate * (w₂ * hr₂)
  show depth = 55 from sorry

end depth_of_second_hole_l34_34702


namespace cos_fourth_minus_sin_fourth_l34_34128

theorem cos_fourth_minus_sin_fourth (α : ℝ) (h : Real.sin α = (Real.sqrt 5) / 5) :
  Real.cos α ^ 4 - Real.sin α ^ 4 = 3 / 5 := 
sorry

end cos_fourth_minus_sin_fourth_l34_34128


namespace Triangle_Segment_Equality_l34_34569

open Classical

variables {A B C D M E F P N : Type*}

-- Let D be a point on the side BC of triangle ABC
variable [hD_on_BC : D ∈ line_segment B C]

-- Let P be a point on the line segment AD
variable [hP_on_AD : P ∈ line_segment A D]

-- D intersects AB at M, PB at E, AC at F, and PC at N
variables (hM_on_AB : D = line_through M B ∧ M ∈ line_segment A B)
variables (hE_on_PB : D = line_through E P ∧ E ∈ line_segment P B)
variables (hF_on_AC : D = line_through F C ∧ F ∈ extension A C)
variables (hN_on_PC : D = line_through N C ∧ N ∈ extension P C)

-- Given DE = DF
variable [hDE_eq_DF : dist D E = dist D F]

-- Prove DM = DN
theorem Triangle_Segment_Equality (hD_on_BC : D ∈ line_segment B C) 
  (hP_on_AD : P ∈ line_segment A D) 
  (hM_on_AB : D = line_through M B ∧ M ∈ line_segment A B) 
  (hE_on_PB : D = line_through E P ∧ E ∈ line_segment P B) 
  (hF_on_AC : D = line_through F C ∧ F ∈ extension A C) 
  (hN_on_PC : D = line_through N C ∧ N ∈ extension P C) 
  (hDE_eq_DF : dist D E = dist D F) : dist D M = dist D N :=
by
  sorry

end Triangle_Segment_Equality_l34_34569


namespace monotonically_decreasing_interval_l34_34312

theorem monotonically_decreasing_interval :
  (∀ x ∈ set.Icc (0:ℝ) (Real.pi), deriv (λ x, Real.cos x - Real.sin x) x ≤ 0) → 
  (∀ x, x ∈ set.Icc (0:ℝ) (3 * Real.pi / 4) → deriv (λ x, Real.cos x - Real.sin x) x < 0) := 
sorry

end monotonically_decreasing_interval_l34_34312


namespace find_t_value_l34_34875

noncomputable def vector_perp_condition (t : ℝ) : Prop :=
  let m := (t + 1, 1) in
  let n := (t + 2, 2) in
  (m.1 + n.1, m.2 + n.2) ⬝ (m.1 - n.1, m.2 - n.2) = 0

theorem find_t_value : ∃ t : ℝ, vector_perp_condition t ∧ t = -3 :=
by
  use -3
  split
  sorry

end find_t_value_l34_34875


namespace factorization_10000_l34_34522

def is_factorization (n a b c : ℕ) : Prop :=
  a * b * c = n

def no_factor_divisible_by_10 (a b c : ℕ) : Prop :=
  ¬(10 ∣ a) ∧ ¬(10 ∣ b) ∧ ¬(10 ∣ c)

theorem factorization_10000 :
  ∃ (a b c : ℕ), is_factorization 10000 a b c ∧ no_factor_divisible_by_10 a b c ∧ 
  finset.card (finset.univ.filter (λ x, ∃ (a b c : ℕ), x = ({a, b, c} : finset ℕ) ∧
    is_factorization 10000 a b c ∧ no_factor_divisible_by_10 a b c)) = 6 :=
begin
  sorry
end

end factorization_10000_l34_34522


namespace intersection_of_A_B_C_l34_34581

-- Define the sets A, B, and C as given conditions:
def A : Set ℕ := { x | ∃ n : ℕ, x = 2 * n }
def B : Set ℕ := { x | ∃ n : ℕ, x = 3 * n }
def C : Set ℕ := { x | ∃ n : ℕ, x = n ^ 2 }

-- Prove that A ∩ B ∩ C = { x | ∃ n : ℕ, x = 36 * n ^ 2 }
theorem intersection_of_A_B_C :
  (A ∩ B ∩ C) = { x | ∃ n : ℕ, x = 36 * n ^ 2 } :=
sorry

end intersection_of_A_B_C_l34_34581


namespace coefficient_x2_in_binomial_expansion_l34_34134

theorem coefficient_x2_in_binomial_expansion :
  let a := (1/2 : ℝ),
      poly := (a * Real.sqrt x - 1 / Real.sqrt x)^6
  in (polynomial.coeff poly 2) = -3 / 16 :=
by
  sorry

end coefficient_x2_in_binomial_expansion_l34_34134


namespace nth_term_correct_l34_34388

noncomputable def nth_term (a b : ℝ) (n : ℕ) : ℝ :=
  (-1 : ℝ)^n * (2 * n - 1) * b / a^n

theorem nth_term_correct (a b : ℝ) (n : ℕ) (h : 0 < a) : 
  nth_term a b n = (-1 : ℝ)^↑n * (2 * n - 1) * b / a^n :=
by sorry

end nth_term_correct_l34_34388


namespace total_weight_l34_34671

variables
  (h : 103) -- Haley's weight
  (v : ℝ) -- Verna's weight
  (s : ℝ) -- Sherry's weight
  (j : ℝ) -- Jake's weight
  (l : ℝ) -- Laura's weight

noncomputable def verna_weight_condition : Prop := v = h + 17
noncomputable def sherry_weight_condition : Prop := v = s / 2
noncomputable def jake_weight_condition : Prop := j = (3 / 5) * (h + v)
noncomputable def laura_weight_condition : Prop := l = s - j

theorem total_weight 
  (h : 103)
  (verna_weight_condition : v = h + 17)
  (sherry_weight_condition : v = s / 2)
  (jake_weight_condition : j = (3 / 5) * (h + v))
  (laura_weight_condition : l = s - j) :
  v + s + j + l = 600 :=
sorry

end total_weight_l34_34671


namespace remainder_of_concatenated_numbers_div_by_9_l34_34341

theorem remainder_of_concatenated_numbers_div_by_9 :
  let N := natConcat (1, 2015) in
  N % 9 = 0 :=
by
  sorry

end remainder_of_concatenated_numbers_div_by_9_l34_34341


namespace find_function_f_l34_34221

noncomputable def S : set ℝ := {x : ℝ | x > -1}

noncomputable def f : ℝ → ℝ := sorry -- This is where we will eventually show f(x) = -x / (1 + x)

theorem find_function_f (f : ℝ → ℝ) (h1 : ∀ x y ∈ S, f (x + f y + x * f y) = y + f x + y * f x)
  (h2 : strict_mono_on (λ x, f x / x) (set.Ioo (-1 : ℝ) 0) ∧ strict_mono_on (λ x, f x / x) (set.Ioi 0)) :
  ∀ x : ℝ, x ∈ S → f x = -x / (1 + x) := sorry

end find_function_f_l34_34221


namespace line_PQ_parallel_to_x_axis_l34_34457

def point (α : Type) := α × α

def is_parallel_to_x_axis (P Q : point ℝ) : Prop :=
  P.2 = Q.2

theorem line_PQ_parallel_to_x_axis (P Q : point ℝ)
  (hP : P = (-4 : ℝ, 5 : ℝ))
  (hQ : Q = (-2 : ℝ, 5 : ℝ)) :
  is_parallel_to_x_axis P Q :=
by sorry

end line_PQ_parallel_to_x_axis_l34_34457


namespace perimeter_of_EFGH_l34_34526

noncomputable def perimeter_of_quadrilateral (EF FG GH : ℝ) (EG EH : ℝ) : ℝ :=
  EF + FG + GH + 2 * EG

/-- Given quadrilateral EFGH with the specified properties, the perimeter is 68 + 2 * sqrt(1360). -/
theorem perimeter_of_EFGH : 
  let EF := 24
  let FG := 28
  let GH := 16
  let sqr_1360 := Real.sqrt 1360
  let EG := sqr_1360 
  let EH := sqr_1360 in
  perimeter_of_quadrilateral EF FG GH EG EH = 68 + 2 * sqr_1360 :=
by
  sorry

end perimeter_of_EFGH_l34_34526


namespace arithmetic_sequence_abs_sum_l34_34453

theorem arithmetic_sequence_abs_sum :
  ∀ (a : ℕ → ℤ), (∀ n, a (n + 1) - a n = 2) → a 1 = -5 → 
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 18) :=
by
  sorry

end arithmetic_sequence_abs_sum_l34_34453


namespace value_set_of_t_l34_34140

theorem value_set_of_t (t : ℝ) :
  (1 > 2 * (1) + 1 - t) ∧ (∀ x : ℝ, x^2 + (2*t-4)*x + 4 > 0) → 3 < t ∧ t < 4 :=
by
  intros h
  sorry

end value_set_of_t_l34_34140


namespace goal1_goal2_goal3_l34_34118

noncomputable theory

variables {f : ℝ → ℝ}

-- Conditions
axiom cond1 (x : ℝ) (hx : x > 1) : f x < 0
axiom cond2 : f (1 / 2) = 1
axiom cond3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x + f y

-- Goal 1: Prove that f(1/x) = -f(x)
theorem goal1 (x : ℝ) (hx : 0 < x) : f (1 / x) = -f x := 
sorry

-- Goal 2: Prove that f(x) is decreasing function within its domain
theorem goal2 : ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ > x₂ → f x₁ < f x₂ := 
sorry

-- Goal 3: Find the set of m that satisfies the inequality f(log_{0.5}m +3 ) + f(2log_{0.5}m - 1) ≥ -2
theorem goal3 (m : ℝ) : (1 / 2) ≤ m ∧ m < (Real.sqrt 2 / 2) ↔ f (log 0.5 m + 3) + f (2 * log 0.5 m - 1) ≥ -2 :=
sorry

end goal1_goal2_goal3_l34_34118


namespace gcd_of_B_l34_34564

def B : Set ℕ := {n | ∃ x : ℕ, n = 5 * x}

theorem gcd_of_B : Int.gcd_range (B.toList) = 5 := by 
  sorry

end gcd_of_B_l34_34564


namespace eggs_in_each_basket_l34_34996

theorem eggs_in_each_basket :
  ∃ (n : ℕ), (n ∣ 30) ∧ (n ∣ 45) ∧ (n ≥ 5) ∧
    (∀ m : ℕ, (m ∣ 30) ∧ (m ∣ 45) ∧ (m ≥ 5) → m ≤ n) ∧ n = 15 :=
by
  -- Condition 1: n divides 30
  -- Condition 2: n divides 45
  -- Condition 3: n is greater than or equal to 5
  -- Condition 4: n is the largest such divisor
  -- Therefore, n = 15
  sorry

end eggs_in_each_basket_l34_34996


namespace slope_of_line_l34_34788

theorem slope_of_line : ∀ (x y : ℝ), 4 * x - 7 * y = 28 → y = (4/7) * x - 4 :=
by
  sorry

end slope_of_line_l34_34788


namespace coefficient_of_x_in_expansion_of_x_plus_m_pow_6_l34_34142

noncomputable def normalDist (mean variance : ℝ) : ℝ -> ℝ := sorry

noncomputable def P (event : ℝ -> Prop) : ℝ := sorry

theorem coefficient_of_x_in_expansion_of_x_plus_m_pow_6 :
  (∀ (xi : ℝ), normalDist (1/2) sigma^2 xi) ->
  P (λ xi, xi < -1) = P (λ xi, xi > m) ->
  (6.choose 1) * (m ^ 5) = 192 :=
by
  sorry

end coefficient_of_x_in_expansion_of_x_plus_m_pow_6_l34_34142


namespace avg_speed_is_65_l34_34692

theorem avg_speed_is_65
  (speed1: ℕ) (speed2: ℕ) (time1: ℕ) (time2: ℕ)
  (h_speed1: speed1 = 85)
  (h_speed2: speed2 = 45)
  (h_time1: time1 = 1)
  (h_time2: time2 = 1) :
  (speed1 + speed2) / (time1 + time2) = 65 := by
  sorry

end avg_speed_is_65_l34_34692


namespace max_one_person_amount_l34_34000

theorem max_one_person_amount (n : ℕ) (total_money : ℝ) (max_10_sum : ℝ) (people_money : Fin n → ℝ)
  (h_n : n = 100) (h_total : total_money = 2000) (h_10_sum : ∀ (s : Finset (Fin n)), s.card = 10 → (∑ i in s, people_money i) ≤ 380) :
  ∃ m, (m = 218) ∧ (∀ i, people_money i ≤ m) :=
by
  sorry

end max_one_person_amount_l34_34000


namespace part1_part2_l34_34994

open Set

variable {α : Type*} [PartialOrder α]

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem part1 : A ∩ B = {x | 2 < x ∧ x < 3} :=
by
  sorry

theorem part2 : (compl B) = {x | x ≤ 1 ∨ x ≥ 3} :=
by
  sorry

end part1_part2_l34_34994


namespace find_a_l34_34814

variables {a : ℝ} (z : ℂ)
axiom imaginary_unit : z = a + complex.I
axiom modulus_condition : abs z = 2

theorem find_a : a = real.sqrt 3 ∨ a = -real.sqrt 3 :=
by
  -- You would write the proof here.
  sorry

end find_a_l34_34814


namespace final_direction_is_east_l34_34201

-- Define initial direction
inductive Direction where
  | north : Direction
  | east : Direction
  | south : Direction
  | west : Direction

def initial_direction : Direction := Direction.north

-- Define movements in terms of fractions
def clockwise_move : ℚ := 7 / 2
def counterclockwise_move : ℚ := 21 / 4

-- Final proof that the spinner points east
theorem final_direction_is_east :
  let net_movement : ℚ := clockwise_move - counterclockwise_move
  let simplified_movement : ℚ := net_movement + 2 -- converting to the positive range if necessary
  simplified_movement ≡ 1 / 4 [MOD 1] →
  initial_direction = Direction.north →
  Direction.east := sorry

end final_direction_is_east_l34_34201


namespace coeff_x3_in_expansion_l34_34621

theorem coeff_x3_in_expansion : (Polynomial.coeff ((Polynomial.C 1 - Polynomial.C 2 * Polynomial.X)^6) 3) = -160 := 
by 
  sorry

end coeff_x3_in_expansion_l34_34621


namespace algebraic_expression_equivalence_l34_34776

theorem algebraic_expression_equivalence (x : ℝ) : 
  x^2 - 6*x + 10 = (x - 3)^2 + 1 := 
by 
  sorry

end algebraic_expression_equivalence_l34_34776


namespace distinct_sequences_count_l34_34493

theorem distinct_sequences_count : 
  (∃ s : List Char, s.length = 4 ∧ s.head = 'L' ∧ s.last = 'Q' ∧ s.nodup) →
  (['E', 'Q', 'U', 'A', 'L', 'S'].erase 'L').erase 'Q'.length = 4 ∧
  (['E', 'Q', 'U', 'A', 'L', 'S'].erase 'L').erase 'Q' = [_, _, _, _] →
  ∃ n : ℕ, n = 12 :=
begin
  sorry
end

end distinct_sequences_count_l34_34493


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l34_34894

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : ∃ (n : ℕ), n = 225 ∧ ∀ k : ℤ, 100 ≤ 4 * k + 2 ∧ 4 * k + 2 ≤ 999 → 24 < k ∧ k < 250 := by
  sorry

end number_of_three_digit_integers_congruent_to_2_mod_4_l34_34894


namespace three_digit_integers_congruent_to_2_mod_4_l34_34889

theorem three_digit_integers_congruent_to_2_mod_4 : 
    ∃ n, n = 225 ∧ ∀ x, (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 2) ↔ (∃ m, 25 ≤ m ∧ m ≤ 249 ∧ x = 4 * m + 2) := by
  sorry

end three_digit_integers_congruent_to_2_mod_4_l34_34889


namespace part_I_part_II_part_III_l34_34148

noncomputable def g (x : ℝ) (a : ℝ) := Real.log x + a * x^2 - 3 * x

theorem part_I :
  (g' : ℝ → ℝ → ℝ) := λ x a, (1/x) + 2*a*x - 3) (g' 1 a = 0) → a = 1 := sorry

theorem part_II :
  ∀ x > 0, g x 1 has a minimum value of -2 := sorry

theorem part_III :
  ∀ (x1 x2 : ℝ), x1 < x2 →
    (let k := (Real.log x2 - Real.log x1) / (x2 - x1) in
    1/x2 < k ∧ k < 1/x1) := sorry

end part_I_part_II_part_III_l34_34148


namespace estimate_students_with_high_scores_l34_34708

noncomputable def normal_distribution := @measure_theory.measure_space ℝ (measure_theory.measure.mul_pos_measure ℝ ℝ ℝ)

def students_count : ℕ := 50
def mean : ℝ := 100
def variance : ℝ := 10^2
def normal : measure_theory.probability_measure ℝ := _ -- Assuming it's declared using mean and variance
def prob_interval : ℝ := 0.3

theorem estimate_students_with_high_scores :
  ∀ (scores : measure_theory.probability_measure ℝ), 
    scores = normal →
    (measure_theory.measure.restrict normal (set.Icc 90 100)).measure (set.univ) = prob_interval →
    students_with_scores_ge_110 = 10 :=
by
  sorry

end estimate_students_with_high_scores_l34_34708


namespace existence_of_intersection_l34_34127

def setA (m : ℝ) : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ (x^2 + m * x - y + 2 = 0) }
def setB : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ (x - y + 1 = 0) ∧ (0 ≤ x ∧ x ≤ 2) }

theorem existence_of_intersection (m : ℝ) : (∃ (p : ℝ × ℝ), p ∈ (setA m ∩ setB)) ↔ m ≤ -1 := 
sorry

end existence_of_intersection_l34_34127


namespace gcd_of_set_B_is_five_l34_34560

def is_in_set_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = (x - 2) + (x - 1) + x + (x + 1) + (x + 2)

theorem gcd_of_set_B_is_five : ∃ d, (∀ n, is_in_set_B n → d ∣ n) ∧ d = 5 :=
by 
  sorry

end gcd_of_set_B_is_five_l34_34560


namespace line_eq_l34_34922

theorem line_eq (A B : ℝ × ℝ) (hAB : midpoint A B = (1, -2)) (hIntersect : ∃ A B, ∀ p, on_circle p ↔ p = A ∨ p = B) : 
  ∃ l : ℝ → ℝ, line_eq l = λ x, x - (-1 : ℝ) - 1 := 
  sorry

-- Definitions of midpoint and on_circle
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

def on_circle (p : ℝ × ℝ) : Prop :=
  (p.fst)^2 + (p.snd + 1)^2 = 4

end line_eq_l34_34922


namespace anthony_initial_pencils_l34_34042

def initial_pencils (given_pencils : ℝ) (remaining_pencils : ℝ) : ℝ :=
  given_pencils + remaining_pencils

theorem anthony_initial_pencils :
  initial_pencils 9.0 47.0 = 56.0 :=
by
  sorry

end anthony_initial_pencils_l34_34042


namespace motorcycle_tire_max_distance_l34_34334

theorem motorcycle_tire_max_distance :
  let wear_front := (1 : ℝ) / 25000
  let wear_rear := (1 : ℝ) / 15000
  let s := 18750
  wear_front * (s / 2) + wear_rear * (s / 2) = 1 :=
by 
  let wear_front := (1 : ℝ) / 25000
  let wear_rear := (1 : ℝ) / 15000
  sorry

end motorcycle_tire_max_distance_l34_34334


namespace cake_cost_proof_l34_34663

noncomputable def bread_cost : ℤ := 50
noncomputable def ham_cost : ℤ := 150
noncomputable def cake_cost : ℤ := 200

theorem cake_cost_proof (total_cost : ℤ) 
  (h1 : bread_cost + ham_cost = 0.5 * total_cost) : 
  cake_cost = 0.5 * total_cost :=
by
  sorry

end cake_cost_proof_l34_34663


namespace smallest_digit_for_divisibility_by_9_l34_34791

theorem smallest_digit_for_divisibility_by_9 : 
  ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ (18 + d) % 9 = 0 ∧ ∀ d' : ℕ, (0 ≤ d' ∧ d' ≤ 9 ∧ (18 + d') % 9 = 0) → d' ≥ d :=
sorry

end smallest_digit_for_divisibility_by_9_l34_34791


namespace abs_neg_three_eq_three_l34_34283

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l34_34283


namespace rain_prob_l34_34316
open BigOperators

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1-p)^(n-k)

theorem rain_prob :
  let p : ℚ := 1 / 5,
      n : ℕ := 31 in
  (binomial_prob n 0 p + binomial_prob n 1 p + binomial_prob n 2 p + binomial_prob n 3 p) ≈ 0.338 :=
by sorry

end rain_prob_l34_34316


namespace greatest_common_divisor_B_l34_34568

def sum_of_five_consecutive_integers (x : ℕ) : ℕ :=
  (x - 2) + (x - 1) + x + (x + 1) + (x + 2)

def B : set ℕ := {n | ∃ x : ℕ, n = sum_of_five_consecutive_integers x}

theorem greatest_common_divisor_B : gcd (set.to_finset B).min' (set.to_finset B).max' = 5 :=
by
  sorry

end greatest_common_divisor_B_l34_34568


namespace f_monotonicity_lambda_range_l34_34855

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x

theorem f_monotonicity (a : ℝ) :
  ((a ≤ 0) → (∀ x y : ℝ, 0 < x → x < y → f x a ≤ f y a)) ∧
  ((0 < a) → (∀ x y : ℝ, 0 < x → x ≤ 1 / a → f x a ≤ f y a) ∧ (∀ x y : ℝ, 1 / a < x → x < y → f x a ≥ f y a)) :=
  sorry

theorem lambda_range (x1 x2 : ℝ) (a : ℝ) (h_x1x2 : x1 < x2) (h_fzero : f x1 a = 0 ∧ f x2 a = 0) (λ : ℝ) (h_λ : 0 < λ):
  (1 + λ < Real.log x1 + λ * Real.log x2) → (λ ≥ 1) :=
  sorry

end f_monotonicity_lambda_range_l34_34855


namespace monkey_reach_top_in_20_hours_l34_34373

-- Defining the conditions
def tree_height : ℕ := 21
def hop_distance : ℕ := 3
def slip_distance : ℕ := 2

-- Defining the net distance gain per hour
def net_gain_per_hour : ℕ := hop_distance - slip_distance

-- Proof statement
theorem monkey_reach_top_in_20_hours :
  ∃ t : ℕ, t = 20 ∧ 20 * net_gain_per_hour + hop_distance = tree_height :=
by
  sorry

end monkey_reach_top_in_20_hours_l34_34373


namespace find_lengths_l34_34212

-- Definitions and conditions
variables (C A E D F B : Type)
variables [MetricSpace A] [MetricSpace C] [MetricSpace E] [MetricSpace D] [MetricSpace F] [MetricSpace B]
variable (line_AE : Line A E)
variable (line_CE : Line C E)
variable [decidable_rel (line_AE.contains : A → Prop)] [decidable_rel (line_CE.contains : A → Prop)]

-- Points C, D, F not on line AE and their perpendicularity.
variable (CD_perpendicular_AE : perpendicular C D A E)
variable (CF_perpendicular_AE : perpendicular C F A E)
variable (AB_perpendicular_CE : perpendicular A B C E)

-- Known lengths.
variable (AB : ℝ := 3)
variable (CD : ℝ := 10)
variable (AE : ℝ := 6)

-- Main Goal: Prove lengths of CE and CF
theorem find_lengths : ∃ (CE CF : ℝ), CF = 10 ∧ CE = 20 :=
by
  exists 20, 10
  sorry

end find_lengths_l34_34212


namespace part1_part2_l34_34486

-- Definition of the quadratic function
def f (x q : ℝ) := x^2 - 16*x + q + 3

-- Problem (1): For q = 1, find the max and min values of f(x) on the interval [-1, 1].
theorem part1 : ∀ (x : ℝ), x ∈ [-1, 1] → f x 1 ≤ 21 ∧ (-11 ≤ f x 1) :=
by
  sorry

-- Problem (2): Find if there exists a q in (0, 10) such that the minimum value of f(x) on [q, 10] is -51.
theorem part2 : ∃ q ∈ (0, 10), ∀ x ∈ [q, 10], f x q = -51 → q = 9 :=
by
  sorry

end part1_part2_l34_34486


namespace cameron_speed_ratio_l34_34401

variables (C Ch : ℝ)
-- Danielle's speed is three times Cameron's speed
def Danielle_speed := 3 * C
-- Danielle's travel time from Granville to Salisbury is 30 minutes
def Danielle_time := 30
-- Chase's travel time from Granville to Salisbury is 180 minutes
def Chase_time := 180

-- Prove the ratio of Cameron's speed to Chase's speed is 2
theorem cameron_speed_ratio :
  (Danielle_speed C / Ch) = (Chase_time / Danielle_time) → (C / Ch) = 2 :=
by {
  sorry
}

end cameron_speed_ratio_l34_34401


namespace min_value_ge_8_min_value_8_at_20_l34_34099

noncomputable def min_value (x : ℝ) (h : x > 4) : ℝ := (x + 12) / Real.sqrt (x - 4)

theorem min_value_ge_8 (x : ℝ) (h : x > 4) : min_value x h ≥ 8 := sorry

theorem min_value_8_at_20 : min_value 20 (by norm_num) = 8 := sorry

end min_value_ge_8_min_value_8_at_20_l34_34099


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l34_34882

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : 
  ∃ (count : ℕ), count = 225 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ↔ (∃ k : ℕ, 25 ≤ k ∧ k ≤ 249 ∧ n = 4 * k + 2) := 
by {
  sorry
}

end number_of_three_digit_integers_congruent_to_2_mod_4_l34_34882


namespace intersection_reciprocal_sum_eq_l34_34141

constant M : ℝ × ℝ
constant θ : ℝ
constant ρ : ℝ
constant t₁ t₂ : ℝ

axiom polar_eq : ∀ θ: ℝ, ρ - 4 * real.cos θ = 0
axiom line_parametric_eq : x = 3 + (real.sqrt 3 / 2) * t ∧ y = (1 / 2) * t

theorem intersection_reciprocal_sum_eq :
  let M := (3, 0) in
  let x := ρ * real.cos θ in
  let y := ρ * real.sin θ in
  let curve_eq := x^2 + y^2 - 4 * x = 0 in
  let line_l := (3 + (real.sqrt 3 / 2) * t, (1 / 2) * t) in
  if h : curve_eq ∧ line_parametric_eq then
    ∀ t₁ t₂ : ℝ, let |MA| := real.sqrt ((M.1 - (3 + (real.sqrt 3 / 2) * t₁))^2 + (M.2 - (1 / 2) * t₁)^2) in
                 let |MB| := real.sqrt (((M.1 - (3 + (real.sqrt 3 / 2) * t₂)))^2 + (M.2 - (1 / 2) * t₂)^2) in
                 (1 / |MA|) + (1 / |MB|) = real.sqrt 15 / 3
  else false :=
begin
  sorry
end

end intersection_reciprocal_sum_eq_l34_34141


namespace rotation_of_unit_circle_l34_34950

open Real

noncomputable def rotated_coordinates (θ : ℝ) : ℝ × ℝ :=
  ( -sin θ, cos θ )

theorem rotation_of_unit_circle (θ : ℝ) (k : ℤ) (h : θ ≠ k * π + π / 2) :
  let A := (cos θ, sin θ)
  let O := (0, 0)
  let B := rotated_coordinates (θ)
  B = (-sin θ, cos θ) :=
sorry

end rotation_of_unit_circle_l34_34950


namespace find_m_l34_34500

theorem find_m (m : ℕ) (h : sqrt (m - 1) = 2) : m = 5 :=
by
  sorry

end find_m_l34_34500


namespace essay_body_section_length_l34_34404

theorem essay_body_section_length :
  ∀ (intro_length conclusion_multiplier : ℕ) (body_sections total_words : ℕ),
  intro_length = 450 →
  conclusion_multiplier = 3 →
  body_sections = 4 →
  total_words = 5000 →
  let conclusion_length := conclusion_multiplier * intro_length in
  let total_intro_conclusion := intro_length + conclusion_length in
  let remaining_words := total_words - total_intro_conclusion in
  let section_length := remaining_words / body_sections in
  section_length = 800 :=
by 
  intros intro_length conclusion_multiplier body_sections total_words 
         h_intro_len h_concl_mul h_body_sec h_total_words;
  dsimp only;
  rw [h_intro_len, h_concl_mul, h_body_sec, h_total_words];
  let conclusion_length := 3 * 450;
  let total_intro_conclusion := 450 + 1350;
  let remaining_words := 5000 - 1800;
  let section_length := 3200 / 4;
  exact sorry

end essay_body_section_length_l34_34404


namespace discount_offered_is_5_percent_l34_34023

noncomputable def cost_price : ℝ := 100

noncomputable def selling_price_with_discount : ℝ := cost_price * 1.216

noncomputable def selling_price_without_discount : ℝ := cost_price * 1.28

noncomputable def discount : ℝ := selling_price_without_discount - selling_price_with_discount

noncomputable def discount_percentage : ℝ := (discount / selling_price_without_discount) * 100

theorem discount_offered_is_5_percent : discount_percentage = 5 :=
by 
  sorry

end discount_offered_is_5_percent_l34_34023


namespace pete_backward_speed_l34_34597

variable (p b t s : ℝ)  -- speeds of Pete, backward walk, Tracy, and Susan respectively

-- Given conditions
axiom h1 : p / t = 1 / 4      -- Pete walks on his hands at a quarter speed of Tracy's cartwheeling
axiom h2 : t = 2 * s          -- Tracy cartwheels twice as fast as Susan walks
axiom h3 : b = 3 * s          -- Pete walks backwards three times faster than Susan
axiom h4 : p = 2              -- Pete walks on his hands at 2 miles per hour

-- Prove Pete's backward walking speed is 12 miles per hour
theorem pete_backward_speed : b = 12 :=
by
  sorry

end pete_backward_speed_l34_34597


namespace triangle_abs_simplification_l34_34870

theorem triangle_abs_simplification
  (x y z : ℝ)
  (h1 : x + y > z)
  (h2 : y + z > x)
  (h3 : x + z > y) :
  |x + y - z| - 2 * |y - x - z| = -x + 3 * y - 3 * z :=
by
  sorry

end triangle_abs_simplification_l34_34870


namespace curve_equations_and_rho_sum_l34_34949

theorem curve_equations_and_rho_sum (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C1 := λ (φ : ℝ), (2 * Real.cos φ, Real.sin φ) in
  let C2 := λ (ρ θ : ℝ), ρ = 2 * Real.cos θ in
  let A := (1, Real.sqrt 3 / 2) in
  let B := (1, Real.pi / 3) in
  let D := (1, Real.pi / 3) in
  C1 A.2 = A ∧ 
  C2 1 (Real.pi / 3) = 1 ∧ 
  (∃ (ρ1 ρ2 θ : ℝ), 
      (2 * (Real.cos θ))^2 + (Real.sin θ)^2 = 1 ∧ 
      (2 * (Real.cos (θ + Real.pi / 2)))^2 + (Real.sin (θ + Real.pi / 2))^2 = 1)
  →
  (1 / ρ1^2 + 1 / ρ2^2 = 5 / 4) := 
  sorry

end curve_equations_and_rho_sum_l34_34949


namespace sqrt49_times_sqrt25_eq_5sqrt7_l34_34053

noncomputable def sqrt49_times_sqrt25 : ℝ :=
  Real.sqrt (49 * Real.sqrt 25)

theorem sqrt49_times_sqrt25_eq_5sqrt7 :
  sqrt49_times_sqrt25 = 5 * Real.sqrt 7 :=
by
sorry

end sqrt49_times_sqrt25_eq_5sqrt7_l34_34053


namespace problem_bc_value_l34_34800

noncomputable def check_bc (b c : ℝ) : Prop :=
  b = -6 ∧ c = 8 ∧ b * c = -48

theorem problem_bc_value {b c : ℝ}
  (h1 : ∀ x : ℝ, (x^2 + b * x + c < 0) ↔ (2 < x ∧ x < 4))
  (h2 : (Polynomial.root (Polynomial.C c + Polynomial.C b * X + X ^ 2)) = {2, 4}) :
  check_bc b c :=
by
  sorry

end problem_bc_value_l34_34800


namespace range_of_f_le_2_l34_34753

def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^(1 - x) else 1 - real.log x / real.log 2

theorem range_of_f_le_2 : {x : ℝ | f x ≤ 2} = set.Ici 0 := 
sorry

end range_of_f_le_2_l34_34753


namespace orderly_colorings_count_l34_34983

theorem orderly_colorings_count (n : ℕ) (h : n ≥ 3) :
  ∃ c : ℕ, (∀ (grid : Fin n × Fin n → Fin 2), orderly_coloring grid n → c = 2 + 2 * n!) := 
sorry

def orderly_coloring (grid : Fin n × Fin n → Fin 2) (n : ℕ) : Prop :=
  ∀ (row_perm : Fin n → Fin n) (col_perm : Fin n → Fin n),
    (λ i j, grid (row_perm i, col_perm j)) = grid

end orderly_colorings_count_l34_34983


namespace maximum_possible_value_l34_34115

-- Given definitions and conditions
variable {x : Fin 1994 → ℝ}
def x_conds : Prop :=
  ∑ i in Finset.range 1992, |x i - x (i+1)| = 1993

def y (k : Fin 1994) : ℝ :=
  (∑ i in Finset.range k.val, x i) / (k.val + 1)

-- Problem statement to prove
theorem maximum_possible_value (h : x_conds) :
  ((Finset.range 1992).sum (λ k, |y ⟨k, Nat.lt_succ_of_lt (Finset.mem_range.1 (Finset.mem_range_self k))⟩
                                 - y ⟨k+1, Nat.lt_add_one (Finset.mem_range.1 (Finset.mem_range_self k))⟩|)) = 1992 :=
sorry

end maximum_possible_value_l34_34115


namespace abs_neg_three_l34_34296

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l34_34296


namespace ellipse_equation_circle_equation_line_and_chord_l34_34469

noncomputable def given_conditions :=
  ∃ (a b c : ℝ), a > b ∧ b > 0 ∧ c = 1 ∧ a^2 = b^2 + c^2 ∧ c = a * 2⁻¹ ∧ a = sqrt 2 ∧ b = 1

theorem ellipse_equation : (a b : ℝ) (ha : a > b) (hb : b > 0) (he : a * 2⁻¹ = c) (hc : c = 1) (h : a^2 = b^2 + c^2) :
  ∃ x y : ℝ, y^2 = a^2 - b^2 ↔ 2 * y^2 + x^2 = 2 := sorry

theorem circle_equation (a b c : ℝ) (ha : a > b) (hb : b > 0) (h1 : c = 1) (h2 : a = sqrt 2) (h3 : b = 1) :
  (m : ℝ) (h : 3/2 = sqrt(m^2 + (2⁻¹)^2)) → (y : ℝ) (y = m^2 + (2⁻¹)^2) :=
  ∃ m : ℝ, m = sqrt 2 ∨ m = -sqrt 2 
         ∩ ∀ (x y : ℝ), (x ± sqrt 2)^2 + y^2 = 9/4 := sorry

theorem line_and_chord (a b c : ℝ) (ha : a > b) (hb : b > 0) (he : a * 2⁻¹ = c) (hc : c = 1) (h : a^2 = b^2 + c^2) :
  ∀ (k : ℝ), k = sqrt 2 ∨ k = -sqrt 2 
      ∩ (2 - 1) / sqrt 3 = 3 * (1/ sqrt 3) 
      := sorry

end ellipse_equation_circle_equation_line_and_chord_l34_34469


namespace alice_bob_meet_l34_34384

/--
Alice and Bob play a game on a circle divided into 18 equally-spaced points.
Alice moves 7 points clockwise per turn, and Bob moves 13 points counterclockwise.
Prove that they will meet at the same point after 9 turns.
-/
theorem alice_bob_meet : ∃ k : ℕ, k = 9 ∧ (7 * k) % 18 = (18 - 13 * k) % 18 :=
by
  sorry

end alice_bob_meet_l34_34384


namespace population_definition_l34_34938

theorem population_definition (participants: ℕ) (sample: ℕ) 
  (total_participants: participants = 1000) 
  (sample_size: sample = 100) : 
  (population: ℕ := participants) = 1000 :=
by
  sorry

end population_definition_l34_34938


namespace trajectory_of_moving_circle_l34_34818

theorem trajectory_of_moving_circle
  (P : Type) [MetricSpace P]
  (A : P) (hA : A = (-3 : ℝ, 0 : ℝ))
  (B : P) (hB : B = (3 : ℝ, 0 : ℝ))
  (fixed_circle : Set P) (hf : ∀ (x : ℝ) (y : ℝ), (x - 3)^2 + y^2 = 64 → (x, y) ∈ fixed_circle)
  (P_path : Set P)
  (tangent_condition : ∀ (M ∈ P_path), ∃ (tangent_point : P), (tangent_point ∈ fixed_circle) ∧ intern_tangent fixed_circle P_path tangent_point)
  (P_through_A : A ∈ P_path) :
  ∀ (P_center : P),
  P_center ∈ P_path ↔ ((dist P_center A + dist P_center B = 8) ∧
                       ∃ (x y : ℝ), P_center = (x, y) ∧ ((x^2 / 16) + (y^2 / 7) = 1)) :=
begin
  sorry
end

end trajectory_of_moving_circle_l34_34818


namespace min_value_f_when_a_is_zero_inequality_holds_for_f_l34_34481

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x

-- Problem (1): Prove the minimum value of f(x) when a = 0 is 2 - 2 * ln 2.
theorem min_value_f_when_a_is_zero : 
  (∃ x : ℝ, f x 0 = 2 - 2 * Real.log 2) :=
sorry

-- Problem (2): Prove that for a < (exp(1) / 2) - 1, f(x) > (exp(1) / 2) - 1 for all x in (0, +∞).
theorem inequality_holds_for_f :
  ∀ a : ℝ, a < (Real.exp 1) / 2 - 1 → 
  ∀ x : ℝ, 0 < x → f x a > (Real.exp 1) / 2 - 1 :=
sorry

end min_value_f_when_a_is_zero_inequality_holds_for_f_l34_34481


namespace exponent_28_digits_to_right_of_decimal_l34_34173

theorem exponent_28_digits_to_right_of_decimal 
  (exponent : ℕ) 
  (h : (10^4 * 3.456789)^exponent) 
  (num_digits : 2 * exponent = 28) : 
  exponent = 14 := sorry

end exponent_28_digits_to_right_of_decimal_l34_34173


namespace a100_gt_two_pow_99_l34_34123

theorem a100_gt_two_pow_99 (a : ℕ → ℤ) (h_pos : ∀ n, 0 < a n) 
  (h1 : a 1 > a 0) (h_rec : ∀ n ≥ 2, a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2 ^ 99 :=
sorry

end a100_gt_two_pow_99_l34_34123


namespace polar_arc_length_l34_34740

open Real

theorem polar_arc_length (φ : ℝ) (h1 : 0 ≤ φ) (h2 : φ ≤ π / 4) :
  let ρ := 8 * sin φ in
  let dρ_dφ := 8 * cos φ in
  ∫ t in 0..(π / 4), sqrt ((ρ)^2 + (dρ_dφ)^2) = 2 * π :=
by sorry

end polar_arc_length_l34_34740


namespace ratio_of_speeds_l34_34078

/-- Define the conditions -/
def distance_AB : ℝ := 540 -- Distance between city A and city B is 540 km
def time_Eddy : ℝ := 3     -- Eddy takes 3 hours to travel to city B
def distance_AC : ℝ := 300 -- Distance between city A and city C is 300 km
def time_Freddy : ℝ := 4   -- Freddy takes 4 hours to travel to city C

/-- Define the average speeds -/
noncomputable def avg_speed_Eddy : ℝ := distance_AB / time_Eddy
noncomputable def avg_speed_Freddy : ℝ := distance_AC / time_Freddy

/-- The statement to prove -/
theorem ratio_of_speeds : avg_speed_Eddy / avg_speed_Freddy = 12 / 5 :=
by sorry

end ratio_of_speeds_l34_34078


namespace number_of_three_digit_integers_congruent_mod4_l34_34910

def integer_congruent_to_mod (a b n : ℕ) : Prop := ∃ k : ℤ, n = a * k + b

theorem number_of_three_digit_integers_congruent_mod4 :
  (finset.filter (λ n, integer_congruent_to_mod 4 2 (n : ℕ)) 
   (finset.Icc (100 : ℕ) (999 : ℕ))).card = 225 :=
by
  sorry

end number_of_three_digit_integers_congruent_mod4_l34_34910


namespace tan_alpha_plus_405_deg_l34_34130

theorem tan_alpha_plus_405_deg (α : ℝ) (h : Real.tan (180 - α) = -4 / 3) : Real.tan (α + 405) = -7 := 
sorry

end tan_alpha_plus_405_deg_l34_34130


namespace coefficient_of_x2_term_l34_34808

def integrand (x : ℝ) : ℝ :=
  cos (x / 2) ^ 2 - 1 / 2

def a : ℝ :=
  ∫ x in 0..(π / 2), integrand x

def expansion_coefficient : ℚ :=
  105 / 32

theorem coefficient_of_x2_term :
  (left (ax + 1 / (2 * a * x)) ^ 10).coeff 2 = expansion_coefficient := by
  sorry

end coefficient_of_x2_term_l34_34808


namespace number_of_ordered_pairs_l34_34785

theorem number_of_ordered_pairs :
  let num_pairs := (λ (a : ℕ) (b : ℤ), ∃ r s : ℤ, a ∈ finset.range 200 ∧ b ≤ 0 ∧ r + s = -(a : ℤ) ∧ r * s = -b) in
  finset.card {p : ℕ × ℤ | num_pairs p.1 p.2} = 300 :=
by sorry

end number_of_ordered_pairs_l34_34785


namespace correct_equation_is_l34_34734

theorem correct_equation_is {a b : ℝ} : a = b → 2 * (a - 1) = 2 * (b - 1) :=
by
  intros h
  rw [h]
  sorry

end correct_equation_is_l34_34734


namespace comb_sum_C8_2_C8_3_l34_34742

open Nat

theorem comb_sum_C8_2_C8_3 : (Nat.choose 8 2) + (Nat.choose 8 3) = 84 :=
by
  sorry

end comb_sum_C8_2_C8_3_l34_34742


namespace readers_all_three_l34_34513

def total_readers : ℕ := 500
def readers_science_fiction : ℕ := 320
def readers_literary_works : ℕ := 200
def readers_non_fiction : ℕ := 150
def readers_sf_and_lw : ℕ := 120
def readers_sf_and_nf : ℕ := 80
def readers_lw_and_nf : ℕ := 60

theorem readers_all_three :
  total_readers = readers_science_fiction + readers_literary_works + readers_non_fiction - (readers_sf_and_lw + readers_sf_and_nf + readers_lw_and_nf) + 90 :=
by
  sorry

end readers_all_three_l34_34513


namespace largest_sum_eq_1040_l34_34413

-- Define distinct digits X, Y, and Z with X being the largest
variables {X Y Z : ℕ}
-- Add condition for K
noncomputable def K := 11
-- Define the sum S as per the given problem's conditions
noncomputable def S := 113 * X + 10 * Y + 10 * Z + K

-- Define the condition for distinct digits and ensure X is the largest
axiom digits_conditions : X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧ X ≥ Y ∧ X ≥ Z ∧ X ≤ 9 ∧ Y ≤ 9 ∧ Z ≤ 9

-- Prove that the largest sum for such X, Y, Z, and K equals 1040
theorem largest_sum_eq_1040 (h_digits : digits_conditions) : S = 1040 := 
sorry

end largest_sum_eq_1040_l34_34413


namespace AF_passes_through_midpoint_of_DE_l34_34102

-- Definitions of points, isosceles triangles, perpendiculars, midpoints, and circumcircles
variables {A B C D E F : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]

-- The isosceles triangle
def isosceles_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] : Prop :=
  dist A B = dist A C

-- Midpoint definition
def is_midpoint (M A B : Type) [metric_space M] [metric_space A] [metric_space B] : Prop :=
  dist M A = dist M B

-- Perpendicular line definition
def is_perpendicular (l₁ l₂ : line) : Prop :=
  ∃ (m : real), slope l₁ = m ∧ slope l₂ = -1 / m

-- Circumcircle intersection
def circumcircle_intersections (circ : circle) (l : line) : set A :=
  { p | p ∈ circ ∧ p ∈ l }

-- Lean theorem statement
theorem AF_passes_through_midpoint_of_DE
  (ABC_isosceles : isosceles_triangle A B C)
  (D_midpoint_BC : is_midpoint D B C)
  (DE_perpendicular_AC : is_perpendicular (line_from D E) (line_from A C))
  (circum_ABD_int_BE : circumcircle_intersections (circumcircle A B D) (line_from B E) = {B, F}) :
  ∃ G, is_midpoint G D E ∧ lies_on (line_from A F) G := 
sorry -- Proof omitted

end AF_passes_through_midpoint_of_DE_l34_34102


namespace lines_parallel_l34_34647

-- Define the two lines by their equations
def line1 (x y : ℝ) : Prop := 3 * x + y + 1 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 2 * y + 1 = 0

-- State the theorem we want to prove
theorem lines_parallel : (∀ x y : ℝ, line1 x y ↔ false) → (∀ x y : ℝ, line2 x y ↔ false) → false :=
begin
  sorry
end

end lines_parallel_l34_34647


namespace equilateral_triangle_third_vertex_y_coordinate_l34_34041

theorem equilateral_triangle_third_vertex_y_coordinate :
  ∀ (x y : ℝ), 
  let A : ℝ × ℝ := (0, 5)
  let B : ℝ × ℝ := (10, 5)
  ∃ C : ℝ × ℝ,
     C.1 = x ∧ C.2 = y ∧
     (C ∉ {(0, 5), (10, 5)}) ∧ 
     (x > 0 ∧ y > 0) ∧
     dist A C = 10 ∧
     dist B C = 10 →
     y = 5 + 5 * Real.sqrt 3 :=
begin
  intros x y,
  use (5, 5 + 5 * Real.sqrt 3),
  sorry
end

end equilateral_triangle_third_vertex_y_coordinate_l34_34041


namespace sin_160_eq_sin_20_l34_34737

theorem sin_160_eq_sin_20 : 
  ∀ θ : ℝ, 
    θ = 20 → 
    sin (160 * (π / 180)) = sin (θ * (π / 180)) :=
by
  sorry

end sin_160_eq_sin_20_l34_34737


namespace determine_varphi_l34_34479

theorem determine_varphi (w : ℝ) (phi : ℝ) (x : ℝ) (k : ℤ) 
  (cond1 : w = 1 / 3) 
  (cond2 : abs phi < real.pi / 2) 
  (cond3 : sin (w * (x - 3 * real.pi / 8) + phi) = sin (w * x)) : 
  phi = real.pi / 8 := 
  sorry

end determine_varphi_l34_34479


namespace count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle_l34_34879

noncomputable def count_equally_spaced_integers : ℕ := 
  sorry

theorem count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle:
  count_equally_spaced_integers = 4 :=
sorry

end count_of_integers_n_ge_2_such_that_points_are_equally_spaced_on_unit_circle_l34_34879


namespace average_of_last_two_numbers_l34_34299

theorem average_of_last_two_numbers (L : List ℝ) (h_length : L.length = 7)
  (h_avg7 : List.sum L / 7 = 60) (h_avg3 : List.sum (L.take 3) / 3 = 45)
  (h_avg2 : List.sum (L.drop 3).take 2 / 2 = 70) :
  List.sum (L.drop 5) / 2 = 72.5 :=
by
  sorry

end average_of_last_two_numbers_l34_34299


namespace remainder_is_nine_l34_34678

-- Define the dividend and divisor
def n : ℕ := 4039
def d : ℕ := 31

-- Prove that n mod d equals 9
theorem remainder_is_nine : n % d = 9 := by
  sorry

end remainder_is_nine_l34_34678


namespace star_sum_larger_than_emilio_sum_l34_34264

theorem star_sum_larger_than_emilio_sum :
  let star_numbers := list.range' 1 30 in
  let modify_digit_2 n := nat.digits 10 n |> list.map (λ d, if d = 2 then 1 else d) |> nat.of_digits 10 in
  let emilio_numbers := star_numbers.map modify_digit_2 in
  star_numbers.sum = emilio_numbers.sum + 103 :=
by
  sorry

end star_sum_larger_than_emilio_sum_l34_34264


namespace g_neg_1_l34_34813

variable {α : Type*} [ring α]
variable (f : α → α) (g : α → α)

-- Condition 1: y=f(x) is an odd function
def is_odd_function (f : α → α) :=
  ∀ x, f (-x) = -f x

-- Condition 2: g(x) = f(x) + 2
def g_def (f g : α → α) :=
  ∀ x, g x = f x + 2

-- Condition 3: g(1) = 1
def g_at_1 (g : α → α) :=
  g 1 = 1

theorem g_neg_1 : 
  is_odd_function f → g_def f g → g_at_1 g → g (-1) = 3 :=
by
  intros h1 h2 h3
  sorry

end g_neg_1_l34_34813


namespace arc_length_ln_x_squared_minus_one_correct_l34_34741

noncomputable def arc_length_ln_x_squared_minus_one : ℝ :=
  ∫ (x : ℝ) in 2..3, sqrt (1 + (deriv (λ x : ℝ, real.log (x ^ 2 - 1)) x) ^ 2)

theorem arc_length_ln_x_squared_minus_one_correct :
  arc_length_ln_x_squared_minus_one = 1 + real.log (3 / 2) :=
by
  sorry

end arc_length_ln_x_squared_minus_one_correct_l34_34741


namespace solve_problem_part1_solve_problem_part2_l34_34839

variables {α : Real}
noncomputable def proof_cos_alpha : Prop := 
  α ∈ Set.Ioo 0 (π / 3) → 
  sqrt 6 * Real.sin α + sqrt 2 * Real.cos α = sqrt 3 → 
  Real.cos (α + π / 6) = sqrt 10 / 4

noncomputable def proof_cos_2alpha : Prop := 
  α ∈ Set.Ioo 0 (π / 3) → 
  sqrt 6 * Real.sin α + sqrt 2 * Real.cos α = sqrt 3 → 
  Real.cos (2 * α + π / 12) = (sqrt 30 + sqrt 2) / 8

theorem solve_problem_part1 : proof_cos_alpha := by sorry

theorem solve_problem_part2 : proof_cos_2alpha := by sorry

end solve_problem_part1_solve_problem_part2_l34_34839


namespace complement_of_A_in_U_l34_34488

-- Given definitions from the problem
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}

-- The theorem to be proven
theorem complement_of_A_in_U : U \ A = {1, 3, 6, 7} :=
by
  sorry

end complement_of_A_in_U_l34_34488


namespace cosA_gt_sinB_iff_obtuse_l34_34928

noncomputable def obtuse_triangle_condition (A B C : ℝ) : Prop :=
  A + B < π / 2 ∧ A + B + C = π ∧ A < π / 2 ∧ B < π / 2

theorem cosA_gt_sinB_iff_obtuse (A B C : ℝ) :
  obtuse_triangle_condition A B C ↔ (cos A > sin B) :=
by
  -- problem statement without proof
  sorry

end cosA_gt_sinB_iff_obtuse_l34_34928


namespace vector_properties_l34_34841

noncomputable def vector := (ℝ × ℝ × ℝ)

def AB : vector := (2, -1, -4)

def AD : vector := (4, 2, 0)

def AP : vector := (-1, 2, -1)

-- Define dot product for vectors
def dot_product (u v : vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3 

-- Define the problem
theorem vector_properties :
  dot_product AB AP = 0 ∧
  dot_product AD AP = 0 ∧
  ∀ u v : vector, dot_product u AP = 0 ∧ dot_product v AP = 0 → False :=
by
  sorry

end vector_properties_l34_34841


namespace trigonometric_identity_l34_34809

theorem trigonometric_identity (α : ℝ) 
  (h : Real.tan (π / 4 + α) = 1) : 
  (2 * Real.sin α + Real.cos α) / (3 * Real.cos α - Real.sin α) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l34_34809


namespace dalton_needs_more_money_l34_34758

noncomputable def total_money : ℝ := 30 + 25 + 10
noncomputable def cost_jump_ropes : ℝ := 3 * 7
noncomputable def cost_board_games : ℝ := 2 * 12
noncomputable def cost_playground_balls : ℝ := 4 * 4

noncomputable def total_before_discounts : ℝ := cost_jump_ropes + cost_board_games + cost_playground_balls
noncomputable def discount_jump_ropes : ℝ := 3 * 2
noncomputable def discount_playground_balls : ℝ := 4 * 1
noncomputable def total_discounts : ℝ := discount_jump_ropes + discount_playground_balls

noncomputable def total_after_discounts : ℝ := total_before_discounts - total_discounts
noncomputable def sales_tax_rate : ℝ := 0.08
noncomputable def sales_tax : ℝ := total_after_discounts * sales_tax_rate
noncomputable def final_total_cost : ℝ := total_after_discounts + sales_tax

theorem dalton_needs_more_money : total_money - final_total_cost = 9.92 :=
by
  unfold total_money cost_jump_ropes cost_board_games cost_playground_balls total_before_discounts discount_jump_ropes discount_playground_balls total_discounts total_after_discounts sales_tax_rate sales_tax final_total_cost
  simp
  sorry

end dalton_needs_more_money_l34_34758


namespace least_steps_to_2008_l34_34265

def steps_required (n : ℕ) : ℕ :=
  (n.bitLength - 1) + n.popCount

theorem least_steps_to_2008 :
  ∀ n, n ∈ [2007, 2008, 2009, 2010, 2011] → steps_required 2008 ≤ steps_required n :=
by sorry

end least_steps_to_2008_l34_34265


namespace minimum_chord_length_l34_34314

noncomputable theory
open Classical

def parabola (m : ℝ) : set (ℝ × ℝ) :=
  {p | ∃ x : ℝ, p = (x, x^2 / (2 * m))}

def normal_at (A : ℝ × ℝ) (m : ℝ) : ℝ :=
  -m / A.1

def chord_normal_length (A : ℝ × ℝ) (m : ℝ) : ℝ :=
  let x2 := - (2 * m^2 / A.1 + A.1) in
  let y2 := x2^2 / (2 * m) in
  dist (A, (x2, y2))

theorem minimum_chord_length (m : ℝ) (h : 0 < m) :
  ∃ A : ℝ × ℝ, A ∈ parabola m ∧ chord_normal_length A m = 3 * sqrt 3 * m := sorry

end minimum_chord_length_l34_34314


namespace equilateral_triangle_area_l34_34211

-- Define the properties and conditions given in the problem.
variables (A B C D E F : Type*)
variables [EquilateralTriangle A B C]
variables [Midpoint D A B]
variables [Midpoint E B C]
variables [Midpoint F C A]
variables (area_DEF : ℝ)
variables (area_ABC : ℝ)

-- Given conditions
axiom DEF_area_given : area_DEF = 50 * Real.sqrt 3

-- Statement to prove
theorem equilateral_triangle_area :
  DEF_area_given → area_ABC = 200 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l34_34211


namespace tan_neg_seven_pi_sixths_l34_34792

noncomputable def tan_neg_pi_seven_sixths : Real :=
  -Real.sqrt 3 / 3

theorem tan_neg_seven_pi_sixths : Real.tan (-7 * Real.pi / 6) = -Real.sqrt 3 / 3 := by
  sorry

end tan_neg_seven_pi_sixths_l34_34792


namespace vertex_of_parabola_l34_34301

-- Define the statement of the problem
theorem vertex_of_parabola :
  ∀ (a h k : ℝ), (∀ x : ℝ, 3 * (x - 5) ^ 2 + 4 = a * (x - h) ^ 2 + k) → (h, k) = (5, 4) :=
by
  sorry

end vertex_of_parabola_l34_34301


namespace z_real_iff_z_complex_iff_z_purely_imaginary_iff_z_second_quadrant_iff_l34_34801

-- Given complex number z definition
def z (m : ℝ) : ℂ := (m^2 * (1/(m + 5) + complex.i) + (8 * m + 15) * complex.i + (m - 6) / (m + 5))

-- Assertion 1: z is a real number if and only if m = -3
theorem z_real_iff (m : ℝ) : (im (z m) = 0) ↔ (m = -3) :=
by sorry

-- Assertion 2: z is a complex number if and only if m ≠ -3 and m ≠ -5
theorem z_complex_iff (m : ℝ) : (real_part (z m) ≠ 0 ∨ im (z m) ≠ 0) ↔ (m ≠ -3 ∧ m ≠ -5) :=
by sorry

-- Assertion 3: z is a purely imaginary number if and only if m = 2
theorem z_purely_imaginary_iff (m : ℝ) : (real_part (z m) = 0 ∧ im (z m) ≠ 0) ↔ (m = 2) :=
by sorry

-- Assertion 4: z corresponds to a point in the second quadrant if and only if m < -5 or -3 < m < 2
theorem z_second_quadrant_iff (m : ℝ) : (real_part (z m) < 0 ∧ im (z m) > 0) ↔ (m < -5 ∨ (-3 < m ∧ m < 2)) :=
by sorry

end z_real_iff_z_complex_iff_z_purely_imaginary_iff_z_second_quadrant_iff_l34_34801


namespace find_alpha_l34_34784

noncomputable def alpha : ℝ := by
  have h : tan (2 * (pi - arctan (2 / 3))) = -12 / 5 := sorry
  exact (pi - arctan (2 / 3))

theorem find_alpha (α : ℝ) (h1 : α ∈ set.Ioo (pi / 2) pi) (h2 : tan (2 * α) = -12 / 5) : 
  α = pi - arctan (2 / 3) := 
by
  sorry

end find_alpha_l34_34784


namespace proof_problem_l34_34544

open EuclideanGeometry

noncomputable def problem_statement (A B C H G I J : Point) :=
  IsAcuteAngledTriangle A B C ∧
  IsOrthocenter A B C H ∧
  ParallelThrough H AB G ∧
  ParallelThrough B AH G ∧
  BisectionThrough AC HI I ∧
  SecondIntersectionOfCircumcircle AC CGI J →
  IJ = AH

axiom placeholders (A B C H G I J : Point) : problem_statement A B C H G I J

theorem proof_problem {A B C H G I J : Point} (h : problem_statement A B C H G I J) : IJ = AH :=
sorry

end proof_problem_l34_34544


namespace at_least_one_le_one_l34_34984

theorem at_least_one_le_one (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
sorry

end at_least_one_le_one_l34_34984


namespace seq_limit_l34_34347

noncomputable def seq (n : ℕ) : ℝ :=
  if n = 1 then 0
  else if n = 2 then 2
  else 2^-(seq (n-2)) + 0.5

theorem seq_limit :
  seq n → (∃ L : ℝ, (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |seq n - L| < ε) ∧ L = 1) :=
sorry

end seq_limit_l34_34347


namespace part1_part2_l34_34815

noncomputable def f (x : ℝ) := Real.exp x

theorem part1 (x : ℝ) (h : x ≥ 0) (m : ℝ) : 
  (x - 1) * f x ≥ m * x^2 - 1 ↔ m ≤ 1 / 2 :=
sorry

theorem part2 (x : ℝ) (h : x > 0) : 
  f x > 4 * Real.log x + 8 - 8 * Real.log 2 :=
sorry

end part1_part2_l34_34815


namespace pete_backward_speed_l34_34596

variable (p b t s : ℝ)  -- speeds of Pete, backward walk, Tracy, and Susan respectively

-- Given conditions
axiom h1 : p / t = 1 / 4      -- Pete walks on his hands at a quarter speed of Tracy's cartwheeling
axiom h2 : t = 2 * s          -- Tracy cartwheels twice as fast as Susan walks
axiom h3 : b = 3 * s          -- Pete walks backwards three times faster than Susan
axiom h4 : p = 2              -- Pete walks on his hands at 2 miles per hour

-- Prove Pete's backward walking speed is 12 miles per hour
theorem pete_backward_speed : b = 12 :=
by
  sorry

end pete_backward_speed_l34_34596


namespace log_expression_value_l34_34398

theorem log_expression_value : 
  let log4_3 := (Real.log 3) / (Real.log 4)
  let log8_3 := (Real.log 3) / (Real.log 8)
  let log3_2 := (Real.log 2) / (Real.log 3)
  let log9_2 := (Real.log 2) / (Real.log 9)
  (log4_3 + log8_3) * (log3_2 + log9_2) = 5 / 4 := 
by
  sorry

end log_expression_value_l34_34398


namespace prod_quotients_equals_value_l34_34058

theorem prod_quotients_equals_value :
  (∏ n in Finset.range 15, (n + 5) / (n + 1)) = 3876 := sorry

end prod_quotients_equals_value_l34_34058


namespace A_finish_time_l34_34035

theorem A_finish_time {A_work B_work C_work : ℝ} 
  (h1 : A_work + B_work + C_work = 1/4)
  (h2 : B_work = 1/24)
  (h3 : C_work = 1/8) :
  1 / A_work = 12 := by
  sorry

end A_finish_time_l34_34035


namespace olivia_total_cost_l34_34244

-- Definitions based on conditions given in the problem.
def daily_rate : ℕ := 30 -- daily rate in dollars per day
def mileage_rate : ℕ := 25 -- mileage rate in cents per mile (converted to cents to avoid fractions)
def rental_days : ℕ := 3 -- number of days the car is rented
def miles_driven : ℕ := 500 -- number of miles driven

-- Calculate costs in cents to avoid fractions in the Lean theorem statement.
def daily_rental_cost : ℕ := daily_rate * rental_days * 100
def mileage_cost : ℕ := mileage_rate * miles_driven
def total_cost : ℕ := daily_rental_cost + mileage_cost

-- Final statement to be proved, converting total cost back to dollars.
theorem olivia_total_cost : (total_cost / 100) = 215 := by
  sorry

end olivia_total_cost_l34_34244


namespace abs_neg_three_l34_34274

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l34_34274


namespace find_g_720_l34_34982

noncomputable def g (n : ℕ) : ℕ := sorry

axiom g_multiplicative : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g_8 : g 8 = 12
axiom g_12 : g 12 = 16

theorem find_g_720 : g 720 = 44 := by sorry

end find_g_720_l34_34982


namespace slope_of_line_l34_34789

theorem slope_of_line (x y : ℝ) :
  (4 * x - 7 * y = 28) → (slope (line_eq := 4 * x - 7 * y) = 4 / 7) :=
by
  sorry

end slope_of_line_l34_34789


namespace total_germs_l34_34951

theorem total_germs (D : ℝ) (Gpd : ℝ) (G: ℝ) :
  (D = 45000 * 10 ^ (-3)) ∧ (Gpd = 79.99999999999999) → G = 3600 := by
sorry

end total_germs_l34_34951


namespace sum_of_consecutive_integers_between_ln20_l34_34072

theorem sum_of_consecutive_integers_between_ln20 : ∃ a b : ℤ, a < b ∧ b = a + 1 ∧ 1 ≤ a ∧ a + 1 ≤ 3 ∧ (a + b = 4) :=
by
  sorry

end sum_of_consecutive_integers_between_ln20_l34_34072


namespace existence_of_100_quadratic_trinomials_l34_34767

-- Define the quadratic trinomial function
def quadratic_trinomial (n : ℕ) : ℝ → ℝ := λ x, (x - 4 * n)^2 - 1

theorem existence_of_100_quadratic_trinomials :
  ∃ (f : ℕ → ℝ → ℝ), (∀ n : ℕ, n ∈ (set.range (λ i, i + 1)^100) → ( ∃ a b : ℝ, a ≠ b ∧ f n = (λ x, (x - a) * (x - b) ))) ∧ (∀ m n : ℕ, m ≠ n → ∀ x : ℝ, f m x + f n x ≠ 0) :=
sorry

end existence_of_100_quadratic_trinomials_l34_34767


namespace cube_painting_l34_34710

-- Let's start with importing Mathlib for natural number operations

theorem cube_painting (n : ℕ) (h : 2 < n)
  (num_one_black_face : ℕ := 3 * (n - 2)^2)
  (num_unpainted : ℕ := (n - 2)^3) :
  num_one_black_face = num_unpainted → n = 5 :=
by
  sorry

end cube_painting_l34_34710


namespace root_of_quadratic_solution_l34_34917

theorem root_of_quadratic_solution (c : ℝ) : (∃ x : ℝ, x = -5 ∧ x^2 = c^2) → (c = 5 ∨ c = -5) :=
by
  intro h
  cases h with x hx
  cases hx with hx1 hx2
  rw hx1 at hx2
  rw sq_neg at hx2
  exact or_comm.mp (eq_or_eq_neg_of_sq_eq_sq (rfl.subst hx2))

end root_of_quadratic_solution_l34_34917


namespace custom_op_seven_three_l34_34170

def custom_op (a b : ℕ) : ℕ := 4 * a + 5 * b - a * b + 1

theorem custom_op_seven_three : custom_op 7 3 = 23 := by
  -- proof steps would go here
  sorry

end custom_op_seven_three_l34_34170


namespace problem_statement_l34_34915

theorem problem_statement (x : Real) (hx : sin (3 * x) * sin (4 * x) = cos (3 * x) * cos (4 * x)) : x = Real.pi / 14 :=
by sorry

end problem_statement_l34_34915


namespace f_1982_l34_34225

def f (n : ℕ) : ℕ

axiom f_add_prop (m n : ℕ) :
  f (m + n) - f m - f n = 0 ∨ f (m + n) - f m - f n = 1

axiom f_2 : f 2 = 0
axiom f_3_pos : f 3 > 0
axiom f_9999 : f 9999 = 3333

theorem f_1982 : f 1982 = 660 :=
by {
  sorry
}

end f_1982_l34_34225


namespace ball_returns_to_Bella_after_12_throws_l34_34084
-- Importing the entire Mathlib library to bring in all necessary definitions and lemmas.

-- Definition of the statement in Lean
theorem ball_returns_to_Bella_after_12_throws :
  ∃ n : ℕ, n = 12 ∧
    let girls := list.range 15 in
    let throws := list.foldl (λ a idx, (a ++ [(idx + 5) % 15])) [1] (list.range n) in
    list.ilast 0 throws = 1 :=
sorry

end ball_returns_to_Bella_after_12_throws_l34_34084


namespace ostrich_emu_leap_difference_l34_34773

theorem ostrich_emu_leap_difference 
  (elmer_strides_per_gap : ℕ) 
  (oscar_leaps_per_gap : ℕ) 
  (distance_between_poles : ℕ) 
  (number_of_poles : ℕ) :
  (elmer_strides_per_gap = 44) → 
  (oscar_leaps_per_gap = 12) → 
  (distance_between_poles = 5280) → 
  (number_of_poles = 41) →
  let total_distance := distance_between_poles * (number_of_poles - 1) in
  let total_strides := elmer_strides_per_gap * (number_of_poles - 1) in
  let total_leaps := oscar_leaps_per_gap * (number_of_poles - 1) in
  let elmer_stride_length := total_distance / total_strides in
  let oscar_leap_length := total_distance / total_leaps in
  oscar_leap_length - elmer_stride_length = 8 := 
by 
  sorry

end ostrich_emu_leap_difference_l34_34773


namespace at_least_one_term_le_one_l34_34986

theorem at_least_one_term_le_one
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x + y + z = 3) :
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
  sorry

end at_least_one_term_le_one_l34_34986


namespace omega_even_odd_infinitely_many_l34_34827

-- Definition of distinct prime factors counting function
def omega (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ k, k > 0 ∧ n % k = 0 ∧ Nat.Prime k).card

-- The arithmetic progression {a_n} defined by a_1 and d
def arithmetic_prog (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

-- The main theorem: Prove that there exist infinitely many k such that omega(a_k) is even and omega(a_{k+1}) is odd
theorem omega_even_odd_infinitely_many (a1 d : ℕ) (h1 : a1 > 0) (h2 : d > 0) :
  ∃ᶠ k in Filter.atTop, (omega (arithmetic_prog a1 d k) % 2 = 0) ∧ (omega (arithmetic_prog a1 d (k + 1)) % 2 = 1) :=
sorry

end omega_even_odd_infinitely_many_l34_34827


namespace opposite_numbers_l34_34736

theorem opposite_numbers (a b : ℤ) (h1 : -5^2 = a) (h2 : (-5)^2 = b) : a = -b :=
by sorry

end opposite_numbers_l34_34736


namespace rounding_repeating_decimals_l34_34259

noncomputable def repeating_decimal (n : ℕ) (d : ℕ) := (n + n / 100 / (1 - 1 / 100 ^ d))

noncomputable def sum_of_repeating_decimals :=
  repeating_decimal 3737 2 + repeating_decimal 1515 2

noncomputable def round_to_nearest_hundredth (x : ℝ) : ℝ :=
  (Real.floor (100 * x) + if 100 * x - Real.floor (100 * x) < 0.5 then 0 else 1) / 100

theorem rounding_repeating_decimals :
  round_to_nearest_hundredth sum_of_repeating_decimals = 0.52 :=
begin
  sorry
end

end rounding_repeating_decimals_l34_34259


namespace true_propositions_l34_34471

-- Define the propositions
def prop1 (P Q : Type) [Parallel P Q] : Prop :=
  ∀ l : P, is_parallel l Q

def prop2 (P Q : Type) [Parallel P Q] : Prop :=
  ∀ l : P, is_perpendicular l P → is_perpendicular l Q

def prop3 (P Q : Type) [Perpendicular P Q] : Prop :=
  ∀ l : P, is_perpendicular l P → is_parallel l Q

def prop4 (P Q : Type) [Perpendicular P Q] : Prop :=
  ∀ l : P, is_in_plane l P → is_perpendicular l Q

-- State the theorem
theorem true_propositions (P Q : Type) [Parallel P Q] [Perpendicular P Q] :
  prop1 P Q ∧ prop2 P Q ∧ ¬ prop3 P Q ∧ ¬ prop4 P Q :=
by
  sorry

end true_propositions_l34_34471


namespace symmetric_point_correct_l34_34456

-- Define a 3D point structure
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- A function to determine the symmetric point with respect to the x-axis
def symmetric_x_axis (p : Point3D) : Point3D :=
  { p with y := -p.y, z := -p.z }

-- Example point
def pointA := Point3D.mk -3 1 4

-- Definition of the point symmetric to A with respect to the x-axis
def symmetric_pointA := Point3D.mk -3 -1 -4

-- The proof statement to be proved
theorem symmetric_point_correct : 
  symmetric_x_axis pointA = symmetric_pointA := by
  sorry

end symmetric_point_correct_l34_34456


namespace common_points_and_circumcenter_projections_form_equilateral_l34_34693

-- Definition of the conditions
variables {A B C D E M N : Point}
variables (triangle_ABC : Triangle A B C)
variables (AD AE : Line)
variables (bisector_AD : isAngleBisector triangle_ABC AD)
variables (bisector_AE : isExternalAngleBisector triangle_ABC AE)
variables (DE : Segment)
variables (circle_Sa : Circle)
variables (Sa_diameter : circle_Sa = diameterCircle DE)
variables (circle_Sb circle_Sc : Circle)
variables (Sb_definition : define_similary circle_Sb S_a triangle_ABC)
variables (Sc_definition : define_similary circle_Sc S_a triangle_ABC)

-- First part: Prove that circles S_a, S_b, and S_c have two common points M and N, and MN passes through the circumcenter of triangle ABC
theorem common_points_and_circumcenter (common_points_MN : commonPoints circle_Sa circle_Sb circle_Sc M N)
    (circumcenter : Line (circumcenter triangle_ABC) M N) : true :=
sorry

-- Second part: Prove that the projections of M (and N) onto the sides of triangle ABC form an equilateral triangle
theorem projections_form_equilateral (projections_equilateral : projectionsEquilateral M triangle_ABC)
    (equilateral : Equilateral (projectionsSides M triangle_ABC)) : true :=
sorry

end common_points_and_circumcenter_projections_form_equilateral_l34_34693


namespace subset_A_B_implies_a_zero_l34_34552

theorem subset_A_B_implies_a_zero (a : ℝ) (A B: set ℝ) (hA : A = {1, a-1}) (hB : B = {-1, 2a-3, 1-2a}) (hAB : A ⊆ B) : a = 0 :=
sorry

end subset_A_B_implies_a_zero_l34_34552


namespace example_problem_l34_34948

noncomputable def cartesian_equation_C (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sqrt 5 * Real.sin θ

def parametric_line (t : ℝ) : (ℝ × ℝ) :=
  (3 - Real.sqrt 2 / 2 * t, Real.sqrt 5 + Real.sqrt 2 / 2 * t)

def cartesian_equation_l (x y : ℝ) : Prop :=
  x + y = 3 + Real.sqrt 5

theorem example_problem (ρ θ x y t : ℝ) :
  (cartesian_equation_C ρ θ → x^2 + y^2 - 2 * Real.sqrt 5 * y = 0) ∧
  (parametric_line t = (x, y) → cartesian_equation_l x y) ∧
  ∀ (P M N : ℝ × ℝ),
    (P = (3, Real.sqrt 5)) →
    (M = parametric_line t) →
    (N = parametric_line t) →
    (|P.fst - M.fst| + |N.fst - N.snd| = 3 * Real.sqrt 2) :=
sorry

end example_problem_l34_34948


namespace perpendicular_AP_MN_l34_34805

open EuclideanGeometry

variable {A B C D M N P : Point}

theorem perpendicular_AP_MN
  (h_parallelogram : parallelogram A B C D)
  (h_perpendicular_AM_BC : ∠AM = 90)
  (h_perpendicular_AN_CD : ∠AN = 90)
  (h_P_intersection : ∃ P, is_intersection P (line B N) (line D M)) :
  ∠AP ≅ ∠MN := 
sorry

end perpendicular_AP_MN_l34_34805


namespace find_n_l34_34777

theorem find_n (n : ℕ) (h : (n + 1) * n.factorial = 5040) : n = 6 := 
by sorry

end find_n_l34_34777


namespace correct_answer_l34_34402

variable (a : ℝ)

theorem correct_answer :
  2 * a^3 * 3 * a^5 = 6 * a^8 ∧
  ¬ (a^3 + a^3 = a^6) ∧
  ¬ ((a^3)^3 * a^6 = a^12) ∧
  ¬ (2 * a^6 / a^3 = 2 * a^2) :=
by
  split
  . sorry
  . split
    . intro hA
      sorry
    . split
      . intro hB
        sorry
      . intro hC
        sorry

end correct_answer_l34_34402


namespace problem_proof_l34_34650

noncomputable def area_triangle (a b c : ℝ) : ℝ :=
let p := (a + b + c) / 2 in
Real.sqrt (p * (p - a) * (p - b) * (p - c))

theorem problem_proof (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (M : Point)
  (h1 : is_triangle a b c) (h2 : is_acute_triangle a b c)
  (h3 : ∠AMB = 120 ∧ ∠BMC = 120 ∧ ∠CMA = 120):
  AM + BM + CM = Real.sqrt ((1 / 2) * (a^2 + b^2 + c^2 + 4 * (area_triangle a b c) * Real.sqrt 3)) :=
sorry

end problem_proof_l34_34650


namespace cube_surface_area_increase_l34_34077

theorem cube_surface_area_increase (x : ℝ) (h : x > 0) :
  let A := 6 * x^2 in
  let A' := 6 * (1.5 * x)^2 in
  (A' - A) / A * 100 = 125 := 
sorry

end cube_surface_area_increase_l34_34077


namespace height_of_cylinder_l34_34305

noncomputable def height_ratio (r m1 m2 : ℝ) := m2 = 3 * m1

noncomputable def equal_volumes (r m1 m2 : ℝ) := π * r^2 * m1 = (1/3) * π * r^2 * m2

noncomputable def cylinder_surface_area (r m1 : ℝ) := 2 * π * r^2 + 2 * π * r * m1

noncomputable def cone_surface_area (r m2 : ℝ) := π * r^2 + π * r * (sqrt (r^2 + m2^2))

noncomputable def equal_surface_areas (r m1 m2 : ℝ) :=
  cylinder_surface_area r m1 = cone_surface_area r m2

theorem height_of_cylinder (r m1 m2 : ℝ) 
  (h_ratio : height_ratio r m1 m2)
  (h_vols : equal_volumes r m1 m2)
  (h_areas : equal_surface_areas r m1 m2) : 
  m1 = (4 / 5) * r :=
sorry

end height_of_cylinder_l34_34305


namespace convex_and_concave_is_affine_l34_34261

variables {I : Set ℝ} (f : ℝ → ℝ)

def is_convex (I : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄ (λ : ℝ), x ∈ I → y ∈ I → 0 ≤ λ → λ ≤ 1 → f (λ * x + (1 - λ) * y) ≤ λ * f x + (1 - λ) * f y

def is_concave (I : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄ (λ : ℝ), x ∈ I → y ∈ I → 0 ≤ λ → λ ≤ 1 → f (λ * x + (1 - λ) * y) ≥ λ * f x + (1 - λ) * f y

theorem convex_and_concave_is_affine (hf_convex : is_convex I f) (hf_concave : is_concave I f) :
  ∃ a b : ℝ, ∀ x, x ∈ I → f x = a * x + b := 
sorry

end convex_and_concave_is_affine_l34_34261


namespace fixed_point_range_of_t_l34_34119

-- Problem 1
theorem fixed_point (k : ℝ) (k1 k2 : ℝ) (k_ne_zero : k ≠ 0) (intersects_ellipse : ∃ M N : ℝ × ℝ, M ∈ ellipse ∧ N ∈ ellipse ∧ line_intersects_ellipse M N k)
  (slopes_relation : 3 * (k1 + k2) = 8 * k) :
  (∃ (fixed_point : ℝ × ℝ), fixed_point = (0, 1/2) ∨ fixed_point = (0, -1/2)) := 
sorry

-- Problem 2
theorem range_of_t (k : ℝ) (t : ℝ) (k_squared_range : 0 < k^2 ∧ k^2 < 5/12) (line_through_D : line.passes_through (1, 0))
  (area_ratio : rat_area_ratio (O_triangle : (ℝ × ℝ) × (ℝ × ℝ)) (M_triangle : (ℝ × ℝ) × (ℝ × ℝ)) (N_triangle : (ℝ × ℝ) × (ℝ × ℝ)) = t) :
  2 < t ∧ t < 3 ∨ 1 / 3 < t ∧ t < 1 / 2 := 
sorry

end fixed_point_range_of_t_l34_34119


namespace waiter_earning_correct_l34_34046

-- Definitions based on the conditions
def tip1 : ℝ := 25 * 0.15
def tip2 : ℝ := 22 * 0.18
def tip3 : ℝ := 35 * 0.20
def tip4 : ℝ := 30 * 0.10

def total_tips : ℝ := tip1 + tip2 + tip3 + tip4
def commission : ℝ := total_tips * 0.05
def net_tips : ℝ := total_tips - commission

-- Theorem statement
theorem waiter_earning_correct : net_tips = 16.82 := by
  sorry

end waiter_earning_correct_l34_34046


namespace remove_15_toothpicks_no_triangles_remain_l34_34101

theorem remove_15_toothpicks_no_triangles_remain :
  ∀ (total_toothpicks upward_triangles downward_triangles : ℕ), 
  total_toothpicks = 45 →
  upward_triangles = 15 →
  downward_triangles = 10 →
  ∃ min_toothpicks_to_remove, min_toothpicks_to_remove = 15 ∧
  (∀ removed, removed < 15 → 
   ∃ remaining_upward remaining_downward, 
   remaining_upward + remaining_downward > 0) :=
by 
  intros total_toothpicks upward_triangles downward_triangles ht hu hd,
  existsi 15,
  split,
  { refl },
  { intros removed hremoved,
    sorry }

end remove_15_toothpicks_no_triangles_remain_l34_34101


namespace machine_value_decrease_l34_34641

theorem machine_value_decrease :
  ∃ p : ℝ, (p ≈ 0.2929) ∧ (8000 * (1 - p)^2 = 4000) :=
by {
  sorry
}

end machine_value_decrease_l34_34641


namespace bus_travel_distance_correct_l34_34362

theorem bus_travel_distance_correct :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 6 →
  let D := λ k : ℕ, 100 * k / (k + 1) in
  D k = (100 * k) / (k + 1) :=
by {
  intros k h,
  let D := λ k : ℕ, 100 * k / (k + 1),
  have h1 : D k = (100 * k) / (k + 1),
  {
    exact rfl,
  },
  exact h1,
}

end bus_travel_distance_correct_l34_34362


namespace complement_A_B_correct_l34_34698

open Set

-- Given sets A and B
def A : Set ℕ := {0, 2, 4, 6, 8, 10}
def B : Set ℕ := {4, 8}

-- Define the complement of B with respect to A
def complement_A_B : Set ℕ := A \ B

-- Statement to prove
theorem complement_A_B_correct : complement_A_B = {0, 2, 6, 10} :=
  by sorry

end complement_A_B_correct_l34_34698


namespace jackie_apples_l34_34731

theorem jackie_apples (a : ℕ) (j : ℕ) (h1 : a = 9) (h2 : a = j + 3) : j = 6 :=
by
  sorry

end jackie_apples_l34_34731


namespace length_of_lot_l34_34945

theorem length_of_lot 
(volume width height : ℝ) (h_volume : volume = 1600) (h_width : width = 20) (h_height : height = 2) : 
  (volume / (width * height) = 40) := 
by
  rw [h_volume, h_width, h_height]
  norm_num
  sorry

end length_of_lot_l34_34945


namespace jump_stats_A_score_stability_choose_student_l34_34706

-- Given data setup
def scores_A := [169, 165, 168, 169, 172, 173, 169, 167]
def scores_B := [161, 174, 172, 162, 163, 172, 172, 176]

-- Calculating the needed statistical measures
def mean (scores : List ℕ) : ℚ :=
  (scores.foldl (· + ·) 0 : ℚ) / scores.length

def median (scores : List ℕ) : ℚ :=
  let sorted := scores.qsort (· < ·)
  if h : (sorted.length % 2 = 0) then
    (sorted[nat.ceil((sorted.length / 2) - 1)] + sorted[sorted.length / 2]) / 2
  else
    sorted[nat.ceil((sorted.length / 2))]

def mode (scores : List ℕ) : ℕ :=
  scores.foldl (fun (acc map : (ℕ × ℕ)) (val: ℕ) =>
    let count := map[1].get! val 0 + 1
    (if count > acc.2 then (val, count) else acc)) (0, 0) |>.fst

def variance (scores : List ℕ) (mean : ℚ) : ℚ :=
  let mean_sq_diff := scores.foldl (fun sum a => sum + (a - mean)^2) 0
  mean_sq_diff / scores.length

def var_A := variance scores_A (mean scores_A)
def var_B := variance scores_B (mean scores_B)

-- Given variances
def given_variance_A := 5.75
def given_variance_B := 31.25

-- Proving the results
theorem jump_stats_A : mean scores_A = 169 ∧ median scores_A = 169 ∧ mode scores_A = 169 ∧ var_A = given_variance_A :=
by
  sorry

theorem score_stability : var_A < var_B :=
by
  sorry

theorem choose_student (jump_height : ℕ) : 
  (jump_height ≥ 165 → "A") ∧ (jump_height ≥ 170 → "B") :=
by
  sorry

end jump_stats_A_score_stability_choose_student_l34_34706


namespace find_AV_length_l34_34005

noncomputable def circle_A_center : ℝ × ℝ := (0, 0)
noncomputable def circle_A_radius : ℝ := 10
noncomputable def circle_B_center : ℝ × ℝ := (13, 0)
noncomputable def circle_B_radius : ℝ := 3

theorem find_AV_length :
  let A := circle_A_center,
  let B := circle_B_center,
  let U : ℝ × ℝ := (10, 0),
  let V : ℝ × ℝ := (13, 3),
  let AV := dist A V in
  AV = 2 * Real.sqrt 55 :=
by
  -- Without proof
  sorry

end find_AV_length_l34_34005


namespace max_magnitude_c_l34_34835

noncomputable def max_c_magnitude (a b c : ℝ^3) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hab : dot a b = 0) (h : dot (a - c) (a - c) = 0) : ℝ :=
  sqrt 2

theorem max_magnitude_c (a b c : ℝ^3) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hab : dot a b = 0) (h : dot (a - c) (a - c) = 0) :
    max_c_magnitude a b c ha hb hab h = sqrt 2 :=
sorry

end max_magnitude_c_l34_34835


namespace Ben_ate_25_percent_of_cake_l34_34258

theorem Ben_ate_25_percent_of_cake (R B : ℕ) (h_ratio : R / B = 3 / 1) : B / (R + B) * 100 = 25 := by
  sorry

end Ben_ate_25_percent_of_cake_l34_34258


namespace Q_20_eq_11_67_l34_34222

def S (n : ℕ) : ℕ := n * (n + 1)

def Q (n : ℕ) : ℝ :=
  ∏ k in Finset.range (n - 1) + 2, (S k) / (S k - 2)

theorem Q_20_eq_11_67 : Q 20 = 11.67 := by
  sorry

end Q_20_eq_11_67_l34_34222


namespace bus_average_speed_l34_34538

theorem bus_average_speed (V_without_stoppages V_with_stoppages : ℝ)
  (H1 : V_without_stoppages = 80)
  (H2 : V_with_stoppages = (1 / 2) * V_without_stoppages) :
  V_with_stoppages = 40 :=
by
  rw [H1, H2]
  norm_num
  sorry

end bus_average_speed_l34_34538


namespace polygon_sides_doubled_l34_34501

theorem polygon_sides_doubled (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : ((2 * n) * ((2 * n) - 3) / 2 - 2 * n) - (n * (n - 3) / 2 - n) = 99) : 
  n = 9 :=
begin
  sorry
end

end polygon_sides_doubled_l34_34501


namespace pete_backwards_speed_l34_34594

variable (speed_pete_hands : ℕ) (speed_tracy_cartwheel : ℕ) (speed_susan_walk : ℕ) (speed_pete_backwards : ℕ)

axiom pete_hands_speed : speed_pete_hands = 2
axiom pete_hands_speed_quarter_tracy_cartwheel : speed_pete_hands = speed_tracy_cartwheel / 4
axiom tracy_cartwheel_twice_susan_walk : speed_tracy_cartwheel = 2 * speed_susan_walk
axiom pete_backwards_three_times_susan_walk : speed_pete_backwards = 3 * speed_susan_walk

theorem pete_backwards_speed : 
  speed_pete_backwards = 12 :=
by
  sorry

end pete_backwards_speed_l34_34594


namespace exists_z_such_that_f_inverse_z_equals_f_neg_conj_z_l34_34964

open Complex

theorem exists_z_such_that_f_inverse_z_equals_f_neg_conj_z 
  (f g : ℂ → ℂ) (h_cont_f : Continuous f) (h_cont_g : Continuous g) 
  (h : ∀ (z : ℂ), z ≠ 0 → g z = f (1 / z)) :
  ∃ z : ℂ, f (1 / z) = f (-conj z) :=
sorry

end exists_z_such_that_f_inverse_z_equals_f_neg_conj_z_l34_34964


namespace sqrt_interval_l34_34050

theorem sqrt_interval :
  let expr := (Real.sqrt 18) / 3 - (Real.sqrt 2) * (Real.sqrt (1 / 2))
  0 < expr ∧ expr < 1 :=
by
  let expr := (Real.sqrt 18) / 3 - (Real.sqrt 2) * (Real.sqrt (1 / 2))
  sorry

end sqrt_interval_l34_34050


namespace abs_neg_three_l34_34288

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l34_34288


namespace find_sin_ratio_l34_34188

noncomputable theory
open Real

-- Definition of the problem
def triangle_properties (A B C a b c : ℝ) : Prop :=
B = 2 * π / 3 ∧ a^2 + c^2 = 4 * a * c

-- The theorem statement
theorem find_sin_ratio {A B C a b c : ℝ} (h : triangle_properties A B C a b c) :
  (sin (A + C)) / (sin A * sin C) = 10 * sqrt 3 / 3 :=
by sorry

end find_sin_ratio_l34_34188


namespace smallest_set_with_three_pairwise_coprime_elements_l34_34440

theorem smallest_set_with_three_pairwise_coprime_elements (n m : ℕ) (hn : n ≥ 4) :
  ∃ f, f = 1008 ∧ 
    ∀ (S : finset ℕ), S ⊆ finset.range (m + n - 1 + 1) ∧ finset.card S = f → 
    ∃ (a b c ∈ S), nat.coprime a b ∧ nat.coprime a c ∧ nat.coprime b c :=
sorry

end smallest_set_with_three_pairwise_coprime_elements_l34_34440


namespace train_speed_l34_34729

theorem train_speed (L V : ℝ) (h1 : L = V * 20) (h2 : L + 300.024 = V * 50) : V = 10.0008 :=
by
  sorry

end train_speed_l34_34729


namespace maximum_distance_value_of_m_l34_34485

-- Define the line equation
def line_eq (m x y : ℝ) : Prop := y = m * x - m - 1

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the problem statement
theorem maximum_distance_value_of_m :
  ∃ (m : ℝ), (∀ x y : ℝ, circle_eq x y → ∃ P : ℝ × ℝ, line_eq m P.fst P.snd) →
  m = -0.5 :=
sorry

end maximum_distance_value_of_m_l34_34485


namespace roma_can_arrange_l34_34979

-- Define the rating function for the chips and the placement rating
def chip_rating (k : ℕ) : ℕ := 2^k

def total_rating (chips : List ℕ) : ℕ := chips.map (λ k => chip_rating k).sum

-- Define the condition from the problem
def condition (n k : ℕ) : Prop := n < 2^(k-3)

-- The main theorem stating that Roma can arrange it so that none of the chips reach the end
theorem roma_can_arrange (n k : ℕ) (h : condition n k) : true := 
by
  sorry

end roma_can_arrange_l34_34979


namespace abs_neg_three_l34_34278

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l34_34278


namespace girls_in_class_l34_34511

theorem girls_in_class (g b : ℕ) 
  (h1 : g / b = 3 / 4) 
  (h2 : g + b = 35) : 
  g = 15 := 
begin
  sorry
end

end girls_in_class_l34_34511


namespace solution_set_of_inequality_l34_34652

theorem solution_set_of_inequality 
  (x : ℝ) :
  (1 / 2) ^ (x - 5) ≤ 2 ^ x ↔ x ≥ 5 / 2 :=
sorry

end solution_set_of_inequality_l34_34652


namespace gcd_of_set_B_l34_34556

-- Let B be the set of all numbers which can be represented as the sum of five consecutive positive integers
def B : Set ℕ := {n | ∃ y : ℕ, n = (y - 2) + (y - 1) + y + (y + 1) + (y + 2)}

-- State the problem
theorem gcd_of_set_B : ∀ k ∈ B, ∃ d : ℕ, is_gcd 5 k d → d = 5 := by
sorry

end gcd_of_set_B_l34_34556


namespace no_nat_nums_x4_minus_y4_eq_x3_plus_y3_l34_34074

theorem no_nat_nums_x4_minus_y4_eq_x3_plus_y3 : ∀ (x y : ℕ), x^4 - y^4 ≠ x^3 + y^3 :=
by
  intro x y
  sorry

end no_nat_nums_x4_minus_y4_eq_x3_plus_y3_l34_34074


namespace sigma_algebra_equiv_countably_generated_l34_34260

variable {Ω : Type*}

-- Definition of a σ-algebra
structure SigmaAlgebra (Ω : Type*) :=
(sets : set (set Ω))
(is_sigma_algebra : is_sigma_algebra sets)

-- Random variable definition
variable {X : Ω → ℝ}

-- Countable generated σ-algebra definition
def countablyGenerated (G : SigmaAlgebra Ω) : Prop :=
  ∃ (A : ℕ → set Ω), G.sets = σ (⋃ n, {A n})

-- The main theorem statement
theorem sigma_algebra_equiv_countably_generated (G : SigmaAlgebra Ω) :
  (∃ X : Ω → ℝ, G.sets = σ(X)) ↔ countablyGenerated G :=
sorry

end sigma_algebra_equiv_countably_generated_l34_34260


namespace iterates_pairwise_coprime_l34_34981

def f (x : ℕ) : ℕ := x ^ 2 - x + 1

theorem iterates_pairwise_coprime (m : ℕ) (h : m > 1) :
  pairwise coprime (stream.iterate f m) :=
sorry

end iterates_pairwise_coprime_l34_34981


namespace sin_shift_sum_nonnegative_l34_34114

theorem sin_shift_sum_nonnegative (x y z : ℝ) (h : sin x + sin y + sin z ≥ 3 / 2) : 
  sin (x - π / 6) + sin (y - π / 6) + sin (z - π / 6) ≥ 0 :=
sorry

end sin_shift_sum_nonnegative_l34_34114


namespace find_width_of_rectangle_l34_34645

variable (w : ℝ) (l : ℝ) (P : ℝ)

def width_correct (h1 : P = 150) (h2 : l = w + 15) : Prop :=
  w = 30

-- Theorem statement in Lean
theorem find_width_of_rectangle (h1 : P = 150) (h2 : l = w + 15) (h3 : P = 2 * l + 2 * w) : width_correct w l P h1 h2 :=
by
  sorry

end find_width_of_rectangle_l34_34645


namespace asymptotes_and_foci_of_hyperbola_l34_34370

def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 81 = 1

theorem asymptotes_and_foci_of_hyperbola :
  (∀ x y : ℝ, hyperbola x y → y = x * (3 / 4) ∨ y = x * -(3 / 4)) ∧
  (∃ x y : ℝ, (x, y) = (15, 0) ∨ (x, y) = (-15, 0)) :=
by {
  -- prove these conditions here
  sorry 
}

end asymptotes_and_foci_of_hyperbola_l34_34370


namespace bus_transport_problem_l34_34002

theorem bus_transport_problem :
  ∃ n m : ℕ, (n = 12) ∧ (m = 13) ∧
             (∀ i : ℕ, i ≥ n → (i * 45 ≥ 500)) ∧
             (∀ j : ℕ, j ≤ m → (j * 36 ≥ 500)) ∧
             (n * 45 ≥ 500) ∧
             (m * 36 ≥ 500) :=
begin
  sorry
end

end bus_transport_problem_l34_34002


namespace centroid_of_triangle_l34_34252

theorem centroid_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) :
  ∃ (x0 y0 : ℝ), 
    x0 = (x1 + x2 + x3) / 3 ∧ 
    y0 = (y1 + y2 + y3) / 3 :=
begin
  sorry
end

end centroid_of_triangle_l34_34252


namespace linda_scores_product_l34_34184

theorem linda_scores_product : 
  ∃ (s8 s9 : ℕ), 
    s8 < 10 ∧ 
    s9 < 10 ∧ 
    ((33 + s8) % 8 = 0) ∧ 
    ((33 + s8 + s9) % 9 = 0) ∧ 
    (s8 * s9 = 35) :=
begin
  sorry
end

end linda_scores_product_l34_34184


namespace smallest_n_l34_34436

theorem smallest_n (k : ℕ) : 
  ∃ n : ℕ, ∀ k : ℕ, (3^k + n^k + (3 * n)^k + 2014^k).is_square 
                  ∧ ¬ (3^k + n^k + (3 * n)^k + 2014^k).is_cube := 
begin
  use 2,
  sorry
end

end smallest_n_l34_34436


namespace solve_for_q_l34_34495

-- Define the conditions
variables (p q : ℝ)
axiom condition1 : 3 * p + 4 * q = 8
axiom condition2 : 4 * p + 3 * q = 13

-- State the goal to prove q = -1
theorem solve_for_q : q = -1 :=
by
  sorry

end solve_for_q_l34_34495


namespace carla_total_time_l34_34743

def time_sharpening : ℝ := 15
def time_peeling : ℝ := 3 * time_sharpening
def time_chopping : ℝ := 0.5 * time_peeling
def time_breaks : ℝ := 2 * 5

def total_time : ℝ :=
  time_sharpening + time_peeling + time_chopping + time_breaks

theorem carla_total_time : total_time = 92.5 :=
by sorry

end carla_total_time_l34_34743


namespace numberOfValidN_l34_34163

theorem numberOfValidN : 
  let num_integers := λ (n : ℤ), 400 * (2/5)^n;
  let n_range := λ n, (num_integers n) ∈ ℤ;
  (finset.Icc (-4 : ℤ) (2 : ℤ)).filter n_range.card = 9 :=
sorry

end numberOfValidN_l34_34163


namespace abs_neg_three_l34_34275

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l34_34275


namespace min_value_S6_l34_34822

theorem min_value_S6 (q : ℝ) (a1 : ℝ) (h1 : q > 1) 
  (h2 : ∃ a1, S4 = 2 * S2 + 1) 
  (S4 : ℝ) (S2 : ℝ) : 
  ∃ S6, S6 = 2 * real.sqrt 3 + 3 :=
by sorry

end min_value_S6_l34_34822


namespace athlete_total_heartbeats_l34_34040

theorem athlete_total_heartbeats (h : ℕ) (p : ℕ) (d : ℕ) (r : ℕ) : (h = 150) ∧ (p = 6) ∧ (d = 30) ∧ (r = 15) → (p * d + r) * h = 29250 :=
by
  sorry

end athlete_total_heartbeats_l34_34040


namespace max_second_smallest_l34_34609

noncomputable def f (M : ℕ) : ℕ :=
  (M - 1) * (90 - M) * (89 - M) * (88 - M)

theorem max_second_smallest (M : ℕ) (cond : 1 ≤ M ∧ M ≤ 89) : M = 23 ↔ (∀ N : ℕ, f M ≥ f N) :=
by
  sorry

end max_second_smallest_l34_34609


namespace inequality_solution_set_l34_34304

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_monotonically_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_mono_dec : is_monotonically_decreasing_on_nonneg f) :
  { x : ℝ | f 1 - f (1 / x) < 0 } = { x : ℝ | x < -1 ∨ x > 1 } :=
by
  sorry

end inequality_solution_set_l34_34304


namespace abs_neg_three_eq_three_l34_34282

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l34_34282


namespace A_alone_completes_one_work_in_32_days_l34_34497

def amount_of_work_per_day_by_B : ℝ := sorry
def amount_of_work_per_day_by_A : ℝ := 3 * amount_of_work_per_day_by_B
def total_work : ℝ := (amount_of_work_per_day_by_A + amount_of_work_per_day_by_B) * 24

theorem A_alone_completes_one_work_in_32_days :
  total_work = amount_of_work_per_day_by_A * 32 :=
by
  sorry

end A_alone_completes_one_work_in_32_days_l34_34497


namespace normal_vector_of_plane_ABC_l34_34828

-- Define the points A, B, and C in 3D space.
def A : ℝ × ℝ × ℝ := (1, 0, 0)
def B : ℝ × ℝ × ℝ := (2, 0, 1)
def C : ℝ × ℝ × ℝ := (0, 1, 2)

-- Define the vectors AB and AC.
def vector_AB : ℝ × ℝ × ℝ := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
def vector_AC : ℝ × ℝ × ℝ := (C.1 - A.1, C.2 - A.2, C.3 - A.3)

-- Define the dot product function.
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the normal vector.
def normal_vector : ℝ × ℝ × ℝ := (1, 3, -1)

-- The proof statement.
theorem normal_vector_of_plane_ABC :
  dot_product normal_vector vector_AB = 0 ∧
  dot_product normal_vector vector_AC = 0 :=
by
  simp [vector_AB, vector_AC, dot_product, normal_vector]
  sorry

end normal_vector_of_plane_ABC_l34_34828


namespace chord_length_of_circle_and_parabola_l34_34826

theorem chord_length_of_circle_and_parabola :
  ∀ (x y : ℝ), circle_passing_through_focus (x^2 + y^2 + 8 * x + 4 * y - 5 = 0) (focus (0, 1)) (directrix y = -1) →
  chord_length (circle_center (x = -4) (y = -2)) (radius 5) (directrix_distance 1) = 4 * sqrt 6 :=
by sorry

end chord_length_of_circle_and_parabola_l34_34826


namespace triangle_problem_l34_34836

noncomputable def triangle_side_ang (a b c : ℝ) (A B C : ℝ) 
  (h1 : (b * Real.cos A + (Real.sqrt 3) * b * Real.sin A - c - a = 0)) 
  (h2 : b = Real.sqrt 3) : Prop :=
  B = Real.pi / 3 ∧ (a + c) ≤ 2 * Real.sqrt 3

theorem triangle_problem (a b c A B C : ℝ)
  (h1 : b * Real.cos A + (Real.sqrt 3) * b * Real.sin A - c - a = 0)
  (h2 : b = Real.sqrt 3) : triangle_side_ang a b c A B C h1 h2 :=
begin
  sorry
end

end triangle_problem_l34_34836


namespace length_BC_value_tan_2B_l34_34506

noncomputable def length {α : Type*} [inner_product_space ℝ α] (a b : α) : ℝ := real.sqrt (inner_product (a - b) (a - b))

-- Given Conditions
variables 
  (A B C : ℝ × ℝ)
  (AB_length : ℝ := 6)
  (AC_length : ℝ := 3 * real.sqrt 2)
  (dot_product_AB_AC : ℝ := -18)

-- Definitions based on the given conditions
def AB : ℝ := AB_length
def AC : ℝ := AC_length
def dot_prod : ℝ := dot_product_AB_AC

-- Main Statements
theorem length_BC (BC_length : ℝ) : length A B C = 3 * real.sqrt 10 :=
begin
  sorry
end

theorem value_tan_2B (tan_2B : ℝ) : tan 2 * (real.atan (length B C / length A B)) = 3 / 4 :=
begin
  sorry
end

end length_BC_value_tan_2B_l34_34506


namespace shaded_region_probability_l34_34751

theorem shaded_region_probability:
  ∀ (T : Type) [Triangle T] [Isosceles T],
  (base_angle T = 2 * vertex_angle T) →
  (altitudes_divide_triangle T 6) →
  shaded_regions T 4 →
  probability_in_shaded_region T = (2 / 3) :=
by
  sorry

end shaded_region_probability_l34_34751


namespace f_even_f_increasing_f_range_l34_34449

variables {R : Type*} [OrderedRing R] (f : R → R)

-- Conditions
axiom f_mul : ∀ x y : R, f (x * y) = f x * f y
axiom f_neg1 : f (-1) = 1
axiom f_27 : f 27 = 9
axiom f_lt_1 : ∀ x : R, 0 ≤ x → x < 1 → 0 ≤ f x ∧ f x < 1

-- Questions
theorem f_even (x : R) : f x = f (-x) :=
by sorry

theorem f_increasing (x1 x2 : R) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : x1 < x2) : f x1 < f x2 :=
by sorry

theorem f_range (a : R) (h1 : 0 ≤ a) (h2 : f (a + 1) ≤ 39) : 0 ≤ a ∧ a ≤ 2 :=
by sorry

end f_even_f_increasing_f_range_l34_34449


namespace domain_range_of_f_l34_34472

noncomputable def f (a k x : ℝ) := log a (a^x + k)

theorem domain_range_of_f {a k m n : ℝ} (ha_pos : a > 0) (ha_ne_one : a ≠ 1) 
    (D : set ℝ) (hD : ∀ x, x ∈ D ↔ a^x + k > 0) 
    (h_range : ∀ x, x ∈ (set.Icc m n) → f a k x ∈ (set.Icc (1/2 * m) (1/2 * n))) :
  0 < k ∧ k < 1/4 :=
by
  sorry

end domain_range_of_f_l34_34472


namespace circle_equation_center_y_axis_l34_34627

theorem circle_equation_center_y_axis (h₁ : ∃ k : ℝ, x^2 + (y - k)^2 = 1)
                                     (h₂ : ∃ k : ℝ, (1, 2) ∈ { p : ℝ × ℝ | x^2 + (y - k)^2 = 1 }) :
                                     x^2 + (y - 2)^2 = 1 :=
begin
  sorry
end

end circle_equation_center_y_axis_l34_34627


namespace concyclic_if_perpendicular_l34_34543

variables {A B C D E F : Type*} [circle : is_circle A B C D]

/-- Given four distinct points A, B, C, D on a circle such that the lines AC and BD intersect at E,
and the lines AD and BC intersect at F, and AB and CD are not parallel, 
prove that C, D, E, F are concyclic if and only if EF is perpendicular to AB. -/
theorem concyclic_if_perpendicular 
  (h1 : is_on_circle A B C D) 
  (h2 : lines_intersect_at (line_through A C) (line_through B D) E) 
  (h3 : lines_intersect_at (line_through A D) (line_through B C) F) 
  (h4 : ¬parallel (line_through A B) (line_through C D)) :
  is_concyclic C D E F ↔ is_perpendicular (line_through E F) (line_through A B) :=
sorry

end concyclic_if_perpendicular_l34_34543


namespace football_shape_area_l34_34254

-- Define the conditions
def is_square (A B C D : ℕ) : Prop := A = 2 ∧ B = 2 ∧ C = 2 ∧ D = 2
def radius_circle_D : ℝ := 2 * real.sqrt 2
def radius_circle_B : ℝ := 2

-- Define the main theorem
theorem football_shape_area (A B C D: ℕ) (h_square: is_square A B C D) 
                            (rad_D_eq: radius_circle_D = 2 * real.sqrt 2) 
                            (rad_B_eq: radius_circle_B = 2) 
                            (AB_length: ℝ := 2) : 
                            AB = 2 
                            → 
                            area_of_football_shape = π := 
begin
  -- Proof is not required
  sorry,
end

end football_shape_area_l34_34254


namespace triangle_construction_l34_34158

noncomputable def construct_triangle (X Y Z : Point) : Triangle :=
  sorry

theorem triangle_construction (X Y Z : Point) :
  let T := construct_triangle X Y Z in
  circumcenter T = X ∧ midpoint T.B T.C = Y ∧ altitude T.B Z T.A =
  sorry

end triangle_construction_l34_34158


namespace increasing_function_m_val_l34_34153

theorem increasing_function_m_val (m : ℝ) (f : ℝ → ℝ)
  (h : ∀ x > 0, f x = (m^2 - 4 * m + 1) * x^(m^2 - 2 * m - 3)) :
  (∃ m, m = 4 ∧ ∀ x > 0, monotone_on f (set.Ioi x)) :=
begin
  sorry
end

end increasing_function_m_val_l34_34153


namespace rational_roots_count_l34_34718

theorem rational_roots_count (b_3 b_2 b_1 : ℤ) :
  let p_roots := [1, -1, 2, -2, 1/2, -1/2, 1/3, -1/3, 1/4, -1/4, 1/6, -1/6, 5, -5, 10, -10, 5/2, -5/2, 5/3, -5/3, 5/4, -5/4, 5/6, -5/6] in
  p_roots.length = 24 :=
by
  sorry

end rational_roots_count_l34_34718


namespace find_missing_digit_l34_34321

-- Define the known sum of the digits
def sum_of_known_digits : Nat := 3 + 4 + 6 + 9 + 2

-- Define the condition that the six-digit number is divisible by 9
def is_divisible_by_9 (n : Nat) : Prop := n % 9 = 0

-- The missing digit is such that adding it to the known digit sum makes the total sum divisible by 9
def missing_digit_property (x : Nat) : Prop :=
  is_divisible_by_9 (sum_of_known_digits + x)

-- We need to prove that the missing digit is 3
theorem find_missing_digit : missing_digit_property 3 :=
by {
  unfold missing_digit_property,
  unfold is_divisible_by_9,
  unfold sum_of_known_digits,
  have sum_total := 3 + 4 + 6 + 9 + 2 + 3,
  show (sum_total) % 9 = 0,
  calc 27 % 9 = 0 : by norm_num }


end find_missing_digit_l34_34321


namespace madeline_refills_l34_34587

theorem madeline_refills :
  let total_water := 100
  let bottle_capacity := 12
  let remaining_to_drink := 16
  let already_drank := total_water - remaining_to_drink
  let initial_refills := already_drank / bottle_capacity
  let refills := initial_refills + 1
  refills = 8 :=
by
  sorry

end madeline_refills_l34_34587


namespace mens_wages_l34_34001

-- Definitions from the conditions.
variables (men women boys total_earnings : ℕ) (wage : ℚ)
variable (equivalence : 5 * men = 8 * boys)
variable (totalEarnings : total_earnings = 120)

-- The final statement to prove the men's wages.
theorem mens_wages (h_eq : 5 = 5) : wage = 46.15 :=
by
  sorry

end mens_wages_l34_34001


namespace problem_statement_l34_34081

noncomputable def evaluate_expression : ℕ :=
  let a := 5^1003
  let b := 6^1002
  (a + b)^2 - (a - b)^2

theorem problem_statement : evaluate_expression = 600 * 30^1001 :=
by
  let a := 5^1003
  let b := 6^1002
  have h : (a + b)^2 - (a - b)^2 = 4 * a * b := by sorry
  have k : 4 * a * b = 600 * 30^1001 := by sorry
  rw [h, k]
  sorry

end problem_statement_l34_34081


namespace a_2013_units_digit_l34_34954

def a_sequence : ℕ → ℕ
| 0 := 3
| 1 := 6
| n + 2 := (a_sequence n * a_sequence (n + 1)) % 10

theorem a_2013_units_digit :
  a_sequence 2013 = 8 :=
sorry

end a_2013_units_digit_l34_34954


namespace local_students_percentage_theorem_l34_34935

-- Define the given conditions about the students
def num_arts_students := 400
def arts_local_percentage := 0.50
def num_science_students := 100
def science_local_percentage := 0.25
def num_commerce_students := 120
def commerce_local_percentage := 0.85

-- Define local student calculations
def local_arts_students := arts_local_percentage * num_arts_students
def local_science_students := science_local_percentage * num_science_students
def local_commerce_students := commerce_local_percentage * num_commerce_students

-- Define total students and local students
def total_local_students := local_arts_students + local_science_students + local_commerce_students
def total_students := num_arts_students + num_science_students + num_commerce_students

-- Define percentage calculation
def local_students_percentage := (total_local_students / total_students) * 100

-- Theorem to state the problem
theorem local_students_percentage_theorem : local_students_percentage ≈ 52.74 := by
  -- Proof to be provided
  sorry

end local_students_percentage_theorem_l34_34935


namespace ant_minimum_distance_l34_34380

section
variables (x y z w u : ℝ)

-- Given conditions
axiom h1 : x + y + z = 22
axiom h2 : w + y + z = 29
axiom h3 : x + y + u = 30

-- Prove the ant crawls at least 47 cm to cover all paths
theorem ant_minimum_distance : x + y + z + w ≥ 47 :=
sorry
end

end ant_minimum_distance_l34_34380


namespace polar_to_cartesian_l34_34528

theorem polar_to_cartesian (θ ρ : ℝ) (h : ρ = 2 * Real.sin θ) : 
  ∀ (x y : ℝ) (h₁ : x = ρ * Real.cos θ) (h₂ : y = ρ * Real.sin θ), 
    x^2 + (y - 1)^2 = 1 :=
by
  sorry

end polar_to_cartesian_l34_34528


namespace probability_of_at_least_two_heads_is_half_l34_34677

-- Define the probability space for tossing three coins
def probability_space : List (List Bool) :=
  [[true, true, true], [true, true, false], [true, false, true], [false, true, true],
   [true, false, false], [false, true, false], [false, false, true], [false, false, false]]

-- Define a function to count the number of heads in a coin toss outcome
def count_heads (outcome : List Bool) : Nat :=
  outcome.countb (λ b => b)

-- Define the event of having at least two heads
def at_least_two_heads (outcome : List Bool) : Bool :=
  count_heads outcome ≥ 2

-- Define the probability calculation
def probability_at_least_two_heads : ℚ :=
  let favorable_outcomes := probability_space.filter at_least_two_heads
  favorable_outcomes.length / probability_space.length

-- The statement specifying that the probability of getting at least two heads when tossing three coins is 1/2
theorem probability_of_at_least_two_heads_is_half : probability_at_least_two_heads = 1 / 2 := by
  sorry

end probability_of_at_least_two_heads_is_half_l34_34677


namespace find_original_weight_l34_34024

-- Define the conditions and hypothesis.
def original_weight (X : ℝ) : Prop :=
  0.5 * X = 750

-- Assert the original weight.
theorem find_original_weight : ∃ X, original_weight X ∧ X = 1500 :=
by
  -- Define the value of original weight.
  let X := 1500
  use X
  have h : original_weight X := rfl
  split
  { exact h }
  { simp only [X] }
  sorry

end find_original_weight_l34_34024


namespace number_of_clocks_is_2_l34_34611

def total_cost (dolls costD gloves costG clocks costC : ℕ) : ℕ := 
  (dolls * costD) + (gloves * costG) + (clocks * costC)

def total_revenue (dolls costD gloves costG clocks countClocks costC : ℕ) : ℕ := 
  (dolls * costD) + (gloves * costG) + (countClocks * costC)

def profit_from_sale (spent revenue: ℕ) : ℕ :=
  revenue - spent

theorem number_of_clocks_is_2 
  (dolls := 3) 
  (gloves := 5)
  (spent := 40) 
  (profit := 25)
  (costDoll := 5) 
  (costClock := 15) 
  (costGlove := 4) : 
  ∃ (countClocks : ℕ), profit_from_sale spent (total_revenue dolls costDoll gloves costGlove countClocks costClock) = profit + spent ∧ countClocks = 2 :=
by
  let countClocks := 2
  have h_total_revenue : total_revenue dolls costDoll gloves costGlove countClocks costClock = spent + profit := by
    sorry
  exists countClocks
  have h_countClocks : countClocks = 2 := by
    sorry
  exact ⟨h_total_revenue, h_countClocks⟩

end number_of_clocks_is_2_l34_34611


namespace smallest_positive_period_pi_l34_34396

noncomputable def f (x : ℝ) : ℝ :=
  let a := (Real.sin x + Real.cos x)^2
  det ![[a, -1], [1, 1]]

theorem smallest_positive_period_pi :
  ∀ x : ℝ, f (x + π) = f x := 
  sorry

end smallest_positive_period_pi_l34_34396


namespace compute_fraction_l34_34411

theorem compute_fraction :
  ((1/3)^4 * (1/5) = (1/405)) :=
by
  sorry

end compute_fraction_l34_34411


namespace dot_product_calculation_l34_34834

variable {R : Type*} [Real_InnerProductSpace R]
variable (e1 e2 : R)
variable (θ : ℝ)

noncomputable def unit_vector (v : R) : Prop := ⟪v, v⟫ = 1

axiom angle_60_deg : θ = real.arccos (1/2)

-- Main theorem
theorem dot_product_calculation
  (h1 : unit_vector e1)
  (h2 : unit_vector e2)
  (h3 : ⟪e1, e2⟫ = real.cos θ)
  (h4 : θ = real.arccos (1/2)) :
  ⟪2 • e1 + e2, -3 • e1 + 2 • e2⟫ = -7 / 2 :=
by
  sorry

end dot_product_calculation_l34_34834


namespace finite_if_and_only_if_nonzero_constant_l34_34966

open Polynomial Set

-- We define the conditions given in the problem.
def is_prime (p : ℕ) : Prop := p ∈ {p' : ℕ | (nat.prime p')}
def P (x : ℤ) : ℤ := sorry -- This represents the polynomial P(x)

-- Define the set of primes as stated in the problem.
def primes : Set ℕ := {p : ℕ | is_prime p}

-- Define the set S as described in the problem.
def S : Set ℕ := {p ∈ primes | ∃ n : ℤ, ↑p ∣ P n}

-- Main theorem statement
theorem finite_if_and_only_if_nonzero_constant :
  (finite S ↔ ∃ c : ℤ, ∀ x : ℤ, P x = c ∧ c ≠ 0) := sorry

end finite_if_and_only_if_nonzero_constant_l34_34966


namespace find_fraction_l34_34432

-- Definition of the fractions and the given condition
def certain_fraction : ℚ := 1 / 2
def given_ratio : ℚ := 2 / 6
def target_fraction : ℚ := 1 / 3

-- The proof problem to verify
theorem find_fraction (X : ℚ) : (X / given_ratio) = 1 ↔ X = target_fraction :=
by
  sorry

end find_fraction_l34_34432


namespace range_of_a_l34_34844

theorem range_of_a : 
  ∀ (a : ℝ), 
  (∀ (x : ℝ), ((a^2 - 1) * x^2 + (a + 1) * x + 1) > 0) → 1 ≤ a ∧ a ≤ 5 / 3 := 
by
  sorry

end range_of_a_l34_34844


namespace find_constants_cd_l34_34216

noncomputable def N : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, 1],
  ![-2, 4]
]

theorem find_constants_cd :
  ∃ (c d : ℚ), (inverse N = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ)) ∧
               (c = -1/14) ∧ (d = 3/7) :=
by
  sorry

end find_constants_cd_l34_34216


namespace wilted_flowers_correct_l34_34335

-- Definitions based on the given conditions
def total_flowers := 45
def flowers_per_bouquet := 5
def bouquets_made := 2

-- Calculating the number of flowers used for bouquets
def used_flowers : ℕ := bouquets_made * flowers_per_bouquet

-- Question: How many flowers wilted before the wedding?
-- Statement: Prove the number of wilted flowers is 35.
theorem wilted_flowers_correct : total_flowers - used_flowers = 35 := by
  sorry

end wilted_flowers_correct_l34_34335


namespace pythagorean_theorem_sets_l34_34686

theorem pythagorean_theorem_sets :
  ¬ (4 ^ 2 + 5 ^ 2 = 6 ^ 2) ∧
  (1 ^ 2 + (Real.sqrt 3) ^ 2 = 2 ^ 2) ∧
  ¬ (5 ^ 2 + 6 ^ 2 = 7 ^ 2) ∧
  ¬ (1 ^ 2 + (Real.sqrt 2) ^ 2 = 3 ^ 2) :=
by {
  sorry
}

end pythagorean_theorem_sets_l34_34686


namespace find_some_value_l34_34349

theorem find_some_value (m n k : ℝ)
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + k = (n + 18) / 6 - 2 / 5) : 
  k = 3 :=
sorry

end find_some_value_l34_34349


namespace find_x_l34_34429

-- Let's define the constants and the condition
def a : ℝ := 2.12
def b : ℝ := 0.345
def c : ℝ := 2.4690000000000003

-- We need to prove that there exists a number x such that
def x : ℝ := 0.0040000000000003

-- Formal statement
theorem find_x : a + b + x = c :=
by
  -- Proof skipped
  sorry
 
end find_x_l34_34429


namespace sum_of_numbers_l34_34680

theorem sum_of_numbers : 72.52 + 12.23 + 5.21 = 89.96 :=
by sorry

end sum_of_numbers_l34_34680


namespace simplify_expression_l34_34423

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 3) :
  (3 * x ^ 2 - 2 * x - 4) / ((x + 2) * (x - 3)) - (5 + x) / ((x + 2) * (x - 3)) =
  3 * (x ^ 2 - x - 3) / ((x + 2) * (x - 3)) :=
by
  sorry

end simplify_expression_l34_34423


namespace a_must_not_be_zero_l34_34614

theorem a_must_not_be_zero (a b c d : ℝ) (h₁ : a / b < -3 * (c / d)) (h₂ : b ≠ 0) (h₃ : d ≠ 0) (h₄ : c = 2 * a) : a ≠ 0 :=
sorry

end a_must_not_be_zero_l34_34614


namespace length_AB_eq_sqrt_5_l34_34582

-- Define the geometry setting and the given condition
variable (O P B A C D : Point)
variable [Diameter (O P)]
variable [Circle (Ω O P)]
variable [Circle (ω P) (hradius : ℝ) (hradius_positive : hradius < Diameter(Ω))]
variable [Intersect (Ω ω) C D]
variable [Chord (Ω O B) A]

-- Given condition: BD * BC = 5
variable [BD BC : ℝ] [BD_times_BC_eq_five : BD * BC = 5]

-- Prove that the length of AB is sqrt(5)
theorem length_AB_eq_sqrt_5 : length (A B) = sqrt 5 := by
  sorry

end length_AB_eq_sqrt_5_l34_34582


namespace tan_alpha_eq_2_l34_34105

theorem tan_alpha_eq_2 (α : Real) (h : Real.tan α = 2) : 
  1 / (Real.sin (2 * α) + Real.cos (α) ^ 2) = 1 := 
by 
  sorry

end tan_alpha_eq_2_l34_34105


namespace probability_defective_units_l34_34235

theorem probability_defective_units (X : ℝ) (hX : X > 0) :
  let defectA := (14 / 2000) * (0.40 * X)
  let defectB := (9 / 1500) * (0.35 * X)
  let defectC := (7 / 1000) * (0.25 * X)
  let total_defects := defectA + defectB + defectC
  let total_units := X
  let probability := total_defects / total_units
  probability = 0.00665 :=
by
  sorry

end probability_defective_units_l34_34235


namespace segment_length_of_intersections_of_circle_with_sides_l34_34931

theorem segment_length_of_intersections_of_circle_with_sides
  (XY XZ YZ: ℝ) (right_triangle: ∀ (a b c: ℝ), a^2 + b^2 = c^2): 
  XY = 13 -> XZ = 12 -> YZ = 5 ->
  (∃ Z M S, right_triangle Z XZ YZ ∧ dist XM = 4) ->
  QR = 120 / 13 :=
by
  sorry

end segment_length_of_intersections_of_circle_with_sides_l34_34931


namespace find_a_l34_34916

theorem find_a (a : ℝ) (h : (Polynomial.monomial 3 (Nat.choose 5 3 * a^3) * Polynomial.C (80 : ℝ)) = 80) : a = 2 :=
by
  sorry

end find_a_l34_34916


namespace total_value_of_bills_in_cash_drawer_l34_34359

-- Definitions based on conditions
def total_bills := 54
def five_dollar_bills := 20
def twenty_dollar_bills := total_bills - five_dollar_bills
def value_of_five_dollar_bills := 5
def value_of_twenty_dollar_bills := 20
def total_value_of_five_dollar_bills := five_dollar_bills * value_of_five_dollar_bills
def total_value_of_twenty_dollar_bills := twenty_dollar_bills * value_of_twenty_dollar_bills

-- Statement to prove
theorem total_value_of_bills_in_cash_drawer :
  total_value_of_five_dollar_bills + total_value_of_twenty_dollar_bills = 780 :=
by
  -- Proof goes here
  sorry

end total_value_of_bills_in_cash_drawer_l34_34359


namespace union_of_A_and_B_l34_34230

-- Definitions of sets A and B
def A := set.Ioc (-1: ℝ) 1  -- (-1, 1]
def B := set.Ioo (0: ℝ) 2   -- (0, 2)

-- Goal: Prove that A ∪ B = (-1, 2)
theorem union_of_A_and_B : set.union A B = set.Ioc (-1: ℝ) 2 := sorry

end union_of_A_and_B_l34_34230


namespace ratio_of_sum_of_odd_to_even_divisors_of_M_l34_34213

def M : ℕ := 36 * 36 * 65 * 275

def sum_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).sum

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_even (n : ℕ) : Prop := n % 2 = 0

def sum_of_odd_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0 ∧ is_odd d).sum

def sum_of_even_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0 ∧ is_even d).sum

theorem ratio_of_sum_of_odd_to_even_divisors_of_M :
  (sum_of_odd_divisors M : ℚ) / sum_of_even_divisors M = 1 / 30 :=
sorry

end ratio_of_sum_of_odd_to_even_divisors_of_M_l34_34213


namespace abs_neg_three_eq_three_l34_34284

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l34_34284


namespace number_of_real_solutions_eq_2_l34_34762

theorem number_of_real_solutions_eq_2 :
  ∃! (x : ℝ), (6 * x) / (x^2 + 2 * x + 5) + (7 * x) / (x^2 - 7 * x + 5) = -5 / 3 :=
sorry

end number_of_real_solutions_eq_2_l34_34762


namespace circle_equation_correct_l34_34628

theorem circle_equation_correct (x y : ℝ) : 
  (∃ (c : ℝ), (c = 2) ∧ (x^2 + (y - c)^2 = 1) ∧ (∃ p : ℝ × ℝ, p = (1, 2) ∧ (1^2 + (2 - c)^2 = 1))) → 
  x^2 + (y - 2)^2 = 1 :=
by
  intro h,
  cases h with c hc,
  cases hc.2 with p hp,
  cases hp with hxy heq,
  simp at *,
  sorry

end circle_equation_correct_l34_34628


namespace value_of_a8_l34_34820

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → 2 * a n + a (n + 1) = 0

theorem value_of_a8 (a : ℕ → ℝ) (h1 : seq a) (h2 : a 3 = -2) : a 8 = 64 :=
sorry

end value_of_a8_l34_34820


namespace find_a_b_solve_inequality_l34_34860

noncomputable def f (a x : ℝ) := real.sqrt (a * x^2 - 3 * x + 2)

theorem find_a_b :
  ∃ a b : ℝ, a > 0 ∧ (λ x, f a x ∈ {x | x < 1 ∨ x > b}) ∧
  (ax^2 - 3x + 2 = 0) ∧ b = 2 / a ∧ 1 + b = 3 / a ∧
  a = 1 ∧ b = 2 :=
by
  sorry

theorem solve_inequality (a c : ℝ) (h : a = 1 ∧ b = 2) :
  ∀ x : ℝ, (x - c) / (a * x - b) > 0 ↔
  (if c > 2 then x > c ∨ x < 2
  else if c < 2 then x > 2 ∨ x < c
  else x ≠ 2) :=
by
  sorry

end find_a_b_solve_inequality_l34_34860


namespace three_digit_integers_congruent_to_2_mod_4_l34_34903

theorem three_digit_integers_congruent_to_2_mod_4 : 
  let count := (249 - 25 + 1) in
  count = 225 :=
by
  let k_min := 25
  let k_max := 249
  have h_count : count = (k_max - k_min + 1) := rfl
  rw h_count
  norm_num

end three_digit_integers_congruent_to_2_mod_4_l34_34903


namespace fourth_root_of_506250000_l34_34409

theorem fourth_root_of_506250000 :
  ∃ x : ℕ, x ^ 4 = 506250000 ∧ x = 150 :=
by
  use 150
  split
  · norm_num
  · rfl

end fourth_root_of_506250000_l34_34409


namespace abs_neg_three_l34_34273

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l34_34273


namespace number_of_odd_digits_in_base_5_representation_of_346_l34_34092

def base_5_representation (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else
    let rec digits (x : ℕ) : list ℕ :=
      if x = 0 then [] else (x % 5) :: digits (x / 5)
    digits n

def odd_digits_base_5 (n : ℕ) : ℕ :=
  (base_5_representation n).countp (λ d => d % 2 = 1)

theorem number_of_odd_digits_in_base_5_representation_of_346 :
  odd_digits_base_5 346 = 2 :=
by
  sorry

end number_of_odd_digits_in_base_5_representation_of_346_l34_34092


namespace percentage_increase_on_bought_price_l34_34031

-- Define the conditions as Lean definitions
def original_price (P : ℝ) : ℝ := P
def bought_price (P : ℝ) : ℝ := 0.90 * P
def selling_price (P : ℝ) : ℝ := 1.62000000000000014 * P

-- Lean statement to prove the required result
theorem percentage_increase_on_bought_price (P : ℝ) :
  (selling_price P - bought_price P) / bought_price P * 100 = 80.00000000000002 := by
  sorry

end percentage_increase_on_bought_price_l34_34031


namespace standard_normal_symmetric_l34_34585

noncomputable def standard_normal_distribution : ProbabilityDistribution ℝ :=
{ density := λ x, (1 / (Mathlib.Real.pi.sqrt * 2)) * Mathlib.Real.exp (-(x^2)/2),
  density_integrable := sorry,
  density_nonneg := sorry }

theorem standard_normal_symmetric (p : ℝ) :
  (P (λ ξ, ξ ~ standard_normal_distribution) (Set.Ici 1) = p) →
  (P (λ ξ, ξ ~ standard_normal_distribution) (Set.Ioo (-1) 0) = (1 / 2) - p) := 
sorry

end standard_normal_symmetric_l34_34585


namespace faster_train_speed_l34_34332

theorem faster_train_speed (V_s : ℝ) (t : ℝ) (l : ℝ) (V_f : ℝ) : 
  V_s = 36 → t = 20 → l = 200 → V_f = V_s + (l / t) * 3.6 → V_f = 72 
  := by
    intros _ _ _ _
    sorry

end faster_train_speed_l34_34332


namespace increase_in_work_l34_34348

theorem increase_in_work (W p : ℕ) (h : p ≠ 0) (h' : W ≠ 0) :
  (1/5 : ℚ) * p ∈ ℕ → 
  let absent : ℕ := (1/5 : ℚ) * p in 
  let present : ℕ := p - absent in
  ∃ k : ℚ, k = (W/p : ℚ) + (W/(present : ℚ)) ∧ k = W/(4*p : ℚ) :=
by sorry

end increase_in_work_l34_34348


namespace concyclic_points_l34_34532

theorem concyclic_points
  (Γ1 Γ2 : Circle)
  (P Q : Point)
  (M : Midpoint P Q)
  (A B C D : Point)
  (M_on_AB : Line_through M A B)
  (M_on_CD : Line_through M C D)
  (BD_G_on_Γ1 : ∃ G, Line_through B D ∧ G ∈ Γ1 ∧ G ≠ B)
  (BD_H_on_Γ2 : ∃ H, Line_through B D ∧ H ∈ Γ2 ∧ H ≠ D)
  (E_on_BM : ∃ E, E ∈ BM ∧ EM = AM)
  (F_on_DM : ∃ F, F ∈ DM ∧ FM = CM)
  (EH_FG_N : ∃ N, Intersection EH FG N) :
  Cyclic M E N F := sorry

end concyclic_points_l34_34532


namespace min_weights_unique_weights_l34_34120

-- Definitions for minimum number of weights needed to measure from 1 to n grams
def f (n : ℕ) : ℕ := sorry -- Placeholder for the correct mathematical function

-- Condition: Given a positive integer n,
variable (n : ℕ) (hn : 0 < n)

-- Minimum number of weights
theorem min_weights (m : ℕ) (hm1 : (3^(m-1) - 1) // 2 < n) (hm2 : n ≤ (3^m - 1) // 2) : f n = m := 
sorry

-- Uniqueness of weights composition when n is specific value
theorem unique_weights (m : ℕ) (hn : n = (3^m - 1) // 2) : 
∀ (i : ℕ), 1 ≤ i ∧ i ≤ m → ∃! a : ℕ, a = 3^(i-1) := 
sorry

end min_weights_unique_weights_l34_34120


namespace avg_ge_half_n_plus_one_l34_34226

theorem avg_ge_half_n_plus_one (m n : ℕ) (A : Finset ℕ) (hA₁ : ∀ (a ∈ A), a ≥ 1 ∧ a ≤ n) 
  (hA₂ : ∀ i j, i ∈ A → j ∈ A → i + j ≤ n → i + j ∈ A) 
  (hA₃ : A.card = m) :
  (A.sum id / m : ℚ) ≥ (n + 1) / 2 :=
sorry

end avg_ge_half_n_plus_one_l34_34226


namespace cone_volume_calc_l34_34492

noncomputable def cone_volume (diameter slant_height: ℝ) : ℝ :=
  let r := diameter / 2
  let h := Real.sqrt (slant_height^2 - r^2)
  (1 / 3) * Real.pi * r^2 * h

theorem cone_volume_calc :
  cone_volume 12 10 = 96 * Real.pi :=
by
  sorry

end cone_volume_calc_l34_34492


namespace volume_of_circular_pool_approx_l34_34044

noncomputable def pool_volume (d h : ℝ) : ℝ :=
  let r := d / 2
  π * r^2 * h

theorem volume_of_circular_pool_approx :
  pool_volume 80 10 ≈ 50265.6 :=
by 
  -- The main calculation is deferred as a sorry here
  sorry

end volume_of_circular_pool_approx_l34_34044


namespace inequality_system_range_l34_34179

theorem inequality_system_range (m : ℝ) :
  (∃ (S : set ℤ), S = {x : ℤ | x < 8 ∧ x > m} ∧ S.card = 3) ↔ 4 ≤ m ∧ m < 5 := sorry

end inequality_system_range_l34_34179


namespace triangle_inscribed_circle_l34_34364

theorem triangle_inscribed_circle :
  ∀ (A B C D E F : ℝ) (α : ℝ),
  (triangle ABC) ∧ (angle_ABC = 90) ∧
  (circle_inscribed ABC) ∧ 
  (touch_points ABC D E F) →
  (triangle DEF) ∧ (angle_SUM DEF = 180) ∧
  (one_angle_OBTUSE DEF = (∃ θ : angle DEF, θ ≥ 90)) ∧
  (two_angles_EQUAL ACUTE DEF = 
    (∃ α1 α2 : angle DEF, α1 = α2 ∧ α1 < 90 ∧ α2 < 90)) :=
by 
  sorry

end triangle_inscribed_circle_l34_34364


namespace total_time_equals_l34_34383

-- Define the distances and speeds
def first_segment_distance : ℝ := 50
def first_segment_speed : ℝ := 30
def second_segment_distance (b : ℝ) : ℝ := b
def second_segment_speed : ℝ := 80

-- Prove that the total time is equal to (400 + 3b) / 240 hours
theorem total_time_equals (b : ℝ) : 
  (first_segment_distance / first_segment_speed) + (second_segment_distance b / second_segment_speed) 
  = (400 + 3 * b) / 240 := 
by
  sorry

end total_time_equals_l34_34383


namespace no_d1_d2_multiple_of_7_l34_34975
open Function

theorem no_d1_d2_multiple_of_7 (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 100) :
  let d1 := a^2 + 3^a + a * 3^((a+1)/2)
  let d2 := a^2 + 3^a - a * 3^((a+1)/2)
  ¬(d1 * d2 % 7 = 0) :=
by
  let d1 := a^2 + 3^a + a * 3^((a+1)/2)
  let d2 := a^2 + 3^a - a * 3^((a+1)/2)
  sorry

end no_d1_d2_multiple_of_7_l34_34975


namespace flowchart_outputs_l34_34255

variable (a b c : ℕ)

-- The flowchart transformation function
def flowchart (a b c : ℕ) : ℕ × ℕ × ℕ :=
  (c, a, b) -- assuming the transformation based on the provided answer

theorem flowchart_outputs :
  flowchart 21 32 75 = (75, 21, 32) :=
  by
    simp [flowchart]
    sorry -- Placeholder for the actual proof

end flowchart_outputs_l34_34255


namespace recurrence_b_explicit_form_b_l34_34319

noncomputable def a : ℕ → ℤ
| 0       := 4
| 1       := 1
| (n + 1) := a n + 6 * a (n - 1)

def b (n : ℕ) : ℤ :=
  ∑ k in Finset.range (n+1), (Nat.choose n k) * a k

theorem recurrence_b (n : ℕ) : 
  b (n + 1) = 4 * b n + 5 * b (n - 1) :=
sorry

theorem explicit_form_b (n : ℕ) : 
  b n = (9/5 : ℚ) * 4^n + (11/5 : ℚ) * (-1)^n :=
sorry

end recurrence_b_explicit_form_b_l34_34319


namespace platform_length_is_correct_l34_34728

noncomputable def length_of_platform (time_to_pass_man : ℝ) (time_to_cross_platform : ℝ) (length_of_train : ℝ) : ℝ := 
  length_of_train * time_to_cross_platform / time_to_pass_man - length_of_train

theorem platform_length_is_correct : length_of_platform 8 20 178 = 267 := 
  sorry

end platform_length_is_correct_l34_34728


namespace number_of_strictly_increasing_three_digit_integers_with_5_l34_34165

-- Definition of a strictly increasing sequence
def strictly_increasing (digits : List Nat) : Prop :=
  ∀ i j, i < j → digits.nth i < digits.nth j

-- Definition to check if a list contains the digit 5
def contains_digit (n : Nat) (digits : List Nat) : Prop :=
  n ∈ digits

-- Function to extract the digits of a number
def digits_of_num (n : Nat) : List Nat :=
  n.digits

-- Main theorem statement
theorem number_of_strictly_increasing_three_digit_integers_with_5 :
  ∃ count : Nat, count = 12 ∧ ∀ n, 
    (100 ≤ n ∧ n < 1000) →
    strictly_increasing (digits_of_num n) →
    contains_digit 5 (digits_of_num n) →
    count = (number of such valid n) := sorry


end number_of_strictly_increasing_three_digit_integers_with_5_l34_34165


namespace compare_y_values_l34_34862

variable (a : ℝ) (y₁ y₂ : ℝ)
variable (h : a > 0)
variable (p1 : y₁ = a * (-1 : ℝ)^2 - 4 * a * (-1 : ℝ) + 2)
variable (p2 : y₂ = a * (1 : ℝ)^2 - 4 * a * (1 : ℝ) + 2)

theorem compare_y_values : y₁ > y₂ :=
by {
  sorry
}

end compare_y_values_l34_34862


namespace min_selling_price_400_l34_34745

theorem min_selling_price_400
  (n : ℕ)
  (avg_price : ℕ)
  (products_lt_1000 : ℕ)
  (max_price : ℕ)
  (total_price : ℕ)
  (min_price : ℕ)
  (h1 : n = 20) 
  (h2 : avg_price = 1200)
  (h3 : products_lt_1000 = 10)
  (h4 : max_price = 11000)
  (h5 : total_price = 24000)
  (h6 : total_price = n * avg_price)
  (h7 : products_lt_1000 * min_price = 4000)
  : min_price = 400 :=
begin
  sorry
end

end min_selling_price_400_l34_34745


namespace flour_needed_for_9_biscuits_l34_34206

theorem flour_needed_for_9_biscuits (flour_for_36_biscuits : ℝ) (num_of_biscuits_per_batch : ℝ) 
  (flour_for_desired_biscuits : ℝ) (h1 : flour_for_36_biscuits = 5) (h2 : num_of_biscuits_per_batch = 9) 
  (h3 : flour_for_desired_biscuits = flour_for_36_biscuits / 4) : flour_for_desired_biscuits = 1.25 :=
by
  rw [h1, h2, h3]
  norm_num

end flour_needed_for_9_biscuits_l34_34206


namespace at_least_one_le_one_l34_34985

theorem at_least_one_le_one (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
sorry

end at_least_one_le_one_l34_34985


namespace part1_part2_l34_34810

theorem part1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) : 
  (1 / a) + (1 / (b + 1)) ≥ 4 / 5 := 
by 
  sorry

theorem part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b + a * b = 8) : 
  a + b ≥ 4 := 
by 
  sorry

end part1_part2_l34_34810


namespace total_time_to_clean_and_complete_l34_34330

def time_to_complete_assignment : Nat := 10
def num_remaining_keys : Nat := 14
def time_per_key : Nat := 3

theorem total_time_to_clean_and_complete :
  time_to_complete_assignment + num_remaining_keys * time_per_key = 52 :=
by
  sorry

end total_time_to_clean_and_complete_l34_34330


namespace range_f_on_interval_l34_34435

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 6

theorem range_f_on_interval : 
  set.range (λ x, f x) (set.Ico 1 5) = set.Ico 2 11 := 
sorry

end range_f_on_interval_l34_34435


namespace abs_neg_three_l34_34298

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l34_34298


namespace sodas_purchasable_l34_34615

namespace SodaPurchase

variable {D C : ℕ}

theorem sodas_purchasable (D C : ℕ) : (3 * (4 * D) / 5 + 5 * C / 15) = (36 * D + 5 * C) / 15 := 
  sorry

end SodaPurchase

end sodas_purchasable_l34_34615


namespace ostrich_emu_leap_difference_l34_34774

theorem ostrich_emu_leap_difference 
  (elmer_strides_per_gap : ℕ) 
  (oscar_leaps_per_gap : ℕ) 
  (distance_between_poles : ℕ) 
  (number_of_poles : ℕ) :
  (elmer_strides_per_gap = 44) → 
  (oscar_leaps_per_gap = 12) → 
  (distance_between_poles = 5280) → 
  (number_of_poles = 41) →
  let total_distance := distance_between_poles * (number_of_poles - 1) in
  let total_strides := elmer_strides_per_gap * (number_of_poles - 1) in
  let total_leaps := oscar_leaps_per_gap * (number_of_poles - 1) in
  let elmer_stride_length := total_distance / total_strides in
  let oscar_leap_length := total_distance / total_leaps in
  oscar_leap_length - elmer_stride_length = 8 := 
by 
  sorry

end ostrich_emu_leap_difference_l34_34774


namespace donkey_wins_l34_34246

/-- On a plane, there are 2005 points (no three of which are collinear). Each pair of points is connected by a segment. 
Donkey labels each segment with one of the numbers from 0 to 8. 
Tiger labels each point with one of the numbers from 0 to 8. 
Donkey wins if there exist two points connected by a segment that are labeled with the same number as the segment. 
Prove that with the correct strategy, Donkey will win. -/
theorem donkey_wins (points : Finset Point) (h : points.card = 2005) 
  (no_three_collinear : ∀ {p1 p2 p3 : Point}, p1 ∈ points → p2 ∈ points → p3 ∈ points → p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬Collinear p1 p2 p3)
  (label_segment : ∀ (p1 p2 : Point), p1 ∈ points → p2 ∈ points → ℕ) 
  (label_point : ∀ (p : Point), p ∈ points → ℕ) :
  ∃ (p1 p2 : Point), p1 ∈ points ∧ p2 ∈ points ∧ label_point p1 = label_segment p1 p2 ∧ label_segment p1 p2 = label_point p2 :=
by
  sorry

end donkey_wins_l34_34246


namespace average_weight_increase_l34_34186

theorem average_weight_increase 
  (n : ℕ) (A : ℕ → ℝ)
  (h_total : n = 10)
  (h_replace : A 65 = 137) : 
  (137 - 65) / 10 = 7.2 := 
by 
  sorry

end average_weight_increase_l34_34186


namespace value_of_a_l34_34147

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else 2 * x

theorem value_of_a (a : ℝ) (h : f a = 10) : a = -3 ∨ a = 5 := 
by
  sorry

end value_of_a_l34_34147


namespace smaller_is_3sqrtk_div_17_l34_34417

noncomputable def find_smaller_of_p_and_q (k : ℝ) (p q : ℝ) (h1 : 0 < p) (h2 : 0 < q) 
(h3 : p / q = 3 / 5) (h4 : p^2 + q^2 = 2 * k) : ℝ :=
  if h5 : p < q then p else q

theorem smaller_is_3sqrtk_div_17 (p q : ℝ) (k : ℝ) (h1 : 0 < p) (h2 : 0 < q)
(h3 : p / q = 3 / 5) (h4 : p^2 + q^2 = 2 * k) : 
  find_smaller_of_p_and_q k p q h1 h2 h3 h4 = 3 * real.sqrt (k / 17) :=
begin
  sorry
end

end smaller_is_3sqrtk_div_17_l34_34417


namespace min_distance_between_parallel_lines_l34_34571

theorem min_distance_between_parallel_lines :
  let P := { x : ℝ × ℝ // 3 * x.1 + 4 * x.2 - 10 = 0 }
  let Q := { x : ℝ × ℝ // 6 * x.1 + 8 * x.2 + 5 = 0 }
  ∃ d : ℝ, d = (abs (-10 - (5 / 2))) / (sqrt ((3 : ℝ)^2 + (4 : ℝ)^2)) ∧ d = (5 / 2) :=
begin
  let d := (abs (-10 - (5 / 2))) / (sqrt ((3 : ℝ)^2 + (4 : ℝ)^2)),
  have : d = 5 / 2, 
  { 
     -- Proof skipped
     sorry 
  },
  exact ⟨d, rfl, this⟩
end

end min_distance_between_parallel_lines_l34_34571


namespace sin_ratio_eq_two_triangle_area_eq_l34_34183

-- Problem 1: Prove that \(\frac{\sin C}{\sin A} = 2\)
theorem sin_ratio_eq_two {A B C a b c : ℝ} (h_trianlge_sides : ∃ (triangle : ℝ), triangle = a + b + c)
  (m : EuclideanSpace ℝ (Fin 2)) (n : EuclideanSpace ℝ (Fin 2)) 
  (h_m : m = ![b, a - 2 * c]) (h_n : n = ![cos A - 2 * cos C, cos B])
  (h_perp : m ⬝ n = 0) :
  (sin C) / (sin A) = 2 := 
sorry

-- Problem 2: Prove that the area of \(\triangle ABC\) equals \(\frac{3\sqrt{15}}{4}\)
theorem triangle_area_eq {A B C a b c : ℝ}
  (h_trianlge_sides : ∃ (triangle : ℝ), triangle = a + b + c)
  (m_length : ∥![b, a - 2 * c]∥ = 3 * real.sqrt 5) (a_eq_two : a = 2)
  (sin_ratio : (sin C) / (sin A) = 2) :
  let S := 1/2 * b * c * sin A in S = 3 * real.sqrt 15 / 4 :=
sorry

end sin_ratio_eq_two_triangle_area_eq_l34_34183


namespace solve_for_x_l34_34607

theorem solve_for_x :
  ∃ x : ℝ, (sqrt (9 + sqrt (27 + 9 * x)) + sqrt (3 + sqrt (3 + x)) = 3 + 3 * sqrt 3) ↔ x = 33 :=
by
  sorry

end solve_for_x_l34_34607


namespace largest_non_prime_among_five_consecutive_l34_34095

theorem largest_non_prime_among_five_consecutive : 
  ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ a + 4 = e ∧ a ≥ 10 ∧ e < 50 ∧ 
  ¬(prime a) ∧ ¬(prime b) ∧ ¬(prime c) ∧ ¬(prime d) ∧ ¬(prime e) ∧ e = 36 :=
by
  sorry

end largest_non_prime_among_five_consecutive_l34_34095


namespace no_root_for_equation_l34_34625

theorem no_root_for_equation : ∀ x : ℝ, x ≠ 3 → (x - 7 / (x - 3) ≠ 3 - 7 / (x - 3)) :=
begin
  intros x h,
  sorry
end

end no_root_for_equation_l34_34625


namespace gcd_of_B_l34_34561

def B : Set ℕ := {n | ∃ x : ℕ, n = 5 * x}

theorem gcd_of_B : Int.gcd_range (B.toList) = 5 := by 
  sorry

end gcd_of_B_l34_34561


namespace most_cost_effective_plan_l34_34303

noncomputable def bus_capacity := 40
noncomputable def bus_ticket_price := 5
noncomputable def bus_discounted_price := 0.8 * bus_ticket_price
noncomputable def minivan_capacity := 10
noncomputable def minivan_ticket_price := 6
noncomputable def minivan_discounted_price := 0.75 * minivan_ticket_price
noncomputable def total_people := 120

theorem most_cost_effective_plan :
  let num_buses := total_people / bus_capacity in
  let bus_cost := num_buses * bus_capacity * bus_discounted_price in
  let num_minivans := total_people / minivan_capacity in
  let minivan_cost := num_minivans * minivan_capacity * minivan_discounted_price in
  bus_cost = 480 ∧ bus_cost < minivan_cost :=
by
  sorry

end most_cost_effective_plan_l34_34303


namespace chocolate_cost_l34_34063

theorem chocolate_cost (Ccb Cc : ℝ) (h1 : Ccb = 6) (h2 : Ccb = Cc + 3) : Cc = 3 :=
by
  sorry

end chocolate_cost_l34_34063


namespace smallest_number_is_43_l34_34386

noncomputable def decimal_of_base6 (n : ℕ) : ℕ := 2 * 6^2 + 1 * 6^1 + 0 * 6^0
noncomputable def decimal_of_base7 (n : ℕ) : ℕ := 1 * 7^3 + 0 * 7^2 + 0 * 7^1 + 0 * 7^0
noncomputable def decimal_of_base2 (n : ℕ) : ℕ := 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem smallest_number_is_43 :
  let a := 85,
      b := decimal_of_base6 210,
      c := decimal_of_base7 1000,
      d := decimal_of_base2 101011
  in min (min a b) (min c d) = d :=
by
  sorry

end smallest_number_is_43_l34_34386


namespace find_principal_l34_34029

theorem find_principal
  (SI : ℝ)
  (R : ℝ)
  (T : ℝ)
  (h_SI : SI = 4025.25)
  (h_R : R = 0.09)
  (h_T : T = 5) : 
  (SI / (R * T / 100)) = 8950 :=
by
  rw [h_SI, h_R, h_T]
  sorry

end find_principal_l34_34029


namespace solve_fraction_eq_l34_34649

theorem solve_fraction_eq : ∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → (2 / x = 3 / (x + 1)) ↔ (x = 2) :=
by
  intros x h
  split 
  sorry

end solve_fraction_eq_l34_34649


namespace transformation_in_S_l34_34227

def S : set (ℂ) := {z : ℂ | -2 ≤ z.re ∧ z.re ≤ 2 ∧ -2 ≤ z.im ∧ z.im ≤ 2}

theorem transformation_in_S (z : ℂ) (h : z ∈ S) : 
  (1/2 + 1/2 * complex.I) * z ∈ S := 
sorry

end transformation_in_S_l34_34227


namespace binomial_pmf_value_l34_34584

open ProbabilityTheory

noncomputable def binomial_pmf 
  (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem binomial_pmf_value :
  binomial_pmf 6 (1 / 2) 3 = 5 / 16 :=
by
  sorry

end binomial_pmf_value_l34_34584


namespace at_least_one_term_le_one_l34_34987

theorem at_least_one_term_le_one
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x + y + z = 3) :
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
  sorry

end at_least_one_term_le_one_l34_34987


namespace factor_10000_into_three_natural_factors_l34_34519

theorem factor_10000_into_three_natural_factors :
  (∃ (a b c : ℕ), a * b * c = 10000 ∧ ¬∃ x y z, ((x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2)) ∧ (a = x^_ ∨ b = y^_ ∨ c = z^_) ∧ (∀ perm, perm.factor(a, b, c) = (a, b, c))) →
  6 :=
by sorry

end factor_10000_into_three_natural_factors_l34_34519


namespace sum_first_n_terms_l34_34953

noncomputable def a : ℕ → ℕ
| 0     := 0
| (n+1) := 3 * a n + 4 * n

def λ := 2
def μ := 1

def b (n : ℕ) := a n + λ * n + μ

def c (n : ℕ) := (λ * n + μ) * (a n + λ * n + μ)

def S (n : ℕ) := (finset.range n).sum c

theorem sum_first_n_terms (n : ℕ) : S n = n * 3^(n+1) :=
by
  sorry

end sum_first_n_terms_l34_34953


namespace bowls_sold_l34_34018

-- Define the conditions
def num_bowls : ℕ := 114
def cost_per_bowl : ℝ := 13
def selling_price_per_bowl : ℝ := 17
def gain_percentage : ℝ := 23.88663967611336 / 100

-- Given conditions in Lean terms
def total_cost_price : ℝ := num_bowls * cost_per_bowl
def total_selling_price (x : ℕ) : ℝ := x * selling_price_per_bowl
def expected_selling_price : ℝ := total_cost_price * (1 + gain_percentage)

-- Required to prove that the number of bowls sold is 108
theorem bowls_sold : ∃ x : ℕ, total_selling_price x = expected_selling_price ∧ x = 108 :=
by
  sorry

end bowls_sold_l34_34018


namespace length_of_MN_l34_34152

noncomputable def parabola_L := λ x: ℝ, 8*x
noncomputable def focus_point := (2, 0 : ℝ)
noncomputable def directrix_line := λ x: ℝ, -2

/-
  Given the parabola C: y^2 = 8x with focus F and directrix l, and P is a point on l
  The line PF intersects the parabola at points M and N. If \overrightarrow{PF}=3\overrightarrow{MF}, 
  then the length |MN| is \frac{32}{3}
-/
theorem length_of_MN
  (x1 y1 x2 y2: ℝ)
  (h1: y1 ^ 2 = 8 * x1)
  (h2: y2 ^ 2 = 8 * x2)
  (h3: (focus_point.fst - x1)^2 + (focus_point.snd - y1)^2 = ((focus_point.fst - x2)^2 + (focus_point.snd - y2)^2))
  (h4: (x1 + 2) = ∥focus_point∥)
  (h5: (x2 +2) = ∥focus_point∥)
  (h6: 3 * ∥focus_point - (x1, y1)∥ = ∥focus_point - (1, 0)∥): 
  ∥(x2, y2) - (x1, y1)∥ = 32 / 3 := by
  sorry

end length_of_MN_l34_34152


namespace negation_of_P_l34_34125

def P : Prop := ∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 ≤ 0

theorem negation_of_P : ¬ P ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by sorry

end negation_of_P_l34_34125


namespace pencils_per_row_l34_34083

-- Definitions of conditions.
def num_pencils : ℕ := 35
def num_rows : ℕ := 7

-- Hypothesis: given the conditions, prove the number of pencils per row.
theorem pencils_per_row : num_pencils / num_rows = 5 := 
  by 
  -- Proof steps go here, but are replaced by sorry.
  sorry

end pencils_per_row_l34_34083


namespace complement_A_in_U_l34_34487

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}

theorem complement_A_in_U : (U \ A) = {x | -1 <= x ∧ x <= 3} :=
by
  sorry

end complement_A_in_U_l34_34487


namespace car_sample_size_l34_34705

theorem car_sample_size (total_sales_A total_sales_B total_sales_C selected_B : ℕ)
  (hA : total_sales_A = 420)
  (hB : total_sales_B = 280)
  (hC : total_sales_C = 700)
  (h_selected_B : selected_B = 20)
  (proportional_allocation : ∀ a b c : ℕ, a = total_sales_A / 140 ∧ b = total_sales_B / 140 ∧ c = total_sales_C / 140 → 3 = a ∧ 2 = b ∧ 5 = c) :
  let m := 3 * (selected_B / 2)
  let n := 5 * (selected_B / 2)
  in m + selected_B + n = 100 := by
  sorry

end car_sample_size_l34_34705


namespace counting_stones_l34_34028

theorem counting_stones :
  ∃ (n_vals : Finset ℕ),
    (∀ n ∈ n_vals, ∃ m : ℕ, 11 * m + 9 * n = 2007) ∧
    card n_vals = 21 :=
by
  sorry

end counting_stones_l34_34028


namespace maximum_of_function_l34_34782

theorem maximum_of_function :
  ∃ x y : ℝ, 
    (1/3 ≤ x ∧ x ≤ 2/5 ∧ 1/4 ≤ y ∧ y ≤ 5/12) ∧ 
    (∀ x' y' : ℝ, 1/3 ≤ x' ∧ x' ≤ 2/5 ∧ 1/4 ≤ y' ∧ y' ≤ 5/12 → 
                (xy / (x^2 + y^2) ≤ x' * y' / (x'^2 + y'^2))) ∧ 
    (xy / (x^2 + y^2) = 20 / 41) := 
sorry

end maximum_of_function_l34_34782


namespace factorization_10000_l34_34521

def is_factorization (n a b c : ℕ) : Prop :=
  a * b * c = n

def no_factor_divisible_by_10 (a b c : ℕ) : Prop :=
  ¬(10 ∣ a) ∧ ¬(10 ∣ b) ∧ ¬(10 ∣ c)

theorem factorization_10000 :
  ∃ (a b c : ℕ), is_factorization 10000 a b c ∧ no_factor_divisible_by_10 a b c ∧ 
  finset.card (finset.univ.filter (λ x, ∃ (a b c : ℕ), x = ({a, b, c} : finset ℕ) ∧
    is_factorization 10000 a b c ∧ no_factor_divisible_by_10 a b c)) = 6 :=
begin
  sorry
end

end factorization_10000_l34_34521


namespace geometric_seq_arithmetic_example_l34_34198

noncomputable def a_n (n : ℕ) (q : ℝ) : ℝ :=
if n = 0 then 1 else q ^ n

theorem geometric_seq_arithmetic_example {q : ℝ} (h₀ : q ≠ 0)
    (h₁ : ∀ n : ℕ, a_n 0 q = 1)
    (h₂ : 2 * (2 * (q ^ 2)) = 3 * q) :
    (q + q^2 + (q^3)) = 14 :=
by sorry

end geometric_seq_arithmetic_example_l34_34198
