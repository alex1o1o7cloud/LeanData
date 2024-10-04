import Mathlib

namespace value_fraction_l667_667166

variables {x y : ℝ}
variables (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + 2 * y) / (x - 4 * y) = 3)

theorem value_fraction : (x + 4 * y) / (4 * x - y) = 10 / 57 :=
by { sorry }

end value_fraction_l667_667166


namespace intercept_x_parallel_lines_l667_667474

theorem intercept_x_parallel_lines (m : ℝ) 
    (line_l : ∀ x y : ℝ, y + m * (x + 1) = 0) 
    (parallel : ∀ x y : ℝ, y * m - (2 * m + 1) * x = 1) : 
    ∃ x : ℝ, x + 1 = -1 :=
by
  sorry

end intercept_x_parallel_lines_l667_667474


namespace impossible_to_maintain_Gini_l667_667690

variables (X Y G0 Y' Z : ℝ)
variables (G1 : ℝ)

-- Conditions
axiom initial_Gini : G0 = 0.1
axiom proportion_poor : X = 0.5
axiom income_poor_initial : Y = 0.4
axiom income_poor_half : Y' = 0.2
axiom population_split : ∀ a b c : ℝ, (a + b + c = 1) ∧ (a = b ∧ b = c)
axiom Gini_constant : G1 = G0

-- Equation system representation final value post situation
axiom Gini_post_reform : 
  G1 = (1 / 2 - ((1 / 6) * 0.2 + (1 / 6) * (0.2 + Z) + (1 / 6) * (1 - 0.2 - Z))) / (1 / 2)

-- Proof problem: to prove inconsistency or inability to maintain Gini coefficient given the conditions
theorem impossible_to_maintain_Gini : false :=
sorry

end impossible_to_maintain_Gini_l667_667690


namespace max_area_of_triangle_l667_667126

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  1/2 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem max_area_of_triangle : 
  let A := (1 : ℝ, 0 : ℝ)
  let B := (4 : ℝ, 3 : ℝ)
  ∃ p, 1 ≤ p ∧ p ≤ 4 ∧ 
  let C := (p, p^2 - 4 * p + 3) in 
  area_of_triangle A B C = 27 / 8 :=
by sorry

end max_area_of_triangle_l667_667126


namespace intersection_of_sets_l667_667039

def setA : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 4}
def setB : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_sets :
  setA ∩ setB = {x : ℝ | -2 ≤ x ∧ x < 4} :=
by
  sorry

end intersection_of_sets_l667_667039


namespace integer_solution_unique_l667_667860

theorem integer_solution_unique (x y z : ℤ) : x^3 - 2*y^3 - 4*z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 :=
by 
  sorry

end integer_solution_unique_l667_667860


namespace convex_polygons_common_point_and_rotation_l667_667043

noncomputable def vec (A B : Point) : Vector := sorry
structure Polygon := (vertices : List Point)

theorem convex_polygons_common_point_and_rotation (A B : Polygon) 
  (h_convex_A : convex A) 
  (h_convex_B : convex B) 
  (h_condition : ∑ i in finset.range A.vertices.length, vec (A.vertices.nth i).get_or_else (0, 0) (B.vertices.nth i).get_or_else (0, 0) = 0) : 
  (∃ P, P ∈ A.vertices ∧ P ∈ B.vertices) ∧ 
  (∃ A' : Polygon, rotated A A' ∧ (vec (A'.vertices.nth 0).get_or_else (0, 0) (A'.vertices.nth 1).get_or_else (0, 0)) ⊥ (vec (B.vertices.nth 0).get_or_else (0, 0) (B.vertices.nth 1).get_or_else (0, 0))) :=
sorry

end convex_polygons_common_point_and_rotation_l667_667043


namespace range_sin_x_plus_cos_2x_l667_667766

-- Define the function
def f (x : ℝ) : ℝ := sin x + cos (2 * x)

-- State the main theorem
theorem range_sin_x_plus_cos_2x : set.range f = set.Icc (-2 : ℝ) (9/8 : ℝ) :=
sorry

end range_sin_x_plus_cos_2x_l667_667766


namespace sum_of_distinct_prime_factors_420_l667_667285

theorem sum_of_distinct_prime_factors_420 : (2 + 3 + 5 + 7 = 17) :=
by
  -- given condition on the prime factorization of 420
  have h : ∀ {n : ℕ}, n = 420 → ∀ (p : ℕ), p ∣ n → p.prime → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7,
  by sorry

  -- proof of summation of these distinct primes
  sorry

end sum_of_distinct_prime_factors_420_l667_667285


namespace find_a_g_is_odd_g_is_increasing_l667_667027

-- Define the function f given a parameter a
def f (a : ℝ) (x : ℝ) := 2 * x - a / x

-- Given condition that f(2) = 9/2
axiom f_at_2 (a : ℝ) : f a 2 = 9 / 2

-- Goal 1: Prove the value of a is -1
theorem find_a : ∃ a : ℝ, (f a 2 = 9 / 2) ∧ (a = -1) :=
by {
  use -1,
  split,
  { exact f_at_2 (-1) },
  { refl }
}

-- Define the specific function f(x) with the determined value of a = -1
def g (x : ℝ) := 2 * x + 1 / x

-- Goal 2: Prove the function g(x) is odd
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) :=
by {
  intro x,
  unfold g,
  rw [neg_mul_eq_neg_mul, add_neg_eq_neg_add, neg_div]
}

-- Goal 3: Prove the function g(x) is increasing on (1, +∞)
theorem g_is_increasing : ∀ (x1 x2 : ℝ), 1 < x1 → x1 < x2 → (1 < x2) → g x1 < g x2 :=
by {
  intros x1 x2 h1 h2 h3,
  unfold g,
  sorry -- Provide proof steps for monotonicity here
}

end find_a_g_is_odd_g_is_increasing_l667_667027


namespace rebecca_bought_2_more_bottles_of_water_l667_667151

noncomputable def number_of_more_bottles_of_water_than_tent_stakes
  (T D W : ℕ) 
  (hT : T = 4) 
  (hD : D = 3 * T) 
  (hTotal : T + D + W = 22) : Prop :=
  W - T = 2

theorem rebecca_bought_2_more_bottles_of_water
  (T D W : ℕ) 
  (hT : T = 4) 
  (hD : D = 3 * T) 
  (hTotal : T + D + W = 22) : 
  number_of_more_bottles_of_water_than_tent_stakes T D W hT hD hTotal :=
by 
  sorry

end rebecca_bought_2_more_bottles_of_water_l667_667151


namespace max_median_soda_l667_667614

theorem max_median_soda (n m : ℕ) (h1 : n = 300) (h2 : m = 120) (h3 : ∀ i : ℕ, (1 ≤ i ∧ i ≤ m) → 1 ≤ f i) : 
  median (n / m) (λ i, f i) = 4 :=
by
  sorry

end max_median_soda_l667_667614


namespace cot_30_deg_l667_667812

-- Define the known values and identities
def tan_30_deg := 1 / Real.sqrt 3

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by 
  sorry

end cot_30_deg_l667_667812


namespace find_lowest_score_l667_667189

noncomputable def lowest_score (total_scores : ℕ) (mean_scores : ℕ) (new_total_scores : ℕ) (new_mean_scores : ℕ) (highest_score : ℕ) : ℕ :=
  let total_sum        := mean_scores * total_scores
  let sum_remaining    := new_mean_scores * new_total_scores
  let removed_sum      := total_sum - sum_remaining
  let lowest_score     := removed_sum - highest_score
  lowest_score

theorem find_lowest_score :
  lowest_score 15 90 13 92 110 = 44 :=
by
  unfold lowest_score
  simp [Nat.mul_sub, Nat.sub_apply, Nat.add_comm]
  rfl

end find_lowest_score_l667_667189


namespace evaluate_polynomial_at_4_l667_667255

noncomputable def polynomial_horner (x : ℤ) : ℤ :=
  (((((3 * x + 6) * x - 20) * x - 8) * x + 15) * x + 9)

theorem evaluate_polynomial_at_4 :
  polynomial_horner 4 = 3269 :=
by
  sorry

end evaluate_polynomial_at_4_l667_667255


namespace minimize_f_sin_65_sin_40_l667_667867

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := m^2 * x^2 + (n + 1) * x + 1

theorem minimize_f_sin_65_sin_40 (m n : ℝ) (h₁ : m = Real.sin (65 * Real.pi / 180))
  (h₂ : n = Real.sin (40 * Real.pi / 180)) : 
  ∃ x, x = -1 ∧ (∀ y, f y m n ≥ f (-1) m n) :=
by
  -- Proof to be completed
  sorry

end minimize_f_sin_65_sin_40_l667_667867


namespace lesser_number_is_32_l667_667628

variable (x y : ℕ)

theorem lesser_number_is_32 (h1 : y = 2 * x) (h2 : x + y = 96) : x = 32 := 
sorry

end lesser_number_is_32_l667_667628


namespace david_pushups_difference_l667_667670

-- Definitions based on conditions
def zachary_pushups : ℕ := 44
def total_pushups : ℕ := 146

-- The number of push-ups David did more than Zachary
def david_more_pushups_than_zachary (D : ℕ) := D - zachary_pushups

-- The theorem we need to prove
theorem david_pushups_difference :
  ∃ D : ℕ, D > zachary_pushups ∧ D + zachary_pushups = total_pushups ∧ david_more_pushups_than_zachary D = 58 :=
by
  -- We leave the proof as an exercise or for further filling.
  sorry

end david_pushups_difference_l667_667670


namespace general_formula_a_n_T_n_greater_than_S_n_l667_667931

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n_l667_667931


namespace inequality_satisfied_l667_667327

-- Define the conditions provided in the problem
def cost_per_kg : ℝ := 5
def total_kg : ℝ := 200
def loss_percentage : ℝ := 0.05
def profit_margin : ℝ := 0.20

-- Total cost calculation
def total_cost := total_kg * cost_per_kg

-- After loss in quantity
def after_loss_kg := total_kg * (1 - loss_percentage)

-- Selling price per kg with increase percentage x
def selling_price_per_kg (x : ℝ) := cost_per_kg * (1 + x / 100)

-- Total selling price
def total_selling_price (x : ℝ) := after_loss_kg * selling_price_per_kg(x)

-- The inequality to be proved
def required_inequality (x : ℝ) : Prop :=
  total_selling_price(x) ≥ total_cost * (1 + profit_margin)

-- Statement of the problem in Lean
theorem inequality_satisfied (x : ℝ) : required_inequality x := 
by sorry

end inequality_satisfied_l667_667327


namespace cot_30_eq_sqrt_3_l667_667834

theorem cot_30_eq_sqrt_3 (θ : ℝ) (hθ : θ = 30) (h1 : tan 30 = 1 / real.sqrt 3) : cot 30 = real.sqrt 3 :=
by
  have h2: cot 30 = 1 / (tan 30), by exact cot_def
  have h3: cot 30 = 1 / (1 / real.sqrt 3), by rw [h1, h2]
  have h4: cot 30 = real.sqrt 3, by exact (one_div_div (1 / real.sqrt 3))
  exact h4


end cot_30_eq_sqrt_3_l667_667834


namespace solve_inequality_l667_667223

theorem solve_inequality (x : ℝ) : ((x + 3) ^ 2 < 1) ↔ (-4 < x ∧ x < -2) := by
  sorry

end solve_inequality_l667_667223


namespace general_formula_a_n_T_n_greater_than_S_n_l667_667924

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n_l667_667924


namespace no_such_polynomial_exists_l667_667156

theorem no_such_polynomial_exists :
  ∀ (P : ℤ[X]),
  ¬ ∃ (a b c : ℤ),
      a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
      P.eval a = b ∧
      P.eval b = c ∧
      P.eval c = a :=
by
  sorry

end no_such_polynomial_exists_l667_667156


namespace simplify_expression_l667_667601

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l667_667601


namespace cot_30_deg_l667_667792

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 := by
  -- Define known values of trigonometric functions at 30 degrees
  have h1 : Real.tan (30 * Real.pi / 180) = 1 / Real.sqrt 3 := sorry
  have h2 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

  -- Prove using these known values
  calc
    Real.cot (30 * Real.pi / 180)
        = 1 / Real.tan (30 * Real.pi / 180) : by sorry
    ... = 1 / (1 / Real.sqrt 3) : by sorry
    ... = Real.sqrt 3 : by sorry

end cot_30_deg_l667_667792


namespace minimum_disks_drawn_l667_667544

theorem minimum_disks_drawn (labels : Fin 60 → ℕ) (total_disks : ℕ) :
  (∀ i, 1 ≤ i ∧ i ≤ 60 → labels i = i) →
  (total_disks = ∑ i in Icc 1 60, i) →
  ∃ n : ℕ, n = 750 ∧ ∀ draws : Fin n → Fin 60,
    ∃ l, 15 ≤ ∑ i in Finset.range n, if draws i = l then 1 else 0 :=
by
  sorry

end minimum_disks_drawn_l667_667544


namespace length_PQ_l667_667552

-- Defining the points and lines
def point := (ℝ × ℝ)
def R : point := (10, 4)
def P (a : ℝ) := (a, (13 * a + 7) / 6)
def Q (b : ℝ) := (b, (2 * b - 5) / 7)

-- Defining the midpoint condition
def midpoint (R P Q : point) :=
  R.1 = (P.1 + Q.1) / 2 ∧ R.2 = (P.2 + Q.2) / 2

-- Defining the distance squared (we will use it to simplify sqrt calculation)
noncomputable def dist_sq (P Q : point) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Main theorem
theorem length_PQ :
  ∃ (a b : ℝ), midpoint R (P a) (Q b) ∧ dist_sq (P a) (Q b) = 1370017 / 11064 :=
by
  sorry

end length_PQ_l667_667552


namespace tan_add_pi_over_8_l667_667050

theorem tan_add_pi_over_8 (θ : ℝ) (h : 3 * sin θ + cos θ = sqrt 10) :
  tan (θ + π / 8) - 1 / tan (θ + π / 8) = -14 :=
by
  sorry

end tan_add_pi_over_8_l667_667050


namespace perpendicular_line_through_center_l667_667624

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + 2 * x + y^2 - 3 = 0

-- Define the equation of the line
def line_eq (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the line we want to prove passes through the center of the circle and is perpendicular to the given line
def wanted_line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- Main statement: Prove that the line that passes through the center of the circle and is perpendicular to the given line has the equation x - y + 1 = 0
theorem perpendicular_line_through_center (x y : ℝ) :
  (circle_eq (-1) 0) ∧ (line_eq x y) → wanted_line_eq x y :=
by
  sorry

end perpendicular_line_through_center_l667_667624


namespace coefficient_x2_expansion_l667_667521

theorem coefficient_x2_expansion : 
  coefficient_of_term (expand (2 * x^3 - 1 / x)^6) x^2 = 60 :=
sorry

end coefficient_x2_expansion_l667_667521


namespace angle_CDE_proof_given_conditions_l667_667523

structure GeometryProblem where
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  angle_AEB : ℝ
  angle_BED : ℝ
  angle_BDE : ℝ

theorem angle_CDE_proof_given_conditions (h : GeometryProblem) (h_angle_A_right : h.angle_A = 90)
  (h_angle_B_right : h.angle_B = 90) (h_angle_C_right : h.angle_C = 90) (h_angle_AEB_30 : h.angle_AEB = 30)
  (h_angle_BED_half_BDE : h.angle_BED = 0.5 * h.angle_BDE) : 
  h.angle_BDE = 60 → h.angle_CDE = 120 := 
  by
  sorry

end angle_CDE_proof_given_conditions_l667_667523


namespace x_pow_y_is_neg_216_l667_667492

theorem x_pow_y_is_neg_216 (x y : ℝ) (h : |x + 2y| + (y - 3)^2 = 0) : x^y = -216 := by
  sorry

end x_pow_y_is_neg_216_l667_667492


namespace math_proof_problem_l667_667587

variables {a b c d e f k : ℝ}

theorem math_proof_problem 
  (h1 : a + b + c = d + e + f)
  (h2 : a^2 + b^2 + c^2 = d^2 + e^2 + f^2)
  (h3 : a^3 + b^3 + c^3 ≠ d^3 + e^3 + f^3) :
  (a + b + c + (d + k) + (e + k) + (f + k) = d + e + f + (a + k) + (b + k) + (c + k) ∧
   a^2 + b^2 + c^2 + (d + k)^2 + (e + k)^2 + (f + k)^2 = d^2 + e^2 + f^2 + (a + k)^2 + (b + k)^2 + (c + k)^2 ∧
   a^3 + b^3 + c^3 + (d + k)^3 + (e + k)^3 + (f + k)^3 = d^3 + e^3 + f^3 + (a + k)^3 + (b + k)^3 + (c + k)^3) 
   ∧ 
  (a^4 + b^4 + c^4 + (d + k)^4 + (e + k)^4 + (f + k)^4 ≠ d^4 + e^4 + f^4 + (a + k)^4 + (b + k)^4 + (c + k)^4) := 
  sorry

end math_proof_problem_l667_667587


namespace more_boys_after_initial_l667_667308

theorem more_boys_after_initial (X Y Z : ℕ) (hX : X = 22) (hY : Y = 35) : Z = Y - X :=
by
  sorry

end more_boys_after_initial_l667_667308


namespace exists_c_x0_l667_667367

open Real

namespace PolynomialTheorem

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (γ : V →ₗ[ℝ] ℝ)

axiom gamma_property (f g : V) : γ (f * g) = 0 → γ f = 0 ∨ γ g = 0

theorem exists_c_x0 (γ : V →ₗ[ℝ] ℝ) :
  ∃ (c x0 : ℝ), ∀ f : V, γ f = c * eval x0 f :=
begin
  sorry
end

end PolynomialTheorem

end exists_c_x0_l667_667367


namespace division_multiplication_relation_l667_667352

theorem division_multiplication_relation (h: 7650 / 306 = 25) :
  25 * 306 = 7650 ∧ 7650 / 25 = 306 := 
by 
  sorry

end division_multiplication_relation_l667_667352


namespace piecewise_continuity_l667_667128

def f (a c : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then a*x + 2
  else if x >= -2 then x - 4
  else 3*x - c

theorem piecewise_continuity (a c : ℝ) (h1 : (∀ x, (x > 2) → f a c x = a*x + 2))
(h2 : (∀ x, (-2 ≤ x ∧ x ≤ 2) → f a c x = x - 4))
(h3 : (∀ x, (x < -2) → f a c x = 3*x - c)) 
(h4 : continuous f) 
: a + c = -3 :=
sorry

end piecewise_continuity_l667_667128


namespace point_in_fourth_quadrant_l667_667985

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a = 2) (h2 : 3 = -b) :
  ∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ (a = x) ∧ (b = y) :=
by 
  use [2, -3]
  sorry

end point_in_fourth_quadrant_l667_667985


namespace sum_first_five_terms_arithmetic_seq_l667_667082

theorem sum_first_five_terms_arithmetic_seq
  (a : ℕ → ℕ)
  (h_arith_seq : ∀ n, a n = a 0 + n * (a 1 - a 0))
  (h_a2 : a 2 = 5)
  (h_a4 : a 4 = 9)
  : (Finset.range 5).sum a = 35 := by
  sorry

end sum_first_five_terms_arithmetic_seq_l667_667082


namespace prob_no_decrease_white_in_A_is_correct_l667_667513

-- Define the conditions of the problem
def bagA_white : ℕ := 3
def bagA_black : ℕ := 5
def bagB_white : ℕ := 4
def bagB_black : ℕ := 6

-- Define the probabilities involved
def prob_draw_black_from_A : ℚ := 5 / 8
def prob_draw_white_from_A : ℚ := 3 / 8
def prob_put_white_back_into_A_conditioned_on_white_drawn : ℚ := 5 / 11

-- Calculate the combined probability
def prob_no_decrease_white_in_A : ℚ := prob_draw_black_from_A + prob_draw_white_from_A * prob_put_white_back_into_A_conditioned_on_white_drawn

-- Prove the probability is as expected
theorem prob_no_decrease_white_in_A_is_correct : prob_no_decrease_white_in_A = 35 / 44 := by
  sorry

end prob_no_decrease_white_in_A_is_correct_l667_667513


namespace general_term_sequence_first_n_terms_sum_l667_667038

-- Proof Problem 1
theorem general_term_sequence 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n^2 + 6 * n)
  (h2 : (a 2 + 6) * (a 5 + 6) = (a 10 + 6)^2) :
  ∀ n, a n = 2 * n + 5 := sorry

-- Proof Problem 2
theorem first_n_terms_sum 
  (b a : ℕ → ℝ)
  (h1 : ∀ n, a n = 2 * n + 5)
  (h2 : ∀ n, b n = 1 + 5 / (a n * a (n + 1))) :
  ∀ n, ∑ i in range (n+1), b i = (14 * n^2 + 54 * n) / (14 * n + 49) := sorry

end general_term_sequence_first_n_terms_sum_l667_667038


namespace longest_side_of_similar_triangle_l667_667631

theorem longest_side_of_similar_triangle (a b c p : ℝ) (h1 : a = 8) (h2: b = 10) (h3: c = 12) (h4: p = 150) :
  let x := (p / (a + b + c)) in (c * x) = 60 :=
by
  sorry

end longest_side_of_similar_triangle_l667_667631


namespace counterexample_to_proposition_l667_667654

theorem counterexample_to_proposition (a b : ℤ) (ha : a = -2) (hb : b = -6) : a > b ∧ a^2 ≤ b^2 :=
by
  have h1 : -2 > -6 := by linarith
  have h2 : (-2)^2 = 4 := by norm_num
  have h3 : (-6)^2 = 36 := by norm_num
  have h4 : 4 ≤ 36 := by linarith
  exact ⟨h1, h4⟩

end counterexample_to_proposition_l667_667654


namespace y_intercept_of_line_l667_667269

theorem y_intercept_of_line (x y : ℝ) (eq : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 := 
by
  sorry

end y_intercept_of_line_l667_667269


namespace angle_C_plus_angle_D_l667_667462

open EuclideanGeometry

theorem angle_C_plus_angle_D {A B C D E : Point}
  (h1 : Triangle A B C)
  (h2 : Triangle A B D)
  (h3 : A ≠ B)
  (h4 : A ≠ C)
  (h5 : B ≠ D)
  (h6 : C ≠ D)
  (h7 : A ≠ E ∧ B ≠ E ∧ C ≠ E ∧ D ≠ E )
  (h8 : B ≠ E ∧ D ≠ E)
  (h9 : A ≠ E)
  (h10 : is_perpendicular BD AC)
  (h11 : AB = AC ∧ AC = BD)
  (h12 : E = intersection_point_of BD AC) :
  measure_angle C + measure_angle D = 135 :=
sorry

end angle_C_plus_angle_D_l667_667462


namespace area_of_square_IJKL_l667_667163

noncomputable def area_of_inner_square : ℝ :=
  let side_length_WXYZ := 10
  let WI := 2
  let y := Real.sqrt 96 - 2
  in (y * y)

-- Theorem statement
theorem area_of_square_IJKL (side_length_WXYZ : ℝ) (WI : ℝ)
  (h1 : side_length_WXYZ = 10) (h2 : WI = 2) :
  area_of_inner_square = 100 - 4 * Real.sqrt 96 := by 
sorry

end area_of_square_IJKL_l667_667163


namespace common_chord_length_l667_667022

-- Circles: x^2 + y^2 - 2x + 10y - 24 = 0, and x^2 + y^2 + 2x + 2y - 8 = 0
noncomputable def circle1 : set (ℝ × ℝ) :=
  { p | ∃ (x y : ℝ), x^2 + y^2 - 2*x + 10*y - 24 = 0 }

noncomputable def circle2 : set (ℝ × ℝ) :=
  { p | ∃ (x y : ℝ), x^2 + y^2 + 2*x + 2*y - 8 = 0 }

-- Prove the length of the common chord
theorem common_chord_length : 
  ∃ l : ℝ, l = 2*Real.sqrt(5) ∧
  ∃ p1 p2, p1 ∈ circle1 ∧ p1 ∈ circle2 ∧ 
           p2 ∈ circle1 ∧ p2 ∈ circle2 ∧ 
           dist p1 p2 = l :=
begin
  sorry
end

end common_chord_length_l667_667022


namespace distance_to_x_axis_l667_667953

noncomputable def hyperbola := set_of (λ p : ℝ × ℝ, (p.1^2 / 9) - (p.2^2 / 16) = 1)

def f1 : ℝ × ℝ := (-5, 0)
def f2 : ℝ × ℝ := (5, 0)

def on_hyperbola (M : ℝ × ℝ) : Prop :=
  (M.1^2 / 9) - (M.2^2 / 16) = 1

def orthogonality_condition (M : ℝ × ℝ) : Prop :=
  let mf1 := λ M : ℝ × ℝ, ((M.1 + 5)^2 + M.2^2)^(1/2)
  let mf2 := λ M : ℝ × ℝ, ((M.1 - 5)^2 + M.2^2)^(1/2)
  ((M.1 + 5) * (M.1 - 5)) + (M.2 * M.2) = 0

theorem distance_to_x_axis (M : ℝ × ℝ) (hM : on_hyperbola M) (hOrth : orthogonality_condition M) : M.2 = 16 / 5 :=
by sorry

end distance_to_x_axis_l667_667953


namespace find_lowest_score_l667_667182

variable (scores : Fin 15 → ℕ)
variable (highest_score : ℕ := 110)

-- Conditions
def arithmetic_mean_15 := (∑ i, scores i) / 15 = 90
def mean_without_highest_lowest := (∑ i in Finset.erase (Finset.erase (Finset.univ : Finset (Fin 15)) (Fin.mk 0 sorry)) (Fin.mk 14 sorry)) / 13 = 92
def highest_value := scores (Fin.mk 14 sorry) = highest_score

-- The condition includes univ meaning all elements in Finset and we are removing the first(lowest) and last(highest) elements (0 and 14).

-- The lowest score
def lowest_score (scores : Fin 15 → ℕ) : ℕ :=
  1350 - 1196 - highest_score

theorem find_lowest_score : lowest_score scores = 44 :=
by
  sorry

end find_lowest_score_l667_667182


namespace find_num_managers_l667_667322

variable (num_associates : ℕ) (avg_salary_managers avg_salary_associates avg_salary_company : ℚ)
variable (num_managers : ℚ)

-- Define conditions based on given problem
def conditions := 
  num_associates = 75 ∧
  avg_salary_managers = 90000 ∧
  avg_salary_associates = 30000 ∧
  avg_salary_company = 40000

-- Proof problem statement
theorem find_num_managers (h : conditions num_associates avg_salary_managers avg_salary_associates avg_salary_company) :
  num_managers = 15 :=
sorry

end find_num_managers_l667_667322


namespace cot_30_eq_sqrt_3_l667_667827

theorem cot_30_eq_sqrt_3 :
  let θ := 30.0
  let cos θ := (real.sqrt 3) / 2
  let sin θ := 1 / 2
  let cot θ := cos θ / sin θ
  cot θ = real.sqrt 3 := by
sorry

end cot_30_eq_sqrt_3_l667_667827


namespace no_19_distinct_pos_integers_with_same_digit_sum_l667_667770

theorem no_19_distinct_pos_integers_with_same_digit_sum (S : Finset ℕ) :
  S.card = 19 ∧ (∀ n ∈ S, n > 0) ∧ (∑ n in S, n = 1999) → ¬(∃ k, ∀ n ∈ S, digit_sum n = k) :=
by
  sorry

end no_19_distinct_pos_integers_with_same_digit_sum_l667_667770


namespace point_P_x_coordinate_l667_667086

variable {P : Type} [LinearOrderedField P]

-- Definitions from the conditions
def line_equation (x : P) : P := 0.8 * x
def y_coordinate_P : P := 6
def x_coordinate_P : P := 7.5

-- Theorems to prove that the x-coordinate of P is 7.5.
theorem point_P_x_coordinate (x : P) :
  line_equation x = y_coordinate_P → x = x_coordinate_P :=
by
  intro h
  sorry

end point_P_x_coordinate_l667_667086


namespace pressure_increases_when_block_submerged_l667_667290

theorem pressure_increases_when_block_submerged 
  (P0 : ℝ) (ρ : ℝ) (g : ℝ) (h0 h1 : ℝ) :
  h1 > h0 → 
  (P0 + ρ * g * h1) > (P0 + ρ * g * h0) :=
by
  intros h1_gt_h0
  sorry

end pressure_increases_when_block_submerged_l667_667290


namespace divisor_correct_l667_667579

theorem divisor_correct : ∃ (x : ℕ), 265 = 12 * x + 1 ∧ x = 22 :=
by
  use 22
  split
  · exact rfl
  · exact rfl

end divisor_correct_l667_667579


namespace collinear_M_N_P_l667_667065

-- Define the geometrical objects and properties
variables {A B C I D E P M N : Type*}
variables {AB AC BC : ℝ} [Inhabited {AB, AC, BC}]
variables {I_incenter : Prop}
variables {incircle_tangency_points : Prop}
variables {intersection_P : Prop}
variables {midpoints_M_and_N : Prop}

-- Assumptions
axiom AB_gt_AC : AB > AC
axiom I_is_incenter : I_incenter
axiom incircle_tangent_at_D_E : incircle_tangency_points
axiom P_is_intersection_AI_DE : intersection_P
axiom M_midpoint_BC_N_midpoint_AB : midpoints_M_and_N

-- Theorem statement
theorem collinear_M_N_P : I_incenter ∧ incircle_tangency_points ∧ intersection_P ∧ midpoints_M_and_N → 
  (collinear ℝ {M, N, P}) :=
begin
  sorry
end

end collinear_M_N_P_l667_667065


namespace question_condition_answer_l667_667052

theorem question_condition_answer (x y : ℝ) (h1 : 3 < x) (h2 : x < 6) (h3 : 6 < y) (h4 : y - x = 3) : ∃ (z : ℝ), z > y ∧ z = 9 :=
by {
  use 9,
  split,
  exact h3.trans (by norm_num),
  refl,
}

end question_condition_answer_l667_667052


namespace range_of_m_l667_667042

noncomputable def C1 : (ℝ × ℝ) → ℝ := λ (p : ℝ × ℝ), p.1^2 + p.2^2
noncomputable def C2 : (ℝ × ℝ) → ℝ := λ (p : ℝ × ℝ), p.1^2 + p.2^2 - 2 * p.1 - 4 * p.2 - 20

theorem range_of_m 
  (m : ℝ)
  (tangent_condition : ∀ p, C1 p = m^2 ∧ C2 p = 0 → (5 - real.sqrt 5 < m ∧ m < 5 + real.sqrt 5)) 
  : 5 - real.sqrt 5 < m ∧ m < 5 + real.sqrt 5 :=
sorry

end range_of_m_l667_667042


namespace lowest_score_of_15_scores_l667_667175

theorem lowest_score_of_15_scores (scores : List ℝ) (h_len : scores.length = 15) (h_mean : (scores.sum / 15) = 90)
    (h_mean_13 : ((scores.erase 110).erase (scores.head)).sum / 13 = 92) (h_highest : scores.maximum = some 110) :
    scores.minimum = some 44 :=
by
    sorry

end lowest_score_of_15_scores_l667_667175


namespace total_spider_legs_l667_667301

-- Definition of the number of spiders
def number_of_spiders : ℕ := 5

-- Definition of the number of legs per spider
def legs_per_spider : ℕ := 8

-- Theorem statement to prove the total number of spider legs
theorem total_spider_legs : number_of_spiders * legs_per_spider = 40 :=
by 
  -- We've planned to use 'sorry' to skip the proof
  sorry

end total_spider_legs_l667_667301


namespace problem_increasing_intervals_problem_area_triangle_l667_667482

noncomputable def f (x : ℝ) : ℝ := (sqrt 3 * sin x) * cos x + (cos x) ^ 2

theorem problem_increasing_intervals :
  (∀ (k : ℤ),
  ∀ (x : ℝ), (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6) → ((f x) = (sqrt 3 / 2) * sin(2 * x) + 1 / 2 * cos(2 * x) + 1 / 2)) ∧ 
  ∀ (k : ℤ),
  ∀ (x : ℝ), ((k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3) → (f x = (sqrt 3 / 2) * sin(2 * x) + 1 / 2 * cos(2 * x) + 1 / 2)) := sorry

noncomputable def triangle_area (a b C : ℝ) : ℝ := 1 / 2 * a * b * sin C

theorem problem_area_triangle :
  let A := π / 6 in
  let a := 1 in
  let b := sqrt 2 in
  let f_A2 := 1 / 2 + sqrt 3 / 2 in
  (f (A / 2) = f_A2) → 
  (sin (A + π / 6) = sqrt 3 / 2) → 
  let B := π / 4 in 
  let C := 7 * π / 12 in
  (triangle_area a b C = (1 + sqrt 3) / 4) ∨ 
  (triangle_area a b (π / 12) = (sqrt 3 - 1) / 4) := sorry

end problem_increasing_intervals_problem_area_triangle_l667_667482


namespace chess_competition_l667_667138

theorem chess_competition : ∃ n : ℕ, n > 5 ∧ 
  (let five_lost_games := 5
   let remaining_won_games := (n - 5) * 3
   let total_games := n * (n - 1) / 2
   in
   5 * (n - 3) + remaining_won_games = five_lost_games + (n - 5) * (n - 4))
   ∧ n = 12 :=
by
  sorry

end chess_competition_l667_667138


namespace min_value_of_D_l667_667551

noncomputable def D (x a : ℝ) : ℝ :=
  Real.sqrt ((x - a) ^ 2 + (Real.exp x - 2 * Real.sqrt a) ^ 2) + a + 2

theorem min_value_of_D (e : ℝ) (h_e : e = 2.71828) :
  ∀ a : ℝ, ∃ x : ℝ, D x a = Real.sqrt 2 + 1 :=
sorry

end min_value_of_D_l667_667551


namespace leak_empty_time_l667_667303

variable (A L : ℝ)

theorem leak_empty_time (hA: A = 1/3) (hAL: A - L = 1/9) : 1 / L = 4.5 :=
by
  -- substitute hA in hAL, solve for L
  have L_def : L = 1/3 - 1/9 := by
    rw [hA]
    rw sub_eq_add_neg
    norm_num
  rw L_def
  norm_num
  sorry -- the intermediate steps of solving can be omitted, skipping to the result

end leak_empty_time_l667_667303


namespace lowest_score_of_15_scores_l667_667178

theorem lowest_score_of_15_scores (scores : List ℝ) (h_len : scores.length = 15) (h_mean : (scores.sum / 15) = 90)
    (h_mean_13 : ((scores.erase 110).erase (scores.head)).sum / 13 = 92) (h_highest : scores.maximum = some 110) :
    scores.minimum = some 44 :=
by
    sorry

end lowest_score_of_15_scores_l667_667178


namespace shopper_savings_percentage_l667_667718

theorem shopper_savings_percentage
  (amount_saved : ℝ) (final_price : ℝ)
  (h_saved : amount_saved = 3)
  (h_final : final_price = 27) :
  (amount_saved / (final_price + amount_saved)) * 100 = 10 := 
by
  sorry

end shopper_savings_percentage_l667_667718


namespace cot_30_deg_l667_667815

-- Define the known values and identities
def tan_30_deg := 1 / Real.sqrt 3

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by 
  sorry

end cot_30_deg_l667_667815


namespace g_1993_at_2_l667_667365

def g (x : ℚ) : ℚ := (2 + x) / (1 - 4 * x^2)

def g_n : ℕ → ℚ → ℚ 
| 0     => id
| (n+1) => λ x => g (g_n n x)

theorem g_1993_at_2 : g_n 1993 2 = 65 / 53 := 
  sorry

end g_1993_at_2_l667_667365


namespace hyperbola_vertex_distance_l667_667841

theorem hyperbola_vertex_distance : 
  (∃ a b : ℝ, a^2 = 144 ∧ b^2 = 64 ∧ ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (distance (a, 0) (-a, 0) = 24)) :=
begin
  use [12, 8],
  split,
  { norm_num },
  split,
  { norm_num },
  intros x y h,
  simp only [distance, real.dist_eq, add_eq_zero_iff_eq_neg, sub_zero, eq_self_iff_true, and_true, zero_eq_mul] at h,
  norm_num,
  linarith,
end

end hyperbola_vertex_distance_l667_667841


namespace minimum_magnitude_l667_667483

open Classical
open Real

variable {a b c : ℝ^3}
variable {x y : ℝ}

def unit_vector (v : ℝ^3) : Prop := ∥v∥ = 1

def dot_product_half (u v : ℝ^3) : Prop := (u • v) = 1 / 2

def positive_real (t : ℝ) : Prop := t > 0

def sum_to_two (x y : ℝ) : Prop := x + y = 2

def linear_combination (c a b : ℝ^3) (x y : ℝ) : Prop :=
  c = x • a + y • b

theorem minimum_magnitude (a b c : ℝ^3) (x y : ℝ)
  (a_unit: unit_vector a)
  (b_unit: unit_vector b)
  (dot_prod: dot_product_half a b)
  (x_pos: positive_real x)
  (y_pos: positive_real y)
  (sum_eq_two: sum_to_two x y)
  (lin_comb: linear_combination c a b x y) :
  ∥c∥ = sqrt 3 :=
by
  sorry

end minimum_magnitude_l667_667483


namespace min_area_ADBE_l667_667450

-- Given conditions in Lean:

def parabola_focus := (1/2, 0)
def parabola_eq (x y : ℝ) := y^2 = 4 * x
def is_focus (p : ℝ × ℝ) := p = parabola_focus

-- Definitions for the problem:

variable (x y : ℝ)
def chord_through_focus (k : ℝ) := y = k * (x - 1/2)
def perpendicular_chord_through_focus (k : ℝ) := y = -1/k * (x - 1/2)

-- Statement of the proof problem:
theorem min_area_ADBE (k : ℝ) (h : 0 < k) :
  let AB := sqrt ((k^2 + 1) * (((x + x)^2 - 4 * x * x)))
  let CD := sqrt (((1/k^2) + 1) * (((x + x)^2 - 4 * x * x)))
  (AB * CD) ≥ 32 :=
sorry

end min_area_ADBE_l667_667450


namespace y_ordering_l667_667446

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 3

-- Define the points and their y-values
def y_val1 : ℝ := quadratic_function (-3)
def y_val2 : ℝ := quadratic_function 1
def y_val3 : ℝ := quadratic_function (-1 / 2)

theorem y_ordering :
  y_val2 < y_val3 ∧ y_val3 < y_val1 :=
by
  sorry

end y_ordering_l667_667446


namespace square_number_divisible_by_five_between_50_and_150_l667_667837

theorem square_number_divisible_by_five_between_50_and_150 (x : ℕ) : 
  (∃ m : ℕ, x = m^2) ∧ (x % 5 = 0) ∧ (50 < x ∧ x < 150) → x = 100 :=
by
  intro h,
  sorry

end square_number_divisible_by_five_between_50_and_150_l667_667837


namespace cot_30_eq_sqrt_3_l667_667820

theorem cot_30_eq_sqrt_3 :
  let θ := 30.0
  let cos θ := (real.sqrt 3) / 2
  let sin θ := 1 / 2
  let cot θ := cos θ / sin θ
  cot θ = real.sqrt 3 := by
sorry

end cot_30_eq_sqrt_3_l667_667820


namespace more_boys_after_initial_l667_667309

theorem more_boys_after_initial (X Y Z : ℕ) (hX : X = 22) (hY : Y = 35) : Z = Y - X :=
by
  sorry

end more_boys_after_initial_l667_667309


namespace order_of_numbers_l667_667212

theorem order_of_numbers (a b c : ℝ) (ha : a = 7^0.3) (hb : b = 0.3^7) (hc : c = Real.log 0.3) : a > b ∧ b > c :=
by
  rw [ha, hb, hc]
  -- The proof follows but is not required as per the instructions
  sorry

end order_of_numbers_l667_667212


namespace find_angle_A_find_area_l667_667942

-- Define triangle structure for convenience
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ)
  (area : ℝ)

-- Define the conditions as hypotheses
variables {a b c angle_A angle_B angle_C area r : ℝ}

-- Condition 1
def condition1 (a b c : ℝ) (angle_A angle_B : ℝ) : Prop :=
  a * Real.cos angle_B - b * Real.cos angle_A = c - b

-- Condition 2
def condition2 (angle_A angle_B angle_C : ℝ) : Prop :=
  Real.tan angle_A + Real.tan angle_B + Real.tan angle_C - Real.sqrt 3 * Real.tan angle_B * Real.tan angle_C = 0

-- Condition 3
def condition3 (a b c angle_A angle_B angle_C : ℝ) : Prop :=
  area = 0.5 * a * (b * Real.sin angle_B + c * Real.sin angle_C - a * Real.sin angle_A)

-- Part 1: Prove A = π/3 given conditions 1, 2, and 3
theorem find_angle_A (h1 : condition1 a b c angle_A angle_B)
                     (h2 : condition2 angle_A angle_B angle_C)
                     (h3 : condition3 a b c angle_A angle_B angle_C) :
  angle_A = Real.pi / 3 :=
  sorry

-- Part 2: Prove area = 11*sqrt(3) given a = 8 and r = sqrt(3)
theorem find_area (h3 : condition3 8 b c angle_A angle_B angle_C)
                  (h_r : r = Real.sqrt 3) :
  area = 11 * Real.sqrt 3 :=
  sorry

end find_angle_A_find_area_l667_667942


namespace cot_30_eq_sqrt_3_l667_667809

theorem cot_30_eq_sqrt_3 (h : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3) : 
  Real.cot (Real.pi / 6) = Real.sqrt 3 :=
by
  sorry

end cot_30_eq_sqrt_3_l667_667809


namespace consecutive_even_sum_l667_667225

theorem consecutive_even_sum (N S : ℤ) (m : ℤ) 
  (hk : 2 * m + 1 > 0) -- k is the number of consecutive even numbers, which is odd
  (h_sum : (2 * m + 1) * N = S) -- The condition of the sum
  (h_even : N % 2 = 0) -- The middle number is even
  : (∃ k : ℤ, k = 2 * m + 1 ∧ k > 0 ∧ (k * N / 2) = S/2 ) := 
  sorry

end consecutive_even_sum_l667_667225


namespace find_added_number_l667_667993

theorem find_added_number (R D Q X : ℕ) (hR : R = 5) (hD : D = 3 * Q) (hDiv : 113 = D * Q + R) (hD_def : D = 3 * R + X) : 
  X = 3 :=
by
  -- Provide the conditions as assumptions
  sorry

end find_added_number_l667_667993


namespace distance_run_when_heard_blast_l667_667334

-- Definitions based on the conditions
def fuse_time : ℝ := 20
def reaction_time : ℝ := 2
def run_rate_yards_per_sec : ℝ := 10
def sound_rate_feet_per_sec : ℝ := 1100

-- Conversion between yards and feet
def yard_to_feet : ℝ := 3

-- Position functions
def position_powderman (t : ℝ) : ℝ := run_rate_yards_per_sec * yard_to_feet * (t - reaction_time)
def position_sound (t : ℝ) : ℝ := sound_rate_feet_per_sec * (t - fuse_time)

theorem distance_run_when_heard_blast :
  let t := 21940 / 1070 in
  let distance_yards := position_powderman t / yard_to_feet in
  distance_yards = 185 :=
by
  let t := 21940 / 1070
  let distance_yards := position_powderman t / yard_to_feet
  have h : distance_yards = 185, sorry
  exact h

end distance_run_when_heard_blast_l667_667334


namespace calculate_expression_l667_667752

theorem calculate_expression : (4 + Real.sqrt 6) * (4 - Real.sqrt 6) = 10 := by
  sorry

end calculate_expression_l667_667752


namespace jack_handing_in_amount_l667_667537

theorem jack_handing_in_amount :
  let total_100_bills := 2 * 100
  let total_50_bills := 1 * 50
  let total_20_bills := 5 * 20
  let total_10_bills := 3 * 10
  let total_5_bills := 7 * 5
  let total_1_bills := 27 * 1
  let total_notes := total_100_bills + total_50_bills + total_20_bills + total_10_bills + total_5_bills + total_1_bills
  let amount_in_till := 300
  let amount_to_hand_in := total_notes - amount_in_till
  amount_to_hand_in = 142 := by
  sorry

end jack_handing_in_amount_l667_667537


namespace set_union_proof_l667_667964

  open Set

  def M : Set ℕ := {0, 1, 3}
  def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 3 * a}

  theorem set_union_proof : M ∪ N = {0, 1, 3, 9} :=
  by
    sorry
  
end set_union_proof_l667_667964


namespace train_speed_in_km_per_hr_l667_667724

def train_length : ℝ := 116.67 -- length of the train in meters
def crossing_time : ℝ := 7 -- time to cross the pole in seconds

theorem train_speed_in_km_per_hr : 
  (train_length / 1000) / (crossing_time / 3600) = 60 := by
  sorry

end train_speed_in_km_per_hr_l667_667724


namespace cylinder_quadrilateral_intersection_l667_667259

-- Definitions for the intersection types for the geometric solids.
def can_intersect_plane_as_quadrilateral (solid : Type) : Prop :=
  solid = Cylinder

theorem cylinder_quadrilateral_intersection :
  (∀ solid, can_intersect_plane_as_quadrilateral solid ↔ solid = Cylinder) :=
begin
  intros solid,
  split,
  { -- Prove that if a solid can intersect plane as a quadrilateral, then it is a cylinder
    intro h,
    exact h
  },
  { -- Prove that a cylinder can intersect plane as a quadrilateral
    intro h,
    rw h,
    refl,
  }
end

end cylinder_quadrilateral_intersection_l667_667259


namespace trigonometric_identity_proof_l667_667865

theorem trigonometric_identity_proof 
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h : tan α = (1 + sin β) / cos β) :
  2 * α - β = π / 2 :=
by
  sorry

end trigonometric_identity_proof_l667_667865


namespace part1_part2_part3_l667_667468

def f (x : ℝ) (k : ℝ) : ℝ := log 4 (4^x + 1) + k * x

-- Part 1: Prove k = -1/2
theorem part1 (k : ℝ) :
  (∀ x : ℝ, f (-x) k = f x k) → k = -1/2 :=
by sorry

-- Part 2: Prove when the equation has no real roots
theorem part2 (a : ℝ) :
  (∀ x : ℝ, log 4 (4^x + 1) - 1/2 * x = 1/2 * x + a) → a ≤ 0 :=
by sorry

-- Part 3: Determine existence of m such that minimum of h(x) is 0
def h (x : ℝ) (m : ℝ) : ℝ := 4^(log 4 (4^x + 1) - 1/2 * x + 1/2 * x) + m * 2^x - 1

theorem part3 : ∃ (m : ℝ), (∃ (x_min : ℝ), (x_min ∈ [0, log 2 3]) ∧ ∀ x ∈ [0, log 2 3], h x m ≥ h x_min m) ∧ h x_min m = 0 :=
by sorry

end part1_part2_part3_l667_667468


namespace n_squared_plus_n_plus_1_is_perfect_square_l667_667418

theorem n_squared_plus_n_plus_1_is_perfect_square (n : ℕ) :
  (∃ k : ℕ, n^2 + n + 1 = k^2) ↔ n = 0 :=
by
  sorry

end n_squared_plus_n_plus_1_is_perfect_square_l667_667418


namespace prime_p_in_range_l667_667982

theorem prime_p_in_range (p : ℕ) (prime_p : Nat.Prime p) 
    (h : ∃ a b : ℤ, a * b = -530 * p ∧ a + b = p) : 43 < p ∧ p ≤ 53 := 
sorry

end prime_p_in_range_l667_667982


namespace evaluate_polynomial_at_4_l667_667254

noncomputable def polynomial_horner (x : ℤ) : ℤ :=
  (((((3 * x + 6) * x - 20) * x - 8) * x + 15) * x + 9)

theorem evaluate_polynomial_at_4 :
  polynomial_horner 4 = 3269 :=
by
  sorry

end evaluate_polynomial_at_4_l667_667254


namespace performance_major_applicants_estimate_l667_667210

-- Define the average of the candidate numbers
def avg_candidate_number (sum : ℕ) (count : ℕ) : ℕ :=
  sum / count

-- Define the estimate of the total number of candidates
def estimate_total_candidates (avg : ℕ) : ℕ :=
  2 * avg

theorem performance_major_applicants_estimate (sum_candidates : ℕ) (num_samples : ℕ)
  (h_sum : sum_candidates = 25025) (h_samples : num_samples = 50) :
  estimate_total_candidates (avg_candidate_number sum_candidates num_samples) ≈ 1000 :=
by
  have avg := avg_candidate_number sum_candidates num_samples
  have estimate := estimate_total_candidates avg
  calc 
    estimate
        ≈ 1001 : by sorry
    ... = 1000 : by sorry

end performance_major_applicants_estimate_l667_667210


namespace sum_a4_a5_a6_l667_667083

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (h_a5 : a 5 = 21)

theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = 63 := by
  sorry

end sum_a4_a5_a6_l667_667083


namespace common_area_proof_l667_667698

noncomputable def commonArea (rectWidth rectHeight circleRadius : ℝ) : ℝ :=
  if rectWidth = 10 ∧ rectHeight = 4 ∧ circleRadius = 5 then 
    40 + 4 * Real.pi 
  else 
    0

theorem common_area_proof :
  commonArea 10 4 5 = 40 + 4 * Real.pi :=
by
  unfold commonArea
  rw [if_pos]
  exact and.intro rfl (and.intro rfl rfl)
  sorry

end common_area_proof_l667_667698


namespace locus_of_right_angle_vertices_l667_667434

-- Define the geometric constructs
variable (A B C : Point)
variable (Q : set Point) -- The locus set we need to prove
variable (segments : segment BC)

-- The locus of vertices of right angle ADE 
theorem locus_of_right_angle_vertices
  (E_in_BC : ∀ E, E ∈ BC) : 
  Q = {D | ∃ R ∈ BC, ∃ midpoint : Point, midpoint = (A + R) / 2 ∧ D ∈ sphere midpoint ((A - midpoint).norm)} :=
begin
  sorry
end

end locus_of_right_angle_vertices_l667_667434


namespace log_and_exponent_conditions_l667_667976

theorem log_and_exponent_conditions (a b : ℝ) (h1 : log 2 a < 0) (h2 : (1 / 2)^b > 1) : 0 < a ∧ a < 1 ∧ b < 0 :=
by
  sorry

end log_and_exponent_conditions_l667_667976


namespace length_of_segment_AB_l667_667064

theorem length_of_segment_AB 
  (ABC : Type) [triangle ABC] 
  (A B C : ABC) 
  (angle_A_eq_90 : angle A = 90)
  (tan_C_eq_3 : tan (angle C) = 3)
  (BC_eq_150 : length BC = 150) :
  length AB = 45 * real.sqrt 10 := 
sorry

end length_of_segment_AB_l667_667064


namespace smallest_norm_of_u_l667_667553

noncomputable def vector : Type := ℝ × ℝ

open Real

theorem smallest_norm_of_u (u : vector) (h : ‖u + (4, 2)‖ = 10) : 
  ∃ c : ℝ, u = c • (4, 2) ∧ ‖u‖ = 10 - 2 * sqrt 5 :=
sorry

end smallest_norm_of_u_l667_667553


namespace trigonometric_identity_l667_667018

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α)) / 
  (Real.cos (3 * Real.pi / 2 - α) + 2 * Real.cos (-Real.pi + α)) = -2 / 5 := 
by
  sorry

end trigonometric_identity_l667_667018


namespace general_formula_a_n_T_n_greater_S_n_l667_667868

-- Define the conditions for the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ (n : ℕ), a n = a1 + (n - 1) * d

def b_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_of_first_n_terms (a S : ℕ → ℕ) :=
  ∀ (n k : ℕ), k ≤ n → S n = (n * (a 1 + a n)) / 2

-- Given S4 = 32 and T3 = 16
def S_4_equals_32 (S : ℕ → ℕ) : Prop := S 4 = 32
def T_3_equals_16 (T : ℕ → ℕ) : Prop := T 3 = 16

-- Prove that the general formula for a_n is 2n + 3
theorem general_formula_a_n (a : ℕ → ℕ) (h : arithmetic_sequence a) : 
  a = λ n, 2 * n + 3 := 
sorry

-- Prove that for n > 5, T_n > S_n
theorem T_n_greater_S_n (a b S T: ℕ → ℕ) 
  (ha : arithmetic_sequence a) 
  (hb : b_sequence a b)
  (hS : sum_of_first_n_terms a S)
  (hT : sum_of_first_n_terms b T)
  (hS4 : S_4_equals_32 S)
  (hT3 : T_3_equals_16 T) 
  (n : ℕ) : n > 5 → T n > S n :=
sorry

end general_formula_a_n_T_n_greater_S_n_l667_667868


namespace minimum_room_size_for_table_l667_667348

theorem minimum_room_size_for_table (S : ℕ) :
  (∃ S, S ≥ 13) := sorry

end minimum_room_size_for_table_l667_667348


namespace harry_morning_routine_l667_667398

theorem harry_morning_routine : 
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  buy_time + read_and_eat_time = 45 :=
by
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  show buy_time + read_and_eat_time = 45
  sorry

end harry_morning_routine_l667_667398


namespace domain_of_g_l667_667762

noncomputable def g (x : ℝ) := real.sqrt (2 - real.sqrt (5 - real.sqrt (4 - x)))

theorem domain_of_g : {x : ℝ | ∃ y, g x = y} = {x : ℝ | x ≤ 3} :=
by {
  sorry
}

end domain_of_g_l667_667762


namespace atleast_one_bellini_work_l667_667742

-- Definitions for the caskets and statements
def gold_casket_statement : Prop := ∀ s : Prop, (s ↔ (s = son_bellini ∨ s = bellini))
def silver_casket_statement : Prop := ∀ g : Prop, (g ↔ (¬(g = son_bellini) ∨ g = bellini))

-- Define son_bellini and bellini as distinct propositions
constants son_bellini bellini : Prop

-- The main theorem to prove the existence of at least one Bellini work
theorem atleast_one_bellini_work 
  (gold_statement : gold_casket_statement)
  (silver_statement : silver_casket_statement) : 
  bellini ∨ ∃ x, x = bellini :=
sorry


end atleast_one_bellini_work_l667_667742


namespace seventh_term_arithmetic_sequence_l667_667226

theorem seventh_term_arithmetic_sequence (a d : ℚ)
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 20)
  (h2 : a + 5 * d = 8) :
  a + 6 * d = 28 / 3 := 
sorry

end seventh_term_arithmetic_sequence_l667_667226


namespace cot_30_deg_l667_667795

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 := by
  -- Define known values of trigonometric functions at 30 degrees
  have h1 : Real.tan (30 * Real.pi / 180) = 1 / Real.sqrt 3 := sorry
  have h2 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

  -- Prove using these known values
  calc
    Real.cot (30 * Real.pi / 180)
        = 1 / Real.tan (30 * Real.pi / 180) : by sorry
    ... = 1 / (1 / Real.sqrt 3) : by sorry
    ... = Real.sqrt 3 : by sorry

end cot_30_deg_l667_667795


namespace find_theta_omega_find_x0_l667_667691

theorem find_theta_omega (ω : ℝ) (θ : ℝ) (h₀ : ω > 0) (h₁ : 0 ≤ θ ∧ θ ≤ π/2) 
  (h₂ : 2 * Real.cos θ = sqrt 3) (h₃ : π = 2 * π / ω) : 
  θ = π / 6 ∧ ω = 2 :=
sorry

theorem find_x0 (x₀ : ℝ) (ω : ℝ := 2) (θ : ℝ := π / 6) 
  (hx₀ : π / 2 ≤ x₀ ∧ x₀ ≤ π) 
  (h₄ : 2 * Real.cos (2 * (2 * x₀ - π/2) + π/6) = sqrt 3) : 
  x₀ = 2 * π / 3 ∨ x₀ = 3 * π / 4 :=
sorry

end find_theta_omega_find_x0_l667_667691


namespace cot_30_eq_sqrt3_l667_667777

theorem cot_30_eq_sqrt3 : Real.cot (π / 6) = Real.sqrt 3 := by
  let tan_30 := 1 / Real.sqrt 3
  have tan_30_eq : Real.tan (π / 6) = tan_30 := sorry
  have cot_def : ∀ x, Real.cot x = 1 / Real.tan x := sorry
  rw [cot_def (π / 6), tan_30_eq]
  simp
  sorry

end cot_30_eq_sqrt3_l667_667777


namespace two_lines_intersecting_skew_lines_are_possibly_intersecting_l667_667657

-- Define the concept of lines intersecting with skew lines
def intersects_skew_lines (l₁ l₂ k₁ k₂ : Line) : Prop :=
  (l₁ ≠ k₁) ∧ (l₁ ≠ k₂) ∧ (l₂ ≠ k₁) ∧ (l₂ ≠ k₂) ∧
  (∃ p₁, p₁ ∈ l₁ ∧ p₁ ∈ k₁) ∧ (∃ p₂, p₂ ∈ l₂ ∧ p₂ ∈ k₂) ∧
  skew k₁ k₂

-- The proof statement translating the math proof problem
theorem two_lines_intersecting_skew_lines_are_possibly_intersecting
  {l₁ l₂ k₁ k₂ : Line} 
  (h : intersects_skew_lines l₁ l₂ k₁ k₂) :
  (∃ q, q ∈ l₁ ∧ q ∈ l₂) ∨ (skew l₁ l₂) :=
sorry

end two_lines_intersecting_skew_lines_are_possibly_intersecting_l667_667657


namespace harry_morning_routine_l667_667396

variable (t1 t2 : ℕ)
variable (h1 : t1 = 15)
variable (h2 : t2 = 2 * t1)

theorem harry_morning_routine : t1 + t2 = 45 := by
  sorry

end harry_morning_routine_l667_667396


namespace leftover_potatoes_l667_667316

theorem leftover_potatoes (fries_per_potato : ℕ) (total_potatoes : ℕ) (required_fries : ℕ)
    (h1 : fries_per_potato = 25) (h2 : total_potatoes = 15) (h3 : required_fries = 200) :
    (total_potatoes - required_fries / fries_per_potato) = 7 :=
sorry

end leftover_potatoes_l667_667316


namespace difference_of_elements_in_set_l667_667341

-- Define the set and its properties
variable (S : Set ℤ)
variable (h1 : ∀ x ∈ S, 2 * x ∈ S)
variable (h2 : ∀ x y ∈ S, x + y ∈ S)
variable (h3 : ∃ x ∈ S, x > 0)
variable (h4 : ∃ y ∈ S, y < 0)

-- The theorem to be proved
theorem difference_of_elements_in_set : ∀ x y ∈ S, (x - y) ∈ S :=
by
  -- Proof goes here
  sorry

end difference_of_elements_in_set_l667_667341


namespace number_of_integer_terms_in_sum_l667_667393

theorem number_of_integer_terms_in_sum : 
  (finset.sum (finset.range 100.succ) 
    (λ n, if (↑((n^2 + 1)!).factorial : ℚ) / (n.factorial ^ (n + 2): ℚ) ∈ ℕ then 1 else 0)) = 1 := 
by 
  sorry

end number_of_integer_terms_in_sum_l667_667393


namespace discriminant_zero_l667_667643

theorem discriminant_zero (a b c : ℝ) (h₁ : a = 1) (h₂ : b = -2) (h₃ : c = 1) :
  (b^2 - 4 * a * c) = 0 :=
by
  sorry

end discriminant_zero_l667_667643


namespace angle_ACB_eq_90_l667_667530

-- Definitions of points and angles
variables {A B C D E F : Type} [Triangle A B C]

-- Equivalence conditions
variables (x : Real)
variable [IsoscelesTriangle C F E (by sorry : CF = FE)]
variable [OnSegment D A B]
variable [OnSegment E B C]
variable [Intersection F A E C D]

-- Given conditions in the problem
variable (h1 : AB = 3 * AC)
variable (h2 : angle BAE = x)
variable (h3 : angle ACD = 2 * x)

-- The theorem statement to prove
theorem angle_ACB_eq_90 :
  ∀ (AB AC : Real) (BAE ACD : Real), AB = 3 * AC ∧ BAE = x ∧ ACD = 2 * x ∧ IsIsoscelesTriangle C F E (by sorry : CF = FE) 
    → angle ACB = 90 :=
sorry

end angle_ACB_eq_90_l667_667530


namespace solve_system_l667_667160

-- Define the conditions as hypotheses
def condition1 (x y : ℝ) := 3 * x^2 - x * y + 3 * y^2 = 16
def condition2 (x y : ℝ) := 7 * x^2 - 4 * x * y + 7 * y^2 = 38

-- The statement of the problem in Lean 4
theorem solve_system (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  (x, y) ≈ (2.274, -0.176) ∨ (x, y) ≈ (0.176, -2.274) ∨ 
  (x, y) ≈ (-0.176, 2.274) ∨ (x, y) ≈ (-2.274, 0.176) := 
by
  sorry

end solve_system_l667_667160


namespace euclidean_division_correct_l667_667144

noncomputable def P (X : ℝ) : ℝ := 2*X^3 + 5*X^2 + 6*X + 1
noncomputable def D (X : ℝ) : ℝ := X^2 - 3*X + 2
noncomputable def Q (X : ℝ) : ℝ := 2*X + 11
noncomputable def R (X : ℝ) : ℝ := 35*X - 21

theorem euclidean_division_correct :
  ∀ X : ℝ, P X = D X * Q X + R X ∧ (∀ X : ℝ, polynomial.degree (polynomial.C (R X)) < polynomial.degree (polynomial.C (D X))) :=
by sorry

end euclidean_division_correct_l667_667144


namespace log_identity_l667_667695

-- Definitions and conditions
def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

-- Hypothesis and conditions
variable {a b : ℝ}
variable (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ 1) (h4 : b ≠ 1)
variable (h_log := @log_base)

-- Equivalent proof problem
theorem log_identity (ha : a = 3) (hb : 1 / b = 3) : h_log a 7 = h_log (1/b) (1/7) :=
by
  sorry

end log_identity_l667_667695


namespace sum_of_distinct_nums_l667_667500

theorem sum_of_distinct_nums (m n p q : ℕ) (hmn : m ≠ n) (hmp : m ≠ p) (hmq : m ≠ q) 
(hnp : n ≠ p) (hnq : n ≠ q) (hpq : p ≠ q) (pos_m : 0 < m) (pos_n : 0 < n) 
(pos_p : 0 < p) (pos_q : 0 < q) (h : (6 - m) * (6 - n) * (6 - p) * (6 - q) = 4) : 
  m + n + p + q = 24 :=
sorry

end sum_of_distinct_nums_l667_667500


namespace problem_statements_l667_667998

noncomputable def is_sym_about_axes (C : set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ C → (-x, y) ∈ C ∧ (x, -y) ∈ C

noncomputable def is_sym_about_origin (C : set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ C → (-x, -y) ∈ C

def curve_C : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in ((x + 1)^2 + y^2) * ((x - 1)^2 + y^2) = 4 }

theorem problem_statements :
  (is_sym_about_axes curve_C) ∧ (is_sym_about_origin curve_C) ∧
  (¬ ∀ p ∈ curve_C, p.1^2 + p.2^2 ≤ 1) ∧
  (¬ ∀ p ∈ curve_C, -1/2 ≤ p.2 ∧ p.2 ≤ 1/2) :=
by
  sorry

end problem_statements_l667_667998


namespace team_person_count_l667_667722

variables
  (n : ℕ) -- Let n be the number of persons in the team
  (score_best_marksman : ℕ)
  (score_total : ℕ)
  (new_score_best_marksman : ℕ)
  (new_average_score : ℕ)

theorem team_person_count
  (h1 : score_best_marksman = 85)
  (h2 : new_score_best_marksman = 92)
  (h3 : new_average_score = 84)
  (h4 : score_total = 665) :
  n = 7 :=
begin
  -- Let the variables n and scores be set up according to the conditions
  -- Set up the equations according to the problem conditions
  let extra_points := new_score_best_marksman - score_best_marksman,
  let new_total := new_average_score * n,
  have eq1 : new_total = score_total - extra_points + extra_points := by sorry,
  -- Simplifying and solving the equation for n
  have eq2 : 84 * n = 665 := by sorry,
  have answer : n = 7 := by sorry,
  exact answer,
end

end team_person_count_l667_667722


namespace triangle_perimeter_l667_667440

theorem triangle_perimeter
  (a : ℝ) (a_gt_5 : a > 5)
  (ellipse : ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / 25 = 1)
  (dist_foci : 8 = 2 * 4) :
  4 * Real.sqrt (41) = 4 * Real.sqrt (41) := by
sorry

end triangle_perimeter_l667_667440


namespace angle_at_4_30_is_45_degrees_l667_667738

theorem angle_at_4_30_is_45_degrees :
  (∃ (angle_4 : ℝ) (num_divisions : ℕ) (angle_per_division : ℝ),
    angle_4 = 120 ∧ num_divisions = 12 ∧ angle_per_division = 30 ∧
    let divisions_apart := 1.5 in
    angle_per_division * divisions_apart = 45) :=
begin
  use 120,
  use 12,
  use 30,
  split,
  { refl, },
  split,
  { refl, },
  split,
  { refl, },
  { sorry } -- this 'sorry' indicates that the proof part is omitted.
end

end angle_at_4_30_is_45_degrees_l667_667738


namespace rita_saving_l667_667590

theorem rita_saving
  (num_notebooks : ℕ)
  (price_per_notebook : ℝ)
  (discount_rate : ℝ) :
  num_notebooks = 7 →
  price_per_notebook = 3 →
  discount_rate = 0.15 →
  (num_notebooks * price_per_notebook) - (num_notebooks * (price_per_notebook * (1 - discount_rate))) = 3.15 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end rita_saving_l667_667590


namespace belinda_percentage_flyers_l667_667745

/-- Conditions and definitions based on the problem given -/
def total_flyers : ℕ := 200
def flyers_ryan : ℕ := 42
def flyers_alyssa : ℕ := 67
def flyers_scott : ℕ := 51
def flyers_belinda : ℕ := total_flyers - (flyers_ryan + flyers_alyssa + flyers_scott)

/-- Theorem statement to prove -/
theorem belinda_percentage_flyers : 
  (flyers_belinda.to_rat / total_flyers.to_rat) * 100 = 20 :=
sorry

end belinda_percentage_flyers_l667_667745


namespace arithmetic_seq_sum_l667_667996

-- Let S be the sum sequence of the arithmetic sequence
-- S(k) is the sum of the first k terms
variable {α : Type} [LinearOrderedField α]

theorem arithmetic_seq_sum (a : ℕ → α) (m : ℕ) (S : ℕ → α)
  (h_a : ArithmeticSequence a) -- assuming we have an arithmetic sequence
  (h1 : S (2 * m) = 100)
  (h2 : (∑ n in range (m + 1) (3 * m + 1), a n) = 200) :
  (∑ n in range (m + 1) (2 * m + 1), a n) = 75 := sorry

end arithmetic_seq_sum_l667_667996


namespace rhombus_perimeter_l667_667619

theorem rhombus_perimeter (d1 d2 : ℝ) (area : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) (h3 : area = 120) :
  4 * sqrt ((d1 / 2)^2 + (d2 / 2)^2) = 52 :=
by {
  -- Working on proof here
  sorry
}

end rhombus_perimeter_l667_667619


namespace cot_30_deg_l667_667798

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 := by
  -- Define known values of trigonometric functions at 30 degrees
  have h1 : Real.tan (30 * Real.pi / 180) = 1 / Real.sqrt 3 := sorry
  have h2 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

  -- Prove using these known values
  calc
    Real.cot (30 * Real.pi / 180)
        = 1 / Real.tan (30 * Real.pi / 180) : by sorry
    ... = 1 / (1 / Real.sqrt 3) : by sorry
    ... = Real.sqrt 3 : by sorry

end cot_30_deg_l667_667798


namespace pq_proof_l667_667914

section ProofProblem

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, ∃ d, a (n + 1) = a n + d

def b_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
if n % 2 = 1 then a n - 6 else 2 * a n

def sum_first_n (seq : ℕ → ℤ) (n : ℕ) : ℤ :=
(n.sum (λ i, seq (i + 1)))

variables (a : ℕ → ℤ)
variables (S T : ℕ → ℤ)

axiom a_is_arithmetic : arithmetic_sequence a
axiom S_4_eq_32 : sum_first_n a 4 = 32
axiom T_3_eq_16 : sum_first_n (b_n a) 3 = 16

theorem pq_proof : (∀ n, a n = 2 * n + 3) ∧ (∀ n > 5, sum_first_n (b_n a) n > sum_first_n a n) :=
sorry

end ProofProblem

end pq_proof_l667_667914


namespace domain_of_sqrt_tan_minus_one_l667_667622

open Real
open Set

def domain_sqrt_tan_minus_one : Set ℝ := 
  ⋃ k : ℤ, Ico (π/4 + k * π) (π/2 + k * π)

theorem domain_of_sqrt_tan_minus_one :
  {x : ℝ | ∃ y : ℝ, y = sqrt (tan x - 1)} = domain_sqrt_tan_minus_one :=
sorry

end domain_of_sqrt_tan_minus_one_l667_667622


namespace matrix_corner_sum_eq_l667_667990

theorem matrix_corner_sum_eq (M : Matrix (Fin 2000) (Fin 2000) ℤ)
  (h : ∀ i j : Fin 1999, M i j + M (i+1) (j+1) = M i (j+1) + M (i+1) j) :
  M 0 0 + M 1999 1999 = M 0 1999 + M 1999 0 :=
sorry

end matrix_corner_sum_eq_l667_667990


namespace cot_30_eq_sqrt3_l667_667791

theorem cot_30_eq_sqrt3 : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by
  have h_cos : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  have h_sin : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  show Real.cot (30 * Real.pi / 180) = Real.sqrt 3
  calc
    Real.cot (30 * Real.pi / 180) = (Real.cos (30 * Real.pi / 180)) / (Real.sin (30 * Real.pi / 180)) := sorry
    ... = (Real.sqrt 3 / 2) / (1 / 2) := sorry
    ... = Real.sqrt 3 := sorry

end cot_30_eq_sqrt3_l667_667791


namespace extremum_at_x_equals_one_range_of_a_l667_667945

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 4 * (a - 1) * Real.log (x + 1)

-- Condition: a < 3
variable (a : ℝ) (h_a : a < 3)

-- Defining the equivalent proof problem in Lean
theorem extremum_at_x_equals_one 
  (h_a : a < 3) : (∃ x : ℝ, x = 1 ∧ ∃ ε > 0, ∀ y ∈ set.Ioc (x - ε) (x + ε), f a y ≤ f a x ∨ f a x ≤ f a y) :=
sorry

theorem range_of_a 
  (h_a : a < 3) (h_nonpos : ∀ x ∈ set.Icc 0 1, f a x ≤ 0) : a ≤ 2 :=
sorry

end extremum_at_x_equals_one_range_of_a_l667_667945


namespace rectangular_prism_diagonals_l667_667335

theorem rectangular_prism_diagonals (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  let edges := 12
  let vertices := 8
  let face_diagonals := 6 * 2 := 12
  let space_diagonals := 4
  let total_diagonals := face_diagonals + space_diagonals := 16
  total_diagonals = 16 :=
by {
  sorry,
}

end rectangular_prism_diagonals_l667_667335


namespace compute_fraction_sum_l667_667164

variable (a b c : ℝ)
open Real

theorem compute_fraction_sum (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -15)
                            (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 6) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = 12 := 
sorry

end compute_fraction_sum_l667_667164


namespace sum_of_squares_of_roots_l667_667756

theorem sum_of_squares_of_roots :
  let a := 10
  let b := 16
  let c := -18
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  sum_of_roots ^ 2 - 2 * product_of_roots = 244 / 25 := by
  sorry

end sum_of_squares_of_roots_l667_667756


namespace ratio_Solomon_Juwan_l667_667159

-- Definitions of the given conditions
variables (Solomon_cans Juwan_cans Levi_cans total_cans : ℕ)
variables (Solomon_collected Levi_collected : ℕ → ℕ)

-- Given conditions as assumptions
axiom Solomon_collected_def : Solomon_collected Solomon_cans = 66
axiom boys_total_cans : total_cans = 99
axiom Levi_collected_def : Levi_collected Juwan_cans = Juwan_cans / 2

theorem ratio_Solomon_Juwan : 
  (Solomon_collected Solomon_cans) = 66 ∧ 
  (total_cans = 99) ∧ 
  (Levi_collected Juwan_cans = Juwan_cans / 2) → 
  Solomon_cans / Juwan_cans = 3 :=
begin
  sorry
end

end ratio_Solomon_Juwan_l667_667159


namespace general_formula_sums_inequality_for_large_n_l667_667903

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def modified_seq (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = (finset.range n).sum a

variables {a b S T : ℕ → ℕ}

-- Given conditions
axiom h_arith_seq : arithmetic_seq a
axiom h_modified_seq : modified_seq a b
axiom h_sum_a : sum_seq a S
axiom h_sum_b : sum_seq b T
axiom h_S4 : S 4 = 32
axiom h_T3 : T 3 = 16

-- Proof Statements
theorem general_formula :
  ∀ n : ℕ, a n = 2 * n + 3 := sorry

theorem sums_inequality_for_large_n :
  ∀ n : ℕ, n > 5 → T n > S n := sorry

end general_formula_sums_inequality_for_large_n_l667_667903


namespace purely_imaginary_complex_l667_667114

theorem purely_imaginary_complex (a : ℝ) (z : ℂ) (h : z = (a + 1) + (1 + a^2) * complex.i) :
  complex.re z = 0 → a = -1 :=
by
  intro hz
  have h_real_part : (a + 1) = 0 := by simp [hz, h]
  sorry

end purely_imaginary_complex_l667_667114


namespace initial_percentage_increase_l667_667639

-- Given conditions
def S_original : ℝ := 4000.0000000000005
def S_final : ℝ := 4180
def reduction : ℝ := 5

-- Predicate to prove the initial percentage increase is 10%
theorem initial_percentage_increase (x : ℝ) 
  (hx : (95/100) * (S_original * (1 + x / 100)) = S_final) : 
  x = 10 :=
sorry

end initial_percentage_increase_l667_667639


namespace least_squares_solution_full_rank_pseudoinverse_l667_667561

open_locale classical

variables {m n : ℕ} (A : matrix (fin m) (fin n) ℝ) (b : vector ℝ m)

/-- Define the pseudoinverse of a matrix satisfying the given properties. -/
noncomputable def pseudoinverse (A : matrix (fin m) (fin n) ℝ) : matrix (fin n) (fin m) ℝ :=
sorry

/-- Solution to the least squares problem. -/
theorem least_squares_solution :
  ∃ x : vector ℝ n, x = (pseudoinverse A) ⬝ b ∧
    ∀ x', (x' ∈ argmin (λ x, ∥A ⬝ x - b∥)) →
    ∥x∥ ≤ ∥x'∥ :=
sorry

/-- If A is of full rank, then the pseudoinverse of A is given by a specific formula. -/
theorem full_rank_pseudoinverse (hA : matrix.rank A = min m n) :
  pseudoinverse A = (Aᵀ ⬝ A)⁻¹ ⬝ Aᵀ :=
sorry

end least_squares_solution_full_rank_pseudoinverse_l667_667561


namespace cube_in_pyramid_side_length_l667_667714

open Real

noncomputable def pyramid_side_length : ℝ := 2
noncomputable def pyramid_height : ℝ := sqrt 6
noncomputable def cube_side_length : ℝ := sqrt 6 / 2

theorem cube_in_pyramid_side_length :
  ∀ (pyramid_side_length = 2) (pyramid_height = sqrt 6),
  ∃ cube_side_length, cube_side_length = sqrt 6 / 2 :=
by
  sorry

end cube_in_pyramid_side_length_l667_667714


namespace degree_of_polynomial_l667_667661

-- Define the polynomial
def polynomial := (2 * X^3 - 5) ^ 8

-- The main theorem
theorem degree_of_polynomial : polynomial.degree = 24 :=
by
  sorry

end degree_of_polynomial_l667_667661


namespace taxi_fare_l667_667096

theorem taxi_fare (base_fare : ℕ) (total_fare_80_miles : ℕ) (fare_100_miles : ℕ) 
(h_base_fare : base_fare = 15) 
(h_fare_80 : total_fare_80_miles = 195) 
(h_fare_100 : fare_100_miles = 240) : fare_100_miles = base_fare + 2.25 * 100 := 
  sorry

end taxi_fare_l667_667096


namespace general_formula_a_n_T_n_greater_S_n_l667_667875

-- Define the conditions for the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ (n : ℕ), a n = a1 + (n - 1) * d

def b_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_of_first_n_terms (a S : ℕ → ℕ) :=
  ∀ (n k : ℕ), k ≤ n → S n = (n * (a 1 + a n)) / 2

-- Given S4 = 32 and T3 = 16
def S_4_equals_32 (S : ℕ → ℕ) : Prop := S 4 = 32
def T_3_equals_16 (T : ℕ → ℕ) : Prop := T 3 = 16

-- Prove that the general formula for a_n is 2n + 3
theorem general_formula_a_n (a : ℕ → ℕ) (h : arithmetic_sequence a) : 
  a = λ n, 2 * n + 3 := 
sorry

-- Prove that for n > 5, T_n > S_n
theorem T_n_greater_S_n (a b S T: ℕ → ℕ) 
  (ha : arithmetic_sequence a) 
  (hb : b_sequence a b)
  (hS : sum_of_first_n_terms a S)
  (hT : sum_of_first_n_terms b T)
  (hS4 : S_4_equals_32 S)
  (hT3 : T_3_equals_16 T) 
  (n : ℕ) : n > 5 → T n > S n :=
sorry

end general_formula_a_n_T_n_greater_S_n_l667_667875


namespace quadratic_roots_distinct_find_m_given_condition_l667_667037

theorem quadratic_roots_distinct (m : ℝ) : ∃ x1 x2 : ℝ, (x^2 + (m+3)*x + (m+1) = 0) ∧ (m + 1) ^ 2 + 4 > 0 := 
by 
  sorry

theorem find_m_given_condition (m : ℝ) : 
  ∀ (x1 x2 : ℝ), 
    (x^2 + (m+3)*x + (m+1) = 0) ∧ |x1 - x2| = 2 * real.sqrt 2 → 
    m = 1 ∨ m = -3 :=
by 
  sorry

end quadratic_roots_distinct_find_m_given_condition_l667_667037


namespace triangle_area_l667_667736

theorem triangle_area (n : ℝ) : 
  (let red_area := (3 / 100 * n) + (7 / 100 * n) + (11 / 100 * n) + (15 / 100 * n) + (19 / 100 * n) in
   let blue_area := (5 / 100 * n) + (9 / 100 * n) + (13 / 100 * n) + (17 / 100 * n) in
   red_area - blue_area = 20) 
  → n = 200 := 
by 
  sorry

end triangle_area_l667_667736


namespace range_f_on_interval_l667_667123

noncomputable def f (x : ℝ) (k : ℝ) (c : ℝ) : ℝ := x^k + c

theorem range_f_on_interval (k c : ℝ) (hk : 0 < k) : 
  set.range (λ x : ℝ, f x k c) ∩ set.Ici 1 = set.Ici (1 + c) :=
by
  sorry

end range_f_on_interval_l667_667123


namespace log_five_one_over_twenty_five_eq_neg_two_l667_667390

theorem log_five_one_over_twenty_five_eq_neg_two
  (h1 : (1 : ℝ) / 25 = 5^(-2))
  (h2 : ∀ (a b c : ℝ), (c > 0) → (a ≠ 0) → (log b (a^c) = c * log b a))
  (h3 : log 5 5 = 1) :
  log 5 (1 / 25) = -2 := 
  by 
    sorry

end log_five_one_over_twenty_five_eq_neg_two_l667_667390


namespace cot_30_deg_l667_667810

-- Define the known values and identities
def tan_30_deg := 1 / Real.sqrt 3

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by 
  sorry

end cot_30_deg_l667_667810


namespace solve_for_m_l667_667609

theorem solve_for_m (m : ℝ) : (m - 4)^3 = (1 / 8)^(-1) → m = 6 := by
  sorry

end solve_for_m_l667_667609


namespace height_and_radius_difference_l667_667994

theorem height_and_radius_difference 
  (X Y Z : ℝ) 
  (h_triangle : isosceles_right_triangle X Y Z (angle_right Y))
  (r : ℝ) 
  (h_circle : inscribed_circle X Y Z r)
  (h_radius : r = 1 / 4) :
  (height_from_vertex_to_base_height (X Y Z) Y) - r = (real.sqrt 2 - 1) / 4 :=
begin
  sorry
end

end height_and_radius_difference_l667_667994


namespace b_17_is_66_l667_667004

noncomputable def a_seq : ℕ → ℤ
noncomputable def b_seq : ℕ → ℤ

axiom roots_of_eq (n : ℕ) : a_seq n + a_seq (n + 1) = n ∧ a_seq n * a_seq (n + 1) = b_seq n
axiom a_10_is_7 : a_seq 10 = 7

theorem b_17_is_66 : b_seq 17 = 66 := by
  sorry

end b_17_is_66_l667_667004


namespace harry_morning_routine_time_l667_667401

-- Define the conditions in Lean.
def buy_coffee_and_bagel_time : ℕ := 15 -- minutes
def read_and_eat_time : ℕ := 2 * buy_coffee_and_bagel_time -- twice the time for buying coffee and bagel is 30 minutes

-- Define the total morning routine time in Lean.
def total_morning_routine_time : ℕ := buy_coffee_and_bagel_time + read_and_eat_time

-- The final proof problem statement.
theorem harry_morning_routine_time :
  total_morning_routine_time = 45 :=
by
  unfold total_morning_routine_time
  unfold read_and_eat_time
  unfold buy_coffee_and_bagel_time
  sorry

end harry_morning_routine_time_l667_667401


namespace prob_product_multiple_of_4_l667_667098

theorem prob_product_multiple_of_4 :
  (∑ i in ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ), ∑ j in ({1, 2, 3, 4, 5} : Finset ℕ), 
    if (i * j) % 4 = 0 then (1 / 8) * (1 / 5) else 0) = 2 / 5 := 
by 
  sorry

end prob_product_multiple_of_4_l667_667098


namespace leftover_potatoes_l667_667315

theorem leftover_potatoes (fries_per_potato : ℕ) (total_potatoes : ℕ) (required_fries : ℕ)
    (h1 : fries_per_potato = 25) (h2 : total_potatoes = 15) (h3 : required_fries = 200) :
    (total_potatoes - required_fries / fries_per_potato) = 7 :=
sorry

end leftover_potatoes_l667_667315


namespace totalCostPerPerson_l667_667364

-- Define the costs as constants
def airfareAndHotel   : Real := 13500
def foodExpenses      : Real := 4500
def transportation    : Real := 3000
def smithsonianCost   : Real := 50
def zooEntryFee       : Real := 75
def zooAllowance      : Real := 15
def riverCruiseCost   : Real := 100

-- Define the number of people in the group
def groupSize         : Real := 15

-- Calculate each cost component for the group
def smithsonianTotalCost := smithsonianCost * groupSize
def zooTotalCost := (zooEntryFee + zooAllowance) * groupSize
def riverCruiseTotalCost := riverCruiseCost * groupSize

-- Calculate the total group cost
def totalGroupCost := airfareAndHotel + foodExpenses + transportation + smithsonianTotalCost + zooTotalCost + riverCruiseTotalCost

-- Calculate the cost per person
def costPerPerson := totalGroupCost / groupSize

-- State the theorem that needs to be proved
theorem totalCostPerPerson : costPerPerson = 1640 := by
  sorry

end totalCostPerPerson_l667_667364


namespace problem_solution_l667_667859

noncomputable def is_saturated_function_of_1 (f : ℝ → ℝ) : Prop :=
  ∃ (x₀ : ℝ), f (x₀ + 1) = f x₀ + f 1

def f1 (x : ℝ) : ℝ := 1 / x
def f2 (x : ℝ) : ℝ := 2 ^ x
def f3 (x : ℝ) : ℝ := Real.log ( x ^ 2 + 2 )

theorem problem_solution :
  ¬ is_saturated_function_of_1 f1 ∧ is_saturated_function_of_1 f2 ∧ ¬ is_saturated_function_of_1 f3 :=
by
  sorry

end problem_solution_l667_667859


namespace parallelepiped_vectors_l667_667565

theorem parallelepiped_vectors (a b c : ℝ^3) : 
  (‖b‖^2 + ‖b + c - a‖^2 + ‖a‖^2 + ‖c‖^2) / (‖a‖^2 + ‖b‖^2 + ‖c‖^2) = 2 := 
  sorry

end parallelepiped_vectors_l667_667565


namespace triangle_angle_property_l667_667087

variables {a b c : ℝ}
variables {A B C : ℝ} -- angles in triangle ABC

-- definition of a triangle side condition
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

-- condition given in the problem
def satisfies_condition (a b c : ℝ) : Prop := b^2 = a^2 + c^2

-- angle property based on given problem
def angle_B_is_right (A B C : ℝ) : Prop := B = 90

theorem triangle_angle_property (a b c : ℝ) (A B C : ℝ)
  (ht : triangle a b c) 
  (hc : satisfies_condition a b c) : 
  angle_B_is_right A B C :=
sorry

end triangle_angle_property_l667_667087


namespace find_angle_A_l667_667933

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h1 : 2 * Real.sin B = Real.sqrt 3 * b) 
  (h2 : a = 2) (h3 : ∃ area : ℝ, area = Real.sqrt 3 ∧ area = (1 / 2) * b * c * Real.sin A) :
  A = Real.pi / 3 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end find_angle_A_l667_667933


namespace cot_30_deg_l667_667814

-- Define the known values and identities
def tan_30_deg := 1 / Real.sqrt 3

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by 
  sorry

end cot_30_deg_l667_667814


namespace range_of_a_l667_667962

-- Define sets A and B
def setA (a : ℝ) : Set ℝ := { x : ℝ | a-1 < x ∧ x < 2a+1 }
def setB : Set ℝ := { x : ℝ | 0 < x ∧ x < 1 }

-- State the theorem we want to prove
theorem range_of_a (a : ℝ) (h : Set.Inter (setA a) setB = ∅) : a ≤ -1/2 ∨ a ≥ 2 :=
by sorry

end range_of_a_l667_667962


namespace reals_equal_if_condition_l667_667147

theorem reals_equal_if_condition {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h : (xy + 1) / (x + 1) = (yz + 1) / (y + 1) ∧ (yz + 1) / (y + 1) = (zx + 1) / (z + 1)) : 
    x = y ∧ y = z :=
begin
  sorry
end

end reals_equal_if_condition_l667_667147


namespace find_solution_pairs_l667_667408

theorem find_solution_pairs (x y : ℝ) (hx : |x| < 2) (hy : y > 1) 
    (h : sqrt(1 / (4 - x^2)) + sqrt(y^2 / (y - 1)) = 5 / 2) : x = 0 ∧ y = 2 := 
sorry

end find_solution_pairs_l667_667408


namespace cot_30_eq_sqrt3_l667_667786

theorem cot_30_eq_sqrt3 : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by
  have h_cos : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  have h_sin : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  show Real.cot (30 * Real.pi / 180) = Real.sqrt 3
  calc
    Real.cot (30 * Real.pi / 180) = (Real.cos (30 * Real.pi / 180)) / (Real.sin (30 * Real.pi / 180)) := sorry
    ... = (Real.sqrt 3 / 2) / (1 / 2) := sorry
    ... = Real.sqrt 3 := sorry

end cot_30_eq_sqrt3_l667_667786


namespace max_min_values_of_trig_function_l667_667844

theorem max_min_values_of_trig_function :
  ∀ (x : ℝ), (π / 3 ≤ x ∧ x ≤ 4 * π / 3) →
  (∃ y_max y_min, 
    (y_max = (sin x) ^ 2 + 2 * cos x ∧ y_max = 7 / 4) ∧
    (y_min = (sin x) ^ 2 + 2 * cos x ∧ y_min = -2)) :=
by
  sorry

end max_min_values_of_trig_function_l667_667844


namespace angle_between_vectors_is_pi_over_six_l667_667852

noncomputable def a : ℝ × ℝ × ℝ := (1, 0, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ × ℝ := (x, -1, 2)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def angle_cosine (a b : ℝ × ℝ × ℝ) : ℝ :=
  dot_product a b / (magnitude a * magnitude b)

theorem angle_between_vectors_is_pi_over_six (x : ℝ) (h : dot_product a (b x) = 3) :
  real.acos (angle_cosine a (b x)) = π / 6 :=
sorry

end angle_between_vectors_is_pi_over_six_l667_667852


namespace problem_l667_667428

def a (x : ℕ) : ℕ := 2005 * x + 2006
def b (x : ℕ) : ℕ := 2005 * x + 2007
def c (x : ℕ) : ℕ := 2005 * x + 2008

theorem problem (x : ℕ) : (a x)^2 + (b x)^2 + (c x)^2 - (a x) * (b x) - (a x) * (c x) - (b x) * (c x) = 3 :=
by sorry

end problem_l667_667428


namespace alpha_is_necessary_but_not_sufficient_for_beta_l667_667449

theorem alpha_is_necessary_but_not_sufficient_for_beta :
  (∀ x: ℝ, |x - 1| ≤ 2 → ∃ x: ℝ, (x > -1 ∧ x ≤ 3) ∧ (x ∈ Icc (-1 : ℝ) 3)) ∧
  (∃ x: ℝ, ¬ ( |x - 1| ≤ 2 → (x > -1 ∧ x ≤ 3))) :=
sorry

end alpha_is_necessary_but_not_sufficient_for_beta_l667_667449


namespace thomas_drives_miles_l667_667237

-- Given conditions as definitions
def miles_per_gallon_before_100 := 40
def miles_per_gallon_after_100 := 50
def cost_per_gallon := 5
def money_available := 25

-- Prove the main statement
theorem thomas_drives_miles : 
  let gallons := money_available / cost_per_gallon in
  let miles_before_threshold := min (gallons * miles_per_gallon_before_100) 100 in
  let remaining_gallons := gallons - (miles_before_threshold / miles_per_gallon_before_100) in
  let miles := miles_before_threshold + remaining_gallons * miles_per_gallon_after_100 in
  miles = 200 :=
by
  sorry

end thomas_drives_miles_l667_667237


namespace find_x_l667_667490

theorem find_x : ∃ x : ℚ, (3 * x + 5) / 5 = 17 ↔ x = 80 / 3 := by
  sorry

end find_x_l667_667490


namespace sin_1320_eq_neg_sqrt_3_div_2_l667_667230

theorem sin_1320_eq_neg_sqrt_3_div_2 : Real.sin (1320 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_1320_eq_neg_sqrt_3_div_2_l667_667230


namespace evaluate_sum_of_ceilings_l667_667392

theorem evaluate_sum_of_ceilings :
  (⌈Real.sqrt 3⌉ + ⌈Real.sqrt 16⌉ + ⌈Real.sqrt 200⌉ = 21) :=
by
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := sorry
  have h2 : Real.sqrt 16 = 4 := sorry
  have h3 : 14 < Real.sqrt 200 ∧ Real.sqrt 200 < 15 := sorry
  -- since lean 4 use different symbol for ceiling 
  have h4 : ⌈Real.sqrt 3⌉ = 2 := sorry
  have h5 : ⌈Real.sqrt 16⌉ = 4 := sorry
  have h6 : ⌈Real.sqrt 200⌉ = 15 := sorry
  -- summing them
  calc
    ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 16⌉ + ⌈Real.sqrt 200⌉
        = 2 + 4 + 15 : by rw [h4, h5, h6]
    ... = 21 : by norm_num

end evaluate_sum_of_ceilings_l667_667392


namespace positive_integer_pairs_count_l667_667856

theorem positive_integer_pairs_count :
  ∃ (pairs : List (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ pairs → a > 0 ∧ b > 0 ∧ (1 : ℚ) / a - (1 : ℚ) / b = (1 : ℚ) / 2021) ∧ 
    pairs.length = 4 :=
by sorry

end positive_integer_pairs_count_l667_667856


namespace y_intercept_of_line_l667_667267

theorem y_intercept_of_line (x y : ℝ) (eq : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 := 
by
  sorry

end y_intercept_of_line_l667_667267


namespace train_speed_in_km_per_hr_l667_667723

def train_length : ℝ := 116.67 -- length of the train in meters
def crossing_time : ℝ := 7 -- time to cross the pole in seconds

theorem train_speed_in_km_per_hr : 
  (train_length / 1000) / (crossing_time / 3600) = 60 := by
  sorry

end train_speed_in_km_per_hr_l667_667723


namespace problem_l667_667116

-- Define first terms
def a_1 : ℕ := 12
def b_1 : ℕ := 48

-- Define the 100th term condition
def a_100 (d_a : ℚ) := 12 + 99 * d_a
def b_100 (d_b : ℚ) := 48 + 99 * d_b

-- Condition that the sum of the 100th terms is 200
def condition (d_a d_b : ℚ) := a_100 d_a + b_100 d_b = 200

-- Define the value of the sum of the first 100 terms
def sequence_sum (d_a d_b : ℚ) := 100 * 60 + (140 / 99) * ((99 * 100) / 2)

-- The proof theorem
theorem problem : ∀ d_a d_b : ℚ, condition d_a d_b → sequence_sum d_a d_b = 13000 :=
by
  intros d_a d_b h_cond
  sorry

end problem_l667_667116


namespace lowest_score_of_15_scores_is_44_l667_667184

theorem lowest_score_of_15_scores_is_44
  (mean_15 : ℕ → ℕ)
  (mean_15 = 90)
  (mean_13 : ℕ → ℕ)
  (mean_13 = 92)
  (highest_score : ℕ)
  (highest_score = 110)
  (sum15 : mean_15 * 15 = 1350)
  (sum13 : mean_13 * 13 = 1196) :
  (lowest_score : ℕ) (lowest_score = 44) :=
by
  sorry

end lowest_score_of_15_scores_is_44_l667_667184


namespace inequalities_hold_l667_667293

theorem inequalities_hold 
  (x y z a b c : ℝ) 
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (ha : a ∈ ℕ) (hb : b ∈ ℕ) (hc : c ∈ ℕ)
  (hxa : x < a) (hyb : y < b) (hzc : z < c) :
  x * y + y * z + z * x < a * b + b * c + c * a ∧
  x^2 + y^2 + z^2 < a^2 + b^2 + c^2 ∧
  x * y * z < a * b * c := 
by
  sorry

end inequalities_hold_l667_667293


namespace measure_angle_HDA_is_270_l667_667524

-- Definitions of the geometric objects
variables (A B C D E F G H I : Type) 
variables [square ABCD] [equilateral_triangle CDE] [regular_hexagon DEFGHI]
variables [isosceles_triangle GHI GH HI]

-- The statement to prove
theorem measure_angle_HDA_is_270 :
  measure_angle H D A = 270 :=
sorry

end measure_angle_HDA_is_270_l667_667524


namespace find_a_l667_667469

noncomputable def f (x a : ℝ) : ℝ := Math.sin (2 * x + Real.pi / 6) + 1 / 2 + a

theorem find_a 
  (h : ∀ (x : ℝ), -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → (f x a) ≤ 1 + 1 / 2 + a ∧ -1 / 2 ≤ Math.sin (2 * x + Real.pi / 6) ∧ Math.sin (2 * x + Real.pi / 6) ≤ 1)
  (sum_max_min: ∃ (max_val min_val : ℝ), max_val = 1 + 1 / 2 + a ∧ min_val = -1 / 2 + 1 / 2 + a ∧ max_val + min_val = 3 / 2) :
  a = 0 :=
sorry

end find_a_l667_667469


namespace log_relation_l667_667980

theorem log_relation (a b : ℝ) (log_7 : ℝ → ℝ) (log_6 : ℝ → ℝ) (log_6_343 : log_6 343 = a) (log_7_18 : log_7 18 = b) :
  a = 6 / (b + 2 * log_7 2) :=
by
  sorry

end log_relation_l667_667980


namespace vanessa_score_l667_667071

-- Define the total score of the team
def total_points : ℕ := 60

-- Define the score of the seven other players
def other_players_points : ℕ := 7 * 4

-- Mathematics statement for proof
theorem vanessa_score : total_points - other_players_points = 32 :=
by
    sorry

end vanessa_score_l667_667071


namespace seq_a_n_general_term_seq_b_n_general_term_sum_T_n_formula_l667_667455

open_locale big_operators

noncomputable def a_n (n : ℕ) : ℕ := 3 * n - 2

noncomputable def b_n (n : ℕ) : ℝ := 1 / 2 ^ (n - 1)

theorem seq_a_n_general_term (S_9 : ℕ) (a_7 : ℕ) (h1 : S_9 = 117) (h2 : a_7 = 19) : 
  ∀ n : ℕ, a_n n = 3 * n - 2 :=
by
  sorry

theorem seq_b_n_general_term (h : ∀ n : ℕ, ∑ i in finset.range n, 2^(i) * b_n (i + 1) = n) :
  ∀ n : ℕ, b_n n = 1 / 2^(n - 1) :=
by 
  sorry

noncomputable def sum_T_n (n : ℕ) : ℝ :=
  finset.sum (finset.range n) (λ k, 1 / (a_n (k + 1) * a_n (k + 2)) + b_n (k + 1))

theorem sum_T_n_formula (n : ℕ) (h1 : ∀ n : ℕ, a_n n = 3 * n - 2) (h2 : ∀ n : ℕ, b_n n = 1 / 2^(n - 1)) :
  sum_T_n n = n / (3 * n + 1) + 2 - 1 / 2^(n - 1) :=
by
  sorry

end seq_a_n_general_term_seq_b_n_general_term_sum_T_n_formula_l667_667455


namespace frog_arrangements_count_l667_667647

/-
Given:
1. There are seven distinguishable frogs sitting in a row.
2. Two frogs are green, three frogs are red, one frog is blue, and one frog is yellow.
3. Green and blue frogs refuse to sit next to the red frogs.

Prove that the number of valid arrangements for these frogs is 144.
-/
theorem frog_arrangements_count :
  let frogs := ["green", "green", "red", "red", "red", "blue", "yellow"] in
  let arrangements := 6 * (Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 2) in
  arrangements = 144 :=
by
  let frogs := ["green", "green", "red", "red", "red", "blue", "yellow"]
  let arrangements := 6 * (Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 2)
  have : arrangements = 144 := sorry
  exact this

end frog_arrangements_count_l667_667647


namespace ratio_of_length_to_breadth_l667_667630

theorem ratio_of_length_to_breadth 
    (breadth : ℝ) (area : ℝ) (h_breadth : breadth = 12) (h_area : area = 432)
    (h_area_formula : area = l * breadth) : 
    l / breadth = 3 :=
sorry

end ratio_of_length_to_breadth_l667_667630


namespace num_distinguishable_large_triangles_l667_667650

-- Given conditions
constant num_colors : ℕ := 8
constant num_corner_triangles : ℕ := 3
constant num_total_positions : ℕ := 4

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Problem Statement
theorem num_distinguishable_large_triangles :
  let corner_color_combinations := binomial num_colors num_corner_triangles
  let corner_color_permutations := corner_color_combinations * Nat.factorial num_corner_triangles
  let center_color_choices := num_colors - num_corner_triangles
  let total_combinations := corner_color_permutations * center_color_choices
  total_combinations = 1680 :=
by
  sorry

end num_distinguishable_large_triangles_l667_667650


namespace second_term_binomial_expansion_l667_667846

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Conditions: Expansion of (x - 2y)^5
def binomial_expansion (x y : ℤ) (n : ℕ) : (ℕ → ℤ) :=
  λ k, binom n k * x ^ (n - k) * (-2 * y) ^ k

-- The specific proof problem
theorem second_term_binomial_expansion (x y : ℤ) :
  binomial_expansion x y 5 1 = -10 * x ^ 4 * y :=
by
  sorry

end second_term_binomial_expansion_l667_667846


namespace math_problem_statement_l667_667426

-- Conditions
variables {a b x : ℝ}
variables (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 2)

-- Part (I): Find the range of x
def part_I := ∀ x, (1 / a^2 + 4 / b^2 >= abs (2 * x - 1) - abs (x - 1)) → x ∈ Icc (-9/2) (9/2)

-- Part (II): Prove the inequality
def part_II := (1 / a + 1 / b) * (a^5 + b^5) >= 4

-- The final Lean 4 statement
theorem math_problem_statement : (part_I ha hb hab) ∧ (part_II ha hb hab) :=
sorry

end math_problem_statement_l667_667426


namespace determine_angle_C_in_DEF_l667_667512

def Triangle := Type

structure TriangleProps (T : Triangle) :=
  (right_angle : Prop)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)

axiom triangle_ABC : Triangle
axiom triangle_DEF : Triangle

axiom ABC_props : TriangleProps triangle_ABC
axiom DEF_props : TriangleProps triangle_DEF

noncomputable def similar (T1 T2 : Triangle) : Prop := sorry

theorem determine_angle_C_in_DEF
  (h1 : ABC_props.right_angle = true)
  (h2 : ABC_props.angle_A = 30)
  (h3 : DEF_props.right_angle = true)
  (h4 : DEF_props.angle_B = 60)
  (h5 : similar triangle_ABC triangle_DEF) :
  DEF_props.angle_C = 30 :=
sorry

end determine_angle_C_in_DEF_l667_667512


namespace refrigerator_cost_l667_667150

theorem refrigerator_cost
  (R : ℝ)
  (mobile_phone_cost : ℝ := 8000)
  (loss_percent_refrigerator : ℝ := 0.04)
  (profit_percent_mobile_phone : ℝ := 0.09)
  (overall_profit : ℝ := 120)
  (selling_price_refrigerator : ℝ := 0.96 * R)
  (selling_price_mobile_phone : ℝ := 8720)
  (total_selling_price : ℝ := selling_price_refrigerator + selling_price_mobile_phone)
  (total_cost_price : ℝ := R + mobile_phone_cost)
  (balance_profit_eq : total_selling_price = total_cost_price + overall_profit):
  R = 15000 :=
by
  sorry

end refrigerator_cost_l667_667150


namespace four_digit_number_multiple_of_37_l667_667416

theorem four_digit_number_multiple_of_37 {a b c d : ℕ} (habcd : 1000 * a + 100 * b + 10 * c + d < 10000) :
  let S := (1000 * a + 100 * b + 10 * c + d) - (1000 * d + 100 * c + 10 * b + a) in
  (S % 37 = 0 ↔ b = c) := 
sorry

end four_digit_number_multiple_of_37_l667_667416


namespace log_base_5_of_reciprocal_of_25_l667_667384

theorem log_base_5_of_reciprocal_of_25 : log 5 (1 / 25) = -2 :=
by
-- Sorry this proof is omitted
sorry

end log_base_5_of_reciprocal_of_25_l667_667384


namespace intersection_point_lattice_k_values_l667_667514

theorem intersection_point_lattice_k_values :
  let lattice_point (p : ℤ × ℤ) := true
  let intersection_exists (k : ℤ) : Prop :=
    let x := 6 / (k - 1)
    let y := 6 / (k - 1) + 2
    ∃ (p : ℤ × ℤ), p = (x, y)
  let valid_values := {0, 2, -2, -1, 3, 7, 4, -5}
  lattice_point (0, 0) → intersection_exists k → #valid_values = 8 := 
sorry

end intersection_point_lattice_k_values_l667_667514


namespace jack_handing_in_amount_l667_667538

theorem jack_handing_in_amount :
  let total_100_bills := 2 * 100
  let total_50_bills := 1 * 50
  let total_20_bills := 5 * 20
  let total_10_bills := 3 * 10
  let total_5_bills := 7 * 5
  let total_1_bills := 27 * 1
  let total_notes := total_100_bills + total_50_bills + total_20_bills + total_10_bills + total_5_bills + total_1_bills
  let amount_in_till := 300
  let amount_to_hand_in := total_notes - amount_in_till
  amount_to_hand_in = 142 := by
  sorry

end jack_handing_in_amount_l667_667538


namespace sum_of_subset_products_l667_667006

/-- Define the original set of numbers. --/
def original_set : set ℤ := {k | k ∈ finset.range (26+1) ∧ k ≠ 0} ∪ -{1, 2, 3, ..., 26}

/-- Define the function that computes the product of elements in a subset. --/
def product_of_subset (s : set ℤ) : ℤ := s.prod (λ x, x)

/-- Define the condition for a subset to be valid (contain at least 2 elements) --/
def valid_subset (s : set ℤ) : Prop := s.card ≥ 2

/-- The main theorem stating the sum of products of all valid subsets is 350. --/
theorem sum_of_subset_products : 
  (finset.powerset original_set).filter valid_subset 
  .sum (λ s, product_of_subset s) = 350 := 
sorry

end sum_of_subset_products_l667_667006


namespace system_of_inequalities_l667_667124

noncomputable def f : ℝ → ℝ := sorry -- Assume f is defined elsewhere

lemma even_function (f : ℝ → ℝ) : ∀ x : ℝ, f (x + 2) = f x :=
sorry -- Given condition about period

lemma even_property (f : ℝ → ℝ) : ∀ x : ℝ, f (-x) = f x :=
sorry -- Given condition about the function being even

lemma strictly_decreasing_f (f : ℝ → ℝ) : ∀ x : ℝ, 1 < x ∧ x < 2 → f' x < 0 :=
sorry -- Given condition about strict decreasing on the interval (1, 2)

lemma evaluate_at_points (f : ℝ → ℝ) : f π = 1 ∧ f (2 * π) = 0 :=
sorry -- Given specific evaluations of f

theorem system_of_inequalities (f : ℝ → ℝ) 
  (H1 : ∀ x : ℝ, f (x + 2) = f x)
  (H2 : ∀ x : ℝ, f (-x) = f x)
  (H3 : ∀ x : ℝ, 1 < x ∧ x < 2 → f' x < 0)
  (H4 : f π = 1)
  (H5 : f (2 * π) = 0) :
  ∀ x : ℝ, (2 * π - 6 ≤ x ∧ x ≤ 4 - π) → 0 ≤ f x ∧ f x ≤ 1 :=
sorry

end system_of_inequalities_l667_667124


namespace correct_multiplication_result_l667_667300

theorem correct_multiplication_result :
  (∃ (n : ℕ), 6 * n = 42 ∧ 3 * n = 21) :=
by
  existsi 7
  split
  case left =>
    exact (by norm_num : 6 * 7 = 42)
  case right =>
    exact (by norm_num : 3 * 7 = 21)

end correct_multiplication_result_l667_667300


namespace solve_for_x_l667_667612

theorem solve_for_x (x : ℝ) : (x - 55) / 3 = (2 - 3*x + x^2) / 4 → (x = 20 / 3 ∨ x = -11) :=
by
  intro h
  sorry

end solve_for_x_l667_667612


namespace solve_for_x_l667_667610

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem solve_for_x :
  let numer := sqrt (7^2 + 24^2)
  let denom := sqrt (49 + 16)
  numer / denom = 25 * (sqrt 65) / 65 :=
by
  let numer := sqrt (7^2 + 24^2)
  let denom := sqrt (49 + 16)
  have : numer = sqrt 625 := by sorry 
  have : denom = sqrt 65 := by sorry
  have : sqrt 625 = 25 := by sorry
  have : numer = 25 := by rw [this]
  have : denom = sqrt 65 := by sorry
  have : numer / denom = 25 / sqrt 65 := by rw [this]
  have : (25 : ℝ) / sqrt 65 = 25 * sqrt 65 / 65 := by
    field_simp [sqrt]; ring
  exact this

end solve_for_x_l667_667610


namespace least_three_digit_divisible_3_4_7_is_168_l667_667666

-- Define the function that checks the conditions
def is_least_three_digit_divisible_by_3_4_7 (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000 ∧ x % 3 = 0 ∧ x % 4 = 0 ∧ x % 7 = 0

-- Define the target value
def least_three_digit_number_divisible_by_3_4_7 : ℕ := 168

-- The theorem we want to prove
theorem least_three_digit_divisible_3_4_7_is_168 :
  ∃ x : ℕ, is_least_three_digit_divisible_by_3_4_7 x ∧ x = least_three_digit_number_divisible_by_3_4_7 := by
  sorry

end least_three_digit_divisible_3_4_7_is_168_l667_667666


namespace find_lowest_score_l667_667181

variable (scores : Fin 15 → ℕ)
variable (highest_score : ℕ := 110)

-- Conditions
def arithmetic_mean_15 := (∑ i, scores i) / 15 = 90
def mean_without_highest_lowest := (∑ i in Finset.erase (Finset.erase (Finset.univ : Finset (Fin 15)) (Fin.mk 0 sorry)) (Fin.mk 14 sorry)) / 13 = 92
def highest_value := scores (Fin.mk 14 sorry) = highest_score

-- The condition includes univ meaning all elements in Finset and we are removing the first(lowest) and last(highest) elements (0 and 14).

-- The lowest score
def lowest_score (scores : Fin 15 → ℕ) : ℕ :=
  1350 - 1196 - highest_score

theorem find_lowest_score : lowest_score scores = 44 :=
by
  sorry

end find_lowest_score_l667_667181


namespace cot_30_deg_l667_667794

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 := by
  -- Define known values of trigonometric functions at 30 degrees
  have h1 : Real.tan (30 * Real.pi / 180) = 1 / Real.sqrt 3 := sorry
  have h2 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

  -- Prove using these known values
  calc
    Real.cot (30 * Real.pi / 180)
        = 1 / Real.tan (30 * Real.pi / 180) : by sorry
    ... = 1 / (1 / Real.sqrt 3) : by sorry
    ... = Real.sqrt 3 : by sorry

end cot_30_deg_l667_667794


namespace f_decreasing_l667_667464

def f (x : ℝ) : ℝ := (1 - 2^x) / (2^x + 1)

theorem f_decreasing (x₁ x₂ : ℝ) (h : x₁ < x₂) : f x₁ > f x₂ :=
sorry

end f_decreasing_l667_667464


namespace general_formula_a_n_T_n_greater_than_S_n_l667_667925

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n_l667_667925


namespace probability_diff_gte_3_l667_667250

-- Define the set of numbers and the condition of selecting two different numbers from the set.
def numSet : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Definition to check if the difference between two numbers is 3 or greater.
def diff_gte_3 (a b : ℕ) : Prop := (a ≠ b ∧ (|a - b| ≥ 3))

-- Calculate the number of combinations for selecting 2 out of 9 (binomial coefficient).
def total_combinations : ℕ := Nat.choose 9 2

-- Define the pairs that have a difference of exactly 1
def diff_1_pairs : List (ℕ × ℕ) := [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]

-- Define the pairs that have a difference of exactly 2
def diff_2_pairs : List (ℕ × ℕ) := [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9)]

-- Sum of pairs with differences less than 3
def diff_lt_3_pairs : ℕ := diff_1_pairs.length + diff_2_pairs.length

-- Number of valid pairs with difference 3 or greater
def valid_pairs_count : ℕ := total_combinations - diff_lt_3_pairs

-- Probability calculation
theorem probability_diff_gte_3 : (valid_pairs_count : ℚ) / (total_combinations : ℚ) = 7 / 12 :=
by
  sorry

end probability_diff_gte_3_l667_667250


namespace coefficient_of_second_term_binomial_expansion_l667_667517

theorem coefficient_of_second_term_binomial_expansion :
  (let expr := (x^2 - (1/x))^5 in 
   let second_term := (-1)^1 * (Nat.choose 5 1) * x^(10-3) in 
   (-1) * (Nat.choose 5 1) = -5) :=
by
  sorry

end coefficient_of_second_term_binomial_expansion_l667_667517


namespace geometric_locus_inside_triangle_l667_667410

theorem geometric_locus_inside_triangle {A B C M : Type} 
(distance_to_sides : A → B → C → M → (ℝ × ℝ × ℝ)) :
  let d := distance_to_sides A B C M in
  (d.1 < d.2 + d.3) →
  (d.2 < d.3 + d.1) →
  (d.3 < d.1 + d.2) →
  ∃ (P Q R : Type), M ∈ triangle_bases_of_angle_bisectors A B C P Q R :=
sorry

end geometric_locus_inside_triangle_l667_667410


namespace total_guppies_l667_667047

-- Define conditions
def Haylee_guppies : ℕ := 3 * 12
def Jose_guppies : ℕ := Haylee_guppies / 2
def Charliz_guppies : ℕ := Jose_guppies / 3
def Nicolai_guppies : ℕ := Charliz_guppies * 4

-- Theorem statement: total number of guppies is 84
theorem total_guppies : Haylee_guppies + Jose_guppies + Charliz_guppies + Nicolai_guppies = 84 := 
by 
  sorry

end total_guppies_l667_667047


namespace triangle_geometry_problem_l667_667533

theorem triangle_geometry_problem
  (A B C D : Type*) [euclidean_space A]
  (angle_BAD angle_ABD angle_BCD : ℝ)
  (AB CD : nnreal)
  (hBAD : angle_BAD = 60) (hABC : angle_ABD = 30) (hBCD : angle_BCD = 30)
  (hAB : AB = 15) (hCD : CD = 8)
  (D_interior : D ∈ interior (triangle A B C)) :
  length (seg A D) = 3.5 :=
sorry

end triangle_geometry_problem_l667_667533


namespace smallest_mn_sum_l667_667090

theorem smallest_mn_sum (m n : ℕ) (h : 3 * n ^ 3 = 5 * m ^ 2) : m + n = 60 :=
sorry

end smallest_mn_sum_l667_667090


namespace toms_dog_age_is_twelve_l667_667243

-- Definitions based on given conditions
def toms_cat_age : ℕ := 8
def toms_rabbit_age : ℕ := toms_cat_age / 2
def toms_dog_age : ℕ := 3 * toms_rabbit_age

-- The statement to be proved
theorem toms_dog_age_is_twelve : toms_dog_age = 12 := by
  sorry

end toms_dog_age_is_twelve_l667_667243


namespace pq_proof_l667_667913

section ProofProblem

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, ∃ d, a (n + 1) = a n + d

def b_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
if n % 2 = 1 then a n - 6 else 2 * a n

def sum_first_n (seq : ℕ → ℤ) (n : ℕ) : ℤ :=
(n.sum (λ i, seq (i + 1)))

variables (a : ℕ → ℤ)
variables (S T : ℕ → ℤ)

axiom a_is_arithmetic : arithmetic_sequence a
axiom S_4_eq_32 : sum_first_n a 4 = 32
axiom T_3_eq_16 : sum_first_n (b_n a) 3 = 16

theorem pq_proof : (∀ n, a n = 2 * n + 3) ∧ (∀ n > 5, sum_first_n (b_n a) n > sum_first_n a n) :=
sorry

end ProofProblem

end pq_proof_l667_667913


namespace circle_and_line_problem_l667_667453

-- Definitions based on the provided conditions
def circle_center_x (a : ℝ) : Prop := a > 0
def circle_tangent_positive_axes (C : ℝ × ℝ) : Prop := C.1 > 0 ∧ C.2 > 0
def circle_distance_to_line (C : ℝ × ℝ) (d : ℝ) : Prop := sqrt 2 * C.fst = d

-- Mathematical problem definitions based on the questions and answers
def circle_equation (x y : ℝ) := (x - 1)^2 + (y - 1)^2 = 1
def line_tangent_to_circle (m n : ℝ) (C : ℝ × ℝ) : Prop :=
  m > 2 ∧ n > 2 ∧ let AMGM := m + n ≥ 2 * sqrt (m * n) in
  let min_mn := (m * n) ≥ (6 + 4 * sqrt 2) in
  circle_tangent_positive_axes C ∧ circle_distance_to_line C 1 ∧ AMGM ∧ min_mn

-- The actual statement
theorem circle_and_line_problem : 
  ∃ (x y : ℝ), circle_equation x y ∧ ∀ (m n : ℝ), line_tangent_to_circle m n (x, y) := sorry

end circle_and_line_problem_l667_667453


namespace cot_30_eq_sqrt3_l667_667788

theorem cot_30_eq_sqrt3 : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by
  have h_cos : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  have h_sin : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  show Real.cot (30 * Real.pi / 180) = Real.sqrt 3
  calc
    Real.cot (30 * Real.pi / 180) = (Real.cos (30 * Real.pi / 180)) / (Real.sin (30 * Real.pi / 180)) := sorry
    ... = (Real.sqrt 3 / 2) / (1 / 2) := sorry
    ... = Real.sqrt 3 := sorry

end cot_30_eq_sqrt3_l667_667788


namespace prob_one_side_of_tri_in_decagon_is_half_l667_667850

noncomputable def probability_one_side_of_tri_in_decagon : ℚ :=
  let num_vertices := 10
  let total_triangles := Nat.choose num_vertices 3
  let favorable_triangles := 10 * 6
  favorable_triangles / total_triangles

theorem prob_one_side_of_tri_in_decagon_is_half :
  probability_one_side_of_tri_in_decagon = 1 / 2 := by
  sorry

end prob_one_side_of_tri_in_decagon_is_half_l667_667850


namespace gcd_even_number_greater_than_six_l667_667149

theorem gcd_even_number_greater_than_six (n : ℕ) (hn : n > 6) (heven : n % 2 = 0) : 
  ∃ p q : ℕ, p = 3 ∧ q = 5 ∧ p.prime ∧ q.prime ∧ gcd (n - p) (n - q) = 1 :=
by 
  sorry

end gcd_even_number_greater_than_six_l667_667149


namespace log_base_5_of_1_div_25_eq_neg_2_l667_667381

theorem log_base_5_of_1_div_25_eq_neg_2 :
  log 5 (1/25) = -2 :=
by sorry

end log_base_5_of_1_div_25_eq_neg_2_l667_667381


namespace angle_CAD_l667_667586

noncomputable def angle_arc (degree: ℝ) (minute: ℝ) : ℝ :=
  degree + minute / 60

theorem angle_CAD :
  angle_arc 117 23 / 2 + angle_arc 42 37 / 2 = 80 :=
by
  sorry

end angle_CAD_l667_667586


namespace symmetric_points_x_axis_l667_667502

theorem symmetric_points_x_axis (a b : ℝ) (P : ℝ × ℝ := (a, 1)) (Q : ℝ × ℝ := (-4, b)) :
  (Q.1 = -P.1 ∧ Q.2 = -P.2) → (a = -4 ∧ b = -1) :=
by {
  sorry
}

end symmetric_points_x_axis_l667_667502


namespace problem_l667_667854

variable (a b : ℝ)

theorem problem (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 4) : a^2 + b^2 ≥ 8 := 
sorry

end problem_l667_667854


namespace parabola_focus_l667_667957

-- Defining the conditions
def parabola (p : ℝ) (h : p > 0) := ∀ x y : ℝ, x^2 = 2 * p * y

def directrix_condition (p : ℝ) := (-1, -2) ∈ (λ y, y = - p / 2)

-- Stating the coordinates of the focus
def focus_coordinates (x y : ℝ) := x = 0 ∧ y = 2

-- Main theorem statement
theorem parabola_focus : ∀ p : ℝ, p > 0 ∧ directrix_condition p → focus_coordinates 0 2 := by
  sorry

end parabola_focus_l667_667957


namespace sum_distinct_prime_factors_of_420_l667_667282

theorem sum_distinct_prime_factors_of_420 : 
  ∑ (p : ℕ) in {2, 3, 5, 7}, p = 17 := 
by 
  sorry

end sum_distinct_prime_factors_of_420_l667_667282


namespace sequence_divisible_by_3p_for_prime_l667_667546

open Nat

def sequence (n : ℕ) : ℤ :=
  match n with
  | 1   => 1
  | 2   => 7
  | n+1 => have h1 : ℤ := sequence n
           have h2 : ℤ := sequence (n - 1)
           h1 + 3 * h2

theorem sequence_divisible_by_3p_for_prime (p : ℕ) (hp : Prime p) : 3 * p ∣ (sequence p - 1) := by
  sorry

end sequence_divisible_by_3p_for_prime_l667_667546


namespace max_omega_for_increasing_sin_l667_667592

theorem max_omega_for_increasing_sin (ω : ℝ) (hω : ω > 0) : 
  (∀ x ∈ set.Icc (0 : ℝ) (π / 4), (deriv (λ x, 2 * sin (ω * x)) x ≥ 0)) ↔ ω ≤ 2 :=
sorry

end max_omega_for_increasing_sin_l667_667592


namespace robot_path_area_l667_667262

structure RobotState where
  A B : ℤ
  deriving Repr, DecidableEq, Inhabited

inductive Move
  | Up
  | Down
  | Left
  | Right

open Move

def updateState (s : RobotState) (m : Move) : RobotState :=
  match m with
  | Up    => {s with A := s.A + 1}
  | Down  => {s with A := s.A - 1}
  | Right => {s with B := s.B + s.A}
  | Left  => {s with B := s.B - s.A}

noncomputable def finalState (moves : List Move) : RobotState :=
  moves.foldl updateState {A := 0, B := 0}

theorem robot_path_area (moves : List Move) (path_closes_itself : Bool) :
  path_closes_itself = tt → 
  ∃ area : ℤ, area = Int.natAbs (finalState moves).B ∧
              area = enclosed_area_by_path moves := 
by
  sorry

end robot_path_area_l667_667262


namespace max_k_partition_l667_667763

theorem max_k_partition :
  ∃ (k : ℕ), (∀ (A : fin k → set ℕ), (∀ n ≥ 15, ∃ i, ∃ a b ∈ A i, a ≠ b ∧ a + b = n)) ∧ 
  (∀ (j : ℕ), (∀ (A : fin j → set ℕ), (∀ n ≥ 15, ∃ i, ∃ a b ∈ A i, a ≠ b ∧ a + b = n) → j ≤ 3)) :=
sorry -- Proof is omitted

end max_k_partition_l667_667763


namespace y_intercept_of_line_l667_667272

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 :=
by
  sorry

end y_intercept_of_line_l667_667272


namespace union_sets_l667_667968

variables (a : ℕ) (A B : set ℕ)
hypothesis hA : A = {0, 1, a}
hypothesis hB : B = {0, 3, 3 * a}
hypothesis h_inter : A ∩ B = {0, 3}

theorem union_sets (a = 3) : A ∪ B = {0, 1, 3, 9} :=
sorry

end union_sets_l667_667968


namespace number_of_zeros_of_f_l667_667636

noncomputable def f (x : ℝ) : ℝ := 2 * x * |Real.log x / Real.log 0.5| - 1

theorem number_of_zeros_of_f : {x : ℝ | f x = 0}.finite ∧ {x : ℝ | f x = 0}.card = 2 :=
by
  sorry

end number_of_zeros_of_f_l667_667636


namespace part_I_part_II_l667_667949

def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

-- Part I
theorem part_I (a b : ℝ) (h₁ : -1 < a) (h₂ : a < 1) (h₃ : -1 < b) (h₄ : b < 1) :
  f a + f b = f ((a + b) / (1 + a * b)) :=
by
  sorry

-- Part II
theorem part_II (m : ℝ) :
  (∃ x ∈ Icc 0 (Real.pi / 2), f (Real.sin x ^ 2) + f (m * Real.cos x + 2 * m) = 0) ↔ 
  -1/2 < m ∧ m ≤ 0 :=
by
  sorry

end part_I_part_II_l667_667949


namespace revenue_decrease_percentage_is_sixteen_l667_667228

noncomputable theory

def decrease_percent_in_revenue (T C : ℝ) : ℝ :=
  let original_revenue := T * C in
  let new_tax := 0.70 * T in
  let new_consumption := 1.20 * C in
  let new_revenue := new_tax * new_consumption in
  let decrease_in_revenue := original_revenue - new_revenue in
  (decrease_in_revenue / original_revenue) * 100

theorem revenue_decrease_percentage_is_sixteen :
  ∀ T C : ℝ, decrease_percent_in_revenue T C = 16 :=
by
  intros T C
  let original_revenue := T * C
  let new_tax := 0.70 * T
  let new_consumption := 1.20 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  have decrease_in_revenue_eq : decrease_in_revenue = 0.16 * original_revenue :=
    by
      simp [original_revenue, new_tax, new_consumption, new_revenue]
      linarith
  rw [decrease_percent_in_revenue, decrease_in_revenue_eq]
  simp
  rfl

end revenue_decrease_percentage_is_sixteen_l667_667228


namespace expected_games_is_correct_l667_667699

open Probability

/-- Define the conditions of the problem -/
variables
  (prob_A_win : ℚ := 2 / 3)
  (prob_B_win : ℚ := 1 / 3)
  (max_games : ℕ := 6)
  (point_difference : ℕ := 2)

/-- The expected number of games played, given the match conditions -/
def expected_number_of_games : ℚ :=
  (2 * (prob_A_win^2 + prob_B_win^2)) +
  (4 * (3 * prob_A_win^3 * prob_B_win + 3 * prob_B_win^3 * prob_A_win)) +
  (6 * (6 * (prob_A_win^2 * prob_B_win^2)))

theorem expected_games_is_correct :
   expected_number_of_games = 266 / 81 :=
sorry

end expected_games_is_correct_l667_667699


namespace relative_positions_of_points_on_parabola_l667_667445

/-- Given a function f(x) = x^2 - 2x + 3 and points on this function, -/
def check_relative_positions (y : ℝ → ℝ) (y1 y2 y3 : ℝ) : Prop :=
  y (-3) = y1 ∧ y (1) = y2 ∧ y (-1/2) = y3 → y2 < y3 ∧ y3 < y1

theorem relative_positions_of_points_on_parabola :
  check_relative_positions (λ x, x^2 - 2 * x + 3) y1 y2 y3 :=
sorry

end relative_positions_of_points_on_parabola_l667_667445


namespace cot_30_eq_sqrt_3_l667_667805

theorem cot_30_eq_sqrt_3 (h : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3) : 
  Real.cot (Real.pi / 6) = Real.sqrt 3 :=
by
  sorry

end cot_30_eq_sqrt_3_l667_667805


namespace largest_n_under_100000_l667_667662

theorem largest_n_under_100000 (n : ℕ) : 
  n < 100000 ∧ (9 * (n - 3)^6 - 3 * n^3 + 21 * n - 42) % 7 = 0 → n = 99996 :=
by
  sorry

end largest_n_under_100000_l667_667662


namespace ducks_and_geese_meet_l667_667693

theorem ducks_and_geese_meet (x : ℕ) : 
  (1 / 7 : ℚ) * x + (1 / 9 : ℚ) * x = 1 → True := 
by
  assume h
  sorry

end ducks_and_geese_meet_l667_667693


namespace log_not_fraction_of_polynomials_l667_667148

theorem log_not_fraction_of_polynomials (a : Real) (ha : 1 < a) :
  ¬ ∃ (f g : Real[X]), f ≠ 0 ∧ g ≠ 0 ∧ (∀ x, log a x = f.eval x / g.eval x ∧ f.gcd g = 1) :=
by
  sorry

end log_not_fraction_of_polynomials_l667_667148


namespace lee_charged_per_action_figure_l667_667104

theorem lee_charged_per_action_figure :
  ∀ (sneakers_cost savings action_figures leftovers price_per_fig),
    sneakers_cost = 90 →
    savings = 15 →
    action_figures = 10 →
    leftovers = 25 →
    price_per_fig = 10 →
    (savings + action_figures * price_per_fig) - sneakers_cost = leftovers → price_per_fig = 10 :=
by
  intros sneakers_cost savings action_figures leftovers price_per_fig
  intros h_sneakers_cost h_savings h_action_figures h_leftovers h_price_per_fig
  intros h_total
  sorry

end lee_charged_per_action_figure_l667_667104


namespace gcd_values_l667_667125

theorem gcd_values (n : ℕ) (h : n > 0) : let d := Int.gcd (n^2 + 1) ((n+1)^2 + 1) in d = 1 ∨ d = 5 :=
by
  sorry

end gcd_values_l667_667125


namespace triangle_LM_length_l667_667088

theorem triangle_LM_length 
  (A B C K L M : Point) 
  (hABC_right : ∠A = 90°)
  (hB_30 : ∠B = 30°)
  (hAK : length(A, K) = 4)
  (hBL : length(B, L) = 31)
  (hMC : length(M, C) = 3)
  (hKL_KM : length(K, L) = length(K, M))
  : length(L, M) = 14 := 
sorry

end triangle_LM_length_l667_667088


namespace problem_statement_l667_667558

theorem problem_statement (n : ℤ) (hn1 : 0 ≤ n) (hn2 : n < 37) (h : 6 * n ≡ 1 [MOD 37]) : 
  ((2 ^ n)^4 - 3) % 37 = 35 :=
by
  sorry

end problem_statement_l667_667558


namespace cot_30_deg_l667_667813

-- Define the known values and identities
def tan_30_deg := 1 / Real.sqrt 3

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by 
  sorry

end cot_30_deg_l667_667813


namespace calculate_x_l667_667679

theorem calculate_x :
  (422 + 404) ^ 2 - (4 * 422 * 404) = 324 :=
by
  -- proof goes here
  sorry

end calculate_x_l667_667679


namespace cot_30_eq_sqrt_3_l667_667825

theorem cot_30_eq_sqrt_3 :
  let θ := 30.0
  let cos θ := (real.sqrt 3) / 2
  let sin θ := 1 / 2
  let cot θ := cos θ / sin θ
  cot θ = real.sqrt 3 := by
sorry

end cot_30_eq_sqrt_3_l667_667825


namespace angle_PBC_eq_90_l667_667548

theorem angle_PBC_eq_90 
  (A B C : Point) 
  (E D F P : Point) 
  (h_triangle : is_triangle A B C)
  (h_sum : dist A B + dist A C = 3 * dist B C)
  (h_B_excircle_touch : B_excircle_touches_at B_Excircle A C E D)
  (h_C_excircle_touch : C_excircle_touches_at C_Excircle A B F)
  (h_lines_meet : lines_meet_at (line_through C F) (line_through D E) P) :
  angle_btw_lines (line_through B C) (line_through B P) = 90 :=
sorry

end angle_PBC_eq_90_l667_667548


namespace trajectory_M_ratio_DE_AB_l667_667433

noncomputable def P : ℝ × ℝ := sorry 
noncomputable def Q : ℝ × ℝ := sorry 
noncomputable def M : ℝ × ℝ := sorry 
noncomputable def E : ℝ × ℝ := (-4, 0)

-- Part (I)
theorem trajectory_M :
  ∃ x y : ℝ, P.1^2 + (3 * P.2)^2 = 18 → M = (P.1, P.2 / 3) →
  ∀ x y, x^2 / 18 + y^2 / 2 = 1 := sorry

-- Part (II)
theorem ratio_DE_AB (m : ℝ) (hm : m ≠ 0) :
  let A := (m * (8 * m / (m^2 + 9)) - 8, 8 * m / (m^2 + 9))
  let B := (m * (-8 * m / (m^2 + 9)) - 8, -8 * m / (m^2 + 9))
  let D := (-32 / (m^2 + 9), 0)
  let DE := abs (D.1 + 4)
  let AB := (sqrt (m^2 + 1) * abs ((8 * m / (m^2 + 9)) - (-8 * m / (m^2 + 9))))
  ∃ D A B, DE / AB = (sqrt 2) / 3 := sorry

end trajectory_M_ratio_DE_AB_l667_667433


namespace michael_savings_l667_667700

theorem michael_savings :
  let price := 45
  let tax_rate := 0.08
  let promo_A_dis := 0.40
  let promo_B_dis := 15
  let before_tax_A := price + price * (1 - promo_A_dis)
  let before_tax_B := price + (price - promo_B_dis)
  let after_tax_A := before_tax_A * (1 + tax_rate)
  let after_tax_B := before_tax_B * (1 + tax_rate)
  after_tax_B - after_tax_A = 3.24 :=
by
  sorry

end michael_savings_l667_667700


namespace brazilNutsPrice_l667_667170

noncomputable def pricePerPoundBrazilNuts : Real :=
  let priceCashew := 6.75
  let weightCashew := 20
  let totalWeightMixture := 50
  let priceMixture := 5.70
  let costCashew := priceCashew * weightCashew
  let weightBrazilNuts := totalWeightMixture - weightCashew
  let totalCostMixture := priceMixture * totalWeightMixture
  let costBrazilNuts := totalCostMixture - costCashew
  let priceBrazilNuts := costBrazilNuts / weightBrazilNuts
  priceBrazilNuts

theorem brazilNutsPrice :
  pricePerPoundBrazilNuts = 5 :=
by
  unfold pricePerPoundBrazilNuts
  let priceCashew := 6.75
  let weightCashew := 20
  let totalWeightMixture := 50
  let priceMixture := 5.70
  let costCashew := priceCashew * weightCashew
  let weightBrazilNuts := totalWeightMixture - weightCashew
  let totalCostMixture := priceMixture * totalWeightMixture
  let costBrazilNuts := totalCostMixture - costCashew
  let priceBrazilNuts := costBrazilNuts / weightBrazilNuts
  show priceBrazilNuts = 5
  sorry

end brazilNutsPrice_l667_667170


namespace triangle_trig_identity_l667_667063

variable {A B C : RealAngle} {a b c : ℝ}

theorem triangle_trig_identity (h1 : a^2 + b^2 = 2018 * c^2) (h2 : ∠A + ∠B + ∠C = π) :
  (2 * sin A * sin B * cos C) / (1 - cos C ^ 2) = 2017 :=
by
  sorry

end triangle_trig_identity_l667_667063


namespace tim_pays_300_l667_667241

def mri_cost : ℕ := 1200
def doctor_rate_per_hour : ℕ := 300
def examination_time_in_hours : ℕ := 1 / 2
def consultation_fee : ℕ := 150
def insurance_coverage : ℚ := 0.8

def examination_cost : ℕ := doctor_rate_per_hour * examination_time_in_hours
def total_cost_before_insurance : ℕ := mri_cost + examination_cost + consultation_fee
def insurance_coverage_amount : ℚ := total_cost_before_insurance * insurance_coverage
def amount_tim_pays : ℚ := total_cost_before_insurance - insurance_coverage_amount

theorem tim_pays_300 : amount_tim_pays = 300 := 
by
  -- proof goes here
  sorry

end tim_pays_300_l667_667241


namespace starting_number_of_range_divisible_by_11_l667_667234

theorem starting_number_of_range_divisible_by_11 (a : ℕ) : 
  a ≤ 79 ∧ (a + 22 = 77) ∧ ((a + 11) + 11 = 77) → a = 55 := 
by
  sorry

end starting_number_of_range_divisible_by_11_l667_667234


namespace Jen_visits_either_but_not_both_l667_667649

-- Define the events and their associated probabilities
def P_Chile : ℝ := 0.30
def P_Madagascar : ℝ := 0.50

-- Define the probability of visiting both assuming independence
def P_both : ℝ := P_Chile * P_Madagascar

-- Define the probability of visiting either but not both
def P_either_but_not_both : ℝ := P_Chile + P_Madagascar - 2 * P_both

-- The problem statement
theorem Jen_visits_either_but_not_both : P_either_but_not_both = 0.65 := by
  /- The proof goes here -/
  sorry

end Jen_visits_either_but_not_both_l667_667649


namespace sixty_times_reciprocal_l667_667494

theorem sixty_times_reciprocal (x : ℚ) (h : 8 * x = 6) : 60 * (1 / x) = 80 := by
  have hx : x = 3 / 4 := by
    have h1 : 8 * x = 6 := h
    have h2 : x = 6 / 8 := by
      linarith
    exact h2
  have reciprocal_x : 1 / x = 4 / 3 := by
    rw [hx]
    norm_num
  rw [reciprocal_x]
  norm_num
  sorry

end sixty_times_reciprocal_l667_667494


namespace area_of_largest_triangle_in_semicircle_l667_667173

theorem area_of_largest_triangle_in_semicircle (r : ℝ) : 
  let d := 2 * r in
  let a := r * Real.sqrt 2 in
  let triangle_area := (1 / 2) * a * a in
  triangle_area = r^2 := 
by
  sorry

end area_of_largest_triangle_in_semicircle_l667_667173


namespace hyperbola_sum_real_axis_and_focal_distance_l667_667642

/-- Given the hyperbola equation, the sum of the length of the real axis and the focal distance is 3. -/
theorem hyperbola_sum_real_axis_and_focal_distance :
  ∀ (x y : ℝ), 12 * x^2 - 4 * y^2 = 3 → (2 * (Real.sqrt (1 / 4)) + 2 * (Real.sqrt ( (1 / 4) + (3 / 4) )) = 3) :=
begin
  intros x y h,
  sorry,
end

end hyperbola_sum_real_axis_and_focal_distance_l667_667642


namespace cot_30_eq_sqrt_3_l667_667835

theorem cot_30_eq_sqrt_3 (θ : ℝ) (hθ : θ = 30) (h1 : tan 30 = 1 / real.sqrt 3) : cot 30 = real.sqrt 3 :=
by
  have h2: cot 30 = 1 / (tan 30), by exact cot_def
  have h3: cot 30 = 1 / (1 / real.sqrt 3), by rw [h1, h2]
  have h4: cot 30 = real.sqrt 3, by exact (one_div_div (1 / real.sqrt 3))
  exact h4


end cot_30_eq_sqrt_3_l667_667835


namespace selling_price_to_achieve_profit_l667_667314

theorem selling_price_to_achieve_profit :
  ∃ (x : ℝ), let original_price := 210
              let purchase_price := 190
              let avg_sales_initial := 8
              let profit_goal := 280
              (210 - x = 200) ∧
              let profit_per_item := original_price - purchase_price - x
              let avg_sales_quantity := avg_sales_initial + 2 * x
              profit_per_item * avg_sales_quantity = profit_goal := by
  sorry

end selling_price_to_achieve_profit_l667_667314


namespace ant_distance_l667_667734

-- Lean Definitions
def eastward_series (n : ℕ) : ℝ := if even n then (1:ℝ) / (2:ℝ) ^ n else 0
def northeast_series (n : ℕ) : ℝ := if odd n then (1:ℝ) / (2:ℝ) ^ n else 0

-- Lean Statement
theorem ant_distance :
  let east_sum := ∑' n, eastward_series n
  let northeast_sum := ∑' n, northeast_series n
  let total_distance := real.sqrt ((east_sum + northeast_sum / real.sqrt 2) ^ 2 + (northeast_sum / real.sqrt 2) ^ 2)
  total_distance = (2 / 3) * real.sqrt (2 * real.sqrt 2 + 5) :=
by
  sorry

end ant_distance_l667_667734


namespace isabel_games_problem_l667_667534

noncomputable def prime_sum : ℕ := 83 + 89 + 97

theorem isabel_games_problem (initial_games : ℕ) (X : ℕ) (H1 : initial_games = 90) (H2 : X = prime_sum) : X > initial_games :=
by 
  sorry

end isabel_games_problem_l667_667534


namespace simplify_expression_l667_667607

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b ^ 2 + 2 * b) - 2 * b ^ 2 = 9 * b ^ 3 + 4 * b ^ 2 :=
by
  sorry

end simplify_expression_l667_667607


namespace harry_morning_routine_l667_667399

theorem harry_morning_routine : 
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  buy_time + read_and_eat_time = 45 :=
by
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  show buy_time + read_and_eat_time = 45
  sorry

end harry_morning_routine_l667_667399


namespace Tn_gt_Sn_l667_667877

-- Definitions of the sequences and initial conditions
def a : ℕ → ℕ := λ n, 2 * n + 3
def b (n : ℕ) : ℕ := if n % 2 = 1 then a n - 6 else 2 * a n
def S (n : ℕ) : ℕ := (n * (2 * n + 8))/2
def T : ℕ → ℕ
| 0     := 0
| (n+1) := T n + b (n + 1)

-- Given initial conditions
axiom S_4_eq_32 : S 4 = 32
axiom T_3_eq_16 : T 3 = 16
axiom a_general : ∀ n : ℕ, a n = 2 * n + 3

-- Proof of the main theorem
theorem Tn_gt_Sn (n : ℕ) (h : n > 5) : T n > S n := by
  sorry

end Tn_gt_Sn_l667_667877


namespace ruffles_on_neck_l667_667356

theorem ruffles_on_neck :
  (let cuffs_total := 2 * 50 in
   let hem := 300 in
   let waist := hem / 3 in
   let lace_total := cuffs_total + waist + hem in
   let lace_total_m := lace_total / 100 in
   let cost_lace := 6 * lace_total_m in
   let amount_remaining := 36 - cost_lace in
   let cost_per_cm := 6 / 100 in
   let lace_for_ruffles := amount_remaining / cost_per_cm in
   let ruffles := lace_for_ruffles / 20 
   in ruffles = 5) :=
sorry

end ruffles_on_neck_l667_667356


namespace equal_functions_l667_667466

def f (x : ℝ) : ℝ := abs x
def g (x : ℝ) : ℝ := (sqrt x) ^ 2
def h (x : ℝ) : ℝ := sqrt (x ^ 2)
def s (x : ℝ) : ℝ := x
def piecewise (x : ℝ) : ℝ := if x >= 0 then x else -x

theorem equal_functions :
    (∀ x, f x = g x) ↔ False ∧
    (∀ x, f x = h x) ↔ True ∧
    (∀ x, f x = s x) ↔ False ∧
    (∀ x, f x = piecewise x) ↔ True :=
by sorry

end equal_functions_l667_667466


namespace trig_identity_l667_667853

theorem trig_identity (x : ℝ) (h : Real.sin (π / 6 - x) = 1 / 2) :
  Real.sin (19 * π / 6 - x) + Real.sin (-2 * π / 3 + x) ^ 2 = 1 / 4 :=
by
  sorry

end trig_identity_l667_667853


namespace cot_30_eq_sqrt3_l667_667784

theorem cot_30_eq_sqrt3 : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by
  have h_cos : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  have h_sin : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  show Real.cot (30 * Real.pi / 180) = Real.sqrt 3
  calc
    Real.cot (30 * Real.pi / 180) = (Real.cos (30 * Real.pi / 180)) / (Real.sin (30 * Real.pi / 180)) := sorry
    ... = (Real.sqrt 3 / 2) / (1 / 2) := sorry
    ... = Real.sqrt 3 := sorry

end cot_30_eq_sqrt3_l667_667784


namespace congruent_triangles_properties_l667_667347

-- Definitions for Lean statements:
structure Triangle :=
  (a b c : ℝ) -- sides
  (A B C : ℝ) -- angles in radians

def CongruentTriangles (Δ1 Δ2 : Triangle) : Prop :=
  Δ1.a = Δ2.a ∧ Δ1.b = Δ2.b ∧ Δ1.c = Δ2.c ∧
  Δ1.A = Δ2.A ∧ Δ1.B = Δ2.B ∧ Δ1.C = Δ2.C

-- Statement only, no proof required.
theorem congruent_triangles_properties (Δ1 Δ2 : Triangle) (h : CongruentTriangles Δ1 Δ2) :
  (Δ1.a = Δ2.a ∧ Δ1.b = Δ2.b ∧ Δ1.c = Δ2.c) ∧
  (Δ1.A = Δ2.A ∧ Δ1.B = Δ2.B ∧ Δ1.C = Δ2.C) ∧
  Triangle.perimeter Δ1 = Triangle.perimeter Δ2 ∧
  Triangle.area Δ1 = Triangle.area Δ2 ∧
  ¬ (Triangle.area Δ1 ≠ Triangle.area Δ2) :=
by {
  sorry -- proof goes here
}

end congruent_triangles_properties_l667_667347


namespace girl_scouts_permission_slips_67_l667_667326

noncomputable def percentage_girl_scouts_with_permission_slips 
  (total_scouts : ℕ) 
  (percent_signed_permission_slips : ℝ) 
  (percent_boy_scouts : ℝ) 
  (percent_boy_scouts_with_permission : ℝ) 
  : ℝ :=
  let total_signed_permission_slips := percent_signed_permission_slips / 100 * total_scouts in
  let total_boy_scouts := percent_boy_scouts / 100 * total_scouts in
  let total_boy_scouts_with_permission := percent_boy_scouts_with_permission / 100 * total_boy_scouts |> Float.round in
  let total_girl_scouts_with_permission := total_signed_permission_slips - total_boy_scouts_with_permission in
  let total_girl_scouts := total_scouts - total_boy_scouts in
  (total_girl_scouts_with_permission / total_girl_scouts) * 100

theorem girl_scouts_permission_slips_67
  : percentage_girl_scouts_with_permission_slips 100 60 45 50 = 67 :=
by
  sorry

end girl_scouts_permission_slips_67_l667_667326


namespace problem_statement_l667_667938

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (a : ℝ)
variable (h_even : ∀ x, f (x + 2) = f (-x + 2))
variable (h_deriv : ∀ x, x ≠ 2 → (x - 2) * f' x > 0)
variable (h_a : 2 < a ∧ a < 3)

theorem problem_statement :
  2 < a ∧ a < 3 →
  (∀ x, f (x + 2) = f (-x + 2)) →
  (∀ x, x ≠ 2 → (x - 2) * f' x > 0) →
  f (Real.log 2 a) < f 3 ∧ f 3 < f (2 ^ a) :=
by
  intro h_a h_even h_deriv
  sorry

end problem_statement_l667_667938


namespace pies_sold_in_a_week_l667_667715

theorem pies_sold_in_a_week : 
  let Monday := 8
  let Tuesday := 12
  let Wednesday := 14
  let Thursday := 20
  let Friday := 20
  let Saturday := 20
  let Sunday := 20
  Monday + Tuesday + Wednesday + Thursday + Friday + Saturday + Sunday = 114 :=
by 
  let Monday := 8
  let Tuesday := 12
  let Wednesday := 14
  let Thursday := 20
  let Friday := 20
  let Saturday := 20
  let Sunday := 20
  have h1 : Monday + Tuesday + Wednesday + Thursday + Friday + Saturday + Sunday = 8 + 12 + 14 + 20 + 20 + 20 + 20 := by rfl
  have h2 : 8 + 12 + 14 + 20 + 20 + 20 + 20 = 114 := by norm_num
  exact h1.trans h2

end pies_sold_in_a_week_l667_667715


namespace simplify_expression_l667_667602

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l667_667602


namespace find_2a_plus_b_l667_667165

noncomputable def f (a b x : ℝ) : ℝ := a * x - b
noncomputable def g (x : ℝ) : ℝ := -4 * x + 6
noncomputable def h (a b x : ℝ) : ℝ := f a b (g x)
noncomputable def h_inv (x : ℝ) : ℝ := x + 9

theorem find_2a_plus_b (a b : ℝ) (h_inv_eq: ∀ x : ℝ, h a b (h_inv x) = x) : 2 * a + b = 7 :=
sorry

end find_2a_plus_b_l667_667165


namespace smallest_b_periodicity_l667_667167

variables {α : Type} [add_group α] (f : α → α)

-- Given condition
def periodic_30 (f : α → α) : Prop :=
  ∀ x, f (x - 30) = f x

-- The desired proof
theorem smallest_b_periodicity (hf : periodic_30 f) : ∃ b > 0, (∀ x, f (2x - b) = f (2x)) ∧ b = 60 :=
  sorry

end smallest_b_periodicity_l667_667167


namespace log2_cos_monotonic_decreasing_interval_l667_667633

theorem log2_cos_monotonic_decreasing_interval (k : ℤ) :
  ∀ x, (2 * k * π - π / 4) ≤ x ∧ x < (2 * k * π + π / 4) → 
       ∃ y, y = log 2 (cos (x + π / 4)) :=
by
  sorry

end log2_cos_monotonic_decreasing_interval_l667_667633


namespace C1_cartesian_eq_C2_rect_eq_min_dist_PQ_l667_667526

noncomputable theory

section

-- Conditions
variables (α θ : ℝ)

def C1_param_x (α : ℝ) := (√3) * Real.cos α
def C1_param_y (α : ℝ) := Real.sin α

def C2_polar_eq (ρ θ : ℝ) := ρ * Real.sin (θ + (π / 4)) = 2 * (√2)

-- Cartesian equation for C1
theorem C1_cartesian_eq :
  ∀ (α : ℝ), (C1_param_x α) ^ 2 / 3 + (C1_param_y α) ^ 2 = 1 := 
by
  assume α
  have h1 : (C1_param_x α) ^ 2 = (√3 * Real.cos α) ^ 2 := rfl
  have h2: (C1_param_y α) ^ 2 = (Real.sin α) ^ 2 := rfl
  rw [h1, h2]
  sorry

-- Rectangular equation for C2
theorem C2_rect_eq (ρ θ : ℝ) (hρ : ρ * Real.sin (θ + (π / 4)) = 2 * (√2)) :
  ρ * Real.cos θ + ρ * Real.sin θ = 4 :=
by
  have h3: Real.sin (θ + π / 4) = (Real.sqrt 2 / 2) * Real.sin θ + (Real.sqrt 2 / 2) * Real.cos θ,
  { rw [Real.sin_add, Real.cos_pi_div_four, Real.sin_pi_div_four] }
  rw [h3] at hρ
  sorry

-- Minimum distance |PQ|
theorem min_dist_PQ (α θ : ℝ) (hρ : ρ * Real.sin (θ + (π / 4)) = 2 * (√2)) :
  ∃ P : ℝ × ℝ, ∃ Q : ℝ × ℝ, 
    P = (C1_param_x α, C1_param_y α) ∧
    (Q.1 + Q.2 = 4) ∧
    dist P Q = √2 ∧
    P = (3 / 2, 1 / 2) :=
by
  sorry

end

end C1_cartesian_eq_C2_rect_eq_min_dist_PQ_l667_667526


namespace sum_of_coordinates_l667_667020

-- Definition: point (3, 8) is on y = g(x), meaning g(3) = 8
def g (x : ℝ) : ℝ

axiom g_3_eq_8 : g 3 = 8

-- Problem: Prove that the sum of the coordinates of the point that must be on the graph of 5y = 4g(2x) + 6 is 9.1
theorem sum_of_coordinates : ∃ x y : ℝ, 5 * y = 4 * g (2 * x) + 6 ∧ x + y = 9.1 :=
by
  use (3 / 2)
  use (38 / 5)
  rw [g_3_eq_8]
  sorry

end sum_of_coordinates_l667_667020


namespace sum_of_distinct_prime_factors_420_l667_667284

theorem sum_of_distinct_prime_factors_420 : (2 + 3 + 5 + 7 = 17) :=
by
  -- given condition on the prime factorization of 420
  have h : ∀ {n : ℕ}, n = 420 → ∀ (p : ℕ), p ∣ n → p.prime → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7,
  by sorry

  -- proof of summation of these distinct primes
  sorry

end sum_of_distinct_prime_factors_420_l667_667284


namespace cot_30_eq_sqrt_3_l667_667821

theorem cot_30_eq_sqrt_3 :
  let θ := 30.0
  let cos θ := (real.sqrt 3) / 2
  let sin θ := 1 / 2
  let cot θ := cos θ / sin θ
  cot θ = real.sqrt 3 := by
sorry

end cot_30_eq_sqrt_3_l667_667821


namespace each_employee_gets_five_dollars_l667_667053

-- Define the conditions
def profit_per_day : ℝ := 50
def number_of_employees : ℕ := 9
def percentage_kept_by_owner : ℝ := 0.10

-- Calculate the amount kept by the owner
def amount_kept_by_owner (profit : ℝ) (percentage : ℝ) : ℝ :=
  profit * percentage

-- Calculate the remaining amount to distribute among employees
def remaining_amount (profit : ℝ) (amount_kept : ℝ) : ℝ :=
  profit - amount_kept

-- Calculate how much each employee gets
def amount_per_employee (remaining : ℝ) (employees : ℕ) : ℝ :=
  remaining / employees

-- The equivalent proof problem
theorem each_employee_gets_five_dollars :
  amount_per_employee (remaining_amount profit_per_day (amount_kept_by_owner profit_per_day percentage_kept_by_owner)) number_of_employees = 5 := by
  sorry

end each_employee_gets_five_dollars_l667_667053


namespace square_is_six_l667_667085

def represents_digit (square triangle circle : ℕ) : Prop :=
  square < 10 ∧ triangle < 10 ∧ circle < 10 ∧
  square ≠ triangle ∧ square ≠ circle ∧ triangle ≠ circle

theorem square_is_six :
  ∃ (square triangle circle : ℕ), represents_digit square triangle circle ∧ triangle = 1 ∧ circle = 9 ∧ (square + triangle + 100 * 1 + 10 * 9) = 117 ∧ square = 6 :=
by {
  sorry
}

end square_is_six_l667_667085


namespace quadratic_y_axis_intersection_l667_667202

theorem quadratic_y_axis_intersection :
  (∃ y, (y = (0 - 1) ^ 2 + 2) ∧ (0, y) = (0, 3)) :=
sorry

end quadratic_y_axis_intersection_l667_667202


namespace ducks_joined_l667_667697

theorem ducks_joined (initial_ducks total_ducks ducks_joined : ℕ) 
  (h_initial: initial_ducks = 13)
  (h_total: total_ducks = 33) :
  initial_ducks + ducks_joined = total_ducks → ducks_joined = 20 :=
by
  intros h_equation
  rw [h_initial, h_total] at h_equation
  sorry

end ducks_joined_l667_667697


namespace tan_half_A_mul_tan_half_C_eq_third_l667_667507

variable {A B C : ℝ}
variable {a b c : ℝ}

theorem tan_half_A_mul_tan_half_C_eq_third (h : a + c = 2 * b) :
  (Real.tan (A / 2)) * (Real.tan (C / 2)) = 1 / 3 :=
sorry

end tan_half_A_mul_tan_half_C_eq_third_l667_667507


namespace initial_bottles_count_l667_667154

theorem initial_bottles_count
  (players : ℕ)
  (bottles_per_player_first_break : ℕ)
  (bottles_per_player_end_game : ℕ)
  (remaining_bottles : ℕ)
  (total_bottles_taken_first_break : bottles_per_player_first_break * players = 22)
  (total_bottles_taken_end_game : bottles_per_player_end_game * players = 11)
  (total_remaining_bottles : remaining_bottles = 15) :
  players * bottles_per_player_first_break + players * bottles_per_player_end_game + remaining_bottles = 48 :=
by 
  -- skipping the proof
  sorry

end initial_bottles_count_l667_667154


namespace option_A_correct_l667_667981

theorem option_A_correct (a b : ℝ) (h : a > b) : a + 2 > b + 2 :=
by sorry

end option_A_correct_l667_667981


namespace harry_morning_routine_l667_667400

theorem harry_morning_routine : 
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  buy_time + read_and_eat_time = 45 :=
by
  let buy_time := 15
  let read_and_eat_time := 2 * buy_time
  show buy_time + read_and_eat_time = 45
  sorry

end harry_morning_routine_l667_667400


namespace totalEquilateralTriangles_count_l667_667519

-- Assume we have a diagram with a specific arrangement of equilateral triangles.
variable (diagram : Type)
variable isEquilateralTriangle : diagram → Prop
variable containsOnlyEquilateralTriangles : ∀ t : diagram, isEquilateralTriangle t

-- Define a predicate to count the number of equilateral triangles.
def countEquilateralTriangles (d : diagram) : Nat := sorry -- Placeholder for the actual counting function

-- Statement of the theorem
theorem totalEquilateralTriangles_count (d : diagram) (h : containsOnlyEquilateralTriangles d) : countEquilateralTriangles d = 28 :=
by
  sorry

end totalEquilateralTriangles_count_l667_667519


namespace general_formula_T_greater_S_l667_667899

section arithmetic_sequence

variable {a : ℕ → ℤ} -- Define the arithmetic sequence {a_n}
variable {b : ℕ → ℤ} -- Define the sequence {b_n}
variable {S : ℕ → ℤ} -- Define the sum sequence {S_n}
variable {T : ℕ → ℤ} -- Define the sum sequence {T_n}
variable (a1 : ℤ) (d : ℤ) -- Variables for the first term and common difference 

-- Conditions
def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n, a n = a1 + n * d

def sequence_b (b a : ℕ → ℤ) : Prop :=
  ∀ n, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_sequence_S (S a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (2 * a1 + (n - 1) * d)) / 2

def sum_sequence_T (T b : ℕ → ℤ) : Prop :=
  ∀ n, T n = ∑ i in Finset.range n, b i

def conditions (a b S T : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  arithmetic_sequence a a1 d ∧ sequence_b b a ∧ sum_sequence_S S a ∧ sum_sequence_T T b ∧ S 4 = 32 ∧ T 3 = 16

-- Problem 1: Prove the general formula for {a_n}
theorem general_formula (a : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, a n = 2 * n + 3 :=
sorry

-- Problem 2: Prove that for n > 5, T_n > S_n
theorem T_greater_S (a b S T : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, n > 5 → T n > S n :=
sorry

end arithmetic_sequence

end general_formula_T_greater_S_l667_667899


namespace distance_estimation_l667_667732

theorem distance_estimation (d : ℝ) (h1 : d < 8) (h2 : d > 6) (h3 : d ≠ 7) :
    d ∈ set.Ioo 6 7 ∨ d ∈ set.Ioo 7 8 :=
by
  sorry

end distance_estimation_l667_667732


namespace problem_l667_667220

open Nat

def geom_seq (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, ∀ n : ℕ, a (n + 1) = r * a n

def sum_seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in range (n + 1), a i

def arith_seq (b : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, b (n + 2) = b (n + 1) + b n + d

def Tn (b : ℕ → ℕ) (T : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, T n = ∑ i in range (n + 1), b i

def condition (a b : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 + b 2 = 8 ∧ a 3 + b 3 = 64

theorem problem 
  (a b : ℕ → ℕ) (S T : ℕ → ℕ) (d : ℕ)
  (ha : a 1 = 1)
  (hS : sum_seq a S)
  (ha' : ∀ n : ℕ, a (n + 1) = 2 * S n + 1)
  (hg : geom_seq a)
  (hT : Tn b T)
  (hcond : condition a b)
  (hT3 : T 3 = 15)
  (hd_max : ∀ d' : ℕ, ∃ d : ℕ, arith_seq b d ∧ (T 3 = 15 → b 2 = 5 → T n ≤ T n))
  : T = λ n, 20 * n - 5 * n * n :=
sorry -- proof goes here

end problem_l667_667220


namespace carrot_broccoli_ratio_l667_667572

variables (total_earnings broccoli_earnings cauliflower_earnings spinach_earnings carrot_earnings : ℕ)

-- Define the conditions
def is_condition_satisfied :=
  total_earnings = 380 ∧
  broccoli_earnings = 57 ∧
  cauliflower_earnings = 136 ∧
  spinach_earnings = (carrot_earnings / 2 + 16)

-- Define the proof problem that checks the ratio
theorem carrot_broccoli_ratio (h : is_condition_satisfied total_earnings broccoli_earnings cauliflower_earnings spinach_earnings carrot_earnings) :
  ((carrot_earnings + ((carrot_earnings / 2) + 16)) + broccoli_earnings + cauliflower_earnings = total_earnings) →
  (carrot_earnings / broccoli_earnings = 2) :=
sorry

end carrot_broccoli_ratio_l667_667572


namespace polygon_diagonals_integer_l667_667211

theorem polygon_diagonals_integer (n : ℤ) : ∃ k : ℤ, 2 * k = n * (n - 3) := by
sorry

end polygon_diagonals_integer_l667_667211


namespace average_minutes_per_day_is_correct_l667_667740
-- Import required library for mathematics

-- Define the conditions
def sixth_grade_minutes := 10
def seventh_grade_minutes := 12
def eighth_grade_minutes := 8
def sixth_grade_ratio := 3
def eighth_grade_ratio := 1/2

-- We use noncomputable since we'll rely on some real number operations that are not trivially computable.
noncomputable def total_minutes_per_week (s : ℝ) : ℝ :=
  sixth_grade_minutes * (sixth_grade_ratio * s) * 2 + 
  seventh_grade_minutes * s * 2 + 
  eighth_grade_minutes * (eighth_grade_ratio * s) * 1

noncomputable def total_students (s : ℝ) : ℝ :=
  sixth_grade_ratio * s + s + eighth_grade_ratio * s

noncomputable def average_minutes_per_day : ℝ :=
  (total_minutes_per_week 1) / (total_students 1 / 5)

theorem average_minutes_per_day_is_correct : average_minutes_per_day = 176 / 9 :=
by
  sorry

end average_minutes_per_day_is_correct_l667_667740


namespace find_angle_between_vectors_l667_667113

open Real

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_angle_between_vectors 
  (a b : ℝ × ℝ)
  (a_nonzero : a ≠ (0, 0))
  (b_nonzero : b ≠ (0, 0))
  (ha : vector_norm a = 2)
  (hb : vector_norm b = 3)
  (h_sum : vector_norm (a.1 + b.1, a.2 + b.2) = 1)
  : arccos (dot_product a b / (vector_norm a * vector_norm b)) = π :=
sorry

end find_angle_between_vectors_l667_667113


namespace subset_sum_divisible_l667_667340

open Finset

theorem subset_sum_divisible {n : ℕ} (hn : n ≥ 3) (S : Finset ℕ) (hS : S.card = n-1)
  (h_diff : ∃ x y ∈ S, (x - y) % n ≠ 0) :
  ∃ T ⊆ S, T.nonempty ∧ (T.sum id) % n = 0 :=
sorry

end subset_sum_divisible_l667_667340


namespace cos_diff_identity_l667_667014

theorem cos_diff_identity {α : ℝ} (h1 : α > π / 2 ∧ α < π) (h2 : cos α = -3/5) :
  cos (π / 4 - α) = √2 / 10 := 
sorry

end cos_diff_identity_l667_667014


namespace precision_mult_10_decreases_precision_mult_35_decreases_precision_div_10_increases_precision_div_35_increases_l667_667048

-- Given definitions for precision adjustment
def initial_precision := 3

def new_precision_mult (x : ℕ): ℕ :=
  initial_precision - 1   -- Example: Multiplying by 10 moves decimal point right decreasing precision by 1

def new_precision_mult_large (x : ℕ): ℕ := 
  initial_precision - 2   -- Example: Multiplying by 35 generally decreases precision by 2

def new_precision_div (x : ℕ): ℕ := 
  initial_precision + 1   -- Example: Dividing by 10 moves decimal point left increasing precision by 1

def new_precision_div_large (x : ℕ): ℕ := 
  initial_precision + 1   -- Example: Dividing by 35 generally increases precision by 1

-- Statements to prove
theorem precision_mult_10_decreases: 
  new_precision_mult 10 = 2 := 
by 
  sorry

theorem precision_mult_35_decreases: 
  new_precision_mult_large 35 = 1 := 
by 
  sorry

theorem precision_div_10_increases: 
  new_precision_div 10 = 4 := 
by 
  sorry

theorem precision_div_35_increases: 
  new_precision_div_large 35 = 4 := 
by 
  sorry

end precision_mult_10_decreases_precision_mult_35_decreases_precision_div_10_increases_precision_div_35_increases_l667_667048


namespace tim_pays_300_l667_667242

def mri_cost : ℕ := 1200
def doctor_rate_per_hour : ℕ := 300
def examination_time_in_hours : ℕ := 1 / 2
def consultation_fee : ℕ := 150
def insurance_coverage : ℚ := 0.8

def examination_cost : ℕ := doctor_rate_per_hour * examination_time_in_hours
def total_cost_before_insurance : ℕ := mri_cost + examination_cost + consultation_fee
def insurance_coverage_amount : ℚ := total_cost_before_insurance * insurance_coverage
def amount_tim_pays : ℚ := total_cost_before_insurance - insurance_coverage_amount

theorem tim_pays_300 : amount_tim_pays = 300 := 
by
  -- proof goes here
  sorry

end tim_pays_300_l667_667242


namespace axis_of_symmetry_l667_667081

theorem axis_of_symmetry (a b c n : ℝ)
  (pointA_on_parabola : n = a * 1^2 + b * 1 + c)
  (pointD_on_parabola : n = a * 3^2 + b * 3 + c):
  (axis : ℝ) := axis = 2 :=
by
  sorry

end axis_of_symmetry_l667_667081


namespace pp_eq_zero_has_more_roots_l667_667562

/-- Let \( P(x) \) be a polynomial of odd degree.
  Prove that the equation \( P(P(x)) = 0 \) 
  has at least as many distinct real roots as the equation \( P(x) = 0 \). --/
theorem pp_eq_zero_has_more_roots (P : Polynomial ℝ) (h1 : P.degree % 2 = 1) :
  (∃ k, P.rootMultiplicity k (P.comp P) > 0) ∧ 
  (∀ x, P.is_root x → ∃ y, P y = x ∧ P.is_root y) →
  (P.rootMultiplicity k (P.comp P) > P.rootMultiplicity k P) :=
sorry

end pp_eq_zero_has_more_roots_l667_667562


namespace problem_A_problem_B_problem_C_problem_D_l667_667668

theorem problem_A (a b: ℝ) (h : b > 0 ∧ a > b) : ¬(1/a > 1/b) := 
by {
  sorry
}

theorem problem_B (a b: ℝ) (h : a < b ∧ b < 0): (a^2 > a*b) := 
by {
  sorry
}

theorem problem_C (a b: ℝ) (h : a > b): ¬(|a| > |b|) := 
by {
  sorry
}

theorem problem_D (a: ℝ) (h : a > 2): (a + 4/(a-2) ≥ 6) := 
by {
  sorry
}

end problem_A_problem_B_problem_C_problem_D_l667_667668


namespace length_of_segment_inside_cube_l667_667419

open Real

-- Define the points X and Y
def X : ℝ × ℝ × ℝ := (0, 0, 0)
def Y : ℝ × ℝ × ℝ := (5, 5, 13)

-- Define the distance function in 3D space
def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2 + (q.3 - p.3)^2)

-- Coordinates of the segment within the cube with edge length 4
def segment_start : ℝ × ℝ × ℝ := (0, 0, 4)
def segment_end : ℝ × ℝ × ℝ := (4, 4, 8)

-- Problem statement
theorem length_of_segment_inside_cube :
  distance segment_start segment_end = 4 * sqrt 3 :=
by
  sorry

end length_of_segment_inside_cube_l667_667419


namespace cot_30_eq_sqrt_3_l667_667807

theorem cot_30_eq_sqrt_3 (h : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3) : 
  Real.cot (Real.pi / 6) = Real.sqrt 3 :=
by
  sorry

end cot_30_eq_sqrt_3_l667_667807


namespace definite_integral_sin4_cos4_l667_667751

theorem definite_integral_sin4_cos4 :
  ∫ x in 0..(2 * Real.pi), (Real.sin (3 * x)) ^ 4 * (Real.cos (3 * x)) ^ 4 = 3 * Real.pi / 64 := 
sorry

end definite_integral_sin4_cos4_l667_667751


namespace cot_30_eq_sqrt_3_l667_667803

theorem cot_30_eq_sqrt_3 (h : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3) : 
  Real.cot (Real.pi / 6) = Real.sqrt 3 :=
by
  sorry

end cot_30_eq_sqrt_3_l667_667803


namespace amount_distribution_l667_667682

theorem amount_distribution :
  ∃ (P Q R S T : ℝ), 
    (P + Q + R + S + T = 24000) ∧ 
    (R = (3 / 5) * (P + Q)) ∧ 
    (S = 0.45 * 24000) ∧ 
    (T = (1 / 2) * R) ∧ 
    (P + Q = 7000) ∧ 
    (R = 4200) ∧ 
    (S = 10800) ∧ 
    (T = 2100) :=
by
  sorry

end amount_distribution_l667_667682


namespace petrol_price_increase_l667_667215

variable (P C : ℝ)

/- The original price of petrol is P per unit, and the user consumes C units of petrol.
   The new consumption after a 28.57142857142857% reduction is (5/7) * C units.
   The expenditure remains constant, i.e., P * C = P' * (5/7) * C.
-/
theorem petrol_price_increase (h : P * C = (P * (7/5)) * (5/7) * C) :
  (P * (7/5) - P) / P * 100 = 40 :=
by
  sorry

end petrol_price_increase_l667_667215


namespace count_divisibles_l667_667485

theorem count_divisibles : (finset.range (51 + 1)).filter (λ n, n % 4 = 0 ∨ n % 6 = 0).card = 16 := by
  sorry

end count_divisibles_l667_667485


namespace cot_30_deg_l667_667811

-- Define the known values and identities
def tan_30_deg := 1 / Real.sqrt 3

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by 
  sorry

end cot_30_deg_l667_667811


namespace pressure_increases_when_block_submerged_l667_667289

theorem pressure_increases_when_block_submerged 
  (P0 : ℝ) (ρ : ℝ) (g : ℝ) (h0 h1 : ℝ) :
  h1 > h0 → 
  (P0 + ρ * g * h1) > (P0 + ρ * g * h0) :=
by
  intros h1_gt_h0
  sorry

end pressure_increases_when_block_submerged_l667_667289


namespace volume_surface_area_ratio_l667_667323

theorem volume_surface_area_ratio
  (V : ℕ := 9)
  (S : ℕ := 34)
  (shape_conditions : ∃ n : ℕ, n = 9 ∧ ∃ m : ℕ, m = 2) :
  V / S = 9 / 34 :=
by
  sorry

end volume_surface_area_ratio_l667_667323


namespace sum_distinct_prime_factors_of_420_l667_667280

theorem sum_distinct_prime_factors_of_420 : 
  ∑ (p : ℕ) in {2, 3, 5, 7}, p = 17 := 
by 
  sorry

end sum_distinct_prime_factors_of_420_l667_667280


namespace sqrt_7225_minus_55_eq_form_c_d_l667_667168

theorem sqrt_7225_minus_55_eq_form_c_d :
  ∃ (c d : ℕ), (c > 0) ∧ (d > 0) ∧ (sqrt 7225 - 55 = (sqrt c - d) ^ 3) ∧ (c + d = 19) :=
by
  sorry

end sqrt_7225_minus_55_eq_form_c_d_l667_667168


namespace max_distance_positions_l667_667311

noncomputable def hook : (ℝ × ℝ) := (0, 0)

noncomputable def M0 : (ℝ × ℝ) := (-1, 2 * sqrt 2)
noncomputable def N0 : (ℝ × ℝ) := (-2, 4 * sqrt 2)

noncomputable def speed_ratio : ℝ := 2.5

theorem max_distance_positions :
  ∃ (positions : list (ℝ × ℝ)), positions = [(4 - sqrt 2, 4 + sqrt 2), (4 + sqrt 2, sqrt 2 - 4), (sqrt 2 - 4, -4 - sqrt 2), (-sqrt 2 - 4, 4 - sqrt 2)] :=
sorry

end max_distance_positions_l667_667311


namespace equation_of_line_l667_667577

def Point := (ℝ × ℝ)

def P0 : Point := (1, 2)
def P1 : Point := (3, 2)

theorem equation_of_line : ∀ (P : Point), (∃ x : ℝ, P = (x, 2)) ∧ (P = P0 ∨ P = P1) → ∀ y : ℝ, y = 2 :=
by
  intros P h
  cases h with hx hy
  cases hy with hp0 hp1
  . have hpx : P = P0 := hp0
    cases P0 with x0 y0
    have hy0 : y0 = 2 := by simp [P0] 
    rw [hpx, ← hy0]
    exact hx
  . have hpx : P = P1 := hp1
    cases P1 with x1 y1
    have hy1 : y1 = 2 := by simp [P1] 
    rw [hpx, ← hy1]
    exact hx
  sorry

end equation_of_line_l667_667577


namespace Tn_gt_Sn_l667_667876

-- Definitions of the sequences and initial conditions
def a : ℕ → ℕ := λ n, 2 * n + 3
def b (n : ℕ) : ℕ := if n % 2 = 1 then a n - 6 else 2 * a n
def S (n : ℕ) : ℕ := (n * (2 * n + 8))/2
def T : ℕ → ℕ
| 0     := 0
| (n+1) := T n + b (n + 1)

-- Given initial conditions
axiom S_4_eq_32 : S 4 = 32
axiom T_3_eq_16 : T 3 = 16
axiom a_general : ∀ n : ℕ, a n = 2 * n + 3

-- Proof of the main theorem
theorem Tn_gt_Sn (n : ℕ) (h : n > 5) : T n > S n := by
  sorry

end Tn_gt_Sn_l667_667876


namespace polynomial_value_at_4_l667_667257

def f (x : ℝ) : ℝ := 9 + 15 * x - 8 * x^2 - 20 * x^3 + 6 * x^4 + 3 * x^5

theorem polynomial_value_at_4 :
  f 4 = 3269 :=
by
  sorry

end polynomial_value_at_4_l667_667257


namespace hyperbola_sum_proof_l667_667069

theorem hyperbola_sum_proof :
  let h := 3
  let k := -1
  let a := |7 - 3|
  let c := sqrt 41
  let b := sqrt (c^2 - a^2)
  h + k + a + b = 11 := by
  sorry

end hyperbola_sum_proof_l667_667069


namespace general_formula_T_greater_S_l667_667896

section arithmetic_sequence

variable {a : ℕ → ℤ} -- Define the arithmetic sequence {a_n}
variable {b : ℕ → ℤ} -- Define the sequence {b_n}
variable {S : ℕ → ℤ} -- Define the sum sequence {S_n}
variable {T : ℕ → ℤ} -- Define the sum sequence {T_n}
variable (a1 : ℤ) (d : ℤ) -- Variables for the first term and common difference 

-- Conditions
def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n, a n = a1 + n * d

def sequence_b (b a : ℕ → ℤ) : Prop :=
  ∀ n, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_sequence_S (S a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (2 * a1 + (n - 1) * d)) / 2

def sum_sequence_T (T b : ℕ → ℤ) : Prop :=
  ∀ n, T n = ∑ i in Finset.range n, b i

def conditions (a b S T : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  arithmetic_sequence a a1 d ∧ sequence_b b a ∧ sum_sequence_S S a ∧ sum_sequence_T T b ∧ S 4 = 32 ∧ T 3 = 16

-- Problem 1: Prove the general formula for {a_n}
theorem general_formula (a : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, a n = 2 * n + 3 :=
sorry

-- Problem 2: Prove that for n > 5, T_n > S_n
theorem T_greater_S (a b S T : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, n > 5 → T n > S n :=
sorry

end arithmetic_sequence

end general_formula_T_greater_S_l667_667896


namespace find_x_minus_y_l667_667430

theorem find_x_minus_y (x y : ℤ) (h1 : |x| = 5) (h2 : y^2 = 16) (h3 : x + y > 0) : x - y = 1 ∨ x - y = 9 := 
by sorry

end find_x_minus_y_l667_667430


namespace part1_and_part2_l667_667891

-- Define the arithmetic sequence {a_n}
def a (n : Nat) : Nat := 2 * n + 3

-- Define the sequence {b_n}
def b (n : Nat) : Nat :=
  if n % 2 = 0 then 4 * n + 6 else 2 * n - 3

-- Define the sum of the first n terms of a sequence
def summation (seq : Nat → Nat) (n : Nat) : Nat :=
  (List.range n).map seq |>.sum

-- Define S_n as the sum of the first n terms of {a_n}
def S (n : Nat) : Nat := summation a n

-- Define T_n as the sum of the first n terms of {b_n}
def T (n : Nat) : Nat := summation b n

-- Given conditions
axiom S4_eq_32 : S 4 = 32
axiom T3_eq_16 : T 3 = 16

-- Prove the general formula for {a_n} and that T_n > S_n for n > 5
theorem part1_and_part2 (n : Nat) (h : n > 5) : a n = 2 * n + 3 ∧ T n > S n :=
  by
  sorry

end part1_and_part2_l667_667891


namespace volume_of_right_prism_l667_667200

theorem volume_of_right_prism (α β d : ℝ) 
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hd : 0 < d) : 
  volume_of_right_prism α β d = (d^3 * sin β * sin (2 * β) * sin (2 * α)) / 8 := 
sorry

end volume_of_right_prism_l667_667200


namespace triangle_area_heron_l667_667007

theorem triangle_area_heron (a b c : ℝ) (ha : a = 12) (hb : b = 15) (hc : c = 20) : 
  let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = 85.3 :=
by
  sorry

end triangle_area_heron_l667_667007


namespace width_of_domain_of_g_l667_667557

-- Define the function h with its domain
def h (x : ℝ) : ℝ := sorry

noncomputable def g (x : ℝ) := h (x / 3)

theorem width_of_domain_of_g :
  ∀ (h : ℝ → ℝ), (∀ (x : ℝ), -10 ≤ x ∧ x ≤ 10 → (h x) = (h x)) → 
  (∃ a b : ℝ, a = -30 ∧ b = 30 ∧ b - a = 60)  := 
by 
  intros h domain_condition
  use [-30, 30]
  split
  sorry


end width_of_domain_of_g_l667_667557


namespace part1_part2_part3_l667_667033

-- Definitions
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 1
def g (x : ℝ) : ℝ := Real.exp x
def y (x : ℝ) (a : ℝ) : ℝ := f x a * g x
def h (x : ℝ) (a : ℝ) : ℝ := f x a / g x

-- Part 1
theorem part1 (a : ℝ) (h_a : a = -1) : 
  ∃ x ∈ Icc (-1 : ℝ) 2, y x a = 3 * Real.exp 2 :=
sorry

-- Part 2
theorem part2 (a : ℝ) (h_a : a = -1) (k : ℝ) :
  (∀ x : ℝ, h x a ≠ k) ∨ (∃! x : ℝ, h x a = k) ↔
  (k > 3 / Real.exp 2) ∨ (0 < k ∧ k < 1 / Real.exp 1) :=
sorry

-- Part 3
theorem part3 : 
  (∀ x1 x2 ∈ Icc 0 2, x1 ≠ x2 → |f x1 a - f x2 a| < |g x1 - g x2|) ↔ 
  -1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2 :=
sorry

end part1_part2_part3_l667_667033


namespace david_invested_time_l667_667583

-- Definitions based on the problem conditions
def P := 698               -- The principal amount
def A1 := 815              -- Amount Peter got back
def A2 := 854              -- Amount David got back
def t1 := 3                -- Time Peter invested
def rate := (11700 / 2094) -- Rate of interest found from Peter's case

-- The statement to be proved
theorem david_invested_time : ∃ t2 : ℝ, A2 = P + (P * rate * t2) / 100 ∧ t2 = 4 :=
begin
  -- show the time t2 that David invested, which should be 4 years
  sorry
end

end david_invested_time_l667_667583


namespace angle_C_in_triangle_l667_667508

theorem angle_C_in_triangle (a c : ℝ) (angle_A : ℝ) (h₁ : a = 2 * sqrt 3) (h₂ : c = 2 * sqrt 2) (h₃ : angle_A = π / 3) : 
  ∃ angle_C : ℝ, angle_C = π / 4 :=
by
  sorry

end angle_C_in_triangle_l667_667508


namespace sum_distinct_prime_factors_of_420_l667_667279

theorem sum_distinct_prime_factors_of_420 : 
  ∑ (p : ℕ) in {2, 3, 5, 7}, p = 17 := 
by 
  sorry

end sum_distinct_prime_factors_of_420_l667_667279


namespace root_in_interval_l667_667172

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem root_in_interval :
  (0 < 2) → (0 < 3) →
  (∀ x, 0 < x → continuous_at (f) x) →
  f 2 < 0 →
  f 3 > 0 →
  ∃ c, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  intros h2pos h3pos hcont hf2 hf3
  sorry

end root_in_interval_l667_667172


namespace regression_line_mean_y_l667_667959

def mean (s : Set ℝ) : ℝ := (s.toList.sum) / (s.card : ℝ)

theorem regression_line_mean_y :
  let x : Set ℝ := {1, 7, 10, 13, 19}
  let regression_line := λ x, 1.5 * x + 45
  let x_bar := mean x
  let y_bar := regression_line x_bar
  y_bar = 60 := by
  sorry

end regression_line_mean_y_l667_667959


namespace max_sector_area_l667_667939

noncomputable def area_of_sector (l r : ℝ) : ℝ :=
  (20 - r) * r

theorem max_sector_area (l r : ℝ) (h1 : l + 2 * r = 40) : ∃ r, r = 10 ∧ area_of_sector l r = 100 :=
by
  use 10
  split
  . rfl
  . unfold area_of_sector
    calc
    (20 - 10) * 10 = 10 * 10 : by norm_num
         ... = 100 : by norm_num
sorry

end max_sector_area_l667_667939


namespace vinegar_solution_max_volume_l667_667645

theorem vinegar_solution_max_volume:
  ∃ V p, (250 * 0.70 + 500 * 0.05 = V * (p / 100)) ∧ 
  (0 < V) ∧ (5 < p) ∧ (p <= 80/3) :=
begin
  sorry
end

end vinegar_solution_max_volume_l667_667645


namespace general_formula_a_n_T_n_greater_than_S_n_l667_667927

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n_l667_667927


namespace part1_and_part2_l667_667884

-- Define the arithmetic sequence {a_n}
def a (n : Nat) : Nat := 2 * n + 3

-- Define the sequence {b_n}
def b (n : Nat) : Nat :=
  if n % 2 = 0 then 4 * n + 6 else 2 * n - 3

-- Define the sum of the first n terms of a sequence
def summation (seq : Nat → Nat) (n : Nat) : Nat :=
  (List.range n).map seq |>.sum

-- Define S_n as the sum of the first n terms of {a_n}
def S (n : Nat) : Nat := summation a n

-- Define T_n as the sum of the first n terms of {b_n}
def T (n : Nat) : Nat := summation b n

-- Given conditions
axiom S4_eq_32 : S 4 = 32
axiom T3_eq_16 : T 3 = 16

-- Prove the general formula for {a_n} and that T_n > S_n for n > 5
theorem part1_and_part2 (n : Nat) (h : n > 5) : a n = 2 * n + 3 ∧ T n > S n :=
  by
  sorry

end part1_and_part2_l667_667884


namespace red_hat_ratio_l667_667646

open Nat

noncomputable def garden_gnomes : Type := sorry

variables (total_gnomes : garden_gnomes)
          (red_hats blue_hats : garden_gnomes)
          (big_noses small_noses : garden_gnomes)
          (red_hats_big_noses blue_hats_big_noses red_hats_small_noses : garden_gnomes)

-- The total number of gnomes is 28.
axiom total_gnomes_count : total_gnomes = 28

-- Half of the garden gnomes have big noses, which means 28 / 2 = 14 gnomes with big noses.
axiom half_big_noses : big_noses = 14

-- Six gnomes with blue hats have big noses.
axiom six_blue_hats_big_noses : blue_hats_big_noses = 6

-- There are 13 gnomes with red hats that have small noses.
axiom thirteen_red_hats_small_noses : red_hats_small_noses = 13

theorem red_hat_ratio :
  (8 + 13) / 28 = 3 / 4 := by
    sorry

end red_hat_ratio_l667_667646


namespace framed_painting_ratio_l667_667712

theorem framed_painting_ratio
  (width_painting : ℕ)
  (height_painting : ℕ)
  (frame_side : ℕ)
  (frame_top_bottom : ℕ)
  (h1 : width_painting = 20)
  (h2 : height_painting = 30)
  (h3 : frame_top_bottom = 3 * frame_side)
  (h4 : (width_painting + 2 * frame_side) * (height_painting + 2 * frame_top_bottom) = 2 * width_painting * height_painting):
  (width_painting + 2 * frame_side) = 1/2 * (height_painting + 2 * frame_top_bottom) := 
by
  sorry

end framed_painting_ratio_l667_667712


namespace smallest_mn_sum_l667_667091

theorem smallest_mn_sum (m n : ℕ) (h : 3 * n ^ 3 = 5 * m ^ 2) : m + n = 60 :=
sorry

end smallest_mn_sum_l667_667091


namespace trajectory_C2_AM_AN_constant_l667_667442

-- Define the geometric entities and their relationships as conditions
variable (x y k : ℝ)

def is_circle (c : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.fst - c.fst)^2 + (p.snd - c.snd)^2 = r^2

def point_G : ℝ × ℝ := (5, 4)
def circle_C1 := is_circle (1, 4) 5
def line_l (G : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
    (5 * p.fst + 4 * p.snd = 1) -- Equation of the line passing through point G and some transformation

variable {p E F : ℝ × ℝ}

-- Midpoint condition
def is_midpoint (A B M : ℝ × ℝ) : Prop :=
  M.fst = (A.fst + B.fst) / 2 ∧ M.snd = (A.snd + B.snd) / 2

-- Line definition
def line_l1 (k : ℝ) (A : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  k * p.fst - p.snd - k = 0 

def line_l2 (p : ℝ × ℝ) : Prop :=
  p.fst + 2 * p.snd + 2 = 0 

-- Definitions of points and results
def point_N (k : ℝ) : ℝ × ℝ :=
  (2 * k - 2) / (2 * k + 1), - 3 * k / (2 * k + 1)

def point_M (k : ℝ) : ℝ × ℝ :=
  (k^2 + 4 * k + 3) / (1 + k^2), - (4 * k^2 + 2 * k) / (1 + k^2)

def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.fst - B.fst)^2 + (A.snd - B.snd)^2)

-- Theorem Statements
theorem trajectory_C2 : 
  (C : ℝ × ℝ) → ∀ {G : ℝ × ℝ} (E F : ℝ × ℝ), line_l G E → line_l G F → is_midpoint E F C → circle_C1 E → circle_C1 F → 
  is_circle (3, 4) 2 C := sorry

theorem AM_AN_constant :
  ∀ (k : ℝ), let A : ℝ × ℝ := (1, 0) in 
  let M := point_M k in 
  let N := point_N k in 
  distance A M * distance A N = 6 := sorry

end trajectory_C2_AM_AN_constant_l667_667442


namespace max_books_bound_l667_667070

def Students : Type := Fin 35

def num_students : ℕ := 35
def num_borrowed (s : Students) : ℕ
-- Specifically given numbers:
def zero_borrowed : ℕ := 2
def one_borrowed : ℕ := 12
def two_borrowed : ℕ := 10

-- Average books per student:
def avg_books_per_student : ℕ := 2
def total_books : ℕ := num_students * avg_books_per_student

-- Known number of books borrowed:
def books_by_known_students : ℕ := (zero_borrowed * 0) + (one_borrowed * 1) + (two_borrowed * 2)

-- Remaining students who borrowed at least 3 books:
def remaining_students : ℕ := num_students - zero_borrowed - one_borrowed - two_borrowed
def min_books_by_remaining : ℕ := remaining_students * 3
def remaining_books : ℕ := total_books - books_by_known_students

-- Maximum number of books a single student can borrow
def max_books_a_student_can_borrow : ℕ := min_books_by_remaining + (remaining_books - min_books_by_remaining)

theorem max_books_bound (s : Students) :
  max_books_a_student_can_borrow = 8 :=
by
  sorry

end max_books_bound_l667_667070


namespace log_five_one_over_twenty_five_eq_neg_two_l667_667388

theorem log_five_one_over_twenty_five_eq_neg_two
  (h1 : (1 : ℝ) / 25 = 5^(-2))
  (h2 : ∀ (a b c : ℝ), (c > 0) → (a ≠ 0) → (log b (a^c) = c * log b a))
  (h3 : log 5 5 = 1) :
  log 5 (1 / 25) = -2 := 
  by 
    sorry

end log_five_one_over_twenty_five_eq_neg_two_l667_667388


namespace find_lowest_score_l667_667195

constant mean15 : ℝ := 90
constant mean13 : ℝ := 92
constant numScores : ℕ := 15
constant maxScore : ℝ := 110

theorem find_lowest_score : 
  let totalSum := mean15 * numScores
  let remainingScores := numScores - 2
  let subtractedSum := mean13 * remainingScores
  let highLowSum := totalSum - subtractedSum
  highLowSum - maxScore = 44 := 
by
  let totalSum := mean15 * numScores
  let remainingScores := numScores - 2
  let subtractedSum := mean13 * remainingScores
  let highLowSum := totalSum - subtractedSum
  show highLowSum - maxScore = 44 from sorry

end find_lowest_score_l667_667195


namespace T3_area_is_sqrt3_l667_667719

def T1_area : ℝ := 36
def T1_side : ℝ := real.sqrt T1_area
def T2_side : ℝ := (2 / 3) * T1_side
def T3_side : ℝ := T2_side / 2
def T3_area : ℝ := (real.sqrt 3 / 4) * T3_side^2

theorem T3_area_is_sqrt3 : T3_area = real.sqrt 3 := sorry

end T3_area_is_sqrt3_l667_667719


namespace limit_of_f_is_x_l667_667030

noncomputable def f (x : ℝ) : ℝ := x

axiom limit_eq_derivative_at_one : 
  (∀ (Δx : ℝ), Δx ≠ 0 → (f (1 + Δx) - f 1) / Δx = 1) →

  ∃ (L : ℝ), filter.tendsto (λ Δx, (f (1 + Δx) - f 1) / Δx)
    (filter.nhds 0) (𝓝 L) ∧ L = 1

theorem limit_of_f_is_x :
  ∃ (L : ℝ), filter.tendsto (λ Δx, (f (1 + Δx) - f 1) / Δx) 
    (filter.nhds 0) (𝓝 L) ∧ L = 1 :=
begin
  apply limit_eq_derivative_at_one,
  intro Δx,
  intro hΔx,
  rw [f, f],
  calc 
    (1 + Δx - 1) / Δx = Δx / Δx : by ring
    ... = 1 : by rw div_self hΔx,
end

end limit_of_f_is_x_l667_667030


namespace last_number_remaining_128_l667_667097

-- Define a function to simulate the marking-out process
def lastRemainingNumber (n : Nat) : Nat :=
  let rec markOut (remaining : List Nat) (step : Nat) : List Nat :=
    remaining.enum.filter (λ (i, x) => (i + 1) % (step + 1) != 0) |>.map Prod.snd
  let rec loop (remaining : List Nat) (step : Nat) : Nat :=
    match markOut remaining step with
    | []      => remaining.head! -- In case the list is empty, return the head of remaining
    | [x]     => x               -- Only one number left, return it
    | newLst  => loop newLst (step + 1)
  loop (List.range' 1 (n+1)) 1
  
theorem last_number_remaining_128 : lastRemainingNumber 150 = 128 := by
  sorry

end last_number_remaining_128_l667_667097


namespace coordinates_of_A_l667_667515

/-- The initial point A and the transformations applied to it -/
def initial_point : Prod ℤ ℤ := (-3, 2)

def translate_right (p : Prod ℤ ℤ) (units : ℤ) : Prod ℤ ℤ :=
  (p.1 + units, p.2)

def translate_down (p : Prod ℤ ℤ) (units : ℤ) : Prod ℤ ℤ :=
  (p.1, p.2 - units)

/-- Proof that the point A' has coordinates (1, -1) -/
theorem coordinates_of_A' : 
  translate_down (translate_right initial_point 4) 3 = (1, -1) :=
by
  sorry

end coordinates_of_A_l667_667515


namespace worker_B_time_l667_667298

-- Defining the work rates and the combined completion time
variable (t : ℝ)
variable (A_rate : ℝ) (B_rate : ℝ)

-- Conditions based on the problem statement
def worker_A_rate := 1 / 5
def combined_rate := 1 / 3.75

theorem worker_B_time :
  A_rate = worker_A_rate →
  B_rate = 1 / t →
  (A_rate + B_rate = combined_rate) →
  t = 15 :=
  by sorry

end worker_B_time_l667_667298


namespace range_of_a_l667_667944

theorem range_of_a (a : ℝ) (h : ∃ c ∈ set.Icc (-2 : ℝ) 1, 2 * c - a + 1 = 0) : -3 ≤ a ∧ a ≤ 3 :=
by {
  sorry
}

end range_of_a_l667_667944


namespace log_base_5_of_fraction_l667_667377

theorem log_base_5_of_fraction (a b : ℕ) (h1 : b ≠ 0) (h2 : a = 5) (h3 : b = 2) : log 5 (1 / (a ^ b)) = -2 := by
  sorry

end log_base_5_of_fraction_l667_667377


namespace find_lowest_score_l667_667196

constant mean15 : ℝ := 90
constant mean13 : ℝ := 92
constant numScores : ℕ := 15
constant maxScore : ℝ := 110

theorem find_lowest_score : 
  let totalSum := mean15 * numScores
  let remainingScores := numScores - 2
  let subtractedSum := mean13 * remainingScores
  let highLowSum := totalSum - subtractedSum
  highLowSum - maxScore = 44 := 
by
  let totalSum := mean15 * numScores
  let remainingScores := numScores - 2
  let subtractedSum := mean13 * remainingScores
  let highLowSum := totalSum - subtractedSum
  show highLowSum - maxScore = 44 from sorry

end find_lowest_score_l667_667196


namespace equivalence_of_statements_l667_667078

noncomputable theory

variable (n : ℕ) -- number of castles
variable (A B : finset ℕ) -- castles in country A and B
variable (adj : ℕ → ℕ → Prop) -- adjacency relation between castles
variable (commander_A commander_B : ℕ → ℕ) -- number of commanders in each castle from countries A and B

-- Definition for statement 1
def statement1 : Prop :=
  ∀ (B_attack A_defend : ℕ → ℕ),
    (∀ i ∈ B, ∀ j ∈ A, adj i j → B_attack i ≤ commander_B i) ∧
    (∀ i ∈ A, A_defend i ≤ commander_A i) ∧
    (∀ i ∈ A, A_defend i ≥ ∑ j in B, if adj j i then B_attack j else 0)

-- Definition for statement 2
def statement2 : Prop :=
  ∀ X ⊆ A,
    (∀ Y : finset ℕ, Y = {i ∈ B | ∃ j ∈ X, adj i j} →
     X.card + {i ∈ A | ∃ j ∈ X, adj j i}.card ≥ Y.card)

-- Proof problem for equivalence
theorem equivalence_of_statements : statement1 n A B adj commander_A commander_B ↔ statement2 n A B adj commander_A commander_B := sorry

end equivalence_of_statements_l667_667078


namespace timothy_percentage_l667_667702

noncomputable def grass_problem : Prop :=
  let mixture_per_acre := 600 / 15 in
  let timothy_per_acre := 2 in
  let percentage_timothy := (timothy_per_acre / mixture_per_acre) * 100 in
  percentage_timothy = 5

theorem timothy_percentage : grass_problem := by
  sorry

end timothy_percentage_l667_667702


namespace true_propositions_l667_667366

def nearest_integer (x : ℝ) : ℤ :=
  if H : ∃ m : ℤ, m - 1/2 < x ∧ x ≤ m + 1/2 then (classical.some H) else 0

def f (x : ℝ) : ℝ := 
  let m := nearest_integer x in x - m

theorem true_propositions : 
  ∃ (p1 p3 : Prop), p1 ∧ p3 ∧ 
  (p1 → (domain : set.univ = set.univ ∧ (range f = set.Icc (-1/2) (1/2)))) ∧ 
  (p3 → (has_period f 1)) :=
by sorry

end true_propositions_l667_667366


namespace bubble_mix_l667_667330

theorem bubble_mix (
  (soap_per_cup : ℝ) -- 3 tablespoons of soap per cup of water
  (capacity_oz : ℝ) -- 40 ounces
  (oz_per_cup : ℝ) -- 8 ounces per cup
  (soap_needed : ℝ) -- tablespoons of soap needed
) :
  soap_per_cup = 3 → capacity_oz = 40 → oz_per_cup = 8 → soap_needed = (capacity_oz / oz_per_cup) * soap_per_cup → soap_needed = 15 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end bubble_mix_l667_667330


namespace cot_30_eq_sqrt_3_l667_667808

theorem cot_30_eq_sqrt_3 (h : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3) : 
  Real.cot (Real.pi / 6) = Real.sqrt 3 :=
by
  sorry

end cot_30_eq_sqrt_3_l667_667808


namespace sum_of_intersection_points_l667_667950

theorem sum_of_intersection_points :
  let f (x : ℝ) := 2 * (Real.sin (x + Real.pi / 4)) ^ 2
  let g (x : ℝ) := 1 + Real.cos (x / 2 + Real.pi / 4)
  let interval := Set.Ioo (Real.pi / 2 - m) (Real.pi / 2 + m)
  -- Let xi be the x-coordinates of the intersection points
  --  such that ∀ i ∈ {1, 2, ..., 9}, (xi, yi) are intersection points of f and g in the given interval
  {Σi_1 i_2 i_3 i_4 i_5 i_6 i_7 i_8 i_9 yi_1 yi_2 yi_3 yi_4 yi_5 yi_6 yi_7 yi_8 yi_9,
    ∀ i : 9, ∃ xi yi, 
      xi ∈ interval ∧
      (f xi = yi ∧ g xi = yi) ∧
      xi = [i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8, i_9][i] ∧
      yi = [yi_1, yi_2, yi_3, yi_4, yi_5, yi_6, yi_7, yi_8, yi_9][i]
  } →
  (Σi_1 i_2 i_3 i_4 i_5 i_6 i_7 i_8 i_9 y_ij_1 y_ij_2 y_ij_3 y_ij_4 y_ij_5 y_ij_6 y_ij_7 y_ij_8 y_ij_9,
    ∑i_1...i_9 + ∑y_ij_1...y_ij_9 = (9 * Real.pi) / 2 + 9) :=
by sorry


end sum_of_intersection_points_l667_667950


namespace angle_MNB_l667_667563

theorem angle_MNB (A B C D M N : Type*) 
  (h1 : is_quadrilateral A B C D)
  (h2 : ∠A B C = 90)
  (h3 : ∠B A D = 80)
  (h4 : ∠A D C = 80)
  (h5 : lies_on M [A, D])
  (h6 : lies_on N [B, C])
  (h7 : ∠C D N = 20)
  (h8 : ∠A B M = 20)
  (h9 : dist M D = dist A B) : 
  ∠M N B = 70 :=
sorry

end angle_MNB_l667_667563


namespace center_of_symmetry_monotonicity_interval_1_monotonicity_interval_2_l667_667470

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x)^2 - (Real.cos (x + Real.pi / 3))^2

-- 1. Proving the center of symmetry
theorem center_of_symmetry (k : ℤ) :
  ∃ x₀, x₀ = (k * Real.pi / 2) + (Real.pi / 12) ∧ f x₀ = 0 := sorry

-- 2. Discussing monotonicity on the interval [-π/3, π/4]
theorem monotonicity_interval_1 :
  ∀ x y ∈ Icc (-Real.pi / 3) (-Real.pi / 6),
  x < y → f x > f y := sorry

theorem monotonicity_interval_2 :
  ∀ x y ∈ Icc (-Real.pi / 6) (Real.pi / 4),
  x < y → f x < f y := sorry

end center_of_symmetry_monotonicity_interval_1_monotonicity_interval_2_l667_667470


namespace general_formula_T_greater_S_l667_667892

section arithmetic_sequence

variable {a : ℕ → ℤ} -- Define the arithmetic sequence {a_n}
variable {b : ℕ → ℤ} -- Define the sequence {b_n}
variable {S : ℕ → ℤ} -- Define the sum sequence {S_n}
variable {T : ℕ → ℤ} -- Define the sum sequence {T_n}
variable (a1 : ℤ) (d : ℤ) -- Variables for the first term and common difference 

-- Conditions
def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n, a n = a1 + n * d

def sequence_b (b a : ℕ → ℤ) : Prop :=
  ∀ n, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_sequence_S (S a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (2 * a1 + (n - 1) * d)) / 2

def sum_sequence_T (T b : ℕ → ℤ) : Prop :=
  ∀ n, T n = ∑ i in Finset.range n, b i

def conditions (a b S T : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  arithmetic_sequence a a1 d ∧ sequence_b b a ∧ sum_sequence_S S a ∧ sum_sequence_T T b ∧ S 4 = 32 ∧ T 3 = 16

-- Problem 1: Prove the general formula for {a_n}
theorem general_formula (a : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, a n = 2 * n + 3 :=
sorry

-- Problem 2: Prove that for n > 5, T_n > S_n
theorem T_greater_S (a b S T : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, n > 5 → T n > S n :=
sorry

end arithmetic_sequence

end general_formula_T_greater_S_l667_667892


namespace part1_part2_l667_667473

def f (x y : ℝ) : ℝ := (2 * x + y) / (x + 2 * y)

theorem part1 : f 1 2 = 1 / f 2 1 :=
by { sorry }

theorem part2 (c : ℝ) : f c c = - f (-c) c :=
by { sorry }

end part1_part2_l667_667473


namespace sum_of_reciprocals_l667_667498

noncomputable def solve (x y : ℚ) : ℚ := x + y

theorem sum_of_reciprocals (x y : ℚ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = -2) :
  solve x y = 4 / 3 :=
by
  -- Placeholder for the actual proof
  sorry

end sum_of_reciprocals_l667_667498


namespace cot_30_eq_sqrt3_l667_667780

theorem cot_30_eq_sqrt3 : Real.cot (π / 6) = Real.sqrt 3 := by
  let tan_30 := 1 / Real.sqrt 3
  have tan_30_eq : Real.tan (π / 6) = tan_30 := sorry
  have cot_def : ∀ x, Real.cot x = 1 / Real.tan x := sorry
  rw [cot_def (π / 6), tan_30_eq]
  simp
  sorry

end cot_30_eq_sqrt3_l667_667780


namespace pressure_increases_when_block_submerged_l667_667291

-- Definitions and conditions
variables (P_0 : ℝ) (ρ : ℝ) (g : ℝ) (h_0 : ℝ) (h_1 : ℝ)
hypothesis (h1_gt_h0 : h_1 > h_0)

-- The proof goal
theorem pressure_increases_when_block_submerged (P_0 ρ g h_0 h_1 : ℝ) (h1_gt_h0 : h_1 > h_0) : 
  let P := P_0 + ρ * g * h_0 in
  let P_1 := P_0 + ρ * g * h_1 in
  P_1 > P := sorry

end pressure_increases_when_block_submerged_l667_667291


namespace pq_proof_l667_667911

section ProofProblem

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, ∃ d, a (n + 1) = a n + d

def b_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
if n % 2 = 1 then a n - 6 else 2 * a n

def sum_first_n (seq : ℕ → ℤ) (n : ℕ) : ℤ :=
(n.sum (λ i, seq (i + 1)))

variables (a : ℕ → ℤ)
variables (S T : ℕ → ℤ)

axiom a_is_arithmetic : arithmetic_sequence a
axiom S_4_eq_32 : sum_first_n a 4 = 32
axiom T_3_eq_16 : sum_first_n (b_n a) 3 = 16

theorem pq_proof : (∀ n, a n = 2 * n + 3) ∧ (∀ n > 5, sum_first_n (b_n a) n > sum_first_n a n) :=
sorry

end ProofProblem

end pq_proof_l667_667911


namespace proposition_converses_correct_count_l667_667203

theorem proposition_converses_correct_count :
  let p1 := ∀ {α β : ℝ}, α = β → ⟨(α,β⟩ ∧ α = -β;
      p2 := ∀ {α β : ℝ}, α = β → (α = β) → α = -β;
      p3 := ∀ {α β : ℝ}, α = β → (α = β) → α = β;
      p4 := ∀ {α β : ℝ}, α = β ↔ α = β;
  (∀ p1.virtual_angles_are_equal : ¬ p1,
  ∀ p2.supplementary_angles_complementary_parallel : p2,
  ∀ p3.corresponding_angles_congruent_triangles : ¬ p3).
  ∃ p : p4 → Prop,
  (#(
      if not_p2 ∧ if not_p4) = 2 ∧
  ) :=
  sorry

end proposition_converses_correct_count_l667_667203


namespace log_base_5_of_fraction_l667_667375

theorem log_base_5_of_fraction (a b : ℕ) (h1 : b ≠ 0) (h2 : a = 5) (h3 : b = 2) : log 5 (1 / (a ^ b)) = -2 := by
  sorry

end log_base_5_of_fraction_l667_667375


namespace problem_statement_l667_667000

noncomputable def f1 (x : ℝ) := Real.sin x + Real.cos x
noncomputable def f (n : ℕ) (x : ℝ) : ℝ := 
  (Real.iterate (λ g : ℝ → ℝ, λ x, Real.deriv g x) n f1) x

theorem problem_statement :
  f 1 (Real.pi / 3) + f 2 (Real.pi / 3) + f 3 (Real.pi / 3) + 
  ∑ i in Finset.range 2014, f (i + 4 + 1) (Real.pi / 3) = 
  (1 + Real.sqrt 3) / 2 :=
sorry

end problem_statement_l667_667000


namespace log_five_one_over_twenty_five_eq_neg_two_l667_667387

theorem log_five_one_over_twenty_five_eq_neg_two
  (h1 : (1 : ℝ) / 25 = 5^(-2))
  (h2 : ∀ (a b c : ℝ), (c > 0) → (a ≠ 0) → (log b (a^c) = c * log b a))
  (h3 : log 5 5 = 1) :
  log 5 (1 / 25) = -2 := 
  by 
    sorry

end log_five_one_over_twenty_five_eq_neg_two_l667_667387


namespace correct_k_λ_l667_667969

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b (k : ℝ) : ℝ × ℝ := (-1, k)
noncomputable def λ_solution : ℝ := -2
noncomputable def k_solution : ℝ := -1 / 2

theorem correct_k_λ (k : ℝ) (λ : ℝ) 
  (h1 : vec_a = (λ * vec_b k)) : 
  λ = λ_solution ∧ k = k_solution :=
sorry

end correct_k_λ_l667_667969


namespace smaller_circle_radius_l667_667072

theorem smaller_circle_radius :
  ∃ r : ℝ, (10 - r = 2 * r) ∧ r = 10 / 3 :=
begin
  existsi (10 / 3 : ℝ),
  simp,
end

end smaller_circle_radius_l667_667072


namespace product_of_fractions_l667_667278

-- Definitions from the conditions
def a : ℚ := 2 / 3 
def b : ℚ := 3 / 5
def c : ℚ := 4 / 7
def d : ℚ := 5 / 9

-- Statement of the proof problem
theorem product_of_fractions : a * b * c * d = 8 / 63 := 
by
  sorry

end product_of_fractions_l667_667278


namespace range_of_ax2_l667_667429

theorem range_of_ax2 (f : ℝ → ℝ) (a : ℝ) (x1 x2 : ℝ)
  (h_f : ∀ x : ℝ, f x = exp x / (x^2 + a))
  (h_a : a > 0)
  (h_extreme_points : (x1 < x2) ∧ (∀ x : ℝ, f' x = (exp x * (x^2 - 2*x + a)) / ((x^2 + a)^2)) ∧ (QuadraticDiscriminant (λ x, x^2 - 2*x + a) = 4 - 4*a) ∧
                      (QuadraticRoots (λ x, x^2 - 2*x + a) = (1 - Real.sqrt(1 - a), 1 + Real.sqrt(1 - a)))
                      ) :
  ∀ x2 : ℝ, g : ℝ → ℝ,
  (ax2 : ℝ := a * (1 + Real.sqrt(1 - a))),
  (t := Real.sqrt(1 - a)),
  (g t = 1 - t^2 * (1 + t) = - t^3 - t^2 + t + 1) →
  ∃ t : ℝ, 0 < t < 1 ∧ (ax2_range : ℝ := g t) ∧ (ax2_range ∈ (0, 32/27]) :=
begin 
  sorry
end

end range_of_ax2_l667_667429


namespace cot_30_eq_sqrt_3_l667_667823

theorem cot_30_eq_sqrt_3 :
  let θ := 30.0
  let cos θ := (real.sqrt 3) / 2
  let sin θ := 1 / 2
  let cot θ := cos θ / sin θ
  cot θ = real.sqrt 3 := by
sorry

end cot_30_eq_sqrt_3_l667_667823


namespace parallel_lines_sufficient_not_necessary_l667_667616

theorem parallel_lines_sufficient_not_necessary (a : ℝ) :
  ((a = 3) → (∀ x y : ℝ, (a * x + 2 * y + 1 = 0) → (3 * x + (a - 1) * y - 2 = 0)) ∧ 
  (∀ x y : ℝ, (a * x + 2 * y + 1 = 0) ∧ (3 * x + (a - 1) * y - 2 = 0) → (a = 3 ∨ a = -2))) :=
sorry

end parallel_lines_sufficient_not_necessary_l667_667616


namespace ratio_average_speed_l667_667372

-- Define the conditions based on given distances and times
def distanceAB : ℕ := 600
def timeAB : ℕ := 3

def distanceBD : ℕ := 540
def timeBD : ℕ := 2

def distanceAC : ℕ := 460
def timeAC : ℕ := 4

def distanceCE : ℕ := 380
def timeCE : ℕ := 3

-- Define the total distances and times for Eddy and Freddy
def distanceEddy : ℕ := distanceAB + distanceBD
def timeEddy : ℕ := timeAB + timeBD

def distanceFreddy : ℕ := distanceAC + distanceCE
def timeFreddy : ℕ := timeAC + timeCE

-- Define the average speeds for Eddy and Freddy
def averageSpeedEddy : ℚ := distanceEddy / timeEddy
def averageSpeedFreddy : ℚ := distanceFreddy / timeFreddy

-- Prove the ratio of their average speeds is 19:10
theorem ratio_average_speed (h1 : distanceAB = 600) (h2 : timeAB = 3) 
                           (h3 : distanceBD = 540) (h4 : timeBD = 2)
                           (h5 : distanceAC = 460) (h6 : timeAC = 4) 
                           (h7 : distanceCE = 380) (h8 : timeCE = 3):
  averageSpeedEddy / averageSpeedFreddy = 19 / 10 := by sorry

end ratio_average_speed_l667_667372


namespace b_arithmetic_sequence_sum_of_a_n_l667_667527

-- Definitions for the sequences a_n and b_n
def a : ℕ → ℝ
| 0       := 1
| (n + 1) := (1/3) * a n + (2 / 3^n) - (2 / 3)

def b (n : ℕ) : ℝ := 3^(n-1) * (a n + 1)

-- Sum of the first n terms of {a_n}
def S (n : ℕ) : ℝ :=
  let T := (Finset.range n).sum (λ k, 2 * (↑k + 1) / 3^k)
  (T - n)

-- First problem: {b_n} is an arithmetic sequence
theorem b_arithmetic_sequence : ∀ n: ℕ, b n = 2*n :=
  sorry

-- Second problem: Sum of the first n terms of {a_n} equals to given expression
theorem sum_of_a_n (n : ℕ) : S n = (9 / 2) - ((2 * n + 3) / (2 * 3^(n-1))) - n :=
  sorry

end b_arithmetic_sequence_sum_of_a_n_l667_667527


namespace incorrect_statement_D_l667_667955

noncomputable def hyperbola_C1 := x^2 / 4 - y^2 / 3 = 1
noncomputable def hyperbola_C2 := x^2 / 4 - y^2 / 3 = -1

theorem incorrect_statement_D :
  ∀ (C1 C2 : ℝ → ℝ → Prop),
    (C1 = (λ x y, x^2 / 4 - y^2 / 3 = 1)) →
    (C2 = (λ x y, x^2 / 4 - y^2 / 3 = -1)) →
    ¬ (eccentricity C1 = eccentricity C2) :=
by sorry

end incorrect_statement_D_l667_667955


namespace simplify_expression_l667_667597

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l667_667597


namespace rhombus_side_is_24_approx_l667_667222

noncomputable def rhombus_side_length (d1 : ℝ) (area : ℝ) : ℝ :=
  let d2 := (2 * area) / d1
  sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)

theorem rhombus_side_is_24_approx (d1 : ℝ) (area : ℝ) :
  d1 = 18 → area = 400.47 → abs (rhombus_side_length d1 area - 24) < 0.01 :=
by
  intros h1 h2
  rw [h1, h2]
  -- Skipping actual calculation verification
  sorry

end rhombus_side_is_24_approx_l667_667222


namespace AM_GM_Inequality_four_vars_l667_667486

theorem AM_GM_Inequality_four_vars (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / b + b / c + c / d + d / a ≥ 4 :=
by sorry

end AM_GM_Inequality_four_vars_l667_667486


namespace polynomial_min_bound_l667_667010

theorem polynomial_min_bound (n : ℕ) (a : Fin n → ℝ) (P x : ℝ → ℝ) 
    (hp : ∀ x, P(x) = ∑ i in Finset.range(n+1), a i * x^(n - i)) 
    (m : ℝ) 
    (hm : m = Finset.min' {Finset.sum (Finset.range k) (λ j, a j) | k in Finset.range(n+1)}) 
    (hx : x ≥ 1) :
  P(x) ≥ m * x^n := 
sorry

end polynomial_min_bound_l667_667010


namespace f_is_odd_f_min_value_pos_f_minimum_at_2_f_increasing_intervals_l667_667554

noncomputable def f (x : ℝ) : ℝ := x + 4/x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

theorem f_min_value_pos : ∀ x : ℝ, x > 0 → f x ≥ 4 :=
by
  sorry

theorem f_minimum_at_2 : f 2 = 4 :=
by
  sorry

theorem f_increasing_intervals : (MonotoneOn f {x | x ≤ -2} ∧ MonotoneOn f {x | x ≥ 2}) :=
by
  sorry

end f_is_odd_f_min_value_pos_f_minimum_at_2_f_increasing_intervals_l667_667554


namespace empty_subset_singleton_zero_l667_667295

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) :=
by
  sorry

end empty_subset_singleton_zero_l667_667295


namespace find_K_l667_667541

noncomputable def cylinder_paint (r h : ℝ) : ℝ := 2 * Real.pi * r * h
noncomputable def cube_surface_area (s : ℝ) : ℝ := 6 * s^2
noncomputable def cube_volume (s : ℝ) : ℝ := s^3

theorem find_K :
  (cylinder_paint 3 4 = 24 * Real.pi) →
  (∃ s, cube_surface_area s = 24 * Real.pi ∧ cube_volume s = 48 / Real.sqrt K) →
  K = 36 / Real.pi^3 :=
by
  sorry

end find_K_l667_667541


namespace immune_response_problem_l667_667701

open ProbabilityTheory

variables (M: ℝ → ℝ) (σ: ℝ) (μ: ℝ)
variables (number_of_people: ℝ)
variables (condition1 : 1 ≤ 1000)
variables (condition2 : ∀ x, M x ≈ Normal 15 σ^2)
variables (condition3 : (P (λ x, 10 < x ∧ x < 20) (Normal 15 σ^2)) = 19 / 25)

theorem immune_response_problem
  (h1 : ∀ x, True → P (λ x, x ≤ 10) (Normal 15 σ^2) = 3 / 25)
  (h2 : ∀ x, True → P (λ x, x ≥ 20) (Normal 15 σ^2) = 3 / 25)
  (h3 : ∀ x, True → P (λ x, x ≤ 20) (Normal 15 σ^2) = 22 / 25)
  : (number_of_people = 1000) →
    number_of_people * 22 / 25 = 880 := 
begin
  intros,
  sorry
end

end immune_response_problem_l667_667701


namespace probability_of_draws_l667_667066

noncomputable def problem_probability (draws : ℕ) : ℚ :=
  if draws = 16 then (10 : ℚ) / 16 else 0

theorem probability_of_draws
  (balls : Finset ℕ)
  (m n : ℕ)
  (hballs : balls = {1, 2, 3, 4}) 
  (hm_mem : m ∈ balls)
  (hn_mem : n ∈ balls)
  (total_outcomes := balls.card * balls.card)
  (favorable_outcomes := 
    ((∑ m in balls, ∑ n in balls, if n < m + 1 then 1 else 0) : ℚ)) :
  problem_probability total_outcomes = favorable_outcomes / total_outcomes := 
sorry

end probability_of_draws_l667_667066


namespace probability_factors_of_120_l667_667331

noncomputable def factorial (n: ℕ) : ℕ :=
nat.factorial n

def set_24 := {x : ℕ | 1 ≤ x ∧ x ≤ 24}

def factors_of_120 := {x ∈ set_24 | x ∣ factorial 5}

def count (s : set ℕ) : ℕ := fintype.card {x // set_24 x}

def probability : ℚ := count factors_of_120 / count set_24

theorem probability_factors_of_120 :
  probability = 1 / 2 :=
by sorry

end probability_factors_of_120_l667_667331


namespace probability_odd_ball_l667_667079

def balls := {1, 2, 3, 4, 5}
def odd_balls := {x ∈ balls | x % 2 = 1}
def total_balls := balls.size
def number_of_odd_balls := odd_balls.size

theorem probability_odd_ball
  (h_total : total_balls = 5)
  (h_odd : number_of_odd_balls = 3) :
  (number_of_odd_balls * 1.0) / (total_balls * 1.0) = 3 / 5 :=
by
  sorry

end probability_odd_ball_l667_667079


namespace cost_per_mile_first_plan_l667_667046

theorem cost_per_mile_first_plan 
  (initial_fee : ℝ) (cost_per_mile_first : ℝ) (cost_per_mile_second : ℝ) (miles : ℝ)
  (h_first : initial_fee = 65)
  (h_cost_second : cost_per_mile_second = 0.60)
  (h_miles : miles = 325)
  (h_equal_cost : initial_fee + miles * cost_per_mile_first = miles * cost_per_mile_second) :
  cost_per_mile_first = 0.40 :=
by
  sorry

end cost_per_mile_first_plan_l667_667046


namespace find_f_l667_667452

theorem find_f {f : ℝ → ℝ} (h : ∀ x : ℝ, f (x - 1) = x^2 - 1) : ∀ x : ℝ, f x = x^2 + 2*x := 
by
  sorry

end find_f_l667_667452


namespace part1_and_part2_l667_667886

-- Define the arithmetic sequence {a_n}
def a (n : Nat) : Nat := 2 * n + 3

-- Define the sequence {b_n}
def b (n : Nat) : Nat :=
  if n % 2 = 0 then 4 * n + 6 else 2 * n - 3

-- Define the sum of the first n terms of a sequence
def summation (seq : Nat → Nat) (n : Nat) : Nat :=
  (List.range n).map seq |>.sum

-- Define S_n as the sum of the first n terms of {a_n}
def S (n : Nat) : Nat := summation a n

-- Define T_n as the sum of the first n terms of {b_n}
def T (n : Nat) : Nat := summation b n

-- Given conditions
axiom S4_eq_32 : S 4 = 32
axiom T3_eq_16 : T 3 = 16

-- Prove the general formula for {a_n} and that T_n > S_n for n > 5
theorem part1_and_part2 (n : Nat) (h : n > 5) : a n = 2 * n + 3 ∧ T n > S n :=
  by
  sorry

end part1_and_part2_l667_667886


namespace range_of_a_l667_667057

structure MonotonicFunction (a : ℝ) :=
  (f : ℝ → ℝ)
  (monotonic : ∀ x y : ℝ, x ≤ y → f x ≤ f y)
  
def piecewiseFunction (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ 7 then (3 - a) * x - 3 else a ^ (x - 6)

theorem range_of_a :
  {a : ℝ | ∃ f : ℝ → ℝ, 
    (MonotonicFunction.mk f 
      (λ x y h, if x ≤ 7 then 
        by sorry 
      else 
        by sorry) 
    ) ∧ f = piecewiseFunction a 
  } = set.Ico (9 / 4) 3 :=
by sorry

end range_of_a_l667_667057


namespace pq_proof_l667_667908

section ProofProblem

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, ∃ d, a (n + 1) = a n + d

def b_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
if n % 2 = 1 then a n - 6 else 2 * a n

def sum_first_n (seq : ℕ → ℤ) (n : ℕ) : ℤ :=
(n.sum (λ i, seq (i + 1)))

variables (a : ℕ → ℤ)
variables (S T : ℕ → ℤ)

axiom a_is_arithmetic : arithmetic_sequence a
axiom S_4_eq_32 : sum_first_n a 4 = 32
axiom T_3_eq_16 : sum_first_n (b_n a) 3 = 16

theorem pq_proof : (∀ n, a n = 2 * n + 3) ∧ (∀ n > 5, sum_first_n (b_n a) n > sum_first_n a n) :=
sorry

end ProofProblem

end pq_proof_l667_667908


namespace rectangular_to_polar_l667_667758

theorem rectangular_to_polar (x y : ℝ) (r θ : ℝ) (hx : x = 6) (hy : y = -2 * real.sqrt 3) 
  (hr : r = 4 * real.sqrt 3) (hθ : θ = 11 * real.pi / 6) : 
  (r > 0 ∧ 0 ≤ θ ∧ θ < 2 * real.pi) ∧ (x = r * real.cos θ ∧ y = r * real.sin θ) := 
sorry

end rectangular_to_polar_l667_667758


namespace acute_angles_sine_relation_l667_667505

theorem acute_angles_sine_relation (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_eq : 2 * Real.sin α = Real.sin α * Real.cos β + Real.cos α * Real.sin β) : α < β :=
by
  sorry

end acute_angles_sine_relation_l667_667505


namespace conic_decomposition_l667_667129

variables {ℂ : Type*} [IsAlgClosed ℂ] [Field ℂ]

structure Point :=
(x : ℂ)
(y : ℂ)

structure Line (p q : Point) :=
(equation : ℂ → ℂ → ℂ)
(on_line : ∀ r : Point, equation r.x r.y = 0 ↔ (r = p ∨ r = q))

structure Conic :=
(equation : ℂ → ℂ → ℂ)
(on_conic : ∀ p : Point, equation p.x p.y = 0)

variables {A B C D : Point}
variables (conic : Conic) (hA : conic.on_conic A) (hB : conic.on_conic B) (hC : conic.on_conic C) (hD : conic.on_conic D)

noncomputable def λ : ℂ := sorry
noncomputable def μ : ℂ := sorry

noncomputable def l_AB : Line A B := sorry
noncomputable def l_CD : Line C D := sorry
noncomputable def l_BC : Line B C := sorry
noncomputable def l_AD : Line A D := sorry

theorem conic_decomposition :
  conic.equation = λ * l_AB.equation + μ * l_CD.equation + λ * l_BC.equation + μ * l_AD.equation :=
sorry

end conic_decomposition_l667_667129


namespace radius_of_circle_point_on_circle_exists_l667_667009

-- Define the circle equation and given conditions
def circle (r : ℝ) : set (ℝ × ℝ) := { P : ℝ × ℝ | P.1^2 + P.2^2 = r^2 }
def A : ℝ × ℝ := (0, 5)
def B : ℝ × ℝ := (5, 0)

-- Define the first question's condition and answer
theorem radius_of_circle : ∃ (r : ℝ), r > 0 ∧ (A ∈ circle r) ∧ r = 5 :=
by {
  use 5,
  split,
  { exact zero_lt_five },
  split, 
  { simp [A, circle] },
  { refl }
}

-- Define the condition for the second question
def P (x y : ℝ) : Prop := 
  (x, y) ∈ circle 5 ∧ abs (x + y - 5) = 6 * real.sqrt 2

-- The second theorem to confirm the existence of point P
theorem point_on_circle_exists : ∃ (x y : ℝ), P x y :=
by {
  let solutions := [(3, -4), (-4, 3)],
  cases solutions with coords,
  cases coords with x y,
  use x,
  use y,
  split,
  { simp [circle, x, y], norm_num },
  { simp [x, y], norm_num }
}

end radius_of_circle_point_on_circle_exists_l667_667009


namespace company_workers_count_l667_667673

-- Definitions
def num_supervisors := 13
def team_leads_per_supervisor := 3
def workers_per_team_lead := 10

-- Hypothesis
def team_leads := num_supervisors * team_leads_per_supervisor
def workers := team_leads * workers_per_team_lead

-- Theorem to prove
theorem company_workers_count : workers = 390 :=
by
  sorry

end company_workers_count_l667_667673


namespace cot_30_deg_l667_667800

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 := by
  -- Define known values of trigonometric functions at 30 degrees
  have h1 : Real.tan (30 * Real.pi / 180) = 1 / Real.sqrt 3 := sorry
  have h2 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

  -- Prove using these known values
  calc
    Real.cot (30 * Real.pi / 180)
        = 1 / Real.tan (30 * Real.pi / 180) : by sorry
    ... = 1 / (1 / Real.sqrt 3) : by sorry
    ... = Real.sqrt 3 : by sorry

end cot_30_deg_l667_667800


namespace gini_inconsistency_l667_667687

-- Definitions of given conditions
def X : ℝ := 0.5
def G0 : ℝ := 0.1
def Y : ℝ := 0.4
def Y' : ℝ := 0.2

-- After the reform, defining the shares
def Z : ℝ -- income share for the middle class
def share_rich : ℝ := 1 - (Y' + Z)

-- Correct value to be checked
def G : ℝ := 0.1

-- Proof object to express that maintaining the same Gini coefficient is impossible
theorem gini_inconsistency : ∀ (Z : ℝ), G ≠ (1 / 2 - (1 / 6 * 0.2 + 1 / 6 * (0.2 + Z) + 1 / 6)) / (1 / 2) := 
by
  sorry

end gini_inconsistency_l667_667687


namespace reema_loan_rate_of_interest_l667_667683

theorem reema_loan_rate_of_interest :
  ∃ (R : ℝ), 
    (∀ (P L : ℝ), P = 1200 ∧ L = 432 → 
    ((L = P * R * R / 100) ∧ R * R = 36)) ↔ R = 6 :=
begin
  sorry
end

end reema_loan_rate_of_interest_l667_667683


namespace annual_interest_rate_is_6_percent_l667_667095

-- Definitions from the conditions
def principal : ℕ := 150
def total_amount_paid : ℕ := 159
def interest := total_amount_paid - principal
def interest_rate := (interest * 100) / principal

-- The theorem to prove
theorem annual_interest_rate_is_6_percent :
  interest_rate = 6 := by sorry

end annual_interest_rate_is_6_percent_l667_667095


namespace billy_reads_three_books_on_weekend_l667_667748

def billy_weekend_reading (hours_per_day : ℕ) (days : ℕ) (video_game_time_fraction : ℚ) (reading_speed : ℕ) (pages_per_book : ℕ) : ℕ :=
  let total_hours := hours_per_day * days
  let reading_hours := (1 - video_game_time_fraction) * total_hours
  let total_pages := reading_speed * reading_hours.toNat
  total_pages / pages_per_book

theorem billy_reads_three_books_on_weekend : billy_weekend_reading 8 2 (75 / 100 : ℚ) 60 80 = 3 := by
  sorry

end billy_reads_three_books_on_weekend_l667_667748


namespace melody_snowman_drawings_l667_667574

theorem melody_snowman_drawings (cards: ℕ) (drawings_per_card: ℕ) (h1: cards = 13) (h2: drawings_per_card = 4) : cards * drawings_per_card = 52 := by
  rw [h1, h2]
  -- Proof goes here
  sorry

end melody_snowman_drawings_l667_667574


namespace female_students_not_at_ends_has_36_arrangements_l667_667238

-- Definitions
def male_students : ℕ := 3
def female_students : ℕ := 2
def positions : ℕ := 5

-- Condition: Female students do not stand at the ends
def valid_positions_for_females : ℕ := 3

-- Total arrangements where female students do not stand at the ends
theorem female_students_not_at_ends_has_36_arrangements :
  (∃ posF : fin 3 → fin 5, (∀ i, 1 ≤ posF i ∧ posF i ≤ 3) ∧ ∃ permM : fin 3 → fin 5, (∀ i, permM i ∈ {posF 0, posF 1, posF 2} ∧ permM i ≠ posF i)) →
  ∃ (arrangements : ℕ), arrangements = 36 :=
sorry

end female_students_not_at_ends_has_36_arrangements_l667_667238


namespace server_multiplications_in_half_hour_l667_667717

theorem server_multiplications_in_half_hour : 
  let rate := 5000
  let seconds_in_half_hour := 1800
  rate * seconds_in_half_hour = 9000000 := by
  sorry

end server_multiplications_in_half_hour_l667_667717


namespace chocolate_chips_per_batch_l667_667319

theorem chocolate_chips_per_batch (total_chocolate_chips : ℝ) (number_of_batches : ℝ) : (total_chocolate_chips = 23.0) ∧ (number_of_batches = 11.5) → (total_chocolate_chips / number_of_batches = 2) :=
by
  intro h
  cases h with h_total h_batches
  rw [h_total, h_batches]
  norm_num
  sorry

end chocolate_chips_per_batch_l667_667319


namespace BE_length_l667_667145

-- Define a structure for points in the plane.
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a predicate for points being collinear.
def collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) = (C.x - A.x) * (B.y - A.y)

-- Define a predicate for intersection of lines.
def intersect (A B C D : Point) : Point :=
  let t := ((C.x - A.x) * (A.y - B.y) - (C.y - A.y) * (A.x - B.x)) /
            ((D.x - C.x) * (A.y - B.y) - (D.y - C.y) * (A.x - B.x))
  ⟨C.x + t * (D.x - C.x), C.y + t * (D.y - C.y)⟩

variables (A B C D F : Point) 

-- Assume necessary conditions for the problem.
axiom parallelogram : collinear A B D ∧ collinear B C D ∧ collinear A D F
axiom BF_inter_AC_at_E : intersect B F A C
axiom BF_inter_DC_at_G : intersect B F D C
axiom EF_length : (EF : ℝ) = 32
axiom GF_length : (GF : ℝ) = 24

-- Define points E and G at the intersection locations
def E : Point := intersect B F A C
def G : Point := intersect B F D C

-- Create the theorem statement
theorem BE_length : dist B E = 16 :=
sorry

end BE_length_l667_667145


namespace transformed_sine_function_l667_667575

theorem transformed_sine_function :
  (∀ x : ℝ, true) →
  (∀ x : ℝ, sin x = sin (2 * (x + π / 3))) :=
by
  sorry

end transformed_sine_function_l667_667575


namespace problem_statement_l667_667110

variable (a b c p q r α β γ : ℝ)

-- Given conditions
def plane_condition : Prop := (a / α) + (b / β) + (c / γ) = 1
def sphere_conditions : Prop := p^3 = α ∧ q^3 = β ∧ r^3 = γ

-- The statement to prove
theorem problem_statement (h_plane : plane_condition a b c α β γ) (h_sphere : sphere_conditions p q r α β γ) :
  (a / p^3) + (b / q^3) + (c / r^3) = 1 := sorry

end problem_statement_l667_667110


namespace iterated_g_l667_667118

def g (x : ℝ) : ℝ := 1 / (1 - x)

theorem iterated_g (x : ℝ) (hx : g (g x) = x) : g (g (g (g (g (g x))))) = x :=
by
  -- Assuming that hx holds, we don't need to provide proof details here.
  sorry

example : g (g (g (g (g (g (1 / 2)))))) = 1 / 2 :=
by
  have hx : g (g (1 / 2)) = 1 / 2 := by
    calc
      g (1 / 2) = 1 / (1 - 1 / 2) : by simp [g]
      ... = 2 : by simp [g]
      g (g (1 / 2)) = g 2 : by simp
      ... = 1 / (1 - 2) : by simp [g, hx]
      ... = -1 : by simp [g, hx]
      
  apply iterated_g
  exact hx

end iterated_g_l667_667118


namespace Kelly_remaining_games_l667_667100

-- Definitions according to the conditions provided
def initial_games : ℝ := 121.0
def given_away : ℝ := 99.0
def remaining_games : ℝ := initial_games - given_away

-- The proof problem statement
theorem Kelly_remaining_games : remaining_games = 22.0 :=
by
  -- sorry is used here to skip the proof
  sorry

end Kelly_remaining_games_l667_667100


namespace train_speed_l667_667708

theorem train_speed (train_length platform_length time_taken : ℕ) 
(h_train_length : train_length = 300)
(h_platform_length : platform_length = 220)
(h_time_taken : time_taken = 26) :
(train_length + platform_length) / time_taken * 3.6 = 72 := 
sorry

end train_speed_l667_667708


namespace fraction_power_mul_reciprocal_frac_mul_reciprocal_result_l667_667659

theorem fraction_power_mul_reciprocal 
  (a b : ℚ) (h1 : a = 5) (h2 : b = 6) :
  (a / b)^4 * (a / b)^(-2) = (a / b)^2 := by
  sorry

theorem frac_mul_reciprocal_result : 
  (5 / 6)^4 * (5 / 6)^(-2) = 25 / 36 := by 
  have h : (5 / 6)^4 * (5 / 6)^(-2) = (5 / 6)^2 := by apply fraction_power_mul_reciprocal 5 6 rfl rfl 
  rw h
  norm_num

end fraction_power_mul_reciprocal_frac_mul_reciprocal_result_l667_667659


namespace number_of_chickens_l667_667139

def cost_per_chicken := 3
def total_cost := 15
def potato_cost := 6
def remaining_amount := total_cost - potato_cost

theorem number_of_chickens : (total_cost - potato_cost) / cost_per_chicken = 3 := by
  sorry

end number_of_chickens_l667_667139


namespace lowest_score_of_15_scores_l667_667176

theorem lowest_score_of_15_scores (scores : List ℝ) (h_len : scores.length = 15) (h_mean : (scores.sum / 15) = 90)
    (h_mean_13 : ((scores.erase 110).erase (scores.head)).sum / 13 = 92) (h_highest : scores.maximum = some 110) :
    scores.minimum = some 44 :=
by
    sorry

end lowest_score_of_15_scores_l667_667176


namespace find_ω_find_a_l667_667467

noncomputable def given_function (ω : ℝ) (x : ℝ) : ℝ :=
  4 * Real.cos (ω * x) * Real.sin (ω * x - π / 6)

theorem find_ω :
  ∃ ω > 0, (∀ x, given_function ω (x + π / (2 * ω)) = given_function ω x) ∧ 1 = ω :=
sorry

def given_triangle (A B C : ℝ) (a b c S: ℝ) : Prop :=
  (∀ A, A ∈ (0, π / 2)) ∧
  (∀ x, given_function (1) x = 0 → x = A) ∧
  Real.sin B = sqrt 3 * Real.sin C ∧
  S = 2 * sqrt 3 ∧
  A = π / 6 ∧
  1 / 2 * b * c * Real.sin A = 2 * sqrt 3

theorem find_a (A B C a b c S : ℝ) (h : given_triangle A B C a b c S) :
  a = 2 * sqrt 2 :=
sorry

end find_ω_find_a_l667_667467


namespace hypotenuse_proof_l667_667716

noncomputable def hypotenuse_length (a b c : ℝ) (k : ℝ) : Prop :=
  a + b + c = 40 ∧ (1 / 2) * a * b = 30 ∧ a = 3 * k ∧ b = 4 * k ∧ c = 5 * sqrt 5

theorem hypotenuse_proof (a b c k : ℝ) : hypotenuse_length a b c k → c = 5 * sqrt 5 :=
by
  intro h
  -- Proof skipped
  sorry

end hypotenuse_proof_l667_667716


namespace find_lowest_score_l667_667183

variable (scores : Fin 15 → ℕ)
variable (highest_score : ℕ := 110)

-- Conditions
def arithmetic_mean_15 := (∑ i, scores i) / 15 = 90
def mean_without_highest_lowest := (∑ i in Finset.erase (Finset.erase (Finset.univ : Finset (Fin 15)) (Fin.mk 0 sorry)) (Fin.mk 14 sorry)) / 13 = 92
def highest_value := scores (Fin.mk 14 sorry) = highest_score

-- The condition includes univ meaning all elements in Finset and we are removing the first(lowest) and last(highest) elements (0 and 14).

-- The lowest score
def lowest_score (scores : Fin 15 → ℕ) : ℕ :=
  1350 - 1196 - highest_score

theorem find_lowest_score : lowest_score scores = 44 :=
by
  sorry

end find_lowest_score_l667_667183


namespace correct_number_of_propositions_is_one_l667_667025

theorem correct_number_of_propositions_is_one (p q a b x : Prop) (h1 : (¬ (p ∧ q) → ¬p ∧ ¬q)) (h2 : (¬(a > b → 2^a > 2^b - 1) ↔ (a ≤ b ∧ 2^a ≤ 2^b - 1))) (h3 : ¬ (∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 ≤ 1)) :
  1 = 1 := sorry

end correct_number_of_propositions_is_one_l667_667025


namespace problem_statement_l667_667479

variable {α : Type*} [Field α]

-- Definition of the sequence
noncomputable def a_seq (n : ℕ) : α :=
if n = 0 then 0 else if n = 1 then 1 else sorry -- Placeholder for recursion relation

-- Definition of the arithmetic sequence term
noncomputable def arith_seq (n : ℕ) : α :=
1 + (n - 1) / 2

-- Definition of the sequence b_n
noncomputable def b_seq (n : ℕ) : α :=
1 / a_seq n

-- Definition of the partial sum of b_n sequence
noncomputable def S (n : ℕ) : α :=
Finset.sum (Finset.range n) (λ i => b_seq (i + 1))

-- Proof problem statement
theorem problem_statement (n : ℕ) (h : n > 0) :
  (∀ k, is_arith_seq (a_seq k / k)) ∧
  S n = 2 * n / (n + 1) :=
by sorry

end problem_statement_l667_667479


namespace find_f2_l667_667471

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x^2) : f 2 = 1 := 
by
  sorry

end find_f2_l667_667471


namespace find_y_l667_667838

theorem find_y (y : ℝ) (h : 9^(Real.log y / Real.log 8) = 81) : y = 64 :=
by
  sorry

end find_y_l667_667838


namespace polynomial_value_at_4_l667_667256

def f (x : ℝ) : ℝ := 9 + 15 * x - 8 * x^2 - 20 * x^3 + 6 * x^4 + 3 * x^5

theorem polynomial_value_at_4 :
  f 4 = 3269 :=
by
  sorry

end polynomial_value_at_4_l667_667256


namespace surface_area_of_cube_l667_667851

noncomputable def cube_edge_length : ℝ := 20

theorem surface_area_of_cube (edge_length : ℝ) (h : edge_length = cube_edge_length) : 
    6 * edge_length ^ 2 = 2400 :=
by
  rw [h]
  sorry  -- proof placeholder

end surface_area_of_cube_l667_667851


namespace belinda_passed_out_percentage_l667_667744

variable (total_flyers ryan_flyers alyssa_flyers scott_flyers : ℕ)
variable (belinda_flyers : ℕ)

-- Define the total number of flyers
def total_flyers := 200
-- Define the number of flyers passed out by Ryan
def ryan_flyers := 42
-- Define the number of flyers passed out by Alyssa
def alyssa_flyers := 67
-- Define the number of flyers passed out by Scott
def scott_flyers := 51

-- Define the flyers that Belinda passed out
def belinda_flyers := total_flyers - (ryan_flyers + alyssa_flyers + scott_flyers)

-- Define the percentage calculation
def belinda_percentage : ℕ := (belinda_flyers * 100) / total_flyers

theorem belinda_passed_out_percentage :
  belinda_percentage total_flyers ryan_flyers alyssa_flyers scott_flyers belinda_flyers = 20 :=
by
  -- proof goes here
  sorry

end belinda_passed_out_percentage_l667_667744


namespace difference_between_payments_l667_667093

def principal : ℝ := 12000
def rate1 : ℝ := 0.05
def rate2 : ℝ := 0.03
def n1 : ℝ := 12
def t1 : ℕ := 10
def half_time : ℕ := 5

def plan1_payment1 : ℝ := principal * (1 + rate1 / n1)^(n1 * (half_time : ℝ)) / 2
def remaining_principal : ℝ := principal * (1 + rate1 / n1)^(n1 * (half_time : ℝ)) / 2
def plan1_payment2 : ℝ := remaining_principal * (1 + rate1 / n1)^(n1 * (half_time : ℝ))
def total_plan1_payment : ℝ := plan1_payment1 + plan1_payment2

def total_plan2_payment : ℝ := principal * (1 + rate2)^t1

def difference : ℝ := abs (total_plan1_payment - total_plan2_payment)

theorem difference_between_payments : round difference = 1411 := by 
  sorry

end difference_between_payments_l667_667093


namespace general_formula_a_n_T_n_greater_S_n_l667_667869

-- Define the conditions for the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ (n : ℕ), a n = a1 + (n - 1) * d

def b_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_of_first_n_terms (a S : ℕ → ℕ) :=
  ∀ (n k : ℕ), k ≤ n → S n = (n * (a 1 + a n)) / 2

-- Given S4 = 32 and T3 = 16
def S_4_equals_32 (S : ℕ → ℕ) : Prop := S 4 = 32
def T_3_equals_16 (T : ℕ → ℕ) : Prop := T 3 = 16

-- Prove that the general formula for a_n is 2n + 3
theorem general_formula_a_n (a : ℕ → ℕ) (h : arithmetic_sequence a) : 
  a = λ n, 2 * n + 3 := 
sorry

-- Prove that for n > 5, T_n > S_n
theorem T_n_greater_S_n (a b S T: ℕ → ℕ) 
  (ha : arithmetic_sequence a) 
  (hb : b_sequence a b)
  (hS : sum_of_first_n_terms a S)
  (hT : sum_of_first_n_terms b T)
  (hS4 : S_4_equals_32 S)
  (hT3 : T_3_equals_16 T) 
  (n : ℕ) : n > 5 → T n > S n :=
sorry

end general_formula_a_n_T_n_greater_S_n_l667_667869


namespace tom_buys_oranges_l667_667582

theorem tom_buys_oranges (o a : ℕ) (h₁ : o + a = 7) (h₂ : (90 * o + 60 * a) % 100 = 0) : o = 6 := 
by 
  sorry

end tom_buys_oranges_l667_667582


namespace interval_monotonicity_boundary_line_exists_and_unique_l667_667951

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2
noncomputable def g (x : ℝ) : ℝ := exp (log x)
noncomputable def F (x : ℝ) : ℝ := f x - g x

theorem interval_monotonicity : 
  (∀ x, 0 < x → x < real.sqrt real.e → deriv F x < 0) ∧ 
  (∀ x, x > real.sqrt real.e → deriv F x > 0) :=
by
  sorry

theorem boundary_line_exists_and_unique (k m : ℝ) : 
  (∀ x, f x ≥ k * x + m) ∧ (∀ x, 0 < x → g x ≤ k * x + m) ↔ 
  (k = real.sqrt real.e ∧ m = real.sqrt real.e * x - real.e / 2) :=
by
  sorry

end interval_monotonicity_boundary_line_exists_and_unique_l667_667951


namespace rewritten_expression_form_l667_667987

theorem rewritten_expression_form :
  ∃ a b d c : ℕ, c > 0 ∧ (a + b + d + c = 1959) ∧ 
  (sqrt(5) + 1/sqrt(5) + sqrt(7) + 1/sqrt(7) + sqrt(11) + 1/sqrt(11) = (a*sqrt(5) + b*sqrt(7) + d*sqrt(11)) / c) :=
sorry

end rewritten_expression_form_l667_667987


namespace tau_tau_odd_count_50_l667_667415

def tau (n : ℕ) : ℕ :=
  if n = 0 then 0 else (Nat.divisors n).length

def tau_tau_is_odd (n : ℕ) : Prop :=
  Nat.odd (tau (tau n))

theorem tau_tau_odd_count_50 :
  (Finset.range 50.succ).filter tau_tau_is_odd).card = 17 := sorry

end tau_tau_odd_count_50_l667_667415


namespace log_five_one_over_twenty_five_eq_neg_two_l667_667389

theorem log_five_one_over_twenty_five_eq_neg_two
  (h1 : (1 : ℝ) / 25 = 5^(-2))
  (h2 : ∀ (a b c : ℝ), (c > 0) → (a ≠ 0) → (log b (a^c) = c * log b a))
  (h3 : log 5 5 = 1) :
  log 5 (1 / 25) = -2 := 
  by 
    sorry

end log_five_one_over_twenty_five_eq_neg_two_l667_667389


namespace range_of_a_l667_667626

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x > 0 → (exp x - a * x^2) ≥ 0) ∧ (x ≤ 0 → (-x^2 + (a - 2) * x + 2 * a) ≥ 0) → (x ∈ [-2, ∞))) ↔ (0 ≤ a ∧ a ≤ (exp 2) / 4) :=
sorry

end range_of_a_l667_667626


namespace quadratic_roots_bounds_l667_667218

theorem quadratic_roots_bounds (a b c : ℤ) (p1 p2 : ℝ) (h_a_pos : a > 0) 
  (h_int_coeff : ∀ x : ℤ, x = a ∨ x = b ∨ x = c) 
  (h_distinct_roots : p1 ≠ p2) 
  (h_roots : a * p1^2 + b * p1 + c = 0 ∧ a * p2^2 + b * p2 + c = 0) 
  (h_roots_bounds : 0 < p1 ∧ p1 < 1 ∧ 0 < p2 ∧ p2 < 1) : 
     a ≥ 5 := 
sorry

end quadratic_roots_bounds_l667_667218


namespace intersection_A_B_l667_667013

open Set

noncomputable def A : Set ℝ := {x | log 2 x < 1}
noncomputable def B : Set ℝ := {x | x^2 + x - 2 ≤ 0}

theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_A_B_l667_667013


namespace b_geometric_sum_T_l667_667005

-- Define the sequence {a_n}
noncomputable def a : ℕ → ℚ
| 0 := 2
| (n+1) := if n % 2 = 0 then (a n + 1/2) else (1/2 * a n)

-- Define the sequence {b_n}
def b (n : ℕ) : ℚ := a (2*n + 1) - a (2*n - 1)

-- Define the sequence {c_n}
def c (n : ℕ) : ℚ := a (2*n) - 1/2

-- Theorem 1: Prove {b_n} is a geometric series with first term -1/2 and common ratio 1/2
theorem b_geometric : b 1 = -1/2 ∧ ∀ n : ℕ, b (n + 1) = 1/2 * b n := by
  sorry

-- Define the sum {T_n} of the first n terms of the sequence {nc_n}
def T (n : ℕ) : ℚ := ∑ i in Finset.range n, (i + 1) * c (i + 1)

-- Theorem 2: Prove T(n) = 2 - (n+2) * (1/2)^n
theorem sum_T : ∀ n : ℕ, T n = 2 - (n + 2) * (1/2)^n := by
  sorry

end b_geometric_sum_T_l667_667005


namespace find_m_find_t_range_l667_667026

def f (m : ℝ) (x : ℝ) := m - |x - 2|

theorem find_m : (∀ x, f m (x+2) ≥ 0 ↔ x ∈ [-2, 2]) → m = 2 :=
by
  sorry

theorem find_t_range (t : ℝ) : (∀ x, f 2 x ≥ -|x + 6| - t^2 + t) ↔ t ∈ (-∞, -2] ∪ [3, ∞) :=
by
  sorry

end find_m_find_t_range_l667_667026


namespace angle_CPE_4_degrees_l667_667757

open EuclideanGeometry

theorem angle_CPE_4_degrees
  (A B C D E P : Point)
  (BAC ACB : Angle)
  (hBAC : BAC = 24)
  (hACB : ACB = 28)
  (h1 : Parallel AB CD)
  (h2 : AD = BC)
  (h3 : ¬Parallel AD BC)
  (h4 : Parallel AE BC)
  (h5 : AB = CE)
  (h6 : ¬Parallel AB CE)
  (h7 : Intersect DE AC P) :
  angle C P E = 4 := by
sorry

end angle_CPE_4_degrees_l667_667757


namespace cot_30_eq_sqrt_3_l667_667836

theorem cot_30_eq_sqrt_3 (θ : ℝ) (hθ : θ = 30) (h1 : tan 30 = 1 / real.sqrt 3) : cot 30 = real.sqrt 3 :=
by
  have h2: cot 30 = 1 / (tan 30), by exact cot_def
  have h3: cot 30 = 1 / (1 / real.sqrt 3), by rw [h1, h2]
  have h4: cot 30 = real.sqrt 3, by exact (one_div_div (1 / real.sqrt 3))
  exact h4


end cot_30_eq_sqrt_3_l667_667836


namespace pq_proof_l667_667912

section ProofProblem

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, ∃ d, a (n + 1) = a n + d

def b_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
if n % 2 = 1 then a n - 6 else 2 * a n

def sum_first_n (seq : ℕ → ℤ) (n : ℕ) : ℤ :=
(n.sum (λ i, seq (i + 1)))

variables (a : ℕ → ℤ)
variables (S T : ℕ → ℤ)

axiom a_is_arithmetic : arithmetic_sequence a
axiom S_4_eq_32 : sum_first_n a 4 = 32
axiom T_3_eq_16 : sum_first_n (b_n a) 3 = 16

theorem pq_proof : (∀ n, a n = 2 * n + 3) ∧ (∀ n > 5, sum_first_n (b_n a) n > sum_first_n a n) :=
sorry

end ProofProblem

end pq_proof_l667_667912


namespace prism_angle_60degrees_l667_667304

open EuclideanGeometry

def right_triangular_prism_angle (A B C A₁ : Point) : Prop :=
  ∠BAC = 90 ∧
  dist A B = dist A C ∧
  dist A A₁ = dist A C →
  angle_between_skew_lines (line_through B A₁) (line_through A C₁) = 60

theorem prism_angle_60degrees (A B C A₁ C₁ : Point)
  (h1 : ∠BAC = 90)
  (h2 : dist A B = dist A C)
  (h3 : dist A A₁ = dist A C₁) :
  angle_between_skew_lines (line_through B A₁) (line_through A C₁) = 60 :=
sorry

end prism_angle_60degrees_l667_667304


namespace dots_not_visible_l667_667420

def total_dots (n_dice : ℕ) : ℕ := n_dice * 21

def sum_visible_dots (visible : List ℕ) : ℕ := visible.foldl (· + ·) 0

theorem dots_not_visible (visible : List ℕ) (h : visible = [1, 1, 2, 3, 4, 5, 5, 6]) :
  total_dots 4 - sum_visible_dots visible = 57 :=
by
  rw [total_dots, sum_visible_dots]
  simp
  sorry

end dots_not_visible_l667_667420


namespace duck_travel_days_l667_667536

theorem duck_travel_days (x : ℕ) (h1 : 40 + 2 * 40 + x = 180) : x = 60 := by
  sorry

end duck_travel_days_l667_667536


namespace crayon_count_l667_667143

theorem crayon_count :
  ∀ (lost given_away total : ℕ), 
  lost = 535 → total = 587 → total - lost = 52 :=
by
  intros lost given_away total h_lost h_total
  rw [h_lost, h_total]
  exact Nat.sub_self 535

end crayon_count_l667_667143


namespace amount_spent_on_raw_materials_l667_667153

-- Given conditions
def spending_on_machinery : ℝ := 125
def spending_as_cash (total_amount : ℝ) : ℝ := 0.10 * total_amount
def total_amount : ℝ := 250

-- Mathematically equivalent problem
theorem amount_spent_on_raw_materials :
  (X : ℝ) → X + spending_on_machinery + spending_as_cash total_amount = total_amount →
    X = 100 :=
by
  (intro X h)
  sorry

end amount_spent_on_raw_materials_l667_667153


namespace balls_in_boxes_l667_667973

theorem balls_in_boxes : ∃ n : ℕ, 
  n = 5 -> -- 5 balls
  (∃ m : ℕ, m = 3) -> -- 3 boxes
  (∃ k : ℕ, k = 18) -- 18 ways
  sorry

end balls_in_boxes_l667_667973


namespace determine_range_of_p_l667_667956

-- Define the line
def line (m : ℝ) : ℝ × ℝ → Prop := 
  λ (x y : ℝ), x = m * y - 1

-- Define the circle
def circle (m n p : ℝ) : ℝ × ℝ → Prop := 
  λ (x y : ℝ), x^2 + y^2 + m * x + n * y + p = 0

-- Define symmetric about the line y=x
def symmetric_about_line_y_eq_x (A B : ℝ × ℝ) : Prop := 
  ∀ (A₁ A₂ : ℝ), (A = (A₁, A₂)) → (B = (A₂, A₁))

-- Theorem to determine the range of values for p
theorem determine_range_of_p (m n p : ℝ) (h : symmetric_about_line_y_eq_x (A B)) : 
  m = -1 ∧ n = -1 ∧ p < -3/2 :=
by 
  sorry

end determine_range_of_p_l667_667956


namespace functional_equation_solution_l667_667839

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f(x + y) - f(x - y) = 2 * y * (3 * x^2 + y^2)) → 
  ∃ a : ℝ, ∀ x : ℝ, f(x) = x^3 + a := 
by
  sorry

end functional_equation_solution_l667_667839


namespace charge_per_action_figure_l667_667102

-- Definitions according to given conditions
def cost_of_sneakers : ℕ := 90
def saved_amount : ℕ := 15
def num_action_figures : ℕ := 10
def left_after_purchase : ℕ := 25

-- Theorem to prove the charge per action figure
theorem charge_per_action_figure : 
  (cost_of_sneakers - saved_amount + left_after_purchase) / num_action_figures = 10 :=
by 
  sorry

end charge_per_action_figure_l667_667102


namespace problem_l667_667861

variable (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℝ) (T : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def a_n_seq := ∀ n, a n = 2 * n + 1

noncomputable def Sn := ∀ n, S n = n^2 + 2 * n

noncomputable def bn := ∀ n, b n = 1 / (a n ^ 2 - 1 : ℝ)

noncomputable def Tn := ∀ n, T n = n / real.of_int (4 * n + 4)

theorem problem
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (b : ℕ → ℝ)
  (T : ℕ → ℝ)
  (h1 : is_arithmetic_sequence a 2)
  (h2 : a 3 = 7)
  (h3 : a 5 + a 7 = 26)
  : a_n_seq a ∧ Sn S ∧ bn a b ∧ Tn b T := by
  sorry

end problem_l667_667861


namespace eugene_swim_time_l667_667373

-- Define the conditions
variable (S : ℕ) -- Swim time on Sunday
variable (swim_time_mon : ℕ := 30) -- Swim time on Monday
variable (swim_time_tue : ℕ := 45) -- Swim time on Tuesday
variable (average_swim_time : ℕ := 34) -- Average swim time over three days

-- The total swim time over three days
def total_swim_time := S + swim_time_mon + swim_time_tue

-- The problem statement: Prove that given the conditions, Eugene swam for 27 minutes on Sunday.
theorem eugene_swim_time : total_swim_time S = 3 * average_swim_time → S = 27 := by
  -- Proof process will follow here
  sorry

end eugene_swim_time_l667_667373


namespace geometric_problem_l667_667044

variables {Point : Type} {Line Plane : Type} [HasMem Point Line] [HasMem Point Plane]
variables [HasMem Line Plane] [HasPerp Line Plane] [HasPerp Plane Plane]
variables (m n : Line) (α β γ : Plane) 

theorem geometric_problem
  (hαβ : α ⊥ β)
  (hα_cap_β : ∃ p : Point, p ∈ α ∧ p ∈ β ∧ p ∈ m) 
  (hnα : n ⊥ α) 
  (hnγ : ∀ p : Point, p ∈ n → p ∈ γ) :
  (m ⊥ n ∧ α ⊥ γ) :=
sorry

end geometric_problem_l667_667044


namespace speed_of_second_part_l667_667325

theorem speed_of_second_part
  (total_distance : ℝ)
  (distance_part1 : ℝ)
  (speed_part1 : ℝ)
  (average_speed : ℝ)
  (speed_part2 : ℝ) :
  total_distance = 70 →
  distance_part1 = 35 →
  speed_part1 = 48 →
  average_speed = 32 →
  speed_part2 = 24 :=
by
  sorry

end speed_of_second_part_l667_667325


namespace find_k_l667_667711

variable {α : Type} [Add α] [Mul α] [Div α] [Neg α] [Sub α] [Zero α] [One α]

def line (a b : α) (t : α) : α := a + t * (b - a)

theorem find_k (a b : α) (ha : a ≠ b) :
  ∃ k, k * a + (3 / 4:α) * b = a + (3 / 4:α) * (b - a) → k = (1 / 4:α) :=
by
  intro h
  use (1 / 4:α)
  rw [line, ←add_sub_assoc, sub_add_cancel (b * (3 / 4:α))]
  sorry

end find_k_l667_667711


namespace find_average_speed_l667_667678

noncomputable def average_speed (distance1 distance2 : ℝ) (time1 time2 : ℝ) : ℝ := 
  (distance1 + distance2) / (time1 + time2)

theorem find_average_speed :
  average_speed 1000 1000 10 4 = 142.86 := by
  sorry

end find_average_speed_l667_667678


namespace amplitude_and_phase_shift_l667_667974

def amplitude (A : ℝ) (f : ℝ → ℝ) : Prop :=
∃ (g : ℝ → ℝ), f = λ x, A * g x ∧ ∀ x, g x ∈ [-1, 1]

def phase_shift (φ : ℝ) (f : ℝ → ℝ) : Prop :=
∃ (g : ℝ → ℝ), f = λ x, g (x - φ)

noncomputable def given_function : ℝ → ℝ :=
  λ x, 5 * Real.sin (3 * x - (Real.pi / 3))

theorem amplitude_and_phase_shift :
  amplitude 5 given_function ∧ phase_shift (Real.pi / 9) given_function :=
sorry

end amplitude_and_phase_shift_l667_667974


namespace length_PH_l667_667146

theorem length_PH (G R H : ℝ × ℝ) (PQRS_area : ℝ) (△RH_area : ℝ) (s RG RH PH : ℝ)  
  (h1 : PQRS_area = 324) 
  (h2 : △RH_area = 252) 
  (h3 : s = real.sqrt 324)
  (h4 : RG = real.sqrt 504) 
  (h5 : RH = real.sqrt 504)
  (h6 : PH = real.sqrt (s^2 + RH^2)) :
  PH = 6 * real.sqrt 23 := 
by 
  sorry

end length_PH_l667_667146


namespace volume_of_prism_l667_667615

-- Define the conditions
variables {a b c : ℝ}
-- Areas of the faces
def ab := 50
def ac := 72
def bc := 45

-- Theorem stating the volume of the prism
theorem volume_of_prism : a * b * c = 180 * Real.sqrt 5 :=
by
  sorry

end volume_of_prism_l667_667615


namespace cot_30_deg_l667_667817

-- Define the known values and identities
def tan_30_deg := 1 / Real.sqrt 3

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by 
  sorry

end cot_30_deg_l667_667817


namespace inequality_proof_l667_667733

variables {R : Type} [LinearOrderedCommRing R]

theorem inequality_proof (a b c : R) (hc : c ≠ 0) :
  ac^2 > bc^2 → a > b := 
sorry

end inequality_proof_l667_667733


namespace harry_morning_routine_time_l667_667403

-- Define the conditions in Lean.
def buy_coffee_and_bagel_time : ℕ := 15 -- minutes
def read_and_eat_time : ℕ := 2 * buy_coffee_and_bagel_time -- twice the time for buying coffee and bagel is 30 minutes

-- Define the total morning routine time in Lean.
def total_morning_routine_time : ℕ := buy_coffee_and_bagel_time + read_and_eat_time

-- The final proof problem statement.
theorem harry_morning_routine_time :
  total_morning_routine_time = 45 :=
by
  unfold total_morning_routine_time
  unfold read_and_eat_time
  unfold buy_coffee_and_bagel_time
  sorry

end harry_morning_routine_time_l667_667403


namespace expansion_non_integer_terms_count_l667_667488

/-- Given a function f, defined as (x + |x|), and let a be defined as 2 times the integral
of f from -3 to 3, the number of terms in the expansion of (sqrt(x) - (1 / cbrt(x)))^a
where the exponent of x is not an integer is 15 -/
theorem expansion_non_integer_terms_count :
  let f := fun x : ℝ => x + |x| 
  let a := 2 * ∫ x in (-3 : ℝ)..(3 : ℝ), f x 
  ∃ n : ℕ, 
    a = 18 ∧
    n = 15 ∧
    (∀ r : ℕ, (0 ≤ r ∧ r ≤ 18) → 
      (∃ (m : ℤ), 9 - 5 * (r : ℚ) / 6 ≠ m) ↔ 
      (r % 6 ≠ 0)) ∧
    sorry

end expansion_non_integer_terms_count_l667_667488


namespace lowest_score_of_15_scores_is_44_l667_667185

theorem lowest_score_of_15_scores_is_44
  (mean_15 : ℕ → ℕ)
  (mean_15 = 90)
  (mean_13 : ℕ → ℕ)
  (mean_13 = 92)
  (highest_score : ℕ)
  (highest_score = 110)
  (sum15 : mean_15 * 15 = 1350)
  (sum13 : mean_13 * 13 = 1196) :
  (lowest_score : ℕ) (lowest_score = 44) :=
by
  sorry

end lowest_score_of_15_scores_is_44_l667_667185


namespace simplify_expression_l667_667603

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l667_667603


namespace simplify_fraction_l667_667299

theorem simplify_fraction (c : ℚ) : (6 + 5 * c) / 9 + 3 = (33 + 5 * c) / 9 := 
sorry

end simplify_fraction_l667_667299


namespace solve_for_x_l667_667204

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ -1 then x + 2 
  else if x < 2 then x^2 
  else 2 * x

theorem solve_for_x (x : ℝ) : f x = 3 ↔ x = Real.sqrt 3 := by
  sorry

end solve_for_x_l667_667204


namespace intersecting_to_quadrilateral_l667_667260

-- Define the geometric solids
inductive GeometricSolid
| cone : GeometricSolid
| sphere : GeometricSolid
| cylinder : GeometricSolid

-- Define a function that checks if intersecting a given solid with a plane can produce a quadrilateral
def can_intersect_to_quadrilateral (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.cone => false
  | GeometricSolid.sphere => false
  | GeometricSolid.cylinder => true

-- State the theorem
theorem intersecting_to_quadrilateral (solid : GeometricSolid) :
  can_intersect_to_quadrilateral solid ↔ solid = GeometricSolid.cylinder :=
sorry

end intersecting_to_quadrilateral_l667_667260


namespace arthur_total_walk_l667_667350

def blocks_to_miles (blocks : ℕ) : ℝ :=
  (blocks : ℝ) * (1 / 4)

theorem arthur_total_walk :
  let east_blocks := 8
  let north_blocks := 10
  let south_blocks := 5
  let total_blocks := east_blocks + north_blocks + south_blocks
  blocks_to_miles total_blocks = 5.75 :=
by
  -- definition and conditions
  let east_blocks := 8
  let north_blocks := 10
  let south_blocks := 5
  let total_blocks := east_blocks + north_blocks + south_blocks

  -- calculate the total distance in miles
  have h_total_blocks : total_blocks = 23 := rfl
  have h_miles : blocks_to_miles total_blocks = 23 * (1 / 4) := rfl
  
  -- final assertion and proof placeholder
  show blocks_to_miles total_blocks = 5.75 by
    rw [h_miles]
    show 23 * (1 / 4) = 5.75
    sorry

end arthur_total_walk_l667_667350


namespace min_combined_horses_and_ponies_l667_667677

theorem min_combined_horses_and_ponies : 
  ∀ (P : ℕ), 
  (∃ (P' : ℕ), 
    (P = P' ∧ (∃ (x : ℕ), x = 3 * P' / 10 ∧ x = 3 * P' / 16) ∧
     (∃ (y : ℕ), y = 5 * x / 8) ∧ 
      ∀ (H : ℕ), (H = 3 + P')) → 
  P + (3 + P) = 35) := 
sorry

end min_combined_horses_and_ponies_l667_667677


namespace sufficient_not_necessary_l667_667120

variable {x y : ℝ}

def p : Prop := (x > 1) ∧ (y > 1)
def q : Prop := x + y > 2

theorem sufficient_not_necessary (h1 : p) : q ∧ ¬ (∀ x y, q → p) :=
by
  split
  -- Sufficiency part
  · show q from sorry
  -- Not necessary part
  · show ¬ (∀ x y, q → p) from sorry

end sufficient_not_necessary_l667_667120


namespace parallel_lines_l667_667481

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, (a-1) * x + 2 * y + 10 = 0) → (∀ x y : ℝ, x + a * y + 3 = 0) → (a = -1 ∨ a = 2) :=
sorry

end parallel_lines_l667_667481


namespace general_formula_a_n_T_n_greater_S_n_l667_667871

-- Define the conditions for the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ (n : ℕ), a n = a1 + (n - 1) * d

def b_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_of_first_n_terms (a S : ℕ → ℕ) :=
  ∀ (n k : ℕ), k ≤ n → S n = (n * (a 1 + a n)) / 2

-- Given S4 = 32 and T3 = 16
def S_4_equals_32 (S : ℕ → ℕ) : Prop := S 4 = 32
def T_3_equals_16 (T : ℕ → ℕ) : Prop := T 3 = 16

-- Prove that the general formula for a_n is 2n + 3
theorem general_formula_a_n (a : ℕ → ℕ) (h : arithmetic_sequence a) : 
  a = λ n, 2 * n + 3 := 
sorry

-- Prove that for n > 5, T_n > S_n
theorem T_n_greater_S_n (a b S T: ℕ → ℕ) 
  (ha : arithmetic_sequence a) 
  (hb : b_sequence a b)
  (hS : sum_of_first_n_terms a S)
  (hT : sum_of_first_n_terms b T)
  (hS4 : S_4_equals_32 S)
  (hT3 : T_3_equals_16 T) 
  (n : ℕ) : n > 5 → T n > S n :=
sorry

end general_formula_a_n_T_n_greater_S_n_l667_667871


namespace percentage_of_boys_passed_l667_667077

theorem percentage_of_boys_passed (total_candidates girls boys failed_candidates_passing_percentage : ℕ)
  (girls_passed_percentage : ℝ)
  (H1 : total_candidates = 2000)
  (H2 : girls = 900)
  (H3 : boys = total_candidates - girls)
  (H4 : girls_passed_percentage = 0.32)
  (H5 : failed_candidates_passing_percentage = 69.1) : 
  (boys_passed_percentage : ℝ) => 
  boys_passed_percentage = 30 := by
  -- proof omitted
  sorry

end percentage_of_boys_passed_l667_667077


namespace sin_pi_minimum_period_l667_667208

def minimum_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ ∀ x, f(x + T) = f(x) ∧ ∀ T', (∀ x, f(x + T') = f(x)) → T' >= T

theorem sin_pi_minimum_period : minimum_positive_period (λ x, Real.sin (π * x)) 2 :=
sorry

end sin_pi_minimum_period_l667_667208


namespace solve_for_x_l667_667611

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem solve_for_x :
  let numer := sqrt (7^2 + 24^2)
  let denom := sqrt (49 + 16)
  numer / denom = 25 * (sqrt 65) / 65 :=
by
  let numer := sqrt (7^2 + 24^2)
  let denom := sqrt (49 + 16)
  have : numer = sqrt 625 := by sorry 
  have : denom = sqrt 65 := by sorry
  have : sqrt 625 = 25 := by sorry
  have : numer = 25 := by rw [this]
  have : denom = sqrt 65 := by sorry
  have : numer / denom = 25 / sqrt 65 := by rw [this]
  have : (25 : ℝ) / sqrt 65 = 25 * sqrt 65 / 65 := by
    field_simp [sqrt]; ring
  exact this

end solve_for_x_l667_667611


namespace find_lowest_score_l667_667179

variable (scores : Fin 15 → ℕ)
variable (highest_score : ℕ := 110)

-- Conditions
def arithmetic_mean_15 := (∑ i, scores i) / 15 = 90
def mean_without_highest_lowest := (∑ i in Finset.erase (Finset.erase (Finset.univ : Finset (Fin 15)) (Fin.mk 0 sorry)) (Fin.mk 14 sorry)) / 13 = 92
def highest_value := scores (Fin.mk 14 sorry) = highest_score

-- The condition includes univ meaning all elements in Finset and we are removing the first(lowest) and last(highest) elements (0 and 14).

-- The lowest score
def lowest_score (scores : Fin 15 → ℕ) : ℕ :=
  1350 - 1196 - highest_score

theorem find_lowest_score : lowest_score scores = 44 :=
by
  sorry

end find_lowest_score_l667_667179


namespace lowest_score_of_15_scores_is_44_l667_667188

theorem lowest_score_of_15_scores_is_44
  (mean_15 : ℕ → ℕ)
  (mean_15 = 90)
  (mean_13 : ℕ → ℕ)
  (mean_13 = 92)
  (highest_score : ℕ)
  (highest_score = 110)
  (sum15 : mean_15 * 15 = 1350)
  (sum13 : mean_13 * 13 = 1196) :
  (lowest_score : ℕ) (lowest_score = 44) :=
by
  sorry

end lowest_score_of_15_scores_is_44_l667_667188


namespace minimize_cost_l667_667728

noncomputable def avg_comprehensive_cost (x : ℝ) : ℝ :=
  (560 + 48 * x) + 10800 / x

theorem minimize_cost (x : ℝ) (hx : x ≥ 10) (hx_int : ∃ n : ℕ, x = n) :
  argmin avg_comprehensive_cost (≥ 10) = 15 :=
by
  sorry

end minimize_cost_l667_667728


namespace general_formula_a_n_T_n_greater_S_n_l667_667874

-- Define the conditions for the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ (n : ℕ), a n = a1 + (n - 1) * d

def b_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_of_first_n_terms (a S : ℕ → ℕ) :=
  ∀ (n k : ℕ), k ≤ n → S n = (n * (a 1 + a n)) / 2

-- Given S4 = 32 and T3 = 16
def S_4_equals_32 (S : ℕ → ℕ) : Prop := S 4 = 32
def T_3_equals_16 (T : ℕ → ℕ) : Prop := T 3 = 16

-- Prove that the general formula for a_n is 2n + 3
theorem general_formula_a_n (a : ℕ → ℕ) (h : arithmetic_sequence a) : 
  a = λ n, 2 * n + 3 := 
sorry

-- Prove that for n > 5, T_n > S_n
theorem T_n_greater_S_n (a b S T: ℕ → ℕ) 
  (ha : arithmetic_sequence a) 
  (hb : b_sequence a b)
  (hS : sum_of_first_n_terms a S)
  (hT : sum_of_first_n_terms b T)
  (hS4 : S_4_equals_32 S)
  (hT3 : T_3_equals_16 T) 
  (n : ℕ) : n > 5 → T n > S n :=
sorry

end general_formula_a_n_T_n_greater_S_n_l667_667874


namespace seventh_term_arithmetic_sequence_l667_667227

theorem seventh_term_arithmetic_sequence (a d : ℚ)
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 20)
  (h2 : a + 5 * d = 8) :
  a + 6 * d = 28 / 3 := 
sorry

end seventh_term_arithmetic_sequence_l667_667227


namespace Tn_gt_Sn_l667_667883

-- Definitions of the sequences and initial conditions
def a : ℕ → ℕ := λ n, 2 * n + 3
def b (n : ℕ) : ℕ := if n % 2 = 1 then a n - 6 else 2 * a n
def S (n : ℕ) : ℕ := (n * (2 * n + 8))/2
def T : ℕ → ℕ
| 0     := 0
| (n+1) := T n + b (n + 1)

-- Given initial conditions
axiom S_4_eq_32 : S 4 = 32
axiom T_3_eq_16 : T 3 = 16
axiom a_general : ∀ n : ℕ, a n = 2 * n + 3

-- Proof of the main theorem
theorem Tn_gt_Sn (n : ℕ) (h : n > 5) : T n > S n := by
  sorry

end Tn_gt_Sn_l667_667883


namespace Tn_gt_Sn_l667_667881

-- Definitions of the sequences and initial conditions
def a : ℕ → ℕ := λ n, 2 * n + 3
def b (n : ℕ) : ℕ := if n % 2 = 1 then a n - 6 else 2 * a n
def S (n : ℕ) : ℕ := (n * (2 * n + 8))/2
def T : ℕ → ℕ
| 0     := 0
| (n+1) := T n + b (n + 1)

-- Given initial conditions
axiom S_4_eq_32 : S 4 = 32
axiom T_3_eq_16 : T 3 = 16
axiom a_general : ∀ n : ℕ, a n = 2 * n + 3

-- Proof of the main theorem
theorem Tn_gt_Sn (n : ℕ) (h : n > 5) : T n > S n := by
  sorry

end Tn_gt_Sn_l667_667881


namespace maximize_total_profit_maximize_average_profit_l667_667705

def initial_investment := 98 * 10^6
def first_year_expenses := 12 * 10^6
def annual_income := 50 * 10^6
def annual_expense_increment := 4 * 10^6

noncomputable def total_profit (n : ℕ) : ℝ :=
  let expenses := first_year_expenses + (n-1) * annual_expense_increment
  (annual_income - expenses) * n - initial_investment

noncomputable def average_profit (n : ℕ) : ℝ :=
  total_profit n / n

theorem maximize_total_profit : ∃ n, is_maximum (total_profit n) := sorry

theorem maximize_average_profit : ∃ n, is_maximum (average_profit n) := sorry

end maximize_total_profit_maximize_average_profit_l667_667705


namespace last_remaining_number_is_64_l667_667543

theorem last_remaining_number_is_64 :
  (∃! n : ℕ, 1 ≤ n ∧ n ≤ 200 ∧ last_remaining_number (finset.range 200) = n) → n = 64 :=
sorry

end last_remaining_number_is_64_l667_667543


namespace cos4_sum_l667_667359

theorem cos4_sum (T : ℝ) :
  T = ∑ k in Finset.range 181, (Real.cos (k * Real.pi / 180)) ^ 4 →
  T = (543 : ℝ) / 8 :=
sorry

end cos4_sum_l667_667359


namespace max_value_of_ex1_ex2_l667_667948

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then exp x else -(x^3)

-- Define the function g
noncomputable def g (x a : ℝ) : ℝ := 
  f (f x) - a

-- Define the condition that g(x) = 0 has two distinct zeros
def has_two_distinct_zeros (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0

-- Define the target function h
noncomputable def h (m : ℝ) : ℝ := 
  m^3 * exp (-m)

-- Statement of the final proof
theorem max_value_of_ex1_ex2 (a : ℝ) (hpos : 0 < a) (zeros : has_two_distinct_zeros a) :
  (∃ x1 x2 : ℝ, e^x1 * e^x2 = (27 : ℝ) / (exp 3) ∧ g x1 a = 0 ∧ g x2 a = 0) :=
sorry

end max_value_of_ex1_ex2_l667_667948


namespace rectangles_in_5x5_grid_l667_667972

theorem rectangles_in_5x5_grid : 
  let n := 5 in
  (finset.card (finset.choose 2 (finset.range n))) ^ 2 = 100 :=
by
  sorry

end rectangles_in_5x5_grid_l667_667972


namespace monotonicity_and_extremes_tangent_with_slope_neg3_tangent_passing_through_origin_l667_667463

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 1

theorem monotonicity_and_extremes :
  (∀ x, x < 0 ∨ x > 2 → f' x > 0) ∧ (∀ x, 0 < x ∧ x < 2 → f' x < 0) ∧
  (f(0) = -1) ∧ (f(2) = -5) :=
sorry

theorem tangent_with_slope_neg3 :
  ∃ x, (f' x = -3) ∧ (3 * x + f x = 0) :=
sorry

theorem tangent_passing_through_origin :
  ∃ x₀, (x₀^3 - 3*x₀^2 + 1 = 0) ∧ 
  ((3*x₀ + 1 = 0 ∧ 3 * x₀ + f x₀ = 0) ∨ (15*x₀ + 4 = 0 ∧ 15 * x₀ + 4 * f x₀ = 0)) :=
sorry

end monotonicity_and_extremes_tangent_with_slope_neg3_tangent_passing_through_origin_l667_667463


namespace normal_probability_example_l667_667312

noncomputable def normal_distribution_example : Prop :=
  let μ := 100
  let σ := 10
  let X : MeasureTheory.Measure ℝ := MeasureTheory.probabilityMeasure (MeasureTheory.Probability.badMeasure.measurableSpace set.univ (MeasureTheory.Probability.badMeasure μ σ))
  MeasureTheory.Probability.Measure (0.9544) (Set.Ioc (80 : ℝ) (120 : ℝ)) 

theorem normal_probability_example :
  normal_distribution_example := sorry

end normal_probability_example_l667_667312


namespace equation_of_line_l667_667036

theorem equation_of_line
  (F : ℝ × ℝ) (C : ℝ → ℝ → Prop)
  (hF : F = (1, 0))
  (hC : ∀ x y, C x y ↔ y^2 = 4 * x)
  (A B : ℝ × ℝ)
  (L : ℝ → ℝ)
  (hL : ∀ x, L x = (A.snd - B.snd) / (A.fst - B.fst) * (x - 1))
  (hA : C A.fst A.snd)
  (hB : C B.fst B.snd)
  (hD : dist A F = 3 * dist B F) :
  L = λ x, - sqrt 3 * (x - 1) ∨ L = λ x, sqrt 3 * (x - 1) :=
by {
  sorry -- proof not included
}

end equation_of_line_l667_667036


namespace perimeter_of_one_rectangle_l667_667640

theorem perimeter_of_one_rectangle (s : ℝ) (rectangle_perimeter rectangle_length rectangle_width : ℝ) (h1 : 4 * s = 240) (h2 : rectangle_width = (1/2) * s) (h3 : rectangle_length = s) (h4 : rectangle_perimeter = 2 * (rectangle_length + rectangle_width)) :
  rectangle_perimeter = 180 := 
sorry

end perimeter_of_one_rectangle_l667_667640


namespace unique_injective_multiplicative_function_l667_667760

open Nat

-- Define the problem conditions
def injective (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, f x = f y → x = y

def condition_a (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, f (m * n) = f m * f n

def condition_b (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, f (m^2 + n^2) ∣ f (m^2) + f (n^2)

-- Pose the main theorem
theorem unique_injective_multiplicative_function (f : ℕ+ → ℕ+) :
  injective f → condition_a f → condition_b f → ∀ n : ℕ+, f n = n :=
by
  intro hinjective ha hb
  sorry

end unique_injective_multiplicative_function_l667_667760


namespace simplify_expression_l667_667594

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l667_667594


namespace log_base_5_of_1_div_25_eq_neg_2_l667_667380

theorem log_base_5_of_1_div_25_eq_neg_2 :
  log 5 (1/25) = -2 :=
by sorry

end log_base_5_of_1_div_25_eq_neg_2_l667_667380


namespace new_pressure_correct_l667_667741
-- Importing the complete library for broader utility

-- Definitions based on given conditions
def p1 : ℝ := 7          -- initial pressure
def v1 : ℝ := 3.5        -- initial volume
def v2 : ℝ := 10.5       -- new volume
def k : ℝ := p1 * v1     -- constant k

-- Main statement to prove
theorem new_pressure_correct :
  ∃ (p2 : ℝ), p2 * v2 = k ∧ p2 = (49 / 21) := 
by
  -- Use k = 24.5
  have h : k = 24.5 := by
    simp [k, p1, v1]; norm_num,
  
  -- Find p2
  have hp2 : ∃ p2, p2 * v2 = k := by
    use k / v2,
    field_simp [v2],
    exact h,

  -- Prove the correct new pressure
  cases hp2 with p2 hp2_eq,
  use p2,
  constructor,
  exact hp2_eq,
  have : p2 = 49 / 21 := by 
    rw [hp2_eq],
    simp [v2],
    norm_num,
  exact this

end new_pressure_correct_l667_667741


namespace cyclic_quadrilateral_and_BF_midpoint_of_CG_l667_667003

-- Define given conditions
variables (A B C H D A_1 E F G : Type)

-- Assume and describe the conditions for the points
axiom triangle_with_orthocenter : ∃ (ABC : Type) (H : Type), true
axiom parallelogram_AHCD : ∃ (D : Type), true
axiom A1_is_midpoint_of_BC : ∃ (A_1 : Type), true
axiom perpendicular_p_through_A1 : ∃ (p : Type), true
axiom E_is_intersection_of_p_and_AB : ∃ (E : Type), true
axiom F_is_midpoint_of_A1E : ∃ (F : Type), true
axiom G_is_intersection_of_parallel_BD_through_A_with_p :  ∃ (G : Type), true

-- The theorem to prove
theorem cyclic_quadrilateral_and_BF_midpoint_of_CG :
  (cyclic_quadrilateral A F A_1 C) ↔ (BF_passes_through_midpoint_CG A B F C G A_1 E D H) :=
sorry

end cyclic_quadrilateral_and_BF_midpoint_of_CG_l667_667003


namespace angle_between_tangents_constant_l667_667585

-- Definitions based on conditions
variables {A B C P Ta Tb : Type*}

-- Define triangle ABC
def is_triangle (A B C : Type*) : Prop := 
  -- Assume it forms a triangle, as actual definition may require coordinate usage
  
-- Point P is on AB means there exists t such that P is a linear combination of A and B
def on_segment (P A B : Type*) : Prop :=
  -- Assume this condition abstracts that P is on AB or its extension
  
-- Define circumcircles of triangles
def circumcircle (A B C : Type*) : Type* := sorry

def tangent_at_point (P : Type*) (ω : Type*) : Type* := sorry

-- Convert the problem into a theorem
theorem angle_between_tangents_constant 
  (ABC_triangle : is_triangle A B C) 
  (P_on_segment : on_segment P A B)
  (ω1 : circumcircle P A C)
  (ω2 : circumcircle P B C)
  (Ta : tangent_at_point P ω1)
  (Tb : tangent_at_point P ω2) :
  -- The angle between the tangents at P is γ, the angle ACB.
  angle Ta P Tb = angle A C B := 
begin
  -- The proof direction
  sorry
end

end angle_between_tangents_constant_l667_667585


namespace cot_30_eq_sqrt_3_l667_667833

theorem cot_30_eq_sqrt_3 (θ : ℝ) (hθ : θ = 30) (h1 : tan 30 = 1 / real.sqrt 3) : cot 30 = real.sqrt 3 :=
by
  have h2: cot 30 = 1 / (tan 30), by exact cot_def
  have h3: cot 30 = 1 / (1 / real.sqrt 3), by rw [h1, h2]
  have h4: cot 30 = real.sqrt 3, by exact (one_div_div (1 / real.sqrt 3))
  exact h4


end cot_30_eq_sqrt_3_l667_667833


namespace artist_painted_total_pictures_l667_667735

theorem artist_painted_total_pictures :
  ∃ x : ℕ, (x / 8 = 9) ∧ (x + 72 = 153) :=
begin
  use 81,
  split,
  {
    show 81 / 8 = 9,
    sorry
  },
  {
    show 81 + 72 = 153,
    sorry
  }
end

end artist_painted_total_pictures_l667_667735


namespace scientific_notation_of_0_0000023_l667_667772

theorem scientific_notation_of_0_0000023 : 
  0.0000023 = 2.3 * 10 ^ (-6) :=
by
  sorry

end scientific_notation_of_0_0000023_l667_667772


namespace min_value_f_l667_667845

noncomputable def f (x : ℝ) : ℝ :=
  (Real.log x / Real.log 2) * (Real.log (2 * x) / (Real.log (sqrt 2)))

theorem min_value_f : ∃ x > 0, f x = -1/4 := 
by
  use (sqrt 2 / 2)
  split
  · apply div_pos
    · exact Real.sqrt_pos.mpr (by norm_num)
    · norm_num
  sorry

end min_value_f_l667_667845


namespace cot_30_eq_sqrt_3_l667_667801

theorem cot_30_eq_sqrt_3 (h : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3) : 
  Real.cot (Real.pi / 6) = Real.sqrt 3 :=
by
  sorry

end cot_30_eq_sqrt_3_l667_667801


namespace eccentricity_of_hyperbola_l667_667954

theorem eccentricity_of_hyperbola 
  (a b : ℝ) 
  (h_eq : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) 
  (h_asymptote : ∀ x : ℝ, x ≠ 0 → ∃ y : ℝ, y = sqrt 2 * x) : 
  b = sqrt 2 * a → sqrt (1 + (b / a)^2) = sqrt 3 := 
by
  intros h_rel
  rw h_rel
  simp
  sorry

end eccentricity_of_hyperbola_l667_667954


namespace digital_earth_sustainable_development_l667_667731

theorem digital_earth_sustainable_development :
  (after_realization_digital_earth : Prop) → (scientists_can : Prop) :=
sorry

end digital_earth_sustainable_development_l667_667731


namespace solve_for_x_l667_667978

theorem solve_for_x (x : ℝ) (h : sqrt (3 / x + 3) = 5 / 3) : x = -27 / 2 :=
sorry

end solve_for_x_l667_667978


namespace intersection_M_N_l667_667961

open Set

variable (x : ℝ)
def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
def N : Set ℝ := {-2, 0, 2}

theorem intersection_M_N : M ∩ N = {0, 2} := sorry

end intersection_M_N_l667_667961


namespace polynomial_sum_l667_667564

noncomputable def p (x : ℝ) : ℝ := -2 * x^2 + 2 * x - 5
noncomputable def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
noncomputable def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = 12 * x - 12 := by
  sorry

end polynomial_sum_l667_667564


namespace decimal_15_to_binary_l667_667363

theorem decimal_15_to_binary : (15 : ℕ) = (4*1 + 2*1 + 1*1)*2^3 + (4*1 + 2*1 + 1*1)*2^2 + (4*1 + 2*1 + 1*1)*2 + 1 := by
  sorry

end decimal_15_to_binary_l667_667363


namespace gcd_polynomials_l667_667934

theorem gcd_polynomials (b : ℕ) (hb : 2160 ∣ b) : 
  Nat.gcd (b ^ 2 + 9 * b + 30) (b + 6) = 12 := 
  sorry

end gcd_polynomials_l667_667934


namespace a_even_is_perfect_square_l667_667122

def isNDigitSum (n : ℕ) (N : ℕ) : Prop :=
  ∑ i in N.digits 10, i = n ∧ (∀ i ∈ N.digits 10, i ∈ {1, 3, 4})

def a (n : ℕ) : ℕ :=
  { N : ℕ | isNDigitSum n N }.toFinset.card

theorem a_even_is_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k * k := by
  sorry

end a_even_is_perfect_square_l667_667122


namespace log_base_5_of_reciprocal_of_25_l667_667383

theorem log_base_5_of_reciprocal_of_25 : log 5 (1 / 25) = -2 :=
by
-- Sorry this proof is omitted
sorry

end log_base_5_of_reciprocal_of_25_l667_667383


namespace product_of_reciprocals_plus_one_geq_nine_l667_667427

theorem product_of_reciprocals_plus_one_geq_nine
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hab : a + b = 1) :
  (1 / a + 1) * (1 / b + 1) ≥ 9 :=
sorry

end product_of_reciprocals_plus_one_geq_nine_l667_667427


namespace quadrilateral_ABCD_l667_667589

noncomputable def ratio_AB_AD (AB AD : ℝ) [AD ≠ 0] := AB / AD

theorem quadrilateral_ABCD 
  (A B C D E : Type) -- points in plane
  [quadrilateral ABCD : Type]  -- ABCD is a quadrilateral
  (right_angle_A : ∠A = 90) -- right angle at A
  (right_angle_C : ∠C = 90) -- right angle at C
  (similar_ABC_ACD : similar (triangle A B C) (triangle A C D)) -- ΔABC ∼ ΔACD
  (AB_gt_AD : AB > AD) -- AB > AD
  (point_E_interior : E ∈ interior ABCD) -- E is in the interior of ABCD
  (similar_ABE_CED : similar (triangle A B E) (triangle C E D)) -- ΔABE ∼ ΔCED
  (area_BDE_twice_area_CED : area (triangle B D E) = 20 * area (triangle C E D)) -- area condition
  : ratio_AB_AD AB AD = 1 / real.sqrt 399 :=
sorry

end quadrilateral_ABCD_l667_667589


namespace part1_part2_l667_667864

-- Define Set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3/2}

-- Define Set B, parameterized by m
def B (m : ℝ) : Set ℝ := {x | 1-m < x ∧ x ≤ 3*m + 1}

-- Proof Problem (1): When m = 1, A ∩ B = {x | 0 < x ∧ x ≤ 3/2}
theorem part1 (x : ℝ) : (x ∈ A ∩ B 1) ↔ (0 < x ∧ x ≤ 3/2) := by
  sorry

-- Proof Problem (2): If ∀ x, x ∈ A → x ∈ B m, then m ∈ (-∞, 1/6]
theorem part2 (m : ℝ) : (∀ x, x ∈ A → x ∈ B m) → m ≤ 1/6 := by
  sorry

end part1_part2_l667_667864


namespace students_not_in_sports_l667_667578

/-- Of the 75 students in a sports club, 46 take swimming, 34 take basketball,
and 22 students take both swimming and basketball. How many sports club 
students do not participate in either swimming or basketball? -/
theorem students_not_in_sports (total_students swimming_students basketball_students both_students : ℕ)
  (h_total : total_students = 75)
  (h_swimming : swimming_students = 46)
  (h_basketball : basketball_students = 34)
  (h_both : both_students = 22) :
  total_students - (swimming_students - both_students + basketball_students - both_students + both_students) = 17 :=
by {
  rw [h_total, h_swimming, h_basketball, h_both],
  norm_num,
  sorry
}

end students_not_in_sports_l667_667578


namespace cot_30_eq_sqrt3_l667_667785

theorem cot_30_eq_sqrt3 : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by
  have h_cos : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  have h_sin : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  show Real.cot (30 * Real.pi / 180) = Real.sqrt 3
  calc
    Real.cot (30 * Real.pi / 180) = (Real.cos (30 * Real.pi / 180)) / (Real.sin (30 * Real.pi / 180)) := sorry
    ... = (Real.sqrt 3 / 2) / (1 / 2) := sorry
    ... = Real.sqrt 3 := sorry

end cot_30_eq_sqrt3_l667_667785


namespace cot_30_eq_sqrt_3_l667_667806

theorem cot_30_eq_sqrt_3 (h : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3) : 
  Real.cot (Real.pi / 6) = Real.sqrt 3 :=
by
  sorry

end cot_30_eq_sqrt_3_l667_667806


namespace three_divides_n_of_invertible_diff_l667_667547

theorem three_divides_n_of_invertible_diff
  (n : ℕ)
  (A B : Matrix (Fin n) (Fin n) ℝ)
  (h1 : A * A + B * B = A * B)
  (h2 : Invertible (B * A - A * B)) :
  3 ∣ n :=
sorry

end three_divides_n_of_invertible_diff_l667_667547


namespace remaining_insects_total_l667_667591

theorem remaining_insects_total :
  let bees := 15 - 6 in
  let beetles := 7 - 2 in
  let ants := 12 - 4 in
  let termites := 10 - 3 in
  let praying_mantises := 2 in
  let ladybugs := 10 - 2 in
  let butterflies := 11 - 3 in
  let dragonflies := 8 - 1 in
  bees + beetles + ants + termites + praying_mantises + ladybugs + butterflies + dragonflies = 54 :=
by
  sorry

end remaining_insects_total_l667_667591


namespace general_formula_T_greater_S_l667_667898

section arithmetic_sequence

variable {a : ℕ → ℤ} -- Define the arithmetic sequence {a_n}
variable {b : ℕ → ℤ} -- Define the sequence {b_n}
variable {S : ℕ → ℤ} -- Define the sum sequence {S_n}
variable {T : ℕ → ℤ} -- Define the sum sequence {T_n}
variable (a1 : ℤ) (d : ℤ) -- Variables for the first term and common difference 

-- Conditions
def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n, a n = a1 + n * d

def sequence_b (b a : ℕ → ℤ) : Prop :=
  ∀ n, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_sequence_S (S a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (2 * a1 + (n - 1) * d)) / 2

def sum_sequence_T (T b : ℕ → ℤ) : Prop :=
  ∀ n, T n = ∑ i in Finset.range n, b i

def conditions (a b S T : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  arithmetic_sequence a a1 d ∧ sequence_b b a ∧ sum_sequence_S S a ∧ sum_sequence_T T b ∧ S 4 = 32 ∧ T 3 = 16

-- Problem 1: Prove the general formula for {a_n}
theorem general_formula (a : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, a n = 2 * n + 3 :=
sorry

-- Problem 2: Prove that for n > 5, T_n > S_n
theorem T_greater_S (a b S T : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, n > 5 → T n > S n :=
sorry

end arithmetic_sequence

end general_formula_T_greater_S_l667_667898


namespace find_x_l667_667680

theorem find_x : 
  ∀ (x : ℝ), (0.75 / x = 5 / 11) → x = 1.65 :=
by 
  assume (x : ℝ) (h : 0.75 / x = 5 / 11),
  sorry

end find_x_l667_667680


namespace cot_30_eq_sqrt_3_l667_667819

theorem cot_30_eq_sqrt_3 :
  let θ := 30.0
  let cos θ := (real.sqrt 3) / 2
  let sin θ := 1 / 2
  let cot θ := cos θ / sin θ
  cot θ = real.sqrt 3 := by
sorry

end cot_30_eq_sqrt_3_l667_667819


namespace quadratic_diff_larger_smaller_proof_l667_667764

noncomputable def quadratic_diff_larger_smaller : ℝ :=
  let a := 9 + 6 * Real.sqrt 2
  let b := 3 + 2 * Real.sqrt 2
  let c := -3
  let delta := b^2 - 4 * a * c
  let root1 := (-b + Real.sqrt delta) / (2 * a)
  let root2 := (-b - Real.sqrt delta) / (2 * a)
  Real.abs (root1 - root2)

theorem quadratic_diff_larger_smaller_proof :
  quadratic_diff_larger_smaller = (Real.sqrt (97 - 72 * Real.sqrt 2)) / 3 :=
sorry

end quadratic_diff_larger_smaller_proof_l667_667764


namespace minimum_value_x2_y2_l667_667011

variable {x y : ℝ}

theorem minimum_value_x2_y2 (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x * y = 1) : x^2 + y^2 = 2 :=
sorry

end minimum_value_x2_y2_l667_667011


namespace Jack_hands_in_l667_667539

def num_hundred_bills := 2
def num_fifty_bills := 1
def num_twenty_bills := 5
def num_ten_bills := 3
def num_five_bills := 7
def num_one_bills := 27
def to_leave_in_till := 300

def total_money_in_notes : Nat :=
  (num_hundred_bills * 100) +
  (num_fifty_bills * 50) +
  (num_twenty_bills * 20) +
  (num_ten_bills * 10) +
  (num_five_bills * 5) +
  (num_one_bills * 1)

def money_to_hand_in := total_money_in_notes - to_leave_in_till

theorem Jack_hands_in : money_to_hand_in = 142 := by
  sorry

end Jack_hands_in_l667_667539


namespace max_vertices_of_divided_triangle_l667_667344

theorem max_vertices_of_divided_triangle (n : ℕ) (h : n ≥ 1) : 
  (∀ t : ℕ, t = 1000 → exists T : ℕ, T = (n + 2)) :=
by sorry

end max_vertices_of_divided_triangle_l667_667344


namespace equation_of_line_BM_l667_667965

noncomputable def midpoint (A C : Point) : Point :=
  ⟨(A.1 + C.1) / 2, (A.2 + C.2) / 2⟩

def line_eq (B M : Point) : AffineLine :=
  AffineLine.mk' B M

def Point := Prod ℝ ℝ

def line_eq_check (p : ℝ × ℝ) : Prop :=
  3 * p.1 - 2 * p.2 + 14 = 0

theorem equation_of_line_BM : 
  let A : Point := (1, 2)
  let B : Point := (4, 1)
  let C : Point := (3, 6)
  let M : Point := midpoint A C
  line_eq_check B ∧ line_eq_check M :=
by
  sorry

end equation_of_line_BM_l667_667965


namespace general_formula_sums_inequality_for_large_n_l667_667902

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def modified_seq (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = (finset.range n).sum a

variables {a b S T : ℕ → ℕ}

-- Given conditions
axiom h_arith_seq : arithmetic_seq a
axiom h_modified_seq : modified_seq a b
axiom h_sum_a : sum_seq a S
axiom h_sum_b : sum_seq b T
axiom h_S4 : S 4 = 32
axiom h_T3 : T 3 = 16

-- Proof Statements
theorem general_formula :
  ∀ n : ℕ, a n = 2 * n + 3 := sorry

theorem sums_inequality_for_large_n :
  ∀ n : ℕ, n > 5 → T n > S n := sorry

end general_formula_sums_inequality_for_large_n_l667_667902


namespace sum_of_distinct_prime_factors_420_l667_667286

theorem sum_of_distinct_prime_factors_420 : (2 + 3 + 5 + 7 = 17) :=
by
  -- given condition on the prime factorization of 420
  have h : ∀ {n : ℕ}, n = 420 → ∀ (p : ℕ), p ∣ n → p.prime → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7,
  by sorry

  -- proof of summation of these distinct primes
  sorry

end sum_of_distinct_prime_factors_420_l667_667286


namespace find_lowest_score_l667_667180

variable (scores : Fin 15 → ℕ)
variable (highest_score : ℕ := 110)

-- Conditions
def arithmetic_mean_15 := (∑ i, scores i) / 15 = 90
def mean_without_highest_lowest := (∑ i in Finset.erase (Finset.erase (Finset.univ : Finset (Fin 15)) (Fin.mk 0 sorry)) (Fin.mk 14 sorry)) / 13 = 92
def highest_value := scores (Fin.mk 14 sorry) = highest_score

-- The condition includes univ meaning all elements in Finset and we are removing the first(lowest) and last(highest) elements (0 and 14).

-- The lowest score
def lowest_score (scores : Fin 15 → ℕ) : ℕ :=
  1350 - 1196 - highest_score

theorem find_lowest_score : lowest_score scores = 44 :=
by
  sorry

end find_lowest_score_l667_667180


namespace ratio_of_shaded_area_l667_667671

theorem ratio_of_shaded_area 
  (AC : ℝ) (CB : ℝ) 
  (AB : ℝ := AC + CB) 
  (radius_AC : ℝ := AC / 2) 
  (radius_CB : ℝ := CB / 2)
  (radius_AB : ℝ := AB / 2) 
  (shaded_area : ℝ := (radius_AB ^ 2 * Real.pi / 2) - (radius_AC ^ 2 * Real.pi / 2) - (radius_CB ^ 2 * Real.pi / 2))
  (CD : ℝ := Real.sqrt (AC^2 - radius_CB^2))
  (circle_area : ℝ := CD^2 * Real.pi) :
  (shaded_area / circle_area = 21 / 187) := 
by 
  sorry

end ratio_of_shaded_area_l667_667671


namespace f_12_16_plus_f_16_12_l667_667566

noncomputable def f : ℕ × ℕ → ℕ :=
sorry

axiom ax1 : ∀ (x : ℕ), f (x, x) = x
axiom ax2 : ∀ (x y : ℕ), f (x, y) = f (y, x)
axiom ax3 : ∀ (x y : ℕ), (x + y) * f (x, y) = y * f (x, x + y)

theorem f_12_16_plus_f_16_12 : f (12, 16) + f (16, 12) = 96 :=
by sorry

end f_12_16_plus_f_16_12_l667_667566


namespace common_ratio_nonzero_sequence_arithmetic_ratio_l667_667528

/-- Define what it means for a sequence to be an arithmetic ratio sequence -/
def is_arithmetic_ratio_sequence (a : ℕ → ℝ) (k : ℝ) :=
  ∀ n : ℕ, (a (n + 2) - a (n + 1)) / (a (n + 1) - a n) = k

/-- Prove that the common ratio of an arithmetic ratio sequence cannot be zero -/
theorem common_ratio_nonzero {a : ℕ → ℝ} (h : ∃ k : ℝ, is_arithmetic_ratio_sequence a k) : ∀ k, k ≠ 0 :=
sorry

/-- Prove that the sequence defined by a_n = -3^n + 2 is an arithmetic ratio sequence with ratio 3  -/
theorem sequence_arithmetic_ratio : is_arithmetic_ratio_sequence (λ n, -3^n + 2) 3 :=
sorry

#check common_ratio_nonzero
#check sequence_arithmetic_ratio

end common_ratio_nonzero_sequence_arithmetic_ratio_l667_667528


namespace angle_AEB_is_50_l667_667062

-- Define the points and triangle
variables (A B C E : Type) 
variables [Point (A B C E)]
variables [Triangle (A B C)]

-- Define the lengths and angles
variable (AE EC : ℝ)
variable (angle_BEC angle_AEB : ℝ)
variables (is_ae_eq_ec : AE = EC)
variables (interior_angle_BEC : angle_BEC = 50)

-- Define condition that BE bisects angle ABC
variable (BE_bisects_angle_ABC : Bisects B E A C)

-- The statement we want to prove
theorem angle_AEB_is_50 :
  angle_AEB = 50 :=
by sorry

end angle_AEB_is_50_l667_667062


namespace total_bees_is_25_l667_667232

def initial_bees : ℕ := 16
def additional_bees : ℕ := 9

theorem total_bees_is_25 : initial_bees + additional_bees = 25 := by
  sorry

end total_bees_is_25_l667_667232


namespace calculate_AM_l667_667522

-- Given definitions
variable (ABC : Triangle)
variable (O : Point)
variable (M : Point)
variable (DM ME AM : ℝ)
variable (midpoint_M_AC : midpoint M ABC.A ABC.C)
variable (center_O_circle : center O ABC.circumcircle)
variable (DM_eq_9 : DM = 9)
variable (ME_eq_4 : ME = 4)

-- Goal: To prove AM = 6
theorem calculate_AM 
  (h_DM : DM = 9)
  (h_ME : ME = 4)
  (h_midpoint : midpoint M ABC.A ABC.C)
  (h_center : center O ABC.circumcircle)
  : AM = 6 :=
by
  -- Proof here
  sorry

end calculate_AM_l667_667522


namespace B_in_interval_l667_667755

noncomputable def g : ℕ → ℝ
| 4     := real.log 4
| (n+1) := real.log (n + 1 + g n)

theorem B_in_interval : g 2020 > real.log 2023 ∧ g 2020 < real.log 2024 :=
sorry

end B_in_interval_l667_667755


namespace impossible_to_use_up_components_l667_667997

theorem impossible_to_use_up_components (p q r : ℕ) : 
  let A_used := 2*p + 2*r + 2 in
  let B_used := 2*p + q + 1 in
  let C_used := q + r in
  ∀ x y z : ℕ,
    2*x + 2*z = A_used ∧
    2*x + y = B_used ∧
    y + z = C_used →
    false :=
sorry

end impossible_to_use_up_components_l667_667997


namespace sum_max_min_squares_l667_667857

/-- Given that x₁, x₂, ..., x₄₀ are positive integers and their sum equals 58,
    prove that the sum of the maximum and minimum values of x₁² + x₂² + ... + x₄₀² equals 494. -/
theorem sum_max_min_squares (x : Fin 40 → ℕ) (h_sum : (∑ i, x i) = 58) (h_pos : ∀ i, x i > 0) :
    let A := max (∑ i, (x i)^2)
    let B := min (∑ i, (x i)^2)
    A + B = 494 :=
by
  sorry

end sum_max_min_squares_l667_667857


namespace area_of_CMKD_l667_667584

-- Definitions from the problem conditions
variables {A B C D M K : Type*}
variables (BC_area : ℝ) (parallelogram_area : ℝ := 1)

-- Condition: Point M divides BC in the ratio 2:1
def M_divides_BC (BM MC : ℝ) : Prop := BM / MC = 2

-- Condition: Line AM intersects diagonal BD at point K
def AM_intersects_BD_at_K (AM BD KM : ℝ) : Prop := AM > 0 ∧ BD > 0 ∧ KM > 0

-- Main theorem statement
theorem area_of_CMKD
        (h_div : M_divides_BC 2 1)
        (h_intersect : AM_intersects_BD_at_K 3 1 1)
        (parallelogram_area : ℝ := 1) :
    let triangle_BCD_area := parallelogram_area / 2 in
    let triangle_BMK_area := (2 / 3) * (2 / 5) * (triangle_BCD_area / 2) in
    let quadrilateral_CMKD_area := triangle_BCD_area - triangle_BMK_area in
    quadrilateral_CMKD_area = 11 / 30 := 
sorry

end area_of_CMKD_l667_667584


namespace area_between_chords_is_correct_l667_667509

noncomputable def circle_radius : ℝ := 10
noncomputable def chord_distance_apart : ℝ := 12
noncomputable def area_between_chords : ℝ := 44.73

theorem area_between_chords_is_correct 
    (r : ℝ) (d : ℝ) (A : ℝ) 
    (hr : r = circle_radius) 
    (hd : d = chord_distance_apart) 
    (hA : A = area_between_chords) : 
    ∃ area : ℝ, area = A := by 
  sorry

end area_between_chords_is_correct_l667_667509


namespace percentage_increase_l667_667337

theorem percentage_increase (W E : ℝ) (P : ℝ) :
  W = 200 →
  E = 204 →
  (∃ P, E = W * (1 + P / 100) * 0.85) →
  P = 20 :=
by
  intros hW hE hP
  -- Proof could be added here.
  sorry

end percentage_increase_l667_667337


namespace binary_operation_result_l667_667754

theorem binary_operation_result :
  let a := 0b1101
  let b := 0b111
  let c := 0b1010
  let d := 0b1001
  a + b - c + d = 0b10011 :=
by {
  sorry
}

end binary_operation_result_l667_667754


namespace washingMachineCapacity_l667_667971

-- Definitions based on the problem's conditions
def numberOfShirts : ℕ := 2
def numberOfSweaters : ℕ := 33
def numberOfLoads : ℕ := 5

-- Statement we need to prove
theorem washingMachineCapacity : 
  (numberOfShirts + numberOfSweaters) / numberOfLoads = 7 := sorry

end washingMachineCapacity_l667_667971


namespace least_three_digit_divisible_3_4_7_is_168_l667_667665

-- Define the function that checks the conditions
def is_least_three_digit_divisible_by_3_4_7 (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000 ∧ x % 3 = 0 ∧ x % 4 = 0 ∧ x % 7 = 0

-- Define the target value
def least_three_digit_number_divisible_by_3_4_7 : ℕ := 168

-- The theorem we want to prove
theorem least_three_digit_divisible_3_4_7_is_168 :
  ∃ x : ℕ, is_least_three_digit_divisible_by_3_4_7 x ∧ x = least_three_digit_number_divisible_by_3_4_7 := by
  sorry

end least_three_digit_divisible_3_4_7_is_168_l667_667665


namespace calc_expression_l667_667354

theorem calc_expression :
  sqrt (6 + 1/4) - (Real.pi - 1)^0 - (3 + 3/8)^(1/3) + (1/64)^(-2/3) = 16 :=
by sorry

end calc_expression_l667_667354


namespace gini_inconsistency_l667_667688

-- Definitions of given conditions
def X : ℝ := 0.5
def G0 : ℝ := 0.1
def Y : ℝ := 0.4
def Y' : ℝ := 0.2

-- After the reform, defining the shares
def Z : ℝ -- income share for the middle class
def share_rich : ℝ := 1 - (Y' + Z)

-- Correct value to be checked
def G : ℝ := 0.1

-- Proof object to express that maintaining the same Gini coefficient is impossible
theorem gini_inconsistency : ∀ (Z : ℝ), G ≠ (1 / 2 - (1 / 6 * 0.2 + 1 / 6 * (0.2 + Z) + 1 / 6)) / (1 / 2) := 
by
  sorry

end gini_inconsistency_l667_667688


namespace weight_difference_is_480_l667_667152

def Robbie_weight : ℕ := 100
def Patty_orig_weight : ℕ := 4.5 * Robbie_weight
def Jim_orig_weight : ℕ := 3 * Robbie_weight
def Mary_orig_weight : ℕ := 2 * Robbie_weight

def Patty_curr_weight : ℕ := Patty_orig_weight - 235
def Jim_curr_weight : ℕ := Jim_orig_weight - 180
def Mary_curr_weight : ℕ := Mary_orig_weight + 45

def Combined_weight_diff : ℕ := (Patty_curr_weight + Jim_curr_weight + Mary_curr_weight) - Robbie_weight

theorem weight_difference_is_480 : Combined_weight_diff = 480 := by
  sorry

end weight_difference_is_480_l667_667152


namespace vector_parallel_and_sum_l667_667958

theorem vector_parallel_and_sum
  (m : ℝ)
  (a b : ℝ × ℝ)
  (h₀ : a = (1, 2))
  (h₁ : b = (-2, m))
  (h_parallel : ∃ k : ℝ, b = k • a) :
  2 • a + 3 • b = (-4, -8) :=
by
  sorry

end vector_parallel_and_sum_l667_667958


namespace total_games_eq_64_l667_667573

def games_attended : ℕ := 32
def games_missed : ℕ := 32
def total_games : ℕ := games_attended + games_missed

theorem total_games_eq_64 : total_games = 64 := by
  sorry

end total_games_eq_64_l667_667573


namespace arithmetic_sequence_sum_l667_667641

theorem arithmetic_sequence_sum (n : ℕ) (a1 : ℤ) (d : ℤ) 
  (h1 : ∑ k in finset.range n, (a1 + k * d) = 2000)
  (h2 : d = 2)
  (h3 : a1 ∈ ℤ)
  (h4 : n > 1) :
  ∑ k in finset.filter (λ k, k > 1) (finset.divisors 2000), k = 4835 :=
sorry

end arithmetic_sequence_sum_l667_667641


namespace final_sum_l667_667121

def Q (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 4

noncomputable def probability_condition_holds : ℝ :=
  by sorry

theorem final_sum :
  let m := 1
  let n := 1
  let o := 1
  let p := 0
  let q := 8
  (m + n + o + p + q) = 11 :=
  by
    sorry

end final_sum_l667_667121


namespace pieces_length_l667_667049

theorem pieces_length :
  let total_length_meters := 29.75
  let number_of_pieces := 35
  let length_per_piece_meters := total_length_meters / number_of_pieces
  let length_per_piece_centimeters := length_per_piece_meters * 100
  length_per_piece_centimeters = 85 :=
by
  sorry

end pieces_length_l667_667049


namespace determine_num_students_first_class_l667_667199
noncomputable def num_students_first_class : ℕ :=
let x := 35,
    avg_marks_first_class := 40,
    avg_marks_second_class := 60,
    num_students_second_class := 45,
    avg_marks_all_students := 51.25 in
  if avg_marks_all_students * (x + num_students_second_class) = (avg_marks_first_class * x + avg_marks_second_class * num_students_second_class) then x else 0

theorem determine_num_students_first_class :
  let avg_marks_first_class := 40,
      avg_marks_second_class := 60,
      num_students_second_class := 45,
      avg_marks_all_students := 51.25 in
  ∃ x : ℕ, avg_marks_all_students * (x + num_students_second_class) = (avg_marks_first_class * x + avg_marks_second_class * num_students_second_class) ∧ x = 35 :=
begin
  let x := 35,
  let avg_marks_first_class := 40,
  let avg_marks_second_class := 60,
  let num_students_second_class := 45,
  let avg_marks_all_students := 51.25,
  use x,
  simp [avg_marks_first_class, avg_marks_second_class, num_students_second_class, avg_marks_all_students, x],
  split,
  { sorry }, -- Here we skip the actual proof for simplicity
  { refl },
end

end determine_num_students_first_class_l667_667199


namespace general_formula_a_n_T_n_greater_than_S_n_l667_667930

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n_l667_667930


namespace train_lengths_l667_667251

variable (P L_A L_B : ℝ)

noncomputable def speedA := 180 * 1000 / 3600
noncomputable def speedB := 240 * 1000 / 3600

-- Train A crosses platform P in one minute
axiom hA : speedA * 60 = L_A + P

-- Train B crosses platform P in 45 seconds
axiom hB : speedB * 45 = L_B + P

-- Sum of the lengths of Train A and platform P is twice the length of Train B
axiom hSum : L_A + P = 2 * L_B

theorem train_lengths : L_A = 1500 ∧ L_B = 1500 :=
by
  sorry

end train_lengths_l667_667251


namespace mystical_creatures_are_enchanted_beings_l667_667984

universe u

variable (Dragon MysticalCreature EnchantedBeing : Type u)

variable (AllDragonsAreMystical : ∀ (d : Dragon), MysticalCreature)
variable (SomeEnchantedAreDragons : ∃ (e : EnchantedBeing), Dragon)

theorem mystical_creatures_are_enchanted_beings :
  ∃ (m : MysticalCreature), EnchantedBeing :=
sorry

end mystical_creatures_are_enchanted_beings_l667_667984


namespace cot_30_eq_sqrt_3_l667_667831

theorem cot_30_eq_sqrt_3 (θ : ℝ) (hθ : θ = 30) (h1 : tan 30 = 1 / real.sqrt 3) : cot 30 = real.sqrt 3 :=
by
  have h2: cot 30 = 1 / (tan 30), by exact cot_def
  have h3: cot 30 = 1 / (1 / real.sqrt 3), by rw [h1, h2]
  have h4: cot 30 = real.sqrt 3, by exact (one_div_div (1 / real.sqrt 3))
  exact h4


end cot_30_eq_sqrt_3_l667_667831


namespace max_subset_T_l667_667112

def valid_subset (T : Set ℕ) : Prop :=
  ∀ (x y : ℕ), x ∈ T → y ∈ T → x ≠ y → abs (x - y) ≠ 3 ∧ abs (x - y) ≠ 6

def max_elements (s : ℕ) : ℕ :=
  if (s ≤ 10) then 4 else 199 * 4 + 4

theorem max_subset_T (T : Set ℕ) (H1 : ∀ x, x ∈ T → 1 ≤ x ∧ x ≤ 1999) (H2 : valid_subset T) :
  ∃ S ⊆ {n : ℕ | 1 ≤ n ∧ n ≤ 1999}, valid_subset S ∧ card S = 800 :=
sorry

end max_subset_T_l667_667112


namespace find_lowest_score_l667_667190

noncomputable def lowest_score (total_scores : ℕ) (mean_scores : ℕ) (new_total_scores : ℕ) (new_mean_scores : ℕ) (highest_score : ℕ) : ℕ :=
  let total_sum        := mean_scores * total_scores
  let sum_remaining    := new_mean_scores * new_total_scores
  let removed_sum      := total_sum - sum_remaining
  let lowest_score     := removed_sum - highest_score
  lowest_score

theorem find_lowest_score :
  lowest_score 15 90 13 92 110 = 44 :=
by
  unfold lowest_score
  simp [Nat.mul_sub, Nat.sub_apply, Nat.add_comm]
  rfl

end find_lowest_score_l667_667190


namespace simplify_expression_l667_667593

theorem simplify_expression :
  (sin (35 * (Real.pi / 180))) ^ 2 - 1 / 2) / sin (20 * (Real.pi / 180)) = - 1 / 2 := 
  by sorry

end simplify_expression_l667_667593


namespace sequence_12th_term_l667_667499

theorem sequence_12th_term (C : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 3) (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = C / a n) (h4 : C = 12) : a 12 = 4 :=
sorry

end sequence_12th_term_l667_667499


namespace correct_equation_l667_667233

theorem correct_equation (x : ℤ) : 232 + x = 3 * (146 - x) :=
sorry

end correct_equation_l667_667233


namespace complex_expression_evaluation_l667_667302

noncomputable def x : ℝ := Real.root 4 6
noncomputable def y : ℝ := Real.root 8 2

theorem complex_expression_evaluation :
  ( (x + 2 * y) / (8 * y^3 * (x^2 + 2 * x * y + 2 * y^2))
  - (x - 2 * y) / (8 * y^3 * (x^2 - 2 * x * y + 2 * y^2))
  +  y^(-2) / (4 * x^2 - 8 * y^2)
  - 1 / (4 * x^2 * y^2 + 8 * y^4) )
  = -13 / 28 := by
  sorry

end complex_expression_evaluation_l667_667302


namespace propositions_validity_l667_667294

theorem propositions_validity
  (a b x y λ : ℝ)
  (A B C : set ℝ)
  (ha : a < b ∧ b < 0)
  (hλ : -2 ≤ λ ∧ λ ≤ 3)
  (hx2y2 : x^2 ≠ y^2)
  (hxABC : x ∈ (A ∪ B) ∩ C) :
  (a < b → b < 0 → (1/a > 1/b) ∧ ¬((1/a > 1/b) → a < b < 0)) ∧
  (-2 ≤ λ → λ ≤ 3 → (λ ∈ set.Icc (-2 : ℝ) 3) ∧ ¬((λ ∈ set.Icc (-2 : ℝ) 3) → λ ∈ set.Icc (-1 : ℝ) 3)) ∧
  ¬(x^2 ≠ y^2 ↔ x ≠ y) ∧
  ¬((x ∈ (A ∪ B) ∩ C) → x ∈ (A ∩ B) ∪ C) :=
by
  -- Placeholder for the proof
  apply and.intro
  { intro h1 ha hb
    exact ⟨fun _ _ => sorry, fun _ => ⟨sorry, sorry⟩⟩
  }
  apply and.intro
  { intro h1 hλh2
    exact ⟨sorry, sorry⟩
  }
  apply and.intro
  { exact sorry }
  { exact sorry }

end propositions_validity_l667_667294


namespace stratified_sampling_l667_667709

theorem stratified_sampling 
  (students_first_grade : ℕ)
  (students_second_grade : ℕ)
  (selected_first_grade : ℕ)
  (x : ℕ)
  (h1 : students_first_grade = 400)
  (h2 : students_second_grade = 360)
  (h3 : selected_first_grade = 60)
  (h4 : (selected_first_grade / students_first_grade : ℚ) = (x / students_second_grade : ℚ)) :
  x = 54 :=
sorry

end stratified_sampling_l667_667709


namespace tangent_line_equation_l667_667842

theorem tangent_line_equation :
  let y := λ x : ℝ, x / (x + 2)
  let p := (-1, -1)
  TangentLineEquation y p "2x - y + 1 = 0" :=
sorry

end tangent_line_equation_l667_667842


namespace siblings_pizza_order_l667_667421

/-- Four siblings ordered an extra large pizza. 
    Alex ate 1/6, Beth 2/5, and Cyril 1/4 of the pizza. 
    Dan got the leftovers. Prove the sequence of the siblings 
    in decreasing order of the part of the pizza they consumed. --/
theorem siblings_pizza_order : 
  let total_pieces := 60 in
  let Alex_share := 1 / 6 in
  let Beth_share := 2 / 5 in
  let Cyril_share := 1 / 4 in
  let Alex_pieces := Alex_share * total_pieces in
  let Beth_pieces := Beth_share * total_pieces in
  let Cyril_pieces := Cyril_share * total_pieces in
  let remaining_pieces := total_pieces - (Alex_pieces + Beth_pieces + Cyril_pieces) in
  let Dan_pieces := remaining_pieces in
  (Beth_pieces >= Cyril_pieces) ∧ (Cyril_pieces >= Dan_pieces) ∧ (Dan_pieces >= Alex_pieces)
: 
  Beth, Cyril, Dan, Alex := sorry

end siblings_pizza_order_l667_667421


namespace harry_morning_routine_l667_667395

variable (t1 t2 : ℕ)
variable (h1 : t1 = 15)
variable (h2 : t2 = 2 * t1)

theorem harry_morning_routine : t1 + t2 = 45 := by
  sorry

end harry_morning_routine_l667_667395


namespace find_a_l667_667472

theorem find_a (a : ℝ) (h₀ : 0 < a) (h₁ : (a + 1)^(2 - 2) + 1 = 2) (h₂ : log 3 (2 + a) = 2) : a = 7 :=  
sorry

end find_a_l667_667472


namespace last_digit_one_over_three_pow_neg_ten_l667_667275

theorem last_digit_one_over_three_pow_neg_ten : (3^10) % 10 = 9 := by
  sorry

end last_digit_one_over_three_pow_neg_ten_l667_667275


namespace simplify_expression_l667_667604

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b ^ 2 + 2 * b) - 2 * b ^ 2 = 9 * b ^ 3 + 4 * b ^ 2 :=
by
  sorry

end simplify_expression_l667_667604


namespace relationship_between_abc_l667_667863

-- Given conditions as definitions
variables {a b c x : ℝ}
axiom h1 : 3 * 2^a = 2^(b + 1)
axiom h2 : a = c + x^2 - x + 1

-- Goal definition relating a, b, and c
theorem relationship_between_abc : b > a ∧ a > c :=
sorry

end relationship_between_abc_l667_667863


namespace parabola_equation_l667_667231

-- Definitions of the conditions
def is_vertex_at_origin : Prop := vertex = (0, 0)
def is_focus_on_y_axis (focus : ℝ × ℝ) : Prop := focus.1 = 0
def distance (P focus : ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2)
def point_on_parabola (P : ℝ × ℝ) (p : ℝ) : Prop := P.2 = p/2

-- Stating the theorem to be proven
theorem parabola_equation (vertex focus : ℝ × ℝ) (m : ℝ) (p : ℝ) :
  is_vertex_at_origin ∧ is_focus_on_y_axis focus ∧ point_on_parabola (m, 1) p ∧ distance (m, 1) focus = 5 →
  2 * p = 16 :=
sorry

end parabola_equation_l667_667231


namespace cot_30_eq_sqrt3_l667_667783

theorem cot_30_eq_sqrt3 : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by
  have h_cos : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  have h_sin : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  show Real.cot (30 * Real.pi / 180) = Real.sqrt 3
  calc
    Real.cot (30 * Real.pi / 180) = (Real.cos (30 * Real.pi / 180)) / (Real.sin (30 * Real.pi / 180)) := sorry
    ... = (Real.sqrt 3 / 2) / (1 / 2) := sorry
    ... = Real.sqrt 3 := sorry

end cot_30_eq_sqrt3_l667_667783


namespace exists_consecutive_integers_prime_ratios_l667_667157

theorem exists_consecutive_integers_prime_ratios :
  ∃ (a : ℕ), ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 2009 →
  let n := a + i
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ∣ n ∧ q ∣ n ∧ q > 20 * p :=
begin
  sorry
end

end exists_consecutive_integers_prime_ratios_l667_667157


namespace y_intercept_of_line_l667_667266

theorem y_intercept_of_line : ∃ y : ℝ, 2 * 0 - 3 * y = 6 ∧ y = -2 :=
by
  exists (-2)
  split
  . simp
  . rfl

end y_intercept_of_line_l667_667266


namespace perfect_square_divisors_product_factorials_l667_667484

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def product_factorials (n : ℕ) : ℕ :=
  ∏ i in finset.range (n + 1), factorial i

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def number_of_perfect_square_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ k, is_perfect_square k).card

theorem perfect_square_divisors_product_factorials :
  number_of_perfect_square_divisors (product_factorials 12) = 5280 :=
by sorry

end perfect_square_divisors_product_factorials_l667_667484


namespace tetrahedron_edge_lengths_l667_667658

noncomputable def radius_of_sphere := (125: ℝ)^(1/3 : ℝ)
def base_edge_1 := 2 * radius_of_sphere
def base_edge_2 := 8 * radius_of_sphere / 5
def base_edge_3 := 6 * radius_of_sphere / 5
def height_of_pyramid := radius_of_sphere
def volume_of_pyramid := 40
def base_right_triangle_area := 24 * (radius_of_sphere^2) / 25

theorem tetrahedron_edge_lengths :
  ∃ r : ℝ,
  r = radius_of_sphere ∧
  base_edge_1 = 2 * r ∧
  base_edge_2 = 8 * r / 5 ∧
  base_edge_3 = 6 * r / 5 ∧
  height_of_pyramid = r ∧
  (1/3) * base_right_triangle_area * height_of_pyramid = volume_of_pyramid ∧
  base_edge_1 = 10 ∧
  base_edge_2 = 8 ∧
  base_edge_3 = 6 :=
begin
  use radius_of_sphere,
  split, { refl },
  split, { rw base_edge_1, refl },
  split, { rw base_edge_2, refl },
  split, { rw base_edge_3, refl },
  split, { rw height_of_pyramid, refl },
  split, {
    rw [base_right_triangle_area, height_of_pyramid],
    norm_num,
    ring,
  },
  norm_num,
end

end tetrahedron_edge_lengths_l667_667658


namespace relative_positions_of_points_on_parabola_l667_667444

/-- Given a function f(x) = x^2 - 2x + 3 and points on this function, -/
def check_relative_positions (y : ℝ → ℝ) (y1 y2 y3 : ℝ) : Prop :=
  y (-3) = y1 ∧ y (1) = y2 ∧ y (-1/2) = y3 → y2 < y3 ∧ y3 < y1

theorem relative_positions_of_points_on_parabola :
  check_relative_positions (λ x, x^2 - 2 * x + 3) y1 y2 y3 :=
sorry

end relative_positions_of_points_on_parabola_l667_667444


namespace cot_30_eq_sqrt3_l667_667775

theorem cot_30_eq_sqrt3 : Real.cot (π / 6) = Real.sqrt 3 := by
  let tan_30 := 1 / Real.sqrt 3
  have tan_30_eq : Real.tan (π / 6) = tan_30 := sorry
  have cot_def : ∀ x, Real.cot x = 1 / Real.tan x := sorry
  rw [cot_def (π / 6), tan_30_eq]
  simp
  sorry

end cot_30_eq_sqrt3_l667_667775


namespace sin_arctan_eq_x_l667_667235

noncomputable def find_x : ℝ := Real.sqrt ((-1 + Real.sqrt 5) / 2)

theorem sin_arctan_eq_x : 
  ∃ (x : ℝ), x > 0 ∧ Real.sin (Real.arctan x) = x ∧ x = find_x :=
by
  use find_x
  split
  { -- Prove x > 0
    sorry }
  split
  { -- Prove sin(arctan(x)) = x
    sorry }
  { -- Prove x = find_x
    sorry }

end sin_arctan_eq_x_l667_667235


namespace rabbit_total_distance_l667_667249

theorem rabbit_total_distance 
    (r_small r_large : ℝ) (arc_fraction : ℝ) (arc_r_small : ℝ) (arc_r_large : ℝ) (radial_distance : ℝ)
    (h1 : r_small = 5) 
    (h2 : r_large = 15) 
    (h3 : arc_fraction = 1/3)
    (h4 : arc_r_small = arc_fraction * 2 * Real.pi * r_small)
    (h5 : arc_r_large = arc_fraction * 2 * Real.pi * r_large)
    (h6 : radial_distance = r_large - r_small) : 
    arc_r_small + radial_distance + arc_r_large + radial_distance = (40 * Real.pi) / 3 + 20 :=
by
    rw [h4, h5, h6],
    simp only [h1, h2, h3],
    norm_num,
    sorry

end rabbit_total_distance_l667_667249


namespace circle_eq_l667_667432

theorem circle_eq (M : ℝ × ℝ) (P Q : ℝ × ℝ) (l : ℝ → ℝ) (h1 : M = (-1, 2))
  (h2 : (l = fun x => x + 4) → (2 * (P.1 + 1) * (P.2 - 2) = (P.1 + 1) * (P.1 + 1) + (P.2 - 2) * (P.2 - 2) = 1))
  (h3 : l P.1 = P.2 - 1)
  (h4 : (dist (P.1 + 1) (P.2 - 2) = 4 * dist (Q.1 - 1) (Q.2 - 2)))
  : ∃ P : ℝ × ℝ, (P = (-1, -2) ∨ P = (3, 2)) := 
  sorry

end circle_eq_l667_667432


namespace possible_degrees_of_remainder_l667_667288

noncomputable def divisor : Polynomial ℤ := 5 * Polynomial.X ^ 7 - 7 * Polynomial.X ^ 3 + 3 * Polynomial.X - 8

theorem possible_degrees_of_remainder (f : Polynomial ℤ) :
  ∃ r : Polynomial ℤ, ∀ d (q : Polynomial ℤ), (f = divisor * q + r) →
  (r.degree < divisor.degree) ∧ (r.degree.toNat = d) →
  d ∈ {0, 1, 2, 3, 4, 5, 6} :=
sorry

end possible_degrees_of_remainder_l667_667288


namespace least_five_digit_congruence_l667_667276

theorem least_five_digit_congruence : ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 18 = 7 ∧ n % 9 = 4 ∧ n = 10012 := 
by {
  -- existence proof skipped
  existsi 10012,
  -- checking properties
  split, exact nat.le_refl 10012,
  split, exact nat.lt_succ_self 10012,
  split, exact nat.mod_eq_of_lt (nat.le_succ_of_le (nat.le_of_lt_succ (nat.lt_succ_self 10012))),
  split, exact nat.mod_eq_of_lt (nat.le_succ_of_le (nat.le_of_lt_succ (nat.lt_succ_self 10012))),
  exact rfl,
}

end least_five_digit_congruence_l667_667276


namespace second_day_sold_pear_percentage_l667_667345

noncomputable def initial_pear_count : ℝ := 100
noncomputable def first_day_sold_pear_percentage : ℝ := 0.8
noncomputable def first_day_thrown_away_percentage : ℝ := 0.5
noncomputable def total_pear_thrown_away_percentage : ℝ := 0.11999999999999996

theorem second_day_sold_pear_percentage :
  let remaining_after_sell_first_day := initial_pear_count * (1 - first_day_sold_pear_percentage),
      remaining_after_throw_first_day := remaining_after_sell_first_day * (1 - first_day_thrown_away_percentage),
      total_thrown_away := initial_pear_count * total_pear_thrown_away_percentage,
      second_day_thrown_away := total_thrown_away - (initial_pear_count - remaining_after_sell_first_day),
      second_day_sold_pear := remaining_after_throw_first_day - second_day_thrown_away,
      second_day_sold_percentage := second_day_sold_pear / remaining_after_throw_first_day
  in second_day_sold_percentage = 0.8 := by have : sorry; sorry

end second_day_sold_pear_percentage_l667_667345


namespace general_formula_a_n_T_n_greater_S_n_l667_667870

-- Define the conditions for the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ (n : ℕ), a n = a1 + (n - 1) * d

def b_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_of_first_n_terms (a S : ℕ → ℕ) :=
  ∀ (n k : ℕ), k ≤ n → S n = (n * (a 1 + a n)) / 2

-- Given S4 = 32 and T3 = 16
def S_4_equals_32 (S : ℕ → ℕ) : Prop := S 4 = 32
def T_3_equals_16 (T : ℕ → ℕ) : Prop := T 3 = 16

-- Prove that the general formula for a_n is 2n + 3
theorem general_formula_a_n (a : ℕ → ℕ) (h : arithmetic_sequence a) : 
  a = λ n, 2 * n + 3 := 
sorry

-- Prove that for n > 5, T_n > S_n
theorem T_n_greater_S_n (a b S T: ℕ → ℕ) 
  (ha : arithmetic_sequence a) 
  (hb : b_sequence a b)
  (hS : sum_of_first_n_terms a S)
  (hT : sum_of_first_n_terms b T)
  (hS4 : S_4_equals_32 S)
  (hT3 : T_3_equals_16 T) 
  (n : ℕ) : n > 5 → T n > S n :=
sorry

end general_formula_a_n_T_n_greater_S_n_l667_667870


namespace polynomial_average_t_l667_667461

-- Definitions for conditions of the problem
def is_polynomial (f : ℤ → ℤ) : Prop := 
  ∃ (p q : ℤ), (∀ x, f x = x ^ 2 + p * x + q)

def positive_integer_root (r : ℤ) : Prop := 
  r > 1 ∧ ∃ n : ℕ, n > 0 ∧ r = n

-- Main statement
theorem polynomial_average_t :
  ∀ t : ℤ, 
    (∀ r1 r2 : ℤ, is_polynomial (λ x, x^2 - 7 * x + t) ∧ positive_integer_root r1 ∧ positive_integer_root r2 ∧
     r1 + r2 = 7 ∧ r1 * r2 = t) →
    t = 10 ∨ t = 12 → 
    (t = 10 ∨ t = 12) → 
    (10 + 12) / 2 = 11 :=
by sorry

end polynomial_average_t_l667_667461


namespace find_angle_B_find_a_c_l667_667989

open Real

noncomputable section

-- Define the problem conditions as variables
variables {A B C a b c : ℝ}

-- Define the given equation condition
def given_equation := 4 * sin (A + C) / 2 ^ 2 - cos 2 * B = 7 / 2

-- First part: Prove the measure of angle B
theorem find_angle_B (h : given_equation) : B = π / 3 :=
by {
  sorry
}

-- Second part: Prove the values of a and c
theorem find_a_c (ha : a + c = 3) (hb : b = sqrt 3) (hB : B = π / 3) : ((a = 1 ∧ c = 2) ∨ (a = 2 ∧ c = 1)) :=
by {
  sorry
}

end find_angle_B_find_a_c_l667_667989


namespace harmonious_interval_real_roots_l667_667001

theorem harmonious_interval_real_roots (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 1) ↔ ∃ m n : ℝ, m ≠ n ∧ (m + n = (a + 1) / a ∧ m * n = 1) ∧ 
  (ax^2 - (a + 1)x + a = 0) :=
sorry

end harmonious_interval_real_roots_l667_667001


namespace distance_between_O_and_plane_l667_667343

noncomputable def distance_from_sphere_center_to_triangle_plane {r : ℝ} {a b c : ℝ} (h_r : r = 10) 
  (h_triangle : a = 8 ∧ b = 15 ∧ c = 17) (h_tangent : ∀ x, x = a ∨ x = b ∨ x = c → tangent_to_sphere x r) : ℝ :=
  let area := 1 / 2 * a * b in
  let s := (a + b + c) / 2 in
  let inradius := area / s in
  let x := sqrt (r^2 - inradius^2) in
  x

-- Dummy definition to indicate the tangent condition, which should be refined to a proper mathematical definition
def tangent_to_sphere (x r : ℝ) : Prop := True

theorem distance_between_O_and_plane (r : ℝ) (a b c : ℝ) (h_r : r = 10) 
  (h_triangle : a = 8 ∧ b = 15 ∧ c = 17) (h_tangent : ∀ x, x = a ∨ x = b ∨ x = c → tangent_to_sphere x r) :
  distance_from_sphere_center_to_triangle_plane h_r h_triangle h_tangent = sqrt 91 :=
sorry

end distance_between_O_and_plane_l667_667343


namespace Tn_gt_Sn_l667_667882

-- Definitions of the sequences and initial conditions
def a : ℕ → ℕ := λ n, 2 * n + 3
def b (n : ℕ) : ℕ := if n % 2 = 1 then a n - 6 else 2 * a n
def S (n : ℕ) : ℕ := (n * (2 * n + 8))/2
def T : ℕ → ℕ
| 0     := 0
| (n+1) := T n + b (n + 1)

-- Given initial conditions
axiom S_4_eq_32 : S 4 = 32
axiom T_3_eq_16 : T 3 = 16
axiom a_general : ∀ n : ℕ, a n = 2 * n + 3

-- Proof of the main theorem
theorem Tn_gt_Sn (n : ℕ) (h : n > 5) : T n > S n := by
  sorry

end Tn_gt_Sn_l667_667882


namespace length_of_EC_l667_667694

variable (AC : ℝ) (AB : ℝ) (CD : ℝ) (EC : ℝ)

def is_trapezoid (AB CD : ℝ) : Prop := AB = 3 * CD
def perimeter (AB CD AC : ℝ) : Prop := AB + CD + AC + (AC / 3) = 36

theorem length_of_EC
  (h1 : is_trapezoid AB CD)
  (h2 : AC = 18)
  (h3 : perimeter AB CD AC) :
  EC = 9 / 2 :=
  sorry

end length_of_EC_l667_667694


namespace cot_30_eq_sqrt3_l667_667779

theorem cot_30_eq_sqrt3 : Real.cot (π / 6) = Real.sqrt 3 := by
  let tan_30 := 1 / Real.sqrt 3
  have tan_30_eq : Real.tan (π / 6) = tan_30 := sorry
  have cot_def : ∀ x, Real.cot x = 1 / Real.tan x := sorry
  rw [cot_def (π / 6), tan_30_eq]
  simp
  sorry

end cot_30_eq_sqrt3_l667_667779


namespace cot_30_deg_l667_667793

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 := by
  -- Define known values of trigonometric functions at 30 degrees
  have h1 : Real.tan (30 * Real.pi / 180) = 1 / Real.sqrt 3 := sorry
  have h2 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

  -- Prove using these known values
  calc
    Real.cot (30 * Real.pi / 180)
        = 1 / Real.tan (30 * Real.pi / 180) : by sorry
    ... = 1 / (1 / Real.sqrt 3) : by sorry
    ... = Real.sqrt 3 : by sorry

end cot_30_deg_l667_667793


namespace tenisha_dogs_female_percentage_l667_667613

def tenishaDogs (totalDogs femalePercentage : ℕ) (puppiesLeft donatedPuppies : ℕ) :=
  let femaleDogs := femalePercentage * totalDogs / 100
  let birthingFemaleDogs := 3 * femaleDogs / 4
  let totalPuppiesBorn := 10 * birthingFemaleDogs
  totalPuppiesBorn = puppiesLeft + donatedPuppies

/-- What percentage of the dogs are female? -/
theorem tenisha_dogs_female_percentage (F : ℕ) (h : tenishaDogs 40 F 50 130) : F = 60 :=
begin
  sorry
end

end tenisha_dogs_female_percentage_l667_667613


namespace Mr_Mayer_purchase_price_l667_667576

theorem Mr_Mayer_purchase_price 
  (P : ℝ) 
  (H1 : (1.30 * 2) * P = 2600) : 
  P = 1000 := 
by
  sorry

end Mr_Mayer_purchase_price_l667_667576


namespace pencil_length_difference_l667_667713

theorem pencil_length_difference (a b : ℝ) (h1 : a = 1) (h2 : b = 4/9) :
  a - b - b = 1/9 :=
by
  rw [h1, h2]
  sorry

end pencil_length_difference_l667_667713


namespace sum_le_five_thirds_l667_667106

-- Let x_i be a sequence of non-negative real numbers such that x_i * x_j <= 4^(-|i - j|)
open Real

theorem sum_le_five_thirds (n : ℕ) 
  (x : Fin n → ℝ) 
  (hx : ∀i j : Fin n, x i * x j ≤ 4^(-|i.val - j.val|)) 
  (hnonneg : ∀i : Fin n, 0 ≤ x i) : 
  (∑ i in Finset.finRange n, x i) ≤ 5 / 3 :=
by
  sorry

end sum_le_five_thirds_l667_667106


namespace geometric_mean_condition_l667_667108

variable {A B C D : Type} [EuclideanGeometry A B C D]
variables (alpha beta gamma : Angle) (sin_alpha sin_beta sin_gamma_half : Real)

def sin (x : Angle) : Real := sorry
def sin_half (x : Angle) : Real := sorry
def isTriangle (A B C : Type) : Prop := sorry
def existsPointOnSegment (A B : Type) (D : Type) : Prop := sorry
def geometricMeanLength (AD BD CD : Real) : Prop := CD = Real.sqrt (AD * BD)

theorem geometric_mean_condition (A B C : Type)
                                  [isTriangle A B C]
                                  (alpha beta gamma : Angle)
                                  (sin_alpha := sin alpha)
                                  (sin_beta := sin beta)
                                  (sin_gamma_half := sin_half gamma)
                                  (D : Type)
                                  [existsPointOnSegment A B D]
                                  (x y : Real) : 
  geometricMeanLength x y (Real.sqrt (x * y)) -> 
  sin_alpha * sin_beta <= sin_gamma_half^2 := by
  sorry

end geometric_mean_condition_l667_667108


namespace lowest_score_of_15_scores_l667_667174

theorem lowest_score_of_15_scores (scores : List ℝ) (h_len : scores.length = 15) (h_mean : (scores.sum / 15) = 90)
    (h_mean_13 : ((scores.erase 110).erase (scores.head)).sum / 13 = 92) (h_highest : scores.maximum = some 110) :
    scores.minimum = some 44 :=
by
    sorry

end lowest_score_of_15_scores_l667_667174


namespace sufficient_but_not_necessary_l667_667967

def l1 (m : ℝ) : ℝ × ℝ → ℝ
| (x, y) => m * x + (m + 1) * y + 2

def l2 (m : ℝ) : ℝ × ℝ → ℝ
| (x, y) => (m + 1) * x + (m + 4) * y - 3

def perpendicular_slopes (m : ℝ) : Prop :=
  let slope_l1 := -m / (m + 1)
  let slope_l2 := -(m + 1) / (m + 4)
  slope_l1 * slope_l2 = -1

theorem sufficient_but_not_necessary (m : ℝ) : m = -2 → (∃ k, m = -k ∧ perpendicular_slopes k) :=
by
  sorry

end sufficient_but_not_necessary_l667_667967


namespace airplane_speed_and_distance_l667_667297

theorem airplane_speed_and_distance (wind_speed : ℝ) (time_with_wind : ℝ) (time_against_wind : ℝ) 
  (speed_with_wind : ℝ) (speed_against_wind : ℝ) 
  (h1 : wind_speed = 24) 
  (h2 : time_with_wind = 2.8) 
  (h3 : time_against_wind = 3)
  (h4 : speed_with_wind = 696 + 24) 
  (h5 : speed_against_wind = 696 - 24) : 
  let avg_speed := 696 in
  let distance := 2016 in
  avg_speed = 696 ∧ distance = 2016 :=
by
  sorry

end airplane_speed_and_distance_l667_667297


namespace count_odd_units_digit_in_range_l667_667417

-- Define the range of n
def range := {n | n ∈ (list.range 200).map (λ i, i + 1)}

-- Define a predicate to check if the unit digit of n^2 is odd
def units_digit_odd (n : ℕ) : Prop :=
  let units_digit := n % 10 in
  let odd_units := [1, 3, 5, 7, 9] in
  (units_digit * units_digit) % 10 ∈ odd_units

-- Lean statement to prove the count of n in the range with odd unit digit of n^2 is 100
theorem count_odd_units_digit_in_range :
  (list.filter (λ n, units_digit_odd n) (list.range' 1 200)).length = 100 := 
sorry

end count_odd_units_digit_in_range_l667_667417


namespace lowest_score_of_15_scores_l667_667177

theorem lowest_score_of_15_scores (scores : List ℝ) (h_len : scores.length = 15) (h_mean : (scores.sum / 15) = 90)
    (h_mean_13 : ((scores.erase 110).erase (scores.head)).sum / 13 = 92) (h_highest : scores.maximum = some 110) :
    scores.minimum = some 44 :=
by
    sorry

end lowest_score_of_15_scores_l667_667177


namespace isosceles_triangle_perimeter_l667_667229

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 4) (h₂ : b = 9) (h_iso : ¬(4 + 4 > 9 ∧ 4 + 9 > 4 ∧ 9 + 4 > 4))
  (h_ineq : (9 + 9 > 4) ∧ (9 + 4 > 9) ∧ (4 + 9 > 9)) : 2 * b + a = 22 :=
by sorry

end isosceles_triangle_perimeter_l667_667229


namespace evaluate_ceiling_neg_cubed_frac_l667_667374

theorem evaluate_ceiling_neg_cubed_frac :
  (Int.ceil ((- (5 : ℚ) / 3) ^ 3 + 1) = -3) :=
sorry

end evaluate_ceiling_neg_cubed_frac_l667_667374


namespace find_x_l667_667491

theorem find_x : ∃ x : ℚ, (3 * x + 5) / 5 = 17 ↔ x = 80 / 3 := by
  sorry

end find_x_l667_667491


namespace power_function_value_l667_667034

noncomputable def f (x : ℝ) : ℝ := x^(1 / 2)

theorem power_function_value :
  (f 9) = 3 :=
by
  have h : ∃ α : ℝ, (2 : ℝ)^α = sqrt 2 := ⟨1 / 2, rfl⟩
  have hα : (2 : ℝ)^(1 / 2) = sqrt 2 := rfl
  show (9)^(1 / 2) = 3 from rfl
  sorry

end power_function_value_l667_667034


namespace additional_boys_l667_667306

theorem additional_boys (initial additional total : ℕ) 
  (h1 : initial = 22) 
  (h2 : total = 35) 
  (h3 : additional = total - initial) : 
  additional = 13 := 
by 
  rw [h1, h2]
  simp
  sorry

end additional_boys_l667_667306


namespace arithmetic_seq_b_seq_formula_min_value_m_l667_667008

open Nat

noncomputable def a_seq (n : ℕ) : ℕ := 3^(n-1)

noncomputable def b_seq (n : ℕ) : ℕ := 2*n - 1

theorem arithmetic_seq (q : ℕ) (hq1 : q > 1) (ha1 : a_seq 1 = 1) 
  (harith : 2 * a_seq 3 = a_seq 1 + a_seq 2 + 14) : 
  q = 3 ∧ ∀ n, a_seq n = 3^(n-1) :=
sorry

theorem b_seq_formula (n : ℕ) :
  a_seq 1 * b_seq 1 + a_seq 2 * b_seq 2 + ... + a_seq n * b_seq n = 
  (n - 1) * 3^n + 1 →
  ∀ n, b_seq n = 2*n - 1 :=
sorry

theorem min_value_m (m : ℝ) :
  (∀ n, m * a_seq n ≥ b_seq n - 8) →
  m ≥ 1/81 :=
sorry

end arithmetic_seq_b_seq_formula_min_value_m_l667_667008


namespace banker_l667_667684

/-
Conditions:
1. Time in years (T) is 3.
2. Rate per annum (R) is 0.12.
3. Banker's gain (BG) is 270.
-/
def T : ℕ := 3
def R : ℝ := 0.12
def BG : ℝ := 270

/-
Definitions:
TD - True Discount
BD - Banker's Discount
-/
def TD (P : ℝ) : ℝ := (P * R * T) / 100

theorem banker's_discount (BD : ℝ) (P : ℝ) (TD : ℝ) : BG = (BD * TD) / (BD - TD) →
  P = 750 → BD = 421.88 :=
by
  sorry

end banker_l667_667684


namespace triangle_side_b_value_l667_667988

theorem triangle_side_b_value (a b c : ℝ) (h_seq : 2 * b = a + c) (h_angle : B = 60) (h_area : (a * c * Math.sin (B * Real.pi / 180)) / 2 = 3 * Real.sqrt 3) : b = 2 * Real.sqrt 3 :=
by
  sorry

end triangle_side_b_value_l667_667988


namespace trajectory_proof_slope_proof_line_equation_proof_l667_667002

variable {P : ℝ × ℝ} (x y : ℝ)
variable {A B M : ℝ × ℝ} 
variable (t m : ℝ)

noncomputable def trajectory_condition := 
  ∀ P = (x, y), y > 0 → (sqrt (x^2 + (y - 1)^2) - y) = 1

noncomputable def trajectory_equation : Prop := 
  x^2 = 4 * y ∧ y > 0

noncomputable def curve_condition (A B : ℝ × ℝ) :=
  A.1 ≠ B.1 ∧ (A.1 + B.1 = 4) ∧ (A.1 ^ 2 = 4 * A.2) ∧ (B.1 ^ 2 = 4 * B.2)

noncomputable def slope (A B : ℝ × ℝ) : ℝ :=
  (A.2 - B.2) / (A.1 - B.1)

noncomputable def slope_condition (A B : ℝ × ℝ) :=
  slope A B = 1

noncomputable def line_equation (y m : ℝ) : Prop :=
  y = x + m

noncomputable def tangent_condition (M : ℝ × ℝ) :=
  ∀ (x : ℝ), (M = (x, x + t)) ∧ (x^2 = 4 * (x + t)) ∧ t = -1

noncomputable def perpendicular_condition (A M B : ℝ × ℝ) :=
  (A.1 + B.1 = 4) ∧ (A.1 * B.1 = -4 * m) ∧ |A.1 - M.1| = |M.1 - B.1|

theorem trajectory_proof (x y : ℝ) :
  trajectory_condition P x y → trajectory_equation x y := 
sorry

theorem slope_proof (x1 y1 x2 y2 : ℝ) :
  curve_condition (x1, y1) (x2, y2) → slope_condition (x1, y1) (x2, y2) :=
sorry

theorem line_equation_proof (x y t m x1 y1 x2 y2 : ℝ) :
  curve_condition (x1, y1) (x2, y2) →
  tangent_condition (x, y) t →
  perpendicular_condition (x1, y1) (x, y) (x2, y2) → 
  line_equation y m :=
sorry

end trajectory_proof_slope_proof_line_equation_proof_l667_667002


namespace voted_in_favor_of_both_l667_667739

variable (U A B : Finset ℕ)

def eligible_students : ℕ := 250
def voted_first : ℕ := 175
def voted_second : ℕ := 140
def against_both : ℕ := 35
def abstained : ℕ := 10

theorem voted_in_favor_of_both : 
  let V := eligible_students - abstained in
  let A_union_B := V - against_both in
  A ∩ B.card = voted_first + voted_second - A_union_B := by sorry

end voted_in_favor_of_both_l667_667739


namespace cylinder_quadrilateral_intersection_l667_667258

-- Definitions for the intersection types for the geometric solids.
def can_intersect_plane_as_quadrilateral (solid : Type) : Prop :=
  solid = Cylinder

theorem cylinder_quadrilateral_intersection :
  (∀ solid, can_intersect_plane_as_quadrilateral solid ↔ solid = Cylinder) :=
begin
  intros solid,
  split,
  { -- Prove that if a solid can intersect plane as a quadrilateral, then it is a cylinder
    intro h,
    exact h
  },
  { -- Prove that a cylinder can intersect plane as a quadrilateral
    intro h,
    rw h,
    refl,
  }
end

end cylinder_quadrilateral_intersection_l667_667258


namespace probability_of_soybean_in_triangle_PBC_l667_667451

-- Definitions related to points and vectors in the plane
variables {A B C P : Type}
variables [affine_space ℝ A]

-- Given conditions
variable (h : ((vector ℝ B - vector ℝ P) + (vector ℝ C - vector ℝ P) + 2 * (vector ℝ A - vector ℝ P) = (0 : vector ℝ)))

theorem probability_of_soybean_in_triangle_PBC :
  (∃ (A B C P : A), h → (measure (P ∈ interior (triangle A B C))) / (measure (P ∈ interior (triangle A B C))) = 1/2) :=
sorry

end probability_of_soybean_in_triangle_PBC_l667_667451


namespace rectangle_width_is_16_l667_667132

-- Definitions based on the conditions
def length : ℝ := 24
def ratio := 6 / 5
def perimeter := 80

-- The proposition to prove
theorem rectangle_width_is_16 (W : ℝ) (h1 : length = 24) (h2 : length = ratio * W) (h3 : 2 * length + 2 * W = perimeter) :
  W = 16 :=
by
  sorry

end rectangle_width_is_16_l667_667132


namespace CB_equals_BE_l667_667511

-- Defining a right triangle with right angle at C and height CK from C to AB
-- and angle bisector CE in triangle ACK
variables {A B C K E : Type*}
variables {a b c k e : Point}
variables [IsRightTriangle a b c] 
variables [IsHeight a c k b]
variables [IsAngleBisector a c k e]

-- Defining the proof problem: Prove CB = BE
theorem CB_equals_BE 
  (h1 : angle a c b = 90) 
  (h2 : is_height c k (line_segment a b))
  (h3 : is_angle_bisector c e (triangle a c k)) : 
  (distance c b) = (distance b e) := 
sorry

end CB_equals_BE_l667_667511


namespace log_base_5_of_fraction_l667_667378

theorem log_base_5_of_fraction (a b : ℕ) (h1 : b ≠ 0) (h2 : a = 5) (h3 : b = 2) : log 5 (1 / (a ^ b)) = -2 := by
  sorry

end log_base_5_of_fraction_l667_667378


namespace regular_polygon_distances_l667_667644

-- Define a regular polygon with 2n sides.
structure RegularPolygon (n : ℕ) :=
  (A : Fin (2 * n) → Point)
  (is_regular : ∀ (i j : Fin (2 * n)), dist (A i) (A j) = dist (A (i + 1)) (A (j + 1)))

noncomputable def point_A1 (p : RegularPolygon n) : Point := p.A ⟨0, by linarith⟩
noncomputable def point_A2 (p : RegularPolygon n) : Point := p.A ⟨1, by linarith⟩
noncomputable def point_A3 (p : RegularPolygon n) : Point := p.A ⟨2, by linarith⟩
noncomputable def point_An (p : RegularPolygon n) : Point := p.A ⟨n, by linarith⟩

theorem regular_polygon_distances (n : ℕ) (p : RegularPolygon n) :
  (1 / (dist (point_A1 p) (point_A2 p))^2) + (1 / (dist (point_A1 p) (point_An p))^2)
  = 4 / (dist (point_A1 p) (point_A3 p))^2 :=
by
  sorry

end regular_polygon_distances_l667_667644


namespace mary_income_more_than_tim_income_l667_667571

variables (J T M : ℝ)
variables (h1 : T = 0.60 * J) (h2 : M = 0.8999999999999999 * J)

theorem mary_income_more_than_tim_income : (M - T) / T * 100 = 50 :=
by
  sorry

end mary_income_more_than_tim_income_l667_667571


namespace twelve_year_olds_count_l667_667510

theorem twelve_year_olds_count (x y z w : ℕ) 
  (h1 : x + y + z + w = 23)
  (h2 : 10 * x + 11 * y + 12 * z + 13 * w = 253)
  (h3 : z = 3 * w / 2) : 
  z = 6 :=
by sorry

end twelve_year_olds_count_l667_667510


namespace ellipse_properties_l667_667439

noncomputable def ellipse_equation (e : ℝ) (b : ℝ) (vertex : ℝ × ℝ) : ℝ × ℝ :=
  let a := b / sqrt(1 - e^2) in
  (a, b)

theorem ellipse_properties (P : ℝ × ℝ) :
  let e := (sqrt 3) / 2 in
  let b := 1 in
  let (0, 1) = vertex in
  let (a, b) := ellipse_equation e b vertex in
  a = 2 ∧ b = 1 ∧ P ∈ { (x, y) | x^2 + y^2 = 16 } →
  ellipse C (x y : ℝ) : ℝ := x^2 / (2^2) + y^2 = 1 →
  minMN (C P : ℝ) : ℝ := (inf MN) = 5 / 4 :=
sorry

end ellipse_properties_l667_667439


namespace find_lowest_score_l667_667198

constant mean15 : ℝ := 90
constant mean13 : ℝ := 92
constant numScores : ℕ := 15
constant maxScore : ℝ := 110

theorem find_lowest_score : 
  let totalSum := mean15 * numScores
  let remainingScores := numScores - 2
  let subtractedSum := mean13 * remainingScores
  let highLowSum := totalSum - subtractedSum
  highLowSum - maxScore = 44 := 
by
  let totalSum := mean15 * numScores
  let remainingScores := numScores - 2
  let subtractedSum := mean13 * remainingScores
  let highLowSum := totalSum - subtractedSum
  show highLowSum - maxScore = 44 from sorry

end find_lowest_score_l667_667198


namespace Tn_gt_Sn_l667_667879

-- Definitions of the sequences and initial conditions
def a : ℕ → ℕ := λ n, 2 * n + 3
def b (n : ℕ) : ℕ := if n % 2 = 1 then a n - 6 else 2 * a n
def S (n : ℕ) : ℕ := (n * (2 * n + 8))/2
def T : ℕ → ℕ
| 0     := 0
| (n+1) := T n + b (n + 1)

-- Given initial conditions
axiom S_4_eq_32 : S 4 = 32
axiom T_3_eq_16 : T 3 = 16
axiom a_general : ∀ n : ℕ, a n = 2 * n + 3

-- Proof of the main theorem
theorem Tn_gt_Sn (n : ℕ) (h : n > 5) : T n > S n := by
  sorry

end Tn_gt_Sn_l667_667879


namespace op_identity_l667_667362

-- Define the operation ⊕ as given by the table
def op (x y : ℕ) : ℕ :=
  match (x, y) with
  | (1, 1) => 4
  | (1, 2) => 1
  | (1, 3) => 2
  | (1, 4) => 3
  | (2, 1) => 1
  | (2, 2) => 3
  | (2, 3) => 4
  | (2, 4) => 2
  | (3, 1) => 2
  | (3, 2) => 4
  | (3, 3) => 1
  | (3, 4) => 3
  | (4, 1) => 3
  | (4, 2) => 2
  | (4, 3) => 3
  | (4, 4) => 4
  | _ => 0  -- default case for completeness

-- State the theorem
theorem op_identity : op (op 4 1) (op 2 3) = 3 := by
  sorry

end op_identity_l667_667362


namespace y_intercept_of_line_l667_667271

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 :=
by
  sorry

end y_intercept_of_line_l667_667271


namespace initial_overs_played_l667_667992

theorem initial_overs_played 
    (target_runs : ℕ)
    (remaining_runs_rate : ℚ)
    (initial_runs_rate : ℚ)
    (remaining_overs : ℕ) 
    (initial_overs : ℕ) :
    target_runs = 275 →
    initial_runs_rate = 3.2 →
    remaining_runs_rate = 6.485714285714286 →
    remaining_overs = 35 →
    initial_overs = 15 :=
begin
  intros,
  sorry

end initial_overs_played_l667_667992


namespace T_gt_S_for_n_gt_5_l667_667922

-- Define the arithmetic sequence a_n
def a : ℕ → ℕ
| n => 2 * n + 3

-- Define the sequence b_n
def b : ℕ → ℕ
| n => if n % 2 = 1 then a n - 6 else 2 * (a n)

-- Sum of first n terms of a
def S (n : ℕ) : ℕ := (List.range n).sum (λ k => a (k + 1))

-- Sum of first n terms of b
def T (n : ℕ) : ℕ := (List.range n).sum (λ k => b (k + 1))

-- Given conditions in Lean
axiom S_4 : S 4 = 32
axiom T_3 : T 3 = 16

-- The theorem to prove
theorem T_gt_S_for_n_gt_5 (n : ℕ) (hn : n > 5) : T n > S n :=
  sorry

end T_gt_S_for_n_gt_5_l667_667922


namespace y_intercept_of_line_l667_667264

theorem y_intercept_of_line : ∃ y : ℝ, 2 * 0 - 3 * y = 6 ∧ y = -2 :=
by
  exists (-2)
  split
  . simp
  . rfl

end y_intercept_of_line_l667_667264


namespace tony_total_gas_expenses_l667_667245

def weekly_work_miles : ℕ := 50 * 5
def weekly_family_miles : ℕ := 15 * 2
def weekly_miles : ℤ := weekly_work_miles + weekly_family_miles
def mpg : ℤ := 25
def weekly_gallons : ℝ := weekly_miles / mpg
def gas_price : List ℝ := [2.00, 2.20, 1.90, 2.10]
def tank_capacity : ℤ := 10

def weekly_expense (price: ℝ) : ℝ := weekly_gallons * price
def total_expenses : ℝ := gas_price.map weekly_expense.sum

theorem tony_total_gas_expenses :
  total_expenses = 91.84 :=
by
  sorry

end tony_total_gas_expenses_l667_667245


namespace complement_union_l667_667041

open Set

variable U : Set ℕ := {1, 2, 3, 4}
variable A : Set ℕ := {1, 2}
variable B : Set ℕ := {2, 3}

theorem complement_union (hU : U = {1, 2, 3, 4}) (hA : A = {1, 2}) (hB : B = {2, 3}) :
  compl (A ∪ B) = {4} :=
by
  sorry

end complement_union_l667_667041


namespace tim_out_of_pocket_expense_l667_667239

noncomputable def total_cost (mri_cost doc_cost seen_fee : ℝ) := mri_cost + doc_cost + seen_fee
noncomputable def insurance_covered (total insurance_percentage : ℝ) := total * insurance_percentage
noncomputable def out_of_pocket (total insurance_covered : ℝ) := total - insurance_covered

theorem tim_out_of_pocket_expense :
  let mri_cost := 1200 in
  let doc_rate := 300 in
  let exam_duration_hr := 0.5 in
  let seen_fee := 150 in
  let insurance_percentage := 0.8 in
  let doc_cost := doc_rate * exam_duration_hr in
  let total := total_cost mri_cost doc_cost seen_fee in
  let covered := insurance_covered total insurance_percentage in
  let out_of_pocket_expense := out_of_pocket total covered in
  out_of_pocket_expense = 300 :=
by
  sorry

end tim_out_of_pocket_expense_l667_667239


namespace pressure_increases_when_block_submerged_l667_667292

-- Definitions and conditions
variables (P_0 : ℝ) (ρ : ℝ) (g : ℝ) (h_0 : ℝ) (h_1 : ℝ)
hypothesis (h1_gt_h0 : h_1 > h_0)

-- The proof goal
theorem pressure_increases_when_block_submerged (P_0 ρ g h_0 h_1 : ℝ) (h1_gt_h0 : h_1 > h_0) : 
  let P := P_0 + ρ * g * h_0 in
  let P_1 := P_0 + ρ * g * h_1 in
  P_1 > P := sorry

end pressure_increases_when_block_submerged_l667_667292


namespace belinda_percentage_flyers_l667_667746

/-- Conditions and definitions based on the problem given -/
def total_flyers : ℕ := 200
def flyers_ryan : ℕ := 42
def flyers_alyssa : ℕ := 67
def flyers_scott : ℕ := 51
def flyers_belinda : ℕ := total_flyers - (flyers_ryan + flyers_alyssa + flyers_scott)

/-- Theorem statement to prove -/
theorem belinda_percentage_flyers : 
  (flyers_belinda.to_rat / total_flyers.to_rat) * 100 = 20 :=
sorry

end belinda_percentage_flyers_l667_667746


namespace collinear_vectors_final_answer_l667_667080

noncomputable def sequence_a (n : ℕ) : ℕ :=
  3 * n ^ 2 - 9 * n + 6

def sequence_b (n : ℕ) : ℕ :=
  6 * n - 6

axiom a1 : sequence_a 1 = 0
axiom b1 : sequence_b 1 = 0

theorem collinear_vectors (n : ℕ) (hn : n > 0) : 
  let A_n := (n, sequence_a n) in
  let A_n_1 := (n + 1, sequence_a (n + 1)) in
  let B_n := (n, sequence_b n) in
  let C_n := (n - 1, 0) in
  (sequence_a (n + 1) - sequence_a n) = sequence_b n :=
by {
  sorry
}

theorem final_answer (n : ℕ) (hn : n > 0) : sequence_a n = 3 * n ^ 2 - 9 * n + 6 :=
by {
  sorry
}

end collinear_vectors_final_answer_l667_667080


namespace general_formula_sums_inequality_for_large_n_l667_667907

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def modified_seq (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = (finset.range n).sum a

variables {a b S T : ℕ → ℕ}

-- Given conditions
axiom h_arith_seq : arithmetic_seq a
axiom h_modified_seq : modified_seq a b
axiom h_sum_a : sum_seq a S
axiom h_sum_b : sum_seq b T
axiom h_S4 : S 4 = 32
axiom h_T3 : T 3 = 16

-- Proof Statements
theorem general_formula :
  ∀ n : ℕ, a n = 2 * n + 3 := sorry

theorem sums_inequality_for_large_n :
  ∀ n : ℕ, n > 5 → T n > S n := sorry

end general_formula_sums_inequality_for_large_n_l667_667907


namespace sum_distinct_prime_factors_of_420_l667_667281

theorem sum_distinct_prime_factors_of_420 : 
  ∑ (p : ℕ) in {2, 3, 5, 7}, p = 17 := 
by 
  sorry

end sum_distinct_prime_factors_of_420_l667_667281


namespace calculate_expression_l667_667750

theorem calculate_expression : 3 * ((-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4) = 81 := 
by sorry

end calculate_expression_l667_667750


namespace determine_k_l667_667768

theorem determine_k (k : ℕ) : 2^2004 - 2^2003 - 2^2002 + 2^2001 = k * 2^2001 → k = 3 :=
by
  intro h
  -- now we would proceed to prove it, but we'll skip proof here
  sorry

end determine_k_l667_667768


namespace rate_of_interest_is_8_l667_667676

-- Define the problem variables and conditions
variables (P₁ P₂ T₁ T₂ total_interest : ℝ) (R : ℝ)

-- Conditions based on the problem statement
def B_loan := (P₁ = 5000) ∧ (T₁ = 2)
def C_loan := (P₂ = 3000) ∧ (T₂ = 4)
def same_interest_rate := (total_interest = 1760)
def interest_formula := total_interest = (P₁ * T₁ * R / 100) + (P₂ * T₂ * R / 100)

-- The question to prove
theorem rate_of_interest_is_8 :
  B_loan P₁ P₂ T₁ T₂ ∧ C_loan P₁ P₂ T₁ T₂ ∧ same_interest_rate P₁ P₂ T₁ T₂ total_interest ∧ interest_formula P₁ P₂ T₁ T₂ R total_interest
  → R = 8 :=
by
  sorry

end rate_of_interest_is_8_l667_667676


namespace locus_of_point_P_l667_667721

theorem locus_of_point_P (
  P : ℝ × ℝ
) : 
  ((∀ P : ℝ × ℝ,
    let (x, y) := P in
    let PA_sq := (x - 1)^2 + (y - 3/2)^2 - (real.sqrt 5 / 2)^2 in
    let PB_sq := (x + 3/2)^2 + (y - 1)^2 - (real.sqrt 5 / 2)^2 in
    PA_sq = PB_sq) →
  (5 * P.1 + P.2 - 1 = 0)) := 
sorry

end locus_of_point_P_l667_667721


namespace neg_P_is_univ_l667_667448

noncomputable def P : Prop :=
  ∃ x0 : ℝ, x0^2 + 2 * x0 + 2 ≤ 0

theorem neg_P_is_univ :
  ¬ P ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by {
  sorry
}

end neg_P_is_univ_l667_667448


namespace percentage_alcohol_new_solution_l667_667672

theorem percentage_alcohol_new_solution
  (V₀ : ℝ) (P₀ : ℝ) (V₁ : ℝ)
  (hV₀ : V₀ = 50)
  (hP₀ : P₀ = 0.30)
  (hV₁ : V₁ = 30) :
  let V_new := V₀ + V₁ in
  let Volume_alcohol := V₀ * P₀ in
  let P_new := (Volume_alcohol / V_new) * 100 in
  P_new = 18.75 := by
  sorry

end percentage_alcohol_new_solution_l667_667672


namespace no_x2_term_imp_a_eq_half_l667_667503

theorem no_x2_term_imp_a_eq_half (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x^2 - 2 * a * x + a^2) = x^3 + (1 - 2 * a) * x^2 + ((a^2 - 2 * a) * x + a^2)) →
  (∀ c : ℝ, (1 - 2 * a) = 0) →
  a = 1 / 2 :=
by
  intros h_prod h_eq
  have h_eq' : 1 - 2 * a = 0 := h_eq 0
  linarith

end no_x2_term_imp_a_eq_half_l667_667503


namespace Jack_hands_in_l667_667540

def num_hundred_bills := 2
def num_fifty_bills := 1
def num_twenty_bills := 5
def num_ten_bills := 3
def num_five_bills := 7
def num_one_bills := 27
def to_leave_in_till := 300

def total_money_in_notes : Nat :=
  (num_hundred_bills * 100) +
  (num_fifty_bills * 50) +
  (num_twenty_bills * 20) +
  (num_ten_bills * 10) +
  (num_five_bills * 5) +
  (num_one_bills * 1)

def money_to_hand_in := total_money_in_notes - to_leave_in_till

theorem Jack_hands_in : money_to_hand_in = 142 := by
  sorry

end Jack_hands_in_l667_667540


namespace compute_105_times_95_l667_667358

theorem compute_105_times_95 : (105 * 95 = 9975) :=
by
  sorry

end compute_105_times_95_l667_667358


namespace sin_2alpha_given_cos_diff_l667_667866

theorem sin_2alpha_given_cos_diff :
  (cos ((Real.pi / 4) - α) = 3 / 5) → sin (2 * α) = -7 / 25 :=
by
  sorry

end sin_2alpha_given_cos_diff_l667_667866


namespace rhombus_diagonal_length_l667_667618

theorem rhombus_diagonal_length (d1 : ℝ) (Area : ℝ) (h_d1 : d1 = 30) (h_Area : Area = 180) : 
  ∃ (d2 : ℝ), d2 = 12 ∧ Area = (d1 * d2) / 2 :=
by
  use 12
  split
  · simp
  sorry

end rhombus_diagonal_length_l667_667618


namespace T_gt_S_for_n_gt_5_l667_667916

-- Define the arithmetic sequence a_n
def a : ℕ → ℕ
| n => 2 * n + 3

-- Define the sequence b_n
def b : ℕ → ℕ
| n => if n % 2 = 1 then a n - 6 else 2 * (a n)

-- Sum of first n terms of a
def S (n : ℕ) : ℕ := (List.range n).sum (λ k => a (k + 1))

-- Sum of first n terms of b
def T (n : ℕ) : ℕ := (List.range n).sum (λ k => b (k + 1))

-- Given conditions in Lean
axiom S_4 : S 4 = 32
axiom T_3 : T 3 = 16

-- The theorem to prove
theorem T_gt_S_for_n_gt_5 (n : ℕ) (hn : n > 5) : T n > S n :=
  sorry

end T_gt_S_for_n_gt_5_l667_667916


namespace hexagon_trapezoid_height_l667_667542

theorem hexagon_trapezoid_height (width height y : ℕ) (A1 : width = 9) (A2 : height = 16) 
    (B1 : ∀ A1 A2, ∃ width' height', width' = 12 ∧ height' = 12)
    (B2 : ∀ A1 A2, ∃ hexagon_height y', hexagon_height = height' ∧ y' = 12) : y = 12 :=
by
  sorry

end hexagon_trapezoid_height_l667_667542


namespace minimum_colors_for_3x3x3_tessellation_l667_667075

theorem minimum_colors_for_3x3x3_tessellation {
    -- Let C(n) be the minimum number of colors required to color an n x n x n grid
    -- such that no two adjacent cubes share the same color.

    -- Given that we have a 3x3x3 grid of cubes, we want to prove that C(3) = 3.
    n : ℕ,
    tessellation : fin 3 → fin 3 → fin 3 → ℕ → Prop,
    no_two_adjacent_same_color : ∀ (i j k : fin 3) (c : ℕ),
        (tessellation i j k c) → 
        ((i > 0 → tessellation (i-1) j k c → false) ∧ 
         (j > 0 → tessellation i (j-1) k c → false) ∧
         (k > 0 → tessellation i j (k-1) c → false) ∧
         (i < 2 → tessellation (i+1) j k c → false) ∧
         (j < 2 → tessellation i (j+1) k c → false) ∧
         (k < 2 → tessellation i j (k+1) c → false))
} :
∀ (i j k : fin 3), tessellation i j k c → C(3) = 3 :=
by
  sorry

end minimum_colors_for_3x3x3_tessellation_l667_667075


namespace intersection_A_B_complementA_intersection_B_l667_667012

open Set

variable {R : Type} [LinearOrder R]

def A : Set R := {x | -1 ≤ x ∧ x < 3}
def B : Set R := {x | 2 < x ∧ x ≤ 5}

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

theorem complementA_intersection_B :
  (compl A) ∩ B = {x | 3 ≤ x ∧ x ≤ 5} := by
  sorry

end intersection_A_B_complementA_intersection_B_l667_667012


namespace minimum_number_of_small_pipes_l667_667333

def volume_of_pipe (r h : ℝ) : ℝ :=
  π * r^2 * h

theorem minimum_number_of_small_pipes
  (h : ℝ)   -- length of the pipes
  (r_main : ℝ := 4)  -- radius of the main pipe (8 inches diameter)
  (r_small : ℝ := 1) -- radius of a smaller pipe (2 inches diameter)
  : 16 = (volume_of_pipe r_main h) / (volume_of_pipe r_small h) :=
by
  sorry

end minimum_number_of_small_pipes_l667_667333


namespace common_ratio_of_gp_l667_667681

theorem common_ratio_of_gp (a r : ℝ) (h1 : r ≠ 1) 
  (h2 : (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 343) : r = 6 := 
by
  sorry

end common_ratio_of_gp_l667_667681


namespace intersection_equiv_l667_667960

theorem intersection_equiv (A B : Set ℝ) :
  (A = { x : ℝ | -2 < x ∧ x < 2 }) →
  (B = { x : ℝ | 1 < x }) →
  (A ∩ B = { x : ℝ | 1 < x ∧ x < 2 }) :=
by
  intro hA
  intro hB
  rw [hA, hB]
  ext
  split
  sorry

end intersection_equiv_l667_667960


namespace least_three_digit_multiple_of_3_4_7_l667_667663

theorem least_three_digit_multiple_of_3_4_7 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 7 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 → m % 3 = 0 ∧ m % 4 = 0 ∧ m % 7 = 0 → n ≤ m :=
  sorry

end least_three_digit_multiple_of_3_4_7_l667_667663


namespace range_of_f_l667_667765

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctanh x

theorem range_of_f : Set.range f = Set.univ :=
  sorry

end range_of_f_l667_667765


namespace cot_30_deg_l667_667799

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 := by
  -- Define known values of trigonometric functions at 30 degrees
  have h1 : Real.tan (30 * Real.pi / 180) = 1 / Real.sqrt 3 := sorry
  have h2 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

  -- Prove using these known values
  calc
    Real.cot (30 * Real.pi / 180)
        = 1 / Real.tan (30 * Real.pi / 180) : by sorry
    ... = 1 / (1 / Real.sqrt 3) : by sorry
    ... = Real.sqrt 3 : by sorry

end cot_30_deg_l667_667799


namespace harry_morning_routine_l667_667397

variable (t1 t2 : ℕ)
variable (h1 : t1 = 15)
variable (h2 : t2 = 2 * t1)

theorem harry_morning_routine : t1 + t2 = 45 := by
  sorry

end harry_morning_routine_l667_667397


namespace cot_30_eq_sqrt_3_l667_667802

theorem cot_30_eq_sqrt_3 (h : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3) : 
  Real.cot (Real.pi / 6) = Real.sqrt 3 :=
by
  sorry

end cot_30_eq_sqrt_3_l667_667802


namespace part1_real_roots_part2_integer_roots_l667_667024

-- Equation structure
variable (k : ℝ) (k_ne_zero : k ≠ 0)

-- Define the quadratic equation kx^2 + (k-2)x - 2 = 0
noncomputable def quadratic_eq (x : ℝ) : ℝ :=
  k * x^2 + (k - 2) * x - 2

-- Part (1): Proving the equation always has real roots
theorem part1_real_roots : ∃ x₁ x₂ : ℝ, quadratic_eq k_ne_zero x₁ = 0 ∧ quadratic_eq k_ne_zero x₂ = 0 :=
by sorry

-- Part (2): Finding values of k for which the equation has two distinct integer roots
theorem part2_integer_roots (k_int : k ∈ Int) : k = 1 ∨ k = -1 ∨ k = 2 :=
by sorry

end part1_real_roots_part2_integer_roots_l667_667024


namespace general_formula_sums_inequality_for_large_n_l667_667900

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def modified_seq (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = (finset.range n).sum a

variables {a b S T : ℕ → ℕ}

-- Given conditions
axiom h_arith_seq : arithmetic_seq a
axiom h_modified_seq : modified_seq a b
axiom h_sum_a : sum_seq a S
axiom h_sum_b : sum_seq b T
axiom h_S4 : S 4 = 32
axiom h_T3 : T 3 = 16

-- Proof Statements
theorem general_formula :
  ∀ n : ℕ, a n = 2 * n + 3 := sorry

theorem sums_inequality_for_large_n :
  ∀ n : ℕ, n > 5 → T n > S n := sorry

end general_formula_sums_inequality_for_large_n_l667_667900


namespace chef_leftover_potatoes_l667_667317

-- Defining the conditions as variables
def fries_per_potato := 25
def total_potatoes := 15
def fries_needed := 200

-- Calculating the number of potatoes needed.
def potatoes_needed : ℕ :=
  fries_needed / fries_per_potato

-- Calculating the leftover potatoes.
def leftovers : ℕ :=
  total_potatoes - potatoes_needed

-- The theorem statement
theorem chef_leftover_potatoes :
  leftovers = 7 :=
by
  -- the actual proof is omitted.
  sorry

end chef_leftover_potatoes_l667_667317


namespace log_sum_example_exponent_log_example_l667_667753

theorem log_sum_example : log 10 4 + log 10 500 - log 10 2 = 3 := by
  sorry

theorem exponent_log_example : (27:ℝ)^(1/3) + (log 3 2) * (log 2 3) = 4 := by
  sorry

end log_sum_example_exponent_log_example_l667_667753


namespace perpendicular_lines_l667_667059

theorem perpendicular_lines (a : ℝ) 
  (h_perpendicular : ∀ x y, ax + (1 - a)y = 3 → (a - 1)x + (2a + 3)y = 2 → 
    -a / (1 - a) * -(a - 1) / (2a + 3) = -1) :
  a = 1 ∨ a = -3 :=
sorry

end perpendicular_lines_l667_667059


namespace divisible_by_6_implies_divisible_by_2_not_divisible_by_2_implies_not_divisible_by_6_equivalence_of_propositions_l667_667346

theorem divisible_by_6_implies_divisible_by_2 :
  ∀ (n : ℤ), (6 ∣ n) → (2 ∣ n) :=
by sorry

theorem not_divisible_by_2_implies_not_divisible_by_6 :
  ∀ (n : ℤ), ¬ (2 ∣ n) → ¬ (6 ∣ n) :=
by sorry

theorem equivalence_of_propositions :
  (∀ (n : ℤ), (6 ∣ n) → (2 ∣ n)) ↔ (∀ (n : ℤ), ¬ (2 ∣ n) → ¬ (6 ∣ n)) :=
by sorry


end divisible_by_6_implies_divisible_by_2_not_divisible_by_2_implies_not_divisible_by_6_equivalence_of_propositions_l667_667346


namespace complex_ratio_identity_l667_667127

variable {x y : ℂ}

theorem complex_ratio_identity :
  ( (x + y) / (x - y) - (x - y) / (x + y) = 3 ) →
  ( (x^4 + y^4) / (x^4 - y^4) - (x^4 - y^4) / (x^4 + y^4) = 49 / 600) :=
by
  sorry

end complex_ratio_identity_l667_667127


namespace coeff_x2_term_l667_667932

noncomputable def integral_value : ℝ := ∫ x in 0..2, (2 * x + 1)

theorem coeff_x2_term (a : ℝ) (h : a = integral_value) :
  let binom_expansion := (x - a / (2 * x)) ^ 6 in
  let coeff_x2 := 135 in
  True :=
sorry

end coeff_x2_term_l667_667932


namespace simplify_expression_l667_667608

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b ^ 2 + 2 * b) - 2 * b ^ 2 = 9 * b ^ 3 + 4 * b ^ 2 :=
by
  sorry

end simplify_expression_l667_667608


namespace smaller_square_area_percentage_is_zero_l667_667321

noncomputable def area_smaller_square_percentage (r : ℝ) : ℝ :=
  let side_length_larger_square := 2 * r
  let x := 0  -- Solution from the Pythagorean step
  let area_larger_square := side_length_larger_square ^ 2
  let area_smaller_square := x ^ 2
  100 * area_smaller_square / area_larger_square

theorem smaller_square_area_percentage_is_zero (r : ℝ) :
    area_smaller_square_percentage r = 0 :=
  sorry

end smaller_square_area_percentage_is_zero_l667_667321


namespace special_naturals_in_interval_l667_667840

noncomputable def find_special_naturals : Set ℕ := { n | (57121 + 35 * n).sqrt ∈ ℕ }

theorem special_naturals_in_interval :
  ∀ (n : ℕ), 1000 ≤ n ∧ n ≤ 2000 → n ∈ find_special_naturals ↔ n ∈ {1096, 1221, 1185, 1312, 1749, 1888, 1848, 1989} := by
  sorry

end special_naturals_in_interval_l667_667840


namespace geometric_sequence_product_geometric_sequence_sum_not_definitely_l667_667017

theorem geometric_sequence_product (a b : ℕ → ℝ) (r1 r2 : ℝ) 
  (ha : ∀ n, a (n+1) = r1 * a n)
  (hb : ∀ n, b (n+1) = r2 * b n) :
  ∃ r3, ∀ n, (a n * b n) = r3 * (a (n-1) * b (n-1)) :=
sorry

theorem geometric_sequence_sum_not_definitely (a b : ℕ → ℝ) (r1 r2 : ℝ) 
  (ha : ∀ n, a (n+1) = r1 * a n)
  (hb : ∀ n, b (n+1) = r2 * b n) :
  ¬ ∀ r3, ∃ N, ∀ n ≥ N, (a n + b n) = r3 * (a (n-1) + b (n-1)) :=
sorry

end geometric_sequence_product_geometric_sequence_sum_not_definitely_l667_667017


namespace negation_proposition_l667_667209

theorem negation_proposition (h : ¬ (∀ x : ℝ, exp x ≥ x + 1)) : ∃ x₀ : ℝ, exp x₀ < x₀ + 1 :=
sorry

end negation_proposition_l667_667209


namespace length_of_one_side_of_regular_octagon_l667_667501

-- Define the conditions of the problem
def is_regular_octagon (n : ℕ) (P : ℝ) (length_of_side : ℝ) : Prop :=
  n = 8 ∧ P = 72 ∧ length_of_side = P / n

-- State the theorem
theorem length_of_one_side_of_regular_octagon : is_regular_octagon 8 72 9 :=
by
  -- The proof is omitted; only the statement is required
  sorry

end length_of_one_side_of_regular_octagon_l667_667501


namespace coin_probability_l667_667638

theorem coin_probability (a r : ℝ) (h : r < a / 2) :
  let favorable_cells := 3
  let larger_cell_area := 9 * a^2
  let favorable_area_per_cell := (a - 2 * r)^2
  let favorable_area := favorable_cells * favorable_area_per_cell
  let probability := favorable_area / larger_cell_area
  probability = (a - 2 * r)^2 / (3 * a^2) :=
by
  sorry

end coin_probability_l667_667638


namespace min_value_expr_l667_667351

theorem min_value_expr (x : ℝ) (h : x > 1) : ∃ m, m = 5 ∧ ∀ y, y = x + 4 / (x - 1) → y ≥ m :=
by
  sorry

end min_value_expr_l667_667351


namespace log_base_5_of_reciprocal_of_25_l667_667386

theorem log_base_5_of_reciprocal_of_25 : log 5 (1 / 25) = -2 :=
by
-- Sorry this proof is omitted
sorry

end log_base_5_of_reciprocal_of_25_l667_667386


namespace T_gt_S_for_n_gt_5_l667_667919

-- Define the arithmetic sequence a_n
def a : ℕ → ℕ
| n => 2 * n + 3

-- Define the sequence b_n
def b : ℕ → ℕ
| n => if n % 2 = 1 then a n - 6 else 2 * (a n)

-- Sum of first n terms of a
def S (n : ℕ) : ℕ := (List.range n).sum (λ k => a (k + 1))

-- Sum of first n terms of b
def T (n : ℕ) : ℕ := (List.range n).sum (λ k => b (k + 1))

-- Given conditions in Lean
axiom S_4 : S 4 = 32
axiom T_3 : T 3 = 16

-- The theorem to prove
theorem T_gt_S_for_n_gt_5 (n : ℕ) (hn : n > 5) : T n > S n :=
  sorry

end T_gt_S_for_n_gt_5_l667_667919


namespace max_MA_plus_MB_l667_667862

-- Definitions of points and the ellipse
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨4, 0⟩
def B : Point := ⟨2, 2⟩
def ellipse (M : Point) : Prop := (M.x^2 / 25) + (M.y^2 / 9) = 1

-- Statement of the proof problem
theorem max_MA_plus_MB :
  ∀ M : Point, ellipse M → dist M A + dist M B ≤ 10 + 2 * real.sqrt 10 := 
  sorry

end max_MA_plus_MB_l667_667862


namespace log_identity_l667_667054

theorem log_identity {x : ℝ} (h : log 8 (5 * x) = 2) : log x 125 = 3 * (log 5 / (6 * log 2 - log 5)) :=
by
  sorry

end log_identity_l667_667054


namespace normal_peak_at_given_condition_l667_667940
open Real

noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  1 / (σ * sqrt (2 * π)) * exp (-((x - μ) ^ 2) / (2 * σ ^ 2))

theorem normal_peak_at_given_condition (μ σ : ℝ) (h : 0 < σ) :
  (∃ x, P(X < 0.3) = 0.5) → ∃ x, normal_pdf μ σ x = normal_pdf μ σ μ :=
sorry

end normal_peak_at_given_condition_l667_667940


namespace T_gt_S_for_n_gt_5_l667_667918

-- Define the arithmetic sequence a_n
def a : ℕ → ℕ
| n => 2 * n + 3

-- Define the sequence b_n
def b : ℕ → ℕ
| n => if n % 2 = 1 then a n - 6 else 2 * (a n)

-- Sum of first n terms of a
def S (n : ℕ) : ℕ := (List.range n).sum (λ k => a (k + 1))

-- Sum of first n terms of b
def T (n : ℕ) : ℕ := (List.range n).sum (λ k => b (k + 1))

-- Given conditions in Lean
axiom S_4 : S 4 = 32
axiom T_3 : T 3 = 16

-- The theorem to prove
theorem T_gt_S_for_n_gt_5 (n : ℕ) (hn : n > 5) : T n > S n :=
  sorry

end T_gt_S_for_n_gt_5_l667_667918


namespace infinite_solutions_exists_l667_667588

theorem infinite_solutions_exists :
  ∃ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x ≠ y) ∧ (y ≠ z) ∧ (z ≠ x) ∧
  (x - y + z = 1) ∧ ((x * y) % z = 0) ∧ ((y * z) % x = 0) ∧ ((z * x) % y = 0) ∧
  ∀ n : ℕ, ∃ x y z : ℕ, (n > 0) ∧ (x = n * (n^2 + n - 1)) ∧ (y = (n+1) * (n^2 + n - 1)) ∧ (z = n * (n+1)) := by
  sorry

end infinite_solutions_exists_l667_667588


namespace intersection_A_B_union_A_B_range_of_a_l667_667040

def A := {x : ℝ | 3^1 ≤ 3^x ∧ 3^x ≤ 3^3}
def B := {x : ℝ | log 2 x < 1}
def C (a : ℝ) := {x : ℝ | 1 < x ∧ x < a}

theorem intersection_A_B : A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by sorry

theorem union_A_B : A ∪ B = {x : ℝ | 0 < x ∧ x ≤ 3} :=
by sorry

theorem range_of_a (a : ℝ) (H : C a ⊆ A) : a ≤ 3 :=
by sorry

end intersection_A_B_union_A_B_range_of_a_l667_667040


namespace linear_decreasing_sequence_l667_667627

theorem linear_decreasing_sequence 
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (h_func1 : y1 = -3 * x1 + 1)
  (h_func2 : y2 = -3 * x2 + 1)
  (h_func3 : y3 = -3 * x3 + 1)
  (hx_seq : x1 < x2 ∧ x2 < x3)
  : y3 < y2 ∧ y2 < y1 := 
sorry

end linear_decreasing_sequence_l667_667627


namespace triangle_area_of_tangent_line_l667_667710

theorem triangle_area_of_tangent_line 
  (l : Line) 
  (h_tangent : l.is_tangent_to (circle 1 (0,0)))
  (h_sum_intercepts : l.sum_of_intercepts = √3) : 
  area_of_triangle l = 3/2 :=
sorry

end triangle_area_of_tangent_line_l667_667710


namespace possible_tens_digit_of_N_l667_667111

-- Definitions: Conditions from the problem
def P (n : ℕ) : ℕ :=
  let digits := (n % 10, n / 10)
  digits.fst * digits.snd

def S (n : ℕ) : ℕ :=
  let digits := (n % 10, n / 10)
  digits.fst + digits.snd

def isPrime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m > 1 ∧ m < n → ¬(n % m = 0)

-- Proof statement
theorem possible_tens_digit_of_N (N : ℕ) (h1 : N = P(N) + S(N)) (h2 : isPrime (S N)) (h3 : 10 ≤ N ∧ N < 100):
  ∃ a, a ∈ {2, 4, 8} ∧ 
    ∃ b, b = 9 ∧ 
      N = 10 * a + b :=
by
  sorry

end possible_tens_digit_of_N_l667_667111


namespace simplify_expression_l667_667605

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b ^ 2 + 2 * b) - 2 * b ^ 2 = 9 * b ^ 3 + 4 * b ^ 2 :=
by
  sorry

end simplify_expression_l667_667605


namespace angle_between_vectors_l667_667060

variables {G : Type*} [inner_product_space ℝ G]
variables (a b : G)

-- Condition 1: (a + 3b) is perpendicular to (7a - 5b)
def condition_1 : Prop := ⟪a + 3 • b, 7 • a - 5 • b⟫ = 0

-- Condition 2: (a - 4b) is perpendicular to (7a - 2b)
def condition_2 : Prop := ⟪a - 4 • b, 7 • a - 2 • b⟫ = 0

-- Statement: The angle between vectors a and b is π / 3
theorem angle_between_vectors (h₁ : condition_1 a b) (h₂ : condition_2 a b) : 
  real.angle a b = real.pi / 3 :=
sorry

end angle_between_vectors_l667_667060


namespace range_g_l667_667015

noncomputable def g (x : ℝ) (k : ℝ) (c : ℝ) : ℝ := x^k + c

theorem range_g (k : ℝ) (c : ℝ) (hk : k > 0) : 
  set.range (λ x, g x k c) ∩ set.Ici 1 = set.Ici (1 + c) :=
sorry

end range_g_l667_667015


namespace part1_and_part2_l667_667889

-- Define the arithmetic sequence {a_n}
def a (n : Nat) : Nat := 2 * n + 3

-- Define the sequence {b_n}
def b (n : Nat) : Nat :=
  if n % 2 = 0 then 4 * n + 6 else 2 * n - 3

-- Define the sum of the first n terms of a sequence
def summation (seq : Nat → Nat) (n : Nat) : Nat :=
  (List.range n).map seq |>.sum

-- Define S_n as the sum of the first n terms of {a_n}
def S (n : Nat) : Nat := summation a n

-- Define T_n as the sum of the first n terms of {b_n}
def T (n : Nat) : Nat := summation b n

-- Given conditions
axiom S4_eq_32 : S 4 = 32
axiom T3_eq_16 : T 3 = 16

-- Prove the general formula for {a_n} and that T_n > S_n for n > 5
theorem part1_and_part2 (n : Nat) (h : n > 5) : a n = 2 * n + 3 ∧ T n > S n :=
  by
  sorry

end part1_and_part2_l667_667889


namespace find_lowest_score_l667_667194

constant mean15 : ℝ := 90
constant mean13 : ℝ := 92
constant numScores : ℕ := 15
constant maxScore : ℝ := 110

theorem find_lowest_score : 
  let totalSum := mean15 * numScores
  let remainingScores := numScores - 2
  let subtractedSum := mean13 * remainingScores
  let highLowSum := totalSum - subtractedSum
  highLowSum - maxScore = 44 := 
by
  let totalSum := mean15 * numScores
  let remainingScores := numScores - 2
  let subtractedSum := mean13 * remainingScores
  let highLowSum := totalSum - subtractedSum
  show highLowSum - maxScore = 44 from sorry

end find_lowest_score_l667_667194


namespace three_digit_multiples_of_3_from_0_to_9_l667_667084

theorem three_digit_multiples_of_3_from_0_to_9 : 
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∃ num_sets : finset (set ℕ) :=
    {s ∈ num_sets | {a, b, c} ∈ num_sets ⇒ a ≠ b ≠ c}
  number_of_multiples_of_3 formed 
    (∀ a b c ∈ digits, a ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a + b + c % 3 = 0 ∧ a * 100 + b * 10 + c < 1000)
  in num_sets = 246 :=
begin
  sorry
end

end three_digit_multiples_of_3_from_0_to_9_l667_667084


namespace lee_charged_per_action_figure_l667_667103

theorem lee_charged_per_action_figure :
  ∀ (sneakers_cost savings action_figures leftovers price_per_fig),
    sneakers_cost = 90 →
    savings = 15 →
    action_figures = 10 →
    leftovers = 25 →
    price_per_fig = 10 →
    (savings + action_figures * price_per_fig) - sneakers_cost = leftovers → price_per_fig = 10 :=
by
  intros sneakers_cost savings action_figures leftovers price_per_fig
  intros h_sneakers_cost h_savings h_action_figures h_leftovers h_price_per_fig
  intros h_total
  sorry

end lee_charged_per_action_figure_l667_667103


namespace incorrect_statement_D_l667_667296

theorem incorrect_statement_D :
  (∀ (x y z : Triangle), incenter x = incenter y → incenter y = incenter z → incenter z = incenter x) ∧
  (∀ (x y z : Triangle), circumcenter x = circumcenter y → circumcenter y = circumcenter z → circumcenter z = circumcenter x) ∧
  (∀ (T : Triangle), is_isosceles T → has_angle T 60 → is_equilateral T) ∧
  ¬(∀ (P : Prop), ∃ (Q : Prop), P → Q) :=
sorry

end incorrect_statement_D_l667_667296


namespace tangent_line_at_1_2_l667_667409

theorem tangent_line_at_1_2 :
  ∃ (m b : ℝ), (∀ (x : ℝ), y = -x^3 + 3 * x^2 → ∀ (x₁ y₁ : ℝ), (x₁, y₁) = (1, 2) → 
  (y₁ - y = m * (x₁ - x) + b)) :=
begin
  use [3, -1], -- slope 3, y-intercept -1
  sorry
end

end tangent_line_at_1_2_l667_667409


namespace domain_of_f_l667_667621

-- Definition of the function
def f (x : ℝ) : ℝ := sqrt (x - 2) + 1 / (x - 3)

-- Proving the domain of the function
theorem domain_of_f : 
  {x : ℝ | x ≥ 2 ∧ x ≠ 3} = 
    {x : ℝ | ∃ y : ℝ, f y = sqrt (y - 2) + 1 / (y - 3)} := 
by
  sorry

end domain_of_f_l667_667621


namespace brendan_cuts_84_yards_in_week_with_lawnmower_l667_667353

-- Brendan cuts 8 yards per day
def yards_per_day : ℕ := 8

-- The lawnmower increases his efficiency by fifty percent
def efficiency_increase (yards : ℕ) : ℕ :=
  yards + (yards / 2)

-- Calculate total yards cut in 7 days with the lawnmower
def total_yards_in_week (days : ℕ) (daily_yards : ℕ) : ℕ :=
  days * daily_yards

-- Prove the total yards cut in 7 days with the lawnmower is 84
theorem brendan_cuts_84_yards_in_week_with_lawnmower :
  total_yards_in_week 7 (efficiency_increase yards_per_day) = 84 :=
by
  sorry

end brendan_cuts_84_yards_in_week_with_lawnmower_l667_667353


namespace product_divisible_by_six_l667_667634

theorem product_divisible_by_six (a : ℤ) : 6 ∣ a * (a + 1) * (2 * a + 1) := 
sorry

end product_divisible_by_six_l667_667634


namespace sin_angle_GAC_l667_667305

theorem sin_angle_GAC (a : ℝ) (a_pos : 0 < a) (A G C : ℂ) 
(AC_eq_a : dist A C = a) 
(AG_eq_sqrt3_a : dist A G = a * real.sqrt 3) :
  real.sin (angle A G C) = real.sqrt 3 / 3 := 
sorry

end sin_angle_GAC_l667_667305


namespace orthocenter_proof_l667_667995

noncomputable def is_acute_angled_triangle (ABC : Triangle) : Prop :=
  ∀ (A B C : Point), ∠ABC < π / 2 ∧ ∠BCA < π / 2 ∧ ∠CAB < π / 2

noncomputable def circumradius_eq (A B C H : Point) : Prop :=
  let r1 := circumradius (triangle A H B)
  let r2 := circumradius (triangle B H C)
  let r3 := circumradius (triangle C H A)
  r1 = r2 ∧ r2 = r3

noncomputable def is_orthocenter (A B C H : Point) : Prop :=
  H ∈ line (altitude A B C) ∧ H ∈ line (altitude B C A) ∧ H ∈ line (altitude C A B)

theorem orthocenter_proof {A B C H : Point} (h1 : is_acute_angled_triangle ⟨A, B, C⟩) 
(h2 : circumradius_eq A B C H) : 
is_orthocenter A B C H :=
sorry

end orthocenter_proof_l667_667995


namespace tim_out_of_pocket_expense_l667_667240

noncomputable def total_cost (mri_cost doc_cost seen_fee : ℝ) := mri_cost + doc_cost + seen_fee
noncomputable def insurance_covered (total insurance_percentage : ℝ) := total * insurance_percentage
noncomputable def out_of_pocket (total insurance_covered : ℝ) := total - insurance_covered

theorem tim_out_of_pocket_expense :
  let mri_cost := 1200 in
  let doc_rate := 300 in
  let exam_duration_hr := 0.5 in
  let seen_fee := 150 in
  let insurance_percentage := 0.8 in
  let doc_cost := doc_rate * exam_duration_hr in
  let total := total_cost mri_cost doc_cost seen_fee in
  let covered := insurance_covered total insurance_percentage in
  let out_of_pocket_expense := out_of_pocket total covered in
  out_of_pocket_expense = 300 :=
by
  sorry

end tim_out_of_pocket_expense_l667_667240


namespace rosie_circles_track_24_l667_667394

-- Definition of the problem conditions
def lou_distance := 3 -- Lou's total distance in miles
def track_length := 1 / 4 -- Length of the circular track in miles
def rosie_speed_factor := 2 -- Rosie runs at twice the speed of Lou

-- Define the number of times Rosie circles the track as a result
def rosie_circles_the_track : Nat :=
  let lou_circles := lou_distance / track_length
  let rosie_distance := lou_distance * rosie_speed_factor
  let rosie_circles := rosie_distance / track_length
  rosie_circles

-- The theorem stating that Rosie circles the track 24 times
theorem rosie_circles_track_24 : rosie_circles_the_track = 24 := by
  sorry

end rosie_circles_track_24_l667_667394


namespace find_lowest_score_l667_667193

noncomputable def lowest_score (total_scores : ℕ) (mean_scores : ℕ) (new_total_scores : ℕ) (new_mean_scores : ℕ) (highest_score : ℕ) : ℕ :=
  let total_sum        := mean_scores * total_scores
  let sum_remaining    := new_mean_scores * new_total_scores
  let removed_sum      := total_sum - sum_remaining
  let lowest_score     := removed_sum - highest_score
  lowest_score

theorem find_lowest_score :
  lowest_score 15 90 13 92 110 = 44 :=
by
  unfold lowest_score
  simp [Nat.mul_sub, Nat.sub_apply, Nat.add_comm]
  rfl

end find_lowest_score_l667_667193


namespace width_of_domain_of_g_l667_667556

-- Define the function h with its domain
def h (x : ℝ) : ℝ := sorry

noncomputable def g (x : ℝ) := h (x / 3)

theorem width_of_domain_of_g :
  ∀ (h : ℝ → ℝ), (∀ (x : ℝ), -10 ≤ x ∧ x ≤ 10 → (h x) = (h x)) → 
  (∃ a b : ℝ, a = -30 ∧ b = 30 ∧ b - a = 60)  := 
by 
  intros h domain_condition
  use [-30, 30]
  split
  sorry


end width_of_domain_of_g_l667_667556


namespace triangle_angle_identity_l667_667532

def triangle_angles_arithmetic_sequence (A B C : ℝ) : Prop :=
  A + C = 2 * B

def sum_of_triangle_angles (A B C : ℝ) : Prop :=
  A + B + C = 180

def angle_B_is_60 (B : ℝ) : Prop :=
  B = 60

theorem triangle_angle_identity (A B C a b c : ℝ)
  (h1 : triangle_angles_arithmetic_sequence A B C)
  (h2 : sum_of_triangle_angles A B C)
  (h3 : angle_B_is_60 B) : 
  1 / (a + b) + 1 / (b + c) = 3 / (a + b + c) :=
by 
  sorry

end triangle_angle_identity_l667_667532


namespace probability_king_then_ten_l667_667422

-- Define the conditions
def standard_deck_size : ℕ := 52
def num_kings : ℕ := 4
def num_tens : ℕ := 4

-- Define the event probabilities
def prob_first_card_king : ℚ := num_kings / standard_deck_size
def prob_second_card_ten (remaining_deck_size : ℕ) : ℚ := num_tens / remaining_deck_size

-- The theorem statement to be proved
theorem probability_king_then_ten : 
  prob_first_card_king * prob_second_card_ten (standard_deck_size - 1) = 4 / 663 :=
by
  sorry

end probability_king_then_ten_l667_667422


namespace problem_lemma_l667_667855

def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem problem_lemma : f (f (f (-1))) = Real.pi + 1 := 
  by
    sorry

end problem_lemma_l667_667855


namespace villages_population_equal_l667_667685

def population_x (initial_population rate_decrease : Int) (n : Int) := initial_population - rate_decrease * n
def population_y (initial_population rate_increase : Int) (n : Int) := initial_population + rate_increase * n

theorem villages_population_equal
    (initial_population_x : Int) (rate_decrease_x : Int)
    (initial_population_y : Int) (rate_increase_y : Int)
    (h₁ : initial_population_x = 76000) (h₂ : rate_decrease_x = 1200)
    (h₃ : initial_population_y = 42000) (h₄ : rate_increase_y = 800) :
    ∃ n : Int, population_x initial_population_x rate_decrease_x n = population_y initial_population_y rate_increase_y n ∧ n = 17 :=
by
    sorry

end villages_population_equal_l667_667685


namespace pq_proof_l667_667915

section ProofProblem

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, ∃ d, a (n + 1) = a n + d

def b_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
if n % 2 = 1 then a n - 6 else 2 * a n

def sum_first_n (seq : ℕ → ℤ) (n : ℕ) : ℤ :=
(n.sum (λ i, seq (i + 1)))

variables (a : ℕ → ℤ)
variables (S T : ℕ → ℤ)

axiom a_is_arithmetic : arithmetic_sequence a
axiom S_4_eq_32 : sum_first_n a 4 = 32
axiom T_3_eq_16 : sum_first_n (b_n a) 3 = 16

theorem pq_proof : (∀ n, a n = 2 * n + 3) ∧ (∀ n > 5, sum_first_n (b_n a) n > sum_first_n a n) :=
sorry

end ProofProblem

end pq_proof_l667_667915


namespace animal_population_decay_l667_667068

noncomputable def months (p_d : ℝ) (p_0 : ℝ) (p_s : ℝ) : ℝ :=
  (Real.log p_s / p_0) / Real.log (1 - p_d)

theorem animal_population_decay :
  let p_d := 1 / 10
  let p_0 := 600
  let p_s := 437.4
  months p_d p_0 p_s ≈ 3 :=
by
  sorry

end animal_population_decay_l667_667068


namespace percentage_deposited_l667_667155

-- Define the given conditions
def amountDeposited : ℝ := 3800
def monthlyIncome : ℝ := 17272.73

-- Prove the percentage of monthly income deposited
theorem percentage_deposited : (amountDeposited / monthlyIncome) * 100 ≈ 21.99 :=
by sorry

end percentage_deposited_l667_667155


namespace minimum_a_ge_neg_5_over_4_minimum_value_a_correct_l667_667058

noncomputable def minimum_value_a : ℝ :=
-5 / 4

theorem minimum_a_ge_neg_5_over_4 {a x : ℝ} (h : ∀ x ∈ set.Ioc 0 (1 / 2), x^2 + 2 * a * x + 1 ≥ 0) : 
a ≥ -5 / 4 :=
begin
  sorry
end

theorem minimum_value_a_correct : minimum_value_a = -5 / 4 :=
begin
  refl,
end

end minimum_a_ge_neg_5_over_4_minimum_value_a_correct_l667_667058


namespace simplify_trig_expression_l667_667158

variable (θ : ℝ)

theorem simplify_trig_expression :
  (sin (θ - 5 * π) * cot (π / 2 - θ) * cos (8 * π - θ)) / 
  (tan (3 * π - θ) * tan (θ - 3 * π / 2) * sin (-θ - 4 * π)) = 
  sin θ := 
by 
  sorry

end simplify_trig_expression_l667_667158


namespace least_multiple_remainder_zero_l667_667843

theorem least_multiple_remainder_zero :
  ∃ M : ℕ, (M ≡ 710 [MOD 1821]) ∧ (M % 23 = 0) ∧ (M = 3024) ∧ (M % 24 = 0) :=
begin
  use 3024,
  split,
  { -- M ≡ 710 [MOD 1821]
    exact nat.modeq.symm (nat.modeq.rfl.congr_left 710) },
  split,
  { -- M % 23 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_mul_left 23 132) },
  split,
  { -- M = 3024
    refl },
  { -- M % 24 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_mul_left 126 24) }
end

end least_multiple_remainder_zero_l667_667843


namespace sum_of_integers_between_60_and_460_ending_in_2_is_10280_l667_667411

-- We define the sequence.
def endsIn2Seq : List Int := List.range' 62 (452 + 1 - 62) 10  -- Generates [62, 72, ..., 452]

-- The sum of the sequence.
def sumEndsIn2Seq : Int := endsIn2Seq.sum

-- The theorem to prove the desired sum.
theorem sum_of_integers_between_60_and_460_ending_in_2_is_10280 :
  sumEndsIn2Seq = 10280 := by
  -- Proof is omitted
  sorry

end sum_of_integers_between_60_and_460_ending_in_2_is_10280_l667_667411


namespace find_a_sequence_formula_l667_667424

variable (f : ℝ → ℝ) (a : ℝ) (a_seq : ℕ → ℝ)
variable (h_f_def : ∀ x ≠ -a, f x = (a * x) / (a + x))
variable (h_f_2 : f 2 = 1) (h_seq_def : ∀ n : ℕ, a_seq (n+1) = f (a_seq n)) (h_a1 : a_seq 1 = 1)

theorem find_a : a = 2 :=
  sorry

theorem sequence_formula : ∀ n : ℕ, a_seq n = 2 / (n + 1) :=
  sorry

end find_a_sequence_formula_l667_667424


namespace cot_30_deg_l667_667818

-- Define the known values and identities
def tan_30_deg := 1 / Real.sqrt 3

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by 
  sorry

end cot_30_deg_l667_667818


namespace problem1_problem2_l667_667567

-- Definitions
def f (x a b : ℝ) : ℝ := x^3 - a * x^2 + 3 * x + b
def f' (x a : ℝ) : ℝ := 3 * x^2 - 2 * a * x + 3
def g (x : ℝ) : ℝ := -x^3 + 6 * x^2 - 9 * x + 3

-- Problem 1 statement: tangent line at (1, f(1)) is parallel to the x-axis implies a = 3
theorem problem1 (b : ℝ) : (f' 1 a = 0) → a = 3 :=
by
  intros h1

-- Problem 2 statement: for any x in [-1, 4], f(x) > f'(x) implies b > 19
theorem problem2 (a : ℝ) : (∀ x ∈ Icc (-1 : ℝ) 4, f x a b > f' x a) → b > 19 :=
by
  intros h2
  sorry

end problem1_problem2_l667_667567


namespace arithmetic_seq_a7_l667_667999

theorem arithmetic_seq_a7 (a : ℕ → ℕ) (d : ℕ)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 8)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  a 7 = 6 := by
  sorry

end arithmetic_seq_a7_l667_667999


namespace general_formula_T_greater_S_l667_667893

section arithmetic_sequence

variable {a : ℕ → ℤ} -- Define the arithmetic sequence {a_n}
variable {b : ℕ → ℤ} -- Define the sequence {b_n}
variable {S : ℕ → ℤ} -- Define the sum sequence {S_n}
variable {T : ℕ → ℤ} -- Define the sum sequence {T_n}
variable (a1 : ℤ) (d : ℤ) -- Variables for the first term and common difference 

-- Conditions
def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n, a n = a1 + n * d

def sequence_b (b a : ℕ → ℤ) : Prop :=
  ∀ n, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_sequence_S (S a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (2 * a1 + (n - 1) * d)) / 2

def sum_sequence_T (T b : ℕ → ℤ) : Prop :=
  ∀ n, T n = ∑ i in Finset.range n, b i

def conditions (a b S T : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  arithmetic_sequence a a1 d ∧ sequence_b b a ∧ sum_sequence_S S a ∧ sum_sequence_T T b ∧ S 4 = 32 ∧ T 3 = 16

-- Problem 1: Prove the general formula for {a_n}
theorem general_formula (a : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, a n = 2 * n + 3 :=
sorry

-- Problem 2: Prove that for n > 5, T_n > S_n
theorem T_greater_S (a b S T : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, n > 5 → T n > S n :=
sorry

end arithmetic_sequence

end general_formula_T_greater_S_l667_667893


namespace triangle_area_PQR_l667_667328

noncomputable def area_of_triangle : ℝ :=
let P := (1, 6)
let Q := (-5, 0)
let R := (-2, 0) in
1 / 2 * (Q.1 - R.1).abs * P.2

theorem triangle_area_PQR :
  let P := (1, 6) in
  let Q := (-5, 0) in
  let R := (-2, 0) in
  (1 / 2 * (Q.1 - R.1).abs * P.2) = 9 :=
by
  let P := (1, 6)
  let Q := (-5, 0)
  let R := (-2, 0)
  show (1 / 2 * (Q.1 - R.1).abs * P.2) = 9, from sorry

end triangle_area_PQR_l667_667328


namespace age_ratio_proof_l667_667675

-- Defining the ages of a, b, and c
def a := b + 2
def b := 10
def c := 27 - a - b

-- Definition to express the ratio of b's age to c's age
def ratio := b / c

-- Theorem stating the problem
theorem age_ratio_proof :
  ∃ ratio, ratio = 2 / 1 :=
by
  sorry

end age_ratio_proof_l667_667675


namespace cot_30_deg_l667_667797

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 := by
  -- Define known values of trigonometric functions at 30 degrees
  have h1 : Real.tan (30 * Real.pi / 180) = 1 / Real.sqrt 3 := sorry
  have h2 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

  -- Prove using these known values
  calc
    Real.cot (30 * Real.pi / 180)
        = 1 / Real.tan (30 * Real.pi / 180) : by sorry
    ... = 1 / (1 / Real.sqrt 3) : by sorry
    ... = Real.sqrt 3 : by sorry

end cot_30_deg_l667_667797


namespace lowest_score_of_15_scores_is_44_l667_667186

theorem lowest_score_of_15_scores_is_44
  (mean_15 : ℕ → ℕ)
  (mean_15 = 90)
  (mean_13 : ℕ → ℕ)
  (mean_13 = 92)
  (highest_score : ℕ)
  (highest_score = 110)
  (sum15 : mean_15 * 15 = 1350)
  (sum13 : mean_13 * 13 = 1196) :
  (lowest_score : ℕ) (lowest_score = 44) :=
by
  sorry

end lowest_score_of_15_scores_is_44_l667_667186


namespace zero_point_not_less_than_4_g_x_greater_than_l667_667031

-- Part (1)
theorem zero_point_not_less_than_4 (a x : ℝ) (h1 : 0 < x) (h2 : 1 ≤ a) : 
  let f := λ x, (1/2) * a * x^2 - (a^2 + 1) * x
  in (∃ x, f x = 0) → x ≥ 4 :=
sorry

-- Part (2)
theorem g_x_greater_than (a x : ℝ) (h3 : 1 ≤ x) (h4 : 1 ≤ a) : 
  let f := λ x, (1/2) * a * x^2 - (a^2 + 1) * x
  let g := λ x, f x + a * Real.log x
  g x > - (1/2) * a^3 - (Real.exp a / a) + Real.exp 1 - 2 * a :=
sorry

end zero_point_not_less_than_4_g_x_greater_than_l667_667031


namespace y_intercept_of_line_l667_667265

theorem y_intercept_of_line : ∃ y : ℝ, 2 * 0 - 3 * y = 6 ∧ y = -2 :=
by
  exists (-2)
  split
  . simp
  . rfl

end y_intercept_of_line_l667_667265


namespace arithmetic_sequence_middle_term_l667_667516

theorem arithmetic_sequence_middle_term:
  ∀ (a : ℕ → ℝ), (a 0 = 12) ∧ (a 6 = 54) ∧ (∀ n, a (n + 1) - a n = a 1 - a 0) → a 3 = 33 :=
by
  intros a h,
  let h1 := h.1,
  let h2 := h.2.1,
  let h3 := h.2.2,
  sorry

end arithmetic_sequence_middle_term_l667_667516


namespace general_formula_T_greater_S_l667_667894

section arithmetic_sequence

variable {a : ℕ → ℤ} -- Define the arithmetic sequence {a_n}
variable {b : ℕ → ℤ} -- Define the sequence {b_n}
variable {S : ℕ → ℤ} -- Define the sum sequence {S_n}
variable {T : ℕ → ℤ} -- Define the sum sequence {T_n}
variable (a1 : ℤ) (d : ℤ) -- Variables for the first term and common difference 

-- Conditions
def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n, a n = a1 + n * d

def sequence_b (b a : ℕ → ℤ) : Prop :=
  ∀ n, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_sequence_S (S a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (2 * a1 + (n - 1) * d)) / 2

def sum_sequence_T (T b : ℕ → ℤ) : Prop :=
  ∀ n, T n = ∑ i in Finset.range n, b i

def conditions (a b S T : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  arithmetic_sequence a a1 d ∧ sequence_b b a ∧ sum_sequence_S S a ∧ sum_sequence_T T b ∧ S 4 = 32 ∧ T 3 = 16

-- Problem 1: Prove the general formula for {a_n}
theorem general_formula (a : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, a n = 2 * n + 3 :=
sorry

-- Problem 2: Prove that for n > 5, T_n > S_n
theorem T_greater_S (a b S T : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, n > 5 → T n > S n :=
sorry

end arithmetic_sequence

end general_formula_T_greater_S_l667_667894


namespace find_lowest_score_l667_667191

noncomputable def lowest_score (total_scores : ℕ) (mean_scores : ℕ) (new_total_scores : ℕ) (new_mean_scores : ℕ) (highest_score : ℕ) : ℕ :=
  let total_sum        := mean_scores * total_scores
  let sum_remaining    := new_mean_scores * new_total_scores
  let removed_sum      := total_sum - sum_remaining
  let lowest_score     := removed_sum - highest_score
  lowest_score

theorem find_lowest_score :
  lowest_score 15 90 13 92 110 = 44 :=
by
  unfold lowest_score
  simp [Nat.mul_sub, Nat.sub_apply, Nat.add_comm]
  rfl

end find_lowest_score_l667_667191


namespace part1_and_part2_l667_667887

-- Define the arithmetic sequence {a_n}
def a (n : Nat) : Nat := 2 * n + 3

-- Define the sequence {b_n}
def b (n : Nat) : Nat :=
  if n % 2 = 0 then 4 * n + 6 else 2 * n - 3

-- Define the sum of the first n terms of a sequence
def summation (seq : Nat → Nat) (n : Nat) : Nat :=
  (List.range n).map seq |>.sum

-- Define S_n as the sum of the first n terms of {a_n}
def S (n : Nat) : Nat := summation a n

-- Define T_n as the sum of the first n terms of {b_n}
def T (n : Nat) : Nat := summation b n

-- Given conditions
axiom S4_eq_32 : S 4 = 32
axiom T3_eq_16 : T 3 = 16

-- Prove the general formula for {a_n} and that T_n > S_n for n > 5
theorem part1_and_part2 (n : Nat) (h : n > 5) : a n = 2 * n + 3 ∧ T n > S n :=
  by
  sorry

end part1_and_part2_l667_667887


namespace find_BL_l667_667248

theorem find_BL
  (K L A B : Point)
  (h1 : touches_internally K)
  (h2 : chord_tangent_to_smaller_circle AB L)
  (h3 : distance A L = 10)
  (h4 : ratio_of_segments (distance A K) (distance B K) = 2 / 5) :
  distance B L = 25 :=
begin
  sorry
end

end find_BL_l667_667248


namespace evaluate_function_at_neg_one_l667_667943

def f (x : ℝ) : ℝ := -2 * x^2 + 1

theorem evaluate_function_at_neg_one : f (-1) = -1 :=
by
  sorry

end evaluate_function_at_neg_one_l667_667943


namespace T_gt_S_for_n_gt_5_l667_667920

-- Define the arithmetic sequence a_n
def a : ℕ → ℕ
| n => 2 * n + 3

-- Define the sequence b_n
def b : ℕ → ℕ
| n => if n % 2 = 1 then a n - 6 else 2 * (a n)

-- Sum of first n terms of a
def S (n : ℕ) : ℕ := (List.range n).sum (λ k => a (k + 1))

-- Sum of first n terms of b
def T (n : ℕ) : ℕ := (List.range n).sum (λ k => b (k + 1))

-- Given conditions in Lean
axiom S_4 : S 4 = 32
axiom T_3 : T 3 = 16

-- The theorem to prove
theorem T_gt_S_for_n_gt_5 (n : ℕ) (hn : n > 5) : T n > S n :=
  sorry

end T_gt_S_for_n_gt_5_l667_667920


namespace triangle_solution_l667_667727

noncomputable def distance (p1 p2 : Real × Real) : Real :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def area_of_triangle (A B C : Real × Real) : Real :=
  let x1 := A.1; let y1 := A.2
  let x2 := B.1; let y2 := B.2
  let x3 := C.1; let y3 := C.2
  (Real.abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))) / 2

theorem triangle_solution :
  let A := (-2, 1)
  let B := (7, -3)
  let C := (4, 6)
  area_of_triangle A B C = 28.5 ∧ 
  (distance A B + distance B C + distance C A).toReal ≈ 29.2 :=
by
  -- Definitions
  let A := (-2, 1)
  let B := (7, -3)
  let C := (4, 6)

  -- Area calculation
  have area : area_of_triangle A B C = 28.5 := by sorry

  -- Perimeter calculation
  have perimeter : (distance A B + distance B C + distance C A).toReal ≈ 29.2 := by sorry

  -- Proof
  exact ⟨area, perimeter⟩

end triangle_solution_l667_667727


namespace jake_peaches_l667_667092

noncomputable def steven_peaches : ℕ := 15
noncomputable def jake_fewer : ℕ := 7

theorem jake_peaches : steven_peaches - jake_fewer = 8 :=
by
  sorry

end jake_peaches_l667_667092


namespace soccer_season_length_l667_667236

def total_games : ℕ := 27
def games_per_month : ℕ := 9
def months_in_season : ℕ := total_games / games_per_month

theorem soccer_season_length : months_in_season = 3 := by
  unfold months_in_season
  unfold total_games
  unfold games_per_month
  sorry

end soccer_season_length_l667_667236


namespace fraction_work_completed_by_third_group_l667_667074

def working_speeds (name : String) : ℚ :=
  match name with
  | "A"  => 1
  | "B"  => 2
  | "C"  => 1.5
  | "D"  => 2.5
  | "E"  => 3
  | "F"  => 2
  | "W1" => 1
  | "W2" => 1.5
  | "W3" => 1
  | "W4" => 1
  | "W5" => 0.5
  | "W6" => 1
  | "W7" => 1.5
  | "W8" => 1
  | _    => 0

def work_done_per_hour (workers : List String) : ℚ :=
  workers.map working_speeds |>.sum

def first_group : List String := ["A", "B", "C", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]
def second_group : List String := ["A", "B", "C", "D", "E", "F", "W1", "W2"]
def third_group : List String := ["A", "B", "C", "D", "E", "W1", "W2"]

theorem fraction_work_completed_by_third_group :
  (work_done_per_hour third_group) / (work_done_per_hour second_group) = 25 / 29 :=
by
  sorry

end fraction_work_completed_by_third_group_l667_667074


namespace pq_proof_l667_667909

section ProofProblem

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, ∃ d, a (n + 1) = a n + d

def b_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
if n % 2 = 1 then a n - 6 else 2 * a n

def sum_first_n (seq : ℕ → ℤ) (n : ℕ) : ℤ :=
(n.sum (λ i, seq (i + 1)))

variables (a : ℕ → ℤ)
variables (S T : ℕ → ℤ)

axiom a_is_arithmetic : arithmetic_sequence a
axiom S_4_eq_32 : sum_first_n a 4 = 32
axiom T_3_eq_16 : sum_first_n (b_n a) 3 = 16

theorem pq_proof : (∀ n, a n = 2 * n + 3) ∧ (∀ n > 5, sum_first_n (b_n a) n > sum_first_n a n) :=
sorry

end ProofProblem

end pq_proof_l667_667909


namespace general_formula_sums_inequality_for_large_n_l667_667906

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def modified_seq (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = (finset.range n).sum a

variables {a b S T : ℕ → ℕ}

-- Given conditions
axiom h_arith_seq : arithmetic_seq a
axiom h_modified_seq : modified_seq a b
axiom h_sum_a : sum_seq a S
axiom h_sum_b : sum_seq b T
axiom h_S4 : S 4 = 32
axiom h_T3 : T 3 = 16

-- Proof Statements
theorem general_formula :
  ∀ n : ℕ, a n = 2 * n + 3 := sorry

theorem sums_inequality_for_large_n :
  ∀ n : ℕ, n > 5 → T n > S n := sorry

end general_formula_sums_inequality_for_large_n_l667_667906


namespace general_formula_T_greater_S_l667_667897

section arithmetic_sequence

variable {a : ℕ → ℤ} -- Define the arithmetic sequence {a_n}
variable {b : ℕ → ℤ} -- Define the sequence {b_n}
variable {S : ℕ → ℤ} -- Define the sum sequence {S_n}
variable {T : ℕ → ℤ} -- Define the sum sequence {T_n}
variable (a1 : ℤ) (d : ℤ) -- Variables for the first term and common difference 

-- Conditions
def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n, a n = a1 + n * d

def sequence_b (b a : ℕ → ℤ) : Prop :=
  ∀ n, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_sequence_S (S a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (2 * a1 + (n - 1) * d)) / 2

def sum_sequence_T (T b : ℕ → ℤ) : Prop :=
  ∀ n, T n = ∑ i in Finset.range n, b i

def conditions (a b S T : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  arithmetic_sequence a a1 d ∧ sequence_b b a ∧ sum_sequence_S S a ∧ sum_sequence_T T b ∧ S 4 = 32 ∧ T 3 = 16

-- Problem 1: Prove the general formula for {a_n}
theorem general_formula (a : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, a n = 2 * n + 3 :=
sorry

-- Problem 2: Prove that for n > 5, T_n > S_n
theorem T_greater_S (a b S T : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, n > 5 → T n > S n :=
sorry

end arithmetic_sequence

end general_formula_T_greater_S_l667_667897


namespace relationship_y1_y2_l667_667035

theorem relationship_y1_y2 :
  ∀ (b y1 y2 : ℝ), 
  (∃ b y1 y2, y1 = -2023 * (-2) + b ∧ y2 = -2023 * (-1) + b) → y1 > y2 :=
by
  intro b y1 y2 h
  sorry

end relationship_y1_y2_l667_667035


namespace minimize_G_l667_667497

-- Define F(p, q)
def F (p q : ℝ) : ℝ := -4 * p * q + 5 * p * (1 - q) + 2 * (1 - p) * q - 3 * (1 - p) * (1 - q)

-- Define G(p) as the maximum of F(p, q) over 0 <= q <= 1
def G (p : ℝ) : ℝ := max (F p 0) (F p 1)

theorem minimize_G :
  ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ p = 5 / 14 ∧ (∀ p' : ℝ, 0 ≤ p' ∧ p' ≤ 1 → G p ≤ G p') :=
sorry

end minimize_G_l667_667497


namespace cosine_identity_l667_667580

variables (A B C P Q : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space P] [metric_space Q]
variables [add_group A] [add_group B] [add_group C] [add_group P] [add_group Q]

/-- Given a right triangle ABC with hypotenuse AB, and on AB an external square ABPQ is constructed. 
    Let α = ∠ACQ, β = ∠QCP, and γ = ∠PCB. We aim to prove that cos β = cos α cos γ. -/
theorem cosine_identity
  (h₁ : is_right_triangle A B C)
  (h₂ : is_square (square AB P Q))
  (alpha beta gamma : ℝ)
  (hα : α = ∠ACQ A B C P Q)
  (hβ : β = ∠QCP A B C P Q)
  (hγ : γ = ∠PCB A B C P Q) :
  cos β = (cos α) * (cos γ) :=
sorry

end cosine_identity_l667_667580


namespace ellipse_problem_l667_667459

noncomputable def ellipse_focus_distance_product (P F1 F2 : ℝ × ℝ) (a b : ℝ) : ℝ :=
  let PF1 := (P.1 - F1.1)^2 + (P.2 - F1.2)^2
  let PF2 := (P.1 - F2.1)^2 + (P.2 - F2.2)^2
  let dot_product := (P.1 - F1.1)*(P.1 - F2.1) + (P.2 - F1.2)*(P.2 - F2.2)
  if h : PF1 + PF2 = 8 ∧ |F1.1 - F2.1| = 4 ∧ dot_product = 9 then
    sqrt PF1 * sqrt PF2
  else 0

theorem ellipse_problem (x y : ℝ)
  (F1_x F1_y F2_x F2_y : ℝ)
  (h1 : (x^2 / 16) + (y^2 / 12) = 1)
  (h2 : F1_x = -2 ∧ F1_y = 0 ∧ F2_x = 2 ∧ F2_y = 0)  -- We need to specify exact positions of foci
  (h3 : x^2 + y^2 = 15 ∧ (x*4 = 9) )
  : ellipse_focus_distance_product (x, y) (F1_x, F1_y) (F2_x, F2_y) 4 3 = 15 :=
by
  sorry

end ellipse_problem_l667_667459


namespace luncheon_tables_needed_l667_667692

theorem luncheon_tables_needed (invited : ℕ) (no_show : ℕ) (people_per_table : ℕ) (people_attended : ℕ) (tables_needed : ℕ) :
  invited = 47 →
  no_show = 7 →
  people_per_table = 5 →
  people_attended = invited - no_show →
  tables_needed = people_attended / people_per_table →
  tables_needed = 8 := by {
  -- Proof here
  sorry
}

end luncheon_tables_needed_l667_667692


namespace line_passes_through_center_l667_667217

theorem line_passes_through_center : 
  let center := (-1, 0 : ℝ)
  let radius := 1
  let line (x y : ℝ) := x - y + 1 = 0
  let circle (x y : ℝ) := (x + 1)^2 + y^2 = 1
  let distance_point_line (x0 y0 : ℝ) (A B C : ℝ) := 
    |A*x0 + B*y0 + C| / Real.sqrt (A^2 + B^2)
  distance_point_line (-1) 0 1 (-1) 1 = 0 :=
  sorry

end line_passes_through_center_l667_667217


namespace modulus_of_z_l667_667935

-- Define the condition
def imaginary_unit (i : ℂ) : Prop :=
  i^2 = -1

def complex_number_condition (z : ℂ) (i : ℂ) : Prop :=
  z * (-1 + 2 * i) = 5 * i

-- State the theorem
theorem modulus_of_z {i z : ℂ} (h_i : imaginary_unit i) (h_z : complex_number_condition z i) :
  complex.abs z = real.sqrt 5 :=
sorry

end modulus_of_z_l667_667935


namespace intersecting_to_quadrilateral_l667_667261

-- Define the geometric solids
inductive GeometricSolid
| cone : GeometricSolid
| sphere : GeometricSolid
| cylinder : GeometricSolid

-- Define a function that checks if intersecting a given solid with a plane can produce a quadrilateral
def can_intersect_to_quadrilateral (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.cone => false
  | GeometricSolid.sphere => false
  | GeometricSolid.cylinder => true

-- State the theorem
theorem intersecting_to_quadrilateral (solid : GeometricSolid) :
  can_intersect_to_quadrilateral solid ↔ solid = GeometricSolid.cylinder :=
sorry

end intersecting_to_quadrilateral_l667_667261


namespace ratio_areas_is_one_fifty_l667_667162

section squares_and_ratios

variables {t : ℝ} (IJKL WXYZ : Type) [has_area IJKL] [has_area WXYZ]
variable (side_WXYZ : ℝ)
variable (WI IZ : ℝ)

-- Assume square areas and their properties
noncomputable def area_square_WXYZ := (10 * t) ^ 2
noncomputable def area_square_IJKL := (t * real.sqrt 2) ^ 2

-- Given the conditions
axiom length_WI_eq_9x_length_IZ_WZ (h : WI = 9 * IZ) (h2 : WI + IZ = 10 * t) : IZ = t ∧ WI = 9 * t

-- Prove the ratio of the areas
theorem ratio_areas_is_one_fifty (h : WI = 9 * IZ) (h2 : WI + IZ = 10 * t) : 
  area_square_IJKL / area_square_WXYZ = 1 / 50 :=
sorry

end squares_and_ratios

end ratio_areas_is_one_fifty_l667_667162


namespace tangent_f_g_at_one_max_k_f_gt_g_l667_667032

open Real

def f (x : ℝ) : ℝ := 5 + log x
def g (k x : ℝ) : ℝ := k * x / (x + 1)

theorem tangent_f_g_at_one :
  ∃ k : ℝ, 
    ∀ x, 
      (g k x = x + 4) ∧ 
      (diff (g k x)) = 1 ∧ 
      (x ≠ 2) → 
        k = 9 :=
sorry

theorem max_k_f_gt_g :
  ∀ (k : ℕ), 
    (∀ x > 1, f x > g k x) → 
      k ≤ 7 :=
sorry

end tangent_f_g_at_one_max_k_f_gt_g_l667_667032


namespace find_a_l667_667966

theorem find_a
  (a : ℝ)
  (l1 : ∀ x y : ℝ, ax + 2 * y + 6 = 0)
  (l2 : ∀ x y : ℝ, x + y - 4 = 0)
  (l3 : ∀ x y : ℝ, 2 * x - y + 1 = 0)
  (h_intersect : ∃ (x y : ℝ), l1 x y ∧ l2 x y ∧ l3 x y) :
  a = -12 :=
by
  sorry

end find_a_l667_667966


namespace problem_l667_667858

noncomputable def f (x : ℝ) : ℝ := |sin x|

theorem problem (f : ℝ → ℝ) 
  (periodic : ∀ x, f x = f (x + π)) 
  (range : ∀ y, 0 ≤ y ∧ y ≤ 1 → ∃ x, f x = y)
  (even : ∀ x, f x = f (-x)) : 
  f = λ x, |sin x| :=
by sorry

end problem_l667_667858


namespace locus_of_points_M_l667_667435

def semicircle {α : Type*} [metric_space α] (O : α) (R : ℝ) : set α :=
  {p | dist p O = R ∧ p.2 ≥ O.2}

def point_on_extension_diameter (O X : α) (d : ℝ) : Prop :=
  ∃ P, dist O P = d ∧ (O.1 ≤ X.1 ∧ X.1 ≤ P.1 ∨ P.1 ≤ X.1 ∧ X.1 ≤ O.1)

def tangent_ray (X O K : α) (radius : ℝ) : Prop :=
  dist O K = radius ∧ dist X K = ∞

def segment_equal (X O M : α) : Prop :=
  dist X M = dist X O

theorem locus_of_points_M
  {O X K M : α} {R : ℝ} [metric_space α]
  (h1 : semicircle O R)
  (h2 : point_on_extension_diameter O X R)
  (h3 : tangent_ray X O K R)
  (h4 : segment_equal X O M) :
  ∃ L : set α, L = {p | p.2 = O.2 + R ∧ p.1 ∉ {A.1, B.1}} :=
sorry

end locus_of_points_M_l667_667435


namespace solution_set_inequality_l667_667847

noncomputable def cube_root := real.cbrt

theorem solution_set_inequality (x : ℝ) :
  (-2 / 3 ≤ x ∧ x < 0) ↔ (3 * x ^ 2 / (1 - cube_root (3 * x + 1)) ^ 2 ≤ x + 2 + cube_root (3 * x + 1)) ∧ x ≠ 0 :=
by
  sorry

end solution_set_inequality_l667_667847


namespace set_union_proof_l667_667963

  open Set

  def M : Set ℕ := {0, 1, 3}
  def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 3 * a}

  theorem set_union_proof : M ∪ N = {0, 1, 3, 9} :=
  by
    sorry
  
end set_union_proof_l667_667963


namespace add_base3_numbers_l667_667730

theorem add_base3_numbers : 
  (2 + 1 * 3) + (0 + 2 * 3 + 1 * 3^2) + 
  (1 + 2 * 3 + 0 * 3^2 + 2 * 3^3) + (2 + 0 * 3 + 1 * 3^2 + 2 * 3^3)
  = 2 + 2 * 3 + 2 * 3^2 + 2 * 3^3 := 
by sorry

end add_base3_numbers_l667_667730


namespace find_m_l667_667480

noncomputable theory

-- Define the slope of a line given by Ax + By + C = 0 as -A / B.
def slope (A B : ℝ) : ℝ := -A / B

-- Define the lines l1 and l2
def l1 (x y : ℝ) (m : ℝ) : Prop := x + m * y + 7 = 0
def l2 (x y : ℝ) (m : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

-- State the theorem that given l1 and l2 are parallel, 
-- find the value of m that makes this true
theorem find_m (m : ℝ) (h_parallel : ∀ (x y : ℝ), l1 x y m → l2 x y m → (slope 1 m = slope (m-2) 3)) :
  m = -1 ∨ m = 3 :=
sorry

end find_m_l667_667480


namespace monotonic_intervals_l667_667028

def f (x : Real) (φ : Real) : Real := Real.cos (2 * x + φ)

def g (x : Real) : Real := Real.cos ((2 / 3) * x - 5 * Real.pi / 12)

theorem monotonic_intervals (φ : Real) (h1 : -Real.pi / 2 < φ) (h2 : φ < 0)
    (symmetry_axis : ∀ x : Real, f x φ = f (Real.pi / 8 - x) φ) :
    φ = -Real.pi / 4 ∧ 
    (∀ k : Int, 
          [((5 * Real.pi) / 8 + 3 * k * Real.pi), 
           (17 * Real.pi) / 8 + 3 * k * Real.pi] ⊆ 
           {x : Real | Deriv.g x < 0}) ∧ 
    ([Real.pi, 2 * Real.pi] ⊆ {x : Real | Deriv.g x < 0}) ∧
    ([-2 * Real.pi, -Real.pi] ⊆ {x : Real | Deriv.g x < 0}) ∧ 
    ([7 * Real.pi / 2, 9 * Real.pi / 2] ⊆ {x : Real | Deriv.g x < 0}) ∧
    ([-9 * Real.pi / 2, -4 * Real.pi] ⊆ {x : Real | Deriv.g x < 0}) :=
  sorry

end monotonic_intervals_l667_667028


namespace sqrt_1_0201_l667_667977

theorem sqrt_1_0201 (h : Real.sqrt 102.01 = 10.1) : 
    Set.eq (Set.insert (-1.01) (Set.singleton 1.01)) (Set.insert (- Real.sqrt 1.0201) (Set.singleton (Real.sqrt 1.0201))) := 
by
  sorry

end sqrt_1_0201_l667_667977


namespace log_base_5_of_fraction_l667_667376

theorem log_base_5_of_fraction (a b : ℕ) (h1 : b ≠ 0) (h2 : a = 5) (h3 : b = 2) : log 5 (1 / (a ^ b)) = -2 := by
  sorry

end log_base_5_of_fraction_l667_667376


namespace max_abs_value_l667_667849

theorem max_abs_value (x y : ℝ) (hx : |x - 1| ≤ 2) (hy : |y - 1| ≤ 2) : |x - 2 * y + 1| ≤ 6 :=
sorry

end max_abs_value_l667_667849


namespace distance_between_foci_after_rotation_l667_667205

noncomputable def hyperbola : set (ℝ × ℝ) := {p | p.1 * p.2 = 1}

def rotate_45 (p : ℝ × ℝ) : ℝ × ℝ :=
((p.1 + p.2) / Real.sqrt 2, (p.2 - p.1) / Real.sqrt 2)

theorem distance_between_foci_after_rotation :
  let rotated_hyperbola : set (ℝ × ℝ) := rotate_45 '' hyperbola
  ∃ f1 f2 : ℝ × ℝ, f1 ∈ rotated_hyperbola ∧ f2 ∈ rotated_hyperbola ∧
    Real.dist f1 f2 = 4 := 
sorry

end distance_between_foci_after_rotation_l667_667205


namespace smallest_abs_value_l667_667438

noncomputable def a_n : ℕ → ℝ := sorry -- Definition of the arithmetic sequence

def S (n : ℕ) : ℝ := ∑ i in range (n + 1), a_n i -- Definition of the sum of the first n terms

theorem smallest_abs_value :
  S 13 < 0 ∧ S 12 > 0 → (abs (a_n 7) < abs (a_n 5) ∧ abs (a_n 7) < abs (a_n 6) ∧ abs (a_n 7) < abs (a_n 8)) :=
by
  intros h
  sorry

end smallest_abs_value_l667_667438


namespace part1_and_part2_l667_667885

-- Define the arithmetic sequence {a_n}
def a (n : Nat) : Nat := 2 * n + 3

-- Define the sequence {b_n}
def b (n : Nat) : Nat :=
  if n % 2 = 0 then 4 * n + 6 else 2 * n - 3

-- Define the sum of the first n terms of a sequence
def summation (seq : Nat → Nat) (n : Nat) : Nat :=
  (List.range n).map seq |>.sum

-- Define S_n as the sum of the first n terms of {a_n}
def S (n : Nat) : Nat := summation a n

-- Define T_n as the sum of the first n terms of {b_n}
def T (n : Nat) : Nat := summation b n

-- Given conditions
axiom S4_eq_32 : S 4 = 32
axiom T3_eq_16 : T 3 = 16

-- Prove the general formula for {a_n} and that T_n > S_n for n > 5
theorem part1_and_part2 (n : Nat) (h : n > 5) : a n = 2 * n + 3 ∧ T n > S n :=
  by
  sorry

end part1_and_part2_l667_667885


namespace probability_of_two_hearts_and_three_diff_suits_l667_667169

def prob_two_hearts_and_three_diff_suits (n : ℕ) : ℚ :=
  if n = 5 then 135 / 1024 else 0

theorem probability_of_two_hearts_and_three_diff_suits :
  prob_two_hearts_and_three_diff_suits 5 = 135 / 1024 :=
by
  sorry

end probability_of_two_hearts_and_three_diff_suits_l667_667169


namespace packs_per_box_correct_l667_667137

-- Define the problem conditions
def weekly_boxes : ℕ := 30
def diapers_per_pack : ℕ := 160
def price_per_diaper : ℝ := 5
def total_earnings : ℝ := 960000

-- Define the total number of diapers sold equation
def total_diapers : ℕ := (total_earnings / price_per_diaper).toInt

-- Define the total number of packs sold equation
def total_packs : ℕ := total_diapers / diapers_per_pack

-- Define the number of packs per box equation
def packs_per_box : ℕ := total_packs / weekly_boxes

-- The theorem we need to prove
theorem packs_per_box_correct : packs_per_box = 40 := by
  sorry

end packs_per_box_correct_l667_667137


namespace G_at_16_l667_667109

noncomputable def G : ℝ → ℝ := sorry

-- Condition 1: G is a polynomial, implicitly stated
-- Condition 2: Given G(8) = 21
axiom G_at_8 : G 8 = 21

-- Condition 3: Given that
axiom G_fraction_condition : ∀ (x : ℝ), 
  (x^2 + 6*x + 8) ≠ 0 ∧ ((x+4)*(x+2)) ≠ 0 → 
  (G (2*x) / G (x+4) = 4 - (16*x + 32) / (x^2 + 6*x + 8))

-- The problem: Prove G(16) = 90
theorem G_at_16 : G 16 = 90 := 
sorry

end G_at_16_l667_667109


namespace smallest_perimeter_l667_667360

noncomputable def triangle_perimeter (AC CD : ℕ) : ℝ := 
  let BD := real.sqrt 64
  let DE := real.sqrt 16
  let AD := AC - CD
  let AB := real.sqrt (AD^2 + BD^2)
  let BC := 2 * real.sqrt ((DE + BD) * (DE - BD + 4))
  in AB + BC + AC

theorem smallest_perimeter :
  ∃ (AC CD : ℕ), 
    AC > 0 ∧ 
    CD > 0 ∧ 
    triangle_perimeter AC CD = 20 + 4 * real.sqrt 5 :=
begin
  sorry
end

end smallest_perimeter_l667_667360


namespace num_stripes_on_us_flag_l667_667171

-- Definitions based on conditions in the problem
def num_stars : ℕ := 50

def num_circles : ℕ := (num_stars / 2) - 3

def num_squares (S : ℕ) : ℕ := 2 * S + 6

def total_shapes (num_squares : ℕ) : ℕ := num_circles + num_squares

-- The theorem stating the number of stripes
theorem num_stripes_on_us_flag (S : ℕ) (h1 : num_circles = 22) (h2 : total_shapes (num_squares S) = 54) : S = 13 := by
  sorry

end num_stripes_on_us_flag_l667_667171


namespace inverse_function_correct_l667_667206

noncomputable def f (x : ℝ) : ℝ := 2 ^ x - 1

noncomputable def g (y : ℝ) : ℝ := Real.log2 (y + 1)

theorem inverse_function_correct (y : ℝ) (x : ℝ) (h1 : x > -1) (h2 : y > -1) :
  f (g y) = y ∧ g (f x) = x := by
  sorry

end inverse_function_correct_l667_667206


namespace y_days_do_work_l667_667686

theorem y_days_do_work (d : ℝ) (h : (1 / 30) + (1 / d) = 1 / 18) : d = 45 := 
by
  sorry

end y_days_do_work_l667_667686


namespace y_ordering_l667_667447

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 3

-- Define the points and their y-values
def y_val1 : ℝ := quadratic_function (-3)
def y_val2 : ℝ := quadratic_function 1
def y_val3 : ℝ := quadratic_function (-1 / 2)

theorem y_ordering :
  y_val2 < y_val3 ∧ y_val3 < y_val1 :=
by
  sorry

end y_ordering_l667_667447


namespace caps_second_week_l667_667310

theorem caps_second_week (x : ℕ) (h1 : 320 + x + 300 = 620 + x) (h2 : (620 + x) / 3 * 4 = 1360) : x = 400 :=
by
  -- Using Lean to define the known condition
  have : 4 * (620 + x) = 4080 := by
    rw [h2]
    norm_num

  -- Simplify the equation to isolate x
  have : 4x = 1600 := by
    linarith

  -- Solve for x
  have : x = 400 := by
    linarith

  assumption

end caps_second_week_l667_667310


namespace odd_pow_sum_divisible_l667_667253

theorem odd_pow_sum_divisible (x y : ℤ) (k : ℕ) (h : k > 0) :
  x^(2*k-1) + y^(2*k-1) % (x + y) = 0 :=
begin
  sorry
end

end odd_pow_sum_divisible_l667_667253


namespace T_gt_S_for_n_gt_5_l667_667923

-- Define the arithmetic sequence a_n
def a : ℕ → ℕ
| n => 2 * n + 3

-- Define the sequence b_n
def b : ℕ → ℕ
| n => if n % 2 = 1 then a n - 6 else 2 * (a n)

-- Sum of first n terms of a
def S (n : ℕ) : ℕ := (List.range n).sum (λ k => a (k + 1))

-- Sum of first n terms of b
def T (n : ℕ) : ℕ := (List.range n).sum (λ k => b (k + 1))

-- Given conditions in Lean
axiom S_4 : S 4 = 32
axiom T_3 : T 3 = 16

-- The theorem to prove
theorem T_gt_S_for_n_gt_5 (n : ℕ) (hn : n > 5) : T n > S n :=
  sorry

end T_gt_S_for_n_gt_5_l667_667923


namespace largest_possible_sum_l667_667414

theorem largest_possible_sum :
  let a := 12
  let b := 6
  let c := 6
  let d := 12
  a + b = c + d ∧ a + b + 15 = 33 :=
by
  have h1 : 12 + 6 = 6 + 12 := by norm_num
  have h2 : 12 + 6 + 15 = 33 := by norm_num
  exact ⟨h1, h2⟩

end largest_possible_sum_l667_667414


namespace log_base_5_of_1_div_25_eq_neg_2_l667_667382

theorem log_base_5_of_1_div_25_eq_neg_2 :
  log 5 (1/25) = -2 :=
by sorry

end log_base_5_of_1_div_25_eq_neg_2_l667_667382


namespace cot_30_eq_sqrt3_l667_667778

theorem cot_30_eq_sqrt3 : Real.cot (π / 6) = Real.sqrt 3 := by
  let tan_30 := 1 / Real.sqrt 3
  have tan_30_eq : Real.tan (π / 6) = tan_30 := sorry
  have cot_def : ∀ x, Real.cot x = 1 / Real.tan x := sorry
  rw [cot_def (π / 6), tan_30_eq]
  simp
  sorry

end cot_30_eq_sqrt3_l667_667778


namespace tangent_of_perpendicular_line_l667_667460

/-- Proof that tan(pi + alpha) = 2 given certain conditions --/
theorem tangent_of_perpendicular_line (α : ℝ) (λ : ℝ) (l k2 : ℝ) 
  (slope_eq : k2 = -1 / 2)
  (perpendicular_condition : l = 2)
  (tanα : Real.tan α = l)
  : Real.tan (Real.pi + α) = 2 :=
sorry

end tangent_of_perpendicular_line_l667_667460


namespace sqrt_inequality_l667_667848

theorem sqrt_inequality (a : fin 6 → ℝ) (h : ∀ i, 0 < a i) :
  (∑ i, (a i / (a ((i + 1) % 6) + a ((i + 2) % 6) + a ((i + 3) % 6))) ^ (1/4)) >= 2 :=
by sorry

end sqrt_inequality_l667_667848


namespace area_of_right_triangle_l667_667629

theorem area_of_right_triangle (hypotenuse : ℝ) (angle : ℝ) (h_hypotenuse : hypotenuse = 10) (h_angle : angle = 45) :
  (1 / 2) * (5 * Real.sqrt 2) * (5 * Real.sqrt 2) = 25 :=
by
  -- Proof goes here
  sorry

end area_of_right_triangle_l667_667629


namespace minimum_b_value_l667_667637

noncomputable def f : ℝ → ℝ := λ x, if x ≥ 0 then 2 * x - x^2 else -(2 * (-x) - (-x)^2)

theorem minimum_b_value :
  ∃ (a b : ℝ), b = -1 ∧ (∀ x, x ∈ Icc a b → f(x) = 2 * x - x^2 → x ∈ Icc (1/b) (1/a)) :=
sorry

end minimum_b_value_l667_667637


namespace min_value_of_quadratic_l667_667667

theorem min_value_of_quadratic :
  ∀ x : ℝ, let y := 5 * x^2 - 10 * x + 14 in ∃ m : ℝ, (∀ x : ℝ, y ≥ m) ∧ (∃ x : ℝ, y = m) :=
by
  sorry

end min_value_of_quadratic_l667_667667


namespace solve_for_x_l667_667979

theorem solve_for_x (x : ℝ) (h : sqrt (3 / x + 3) = 5 / 3) : x = -27 / 2 :=
sorry

end solve_for_x_l667_667979


namespace distance_to_asymptote1_l667_667443

-- Define the hyperbola and point P
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1
def point_P := (2 : ℝ, 0 : ℝ)

-- Define the asymptotes equations
def asymptote1 (x y : ℝ) : Prop := 4 * x - 3 * y = 0
def asymptote2 (x y : ℝ) : Prop := 4 * x + 3 * y = 0

-- Distance formula between point and line
def distance_point_line (x1 y1 a b c : ℝ) : ℝ :=
  (abs (a * x1 + b * y1 + c)) / (real.sqrt (a^2 + b^2))

-- Prove the distance between point P and asymptote1 is 8/5
theorem distance_to_asymptote1 : distance_point_line (2 : ℝ) (0 : ℝ) (4 : ℝ) (-3 : ℝ) (0 : ℝ) = 8 / 5 := by
  -- Sorry is used to skip the proof steps
  sorry

end distance_to_asymptote1_l667_667443


namespace max_value_of_a3_a18_l667_667936

theorem max_value_of_a3_a18 (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) 
(h_sum : ∑ i in range 20, a (i + 1) = 100) : 
  (∃ a_3 a_18 : ℝ, a_3 = a 3 ∧ a_18 = a 18 ∧ a_3 * a_18 = 25) :=
sorry

end max_value_of_a3_a18_l667_667936


namespace worker_hourly_rate_l667_667729

theorem worker_hourly_rate (x : ℝ) (h1 : 8 * 0.90 = 7.20) (h2 : 42 * x + 7.20 = 32.40) : x = 0.60 :=
by
  sorry

end worker_hourly_rate_l667_667729


namespace reading_speed_increase_factor_l667_667371

-- Define Tom's normal reading speed as a constant rate
def tom_normal_speed := 12 -- pages per hour

-- Define the time period
def hours := 2 -- hours

-- Define the number of pages read in the given time period
def pages_read := 72 -- pages

-- Calculate the expected pages read at normal speed in the given time
def expected_pages := tom_normal_speed * hours -- should be 24 pages

-- Define the calculated factor by which the reading speed has increased
def expected_factor := pages_read / expected_pages -- should be 3

-- Prove that the factor is indeed 3
theorem reading_speed_increase_factor :
  expected_factor = 3 := by
  sorry

end reading_speed_increase_factor_l667_667371


namespace arithmetic_lemma_l667_667357

theorem arithmetic_lemma : 45 * 52 + 48 * 45 = 4500 := by
  sorry

end arithmetic_lemma_l667_667357


namespace total_profit_equals_7_million_5_months_total_profit_equal_if_x_3_total_profit_after_two_years_l667_667349

variable {x : ℕ} (w : ℕ → ℕ) (y : ℕ → ℕ)

-- Define the monthly average profit function
def monthly_profit (x : ℕ) : ℕ := 10 * x + 90

-- Define the total profit function till month x
def total_profit (x : ℕ) : ℕ := x * monthly_profit x

-- Define the original total profit without equipment
def original_total_profit (x : ℕ) : ℕ := 120 * x

-- Theorem to prove the total profit equals 7 million yuan after 5 months
theorem total_profit_equals_7_million_5_months : total_profit 5 = 700 :=
by
  unfold total_profit monthly_profit
  simp [Nat.mul_eq_zero, Nat.add_eq_zero, Nat.succ_eq_one, Nat.mul_succ, Nat.zero_lt_succ, Nat.add_mul, Nat.one_mul]
  norm_num

-- Theorem to find value of x where total profit with and without equipment are equal
theorem total_profit_equal_if_x_3 : ∃ x, total_profit x = original_total_profit x ∧ x = 3 :=
by
  use 3
  unfold total_profit original_total_profit monthly_profit
  simp [Nat.mul_eq_zero, Nat.add_eq_zero, Nat.succ_eq_one, Nat.mul_succ, Nat.zero_lt_succ, Nat.add_mul, Nat.one_mul]
  norm_num

-- Theorem to calculate total profit after two years
theorem total_profit_after_two_years : (12 * monthly_profit 12) + (12 * monthly_profit 12) = 6360 :=
by
  unfold monthly_profit
  simp [Nat.mul_eq_zero, Nat.add_eq_zero, Nat.succ_eq_one, Nat.zero_lt_succ]
  norm_num

end total_profit_equals_7_million_5_months_total_profit_equal_if_x_3_total_profit_after_two_years_l667_667349


namespace constant_term_in_expansion_l667_667520

  theorem constant_term_in_expansion (x : ℝ) (hx : x ≠ 0) :
  let term := (x - x.pow (-1 / 3))^4
  in @has_polynomial_domain.hasPolynomialDomain ℕ x term.eval ⟨term⟩ 0 = -4 :=
begin
  sorry
end


end constant_term_in_expansion_l667_667520


namespace fill_grid_ways_l667_667773

-- Define the problem conditions.
def is_valid_row (row : List ℕ) : Prop :=
  (row.length = 3) ∧ (∀ i j, i ≠ j → row.nth i ≠ row.nth j)

def is_valid_grid (grid : List (List ℕ)) : Prop :=
  (grid.length = 3) ∧ (∀ i, (grid.nth i).is_some ∧ is_valid_row (grid.nth i).get) ∧
  (∀ j, ((List.range 3).map (λ i => (grid.nth i).get.nth j)).dedup.length = 3)

-- Define the question and answer tuple as a proof problem.
theorem fill_grid_ways : 
  (∃ (grid : List (List ℕ)), 
    is_valid_grid grid 
    ∧ (∀ n ∈ grid.bind id, n ∈ [1, 2, 3])
    ∧ ∃ ways : ℕ, ways = 12
  ) :=
  sorry

end fill_grid_ways_l667_667773


namespace matrix_product_correct_l667_667475

-- Define the given matrices
def A : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 2], ![0, -2]]
def B_inv : Matrix (Fin 2) (Fin 2) ℚ := ![![1, -1/2], ![0, 2]]

-- Define B as the inverse of B_inv
noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ :=
  Matrix.inv B_inv

-- Define the target matrix AB
def AB_expected : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 5/4], ![0, -1]]

-- The theorem stating that AB equals the expected result
theorem matrix_product_correct :
  A ⬝ B = AB_expected :=
by sorry

end matrix_product_correct_l667_667475


namespace simplify_and_find_ratio_l667_667055

theorem simplify_and_find_ratio (m : ℤ) (c d : ℤ) (h : (5 * m + 15) / 5 = c * m + d) : d / c = 3 := by
  sorry

end simplify_and_find_ratio_l667_667055


namespace chef_leftover_potatoes_l667_667318

-- Defining the conditions as variables
def fries_per_potato := 25
def total_potatoes := 15
def fries_needed := 200

-- Calculating the number of potatoes needed.
def potatoes_needed : ℕ :=
  fries_needed / fries_per_potato

-- Calculating the leftover potatoes.
def leftovers : ℕ :=
  total_potatoes - potatoes_needed

-- The theorem statement
theorem chef_leftover_potatoes :
  leftovers = 7 :=
by
  -- the actual proof is omitted.
  sorry

end chef_leftover_potatoes_l667_667318


namespace num_coloring_methods_l667_667141

theorem num_coloring_methods : 
  ∃ (f : Fin 6 → Bool), -- f is a function that assigns a color to each of the 6 fish
  (∀ i, f i ≠ f (i + 1)) ∧ -- no two adjacent fish are both colored red
  (∃ j, f j = true) ∧    -- there exists at least one red fish
  (∃ k, f k = false) ∧   -- there exists at least one blue fish
  (fin 6).card = 20 := sorry -- the number of such valid colorings is 20


end num_coloring_methods_l667_667141


namespace prove_dan_average_rate_l667_667246

noncomputable def dan_average_rate
  (run_flat_dist run_flat_speed run_uphill_dist run_uphill_speed run_downhill_dist run_downhill_speed 
   swim_dist swim_speed bike_dist bike_speed : ℝ) : ℝ :=
let run_flat_time := run_flat_dist / run_flat_speed,
    run_uphill_time := run_uphill_dist / run_uphill_speed,
    run_downhill_time := run_downhill_dist / run_downhill_speed,
    swim_time := swim_dist / swim_speed,
    bike_time := bike_dist / bike_speed,
    total_time := run_flat_time + run_uphill_time + run_downhill_time + swim_time + bike_time in
    (run_flat_dist + run_uphill_dist + run_downhill_dist + swim_dist + bike_dist) / (total_time * 60)

theorem prove_dan_average_rate :
  dan_average_rate 1 10 1 6 1 14 3 4 3 12 = 0.1121 :=
by sorry

end prove_dan_average_rate_l667_667246


namespace contrapositive_l667_667201

variable {α : Type} (M : α → Prop) (a b : α)

theorem contrapositive (h : (M a → ¬ M b)) : (M b → ¬ M a) := 
by
  sorry

end contrapositive_l667_667201


namespace min_value_parabola_l667_667632

theorem min_value_parabola : 
  ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 4 ∧ (-x^2 + 4 * x - 2) = -2 :=
by
  sorry

end min_value_parabola_l667_667632


namespace bm_perp_bn_l667_667535

-- Definitions and theorem
variables {A B C N M : Type} [plane_geometry A B C N M]

-- Equilateral triangle condition
def equilateral_triangle (ABC : triangle) : Prop :=
  ABC.AB = ABC.BC ∧ ABC.BC = ABC.CA ∧
  ABC.∠A = 60 ∧ ABC.∠B = 60 ∧ ABC.∠C = 60

-- Isosceles right triangles condition
def isosceles_right_triangle_ACN (ACN : triangle) : Prop :=
  ACN.∠A = 90 ∧ ACN.AC = ACN.AN

def isosceles_right_triangle_BCM (BCM : triangle) : Prop :=
  BCM.∠C = 90 ∧ BCM.BC = BCM.CM

-- Theorem statement
theorem bm_perp_bn
  (ABC : triangle) (ACN BCM : triangle)
  (h_eq_tri : equilateral_triangle ABC)
  (h_iso_right_ACN : isosceles_right_triangle_ACN ACN)
  (h_iso_right_BCM : isosceles_right_triangle_BCM BCM) :
  BM ⊥ BN :=
sorry

end bm_perp_bn_l667_667535


namespace probability_heads_less_than_tails_l667_667277

noncomputable def biased_coin_heads_prob : ℝ := 3 / 4
noncomputable def unbiased_coin_heads_prob : ℝ := 1 / 2
noncomputable def total_coins : ℕ := 12
noncomputable def biased_index : ℕ := 1
noncomputable def unbiased_coins : ℕ := 11

noncomputable def probability_fewer_heads_than_tails : ℝ :=
  -- This is a placeholder definition corresponding to the computed answer in b)
  3172 / 8192

theorem probability_heads_less_than_tails :
  (calculate_probability_fewer_heads_than_tails total_coins unbiased_coins biased_coin_heads_prob unbiased_coin_heads_prob) 
  = probability_fewer_heads_than_tails :=
sorry

end probability_heads_less_than_tails_l667_667277


namespace solution_l667_667458

noncomputable def f : ℝ → ℝ := sorry -- Definition of f based on conditions provided

axiom cond1 : ∀ x : ℝ, f (-x) = f x
axiom cond2 : ∀ x : ℝ, f (x + 2) = f (x - 2)
axiom cond3 : ∀ x : ℝ, (0 < x ∧ x < 2) → f x = log (x + 1)

theorem solution : f (-3 / 2) > f 1 ∧ f 1 > f (7 / 2) :=
by { sorry }

end solution_l667_667458


namespace fiona_pairs_l667_667412

-- Define the combinatorial calculation using the combination formula
def combination (n k : ℕ) := n.choose k

-- The main theorem stating that the number of pairs from 6 people is 15
theorem fiona_pairs : combination 6 2 = 15 :=
by
  sorry

end fiona_pairs_l667_667412


namespace find_lowest_score_l667_667197

constant mean15 : ℝ := 90
constant mean13 : ℝ := 92
constant numScores : ℕ := 15
constant maxScore : ℝ := 110

theorem find_lowest_score : 
  let totalSum := mean15 * numScores
  let remainingScores := numScores - 2
  let subtractedSum := mean13 * remainingScores
  let highLowSum := totalSum - subtractedSum
  highLowSum - maxScore = 44 := 
by
  let totalSum := mean15 * numScores
  let remainingScores := numScores - 2
  let subtractedSum := mean13 * remainingScores
  let highLowSum := totalSum - subtractedSum
  show highLowSum - maxScore = 44 from sorry

end find_lowest_score_l667_667197


namespace parallel_vectors_perpendicular_vectors_l667_667970

open Real

variables {m : ℝ}

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Problem 1: If a is parallel to b, then m = -4
theorem parallel_vectors (h : ∃ k : ℝ, k • vector_a = vector_b m) : m = -4 :=
sorry

-- Problem 2: If a is perpendicular to b, then m = 1
theorem perpendicular_vectors (h : vector_a.1 * vector_b(m).1 + vector_a.2 * vector_b(m).2 = 0) : m = 1 :=
sorry

end parallel_vectors_perpendicular_vectors_l667_667970


namespace cot_30_eq_sqrt_3_l667_667804

theorem cot_30_eq_sqrt_3 (h : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3) : 
  Real.cot (Real.pi / 6) = Real.sqrt 3 :=
by
  sorry

end cot_30_eq_sqrt_3_l667_667804


namespace probability_of_both_events_l667_667313

theorem probability_of_both_events (P_A P_B P_union : ℝ) (hA : P_A = 0.20) (hB : P_B = 0.30) (h_union : P_union = 0.35) : 
  P_A + P_B - P_union = 0.15 :=
by
  simp [hA, hB, h_union]
  sorry

end probability_of_both_events_l667_667313


namespace angle_between_nonzero_plane_vectors_l667_667216

theorem angle_between_nonzero_plane_vectors (u v : ℝ × ℝ) (hu : u ≠ (0, 0)) (hv : v ≠ (0, 0)) :
  ∃ θ, 0 ≤ θ ∧ θ ≤ real.pi :=
sorry

end angle_between_nonzero_plane_vectors_l667_667216


namespace general_formula_a_n_T_n_greater_than_S_n_l667_667928

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n_l667_667928


namespace simplify_expression_l667_667596

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l667_667596


namespace find_expression_value_l667_667504

theorem find_expression_value (x : ℝ) (h : x + 1/x = 3) : 
  x^10 - 5 * x^6 + x^2 = 8436*x - 338 := 
by {
  sorry
}

end find_expression_value_l667_667504


namespace cot_30_eq_sqrt3_l667_667790

theorem cot_30_eq_sqrt3 : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by
  have h_cos : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  have h_sin : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  show Real.cot (30 * Real.pi / 180) = Real.sqrt 3
  calc
    Real.cot (30 * Real.pi / 180) = (Real.cos (30 * Real.pi / 180)) / (Real.sin (30 * Real.pi / 180)) := sorry
    ... = (Real.sqrt 3 / 2) / (1 / 2) := sorry
    ... = Real.sqrt 3 := sorry

end cot_30_eq_sqrt3_l667_667790


namespace sum_of_digits_3_to_300_sum_of_digits_3_to_3000_l667_667759

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem sum_of_digits_3_to_300 :
  ∑ k in finset.range 100, sum_of_digits (3 * k) = 1059 :=
sorry

theorem sum_of_digits_3_to_3000 :
  ∑ k in finset.range 1001, sum_of_digits (3 * k) = 16059 :=
sorry

end sum_of_digits_3_to_300_sum_of_digits_3_to_3000_l667_667759


namespace y_intercept_of_line_l667_667268

theorem y_intercept_of_line (x y : ℝ) (eq : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 := 
by
  sorry

end y_intercept_of_line_l667_667268


namespace simplify_expression_l667_667595

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l667_667595


namespace part1_and_part2_l667_667890

-- Define the arithmetic sequence {a_n}
def a (n : Nat) : Nat := 2 * n + 3

-- Define the sequence {b_n}
def b (n : Nat) : Nat :=
  if n % 2 = 0 then 4 * n + 6 else 2 * n - 3

-- Define the sum of the first n terms of a sequence
def summation (seq : Nat → Nat) (n : Nat) : Nat :=
  (List.range n).map seq |>.sum

-- Define S_n as the sum of the first n terms of {a_n}
def S (n : Nat) : Nat := summation a n

-- Define T_n as the sum of the first n terms of {b_n}
def T (n : Nat) : Nat := summation b n

-- Given conditions
axiom S4_eq_32 : S 4 = 32
axiom T3_eq_16 : T 3 = 16

-- Prove the general formula for {a_n} and that T_n > S_n for n > 5
theorem part1_and_part2 (n : Nat) (h : n > 5) : a n = 2 * n + 3 ∧ T n > S n :=
  by
  sorry

end part1_and_part2_l667_667890


namespace g3_eq_6_f_neg1_eq_neg1_sum_f_eq_neg2025_l667_667019

section
variables {f g : ℝ → ℝ}

-- Conditions
axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom domain_g : ∀ x : ℝ, g x ∈ ℝ
axiom eq1 : ∀ x : ℝ, f x + g (2 - x) = 5
axiom eq2 : ∀ x : ℝ, g x - f (x - 4) = 7
axiom symmetry_g : ∀ x : ℝ, g (2 - x) = g (2 + x)
axiom g2 : g 2 = 4

-- Proof goals
theorem g3_eq_6 : g 3 = 6 := sorry
theorem f_neg1_eq_neg1 : f (-1) = -1 := sorry
theorem sum_f_eq_neg2025 : ∑ k in finset.range 2023, f (k + 1 : ℝ) = -2025 := sorry

end

end g3_eq_6_f_neg1_eq_neg1_sum_f_eq_neg2025_l667_667019


namespace exists_trio_interacted_l667_667651

noncomputable theory

-- Definitions
def Country : Type := ℕ
def Mathematician : Country → Type := λ c, Fin n

-- Assumptions
variables (n : ℕ) (A B C : Country) (iA : Mathematician A → Fin (3 * n - n - 1) → Mathematician B ⊕ Mathematician C)
variables (iB : Mathematician B → Fin (3 * n - n - 1) → Mathematician A ⊕ Mathematician C)
variables (iC : Mathematician C → Fin (3 * n - n - 1) → Mathematician A ⊕ Mathematician B)

axiom interactA (a : Mathematician A) : ∃ (b : Mathematician B) (c : Mathematician C),
  iA a (Fin.cast (3 * n - 2) (nat.succ (nat.pred (iB b (Fin.cast (3 * n - 2) (nat.succ (nat.pred (iC c))))) = sum.inl a ∧
  iB b (Fin.cast (3 * n - 2) (nat.succ (nat.pred (iC c)))) = sum.inl a

theorem exists_trio_interacted :
  ∃ (a : Mathematician A) (b : Mathematician B) (c : Mathematician C),
  (iA a (Fin.cast (3 * n - 2) (nat.succ (nat.pred (iB b (Fin.cast (3 * n - 2) (nat.succ (nat.pred (iC c))))))) = sum.inl a ∧
  (iB b (Fin.cast (3 * n - 2) (nat.succ (nat.pred (iC c)))) = sum.inl b ∧
  (iC c (Fin.cast (3 * n - 2) (nat.succ (nat.pred (iA a)))) = sum.inl c :=
by sorry

end exists_trio_interacted_l667_667651


namespace cot_30_eq_sqrt_3_l667_667829

theorem cot_30_eq_sqrt_3 (θ : ℝ) (hθ : θ = 30) (h1 : tan 30 = 1 / real.sqrt 3) : cot 30 = real.sqrt 3 :=
by
  have h2: cot 30 = 1 / (tan 30), by exact cot_def
  have h3: cot 30 = 1 / (1 / real.sqrt 3), by rw [h1, h2]
  have h4: cot 30 = real.sqrt 3, by exact (one_div_div (1 / real.sqrt 3))
  exact h4


end cot_30_eq_sqrt_3_l667_667829


namespace r_expansion_infinite_l667_667404

theorem r_expansion_infinite {r : ℝ} (h_r_not_int : ¬ r ∈ ℤ) :
  ∃ p : ℝ, p ∈ set.Icc 0 (n / (r - 1)) ∧
  ∃ (a : ℕ → ℕ), (∀ i, 0 ≤ a i ∧ a i < floor r) ∧
  ∀ N : ℕ, ∃ (x : ℝ), (∀ m < N, x ≠ ∑ k in finset.range m, a k * r^(-m)) :=
begin
  sorry
end

end r_expansion_infinite_l667_667404


namespace solve_dfrac_eq_l667_667487

theorem solve_dfrac_eq (x : ℝ) (h : (x / 5) / 3 = 3 / (x / 5)) : x = 15 ∨ x = -15 := by
  sorry

end solve_dfrac_eq_l667_667487


namespace no_pair_more_than_one_km_l667_667652

noncomputable def radius : ℝ := 1

def cyclers_on_circle (time : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  -- Assume three cyclers with different constant angular speeds ω1, ω2, ω3
  let ω1 := 1
  let ω2 := 2
  let ω3 := 3
  -- Positions on the circle parametrized by angle (using polar coordinates)
  (
    (radius * real.cos (ω1 * time), radius * real.sin (ω1 * time)),
    (radius * real.cos (ω2 * time), radius * real.sin (ω2 * time)),
    (radius * real.cos (ω3 * time), radius * real.sin (ω3 * time))
  )

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem no_pair_more_than_one_km (t : ℝ) :
  let
    p1 := (radius * real.cos (1 * t), radius * real.sin (1 * t))
    p2 := (radius * real.cos (2 * t), radius * real.sin (2 * t))
    p3 := (radius * real.cos (3 * t), radius * real.sin (3 * t))
  in
  ¬ (distance p1 p2 > 1 ∧ distance p2 p3 > 1 ∧ distance p1 p3 > 1) := sorry

end no_pair_more_than_one_km_l667_667652


namespace shortest_side_of_similar_triangle_l667_667338

theorem shortest_side_of_similar_triangle (a1 a2 h1 h2 : ℝ)
  (h1_eq : a1 = 24)
  (h2_eq : h1 = 37)
  (h2_eq' : h2 = 74)
  (h_similar : h2 / h1 = 2)
  (h_a2_eq : a2 = 2 * Real.sqrt 793):
  a2 = 2 * Real.sqrt 793 := by
  sorry

end shortest_side_of_similar_triangle_l667_667338


namespace maximum_triangle_area_l667_667105

variable (A B C D E F P : Type) [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C]
variable [AffineSpace ℝ D] [AffineSpace ℝ E] [AffineSpace ℝ F] [AffineSpace ℝ P]
variable (parallelogram : Parallelogram A B C D)
variable (midpoint_E : E = midpoint E A B)
variable (midpoint_F : F = midpoint F B C)
variable (area_abcd : ∀ rect, Parallelogram.area parallelogram = 100)
variable (intersection_P : intersect EC FD = P)

theorem maximum_triangle_area (parallelogram ABCD)
    (midpoint E A B)
    (midpoint F B C)
    (area (parallelogram) ABCD 100)
    (intersection P (EC FD) P):
    ∃ (max_area : ℝ), max_area = 40 :=
  sorry

end maximum_triangle_area_l667_667105


namespace length_BC_l667_667061

-- Definitions representing the given conditions
variables {A B C Y Z : Point}
variables (hAB : dist A B = 75) (hAC : dist A C = 100)
variables (circleABC : Circle A 75) (hBY : B ∈ circleABC ∧ Y ∈ circleABC)
variables (hAZ : Z ∈ circleABC) (hDistinct : Z ≠ A)

-- Goal to prove the length of BC (BY + CY) is 125
theorem length_BC :
  dist B C = 125 :=
sorry

end length_BC_l667_667061


namespace problem_statement_l667_667369

theorem problem_statement :
  (\frac{\sum_{k=1}^{2018} (2019 - k) / k}
        { \sum_{k=2}^{2019} (1 / k)}) = 2019 :=
sorry

end problem_statement_l667_667369


namespace equation_graph_is_ellipse_l667_667761

theorem equation_graph_is_ellipse :
  ∃ a b c d : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b * (x^2 - 72 * y^2) + a * x + d = a * c * (y - 6)^2 :=
sorry

end equation_graph_is_ellipse_l667_667761


namespace soda_consumption_l667_667413

theorem soda_consumption 
    (dozens : ℕ)
    (people_per_dozen : ℕ)
    (cost_per_box : ℕ)
    (cans_per_box : ℕ)
    (family_members : ℕ)
    (payment_per_member : ℕ)
    (dozens_eq : dozens = 5)
    (people_per_dozen_eq : people_per_dozen = 12)
    (cost_per_box_eq : cost_per_box = 2)
    (cans_per_box_eq : cans_per_box = 10)
    (family_members_eq : family_members = 6)
    (payment_per_member_eq : payment_per_member = 4) :
  (60 * (cans_per_box)) / 60 = 2 :=
by
  -- proof would go here eventually
  sorry

end soda_consumption_l667_667413


namespace outfits_count_l667_667669

-- Definitions of the counts of each type of clothing item
def num_blue_shirts : Nat := 6
def num_green_shirts : Nat := 4
def num_pants : Nat := 7
def num_blue_hats : Nat := 9
def num_green_hats : Nat := 7

-- Statement of the problem to prove
theorem outfits_count :
  (num_blue_shirts * num_pants * num_green_hats) + (num_green_shirts * num_pants * num_blue_hats) = 546 :=
by
  sorry

end outfits_count_l667_667669


namespace solution_set_x_squared_f_x_l667_667117

theorem solution_set_x_squared_f_x (f : ℝ → ℝ) 
  (odd_f : ∀ x, f (-x) = -f x)
  (f_2_eq_0 : f 2 = 0)
  (cond : ∀ x > 0, x * (deriv f x) - f x < 0) 
  : {x : ℝ | x^2 * f x > 0} = {x : ℝ | x ∈ (Set.Ioo (-∞) (-2)) ∪ Set.Ioo (0) (2)} :=
begin
  sorry
end

end solution_set_x_squared_f_x_l667_667117


namespace expression_defined_interval_l667_667370

theorem expression_defined_interval (x : ℝ) : 
  (5 - x > 0) → (x - 2 > 0) → (2 < x ∧ x < 5) := 
by 
  intros h1 h2
  split
  · linarith
  · linarith

end expression_defined_interval_l667_667370


namespace ratio_of_photos_l667_667142

-- Definitions based on conditions
def initial_photos : ℕ := 100
def first_week_photos : ℕ := 50
def third_fourth_weeks_photos : ℕ := 80
def final_photos : ℕ := 380

-- Derived definition from the conditions
def total_new_photos : ℕ := final_photos - initial_photos
def second_week_photos : ℕ := total_new_photos - first_week_photos - third_fourth_weeks_photos

-- Statement to prove the ratio
theorem ratio_of_photos (h1 : total_new_photos = final_photos - initial_photos) 
                        (h2 : second_week_photos = total_new_photos - first_week_photos - third_fourth_weeks_photos)
                        (h3 : final_photos = 380) :
    (second_week_photos / first_week_photos) = 3 :=
by
  -- Using the conditions and derived definitions
  have h_total : total_new_photos = 280 :=
    calc total_new_photos = final_photos - initial_photos : h1
                    ...  = 380 - 100 : by rw h3
                    ...  = 280 : by norm_num,
  have h_second : second_week_photos = 150 :=
    calc second_week_photos = total_new_photos - first_week_photos - third_fourth_weeks_photos : h2
                      ...  = 280 - 50 - 80 : by rw [h_total]
                      ...  = 150 : by norm_num,
  show (second_week_photos / first_week_photos) = 3,
  rw [h_second],
  norm_num,

-- Sorry to skip the proof, assuming all the conditions are met
sorry

end ratio_of_photos_l667_667142


namespace exists_long_diagonal_le_two_l667_667560

theorem exists_long_diagonal_le_two (A B C D E F : ℝ^2)
    (convex_hex : convex (set.of_list [A, B, C, D, E, F]))
    (side_lengths : ∀ {X Y : ℝ^2}, (X, Y) ∈ set.pairwise [A, B, C, D, E, F] → dist X Y ≤ 1) :
    ∃ (d₁ d₂ : ℝ^2), (d₁, d₂) ∈ set.pairwise [A, B, C, D, E, F] ∧ dist d₁ d₂ ≤ 2 := by
    sorry

end exists_long_diagonal_le_two_l667_667560


namespace general_formula_sums_inequality_for_large_n_l667_667904

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def modified_seq (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = (finset.range n).sum a

variables {a b S T : ℕ → ℕ}

-- Given conditions
axiom h_arith_seq : arithmetic_seq a
axiom h_modified_seq : modified_seq a b
axiom h_sum_a : sum_seq a S
axiom h_sum_b : sum_seq b T
axiom h_S4 : S 4 = 32
axiom h_T3 : T 3 = 16

-- Proof Statements
theorem general_formula :
  ∀ n : ℕ, a n = 2 * n + 3 := sorry

theorem sums_inequality_for_large_n :
  ∀ n : ℕ, n > 5 → T n > S n := sorry

end general_formula_sums_inequality_for_large_n_l667_667904


namespace exists_infinite_pos_sequence_l667_667769

theorem exists_infinite_pos_sequence :
  ∃ (a : ℕ → ℕ), (∀ n, 0 < a n) ∧ (∀ m n, m ≠ n → a m ≠ a n) ∧ (∀ n, ∃ k : ℕ, (∏ i in finset.range (n + 1), a i) = k^(n + 1)) :=
by
  sorry

end exists_infinite_pos_sequence_l667_667769


namespace general_formula_a_n_T_n_greater_S_n_l667_667873

-- Define the conditions for the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ (n : ℕ), a n = a1 + (n - 1) * d

def b_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_of_first_n_terms (a S : ℕ → ℕ) :=
  ∀ (n k : ℕ), k ≤ n → S n = (n * (a 1 + a n)) / 2

-- Given S4 = 32 and T3 = 16
def S_4_equals_32 (S : ℕ → ℕ) : Prop := S 4 = 32
def T_3_equals_16 (T : ℕ → ℕ) : Prop := T 3 = 16

-- Prove that the general formula for a_n is 2n + 3
theorem general_formula_a_n (a : ℕ → ℕ) (h : arithmetic_sequence a) : 
  a = λ n, 2 * n + 3 := 
sorry

-- Prove that for n > 5, T_n > S_n
theorem T_n_greater_S_n (a b S T: ℕ → ℕ) 
  (ha : arithmetic_sequence a) 
  (hb : b_sequence a b)
  (hS : sum_of_first_n_terms a S)
  (hT : sum_of_first_n_terms b T)
  (hS4 : S_4_equals_32 S)
  (hT3 : T_3_equals_16 T) 
  (n : ℕ) : n > 5 → T n > S n :=
sorry

end general_formula_a_n_T_n_greater_S_n_l667_667873


namespace whale_length_l667_667252

theorem whale_length
  (velocity_fast : ℕ)
  (velocity_slow : ℕ)
  (time : ℕ)
  (h1 : velocity_fast = 18)
  (h2 : velocity_slow = 15)
  (h3 : time = 15) :
  (velocity_fast - velocity_slow) * time = 45 := 
by
  sorry

end whale_length_l667_667252


namespace coat_price_reduction_l667_667214

variables (original_price reduction_amount : ℝ)

def percent_reduction (original_price reduction_amount : ℝ) : ℝ :=
  (reduction_amount / original_price) * 100

theorem coat_price_reduction
  (h_original_price : original_price = 500)
  (h_reduction_amount : reduction_amount = 250) :
  percent_reduction original_price reduction_amount = 50 := 
by
  sorry

end coat_price_reduction_l667_667214


namespace cot_30_eq_sqrt3_l667_667776

theorem cot_30_eq_sqrt3 : Real.cot (π / 6) = Real.sqrt 3 := by
  let tan_30 := 1 / Real.sqrt 3
  have tan_30_eq : Real.tan (π / 6) = tan_30 := sorry
  have cot_def : ∀ x, Real.cot x = 1 / Real.tan x := sorry
  rw [cot_def (π / 6), tan_30_eq]
  simp
  sorry

end cot_30_eq_sqrt3_l667_667776


namespace find_lowest_score_l667_667192

noncomputable def lowest_score (total_scores : ℕ) (mean_scores : ℕ) (new_total_scores : ℕ) (new_mean_scores : ℕ) (highest_score : ℕ) : ℕ :=
  let total_sum        := mean_scores * total_scores
  let sum_remaining    := new_mean_scores * new_total_scores
  let removed_sum      := total_sum - sum_remaining
  let lowest_score     := removed_sum - highest_score
  lowest_score

theorem find_lowest_score :
  lowest_score 15 90 13 92 110 = 44 :=
by
  unfold lowest_score
  simp [Nat.mul_sub, Nat.sub_apply, Nat.add_comm]
  rfl

end find_lowest_score_l667_667192


namespace find_AC_l667_667518

open Real

-- Definitions from conditions
def BD : ℝ := 16
def AB : ℝ := 9
def CE : ℝ := 5
def DE : ℝ := 3

-- Derived quantities from the solution using conditions
def CD : ℝ := sqrt (CE^2 - DE^2)
def BC : ℝ := BD - CD

-- Final proof statement
theorem find_AC : AC = sqrt (AB^2 + BC^2) :=
  by 
    unfold AC
    sorry

end find_AC_l667_667518


namespace find_a_in_expansion_l667_667456

theorem find_a_in_expansion (a : ℝ) : 
  (let term_coeff := 6 + 4 * a in term_coeff = 10) → a = 1 :=
by 
  intro h
  simp at h
  linarith

end find_a_in_expansion_l667_667456


namespace nearest_integer_to_sum_l667_667496

theorem nearest_integer_to_sum (x y : ℝ) (h1 : |x| - y = 1) (h2 : |x| * y + x^2 = 2) : Int.ceil (x + y) = 2 :=
sorry

end nearest_integer_to_sum_l667_667496


namespace prob_intersection_l667_667937

-- Define the probability of an event
variable (Ω : Type*) [ProbabilityMeasure Ω]

-- Define events A and B
variables (A B : Set Ω)

-- Given conditions
variable (h_independent : Independent A B)
variable (h_PA : ProbabilityMeasure.probability A = 1/4)
variable (h_PB : ProbabilityMeasure.probability B = 1/13)

-- Theorem statement
theorem prob_intersection : ProbabilityMeasure.probability (A ∩ B) = 1/52 := by
  sorry

end prob_intersection_l667_667937


namespace cot_30_deg_l667_667796

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 := by
  -- Define known values of trigonometric functions at 30 degrees
  have h1 : Real.tan (30 * Real.pi / 180) = 1 / Real.sqrt 3 := sorry
  have h2 : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

  -- Prove using these known values
  calc
    Real.cot (30 * Real.pi / 180)
        = 1 / Real.tan (30 * Real.pi / 180) : by sorry
    ... = 1 / (1 / Real.sqrt 3) : by sorry
    ... = Real.sqrt 3 : by sorry

end cot_30_deg_l667_667796


namespace ratio_of_pens_to_pencils_l667_667720

/-
The store ordered pens and pencils:
1. The number of pens was some multiple of the number of pencils plus 300.
2. The cost of a pen was $5.
3. The cost of a pencil was $4.
4. The store ordered 15 boxes, each having 80 pencils.
5. The store paid a total of $18,300 for the stationery.
Prove that the ratio of the number of pens to the number of pencils is 2.25.
-/

variables (e p k : ℕ)
variables (cost_pen : ℕ := 5) (cost_pencil : ℕ := 4) (total_cost : ℕ := 18300)

def number_of_pencils := 15 * 80

def number_of_pens := p -- to be defined in terms of e and k

def total_cost_pens := p * cost_pen
def total_cost_pencils := e * cost_pencil

theorem ratio_of_pens_to_pencils :
  p = k * e + 300 →
  e = 1200 →
  5 * p + 4 * e = 18300 →
  (p : ℚ) / e = 2.25 :=
by
  intros hp he htotal
  sorry

end ratio_of_pens_to_pencils_l667_667720


namespace evaluate_power_sum_l667_667391

noncomputable def i : ℂ := Complex.I

lemma powers_of_i_repeat (n : ℕ) : i^(n % 4) = i^n :=
begin
  sorry
end

theorem evaluate_power_sum : i^20 + i^35 = 1 - i :=
begin
  have h1 : i^20 = 1,
  {
    sorry
  },
  have h2 : i^35 = -i,
  {
    sorry
  },
  calc
    i^20 + i^35 = 1 + (-i) : by rw [h1, h2]
              ... = 1 - i   : by rw add_neg_eq_sub,
end

end evaluate_power_sum_l667_667391


namespace T_gt_S_for_n_gt_5_l667_667921

-- Define the arithmetic sequence a_n
def a : ℕ → ℕ
| n => 2 * n + 3

-- Define the sequence b_n
def b : ℕ → ℕ
| n => if n % 2 = 1 then a n - 6 else 2 * (a n)

-- Sum of first n terms of a
def S (n : ℕ) : ℕ := (List.range n).sum (λ k => a (k + 1))

-- Sum of first n terms of b
def T (n : ℕ) : ℕ := (List.range n).sum (λ k => b (k + 1))

-- Given conditions in Lean
axiom S_4 : S 4 = 32
axiom T_3 : T 3 = 16

-- The theorem to prove
theorem T_gt_S_for_n_gt_5 (n : ℕ) (hn : n > 5) : T n > S n :=
  sorry

end T_gt_S_for_n_gt_5_l667_667921


namespace integral_inequality_l667_667706

theorem integral_inequality 
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x ≥ 0)
  (integrable_f : integrable_on f (Ici 0))
  (hx_int_f2 : ∫ x in 0..∞, f x ^ 2 < ∞)
  (hx_int_xf : ∫ x in 0..∞, x * f x < ∞) :
  (∫ x in 0..∞, f x) ^ 3 ≤ 8 * (∫ x in 0..∞, f x ^ 2) * (∫ x in 0..∞, x * f x) :=
by
  sorry

end integral_inequality_l667_667706


namespace rhombus_C_coordinates_sum_l667_667653

section RhombusCoordinates

variables {A B D C : (ℝ × ℝ)}
variables {Ax Ay Bx By Dx Dy Cx Cy : ℝ}

-- Definitions based on given conditions
def vertex_A := (-3, -2 : ℝ)
def vertex_B := (1, -5 : ℝ)
def vertex_D := (9, 1  : ℝ)

-- Assume C is (Cx, Cy)
def vertex_C := (Cx, Cy : ℝ)

-- Function to calculate the midpoint of two vertices
def midpoint (P Q : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Use the property of rhombus that diagonals bisect each other perpendicularly
theorem rhombus_C_coordinates_sum
  (H : midpoint vertex_A vertex_D = midpoint vertex_B vertex_C) :
  Cx + Cy = 9 :=
sorry

end RhombusCoordinates

end rhombus_C_coordinates_sum_l667_667653


namespace problem_4_pts_sol_exist_l667_667361

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def perimeter (A B C : ℝ × ℝ) : ℝ :=
  distance A B + distance B C + distance C A

noncomputable def area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))

theorem problem_4_pts_sol_exist :
  ∃ (C1 C2 C3 C4 : ℝ × ℝ), 
  let A := (0, 0) in 
  let B := (8, 0) in
  distance A B = 8 ∧ 
  perimeter A B C1 = 36 ∧
  area A B C1 = 48 ∧
  perimeter A B C2 = 36 ∧
  area A B C2 = 48 ∧
  perimeter A B C3 = 36 ∧
  area A B C3 = 48 ∧
  perimeter A B C4 = 36 ∧
  area A B C4 = 48 :=
sorry

end problem_4_pts_sol_exist_l667_667361


namespace maximum_value_of_f_l667_667946

def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem maximum_value_of_f : ∃ x ∈ set.Icc (-2 : ℝ) (4 : ℝ), ∀ y ∈ set.Icc (-2) 4, f y ≤ f x :=
by
  sorry

end maximum_value_of_f_l667_667946


namespace sphere_surface_area_l667_667983

theorem sphere_surface_area (S : ℝ) (hS : S = 6) :
  let a := sqrt (S / 6) in
  let d := a * sqrt 3 in
  let R := d / 2 in
  4 * π * R^2 = 3 * π := by
  let a := sqrt (S / 6)
  let d := a * sqrt 3
  let R := d / 2
  rw [hS, a, d, R]
  sorry

end sphere_surface_area_l667_667983


namespace hikers_count_l667_667506

theorem hikers_count (B H K : ℕ) (h1 : H = B + 178) (h2 : K = B / 2) (h3 : H + B + K = 920) : H = 474 :=
by
  sorry

end hikers_count_l667_667506


namespace find_lambda_l667_667454

variables {α : Type*} [inner_product_space ℝ α] 

-- Definitions of points A, B, C, and G, and the position of the centroid G  
variables {A B C G : α}

-- Given conditions
def is_centroid (G : α) (A B C : α) : Prop := 
G = (A + B + C) / 3

def perp_vectors (G : α) (B C: α) : Prop :=
inner_product_space.dot_product (B - G) (C - G) = 0

-- The equation to prove
def given_eq (A B C : α) (λ : ℝ) : Prop :=
1 / real.tan (∠ B C A) + 1 / real.tan (∠ C B A) = λ / real.tan (∠ A B C)

-- Proof statement
theorem find_lambda (A B C G : α) (λ : ℝ) 
  (h1 : is_centroid G A B C) 
  (h2 : perp_vectors G B C) 
  (h3 : given_eq A B C λ) : 
  λ = 1/2 := 
sorry

end find_lambda_l667_667454


namespace number_of_integers_divisible_by_neither_6_nor_8_l667_667635

theorem number_of_integers_divisible_by_neither_6_nor_8 :
  (finset.filter (λ n, n % 6 ≠ 0 ∧ n % 8 ≠ 0) (finset.range 1200)).card = 900 :=
by sorry

end number_of_integers_divisible_by_neither_6_nor_8_l667_667635


namespace value_range_for_inequality_solution_set_l667_667495

-- Define the condition
def condition (a : ℝ) : Prop := a > 0

-- Define the inequality
def inequality (x a : ℝ) : Prop := |x - 4| + |x - 3| < a

-- State the theorem to be proven
theorem value_range_for_inequality_solution_set (a : ℝ) (h: condition a) : (a > 1) ↔ ∃ x : ℝ, inequality x a := 
sorry

end value_range_for_inequality_solution_set_l667_667495


namespace polynomial_coefficients_equal_l667_667263

theorem polynomial_coefficients_equal {n : ℕ} (A B : ℕ → ℝ) :
  (∀ x : ℝ, (∑ i in Finset.range (n + 1), A i * x^i) = (∑ i in Finset.range (n + 1), B i * x^i)) →
  (∀ i : ℕ, i ≤ n → A i = B i) :=
by
  sorry

end polynomial_coefficients_equal_l667_667263


namespace identity_implies_a_minus_b_l667_667975

theorem identity_implies_a_minus_b (a b : ℚ) (y : ℚ) (h : y > 0) :
  (∀ y, y > 0 → (a / (y - 3) + b / (y + 5) = (3 * y + 7) / ((y - 3) * (y + 5)))) → (a - b = 1) :=
by
  sorry

end identity_implies_a_minus_b_l667_667975


namespace incorrect_statement_in_regression_model_l667_667991

theorem incorrect_statement_in_regression_model:
  let model : ℕ → ℝ := λ t, 0.92 + 0.1 * t in
  ¬ (model 9 = 1.82) :=
by
  sorry

end incorrect_statement_in_regression_model_l667_667991


namespace minimize_g_function_l667_667771

noncomputable def g (x : ℝ) : ℝ := (9 * x^2 + 18 * x + 29) / (8 * (2 + x))

theorem minimize_g_function : ∀ x : ℝ, x ≥ -1 → g x = 29 / 8 :=
sorry

end minimize_g_function_l667_667771


namespace belinda_passed_out_percentage_l667_667743

variable (total_flyers ryan_flyers alyssa_flyers scott_flyers : ℕ)
variable (belinda_flyers : ℕ)

-- Define the total number of flyers
def total_flyers := 200
-- Define the number of flyers passed out by Ryan
def ryan_flyers := 42
-- Define the number of flyers passed out by Alyssa
def alyssa_flyers := 67
-- Define the number of flyers passed out by Scott
def scott_flyers := 51

-- Define the flyers that Belinda passed out
def belinda_flyers := total_flyers - (ryan_flyers + alyssa_flyers + scott_flyers)

-- Define the percentage calculation
def belinda_percentage : ℕ := (belinda_flyers * 100) / total_flyers

theorem belinda_passed_out_percentage :
  belinda_percentage total_flyers ryan_flyers alyssa_flyers scott_flyers belinda_flyers = 20 :=
by
  -- proof goes here
  sorry

end belinda_passed_out_percentage_l667_667743


namespace omega_value_l667_667056

theorem omega_value (ω : ℝ) (hω_pos : ω > 0) :
  (∀ x1 x2, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 3 → f x1 ≤ f x2) ∧
  (∀ x1 x2, π / 3 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 2 → f x1 ≥ f x2) →
  ω = 3 / 2 :=
by
  -- define the function f(x) = sin(ω * x)
  let f := λ x : ℝ, Real.sin (ω * x)
  -- assume the given conditions as hypotheses

  -- monotonicity conditions as hypotheses would be applied here

  -- the maximum value condition implies the value for ω given the intervals
  sorry

end omega_value_l667_667056


namespace Luke_spent_per_week_l667_667134

-- Definitions based on the conditions
def money_from_mowing := 9
def money_from_weeding := 18
def total_money := money_from_mowing + money_from_weeding
def weeks := 9
def amount_spent_per_week := total_money / weeks

-- The proof statement
theorem Luke_spent_per_week :
  amount_spent_per_week = 3 := 
  sorry

end Luke_spent_per_week_l667_667134


namespace part1_part2_l667_667431

def P : Set ℝ := {x | x ≥ 1 / 2 ∧ x ≤ 2}

def Q (a : ℝ) : Set ℝ := {x | a * x^2 - 2 * x + 2 > 0}

def R (a : ℝ) : Set ℝ := {x | a * x^2 - 2 * x + 2 = 0}

theorem part1 (a : ℝ) : (∃ x, x ∈ P ∧ x ∈ Q a) → a > -1 / 2 :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x, x ∈ P ∧ x ∈ R a) → a ≥ -1 / 2 ∧ a ≤ 1 / 2 :=
by
  sorry

end part1_part2_l667_667431


namespace family_snails_l667_667329

def total_snails_family (n1 n2 n3 n4 : ℕ) (mother_find : ℕ) : ℕ :=
  n1 + n2 + n3 + mother_find

def first_ducklings_snails (num_ducklings : ℕ) (snails_per_duckling : ℕ) : ℕ :=
  num_ducklings * snails_per_duckling

def remaining_ducklings_snails (num_ducklings : ℕ) (mother_snails : ℕ) : ℕ :=
  num_ducklings * (mother_snails / 2)

def mother_find_snails (snails_group1 : ℕ) (snails_group2 : ℕ) : ℕ :=
  3 * (snails_group1 + snails_group2)

theorem family_snails : 
  ∀ (ducklings : ℕ) (group1_ducklings group2_ducklings : ℕ) 
    (snails1 snails2 : ℕ) 
    (total_ducklings : ℕ), 
    ducklings = 8 →
    group1_ducklings = 3 → 
    group2_ducklings = 3 → 
    snails1 = 5 →
    snails2 = 9 →
    total_ducklings = group1_ducklings + group2_ducklings + 2 →
    total_snails_family 
      (first_ducklings_snails group1_ducklings snails1)
      (first_ducklings_snails group2_ducklings snails2)
      (remaining_ducklings_snails 2 (mother_find_snails 
        (first_ducklings_snails group1_ducklings snails1)
        (first_ducklings_snails group2_ducklings snails2)))
      (mother_find_snails 
        (first_ducklings_snails group1_ducklings snails1)
        (first_ducklings_snails group2_ducklings snails2)) 
    = 294 :=
by intros; sorry

end family_snails_l667_667329


namespace log_base_5_of_1_div_25_eq_neg_2_l667_667379

theorem log_base_5_of_1_div_25_eq_neg_2 :
  log 5 (1/25) = -2 :=
by sorry

end log_base_5_of_1_div_25_eq_neg_2_l667_667379


namespace cot_30_eq_sqrt_3_l667_667828

theorem cot_30_eq_sqrt_3 (θ : ℝ) (hθ : θ = 30) (h1 : tan 30 = 1 / real.sqrt 3) : cot 30 = real.sqrt 3 :=
by
  have h2: cot 30 = 1 / (tan 30), by exact cot_def
  have h3: cot 30 = 1 / (1 / real.sqrt 3), by rw [h1, h2]
  have h4: cot 30 = real.sqrt 3, by exact (one_div_div (1 / real.sqrt 3))
  exact h4


end cot_30_eq_sqrt_3_l667_667828


namespace least_three_digit_multiple_of_3_4_7_l667_667664

theorem least_three_digit_multiple_of_3_4_7 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 7 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 → m % 3 = 0 ∧ m % 4 = 0 ∧ m % 7 = 0 → n ≤ m :=
  sorry

end least_three_digit_multiple_of_3_4_7_l667_667664


namespace general_formula_T_greater_S_l667_667895

section arithmetic_sequence

variable {a : ℕ → ℤ} -- Define the arithmetic sequence {a_n}
variable {b : ℕ → ℤ} -- Define the sequence {b_n}
variable {S : ℕ → ℤ} -- Define the sum sequence {S_n}
variable {T : ℕ → ℤ} -- Define the sum sequence {T_n}
variable (a1 : ℤ) (d : ℤ) -- Variables for the first term and common difference 

-- Conditions
def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n, a n = a1 + n * d

def sequence_b (b a : ℕ → ℤ) : Prop :=
  ∀ n, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_sequence_S (S a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (2 * a1 + (n - 1) * d)) / 2

def sum_sequence_T (T b : ℕ → ℤ) : Prop :=
  ∀ n, T n = ∑ i in Finset.range n, b i

def conditions (a b S T : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  arithmetic_sequence a a1 d ∧ sequence_b b a ∧ sum_sequence_S S a ∧ sum_sequence_T T b ∧ S 4 = 32 ∧ T 3 = 16

-- Problem 1: Prove the general formula for {a_n}
theorem general_formula (a : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, a n = 2 * n + 3 :=
sorry

-- Problem 2: Prove that for n > 5, T_n > S_n
theorem T_greater_S (a b S T : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, n > 5 → T n > S n :=
sorry

end arithmetic_sequence

end general_formula_T_greater_S_l667_667895


namespace sum_of_digits_y_C_l667_667107

theorem sum_of_digits_y_C :
  ∀ (a b : ℝ), 
    a ≠ b → 
    (2 * a^2 = 2 * b^2) → 
    (∃ x y z : ℝ, a * b * 1 / 2 = 250) → 
    let C := ((a + b) / 2, 2 * ((a + b) / 2)^2) in 
    (C.snd.digits.sum = 5) := sorry

end sum_of_digits_y_C_l667_667107


namespace cot_30_eq_sqrt_3_l667_667830

theorem cot_30_eq_sqrt_3 (θ : ℝ) (hθ : θ = 30) (h1 : tan 30 = 1 / real.sqrt 3) : cot 30 = real.sqrt 3 :=
by
  have h2: cot 30 = 1 / (tan 30), by exact cot_def
  have h3: cot 30 = 1 / (1 / real.sqrt 3), by rw [h1, h2]
  have h4: cot 30 = real.sqrt 3, by exact (one_div_div (1 / real.sqrt 3))
  exact h4


end cot_30_eq_sqrt_3_l667_667830


namespace cost_per_mile_first_plan_l667_667045

theorem cost_per_mile_first_plan 
  (initial_fee : ℝ) (cost_per_mile_first : ℝ) (cost_per_mile_second : ℝ) (miles : ℝ)
  (h_first : initial_fee = 65)
  (h_cost_second : cost_per_mile_second = 0.60)
  (h_miles : miles = 325)
  (h_equal_cost : initial_fee + miles * cost_per_mile_first = miles * cost_per_mile_second) :
  cost_per_mile_first = 0.40 :=
by
  sorry

end cost_per_mile_first_plan_l667_667045


namespace interesting_arrays_count_l667_667441

def is_interesting_array (n : ℕ) (T : Array (ℕ × ℕ)) : Prop :=
  (T.size == 2 * n) ∧
  (∀ p q, 1 ≤ p → p < q → q ≤ n →
    (T[2*p - 2].fst, T[2*p - 2].snd, T[2*q - 1].snd) ≠ (1, 0, 1) ∧
    (T[2*q - 2].fst, T[2*q - 2].snd, T[2*p - 2].fst) ≠ (1, 0, 0))

theorem interesting_arrays_count (n : ℕ) (h : 2 ≤ n) :
  ∃ count, count = 3^n + (n * (n + 1)) / 2 :=
sorry

end interesting_arrays_count_l667_667441


namespace cot_30_eq_sqrt3_l667_667787

theorem cot_30_eq_sqrt3 : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by
  have h_cos : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  have h_sin : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  show Real.cot (30 * Real.pi / 180) = Real.sqrt 3
  calc
    Real.cot (30 * Real.pi / 180) = (Real.cos (30 * Real.pi / 180)) / (Real.sin (30 * Real.pi / 180)) := sorry
    ... = (Real.sqrt 3 / 2) / (1 / 2) := sorry
    ... = Real.sqrt 3 := sorry

end cot_30_eq_sqrt3_l667_667787


namespace smallest_k_l667_667559

noncomputable def k_value (p : ℕ) : ℕ :=
  if (is_prime p) ∧ (digit_count p = 2023) then 1 else 0

theorem smallest_k (p : ℕ) (hp1 : is_prime p) (hp2 : digit_count p = 2023) : 
  ∃ k : ℕ, k = 1 ∧ (p^3 - k) % 30 = 0 :=
by
  use k_value p
  have h_prime_digits := (hp1, hp2)
  have k := k_value p
  rw k_value at k
  simp only [h_prime_digits] at k
  rw k
  sorry  -- Proof steps would follow here

end smallest_k_l667_667559


namespace general_formula_a_n_T_n_greater_S_n_l667_667872

-- Define the conditions for the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ (n : ℕ), a n = a1 + (n - 1) * d

def b_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_of_first_n_terms (a S : ℕ → ℕ) :=
  ∀ (n k : ℕ), k ≤ n → S n = (n * (a 1 + a n)) / 2

-- Given S4 = 32 and T3 = 16
def S_4_equals_32 (S : ℕ → ℕ) : Prop := S 4 = 32
def T_3_equals_16 (T : ℕ → ℕ) : Prop := T 3 = 16

-- Prove that the general formula for a_n is 2n + 3
theorem general_formula_a_n (a : ℕ → ℕ) (h : arithmetic_sequence a) : 
  a = λ n, 2 * n + 3 := 
sorry

-- Prove that for n > 5, T_n > S_n
theorem T_n_greater_S_n (a b S T: ℕ → ℕ) 
  (ha : arithmetic_sequence a) 
  (hb : b_sequence a b)
  (hS : sum_of_first_n_terms a S)
  (hT : sum_of_first_n_terms b T)
  (hS4 : S_4_equals_32 S)
  (hT3 : T_3_equals_16 T) 
  (n : ℕ) : n > 5 → T n > S n :=
sorry

end general_formula_a_n_T_n_greater_S_n_l667_667872


namespace simplify_expression_l667_667599

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l667_667599


namespace arithmetic_mean_neg6_to_7_l667_667273

theorem arithmetic_mean_neg6_to_7 :
  let seq := list.range' (-6) 14 in
  (seq.sum : ℤ) / seq.length = 0.5 :=
by
  let seq := list.range' (-6) 14
  have h_length : seq.length = 14 := by rfl
  have h_sum : seq.sum = 7 := by sorry
  rw [h_sum, h_length]
  norm_num

end arithmetic_mean_neg6_to_7_l667_667273


namespace maximize_profit_at_25_units_l667_667704

-- Definitions from conditions
def cost (x : ℕ) := 1200 + (2 / 75) * x ^ 3
def unit_price (x : ℕ) (k : ℝ) := real.sqrt (k / x)
def total_profit (x : ℕ) (k : ℝ) := x * (500 / real.sqrt x) - 1200 - (2 / 75) * x ^ 3

-- Given values
def k_value : ℝ := 25 * 10 ^ 4
def unit_price_for_100_units : ℝ := 50

-- Theorem stating the problem
theorem maximize_profit_at_25_units : ∃ x : ℕ, x = 25 ∧ 
  (∀ y : ℕ, total_profit x k_value ≥ total_profit y k_value) :=
by
  sorry

end maximize_profit_at_25_units_l667_667704


namespace choir_problem_l667_667320

theorem choir_problem : 
  let num_ways := 
    (∑ (k : ℕ) in {0, 1, 2, 3}, nat.choose 7 k * nat.choose 9 (k + 4)) +
    (∑ (k : ℕ) in finset.range (7 + 1), nat.choose 7 k * nat.choose 9 k - 1) +
    (∑ (k : ℕ) in {0, 1, 2, 3, 4, 5}, nat.choose 7 (k + 4) * nat.choose 9 k) + 9
  in num_ways % 100 = 28 := 
by {
  sorry
}

end choir_problem_l667_667320


namespace find_solutions_l667_667407

def is_solution (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ Int.gcd (Int.gcd a b) c = 1 ∧
  (a + b + c) ∣ (a^12 + b^12 + c^12) ∧
  (a + b + c) ∣ (a^23 + b^23 + c^23) ∧
  (a + b + c) ∣ (a^11004 + b^11004 + c^11004)

theorem find_solutions :
  (is_solution 1 1 1) ∧ (is_solution 1 1 4) ∧ 
  (∀ a b c : ℕ, is_solution a b c → 
    (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 4)) := 
sorry

end find_solutions_l667_667407


namespace cot_30_eq_sqrt3_l667_667781

theorem cot_30_eq_sqrt3 : Real.cot (π / 6) = Real.sqrt 3 := by
  let tan_30 := 1 / Real.sqrt 3
  have tan_30_eq : Real.tan (π / 6) = tan_30 := sorry
  have cot_def : ∀ x, Real.cot x = 1 / Real.tan x := sorry
  rw [cot_def (π / 6), tan_30_eq]
  simp
  sorry

end cot_30_eq_sqrt3_l667_667781


namespace solution_set_inequality_l667_667457

-- Conditions
variable {f : ℝ → ℝ}
variable {f'} : ℝ → ℝ 
variable (h_dom : ∀ x, x > 0 → f x ∈ (0, +∞))
variable (h_deriv : ∀ x, f' x = (differentiable f x).deriv)
variable (h_deriv_cond : ∀ x, x > 0 → x * f' x > f x)

-- Question: Prove the solution set for the inequality (x-1)f(x+1) > f(x^2-1) is (1,2)
theorem solution_set_inequality (h_s : ∀ x, x > 0 → (x - 1) * f (x + 1) > f (x^2 - 1)) : 
  setOf (λ x, (x - 1) * f (x + 1) > f (x^2 - 1) ∧ x > 1) = setOf (λ x, 1 < x ∧ x < 2) :=
sorry

end solution_set_inequality_l667_667457


namespace find_S2019_l667_667941

-- Conditions given in the problem
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Definitions and conditions extracted: conditions for sum of arithmetic sequence
axiom arithmetic_sum (n : ℕ) : S n = n * a (n / 2)
axiom OB_condition : a 3 + a 2017 = 1

-- Lean statement to prove S2019
theorem find_S2019 : S 2019 = 2019 / 2 := by
  sorry

end find_S2019_l667_667941


namespace log_base_5_of_reciprocal_of_25_l667_667385

theorem log_base_5_of_reciprocal_of_25 : log 5 (1 / 25) = -2 :=
by
-- Sorry this proof is omitted
sorry

end log_base_5_of_reciprocal_of_25_l667_667385


namespace prove_inequality_l667_667707

variable (f : ℝ → ℝ)

-- Conditions
axiom condition : ∀ x y : ℝ, (f x + y) * (f y + x) > 0 → f x + y = f y + x

-- Proof of the desired statement
theorem prove_inequality : ∀ x y : ℝ, x > y → f x + y ≤ f y + x :=
by
  sorry

end prove_inequality_l667_667707


namespace pq_proof_l667_667910

section ProofProblem

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, ∃ d, a (n + 1) = a n + d

def b_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
if n % 2 = 1 then a n - 6 else 2 * a n

def sum_first_n (seq : ℕ → ℤ) (n : ℕ) : ℤ :=
(n.sum (λ i, seq (i + 1)))

variables (a : ℕ → ℤ)
variables (S T : ℕ → ℤ)

axiom a_is_arithmetic : arithmetic_sequence a
axiom S_4_eq_32 : sum_first_n a 4 = 32
axiom T_3_eq_16 : sum_first_n (b_n a) 3 = 16

theorem pq_proof : (∀ n, a n = 2 * n + 3) ∧ (∀ n > 5, sum_first_n (b_n a) n > sum_first_n a n) :=
sorry

end ProofProblem

end pq_proof_l667_667910


namespace range_of_a_l667_667952

noncomputable def f (x a : ℝ) : ℝ := 2 * x + a

noncomputable def g (x : ℝ) : ℝ := Real.log x - 2 * x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 ∈ set.Icc (1/2 : ℝ) 2, f x1 a ≤ g x2) → a ≤ Real.log 2 - 5 := 
by
  sorry

end range_of_a_l667_667952


namespace angle_EKP_correct_l667_667531

-- Variables for the vertices of triangle DEF
variables {D E F P Q K : Type}
-- Angles in degrees
variables (angle_DFE : ℝ) (angle_DEF : ℝ)

-- The given conditions
def conditions_triangle_DEF (D E F P Q K : Type) (angle_DFE : ℝ) (angle_DEF : ℝ) : Prop :=
  ∃ (D E F P Q K : Type), 
    angle_DFE = 64 ∧
    angle_DEF = 67 ∧
    -- additional conditions about the orthocenter and altitudes intersection
    (orthocenter DEF K) ∧ 
    (altitude D P K) ∧ 
    (altitude E Q K)

-- The angle we need to prove
def angle_EKP : ℝ := 41

-- Main statement
theorem angle_EKP_correct (D E F P Q K : Type) (angle_DFE : ℝ) (angle_DEF : ℝ) 
  (h : conditions_triangle_DEF D E F P Q K angle_DFE angle_DEF) : 
  ∠EKP = angle_EKP 
  :=
sorry

end angle_EKP_correct_l667_667531


namespace train_speed_is_60_kmh_l667_667726

def length_of_train := 116.67 -- in meters
def time_to_cross_pole := 7 -- in seconds

def length_in_km := length_of_train / 1000 -- converting meters to kilometers
def time_in_hours := time_to_cross_pole / 3600 -- converting seconds to hours

theorem train_speed_is_60_kmh :
  (length_in_km / time_in_hours) = 60 :=
sorry

end train_speed_is_60_kmh_l667_667726


namespace numeral_sevens_place_value_diff_l667_667274

theorem numeral_sevens_place_value_diff (A B : ℕ) (h₁ : A - B = 69930) (h₂ : A = 10 * B) : 
  ∃ n : ℕ, n = 7700070 :=
begin
  sorry
end

end numeral_sevens_place_value_diff_l667_667274


namespace proof_problem_l667_667477

def line_l (x y : ℝ) : Prop := x - y + 1 = 0
def curve_C (x y : ℝ) : Prop := y = x^2

def M : ℝ × ℝ := (1/2 - (Real.sqrt 5)/2 , 1/2 - (Real.sqrt 5)/2)
def A : ℝ × ℝ := (1/2 - (Real.sqrt 5)/2 , 3/2 - (Real.sqrt 5)/2)
def B : ℝ × ℝ := (1/2 + (Real.sqrt 5)/2 , 3/2 + (Real.sqrt 5)/2)

noncomputable def distance_squared (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem proof_problem :
  let MA2 := distance_squared M A in
  let MB2 := distance_squared M B in
  MA2 * MB2 = 11 + 2 * Real.sqrt 5 :=
by sorry

end proof_problem_l667_667477


namespace part1_and_part2_l667_667888

-- Define the arithmetic sequence {a_n}
def a (n : Nat) : Nat := 2 * n + 3

-- Define the sequence {b_n}
def b (n : Nat) : Nat :=
  if n % 2 = 0 then 4 * n + 6 else 2 * n - 3

-- Define the sum of the first n terms of a sequence
def summation (seq : Nat → Nat) (n : Nat) : Nat :=
  (List.range n).map seq |>.sum

-- Define S_n as the sum of the first n terms of {a_n}
def S (n : Nat) : Nat := summation a n

-- Define T_n as the sum of the first n terms of {b_n}
def T (n : Nat) : Nat := summation b n

-- Given conditions
axiom S4_eq_32 : S 4 = 32
axiom T3_eq_16 : T 3 = 16

-- Prove the general formula for {a_n} and that T_n > S_n for n > 5
theorem part1_and_part2 (n : Nat) (h : n > 5) : a n = 2 * n + 3 ∧ T n > S n :=
  by
  sorry

end part1_and_part2_l667_667888


namespace chimney_bricks_l667_667749

theorem chimney_bricks :
  ∃ (x : ℕ), (x = 900) ∧ 
             (1 / 9 * x) + (1 / 10 * x) - 10) * 5 = x :=
by
  sorry

end chimney_bricks_l667_667749


namespace find_speed_second_car_l667_667247

noncomputable def speed_second_car := {v : ℝ | v = 60 ∨ v = 140}

theorem find_speed_second_car (d_AB : ℝ) (speed_A : ℝ) (t : ℝ) (d_after_two_hours : ℝ)
  (h1 : d_AB = 300) (h2 : speed_A = 40) (h3 : t = 2) (h4 : d_after_two_hours = 100) :
  ∃ v ∈ speed_second_car, true :=
begin
  sorry -- Proof is omitted
end

end find_speed_second_car_l667_667247


namespace value_of_a_l667_667130

theorem value_of_a (a : ℝ) (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-1, 1, 2}) 
  (hB : B = {a + 1, a ^ 2 + 3}) 
  (h_inter : A ∩ B = {2}) : 
  a = 1 := 
by sorry

end value_of_a_l667_667130


namespace simplify_expression_l667_667600

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l667_667600


namespace original_solution_percentage_l667_667703

theorem original_solution_percentage (P : ℝ) (h1 : 0.5 * P + 0.5 * 30 = 40) : P = 50 :=
by
  sorry

end original_solution_percentage_l667_667703


namespace constant_value_expression_l667_667476

-- Given conditions
variables (p : ℝ) (hp : p > 0) (H : ℝ × ℝ) (H_fixed : H = (3, 0))
def parabola_equation : (y : ℝ) → (x : ℝ) → Prop := λ y x, y^2 = 2 * p * x
def focus : ℝ × ℝ := (p / 2, 0)

-- Main statement to be proven
theorem constant_value_expression (h_p : p = 3) (H_intersects_parabola : ∀ (m : ℝ), ∃ y1 y2, parabola_equation p y1 (m * y1 + 3) 
                                                                         ∧ parabola_equation p y2 (m * y2 + 3)) :
  let y1 := (λ m, some (H_intersects_parabola m)),
      y2 := (λ m, some (H_intersects_parabola m)),
      HM_squared := (λ m, (1 + m^2) * (y1 m)^2),
      HN_squared := (λ m, (1 + m^2) * (y2 m)^2),
      MN_squared := (λ m, (1 + m^2) * (y1 m - y2 m)^2) in
  ∀ m : ℝ, y1 m * y2 m = -18 ∧ y1 m + y2 m = 6 * m →
  (HM_squared m * HN_squared m) / (MN_squared m - 2 * (HM_squared m * HN_squared m)^0.5) = 9 :=
begin
  sorry
end

end constant_value_expression_l667_667476


namespace value_of_c_l667_667287

theorem value_of_c (c : ℝ) : (∀ x : ℝ, (-x^2 + c*x + 12 > 0 ↔ x ∈ Set.Ioo (-∞) (-3) ∪ Set.Ioo 4 ∞)) → c = 1 :=
by
  intro h
  sorry

end value_of_c_l667_667287


namespace length_MN_eq_600_l667_667529

-- Condition definitions based on the problem statement
variables (A B C D E M N : Type)
variables (BC AD : ℝ)
variables [hBC : BC = 1200] [hAD : AD = 2400]
variables [h1 : ∠A = 45] [h2 : ∠D = 45] [hMN : M ∈ BC ∧ N ∈ AD ∧ midpoint(M, BC) ∧ midpoint(N, AD)]

-- Length of MN proof objective
theorem length_MN_eq_600 : MN = 600 :=
sorry

end length_MN_eq_600_l667_667529


namespace abc_sum_square_identity_l667_667016

theorem abc_sum_square_identity (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 941) (h2 : a + b + c = 31) :
  ab + bc + ca = 10 :=
by
  sorry

end abc_sum_square_identity_l667_667016


namespace S_5_equals_31_l667_667437

-- Define the sequence sum function S
def S (n : Nat) : Nat := 2^n - 1

-- The theorem to prove that S(5) = 31
theorem S_5_equals_31 : S 5 = 31 :=
by
  rw [S]
  sorry

end S_5_equals_31_l667_667437


namespace smallest_k_l667_667767

variable (A B C D A1 B1 C1 D1 : Point)
variable [ConvexQuadrilateral ABCD]
variable (A1_on_AB : OnSegment A B A1)
variable (B1_on_BC : OnSegment B C B1)
variable (C1_on_CD : OnSegment C D C1)
variable (D1_on_DA : OnSegment D A D1)

def area_triangle (X Y Z : Point) : ℝ := sorry
def area_quadrilateral (W X Y Z : Point) : ℝ := sorry

noncomputable def S : ℝ :=
  min (area_triangle A A1 D1) (min (area_triangle B B1 A1) (min (area_triangle C C1 B1) (area_triangle D D1 C1))) +
  sorry -- The second smallest triangle's area, assuming it's understood it's coded similarly

noncomputable def S1 : ℝ := area_quadrilateral A1 B1 C1 D1

theorem smallest_k : ∀ (A B C D A1 B1 C1 D1 : Point) 
  [ConvexQuadrilateral ABCD]
  (A1_on_AB : OnSegment A B A1)
  (B1_on_BC : OnSegment B C B1)
  (C1_on_CD : OnSegment C D C1)
  (D1_on_DA : OnSegment D A D1),
  S1 ≥ S := sorry

end smallest_k_l667_667767


namespace mary_paid_for_berries_l667_667570

theorem mary_paid_for_berries : 
  let T := 20.00 - 5.98 in
  T - 6.83 = 7.19 :=
by
  sorry

end mary_paid_for_berries_l667_667570


namespace first_group_hours_per_day_l667_667493

theorem first_group_hours_per_day (X : ℕ) :
  (10 * 20 * X * 3 = 600 * X) ∧
  (30 * 32 * 8 * 2 = 7680) →
  (600 * X = 7680) ↔ X = 12.8 :=
by
  intro h
  sorry

end first_group_hours_per_day_l667_667493


namespace karen_max_avg_speed_l667_667099

theorem karen_max_avg_speed {s e : ℕ} (h1 : s = 12321) (h2 : (∀ x, Nat.digits 10 x = Nat.digits 10 x.reverse) → (Nat.digits 10 e = Nat.digits 10 e.reverse)) (h3 : e - s ≤ 260) : (e - s) / 4 = 50 :=
by
  sorry

end karen_max_avg_speed_l667_667099


namespace a_n_sequence_term2015_l667_667221

theorem a_n_sequence_term2015 :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ a 2 = 1/2 ∧ (∀ n ≥ 2, a n * (a (n-1) + a (n+1)) = 2 * a (n+1) * a (n-1)) ∧ a 2015 = 1/2015 :=
sorry

end a_n_sequence_term2015_l667_667221


namespace inequality_proof_l667_667549

variable (a b : ℝ)

theorem inequality_proof (h : a ≥ b ∧ b ≥ 0) :
  (Real.sqrt (a^2 + b^2) + Real.cbrt (a^3 + b^3) + Real.quadrt (a^4 + b^4) ≤ 3 * a + b) :=
sorry

end inequality_proof_l667_667549


namespace inscribed_squares_area_relation_l667_667161

theorem inscribed_squares_area_relation
    (ABCD_area : ℝ)
    (I_on_CD : I ∈ segment C D)
    (J_on_CD : J ∈ segment C D)
    (K_on_circle : K ∈ circle O (sqrt 2))
    (L_on_circle : L ∈ circle O (sqrt 2))
    (H_area : area body_sq ABCD = 4) :
    ∃ (p q : ℕ), p < q ∧ Nat.gcd p q = 1 ∧ (area body_sq IJKL) = p / q ∧ 10 * q + p = 251 := 
sorry

end inscribed_squares_area_relation_l667_667161


namespace percentage_markup_correct_l667_667213

-- Define the selling price and cost price
def selling_price : ℝ := 3600
def cost_price : ℝ := 3000

-- Define the markup as the difference between selling price and cost price
def markup : ℝ := selling_price - cost_price

-- Define the percentage markup
def percentage_markup : ℝ := (markup / cost_price) * 100

-- The proof statement to be proved
theorem percentage_markup_correct :
  percentage_markup = 20 := by
  sorry

end percentage_markup_correct_l667_667213


namespace mary_baseball_cards_l667_667136

variable (original_cards : ℕ) (torn_cards : ℕ) (new_cards_from_fred : ℕ) (bought_cards : ℕ)

theorem mary_baseball_cards :
  (original_cards = 18) →
  (torn_cards = 8) →
  (new_cards_from_fred = 26) →
  (bought_cards = 40) →
  (original_cards - torn_cards + new_cards_from_fred + bought_cards = 76) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact dec_trivial

end mary_baseball_cards_l667_667136


namespace complex_diff_arg_min_120_l667_667089

open Complex

theorem complex_diff_arg_min_120 (n : ℕ) (Z : Fin n → ℂ) (h : (∑ k, Z k) = 0) :
  ∃ i j : Fin n, (∠(Z i / Z j)).angle.toReal ≥ 120 :=
begin
  sorry
end

end complex_diff_arg_min_120_l667_667089


namespace unique_function_satisfying_conditions_l667_667406

theorem unique_function_satisfying_conditions (f : ℤ → ℤ) :
  (∀ n : ℤ, f (f n) + f n = 2 * n + 3) → 
  (f 0 = 1) → 
  (∀ n : ℤ, f n = n + 1) :=
by
  intro h1 h2
  sorry

end unique_function_satisfying_conditions_l667_667406


namespace calories_in_300g_lemonade_proof_l667_667568

def g_lemon := 150
def g_sugar := 200
def g_water := 450

def c_lemon_per_100g := 30
def c_sugar_per_100g := 400
def c_water := 0

def total_calories :=
  g_lemon * c_lemon_per_100g / 100 +
  g_sugar * c_sugar_per_100g / 100 +
  g_water * c_water

def total_weight := g_lemon + g_sugar + g_water

def caloric_density := total_calories / total_weight

def calories_in_300g_lemonade := 300 * caloric_density

theorem calories_in_300g_lemonade_proof : calories_in_300g_lemonade = 317 := by
  sorry

end calories_in_300g_lemonade_proof_l667_667568


namespace cot_30_eq_sqrt_3_l667_667822

theorem cot_30_eq_sqrt_3 :
  let θ := 30.0
  let cos θ := (real.sqrt 3) / 2
  let sin θ := 1 / 2
  let cot θ := cos θ / sin θ
  cot θ = real.sqrt 3 := by
sorry

end cot_30_eq_sqrt_3_l667_667822


namespace functional_equation_solution_form_l667_667405

noncomputable def functional_equation_problem (f : ℝ → ℝ) :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

theorem functional_equation_solution_form :
  (∀ f : ℝ → ℝ, (functional_equation_problem f) → (∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ 2 + b * x)) :=
by 
  sorry

end functional_equation_solution_form_l667_667405


namespace angle_between_planes_90_degrees_l667_667342

noncomputable def regular_tetrahedron (A B C D : Point) : Prop :=
  regular_tetrahedron A B C D

noncomputable def sphere_circumscribed (A B C D : Point) (sphere : Sphere) : Prop :=
  sphere_circumscribed A B C D sphere

noncomputable def pyramid_constructed (A B C D A' B' C' D' : Point) : Prop :=
  pyramid_constructed A B C D A' B' C' D'

theorem angle_between_planes_90_degrees
(A B C D A' B' C' D' : Point) (sphere : Sphere)
(h1 : regular_tetrahedron A B C D)
(h2 : sphere_circumscribed A B C D sphere)
(h3 : pyramid_constructed A B C D A' B' C' D') :
  angle_between_planes (plane_of_points A B C) (plane_of_points A C D') = 90 :=
by
  sorry

end angle_between_planes_90_degrees_l667_667342


namespace simplify_expression_l667_667598

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l667_667598


namespace evaluate_expression_l667_667489

theorem evaluate_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(2*x)) / (y^(2*y) * x^(2*x)) = (x / y)^(2 * (y - x)) :=
by
  sorry

end evaluate_expression_l667_667489


namespace total_cost_of_movie_visit_l667_667423

-- Let define all conditions as Lean variables and constants

def ticket_cost := 16
def nachos_cost := ticket_cost / 2
def popcorn_cost := nachos_cost * 0.75
def soda_cost := popcorn_cost * 0.75
def total_food_before_discount := nachos_cost + popcorn_cost + soda_cost
def discount := total_food_before_discount * 0.10
def total_food_after_discount := total_food_before_discount - discount
def total_purchase_before_tax := ticket_cost + total_food_after_discount
def sales_tax := total_purchase_before_tax * 0.05
def rounded_sales_tax := 1.63  -- rounding to nearest cent
def final_total_cost := total_purchase_before_tax + rounded_sales_tax

-- Proof statement
theorem total_cost_of_movie_visit :
  final_total_cost = 34.28 :=
by
  sorry

end total_cost_of_movie_visit_l667_667423


namespace tangent_line_value_at_2_l667_667696

theorem tangent_line_value_at_2
  (f : ℝ → ℝ)
  (tangent_line_at_2 : ∀ x : ℝ, 2 * x + f(2) - 3 = 0) :
  f(2) + deriv f 2 = -3 := 
  sorry

end tangent_line_value_at_2_l667_667696


namespace min_sum_squares_difference_rem_l667_667620

theorem min_sum_squares_difference_rem :
  let squares := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19].map (λ n, n^2)
  let sum_squares := squares.sum
  let remainder := sum_squares % 4
  sum_squares - remainder = 1328 :=
by
  let squares := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19].map (λ n, n^2)
  let sum_squares := squares.sum
  let remainder := sum_squares % 4
  have : sum_squares = 1330 := by sorry
  have : remainder = 2 := by sorry
  calc
    sum_squares - remainder
    = 1330 - 2 := by rw [‹sum_squares = 1330›, ‹remainder = 2›]
    ... = 1328 := by norm_num

end min_sum_squares_difference_rem_l667_667620


namespace cot_30_eq_sqrt_3_l667_667824

theorem cot_30_eq_sqrt_3 :
  let θ := 30.0
  let cos θ := (real.sqrt 3) / 2
  let sin θ := 1 / 2
  let cot θ := cos θ / sin θ
  cot θ = real.sqrt 3 := by
sorry

end cot_30_eq_sqrt_3_l667_667824


namespace base8_to_base10_conversion_l667_667094

theorem base8_to_base10_conversion : 
  (6 * 8^3 + 3 * 8^2 + 7 * 8^1 + 5 * 8^0) = 3325 := 
by 
  sorry

end base8_to_base10_conversion_l667_667094


namespace part_one_part_two_part_three_l667_667029

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x^2 - x) - Real.log x

theorem part_one :
  ∀ (a : ℝ), (∀ x : ℝ, x > 0 → 0 < x - 1 → D (f a) x = 0) → f a 1 < f a 2 → a = 1 :=
by
  sorry

theorem part_two :
  ∀ (a : ℝ), (∀ x : ℝ, x ≥ 1 → f a x ≥ 0) → a ≥ 1 :=
by
  sorry

theorem part_three :
  ∀ n : ℕ, n ≥ 2 → (∑ k in Finset.range n, (1 / Real.log (k + 2))) > (n - 1) / n :=
by
  sorry

end part_one_part_two_part_three_l667_667029


namespace determine_g_x2_l667_667555

noncomputable def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem determine_g_x2 (x : ℝ) (h : x^2 ≠ 4) : g (x^2) = (2 * x^2 + 3) / (x^2 - 2) :=
by sorry

end determine_g_x2_l667_667555


namespace cot_30_deg_l667_667816

-- Define the known values and identities
def tan_30_deg := 1 / Real.sqrt 3

theorem cot_30_deg : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by 
  sorry

end cot_30_deg_l667_667816


namespace sum_of_distinct_prime_factors_420_l667_667283

theorem sum_of_distinct_prime_factors_420 : (2 + 3 + 5 + 7 = 17) :=
by
  -- given condition on the prime factorization of 420
  have h : ∀ {n : ℕ}, n = 420 → ∀ (p : ℕ), p ∣ n → p.prime → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7,
  by sorry

  -- proof of summation of these distinct primes
  sorry

end sum_of_distinct_prime_factors_420_l667_667283


namespace general_formula_sums_inequality_for_large_n_l667_667905

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def modified_seq (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = (finset.range n).sum a

variables {a b S T : ℕ → ℕ}

-- Given conditions
axiom h_arith_seq : arithmetic_seq a
axiom h_modified_seq : modified_seq a b
axiom h_sum_a : sum_seq a S
axiom h_sum_b : sum_seq b T
axiom h_S4 : S 4 = 32
axiom h_T3 : T 3 = 16

-- Proof Statements
theorem general_formula :
  ∀ n : ℕ, a n = 2 * n + 3 := sorry

theorem sums_inequality_for_large_n :
  ∀ n : ℕ, n > 5 → T n > S n := sorry

end general_formula_sums_inequality_for_large_n_l667_667905


namespace boat_speed_in_still_water_l667_667224

variable (b r d v t : ℝ)

theorem boat_speed_in_still_water (hr : r = 3) 
                                 (hd : d = 3.6) 
                                 (ht : t = 1/5) 
                                 (hv : v = b + r) 
                                 (dist_eq : d = v * t) : 
  b = 15 := 
by
  sorry

end boat_speed_in_still_water_l667_667224


namespace scout_troop_profit_l667_667339

noncomputable def buy_price_per_bar : ℚ := 3 / 4
noncomputable def sell_price_per_bar : ℚ := 2 / 3
noncomputable def num_candy_bars : ℕ := 800

theorem scout_troop_profit :
  num_candy_bars * (sell_price_per_bar : ℚ) - num_candy_bars * (buy_price_per_bar : ℚ) = -66.64 :=
by
  sorry

end scout_troop_profit_l667_667339


namespace circle_plane_division_l667_667073

noncomputable def f : ℕ → ℕ :=
  sorry

theorem circle_plane_division (k : ℕ) :
  ∀ k : ℕ, ∀ f : ℕ → ℕ,
  ∀ h1 : ∀ n m : ℕ, n ≠ m → circles n ⊓ circles m = 2,
  ∀ h2 : ∀ i j k : ℕ, i ≠ j ∧ i ≠ k ∧ j ≠ k →
          ¬ ∃ P, P ∈ circles i ∧ P ∈ circles j ∧ P ∈ circles k,
  f (k + 1) = f k + 2 * k :=
sorry

-- Definitions used:
def circles (n : ℕ) : Set (Set Point) :=
  sorry

def Point : Type :=
  sorry

end circle_plane_division_l667_667073


namespace anne_equals_bob_l667_667219

-- Define the conditions as constants and functions
def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.06
def discount_rate : ℝ := 0.25

-- Calculation models for Anne and Bob
def anne_total (price : ℝ) (tax : ℝ) (discount : ℝ) : ℝ :=
  (price * (1 + tax)) * (1 - discount)

def bob_total (price : ℝ) (tax : ℝ) (discount : ℝ) : ℝ :=
  (price * (1 - discount)) * (1 + tax)

-- The theorem that states what we need to prove
theorem anne_equals_bob : anne_total original_price tax_rate discount_rate = bob_total original_price tax_rate discount_rate :=
by
  sorry

end anne_equals_bob_l667_667219


namespace average_of_integers_between_two_sevenths_and_one_sixth_l667_667660

theorem average_of_integers_between_two_sevenths_and_one_sixth :
  (∑ n in Finset.Icc 15 23, n) / 9 = 19 :=
by
  sorry

end average_of_integers_between_two_sevenths_and_one_sixth_l667_667660


namespace min_cylinder_volume_eq_surface_area_l667_667986

theorem min_cylinder_volume_eq_surface_area (r h V S : ℝ) (hr : r > 0) (hh : h > 0)
  (hV : V = π * r^2 * h) (hS : S = 2 * π * r^2 + 2 * π * r * h) (heq : V = S) :
  V = 54 * π :=
by
  -- Placeholder for the actual proof
  sorry

end min_cylinder_volume_eq_surface_area_l667_667986


namespace geometry_problem_l667_667550

-- Definitions for geometric elements
variables {A B C H M N X Y : Type*} 
variables [triangle : Triangle A B C] 
variables [orthocenter H A B C]
variables [on_segment M A B] [on_segment N A C]
variables [circle1 : Circle (segment B N), circle2 : Circle (segment C M)]
variables [intersect : Intersects circle1 circle2 X Y]

-- The statement to be proved
theorem geometry_problem :
  lies_on_line (segment X Y) H :=
sorry

end geometry_problem_l667_667550


namespace otto_assignment_l667_667581

noncomputable def otto_function (x y : ℝ) : ℝ := (x + y) / 2

theorem otto_assignment : otto_function 1975 1976 = 1975.5 :=
by
  -- Definitions from conditions to ensure equivalence:
  -- 1. Symmetry property
  have h_symm : ∀ (x y : ℝ), otto_function x y = otto_function y x := by
    intros x y
    unfold otto_function
    rw [add_comm]

  -- 2. Distribution over some operation
  have h_dist : ∀ (x y z : ℝ), otto_function (otto_function x y) z = otto_function (otto_function x z) (otto_function y z) := by
    intros x y z
    unfold otto_function
    linarith

  -- 3. Translation invariance
  have h_trans : ∀ (x y z : ℝ), otto_function x y + z = otto_function (x + z) (y + z) := by
    intros x y z
    unfold otto_function
    ring

  -- Now showing the function value for given problem
  show otto_function 1975 1976 = 1975.5,
  by smt
    unfold otto_function
    linarith

  sorry -- Finish the proof

end otto_assignment_l667_667581


namespace xiao_ming_excellent_score_probability_l667_667067

theorem xiao_ming_excellent_score_probability :
  let P_M : ℝ := 0.5
  let P_L : ℝ := 0.3
  let P_E := 1 - P_M - P_L
  P_E = 0.2 :=
by
  let P_M : ℝ := 0.5
  let P_L : ℝ := 0.3
  let P_E := 1 - P_M - P_L
  sorry

end xiao_ming_excellent_score_probability_l667_667067


namespace least_number_of_trees_l667_667674

theorem least_number_of_trees (n : ℕ) :
  (∃ k₄ k₅ k₆, n = 4 * k₄ ∧ n = 5 * k₅ ∧ n = 6 * k₆) ↔ n = 60 :=
by 
  sorry

end least_number_of_trees_l667_667674


namespace cot_30_eq_sqrt3_l667_667789

theorem cot_30_eq_sqrt3 : Real.cot (30 * Real.pi / 180) = Real.sqrt 3 :=
by
  have h_cos : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  have h_sin : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  show Real.cot (30 * Real.pi / 180) = Real.sqrt 3
  calc
    Real.cot (30 * Real.pi / 180) = (Real.cos (30 * Real.pi / 180)) / (Real.sin (30 * Real.pi / 180)) := sorry
    ... = (Real.sqrt 3 / 2) / (1 / 2) := sorry
    ... = Real.sqrt 3 := sorry

end cot_30_eq_sqrt3_l667_667789


namespace general_formula_a_n_T_n_greater_than_S_n_l667_667926

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n_l667_667926


namespace find_a_perpendicular_l667_667051

theorem find_a_perpendicular:
  ∃ a: ℝ, (∀ x y: ℝ, ax + y + 2 = 0 → 3x - y - 2 = 0 → - (a / 1) * (-3) = -1) → a = 2 / 3 :=
begin
  sorry
end

end find_a_perpendicular_l667_667051


namespace general_formula_a_n_T_n_greater_than_S_n_l667_667929

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n_l667_667929


namespace hyperbola_condition_sufficiency_l667_667617

theorem hyperbola_condition_sufficiency (k : ℝ) :
  (k > 3) → (∃ x y : ℝ, (x^2)/(3-k) + (y^2)/(k-1) = 1) :=
by
  sorry

end hyperbola_condition_sufficiency_l667_667617


namespace percentage_of_students_in_second_division_is_54_l667_667076

-- Definition of the problem conditions
def total_students : ℕ := 300
def first_division_percentage : ℝ := 28 / 100
def just_passed_students : ℕ := 54

-- Calculation
def first_division_students : ℕ := (first_division_percentage * total_students).to_nat
def first_or_second_division_students : ℕ := total_students - just_passed_students
def second_division_students : ℕ := first_or_second_division_students - first_division_students

-- Question: What percentage of students got second division?
def second_division_percentage : ℝ :=
  (second_division_students : ℝ) / (total_students : ℝ) * 100

-- Proof statement
theorem percentage_of_students_in_second_division_is_54 :
  second_division_percentage = 54 := by
  sorry

end percentage_of_students_in_second_division_is_54_l667_667076


namespace charge_per_action_figure_l667_667101

-- Definitions according to given conditions
def cost_of_sneakers : ℕ := 90
def saved_amount : ℕ := 15
def num_action_figures : ℕ := 10
def left_after_purchase : ℕ := 25

-- Theorem to prove the charge per action figure
theorem charge_per_action_figure : 
  (cost_of_sneakers - saved_amount + left_after_purchase) / num_action_figures = 10 :=
by 
  sorry

end charge_per_action_figure_l667_667101


namespace cot_30_eq_sqrt3_l667_667782

theorem cot_30_eq_sqrt3 : Real.cot (π / 6) = Real.sqrt 3 := by
  let tan_30 := 1 / Real.sqrt 3
  have tan_30_eq : Real.tan (π / 6) = tan_30 := sorry
  have cot_def : ∀ x, Real.cot x = 1 / Real.tan x := sorry
  rw [cot_def (π / 6), tan_30_eq]
  simp
  sorry

end cot_30_eq_sqrt3_l667_667782


namespace dot_product_a_b_c_l667_667425

variables {V : Type} [inner_product_space ℝ V]

variables (a b c : V)
variables (h1 : 3 • a + 4 • b + 5 • c = 0)
variables (h2 : ∥a∥ = 1)
variables (h3 : ∥b∥ = 1)
variables (h4 : ∥c∥ = 1)

theorem dot_product_a_b_c : inner_product_space ℝ V :=
by 
  have : 3 • a + 4 • b + 5 • c = 0 := h1
  have : ∥a∥ = 1 := h2
  have : ∥b∥ = 1 := h3
  have : ∥c∥ = 1 := h4
  sorry

end dot_product_a_b_c_l667_667425


namespace y_intercept_of_line_l667_667270

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 :=
by
  sorry

end y_intercept_of_line_l667_667270


namespace circle_center_l667_667623

theorem circle_center (A B : ℝ × ℝ) (hA : A = (2, -3)) (hB : B = (8, 9)) :
  let mid := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  in mid = (5, 3) :=
by
  -- Conditions for the endpoints
  have hA : A = (2, -3) := hA
  have hB : B = (8, 9) := hB

  -- specify the midpoint calculation
  let mid := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

  -- State the expected midpoint
  have hmid : mid = (5, 3) := by
    -- The calculation is straightforward, exact proof is not provided here
    sorry

  exact hmid

end circle_center_l667_667623


namespace recreation_spent_percent_l667_667545

variable (W : ℝ) -- Assume W is the wages last week

-- Conditions
def last_week_spent_on_recreation (W : ℝ) : ℝ := 0.25 * W
def this_week_wages (W : ℝ) : ℝ := 0.70 * W
def this_week_spent_on_recreation (W : ℝ) : ℝ := 0.50 * (this_week_wages W)

-- Proof statement
theorem recreation_spent_percent (W : ℝ) :
  (this_week_spent_on_recreation W / last_week_spent_on_recreation W) * 100 = 140 := by
  sorry

end recreation_spent_percent_l667_667545


namespace math_problem_proof_l667_667465

noncomputable def f (x : ℝ) (m : ℤ) : ℝ := x ^ (-2 * m ^ 2 + m + 3)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ {a b : ℝ}, a ∈ s → b ∈ s → a < b → f a < f b

def g (x a : ℝ) : ℝ := log a (x^2 - a*x)

theorem math_problem_proof (m : ℤ) (a : ℝ)
  (hf1: f x m = x ^ (-2 * m ^ 2 + m + 3))
  (hf2: is_even_function (λ x, f x m))
  (hf3: is_increasing_on (λ x, f x m) {x : ℝ | 0 < x})
  (ha1: 1 < a) (ha2: a < 2)
  : ∃ (m : ℤ), (m = 1 ∧ f x m = x^2 ∧ (∀ x, 2 ≤ x ∧ x ≤ 3 → is_increasing_on (g x a) {x | 2 ≤ x ∧ x ≤ 3})) :=
begin
  sorry
end

end math_problem_proof_l667_667465


namespace max_area_triangle_PAB_l667_667023

-- Definitions of the problem
def ellipse (x y : ℝ) : Prop := 3 * x^2 + y^2 = 6
def pointP : ℝ × ℝ := (1, sqrt 3)

-- Statement of the theorem
theorem max_area_triangle_PAB : 
  (∃ A B : ℝ × ℝ, ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ 
   ∀ A B : ℝ × ℝ, ellipse A.1 A.2 ∧ ellipse B.1 B.2 → 
   ∃ max_area: ℝ, max_area = sqrt 3) :=
sorry

end max_area_triangle_PAB_l667_667023


namespace complement_domain_l667_667368

-- Given conditions:
def f (x : ℝ) := (x + 1) / (x^2 - 3 * x + 2)

-- Quadratic polynomial used in the denominator
def quadratic (x : ℝ) := x^2 - 3 * x + 2

-- Define T as the domain of f
def T : set ℝ := {x ∈ ℝ | quadratic x ≠ 0}

-- Define the universal set U
def U : set ℝ := set.univ

-- The statement to prove
theorem complement_domain (x : ℝ) : x ∈ (U \ T) ↔ x = 1 ∨ x = 2 :=
by sorry

end complement_domain_l667_667368


namespace Tn_gt_Sn_l667_667880

-- Definitions of the sequences and initial conditions
def a : ℕ → ℕ := λ n, 2 * n + 3
def b (n : ℕ) : ℕ := if n % 2 = 1 then a n - 6 else 2 * a n
def S (n : ℕ) : ℕ := (n * (2 * n + 8))/2
def T : ℕ → ℕ
| 0     := 0
| (n+1) := T n + b (n + 1)

-- Given initial conditions
axiom S_4_eq_32 : S 4 = 32
axiom T_3_eq_16 : T 3 = 16
axiom a_general : ∀ n : ℕ, a n = 2 * n + 3

-- Proof of the main theorem
theorem Tn_gt_Sn (n : ℕ) (h : n > 5) : T n > S n := by
  sorry

end Tn_gt_Sn_l667_667880


namespace simplify_expression_l667_667606

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b ^ 2 + 2 * b) - 2 * b ^ 2 = 9 * b ^ 3 + 4 * b ^ 2 :=
by
  sorry

end simplify_expression_l667_667606


namespace expr_I_equals_expr_II_equals_l667_667355

noncomputable def calculate_expr_I : ℚ :=
  0.064 ^ (-1 / 3 : ℚ) - (-4 / 5 : ℚ) ^ 0 + 0.01 ^ (1 / 2 : ℚ)

noncomputable def calculate_expr_II : ℚ :=
  2 * log 5 + log 4 + log (sqrt real.exp 1)

theorem expr_I_equals :
  calculate_expr_I = (8 / 5 : ℚ) :=
by
  unfold calculate_expr_I
  sorry

theorem expr_II_equals :
  calculate_expr_II = (5 / 2 : ℚ) :=
by
  unfold calculate_expr_II
  sorry

end expr_I_equals_expr_II_equals_l667_667355


namespace player_A_wins_l667_667648

-- Define the 3x3 grid and 9 cards with real numbers
def grid := matrix (fin 3) (fin 3) ℝ
def cards : fin 9 → ℝ := sorry  -- The card values are given but not specified for the proof

-- Define the sum function for rows and columns
def row_sum (g : grid) (r : fin 3) : ℝ :=
  g r 0 + g r 1 + g r 2

def column_sum (g : grid) (c : fin 3) : ℝ :=
  g 0 c + g 1 c + g 2 c

-- Define Player A's and Player B's sum
def player_A_sum (g : grid) : ℝ :=
  row_sum g 0 + row_sum g 2

def player_B_sum (g : grid) : ℝ :=
  column_sum g 0 + column_sum g 2

-- The theorem to be proven
theorem player_A_wins (g : grid) : 
  (∃ optimal_placement : (fin 9) → (fin 3 × fin 3), 
    ∀ placement : (fin 9) → (fin 3 × fin 3), 
      player_A_sum (matrix.map sorry (optimal_placement sorry)) >= player_B_sum (matrix.map sorry (placement sorry))) :=
sorry

end player_A_wins_l667_667648


namespace additional_boys_l667_667307

theorem additional_boys (initial additional total : ℕ) 
  (h1 : initial = 22) 
  (h2 : total = 35) 
  (h3 : additional = total - initial) : 
  additional = 13 := 
by 
  rw [h1, h2]
  simp
  sorry

end additional_boys_l667_667307


namespace max_sum_is_38_l667_667133

-- Definition of the problem variables and conditions
def number_set : Set ℤ := {2, 3, 8, 9, 14, 15}
variable (a b c d e : ℤ)

-- Conditions translated to Lean
def condition1 : Prop := b = c
def condition2 : Prop := a = d

-- Sum condition to find maximum sum
def max_combined_sum : ℤ := a + b + e

theorem max_sum_is_38 : 
  ∃ a b c d e, 
    {a, b, c, d, e} ⊆ number_set ∧
    b = c ∧ 
    a = d ∧ 
    a + b + e = 38 :=
sorry

end max_sum_is_38_l667_667133


namespace ideal_sleep_hours_l667_667656

open Nat

theorem ideal_sleep_hours 
  (weeknight_sleep : Nat)
  (weekend_sleep : Nat)
  (sleep_deficit : Nat)
  (num_weeknights : Nat := 5)
  (num_weekend_nights : Nat := 2)
  (total_nights : Nat := 7) :
  weeknight_sleep = 5 →
  weekend_sleep = 6 →
  sleep_deficit = 19 →
  ((num_weeknights * weeknight_sleep + num_weekend_nights * weekend_sleep) + sleep_deficit) / total_nights = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end ideal_sleep_hours_l667_667656


namespace crayons_in_new_set_l667_667569

theorem crayons_in_new_set (initial_crayons : ℕ) (half_loss : ℕ) (total_after_purchase : ℕ) (initial_crayons_eq : initial_crayons = 18) (half_loss_eq : half_loss = initial_crayons / 2) (total_eq : total_after_purchase = 29) :
  total_after_purchase - (initial_crayons - half_loss) = 20 :=
by
  sorry

end crayons_in_new_set_l667_667569


namespace lowest_score_of_15_scores_is_44_l667_667187

theorem lowest_score_of_15_scores_is_44
  (mean_15 : ℕ → ℕ)
  (mean_15 = 90)
  (mean_13 : ℕ → ℕ)
  (mean_13 = 92)
  (highest_score : ℕ)
  (highest_score = 110)
  (sum15 : mean_15 * 15 = 1350)
  (sum13 : mean_13 * 13 = 1196) :
  (lowest_score : ℕ) (lowest_score = 44) :=
by
  sorry

end lowest_score_of_15_scores_is_44_l667_667187


namespace cot_30_eq_sqrt_3_l667_667832

theorem cot_30_eq_sqrt_3 (θ : ℝ) (hθ : θ = 30) (h1 : tan 30 = 1 / real.sqrt 3) : cot 30 = real.sqrt 3 :=
by
  have h2: cot 30 = 1 / (tan 30), by exact cot_def
  have h3: cot 30 = 1 / (1 / real.sqrt 3), by rw [h1, h2]
  have h4: cot 30 = real.sqrt 3, by exact (one_div_div (1 / real.sqrt 3))
  exact h4


end cot_30_eq_sqrt_3_l667_667832


namespace cot_30_eq_sqrt3_l667_667774

theorem cot_30_eq_sqrt3 : Real.cot (π / 6) = Real.sqrt 3 := by
  let tan_30 := 1 / Real.sqrt 3
  have tan_30_eq : Real.tan (π / 6) = tan_30 := sorry
  have cot_def : ∀ x, Real.cot x = 1 / Real.tan x := sorry
  rw [cot_def (π / 6), tan_30_eq]
  simp
  sorry

end cot_30_eq_sqrt3_l667_667774


namespace toms_dog_age_is_twelve_l667_667244

-- Definitions based on given conditions
def toms_cat_age : ℕ := 8
def toms_rabbit_age : ℕ := toms_cat_age / 2
def toms_dog_age : ℕ := 3 * toms_rabbit_age

-- The statement to be proved
theorem toms_dog_age_is_twelve : toms_dog_age = 12 := by
  sorry

end toms_dog_age_is_twelve_l667_667244


namespace train_speed_is_60_kmh_l667_667725

def length_of_train := 116.67 -- in meters
def time_to_cross_pole := 7 -- in seconds

def length_in_km := length_of_train / 1000 -- converting meters to kilometers
def time_in_hours := time_to_cross_pole / 3600 -- converting seconds to hours

theorem train_speed_is_60_kmh :
  (length_in_km / time_in_hours) = 60 :=
sorry

end train_speed_is_60_kmh_l667_667725


namespace num_remainders_p_squared_mod_210_l667_667737

theorem num_remainders_p_squared_mod_210 (p : ℕ) (hp : p.prime) (h : p > 7) :
  ∃ R : Finset ℕ, (∀ q ∈ R, q < 210) ∧ (∀ q1 q2 ∈ R, p^2 % 210 = q1 → p^2 % 210 = q2 → q1 = q2) ∧ R.card = 6 :=
by
  sorry

end num_remainders_p_squared_mod_210_l667_667737


namespace cot_30_eq_sqrt_3_l667_667826

theorem cot_30_eq_sqrt_3 :
  let θ := 30.0
  let cos θ := (real.sqrt 3) / 2
  let sin θ := 1 / 2
  let cot θ := cos θ / sin θ
  cot θ = real.sqrt 3 := by
sorry

end cot_30_eq_sqrt_3_l667_667826


namespace impossible_to_maintain_Gini_l667_667689

variables (X Y G0 Y' Z : ℝ)
variables (G1 : ℝ)

-- Conditions
axiom initial_Gini : G0 = 0.1
axiom proportion_poor : X = 0.5
axiom income_poor_initial : Y = 0.4
axiom income_poor_half : Y' = 0.2
axiom population_split : ∀ a b c : ℝ, (a + b + c = 1) ∧ (a = b ∧ b = c)
axiom Gini_constant : G1 = G0

-- Equation system representation final value post situation
axiom Gini_post_reform : 
  G1 = (1 / 2 - ((1 / 6) * 0.2 + (1 / 6) * (0.2 + Z) + (1 / 6) * (1 - 0.2 - Z))) / (1 / 2)

-- Proof problem: to prove inconsistency or inability to maintain Gini coefficient given the conditions
theorem impossible_to_maintain_Gini : false :=
sorry

end impossible_to_maintain_Gini_l667_667689


namespace reduced_price_per_dozen_bananas_l667_667336

-- Definitions
def original_price_per_dozen (P : ℝ) := P
def price_per_dozen_after_reduction (P : ℝ) := 0.6 * P
def bananas_per_rs (P : ℝ) := 40 / P
def bananas_with_reduced_price (P : ℝ) := 40 / (0.6 * P)
def additional_bananas (P : ℝ) := bananas_with_reduced_price P - bananas_per_rs P
def total_bananas (B : ℝ) := B + 66

-- Condition: Bananas at original and reduced price
axiom orig_bananas (P : ℝ) : (additional_bananas P = 66)

-- Desired proof goal
theorem reduced_price_per_dozen_bananas {P : ℝ} (h1 : orig_bananas P) : price_per_dozen_after_reduction P = 2.08 := 
sorry

end reduced_price_per_dozen_bananas_l667_667336


namespace range_of_distances_l667_667525

-- Definitions of points A and B in polar coordinates
def A_polar : ℝ × ℝ := (sqrt 3, π / 6)
def B_polar : ℝ × ℝ := (sqrt 3, π / 2)

-- Definition of the curve C in polar coordinates
def C_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos (θ - π / 3) ∧ ρ ≥ 0

-- Definition of the rectangular coordinates for conversion purposes
def polar_to_rect (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Problem Statement
theorem range_of_distances (M : ℝ → ℝ × ℝ) (hC : ∀ θ, M θ = polar_to_rect 1 (θ - π / 6)) :
  ∃ m M : ℝ, (∀ θ, 2 ≤ |M θ - polar_to_rect (sqrt 3) (π / 6)|^2 + |M θ - polar_to_rect (sqrt 3) (π / 2)|^2 ∧
                  |M θ - polar_to_rect (sqrt 3) (π / 6)|^2 + |M θ - polar_to_rect (sqrt 3) (π / 2)|^2 ≤ 6) :=
sorry

end range_of_distances_l667_667525


namespace sum_of_three_element_subsets_l667_667131

theorem sum_of_three_element_subsets (A : set ℤ) (hA : A = {a1, a2, a3, a4}) :
  let B := {∑ s in (A.subsets 3), (λ s, ∑ x in s, x)} in
  B = {-1, 3, 5, 8} → A = {-3, 0, 2, 6} :=
by
  intros
  let B := {∑ s in (A.subsets 3), (λ s, ∑ x in s, x)}
  have hB : B = {-1, 3, 5, 8} := by sorry
  have hSum : a1 + a2 + a3 + a4 = 5 := by sorry
  have hElems : A = {-3, 0, 2, 6} := by sorry
  exact hElems

end sum_of_three_element_subsets_l667_667131


namespace value_to_add_l667_667207

theorem value_to_add (a b c d : ℕ) (x : ℕ) 
  (h1 : a = 24) (h2 : b = 32) (h3 : c = 36) (h4 : d = 54)
  (h_lcm : Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = 864) :
  x + 857 = 864 → x = 7 := 
by
  intros h
  rw [h]
  exact rfl

end value_to_add_l667_667207


namespace bricks_needed_l667_667324

-- Definitions
def courtyard_length : ℝ := 25 
def courtyard_width : ℝ := 15 
def brick_length_cm : ℝ := 20 
def brick_width_cm : ℝ := 10 

-- Conversion factor from cm^2 to m^2
def cm2_to_m2 (cm2 : ℝ) : ℝ := cm2 / 10000

-- Area calculations
def courtyard_area : ℝ := courtyard_length * courtyard_width
def brick_area_cm2 : ℝ := brick_length_cm * brick_width_cm
def brick_area_m2 : ℝ := cm2_to_m2 brick_area_cm2 

-- Number of bricks required
def required_bricks : ℝ := courtyard_area / brick_area_m2

-- Statement of the theorem to be proved
theorem bricks_needed : required_bricks = 18750 := by
  sorry

end bricks_needed_l667_667324


namespace general_formula_sums_inequality_for_large_n_l667_667901

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def modified_seq (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = (finset.range n).sum a

variables {a b S T : ℕ → ℕ}

-- Given conditions
axiom h_arith_seq : arithmetic_seq a
axiom h_modified_seq : modified_seq a b
axiom h_sum_a : sum_seq a S
axiom h_sum_b : sum_seq b T
axiom h_S4 : S 4 = 32
axiom h_T3 : T 3 = 16

-- Proof Statements
theorem general_formula :
  ∀ n : ℕ, a n = 2 * n + 3 := sorry

theorem sums_inequality_for_large_n :
  ∀ n : ℕ, n > 5 → T n > S n := sorry

end general_formula_sums_inequality_for_large_n_l667_667901


namespace bernardo_larger_prob_l667_667747

open ProbabilityTheory

-- Definitions
def bernardo_set : Finset ℕ := (Finset.range 10)  -- {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def silvia_set : Finset ℕ := (Finset.range 9).erase 0  -- {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Main Statement
theorem bernardo_larger_prob :
  let P := (Finset.choose 2 (bernardo_set.card - 1)) / (Finset.choose 3 bernardo_set.card)
          + (1 - (Finset.choose 2 (bernardo_set.card - 1)) / (Finset.choose 3 bernardo_set.card))
            * (1 - (Finset.choose 2 (silvia_set.card - 1)) / (Finset.choose 2 bernardo_set.card)) in
  P = 41 / 90 :=
by
  sorry

end bernardo_larger_prob_l667_667747


namespace smallest_positive_period_of_f_range_of_f_in_interval_l667_667947

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * sin (x + π / 6)

theorem smallest_positive_period_of_f : ∃ T > 0, T = π ∧ ∀ x ∈ Icc 0 (2 * π), f (x + T) = f x := sorry

theorem range_of_f_in_interval : set.Icc 0 (1 + sqrt 3 / 2) = set.image f (set.Icc 0 (π / 2)) := sorry

end smallest_positive_period_of_f_range_of_f_in_interval_l667_667947


namespace T_gt_S_for_n_gt_5_l667_667917

-- Define the arithmetic sequence a_n
def a : ℕ → ℕ
| n => 2 * n + 3

-- Define the sequence b_n
def b : ℕ → ℕ
| n => if n % 2 = 1 then a n - 6 else 2 * (a n)

-- Sum of first n terms of a
def S (n : ℕ) : ℕ := (List.range n).sum (λ k => a (k + 1))

-- Sum of first n terms of b
def T (n : ℕ) : ℕ := (List.range n).sum (λ k => b (k + 1))

-- Given conditions in Lean
axiom S_4 : S 4 = 32
axiom T_3 : T 3 = 16

-- The theorem to prove
theorem T_gt_S_for_n_gt_5 (n : ℕ) (hn : n > 5) : T n > S n :=
  sorry

end T_gt_S_for_n_gt_5_l667_667917


namespace farmer_apples_after_giving_away_l667_667625

def initial_apples : ℕ := 127
def given_away_apples : ℕ := 88
def remaining_apples : ℕ := 127 - 88

theorem farmer_apples_after_giving_away : remaining_apples = 39 := by
  sorry

end farmer_apples_after_giving_away_l667_667625


namespace polynomial_inequality_l667_667119
noncomputable def p (x : ℝ) : ℝ := sorry

theorem polynomial_inequality
  (h1 : ∀ x : ℝ, p x = 0 → ∃ x1 x2 ... xn, x1 < x2 < ... < xn)
  (h2 : polynomial p) :
  (∃ p' p'': ℝ → ℝ, ∀ x : ℝ, (p' x)^2 ≥ p x * p'' x) :=
sorry

end polynomial_inequality_l667_667119


namespace general_formula_min_n_S_gt_2015_div_2016_l667_667436

-- Define the sequence {a_n} and the conditions a_1 = 2 and a_2 = 6
def a : ℕ → ℕ
| 1 := 2
| 2 := 6
| (n + 1) := a n + 2 * n + 2 

-- Prove the general formula for {a_n}
theorem general_formula (n : ℕ) : a n = n * n + n := sorry

-- Define S_n as the sum of the sequence {1 / a_n}
def S : ℕ → ℚ 
| n := (∑ i in finset.range n, 1 / (a (i+1)))

-- Prove the minimum value of n such that S_n > 2015 / 2016
theorem min_n_S_gt_2015_div_2016 (n : ℕ) : S n > 2015 / 2016 ↔ n > 2015 := sorry

end general_formula_min_n_S_gt_2015_div_2016_l667_667436


namespace Tn_gt_Sn_l667_667878

-- Definitions of the sequences and initial conditions
def a : ℕ → ℕ := λ n, 2 * n + 3
def b (n : ℕ) : ℕ := if n % 2 = 1 then a n - 6 else 2 * a n
def S (n : ℕ) : ℕ := (n * (2 * n + 8))/2
def T : ℕ → ℕ
| 0     := 0
| (n+1) := T n + b (n + 1)

-- Given initial conditions
axiom S_4_eq_32 : S 4 = 32
axiom T_3_eq_16 : T 3 = 16
axiom a_general : ∀ n : ℕ, a n = 2 * n + 3

-- Proof of the main theorem
theorem Tn_gt_Sn (n : ℕ) (h : n > 5) : T n > S n := by
  sorry

end Tn_gt_Sn_l667_667878


namespace order_of_a_b_c_l667_667115

def a := 2 ^ 0.3
def b := 0.3 ^ 2
def c := Real.log 0.3 / Real.log 2  -- Using change of base for log

theorem order_of_a_b_c : c < b ∧ b < a := 
by
  sorry

end order_of_a_b_c_l667_667115


namespace harry_morning_routine_time_l667_667402

-- Define the conditions in Lean.
def buy_coffee_and_bagel_time : ℕ := 15 -- minutes
def read_and_eat_time : ℕ := 2 * buy_coffee_and_bagel_time -- twice the time for buying coffee and bagel is 30 minutes

-- Define the total morning routine time in Lean.
def total_morning_routine_time : ℕ := buy_coffee_and_bagel_time + read_and_eat_time

-- The final proof problem statement.
theorem harry_morning_routine_time :
  total_morning_routine_time = 45 :=
by
  unfold total_morning_routine_time
  unfold read_and_eat_time
  unfold buy_coffee_and_bagel_time
  sorry

end harry_morning_routine_time_l667_667402


namespace number_of_points_visited_l667_667140

def point := (ℤ × ℤ)

def is_within_bounds (p : point) : Prop :=
  let (x, y) := p in
  |x + 4| + |x - 4| + |y - 3| + |y + 3| ≤ 22

def avoids_obstacle (p : point) : Prop :=
  let (x, y) := p in
  ¬(x = 0 ∧ -1 ≤ y ∧ y ≤ 1)

def possible_point (p : point) : Prop :=
  is_within_bounds p ∧ avoids_obstacle p

def points_visited := {p : point | possible_point p}

theorem number_of_points_visited : set.card points_visited = 206 :=
sorry

end number_of_points_visited_l667_667140


namespace truck_cargo_solution_l667_667332

def truck_cargo_problem (x : ℝ) (n : ℕ) : Prop :=
  (∀ (x : ℝ) (n : ℕ), x = (x / n - 0.5) * (n + 4)) ∧ (55 ≤ x ∧ x ≤ 64)

theorem truck_cargo_solution :
  ∃ y : ℝ, y = 2.5 :=
sorry

end truck_cargo_solution_l667_667332


namespace negation_of_universal_prop_l667_667478

-- Define the proposition p
def p (f : ℝ → ℝ) : Prop := ∀ x ∈ Ioo 0 (π / 2), f x < 0

-- Statement: Negation of the proposition p is ∃ x ∈ (0, π/2), f(x) ≥ 0
theorem negation_of_universal_prop (f : ℝ → ℝ) : ¬ p f ↔ ∃ x ∈ Ioo 0 (π / 2), f x ≥ 0 :=
by 
  sorry

end negation_of_universal_prop_l667_667478


namespace smallest_equal_cost_l667_667655

def decimal_cost (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_cost (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

theorem smallest_equal_cost :
  ∃ n : ℕ, n < 200 ∧ decimal_cost n = binary_cost n ∧ (∀ m : ℕ, m < 200 ∧ decimal_cost m = binary_cost m → m ≥ n) :=
by
  -- Proof goes here
  sorry

end smallest_equal_cost_l667_667655


namespace quadratic_polynomial_coefficients_l667_667021

theorem quadratic_polynomial_coefficients (a b : ℝ)
  (h1 : 2 * a - 1 - b = 0)
  (h2 : 5 * a + b - 13 = 0) :
  a^2 + b^2 = 13 := 
by 
  sorry

end quadratic_polynomial_coefficients_l667_667021


namespace packs_of_green_bouncy_balls_l667_667135

/-- Maggie bought 10 bouncy balls in each pack of red, yellow, and green bouncy balls.
    She bought 4 packs of red bouncy balls, 8 packs of yellow bouncy balls, and some 
    packs of green bouncy balls. In total, she bought 160 bouncy balls. This theorem 
    aims to prove how many packs of green bouncy balls Maggie bought. 
 -/
theorem packs_of_green_bouncy_balls (red_packs : ℕ) (yellow_packs : ℕ) (total_balls : ℕ) (balls_per_pack : ℕ) 
(pack : ℕ) :
  red_packs = 4 →
  yellow_packs = 8 →
  balls_per_pack = 10 →
  total_balls = 160 →
  red_packs * balls_per_pack + yellow_packs * balls_per_pack + pack * balls_per_pack = total_balls →
  pack = 4 :=
by
  intros h_red h_yellow h_balls_per_pack h_total_balls h_eq
  sorry

end packs_of_green_bouncy_balls_l667_667135
