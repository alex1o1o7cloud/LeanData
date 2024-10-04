import Mathlib
import Mathlib.Algebra.ArithmeticSequence
import Mathlib.Algebra.Field
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Perm
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.Probability.Prob
import Mathlib.ProbabilityTheory.PairwiseIndep
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LinearCombination
import Real

namespace length_MP_l796_796492

variable (A B C M P : Type)
variable (a b c : ℝ)
variable [h_nonneg : 0 ≤ a] [h_nonneg : 0 ≤ b] [h_nonneg : 0 ≤ c]
variable (ω : Circle A B)
variable (triangle_ABC : Triangle A B C)
variable (intersection_AC : ω.Intersects AC M)
variable (intersection_BC : ω.Intersects BC P)
variable (inscribed_center : MP.Contains (triangle_ABC.Incenter))

theorem length_MP
  (AB_eq_c : AB = c)
  (BC_eq_a : BC = a)
  (CA_eq_b : CA = b) :
  length MP = c * (a + b) / (a + b + c) :=
sorry

end length_MP_l796_796492


namespace trig_identity_l796_796333

theorem trig_identity (α : ℝ) (h : Real.tan (π - α) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 3 :=
by
  sorry

end trig_identity_l796_796333


namespace problem_1_problem_2_l796_796458

def p (a : ℝ) : Prop := ∀ (x : ℝ), 2^x + 1 ≥ a
def q (a : ℝ) : Prop := ∀ (x : ℝ), ax^2 - x + a > 0
def m (b a : ℝ) : Prop := ∃ (x : ℝ), x^2 + b * x + a = 0

theorem problem_1 (a : ℝ) (h : p a ∧ q a) : a ∈ Ioc (1 / 2) 1 := sorry

theorem problem_2 (a b : ℝ) (h : ∀ h₁ : ¬p a, ¬m b a → ¬h₁) : b ∈ Ioo (-2) 2 := sorry

end problem_1_problem_2_l796_796458


namespace circle_equation_proof_l796_796343

noncomputable def circle_equation (a r : ℝ) := ∀ x y : ℝ, (x - a)^2 + y^2 = r^2

theorem circle_equation_proof :
  ∃ a r : ℝ, a > 0 ∧ (0, real.sqrt 5) ∈ { p : ℝ × ℝ | circle_equation a r p.1 p.2 } ∧
    (2 * a) / real.sqrt 5 = 4 * real.sqrt 5 / 5 ∧ circle_equation a r = (λ x y => (x - 2)^2 + y^2 = 9) :=
by
  sorry

end circle_equation_proof_l796_796343


namespace incorrect_population_growth_statement_l796_796940

def population_growth_behavior (p: ℝ → ℝ) : Prop :=
(p 0 < p 1) ∧ (∃ t₁ t₂, t₁ < t₂ ∧ (∀ t < t₁, p t < p (t + 1)) ∧
 (∀ t > t₁, (p t < p (t - 1)) ∨ (p t = p (t - 1))))

def stabilizes_at_K (p: ℝ → ℝ) (K: ℝ) : Prop :=
∃ t₀, ∀ t > t₀, p t = K

def K_value_definition (K: ℝ) (environmental_conditions: ℝ → ℝ) : Prop :=
∀ t, environmental_conditions t = K

theorem incorrect_population_growth_statement (p: ℝ → ℝ) (K: ℝ) (environmental_conditions: ℝ → ℝ)
(h1: population_growth_behavior p)
(h2: stabilizes_at_K p K)
(h3: K_value_definition K environmental_conditions) :
(p 0 > p 1) ∨ (¬ (∃ t₁ t₂, t₁ < t₂ ∧ (∀ t < t₁, p t < p (t + 1)) ∧
 (∀ t > t₁, (p t < p (t - 1)) ∨ (p t = p (t - 1))))) :=
sorry

end incorrect_population_growth_statement_l796_796940


namespace csc_315_eq_neg_sqrt_two_l796_796995

theorem csc_315_eq_neg_sqrt_two : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_two_l796_796995


namespace john_reading_time_l796_796786

theorem john_reading_time:
  let weekday_hours_moses := 1.5
  let weekday_rate_moses := 30
  let saturday_hours_moses := 2
  let saturday_rate_moses := 40
  let pages_moses := 450
  let weekday_hours_rest := 1.5
  let weekday_rate_rest := 45
  let saturday_hours_rest := 2.5
  let saturday_rate_rest := 60
  let pages_rest := 2350
  let weekdays_per_week := 5
  let saturdays_per_week := 1
  let total_pages_per_week_moses := (weekday_hours_moses * weekday_rate_moses * weekdays_per_week) + 
                                    (saturday_hours_moses * saturday_rate_moses * saturdays_per_week)
  let total_pages_per_week_rest := (weekday_hours_rest * weekday_rate_rest * weekdays_per_week) + 
                                   (saturday_hours_rest * saturday_rate_rest * saturdays_per_week)
  let weeks_moses := (pages_moses / total_pages_per_week_moses).ceil
  let weeks_rest := (pages_rest / total_pages_per_week_rest).ceil
  let total_weeks := weeks_moses + weeks_rest
  total_weeks = 7 :=
by
  -- placeholders for the proof steps.
  sorry

end john_reading_time_l796_796786


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796024

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796024


namespace mean_value_of_quadrilateral_angles_l796_796004

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l796_796004


namespace coefficient_x_squared_l796_796968

-- Problem conditions
variables (x : ℝ) -- Assuming we're working with real numbers here

-- Given expression
def expr : ℝ := 7 * (x - x^4) - 5 * (x^2 - x^3 + x^6) + 3 * (5 * x^2 - x^10)

-- Statement to prove
theorem coefficient_x_squared : coefficient expr x^2 = 10 := by
  sorry

end coefficient_x_squared_l796_796968


namespace tan_double_angle_l796_796366

-- Define the conditions
def angle_condition (α : ℝ) : Prop :=
  let θ := α - Real.pi / 4
  θ = Real.arctan(-2)

-- Define the target statement to prove
theorem tan_double_angle (α : ℝ) (h : angle_condition α) : Real.tan (2 * α) = -3 / 4 := 
  sorry

end tan_double_angle_l796_796366


namespace mean_value_of_quadrilateral_angles_l796_796214

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796214


namespace centipede_shoe_sock_permutations_l796_796916

theorem centipede_shoe_sock_permutations : 
  let legs := 10
  let items := 2 * legs
  let permutations := (20.factorial : ℤ)
  let valid_orders := permutations / 2 ^ legs
  valid_orders = 20.factorial / 2 ^ 10 := 
by
  sorry

end centipede_shoe_sock_permutations_l796_796916


namespace calculate_expression_l796_796235

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := 
by
  -- proof goes here
  sorry

end calculate_expression_l796_796235


namespace coordinates_P_l796_796677

theorem coordinates_P 
  (P1 P2 P : ℝ × ℝ)
  (hP1 : P1 = (2, -1))
  (hP2 : P2 = (0, 5))
  (h_ext_line : ∃ t : ℝ, P = (P1.1 + t * (P2.1 - P1.1), P1.2 + t * (P2.2 - P1.2)) ∧ t ≠ 1)
  (h_distance : dist P1 P = 2 * dist P P2) :
  P = (-2, 11) := 
by
  sorry

end coordinates_P_l796_796677


namespace mean_value_of_quadrilateral_angles_l796_796208

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796208


namespace min_employees_needed_l796_796937

-- Definitions for the problem conditions
def hardware_employees : ℕ := 150
def software_employees : ℕ := 130
def both_employees : ℕ := 50

-- Statement of the proof problem
theorem min_employees_needed : hardware_employees + software_employees - both_employees = 230 := 
by 
  -- Calculation skipped with sorry
  sorry

end min_employees_needed_l796_796937


namespace circle_bisectors_chord_sum_l796_796577

theorem circle_bisectors_chord_sum {
  ABC : Type* 
  [triangle ABC]
  (C : circle) 
  (hC : ∀ P ∈ C.bases_of_angle_bisectors ABC, ∃ K, L, P = K ∩ L) :
  ∃ K₁ L₁ K L, 
  K₁ ∈ C.intersections (sides_of_triangle ABC BC)
  ∧ L₁ ∈ C.intersections (sides_of_triangle ABC BA)
  ∧ K ∈ C.intersections (sides_of_triangle ABC CA)
  ∧ length_chord K₁K = length_chord K₁K + length_chord L₁L :=
sorry


end circle_bisectors_chord_sum_l796_796577


namespace zero_not_in_empty_set_l796_796971

theorem zero_not_in_empty_set : ¬ (0 ∈ (∅ : set ℕ)) :=
sorry

end zero_not_in_empty_set_l796_796971


namespace sine_angle_prism_l796_796689

-- Define the structure of a regular triangular prism
structure RegularTriangularPrism (V : Type) :=
  (A B C A1 B1 C1 : V)
  (is_regular : ∀ (u v : set V), set.equilateral u → set.equilateral v)
  (side_equal_base : ∀ (u : V), (u ≠ A ∧ u ≠ B ∧ u ≠ C) → (∃ v, u = v ∧ v = A1 ∨ v = B1 ∨ v = C1))

--Define the vertices
variables {V : Type} [inner_product_space ℝ V] (prism : RegularTriangularPrism V)

-- Define the midpoint of A1C1
def midpoint (u v : V) := (u + v) / 2

-- Define the angle between two vectors 
def angle_between (u v : V) := real.arccos ((inner_product_space.il_inner u v) / (norm u * norm v))

-- Define the sine of an angle
def sin_angle (θ : ℝ) := real.sin θ

-- Prove that the sine of the angle formed by AB1 and the lateral face ACCA1 is sqrt(6)/4
theorem sine_angle_prism
  (D1 := midpoint prism.A1 prism.C1)
  (B1D1 := prism.B1 - D1)
  (AD1 := prism.A - D1):
  sin_angle (angle_between (prism.A - prism.B1) B1D1) = sqrt(6) / 4 :=
sorry

end sine_angle_prism_l796_796689


namespace mean_of_quadrilateral_angles_is_90_l796_796092

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796092


namespace length_of_chord_AB_is_sqrt15_div_2_l796_796351

-- Definitions for the circles
def C1 (x y : ℝ) := x^2 + y^2 = 1
def C2 (x y : ℝ) := x^2 + y^2 - 4 * x = 0

-- Theorem statement for proving the length of the chord AB
theorem length_of_chord_AB_is_sqrt15_div_2 :
  ∃ (A B : ℝ × ℝ), (C1 A.1 A.2) ∧ (C2 A.1 A.2) ∧
                   (C1 B.1 B.2) ∧ (C2 B.1 B.2) ∧
                   dist A B = sqrt(15) / 2 :=
sorry

end length_of_chord_AB_is_sqrt15_div_2_l796_796351


namespace part1_part2_l796_796349

open Real

-- Definitions for part (1)
variables {x y z : ℕ} (h : x > y) (h2 : y > z) (h3 : z > 0)

def A : ℝ := (x.to_real / (y.to_real + z.to_real))
def B : ℝ := (y.to_real / (x.to_real + z.to_real))
def C : ℝ := (z.to_real / (x.to_real + y.to_real))

theorem part1 (h1 : x > y) (h2 : y > z) (h3 : z > 0) : A h1 h3 > B h1 h3 ∧ B h1 h3 > C h1 h3 := sorry

-- Definitions for part (2)
variable (z : ℝ) 

def A₂ : ℝ := 1 / (1 + z)
def B₂ : ℝ := 1 / (1 + z)
def C₂ : ℝ := z / 2

def is_root (z : ℝ) : Prop := z^2 - 2023 * z + 1 = 0

theorem part2 (hz : is_root z) : (1 / A₂ z) + (1 / B₂ z) + (1 / C₂ z) = 4048 := sorry

end part1_part2_l796_796349


namespace product_of_roots_l796_796537

theorem product_of_roots :
  ∀ α β : ℝ, (Polynomial.roots (Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 5 * Polynomial.X + Polynomial.C (-12))).prod = -6 :=
by
  sorry

end product_of_roots_l796_796537


namespace shopkeeper_profit_is_30_l796_796900

variable (X : ℝ) -- cost price per kg of apples
variable (CP_total : ℝ) -- total cost price of apples
variable (SP_112kg : ℝ) -- selling price for first part
variable (SP_168kg : ℝ) -- selling price for second part
variable (TP : ℝ) -- total profit
variable (percentage_profit : ℝ) -- percentage profit

/-- Assume the known quantities -/
def shopkeeper_data := (CP_total = 280 * X) ∧ 
                       (SP_112kg = 112 * 1.30 * X) ∧ 
                       (SP_168kg = 168 * 1.30 * X) ∧ 
                       (TP = (SP_112kg + SP_168kg) - CP_total) ∧
                       (percentage_profit = (TP / CP_total) * 100)

/-- The goal is to prove the shopkeeper's total percentage profit is 30% -/
theorem shopkeeper_profit_is_30 (h : shopkeeper_data) : percentage_profit = 30 := by
  sorry

end shopkeeper_profit_is_30_l796_796900


namespace problem_l796_796910

-- Define i as the imaginary unit
def i : ℂ := Complex.I

-- The statement to be proved
theorem problem : i * (1 - i) ^ 2 = 2 := by
  sorry

end problem_l796_796910


namespace edward_final_money_l796_796646

theorem edward_final_money 
  (spring_earnings : ℕ)
  (summer_earnings : ℕ)
  (supplies_cost : ℕ)
  (h_spring : spring_earnings = 2)
  (h_summer : summer_earnings = 27)
  (h_supplies : supplies_cost = 5)
  : spring_earnings + summer_earnings - supplies_cost = 24 := 
sorry

end edward_final_money_l796_796646


namespace cone_vertex_angle_l796_796222

theorem cone_vertex_angle (r l : ℝ) (h_ratio : π * r * l / (π * r^2) = 2) : 
  let α := Real.arcsin (r / l) in 2 * α = 60 :=
by
  sorry

end cone_vertex_angle_l796_796222


namespace csc_315_eq_neg_sqrt_two_l796_796989

theorem csc_315_eq_neg_sqrt_two : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_two_l796_796989


namespace eccentricity_of_hyperbola_l796_796850

theorem eccentricity_of_hyperbola :
  let a_squared := (2 : ℝ)
  let b_squared := (1 : ℝ)
  let c_squared := a_squared + b_squared
  let c := Real.sqrt c_squared
  let a := Real.sqrt a_squared
  let e := c / a
  e = Real.sqrt 6 / 2 := by
  sorry

end eccentricity_of_hyperbola_l796_796850


namespace cereal_discount_l796_796925

theorem cereal_discount (milk_normal_cost milk_discounted_cost total_savings milk_quantity cereal_quantity: ℝ) 
  (total_milk_savings cereal_savings_per_box: ℝ) 
  (h1: milk_normal_cost = 3)
  (h2: milk_discounted_cost = 2)
  (h3: total_savings = 8)
  (h4: milk_quantity = 3)
  (h5: cereal_quantity = 5)
  (h6: total_milk_savings = milk_quantity * (milk_normal_cost - milk_discounted_cost)) 
  (h7: total_milk_savings + cereal_quantity * cereal_savings_per_box = total_savings):
  cereal_savings_per_box = 1 :=
by 
  sorry

end cereal_discount_l796_796925


namespace polynomials_with_desired_properties_l796_796297

theorem polynomials_with_desired_properties :
  ∃ (g h : ℤ[X]),
    (∃ c ∈ g.coeffs, |c| > 2015) ∧ 
    (∃ c ∈ h.coeffs, |c| > 2015) ∧ 
    ∀ c ∈ (g * h).coeffs, |c| ≤ 1 :=
sorry

end polynomials_with_desired_properties_l796_796297


namespace marble_count_l796_796763

theorem marble_count (r g b : ℝ) (h1 : g + b = 9) (h2 : r + b = 7) (h3 : r + g = 5) :
  r + g + b = 10.5 :=
by sorry

end marble_count_l796_796763


namespace bill_score_l796_796553

variable {J B S : ℕ}

theorem bill_score (h1 : B = J + 20) (h2 : B = S / 2) (h3 : J + B + S = 160) : B = 45 :=
sorry

end bill_score_l796_796553


namespace mean_value_of_quadrilateral_interior_angles_l796_796170

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796170


namespace train_speed_is_300_kmph_l796_796268

noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

theorem train_speed_is_300_kmph :
  train_speed 1250 15 = 300 := by
  sorry

end train_speed_is_300_kmph_l796_796268


namespace minimum_value_l796_796811

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (1 / x + 9 / y) = 16 :=
sorry

end minimum_value_l796_796811


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796023

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796023


namespace proof_problem_A_proof_problem_C_l796_796894

theorem proof_problem_A (x : ℝ) (h : x > 1) : 
  ∃ m, m = 3 ∧ (∀ x, x > 1 → x + 1 / (x - 1) ≥ m) :=
sorry

theorem proof_problem_C (x : ℝ) (h : 0 < x) (h' : x < 10) : 
  ∃ M, M = 5 ∧ (∀ x, 0 < x → x < 10 → sqrt (x * (10 - x)) ≤ M) :=
sorry

end proof_problem_A_proof_problem_C_l796_796894


namespace train_length_l796_796259

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (h_speed : speed_kmph = 36) (h_time : time_sec = 6.5) : 
  (speed_kmph * 1000 / 3600) * time_sec = 65 := 
by {
  -- Placeholder for proof
  sorry
}

end train_length_l796_796259


namespace candy_cost_l796_796832

theorem candy_cost 
  (saved : ℝ)
  (mother : ℝ)
  (father : ℝ)
  (candies : ℝ)
  (jellybeans : ℝ)
  (remaining : ℝ)
  (total_cost : ℝ)
  (candies_cost : ℝ) :
  saved = 10 →
  mother = 4 →
  father = mother * 2 →
  candies = 14 →
  jellybeans = 20 →
  remaining = 11 →
  total_cost = saved + mother + father - remaining →
  candies_cost = total_cost / candies →
  candies_cost =  0.79 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  have h8 : total_cost = 10 + 4 + 8 - 11 := by sorry
  have h9 : total_cost = 11 := by sorry
  have h10 : candies_cost = 11 / 14 := by sorry
  have h11 : candies_cost = 0.7857 := by sorry
  have h12 : candies_cost ≈ 0.79 := by sorry
  sorry

end candy_cost_l796_796832


namespace digit_sum_at_least_18_l796_796481

-- Definitions corresponding to the conditions
def divisible_by_11 (n : ℕ) : Prop := 
  (n.digits.sum % 11 = 0)

def divisible_by_9 (n : ℕ) : Prop := 
  (n.digits.sum % 9 = 0)

def divisible_by_99 (n : ℕ) : Prop := 
  divisible_by_11 n ∧ divisible_by_9 n

-- Main statement to prove: Any number divisible by 99 has digit sum at least 18
theorem digit_sum_at_least_18 (n : ℕ) (h : divisible_by_99 n) : 18 ≤ n.digits.sum :=
by { sorry }

end digit_sum_at_least_18_l796_796481


namespace min_colors_needed_l796_796252

theorem min_colors_needed (n : ℕ) : 
  (n + (n * (n - 1)) / 2 ≥ 12) → (n = 5) :=
by
  sorry

end min_colors_needed_l796_796252


namespace polynomial_root_sum_l796_796798

noncomputable def poly (p q : ℝ) : Polynomial ℂ :=
  Polynomial.C p * Polynomial.X + Polynomial.C q + Polynomial.X^³

theorem polynomial_root_sum (p q : ℝ) 
  (h1 : is_root ((X^3 : Polynomial ℂ) + p • X + q • (1 : Polynomial ℂ)) (2 + I)) :
  p + q = 9 :=
by
  sorry

end polynomial_root_sum_l796_796798


namespace interval_between_prizes_l796_796769

theorem interval_between_prizes (total_prize : ℝ) (first_place : ℝ) (interval : ℝ) :
  total_prize = 4800 ∧
  first_place = 2000 ∧
  (first_place - interval) + (first_place - 2 * interval) = total_prize - 2000 →
  interval = 400 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  rw [h1, h2] at h3
  sorry

end interval_between_prizes_l796_796769


namespace area_AMDN_eq_area_ABC_l796_796771

-- Definitions of the points and conditions
variables {A B C E F M N D : Point}
variables {circumcircle_ABC : Circle}
variables {h1 : E ∈ segment B C}
variables {h2 : F ∈ segment B C}
variables {h3 : ∠ BAE = ∠ CAF}
variables {h4 : M = foot_of_perpendicular F A B}
variables {h5 : N = foot_of_perpendicular F A C}
variables {h6 : D ∈ circumcircle_ABC}
variables {h7 : ∃ X, X ∈ line A E ∧ X ∈ circumcircle_ABC ∧ X ≠ A ∧ D = X}

-- Statement of the problem
theorem area_AMDN_eq_area_ABC 
  (h8 : are_cyclic [A, M, F, N]) 
  (h9 : ∠ AMN + ∠ BAE = 90)
  (h10 : MN ⊥ AD) 
  (h11 : similar AFC ABD) 
  (h12 : area (quadrilateral A M D N) = area (triangle A B C)) : 
  area (quadrilateral A M D N) = area (triangle A B C) :=
sorry

end area_AMDN_eq_area_ABC_l796_796771


namespace sum_cos_squares_equal_23_div_2_l796_796958

theorem sum_cos_squares_equal_23_div_2 : 
  (∑ i in (Finset.range 46).filter (λ i, i % 2 = 0), (Real.cos (i * (π / 180)))^2) = 23 / 2 := 
  sorry

end sum_cos_squares_equal_23_div_2_l796_796958


namespace even_stones_fraction_odd_stones_fraction_l796_796531

/-- The fraction of the distributions of 12 indistinguishable stones to 4 distinguishable boxes where every box contains an even number of stones is 12/65. -/
theorem even_stones_fraction : (∀ (B1 B2 B3 B4 : ℕ), B1 % 2 = 0 ∧ B2 % 2 = 0 ∧ B3 % 2 = 0 ∧ B4 % 2 = 0 ∧ B1 + B2 + B3 + B4 = 12) → (84 / 455 = 12 / 65) := 
by sorry

/-- The fraction of the distributions of 12 indistinguishable stones to 4 distinguishable boxes where every box contains an odd number of stones is 1/13. -/
theorem odd_stones_fraction : (∀ (B1 B2 B3 B4 : ℕ), B1 % 2 = 1 ∧ B2 % 2 = 1 ∧ B3 % 2 = 1 ∧ B4 % 2 = 1 ∧ B1 + B2 + B3 + B4 = 12) → (35 / 455 = 1 / 13) := 
by sorry

end even_stones_fraction_odd_stones_fraction_l796_796531


namespace right_prism_condition_l796_796761

theorem right_prism_condition (n : ℕ) (convex_prism : Prop)
  (congruent_lateral_faces : Prop) : (convex_prism ∧ congruent_lateral_faces) → ((n ≠ 4) ↔ (is_right_prism n)) :=
by
  sorry

end right_prism_condition_l796_796761


namespace school_distance_l796_796748

theorem school_distance (T D : ℝ) (h1 : 5 * (T + 6) = 630) (h2 : 7 * (T - 30) = 630) :
  D = 630 :=
sorry

end school_distance_l796_796748


namespace mean_value_of_quadrilateral_angles_l796_796044

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796044


namespace max_modulus_distance_unit_circle_l796_796341

theorem max_modulus_distance_unit_circle (z : ℂ) (hz : |z| = 1) : 
  (∃ w : ℂ, w = 3 - 4 * complex.I ∧ ∀ (z : ℂ), |z| = 1 → |z - w| ≤ 6) :=
sorry

end max_modulus_distance_unit_circle_l796_796341


namespace angle_quadrant_l796_796641

theorem angle_quadrant (θ : ℝ) (hq : θ ≡ 2016 [MOD 360]) : 180 < θ ∧ θ < 270 → quadrant θ = Quadrant.III :=
by
  intro hθ
  sorry

end angle_quadrant_l796_796641


namespace solve_for_a_and_b_l796_796362

theorem solve_for_a_and_b (a b : ℝ) (hp1 : a > 0) (hp2 : a ≠ 1)
  (h1 : log a (-1 + b) = 0) (h2 : log a b = 1) : a = 2 ∧ b = 2 := by
  sorry

end solve_for_a_and_b_l796_796362


namespace ball_hits_ground_time_l796_796493

def height (t : ℝ) : ℝ := -16 * t^2 + 32 * t + 60

theorem ball_hits_ground_time :
  ∃ t : ℝ, height t = 0 ∧ t = 1 + sqrt 19 / 2 :=
by
  sorry

end ball_hits_ground_time_l796_796493


namespace coefficient_of_m5n4_in_expansion_l796_796534

def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_of_m5n4_in_expansion : binomial_coefficient 9 5 = 126 := by
  sorry

end coefficient_of_m5n4_in_expansion_l796_796534


namespace unique_property_rhombus_l796_796892

-- Definitions of shapes
structure Rhombus (R : Type) :=
  (side : R)
  (angle : R)
  (diagonals_are_perpendicular : Prop)

structure Rectangle (R : Type) :=
  (length : R)
  (width : R)
  (diagonals_are_equal : Prop)

-- Main theorem
theorem unique_property_rhombus (R : Type) [NontrivialRing R] :
  ∀ (r : Rhombus R) (s : Rectangle R), r.diagonals_are_perpendicular ∧ ¬ s.diagonals_are_perpendicular :=
by
  intros r s
  sorry

end unique_property_rhombus_l796_796892


namespace largest_number_of_red_faces_l796_796944

def small_cube := { faces_red : ℕ // faces_red ≤ 2 }

def large_cube :=
  (| side_length : ℕ, total_cubes : ℕ | side_length = 3 ∧ total_cubes = 27)

def max_red_faces (lc : large_cube) (cubes : list small_cube) : ℕ :=
  if lc.side_length = 3 ∧ lc.total_cubes = 27 ∧ ∀ (c : small_cube), c.faces_red ≤ 2 ∧ list.length cubes = 27
  then 4 else 0

theorem largest_number_of_red_faces (lc : large_cube) (cubes : list small_cube) :
  lc.side_length = 3 ∧ lc.total_cubes = 27 ∧ (∀ c : small_cube, c.faces_red ≤ 2) ∧ list.length cubes = 27 →
  max_red_faces lc cubes = 4 := by sorry

end largest_number_of_red_faces_l796_796944


namespace total_vessels_proof_l796_796573

def cruise_ships : Nat := 4
def cargo_ships : Nat := cruise_ships * 2
def sailboats : Nat := cargo_ships + 6
def fishing_boats : Nat := sailboats / 7
def total_vessels : Nat := cruise_ships + cargo_ships + sailboats + fishing_boats

theorem total_vessels_proof : total_vessels = 28 := by
  sorry

end total_vessels_proof_l796_796573


namespace rectangles_exist_l796_796973

noncomputable def s : ℕ → ℝ
| 0       := 1
| (n + 1) := s n / 4

def d : ℕ → ℝ 
| 0       := real.sqrt 2
| (n + 1) := d n -- the actual form will depend on the specifics of each recursive rectangle definition

def l : ℕ → ℝ 
| 0       := 1
| (n + 1) := 2 * (finset.range (n + 1)).sum d 

def exists_rectangles : Prop :=
∃ (rectangles : fin 100 → rect), 
  (∀ i : fin 100, rectangles i = ⟨l i, _, _⟩) ∧
  ∀ (i j : fin 100), i ≠ j → ¬(rectangles i ⊆ rectangles j)

theorem rectangles_exist : exists_rectangles :=
begin
  -- corresponding proof would go here
  sorry
end

end rectangles_exist_l796_796973


namespace sequence_general_formula_l796_796410

theorem sequence_general_formula (n : ℕ) (hn : n > 0) 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (hS : ∀ n, S n = 1 - n * a n) 
  (hpos : ∀ n, a n > 0) : 
  (a n = 1 / (n * (n + 1))) :=
sorry

end sequence_general_formula_l796_796410


namespace polynomial_property_solution_l796_796452

open Polynomial

noncomputable def satisfy_polynomial_property (P : Polynomial ℝ) (k : ℕ) : Prop :=
  P.comp(P) = P^k

noncomputable def is_constant (P : Polynomial ℝ) : Prop :=
  P.degree = 0

theorem polynomial_property_solution (P : Polynomial ℝ) (k : ℕ) (hk : k > 0) :
  satisfy_polynomial_property P k ↔ 
  ( (is_constant P ∧ 
      (P.coeff 0 ≠ 0 ∧ k = 1 ∧ ∃ c : ℝ, P = Polynomial.C c) ∨ 
      (P.coeff 0 = 1 ∧ even k) ∨ 
      (P.coeff 0 = 1 ∨ P.coeff 0 = -1 ∧ odd k ∧ k > 1)) ∨ 
    (¬ is_constant P ∧ P = Polynomial.X ^ k) ) :=
by
  sorry

end polynomial_property_solution_l796_796452


namespace sum_of_angles_l796_796357

open Real

theorem sum_of_angles
  (α β : ℝ)
  (hα : α ∈ Ioo (-π / 2) (π / 2))
  (hβ : β ∈ Ioo (-π / 2) (π / 2))
  (h_roots : (tan α) ∈ {x : ℝ | x^2 + 3 * sqrt 3 * x + 4 = 0} 
              ∧ (tan β) ∈ {x : ℝ | x^2 + 3 * sqrt 3 * x + 4 = 0}) :
  α + β = -2 * π / 3 :=
sorry

end sum_of_angles_l796_796357


namespace mean_value_of_quadrilateral_interior_angles_l796_796176

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796176


namespace construct_rhombus_given_diagonals_sum_and_angle_l796_796966

theorem construct_rhombus_given_diagonals_sum_and_angle 
  (d1 d2 : ℝ) (α : ℝ) :
  ∃ (a : ℝ) (b : ℝ), 
    a * b = (d1 ^ 2 + d2 ^ 2) / 4 ∧ -- Condition derived from diagonals and geometry of the rhombus
    ∠ a b = α :=               -- Condition that angles in the geometry of the rhombus are as specified 
sorry

end construct_rhombus_given_diagonals_sum_and_angle_l796_796966


namespace increasing_sequence_l796_796728

   theorem increasing_sequence (a : ℕ → ℝ) (h_init : a 1 > 0) (h_def : ∀ n, a (n + 1) = real.sqrt ((3 + a n) / 2)) :
     0 < a 1 ∧ a 1 < 3 / 2 → ∀ n, a (n + 1) > a n :=
   sorry
   
end increasing_sequence_l796_796728


namespace continuous_at_4_l796_796823

noncomputable def f (x : ℝ) := 3 * x^2 - 3

theorem continuous_at_4 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 4| < δ → |f x - f 4| < ε :=
by
  sorry

end continuous_at_4_l796_796823


namespace octagon_area_in_circle_l796_796931

theorem octagon_area_in_circle (r : ℝ) (h : r = 3) :
  let A := 72 * Real.sqrt ((1 - Real.sqrt 2 / 2) / 2)
  in A = 72 * Real.sqrt ((1 - Real.sqrt 2 / 2) / 2) :=
by
  let A := 72 * (Real.sqrt ((1 - Real.sqrt 2 / 2) / 2))
  exact rfl

end octagon_area_in_circle_l796_796931


namespace mean_of_quadrilateral_angles_l796_796197

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796197


namespace distinct_integer_sums_count_l796_796638

def is_special_fraction (a b : ℕ) : Prop := a + b = 20

def special_fractions : Set ℚ :=
  {q | ∃ (a b : ℕ), is_special_fraction a b ∧ q = (a : ℚ) / (b : ℚ)}

def possible_sums_of_special_fractions : Set ℚ :=
  { sum | ∃ (q1 q2 : ℚ), q1 ∈ special_fractions ∧ q2 ∈ special_fractions ∧ sum = q1 + q2 }

def integer_sums : Set ℤ :=
  { z | z ∈ possible_sums_of_special_fractions ∧ z.den = 1 }

theorem distinct_integer_sums_count : ∃ n : ℕ, n = 15 ∧ n = integer_sums.to_finset.card :=
by
  sorry

end distinct_integer_sums_count_l796_796638


namespace mean_value_of_quadrilateral_interior_angles_l796_796174

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796174


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796021

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796021


namespace inverse_function_property_l796_796744

noncomputable def f (x : ℝ) : ℝ := 16 / (5 + 3 * x)

theorem inverse_function_property :
  ((Function.inverse f 2) ^ -2) = 1 := by
  sorry

end inverse_function_property_l796_796744


namespace sequence_property_l796_796727

noncomputable def seq (n : ℕ) : ℕ := 
if n = 0 then 1 else 
if n = 1 then 3 else 
seq (n-2) + 3 * 2^(n-2)

theorem sequence_property {n : ℕ} (h_pos : n > 0) :
(∀ n : ℕ, n > 0 → seq (n + 2) ≤ seq n + 3 * 2^n) →
(∀ n : ℕ, n > 0 → seq (n + 1) ≥ 2 * seq n + 1) →
seq n = 2^n - 1 := 
sorry

end sequence_property_l796_796727


namespace value_of_lambda_l796_796707

theorem value_of_lambda (f : ℝ → ℝ) (hf_odd : ∀ x, f (-x) = -f x) (hf_monotonic : ∀ {x y}, x ≤ y → f x ≤ f y)
  (h : ∃! x, f (2 * x ^ 2 + 1) + f (λ - x) = 0) :
  λ = -7 / 8 := 
sorry

end value_of_lambda_l796_796707


namespace tshirt_cost_l796_796411

-- Definitions based on conditions
def pants_cost : ℝ := 80
def shoes_cost : ℝ := 150
def discount : ℝ := 0.1
def total_paid : ℝ := 558

-- Variables based on the problem
variable (T : ℝ) -- Cost of one T-shirt
def num_tshirts : ℝ := 4
def num_pants : ℝ := 3
def num_shoes : ℝ := 2

-- Theorem: The cost of one T-shirt is $20
theorem tshirt_cost : T = 20 :=
by
  have total_cost : ℝ := (num_tshirts * T) + (num_pants * pants_cost) + (num_shoes * shoes_cost)
  have discounted_total : ℝ := (1 - discount) * total_cost
  have payment_condition : discounted_total = total_paid := sorry
  sorry -- detailed proof

end tshirt_cost_l796_796411


namespace range_of_a_satisfying_equation_l796_796715

def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 else 2 ^ x

theorem range_of_a_satisfying_equation :
  {a : ℝ | f (f a) = 2 ^ (f a)} = {a : ℝ | a ≥ 2 / 3} :=
sorry

end range_of_a_satisfying_equation_l796_796715


namespace circle_black_points_l796_796676

theorem circle_black_points (n : ℕ) (hn : 3 ≤ n) :
  ∀ (points : Finset ℕ) (black_points : Finset ℕ),
  points.card = 2 * n - 1 →
  black_points.card = n →
  black_points ⊆ points →
  ∃ (a b : ℕ), a ∈ black_points ∧ b ∈ black_points ∧ (
    points.filter (λ x, x > min a b ∧ x < max a b)).card = n :=
by sorry

end circle_black_points_l796_796676


namespace totalMoney_l796_796869

noncomputable def totalAmount (x : ℝ) : ℝ := 15 * x

theorem totalMoney (x : ℝ) (h : 1.8 * x = 9) : totalAmount x = 75 :=
by sorry

end totalMoney_l796_796869


namespace bisectors_lengths_l796_796309

noncomputable def right_triangle_bisectors 
  (AC BC : ℝ)
  (hAC : AC = 24)
  (hBC : BC = 18) : 
  ℝ × ℝ :=
  let AB := Real.sqrt (AC^2 + BC^2) in
  let CD := (AC * AB - AC * BC / (AC + BC)) / AB in
  let AD := Real.sqrt (AC * AB - CD * (AC - CD)) in
  let AM := (BC * AB - AC * BC / (AC + BC)) / AB in
  let CM := Real.sqrt (AC * BC - AM * (BC - AM)) in
  (AD, CM)

theorem bisectors_lengths 
  (AC BC : ℝ)
  (hAC : AC = 24)
  (hBC : BC = 18) : 
  right_triangle_bisectors AC BC hAC hBC = (9 * Real.sqrt 5, 8 * Real.sqrt 10) := 
  by 
    sorry

end bisectors_lengths_l796_796309


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796019

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796019


namespace factorize_difference_of_squares_l796_796977

-- Define the variables a and b
variables (a b : ℝ)

-- Theorem statement: a^2 - 4b^2 = (a + 2b) * (a - 2b)
theorem factorize_difference_of_squares : a^2 - 4b^2 = (a + 2b) * (a - 2b) :=
by sorry

end factorize_difference_of_squares_l796_796977


namespace band_row_lengths_l796_796567

theorem band_row_lengths (n : ℕ) (h1 : n = 108) (h2 : ∃ k, 10 ≤ k ∧ k ≤ 18 ∧ 108 % k = 0) : 
  (∃ count : ℕ, count = 2) :=
by 
  sorry

end band_row_lengths_l796_796567


namespace max_profit_l796_796919

def salesRevenue (x : ℕ) (m : ℚ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 3 then 9 / (x + 1) + 2 * x + 3
  else if 3 < x ∧ x ≤ 6 then 9 * m * x / (x + 3) + 3 * x
  else if 6 < x then 21
  else 0

def dailyCost (x : ℕ) : ℚ :=
  x 

def salesProfit (x : ℕ) (m : ℚ) : ℚ :=
  salesRevenue x m - dailyCost x

theorem max_profit 
  (m : ℚ) 
  (h_m : m = 1 / 2)
  (P_max : ∀ x, 3 < x ∧ x ≤ 6 → salesProfit x m ≤ 15) : 
  m = 1 / 2 ∧ 
  (∀ x, 1 ≤ x ∧ x ≤ 3 → 
    let P := salesRevenue x m - dailyCost x in P = 9 / (x + 1) + x + 3) ∧
  (∀ x, 3 < x ∧ x ≤ 6 → 
    let P := salesRevenue x m - dailyCost x in P = 9 * x / (2 * (x + 3)) + 2 * x) ∧
  (∀ x, 6 < x →
    let P := salesRevenue x m - dailyCost x in P = 21 - x) ∧
  (∃ x, 3 < x ∧ x ≤ 6 ∧ salesProfit x m = 15) :=
  sorry

end max_profit_l796_796919


namespace max_stickers_single_player_l796_796250

noncomputable def max_stickers (num_players : ℕ) (average_stickers : ℕ) : ℕ :=
  let total_stickers := num_players * average_stickers
  let min_stickers_one_player := 1
  let min_stickers_others := (num_players - 1) * min_stickers_one_player
  total_stickers - min_stickers_others

theorem max_stickers_single_player : 
  ∀ (num_players average_stickers : ℕ), 
    num_players = 25 → 
    average_stickers = 4 →
    ∀ player_stickers : ℕ, player_stickers ≤ max_stickers num_players average_stickers → player_stickers = 76 :=
    by
      intro num_players average_stickers players_eq avg_eq player_stickers player_le_max
      sorry

end max_stickers_single_player_l796_796250


namespace polynomial_root_sum_l796_796799

noncomputable def poly (p q : ℝ) : Polynomial ℂ :=
  Polynomial.C p * Polynomial.X + Polynomial.C q + Polynomial.X^³

theorem polynomial_root_sum (p q : ℝ) 
  (h1 : is_root ((X^3 : Polynomial ℂ) + p • X + q • (1 : Polynomial ℂ)) (2 + I)) :
  p + q = 9 :=
by
  sorry

end polynomial_root_sum_l796_796799


namespace sqrt_expression_eq_l796_796970

theorem sqrt_expression_eq :
  Real.sqrt (3 * (7 + 4 * Real.sqrt 3)) = 2 * Real.sqrt 3 + 3 := 
  sorry

end sqrt_expression_eq_l796_796970


namespace cross_section_shape_of_right_prism_l796_796849

theorem cross_section_shape_of_right_prism (P : Prism) (h_right : P.is_right_prism) (h_centers : P.cross_section_through_centers) : 
  P.cross_section_shape = Shape.GeneralTrapezoid ∨ P.cross_section_shape = Shape.IsoscelesTrapezoid := 
sorry

end cross_section_shape_of_right_prism_l796_796849


namespace ratio_last_to_first_two_songs_l796_796464

noncomputable def total_days : ℕ := 14
noncomputable def gigs_per_week : ℕ := total_days / 2
noncomputable def songs_per_gig : ℕ := 3
noncomputable def length_first_song : ℕ := 5
noncomputable def length_second_song : ℕ := 5
noncomputable def total_playing_time : ℕ := 280
noncomputable def gigs_per_two_weeks : ℕ := total_days / 2 -- since 2 weeks have 14 days, and he gigs every other day

theorem ratio_last_to_first_two_songs :
  let length_first_two_songs_per_gig := length_first_song + length_second_song in
  let total_length_first_two_songs := gigs_per_two_weeks * length_first_two_songs_per_gig in
  let total_length_last_song := total_playing_time - total_length_first_two_songs in
  let length_last_song_per_gig := total_length_last_song / gigs_per_two_weeks in
  length_last_song_per_gig / length_first_two_songs_per_gig = 3 := 
by
  sorry

end ratio_last_to_first_two_songs_l796_796464


namespace number_of_triangles_number_of_rays_l796_796877

-- Definitions based on the problem conditions
def total_points : ℕ := 9
def collinear_points : ℕ := 4
def non_collinear_points : ℕ := 5

-- Lean proof problem:
theorem number_of_triangles (h1 : collinear_points = 4) (h2 : total_points = 9) :
  nat.choose total_points 3 - nat.choose collinear_points 3 = 80 :=
sorry

theorem number_of_rays (h3 : non_collinear_points = 5) (h4 : collinear_points = 4) (h5 : total_points = 9) :
  nat.perm non_collinear_points 2 + 2 * 1 + 2 * 2 + (nat.choose collinear_points 1 * nat.choose non_collinear_points 1 * nat.perm 2 2) = 66 :=
sorry

end number_of_triangles_number_of_rays_l796_796877


namespace complex_point_in_first_quadrant_l796_796420

theorem complex_point_in_first_quadrant :
  let z := (1 / (1 + complex.I)) + complex.I in
  0 < z.re ∧ 0 < z.im :=
by
  sorry

end complex_point_in_first_quadrant_l796_796420


namespace solution_set_empty_for_k_l796_796661

theorem solution_set_empty_for_k (k : ℝ) :
  (∀ x : ℝ, ¬ (kx^2 - 2 * |x - 1| + 3 * k < 0)) ↔ (1 ≤ k) :=
by
  sorry

end solution_set_empty_for_k_l796_796661


namespace closest_integer_percentage_increase_l796_796489

theorem closest_integer_percentage_increase :
  let radius1 := 5
  let radius2 := 4
  let area1 := Real.pi * radius1^2
  let area2 := Real.pi * radius2^2
  let difference := area1 - area2
  let percentage_increase := (difference / area2) * 100
  let closest_integer := Int.ofReal (Real.floor (percentage_increase / 1) * 1)
  closest_integer = 56 := by
  sorry

end closest_integer_percentage_increase_l796_796489


namespace compound_interest_rate_l796_796848

-- Conditions
def principal : ℝ := 2000
def compound_interest : ℝ := 662
def future_value : ℝ := 2662
def period : ℝ := 3.0211480362537766
def n : ℕ := 1

-- Target interest rate to verify
def target_rate : ℝ := 10/100

-- Lean 4 statement for the given problem
theorem compound_interest_rate :
  future_value = principal * (1 + target_rate / n) ^ (n * period) :=
by
  -- The proof steps would go here
  sorry

end compound_interest_rate_l796_796848


namespace area_triangle_OKT_l796_796847

open EuclideanGeometry

variables {A B C M N O K T : Point}
variables {a : ℝ}

-- Conditions
def condition1 (ω : Circle) := tangent ω BA M ∧ tangent ω BC N ∧ center ω O
def condition2 := parallel (lineThrough M parallelTo BC) (ray BO)
def condition3 := ∃ x y, pointOnRay x y ∧ angleBetween M T K = (1 / 2) * angle† A B C
def condition4 (ω : Circle) := tangent (lineThrough K T) ω
def condition5 := length B M = a

-- Goal: The area of triangle OKT
theorem area_triangle_OKT (ω : Circle) (h1 : condition1 ω) (h2 : condition2) (h3 : condition3) (h4 : condition4 ω) (h5 : condition5) :
  area ( ∆ O K T ) = a^2 / 2 :=
sorry

end area_triangle_OKT_l796_796847


namespace find_starting_number_l796_796876

-- Define the conditions and the question
def starting_number (x y : ℕ) : Prop :=
  ∃ (n : ℕ), y = x + n * 3 ∧ n = 11 ∧ y <= 47

theorem find_starting_number : starting_number 12 45 := 
by
  have h : 45 = 12 + 11 * 3 by norm_num
  have h_le : 45 <= 47 := by norm_num
  use 11
  exact ⟨h, rfl, h_le⟩
  sorry

end find_starting_number_l796_796876


namespace solve_z_eqn_l796_796306

theorem solve_z_eqn (z : ℂ) : (z^6 - 6*z^4 + 9*z^2 = 0) ↔ (z = -complex.sqrt 3 ∨ z = 0 ∨ z = complex.sqrt 3) := 
by sorry

end solve_z_eqn_l796_796306


namespace max_value_of_a_l796_796723

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a ≥ 0) → a ≤ 4 := 
by {
  sorry
}

end max_value_of_a_l796_796723


namespace chemistry_class_size_l796_796278

theorem chemistry_class_size
  (total_students : ℕ)
  (chem_bio_both : ℕ)
  (bio_students : ℕ)
  (chem_students : ℕ)
  (both_students : ℕ)
  (H1 : both_students = 8)
  (H2 : bio_students + chem_students + both_students = total_students)
  (H3 : total_students = 70)
  (H4 : chem_students = 2 * (bio_students + both_students)) :
  chem_students + both_students = 52 :=
by
  sorry

end chemistry_class_size_l796_796278


namespace mean_of_quadrilateral_angles_l796_796030

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796030


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796059

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796059


namespace integral_sin_x_l796_796302

theorem integral_sin_x :
    ∫ x in 0..1, sin x = 1 - cos 1 :=
by
  sorry

end integral_sin_x_l796_796302


namespace g_n_plus_1_minus_g_n_minus_1_l796_796639

def g (n : ℤ) : ℚ := 
  (5 + 4 * Real.sqrt 2) / 2 * (Real.sqrt 2)^n - 
  (5 - 4 * Real.sqrt 2) / 2 * (- Real.sqrt 2)^n 

theorem g_n_plus_1_minus_g_n_minus_1 (n : ℤ) : 
  g (n + 1) - g (n - 1) = (Real.sqrt 2 / 2) * g n := 
sorry

end g_n_plus_1_minus_g_n_minus_1_l796_796639


namespace ratio_of_time_l796_796544

theorem ratio_of_time (tX tY tZ : ℕ) (h1 : tX = 16) (h2 : tY = 12) (h3 : tZ = 8) :
  (tX : ℚ) / (tY * tZ / (tY + tZ) : ℚ) = 10 / 3 := 
by 
  sorry

end ratio_of_time_l796_796544


namespace prime_roots_range_l796_796389

theorem prime_roots_range (p : ℕ) (hp : Prime p) (h : ∃ x₁ x₂ : ℤ, x₁ + x₂ = -p ∧ x₁ * x₂ = -444 * p) : 31 < p ∧ p ≤ 41 :=
by sorry

end prime_roots_range_l796_796389


namespace ratio_value_l796_796400

theorem ratio_value (c d : ℝ) (h1 : c = 15 - 4 * d) (h2 : c / d = 4) : d = 15 / 8 :=
by sorry

end ratio_value_l796_796400


namespace mean_of_quadrilateral_angles_l796_796146

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796146


namespace probability_sum_A1_A2_le_0_5_l796_796407

-- Declare three events with given probabilities
constants {A1 A2 A3 : Event}
variable (P : Event → ℝ)
axiom P_A1 : P A1 = 0.2
axiom P_A2 : P A2 = 0.3
axiom P_A3 : P A3 = 0.5

-- State the goal to prove
theorem probability_sum_A1_A2_le_0_5 : P (A1 ∪ A2) ≤ 0.5 :=
sorry

end probability_sum_A1_A2_le_0_5_l796_796407


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796069

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796069


namespace number_of_valid_initial_numbers_l796_796231

theorem number_of_valid_initial_numbers : 
  ∃ n : ℕ, n = 39 ∧ 
  ∀ (start : ℕ), 
    (start < 10^6) →
    (∀ (m : ℕ), (m = ∑ d in (start.digits 10), d ∨ m = ∏ d in (start.digits 10), d) → (m % 2 = 1)) →
    (∃ valid_starts : List ℕ, valid_starts.length n ∧ ∀ s ∈ valid_starts, (s < 10^6 ∧
      (∀ (m : ℕ), (m = ∑ d in (s.digits 10), d ∨ m = ∏ d in (s.digits 10), d) → (m % 2 = 1)))) :=
begin
  sorry  -- Proof to be completed
end

end number_of_valid_initial_numbers_l796_796231


namespace csc_315_eq_neg_sqrt_2_l796_796998

theorem csc_315_eq_neg_sqrt_2 :
  Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := 
by
  sorry

end csc_315_eq_neg_sqrt_2_l796_796998


namespace mean_value_of_quadrilateral_angles_l796_796161

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796161


namespace mean_of_quadrilateral_angles_l796_796200

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796200


namespace mean_value_of_quadrilateral_angles_l796_796162

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796162


namespace inequality_solution_minimum_value_l796_796443

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 4)

theorem inequality_solution (x : ℝ) : f(x) ≥ 2 ↔ x ≤ -7 ∨ x ≥ 5 / 3 := sorry

noncomputable def min_f : ℝ := -9 / 2

theorem minimum_value : ∃ x : ℝ, f(x) = min_f := 
  exists.intro (-1/2) (by
    unfold f
    simp
    norm_num
    sorry
  )

end inequality_solution_minimum_value_l796_796443


namespace mean_value_of_quadrilateral_angles_l796_796165

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796165


namespace Michelle_bought_14_chocolate_bars_l796_796814

-- Definitions for conditions
def sugar_per_chocolate_bar : ℕ := 10
def sugar_in_lollipop : ℕ := 37
def total_sugar_in_candy : ℕ := 177

-- Theorem to prove
theorem Michelle_bought_14_chocolate_bars :
  (total_sugar_in_candy - sugar_in_lollipop) / sugar_per_chocolate_bar = 14 :=
by
  -- Proof steps will go here, but are omitted as per the requirements.
  sorry

end Michelle_bought_14_chocolate_bars_l796_796814


namespace negation_proposition_l796_796504

theorem negation_proposition (x y : ℝ) :
  (¬ (x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔ (x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) := 
sorry

end negation_proposition_l796_796504


namespace mean_value_of_quadrilateral_angles_l796_796001

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l796_796001


namespace mean_of_quadrilateral_angles_is_90_l796_796091

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796091


namespace jess_remaining_blocks_l796_796625

-- Define the number of blocks for each segment of Jess's errands
def blocks_to_post_office : Nat := 24
def blocks_to_store : Nat := 18
def blocks_to_gallery : Nat := 15
def blocks_to_library : Nat := 14
def blocks_to_work : Nat := 22
def blocks_already_walked : Nat := 9

-- Calculate the total blocks to be walked
def total_blocks : Nat :=
  blocks_to_post_office + blocks_to_store + blocks_to_gallery + blocks_to_library + blocks_to_work

-- The remaining blocks Jess needs to walk
def blocks_remaining : Nat :=
  total_blocks - blocks_already_walked

-- The statement to be proved
theorem jess_remaining_blocks : blocks_remaining = 84 :=
by
  sorry

end jess_remaining_blocks_l796_796625


namespace dried_fruit_percentage_l796_796903

-- Define the percentages for Sue, Jane, and Tom's trail mixes.
structure TrailMix :=
  (nuts : ℝ)
  (dried_fruit : ℝ)

def sue : TrailMix := { nuts := 0.30, dried_fruit := 0.70 }
def jane : TrailMix := { nuts := 0.60, dried_fruit := 0.00 }  -- Note: No dried fruit
def tom : TrailMix := { nuts := 0.40, dried_fruit := 0.50 }

-- Condition: Combined mix contains 45% nuts.
def combined_nuts (sue_nuts jane_nuts tom_nuts : ℝ) : Prop :=
  0.33 * sue_nuts + 0.33 * jane_nuts + 0.33 * tom_nuts = 0.45

-- Condition: Each contributes equally to the total mixture.
def equal_contribution (sue_cont jane_cont tom_cont : ℝ) : Prop :=
  sue_cont = jane_cont ∧ jane_cont = tom_cont

-- Theorem to be proven: Combined mixture contains 40% dried fruit.
theorem dried_fruit_percentage :
  combined_nuts sue.nuts jane.nuts tom.nuts →
  equal_contribution (1 / 3) (1 / 3) (1 / 3) →
  0.33 * sue.dried_fruit + 0.33 * tom.dried_fruit = 0.40 :=
by sorry

end dried_fruit_percentage_l796_796903


namespace minimum_m_n_l796_796335

variable (a b : ℝ)

-- Conditions
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom geom_mean_one : Real.sqrt (a * b) = 1

-- Definitions
def m := b + (1 / a)
def n := a + (1 / b)

-- Goal
theorem minimum_m_n : m + n ≥ 4 := 
by {
  -- proof goes here
  sorry
}

end minimum_m_n_l796_796335


namespace matrix_property_l796_796657

open Matrix

noncomputable def M : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0,  8, 2],
    ![-8, 0, -5],
    ![-2, 5, 0]
  ]

theorem matrix_property (v : Vector ℝ 3) :
    (M.mul_vec v) = ![5, 2, -8].cross v :=
  sorry

end matrix_property_l796_796657


namespace mean_value_of_quadrilateral_interior_angles_l796_796173

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796173


namespace syrup_existence_iff_l796_796626

def can_obtain_30_percent_syrup (n : ℕ) : Prop :=
  ∃ (operations : list (char × char)), 
  let final_state := (3, n, 0) in -- initial (A, B, C)
  final_state = (10, 0, 0) ∧ -- final state (10 litres in A, 0 in B and C)
  ( -- Check if operations are valid
    ∀ op ε,
      ((op = ('B', 'C') ∧ (transferring_B_to_C op ε)) ∨
      (op = ('C', 'A') ∧ (transferring_C_to_A op ε)) ∨
      (op = ('A', 'Y') ∧ (pouring_away_A op ε))) 

noncomputable def valid_n_values : set ℕ :=
  { n | ∃ k ≥ 2, n = 3*k + 1 ∨ n = 3*k + 2 }

theorem syrup_existence_iff (n : ℕ) : 
  can_obtain_30_percent_syrup n ↔ n ∈ valid_n_values :=
begin
  sorry
end

end syrup_existence_iff_l796_796626


namespace percentage_discount_is_10_percent_l796_796599

def wholesale_price : ℝ := 90
def profit_percentage : ℝ := 0.20
def retail_price : ℝ := 120

theorem percentage_discount_is_10_percent :
  let profit := profit_percentage * wholesale_price in
  let selling_price := wholesale_price + profit in
  let discount_amount := retail_price - selling_price in
  let percentage_discount := (discount_amount / retail_price) * 100 in
  percentage_discount = 10 :=
by
  sorry

end percentage_discount_is_10_percent_l796_796599


namespace mean_value_interior_angles_quadrilateral_l796_796107

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796107


namespace maximum_subset_propertyT_l796_796936

/-- 
Property T definition: A subset B of {1, 2, ..., 2017} such that any three elements of B are
the sides of a nondegenerate triangle.
-/
def propertyT (B : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ B → b ∈ B → c ∈ B → a + b > c ∧ a + c > b ∧ b + c > a

/-- 
Main theorem stating the maximum number of elements a subset with property T
can contain.
-/
theorem maximum_subset_propertyT :
  ∃ (B : Finset ℕ), B ⊆ (Finset.range 2018).image (λ x, x + 1) ∧ propertyT B ∧ B.card = 1009 := 
sorry

end maximum_subset_propertyT_l796_796936


namespace csc_315_eq_neg_sqrt_two_l796_796990

theorem csc_315_eq_neg_sqrt_two : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_two_l796_796990


namespace total_tiles_l796_796610

theorem total_tiles (s : ℕ) (h1 : true) (h2 : true) (h3 : true) (h4 : true) (h5 : 4 * s - 4 = 100): s * s = 676 :=
by
  sorry

end total_tiles_l796_796610


namespace gcd_sequence_condition_l796_796438

open Nat Int

theorem gcd_sequence_condition (x y : ℤ) (a : ℕ → ℤ)
  (h0 : a 0 = 0) (h1 : a 1 = 0)
  (h_rec : ∀ n : ℕ, a (n + 2) = x * a (n + 1) + y * a n + 1)
  (p : ℕ) (hp : Prime p) : gcd (a p) (a (p + 1)) = 1 ∨ gcd (a p) (a (p + 1)) > sqrt p := by
  sorry

end gcd_sequence_condition_l796_796438


namespace find_original_number_l796_796247

def satisfies_conditions (N : ℕ) : Prop :=
  let N_sq_digits := to_base6_digits (N * N) in
  N * N ≥ 12345 ∧ N * N ≤ 54321 ∧ (∀ d, d ∈ N_sq_digits → d ≠ 0) ∧ (∀ d1 d2, d1 ∈ N_sq_digits → d2 ∈ N_sq_digits → d1 ≠ d2) ∧
  let rearranged := move_units_digit_to_beginning (N * N) in
  sqrt (to_nat rearranged) = to_nat (reverse (to_base10_digits (N * N)))

theorem find_original_number : ∃ (N : ℕ), 113 ≤ N ∧ N ≤ 221 ∧ satisfies_conditions N :=
begin
  use 221,
  split,
  { -- prove 113 ≤ 221
    norm_num },
  split,
  { -- prove 221 ≤ 221
    norm_num },
  { -- prove satisfies_conditions 221
    sorry
  }
end

end find_original_number_l796_796247


namespace find_t_l796_796300

def utility (hours_math hours_reading hours_painting : ℕ) : ℕ :=
  hours_math^2 + hours_reading * hours_painting

def utility_wednesday (t : ℕ) : ℕ :=
  utility 4 t (12 - t)

def utility_thursday (t : ℕ) : ℕ :=
  utility 3 (t + 1) (11 - t)

theorem find_t (t : ℕ) (h : utility_wednesday t = utility_thursday t) : t = 2 :=
by
  sorry

end find_t_l796_796300


namespace find_alpha_in_square_l796_796424

-- Definitions used in conditions
def is_square (ABCD : Type) : Prop := 
  ∃ A B C D : ABCD, 
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ 
    dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧ 
    ∀ (a b c d : ABCD), angle a b c = 90 ∧ angle b c d = 90 ∧ angle c d a = 90 ∧ angle d a b = 90

def segment_in_square (CE : Type) (square : Type) : Prop :=
  ∃ (C E : square), 
    C ≠ E

def triangle_with_angles (DEF : Type) (α : ℝ) : Prop :=
  ∃ D E F : DEF, 
    angle D E F = 7 * α ∧ angle E F D = 8 * α ∧ angle F D E = 90

-- The problem statement in Lean 4 proof format
theorem find_alpha_in_square
  (ABCD : Type) [is_square ABCD]
  (CE : Type) [segment_in_square CE ABCD]
  (DEF : Type) [triangle_with_angles DEF α]
  : α = 9° :=
sorry

end find_alpha_in_square_l796_796424


namespace ride_count_l796_796500

noncomputable def initial_tickets : ℕ := 287
noncomputable def spent_on_games : ℕ := 134
noncomputable def earned_tickets : ℕ := 32
noncomputable def cost_per_ride : ℕ := 17

theorem ride_count (initial_tickets : ℕ) (spent_on_games : ℕ) (earned_tickets : ℕ) (cost_per_ride : ℕ) : 
  initial_tickets = 287 ∧ spent_on_games = 134 ∧ earned_tickets = 32 ∧ cost_per_ride = 17 → (initial_tickets - spent_on_games + earned_tickets) / cost_per_ride = 10 :=
by
  intros
  sorry

end ride_count_l796_796500


namespace mean_value_of_quadrilateral_angles_l796_796009

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l796_796009


namespace option_a_option_b_option_c_option_d_l796_796896

theorem option_a (x : ℝ) (h : x > 1) : (min (x + 1 / (x - 1)) = 3) := 
sorry

theorem option_b (x : ℝ) : (min ((x ^ 2 + 5) / (sqrt (x ^ 2 + 4))) ≠ 2) := 
sorry

theorem option_c (x : ℝ) (h : 0 < x ∧ x < 10) : (max (sqrt (x * (10 - x))) = 5) := 
sorry

theorem option_d (x y : ℝ) (hx : 0 < x) (hy : true) (h : x + 2 * y = 3 * x * y) : (max (2 * x + y) ≠ 3) := 
sorry

end option_a_option_b_option_c_option_d_l796_796896


namespace calculate_speed_l796_796430

-- Given conditions
def distance : ℕ := 500  -- in meters
def time_in_minutes : ℕ := 6  -- in minutes

-- Conversion of time from minutes to hours
def time_in_hours : ℝ := time_in_minutes / 60.0

-- Statement to prove
theorem calculate_speed : (distance / time_in_hours) = 5000 := by
  sorry

end calculate_speed_l796_796430


namespace new_game_cost_l796_796899

theorem new_game_cost 
  (initial_money : ℕ) 
  (G : ℕ)
  (toys_cost : ℕ)
  (total_toys : ℕ)
  (remaining_money : initial_money - G = toys_cost * total_toys)
  (initial_money = 57)
  (toys_cost = 6)
  (total_toys = 5) :
  G = 27 :=
by
  sorry

end new_game_cost_l796_796899


namespace compute_sum_of_cubes_l796_796441

-- Define the polynomial
def polynomial := 3 * x^3 - 5 * x^2 + 90 * x - 2

-- Assume the roots of the polynomial
variables {a b c : ℝ}

-- Roots of the polynomial
axiom roots_of_polynomial : polynomial(a) = 0 ∧ polynomial(b) = 0 ∧ polynomial(c) = 0

-- Prove the given expression
theorem compute_sum_of_cubes :
  (a + b + 1)^3 + (b + c + 1)^3 + (c + a + 1)^3 = 259 + 1 / 3 :=
begin
  -- Use the axioms and the properties derived to prove the given expression
  sorry
end

end compute_sum_of_cubes_l796_796441


namespace probability_one_unit_apart_l796_796304

theorem probability_one_unit_apart :
  let n := 15
  let total_points := (3 * 2) + (2 * 2)
  let total_choices := Nat.choose n 2
  let favorable_choices := 16
  total_choices = 105 →
  favorable_choices = 16 →
  (favorable_choices.toRat / total_choices.toRat) = (16 / 105 : ℚ) :=
by
  intros
  sorry

end probability_one_unit_apart_l796_796304


namespace speed_of_stream_l796_796554

variable (x : ℝ)
variable (upstream_time downstream_time : ℝ)
variable (speed_boat_still_water : ℝ := 18)

-- Conditions:
def condition1 : Prop := upstream_time = 2 * downstream_time
def condition2 : Prop := sorry -- From speed of boat still water and effective speed expressions, derive the equation

-- Theorem to prove:
theorem speed_of_stream (h1 : condition1) (h2 : condition2) : x = 6 :=
sorry

end speed_of_stream_l796_796554


namespace jinsu_work_per_hour_l796_796429

theorem jinsu_work_per_hour (t : ℝ) (h : t = 4) : (1 / t = 1 / 4) :=
by {
    sorry
}

end jinsu_work_per_hour_l796_796429


namespace mean_of_quadrilateral_angles_l796_796150

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796150


namespace mean_value_of_quadrilateral_angles_l796_796132

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796132


namespace angle_BAC_is_72_degrees_l796_796529

theorem angle_BAC_is_72_degrees 
    {A B C : Point} 
    {O : Circle} 
    (tangent1 : Tangent A B O) 
    (tangent2 : Tangent A C O) 
    (arc_ratio : ArcLengthRatio O B C 1 4) 
    (larger_arc_semicircle : LargerArcSemicircle O B C) 
    : MeasureAngle A B C = 72 := 
by 
    sorry

end angle_BAC_is_72_degrees_l796_796529


namespace peter_son_is_nikolay_l796_796471

variable (x y : ℕ)

/-- Within the stated scenarios of Nikolai/Peter paired fishes caught -/
theorem peter_son_is_nikolay :
  (∀ n p ns ps : ℕ, (
    n = ns ∧              -- Nikolai caught as many fish as his son
    p = 3 * ps ∧          -- Peter caught three times more fish than his son
    n + ns + p + ps = 25  -- A total of 25 fish were caught
  ) → ("Nikolay" = "Peter's son")) := 
sorry

end peter_son_is_nikolay_l796_796471


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796066

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796066


namespace fraction_of_clerical_staff_is_one_third_l796_796921

-- Defining the conditions
variables (employees clerical_f clerical employees_reduced employees_remaining : ℝ)

def company_conditions (employees clerical_f clerical employees_reduced employees_remaining : ℝ) : Prop :=
  employees = 3600 ∧
  clerical = 3600 * clerical_f ∧
  employees_reduced = clerical * (2 / 3) ∧
  employees_remaining = employees - clerical * (1 / 3) ∧
  employees_reduced = 0.25 * employees_remaining

-- The statement to prove the fraction of clerical employees given the conditions
theorem fraction_of_clerical_staff_is_one_third
  (hc : company_conditions employees clerical_f clerical employees_reduced employees_remaining) :
  clerical_f = 1 / 3 :=
sorry

end fraction_of_clerical_staff_is_one_third_l796_796921


namespace mohit_discount_l796_796815

variable (SP : ℝ) -- Selling price
variable (CP : ℝ) -- Cost price
variable (discount_percentage : ℝ) -- Discount percentage

-- Conditions
axiom h1 : SP = 21000
axiom h2 : CP = 17500
axiom h3 : discount_percentage = ( (SP - (CP + 0.08 * CP)) / SP) * 100

-- Theorem to prove
theorem mohit_discount : discount_percentage = 10 :=
  sorry

end mohit_discount_l796_796815


namespace eval_f_at_800_l796_796808

-- Given conditions in Lean 4:
def f : ℝ → ℝ := sorry -- placeholder for the function definition
axiom func_eqn (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom f_at_1000 : f 1000 = 4

-- The goal/proof statement:
theorem eval_f_at_800 : f 800 = 5 := sorry

end eval_f_at_800_l796_796808


namespace sum_interior_angles_new_vertices_l796_796633

theorem sum_interior_angles_new_vertices (n : ℕ) (h : n ≥ 7) :
  let S := ∑ k in Finset.range n, (180 - 1080 / n) in
  S = 180 * (n - 6) := by
sorry

end sum_interior_angles_new_vertices_l796_796633


namespace estimated_population_correct_correlation_coefficient_correct_l796_796614

noncomputable def estimated_population
  (n_plots : ℕ) (n_sampled : ℕ) (sum_y_sampled : ℕ) : ℕ :=
  (sum_y_sampled / n_sampled) * n_plots

def correlation_coefficient
  (sum_xy_dev : ℝ) (sum_x_dev2 : ℝ) (sum_y_dev2 : ℝ) : ℝ :=
  sum_xy_dev / (Real.sqrt (sum_x_dev2 * sum_y_dev2))

theorem estimated_population_correct :
  estimated_population 200 20 1200 = 12000 :=
by 
  -- We skip the proof for GHC purposes
  sorry

theorem correlation_coefficient_correct :
  correlation_coefficient 800 (80 : ℝ) (9000 : ℝ) ≈ 0.94 :=
by
  -- We skip the proof for GHC purposes
  sorry

end estimated_population_correct_correlation_coefficient_correct_l796_796614


namespace mean_value_of_quadrilateral_angles_l796_796046

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796046


namespace power_reciprocal_identity_my_problem_l796_796888

theorem power_reciprocal_identity :
  ∀ (a : ℝ) (b : ℤ), a ≠ 0 → a^b * a^(-b) = 1 := 
by {
  intros,
  sorry
}

theorem my_problem : 
  (\(\frac{9}{10})^5 * (\(\frac{9}{10})^{-5} = 1 := 
by
  exact power_reciprocal_identity (\(\frac{9}{10} common
  sorry

#check power_reciprocal_identity

end power_reciprocal_identity_my_problem_l796_796888


namespace exists_cylinder_touching_planes_and_sphere_l796_796939

-- Conditions
variable {R : Type} [RealField R] -- We are working in a real field
variables (s₁ s₂ : Plane R) (S : Sphere R)

-- Theorem Statement
theorem exists_cylinder_touching_planes_and_sphere :
  ∃ (C : Cylinder R), touches C s₁ ∧ touches C s₂ ∧ touches C S :=
sorry

end exists_cylinder_touching_planes_and_sphere_l796_796939


namespace find_n_coeff_of_x4_l796_796365

open Classical

-- Define the condition
def condition (n : ℕ) : Prop := 3^n + 2^n = 275

-- Define the statement to prove the value of n
theorem find_n : ∃ n : ℕ, condition n ∧ n = 5 :=
by
  use 5
  unfold condition
  norm_num
  exact rfl

-- Define the binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the general term of the expansion
def general_term (n r : ℕ) : ℤ :=
  binomial_coefficient n r * 2^r * x^(2*(n - r) - r)

-- Define the statement to prove the coefficient of x^4
theorem coeff_of_x4 :
  (∃ (n : ℕ), condition n ∧ n = 5) →
  let r := 2 in
  let coeff := binomial_coefficient 5 r * 2^r in
  coeff = 40 :=
by
  intros h
  simp
  use 2
  unfold binomial_coefficient
  norm_num
  exact rfl

end find_n_coeff_of_x4_l796_796365


namespace mean_value_of_quadrilateral_angles_l796_796133

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796133


namespace second_chapter_pages_l796_796569

theorem second_chapter_pages (x : ℕ) (h1 : 48 = x + 37) : x = 11 := 
sorry

end second_chapter_pages_l796_796569


namespace mean_of_quadrilateral_angles_l796_796148

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796148


namespace probability_no_one_receives_own_letter_l796_796881

-- Define the factorial function for computing total permutations
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the number of derangements function
noncomputable def derangements (n : ℕ) : ℕ :=
  (factorial n) * ∑ k in finset.range (n + 1), ((-1 : ℚ)^k) / factorial k

-- The theorem stating the desired probability
theorem probability_no_one_receives_own_letter : 
  (derangements 4 : ℚ) / factorial 4 = 3 / 8 :=
by
  sorry

end probability_no_one_receives_own_letter_l796_796881


namespace solve_equation_l796_796840

theorem solve_equation (x : ℝ) : (x + 3) * (x - 1) = 12 ↔ (x = -5 ∨ x = 3) := sorry

end solve_equation_l796_796840


namespace number_of_possible_winning_scores_l796_796405

noncomputable def sum_of_first_n_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem number_of_possible_winning_scores : 
  let total_sum := sum_of_first_n_integers 12
  let max_possible_score := total_sum / 2
  let min_possible_score := sum_of_first_n_integers 6
  39 - 21 + 1 = 19 := 
by
  sorry

end number_of_possible_winning_scores_l796_796405


namespace gain_percent_l796_796228

variables (MP CP SP : ℝ)

-- problem conditions
axiom h1 : CP = 0.64 * MP
axiom h2 : SP = 0.84 * MP

-- To prove: Gain percent is 31.25%
theorem gain_percent (CP MP SP : ℝ) (h1 : CP = 0.64 * MP) (h2 : SP = 0.84 * MP) :
  ((SP - CP) / CP) * 100 = 31.25 :=
by sorry

end gain_percent_l796_796228


namespace distance_between_adjacent_symmetry_axes_l796_796714

noncomputable def f (x : ℝ) : ℝ := (Real.cos (3 * x))^2 - 1/2

theorem distance_between_adjacent_symmetry_axes :
  (∃ x : ℝ, f x = f (x + π / 3)) → (∃ d : ℝ, d = π / 6) :=
by
  -- Prove the distance is π / 6 based on the properties of f(x).
  sorry

end distance_between_adjacent_symmetry_axes_l796_796714


namespace honey_harvested_correct_l796_796972

def honey_harvested_last_year : ℕ := 2479
def honey_increase_this_year : ℕ := 6085
def honey_harvested_this_year : ℕ := 8564

theorem honey_harvested_correct :
  honey_harvested_last_year + honey_increase_this_year = honey_harvested_this_year :=
sorry

end honey_harvested_correct_l796_796972


namespace units_digit_sum_sequence_is_1_l796_796538

def a (n : Nat) : Nat := n.factorial + n

theorem units_digit_sum_sequence_is_1 : 
  let sum_units_digits := (∑ n in Finset.range 12 | 1 ≤ n ∧ n ≤ 12, (a n) % 10)
  (sum_units_digits % 10 = 1) := by
  sorry

end units_digit_sum_sequence_is_1_l796_796538


namespace slope_angle_of_line_l796_796872

theorem slope_angle_of_line : 
  ∀ α : ℝ, 0 ≤ α ∧ α < π ∧ (∀ x y : ℝ, (y = sqrt 3 * x + 2) → tan α = sqrt 3) → α = π / 3 :=
by
  intros α h₁ h₂ x y h₃
  sorry

end slope_angle_of_line_l796_796872


namespace find_eccentricity_l796_796870

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : ∀ (P Q : ℝ × ℝ), P ≠ Q → 
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧ 
    (Q.1^2 / a^2 + Q.2^2 / b^2 = 1) ∧
    (sqrt((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = a) ∧ 
    (P.1 = a) ∧ 
    (Q.1 ≠ P.1 ∨ Q.2 ≠ P.2) ∧ 
    let O : ℝ × ℝ := (0, 0) in (P.1-O.1)*(Q.1-O.1) + (P.2-O.2)*(Q.2-O.2) = 0) : 
  ℝ := 
    let c_squared : ℝ := (a^2 - let b_squared := b^2 in b_squared) in 
    sqrt (c_squared / a^2)

theorem find_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : ∀ (P Q : ℝ × ℝ), P ≠ Q → (P.1^2 / a^2 + P.2^2 / b^2 = 1 ) ∧ 
    (Q.1^2 / a^2 + Q.2^2 / b^2 = 1) ∧
    (sqrt((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = a) ∧ 
    (P.1 = a) ∧ 
    (Q.1 ≠ P.1 ∨ Q.2 ≠ P.2) ∧ 
    let O : ℝ × ℝ := (0, 0) in (P.1-O.1)*(Q.1-O.1) + (P.2-O.2)*(Q.2-O.2) = 0) : 
  eccentricity_of_ellipse a b h1 h2 h3 h4 = 2 * sqrt(5) / 5 :=
begin
  sorry
end

end find_eccentricity_l796_796870


namespace mean_value_interior_angles_quadrilateral_l796_796114

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796114


namespace mean_value_of_quadrilateral_angles_l796_796131

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796131


namespace division_problem_l796_796820

theorem division_problem (n : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h_div : divisor = 12) (h_quo : quotient = 9) (h_rem : remainder = 1) 
  (h_eq: n = divisor * quotient + remainder) : n = 109 :=
by
  sorry

end division_problem_l796_796820


namespace factorize_cubic_expression_l796_796650

theorem factorize_cubic_expression (m : ℝ) : 
  m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end factorize_cubic_expression_l796_796650


namespace cotangent_inequality_l796_796834

variable {A B C : ℕ}
variable {a b c : ℝ} -- side lengths of triangle ABC
variable {cotA cotB cotC : ℝ} -- cotangents of angles A, B, and C
variable {area : ℝ} -- area of triangle ABC

-- Assuming the triangle ABC is acute
axiom acute_triangle (A B C : ℕ) : A < 90 ∧ B < 90 ∧ C < 90

-- Define the cotangent of the angles using side lengths
def cotangent (A B C : ℝ) : ℝ := cotA + cotB + cotC

-- Define the area of the triangle
def triangle_area (a b c : ℝ) : ℝ := area

-- Main theorem to be proved
theorem cotangent_inequality (a b c : ℝ) (area : ℝ) (cotA cotB cotC : ℝ) (A B C : ℕ) :
  acute_triangle A B C →
  cotangent A B C ≥ 12 * triangle_area a b c / (a^2 + b^2 + c^2) :=
begin
  sorry
end

end cotangent_inequality_l796_796834


namespace csc_315_equals_neg_sqrt_2_l796_796982

-- Define the conditions
def csc (x : Real) : Real := 1 / Real.sin x
axiom sin_subtraction_360 (x : Real) : Real.sin (360 - x) = -Real.sin x

-- State the problem
theorem csc_315_equals_neg_sqrt_2 : csc 315 = -Real.sqrt 2 := by
  sorry

end csc_315_equals_neg_sqrt_2_l796_796982


namespace parabola_line_intersection_l796_796582

noncomputable def parabolaFocus : ℝ × ℝ := (1, 0)
def parabolaEquation (x y : ℝ) : Prop := y^2 = 4 * x
def lineInclinationAngle : ℝ := real.pi / 4
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
def absDifference (a b : ℝ) : ℝ := abs (b - a)

theorem parabola_line_intersection :
  ∃ A B : ℝ × ℝ, parabolaEquation A.1 A.2 ∧ parabolaEquation B.1 B.2 ∧ 
  distance (1, 0) A = 4 + 2 * real.sqrt 2 ∧ distance (1, 0) B = 4 - 2 * real.sqrt 2 ∧
  absDifference (distance (1, 0) A) (distance (1, 0) B) = 4 * real.sqrt 2 :=
by
  sorry

end parabola_line_intersection_l796_796582


namespace sphere_eqn_standard_form_l796_796656

-- Define the center and radius of the sphere
variables {a b c R : ℝ}

-- Define the sphere equation as the target proposition
def sphere_equation (x y z : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 + (z - c)^2 = R^2

-- State the theorem for the sphere equation
theorem sphere_eqn_standard_form (x y z : ℝ) (h : (x - a)^2 + (y - b)^2 + (z - c)^2 = R^2) :
  sphere_equation x y z :=
  by
    sorry

end sphere_eqn_standard_form_l796_796656


namespace cars_on_tuesday_is_25_l796_796523

-- Define the number of cars passing on Tuesday
def cars_on_tuesday (T : ℝ) : ℝ :=
  T

-- Define the different day conditions based on T
def cars_on_monday (T : ℝ) : ℝ :=
  0.80 * T

def cars_on_wednesday (T : ℝ) : ℝ :=
  0.80 * T + 2

def cars_thursday_friday : ℝ :=
  10 + 10

def cars_weekend : ℝ :=
  5 + 5

-- Define the total number of cars over the week in terms of T
def total_cars (T : ℝ) : ℝ :=
  cars_on_monday T + T + cars_on_wednesday T + cars_thursday_friday + cars_weekend

-- The proof statement in Lean 4
theorem cars_on_tuesday_is_25 : ∃ T : ℝ, total_cars T = 97 ∧ cars_on_tuesday T = 25 :=
by
  use 25
  calc
    total_cars 25 = (0.80 * 25 : ℝ) + 25 + (0.80 * 25 + 2) + 20 + 10 := rfl
    ... = 20 + 25 + 22 + 20 + 10  := by norm_num
    ... = 97 := by norm_num
  sorry -- Proof statement placeholder

end cars_on_tuesday_is_25_l796_796523


namespace find_dot_product_l796_796353

variables {V : Type*} [inner_product_space ℝ V]
variables {A B C O : V}

-- Definitions as per conditions
def condition1 : Prop := 
  (∥O - A∥ = 1) ∧ (∥O - B∥ = 1) ∧ (∥O - C∥ = 1)

def condition2 : Prop := 
  (3 • (O - A) + 4 • (O - B) + 5 • (O - C) = 0)

theorem find_dot_product 
  (h1 : condition1) 
  (h2 : condition2) : 
  inner (B - A) (C - A) = 4 / 5 :=
by
  sorry

end find_dot_product_l796_796353


namespace mean_of_quadrilateral_angles_l796_796143

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796143


namespace ax0_eq_b_condition_l796_796334

theorem ax0_eq_b_condition (a b x0 : ℝ) (h : a < 0) : (ax0 = b) ↔ (∀ x : ℝ, (1/2 * a * x^2 - b * x) ≤ (1/2 * a * x0^2 - b * x0)) :=
sorry

end ax0_eq_b_condition_l796_796334


namespace mean_value_of_quadrilateral_interior_angles_l796_796172

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796172


namespace order_A_C_B_l796_796387

noncomputable def A (a b : ℝ) : ℝ := Real.log ((a + b) / 2)
noncomputable def B (a b : ℝ) : ℝ := Real.sqrt (Real.log a * Real.log b)
noncomputable def C (a b : ℝ) : ℝ := (Real.log a + Real.log b) / 2

theorem order_A_C_B (a b : ℝ) (h1 : 1 < b) (h2 : b < a) :
  A a b > C a b ∧ C a b > B a b :=
by 
  sorry

end order_A_C_B_l796_796387


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796015

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796015


namespace mean_value_of_quadrilateral_angles_l796_796122

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796122


namespace exists_24_skew_colored_quadrilaterals_l796_796408

theorem exists_24_skew_colored_quadrilaterals :
  ∀ (V : Finset (Fin 100))
    (coloring : Fin 100 → Prop)
    (black_vertices white_vertices : Finset (Fin 100)),
    V.card = 100 →
    black_vertices.card = 41 →
    white_vertices.card = 59 →
    (∀ v ∈ V, (coloring v ↔ v ∈ black_vertices) ∨ (¬ coloring v ↔ v ∈ white_vertices)) →
    ∃ Q : Finset (Finset (Fin 100)), 
      Q.card = 24 ∧
      (∀ q ∈ Q, q.card = 4 ∧ q.subset V ∧
        ((∃ b ∈ q, ∀ w ∈ q, w ≠ b → ¬coloring w) ∨
        (∃ w ∈ q, ∀ b ∈ q, b ≠ w → coloring b))) ∧
      (∀ q1 q2 ∈ Q, q1 ≠ q2 → q1 ∩ q2 = ∅) :=
sorry

end exists_24_skew_colored_quadrilaterals_l796_796408


namespace part_a_part_b_l796_796267

-- Part (a): Show that every subset S of {1, 2, ..., n} with |S| > 3n/4 is balanced when n = 2k and k > 1
theorem part_a (k : ℕ) (h1 : k > 1) (S : Finset ℕ) 
  (h2 : S ⊆ Finset.range (2 * k + 1)) (h3 : S.card > (3 * (2 * k)) / 4) :
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ((a + b) / 2) ∈ S :=
by sorry

-- Part (b): Show that there exists some k > 1 such that for n = 2k, there exists a subset S with |S| > 2n/3 that is not balanced
theorem part_b : ∃ (k : ℕ) (S : Finset ℕ), k > 1 ∧ S ⊆ Finset.range (2 * k + 1) ∧ 
  S.card > (2 * (2 * k)) / 3 ∧ ¬(∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ((a + b) / 2) ∈ S) :=
by
  exists 3
  let S := Finset.ofList [1, 2, 3, 4, 5]
  use S
  split
  sorry

end part_a_part_b_l796_796267


namespace wine_remaining_percentage_l796_796933

theorem wine_remaining_percentage :
  let initial_wine := 250.0 -- initial wine in liters
  let daily_fraction := (249.0 / 250.0)
  let days := 50
  let remaining_wine := (daily_fraction ^ days) * initial_wine
  let percentage_remaining := (remaining_wine / initial_wine) * 100
  percentage_remaining = 81.846 :=
by
  sorry

end wine_remaining_percentage_l796_796933


namespace find_first_discount_l796_796510

theorem find_first_discount (price_initial : ℝ) (price_final : ℝ) (discount_additional : ℝ) (x : ℝ) :
  price_initial = 350 → price_final = 266 → discount_additional = 5 →
  price_initial * (1 - x / 100) * (1 - discount_additional / 100) = price_final →
  x = 20 :=
by
  intros h1 h2 h3 h4
  -- skippable in proofs, just holds the place
  sorry

end find_first_discount_l796_796510


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796014

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796014


namespace mean_of_quadrilateral_angles_l796_796037

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796037


namespace circle_area_increase_l796_796399

theorem circle_area_increase (r : ℝ) : 
  let r_new := 2.5 * r
      A_original := π * r^2
      A_new := π * r_new^2
  in (A_new - A_original) / A_original * 100 = 525 := 
by
  let r_new := 2.5 * r
  let A_original := π * r^2
  let A_new := π * r_new^2
  have h : A_new = π * (2.5 * r)^2 := by sorry
  have h1 : A_original = π * r^2 := by sorry
  have h2 : A_new - A_original = π * (2.5 * r)^2 - π * r^2 := by sorry
  have h3 : (A_new - A_original) / A_original * 100 = (π * (2.5 * r)^2 - π * r^2) / (π * r^2) * 100 := by sorry
  have h4 : (A_new - A_original) / A_original * 100 = (π * 6.25 * r^2 - π * r^2) / (π * r^2) * 100 := by sorry
  have h5 : (A_new - A_original) / A_original * 100 = (π * 5.25 * r^2) / (π * r^2) * 100 := by sorry
  have h6 : (A_new - A_original) / A_original * 100 = 5.25 * 100 := by sorry
  exact eq.trans h6 (by sorry)

end circle_area_increase_l796_796399


namespace find_u_v_w_x_l796_796345

variables (P P' Q Q' R R' S S' : Type)

-- Conditions
def is_midpoint (X Y Z : Type) : Prop := X = (1 / 2 * Y + 1 / 2 * Z)

-- Proof problem
theorem find_u_v_w_x
  (hPQ : is_midpoint P Q P')
  (hQR : is_midpoint Q R Q')
  (hRS : is_midpoint R S R')
  (hSP : is_midpoint S P S') :
  ∃ (u v w x : ℝ), P = u * P' + v * Q' + w * R' + x * S' ∧
  u = 1 / 15 ∧ v = 2 / 15 ∧ w = 4 / 15 ∧ x = 8 / 15 := sorry

end find_u_v_w_x_l796_796345


namespace mean_value_of_quadrilateral_angles_l796_796206

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796206


namespace find_x_given_area_l796_796320

theorem find_x_given_area (x : ℝ) (h_pos : x > 0) (h_area : 1/2 * x * 3 * x = 72) : x = 4 * real.sqrt 3 :=
sorry

end find_x_given_area_l796_796320


namespace correct_inequality_condition_l796_796240

theorem correct_inequality_condition : 
  ∀ (a b c d : Prop), 
    (a = (1.7^2.5 > 1.7^3)) ∧ 
    (b = (log 0.2 3 < log 0.2 5)) ∧ 
    (c = (1.7^3.1 < 0.9^3.1)) ∧ 
    (d = (log 3 0.2 < log 0.2 0.3)) → 
    d = true :=
by
  intros a b c d h
  sorry

end correct_inequality_condition_l796_796240


namespace largest_divisor_of_expression_l796_796388

theorem largest_divisor_of_expression
  (x : ℤ) (h_odd : x % 2 = 1) : 
  ∃ k : ℤ, k = 40 ∧ 40 ∣ (12 * x + 2) * (8 * x + 14) * (10 * x + 10) :=
by
  sorry

end largest_divisor_of_expression_l796_796388


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796071

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796071


namespace building_height_l796_796226

theorem building_height
    (flagpole_height : ℝ)
    (flagpole_shadow_length : ℝ)
    (building_shadow_length : ℝ)
    (h : ℝ)
    (h_eq : flagpole_height / flagpole_shadow_length = h / building_shadow_length)
    (flagpole_height_eq : flagpole_height = 18)
    (flagpole_shadow_length_eq : flagpole_shadow_length = 45)
    (building_shadow_length_eq : building_shadow_length = 65) :
  h = 26 := by
  sorry

end building_height_l796_796226


namespace no_polynomials_with_given_roots_l796_796635

def polynomial_has_roots (b : Fin 10 → ℕ) : Prop :=
  ∃ p : ℕ[X], (p.coeff 0 = 0) ∧
              ∀ r ∈ [-1, 0, 1], p.eval r = 0 ∧
              ∀ i : Fin 10, p.coeff (i + 1) ∈ {0, 1}

theorem no_polynomials_with_given_roots :
  ¬ ∃ b : Fin 10 → ℕ, polynomial_has_roots b :=
by
  sorry

end no_polynomials_with_given_roots_l796_796635


namespace seating_arrangements_count_l796_796879

noncomputable def total_seating_arrangements : ℕ := 276

theorem seating_arrangements_count :
  let front_row_seats := 10 in
  let back_row_seats := 11 in
  let middle_back_not_used := 3 in
  let total_seats := front_row_seats + back_row_seats - middle_back_not_used in
  let cannot_be_next_to_each_other := true in
  (∀ people : ℕ, people = 2 → 
  (total_seats - middle_back_not_used) ≥ people *
  (cannot_be_next_to_each_other → 
  (front_row_seats * (back_row_seats - middle_back_not_used)) +
  (binomial (back_row_seats - middle_back_not_used, 2)) +
  (front_row_seats * (back_row_seats - middle_back_not_used - people))
  = total_seating_arrangements)) := sorry

end seating_arrangements_count_l796_796879


namespace factorization_bound_l796_796322

-- Definitions based on conditions
def f (k : ℕ) : ℕ := sorry  -- placeholder for the function definition that should be given

-- Main statement to prove
theorem factorization_bound (n p : ℕ) (h1 : 1 < n) (h2 : p.prime) (h3 : p ∣ n) : f n ≤ n / p :=
sorry  -- proof not required

end factorization_bound_l796_796322


namespace expression_equals_6_l796_796539

-- Define the expression as a Lean definition.
def expression : ℤ := 1 - (-2) - 3 - (-4) - 5 - (-6) - 7 - (-8)

-- The statement to prove that the expression equals 6.
theorem expression_equals_6 : expression = 6 := 
by 
  -- This is a placeholder for the actual proof.
  sorry

end expression_equals_6_l796_796539


namespace probability_A_complement_exactly_k_times_l796_796536

theorem probability_A_complement_exactly_k_times (n k : ℕ) (P : ℝ) (hP : 0 ≤ P ∧ P ≤ 1) :
  (nat.choose n k) * ((1 - P) ^ k) * (P ^ (n - k)) = C_n^k(1-P)^k P^{n-k} :=
by
  sorry

end probability_A_complement_exactly_k_times_l796_796536


namespace minimum_positive_period_of_sec_mul_sin_l796_796503

open Real

-- Define the function sec(x) * sin(x)
noncomputable def f (x : ℝ) : ℝ := sec x * sin x

-- Statement of the problem
theorem minimum_positive_period_of_sec_mul_sin : ∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T' :=
sorry

end minimum_positive_period_of_sec_mul_sin_l796_796503


namespace mean_value_of_quadrilateral_angles_l796_796136

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796136


namespace new_weekly_hours_l796_796740

-- Conditions
def initial_hours_per_week : ℝ := 25
def total_weeks : ℝ := 15
def total_earnings : ℝ := 3750
def missed_weeks : ℝ := 3

-- Remaining weeks
def remaining_weeks : ℝ := total_weeks - missed_weeks

-- The new required weekly hours
def new_required_weekly_hours (initial_hours_per_week: ℝ) (remaining_weeks: ℝ) (total_weeks: ℝ): ℝ :=
  (initial_hours_per_week * total_weeks) / remaining_weeks

-- Statement to prove
theorem new_weekly_hours : new_required_weekly_hours initial_hours_per_week remaining_weeks total_weeks = 31.25 := by
  sorry

end new_weekly_hours_l796_796740


namespace product_not_end_in_1999_l796_796873

theorem product_not_end_in_1999 (a b c d e : ℕ) (h : a + b + c + d + e = 200) : 
  ¬(a * b * c * d * e % 10000 = 1999) := 
by
  sorry

end product_not_end_in_1999_l796_796873


namespace problem_statement_l796_796792

theorem problem_statement (p q : ℝ)
  (α β : ℝ) (h1 : α ≠ β) (h1' : α + β = -p) (h1'' : α * β = -2)
  (γ δ : ℝ) (h2 : γ ≠ δ) (h2' : γ + δ = -q) (h2'' : γ * δ = -3) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = 3 * (q ^ 2 - p ^ 2) - 2 * q + 1 := by
  sorry

end problem_statement_l796_796792


namespace log_sum_value_l796_796229

theorem log_sum_value :
  real.logb 9 27 + real.logb 8 32 = 19 / 6 := by
  sorry

end log_sum_value_l796_796229


namespace calculate_fraction_l796_796891

theorem calculate_fraction :
  (5 * 6 - 4) / 8 = 13 / 4 := 
by
  sorry

end calculate_fraction_l796_796891


namespace mean_value_of_quadrilateral_angles_l796_796005

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l796_796005


namespace minor_arc_prob_l796_796477

theorem minor_arc_prob (A B : ℝ) (hA : 0 ≤ A ∧ A < 3) (hB : 0 ≤ B ∧ B < 3) :
  let length_of_minor_arc := if B < A then min (3 - A + B) (A - B) else min (B - A) (3 - B + A)
  in ∃ p : ℝ, (∀ B, p = if length_of_minor_arc < 1 then 1/2 else 1/2) ∧ p = 1/2 :=
by
  sorry

end minor_arc_prob_l796_796477


namespace log_prod_eq_six_l796_796631

theorem log_prod_eq_six : log 2 9 * log 3 8 = 6 := 
by sorry

end log_prod_eq_six_l796_796631


namespace mean_value_of_quadrilateral_angles_l796_796128

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796128


namespace xiaodong_election_l796_796225

theorem xiaodong_election (V : ℕ) (h1 : 0 < V) :
  let total_needed := (3 : ℚ) / 4 * V
  let votes_obtained := (5 : ℚ) / 6 * (2 : ℚ) / 3 * V
  let remaining_votes := V - (2 : ℚ) / 3 * V
  total_needed - votes_obtained = (7 : ℚ) / 12 * remaining_votes :=
by 
  sorry

end xiaodong_election_l796_796225


namespace earnings_last_friday_l796_796431

theorem earnings_last_friday 
  (price_per_kg : ℕ := 2)
  (earnings_wednesday : ℕ := 30)
  (earnings_today : ℕ := 42)
  (total_kg_sold : ℕ := 48)
  (total_earnings : ℕ := total_kg_sold * price_per_kg) 
  (F : ℕ) :
  earnings_wednesday + F + earnings_today = total_earnings → F = 24 := by
  sorry

end earnings_last_friday_l796_796431


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796016

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796016


namespace molecular_weight_cao_is_correct_l796_796282

-- Define the atomic weights of calcium and oxygen
def atomic_weight_ca : ℝ := 40.08
def atomic_weight_o : ℝ := 16.00

-- Define the molecular weight of CaO
def molecular_weight_cao : ℝ := atomic_weight_ca + atomic_weight_o

-- State the theorem to prove
theorem molecular_weight_cao_is_correct : molecular_weight_cao = 56.08 :=
by
  sorry

end molecular_weight_cao_is_correct_l796_796282


namespace intersection_points_of_curve_and_line_l796_796295

theorem intersection_points_of_curve_and_line :
  let f := λ x : ℝ, 9 / (x^2 + 1)
  let g := λ x : ℝ, 3 - x
  (f 2 = g 2) ∧ (f (1 + sqrt 13) / 2 = g (1 + sqrt 13) / 2) ∧ (f (1 - sqrt 13) / 2 = g (1 - sqrt 13) / 2) :=
by
  sorry

end intersection_points_of_curve_and_line_l796_796295


namespace count_valid_arrangements_l796_796619

-- Each circle is represented as a vertex in a graph, with edges representing adjacency between circles
structure CircleArrangement :=
  (a b c d e : ℕ)
  (distinct : List.nodup [a, b, c, d, e])
  (diff_adj : (|a - d| ≥ 2) ∧ (|a - e| ≥ 2) ∧ (|b - d| ≥ 2) ∧ (|b - e| ≥ 2) ∧
               (|c - d| ≥ 2) ∧ (|c - e| ≥ 2))
  (numbers : a ∈ [2, 3, 4, 5, 6] ∧ b ∈ [2, 3, 4, 5, 6] ∧ c ∈ [2, 3, 4, 5, 6] ∧ 
            d ∈ [2, 3, 4, 5, 6] ∧ e ∈ [2, 3, 4, 5, 6])

theorem count_valid_arrangements : 
  (finset.univ.filter (λ s : CircleArrangement, true)).card = 3 :=
by
  sorry

end count_valid_arrangements_l796_796619


namespace area_in_terms_of_x_range_of_x_coordinates_when_S_is_6_l796_796352

-- Definition of the points and the condition
def A : ℝ × ℝ := (4, 0)
variable x y : ℝ
axiom h1 : x + y = 6
axiom h2 : 0 < x
axiom h3 : x < 6

def S (x y : ℝ) : ℝ := 2 * y

-- Statements to be proved
theorem area_in_terms_of_x : S x (6 - x) = -2 * x + 12 :=
by sorry

theorem range_of_x : 0 < x ∧ x < 6 :=
by sorry

theorem coordinates_when_S_is_6 : S x (6 - x) = 6 → (x, 6 - x) = (3, 3) :=
by sorry

end area_in_terms_of_x_range_of_x_coordinates_when_S_is_6_l796_796352


namespace geese_more_than_ducks_l796_796624

-- Define initial conditions
def initial_ducks : ℕ := 25
def initial_geese : ℕ := 2 * initial_ducks - 10
def additional_ducks : ℕ := 4
def geese_leaving : ℕ := 15 - 5

-- Calculate the number of remaining ducks
def remaining_ducks : ℕ := initial_ducks + additional_ducks

-- Calculate the number of remaining geese
def remaining_geese : ℕ := initial_geese - geese_leaving

-- Prove the final statement
theorem geese_more_than_ducks : (remaining_geese - remaining_ducks) = 1 := by
  sorry

end geese_more_than_ducks_l796_796624


namespace express_p_in_terms_of_q_l796_796745

variable (p q : ℝ)
variable (h1 : p = log 4 400)
variable (h2 : q = log 2 20)

theorem express_p_in_terms_of_q : p = q := by
  sorry

end express_p_in_terms_of_q_l796_796745


namespace inequality_min_bound_l796_796437

theorem inequality_min_bound (n : ℕ) (x y : Fin n → ℝ) 
  (h_pos_x : ∀ i, 0 < x i) (h_pos_y : ∀ i, 0 < y i)
  (h_sum_x : ∑ i, x i = 1) (h_sum_y : ∑ i, y i = 1) :
  ∑ i, abs (x i - y i) ≤ 2 - min (1 < n) (λ i, x i / y i) - min (1 < n) (λ i, y i / x i) := 
sorry

end inequality_min_bound_l796_796437


namespace tan_alpha_is_half_calc_value_l796_796368

-- Define the angle α
variable {α : ℝ}

-- Given condition: (5 * sin(α) - cos(α)) / (cos(α) + sin(α)) = 1
def condition_1 : Prop :=
  (5 * Real.sin α - Real.cos α) / (Real.cos α + Real.sin α) = 1

-- Question 1: Prove that tan(α) = 1/2 given the condition
theorem tan_alpha_is_half (h : condition_1) : Real.tan α = 1 / 2 :=
  sorry

-- Question 2: Prove the specified expression equals 17/5 given tan(α) = 1/2
theorem calc_value (h : Real.tan α = 1 / 2) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) + Real.sin α * Real.cos α = 17 / 5 :=
  sorry

end tan_alpha_is_half_calc_value_l796_796368


namespace mean_value_of_quadrilateral_interior_angles_l796_796180

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796180


namespace volume_of_cone_l796_796706

def coneGeneratrixLength := 2
def sectorArea := 2 * Real.pi

theorem volume_of_cone (r h V : ℝ) :
  (1 / 2 * (2 * Real.pi * r) * coneGeneratrixLength = sectorArea) →
  (h = Real.sqrt (coneGeneratrixLength^2 - r^2)) →
  (V = 1 / 3 * Real.pi * r^2 * h) →
  V = Real.sqrt(3) * Real.pi / 3 :=
by
  sorry

end volume_of_cone_l796_796706


namespace rectangle_area_difference_l796_796739

theorem rectangle_area_difference :
  let area (l w : ℝ) := l * w
  let combined_area (l w : ℝ) := 2 * area l w
  combined_area 11 19 - combined_area 9.5 11 = 209 :=
by
  sorry

end rectangle_area_difference_l796_796739


namespace mean_value_interior_angles_quadrilateral_l796_796110

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796110


namespace mean_of_quadrilateral_angles_l796_796192

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796192


namespace largest_k_996_l796_796321

def euler_totient (n : ℕ) : ℕ := n.totient

def S (k : ℕ) : ℚ :=
  ∑ d in (finset.range (42^k).succ).filter (λ n, (42^k) % n = 0), (euler_totient d : ℚ) / d

theorem largest_k_996 :
  ∃ k, k < 1000 ∧ (S k).denom = 1 ∧ ∀ m, m < 1000 → (S m).denom = 1 → m ≤ k :=
begin
  use 996,
  split,
  { exact nat.lt_trans (996 : ℕ).lt_succ_self 1000 }
  sorry
end

end largest_k_996_l796_796321


namespace find_a_range_l796_796369

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else (a + 1) / x

theorem find_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) → 
  - (7 / 2) ≤ a ∧ a ≤ -2 :=
by
  sorry

end find_a_range_l796_796369


namespace mean_value_of_quadrilateral_interior_angles_l796_796185

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796185


namespace total_earrings_l796_796947

-- Definitions based on the given conditions
def bella_earrings : ℕ := 10
def monica_earrings : ℕ := 4 * bella_earrings
def rachel_earrings : ℕ := monica_earrings / 2
def olivia_earrings : ℕ := bella_earrings + monica_earrings + rachel_earrings + 5

-- The theorem to prove the total number of earrings
theorem total_earrings : bella_earrings + monica_earrings + rachel_earrings + olivia_earrings = 145 := by
  sorry

end total_earrings_l796_796947


namespace mean_value_of_quadrilateral_angles_l796_796158

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796158


namespace standard_equation_of_C2_equation_of_locus_minimal_area_AMB_l796_796710

noncomputable def conditions : Prop :=
  ∃ (a b : ℝ),
  a > b ∧ b > 0 ∧
  2 * a * b = 4 * sqrt 5 ∧
  (a * b) / sqrt (a^2 + b^2) = (2 * sqrt 5) / 3

-- 1. Prove the standard equation of ellipse C2
theorem standard_equation_of_C2 : conditions → 
  ∃ (a b : ℝ), (a = 5 ∧ b = 4) → (∀ x y : ℝ, (x^2 / a + y^2 / b = 1)) :=
begin
  sorry
end

-- 2.1 Prove the equation of the locus of point M
theorem equation_of_locus (λ : ℝ) : conditions →
  ∀ (M : ℝ × ℝ) (O : ℝ × ℝ) (A : ℝ × ℝ), (M ≠ O) →
  |(M.1 - O.1)^2 + (M.2 - O.2)^2| = λ^2 * |(A.1 - O.1)^2 + (A.2 - O.2)^2| →
  ( ∀ x y : ℝ, (x^2 / 4 + y^2 / 5 = λ^2) ) :=
begin
  sorry
end

-- 2.2 Prove the minimal value of the area of ΔAMB
theorem minimal_area_AMB : conditions → 
  ∀ (A M B O : ℝ × ℝ), A ≠ B → M ≠ O →
  area_of_triangle AMB O = 40 / 9 :=
begin
  sorry
end

end standard_equation_of_C2_equation_of_locus_minimal_area_AMB_l796_796710


namespace range_of_a_for_local_min_l796_796339

noncomputable def f (a x : ℝ) : ℝ := (x - 2 * a) * (x^2 + a^2 * x + 2 * a^3)

theorem range_of_a_for_local_min :
  (∀ a : ℝ, (∃ δ > 0, ∀ ε ∈ Set.Ioo (-δ) δ, f a ε > f a 0) → a < 0 ∨ a > 2) :=
by
  sorry

end range_of_a_for_local_min_l796_796339


namespace not_appear_during_calculation_l796_796629

def polynomial (x : ℤ) : ℤ := 7 * x^3 + 3 * x^2 - 5 * x + 11

def horner_step1 (x : ℤ) := 7

def horner_step2 (v0 x : ℤ) := v0 * x + 3

def horner_step3 (v1 x : ℤ) := v1 * x - 5

def horner_step4 (v2 x : ℤ) := v2 * x + 11

theorem not_appear_during_calculation : 
  ∀ (x : ℤ), x = 23 → 
  let v0 := horner_step1 x in
  let v1 := horner_step2 v0 x in
  let v2 := horner_step3 v1 x in
  let v3 := horner_step4 v2 x in
  v0 ≠ 85169 ∧
  v1 ≠ 85169 ∧
  v2 ≠ 85169 ∧
  v3 ≠ 85169 :=
by
  intros
  unfold polynomial horner_step1 horner_step2 horner_step3 horner_step4
  sorry

end not_appear_during_calculation_l796_796629


namespace mean_value_of_quadrilateral_angles_l796_796002

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l796_796002


namespace tangent_AX_circumcircle_l796_796807

noncomputable theory

variables {A B C D P Q X : Type} [triangle_ABC : triangle A B C] [circumcircle_ABC : circle A B C]
variables [AB_gt_AC : AB > AC] [angle_bisector_BAC : angle_bisector A B C D]
variables [circle_BD : circle B D] [circle_CD : circle C D]
variables [P_on_circumcircle : P ≠ B] [Q_on_circumcircle : Q ≠ C]
variables [PQ_intersect_BC : intersection PQ BC X]

theorem tangent_AX_circumcircle (hx : ∃ X, lies_on X (intersection_line PQ BC)):
  is_tangent ((segment A X) : line) (circumcircle_ABC : circle) :=
sorry

end tangent_AX_circumcircle_l796_796807


namespace line_symmetry_probability_l796_796609

-- Definitions of the conditions
def square_grid_7x7 : set (ℕ × ℕ) := { p | p.1 < 7 ∧ p.2 < 7 }
def point_P : ℕ × ℕ := (3, 4)
def valid_points : set (ℕ × ℕ) := square_grid_7x7 \ {point_P}

-- The statement to prove
theorem line_symmetry_probability : 
  let possible_Qs := valid_points 
  let symmetric_lines_through_P_Q := { q | (q.1 = point_P.1) ∨ (q.2 = point_P.2) }
  let favorable_Qs := possible_Qs ∩ symmetric_lines_through_P_Q
  (favorable_Qs.card.to_real / possible_Qs.card.to_real) = 1 / 4 :=
sorry

end line_symmetry_probability_l796_796609


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796013

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796013


namespace a_greater_than_1_and_b_less_than_1_l796_796702

theorem a_greater_than_1_and_b_less_than_1
  (a b c : ℝ) 
  (h1 : a ≥ b)
  (h2 : b ≥ c)
  (h3 : c > 0)
  (h4 : a * b * c = 1)
  (h5 : a + b + c > 1/a + 1/b + 1/c) : a > 1 ∧ b < 1 :=
by
  sorry

end a_greater_than_1_and_b_less_than_1_l796_796702


namespace smallest_positive_omega_l796_796720

def sine_shift_equivalence (ω : ℝ) : Prop :=
  ∃ (k : ℤ), ∀ (x : ℝ), ω * x + (π / 3) - (π / 3) = 2 * π * k

theorem smallest_positive_omega : sine_shift_equivalence (2 * π) :=
sorry

end smallest_positive_omega_l796_796720


namespace construct_forms_l796_796670

-- Define the type for shapes that can be rotated and flipped
inductive Shape where
  | LShaped : Shape
  | otherShapes : Shape -- other shapes that exist in the problem statement

-- Define a function to represent the shapes being used to form the square and rectangle
def canForm (shapes : List Shape) (dimension1 : Nat) (dimension2 : Nat) : Prop :=
  sorry -- This would represent the actual arrangement logic which is not required to be shown

-- The main proof statement
theorem construct_forms (shapes : List Shape) (cond1 : (shapes : List Shape → Prop)) : 
  (canForm shapes 9 9 ∧ canForm shapes 9 12) :=
by
  -- We assume the provided shapes and their conditions are given
  have h1 : canForm shapes 9 9 := sorry -- Proof that the shapes can form a 9x9 with a 3x3 cutout
  have h2 : canForm shapes 9 12 := sorry -- Proof that the shapes can form a 9x12 rectangle
  exact ⟨h1, h2⟩

end construct_forms_l796_796670


namespace general_solution_of_PDE_l796_796310

theorem general_solution_of_PDE (u : ℝ → ℝ → ℝ) (C_1 C_2 : ℝ → ℝ) :
  (∂ (u x y) / ∂ x ∂ x) - 2 * (∂ (u x y) / ∂ x ∂ y) + (∂ (u x y) / ∂ y ∂ y) - (∂ (u x y) / ∂ x) + (∂ (u x y) / ∂ y) = 0 ↔
  u(x, y) = C_1(y - x) + C_2(y - x) * exp(-y) := 
sorry

end general_solution_of_PDE_l796_796310


namespace octagon_properties_l796_796932

-- Definitions for a regular octagon inscribed in a circle
def regular_octagon (r : ℝ) := ∀ (a b : ℝ), abs (a - b) = r
def side_length := 5
def inscribed_in_circle (r : ℝ) := ∃ (a b : ℝ), a * a + b * b = r * r

-- Main theorem statement
theorem octagon_properties (r : ℝ) (h : r = side_length) (h1 : regular_octagon r) (h2 : inscribed_in_circle r) :
  let arc_length := (5 * π) / 4
  let area_sector := (25 * π) / 8
  arc_length = (5 * π) / 4 ∧ area_sector = (25 * π) / 8 := by
  sorry

end octagon_properties_l796_796932


namespace survey_preference_l796_796473

theorem survey_preference (X Y : ℕ) 
  (ratio_condition : X / Y = 5)
  (total_respondents : X + Y = 180) :
  X = 150 := 
sorry

end survey_preference_l796_796473


namespace quadrilateral_ABCD_area_l796_796421

noncomputable def area_of_ABCD (A B C D E : Point) 
  (h1 : right_angle (A, B, E)) (h2 : right_angle (B, C, E)) (h3 : right_angle (C, D, E))
  (h4 : ∠AEB = 60) (h5 : ∠BEC = 60) (h6 : ∠CED = 45) (h7: AE = 32) : Real :=
  sorry

theorem quadrilateral_ABCD_area :
  ∀ (A B C D E : Point), 
  right_angle (A, B, E) → right_angle (B, C, E) → right_angle (C, D, E) →
  ∠AEB = 60 → ∠BEC = 60 → ∠CED = 45 → AE = 32 →
  area_of_ABCD (A, B, C, D, E) = 2176 * sqrt 3 / 9 :=
by sorry

end quadrilateral_ABCD_area_l796_796421


namespace mean_value_of_quadrilateral_angles_l796_796213

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796213


namespace least_distance_between_ticks_l796_796563

theorem least_distance_between_ticks :
  let fifths := 1 / 5,
      sevenths := 1 / 7,
      lcm := Nat.lcm 5 7,
      z := 1 / lcm
  in z = 1 / 35 :=
  by
  -- Definitions
  let fifths := (1 / 5 : ℝ)
  let sevenths := (1 / 7 : ℝ)
  let lcm := Nat.lcm 5 7
  let z := 1 / (lcm : ℝ)
  
  -- Goal
  have z_goal : z = 1 / 35 := sorry
  exact z_goal

end least_distance_between_ticks_l796_796563


namespace sum_f_sequence_l796_796680

noncomputable def f (x : ℝ) : ℝ := 1 / (4^x + 2)

theorem sum_f_sequence :
  f (1/10) + f (2/10) + f (3/10) + f (4/10) + f (5/10) + f (6/10) + f (7/10) + f (8/10) + f (9/10) = 9 / 4 :=
by {
  sorry
}

end sum_f_sequence_l796_796680


namespace cannot_become_uniform_l796_796818

-- Define the number type as an inductive type
inductive Num
| Zero : Num
| One  : Num

-- Define the list of numbers as a type alias for list of Num
def num_list := list Num

-- Initial configuration: four ones and five zeros
def initial_config : num_list := [Num.One, Num.One, Num.One, Num.One, Num.Zero, Num.Zero, Num.Zero, Num.Zero, Num.Zero]

-- Define the operation on pairs of numbers
def operation (n1 n2 : Num) : Num :=
match n1, n2 with
| Num.Zero, Num.Zero => Num.One
| Num.One, Num.One => Num.One
| _, _ => Num.Zero
end

-- Define the operation on the entire circle configuration
-- This is a helper function that returns the new configuration
def next_state (config : num_list) : num_list :=
(config.zip (config.tail ++ [config.head])).map (λ p => operation p.1 p.2)

-- Define the eventual uniform state condition checker
def is_uniform (config : num_list) : Prop :=
config.all (λ n => n = Num.One) ∨ config.all (λ n => n = Num.Zero)

-- Define the theorem to prove that we cannot reach a uniform state
theorem cannot_become_uniform : ∀ config, (config = initial_config) → ¬ (∃ t, is_uniform (nat.iterate next_state t config)) :=
sorry

end cannot_become_uniform_l796_796818


namespace garage_sale_total_items_l796_796277

-- Definitions for the conditions
def items (n : ℕ) := List ℕ -- List of items sold represented as natural numbers
def different_prices (l : List ℕ) := l.nodup -- All items have different prices
def divisible_by_five (l : List ℕ) := ∀ x ∈ l, x % 5 = 0 -- Prices are divisible by 5
def radio_price_position (l : List ℕ) (p : ℕ) := 
  let sorted_l := l.qsort (· < ·) in
    sorted_l.get? 47 = some p ∧ List.reverse_sorted sorted_l |>.get? 36 = some p -- Radio's price position

-- The main theorem
theorem garage_sale_total_items (l : List ℕ) (p : ℕ) (h_diff : different_prices l) (h_div5 : divisible_by_five l) (h_pos : radio_price_position l p) : 
  l.length = 84 := 
by
  sorry

end garage_sale_total_items_l796_796277


namespace transfer_increases_averages_l796_796906

variables (score_A : ℝ ≥ 0)
variables (score_B : ℝ ≥ 0)

def initial_avg_A := 47.2
def initial_avg_B := 41.8
def initial_size_A := 10
def initial_size_B := 10
def total_score_A := initial_avg_A * initial_size_A
def total_score_B := initial_avg_B * initial_size_B

variables (score_L : ℝ) (score_F : ℝ)

def cond_Lopatin := 41.8 < score_L ∧ score_L < 47.2
def cond_Filin := 41.8 < score_F ∧ score_F < 47.2

def new_avg_A_after_both_transferred := (total_score_A - score_L - score_F) / (initial_size_A - 2)
def new_avg_B_after_both_transferred := (total_score_B + score_L + score_F) / (initial_size_B + 2)

theorem transfer_increases_averages :
  cond_Lopatin →
  cond_Filin →
  new_avg_A_after_both_transferred > initial_avg_A ∧
  new_avg_B_after_both_transferred > initial_avg_B :=
sorry

end transfer_increases_averages_l796_796906


namespace cyclic_quadrilateral_l796_796809

theorem cyclic_quadrilateral 
  (ABC : triangle)
  (AD BE CF : line)
  (I : point)
  (X Y : point)
  (M : point) 
  (D E F : point) 
  (perpendicular_bisector_AD : line)
  (intersects_BE : intersects perpendicular_bisector_AD BE X)
  (intersects_CF : intersects perpendicular_bisector_AD CF Y)
  (midpoint_AD : midpoint AD M) 
  (incenter : incenter ABC I)
  : cyclic_quad A X Y I := 
sorry

end cyclic_quadrilateral_l796_796809


namespace cauchy_davenport_l796_796562

open Finset

theorem cauchy_davenport {p : ℕ} (hp : Nat.Prime p) (A B : Finset (ZMod p))
  (hA : A.Nonempty) (hB : B.Nonempty) : 
  (A + B).card ≥ min (A.card + B.card - 1) p :=
sorry

end cauchy_davenport_l796_796562


namespace at_least_10_percent_non_empty_l796_796773

theorem at_least_10_percent_non_empty (N : ℕ) 
  (h1 : (∑ i in finset.range N, (num_nuts i) / N) = 10) 
  (h2 : (∑ i in finset.range N, (num_nuts i)^2 / N) < 1000) :
  ∃ n, n ≥ N / 10 ∧ ∀ i ∈ finset.range N, num_nuts i ≠ 0 → i ∈ finset.range n := 
sorry

end at_least_10_percent_non_empty_l796_796773


namespace min_f_in_S_max_f_in_S_l796_796448

def S (m n : ℕ) : Prop :=
  m < 4002 ∧ 2 * n ∣ 4002 * m - m^2 + n^2 ∧ n^2 - m^2 + 2 * m * n ≤ 4002 * (n - m)

def f (m n : ℕ) : ℝ :=
  (4002 * m - m^2 - m * n : ℝ) / n

theorem min_f_in_S : ∃ (m n : ℕ), S m n ∧ f m n = 2 :=
sorry

theorem max_f_in_S : ∃ (m n : ℕ), S m n ∧ f m n = 3750 :=
sorry

end min_f_in_S_max_f_in_S_l796_796448


namespace determine_k_l796_796294

noncomputable def k_value (k : ℝ) : Prop :=
  let p1 := (1:ℝ, 2:ℝ)
  let p2 := (3:ℝ, 8:ℝ)
  let p3 := (4:ℝ, k / 3)
  ∃ m b, 
    (p1.2 = m * p1.1 + b) ∧ 
    (p2.2 = m * p2.1 + b) ∧ 
    (p3.2 = m * p3.1 + b)

theorem determine_k : ∀ (k : ℝ), k_value k ↔ k = 33 :=
by
  intro k
  split
  sorry -- Proof goes here
  sorry -- Proof goes here

end determine_k_l796_796294


namespace mean_of_quadrilateral_angles_l796_796193

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796193


namespace average_score_eliminated_l796_796759

theorem average_score_eliminated (n : ℕ) (h : n > 0) :
  (∀ participants, 
    participants = 2 * n ∧
    (∀ avg_score_total adv_avg_score elim_avg_score max_score,
      max_score = 10 ∧
      avg_score_total = 6 ∧
      adv_avg_score = 8 ∧
      adv_avg_score = avg_score_total + 2 →
      elim_avg_score = 4)) :=
begin
  sorry
end

end average_score_eliminated_l796_796759


namespace mean_value_of_quadrilateral_angles_l796_796168

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796168


namespace beth_cans_l796_796949

theorem beth_cans (cans_peas cans_corn : ℕ) (h_peas : cans_peas = 35) (h_corn : cans_corn = 10) :
  cans_peas - 2 * cans_corn = 15 :=
by
  rw [h_peas, h_corn]
  simp
  sorry

end beth_cans_l796_796949


namespace total_vessels_proof_l796_796572

def cruise_ships : Nat := 4
def cargo_ships : Nat := cruise_ships * 2
def sailboats : Nat := cargo_ships + 6
def fishing_boats : Nat := sailboats / 7
def total_vessels : Nat := cruise_ships + cargo_ships + sailboats + fishing_boats

theorem total_vessels_proof : total_vessels = 28 := by
  sorry

end total_vessels_proof_l796_796572


namespace dissimilar_terms_expansion_l796_796954

theorem dissimilar_terms_expansion (a b c d : ℕ) :
  (∑ i : ℕ in finset.range 11, 
    ∑ j : ℕ in finset.range (11 - i),
    ∑ k : ℕ in finset.range (11 - i - j),
    (i + j + k + (10 - i - j - k)) = 10) = 
  (∑ i : ℕ in finset.range 11, 
    ∑ j : ℕ in finset.range (11 - i),
    ∑ k : ℕ in finset.range (11 - i - j),
    (4! / (i! * j! * k! * (10-i-j-k)!))) := 
begin
  sorry
end

end dissimilar_terms_expansion_l796_796954


namespace sum_of_roots_inequality_l796_796323

theorem sum_of_roots_inequality (n : ℕ) (h : 0 < n) :
  (2 / 3) * n * real.sqrt n < (∑ k in finset.range (n + 1), real.sqrt k) ∧
  (∑ k in finset.range (n + 1), real.sqrt k) < ((4 * n + 3) / 6) * real.sqrt n := 
sorry

end sum_of_roots_inequality_l796_796323


namespace find_two_digit_number_l796_796587

theorem find_two_digit_number (N : ℕ) (h1 : 10 ≤ N ∧ N < 100) 
                              (h2 : N % 2 = 0) (h3 : N % 11 = 0) 
                              (h4 : ∃ k : ℕ, (N / 10) * (N % 10) = k^3) :
  N = 88 :=
by {
  sorry
}

end find_two_digit_number_l796_796587


namespace dinosaur_book_cost_l796_796432

-- Define the constants for costs and savings/needs
def dict_cost : ℕ := 11
def cookbook_cost : ℕ := 7
def savings : ℕ := 8
def needed : ℕ := 29
def total_cost : ℕ := savings + needed
def dino_cost : ℕ := 19

-- Mathematical statement to prove
theorem dinosaur_book_cost :
  dict_cost + dino_cost + cookbook_cost = total_cost :=
by
  -- The proof steps would go here
  sorry

end dinosaur_book_cost_l796_796432


namespace mean_value_of_quadrilateral_angles_l796_796210

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796210


namespace x_is_square_l796_796512

noncomputable def x : ℕ → ℚ
| 0       := 1
| 1       := 0
| 2       := 1
| 3       := 1
| (n+3) :=
  if n ≥ 1 then 
    (n^2 + n + 1) * (n + 1) / n * x (n + 2) +
    (n^2 + n + 1) * x (n + 1) -
    (n + 1) / n * x n 
  else 
    0 -- this branch is unreachable, defensive coding

noncomputable def y : ℕ → ℚ
| 0       := 1
| 1       := 0
| (n+2)   := n * y (n + 1) + y n

theorem x_is_square (n : ℕ) : ∃ k, x n = k * k :=
by sorry

end x_is_square_l796_796512


namespace second_diff_const_one_second_diff_const_seven_l796_796826

-- Definitions of second finite differences for natural numbers

def second_finite_difference (f : ℕ → ℝ) (n : ℕ) : ℝ := f(n+2) - 2 * f(n+1) + f(n)

-- Scenario a): The second finite difference is constantly equal to 1
theorem second_diff_const_one (f : ℕ → ℝ) (h : ∀ n, second_finite_difference f n = 1) :
  ∃ a b : ℝ, ∀ n, f(n) = (1/2) * n^2 + a * n + b := sorry

-- Scenario b): The second finite difference is constantly equal to 7
theorem second_diff_const_seven (f : ℕ → ℝ) (h : ∀ n, second_finite_difference f n = 7) :
  ∃ a b : ℝ, ∀ n, f(n) = (7/2) * n^2 + a * n + b := sorry

end second_diff_const_one_second_diff_const_seven_l796_796826


namespace mean_of_quadrilateral_angles_l796_796139

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796139


namespace csc_315_eq_neg_sqrt_two_l796_796994

theorem csc_315_eq_neg_sqrt_two : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_two_l796_796994


namespace sum_of_x_and_y_l796_796757

theorem sum_of_x_and_y 
  (x y : ℤ)
  (h1 : x - y = 36) 
  (h2 : x = 28) : 
  x + y = 20 :=
by 
  sorry

end sum_of_x_and_y_l796_796757


namespace general_terms_l796_796732

noncomputable theory

-- Define the sequences a_n and b_n
def a (n : ℕ) : ℝ := if n = 0 then 1 else (n * (n + 1)) / 2
def b (n : ℕ) : ℝ := ((n + 1) * (n + 1)) / 2

lemma arithmetic_seq (n : ℕ) : 2 * b n = a n + a (n + 1) := by
  sorry

lemma geometric_seq (n : ℕ) : a (n + 1) * a (n + 1) = b n * b (n + 1) := by
  sorry

lemma initial_conditions : a 0 = 1 ∧ b 0 = 2 ∧ a 1 = 3 := by
  sorry

theorem general_terms :
  (∀ n : ℕ, a n = n * (n + 1) / 2) ∧ (∀ n : ℕ, b n = (n + 1) * (n + 1) / 2) := by
  split
  · intro n
    sorry
  · intro n
    sorry

end general_terms_l796_796732


namespace sum_of_cubes_geq_sum_squared_l796_796607

theorem sum_of_cubes_geq_sum_squared {a : ℕ → ℕ} (n : ℕ) 
  (h₀ : a 0 = 0) 
  (h₁ : ∀ k < n, a (k + 1) ≥ a k + 1) :
  ∑ k in Finset.range (n + 1), a (k + 1) ^ 3 ≥ (∑ k in Finset.range (n + 1), a (k + 1)) ^ 2 :=
sorry

end sum_of_cubes_geq_sum_squared_l796_796607


namespace range_of_a_for_two_critical_points_l796_796375

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - Real.exp 1 * x^2 + 18

theorem range_of_a_for_two_critical_points (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) ↔ (a ∈ Set.Ioo (1 / Real.exp 1) 1 ∪ Set.Ioo 1 (Real.exp 1)) :=
sorry

end range_of_a_for_two_critical_points_l796_796375


namespace probability_greater_than_two_l796_796768

noncomputable def probability_of_greater_than_two : ℚ :=
  let total_outcomes := 6
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_greater_than_two (total_outcomes favorable_outcomes : ℕ)
  (ht : total_outcomes = 6) (hf : favorable_outcomes = 4) :
  probability_of_greater_than_two = 2 / 3 :=
by
  rw [probability_of_greater_than_two, ht, hf]
  norm_num
  sorry

end probability_greater_than_two_l796_796768


namespace shaded_region_area_l796_796265

open Real

-- Define the side length of the square
def side_length : ℝ := 8

-- Define the radius of the circles centered at each vertex of the square
def circle_radius : ℝ := 3

-- Define the area of the square
def square_area : ℝ := side_length ^ 2

-- Define the area of one circle
def circle_area : ℝ := π * circle_radius ^ 2

-- Define the area of one 90-degree sector of the circle
def sector_area : ℝ := (π * circle_radius ^ 2) / 4

-- Define the total area covered by the four sectors
def total_sectors_area : ℝ := 4 * sector_area

-- Define the area of the shaded region
def shaded_area : ℝ := square_area - total_sectors_area

theorem shaded_region_area : shaded_area = 64 - 9 * π :=
by
  -- Sorry is used here to skip the proof
  sorry

end shaded_region_area_l796_796265


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796012

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796012


namespace range_of_a_for_monotonicity_l796_796718

noncomputable def f (x a : ℝ) := -x^3 + a * x^2 - x - 2
noncomputable def f' (x a : ℝ) := -3 * x^2 + 2 * a * x - 1

theorem range_of_a_for_monotonicity (a : ℝ) :
  (∀ x : ℝ, f' x a ≤ 0) ↔ -real.sqrt 3 ≤ a ∧ a ≤ real.sqrt 3 :=
by
  sorry

end range_of_a_for_monotonicity_l796_796718


namespace num_pairs_satisfying_eq_l796_796384

theorem num_pairs_satisfying_eq :
  {p : ℤ × ℤ // p.1^2022 + p.2^2 = 4 * p.2}.toFinset.card = 4 :=
  sorry

end num_pairs_satisfying_eq_l796_796384


namespace plant_current_height_l796_796246

theorem plant_current_height (constant_growth_rate : ℕ) (future_height : ℕ) (months_in_year : ℕ)
    (H1 : constant_growth_rate = 5)
    (H2 : future_height = 80)
    (H3 : months_in_year = 12) :
    future_height - constant_growth_rate * months_in_year = 20 :=
by
  rw [H1, H2, H3]
  exact eq.refl 20

end plant_current_height_l796_796246


namespace purple_balls_count_l796_796917

theorem purple_balls_count :
  ∃ P : ℕ, 
    P = 3 ∧ 
    100 = 50 + 20 + 10 + 17 + P ∧ 
    80 / 100 = 0.8 :=
begin
  use 3,
  split,
  { refl, },  -- This proves P = 3
  split,
  { norm_num, },  -- This proves 100 = 50 + 20 + 10 + 17 + 3
  { norm_num, }  -- This proves 80 / 100 = 0.8
end

end purple_balls_count_l796_796917


namespace solve_for_s_l796_796637

def E (a b c : ℝ) : ℝ := a * b^c

theorem solve_for_s (s : ℝ) : E(s, s, 2) = 256 → s = 2^(8 / 3) :=
by
  intro h
  sorry

end solve_for_s_l796_796637


namespace fixed_point_of_line_l796_796472

theorem fixed_point_of_line (m : ℝ) : 
  ∀ (x y : ℝ), (3 * x - 2 * y + 7 = 0) ∧ (4 * x + 5 * y - 6 = 0) → x = -1 ∧ y = 2 :=
sorry

end fixed_point_of_line_l796_796472


namespace problem_solution_l796_796803

theorem problem_solution (u v : ℤ) (h₁ : 0 < v) (h₂ : v < u) (h₃ : u^2 + 3 * u * v = 451) : u + v = 21 :=
sorry

end problem_solution_l796_796803


namespace pyramid_volume_l796_796930

-- Definitions for the conditions
def AB := 10
def BC := 6
def PA_perpendicular_AD := true
def PA_perpendicular_AB := true
def PB := 20

noncomputable def volume_PABCD : ℝ := 200 * real.sqrt 3

-- Lean statement to prove the volume of the pyramid PABCD
theorem pyramid_volume :
  ∃ (PA height_PABCD : ℝ),  
    height_PABCD = real.sqrt (PB^2 - AB^2) ∧ PA = height_PABCD ∧ 
    ∃ (base_area : ℝ), base_area = AB * BC ∧ 
    volume_PABCD = (1/3:ℝ) * base_area * height_PABCD := 
begin
  sorry
end

end pyramid_volume_l796_796930


namespace John_break_time_l796_796784

-- Define the constants
def John_dancing_hours : ℕ := 8

-- Define the condition for James's dancing time 
def James_dancing_time (B : ℕ) : ℕ := 
  let total_time := John_dancing_hours + B
  total_time + total_time / 3

-- State the problem as a theorem
theorem John_break_time (B : ℕ) : John_dancing_hours + James_dancing_time B = 20 → B = 1 := 
  by sorry

end John_break_time_l796_796784


namespace monotonicity_of_f_l796_796374

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x - (1 / x) + 2 * m * (Real.log x)

theorem monotonicity_of_f (m : ℝ) : 
  (∀ x > 0, f x m is_increasing) ↔ (m >= -1 ∧ ∀ x > 0, f x m is_decreasing_on (Ioi x_2)) ∨ 
  (m < -1 ∧ ∀ x > 0, f x m is_decreasing_on (Ioc x_1 x_2)) ∨
  (m < -1 ∧ ∀ x > 0, f x m is_increasing_on (Ioo x_1 0 ∪ Ioi x_2)) :=
sorry

end monotonicity_of_f_l796_796374


namespace second_discount_percentage_l796_796245

theorem second_discount_percentage 
    (original_price : ℝ) (final_price : ℝ) (first_discount : ℝ) (third_discount : ℝ) (second_discount : ℝ) :
      original_price = 9795.3216374269 →
      final_price = 6700 →
      first_discount = 0.20 →
      third_discount = 0.05 →
      (original_price * (1 - first_discount) * (1 - second_discount / 100) * (1 - third_discount) = final_price) →
      second_discount = 10 :=
by
  intros h_orig h_final h_first h_third h_eq
  sorry

end second_discount_percentage_l796_796245


namespace count_valid_arrays_l796_796328

-- Define the integer array condition
def valid_array (x1 x2 x3 x4 : ℕ) : Prop :=
  0 < x1 ∧ x1 ≤ x2 ∧ x2 < x3 ∧ x3 ≤ x4 ∧ x4 < 7

-- State the theorem that proves the number of valid arrays is 70
theorem count_valid_arrays : ∃ (n : ℕ), n = 70 ∧ 
    ∀ (x1 x2 x3 x4 : ℕ), valid_array x1 x2 x3 x4 -> ∃ (n : ℕ), n = 70 :=
by
  -- The proof can be filled in later
  sorry

end count_valid_arrays_l796_796328


namespace trigonometric_quadrant_l796_796743

theorem trigonometric_quadrant (α : ℝ) (h1 : sin α * tan α < 0) (h2 : cos α / tan α < 0) : 
  (π < α ∧ α < 3 * π / 2) :=
by
  sorry

end trigonometric_quadrant_l796_796743


namespace triangle_ratios_l796_796780

theorem triangle_ratios 
  (A B C F G E : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited F] [inhabited G] [inhabited E]
  (r1 r2 : ℝ)
  (hF : r1 ≠ 0 ∧ r2 ≠ 0)
  (h1 : F = (r1 * A + r2 * C) / (r1 + r2))  -- F divides AC in the ratio 1:2
  (h2 : G = (B + F) / 2)                    -- G is the midpoint of BF
  (h3 : E = intersection (line_through A G) (line_through B C)) -- E is the intersection of AG with BC
  : divides E B C 1 3 := 
sorry

end triangle_ratios_l796_796780


namespace angle_4_measure_l796_796690

theorem angle_4_measure (a1 a2 a3 a4 : ℝ)
  (h1 : a1 = 34)
  (h2 : a2 = 53)
  (h3 : a3 = 27)
  (h_sum : a4 = 180 - (a1 + a2 + a3)) :
  a4 = 114 :=
by
  rw [h1, h2, h3] at h_sum
  linarith

end angle_4_measure_l796_796690


namespace symmetric_circle_eq_l796_796852

-- Define the initial circle equation C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line equation l
def line_l (x y : ℝ) : Prop := x - y = 2

-- Define the new circle equation we are to prove
def symmetric_circle (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 1

-- Theorem to prove that the equation of the circle symmetric to C with respect to l is as stated
theorem symmetric_circle_eq :
  (∀ x y : ℝ, circle_C x y ↔ ((x-1)^2 + (y-1)^2 = 1)) →
  (∃ a b : ℝ, ((a, b) = (3, -1) ∧
    symmetric_point (1, 1) (a, b) line_l)) →
  (∀ x y : ℝ, symmetric_circle x y) :=
by
  intros h₁ h₂,
  -- Proof steps go here
  sorry

end symmetric_circle_eq_l796_796852


namespace ellipse_equation_fixed_point_intersection_l796_796691

-- Definitions related to the problem conditions
variables (a b : ℝ) (ecc : ℝ) (F1 F2 P : ℝ × ℝ) (k : ℝ) (x1 x2 y1 y2 : ℝ)

axiom a_gt_b : a > b
axiom b_gt_0 : b > 0
axiom ecc_eq : ecc = real.sqrt 3 / 2
axiom e_def : ecc = real.sqrt (a^2 - b^2) / a

noncomputable def foci := (F1, F2)
noncomputable def vertex := P

axiom F1_coords : F1 = (-real.sqrt 3 * b, 0)
axiom F2_coords : F2 = (real.sqrt 3 * b, 0)
axiom P_coords : P = (2 * b, 0)

axiom dot_prod_eq_1 : (-(real.sqrt 3)*b - 2*b) * (real.sqrt 3 * b - 2*b) = 1

-- Conditions for the second part of the problem
variables (A M B : ℝ × ℝ)

axiom k_line_intersects_ellipse : A = (x1, y1) ∧ B = (x2, y2) ∧ 
                                      (A.1 = k * A.2 - 1 ∧ B.1 = k * B.2 - 1)

axiom A_symmetric_to_M : M = (x1, -y1) ∧ M ≠ B

-- Statements to prove
theorem ellipse_equation : a = 2 * b ∧ b = 1 → (∀ x y, (x^2) / 4 + y^2 = 1 ↔ x^2 + 4 * y^2 = 4) :=
sorry

theorem fixed_point_intersection : (x2*y1 + x1*y2) / (y1 + y2) = -4 :=
sorry

end ellipse_equation_fixed_point_intersection_l796_796691


namespace polynomial_multiplication_l796_796466

theorem polynomial_multiplication :
  ∀ (x : ℝ),
    let P := 2 * x ^ 3 - 4 * x ^ 2 + 3 * x,
        Q := 2 * x ^ 4 + 3 * x ^ 3 in
    (P * Q) = (4 * x ^ 7 - 2 * x ^ 6 - 6 * x ^ 5 + 9 * x ^ 4) := 
sorry

end polynomial_multiplication_l796_796466


namespace at_least_15_same_class_l796_796520

variable (Club : Type) [Fintype Club] (Student : Club → Prop)

-- Conditions
variable (h_students : Fintype.card Club = 60)
variable (h_at_least_three : ∀ (s : Finset Club), s.card = 10 → ∃ (c : Club), 3 ≤ (s.filter Student).card)

theorem at_least_15_same_class :
  ∃ (c : Club), 15 ≤ (Finset.univ.filter Student).filter (λ x, x = c).card :=
by
  sorry

end at_least_15_same_class_l796_796520


namespace largest_sum_prime_factors_l796_796462

theorem largest_sum_prime_factors (a b c d : ℕ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : a ≠ d) (h₄ : b ≠ c) (h₅ : b ≠ d) (h₆ : c ≠ d)
  (h₇ : 1 ≤ a ∧ a ≤ 9) (h₈ : 1 ≤ b ∧ b ≤ 9) (h₉ : 1 ≤ c ∧ c ≤ 9) (h₁₀ : 1 ≤ d ∧ d ≤ 9) :
  let N := 6666 * (a + b + c + d) in
  prime_factors (N) → 
  (2 + 3 + 11 + 101 + 29 = 146) :=
begin
  let N := 6666 * (a + b + c + d),
  sorry
end

end largest_sum_prime_factors_l796_796462


namespace count_zero_sequences_l796_796791

def ordered_triple := {x : ℤ // 1 ≤ x ∧ x ≤ 15}

def is_zero_sequence (s : ℕ → ℤ) (triple : ordered_triple × ordered_triple × ordered_triple) : Prop :=
  let ⟨b1, hb1⟩ := triple.fst,
      ⟨b2, hb2⟩ := triple.snd.fst,
      ⟨b3, hb3⟩ := triple.snd.snd in
  ∃ n : ℕ, n ≥ 4 ∧ s n = 0

theorem count_zero_sequences : ∃ s : ordered_triple → ordered_triple → ordered_triple → (ℕ → ℤ), 
  (∀ t : ordered_triple × ordered_triple × ordered_triple, is_zero_sequence (s t.fst t.snd.fst t.snd.snd) t) →
  (finset.card {t : ordered_triple × ordered_triple × ordered_triple | is_zero_sequence (s t.fst t.snd.fst t.snd.snd) t} = 855) :=
sorry

end count_zero_sequences_l796_796791


namespace minNumberOfGloves_l796_796498

-- Define the number of participants
def numParticipants : ℕ := 43

-- Define the number of gloves needed per participant
def glovesPerParticipant : ℕ := 2

-- Define the total number of gloves
def totalGloves (participants glovesPerParticipant : ℕ) : ℕ := 
  participants * glovesPerParticipant

-- Theorem proving the minimum number of gloves required
theorem minNumberOfGloves : totalGloves numParticipants glovesPerParticipant = 86 :=
by
  sorry

end minNumberOfGloves_l796_796498


namespace no_solution_eq_eight_diff_l796_796428

theorem no_solution_eq_eight_diff (k : ℕ) (h1 : k > 0) (h2 : k ≤ 99) 
  (h3 : ∀ x y : ℕ, x^2 - k * y^2 ≠ 8) : 
  (99 - 3 = 96) := 
by 
  sorry

end no_solution_eq_eight_diff_l796_796428


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796011

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796011


namespace one_of_15_coprime_numbers_is_prime_l796_796684

theorem one_of_15_coprime_numbers_is_prime  
(n : ℕ) (a : Fin n → ℕ) 
(h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
(h_coprime : ∀ i j, i ≠ j → Nat.coprime (a i) (a j)) 
(h_interval : ∀ i, 2 ≤ a i ∧ a i ≤ 2022) 
(h_n : n = 15) : 
  ∃ i, Nat.Prime (a i) := 
sorry

end one_of_15_coprime_numbers_is_prime_l796_796684


namespace triple_integral_value_l796_796959

open Real

noncomputable def f (x y z : ℝ) : ℝ := (x^2) / (x^2 + y^2)

def region (x y z : ℝ) : Prop :=
  z ≤ sqrt(36 - x^2 - y^2) ∧ z ≥ sqrt((x^2 + y^2) / 3)

theorem triple_integral_value :
  ∫∫∫ (λ (x y z : ℝ), f x y z) in region = 36 * π :=
sorry

end triple_integral_value_l796_796959


namespace proof_problem_A_proof_problem_C_l796_796893

theorem proof_problem_A (x : ℝ) (h : x > 1) : 
  ∃ m, m = 3 ∧ (∀ x, x > 1 → x + 1 / (x - 1) ≥ m) :=
sorry

theorem proof_problem_C (x : ℝ) (h : 0 < x) (h' : x < 10) : 
  ∃ M, M = 5 ∧ (∀ x, 0 < x → x < 10 → sqrt (x * (10 - x)) ≤ M) :=
sorry

end proof_problem_A_proof_problem_C_l796_796893


namespace fuchsia_purply_belief_l796_796564

theorem fuchsia_purply_belief :
  ∀ (total kind_pink both neither : ℕ),
  total = 100 →
  kind_pink = 60 →
  both = 27 →
  neither = 17 →
  (total - neither - (kind_pink - both)) = 50 :=
by
  intros total kind_pink both neither h_total h_kind_pink h_both h_neither
  calc
    total - neither - (kind_pink - both) = 100 - 17 - (60 - 27) : by
      rw [h_total, h_kind_pink, h_both, h_neither]
    ... = 50 : by
      norm_num

end fuchsia_purply_belief_l796_796564


namespace total_vessels_l796_796571

theorem total_vessels (C G S F : ℕ) (h1 : C = 4) (h2 : G = 2 * C) (h3 : S = G + 6) (h4 : S = 7 * F) : 
  C + G + S + F = 28 :=
by
  sorry

end total_vessels_l796_796571


namespace finite_couples_l796_796325

theorem finite_couples (n k : ℕ) (h_n : n > 0) (h_k : k > 1) :
    ∃ N : ℕ, ∀ a b : ℕ, a > 0 → b > 0 → F_{n, k} a b = 0 → a < N ∧ b < N :=
sorry

def F_{n,k} (n k x y : ℕ) : ℕ := x.factorial + n^k + n + 1 - y^k

end finite_couples_l796_796325


namespace complex_value_l796_796640

noncomputable def det (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_value (z : ℂ) (h : det z 1 z (2 * complex.I) = 3 + 2 * complex.I) :
  z = (1 / 5 : ℂ) - (8 / 5 : ℂ) * complex.I :=
by
  sorry

end complex_value_l796_796640


namespace rationalization_sum_l796_796483

noncomputable def numerator : ℚ√ℚ := 
  -13*real.sqrt 3 - 9*real.sqrt 5 + 3*real.sqrt 11 - 6*real.sqrt 15 + 6*real.sqrt 55

noncomputable def denominator : ℚ :=
  51

noncomputable def final_sum : ℤ :=
  -13 + 3 + (-9 + 5) + 3 + 11 + (-6 + 15) + 6 + 55 + 51

theorem rationalization_sum : final_sum = 116 := by
  exact (calc
    -13 + 3 + (-9 + 5) + 3 + 11 + (-6 + 15) + 6 + 55 + 51 = 116 : by norm_num)

end rationalization_sum_l796_796483


namespace percentage_discount_is_10_percent_l796_796600

def wholesale_price : ℝ := 90
def profit_percentage : ℝ := 0.20
def retail_price : ℝ := 120

theorem percentage_discount_is_10_percent :
  let profit := profit_percentage * wholesale_price in
  let selling_price := wholesale_price + profit in
  let discount_amount := retail_price - selling_price in
  let percentage_discount := (discount_amount / retail_price) * 100 in
  percentage_discount = 10 :=
by
  sorry

end percentage_discount_is_10_percent_l796_796600


namespace coefficient_x3_l796_796533

def f (x : ℝ) : ℝ := (3 * x + 1 + Real.sqrt 3) ^ 8

theorem coefficient_x3 (x : ℝ) : 
  let term := (Finset.sum (Finset.range 9) (λ k : ℕ, (Nat.choose 8 k) * (3 * x) ^ k * (1 + Real.sqrt 3) ^ (8 - k))) in
    (term.coeff 3) = 114912 :=
by sorry

end coefficient_x3_l796_796533


namespace trig_expression_equals_neg_one_l796_796317

theorem trig_expression_equals_neg_one :
  tan 70 * cos 10 * (sqrt 3 * tan 20 - 1) = -1 :=
by sorry

end trig_expression_equals_neg_one_l796_796317


namespace hot_dogs_served_today_l796_796262

-- Define the number of hot dogs served during lunch
def h_dogs_lunch : ℕ := 9

-- Define the number of hot dogs served during dinner
def h_dogs_dinner : ℕ := 2

-- Define the total number of hot dogs served today
def total_h_dogs : ℕ := h_dogs_lunch + h_dogs_dinner

-- Theorem stating that the total number of hot dogs served today is 11
theorem hot_dogs_served_today : total_h_dogs = 11 := by
  sorry

end hot_dogs_served_today_l796_796262


namespace measure_of_angle_A_sum_of_sides_b_c_l796_796770

open Real

-- Define the acute triangle and the given conditions
structure AcuteTriangle :=
(a b c : ℝ) -- sides
(A B C : ℝ) -- angles
(acuteness : A < 90 ∧ B < 90 ∧ C < 90)
(opposite_sides : a = c * sin A ∧ b = a * sin B ∧ c = b * sin C)
(given_eq : sqrt 3 * sin C - cos B = cos (A - C))
(area_given : 3 * sqrt 3 = 1 / 2 * b * c * sin A)

-- Define the sides and area given
def acute_triangle_specific : AcuteTriangle :=
{ a := 2 * sqrt 3,
  b := sorry,
  c := sorry,
  A := 60,
  B := sorry,
  C := sorry,
  acuteness := ⟨sorry, sorry, sorry⟩,
  opposite_sides := ⟨sorry, sorry, sorry⟩,
  given_eq := sorry,
  area_given := sorry
}

-- Prove the measure of angle A
theorem measure_of_angle_A (T : AcuteTriangle) : T.A = 60 :=
sorry

-- Prove the sum of sides b and c
theorem sum_of_sides_b_c (T : AcuteTriangle) (a_eq : T.a = 2 * sqrt 3) (area_eq : 3 * sqrt 3 = 1 / 2 * T.b * T.c * sin T.A) : T.b + T.c = 4 * sqrt 3 :=
sorry

end measure_of_angle_A_sum_of_sides_b_c_l796_796770


namespace mean_of_quadrilateral_angles_l796_796189

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796189


namespace csc_315_eq_neg_sqrt_two_l796_796987

theorem csc_315_eq_neg_sqrt_two : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_two_l796_796987


namespace sum_possible_values_of_abs_diff_l796_796802

theorem sum_possible_values_of_abs_diff 
  (p q r s : ℝ) 
  (h₀ : |p - q| = 3) 
  (h₁ : |q - r| = 4) 
  (h₂ : |r - s| = 5) : 
  (finset.univ.bUnion (λ q, finset.univ.bUnion (λ r, finset.univ.image (λ s, |p - s|)))).sum = 24 :=
sorry

end sum_possible_values_of_abs_diff_l796_796802


namespace probability_is_one_fourth_l796_796257

-- Define the set of numbers
def number_set : set ℝ := { -10, -6, -5, -4, -2.5, -1, 0, 2.5, 4, 6, 7, 10 }

-- Define the equation
def equation (x : ℝ) : Prop := (x + 5) * (x + 10) * (2*x - 5) = 0

-- Define the roots of the equation
def roots : set ℝ := { -5, -10, 2.5 }

-- Define the probability function
def probability_of_solution_in_set (sols set : set ℝ) : ℝ :=
  (sols ∩ set).size / set.size

-- Define a theorem stating that the probability is 1/4
theorem probability_is_one_fourth :
  probability_of_solution_in_set roots number_set = 1/4 :=
by
  sorry

end probability_is_one_fourth_l796_796257


namespace escalator_steps_l796_796272

theorem escalator_steps (T : ℝ) (E : ℝ) (N : ℝ) (h1 : N - 11 = 2 * (N - 29)) : N = 47 :=
by
  sorry

end escalator_steps_l796_796272


namespace a3_value_l796_796346

def sequence (a : ℕ → ℚ) := 
  (a 1 = 1) ∧ (∀ n, n > 1 → a n = 1 / (a (n-1) + 1))

theorem a3_value (a : ℕ → ℚ) (h : sequence a) : a 3 = 2 / 3 :=
by {
  obtain ⟨h₁, h₂⟩ := h,
  have a2 : a 2 = 1 / 2 := by rw [h₂ 2]; try linarith; assumption,
  rw [h₂ 3]; try linarith,
  rw [a2],
  norm_num,
  simp,
}


end a3_value_l796_796346


namespace vector_magnitude_l796_796697

open Real

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Define the conditions that a and b are unit vectors
def is_unit_vector (v : EuclideanSpace ℝ (Fin 3)) : Prop := ‖v‖ = 1

-- Define the condition that the angle between a and b is 120 degrees
def angle_120_degrees (a b : EuclideanSpace ℝ (Fin 3)) : Prop :=
  a • b = - (1 / 2)

theorem vector_magnitude
  (ha : is_unit_vector a)
  (hb : is_unit_vector b)
  (hab : angle_120_degrees a b) :
  ‖a - 2 • b‖ = sqrt 7 := by
  sorry

end vector_magnitude_l796_796697


namespace count_zeros_g_l796_796289

def g (x : ℝ) : ℝ := sin (log x) + 1 / 2

theorem count_zeros_g :
  let interval := (0 : ℝ, (exp (Real.pi) : ℝ)) in
  let zeros := {x | 0 < x ∧ x < exp (Real.pi) ∧ g x = 0} in
  zeros.finite ∧ zeros.card = 2 :=
by
  let interval := (0 : ℝ, (exp (Real.pi) : ℝ))
  let zeros := {x | 0 < x ∧ x < exp (Real.pi) ∧ g x = 0}
  have fin_zeros : zeros.finite := sorry
  have card_zeros : zeros.card = 2 := sorry
  exact ⟨fin_zeros, card_zeros⟩

end count_zeros_g_l796_796289


namespace compare_values_l796_796376

noncomputable def f (x : ℝ) : ℝ := Real.exp (-abs x)

def a : ℝ := f (Real.log (1 / 3))
def b : ℝ := f (Real.log (1 / Real.exp 1) / Real.log 3)
def c : ℝ := f (Real.log 9 / Real.log (1 / Real.exp 1))

theorem compare_values : b > a ∧ a > c := 
by sorry

end compare_values_l796_796376


namespace find_vector_b_l796_796793

variable (x y z : ℝ)

def a : ℝ × ℝ × ℝ := (3, 2, 4)

def b : ℝ × ℝ × ℝ := (x, y, z)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

theorem find_vector_b (h1 : dot_product a b = 20)
                      (h2 : cross_product a b = (-10, -5, 1)) :
  b = (4, 3, 1) :=
  sorry

end find_vector_b_l796_796793


namespace find_natural_number_n_l796_796305

def sum_of_squares_of_proper_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d, d < n) (Finset.divisors n)).sum (λ d, d * d)

theorem find_natural_number_n : ∃ n : ℕ, sum_of_squares_of_proper_divisors n = 5 * (n + 1) ∧ n = 16 :=
by
  sorry

end find_natural_number_n_l796_796305


namespace complex_operation_l796_796461

namespace ComplexNumberProof

-- Define the given condition
def z : ℂ := 1 + complex.i

-- The theorem statement
theorem complex_operation : z^2 + 2 / z = 1 + complex.i := 
by
  -- Complete the proof here
  sorry

end ComplexNumberProof

end complex_operation_l796_796461


namespace mean_value_of_quadrilateral_angles_l796_796084

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796084


namespace unique_monic_poly_of_degree_5_real_coeffs_l796_796288

noncomputable def p : Polynomial ℝ := Polynomial.monicOfDegree (Polynomial.degree (Polynomial.C 1 - Polynomial.X) = 5) sorry

theorem unique_monic_poly_of_degree_5_real_coeffs :
  ∀ (p : Polynomial ℝ), 
  Polynomial.degree p = 5 ∧ Polynomial.monic p ∧ 
  (∀ (a : ℂ), Polynomial.isRoot p a → Polynomial.isRoot p (1 / a) ∧ Polynomial.isRoot p (1 - a)) → 
  p = (Polynomial.C 1 + Polynomial.X) * 
      (Polynomial.C 2 - Polynomial.X) *
      (Polynomial.C (1 / 2) - Polynomial.X) *
      (Polynomial.X^2 - Polynomial.C 1 * Polynomial.X + Polynomial.C 1) :=
by sorry

end unique_monic_poly_of_degree_5_real_coeffs_l796_796288


namespace bounded_region_area_l796_796308

theorem bounded_region_area :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 9}
  let lines := {p : ℝ × ℝ | p.2 = 2 * | p.1 |}
  set bounded_region := intersection circle lines
  let region_area := 9 * π / 2
in
  area bounded_region = region_area :=
sorry

end bounded_region_area_l796_796308


namespace eduardo_needs_l796_796588

variable (flour_per_24_cookies sugar_per_24_cookies : ℝ)
variable (num_cookies : ℝ)

axiom h_flour : flour_per_24_cookies = 1.5
axiom h_sugar : sugar_per_24_cookies = 0.5
axiom h_cookies : num_cookies = 120

theorem eduardo_needs (scaling_factor : ℝ) 
    (flour_needed : ℝ)
    (sugar_needed : ℝ)
    (h_scaling : scaling_factor = num_cookies / 24)
    (h_flour_needed : flour_needed = flour_per_24_cookies * scaling_factor)
    (h_sugar_needed : sugar_needed = sugar_per_24_cookies * scaling_factor) :
  flour_needed = 7.5 ∧ sugar_needed = 2.5 :=
sorry

end eduardo_needs_l796_796588


namespace geese_more_than_ducks_l796_796623

-- Define initial conditions
def initial_ducks : ℕ := 25
def initial_geese : ℕ := 2 * initial_ducks - 10
def additional_ducks : ℕ := 4
def geese_leaving : ℕ := 15 - 5

-- Calculate the number of remaining ducks
def remaining_ducks : ℕ := initial_ducks + additional_ducks

-- Calculate the number of remaining geese
def remaining_geese : ℕ := initial_geese - geese_leaving

-- Prove the final statement
theorem geese_more_than_ducks : (remaining_geese - remaining_ducks) = 1 := by
  sorry

end geese_more_than_ducks_l796_796623


namespace other_bullet_train_speed_is_correct_l796_796912

noncomputable def speed_of_other_bullet_train 
    (length_first_train : ℝ) 
    (speed_first_train_kmph : ℝ)
    (length_second_train : ℝ) 
    (crossing_time : ℝ) 
    (relative_speed_mps : ℝ) : ℝ :=
    speed_first_train_kmph * (1000 / 3600) + length_second_train / crossing_time  * 3600 / 1000 + sorry

theorem other_bullet_train_speed_is_correct
    (length_first_train : ℝ := 270)
    (speed_first_train_kmph : ℝ := 120)
    (length_second_train : ℝ := 230.04)
    (crossing_time : ℝ := 9) : 
    speed_of_other_bullet_train length_first_train speed_first_train_kmph length_second_train crossing_time = 79.98 :=
sorry

end other_bullet_train_speed_is_correct_l796_796912


namespace gum_ratio_correct_l796_796957

variable (y : ℝ)
variable (cherry_pieces : ℝ := 30)
variable (grape_pieces : ℝ := 40)
variable (pieces_per_pack : ℝ := y)

theorem gum_ratio_correct:
  ((cherry_pieces - 2 * pieces_per_pack) / grape_pieces = cherry_pieces / (grape_pieces + 4 * pieces_per_pack)) ↔ y = 5 :=
by
  sorry

end gum_ratio_correct_l796_796957


namespace yadav_spends_50_percent_on_clothes_and_transport_l796_796816

variable (S : ℝ)
variable (monthly_savings : ℝ := 46800 / 12)
variable (clothes_transport_expense : ℝ := 3900)
variable (remaining_salary : ℝ := 0.40 * S)

theorem yadav_spends_50_percent_on_clothes_and_transport (h1 : remaining_salary = 2 * 3900) :
  (clothes_transport_expense / remaining_salary) * 100 = 50 :=
by
  -- skipping the proof steps
  sorry

end yadav_spends_50_percent_on_clothes_and_transport_l796_796816


namespace tan_of_cos_l796_796742

theorem tan_of_cos (α : ℝ) (h_cos : Real.cos α = -4 / 5) (h_alpha : 0 < α ∧ α < Real.pi) : 
  Real.tan α = -3 / 4 :=
sorry

end tan_of_cos_l796_796742


namespace area_of_rectangle_l796_796506

theorem area_of_rectangle (l w : ℝ) (h_perimeter : 2 * (l + w) = 126) (h_difference : l - w = 37) : l * w = 650 :=
sorry

end area_of_rectangle_l796_796506


namespace max_cookies_Andy_can_eat_l796_796274

theorem max_cookies_Andy_can_eat 
  (x y : ℕ) 
  (h1 : x + y = 36)
  (h2 : y ≥ 2 * x) : 
  x ≤ 12 := by
  sorry

end max_cookies_Andy_can_eat_l796_796274


namespace root_polynomial_p_q_l796_796800

theorem root_polynomial_p_q (p q : ℝ) (h_root : is_root (λ x : ℂ, x^3 + (p : ℂ) * x + (q : ℂ)) (2 + 1 * Complex.I)) :
  p + q = 9 := 
by
  sorry

end root_polynomial_p_q_l796_796800


namespace line_perp_to_plane_imp_perp_to_line_l796_796725

def Line := Type
def Plane := Type

variables (m n : Line) (α : Plane)

def is_parallel (l : Line) (p : Plane) : Prop := sorry
def is_perpendicular (l1 l2 : Line) : Prop := sorry
def is_contained (l : Line) (p : Plane) : Prop := sorry

theorem line_perp_to_plane_imp_perp_to_line :
  (is_perpendicular m α) ∧ (is_contained n α) → (is_perpendicular m n) :=
sorry

end line_perp_to_plane_imp_perp_to_line_l796_796725


namespace length_of_XZ_is_correct_l796_796425

noncomputable def length_of_side_XZ : Real :=
  let XT := 15
  let YS := 20
  let XG := (2/3) * XT
  let YG := (2/3) * YS
  Real.sqrt (XG^2 + YG^2)

theorem length_of_XZ_is_correct :
  length_of_side_XZ = 50 / 3 :=
by
  let XT := 15
  let YS := 20
  have XG := (2/3) * XT
  have YG := (2/3) * YS
  have sqr_XG := XG^2
  have sqr_YG := YG^2
  have XZ_sqr := sqr_XG + sqr_YG
  have XZ := Real.sqrt XZ_sqr
  have correct_XZ := 50 / 3
  show _ = _
  calc
  length_of_side_XZ
    = Real.sqrt (XG^2 + YG^2) : rfl
    ... = Real.sqrt (10^2 + (40/3)^2) : by sorry -- Calculation to be refined
    ... = 50 / 3 : by sorry -- Final proof step

end length_of_XZ_is_correct_l796_796425


namespace beatrice_has_highest_value_l796_796270

def initial_value := 15

def albert_final_value (x : ℕ) : ℕ :=
let step1 := x * 3 in
let step2 := step1 + 5 in
let step3 := step2 - 3 in
step3 * 2

def beatrice_final_value (x : ℕ) : ℕ :=
let step1 := x ^ 2 in
let step2 := step1 + 3 in
let step3 := step2 - 7 in
step3 * 2

def carlos_final_value (x : ℕ) : ℚ :=
let step1 := x * 5 in
let step2 := step1 - 4 in
let step3 := step2 + 6 in
step3 / 2

theorem beatrice_has_highest_value :
  beatrice_final_value initial_value > albert_final_value initial_value ∧
  beatrice_final_value initial_value > carlos_final_value initial_value :=
by
  sorry

end beatrice_has_highest_value_l796_796270


namespace staff_meeting_doughnuts_l796_796975

theorem staff_meeting_doughnuts (n_d n_s n_l : ℕ) (h₁ : n_d = 50) (h₂ : n_s = 19) (h₃ : n_l = 12) :
  (n_d - n_l) / n_s = 2 :=
by
  sorry

end staff_meeting_doughnuts_l796_796975


namespace solution_sets_correct_l796_796299

noncomputable def problem : ℕ :=
  let a₁ := 3
  let b₁ := 4
  let c₁ := 6
  let height := 45
  let total_crates := 11
  let possible_combinations := 
    [(7, 0, 4), (4, 3, 3), (1, 6, 2), (2, 9, 1), (11, 0, 0)]
  let multinomial (a b c : ℕ) := Nat.factorial 11 / (Nat.factorial a * Nat.factorial b * Nat.factorial c)
  let possibilities := List.map (λ (triple : ℕ × ℕ × ℕ), multinomial triple.1 triple.2 triple.3) possible_combinations
  let total_permutations := possibilities.foldr (· + ·) 0
  total_permutations

theorem solution_sets_correct : ∃ m : ℕ, m = problem :=
begin
  use problem,
  sorry
end

end solution_sets_correct_l796_796299


namespace mean_of_quadrilateral_angles_is_90_l796_796105

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796105


namespace hit_coordinate_axes_probability_l796_796928

noncomputable def m : ℕ := 245
noncomputable def n : ℕ := 7

/-- Define the probability function -/
noncomputable def P : ℕ × ℕ → ℚ
| (0, 0) := 1
| (x, 0) := 0
| (0, y) := 0
| (x, y) := if x > 0 ∧ y > 0 then (P (x - 1, y) + P (x, y - 1) + P (x - 1, y - 1)) / 3 
           else 0

theorem hit_coordinate_axes_probability : P (4, 4) = 245 / 2187 ∧ 245 % 3 ≠ 0 ∧ m + n = 252 := 
by {
  -- The actual proof goes here.
  sorry
}

end hit_coordinate_axes_probability_l796_796928


namespace garden_fencing_l796_796590

/-- A rectangular garden has a length of 50 yards and the width is half the length.
    Prove that the total amount of fencing needed to enclose the garden is 150 yards. -/
theorem garden_fencing : 
  ∀ (length width : ℝ), 
  length = 50 ∧ width = length / 2 → 
  2 * (length + width) = 150 :=
by
  intros length width
  rintro ⟨h1, h2⟩
  sorry

end garden_fencing_l796_796590


namespace general_formula_sum_first_20_b_l796_796694

noncomputable theory

-- Defining the geometric sequence a_n
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * 2)

-- Defining the arithmetic condition
def arithmetic_condition (a : ℕ → ℝ) : Prop :=
  let a1 := a 1 in
  let a2 := a 2 in
  let a3 := a 3 in
  2 * a2 = a1 + (a3 - 1)

-- Definition of b_n sequence
def b_sequence (a b : ℕ → ℝ) : Prop :=
  (∀ n, if n % 2 = 1 then b n = a n - 1 else b n = a n / 2)

-- The main theorem to prove the general formula for a_n
theorem general_formula (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 = 2) (h3 : arithmetic_condition a) :
  ∀ n, a n = 2 ^ (n - 1) :=
sorry

-- The main theorem to calculate the sum of the first 20 terms of b_n
theorem sum_first_20_b (a b : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 = 2) (h3 : arithmetic_condition a) (h4 : b_sequence a b) :
  (∑ i in finset.range 20, b (i + 1)) = (2 ^ 21 - 32) / 3 :=
sorry

end general_formula_sum_first_20_b_l796_796694


namespace sin_alpha_expression_l796_796678

noncomputable def alpha : ℝ := sorry

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (Real.sin (alpha - Real.pi / 3), Real.cos alpha + Real.pi / 3)
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- The theorem to be proved
theorem sin_alpha_expression :
  parallel a b →
  (Real.sin alpha)^2 + 2 * (Real.sin alpha) * (Real.cos alpha) = 3 / 2 :=
by
  sorry

end sin_alpha_expression_l796_796678


namespace second_number_is_90_l796_796514

theorem second_number_is_90 (a b c : ℕ) 
  (h1 : a + b + c = 330) 
  (h2 : a = 2 * b) 
  (h3 : c = (1 / 3) * a) : 
  b = 90 := 
by
  sorry

end second_number_is_90_l796_796514


namespace math_problem_l796_796790

noncomputable def problem_statement : Prop :=
  let A := 60 * (Real.pi / 180) in
  ∃ (B C : ℝ) (x y z w : ℕ),
    (cos B)^2 + (cos C)^2 + 2 * sin B * sin C * cos A = 13 / 7 ∧
    (cos C)^2 + (cos A)^2 + 2 * sin C * sin A * cos B = 12 / 5 ∧
    (cos A)^2 + (cos B)^2 + 2 * sin A * sin B * cos C = (x - y * (z^.sqrt)) / w ∧
    Nat.gcd (x + y) w = 1 ∧
    ¬ ∃ p : ℕ, prime p ∧ p^2 ∣ z ∧
    x + y + z + w = 14

theorem math_problem : problem_statement :=
  by sorry

end math_problem_l796_796790


namespace mean_value_of_quadrilateral_angles_l796_796051

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796051


namespace arithmetic_sequence_sum_l796_796627

theorem arithmetic_sequence_sum :
  ∃ (a l d n : ℕ), a = 71 ∧ l = 109 ∧ d = 2 ∧ n = ((l - a) / d) + 1 ∧ 
    (3 * (n * (a + l) / 2) = 5400) := sorry

end arithmetic_sequence_sum_l796_796627


namespace jiaqi_grade_is_95_3_l796_796827

def extracurricular_score : ℝ := 96
def mid_term_score : ℝ := 92
def final_exam_score : ℝ := 97

def extracurricular_weight : ℝ := 0.2
def mid_term_weight : ℝ := 0.3
def final_exam_weight : ℝ := 0.5

def total_grade : ℝ :=
  extracurricular_score * extracurricular_weight +
  mid_term_score * mid_term_weight +
  final_exam_score * final_exam_weight

theorem jiaqi_grade_is_95_3 : total_grade = 95.3 :=
by
  simp [total_grade, extracurricular_score, mid_term_score, final_exam_score,
    extracurricular_weight, mid_term_weight, final_exam_weight]
  sorry

end jiaqi_grade_is_95_3_l796_796827


namespace counterexample_not_coprime_l796_796654

theorem counterexample_not_coprime (a n : ℕ) (h : ¬ a.gcd n = 1) : ¬ (a ^ n ≡ a [MOD n]) :=
by
  let a := 4
  let n := 8
  have h : ¬ a.gcd n = 1 := by
    simp [Nat.gcd]
  have h_pow : (a ^ n) % n = 0 := by
    norm_num [Nat.pow] 
  have h_mod : a % n = 4 := by
    norm_num 
  have h_neq : (0 : ℕ) ≠ (4 : ℕ) := by
    norm_num 
  exact h_neq


end counterexample_not_coprime_l796_796654


namespace sum_of_solutions_eq_239_l796_796665

theorem sum_of_solutions_eq_239 :
  let solutions := { n : ℕ | (∑ i in finset.range(n + 1), i^3) % (n + 5) = 17 }
  solutions.sum = 239 :=
by
  sorry

end sum_of_solutions_eq_239_l796_796665


namespace mean_of_quadrilateral_angles_l796_796028

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796028


namespace mean_value_of_quadrilateral_angles_l796_796057

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796057


namespace base_3_to_base_9_first_digit_l796_796490

theorem base_3_to_base_9_first_digit (x : ℝ) 
  (h : (33.67 * x : ℚ) = 33 + 2 * 3^1 + 1 * 3^2 + 2 * 3^3 + 2 * 3^4 + 1 * 3^5 + 1 * 3^6 + 2 * 3^7 + 2 * 3^8 + 1 * 3^9 + 1 * 3^{10} + 2 * 3^{11} + 2 * 3^{12} + 2 * 3^{13} + 1 * 3^{14} + 1 * 3^{15} + 1 * 3^{16} + 2 * 3^{17} + 2 * 3^{18} + 2 * 3^{19}) :
  x.toBase 9.headDigit = 5 := 
sorry

end base_3_to_base_9_first_digit_l796_796490


namespace sequence_comparison_l796_796358

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Assume conditions
axiom geometric_sequence (h0 : ∀ n, a n = a 0 * q^n)
axiom all_terms_positive (h1 : ∀ n, a n > 0)
axiom common_ratio_not_one (h2 : q ≠ 1)

-- Statement to prove
theorem sequence_comparison (h0 : ∀ n, a n = a 0 * q^n)
  (h1 : ∀ n, a n > 0) (h2 : q ≠ 1) (h3 : q > 0) :
  a 0 + a 7 > a 3 + a 4 :=
by
  sorry

end sequence_comparison_l796_796358


namespace csc_315_equals_neg_sqrt_2_l796_796980

-- Define the conditions
def csc (x : Real) : Real := 1 / Real.sin x
axiom sin_subtraction_360 (x : Real) : Real.sin (360 - x) = -Real.sin x

-- State the problem
theorem csc_315_equals_neg_sqrt_2 : csc 315 = -Real.sqrt 2 := by
  sorry

end csc_315_equals_neg_sqrt_2_l796_796980


namespace magnitude_of_complex_given_condition_l796_796337

theorem magnitude_of_complex_given_condition (b : ℝ) (h : (2 + b * complex.I) * (1 - complex.I)).re = 0 : abs (1 + b * complex.I) = sqrt 5 :=
by
  have b_eq_2 : b = 2 := sorry
  rw [b_eq_2]
  norm_num
  sorry

end magnitude_of_complex_given_condition_l796_796337


namespace starred_result_l796_796667

def star (x y : ℝ) : ℝ := 
x + y / x - y

theorem starred_result : 
    (star (star (-1) 2) (-3)) = -5 / 4 :=
    sorry

end starred_result_l796_796667


namespace interest_rate_calculation_l796_796927

theorem interest_rate_calculation
  (SI : ℕ) (P : ℕ) (T : ℕ) (R : ℕ)
  (h1 : SI = 2100) (h2 : P = 875) (h3 : T = 20) :
  (SI * 100 = P * R * T) → R = 12 :=
by
  sorry

end interest_rate_calculation_l796_796927


namespace find_f_f_of_2_l796_796716

def f (x : ℝ) : ℝ :=
  if x > 1 then log (x + 1) / log 3
  else 2^x + 1

theorem find_f_f_of_2 : f (f 2) = 3 := by
  sorry

end find_f_f_of_2_l796_796716


namespace magic_number_third_operation_l796_796390

-- Definition of cube_sum function
def cube_sum (n : ℕ) : ℕ :=
  (n.toString.data.map (λ c, (c.toNat - '0'.toNat) ^ 3)).sum

-- Theorem: Third cube_sum operation result is 513, and 153 is a "magic" number
theorem magic_number_third_operation (n : ℕ) (h1 : n = 9) : 
  cube_sum (cube_sum (cube_sum n)) = 153 ∧ cube_sum (cube_sum (cube_sum n)) = 513 :=
by {
  have h2 : cube_sum n = 729,
  { calc cube_sum 9 = 9 ^ 3 : by sorry }, -- Compute 9^3
  have h3 : cube_sum (cube_sum n) = 1080,
  { calc cube_sum (cube_sum 9) = 7^3 + 2^3 + 9^3 : by sorry }, -- Compute 7^3 + 2^3 + 9^3 = 1080
  have h4 : cube_sum (cube_sum (cube_sum n)) = 513,
  { calc cube_sum (cube_sum (cube_sum 9)) = 1^3 + 0^3 + 8^3 + 0^3 : by sorry }, -- Compute 1^3 + 0^3 + 8^3 + 0^3 = 513
  have h5 : cube_sum 513 = 153,
  { calc cube_sum 513 = 5^3 + 1^3 + 3^3 : by sorry }, -- Compute 5^3 + 1^3 + 3^3 = 153
  split,
  exact h4, -- Use fourth operation result
  exact h5, -- Verify "magic" number
}

end magic_number_third_operation_l796_796390


namespace range_of_t_sum_of_squares_l796_796724

-- Define the conditions and the problem statement in Lean

variables (a b c t x : ℝ)
variables (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
variables (ineq1 : |x + 1| - |x - 2| ≥ |t - 1| + t)
variables (sum_pos : 2 * a + b + c = 2)

theorem range_of_t :
  (∃ x, |x + 1| - |x - 2| ≥ |t - 1| + t) → t ≤ 2 :=
sorry

theorem sum_of_squares :
  2 * a + b + c = 2 → 0 < a → 0 < b → 0 < c → a^2 + b^2 + c^2 ≥ 2 / 3 :=
sorry

end range_of_t_sum_of_squares_l796_796724


namespace mean_value_interior_angles_quadrilateral_l796_796119

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796119


namespace sum_of_arithmetic_sequence_l796_796777

variables (a_n : Nat → Int) (S_n : Nat → Int)
variable (n : Nat)

-- Definitions based on given conditions:
def is_arithmetic_sequence (a_n : Nat → Int) : Prop :=
∀ n, a_n (n + 1) = a_n n + a_n 1 - a_n 0

def a_1 : Int := -2018

def arithmetic_sequence_sum (S_n : Nat → Int) (a_n : Nat → Int) (n : Nat) : Prop :=
S_n n = n * a_n 0 + (n * (n - 1) / 2 * (a_n 1 - a_n 0))

-- Given condition S_12 / 12 - S_10 / 10 = 2
def condition (S_n : Nat → Int) : Prop :=
S_n 12 / 12 - S_n 10 / 10 = 2

-- Goal: Prove S_2018 = -2018
theorem sum_of_arithmetic_sequence (a_n S_n : Nat → Int)
  (h1 : a_n 1 = -2018)
  (h2 : is_arithmetic_sequence a_n)
  (h3 : ∀ n, arithmetic_sequence_sum S_n a_n n)
  (h4 : condition S_n) :
  S_n 2018 = -2018 :=
sorry

end sum_of_arithmetic_sequence_l796_796777


namespace correct_statements_l796_796237

/-- Define planes α and β as distinct planes -/
constant α β : Type
constant different_planes : α ≠ β

/-- Define the statements as conditions -/
constant statement_1 : ∀ (l : α), (∀ (m : β), perpendicular l m) → perpendicular α β
constant statement_2 : (∀ (l : α), parallel l β) → parallel α β
constant statement_3 : perpendicular α β → (∀ (l : α), perpendicular l β)
constant statement_4 : parallel α β → (∀ (l : α), parallel l β)

/-- Prove that the number of correct statements is 3 -/
theorem correct_statements : 
  (statement_1 and statement_2 and ¬statement_3 and statement_4) → 3 = 3 := by
  sorry

end correct_statements_l796_796237


namespace probability_greater_than_two_l796_796767

noncomputable def probability_of_greater_than_two : ℚ :=
  let total_outcomes := 6
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_greater_than_two (total_outcomes favorable_outcomes : ℕ)
  (ht : total_outcomes = 6) (hf : favorable_outcomes = 4) :
  probability_of_greater_than_two = 2 / 3 :=
by
  rw [probability_of_greater_than_two, ht, hf]
  norm_num
  sorry

end probability_greater_than_two_l796_796767


namespace mean_value_of_quadrilateral_angles_l796_796211

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796211


namespace average_minutes_per_day_l796_796279

-- Definitions based on the conditions
variables (f : ℕ)
def third_graders := 6 * f
def fourth_graders := 2 * f
def fifth_graders := f

def third_graders_time := 10 * third_graders f
def fourth_graders_time := 12 * fourth_graders f
def fifth_graders_time := 15 * fifth_graders f

def total_students := third_graders f + fourth_graders f + fifth_graders f
def total_time := third_graders_time f + fourth_graders_time f + fifth_graders_time f

-- Proof statement
theorem average_minutes_per_day : total_time f / total_students f = 11 := sorry

end average_minutes_per_day_l796_796279


namespace mean_value_of_quadrilateral_angles_l796_796212

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796212


namespace mean_of_quadrilateral_angles_l796_796036

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796036


namespace flat_fee_first_night_l796_796254

theorem flat_fee_first_night :
  ∃ f n : ℚ, (f + 3 * n = 195) ∧ (f + 6 * n = 350) ∧ (f = 40) :=
by
  -- Skipping the detailed proof:
  sorry

end flat_fee_first_night_l796_796254


namespace max_perpendicular_faces_correct_l796_796535

noncomputable def max_perpendicular_faces (n : ℕ) : ℕ :=
if even n then n / 2 else (n + 1) / 2

theorem max_perpendicular_faces_correct (n : ℕ) :
  max_perpendicular_faces n =
  if even n then n / 2 else (n + 1) / 2 := 
by sorry

end max_perpendicular_faces_correct_l796_796535


namespace find_C_l796_796860

noncomputable def A_annual_income : ℝ := 403200.0000000001
noncomputable def A_monthly_income : ℝ := A_annual_income / 12 -- 33600.00000000001

noncomputable def x : ℝ := A_monthly_income / 5 -- 6720.000000000002

noncomputable def C : ℝ := (2 * x) / 1.12 -- should be 12000.000000000004

theorem find_C : C = 12000.000000000004 := 
by sorry

end find_C_l796_796860


namespace mean_of_quadrilateral_angles_l796_796039

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796039


namespace cannot_be_square_of_binomial_B_l796_796542

theorem cannot_be_square_of_binomial_B (x y m n : ℝ) :
  (∃ (a b : ℝ), (3*x + 7*y) * (3*x - 7*y) = a^2 - b^2) ∧
  (∃ (a b : ℝ), ( -0.2*x - 0.3) * ( -0.2*x + 0.3) = a^2 - b^2) ∧
  (∃ (a b : ℝ), ( -3*n - m*n) * ( 3*n - m*n) = a^2 - b^2) ∧
  ¬(∃ (a b : ℝ), ( 5*m - n) * ( n - 5*m) = a^2 - b^2) :=
by
  sorry

end cannot_be_square_of_binomial_B_l796_796542


namespace mean_of_quadrilateral_angles_is_90_l796_796093

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796093


namespace number_of_elements_in_M_l796_796324

def Delta (m n : ℕ) : ℕ :=
  if (m % 2 = n % 2) then m + n else m * n

def isValidPair (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ Delta x y = 36

def M : set (ℕ × ℕ) := { p | isValidPair p.1 p.2 }

theorem number_of_elements_in_M :
  (finset.filter (λ (p : ℕ × ℕ), isValidPair p.1 p.2) (finset.product (finset.range 37) (finset.range 37))).card = 40 :=
sorry

end number_of_elements_in_M_l796_796324


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796065

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796065


namespace sum_slope_y_intercept_l796_796779

theorem sum_slope_y_intercept (A B C F : ℝ × ℝ) (midpoint_A_C : F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) 
  (coords_A : A = (0, 6)) (coords_B : B = (0, 0)) (coords_C : C = (8, 0)) :
  let slope : ℝ := (F.2 - B.2) / (F.1 - B.1)
  let y_intercept : ℝ := B.2
  slope + y_intercept = 3 / 4 := by
{
  -- proof steps
  sorry
}

end sum_slope_y_intercept_l796_796779


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796072

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796072


namespace triangle_angle_condition_l796_796905

theorem triangle_angle_condition (A B C O F E D M N K : Type) 
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited O] 
  [Inhabited F] [Inhabited E] [Inhabited D] [Inhabited M] 
  [Inhabited N] [Inhabited K]
  (h1 : is_triangle A B C)
  (h2 : is_circumcenter O A B C)
  (h3 : line BO ∩ AC = F)
  (h4 : line CO ∩ AB = E)
  (h5 : perp_bisector FE ∩ BC = D)
  (h6 : line DE ∩ line BF = M)
  (h7 : line DF ∩ line CE = N)
  (h8 : intersect_perp_bisectors EM FN K)
  (h9 : on_line K EF) :
  angle A B C = 60 := 
sorry

end triangle_angle_condition_l796_796905


namespace harry_geckos_count_l796_796736

theorem harry_geckos_count 
  (G : ℕ)
  (iguanas : ℕ := 2)
  (snakes : ℕ := 4)
  (cost_snake : ℕ := 10)
  (cost_iguana : ℕ := 5)
  (cost_gecko : ℕ := 15)
  (annual_cost : ℕ := 1140) :
  12 * (snakes * cost_snake + iguanas * cost_iguana + G * cost_gecko) = annual_cost → 
  G = 3 := 
by 
  intros h
  sorry

end harry_geckos_count_l796_796736


namespace extreme_points_minimum_value_l796_796722

open Real

-- Define functions f and g according to the problem statement
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := ln x + 2 * x^2 + m * x
noncomputable def g (x : ℝ) (n : ℝ) : ℝ := exp x + 3 * x^2 - n * x

-- Prove that for m ≥ -4, f(x) has no extreme points on (0, +∞), and for m < -4, f(x) has two extreme points on (0, +∞)
theorem extreme_points (m : ℝ) : 
  (m ≥ -4 → ∀ x > 0, deriv (f x m) x ≠ 0) ∧ (m < -4 → ∃ x1 x2 > 0, x1 ≠ x2 ∧ deriv (f x m) x1 = 0 ∧ deriv (f x m) x2 = 0) := sorry

-- Given the inequality f(x) ≤ g(x) ∀ x > 0, and m, n are positive, prove the minimum value of 1/m + 4/n is 9/(e + 1)
theorem minimum_value (m n : ℝ) (hp_m : 0 < m) (hp_n : 0 < n) (H : ∀ x > 0, f x m ≤ g x n) : 
  1 / m + 4 / n ≥ 9 / (exp 1 + 1) := sorry

end extreme_points_minimum_value_l796_796722


namespace twelfth_prime_is_37_and_sum_is_156_l796_796833

theorem twelfth_prime_is_37_and_sum_is_156 :
  let ps := [17, 19, 23, 29, 31, 37] in
  (nth_prime 12 = 37) ∧ (ps.sum = 156) :=
by
  sorry

end twelfth_prime_is_37_and_sum_is_156_l796_796833


namespace river_width_l796_796602

theorem river_width 
  (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) 
  (h_depth : depth = 2) 
  (h_flow_rate: flow_rate_kmph = 3) 
  (h_volume : volume_per_minute = 4500) : 
  the_width_of_the_river = 45 :=
by
  sorry 

end river_width_l796_796602


namespace negation_exists_of_forall_l796_796862

theorem negation_exists_of_forall (P : ∀ (x : ℝ), x^2 + 1 > 0) : ¬ P ↔ ∃ (x : ℝ), x^2 + 1 ≤ 0 := 
sorry

end negation_exists_of_forall_l796_796862


namespace mean_of_quadrilateral_angles_l796_796038

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796038


namespace min_marked_squares_l796_796427

/-
  Let’s define the problem formally in Lean:
  Given a 1001×1001 board, we need to mark m squares such that:
  1. If two squares are adjacent, at least one of them is marked.
  2. If there are six consecutive squares in a row or column, at least two adjacent squares among them are marked.
  We need to prove that the minimum number of marked squares is 601200.
-/

def is_adjacent (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 + 1 = y2)) ∨ (y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 + 1 = x2))

def satisfies_conditions (markedsquares : finset (ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ), x < 1001 ∧ y < 1001 → 
    (if x + 1 < 1001 then (markedsquares.contains (x, y) ∨ markedsquares.contains (x + 1, y)) else True) ∧
    (if y + 1 < 1001 then (markedsquares.contains (x, y) ∨ markedsquares.contains (x, y + 1)) else True) ∧
    ∀ i j, i < 996 ∧ j < 1001 →
      markedsquares.filter (λ (p : ℕ × ℕ), p.1 = i) ∧ markedsquares.filter (λ (p : ℕ × ℕ), p.2 = j).card ≥ 3

theorem min_marked_squares : ∃ (markedsquares : finset (ℕ × ℕ)), 
  satisfies_conditions markedsquares ∧ markedsquares.card = 601200 := 
sorry

end min_marked_squares_l796_796427


namespace area_rectangle_l796_796509

theorem area_rectangle 
    (x y : ℝ)
    (h1 : 5 * x + 4 * y = 10)
    (h2 : 3 * x = 2 * y) :
    5 * (x * y) = 3000 / 121 :=
by
  sorry

end area_rectangle_l796_796509


namespace prove_equilateral_l796_796412

-- Definition of the general problem setup
variables {A B C D E F O H P : Type}
variable (acute : Π (A B C : Type), Triangle A B C → Prop)
variable (altitudes : Π (D E F : Type) (A B C : Type), Triangle A B C → Prop)
variable (circumcenter : Π (O : Type) (A B C : Type), Triangle A B C → Prop)
variable (orthocenter : Π (H : Type) (A B C : Type), Triangle A B C → Prop)
variable (intersection : Π (P : Type) (CF AO : Type), Point CF AO → Prop)
variable (equality_of_segments : Π (FP EH : Type), Segment FP EH → Prop)
variable (circle_passing_through : Π (A O H C : Type), Circle A O H C → Prop)

theorem prove_equilateral
  (h_acute : acute A B C <| Triangle.mk)
  (h_altitudes : altitudes D E F A B C <| Triangle.mk)
  (h_circumcenter : circumcenter O A B C <| Triangle.mk)
  (h_orthocenter : orthocenter H A B C <| Triangle.mk)
  (h_intersection : intersection P CF AO <| Point.mk)
  (h_equality_of_segments : equality_of_segments FP EH <| Segment.mk)
  (h_circle_passing_through : circle_passing_through A O H C <| Circle.mk)
: equilateral △ ABC :=
sorry

end prove_equilateral_l796_796412


namespace vector_perpendicular_l796_796735

variables (a b : Vector ℝ)
def a := ⟨1, 0⟩
def b := ⟨(1 / 2), (1 / 2)⟩

theorem vector_perpendicular (a b : Vector ℝ) (h₁: a = ⟨1, 0⟩) (h₂: b = ⟨1/2, 1/2⟩) : (a - b) ⬝ b = 0 :=
by
  sorry

end vector_perpendicular_l796_796735


namespace length_of_AC_l796_796417

-- Define the conditions: lengths and angle
def AB : ℝ := 10
def BC : ℝ := 10
def CD : ℝ := 15
def DA : ℝ := 15
def angle_ADC : ℝ := 120

-- Prove the length of diagonal AC is 15*sqrt(3)
theorem length_of_AC : 
  (CD ^ 2 + DA ^ 2 - 2 * CD * DA * Real.cos (angle_ADC * Real.pi / 180)) = (15 * Real.sqrt 3) ^ 2 :=
by
  sorry

end length_of_AC_l796_796417


namespace accessories_cost_is_200_l796_796785

variable (c_cost a_cost : ℕ)
variable (ps_value ps_sold : ℕ)
variable (john_paid : ℕ)

-- Given Conditions
def computer_cost := 700
def accessories_cost := a_cost
def playstation_value := 400
def playstation_sold := ps_value - (ps_value * 20 / 100)
def john_paid_amount := 580

-- Theorem to be proved
theorem accessories_cost_is_200 :
  ps_value = 400 →
  ps_sold = playstation_sold →
  c_cost = 700 →
  john_paid = 580 →
  john_paid + ps_sold - c_cost = a_cost →
  a_cost = 200 :=
by
  intros
  sorry

end accessories_cost_is_200_l796_796785


namespace fair_attendance_l796_796467

theorem fair_attendance (P_this_year P_next_year P_last_year : ℕ) 
  (h1 : P_this_year = 600) 
  (h2 : P_next_year = 2 * P_this_year) 
  (h3 : P_last_year = P_next_year - 200) : 
  P_this_year = 600 ∧ P_next_year = 1200 ∧ P_last_year = 1000 :=
by
  split
  . exact h1
  split
  . rw [h1]
    rw [h2]
    norm_num
  . rw [h1]
    rw [h2]
    rw [h3]
    norm_num

example : fair_attendance 600 1200 1000 :=
by sorry

end fair_attendance_l796_796467


namespace mean_value_of_quadrilateral_angles_l796_796134

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796134


namespace mean_value_of_quadrilateral_angles_l796_796078

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796078


namespace min_attacking_rook_pairs_l796_796475

theorem min_attacking_rook_pairs : 
  ∀ (B : fin 8 → fin 8 → Prop) (R : fin 8 → fin 8 → Prop), 
    (∀ i j, B i j → R i j) →
    (∃ K : set (fin 8 × fin 8), (K.card = 16 ∧ (∀ (i j : fin 8) (hi: i ≠ j), K (i, j) → ∃ row col, R row col ∧ ((row = i) ∨ (col = j)) ∧ ∀ rook ∈ K, rook.1 ≠ row ∨ rook.2 ≠ col))
     → K.card = 16 ∧ (minimum pairs that can attack each other) = 16
:= sorry

end min_attacking_rook_pairs_l796_796475


namespace total_money_before_spending_l796_796242

-- Define the amounts for each friend
variables (J P Q A: ℝ)

-- Define the conditions from the problem
def condition1 := P = 2 * J
def condition2 := Q = P + 20
def condition3 := A = 1.15 * Q
def condition4 := J + P + Q + A = 1211
def cost_of_item : ℝ := 1200

-- The total amount before buying the item
theorem total_money_before_spending (J P Q A : ℝ)
  (h1 : condition1 J P)
  (h2 : condition2 P Q)
  (h3 : condition3 Q A)
  (h4 : condition4 J P Q A) : 
  J + P + Q + A - cost_of_item = 11 :=
by
  sorry

end total_money_before_spending_l796_796242


namespace mean_value_interior_angles_quadrilateral_l796_796109

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796109


namespace mean_value_interior_angles_quadrilateral_l796_796121

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796121


namespace mean_value_of_quadrilateral_angles_l796_796047

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796047


namespace mean_value_of_quadrilateral_interior_angles_l796_796183

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796183


namespace arrangement_books_not_adjacent_distribution_books_students_l796_796546

open Finset

-- Given 5 different books including 2 mathematics books
variable (books : Finset Nat) (math_books : Finset Nat)
variable (mathematic_books : books ⊆ math_books)

-- We need to prove:
-- 1. There are 60 ways to arrange the books such that the mathematics books are neither adjacent nor at both ends simultaneously.
theorem arrangement_books_not_adjacent (h1 : books.card = 5) (h2 : math_books.card = 2) :
  (arrangements books math_books) = 60 := sorry

-- 2. There are 150 ways to distribute the books to 3 students such that each student receives at least 1 book.
theorem distribution_books_students (h1 : books.card = 5) (h3 : students.card = 3) (every_student_one : ∀ (s ∈ students), (s.card ≥ 1)):
  (distribution books students) = 150 := sorry

end arrangement_books_not_adjacent_distribution_books_students_l796_796546


namespace bug_travel_points_l796_796286

def point := (ℤ × ℤ)

def A : point := (-2, 3)
def B : point := (4, -3)

def manhattan_distance (p1 p2 : point) : ℤ :=
  abs (p2.1 - p1.1) + abs (p2.2 - p1.2)

def valid_path_length : ℤ := 22

def on_valid_path (p : point) : Prop :=
  abs (p.1 + 2) + abs (p.2 - 3) + abs (p.1 - 4) + abs (p.2 + 3) ≤ valid_path_length

def integer_coordinate_points (A B : point) : ℤ :=
  (finset.univ.filter on_valid_path).card

theorem bug_travel_points : integer_coordinate_points A B = 229 :=
  sorry

end bug_travel_points_l796_796286


namespace mean_of_quadrilateral_angles_l796_796041

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796041


namespace non_similar_isosceles_triangles_distinct_positive_arithmetic_l796_796737

-- Define the conditions and the theorem in Lean 4
theorem non_similar_isosceles_triangles_distinct_positive_arithmetic :
  ∃! d : ℕ, (0 < d ∧ d < 15) ∧ (∀ n : ℕ, 3 * n + 2 * d = 180 ∧
  n > 0 ∧ (180 - 2 * d) % 3 = 0) := 
begin
  sorry
end

end non_similar_isosceles_triangles_distinct_positive_arithmetic_l796_796737


namespace koala_fiber_l796_796788

theorem koala_fiber (absorption_percent: ℝ) (absorbed_fiber: ℝ) (total_fiber: ℝ) 
  (h1: absorption_percent = 0.25) 
  (h2: absorbed_fiber = 10.5) 
  (h3: absorbed_fiber = absorption_percent * total_fiber) : 
  total_fiber = 42 :=
by
  rw [h1, h2] at h3
  have h : 10.5 = 0.25 * total_fiber := h3
  sorry

end koala_fiber_l796_796788


namespace negation_of_exists_prop_l796_796861

theorem negation_of_exists_prop (x : ℝ) :
  (¬ ∃ (x : ℝ), (x > 0) ∧ (|x| + x >= 0)) ↔ (∀ (x : ℝ), x > 0 → |x| + x < 0) := 
sorry

end negation_of_exists_prop_l796_796861


namespace value_of_f_neg_one_l796_796855

noncomputable def f : ℝ → ℝ 
| x => if x < 3 then f (x + 3) else Real.log2 (x - 1)

theorem value_of_f_neg_one : f (-1) = 2 := by 
  sorry

end value_of_f_neg_one_l796_796855


namespace find_m_l796_796273

def is_good (n : ℤ) : Prop :=
  ¬ ∃ k : ℕ, |n| = k^2

noncomputable def is_square_of_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k^2 ∧ odd k

theorem find_m (m : ℤ) (k : ℤ) :
  (∃ u v w : ℤ, u ≠ v ∧ v ≠ w ∧ w ≠ u ∧ is_good u ∧ is_good v ∧ is_good w ∧ m = u + v + w ∧ is_square_of_odd (u * v * w)) ↔ m = 4 * k + 3 :=
sorry

end find_m_l796_796273


namespace limit_of_expr_is_1_div_6_l796_796953

noncomputable def limit_expr (x : ℝ) : ℝ :=
  (x - Real.sin x) / (x ^ 3)

theorem limit_of_expr_is_1_div_6 : 
  Mathlib.analysis.special_functions.trigonometric.has_limit (λ x, limit_expr x) 0 (1 / 6) :=
by {
  sorry
}

end limit_of_expr_is_1_div_6_l796_796953


namespace additional_discount_during_special_sale_l796_796934

-- Definitions for the conditions
def list_price : ℝ := 80
def usual_discount_min : ℝ := 0.30
def usual_discount_max : ℝ := 0.50
def special_sale_discounted_price : ℝ := 0.40 * list_price

-- The theorem to be proved
theorem additional_discount_during_special_sale :
  ∃ (additional_discount : ℝ), additional_discount = 0.20 :=
by {
  -- The price after the maximum usual discount
  let usual_max_discounted_price := list_price * (1 - usual_discount_max),
  -- The additional discount from the usual max discounted price to the special sale price
  let actual_additional_discount := (usual_max_discounted_price - special_sale_discounted_price) / usual_max_discounted_price,
  -- The actual additional discount should be 0.20
  exact ⟨actual_additional_discount, by norm_num⟩,
}

end additional_discount_during_special_sale_l796_796934


namespace positive_t_for_magnitude_l796_796669

theorem positive_t_for_magnitude : ∃ t : ℕ, 0 < t ∧ absC (-3 + t * Complex.I) = 3 * Real.sqrt 5 ∧ t = 6 :=
by
  use 6
  sorry

end positive_t_for_magnitude_l796_796669


namespace stefan_total_spent_l796_796484

def total_spent (appetizer_cost : ℝ) (entree_cost : ℝ) (num_entrees : ℕ)
                (entree_discount : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
let discounted_entree_cost := entree_cost * (1 - entree_discount)
    total_entree_cost := discounted_entree_cost * num_entrees
    subtotal := appetizer_cost + total_entree_cost
    tax := subtotal * sales_tax_rate
    total_before_tip := subtotal + tax
    tip := total_before_tip * tip_rate
    total := total_before_tip + tip
in total

theorem stefan_total_spent :
  total_spent 10 20 4 0.10 0.08 0.20 = 106.27 :=
by
  sorry

end stefan_total_spent_l796_796484


namespace mean_value_of_quadrilateral_angles_l796_796055

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796055


namespace tetrahedron_surface_area_l796_796409

theorem tetrahedron_surface_area (x : ℝ) (M N : ℝ) (S : ℝ) (h_MN : M = x / (real.sqrt 2))
  (h_S : S = (1/2) * (x * real.cos (pi / 3) + x * real.sin (pi / 3)) * (x / (real.sqrt 2)))
  (h_angle : real.abs (real.cos (pi / 3) - real.sin (pi / 3)) = real.sqrt (2 / 3)) :
  (4 * S = (2 * x^2 * real.sqrt 2) / 3) :=
sorry

end tetrahedron_surface_area_l796_796409


namespace diana_larger_than_apollo_l796_796296

def eight_sided_die : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

theorem diana_larger_than_apollo :
  let outcomes := List.product eight_sided_die eight_sided_die;
  let successful_outcomes := outcomes.filter (λ p, p.1 > p.2);
  (successful_outcomes.length / outcomes.length.toFloat) = 7 / 16 := by
  sorry

end diana_larger_than_apollo_l796_796296


namespace min_value_4x_3y_l796_796392

theorem min_value_4x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y = 5 * x * y) : 
  4 * x + 3 * y ≥ 5 :=
sorry

end min_value_4x_3y_l796_796392


namespace four_numbers_pairwise_coprime_from_five_l796_796480

theorem four_numbers_pairwise_coprime_from_five (a₁ a₂ a₃ a₄ a₅ : ℕ) 
  (h1 : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₄ ≠ a₅)
  (h2 : ∀ (i j : ℕ), i ≠ j → i ∈ {a₁, a₂, a₃, a₄, a₅} → j ∈ {a₁, a₂, a₃, a₄, a₅} → Nat.gcd i j = 1) :
  ∃ (b₁ b₂ b₃ b₄ : ℕ), b₁ ≠ b₂ ∧ b₁ ≠ b₃ ∧ b₁ ≠ b₄ ∧ b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₃ ≠ b₄ ∧ 
  b₁ ∈ {a₁, a₂, a₃, a₄, a₅} ∧ b₂ ∈ {a₁, a₂, a₃, a₄, a₅} ∧ b₃ ∈ {a₁, a₂, a₃, a₄, a₅} ∧ b₄ ∈ {a₁, a₂, a₃, a₄, a₅} ∧ 
  ∀ (i j : ℕ), i ≠ j → i ∈ {b₁, b₂, b₃, b₄} → j ∈ {b₁, b₂, b₃, b₄} → Nat.gcd i j = 1 :=
begin
  sorry
end

end four_numbers_pairwise_coprime_from_five_l796_796480


namespace mean_value_of_quadrilateral_angles_l796_796164

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796164


namespace mean_of_quadrilateral_angles_l796_796196

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796196


namespace mean_of_quadrilateral_angles_l796_796032

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796032


namespace mean_of_quadrilateral_angles_l796_796029

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796029


namespace trapezoid_AD_condition_l796_796846

theorem trapezoid_AD_condition (A B C D M N O : Point) (AD BC AO : ℝ) 
  (h1 : AD = 4 * BC)
  (h2 : ∠A + ∠D = 120)
  (h3 : CN / ND = 1 / 2)
  (h4 : BM / MA = 1 / 2)
  (h5 : perpendicular M A O)
  (h6 : perpendicular N D O)
  (h7 : AO = 1)
  : AD = sqrt 3 := 
sorry

end trapezoid_AD_condition_l796_796846


namespace intersection_point_l796_796398

theorem intersection_point (k : ℝ) :
  let l1 : ℝ → ℝ := λ x, k * x + 2
  let l2 : ℝ → ℝ := λ x, -x - 2
  (∃ x y, l1 x = y ∧ l2 x = y) → (l1 (-2) = 0 ∧ l2 (-2) = 0) :=
by
  sorry

end intersection_point_l796_796398


namespace avg_length_remaining_wires_l796_796565

theorem avg_length_remaining_wires (N : ℕ) (avg_length : ℕ) 
    (third_wires_count : ℕ) (third_wires_avg_length : ℕ) 
    (total_length : ℕ := N * avg_length) 
    (third_wires_total_length : ℕ := third_wires_count * third_wires_avg_length) 
    (remaining_wires_count : ℕ := N - third_wires_count) 
    (remaining_wires_total_length : ℕ := total_length - third_wires_total_length) :
    N = 6 → 
    avg_length = 80 → 
    third_wires_count = 2 → 
    third_wires_avg_length = 70 → 
    remaining_wires_count = 4 → 
    remaining_wires_total_length / remaining_wires_count = 85 :=
by 
  intros hN hAvg hThirdCount hThirdAvg hRemainingCount
  sorry

end avg_length_remaining_wires_l796_796565


namespace transfer_increases_averages_l796_796908

noncomputable def initial_average_A : ℚ := 47.2
noncomputable def initial_average_B : ℚ := 41.8
def initial_students : ℕ := 10
noncomputable def score_filin : ℚ := 44
noncomputable def score_lopatin : ℚ := 47

theorem transfer_increases_averages :
  let sum_A := initial_students * initial_average_A,
      sum_B := initial_students * initial_average_B,
      new_sum_A := sum_A - score_lopatin - score_filin,
      new_average_A := new_sum_A / (initial_students - 2),
      new_sum_B := sum_B + score_lopatin + score_filin,
      new_average_B := new_sum_B / (initial_students + 2)
  in new_average_A > initial_average_A / 2 ∧ new_average_B > initial_average_B / 2 :=
by {
  sorry
}

end transfer_increases_averages_l796_796908


namespace probability_greater_than_two_on_die_l796_796765

variable {favorable_outcomes : ℕ := 4}
variable {total_outcomes : ℕ := 6}

theorem probability_greater_than_two_on_die : (favorable_outcomes : ℚ) / total_outcomes = 2 / 3 := 
by
  sorry

end probability_greater_than_two_on_die_l796_796765


namespace mean_of_quadrilateral_angles_l796_796031

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796031


namespace triangle_inequality_lemma_l796_796556

theorem triangle_inequality_lemma 
  {A B C D E : Type}
  [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E]
  (AC AD BC BD EC ED : ℝ)
  (D_inside : D)
  (E_on_AB : E → A → B → Prop)
  (h1 : AC - AD ≥ 1)
  (h2 : BC - BD ≥ 1) 
  : E_on_AB E A B → EC - ED ≥ 1 := 
sorry

end triangle_inequality_lemma_l796_796556


namespace imaginary_part_2_l796_796859

def complex := ℂ

variable (z : complex)
variable (x y : ℂ)

theorem imaginary_part_2 : (z = (2 + complex.i) * complex.i) → (z.im = 2) :=
by
  sorry

end imaginary_part_2_l796_796859


namespace csc_315_eq_neg_sqrt_two_l796_796991

theorem csc_315_eq_neg_sqrt_two : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_two_l796_796991


namespace percentage_discount_is_10_l796_796592

-- Definition of the wholesale price
def wholesale_price : ℝ := 90

-- Definition of the retail price
def retail_price : ℝ := 120

-- Definition of the profit in terms of percentage of the wholesale price
def profit : ℝ := 0.2 * wholesale_price

-- Definition of the selling price based on the wholesale price and profit
def selling_price : ℝ := wholesale_price + profit

-- Definition of the discount amount
def discount_amount : ℝ := retail_price - selling_price

-- Definition of the percentage discount
def percentage_discount : ℝ := (discount_amount / retail_price) * 100

-- The theorem we want to prove: the percentage discount is 10%
theorem percentage_discount_is_10 : percentage_discount = 10 := 
by
  sorry

end percentage_discount_is_10_l796_796592


namespace solve_for_S_l796_796576

variable (D S : ℝ)
variable (h1 : D > 0)
variable (h2 : S > 0)
variable (h3 : ((0.75 * D) / 50 + (0.25 * D) / S) / D = 1 / 50)

theorem solve_for_S :
  S = 50 :=
by
  sorry

end solve_for_S_l796_796576


namespace arctan_sum_eq_pi_over_3_l796_796314

noncomputable def find_n : ℕ := 14

theorem arctan_sum_eq_pi_over_3 (n : ℕ) 
  (h : Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/6) + Real.arctan (1/n) = Real.pi / 3) : 
  n = find_n := 
begin 
  sorry 
end

end arctan_sum_eq_pi_over_3_l796_796314


namespace mean_value_of_quadrilateral_angles_l796_796050

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796050


namespace total_vessels_l796_796570

theorem total_vessels (C G S F : ℕ) (h1 : C = 4) (h2 : G = 2 * C) (h3 : S = G + 6) (h4 : S = 7 * F) : 
  C + G + S + F = 28 :=
by
  sorry

end total_vessels_l796_796570


namespace speed_of_canoe_downstream_l796_796574

-- Definition of the problem conditions
def speed_of_canoe_in_still_water (V_c : ℝ) (V_s : ℝ) (upstream_speed : ℝ) : Prop :=
  V_c - V_s = upstream_speed

def speed_of_stream (V_s : ℝ) : Prop :=
  V_s = 4

-- The statement we want to prove
theorem speed_of_canoe_downstream (V_c V_s : ℝ) (upstream_speed : ℝ) 
  (h1 : speed_of_canoe_in_still_water V_c V_s upstream_speed)
  (h2 : speed_of_stream V_s)
  (h3 : upstream_speed = 4) :
  V_c + V_s = 12 :=
by
  sorry

end speed_of_canoe_downstream_l796_796574


namespace rectangle_area_l796_796331

/-- Four congruent circles with centers P, Q, R, and S are tangent to the sides
    of rectangle ABCD. The circle centered at P has a radius of 3 and passes
    through points Q, R, and S. The line passing through the centers P, Q, R, and S
    is horizontal and touches the sides AB and CD of the rectangle. 
    Prove that the area of the rectangle ABCD is 72. -/
theorem rectangle_area (r : ℝ) (h_congruent : ∀ {P Q R S : Type}, r = 3)
    (h_tangent : ∀ {A B C D P Q R S : Type}, 
                 (P.1 = A.1 + r ∧ P.2 = A.2 + r) ∧
                 (Q.1 = A.1 + 3*r ∧ Q.2 = A.2 + r) ∧
                 (R.1 = A.1 + 5*r ∧ R.2 = A.2 + r) ∧
                 (S.1 = A.1 + 7*r ∧ S.2 = A.2 + r)) :
  let height := 2 * r,
      width := 4 * r + 2 * r in
  height * width = 72 :=
by
  sorry

end rectangle_area_l796_796331


namespace sin_600_eq_neg_sqrt_3_div_2_l796_796559

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * (Real.pi / 180)) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_600_eq_neg_sqrt_3_div_2_l796_796559


namespace coefficient_of_x_term_l796_796851

theorem coefficient_of_x_term (k : ℝ) (h : k = 1.7777777777777777) (h_discriminant : (3 * k)^2 - 4 * (2 * k) * 2 = 0) :
  3 * k = 5.333333333333333 :=
by
  rw h
  sorry

end coefficient_of_x_term_l796_796851


namespace mean_of_quadrilateral_angles_is_90_l796_796090

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796090


namespace mean_value_of_quadrilateral_angles_l796_796079

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796079


namespace line_through_circle_center_l796_796501

theorem line_through_circle_center {m : ℝ} :
  (∃ (x y : ℝ), x - 2*y + m = 0 ∧ x^2 + y^2 + 2*x - 4*y = 0) → m = 5 :=
by
  sorry

end line_through_circle_center_l796_796501


namespace number_of_correct_statements_is_2_l796_796495

def statement_1 : Prop :=
  ∀ (x : ℝ), irrational x → ∃ a, a ∈ Set.RealNumbersNonTerminatingDecimals x

def statement_2 : Prop :=
  ∀ (x : ℚ), ∃ (y : ℝ), y = x ∧ (∃ (z : ℝ), z ≠ x)

def statement_3 : Prop :=
  ∀ (x : ℝ), ∀ (y : ℝ), abs x = x ↔ x = 0 ∨ x = y

def statement_4 : Prop :=
  fraction (π / 2)

def statement_5 : Prop :=
  ∀ (a : ℝ), a = 7.30 → 7.295 ≤ a ∧ a < 7.305

def all_statements : List Prop :=
  [statement_1, statement_2, statement_3, statement_4, statement_5]

def count_correct_statements : ℕ :=
  all_statements.filter (λ s, s).length

theorem number_of_correct_statements_is_2 :
  count_correct_statements = 2 := sorry

end number_of_correct_statements_is_2_l796_796495


namespace games_needed_to_declare_winner_l796_796938

def single_elimination_games (T : ℕ) : ℕ :=
  T - 1

theorem games_needed_to_declare_winner (T : ℕ) :
  (single_elimination_games 23 = 22) :=
by
  sorry

end games_needed_to_declare_winner_l796_796938


namespace number_of_eaten_tourists_l796_796580

def tourists_eaten_by_anacondas (E : ℕ) : Prop :=
  let remaining = 30 - E in
  let poisoned = (remaining) / 2 in
  let not_recovered = (6 / 7 : ℝ) * poisoned in
  30 - E - not_recovered = 16

theorem number_of_eaten_tourists : ∃ E : ℕ, tourists_eaten_by_anacondas E ∧ E = 2 :=
by
  unfold tourists_eaten_by_anacondas
  sorry

end number_of_eaten_tourists_l796_796580


namespace probability_of_rolling_perfect_square_on_eight_sided_die_l796_796488

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k^2 = n

theorem probability_of_rolling_perfect_square_on_eight_sided_die :
  ∃ p : ℚ, p = 1 / 4 ∧
  ∀ (rolling : ℕ), rolling ∈ {1, 2, 3, 4, 5, 6, 7, 8} →
    (is_perfect_square rolling ↔ rolling = 1 ∨ rolling = 4) →
    (prob_of_perfect_square : ℚ) = 1 / 4 := sorry

end probability_of_rolling_perfect_square_on_eight_sided_die_l796_796488


namespace area_of_defined_region_l796_796290

theorem area_of_defined_region : 
  ∀ (x y : ℝ), x^2 + y^2 + 4 * x - 6 * y + 9 = 0 → (∀ r : ℝ, r = 2 → real.pi * r^2 = 4 * real.pi) :=
by
  sorry

end area_of_defined_region_l796_796290


namespace min_weight_requires_at_least_three_weights_l796_796682

def weights : List ℕ := [1, 2, 3, 5]

def can_measure_with_at_least_three_weights (w : ℕ) : Prop :=
  ∃ w1 w2 w3 (h1 : w1 ∈ weights) (h2 : w2 ∈ weights) (h3 : w3 ∈ weights) (w_comb : List ℕ), 
  (w = w1 + w2 + w3) ∧ (w_comb = [w1, w2, w3]) ∧ (w_comb.nodup) ∧ (w_comb.length ≥ 3)

theorem min_weight_requires_at_least_three_weights : ∃ (w : ℕ), can_measure_with_at_least_three_weights w ∧ (¬ ∃ (w' : ℕ), w' < w ∧ can_measure_with_at_least_three_weights w') :=
  sorry

end min_weight_requires_at_least_three_weights_l796_796682


namespace mean_value_of_quadrilateral_angles_l796_796159

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796159


namespace shopkeeper_total_profit_percentage_l796_796263

noncomputable def profit_percentage (actual_weight faulty_weight ratio : ℕ) : ℝ :=
  (actual_weight - faulty_weight) / actual_weight * 100 * ratio

noncomputable def total_profit_percentage (ratios profits : List ℝ) : ℝ :=
  (List.sum (List.zipWith (· * ·) ratios profits)) / (List.sum ratios)

theorem shopkeeper_total_profit_percentage :
  let actual_weight := 1000
  let faulty_weights := [900, 850, 950]
  let profit_percentages := [10, 15, 5]
  let ratios := [3, 2, 1]
  total_profit_percentage ratios profit_percentages = 10.83 :=
by
  sorry

end shopkeeper_total_profit_percentage_l796_796263


namespace f_one_zero_inequality_solution_l796_796547

noncomputable def f : ℝ → ℝ := sorry

axiom increasing_f : ∀ x y, 0 < x → 0 < y → x < y → f x < f y
axiom functional_eq : ∀ x y, 0 < x → 0 < y → f (x / y) = f x - f y
axiom f_six : f 6 = 1

-- Part 1: Prove that f(1) = 0
theorem f_one_zero : f 1 = 0 := sorry

-- Part 2: Prove that ∀ x ∈ (0, (-3 + sqrt 153) / 2), f(x + 3) - f(1 / x) < 2
theorem inequality_solution : ∀ x, 0 < x → x < (-3 + Real.sqrt 153) / 2 → f (x + 3) - f (1 / x) < 2 := sorry

end f_one_zero_inequality_solution_l796_796547


namespace sin_beta_value_l796_796356

theorem sin_beta_value 
  (α β : ℝ)
  (h1 : 0 < α) 
  (h2 : α < β) 
  (h3 : β < π / 2) 
  (h4 : sin α = 3 / 5) 
  (h5 : cos (β - α) = 12 / 13) : 
  sin β = 56 / 65 :=
by
  sorry

end sin_beta_value_l796_796356


namespace mean_value_of_quadrilateral_angles_l796_796075

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796075


namespace avg_time_stopped_per_hour_l796_796644

-- Definitions and conditions
def avgSpeedInMotion : ℝ := 75
def overallAvgSpeed : ℝ := 40

-- Statement to prove
theorem avg_time_stopped_per_hour :
  (1 - overallAvgSpeed / avgSpeedInMotion) * 60 = 28 := 
by
  sorry

end avg_time_stopped_per_hour_l796_796644


namespace positional_relationships_between_line_and_plane_l796_796507

-- Mathematical Definitions
def in_plane : Type := sorry -- Definition of a line in the plane
def intersects_plane : Type := sorry -- Definition of a line intersecting the plane
def parallel_to_plane : Type := sorry -- Definition of a line parallel to the plane

-- We now state the proposition to be proved: that there are exactly 3 types of positional relationships
theorem positional_relationships_between_line_and_plane : 
  (in_plane + intersects_plane + parallel_to_plane).sum = 3 :=
sorry

end positional_relationships_between_line_and_plane_l796_796507


namespace sum_is_945_l796_796558

def sum_of_integers_from_90_to_99 : ℕ :=
  90 + 91 + 92 + 93 + 94 + 95 + 96 + 97 + 98 + 99

theorem sum_is_945 : sum_of_integers_from_90_to_99 = 945 := 
by
  sorry

end sum_is_945_l796_796558


namespace log_eq_implies_x_gt_y_l796_796701

theorem log_eq_implies_x_gt_y {x y : ℝ} (hx : 0 < x) (hy : 0 < y)
  (h : log 2 x + 2 * x = log 2 y + 3 * y) : x > y :=
sorry

end log_eq_implies_x_gt_y_l796_796701


namespace exists_basis_of_order_2_bounded_by_sqrt_l796_796449

-- Define parameters and properties
def is_basis_of_order_2 (A : Set ℕ) : Prop :=
∀ n ≥ 1, ∃ a b ∈ A, n = a + b

def bounded_by_sqrt (A : Set ℕ) (C : ℝ) : Prop :=
∀ x ≥ 1, (A ∩ {n | n ≤ x}).to_finset.card ≤ C * real.sqrt x

-- Our main theorem statement
theorem exists_basis_of_order_2_bounded_by_sqrt :
  ∃ (A : Set ℕ) (C : ℝ), A.nonempty ∧ is_basis_of_order_2 A ∧ C > 0 ∧ bounded_by_sqrt A C :=
sorry

end exists_basis_of_order_2_bounded_by_sqrt_l796_796449


namespace polynomial_degree_le_one_l796_796756

theorem polynomial_degree_le_one {P : ℝ → ℝ} (h : ∀ x : ℝ, 2 * P x = P (x + 3) + P (x - 3)) :
  ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x + b :=
sorry

end polynomial_degree_le_one_l796_796756


namespace mean_of_quadrilateral_angles_l796_796035

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796035


namespace correct_answers_is_70_l796_796819

variable {C I U : ℕ} -- C: correct answers, I: incorrect answers, U: unanswered questions

-- Given conditions
def condition1 : Prop := C + I + U = 82
def condition2 : Prop := U = 85 - 82
def condition3 : Prop := C - 0.25 * I = 67

-- Prove that the number of correct answers is 70
theorem correct_answers_is_70 :
  (C + I + U = 82) ∧ (U = 85 - 82) ∧ (C - 0.25 * I = 67) → C = 70 :=
by
  intro h,
  cases h with h1 h2,
  cases h2 with h2 h3,
  sorry  -- proof to be filled

end correct_answers_is_70_l796_796819


namespace xiao_bin_analysis_l796_796545

noncomputable def avg_points_team_A (points : List ℕ) : ℕ :=
  (points.sum) / points.length

def mode (nums : List ℕ) : ℕ :=
  let freq := nums.foldl (λ mp n, mp.insert n (mp.find! n + 1)) (RBMap.empty _ _)
  freq.foldl (λ m k v, if h : v > freq.find! m then k else m) 0

def median (nums : List ℕ) : ℕ :=
  let sorted := nums.qsort (λ a b, a < b)
  let n := sorted.length
  if n % 2 == 0 then (sorted.get (n/2 - 1) + sorted.get (n/2)) / 2
  else sorted.get (n/2)

def comprehensive_score (avg_points avg_rebounds avg_turnovers : ℚ) : ℚ :=
  avg_points * 1 + avg_rebounds * 1.2 - avg_turnovers * 1

theorem xiao_bin_analysis :
  (avg_points_team_A [21, 29, 24, 26] = 25) ∧
  (mode [10, 10, 10, 17, 15, 14, 12, 8] = 10) ∧
  (median [10, 10, 10, 17, 15, 14, 12, 8] = 11) ∧
  (comprehensive_score 25 11 2 = 36.2) ∧
  (37.1 > 36.2) :=
by
  sorry

end xiao_bin_analysis_l796_796545


namespace numberOfKidsInOtherClass_l796_796499

-- Defining the conditions as given in the problem
def kidsInSwansonClass := 25
def averageZitsSwansonClass := 5
def averageZitsOtherClass := 6
def additionalZitsInOtherClass := 67

-- Total number of zits in Ms. Swanson's class
def totalZitsSwansonClass := kidsInSwansonClass * averageZitsSwansonClass

-- Total number of zits in the other class
def totalZitsOtherClass := totalZitsSwansonClass + additionalZitsInOtherClass

-- Proof that the number of kids in the other class is 32
theorem numberOfKidsInOtherClass : 
  (totalZitsOtherClass / averageZitsOtherClass = 32) :=
by
  -- Proof is left as an exercise.
  sorry

end numberOfKidsInOtherClass_l796_796499


namespace modulus_of_euler_formula_l796_796647

theorem modulus_of_euler_formula : ∀ x : ℝ, 
  ∥exp (complex.I * x)∥ = 1 :=
by {
  sorry
}

end modulus_of_euler_formula_l796_796647


namespace shaded_area_limit_l796_796601

open Real

theorem shaded_area_limit (AB BC : ℝ) (h_AB : AB = 12) (h_BC : BC = 12) : 
  let Area₀ := (1/2) * AB * BC
  let a := (1/4) * Area₀
  let r := (1/4)
  S = a / (1 - r) :=
begin
  have Area₀_def : Area₀ = (1/2) * 12 * 12, 
  {
    rw [h_AB, h_BC],
    norm_num,
  },
  have Area₀_val : Area₀ = 72, by norm_num[Area₀_def],
  have a_def : a = (1/4) * 72, by rw [Area₀_val],
  have r_val : r = 1/4, by norm_num,
  have sum_val : S = 18 / (1 - 1/4), by rw [a_def, r_val],
  norm_num at sum_val,
  exact sum_val,
end

end shaded_area_limit_l796_796601


namespace mean_of_quadrilateral_angles_l796_796033

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796033


namespace length_AK_independent_of_D_l796_796456

theorem length_AK_independent_of_D
  (A B C D K : Point)
  (hD : D ∈ line_segment B C)
  (incircle_ABD incircle_ACD : Circle)
  (h1 : tangent_line incircle_ABD ∩ tangent_line incircle_ACD = K)
  (h2 : tangent_line incircle_ABD ∩ AD = K)
  (h3 : tangent_line incircle_ACD ∩ AD = K) :
  distance A K = (distance A B + distance A C - distance B C) / 2 := 
sorry

end length_AK_independent_of_D_l796_796456


namespace mean_value_of_quadrilateral_angles_l796_796081

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796081


namespace neg_p_equiv_l796_796749

def p : Prop := ∃ x₀ : ℝ, x₀^2 + 1 > 3 * x₀

theorem neg_p_equiv :
  ¬ p ↔ ∀ x : ℝ, x^2 + 1 ≤ 3 * x := by
  sorry

end neg_p_equiv_l796_796749


namespace profit_percentage_l796_796750

theorem profit_percentage (SP CP : ℝ) (h : CP = 0.98 * SP) : 
  ((SP - CP) / CP) * 100 ≈ 2.04 :=
by
  sorry

end profit_percentage_l796_796750


namespace length_of_hypotenuse_l796_796868

-- Definition of the right triangle with one angle of 45 degrees and radius of the inscribed circle being 4 cm
def right_triangle_with_inscribed_circle_radius_4 (r : ℝ) (θ : ℝ) (a b c : ℝ) : Prop :=
  (θ = 45) ∧ (r = 4) ∧ (a = b) ∧ (c = a * √2)

-- The theorem to prove the length of the hypotenuse
theorem length_of_hypotenuse (a b c : ℝ) (r : ℝ) (θ : ℝ) 
  (h : right_triangle_with_inscribed_circle_radius_4 r θ a b c) : c = 16 + 8 * √2 :=
by 
  sorry

end length_of_hypotenuse_l796_796868


namespace mean_value_of_quadrilateral_angles_l796_796089

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796089


namespace mean_value_interior_angles_quadrilateral_l796_796106

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796106


namespace distribution_denominations_counts_l796_796887

theorem distribution_denominations_counts :
  ∃ (c_2 c_5 c_{10} c_{20} c_{50} c_{100} c_{200} : ℕ),
    (c_2 + 1) * (c_5 + 1) * (c_{10} + 1) * (c_{20} + 1) * (c_{50} + 1) * (c_{100} + 1) * (c_{200} + 1) = 1512 ∧ 
    2 * c_2 + 5 * c_5 + 10 * c_{10} + 20 * c_{20} + 50 * c_{50} + 100 * c_{100} + 200 * c_{200} = 1512 ∧ 
    c_2 = 1 ∧ c_5 = 2 ∧ c_{10} = 1 ∧ c_{20} = 2 ∧ c_{50} = 1 ∧ c_{100} = 2 ∧ c_{200} = 6 := 
by
  use 1, 2, 1, 2, 1, 2, 6
  split
  exact Nat.eq_of_mul_eq_mul_left (by decide) 
    (by simp [Nat.mul_add, Nat.add_mul])
  split
  { norm_num }
  repeat { split; refl }
  sorry -- Proof omitted for brevity

end distribution_denominations_counts_l796_796887


namespace exists_k_ratio_2015_l796_796319

def P (n : ℕ) : ℕ :=
  if n = 0 then 0 else if n = 1 then 1 else (((nat.factor_multiset n).to_list).to_finset.prod id)

def sequence (a : ℕ) : ℕ → ℕ
| 0 := a
| (k+1) := sequence k + P (sequence k)

theorem exists_k_ratio_2015 (a₀ : ℕ) : ∃ k : ℕ, (sequence a₀ k) / (P (sequence a₀ k)) = 2015 :=
sorry

end exists_k_ratio_2015_l796_796319


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796064

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796064


namespace highest_probability_ksi_expected_value_ksi_equals_l796_796406

noncomputable def probability_ksi_equals (k : ℕ) : ℚ :=
  match k with
  | 2 => 9 / 64
  | 3 => 18 / 64
  | 4 => 21 / 64
  | 5 => 12 / 64
  | 6 => 4 / 64
  | _ => 0

noncomputable def expected_value_ksi : ℚ :=
  2 * (9 / 64) + 3 * (18 / 64) + 4 * (21 / 64) + 5 * (12 / 64) + 6 * (4 / 64)

theorem highest_probability_ksi :
  ∃ k : ℕ, (∀ m : ℕ, probability_ksi_equals k ≥ probability_ksi_equals m) ∧ k = 4 :=
by
  sorry

theorem expected_value_ksi_equals :
  expected_value_ksi = 15 / 4 :=
by
  sorry

end highest_probability_ksi_expected_value_ksi_equals_l796_796406


namespace mean_value_of_quadrilateral_angles_l796_796129

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796129


namespace profit_percentage_50_l796_796922

theorem profit_percentage_50 (SP : ℝ) (Profit : ℝ) (hSP : SP = 900) (hProfit : Profit = 300) : 
  (Profit / (SP - Profit)) * 100 = 50 :=
by
  rw [hSP, hProfit]
  norm_num
  sorry

end profit_percentage_50_l796_796922


namespace mean_of_quadrilateral_angles_is_90_l796_796096

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796096


namespace mean_of_quadrilateral_angles_is_90_l796_796095

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796095


namespace max_value_expression_l796_796455

open_locale big_operators

theorem max_value_expression (x y z : ℝ)
  (h : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z)
  (sum_eq : x + y + z = 3) :
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 27 / 8 :=
sorry

end max_value_expression_l796_796455


namespace union_A_B_l796_796695

noncomputable def A : Set ℝ := {x | x^2 + x - 6 = 0}
noncomputable def B : Set ℝ := {x | x^2 - 4 = 0}

theorem union_A_B : A ∪ B = {-3, -2, 2} := by
  sorry

end union_A_B_l796_796695


namespace factorize_m_cubed_minus_16m_l796_796653

theorem factorize_m_cubed_minus_16m (m : ℝ) : m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end factorize_m_cubed_minus_16m_l796_796653


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796068

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796068


namespace domain_f_parity_f_inequality_f_l796_796719

def f (x : ℝ) : ℝ := Real.log (x + 2) - Real.log (2 - x)

theorem domain_f : ∀ x : ℝ, f x ∈ ℝ → -2 < x ∧ x < 2 := 
by
  sorry

theorem parity_f : ∀ x : ℝ, x ∈ Ioo (-2 : ℝ) (2 : ℝ) → f (-x) = -f x := 
by 
  sorry

theorem inequality_f : ∀ x : ℝ, x ∈ Ioo (-2 : ℝ) (2 : ℝ) → (f x > 1 ↔ x > 18 / 11) := 
by
  sorry

end domain_f_parity_f_inequality_f_l796_796719


namespace gcd_of_7163_and_209_l796_796377

theorem gcd_of_7163_and_209 :
  greatest_common_divisor 7163 209 = 19 :=
by
  have h1 : 7163  = 209 * 34 + 57 := by sorry
  have h2 : 209 = 57 * 3 + 38 := by sorry
  have h3 : 57 = 38 * 1 + 19 := by sorry
  have h4 : 38 = 19 * 2 := by sorry
  exact sorry

end gcd_of_7163_and_209_l796_796377


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796010

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796010


namespace mean_of_quadrilateral_angles_is_90_l796_796101

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796101


namespace new_person_weight_is_55_l796_796845

variable (W : ℝ) -- Total weight of the original 8 people
variable (new_person_weight : ℝ) -- Weight of the new person
variable (avg_increase : ℝ := 2.5) -- The average weight increase

-- Given conditions
def condition (W new_person_weight : ℝ) : Prop :=
  new_person_weight = W + (8 * avg_increase) + 35 - W

-- The proof statement
theorem new_person_weight_is_55 (W : ℝ) : (new_person_weight = 55) :=
by
  sorry

end new_person_weight_is_55_l796_796845


namespace base_conversion_least_sum_l796_796864

theorem base_conversion_least_sum
    (c d : ℕ) (hc : c > 1) (hd : d > 1)
    (h1 : 3 * c + 8 = 8 * d + 3) :
    c + d = 13 :=
by
  have h : 3 * c - 8 * d = -5 := by linarith
  sorry

end base_conversion_least_sum_l796_796864


namespace mean_of_quadrilateral_angles_l796_796186

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796186


namespace exists_four_consecutive_with_square_divisors_l796_796675

theorem exists_four_consecutive_with_square_divisors :
  ∃ n : ℕ, n = 3624 ∧
  (∃ d1, d1^2 > 1 ∧ d1^2 ∣ n) ∧ 
  (∃ d2, d2^2 > 1 ∧ d2^2 ∣ (n + 1)) ∧ 
  (∃ d3, d3^2 > 1 ∧ d3^2 ∣ (n + 2)) ∧ 
  (∃ d4, d4^2 > 1 ∧ d4^2 ∣ (n + 3)) :=
sorry

end exists_four_consecutive_with_square_divisors_l796_796675


namespace sum_max_min_ratio_l796_796617

def ellipse_eq (x y : ℝ) : Prop :=
  5 * x^2 + x * y + 4 * y^2 - 15 * x - 24 * y + 56 = 0

theorem sum_max_min_ratio (p q : ℝ) (x y : ℝ) :
  (∀ x y : ℝ, ellipse_eq x y → y / x = p ∨ y / x = q) → 
  p + q = 31 / 34 :=
by
  sorry

end sum_max_min_ratio_l796_796617


namespace problem1_problem2_l796_796758

/-
Conditions and definitions:
1. In $\triangle ABC$, sides opposite angles $A$, $B$, and $C$ are $a$, $b$, and $c$ respectively.
2. Function $f(x) = 2 \cos x \sin (x - A) + \sin A$ reaches its maximum value at $x = \frac{5\pi}{12}$.
3. Specifically, $a = 7$ and $\sin B + \sin C = \frac{13\sqrt{3}}{14}$.
4. Validate range of $f(x)$ for $x \in (0, \frac{\pi}{2})$.
-/

noncomputable def range_f_x (A : ℝ) (x : ℝ) : set ℝ :=
  {f | ∃ (f : ℝ), f = 2 * cos x * sin (x - A) + sin A ∧ x ∈ Ioo 0 (π / 2)}

theorem problem1 (A : ℝ) : 
  (∃ x : ℝ, x = 5 * π / 12 ∧ f (x) = 2 * cos x * sin (x - A) + sin A) → 
  A = π / 3 → 
  range_f_x A x = set.Ioc (-sqrt 3 / 2) 1 :=
sorry

noncomputable def area_triangle (a b c A : ℝ) :=
  (1 / 2) * b * c * sin A

theorem problem2 (a b c A : ℝ) :
  a = 7 → 
  sin B + sin C = 13 * sqrt 3 / 14 → 
  A = π / 3 → 
  b + c = 13 → 
  (b * c = 40) → 
  area_triangle a b c A = 10 * sqrt 3 :=
sorry

end problem1_problem2_l796_796758


namespace imaginary_part_of_complex_division_l796_796858

-- Define the complex number division
def complex_division := (1 + 3 * Complex.i) / (1 - Complex.i)

-- State the theorem
theorem imaginary_part_of_complex_division : complex_division.im = 2 := 
sorry

end imaginary_part_of_complex_division_l796_796858


namespace find_length_l796_796666

variables (w h A l : ℕ)
variable (A_eq : A = 164)
variable (w_eq : w = 4)
variable (h_eq : h = 3)

theorem find_length : 2 * l * w + 2 * l * h + 2 * w * h = A → l = 10 :=
by
  intros H
  rw [w_eq, h_eq, A_eq] at H
  linarith

end find_length_l796_796666


namespace coin_same_side_probability_l796_796541

noncomputable def probability_same_side_5_tosses (p : ℚ) := (p ^ 5) + (p ^ 5)

theorem coin_same_side_probability : probability_same_side_5_tosses (1/2) = 1/16 := by
  sorry

end coin_same_side_probability_l796_796541


namespace hexagon_area_ratio_l796_796871

-- Define the hexagon and its properties
structure Hexagon :=
  (side : ℝ)

-- Define the ratio p : q
structure Ratio :=
  (p : ℝ)
  (q : ℝ)

-- Define the property that the sum of areas of inscribed hexagons is four times the original hexagon
def area_condition (hex : Hexagon) (r : Ratio) : Prop :=
  let a := hex.side in
  let p := r.p in
  let q := r.q in
  (p ≠ 0) ∧ (q ≠ 0) ∧
  (
    3 * sqrt 3 / 2 * a^2 * ((p + q)^2 / (p * q)) = 4 * 3 * sqrt 3 / 2 * a^2
  )
  
-- The proof statement
theorem hexagon_area_ratio (hex : Hexagon) (r : Ratio) : area_condition hex r ↔ r.p = r.q :=
  sorry

end hexagon_area_ratio_l796_796871


namespace mean_value_of_quadrilateral_angles_l796_796156

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796156


namespace map_scale_to_yards_l796_796584

theorem map_scale_to_yards :
  (6.25 * 500) / 3 = 1041 + 2 / 3 := 
by sorry

end map_scale_to_yards_l796_796584


namespace probability_point_within_two_units_of_origin_l796_796586

theorem probability_point_within_two_units_of_origin :
  ∀ (P : ℝ × ℝ), 
    (P.fst ≥ -3 ∧ P.fst ≤ 3 ∧ P.snd ≥ -3 ∧ P.snd ≤ 3) → 
    (∀ (p : ℝ × ℝ), p.fst^2 + p.snd^2 ≤ 4) → 
    (real.pi / 9) :=
begin
  sorry
end

end probability_point_within_two_units_of_origin_l796_796586


namespace mean_value_of_quadrilateral_angles_l796_796157

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796157


namespace problem_solution_l796_796918

def planned_production_week : ℕ := 2100
def daily_average : ℕ := 300
def deviations : List ℤ := [5, -2, -5, 15, -10, 16, -9]
def daily_productions := deviations.map (λ d, daily_average + d)
def sum_deviations : ℤ := deviations.sum
def total_production : ℕ := (daily_average * 7) + sum_deviations.to_nat
def highest_production : ℕ := daily_productions.foldl max 0
def lowest_production : ℕ := daily_productions.foldl min daily_average
def base_wage_per_unit : ℤ := 60
def bonus_per_excess_unit : ℤ := 50
def penalty_per_deficit_unit : ℤ := 80
def total_wage : ℤ := (base_wage_per_unit * total_production) + 
                      if sum_deviations > 0 then bonus_per_excess_unit * sum_deviations
                      else penalty_per_deficit_unit * sum_deviations


theorem problem_solution :
  (daily_productions.head! = (305 : ℕ)) ∧
  ((highest_production - lowest_production) = 26) ∧
  (total_production = 2110) ∧
  (total_wage = 127100) :=
by
  -- Please provide the proof steps here in Lean 4
  sorry

end problem_solution_l796_796918


namespace total_distance_covered_l796_796575

theorem total_distance_covered (up_speed down_speed up_time down_time : ℕ) (H1 : up_speed = 30) (H2 : down_speed = 50) (H3 : up_time = 5) (H4 : down_time = 5) :
  (up_speed * up_time + down_speed * down_time) = 400 := 
by
  sorry

end total_distance_covered_l796_796575


namespace twice_joan_more_than_karl_l796_796782

-- Define the conditions
def J : ℕ := 158
def total : ℕ := 400
def K : ℕ := total - J

-- Define the theorem to be proven
theorem twice_joan_more_than_karl :
  2 * J - K = 74 := by
    -- Skip the proof steps using 'sorry'
    sorry

end twice_joan_more_than_karl_l796_796782


namespace mean_value_of_quadrilateral_angles_l796_796154

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796154


namespace sum_of_fourth_powers_l796_796628

-- Define the sum of fourth powers as per the given formula
noncomputable def sum_fourth_powers (n: ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1) * (3 * n^2 + 3 * n - 1)) / 30

-- Define the statement to be proved
theorem sum_of_fourth_powers :
  2 * sum_fourth_powers 100 = 41006666600 :=
by sorry

end sum_of_fourth_powers_l796_796628


namespace flavored_drink_ratio_l796_796762

theorem flavored_drink_ratio :
  ∃ (F C W: ℚ), F / C = 1 / 7.5 ∧ F / W = 1 / 56.25 ∧ C/W = 6/90 ∧ F / C / 3 = ((F / W) * 2)
:= sorry

end flavored_drink_ratio_l796_796762


namespace mean_value_of_quadrilateral_angles_l796_796124

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796124


namespace planes_intersect_single_line_l796_796608

noncomputable theory
open_locale real

-- Given a sphere inscribed in a trihedral angle SABC, touching the faces at points A₁, B₁, and C₁ respectively,
-- prove that the planes SAA₁, SBB₁, and SCC₁ intersect at a single line.
theorem planes_intersect_single_line 
  (S A B C A₁ B₁ C₁ : Type*)
  [plane S A B C A₁ plane] [plane S B₁ C A plane] [plane S C₁ A B plane]
  (h_inscribed : inscribed_sphere S A B C A₁ B₁ C₁)
  (h_touch_SBC : touches_faces_at A₁ sphere_face_SBC)
  (h_touch_SCA : touches_faces_at B₁ sphere_face_SCA)
  (h_touch_SAB : touches_faces_at C₁ sphere_face_SAB) :
  planes_intersect (plane_SAA₁) (plane_SBB₁) (plane_SCC₁) :=
sorry

end planes_intersect_single_line_l796_796608


namespace conjugate_of_z_magnitude_of_z_l796_796687

def z : ℂ := 4 - 3 * Complex.I
def z_conjugate := Complex.conj(z)
def z_magnitude := Complex.abs(z)

theorem conjugate_of_z : z_conjugate = 4 + 3 * Complex.I :=
by 
  sorry

theorem magnitude_of_z : z_magnitude = 5 :=
by 
  sorry

end conjugate_of_z_magnitude_of_z_l796_796687


namespace mean_of_quadrilateral_angles_l796_796034

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796034


namespace percentile_85_l796_796606

theorem percentile_85 (data : List ℕ) (h : data = [16, 21, 23, 26, 33, 33, 37, 37]) : 
  data.nth (Int.toNat (Float.ceil (8 * 0.85) - 1)) = some 37 :=
by
  have data_positions : List (ℕ × ℕ) := data.zip ((List.range data.length).map (· + 1))  -- Positions along with data
  let p85_index := Int.toNat (Float.ceil (8 * 0.85) - 1) -- 7th position (index 6)
  rw [h]
  simp only
  norm_num
  sorry

end percentile_85_l796_796606


namespace trader_profit_l796_796611

theorem trader_profit (P : ℝ) (hP : 0 < P) :
  let buy_price := 0.80 * P
  let sell_price := 1.6000000000000001 * P
  let increase := sell_price - buy_price
  let percentage_increase := (increase / buy_price) * 100
  percentage_increase = 100 :=
by
  let buy_price := 0.80 * P
  let sell_price := 1.6000000000000001 * P
  let increase := sell_price - buy_price
  let percentage_increase := (increase / buy_price) * 100
  have : percentage_increase = 100,
    from sorry
  exact this

end trader_profit_l796_796611


namespace cube_inequality_of_greater_l796_796336

theorem cube_inequality_of_greater {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_of_greater_l796_796336


namespace max_min_values_of_f_measure_of_angle_C_value_of_cos_B_l796_796361

-- (I) Proof Problem
theorem max_min_values_of_f (ω : ℝ) (x : ℝ) (hω : ω > 0) (h_period : IsPeriod (λ x, sqrt 3 * sin (ω * x) - 2 * sin (ω * x / 2) ^ 2) (3 * π)) :
  (∀ x ∈ Icc (-π) (3 * π / 4), (sqrt 3 * sin (ω * x) - 2 * sin (ω * x / 2) ^ 2) ≤ 1) ∧
  (∀ x ∈ Icc (-π) (3 * π / 4), (sqrt 3 * sin (ω * x) - 2 * sin (ω * x / 2) ^ 2) ≥ -3) :=
sorry

-- (II) Proof Problem
theorem measure_of_angle_C (a b c A : ℝ) (h1 : a < b) (h2 : b < c) (h3 : sqrt 3 * a = 2 * c * sin A) :
  let C := 2 * π / 3 in 0 < C ∧ C < π :=
sorry

-- (III) Proof Problem
theorem value_of_cos_B (a b c A B C : ℝ) (h1 : a < b) (h2 : b < c) (h3 : sqrt 3 * a = 2 * c * sin A)
  (h4 : C = 2 * π / 3) (h5 : (sqrt 3 * sin(2 * π / 3 * A + π / 6) -1) = 11 / 13) :
  cos B = (12 + 5 * sqrt 3) / 26 :=
sorry

end max_min_values_of_f_measure_of_angle_C_value_of_cos_B_l796_796361


namespace reflections_concurrent_l796_796440

theorem reflections_concurrent
    (A B C O A' B' C' : Point) 
    (h_A' : A' = reflection O (internal_angle_bisector A (B, C)))
    (h_B' : B' = reflection O (internal_angle_bisector B (A, C)))
    (h_C' : C' = reflection O (internal_angle_bisector C (A, B))) :
    concurrent (line A A') (line B B') (line C C') := 
sorry

end reflections_concurrent_l796_796440


namespace parabola_standard_equation_l796_796712

variable {a : ℝ} (h : a < 0)

theorem parabola_standard_equation (h : a < 0) :
  ∃ (p : ℝ), p = -2 * a ∧ (∀ (x y : ℝ), y^2 = -2 * p * x ↔ y^2 = 4 * a * x) :=
sorry

end parabola_standard_equation_l796_796712


namespace mean_value_interior_angles_quadrilateral_l796_796111

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796111


namespace geometric_figure_perimeter_l796_796960

theorem geometric_figure_perimeter (A : ℝ) (n : ℝ) (area : ℝ) (side_length : ℝ) (perimeter : ℝ) : 
  A = 216 ∧ n = 6 ∧ area = A / n ∧ side_length = Real.sqrt area ∧ perimeter = 2 * (3 * side_length + 2 * side_length) + 2 * side_length →
  perimeter = 72 := 
by 
  sorry

end geometric_figure_perimeter_l796_796960


namespace polynomial_remainder_l796_796662

theorem polynomial_remainder :
  let f := X^2023 + 1
  let g := X^6 - X^4 + X^2 - 1
  ∃ (r : Polynomial ℤ), (r = -X^3 + 1) ∧ (∃ q : Polynomial ℤ, f = q * g + r) :=
by
  sorry

end polynomial_remainder_l796_796662


namespace fractional_equation_no_solution_l796_796753

theorem fractional_equation_no_solution (a : ℝ) :
  (¬ ∃ x, x ≠ 1 ∧ x ≠ 0 ∧ ((x - a) / (x - 1) - 3 / x = 1)) → (a = 1 ∨ a = -2) :=
by
  sorry

end fractional_equation_no_solution_l796_796753


namespace mean_value_of_quadrilateral_angles_l796_796216

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796216


namespace find_last_number_l796_796878

-- All definitions come from the conditions, no solution steps used here.
variables (S n x : ℕ)

-- Use noncomputable theory to avoid issues with real number division operations in Lean
noncomputable theory

-- Assuming initial conditions
def condition1 := S = 30 * n
def condition2 := S + 100 = 40 * (n + 1)
def condition3 := S + 100 + x = 50 * (n + 2)

-- The main statement we want to prove
theorem find_last_number (S n x : ℕ) : 
  condition1 S n ∧ condition2 S n ∧ condition3 S n x → x = 120 :=
by 
  sorry

end find_last_number_l796_796878


namespace triangle_external_properties_l796_796772

open EuclideanGeometry

variables (A B C P Q R : Point)
variables (α β γ : ℝ)
variables [hα : α = 45] [hβ : β = 30] [hγ : γ = 15]

def PBC_external : Prop := ∠ P B C = α
def CAQ_external : Prop := ∠ C A Q = α
def BCP_external : Prop := ∠ B C P = β
def QCA_external : Prop := ∠ Q C A = β
def ABR_external : Prop := ∠ A B R = γ
def BAR_external : Prop := ∠ B A R = γ

theorem triangle_external_properties :
  PBC_external A B C P α → 
  CAQ_external A C Q α → 
  BCP_external B C P β → 
  QCA_external C Q A β → 
  ABR_external A B R γ → 
  BAR_external B A R γ → 
  (∠ Q R P = 90) ∧ (dist Q R = dist R P) :=
by
  sorry

end triangle_external_properties_l796_796772


namespace value_of_a_l796_796709

theorem value_of_a (a x : ℝ) (h : (3 * x^2 + 2 * a * x = 0) → (x^3 + a * x^2 - (4 / 3) * a = 0)) :
  a = 0 ∨ a = 3 ∨ a = -3 :=
by
  sorry

end value_of_a_l796_796709


namespace laundry_loads_needed_l796_796243

theorem laundry_loads_needed
  (families : ℕ) (people_per_family : ℕ)
  (towels_per_person_per_day : ℕ) (days : ℕ)
  (washing_machine_capacity : ℕ)
  (h_f : families = 7)
  (h_p : people_per_family = 6)
  (h_t : towels_per_person_per_day = 2)
  (h_d : days = 10)
  (h_w : washing_machine_capacity = 10) : 
  ((families * people_per_family * towels_per_person_per_day * days) / washing_machine_capacity) = 84 := 
by
  sorry

end laundry_loads_needed_l796_796243


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796017

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796017


namespace candies_left_is_correct_l796_796974

-- Define the number of candies bought on different days
def candiesBoughtTuesday : ℕ := 3
def candiesBoughtThursday : ℕ := 5
def candiesBoughtFriday : ℕ := 2

-- Define the number of candies eaten
def candiesEaten : ℕ := 6

-- Define the total candies left
def candiesLeft : ℕ := (candiesBoughtTuesday + candiesBoughtThursday + candiesBoughtFriday) - candiesEaten

theorem candies_left_is_correct : candiesLeft = 4 := by
  -- Placeholder proof: replace 'sorry' with the actual proof when necessary
  sorry

end candies_left_is_correct_l796_796974


namespace mean_value_of_quadrilateral_angles_l796_796003

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l796_796003


namespace length_minor_axis_l796_796752

theorem length_minor_axis (k : ℝ) (h1 : (∀ x y : ℝ, x^2 + k * y^2 = 2 → (k > 0) ∧ (x^2 + k * y^2 = 2)))
  (h2 : ∃ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1))
  (h3 : ∀ a b c : ℝ, (c^2 = a^2 - b^2) → c = √3) :
  2 * √((2 / k)) = sqrt 5 :=
by
  sorry

end length_minor_axis_l796_796752


namespace second_train_length_l796_796530

noncomputable def length_of_second_train (speed1_kmph speed2_kmph time_sec length1_m : ℝ) : ℝ :=
  let relative_speed_mps := (speed1_kmph + speed2_kmph) * (1000 / 3600)
  let total_distance := relative_speed_mps * time_sec
  total_distance - length1_m

theorem second_train_length :
  length_of_second_train 60 48 9.99920006399488 140 = 159.9760019198464 :=
by
  sorry

end second_train_length_l796_796530


namespace mean_of_quadrilateral_angles_l796_796152

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796152


namespace attendance_proof_l796_796470

noncomputable def next_year_attendance (this_year: ℕ) := 2 * this_year
noncomputable def last_year_attendance (next_year: ℕ) := next_year - 200
noncomputable def total_attendance (last_year this_year next_year: ℕ) := last_year + this_year + next_year

theorem attendance_proof (this_year: ℕ) (h1: this_year = 600):
    total_attendance (last_year_attendance (next_year_attendance this_year)) this_year (next_year_attendance this_year) = 2800 :=
by
  sorry

end attendance_proof_l796_796470


namespace solution_set_eq_l796_796663

noncomputable def f (x : ℝ) : ℝ := x^6 + x^2
noncomputable def g (x : ℝ) : ℝ := (2*x + 3)^3 + 2*x + 3

theorem solution_set_eq : {x : ℝ | f x = g x} = {-1, 3} :=
by
  sorry

end solution_set_eq_l796_796663


namespace distinct_initial_terms_l796_796924

def collatz_step (n : ℕ) : ℕ :=
if n % 2 = 0 then n / 2 else 3 * n + 1

def collatz_sequence (n : ℕ) : ℕ → ℕ
| 0     := n
| (k+1) := collatz_step (collatz_sequence k)

lemma eighth_term_is_one (n : ℕ) (h : collatz_sequence n 8 = 1) : 
  n ∈ {2, 3, 16, 21, 32, 128} :=
sorry

theorem distinct_initial_terms :
  {n : ℕ | collatz_sequence n 8 = 1}.to_finset.card = 6 :=
sorry

end distinct_initial_terms_l796_796924


namespace R_depends_on_a_d_m_l796_796447

theorem R_depends_on_a_d_m (a d m : ℝ) :
    let s1 := (m / 2) * (2 * a + (m - 1) * d)
    let s2 := m * (2 * a + (2 * m - 1) * d)
    let s3 := 2 * m * (2 * a + (4 * m - 1) * d)
    let R := s3 - 2 * s2 + s1
    R = m * (a + 12 * m * d - (d / 2)) := by
  sorry

end R_depends_on_a_d_m_l796_796447


namespace mean_value_of_quadrilateral_angles_l796_796045

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796045


namespace area_inequality_l796_796789

variable {n : ℝ}
variable {AB AE CD DF AD BC : ℝ}
variable {S : ℝ}
variable {E F : Point}
variable {ABCD AEFD : Quadrilateral}

-- Given Conditions
axiom convex_quadrilateral: convex_quadrilateral ABCD
axiom points_on_sides: E ∈ line_segment AB ∧ F ∈ line_segment CD
axiom ratio_condition: AB / AE = n ∧ CD / DF = n

-- Prove Statement
theorem area_inequality: S ≤ (AB * CD + n * (n - 1) * AD^2 + n^2 * DA * BC) / (2 * n^2) :=
sorry

end area_inequality_l796_796789


namespace equilateral_triangle_b_l796_796380

noncomputable def parabola := { C : ℝ × ℝ // C.2^2 = 4 * C.1 }

theorem equilateral_triangle_b (b : ℝ) :
  (∃ (C : parabola), let ⟨(x, y), hx⟩ := C in 
    ∃ (A B : ℝ × ℝ), A = (1, 0) ∧ B = (b, 0) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - x)^2 + (A.2 - y)^2) ∧
    ((B.1 - x)^2 + (B.2 - y)^2 = (A.1 - x)^2 + (A.2 - y)^2)) ↔
    (b = 5 ∨ b = -1/3) :=
begin
  sorry
end

end equilateral_triangle_b_l796_796380


namespace power_mod_2040_l796_796219

theorem power_mod_2040 : (6^2040) % 13 = 1 := by
  -- Skipping the proof as the problem only requires the statement
  sorry

end power_mod_2040_l796_796219


namespace csc_315_equals_neg_sqrt_2_l796_796986

-- Define the conditions
def csc (x : Real) : Real := 1 / Real.sin x
axiom sin_subtraction_360 (x : Real) : Real.sin (360 - x) = -Real.sin x

-- State the problem
theorem csc_315_equals_neg_sqrt_2 : csc 315 = -Real.sqrt 2 := by
  sorry

end csc_315_equals_neg_sqrt_2_l796_796986


namespace calculate_neg4_mul_three_div_two_l796_796630

theorem calculate_neg4_mul_three_div_two : (-4) * (3 / 2) = -6 := 
by
  sorry

end calculate_neg4_mul_three_div_two_l796_796630


namespace perimeter_of_triangle_l796_796344

noncomputable def point_on_ellipse (P : ℝ × ℝ) : Prop := 
  (P.1^2 / 9) + (P.2^2 / 5) = 1

noncomputable def foci_position (F1 F2 : ℝ × ℝ) : Prop := 
  F1 = (-2, 0) ∧ F2 = (2, 0)

theorem perimeter_of_triangle :
  ∀ (P F1 F2 : ℝ × ℝ),
    point_on_ellipse P →
    foci_position F1 F2 →
    dist P F1 + dist P F2 + dist F1 F2 = 10 :=
by
  sorry

end perimeter_of_triangle_l796_796344


namespace mean_value_of_quadrilateral_angles_l796_796054

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796054


namespace mean_value_interior_angles_quadrilateral_l796_796113

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796113


namespace domino_cover_grid_l796_796797

-- Definitions representing the conditions:
def isPositive (n : ℕ) : Prop := n > 0
def divides (a b : ℕ) : Prop := ∃ k, b = k * a
def canCoverWithDominos (n k : ℕ) : Prop := ∀ i j, (i < n) → (j < n) → (∃ r, i = r * k ∨ j = r * k)

-- The hypothesis: n and k are positive integers
axiom n : ℕ
axiom k : ℕ
axiom n_positive : isPositive n
axiom k_positive : isPositive k

-- The main theorem
theorem domino_cover_grid (n k : ℕ) (n_positive : isPositive n) (k_positive : isPositive k) :
  canCoverWithDominos n k ↔ divides k n := by
  sorry

end domino_cover_grid_l796_796797


namespace circle_tangent_line_l796_796920

noncomputable def line_eq (x : ℝ) : ℝ := 2 * x + 1
noncomputable def circle_eq (x y b : ℝ) : ℝ := x^2 + (y - b)^2

theorem circle_tangent_line 
  (b : ℝ) 
  (tangency : ∃ b, (1 - b) / (0 - 1) = -(1 / 2)) 
  (center_point : 1^2 + (3 - b)^2 = 5 / 4) : 
  circle_eq 1 3 b = circle_eq 0 b (7/2) :=
sorry

end circle_tangent_line_l796_796920


namespace sum_of_inscribed_circle_areas_eq_area_of_circle_l796_796435

theorem sum_of_inscribed_circle_areas_eq_area_of_circle 
  (r : ℝ) (r0 : ℝ) (α : ℝ) (h1 : 0 < r) (h2 : 0 < r0)
  (h3 : 0 < α ∧ α < π/2)
  (h_eq : r = (1/2) * r0 * (Real.sqrt (Real.sin α) + Real.sqrt (Real.csc α))) : 
  (∑ i, (π * (r0 * ((1 - Real.sin α) / (1 + Real.sin α))^i)^2)) = π * r^2 :=
begin
  sorry -- Proof is omitted as per the instructions
end

end sum_of_inscribed_circle_areas_eq_area_of_circle_l796_796435


namespace find_angle_C_max_perimeter_triangle_ABC_l796_796426

noncomputable def triangle_sides (a b c: ℝ) (A B C: ℝ) :=
  a = c * cos(C) / 2 + b * cos(A) / 2

theorem find_angle_C (a b c A B C: ℝ)
  (h1: triangle_sides a b c A B C)
  : C = π / 3 :=
sorry

theorem max_perimeter_triangle_ABC (a b c: ℝ)
  (h_c: c = 2)
  : ∃ a b, a + b + c = 6 ∧ a + b ≤ 4 ∧ a = 2 ∧ b = 2 :=
sorry

end find_angle_C_max_perimeter_triangle_ABC_l796_796426


namespace tangent_line_equation_is_correct_l796_796853

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_equation_is_correct :
  let p : ℝ × ℝ := (0, 1)
  let f' := fun x => x * Real.exp x + Real.exp x
  let slope := f' 0
  let tangent_line := fun x y => slope * (x - p.1) - (y - p.2)
  tangent_line = (fun x y => x - y + 1) :=
by
  intros
  sorry

end tangent_line_equation_is_correct_l796_796853


namespace union_of_P_complementQ_eq_set_a_l796_796378

variable (U : Set ℕ) (P : Set ℕ) (Q : Set ℕ)

-- Define the specific sets: U, P, Q
def U := {1, 2, 3, 4, 5, 6}
def P := {1, 2, 4}
def Q := {2, 3, 4, 6}

-- Define the complement of Q with respect to U
def complementQ : Set ℕ := U \ Q

-- The actual proof statement
theorem union_of_P_complementQ_eq_set_a :
  P ∪ complementQ = {1, 2, 4, 5} :=
  sorry

end union_of_P_complementQ_eq_set_a_l796_796378


namespace convex_polygon_contains_segment_l796_796516

variables {a t : ℝ} (P : Set (ℝ × ℝ)) (hconvex : Convex ℝ P) (hsquare : ∃ Q : Finset (ℝ × ℝ), (Q.card = 4) ∧ ∀ (v ∈ Q), (v ∈ Boundary P) ∧ Square (a) (Q)) (harea : t = area P)

theorem convex_polygon_contains_segment (hconvex : Convex ℝ P) (hsquare : ∃ Q : Finset (ℝ × ℝ), (Q.card = 4) ∧ ∀ (v ∈ Q), (v ∈ Boundary P) ∧ Square (a) (Q)) (harea : t = area P):
  ∃ (x y : (ℝ × ℝ)), (x ∈ P ∧ y ∈ P) ∧ dist x y ≥ t / a := 
sorry

end convex_polygon_contains_segment_l796_796516


namespace mean_of_quadrilateral_angles_is_90_l796_796100

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796100


namespace f_neg_f_f_periodic_f_value_at_neg_5_over_2_l796_796688

def f : ℝ → ℝ 
  := λ x, if (0 ≤ x ∧ x ≤ 1) then 2 * x * (1 - x) else 0 -- Define f piecewise

theorem f_neg_f (x : ℝ) : f (-x) = -f (x) := by
  -- Add some assumptions, sorry to skip the proof
  sorry

theorem f_periodic (x : ℝ) : f (x + 2) = f (x) := by
  -- Add some assumptions, sorry to skip the proof
  sorry

theorem f_value_at_neg_5_over_2 : f (-5/2) = -1/2 := by
  have periodic : f (-5/2) = f (-5/2 + 2),
    from f_periodic _,                 -- Use periodic condition
  have reflective : f (-1/2) = -f (1/2),
    from f_neg_f _,                    -- Use odd function property
  have interval : f (1/2) = 2 * (1/2) * (1 - (1/2)),
    from if_pos (and.intro (by norm_num) (by norm_num)), -- Use given function when 0 ≤ x ≤ 1
  have simplify : 2 * (1/2) * (1 - (1/2)) = 1/2,
    by norm_num,                       -- Simplify the math
  rw interval at reflective,
  rw simplify at reflective,
  rw periodic,
  exact reflective

end f_neg_f_f_periodic_f_value_at_neg_5_over_2_l796_796688


namespace mean_of_quadrilateral_angles_l796_796144

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796144


namespace perfect_square_of_integer_expression_l796_796487

theorem perfect_square_of_integer_expression (n : ℤ) (k : ℤ) 
  (h : 2 + 2 * real.sqrt (1 + 12 * n ^ 2) = k) : ∃ m : ℤ, k = m ^ 2 :=
by 
  sorry

end perfect_square_of_integer_expression_l796_796487


namespace inverse_function_property_l796_796397

theorem inverse_function_property {α β : Type*} [hα : Nonempty α] [hβ : Nonempty β] {f : α → β} {g : β → α} (h₁ : Function.LeftInverse g f) (h₂ : Function.RightInverse g f) {a : α} {b : β} (h3 : f a = b) (h₄ : a ≠ 0) (h₅ : b ≠ 0) : g b = a :=
by
  sorry

end inverse_function_property_l796_796397


namespace csc_315_eq_neg_sqrt_two_l796_796993

theorem csc_315_eq_neg_sqrt_two : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_two_l796_796993


namespace range_of_m_satisfying_obtuse_triangle_l796_796700

theorem range_of_m_satisfying_obtuse_triangle (m : ℝ) 
(h_triangle: m > 0 
  → m + (m + 1) > (m + 2) 
  ∧ m + (m + 2) > (m + 1) 
  ∧ (m + 1) + (m + 2) > m
  ∧ (m + 2) ^ 2 > m ^ 2 + (m + 1) ^ 2) : 1 < m ∧ m < 1.5 :=
by
  sorry

end range_of_m_satisfying_obtuse_triangle_l796_796700


namespace csc_315_eq_neg_sqrt_2_l796_796999

theorem csc_315_eq_neg_sqrt_2 :
  Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := 
by
  sorry

end csc_315_eq_neg_sqrt_2_l796_796999


namespace each_persons_share_l796_796230

theorem each_persons_share 
  (total_bill : ℝ) (num_people : ℝ) (tip_percent : ℝ) 
  (h_total_bill : total_bill = 211.00) (h_num_people : num_people = 9) (h_tip_percent : tip_percent = 0.15) :
  (total_bill * (1 + tip_percent)) / num_people ≈ 26.96 :=
by
  sorry

end each_persons_share_l796_796230


namespace divisibility_of_poly_l796_796726

theorem divisibility_of_poly (P Q R S : Polynomial ℤ) (h: P.eval (x^5) + x * Q.eval (x^5) + x^2 * R.eval (x^5) = (x^4 + x^3 + x^2 + x + 1) * S.eval x) :
  (x + 1) ∣ P :=
sorry

end divisibility_of_poly_l796_796726


namespace slices_needed_l796_796830

def slices_per_sandwich : ℕ := 3
def number_of_sandwiches : ℕ := 5

theorem slices_needed : slices_per_sandwich * number_of_sandwiches = 15 :=
by {
  sorry
}

end slices_needed_l796_796830


namespace problem_solution_l796_796419

open Real

-- The parametric equations of the circle in Cartesian coordinates
def x (φ : ℝ) : ℝ := 1 + cos φ
def y (φ : ℝ) : ℝ := sin φ

-- The Cartesian equation of the circle
def cartesian_circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- The polar form of the circle
def polar_circle_eq (ρ θ : ℝ) : Prop := ρ = 2 * cos θ

-- Given special angle
def special_angle : ℝ := π / 3

-- The intersection points of ray OM with circle C and line l
def ρ_P : ℝ := 1
def θ_P : ℝ := special_angle

def line_l_eq (ρ θ : ℝ) : Prop := ρ * (sin θ + cos θ) = 3 * sqrt 3

def ρ_Q : ℝ := 3
def θ_Q : ℝ := special_angle

-- The length of segment PQ
def length_PQ (ρ_P ρ_Q : ℝ) : ℝ := abs (ρ_P - ρ_Q)

theorem problem_solution :
    (polar_circle_eq ρ_P θ_P) →
    (θ_P = special_angle) →
    (line_l_eq ρ_Q θ_Q) →
    (θ_Q = special_angle) →
    length_PQ ρ_P ρ_Q = 2 :=
by
  intros h1 h2 l1 l2
  rw [polar_circle_eq, line_l_eq, length_PQ]
  simp [h1, h2, l1, l2]
  sorry

end problem_solution_l796_796419


namespace students_play_both_football_and_cricket_l796_796821

def total_students : ℕ := 420
def football_players : ℕ := 325
def cricket_players : ℕ := 175
def neither_players : ℕ := 50

theorem students_play_both_football_and_cricket : 
  (420 - neither_players) = (football_players + cricket_players - (500 - 370)) ↔ 130 :=
by sorry

end students_play_both_football_and_cricket_l796_796821


namespace calculate_total_shaded_area_l796_796423

theorem calculate_total_shaded_area
(smaller_square_side larger_square_side smaller_circle_radius larger_circle_radius : ℝ)
(h1 : smaller_square_side = 6)
(h2 : larger_square_side = 12)
(h3 : smaller_circle_radius = 3)
(h4 : larger_circle_radius = 6) :
  (smaller_square_side^2 - π * smaller_circle_radius^2) + 
  (larger_square_side^2 - π * larger_circle_radius^2) = 180 - 45 * π :=
by
  sorry

end calculate_total_shaded_area_l796_796423


namespace watermelon_slices_l796_796885

theorem watermelon_slices (total_seeds slices_black seeds_white seeds_per_slice num_slices : ℕ)
  (h1 : seeds_black = 20)
  (h2 : seeds_white = 20)
  (h3 : seeds_per_slice = seeds_black + seeds_white)
  (h4 : total_seeds = 1600)
  (h5 : num_slices = total_seeds / seeds_per_slice) :
  num_slices = 40 :=
by
  sorry

end watermelon_slices_l796_796885


namespace mean_of_quadrilateral_angles_is_90_l796_796094

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796094


namespace eunjeong_cases_count_l796_796221

theorem eunjeong_cases_count :
  let students := ["Hyunji", "Eunjeong", "Migeum", "Youngmi"]
  in (∃ perm : List String, perm = ["Eunjeong", "Hyunji", "Migeum", "Youngmi"]) ∨
     (∃ perm : List String, perm = ["Hyunji", "Migeum", "Youngmi", "Eunjeong"]) →
     ∃ n : Nat, n = 12 :=
by
  sorry

end eunjeong_cases_count_l796_796221


namespace mean_value_of_quadrilateral_angles_l796_796163

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796163


namespace basketball_team_min_score_l796_796568

theorem basketball_team_min_score (n : ℕ) (min_score : ℕ) (max_score : ℕ) (players : ℕ) :
  players = 12 → min_score = 7 → max_score = 23 →
  (∀ i, 1 ≤ i ∧ i ≤ players → min_score ≤ (score i) ∧ (score i) ≤ max_score) →
  (∑ i in finset.range players, score i) ≥ 100 :=
by
  intro h_players h_min_score h_max_score h_scores
  sorry

end basketball_team_min_score_l796_796568


namespace mean_value_of_quadrilateral_angles_l796_796086

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796086


namespace mean_of_quadrilateral_angles_is_90_l796_796103

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796103


namespace mean_of_quadrilateral_angles_l796_796195

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796195


namespace rational_terms_in_expansion_l796_796704

theorem rational_terms_in_expansion (n : ℕ) (h : (n^2 - 9 * n + 8 = 0 ∨ n = 8)) :
  (let expansion_term (r : ℕ) := Nat.choose 8 r * (1/2)^r * x^((16 - 3 * r) / 4) in
  ( ∃ k : ℕ, k ∈ {0, 4, 8} ∧ expansion_term k ∈ {x^4, (35/8)*x, 1/(256*x^2)} ))
  ∧ (let k : ℕ := max_at (λ k, Nat.choose 8 (k-1) * (1/2)^(k-1)) in
  k ∈ {3, 4} ∧ (Nat.choose 8 (k-1) * (1/2)^(k-1) == 7*x^(5/2) ∨ 7*x^(7/4))) :=
sorry

end rational_terms_in_expansion_l796_796704


namespace k_at_1_eq_one_point_one_two_five_l796_796444

noncomputable def h (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * x + 4

noncomputable def k (x : ℝ) : ℝ := let s := 1.0 in let t := 2.0 in let u := 3.0 in (x - s^2) * (x - t^2) * (x - u^2)

theorem k_at_1_eq_one_point_one_two_five : k 1 = 1.125 :=
by
  sorry

end k_at_1_eq_one_point_one_two_five_l796_796444


namespace find_n_times_s_l796_796796

-- Mathematical conditions as Lean definitions
variables (f : ℝ → ℝ)

-- Condition: ∀ x y z ∈ ℝ, f(x^2 + yf(z)) = xf(x) + zf(y) + y^2
def property (f : ℝ → ℝ) : Prop :=
∀ x y z : ℝ, f(x^2 + yf(z)) = xf(x) + zf(y) + y^2

-- Lean statement for the proof goal
theorem find_n_times_s (hf : property f) : 
    let n := 2 in
    let s := 7 in
    n * s = 14 :=
by
  -- This is where the proof would go, but for now we skip it
  sorry

end find_n_times_s_l796_796796


namespace mean_value_interior_angles_quadrilateral_l796_796120

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796120


namespace divides_floor_factorial_div_l796_796835

theorem divides_floor_factorial_div {m n : ℕ} (h1 : 1 < m) (h2 : m < n + 2) (h3 : 3 < n) :
  (m - 1) ∣ (n! / m) :=
sorry

end divides_floor_factorial_div_l796_796835


namespace calculate_expression_l796_796956

theorem calculate_expression : (2 * Real.sqrt 3 - Real.pi)^0 - abs (1 - Real.sqrt 3) + 3 * Real.tan (Real.pi / 6) + (-1 / 2)^(-2) = 6 :=
by
  sorry

end calculate_expression_l796_796956


namespace mean_value_of_quadrilateral_angles_l796_796083

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796083


namespace probability_correct_guessing_four_bins_l796_796249

theorem probability_correct_guessing_four_bins :
  let bins := 4 in
  let arrangements := bins.factorial in
  (1 / arrangements) = 1 / 24 := by
  sorry

end probability_correct_guessing_four_bins_l796_796249


namespace triangle_distance_sum_equal_l796_796643

noncomputable def triangle_distances (A B C P Q : ℝ × ℝ) (a b c a1 b1 c1 : ℝ) : Prop :=
  let d := λ (X Y : ℝ × ℝ), abs ((P.1 - Q.1) * (X.2 - Y.2) - (P.2 - Q.2) * (X.1 - Y.1)) /
                             sqrt (((P.1 - Q.1)^2) + ((P.2 - Q.2)^2))
  in d A P = a ∧ d B P = b ∧ d C P = c ∧
     d ((A.1 + B.1) / 2, (A.2 + B.2) / 2) P = a1 ∧
     d ((B.1 + C.1) / 2, (B.2 + C.2) / 2) P = b1 ∧
     d ((C.1 + A.1) / 2, (C.2 + A.2) / 2) P = c1 ∧
     a + b + c = a1 + b1 + c1

theorem triangle_distance_sum_equal (A B C P Q : ℝ × ℝ) (a b c a1 b1 c1 : ℝ) :
  triangle_distances A B C P Q a b c a1 b1 c1 :=
by
  sorry

end triangle_distance_sum_equal_l796_796643


namespace problem_solution_l796_796856

theorem problem_solution :
  ∀ (f : ℝ → ℝ),
  (∀ (x : ℝ), f(x + 1) = (1 + f(x + 3)) / (1 - f(x + 3))) →
  (f 1 * f 2 * f 3 * ⋯ * f 2008 + 2009 = 2010) :=
by
  sorry

end problem_solution_l796_796856


namespace percentage_discount_is_10_l796_796594

-- Definition of the wholesale price
def wholesale_price : ℝ := 90

-- Definition of the retail price
def retail_price : ℝ := 120

-- Definition of the profit in terms of percentage of the wholesale price
def profit : ℝ := 0.2 * wholesale_price

-- Definition of the selling price based on the wholesale price and profit
def selling_price : ℝ := wholesale_price + profit

-- Definition of the discount amount
def discount_amount : ℝ := retail_price - selling_price

-- Definition of the percentage discount
def percentage_discount : ℝ := (discount_amount / retail_price) * 100

-- The theorem we want to prove: the percentage discount is 10%
theorem percentage_discount_is_10 : percentage_discount = 10 := 
by
  sorry

end percentage_discount_is_10_l796_796594


namespace directrix_of_parabola_l796_796494

theorem directrix_of_parabola (x y : ℝ) (h : y = x^2) : y = -1 / 4 :=
sorry

end directrix_of_parabola_l796_796494


namespace problem_translation_symmetry_l796_796540

noncomputable def f : ℝ → ℝ :=
  λ x, real.sin (2 * x + π / 3)

/-- The graph of the function y = sin (2x + π / 3) translated right by π / 12 units 
    is symmetric about the point (-π / 12, 0) 
--/
theorem problem_translation_symmetry :
  ∀ x : ℝ, f (x - π / 12) = f (-π / 6 - x) :=
sorry

end problem_translation_symmetry_l796_796540


namespace smallest_integer_l796_796220

-- Define a function to calculate the LCM of a list of numbers
def lcm_list (l : List ℕ) : ℕ :=
  l.foldl Nat.lcm 1

-- List of divisors
def divisors : List ℕ := [4, 5, 6, 7, 8, 9, 10]

-- Calculating the required integer
noncomputable def required_integer : ℕ := lcm_list divisors + 1

-- The proof statement
theorem smallest_integer : required_integer = 2521 :=
  by 
  sorry

end smallest_integer_l796_796220


namespace simplify_expression_l796_796841

variable (x : ℝ)

theorem simplify_expression : 
  (3 * x + 6 - 5 * x) / 3 = (-2 / 3) * x + 2 := by
  sorry

end simplify_expression_l796_796841


namespace exists_natural_2001_digits_l796_796781

theorem exists_natural_2001_digits (N : ℕ) (hN: N = 5 * 10^2000 + 1) : 
  ∃ K : ℕ, (K = N) ∧ (N^(2001) % 10^2001 = N % 10^2001) :=
by
  sorry

end exists_natural_2001_digits_l796_796781


namespace min_value_of_t_l796_796679

theorem min_value_of_t (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : 
  ∃ t : ℝ, t = 3 + 2 * Real.sqrt 2 ∧ t = 1 / a + 1 / b :=
sorry

end min_value_of_t_l796_796679


namespace arcsin_sufficient_not_necessary_l796_796557

theorem arcsin_sufficient_not_necessary (α : ℝ) (k : ℤ) :
  (∃ (k : ℤ), α = arcsin (1 / 3) + 2 * k * Real.pi ∨ α = Real.pi - arcsin (1/3) + 2 * k * Real.pi) ↔ 
  (α = arcsin (1 / 3) ∨ ∃ (k : ℤ), α = arcsin (1 / 3) + 2 * k * Real.pi ∨ α = Real.pi - arcsin (1/3) + 2 * k * Real.pi ∧ α ≠ arcsin (1 / 3)) :=
sorry

end arcsin_sufficient_not_necessary_l796_796557


namespace exists_rectangle_of_area_sqrt2_packing_squares_l796_796311

theorem exists_rectangle_of_area_sqrt2_packing_squares :
  ∀ (a b : ℝ), a^2 + b^2 = 1 → ∃ (A: ℝ), A = sqrt 2 ∧ ∃ (l w : ℝ), l * w = A ∧ a <= l ∧ b <= l ∧ a <= w ∧ b <= w :=
by
  intro a b h
  use sqrt 2
  constructor
  simp
  use [1, sqrt 2]
  constructor
  ring_nf
  simp
  sorry

end exists_rectangle_of_area_sqrt2_packing_squares_l796_796311


namespace mean_value_of_quadrilateral_angles_l796_796215

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796215


namespace problem_equivalence_l796_796283

theorem problem_equivalence : 4 * 4^3 - 16^60 / 16^57 = -3840 := by
  sorry

end problem_equivalence_l796_796283


namespace find_a_l796_796496

def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2 - 6 * a

theorem find_a (a b : ℝ) (h1 : deriv (λ x, f x a b) 2 = 0)
    (h2 : f 2 a b = 8) : a = -4 := 
by
  sorry

end find_a_l796_796496


namespace sales_tax_is_10_percent_l796_796578

-- Price of food before tax and tip
constant food_price : ℝ := 160

-- Total amount paid
constant total_paid : ℝ := 211.20

-- Tip percentage
constant tip_percentage : ℝ := 0.20

-- Define sales_tax_percentage 'x' and calculate total cost to prove it's correct
noncomputable def sales_tax_percentage (x : ℝ) : Prop :=
  let sales_tax := x / 100 * food_price
  let total_with_tax := food_price + sales_tax
  let total_with_tip := total_with_tax + (tip_percentage * total_with_tax)
  total_with_tip = total_paid

-- Prove sales tax percentage is 10%
theorem sales_tax_is_10_percent : sales_tax_percentage 10 :=
  by
  sorry

end sales_tax_is_10_percent_l796_796578


namespace count_intersection_points_eq_17_l796_796965

-- Definitions of the conditions
def sphere1 (x y z : ℤ) : Prop :=
  x^2 + y^2 + (z - 10)^2 ≤ 49

def sphere2 (x y z : ℤ) : Prop :=
  x^2 + y^2 + (z - 2)^2 ≤ 25

-- Definition of the intersection
def intersection (p : ℤ × ℤ × ℤ) : Prop :=
  let (x, y, z) := p in sphere1 x y z ∧ sphere2 x y z

-- Definition that counts the integer points in the intersection
def count_points_in_intersection : ℕ :=
  Finset.card (Finset.filter intersection
    (Finset.product (Finset.fin_range 15) (Finset.product 
    (Finset.fin_range 15) (Finset.fin_range 15))))

-- The proof statement
theorem count_intersection_points_eq_17 : count_points_in_intersection = 17 :=
  sorry

end count_intersection_points_eq_17_l796_796965


namespace triangle_ratios_l796_796806

theorem triangle_ratios 
  (A B C D : Point)
  (h1 : InTriangle D A B C)
  (h2 : ∠ A D B = ∠ A C B + 90)
  (h3 : AC * BD = AD * BC) :
  AB * CD / (AC * BD) = sqrt(2) := 
sorry

end triangle_ratios_l796_796806


namespace mean_of_quadrilateral_angles_l796_796147

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796147


namespace injective_and_monotonic_on_irrationals_implies_on_reals_l796_796436

theorem injective_and_monotonic_on_irrationals_implies_on_reals {f : ℝ → ℝ} (h_cont : Continuous f)
  (h_inj_irr : ∀ (x₁ x₂ : ℝ), (x₁ ∉ ℚ) → (x₂ ∉ ℚ) → (f x₁ = f x₂ → x₁ = x₂)) :
  (Injective f) ∧ (StrictMono f) := by
  sorry

end injective_and_monotonic_on_irrationals_implies_on_reals_l796_796436


namespace correct_cost_per_piece_l796_796474

-- Definitions for the given conditions
def totalPaid : ℝ := 20700
def reimbursement : ℝ := 600
def numberOfPieces : ℝ := 150
def correctTotal := totalPaid - reimbursement

-- Theorem stating the correct cost per piece of furniture
theorem correct_cost_per_piece : correctTotal / numberOfPieces = 134 := 
by
  sorry

end correct_cost_per_piece_l796_796474


namespace fraction_ordering_l796_796889

theorem fraction_ordering :
  (8 / 25 : ℚ) < 6 / 17 ∧ 6 / 17 < 10 / 27 ∧ 8 / 25 < 10 / 27 :=
by
  sorry

end fraction_ordering_l796_796889


namespace max_modulus_complex_number_l796_796686

theorem max_modulus_complex_number (z : ℂ) (h : complex.abs z = 1) : complex.abs (z + 2 * real.sqrt 2 + complex.I) ≤ 4 :=
sorry

end max_modulus_complex_number_l796_796686


namespace probability_two_heads_two_fair_coins_l796_796223

theorem probability_two_heads_two_fair_coins :
  let outcomes := {("Head", "Head"), ("Head", "Tail"), ("Tail", "Head"), ("Tail", "Tail")} in
  (finset.card {outcome ∈ outcomes | outcome = ("Head", "Head")} : ℚ) / finset.card outcomes = 1 / 4 :=
by
  sorry

end probability_two_heads_two_fair_coins_l796_796223


namespace geometry_problem_l796_796526

noncomputable def proof_problem (circle1 circle2 : ℝ) (M K A B C D : Type) 
(Intersec1 : M ∈ circle1)
(Intersec2 : K ∈ circle2)
(Line_AB : ∀ A B, A ≠ B → A ∉ B → AB ∈ circle1)
(Line_CD : ∀ C D, C ≠ D → C ∉ D → CD ∈ circle2)
(Intersec1M : ∀ A M, M ≠ K → M ∉ K → AM ∈ circle1)
(Intersec2M : ∀ C M, M ≠ K → M ∉ K → CM ∈ circle2)
(CD_MK : C ∈ M → D ∈ K) : Prop :=
AC ∥ BD

theorem geometry_problem {circle1 circle2 : ℝ} {M K A B C D : Type}
  (Intersec1 : M ∈ circle1)
  (Intersec2 : K ∈ circle2)
  (Line_AB : ∀ A B, A ≠ B → A ∉ B → AB ∈ circle1)
  (Line_CD : ∀ C D, C ≠ D → C ∉ D → CD ∈ circle2)
  (Intersec1M : ∀ A M, M ≠ K → M ∉ K → AM ∈ circle1)
  (Intersec2M : ∀ C M, M ≠ K → M ∉ K → CM ∈ circle2)
  (CD_MK : C ∈ M → D ∈ K) : proof_problem circle1 circle2 M K A B C D :=
  sorry

end geometry_problem_l796_796526


namespace mean_of_quadrilateral_angles_is_90_l796_796099

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796099


namespace proof_quadrilateral_AB_r_s_l796_796774

-- Definition of the problem's conditions
def quadrilateral (A B C D : Type) := True

-- Constants defining the conditions in the problem
constants (A B C D : Type)
constants (BC CD AD : ℝ)
constants (angle_A angle_B : ℝ)

-- Given conditions
axiom A1 : quadrilateral A B C D
axiom A2 : BC = 10
axiom A3 : CD = 15
axiom A4 : AD = 12
axiom A5 : angle_A = 60
axiom A6 : angle_B = 60

-- Definition of r and s
def r : ℝ := 21
def s : ℝ := 275

-- The statement to be proved in Lean
theorem proof_quadrilateral_AB_r_s :
  ∃ r s : ℝ, r = 21 ∧ s = 275 ∧ BC = 10 ∧ CD = 15 ∧ AD = 12 ∧ angle_A = 60 ∧ angle_B = 60 →
  AB = r + sqrt(s) :=
sorry

end proof_quadrilateral_AB_r_s_l796_796774


namespace max_marks_l796_796227

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 59 + 40) : M = 300 :=
by
  sorry

end max_marks_l796_796227


namespace mean_value_of_quadrilateral_angles_l796_796080

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796080


namespace mean_value_of_quadrilateral_angles_l796_796007

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l796_796007


namespace exists_p_q_l796_796234

noncomputable def A_k (a : ℕ → ℝ) (k m : ℕ) : ℝ :=
  (1 / (m : ℝ)) * (∑ i in Finset.range m, a (k + i + 1))

noncomputable def G_k (a : ℕ → ℝ) (k m : ℕ) : ℝ :=
  ((∏ i in Finset.range m, a (k + i + 1))^(1 / (m : ℝ)))

theorem exists_p_q (a : ℕ → ℝ) : ∃ (p q : ℕ), ∀ (s t : ℕ), G_k a p s ≤ A_k a q t :=
  sorry

end exists_p_q_l796_796234


namespace related_sequence_exists_l796_796238

theorem related_sequence_exists :
  ∃ b : Fin 5 → ℕ, b = ![11, 10, 9, 8, 7] :=
by
  let a : Fin 5 → ℕ := ![1, 5, 9, 13, 17]
  let b : Fin 5 → ℕ := ![
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 0) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 1) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 2) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 3) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 4) / 4
  ]
  existsi b
  sorry

end related_sequence_exists_l796_796238


namespace triangle_combinations_count_l796_796241

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def valid_combinations_count (sides : list ℕ) : ℕ :=
  (sides.combination 3).countp (λ t, is_triangle t.head! t.tail.head! t.tail.tail.head!)

theorem triangle_combinations_count :
  valid_combinations_count [13, 10, 5, 7] = 3 :=
by
  -- We assert the combinations and their validity
  have h1 : is_triangle 13 10 5 := 
    by
      simp [is_triangle]
      exact ⟨by norm_num, by norm_num, by norm_num⟩
  have h2 : is_triangle 13 10 7 :=
    by
      simp [is_triangle]
      exact ⟨by norm_num, by norm_num, by norm_num⟩
  have h3 : is_triangle 10 7 5 :=
    by
      simp [is_triangle]
      exact ⟨by norm_num, by norm_num, by norm_num⟩
  have not_h4 : ¬ is_triangle 13 7 5 :=
    by
      simp [is_triangle]
      exact not_and_of_not_left _ (by norm_num)
  sorry

end triangle_combinations_count_l796_796241


namespace number_of_correct_propositions_l796_796730

-- Definitions of lines, planes, and relations
variables (l m : Line) (α β : Plane)

-- Condition (1)
def condition1 := ∀ (l : Line) (α : Plane) (x y : Line),
  (x ≠ y) → (x ≠ l) → (y ≠ l) → (x ∈ α) → (y ∈ α) → (x ∩ y ≠ ∅) → (l ⟂ x) → (l ⟂ y) → (l ⟂ α)

-- Condition (2)
def condition2 := ∀ (l : Line) (α : Plane),
  (l ∥ α → False) → ∀ (x : Line), (x ∈ α) → (l ∥ x)

-- Condition (3)
def condition3 := ∀ (m : Line) (α : Plane) (l : Line) (β : Plane),
  (m ∈ α) → (l ∈ β) → (l ⟂ m) → (α ⟂ β)

-- Condition (4)
def condition4 := ∀ (l : Line) (β : Plane) (α : Plane),
  (l ∈ β) → (l ⟂ α) → (α ⟂ β)

-- Condition (5)
def condition5 := ∀ (m : Line) (α : Plane) (l : Line) (β : Plane),
  (m ∈ α) → (l ∈ β) → (α ∥ β → False) → (m ∥ l → False)

-- Proof statement with conditions and expected answer
theorem number_of_correct_propositions : 
  (↑2) = 
    (if condition1 then 1 else 0) +
    (if condition2 then 1 else 0) +
    (if condition3 then 1 else 0) +
    (if condition4 then 1 else 0) +
    (if condition5 then 1 else 0) := 
by sorry

end number_of_correct_propositions_l796_796730


namespace mean_of_quadrilateral_angles_l796_796141

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796141


namespace circumscribed_circle_area_l796_796248

/-- 
Statement: The area of the circle circumscribed about an equilateral triangle with side lengths of 9 units is 27π square units.
-/
theorem circumscribed_circle_area (s : ℕ) (h : s = 9) : 
  let radius := (2/3) * (4.5 * Real.sqrt 3)
  let area := Real.pi * (radius ^ 2)
  area = 27 * Real.pi :=
by
  -- Axis and conditions definitions
  have := h

  -- Definition for the area based on the radius
  let radius := (2/3) * (4.5 * Real.sqrt 3)
  let area := Real.pi * (radius ^ 2)

  -- Statement of the equality to be proven
  show area = 27 * Real.pi
  sorry

end circumscribed_circle_area_l796_796248


namespace rate_of_interest_l796_796829

-- Given conditions
def P : ℝ := 1500
def SI : ℝ := 735
def r : ℝ := 7
def t := r  -- The time period in years is equal to the rate of interest

-- The formula for simple interest and the goal
theorem rate_of_interest : SI = P * r * t / 100 ↔ r = 7 := 
by
  -- We will use the given conditions and check if they support r = 7
  sorry

end rate_of_interest_l796_796829


namespace find_x_squared_plus_y_squared_l796_796741

theorem find_x_squared_plus_y_squared (x y : ℝ) (h₁ : x * y = -8) (h₂ : x^2 * y + x * y^2 + 3 * x + 3 * y = 100) : x^2 + y^2 = 416 :=
sorry

end find_x_squared_plus_y_squared_l796_796741


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796061

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796061


namespace csc_315_equals_neg_sqrt_2_l796_796985

-- Define the conditions
def csc (x : Real) : Real := 1 / Real.sin x
axiom sin_subtraction_360 (x : Real) : Real.sin (360 - x) = -Real.sin x

-- State the problem
theorem csc_315_equals_neg_sqrt_2 : csc 315 = -Real.sqrt 2 := by
  sorry

end csc_315_equals_neg_sqrt_2_l796_796985


namespace evaluate_expression_l796_796976

theorem evaluate_expression :
  (floor (-3 - 0.5) * ceil (3 + 0.5)) *
  (floor (-2 - 0.5) * ceil (2 + 0.5)) *
  (floor (-1 - 0.5) * ceil (1 + 0.5)) = -576 :=
by
  have h : ∀ n : Int, (floor (-n - 0.5) * ceil (n + 0.5)) = -(n + 1) * (n + 1) :=
    sorry -- This should be provided as a given condition
  calc
    (floor (-3 - 0.5) * ceil (3 + 0.5)) *
    (floor (-2 - 0.5) * ceil (2 + 0.5)) *
    (floor (-1 - 0.5) * ceil (1 + 0.5)) =
    ((-(3 + 1) * (3 + 1)) * (-(2 + 1) * (2 + 1)) * (-(1 + 1) * (1 + 1))) : by
      rw [h 3, h 2, h 1]
    ... = -576 : sorry -- Calculation step to show -576 as the result

end evaluate_expression_l796_796976


namespace log_base_half_strictly_decreasing_interval_l796_796969

noncomputable def is_strictly_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f x > f y

theorem log_base_half_strictly_decreasing_interval :
  (∀ x, x ∈ (Ici 2) → is_strictly_decreasing (λ x, Real.log2 (x^2 - 2*x)) (Ici 2)) := by
  sorry

end log_base_half_strictly_decreasing_interval_l796_796969


namespace max_profit_l796_796842

noncomputable def annual_profit (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 60 then
    6 * x - (1/2 * x^2 + x) - 4
  else
    6 * x - (7 * x + 100 / x - 39) - 4

theorem max_profit :
  (∃ x, annual_profit x = 15 ∧ 0 < x ∧ x = 10) :=
begin
  sorry
end

end max_profit_l796_796842


namespace expected_total_rainfall_over_week_l796_796269

noncomputable def daily_rain_expectation : ℝ :=
  (0.5 * 0) + (0.2 * 2) + (0.3 * 5)

noncomputable def total_rain_expectation (days: ℕ) : ℝ :=
  days * daily_rain_expectation

theorem expected_total_rainfall_over_week : total_rain_expectation 7 = 13.3 :=
by 
  -- calculation of expected value here
  -- daily_rain_expectation = 1.9
  -- total_rain_expectation 7 = 7 * 1.9 = 13.3
  sorry

end expected_total_rainfall_over_week_l796_796269


namespace median_intersection_on_angle_bisector_l796_796347

variables {A B C D P Q : Type}
variables [trapezoid ABCD : Prop] (base1 : is_base AD ABCD) (base2 : is_base BC ABCD)
variables (medians_of_ABD : intersection_of_medians ABD P)
variables (angle_bisector_BCD : is_on_angle_bisector P BCD)

theorem median_intersection_on_angle_bisector
  (h_trapezoid : trapezoid ABCD)
  (h_base1 : is_base AD ABCD)
  (h_base2 : is_base BC ABCD)
  (h_medians_ABD : intersection_of_medians ABD P)
  (h_angle_bisector_BCD : is_on_angle_bisector P BCD) :
  ∃ Q, intersection_of_medians ABC Q ∧ is_on_angle_bisector Q ADC :=
sorry

end median_intersection_on_angle_bisector_l796_796347


namespace angle_of_vectors_l796_796734

noncomputable def a : ℝ → ℝ → ℝ × ℝ := λ x y, (x, y)
noncomputable def b : ℝ → ℝ → ℝ × ℝ := λ x y, (x, y)

def angle_between (u v : ℝ × ℝ) : ℝ :=
  real.arccos ((u.1 * v.1 + u.2 * v.2) / (real.sqrt (u.1 ^ 2 + u.2 ^ 2) * real.sqrt (v.1 ^ 2 + v.2 ^ 2)))

theorem angle_of_vectors :
  ∀ (u v : ℝ × ℝ), 
    (u.1 * (u.1 - v.1) + u.2 * (u.2 - v.2) = 0) →
    (real.sqrt (u.1 ^ 2 + u.2 ^ 2) = 3) →
    (real.sqrt (v.1 ^ 2 + v.2 ^ 2) = 2 * real.sqrt 3) →
    angle_between u v = real.pi / 6 :=
by
  sorry

end angle_of_vectors_l796_796734


namespace mean_of_quadrilateral_angles_l796_796194

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796194


namespace sum_of_coordinates_of_B_l796_796822

-- Definitions
def Point := (ℝ × ℝ)
def isMidpoint (M A B : Point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Given conditions
def M : Point := (4, 8)
def A : Point := (10, 4)

-- Statement to prove
theorem sum_of_coordinates_of_B (B : Point) (h : isMidpoint M A B) :
  B.1 + B.2 = 10 :=
by
  sorry

end sum_of_coordinates_of_B_l796_796822


namespace multiple_of_six_l796_796836

theorem multiple_of_six (n : ℤ) (h : (n^5 / 120 + n^3 / 24 + n / 30 : ℚ) = int.natAbs (n^5 / 120 + n^3 / 24 + n / 30)) :
  ∃ k : ℤ, n = 6 * k :=
by
  sorry

end multiple_of_six_l796_796836


namespace sufficient_but_not_necessary_condition_l796_796681

-- Definitions based on conditions
def z1 (m : ℝ) : ℂ := (m^2 + m + 1 : ℝ) + (m^2 + m - 4 : ℝ) * complex.i
def z2 : ℂ := 3 - 2 * complex.i

-- Lean statement matching the equivalent proof problem
theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (z1 1 = z2) ∧ (∃ (m' : ℝ), m' ≠ 1 ∧ z1 m' = z2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l796_796681


namespace probability_black_ball_BoxB_higher_l796_796950

def boxA_red_balls : ℕ := 40
def boxA_black_balls : ℕ := 10
def boxB_red_balls : ℕ := 60
def boxB_black_balls : ℕ := 40
def boxB_white_balls : ℕ := 50

theorem probability_black_ball_BoxB_higher :
  (boxA_black_balls : ℚ) / (boxA_red_balls + boxA_black_balls) <
  (boxB_black_balls : ℚ) / (boxB_red_balls + boxB_black_balls + boxB_white_balls) :=
by
  sorry

end probability_black_ball_BoxB_higher_l796_796950


namespace exists_small_triangle_area_l796_796434

theorem exists_small_triangle_area
  (A B C D : Type)
  (T : Fin 2000 → Type)
  (no_three_collinear : ∀ (P Q R : Type), P ≠ Q → Q ≠ R → R ≠ P → ¬ collinear P Q R)
  (square_side : nat := 20)
  (square_area : nat := 400) :
  ∃ (P Q R : S), area P Q R < 1 / 10 :=
by
  let S := insert A (insert B (insert C (insert D (finset.image T finset.univ))))
  sorry

end exists_small_triangle_area_l796_796434


namespace mean_value_of_quadrilateral_angles_l796_796160

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796160


namespace city_X_coordinates_l796_796616

def oslo_latitude : ℝ := 59 + 55 / 60
def oslo_longitude : ℝ := 10 + 43 / 60
def equator_latitude : ℝ := 0

def shortest_route_west (longitude : ℝ) (distance_west : ℝ) : ℝ :=
    longitude - distance_west

theorem city_X_coordinates :
  let X_latitude := equator_latitude in
  let X_longitude := shortest_route_west oslo_longitude 90 in
  (X_latitude = 0) ∧ (X_longitude = -(79 + 17 / 60)) :=
by
  -- Here, we expect the proof to be provided
  sorry

end city_X_coordinates_l796_796616


namespace mean_value_of_quadrilateral_angles_l796_796043

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796043


namespace torchbearers_consecutive_probability_l796_796645

theorem torchbearers_consecutive_probability :
  let torchbearers := {1, 2, 3, 4, 5}
  let pairs := (Finset.powersetLen 2 torchbearers).val
  let consecutive_pairs := {(1, 2), (2, 3), (3, 4), (4, 5)}
  prob_consecutive :
    (consecutive_pairs.card).natAbs.toRat / (pairs.card).natAbs.toRat = 2 / 5 :=
by
  sorry

end torchbearers_consecutive_probability_l796_796645


namespace transfer_increases_averages_l796_796909

noncomputable def initial_average_A : ℚ := 47.2
noncomputable def initial_average_B : ℚ := 41.8
def initial_students : ℕ := 10
noncomputable def score_filin : ℚ := 44
noncomputable def score_lopatin : ℚ := 47

theorem transfer_increases_averages :
  let sum_A := initial_students * initial_average_A,
      sum_B := initial_students * initial_average_B,
      new_sum_A := sum_A - score_lopatin - score_filin,
      new_average_A := new_sum_A / (initial_students - 2),
      new_sum_B := sum_B + score_lopatin + score_filin,
      new_average_B := new_sum_B / (initial_students + 2)
  in new_average_A > initial_average_A / 2 ∧ new_average_B > initial_average_B / 2 :=
by {
  sorry
}

end transfer_increases_averages_l796_796909


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796025

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796025


namespace gaeun_taller_than_nana_l796_796817

noncomputable def nana_height_m : Real := 1.618
def nana_height_cm : Real := nana_height_m * 100

def gaeun_height_cm : Real := 162.3

theorem gaeun_taller_than_nana : gaeun_height_cm > nana_height_cm := 
by
  -- The proof will be provided here
  sorry

end gaeun_taller_than_nana_l796_796817


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796070

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796070


namespace first_mechanic_hourly_charge_l796_796528

theorem first_mechanic_hourly_charge (M_1 : ℝ) (M_2 : ℝ) (T : ℝ) (C : ℝ) (T_2 : ℝ) :
  M_2 = 85 → T = 20 → C = 1100 → T_2 = 5 → 
  let T_1 := T - T_2 in
  let C_2 := M_2 * T_2 in
  let C_1 := C - C_2 in
  M_1 = C_1 / T_1 →
  M_1 = 45 :=
begin
  intros,
  let T_1 := T - T_2,
  let C_2 := M_2 * T_2,
  let C_1 := C - C_2,
  have M_1_def : M_1 = C_1 / T_1,
  { assumption },
  have goal : M_1 = 45,
  { -- proves that if we get to this point, M_1 is indeed 45, using values for terms
    rw M_2 at *,
    rw T at *,
    rw C at *,
    rw T_2 at *,
    simp at *,
    sorry },
  exact goal
end

end first_mechanic_hourly_charge_l796_796528


namespace odd_function_a_eq_one_l796_796395

theorem odd_function_a_eq_one (a : ℝ) :
  (∀ x : ℝ, (e^{x} + a * e^{-x}) * sin x = -(e^{-x} + a * e^{x}) * sin (-x)) → a = 1 := by
  intro h
  -- Proof is omitted
  sorry

end odd_function_a_eq_one_l796_796395


namespace arithmetic_sequence_problem_l796_796778

variable {a₁ d : ℝ} (S : ℕ → ℝ)

axiom Sum_of_terms (n : ℕ) : S n = n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_problem
  (h : S 10 = 4 * S 5) :
  (a₁ / d) = 1 / 2 :=
by
  -- definitional expansion and algebraic simplification would proceed here
  sorry

end arithmetic_sequence_problem_l796_796778


namespace min_value_geometric_sequence_l796_796442

theorem min_value_geometric_sequence (a1 a2 a3 : ℝ) (r : ℝ) (h1 : a1 = 1)
  (h2 : a2 = r) (h3 : a3 = r^2) : ∃ (m : ℝ), m = 3a2 + 7a3 ∧ m = -27/28 :=
by
  sorry

end min_value_geometric_sequence_l796_796442


namespace find_area_triangle_ABC_l796_796618

-- Define the triangles and conditions
variables {P A B C D E F : Type*}

-- Define the isosceles right triangles and areas
def isosceles_right_triangle (T : Type*) :=
  ∃ (a b c : ℝ), T = {a, b, c} ∧ (a * a + b * b = c * c)

def hypotenuse_on_hypotenuse (T1 T2 : Type*) :=
  ∃ (a b c : ℝ), T1 = {a, b, c} ∧ ∃ (d e f : ℝ), T2 = {d, e, f} ∧ (c = f)

axiom similar_triangles (T1 T2 : Type*) : 
  isosceles_right_triangle T1 → isosceles_right_triangle T2 → 
  (T1 : set ℝ) ⊆ T2 → true

axiom area_ratios : 
  area_ratios : real := 36

-- Main theorem stating the area of triangle ABC
theorem find_area_triangle_ABC :
  ∀ (DEF ABC : Type*), isosceles_right_triangle DEF → isosceles_right_triangle ABC → 
  hypotenuse_on_hypotenuse DEF ABC → 
  area_ratios DEF ABC → 
  area ABC = 36 :=
begin
  intros DEF ABC h1 h2 h3 h4,
  sorry
end

end find_area_triangle_ABC_l796_796618


namespace function_properties_l796_796942

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem function_properties :
  (∀ x : ℝ, f (Real.pi / 12 + x) + f (Real.pi / 12 - x) = 0) ∧
  (∀ x : ℝ, -Real.pi / 6 < x ∧ x < Real.pi / 3 → (Real.sin (2 * x - Real.pi / 6)).deriv.deriv > 0) :=
by 
  sorry

end function_properties_l796_796942


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796018

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796018


namespace find_cone_radius_l796_796363

-- Define the given conditions
def cone_height := 4
def cone_volume := 4 * Real.pi

-- Use the volume formula for a cone V = (1/3) * π * r^2 * h
def cone_radius (V : ℝ) (h : ℝ) : ℝ := 
  Real.sqrt ((3 * V) / (π * h))

-- Theorem statement: Prove that the base radius r of the cone is √3 given the conditions
theorem find_cone_radius : 
  cone_radius cone_volume cone_height = Real.sqrt 3 := by 
  sorry

end find_cone_radius_l796_796363


namespace baseball_opponents_score_l796_796913

theorem baseball_opponents_score 
  (team_scores : List ℕ)
  (team_lost_scores : List ℕ)
  (team_won_scores : List ℕ)
  (opponent_lost_scores : List ℕ)
  (opponent_won_scores : List ℕ)
  (h1 : team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  (h2 : team_lost_scores = [1, 3, 5, 7, 9, 11])
  (h3 : team_won_scores = [6, 9, 12])
  (h4 : opponent_lost_scores = [3, 5, 7, 9, 11, 13])
  (h5 : opponent_won_scores = [2, 3, 4]) :
  (List.sum opponent_lost_scores + List.sum opponent_won_scores = 57) :=
sorry

end baseball_opponents_score_l796_796913


namespace mean_of_quadrilateral_angles_is_90_l796_796097

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796097


namespace number_of_integers_on_or_inside_circle_l796_796326

theorem number_of_integers_on_or_inside_circle :
  {x : ℤ | (x - 5)^2 + (-x - 5)^2 ≤ 100}.finite.toFinset.card = 11 := by
  sorry

end number_of_integers_on_or_inside_circle_l796_796326


namespace mean_value_of_quadrilateral_interior_angles_l796_796171

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796171


namespace volume_calculation_l796_796285

theorem volume_calculation (m n p : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) (p_pos : 0 < p)
  (rel_prime : Nat.coprime n p) :
  let volume := (3 * 4 * 5) + (2 * (3 * 4 * 1) + 2 * (3 * 5 * 1) + 2 * (4 * 5 * 1)) + (8 * ((1/8) * (4/3) * Real.pi * 1^3)) + (12 * (Real.pi * 1^2 * (3 + 4 + 5)))
  in m = 462 ∧ n = 40 ∧ p = 3 ∧ volume = (m + n * Real.pi) / p ∧ m + n + p = 505 :=
by
  sorry

end volume_calculation_l796_796285


namespace mean_value_of_quadrilateral_angles_l796_796056

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796056


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796020

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796020


namespace people_needed_to_mow_lawn_l796_796838

/-- 
Given that six people can mow a lawn in 8 hours and each person mows at the same rate,
prove that 10 more people are needed to mow the lawn in 3 hours.
-/
theorem people_needed_to_mow_lawn :
  let n1 := 6 in
  let t1 := 8 in
  let constant := n1 * t1 in
  let t2 := 3 in
  let m := constant / t2 in
  m - n1 = 10 :=
by
  let n1 := 6
  let t1 := 8
  let constant := n1 * t1
  let t2 := 3
  let m := constant / t2
  have h : m - n1 = 10 := by sorry
  exact h


end people_needed_to_mow_lawn_l796_796838


namespace mean_value_of_quadrilateral_angles_l796_796204

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796204


namespace mean_value_of_quadrilateral_angles_l796_796087

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796087


namespace attendees_receive_all_items_l796_796946

theorem attendees_receive_all_items (capacity : ℕ) (n1 n2 n3 : ℕ) 
  (h_capacity : capacity = 5400) 
  (h_n1 : n1 = 90) 
  (h_n2 : n2 = 45) 
  (h_n3 : n3 = 60) : 
  (finset.card (finset.filter (λ x, x % n1 = 0 ∧ x % n2 = 0 ∧ x % n3 = 0) 
  (finset.range capacity))) = 30 :=
by 
  sorry

end attendees_receive_all_items_l796_796946


namespace csc_315_equals_neg_sqrt_2_l796_796983

-- Define the conditions
def csc (x : Real) : Real := 1 / Real.sin x
axiom sin_subtraction_360 (x : Real) : Real.sin (360 - x) = -Real.sin x

-- State the problem
theorem csc_315_equals_neg_sqrt_2 : csc 315 = -Real.sqrt 2 := by
  sorry

end csc_315_equals_neg_sqrt_2_l796_796983


namespace exponential_inequality_l796_796804

variables (x a b : ℝ)

theorem exponential_inequality (h1 : x > 0) (h2 : 1 < b^x) (h3 : b^x < a^x) : 1 < b ∧ b < a :=
by
   sorry

end exponential_inequality_l796_796804


namespace reciprocal_sum_fraction_l796_796218

theorem reciprocal_sum_fraction : (1/3 + 3/4)⁻¹ = 12/13 := 
by
  sorry

end reciprocal_sum_fraction_l796_796218


namespace mean_value_of_quadrilateral_angles_l796_796053

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796053


namespace find_other_parallel_side_l796_796655

theorem find_other_parallel_side 
  (a b d : ℝ) 
  (area : ℝ) 
  (h_area : area = 285) 
  (h_a : a = 20) 
  (h_d : d = 15)
  : (∃ x : ℝ, area = 1/2 * (a + x) * d ∧ x = 18) :=
by
  sorry

end find_other_parallel_side_l796_796655


namespace interest_difference_l796_796385

theorem interest_difference (P P_B : ℝ) (R_A R_B T : ℝ)
    (h₁ : P = 10000)
    (h₂ : P_B = 4000.0000000000005)
    (h₃ : R_A = 15)
    (h₄ : R_B = 18)
    (h₅ : T = 2) :
    let P_A := P - P_B
    let I_A := (P_A * R_A * T) / 100
    let I_B := (P_B * R_B * T) / 100
    I_A - I_B = 359.99999999999965 := 
by
  sorry

end interest_difference_l796_796385


namespace part_a_part_b_l796_796479

variable (poly : Type) [SimplyConnectedPolyhedron poly]
variables (𝒯 𝒢 𝒫 ℬ : ℕ) -- Representing T, G, P, B respectively

theorem part_a (h1 : ℯulerFormulaHolds poly 𝒯 𝒢 𝒫 ℬ) (h2: 𝒯 ≥ 3 / 2 * 𝒢) : 
  3 / 2 ≤ 𝒯 / 𝒢 ∧ 𝒯 / 𝒢 < 3 := 
sorry

theorem part_b (h3 : 𝒫 ≥ 3 / 2 * ℬ) : 
  3 / 2 ≤ 𝒫 / ℬ ∧ 𝒫 / ℬ < 3 := 
sorry

end part_a_part_b_l796_796479


namespace winter_distances_equal_l796_796843

noncomputable def equilateral_triangle (a : ℝ) : Prop :=
  ∀ (A B C : point), distance A B = a → distance B C = a → distance C A = a

variables {A B C K L N : point}
variables (x : ℝ) {side_length : ℝ} (known_distance : ℝ) (summer_distance : ℝ)

def krosh_barash_condition (Krosh Barash : point) : Prop :=
  distance Krosh Barash = 300

def summer_distances (Losyash Krosh Barash Nyusha : point) : Prop :=
  distance Losyash Krosh = 900 ∧ distance Barash Nyusha = 900

theorem winter_distances_equal (Losyash Krosh Barash Nyusha : point) :
  equilateral_triangle 600 →
  krosh_barash_condition Krosh Barash →
  summer_distances Losyash Krosh Barash Nyusha →
  distance Losyash Krosh = distance Barash Nyusha :=
begin
  sorry
end

end winter_distances_equal_l796_796843


namespace mean_value_of_quadrilateral_interior_angles_l796_796181

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796181


namespace mean_value_interior_angles_quadrilateral_l796_796108

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796108


namespace y_intercept_of_line_l796_796517

theorem y_intercept_of_line (a b : ℝ) : 
  ∃ y : ℝ, (∃ x : ℝ, x = 0) → (x / a - y / b = 1) → y = -b :=
by
  intro a b
  exists -b
  intro h1 h2
  have h3 : x = 0 := sorry
  have h4 : -y / b = 1 := h2
  sorry

end y_intercept_of_line_l796_796517


namespace fraction_decomposition_roots_sum_l796_796454

theorem fraction_decomposition_roots_sum :
  ∀ (p q r A B C : ℝ),
  p ≠ q → p ≠ r → q ≠ r →
  (∀ (s : ℝ), s ≠ p → s ≠ q → s ≠ r →
          1 / (s^3 - 15 * s^2 + 50 * s - 56) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 225 :=
by
  intros p q r A B C hpq hpr hqr hDecomp
  -- Skip proof
  sorry

end fraction_decomposition_roots_sum_l796_796454


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796060

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796060


namespace six_digit_numbers_correct_count_l796_796330

noncomputable def six_digit_numbers_count : Nat :=
  let digits := {1, 2, 3, 4, 5, 6}
  let is_even (n : Nat) : Bool := n % 2 == 0
  let is_odd (n : Nat) : Bool := not (is_even n)
  let position_valid (seq : List Nat) : Bool :=
    seq.length == 6 ∧
    (∀ i < 5, seq.nth i % 2 == 0 → seq.nth (i + 1) % 2 == 1) ∧
    (∀ i < 5, seq.nth i % 2 == 1 → seq.nth (i + 1) % 2 == 0) ∧
    seq.nth 3 ≠ some 4
  let valid_permutations : List (List Nat) :=
    (List.permutations digits.toList).filter position_valid
  valid_permutations.length

theorem six_digit_numbers_correct_count : six_digit_numbers_count = 120 :=
  by
    sorry

end six_digit_numbers_correct_count_l796_796330


namespace B_is_right_and_hypotenuse_is_sqrt65_l796_796381

open Real

-- Define points and triangles
def point := (ℝ × ℝ)
def triangle := point × point × point

-- Define vertices of Triangle A and Triangle B
def A : triangle :=
  ((0, 0), (3, 4), (0, 8))

def B : triangle :=
  ((3, 4), (10, 4), (3, 0))

-- Function to calculate the square of the distance between two points
def dist_squared (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

-- Function to check if a triangle is a right triangle
def is_right_triangle (t : triangle) : Prop :=
  let (p1, p2, p3) := t
  let d12 := dist_squared p1 p2
  let d13 := dist_squared p1 p3
  let d23 := dist_squared p2 p3
  d12 + d13 = d23 ∨ d12 + d23 = d13 ∨ d13 + d23 = d12

-- Function to determine the length of the hypotenuse
def hypotenuse_length (t : triangle) : ℝ :=
  let (p1, p2, p3) := t
  my_real.sqrt (max (dist_squared p1 p2) (max (dist_squared p1 p3) (dist_squared p2 p3)))

-- Prove that B is a right triangle and the hypotenuse is √65
theorem B_is_right_and_hypotenuse_is_sqrt65 : is_right_triangle B ∧ hypotenuse_length B = sqrt 65 := by
  sorry

end B_is_right_and_hypotenuse_is_sqrt65_l796_796381


namespace general_formula_sum_first_20_b_l796_796693

noncomputable theory

-- Defining the geometric sequence a_n
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * 2)

-- Defining the arithmetic condition
def arithmetic_condition (a : ℕ → ℝ) : Prop :=
  let a1 := a 1 in
  let a2 := a 2 in
  let a3 := a 3 in
  2 * a2 = a1 + (a3 - 1)

-- Definition of b_n sequence
def b_sequence (a b : ℕ → ℝ) : Prop :=
  (∀ n, if n % 2 = 1 then b n = a n - 1 else b n = a n / 2)

-- The main theorem to prove the general formula for a_n
theorem general_formula (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 = 2) (h3 : arithmetic_condition a) :
  ∀ n, a n = 2 ^ (n - 1) :=
sorry

-- The main theorem to calculate the sum of the first 20 terms of b_n
theorem sum_first_20_b (a b : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 = 2) (h3 : arithmetic_condition a) (h4 : b_sequence a b) :
  (∑ i in finset.range 20, b (i + 1)) = (2 ^ 21 - 32) / 3 :=
sorry

end general_formula_sum_first_20_b_l796_796693


namespace toaster_sales_l796_796287

-- Define the context of the problem
variable (p c a : ℝ)  -- p: popularity, c: cost, a: age
variable (k : ℝ)  -- proportionality constant

-- Given conditions
axiom H1 : p = 16 → c = 400 → a = 2 → k * a / c = p
axiom H2 : ∀ c a, p = k * a / c

-- Prove that for a new toaster costing $800 and 4 years old, the popularity is 16
theorem toaster_sales : ∃ p k, (c = 800) → (a = 4) → (p = 16) :=
by
  have h1 : k = 3200 := sorry,
  have h2 : ∀ c a, p = k * a / c := by
    intro c a
    sorry,
  show ∃ (p k), c = 800 → a = 4 → p = 16
  sorry

end toaster_sales_l796_796287


namespace solve_for_x_l796_796839

theorem solve_for_x : 
  ∃! x : ℝ, (1/8)^(3 * x + 12) = 32^(3 * x + 4) ∧ x = -7/3 :=
begin
  use -7/3,
  split,
  {
    -- Show that the solution satisfies the equation
    sorry
  },
  {
    -- Show that it is the only solution
    intro y,
    rw [←this],
    sorry
  }
end

end solve_for_x_l796_796839


namespace csc_315_eq_neg_sqrt_two_l796_796988

theorem csc_315_eq_neg_sqrt_two : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_two_l796_796988


namespace ants_meet_time_l796_796527

noncomputable def circumference (r : ℝ) : ℝ :=
  2 * r * Real.pi

def time_to_complete_rotation (circumference velocity : ℝ) : ℝ :=
  circumference / velocity

def lcm (a b : ℕ) : ℕ := sorry

theorem ants_meet_time :
  let r1 := 7
  let r2 := 3
  let v1 := 5 * Real.pi
  let v2 := 3 * Real.pi
  let t1 := time_to_complete_rotation (circumference r1) v1
  let t2 := time_to_complete_rotation (circumference r2) v2
  (lcm (Nat.ceil t1) (Nat.ceil t2) = 70) :=
by
  sorry

end ants_meet_time_l796_796527


namespace sum_of_greater_set_diff_l796_796904

theorem sum_of_greater_set_diff (C : ℤ) (h1 : 2 * C + 1 > 0) (h2 : C % 2 = 1) : 
  let set1 := [(C-2), C, (C+2)] in
  let set2 := [(C-4), (C-2), C] in
  (set1.sum - set2.sum = 6) := 
by
  sorry

end sum_of_greater_set_diff_l796_796904


namespace runners_meet_opposite_dir_l796_796883

theorem runners_meet_opposite_dir 
  {S x y : ℝ}
  (h1 : S / x + 5 = S / y)
  (h2 : S / (x - y) = 30) :
  S / (x + y) = 6 := 
sorry

end runners_meet_opposite_dir_l796_796883


namespace mean_value_of_quadrilateral_interior_angles_l796_796177

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796177


namespace mean_value_of_quadrilateral_angles_l796_796207

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796207


namespace ducks_and_geese_difference_l796_796621

variable (d g d' l : ℕ)
variables (hd : d = 25)
variables (hg : g = 2 * d - 10)
variables (hd' : d' = d + 4)
variables (hl : l = 15 - 5)

theorem ducks_and_geese_difference :
  let geese_remain := g - l
  let ducks_remain := d'
  geese_remain - ducks_remain = 1 :=
by
  sorry

end ducks_and_geese_difference_l796_796621


namespace interchangeable_statements_l796_796393

theorem interchangeable_statements:
  (∀ l₁ l₂ p, (l₁ ⊥ p ∧ l₂ ⊥ p) → l₁ ∥ l₂) ↔ (∀ p₁ p₂ l, (p₁ ⊥ l ∧ p₂ ⊥ l) → p₁ ∥ p₂) ∧
  (∀ l₁ l₂ l₃, (l₁ ∥ l₃ ∧ l₂ ∥ l₃) → l₁ ∥ l₂) ↔ (∀ p₁ p₂ p₃, (p₁ ∥ p₃ ∧ p₂ ∥ p₃) → p₁ ∥ p₂) :=
sorry

end interchangeable_statements_l796_796393


namespace Petya_can_take_last_stone_l796_796522

theorem Petya_can_take_last_stone 
    (n : ℕ) 
    (pile_size : ℕ) 
    (hn : pile_size > n^2) 
    (take_stones : ℕ → Prop) :
    (∀ k, (prime k ∧ k < n) ∨ k = 1 ∨ (∃ m, k = n * m)) → 
    ∃ strategy : (ℕ → ℕ), 
    ∀ (pile : ℕ), (pile > 0) → 
    (pile ≤ pile_size) → 
    (strategy pile ≤ min n pile) → 
    (∃ (last_pile : ℕ), (last_pile = strategy pile ∨ last_pile = 0)) → 
    last_pile = 0 :=
by
  sorry

end Petya_can_take_last_stone_l796_796522


namespace coordinates_of_C_l796_796583

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-3, 7⟩
def B : Point := ⟨9, -1⟩

-- Define a function to calculate the midpoint between two points
def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

-- Define the condition that BC = 1/2 * AB
def BC1_2AB (A B C : Point) : Prop :=
  let deltaX := B.x - A.x
  let deltaY := B.y - A.y
  let Cx := B.x + deltaX / 2
  let Cy := B.y + deltaY / 2
  (C.x = Cx) ∧ (C.y = Cy)

-- Theorem to prove the coordinates of C
theorem coordinates_of_C (C : Point) (h : BC1_2AB A B C) : C = { x := 15, y := -5 } :=
by
  -- The proof will go here
  sorry

end coordinates_of_C_l796_796583


namespace students_not_A_l796_796404

noncomputable def num_students := 40
noncomputable def num_A_history := 12
noncomputable def num_A_math := 18
noncomputable def num_A_both := 6

def num_A_either := num_A_history + num_A_math - num_A_both
def num_not_A := num_students - num_A_either

theorem students_not_A : num_not_A = 16 := by
  unfold num_not_A num_students num_A_either
  rw [num_A_either, num_A_history, num_A_math, num_A_both]
  norm_num
  done

end students_not_A_l796_796404


namespace mean_of_quadrilateral_angles_l796_796149

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796149


namespace mean_value_of_quadrilateral_angles_l796_796008

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l796_796008


namespace root_polynomial_p_q_l796_796801

theorem root_polynomial_p_q (p q : ℝ) (h_root : is_root (λ x : ℂ, x^3 + (p : ℂ) * x + (q : ℂ)) (2 + 1 * Complex.I)) :
  p + q = 9 := 
by
  sorry

end root_polynomial_p_q_l796_796801


namespace marlon_goals_l796_796764

theorem marlon_goals :
  ∃ g : ℝ,
    (∀ p f : ℝ, p + f = 40 → g = 0.4 * p + 0.5 * f) → g = 20 :=
by
  sorry

end marlon_goals_l796_796764


namespace total_time_required_l796_796642

-- Define Doug's painting rate
def doug_rate : ℝ := 1 / 4

-- Define Dave's painting rate
def dave_rate : ℝ := 1 / 6

-- Combined painting rate
def combined_rate : ℝ := doug_rate + dave_rate

-- Definition of total time s, including break
def total_time_incl_break (s : ℝ) : Prop := (combined_rate * (s - 2)) = 1

-- Prove the total time including the break is 22/5 hours
theorem total_time_required : ∃ s : ℝ, total_time_incl_break s ∧ s = 22 / 5 :=
by
  use 22 / 5
  split
  . sorry  -- placeholder for proof
  . rfl

end total_time_required_l796_796642


namespace mean_value_of_quadrilateral_angles_l796_796085

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796085


namespace csc_315_equals_neg_sqrt_2_l796_796978

-- Define the conditions
def csc (x : Real) : Real := 1 / Real.sin x
axiom sin_subtraction_360 (x : Real) : Real.sin (360 - x) = -Real.sin x

-- State the problem
theorem csc_315_equals_neg_sqrt_2 : csc 315 = -Real.sqrt 2 := by
  sorry

end csc_315_equals_neg_sqrt_2_l796_796978


namespace find_vector_difference_magnitude_l796_796703

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Assuming a and b are unit vectors 
def is_unit_vector (v : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∥v∥ = 1

-- Assuming the angle between a and b is π/3
def angle_is_pi_over_3 (a b : EuclideanSpace ℝ (Fin 3)) : Prop :=
  real_inner_product_space.angle a b = π/3

-- Problem Statement
theorem find_vector_difference_magnitude
  (hu_a : is_unit_vector a)
  (hu_b : is_unit_vector b)
  (angle_ab : angle_is_pi_over_3 a b) :
  ∥a - 2 • b∥ = 2 :=
sorry

end find_vector_difference_magnitude_l796_796703


namespace distinct_terms_in_expansion_l796_796863

theorem distinct_terms_in_expansion (a b : ℝ) :
  let expr := (a + 2*b)^3 * (a - 2*b)^3 
  in (expr^2).polynomial.num_terms = 7 := 
by 
  sorry

end distinct_terms_in_expansion_l796_796863


namespace range_of_k_l796_796755

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3 / 8 < 0) ↔ k ∈ Icc (-3 : ℝ) 0 :=
by
  sorry

end range_of_k_l796_796755


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796063

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796063


namespace sum_of_powers_of_i_l796_796664

theorem sum_of_powers_of_i (i : ℂ) (h : i^2 = -1) : i + i^2 + i^3 + i^4 + i^5 = i := by
  sorry

end sum_of_powers_of_i_l796_796664


namespace fraction_alice_gives_to_charlie_is_zero_l796_796941

variable {c : ℕ} -- number of stickers Charlie has

/-- Conditions -/
def stickers_bob : ℕ := 3 * c -- Bob has 3 times as many as Charlie
def stickers_dave : ℕ := c -- Dave has the same number as Charlie
def stickers_alice : ℕ := 4 * stickers_bob -- Alice has 4 times of Bob
def stickers_bob_final : ℕ := 2 * c -- Bob will have twice as Charlie after redistribution
def stickers_dave_final : ℕ := 3 * c -- Dave will have three times as Charlie after redistribution

/-- Number of stickers Alice needs to distribute in total -/
def stickers_to_distribute : ℕ := stickers_alice - (stickers_bob_final + c + stickers_dave_final)

/-- Fraction of Alice's stickers given to Charlie is zero -/
theorem fraction_alice_gives_to_charlie_is_zero (h : stickers_to_distribute = 6 * c) : 
  0 = stickers_to_distribute / stickers_alice := by
sorry

end fraction_alice_gives_to_charlie_is_zero_l796_796941


namespace calculate_overall_profit_percentage_l796_796264

noncomputable def overall_profit_percentage : ℚ :=
  let price_cricket_bat := 850
  let price_tennis_racket := 600
  let price_baseball_glove := 100
  
  let profit_cricket_bat := 230
  let profit_tennis_racket := 120
  let profit_baseball_glove := 30
  
  let qty_cricket_bat := 4
  let qty_tennis_racket := 5
  let qty_baseball_glove := 10
  
  let cp_cricket_bat := price_cricket_bat - profit_cricket_bat
  let cp_tennis_racket := price_tennis_racket - profit_tennis_racket
  let cp_baseball_glove := price_baseball_glove - profit_baseball_glove
  
  let total_sp := qty_cricket_bat * price_cricket_bat + qty_tennis_racket * price_tennis_racket + qty_baseball_glove * price_baseball_glove
  let total_cp := qty_cricket_bat * cp_cricket_bat + qty_tennis_racket * cp_tennis_racket + qty_baseball_glove * cp_baseball_glove
  
  let total_profit := (total_sp : ℚ) - total_cp

  let profit_percentage := (total_profit / total_cp) * 100

  profit_percentage

theorem calculate_overall_profit_percentage : overall_profit_percentage ≈ 32.62 := 
  sorry

end calculate_overall_profit_percentage_l796_796264


namespace first_divisor_is_six_l796_796312

theorem first_divisor_is_six {d : ℕ} 
  (h1: (1394 - 14) % d = 0)
  (h2: (2535 - 1929) % d = 0)
  (h3: (40 - 34) % d = 0)
  : d = 6 :=
sorry

end first_divisor_is_six_l796_796312


namespace mean_value_of_quadrilateral_angles_l796_796130

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796130


namespace parallel_vectors_y_value_l796_796379

theorem parallel_vectors_y_value 
  (y : ℝ) 
  (a : ℝ × ℝ := (6, 2)) 
  (b : ℝ × ℝ := (y, 3)) 
  (h : ∃ k : ℝ, b = k • a) : y = 9 :=
sorry

end parallel_vectors_y_value_l796_796379


namespace mean_value_of_quadrilateral_angles_l796_796077

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796077


namespace probability_of_different_colors_is_11_over_18_l796_796519

noncomputable def probability_different_colors (blue yellow red : ℕ) : ℚ :=
  let total := blue + yellow + red in
  let p_blue_then_diff := (blue / total : ℚ) * (yellow / total + red / total) in
  let p_yellow_then_diff := (yellow / total : ℚ) * (blue / total + red / total) in
  let p_red_then_diff := (red / total : ℚ) * (blue / total + yellow / total) in
  p_blue_then_diff + p_yellow_then_diff + p_red_then_diff

theorem probability_of_different_colors_is_11_over_18 :
  probability_different_colors 6 4 2 = 11 / 18 := by
  sorry

end probability_of_different_colors_is_11_over_18_l796_796519


namespace diagonal_AC_length_l796_796415

variable (A B C D : Type)
variables [InnerProductSpace ℝ V] [CompleteSpace V]
variables (AB BC CD DA : ℝ) (angle_ADC : ℝ)

-- Conditions
notation "length_AB" => AB = 10
notation "length_BC" => BC = 10
notation "length_CD" => CD = 15
notation "length_DA" => DA = 15
notation "angle_ADC_120" => angle_ADC = 120

-- The proof problem statement
theorem diagonal_AC_length : 
  length_AB ∧ length_BC ∧ length_CD ∧ length_DA ∧ angle_ADC_120 → 
  ∃ AC : ℝ, AC = 15 * Real.sqrt 3 :=
by
  intros
  sorry

end diagonal_AC_length_l796_796415


namespace attendance_proof_l796_796469

noncomputable def next_year_attendance (this_year: ℕ) := 2 * this_year
noncomputable def last_year_attendance (next_year: ℕ) := next_year - 200
noncomputable def total_attendance (last_year this_year next_year: ℕ) := last_year + this_year + next_year

theorem attendance_proof (this_year: ℕ) (h1: this_year = 600):
    total_attendance (last_year_attendance (next_year_attendance this_year)) this_year (next_year_attendance this_year) = 2800 :=
by
  sorry

end attendance_proof_l796_796469


namespace sum_of_coefficients_l796_796964

theorem sum_of_coefficients:
  ∀ (a : ℕ → ℤ), 
  ((x^2 + 1) * (2 * x + 1)^9 = ∑ i in finset.range 12, a i * (x + 2)^i) → 
  (∑ i in finset.range 12, a i) = -2 :=
by
  intros a h
  sorry

end sum_of_coefficients_l796_796964


namespace student_knows_german_l796_796585

-- Definitions for each classmate's statement
def classmate1 (lang: String) : Prop := lang ≠ "French"
def classmate2 (lang: String) : Prop := lang = "Spanish" ∨ lang = "German"
def classmate3 (lang: String) : Prop := lang = "Spanish"

-- Conditions: at least one correct and at least one incorrect
def at_least_one_correct (lang: String) : Prop :=
  classmate1 lang ∨ classmate2 lang ∨ classmate3 lang

def at_least_one_incorrect (lang: String) : Prop :=
  ¬classmate1 lang ∨ ¬classmate2 lang ∨ ¬classmate3 lang

-- The statement to prove
theorem student_knows_german : ∀ lang : String,
  at_least_one_correct lang → at_least_one_incorrect lang → lang = "German" :=
by
  intros lang Hcorrect Hincorrect
  revert Hcorrect Hincorrect
  -- sorry stands in place of direct proof
  sorry

end student_knows_german_l796_796585


namespace mean_of_quadrilateral_angles_l796_796138

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796138


namespace mean_of_quadrilateral_angles_l796_796151

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796151


namespace problem_statement_l796_796370

def f (x : ℝ) : ℝ :=
if x ≤ 0 then 1 + 4^x else log 2 x

theorem problem_statement : f(f(√2 / 4)) = 9 / 8 := 
by sorry

end problem_statement_l796_796370


namespace increasing_implies_a_nonnegative_l796_796396

theorem increasing_implies_a_nonnegative (a : ℝ) :
  (∀ x : ℝ, 3 * x^2 - a ≥ 0) → a ≥ 0 :=
begin
  sorry
end

end increasing_implies_a_nonnegative_l796_796396


namespace triangle_CSE_is_equilateral_l796_796451

-- Definitions of the geometric entities involved
variable (k : Circle) (r : ℝ) (A B S C D E : Point)

-- Given conditions as mathematical conditions
axiom circle_radius (hk : k.radius = r)
axiom chord_greater_than_radius (hAB : (chord_length k A B) > r)
axiom point_S (hS : distance A S = r)
axiom perp_bisector_intersects (hCD : is_perpendicular_bisector (k.radius A B) C D)
axiom DS_line_intersects (hE : E ∈ (line_through D S ∩ k))

-- To prove: triangle CSE is equilateral
theorem triangle_CSE_is_equilateral :
  is_equilateral_triangle C S E := 
sorry

end triangle_CSE_is_equilateral_l796_796451


namespace quadratic_roots_inverse_sum_l796_796298

theorem quadratic_roots_inverse_sum (t q α β : ℝ) (h1 : α + β = t) (h2 : α * β = q) 
  (h3 : ∀ n : ℕ, n ≥ 1 → α^n + β^n = t) : (1 / α^2011 + 1 / β^2011) = 2 := 
by 
  sorry

end quadratic_roots_inverse_sum_l796_796298


namespace midpoint_and_distance_l796_796890

-- Definitions
def midpoint (x1 y1 x2 y2 : ℝ) : (ℝ × ℝ) :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Conditions
def endpoint1 : (ℝ × ℝ) := (9, -8)
def endpoint2 : (ℝ × ℝ) := (-1, 6)

-- Statement
theorem midpoint_and_distance :
  let p1 := endpoint1
      p2 := endpoint2
      M := midpoint p1.1 p1.2 p2.1 p2.2
  in M = (4, -1) ∧ distance M.1 M.2 p1.1 p1.2 = Real.sqrt 74 :=
by
  sorry

end midpoint_and_distance_l796_796890


namespace drawing_lots_is_correct_random_number_method_is_correct_l796_796882

noncomputable def draw_lots_method (students : Finset ℕ) (draws: Fin 8 -> ℕ) : Prop :=
  (students.card = 48) →
  (∀ i, draws i ∈ students) → 
  (draws.card = 8)

noncomputable def random_number_method (students : Finset ℕ) (draws: Fin 8 -> ℕ) :
Prop :=
  (students.card = 48) →
  (∀ i, draws i ∈ students) → 
  (draws.card = 8) → 
  (∀ i, draws i < 48) → 
  (function.injective draws)

theorem drawing_lots_is_correct (students : Finset ℕ) (draws: Fin 8 -> ℕ) :
  draw_lots_method students draws :=
sorry

theorem random_number_method_is_correct (students : Finset ℕ) (draws: Fin 8 -> ℕ) :
  random_number_method students draws :=
sorry

end drawing_lots_is_correct_random_number_method_is_correct_l796_796882


namespace regular_polygon_sides_l796_796591

theorem regular_polygon_sides (n : ℕ) (h_regular : ∀ i, ∠i = 170) : n = 36 :=
sorry

end regular_polygon_sides_l796_796591


namespace pipes_fill_cistern_together_time_l796_796884

theorem pipes_fill_cistern_together_time
  (t : ℝ)
  (h1 : t * (1 / 12 + 1 / 15) + 6 * (1 / 15) = 1) : 
  t = 4 := 
by
  -- Proof is omitted here as instructed
  sorry

end pipes_fill_cistern_together_time_l796_796884


namespace mean_value_of_quadrilateral_angles_l796_796155

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796155


namespace correct_statements_l796_796615

def condition1 : Prop := ¬monotone (λ x : ℝ, -1 / x)
def condition2 : Prop := ¬∀ (f : ℝ → ℝ), odd f → f 0 = 0
def condition3 : Prop := ∀ (f : ℝ → ℝ), (∀ x, f x = f (-x)) → (∀ x, f x = f x) 
def condition4 : Prop := ∀ (f : ℝ → ℝ), (∀ x, f x = -f x) ∧ (∀ x, f x = f (-x)) → (∀ x, f x = 0) 

theorem correct_statements : (cond1 ∧ ¬cond2 ∧ cond3 ∧ ¬cond4) ∨
                      (¬cond1 ∧ cond2 ∧ cond3 ∧ cond4) ∨
                      (¬cond1 ∧ ¬cond2 ∧ cond3 ∧ ¬cond4) ∨
                      (¬cond1 ∧ ¬cond2 ∧ ¬cond3 ∧ ¬cond4) → 1 = 1 :=
by sorry

end correct_statements_l796_796615


namespace mean_value_of_quadrilateral_angles_l796_796000

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l796_796000


namespace gavrila_travel_distance_l796_796673

-- Declaration of the problem conditions
noncomputable def smartphone_discharge_rate_video : ℝ := 1 / 3
noncomputable def smartphone_discharge_rate_tetris : ℝ := 1 / 5
def average_speed_first_half : ℝ := 80
def average_speed_second_half : ℝ := 60

-- Main theorem statement
theorem gavrila_travel_distance : ∃ d : ℝ, d = 257 := by
  -- Solve for the total travel time t
  let smartphone_discharge_time : ℝ := 15 / 4
  -- Expressing the distance formula
  let distance_1 := smartphone_discharge_time * average_speed_first_half
  let distance_2 := smartphone_discharge_time * average_speed_second_half
  let total_distance := (distance_1 / 2 + distance_2 / 2)
  -- Assert the final distance
  exact ⟨257, by decide⟩

end gavrila_travel_distance_l796_796673


namespace mean_of_quadrilateral_angles_l796_796140

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796140


namespace mean_value_of_quadrilateral_angles_l796_796082

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796082


namespace condition_suff_and_nec_l796_796446

def p (x : ℝ) : Prop := |x + 2| ≤ 3
def q (x : ℝ) : Prop := x < -8

theorem condition_suff_and_nec (x : ℝ) : p x ↔ ¬ q x :=
by
  sorry

end condition_suff_and_nec_l796_796446


namespace max_dominos_after_removal_l796_796672

def chessboard (n : Nat) := List (List (Option Bool))

def removed_cells (board : chessboard 8) (removed_count : Nat) :=
  removed_count = 10 ∧
  (List.countp id (List.map (Option.isSome) (List.join board)) = 10)

def guarantee_dominos_rem (board : chessboard 8) (domino_count: Nat) :=
  (∃ (removed_cells : List (Nat × Nat)),
      List.length removed_cells = 10 ∧
      ∀ (x y : Nat × Nat), x ≠ y → x ∈ removed_cells ∧ y ∈ removed_cells → board[x.fst][x.snd] ≠ board[y.fst][y.snd]) →
  List.sum (List.map (List.countp (Option.isSome)) board) = 54 →
  domino_count = 23

theorem max_dominos_after_removal : 
  chessboard 8 →  ∀ removed_count, (removed_cells 8 removed_count → guarantee_dominos_rem 8 23) :=
by
  sorry

end max_dominos_after_removal_l796_796672


namespace flowers_lost_l796_796787

theorem flowers_lost 
  (time_per_flower : ℕ)
  (gathered_time : ℕ) 
  (additional_time : ℕ) 
  (classmates : ℕ) 
  (collected_flowers : ℕ) 
  (total_needed : ℕ)
  (lost_flowers : ℕ) 
  (H1 : time_per_flower = 10)
  (H2 : gathered_time = 120)
  (H3 : additional_time = 210)
  (H4 : classmates = 30)
  (H5 : collected_flowers = gathered_time / time_per_flower)
  (H6 : total_needed = classmates + (additional_time / time_per_flower))
  (H7 : lost_flowers = total_needed - classmates) :
lost_flowers = 3 := 
sorry

end flowers_lost_l796_796787


namespace intersect_sets_l796_796459

section ProofProblem

variables (A B : set ℝ)

def A := {x : ℝ | abs x ≤ 1}
def B := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem intersect_sets : A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by
  sorry -- Proof to be filled in

end ProofProblem

end intersect_sets_l796_796459


namespace arithmetic_sequence_sum_l796_796513

theorem arithmetic_sequence_sum :
  ∀ (a_2 a_3 : ℤ) (S : ℕ → ℤ),
  a_2 = 1 → a_3 = 3 →
  (∀ n, S n = n * (2 * a_1 + (n - 1) * d) / 2) →
  S 4 = 8 :=
by
  intros a_2 a_3 S h_a2 h_a3 h_S
  sorry

end arithmetic_sequence_sum_l796_796513


namespace mean_value_interior_angles_quadrilateral_l796_796118

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796118


namespace wicket_keeper_older_than_captain_l796_796491

-- Define the team and various ages
def captain_age : ℕ := 28
def average_age_team : ℕ := 25
def number_of_players : ℕ := 11
def number_of_remaining_players : ℕ := number_of_players - 2
def average_age_remaining_players : ℕ := average_age_team - 1

theorem wicket_keeper_older_than_captain :
  ∃ (W : ℕ), W = captain_age + 3 ∧
  275 = number_of_players * average_age_team ∧
  216 = number_of_remaining_players * average_age_remaining_players ∧
  59 = 275 - 216 ∧
  W = 59 - captain_age :=
by
  sorry

end wicket_keeper_older_than_captain_l796_796491


namespace f_monotonicity_f_inequality_a_1_l796_796372

-- Definitions of the functions
def f (a : ℝ) (x : ℝ) : ℝ := (1 / 2) * x^2 - a * log x + (1 - a) * x + 1
def g (x : ℝ) : ℝ := x * exp x - log x - x - 1

-- Problem 1: Monotonicity of the function f(x)
theorem f_monotonicity (a : ℝ) :
    (∀ x : ℝ, 0 < x → 0 ≤ a → f a x = (1 / 2) * x^2 - a * log x + (1 - a) * x + 1) ∧
    (∀ x : ℝ, 0 < x → a ≤ 0 → ∀ x' : ℝ, 0 < x' → x ≤ x' → f a x ≤ f a x') ∧
    (∀ x : ℝ, 0 < x → 0 < a → x ≤ a → ∀ x' : ℝ, 0 < x' → x' ≤ a → x ≤ x' → f a x ≥ f a x') ∧
    (∀ x : ℝ, 0 < x → 0 < a → a ≤ x → ∀ x' : ℝ, 0 < x' → a ≤ x' → x ≤ x' → f a x ≤ f a x')
    := sorry

-- Problem 2: Inequality when a = 1
theorem f_inequality_a_1 (x : ℝ) (h1 : 0 < x) :
    let f1 := (λ x => (1 / 2) * x^2 - log x + x)
    f1 x ≤ x * (exp x - 1) + (1 / 2) * x^2 - 2 * log x
    := sorry

end f_monotonicity_f_inequality_a_1_l796_796372


namespace weight_distribution_l796_796518

theorem weight_distribution (x y z : ℕ) 
  (h1 : x + y + z = 100) 
  (h2 : x + 10 * y + 50 * z = 500) : 
  x = 60 ∧ y = 39 ∧ z = 1 :=
by {
  sorry
}

end weight_distribution_l796_796518


namespace find_length_of_smaller_rectangle_l796_796911

theorem find_length_of_smaller_rectangle
  (w : ℝ)
  (h_original : 10 * 15 = 150)
  (h_new_rectangle : 2 * w * w = 150)
  (h_z : w = 5 * Real.sqrt 3) :
  z = 5 * Real.sqrt 3 :=
by
  sorry

end find_length_of_smaller_rectangle_l796_796911


namespace relationship_between_abc_l796_796386

theorem relationship_between_abc 
  (a b c : ℝ) 
  (h1 : log 2 a = 0.3)
  (h2 : 0.3^b = 2)
  (h3 : c = 0.3^2) : 
  a > c ∧ c > b :=
sorry

end relationship_between_abc_l796_796386


namespace joan_total_cents_l796_796783

-- Conditions
def quarters : ℕ := 12
def dimes : ℕ := 8
def nickels : ℕ := 15
def pennies : ℕ := 25

def value_of_quarter : ℕ := 25
def value_of_dime : ℕ := 10
def value_of_nickel : ℕ := 5
def value_of_penny : ℕ := 1

-- The problem statement
theorem joan_total_cents : 
  (quarters * value_of_quarter + dimes * value_of_dime + nickels * value_of_nickel + pennies * value_of_penny) = 480 := 
  sorry

end joan_total_cents_l796_796783


namespace integral_correct_l796_796648

open Real
open BigOperators

noncomputable def integral_value : Real :=
  12.55 * ( ( ∫ t in 0 .. (π / 4), (sin (2 * t) - cos (2 * t)) ^ 2) )

theorem integral_correct :
  integral_value = 0.25 * (π - 2) :=
  by sorry

end integral_correct_l796_796648


namespace sufficient_but_not_necessary_condition_l796_796561

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a^2 ≠ 4) → (a ≠ 2) ∧ ¬ ((a ≠ 2) → (a^2 ≠ 4)) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l796_796561


namespace cafeteria_extra_fruits_l796_796511

def num_apples_red := 75
def num_apples_green := 35
def num_oranges := 40
def num_bananas := 20
def num_students := 17

def total_fruits := num_apples_red + num_apples_green + num_oranges + num_bananas
def fruits_taken_by_students := num_students
def extra_fruits := total_fruits - fruits_taken_by_students

theorem cafeteria_extra_fruits : extra_fruits = 153 := by
  -- proof goes here
  sorry

end cafeteria_extra_fruits_l796_796511


namespace log_sum_zero_l796_796952

theorem log_sum_zero : log 2 + log (1 / 2) = 0 := by
  sorry

end log_sum_zero_l796_796952


namespace mean_value_of_quadrilateral_angles_l796_796006

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l796_796006


namespace csc_315_eq_neg_sqrt_two_l796_796992

theorem csc_315_eq_neg_sqrt_two : Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 :=
by
  sorry

end csc_315_eq_neg_sqrt_two_l796_796992


namespace find_original_function_l796_796721

theorem find_original_function :
  ∀ (f : ℝ → ℝ),
  (∀ x, (λ x, y = f(x)) x = y) →
  (∀ x, y = y × (1:ℝ) = 1 * x) →
  (∀ x, y = y + (-1:ℝ)) →
  (∀ x, y = (y * y)) →
  (∀ x, y = 2 * y) →
  (∀ x, f(x) = (λ x, (1/2) * sin(2 * x - π/2) + 1)) :=
begin
  sorry
end

end find_original_function_l796_796721


namespace solve_for_x_l796_796555

theorem solve_for_x (x y : ℤ) (h1 : x + 2 * y = 10) (h2 : y = 3) : x = 4 := by
  sorry

end solve_for_x_l796_796555


namespace sum_of_three_integers_eq_57_l796_796866

theorem sum_of_three_integers_eq_57
  (a b c : ℕ) (h1: a * b * c = 7^3) (h2: a ≠ b) (h3: b ≠ c) (h4: a ≠ c) :
  a + b + c = 57 :=
sorry

end sum_of_three_integers_eq_57_l796_796866


namespace csc_315_eq_neg_sqrt_2_l796_796997

theorem csc_315_eq_neg_sqrt_2 :
  Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := 
by
  sorry

end csc_315_eq_neg_sqrt_2_l796_796997


namespace exists_projection_less_than_two_over_pi_l796_796837

theorem exists_projection_less_than_two_over_pi (n : ℕ) (a : Fin n → ℝ) (h_sum : (∑ i, a i) = 1) :
  ∃ l : ℝ, (∑ i, |a i * (Real.cos l)|) < (2 / Real.pi) := 
sorry

end exists_projection_less_than_two_over_pi_l796_796837


namespace plot_area_in_acres_l796_796929

-- Definitions based on the problem conditions
def shorter_base_cm : ℝ := 12
def longer_base_cm : ℝ := 18
def height_cm : ℝ := 15
def cm_to_miles : ℝ := 3
def mile_to_acre : ℝ := 640

-- Theorem stating the calculation combines all definitions
theorem plot_area_in_acres :
  let area_cm2 := (1 / 2) * (shorter_base_cm + longer_base_cm) * height_cm,
      area_miles2 := area_cm2 * (cm_to_miles ^ 2),
      area_acres := area_miles2 * mile_to_acre
  in area_acres = 1296000 := by {
  -- All proof steps to be filled here
  sorry
}

end plot_area_in_acres_l796_796929


namespace election_total_polled_votes_l796_796901

theorem election_total_polled_votes (V : ℝ) (invalid_votes : ℝ) (candidate_votes : ℝ) (margin : ℝ)
  (h1 : candidate_votes = 0.3 * V)
  (h2 : margin = 5000)
  (h3 : V = 0.3 * V + (0.3 * V + margin))
  (h4 : invalid_votes = 100) :
  V + invalid_votes = 12600 :=
by
  sorry

end election_total_polled_votes_l796_796901


namespace remainder_of_b_mod_13_l796_796795

theorem remainder_of_b_mod_13 :
  ∃ b : ℕ, ((b ≡ (1/4 + 1/6 + 1/9)⁻¹ [MOD 13]) ∧ b % 13 = 7) :=
sorry

end remainder_of_b_mod_13_l796_796795


namespace monotonic_interval_a1_range_of_a_sum_inequality_l796_796373

-- Problem 1: Prove that f(x) is monotonically increasing on (0, +∞) for a = 1.
theorem monotonic_interval_a1 (x : ℝ) (hx : x > 0) : 
  let f := λ x : ℝ, x - (1 / x) - Real.log x in
  (∀ y : ℝ, y > 0 → f' y > 0) :=
sorry

-- Problem 2: Prove that the range of a such that f(x) > 0 for all x in (1, +∞) is [1/2, +∞).
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x > 1 → (a*x - a/x - Real.log x) > 0) : 
  a ∈ Set.Ici (1 / 2) :=
sorry

-- Problem 3: Prove the inequality for the sum.
theorem sum_inequality (n : ℕ) (hn : 2 ≤ n) :
  ∑ i in Finset.range (n+1), if 2 ≤ i then (1 / (i^2 * Real.log i)) else 0 > (n + 2) * (n-1) /
  2 / n / (n + 1) :=
sorry

end monotonic_interval_a1_range_of_a_sum_inequality_l796_796373


namespace polynomial_solution_l796_796967

theorem polynomial_solution (P : Polynomial ℝ) :
  (∀ x : ℝ, 1 + P.eval x = (1 / 2) * (P.eval (x - 1) + P.eval (x + 1))) →
  ∃ b c : ℝ, ∀ x : ℝ, P.eval x = x^2 + b * x + c := 
sorry

end polynomial_solution_l796_796967


namespace general_formula_a_n_min_length_range_interval_l796_796692

noncomputable theory

open_locale big_operators

/-- Given a geometric sequence {a_n} with the first term 3/2 -/
def a (n : ℕ) : ℝ := (3/2) * (-1/2)^(n-1)

/-- The sum of the first n terms of the sequence {a_n} -/
def S (n : ℕ) : ℝ := ∑ i in finset.range n, a (i+1)

/-- Define b_n as S_n + 1/S_n -/
def b (n : ℕ) : ℝ := S n + 1 / S n

theorem general_formula_a_n :
  ∀ n, ∃ q, q ≠ (0 : ℝ) ∧ a n = (3/2) * q^(n-1) :=
begin
  intro n,
  use [-1/2],
  split,
  { norm_num },
  { dsimp [a], ring },
end

theorem min_length_range_interval :
  ∃ M : set ℝ, (∀ i, b (i+1) ∈ M) ∧ ∀ x y ∈ M, x ≠ y → (|x - y| = 1 / 6) :=
begin
  sorry
end

end general_formula_a_n_min_length_range_interval_l796_796692


namespace find_k_l796_796329

theorem find_k (x y k : ℝ) (h_line : 2 - k * x = -4 * y) (h_point : x = 3 ∧ y = -2) : k = -2 :=
by
  -- Given the conditions that the point (3, -2) lies on the line 2 - kx = -4y, 
  -- we want to prove that k = -2
  sorry

end find_k_l796_796329


namespace projection_of_a_on_b_l796_796359

-- Definitions of conditions
variables {a b : Vector ℝ}
variable (θ : ℝ)
axiom norm_a : ∥a∥ = 2
axiom norm_b : ∥b∥ = 4
axiom angle_ab : θ = (5 * Real.pi) / 6

-- Main statement to prove
theorem projection_of_a_on_b 
  (ha : ∥a∥ = 2)
  (hb : ∥b∥ = 4)
  (hab : θ = (5 * Real.pi) / 6) :
  ∥a∥ * Real.cos θ = -√3 :=
by
  rw [norm_a, norm_b, angle_ab]
  sorry

end projection_of_a_on_b_l796_796359


namespace initial_amount_is_30000_l796_796271

-- Conditions as definitions
def yearly_compounded_amount (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := 
  P * ((1 + r) ^ n)

def half_yearly_compounded_amount (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := 
  P * ((1 + r/2) ^ (2 * n))

def initial_amount_after_difference (diff : ℝ) (half_yearly : ℝ) (yearly : ℝ) : ℝ :=
  diff / (half_yearly - yearly)

-- Given values
def r_yearly := 0.20
def r_half_yearly := 0.20
def n := 2
def difference := 723.0000000000146
def expected_initial_amount := 30000

-- Prove the initial amount P is Rs. 30,000
theorem initial_amount_is_30000 : 
  initial_amount_after_difference difference 1.4641 1.44 = expected_initial_amount := 
by
  sorry

end initial_amount_is_30000_l796_796271


namespace rectangle_k_value_l796_796260

theorem rectangle_k_value (x d : ℝ)
  (h_ratio : ∃ x, ∀ l w, l = 5 * x ∧ w = 4 * x)
  (h_diagonal : ∀ l w, l = 5 * x ∧ w = 4 * x → d^2 = (5 * x)^2 + (4 * x)^2)
  (h_area_written : ∃ k, ∀ A, A = (5 * x) * (4 * x) → A = k * d^2) :
  ∃ k, k = 20 / 41 := sorry

end rectangle_k_value_l796_796260


namespace mean_of_quadrilateral_angles_l796_796153

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796153


namespace percentage_discount_l796_796597

theorem percentage_discount (W R : ℕ) (profit : ℕ) (selling_price discount_amount : ℕ) (D : ℕ)
  (h1 : W = 90) (h2 : R = 120) (h3 : profit = 0.2 * W)
  (h4 : selling_price = W + profit) (h5 : discount_amount = R - selling_price)
  (h6 : D = (discount_amount / R) * 100) :
  D = 10 := 
sorry

end percentage_discount_l796_796597


namespace part1_part2_l796_796717

def f (x a : ℝ) := (2 - a) * (x - 1) - 2 * Real.log x

theorem part1 (a : ℝ) (h1 : a = 1) : 
  (∀ x, 0 < x ∧ x ≤ 2 → f x 1 ≤ f 2 1) ∧ 
  (∀ x, 2 ≤ x → f (x - 1) 1 ≤ f (x - 1) 1) :=
sorry

theorem part2 : 
  (∀ x, 0 < x ∧ x < (1 / 2) → (2 - a) * (x - 1) - 2 * Real.log x > 0) → 
  (∃ m : ℝ, a ≥ m ∧ m = 2 - 4 * Real.log 2) :=
sorry


end part1_part2_l796_796717


namespace error_in_area_l796_796413

-- Define the length and width of the rectangle
variable {L W : ℝ}

-- Define the measured length and width
def L_measured := 1.07 * L
def W_measured := 0.94 * W

-- Define the actual area and the measured (incorrect) area
def actual_area := L * W
def measured_area := L_measured * W_measured

-- Define the error percent in the area
def error_percent := ((measured_area - actual_area) / actual_area) * 100

-- The theorem to prove the error percent is 0.58%
theorem error_in_area : error_percent = 0.58 := by
  sorry

end error_in_area_l796_796413


namespace range_of_m_l796_796478

-- Definition of the propositions and conditions
def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := 1 ≤ m ∧ m ≤ 3
def prop (m : ℝ) : Prop := (¬(p m ∧ q m) ∧ (p m ∨ q m))

-- The proof statement showing the range of m
theorem range_of_m (m : ℝ) : prop m ↔ (1 ≤ m ∧ m ≤ 2) ∨ (m > 3) :=
by
  sorry

end range_of_m_l796_796478


namespace mean_of_quadrilateral_angles_l796_796142

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796142


namespace candle_height_comparison_l796_796525

def first_candle_height (t : ℝ) : ℝ := 10 - 2 * t
def second_candle_height (t : ℝ) : ℝ := 8 - 2 * t

theorem candle_height_comparison (t : ℝ) :
  first_candle_height t = 3 * second_candle_height t → t = 3.5 :=
by
  -- the main proof steps would be here
  sorry

end candle_height_comparison_l796_796525


namespace fair_attendance_l796_796468

theorem fair_attendance (P_this_year P_next_year P_last_year : ℕ) 
  (h1 : P_this_year = 600) 
  (h2 : P_next_year = 2 * P_this_year) 
  (h3 : P_last_year = P_next_year - 200) : 
  P_this_year = 600 ∧ P_next_year = 1200 ∧ P_last_year = 1000 :=
by
  split
  . exact h1
  split
  . rw [h1]
    rw [h2]
    norm_num
  . rw [h1]
    rw [h2]
    rw [h3]
    norm_num

example : fair_attendance 600 1200 1000 :=
by sorry

end fair_attendance_l796_796468


namespace problem_l796_796813

noncomputable def f (x : ℝ) : ℝ := sorry -- We don't define the function explicitly

theorem problem 
  (f : ℝ → ℝ)
  (h_cont : ∀ x ≠ 0, continuous_at f x)
  (h_ineq : ∀ x y ≠ 0, 1 / f (x + y) ≤ (f x + f y) / (f x * f y))
  (h_nonzero : ∀ x ≠ 0, f x ≠ 0)
  (h_limit : tendsto f at_top at_top) :
  ∃ a ≠ 0, ∀ x ≠ 0, f x = a / x :=
sorry

end problem_l796_796813


namespace only_curve_of_constant_width_with_center_of_symmetry_is_circle_l796_796824

noncomputable theory

def curve_of_constant_width (K : Type) (h : ℝ) : Prop :=
∀ (A B : K), ∃ (d : ℝ), (d = h) ∧ ∀ (A' B' : K), (A' ≠ A ∨ B' ≠ B) → (d = h)

def center_of_symmetry (O : Type) (K : Type) : Prop :=
∀ (A B : K), ∃ (A' B' : K), (A' ≠ A ∨ B' ≠ B) → (d = h) 

theorem only_curve_of_constant_width_with_center_of_symmetry_is_circle
  (K : Type) (h : ℝ) (O : Type) 
  (constant_width : curve_of_constant_width K h)
  (center_symmetry : center_of_symmetry O K) :
  ∃ r : ℝ, r = h / 2 ∧ is_circle K r O :=
sorry

end only_curve_of_constant_width_with_center_of_symmetry_is_circle_l796_796824


namespace min_vec_OC_length_l796_796342

variables (OA OB OC : ℝ) (x y : ℝ)

def vec_OA_length : ℝ := 4
def vec_OB_length : ℝ := 2
def angle_AOB : ℝ := 2 * Real.pi / 3
def vec_OC : ℝ := x * vec_OA_length + y * vec_OB_length

def constraint : Prop := x + 2 * y = 1

theorem min_vec_OC_length (h1 : vec_OA_length = 4) (h2 : vec_OB_length = 2)
  (h3 : angle_AOB = 2 * Real.pi / 3) (h4 : constraint) :
  ∃ y : ℝ, (x + 2 * y = 1) ∧ ∀ x y, (vec_OC ^ 2 = (1 - 2 * y) ^ 2 * 16 + 2 * y * (1 - 2 * y) * 2 * 4 * -0.5 + 4 * y ^ 2) → 
  ∃ y : ℝ, y = 3 / 7 ∧ vec_OC = 2 * Real.sqrt 7 / 7 :=
sorry

end min_vec_OC_length_l796_796342


namespace exponential_product_l796_796338

theorem exponential_product (a b : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f(x) = Real.exp x) (h₂ : f(a + b) = 2) : 
  f(2*a) * f(2*b) = 4 := 
by
  sorry

end exponential_product_l796_796338


namespace find_smallest_n_series_l796_796315

def series_sum (n : ℕ) : ℤ :=
  (finset.range (n + 1)).sum (λ i, i * 3 ^ i)

theorem find_smallest_n_series :
  ∃ n, n = 2000 ∧ series_sum n > 3 ^ 2007 :=
by
  sorry

end find_smallest_n_series_l796_796315


namespace percentage_discount_is_10_l796_796593

-- Definition of the wholesale price
def wholesale_price : ℝ := 90

-- Definition of the retail price
def retail_price : ℝ := 120

-- Definition of the profit in terms of percentage of the wholesale price
def profit : ℝ := 0.2 * wholesale_price

-- Definition of the selling price based on the wholesale price and profit
def selling_price : ℝ := wholesale_price + profit

-- Definition of the discount amount
def discount_amount : ℝ := retail_price - selling_price

-- Definition of the percentage discount
def percentage_discount : ℝ := (discount_amount / retail_price) * 100

-- The theorem we want to prove: the percentage discount is 10%
theorem percentage_discount_is_10 : percentage_discount = 10 := 
by
  sorry

end percentage_discount_is_10_l796_796593


namespace even_sum_count_l796_796401

theorem even_sum_count (x y : ℕ) 
  (hx : x = (40 + 42 + 44 + 46 + 48 + 50 + 52 + 54 + 56 + 58 + 60)) 
  (hy : y = ((60 - 40) / 2 + 1)) : 
  x + y = 561 := 
by 
  sorry

end even_sum_count_l796_796401


namespace mean_value_of_quadrilateral_angles_l796_796217

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796217


namespace monthly_payment_amount_l796_796253

theorem monthly_payment_amount :
  ∀ (total_cost down_payment num_payments final_payment: ℝ)
  (interest_rate: ℝ),
  total_cost = 750 →
  down_payment = 300 →
  num_payments = 9 →
  final_payment = 21 →
  interest_rate = 0.18666666666666668 →
  let amount_borrowed := total_cost - down_payment in
  let interest := interest_rate * amount_borrowed in
  let total_paid_back := amount_borrowed + interest in
  ∃ (P : ℝ), 
    9 * P + final_payment = total_paid_back ∧ 
    P = 57 :=
by
  intros total_cost down_payment num_payments final_payment interest_rate
  intro ht hc hn hf hi
  let amount_borrowed := total_cost - down_payment
  let interest := interest_rate * amount_borrowed
  let total_paid_back := amount_borrowed + interest
  use 57
  split
  sorry
  refl

end monthly_payment_amount_l796_796253


namespace problem_l796_796239

-- Given the definitions and conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

-- Statement of the proof problem
theorem problem (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_mono_dec : monotonically_decreasing (λ x, f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 :=
by
  sorry

end problem_l796_796239


namespace arithmetic_square_root_of_5_l796_796532

theorem arithmetic_square_root_of_5 : ∃ x : ℝ, x * x = 5 ∧ x = Real.sqrt 5 :=
by
  have h_sqrt_eq := Real.sqrt_sqr (Real.sqrt 5)
  have h_cond := Real.sqrt_nonneg 5
  use Real.sqrt 5
  exact ⟨h_sqrt_eq, h_cond⟩

end arithmetic_square_root_of_5_l796_796532


namespace find_angle_l796_796810

open Real EuclideanSpace

noncomputable def angle_between_vectors (a b c : ℝ^3) (h_unit_a : ∥a∥ = 1) (h_unit_b : ∥b∥ = 1) (h_unit_c : ∥c∥ = 1) (h_eq : a + 2 • b - 2 • c = 0) : ℝ :=
  Real.arccos (a ⬝ b / (∥a∥ * ∥b∥))

theorem find_angle : ∀ (a b c : ℝ^3),
  ∥a∥ = 1 → ∥b∥ = 1 → ∥c∥ = 1 →
  a + 2 • b - 2 • c = 0 →
  angle_between_vectors a b c = 104.48 :=
by
  intro a b c h_unit_a h_unit_b h_unit_c h_eq
  sorry

end find_angle_l796_796810


namespace frisbee_total_distance_l796_796281

-- Definitions for the conditions
def bess_initial_distance : ℝ := 20
def bess_throws : ℕ := 4
def bess_reduction : ℝ := 0.90
def holly_initial_distance : ℝ := 8
def holly_throws : ℕ := 5
def holly_reduction : ℝ := 0.95

-- Function to calculate the total distance for Bess
def total_distance_bess : ℝ :=
  let distances := List.range bess_throws |>.map (λ i => bess_initial_distance * bess_reduction ^ i)
  (distances.sum) * 2

-- Function to calculate the total distance for Holly
def total_distance_holly : ℝ :=
  let distances := List.range holly_throws |>.map (λ i => holly_initial_distance * holly_reduction ^ i)
  distances.sum

-- Proof statement
theorem frisbee_total_distance : 
  total_distance_bess + total_distance_holly = 173.76 :=
by
  sorry

end frisbee_total_distance_l796_796281


namespace set_C_cannot_form_right_triangle_l796_796943

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem set_C_cannot_form_right_triangle :
  ¬ is_right_triangle 7 8 9 :=
by
  sorry

end set_C_cannot_form_right_triangle_l796_796943


namespace mean_value_of_quadrilateral_angles_l796_796135

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796135


namespace pyramid_surface_area_l796_796292

theorem pyramid_surface_area
(base_edge lateral_edge : ℕ)
(h_base : base_edge = 10)
(h_lateral : lateral_edge = 7) :
  let altitude := √24,
      triangle_area := 10 * altitude / 2 in
  (4 * triangle_area) = 40 * √6 := 
by
  sorry

end pyramid_surface_area_l796_796292


namespace mean_value_of_quadrilateral_interior_angles_l796_796178

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796178


namespace mean_of_quadrilateral_angles_l796_796040

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796040


namespace probability_horizontal_or_vertical_line_symmetry_l796_796935

theorem probability_horizontal_or_vertical_line_symmetry : 
  let grid_points := 100
  let remaining_points := 99
  let lines_of_symmetry := 18
in 
  (lines_of_symmetry / remaining_points) = (2 / 11) := 
by
  sorry

end probability_horizontal_or_vertical_line_symmetry_l796_796935


namespace terminal_velocity_steady_speed_l796_796914

variable (g : ℝ) (t₁ t₂ : ℝ) (a₀ a₁ : ℝ) (v_terminal : ℝ)

-- Conditions
def acceleration_due_to_gravity := g = 10 -- m/s²
def initial_time := t₁ = 0 -- s
def intermediate_time := t₂ = 2 -- s
def initial_acceleration := a₀ = 50 -- m/s²
def final_acceleration := a₁ = 10 -- m/s²

-- Question: Prove the terminal velocity
theorem terminal_velocity_steady_speed 
  (h_g : acceleration_due_to_gravity g)
  (h_t1 : initial_time t₁)
  (h_t2 : intermediate_time t₂)
  (h_a0 : initial_acceleration a₀)
  (h_a1 : final_acceleration a₁) :
  v_terminal = 25 :=
  sorry

end terminal_velocity_steady_speed_l796_796914


namespace number_of_subsets_of_set_l796_796515

theorem number_of_subsets_of_set {α : Type} [fintype α] {n: ℕ} (s : finset α) (h: s.card = n) :
  ∃ (m : ℕ), m = 2 ^ n :=
begin
  use 2 ^ n,
  reflexivity, -- This will conclude that it's equal to the expression provided
end

example : ∃ n, n = 2^(finset.card {0, 1, 2}) :=
begin
  have h_card: (finset.card {0, 1, 2}) = 3 := by refl,
  rw h_card,
  use 8,
  norm_num,
end

end number_of_subsets_of_set_l796_796515


namespace mean_value_of_quadrilateral_angles_l796_796127

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796127


namespace annie_first_passes_bonnie_after_5_laps_l796_796945

-- Start of Lean code

-- Conditions
def track_length : ℝ := 400
def bonnie_speed : ℝ := v
def annie_speed : ℝ := 1.25 * v
def time_to_pass := (400 * 400) / (0.25 * v)

-- Statement
theorem annie_first_passes_bonnie_after_5_laps :
  let bonnie_distance := v * ((400 * 400) / (0.25 * v))
  let annie_distance := 1.25 * v * ((400 * 400) / (0.25 * v))
  annie_distance / track_length = 5 :=
by
  -- The proof will be filled in here
  sorry

end annie_first_passes_bonnie_after_5_laps_l796_796945


namespace probability_of_x_plus_y_le_5_l796_796258

theorem probability_of_x_plus_y_le_5 :
  (∫ x in 0..4, ∫ y in 0.. 4, ite (x + y ≤ 5) 1 0) / (∫ x in 0..4, ∫ y in 0..4, 1) = 1 / 4 :=
sorry

end probability_of_x_plus_y_le_5_l796_796258


namespace p_plus_q_correct_l796_796857

noncomputable def p := λ x : ℝ, ((8 / 3) * x) + 2
noncomputable def q := λ x : ℝ, ((1 / 3) * (x + 3) * (x - 1) * (x - 2))

theorem p_plus_q_correct : ∀ x : ℝ, p x + q x = (1 / 3) * x^3 - (1 / 3) * x^2 + (11 / 3) * x + 4 := 
by {
    intro x,
    sorry
}

end p_plus_q_correct_l796_796857


namespace range_of_m_l796_796731

variable (m : ℝ)

def p := ∀ x : ℝ, sin x + cos x > m
def q := ∀ x : ℝ, (2 * m^2 - m)^x > (2 * m^2 - m)^(x - 1)

theorem range_of_m :
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ 
  m ∈ (-∞, -√2) ∪ (-√2, -1/2) ∪ (1, ∞) := 
by
  sorry

end range_of_m_l796_796731


namespace Ram_Shyam_weight_ratio_l796_796874

theorem Ram_Shyam_weight_ratio :
  ∃ (R S : ℝ), 
    (1.10 * R + 1.21 * S = 82.8) ∧ 
    (1.15 * (R + S) = 82.8) ∧ 
    (R / S = 1.20) :=
by {
  sorry
}

end Ram_Shyam_weight_ratio_l796_796874


namespace problem_statement_l796_796696

open Set

def A : Set ℝ := {x | x^2 ≥ 4}
def B : Set ℝ := {y | ∃ x : ℝ, y = abs (tan x)}

theorem problem_statement : 
  ((univ \ A) ∩ B) = {x | 0 ≤ x ∧ x < 2} := 
by 
  sorry

end problem_statement_l796_796696


namespace regular_tetrahedron_angles_constant_l796_796275

-- Definitions based on conditions
variables {α β λ : ℝ}
def is_regular_tetrahedron (ABCD : set (ℝ × ℝ × ℝ)) : Prop := sorry -- Formal definition of the regular tetrahedron
def on_edge (E point A B : (ℝ × ℝ × ℝ)) (λ : ℝ) : Prop := sorry -- Formal definition for point E on edge AB
def angle (E F C D : (ℝ × ℝ × ℝ)) : ℝ := sorry -- Formal definition to compute the needed angles

-- Given problem conditions and result
theorem regular_tetrahedron_angles_constant (A B C D E F : (ℝ × ℝ × ℝ)) (h_tetra : is_regular_tetrahedron {A, B, C, D})
  (h_ratio1 : on_edge E A B λ) (h_ratio2 : on_edge F C D λ) (h_lambda : 0 < λ) 
  : let α := angle E F A C in 
    let β := angle E F B D in 
    α + β = π :=
by
  sorry

end regular_tetrahedron_angles_constant_l796_796275


namespace eccentricity_of_hyperbola_l796_796457

noncomputable def hyperbolaData : Type := 
  {P : ℝ × ℝ, F1 : ℝ × ℝ, F2 : ℝ × ℝ, a : ℝ, b : ℝ // a > 0 ∧ b > 0}

noncomputable def hyperbolaEccentricity (data : hyperbolaData) : ℝ := 
  let ⟨P, F1, F2, a, b, hab⟩ := data
  let PF1 := P - F1
  let PF2 := P - F2
  let dot_product := PF1.1 * PF2.1 + PF1.2 * PF2.2
  let angle_tan := (PF1.2 * PF2.1 - PF1.1 * PF2.2) / (PF1.1 * PF2.1 + PF1.2 * PF2.2)
  if (dot_product = 0) ∧ (angle_tan = 2 / 3) then 
    sqrt 13 
  else 
    0 -- default case which should not occur

theorem eccentricity_of_hyperbola (data : hyperbolaData) :
  hyperbolaEccentricity data = sqrt 13 :=
  sorry

end eccentricity_of_hyperbola_l796_796457


namespace find_six_numbers_l796_796307

theorem find_six_numbers
  (a1 a2 a3 a4 a5 a6 : ℕ)
  (h_distinct : list.nodup [a1, a2, a3, a4, a5, a6])
  (h_set : {a1, a2, a3, a4, a5, a6} = {27720, 55440, 83160, 110880, 138600, 166320})
  (h_divisibility : ∀ (x y : ℕ),
    x ∈ [a1, a2, a3, a4, a5, a6] → y ∈ [a1, a2, a3, a4, a5, a6] → x ≠ y → (x * y) % (x + y) = 0) :
  true :=
sorry

end find_six_numbers_l796_796307


namespace max_n_ellipse_sequence_l796_796963

theorem max_n_ellipse_sequence :
  ∃ n : ℕ, 
  let a := 2
  let c := 1
  let P_nF := (n : ℕ) → ℕ 
  let d := P_nF (n+1) - P_nF n
  let min_dist := a - c
  let max_dist := a + c
  (n ≤ 201) ∧ (∀ n', d ≥ 1/100) :=
sorry

end max_n_ellipse_sequence_l796_796963


namespace sufficient_condition_B_is_proper_subset_of_A_l796_796460

def A : Set ℝ := {x | x^2 + x = 6}
def B (m : ℝ) : Set ℝ := {-1 / m}

theorem sufficient_condition_B_is_proper_subset_of_A (m : ℝ) : 
  m = -1/2 → B m ⊆ A ∧ B m ≠ A :=
by
  sorry

end sufficient_condition_B_is_proper_subset_of_A_l796_796460


namespace remainder_when_n_add_3006_divided_by_6_l796_796367

theorem remainder_when_n_add_3006_divided_by_6 (n : ℤ) (h : n % 6 = 1) : (n + 3006) % 6 = 1 := by
  sorry

end remainder_when_n_add_3006_divided_by_6_l796_796367


namespace angle_between_line_and_plane_l796_796581

-- Definitions for points and the line.
def point1 : ℝ × ℝ × ℝ := (2, 1, -1)
def point2 : ℝ × ℝ × ℝ := (5/2, 1/4, -5/4)

-- Definition for the plane.
def plane1 (x y z : ℝ) : Prop := x + 2*y - z + 3 = 0

-- Vector dot product for angle calculation
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Vector length calculation for angle calculation
def vector_length (v : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (v.1*v.1 + v.2*v.2 + v.3*v.3)

-- Angle computation between two vectors
def angle_between_vectors (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
Real.arcsin (dot_product v1 v2 / (vector_length v1 * vector_length v2))

-- Vectors from points
def vector1 : ℝ × ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2, point2.3 - point1.3)
def vector2 : ℝ × ℝ × ℝ := (1, 2, -1)  -- Normal vector to the plane x + 2y - z + 3 = 0

-- Final angle in degrees (approximating arc sine value)
noncomputable def angle_formed : ℝ :=
angle_between_vectors vector1 vector2 * (180 / Real.pi)

-- Proof in Lean
theorem angle_between_line_and_plane (h : plane1 (5/2) (1/4) (-5/4)) : 
angle_formed ≈ 19 + 7 / 60 := 
by
  sorry

end angle_between_line_and_plane_l796_796581


namespace mean_value_interior_angles_quadrilateral_l796_796112

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796112


namespace mutually_exclusive_events_l796_796403

/-- There are 5 ballpoint pens in a box, including 3 first-grade pens and 2 second-grade pens. Pens are of the same size and quality. 3 pens are randomly selected from the box. Define the events:
A: Selecting at least 2 first-grade pens from the 3 selected pens.
B: Selecting at most 1 second-grade pen from the 3 selected pens.
C: Selecting both first-grade and second-grade pens from the 3 selected pens.
D: Selecting no second-grade pen from the 3 selected pens.
Prove that events A, B, D are mutually exclusive with the event of selecting 1 first-grade pen and 2 second-grade pens.
-/
theorem mutually_exclusive_events (p1 p2 p3 p4 p5 : ℕ) (h_condition1 : (p1, p2, p3, p4, p5) = (3, 3, 1, 2, 2)) :
  (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) (h_condition2 : (A, B, C, D) = (2, 1, 0, 3)) :
  (mutually_exclusive (A) (select_f2_sg3 (p2, p4)) → mutually_exclusive (B) (select_f2_sg3 (p2, p4)) → mutually_exclusive (D) (select_f2_sg3 (p2, p4))) :=
begin
  sorry
end

end mutually_exclusive_events_l796_796403


namespace smallest_n_l796_796685

theorem smallest_n (n : ℕ) (h1 : n > 4) 
  (A : Finset ℕ) (hA : A = Finset.range n) 
  (m : ℕ) (A_i : Fin m → Finset ℕ) 
  (h_union : (Finset.bUnion Finset.univ A_i) = A) 
  (h_size : ∀ i, (A_i i).card = 4) 
  (h_unique : ∀ X ∈ (A.powersetLen 2), ∃! j, X ⊆ A_i j) 
  : n = 13 :=
by
  sorry

end smallest_n_l796_796685


namespace prairie_total_area_l796_796579

theorem prairie_total_area :
  let dust_covered := 64535
  let untouched := 522
  (dust_covered + untouched) = 65057 :=
by {
  let dust_covered := 64535
  let untouched := 522
  trivial
}

end prairie_total_area_l796_796579


namespace factorize_cubic_expression_l796_796651

theorem factorize_cubic_expression (m : ℝ) : 
  m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end factorize_cubic_expression_l796_796651


namespace boys_in_other_communities_l796_796552

theorem boys_in_other_communities:
  let total_boys := 850
  let percentage_muslims := 0.4
  let percentage_hindus := 0.28
  let percentage_sikhs := 0.1
  let percentage_other := 1 - (percentage_muslims + percentage_hindus + percentage_sikhs)
  let number_other := percentage_other * total_boys in
  number_other = 187 :=
by
  sorry

end boys_in_other_communities_l796_796552


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796062

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796062


namespace mean_of_quadrilateral_angles_l796_796198

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796198


namespace probability_defective_third_draw_l796_796683

-- Define the problem conditions
def initial_total_products : ℕ := 10
def defective_products : ℕ := 3
def draws : ℕ := 3

-- Define the problem statement
theorem probability_defective_third_draw :
  (first_draw_defective : true) -- condition implied: the first draw is defective
  ∧ (total_products : ℕ = initial_total_products)
  ∧ (defective : ℕ = defective_products)
  ∧ (draw_count : ℕ = draws) →
  (prob_defective_third_draw : ℚ = 2/9) := 
sorry

end probability_defective_third_draw_l796_796683


namespace gavrila_travel_distance_l796_796674

-- Declaration of the problem conditions
noncomputable def smartphone_discharge_rate_video : ℝ := 1 / 3
noncomputable def smartphone_discharge_rate_tetris : ℝ := 1 / 5
def average_speed_first_half : ℝ := 80
def average_speed_second_half : ℝ := 60

-- Main theorem statement
theorem gavrila_travel_distance : ∃ d : ℝ, d = 257 := by
  -- Solve for the total travel time t
  let smartphone_discharge_time : ℝ := 15 / 4
  -- Expressing the distance formula
  let distance_1 := smartphone_discharge_time * average_speed_first_half
  let distance_2 := smartphone_discharge_time * average_speed_second_half
  let total_distance := (distance_1 / 2 + distance_2 / 2)
  -- Assert the final distance
  exact ⟨257, by decide⟩

end gavrila_travel_distance_l796_796674


namespace log_a_min_value_iff_l796_796754

noncomputable def log_function_has_min_value (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - a * x + 1 > 0 ∧ a > 0 ∧ a ≠ 1 ∧ (∃ y : ℝ, ∀ x : ℝ, y ≤ log a (x^2 - a * x + 1))

theorem log_a_min_value_iff (a : ℝ) : log_function_has_min_value a ↔ 1 < a ∧ a < 2 := 
by
  sorry

end log_a_min_value_iff_l796_796754


namespace cat_roaming_area_l796_796465

-- Define conditions
constant leash_length : ℝ := 15
constant side_length : ℝ := 10
constant midpoint_area : ℝ := (1 / 2) * Real.pi * leash_length ^ 2
constant corner_area : ℝ := (3 / 4) * Real.pi * leash_length ^ 2 + (1 / 4) * Real.pi * 5 ^ 2

-- Problem statement
theorem cat_roaming_area :
  corner_area - midpoint_area = 62.5 * Real.pi := by
  sorry

end cat_roaming_area_l796_796465


namespace woman_swim_upstream_distance_l796_796613

variable (v c d : ℝ)

-- Conditions
def speed_of_current := c = 2.5
def downstream_distance := (v + c) * 9 = 81
def upstream_distance := (v - c) * 9 = d

-- Key Theorem to Prove
theorem woman_swim_upstream_distance :
  speed_of_current → downstream_distance → upstream_distance → d = 36 :=
by
  intro h_speed h_downstream h_upstream
  -- The actual proof steps go here
  sorry

end woman_swim_upstream_distance_l796_796613


namespace exists_a_star_b_eq_a_l796_796825

variable {S : Type*} [CommSemigroup S]

def exists_element_in_S (star : S → S → S) : Prop :=
  ∃ a : S, ∀ b : S, star a b = a

theorem exists_a_star_b_eq_a
  (star : S → S → S)
  (comm : ∀ a b : S, star a b = star b a)
  (assoc : ∀ a b c : S, star (star a b) c = star a (star b c))
  (exists_a : ∃ a : S, star a a = a) :
  exists_element_in_S star := sorry

end exists_a_star_b_eq_a_l796_796825


namespace value_of_f_AT_l796_796854

noncomputable def f (x : ℝ) : ℝ := (1 - Real.cos (2 * x)) * (Real.cos x) ^ 2

theorem value_of_f_AT : 
  let A := 1 / 2 in 
  let T := Real.pi / 2 in 
  f (A * T) = 1 / 2 :=
by 
  sorry

end value_of_f_AT_l796_796854


namespace president_wins_the_game_l796_796875

noncomputable def pres_wins_game : Prop :=
  ∀ G : SimpleGraph (Fin 2010), ∀ color : Edge G → Fin 3,
    (∀ v : V G, (G.degree v = 3)) →
    (∃ v : V G, ∃ c1 c2 c3 : Fin 3,
      c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
      ∃ (e1 e2 e3 : Edge G), 
        e1 ∈ G.incidence_set v ∧ e2 ∈ G.incidence_set v ∧ e3 ∈ G.incidence_set v ∧
        color e1 = c1 ∧ color e2 = c2 ∧ color e3 = c3)

theorem president_wins_the_game : pres_wins_game :=
sorry

end president_wins_the_game_l796_796875


namespace product_value_l796_796649

theorem product_value :
  (∏ k in Finset.range 149, (1 - (1 / (k + 2 : ℝ)))^2) = 1 / 22500 := 
sorry

end product_value_l796_796649


namespace transfer_increases_averages_l796_796907

variables (score_A : ℝ ≥ 0)
variables (score_B : ℝ ≥ 0)

def initial_avg_A := 47.2
def initial_avg_B := 41.8
def initial_size_A := 10
def initial_size_B := 10
def total_score_A := initial_avg_A * initial_size_A
def total_score_B := initial_avg_B * initial_size_B

variables (score_L : ℝ) (score_F : ℝ)

def cond_Lopatin := 41.8 < score_L ∧ score_L < 47.2
def cond_Filin := 41.8 < score_F ∧ score_F < 47.2

def new_avg_A_after_both_transferred := (total_score_A - score_L - score_F) / (initial_size_A - 2)
def new_avg_B_after_both_transferred := (total_score_B + score_L + score_F) / (initial_size_B + 2)

theorem transfer_increases_averages :
  cond_Lopatin →
  cond_Filin →
  new_avg_A_after_both_transferred > initial_avg_A ∧
  new_avg_B_after_both_transferred > initial_avg_B :=
sorry

end transfer_increases_averages_l796_796907


namespace sum_and_count_even_l796_796402

-- Sum of integers from a to b (inclusive)
def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

-- Number of even integers from a to b (inclusive)
def count_even_integers (a b : ℕ) : ℕ :=
  ((b - if b % 2 == 0 then 0 else 1) - (a + if a % 2 == 0 then 0 else 1)) / 2 + 1

theorem sum_and_count_even (x y : ℕ) :
  x = sum_of_integers 20 40 →
  y = count_even_integers 20 40 →
  x + y = 641 :=
by
  intros
  sorry

end sum_and_count_even_l796_796402


namespace hypotenuse_length_l796_796476

variables {α : Type} [OrderedField α]

theorem hypotenuse_length (h x y BC : α)
  (h_hyp : ∀ (A B C : α), ∃ (D : α), (AD = h) ∧ (CD - BD = h) ∧ (AD ⊥ BC))
  (geo_mean : h^2 = x * y)
  (diff : x - y = h)
  (BC_val : BC = x + y) :
  BC = h * Real.sqrt 5 :=
by sorry

end hypotenuse_length_l796_796476


namespace max_min_values_of_function_l796_796313

def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 3 * (Real.sin x)

theorem max_min_values_of_function :
  let interval := Set.Icc (-Real.pi / 2) (Real.pi / 2) in
  (∀ x ∈ interval, f x ≤ 25 / 8) ∧
  (∃ x ∈ interval, f x = 25 / 8) ∧
  (∀ x ∈ interval, f x ≥ -3) ∧
  (∃ x ∈ interval, f x = -3) := by
  sorry

end max_min_values_of_function_l796_796313


namespace csc_315_eq_neg_sqrt_2_l796_796996

theorem csc_315_eq_neg_sqrt_2 :
  Real.csc (315 * Real.pi / 180) = -Real.sqrt 2 := 
by
  sorry

end csc_315_eq_neg_sqrt_2_l796_796996


namespace mean_of_quadrilateral_angles_l796_796199

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796199


namespace sqrt_expression_value_l796_796318

theorem sqrt_expression_value (w : ℝ) (h_w : w = 0.49) :
  (sqrt 1.21 / sqrt 0.81) + (sqrt 1.00 / sqrt w) = 2.650793650793651 := by
  sorry

end sqrt_expression_value_l796_796318


namespace planning_committee_selection_l796_796266

theorem planning_committee_selection (n : ℕ) (h : n * (n - 1) / 2 = 6) : nat.choose 4 4 = 1 := 
by {
  have h1 : n * (n - 1) = 12,
  { sorry },  -- Intermediate step of multiplying both sides by 2
  have h2 : n = 4,
  { sorry },  -- Solving quadratic equation n^2 - n - 12 = 0
  rw h2,
  apply nat.choose_self,
}

end planning_committee_selection_l796_796266


namespace slices_needed_l796_796831

def slices_per_sandwich : ℕ := 3
def number_of_sandwiches : ℕ := 5

theorem slices_needed : slices_per_sandwich * number_of_sandwiches = 15 :=
by {
  sorry
}

end slices_needed_l796_796831


namespace statement_B_is_algorithm_l796_796897

def is_algorithm (statement : String) : Prop := 
  statement = "Cooking rice involves the steps of washing the pot, rinsing the rice, adding water, and heating."

def condition_A : String := "At home, it is generally the mother who cooks."
def condition_B : String := "Cooking rice involves the steps of washing the pot, rinsing the rice, adding water, and heating."
def condition_C : String := "Cooking outdoors is called a picnic."
def condition_D : String := "Rice is necessary for cooking."

theorem statement_B_is_algorithm : is_algorithm condition_B :=
by
  sorry

end statement_B_is_algorithm_l796_796897


namespace mean_value_of_quadrilateral_interior_angles_l796_796182

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796182


namespace determine_x_l796_796293

theorem determine_x (x : ℝ) (hx_pos : 0 < x) (eq_xfx : x * floor x = 90) (bounds_x : 9 ≤ x ∧ x < 10) : x = 10 := by
  sorry

end determine_x_l796_796293


namespace charlie_coins_l796_796747

variables (a c : ℕ)

axiom condition1 : c + 2 = 5 * (a - 2)
axiom condition2 : c - 2 = 4 * (a + 2)

theorem charlie_coins : c = 98 :=
by {
    sorry
}

end charlie_coins_l796_796747


namespace add_fractions_l796_796951

theorem add_fractions :
  (8:ℚ) / 19 + 5 / 57 = 29 / 57 :=
sorry

end add_fractions_l796_796951


namespace mean_value_interior_angles_quadrilateral_l796_796116

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796116


namespace mean_value_of_quadrilateral_angles_l796_796209

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796209


namespace feeding_ways_l796_796603

-- Definitions
def num_pairs : ℕ := 5

-- Conditions
def starts_with_male_lion : Prop := true

def no_two_same_gender_consecutively (sequence : list (ℕ × char)) : Prop :=
  ∀ (i : ℕ), i < sequence.length - 1 → (sequence[i].2 ≠ sequence[i+1].2)

-- The final statement
theorem feeding_ways : 
  ∃ sequence : list (ℕ × char), 
    length sequence = 2 * num_pairs ∧ 
    starts_with_male_lion ∧ 
    no_two_same_gender_consecutively sequence ∧ 
    (sequence.head = (1, 'M')) ∧
    sequence.foldl (λ acc x, acc + fst x) 1 = 2880 :=
sorry

end feeding_ways_l796_796603


namespace closest_whole_number_to_ratio_l796_796303

theorem closest_whole_number_to_ratio :
  let ratio := (2^3000 + 2^3003) / (2^3001 + 2^3002)
  | ratio - 2 | < | ratio - 1 | :=
sorry

end closest_whole_number_to_ratio_l796_796303


namespace mean_value_of_interior_angles_of_quadrilateral_l796_796022

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l796_796022


namespace smallest_AC_l796_796961

-- Definitions of the given conditions
variable (ABC : Triangle)
variable (AB AC BD CD AD : ℝ)
variable (D : Point)
variable (h1 : AB = AC)
variable (h2 : D ∈ LineSegment AC)
variable (h3 : (BD⊥AC))
variable (h4 : is_int AC)
variable (h5 : is_int CD)
variable (h6 : BD^2 = 57)

-- Proof statement for the smallest possible value of AC == 11
theorem smallest_AC {AC : ℝ} (h1 : AB = AC) (h2 : D ∈ LineSegment AC)
                    (h3 : BD⊥AC) (h4 : is_int AC) (h5 : is_int CD) (h6 : BD^2 = 57) :
                    AC = 11 :=
by
    sorry

end smallest_AC_l796_796961


namespace csc_315_equals_neg_sqrt_2_l796_796979

-- Define the conditions
def csc (x : Real) : Real := 1 / Real.sin x
axiom sin_subtraction_360 (x : Real) : Real.sin (360 - x) = -Real.sin x

-- State the problem
theorem csc_315_equals_neg_sqrt_2 : csc 315 = -Real.sqrt 2 := by
  sorry

end csc_315_equals_neg_sqrt_2_l796_796979


namespace mean_value_of_quadrilateral_angles_l796_796137

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796137


namespace PQ_fixed_point_l796_796711

def ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

theorem PQ_fixed_point (a b e : ℝ) (ha : a > 0) (hb : b > 0)
  (h_cond : a > b) (h_ecc : e = 1 / 2)
  (h_ellipse : ellipse a b (sqrt 3, sqrt 3 / 2))
  (P Q A: ℝ × ℝ) (hA: A = (-2, 0))
  (hP : P ∈ ellipse 2 (sqrt 3)) (hQ : Q ∈ ellipse 2 (sqrt 3))
  (h_perp : ∃ l1 l2 : ℝ → ℝ, (∀ x, l1 x = x + 2) ∧ (∀ x, l2 x = -x + 2) ∧ 
             (l1 = λ x, l2 x)) :
  ∃ M : ℝ × ℝ, M = (-2/7, 0) :=
sorry

end PQ_fixed_point_l796_796711


namespace compute_expression_l796_796632

theorem compute_expression : 11 * (1 / 17) * 34 = 22 := 
sorry

end compute_expression_l796_796632


namespace eqn_has_real_root_in_interval_l796_796394

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + x - 3

theorem eqn_has_real_root_in_interval (k : ℤ) :
  (∃ (x : ℝ), x > k ∧ x < (k + 1) ∧ f x = 0) → k = 2 :=
by
  sorry

end eqn_has_real_root_in_interval_l796_796394


namespace range_of_tangent_line_inclination_l796_796350

noncomputable def curve (x : ℝ) : ℝ := x^3 - real.sqrt 3 * x + 3 / 5

noncomputable def derivative (x : ℝ) : ℝ := 3 * x^2 - real.sqrt 3

theorem range_of_tangent_line_inclination :
  ∀ (x : ℝ),
    let α := real.arctan (derivative x) in
    (0 ≤ α ∧ α < real.pi / 2) ∨
    (2 * real.pi / 3 ≤ α ∧ α < real.pi) :=
sorry

end range_of_tangent_line_inclination_l796_796350


namespace mean_of_quadrilateral_angles_is_90_l796_796098

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796098


namespace sixth_diagram_shaded_fraction_l796_796604

theorem sixth_diagram_shaded_fraction :
  ∀ {n : ℕ}, (n ≥ 6) →
  let shaded_tris := (n - 1) ^ 2 in
  let total_tris := n ^ 2 in
  shaded_tris / total_tris = (25 : ℚ) / 36 :=
by
  intros n hn
  have hn_eq : n = 6 := sorry
  rw [hn_eq]
  have shaded_tris_eq : (n - 1) ^ 2 = 25 := by norm_num
  have total_tris_eq : n ^ 2 = 36 := by norm_num
  rw [shaded_tris_eq, total_tris_eq]
  norm_num
  sorry

end sixth_diagram_shaded_fraction_l796_796604


namespace find_b_of_hyperbola_l796_796775

theorem find_b_of_hyperbola (a b : ℝ) (h : 0 < a) (k : 0 < b)
  (h1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) = (2, 0))
  (h2 : ∀ m x1 y1 c, abs (m * x1 - y1 + c) / √(m^2 + 1) = √2 ↔ d = 2) :
  b = 2 :=
sorry

end find_b_of_hyperbola_l796_796775


namespace percentage_discount_l796_796595

theorem percentage_discount (W R : ℕ) (profit : ℕ) (selling_price discount_amount : ℕ) (D : ℕ)
  (h1 : W = 90) (h2 : R = 120) (h3 : profit = 0.2 * W)
  (h4 : selling_price = W + profit) (h5 : discount_amount = R - selling_price)
  (h6 : D = (discount_amount / R) * 100) :
  D = 10 := 
sorry

end percentage_discount_l796_796595


namespace height_from_A_to_BC_median_from_B_to_AC_l796_796348

structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := ⟨4, 0⟩
def B : Point := ⟨6, 6⟩
def C : Point := ⟨0, 2⟩

noncomputable def height_eq : ℝ → ℝ → Prop := 
λ x y, 3 * x + 2 * y - 12 = 0

noncomputable def median_eq : ℝ → ℝ → Prop := 
λ x y, x + 2 * y - 18 = 0

theorem height_from_A_to_BC :
  ∃ x y, height_eq x y :=
sorry

theorem median_from_B_to_AC :
  ∃ x y, median_eq x y :=
sorry

end height_from_A_to_BC_median_from_B_to_AC_l796_796348


namespace mean_of_quadrilateral_angles_l796_796145

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l796_796145


namespace mean_of_quadrilateral_angles_l796_796190

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796190


namespace sequence_sum_expression_l796_796316

theorem sequence_sum_expression (n : ℕ) :
  2 * (∑ (i : ℕ) in Finset.range (n+1), ∑ (j : ℕ) in Finset.range (n+1), if i < j then i * j else 0) =
  n * (n + 1) * (3 * n^2 - n - 2) / 12 := 
by 
  sorry

end sequence_sum_expression_l796_796316


namespace limit_eval_l796_796233

noncomputable def limit_problem := sorry

theorem limit_eval :
  (∀ x, 0 < x) →
  limit (λ x, (1 - 2^(4 - x^2)) / (2 * (sqrt (2 * x) - sqrt(3 * x^2 - 5 * x + 2)))) 2 = -8 * real.log 2 / 5 :=
begin
  sorry
end

end limit_eval_l796_796233


namespace find_S_10_l796_796699

variable (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Definition for a geometric sequence composed of positive numbers.
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ q > 0, ∀ n, a (n + 1) = a n * q

-- Definition for the sum of the first n terms of a sequence.
def sum_sequence (S a : ℕ → ℕ) : Prop :=
  ∀ n, S n = (∑ i in Finset.range n, a (i + 1))

-- Problem Conditions
axiom a1_pos : a 1 = 3
axiom product_cond : a 2 * a 4 = 144

-- Proof Statement
theorem find_S_10 : geometric_sequence a → sum_sequence S a → S 10 = 3069 := by
  sorry

end find_S_10_l796_796699


namespace fourth_root_of_x_cbrt_of_x_l796_796276

theorem fourth_root_of_x_cbrt_of_x (x : ℝ) (hx : 0 < x) : 
  real.sqrt (real.sqrt (x * real.cbrt x)) = x^(1/3) :=
sorry

end fourth_root_of_x_cbrt_of_x_l796_796276


namespace intersections_distance_l796_796502

theorem intersections_distance (y : Real) (m n : ℕ) 
  (h₀ : ∀ x : Real, x^2 ≠ 0 → n * 4 = x^2 ∧ m = 65 ∧ n = 4)
  (hn_coprime : Nat.gcd m n = 1)
  (h_distance : y = Real.abs (Real.sqrt 65) / 4) 
  : m - n = 61 :=
by 
  sorry

end intersections_distance_l796_796502


namespace mean_value_of_quadrilateral_angles_l796_796126

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796126


namespace find_ZW_length_l796_796794

variable {X Y Z W : Type}
variable [MetricSpace X] [MetricSpace Y] [MetricSpace Z] [MetricSpace W]

variable (triangle : Triangle XYZ)
variable (right_angle_at_Y : triangle.Angle Y = 90)
variable (circle : Circle Y Z)
variable (diameter_YZ : circle.diameter = segment Y Z)
variable (intersects_at_W : circle.intersects (segment X Z) at W)
variable (distance_XW : distance X W = 2)
variable (distance_YW : distance Y W = 3)

theorem find_ZW_length :
  distance Z W = 4.5 :=
sorry

end find_ZW_length_l796_796794


namespace cars_sold_l796_796244

theorem cars_sold (sales_Mon sales_Tue sales_Wed cars_Thu_Fri_Sat : ℕ) 
  (mean : ℝ) (h1 : sales_Mon = 8) 
  (h2 : sales_Tue = 3) 
  (h3 : sales_Wed = 10) 
  (h4 : mean = 5.5) 
  (h5 : mean * 6 = sales_Mon + sales_Tue + sales_Wed + cars_Thu_Fri_Sat):
  cars_Thu_Fri_Sat = 12 :=
sorry

end cars_sold_l796_796244


namespace min_distance_to_line_l796_796705

noncomputable def f (x : ℝ) : ℝ := 2 * f (2 - x) - x^2 + 8 * x - 8

def min_distance : ℝ := (4 * Real.sqrt 5) / 5

theorem min_distance_to_line : 
  ∃ x : ℝ, (∀ x : ℝ, f x = x^2) → 
  ∀ p : ℝ × ℝ, p ∈ { (x, f x) | x : ℝ } → 
  distance p (simpler_line : ℝ × ℝ) = min_distance :=
sorry

end min_distance_to_line_l796_796705


namespace mean_value_interior_angles_quadrilateral_l796_796117

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796117


namespace problem_l796_796551

theorem problem (w : ℝ) (h : 13 = 13 * w / (1 - w)) : w^2 = 1 / 4 :=
by
  sorry

end problem_l796_796551


namespace digits_equiv_l796_796482

theorem digits_equiv (n : ℕ) (k : ℕ) (h : n ≥ k) : 
  (number_of_digits (2^n + 1974^n) = number_of_digits (1974^n)) :=
sorry

end digits_equiv_l796_796482


namespace convert_euler_to_rectangular_form_l796_796636

noncomputable def convert_to_rectangular_form (x : ℂ) : ℂ :=
  let θ := 13 * Real.pi / 3
  in complex.exp (θ * complex.I)

theorem convert_euler_to_rectangular_form :
  convert_to_rectangular_form (13 * Real.pi / 3 * complex.I) = (1/2 : ℂ) + (complex.I * (Real.sqrt 3 / 2)) :=
sorry

end convert_euler_to_rectangular_form_l796_796636


namespace quadratic_inequality_solution_l796_796708

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : a < 0)
  (h2 : (∀ x, ax^2 + bx + c = 0 ↔ x = 1 ∨ x = 3)) : 
  ∀ x, cx^2 + bx + a > 0 ↔ (1/3 < x ∧ x < 1) :=
by
  sorry

end quadratic_inequality_solution_l796_796708


namespace no_valid_n_l796_796327

theorem no_valid_n (n : ℕ) : (100 ≤ n / 4 ∧ n / 4 ≤ 999) → (100 ≤ 4 * n ∧ 4 * n ≤ 999) → false :=
by
  intro h1 h2
  sorry

end no_valid_n_l796_796327


namespace mean_of_quadrilateral_angles_l796_796188

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796188


namespace runners_meet_l796_796671

theorem runners_meet (v1 v2 v3 v4 : ℕ) (L : ℕ) (t : ℕ) :
  v1 = 5 ∧ v2 = 5.5 ∧ v3 = 6 ∧ v4 = 6.5 ∧ L = 600 ∧
  (∀ t1, (5 * t1) % L = (5.5 * t1) % L ∧
         (5.5 * t1) % L = (6 * t1) % L ∧
         (6 * t1) % L = (6.5 * t1) % L) →
  t = 1200 :=
by
  sorry

end runners_meet_l796_796671


namespace measure_minor_arc_MN_l796_796760

-- Define point, circle, angle, and measure in a Euclidean geometry context.
variable {O A M N : Point}
variable {circle : Circle O}
variable (on_circle_M : M ∈ circle)
variable (on_circle_N : N ∈ circle)
variable (center_A : A = O)
variable (angle_MAN : Angle A M N)
variable (angle_MAN_measure : angle_MAN = 60)

theorem measure_minor_arc_MN : measure (arc M N) = 60 :=
by
  sorry

end measure_minor_arc_MN_l796_796760


namespace range_of_a_l796_796355

variable {a : ℝ}
def P : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0 
def Q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0 

theorem range_of_a (h : (P ∨ Q) ∧ ¬(P ∧ Q)) : (-1 < a ∧ a ≤ 1) ∨ (a ≥ 2) :=
  sorry

end range_of_a_l796_796355


namespace mean_value_of_quadrilateral_angles_l796_796169

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796169


namespace rectangle_area_l796_796255

theorem rectangle_area (total_area a1 a2 a3 : ℕ) (h_total : total_area = 72) (h_a1 : a1 = 15) (h_a2 : a2 = 12) (h_a3 : a3 = 18) : 
  (total_area = a1 + a2 + a3 + 27) :=
by
  rw [h_total, h_a1, h_a2, h_a3]
  norm_num
  sorry

end rectangle_area_l796_796255


namespace quadrilateral_is_trapezoid_l796_796418

variables {V : Type*} [add_comm_group V] [module ℝ V]
variables (a b : V)

def AB : V := a + 2 • b
def BC : V := -4 • a - b
def CD : V := -5 • a - 3 • b
def AD : V := AB a b + BC a b + CD a b

/-- Proof that quadrilateral ABCD is a trapezoid -/
theorem quadrilateral_is_trapezoid (a b : V) :
  AD a b = 2 • BC a b → 
  trapezoid (AB a b) (BC a b) (CD a b) :=
sorry

end quadrilateral_is_trapezoid_l796_796418


namespace conic_section_eccentricity_l796_796729

theorem conic_section_eccentricity (m : ℝ) (h : 2 * 8 = m^2) :
    (∃ e : ℝ, ((e = (Real.sqrt 2) / 2) ∨ (e = Real.sqrt 3))) :=
by
  sorry

end conic_section_eccentricity_l796_796729


namespace probability_neither_event_l796_796508

-- Definitions and conditions
variable (P : Set (Set Prop) → ℝ)
variable (A B : Prop)
variable [ProbabilityMeasure P] 

-- Additional given conditions
axiom PA : P {A} = 0.20
axiom PB : P {B} = 0.40
axiom PAB : P {A, B} = 0.15

-- The theorem to prove
theorem probability_neither_event : P (neither A nor B) = 0.55 :=
by
  -- Definitions for 'or' and 'not' in terms of event A and event B
  let not_A := P {¬A}
  let not_B := P {¬B}
  let A_or_B := P {A ∨ B}
  let not_A_and_not_B := P {¬A ∧ ¬B}
  have h1 : A_or_B = PA + PB - PAB, from sorry
  have h2 : not_A_and_not_B = 1 - A_or_B, from sorry
  show not_A_and_not_B = 0.55, from sorry

end probability_neither_event_l796_796508


namespace mean_value_of_quadrilateral_angles_l796_796042

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796042


namespace equal_angles_S3_S1_S2_l796_796880

-- Define points A and B
variables {A B : Point}

-- Define circles S, S1, S2, and S3
variables {S S1 S2 S3 : Circle}

-- Define tangency and perpendicularity conditions
variables (tangent_S_S1 : Tangent S S1)
          (tangent_S_S2 : Tangent S S2)
          (perpendicular_S_S3 : Perpendicular S S3)
          (passes_through_A_S1 : PassesThrough S1 A)
          (passes_through_B_S1 : PassesThrough S1 B)
          (passes_through_A_S2 : PassesThrough S2 A)
          (passes_through_B_S2 : PassesThrough S2 B)

-- Prove that S3 forms equal angles with S1 and S2
theorem equal_angles_S3_S1_S2 : forms_equal_angles S3 S1 S2 :=
by sorry

end equal_angles_S3_S1_S2_l796_796880


namespace mean_of_quadrilateral_angles_l796_796201

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796201


namespace mean_value_of_quadrilateral_angles_l796_796076

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796076


namespace sum_bound_l796_796828

variable (n : ℕ)
variable (x : Fin n → ℝ)

theorem sum_bound (h : ∑ i in Finset.univ, (x i)^2 = 1) :
  (∑ i in Finset.univ, x i / (1 + ∑ j in Finset.range (i+1), (x j)^2)) < Real.sqrt (n / 2) :=
sorry

end sum_bound_l796_796828


namespace total_bees_l796_796550

theorem total_bees 
    (B : ℕ) 
    (h1 : (1/5 : ℚ) * B + (1/3 : ℚ) * B + (2/5 : ℚ) * B + 1 = B) : 
    B = 15 := sorry

end total_bees_l796_796550


namespace floor_ceil_product_correct_l796_796301

noncomputable def floor_ceil_product : ℤ :=
  ∏ (n : ℕ) in Finset.range 7, (Int.floor (- (n : ℝ) - 0.5) * Int.ceil ((n : ℝ) + 0.5))

theorem floor_ceil_product_correct :
  floor_ceil_product = -2116800 := by
  sorry

end floor_ceil_product_correct_l796_796301


namespace secant_product_l796_796224

theorem secant_product : 
  (∃ (m n : ℕ), (m > 1) ∧ (n > 1) ∧ (∏ k in (Finset.range 30).map (λ k, 3 * (k + 1) - 1), (Real.sec (k : ℝ) * (π / 180))^2 = m ^ n) ∧ (m + n = 31)) :=
by {
  sorry
}

end secant_product_l796_796224


namespace minimum_time_to_reenter_classroom_l796_796524

def concentration (t : ℝ) : ℝ :=
  if 0 ≤ t ∧ t ≤ 0.1 then 10 * t
  else (1 / 16) ^ (t - 0.1)

theorem minimum_time_to_reenter_classroom : 
  ∃ t, concentration t ≤ 0.25 ∧ (∀ t', concentration t' ≤ 0.25 → t' ≥ t) :=
begin
  use 0.6,
  split,
  play,
  refine₂,
end

end minimum_time_to_reenter_classroom_l796_796524


namespace basketball_or_cricket_l796_796902

theorem basketball_or_cricket (B C B_and_C : ℕ) (hB : B = 9) (hC : C = 8) (hB_and_C : B_and_C = 6) :
    B + C - B_and_C = 11 :=
by
  rw [hB, hC, hB_and_C]
  exact rfl

end basketball_or_cricket_l796_796902


namespace apprentice_completion_time_l796_796915

/-- A brigade of mechanics can complete a task 15 hours faster than a brigade of apprentices.
    If the brigade of apprentices works for 18 hours on the task, and then the brigade of mechanics
    continues working on the task for 6 hours, only 0.6 of the entire task will be completed.
    Determine the number of hours required for the brigade of apprentices to complete the entire task alone. -/
def time_required_for_apprentices (x : ℝ) (hx : 0 < x) : ℝ :=
  if (18/x + 6/(x-15) = 0.6) then x else 0

theorem apprentice_completion_time (x : ℝ) (hx : 0 < x) (hx_mechanic : x > 15) :
  (18/x + 6/(x-15) = 0.6) → time_required_for_apprentices x hx = 62.25 :=
by sorry

end apprentice_completion_time_l796_796915


namespace percentage_discount_l796_796596

theorem percentage_discount (W R : ℕ) (profit : ℕ) (selling_price discount_amount : ℕ) (D : ℕ)
  (h1 : W = 90) (h2 : R = 120) (h3 : profit = 0.2 * W)
  (h4 : selling_price = W + profit) (h5 : discount_amount = R - selling_price)
  (h6 : D = (discount_amount / R) * 100) :
  D = 10 := 
sorry

end percentage_discount_l796_796596


namespace number_of_rational_points_in_region_l796_796505

/-- The statement to prove the number of points with positive rational coordinates
within the prescribed region -/
theorem number_of_rational_points_in_region : 
  let region := {p : ℚ × ℚ | 1 ≤ p.1 ∧ 0 < p.1 ∧ 0 < p.2 ∧ p.1 + p.2 ≤ 4}
  ∃ (s : Finset (ℚ × ℚ)), s.card = 10 ∧ ∀ p ∈ s, p ∈ region :=
sorry

end number_of_rational_points_in_region_l796_796505


namespace sector_area_l796_796364

-- Define radius and central angle as conditions
def radius : ℝ := 1
def central_angle : ℝ := 2

-- Define the theorem to prove that the area of the sector is 1 cm² given the conditions
theorem sector_area : (1 / 2) * radius * central_angle = 1 := 
by 
  -- sorry is used to skip the actual proof
  sorry

end sector_area_l796_796364


namespace kolya_start_time_l796_796280

-- Definitions of conditions as per the initial problem statement
def angle_moved_by_minute_hand (x : ℝ) : ℝ := 6 * x
def angle_moved_by_hour_hand (x : ℝ) : ℝ := 30 + 0.5 * x

theorem kolya_start_time (x : ℝ) :
  (angle_moved_by_minute_hand x = (angle_moved_by_hour_hand x + angle_moved_by_hour_hand x + 60) / 2) ∨
  (angle_moved_by_minute_hand x - 180 = (angle_moved_by_hour_hand x + angle_moved_by_hour_hand x + 60) / 2) :=
sorry

end kolya_start_time_l796_796280


namespace pedal_circles_common_point_l796_796962

theorem pedal_circles_common_point {A B C D : Type} [plane A B C D] 
  (h_non_collinear : ¬collinear A B C D) : 
  ∃ P : point, ∀ P_circle : pedal_circle, 
  P ∈ P_circle := sorry

end pedal_circles_common_point_l796_796962


namespace upper_derivative_borel_measurable_lower_derivative_borel_measurable_l796_796433

noncomputable def upper_derivative (f : ℝ → ℝ) (x : ℝ) :=
  filter.limsup (λ (hk : ℝ × ℝ), (f (x + hk.1) - f (x - hk.2)) / (hk.1 + hk.2))
  (filter.prod (filter.inf filter.at_top (filter.principal {h | h > 0}))
               (filter.inf filter.at_top (filter.principal {k | k > 0})))

noncomputable def lower_derivative (f : ℝ → ℝ) (x : ℝ) :=
  filter.liminf (λ (hk : ℝ × ℝ), (f (x + hk.1) - f (x - hk.2)) / (hk.1 + hk.2))
  (filter.prod (filter.inf filter.at_top (filter.principal {h | h > 0}))
               (filter.inf filter.at_top (filter.principal {k | k > 0})))

theorem upper_derivative_borel_measurable {f : ℝ → ℝ} (hf : ∀ x, is_finite (f x)) : 
  measurable (upper_derivative f) :=
sorry

theorem lower_derivative_borel_measurable {f : ℝ → ℝ} (hf : ∀ x, is_finite (f x)) : 
  measurable (lower_derivative f) :=
sorry

end upper_derivative_borel_measurable_lower_derivative_borel_measurable_l796_796433


namespace b_n_formula_c_n_bound_l796_796634

-- Definitions of the sequences based on conditions.
variables {a_n : ℕ → ℝ} {b_n : ℕ → ℝ} {c_n : ℕ → ℝ}

-- Given conditions:
axiom geometric_sequence (n : ℕ) : 0 < a_n n
axiom sum_a1_a3 : a_n 1 + a_n 3 = 10
axiom sum_a3_a5 : a_n 3 + a_n 5 = 40
noncomputable def b_n (n : ℕ) := log 2 (a_n n)
def c_1 := (1 : ℝ)
def c_n (n : ℕ) := nat.rec_on n c_1 (λ n ih, ih + b_n n / a_n n)

-- To prove:
theorem b_n_formula (n : ℕ) : b_n n = n :=
sorry

theorem c_n_bound (n : ℕ) (hn : 0 < n) : c_n n < 3 :=
sorry

end b_n_formula_c_n_bound_l796_796634


namespace mean_value_of_quadrilateral_angles_l796_796167

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796167


namespace probability_Bernardo_number_larger_l796_796948

noncomputable def set_Bernardo := {x | x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
noncomputable def set_Silvia := {x | x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 10}}

noncomputable def probability_Bernardo_larger := 
((ℕ.choose 9 2).to_real / (ℕ.choose 10 3).to_real * 1 / 6) + 
((ℕ.choose 8 3).to_real / (ℕ.choose 9 3).to_real * (69 / 140))

theorem probability_Bernardo_number_larger : 
  probability_Bernardo_larger = 0.395 := 
by
  -- Calculations and proof go here 
  sorry

end probability_Bernardo_number_larger_l796_796948


namespace collinear_M_N_P_l796_796805

variables {A B C I D E P M N : Type*}
variables [is_triangle A B C] [incenter I A B C]
variables [incircle_touches D I B C] [incircle_touches E I A C]
variables [intersects P (line AI) (line DE)]
variables [midpoint M B C] [midpoint N A B]

theorem collinear_M_N_P :
  collinear M N P :=
sorry

end collinear_M_N_P_l796_796805


namespace cos_angle_AND_regular_tetrahedron_l796_796236

-- Definitions based on conditions
variables {A B C D N : Point}
variable {s : ℝ} -- side length of the tetrahedron

-- Assume the necessary properties of a regular tetrahedron
def regular_tetrahedron (A B C D : Point) (s : ℝ) : Prop :=
  dist A B = s ∧ dist A C = s ∧ dist A D = s ∧
  dist B C = s ∧ dist B D = s ∧ dist C D = s

-- Assume N is the midpoint of segment BC
def midpoint (N B C : Point) : Prop :=
  2 * dist N B = dist B C ∧ 2 * dist N C = dist B C

-- Main theorem statement
theorem cos_angle_AND_regular_tetrahedron (h1 : regular_tetrahedron A B C D s)
    (h2 : midpoint N B C) :
    cos (angle A N D) = 2/3 :=
sorry

end cos_angle_AND_regular_tetrahedron_l796_796236


namespace mean_value_of_quadrilateral_angles_l796_796049

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796049


namespace option_a_option_b_option_c_option_d_l796_796895

theorem option_a (x : ℝ) (h : x > 1) : (min (x + 1 / (x - 1)) = 3) := 
sorry

theorem option_b (x : ℝ) : (min ((x ^ 2 + 5) / (sqrt (x ^ 2 + 4))) ≠ 2) := 
sorry

theorem option_c (x : ℝ) (h : 0 < x ∧ x < 10) : (max (sqrt (x * (10 - x))) = 5) := 
sorry

theorem option_d (x y : ℝ) (hx : 0 < x) (hy : true) (h : x + 2 * y = 3 * x * y) : (max (2 * x + y) ≠ 3) := 
sorry

end option_a_option_b_option_c_option_d_l796_796895


namespace mean_of_quadrilateral_angles_l796_796191

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796191


namespace mean_of_quadrilateral_angles_is_90_l796_796102

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796102


namespace mean_value_of_quadrilateral_angles_l796_796125

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796125


namespace mean_of_quadrilateral_angles_l796_796027

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796027


namespace ratio_of_areas_l796_796589

theorem ratio_of_areas (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  let A_original := l * w
  let A_new := (2 * l) * (3 * w) in
  A_original / A_new = 1 / 6 := 
by 
  sorry

end ratio_of_areas_l796_796589


namespace largest_cardinality_A_l796_796605

noncomputable def largest_cardinality (A : Type) [fintype A] [has_mul A] : ℕ :=
fintype.card A

variables {A : Type} [fintype A] [has_mul A]
variables (associative : ∀ (a b c : A), a * (b * c) = (a * b) * c)
variables (cancellation : ∀ (a b c : A), a * c = b * c → a = b)
variables (identity_exists : ∃ e : A, ∀ a : A, a * e = a)
variables (special_condition : ∀ (a b : A) (e : A), a ≠ e → b ≠ e → (a * a * a) * b = (b * b * b) * (a * a))

theorem largest_cardinality_A : largest_cardinality A = 3 :=
sorry

end largest_cardinality_A_l796_796605


namespace rectangle_area_l796_796422

noncomputable def area_of_rectangle (radius : ℝ) (ab ad : ℝ) : ℝ :=
  ab * ad

theorem rectangle_area (radius : ℝ) (ad : ℝ) (ab : ℝ) 
  (h_radius : radius = Real.sqrt 5)
  (h_ab_ad_relation : ab = 4 * ad) : 
  area_of_rectangle radius ab ad = 16 / 5 :=
by
  sorry

end rectangle_area_l796_796422


namespace probability_greater_than_two_on_die_l796_796766

variable {favorable_outcomes : ℕ := 4}
variable {total_outcomes : ℕ := 6}

theorem probability_greater_than_two_on_die : (favorable_outcomes : ℚ) / total_outcomes = 2 / 3 := 
by
  sorry

end probability_greater_than_two_on_die_l796_796766


namespace incorrect_supplementary_congruent_l796_796898

def supplementary_angles (α β : ℝ) : Prop :=
  α + β = 180

def congruent_angles (α β : ℝ) : Prop :=
  α = β

def vertical_angles (α β : ℝ) : Prop :=
  α = β

def cube_root (x y : ℝ) : Prop :=
  y^3 = x

def perpendicular_lines (l1 l2 : ℝ) : Prop :=
  l1 * l2 = -1

def parallel_lines (l1 l2 : ℝ) : Prop :=
  ∀ᶠ x in (𝓝 l1), ∀ᶠ y in (𝓝 l2), x = y

theorem incorrect_supplementary_congruent (α β : ℝ) (h : supplementary_angles α β) : ¬ congruent_angles α β :=
by
  sorry

end incorrect_supplementary_congruent_l796_796898


namespace factorize_m_cubed_minus_16m_l796_796652

theorem factorize_m_cubed_minus_16m (m : ℝ) : m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end factorize_m_cubed_minus_16m_l796_796652


namespace mean_value_of_quadrilateral_angles_l796_796166

theorem mean_value_of_quadrilateral_angles (angles : Fin 4 → ℝ) 
  (h_sum : ∑ i, angles i = 360) : ∑ i, angles i / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796166


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796073

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796073


namespace mean_value_of_quadrilateral_angles_l796_796088

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796088


namespace prod_mod_25_l796_796485

theorem prod_mod_25 (m : ℤ) (h : 0 ≤ m ∧ m < 25) : 93 * 59 * 84 % 25 = 8 := 
by {
  sorry,
}

end prod_mod_25_l796_796485


namespace find_f_pi34_find_cos_alpha_minus_beta_l796_796733

section Problem

-- Vector definitions
def a (ω x : ℝ) : ℝ × ℝ := (real.sqrt 2 * real.cos (ω * x), 1)
def b (ω x : ℝ) : ℝ × ℝ := (2 * real.sin (ω * x + real.pi / 4), -1)

-- Function f definition
def f (ω x : ℝ) : ℝ := (a ω x).fst * (b ω x).fst + (a ω x).snd * (b ω x).snd

-- Conditions
axiom omega_bounds (ω : ℝ) : 1/4 ≤ ω ∧ ω ≤ 3/2

-- Problem statements
axiom axis_of_symmetry (ω : ℝ) : 1/4 ≤ ω ∧ ω ≤ 3/2 → 2 * ω * (5 * real.pi / 8) + real.pi / 4 = real.pi / 2 + real.pi -- axis of symmetry

axiom values_f1 (α β : ℝ) :
  f 1 (α/2 - real.pi/8) = real.sqrt 2 / 3 ∧
  f 1 (β/2 - real.pi/8) = 2 * real.sqrt 2 / 3 ∧
  α ∈ (-real.pi/2, real.pi/2) ∧
  β ∈ (-real.pi/2, real.pi/2)

-- The first part of the problem
theorem find_f_pi34 :
  f 1 (3 * real.pi / 4) = -1 := 
sorry

-- The second part of the problem
theorem find_cos_alpha_minus_beta (α β : ℝ) 
  (hval : f 1 (α/2 - real.pi / 8) = real.sqrt 2 / 3)
  (hval' : f 1 (β/2 - real.pi / 8) = 2 * real.sqrt 2 / 3)
  (halpha : α > -real.pi/2 ∧ α < real.pi/2)
  (hbeta : β > -real.pi/2 ∧ β < real.pi/2) :
  real.cos (α - β) = (2 * real.sqrt 10 + 2) / 9 := 
sorry
   
end Problem

end find_f_pi34_find_cos_alpha_minus_beta_l796_796733


namespace mean_value_of_quadrilateral_angles_l796_796052

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796052


namespace average_weight_of_all_boys_l796_796844

theorem average_weight_of_all_boys 
  (n₁ n₂ : ℕ) (w₁ w₂ : ℝ) 
  (h₁ : n₁ = 20) (h₂ : w₁ = 50.25) 
  (h₃ : n₂ = 8) (h₄ : w₂ = 45.15) :
  (n₁ * w₁ + n₂ * w₂) / (n₁ + n₂) = 48.79 := 
by
  sorry

end average_weight_of_all_boys_l796_796844


namespace mean_value_interior_angles_quadrilateral_l796_796115

theorem mean_value_interior_angles_quadrilateral :
  let n := 4 in
  (n - 2) * 180 / n = 90 :=
by
  let n := 4
  sorry

end mean_value_interior_angles_quadrilateral_l796_796115


namespace abs_sub_abs_eq_neg_two_a_l796_796391

theorem abs_sub_abs_eq_neg_two_a (a : ℝ) (h : |a| = -a) : |a - |a|| = -2a := 
by 
  sorry

end abs_sub_abs_eq_neg_two_a_l796_796391


namespace mean_value_of_quadrilateral_angles_l796_796205

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796205


namespace mean_value_of_quadrilateral_interior_angles_l796_796175

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796175


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796058

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796058


namespace num_sequences_with_6_heads_in_10_flips_l796_796251

theorem num_sequences_with_6_heads_in_10_flips : (finset.card (finset.filter (λ seq : fin 10 → bool, (finset.card (finset.filter id (finset.univ.image seq))) = 6) finset.univ) = 210) :=
sorry

end num_sequences_with_6_heads_in_10_flips_l796_796251


namespace six_digit_number_representation_l796_796560

-- Defining that a is a two-digit number
def isTwoDigitNumber (a : ℕ) : Prop := a >= 10 ∧ a < 100

-- Defining that b is a four-digit number
def isFourDigitNumber (b : ℕ) : Prop := b >= 1000 ∧ b < 10000

-- The statement that placing a to the left of b forms the number 10000*a + b
theorem six_digit_number_representation (a b : ℕ) 
  (ha : isTwoDigitNumber a) 
  (hb : isFourDigitNumber b) : 
  (10000 * a + b) = (10^4 * a + b) :=
by
  sorry

end six_digit_number_representation_l796_796560


namespace least_non_lucky_multiple_of_7_correct_l796_796926

def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

def least_non_lucky_multiple_of_7 : ℕ :=
  14

theorem least_non_lucky_multiple_of_7_correct : 
  ¬ is_lucky 14 ∧ ∀ m, m < 14 → m % 7 = 0 → ¬ ¬ is_lucky m :=
by
  sorry

end least_non_lucky_multiple_of_7_correct_l796_796926


namespace csc_315_equals_neg_sqrt_2_l796_796984

-- Define the conditions
def csc (x : Real) : Real := 1 / Real.sin x
axiom sin_subtraction_360 (x : Real) : Real.sin (360 - x) = -Real.sin x

-- State the problem
theorem csc_315_equals_neg_sqrt_2 : csc 315 = -Real.sqrt 2 := by
  sorry

end csc_315_equals_neg_sqrt_2_l796_796984


namespace sum_G_eq_10094_l796_796812

/-- 
G(n) counts the number of solutions to \cos x = \sin (n * x) on the interval [0, 2π] 
for each integer n greater than 2.
-/
noncomputable def G (n : ℕ) : ℕ :=
if n > 2 then 2 * n else 0

theorem sum_G_eq_10094 : (∑ n in Finset.range 98 | (n + 3 > 2), G (n + 3)) = 10094 := by
  sorry

end sum_G_eq_10094_l796_796812


namespace problem_solution_l796_796776

-- Define the Cartesian points M, N, and A
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨-2, 0⟩
def N : Point := ⟨1, 0⟩
def A : Point := ⟨3, 1⟩

-- Define the distance formula
def dist (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Define the locus condition
def locus (P : Point) : Prop :=
  dist P M = real.sqrt 2 * dist P N

-- Define the correct answer statements
def statement_B : Prop :=
  ∃ P : Point, locus P ∧ (let area := (real.abs ((4 * (N.y - M.y) - (1 * (N.x - M.x))) / 2)) in
                          area = 9 * real.sqrt 2 / 2)

def statement_D : Prop :=
  ∃ D E : Point, locus D ∧ locus E ∧ 
  (let line := λ x : ℝ, 2 * x + 1 in
   let distance_center := real.abs (2 * 4 + 0 - 1) / real.sqrt (2^2 + 1^2) in
   let radius := real.sqrt 18 in
   real.sqrt (radius^2 - distance_center^2) = 3 * real.sqrt 5 / 5)

-- Main theorem stating that both statements B and D are correct
theorem problem_solution : statement_B ∧ statement_D :=
  sorry

end problem_solution_l796_796776


namespace ducks_and_geese_difference_l796_796622

variable (d g d' l : ℕ)
variables (hd : d = 25)
variables (hg : g = 2 * d - 10)
variables (hd' : d' = d + 4)
variables (hl : l = 15 - 5)

theorem ducks_and_geese_difference :
  let geese_remain := g - l
  let ducks_remain := d'
  geese_remain - ducks_remain = 1 :=
by
  sorry

end ducks_and_geese_difference_l796_796622


namespace unique_fundamental_solution_l796_796360

theorem unique_fundamental_solution 
  (x0 y0 : ℕ) 
  (hx0y0 : x0^2 - 2003 * y0^2 = 1) 
  (x y : ℕ) 
  (hxpy : x^2 - 2003 * y^2 = 1) 
  (pos_x : x > 0) 
  (pos_y : y > 0)
  (prime_div_x : ∀ p : ℕ, prime p → p ∣ x → p ∣ x0) : 
  x = x0 ∧ y = y0 := 
sorry

end unique_fundamental_solution_l796_796360


namespace mean_value_of_quadrilateral_angles_l796_796048

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796048


namespace correct_statements_about_triangular_prism_l796_796543

-- Definitions regarding triangular prisms and spheres
structure TriangularPrism :=
  (base_triangle : Triangle)
  (height : ℝ)

structure Sphere :=
  (center : Point3D)
  (radius : ℝ)

def has_circumscribed_sphere (prism : TriangularPrism) : Prop :=
  ∃ (sphere : Sphere), 
  (∀ v : Point3D, v ∈ vertices prism.base_triangle ∪ top_vertices prism.height prism.base_triangle -> distance sphere.center v = sphere.radius)

def has_inscribed_sphere (prism : TriangularPrism) : Prop :=
  ∃ (sphere : Sphere), 
  (∀ v : Point3D, v ∈ lateral_faces prism.base_triangle prism.height -> distance sphere.center v = sphere.radius)

def is_regular (prism : TriangularPrism) : Prop :=
  is_equilateral prism.base_triangle

def volume_of_prism (prism : TriangularPrism) : ℝ :=
  area prism.base_triangle * prism.height

theorem correct_statements_about_triangular_prism {prism : TriangularPrism} :
  has_circumscribed_sphere prism ∧ 
  ¬has_inscribed_sphere prism ∧ 
  (is_regular prism → prism.height = 2 → volume_of_prism prism = 6 * sqrt 3) ∧
  (has_circumscribed_sphere prism → 
    (∃ (face : Face), contains prism face prism.circumscribed_sphere.center) ->
    is_right_angled prism.base_triangle) :=
begin
  sorry
end

end correct_statements_about_triangular_prism_l796_796543


namespace range_of_f_l796_796660

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_f :
  Set.range f = Set.Icc (Real.pi / 2 + Real.arctan (-2)) (Real.pi / 2 + Real.arctan 2) :=
sorry

end range_of_f_l796_796660


namespace trig_identity_solution_l796_796549

-- Define the necessary trigonometric functions
noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x
noncomputable def cot (x : ℝ) : ℝ := Real.cos x / Real.sin x

-- Statement of the theorem
theorem trig_identity_solution (x : ℝ) (k : ℤ) (hcos : Real.cos x ≠ 0) (hsin : Real.sin x ≠ 0) :
  (Real.sin x) ^ 2 * tan x + (Real.cos x) ^ 2 * cot x + 2 * Real.sin x * Real.cos x = (4 * Real.sqrt 3) / 3 →
  ∃ k : ℤ, x = (-1) ^ k * (Real.pi / 6) + (Real.pi / 2) :=
sorry

end trig_identity_solution_l796_796549


namespace polynomial_difference_l796_796751

variable {R : Type*} [CommRing R]

theorem polynomial_difference (x y k : R) (h : ∀ (p : R), p ≠ 0 → (x^3 - 2 * k * x * y) - (y^2 + 4 * x * y) ≠ 0 → ¬ (p * x * y)) :
(k = -2) :=
by
  have pol_diff := (x^3 - 2 * k * x * y) - (y^2 + 4 * x * y)
  rw [pol_diff]
  sorry

end polynomial_difference_l796_796751


namespace strawberries_picking_problem_l796_796463

noncomputable def StrawberriesPicked : Prop :=
  let kg_to_lb := 2.2
  let marco_pounds := 1 + 3 * kg_to_lb
  let sister_pounds := 1.5 * marco_pounds
  let father_pounds := 2 * sister_pounds
  marco_pounds = 7.6 ∧ sister_pounds = 11.4 ∧ father_pounds = 22.8

theorem strawberries_picking_problem : StrawberriesPicked :=
  sorry

end strawberries_picking_problem_l796_796463


namespace tetromino_symmetry_count_l796_796738

-- Defining the conditions
def standard_tetrominoes : Set (Set (ℕ × ℕ)) := { ... } -- Set of standard shapes
def non_standard_tetrominoes : Set (Set (ℕ × ℕ)) := { ... } -- Set of non-standard shapes

-- Predicates to check if a given tetromino has reflectional symmetry
def has_reflectional_symmetry (tetromino : Set (ℕ × ℕ)) : Prop := sorry -- Define reflectional symmetry condition

-- Counting function for symmetric tetrominoes
def count_symmetric_tetrominoes (tetrominoes : Set (Set (ℕ × ℕ))) : ℕ :=
  (tetrominoes.filter has_reflectional_symmetry).card

-- Number of symmetric standard and non-standard tetrominoes
def num_symmetric_standard : ℕ := count_symmetric_tetrominoes standard_tetrominoes
def num_symmetric_non_standard : ℕ := count_symmetric_tetrominoes non_standard_tetrominoes
def total_symmetric_tetrominoes : ℕ := num_symmetric_standard + num_symmetric_non_standard

-- Proof statement
theorem tetromino_symmetry_count : total_symmetric_tetrominoes = 5 := by
  sorry

end tetromino_symmetry_count_l796_796738


namespace mean_of_quadrilateral_angles_l796_796187

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360) → 90 :=
by
  intro h
  sorry

end mean_of_quadrilateral_angles_l796_796187


namespace count_triangles_in_figure_l796_796284

open Finset

theorem count_triangles_in_figure 
  (square : Type)
  (vertices : Finset square)
  (diagonals segments : Finset (Finset square)) :
  (vertices.card = 4) →
  (diagonals.card = 2) →
  (segments.card = 4) →
  -- Each segment should join the midpoints of opposite sides
  (∀ seg ∈ segments, ∃ x y ∈ vertices, x ≠ y ∧ midpoints x y seg) →
  -- Each diagonal should join two opposite vertices
  (∀ diag ∈ diagonals, ∃ x y ∈ vertices, x ≠ y ∧ diagonal x y diag) →
  count_total_triangles vertices diagonals segments = 16 :=
sorry

end count_triangles_in_figure_l796_796284


namespace mean_value_of_quadrilateral_angles_l796_796123

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796123


namespace mean_of_quadrilateral_angles_is_90_l796_796104

theorem mean_of_quadrilateral_angles_is_90 :
  let n := 4 in
  let sum_of_interior_angles (n : ℕ) := (n - 2) * 180 in
  let quadrilateral_interior_angle_sum := sum_of_interior_angles 4 in
  quadrilateral_interior_angle_sum = 360 →
  360 / 4 = 90 :=
by
  intros n sum_of_interior_angles quadrilateral_interior_angle_sum h
  exact sorry

end mean_of_quadrilateral_angles_is_90_l796_796104


namespace probability_at_least_one_hit_l796_796620

theorem probability_at_least_one_hit (P_A P_B : ℝ) (hA : P_A = 1/2) (hB : P_B = 1/3) (independent : independent_events A B) : 
  ∃ P_at_least_one : ℝ, P_at_least_one = 1 - (1 - P_A) * (1 - P_B) ∧ P_at_least_one = 2 / 3 :=
by
  sorry

end probability_at_least_one_hit_l796_796620


namespace percentage_discount_is_10_percent_l796_796598

def wholesale_price : ℝ := 90
def profit_percentage : ℝ := 0.20
def retail_price : ℝ := 120

theorem percentage_discount_is_10_percent :
  let profit := profit_percentage * wholesale_price in
  let selling_price := wholesale_price + profit in
  let discount_amount := retail_price - selling_price in
  let percentage_discount := (discount_amount / retail_price) * 100 in
  percentage_discount = 10 :=
by
  sorry

end percentage_discount_is_10_percent_l796_796598


namespace restaurant_min_vegetable_dishes_l796_796261

theorem restaurant_min_vegetable_dishes (x : ℕ) (h : nat.choose 5 2 * nat.choose x 2 ≥ 200) : x ≥ 7 :=
sorry

end restaurant_min_vegetable_dishes_l796_796261


namespace inequality_proof_l796_796439

theorem inequality_proof (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) 
  (h : 1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1) :
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1 := 
by {
  sorry
}

end inequality_proof_l796_796439


namespace csc_315_equals_neg_sqrt_2_l796_796981

-- Define the conditions
def csc (x : Real) : Real := 1 / Real.sin x
axiom sin_subtraction_360 (x : Real) : Real.sin (360 - x) = -Real.sin x

-- State the problem
theorem csc_315_equals_neg_sqrt_2 : csc 315 = -Real.sqrt 2 := by
  sorry

end csc_315_equals_neg_sqrt_2_l796_796981


namespace logarithmic_function_fixed_point_l796_796497

variable (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1)

theorem logarithmic_function_fixed_point :
  ∀ x : ℝ, ∀ y : ℝ, (y = log a (x - 3) + 1) → (x = 4) ∧ (y = 1) :=
by
  intros x y h
  have hx : x - 3 = 1 := by sorry
  have hy : y = 1 := by sorry
  exact And.intro hx hy

end logarithmic_function_fixed_point_l796_796497


namespace possible_values_for_p_t_l796_796445

theorem possible_values_for_p_t (p q r s t : ℝ)
(h₁ : |p - q| = 3)
(h₂ : |q - r| = 4)
(h₃ : |r - s| = 5)
(h₄ : |s - t| = 6) :
  ∃ (v : Finset ℝ), v = {0, 2, 4, 6, 8, 10, 12, 18} ∧ |p - t| ∈ v :=
sorry

end possible_values_for_p_t_l796_796445


namespace roots_greater_than_one_implies_s_greater_than_zero_l796_796668

theorem roots_greater_than_one_implies_s_greater_than_zero
  (b c : ℝ)
  (h : ∃ α β : ℝ, α > 0 ∧ β > 0 ∧ (1 + α) + (1 + β) = -b ∧ (1 + α) * (1 + β) = c) :
  b + c + 1 > 0 :=
sorry

end roots_greater_than_one_implies_s_greater_than_zero_l796_796668


namespace mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796067

-- Defining the sum of interior angles of a quadrilateral.
def sum_of_interior_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  360

-- Defining the number of angles in a quadrilateral.
def number_of_angles (quad : Type) [IsQuadrilateral quad] : ℕ :=
  4

-- The theorem we want to prove: the mean value of the measures of the four interior angles of any quadrilateral.
theorem mean_value_of_angles (quad : Type) [IsQuadrilateral quad] : ℝ :=
  sum_of_interior_angles quad / number_of_angles quad
 
-- Proving the theorem.
theorem mean_value_of_interior_angles_of_quadrilateral (quad : Type) [IsQuadrilateral quad]:
  mean_value_of_angles quad = 90 :=
by
  sorry

end mean_value_of_angles_mean_value_of_interior_angles_of_quadrilateral_l796_796067


namespace maximizing_sum_of_arithmetic_sequence_l796_796923

theorem maximizing_sum_of_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_decreasing : ∀ n, a n > a (n + 1))
  (h_sum : S 5 = S 10) :
  (S 7 >= S n ∧ S 8 >= S n) := sorry

end maximizing_sum_of_arithmetic_sequence_l796_796923


namespace quadrilateral_sum_of_sides_l796_796867

theorem quadrilateral_sum_of_sides {A B C D P Q R S : ℝ} 
  (AB AD BC CD : ℝ)
  (incircle_tangent : true)  -- Placeholder for the incircle tangent property
  (tangent_eq_BP_BQ : B = P → B = Q → ∀ (x : ℝ), x ∈ set {B P Q})  -- Placeholder for BP = BQ = x
  (tangent_eq_CR_CQ : C = R → C = Q → ∀ (y : ℝ), y ∈ set {C R Q})  -- Placeholder for CR = CQ = y
  (PQ_RS_eq : P = Q → R = S → true) :  -- Placeholder for AP = AS
  AB + BC = AD + CD :=
by
  sorry

end quadrilateral_sum_of_sides_l796_796867


namespace mean_value_of_quadrilateral_angles_l796_796202

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796202


namespace count_permutations_divisible_by_9_l796_796383

theorem count_permutations_divisible_by_9 : 
    let three_digit_numbers := {n | 100 ≤ n ∧ n ≤ 999 ∧ ∃ (a b c : ℕ), a + b + c = n ∧ a < 10 ∧ b < 10 ∧ c < 10},
    let is_divisible_by_9 (n : ℕ) := ∃ d : ℕ, n = 9 * d,
    let permutations (n : ℕ) := { m | ∃ (a b c : ℕ), a + b + c = n ∧ m = a * 100 + b * 10 + c ∧ m ∈ three_digit_numbers },
    finset.card { n ∈ three_digit_numbers | ∃ m ∈ (permutations n), is_divisible_by_9 m } = 126 := by sorry

end count_permutations_divisible_by_9_l796_796383


namespace mean_of_quadrilateral_angles_l796_796026

theorem mean_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
begin
  intros a b c d h,
  calc
    (a + b + c + d) / 4 = 360 / 4 : by rw h
                    ... = 90 : by norm_num,
end

end mean_of_quadrilateral_angles_l796_796026


namespace find_square_subtraction_l796_796340

theorem find_square_subtraction (x y : ℝ) (h1 : x = Real.sqrt 5) (h2 : y = Real.sqrt 2) : (x - y)^2 = 7 - 2 * Real.sqrt 10 :=
by
  sorry

end find_square_subtraction_l796_796340


namespace monotonic_increasing_D_l796_796291

def f_A (x : ℝ) : ℝ := (1 / (x + 2))
def f_B (x : ℝ) : ℝ := -(x + 1)^2
def f_C (x : ℝ) : ℝ := 1 + 2 * x^2
def f_D (x : ℝ) : ℝ := -|x|

theorem monotonic_increasing_D : ∀ x y : ℝ, x < y → y < 0 → f_D x < f_D y :=
by
  sorry

end monotonic_increasing_D_l796_796291


namespace mean_value_of_quadrilateral_angles_l796_796203

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l796_796203


namespace PolyCoeffInequality_l796_796548

open Real

variable (p q : ℝ[X])
variable (a : ℝ)
variable (n : ℕ)
variable (h k : ℝ)
variable (deg_p : p.degree = n)
variable (deg_q : q.degree = n - 1)
variable (hp : ∀ i, i ≤ n → |p.coeff i| ≤ h)
variable (hq : ∀ i, i < n → |q.coeff i| ≤ k)
variable (hpq : p = (X + C a) * q)

theorem PolyCoeffInequality : k ≤ h^n := by
  sorry

end PolyCoeffInequality_l796_796548


namespace max_a_plus_b_l796_796450

theorem max_a_plus_b (a b c d e : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : a + 2*b + 3*c + 4*d + 5*e = 300) : a + b ≤ 35 :=
sorry

end max_a_plus_b_l796_796450


namespace vector_subtraction_l796_796382

theorem vector_subtraction (a b : ℝ^3) (ha : a = ![-3, 4, 2]) (hb : b = ![5, -1, 3]) :
  a - 2 • b = ![-13, 6, -4] :=
by
  rw [ha, hb]
  simp
  sorry

end vector_subtraction_l796_796382


namespace find_x_coordinate_l796_796256

theorem find_x_coordinate : 
  ∃ x : ℚ, ∃ t : ℚ, (1, 3, 2) + t • (5, -3, 0) = (x, 1, 2) :=
sorry

end find_x_coordinate_l796_796256


namespace sum_of_good_polygons_is_zero_l796_796886

noncomputable def sum_of_good_polygons (n : ℕ) (c : ℕ → ℕ) : ℤ :=
  ∑ i in Finset.range (n + 1), if h : 3 ≤ i then (-1 : ℤ) ^ i * c i else 0

theorem sum_of_good_polygons_is_zero
  (n : ℕ)
  (h : n ≥ 3)
  (no_three_collinear : ∀ (points : Finset (ℝ × ℝ)), points.card = n → ∃ subset, 
  (subset.card = 3) ∧ (∀ p1 p2 p3, p1 ∈ subset → p2 ∈ subset → p3 ∈ subset → 
  ¬ collinear p1 p2 p3)) 
  (c : ℕ → ℕ)
  (c_def : ∀ k, c k = ∥ {subset : Finset (ℝ × ℝ) // good_set subset k} ∥) :
  sum_of_good_polygons n c = 0 :=
sorry

end sum_of_good_polygons_is_zero_l796_796886


namespace distinct_monic_quadratic_polynomials_count_l796_796659

theorem distinct_monic_quadratic_polynomials_count :
  ∀ (p : ℤ[X]), monic p ∧ (p.degree = 2) ∧ (∀ r ∈ p.roots, ∃ k ∈ ℤ, r = 7^k) ∧
    (∀ c ∈ p.coeffs, |c| ≤ 343^36) → p.distinct_roots ∧ p.coeffs_bound → p.count = 2969 :=
sorry

end distinct_monic_quadratic_polynomials_count_l796_796659


namespace sum_roots_of_unity_l796_796955

theorem sum_roots_of_unity :
  (∑ z in {z : ℂ | z^8 = 1}.toFinset, 1 / (1 + Complex.normSq z)) = 4 := by 
sorry

end sum_roots_of_unity_l796_796955


namespace min_value_on_interval_l796_796698

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x + 2^x

theorem min_value_on_interval (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (max_on_01 : ∀ x ∈ set.Icc (0 : ℝ) (1 : ℝ), f a b x ≤ 4) :
  ∃ x ∈ set.Icc (-1 : ℝ) (0 : ℝ), f a b x = -3 / 2 :=
by
  have ab_eq_2 : a + b = 2 := sorry

  use -1
  split
  linarith
  linarith
  change a * (-1)^3 + b * (-1) + 2^(-1) = -3 / 2
  simp [ab_eq_2]
  sorry

end min_value_on_interval_l796_796698


namespace monotonically_decreasing_interval_l796_796371

noncomputable def f (x : ℝ) : ℝ := cos (2 * x - π / 6) + sin (2 * x)

theorem monotonically_decreasing_interval :
  ∃ a b, (a = π / 6 ∧ b = 2 * π / 3) ∧ (∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x) :=
sorry

end monotonically_decreasing_interval_l796_796371


namespace max_min_values_of_f_l796_796658

def f (x : ℝ) : ℝ := 1 - 2 * (sin x) ^ 2 + 2 * cos x

theorem max_min_values_of_f :
  ∀ x : ℝ, -3/2 ≤ f x ∧ f x ≤ 3 :=
by
  sorry

end max_min_values_of_f_l796_796658


namespace mean_value_of_quadrilateral_interior_angles_l796_796179

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796179


namespace find_x_l796_796566

theorem find_x : ∃ x : ℝ, 0.65 * x = 0.20 * 422.50 ∧ x = 130 :=
begin
  use 130,
  split,
  { 
    sorry 
  },
  {
    refl,
  }
end

end find_x_l796_796566


namespace limit_eval_l796_796232

noncomputable def limit_problem := sorry

theorem limit_eval :
  (∀ x, 0 < x) →
  limit (λ x, (1 - 2^(4 - x^2)) / (2 * (sqrt (2 * x) - sqrt(3 * x^2 - 5 * x + 2)))) 2 = -8 * real.log 2 / 5 :=
begin
  sorry
end

end limit_eval_l796_796232


namespace length_of_AC_l796_796416

-- Define the conditions: lengths and angle
def AB : ℝ := 10
def BC : ℝ := 10
def CD : ℝ := 15
def DA : ℝ := 15
def angle_ADC : ℝ := 120

-- Prove the length of diagonal AC is 15*sqrt(3)
theorem length_of_AC : 
  (CD ^ 2 + DA ^ 2 - 2 * CD * DA * Real.cos (angle_ADC * Real.pi / 180)) = (15 * Real.sqrt 3) ^ 2 :=
by
  sorry

end length_of_AC_l796_796416


namespace other_number_has_8_common_factors_l796_796521

-- Definition of the factors of a number
def factors (n : ℕ) : Finset ℕ := (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0)

-- Define the given number and the set of positive factors it has
def number := 90
def numberFactors : Finset ℕ := factors number

-- Define the other number to be found
def otherNumber := 30
def commonFactors := {1, 2, 3, 5, 6, 10, 15, 30}

-- State the theorem
theorem other_number_has_8_common_factors : factors otherNumber ∩ numberFactors = commonFactors := by
  sorry

end other_number_has_8_common_factors_l796_796521


namespace num_impossible_d_l796_796865

theorem num_impossible_d (t s d : ℕ) (d_pos : d > 0) 
    (triangle_perimeter : 3 * t) 
    (square_perimeter : 4 * s) 
    (perimeter_condition : 3 * t = 4 * s + 2016) 
    (side_length_condition : t = s + d)
    (square_perimeter_pos : square_perimeter > 0) : 
    672 = ∑ n in finset.range 673, 1 := 
by
  sorry

end num_impossible_d_l796_796865


namespace mean_value_of_quadrilateral_angles_l796_796074

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l796_796074


namespace proof_of_problem_l796_796354

noncomputable def problem_statement (α β γ δ : ℝ) : Prop :=
  (∀ n : ℕ, 0 < n → (Int.floor (α * n) * Int.floor (β * n) = Int.floor (γ * n) * Int.floor (δ * n))) ∧
  (α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0) ∧
  ({α, β} ≠ {γ, δ})

theorem proof_of_problem (α β γ δ : ℝ) (h : problem_statement α β γ δ) :
  α * β = γ * δ ∧
  (∃ (a b c d : ℤ), α = a ∧ β = b ∧ γ = c ∧ δ = d ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :=
by
  sorry

end proof_of_problem_l796_796354


namespace exists_two_numbers_with_gcd_quotient_ge_p_plus_one_l796_796453

theorem exists_two_numbers_with_gcd_quotient_ge_p_plus_one (p : ℕ) (hp : Nat.Prime p)
  (l : List ℕ) (hl_len : l.length = p + 1) (hl_distinct : l.Nodup) :
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ l ∧ b ∈ l ∧ a > b ∧ a / (Nat.gcd a b) ≥ p + 1 := sorry

end exists_two_numbers_with_gcd_quotient_ge_p_plus_one_l796_796453


namespace diagonal_AC_length_l796_796414

variable (A B C D : Type)
variables [InnerProductSpace ℝ V] [CompleteSpace V]
variables (AB BC CD DA : ℝ) (angle_ADC : ℝ)

-- Conditions
notation "length_AB" => AB = 10
notation "length_BC" => BC = 10
notation "length_CD" => CD = 15
notation "length_DA" => DA = 15
notation "angle_ADC_120" => angle_ADC = 120

-- The proof problem statement
theorem diagonal_AC_length : 
  length_AB ∧ length_BC ∧ length_CD ∧ length_DA ∧ angle_ADC_120 → 
  ∃ AC : ℝ, AC = 15 * Real.sqrt 3 :=
by
  intros
  sorry

end diagonal_AC_length_l796_796414


namespace mean_value_of_quadrilateral_interior_angles_l796_796184

theorem mean_value_of_quadrilateral_interior_angles (a b c d : ℝ) 
  (h_sum : a + b + c + d = 360) : 
  (a + b + c + d) / 4 = 90 :=
by
  sorry

end mean_value_of_quadrilateral_interior_angles_l796_796184


namespace value_of_a_plus_b_l796_796746

variables (a b c d x : ℕ)

theorem value_of_a_plus_b : (b + c = 9) → (c + d = 3) → (a + d = 8) → (a + b = x) → x = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end value_of_a_plus_b_l796_796746


namespace problem_one_problem_two_problem_three_l796_796713

noncomputable def f (x : ℝ) : ℝ := 3 - 2 * Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def h (x : ℝ) : ℝ := (f x + 1) * g x

noncomputable def M (x : ℝ) : ℝ :=
  if f x >= g x then g x else f x

noncomputable def condition_one : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 8 → -6 ≤ h x ∧ h x ≤ 2

noncomputable def condition_two : Prop :=
  ∃ x, (M x = 1 ∧ 0 < x ∧ x ≤ 2) ∧ (∀ y, 0 < y ∧ y < x → M y < 1)

noncomputable def condition_three : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 8 → f (x^2) * f (Real.sqrt x) ≥ g x * -3

theorem problem_one : condition_one := sorry
theorem problem_two : condition_two := sorry
theorem problem_three : condition_three := sorry

end problem_one_problem_two_problem_three_l796_796713


namespace volunteer_org_percentage_change_l796_796612

theorem volunteer_org_percentage_change 
  (initial_membership : ℝ)
  (fall_increase_rate : ℝ)
  (spring_decrease_rate : ℝ) :
  (initial_membership = 100) →
  (fall_increase_rate = 0.05) →
  (spring_decrease_rate = 0.19) →
  (14.95 : ℝ) =
  ((initial_membership * (1 + fall_increase_rate)) * (1 - spring_decrease_rate)
  - initial_membership) / initial_membership * 100 := by
  sorry

end volunteer_org_percentage_change_l796_796612


namespace q_satisfies_all_conditions_l796_796486

-- Define the monic quartic polynomial q with given conditions.
def q (x : ℝ) : ℝ := x^4 - 7.8*x^3 + 25.4*x^2 - 23.8*x + 48

-- Define the conditions
lemma q_is_monic : ∀ x, q x = x^4 - 7.8*x^3 + 25.4*x^2 - 23.8*x + 48 := 
by sorry

lemma q_has_real_coeffs : ∀ x, q x ∈ ℝ := 
by sorry

lemma q_root_1_sub_3i : q (1 - 3*complex.I) = 0 := 
by sorry

lemma q_eval_0 : q 0 = -48 := 
by sorry

lemma q_eval_1 : q 1 = 0 := 
by sorry

-- Proof statement that the given polynomial satisfies all conditions
theorem q_satisfies_all_conditions : 
  ∀ x, (q_has_real_coeffs x) ∧ (q (1 - 3*complex.I) = 0) ∧ (q 0 = -48) ∧ (q 1 = 0) → q x = x^4 - 7.8*x^3 + 25.4*x^2 - 23.8*x + 48 :=
by sorry

end q_satisfies_all_conditions_l796_796486


namespace balloons_and_cost_l796_796332

variables (x y z d : ℕ)
def total_initial_balloons : ℕ := x + y
def total_cost : ℕ := total_initial_balloons x y * z
def final_remaining_balloons : ℕ := total_initial_balloons x y - d

theorem balloons_and_cost (hx : x = 10) (hy : y = 46) (hz : z = 10) (hd : d = 16) :
  final_remaining_balloons x y d = 40 ∧ total_cost x y z = 560 :=
by
  rw [hx, hy, hz, hd]
  simpl
  sorry

end balloons_and_cost_l796_796332
