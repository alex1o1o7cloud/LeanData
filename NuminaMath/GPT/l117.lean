import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.GcdMonoid
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.RelationDefs
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.IntermediateValue
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Integral
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Polynomial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Pi
import Mathlib.Data.Set.Basic
import Mathlib.NumberTheory.PythagoreanTriples
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.ProbabilityMeasure
import Mathlib.Tactic
import Mathlib.Topology.Probability
import Real
import data.finset
import data.list.perm

namespace probability_product_divisible_by_8_l117_117863

theorem probability_product_divisible_by_8 (dice_rolls : Fin 8 → Fin 6) :
  let N := 6  -- number of sides on a die
  let prob_not_div_2 := (3 / N) ^ 8  -- P(All dice show odd numbers)
  let prob_div_2_not_4 := 8 * (1 / N) * (1 / 2) ^ 7  -- P(Exactly one die shows 2)
  let prob_div_4_not_8 := 
        (binomial 8 2 * (1 / N) ^ 2 * (1 / 2) ^ 6) + 
        (8 * (1 / N) * (1 / 2) ^ 7)  -- P(Exactly two dice show 2 or one die shows 4)
  let prob_not_div_8 := prob_not_div_2 + prob_div_2_not_4 + prob_div_4_not_8
  let prob_div_8 := 1 - prob_not_div_8
  prob_div_8 = 1070 / 1152 :=
by {
    sorry
}

end probability_product_divisible_by_8_l117_117863


namespace deepak_speed_proof_l117_117213

noncomputable def deepak_speed (circumference : ℝ) (meeting_time : ℝ) (wife_speed_kmh : ℝ) : ℝ :=
  let wife_speed_mpm := wife_speed_kmh * 1000 / 60
  let wife_distance := wife_speed_mpm * meeting_time
  let deepak_speed_mpm := ((circumference - wife_distance) / meeting_time)
  deepak_speed_mpm * 60 / 1000

theorem deepak_speed_proof :
  deepak_speed 726 5.28 3.75 = 4.5054 :=
by
  -- The functions and definitions used here come from the problem statement
  -- Conditions:
  -- circumference = 726
  -- meeting_time = 5.28 minutes
  -- wife_speed_kmh = 3.75 km/hr
  sorry

end deepak_speed_proof_l117_117213


namespace arithmetic_sequence_sum_l117_117710

theorem arithmetic_sequence_sum :
  ∀ {a : ℕ → ℤ} {d : ℤ},
    (∀ n, a (n + 1) = a n + d) →
    a 10 = (1 / 2 : ℤ) * a 14 - 6 →
    (∑ i in (finset.range 11), a i) = -132 :=
by
  sorry

end arithmetic_sequence_sum_l117_117710


namespace negative_reciprocal_max_l117_117957

noncomputable def max_value (A : Set ℝ) (a : ℝ) :=
  a ∈ A ∧ ∀ x ∈ A, x ≤ a

theorem negative_reciprocal_max {A : Set ℝ} {a : ℝ}
  (hA_nonempty : A.Nonempty)
  (hA_no_zero : ¬(0 : ℝ) ∈ A)
  (hA_max : max_value A a)
  (ha_neg : a < 0) :
  max_value (Set.image (λ x, -x⁻¹) A) (-a⁻¹) :=
sorry

end negative_reciprocal_max_l117_117957


namespace opposite_of_negative_2023_l117_117043

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117043


namespace sum_of_decimals_as_fraction_l117_117593

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 :=
by
  sorry

end sum_of_decimals_as_fraction_l117_117593


namespace prob_X_wins_l117_117349

-- Definitions of the given probabilities
def P_Y := 1/12
def P_Z := 1/6
def P_total := 0.375

-- The sum of probabilities for mutually exclusive events X, Y, and Z
noncomputable def P_X : ℝ := P_total - (P_Y + P_Z)

-- The main theorem to prove
theorem prob_X_wins : P_X = 0.125 := 
by
  simp [P_X, P_total, P_Y, P_Z]
  norm_num
  sorry

end prob_X_wins_l117_117349


namespace sin_product_inequality_l117_117871

theorem sin_product_inequality
  (α β γ : ℝ)
  (h1 : ∀ x : ℝ, sin x = 2 * sin (x / 2) * cos (x / 2))
  (h2 : ∀ {p r R : ℝ}, sin α * sin β * sin γ = p * r / (2 * R^2))
  (h3 : ∀ {p R : ℝ}, p ≤ 3 * real.sqrt 3 * R / 2)
  (h4 : ∀ {r R : ℝ}, r ≤ R / 2) :
  sin α * sin β * sin γ ≤ 3 * real.sqrt 3 / 8 := 
sorry

end sin_product_inequality_l117_117871


namespace domain_of_function_l117_117813

noncomputable def domain : Set ℝ := {x | x ≥ 1/2 ∧ x ≠ 1}

theorem domain_of_function : ∀ (x : ℝ), (2 * x - 1 ≥ 0) ∧ (x ^ 2 + x - 2 ≠ 0) ↔ (x ∈ domain) :=
by 
  sorry

end domain_of_function_l117_117813


namespace smaller_angle_at_3_15_l117_117934

/-- Given a clock where the hour hand moves 30 degrees per hour
and the minute hand moves 6 degrees per minute,
prove that the smaller angle between the minute hand and the hour hand 
at 3:15 p.m. is 7.5 degrees. 
-/
theorem smaller_angle_at_3_15 : 
  let hour_angle_at_3_15 := 90 + (30 / 60) * 15,
      minute_angle_at_15 := (15 / 60) * 360 in
  |hour_angle_at_3_15 - minute_angle_at_15| = 7.5 := by
  sorry

end smaller_angle_at_3_15_l117_117934


namespace factor_81_minus_36x4_l117_117970

theorem factor_81_minus_36x4 (x : ℝ) : 
    81 - 36 * x^4 = 9 * (Real.sqrt 3 - Real.sqrt 2 * x) * (Real.sqrt 3 + Real.sqrt 2 * x) * (3 + 2 * x^2) :=
sorry

end factor_81_minus_36x4_l117_117970


namespace solution_set_f_lt_2exp_eq_0_to_infty_l117_117384

noncomputable theory

open Set

theorem solution_set_f_lt_2exp_eq_0_to_infty
  (f : ℝ → ℝ)
  (hf_diff : Differentiable ℝ f)
  (hf_deriv : ∀ x : ℝ, deriv f x < f x)
  (hf_at_0 : f 0 = 2) :
  {x : ℝ | f x < 2 * Real.exp x} = Ioi 0 :=
sorry

end solution_set_f_lt_2exp_eq_0_to_infty_l117_117384


namespace six_people_line_up_l117_117444

theorem six_people_line_up : (∃ (n : ℕ), n = 6 * 5 * 4 * 3 * 2 * 1) → ∃ (num_ways : ℕ), num_ways = 720 :=
by
  intro h
  use 720
  exact h

end six_people_line_up_l117_117444


namespace gcd_10010_15015_l117_117238

theorem gcd_10010_15015 :
  Int.gcd 10010 15015 = 5005 :=
by 
  sorry

end gcd_10010_15015_l117_117238


namespace CarrieSpent_l117_117564

variable (CostPerShirt NumberOfShirts : ℝ)

def TotalCost (CostPerShirt NumberOfShirts : ℝ) : ℝ :=
  CostPerShirt * NumberOfShirts

theorem CarrieSpent {CostPerShirt NumberOfShirts : ℝ} 
  (h1 : CostPerShirt = 9.95) 
  (h2 : NumberOfShirts = 20) : 
  TotalCost CostPerShirt NumberOfShirts = 199.00 :=
by
  sorry

end CarrieSpent_l117_117564


namespace circle_radius_from_tangents_l117_117627

theorem circle_radius_from_tangents (A B O T : Point) (r : ℝ) 
  (h_tangents : tangent_point T A O ∧ tangent_point T B O)
  (h_AT : dist A T = 7)
  (h_BT : dist A T = 2 * dist B T) :
  r = 3.5 :=
sorry

end circle_radius_from_tangents_l117_117627


namespace rationalize_denominator_l117_117419

theorem rationalize_denominator :
  (35 / Real.sqrt 35) = Real.sqrt 35 :=
sorry

end rationalize_denominator_l117_117419


namespace term_with_x_3_over_2_in_expansion_term_with_largest_coefficient_l117_117647

noncomputable def general_term (n r : ℕ) : ℚ :=
  nat.choose n r * (-2)^r

theorem term_with_x_3_over_2_in_expansion (n : ℕ) (h : (general_term n 4) / (general_term n 2) = 10) : 
  n = 8 → (general_term 8 1) * x^(3/2) = -16 * x^(3/2) :=
by
  intro h_n
  have n_eq_8 : n = 8 := sorry -- The detailed proof step
  rw [n_eq_8] at h
  sorry

theorem term_with_largest_coefficient (n : ℕ) (h : (general_term n 4) / (general_term n 2) = 10) :
  n = 8 → (general_term 8 6) * x^(-12) = 1792 * x^(-12) :=
by
  intro h_n
  have n_eq_8 : n = 8 := sorry -- The detailed proof step
  rw [n_eq_8] at h
  sorry

end term_with_x_3_over_2_in_expansion_term_with_largest_coefficient_l117_117647


namespace find_xsq_plus_inv_xsq_l117_117336

theorem find_xsq_plus_inv_xsq (x : ℝ) (h : 35 = x^6 + 1/(x^6)) : x^2 + 1/(x^2) = 37 :=
sorry

end find_xsq_plus_inv_xsq_l117_117336


namespace prob_first_gun_hit_l117_117146

variable (p1 p2 p3 : ℝ)
variable (h_p1 : p1 = 0.4)
variable (h_p2 : p2 = 0.3)
variable (h_p3 : p3 = 0.5)

theorem prob_first_gun_hit (p1 p2 p3 : ℝ) (h_p1 : p1 = 0.4) (h_p2 : p2 = 0.3) (h_p3 : p3 = 0.5) :
  (hidden_answer) = 20 / 29  :=
sorry

end prob_first_gun_hit_l117_117146


namespace min_edges_in_graph_with_common_neighbor_l117_117829

theorem min_edges_in_graph_with_common_neighbor (n : ℕ) (h : n = 12):
  ∃ G : SimpleGraph (Fin n), (∀ (A B : Fin n), ∃ (C : Fin n), G.adj A C ∧ G.adj B C) ∧ G.edgeFinset.card = 20 :=
by
  sorry

end min_edges_in_graph_with_common_neighbor_l117_117829


namespace biker_bob_east_distance_l117_117938

theorem biker_bob_east_distance :
  ∀ (w n1 n2 : ℝ) (d : ℝ), 
  w = 8 → n1 = 5 → n2 = 15 →
  d = 20.396078054371138 →
  sqrt ((w - 4)^2 + (n1 + n2)^2) = d :=
begin
  intros w n1 n2 d hw hn1 hn2 hd,
  rw [hw, hn1, hn2, hd],
  sorry
end

end biker_bob_east_distance_l117_117938


namespace cid_earnings_l117_117566

variable (x : ℕ)
variable (oil_change_price repair_price car_wash_price : ℕ)
variable (cars_repaired cars_washed total_earnings : ℕ)

theorem cid_earnings :
  (oil_change_price = 20) →
  (repair_price = 30) →
  (car_wash_price = 5) →
  (cars_repaired = 10) →
  (cars_washed = 15) →
  (total_earnings = 475) →
  (oil_change_price * x + repair_price * cars_repaired + car_wash_price * cars_washed = total_earnings) →
  x = 5 := by sorry

end cid_earnings_l117_117566


namespace Perry_income_l117_117439

theorem Perry_income (income : ℝ) : 
  let tax1 := 0.05 * 5000
  let tax2 := 0.10 * 10000
  ∃ x, (income = 15000 + x) → (x ≥ 0) → (tax1 + tax2 + 0.15 * x = 3800) → income = 32000 :=
begin
  -- proof is to be provided
  sorry
end

end Perry_income_l117_117439


namespace B_works_alone_in_20_days_C_works_alone_in_60_days_AC_together_work_in_8_57_days_l117_117869

-- Definitions of work rates
variables (A B C : ℝ)

-- Conditions
def A_eq_2B : Prop := A = 2 * B
def B_eq_3C : Prop := B = 3 * C
def together_complete_6_days : Prop := 6 * (A + B + C) = 1

-- Prove the duration each workman needs to complete the job alone
theorem B_works_alone_in_20_days (A B C : ℝ) (h1 : A_eq_2B A B)
  (h2 : B_eq_3C B C) (h3 : together_complete_6_days A B C) : 
  B * 20 = 1 := by sorry

theorem C_works_alone_in_60_days (A B C : ℝ) (h1 : A_eq_2B A B)
  (h2 : B_eq_3C B C) (h3 : together_complete_6_days A B C) : 
  C * 60 = 1 := by sorry

theorem AC_together_work_in_8_57_days (A B C : ℝ) (h1 : A_eq_2B A B)
  (h2 : B_eq_3C B C) (h3 : together_complete_6_days A B C) : 
  (A + C) * 8.57 ≈ 1 := by sorry

end B_works_alone_in_20_days_C_works_alone_in_60_days_AC_together_work_in_8_57_days_l117_117869


namespace find_u_l117_117382

noncomputable def c : ℝ × ℝ × ℝ := (4, -3, 0)
noncomputable def d : ℝ × ℝ × ℝ := (2, 0, 2)

def is_unit_vector (u : ℝ × ℝ × ℝ) : Prop :=
  (u.1 ^ 2 + u.2 ^ 2 + u.3 ^ 2 = 1)

def bisects_angle_condition (u : ℝ × ℝ × ℝ) (c d : ℝ × ℝ × ℝ) (k : ℝ) : Prop :=
  d = k • ((c.1 + 3 * u.1) / 2, (c.2 + 3 * u.2) / 2, (c.3 + 3 * u.3) / 2) 

theorem find_u : 
  ∃ u : ℝ × ℝ × ℝ, 
    is_unit_vector u ∧ 
    ∃ k : ℝ, bisects_angle_condition u c d k ∧
    u = (-16/25, 16/25, 1) :=
  sorry

end find_u_l117_117382


namespace circle_properties_l117_117263

-- Define the given circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Standard form of the circle
def standard_form_circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9

-- Define what it means for a line to have a slope of 1 and passes through some y-intercept b
def line_slope1 (b : ℝ) (x y : ℝ) : Prop := y = x + b

-- Origin point
def origin : Prop := true

-- Perpendicularity condition OA ⊥ OB (O is origin)
def perp_condition (A B : ℝ × ℝ) : Prop :=
  let O := (0, 0)
  (A.1 * B.1 + A.2 * B.2 = 0)

-- Prove that the circle's standard form is correct and there exist lines that intersect it 
-- under the given conditions
theorem circle_properties:
  (∀ x y, circle_eq x y ↔ standard_form_circle x y) ∧
  (∃ b : ℝ, 
    (line_slope1 b).line_slope1 ↔
    (∃ A B : ℝ × ℝ, circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧ perp_condition A B)) :=
by
  sorry

end circle_properties_l117_117263


namespace sum_of_decimals_as_fraction_l117_117592

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 :=
by
  sorry

end sum_of_decimals_as_fraction_l117_117592


namespace price_increase_count_l117_117450

-- Conditions
def original_price (P : ℝ) : ℝ := P
def increase_factor : ℝ := 1.15
def final_factor : ℝ := 1.3225

-- The theorem that states the number of times the price increased
theorem price_increase_count (n : ℕ) :
  increase_factor ^ n = final_factor → n = 2 :=
by
  sorry

end price_increase_count_l117_117450


namespace isosceles_triangle_relationship_l117_117631

theorem isosceles_triangle_relationship 
  (A B C D E F H M N : Type*)
  [linear_ordered_field A]
  [metric_space B] [metric_space C] [metric_space D]
  [metric_space E] [metric_space F]
  [metric_space H] [metric_space M] [metric_space N]
  (isosceles_triangle : Π {X Y Z : Type*}, X = Y ∧ Y = Z → X = Z)
  (altitude_AD : Π {A B C D : Type*}, ∀ (angle_A : A) (base_BC : B),
    altitude angle_A base_BC D)
  (altitude_BE : Π {B A C E : Type*}, ∀ (angle_B : B) (side_AC : A),
    altitude angle_B side_AC E)
  (perpendicular_EF : Π {E C F : Type*}, ∀ (from_E : E) (to_BC : C), 
    perpendicular from_E to_BC F)
  (midpoint_N : Π {A H N : Type*},
    midpoint A H N)
  (DM_EF : Π {D M : Type*}, ∀ (D : D) (EF : E),
    distance(D, M) = distance(E, F))
  (MN_sq : Π {M N : Type*}, ∀ (M : M) (N : N), 
    distance(M, N)^2 = b)
  (BN_sq : Π {B N : Type*}, ∀ (B : B) (N : N), 
    distance(B, N)^2 = m)
  (BM_sq : Π {B M : Type*}, ∀ (B : B) (M : M), 
    distance(B, M)^2 = n) :
  b = m + n := by
  sorry

end isosceles_triangle_relationship_l117_117631


namespace gcd_proof_l117_117246

def gcd_10010_15015 := Nat.gcd 10010 15015 = 5005

theorem gcd_proof : gcd_10010_15015 :=
by
  sorry

end gcd_proof_l117_117246


namespace scientific_notation_correct_l117_117789

-- Define the input number
def input_number : ℕ := 858000000

-- Define the expected scientific notation result
def scientific_notation (n : ℕ) : ℝ := 8.58 * 10^8

-- The theorem states that the input number in scientific notation is indeed 8.58 * 10^8
theorem scientific_notation_correct :
  scientific_notation input_number = 8.58 * 10^8 :=
sorry

end scientific_notation_correct_l117_117789


namespace sec_240_eq_neg2_l117_117598

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_240_eq_neg2 : sec 240 = -2 := by
  -- Proof goes here
  sorry

end sec_240_eq_neg2_l117_117598


namespace original_number_l117_117121

theorem original_number (N : ℕ) :
  (∃ k m n : ℕ, N - 6 = 5 * k + 3 ∧ N - 6 = 11 * m + 3 ∧ N - 6 = 13 * n + 3) → N = 724 :=
by
  sorry

end original_number_l117_117121


namespace number_of_bicycles_l117_117832

theorem number_of_bicycles (B T : ℕ) (h1 : T = 14) (h2 : 2 * B + 3 * T = 90) : B = 24 := by
  sorry

end number_of_bicycles_l117_117832


namespace find_prime_p_l117_117140

theorem find_prime_p (p q : ℕ) (hp : p.prime) (hq : q.prime)
    (h1 : ∃ x : ℕ, x^2 = p + q)
    (h2 : ∃ y : ℕ, y^2 = p + 7q) :
    p = 2 :=
by
  sorry

end find_prime_p_l117_117140


namespace opposite_of_neg_2023_l117_117072

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117072


namespace exists_monochromatic_triangle_l117_117502

-- Defining the necessary predicates and objects.
def convex_2000gon (P : set (ℕ × ℕ)) : Prop :=
  P.card = 2000 ∧ convex P

def no_three_diagonals_intersect (D : set (ℕ × ℕ)) : Prop :=
  ∀ d1 d2 d3 ∈ D, d1 ≠ d2 → d1 ≠ d3 → d2 ≠ d3 → 
  ¬ (intersect_at_single_point d1 d2 d3)

def colored_by_999_colors (C : set (ℕ × ℕ) → ℕ) : Prop :=
  ∀ d ∈ C, 1 ≤ d ∧ d ≤ 999

-- Formalizing the existence of the triangle.
theorem exists_monochromatic_triangle 
  (P : set (ℕ × ℕ)) (D : set (ℕ × ℕ)) (C : set (ℕ × ℕ) → ℕ)
  (hP : convex_2000gon P)
  (hD : no_three_diagonals_intersect D)
  (hC : colored_by_999_colors C) : 
  ∃ t : set (ℕ × ℕ), t.card = 3 ∧ (∀ d ∈ t, C d = C (t.some)) :=
sorry

end exists_monochromatic_triangle_l117_117502


namespace sum_as_common_fraction_l117_117596

/-- The sum of 0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 as a common fraction -/
theorem sum_as_common_fraction : (0.2 + 0.04 + 0.006 + 0.0008 + 0.00010) = (12345 / 160000) := by
  sorry

end sum_as_common_fraction_l117_117596


namespace area_of_PQR_is_32_l117_117855

/-- Triangle PQR is an isosceles right triangle with angle P = 90 degrees, and the length of segment PR = 8 cm. The goal is to prove that the area of triangle PQR is 32 square centimeters. -/
theorem area_of_PQR_is_32 
  (P Q R : Type) 
  [triangle P Q R] 
  (right_angle_P : angle P = 90)
  (isosceles_right : isosceles_right_triangle P Q R P Q P R)
  (length_PR : segment_length P R = 8) 
  : area_triangle P Q R = 32 := 
sorry

end area_of_PQR_is_32_l117_117855


namespace gcd_10010_15015_l117_117236

def a := 10010
def b := 15015

theorem gcd_10010_15015 : Nat.gcd a b = 5005 := by
  sorry

end gcd_10010_15015_l117_117236


namespace cab_company_charge_l117_117477

-- Defining the conditions
def total_cost : ℝ := 23
def base_price : ℝ := 3
def distance_to_hospital : ℝ := 5

-- Theorem stating the cost per mile
theorem cab_company_charge : 
  (total_cost - base_price) / distance_to_hospital = 4 :=
by
  -- Proof is omitted
  sorry

end cab_company_charge_l117_117477


namespace opposite_of_negative_2023_l117_117061

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117061


namespace not_like_terms_setD_l117_117125

def monomial (vars : Fin n → ℕ) (coeff : ℚ) : Fin n → ℕ × ℚ := vars × coeff

def like_terms (m1 m2 : Fin n → ℕ × ℚ) : Prop :=
  (∀ i, (m1 i).1 = (m2 i).1) -- Variables and their exponents must match.

-- Sets of monomials
def setA := (monomial ![2, 1] 4, monomial ![2, 1] (2 / 3))
def setB := (monomial ![1, 3] (1 / 3), monomial ![1, 3] (-1 / 3))
def setC := (monomial ![1, 1, 2] 2, monomial ![1, 1, 2] (2 / 3))
def setD := (monomial ![2, 1] 7, monomial ![1, 2] (-9))

theorem not_like_terms_setD :
  ¬ like_terms (fst setD) (snd setD) ∧
  (like_terms (fst setA) (snd setA)) ∧
  (like_terms (fst setB) (snd setB)) ∧
  (like_terms (fst setC) (snd setC)) :=
sorry

end not_like_terms_setD_l117_117125


namespace express_line_away_l117_117504

variables {Point Line Plane : Type}
variables (a : Line) (P : Point) (α : Plane)

def passes_through_outside (a : Line) (P : Point) (α : Plane) : Prop :=
  P ∈ a ∧ P ∉ α

theorem express_line_away (h1 : P ∈ a) (h2 : P ∉ α) :
  passes_through_outside a P α :=
by
  split
  · exact h1
  · exact h2

end express_line_away_l117_117504


namespace rationalize_sqrt_35_l117_117417

theorem rationalize_sqrt_35 : (35 / Real.sqrt 35) = Real.sqrt 35 :=
  sorry

end rationalize_sqrt_35_l117_117417


namespace vertex_of_parabola_y_eq_x2_minus_2_l117_117801

theorem vertex_of_parabola_y_eq_x2_minus_2 :
  vertex (λ x : ℝ, x^2 - 2) = (0, -2) := 
sorry

end vertex_of_parabola_y_eq_x2_minus_2_l117_117801


namespace opposite_of_neg_2023_l117_117079

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117079


namespace average_cost_of_fruit_l117_117395

variable (apples bananas oranges total_cost total_pieces avg_cost : ℕ)

theorem average_cost_of_fruit (h1 : apples = 12)
                              (h2 : bananas = 4)
                              (h3 : oranges = 4)
                              (h4 : total_cost = apples * 2 + bananas * 1 + oranges * 3)
                              (h5 : total_pieces = apples + bananas + oranges)
                              (h6 : avg_cost = total_cost / total_pieces) :
                              avg_cost = 2 :=
by sorry

end average_cost_of_fruit_l117_117395


namespace rationalize_sqrt_35_l117_117418

theorem rationalize_sqrt_35 : (35 / Real.sqrt 35) = Real.sqrt 35 :=
  sorry

end rationalize_sqrt_35_l117_117418


namespace valid_triangle_count_l117_117323

def point := (ℤ × ℤ)

def isValidPoint (p : point) : Prop := 
  1 ≤ p.1 ∧ p.1 ≤ 4 ∧ 1 ≤ p.2 ∧ p.2 ≤ 4

def isCollinear (p1 p2 p3 : point) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

def isValidTriangle (p1 p2 p3 : point) : Prop :=
  isValidPoint p1 ∧ isValidPoint p2 ∧ isValidPoint p3 ∧ ¬isCollinear p1 p2 p3

def numberOfValidTriangles : ℕ :=
  sorry -- This will contain the combinatorial calculations from the solution.

theorem valid_triangle_count : numberOfValidTriangles = 520 :=
  sorry -- Proof will show combinatorial result from counting non-collinear combinations.

end valid_triangle_count_l117_117323


namespace total_amount_received_l117_117868

-- Definition of the conditions
def total_books (B : ℕ) : Prop := (1 / 3 : ℝ) * B = 36

def sold_books (B : ℕ) : ℕ := (2 / 3 : ℝ) * B

def price_per_book : ℝ := 3.50

-- Main theorem that needs to be proven
theorem total_amount_received (B : ℕ) (H : total_books B) : sold_books B * price_per_book = 252 := 
  sorry

end total_amount_received_l117_117868


namespace problem1_problem2_problem3_l117_117560

-- Problem 1
theorem problem1 : (-10) - (-22) + (-8) - 13 = -9 :=
by sorry

-- Problem 2
theorem problem2 : (5 / 9 + 5 / 6 - 3 / 4) / (-1 / 36) = -23 :=
by sorry

-- Problem 3
theorem problem3 : -1^4 - |0.5 - 1| * 2 - ( -3 )^2 / ( -3 / 2 ) = 4 :=
by sorry

end problem1_problem2_problem3_l117_117560


namespace part1_part2_part3_l117_117309

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x
noncomputable def g (x : ℝ) : ℝ := (1/2)^x - x

theorem part1 (f : ℝ → ℝ) (h1 : ∃ a, 0 < a ∧ a ≠ 1 ∧ f = λ x, a^x)
              (h2 : f 3 = 1/8) : ∃ a, a = 1/2 :=
by
  obtain ⟨a, _, _, rfl⟩ := h1
  use (1/8)^(1/3)
  have ha : a^3 = 1/8 := by assumptions
  have ha' : (1/8)^(1/3) = 1/2 := by norm_num
  exact ha'

theorem part2 : ∃ x, x ∈ Icc (-(1/2)) 2 ∧ f (1/2) x = sqrt 2 :=
by
  use -1/2
  split
  · norm_num
  ·
    dsimp [f]
    rw [pow_neg, pow_one_half, one_div, inv_pow, real.sqrt_inv]
    norm_num

theorem part3 : ∃ c ∈ Ioo 0 1, g c = 0 :=
by
  let g : ℝ → ℝ := λ x, (1/2)^x - x
  have h0 : g 0 > 0 := by
    dsimp [g]
    norm_num
  have h1 : g 1 < 0 := by
    dsimp [g]
    norm_num
  exact intermediate_value_Ioo _ _ h0 h1
  · continuity
  · exact h0
  · exact h1

end part1_part2_part3_l117_117309


namespace ratio_MRI_xray_eq_three_l117_117756

-- Definitions and conditions
def cost_xray : ℝ := 250
def k : ℝ := sorry -- to be proven
def cost_MRI (k : ℝ) : ℝ := k * cost_xray
def total_cost (k : ℝ) : ℝ := cost_xray + cost_MRI(k)

-- Statement for proof
theorem ratio_MRI_xray_eq_three :
  (total_cost(k) * 0.2 = 200) → (1 + k = 4) → k = 3 :=
by
  sorry

end ratio_MRI_xray_eq_three_l117_117756


namespace opposite_of_negative_2023_l117_117050

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117050


namespace min_value_of_y_l117_117300

noncomputable def y (x : ℝ) : ℝ :=
  (Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)

theorem min_value_of_y :
  ∃ x ∈ set.Icc (0 : ℝ) 1, y x = 2 + Real.sqrt 2 :=
begin
  sorry
end

end min_value_of_y_l117_117300


namespace cost_to_paint_cube_is_correct_l117_117809

-- side length of the cube in feet
def side_length : ℝ := 30

-- cost of paint per kg
def cost_per_kg : ℝ := 100

-- area covered by 1 kg of paint in sq. ft.
def coverage_per_kg : ℝ := 15

-- total surface area of a cube with given side length
def total_surface_area (side_length : ℝ) : ℝ :=
  6 * (side_length ^ 2)

-- amount of paint required to cover the total surface area
def amount_of_paint_required (surface_area : ℝ) (coverage_per_kg : ℝ) : ℝ :=
  surface_area / coverage_per_kg

-- total cost to paint the cube
def total_cost (amount_of_paint : ℝ) (cost_per_kg : ℝ) : ℝ :=
  amount_of_paint * cost_per_kg

-- proof statement
theorem cost_to_paint_cube_is_correct :
  total_cost (amount_of_paint_required (total_surface_area side_length) coverage_per_kg) cost_per_kg = 36000 :=
by
  sorry

end cost_to_paint_cube_is_correct_l117_117809


namespace monotonicity_x2_a_div_x2_monotonicity_xn_a_div_xn_l117_117310

-- Define the monotonicity of y = x^2 + a/x^2 for a > 0
theorem monotonicity_x2_a_div_x2 (a : ℝ) (ha : a > 0) :
  (∀ x, 0 < x ∧ x ≤ real.sqrt (real.sqrt a) → 
    (λ x, x^2 + a / x^2) x ≤ (λ x, x^2 + a / x^2) (real.sqrt (real.sqrt a))) ∧
  (∀ x, real.sqrt (real.sqrt a) ≤ x → 
    (λ x, x^2 + a / x^2) (real.sqrt (real.sqrt a)) ≤ (λ x, x^2 + a / x^2) x) :=
sorry

-- Define the monotonicity of the generalized function y = x^n + a/x^n for a > 0 and n > 0
theorem monotonicity_xn_a_div_xn (a : ℝ) (n : ℕ) (ha : a > 0) (hn : n > 0) :
  (even n → 
    (∀ x, 0 < x ∧ x ≤ real.sqrt (2 * n * real.sqrt a) → 
      (λ x, x^n + a / x^n) x ≤ (λ x, x^n + a / x^n) (real.sqrt (2 * n * real.sqrt a))) ∧
    (∀ x, real.sqrt (2 * n * real.sqrt a) ≤ x → 
      (λ x, x^n + a / x^n) (real.sqrt (2 * n * real.sqrt a)) ≤ (λ x, x^n + a / x^n) x)) ∧
  (odd n →
    (∀ x, 0 < x ∧ x ≤ real.sqrt (2 * n * real.sqrt a) → 
      (λ x, x^n + a / x^n) x ≤ (λ x, x^n + a / x^n) (real.sqrt (2 * n * real.sqrt a))) ∧
    (∀ x, real.sqrt (2 * n * real.sqrt a) ≤ x → 
      (λ x, x^n + a / x^n) (real.sqrt (2 * n * real.sqrt a)) ≤ (λ x, x^n + a / x^n) x)) :=
sorry

end monotonicity_x2_a_div_x2_monotonicity_xn_a_div_xn_l117_117310


namespace valid_votes_for_candidate_a_l117_117132

theorem valid_votes_for_candidate_a (total_votes : ℕ) (invalid_percentage : ℝ) (candidate_a_percentage : ℝ) (valid_votes_a : ℝ) :
  total_votes = 560000 ∧ invalid_percentage = 0.15 ∧ candidate_a_percentage = 0.80 →
  valid_votes_a = (candidate_a_percentage * (1 - invalid_percentage) * total_votes) := 
sorry

end valid_votes_for_candidate_a_l117_117132


namespace parabola_vertex_coordinates_l117_117808

theorem parabola_vertex_coordinates :
  (∃ x : ℝ, (λ x, x^2 - 2) = (0, -2)) :=
sorry

end parabola_vertex_coordinates_l117_117808


namespace opposite_of_negative_2023_l117_117051

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117051


namespace range_of_a_l117_117658

theorem range_of_a {a : ℝ} (h : ∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) :
  a ≤ -1 ∧ a ≠ -2 := 
sorry

end range_of_a_l117_117658


namespace opposite_of_neg2023_l117_117002

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l117_117002


namespace symmedian_A_l117_117744

variables {A B C D E F D1 E1 F1 M N : Type*}
variables [affine_space A B C D E F]
variables [affine_space A B D1 E1 F1]
variables [affine_space D E F M N]

structure Triangle (A B C : Type*) := 
  (side_AB : A → B → Type*)
  (side_BC : B → C → Type*)
  (side_CA : C → A → Type*)

-- Definitions of parallelism and intersection
def parallel (l1 l2 : line) : Prop := sorry

def intersect (l1 l2 : line) (P : point) : Prop := sorry

-- Given definitions according to the problem
variables (DE : line) (AB : line) (DF : line) (AC : line) 
  (circle_passing_D_E_F : circle) (BC : line) (CA : line) (D1 E1 F1 : point)
  (F1D1 : line) (D1E1 : line)

def problem (A B C : Type*) [triangle : Triangle A B C] (D F : Type*) [line_parallel : parallel DE AB]
  [line_parallel' : parallel DF AC] (circle : circle) [intersect : intersect circle BC D1]
  [intersect' : intersect circle CA E1] [intersect' : intersect circle AB F1]
  [intersect'' : intersect DE F1D1 M] [intersect''' : intersect DF D1E1 N] : Prop :=
  sorry

theorem symmedian_A (A B C D E F D1 E1 F1 M N : Type*) 
  [triangle : Triangle A B C] (DE : line) (DF : line) (circle_passing_D_E_F : circle)
  [line_parallel : parallel DE AB] [line_parallel' : parallel DF AC]
  [intersect : intersect circle_passing_D_E_F BC D1] [intersect' : intersect circle_passing_D_E_F CA E1]
  [intersect'' : intersect circle_passing_D_E_F AB F1]
  [intersect_1 : intersect DE F1D1 M] [intersect_2 : intersect DF D1E1 N] : 
  problem A B C D E F D1 E1 F1 M N :=
sorry

end symmedian_A_l117_117744


namespace buses_needed_l117_117827

theorem buses_needed (num_classrooms : ℕ) (students_per_classroom : ℕ) (seats_per_bus : ℕ) :
  num_classrooms = 67 →
  students_per_classroom = 66 →
  seats_per_bus = 6 →
  (67 * 66 + 5) / 6 = 738 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end buses_needed_l117_117827


namespace problem1_line_equation_problem2_circle_equation_l117_117505

-- Problem 1: Equation of a specific line
def line_intersection (x y : ℝ) : Prop := 
  2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0

def line_perpendicular (x y : ℝ) : Prop :=
  6 * x - 8 * y + 3 = 0

noncomputable def find_line (x y : ℝ) : Prop :=
  ∃ (l : ℝ), (8 * x + 6 * y + l = 0) ∧ 
  line_intersection x y ∧ line_perpendicular x y

theorem problem1_line_equation : ∃ (x y : ℝ), find_line x y :=
sorry

-- Problem 2: Equation of a specific circle
def point_A (x y : ℝ) : Prop := 
  x = 5 ∧ y = 2

def point_B (x y : ℝ) : Prop := 
  x = 3 ∧ y = -2

def center_on_line (x y : ℝ) : Prop :=
  2 * x - y = 3

noncomputable def find_circle (x y r : ℝ) : Prop :=
  ((x - 2)^2 + (y - 1)^2 = r) ∧
  ∃ x1 y1 x2 y2, point_A x1 y1 ∧ point_B x2 y2 ∧ center_on_line x y ∧ ((x1 - x)^2 + (y1 - y)^2 = r)

theorem problem2_circle_equation : ∃ (x y r : ℝ), find_circle x y 10 :=
sorry

end problem1_line_equation_problem2_circle_equation_l117_117505


namespace find_j_parallel_line_l117_117906

theorem find_j_parallel_line :
  ∃ j : ℝ, (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) = (2, -9) ∧ (x₂, y₂) = (j, 17) →
    (2 * x + 3 * y = 21) → (17 - (-9)) / (j - 2) = -2 / 3) :=
begin
  use -37,
  intros x₁ y₁ x₂ y₂ h1 h2,
  cases h1,
  cases h2,
  sorry
end

end find_j_parallel_line_l117_117906


namespace blue_notebook_cost_l117_117402

theorem blue_notebook_cost
  (total_spent : ℕ)
  (total_notebooks : ℕ)
  (red_notebooks : ℕ)
  (red_notebook_cost : ℕ)
  (green_notebooks : ℕ)
  (green_notebook_cost : ℕ)
  (blue_notebook_cost : ℕ)
  (h₀ : total_spent = 37)
  (h₁ : total_notebooks = 12)
  (h₂ : red_notebooks = 3)
  (h₃ : red_notebook_cost = 4)
  (h₄ : green_notebooks = 2)
  (h₅ : green_notebook_cost = 2)
  (h₆ : total_spent = red_notebooks * red_notebook_cost + green_notebooks * green_notebook_cost + blue_notebook_cost * (total_notebooks - red_notebooks - green_notebooks)) :
  blue_notebook_cost = 3 := by
  sorry

end blue_notebook_cost_l117_117402


namespace average_k_of_polynomial_with_int_roots_l117_117295

theorem average_k_of_polynomial_with_int_roots :
  (∀ r1 r2 : ℕ, r1 * r2 = 36 → ∃ k : ℕ, k = r1 + r2) ∧ 
  (list.avg [37, 20, 15, 13, 12] = 19.4) :=
by
  sorry

end average_k_of_polynomial_with_int_roots_l117_117295


namespace emus_count_l117_117973

theorem emus_count (E : ℕ) (heads : ℕ) (legs : ℕ) 
  (h_heads : ∀ e : ℕ, heads = e) 
  (h_legs : ∀ e : ℕ, legs = 2 * e)
  (h_total : heads + legs = 60) : 
  E = 20 :=
by sorry

end emus_count_l117_117973


namespace range_of_a_l117_117265

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f''' x > 0)
  (h2 : ∀ x, f (f x - exp x) = 1)
  (h3 : ∀ x, f x ≥ a * x + a) :
  0 ≤ a ∧ a ≤ 1 := 
sorry

end range_of_a_l117_117265


namespace area_OAB_l117_117192

variable {x y : ℝ}

def ellipse_eq (x : ℝ) (y : ℝ) : Prop := (x^2 / 5) + (y^2 / 4) = 1
def line_eq (x : ℝ) (y : ℝ) : Prop := y = 2 * (x - 1)
def O : (ℝ × ℝ) := (0, 0)
def A : (ℝ × ℝ) := (0, -2)
def B : (ℝ × ℝ) := (5 / 3, 4 / 3)
def F : (ℝ × ℝ) := (1, 0)

theorem area_OAB : 
  let area_triangle (O A B : (ℝ × ℝ)) : ℝ := 1 / 2 * abs (O.1 * (A.2 - B.2) + A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2)) in
  ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2 ∧ line_eq A.1 A.2 ∧ line_eq B.1 B.2 ∧ F = (1, 0) -> 
  area_triangle O A B = 5 / 3 :=
by
  sorry

end area_OAB_l117_117192


namespace concurrency_of_lines_l117_117761

theorem concurrency_of_lines 
  (A B C F E : Type) 
  [triangle : Triangle ABC] 
  (ABF : Triangle ABF) 
  (ACE : Triangle ACE)
  (RightAngle : ∠ ABF = 90)
  (SimTriangles : ∼ ACE ABF) :
  Concurrent BF CF (altitude AH) :=
by
  sorry

end concurrency_of_lines_l117_117761


namespace train_length_l117_117524

-- Defining the given speed in kmph
def speed_kmph : Float := 50

-- Converting speed to m/s
def speed_mps : Float := (speed_kmph * 1000) / 3600

-- Defining the time taken to pass the telegraph post in seconds
def time_seconds : Float := 5.399568034557236

-- The distance is given by speed * time
def distance : Float := speed_mps * time_seconds

theorem train_length : distance ≈ 75 := sorry

end train_length_l117_117524


namespace range_of_a_l117_117996

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

noncomputable def g (a x : ℝ) : ℝ := a * x + 2

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x1 ∈ set.Icc (-1 : ℝ) 2, ∃ x0 ∈ set.Icc (-1 : ℝ) 2, g a x1 = f x0) → (0 < a ∧ a ≤ (1/2)) :=
sorry

end range_of_a_l117_117996


namespace minor_axis_length_max_AF2_BF2_equal_min_distance_AB_l117_117271

noncomputable def ellipse : Type := sorry

structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < 2
  eq : ∀ x y : ℝ, x^2 / 4 + y^2 / (b^2) = 1

structure Points where
  A : ellipse → ℝ × ℝ
  B : ellipse → ℝ × ℝ
  F1 : ellipse → ℝ × ℝ
  F2 : ellipse → ℝ × ℝ

def max_AF2_BF2 (e : ellipse) (p : Points) : ℝ := 5

theorem minor_axis_length (e : ellipse) (p : Points) : 2 * e.b = 2 * Real.sqrt 3 :=
by sorry

theorem max_AF2_BF2_equal (e : ellipse) (p : Points) : max_AF2_BF2 e p = 5 → |p.A.2 - p.B.2| = 0 :=
by sorry

theorem min_distance_AB (e : ellipse) (p : Points) : |p.A.1 - p.B.1| = 3 :=
by sorry

end minor_axis_length_max_AF2_BF2_equal_min_distance_AB_l117_117271


namespace area_of_min_def_l117_117352

variables {A B C D E F : Type*}
variables [EuclideanGeometry A B C]
variables {a b c : ℝ} (S : ℝ)

-- Assume that ABC is an acute triangle with sides BC = a, AC = b, AB = c
-- and points D, E, F are on sides BC, CA, AB respectively.
-- We want to establish the relationship for the area of DEF.

theorem area_of_min_def (h_acute : ∀ {x y z : Type*}, acute_triangle x y z) (h_bc : BC = a) (h_ca : AC = b) (h_ab : AB = c) :
  let S := area_of_triangle A B C in
  let minimized_area := (12 * S ^ 3) / (a ^ 2 + b ^ 2 + c ^ 2) ^ 2 in
  area_of_triangle D E F = minimized_area :=
sorry

end area_of_min_def_l117_117352


namespace number_of_boys_false_l117_117822

def total_students : ℝ := 1
def percentage_boys : ℝ := 0.54
def number_of_boys : ℝ := 54

theorem number_of_boys_false :
  ¬ (number_of_boys = percentage_boys * total_students * total_students.recip) :=
by
  sorry

end number_of_boys_false_l117_117822


namespace sum_of_valid_c_l117_117573

theorem sum_of_valid_c (h : ∀ c : ℤ, c ≤ 19 → ∃ k : ℤ, 25 + 8 * c = k * k) : (∑ c in [-3, -2, 3, 12], c) = 10 :=
by
  sorry

end sum_of_valid_c_l117_117573


namespace cost_of_paving_floor_l117_117819

theorem cost_of_paving_floor 
  (Length : ℝ) (Width : ℝ) (Rate : ℝ) 
  (h_Length : Length = 5.5) (h_Width : Width = 3.75) (h_Rate : Rate = 1400) : 
  Length * Width * Rate = 28875 :=
by
  rw [h_Length, h_Width, h_Rate]
  norm_num
  sorry

end cost_of_paving_floor_l117_117819


namespace perimeter_of_new_figure_l117_117530

-- Define the initial conditions
def side_length_of_square (perimeter : ℝ) : ℝ := perimeter / 4
def hypotenuse_of_right_triangle (leg : ℝ) : ℝ := Real.sqrt (leg^2 + leg^2)

-- Prove the final perimeter of the new figure
theorem perimeter_of_new_figure (perimeter : ℝ) (leg : ℝ) (h1 : perimeter = 48) (h2 : leg = 12) :
  let side := side_length_of_square perimeter in
  let hypotenuse := hypotenuse_of_right_triangle leg in
  3 * side + hypotenuse = 36 + 12 * Real.sqrt 2 :=
by
  let side := side_length_of_square perimeter
  let hypotenuse := hypotenuse_of_right_triangle leg
  -- the actual proof will go here
  sorry

end perimeter_of_new_figure_l117_117530


namespace percentage_cats_less_dogs_l117_117704

theorem percentage_cats_less_dogs (C D F : ℕ) (h1 : C < D) (h2 : F = 2 * D) (h3 : C + D + F = 304) (h4 : F = 160) :
  ((D - C : ℕ) * 100 / D : ℕ) = 20 := 
sorry

end percentage_cats_less_dogs_l117_117704


namespace proof_probability_and_expectations_l117_117095

/-- Number of white balls drawn from two boxes --/
def X : ℕ := 1

/-- Number of red balls drawn from two boxes --/
def Y : ℕ := 1

/-- Given the conditions, the probability of drawing one white ball is 1/2, and
the expected value of white balls drawn is greater than the expected value of red balls drawn --/
theorem proof_probability_and_expectations :
  (∃ (P_X : ℚ), P_X = 1 / 2) ∧ (∃ (E_X E_Y : ℚ), E_X > E_Y) :=
by {
  sorry
}

end proof_probability_and_expectations_l117_117095


namespace distinct_triangles_count_l117_117698

def side_lengths_are_valid (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ 5 ∧ b ≤ 5 ∧ c ≤ 5

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the problem statement: There exist exactly 22 distinct triangles
-- where the sides are positive integers not greater than 5 and satisfy
-- the triangle inequality.
theorem distinct_triangles_count : 
  set.count (λ (a b c : ℕ), side_lengths_are_valid a b c ∧ triangle_inequality a b c) {<a | (a,b) | (a,b,c)} = 22 :=
sorry

end distinct_triangles_count_l117_117698


namespace constant_term_expansion_l117_117619

theorem constant_term_expansion :
  let a := ∫ x in (0 : ℝ)..(Real.pi / 6), Real.cos x
  x * (x - (1 / (a * x))) ^ 7
  has_constant_term -128 :=
by 
  sorry

end constant_term_expansion_l117_117619


namespace find_BP_l117_117890

-- Define points
variables {A B C D P : Type}  

-- Define lengths
variables (AP PC BP DP BD : ℝ)

-- Provided conditions
axiom h1 : AP = 10
axiom h2 : PC = 2
axiom h3 : BD = 9

-- Assume intersect and lengths relations setup
axiom intersect : BP < DP
axiom power_of_point : AP * PC = BP * DP

-- Target statement
theorem find_BP (h1 : AP = 10) (h2 : PC = 2) (h3 : BD = 9)
  (intersect : BP < DP) (power_of_point : AP * PC = BP * DP) : BP = 4 :=
  sorry

end find_BP_l117_117890


namespace equal_sunday_tuesday_count_l117_117167

theorem equal_sunday_tuesday_count (h : ∀ (d : ℕ), d < 7 → d ≠ 0 → d ≠ 1 → d ≠ 2 → d ≠ 3) :
  ∃! d, d = 4 :=
by
  -- proof here
  sorry

end equal_sunday_tuesday_count_l117_117167


namespace opposite_of_neg_2023_l117_117073

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117073


namespace savings_difference_correct_l117_117964

noncomputable def savings_1989_dick : ℝ := 5000
noncomputable def savings_1989_jane : ℝ := 5000

noncomputable def savings_1990_dick : ℝ := savings_1989_dick + 0.10 * savings_1989_dick
noncomputable def savings_1990_jane : ℝ := savings_1989_jane - 0.05 * savings_1989_jane

noncomputable def savings_1991_dick : ℝ := savings_1990_dick + 0.07 * savings_1990_dick
noncomputable def savings_1991_jane : ℝ := savings_1990_jane + 0.08 * savings_1990_jane

noncomputable def savings_1992_dick : ℝ := savings_1991_dick - 0.12 * savings_1991_dick
noncomputable def savings_1992_jane : ℝ := savings_1991_jane + 0.15 * savings_1991_jane

noncomputable def total_savings_dick : ℝ :=
savings_1989_dick + savings_1990_dick + savings_1991_dick + savings_1992_dick

noncomputable def total_savings_jane : ℝ :=
savings_1989_jane + savings_1990_jane + savings_1991_jane + savings_1992_jane

noncomputable def difference_of_savings : ℝ :=
total_savings_dick - total_savings_jane

theorem savings_difference_correct :
  difference_of_savings = 784.30 :=
by sorry

end savings_difference_correct_l117_117964


namespace sum_of_distances_l117_117709

noncomputable def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (t + 1/t, t/2 - 1/(2*t))

noncomputable def polar_line (ρ θ : ℝ) : Prop :=
  ρ * real.sin (θ + π / 4) = 2 * real.sqrt 2

def cartesian_point : ℝ × ℝ := (0, 4)

theorem sum_of_distances
  (A B : ℝ × ℝ)
  (t1 t2 : ℝ)
  (hA : parametric_curve t1 = A)
  (hB : parametric_curve t2 = B)
  (intersect : ∃ ρ θ, polar_line ρ θ ∧ ∃ x y, A = (x, y) ∧ B = (x, y) ∧ x + y = 4) :
  abs (real.sqrt ((A.fst - cartesian_point.fst)^2 + (A.snd - cartesian_point.snd)^2)) +
  abs (real.sqrt ((B.fst - cartesian_point.fst)^2 + (B.snd - cartesian_point.snd)^2)) =
  32 * real.sqrt 2 / 3 := 
sorry

end sum_of_distances_l117_117709


namespace opposite_of_neg_2023_l117_117034

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117034


namespace range_of_a_l117_117655

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) ↔ (a ≤ -1 ∧ a ≠ -2) :=
by
  sorry

end range_of_a_l117_117655


namespace perpendicular_distance_from_centroid_l117_117719

-- Define the points A, B, C
variables (A B C D E F G : Type)
variables [AffineSpace A B] [AffineSpace A C] [AffineSpace A D] [AffineSpace A E] [AffineSpace A F] [AffineSpace A G]

-- Define the lengths AD, BE, CF
variables (AD BE CF : ℝ) 

-- The given conditions
axiom AD_eq_15 : AD = 15
axiom BE_eq_9 : BE = 9
axiom CF_eq_27 : CF = 27

-- Prove the length of GH, where G is the centroid of triangle ABC
theorem perpendicular_distance_from_centroid
  (AD BE CF : ℝ)
  (AD_eq_15 : AD = 15)
  (BE_eq_9 : BE = 9)
  (CF_eq_27 : CF = 27)
  : 
  let y_G := (15 + 9 + 27) / 3 in 
  y_G = 17 :=
by 
  sorry

end perpendicular_distance_from_centroid_l117_117719


namespace minimum_value_of_f_range_of_m_l117_117302

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + (1/x) - 1 else -x^3 + 1

def p (m : ℝ) : Prop := ∀ x : ℝ, f x ≥ m^2 + 2*m - 2

def q (m : ℝ) : Prop := (m^2 - 1) > 1

theorem minimum_value_of_f : ∃ x : ℝ, f x = 1 :=
  sorry

theorem range_of_m (m : ℝ) :
  ((∀ x : ℝ, f x ≥ m^2 + 2 * m - 2) ∨ (m^2 - 1 > 1)) ∧ ¬((∀ x : ℝ, f x ≥ m^2 + 2 * m - 2) ∧ (m^2 - 1 > 1)) ↔
    (m < -3 ∨ (-real.sqrt 2 ≤ m ∧ m ≤ 1) ∨ m > real.sqrt 2) :=
  sorry

end minimum_value_of_f_range_of_m_l117_117302


namespace polynomial_coeff_ratio_l117_117641

theorem polynomial_coeff_ratio :
  let p := (2 - x)^5
  let a_0 := polynomial.coeff p 0
  let a_1 := polynomial.coeff p 1
  let a_2 := polynomial.coeff p 2
  let a_3 := polynomial.coeff p 3
  let a_4 := polynomial.coeff p 4
  let a_5 := polynomial.coeff p 5
  (a_2 + a_4) / (a_1 + a_3) = -3 / 4 := 
by
  sorry

end polynomial_coeff_ratio_l117_117641


namespace number_of_emus_l117_117971

theorem number_of_emus (total_heads_and_legs : ℕ) (heads_per_emu legs_per_emu : ℕ) (total_emu : ℕ) :
  total_heads_and_legs = 60 → 
  heads_per_emu = 1 → 
  legs_per_emu = 2 → 
  total_emu = total_heads_and_legs / (heads_per_emu + legs_per_emu) → 
  total_emu = 20 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  refine h4
  sorry

end number_of_emus_l117_117971


namespace initial_percentage_water_is_80_l117_117512

noncomputable def initial_kola_solution := 340
noncomputable def added_sugar := 3.2
noncomputable def added_water := 10
noncomputable def added_kola := 6.8
noncomputable def final_percentage_sugar := 14.111111111111112
noncomputable def percentage_kola := 6

theorem initial_percentage_water_is_80 :
  ∃ (W : ℝ), W = 80 :=
by
  sorry

end initial_percentage_water_is_80_l117_117512


namespace isosceles_triangle_circumcircle_condition_l117_117112

theorem isosceles_triangle_circumcircle_condition (ABC : Triangle)
  (h1 : ABC.isIsosceles AB BC)
  (M : Point)
  (h2 : M ∈ circumcircle ABC AC ∧ M ≠ B) :
  MA + MC = 2 * MB * cos (angle A) :=
by
  sorry

end isosceles_triangle_circumcircle_condition_l117_117112


namespace probability_uniform_same_color_l117_117937

noncomputable def probability_same_color (choices : List String) (athleteA: ℕ) (athleteB: ℕ) : ℚ :=
  if choices.length = 3 ∧ athleteA ∈ [0,1,2] ∧ athleteB ∈ [0,1,2] then
    1 / 3
  else
    0

theorem probability_uniform_same_color :
  probability_same_color ["red", "white", "blue"] 0 1 = 1 / 3 :=
by
  sorry

end probability_uniform_same_color_l117_117937


namespace proof_l117_117981

noncomputable def problem_statement : Prop :=
  ( ( (Real.sqrt 1.21 * Real.sqrt 1.44) / (Real.sqrt 0.81 * Real.sqrt 0.64)
    + (Real.sqrt 1.0 * Real.sqrt 3.24) / (Real.sqrt 0.49 * Real.sqrt 2.25) ) ^ 3 
  = 44.6877470366 )

theorem proof : problem_statement := 
  by
  sorry

end proof_l117_117981


namespace alan_glasses_drank_l117_117759

-- Definition for the rate of drinking water
def glass_per_minutes := 1 / 20

-- Definition for the total time in minutes
def total_minutes := 5 * 60

-- Theorem stating the number of glasses Alan will drink in the given time
theorem alan_glasses_drank : (glass_per_minutes * total_minutes) = 15 :=
by 
  sorry

end alan_glasses_drank_l117_117759


namespace profit_percent_is_25_l117_117915

-- Given conditions as definitions
def purchase_price : ℝ := 225
def overhead_expenses : ℝ := 15
def selling_price : ℝ := 300

-- Proof statement
theorem profit_percent_is_25 :
  let cost_price := purchase_price + overhead_expenses in
  let profit := selling_price - cost_price in
  let profit_percent := (profit / cost_price) * 100 in
  profit_percent = 25 := 
by
  sorry

end profit_percent_is_25_l117_117915


namespace shopkeeper_overall_profit_l117_117533

-- Define the initial quantities of fruits
def apples : ℝ := 280
def oranges : ℝ := 150
def bananas : ℝ := 100

-- Define the percentages sold at higher profit and corresponding profit percentages
def higher_profit_percentage (fruit: ℝ) (percentage: ℝ) : ℝ := (percentage / 100) * fruit
def higher_apples : ℝ := higher_profit_percentage apples 40
def higher_oranges : ℝ := higher_profit_percentage oranges 45
def higher_bananas : ℝ := higher_profit_percentage bananas 50

def profit_percentage (quantity: ℝ) (percentage: ℝ) : ℝ := (percentage / 100) * quantity
def higher_apples_profit : ℝ := profit_percentage higher_apples 20
def higher_oranges_profit : ℝ := profit_percentage higher_oranges 25
def higher_bananas_profit : ℝ := profit_percentage higher_bananas 30

-- Define the amounts sold at lower profit
def lower_apples : ℝ := apples - higher_apples
def lower_oranges : ℝ := oranges - higher_oranges
def lower_bananas : ℝ := bananas - higher_bananas

def lower_apples_profit : ℝ := profit_percentage lower_apples 15
def lower_oranges_profit : ℝ := profit_percentage lower_oranges 15
def lower_bananas_profit : ℝ := profit_percentage lower_bananas 15

-- Calculate the total profits
def total_profit : ℝ := higher_apples_profit + higher_oranges_profit + higher_bananas_profit +
                        lower_apples_profit + lower_oranges_profit + lower_bananas_profit

-- Calculate the total weight of fruits sold
def total_weight : ℝ := apples + oranges + bananas

-- Calculate the overall percentage profit
def overall_profit_percentage : ℝ := (total_profit / total_weight) * 100

-- The statement to prove
theorem shopkeeper_overall_profit : overall_profit_percentage = 18.75 := by
  sorry

end shopkeeper_overall_profit_l117_117533


namespace find_number_l117_117141

theorem find_number (x : ℝ) :
  (1.5 * 1265) / x = 271.07142857142856 → x = 7 :=
by
  intro h
  sorry

end find_number_l117_117141


namespace dodecahedron_coloring_inscribed_cube_transformations_volume_of_union_of_inscribed_cubes_l117_117500

def regular_dodecahedron := Type
def four_colors := Type
def color_dodecahedron_faces_different_colors (d : regular_dodecahedron) (c : four_colors) := True
def ways_to_color_dodecahedron (d : regular_dodecahedron) (c : four_colors) := 4

def inscribed_cube := Type
def ways_to_inscribe_cube (d : regular_dodecahedron) (c : inscribed_cube) := 5

def volume (shape : Type) := ℝ 
def dodecahedron_volume := 1
def union_of_all_inscribed_cubes_volume (d : regular_dodecahedron) :=  (1/2) * (79 * real.sqrt 5 - 175)

theorem dodecahedron_coloring:
  ∀ (d : regular_dodecahedron) (c : four_colors), 
  color_dodecahedron_faces_different_colors d c → ways_to_color_dodecahedron d c = 4 := 
sorry

theorem inscribed_cube_transformations:
  ∀ (d : regular_dodecahedron) (c : inscribed_cube), 
  true → ways_to_inscribe_cube d c = 5 :=
sorry

theorem volume_of_union_of_inscribed_cubes:
  ∀ (d : regular_dodecahedron), 
  dodecahedron_volume = 1 → volume (union_of_all_inscribed_cubes_volume d) = (1 / 2) * (79 * real.sqrt 5 - 175) :=
sorry

end dodecahedron_coloring_inscribed_cube_transformations_volume_of_union_of_inscribed_cubes_l117_117500


namespace A_is_perfect_square_l117_117217

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def A := factorial 120 * factorial 121
def B := factorial 121 * factorial 123
def C := factorial 122 * factorial 123
def D := factorial 122 * factorial 124
def E := factorial 123 * factorial 124

theorem A_is_perfect_square : ∃ k : ℕ, A = k * k := by
  sorry

end A_is_perfect_square_l117_117217


namespace circumcircle_contains_symmetric_point_l117_117520

universe u
variables {α : Type u} [MetricSpace α]

-- Define the triangle ABC and point A' on BC
variables {A B C A' : α}
variables (triangle_ABC : ABC α A B C)

-- Define points M and N as intersections of perpendicular bisectors
def is_perpendicular_bisector {p₁ p₂ q : α} : Prop :=
  dist p₁ q = dist p₂ q

variables {M : α} (hM : is_perpendicular_bisector A' B M)
variables {N : α} (hN : is_perpendicular_bisector A' C N)

-- Symmetric point of A' with respect to line MN
def symmetric_point (P Q R : α) : α := sorry -- definition of the symmetric point

-- Definition of circumcircle and membership
def is_on_circumcircle {P Q R Q' : α} : Prop :=
  let center := sorry -- the center of circumcircle
  let radius := sorry -- the radius of circumcircle
  dist center Q = radius ∧ dist center Q' = radius ∧ dist center R = radius 

theorem circumcircle_contains_symmetric_point :
  is_on_circumcircle A B C (symmetric_point A' M N) :=
sorry

end circumcircle_contains_symmetric_point_l117_117520


namespace sum_as_common_fraction_l117_117595

/-- The sum of 0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 as a common fraction -/
theorem sum_as_common_fraction : (0.2 + 0.04 + 0.006 + 0.0008 + 0.00010) = (12345 / 160000) := by
  sorry

end sum_as_common_fraction_l117_117595


namespace opposite_of_negative_2023_l117_117044

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117044


namespace sasha_mistake_l117_117484

/-- If Sasha obtained three numbers by raising 4 to various powers, such that all three units digits are different, 
     then Sasha's numbers cannot have three distinct last digits. -/
theorem sasha_mistake (h : ∀ n1 n2 n3 : ℕ, ∃ k1 k2 k3, n1 = 4^k1 ∧ n2 = 4^k2 ∧ n3 = 4^k3 ∧ (n1 % 10 ≠ n2 % 10) ∧ (n2 % 10 ≠ n3 % 10) ∧ (n1 % 10 ≠ n3 % 10)) :
False :=
sorry

end sasha_mistake_l117_117484


namespace find_a_n_and_T_n_l117_117361

-- Definitions for the geometric sequence and given conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = a n * q
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = (finset.range n).sum (λ k, a k)
def S2_condition (S : ℕ → ℝ) := S 2 = 3
def S4_condition (S : ℕ → ℝ) := S 4 = 15
def geometric_sequence_sum_formula (a : ℕ → ℝ) (q : ℝ) : ℕ → ℝ 
| 0       := 0
| (n + 1) := a 0 * ((1 - q^(n + 1)) / (1 - q))

-- The main statement that encapsulates the proof goals
theorem find_a_n_and_T_n (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ) (q : ℝ) 
  (hq : q > 0)
  (ha : geometric_sequence a q)
  (hS : sum_of_first_n_terms a S)
  (hS2 : S2_condition S)
  (hS4 : S4_condition S) :
  (∀ n, a n = 2^(n-1)) ∧ (∀ n, T n = 2^(n+1) - n - 2) :=
by
  sorry

end find_a_n_and_T_n_l117_117361


namespace integral_solution_l117_117203

noncomputable def integral_problem (a b : ℝ) (h : a > b ∧ b > 0) : ℝ :=
  ∫ x in 0..(2 * Real.pi), (1 / (a + b * Real.cos x)^2)

theorem integral_solution (a b : ℝ) (h : a > b ∧ b > 0) :
  integral_problem a b h = (2 * Real.pi * a) / (a^2 - b^2)^(3/2) :=
by
  sorry

end integral_solution_l117_117203


namespace vector_identity_l117_117643

variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables (P A B C : V)
variables (PA PB PC AB AC AP : V)

theorem vector_identity
  (h : PA + 2 • PB + 3 • PC = 0)
  (hPA : PA = -AP)
  (hPB : PB = B - P)
  (hPC : PC = C - P)
  (hAB : AB = B - A)
  (hAC : AC = C - A) :
  AP = 1/3 • AB + 1/2 • AC :=
by sorry

end vector_identity_l117_117643


namespace sequence_bound_exists_l117_117954

def sequence (n : ℕ) : ℕ → ℝ
| 0 := 5
| (n + 1) := (sequence n ^ 2 + 5 * sequence n + 4) / (sequence n + 6)

theorem sequence_bound_exists :
  ∃ m : ℕ, 81 ≤ m ∧ m ≤ 242 ∧ sequence m ≤ 4 + 1 / 2^20 :=
sorry

end sequence_bound_exists_l117_117954


namespace _l117_117925

noncomputable def angle_ratio (O A B C E : Type*) [inner_product_space ℝ O] 
  [metric_space O] [normed_group O] [normed_space ℝ O] [normed_add_comm_group O] 
  [normed_linear_ordered_field O] (h₁ : angle (∠ A O B) = 100) 
  (h₂ : angle (∠ B O C) = 90) (h₃ : angle (∠ B A C) * 2 = angle (∠ B O C)) 
  (h₄ : dist O E = dist O B) (h₅ : ∠ E O A = 90) : ℝ :=
  angle (∠ O B E) / angle (∠ B A C)

noncomputable theorem angle_ratio_proof (O A B C E : Type*) [inner_product_space ℝ O] 
  [metric_space O] [normed_group O] [normed_space ℝ O] [normed_add_comm_group O] 
  [normed_linear_ordered_field O] (h₁ : angle (∠ A O B) = 100) 
  (h₂ : angle (∠ B O C) = 90) (h₃ : angle (∠ B A C) * 2 = angle (∠ B O C)) 
  (h₄ : dist O E = dist O B) (h₅ : ∠ E O A = 90) : 
  angle_ratio O A B C E h₁ h₂ h₃ h₄ h₅ = 19/18 :=
by
  sorry

end _l117_117925


namespace percentage_paid_l117_117114

theorem percentage_paid (X Y : ℝ) (h_sum : X + Y = 572) (h_Y : Y = 260) : (X / Y) * 100 = 120 :=
by
  -- We'll prove this result by using the conditions and solving for X.
  sorry

end percentage_paid_l117_117114


namespace range_of_a_l117_117632

variable {x a : ℝ}

def p (x : ℝ) := 2*x^2 - 3*x + 1 ≤ 0
def q (x : ℝ) (a : ℝ) := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a (h : ¬ p x → ¬ q x a) : 0 ≤ a ∧ a ≤ 1/2 := by
  sorry

end range_of_a_l117_117632


namespace cosine_angle_AM_BC_l117_117392

noncomputable theory
open_locale real_inner_product_space

def_point_vectors
  (A B C M : ℝ^2) 
  (h_equilateral : abs (B - A) = abs (C - B) ∧ abs (C - B) = abs (C - A))
  (h_AM : M - A = (1 / 3) * (B - A) + (2 / 3) * (C - A))
  (BC := C - B)
  (AM := M - A)
: ℝ := 
  real.arccos ((AM ⬝ BC) / (|AM| * |BC|))
  
theorem cosine_angle_AM_BC 
  (A B C M : ℝ^2) 
  (h_equilateral : abs (B - A) = abs (C - B) ∧ abs (C - B) = abs (C - A))
  (h_AM : M - A = (1 / 3) * (B - A) + (2 / 3) * (C - A)) 
  (h_cos : def_point_vectors
            A B C M
            h_equilateral
            h_AM
            ((C - B) = BC)
            ((M - A) = AM)) : 
  arccos ((AM ⬝ BC) / (|AM| * |BC|)) = √7 / 14 :=
begin
  sorry
end

end cosine_angle_AM_BC_l117_117392


namespace stan_average_speed_l117_117787

def first_segment_distance : ℝ := 320
def first_segment_time : ℝ := 4 + 30 / 60

def second_segment_distance : ℝ := 400
def second_segment_time : ℝ := 6 + 20 / 60

def total_distance : ℝ := first_segment_distance + second_segment_distance
def total_time : ℝ := first_segment_time + second_segment_time

def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem stan_average_speed :
  average_speed total_distance total_time ≈ 66.48 := sorry

end stan_average_speed_l117_117787


namespace cos_difference_simplification_l117_117774

theorem cos_difference_simplification :
  let x := Real.cos (20 * Real.pi / 180)
  let y := Real.cos (40 * Real.pi / 180)
  (y = 2 * x^2 - 1) →
  (x = 1 - 2 * y^2) →
  x - y = 1 / 2 :=
by
  intros x y h1 h2
  sorry

end cos_difference_simplification_l117_117774


namespace opposite_of_neg_2023_l117_117064

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117064


namespace part1_part2_part3_l117_117636

open Real

noncomputable def g (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x
noncomputable def h (x : ℝ) : ℝ := log x
noncomputable def f (a x : ℝ) : ℝ := g a x + h x

theorem part1 (a : ℝ) (h_a : a = 1) : 
  let g := g a
  let g' := 2 * x - 3
  ∀ (x y : ℝ), y = g 1 → (x + y + 1 = 0) → (g 1 = g' 1 := by sorry

theorem part2 (a : ℝ) (h_positive : 0 < a) (h_fmin : ∀ x, 1 ≤ x ∧ x ≤ exp 1 → -2 ≤ f a x) : 
  a = 1 := by sorry

theorem part3 (a : ℝ) (h_ineq : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → 
  (f a x1 - f a x2) / (x1 - x2) > -2) : 
  0 ≤ a ∧ a ≤ 8 := by sorry

end part1_part2_part3_l117_117636


namespace oranges_remaining_l117_117097

theorem oranges_remaining (initial_oranges jonathan_takes : ℕ) (h1 : initial_oranges = 96) (h2 : jonathan_takes = 45) : initial_oranges - jonathan_takes = 51 :=
by {
  rw [h1, h2],
  norm_num,
  }

end oranges_remaining_l117_117097


namespace total_apples_picked_l117_117198

def benny_apples : Nat := 2
def dan_apples : Nat := 9

theorem total_apples_picked : benny_apples + dan_apples = 11 := 
by
  sorry

end total_apples_picked_l117_117198


namespace question_1_question_2_l117_117150

def profit (x : ℝ) : ℝ := 5 * x + 1 - 3 / x

theorem question_1 (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 10) : 3 ≤ x ∧ x ≤ 10 ↔ 2 * (profit x) ≥ 30 :=
sorry

theorem question_2 (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 10) (y : ℝ) : 
  (∀ t, (1 : ℝ) ≤ t → t ≤ 10 → 120 * (-3 / t^2 + 1 / t + 5) ≤ y) →
  x = 6 ∧ y = 610 :=
sorry

end question_1_question_2_l117_117150


namespace sec_240_eq_neg2_l117_117597

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_240_eq_neg2 : sec 240 = -2 := by
  -- Proof goes here
  sorry

end sec_240_eq_neg2_l117_117597


namespace alex_activities_40_points_l117_117183

theorem alex_activities_40_points :
  let activities (points: ℕ) : ℕ :=
    if points ≤ 10 then points * 1
    else if points ≤ 20 then 10 * 1 + (points - 10) * 2
    else if points ≤ 30 then 10 * 1 + 10 * 2 + (points - 20) * 3
    else 10 * 1 + 10 * 2 + 10 * 3 + (points - 30) * 4
  in activities 40 = 100 :=
by
  intro activities
  unfold activities
  simp
  sorry

end alex_activities_40_points_l117_117183


namespace total_money_tshirts_l117_117429

-- Conditions
def price_per_tshirt : ℕ := 62
def num_tshirts_sold : ℕ := 183

-- Question: prove the total money made from selling the t-shirts
theorem total_money_tshirts :
  num_tshirts_sold * price_per_tshirt = 11346 := 
by
  -- Proof goes here
  sorry

end total_money_tshirts_l117_117429


namespace solution_set_of_inequality_l117_117085

theorem solution_set_of_inequality (x : ℝ) : 2 * x - 6 < 0 ↔ x < 3 := 
by
  sorry

end solution_set_of_inequality_l117_117085


namespace binom_identity_l117_117767

-- Definition: Combinatorial coefficient (binomial coefficient)
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binom_identity (n k : ℕ) (h : k ≤ n) :
  binom (n + 1) k = binom n k + binom n (k - 1) := by
  sorry

end binom_identity_l117_117767


namespace octagon_diagonals_eq_20_l117_117686

theorem octagon_diagonals_eq_20 :
  let n := 8 in
  D = n * (n - 3) / 2 →
  D = 20 :=
by
  intros n H
  sorry

end octagon_diagonals_eq_20_l117_117686


namespace euler_line_l117_117773

variables {V : Type*} [inner_product_space ℝ V]

-- Define the points A, B, C in the vector space V
variables (A B C : V)

-- Define O as the circumcenter in the vector space V
variables (O : V)

-- Define the centroid G in terms of vectors
def centroid (A B C O : V) : V := (A + B + C) / 3

-- Define the orthocenter H in terms of vectors
def orthocenter (A B C O : V) : V := A + B + C

-- The proof statement asserting collinearity using vector conditions
theorem euler_line (A B C O : V) : let G : V := centroid A B C O,
                                      H : V := orthocenter A B C O
                                   in H = 3 • G :=
by sorry

end euler_line_l117_117773


namespace variance_comparison_l117_117902

variables (x1 x2 : ℝ) (S1 S2 : ℝ)

-- Axiom: The customer selected large and uniform oranges from a diverse batch.
axiom chosen_criteria : ∀ x, (x ∈ original_oranges → ∃ x' ∈ selected_oranges, x' > x)

-- Definition of average and variance for original and selected oranges
def avg_original_weight : ℝ := x1
def variance_original_weight : ℝ := S1^2
def avg_selected_weight : ℝ := x2
def variance_selected_weight : ℝ := S2^2

-- Theorem: Proving S1^2 > S2^2 based on the problem's conditions
theorem variance_comparison : S1^2 > S2^2 :=
sorry

end variance_comparison_l117_117902


namespace solve_equation_l117_117783

theorem solve_equation (x : ℚ) :
  (x^2 + 3 * x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 :=
by
  sorry

end solve_equation_l117_117783


namespace ellipse_properties_correct_l117_117272

-- Define the semi-major axis
def semi_major_axis (a : ℝ) : Prop := a = 2

-- Define the semi-minor axis
def minor_axis_length (b : ℝ) : Prop := 2 * b = 2 * Real.sqrt 3

-- Define the foci-related properties
def foci_properties (b : ℝ) (AF2 BF2 : ℝ) : Prop :=
  (|AF2| + |BF2| = 5) ∧ (|AF2| = |BF2|)

-- Validate the line intersecting points
def intersection_properties (a b : ℝ) (AB : ℝ) : Prop :=
  (|AF2| + |BF2| + |AB| = 8) ∧ (|AB| = 3)

-- Define the eccentricity
def eccentricity (a c : ℝ) : Prop :=
  (c = Real.sqrt (a^2 - b^2)) ∧ (c / a ≠ Real.sqrt 3 / 3)

-- Define the final theorem to combine all properties
theorem ellipse_properties_correct (b AF2 BF2 AB : ℝ) :
  semi_major_axis 2 ∧ minor_axis_length b ∧ foci_properties b AF2 BF2 ∧ intersection_properties 2 b AB ∧ eccentricity 2 (Real.sqrt (4 - b^2)) :=
  sorry

end ellipse_properties_correct_l117_117272


namespace decreasing_function_positive_at_endpoint_implies_positive_within_interval_l117_117287

theorem decreasing_function_positive_at_endpoint_implies_positive_within_interval
  (f : ℝ → ℝ) (a b : ℝ)
  (h_deriv : ∀ x, a < x ∧ x < b → f' x < 0)
  (h_fb_gt_zero : f b > 0) :
  ∀ x, a < x ∧ x < b → f x > 0 := 
sorry

end decreasing_function_positive_at_endpoint_implies_positive_within_interval_l117_117287


namespace grazing_months_of_B_l117_117511

variable (A_cows A_months C_cows C_months D_cows D_months A_rent total_rent : ℕ)
variable (B_cows x : ℕ)

theorem grazing_months_of_B
  (hA_cows : A_cows = 24)
  (hA_months : A_months = 3)
  (hC_cows : C_cows = 35)
  (hC_months : C_months = 4)
  (hD_cows : D_cows = 21)
  (hD_months : D_months = 3)
  (hA_rent : A_rent = 1440)
  (htotal_rent : total_rent = 6500)
  (hB_cows : B_cows = 10) :
  x = 5 := 
sorry

end grazing_months_of_B_l117_117511


namespace amin_ali_equivalence_l117_117837

-- Define the conditions of the problem
def table (n : ℕ) := list (list (option bool))  -- A table of size n*n where each cell is either black (some false), white (some true), or not colored (none).

-- Define the coloring rules for Amin and Ali:
def can_amins_turn_make_all_red (n : ℕ) (t : table n) : Prop :=
sorry -- (Implement the condition based on the problem statement.)

def can_ali_turn_make_all_red (n : ℕ) (t : table n) : Prop :=
sorry -- (Implement the condition based on the problem statement.)

-- The theorem to be proved
theorem amin_ali_equivalence {n : ℕ} (t : table n) : 
  can_amins_turn_make_all_red n t ↔ can_ali_turn_make_all_red n t :=
sorry

end amin_ali_equivalence_l117_117837


namespace find_digit_l117_117426

def nat_of_base (b : ℕ) (d : ℕ) (base_number : List ℕ) : ℕ :=
base_number.reverse.zipWith (λ digit, λ idx, d * b ^ idx) base_number

theorem find_digit (triangle : ℕ) : (35 + triangle = 9 * triangle + 3) -> triangle = 4 :=
by
  sorry

end find_digit_l117_117426


namespace opposite_of_neg_2023_l117_117037

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117037


namespace quadrilateral_area_ratio_l117_117768

noncomputable def area_of_octagon (a : ℝ) : ℝ := 2 * a^2 * (1 + Real.sqrt 2)

noncomputable def area_of_square (s : ℝ) : ℝ := s^2

theorem quadrilateral_area_ratio (a : ℝ) (s : ℝ)
    (h1 : s = a * Real.sqrt (2 + Real.sqrt 2))
    : (area_of_square s) / (area_of_octagon a) = Real.sqrt 2 / 2 :=
by
  sorry

end quadrilateral_area_ratio_l117_117768


namespace max_chord_length_of_parabola_l117_117673

-- Definitions based on the problem conditions
def parabola (x y : ℝ) : Prop := x^2 = 8 * y
def y_midpoint_condition (y1 y2 : ℝ) : Prop := (y1 + y2) / 2 = 4

-- The theorem to prove that the maximum length of the chord AB is 12
theorem max_chord_length_of_parabola (x1 y1 x2 y2 : ℝ) 
  (h1 : parabola x1 y1) 
  (h2 : parabola x2 y2) 
  (h_mid : y_midpoint_condition y1 y2) : 
  abs ((y1 + y2) + 2 * 2) = 12 :=
sorry

end max_chord_length_of_parabola_l117_117673


namespace opposite_of_neg_2023_l117_117012

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117012


namespace committee_count_l117_117900

noncomputable def num_acceptable_committees (total_people : ℕ) (committee_size : ℕ) (conditions : List (Set ℕ)) : ℕ := sorry

theorem committee_count :
  num_acceptable_committees 9 5 [ {1, 2}, {3, 4} ] = 41 := sorry

end committee_count_l117_117900


namespace general_term_sum_terms_l117_117654

-- Sequence a_n
def S (n : ℕ) : ℤ := 4 * n - n^2
def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = 5 - 2 * n :=
by sorry

-- Sequence b_n and sum T_n
def b (n : ℕ) : ℤ := (7 - a n) / 2 ^ n

def T (n : ℕ) : ℤ :=
  (1 : ℤ)  * ∑ i in finset.range n, ((i + 1) : ℤ) / 2 ^ i

theorem sum_terms (n : ℕ) (hn : n ≥ 1) : T n = 6 - ((n + 3) : ℤ) / 2 ^ (n - 1) :=
by sorry

end general_term_sum_terms_l117_117654


namespace geometric_series_sum_l117_117204

theorem geometric_series_sum :
  let a := (1 / 2 : ℝ)
  let r := (1 / 2 : ℝ)
  let n := 6
  (a * (1 - r^n) / (1 - r)) = (63 / 64 : ℝ) := 
by 
  sorry

end geometric_series_sum_l117_117204


namespace james_hears_beats_per_week_l117_117366

theorem james_hears_beats_per_week
  (beats_per_minute : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (H1 : beats_per_minute = 200)
  (H2 : hours_per_day = 2)
  (H3 : days_per_week = 7) :
  beats_per_minute * hours_per_day * 60 * days_per_week = 168000 := 
by
  -- sorry proof step placeholder
  sorry

end james_hears_beats_per_week_l117_117366


namespace parabola_vertex_l117_117804

theorem parabola_vertex :
  (∃ h k, ∀ x, (x^2 - 2 = ((x - h) ^ 2) + k) ∧ (h = 0) ∧ (k = -2)) :=
by
  sorry

end parabola_vertex_l117_117804


namespace normal_distribution_condition_l117_117296

open ProbabilityTheory

noncomputable def xis_normal : MeasureSpace.ProbMeasure ℝ :=
  MeasureSpace.ProbabilityMeasure.stdNormal

variable {ξ : ℝ → Prop}

theorem normal_distribution_condition (a : ℝ)
  (h0 : ∀ x, ξ x ↔ (MeasureSpace.ProbabilityMeasure.density xis_normal ((MeasureSpace.ProbabilityMeasure.stdNormal : MeasureSpace.ProbMeasure ℝ) > 1) x).mass = a) :
  MeasureSpace.ProbabilityMeasure.density xis_normal ((-1 ≤ ξ) ∧ (ξ ≤ 0)) = 1/2 - a :=
by
  sorry

end normal_distribution_condition_l117_117296


namespace number_of_emus_l117_117972

theorem number_of_emus (total_heads_and_legs : ℕ) (heads_per_emu legs_per_emu : ℕ) (total_emu : ℕ) :
  total_heads_and_legs = 60 → 
  heads_per_emu = 1 → 
  legs_per_emu = 2 → 
  total_emu = total_heads_and_legs / (heads_per_emu + legs_per_emu) → 
  total_emu = 20 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  refine h4
  sorry

end number_of_emus_l117_117972


namespace abs_a_minus_b_l117_117985

-- Define τ function as the number of divisors
def tau (n : ℕ) : ℕ := (List.filter (λ d, n % d = 0) (List.range (n + 1))).length

-- Define S function
def S (n : ℕ) : ℕ := (List.range (n + 1)).sum (λ k, tau (k + 1))

-- Define a and b as count of n ≤ 3000 with S(n) odd or even respectively
def a : ℕ := (List.range 3001).count (λ n, S n % 2 = 1)
def b : ℕ := (List.range 3001).count (λ n, S n % 2 = 0)

-- Prove the main statement |a - b| = 54
theorem abs_a_minus_b : |a - b| = 54 := by
  sorry

end abs_a_minus_b_l117_117985


namespace golu_distance_after_turning_left_l117_117685

theorem golu_distance_after_turning_left :
  ∀ (a c b : ℝ), a = 8 → c = 10 → (c ^ 2 = a ^ 2 + b ^ 2) → b = 6 :=
by
  intros a c b ha hc hpyth
  rw [ha, hc] at hpyth
  sorry

end golu_distance_after_turning_left_l117_117685


namespace meaningful_domain_l117_117826

def is_meaningful (x : ℝ) : Prop :=
  (x - 1) ≠ 0

theorem meaningful_domain (x : ℝ) : is_meaningful x ↔ (x ≠ 1) :=
  sorry

end meaningful_domain_l117_117826


namespace valid_n_count_is_399_l117_117606

def count_valid_n : ℕ :=
  (count (λ n, (1 ≤ n ∧ n ≤ 1000) ∧ (
    ∃ (x : ℝ), n = ⌊x⌋ + ⌊2*x⌋ + ⌊3*x⌋ + ⌊4*x⌋)))

theorem valid_n_count_is_399 : count_valid_n = 399 :=
by
  sorry

end valid_n_count_is_399_l117_117606


namespace solve_for_t_l117_117622

variable (f : ℝ → ℝ)
variable (x t : ℝ)

-- Conditions
def cond1 : Prop := ∀ x, f ((1 / 2) * x - 1) = 2 * x + 3
def cond2 : Prop := f t = 4

-- Theorem statement
theorem solve_for_t (h1 : cond1 f) (h2 : cond2 f t) : t = -3 / 4 := by
  sorry

end solve_for_t_l117_117622


namespace opposite_of_neg_2023_l117_117068

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117068


namespace isosceles_right_triangle_area_l117_117849

noncomputable def triangle_area (a b c : ℝ) (angle : ℝ) : ℝ :=
  if angle = 90 then 0.5 * a * b
  else sorry -- General case for arbitrary angle, to be filled in as needed.

theorem isosceles_right_triangle_area :
  ∀ (PQ PR : ℝ), (PQ = 8) → (PR = 8) → (triangle_area PQ PR 0 90) = 32 :=
by
  intros PQ PR hPQ hPR
  simp [triangle_area]
  rw [hPQ, hPR]
  norm_num
  sorry

end isosceles_right_triangle_area_l117_117849


namespace mary_needs_more_apples_l117_117399

theorem mary_needs_more_apples :
  ∀ (number_of_pies apples_per_pie apples_harvested : ℕ),
    number_of_pies = 10 →
    apples_per_pie = 8 →
    apples_harvested = 50 →
    let total_apples_needed := number_of_pies * apples_per_pie in
    let apples_to_buy := total_apples_needed - apples_harvested in
    apples_to_buy = 30 :=
by
  intros number_of_pies apples_per_pie apples_harvested h_pies h_apples_per_pie h_apples_harvested
  rw [h_pies, h_apples_per_pie, h_apples_harvested]
  let total_apples_needed := number_of_pies * apples_per_pie
  let apples_to_buy := total_apples_needed - apples_harvested
  Sorry -- Here, we would put the proof if necessary.

end mary_needs_more_apples_l117_117399


namespace mean_proportional_234_104_l117_117134

theorem mean_proportional_234_104 : Real.sqrt (234 * 104) = 156 :=
by 
  sorry

end mean_proportional_234_104_l117_117134


namespace parabola_vertex_l117_117803

theorem parabola_vertex :
  (∃ h k, ∀ x, (x^2 - 2 = ((x - h) ^ 2) + k) ∧ (h = 0) ∧ (k = -2)) :=
by
  sorry

end parabola_vertex_l117_117803


namespace angle_inequality_l117_117354

variable (A B C D : Type)
variable [IsoscelesTrapezoid A B C D]

def lengths (AD BC height : ℝ) : Prop :=
  AD = 12 ∧ BC = 6 ∧ height = 4

theorem angle_inequality (h : lengths AD BC 4)
  (h_trap : IsoscelesTrapezoid A B C D) :
  angle A B C > angle B C A := sorry

end angle_inequality_l117_117354


namespace fold_triangle_proof_l117_117519

noncomputable theory

open_locale real

def triangle := {a b c : ℝ // a^2 + b^2 = c^2}
def midpoint (a b : ℝ) := (a + b) / 2

def length_of_crease (a b c : ℝ) (ht : triangle ∧ a = 5 ∧ b = 12 ∧ c = 13) : ℝ := 
  let m := midpoint b c in
    real.sqrt (pow m - 0)^2 + (pow b - m)^2
    sorry

theorem fold_triangle_proof : ∀ a b c : ℝ, 
  triangle a b c ∧ a = 5 ∧ b = 12 ∧ c = 13 → 
  length_of_crease a b c = 7.387 := 
by 
  sorry

end fold_triangle_proof_l117_117519


namespace opposite_of_negative_2023_l117_117054

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117054


namespace problem1_problem2_l117_117683

noncomputable theory

open Real

-- Given conditions
def a (α : ℝ) : vector ℝ := (cos α, sin α)
def b : vector ℝ := (-2, 2)

-- Problem 1: Prove (sin α + cos α)^2 = 1/25 given a · b = 14/5
theorem problem1 (α : ℝ) (h : (-2 : ℝ) * cos α + 2 * sin α = 14/5) :
  (sin α + cos α)^2 = 1/25 :=
begin
  sorry,
end

-- Problem 2: Prove sin (π - α) · sin (π/2 + α) = -1/2 given a is parallel to b
theorem problem2 (α : ℝ) (h : cos α = sin α) :
  sin (π - α) * sin (π/2 + α) = -1/2 :=
begin
  sorry,
end

end problem1_problem2_l117_117683


namespace john_subtracts_79_l117_117102

theorem john_subtracts_79 (x : ℕ) (h : x = 40) : (x - 1)^2 = x^2 - 79 :=
by sorry

end john_subtracts_79_l117_117102


namespace length_of_train_l117_117186

theorem length_of_train 
    (speed_kmph : ℤ) (time_sec : ℤ) (length_platform_m : ℝ) 
    (h1 : speed_kmph = 54)
    (h2 : time_sec = 25)
    (h3 : length_platform_m = 175.03) : 
    let speed_m_s := (speed_kmph * 1000) / 3600,
        distance_covered_m := speed_m_s * time_sec,
        length_train_m := distance_covered_m - length_platform_m in
    length_train_m = 199.97 :=
by
  sorry

end length_of_train_l117_117186


namespace james_hears_beats_per_week_l117_117367

theorem james_hears_beats_per_week
  (beats_per_minute : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (H1 : beats_per_minute = 200)
  (H2 : hours_per_day = 2)
  (H3 : days_per_week = 7) :
  beats_per_minute * hours_per_day * 60 * days_per_week = 168000 := 
by
  -- sorry proof step placeholder
  sorry

end james_hears_beats_per_week_l117_117367


namespace verify_eq_l117_117230

theorem verify_eq (x : ℝ) (h_ne_1 : x ≠ -1) (h_ne_3 : x ≠ -3) : 
  (x = -5) 
  → (∃ x, (x^3 + 3 * x^2 - x) / (x^2 + 4 * x + 3) + x = -7) :=
by
  intros h
  rw h
  have h_x_ne_1 : (-5 : ℝ) ≠ -1 := by norm_num
  have h_x_ne_3 : (-5 : ℝ) ≠ -3 := by norm_num
  existsi (-5 : ℝ)
  sorry

end verify_eq_l117_117230


namespace heartsuit_computation_l117_117571

def heartsuit (a b : ℝ) : ℝ :=
  (a + b) / (a - b)

theorem heartsuit_computation :
  heartsuit (heartsuit 7 (heartsuit 2 5)) 4 = -9/7 := by
  sorry

end heartsuit_computation_l117_117571


namespace opposite_of_neg_2023_l117_117067

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117067


namespace sum_divisible_by_11_l117_117966

theorem sum_divisible_by_11 (n : ℕ) : (6^(2*n) + 3^n + 3^(n+2)) % 11 = 0 := by
  sorry

end sum_divisible_by_11_l117_117966


namespace math_problem_l117_117684

-- Define the given vectors
def vector_m (α : ℝ) : Vector ℝ := ⟨2, Real.sin α⟩
def vector_n (α : ℝ) : Vector ℝ := ⟨Real.cos α, -1⟩

-- Define the orthogonality condition
def orthogonal (v1 v2 : Vector ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Define the intervals for alpha and beta
axiom alpha_interval (α : ℝ) : 0 < α ∧ α < (Real.pi / 2)
axiom beta_interval (β : ℝ) : 0 < β ∧ β < (Real.pi / 2)

-- Define the given sine condition
axiom sin_alpha_minus_beta (α β : ℝ) : Real.sin (α - β) = Real.sqrt 10 / 10

-- The statement to be proven
theorem math_problem (α β : ℝ)
  (h₁ : orthogonal (vector_m α) (vector_n α))
  (h₂ : alpha_interval α)
  (h₃ : sin_alpha_minus_beta α β)
  (h₄ : beta_interval β)
  : Real.sin (2 * α) = 4 / 5 ∧ Real.cos (2 * α) = -(3 / 5) ∧ β = Real.pi / 4 := by
  sorry

end math_problem_l117_117684


namespace solve_triangle_problem_l117_117720

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) (S : ℝ) : Prop :=
  (sqrt 3 * c = 2 * a * sin C) ∧
  (A = π / 3 ∨ A = 2 * π / 3) ∧
  (A = π / 3) ∧
  (S = 2 * sqrt 3) ∧
  ((b = 4 ∧ c = 2) ∨ (b = 2 ∧ c = 4))

theorem solve_triangle_problem : 
  ∃ (a b c : ℝ) (A B C : ℝ) (S : ℝ),
  triangle_problem a b c A B C S :=
by
  sorry

end solve_triangle_problem_l117_117720


namespace quadratic_inequality_condition_l117_117909

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) → 0 ≤ a ∧ a < 4 :=
sorry

end quadratic_inequality_condition_l117_117909


namespace alex_walking_distance_correct_l117_117542

def average_speed_on_flat_ground := 20
def time_on_flat_ground := 4.5
def average_speed_uphill := 12
def time_uphill := 2.5
def average_speed_downhill := 24
def time_downhill := 1.5
def total_distance_to_next_town := 164

def distance_covered_flat := average_speed_on_flat_ground * time_on_flat_ground
def distance_covered_uphill := average_speed_uphill * time_uphill
def distance_covered_downhill := average_speed_downhill * time_downhill

def total_distance_covered_before_puncture := distance_covered_flat + distance_covered_uphill + distance_covered_downhill

def distance_to_walk := total_distance_to_next_town - total_distance_covered_before_puncture

theorem alex_walking_distance_correct : distance_to_walk = 8 := by
  unfold distance_to_walk
  unfold total_distance_to_next_town
  unfold total_distance_covered_before_puncture
  unfold distance_covered_flat distance_covered_uphill distance_covered_downhill
  unfold average_speed_on_flat_ground time_on_flat_ground
  unfold average_speed_uphill time_uphill
  unfold average_speed_downhill time_downhill
  norm_num
  sorry

end alex_walking_distance_correct_l117_117542


namespace light_color_94_is_green_l117_117182

def color_sequence := ['red', 'red', 'green', 'green', 'green', 'yellow', 'yellow']

def color_of_nth_light (n : Nat) : String :=
  color_sequence[(n % color_sequence.length) - 1]

theorem light_color_94_is_green : color_of_nth_light 94 = 'green' :=
sorry

end light_color_94_is_green_l117_117182


namespace trapezoid_isosceles_AD_squared_eq_BC_times_AD_plus_AB_l117_117736

theorem trapezoid_isosceles_AD_squared_eq_BC_times_AD_plus_AB
  (ABCD : Type)
  [h_isosceles_trapezoid : is_isosceles_trapezoid ABCD]
  (B C D A : ABCD)
  (h_base_BC : is_base B C)
  (h_base_AD : is_base A D)
  (angle_BDC : ∠ B D C = 10)
  (angle_BDA : ∠ B D A = 70) :
  (AD ^ 2) = BC * (AD + AB) :=
sorry

end trapezoid_isosceles_AD_squared_eq_BC_times_AD_plus_AB_l117_117736


namespace ellipse_properties_correct_l117_117273

-- Define the semi-major axis
def semi_major_axis (a : ℝ) : Prop := a = 2

-- Define the semi-minor axis
def minor_axis_length (b : ℝ) : Prop := 2 * b = 2 * Real.sqrt 3

-- Define the foci-related properties
def foci_properties (b : ℝ) (AF2 BF2 : ℝ) : Prop :=
  (|AF2| + |BF2| = 5) ∧ (|AF2| = |BF2|)

-- Validate the line intersecting points
def intersection_properties (a b : ℝ) (AB : ℝ) : Prop :=
  (|AF2| + |BF2| + |AB| = 8) ∧ (|AB| = 3)

-- Define the eccentricity
def eccentricity (a c : ℝ) : Prop :=
  (c = Real.sqrt (a^2 - b^2)) ∧ (c / a ≠ Real.sqrt 3 / 3)

-- Define the final theorem to combine all properties
theorem ellipse_properties_correct (b AF2 BF2 AB : ℝ) :
  semi_major_axis 2 ∧ minor_axis_length b ∧ foci_properties b AF2 BF2 ∧ intersection_properties 2 b AB ∧ eccentricity 2 (Real.sqrt (4 - b^2)) :=
  sorry

end ellipse_properties_correct_l117_117273


namespace institutions_made_happy_l117_117839

theorem institutions_made_happy (ppl_per_inst happy_ppl : ℕ) (h1 : ppl_per_inst = 80) (h2 : happy_ppl = 480) :
  happy_ppl / ppl_per_inst = 6 :=
by
  rw [h1, h2]
  norm_num

end institutions_made_happy_l117_117839


namespace minute_hand_120_degrees_l117_117445

/-- The minute hand starts at 12 and rotates 120 degrees clockwise.
    Prove that it points at the number 4 on the clock. -/
theorem minute_hand_120_degrees :
  let total_numbers := 12 in
  let total_degrees := 360 in
  let start_position := 12 in
  let rotation_degrees := 120 in
  let angle_per_position := total_degrees / total_numbers in
  let positions_moved := rotation_degrees / angle_per_position in
  (start_position + positions_moved) % total_numbers = 4 :=
by {
  sorry
}

end minute_hand_120_degrees_l117_117445


namespace axis_rotation_regular_polyhedron_l117_117765

noncomputable def axisOfRotation 
  (P : Type) [RegularPolyhedron P] 
  (axis : Line) : Prop :=
  ∃ (center : Point) (point_on_surface : Point),
    is_center P center ∧
    (is_vertex P point_on_surface ∨ is_midpoint_edge P point_on_surface ∨ is_center_face P point_on_surface) ∧ 
    passes_through axis center ∧
    passes_through axis point_on_surface

-- Conditions Definitions
axiom RegularPolyhedron (P : Type) : Type
axiom Point : Type
axiom Line : Type
axiom is_center (P : Type) : Point → Prop
axiom is_vertex (P : Type) : Point → Prop
axiom is_midpoint_edge (P : Type) : Point → Prop
axiom is_center_face (P : Type) : Point → Prop
axiom passes_through : Line → Point → Prop

-- Proof Statement
theorem axis_rotation_regular_polyhedron 
  (P : Type) [hP : RegularPolyhedron P] 
  (axis : Line) :
  axisOfRotation P axis :=
sorry

end axis_rotation_regular_polyhedron_l117_117765


namespace value_of_a_if_perpendicular_l117_117340

theorem value_of_a_if_perpendicular (a l : ℝ) :
  (∀ x y : ℝ, (a + l) * x + 2 * y = 0 → x - a * y = 1 → false) → a = 1 :=
by
  -- Proof is omitted
  sorry

end value_of_a_if_perpendicular_l117_117340


namespace opposite_of_neg_2023_l117_117018

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117018


namespace min_value_f_range_a_f_l117_117304

-- Statement for part 1
theorem min_value_f (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x + a| + |x - 3|) :
  (∀ x, f x ≥ |a + 3|) ∧ (∃ x, f x = |a + 3|) → (a = 2 ∨ a = -8) :=
by
  sorry

-- Statement for part 2
theorem range_a_f (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x + a| + |x - 3|) :
  (∀ x, -1 ≤ x → x ≤ 0 → f x ≤ |x - 4|) → (a ∈ Icc (-1 : ℝ) 1) :=
by
  sorry

end min_value_f_range_a_f_l117_117304


namespace length_of_median_AC_l117_117364

-- Define the parameters of the triangle
variables {A B C : Type} 
-- Define the lengths of the sides
variables (BC AB AC : ℝ)
-- Define the parameter b in the quadratic equation
variable (b : ℝ)

-- Conditions from the problem
def is_right_triangle (BC AB AC : ℝ) : Prop :=
  BC^2 + AB^2 = AC^2

def equal_roots_quadratic (b : ℝ) : Prop :=
  let discriminant := 16 - 4 * b in
  discriminant = 0

def median_on_side_AC (BC AB AC : ℝ) : ℝ :=
  if is_right_triangle BC AB AC then AC / 2 else 0

-- Prove the length of the median given the conditions
theorem length_of_median_AC :
  ∀ (BC AB : ℝ) (b : ℝ),
    BC = 2 → AB = 2 * Real.sqrt 3 → equal_roots_quadratic b → b = 4 →
    median_on_side_AC BC AB b = 2 :=
by
  intros BC AB b hBC hAB hEqRoots hb
  rw [hBC, hAB, hb]
  unfold equal_roots_quadratic at hEqRoots
  unfold median_on_side_AC
  suffices hTri: is_right_triangle 2 (2 * Real.sqrt 3) 4, from calc
    4 / 2 = 2 : by norm_num,
  unfold is_right_triangle
  norm_num

#check length_of_median_AC

end length_of_median_AC_l117_117364


namespace presidency_vp_meeting_ways_l117_117151

theorem presidency_vp_meeting_ways : 
  let schools := 4,
      members_per_school := 4,
      host_reps := 3,
      non_host_reps := 1 in
  (schools * (nat.choose members_per_school 2 * nat.choose (members_per_school - 2) 1 * (members_per_school ^ (schools - 1)))) = 3072 := 
by
  let schools := 4
  let members_per_school := 4
  let host_reps := 3
  let non_host_reps := 1
  have host_choices := 6 * 2 -- Choose president and vice-president, then one more representative from the host school
  have non_host_choices := 4 * 4 * 4 -- Choose one representative from each of the other three schools
  have total_choices := 4 * (host_choices * non_host_choices) -- Combining all choices and considering any school can be the host
  show total_choices = 3072 from sorry

end presidency_vp_meeting_ways_l117_117151


namespace ana_correct_percentage_l117_117547

-- Definitions for conditions
noncomputable def score1 := 0.75 * 20
noncomputable def score2 := 0.85 * 50
noncomputable def score3 := 0.8 * 30
noncomputable def total_score := score1 + score2 + score3
noncomputable def total_problems := 20 + 50 + 30
noncomputable def overall_percentage := (total_score / total_problems) * 100

-- The main theorem to prove
theorem ana_correct_percentage : overall_percentage = 81.5 := by
  sorry

end ana_correct_percentage_l117_117547


namespace complex_conjugate_sum_l117_117691

theorem complex_conjugate_sum (z : ℂ) (i : ℂ) (h : i * (z - 1) = 1) : conj z + z = 2 := 
by sorry

end complex_conjugate_sum_l117_117691


namespace find_n_l117_117642

-- Define the condition as a function relation
def sequence (n : ℕ) : ℕ := sorry 

-- Given statement for proof
theorem find_n (n : ℕ) (h : sequence n * sequence n = 7 * (sequence (n - 4) * sequence (n - 4))) : n = 7 := 
sorry

end find_n_l117_117642


namespace prism_faces_l117_117522

-- Define the conditions of the problem
def prism (E : ℕ) : Prop :=
  ∃ (L : ℕ), 3 * L = E

-- Define the main proof statement
theorem prism_faces (E : ℕ) (hE : prism E) : E = 27 → 2 + E / 3 = 11 :=
by
  sorry -- Proof is not required

end prism_faces_l117_117522


namespace multiplication_identity_l117_117331

theorem multiplication_identity : 5 ^ 29 * 4 ^ 15 = 2 * 10 ^ 29 := by
  sorry

end multiplication_identity_l117_117331


namespace probability_of_pink_gumball_l117_117722

theorem probability_of_pink_gumball (P_B P_P : ℝ)
    (h1 : P_B ^ 2 = 25 / 49)
    (h2 : P_B + P_P = 1) :
    P_P = 2 / 7 := 
    sorry

end probability_of_pink_gumball_l117_117722


namespace find_subsets_P_Q_l117_117137

variable {X : Type} [Fintype X]
variable (f : (Set X) → ℝ)
variable (E D : Set X) (hE : ∃ D, EvenCard D ∧ f D > 1990)
variable (h_f : ∀ (A B : Set X), EvenCard A ∧ EvenCard B ∧ Disjoint A B → f (A ∪ B) = f A + f B - 1990)

noncomputable def exists_subsets_P_Q : Prop :=
  ∃ P Q : Set X, 
    P ∩ Q = ∅ ∧ P ∪ Q = Set.univ ∧ 
    (∀ S : Set X, EvenCard S ∧ S ⊆ P ∧ S ≠ ∅ → f S > 1990) ∧
    (∀ T : Set X, EvenCard T ∧ T ⊆ Q → f T ≤ 1990)

theorem find_subsets_P_Q (hE : ∃ D, EvenCard D ∧ f D > 1990)
                         (h_f : ∀ (A B : Set X), EvenCard A ∧ EvenCard B ∧ Disjoint A B → f (A ∪ B) = f A + f B - 1990) :
  exists_subsets_P_Q f :=
sorry

end find_subsets_P_Q_l117_117137


namespace coordinates_of_C_l117_117717

noncomputable def point := (ℚ × ℚ)

def A : point := (2, 8)
def B : point := (6, 14)
def M : point := (4, 11)
def L : point := (6, 6)
def C : point := (14, 2)

-- midpoint formula definition
def is_midpoint (M A B : point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Main statement to prove
theorem coordinates_of_C (hM : is_midpoint M A B) : C = (14, 2) :=
  sorry

end coordinates_of_C_l117_117717


namespace opposite_of_neg_2023_l117_117027

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117027


namespace shooter_A_better_stable_l117_117115

variables {A B : Type} [has_shots A] [has_shots B] 
  {avgA avgB varA varB : ℝ}

/-- Shooter A has better performance and more stability than shooter B, if the average score
of A is greater than the average score of B and the variance of A's scores is less than
the variance of B's scores. -/
theorem shooter_A_better_stable 
  (h_avg: avgA > avgB)
  (h_var: varA < varB) :
  better_performance_and_stability A B :=
sorry

class has_shots (α : Type) :=
  (shots : ℕ)

class better_performance_and_stability (A B : Type) :=
  (better_stability : varA < varB)
  (better_performance : avgA > avgB)

end shooter_A_better_stable_l117_117115


namespace work_done_by_force_l117_117169

variable (x : ℝ)
def F (x : ℝ) : ℝ := 1 + Real.exp x

-- Statement that needs to be proved:
theorem work_done_by_force :
  (∫ x in 0..1, F x) = Real.exp 1 :=
by
  sorry

end work_done_by_force_l117_117169


namespace marcy_total_people_served_l117_117749

noncomputable def total_people_served_lip_gloss
  (tubs_lip_gloss : ℕ) (tubes_per_tub_lip_gloss : ℕ) (people_per_tube_lip_gloss : ℕ) : ℕ :=
  tubs_lip_gloss * tubes_per_tub_lip_gloss * people_per_tube_lip_gloss

noncomputable def total_people_served_mascara
  (tubs_mascara : ℕ) (tubes_per_tub_mascara : ℕ) (people_per_tube_mascara : ℕ) : ℕ :=
  tubs_mascara * tubes_per_tub_mascara * people_per_tube_mascara

theorem marcy_total_people_served :
  ∀ (tubs_lip_gloss tubs_mascara : ℕ) 
    (tubes_per_tub_lip_gloss tubes_per_tub_mascara 
     people_per_tube_lip_gloss people_per_tube_mascara : ℕ),
    tubs_lip_gloss = 6 → 
    tubes_per_tub_lip_gloss = 2 → 
    people_per_tube_lip_gloss = 3 → 
    tubs_mascara = 4 → 
    tubes_per_tub_mascara = 3 → 
    people_per_tube_mascara = 5 → 
    total_people_served_lip_gloss tubs_lip_gloss 
                                 tubes_per_tub_lip_gloss 
                                 people_per_tube_lip_gloss = 36 :=
by
  intros tubs_lip_gloss tubs_mascara 
         tubes_per_tub_lip_gloss tubes_per_tub_mascara 
         people_per_tube_lip_gloss people_per_tube_mascara
         h_tubs_lip_gloss h_tubes_per_tub_lip_gloss h_people_per_tube_lip_gloss
         h_tubs_mascara h_tubes_per_tub_mascara h_people_per_tube_mascara
  rw [h_tubs_lip_gloss, h_tubes_per_tub_lip_gloss, h_people_per_tube_lip_gloss]
  exact rfl


end marcy_total_people_served_l117_117749


namespace problem1_problem2_problem3_l117_117559

-- Problem 1
theorem problem1 : (-10) - (-22) + (-8) - 13 = -9 :=
by sorry

-- Problem 2
theorem problem2 : (5 / 9 + 5 / 6 - 3 / 4) / (-1 / 36) = -23 :=
by sorry

-- Problem 3
theorem problem3 : -1^4 - |0.5 - 1| * 2 - ( -3 )^2 / ( -3 / 2 ) = 4 :=
by sorry

end problem1_problem2_problem3_l117_117559


namespace min_value_of_S_l117_117984

-- Prove the minimum value of S given the conditions

theorem min_value_of_S (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_prod : a * b * c = 1):
  let S := (1 / (2 * a + 1)) + (1 / (2 * b + 1)) + (1 / (2 * c + 1))
  in S = 1 :=
sorry

end min_value_of_S_l117_117984


namespace solve_for_N_l117_117087

theorem solve_for_N (N : ℤ) (h : 2 * N^2 + N = 12) (h_neg : N < 0) : N = -3 := 
by 
  sorry

end solve_for_N_l117_117087


namespace exists_rectangle_with_diagonal_zeros_and_ones_l117_117356

-- Define the problem parameters
def n := 2012
def table := Matrix (Fin n) (Fin n) (Fin 2)

-- Conditions
def row_contains_zero_and_one (m : table) (r : Fin n) : Prop :=
  ∃ c1 c2 : Fin n, m r c1 = 0 ∧ m r c2 = 1

def col_contains_zero_and_one (m : table) (c : Fin n) : Prop :=
  ∃ r1 r2 : Fin n, m r1 c = 0 ∧ m r2 c = 1

-- Problem statement
theorem exists_rectangle_with_diagonal_zeros_and_ones
  (m : table)
  (h_rows : ∀ r : Fin n, row_contains_zero_and_one m r)
  (h_cols : ∀ c : Fin n, col_contains_zero_and_one m c) :
  ∃ (r1 r2 : Fin n) (c1 c2 : Fin n),
    m r1 c1 = 0 ∧ m r2 c2 = 0 ∧ m r1 c2 = 1 ∧ m r2 c1 = 1 :=
sorry

end exists_rectangle_with_diagonal_zeros_and_ones_l117_117356


namespace opposite_of_neg_2023_l117_117065

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117065


namespace opposite_of_neg_2023_l117_117023

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117023


namespace total_weight_lifted_l117_117924

-- Given definitions from the conditions
def weight_left_hand : ℕ := 10
def weight_right_hand : ℕ := 10

-- The proof problem statement
theorem total_weight_lifted : weight_left_hand + weight_right_hand = 20 := 
by 
  -- Proof goes here
  sorry

end total_weight_lifted_l117_117924


namespace opposite_of_neg2023_l117_117004

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l117_117004


namespace mary_needs_more_apples_l117_117400

theorem mary_needs_more_apples :
  ∀ (number_of_pies apples_per_pie apples_harvested : ℕ),
    number_of_pies = 10 →
    apples_per_pie = 8 →
    apples_harvested = 50 →
    let total_apples_needed := number_of_pies * apples_per_pie in
    let apples_to_buy := total_apples_needed - apples_harvested in
    apples_to_buy = 30 :=
by
  intros number_of_pies apples_per_pie apples_harvested h_pies h_apples_per_pie h_apples_harvested
  rw [h_pies, h_apples_per_pie, h_apples_harvested]
  let total_apples_needed := number_of_pies * apples_per_pie
  let apples_to_buy := total_apples_needed - apples_harvested
  Sorry -- Here, we would put the proof if necessary.

end mary_needs_more_apples_l117_117400


namespace min_distance_l117_117316

def vector (α : Type*) := (α × α)

variables (x y : ℝ)
def a := (x, y)
def b := (1, 2)

theorem min_distance (h : x^2 + y^2 = 1) : 
  real.norm (a x y - b) = real.sqrt 5 - 1 :=
sorry

end min_distance_l117_117316


namespace monotonic_increasing_interval_of_log_l117_117821

theorem monotonic_increasing_interval_of_log (f : ℝ → ℝ) 
  (h : ∀ x, f x = log (x^2 - 4 * x + 3)) :
  ∀ x, x > 3 → monotone (λ x, f x) :=
by
  sorry

end monotonic_increasing_interval_of_log_l117_117821


namespace minimum_value_correct_l117_117625

noncomputable def minimum_value (x y : ℝ) (h_posx : 0 < x) (h_posy : 0 < y) (h_eq : 2 * x + 3 * y = 1) : ℝ :=
  min_value where
    min_value : ℝ := 5 + 2 * Real.sqrt 6

theorem minimum_value_correct : ∀ (x y : ℝ), 0 < x → 0 < y → 2 * x + 3 * y = 1 → minimum_value x y = 5 + 2 * Real.sqrt 6 :=
by
  -- proof omitted
  sorry

end minimum_value_correct_l117_117625


namespace richard_cleans_in_45_minutes_l117_117424
noncomputable def richard_time (R : ℝ) := 
  let cory_time := R + 3
  let blake_time := (R + 3) - 4
  (R + cory_time + blake_time = 136) -> R = 45

theorem richard_cleans_in_45_minutes : 
  ∃ R : ℝ, richard_time R := 
sorry

end richard_cleans_in_45_minutes_l117_117424


namespace difference_in_pages_l117_117583

def purple_pages_per_book : ℕ := 230
def orange_pages_per_book : ℕ := 510
def purple_books_read : ℕ := 5
def orange_books_read : ℕ := 4

theorem difference_in_pages : 
  orange_books_read * orange_pages_per_book - purple_books_read * purple_pages_per_book = 890 :=
by
  sorry

end difference_in_pages_l117_117583


namespace gcd_proof_l117_117245

def gcd_10010_15015 := Nat.gcd 10010 15015 = 5005

theorem gcd_proof : gcd_10010_15015 :=
by
  sorry

end gcd_proof_l117_117245


namespace mike_bicycle_distance_l117_117757

def speed : ℝ := 30
def time : ℝ := (1 / 3)

theorem mike_bicycle_distance :
  speed * time = 10 :=
by sorry

end mike_bicycle_distance_l117_117757


namespace next_term_geometric_sequence_l117_117119

theorem next_term_geometric_sequence (x : ℝ) :
  let a_1 := 4
      r := 5 * x
      a_4 := 500 * x^3
  in a_4 * r = 2500 * x^4 :=
by
  let a_4 := 500 * x^3
  let r := 5 * x
  have h : a_4 * r = 2500 * x^4 := sorry
  exact h

end next_term_geometric_sequence_l117_117119


namespace opposite_of_negative_2023_l117_117045

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117045


namespace smallest_year_with_digit_sum_16_l117_117232

def sum_of_digits (n : Nat) : Nat :=
  let digits : List Nat := n.digits 10
  digits.foldl (· + ·) 0

theorem smallest_year_with_digit_sum_16 :
  ∃ (y : Nat), 2010 < y ∧ sum_of_digits y = 16 ∧
  (∀ (z : Nat), 2010 < z ∧ sum_of_digits z = 16 → z ≥ y) → y = 2059 :=
by
  sorry

end smallest_year_with_digit_sum_16_l117_117232


namespace gcd_10010_15015_l117_117242

theorem gcd_10010_15015 :
  let n1 := 10010
  let n2 := 15015
  ∃ d, d = Nat.gcd n1 n2 ∧ d = 5005 :=
by
  let n1 := 10010
  let n2 := 15015
  -- ... omitted proof steps
  sorry

end gcd_10010_15015_l117_117242


namespace problem_statement_l117_117610

variables {R : Type*} [LinearOrderedField R]

-- Definitions of f and its derivatives
variable (f : R → R)
variable (f' : R → R) 
variable (f'' : R → R)

-- Conditions given in the math problem
axiom decreasing_f : ∀ x1 x2 : R, x1 < x2 → f x1 > f x2
axiom derivative_condition : ∀ x : R, f'' x ≠ 0 → f x / f'' x < 1 - x

-- Lean 4 statement for the proof problem
theorem problem_statement (decreasing_f : ∀ x1 x2 : R, x1 < x2 → f x1 > f x2)
    (derivative_condition : ∀ x : R, f'' x ≠ 0 → f x / f'' x < 1 - x) :
    ∀ x : R, f x > 0 :=
by
  sorry

end problem_statement_l117_117610


namespace percentage_of_original_solution_l117_117911

-- Define the problem and conditions
variable (P : ℝ)
variable (h1 : (0.5 * P + 0.5 * 60) = 55)

-- The theorem to prove
theorem percentage_of_original_solution : P = 50 :=
by
  -- Proof will go here
  sorry

end percentage_of_original_solution_l117_117911


namespace eastern_rattlesnake_segments_l117_117968

theorem eastern_rattlesnake_segments (W : ℕ) (H₁ : W = 8) (P : ℝ) (H₂ : P = 0.25) : 
  let E := W - P * W in E = 6 :=
by
  sorry

end eastern_rattlesnake_segments_l117_117968


namespace opposite_of_negative_2023_l117_117052

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117052


namespace triangle_ABC_properties_l117_117501

variables {A B C M : Type} [EuclideanGeometry A B C M]
  (angle_BAC : ∠BAC = 45) (angle_ACB : ∠ACB = 30)
  (is_midpoint_M : Midpoint M B C)

theorem triangle_ABC_properties :
  ∠AMB = 45 ∧ BC * AC = 2 * AM * AB :=
by
  sorry

end triangle_ABC_properties_l117_117501


namespace gcd_10010_15015_l117_117240

theorem gcd_10010_15015 :
  Int.gcd 10010 15015 = 5005 :=
by 
  sorry

end gcd_10010_15015_l117_117240


namespace solution_set_of_inequality_l117_117086

theorem solution_set_of_inequality :
  { x : ℝ | |x + 1| + |x - 4| ≥ 7 } = { x : ℝ | x ≤ -2 ∨ x ≥ 5 } := sorry

end solution_set_of_inequality_l117_117086


namespace combination_sum_eq_13_l117_117941

theorem combination_sum_eq_13 :
  ∑ n in {n | binom 24 n + binom 24 12 = binom 25 14}.to_finset, n = 13 :=
by {
  -- placeholder for the proof
  sorry
}

end combination_sum_eq_13_l117_117941


namespace measure_of_angle_EHG_l117_117357

-- Define degrees to be the units for angles
abbreviation degrees := ℝ

-- Define the angles involved in the problem
variables (EFG FGH EHG : degrees)

-- Establish conditions given in the problem
variables (parallelogram_EFGH : parallelogram)
variable (angle_relation : EFG = 4 * FGH)

-- Lean statement that proves the measure of angle EHG
theorem measure_of_angle_EHG (parallelogram_EFGH : Prop) (angle_relation : EFG = 4 * FGH) : EHG = 144 :=
  by
    -- Conditions provided already, skipping proof
    sorry

end measure_of_angle_EHG_l117_117357


namespace find_abc_sum_l117_117377

theorem find_abc_sum :
  ∃ (a b c : ℕ), 
  (∀ x, 
    if x > 0 then f x = a * x + b + 3 
    else if x = 0 then f x = a + b 
    else f x = 2 * b * x + c) ∧ 
  f 2 = 7 ∧ 
  f 0 = 6 ∧ 
  f (-1) = -4 ∧ 
  a + b + c = 10 :=
by
  sorry

end find_abc_sum_l117_117377


namespace initial_mean_corrected_l117_117820

theorem initial_mean_corrected
  (M : ℝ)
  (h : 30 * M + 10 = 30 * 140.33333333333334) :
  M = 140 :=
by
  sorry

end initial_mean_corrected_l117_117820


namespace value_of_each_bill_l117_117475

theorem value_of_each_bill (bank1_withdrawal bank2_withdrawal number_of_bills : ℕ)
  (h1 : bank1_withdrawal = 300) 
  (h2 : bank2_withdrawal = 300) 
  (h3 : number_of_bills = 30) : 
  (bank1_withdrawal + bank2_withdrawal) / number_of_bills = 20 :=
by
  sorry

end value_of_each_bill_l117_117475


namespace perimeter_of_polygon_l117_117448

-- Conditions
variables (a b : ℝ) (polygon_is_part_of_rectangle : 0 < a ∧ 0 < b)

-- Prove that if the polygon completes a rectangle with perimeter 28,
-- then the perimeter of the polygon is 28.
theorem perimeter_of_polygon (h : 2 * (a + b) = 28) : 2 * (a + b) = 28 :=
by
  exact h

end perimeter_of_polygon_l117_117448


namespace exists_subset_X_l117_117958

open Nat

theorem exists_subset_X (N : Set ℕ) (hN : ∀ n, n ∈ N) :
  (∃ X ⊆ N, ∀ n ∈ N, ∃! a b ∈ X, n = a - b) :=
sorry

end exists_subset_X_l117_117958


namespace solve_equation_l117_117781

theorem solve_equation 
  (x : ℚ)
  (h : (x^2 + 3*x + 4)/(x + 5) = x + 6) :
  x = -13/4 := 
by
  sorry

end solve_equation_l117_117781


namespace problem1_problem2_problem3_l117_117558

open Real

theorem problem1 : (-10) - (-22) + (-8) - 13 = -9 :=
by sorry

theorem problem2 : ((5/9) + (5/6) - (3/4)) / (-1/36) = -23 :=
by sorry

theorem problem3 : -(1:ℝ)^4 - abs(0.5 - 1) * 2 - ((-3)^2 / (-3/2)) = 4 :=
by sorry

end problem1_problem2_problem3_l117_117558


namespace generalization_condition_l117_117258

noncomputable def x_i_base_repr (x_i : ℕ) (k : ℕ) : ℕ → ℕ
| j := (x_i / (k+1)^j) % (k+1)

def sum_mod_k1 (xs : List ℕ) (k : ℕ) (j : ℕ) : ℕ :=
(xs.map (λ x_i, x_i_base_repr x_i k j)).sum % (k+1)

def combined_int (xs : List ℕ) (k : ℕ) : ℕ :=
List.sum (List.zipWith (λ y j, y * (k+1)^j) 
  (List.range (xs.map (λ x_i, Nat.log (x_i + 1))).maximum).pmap (sum_mod_k1 xs k) id) sorry

def belongs_to_P (xs : List ℕ) (k : ℕ) : Prop :=
combined_int xs k = 0

theorem generalization_condition (xs : List ℕ) (k : ℕ) :
  belongs_to_P xs k ↔ combined_int xs k = 0 :=
by sorry

end generalization_condition_l117_117258


namespace plums_for_20_oranges_l117_117693

noncomputable def oranges_to_pears (oranges : ℕ) : ℕ :=
  (oranges / 5) * 3

noncomputable def pears_to_plums (pears : ℕ) : ℕ :=
  (pears / 4) * 6

theorem plums_for_20_oranges :
  oranges_to_pears 20 = 12 ∧ pears_to_plums 12 = 18 :=
by
  sorry

end plums_for_20_oranges_l117_117693


namespace range_of_a_l117_117656

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) ↔ (a ≤ -1 ∧ a ≠ -2) :=
by
  sorry

end range_of_a_l117_117656


namespace Maria_high_school_students_l117_117750

theorem Maria_high_school_students (M J : ℕ) (h1 : M = 4 * J) (h2 : M + J = 3600) : M = 2880 :=
sorry

end Maria_high_school_students_l117_117750


namespace coefficient_a8_value_l117_117991

theorem coefficient_a8_value :
  let f := (1 : ℝ) + x 
  let rhs := ∑ k in Finset.range 11, (a k) * (1 - x)^k
  f^10 = rhs →
  a 8 = 180 :=
by
  sorry

end coefficient_a8_value_l117_117991


namespace cost_price_of_articles_l117_117534

theorem cost_price_of_articles (C1 C2 C3 : ℝ) :
  let S1, S2, S3 := 1110, 1575, 2040 in
  S1 = C1 + 0.20 * C1 → 
  S2 = C2 - 0.15 * C2 → 
  S3 = C3 →
  C1 = 925 ∧ C2 ≈ 1852.94 ∧ C3 = 2040 :=
sorry

end cost_price_of_articles_l117_117534


namespace parabola_vertex_l117_117805

theorem parabola_vertex :
  (∃ h k, ∀ x, (x^2 - 2 = ((x - h) ^ 2) + k) ∧ (h = 0) ∧ (k = -2)) :=
by
  sorry

end parabola_vertex_l117_117805


namespace find_m_interval_l117_117955

-- Define the sequence recursively
def sequence_recursive (x : ℕ → ℝ) (n : ℕ) : Prop :=
  x 0 = 5 ∧ ∀ n, x (n + 1) = (x n ^ 2 + 5 * x n + 4) / (x n + 6)

-- The left-hand side of the inequality
noncomputable def target_value : ℝ := 4 + 1 / (2 ^ 20)

-- The condition that the sequence element must satisfy
def condition (x : ℕ → ℝ) (m : ℕ) : Prop :=
  x m ≤ target_value

-- The proof problem statement, m lies within the given interval
theorem find_m_interval (x : ℕ → ℝ) (m : ℕ) :
  sequence_recursive x n →
  condition x m →
  81 ≤ m ∧ m ≤ 242 :=
sorry

end find_m_interval_l117_117955


namespace right_triangle_special_count_l117_117322

def count_special_right_triangles : Nat :=
  {n | ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c = Int.sqrt (a^2 + b^2) ∧ ab = 8 * (a + b + c) ∧ a < b ∧ c ∉ {a, b}}.card

theorem right_triangle_special_count : count_special_right_triangles = 4 := by
  sorry

end right_triangle_special_count_l117_117322


namespace bounded_size_of_good_set_l117_117611

open Finset

def is_good (A : Finset ℕ) (n : ℕ) : Prop :=
  (∀ u v ∈ A, (u + v < 1) ∨ (u + v > n) ∧ (u - v < 1) ∨ (u - v > n))

theorem bounded_size_of_good_set :
  ∃ c : ℝ, (c = 6 / 5) ∧ ∀ (n : ℕ) (A : Finset ℕ),
  (0 < n) →
  (A ⊆ (Finset.range (2 * n + 1)).filter (≥ 1)) →
  is_good A n →
  (A.card ≤ c * n) :=
sorry

end bounded_size_of_good_set_l117_117611


namespace a8_val_l117_117629

noncomputable def sum_seq (a : ℕ → ℕ) : ℕ → ℕ
| 0     := 0
| (n+1) := (sum_seq n) + a (n+1)

def a : ℕ → ℕ
| 0     := 0  -- not defined for 0, placeholder value
| 1     := 1
| (n+1) := 3 * sum_seq a n

theorem a8_val : a 8 = 12288 :=
by
  sorry

end a8_val_l117_117629


namespace julie_age_end_of_period_is_15_l117_117376

-- Define necessary constants and variables
def hours_per_day : ℝ := 3
def pay_rate_per_hour_per_year : ℝ := 0.75
def total_days_worked : ℝ := 60
def total_earnings : ℝ := 810

-- Define Julie's age at the end of the four-month period
def julies_age_end_of_period (age: ℝ) : Prop :=
  hours_per_day * pay_rate_per_hour_per_year * age * total_days_worked = total_earnings

-- The final Lean 4 statement that needs proof
theorem julie_age_end_of_period_is_15 : ∃ age : ℝ, julies_age_end_of_period age ∧ age = 15 :=
by {
  sorry
}

end julie_age_end_of_period_is_15_l117_117376


namespace unique_solution_2021_l117_117231

theorem unique_solution_2021 (x y z : ℝ) :
  (sqrt (x - 997) + sqrt (y - 932) + sqrt (z - 796) = 100) ∧
  (sqrt (x - 1237) + sqrt (y - 1121) + sqrt (3045 - z) = 90) ∧
  (sqrt (x - 1621) + sqrt (2805 - y) + sqrt (z - 997) = 80) ∧
  (sqrt (2102 - x) + sqrt (y - 1237) + sqrt (z - 932) = 70) →
  (x = 2021 ∧ y = 2021 ∧ z = 2021) :=
by
  sorry

end unique_solution_2021_l117_117231


namespace find_number_l117_117609

theorem find_number : 
  (15^2 * 9^2) / x = 51.193820224719104 → x = 356 :=
by
  sorry

end find_number_l117_117609


namespace evaluate_expression_l117_117587

theorem evaluate_expression :
  ((Int.ceil ((21 : ℚ) / 5 - Int.ceil ((35 : ℚ) / 23))) : ℚ) /
  (Int.ceil ((35 : ℚ) / 5 + Int.ceil ((5 * 23 : ℚ) / 35))) = 3 / 11 := by
  sorry

end evaluate_expression_l117_117587


namespace max_min_values_on_circle_l117_117280

def on_circle (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 - 4 * x - 4 * y + 7 = 0

theorem max_min_values_on_circle (x y : ℝ) (h : on_circle x y) :
  16 ≤ (x + 1) ^ 2 + (y + 2) ^ 2 ∧ (x + 1) ^ 2 + (y + 2) ^ 2 ≤ 36 :=
  sorry

end max_min_values_on_circle_l117_117280


namespace difference_in_pages_l117_117582

def purple_pages_per_book : ℕ := 230
def orange_pages_per_book : ℕ := 510
def purple_books_read : ℕ := 5
def orange_books_read : ℕ := 4

theorem difference_in_pages : 
  orange_books_read * orange_pages_per_book - purple_books_read * purple_pages_per_book = 890 :=
by
  sorry

end difference_in_pages_l117_117582


namespace smallest_possible_value_of_a_l117_117788

theorem smallest_possible_value_of_a :
  ∃ (a : ℝ), 0 < a ∧ ∀ (b c : ℝ), a = 9 ∧ a + b + c ∈ ℤ ∧ ∃ d : ℝ,
  d = (∃ x y : ℝ, x = 3 / 4 ∧ y = -25 / 16 ∧ -y = a * (x - 3 / 4) ^ 2 + (b * x) + c) :=
begin
  sorry
end

end smallest_possible_value_of_a_l117_117788


namespace number_of_girls_in_circle_l117_117986

theorem number_of_girls_in_circle :
  ∀ (total_children children_holding_boy_hand children_holding_girl_hand : ℕ),
  total_children = 40 →
  children_holding_boy_hand = 22 →
  children_holding_girl_hand = 30 →
  let children_holding_both := children_holding_boy_hand + children_holding_girl_hand - total_children in
  let children_holding_only_girl := children_holding_girl_hand - children_holding_both in
  let total_girl_hands := (children_holding_only_girl * 2) + children_holding_both in
  (total_girl_hands / 2) = 24 :=
by
  intros total_children children_holding_boy_hand children_holding_girl_hand total_def boy_def girl_def
  let children_holding_both := children_holding_boy_hand + children_holding_girl_hand - total_children
  let children_holding_only_girl := children_holding_girl_hand - children_holding_both
  let total_girl_hands := (children_holding_only_girl * 2) + children_holding_both
  have total_girls : (total_girl_hands / 2) = 24 := sorry
  exact total_girls

end number_of_girls_in_circle_l117_117986


namespace jenny_relationship_l117_117723

noncomputable def relationship (x y r w d : ℝ) : Prop :=
  x * r - y * w = 60 * d

theorem jenny_relationship (x y r w d: ℝ) (hx : x > 0) (hy : y > 0) (hr : r > 0) (hw : w > 0) (hd : d > 0) :
  relationship x y r w d :=
begin
  sorry
end

end jenny_relationship_l117_117723


namespace opposite_of_neg_2023_l117_117006

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117006


namespace sample_size_l117_117576

theorem sample_size (f1 f2 f3 : ℝ) (h1 : f1 = 10) (h2 : f2 = 0.35) (h3 : f3 = 0.45) :
  10 / (1 - f2 - f3) = 50 :=
by
  rw [h2, h3]
  norm_num
  sorry

end sample_size_l117_117576


namespace remaining_macaroons_weight_l117_117728

-- Problem conditions
variables (macaroons_per_bake : ℕ) (weight_per_macaroon : ℕ) (bags : ℕ) (macaroons_eaten : ℕ)

-- Definitions from problem conditions
def macaroons_per_bake := 12
def weight_per_macaroon := 5
def bags := 4
def macaroons_per_bag := macaroons_per_bake / bags
def macaroons_eaten := macaroons_per_bag

-- Lean theorem
theorem remaining_macaroons_weight : (macaroons_per_bake - macaroons_eaten) * weight_per_macaroon = 45 :=
by
  have h1 : macaroons_per_bag = 12 / 4 := rfl
  have h2 : macaroons_per_bag = 3 := by norm_num [h1]
  have h3 : macaroons_eaten = 3 := h2
  have h4 : macaroons_per_bake - macaroons_eaten = 12 - 3 := rfl
  have h5 : macaroons_per_bake - macaroons_eaten = 9 := by norm_num [h4]
  have h6 : (macaroons_per_bake - macaroons_eaten) * weight_per_macaroon = 9 * 5 := by rw [h5]
  calc
    (macaroons_per_bake - macaroons_eaten) * weight_per_macaroon = 9 * 5 : by rw [h6]
    ... = 45 : by norm_num

end remaining_macaroons_weight_l117_117728


namespace correct_factorization_l117_117122

theorem correct_factorization (a b : ℝ) : a^2 - 4 * a * b + 4 * b^2 = (a - 2 * b)^2 :=
by sorry

end correct_factorization_l117_117122


namespace simple_interest_rate_l117_117880

theorem simple_interest_rate (SI P T : ℕ) (H : SI = 160 ∧ P = 800 ∧ T = 4) : 
  (R : ℕ) (R = 5) :=
by
  sorry

end simple_interest_rate_l117_117880


namespace initial_average_weight_l117_117798

theorem initial_average_weight
  (A : ℝ)
  (h : 30 * 27.4 - 10 = 29 * A) : 
  A = 28 := 
by
  sorry

end initial_average_weight_l117_117798


namespace Gideon_age_l117_117457
-- Import necessary libraries

-- Define the initial conditions and the final proof statement
theorem Gideon_age :
  ∀ (total_marbles gideon_age_now : ℕ) (frac : ℚ),
    total_marbles = 100 →
    frac = 3 / 4 →
    let marbles_given_away := (frac * total_marbles).to_nat in
    let remaining_marbles := total_marbles - marbles_given_away in
    let age_five_years_from_now := 2 * remaining_marbles in
    age_five_years_from_now = gideon_age_now + 5 →
    gideon_age_now = 45 :=
by
  intros total_marbles gideon_age_now frac H1 H2 H3
  sorry

end Gideon_age_l117_117457


namespace sqrt_arcsin_sum_eq_pi_div_two_l117_117997

noncomputable def sqrt_arcsin_sum (x : ℝ) : ℝ :=
  Real.sqrt(x * (x + 1)) + Real.arcsin (Real.sqrt(x^2 + x + 1))

theorem sqrt_arcsin_sum_eq_pi_div_two (x : ℝ) :
  sqrt_arcsin_sum x = π / 2 :=
by
  sorry

end sqrt_arcsin_sum_eq_pi_div_two_l117_117997


namespace find_BP_l117_117891

-- Define points
variables {A B C D P : Type}  

-- Define lengths
variables (AP PC BP DP BD : ℝ)

-- Provided conditions
axiom h1 : AP = 10
axiom h2 : PC = 2
axiom h3 : BD = 9

-- Assume intersect and lengths relations setup
axiom intersect : BP < DP
axiom power_of_point : AP * PC = BP * DP

-- Target statement
theorem find_BP (h1 : AP = 10) (h2 : PC = 2) (h3 : BD = 9)
  (intersect : BP < DP) (power_of_point : AP * PC = BP * DP) : BP = 4 :=
  sorry

end find_BP_l117_117891


namespace minimizes_sum_distances_l117_117277

variable (P : ℝ) (Ps : Fin 9 → ℝ)

-- Define the sum of distances function
def sum_distances (Ps : Fin 9 → ℝ) (P : ℝ) : ℝ :=
  ∑ i, (abs (P - Ps i))

-- Define the median point P5 as Ps 4 (since Lean is zero-indexed)
def P5 : ℝ := Ps 4

-- The theorem that proves P5 minimizes the sum of distances
theorem minimizes_sum_distances : ∀ P : ℝ, sum_distances Ps P ≥ sum_distances Ps P5
:= sorry

end minimizes_sum_distances_l117_117277


namespace arc_length_semicubical_parabola_correct_l117_117554

noncomputable def arc_length_semicubical_parabola : ℝ :=
∫ x in 0..9, sqrt(1 + (3 / 2 * x ^ (1 / 2)) ^ 2)

theorem arc_length_semicubical_parabola_correct :
  arc_length_semicubical_parabola = 28.552 :=
sorry

end arc_length_semicubical_parabola_correct_l117_117554


namespace sum_of_disk_areas_l117_117776

-- Definitions corresponding to the problem conditions
def radius_of_large_circle : ℝ := 2
def number_of_disks : ℕ := 16

-- The lean statement for the problem
theorem sum_of_disk_areas (r : ℝ)
  (h1 : ∀ i j : Fin number_of_disks, i ≠ j → ¬∃ x : Fin number_of_disks, (x = i ∧ x = j))
  (h2 : ∀ i : Fin number_of_disks, ∃! p : ℝ × ℝ, (p.1^2 + p.2^2 = radius_of_large_circle^2))
  (h3 : ∀ i, ∃! p : ℝ × ℝ, (p.1^2 + p.2^2 = r^2)) :
  (16 * (Real.pi * r^2) = Real.pi * (112 - 64 * Real.sqrt 3))
:= by
  sorry

end sum_of_disk_areas_l117_117776


namespace find_fourth_vertex_of_square_l117_117987

def complex_square_vertex (z₁ z₂ z₃: ℂ) (z₄: ℂ) : Prop :=
  let midpoint (a b : ℂ) := (a + b) / 2
  ∃ (w₁ w₂: ℂ), -- opposite vertices of the square
    midpoint z₁ z₃ = midpoint w₁ w₂ ∧
    w₁ = z₂ ∧
    w₂ = z₄

theorem find_fourth_vertex_of_square :
  complex_square_vertex (2 + 3*I) (-1 + 2*I) (-2 - 3*I) (1 - 2*I) :=
begin
  -- Here the proof would be added to confirm the fourth vertex
  -- For now, we leave it as sorry because the problem only requires the statement
  sorry
end

end find_fourth_vertex_of_square_l117_117987


namespace num_bricks_required_l117_117161

def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def brick_length_cm : ℝ := 20
def brick_width_cm : ℝ := 10
def cm_to_m (cm : ℝ) : ℝ := cm / 100
def area_of_courtyard : ℝ := courtyard_length * courtyard_width
def area_of_one_brick : ℝ := cm_to_m(brick_length_cm) * cm_to_m(brick_width_cm)
def num_of_bricks (courtyard_area brick_area : ℝ) : ℝ := courtyard_area / brick_area

theorem num_bricks_required :
  num_of_bricks area_of_courtyard area_of_one_brick = 20000 := by
sorry

end num_bricks_required_l117_117161


namespace problem1_problem2_problem3_l117_117557

open Real

theorem problem1 : (-10) - (-22) + (-8) - 13 = -9 :=
by sorry

theorem problem2 : ((5/9) + (5/6) - (3/4)) / (-1/36) = -23 :=
by sorry

theorem problem3 : -(1:ℝ)^4 - abs(0.5 - 1) * 2 - ((-3)^2 / (-3/2)) = 4 :=
by sorry

end problem1_problem2_problem3_l117_117557


namespace max_levels_passed_probability_passed_three_levels_l117_117894

-- Definition of the conditions
def sum_greater (n : ℕ) (throw : ℕ → ℕ) := (Finset.range n).sum throw > 2^n

def die_faces := {1, 2, 3, 4, 5, 6}

-- Problem 1: Maximum number of levels that can be passed
theorem max_levels_passed : 
  (∀ (n : ℕ), sum_greater n (λ i, 6) → n ≤ 4) :=
sorry

-- Problem 2: Probability of passing the first three levels consecutively
theorem probability_passed_three_levels :
  ∑' (n : ℕ), ite (n = 0 ∨ n = 3 ∨ n > 3) 0 
                   (P_No_Allowed n (χ = 5)) = 100 / 243 :=
sorry

end max_levels_passed_probability_passed_three_levels_l117_117894


namespace sum_infinite_series_l117_117553

theorem sum_infinite_series :
  (∑' n : ℕ, n ≥ 1 → (3^n) / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = 1 / 4 := by
    sorry

end sum_infinite_series_l117_117553


namespace fold_points_area_EQ_144π_minus_144sqrt3_l117_117628

noncomputable def triangleDEF := { 
  D: (ℝ × ℝ),
  E: (ℝ × ℝ),
  F: (ℝ × ℝ) // ensuring coordinates form a right triangle with DE = 24, DF = 48, angle E = 90°
}

theorem fold_points_area_EQ_144π_minus_144sqrt3 :
  (∀ Q : (ℝ × ℝ), fold_point Q triangleDEF) →
  (area_of_fold_points Q) = 144 * π - 144 * sqrt 3 :=
by 
  intro Q fold_point,
  specialize fold_point Q,
  have h1: fold_point Q triangleDEF,
  from fold_point_def,
  sorry

end fold_points_area_EQ_144π_minus_144sqrt3_l117_117628


namespace more_orange_pages_read_l117_117585

-- Define the conditions
def purple_pages_per_book : Nat := 230
def orange_pages_per_book : Nat := 510
def purple_books_read : Nat := 5
def orange_books_read : Nat := 4

-- Calculate the total pages read from purple and orange books respectively
def total_purple_pages_read : Nat := purple_pages_per_book * purple_books_read
def total_orange_pages_read : Nat := orange_pages_per_book * orange_books_read

-- State the theorem to be proved
theorem more_orange_pages_read : total_orange_pages_read - total_purple_pages_read = 890 :=
by
  -- This is where the proof steps would go, but we'll leave it as sorry to indicate the proof is not provided
  sorry

end more_orange_pages_read_l117_117585


namespace gcd_10010_15015_l117_117243

theorem gcd_10010_15015 :
  let n1 := 10010
  let n2 := 15015
  ∃ d, d = Nat.gcd n1 n2 ∧ d = 5005 :=
by
  let n1 := 10010
  let n2 := 15015
  -- ... omitted proof steps
  sorry

end gcd_10010_15015_l117_117243


namespace at_most_one_side_shorter_than_altitude_l117_117413

theorem at_most_one_side_shorter_than_altitude (A B C : Point) :
  ∀ (AB BC CA m_a m_b m_c : ℝ), 
  is_triangle A B C → 
  altitude A B C m_a → 
  altitude B C A m_b → 
  altitude C A B m_c → 
  (AB < m_c → ¬ (BC < m_a ∧ CA < m_b)) :=
sorry

end at_most_one_side_shorter_than_altitude_l117_117413


namespace opposite_of_neg_2023_l117_117015

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117015


namespace compute_fraction_l117_117946

theorem compute_fraction : 
  (2045^2 - 2030^2) / (2050^2 - 2025^2) = 3 / 5 :=
by
  sorry

end compute_fraction_l117_117946


namespace optimal_play_probability_Reimu_l117_117769

noncomputable def probability_Reimu_wins : ℚ :=
  5 / 16

theorem optimal_play_probability_Reimu :
  probability_Reimu_wins = 5 / 16 := 
by
  sorry

end optimal_play_probability_Reimu_l117_117769


namespace relationship_among_abc_l117_117620

noncomputable def a : ℝ := Real.logb 1.2 0.8
noncomputable def b : ℝ := Real.logb 0.7 0.8
noncomputable def c : ℝ := 1.2 ^ 0.8

theorem relationship_among_abc : a < b ∧ b < c := by
  sorry

end relationship_among_abc_l117_117620


namespace interest_rate_per_annum_l117_117980

theorem interest_rate_per_annum (P A : ℝ) (T : ℝ)
  (principal_eq : P = 973.913043478261)
  (amount_eq : A = 1120)
  (time_eq : T = 3):
  (A - P) / (T * P) * 100 = 5 := 
by 
  sorry

end interest_rate_per_annum_l117_117980


namespace find_QR_l117_117342

variable {P Q R Y : Type}
variables [metric_space P] [metric_space Q] [metric_space R] [metric_space Y]

-- Define the lengths
def length_PQ : ℝ := 95
def length_PR : ℝ := 103

-- Define the condition that PQ and PR are fixed lengths
axiom PQ_fixed : dist P Q = length_PQ
axiom PR_fixed : dist P R = length_PR

-- Define the condition that a circle centered at P with radius PQ intersects at Q and Y on QR
axiom circle_intersect : dist P Q = dist P Y

-- Define the distances between intersecting points QY and RY having integer lengths
variables (QY RY : ℕ)
axiom integer_lengths : (QY:ℝ) * (QY + RY:ℝ) = 1584

-- Define the required proof that QR = 132 given the above conditions
theorem find_QR : dist Q R = 132 :=
sorry

end find_QR_l117_117342


namespace marble_color_197th_l117_117919

theorem marble_color_197th (n : ℕ) (total_marbles : ℕ) (marble_color : ℕ → ℕ)
                          (h_total : total_marbles = 240)
                          (h_pattern : ∀ k, marble_color (k + 15) = marble_color k)
                          (h_colors : ∀ i, (0 ≤ i ∧ i < 15) →
                                   (marble_color i = if i < 6 then 1
                                   else if i < 11 then 2
                                   else if i < 15 then 3
                                   else 0)) :
  marble_color 197 = 1 := sorry

end marble_color_197th_l117_117919


namespace product_zero_l117_117223

theorem product_zero (x : ℤ) (h : x = 5) : (x - 15) * (x - 14) * (x - 13) * (x - 12) * (x - 11) * (x - 10) * (x - 9) * (x - 8) * (x - 7) * (x - 6) * (x - 5) * (x - 4) * (x - 3) * (x - 2) * (x - 1) * x = 0 :=
by 
  rw h
  sorry

end product_zero_l117_117223


namespace opposite_of_neg_2023_l117_117010

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117010


namespace money_spent_on_paintbrushes_l117_117843

-- Define the conditions
def total_spent : ℝ := 90.00
def cost_canvases : ℝ := 40.00
def cost_paints : ℝ := cost_canvases / 2
def cost_easel : ℝ := 15.00

-- Define the problem
theorem money_spent_on_paintbrushes : total_spent - (cost_canvases + cost_paints + cost_easel) = 15.00 :=
by sorry

end money_spent_on_paintbrushes_l117_117843


namespace first_player_wins_l117_117838

theorem first_player_wins :
  ∃ (a : ℤ), ∀ (p : polynomial ℤ), 
    (p = polynomial.monomial 3 1 + polynomial.monomial 2 (-a) + polynomial.monomial 1 (-1) + polynomial.monomial 0 a →
    ∃ r1 r2 r3 : ℤ, p = (polynomial.monomial 2 1 - polynomial.monomial 0 1) * (polynomial.monomial 1 1 - polynomial.monomial 0 a)) :=
begin
  sorry
end

end first_player_wins_l117_117838


namespace gideon_age_l117_117458

noncomputable def gideon_current_age : ℕ :=
  let total_marbles := 100
  let remaining_marbles := total_marbles * (1/4 : ℚ)
  let age_in_five_years := remaining_marbles * 2
  let curr_age := age_in_five_years - 5
  curr_age

theorem gideon_age : gideon_current_age = 45 :=
by
  let total_marbles := 100
  let remaining_marbles := total_marbles * (1/4 : ℚ)
  let age_in_five_years := remaining_marbles * 2
  let curr_age := age_in_five_years - 5
  have h1 : total_marbles = 100 := rfl
  have h2 : remaining_marbles = 25 := by norm_num
  have h3 : age_in_five_years = 50 := by norm_num
  have h4 : curr_age = 45 := by norm_num
  exact h4

end gideon_age_l117_117458


namespace smallest_n_for_candy_distribution_l117_117579

theorem smallest_n_for_candy_distribution : ∃ (n : ℕ), (∀ (a : ℕ), ∃ (x : ℕ), (x * (x + 1)) / 2 % n = a % n) ∧ n = 2 :=
sorry

end smallest_n_for_candy_distribution_l117_117579


namespace circle_radius_tangent_to_ellipse_l117_117113

theorem circle_radius_tangent_to_ellipse :
  ∃ r : ℝ, (∀ x y : ℝ, (x - r)^2 + y^2 = r^2 ∧ (x + r)^2 + y^2 = r^2 ∧ 4 * x^2 + 9 * y^2 = 18) →
    r = 3 * real.sqrt 5 / 5 :=
by
  -- Lean proof will be here, but we skip it as per instructions
  sorry

end circle_radius_tangent_to_ellipse_l117_117113


namespace living_room_is_big_as_other_rooms_l117_117751

def total_area (length width : ℕ) : ℕ := length * width

def remaining_area (total living_room : ℕ) : ℕ := total - living_room

def area_each_other_room (remaining num_rooms : ℕ) : ℕ := remaining / num_rooms

def number_of_rooms_the_living_room_is_as_big_as (living_room each_room : ℕ) : ℕ := living_room / each_room

example : total_area 16 10 = 160 := by
  rw [total_area]
  norm_num

example : remaining_area (total_area 16 10) 60 = 100 := by
  rw [total_area, remaining_area]
  norm_num

example : (area_each_other_room (remaining_area (total_area 16 10) 60) 5) = 20 := by
  rw [total_area, remaining_area, area_each_other_room]
  norm_num

example : number_of_rooms_the_living_room_is_as_big_as 60 20 = 3 := by
  rw [number_of_rooms_the_living_room_is_as_big_as]
  norm_num

theorem living_room_is_big_as_other_rooms (length width living_room total_rooms : ℕ) (h1 : length = 16) (h2 : width = 10) (h3 : living_room = 60) (h4 : total_rooms = 6) :
  total_area length width = 160 ∧
  remaining_area (total_area length width) living_room = 100 ∧
  area_each_other_room (remaining_area (total_area length width) living_room) (total_rooms - 1) = 20 ∧
  number_of_rooms_the_living_room_is_as_big_as living_room (area_each_other_room (remaining_area (total_area length width) living_room) (total_rooms - 1)) = 3 :=
by {
  rw [h1, h2, h3, h4],
  simp [total_area, remaining_area, area_each_other_room, number_of_rooms_the_living_room_is_as_big_as],
  norm_num,
}

#check living_room_is_big_as_other_rooms 

end living_room_is_big_as_other_rooms_l117_117751


namespace math_problem_l117_117959

theorem math_problem :
  let x1 := 0.60 * 50 in
  let x2 := 0.45 * 30 in
  let part1 := x1 * x2 in
  let y1 := 0.40 * 35 in
  let y2 := 0.25 * 20 in
  let part2 := y1 / y2 in
  let difference := part1 - part2 in
  let z1 := (3 / 5) * 100 in
  let z2 := (2 / 7) * 49 in
  let sum := z1 + z2 in
  let result := difference * sum in
  result = 29762.8 :=
by
  sorry

end math_problem_l117_117959


namespace max_Q_value_l117_117612

noncomputable def Q (b : ℝ) : ℝ :=
  let f := λ (x y : ℝ), (Real.sin (Real.pi * x))^2 + (Real.sin (Real.pi * y))^2
  ∑ in (0 ≤ x ∧ x ≤ b ∧ 0 ≤ y ∧ y ≤ min (2 * b) 1, 
    if f x y > 1 + b^2 then 
      1 
    else 
      0) / (b * min (2 * b) 1)

theorem max_Q_value : 
  let Q := λ (b : ℝ), ∫ (x : ℝ) in (0..b), ∫ (y : ℝ) in (0..min (2 * b) 1), 
  ite ((Real.sin (Real.pi * x))^2 + (Real.sin (Real.pi * y))^2 > 1 + b^2) 1 0 
  in ∀ (b : ℝ), 0 ≤ b ∧ b ≤ 1 → Q b ≤ 1 ∧ (Q 1 = 1) :=
by
  sorry

end max_Q_value_l117_117612


namespace rectangle_perimeter_l117_117793

theorem rectangle_perimeter (A : ℝ) (n : ℝ) (s : ℝ) (P : ℝ) :
  A = 512 → n = 8 → A / n = s^2 → s = 8 → P = 4 * 2 * s → P = 160 :=
by 
  intros hA hn hs hs8 hP
  rw [hA, hn, hs]
  simp [hs8, hP]
  linarith

end rectangle_perimeter_l117_117793


namespace find_a_l117_117995

def f (x : ℝ) : ℝ := 
  if 0 ≤ x then x^(1/2) 
  else |sin x|

theorem find_a (a : ℝ) (h : f a = 1/2) : a = 1/4 ∨ a = -π/6 :=
by
  sorry

end find_a_l117_117995


namespace smaller_angle_at_3_15_l117_117935

/-- Given a clock where the hour hand moves 30 degrees per hour
and the minute hand moves 6 degrees per minute,
prove that the smaller angle between the minute hand and the hour hand 
at 3:15 p.m. is 7.5 degrees. 
-/
theorem smaller_angle_at_3_15 : 
  let hour_angle_at_3_15 := 90 + (30 / 60) * 15,
      minute_angle_at_15 := (15 / 60) * 360 in
  |hour_angle_at_3_15 - minute_angle_at_15| = 7.5 := by
  sorry

end smaller_angle_at_3_15_l117_117935


namespace three_layers_coverage_l117_117840

/--
Three table runners have a combined area of 208 square inches. 
By overlapping the runners to cover 80% of a table of area 175 square inches, 
the area that is covered by exactly two layers of runner is 24 square inches. 
Prove that the area of the table that is covered with three layers of runner is 22 square inches.
--/
theorem three_layers_coverage :
  ∀ (A T two_layers total_table_coverage : ℝ),
  A = 208 ∧ total_table_coverage = 0.8 * 175 ∧ two_layers = 24 →
  A = (total_table_coverage - two_layers - T) + 2 * two_layers + 3 * T →
  T = 22 :=
by
  intros A T two_layers total_table_coverage h1 h2
  sorry

end three_layers_coverage_l117_117840


namespace unique_solution_a_l117_117600

-- Define the function f(x) as described in the problem
def f (x : ℝ) : ℝ :=
  ∑ k in Finset.range 100, |x - (k + 1)|

-- Theorem stating the value of a for which the equation has a unique real solution
theorem unique_solution_a :
  ∃! x : ℝ, f x = 2450 :=
sorry

end unique_solution_a_l117_117600


namespace shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder_l117_117982

theorem shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder
  (c : ℝ)
  (r : ℝ)
  (θ : ℝ)
  (hr : r ≥ 0)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  ∃ (x y z : ℝ), (z = c) ∧ 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ :=
by
  sorry

end shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder_l117_117982


namespace opposite_of_negative_2023_l117_117060

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117060


namespace arithmetic_seq_general_term_geometric_seq_general_term_sequence_sum_l117_117796

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℕ := 2^n

def seq_sum (n : ℕ) (seq : ℕ → ℕ) : ℕ :=
  (Finset.range n).sum seq

noncomputable def T_n (n : ℕ) : ℕ :=
  seq_sum n (λ i => (a_n (i + 1) + 1) * b_n (i + 1))

theorem arithmetic_seq_general_term (n : ℕ) : a_n n = 2 * n - 1 := by
  sorry

theorem geometric_seq_general_term (n : ℕ) : b_n n = 2^n := by
  sorry

theorem sequence_sum (n : ℕ) : T_n n = (n - 1) * 2^(n+2) + 4 := by
  sorry

end arithmetic_seq_general_term_geometric_seq_general_term_sequence_sum_l117_117796


namespace gcd_101_pow_11_plus_1_and_101_pow_11_plus_101_pow_3_plus_1_l117_117210

open Nat

theorem gcd_101_pow_11_plus_1_and_101_pow_11_plus_101_pow_3_plus_1 :
  gcd (101 ^ 11 + 1) (101 ^ 11 + 101 ^ 3 + 1) = 1 := 
by
  sorry

end gcd_101_pow_11_plus_1_and_101_pow_11_plus_101_pow_3_plus_1_l117_117210


namespace fraction_division_calculation_l117_117423

theorem fraction_division_calculation :
  (\( (1 : ℝ) / 24 \) / 
   ((1 / 12) - (5 / 16) + (7 / 24) - (2 / 3)) = - (2 / 29)) :=
by sorry

end fraction_division_calculation_l117_117423


namespace area_of_triangle_PQR_l117_117852

-- Definitions based on conditions
def is_isosceles_right_triangle (P Q R : Type) (angleP : ℝ) (lengthPR : ℝ) := 
  angleP = 90 ∧ lengthPR = 8

-- The proof goal
theorem area_of_triangle_PQR {P Q R : Type} (h : is_isosceles_right_triangle P Q R 90 8) : 
  ∃ (area : ℝ), area = 32 :=
begin
  sorry
end

end area_of_triangle_PQR_l117_117852


namespace opposite_of_negative_2023_l117_117057

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117057


namespace johns_total_biking_distance_is_9_25_l117_117375

-- Define constants and variables
variable (d : ℝ)
def biking_speed : ℝ := 15
def walking_distance : ℝ := 3
def walking_speed : ℝ := 4
def total_time : ℝ := (7 / 6) -- 1 hour and 10 minutes in hours

-- Define the times taken for biking and walking
def biking_time := (d - walking_distance) / biking_speed
def walking_time := walking_distance / walking_speed

-- Define the total time journey equation
def journey_equation := biking_time + walking_time = total_time

-- Prove that the total biking distance b is 9.25 kilometers
theorem johns_total_biking_distance_is_9_25 (h : journey_equation d) : d = 9.25 := by
  sorry

end johns_total_biking_distance_is_9_25_l117_117375


namespace path_area_correct_l117_117931

-- Define the constants for the problem
noncomputable def major_axis_lawn := 50.0
noncomputable def minor_axis_lawn := 35.0
noncomputable def semi_major_axis_lawn := major_axis_lawn / 2
noncomputable def semi_minor_axis_lawn := minor_axis_lawn / 2

-- Define the path widths
noncomputable def width_at_major_axis := 7.0
noncomputable def width_at_minor_axis := 3.0

-- Calculate the average width of the path
noncomputable def average_width := (width_at_major_axis + width_at_minor_axis) / 2

-- Calculate the semi-major and semi-minor axes of the larger ellipse
noncomputable def semi_major_axis_path := semi_major_axis_lawn + average_width
noncomputable def semi_minor_axis_path := semi_minor_axis_lawn + average_width

-- Define the areas of the ellipses
noncomputable def area_lawn := Real.pi * semi_major_axis_lawn * semi_minor_axis_lawn
noncomputable def area_path := Real.pi * semi_major_axis_path * semi_minor_axis_path

-- Define the area of the path as the difference of the areas
noncomputable def area_of_path := area_path - area_lawn

-- Finally, prove the original problem statement
theorem path_area_correct : 
  area_of_path = 237.5 * Real.pi := by
  sorry

end path_area_correct_l117_117931


namespace fewest_printers_l117_117152

theorem fewest_printers (x y : ℕ) (h1 : 350 * x = 200 * y) : x + y = 11 := 
by
  sorry

end fewest_printers_l117_117152


namespace length_of_third_side_l117_117671

-- Definitions for sides and perimeter condition
variables (a b : ℕ) (h1 : a = 3) (h2 : b = 10) (p : ℕ) (h3 : p % 6 = 0)
variable (c : ℕ)

-- Definition for the triangle inequality
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statement to prove the length of the third side
theorem length_of_third_side (h4 : triangle_inequality a b c)
  (h5 : p = a + b + c) : c = 11 :=
sorry

end length_of_third_side_l117_117671


namespace construct_circle_case_a_construct_circle_case_b_l117_117951

variables {P Q_1 Q_2 R_1 R_2 L_1 L_2 : Type}
variables [metric_space P] [metric_space Q_1] [metric_space Q_2] [metric_space R_1] [metric_space R_2] [metric_space L_1] [metric_space L_2]
variables (p : metric_space)

-- Define the conditions
def is_parallel (L1 L2 : Type) : Prop := ∀ (x : L1) (y : L2), x ∥ y
def is_perpendicular_distance_eq (p : Type) (L1 L2 : Type) : Prop := ∃ m, ∀ (x : p), dist x L1 = m ∧ dist x L2 = m
def center_exists (P : Type) (m : ℝ) (Q_1 Q_2: Type) (p: Type) : Prop := ∃ (c_1 c_2 : P), dist c_1 p = m ∧ dist c_2 p = m

-- Theorems for two cases

-- Case (a): Parallel lines
theorem construct_circle_case_a (P L_1 L_2 p : Type) (Q_1 Q_2 : Type)
  (h1 : is_parallel L_1 L_2) 
  (h2 : is_perpendicular_distance_eq p L_1 L_2) 
  (h3 : center_exists P m Q_1 Q_2 p) :
  true := sorry

-- Case (b): Non-parallel lines
theorem construct_circle_case_b (P L_1 L_2 : Type) (Q R_1 R_2 : Type)
  (h1 : ¬ is_parallel L_1 L_2)
  (h2 : center_exists Q m R_1 R_2 L_1 L_2) :
  true := sorry

end construct_circle_case_a_construct_circle_case_b_l117_117951


namespace inequality_sin_cos_400_40_l117_117123

theorem inequality_sin_cos_400_40 : (sin (400 * Real.pi / 180) < cos (40 * Real.pi / 180)) :=
by
  -- The proof requires verification which is skipped using sorry
  sorry

end inequality_sin_cos_400_40_l117_117123


namespace roots_of_unity_polynomial_l117_117214

theorem roots_of_unity_polynomial (c d : ℤ) (z : ℂ) (hz : z^3 = 1) :
  (z^3 + c * z + d = 0) → (z = 1) :=
sorry

end roots_of_unity_polynomial_l117_117214


namespace alex_had_to_walk_8_miles_l117_117540

theorem alex_had_to_walk_8_miles : 
  let flat_speed := 20 
  let flat_time := 4.5 
  let uphill_speed := 12 
  let uphill_time := 2.5 
  let downhill_speed := 24 
  let downhill_time := 1.5 
  let total_distance := 164 
  let flat_distance := flat_speed * flat_time 
  let uphill_distance := uphill_speed * uphill_time 
  let downhill_distance := downhill_speed * downhill_time 
  let biking_distance := flat_distance + uphill_distance + downhill_distance 
  let walking_distance := total_distance - biking_distance 
  walking_distance = 8 := 
by 
  let flat_speed := 20
  let flat_time := 4.5
  let uphill_speed := 12
  let uphill_time := 2.5
  let downhill_speed := 24
  let downhill_time := 1.5
  let total_distance := 164
  let flat_distance := flat_speed * flat_time
  let uphill_distance := uphill_speed * uphill_time
  let downhill_distance := downhill_speed * downhill_time
  let biking_distance := flat_distance + uphill_distance + downhill_distance
  let walking_distance := total_distance - biking_distance
  sorry

end alex_had_to_walk_8_miles_l117_117540


namespace part1_part2_part3_l117_117283

noncomputable def f (x a : ℝ) : ℝ := (x^2 - 4) * (x - a)
noncomputable def f' (x a : ℝ) : ℝ := 3 * x^2 - 2 * a * x - 4

theorem part1 (a : ℝ) : ∀ (x : ℝ), deriv (λ x, (x^2 - 4) * (x - a)) x = 3 * x^2 - 2 * a * x - 4 :=
by { sorry }

theorem part2 (a : ℝ) (h : f' (-1) a = 0) :
  (∀ x ∈ Icc (-2 : ℝ) 2, f x a ≤ f (-1) a) ∧ (∀ x ∈ Icc (-2 : ℝ) 2, f x a ≥ f (4/3) a) :=
by { sorry }

theorem part3 (a : ℝ) :
  (∀ x < -2, f' x a > 0) ∧ (∀ x > 2, f' x a > 0) → -2 ≤ a ∧ a ≤ 2 :=
by { sorry }

end part1_part2_part3_l117_117283


namespace map_distance_ratio_l117_117430

theorem map_distance_ratio (actual_distance_km : ℕ) (map_distance_cm : ℕ) (h1 : actual_distance_km = 6) (h2 : map_distance_cm = 20) : map_distance_cm / (actual_distance_km * 100000) = 1 / 30000 :=
by
  -- Proof goes here
  sorry

end map_distance_ratio_l117_117430


namespace scientific_notation_of_858_million_l117_117792

theorem scientific_notation_of_858_million :
  858000000 = 8.58 * 10 ^ 8 :=
sorry

end scientific_notation_of_858_million_l117_117792


namespace unit_vector_AB_l117_117639

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem unit_vector_AB :
  let A := (1, 3)
  let B := (4, -1)
  let AB := (B.1 - A.1, B.2 - A.2)
  let mag_AB := magnitude AB
  (AB.1 / mag_AB, AB.2 / mag_AB) = (3 / 5 : ℝ, -4 / 5 : ℝ) :=
by
  let A := (1, 3)
  let B := (4, -1)
  let AB := (B.1 - A.1, B.2 - A.2)
  let mag_AB := magnitude AB
  have : (AB.1 / mag_AB, AB.2 / mag_AB) = (3 / 5 : ℝ, -4 / 5 : ℝ) := sorry
  exact this

end unit_vector_AB_l117_117639


namespace alpha_plus_beta_l117_117281

variable (α β : ℝ)

def α_acute : Prop := 0 < α ∧ α < Real.pi / 2
def β_acute : Prop := 0 < β ∧ β < Real.pi / 2

theorem alpha_plus_beta (h1: cos (α - β) = (√5) / 5)
                        (h2: cos (2 * α) = (√10) / 10)
                        (α_acute : 0 < α ∧ α < Real.pi / 2)
                        (β_acute : 0 < β ∧ β < Real.pi / 2)
                        (h3: α < β) :
                        α + β = 3 * Real.pi / 4 := 
sorry

end alpha_plus_beta_l117_117281


namespace average_k_of_polynomial_with_int_roots_l117_117294

theorem average_k_of_polynomial_with_int_roots :
  (∀ r1 r2 : ℕ, r1 * r2 = 36 → ∃ k : ℕ, k = r1 + r2) ∧ 
  (list.avg [37, 20, 15, 13, 12] = 19.4) :=
by
  sorry

end average_k_of_polynomial_with_int_roots_l117_117294


namespace Tyrone_total_money_l117_117476

theorem Tyrone_total_money :
  let usd_bills := 4 * 1 + 1 * 10 + 2 * 5 + 30 * 0.25 + 5 * 0.5 + 48 * 0.1 + 12 * 0.05 + 4 * 1 + 64 * 0.01 + 3 * 2 + 5 * 0.5
  let euro_to_usd := 20 * 1.1
  let pound_to_usd := 15 * 1.32
  let cad_to_usd := 6 * 0.76
  let total_usd_currency := usd_bills
  let total_foreign_usd_currency := euro_to_usd + pound_to_usd + cad_to_usd
  let total_money := total_usd_currency + total_foreign_usd_currency
  total_money = 98.90 :=
by
  sorry

end Tyrone_total_money_l117_117476


namespace katherine_fruit_count_l117_117333

variables (apples pears bananas total_fruit : ℕ)

theorem katherine_fruit_count (h1 : apples = 4) 
  (h2 : pears = 3 * apples)
  (h3 : total_fruit = 21) 
  (h4 : total_fruit = apples + pears + bananas) : bananas = 5 := 
by sorry

end katherine_fruit_count_l117_117333


namespace chords_parallel_l117_117549

variables {A B M N A1 B1 : Type} [InCircle A B M N] 

-- Assume the conditions
variable (h1 : arc_length B1 N + arc_length A M = arc_length B1 A + arc_length N M)
variable (h2 : arc_length A1 B + arc_length N M = arc_length B M + arc_length A1 N)

-- Define the theorem that states the conclusion
theorem chords_parallel (h1 : arc_length B1 N + arc_length A M = arc_length B1 A + arc_length N M)
                        (h2 : arc_length A1 B + arc_length N M = arc_length B M + arc_length A1 N) :
  parallel (chord A A1) (chord B B1) :=
  sorry

end chords_parallel_l117_117549


namespace total_weight_of_remaining_macaroons_l117_117733

def total_weight_remaining_macaroons (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (bags : ℕ) (bags_eaten : ℕ) : ℕ :=
  let macaroons_per_bag := total_macaroons / bags
  let remaining_macaroons := total_macaroons - macaroons_per_bag * bags_eaten
  remaining_macaroons * weight_per_macaroon

theorem total_weight_of_remaining_macaroons
  (total_macaroons : ℕ)
  (weight_per_macaroon : ℕ)
  (bags : ℕ)
  (bags_eaten : ℕ)
  (h1 : total_macaroons = 12)
  (h2 : weight_per_macaroon = 5)
  (h3 : bags = 4)
  (h4 : bags_eaten = 1)
  : total_weight_remaining_macaroons total_macaroons weight_per_macaroon bags bags_eaten = 45 := by
  sorry

end total_weight_of_remaining_macaroons_l117_117733


namespace multiple_of_deans_height_l117_117539

variable (Dean_height : ℕ) (water_depth : ℕ)

theorem multiple_of_deans_height (h1 : Dean_height = 6) (h2 : water_depth = 60) : water_depth / Dean_height = 10 :=
by
  rw [h1, h2]
  sorry

end multiple_of_deans_height_l117_117539


namespace find_second_number_l117_117149

-- Definitions for the conditions
def ratio_condition (x : ℕ) : Prop := 5 * x = 40

-- The theorem we need to prove, i.e., the second number is 8 given the conditions
theorem find_second_number (x : ℕ) (h : ratio_condition x) : x = 8 :=
by sorry

end find_second_number_l117_117149


namespace sum_of_coefficients_l117_117646

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def polynomial (a : ℝ) (x : ℝ) : ℝ :=
  (2 + a * x) * (1 + x)^5

def x2_coefficient_condition (a : ℝ) : Prop :=
  2 * binomial_coefficient 5 2 + a * binomial_coefficient 5 1 = 15

theorem sum_of_coefficients (a : ℝ) (h : x2_coefficient_condition a) : 
  polynomial a 1 = 64 := 
sorry

end sum_of_coefficients_l117_117646


namespace evaluate_expression_l117_117586

-- Definitions based on the given conditions
def x1 : ℝ := 100^(1/2)
def x2 : ℝ := 64^(-1/3)
def x3 : ℝ := 81^(1/4)

-- Target theorem to prove
theorem evaluate_expression : x1 * x2 * x3 = 15 / 2 := 
by 
  have h1 : x1 = 10 := by
    rw [Real.rpow_mul_eq, Real.zero_add, Real.one, Real.rpow_one]
  have h2 : x2 = 1 / 4 := by
    rw [Real.inv_rpow, Real.pow_mul, Real.one_div, Real.pow_one, Real.pow_inv]
  have h3 : x3 = 3 := by
    rw [Real.rpow_mul_eq, Real.zero_add, Real.one, Real.rpow_one]
  rw [h1, h2, h3]
  norm_num

end evaluate_expression_l117_117586


namespace real_root_of_cubic_l117_117762

theorem real_root_of_cubic (a b : ℝ) (h : (a : ℝ) ≠ 0) :
  (1 : a = 2) ∧ (1 : b = 6) ∧ Root (2 * x ^ 3 + 3 * x ^ 2 + 6 * x - 125) (\frac{5}{2}) :=
by
  sorry

end real_root_of_cubic_l117_117762


namespace ideal_number_of_sequence_2_a1_a2_a100_l117_117390

-- Define the sequence a_n and its sum S_n
variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Define the ideal number T_n as (S_1 + S_2 + ... + S_n) / n
def T (n : ℕ) : ℕ := (Finset.range n).sum S / n

-- Given condition: the ideal number T_100 of the sequence a_1, a_2, ..., a_100 is 101.
axiom T_100_eq_101 : T S 100 = 101

-- We need to prove that the ideal number of the sequence 2, a_1, a_2, ..., a_100 is 102.
theorem ideal_number_of_sequence_2_a1_a2_a100 : 
  T (λ n, if n = 0 then 2 else 2 + S n) 101 = 102 := 
by sorry

end ideal_number_of_sequence_2_a1_a2_a100_l117_117390


namespace geometric_sequence_a6_l117_117327

theorem geometric_sequence_a6 (a_4 a_8 a_6 : ℝ) (a_seq : ℕ → ℝ)
  (h1 : a_4 = a_seq 4)
  (h2 : a_8 = a_seq 8)
  (h3 : a_4 > 0)
  (h4 : a_8 > 0)
  (h5 : a_seq 6 = a_6)
  (h6 : ∀ n, a_seq (n+1) / a_seq n = a_8 / a_4)
  (ineq_sol : {x : ℝ | x^2 - 4*x + 3 < 0} = set.Ioo a_4 a_8) :
  a_6 = real.sqrt 3 := sorry

end geometric_sequence_a6_l117_117327


namespace opposite_of_neg_2023_l117_117036

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117036


namespace arithmetic_sequence_product_l117_117711

theorem arithmetic_sequence_product 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n+1) - a n = a 2 - a 1) -- This condition states that the sequence is arithmetic
  (h2 : a 2 ^ 2 + 2 * a 2 * a 8 + a 6 * a 10 = 16) :
  a 4 * a 6 = 4 :=
begin
  sorry -- Proof to be filled in
end

end arithmetic_sequence_product_l117_117711


namespace r25_moves_to_r35_prob_l117_117949

-- Conditions: sequence of 50 distinct real numbers
def sequence_50 (seq : Fin 50 → ℝ) : Prop :=
  Function.Injective seq

-- Definition of one bubble pass
def bubble_pass (seq : Array ℝ) : Array ℝ :=
  let n := seq.size
  seq.foldl (fun a i => 
    if i + 1 < n && a[i] > a[i + 1] then
      a.swap i (i + 1)
    else
      a) seq (List.range (n - 1))

-- The probability that number at position r_25 moves to the 35th position
def probability_r25_to_r35 (seq : Fin 50 → ℝ) : ℚ :=
  if sequence_50 seq then
    let arr := Array.mkArray 50 (λ i => seq i)
    let new_arr := bubble_pass arr
    if new_arr[34] = seq 24 then
      1 / 1260
    else
      0
  else
    0

-- Main theorem statement
theorem r25_moves_to_r35_prob :
  ∀ (seq : Fin 50 → ℝ), sequence_50 seq →
  probability_r25_to_r35 seq = 1 / 1260 :=
by
  sorry

end r25_moves_to_r35_prob_l117_117949


namespace magnitude_a_2b_l117_117317

variables {E : Type*} [inner_product_space ℝ E]

-- Definitions based solely on the conditions in a)
def vec_a {E : Type*} [inner_product_space ℝ E] := sorry
def vec_b {E : Type*} [inner_product_space ℝ E] := sorry

-- Assumptions from the conditions
axiom mag_a : ∥vec_a∥ = 2
axiom mag_b : ∥vec_b∥ = 1
axiom angle_ab : real.angle.cos_angle (inner_product_geometry.angle vec_a vec_b) = -1/2

-- The proof problem statement (question == answer given conditions)
theorem magnitude_a_2b : ∥vec_a + 2 • vec_b∥ = 2 := sorry

end magnitude_a_2b_l117_117317


namespace gcd_10010_15015_l117_117235

def a := 10010
def b := 15015

theorem gcd_10010_15015 : Nat.gcd a b = 5005 := by
  sorry

end gcd_10010_15015_l117_117235


namespace prime_count_in_range_200_220_l117_117131

theorem prime_count_in_range_200_220 : 
  ∃! (n : ℤ), n > 200 ∧ n < 220 ∧ nat.prime (int.natAbs n) := sorry

end prime_count_in_range_200_220_l117_117131


namespace speed_including_stoppages_l117_117224

-- Definitions
def speed_excluding_stoppages : ℤ := 50 -- kmph
def stoppage_time_per_hour : ℕ := 24 -- minutes

-- Theorem to prove the speed of the train including stoppages
theorem speed_including_stoppages (h1 : speed_excluding_stoppages = 50)
                                  (h2 : stoppage_time_per_hour = 24) :
  ∃ s : ℤ, s = 30 := 
sorry

end speed_including_stoppages_l117_117224


namespace opposite_of_neg_2023_l117_117071

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117071


namespace train_speed_conversion_l117_117922

def speed_kmph := 135
def km_to_m_factor := 1000
def hour_to_sec_factor := 3600
def speed_in_mps := (speed_kmph * km_to_m_factor) / hour_to_sec_factor

theorem train_speed_conversion :
  speed_in_mps = 37.5 := sorry

end train_speed_conversion_l117_117922


namespace domain_implies_range_k_l117_117814

theorem domain_implies_range_k :
  (∀ x : ℝ, kx^2 - 4*k*x + k + 6 ≥ 0) → (0 ≤ k ∧ k ≤ 2) :=
begin
  sorry
end

end domain_implies_range_k_l117_117814


namespace greatest_divisor_4665_6905_l117_117389

def digits_sum (n : ℕ) : ℕ :=
(n.digits 10).sum

theorem greatest_divisor_4665_6905 :
  ∃ n : ℕ, (n ∣ 4665) ∧ (n ∣ 6905) ∧ (digits_sum n = 4) ∧
  (∀ m : ℕ, ((m ∣ 4665) ∧ (m ∣ 6905) ∧ (digits_sum m = 4)) → (m ≤ n)) :=
sorry

end greatest_divisor_4665_6905_l117_117389


namespace find_prices_min_cost_l117_117763

-- Definitions based on conditions
def price_difference (x y : ℕ) : Prop := x - y = 50
def total_cost (x y : ℕ) : Prop := 2 * x + 3 * y = 250
def cost_function (a : ℕ) : ℕ := 50 * a + 6000
def min_items (a : ℕ) : Prop := a ≥ 80
def total_items : ℕ := 200

-- Lean 4 statements for the proof problem
theorem find_prices (x y : ℕ) (h1 : price_difference x y) (h2 : total_cost x y) :
  (x = 80) ∧ (y = 30) :=
sorry

theorem min_cost (a : ℕ) (h1 : min_items a) :
  cost_function a ≥ 10000 :=
sorry

#check find_prices
#check min_cost

end find_prices_min_cost_l117_117763


namespace starting_number_is_400_l117_117096

theorem starting_number_is_400 :
  (∃ N : ℕ, ∀ n, (N ≤ n ∧ n ≤ 1000) → (n % 2 = 1) ∧ (300 = nat.card (set_of (λ (n: ℕ), N ≤ n ∧ n ≤ 1000 ∧ (n % 2 = 1))))) → N = 400 :=
sorry

end starting_number_is_400_l117_117096


namespace volume_of_prism_l117_117175

-- Given conditions
def length : ℕ := 12
def width : ℕ := 8
def depth : ℕ := 8

-- Proving the volume of the rectangular prism
theorem volume_of_prism : length * width * depth = 768 := by
  sorry

end volume_of_prism_l117_117175


namespace right_triangle_inscribed_circle_length_l117_117451

-- Define the problem conditions and the statement in Lean

theorem right_triangle_inscribed_circle_length 
  (r : ℝ) (BAC : ℝ) (ACB : ℝ)
  (angle_BAC : BAC = 60)
  (angle_ACB : ACB = 90)
  (radius : r = 8) 
  (triangle_right : angle_ACB = 90)
  : ∃ (AC : ℝ), AC = 16 := sorry

end right_triangle_inscribed_circle_length_l117_117451


namespace range_of_hx_l117_117480

open Real

theorem range_of_hx (h : ℝ → ℝ) (a b : ℝ) (H_def : ∀ x : ℝ, h x = 3 / (1 + 3 * x^4)) 
  (H_range : ∀ y : ℝ, (y > 0 ∧ y ≤ 3) ↔ ∃ x : ℝ, h x = y) : 
  a + b = 3 := 
sorry

end range_of_hx_l117_117480


namespace area_inside_C_outside_A_B_l117_117567

noncomputable def radius_A := 2
noncomputable def radius_B := 2
noncomputable def radius_C := 1

def tangent_at_midpoint (A B C : ℝ) := 
  ∃ M : ℝ, M = (A + B) / 2 ∧ C = M

lemma area_circle (r : ℝ) : ℝ := π * r * r

theorem area_inside_C_outside_A_B :
    ∀ (A B C : ℝ)
    (h1 : radius_A = 2)
    (h2 : radius_B = 2)
    (h3 : radius_C = 1)
    (h4 : tangent_at_midpoint A B C),
    area_circle radius_C = π := 
by
    intros
    rw [area_circle, h3]
    simp
    sorry

end area_inside_C_outside_A_B_l117_117567


namespace maximum_area_of_triangle_ABC_l117_117680

variables {x1 x2 y1 y2 : ℝ}

def parabola := ∀ (x y : ℝ), y^2 = 6 * x

def point_on_parabola (x y : ℝ) : Prop := y^2 = 6 * x

def points_A_B_on_parabola := point_on_parabola x1 y1 ∧ point_on_parabola x2 y2

def x1_neq_x2 := x1 ≠ x2

def x1_plus_x2_eq_4 := x1 + x2 = 4

noncomputable def max_area_of_triangle_ABC (x1 x2 y1 y2 : ℝ) :=
  let y := λ (x : ℝ), - (x1 + x2) / 6 * x + (12 + y1 * y2) / (x1 + x2)
  let xc := 5
  let area := (λ (a b : ℝ × ℝ), (1 / 2 : ℝ) * abs ((a.1 - b.1) * b.2 - (b.1 - xc) * a.2))
  area (x1 - xc, y1) (x2 - xc, y2)

theorem maximum_area_of_triangle_ABC :
  points_A_B_on_parabola ∧ x1_neq_x2 ∧ x1_plus_x2_eq_4 →
  max_area_of_triangle_ABC x1 x2 y1 y2 = 14 / 3 * real.sqrt 7 :=
by
  sorry

end maximum_area_of_triangle_ABC_l117_117680


namespace angle_CHX_value_l117_117927

-- Define the acute triangle ABC
variable (A B C H X Y : Type)
variable [TriangleGeometry A B C]

-- Define angles and altitudes intersecting at orthocenter
variable (Angle_BAC Angle_ABC : ℝ)
variable (is_orthocenter : Orthocenter_of_triangle H A B C)
variable (altitude_AX : Altitude A X B C)
variable (altitude_BY : Altitude B Y A C)

-- Given conditions
axiom angle_BAC_eq : Angle_BAC = 55
axiom angle_ABC_eq : Angle_ABC = 65

-- Definition for angle CHX
def angle_CHX : ℝ := 90 - (180 - (Angle_BAC + Angle_ABC))

-- Theorem to prove
theorem angle_CHX_value : angle_CHX = 30 := sorry

end angle_CHX_value_l117_117927


namespace angle_ratio_is_two_l117_117387

theorem angle_ratio_is_two
  {A B C I : Type} [EuclideanGeometry]
  (h1 : is_incenter_triangle I A B C)
  (h2 : AB = AC + CI) :
  ∀ {θ₁ θ₂ : ℝ}, angle ACB θ₁ → angle ABC θ₂ → θ₁ / θ₂ = 2 :=
sorry

end angle_ratio_is_two_l117_117387


namespace opposite_of_neg_2023_l117_117026

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117026


namespace number_of_technicians_l117_117797

-- Define the problem statements
variables (T R : ℕ)

-- Conditions based on the problem description
def condition1 : Prop := T + R = 42
def condition2 : Prop := 3 * T + R = 56

-- The main goal to prove
theorem number_of_technicians (h1 : condition1 T R) (h2 : condition2 T R) : T = 7 :=
by
  sorry -- Proof is omitted as per instructions

end number_of_technicians_l117_117797


namespace min_value_reciprocal_sums_l117_117672

theorem min_value_reciprocal_sums {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hline : ∀ x y, (2 * a * x - b * y + 2 = 0) ↔ (x = -1 ∧ y = 2)) :
  (a + b) = 1 → (∃ m, m = a + b ∧ (∃ min_val, min_val = 4 ∧ (1 / a + 1 / b) >= min_val)) :=
by
  intro hsum
  use 1, hsum
  exists 4
  apply sorry

end min_value_reciprocal_sums_l117_117672


namespace opposite_of_neg_2023_l117_117030

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117030


namespace john_subtracts_79_l117_117100

theorem john_subtracts_79 (x : ℕ) (h : x = 40) : (x - 1)^2 = x^2 - 79 :=
by sorry

end john_subtracts_79_l117_117100


namespace find_x_l117_117679

noncomputable def x_solution (x y z : ℝ) : Prop :=
  0.65 * x = 0.20 * y ∧
  y = (617.5^2 - 42) * z ∧
  z = real.sqrt 25 + real.pi ∧
  x = 955283.7148

theorem find_x : ∃ x y z : ℝ, x_solution x y z :=
by
  -- Be explicit in stating we need to skip the proof for now
  sorry

end find_x_l117_117679


namespace total_ways_to_give_gifts_l117_117565

theorem total_ways_to_give_gifts (n k : ℕ) (hn : n = 5) (hk : k = 3) :
  (nat.choose n k) * nat.factorial k = 60 := by
  sorry

end total_ways_to_give_gifts_l117_117565


namespace count_whole_numbers_interval_l117_117324

theorem count_whole_numbers_interval : 
  ∃ n : ℕ, n = 8 ∧ 
  ∀ x : ℤ, (x > Int.ofNat (Real.sqrt 2) ∧ x < Int.ofReal (3 * Real.pi)) ↔ (2 ≤ x ∧ x ≤ 9) := by
sorry

end count_whole_numbers_interval_l117_117324


namespace solve_equation_l117_117780

theorem solve_equation 
  (x : ℚ)
  (h : (x^2 + 3*x + 4)/(x + 5) = x + 6) :
  x = -13/4 := 
by
  sorry

end solve_equation_l117_117780


namespace range_of_a_l117_117657

theorem range_of_a {a : ℝ} (h : ∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) :
  a ≤ -1 ∧ a ≠ -2 := 
sorry

end range_of_a_l117_117657


namespace aquatic_product_profit_l117_117219

noncomputable theory

-- Definitions based on problem conditions
def initial_profit_per_kg : ℝ := 10
def initial_monthly_sales_volume : ℝ := 500
def price_increase_effect_on_sales (x : ℝ) : ℝ := 10 * x
def profit_function (x : ℝ) : ℝ := (initial_profit_per_kg + x) * (initial_monthly_sales_volume - price_increase_effect_on_sales x)

theorem aquatic_product_profit (x : ℝ) :
  profit_function x = 8000 → x = 10 := sorry

end aquatic_product_profit_l117_117219


namespace find_BP_l117_117888

theorem find_BP
    (A B C D P : Type) [Point A] [Point B] [Point C] [Point D] [Point P]
    (h_circle : Circle A B C D) 
    (h_intersect : Intersect AC BD P)
    (h_AP : AP = 10) 
    (h_PC : PC = 2) 
    (h_BD : BD = 9) 
    (h_BP_DP : BP < DP) : 
    BP = 4 := 
sorry

end find_BP_l117_117888


namespace greater_prime_of_lcm_and_sum_l117_117442

-- Define the problem conditions
def is_prime (n: ℕ) : Prop := Nat.Prime n
def is_lcm (a b l: ℕ) : Prop := Nat.lcm a b = l

-- Statement of the theorem to be proved
theorem greater_prime_of_lcm_and_sum (x y: ℕ) 
  (hx: is_prime x) 
  (hy: is_prime y) 
  (hlcm: is_lcm x y 10) 
  (h_sum: 2 * x + y = 12) : 
  x > y :=
sorry

end greater_prime_of_lcm_and_sum_l117_117442


namespace problem_statement_l117_117626

theorem problem_statement (x y : ℝ) (h : complex.exp (-1) * complex.I + 2 = y + x * complex.I) : x ^ 3 + y = 1 := 
sorry

end problem_statement_l117_117626


namespace opposite_of_neg_2023_l117_117005

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117005


namespace opposite_of_neg_2023_l117_117040

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117040


namespace solve_for_t_l117_117682

open Real

def vec_inner (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vec_a : ℝ × ℝ := (5, 12)
def vec_b : ℝ × ℝ := (2, 0)

theorem solve_for_t (t : ℝ) : 
  let vec_c := vec_add vec_a (t, 0)
  in vec_inner vec_a vec_c = vec_inner vec_b vec_c → t = 13 / 2 :=
by
  intro _ h
  sorry

end solve_for_t_l117_117682


namespace chess_piece_problem_l117_117834

theorem chess_piece_problem
  (a b c : ℕ)
  (h1 : b = b * 2 - a)
  (h2 : c = c * 2)
  (h3 : a = a * 2 - b)
  (h4 : c = c * 2 - a + b)
  (h5 : a * 2 = 16)
  (h6 : b * 2 = 16)
  (h7 : c * 2 = 16) : 
  a = 26 ∧ b = 14 ∧ c = 8 := 
sorry

end chess_piece_problem_l117_117834


namespace opposite_of_negative_2023_l117_117053

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117053


namespace trigonometric_expression_l117_117222

noncomputable def sqrt3 := Real.sqrt 3
def sin20 := Real.sin (20 * Real.pi / 180)
def cos20 := Real.cos (20 * Real.pi / 180)
def sin70 := Real.sin (70 * Real.pi / 180)
def sin40 := Real.sin (40 * Real.pi / 180)

theorem trigonometric_expression :
  (sqrt3 / sin20) - (1 / cos20) = 4 :=
sorry

end trigonometric_expression_l117_117222


namespace opposite_of_neg_2023_l117_117063

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117063


namespace sum_of_squares_eq_l117_117143

theorem sum_of_squares_eq :
  (1000^2 + 1001^2 + 1002^2 + 1003^2 + 1004^2 + 1005^2 + 1006^2) = 7042091 :=
by {
  sorry
}

end sum_of_squares_eq_l117_117143


namespace recipe_sugar_amount_l117_117755

-- Definitions from A)
def cups_of_salt : ℕ := 9
def additional_cups_of_sugar (sugar salt : ℕ) : Prop := sugar = salt + 2

-- Statement to prove
theorem recipe_sugar_amount (salt : ℕ) (h : salt = cups_of_salt) : ∃ sugar : ℕ, additional_cups_of_sugar sugar salt ∧ sugar = 11 :=
by
  sorry

end recipe_sugar_amount_l117_117755


namespace two_bridges_problem_l117_117859

-- Definitions of Points and Variables
variables (A B B' d : Point)
variable (AB' : Line)

-- Conditions
def segment_parallel_and_equal (B B' : Point) : Prop := sorry
def on_same_side_closer_to_canal (B B' : Point) : Prop := sorry
def bank_closer_to_A (A d : Point) : Prop := sorry
def midpoint_perpendicular_of_segment (A B' : Point) : Line := sorry
def intersection_on_line (line1 line2 : Line) : Point := sorry

-- Problem Statement
theorem two_bridges_problem
    (hBB' : segment_parallel_and_equal B B')
    (hB_closer : on_same_side_closer_to_canal B B')
    (h_bank : bank_closer_to_A A d) :
    ∃ p1 p2 : Point,
    p1 = intersection_on_line (midpoint_perpendicular_of_segment A B') d ∧
    p2 = intersection_on_line AB' d := 
sorry

end two_bridges_problem_l117_117859


namespace opposite_of_neg_2023_l117_117017

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117017


namespace log_sufficient_not_necessary_l117_117260

theorem log_sufficient_not_necessary (a b : ℝ) (ha_pos : 0 < a) (ha_not_one : a ≠ 1) :
  (log a b > 0 → ((a - 1) * (b - 1) > 0)) ∧ ¬(log a b > 0 ↔ ((a - 1) * (b - 1) > 0)) :=
sorry

end log_sufficient_not_necessary_l117_117260


namespace part_I_part_II_l117_117307

def f (a x : ℝ) : ℝ := |x - a| + |x - 1 / 2|

theorem part_I (x : ℝ) : f (5 / 2) x ≤ x + 10 ↔ - (7 / 3) ≤ x ∧ x ≤ 13 :=
sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f a x ≥ a) ↔ a ≤ 1 / 4 :=
sorry

end part_I_part_II_l117_117307


namespace find_k_for_circle_l117_117253

theorem find_k_for_circle (k : ℝ) : (∃ x y : ℝ, (x^2 + 8*x + y^2 + 4*y - k = 0) ∧ (x + 4)^2 + (y + 2)^2 = 25) → k = 5 := 
by 
  sorry

end find_k_for_circle_l117_117253


namespace sum_of_squares_edges_l117_117799

-- Define Points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define given conditions (4 vertices each on two parallel planes)
def A1 : Point := { x := 0, y := 0, z := 0 }
def A2 : Point := { x := 1, y := 0, z := 0 }
def A3 : Point := { x := 1, y := 1, z := 0 }
def A4 : Point := { x := 0, y := 1, z := 0 }

def B1 : Point := { x := 0, y := 0, z := 1 }
def B2 : Point := { x := 1, y := 0, z := 1 }
def B3 : Point := { x := 1, y := 1, z := 1 }
def B4 : Point := { x := 0, y := 1, z := 1 }

-- Function to calculate distance squared between two points
def dist_sq (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2

-- The Theorem to be proven
theorem sum_of_squares_edges : dist_sq A1 B2 + dist_sq A2 B3 + dist_sq A3 B4 + dist_sq A4 B1 = 8 := by
  sorry

end sum_of_squares_edges_l117_117799


namespace elements_of_sequence_l117_117650

def sequence (n : Nat) : Nat := n^2 + n

theorem elements_of_sequence :
  (∃ n : Nat, sequence n = 12) ∧ (∃ n : Nat, sequence n = 30) := by
  sorry

end elements_of_sequence_l117_117650


namespace solve_equation_l117_117599

theorem solve_equation (x : ℝ) (hx : x ≠ 0) : 
  x^2 + 36 / x^2 = 13 ↔ (x = 2 ∨ x = -2 ∨ x = 3 ∨ x = -3) := by
  sorry

end solve_equation_l117_117599


namespace Renu_work_days_l117_117770

noncomputable def work_days : ℝ :=
  14.4 / 1.8

theorem Renu_work_days :
  (1 / work_days + 1 / 4.8 = 1 / 3) → work_days = 8 :=
begin
  sorry
end

end Renu_work_days_l117_117770


namespace complete_the_square_l117_117487

theorem complete_the_square (x : ℝ) : (x^2 - 6 * x - 3 = 0) → (x - 3)^2 = 12 :=
begin
  sorry
end

end complete_the_square_l117_117487


namespace domain_of_tan_arcsin_squared_l117_117960

theorem domain_of_tan_arcsin_squared :
  (∀ x, (-1 < x ∧ x < 1) → (∃ y, g(x) = y)) :=
begin
  sorry
end

def g (x : ℝ) : ℝ := 
  Real.tan (Real.arcsin (x ^ 2))

end domain_of_tan_arcsin_squared_l117_117960


namespace solve_equation_l117_117782

theorem solve_equation 
  (x : ℚ)
  (h : (x^2 + 3*x + 4)/(x + 5) = x + 6) :
  x = -13/4 := 
by
  sorry

end solve_equation_l117_117782


namespace compare_fractions_difference_l117_117497

theorem compare_fractions_difference :
  let a := (1 : ℝ) / 2
  let b := (1 : ℝ) / 3
  a - b = 1 / 6 :=
by
  sorry

end compare_fractions_difference_l117_117497


namespace opposite_of_neg_2023_l117_117009

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117009


namespace sum_three_times_integers_15_to_25_l117_117555

noncomputable def sumArithmeticSequence (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem sum_three_times_integers_15_to_25 :
  let a := 15
  let d := 1
  let n := 25 - 15 + 1
  3 * sumArithmeticSequence a d n = 660 := by
  -- This part can be filled in with the actual proof
  sorry

end sum_three_times_integers_15_to_25_l117_117555


namespace opposite_of_neg_2023_l117_117019

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117019


namespace gideon_current_age_l117_117463

theorem gideon_current_age :
  let total_marbles := 100 in
  let marbles_given_to_sister := total_marbles * 3 / 4 in
  let remaining_marbles := total_marbles - marbles_given_to_sister in
  let age_five_years_from_now := remaining_marbles * 2 in
  let current_age := age_five_years_from_now - 5 in
  current_age = 45 := 
by
  sorry

end gideon_current_age_l117_117463


namespace translation_right_f_g_l117_117661

-- Define the original function f
def f (x : ℝ) : ℝ := 2^(-x) + x

-- Define the translated function g
def g (x : ℝ) : ℝ := f (x - 3)

-- The theorem to prove that g(x) is the expected expression
theorem translation_right_f_g:
  ∀ x : ℝ, g x = 2^(-x + 3) + x - 3 := by
  sorry

end translation_right_f_g_l117_117661


namespace box_height_l117_117905

noncomputable def length : ℝ := 8
noncomputable def width : ℝ := 15
noncomputable def volume_cube : ℝ := 10
noncomputable def num_cubes : ℝ := 60
noncomputable def height (l w v_c n : ℝ) : ℝ := (n * v_c) / (l * w)

theorem box_height :
  height length width volume_cube num_cubes = 5 :=
by
  sorry

end box_height_l117_117905


namespace piece_length_is_111_l117_117613

-- Define the conditions
axiom condition1 : ∃ (x : ℤ), 9 * x ≤ 1000
axiom condition2 : ∃ (x : ℤ), 9 * x ≤ 1100

-- State the problem: Prove that the length of each piece is 111 centimeters
theorem piece_length_is_111 (x : ℤ) (h1 : 9 * x ≤ 1000) (h2 : 9 * x ≤ 1100) : x = 111 :=
by sorry

end piece_length_is_111_l117_117613


namespace probability_gold_coin_biased_l117_117551

-- Definitions based on conditions
def P_A : ℝ := 0.5
def P_not_A : ℝ := 0.5
def P_B_given_A : ℝ := 0.15
def P_B_given_not_A : ℝ := 0.2

-- Calculate P_B using the law of total probability
def P_B : ℝ := P_B_given_A * P_A + P_B_given_not_A * P_not_A

-- Calculate P_A_given_B using Bayes' theorem
def P_A_given_B : ℝ := (P_B_given_A * P_A) / P_B

-- Prove final result
theorem probability_gold_coin_biased : P_A_given_B = 3 / 7 :=
by 
  sorry

end probability_gold_coin_biased_l117_117551


namespace log3_45_times_40_not_directly_determined_l117_117282

-- Conditions
axiom log2_of_16 : log 2 16 = 4
axiom log3_of_27 : log 3 27 = 3

-- Function Definitions
def log2_256 := log 2 256
def log3_81 := log 3 81
def log2_20_times_12 := log 2 (20 * 12)
def log3_45_times_40 := log 3 (45 * 40)
def log5_25 := log 5 25

-- Main Theorem
theorem log3_45_times_40_not_directly_determined :
  log3_45_times_40 = log 3 (45 * 40) :=
by 
  sorry

end log3_45_times_40_not_directly_determined_l117_117282


namespace subtraction_calculation_l117_117106

theorem subtraction_calculation (a b : ℤ) (h : b = 40) (h1 : a = b - 1) : (a * a) = (b * b) - 79 := 
by
  -- Using the given condition
  have h2 : a * a = (b - 1) * (b - 1),
  from by rw [h1],
  -- Expanding using binomial theorem
  rw [mul_sub, sub_mul, mul_one, ← square_eq, sub_sub, one_mul, one_mul] at h2,
  -- Proving the theorem
  rw [sub_add] at h2,
  exact h2,
  sorry

end subtraction_calculation_l117_117106


namespace find_number_l117_117506

theorem find_number (number : ℝ) (h : 0.001 * number = 0.24) : number = 240 :=
sorry

end find_number_l117_117506


namespace katherine_has_5_bananas_l117_117334

/-- Katherine has 4 apples -/
def apples : ℕ := 4

/-- Katherine has 3 times as many pears as apples -/
def pears : ℕ := 3 * apples

/-- Katherine has a total of 21 pieces of fruit (apples + pears + bananas) -/
def total_fruit : ℕ := 21

/-- Define the number of bananas Katherine has -/
def bananas : ℕ := total_fruit - (apples + pears)

/-- Prove that Katherine has 5 bananas -/
theorem katherine_has_5_bananas : bananas = 5 := by
  sorry

end katherine_has_5_bananas_l117_117334


namespace john_subtracts_79_l117_117101

theorem john_subtracts_79 (x : ℕ) (h : x = 40) : (x - 1)^2 = x^2 - 79 :=
by sorry

end john_subtracts_79_l117_117101


namespace gcd_example_l117_117212

theorem gcd_example : Nat.gcd (101^11 + 1) (101^11 + 101^3 + 1) = 1 := by
  sorry

end gcd_example_l117_117212


namespace opposite_of_neg_2023_l117_117070

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117070


namespace remaining_macaroons_weight_l117_117729

-- Problem conditions
variables (macaroons_per_bake : ℕ) (weight_per_macaroon : ℕ) (bags : ℕ) (macaroons_eaten : ℕ)

-- Definitions from problem conditions
def macaroons_per_bake := 12
def weight_per_macaroon := 5
def bags := 4
def macaroons_per_bag := macaroons_per_bake / bags
def macaroons_eaten := macaroons_per_bag

-- Lean theorem
theorem remaining_macaroons_weight : (macaroons_per_bake - macaroons_eaten) * weight_per_macaroon = 45 :=
by
  have h1 : macaroons_per_bag = 12 / 4 := rfl
  have h2 : macaroons_per_bag = 3 := by norm_num [h1]
  have h3 : macaroons_eaten = 3 := h2
  have h4 : macaroons_per_bake - macaroons_eaten = 12 - 3 := rfl
  have h5 : macaroons_per_bake - macaroons_eaten = 9 := by norm_num [h4]
  have h6 : (macaroons_per_bake - macaroons_eaten) * weight_per_macaroon = 9 * 5 := by rw [h5]
  calc
    (macaroons_per_bake - macaroons_eaten) * weight_per_macaroon = 9 * 5 : by rw [h6]
    ... = 45 : by norm_num

end remaining_macaroons_weight_l117_117729


namespace quadratic_has_single_solution_l117_117574

theorem quadratic_has_single_solution (k : ℚ) : 
  (∀ x : ℚ, 3 * x^2 - 7 * x + k = 0 → x = 7 / 6) ↔ k = 49 / 12 := 
by
  sorry

end quadratic_has_single_solution_l117_117574


namespace gcd_101_pow_11_plus_1_and_101_pow_11_plus_101_pow_3_plus_1_l117_117209

open Nat

theorem gcd_101_pow_11_plus_1_and_101_pow_11_plus_101_pow_3_plus_1 :
  gcd (101 ^ 11 + 1) (101 ^ 11 + 101 ^ 3 + 1) = 1 := 
by
  sorry

end gcd_101_pow_11_plus_1_and_101_pow_11_plus_101_pow_3_plus_1_l117_117209


namespace largest_sphere_radius_l117_117185

theorem largest_sphere_radius
  (inner_radius outer_radius : ℝ)
  (circle_center : ℝ × ℝ × ℝ)
  (circle_radius : ℝ)
  (hx : inner_radius = 3)
  (hy : outer_radius = 5)
  (hz : circle_center = (4, 0, 1))
  (hr : circle_radius = 1) :
  ∃ r : ℝ, r = 4 :=
by
  use 4
  sorry

end largest_sphere_radius_l117_117185


namespace parallelogram_is_output_l117_117346

def isFlowchartSymbol (symbol : String) : Prop :=
  symbol = "parallelogram"

def symbolMeaning (symbol : String) : String :=
  if isFlowchartSymbol symbol then "Output"
  else "Unknown"

theorem parallelogram_is_output : symbolMeaning "parallelogram" = "Output" := by
  rw [symbolMeaning, if_pos rfl]

end parallelogram_is_output_l117_117346


namespace functional_equation_solution_l117_117228

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(f(x) + y) = f(f(x) - y) + 4 * f(x) * y) →
  (∀ x : ℝ, f(x) = 0) ∨ ∃ c : ℝ, ∀ x : ℝ, f(x) = x^2 + c :=
by 
  intro h 
  sorry

end functional_equation_solution_l117_117228


namespace distance_between_intersections_l117_117162

open Function

def cube_vertices : List (ℝ × ℝ × ℝ) :=
  [(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0), (5, 0, 5), (5, 5, 0), (5, 5, 5)]

def intersecting_points : List (ℝ × ℝ × ℝ) :=
  [(0, 3, 0), (2, 0, 0), (2, 5, 5)]

noncomputable def plane_distance_between_points : ℝ :=
  let S := (11 / 3, 0, 5)
  let T := (0, 5, 4)
  Real.sqrt ((11 / 3 - 0)^2 + (0 - 5)^2 + (5 - 4)^2)

theorem distance_between_intersections : plane_distance_between_points = Real.sqrt (355 / 9) :=
  sorry

end distance_between_intersections_l117_117162


namespace opposite_of_neg_2023_l117_117076

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117076


namespace general_term_formula_Tn_formula_l117_117298

noncomputable def a_n (n : ℕ) : ℕ := 
  if n = 1 then 1 else 2 * n - 1

noncomputable def S_n (n : ℕ) : ℕ :=
  n * n

noncomputable def b_n (n : ℕ) : ℝ :=
  1 / ((a_n n : ℝ) * (a_n (n + 1) : ℝ))

noncomputable def T_n (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, b_n (k + 1))

theorem general_term_formula (n : ℕ) (h : n ≥ 1) : a_n n = 2 * n - 1 := by
  induction n with
  | zero =>
    exfalso
    exact Nat.not_lt_zero 1 h
  | succ n ih => sorry

theorem Tn_formula (n : ℕ) (h : n ≥ 1) : T_n n = n / (2 * n + 1) := by
  induction n with
  | zero =>
    exfalso
    exact Nat.not_lt_zero 1 h
  | succ n ih => sorry

end general_term_formula_Tn_formula_l117_117298


namespace gideon_current_age_l117_117461

theorem gideon_current_age :
  let total_marbles := 100 in
  let marbles_given_to_sister := total_marbles * 3 / 4 in
  let remaining_marbles := total_marbles - marbles_given_to_sister in
  let age_five_years_from_now := remaining_marbles * 2 in
  let current_age := age_five_years_from_now - 5 in
  current_age = 45 := 
by
  sorry

end gideon_current_age_l117_117461


namespace volume_between_spheres_l117_117835

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_between_spheres :
  volume_of_sphere 10 - volume_of_sphere 4 = (3744 / 3) * Real.pi := by
  sorry

end volume_between_spheres_l117_117835


namespace f_relationship_l117_117648

noncomputable def f (x : ℝ) : ℝ := sorry -- definition of f needs to be filled in later

-- Conditions given in the problem
variable (h_diff : Differentiable ℝ f)
variable (h_gt : ∀ x: ℝ, deriv f x > f x)
variable (a : ℝ) (h_pos : a > 0)

theorem f_relationship (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_gt : ∀ x: ℝ, deriv f x > f x) (a : ℝ) (h_pos : a > 0) :
  f a > Real.exp a * f 0 :=
sorry

end f_relationship_l117_117648


namespace triangle_angle_bisector_eq_l117_117715

theorem triangle_angle_bisector_eq
  {A B C L F : Type}
  (h_triangle : ∀ (A B C : ℝ), triangle A B C)
  (h_angle_diff : ∃ (B C : ℝ), B - C = 90)
  (h_AL_bisector : ∀ (AL : ℝ), bisector (angle A B L) = bisector (angle A C L))
  (h_ext_bisector : ∀ (ext_bisector : ℝ), ext_bisector (angle A) intersects line BC at F) :
  length_segment AL = length_segment AF := 
sorry

end triangle_angle_bisector_eq_l117_117715


namespace probability_at_least_one_from_each_language_l117_117401

noncomputable def probability_at_least_one_from_each :: ℚ :=
  (76 / 4940 : ℚ)

theorem probability_at_least_one_from_each_language 
  (F S G total : ℕ)
  (students_in_F : F = 26)
  (students_in_S : S = 29)
  (students_in_G : G = 12)
  (total_students : total = 40) :
  let p := (76 / 4940 : ℚ) in
  3 < total →
  F + S + G - total ≥ 0 →
  p = probability_at_least_one_from_each :=
by
  intros,
  sorry

end probability_at_least_one_from_each_language_l117_117401


namespace graph_inequality_l117_117412

variable (G : Type)
variable [Graph G]

noncomputable def min_degree (G : Graph G) : Prop := 
  ∀ v : G, degree v ≥ 3

noncomputable def connected (G : Graph G) : Prop := 
  ∀ u v : G, u ≠ v → ∃ p : Path G u v, nonempty p

theorem graph_inequality (G : Graph G) [connected G] [min_degree G] : 
  let v := Graph.vertices G
  let g := Graph.edges G
  let s := Graph.faces G
  g ≤ 3 * s - 6 :=
by sorry

end graph_inequality_l117_117412


namespace find_initial_friends_l117_117470

-- Define the given constants and conditions
def initial_friends (F : ℕ) : Prop :=
    let total_players := F + 2 in
    let lives_per_player := 6 in
    let total_lives := 24 in
    lives_per_player * total_players = total_lives

-- The theorem stating the number of initial friends
theorem find_initial_friends : ∃ (F : ℕ), initial_friends F ∧ F = 2 := by
    sorry

end find_initial_friends_l117_117470


namespace base8_to_base10_conversion_l117_117546

theorem base8_to_base10_conversion : 
  let num_in_base8 := "276" : List Char,
      base := 8
  ∃ (num_in_base10 : ℕ), 
  (num_in_base10 = 2 * base^2 + 7 * base^1 + 6 * base^0) ∧ num_in_base10 = 190 :=
begin
  let base := 8,
  let num_in_base10 := 2 * base^2 + 7 * base^1 + 6 * base^0,
  use num_in_base10,
  split,
  { refl },
  { norm_num }
end

end base8_to_base10_conversion_l117_117546


namespace gideon_age_l117_117460

noncomputable def gideon_current_age : ℕ :=
  let total_marbles := 100
  let remaining_marbles := total_marbles * (1/4 : ℚ)
  let age_in_five_years := remaining_marbles * 2
  let curr_age := age_in_five_years - 5
  curr_age

theorem gideon_age : gideon_current_age = 45 :=
by
  let total_marbles := 100
  let remaining_marbles := total_marbles * (1/4 : ℚ)
  let age_in_five_years := remaining_marbles * 2
  let curr_age := age_in_five_years - 5
  have h1 : total_marbles = 100 := rfl
  have h2 : remaining_marbles = 25 := by norm_num
  have h3 : age_in_five_years = 50 := by norm_num
  have h4 : curr_age = 45 := by norm_num
  exact h4

end gideon_age_l117_117460


namespace rationalize_denominator_l117_117422

theorem rationalize_denominator :
  (35 / Real.sqrt 35) = Real.sqrt 35 :=
sorry

end rationalize_denominator_l117_117422


namespace final_salary_correct_l117_117753

-- Define conditions
def initialSalary : ℝ := 2500
def raisePercent : ℝ := 25 / 100
def payCutPercent : ℝ := 15 / 100
def taxPercent : ℝ := 10 / 100

-- Define the transformations
def salaryAfterRaise := initialSalary * (1 + raisePercent)
def salaryAfterPayCut := salaryAfterRaise * (1 - payCutPercent)
def finalTakeHomeSalary := salaryAfterPayCut * (1 - taxPercent)

-- The proposition to be proved
theorem final_salary_correct : finalTakeHomeSalary = 2390.625 := by
  sorry

end final_salary_correct_l117_117753


namespace gcd_10010_15015_l117_117234

def a := 10010
def b := 15015

theorem gcd_10010_15015 : Nat.gcd a b = 5005 := by
  sorry

end gcd_10010_15015_l117_117234


namespace Gideon_age_l117_117455
-- Import necessary libraries

-- Define the initial conditions and the final proof statement
theorem Gideon_age :
  ∀ (total_marbles gideon_age_now : ℕ) (frac : ℚ),
    total_marbles = 100 →
    frac = 3 / 4 →
    let marbles_given_away := (frac * total_marbles).to_nat in
    let remaining_marbles := total_marbles - marbles_given_away in
    let age_five_years_from_now := 2 * remaining_marbles in
    age_five_years_from_now = gideon_age_now + 5 →
    gideon_age_now = 45 :=
by
  intros total_marbles gideon_age_now frac H1 H2 H3
  sorry

end Gideon_age_l117_117455


namespace andrew_paid_1428_l117_117548

-- Define the constants for the problem
def rate_per_kg_grapes : ℕ := 98
def kg_grapes : ℕ := 11

def rate_per_kg_mangoes : ℕ := 50
def kg_mangoes : ℕ := 7

-- Calculate the cost of grapes and mangoes
def cost_grapes := rate_per_kg_grapes * kg_grapes
def cost_mangoes := rate_per_kg_mangoes * kg_mangoes

-- Calculate the total amount paid
def total_amount_paid := cost_grapes + cost_mangoes

-- State the proof problem
theorem andrew_paid_1428 :
  total_amount_paid = 1428 :=
by
  -- Add the proof to verify the calculations
  sorry

end andrew_paid_1428_l117_117548


namespace opposite_of_neg_2023_l117_117021

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117021


namespace solve_for_a_l117_117312

-- Define the lines
def l1 (x y : ℝ) := x + y - 2 = 0
def l2 (x y a : ℝ) := 2 * x + a * y - 3 = 0

-- Define orthogonality condition
def perpendicular (m₁ m₂ : ℝ) := m₁ * m₂ = -1

-- The theorem to prove
theorem solve_for_a (a : ℝ) :
  (∀ x y : ℝ, l1 x y → ∀ x y : ℝ, l2 x y a → perpendicular (-1) (-2 / a)) → a = 2 := 
sorry

end solve_for_a_l117_117312


namespace clock_angle_at_3_15_l117_117932

-- Definitions of the parameters given in the problem.
def hour_angle (h : ℕ) (m : ℕ) : ℝ :=
  (h % 12) * 30 + (m / 60.0) * 30

def minute_angle (m : ℕ) : ℝ :=
  m * 6

-- Statement of the problem
theorem clock_angle_at_3_15 :
  let h := 3
  let m := 15
  hour_angle h m = 97.5 → 
  minute_angle m = 90 →
  abs (hour_angle h m - minute_angle m) = 7.5 :=
by
  intros h_eq m_eq
  simp [hour_angle, minute_angle] at *
  sorry

end clock_angle_at_3_15_l117_117932


namespace max_quadratic_expr_l117_117962

noncomputable def quadratic_expr (y : ℝ) : ℝ :=
-9 * y^2 + 15 * y + 3

theorem max_quadratic_expr : ∃ y ∈ set.univ, quadratic_expr y = 6.25 ∧ ∀ z ∈ set.univ, quadratic_expr z ≤ 6.25 :=
by sorry

end max_quadratic_expr_l117_117962


namespace angle_in_triangle_l117_117347

theorem angle_in_triangle (angle1 angle2 angle3: ℝ) (A: ℝ) :
  angle1 = 120 ∧ angle2 = 50 ∧ angle3 = 130 ->
  180 - angle2 - (180 - angle3) + A = 180 ∧ 180 - angle1 - angle2 = 10 ->
  A = 120 :=
by 
  intros hc1 hc2
  cases hc1 with h120 h50 h130 h10
  linarith

end angle_in_triangle_l117_117347


namespace parabola_properties_l117_117225

theorem parabola_properties
  (p q r : ℝ)
  (h1 : ∀ x, ((y = px^2 + qx + r) ↔ (y = p(x-3)^2 + 7)))
  (h2 : 4 = p * (0 - 3)^2 + 7) :
  p + q + r = 13 / 3 :=
begin
  sorry
end

end parabola_properties_l117_117225


namespace hexagon_diagonals_sum_l117_117904

noncomputable def sum_of_diagonals (r : ℝ) (sides : Fin 6 → ℝ) : ℝ :=
  if (sides 0 = 50) ∧ (∀ i : Fin 5, sides (i + 1) = 90) then
    let diagonal_sum := (86.9 + 144 + 145) in -- decimals due to Ptolemy's theorem application
    diagonal_sum
  else
    0

theorem hexagon_diagonals_sum :
  ∀ (r : ℝ) (sides : Fin 6 → ℝ), (sides 0 = 50) → (∀ i : Fin 5, sides (i + 1) = 90) →
  sum_of_diagonals r sides = 376 := 
by
  intros r sides h0 h1
  sorry

end hexagon_diagonals_sum_l117_117904


namespace orchard_problem_l117_117355

theorem orchard_problem (number_of_peach_trees number_of_apple_trees : ℕ) 
  (h1 : number_of_apple_trees = number_of_peach_trees + 1700)
  (h2 : number_of_apple_trees = 3 * number_of_peach_trees + 200) :
  number_of_peach_trees = 750 ∧ number_of_apple_trees = 2450 :=
by
  sorry

end orchard_problem_l117_117355


namespace cookies_baking_l117_117841

theorem cookies_baking : 
  ∀ (milk_quarts : ℕ) (sugar_cups : ℕ), 
    milk_quarts = 4 → sugar_cups = 6 → 
    (∃ (milk_in_pints : ℝ) (sugar_in_cups : ℝ), 
      milk_in_pints = (milk_quarts * 2) * (6 / 24) ∧ sugar_in_cups = sugar_cups * (6 / 24) ∧ 
      milk_in_pints = 2 ∧ sugar_in_cups = 1.5) :=
by
  intros milk_quarts sugar_cups H1 H2
  use [(milk_quarts * 2) * (6 / 24), sugar_cups * (6 / 24)]
  sorry

end cookies_baking_l117_117841


namespace bricks_required_to_pave_courtyard_l117_117154

theorem bricks_required_to_pave_courtyard :
  let courtyard_length : ℝ := 25
  let courtyard_width : ℝ := 16
  let brick_length : ℝ := 0.20
  let brick_width : ℝ := 0.10
  let area_courtyard := courtyard_length * courtyard_width
  let area_brick := brick_length * brick_width
  let number_of_bricks := area_courtyard / area_brick
  number_of_bricks = 20000 := by
    let courtyard_length : ℝ := 25
    let courtyard_width : ℝ := 16
    let brick_length : ℝ := 0.20
    let brick_width : ℝ := 0.10
    let area_courtyard := courtyard_length * courtyard_width
    let area_brick := brick_length * brick_width
    let number_of_bricks := area_courtyard / area_brick
    sorry

end bricks_required_to_pave_courtyard_l117_117154


namespace number_of_teams_l117_117833

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
by
  sorry

end number_of_teams_l117_117833


namespace arithmetic_progression_common_difference_and_first_terms_l117_117544

def sum (n : ℕ) : ℕ := 5 * n ^ 2
def Sn (a1 d n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_progression_common_difference_and_first_terms:
  ∀ n : ℕ, Sn 5 10 n = sum n :=
by
  sorry

end arithmetic_progression_common_difference_and_first_terms_l117_117544


namespace max_sin_sum_l117_117337

theorem max_sin_sum (A B C : ℝ) (hA : 0 < A) (hA_lt_pi : A < π)
  (hB : 0 < B) (hB_lt_pi : B < π) (hC : 0 < C) (hC_lt_pi : C < π)
  (h_sum : A + B + C = π) :
  sin A + sin B + sin C ≤ 3 * (Real.sin (π / 3)) :=
by
  have h_convex : ∀ (x1 x2 x3 : ℝ) (hx1 : x1 ∈ Ioo 0 π) (hx2 : x2 ∈ Ioo 0 π) (hx3 : x3 ∈ Ioo 0 π),
    (Real.sin x1 + Real.sin x2 + Real.sin x3) / 3 ≤ Real.sin ((x1 + x2 + x3) / 3),
    from sorry,
  have key := h_convex A B C ⟨hA, hA_lt_pi⟩ ⟨hB, hB_lt_pi⟩ ⟨hC, hC_lt_pi⟩,
  calc
    sin A + sin B + sin C ≤ 3 * (Real.sin (π / 3)) : by linarith [key, h_sum]

end max_sin_sum_l117_117337


namespace speed_of_man_is_correct_approx_l117_117920

-- Given conditions as Lean definitions
def length_of_train : ℝ := 270 -- in meters
def speed_of_train_kmph : ℝ := 25 -- in kmph
def crossing_time : ℝ := 36 -- in seconds

-- Conversion factors
def km_to_m : ℝ := 1000 -- meters in a kilometer
def hour_to_sec : ℝ := 3600 -- seconds in an hour

-- Compute speed of train in m/s
def speed_of_train_ms : ℝ := speed_of_train_kmph * km_to_m / hour_to_sec -- in m/s

-- Expected speed of man in kmph
def speed_of_man_kmph : ℝ := 2.0016 -- in kmph

-- Relative speed calculation
def relative_speed := speed_of_train_ms + (speed_of_man_kmph * km_to_m / hour_to_sec) -- in m/s

-- Speed calculation using Distance = Speed * Time
def calculated_relative_speed := length_of_train / crossing_time -- in m/s

-- Lean 4 statement to prove the man's speed is approximately 2.0016 kmph
theorem speed_of_man_is_correct_approx : 
  abs (relative_speed - calculated_relative_speed) < 0.0001 := sorry

end speed_of_man_is_correct_approx_l117_117920


namespace find_pairs_l117_117379

theorem find_pairs (p a : ℕ) (hp_prime : Nat.Prime p) (hp_ge_2 : p ≥ 2) (ha_ge_1 : a ≥ 1) (h_p_ne_a : p ≠ a) :
  (a + p) ∣ (a^2 + p^2) → (a = p ∧ p = p) ∨ (a = p^2 - p ∧ p = p) ∨ (a = 2 * p^2 - p ∧ p = p) :=
by
  sorry

end find_pairs_l117_117379


namespace number_of_ordered_pairs_l117_117527

theorem number_of_ordered_pairs (a b : ℕ) (ha : a > 0) (hb : b > a)
  (h : (a - 2) * (b - 2) * 2 = a * b) :
  (a, b) = (5, 12) ∨ (a, b) = (6, 8) :=
begin
  sorry
end

end number_of_ordered_pairs_l117_117527


namespace find_f_l117_117737

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def f (p : ℕ) : ℕ := sorry

axiom P : ∀ p q ∈ {n : ℕ | is_prime n}, f(p)^(f(q)) + q^p = f(q)^(f(p)) + p^q

theorem find_f (p : ℕ) (hp : is_prime p) : f(p) = p :=
sorry

end find_f_l117_117737


namespace gcd_10010_15015_l117_117239

theorem gcd_10010_15015 :
  Int.gcd 10010 15015 = 5005 :=
by 
  sorry

end gcd_10010_15015_l117_117239


namespace find_width_of_plot_l117_117174

def length : ℕ := 90
def poles : ℕ := 52
def distance_between_poles : ℕ := 5
def perimeter : ℕ := poles * distance_between_poles

theorem find_width_of_plot (perimeter_eq : perimeter = 2 * (length + width)) : width = 40 := by
  sorry

end find_width_of_plot_l117_117174


namespace problem_statement_l117_117262

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

variable (a x y t : ℝ) 

theorem problem_statement : 
  (log_base a x + 3 * log_base x a - log_base x y = 3) ∧ (a > 1) ∧ (x = a ^ t) ∧ (0 < t ∧ t ≤ 2) ∧ (y = 8) 
  → (a = 16) ∧ (x = 64) := 
by 
  sorry

end problem_statement_l117_117262


namespace smallest_solution_to_equation_l117_117251

theorem smallest_solution_to_equation :
  let x := 4 - Real.sqrt 2
  ∃ x, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
       ∀ y, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y :=
  by
    let x := 4 - Real.sqrt 2
    sorry

end smallest_solution_to_equation_l117_117251


namespace tangent_line_solution_l117_117975

variables (x y : ℝ)

noncomputable def circle_equation (m : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + m * y = 0

def point_on_circle (m : ℝ) : Prop :=
  circle_equation 1 1 m

noncomputable def tangent_line_equation (x y : ℝ) : Prop :=
  x - 2 * y + 1 = 0

theorem tangent_line_solution (m : ℝ) :
  point_on_circle m →
  m = 2 →
  tangent_line_equation 1 1 :=
by
  sorry

end tangent_line_solution_l117_117975


namespace vertex_of_parabola_y_eq_x2_minus_2_l117_117800

theorem vertex_of_parabola_y_eq_x2_minus_2 :
  vertex (λ x : ℝ, x^2 - 2) = (0, -2) := 
sorry

end vertex_of_parabola_y_eq_x2_minus_2_l117_117800


namespace mutual_exclusivity_conditional_probability_l117_117196

noncomputable def BagA := {white := 3, red := 3, black := 2}
noncomputable def BagB := {white := 2, red := 2, black := 1}

axiom A1_event : Event BagA -> Prop
axiom A2_event : Event BagA -> Prop
axiom A3_event : Event BagA -> Prop
axiom B_event : Event BagB -> Prop

axiom P_A1 : P(A1_event) = 3 / 8
axiom P_A2 : P(A2_event) = 3 / 8
axiom P_A3 : P(A3_event) = 2 / 8

axiom mutually_exclusive : MutuallyExclusive [A1_event, A2_event, A3_event]

axiom P_B_given_A1 : P(B_event | A1_event) = 1 / 3

theorem mutual_exclusivity : MutuallyExclusive [A1_event, A2_event, A3_event] :=
  sorry

theorem conditional_probability : P(B_event | A1_event) = 1 / 3 :=
  sorry

end mutual_exclusivity_conditional_probability_l117_117196


namespace greatest_prime_saturated_96_prime_saturation_condition_value_less_than_sqrt_l117_117910

def prime_factors (n : ℕ) : list ℕ := 
  -- This should return the list of distinct prime factors of n
  sorry

def is_prime_saturated (n : ℕ) : Prop :=
  (prime_factors n).prod < Real.sqrt n

theorem greatest_prime_saturated_96 :
  ∀ n : ℕ, n < 100 → is_prime_saturated n → n ≤ 96 :=
by
  -- Proof is omitted
  sorry

theorem prime_saturation_condition (n : ℕ) :
  is_prime_saturated n ↔ (prime_factors n).prod < Real.sqrt n :=
by
  -- Proof is omitted
  sorry

theorem value_less_than_sqrt (n : ℕ) :
  (prime_factors n).prod < Real.sqrt n :=
by
  -- Proof demonstrating the product of prime factors condition
  sorry

end greatest_prime_saturated_96_prime_saturation_condition_value_less_than_sqrt_l117_117910


namespace solve_equation_l117_117777

theorem solve_equation (x : ℚ) (h : x ≠ -5) : 
  (x^2 + 3*x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 := by
  sorry

end solve_equation_l117_117777


namespace coeff_x3_l117_117712

open BigOperators

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

def coefficient_in_expansion (x : ℝ) :=
  let f := (x ^ 2 + 1) ^ 2
  let g := (x - 1) ^ 6
  (\sum r in Finset.range (7), binomial 6 r * (-1)^r * (x ^ (6 - r)) * 
  (\sum s in Finset.range 3, binomial 3 s * (x^4 + 2 * x^2 + 1))) * 
  (\sum t in Finset.range (9), x ^ t) 

theorem coeff_x3 : coefficient_in_expansion (-1) 3 = -32 := 
  sorry

end coeff_x3_l117_117712


namespace difference_between_c_and_a_l117_117432

variable (a b c : ℝ)

theorem difference_between_c_and_a (h1 : (a + b) / 2 = 30) (h2 : c - a = 60) : c - a = 60 :=
by
  exact h2

end difference_between_c_and_a_l117_117432


namespace num_isosceles_triangles_l117_117702

-- Definition of a point in 2D space
structure Point :=
  (x : ℕ)
  (y : ℕ)

-- Definitions for points A and B on the 7x7 geoboard
def A : Point := ⟨3, 3⟩
def B : Point := ⟨5, 3⟩

-- Define what it means for a triangle to be isosceles given three points
def is_isosceles (P Q R : Point) : Prop :=
  let d1 := (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2
  let d2 := (P.x - R.x) ^ 2 + (P.y - R.y) ^ 2
  let d3 := (Q.x - R.x) ^ 2 + (Q.y - R.y) ^ 2
  d1 = d2 ∨ d1 = d3 ∨ d2 = d3

-- Main theorem statement
theorem num_isosceles_triangles : (finset.univ.filter (λ C, is_isosceles A B C)).card = 10 :=
sorry

end num_isosceles_triangles_l117_117702


namespace max_cone_volume_l117_117630

-- Define the sphere's radius
def sphere_radius : ℝ := 9

-- Define the radius of the base circle of the cone
def r (h : ℝ) : ℝ := real.sqrt (81 - h^2)

-- Define the volume of the cone
def cone_volume (h : ℝ) : ℝ := (1 / 3) * real.pi * (r h) ^ 2 * h

-- Define the maximum height of the cone
def max_height : ℝ := 3 * real.sqrt 3

-- State the theorem to prove the maximum volume of the cone
theorem max_cone_volume : 
  cone_volume max_height = 54 * real.sqrt 3 * real.pi :=
sorry

end max_cone_volume_l117_117630


namespace f_even_function_l117_117817

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_even_function : ∀ x : ℝ, f x = f (-x) :=
by
  intro x
  show f x = f (-x)
  sorry

end f_even_function_l117_117817


namespace find_value_l117_117689

open Real

theorem find_value (a b : ℝ)
  (h : a^2 - 2 * b^2 - 2 = 0) : -3 * a^2 + 6 * b^2 + 2023 = 2017 :=
by 
  have h1 : a^2 - 2 * b^2 = 2, from calc
    a^2 - 2 * b^2 = 2 : by linarith [h],
  calc
    -3 * a^2 + 6 * b^2 + 2023
        = -3 * (a^2 - 2 * b^2) + 2023 : by linarith
    ... = -3 * 2 + 2023 : by rw [h1]
    ... = 2017 : by norm_num

end find_value_l117_117689


namespace circle_outside_square_area_l117_117536

noncomputable def area_circle_outside_square (r : ℝ) (a : ℝ) : ℝ :=
  π * r^2 - a^2

-- Given a square of side length 1 and a circle of radius 1/2 sharing the same center,
-- the area inside the circle but outside the square is (π - 2) / 4.
theorem circle_outside_square_area :
  let r := 1 / 2
  let a := 1
  area_circle_outside_square r a = (π - 2) / 4 :=
by
  sorry

end circle_outside_square_area_l117_117536


namespace probability_exactly_one_first_class_l117_117860

-- Define the probabilities
def prob_first_class_first_intern : ℚ := 2 / 3
def prob_first_class_second_intern : ℚ := 3 / 4
def prob_not_first_class_first_intern : ℚ := 1 - prob_first_class_first_intern
def prob_not_first_class_second_intern : ℚ := 1 - prob_first_class_second_intern

-- Define the event A, which is the event that exactly one of the two parts is of first-class quality
def prob_event_A : ℚ :=
  (prob_first_class_first_intern * prob_not_first_class_second_intern) +
  (prob_not_first_class_first_intern * prob_first_class_second_intern)

theorem probability_exactly_one_first_class (h1 : prob_first_class_first_intern = 2 / 3) 
    (h2 : prob_first_class_second_intern = 3 / 4) 
    (h3 : prob_event_A = 
          (prob_first_class_first_intern * (1 - prob_first_class_second_intern)) + 
          ((1 - prob_first_class_first_intern) * prob_first_class_second_intern)) : 
  prob_event_A = 5 / 12 := 
  sorry

end probability_exactly_one_first_class_l117_117860


namespace fg_of_3_l117_117314

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 + 2
def g (x : ℝ) : ℝ := 3 * x + 4

-- Theorem statement to prove f(g(3)) = 2199
theorem fg_of_3 : f (g 3) = 2199 :=
by
  sorry

end fg_of_3_l117_117314


namespace exam_scheduling_l117_117348

theorem exam_scheduling :
  ∃ (Subjects : Finset String) 
    (C M E : String) 
    (arrangements : Finset (List String)),
    C ∈ Subjects ∧ M ∈ Subjects ∧ E ∈ Subjects ∧
    -- Chinese must be in the first session
    (∀ (l : List String), l ∈ arrangements → List.head? = C) ∧ 
    -- Mathematics and English cannot be adjacent
    (∀ (l : List String), l ∈ arrangements → 
      (∀ i, (List.nth l i = some M ∧ List.nth l (i + 1) ≠ some E) ∧ 
             (List.nth l i = some E ∧ List.nth l (i + 1) ≠ some M)) ∧ 
    arrangements.card = 72 := 
sorry

end exam_scheduling_l117_117348


namespace significant_digits_of_square_side_l117_117447

theorem significant_digits_of_square_side (A : ℝ) (hA : A = 1.1025) :
  significant_digits (sqrt A) = 5 :=
  sorry

-- Definition of significant_digits (hypothetical as significant_digits definition is not standard)
def significant_digits (n : ℝ) : ℕ :=
  sorry

end significant_digits_of_square_side_l117_117447


namespace track_meet_earliest_time_l117_117936

theorem track_meet_earliest_time :
  let bella_lap := 5
  let chris_lap := 9
  let daniel_lap := 10
  let start_time := (7, 30) -- Representing 7:30 AM
  Nat.lcm (Nat.lcm bella_lap chris_lap) daniel_lap = 90 ∧ 
  start_time + 90.minutes = (9, 0) := 
by
  sorry

end track_meet_earliest_time_l117_117936


namespace fish_left_in_sea_l117_117510

theorem fish_left_in_sea : 
  let westward_initial := 1800
  let eastward_initial := 3200
  let north_initial := 500
  let eastward_caught := (2 / 5) * eastward_initial
  let westward_caught := (3 / 4) * westward_initial
  let eastward_left := eastward_initial - eastward_caught
  let westward_left := westward_initial - westward_caught
  let north_left := north_initial
  eastward_left + westward_left + north_left = 2870 := 
by 
  sorry

end fish_left_in_sea_l117_117510


namespace find_BP_l117_117887

theorem find_BP
    (A B C D P : Type) [Point A] [Point B] [Point C] [Point D] [Point P]
    (h_circle : Circle A B C D) 
    (h_intersect : Intersect AC BD P)
    (h_AP : AP = 10) 
    (h_PC : PC = 2) 
    (h_BD : BD = 9) 
    (h_BP_DP : BP < DP) : 
    BP = 4 := 
sorry

end find_BP_l117_117887


namespace man_speed_was_5_kmph_l117_117518

theorem man_speed_was_5_kmph (time_in_minutes : ℕ) (distance_in_km : ℝ)
  (h_time : time_in_minutes = 30)
  (h_distance : distance_in_km = 2.5) :
  (distance_in_km / (time_in_minutes / 60 : ℝ) = 5) :=
by
  sorry

end man_speed_was_5_kmph_l117_117518


namespace min_people_wearing_both_l117_117705

theorem min_people_wearing_both (n : ℕ) (h_lcm : n % 24 = 0) 
  (h_gloves : 3 * n % 8 = 0) (h_hats : 5 * n % 6 = 0) :
  ∃ x, x = 5 := 
by
  let gloves := 3 * n / 8
  let hats := 5 * n / 6
  let both := gloves + hats - n
  have h1 : both = 5 := sorry
  exact ⟨both, h1⟩

end min_people_wearing_both_l117_117705


namespace overall_percent_supporters_l117_117173

theorem overall_percent_supporters
  (percent_A : ℝ) (percent_B : ℝ)
  (members_A : ℕ) (members_B : ℕ)
  (supporters_A : ℕ)
  (supporters_B : ℕ)
  (total_supporters : ℕ)
  (total_members : ℕ)
  (overall_percent : ℝ) 
  (h1 : percent_A = 0.70) 
  (h2 : percent_B = 0.75)
  (h3 : members_A = 200) 
  (h4 : members_B = 800) 
  (h5 : supporters_A = percent_A * members_A) 
  (h6 : supporters_B = percent_B * members_B) 
  (h7 : total_supporters = supporters_A + supporters_B) 
  (h8 : total_members = members_A + members_B) 
  (h9 : overall_percent = (total_supporters : ℝ) / total_members * 100) :
  overall_percent = 74 := by
  sorry

end overall_percent_supporters_l117_117173


namespace value_of_b_minus_d_squared_l117_117874

theorem value_of_b_minus_d_squared
  (a b c d : ℤ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 9) :
  (b - d) ^ 2 = 4 :=
sorry

end value_of_b_minus_d_squared_l117_117874


namespace mary_potatoes_l117_117754

theorem mary_potatoes :
  ∀ (initial_potatoes : ℕ) (eaten_potatoes : ℕ), 
    initial_potatoes = 8 → 
    eaten_potatoes = 3 → 
    initial_potatoes - eaten_potatoes = 5 :=
by
  intros initial_potatoes eaten_potatoes h1 h2
  rw [h1, h2]
  exact rfl

end mary_potatoes_l117_117754


namespace sum_of_squares_of_roots_l117_117601

noncomputable theory
open Complex

theorem sum_of_squares_of_roots (n : ℕ) (h : 1 ≤ n-1) : 
  ∑ k in finset.range (n-1), (i * Complex.cot (k * Complex.pi / n))^2 = (n-1)*(n-2) / 3 := 
begin
  sorry -- proof goes here
end

end sum_of_squares_of_roots_l117_117601


namespace tank_emptying_time_l117_117184

noncomputable def cubic_feet_to_cubic_inches (cubic_feet : ℝ) : ℝ :=
  cubic_feet * (12 ^ 3)

noncomputable def net_emptying_rate (inlet_rate : ℝ) (outlet_rates : List ℝ) : ℝ :=
  (outlet_rates.sum - inlet_rate)

noncomputable def time_to_empty_tank (volume : ℝ) (net_rate : ℝ) : ℝ :=
  volume / net_rate

theorem tank_emptying_time :
  let volume_tank_in_cubic_feet := 45
  let inlet_rate := 5
  let outlet_rates := [12, 9, 6]
  let conversion_factor := 12
  let volume_tank_in_cubic_inches := cubic_feet_to_cubic_inches volume_tank_in_cubic_feet
  let net_rate := net_emptying_rate inlet_rate outlet_rates
  time_to_empty_ttank volume_tank_in_cubic_inches net_rate ≈ 3534.55 := by
    sorry

end tank_emptying_time_l117_117184


namespace alex_had_to_walk_8_miles_l117_117541

theorem alex_had_to_walk_8_miles : 
  let flat_speed := 20 
  let flat_time := 4.5 
  let uphill_speed := 12 
  let uphill_time := 2.5 
  let downhill_speed := 24 
  let downhill_time := 1.5 
  let total_distance := 164 
  let flat_distance := flat_speed * flat_time 
  let uphill_distance := uphill_speed * uphill_time 
  let downhill_distance := downhill_speed * downhill_time 
  let biking_distance := flat_distance + uphill_distance + downhill_distance 
  let walking_distance := total_distance - biking_distance 
  walking_distance = 8 := 
by 
  let flat_speed := 20
  let flat_time := 4.5
  let uphill_speed := 12
  let uphill_time := 2.5
  let downhill_speed := 24
  let downhill_time := 1.5
  let total_distance := 164
  let flat_distance := flat_speed * flat_time
  let uphill_distance := uphill_speed * uphill_time
  let downhill_distance := downhill_speed * downhill_time
  let biking_distance := flat_distance + uphill_distance + downhill_distance
  let walking_distance := total_distance - biking_distance
  sorry

end alex_had_to_walk_8_miles_l117_117541


namespace blanket_collection_ratio_l117_117988

theorem blanket_collection_ratio :
  ∀ (n_people blankets_per_person blankets_last_day total_blankets : ℕ),
  n_people = 15 →
  blankets_per_person = 2 →
  blankets_last_day = 22 →
  total_blankets = 142 →
  let blankets_first_day := n_people * blankets_per_person in
  let blankets_second_day := total_blankets - blankets_first_day - blankets_last_day in
  (blankets_second_day / blankets_first_day) = 3 :=
by
  intros n_people blankets_per_person blankets_last_day total_blankets
  intro h1 h2 h3 h4
  let blankets_first_day := n_people * blankets_per_person
  let blankets_second_day := total_blankets - blankets_first_day - blankets_last_day
  have h5 : blankets_first_day = 30 := by sorry
  have h6 : blankets_second_day = 90 := by sorry
  have ratio := blankets_second_day / blankets_first_day
  have h7 : ratio = 3 := by sorry
  exact h7

end blanket_collection_ratio_l117_117988


namespace find_line_equation_l117_117816

noncomputable def line_equation (l : affine_subspace ℝ) : Prop :=
  ∃ k : ℝ, (l = {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 21 = 0 ∨ p.1 = 3}) ∧
  ∃ P Q : (ℝ × ℝ), P = (3, 3) ∧ Q = (-1, 1) ∧
  distance P l = 4

theorem find_line_equation (l : affine_subspace ℝ) :
  line_equation l ↔
  ((∃ k : ℝ, l = {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 21 = 0}) ∨
  (l = {p : ℝ × ℝ | p.1 = 3})) ∧
  (∃ P Q : (ℝ × ℝ), P = (3, 3) ∧ Q = (-1, 1) ∧
  distance P l = 4) := by
  sorry

end find_line_equation_l117_117816


namespace opposite_of_negative_2023_l117_117058

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117058


namespace find_number_l117_117482

theorem find_number (x : ℝ) (h : (2 / 5) * x = 10) : x = 25 :=
sorry

end find_number_l117_117482


namespace y_give_z_start_l117_117881

variables (Vx Vy Vz T : ℝ)
variables (D : ℝ)

-- Conditions
def condition1 : Prop := Vx * T = Vy * T + 100
def condition2 : Prop := Vx * T = Vz * T + 200
def condition3 : Prop := T > 0

theorem y_give_z_start (h1 : condition1 Vx Vy T) (h2 : condition2 Vx Vz T) (h3 : condition3 T) : (Vy - Vz) * T = 200 := 
by
  sorry

end y_give_z_start_l117_117881


namespace avg_speed_difference_l117_117944

noncomputable def difference_in_speeds (d : ℝ) (v_R : ℝ) (time_diff : ℝ) : ℝ :=
  let t_R := d / v_R
  let t_P := t_R - time_diff
  let v_P := d / t_P
  v_P - v_R

theorem avg_speed_difference :
  difference_in_speeds 300 34.05124837953327 2 ≈ 10 :=
-- Note: ≈ is Lean's way of saying "approximately equal to"
by {
  sorry
}

end avg_speed_difference_l117_117944


namespace drop_perpendicular_l117_117116

structure Point :=
(x : ℝ)
(y : ℝ)

structure Circle :=
(center : Point)
(radius : ℝ)

def is_on_line (M : Point) (A B : Point) : Prop :=
∃ k : ℝ, M.x = A.x + k * (B.x - A.x) ∧ M.y = A.y + k * (B.y - A.y)

def is_on_circle (M : Point) (circle : Circle) : Prop :=
(M.x - circle.center.x)^2 + (M.y - circle.center.y)^2 = circle.radius^2

def is_perpendicular (line1 line2 : Point × Point) : Prop :=
let (A, B) := line1 in
let (C, D) := line2 in
(A.x - B.x) * (C.x - D.x) + (A.y - B.y) * (C.y - D.y) = 0

variables (M A B C D H : Point) (circle : Circle)

theorem drop_perpendicular (M_not_on_circle: ¬ is_on_circle M circle)
  (M_not_on_line: ¬ is_on_line M A B)
  (AB_diameter: is_on_circle A circle ∧ is_on_circle B circle)
  (C_on_circle: is_on_circle C circle)
  (D_on_circle: is_on_circle D circle) :
  is_perpendicular (M, H) (A, B) :=
sorry

end drop_perpendicular_l117_117116


namespace sum_as_common_fraction_l117_117594

/-- The sum of 0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 as a common fraction -/
theorem sum_as_common_fraction : (0.2 + 0.04 + 0.006 + 0.0008 + 0.00010) = (12345 / 160000) := by
  sorry

end sum_as_common_fraction_l117_117594


namespace ab_ac_bc_values_l117_117745

theorem ab_ac_bc_values (a b c : ℝ) (h : a + b + c = 3) :
  ∃ s, (∀ x ∈ s, ab + ac + bc = x) ∧ s = set.Iic 3 := sorry

end ab_ac_bc_values_l117_117745


namespace min_distance_parabola_focus_l117_117649

theorem min_distance_parabola_focus :
  let F : ℝ × ℝ := (1, 0),
      M : ℝ × ℝ := (1/4, a),
      N : ℝ × ℝ := (1/2, b),
      P : ℝ × ℝ := (1, c),
      Q : ℝ × ℝ := (4, d)

  ∀a b c d: ℝ, (a, b, c, d) ∈ {(a, b, c, d) | (1/4)^2 <= 4 * a ∧ (1/2)^2 <= 4 * b ∧ 1^2 <= 4 * c ∧ 4^2 <= 4 * d} ->
  dist (1/4, a) F < dist (1/2, b) F ∧ dist (1/4, a) F < dist (1, c) F ∧ dist (1/4, a) F < dist (4, d) F :=
by
  sorry

end min_distance_parabola_focus_l117_117649


namespace proof_problem_l117_117950

theorem proof_problem (x : ℝ) (hx1 : log (2 * x^3) 8 + log (16 * x^4) 8 = -4) 
  (hx2 : log (4 * x^2) 16 = 1) : (1 / x^14) = 1 / 2^14 :=
sorry

end proof_problem_l117_117950


namespace Tom_initial_investment_l117_117472

-- Initial definitions of the conditions
def Jose_investment : ℝ := 45000
def total_profit : ℝ := 72000
def Jose_share : ℝ := 40000
def Tom_share := total_profit - Jose_share
def Tom_months : ℕ := 12
def Jose_months : ℕ := 10
def proportion (T : ℝ) : ℝ := (T * Tom_months) / (Jose_investment * Jose_months)

theorem Tom_initial_investment : 
  ∃ T : ℝ, proportion T = (Tom_share / Jose_share) ∧ T = 30000 :=
begin
  sorry
end

end Tom_initial_investment_l117_117472


namespace min_value_fraction_l117_117640

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) : 
  ∃x : ℝ, (x = (1/a + 2/b)) ∧ (∀y : ℝ, (y = (1/a + 2/b)) → y ≥ 8) :=
by
  sorry

end min_value_fraction_l117_117640


namespace jane_ate_four_pieces_l117_117255

def total_pieces : ℝ := 12.0
def num_people : ℝ := 3.0
def pieces_per_person : ℝ := 4.0

theorem jane_ate_four_pieces :
  total_pieces / num_people = pieces_per_person := 
  by
    sorry

end jane_ate_four_pieces_l117_117255


namespace students_weekly_total_break_time_l117_117478

def total_week_break_time (mon_assembly fri_tutoring: Nat) : Nat :=
  let daily_break_time_mon_fri := 
      10 +         -- Morning warm-up activity
      30 +         -- Morning recess breaks 
      30 +         -- Lunch break and transition time
      30          -- Afternoon recess breaks (excluding Wednesday)
      in 
  let weekly_break_time := 5 * daily_break_time_mon_fri - daily_break_time_mon_fri + (10 + 30 + 30 + 35) in -- Adjustments for Wednesday
  weekly_break_time + mon_assembly + fri_tutoring

theorem students_weekly_total_break_time : total_week_break_time 30 45 = 590 := 
by
  sorry

end students_weekly_total_break_time_l117_117478


namespace opposite_of_neg_2023_l117_117039

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117039


namespace red_ball_higher_numbered_bin_l117_117098

-- Define the probability of a ball being tossed into bin k
def prob_bin (k : ℕ) : ℝ := 3 ^ (-k)

-- Define the probability that the red ball is in a higher-numbered bin than the green and blue balls
noncomputable def prob_red_higher (prob_bin : ℕ → ℝ) : ℝ :=
  let prob_same_bin := ∑' k:ℕ, (prob_bin k) ^ 3 in
  (1 - prob_same_bin) / 3

-- Prove the final probability is 25/78
theorem red_ball_higher_numbered_bin :
  prob_red_higher prob_bin = 25 / 78 :=
sorry

end red_ball_higher_numbered_bin_l117_117098


namespace point_line_distance_l117_117638

theorem point_line_distance (m : ℝ) :
  let M := (1, 4)
  let l := λ x y : ℝ, m * x + y - 1
  dist (M.1, M.2) l = 3 →
  m = 0 ∨ m = 3 / 4 :=
by
  sorry

end point_line_distance_l117_117638


namespace shoe_selection_ways_l117_117989

theorem shoe_selection_ways : 
  ∃ (n : ℕ), (∃ (pairs : Fin n → Fin 2 × Fin 2),
  n = 5 ∧
  (∃ (ways : ℕ), ways = 120 ∧
  ways = (let p := 5 in
          let left_pairs := 4 in
          let combinations := left_pairs * (left_pairs - 1) / 2 in
          let selected_pair_ways := 2 * 2 in
          p * combinations * selected_pair_ways))) :=
begin
  sorry
end

end shoe_selection_ways_l117_117989


namespace garden_roller_diameter_l117_117812

theorem garden_roller_diameter
  (l : ℝ) (A : ℝ) (r : ℕ) (pi : ℝ)
  (h_l : l = 2)
  (h_A : A = 44)
  (h_r : r = 5)
  (h_pi : pi = 22 / 7) :
  ∃ d : ℝ, d = 1.4 :=
by {
  sorry
}

end garden_roller_diameter_l117_117812


namespace greatest_cars_with_ac_not_racing_stripes_l117_117876

-- Definitions
def total_cars : ℕ := 100
def cars_without_ac : ℕ := 47
def cars_with_ac : ℕ := total_cars - cars_without_ac
def at_least_racing_stripes : ℕ := 53

-- Prove that the greatest number of cars that could have air conditioning but not racing stripes is 53
theorem greatest_cars_with_ac_not_racing_stripes :
  ∃ maximum_cars_with_ac_not_racing_stripes, 
    maximum_cars_with_ac_not_racing_stripes = cars_with_ac - 0 ∧
    maximum_cars_with_ac_not_racing_stripes = 53 := 
by
  sorry

end greatest_cars_with_ac_not_racing_stripes_l117_117876


namespace simplification_correct_l117_117485

noncomputable def given_equation (x : ℚ) : Prop := 
  x / (2 * x - 1) - 3 = 2 / (1 - 2 * x)

theorem simplification_correct (x : ℚ) (h : given_equation x) : 
  x - 3 * (2 * x - 1) = -2 :=
sorry

end simplification_correct_l117_117485


namespace extreme_value_at_one_monotonicity_of_f_l117_117668

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem extreme_value_at_one (a b : ℝ) :
  (f 1 a b = 10 ∧ (3 * (1 : ℝ)^2 + 2 * a * 1 + b = 0)) →
  (a = 4 ∧ b = -11) :=
by sorry


theorem monotonicity_of_f (a : ℝ) :
  let b := -a^2
  in
  (∀ x : ℝ, f x a b = x^3 + a * x^2 - a^2 * x + a^2) →
  (if a > 0 then 
    ( (∀ x < -a, 3 * x^2 + 2 * a * x - a^2 > 0) ∧ (∀ x > a / 3, 3 * x^2 + 2 * a * x - a^2 > 0) ∧ 
      (∀ x, -a < x ∧ x < a / 3 → 3 * x^2 + 2 * a * x - a^2 < 0) )
   else if a < 0 then
    ( (∀ x < a / 3, 3 * x^2 + 2 * a * x - a^2 > 0) ∧ (∀ x > -a, 3 * x^2 + 2 * a * x - a^2 > 0) ∧ 
      (∀ x, a / 3 < x ∧ x < -a → 3 * x^2 + 2 * a * x - a^2 < 0) )
   else
    (∀ x : ℝ, 3 * x^2 > 0)) :=
by sorry

end extreme_value_at_one_monotonicity_of_f_l117_117668


namespace dave_used_tickets_for_toys_l117_117195

-- Define the given conditions
def number_of_tickets_won : ℕ := 18
def tickets_more_for_clothes : ℕ := 10

-- Define the main conjecture
theorem dave_used_tickets_for_toys (T : ℕ) : T + (T + tickets_more_for_clothes) = number_of_tickets_won → T = 4 :=
by {
  -- We'll need the proof here, but it's not required for the statement purpose.
  sorry
}

end dave_used_tickets_for_toys_l117_117195


namespace chords_not_mutually_bisect_l117_117562

-- Definitions of chords, points and midpoints in a circle.
variables {O A B C D M : Point}
variables (circle : Circle) (chord1 : line_segment A B) (chord2 : line_segment C D)

-- Conditions that AB and CD are chords of a circle with O as the center, and M is the intersection.
def chord_of_circle (circle : Circle) (P Q : Point) : Prop :=
  ∃ (O : Point), Circle O P Q = circle

def midpoint (P Q M : Point) : Prop :=
  distance P M = distance M Q

-- Given 1: AB and CD are chords of the circle and intersect at M.
axiom AB_is_chord : chord_of_circle circle A B
axiom CD_is_chord : chord_of_circle circle C D

-- Given 2: Neither AB nor CD is a diameter.
axiom AB_not_diameter : ¬ diameter_circle A B
axiom CD_not_diameter : ¬ diameter_circle C D

-- Claim: It is not possible for AB and CD to mutually bisect each other at their intersection point M.
theorem chords_not_mutually_bisect :
  midpoint A B M ∧ midpoint C D M → False :=
begin
  sorry
end

end chords_not_mutually_bisect_l117_117562


namespace coins_stacks_final_state_l117_117775

theorem coins_stacks_final_state :
  ∃ (seq : List (Fin 2 × Fin 5)), 
  let step (m : Fin 2) (k : Fin 5) (S : Fin 6 → ℕ) : Fin 6 → ℕ :=
    match m, k, S k with
    | 0, k, n => fun i => if i = k then S i - 1 else if i = k + 1 then S (k + 1) + 2 else S i
    | 1, k, n => fun i => if i = k then S i - 1 else if i = k + 1 then S (k + 2) else if i = k + 2 then S (k + 1) else S i
    | _, _, _ => id
  ∃ (S : ℕ → (Fin 6 → ℕ)),
  S 0 = (fun _ => 1) \and 
  (∀ t, t < seq.length → S (t + 1) = step (seq.nth_le t (by linarith)).1 (seq.nth_le t (by linarith)).2 (S t)) ∧ 
  S seq.length = fun j => if j = 5 then 2010^(2010^(2010 : ℕ)) else 0 :=
sorry

end coins_stacks_final_state_l117_117775


namespace brad_age_proof_l117_117374

theorem brad_age_proof :
  ∀ (Shara_age Jaymee_age Average_age Brad_age : ℕ),
  Jaymee_age = 2 * Shara_age + 2 →
  Average_age = (Shara_age + Jaymee_age) / 2 →
  Brad_age = Average_age - 3 →
  Shara_age = 10 →
  Brad_age = 13 :=
by
  intros Shara_age Jaymee_age Average_age Brad_age
  intro h1 h2 h3 h4
  sorry

end brad_age_proof_l117_117374


namespace hydrogen_atoms_in_compound_l117_117513

theorem hydrogen_atoms_in_compound (C H : ℕ) (total_mass mass_C mass_H : ℕ) 
  (h1 : C = 6) 
  (h2 : mass_C = 12) 
  (h3 : total_mass = 78) 
  (h4 : mass_H = 1) 
  : H = (total_mass - C * mass_C) / mass_H := by
  sorry

example : hydrogen_atoms_in_compound 6 6 78 12 1 :=
by
  rfl

end hydrogen_atoms_in_compound_l117_117513


namespace correlation_problem_l117_117490

-- Definitions for the problem context
def is_deterministic (X Y : Type) : Prop := sorry -- Placeholder for deterministic relationship definition
def is_correlated (X Y : Type) : Prop := sorry  -- Placeholder for correlation definition

-- Specific pairs as types representing the provided options
def TaxiFareDistance : Type := sorry
def HouseAreaPrice : Type := sorry
def HeightWeight : Type := sorry
def IronBlockVolumeMass : Type := sorry

-- Statement of the problem
theorem correlation_problem :
  (¬ is_correlated TaxiFareDistance ∧ is_deterministic TaxiFareDistance) ∧
  (¬ is_correlated HouseAreaPrice ∧ is_deterministic HouseAreaPrice) ∧
  (is_correlated HeightWeight) ∧
  (¬ is_correlated IronBlockVolumeMass ∧ is_deterministic IronBlockVolumeMass) :=
begin
  sorry
end

end correlation_problem_l117_117490


namespace angle_OAB_regular_polygon_l117_117913

theorem angle_OAB_regular_polygon {n : ℕ} (h : 3 ≤ n) :
  let O := (0 : ℝ × ℝ),
      A := (1 : ℝ × ℝ),
      B := (complex.exp (2 * real.pi * complex.I / n) : ℝ × ℝ) in
  ∠ O A B = 360 / (2 * n) :=
by
  sorry

end angle_OAB_regular_polygon_l117_117913


namespace remainder_3_pow_1000_mod_7_l117_117215

theorem remainder_3_pow_1000_mod_7 : 3 ^ 1000 % 7 = 4 := by
  sorry

end remainder_3_pow_1000_mod_7_l117_117215


namespace neg_pow_eq_pow_four_l117_117867

variable (a : ℝ)

theorem neg_pow_eq_pow_four (a : ℝ) : (-a)^4 = a^4 :=
sorry

end neg_pow_eq_pow_four_l117_117867


namespace calculate_expression_eq_two_l117_117205

theorem calculate_expression_eq_two : (-3)^2 * (3⁻¹) + (-5 + 2) + |(-2)| = 2 := by
  sorry

end calculate_expression_eq_two_l117_117205


namespace find_ab_l117_117963

theorem find_ab (a b : ℕ) (h : (Real.sqrt 30 - Real.sqrt 18) * (3 * Real.sqrt a + Real.sqrt b) = 12) : a = 2 ∧ b = 30 :=
sorry

end find_ab_l117_117963


namespace maximum_surface_area_l117_117633

theorem maximum_surface_area (a b c : ℕ) (h1 : a = 8) (h2 : b = 8) (h3 : c = 9) : 
  let S := 2 * (a * b + b * c + a * c) in
  S = 416 := by
  sorry

end maximum_surface_area_l117_117633


namespace optical_power_and_distance_correct_l117_117171

noncomputable def optical_power_and_distance (x y : ℝ) (hx : x = 0.1) (hy : y = 0.05) :
  ℝ × ℝ :=
let d := 0.1 in  -- distance of light source from lens in meters
let f' := 0.05 in  -- distance of image from lens in meters
let D := (1 / d) - (1 / f') in
let s := real.sqrt ((x - y)^2 + (d - f')^2) in
(D, s)

theorem optical_power_and_distance_correct :
  optical_power_and_distance 0.1 0.05 (by norm_num) (by norm_num) = (-10, real.sqrt 50) :=
by {
  have hx : 0.1 = 10 * 0.01 := by norm_num,
  have hy : 0.05 = 5 * 0.01 := by norm_num,
  simp [optical_power_and_distance, hx, hy],
  norm_num
}

end optical_power_and_distance_correct_l117_117171


namespace find_a2014_l117_117452

def sequence (a : ℕ → ℚ) : Prop :=
∀ n, if (0 ≤ a n ∧ a n ≤ 1 / 2) then a (n + 1) = 2 * a n
     else if (1 / 2 < a n ∧ a n ≤ 1) then a (n + 1) = 2 * a n - 1
     else false

theorem find_a2014 (a : ℕ → ℚ) (h : sequence a) (h_start : a 1 = 3 / 5) : 
  a 2014 = 1 / 5 :=
sorry

end find_a2014_l117_117452


namespace locus_of_tangent_intersection_l117_117291

theorem locus_of_tangent_intersection 
  (x y: ℝ)
  (M N : ℝ × ℝ)
  (hx : (M.1 - 0)^2 - (M.2 / 2)^2 = 1)
  (hy : (M.1, M.2) ≠ (N.1, N.2))
  (tangent_circle : M.1^2 + M.2^2 = 4)
  (curve_c : N.1^2 - (N.2 / 4)^2 = 1)
  (tangent_hyperbola : 4 * M.1 * N.1 - M.2 * N.2 = 4)
  (x_positive : 0 < M.1) 
  (circle_radius : √(16 * N.1^2 + N.2^2) = 4)
  (h_domain : 16 * x^2 + y^2 = 4)
  (h_bound : √5 / 5 < x ∧ x ≤ 1 / 2) :
  16 * x^2 + y^2 = 4 ∧ (√5 / 5 < x ∧ x ≤ 1 / 2) :=
  sorry

end locus_of_tangent_intersection_l117_117291


namespace prism_lateral_edge_and_lateral_face_parallel_l117_117823

def Line := ℝ → ℝ → Prop -- Hypothetical definition of a line (for simplicity)
def Plane := ℝ → ℝ → ℝ → Prop -- Hypothetical definition of a plane (for simplicity)

-- Assume prism is represented by a structure
structure Prism where
  lateral_edges : List Line
  lateral_faces : List Plane

-- Condition: lines containing all lateral edges of the prism are parallel
def all_lateral_edges_parallel (prism : Prism) : Prop :=
  ∀ (e1 e2 : Line), e1 ∈ prism.lateral_edges → e2 ∈ prism.lateral_edges → parallel e1 e2

-- Problem statement: Prove the positional relationship
theorem prism_lateral_edge_and_lateral_face_parallel (prism : Prism)
  (h : all_lateral_edges_parallel prism)
  (e : Line)
  (p : Plane) :
  e ∈ prism.lateral_edges →
  p ∈ prism.lateral_faces →
  ¬ (∃ t, p t e e ∧ e ∈ prism.lateral_edges) →
  parallel e p :=
sorry

end prism_lateral_edge_and_lateral_face_parallel_l117_117823


namespace man_has_2_nickels_l117_117908

theorem man_has_2_nickels
  (d n : ℕ)
  (h1 : 10 * d + 5 * n = 70)
  (h2 : d + n = 8) :
  n = 2 := 
by
  -- omit the proof
  sorry

end man_has_2_nickels_l117_117908


namespace max_non_overlapping_crosses_l117_117486

-- Definitions based on conditions:
def cross (boardsize : ℕ × ℕ) (crosssize : ℕ × ℕ) (total_cross_cells : ℕ) : Prop :=
  (boardsize = (10, 11)) ∧ (crosssize = (3, 3)) ∧ (total_cross_cells = 5)

def within_boundary (boardsize : ℕ × ℕ) (sub_board_size : ℕ × ℕ) : Prop :=
  boardsize = (10, 11) → sub_board_size = (8, 9)

-- The main theorem statement:
theorem max_non_overlapping_crosses :
  ∀ (boardsize : ℕ × ℕ) (cross_cells : ℕ),
  cross boardsize (3, 3) cross_cells ∧
  within_boundary (10, 11) (8, 9) →
  cross_cells * 15 ≤ boardsize.1 * boardsize.2  :=
begin
  sorry
end

end max_non_overlapping_crosses_l117_117486


namespace clock_angle_at_3_15_l117_117933

-- Definitions of the parameters given in the problem.
def hour_angle (h : ℕ) (m : ℕ) : ℝ :=
  (h % 12) * 30 + (m / 60.0) * 30

def minute_angle (m : ℕ) : ℝ :=
  m * 6

-- Statement of the problem
theorem clock_angle_at_3_15 :
  let h := 3
  let m := 15
  hour_angle h m = 97.5 → 
  minute_angle m = 90 →
  abs (hour_angle h m - minute_angle m) = 7.5 :=
by
  intros h_eq m_eq
  simp [hour_angle, minute_angle] at *
  sorry

end clock_angle_at_3_15_l117_117933


namespace opposite_of_neg_2023_l117_117078

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117078


namespace inequality_proof_l117_117378

noncomputable def triangle_incircle (A B C P Q R : Point) :=
  touches_incircle A B P ∧ touches_incircle B C Q ∧ touches_incircle C A R

theorem inequality_proof (A B C P Q R : Point) (h : triangle_incircle A B C P Q R) :
  let BC := distance B C
  let CA := distance C A
  let AB := distance A B
  let PQ := distance P Q
  let QR := distance Q R
  let RP := distance R P
  in
  (BC / PQ) + (CA / QR) + (AB / RP) ≥ 6 :=
by
  sorry

end inequality_proof_l117_117378


namespace length_of_second_platform_l117_117921

theorem length_of_second_platform
  (length_train : ℕ := 100) -- The length of the train is 100 meters
  (time_first_platform : ℕ := 15) -- The train takes 15 seconds to cross the first platform
  (length_first_platform : ℕ := 200) -- The length of the first platform is 200 meters
  (time_second_platform : ℕ := 20) -- The train takes 20 seconds to cross the second platform
  : Σ (L : ℕ), L = 300 :=  -- The length of the second platform is 300 meters
by
  let speed := (length_train + length_first_platform) / time_first_platform
  let L := speed * time_second_platform - length_train
  exact ⟨L, sorry⟩

end length_of_second_platform_l117_117921


namespace winning_callers_prizes_l117_117577

theorem winning_callers_prizes :
  let n := 1125 in
  let eligible_callers (x : ℕ) := (8 * x) / 9 in
  let LCM := nat.lcm 100 (nat.lcm 40 250) in
  LCM = 1000 →
  eligible_callers n = 1000 ∧ n ≤ 2000 := sorry

end winning_callers_prizes_l117_117577


namespace negation_of_prop_equiv_l117_117410

-- Define the proposition
def prop (x : ℝ) : Prop := x^2 + 1 > 0

-- State the theorem that negation of proposition forall x, prop x is equivalent to exists x, ¬ prop x
theorem negation_of_prop_equiv :
  ¬ (∀ x : ℝ, prop x) ↔ ∃ x : ℝ, ¬ prop x :=
by
  sorry

end negation_of_prop_equiv_l117_117410


namespace find_b_find_sinB_l117_117343

variable {α : Type*} [LinearOrderedField α]

-- Given conditions
def A : α := 60 * (Real.pi / 180) -- 60 degrees in radians
def area (b c : α) : α := (b * c * (Real.sqrt 3) / 2) / 2

-- Proof 1: Find b given conditions A = 60°, 3b = 2c, Area = 3√3/2
theorem find_b (B : α) (C : α) (area_BC : α) (hA : A = 60 * (Real.pi / 180)) (h3b2c : 3 * B = 2 * C) (h_area : area B C = 3 * Real.sqrt 3 / 2) :
  B = 2 :=
sorry

-- Proof 2: Find sin B given b = 2, c = 3, A = 60°
theorem find_sinB (B C : α) (a : α) (sin_B : α) (hB : B = 2) (hC : C = 3) (hA60 : A = 60 * (Real.pi / 180)) (ha : a = Real.sqrt 7) :
  sin_B = Real.sqrt 21 / 7 :=
sorry

end find_b_find_sinB_l117_117343


namespace bricks_required_to_pave_courtyard_l117_117153

theorem bricks_required_to_pave_courtyard :
  let courtyard_length : ℝ := 25
  let courtyard_width : ℝ := 16
  let brick_length : ℝ := 0.20
  let brick_width : ℝ := 0.10
  let area_courtyard := courtyard_length * courtyard_width
  let area_brick := brick_length * brick_width
  let number_of_bricks := area_courtyard / area_brick
  number_of_bricks = 20000 := by
    let courtyard_length : ℝ := 25
    let courtyard_width : ℝ := 16
    let brick_length : ℝ := 0.20
    let brick_width : ℝ := 0.10
    let area_courtyard := courtyard_length * courtyard_width
    let area_brick := brick_length * brick_width
    let number_of_bricks := area_courtyard / area_brick
    sorry

end bricks_required_to_pave_courtyard_l117_117153


namespace perimeter_of_plot_is_340_l117_117443

def width : ℝ := 80 -- Derived width from the given conditions
def length (w : ℝ) : ℝ := w + 10 -- Length is 10 meters more than width
def perimeter (w : ℝ) : ℝ := 2 * (w + length w) -- Perimeter of the rectangle
def cost_per_meter : ℝ := 6.5 -- Cost rate per meter
def total_cost : ℝ := 2210 -- Total cost given

theorem perimeter_of_plot_is_340 :
  cost_per_meter * perimeter width = total_cost → perimeter width = 340 := 
by
  sorry

end perimeter_of_plot_is_340_l117_117443


namespace points_on_line_l117_117866

theorem points_on_line (x y : ℝ) (h : x + y = 0) : y = -x :=
by
  sorry

end points_on_line_l117_117866


namespace opposite_of_negative_2023_l117_117049

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117049


namespace gcd_condition_l117_117229

open Nat

theorem gcd_condition (n m : ℕ) (k l : ℕ) (h_n_pos : n > 0) (h_m_pos : m > 0)
  (hk_pos : k > 0) (hl_pos : l > 0) :
  ((gcd ((n + 1) ^ m - n) ((n + 1) ^ (m + 3) - n)) > 1) ↔
  (∃ k l : ℕ, n = 7 * k - 6 ∧ m = 3 * l ∧ k > 0 ∧ l > 0) :=
sorry

end gcd_condition_l117_117229


namespace mixture_volume_l117_117465

noncomputable def weight_ratio (k weight_a weight_b : ℕ) (Va ← Vliters : ℕ) :=
  weight_a * Va + weight_b * Vb = k

theorem mixture_volume
  (weight_a weight_b : ℕ)
  (ratio_a_b : ℚ)
  (total_weight : ℕ)
  (ratio_condition : ratio_a_b = 3 / 2)
  (weight_condition_a : weight_a = 900)
  (weight_condition_b : weight_b = 750)
  (total_weight_condition : total_weight = 3360)
  : (volume : ℚ) :=
begin
  sorry
end

end mixture_volume_l117_117465


namespace work_rate_l117_117128

theorem work_rate (x : ℝ) (h : (1 / x + 1 / 15 = 1 / 6)) : x = 10 :=
sorry

end work_rate_l117_117128


namespace log_seq_arithmetic_T_n_sum_allowed_m_values_l117_117748

-- Definitions and conditions
def a_n (n : ℕ) : ℕ := 
  match n with
  | 0 => 10
  | n+1 => 9 * S_n n + 10

def S_n (n : ℕ) : ℕ := (list.range (n+1)).sum (λ k, a_n k)

-- Part (Ⅰ)
theorem log_seq_arithmetic : ∀ n: ℕ, log(a_n (n+1)) - log(a_n n) = 1 :=
sorry

-- Part (Ⅱ)
def T_n (n : ℕ) : ℝ := (list.range (n+1)).sum (λ k, 3 / (log(a_n k) * log(a_n (k+1))))

theorem T_n_sum : ∀ n: ℕ, T_n n = 3 - 3 / (n+1) :=
sorry

-- Part (Ⅲ)
def condition (m : ℤ) : Prop := ∀ n : ℕ, T_n n > (1/4) * (m^2 - 5*m)

theorem allowed_m_values : ∀ n: ℕ, ∀ m : ℤ, m = 0 ∨ m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 →
  condition m :=
sorry

end log_seq_arithmetic_T_n_sum_allowed_m_values_l117_117748


namespace sum_of_numbers_l117_117089

theorem sum_of_numbers (a b c : ℝ) 
  (h₁ : a^2 + b^2 + c^2 = 62) 
  (h₂ : ab + bc + ca = 131) : 
  a + b + c = 18 :=
sorry

end sum_of_numbers_l117_117089


namespace simplify_and_evaluate_expr1_simplify_and_evaluate_expr2_l117_117142

-- Statement for the first problem
theorem simplify_and_evaluate_expr1 (x y : ℝ) (h1 : x = -3) (h2 : y = -2) :
  3 * x^2 * y - (2 * x * y^2 - 2 * (x * y - 1.5 * x^2 * y) + x * y) + 3 * x * y^2 = xy^2 + xy ∧
  (3 * x^2 * y - (2 * x * y^2 - 2 * (x * y - 1.5 * x^2 * y) + x * y) + 3 * x * y^2).subst(sub2:xy^2xy) = -6 :=
by
  sorry

-- Statement for the second problem
theorem simplify_and_evaluate_expr2 (x y : ℝ) (h1 : x = -3) (h2 : y = 2) :
  (2 * x + 3 * y) - 4 * y - (3 * x - 2 * y) = -x + y ∧
  ((2 * x + 3 * y) - 4 * y - (3 * x - 2 * y)).subst(subs:negxy) = 5 :=
by
  sorry

end simplify_and_evaluate_expr1_simplify_and_evaluate_expr2_l117_117142


namespace derivative_half_l117_117696

noncomputable def f (x : ℝ) : ℝ := Real.log (x - (deriv f 1) * x^2) + 5 * x - 4
noncomputable def f' (x : ℝ) : ℝ := (deriv f x)

theorem derivative_half : f' (1 / 2) = 5 := by
  sorry

end derivative_half_l117_117696


namespace solve_quadratics_and_sum_l117_117437

theorem solve_quadratics_and_sum (d e f : ℤ) 
  (h1 : ∃ d e : ℤ, d + e = 19 ∧ d * e = 88) 
  (h2 : ∃ e f : ℤ, e + f = 23 ∧ e * f = 120) : 
  d + e + f = 31 := by
  sorry

end solve_quadratics_and_sum_l117_117437


namespace conjugate_quadrant_l117_117329

theorem conjugate_quadrant (z : ℂ) (h : Complex.I * z = -1 + Complex.I) : 
  z.conj.im < 0 ∧ z.conj.re > 0 :=
begin
  -- Proof to be provided
  sorry
end

end conjugate_quadrant_l117_117329


namespace second_player_wins_bishops_game_l117_117139

-- Definition of a valid bishop placement on a chessboard.
def is_valid_bishop_move (board : matrix (fin 8) (fin 8) bool) (row col : fin 8) : Prop :=
  ¬ board row col

-- Axial symmetry function for a bishop on an 8x8 board.
def axial_symmetry (row col : fin 8) : (fin 8 × fin 8) :=
  (col, row)

-- Main theorem stating that the second player has a winning strategy.
theorem second_player_wins_bishops_game
  (initial_board : matrix (fin 8) (fin 8) bool)
  (move_strategy : ∀ (board : matrix (fin 8) (fin 8) bool) (row col : fin 8), (fin 8 × fin 8)) :
  ∀ board, ∃ (win_strategy : ∀ (board : matrix (fin 8) (fin 8) bool) (row col : fin 8), bool),
  ∃ row col, win_strategy board row col :=
by
  sorry

end second_player_wins_bishops_game_l117_117139


namespace total_weight_of_remaining_macaroons_l117_117731

def total_weight_remaining_macaroons (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (bags : ℕ) (bags_eaten : ℕ) : ℕ :=
  let macaroons_per_bag := total_macaroons / bags
  let remaining_macaroons := total_macaroons - macaroons_per_bag * bags_eaten
  remaining_macaroons * weight_per_macaroon

theorem total_weight_of_remaining_macaroons
  (total_macaroons : ℕ)
  (weight_per_macaroon : ℕ)
  (bags : ℕ)
  (bags_eaten : ℕ)
  (h1 : total_macaroons = 12)
  (h2 : weight_per_macaroon = 5)
  (h3 : bags = 4)
  (h4 : bags_eaten = 1)
  : total_weight_remaining_macaroons total_macaroons weight_per_macaroon bags bags_eaten = 45 := by
  sorry

end total_weight_of_remaining_macaroons_l117_117731


namespace carina_larger_package_size_l117_117207

theorem carina_larger_package_size :
  ∃ x : ℕ, (7 * x + 45 = 115) ∧ (x = 10) :=
by
  exists 10
  split
  · norm_num
  · refl

end carina_larger_package_size_l117_117207


namespace blue_notebook_cost_l117_117403

theorem blue_notebook_cost
  (total_spent : ℕ)
  (total_notebooks : ℕ)
  (red_notebooks : ℕ)
  (red_notebook_cost : ℕ)
  (green_notebooks : ℕ)
  (green_notebook_cost : ℕ)
  (blue_notebook_cost : ℕ)
  (h₀ : total_spent = 37)
  (h₁ : total_notebooks = 12)
  (h₂ : red_notebooks = 3)
  (h₃ : red_notebook_cost = 4)
  (h₄ : green_notebooks = 2)
  (h₅ : green_notebook_cost = 2)
  (h₆ : total_spent = red_notebooks * red_notebook_cost + green_notebooks * green_notebook_cost + blue_notebook_cost * (total_notebooks - red_notebooks - green_notebooks)) :
  blue_notebook_cost = 3 := by
  sorry

end blue_notebook_cost_l117_117403


namespace total_votes_l117_117129

-- Define the given conditions
def candidate_votes (V : ℝ) : ℝ := 0.35 * V
def rival_votes (V : ℝ) : ℝ := 0.35 * V + 1800

-- Prove the total number of votes cast
theorem total_votes (V : ℝ) (h : candidate_votes V + rival_votes V = V) : V = 6000 :=
by
  sorry

end total_votes_l117_117129


namespace sum_odd_numbers_l117_117494

theorem sum_odd_numbers :
  1 + 2 + 3 + ... + n = n * (n + 1) →
  3 * (1 + 3 + 5 + ... + 79) = 4800 := by
  intro h
  sorry

end sum_odd_numbers_l117_117494


namespace sum_geometric_natural_proof_l117_117993

noncomputable def sum_geometric_natural (a : ℕ) (n : ℕ) : ℕ :=
  if a = 1 then n + 1
  else (1 - a^(n + 1)) / (1 - a)

theorem sum_geometric_natural_proof (a n : ℕ) (ha : a ≠ 0) (hn : n ≠ 0) : 
  1 + a + a^2 + ... + a^n = if a = 1 then n + 1 else (1 - a^(n + 1)) / (1 - a) :=
sorry

end sum_geometric_natural_proof_l117_117993


namespace product_multiple_of_12_probability_l117_117700

open Finset

theorem product_multiple_of_12_probability :
  let S : Finset ℕ := {2, 3, 6, 9}
  let pairs := S.powerset.filter (λ t => t.card = 2)
  let valid_pairs := pairs.filter (λ t => (t.val.prod) % 12 = 0)
  (valid_pairs.card / pairs.card : ℚ) = 1 / 6 :=
by
  let S := {2, 3, 6, 9} : Finset ℕ
  let pairs := S.powerset.filter (λ t => t.card = 2)
  let valid_pairs := pairs.filter (λ t => (t.val.prod) % 12 = 0)
  have total_pairs := pairs.card
  have valid_count := valid_pairs.card
  have prob := valid_count / total_pairs : ℚ
  have expected_prob := 1 / 6
  show prob = expected_prob
  sorry

end product_multiple_of_12_probability_l117_117700


namespace first_term_of_geometric_progression_l117_117090

theorem first_term_of_geometric_progression (a r : ℝ) (S : ℝ) (sum_two_terms : ℝ): 
(S = 5) ∧ (sum_two_terms = 7 / 2) ∧ (a = 5 * (1 - r)) ∧ (a + a * r = sum_two_terms) ∧ (a^2 = 25 * (1 - r^2)^2): 
a = 5 * (1 - sqrt (3 / 10)) ∨ a = 5 * (1 + sqrt (3 / 10)) :=
sorry

end first_term_of_geometric_progression_l117_117090


namespace find_x_l117_117499

theorem find_x (x y: ℤ) (h1: x + 2 * y = 12) (h2: y = 3) : x = 6 := by
  sorry

end find_x_l117_117499


namespace history_book_cost_l117_117117

theorem history_book_cost (
  total_books : ℕ,
  math_books : ℕ,
  total_cost : ℕ,
  math_book_cost : ℕ,
  math_books_bought : ℕ
) : total_books = 80 →
    math_book_cost = 4 →
    total_cost = 368 →
    math_books_bought = 32 →
    let history_books := total_books - math_books_bought in
    let total_math_cost := math_books_bought * math_book_cost in
    let history_book_cost := (total_cost - total_math_cost) / history_books in
    history_book_cost = 5 :=
by
  intros h_total_books h_math_book_cost h_total_cost h_math_books_bought
  let history_books := total_books - math_books_bought
  let total_math_cost := math_books_bought * math_book_cost
  let history_book_cost := (total_cost - total_math_cost) / history_books
  have h_books_eq: total_books - math_books_bought = 48 := by 
    rw [h_total_books, h_math_books_bought]
    simp
  have h_math_cost_eq: math_books_bought * math_book_cost = 128 := by
    rw [h_math_books_bought, h_math_book_cost]
    simp
  have h_history_cost_eq: total_cost - total_math_cost = 240 := by
    rw [h_total_cost, h_math_cost_eq]
    simp
  have h_history_book_cost_eq: 240 / 48 = 5 := by
    rw h_history_cost_eq
    simp
  rw [h_history_book_cost_eq]
  exact h_history_book_cost_eq

end history_book_cost_l117_117117


namespace find_quantities_of_raib_ornaments_and_pendants_l117_117580

theorem find_quantities_of_raib_ornaments_and_pendants (x y : ℕ)
  (h1 : x + y = 90)
  (h2 : 40 * x + 25 * y = 2850) :
  x = 40 ∧ y = 50 :=
sorry

end find_quantities_of_raib_ornaments_and_pendants_l117_117580


namespace tangent_line_at_one_minimum_a_range_of_a_l117_117635

-- Definitions for the given functions
def g (a x : ℝ) := a * x^2 - (a + 2) * x
noncomputable def h (x : ℝ) := Real.log x
noncomputable def f (a x : ℝ) := g a x + h x

-- Part (1): Prove the tangent line equation at x = 1 for a = 1
theorem tangent_line_at_one (x y : ℝ) (h_x : x = 1) (h_a : 1 = (1 : ℝ)) :
  x + y + 1 = 0 := by
  sorry

-- Part (2): Prove the minimum value of a given certain conditions
theorem minimum_a (a : ℝ) (h_a_pos : 0 < a) (h_x : 1 ≤ x ∧ x ≤ Real.exp 1)
  (h_fmin : ∀ x, f a x ≥ -2) : 
  a = 1 := by
  sorry

-- Part (3): Prove the range of values for a given a condition
theorem range_of_a (a x₁ x₂ : ℝ) (h_x : 0 < x₁ ∧ x₁ < x₂) 
  (h_f : ∀ x₁ x₂, (f a x₁ - f a x₂) / (x₁ - x₂) > -2) :
  0 ≤ a ∧ a ≤ 8 := by
  sorry

end tangent_line_at_one_minimum_a_range_of_a_l117_117635


namespace Carlos_books_read_in_July_l117_117563

theorem Carlos_books_read_in_July :
  ∀ (books_in_June books_in_August books_goal books_in_July : ℕ),
    books_in_June = 42 →
    books_in_August = 30 →
    books_goal = 100 →
    books_goal - (books_in_June + books_in_August) = books_in_July →
    books_in_July = 28 :=
by
  intros books_in_June books_in_August books_goal books_in_July h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4
  done

end Carlos_books_read_in_July_l117_117563


namespace inequality_a_inequality_b_l117_117138

theorem inequality_a (R_A R_B R_C R_D d_A d_B d_C d_D : ℝ) :
  (R_A + R_B + R_C + R_D) * (1 / d_A + 1 / d_B + 1 / d_C + 1 / d_D) ≥ 48 :=
sorry

theorem inequality_b (R_A R_B R_C R_D d_A d_B d_C d_D : ℝ) :
  (R_A^2 + R_B^2 + R_C^2 + R_D^2) * (1 / d_A^2 + 1 / d_B^2 + 1 / d_C^2 + 1 / d_D^2) ≥ 144 :=
sorry

end inequality_a_inequality_b_l117_117138


namespace gcd_10010_15015_l117_117233

def a := 10010
def b := 15015

theorem gcd_10010_15015 : Nat.gcd a b = 5005 := by
  sorry

end gcd_10010_15015_l117_117233


namespace xyz_squared_l117_117624

theorem xyz_squared (x y z p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
  (hxy : x + y = p) (hyz : y + z = q) (hzx : z + x = r) :
  x^2 + y^2 + z^2 = (p^2 + q^2 + r^2 - p * q - q * r - r * p) / 2 :=
by
  sorry

end xyz_squared_l117_117624


namespace compound_interest_time_period_l117_117118

theorem compound_interest_time_period (P r I : ℝ) (n A t : ℝ) 
(hP : P = 6000) 
(hr : r = 0.10) 
(hI : I = 1260.000000000001) 
(hn : n = 1)
(hA : A = P + I)
(ht_eqn: (A / P) = (1 + r / n) ^ t) :
t = 2 := 
by sorry

end compound_interest_time_period_l117_117118


namespace Liz_team_deficit_l117_117393

theorem Liz_team_deficit :
  ∀ (initial_deficit liz_free_throws liz_three_pointers liz_jump_shots opponent_points : ℕ),
    initial_deficit = 20 →
    liz_free_throws = 5 →
    liz_three_pointers = 3 →
    liz_jump_shshots = 4 →
    opponent_points = 10 →
    (initial_deficit - (liz_free_throws * 1 + liz_three_pointers * 3 + liz_jump_shshots * 2 - opponent_points)) = 8 := by
  intros initial_deficit liz_free_throws liz_three_pointers liz_jump_shots opponent_points
  intros h_initial_deficit h_liz_free_throws h_liz_three_pointers h_liz_jump_shots h_opponent_points
  sorry

end Liz_team_deficit_l117_117393


namespace roots_poly_l117_117380

theorem roots_poly (u p q r : ℝ) (Q : ℂ → ℂ)
  (hQ : Q = λ z, z^3 + p*z^2 + q*z + r)
  (hroots : ∃ (u : ℂ), Q(u - 2*complex.I) = 0 ∧ Q(2*u + 4*complex.I) = 0 ∧ Q(3*u + 6) = 0) :
  p + q + r = -24 :=
sorry

end roots_poly_l117_117380


namespace cubes_with_even_blue_faces_l117_117538

theorem cubes_with_even_blue_faces :
  let length := 6
  let width := 4
  let height := 2
  let total_cubes := length * width * height
  let corner_cubes := 8
  let edge_cubes := 4 * (length - 2) + 4 * (width - 2) + 4 * (height - 2)
  let face_cubes := 2 * ((length - 2) * (width - 2)) + 2 * ((length - 2) * (height - 2)) + 2 * ((width - 2) * (height - 2))
  let inside_cubes := total_cubes - corner_cubes - edge_cubes - face_cubes
  total_cubes == 48 ∧
  corner_cubes == 8 ∧
  edge_cubes == 24 ∧
  face_cubes == 8 ∧
  inside_cubes == 8 →
  let even_blue_faces_cubes := edge_cubes + inside_cubes
  even_blue_faces_cubes = 32 := 
begin
  intros,
  have total_cubes_correct: total_cubes == 48,
  { exact total_cubes == 48, },
  have corner_cubes_correct: corner_cubes == 8,
  { exact corner_cubes == 8, },
  have edge_cubes_correct: edge_cubes == 24,
  { sorry },
  have face_cubes_correct: face_cubes == 8,
  { sorry },
  have inside_cubes_correct: inside_cubes == 8,
  { sorry },
  have even_blue_faces_cubes_correct: even_blue_faces_cubes == 32,
  { exact even_blue_faces_cubes == 32, },
  exact even_blue_faces_cubes_correct,
end

end cubes_with_even_blue_faces_l117_117538


namespace Bryan_has_more_skittles_l117_117883

-- Definitions for conditions
def Bryan_skittles : ℕ := 50
def Ben_mms : ℕ := 20

-- Main statement to be proven
theorem Bryan_has_more_skittles : Bryan_skittles > Ben_mms ∧ Bryan_skittles - Ben_mms = 30 :=
by
  sorry

end Bryan_has_more_skittles_l117_117883


namespace avg_distinct_k_values_l117_117293

open_locale big_operators

theorem avg_distinct_k_values :
  ∀ k : ℕ, (∃ r1 r2 : ℕ, r1 * r2 = 36 ∧ r1 + r2 = k) →
  (∑ k in ({37, 20, 15, 13, 12} : finset ℕ), k) / 5 = 19.4 :=
by { sorry }

end avg_distinct_k_values_l117_117293


namespace eel_cost_l117_117191

theorem eel_cost (J E : ℝ) (h1 : E = 9 * J) (h2 : J + E = 200) : E = 180 :=
by
  sorry

end eel_cost_l117_117191


namespace part1_part2_part3_l117_117637

open Real

noncomputable def g (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x
noncomputable def h (x : ℝ) : ℝ := log x
noncomputable def f (a x : ℝ) : ℝ := g a x + h x

theorem part1 (a : ℝ) (h_a : a = 1) : 
  let g := g a
  let g' := 2 * x - 3
  ∀ (x y : ℝ), y = g 1 → (x + y + 1 = 0) → (g 1 = g' 1 := by sorry

theorem part2 (a : ℝ) (h_positive : 0 < a) (h_fmin : ∀ x, 1 ≤ x ∧ x ≤ exp 1 → -2 ≤ f a x) : 
  a = 1 := by sorry

theorem part3 (a : ℝ) (h_ineq : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → 
  (f a x1 - f a x2) / (x1 - x2) > -2) : 
  0 ≤ a ∧ a ≤ 8 := by sorry

end part1_part2_part3_l117_117637


namespace find_fe_l117_117662

def f : ℝ → ℝ
| x => if x < 1 then Real.exp x + 1 else f (Real.log x)

theorem find_fe : f Real.e = 2 := sorry

end find_fe_l117_117662


namespace BP_value_l117_117885

-- Define the problem conditions and statement.
theorem BP_value
  (A B C D P : Point)
  (on_circle : ∀ point ∈ {A, B, C, D}, is_on_circle point)
  (intersect : P ∈ (line_through A C) ∧ P ∈ (line_through B D))
  (AP : Real := 10)
  (PC : Real := 2)
  (BD : Real := 9)
  (BP_lt_DP : ∃ x y : Real, BP = x ∧ DP = y ∧ x + y = BD ∧ x < y) :
  BP = 4 :=
by
  sorry -- Proof is omitted

end BP_value_l117_117885


namespace opposite_of_neg_2023_l117_117011

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117011


namespace john_subtraction_number_l117_117104

theorem john_subtraction_number (a b : ℕ) (h1 : a = 40) (h2 : b = 1) :
  40^2 - ((2 * 40 * 1) - 1^2) = 39^2 :=
by
  -- sorry indicates the proof is skipped
  sorry

end john_subtraction_number_l117_117104


namespace pyramid_identity_l117_117350

variable (n : ℕ) (P A1 A2 An O B : ℝ)

-- Condition that n >= 3
axiom n_ge_3 : n ≥ 3

-- Definitions of distances in the pyramid
axiom PO : ℝ
axiom PA1 : ℝ
axiom PB : ℝ

-- Definition of midpoint B of the edge A1 An
axiom midpoint_B : B = (A1 + An) / 2

-- Trigonometric identities
axiom sin_pi_n : ℝ := Real.sin (Real.pi / n)
axiom cos_pi_n : ℝ := Real.cos (Real.pi / n)

-- The theorem to be proven
theorem pyramid_identity :
  PO^2 * (sin_pi_n^2) + PA1^2 * (cos_pi_n^2) = PB^2 := 
sorry

end pyramid_identity_l117_117350


namespace total_eggs_examined_l117_117197

def trays := 7
def eggs_per_tray := 10

theorem total_eggs_examined : trays * eggs_per_tray = 70 :=
by 
  sorry

end total_eggs_examined_l117_117197


namespace distance_from_point_to_line_l117_117436

def distance_point_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

theorem distance_from_point_to_line :
  distance_point_line 1 2 3 4 (-6) = 1 := 
by
  unfold distance_point_line
  -- This part would typically contain the arithmetic steps, summed up here for brevity
  sorry

end distance_from_point_to_line_l117_117436


namespace roots_of_polynomial_eq_l117_117261

theorem roots_of_polynomial_eq (a b : ℝ) (h : a^2 + 2 * a - 5 = 0) (h' : b^2 + 2 * b - 5 = 0) :
  a^2 + a * b + 2 * a = 0 :=
by
  -- conditions based on Vieta's formulas
  have sum_roots : a + b = -2, from sorry,
  have prod_roots : a * b = -5, from sorry,
  -- simplify using given conditions
  sorry

end roots_of_polynomial_eq_l117_117261


namespace num_div_by_8_in_range_l117_117321

theorem num_div_by_8_in_range : 
  ∃ k : ℕ, k = 50 ∧ 
  ∀ n : ℕ, (100 ≤ n ∧ n ≤ 500 → n % 8 = 0) ↔ 
           (∃ m : ℕ, 100 ≤ m * 8 ∧ m * 8 ≤ 500 ∧ n = m * 8) :=
begin
  sorry
end

end num_div_by_8_in_range_l117_117321


namespace percentage_decrease_l117_117081

theorem percentage_decrease (x : ℝ) (P : ℝ) (hx₁ : P > 0) (hx₂ : x / 100 < 1) :
  let final_price := P * (1 - x / 100) * 1.10 in
  (final_price = P * 1.012) → x = 8 :=
by
  intro h_price
  have h1 : final_price = P * (1 - x / 100) * 1.10 := rfl
  rw h_price at h1
  sorry

end percentage_decrease_l117_117081


namespace sum_of_digits_A_n_l117_117252

noncomputable def A (n : ℕ) : ℕ :=
  ∏ i in range (n + 1), (10^(2^i) - 1)

def sum_digits (n : ℕ) : ℕ :=
  if n = 0 then 9
  else 9 * 2^n

theorem sum_of_digits_A_n (n : ℕ) : 
  sum_digits n = 9 * 2^n :=
by
  sorry

end sum_of_digits_A_n_l117_117252


namespace gcd_proof_l117_117248

def gcd_10010_15015 := Nat.gcd 10010 15015 = 5005

theorem gcd_proof : gcd_10010_15015 :=
by
  sorry

end gcd_proof_l117_117248


namespace money_spent_on_paintbrushes_l117_117844

-- Define the conditions
def total_spent : ℝ := 90.00
def cost_canvases : ℝ := 40.00
def cost_paints : ℝ := cost_canvases / 2
def cost_easel : ℝ := 15.00

-- Define the problem
theorem money_spent_on_paintbrushes : total_spent - (cost_canvases + cost_paints + cost_easel) = 15.00 :=
by sorry

end money_spent_on_paintbrushes_l117_117844


namespace problem1_problem2_l117_117279

-- Definitions of sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x > 5 ∨ x < -1}

-- First problem: A ∩ B
theorem problem1 (a : ℝ) (ha : a = 4) : A a ∩ B = {x | 6 < x ∧ x ≤ 7} :=
by sorry

-- Second problem: A ∪ B = B
theorem problem2 (a : ℝ) : (A a ∪ B = B) ↔ (a < -4 ∨ a > 5) :=
by sorry

end problem1_problem2_l117_117279


namespace Brad_age_l117_117371

theorem Brad_age (shara_age : ℕ) (h_shara : shara_age = 10)
  (jaymee_age : ℕ) (h_jaymee : jaymee_age = 2 * shara_age + 2)
  (brad_age : ℕ) (h_brad : brad_age = (shara_age + jaymee_age) / 2 - 3) : brad_age = 13 := by
  sorry

end Brad_age_l117_117371


namespace valid_solutions_count_l117_117687

theorem valid_solutions_count :
  let num_zeros := (Finset.range 50).filter (λ x : ℕ, x + 1)
  let common_zeros := [2, 8, 18, 32, 50].toFinset
  num_zeros.card - common_zeros.card = 45 :=
by {
  let numerator_zeros := (Finset.range 50).filter (λ x : ℕ, x + 1),
  let denominator_zeros := (Finset.range 25).map (λ k : ℕ, 2 * k^2),
  let common_zeros := numerator_zeros ∩ denominator_zeros,
  let valid_solutions := numerator_zeros \ common_zeros,
  have h1 : numerator_zeros = Finset.range 50,
  have h2 : common_zeros = [2, 8, 18, 32, 50].toFinset,
  exact Finset.card (numerator_zeros \ common_zeros) = 45,
  sorry
}

end valid_solutions_count_l117_117687


namespace percentile_75_average_l117_117645

theorem percentile_75_average (data : list ℝ) (h_len : data.length = 100) (h_75 : percentile 75 data = 9.3) :
  (nth_le data 74 (by sorry) + nth_le data 75 (by sorry)) / 2 = 9.3 :=
sorry

end percentile_75_average_l117_117645


namespace roots_opposite_sign_eq_magnitude_l117_117575

theorem roots_opposite_sign_eq_magnitude (c d e n : ℝ) (h : ((n+2) * (x^2 + c*x + d)) = (n-2) * (2*x - e)) :
  n = (-4 - 2 * c) / (c - 2) :=
by
  sorry

end roots_opposite_sign_eq_magnitude_l117_117575


namespace binomial_expansion_coefficient_l117_117286

theorem binomial_expansion_coefficient (a : ℝ) :
  (∃ a : ℝ, (binomial_theorem_coefficient ℝ 6 3 (λ k, a^k) = 5/2) ∧ a = 1/2) :=
by sorry

end binomial_expansion_coefficient_l117_117286


namespace starting_number_divisible_by_3_l117_117467

theorem starting_number_divisible_by_3 (x : ℕ) (h₁ : ∀ n, 1 ≤ n → n < 14 → ∃ k, x + (n - 1) * 3 = 3 * k ∧ x + (n - 1) * 3 ≤ 50) :
  x = 12 :=
by
  sorry

end starting_number_divisible_by_3_l117_117467


namespace opposite_of_neg_2023_l117_117033

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117033


namespace complex_min_modulus_l117_117692

theorem complex_min_modulus (z : ℂ) (h : ∃ x : ℝ, (x^2 - z*x + (4 + 3*complex.I) = 0)) :
  z = (9*sqrt 5 / 5) + (3*sqrt 5 / 5) * complex.I ∨
  z = -(9*sqrt 5 / 5) - (3*sqrt 5 / 5) * complex.I :=
sorry

end complex_min_modulus_l117_117692


namespace solve_for_N_l117_117088

theorem solve_for_N (N : ℤ) (h : 2 * N^2 + N = 12) (h_neg : N < 0) : N = -3 := 
by 
  sorry

end solve_for_N_l117_117088


namespace perpendicular_diagonals_implies_rectangle_l117_117811

-- Definitions and theorem statement
def is_perpendicular {α : Type} [InnerProductSpace ℝ α] (u v : α) : Prop := ⟪u, v⟫ = 0

def quadrilateral (α : Type) := (α × α × α × α)

def intersection_point (α : Type) := α

def perpendiculars_from_point_to_sides (α : Type) := List (α × α)

variables {α : Type} [InnerProductSpace ℝ α]

/-- Theorem statement: Given a quadrilateral with perpendicular diagonals, 
    and from the point of intersection, 52 perpendiculars are dropped
    to the sides of the quadrilateral and extended to intersect the opposite 
    sides at points \(M, N, K, L\) respectively, then quadrilateral 
    \(MNKL\) is a rectangle with sides parallel to the diagonals of the 
    given quadrilateral. -/
theorem perpendicular_diagonals_implies_rectangle 
  (quad : quadrilateral α)
  (S : intersection_point α)
  (perp_dropped : perpendiculars_from_point_to_sides α)
  (M N K L : α)
  (h_perpendicular_diagonals : is_perpendicular (quad.1 - quad.3) (quad.2 - quad.4))
  (h_perpendiculars_constructed : perp_dropped.length = 52)
  (h_intersect_points : {M, N, K, L} ⊆ (perp_dropped.map (λ x, x.2)).to_finset)
  : (is_rectangle {M, N, K, L} ∧
     are_sides_parallel_to_diagonals {M, N, K, L} (quad.1, quad.3) (quad.2, quad.4)) :=
sorry

end perpendicular_diagonals_implies_rectangle_l117_117811


namespace expand_polynomial_l117_117969

theorem expand_polynomial (x : ℝ) :
  (x^10 - 4 * x^3 + 2 * x^(-1) - 8) * (3 * x^5) = 3 * x^(15) - 12 * x^(8) + 6 * x^(4) - 24 * x^(5) :=
by
  sorry

end expand_polynomial_l117_117969


namespace opposite_of_neg_2023_l117_117013

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117013


namespace find_b_from_conditions_l117_117330

theorem find_b_from_conditions 
  (x y b : ℝ) 
  (h1 : 3 * x - 5 * y = b) 
  (h2 : x / (x + y) = 5 / 7) 
  (h3 : x - y = 3) : 
  b = 5 := 
by 
  sorry

end find_b_from_conditions_l117_117330


namespace example_100_digit_number_divisible_l117_117226

def is_100_digit_number (n : ℕ) : Prop := n >= 10^99 ∧ n < 10^100
def no_zero_digits (n : ℕ) : Prop := ∀ d ∈ digits 10 n, d ≠ 0 
def sum_of_digits (n : ℕ) : ℕ := digits 10 n |>.foldr Nat.add 0

theorem example_100_digit_number_divisible (n : ℕ) 
    (h1 : is_100_digit_number n) 
    (h2 : no_zero_digits n) 
    (h3 : n % sum_of_digits n = 0) : 
    n = 11111111111111111111111111111111111111111111111111111111111111111111111111111195125 := 
    sorry

end example_100_digit_number_divisible_l117_117226


namespace multiples_sum_squared_l117_117742

theorem multiples_sum_squared :
  let a := 4
  let b := 4
  ((a + b)^2) = 64 :=
by
  sorry

end multiples_sum_squared_l117_117742


namespace problem1_problem2_problem3_l117_117264

-- Define the condition for f(x)
def condition_f (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) - 3 * f(-x) = 12 * x - 2

-- Define the condition for g(x)
def condition_g (g : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, g(x) = a * x^2 + b * x + c ∧ ∀ x : ℝ, g(x) ≥ -16 ∧ g(-2) = -15 ∧ g(0) = -15)

-- Define the function h(x) and its minimum value F(a)
def h (f g : ℝ → ℝ) (a x : ℝ) : ℝ :=
  (2 / 3) * f(x) + g(a - x) - 2 * a

def F (f g : ℝ → ℝ) (a : ℝ) : ℝ :=
  if a ≥ 1 then a^2 - 2*a - 40/3
  else if -1 < a ∧ a < 1 then -43/3
  else a^2 + 2*a - 40/3

theorem problem1 :
  (∃ f : ℝ → ℝ, condition_f f ∧ (∀ x : ℝ, f x = 3 * x + 1)) :=
sorry

theorem problem2 :
  (∃ g : ℝ → ℝ, condition_g g ∧ (∀ x : ℝ, g x = x^2 + 2 * x - 15)) :=
sorry

theorem problem3 :
  (∀ f g : ℝ → ℝ, condition_f f → condition_g g →
  (∀ a : ℝ, ∃ x : ℝ, h f g a x = F f g a)) :=
sorry

end problem1_problem2_problem3_l117_117264


namespace bricks_required_l117_117158

def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def brick_length : ℝ := 20 / 100
def brick_width : ℝ := 10 / 100

theorem bricks_required :
  (courtyard_length * courtyard_width) / (brick_length * brick_width) = 20000 := 
    sorry

end bricks_required_l117_117158


namespace maximize_Sn_l117_117313

def a_n (n : ℕ) : ℤ := 26 - 2 * n

def S_n (n : ℕ) : ℤ := n * (26 - 2 * (n + 1)) / 2 + 26 * n

theorem maximize_Sn : (n = 12 ∨ n = 13) ↔ (∀ m : ℕ, S_n m ≤ S_n 12 ∨ S_n m ≤ S_n 13) :=
by sorry

end maximize_Sn_l117_117313


namespace mary_needs_more_apples_l117_117397

theorem mary_needs_more_apples (total_pies : ℕ) (apples_per_pie : ℕ) (harvested_apples : ℕ) (y : ℕ) :
  total_pies = 10 → apples_per_pie = 8 → harvested_apples = 50 → y = 30 :=
by
  intro h1 h2 h3
  have total_apples_needed := total_pies * apples_per_pie
  have apples_needed_to_buy := total_apples_needed - harvested_apples
  have proof_needed : apples_needed_to_buy = y := sorry
  have proof_given : y = 30 := sorry
  have apples_needed := total_pies * apples_per_pie - harvested_apples
  exact proof_given

end mary_needs_more_apples_l117_117397


namespace quadratic_decreasing_range_a_l117_117697

theorem quadratic_decreasing_range_a {a : ℝ} (h : ∀ x y : ℝ, x ≤ y → x ∈ set.Iic 5 → y ∈ set.Iic 5 → (x^2 + 2*a*x + 1) ≥ (y^2 + 2*a*y + 1)) : a ≤ -5 :=
by {
  -- proof goes here
  sorry
}

end quadratic_decreasing_range_a_l117_117697


namespace ram_weight_increase_l117_117466

/-- The weights of two friends Ram and Shyam are in the ratio of 4:5. Ram's weight increases by a certain percentage,
    and the total weight of Ram and Shyam together becomes 82.8 kg with an increase of 15%. 
    The weight of Shyam increased by 19%. Prove that Ram's weight increased by 10%. -/
theorem ram_weight_increase
  (x : ℝ)
  (hx : x = 8)
  (ram_original : ℝ := 4 * x)
  (shyam_original : ℝ := 5 * x)
  (total_original : ℝ := 9 * x)
  (h_total_weight_after : (9 * x) * 1.15 = 82.8)
  (shyam_increased : shyam_original * 1.19)
  (shyam_new : ℝ := shyam_original * 1.19)
  (total_weight_after : shyam_new + (ram_original +  (ram_original* increase_proportion )) = 82.8)
  (increase_amount_ram : ℝ := ((operation 8  increase_proportion) = 32 + (32 ∗ ( increase_proportion))))
  (increment_rpc : ↑( "  increase_proportion" as percent)) :
  (10 : quality ) :=
by sorry -- Proof is omitted

end ram_weight_increase_l117_117466


namespace opposite_of_neg2023_l117_117003

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l117_117003


namespace oranges_thrown_away_l117_117181

def numOrangesInBin (initial : ℕ) (added : ℕ) (removed : ℕ) : ℕ := initial - removed + added

theorem oranges_thrown_away (initial added total : ℕ) (removed: ℕ) (h : numOrangesInBin initial added removed = total) : removed = 25 :=
by
  have Initial : initial = 40 := rfl
  have Added : added = 21 := rfl
  have Total : total = 36 := rfl
  
  -- Sorry signals that the proof is not provided
  sorry

end oranges_thrown_away_l117_117181


namespace compare_polynomials_l117_117208

noncomputable def f (x : ℝ) : ℝ := 2*x^2 + 5*x + 3
noncomputable def g (x : ℝ) : ℝ := x^2 + 4*x + 2

theorem compare_polynomials (x : ℝ) : f x > g x :=
by sorry

end compare_polynomials_l117_117208


namespace gideon_current_age_l117_117462

theorem gideon_current_age :
  let total_marbles := 100 in
  let marbles_given_to_sister := total_marbles * 3 / 4 in
  let remaining_marbles := total_marbles - marbles_given_to_sister in
  let age_five_years_from_now := remaining_marbles * 2 in
  let current_age := age_five_years_from_now - 5 in
  current_age = 45 := 
by
  sorry

end gideon_current_age_l117_117462


namespace candies_left_correct_l117_117589

def number_of_clowns := 4
def number_of_children := 30
def initial_candies := 700
def candies_per_person := 20

def total_people := number_of_clowns + number_of_children
def total_candies_sold := total_people * candies_per_person
def candies_left := initial_candies - total_candies_sold

theorem candies_left_correct : candies_left = 20 := by
  have h1 : total_people = 4 + 30 := rfl
  have h2 : total_candies_sold = (4 + 30) * 20 := by rw [h1]
  have h3 : total_candies_sold = 680 := rfl
  have h4 : candies_left = 700 - 680 := by rw [h3]
  exact h4

end candies_left_correct_l117_117589


namespace find_b_find_a_a_is_one_final_a_l117_117665

noncomputable def f (a b x : ℝ) := (a^2 / 3) * x^3 - 2 * a * x^2 + b * x

theorem find_b (a : ℝ) (b : ℝ) : (deriv (f a b) 0 = 3) → b = 3 :=
by
  sorry

theorem find_a (a : ℝ) : (deriv (f a 3) 1 = 0) → a^2 - 4 * a + 3 = 0 :=
by
  sorry

theorem a_is_one (a : ℝ) : a^2 - 4 * a + 3 = 0 → (∀ x, deriv (f a 3) x = a^2 * x^2 - 4 * a * x + 3) → 
    ∀ x, deriv (f 1 3) x = (x - 1) * (x - 3) :=
by
  sorry

theorem final_a (a : ℝ) : (deriv (f a 3) 1 = 0) → has_local_min (deriv (f a 3)) 1 = has_local_min (deriv (f 1 3)) 1 → a = 1 :=
by
  sorry

end find_b_find_a_a_is_one_final_a_l117_117665


namespace new_person_weight_is_75_l117_117434

theorem new_person_weight_is_75 
  (avg_increase : 2.5) 
  (persons : 10) 
  (original_weight : ℝ) 
  (old_person_weight : 50) :
  let increase_total_weight := avg_increase * persons
  let new_person_weight := old_person_weight + increase_total_weight
  new_person_weight = 75 := by
sorrt

end new_person_weight_is_75_l117_117434


namespace prob_A_wins_correct_l117_117862

noncomputable def prob_A_wins : ℚ :=
  let outcomes : ℕ := 3^3
  let win_one_draw_two : ℕ := 3
  let win_two_other : ℕ := 6
  let win_all : ℕ := 1
  let total_wins : ℕ := win_one_draw_two + win_two_other + win_all
  total_wins / outcomes

theorem prob_A_wins_correct :
  prob_A_wins = 10/27 :=
by
  sorry

end prob_A_wins_correct_l117_117862


namespace katherine_has_5_bananas_l117_117335

/-- Katherine has 4 apples -/
def apples : ℕ := 4

/-- Katherine has 3 times as many pears as apples -/
def pears : ℕ := 3 * apples

/-- Katherine has a total of 21 pieces of fruit (apples + pears + bananas) -/
def total_fruit : ℕ := 21

/-- Define the number of bananas Katherine has -/
def bananas : ℕ := total_fruit - (apples + pears)

/-- Prove that Katherine has 5 bananas -/
theorem katherine_has_5_bananas : bananas = 5 := by
  sorry

end katherine_has_5_bananas_l117_117335


namespace opposite_of_negative_2023_l117_117056

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117056


namespace diana_statues_painted_l117_117872

theorem diana_statues_painted :
  let paint_remaining := (1 : ℚ) / 2
  let paint_per_statue := (1 : ℚ) / 4
  (paint_remaining / paint_per_statue) = 2 :=
by
  sorry

end diana_statues_painted_l117_117872


namespace nailcutter_sound_count_l117_117694

-- Definitions based on conditions
def nails_per_person : ℕ := 20
def number_of_customers : ℕ := 3
def sound_per_nail : ℕ := 1

-- The statement to prove 
theorem nailcutter_sound_count :
  (nails_per_person * number_of_customers * sound_per_nail) = 60 := by
  sorry

end nailcutter_sound_count_l117_117694


namespace proof_problem_l117_117893

variable {α : Type*} [LinearOrderedField α]
variable (f : α → α) (a b : α)

-- Given the conditions
def is_increasing (f : α → α) := ∀ x y, x < y → f x < f y
def is_odd (f : α → α) := ∀ x, f (-x) = -f x

-- Problem Statement
theorem proof_problem (h₁ : is_increasing f) (h₂ : is_odd f) (h₃ : a + b ≤ 0) :
  f(a) + f(b) ≤ 0 := by
  sorry

end proof_problem_l117_117893


namespace area_of_PQR_is_32_l117_117853

/-- Triangle PQR is an isosceles right triangle with angle P = 90 degrees, and the length of segment PR = 8 cm. The goal is to prove that the area of triangle PQR is 32 square centimeters. -/
theorem area_of_PQR_is_32 
  (P Q R : Type) 
  [triangle P Q R] 
  (right_angle_P : angle P = 90)
  (isosceles_right : isosceles_right_triangle P Q R P Q P R)
  (length_PR : segment_length P R = 8) 
  : area_triangle P Q R = 32 := 
sorry

end area_of_PQR_is_32_l117_117853


namespace discount_difference_l117_117918

variable {P : ℝ} -- Original price

def first_discount (P : ℝ) : ℝ := 0.60 * P
def second_discount (P : ℝ) : ℝ := 0.90 * first_discount P
def actual_discount (P : ℝ) : ℝ := P - second_discount P

def claimed_discount (P : ℝ) : ℝ := 0.50 * P

theorem discount_difference : 
  let diff := claimed_discount P - actual_discount P
  in diff / P = 0.04 :=
by
  sorry

end discount_difference_l117_117918


namespace num_type_A_cubes_internal_diagonal_l117_117176

theorem num_type_A_cubes_internal_diagonal :
  let L := 120
  let W := 350
  let H := 400
  -- Total cubes traversed calculation
  let GCD := Nat.gcd
  let total_cubes_traversed := L + W + H - (GCD L W + GCD W H + GCD H L) + GCD L (GCD W H)
  -- Type A cubes calculation
  total_cubes_traversed / 2 = 390 := by sorry

end num_type_A_cubes_internal_diagonal_l117_117176


namespace problem_I_problem_II_l117_117674

theorem problem_I (α : ℝ) (θ ρ : ℝ) :
  let x := 2 * Real.cos α,
      y := 1 + 2 * Real.sin α,
      x' := ρ * Real.cos θ,
      y' := ρ * Real.sin θ in
  x^2 + (y - 1)^2 = 4 →
  x = x' ∧ y = y' →
  ρ^2 - 2 * ρ * Real.sin θ - 3 = 0 :=
by
  intros α θ ρ x y x' y' h₁ h₂
  sorry

theorem problem_II (t : ℝ) :
  let x := 1 + t * Real.cos (Real.pi / 4),
      y := t * Real.sin (Real.pi / 4) in
  (x^2 + (y - 1)^2 = 4) →
  t = Real.sqrt 2 ∨ t = -Real.sqrt 2 →
  abs (2 * Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  intros t x y h₁ h₂
  sorry

end problem_I_problem_II_l117_117674


namespace problem_l117_117929
open Real Classical

noncomputable def triangle_area (A B C: Point) : ℝ :=
  (1 / 2) * (abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y)))

noncomputable def triangle_area_condition (A B C P K: Point) (APK_area: ℝ) (CPK_area: ℝ) : Prop :=
  let AP := dist A P
  let PK := dist P K
  let AK := dist A K
  let CK := dist C K
  APK_area = 7 ∧ CPK_area = 5

noncomputable def find_AC (A B C O P T K: Point) (angle_ABC: ℝ) : ℝ :=
  let beta := atan(3 / 4)
  let APK_area := 7
  let CPK_area := 5
  let S_AC := triangle_area A C P
  let cond := triangle_area_condition A B C P K APK_area CPK_area
  sqrt ((17 / 7) * S_AC)

theorem problem 
  (A B C O P T K: Point)
  (angle_ABC: ℝ)
  (APK_area: ℝ)
  (CPK_area: ℝ)
  (H1: angle_ABC = atan(3 / 4))
  (H2: APK_area = 7)
  (H3: CPK_area = 5)
  (H4: dist P A * dist P K * dist K A * dist K C = APK_area * CPK_area) :
  triangle_area A B C = 144 / 5 ∧ find_AC A B C O P T K angle_ABC = (4 * sqrt 17) / sqrt 7 :=
by 
  sorry

end problem_l117_117929


namespace isosceles_right_triangle_area_l117_117847

noncomputable def triangle_area (a b c : ℝ) (angle : ℝ) : ℝ :=
  if angle = 90 then 0.5 * a * b
  else sorry -- General case for arbitrary angle, to be filled in as needed.

theorem isosceles_right_triangle_area :
  ∀ (PQ PR : ℝ), (PQ = 8) → (PR = 8) → (triangle_area PQ PR 0 90) = 32 :=
by
  intros PQ PR hPQ hPR
  simp [triangle_area]
  rw [hPQ, hPR]
  norm_num
  sorry

end isosceles_right_triangle_area_l117_117847


namespace solve_equation_simplify_expression_l117_117200

-- Part 1: Solving the equation
theorem solve_equation (x : ℝ) : 9 * (x - 3) ^ 2 - 121 = 0 ↔ x = 20 / 3 ∨ x = -2 / 3 :=
by 
    sorry

-- Part 2: Simplifying the expression
theorem simplify_expression (x y : ℝ) : (x - 2 * y) * (x ^ 2 + 2 * x * y + 4 * y ^ 2) = x ^ 3 - 8 * y ^ 3 :=
by 
    sorry

end solve_equation_simplify_expression_l117_117200


namespace area_triangle_BCD_l117_117360

theorem area_triangle_BCD {A B C D : ℝ×ℝ} 
  (area_triangle_ABC : ½ * 8 * 12.5 = 50)
  (BC : (0 : ℝ) < 8)
  (CD : 8 + 32 = 40) :
  ½ * 32 * 12.5 = 200 :=
sorry

end area_triangle_BCD_l117_117360


namespace triangle_division_max_area_perimeter_eq_l117_117735

-- Define the sides of the triangle
def A := (0, 0)
def B := (13, 0)
def C := (x3, y3)
def AB := 13
def BC := 14
def CA := 15

-- Define the maximum possible area of the triangle formed by the division
def max_area : ℚ := 1323 / 26

-- Define the condition that the line passes through two sides and divides the triangle into two figures with equal perimeter
def divides_equal_perimeter (A B C : ℝ × ℝ) (line : ℝ × ℝ → Prop) : Prop :=
  ∃ D E, line D ∧ line E ∧ D ∈ segment A B ∧ E ∈ segment B C ∧
  perimeter (triangle A B D) + perimeter (quadrilateral D B C E) = perimeter (triangle A B C)

-- Statement to check the maximum possible area of the triangle
theorem triangle_division_max_area_perimeter_eq (A B C : ℝ × ℝ) (line : ℝ × ℝ → Prop) :
  divides_equal_perimeter A B C line →
  ∃ D E, 
  D ∈ segment A B ∧ E ∈ segment B C ∧ 
  max_area = area (triangle A B D) :=
sorry

end triangle_division_max_area_perimeter_eq_l117_117735


namespace correct_transformation_l117_117856

noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def Q : ℝ × ℝ := (3, 0)
noncomputable def P : ℝ × ℝ := (3, 3)  -- This should be calculated based on the given conditions

-- Define the rotation by 60 degrees counterclockwise
def rotate_60 (x y : ℝ) : ℝ × ℝ :=
  let θ := Real.pi / 3
  let cosθ := Real.cos θ
  let sinθ := Real.sin θ
  (x * cosθ - y * sinθ, x * sinθ + y * cosθ)

-- Perform the translation 4 units right
def translate_4_right (x y : ℝ) : ℝ × ℝ :=
  (x + 4, y)

-- Applying the transformations
def transformed_P : ℝ × ℝ :=
  let (x, y) := P
  let (x', y') := rotate_60 x y
  translate_4_right x' y'

-- The expected coordinates after transformations
def image_of_P := (11 - 3 * Real.sqrt 3) / 2, (3 * Real.sqrt 3 + 3) / 2

-- Prove the resulting coordinates are as expected
theorem correct_transformation : transformed_P = image_of_P :=
  sorry

end correct_transformation_l117_117856


namespace quadratic_equation_completing_square_l117_117396

theorem quadratic_equation_completing_square :
  (∃ m n : ℝ, (∀ x : ℝ, 15 * x^2 - 30 * x - 45 = 15 * ((x + m)^2 - m^2 - 3) + 45 ∧ (m + n = 3))) :=
sorry

end quadratic_equation_completing_square_l117_117396


namespace solution_set_proof_l117_117653

noncomputable def solution_set_of_inequality {f g : ℝ → ℝ}
  (odd_f : ∀ x, f (-x) = -f x)
  (even_g : ∀ x, g (-x) = g x)
  (domain_fg : ∀ x, x ≠ 0 → (∃ y, y = f x ∨ y = g x))
  (condition : ∀ x < 0, deriv f x * g x + f x * deriv g x > 0)
  (g_at_neg2 : g (-2) = 0) : Set ℝ :=
  {x | f x * g x > 0}

theorem solution_set_proof (odd_f : ∀ x, f (-x) = -f x)
  (even_g : ∀ x, g (-x) = g x)
  (domain_fg : ∀ x, x ≠ 0 → (∃ y, y = f x ∨ y = g x))
  (condition : ∀ x < 0, deriv f x * g x + f x * deriv g x > 0)
  (g_at_neg2 : g (-2) = 0) :
  solution_set_of_inequality odd_f even_g domain_fg condition g_at_neg2 = (-2, 0) ∪ (2, +∞) :=
sorry

end solution_set_proof_l117_117653


namespace BP_value_l117_117886

-- Define the problem conditions and statement.
theorem BP_value
  (A B C D P : Point)
  (on_circle : ∀ point ∈ {A, B, C, D}, is_on_circle point)
  (intersect : P ∈ (line_through A C) ∧ P ∈ (line_through B D))
  (AP : Real := 10)
  (PC : Real := 2)
  (BD : Real := 9)
  (BP_lt_DP : ∃ x y : Real, BP = x ∧ DP = y ∧ x + y = BD ∧ x < y) :
  BP = 4 :=
by
  sorry -- Proof is omitted

end BP_value_l117_117886


namespace find_somu_age_l117_117878

noncomputable def somu_age (S F : ℕ) : Prop :=
  S = (1/3 : ℝ) * F ∧ S - 6 = (1/5 : ℝ) * (F - 6)

theorem find_somu_age {S F : ℕ} (h : somu_age S F) : S = 12 :=
by sorry

end find_somu_age_l117_117878


namespace extreme_value_at_neg3_l117_117818

theorem extreme_value_at_neg3 {a : ℝ} (h : ∃ x, ∀ y, f'(x) = 0 → x = -3) 
: (differentiation of f(x)) = 0 → let a := 5 sorry :=
sorry

end extreme_value_at_neg3_l117_117818


namespace opposite_of_neg_2023_l117_117075

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117075


namespace opposite_of_negative_2023_l117_117059

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117059


namespace locus_of_centers_l117_117435

theorem locus_of_centers :
  ∃ P R S : ℤ, ∃ a : ℝ, P = 2 ∧ R = -6 ∧ S = -7 ∧ P * a^2 + R * a + S = 0 ∧
  gcd |P| (gcd |R| |S|) = 1 := 
by
  use 2, -6, -7, (-1/2 : ℝ)
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  rw [int.gcd_div_gcd_div_gcd]
  norm_num
  field_simp
  sorry

end locus_of_centers_l117_117435


namespace log_inequality_l117_117259

theorem log_inequality (a b : ℝ) (h1 : log 2 a < log 2 b) (h2 : log 2 b < 0) : 0 < a ∧ a < b ∧ b < 1 :=
sorry

end log_inequality_l117_117259


namespace real_axis_hyperbola_length_l117_117311

-- Define the mathematical problem: the length of the real axis of the hyperbola
theorem real_axis_hyperbola_length : 
  ∀ (a b : ℝ), (a > 0) → (b > 0) → 
  (∃ (p : ℝ × ℝ), p = (1, 2) ∧ p.1^2 / a^2 - p.2^2 / b^2 = 1 ∧ a^2 + b^2 = 1) → 
  2 * a = 2 * Real.sqrt 2 - 2 :=
by
  intro a b ha hb h
  cases h with p hp
  cases hp with hp1 hp2
  cases hp2 with hp3 hp4
  sorry

end real_axis_hyperbola_length_l117_117311


namespace rectangular_prism_sum_l117_117724

-- Definitions based on conditions
def edges := 12
def corners := 8
def faces := 6

-- Lean statement to prove question == answer given conditions.
theorem rectangular_prism_sum : edges + corners + faces = 26 := by
  sorry

end rectangular_prism_sum_l117_117724


namespace exists_perpendicular_line_l117_117266

theorem exists_perpendicular_line (n : Line) (α : Plane) : ∃ m : Line, m ∈ α ∧ m ⊥ n :=
by sorry

end exists_perpendicular_line_l117_117266


namespace opposite_of_neg2023_l117_117000

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l117_117000


namespace bricks_required_l117_117157

def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def brick_length : ℝ := 20 / 100
def brick_width : ℝ := 10 / 100

theorem bricks_required :
  (courtyard_length * courtyard_width) / (brick_length * brick_width) = 20000 := 
    sorry

end bricks_required_l117_117157


namespace james_beats_per_week_l117_117368

def beats_per_minute := 200
def hours_per_day := 2
def days_per_week := 7

def beats_per_week (beats_per_minute: ℕ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℕ :=
  (beats_per_minute * hours_per_day * 60) * days_per_week

theorem james_beats_per_week : beats_per_week beats_per_minute hours_per_day days_per_week = 168000 := by
  sorry

end james_beats_per_week_l117_117368


namespace number_of_subsets_P_l117_117677

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}
def P : Set ℕ := M ∩ N

theorem number_of_subsets_P : (Set.powerset P).card = 4 := by
  sorry

end number_of_subsets_P_l117_117677


namespace minimum_sum_of_lengths_l117_117764

noncomputable def parabola := {p : ℝ × ℝ | p.2^2 = 4 * p.1}
structure PointOnParabola where
  P : ℝ × ℝ
  h : P ∈ parabola

def projection_on_y (P : ℝ × ℝ) : ℝ × ℝ := (0, P.2)

def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Define the given specific problem conditions
def point_M : ℝ × ℝ := (4, 5)
def point_F : ℝ × ℝ := (1, 0)

theorem minimum_sum_of_lengths (P : PointOnParabola) :
  let Q := projection_on_y P.P,
      PM := distance P.P point_M,
      PQ := distance P.P Q,
      PF := distance P.P point_F,
      MF := distance point_M point_F in
  PM + PQ = real.sqrt 34 - 1 :=
sorry

end minimum_sum_of_lengths_l117_117764


namespace eval_f_f_neg3_l117_117301

def piecewise_function (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 1 else real.sqrt (1 - x)

theorem eval_f_f_neg3 : piecewise_function (piecewise_function (-3)) = 5 :=
by 
  sorry

end eval_f_f_neg3_l117_117301


namespace find_x_coordinate_l117_117092

theorem find_x_coordinate
  (x : ℝ)
  (passes_origin : ∀ p : ℝ × ℝ, p.1 = 0 → p.2 = 0 → l p)
  (divides_rectangle : ∃ p1 p2 p3 p4 : ℝ × ℝ, 
                       p1 = (1, 0) ∧ p2 = (x, 0) ∧ 
                       p3 = (1, 2) ∧ p4 = (x, 2) ∧
                       l p1 ∧ l p3)
  (slope_l : ∀ p1 p2 : ℝ × ℝ, p1 = (0, 0) → p2 = (1 + x / 2, 1) → l p1 ∧ l p2 → (p2.2 - p1.2) / (p2.1 - p1.1) = 1 / 3) :
  x = 5 := sorry

end find_x_coordinate_l117_117092


namespace remaining_macaroons_weight_l117_117725

theorem remaining_macaroons_weight (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (total_bags : ℕ) :
  (total_macaroons = 12) → 
  (weight_per_macaroon = 5) → 
  (total_bags = 4) → 
  let macaroons_per_bag := total_macaroons / total_bags in
  let weight_per_bag := macaroons_per_bag * weight_per_macaroon in
  let weight_eaten_by_steve := weight_per_bag in
  let total_weight := total_macaroons * weight_per_macaroon in
  let remaining_weight := total_weight - weight_eaten_by_steve in
  remaining_weight = 45 :=
by {
  sorry
}

end remaining_macaroons_weight_l117_117725


namespace points_on_yg_f_f_x_l117_117428

theorem points_on_yg_f_f_x (f : ℝ → ℝ):
  (f 0 = 4) → (f 1 = 2) → (f 2 = 0) →
  (f (f 1) = 0) ∧ (f (f 2) = 4) →
  (ab + cd = 8) :=
by
  intros h0 h1 h2 h3
  let a := 1
  let b := 0
  let c := 2
  let d := 4
  have hab : a * b = 0, by linarith
  have hcd : c * d = 8, by linarith
  rw [hab, hcd]
  exact 0 + 8 = 8

end points_on_yg_f_f_x_l117_117428


namespace f_decreasing_f_max_min_on_interval_l117_117663

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

theorem f_decreasing (x1 x2 : ℝ) (h1 : 2 ≤ x1) (h2 : x1 ≤ 6) (h3 : 2 ≤ x2) (h4 : x2 ≤ 6) (h5 : x1 < x2) :
  f x1 > f x2 :=
by
  sorry

theorem f_max_min_on_interval : 
  (∀ x ∈ Icc 2 6, f(x) ≤ f 2) ∧ (∀ x ∈ Icc 2 6, f(x) ≥ f 6) :=
by
  sorry

end f_decreasing_f_max_min_on_interval_l117_117663


namespace john_subtraction_number_l117_117105

theorem john_subtraction_number (a b : ℕ) (h1 : a = 40) (h2 : b = 1) :
  40^2 - ((2 * 40 * 1) - 1^2) = 39^2 :=
by
  -- sorry indicates the proof is skipped
  sorry

end john_subtraction_number_l117_117105


namespace vertex_of_parabola_y_eq_x2_minus_2_l117_117802

theorem vertex_of_parabola_y_eq_x2_minus_2 :
  vertex (λ x : ℝ, x^2 - 2) = (0, -2) := 
sorry

end vertex_of_parabola_y_eq_x2_minus_2_l117_117802


namespace probability_shortest_diagonal_l117_117481

theorem probability_shortest_diagonal (n : ℕ) (h : n = 20) :
  let D := n * (n - 3) / 2,
      shortest_diagonals := n / 2
  in (shortest_diagonals : ℚ) / D = 1 / 17 :=
by
  sorry

end probability_shortest_diagonal_l117_117481


namespace kristine_travel_distance_l117_117734

theorem kristine_travel_distance :
  ∃ T : ℝ, T + T / 2 + T / 6 = 500 ∧ T = 300 := by
  sorry

end kristine_travel_distance_l117_117734


namespace exponent_problem_l117_117690

theorem exponent_problem (a : ℝ) (m n : ℕ) (h1 : a ^ m = 3) (h2 : a ^ n = 2) : a ^ (m - 2 * n) = 3 / 4 := by
  sorry

end exponent_problem_l117_117690


namespace opposite_of_neg_2023_l117_117041

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117041


namespace ball_distribution_ratio_l117_117220

noncomputable def total_ways := (choose (20 + 5 - 1) 4)

noncomputable def count_A := (5 * 4 * 3 * (choose 20 2) * (choose 18 6) * (choose 12 4) * (choose 8 4) * (choose 4 4))

noncomputable def count_B := (factorial 5 / factorial 4 * (choose 20 4) * (choose 16 4) * (choose 12 4) * (choose 8 4) * (choose 4 4))

def probability_p := count_A / total_ways
def probability_q := count_B / total_ways


theorem ball_distribution_ratio : probability_p / probability_q = 10 / 3 := by
  sorry

end ball_distribution_ratio_l117_117220


namespace cheryl_mms_l117_117945

/-- Cheryl's m&m problem -/
theorem cheryl_mms (c l g d : ℕ) (h1 : c = 25) (h2 : l = 7) (h3 : g = 13) :
  (c - l - g) = d → d = 5 :=
by
  sorry

end cheryl_mms_l117_117945


namespace proof_problem_l117_117623

theorem proof_problem (p q : Prop) 
  (hp : (2 : ℝ) + Real.sqrt 2 ∉ Rational) 
  (hq : ¬ ∃ x : ℝ, x^2 < 0) : 
  (p ∧ ¬ q) = true :=
by
  -- Adding the necessary propositions for translation.
  have hp_true : p := sorry -- p is true because 2 + sqrt(2) is irrational
  have hq_false : q = false := sorry -- q is false because no real number squared is negative

  -- Therefore, p ∧ ¬ q should hold.
  have h_negq_true : ¬ q := by simp [hq_false]
  have h_conjunction : p ∧ ¬ q := ⟨hp_true, h_negq_true⟩

  -- Showing that p ∧ ¬ q is the only true option.
  exact True.intro

end proof_problem_l117_117623


namespace shaded_cubes_in_large_cube_l117_117948

def isShaded (n : Nat) : Prop := n ∈ {0, 1, 2, 3}

def cubeVertices : List (Nat × Nat × Nat) :=
  [(x, y, z) | x ← [0, 3], y ← [0, 3], z ← [0, 3]]

def cubeEdgesMiddle (n: Nat) : List (Nat × Nat × Nat) :=
  (List.range n).allPairs (λ x z, [(x, 0, z), (x, 3, z), (0, x, z), (3, x, z), 
                                    (x, z, 0), (x, z, 3)]).filterMap (λ x, 
   if x.snd < n && x.fst < n then some (x.fst, x.snd) else none)

def faceCenters (n: Nat): List (Nat × Nat × Nat) := 
  (List.range n).allPairs (λ i j, if i = 3/2 && j = 3/2 then [(i, j, 0), (i, j, 3), 
                                                           (0, i, j), (3, i, j), 
                                                           (i, 0, j), (i, 3, j)]
                                                           else none)

def shadedCubes : List (Nat × Nat × Nat) :=
  cubeVertices ++ cubeEdgesMiddle 4 ++ faceCenters 4

theorem shaded_cubes_in_large_cube :
  ∃ n : Nat, n = shadedCubes.length ∧ n = 33 := by
  sorry

end shaded_cubes_in_large_cube_l117_117948


namespace max_fraction_value_l117_117699

theorem max_fraction_value (x y : ℝ)
  (h1 : sqrt 3 * x - y + sqrt 3 ≥ 0)
  (h2 : sqrt 3 * x + y - sqrt 3 ≤ 0)
  (h3 : y ≥ 0) :
  ∃ t, (y + 1) / (x + 3) = t ∧ ∀ t', t' ≤ t → (x + y = sqrt 3) := 
sorry

end max_fraction_value_l117_117699


namespace scientific_notation_of_858_million_l117_117791

theorem scientific_notation_of_858_million :
  858000000 = 8.58 * 10 ^ 8 :=
sorry

end scientific_notation_of_858_million_l117_117791


namespace remaining_macaroons_weight_l117_117730

-- Problem conditions
variables (macaroons_per_bake : ℕ) (weight_per_macaroon : ℕ) (bags : ℕ) (macaroons_eaten : ℕ)

-- Definitions from problem conditions
def macaroons_per_bake := 12
def weight_per_macaroon := 5
def bags := 4
def macaroons_per_bag := macaroons_per_bake / bags
def macaroons_eaten := macaroons_per_bag

-- Lean theorem
theorem remaining_macaroons_weight : (macaroons_per_bake - macaroons_eaten) * weight_per_macaroon = 45 :=
by
  have h1 : macaroons_per_bag = 12 / 4 := rfl
  have h2 : macaroons_per_bag = 3 := by norm_num [h1]
  have h3 : macaroons_eaten = 3 := h2
  have h4 : macaroons_per_bake - macaroons_eaten = 12 - 3 := rfl
  have h5 : macaroons_per_bake - macaroons_eaten = 9 := by norm_num [h4]
  have h6 : (macaroons_per_bake - macaroons_eaten) * weight_per_macaroon = 9 * 5 := by rw [h5]
  calc
    (macaroons_per_bake - macaroons_eaten) * weight_per_macaroon = 9 * 5 : by rw [h6]
    ... = 45 : by norm_num

end remaining_macaroons_weight_l117_117730


namespace fish_left_in_sea_l117_117508

-- Definitions based on conditions
def total_fish_westward : Nat := 1800
def total_fish_eastward : Nat := 3200
def total_fish_north : Nat := 500

def caught_fraction_westward : Rat := 3/4
def caught_fraction_eastward : Rat := 2/5

-- Theorem statement
theorem fish_left_in_sea : 
  let fish_left_westward := total_fish_westward - (caught_fraction_westward * total_fish_westward).nat
  let fish_left_eastward := total_fish_eastward - (caught_fraction_eastward * total_fish_eastward).nat
  let fish_left_north := total_fish_north
  fish_left_westward + fish_left_eastward + fish_left_north = 2870 := 
by
  -- Placeholder for proof
  sorry

end fish_left_in_sea_l117_117508


namespace opposite_of_negative_2023_l117_117047

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117047


namespace part1_part2_l117_117652

theorem part1 (a : ℝ) (h1 : ∀ x y, y = a * x + 1 → 3 * x^2 - y^2 = 1) (h2 : ∃ x1 y1 x2 y2 : ℝ, y1 = a * x1 + 1 ∧ y2 = a * x2 + 1 ∧ 3 * x1 * x1 - y1 * y1 = 1 ∧ 3 * x2 * x2 - y2 * y2 = 1 ∧ x1 * x2 + (a * x1 + 1) * (a * x2 + 1) = 0) : a = 1 ∨ a = -1 := sorry

theorem part2 (h : ∀ x y, y = a * x + 1 → 3 * x^2 - y^2 = 1) (a : ℝ) (h2 : ∃ x1 y1 x2 y2 : ℝ, y1 = a * x1 + 1 ∧ y2 = a * x2 + 1 ∧ 3 * x1 * x1 - y1 * y1 = 1 ∧ 3 * x2 * x2 - y2 * y2 = 1 ∧ (y1 + y2) / 2 = (1 / 2) * (x1 + x2) / 2 ∧ (y1 - y2) / (x1 - x2) = -2) : false := sorry

end part1_part2_l117_117652


namespace opposite_of_neg_2023_l117_117008

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117008


namespace train_length_l117_117923

theorem train_length
    (V : ℝ) -- train speed in m/s
    (L : ℝ) -- length of the train in meters
    (H1 : L = V * 18) -- condition: train crosses signal pole in 18 sec
    (H2 : L + 333.33 = V * 38) -- condition: train crosses platform in 38 sec
    (V_pos : 0 < V) -- additional condition: speed must be positive
    : L = 300 :=
by
-- here goes the proof which is not required for our task
sorry

end train_length_l117_117923


namespace hypotenuse_length_l117_117977

theorem hypotenuse_length (h : ℝ) (a : ℝ) (c : ℝ) (H1 : h = 1) (H2 : a = 15) (H3 : c = 90) :
  let A := triangle_angle 15
  let C := triangle_angle 90
  let CH := perpendicular_height 1
  let B := hypotenuse_of_right_triangle CH A C in
B = 4 :=
sorry

end hypotenuse_length_l117_117977


namespace find_P_xi_greater_than_4_l117_117992

noncomputable def normalDist (mean variance : ℝ) (xi : ℝ → ℝ) : Prop :=
∀ x, xi x ∼ Normal (mean, variance)

variable {xi : ℝ → ℝ} (h₁ : normalDist 1 (6^2) xi)
          (h₂ : P (-2 ≤ xi ∧ xi ≤ 1) = 0.4)

theorem find_P_xi_greater_than_4 : P (xi > 4) = 0.1 :=
sorry

end find_P_xi_greater_than_4_l117_117992


namespace ratio_arithmetic_sequences_l117_117257

variable (a : ℕ → ℕ) (b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (h : ∀ n : ℕ, S n / T n = (3 * n - 1) / (2 * n + 3))

theorem ratio_arithmetic_sequences :
  a 7 / b 7 = 38 / 29 :=
sorry

end ratio_arithmetic_sequences_l117_117257


namespace opposite_of_neg_2023_l117_117032

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117032


namespace ratio_of_perimeters_l117_117136

theorem ratio_of_perimeters (s d s' d': ℝ) (h1 : d = s * Real.sqrt 2) (h2 : d' = 2.5 * d) (h3 : d' = s' * Real.sqrt 2) : (4 * s') / (4 * s) = 5 / 2 :=
by
  -- Additional tactical details for completion, proof is omitted as per instructions
  sorry

end ratio_of_perimeters_l117_117136


namespace almond_butter_servings_l117_117901

def servings_of_almond_butter (tbsp_in_container : ℚ) (tbsp_per_serving : ℚ) : ℚ :=
  tbsp_in_container / tbsp_per_serving

def container_holds : ℚ := 37 + 2/3

def serving_size : ℚ := 3

theorem almond_butter_servings :
  servings_of_almond_butter container_holds serving_size = 12 + 5/9 := 
by
  sorry

end almond_butter_servings_l117_117901


namespace bowling_ball_surface_area_l117_117147

theorem bowling_ball_surface_area (d : ℝ) (hd : d = 9) : 
  4 * Real.pi * (d / 2)^2 = 81 * Real.pi :=
by
  -- proof goes here
  sorry

end bowling_ball_surface_area_l117_117147


namespace opposite_of_neg_2023_l117_117014

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117014


namespace total_journey_time_is_24_l117_117877

-- Given conditions
def river_speed : ℝ := 2 -- River speed in km/hr
def boat_still_water_speed : ℝ := 6 -- Boat speed in still water in km/hr
def distance : ℝ := 64 -- Distance traveled upstream in km

-- Calculation constants
def upstream_speed := boat_still_water_speed - river_speed
def downstream_speed := boat_still_water_speed + river_speed
def time_upstream := distance / upstream_speed
def time_downstream := distance / downstream_speed
def total_journey_time := time_upstream + time_downstream

-- Theorem to prove
theorem total_journey_time_is_24 : total_journey_time = 24 := by
  sorry

end total_journey_time_is_24_l117_117877


namespace sum_of_roots_quadratic_specific_sum_of_roots_l117_117865

theorem sum_of_roots_quadratic:
  ∀ a b c : ℚ, a ≠ 0 → 
  ∀ x1 x2 : ℚ, (a * x1^2 + b * x1 + c = 0) ∧ 
               (a * x2^2 + b * x2 + c = 0) → 
               x1 + x2 = -b / a := 
by
  sorry

theorem specific_sum_of_roots:
  ∀ x1 x2 : ℚ, (12 * x1^2 + 19 * x1 - 21 = 0) ∧ 
               (12 * x2^2 + 19 * x2 - 21 = 0) → 
               x1 + x2 = -19 / 12 := 
by
  sorry

end sum_of_roots_quadratic_specific_sum_of_roots_l117_117865


namespace area_of_triangle_PQR_l117_117850

-- Definitions based on conditions
def is_isosceles_right_triangle (P Q R : Type) (angleP : ℝ) (lengthPR : ℝ) := 
  angleP = 90 ∧ lengthPR = 8

-- The proof goal
theorem area_of_triangle_PQR {P Q R : Type} (h : is_isosceles_right_triangle P Q R 90 8) : 
  ∃ (area : ℝ), area = 32 :=
begin
  sorry
end

end area_of_triangle_PQR_l117_117850


namespace gcd_proof_l117_117247

def gcd_10010_15015 := Nat.gcd 10010 15015 = 5005

theorem gcd_proof : gcd_10010_15015 :=
by
  sorry

end gcd_proof_l117_117247


namespace problem_statement_l117_117947

theorem problem_statement : 20 * (256 / 4 + 64 / 16 + 16 / 64 + 2) = 1405 := by
  sorry

end problem_statement_l117_117947


namespace solve_problem_l117_117425

noncomputable def problem_statement : Prop :=
  ∃ (a b c : ℤ),
  Polynomial.gcd (Polynomial.C b + Polynomial.C a * Polynomial.X + Polynomial.X^2)
                 (Polynomial.C c + Polynomial.C b * Polynomial.X + Polynomial.X^2) = Polynomial.X + 1 ∧
  Polynomial.lcm (Polynomial.C b + Polynomial.C a * Polynomial.X + Polynomial.X^2)
                 (Polynomial.C c + Polynomial.C b * Polynomial.X + Polynomial.X^2) = Polynomial.X^3 - 5 * Polynomial.X^2 + 7 * Polynomial.X - 3 ∧
  a + b + c = -8

theorem solve_problem : problem_statement := sorry

end solve_problem_l117_117425


namespace circle_radius_on_sphere_l117_117760

theorem circle_radius_on_sphere
  (sphere_radius : ℝ)
  (circle1_radius : ℝ)
  (circle2_radius : ℝ)
  (circle3_radius : ℝ)
  (all_circle_touch_each_other : Prop)
  (smaller_circle_touches_all : Prop)
  (smaller_circle_radius : ℝ) :
  sphere_radius = 2 →
  circle1_radius = 1 →
  circle2_radius = 1 →
  circle3_radius = 1 →
  all_circle_touch_each_other →
  smaller_circle_touches_all →
  smaller_circle_radius = 1 - Real.sqrt (2 / 3) :=
by
  intros h_sphere_radius h_circle1_radius h_circle2_radius h_circle3_radius h_all_circle_touch h_smaller_circle_touch
  sorry

end circle_radius_on_sphere_l117_117760


namespace part1_control_function_part2_control_function_and_value_part3_control_function_l117_117276

-- Part (1)
theorem part1_control_function (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : 
  let f := λ x : ℝ, 2 * x ^ 3 - 3 * x ^ 2 + x
  let g := λ x : ℝ, x
  f x ≤ g x :=
sorry

-- Part (2)
theorem part2_control_function_and_value :
  let f := λ x : ℝ, - x ^ 2 + x
  let h := λ x : ℝ, (½) * x + (1/16)
  ∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1), f x ≤ h x ∧ h (¼) = (3/16) :=
sorry

-- Part (3)
theorem part3_control_function (x0 c : ℝ) (a : ℝ) (h1 : 0 ≤ x0 ∧ x0 ≤ 1) (h2 : x0 ∈ (0,1)) (h3 : c ∈ x0..1) :
  let f := λ x : ℝ, a * x ^ 3 - (a + 1) * x ^ 2 + x
  let t := λ x : ℝ, - (1 / (4 * a)) * (x - 1) 
  f 1 = 0 ∧ (c = x0 ∨ c = 1) ↔ f c = f x0 :=
sorry

end part1_control_function_part2_control_function_and_value_part3_control_function_l117_117276


namespace range_of_m_with_three_tangents_l117_117621

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Define the statement of the theorem
theorem range_of_m_with_three_tangents : 
  ∀ (m : ℝ), (∃ (A : (ℝ × ℝ)), A = (1, m) ∧ ∃ t1 t2 t3 : ℝ → ℝ, 
  (∀ t : ℝ → ℝ, (∃ x0 : ℝ, t x0 = f x0 ∧ t = λ x, f x0 + (f' x0) * (x - x0)) → (t = t1 ∨ t = t2 ∨ t = t3))) ↔ -3 < m ∧ m < -2 :=
sorry

end range_of_m_with_three_tangents_l117_117621


namespace bike_to_tractor_speed_ratio_l117_117433

-- Definitions of the conditions in the problem
def speed_of_tractor := 575 / 23 -- km/h
def speed_of_car := 630 / 7 -- km/h
def average_speed_car := (9 / 5 : ℝ) * speed_of_bike -- car speed to bike speed ratio

-- Defining the key entities
def speed_of_bike : ℝ := speed_of_car * (5 / 9) -- using the given ratio

-- The goal is to prove the ratio of the speed of the bike to the speed of the tractor
theorem bike_to_tractor_speed_ratio : (speed_of_bike / speed_of_tractor) = 2 / 1 :=
by
  have h1 : speed_of_tractor = 25 := by sorry
  have h2 : speed_of_car = 90 := by sorry
  have h3 : speed_of_bike = 50 := by sorry
  have h4 : speed_of_bike / speed_of_tractor = 2 := by
    rw [h3, h1]
    exact rfl
  rw [h4]
  norm_num
  exact rfl

end bike_to_tractor_speed_ratio_l117_117433


namespace shortest_distance_between_inscribed_circles_l117_117916

theorem shortest_distance_between_inscribed_circles :
  ∀ (s : ℝ)(n : ℕ)(p : s = 1)(q : n = 9)(r : ∀ x : set ℝ, x≠ ∅)
  , let radius := s / 2 in let diag := real.sqrt (s^2 + s^2) in
  (dist_between_circles diag radius = 2 * real.sqrt 2 - 1) := 
by
  sorry

end shortest_distance_between_inscribed_circles_l117_117916


namespace Brad_age_l117_117372

theorem Brad_age (shara_age : ℕ) (h_shara : shara_age = 10)
  (jaymee_age : ℕ) (h_jaymee : jaymee_age = 2 * shara_age + 2)
  (brad_age : ℕ) (h_brad : brad_age = (shara_age + jaymee_age) / 2 - 3) : brad_age = 13 := by
  sorry

end Brad_age_l117_117372


namespace range_of_fa_l117_117297

open real

theorem range_of_fa : 
  ∀ (ξ : ℝ → ℝ) (a b : ℝ), 
    (∀ x, ξ x = pdf_normal 2 3 x) → 
    (prob (λ x, ξ x > 3) = a) → 
    (prob (λ x, 1 < ξ x ∧ ξ x ≤ 3) = b) → 
    (2 * a + b = 1) → 
    (0 < a ∧ a < 1 / 2) → 
    ∀ y, 
      (y = (a^2 + a - 1) / (a + 1)) → 
      -1 < y ∧ y < -1 / 6 :=
sorry

end range_of_fa_l117_117297


namespace find_x_range_l117_117278

variable {x : ℝ}

def P (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0

def Q (x : ℝ) : Prop := |1 - x/2| < 1

theorem find_x_range (hP : P x) (hQ : ¬ Q x) : x ≤ -1 ∨ x ≥ 4 :=
  sorry

end find_x_range_l117_117278


namespace cos_minus_sin_eq_neg_sqrt3_div_2_l117_117618

theorem cos_minus_sin_eq_neg_sqrt3_div_2 
  (α : ℝ) 
  (h1 : sin α * cos α = 1 / 8) 
  (h2 : π / 4 < α) 
  (h3 : α < π / 2) : 
  cos α - sin α = - (real.sqrt 3) / 2 :=
by
  sorry

end cos_minus_sin_eq_neg_sqrt3_div_2_l117_117618


namespace probability_correct_l117_117135

-- Definitions of given conditions
def P_AB := 2 / 3
def P_BC := 1 / 2

-- Probability that at least one road is at least 5 miles long
def probability_at_least_one_road_is_5_miles_long : ℚ :=
  1 - (1 - P_AB) * (1 - P_BC)

theorem probability_correct :
  probability_at_least_one_road_is_5_miles_long = 5 / 6 :=
by
  -- Proof goes here
  sorry

end probability_correct_l117_117135


namespace complex_number_problem_l117_117285

theorem complex_number_problem (z : ℂ) (h : z * (1 - complex.I) = 1 + complex.I) : z^(2016) = 1 :=
by
  have h_imaginary_unit := complex.i_eq_i,
  sorry

end complex_number_problem_l117_117285


namespace ratio_of_x_to_y_l117_117341

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + 3 * y) = 1 / 2) : x / y = 7 / 4 :=
sorry

end ratio_of_x_to_y_l117_117341


namespace trapezoid_base_pairs_l117_117794

theorem trapezoid_base_pairs :
  let h := 60 in
  let A := 1800 in
  ∃ (b1 b2 : ℕ), (h * (b1 + b2)) / 2 = A ∧
  (∃ m n : ℕ, b1 = 10 * m ∧ b2 = 10 * n ∧ m + n = 6) ∧
  (finset.card {p : ℕ × ℕ | let (m, n) := p in m + n = 6 ∧ m ≤ n} = 4) :=
by
  let h := 60
  let A := 1800
  existsi (10 * 0, 10 * 6)
  existsi (10 * 1, 10 * 5)
  existsi (10 * 2, 10 * 4)
  existsi (10 * 3, 10 * 3)
  sorry

end trapezoid_base_pairs_l117_117794


namespace fly_travel_time_to_opposite_vertex_l117_117166

noncomputable def cube_side_length (a : ℝ) := 
  a

noncomputable def fly_travel_time_base := 4 -- minutes

noncomputable def fly_speed (a : ℝ) := 
  4 * a / fly_travel_time_base

noncomputable def space_diagonal_length (a : ℝ) := 
  a * Real.sqrt 3

theorem fly_travel_time_to_opposite_vertex (a : ℝ) : 
  fly_speed a ≠ 0 -> 
  space_diagonal_length a / fly_speed a = Real.sqrt 3 :=
by
  intro h
  sorry

end fly_travel_time_to_opposite_vertex_l117_117166


namespace termination_l117_117408

def adjustment (x y z u v : ℤ) : Prop :=
  x + y + z + u + v > 0 ∧ (x + y, -y, z + y, u, v) ≠ (x, y, z, u, v)

theorem termination {x y z u v : ℤ} (h : x + y + z + u + v > 0) :
  ∀ (y : ℤ), y < 0 → ∃ n : ℕ, ∀ k ≥ n, (x, y, z, u, v).nth k ≥ 0 := sorry

end termination_l117_117408


namespace rodney_correct_guess_probability_l117_117771

noncomputable def is_valid_number (n : ℕ) : Prop :=
  (75 < n) ∧ (n < 100) ∧ (even (n / 10)) ∧ (odd (n % 10))

theorem rodney_correct_guess_probability :
  let all_valid_numbers := {n : ℕ | is_valid_number n}.to_finset in
  fintype.card all_valid_numbers = 10 →
  (1 / fintype.card all_valid_numbers = (1 / 10)) :=
by
  intros valid_cardinality valid_set_cardinality_eq
  rw valid_set_cardinality_eq
  norm_num
  apply real.inval.eq
  norm_num

end rodney_correct_guess_probability_l117_117771


namespace specific_gravity_cylinder_l117_117535

noncomputable def specific_gravity_of_cylinder (r m : ℝ) : ℝ :=
  (1 / 3) - (Real.sqrt 3 / (4 * Real.pi))

theorem specific_gravity_cylinder
  (r m : ℝ) 
  (cylinder_floats : r > 0 ∧ m > 0)
  (submersion_depth : r / 2 = r / 2) :
  specific_gravity_of_cylinder r m = 0.1955 :=
sorry

end specific_gravity_cylinder_l117_117535


namespace more_orange_pages_read_l117_117584

-- Define the conditions
def purple_pages_per_book : Nat := 230
def orange_pages_per_book : Nat := 510
def purple_books_read : Nat := 5
def orange_books_read : Nat := 4

-- Calculate the total pages read from purple and orange books respectively
def total_purple_pages_read : Nat := purple_pages_per_book * purple_books_read
def total_orange_pages_read : Nat := orange_pages_per_book * orange_books_read

-- State the theorem to be proved
theorem more_orange_pages_read : total_orange_pages_read - total_purple_pages_read = 890 :=
by
  -- This is where the proof steps would go, but we'll leave it as sorry to indicate the proof is not provided
  sorry

end more_orange_pages_read_l117_117584


namespace find_principal_l117_117898

/-- Given that the simple interest SI is Rs. 90, the rate R is 3.5 percent, and the time T is 4 years,
prove that the principal P is approximately Rs. 642.86 using the simple interest formula. -/
theorem find_principal
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) 
  (h1 : SI = 90) (h2 : R = 3.5) (h3 : T = 4) 
  : P = 90 * 100 / (3.5 * 4) :=
by
  sorry

end find_principal_l117_117898


namespace smallest_special_number_correct_l117_117094

noncomputable def smallest_special_number : ℕ :=
  let digits := [1, 3, 5, 7, 9]
  in
  let five_digit_numbers := (List.permutations digits).map (λ l, l.foldl (λ acc d, acc * 10 + d) 0)
  in
  five_digit_numbers.filter (λ n, (digits.filter (≠ 1)).any (λ d, n % d = 0))
                    .minimum (λ a b, a < b)
                    .getD 0

theorem smallest_special_number_correct : smallest_special_number = 13597 := sorry

end smallest_special_number_correct_l117_117094


namespace jan_total_skips_l117_117721

def jan_initial_speed : ℕ := 70
def jan_training_factor : ℕ := 2
def jan_skipping_time : ℕ := 5

theorem jan_total_skips :
  (jan_initial_speed * jan_training_factor) * jan_skipping_time = 700 := by
  sorry

end jan_total_skips_l117_117721


namespace initial_salt_percentage_l117_117109

theorem initial_salt_percentage (initial_mass : ℝ) (added_salt_mass : ℝ) (final_solution_percentage : ℝ) (final_mass : ℝ) 
  (h1 : initial_mass = 100) 
  (h2 : added_salt_mass = 38.46153846153846) 
  (h3 : final_solution_percentage = 0.35) 
  (h4 : final_mass = 138.46153846153846) : 
  ((10 / 100) * 100) = 10 := 
sorry

end initial_salt_percentage_l117_117109


namespace value_of_expression_l117_117521

theorem value_of_expression (p q r s : ℝ) (h : -27 * p + 9 * q - 3 * r + s = -7) : 
  4 * p - 2 * q + r - s = 7 :=
by
  sorry

end value_of_expression_l117_117521


namespace hexagonal_prism_sum_maximum_l117_117529

noncomputable def hexagonal_prism_max_sum (h_u h_v h_w h_x h_y h_z : ℕ) (u v w x y z : ℝ) : ℝ :=
  u + v + w + x + y + z

def max_sum_possible (h_u h_v h_w h_x h_y h_z : ℕ) : ℝ :=
  if h_u = 4 ∧ h_v = 7 ∧ h_w = 10 ∨
     h_u = 4 ∧ h_x = 7 ∧ h_y = 10 ∨
     h_u = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_v = 4 ∧ h_x = 7 ∧ h_w = 10 ∨
     h_v = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_w = 4 ∧ h_x = 7 ∧ h_z = 10
  then 78
  else 0

theorem hexagonal_prism_sum_maximum (h_u h_v h_w h_x h_y h_z : ℕ) :
  max_sum_possible h_u h_v h_w h_x h_y h_z = 78 → ∃ (u v w x y z : ℝ), hexagonal_prism_max_sum h_u h_v h_w h_x h_y h_z u v w x y z = 78 := 
by 
  sorry

end hexagonal_prism_sum_maximum_l117_117529


namespace geom_series_common_ratio_l117_117828

theorem geom_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hNewS : (ar^3) / (1 - r) = S / 27) : r = 1 / 3 :=
by
  sorry

end geom_series_common_ratio_l117_117828


namespace opposite_of_neg_2023_l117_117020

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117020


namespace line_AB_properties_l117_117315

-- Definitions of points and the equation of the line
def A : (ℝ × ℝ) := (2, 1)
def B : ℝ × ℝ := sorry -- It will be parameterized by m later

-- The statement covering all the proof goals
theorem line_AB_properties (m : ℝ) 
  (h_m : m ∈ set.Icc (2 - Real.sqrt 3) (2 + 3 * Real.sqrt 3)) :
  (if m = 2 then 
    (∀ x y : ℝ, (3x + (2 - m) * y + m - 8 = 0) → x = 2) ∧ 
    (∀ α : ℝ, α = Real.arctan (3 / (m - 2)) → α ∈ set.Icc (Real.pi / 6) (2 * Real.pi / 3))
  else 
    ∃ k : ℝ, k = 3 / (m - 2) ∧ ∀ x y : ℝ, y - 1 = k * (x - 2) → 3x + (2 - m) y + m - 8 = 0 ∧
    ∀ α : ℝ, α = Real.arctan k → α ∈ set.Icc (Real.pi / 6) (2 * Real.pi / 3)) := 
sorry

end line_AB_properties_l117_117315


namespace apple_difference_l117_117468

def apples_at_start := 1200 -- Given the original number of green apples
def red_apples_difference := 3250 -- Given the difference in initial count
def green_apples_delivered := 3600
def red_apples_delivered := 1300
def green_apples_sold := 750
def red_apples_sold := 810

def original_red_apples := apples_at_start + red_apples_difference
def green_apples_after_delivery := apples_at_start + green_apples_delivered
def red_apples_after_delivery := original_red_apples + red_apples_delivered
def green_apples_after_sales := green_apples_after_delivery - green_apples_sold
def red_apples_after_sales := red_apples_after_delivery - red_apples_sold

theorem apple_difference : green_apples_after_sales - red_apples_after_sales = -890 := 
by
  sorry

end apple_difference_l117_117468


namespace arctan_sum_eq_pi_over_4_l117_117979

noncomputable def find_n : ℕ :=
let m := 56 in
m

theorem arctan_sum_eq_pi_over_4 :
  (Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/6) + Real.arctan (1/find_n) = Real.pi / 4) :=
by
  let m := find_n
  have h : m = 56 := rfl
  rw [<-h]
  sorry

end arctan_sum_eq_pi_over_4_l117_117979


namespace problem_1_problem_2_l117_117305

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if -1 < x ∧ x < 2 then x^2
  else 2 * x

-- Prove the first set of values for f(-3), f(4), and f(f(-2))
theorem problem_1 :
  (f (-3) = -1) ∧ (f 4 = 8) ∧ (f (f (-2)) = 8) :=
by
  sorry

-- Prove that if f(m) = 8, then m = 4
theorem problem_2 (m : ℝ) (h : f m = 8) : m = 4 :=
by
  sorry

end problem_1_problem_2_l117_117305


namespace sale_in_third_month_l117_117515

def average_sale (s1 s2 s3 s4 s5 s6 : ℕ) : ℕ :=
  (s1 + s2 + s3 + s4 + s5 + s6) / 6

theorem sale_in_third_month
  (S1 S2 S3 S4 S5 S6 : ℕ)
  (h1 : S1 = 6535)
  (h2 : S2 = 6927)
  (h4 : S4 = 7230)
  (h5 : S5 = 6562)
  (h6 : S6 = 4891)
  (havg : average_sale S1 S2 S3 S4 S5 S6 = 6500) :
  S3 = 6855 := 
sorry

end sale_in_third_month_l117_117515


namespace roots_equation_sum_and_product_l117_117288

theorem roots_equation_sum_and_product (x1 x2 : ℝ) (h1 : x1 ^ 2 - 3 * x1 - 5 = 0) (h2 : x2 ^ 2 - 3 * x2 - 5 = 0) :
  x1 + x2 - x1 * x2 = 8 :=
sorry

end roots_equation_sum_and_product_l117_117288


namespace sam_needs_change_probability_l117_117836

theorem sam_needs_change_probability :
  let total_permutations := factorial 9
      without_change := factorial 8 + factorial 7 + factorial 6
  in (total_permutations - without_change) / total_permutations = (55 : ℚ) / 63 :=
by
  let total_permutations := factorial 9
  let without_change := factorial 8 + factorial 7 + factorial 6
  have h : (total_permutations - without_change) * 63 = 55 * total_permutations, from sorry,
  exact (div_eq_div_iff_of_neq_zero' (by norm_num : (63 : ℚ) ≠ 0) (by norm_num : (factorial 9 : ℚ) ≠ 0)).2 h

end sam_needs_change_probability_l117_117836


namespace opposite_of_neg_2023_l117_117069

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117069


namespace Jane_saves_five_dollars_l117_117164

noncomputable def first_pair_cost : ℝ := 50
noncomputable def second_pair_cost_A : ℝ := first_pair_cost * 0.6
noncomputable def second_pair_cost_B : ℝ := first_pair_cost - 15
noncomputable def promotion_A_total_cost : ℝ := first_pair_cost + second_pair_cost_A
noncomputable def promotion_B_total_cost : ℝ := first_pair_cost + second_pair_cost_B
noncomputable def Jane_savings : ℝ := promotion_B_total_cost - promotion_A_total_cost

theorem Jane_saves_five_dollars : Jane_savings = 5 := by
  sorry

end Jane_saves_five_dollars_l117_117164


namespace gcd_10010_15015_l117_117241

theorem gcd_10010_15015 :
  let n1 := 10010
  let n2 := 15015
  ∃ d, d = Nat.gcd n1 n2 ∧ d = 5005 :=
by
  let n1 := 10010
  let n2 := 15015
  -- ... omitted proof steps
  sorry

end gcd_10010_15015_l117_117241


namespace const_term_in_expansion_l117_117326

noncomputable def a : ℝ := ∫ x in -1..1, real.sqrt (1 - x^2)

theorem const_term_in_expansion :
  a = ∫ x in -1..1, real.sqrt (1 - x^2) →
  let exp := (a / real.pi * x - 1 / x)^6 in
  (∑ k in finset.range 7, nat.choose 6 k * (a / real.pi)^k * (1 / x)^k * (-1 / x)^(6 - k) = -5 / 2) : sorry

end const_term_in_expansion_l117_117326


namespace cos_4theta_l117_117328

theorem cos_4theta (θ : ℂ) (h : exp (complex.I * θ) = (1 + 2 * complex.I) / real.sqrt 5) : 
  complex.cos (4 * θ) = -7 / 25 := 
sorry

end cos_4theta_l117_117328


namespace trapezoid_ratio_l117_117391

theorem trapezoid_ratio (u v : ℝ) (h1 : u > v) (h2 : (u + v) * (14 / u + 6 / v) = 40) : u / v = 7 / 3 :=
sorry

end trapezoid_ratio_l117_117391


namespace rook_paths_eq_catalan_l117_117899

def catalan (k : ℕ) : ℕ := (Nat.factorial (2*k)) / (Nat.factorial k * Nat.factorial (k+1))

theorem rook_paths_eq_catalan (n : ℕ) (h : 2 ≤ n) :
  let paths := {p : List (ℕ × ℕ) | 
      p.head = (1, 1) ∧
      p.last = (n, n) ∧
      ∀ ⟨x, y⟩ ∈ p, x ≤ n ∧ y ≤ n ∧ (x = y → (x, y) = (1, 1) ∨ (x, y) = (n, n)) ∧
      ∀ i : ℕ, i < p.length - 1 → (p[i].1 = p[i+1].1 + 1 ∨ p[i].2 = p[i+1].2 + 1)}
  in paths.card = catalan (n - 2) := sorry

end rook_paths_eq_catalan_l117_117899


namespace fill_buckets_lcm_l117_117701

theorem fill_buckets_lcm :
  (∀ (A B C : ℕ), (2 / 3 : ℚ) * A = 90 ∧ (1 / 2 : ℚ) * B = 120 ∧ (3 / 4 : ℚ) * C = 150 → lcm A (lcm B C) = 1200) :=
by
  sorry

end fill_buckets_lcm_l117_117701


namespace tangent_line_equation_l117_117976

noncomputable def curve (x : ℝ) : ℝ := (x^29 + 6) / (x^4 + 1)
noncomputable def tangent_at_x1 (x : ℝ) : ℝ := 7.5 * x - 4

theorem tangent_line_equation :
  ∃ (x₀ : ℝ), x₀ = 1 ∧ (∀ x, curve x₀ = (curve x₀ + (deriv curve x₀) * (x - x₀)) = 7.5 * x - 4) :=
begin
  sorry
end

end tangent_line_equation_l117_117976


namespace rationalize_sqrt_35_l117_117416

theorem rationalize_sqrt_35 : (35 / Real.sqrt 35) = Real.sqrt 35 :=
  sorry

end rationalize_sqrt_35_l117_117416


namespace opposite_of_neg_2023_l117_117025

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117025


namespace gcd_example_l117_117211

theorem gcd_example : Nat.gcd (101^11 + 1) (101^11 + 101^3 + 1) = 1 := by
  sorry

end gcd_example_l117_117211


namespace polar_eq_C1_rect_eq_C2_length_AB_l117_117714

section problem

open Real

-- Given definitions.
def C1_line_rect (x y : ℝ) : Prop := y = -sqrt 3 * x

def C2_curve_param (φ : ℝ) : ℝ × ℝ :=
  (-sqrt 3 + cos φ, -2 + sin φ)

def polar_conversion (r θ : ℝ) : ℝ × ℝ :=
  (r * cos θ, r * sin θ)

def C2_curve_rect (x y : ℝ) : Prop :=
  (x + sqrt 3) ^ 2 + (y + 2) ^ 2 = 1

def theta_rotation (θ : ℝ) (dθ : ℝ) : ℝ :=
  θ + dθ

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- To prove (I)
theorem polar_eq_C1 :
  ∀ r θ : ℝ, 
  C1_line_rect (polar_conversion r θ).1 (polar_conversion r θ).2 ↔ θ = 2 * π / 3 := 
  sorry

theorem rect_eq_C2 : 
  ∀ x y : ℝ, 
  (∃ φ : ℝ, C2_curve_param φ = (x, y)) ↔ 
  C2_curve_rect x y := 
  sorry

-- To prove (II)
theorem length_AB :
  let C3_line : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.2 = sqrt 3 * p.1
  ∀ A B : ℝ × ℝ, 
  C3_line A ∧ C2_curve_rect A.1 A.2 → 
  C3_line B ∧ C2_curve_rect B.1 B.2 → 
  distance A.1 A.2 B.1 B.2 = sqrt 3 := 
  sorry

end problem

end polar_eq_C1_rect_eq_C2_length_AB_l117_117714


namespace distance_diff_on_curve_l117_117299

theorem distance_diff_on_curve :
  let x := √(2 * y - y^2) in
  let line := λ x y, x - y - 2 = 0 in
  let max_dist := 2 * √2 in
  let min_dist := (3 * √2 / 2) - 1 in
  max_dist - min_dist = √2 / 2 + 1 :=
by
  sorry

end distance_diff_on_curve_l117_117299


namespace no_int_solutions_l117_117738

open Nat

theorem no_int_solutions (p1 p2 α n : ℕ)
  (hp1_prime : p1.Prime)
  (hp2_prime : p2.Prime)
  (hp1_odd : p1 % 2 = 1)
  (hp2_odd : p2 % 2 = 1)
  (hα_pos : 0 < α)
  (hn_pos : 0 < n)
  (hα_gt1 : 1 < α)
  (hn_gt1 : 1 < n) :
  ¬(let lhs := ((p2 - 1) / 2) ^ p1 + ((p2 + 1) / 2) ^ p1
    lhs = α ^ n) :=
sorry

end no_int_solutions_l117_117738


namespace triangle_determination_l117_117491

theorem triangle_determination (base_angle vertex_angle : ℝ) 
  (vertex_length base_length radius_circumscribed_leg_length 
  radius_inscribed_side_length : ℝ) 
  (A B C : Type) 
  [IsoscelesTriangle A base_angle vertex_angle] 
  [IsoscelesTriangle B vertex_angle base_length] 
  [EquilateralTriangle C radius_circumscribed_leg_length] : 
  ¬(∀ (T : Type), IsoscelesTriangle T base_angle vertex_angle → 
    ∃ (Triangle_eqv : Type), T = Triangle_eqv ) ∧ 
  (∀ (T : Type), IsoscelesTriangle T vertex_angle base_length → 
    ∃ (Triangle_eqv : Type), T = Triangle_eqv ) ∧ 
  (∀ (T : Type), EquilateralTriangle T radius_circumscribed_leg_length → 
    ∃ (Triangle_eqv : Type), T = Triangle_eqv ) ∧ 
  (∀ (T : Type), RightTriangle T base_length radius_inscribed_side_length → 
    ¬∃ (Triangle_eqv : Type), T = Triangle_eqv ) ∧ 
  (∀ (T : Type), ScaleneTriangle T vertex_angle base_length → 
    ∃ (Triangle_eqv : Type), T = Triangle_eqv ) := 
begin 
  sorry 
end

end triangle_determination_l117_117491


namespace outlier_count_is_one_l117_117810

def data_set : List ℕ := [8, 22, 36, 36, 44, 45, 45, 48, 56, 62]
def Q1 : ℕ := 36
def Q3 : ℕ := 48

def IQR : ℕ := Q3 - Q1
def lower_threshold : ℕ := Q1 - 2 * IQR
def upper_threshold : ℕ := Q3 + 2 * IQR

def is_outlier (x : ℕ) : Prop :=
  x < lower_threshold ∨ x > upper_threshold

def number_of_outliers (xs : List ℕ) : ℕ :=
  xs.countp is_outlier

theorem outlier_count_is_one : number_of_outliers data_set = 1 :=
  by
  sorry

end outlier_count_is_one_l117_117810


namespace brad_age_proof_l117_117373

theorem brad_age_proof :
  ∀ (Shara_age Jaymee_age Average_age Brad_age : ℕ),
  Jaymee_age = 2 * Shara_age + 2 →
  Average_age = (Shara_age + Jaymee_age) / 2 →
  Brad_age = Average_age - 3 →
  Shara_age = 10 →
  Brad_age = 13 :=
by
  intros Shara_age Jaymee_age Average_age Brad_age
  intro h1 h2 h3 h4
  sorry

end brad_age_proof_l117_117373


namespace birds_total_distance_covered_l117_117858

def round_trip_distance (building_materials_distance : ℕ) : ℕ :=
  2 * building_materials_distance

def total_distance_one_bird (round_trip_distance : ℕ) (num_trips : ℕ) : ℕ :=
  round_trip_distance * num_trips

def total_distance_two_birds (distance_one_bird : ℕ) : ℕ :=
  2 * distance_one_bird

theorem birds_total_distance_covered :
  let distance_one_trip := round_trip_distance 200 in
  let distance_one_bird := total_distance_one_bird distance_one_trip 10 in
  total_distance_two_birds distance_one_bird = 8000 :=
by sorry

end birds_total_distance_covered_l117_117858


namespace minimum_h22_of_tenuous_l117_117545

-- Define tenuous property
def tenuous (h : ℕ → ℤ) : Prop :=
  ∀ x y : ℕ, 0 < x → 0 < y → h(x) + h(y) > y^2

-- Lean statement for the proof problem
theorem minimum_h22_of_tenuous (h : ℕ → ℤ) (h_tenuous : tenuous h)
  (h_sum_min : (finset.range 30).sum (λ n, h (n + 1)) = 6475) :
  h 22 = 357 := 
sorry

end minimum_h22_of_tenuous_l117_117545


namespace count_expressible_integers_l117_117605

theorem count_expressible_integers :
  let S := { n | ∃ (x : ℝ), n = Int.floor x + Int.floor (2 * x) + Int.floor (3 * x) + Int.floor (4 * x) } in
  S.count (finset.Icc 1 1000) = 600 :=
by
  sorry

end count_expressible_integers_l117_117605


namespace difference_batteries_l117_117110

/-
Conditions:
1. Number of batteries used on flashlights is 2.
2. Number of batteries used in toys is 15.

Prove: The difference between the number of batteries in toys and flashlights is 13.
-/
theorem difference_batteries (f t : ℕ) (hf : f = 2) (ht : t = 15) : t - f = 13 := by
  rw [hf, ht]
  simp
  sorry

end difference_batteries_l117_117110


namespace area_of_triangle_l117_117795

theorem area_of_triangle 
  (h : ∀ x y : ℝ, (x / 5 + y / 2 = 1) → ((x = 5 ∧ y = 0) ∨ (x = 0 ∧ y = 2))) : 
  ∃ t : ℝ, t = 1 / 2 * 2 * 5 := 
sorry

end area_of_triangle_l117_117795


namespace range_of_a_l117_117308

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h : f = abs) 
  (h₁ : ∀ x : ℝ, 2 * f (x - 1) + f (2 * x - a) ≥ 1) : 
  a ∈ set.Ioo 0 1 ∪ set.Ici 3 :=
sorry

end range_of_a_l117_117308


namespace polynomial_at_3_l117_117739

noncomputable def P (x : ℝ) : ℝ := x^7 + 2*x^5 + 2*x^3 + x + 1

theorem polynomial_at_3 (x := 3) : P(x) = 2731 := by
  sorry

end polynomial_at_3_l117_117739


namespace total_weight_of_remaining_macaroons_l117_117732

def total_weight_remaining_macaroons (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (bags : ℕ) (bags_eaten : ℕ) : ℕ :=
  let macaroons_per_bag := total_macaroons / bags
  let remaining_macaroons := total_macaroons - macaroons_per_bag * bags_eaten
  remaining_macaroons * weight_per_macaroon

theorem total_weight_of_remaining_macaroons
  (total_macaroons : ℕ)
  (weight_per_macaroon : ℕ)
  (bags : ℕ)
  (bags_eaten : ℕ)
  (h1 : total_macaroons = 12)
  (h2 : weight_per_macaroon = 5)
  (h3 : bags = 4)
  (h4 : bags_eaten = 1)
  : total_weight_remaining_macaroons total_macaroons weight_per_macaroon bags bags_eaten = 45 := by
  sorry

end total_weight_of_remaining_macaroons_l117_117732


namespace pen_collection_l117_117127

noncomputable def ceil (x : ℝ) : ℝ :=
  if x = floor x then x else (floor x + 1)

theorem pen_collection :
  let P0 := 20
  let P1 := P0 + 22
  let P2 := 2 * P1
  let P3 := P2 - ceil (0.15 * P2)
  let Pf := P3 - 19
  Pf = 52 :=
by
  sorry

end pen_collection_l117_117127


namespace sector_area_proof_l117_117879

/-- Define the radius and arc length as given -/
def radius : ℝ := 4
def arc_length : ℝ := 3.5

/-- Define the formula for the area of a sector -/
def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

def area_of_sector (l : ℝ) (circ : ℝ) (circle_area : ℝ) : ℝ := (l / circ) * circle_area

#eval let r := 4
        let l := 3.5
        let circ := circumference r
        let circle_area := area_of_circle r
        area_of_sector l circ circle_area

/-- Lean statement to prove that the area of the sector is 7 cm^2 -/
theorem sector_area_proof : area_of_sector arc_length (circumference radius) (area_of_circle radius) = 7 :=
by
    sorry

end sector_area_proof_l117_117879


namespace opposite_of_neg_2023_l117_117080

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117080


namespace find_m_interval_l117_117956

-- Define the sequence recursively
def sequence_recursive (x : ℕ → ℝ) (n : ℕ) : Prop :=
  x 0 = 5 ∧ ∀ n, x (n + 1) = (x n ^ 2 + 5 * x n + 4) / (x n + 6)

-- The left-hand side of the inequality
noncomputable def target_value : ℝ := 4 + 1 / (2 ^ 20)

-- The condition that the sequence element must satisfy
def condition (x : ℕ → ℝ) (m : ℕ) : Prop :=
  x m ≤ target_value

-- The proof problem statement, m lies within the given interval
theorem find_m_interval (x : ℕ → ℝ) (m : ℕ) :
  sequence_recursive x n →
  condition x m →
  81 ≤ m ∧ m ≤ 242 :=
sorry

end find_m_interval_l117_117956


namespace solve_equation_l117_117784

theorem solve_equation (x : ℚ) :
  (x^2 + 3 * x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 :=
by
  sorry

end solve_equation_l117_117784


namespace number_of_zero_sequences_is_180_l117_117381

def sequence_zero_count : ℕ :=
  let T := {(a1, a2, a3) : ℕ × ℕ × ℕ | 1 ≤ a1 ∧ a1 ≤ 15 ∧ 1 ≤ a2 ∧ a2 ≤ 15 ∧ 1 ≤ a3 ∧ a3 ≤ 5}
  let generates_sequence (a : ℕ × ℕ × ℕ) (n : ℕ) : ℕ :=
    match n with
    | 1 => a.1
    | 2 => a.2
    | 3 => a.3
    | n' + 3 => generates_sequence a (n' + 2) * (abs (generates_sequence a (n' + 1) - generates_sequence a n'))
  let has_zero_in_sequence (a : ℕ × ℕ × ℕ) : Prop :=
    ∃ n ≥ 4, generates_sequence a n = 0
  in
  (finset.filter has_zero_in_sequence T).card

theorem number_of_zero_sequences_is_180 : sequence_zero_count = 180 := sorry

end number_of_zero_sequences_is_180_l117_117381


namespace rainfall_second_week_january_l117_117967

-- Define the conditions
def total_rainfall_2_weeks (rainfall_first_week rainfall_second_week : ℝ) : Prop :=
  rainfall_first_week + rainfall_second_week = 20

def rainfall_second_week_is_1_5_times_first (rainfall_first_week rainfall_second_week : ℝ) : Prop :=
  rainfall_second_week = 1.5 * rainfall_first_week

-- Define the statement to prove
theorem rainfall_second_week_january (rainfall_first_week rainfall_second_week : ℝ) :
  total_rainfall_2_weeks rainfall_first_week rainfall_second_week →
  rainfall_second_week_is_1_5_times_first rainfall_first_week rainfall_second_week →
  rainfall_second_week = 12 :=
by
  sorry

end rainfall_second_week_january_l117_117967


namespace tea_mixture_ratio_l117_117133

theorem tea_mixture_ratio
    (x y : ℝ)
    (h₁ : 62 * x + 72 * y = 64.5 * (x + y)) :
    x / y = 3 := by
  sorry

end tea_mixture_ratio_l117_117133


namespace points_lie_on_hyperbola_l117_117256

noncomputable def point_on_hyperbola (t : ℝ) : Prop :=
  let x := 2 * (Real.exp t + Real.exp (-t))
  let y := 4 * (Real.exp t - Real.exp (-t))
  (x^2 / 16) - (y^2 / 64) = 1

theorem points_lie_on_hyperbola (t : ℝ) : point_on_hyperbola t := 
by
  sorry

end points_lie_on_hyperbola_l117_117256


namespace range_of_a_l117_117269

-- Definitions for the problem conditions
def is_on_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 2)^2 = 1

def right_triangle (Ax Ay Bx By Cx Cy : ℝ) : Prop :=
  (Ax - Bx) * (Ax - Cx) + (Ay - By) * (Ay - Cy) = 0

def is_on_x_axis (x y : ℝ) : Prop := y = 0

-- The main statement
theorem range_of_a (a : ℝ) :
  (3 - (a - 1) / 2)^2 + 4 = (max (1 - (a + 1) / 2) ((a + 1) / 2 + 1))^2 ∧ a ∈ set.Icc (14 / 5) (16 / 3) :=
begin
  sorry
end

end range_of_a_l117_117269


namespace find_inverse_value_l117_117289

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic (x : ℝ) : f (x - 1) = f (x + 3)
axiom defined_interval (x : ℝ) (h : 4 ≤ x ∧ x ≤ 5) : f x = 2 ^ x + 1

noncomputable def f_inv : ℝ → ℝ := sorry
axiom inverse_defined (x : ℝ) (h : -2 ≤ x ∧ x ≤ 0) : f (f_inv x) = x

theorem find_inverse_value : f_inv 19 = 3 - 2 * (Real.log 3 / Real.log 2) := by
  sorry

end find_inverse_value_l117_117289


namespace opposite_of_neg_2023_l117_117038

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117038


namespace number_of_girls_l117_117254

theorem number_of_girls (num_vans : ℕ) (students_per_van : ℕ) (num_boys : ℕ) (total_students : ℕ) (num_girls : ℕ) 
(h1 : num_vans = 5) 
(h2 : students_per_van = 28) 
(h3 : num_boys = 60) 
(h4 : total_students = num_vans * students_per_van) 
(h5 : num_girls = total_students - num_boys) : 
num_girls = 80 :=
by
  sorry

end number_of_girls_l117_117254


namespace opposite_of_neg_2023_l117_117077

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117077


namespace probability_at_least_one_white_l117_117093

def total_number_of_pairs : ℕ := 10
def number_of_pairs_with_at_least_one_white_ball : ℕ := 7

theorem probability_at_least_one_white :
  (number_of_pairs_with_at_least_one_white_ball : ℚ) / (total_number_of_pairs : ℚ) = 7 / 10 :=
by
  sorry

end probability_at_least_one_white_l117_117093


namespace definite_integral_value_l117_117464

theorem definite_integral_value :
  ∫ x in 0..1, (2 * x + Real.exp x) = Real.exp 1 :=
by
  sorry

end definite_integral_value_l117_117464


namespace find_BC_in_acute_triangle_l117_117351

theorem find_BC_in_acute_triangle
  (A B C : Type) [triangle A B C]
  (h_acute : is_acute A B C)
  (h_sin_A : real.sin (angle A B C) = 3/5)
  (h_AB : distance A B = 5)
  (h_AC : distance A C = 6) :
  distance B C = real.sqrt 13 := by
  sorry

end find_BC_in_acute_triangle_l117_117351


namespace wall_length_l117_117179

theorem wall_length (s : ℕ) (d : ℕ) (w : ℕ) (L : ℝ) 
  (hs : s = 18) 
  (hd : d = 20) 
  (hw : w = 32)
  (hcombined : (s ^ 2 + Real.pi * ((d / 2) ^ 2)) = (1 / 2) * (w * L)) :
  L = 39.88 := 
sorry

end wall_length_l117_117179


namespace sum_of_first_ten_super_numbers_l117_117552

def is_proper_divisor (n d : ℕ) : Prop := d > 1 ∧ d < n ∧ n % d = 0

def is_super (n : ℕ) : Prop :=
(n > 1) ∧
((∃ (p q : ℕ), prime p ∧ prime q ∧ p ≠ q ∧ n = p * q) ∨
 (∃ (p q r : ℕ), prime p ∧ prime q ∧ prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ n = p * q * r) ∨
 (∃ (p : ℕ), prime p ∧ n = p ^ 3))

def first_ten_super_numbers : list ℕ := [6, 10, 14, 15, 21, 22, 30, 8, 27, 125]

theorem sum_of_first_ten_super_numbers : 
  list.sum first_ten_super_numbers = 278 :=
by
  sorry

end sum_of_first_ten_super_numbers_l117_117552


namespace scientific_notation_correct_l117_117790

-- Define the input number
def input_number : ℕ := 858000000

-- Define the expected scientific notation result
def scientific_notation (n : ℕ) : ℝ := 8.58 * 10^8

-- The theorem states that the input number in scientific notation is indeed 8.58 * 10^8
theorem scientific_notation_correct :
  scientific_notation input_number = 8.58 * 10^8 :=
sorry

end scientific_notation_correct_l117_117790


namespace identical_digit_square_l117_117365

theorem identical_digit_square {b x y : ℕ} (hb : b ≥ 2) (hx : x < b) (hy : y < b) (hx_pos : x ≠ 0) (hy_pos : y ≠ 0) :
  (x * b + x)^2 = y * b^3 + y * b^2 + y * b + y ↔ b = 7 :=
by
  sorry

end identical_digit_square_l117_117365


namespace compound_interest_time_l117_117083

noncomputable def simple_interest (P R T : ℝ) := P * R * T / 100

noncomputable def compound_interest (P R n : ℝ) := P * ((1 + R / 100)^n - 1)

theorem compound_interest_time
  (P₁ P₂ SI CI R₁ T₁ R₂ : ℝ)
  (h₁ : simple_interest P₁ R₁ T₁ = SI)
  (h₂ : SI = CI / 2)
  (h₃ : compound_interest P₂ R₂ ?n = CI) :
  ?n = 2 := 
sorry

end compound_interest_time_l117_117083


namespace base_conversion_addition_correct_l117_117590

noncomputable def base_8_to_base_10 (n : Nat) : ℕ :=
  match n with
  | 254 => 2 * (8 ^ 2) + 5 * (8 ^ 1) + 4 * (8 ^ 0)
  | _ => 0

noncomputable def base_3_to_base_10 (n : Nat) : ℕ :=
  match n with
  | 13 => 1 * (3 ^ 1) + 3 * (3 ^ 0)
  | _ => 0

noncomputable def base_5_to_base_10 (n : Nat) : ℕ :=
  match n with
  | 202 => 2 * (5 ^ 2) + 0 * (5 ^ 1) + 2 * (5 ^ 0)
  | _ => 0

noncomputable def base_4_to_base_10 (n : Nat) : ℕ :=
  match n with
  | 22 => 2 * (4 ^ 1) + 2 * (4 ^ 0)
  | _ => 0

theorem base_conversion_addition_correct :
  (base_8_to_base_10 254) / (base_3_to_base_10 13) + (base_5_to_base_10 202) / (base_4_to_base_10 22) = 39.6 :=
by
  sorry

end base_conversion_addition_correct_l117_117590


namespace factorize_poly1_min_value_poly2_l117_117199

-- Define the polynomials
def poly1 := fun (x : ℝ) => x^2 + 2 * x - 3
def factored_poly1 := fun (x : ℝ) => (x - 1) * (x + 3)

def poly2 := fun (x : ℝ) => x^2 + 4 * x + 5
def min_value := 1

-- State the theorems without providing proofs
theorem factorize_poly1 : ∀ x : ℝ, poly1 x = factored_poly1 x := 
by { sorry }

theorem min_value_poly2 : ∀ x : ℝ, poly2 x ≥ min_value := 
by { sorry }

end factorize_poly1_min_value_poly2_l117_117199


namespace find_a_l117_117284

def f : ℝ → ℝ := sorry

theorem find_a (x a : ℝ) 
  (h1 : ∀ x, f ((1/2)*x - 1) = 2*x - 5)
  (h2 : f a = 6) : 
  a = 7/4 := 
by 
  sorry

end find_a_l117_117284


namespace translation_up_4_units_l117_117188

def f (x : ℝ) : ℝ := -2 * x + 1

def translate_up (y_translation : ℝ) (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x, f x + y_translation

theorem translation_up_4_units (x : ℝ) : translate_up 4 f x = -2 * x + 5 :=
by
  sorry

end translation_up_4_units_l117_117188


namespace fixed_points_at_most_n_l117_117386

theorem fixed_points_at_most_n {n : ℕ} (h_n : n > 1) (P : polynomial ℤ) (hP : P.degree = n) (k : ℕ) (hk : k > 0) :
  ∃ m ≤ n, ∀ t : ℤ, ¬(Q t = t) ∨ m ≤ n :=
sorry

def Q (P : polynomial ℤ) (k : ℕ) (x : ℤ) : ℤ :=
if k = 0 then x else Q P (k - 1) (P.eval x)

end fixed_points_at_most_n_l117_117386


namespace cookies_bought_l117_117168

/-
Given conditions:
- Each paper bag can hold 16 cookies.
- Edgar needs 19 paper bags.
Question:
- How many cookies did Edgar buy?
Answer:
- Edgar bought 304 cookies.
-/

theorem cookies_bought (bags cookies_per_bag : ℕ) (h_bags : bags = 19) (h_cookies_per_bag : cookies_per_bag = 16) : bags * cookies_per_bag = 304 :=
by
  rw [h_bags, h_cookies_per_bag]
  norm_num
  exact rfl

end cookies_bought_l117_117168


namespace subtraction_calculation_l117_117107

theorem subtraction_calculation (a b : ℤ) (h : b = 40) (h1 : a = b - 1) : (a * a) = (b * b) - 79 := 
by
  -- Using the given condition
  have h2 : a * a = (b - 1) * (b - 1),
  from by rw [h1],
  -- Expanding using binomial theorem
  rw [mul_sub, sub_mul, mul_one, ← square_eq, sub_sub, one_mul, one_mul] at h2,
  -- Proving the theorem
  rw [sub_add] at h2,
  exact h2,
  sorry

end subtraction_calculation_l117_117107


namespace gcd_10010_15015_l117_117244

theorem gcd_10010_15015 :
  let n1 := 10010
  let n2 := 15015
  ∃ d, d = Nat.gcd n1 n2 ∧ d = 5005 :=
by
  let n1 := 10010
  let n2 := 15015
  -- ... omitted proof steps
  sorry

end gcd_10010_15015_l117_117244


namespace Harold_tips_l117_117320

theorem Harold_tips (A x : ℝ) (H_total: xA = 0.5 * (xA + 6A)) : x = 6 :=
by
  sorry

end Harold_tips_l117_117320


namespace count_expressible_integers_l117_117604

theorem count_expressible_integers :
  let S := { n | ∃ (x : ℝ), n = Int.floor x + Int.floor (2 * x) + Int.floor (3 * x) + Int.floor (4 * x) } in
  S.count (finset.Icc 1 1000) = 600 :=
by
  sorry

end count_expressible_integers_l117_117604


namespace operation_B_correct_l117_117489

theorem operation_B_correct : 3 / Real.sqrt 3 = Real.sqrt 3 :=
  sorry

end operation_B_correct_l117_117489


namespace range_of_c_l117_117994

theorem range_of_c (c : ℝ) (h_c : c > 0) :
  ((∀ x : ℝ, y = c^x → y decreases) ∨ (∀ x : ℝ, f x = x + c > 0)) ∧ 
  ¬ ((∀ x : ℝ, y = c^x → y decreases) ∧ (∀ x : ℝ, f x = x + c > 0)) →
  c > 0 ∧ c ≠ 1 :=
sorry

end range_of_c_l117_117994


namespace minimum_value_sum_fractions_inequality_l117_117388

theorem minimum_value_sum_fractions_inequality (x : Fin 150 → ℝ) 
  (hx_pos : ∀ i, 0 < x i) 
  (hx_sum_square : ∑ i in Finset.univ, (x i) ^ 2 = 2) : 
  (∑ i in Finset.univ, x i / (2 - (x i) ^ 2)) ≥ 3 * Real.sqrt 3 := 
sorry

end minimum_value_sum_fractions_inequality_l117_117388


namespace nathan_earnings_l117_117758

-- Conditions
def num_strwb_plants : ℕ := 5
def num_tmt_plants : ℕ := 7
def strwb_per_plant : ℕ := 14
def tmt_per_plant : ℕ := 16
def basket_capacity : ℕ := 7
def price_per_strwb_basket : ℕ := 9
def price_per_tmt_basket : ℕ := 6

-- Outcome to prove
theorem nathan_earnings :
  let total_strwb := num_strwb_plants * strwb_per_plant,
      total_tmt := num_tmt_plants * tmt_per_plant,
      strwb_baskets := total_strwb / basket_capacity,
      tmt_baskets := total_tmt / basket_capacity,
      earnings_strwb := strwb_baskets * price_per_strwb_basket,
      earnings_tmt := tmt_baskets * price_per_tmt_basket
  in earnings_strwb + earnings_tmt = 186 :=
by {
  sorry -- The proof is absent, as directed.
}

end nathan_earnings_l117_117758


namespace max_omega_l117_117303

-- Definitions based on conditions
def f (x : ℝ) (ω : ℝ) : ℝ := cos (ω * x - π / 3)
def interval := Ioc (2 * π / 3) (5 * π / 6)

-- The problem statement to prove
theorem max_omega (ω : ℝ) :
  (ω > 0) →
  (f (2 * π / 3) ω = f (5 * π / 6) ω) →
  (∀ x ∈ interval, f x ω.has_max f x ω.has_min) →
  ω = 100 / 9 :=
sorry

end max_omega_l117_117303


namespace fish_left_in_sea_l117_117507

-- Definitions based on conditions
def total_fish_westward : Nat := 1800
def total_fish_eastward : Nat := 3200
def total_fish_north : Nat := 500

def caught_fraction_westward : Rat := 3/4
def caught_fraction_eastward : Rat := 2/5

-- Theorem statement
theorem fish_left_in_sea : 
  let fish_left_westward := total_fish_westward - (caught_fraction_westward * total_fish_westward).nat
  let fish_left_eastward := total_fish_eastward - (caught_fraction_eastward * total_fish_eastward).nat
  let fish_left_north := total_fish_north
  fish_left_westward + fish_left_eastward + fish_left_north = 2870 := 
by
  -- Placeholder for proof
  sorry

end fish_left_in_sea_l117_117507


namespace avg_distinct_k_values_l117_117292

open_locale big_operators

theorem avg_distinct_k_values :
  ∀ k : ℕ, (∃ r1 r2 : ℕ, r1 * r2 = 36 ∧ r1 + r2 = k) →
  (∑ k in ({37, 20, 15, 13, 12} : finset ℕ), k) / 5 = 19.4 :=
by { sorry }

end avg_distinct_k_values_l117_117292


namespace find_speed_of_first_part_of_trip_l117_117165

-- Define the conditions as hypotheses and state the final theorem to be proven
theorem find_speed_of_first_part_of_trip : ∃ v : ℝ, 
  let distance_first_part := 30 in
  let distance_second_part := 30 in
  let total_distance := 60 in
  let speed_second_part := 24 in
  let avg_speed := 32 in
  let time_first_part := distance_first_part / v in
  let time_second_part := distance_second_part / speed_second_part in
  let total_time := total_distance / avg_speed in
  time_first_part + time_second_part = total_time → v = 60 :=
by
  sorry

end find_speed_of_first_part_of_trip_l117_117165


namespace students_present_l117_117830

theorem students_present (absent_students male_students female_student_diff : ℕ) 
  (h1 : absent_students = 18) 
  (h2 : male_students = 848) 
  (h3 : female_student_diff = 49) : 
  (male_students + (male_students - female_student_diff) - absent_students = 1629) := 

by 
  sorry

end students_present_l117_117830


namespace opposite_of_neg_2023_l117_117062

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117062


namespace proof_of_equivalence_l117_117202

variables (x y : ℝ)

def expression := 49 * x^2 - 36 * y^2
def optionD := (-6 * y + 7 * x) * (6 * y + 7 * x)

theorem proof_of_equivalence : expression x y = optionD x y := 
by sorry

end proof_of_equivalence_l117_117202


namespace find_volume_of_prism_l117_117914

def volume_of_prism_inscribed_in_sphere (r_sphere : ℝ) (AD : ℝ) : ℝ :=
  let d_sphere := 2 * r_sphere
  let h := (4 * sqrt 5 : ℝ)
  let side_length := (4 * sqrt 3 : ℝ)
  let base_area := (sqrt 3 / 4 * (4 * sqrt 3)^2 : ℝ)
  base_area * h

theorem find_volume_of_prism :
  volume_of_prism_inscribed_in_sphere 6 (4 * sqrt 6) = 48 * sqrt 15 :=
by
  sorry

end find_volume_of_prism_l117_117914


namespace bricks_required_to_pave_courtyard_l117_117155

theorem bricks_required_to_pave_courtyard :
  let courtyard_length : ℝ := 25
  let courtyard_width : ℝ := 16
  let brick_length : ℝ := 0.20
  let brick_width : ℝ := 0.10
  let area_courtyard := courtyard_length * courtyard_width
  let area_brick := brick_length * brick_width
  let number_of_bricks := area_courtyard / area_brick
  number_of_bricks = 20000 := by
    let courtyard_length : ℝ := 25
    let courtyard_width : ℝ := 16
    let brick_length : ℝ := 0.20
    let brick_width : ℝ := 0.10
    let area_courtyard := courtyard_length * courtyard_width
    let area_brick := brick_length * brick_width
    let number_of_bricks := area_courtyard / area_brick
    sorry

end bricks_required_to_pave_courtyard_l117_117155


namespace opposite_of_neg_2023_l117_117042

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117042


namespace alex_walking_distance_correct_l117_117543

def average_speed_on_flat_ground := 20
def time_on_flat_ground := 4.5
def average_speed_uphill := 12
def time_uphill := 2.5
def average_speed_downhill := 24
def time_downhill := 1.5
def total_distance_to_next_town := 164

def distance_covered_flat := average_speed_on_flat_ground * time_on_flat_ground
def distance_covered_uphill := average_speed_uphill * time_uphill
def distance_covered_downhill := average_speed_downhill * time_downhill

def total_distance_covered_before_puncture := distance_covered_flat + distance_covered_uphill + distance_covered_downhill

def distance_to_walk := total_distance_to_next_town - total_distance_covered_before_puncture

theorem alex_walking_distance_correct : distance_to_walk = 8 := by
  unfold distance_to_walk
  unfold total_distance_to_next_town
  unfold total_distance_covered_before_puncture
  unfold distance_covered_flat distance_covered_uphill distance_covered_downhill
  unfold average_speed_on_flat_ground time_on_flat_ground
  unfold average_speed_uphill time_uphill
  unfold average_speed_downhill time_downhill
  norm_num
  sorry

end alex_walking_distance_correct_l117_117543


namespace bulbs_always_on_l117_117831

theorem bulbs_always_on (n : ℕ) (h : n ≥ 3) : 
  ∃ initial_state : vector bool n, ∀ t : ℕ, ∃ i : ℕ, i < n ∧ (bulb_state n initial_state t) i = true :=
sorry

where
  -- Function to determine the state of the bulbs at time t
  bulb_state : Π (n : ℕ) (initial_state : vector bool n), ℕ → vector bool n
  bulb_state n initial_state 0 := initial_state
  bulb_state n initial_state (t+1) :=
    vector.of_fn (λ i, (is_on_next (n, bulb_state n initial_state t) i))

  -- Helper function to determine if a bulb will be on in the next state
  is_on_next (n : ℕ) (current_state : vector bool n) (i : ℕ) : bool :=
    if current_state.nth i = tt then ff
    else if i = 0 then current_state.nth 1
    else if i = n-1 then current_state.nth (n-2)
    else current_state.nth (i-1) ≠ current_state.nth (i+1)
  

end bulbs_always_on_l117_117831


namespace mark_has_3_tanks_l117_117752

-- Define conditions
def pregnant_fish_per_tank : ℕ := 4
def young_per_fish : ℕ := 20
def total_young : ℕ := 240

-- Theorem statement that Mark has 3 tanks
theorem mark_has_3_tanks : (total_young / (pregnant_fish_per_tank * young_per_fish)) = 3 :=
by
  sorry

end mark_has_3_tanks_l117_117752


namespace area_of_PQR_is_32_l117_117854

/-- Triangle PQR is an isosceles right triangle with angle P = 90 degrees, and the length of segment PR = 8 cm. The goal is to prove that the area of triangle PQR is 32 square centimeters. -/
theorem area_of_PQR_is_32 
  (P Q R : Type) 
  [triangle P Q R] 
  (right_angle_P : angle P = 90)
  (isosceles_right : isosceles_right_triangle P Q R P Q P R)
  (length_PR : segment_length P R = 8) 
  : area_triangle P Q R = 32 := 
sorry

end area_of_PQR_is_32_l117_117854


namespace largest_unique_base_polynomial_l117_117172

theorem largest_unique_base_polynomial (c : ℝ) :
  ∃! (P : ℝ → ℝ), (∃ (d : ℕ) (a : ℕ → ℤ), 
    (∀ i, 0 ≤ a i ∧ a i < 9) ∧ 
    (0 < a d) ∧ 
    P = λ x, (finset.range (d+1)).sum (λ i, a i * x^i)) ∧ 
    P (Real.sqrt 2 + Real.sqrt 3) = c :=
begin
  sorry
end

end largest_unique_base_polynomial_l117_117172


namespace sin_angle_ADB_l117_117358

variable (A B C D : Point)
variables (x y : ℝ)
variables (angle_ABC angle_ACB angle_BAC angle_CAD angle_CBD angle_ADB : ℝ)

-- Conditions
variable (h1 : angle_ABC = 90)
variable (h2 : angle_ACB = 90)
variable (h3 : angle_BAC = 90)
variable (hx : x = cos angle_CAD)
variable (hy : y = cos angle_CBD)

-- Proof statement
theorem sin_angle_ADB (A B C D : Point) (x y : ℝ) 
  (angle_ABC angle_ACB angle_BAC angle_CAD angle_CBD angle_ADB : ℝ)
  (h1 : angle_ABC = 90) (h2 : angle_ACB = 90) (h3 : angle_BAC = 90)
  (hx : x = cos angle_CAD) (hy : y = cos angle_CBD) : 
  sin angle_ADB = x * y / sqrt (x^2 + y^2) := 
sorry

end sin_angle_ADB_l117_117358


namespace intersection_point_exists_l117_117786

def equation_1 (x y : ℝ) : Prop := 3 * x^2 - 12 * y^2 = 48
def line_eq (x y : ℝ) : Prop := y = - (1 / 3) * x + 5

theorem intersection_point_exists :
  ∃ (x y : ℝ), equation_1 x y ∧ line_eq x y ∧ x = 75 / 8 ∧ y = 15 / 8 :=
sorry

end intersection_point_exists_l117_117786


namespace calculate_expression_l117_117940

theorem calculate_expression :
  ((16^10 / 16^8) ^ 3 * 8 ^ 3) / 2 ^ 9 = 16777216 := by
  sorry

end calculate_expression_l117_117940


namespace opposite_of_negative_2023_l117_117048

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117048


namespace smallest_positive_period_monotonically_increasing_interval_triangle_side_lengths_l117_117666

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 3 * sin (π / 2 - x) * cos (3 * π / 2 + x) + sin x ^ 2

theorem smallest_positive_period :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π := sorry

theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x, k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3 →
  ∀ y, k * π - π / 6 ≤ y ∧ y ≤ k * π + π / 3 →
  (∀ a b, a ≤ b → x = a ∧ y = b → f a ≤ f b) := sorry

structure triangle :=
(a b c : ℝ)
(lt_ab_c : a > 0 ∧ b > 0 ∧ c > 0)
(area : ℝ)
(angles : ℝ)

noncomputable def triangle_C_value : ℝ := π / 3

def side_values (t : triangle) : Prop :=
  t.c = sqrt 3 ∧ f (t.angles) = 3 / 2 ∧
  t.area = sqrt 3 / 2 ∧
  (t.a * t.b = 2 ∧ t.a^2 + t.b^2 - t.a * t.b = 3)

theorem triangle_side_lengths (t : triangle) :
  side_values t → (t.a = 1 ∧ t.b = 2) ∨ (t.a = 2 ∧ t.b = 1) := sorry

end smallest_positive_period_monotonically_increasing_interval_triangle_side_lengths_l117_117666


namespace shift_graph_correct_l117_117842

/-- To obtain the graph of the function y = sin 3x + cos 3x one shifts the graph of y = sqrt(2) cos 3x to the right by π/12 units. -/
theorem shift_graph_correct :
  ∀ x : ℝ, sin (3 * x) + cos (3 * x) = sqrt 2 * cos (3 * (x - π / 12)) :=
by
  sorry

end shift_graph_correct_l117_117842


namespace range_of_fraction_l117_117449

theorem range_of_fraction (x1 y1 : ℝ) (h1 : y1 = -2 * x1 + 8) (h2 : 2 ≤ x1 ∧ x1 ≤ 5) :
  -1/6 ≤ (y1 + 1) / (x1 + 1) ∧ (y1 + 1) / (x1 + 1) ≤ 5/3 :=
sorry

end range_of_fraction_l117_117449


namespace valid_permutations_count_correct_l117_117713

open finset

def valid_permutations_count : ℕ :=
  (univ : finset (perm (fin 5))).filter
    (λ σ, σ 0 < σ 1 ∧ σ 1 > σ 2 ∧ σ 2 < σ 3 ∧ σ 3 > σ 4).card

theorem valid_permutations_count_correct :
  valid_permutations_count = 16 :=
by sorry

end valid_permutations_count_correct_l117_117713


namespace find_2018th_number_l117_117406

-- Definition for the sequence
def sequence : ℕ → ℕ
| n => 
let k := n / 6
let p := n % 6
k + 1 + if p = 0 ∨ p = 1 then 0 else if p = 2 ∨ p = 4 then 1 else if p = 3 then 2 else 3

-- Statement of the main proof problem
theorem find_2018th_number : sequence 2017 = 338 :=
  sorry

end find_2018th_number_l117_117406


namespace card_number_C_l117_117221

theorem card_number_C (A B C : ℤ) 
  (h1 : (A + B + C) / 3 = 143)
  (h2 : A + 4.5 = (B + C) / 2)
  (h3 : C = B - 3) : 
  C = 143 :=
by
  sorry

end card_number_C_l117_117221


namespace sample_size_ratio_l117_117514

theorem sample_size_ratio (n : ℕ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
                          (total_ratio : ℕ) (B_in_sample : ℕ)
                          (h_ratio : ratio_A = 1 ∧ ratio_B = 3 ∧ ratio_C = 5)
                          (h_total : total_ratio = ratio_A + ratio_B + ratio_C)
                          (h_B_sample : B_in_sample = 27)
                          (h_sampling_ratio_B : ratio_B / total_ratio = 1 / 3) :
                          n = 81 :=
by sorry

end sample_size_ratio_l117_117514


namespace problem1_problem2_l117_117681

noncomputable def vector_a : ℝ × ℝ := (1, -2)
noncomputable def vector_b : ℝ × ℝ := (-1, -1)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def add_scalar_mult (λ : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 - λ, -2 - λ)

theorem problem1 : magnitude (2 • vector_a - vector_b) = 3 * real.sqrt 2 := sorry

theorem problem2 (λ : ℝ) (h : real.angle (add_scalar_mult λ vector_b) (2 • vector_a - vector_b) = real.pi / 4) :
  λ = 1 ∨ λ = -2 := sorry

end problem1_problem2_l117_117681


namespace mall_b_better_for_fewer_than_6_mall_equal_for_6_mall_a_better_for_more_than_6_l117_117895

-- Definitions
def original_price : ℕ := 80
def discount_mallA (n : ℕ) : ℕ := min ((4 * n) * n) (80 * n / 2)
def discount_mallB (n : ℕ) : ℕ := (80 * n * 3) / 10

def total_cost_mallA (n : ℕ) : ℕ := (original_price * n) - discount_mallA n
def total_cost_mallB (n : ℕ) : ℕ := (original_price * n) - discount_mallB n

-- Theorem statements
theorem mall_b_better_for_fewer_than_6 (n : ℕ) (h : n < 6) : total_cost_mallA n > total_cost_mallB n := sorry
theorem mall_equal_for_6 (n : ℕ) (h : n = 6) : total_cost_mallA n = total_cost_mallB n := sorry
theorem mall_a_better_for_more_than_6 (n : ℕ) (h : n > 6) : total_cost_mallA n < total_cost_mallB n := sorry

end mall_b_better_for_fewer_than_6_mall_equal_for_6_mall_a_better_for_more_than_6_l117_117895


namespace rationalize_denominator_l117_117421

theorem rationalize_denominator :
  (35 / Real.sqrt 35) = Real.sqrt 35 :=
sorry

end rationalize_denominator_l117_117421


namespace fish_left_in_sea_l117_117509

theorem fish_left_in_sea : 
  let westward_initial := 1800
  let eastward_initial := 3200
  let north_initial := 500
  let eastward_caught := (2 / 5) * eastward_initial
  let westward_caught := (3 / 4) * westward_initial
  let eastward_left := eastward_initial - eastward_caught
  let westward_left := westward_initial - westward_caught
  let north_left := north_initial
  eastward_left + westward_left + north_left = 2870 := 
by 
  sorry

end fish_left_in_sea_l117_117509


namespace Gideon_age_l117_117456
-- Import necessary libraries

-- Define the initial conditions and the final proof statement
theorem Gideon_age :
  ∀ (total_marbles gideon_age_now : ℕ) (frac : ℚ),
    total_marbles = 100 →
    frac = 3 / 4 →
    let marbles_given_away := (frac * total_marbles).to_nat in
    let remaining_marbles := total_marbles - marbles_given_away in
    let age_five_years_from_now := 2 * remaining_marbles in
    age_five_years_from_now = gideon_age_now + 5 →
    gideon_age_now = 45 :=
by
  intros total_marbles gideon_age_now frac H1 H2 H3
  sorry

end Gideon_age_l117_117456


namespace pyramid_height_l117_117528

noncomputable def height_pyramid (perimeter_base : ℝ) (distance_apex_vertex : ℝ) : ℝ :=
  let side_length := perimeter_base / 4
  let half_diagonal := (side_length * Real.sqrt 2) / 2
  Real.sqrt (distance_apex_vertex ^ 2 - half_diagonal ^ 2)

theorem pyramid_height
  (perimeter_base: ℝ)
  (h_perimeter : perimeter_base = 32)
  (distance_apex_vertex: ℝ)
  (h_distance : distance_apex_vertex = 10) :
  height_pyramid perimeter_base distance_apex_vertex = 2 * Real.sqrt 17 :=
by
  sorry

end pyramid_height_l117_117528


namespace projection_of_rectangle_one_side_parallel_to_plane_l117_117526

def is_projection_of_rectangle_side_parallel_to_plane (shape : Type) 
(rect : shape → Prop) (parallel_side : shape → Prop) : Prop :=
∀ s : shape, rect s → parallel_side s → (s = "rectangle" ∨ s = "line_segment")

-- Prove that the projection of a rectangle with one side parallel to the plane 
-- can either be a rectangle or a line segment.
theorem projection_of_rectangle_one_side_parallel_to_plane : 
  ∀ s : Type, (is_projection_of_rectangle_side_parallel_to_plane s (λ x, true) (λ x, true)):
  sorry

end projection_of_rectangle_one_side_parallel_to_plane_l117_117526


namespace triangle_constructability_impossible_l117_117678

theorem triangle_constructability_impossible (a α : ℝ) (ABC : Type) [triangle ABC] 
  (ha : side (opposite ABC α) = a) (h_bisector_C : angle_bisector_length C = a) : 
  ¬ (constructible_with_euclidean_methods ABC a α) :=
sorry

end triangle_constructability_impossible_l117_117678


namespace division_by_fraction_l117_117201

theorem division_by_fraction :
  12 / (3 / 5 : ℚ) = 20 :=
by sorry

end division_by_fraction_l117_117201


namespace unique_function_zero_l117_117961

theorem unique_function_zero (f : (UnitInterval) → ℝ) :
  (∀ x y : UnitInterval, f (x * y) = x * f x + y * f y) → (∀ x : UnitInterval, f x = 0) :=
by
  -- Here we will provide the proof
  sorry

end unique_function_zero_l117_117961


namespace sequence_bound_exists_l117_117953

def sequence (n : ℕ) : ℕ → ℝ
| 0 := 5
| (n + 1) := (sequence n ^ 2 + 5 * sequence n + 4) / (sequence n + 6)

theorem sequence_bound_exists :
  ∃ m : ℕ, 81 ≤ m ∧ m ≤ 242 ∧ sequence m ≤ 4 + 1 / 2^20 :=
sorry

end sequence_bound_exists_l117_117953


namespace x_lt_1_nec_not_suff_l117_117998

theorem x_lt_1_nec_not_suff (x : ℝ) : (x < 1 → x^2 < 1) ∧ (¬(x < 1) → x^2 < 1) := 
by {
  sorry
}

end x_lt_1_nec_not_suff_l117_117998


namespace total_tires_on_lot_l117_117187

def total_vehicles := 75
def perc_motorcycles := 0.40
def perc_cars_with_spare := 0.30

def motorcycles := perc_motorcycles * total_vehicles
def cars := total_vehicles - motorcycles

def cars_with_spare := Int.floor (perc_cars_with_spare * cars)
def cars_without_spare := cars - cars_with_spare

def motorcycle_tires := 2 * motorcycles
def car_tires_without_spare := 4 * cars_without_spare
def car_tires_with_spare := 5 * cars_with_spare

def total_tires := motorcycle_tires + car_tires_without_spare + car_tires_with_spare

theorem total_tires_on_lot : total_tires = 253 :=
by
  sorry

end total_tires_on_lot_l117_117187


namespace ellipse_foci_projections_on_tangent_circle_ellipse_foci_distances_product_constant_l117_117492

-- Definition of the ellipse and its properties
structure Ellipse (a b c : ℝ) :=
  (f₁ : ℝ × ℝ := (-c, 0))
  (f₂ : ℝ × ℝ := (c, 0))
  (is_ellipse : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)

-- Part (a): Projections of the foci of an ellipse onto all tangent lines lie on one circle
theorem ellipse_foci_projections_on_tangent_circle (a b c : ℝ) (e : Ellipse a b c) :
  ∃ r : ℝ, ∀ T : ℝ × ℝ, let l := λ x y : ℝ, (x * T.1 / a^2) + (y * T.2 / b^2) = 1 in 
  ∀ P : ℝ × ℝ, (P.1^2 + P.2^2 = r^2) := 
sorry

-- Part (b): The product of distances from the foci of the ellipse to a tangent line 
-- does not depend on the choice of the tangent line
theorem ellipse_foci_distances_product_constant (a b c : ℝ) (e : Ellipse a b c) (d₁ d₂ : ℝ) :
  ∀ (l : ℝ → ℝ → Prop) (tangent_line : ∀ x y : ℝ, l x y), 
  (d₁ * d₂ = constant) := 
sorry

end ellipse_foci_projections_on_tangent_circle_ellipse_foci_distances_product_constant_l117_117492


namespace shaded_area_converges_l117_117353

noncomputable def equilateral_triangle_area (side_length : ℝ) : ℝ :=
  (sqrt 3 / 4) * side_length ^ 2

noncomputable def total_shaded_area (initial_side_length : ℝ) : ℝ :=
  let initial_area := equilateral_triangle_area initial_side_length
  let first_shaded_area := initial_area / 4
  let common_ratio := 1 / 4
  first_shaded_area / (1 - common_ratio)

theorem shaded_area_converges :
  total_shaded_area 10 = 25 * sqrt 3 / 3 :=
by
  -- Sorry, proof not required.
  sorry

end shaded_area_converges_l117_117353


namespace cot_theta_in_terms_of_x_l117_117741

theorem cot_theta_in_terms_of_x (θ : ℝ) (x : ℝ) (h1 : 0 < θ ∧ θ < π / 2) 
    (h2 : sin (θ / 2) = Real.sqrt ((2 * x - 1) / (3 * x))) : 
    Real.cot θ = (2 - x) / Real.sqrt (x^2 + 8 * x - 4) :=
sorry

end cot_theta_in_terms_of_x_l117_117741


namespace increasing_function_range_iff_l117_117339

def f (a : ℝ) (x : ℝ) : ℝ :=
if x >= 1 then (2 * a + 3) * x - 4 * a + 3 else a ^ x

theorem increasing_function_range_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (1 < a ∧ a ≤ 2) :=
sorry

end increasing_function_range_iff_l117_117339


namespace platform_length_correct_l117_117130

-- Define the given conditions
def train_length : ℝ := 200
def train_speed_kmph : ℝ := 72
def crossing_time : ℝ := 25

-- Converting the train speed from kmph to m/s
def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600

-- Define the total distance covered by the train while crossing the platform
def total_distance_covered := train_speed_mps * crossing_time

-- Define the statement of the length of the platform
def platform_length : ℝ := total_distance_covered - train_length

-- The theorem stating the correct length of the platform
theorem platform_length_correct : 
  platform_length = 300 := by
  sorry

end platform_length_correct_l117_117130


namespace angle_A_triangle_shape_l117_117718

noncomputable def vector_m (A : ℝ) : ℝ × ℝ := (Real.cos (3 * A / 2), Real.sin (3 * A / 2))
noncomputable def vector_n (A : ℝ) : ℝ × ℝ := (Real.cos (A / 2), Real.sin (A / 2))
noncomputable def vector_sum_m_n (A : ℝ) : ℝ := 
  Real.sqrt (vector_m A.1 ^ 2 + vector_m A.2 ^ 2 + 2 * (vector_m A.1 * vector_n A.1 + vector_m A.2 * vector_n A.2))

theorem angle_A (A : ℝ) (h : vector_sum_m_n A = Real.sqrt 3) : A = Real.pi / 3 :=
sorry

noncomputable def side_length (A : ℝ) (b c : ℝ) : ℝ := b + c
noncomputable def scaled_side_length (a b c : ℝ) : ℝ := Real.sqrt 3 * a

theorem triangle_shape (A : ℝ) (a b c : ℝ) (hA : A = Real.pi / 3) 
  (h : side_length A b c = scaled_side_length a b c) : 
  (B C : ℝ) (hB : B = Real.pi/6 ∨ B = Real.pi/2) (hC : C = Real.pi/2 ∨ C = Real.pi/6) : 
  B = Real.pi/2 ∨ C = Real.pi/2 :=
sorry

end angle_A_triangle_shape_l117_117718


namespace smallest_whole_number_larger_than_perimeter_l117_117120

-- Define the sides of the triangle
def side1 : ℕ := 7
def side2 : ℕ := 23

-- State the conditions using the triangle inequality theorem
def triangle_inequality_satisfied (s : ℕ) : Prop :=
  (side1 + side2 > s) ∧ (side1 + s > side2) ∧ (side2 + s > side1)

-- The proof statement
theorem smallest_whole_number_larger_than_perimeter
  (s : ℕ) (h : triangle_inequality_satisfied s) : 
  ∃ n : ℕ, n = 60 ∧ ∀ p : ℕ, (p > side1 + side2 + s) → (p ≥ n) :=
sorry

end smallest_whole_number_larger_than_perimeter_l117_117120


namespace ash_cloud_radius_l117_117537

noncomputable def calculate_radius (deg : ℝ) (factor : ℝ) (height_meters : ℝ) (conversion_factor : ℝ) : ℝ :=
    let hypotenuse := height_meters / (Real.sin (Real.pi * deg / 180))
    let diameter := factor * hypotenuse
    let radius_meters := diameter / 2
    let radius_feet := radius_meters * conversion_factor
    radius_feet

theorem ash_cloud_radius :
  calculate_radius 60 18 300 3.28084 ≈ 10228.74 :=
by
  sorry

end ash_cloud_radius_l117_117537


namespace smallest_001_multiple_of_72_has_12_digits_l117_117170

-- Define a function that verifies if a number meets the required conditions
def is_valid_001_integer (n : ℕ) : Prop :=
  (∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 1) ∧
  (72 ∣ n)

-- Define the smallest valid integer that satisfies the conditions
def smallest_001_multiple_of_72 := 111111111000

-- The main theorem statement: proving the number of digits of the smallest valid integer
theorem smallest_001_multiple_of_72_has_12_digits :
  is_valid_001_integer smallest_001_multiple_of_72 ∧
  smallest_001_multiple_of_72.digits 10 = 12 := by
    sorry

end smallest_001_multiple_of_72_has_12_digits_l117_117170


namespace remaining_macaroons_weight_l117_117726

theorem remaining_macaroons_weight (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (total_bags : ℕ) :
  (total_macaroons = 12) → 
  (weight_per_macaroon = 5) → 
  (total_bags = 4) → 
  let macaroons_per_bag := total_macaroons / total_bags in
  let weight_per_bag := macaroons_per_bag * weight_per_macaroon in
  let weight_eaten_by_steve := weight_per_bag in
  let total_weight := total_macaroons * weight_per_macaroon in
  let remaining_weight := total_weight - weight_eaten_by_steve in
  remaining_weight = 45 :=
by {
  sorry
}

end remaining_macaroons_weight_l117_117726


namespace tan_intersection_points_l117_117667

theorem tan_intersection_points
  (f : ℝ → ℝ)
  (k x1 x2 x3 : ℝ)
  (h1 : f = (λ x, Real.sin (2 * x)))
  (h2 : ❨2 * k * x - 2 * (f x) - k * Real.pi = 0 → ∃ x1 x2 x3, x1 < x2 ∧ x2 < x3 ∧
        f x1 = Real.sin (2 * x1) ∧ f x2 = Real.sin (2 * x2) ∧ f x3 = Real.sin (2 * x3)❩)
  (h3 : k > 0) :
  (x1 - x3) * Real.tan (x2 - 2 * x3) = -1 := sorry

end tan_intersection_points_l117_117667


namespace parabola_vertex_coordinates_l117_117807

theorem parabola_vertex_coordinates :
  (∃ x : ℝ, (λ x, x^2 - 2) = (0, -2)) :=
sorry

end parabola_vertex_coordinates_l117_117807


namespace opposite_of_neg_2023_l117_117074

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117074


namespace length_of_EF_l117_117363

theorem length_of_EF :
  ∀ (A B C D F : Type)
  (AB AC : ℝ) (∠BAC : ℝ)
  (DE DF : ℝ) (∠EDF : ℝ)
  (area_ABC : ℝ),
  AB = 5 → AC = 6 → ∠BAC = 30 → 
  DE = 2 → DF = 3 → ∠EDF = 45 →
  area_ABC = 7.5 →
  2 * (1 / 2 * DE * DF * real.sin (real.pi / 4)) = area_ABC →
  3 * real.sqrt 2 / 2 = 3.75 →
  ∃ EF : ℝ, EF = 3.75 :=
by {
  intros,
  sorry
}

end length_of_EF_l117_117363


namespace find_a_find_monotonic_intervals_l117_117306

noncomputable def f (x : ℝ) (a c : ℝ) : ℝ := x^3 + a * x^2 - x + c

def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x - 1

theorem find_a (c : ℝ) : f' (2 / 3) (f' (2 / 3) - 1) = (f' (2 / 3) - 1) := by
  sorry

def f_a_minus_1 (x c : ℝ) : ℝ := x^3 - x^2 - x + c

def f'_a_minus_1 (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 1

theorem find_monotonic_intervals (c : ℝ) :
  (∀ x, (f'_a_minus_1 x > 0) ↔ (x > 1 ∨ x < -1 / 3)) ∧ 
  (∀ x, (f'_a_minus_1 x < 0) ↔ (-1 / 3 < x ∧ x < 1)) := by
  sorry

end find_a_find_monotonic_intervals_l117_117306


namespace person_covers_escalator_in_10_seconds_l117_117193

/-- A person walking on an escalator -/
def time_to_cover_escalator (escalator_speed person_speed : ℝ) (escalator_length : ℝ) : ℝ :=
  escalator_length / (escalator_speed + person_speed)

theorem person_covers_escalator_in_10_seconds :
  time_to_cover_escalator 15 3 180 = 10 :=
by
  sorry

end person_covers_escalator_in_10_seconds_l117_117193


namespace opposite_of_neg_2023_l117_117016

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117016


namespace unique_quadratic_solution_l117_117227

noncomputable def validates_properties (f : ℝ → ℝ) : Prop :=
(∀ x : ℝ, f x ≥ 0) ∧ 
(∀ a b c d : ℝ, ab + bc + cd = 0 → f (a - b) + f (c - d) = f a + f (b + c) + f d)

theorem unique_quadratic_solution : 
  ∀ f : ℝ → ℝ, validates_properties f → ∃ c : ℝ, c ≥ 0 ∧ ∀ x : ℝ, f x = c * x^2 := 
by sorry

end unique_quadratic_solution_l117_117227


namespace min_sum_of_factors_l117_117824

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1800) : 
  a + b + c = 64 :=
sorry

end min_sum_of_factors_l117_117824


namespace beth_candies_l117_117194

noncomputable def num_candies := 10
noncomputable def min_anne := 3
noncomputable def min_beth := 2
noncomputable def min_chris := 2
noncomputable def max_chris := 3

theorem beth_candies :
  ∀ A B C : ℕ, 
  A + B + C = num_candies →
  A ≥ min_anne →
  B ≥ min_beth →
  min_chris ≤ C ∧ C ≤ max_chris →
  B ∈ {2, 3, 4, 5} :=
by
  intros A B C H_sum H_anne H_beth H_chris
  sorry

end beth_candies_l117_117194


namespace ratio_differences_l117_117675

variable (a b : ℝ)
def f1 (x : ℝ) := x^2 - 2*x + a
def f2 (x : ℝ) := x^2 + b*x - 2
def f3 (x : ℝ) := 4*x^2 + (b-6)*x + 3*a - 2
def f4 (x : ℝ) := 4*x^2 + (3*b-2)*x - 6 + a

noncomputable def A := Real.sqrt (4 - 4 * a)
noncomputable def B := Real.sqrt (b^2 + 8)
noncomputable def C := (1/4) * Real.sqrt ((b-6)^2 - 48*a + 32)
noncomputable def D := (1/4) * Real.sqrt ((3*b-2)^2 - 16*a + 96)

theorem ratio_differences (h : |C| ≠ |D|) : (A^2 - B^2) / (C^2 - D^2) = 2 :=
by {
  sorry
}

end ratio_differences_l117_117675


namespace trail_length_33_l117_117473

-- Definitions based on the conditions
variables (v t L : ℝ)

-- Condition 1: Friend P's rate is 20% faster than Friend Q's.
def p_speed := 1.20 * v

-- Condition 2: Friend P will have walked 18 km when they pass each other.
def p_distance := 18

-- Proof problem: Prove that the total length of the trail is 33 km.
theorem trail_length_33 (h1 : p_distance = p_speed * t)
                       (h2 : L - p_distance = v * t) :
  L = 33 :=
by
  sorry

end trail_length_33_l117_117473


namespace tetrahedron_edges_midpoint_distances_sum_l117_117414

theorem tetrahedron_edges_midpoint_distances_sum (a b c d e f m1 m2 m3 m4 m5 m6 : ℝ) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 4 * (m1^2 + m2^2 + m3^2 + m4^2 + m5^2 + m6^2) :=
sorry

end tetrahedron_edges_midpoint_distances_sum_l117_117414


namespace part1_part2_part3_l117_117659

noncomputable def e : ℝ := 2.718281828459045

def f (x : ℝ) := (x + 1) * Real.log (x + 1)
def g (x k : ℝ) := k * x * Real.exp x
def g' (x k : ℝ) := k * (x + 1) * Real.exp x
def h (x a : ℝ) := (x + 1) * Real.log (x + 1) - a * x
def h' (x a : ℝ) := 1 - a + Real.log (x + 1)

theorem part1 : ∃ k, g' 0 k = 1 := sorry

theorem part2 {k : ℝ} (hk : k = 1) :
  ∀ x > 0, f x < g x k := sorry

theorem part3 :
  (∀ x ≥ 0, f x ≥ (λ x, a) x) ↔ a ≤ 1 := sorry

end part1_part2_part3_l117_117659


namespace line_perpendicular_to_plane_l117_117411

open Set

variable {α : Type*} [EuclideanSpace α]

-- Definitions to use for the conditions
def perpendicular (l1 l2 : Line α) : Prop := ∀ p : Point α, p ∈ l1 ∧ p ∈ l2 → RightAngle l1 l2

def in_plane (l : Line α) (pl : Plane α) : Prop := ∀ p : Point α, p ∈ l → p ∈ pl

def intersects (l1 l2 : Line α) : Prop := ∃ p : Point α, p ∈ l1 ∧ p ∈ l2

-- Theorem statement
theorem line_perpendicular_to_plane
  (a b h : Line α) (α : Plane α)
  (H_intersects : intersects a b)
  (H_in_plane_a : in_plane a α)
  (H_in_plane_b : in_plane b α)
  (H_perpendicular_a : perpendicular h a)
  (H_perpendicular_b : perpendicular h b)
  : ∀ c : Line α, in_plane c α → perpendicular h c :=
sorry

end line_perpendicular_to_plane_l117_117411


namespace opposite_of_neg_2023_l117_117035

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117035


namespace opposite_of_neg_2023_l117_117028

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117028


namespace necessary_but_not_sufficient_condition_l117_117695

variable (α : ℝ)

def in_first_quadrant (α : ℝ) : Prop := 
  0 < α ∧ α < π / 2 ∨ 2 * π < α ∧ α < 5 * π / 2

def is_acute_angle (α : ℝ) : Prop := 
  0 < α ∧ α < π / 2

theorem necessary_but_not_sufficient_condition :
  (in_first_quadrant α → is_acute_angle α) ∧
  ¬ (is_acute_angle α → in_first_quadrant α) := 
  sorry

end necessary_but_not_sufficient_condition_l117_117695


namespace unique_circle_diameter_equilateral_triangle_l117_117740

theorem unique_circle_diameter_equilateral_triangle
  (T : Triangle) (h_equilateral : Equilateral T) :
  ∃! C : Circle, ∃ A B : T.vertex, A ≠ B ∧ C.diameter = Segment A B :=
sorry

end unique_circle_diameter_equilateral_triangle_l117_117740


namespace median_number_of_pets_is_three_l117_117706

noncomputable def list_of_pets : list ℕ := 
  (list.repeat 2 5) ++ (list.repeat 3 6) ++ (list.repeat 4 1) ++
  (list.repeat 5 4) ++ (list.repeat 6 3)

def median (l : list ℕ) : ℕ :=
l.nth_le (l.length / 2) (by {simp, exact nat.div_lt_self (nat.zero_lt_succ _) (by decide)})

theorem median_number_of_pets_is_three :
  median list_of_pets = 3 :=
sorry

end median_number_of_pets_is_three_l117_117706


namespace parabola_vertex_coordinates_l117_117806

theorem parabola_vertex_coordinates :
  (∃ x : ℝ, (λ x, x^2 - 2) = (0, -2)) :=
sorry

end parabola_vertex_coordinates_l117_117806


namespace countPathsBMC1_correct_l117_117864

def countPathsBMC1 (grid : List (List Char)) (start : (Nat, Nat)) : Nat :=
  sorry

theorem countPathsBMC1_correct
  (grid : List (List Char))
  (start : (Nat, Nat))
  (h_adjacent_up : ∀ (x y : Nat), grid.get! x |>.get! y = 'B' → grid.get! (x - 1) |>.get! y = 'M')
  (h_adjacent_down : ∀ (x y : Nat), grid.get! x |>.get! y = 'B' → grid.get! (x + 1) |>.get! y = 'M')
  (h_adjacent_left : ∀ (x y : Nat), grid.get! x |>.get! y = 'B' → grid.get! x |>.get! (y - 1) = 'M')
  (h_adjacent_right : ∀ (x y : Nat), grid.get! x |>.get! y = 'B' → grid.get! x |>.get! (y + 1) = 'M')
  (h_M_to_C : ∀ (x y : Nat), grid.get! x |>.get! y = 'M' → (grid.get! (x + 1) |>.get! y = 'C' ∨ grid.get! (x - 1) |>.get! y = 'C' ∨ grid.get! x |>.get! (y + 1) = 'C' ∨ grid.get! x |>.get! (y - 1) = 'C'))
  (h_C_to_1 : ∀ (x y : Nat), grid.get! x |>.get! y = 'C' → (grid.get! (x + 1) |>.get! y = '1' ∨ grid.get! (x - 1) |>.get! y = '1' ∨ grid.get! x |>.get! (y + 1) = '1' ∨ grid.get! x |>.get! (y - 1) = '1'))
  (h_no_diagonal : ∀ (x y : Nat), ¬(grid.get! x |>.get! y = '1' → (grid.get! (x - 1) |>.get! (y - 1) = 'C'))):
  countPathsBMC1 grid start = 16 := sorry

end countPathsBMC1_correct_l117_117864


namespace find_ratio_product_l117_117716

open Lean

noncomputable theory

variables {A B C A' B' C' O : Type*}

-- Points A', B' and C' are on sides BC, AC, and AB respectively
axiom points_on_sides (h1 : A' ∈ line_segment B C) (h2 : B' ∈ line_segment A C) (h3 : C' ∈ line_segment A B) 

-- AA', BB', and CC' are concurrent at O
axiom concurrency (h1 : concurrent (line_through A A') (line_through B B') (line_through C C')) (O)

-- Given condition
axiom ratio_sum (h1 : ∀ (AO OA' BO OB' CO OC' : ℝ), AO / OA' + BO / OB' + CO / OC' = 92)

-- Proof goal
theorem find_ratio_product (AO OA' BO OB' CO OC' : ℝ) :
  AO / OA' * BO / OB' * CO / OC' = 94 :=
by
  sorry

end find_ratio_product_l117_117716


namespace line_intersects_x_axis_at_6_l117_117517

theorem line_intersects_x_axis_at_6 :
  ∀ (x y : ℝ), (4, -2) ∈ set_of (λ p, p.2 = 1 * (p.1 - 4) - 2) →
               (8, 2) ∈ set_of (λ p, p.2 = 1 * (p.1 - 4) - 2) →
               ∃ (x₀ : ℝ), (0, x₀) ∈ set_of (λ p, p.2 = 1 * (p.1 - 4) - 2) ∧ x₀ = 6 :=
by sorry

end line_intersects_x_axis_at_6_l117_117517


namespace same_terminal_side_l117_117488

open Real

theorem same_terminal_side (k : ℤ) : (∃ k : ℤ, k * 360 - 315 = 9 / 4 * 180) :=
by
  sorry

end same_terminal_side_l117_117488


namespace decrypt_puzzle_l117_117570

-- Definitions based on conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Variables representing respective digits
variables (A E V Z C H K I L : ℕ)

-- Assumptions based on the conditions given
axiom AE_is_even : is_even A ∧ is_even E ∧ is_even V
axiom YHK_mod : (YHK = Y * 100 + H * 10 + K)
axiom Y_is_odd : is_odd Y
axiom H_is_odd : is_odd H
axiom K_is_odd : is_odd K
axiom AEV_rep : AEV = A * 100 + E * 10 + V
axiom correctness : YHK - (AEV * C) = 7557

-- Additional constraints from the problem
axiom A_eq_2 : A = 2
axiom C_eq_3 : C = 3
axiom Y_eq_8 : Y = 8
axiom E_H_K_vals : E = 8 ∧ H = 5 ∧ K = 5
axiom I_eq : I = 2
axiom F_eq : F = 5

-- Final statement to prove the equivalence
theorem decrypt_puzzle : 
  (A = 2) ∧ (C = 3) ∧ (Y = 8) ∧ (E = 8) ∧ (V = 6) ∧ (H = 5) ∧ (K = 5) ∧ (I = 2) ∧ (F = 5) :=
by
  sorry

end decrypt_puzzle_l117_117570


namespace interview_score_l117_117578

theorem interview_score (a b c : ℝ) (pa pb pc : ℝ) :
  a = 92 → b = 87 → c = 94 →
  pa = 0.20 → pb = 0.30 → pc = 0.50 →
  a * pa + b * pb + c * pc = 91.5 :=
by
  intros ha hb hc hpa hpb hpc
  rw [ha, hb, hc, hpa, hpb, hpc]
  norm_num
  sorry

end interview_score_l117_117578


namespace not_increasing_h_l117_117190

open Real

-- Define the functions given in the problem
def f (x : ℝ) := 2^x
def g (x : ℝ) := log (sqrt 2) x
def h (x : ℝ) := 2 / x
def k (x : ℝ) := 2 * x^2 + x + 1

-- Define the theorem to prove that h(x) is not increasing on (0, +∞)
theorem not_increasing_h : ¬(StrictlyIncreasing (λ x : ℝ, h x) (λ x, x > 0)) :=
sorry

end not_increasing_h_l117_117190


namespace num_bricks_required_l117_117159

def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def brick_length_cm : ℝ := 20
def brick_width_cm : ℝ := 10
def cm_to_m (cm : ℝ) : ℝ := cm / 100
def area_of_courtyard : ℝ := courtyard_length * courtyard_width
def area_of_one_brick : ℝ := cm_to_m(brick_length_cm) * cm_to_m(brick_width_cm)
def num_of_bricks (courtyard_area brick_area : ℝ) : ℝ := courtyard_area / brick_area

theorem num_bricks_required :
  num_of_bricks area_of_courtyard area_of_one_brick = 20000 := by
sorry

end num_bricks_required_l117_117159


namespace A_initial_investment_l117_117917

variable (profit_ratio_A : ℚ) (profit_ratio_B : ℚ)
variable (time_A : ℚ) (time_B : ℚ)
variable (B_contribution : ℚ)
variable (A_investment : ℚ)

-- Conditions
def conditions := 
  profit_ratio_A = 2 / 3 ∧
  profit_ratio_B = 3 / 3 ∧
  time_A = 12 ∧
  time_B = 3 ∧
  B_contribution = 21000

-- Theorem to prove
theorem A_initial_investment (h : conditions) : A_investment = 3500 := by
  sorry

end A_initial_investment_l117_117917


namespace number_of_pythagorean_triples_l117_117928

-- Definition for a Pythagorean triple
def is_pythagorean_triple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Given sets to check
def set_1 := (3, 4, 5)
def set_2 := (6/10, 8/10, 1 : ℝ)
def set_3 := (8, 15, 17)
def set_4 := (7, 24, 25)

-- Proving the number of Pythagorean triple sets is equal to 2
theorem number_of_pythagorean_triples : 
  (if is_pythagorean_triple set_1.1 set_1.2 set_1.3 then 1 else 0)
  + (if ∃ (a b c : ℕ), set_2.1 = a ∧ set_2.2 = b ∧ set_2.3 = c ∧ is_pythagorean_triple a b c then 1 else 0)
  + (if is_pythagorean_triple set_3.1 set_3.2 set_3.3 then 1 else 0)
  + (if is_pythagorean_triple set_4.1 set_4.2 set_4.3 then 1 else 0) = 2 := sorry

end number_of_pythagorean_triples_l117_117928


namespace sector_length_OAB_l117_117359

theorem sector_length_OAB
  (O A B X Y : Type) [EuclideanGeometry O A B X Y]
  (OA OB OY : ℝ) (O_angle_AOB : ℝ)
  (OY_perpendicular_AB : ∀ (A B X : Type), is_perpendicular AB OY)
  (OA_unit : OA = 12) (OB_unit : OB = 12) (angle_unit : O_angle_AOB = 120):
  XY = 12 - 6 * real.sqrt 3 := 
by sorry

end sector_length_OAB_l117_117359


namespace total_hamburgers_l117_117943

-- Definitions from the conditions
def total_spent : ℝ := 68.50
def cost_single_burger : ℝ := 1.00
def cost_double_burger : ℝ := 1.50
def num_double_burgers : ℕ := 37

-- Define the proof statement
theorem total_hamburgers (total_spent cost_single_burger cost_double_burger : ℝ) (num_double_burgers : ℕ) :

-- Calculate cost of double burgers
let cost_of_double_burgers := cost_double_burger * num_double_burgers

-- Calculate cost of single burgers
let cost_of_single_burgers := total_spent - cost_of_double_burgers

-- Calculate number of single burgers
let num_single_burgers := cost_of_single_burgers / cost_single_burger

-- Calculate total number of burgers
let total_burgers := num_single_burgers + num_double_burgers in

total_burgers = 50 := by {
    sorry
}

end total_hamburgers_l117_117943


namespace range_of_angle_C_in_triangle_l117_117344

theorem range_of_angle_C_in_triangle (BC AB C : ℝ) (h₁ : BC = 2) (h₂ : AB = sqrt 3)
  (h₃ : 0 < C) (h₄ : C ≤ π / 3) :
  (∃ b : ℝ, 3 = b^2 + 4 - 4 * b * cos C) :=
by
  sorry

end range_of_angle_C_in_triangle_l117_117344


namespace total_apples_l117_117409

def pinky_apples : ℕ := 36
def danny_apples : ℕ := 73

theorem total_apples :
  pinky_apples + danny_apples = 109 :=
by
  sorry

end total_apples_l117_117409


namespace sum_of_decimals_as_fraction_l117_117591

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 :=
by
  sorry

end sum_of_decimals_as_fraction_l117_117591


namespace find_red_cards_l117_117903

-- We use noncomputable here as we are dealing with real numbers in a theoretical proof context.
noncomputable def red_cards (r b : ℕ) (_initial_prob : r / (r + b) = 1 / 5) 
                            (_added_prob : r / (r + b + 6) = 1 / 7) : ℕ := 
r

theorem find_red_cards 
  {r b : ℕ}
  (h1 : r / (r + b) = 1 / 5)
  (h2 : r / (r + b + 6) = 1 / 7) : 
  red_cards r b h1 h2 = 3 :=
sorry  -- Proof not required

end find_red_cards_l117_117903


namespace gideon_age_l117_117459

noncomputable def gideon_current_age : ℕ :=
  let total_marbles := 100
  let remaining_marbles := total_marbles * (1/4 : ℚ)
  let age_in_five_years := remaining_marbles * 2
  let curr_age := age_in_five_years - 5
  curr_age

theorem gideon_age : gideon_current_age = 45 :=
by
  let total_marbles := 100
  let remaining_marbles := total_marbles * (1/4 : ℚ)
  let age_in_five_years := remaining_marbles * 2
  let curr_age := age_in_five_years - 5
  have h1 : total_marbles = 100 := rfl
  have h2 : remaining_marbles = 25 := by norm_num
  have h3 : age_in_five_years = 50 := by norm_num
  have h4 : curr_age = 45 := by norm_num
  exact h4

end gideon_age_l117_117459


namespace intersection_A_B_l117_117676

def setA : Set ℝ := { x | x^2 - 2*x < 3 }
def setB : Set ℝ := { x | x ≤ 2 }
def setC : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_A_B :
  (setA ∩ setB) = setC :=
by
  sorry

end intersection_A_B_l117_117676


namespace white_gumballs_after_replacement_l117_117516

variables (gumballs : ℕ)
variables (orange_gumballs purple_gumballs green_gumballs white_gumballs black_gumballs pink_gumballs : ℕ)

-- Conditions
axiom gumballs_sum : orange_gumballs + purple_gumballs + green_gumballs + white_gumballs + black_gumballs = 0.8 * gumballs
axiom pink_gumballs_45 : pink_gumballs = 45
axiom orange_gumballs_percentage : orange_gumballs = 0.25 * gumballs
axiom purple_gumballs_percentage : purple_gumballs = 0.15 * gumballs
axiom green_gumballs_percentage : green_gumballs = 0.20 * gumballs
axiom white_gumballs_percentage : white_gumballs = 0.10 * gumballs
axiom black_gumballs_percentage : black_gumballs = 0.10 * gumballs
axiom total_gumballs : gumballs = pink_gumballs / 0.20

-- Question to prove
theorem white_gumballs_after_replacement:
  let initial_white_gumballs := 0.10 * gumballs in
  let initial_green_gumballs := 0.20 * gumballs in
  initial_white_gumballs + (initial_green_gumballs / 3) = 37.5 :=
by
  sorry

end white_gumballs_after_replacement_l117_117516


namespace find_cos_angle_l117_117319

namespace VectorProof

variables {V : Type*} [inner_product_space ℝ V]
variables (c d : V)

def cos_angle (c d : V) : ℝ := real_inner c d / (∥c∥ * ∥d∥)

theorem find_cos_angle
  (hc : ∥c∥ = 5)
  (hd : ∥d∥ = 7)
  (hcd : ∥c + d∥ = 10) :
  cos_angle c d = 13 / 35 :=
by
  sorry

end VectorProof

end find_cos_angle_l117_117319


namespace total_students_correct_l117_117469

def total_students (M : ℕ) (E : ℕ) : ℕ := E + M

theorem total_students_correct :
  ∀ (M E : ℕ), (E = 4 * M - 3) → (M = 50) → total_students M E = 247 :=
by {
  intros M E hE hM,
  rw hM at hE,
  rw hM,
  have hE' : E = 4 * 50 - 3 := by { rw hE },
  rw hE',
  norm_num,
  reflexivity,
  sorry,
}

end total_students_correct_l117_117469


namespace number_of_partitions_l117_117338

theorem number_of_partitions (n : ℕ) (A : Finset (Fin (n+1))) (A_parts : Finset (Fin n).Powerset) :
  (∀ (i : Fin n), ∃ (P : Finset (Fin (n+1))), P ∈ A_parts ∧ ((A_parts.image (λ P : Finset (Fin (n+1)), P)).Union = A)) →
  A_parts.card = (2^n - 1) * n + 1 := 
sorry

end number_of_partitions_l117_117338


namespace james_beats_per_week_l117_117369

def beats_per_minute := 200
def hours_per_day := 2
def days_per_week := 7

def beats_per_week (beats_per_minute: ℕ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℕ :=
  (beats_per_minute * hours_per_day * 60) * days_per_week

theorem james_beats_per_week : beats_per_week beats_per_minute hours_per_day days_per_week = 168000 := by
  sorry

end james_beats_per_week_l117_117369


namespace relatively_prime_nonconsecutive_threes_l117_117531

def b : ℕ → ℕ
| 0       := 1
| 1       := 2
| 2       := 4
| (n + 3) := b n + b (n + 1) + b (n + 2)

noncomputable def probability (k : ℕ) : ℚ :=
(b k : ℚ) / 2^k

theorem relatively_prime_nonconsecutive_threes :
  let p := 927
  let q := 4096 in
  Nat.gcd p q = 1 ∧ p + q = 5023 :=
by
  sorry

end relatively_prime_nonconsecutive_threes_l117_117531


namespace subtraction_calculation_l117_117108

theorem subtraction_calculation (a b : ℤ) (h : b = 40) (h1 : a = b - 1) : (a * a) = (b * b) - 79 := 
by
  -- Using the given condition
  have h2 : a * a = (b - 1) * (b - 1),
  from by rw [h1],
  -- Expanding using binomial theorem
  rw [mul_sub, sub_mul, mul_one, ← square_eq, sub_sub, one_mul, one_mul] at h2,
  -- Proving the theorem
  rw [sub_add] at h2,
  exact h2,
  sorry

end subtraction_calculation_l117_117108


namespace percentage_of_women_not_speaking_french_is_65_l117_117144

-- Definitions based on conditions
def men_percentage : ℝ := 0.60
def men_speak_french_percentage : ℝ := 0.60
def employees_speak_french_percentage : ℝ := 0.50

theorem percentage_of_women_not_speaking_french_is_65 :
  let total_employees := 100 in
  let total_men := men_percentage * total_employees in
  let total_women := total_employees - total_men in
  let men_speaking_french := men_speak_french_percentage * total_men in
  let total_speaking_french := employees_speak_french_percentage * total_employees in
  let women_speaking_french := total_speaking_french - men_speaking_french in
  let women_not_speaking_french := total_women - women_speaking_french in
  (women_not_speaking_french / total_women) * 100 = 65 :=
sorry

end percentage_of_women_not_speaking_french_is_65_l117_117144


namespace isosceles_right_triangle_area_l117_117848

noncomputable def triangle_area (a b c : ℝ) (angle : ℝ) : ℝ :=
  if angle = 90 then 0.5 * a * b
  else sorry -- General case for arbitrary angle, to be filled in as needed.

theorem isosceles_right_triangle_area :
  ∀ (PQ PR : ℝ), (PQ = 8) → (PR = 8) → (triangle_area PQ PR 0 90) = 32 :=
by
  intros PQ PR hPQ hPR
  simp [triangle_area]
  rw [hPQ, hPR]
  norm_num
  sorry

end isosceles_right_triangle_area_l117_117848


namespace smallest_value_of_a2019_l117_117503

theorem smallest_value_of_a2019 :
  ∃ (a : ℕ → ℕ),
  a 0 = 1 ∧
  a 1 = 1 ∧
  (∀ i ≥ 2, a i % a (i - 1) = 0 ∧ a i % (a (i - 1) + a (i - 2)) = 0) ∧
  a 2019 = (list.range 2 2020).prod :=
sorry

end smallest_value_of_a2019_l117_117503


namespace choose_five_from_fifteen_l117_117707

theorem choose_five_from_fifteen : nat.choose 15 5 = 3003 := by
  sorry

end choose_five_from_fifteen_l117_117707


namespace sum_minimum_nine_l117_117644

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
a + n * d

theorem sum_minimum_nine (a_1 a_8 a_13 S_n : ℤ) (d : ℤ) :
  a_1 = -26 ∧ a_8 + a_13 = 5 → 
  (∀ n : ℤ, S_n = (3 / 2) * n^2 - (55 / 2) * n) → (∃ n : ℕ, n = 9 ∧ ∀ m : ℕ, S_n n ≤ S_n m) :=
begin
  sorry
end

end sum_minimum_nine_l117_117644


namespace proofFirstProblem_proofSecondProblem_l117_117561

-- Define the first problem
def firstProblem : Prop :=
  sqrt 27 / sqrt 3 + sqrt 12 * sqrt (1 / 3) - sqrt 5 = 5 - sqrt 5

-- Define the second problem
def secondProblem : Prop :=
  (sqrt 5 + 2) * (sqrt 5 - 2) + (2 * sqrt 3 + 1)^2 = 14 + 4 * sqrt 3

-- Proof (omitted)
theorem proofFirstProblem : firstProblem := by
  sorry

theorem proofSecondProblem : secondProblem := by
  sorry

end proofFirstProblem_proofSecondProblem_l117_117561


namespace num_tangent_lines_with_equal_intercepts_l117_117250

theorem num_tangent_lines_with_equal_intercepts :
  let circle_eq := ∀ x y, x^2 + (y-2)^2 = 1,
      intercept_condition := ∀ m c, m * c = c ∨ m = 1 ∨ c = 0 in
  ∃ lines : ℕ, lines = 4 :=
by
  sorry

end num_tangent_lines_with_equal_intercepts_l117_117250


namespace largest_number_of_elements_in_S_l117_117532

-- Define a structure for a triangle with integer sides
structure Triangle :=
(a b c : ℕ)
(h1 : a ≥ b)
(h2 : b ≥ c)
(h3 : a < 7)
(h4 : b < 7)
(h5 : c < 7)
(h_triangle_ineq : b + c > a)

-- Define congruence and similarity
def congruent (T1 T2 : Triangle) : Prop :=
(T1.a = T2.a ∧ T1.b = T2.b ∧ T1.c = T2.c) ∨
(T1.a = T2.a ∧ T1.b = T2.c ∧ T1.c = T2.b) ∨
(T1.a = T2.b ∧ T1.b = T2.a ∧ T1.c = T2.c) ∨
(T1.a = T2.b ∧ T1.b = T2.c ∧ T1.c = T2.a) ∨
(T1.a = T2.c ∧ T1.b = T2.a ∧ T1.c = T2.b) ∨
(T1.a = T2.c ∧ T1.b = T2.b ∧ T1.c = T2.a)

def similar (T1 T2 : Triangle) : Prop :=
let ratio := λ x y, (x : ℚ) / (y : ℚ) in
(ratio T1.a T2.a = ratio T1.b T2.b ∧ ratio T1.a T2.a = ratio T1.c T2.c) ∨
(ratio T1.a T2.b = ratio T1.b T2.a ∧ ratio T1.a T2.b = ratio T1.c T2.c) ∨
(ratio T1.a T2.c = ratio T1.b T2.a ∧ ratio T1.a T2.c = ratio T1.c T2.b) ∨
(ratio T1.b T2.a = ratio T1.a T2.b ∧ ratio T1.b T2.a = ratio T1.c T2.c) ∨
(ratio T1.b T2.b = ratio T1.a T2.a ∧ ratio T1.b T2.b = ratio T1.c T2.c) ∨
(ratio T1.b T2.c = ratio T1.a T2.a ∧ ratio T1.b T2.c = ratio T1.c T2.b) ∨
(ratio T1.c T2.a = ratio T1.a T2.b ∧ ratio T1.c T2.a = ratio T1.b T2.c) ∨
(ratio T1.c T2.b = ratio T1.a T2.a ∧ ratio T1.c T2.b = ratio T1.b T2.c) ∨
(ratio T1.c T2.c = ratio T1.a T2.a ∧ ratio T1.c T2.c = ratio T1.b T2.b)

-- Define a set S of triangles
def S : set Triangle := { T | T.a < 7 ∧ T.b < 7 ∧ T.c < 7 ∧ ∀ T' ∈ S, ¬ congruent T T' ∧ ¬ similar T T' }

-- Formalize the problem statement
theorem largest_number_of_elements_in_S : ∃ n, n ≤ 20 ∧ ∀ S, (∀ T1 T2 ∈ S, ¬ congruent T1 T2 ∧ ¬ similar T1 T2) → finset.card S ≤ n :=
sorry

end largest_number_of_elements_in_S_l117_117532


namespace polynomial_bivariate_l117_117572

-- Define the sets X and Y
variable (X Y : Set ℝ)

-- Define the function f and its properties
variable (f : ℝ × ℝ → ℝ)

-- State the problem in Lean 4
theorem polynomial_bivariate {X Y : Set ℝ} (f : ℝ × ℝ → ℝ)
  (h₁ : ∀ (x : ℝ), x ∈ X → ∃ p : ℝ → ℝ, Polynomial p ∧ ∀ (y : ℝ), y ∈ Y → f (x, y) = p y)
  (h₂ : ∀ (y : ℝ), y ∈ Y → ∃ q : ℝ → ℝ, Polynomial q ∧ ∀ (x : ℝ), x ∈ X → f (x, y) = q x) :
  ∃ p : ℝ × ℝ → ℝ, is_polynomial p ∧ ∀ (x : ℝ) (y : ℝ), x ∈ X → y ∈ Y → f (x, y) = p (x, y) :=
sorry

end polynomial_bivariate_l117_117572


namespace area_triangle_PCF_l117_117274

variables {ℝ : Type*} [LinearOrderedField ℝ]

/-- Given the areas of the triangles formed by dropping perpendiculars from an internal point
    in an equilateral triangle, prove the area of the third triangle. -/
theorem area_triangle_PCF
  (S_ABC : ℝ)
  (S_PAD : ℝ)
  (S_PBE : ℝ) :
  (S_ABC = 2028) →
  (S_PAD = 192) →
  (S_PBE = 192) →
  let S_PCF := S_ABC / 2 - S_PAD - S_PBE in
  S_PCF = 630 := 
by
  intros h1 h2 h3
  dsimp [S_PCF]
  rw [h1, h2, h3]
  norm_num
  sorry

end area_triangle_PCF_l117_117274


namespace four_x_squared_minus_three_is_quadratic_binomial_l117_117126

def is_binomial (f : ℕ → ℝ) : Prop :=
  ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c ∧ a ≠ 0 ∧ c ≠ 0

def quadratic_degree (f : ℕ → ℝ) (d : ℕ) : Prop :=
  ∃ a b : ℝ, f = λ x, a * x^2 + b * x ∧ d = 2

theorem four_x_squared_minus_three_is_quadratic_binomial :
  is_binomial (λ x, 4 * x^2 - 3) ∧ quadratic_degree (λ x, 4 * x^2 - 3) 2 :=
by
  sorry

end four_x_squared_minus_three_is_quadratic_binomial_l117_117126


namespace kombucha_fill_half_time_l117_117431

/--
The area of the kombucha in the jar doubles every day.
It takes 17 days to fill the entire jar.
Prove that it takes 16 days to fill half of the jar.
-/
theorem kombucha_fill_half_time : (∀ n, area n = 2^n) ∧ (area 17 = 2^17) → (∀ half_area, half_area = (2^17) / 2) → ∃ d, d = 16 := sorry

end kombucha_fill_half_time_l117_117431


namespace problem_statement_l117_117275

noncomputable def f : ℝ → ℝ := sorry

variables {x : ℝ}

-- Conditions
def even_function (f : ℝ → ℝ) := ∀ x, f (x) = f (-x)
def periodic_function (f : ℝ → ℝ) := ∀ x, f (x + 2) = f (x)
def increasing_function (f : ℝ → ℝ) := ∀ a b, -1 ≤ a ∧ a < b ∧ b ≤ 0 → f a < f b

-- Proof Goal
theorem problem_statement
  (h_even : even_function f)
  (h_periodic : periodic_function f)
  (h_increasing : increasing_function f) :
  f 3 < f (real.sqrt 2) ∧ f (real.sqrt 2) < f 2 := 
sorry

end problem_statement_l117_117275


namespace boys_camp_total_l117_117703

theorem boys_camp_total {T : ℕ} 
  (h1 : 0.20 * T = (28 / 0.70)) 
  (h2 : 0.30 * (0.20 * T) + 0.70 * (0.20 * T) = 0.20 * T) 
  (h3 : 28 = 0.70 * (0.20 * T)) :
  T = 200 := 
by 
  sorry

end boys_camp_total_l117_117703


namespace find_BP_l117_117889

theorem find_BP
    (A B C D P : Type) [Point A] [Point B] [Point C] [Point D] [Point P]
    (h_circle : Circle A B C D) 
    (h_intersect : Intersect AC BD P)
    (h_AP : AP = 10) 
    (h_PC : PC = 2) 
    (h_BD : BD = 9) 
    (h_BP_DP : BP < DP) : 
    BP = 4 := 
sorry

end find_BP_l117_117889


namespace find_m_of_parallel_vectors_l117_117318

theorem find_m_of_parallel_vectors (m : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (m, m + 1))
  (parallel : a.1 * b.2 = a.2 * b.1) :
  m = 1 :=
by
  -- We assume a parallel condition and need to prove m = 1
  sorry

end find_m_of_parallel_vectors_l117_117318


namespace blue_notebook_cost_l117_117404

theorem blue_notebook_cost
    (total_spent : ℕ)
    (total_notebooks : ℕ)
    (red_notebooks : ℕ) (red_cost : ℕ)
    (green_notebooks : ℕ) (green_cost : ℕ)
    (blue_notebooks : ℕ) (blue_total_cost : ℕ) 
    (blue_cost : ℕ) :
    total_spent = 37 →
    total_notebooks = 12 →
    red_notebooks = 3 →
    red_cost = 4 →
    green_notebooks = 2 →
    green_cost = 2 →
    blue_notebooks = total_notebooks - red_notebooks - green_notebooks →
    blue_total_cost = total_spent - red_notebooks * red_cost - green_notebooks * green_cost →
    blue_cost = blue_total_cost / blue_notebooks →
    blue_cost = 3 := 
    by sorry

end blue_notebook_cost_l117_117404


namespace opposite_of_neg_2023_l117_117024

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117024


namespace paint_canvas_cost_ratio_l117_117930

theorem paint_canvas_cost_ratio (C P : ℝ) (hc : 0.6 * C = C - 0.4 * C) (hp : 0.4 * P = P - 0.6 * P)
 (total_cost_reduction : 0.4 * P + 0.6 * C = 0.44 * (P + C)) :
  P / C = 4 :=
by
  sorry

end paint_canvas_cost_ratio_l117_117930


namespace quotient_of_division_l117_117407

theorem quotient_of_division (dividend divisor remainder quotient : ℕ) 
  (h_dividend : dividend = 271) (h_divisor : divisor = 30) 
  (h_remainder : remainder = 1) (h_division : dividend = divisor * quotient + remainder) : 
  quotient = 9 := 
by 
  sorry

end quotient_of_division_l117_117407


namespace inequality_holds_l117_117615

theorem inequality_holds (x : ℝ) (m : ℝ) :
  (∀ x : ℝ, (x^2 - m * x - 2) / (x^2 - 3 * x + 4) > -1) ↔ (-7 < m ∧ m < 1) :=
by
  sorry

end inequality_holds_l117_117615


namespace find_n_l117_117471

-- Declaring the necessary context and parameters.
variable (n : ℕ)

-- Defining the condition described in the problem.
def reposting_equation (n : ℕ) : Prop := 1 + n + n^2 = 111

-- Stating the theorem to prove that for n = 10, the reposting equation holds.
theorem find_n : ∃ (n : ℕ), reposting_equation n ∧ n = 10 :=
by
  use 10
  unfold reposting_equation
  sorry

end find_n_l117_117471


namespace rationalize_sqrt_35_l117_117415

theorem rationalize_sqrt_35 : (35 / Real.sqrt 35) = Real.sqrt 35 :=
  sorry

end rationalize_sqrt_35_l117_117415


namespace opposite_of_neg_2023_l117_117066

theorem opposite_of_neg_2023 : ∃ y : ℤ, -2023 + y = 0 ∧ y = 2023 :=
by
  use 2023
  split
  · ring
  · rfl

end opposite_of_neg_2023_l117_117066


namespace maximum_value_x_squared_plus_2y_l117_117267

theorem maximum_value_x_squared_plus_2y (x y b : ℝ) (h_curve : x^2 / 4 + y^2 / b^2 = 1) (h_b_positive : b > 0) : 
  x^2 + 2 * y ≤ max (b^2 / 4 + 4) (2 * b) :=
sorry

end maximum_value_x_squared_plus_2y_l117_117267


namespace trapezoid_ACED_l117_117479

-- Definitions from the conditions
structure Rectangle (A B C D : Type u) :=
(is_rectangle : ∀ (a b c d : Type u), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ c ≠ d →
              ∃ (R : set (a × a)), ∀ (p q : a), R p q ↔ q = p + c / 2 → p ∈ C)
              
def circumcircle (A B C D : Type u) : Type u := sorry  -- Definition of circumcircle can be complex

def circle (A B : Type u) : Type u := sorry  -- Definition of circle with center and radius

variable {Point : Type u}

-- Lean 4 statement to prove the condition
theorem trapezoid_ACED {A B C D : Point}
  (h_rect : Rectangle A B C D)
  (h_circleA : circle A B) 
  (h_circumcircle : circumcircle A B C D)
  (E : Point)
  (h_intersect : E ≠ B ∧ E ∈ (h_circumcircle) ∧ E ∈ (h_circleA))
  : ∃ P Q R S : Point, (P = A ∧ Q = C ∧ R = E ∧ S = D) ∧ Trapezoid P Q R S := 
by
  sorry

end trapezoid_ACED_l117_117479


namespace trains_meet_in_9_77_seconds_l117_117498

-- Defining the conditions
def length_train1 : ℕ := 120  -- Length of train 1 in meters
def length_train2 : ℕ := 210  -- Length of train 2 in meters
def distance_apart : ℕ := 80  -- Distance between the two trains in meters
def speed_train1_kmph : ℕ := 69  -- Speed of train 1 in km/h
def speed_train2_kmph : ℕ := 82  -- Speed of train 2 in km/h

-- Comvert speeds from km/h to m/s
def speed_train1_mps : ℝ := (speed_train1_kmph * 1000) / 3600 
def speed_train2_mps : ℝ := (speed_train2_kmph * 1000) / 3600

-- Calculate relative speed in m/s
def relative_speed : ℝ := speed_train1_mps + speed_train2_mps

-- Total distance to be covered in meters
def total_distance : ℕ := length_train1 + length_train2 + distance_apart

-- Calculate the time taken in seconds 
def time_to_meet : ℝ := total_distance / relative_speed

-- The proof statement
theorem trains_meet_in_9_77_seconds : time_to_meet ≈ 9.77 := 
by 
  sorry

end trains_meet_in_9_77_seconds_l117_117498


namespace opposite_of_neg2023_l117_117001

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end opposite_of_neg2023_l117_117001


namespace proof1_proof2_proof3_l117_117556

noncomputable def problem1 : ℝ :=
  ((9/16) ^ (1/2)) + 31000 - ((64/27) ^ (-1/3)) + 3 * (Real.exp 0)

theorem proof1 : problem1 = 13 := by
  sorry

noncomputable def problem2 : ℝ :=
  (Real.logb 10 (sqrt 27) + Real.logb 10 8 - Real.logb 4 8) / ((1/2) * Real.logb 10 0.3 + Real.logb 10 2)

theorem proof2 : problem2 = 3 := by
  sorry

noncomputable def problem3 : ℝ :=
  (Real.logb 10 5) ^ 2 + Real.logb 10 2 * Real.logb 10 50

theorem proof3 : problem3 = 1 := by
  sorry

end proof1_proof2_proof3_l117_117556


namespace bus_stop_time_l117_117873

/-- 
  We are given:
  speed_ns: speed of bus without stoppages (32 km/hr)
  speed_ws: speed of bus including stoppages (16 km/hr)
  
  We need to prove the bus stops for t = 30 minutes each hour.
-/
theorem bus_stop_time
  (speed_ns speed_ws: ℕ)
  (h_ns: speed_ns = 32)
  (h_ws: speed_ws = 16):
  ∃ t: ℕ, t = 30 := 
sorry

end bus_stop_time_l117_117873


namespace words_of_length_10_l117_117978

noncomputable def a : ℕ → ℕ
| 0     := 1  -- Trivial base case, no word
| 1     := 2  -- "a" and "b"
| 2     := 3  -- "aa", "ab", "ba"
| (n+1) := a n + a (n - 1)

theorem words_of_length_10 :
  a 10 = 144 :=
by
  sorry

end words_of_length_10_l117_117978


namespace disjoint_subsets_X_l117_117569

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def prod_set (S : set ℕ) : ℕ :=
  if S = ∅ then 1 else S.to_finset.prod id

theorem disjoint_subsets_X :
  ∃ (A B : set ℕ), 
    A ∪ B = {x | 1 ≤ x ∧ x ≤ 10} ∧ 
    A ∩ B = ∅ ∧ 
    A ≠ ∅ ∧ 
    B ≠ ∅ ∧ 
    ∃ (k : ℕ), prod_set A = k * prod_set B ∧ k = 1 :=
by sorry

end disjoint_subsets_X_l117_117569


namespace paper_pieces_l117_117926

theorem paper_pieces (n : ℕ) (h1 : 20 = 2 * n - 8) : n^2 + 20 = 216 := 
by
  sorry

end paper_pieces_l117_117926


namespace solve_equation_l117_117785

theorem solve_equation (x : ℚ) :
  (x^2 + 3 * x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 :=
by
  sorry

end solve_equation_l117_117785


namespace correct_operations_l117_117124

theorem correct_operations :
  (∀ {a b : ℝ}, -(-a + b) = a + b → False) ∧
  (∀ {a : ℝ}, 3 * a^3 - 3 * a^2 = a → False) ∧
  (∀ {x : ℝ}, (x^6)^2 = x^8 → False) ∧
  (∀ {z : ℝ}, 1 / (2 / 3 : ℝ)⁻¹ = 2 / 3) :=
by
  sorry

end correct_operations_l117_117124


namespace hyperbola_standard_equation_l117_117670

noncomputable def c (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem hyperbola_standard_equation
  (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
  (focus_distance_condition : ∃ (F1 F2 : ℝ), |F1 - F2| = 2 * (c a b))
  (circle_intersects_asymptote : ∃ (x y : ℝ), (x, y) = (1, 2) ∧ y = (b/a) * x + 2): 
  (a = 1) ∧ (b = 2) → (x^2 - (y^2 / 4) = 1) := 
sorry

end hyperbola_standard_equation_l117_117670


namespace common_root_exists_l117_117965

theorem common_root_exists :
  ∃ x, (3 * x^4 + 13 * x^3 + 20 * x^2 + 17 * x + 7 = 0) ∧ (3 * x^4 + x^3 - 8 * x^2 + 11 * x - 7 = 0) → x = -7 / 3 := 
by
  sorry

end common_root_exists_l117_117965


namespace minor_axis_length_max_AF2_BF2_equal_min_distance_AB_l117_117270

noncomputable def ellipse : Type := sorry

structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < 2
  eq : ∀ x y : ℝ, x^2 / 4 + y^2 / (b^2) = 1

structure Points where
  A : ellipse → ℝ × ℝ
  B : ellipse → ℝ × ℝ
  F1 : ellipse → ℝ × ℝ
  F2 : ellipse → ℝ × ℝ

def max_AF2_BF2 (e : ellipse) (p : Points) : ℝ := 5

theorem minor_axis_length (e : ellipse) (p : Points) : 2 * e.b = 2 * Real.sqrt 3 :=
by sorry

theorem max_AF2_BF2_equal (e : ellipse) (p : Points) : max_AF2_BF2 e p = 5 → |p.A.2 - p.B.2| = 0 :=
by sorry

theorem min_distance_AB (e : ellipse) (p : Points) : |p.A.1 - p.B.1| = 3 :=
by sorry

end minor_axis_length_max_AF2_BF2_equal_min_distance_AB_l117_117270


namespace opposite_of_neg_2023_l117_117007

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117007


namespace Gill_arrives_on_time_l117_117617

-- Definitions for times and speeds
def departure_time : ℕ := 9 * 60     -- departure time in minutes (09:00)
def segment1_distance : ℕ := 27      -- distance for the first segment in km
def segment2_distance : ℕ := 29      -- distance for the second segment in km
def speed : ℕ := 96                  -- speed in km/h
def stop_time : ℕ := 3               -- stop time at Lens in minutes

-- Calculate times for each segment and total travel time
def segment1_time : ℚ := segment1_distance * 60 / speed
def segment2_time : ℚ := segment2_distance * 60 / speed
def total_travel_time : ℚ := segment1_time + stop_time + segment2_time

-- Convert total travel time to minutes
def total_travel_time_mins : ℕ := total_travel_time.num

-- Arrival time
def arrival_time : ℕ := departure_time + total_travel_time_mins

-- Theorem stating the arrival time
theorem Gill_arrives_on_time : arrival_time = 9 * 60 + 38 := by
  -- Proof steps would go here, but we add sorry to conclude the statement
  sorry

end Gill_arrives_on_time_l117_117617


namespace moneySpentOnPaintbrushes_l117_117845

def totalExpenditure := 90
def costOfCanvases := 40
def costOfPaints := costOfCanvases / 2
def costOfEasel := 15
def costOfOthers := costOfCanvases + costOfPaints + costOfEasel

theorem moneySpentOnPaintbrushes : totalExpenditure - costOfOthers = 15 := by
  sorry

end moneySpentOnPaintbrushes_l117_117845


namespace blue_notebook_cost_l117_117405

theorem blue_notebook_cost
    (total_spent : ℕ)
    (total_notebooks : ℕ)
    (red_notebooks : ℕ) (red_cost : ℕ)
    (green_notebooks : ℕ) (green_cost : ℕ)
    (blue_notebooks : ℕ) (blue_total_cost : ℕ) 
    (blue_cost : ℕ) :
    total_spent = 37 →
    total_notebooks = 12 →
    red_notebooks = 3 →
    red_cost = 4 →
    green_notebooks = 2 →
    green_cost = 2 →
    blue_notebooks = total_notebooks - red_notebooks - green_notebooks →
    blue_total_cost = total_spent - red_notebooks * red_cost - green_notebooks * green_cost →
    blue_cost = blue_total_cost / blue_notebooks →
    blue_cost = 3 := 
    by sorry

end blue_notebook_cost_l117_117405


namespace area_ratio_of_squares_l117_117495

variable (x : ℝ)

theorem area_ratio_of_squares (h : 4 * x) : ((4 * x)^2) = 16 * (x^2) := 
sorry

end area_ratio_of_squares_l117_117495


namespace opposite_of_neg_2023_l117_117022

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l117_117022


namespace one_third_of_four_l117_117345

theorem one_third_of_four (h : 1/6 * 20 = 15) : 1/3 * 4 = 10 :=
sorry

end one_third_of_four_l117_117345


namespace moneySpentOnPaintbrushes_l117_117846

def totalExpenditure := 90
def costOfCanvases := 40
def costOfPaints := costOfCanvases / 2
def costOfEasel := 15
def costOfOthers := costOfCanvases + costOfPaints + costOfEasel

theorem moneySpentOnPaintbrushes : totalExpenditure - costOfOthers = 15 := by
  sorry

end moneySpentOnPaintbrushes_l117_117846


namespace birds_total_distance_covered_l117_117857

def round_trip_distance (building_materials_distance : ℕ) : ℕ :=
  2 * building_materials_distance

def total_distance_one_bird (round_trip_distance : ℕ) (num_trips : ℕ) : ℕ :=
  round_trip_distance * num_trips

def total_distance_two_birds (distance_one_bird : ℕ) : ℕ :=
  2 * distance_one_bird

theorem birds_total_distance_covered :
  let distance_one_trip := round_trip_distance 200 in
  let distance_one_bird := total_distance_one_bird distance_one_trip 10 in
  total_distance_two_birds distance_one_bird = 8000 :=
by sorry

end birds_total_distance_covered_l117_117857


namespace race_problem_equivalent_l117_117523

noncomputable def race_track_distance (D_paved D_dirt D_muddy : ℝ) : Prop :=
  let v1 := 100 -- speed on paved section in km/h
  let v2 := 70  -- speed on dirt section in km/h
  let v3 := 15  -- speed on muddy section in km/h
  let initial_distance := 0.5 -- initial distance in km (since 500 meters is 0.5 km)
  
  -- Time to cover paved section
  let t_white_paved := D_paved / v1
  let t_red_paved := (D_paved - initial_distance) / v1

  -- Times to cover dirt section
  let t_white_dirt := D_dirt / v2
  let t_red_dirt := D_dirt / v2 -- same time since both start at the same time on dirt

  -- Times to cover muddy section
  let t_white_muddy := D_muddy / v3
  let t_red_muddy := D_muddy / v3 -- same time since both start at the same time on mud

  -- Distances between cars on dirt and muddy sections
  ((t_white_paved - t_red_paved) * v2 = initial_distance) ∧ 
  ((t_white_paved - t_red_paved) * v3 = initial_distance)

-- Prove the distance between the cars when both are on the dirt and muddy sections is 500 meters
theorem race_problem_equivalent (D_paved D_dirt D_muddy : ℝ) : race_track_distance D_paved D_dirt D_muddy :=
by
  -- Insert proof here, for now we use sorry
  sorry

end race_problem_equivalent_l117_117523


namespace num_bricks_required_l117_117160

def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def brick_length_cm : ℝ := 20
def brick_width_cm : ℝ := 10
def cm_to_m (cm : ℝ) : ℝ := cm / 100
def area_of_courtyard : ℝ := courtyard_length * courtyard_width
def area_of_one_brick : ℝ := cm_to_m(brick_length_cm) * cm_to_m(brick_width_cm)
def num_of_bricks (courtyard_area brick_area : ℝ) : ℝ := courtyard_area / brick_area

theorem num_bricks_required :
  num_of_bricks area_of_courtyard area_of_one_brick = 20000 := by
sorry

end num_bricks_required_l117_117160


namespace waiter_tip_amount_l117_117550

theorem waiter_tip_amount (n n_no_tip E : ℕ) (h_n : n = 10) (h_no_tip : n_no_tip = 5) (h_E : E = 15) :
  (E / (n - n_no_tip) = 3) :=
by
  -- Proof goes here (we are only writing the statement with sorry)
  sorry

end waiter_tip_amount_l117_117550


namespace candidate_X_win_by_4_percent_l117_117875

-- Define the conditions
def ratio_rep_dem := 3 / 2
def total_voters (R: ℕ) := 5 * R
def vote_percentage_rep_for_X := 70 / 100
def vote_percentage_dem_for_X := 25 / 100
def vote_percentage_rep_for_Y := 30 / 100
def vote_percentage_dem_for_Y := 75 / 100

-- Define the votes for candidate X
def votes_for_X (R: ℕ) := (2.1 * R) + (0.5 * R)

-- Define the votes for candidate Y
def votes_for_Y (R: ℕ) := (0.9 * R) + (1.5 * R)

-- Define the percentage win for candidate X
def percentage_win (votes_X votes_Y total: ℤ) := ((votes_X - votes_Y) / total.toRational) * 100

theorem candidate_X_win_by_4_percent (R: ℕ) (H: R > 0):
  percentage_win (votes_for_X R).toInt (votes_for_Y R).toInt (total_voters R) = 4 := by
  sorry

end candidate_X_win_by_4_percent_l117_117875


namespace opposite_of_neg_2023_l117_117029

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117029


namespace sales_tax_paid_l117_117952

variable (total_cost : ℝ)
variable (tax_rate : ℝ)
variable (tax_free_cost : ℝ)

theorem sales_tax_paid (h_total : total_cost = 25) (h_rate : tax_rate = 0.10) (h_free : tax_free_cost = 21.7) :
  ∃ (X : ℝ), 21.7 + X + (0.10 * X) = 25 ∧ (0.10 * X = 0.3) := 
by
  sorry

end sales_tax_paid_l117_117952


namespace side_length_square_eq_6_l117_117525

theorem side_length_square_eq_6
  (width length : ℝ)
  (h_width : width = 2)
  (h_length : length = 18) :
  (∃ s : ℝ, s^2 = width * length) ∧ (∀ s : ℝ, s^2 = width * length → s = 6) :=
by
  sorry

end side_length_square_eq_6_l117_117525


namespace percentage_blue_flags_l117_117896

variable (C : ℕ) -- Total number of children
variable (E : ℕ) -- Even number of flags

-- Each child picks up two flags
axiom two_flags_per_child (E = 2 * C) : Prop

-- 70% of children have red flags
axiom seventy_percent_red_flags (0.7 * C) : ℕ

-- 30% of children have flags of both colors
axiom thirty_percent_both_colors (0.3 * C) : ℕ

-- Prove that 60% of children have blue flags
theorem percentage_blue_flags (h1: ∃ (C : ℕ), (2 * C = E)) 
                             (h2: ∃ (a : ℕ), a = (0.7 * C))
                             (h3: ∃ (b : ℕ), b = (0.3 * C)) : 
                             (∃ (p : ℕ), p = (0.6 * C)) :=
by sorry

end percentage_blue_flags_l117_117896


namespace M_plus_2N_equals_330_l117_117688

theorem M_plus_2N_equals_330 (M N : ℕ) :
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + 2 * N = 330 := by
  sorry

end M_plus_2N_equals_330_l117_117688


namespace mabel_petal_count_l117_117394

theorem mabel_petal_count : 
  (daisies_initial petals_per_daisy daisies_given : ℕ) 
  (h1 : daisies_initial = 5) 
  (h2 : petals_per_daisy = 8) 
  (h3 : daisies_given = 2) :
  (daisies_remaining : ℕ := daisies_initial - daisies_given) 
  (petals_remaining : ℕ := daisies_remaining * petals_per_daisy) 
  petals_remaining = 24 := 
by 
  sorry

end mabel_petal_count_l117_117394


namespace evaluate_expression_l117_117588

theorem evaluate_expression (x : ℝ) : 
  (36 + 12 * x) ^ 2 - (12^2 * x^2 + 36^2) = 864 * x :=
by
  sorry

end evaluate_expression_l117_117588


namespace calculate_expression_l117_117206

theorem calculate_expression :
  (Real.sqrt 25) + (Real.cbrt (-27)) - (abs (Real.sqrt 3 - 2)) = Real.sqrt 3 := by
  sorry

end calculate_expression_l117_117206


namespace opposite_of_negative_2023_l117_117055

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117055


namespace solve_equation_l117_117779

theorem solve_equation (x : ℚ) (h : x ≠ -5) : 
  (x^2 + 3*x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 := by
  sorry

end solve_equation_l117_117779


namespace projection_matrix_determinant_l117_117216

theorem projection_matrix_determinant (a c : ℚ) (h : (a^2 + (20 / 49 : ℚ) * c = a) ∧ ((20 / 49 : ℚ) * a + 580 / 2401 = 20 / 49) ∧ (a * c + (29 / 49 : ℚ) * c = c) ∧ ((20 / 49 : ℚ) * c + 841 / 2401 = 29 / 49)) :
  (a = 41 / 49) ∧ (c = 204 / 1225) := 
by {
  sorry
}

end projection_matrix_determinant_l117_117216


namespace orthocentric_tetrahedron_ratio_one_two_l117_117882

theorem orthocentric_tetrahedron_ratio_one_two (H M' N: Point) (T: Tetrahedron) 
(p1: Orthocenter T H) 
(p2: FaceCentroid T M') 
(p3: RayIntersectsCircumscribedSphere T H M' N): 
  dist H M' / dist M' N = 1 / 2 :=
sorry

end orthocentric_tetrahedron_ratio_one_two_l117_117882


namespace binomial_inequality_l117_117907

noncomputable def l (A B z : ℂ) : ℂ := A * z + B

def max_value_on_segment (A B M : ℂ) : Prop :=
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → ∥A * (x : ℂ) + B∥ ≤ M

theorem binomial_inequality
  (A B M : ℂ)
  (h_max : max_value_on_segment A B M) :
  ∀ z : ℂ, ∥l A B z∥ ≤ M * (complex.abs (z - 1) + complex.abs (z + 1)) :=
sorry

end binomial_inequality_l117_117907


namespace rod_length_l117_117616

theorem rod_length (n : ℕ) (l_cm_per_piece : ℕ) (total_length_cm : ℕ) (total_length_m : ℕ) :
  n = 20 →
  l_cm_per_piece = 85 →
  total_length_cm = n * l_cm_per_piece →
  total_length_m = total_length_cm / 100 →
  total_length_m = 17 :=
by
  intros n_eq l_eq len_eq conv_eq
  rw [n_eq, l_eq] at len_eq
  rw len_eq
  rw conv_eq
  sorry

end rod_length_l117_117616


namespace altitude_length_of_right_triangle_l117_117082

theorem altitude_length_of_right_triangle 
    (a b c : ℝ) 
    (h1 : a = 8) 
    (h2 : b = 15) 
    (h3 : c = 17) 
    (h4 : a^2 + b^2 = c^2) 
    : (2 * (1/2 * a * b))/c = 120/17 := 
by {
  sorry
}

end altitude_length_of_right_triangle_l117_117082


namespace sum_numbers_increased_by_three_l117_117942

theorem sum_numbers_increased_by_three :
  let range_start := 85
  let range_end := 100
  let increment := 3
  let n := range_end - range_start + 1
  let original_sum := n * (range_start + range_end) / 2
  let total_increment := n * increment
  let modified_sum := original_sum + total_increment
  modified_sum = 1528 := by
{
  let range_start := 85
  let range_end := 100
  let increment := 3
  let n := range_end - range_start + 1
  let original_sum := n * (range_start + range_end) / 2
  let total_increment := n * increment
  let modified_sum := original_sum + total_increment
  show modified_sum = 1528
}

end sum_numbers_increased_by_three_l117_117942


namespace remaining_macaroons_weight_l117_117727

theorem remaining_macaroons_weight (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (total_bags : ℕ) :
  (total_macaroons = 12) → 
  (weight_per_macaroon = 5) → 
  (total_bags = 4) → 
  let macaroons_per_bag := total_macaroons / total_bags in
  let weight_per_bag := macaroons_per_bag * weight_per_macaroon in
  let weight_eaten_by_steve := weight_per_bag in
  let total_weight := total_macaroons * weight_per_macaroon in
  let remaining_weight := total_weight - weight_eaten_by_steve in
  remaining_weight = 45 :=
by {
  sorry
}

end remaining_macaroons_weight_l117_117727


namespace emus_count_l117_117974

theorem emus_count (E : ℕ) (heads : ℕ) (legs : ℕ) 
  (h_heads : ∀ e : ℕ, heads = e) 
  (h_legs : ∀ e : ℕ, legs = 2 * e)
  (h_total : heads + legs = 60) : 
  E = 20 :=
by sorry

end emus_count_l117_117974


namespace minimum_value_of_expression_l117_117743

noncomputable def minimum_value_expression (x y z : ℝ) : ℝ :=
  1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z))

theorem minimum_value_of_expression : ∀ (x y z : ℝ), -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ -1 < z ∧ z < 0 → 
  minimum_value_expression x y z ≥ 2 := 
by
  intro x y z h
  sorry

end minimum_value_of_expression_l117_117743


namespace katherine_fruit_count_l117_117332

variables (apples pears bananas total_fruit : ℕ)

theorem katherine_fruit_count (h1 : apples = 4) 
  (h2 : pears = 3 * apples)
  (h3 : total_fruit = 21) 
  (h4 : total_fruit = apples + pears + bananas) : bananas = 5 := 
by sorry

end katherine_fruit_count_l117_117332


namespace triangle_is_isosceles_right_triangle_l117_117091

theorem triangle_is_isosceles_right_triangle
  (a b c : ℝ)
  (h1 : (a - b)^2 + (Real.sqrt (2 * a - b - 3)) + (abs (c - 3 * Real.sqrt 2)) = 0) :
  (a = 3) ∧ (b = 3) ∧ (c = 3 * Real.sqrt 2) :=
by
  sorry

end triangle_is_isosceles_right_triangle_l117_117091


namespace deposit_amount_is_140_l117_117145

-- Definitions based on the conditions
variables (P : ℝ)

-- Condition that 10% of the total price is already paid
def deposit_amount := 0.10 * P

-- Condition that the remaining amount to be paid is $1260
def remaining_amount := P - deposit_amount

-- Theorem statement
theorem deposit_amount_is_140 (h₁ : remaining_amount = 1260) : deposit_amount = 140 :=
by
  -- Replace the remaining amount by its definition
  have h₂ : P - deposit_amount = 1260 := h₁
  -- Substitute deposit_amount by its definition
  rw [deposit_amount] at h₂
  -- Factor out P
  have h₃ : P - 0.10 * P = 1260 := h₂
  -- Simplify to 0.90 * P
  have h₄ : 0.90 * P = 1260 := by linarith
  -- Solve for P
  have h₅ : P = 1260 / 0.90 := by linarith
  -- Calculate deposit_amount
  have h₆ : deposit_amount = 0.10 * (1260 / 0.90) := by rw [deposit_amount, h₅]
  -- Simplify to 140
  show deposit_amount = 140, by linarith

end deposit_amount_is_140_l117_117145


namespace imaginary_part_of_fraction_l117_117746

variable (i : ℂ) (h : i^2 = -1)

theorem imaginary_part_of_fraction : (⟨2 * i, 0⟩ / ⟨1 - i.re, 0 - i.im⟩).im = 1 := by sorry

end imaginary_part_of_fraction_l117_117746


namespace range_of_t_l117_117664

noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + 2 * t * x + t^2 else x + 1 / x + t

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f t 0 ≤ f t x) ↔ (0 ≤ t ∧ t ≤ 2) :=
by sorry

end range_of_t_l117_117664


namespace john_subtraction_number_l117_117103

theorem john_subtraction_number (a b : ℕ) (h1 : a = 40) (h2 : b = 1) :
  40^2 - ((2 * 40 * 1) - 1^2) = 39^2 :=
by
  -- sorry indicates the proof is skipped
  sorry

end john_subtraction_number_l117_117103


namespace area_of_triangle_PQR_l117_117851

-- Definitions based on conditions
def is_isosceles_right_triangle (P Q R : Type) (angleP : ℝ) (lengthPR : ℝ) := 
  angleP = 90 ∧ lengthPR = 8

-- The proof goal
theorem area_of_triangle_PQR {P Q R : Type} (h : is_isosceles_right_triangle P Q R 90 8) : 
  ∃ (area : ℝ), area = 32 :=
begin
  sorry
end

end area_of_triangle_PQR_l117_117851


namespace smallest_number_with_cards_l117_117483

-- Definitions corresponding to the conditions in a)
def card1 : ℕ := 2
def card2 : ℕ := 4

-- The proposition to prove
theorem smallest_number_with_cards : (card1 * 10 + card2) = 24 :=
by 
  rw [card1, card2]
  sorry

end smallest_number_with_cards_l117_117483


namespace number_of_sets_satisfying_union_l117_117747

theorem number_of_sets_satisfying_union :
  let A := {1, 2}
  ∃ (B : set ℕ), A ∪ B = {1, 2, 3} ∧ ∃! (B : set ℕ), (B = {3} ∨ B = {1, 3} ∨ B = {2, 3} ∨ B = {1, 2, 3}) :=
by
  sorry

end number_of_sets_satisfying_union_l117_117747


namespace find_m_of_ellipse_l117_117651

theorem find_m_of_ellipse (m : ℝ) (h₀ : m > 0) 
  (h₁ : ∃ (x y : ℝ), x^2 / 25 + y^2 / m^2 = 1) 
  (h₂ : ∀ c, (c = 4) → (∃ a b : ℝ, a = 5 ∧ b = m ∧ 25 = m^2 + 16)) :
  m = 3 :=
by
  sorry

end find_m_of_ellipse_l117_117651


namespace no_such_decreasing_h_exists_l117_117218

-- Define the interval [0, ∞)
def nonneg_reals := {x : ℝ // 0 ≤ x}

-- Define a decreasing function h on [0, ∞)
def is_decreasing (h : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → h x ≥ h y

-- Define the function f based on h
def f (h : ℝ → ℝ) (x : ℝ) : ℝ := (x^2 - x + 1) * h x

-- Define the increasing property for f on [0, ∞)
def is_increasing_on_nonneg_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x ≤ f y

theorem no_such_decreasing_h_exists :
  ¬ ∃ h : ℝ → ℝ, is_decreasing h ∧ is_increasing_on_nonneg_reals (f h) :=
by sorry

end no_such_decreasing_h_exists_l117_117218


namespace mary_needs_more_apples_l117_117398

theorem mary_needs_more_apples (total_pies : ℕ) (apples_per_pie : ℕ) (harvested_apples : ℕ) (y : ℕ) :
  total_pies = 10 → apples_per_pie = 8 → harvested_apples = 50 → y = 30 :=
by
  intro h1 h2 h3
  have total_apples_needed := total_pies * apples_per_pie
  have apples_needed_to_buy := total_apples_needed - harvested_apples
  have proof_needed : apples_needed_to_buy = y := sorry
  have proof_given : y = 30 := sorry
  have apples_needed := total_pies * apples_per_pie - harvested_apples
  exact proof_given

end mary_needs_more_apples_l117_117398


namespace greatest_two_digit_common_multiple_of_3_and_5_l117_117249

def is_two_digit (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100

def is_common_multiple_of_3_and_5 (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 5 = 0

theorem greatest_two_digit_common_multiple_of_3_and_5 : ∃ (x : ℕ), is_two_digit x ∧ is_common_multiple_of_3_and_5 x ∧ ∀ (y : ℕ), is_two_digit y ∧ is_common_multiple_of_3_and_5 y → y ≤ x :=
by {
  use 90,
  split,
  { -- 90 is a two-digit number
    show 90 ≥ 10 ∧ 90 < 100,
    sorry,
  },
  split,
  { -- 90 is a common multiple of 3 and 5
    show 90 % 3 = 0 ∧ 90 % 5 = 0,
    sorry,
  },
  { -- 90 is the greatest such number
    intros y hy,
    cases hy with hy1 hy2,
    sorry,
  }
}

end greatest_two_digit_common_multiple_of_3_and_5_l117_117249


namespace hyperbola_eccentricity_l117_117815

def hyperbola := { x : ℝ // (x^2 / 16) - (x.snd^2 / 9) = 1 }

noncomputable def eccentricity_of_hyperbola (h : hyperbola) : ℝ :=
  let a := 4
  let b := 3
  let c := real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_eccentricity :
  eccentricity_of_hyperbola ⟨4, 3, by sorry⟩ = 5 / 4 :=
by sorry

end hyperbola_eccentricity_l117_117815


namespace speed_in_still_water_l117_117496

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) :
  upstream_speed = 10 → downstream_speed = 20 → (upstream_speed + downstream_speed) / 2 = 15 :=
by
  intros h_upstream h_downstream
  rw [h_upstream, h_downstream]
  norm_num
  sorry

end speed_in_still_water_l117_117496


namespace valid_n_count_is_399_l117_117607

def count_valid_n : ℕ :=
  (count (λ n, (1 ≤ n ∧ n ≤ 1000) ∧ (
    ∃ (x : ℝ), n = ⌊x⌋ + ⌊2*x⌋ + ⌊3*x⌋ + ⌊4*x⌋)))

theorem valid_n_count_is_399 : count_valid_n = 399 :=
by
  sorry

end valid_n_count_is_399_l117_117607


namespace shopkeeper_profit_l117_117177

theorem shopkeeper_profit
  (CP SP : ℕ)
  (h1 : CP = 1)
  (h2 : SP = 2) :
  ((SP - CP) / CP) * 100 = 100 :=
by
  rw [h1, h2]
  simp
  sorry

end shopkeeper_profit_l117_117177


namespace hyperboloid_surface_l117_117608

theorem hyperboloid_surface (x y z : ℝ) (dx dy dz : ℝ) :
    (dx / (y * z) = dy / (x * z)) →
    (dx / (y * z) = dz / (x * y)) →
    (dx / (y * z) = dy / (x * z) = dz / (x * y)) →
    (x = 0) →
    (y^2 + z^2 = 1) →
    (2 * x^2 + 1 = y^2 + z^2) :=
    by sorry

end hyperboloid_surface_l117_117608


namespace final_price_correct_l117_117581

-- Definitions that follow the given conditions
def initial_price : ℝ := 150
def increase_percentage_year1 : ℝ := 1.5
def decrease_percentage_year2 : ℝ := 0.3

-- Compute intermediate values
noncomputable def price_end_year1 : ℝ := initial_price + (increase_percentage_year1 * initial_price)
noncomputable def price_end_year2 : ℝ := price_end_year1 - (decrease_percentage_year2 * price_end_year1)

-- The final theorem stating the price at the end of the second year
theorem final_price_correct : price_end_year2 = 262.5 := by
  sorry

end final_price_correct_l117_117581


namespace partition_nonnegative_integers_exists_l117_117983

def is_partition_possible (A B : Set ℕ) (r : Set ℕ → ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), r A n = r B n

theorem partition_nonnegative_integers_exists :
  ∃ (A B : Set ℕ), (∀ (n : ℕ), is_partition_possible A B (λ S n, S.countp (λ s1, ∃ s2, s2 ≠ s1 ∧ s1 + s2 = n))) :=
by
  sorry

end partition_nonnegative_integers_exists_l117_117983


namespace opposite_of_neg_2023_l117_117031

-- Define the notion of the additive inverse
def additive_inverse (a : Int) (b : Int) : Prop := a + b = 0

-- The specific problem: What is the opposite of -2023?
theorem opposite_of_neg_2023 : ∃ x : Int, additive_inverse (-2023) x ∧ x = 2023 := by
  use 2023
  unfold additive_inverse
  simp
  sorry

end opposite_of_neg_2023_l117_117031


namespace employees_after_reduction_l117_117178

def reduction (original : Float) (percent : Float) : Float :=
  original - (percent * original)

theorem employees_after_reduction :
  reduction 243.75 0.20 = 195 := by
  sorry

end employees_after_reduction_l117_117178


namespace linear_function_not_passing_through_third_quadrant_l117_117614

theorem linear_function_not_passing_through_third_quadrant
  (m : ℝ)
  (h : 4 + 4 * m < 0) : 
  ∀ x y : ℝ, (y = m * x - m) → ¬ (x < 0 ∧ y < 0) :=
by
  sorry

end linear_function_not_passing_through_third_quadrant_l117_117614


namespace find_alpha_and_evaluate_l117_117290

-- Define the power function f(x) = x ^ α
def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- Given conditions
theorem find_alpha_and_evaluate :
  (∃ α : ℝ, power_function α 8 = 2 ∧ power_function α (1 / 8) = 1 / 2) :=
sorry

end find_alpha_and_evaluate_l117_117290


namespace BP_value_l117_117884

-- Define the problem conditions and statement.
theorem BP_value
  (A B C D P : Point)
  (on_circle : ∀ point ∈ {A, B, C, D}, is_on_circle point)
  (intersect : P ∈ (line_through A C) ∧ P ∈ (line_through B D))
  (AP : Real := 10)
  (PC : Real := 2)
  (BD : Real := 9)
  (BP_lt_DP : ∃ x y : Real, BP = x ∧ DP = y ∧ x + y = BD ∧ x < y) :
  BP = 4 :=
by
  sorry -- Proof is omitted

end BP_value_l117_117884


namespace min_value_a2_plus_b2_l117_117268

theorem min_value_a2_plus_b2 (a b : ℝ) (σ : ℝ) (X : ℝ → ℝ) (hX : ∀ t, X t = pdf_normal t 1 σ) 
  (h_cond_a : P(X ≤ a) = P(X ≥ b)) (h_sum : a + b = 2) : a^2 + b^2 ≥ 2 := by
  sorry

# Check that the Lean environment builds successfully.

end min_value_a2_plus_b2_l117_117268


namespace tangent_line_eqn_l117_117383

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 1
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem tangent_line_eqn (h : f' x = 2) : 2 * x - y - Real.exp 1 + 1 = 0 :=
by
  sorry

end tangent_line_eqn_l117_117383


namespace discontinuity_of_first_kind_at_2_l117_117766

noncomputable def f (x : ℝ) : ℝ := (x - 2) / |x - 2|

theorem discontinuity_of_first_kind_at_2 : 
  discontinuous_at f 2 :=
by
  sorry

end discontinuity_of_first_kind_at_2_l117_117766


namespace reduced_price_per_dozen_l117_117870

theorem reduced_price_per_dozen 
  (P : ℝ) -- original price per apple
  (R : ℝ) -- reduced price per apple
  (A : ℝ) -- number of apples originally bought for Rs. 30
  (H1 : R = 0.7 * P) 
  (H2 : A * P = (A + 54) * R) :
  30 / (A + 54) * 12 = 2 :=
by
  sorry

end reduced_price_per_dozen_l117_117870


namespace f_const_one_l117_117990

-- Mathematical Translation of the Definitions
variable (f g h : ℕ → ℕ)

-- Given conditions
axiom h_injective : Function.Injective h
axiom g_surjective : Function.Surjective g
axiom f_eq : ∀ n, f n = g n - h n + 1

-- Theorem to Prove
theorem f_const_one : ∀ n, f n = 1 :=
by
  sorry

end f_const_one_l117_117990


namespace inequality_solution_set_l117_117084

theorem inequality_solution_set (a b : ℝ) (h : (a > 1) ∧ ((λ x, (ax + 1) / (x + b) > 1) = (λ x, x ∈ Ioo (-∞) (-1) ∪ Ioo (3) (∞)))) :
  (set_of (λ x, x^2 + a * x - 2 * b < 0) = set_of (λ x, -3 < x ∧ x < -2)) :=
by
  sorry

end inequality_solution_set_l117_117084


namespace exists_intersecting_line_l117_117493

theorem exists_intersecting_line (n : ℕ) (rays : Fin 2n → ℝ × ℝ → Prop) :
  (∀ i j, i ≠ j → ¬ parallel (rays i) (rays j)) → 
  (∀ i j, i ≠ j → initial_point (rays i) ≠ initial_point (rays j)) →
  ∃ ℓ, (∃ count : Fin 2n → ℕ, (∀ i, (intersects ℓ (rays i) → count i = 1)) ∧ (∑ i, count i = n))
         ∧ (∀ i, ¬ passes_through ℓ (initial_point (rays i))) :=
sorry

end exists_intersecting_line_l117_117493


namespace find_BP_l117_117892

-- Define points
variables {A B C D P : Type}  

-- Define lengths
variables (AP PC BP DP BD : ℝ)

-- Provided conditions
axiom h1 : AP = 10
axiom h2 : PC = 2
axiom h3 : BD = 9

-- Assume intersect and lengths relations setup
axiom intersect : BP < DP
axiom power_of_point : AP * PC = BP * DP

-- Target statement
theorem find_BP (h1 : AP = 10) (h2 : PC = 2) (h3 : BD = 9)
  (intersect : BP < DP) (power_of_point : AP * PC = BP * DP) : BP = 4 :=
  sorry

end find_BP_l117_117892


namespace rationalize_denominator_l117_117420

theorem rationalize_denominator :
  (35 / Real.sqrt 35) = Real.sqrt 35 :=
sorry

end rationalize_denominator_l117_117420


namespace solve_equation_l117_117778

theorem solve_equation (x : ℚ) (h : x ≠ -5) : 
  (x^2 + 3*x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 := by
  sorry

end solve_equation_l117_117778


namespace find_m_plus_n_l117_117111

noncomputable def area_triangle (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

def beta_fraction (a b c : ℕ) (omega : ℝ) : ℝ :=
  let area_ABC := area_triangle a b c in
  let area_PQRS := 1 / 2 * area_ABC in
  ((4 * area_PQRS) / omega^2)

theorem find_m_plus_n :
  let a := 12
  let b := 25
  let c := 17
  let omega := 25 / 2 in
  let area_ABC := area_triangle a b c in
  let area_PQRS := 1 / 2 * area_ABC in
  let beta := (4 * area_PQRS) / omega^2 in
  beta = 36 / 125 → 36 + 125 = 161 :=
by
  sorry

end find_m_plus_n_l117_117111


namespace domain_of_sqrt_log2_sub_2_l117_117602

noncomputable def domain_of_function : Set ℝ :=
  {x : ℝ | x ≥ 4}

theorem domain_of_sqrt_log2_sub_2 :
  ∀ x : ℝ, x ∈ domain_of_function ↔ ∃ y : ℝ, y = sqrt (log 2 x - 2) :=
by
  sorry

end domain_of_sqrt_log2_sub_2_l117_117602


namespace cos_PQR_expression_l117_117708

-- Define the properties and conditions of the tetrahedron
variables (P Q R S : Point) (k s t : ℝ)
variables (anglePQS anglePRS angleQRS : ℝ)
variables (PQ PR QR RS : ℝ)

-- Conditions
axiom PQS_90 : anglePQS = π / 2
axiom PRS_90 : anglePRS = π / 2
axiom QRS_90 : angleQRS = π / 2
axiom sin_PQS : s = sin(anglePQS)
axiom sin_PRS : t = sin(anglePRS)
axiom length_RS : RS = 2 * k

-- Main theorem
theorem cos_PQR_expression : 
  cos (∠ P Q R) = (k * (s^2 - t^2 - 4)) / sqrt((s^2 - t^2) * (s^2 - 4)) :=
sorry

end cos_PQR_expression_l117_117708


namespace bricks_required_l117_117156

def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def brick_length : ℝ := 20 / 100
def brick_width : ℝ := 10 / 100

theorem bricks_required :
  (courtyard_length * courtyard_width) / (brick_length * brick_width) = 20000 := 
    sorry

end bricks_required_l117_117156


namespace shaded_figure_perimeter_l117_117861

theorem shaded_figure_perimeter (a b : ℝ) (area_overlap : ℝ) (side_length : ℝ) (side_length_overlap : ℝ):
    a = 5 → b = 5 → area_overlap = 4 → side_length_overlap * side_length_overlap = area_overlap →
    side_length_overlap = 2 →
    ((4 * a) + (4 * b) - (4 * side_length_overlap)) = 32 :=
by
  intros
  sorry

end shaded_figure_perimeter_l117_117861


namespace al_initial_amount_l117_117189

-- Definitions
variables {a b c : ℝ}

-- Conditions
def condition1 : Prop := a + b + c = 1200
def condition2 : Prop := a - 150 + 3 * b + 3 * c = 1800
def distinct_amounts : Prop := b ≠ c ∧ b ≠ a ∧ c ≠ a

-- Theorem stating that given these conditions, a = 825
theorem al_initial_amount (h1 : condition1) (h2 : condition2) (h3 : distinct_amounts) : a = 825 :=
by
  sorry

end al_initial_amount_l117_117189


namespace movie_theater_total_revenue_l117_117446

noncomputable def revenue_from_matinee_tickets : ℕ := 20 * 5 * 1 / 2 + 180 * 5
noncomputable def revenue_from_evening_tickets : ℕ := 150 * 12 * 9 / 10 + 75 * 12 * 75 / 100 + 75 * 12
noncomputable def revenue_from_3d_tickets : ℕ := 60 * 23 + 25 * 20 * 85 / 100 + 15 * 20
noncomputable def revenue_from_late_night_tickets : ℕ := 30 * 10 * 12 / 10 + 20 * 10

noncomputable def total_revenue : ℕ :=
  revenue_from_matinee_tickets + revenue_from_evening_tickets +
  revenue_from_3d_tickets + revenue_from_late_night_tickets

theorem movie_theater_total_revenue : total_revenue = 6810 := by
  sorry

end movie_theater_total_revenue_l117_117446


namespace parallelogram_sides_l117_117453

theorem parallelogram_sides (x y : ℕ) 
  (h₁ : 2 * x + 3 = 9) 
  (h₂ : 8 * y - 1 = 7) : 
  x + y = 4 :=
by
  sorry

end parallelogram_sides_l117_117453


namespace intervals_of_monotonicity_interval_max_min_l117_117660

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x - 2

theorem intervals_of_monotonicity :
  (∀ (x : ℝ), x < -1 → deriv f x < 0) ∧ 
  (∀ (x : ℝ), -1 < x ∧ x < 3 → deriv f x > 0) ∧ 
  (∀ (x : ℝ), x > 3 → deriv f x < 0) := 
sorry

theorem interval_max_min :
  f 2 = 20 → f (-1) = -7 := 
sorry

end intervals_of_monotonicity_interval_max_min_l117_117660


namespace fraction_pq_eq_1_l117_117427

theorem fraction_pq_eq_1 (p q : ℝ) (h₀ : 0 < p ∧ 0 < q)
  (h₁ : log p / log 8 = log q / log 18)
  (h₂ : log q / log 18 = log (p + q) / log 32) :
  p / q = 1 :=
sorry

end fraction_pq_eq_1_l117_117427


namespace rectangular_field_area_l117_117912

def area_of_field (L W : ℕ) : ℕ := L * W

theorem rectangular_field_area :
  ∃ (L W : ℕ), L = 20 ∧ L + 2 * W = 60 ∧ area_of_field L W = 400 :=
by
  use 20, 20
  split; exact rfl
  split
  ·
    have eq1 : 20 + 2 * 20 = 60 := by norm_num
    exact eq1
  ·
    have eq2 : area_of_field 20 20 = 400 := by norm_num 
    exact eq2

end rectangular_field_area_l117_117912


namespace product_is_correct_l117_117568

theorem product_is_correct:
  12 * (-0.5) * (3/4) * 0.20 = - (9 / 10) := by 
  have h1 : -0.5 = - (1 / 2) := by norm_num
  have h2 : 0.20 = 1 / 5 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end product_is_correct_l117_117568


namespace gcd_10010_15015_l117_117237

theorem gcd_10010_15015 :
  Int.gcd 10010 15015 = 5005 :=
by 
  sorry

end gcd_10010_15015_l117_117237


namespace root_interval_l117_117440

def f (x : ℝ) : ℝ := 2^x + 3*x - 7

theorem root_interval : (f 1 < 0) ∧ (f 2 > 0) :=
by {
  -- lemma for f(1) < 0
  have h₁ : f 1 = 2^1 + 3*1 - 7 := by rfl,
  have h₂ : f 1 = 2 + 3 - 7 := by rw h₁,
  have h₃ : f 1 = -2 := by linarith,
  exact ⟨by linarith [h₃], -- f(1) < 0

  -- lemma for f(2) > 0
  have h₄ : f 2 = 2^2 + 3*2 - 7 := by rfl,
  have h₅ : f 2 = 4 + 6 - 7 := by rw h₄,
  have h₆ : f 2 = 3 := by linarith,
  exact ⟨by linarith [h₃], by linarith [h₆]⟩ -- f(2) > 0
 }⟩
  sorry  

end root_interval_l117_117440


namespace find_a10_l117_117669

variable {q : ℝ}
variable {a : ℕ → ℝ}

-- Sequence conditions
axiom geo_seq (n : ℕ) : a (n + 1) = a n * q
axiom positive_ratio : 0 < q
axiom condition_1 : a 2 = 1
axiom condition_2 : a 4 * a 8 = 2 * (a 5) ^ 2

theorem find_a10 : a 10 = 16 := by
  sorry

end find_a10_l117_117669


namespace cab_driver_income_l117_117897

variable (income1 income2 income4 income5 income_avg income_total income3 : ℤ)

theorem cab_driver_income :
  income1 = 300 →
  income2 = 150 →
  income4 = 200 →
  income5 = 600 →
  income_avg = 400 →
  income_total = 5 * income_avg →
  income1 + income2 + income3 + income4 + income5 = income_total →
  income3 = 750 :=
begin
  intros h1 h2 h4 h5 h_avg h_total h_sum,
  sorry
end

end cab_driver_income_l117_117897


namespace final_two_pieces_selling_price_l117_117148

variable (a : ℝ)

-- Definitions based on the given conditions
def cost_price_each : ℝ := a
def markup_percentage : ℝ := 0.5
def discount_percentage : ℝ := 0.2

theorem final_two_pieces_selling_price :
  let selling_price_each := cost_price_each a * (1 + markup_percentage a)
  let discounted_price_each := selling_price_each * (1 - discount_percentage a)
  2 * discounted_price_each = 2.4 * cost_price_each a :=
by 
  sorry

-- Proof goal is to show:
--  2 * (a * 1.5 * (1 - 0.2)) = 2.4 * a

end final_two_pieces_selling_price_l117_117148


namespace yeast_population_at_1_20_pm_l117_117772

def yeast_population (initial : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  initial * rate^time

theorem yeast_population_at_1_20_pm : 
  yeast_population 50 3 4 = 4050 :=
by
  -- Proof goes here
  sorry

end yeast_population_at_1_20_pm_l117_117772


namespace identify_coefficients_l117_117825

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : ℝ := x^2 - x

-- Proposition stating the coefficients of the quadratic equation
theorem identify_coefficients :
  ∃ (a b c : ℝ), (∀ x : ℝ, quadratic_equation x = a * x^2 + b * x + c) ∧ a = 1 ∧ b = -1 ∧ c = 0 :=
by
  -- We state the existence of coefficients a, b, and c
  use [1, -1, 0]
  -- State that quadratic_equation x should be equivalent to ax^2 + bx + c
  split
  -- Show that quadratic_equation x is equivalent to 1 * x^2 + (-1) * x + 0
  { intro x
    simp [quadratic_equation] }
  -- Conclude with the values of a, b, and c
  simp
  sorry  -- Placeholder for the proof

end identify_coefficients_l117_117825


namespace inequality_proof_l117_117438

variable (a b c d : ℝ)

theorem inequality_proof (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : a + b < c + d) : ac + bd > ab :=
sorry

end inequality_proof_l117_117438


namespace probability_value_expr_is_7_l117_117370

theorem probability_value_expr_is_7 : 
  let num_ones : ℕ := 15
  let num_ops : ℕ := 14
  let target_value : ℤ := 7
  let total_ways := 2 ^ num_ops
  let favorable_ways := (Nat.choose num_ops 11)  -- Ways to choose positions for +1's
  let prob := (favorable_ways : ℝ) / total_ways
  prob = 91 / 4096 := sorry

end probability_value_expr_is_7_l117_117370


namespace common_area_of_three_intersecting_circles_l117_117099

theorem common_area_of_three_intersecting_circles (r : ℝ) (h : 0 < r) :
  let sector_area := π * r^2 / 6
  let total_sector_area := 3 * sector_area
  let triangle_area := (√3 / 4) * r^2
  let common_area := total_sector_area - 2 * triangle_area
  common_area = r^2 * (π - √3) / 2 := by
  sorry

end common_area_of_three_intersecting_circles_l117_117099


namespace cannot_make_all_equal_cannot_make_all_divisible_by_3_l117_117362

-- Defining the cube vertices and initial conditions
def cube_vertices := Fin 8  -- The 8 vertices of the cube

def initial_numbers (v : cube_vertices) : ℕ :=
  match v with
  | ⟨0, _⟩ => 1
  | _      => 0

-- Definition of the operation: Adding 1 to both numbers at the ends of any edge of the cube
def operation (n : cube_vertices → ℕ) (e : (cube_vertices × cube_vertices)) : cube_vertices → ℕ :=
  fun v => if v = e.1 ∨ v = e.2 then n v + 1 else n v

-- Condition for equal numbers
def all_equal (n : cube_vertices → ℕ) : Prop :=
  ∃ k, ∀ v, n v = k

-- Condition for numbers divisible by 3
def all_divisible_by_3 (n : cube_vertices → ℕ) : Prop :=
  ∀ v, n v % 3 = 0

-- Defining the mathematical problem in Lean 4
theorem cannot_make_all_equal :
  ¬ (∃ (f : ℕ → cube_vertices → ℕ), (f 0 = initial_numbers) ∧ (∀ t, ∃ e, f (t + 1) = operation (f t) e) ∧ all_equal (f t)) := sorry

theorem cannot_make_all_divisible_by_3 :
  ¬ (∃ (f : ℕ → cube_vertices → ℕ), (f 0 = initial_numbers) ∧ (∀ t, ∃ e, f (t + 1) = operation (f t) e) ∧ all_divisible_by_3 (f t)) := sorry

end cannot_make_all_equal_cannot_make_all_divisible_by_3_l117_117362


namespace speed_of_second_train_is_40_kmh_l117_117474

def length_first_train : ℝ := 150
def length_second_train : ℝ := 160
def crossing_time : ℝ := 11.159107271418288
def speed_first_train_kmh : ℝ := 60

def mps_to_kmh (v : ℝ) : ℝ := v * 3600 / 1000
def kmh_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

noncomputable def speed_first_train_mps := kmh_to_mps speed_first_train_kmh

noncomputable def relative_speed := (length_first_train + length_second_train) / crossing_time

noncomputable def speed_second_train_mps := relative_speed - speed_first_train_mps

noncomputable def speed_second_train_kmh := mps_to_kmh speed_second_train_mps

theorem speed_of_second_train_is_40_kmh : speed_second_train_kmh = 40 := 
sorry

end speed_of_second_train_is_40_kmh_l117_117474


namespace opposite_of_negative_2023_l117_117046

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l117_117046


namespace parametric_curve_is_symmetric_l117_117163

def parametric_curve_symmetrical_about_x_axis : Prop :=
  ∀ t ∈ ℝ, (let x := (1 / 2) * Real.cos (2 * t) + Real.sin t ^ 2, 
            y := Real.cos t + Real.sin t 
            in y = Real.cos (-t) + Real.sin (-t))

theorem parametric_curve_is_symmetric :
  parametric_curve_symmetrical_about_x_axis :=
sorry

end parametric_curve_is_symmetric_l117_117163


namespace parabolas_reflection_intersect_l117_117180

section ParabolaReflection

variables {x y h₁ k₁ h₂ k₂ h₃ k₃ : ℝ}
variables {x₁ y₁ x₂ y₂ x₃ y₃ : ℝ}

-- Definitions of standard parabolas
def P1 : ℝ → ℝ := λ x, (x - h₁) * (x - h₁) + k₁
def P2 : ℝ → ℝ := λ x, (x - h₂) * (x - h₂) + k₂
def P3 : ℝ → ℝ := λ x, (x - h₃) * (x - h₃) + k₃

-- Vertices
def V1 := (h₁, k₁)
def V2 := (h₂, k₂)
def V3 := (h₃, k₃)

-- Intersection points
def A1 := (x₁, y₁)
def A2 := (x₂, y₂)
def A3 := (x₃, y₃)

-- Reflected vertices
def sV1 := (h₁, -k₁)
def sV2 := (h₂, -k₂)
def sV3 := (h₃, -k₃)

-- Reflected intersection points
def sA1 := (x₁, -y₁)
def sA2 := (x₂, -y₂)
def sA3 := (x₃, -y₃)

-- Lean statement asserting the intersections of reflected parabolas
theorem parabolas_reflection_intersect :
  (P1 x₁ = y₁ ∧ P2 x₁ = y₁) ∧
  (P2 x₂ = y₂ ∧ P3 x₂ = y₂) ∧
  (P3 x₃ = y₃ ∧ P1 x₃ = y₃) →
  (λ x, (x - h₂) * (x - h₂) - y₁) (h₁) = -k₁ ∧
  (λ x, (x - h₃) * (x - h₃) - y₂) (h₂) = -k₂ ∧
  (λ x, (x - h₁) * (x - h₁) - y₃) (h₃) = -k₃
:= by
  sorry

end ParabolaReflection

end parabolas_reflection_intersect_l117_117180


namespace tangent_line_at_one_minimum_a_range_of_a_l117_117634

-- Definitions for the given functions
def g (a x : ℝ) := a * x^2 - (a + 2) * x
noncomputable def h (x : ℝ) := Real.log x
noncomputable def f (a x : ℝ) := g a x + h x

-- Part (1): Prove the tangent line equation at x = 1 for a = 1
theorem tangent_line_at_one (x y : ℝ) (h_x : x = 1) (h_a : 1 = (1 : ℝ)) :
  x + y + 1 = 0 := by
  sorry

-- Part (2): Prove the minimum value of a given certain conditions
theorem minimum_a (a : ℝ) (h_a_pos : 0 < a) (h_x : 1 ≤ x ∧ x ≤ Real.exp 1)
  (h_fmin : ∀ x, f a x ≥ -2) : 
  a = 1 := by
  sorry

-- Part (3): Prove the range of values for a given a condition
theorem range_of_a (a x₁ x₂ : ℝ) (h_x : 0 < x₁ ∧ x₁ < x₂) 
  (h_f : ∀ x₁ x₂, (f a x₁ - f a x₂) / (x₁ - x₂) > -2) :
  0 ≤ a ∧ a ≤ 8 := by
  sorry

end tangent_line_at_one_minimum_a_range_of_a_l117_117634


namespace inverse_of_37_mod_53_l117_117603

theorem inverse_of_37_mod_53 :
  ∃ z, (37 * z) % 53 = 1 ∧ z = 10 :=
by 
  -- Step: Expressing 1 as Bézout Identity
  have h1 : 1 = 7 * 53 - 10 * 37, from sorry,
  -- Step: Deduce modular inverse
  existsi (10 : ℤ),
  -- Step: Confirm inverse property
  rw [←h1],
  -- Step: Modular arithmetic simplification
  exact sorry

end inverse_of_37_mod_53_l117_117603


namespace valid_outfit_combinations_l117_117325

-- Defining the conditions
def num_colors : ℕ := 8
def num_items : ℕ := 3 -- Shirt, pants, hat

-- The question is equivalent to stating:
theorem valid_outfit_combinations 
  (shirts pants hats : Fin num_colors → Fin num_colors → Fin num_colors → Prop)
  (h_no_two_same_color: ∀ (s p h : Fin num_colors), s ≠ p ∧ p ≠ h ∧ s ≠ h → shirts s p h ∧ pants s p h ∧ hats s p h) :
  let total_combinations := num_colors ^ num_items in
  let unwanted_combinations := 3 * (num_colors * (num_colors - 1)) in
  let valid_combinations := total_combinations - unwanted_combinations in
  valid_combinations = 344 := 
by 
  sorry

end valid_outfit_combinations_l117_117325


namespace inverse_of_f_is_g_l117_117441

-- Define the function f
def f (x : ℝ) := x^2

-- Define the domain condition for f
def f_domain (x : ℝ) := x < -2

-- Define the proposed inverse function g
def g (y : ℝ) := -√y

-- Define the domain condition for g
def g_domain (y : ℝ) := y > 4

-- The theorem stating that g is the inverse of f over the specified domains
theorem inverse_of_f_is_g : 
  ∀ (x : ℝ), f_domain x → f (g (f x)) = x :=
by
  sorry

end inverse_of_f_is_g_l117_117441


namespace carol_weight_l117_117454

theorem carol_weight :
  ∃ (a c : ℝ), a + c = 220 ∧ a + 2 * c = 280 ∧ c = 60 :=
by
  use [220 - 60, 60]
  split
  { sorry }
  split
  { sorry }
  { sorry }

end carol_weight_l117_117454


namespace triangle_equality_J_l117_117385

-- Definitions for geometric constructs
variable (A B C : Point) -- Points A, B, and C
variable (triangle_ABC : Triangle A B C) -- Triangle ABC
variable (I : Point) -- Incenter I
variable (incenter_I : IsIncenter I triangle_ABC) -- I is incenter of triangle ABC
variable (J : Point) -- Point J
variable (line_AI : Line AI) -- Line AI
variable (intersects_circumcircle : IntersectsCircumcircle line_AI J triangle_ABC) -- Line AI intersects circumcircle of triangle ABC at J

-- The final theorem statement to be proven
theorem triangle_equality_J : JB = JC = JI :=
sorry

end triangle_equality_J_l117_117385


namespace plank_cost_l117_117939

-- Defining the conditions
def birdhouses_to_planks (birdhouses : ℕ) : ℕ := 
  7 * birdhouses

def birdhouses_to_nails (birdhouses : ℕ) : ℕ := 
  20 * birdhouses

def total_nail_cost (nails : ℕ) : ℝ := 
  0.05 * nails

def total_plank_cost (total_cost : ℝ) (nail_cost : ℝ) : ℝ := 
  total_cost - nail_cost

def cost_per_plank (total_plank_cost : ℝ) (total_planks : ℕ) : ℝ := 
  total_plank_cost / total_planks

-- Defining the main problem statement
theorem plank_cost :
  cost_per_plank (total_plank_cost 88 (total_nail_cost (birdhouses_to_nails 4))) (birdhouses_to_planks 4) = 3 := 
sorry

end plank_cost_l117_117939


namespace tangent_length_equal_l117_117999

variable {A B C D E F G H M : Type*}
variables [metric_space A]
variables [metric_space B]
variables [metric_space C]
variables [metric_space D]
variables [metric_space E]
variables [metric_space F]
variables [metric_space G]
variables [metric_space H]
variables [metric_space M]

def tangent_segments (A B C D E F G H M : Type*) :=
  ∃ (EF_within_triangle EF_outside_triangle : ℝ),
    ∃ (G H: Type*), EF_within_triangle = EF_outside_triangle

theorem tangent_length_equal (A B C D E F G H M : Type*)
  (EF_within_triangle EF_outside_triangle : ℝ)
  (tangent_segments : ∃ (G H : Type*), EF_within_triangle = EF_outside_triangle) :
  EF_within_triangle = EF_outside_triangle :=
sorry

end tangent_length_equal_l117_117999
