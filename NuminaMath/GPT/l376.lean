import Mathlib

namespace circuminradius_inequality_l376_376103

theorem circuminradius_inequality
  (A B C : ℝ) -- Angles of the triangle
  (R r : ℝ) -- Circumradius and inradius
  (h_triangle : A + B + C = real.pi) -- Condition that angles sum to π (a triangle)
  (h_R : ∀ a b c : ℝ, a / (2 * real.sin (a / 2) * (1 - real.sin (a / 2))) = R) -- Condition of circumradius
  (h_r : ∀ a b c : ℝ, r = a * b * c / (4 * R)) -- Condition of inradius
  : R ≥ r / (2 * (real.sin (A / 2)) * (1 - real.sin (A / 2))) :=
sorry

end circuminradius_inequality_l376_376103


namespace sin_405_eq_sqrt2_div_2_l376_376319

theorem sin_405_eq_sqrt2_div_2 :
  Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_eq_sqrt2_div_2_l376_376319


namespace percentage_filled_l376_376435

/-- Statement of the theorem -/
theorem percentage_filled (total_seats vacant_seats seats_filled: ℕ) :
  total_seats = 600 → vacant_seats = 150 → seats_filled = total_seats - vacant_seats → 
  seats_filled = 450 ∧ (seats_filled: ℚ / total_seats) * 100 = 75 := 
by
  intros h_total h_vacant h_filled
  sorry

end percentage_filled_l376_376435


namespace rate_per_meter_eq_2_5_l376_376346

-- Definitions of the conditions
def diameter : ℝ := 14
def total_cost : ℝ := 109.96

-- The theorem to be proven
theorem rate_per_meter_eq_2_5 (π : ℝ) (hπ : π = 3.14159) : 
  diameter = 14 ∧ total_cost = 109.96 → (109.96 / (π * 14)) = 2.5 :=
by
  sorry

end rate_per_meter_eq_2_5_l376_376346


namespace number_of_zeros_in_interval_l376_376696

noncomputable def f (x : ℝ) : ℝ :=
if hx : -1 < x ∧ x ≤ 4 then
  x^2 - 2^x
else
  sorry -- Placeholder for the extended definition using functional equation

theorem number_of_zeros_in_interval :
  (∀ x : ℝ, f x + f (x + 5) = 16) →
  (∀ x : ℝ, x ∈ Icc (-1 : ℝ) 4 → f x = x^2 - 2^x) →
  (∃ n : ℕ, n = 604 ∧
    ∀ x : ℝ, x ∈ Ioc 0 2013 → (f x = 0 ↔ x ∈ Finset.range 604)) :=
begin
  -- Placeholder for the proof
  sorry 
end

end number_of_zeros_in_interval_l376_376696


namespace proposition_does_not_hold_at_2_l376_376281

variable (P : ℕ+ → Prop)
open Nat

theorem proposition_does_not_hold_at_2
  (h₁ : ¬ P 3)
  (h₂ : ∀ k : ℕ+, P k → P (k + 1)) :
  ¬ P 2 :=
by
  sorry

end proposition_does_not_hold_at_2_l376_376281


namespace period_change_l376_376429

theorem period_change {f : ℝ → ℝ} (T : ℝ) (hT : 0 < T) (h_period : ∀ x, f (x + T) = f x) (α : ℝ) (hα : 0 < α) :
  ∀ x, f (α * (x + T / α)) = f (α * x) :=
by
  sorry

end period_change_l376_376429


namespace initial_avg_height_l376_376182

-- Lean 4 statement for the given problem
theorem initial_avg_height (A : ℝ) (n : ℕ) (wrong_height correct_height actual_avg init_diff : ℝ)
  (h_class_size : n = 35)
  (h_wrong_height : wrong_height = 166)
  (h_correct_height : correct_height = 106)
  (h_actual_avg : actual_avg = 183)
  (h_init_diff : init_diff = wrong_height - correct_height)
  (h_total_height_actual : n * actual_avg = 35 * 183)
  (h_total_height_wrong : n * A = 35 * actual_avg - init_diff) :
  A = 181 :=
by {
  -- The problem and conditions are correctly stated. The proof is skipped with sorry.
  sorry
}

end initial_avg_height_l376_376182


namespace double_tetrahedron_prism_volume_l376_376220

noncomputable def prism_volume (a : ℝ) : ℝ :=
  (√2 / 54) * a^3

theorem double_tetrahedron_prism_volume (a : ℝ) :
  let volume := prism_volume a in
  volume = (√2 / 54) * a^3 := 
by
  sorry

end double_tetrahedron_prism_volume_l376_376220


namespace triangle_area_PQR_l376_376457

noncomputable def area_TRIANGLE_PQR (PQ PR PM : ℕ) :=
  let semi_perimeter := (PQ + PR + 24) / 2
  let area := Real.sqrt (semi_perimeter * (semi_perimeter - PQ) * (semi_perimeter - PR) * (semi_perimeter - 24))
  area

theorem triangle_area_PQR
  (PQ PR PM : ℕ)
  (hPQ : PQ = 8)
  (hPR : PR = 18)
  (hPM : PM = 12) :
  area_TRIANGLE_PQR PQ PR PM = Real.sqrt 2975 :=
by
  -- Given conditions
  have h_semi_perimeter : (PQ + PR + 24) / 2 = 25 := sorry
  -- Calculate the area of triangle \(PRS\) and prove it equals \(\sqrt{2975}\)
  have h_area : Real.sqrt (25 * (25 - PQ) * (25 - PR) * (25 - 24)) = Real.sqrt 2975 := sorry
  -- Hence, the area of \(PQR\) is also \(\sqrt{2975}\) due to the congruency of \(PQR\) and \(PRS\)
  exact h_area

end triangle_area_PQR_l376_376457


namespace max_min_ratio_l376_376368

variable (a : ℝ)
variable (ABCD : Type)
variable (Γ : Type)
variable 
[isConvexQuadrilateral ABCD]
[lengthOfMaxSide ABCD = a]
[lengthOfMinSide ABCD = Real.sqrt (4 - a^2)]
[isInscribedInCircle ABCD Γ]
[isUnitCircle Γ]
[centerInQuadrilateral Γ ABCD]

theorem max_min_ratio (h1 : Real.sqrt 2 < a) (h2 : a < 2) : 
max_min_ratio (A : ABCD) (B : ABCD) (C : ABCD) (D : ABCD) : 
(max_ratio := 8 / (a^2 * (4 - a^2))) ∧ 
(min_ratio := 4 / (a * Real.sqrt (4 - a^2))) := sorry

end max_min_ratio_l376_376368


namespace find_a_l376_376076

theorem find_a (a : ℝ) (x₁ x₂ : ℝ) :
  (2 * x₁ + 1 = 3) →
  (2 - (a - x₂) / 3 = 1) →
  (x₁ = x₂) →
  a = 4 :=
by
  intros h₁ h₂ h₃
  sorry

end find_a_l376_376076


namespace Jed_cards_after_4_weeks_l376_376108

theorem Jed_cards_after_4_weeks :
  (∀ n: ℕ, (if n % 2 = 0 then 20 + 4*n - 2*n else 20 + 4*n - 2*(n-1)) = 40) :=
by {
  sorry
}

end Jed_cards_after_4_weeks_l376_376108


namespace number_of_ah_tribe_residents_l376_376433

theorem number_of_ah_tribe_residents 
  (P A U : Nat) 
  (H1 : 16 < P) 
  (H2 : P ≤ 17) 
  (H3 : A + U = P) 
  (H4 : U = 2) : 
  A = 15 := 
by
  sorry

end number_of_ah_tribe_residents_l376_376433


namespace rectangle_perimeter_l376_376655

noncomputable def perimeter_rect (WA BX WY XZ : ℤ) : Real :=
  (2 * (Real.sqrt (WA^2 + BX^2)) * (Real.sqrt (WY^2 + XZ^2)))

theorem rectangle_perimeter
  (ABCD WXYZ : Type) [Rectangle ABCD] [Rhombus WXYZ]
  (WA BX WY XZ : ℤ)
  (h1 : WA = 10) (h2 : BX = 25) (h3 : WY = 35) (h4 : XZ = 50)
  (h5 : InscribedWXYZ ABCD WXYZ)
  : perimeter_rect WA BX WY XZ = (2 * (200 / Real.sqrt 29 + 5 * Real.sqrt 61)) :=
by
  -- define the entity perimeter_rect
  sorry

end rectangle_perimeter_l376_376655


namespace smallest_k_exists_l376_376257

theorem smallest_k_exists :
  ∃ k : ℕ, k > 0 ∧ (∀ (x : ℕ → ℕ → ℝ),
  (∀ i j, i ∈ {1, 2, ..., 100} → j ∈ {1, 2, ..., 25} → x i j ≥ 0) →
  (∀ i, i ∈ {1, 2, ..., 100} → ∑ j in finset.range(25), x i j ≤ 1) →
  let x' (i j : ℕ) := (finset.univ.image (λ (f : ℕ × ℕ), x (f.1) (f.2))).val.nth_le j sorry in
  (∀ i ≥ k, ∑ j in finset.range(25), x' i j ≤ 1)) :=
begin
  sorry
end

end smallest_k_exists_l376_376257


namespace probability_of_attending_10_minutes_of_second_class_l376_376659

theorem probability_of_attending_10_minutes_of_second_class :
  let first_class_start := 8 * 60
  let first_class_end := first_class_start + 40
  let break_between_classes := 10
  let second_class_start := first_class_end + break_between_classes
  let second_class_end := second_class_start + 40
  let arrival_start := 9 * 60 + 10
  let arrival_end := 10 * 60
  let arrival_window := arrival_end - arrival_start
  let required_arrival_start := arrival_start
  let required_arrival_end := second_class_start + 30
  let required_arrival_window := required_arrival_end - required_arrival_start in
  (required_arrival_window : ℝ) / arrival_window = 1 / 5 :=
by 
  -- proof
  sorry

end probability_of_attending_10_minutes_of_second_class_l376_376659


namespace assignments_for_28_points_l376_376086

def points_per_assignment (n : ℕ) : ℕ := (n + 6) / 7

theorem assignments_for_28_points : 
  (∑ n in range 28, points_per_assignment (n + 1)) = 70 := 
by
  -- This statement calculates the total number of homework assignments for 28 points
  -- given the points per assignment condition
  sorry

end assignments_for_28_points_l376_376086


namespace prob_not_negative_review_A_prob_two_positive_reviews_choose_platform_A_l376_376942

-- Definitions of the problem conditions
def positive_reviews_A := 75
def neutral_reviews_A := 20
def negative_reviews_A := 5
def total_reviews_A := 100

def positive_reviews_B := 64
def neutral_reviews_B := 8
def negative_reviews_B := 8
def total_reviews_B := 80

-- Prove the probability that a buyer's evaluation on platform A is not a negative review
theorem prob_not_negative_review_A : 
  (1 - negative_reviews_A / total_reviews_A) = 19 / 20 := by
  sorry

-- Prove the probability that exactly 2 out of 4 (2 from A and 2 from B) buyers give a positive review
theorem prob_two_positive_reviews :
  ((positive_reviews_A / total_reviews_A) ^ 2 * (1 - positive_reviews_B / total_reviews_B) ^ 2 + 
  2 * (positive_reviews_A / total_reviews_A) * (1 - positive_reviews_A / total_reviews_A) * 
  (positive_reviews_B / total_reviews_B) * (1 - positive_reviews_B / total_reviews_B) +
  (1 - positive_reviews_A / total_reviews_A) ^ 2 * (positive_reviews_B / total_reviews_B) ^ 2) = 
  73 / 400 := by
  sorry

-- Choose platform A based on the given data
theorem choose_platform_A :
  let E_A := (5 * 0.75 + 3 * 0.2 + 1 * 0.05)
  let D_A := (5 - E_A) ^ 2 * 0.75 + (3 - E_A) ^ 2 * 0.2 + (1 - E_A) ^ 2 * 0.05
  let E_B := (5 * 0.8 + 3 * 0.1 + 1 * 0.1)
  let D_B := (5 - E_B) ^ 2 * 0.8 + (3 - E_B) ^ 2 * 0.1 + (1 - E_B) ^ 2 * 0.1
  (E_A = E_B) ∧ (D_A < D_B) → choose_platform = "Platform A" := by
  sorry

end prob_not_negative_review_A_prob_two_positive_reviews_choose_platform_A_l376_376942


namespace excircle_centers_form_acute_triangle_l376_376737

open EuclideanGeometry

variable {A B C O1 O2 O3 : Point}

-- Define that O1, O2, and O3 are the centers of the excircles of triangle ABC.

def is_excenter (A B C O : Point) : Prop :=
  exists (O : Point),
    O = center_of_excircle_opposite A B C

-- The theorem to be proven.
theorem excircle_centers_form_acute_triangle 
  (h₁ : is_excenter A B C O1)
  (h₂ : is_excenter B C A O2)
  (h₃ : is_excenter C A B O3) :
  is_acutetriangle O1 O2 O3 :=
sorry

end excircle_centers_form_acute_triangle_l376_376737


namespace sequence_value_a3_l376_376929

theorem sequence_value_a3 : 
  ∀ (a : ℕ → ℝ) (λ : ℝ), 
  a 1 = 1 → a 2 = 3 → 
  (∀ n : ℕ, a (n + 1) = (2 * n - λ) * a n) → 
  a 3 = 15 :=
by
  intros a λ a1 a2 h
  have hλ : λ = -1 := by sorry
  have new_h : ∀ n : ℕ, a (n + 1) = (2 * n + 1) * a n := by sorry
  sorry

end sequence_value_a3_l376_376929


namespace solve_equation_l376_376760

variable {x y : ℝ}

theorem solve_equation (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2: y ≠ 4) (h : (3 / x) + (2 / y) = 5 / 6) :
  x = 18 * y / (5 * y - 12) :=
sorry

end solve_equation_l376_376760


namespace find_norm_b_and_angle_l376_376405

variables (a b : EuclideanSpace ℝ (Fin 3))
axiom angle_ab : angle a b = real.pi / 3 -- 60° in radians
axiom norm_a : ∥a∥ = 1
axiom norm_expr : ∥2 • a - b∥ = 2 * real.sqrt 3

theorem find_norm_b_and_angle :
  ∥b∥ = 4 ∧ angle b (2 • a - b) = 5 * real.pi / 6 := -- 150° in radians
sorry

end find_norm_b_and_angle_l376_376405


namespace partition_piles_into_two_groups_l376_376227

open Nat

def pebbles_in_piles (pebbles : List Nat) := pebbles.length = 50 ∧ pebbles.all (λ n => 1 ≤ n ∧ n ≤ 50)

def total_pebbles (pebbles : List Nat) := List.sum pebbles = 100

theorem partition_piles_into_two_groups (pebbles : List Nat) (h1 : pebbles_in_piles pebbles) (h2 : total_pebbles pebbles) :
  ∃ (group1 group2 : List Nat), group1 ⊆ pebbles ∧ group2 ⊆ pebbles ∧ List.pairwise Disjoint group1 group2 ∧ List.sum group1 = 50 ∧ List.sum group2 = 50 := by
    sorry

end partition_piles_into_two_groups_l376_376227


namespace triangle_base_length_l376_376535

theorem triangle_base_length (A h b : ℝ) 
  (h1 : A = 30) 
  (h2 : h = 5) 
  (h3 : A = (b * h) / 2) : 
  b = 12 :=
by
  sorry

end triangle_base_length_l376_376535


namespace find_value_a_pow_2m_plus_n_l376_376365

theorem find_value_a_pow_2m_plus_n (a : ℝ) (m n : ℝ) (h1 : log a 2 = m) (h2 : log a 3 = n) : a^(2 * m + n) = 12 :=
by 
  sorry

end find_value_a_pow_2m_plus_n_l376_376365


namespace side_length_of_equilateral_triangle_l376_376297

theorem side_length_of_equilateral_triangle
  (a b c d : ℝ)
  (trapezoid_perimeter : ℝ)
  (h1 : a = b)
  (h2 : b = c)
  (h3 : c = (d / 2))
  (h4 : trapezoid_perimeter = 10 + 5 * Real.sqrt 3)
  (h5 : 3 * (a + b + c + d/2) = trapezoid_perimeter) :
  -- We prove that the side length of the equilateral triangle is \( 6 + 3 \sqrt{3} \).
  let a := (10 + 5 * Real.sqrt 3) / 5 in
  3 * a = 6 + 3 * Real.sqrt 3 := 
by
  sorry

end side_length_of_equilateral_triangle_l376_376297


namespace monthly_repayment_amount_l376_376149

-- Define the parameters
def house_price := 800000 -- in yuan
def down_payment := 300000 -- in yuan
def loan_amount := house_price - down_payment -- in yuan
def monthly_interest_rate := 0.005 -- 0.5%
def loan_repayment_period := 30 * 12 -- in months
def target_monthly_repayment := 2997.75 -- in yuan

-- Define the sequence for loan balance
def loan_balance (n : ℕ) (a_n : ℕ) (x : ℕ) : ℕ :=
  a_n * (1 + monthly_interest_rate) - x

-- Problem statement: Mr. Cheng should repay this much each month
theorem monthly_repayment_amount (x : ℝ) : x = target_monthly_repayment :=
  sorry

end monthly_repayment_amount_l376_376149


namespace no_solutions_for_equation_l376_376524

theorem no_solutions_for_equation:
  ¬ ∃ x : ℝ, sqrt (4 + 2 * x) + sqrt (6 + 3 * x) + sqrt (8 + 4 * x) = 9 + (3 / 2) * x := 
sorry

end no_solutions_for_equation_l376_376524


namespace ron_tickets_sold_l376_376516

theorem ron_tickets_sold 
  (R K : ℕ) 
  (h1 : R + K = 20) 
  (h2 : 2 * R + 9 / 2 * K = 60) : 
  R = 12 := 
by 
  sorry

end ron_tickets_sold_l376_376516


namespace b2_values_count_l376_376660

open Int
open Nat

theorem b2_values_count :
  let sequence (b : ℕ → ℕ) := ∀ n, b (n + 2) = abs (b (n + 1) - b n)
  in ∀ b : ℕ → ℕ, b 1 = 1001 ∧ (b 2 < 1001) ∧ b 2023 = 0 ∧ sequence b →
  Finset.card ((Finset.filter (λ x, x < 1001 ∧ even x ∧ gcd 1001 x = 1) (Finset.range 1001))) = 386 :=
by
  sorry

end b2_values_count_l376_376660


namespace walls_per_room_is_8_l376_376112

-- Definitions and conditions
def total_rooms : Nat := 10
def green_rooms : Nat := 3 * total_rooms / 5
def purple_rooms : Nat := total_rooms - green_rooms
def purple_walls : Nat := 32
def walls_per_room : Nat := purple_walls / purple_rooms

-- Theorem to prove
theorem walls_per_room_is_8 : walls_per_room = 8 := by
  sorry

end walls_per_room_is_8_l376_376112


namespace interest_calculation_years_l376_376074

theorem interest_calculation_years (P n : ℝ) (r : ℝ) (SI CI : ℝ)
  (h₁ : SI = P * r * n / 100)
  (h₂ : r = 5)
  (h₃ : SI = 50)
  (h₄ : CI = P * ((1 + r / 100)^n - 1))
  (h₅ : CI = 51.25) :
  n = 2 := by
  sorry

end interest_calculation_years_l376_376074


namespace intersection_M_N_l376_376403

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 2^x > 1}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l376_376403


namespace hyperbola_equation_l376_376765

theorem hyperbola_equation (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_asymptote: a/b = Real.sqrt 3) (h_focus: ∃ c, c = 6 ∧ c^2 = a^2 + b^2) :
  (∃ (a : ℝ) (b : ℝ), ∀ (x y : ℝ), (a > 0 ∧ b > 0 ∧ (a/b = Real.sqrt 3) ∧ one_focus c (-6)) →
  (a = Real.sqrt 27 ∧ b = Real.sqrt 9 ∧ y^2 / 27 - x^2 / 9 = 1)) :=
by
  sorry

end hyperbola_equation_l376_376765


namespace part1_part2_part3_l376_376008

def f : ℕ → ℕ 
| 0     := 0                -- Using zero as dummy value for clarity
| 1     := 1
| 2     := 2
| (n+2) := f (n + 2 - f (n + 1)) + f (n + 1 - f n)

theorem part1 (n : ℕ) (hn : n ≥ 1) : 0 ≤ f (n + 1) - f n ∧ f (n + 1) - f n ≤ 1 :=
sorry

theorem part2 (n : ℕ) (hn : n ≥ 1) (hn_odd : f n % 2 = 1) : f (n + 1) = f n + 1 :=
sorry

theorem part3 : ∃ n : ℕ, f n = 1025 :=
sorry

end part1_part2_part3_l376_376008


namespace common_point_of_concurrence_l376_376910

theorem common_point_of_concurrence
  (A B C R K L M P Q : Point)
  (circumcircle : Circle)
  (triangle_ABC : is_triangle A B C)
  (angle_bisectors : ∀ (X : Point), X ∈ {A, B, C} → bisector_of ∡ (X.1 -- center X's angle at the origin))
  (circum_circle_points : K ∈ circumcircle ∧ L ∈ circumcircle ∧ M ∈ circumcircle)
  (KLM_triangle : K ≠ L ∧ L ≠ M ∧ M ≠ K)
  (R_internal : R ∈ segment A B)
  (RP_parallel_AK : parallel R P A K)
  (BP_perpendicular_BL : perpendicular B P B L)
  (RQ_parallel_BL : parallel R Q B L)
  (AQ_perpendicular_AK : perpendicular A Q A K) :
  ∃ (X : Point), X ∈ line K P ∧ X ∈ line L Q ∧ X ∈ line M R :=
begin
  sorry
end

end common_point_of_concurrence_l376_376910


namespace payment_equation_1_payment_equation_2_cost_effective_40_combined_cost_effective_40_l376_376649

namespace ShoppingMall

def tea_set_price : ℕ := 200
def tea_bowl_price : ℕ := 20
def discount_option_1 (x : ℕ) : ℕ := 20 * x + 5400
def discount_option_2 (x : ℕ) : ℕ := 19 * x + 5700
def combined_option_40 : ℕ := 6000 + 190

theorem payment_equation_1 (x : ℕ) (hx : x > 30) : 
  discount_option_1 x = 20 * x + 5400 :=
by sorry

theorem payment_equation_2 (x : ℕ) (hx : x > 30) : 
  discount_option_2 x = 19 * x + 5700 :=
by sorry

theorem cost_effective_40 : discount_option_1 40 < discount_option_2 40 :=
by sorry

theorem combined_cost_effective_40 : combined_option_40 < discount_option_1 40 ∧ combined_option_40 < discount_option_2 40 :=
by sorry

end ShoppingMall

end payment_equation_1_payment_equation_2_cost_effective_40_combined_cost_effective_40_l376_376649


namespace tangent_line_to_circle_l376_376707

noncomputable def r_tangent_to_circle : ℝ := 4

theorem tangent_line_to_circle
  (x y r : ℝ)
  (circle_eq : x^2 + y^2 = 2 * r)
  (line_eq : x - y = r) :
  r = r_tangent_to_circle :=
by
  sorry

end tangent_line_to_circle_l376_376707


namespace find_cost_price_l376_376668

noncomputable def cost_price (cp : ℝ) : Prop := 
  let mp := cp * 1.45 in       -- Markup price
  let sp := mp - 45 in         -- Selling price after discount
  sp = cp * 1.20               -- Selling price relation with profit

theorem find_cost_price : ∃ cp : ℝ, cost_price cp ∧ cp = 180 :=
by
  use 180
  rw [cost_price]
  dsimp
  -- sorry to inform Lean that the proof is omitted.
  sorry

end find_cost_price_l376_376668


namespace avg_values_l376_376342

theorem avg_values (z : ℝ) : (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end avg_values_l376_376342


namespace largest_base7_three_digit_is_342_l376_376915

-- Definition of the base-7 number 666
def base7_666 : ℕ := 6 * 7^2 + 6 * 7^1 + 6 * 7^0

-- The largest decimal number represented by a three-digit base-7 number is 342
theorem largest_base7_three_digit_is_342 : base7_666 = 342 := by
  sorry

end largest_base7_three_digit_is_342_l376_376915


namespace beads_per_earring_l376_376462

theorem beads_per_earring :
  (∀ (beads_necklace beads_bracelet total_beads: ℕ), 
    (beads_necklace = 20) →
    (beads_bracelet = 10) →
    (total_beads = 325) →
    let necklaces := 10 + 2 in 
    let bracelets := 5 in 
    let earrings := 7 in 
    let used_beads_for_necklaces := necklaces * beads_necklace in 
    let used_beads_for_bracelets := bracelets * beads_bracelet in 
    let used_beads_for_earrings := total_beads - (used_beads_for_necklaces + used_beads_for_bracelets) in 
    used_beads_for_earrings / earrings = 5) :=
begin
  sorry
end

end beads_per_earring_l376_376462


namespace find_a100_l376_376453

noncomputable def sequence : ℕ → ℕ
| 0     := 1
| (n+1) := sequence n + n

theorem find_a100 : sequence 100 = 4951 :=
by
  sorry

end find_a100_l376_376453


namespace proposition_C_l376_376962

theorem proposition_C (a b : ℝ) : a^3 > b^3 → a > b :=
sorry

end proposition_C_l376_376962


namespace sufficient_but_not_necessary_condition_l376_376754

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x + 1) * (x - 3) < 0 → x > -1 ∧ ((x > -1) → (x + 1) * (x - 3) < 0) = false :=
sorry

end sufficient_but_not_necessary_condition_l376_376754


namespace hyperbola_properties_l376_376051

theorem hyperbola_properties :
  (∃ x y : Real,
    (x^2 / 4 - y^2 / 2 = 1) ∧
    (∃ a b c e : Real,
      2 * a = 4 ∧
      2 * b = 2 * Real.sqrt 2 ∧
      c = Real.sqrt (a^2 + b^2) ∧
      2 * c = 2 * Real.sqrt 6 ∧
      e = c / a)) :=
by
  sorry

end hyperbola_properties_l376_376051


namespace round_to_nearest_whole_l376_376517

theorem round_to_nearest_whole (x : ℝ) (hx : x = 7643.498201) : Int.floor (x + 0.5) = 7643 := 
by
  -- To prove
  sorry

end round_to_nearest_whole_l376_376517


namespace sum_odd_implies_parity_l376_376771

theorem sum_odd_implies_parity (a b c: ℤ) (h: (a + b + c) % 2 = 1) : (a^2 + b^2 - c^2 + 2 * a * b) % 2 = 1 := 
sorry

end sum_odd_implies_parity_l376_376771


namespace median_of_36_consecutive_integers_l376_376567

theorem median_of_36_consecutive_integers (x : ℤ) (sum_eq : (∑ i in finset.range 36, (x + i)) = 6^4) : (17 + 18) / 2 = 36 :=
by
  -- Proof goes here
  sorry

end median_of_36_consecutive_integers_l376_376567


namespace find_a_l376_376027

theorem find_a (a : ℝ) : 
  (a = 0 ∨ a = 2) ↔ 
  (∃ x y : ℝ, x^2 + y^2 - 2*x - 4*y = 0 ∧ 1 / sqrt 2 = sqrt 2 / 2 ∧ abs(a - 1) = 1) :=
by 
  sorry

end find_a_l376_376027


namespace continued_fraction_euclidean_l376_376509

theorem continued_fraction_euclidean (m n : ℕ) (h : m < n) (a : ℕ → ℕ) (r : ℕ → ℕ) (s : ℕ) :
  let x := λ k, [0; a k, a (k + 1), ..., a s] in
  n = a 0 * m + r 1 ∧
  m = a 1 * r 1 + r 2 ∧
  (∀ k < s, r k = a (k + 1) * r (k + 1) + r (k + 2)) ∧
  r (s - 1) = a s * r s →
  x 0 = m / n ∧ x 1 = r 1 / m ∧ x 2 = r 2 / r 1 ∧ ... ∧ x s = r s / r (s-1) :=
sorry

end continued_fraction_euclidean_l376_376509


namespace correct_answer_l376_376417

-- Define the sentence structure and the requirement for a formal object
structure SentenceStructure where
  subject : String := "I"
  verb : String := "like"
  object_placeholder : String := "_"
  clause : String := "when the weather is clear and bright"

-- Correct choices provided
inductive Choice
  | this
  | that
  | it
  | one

-- Problem formulation: Based on SentenceStructure, prove that 'it' is the correct choice
theorem correct_answer {S : SentenceStructure} : Choice.it = Choice.it :=
by
  -- Proof omitted
  sorry

end correct_answer_l376_376417


namespace find_lambda_l376_376790

open Real

variables (m n : ℝ^3) (λ : ℝ)
 
-- All given conditions
def angle_condition := real.angle m n = 2 * π / 3
def magnitude_condition := ∥n∥ = 2 * ∥m∥
def perpendicular_condition := dot (λ • m + n) (m - 2 • n) = 0

-- The theorem statement
theorem find_lambda : angle_condition m n ∧ magnitude_condition m n ∧ perpendicular_condition m n λ → λ = 3 := by
  sorry

end find_lambda_l376_376790


namespace circle_diameter_MN_passes_through_F_l376_376751

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Define the condition for the slope of the line
def line (k x : ℝ) : ℝ := k * (x - 1)

-- Define the left vertex of the ellipse
def A : ℝ × ℝ := (-2, 0)

-- Define the focus F of the ellipse
def F : ℝ × ℝ := (1, 0)

-- Prove that the circle with diameter MN always passes through the focus F
theorem circle_diameter_MN_passes_through_F (x₁ x₂ y₁ y₂ k : ℝ) 
  (hx₁ : ellipse x₁ y₁) (hx₂ : ellipse x₂ y₂) (k_ne_zero : k ≠ 0) 
  (P Q : ℝ × ℝ) (hf₁ : P = (x₁, line k x₁)) (hf₂ : Q = (x₂, line k x₂)) 
  (hpq : (P ≠ Q)) : 
  ∃ M N : ℝ × ℝ, ∃ (diameter_MN := λ M N, ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 4), 
  circle diameter_MN F :=
sorry -- Proof omitted

end circle_diameter_MN_passes_through_F_l376_376751


namespace median_of_36_consecutive_integers_l376_376566

theorem median_of_36_consecutive_integers (x : ℤ) (sum_eq : (∑ i in finset.range 36, (x + i)) = 6^4) : (17 + 18) / 2 = 36 :=
by
  -- Proof goes here
  sorry

end median_of_36_consecutive_integers_l376_376566


namespace tetrahedron_volume_eq_l376_376216

theorem tetrahedron_volume_eq : 
  ∀ (tetrahedron: Type) 
  (faces : tetrahedron → Type)
  (side_length : ℝ), 
  side_length = 1 →
  (∀ (face : faces tetrahedron), 
    is_equilateral face side_length ∨ is_isosceles_right face side_length) →
  volume tetrahedron = √2 / 12 := 
by
  sorry

end tetrahedron_volume_eq_l376_376216


namespace probability_of_product_of_rolls_is_multiple_of_4_l376_376461

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def probability_of_product_multiple_of_4 : ℚ :=
  let juan_die := {x // 1 ≤ x ∧ x ≤ 12}
  let amal_die := {x // 1 ≤ x ∧ x ≤ 8}
  (∑ x in juan_die, ∑ y in amal_die, if is_multiple_of_4 (x * y) then 1 else 0) / (juan_die.card * amal_die.card)

theorem probability_of_product_of_rolls_is_multiple_of_4 :
  probability_of_product_multiple_of_4 = 7 / 16 :=
  sorry

end probability_of_product_of_rolls_is_multiple_of_4_l376_376461


namespace g_decreasing_on_neg1_0_l376_376001

noncomputable def f (x : ℝ) : ℝ := 8 + 2 * x - x^2 
noncomputable def g (x : ℝ) : ℝ := f (2 - x^2)

theorem g_decreasing_on_neg1_0 : 
  ∀ x y : ℝ, -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ x < y → g y < g x :=
sorry

end g_decreasing_on_neg1_0_l376_376001


namespace investment_calculation_l376_376941

noncomputable def initial_investment (final_amount : ℝ) (years : ℕ) (interest_rate : ℝ) : ℝ :=
  final_amount / ((1 + interest_rate / 100) ^ years)

theorem investment_calculation :
  initial_investment 504.32 3 12 = 359 :=
by
  sorry

end investment_calculation_l376_376941


namespace min_a_for_sequence_sum_lt_a_l376_376489

theorem min_a_for_sequence_sum_lt_a {a : ℝ} :
  (∀ (a_n : ℕ → ℝ) (S_n : ℕ → ℝ), 
     a_n 1 = 1/3 ∧ 
     (∀ (m n : ℕ), a_n (m + n) = a_n m * a_n n) →
     S_n n = (∑ i in range n, a_n i) →
     (∀ n, S_n n < a)) → 
  a ≥ 1/2 :=
by
  sorry

end min_a_for_sequence_sum_lt_a_l376_376489


namespace license_plates_count_l376_376140

theorem license_plates_count : 
  let alphabet := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'L', 'M', 'P', 'R'],
      start_letters := ['B', 'G'],
      end_letter := 'R',
      n := 5 in
  (∃ (plates : List (List Char)), ∀ plate ∈ plates, 
    plate.head ∈ start_letters ∧ plate.ilast = end_letter ∧ 
    plate.length = n ∧ 
    (∀ c ∈ plate, c ∈ alphabet ∧ plate.count c = 1) ∧ 
    plates.length = 1008) :=
by
  sorry

end license_plates_count_l376_376140


namespace graph_passes_through_fixed_point_l376_376191

theorem graph_passes_through_fixed_point (a : ℝ) (ha : a > 1) :
  ∃ p : ℝ × ℝ, p = (-2, 6) ∧ ∃ f : ℝ → ℝ, (∀ x, f x = a^(x+2) + 5) ∧ f (-2) = 6 :=
by
  exists (-2, 6)
  split
  · rfl
  · exists (λ x, a^(x+2) + 5)
    split
    · intro x
      rfl
    · rfl

end graph_passes_through_fixed_point_l376_376191


namespace bus_stop_time_per_hour_l376_376625

theorem bus_stop_time_per_hour
  (speed_no_stops : ℝ)
  (speed_with_stops : ℝ)
  (h1 : speed_no_stops = 50)
  (h2 : speed_with_stops = 35) : 
  18 = (60 * (1 - speed_with_stops / speed_no_stops)) :=
by
  sorry

end bus_stop_time_per_hour_l376_376625


namespace fill_tank_time_l376_376958

theorem fill_tank_time (flow_rate : ℕ) (tank_volume : ℕ) (h_flow_rate : flow_rate = 36) (h_tank_volume : tank_volume = 252) : tank_volume / flow_rate = 7 :=
by
  rw [h_flow_rate, h_tank_volume]
  norm_num

end fill_tank_time_l376_376958


namespace domain_of_myFunction_l376_376329

noncomputable def myFunction (x : ℝ) : ℝ := real.log ((2 - x) / (2 + x))

theorem domain_of_myFunction :
  {x : ℝ | (2 - x) / (2 + x) > 0 ∧ 2 + x ≠ 0} = {x : ℝ | -2 < x ∧ x < 2} := 
by
  sorry

end domain_of_myFunction_l376_376329


namespace mike_gave_4_marbles_l376_376146

noncomputable def marbles_given (original_marbles : ℕ) (remaining_marbles : ℕ) : ℕ :=
  original_marbles - remaining_marbles

theorem mike_gave_4_marbles (original_marbles remaining_marbles given_marbles : ℕ) 
  (h1 : original_marbles = 8) (h2 : remaining_marbles = 4) (h3 : given_marbles = marbles_given original_marbles remaining_marbles) : given_marbles = 4 :=
by
  sorry

end mike_gave_4_marbles_l376_376146


namespace shortest_routes_l376_376678

def side_length : ℕ := 10
def refuel_distance : ℕ := 30
def num_squares_per_refuel := refuel_distance / side_length

theorem shortest_routes (A B : Type) (distance_AB : ℕ) (shortest_paths : Π (A B : Type), ℕ) : 
  shortest_paths A B = 54 := by
  sorry

end shortest_routes_l376_376678


namespace sum_of_squares_geq_one_div_n_l376_376369

theorem sum_of_squares_geq_one_div_n (a : ℕ → ℝ) (n : ℕ) (h₁ : ∑ (i : ℕ) in finset.range n, a i = 1) :
  ∑ (i : ℕ) in finset.range n, (a i)^2 ≥ 1 / n :=
sorry

end sum_of_squares_geq_one_div_n_l376_376369


namespace find_bullet_l376_376955

theorem find_bullet (x y : ℝ) (h₁ : 3 * x + y = 8) (h₂ : y = -1) : 2 * x - y = 7 :=
sorry

end find_bullet_l376_376955


namespace max_sum_ak_one_over_k_squared_l376_376005

theorem max_sum_ak_one_over_k_squared (a : Fin 2020 → ℝ)
  (h_pos : ∀ i, 0 < a i)
  (h_sum : ∑ i, a i = 2020) :
  ∑ k, (a k)^(1 / (k + 1)^2) ≤ 2021 := 
sorry

end max_sum_ak_one_over_k_squared_l376_376005


namespace sum_indices_of_reordered_elements_l376_376854

def c_sequence : ℕ → ℝ
| 1 := 0.505
| k := 
    if odd k then (0.50501 : ℝ)^(c_sequence (k - 1))
    else (0.505011 : ℝ)^(c_sequence (k - 1))

def d_sequence : ℕ → ℝ := sorry -- Define d_sequence as the sorted sequence

noncomputable def is_reordered_element (n : ℕ) : Prop := c_sequence n = d_sequence n

noncomputable def sum_reordered_indices : ℕ :=
  ∑ i in (finset.range 1005).filter is_reordered_element, (i + 1)

theorem sum_indices_of_reordered_elements :
  sum_reordered_indices = 504612 :=
begin
  sorry -- Proof goes here
end

end sum_indices_of_reordered_elements_l376_376854


namespace system_equivalence_l376_376887

theorem system_equivalence (f g : ℝ → ℝ) (x : ℝ) (h1 : f x > 0) (h2 : g x > 0) : f x + g x > 0 :=
sorry

end system_equivalence_l376_376887


namespace bag_cost_is_2_l376_376053

-- Define the inputs and conditions
def carrots_per_day := 1
def days_per_year := 365
def carrots_per_bag := 5
def yearly_spending := 146

-- The final goal is to find the cost per bag
def cost_per_bag := yearly_spending / ((carrots_per_day * days_per_year) / carrots_per_bag)

-- Prove that the cost per bag is $2
theorem bag_cost_is_2 : cost_per_bag = 2 := by
  -- Using sorry to complete the proof
  sorry

end bag_cost_is_2_l376_376053


namespace range_of_a_l376_376428

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 - a * x + a else (4 - 2 * a) ^ x

def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ f x ≥ f y

theorem range_of_a (a : ℝ) :
  is_monotonic (f a) → (3 / 2 < a ∧ a < 2) :=
sorry

end range_of_a_l376_376428


namespace sequence_properties_l376_376375

-- Define the sequence with the given conditions
def seq : ℕ → ℕ
| 0       := 1
| 1       := 2
| (n + 2) := if seq n * seq (n + 1) ≠ 1 then seq n + seq (n + 1) else 3  -- Placeholder for the rule

-- Sum of first n terms in the sequence
def sum_seq : ℕ → ℕ
| 0       := seq 0
| (n + 1) := sum_seq n + seq (n + 1)

-- Define the main theorem
theorem sequence_properties :
  (seq 0 + seq 1 + seq 2 = 6) ∧
  (sum_seq 2009 = 4020) :=
by
  -- Placeholder to satisfy Lean as the proof isn't required
  sorry

end sequence_properties_l376_376375


namespace max_volume_of_box_l376_376656

def volume (x : ℝ) : ℝ :=
  4 * x^3 - 56 * x^2 + 192 * x

theorem max_volume_of_box :
  ∃ x : ℝ, 0 < x ∧ 2 * x < 12 ∧ 2 * x < 16 ∧ ∀ y : ℝ, 0 < y ∧ 2 * y < 12 ∧ 2 * y < 16 → volume y ≤ volume x ∧ volume x = 128 :=
begin
  sorry
end

end max_volume_of_box_l376_376656


namespace problem_statement_l376_376327

noncomputable def integerValuesSatisfyingInequality : Nat := 
  ∑ d in Finset.range 10, ite (3 + d * 0.0001 + 0.00003 < 3.0007) 1 0

theorem problem_statement :
  integerValuesSatisfyingInequality = 7 :=
sorry

end problem_statement_l376_376327


namespace bunny_burrows_l376_376635

theorem bunny_burrows (x : ℕ) (h1 : 20 * x * 600 = 36000) : x = 3 :=
by
  -- Skipping proof using sorry
  sorry

end bunny_burrows_l376_376635


namespace max_value_x2_plus_y2_l376_376481

theorem max_value_x2_plus_y2 (x y : ℝ) (h : 5 * x^2 + 4 * y^2 = 10 * x) : 
  x^2 + y^2 ≤ 4 :=
sorry

end max_value_x2_plus_y2_l376_376481


namespace complex_div_conj_l376_376767

theorem complex_div_conj (z : ℂ) (h : z = 1 + 2 * complex.I) : 
  z / (complex.conj z) = - (3 / 5 : ℂ) + (4 / 5 : ℂ) * complex.I :=
by 
  sorry

end complex_div_conj_l376_376767


namespace gather_half_nuts_l376_376673

theorem gather_half_nuts (a b c : ℕ) (h : even (a + b + c)) :
  ∃ i ∈ {a, b, c}, i = (a + b + c) / 2 :=
sorry

end gather_half_nuts_l376_376673


namespace pig_ratio_l376_376275

theorem pig_ratio (avg_bacon_per_pig : ℝ) (price_per_pound : ℝ) (total_revenue : ℝ) (total_revenue = 60) (price_per_pound = 6) (avg_bacon_per_pig = 20) :
  (total_revenue / price_per_pound) / avg_bacon_per_pig = 1/2 :=
by 
  sorry

end pig_ratio_l376_376275


namespace number_of_terms_in_arithmetic_sequence_l376_376414

-- Define the necessary conditions
def a := 2
def d := 5
def l := 1007  -- last term

-- Prove the number of terms in the sequence
theorem number_of_terms_in_arithmetic_sequence : 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 202 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l376_376414


namespace max_value_f_l376_376917

noncomputable def f : ℝ → ℝ := λ x, Real.exp x - 2 * x

theorem max_value_f : ∃ M, (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≤ M) ∧ M = f (Real.exp 1) :=
by
  sorry

end max_value_f_l376_376917


namespace strictly_decreasing_interval_l376_376347

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

theorem strictly_decreasing_interval : ∀ x : ℝ, -1 < x ∧ x < 3 → f' x < 0 := 
by
  sorry

end strictly_decreasing_interval_l376_376347


namespace distribute_stickers_l376_376599

theorem distribute_stickers (n m : ℕ) (h_n : n = 9) (h_m : m = 3) : 
  ∃ k, k = 12 ∧ (num_partitions_with_zeros n m = k) :=
by
  sorry

-- Auxiliary function to calculate number of partitions allowing zeros
def num_partitions_with_zeros (n m : ℕ) : ℕ :=
  sorry

end distribute_stickers_l376_376599


namespace average_of_values_l376_376345

theorem average_of_values (z : ℝ) : 
  (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end average_of_values_l376_376345


namespace initial_investment_l376_376674

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / (n : ℝ)) ^ (n : ℝ * t)

theorem initial_investment :
  (∃ P, compound_interest P 0.10 2 1 = 771.75) :=
begin
  use 700,
  sorry,
end

end initial_investment_l376_376674


namespace collinear_INP_l376_376831

-- Let \(ABC\) be a triangle and define the points as given in the conditions.
variables {A B C I N P M : Point}
variables {A1 C1 : Point}

-- Assume the given conditions in the problem.
-- Define A1 and C1 as the points where the incircle touches sides BC and BA, respectively.
def touches_incircle (I : Point) (BC BA : Line) (A1 C1 : Point) :=
  touches I BC A1 ∧ touches I BA C1

-- Define M as the midpoint of side AC.
def midpoint (AC : Line) (M : Point) :=
  is_midpoint M AC

-- Define P as the foot of the perpendicular from M to line A1C1.
def foot_perpendicular (M : Point) (A1C1 : Line) (P : Point) :=
  is_perpendicular_foot M A1C1 P

-- N is defined as the midpoint of the arc ABC of the circumcircle of triangle ABC.
def arc_midpoint (circ_ABC : Arc) (N : Point) :=
  is_arc_midpoint N circ_ABC

-- State the theorem that needs to be proved.
theorem collinear_INP (A B C I N P M A1 C1 : Point)
  (h_incircle : touches_incircle I (Line B C) (Line B A) A1 C1)
  (h_midpoint_M : midpoint (Line A C) M)
  (h_foot_P : foot_perpendicular M (Line A1 C1) P)
  (h_arc_midpoint_N : arc_midpoint (circumcircle A B C) N) :
  collinear I N P :=
begin
  sorry
end

end collinear_INP_l376_376831


namespace sum_of_powers_modulo_l376_376131

theorem sum_of_powers_modulo (R : Finset ℕ) (S : ℕ) :
  (∀ n < 100, ∃ r, r ∈ R ∧ r = 3^n % 500) →
  S = R.sum id →
  (S % 500) = 0 :=
by {
  -- Proof would go here
  sorry
}

end sum_of_powers_modulo_l376_376131


namespace size_of_johns_donation_l376_376530

theorem size_of_johns_donation (n : ℕ) (new_avg : ℝ) (increase_pct : ℝ) (orig_avg : ℝ) (total_before : ℝ) (total_after : ℝ) (john_donation : ℝ):
  n = 10 → new_avg = 90 → increase_pct = 0.80 → 
  orig_avg = new_avg / (1 + increase_pct) → 
  total_before = n * orig_avg → total_after = total_before + john_donation -> 
  total_after = (n + 1) * new_avg → 
  john_donation = 490 :=
begin
  intros h1 h2 h3 h4 h5 h6,
  sorry
end

end size_of_johns_donation_l376_376530


namespace range_of_m_l376_376473

def f (x : ℝ) : ℝ := x - (1 / x)

theorem range_of_m :
  {m : ℝ | ∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.sqrt 5 → f (m * x) + m * f x < 0} ⊆ 
  {m | (m < -1) ∨ (0 < m ∧ m < 1 / 3)} :=
by
  sorry

end range_of_m_l376_376473


namespace perpendicular_lines_l376_376099

noncomputable def acute_triangle (M K N : Type) [inner_product_space ℝ E] : Prop :=
  is_acute_triangle M K N

noncomputable def bisector (M K N L : Type) [inner_product_space ℝ F] : Prop :=
  is_angle_bisector K L M N

noncomputable def circumcenter (M K N O : Type) [inner_product_space ℝ F] : Prop :=
  is_circumcenter M K N O

noncomputable def isosceles (K X N : Type) [metric_space K] : Prop :=
  dist K X = dist K N

theorem perpendicular_lines {E F : Type}
  [inner_product_space ℝ E] [inner_product_space ℝ F] 
  (M K N L X O : E)
  (h1 : acute_triangle M K N)
  (h2 : bisector M K N L)
  (h3 : isosceles K X N)
  (h4 : circumcenter M K N O) :
  is_perpendicular (line_through K O) (line_through X L) :=
sorry

end perpendicular_lines_l376_376099


namespace sin_alpha_value_l376_376773

-- Definitions based on given conditions
def P : ℝ × ℝ := (-2, 1)
def α : ℝ := sorry -- α as required angle defined 

-- Theorem to prove that sin α is as expected
theorem sin_alpha_value : 
  let (x, y) := P 
  let r := real.sqrt (x*x + y*y)
  α = real.arctan2 y x in
  real.sin α = real.sqrt 5 / 5 :=
by 
  let (x, y) := P
  let r := real.sqrt (x*x + y*y)
  have h : r = real.sqrt 5 := sorry
  have α_val : α = real.arctan2 y x := sorry
  calc
    real.sin α 
      = y / r       : by rw [real.sin_eq_y_div_r α_val]
      ... = 1 / real.sqrt 5 : by sorry
      ... = real.sqrt 5 / 5 : by sorry

end sin_alpha_value_l376_376773


namespace area_parallelogram_ABCD_l376_376826

variables (ABCD : parallelogram)
variables (E F H G : Point)
variable (area_EHGF : ℝ)
variable [midpoint E (side CD ABCD)]
variable [intersection F (line AE) (line BD)]
variable [intersection H (line AC) (line BE)]
variable [intersection G (line AC) (line BD)]

theorem area_parallelogram_ABCD :
  area_EHGF = 15 →
  area ABCD = 90 :=
by
  sorry

end area_parallelogram_ABCD_l376_376826


namespace unique_line_through_two_points_l376_376939

noncomputable theory

open_locale classical

def point : Type := ℝ × ℝ

def distinct (p1 p2 : point) : Prop := p1 ≠ p2

def line_through (p1 p2 : point) : Type := { l : set point // ∀ p : point, p ∈ l ↔ ∃ k : ℝ, p = (1 - k) • p1 + k • p2 }

theorem unique_line_through_two_points (p1 p2 : point) (hp : distinct p1 p2) :
  ∃! l : line_through p1 p2, true :=
sorry

end unique_line_through_two_points_l376_376939


namespace evaluate_expr_l376_376708

theorem evaluate_expr : Int.ceil (5 / 4 : ℚ) + Int.floor (-5 / 4 : ℚ) = 0 := by
  sorry

end evaluate_expr_l376_376708


namespace pure_imaginary_product_imaginary_part_fraction_l376_376380

-- Part 1
theorem pure_imaginary_product (m : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : z1 = m + i) (h3 : z2 = 2 + m * i) :
  (z1 * z2).re = 0 ↔ m = 0 := 
sorry

-- Part 2
theorem imaginary_part_fraction (m : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : z1 = m + i) (h3 : z2 = 2 + m * i)
  (h4 : z1^2 - 2 * z1 + 2 = 0) :
  (z2 / z1).im = -1 / 2 :=
sorry

end pure_imaginary_product_imaginary_part_fraction_l376_376380


namespace total_books_l376_376225

-- Define the number of books Victor originally had and the number he bought
def original_books : ℕ := 9
def bought_books : ℕ := 3

-- The proof problem statement: Prove Victor has a total of original_books + bought_books books
theorem total_books : original_books + bought_books = 12 := by
  -- proof will go here, using sorry to indicate it's omitted
  sorry

end total_books_l376_376225


namespace equilateral_triangle_perimeter_l376_376922

theorem equilateral_triangle_perimeter (s : ℕ) (h1 : 2 * s + 10 = 50) : 3 * s = 60 :=
sorry

end equilateral_triangle_perimeter_l376_376922


namespace print_shop_cost_difference_l376_376162

theorem print_shop_cost_difference :
let
  x_cost_per_copy := 1.25,
  y_cost_per_copy := 2.75,
  x_bulk_discount := 0.10,
  y_bulk_discount := 0.05,
  x_tax_rate := 0.07,
  y_tax_rate := 0.09,
  num_copies := 40
in
  let
    x_initial_cost := num_copies * x_cost_per_copy,
    y_initial_cost := num_copies * y_cost_per_copy,
    x_discount := x_initial_cost * x_bulk_discount,
    y_discount := y_initial_cost * y_bulk_discount,
    x_cost_after_discount := x_initial_cost - x_discount,
    y_cost_after_discount := y_initial_cost - y_discount,
    x_tax := x_cost_after_discount * x_tax_rate,
    y_tax := y_cost_after_discount * y_tax_rate,
    x_total_cost := x_cost_after_discount + x_tax,
    y_total_cost := y_cost_after_discount + y_tax,
    cost_difference := y_total_cost - x_total_cost
  in
    cost_difference = 65.755 :=
by sorry

end print_shop_cost_difference_l376_376162


namespace non_negative_sums_possible_l376_376692

open Matrix

theorem non_negative_sums_possible {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
  ∃ B : Matrix (Fin m) (Fin n) ℝ, 
    (∀ i : Fin m, 0 ≤ ∑ j : Fin n, B i j) ∧ 
    (∀ j : Fin n, 0 ≤ ∑ i : Fin m, B i j) :=
  sorry

end non_negative_sums_possible_l376_376692


namespace max_value_a_l376_376072

noncomputable def minimum_value (f : ℝ → ℝ) : ℝ :=
  Inf (Set.range f)

theorem max_value_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2 * x - 6 ≥ a) ↔ a ≤ minimum_value (λ x, x^2 + 2 * x - 6) := 
by
  sorry

end max_value_a_l376_376072


namespace compare_abc_l376_376367

noncomputable def a : ℝ := 2^0.5
noncomputable def b : ℝ := Real.log 3 / Real.log π
noncomputable def c : ℝ := Real.log ((Real.sqrt 2) / 2) / Real.log 2

theorem compare_abc : a > b ∧ b > c := by
sorry

end compare_abc_l376_376367


namespace range_of_a_l376_376787

variable (a : ℝ)

def A : set ℝ := {x | (x - 3) / (x - 4) < 0}
def B (a : ℝ) : set ℝ := {x | (x - a) * (x - 5) > 0}

theorem range_of_a : (A ⊆ B a) ↔ 4 ≤ a ∧ a < 5 := by
  sorry

end range_of_a_l376_376787


namespace graph_has_inverse_l376_376794

noncomputable def F (x : ℝ) : ℝ := if x < -1 then x + 4
                                   else if x < 1 then -4 * x
                                   else x - 4

def G (x : ℝ) : ℝ := 2 * x + 1

noncomputable def H (x : ℝ) : ℝ := if x < -1 then -1
                                   else if x < 3 then 3
                                   else -2

noncomputable def I (x : ℝ) : ℝ × ℝ := (4 * real.cos x, 4 * real.sin x)

noncomputable def J (x : ℝ) : ℝ := x ^ 3 / 20 + x ^ 2 / 10 - x + 1

theorem graph_has_inverse (g : ℝ → ℝ) (inv_g : ℝ → ℝ) : 
  (∀ x : ℝ, g (inv_g x) = x) ∧
  (∀ y : ℝ, inv_g (g y) = y) → 
  ({G} : set (ℝ → ℝ)) = {G} := 
by {
  sorry
}

end graph_has_inverse_l376_376794


namespace number_of_blue_tiles_is_16_l376_376916

def length_of_floor : ℕ := 20
def breadth_of_floor : ℕ := 10
def tile_length : ℕ := 2

def total_tiles : ℕ := (length_of_floor / tile_length) * (breadth_of_floor / tile_length)

def black_tiles : ℕ :=
  let rows_length := 2 * (length_of_floor / tile_length)
  let rows_breadth := 2 * (breadth_of_floor / tile_length)
  (rows_length + rows_breadth) - 4

def remaining_tiles : ℕ := total_tiles - black_tiles
def white_tiles : ℕ := remaining_tiles / 3
def blue_tiles : ℕ := remaining_tiles - white_tiles

theorem number_of_blue_tiles_is_16 :
  blue_tiles = 16 :=
by
  sorry

end number_of_blue_tiles_is_16_l376_376916


namespace remainder_of_S_mod_500_eq_zero_l376_376125

open Function

def R : Set ℕ := { r | ∃ n : ℕ, r = (3^n % 500) }

def S : ℕ := ∑ r in R.toFinset, r

theorem remainder_of_S_mod_500_eq_zero :
  (S % 500) = 0 := by
  sorry

end remainder_of_S_mod_500_eq_zero_l376_376125


namespace max_fraction_inequality_l376_376770

theorem max_fraction_inequality (a c : ℝ) (h1 : ∀ x : ℝ, a * x^2 + 2 * x + c ≤ 0 ↔ x = -1 / a)
  (h2 : a > c) : ∃ M, M = (a - c) / (a ^ 2 + c ^ 2) ∧ M = sqrt 2 / 4 :=
by
  sorry

end max_fraction_inequality_l376_376770


namespace min_value_polynomial_l376_376813

theorem min_value_polynomial (a b : ℝ) : 
  ∃ c, (∀ a b, c ≤ a^2 + 2 * b^2 + 2 * a + 4 * b + 2008) ∧
       (∀ a b, a = -1 ∧ b = -1 → c = a^2 + 2 * b^2 + 2 * a + 4 * b + 2008) :=
sorry

end min_value_polynomial_l376_376813


namespace fraction_of_orange_juice_in_mixture_l376_376219

theorem fraction_of_orange_juice_in_mixture
  (capacity_pitcher : ℕ)
  (fraction_first_pitcher : ℚ)
  (fraction_second_pitcher : ℚ)
  (condition1 : capacity_pitcher = 500)
  (condition2 : fraction_first_pitcher = 1/4)
  (condition3 : fraction_second_pitcher = 3/7) :
  (125 + 500 * (3/7)) / (2 * 500) = 95 / 280 :=
by
  sorry

end fraction_of_orange_juice_in_mixture_l376_376219


namespace cyclic_hexagon_incircle_circumcircle_relation_l376_376835

theorem cyclic_hexagon_incircle_circumcircle_relation
  (AB CD AD R r c : ℝ)
  (H_symmetric : SymmetricHexagonAboutDiagonal ABCDEF AD)
  (inscribable : CanInscribableCircleInHexagon ABCDEF)
  (AB_CD_AD_eq : AB + CD = AD)
  (circum_radius : CircumcircleRadius ABCDEF = R)
  (incircle_radius : IncircleRadius ABCDEF = r)
  (centers_distance : DistanceBetweenCentersOfCircles = c) :
  3 * (R^2 - c^2)^4 - 4 * (R^2 - c^2)^2 * (R^2 + c^2) * r^2 - 16 * R^2 * c^2 * r^4 = 0 :=
by sorry

end cyclic_hexagon_incircle_circumcircle_relation_l376_376835


namespace sum_of_powers_modulo_l376_376129

theorem sum_of_powers_modulo (R : Finset ℕ) (S : ℕ) :
  (∀ n < 100, ∃ r, r ∈ R ∧ r = 3^n % 500) →
  S = R.sum id →
  (S % 500) = 0 :=
by {
  -- Proof would go here
  sorry
}

end sum_of_powers_modulo_l376_376129


namespace max_intersection_points_of_perpendiculars_l376_376014

theorem max_intersection_points_of_perpendiculars (P1 P2 P3 P4 : Point) :
  (∀ (L1 L2 : Line), connects_two_points L1 P1 P2 → connects_two_points L2 P3 P4 → 
    ¬ (coincident L1 L2 ∨ parallel L1 L2 ∨ perpendicular L1 L2)) →
  max_intersection_points_perpendiculars (P1, P2, P3, P4) = 44 := 
sorry

end max_intersection_points_of_perpendiculars_l376_376014


namespace ratio_of_buses_to_cars_l376_376926

theorem ratio_of_buses_to_cars (cars buses : ℕ) (h1 : cars = 100) (h2 : buses = cars - 90) : buses / gcd buses cars = 1 ∧ cars / gcd buses cars = 10 :=
by
  have h3 : buses = 10 := by rw [h1, h2]; refl
  have h4 : gcd buses cars = 10 := by rw [h3, h1]; exact Nat.gcd_refl 10
  rw [h4]
  exact ⟨Nat.div_self (Nat.pos_of_ne_zero (ne_of_eq_of_ne (h3.symm ▸ h2 ▸ h1.symm ▸ rfl) (by decide))),
         show cars / 10 = 10 by rw [Nat.div_eq_iff_eq_mul_right (dec_trivial: 0 < 10), nat.mul_comm]; rw [h1]; rfl⟩

end ratio_of_buses_to_cars_l376_376926


namespace total_points_scored_l376_376441

theorem total_points_scored
    (Bailey_points Chandra_points Akiko_points Michiko_points : ℕ)
    (h1 : Bailey_points = 14)
    (h2 : Michiko_points = Bailey_points / 2)
    (h3 : Akiko_points = Michiko_points + 4)
    (h4 : Chandra_points = 2 * Akiko_points) :
  Bailey_points + Michiko_points + Akiko_points + Chandra_points = 54 := by
  sorry

end total_points_scored_l376_376441


namespace θ_values_l376_376361

-- Define the given conditions
def terminal_side_coincides (θ : ℝ) : Prop :=
  ∃ k : ℤ, 7 * θ = θ + 360 * k

def θ_in_range (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 360

-- The main theorem
theorem θ_values (θ : ℝ) (h_terminal : terminal_side_coincides θ) (h_range : θ_in_range θ) :
  θ = 0 ∨ θ = 60 ∨ θ = 120 ∨ θ = 180 ∨ θ = 240 ∨ θ = 300 :=
sorry

end θ_values_l376_376361


namespace max_value_of_f_l376_376608

noncomputable def f (x : ℝ) : ℝ := real.sin (π / 6 - x) * real.sin x

theorem max_value_of_f : ∃ x : ℝ, f x = 1 / 2 - real.sqrt 3 / 4 := sorry

end max_value_of_f_l376_376608


namespace correct_statements_l376_376458

noncomputable def triangleA := 
  let b := 19
  let A := 45
  let C := 30
  (b, A, C, ∃! (a c : ℝ), a = 19 * sin 45 / sin 105 ∧ c = 19 * sin 30 / sin 105)

noncomputable def triangleB := 
  let a := sqrt 3
  let b := 2 * sqrt 2
  let A := 45
  (a, b, A, ∀ B : ℝ, sin B ≠ 2 / sqrt 3)

noncomputable def triangleC := 
  let a := 3
  let b := 2 * sqrt 2
  let A := 45
  (a, b, A, ∃! (B : ℝ), sin B = 2 / 3)

noncomputable def triangleD := 
  let a := 7
  let b := 7
  let A := 75
  let B := 75
  (a, b, A, B, ∃! (C : ℝ), C = 30)

theorem correct_statements : (triangleA = false) ∧ (triangleB = false) ∧ (triangleC = true) ∧ (triangleD = true) :=
  by sorry

end correct_statements_l376_376458


namespace shifted_parabola_expression_l376_376531

theorem shifted_parabola_expression (x y x' y' : ℝ) 
  (h_initial : y = (x + 2)^2 + 3)
  (h_shift_right : x' = x - 3)
  (h_shift_down : y' = y - 2)
  : y' = (x' - 1)^2 + 1 := 
sorry

end shifted_parabola_expression_l376_376531


namespace cows_in_group_l376_376817

-- Definitions for the problem conditions
def number_of_cows := C
def number_of_hens := H
def legs_of_cows := 4 * C
def legs_of_hens := 2 * H
def total_legs := legs_of_cows + legs_of_hens
def total_heads := C + H
def condition := total_legs = 2 * total_heads + 8

-- Lean statement for the proof problem
theorem cows_in_group (C H : ℕ) (h : condition C H) : C = 4 :=
by sorry

end cows_in_group_l376_376817


namespace fraction_addition_l376_376419

theorem fraction_addition (x y : ℚ) (h : x / y = 2 / 3) : (x + y) / y = 5 / 3 := 
by 
  sorry

end fraction_addition_l376_376419


namespace median_of_36_consecutive_integers_l376_376573

theorem median_of_36_consecutive_integers (sum_of_integers : ℕ) (num_of_integers : ℕ) 
  (h1 : num_of_integers = 36) (h2 : sum_of_integers = 6 ^ 4) : 
  (sum_of_integers / num_of_integers) = 36 := 
by 
  sorry

end median_of_36_consecutive_integers_l376_376573


namespace f_at_11_l376_376800

def f (n : ℕ) : ℕ := n^2 + n + 17

theorem f_at_11 : f 11 = 149 := sorry

end f_at_11_l376_376800


namespace sin_405_eq_sqrt_2_div_2_l376_376317

theorem sin_405_eq_sqrt_2_div_2 : sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_eq_sqrt_2_div_2_l376_376317


namespace sufficient_questions_l376_376301

theorem sufficient_questions (n : ℕ) (h : n > 0) :
  ∃ q : ℕ, q ≤ 10 * n ∧ (∀ (scientists : fin n → bool) (truthfulness : fin n → Prop),
    (∀ i, ∃ j, truthfulness j) →
    (∃ a : list (fin n × fin n × bool), a.length ≤ q ∧
      ∀ i, ∃ truth_value,
        list.take i a
        |>.map (λ ⟨asker, asked, res⟩, if asker = i && res = truthfulness asked
                                      else if asked = i then res = truthfulness asker
                                      else true)
        = filter (λ res, res = truth_value) a)) :=
by sorry

end sufficient_questions_l376_376301


namespace fraction_subtraction_inequality_l376_376888

theorem fraction_subtraction_inequality (a b n : ℕ) (h1 : a < b) (h2 : 0 < n) (h3 : n < a) : 
  (a : ℚ) / b > (a - n : ℚ) / (b - n) :=
sorry

end fraction_subtraction_inequality_l376_376888


namespace function_classification_l376_376326

theorem function_classification (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) := by
  sorry

end function_classification_l376_376326


namespace find_x_solution_l376_376774

theorem find_x_solution (x b c : ℝ) (h_eq : x^2 + c^2 = (b - x)^2):
  x = (b^2 - c^2) / (2 * b) :=
sorry

end find_x_solution_l376_376774


namespace cos_B_of_arithmetic_angles_sin_A_sin_C_of_geometric_sides_l376_376816

theorem cos_B_of_arithmetic_angles (A B C : ℝ) (h_arith_seq : 2 * B = A + C) (h_sum_180 : A + B + C = 180) : 
  Real.cos B = 1 / 2 := 
  sorry

theorem sin_A_sin_C_of_geometric_sides (A B C a b c : ℝ) 
  (h_arith_seq : 2 * B = A + C) (h_sum_180 : A + B + C = 180)
  (h_cos_B : Real.cos B = 1 / 2) (h_geom_seq : b^2 = a * c) : 
  Real.sin A * Real.sin C = 3 / 4 := 
  sorry

end cos_B_of_arithmetic_angles_sin_A_sin_C_of_geometric_sides_l376_376816


namespace geom_seq_general_term_sum_geometric_arithmetic_l376_376019

noncomputable def a_n (n : ℕ) : ℕ := 2^n
def b_n (n : ℕ) : ℕ := 2*n - 1

theorem geom_seq_general_term (a : ℕ → ℕ) (a1 : a 1 = 2)
  (a2 : a 3 = (a 2) + 4) : ∀ n, a n = a_n n :=
by
  sorry

theorem sum_geometric_arithmetic (a b : ℕ → ℕ) 
  (a_def : ∀ n, a n = 2 ^ n) (b_def : ∀ n, b n = 2 * n - 1) : 
  ∀ n, (Finset.range n).sum (λ i => (a (i + 1) + b (i + 1))) = 2^(n+1) + n^2 - 2 :=
by
  sorry

end geom_seq_general_term_sum_geometric_arithmetic_l376_376019


namespace parabola_focus_distance_area_l376_376553

theorem parabola_focus_distance_area (p : ℝ) (hp : p > 0)
  (A : ℝ × ℝ) (hA : A.2^2 = 2 * p * A.1)
  (hDist : A.1 + p / 2 = 2 * A.1)
  (hArea : 1/2 * (p / 2) * |A.2| = 1) :
  p = 2 :=
sorry

end parabola_focus_distance_area_l376_376553


namespace radius_of_incircle_of_triangle_l376_376542

/-- Coordinates of foci of the hyperbola --/
def F1 : (ℝ × ℝ) := (-2, 0)
def F2 : (ℝ × ℝ) := (2, 0)

/-- Equation of the hyperbola --/
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 - y^2 / 3 = 1

/-- Equation of the circle with diameter F_1F_2 --/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

/-- Conditions for the points P and Q that lie on both the hyperbola and the circle --/
def point_conditions : ∃ (x y : ℝ), hyperbola_eq x y ∧ circle_eq x y :=
  sorry

/-- Proof that the radius of the incircle of ∆F1PQ is √7 - 1 --/
theorem radius_of_incircle_of_triangle :
  ∃ (r : ℝ), r = (√7 - 1) :=
  sorry

end radius_of_incircle_of_triangle_l376_376542


namespace cut_through_centers_divides_squares_in_half_l376_376054

-- Define the problem conditions
def square (p : Type*) := 
  {c : p // true } -- Using Subtype to encapsulate the center of the square

-- Defining the context of the problem
variables {p : Type*} [metric_space p] [linear_ordered_ring ℝ]

-- You have two square pancakes with centers c1 and c2
variables (c1 c2 : p)

-- Main theorem statement
theorem cut_through_centers_divides_squares_in_half :
  ∃ l : p × p, (l.1 = c1 ∧ l.2 = c2) :=
sorry

end cut_through_centers_divides_squares_in_half_l376_376054


namespace transformed_parabola_correct_l376_376533

def f (x : ℝ) : ℝ := (x + 2)^2 + 3
def g (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem transformed_parabola_correct :
  ∀ x : ℝ, g x = f (x - 3) - 2 := by
  sorry

end transformed_parabola_correct_l376_376533


namespace sequence_divisibility_l376_376847

theorem sequence_divisibility :
  ∃ (n : ℕ), ∀ i, (2006 ≤ i ∧ i < 4011) →
  ((fun x : ℕ =>
     if x < 2006 then x + 1
     else (list.func.get (λ j, if j < 2005 then (n + j) else (n + j - 2005))
       (list.func.get (λ j, nat.succ (list.func.get (λ k, if k < 2005 then (n + k) else (n + k - 2005)) n))⁻¹)) (i - 2006)) % 2006) = 0 :=
sorry

end sequence_divisibility_l376_376847


namespace problem_R1_1_l376_376265

theorem problem_R1_1 : 291 + 503 - 91 + 492 - 103 - 392 = 700 := by
  calc
    291 + 503 - 91 + 492 - 103 - 392 = (291 + 503 + 492) - (91 + 103 + 392) := by
                                                                              sorry
    ... = 1286 - 586 := by sorry
    ... = 700 := by sorry

end problem_R1_1_l376_376265


namespace ellipse_standard_equation_l376_376025

theorem ellipse_standard_equation
  (a b : ℝ) (P : ℝ × ℝ) (h_center : P = (3, 0))
  (h_a_eq_3b : a = 3 * b) 
  (h1 : a = 3) 
  (h2 : b = 1) : 
  (∀ (x y : ℝ), (x = 3 → y = 0) → (x = 0 → y = 3)) → 
  ((x^2 / a^2) + y^2 = 1 ∨ (x^2 / b^2) + (y^2 / a^2) = 1) := 
by sorry

end ellipse_standard_equation_l376_376025


namespace totalAlternatingSum_eq_l376_376049

-- Define the alternating sum function for a set
def alternatingSum (A : List ℕ) : ℕ :=
  A.reverse.enum.foldl 
    (λ acc ⟨idx, val⟩ => if idx % 2 = 0 then acc + val else acc - val) 0

-- General sum of the alternating sums of all non-empty subsets of {1, 2, ..., n}
def totalAlternatingSum (n : ℕ) : ℕ :=
  (1 + n).powerset.filter (λ s => s ≠ []).foldl (λ acc s => acc + alternatingSum s) 0

-- Prove the equivalence
theorem totalAlternatingSum_eq (n : ℕ) : totalAlternatingSum n = n * 2 ^ (n - 1) := by
  sorry

end totalAlternatingSum_eq_l376_376049


namespace geometric_sequence_a3_l376_376445

theorem geometric_sequence_a3 (
  a : ℕ → ℝ
) 
(h1 : a 1 = 1)
(h5 : a 5 = 16)
(h_geometric : ∀ (n : ℕ), a (n + 1) / a n = a 2 / a 1) :
a 3 = 4 := by
  sorry

end geometric_sequence_a3_l376_376445


namespace x_squared_minus_y_squared_equiv_l376_376483

theorem x_squared_minus_y_squared_equiv :
  let x := 2023^1012 - 2023^(-1012)
  let y := 2023^1012 + 2023^(-1012)
  in x^2 - y^2 = -4 :=
by
  sorry

end x_squared_minus_y_squared_equiv_l376_376483


namespace AK_perpendicular_BC_l376_376465

theorem AK_perpendicular_BC
  (A B C E F D K : Type*)
  (h1 : acute_triangle A B C)
  (h2 : reflect_point E B AC)
  (h3 : reflect_point F C AB)
  (h4 : intersection_point D BF CE)
  (h5 : circumcenter K DEF) :
  perpendicular AK BC :=
sorry

end AK_perpendicular_BC_l376_376465


namespace inscribed_rectangle_area_l376_376909

variable (a b h x : ℝ)
variable (h_pos : 0 < h) (a_b_pos : a > b) (b_pos : b > 0) (a_pos : a > 0) (x_pos : 0 < x) (hx : x < h)

theorem inscribed_rectangle_area (hb : b > 0) (ha : a > 0) (hx : 0 < x) (hxa : x < h) : 
  x * (a - b) * (h - x) / h = x * (a - b) * (h - x) / h := by
  sorry

end inscribed_rectangle_area_l376_376909


namespace number_of_valid_N_l376_376305

def is_valid_N (N : ℕ) : Prop :=
  let N4 := 
    4 ^ 3 * (N / 1000 % 4) + 
    4 ^ 2 * (N / 100 % 4) + 
    4 * (N / 10 % 4) + 
    (N % 10)
  let N7 := 
    7 ^ 3 * (N / 1000 % 7) + 
    7 ^ 2 * (N / 100 % 7) + 
    7 * (N / 10 % 7) + 
    (N % 7)
  (N4 + N7) % 1000 = (2 * N) % 1000

theorem number_of_valid_N : 
  (finset.range 10000).filter is_valid_N .card = 80 := sorry

end number_of_valid_N_l376_376305


namespace distance_covered_l376_376650

noncomputable def boat_speed_still_water : ℝ := 6.5
noncomputable def current_speed : ℝ := 2.5
noncomputable def time_taken : ℝ := 35.99712023038157

noncomputable def effective_speed_downstream (boat_speed_still_water current_speed : ℝ) : ℝ :=
  boat_speed_still_water + current_speed

noncomputable def convert_kmph_to_mps (speed_in_kmph : ℝ) : ℝ :=
  speed_in_kmph * (1000 / 3600)

noncomputable def calculate_distance (speed_in_mps time_in_seconds : ℝ) : ℝ :=
  speed_in_mps * time_in_seconds

theorem distance_covered :
  calculate_distance (convert_kmph_to_mps (effective_speed_downstream boat_speed_still_water current_speed)) time_taken = 89.99280057595392 :=
by
  sorry

end distance_covered_l376_376650


namespace quadratic_has_two_distinct_real_roots_l376_376731

theorem quadratic_has_two_distinct_real_roots (a : ℝ) (h : a ≠ 0): 
  (a < 4 / 3) ↔ (∃ x y : ℝ, x ≠ y ∧  a * x^2 - 4 * x + 3 = 0 ∧ a * y^2 - 4 * y + 3 = 0) := 
sorry

end quadratic_has_two_distinct_real_roots_l376_376731


namespace weight_first_cat_is_2_l376_376287

-- Definitions for conditions
def weight_second_cat : ℕ := 7
def weight_third_cat : ℕ := 4
def total_weight : ℕ := 13

-- Theorem stating the weight of the first cat
theorem weight_first_cat_is_2 :
  ∃ weight_first_cat : ℕ, weight_first_cat + weight_second_cat + weight_third_cat = total_weight ∧ weight_first_cat = 2 :=
by 
  use 2
  simp [weight_second_cat, weight_third_cat, total_weight]
  exact sorry

end weight_first_cat_is_2_l376_376287


namespace two_A_sub_B_l376_376409

variable (x : ℝ) (A B : ℝ)
hypothesis hA : A = 2 * x ^ 2 + x * y - 3
hypothesis hB : B = -x ^ 2 + 2 * x * y - 1

theorem two_A_sub_B :
  2 * A - B = 5 * x ^ 2 - 5 := by
  rw [hA, hB]
  sorry

end two_A_sub_B_l376_376409


namespace amount_of_flour_already_put_in_l376_376875

theorem amount_of_flour_already_put_in 
  (total_flour_needed : ℕ) (flour_remaining : ℕ) (x : ℕ) 
  (h1 : total_flour_needed = 9) 
  (h2 : flour_remaining = 7) 
  (h3 : total_flour_needed - flour_remaining = x) : 
  x = 2 := 
sorry

end amount_of_flour_already_put_in_l376_376875


namespace analytical_expression_range_of_f_l376_376747

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 1

theorem analytical_expression {f : ℝ → ℝ}
  (h1 : ∀ x, f(x+1) - f(x) = 2*x)
  (h2 : f(0) = 1)
  (h3 : ∃ a b c, f = λ x, a*x^2 + b*x + c) :
  f = λ x, x^2 - x + 1 :=
by
  sorry

theorem range_of_f {f : ℝ → ℝ}
  (h_def : f = λ x, x^2 - x + 1) :
  set.range (λ x, f x) (Icc (-1 : ℝ) 1) = 
  set.Icc (3 / 4 : ℝ) 3 :=
by
  sorry

end analytical_expression_range_of_f_l376_376747


namespace simplify_expr_l376_376630

theorem simplify_expr (a : ℝ) (h_a : a = (8:ℝ)^(1/2) * (1/2) - (3:ℝ)^(1/2)^(0) ) : 
  a = (2:ℝ)^(1/2) - 1 := 
by
  sorry

end simplify_expr_l376_376630


namespace gcd_8m_6n_l376_376801

theorem gcd_8m_6n (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : Nat.gcd m n = 7) : Nat.gcd (8 * m) (6 * n) = 14 := 
by
  sorry

end gcd_8m_6n_l376_376801


namespace min_value_expression_l376_376421

variable {a b : ℝ}

theorem min_value_expression
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : a + b = 4) : 
  (∃ C, (∀ a b, a > 0 → b > 0 → a + b = 4 → (b / a + 4 / b) ≥ C) ∧ 
         (∀ a b, a > 0 → b > 0 → a + b = 4 → (b / a + 4 / b) = C)) ∧ 
         C = 3 :=
  by sorry

end min_value_expression_l376_376421


namespace toyota_to_honda_ratio_l376_376223

theorem toyota_to_honda_ratio
  (T H : ℕ)
  (h : 0.4 * T + 0.6 * H = 52):
  (T : ℚ) / H = 5 / 1 :=
sorry

end toyota_to_honda_ratio_l376_376223


namespace primes_sum_difference_l376_376173

def primes_in_range (a b : ℕ) : List ℕ := (list.range' a (b - a + 1)).filter Nat.Prime

noncomputable def set_A : List ℕ := primes_in_range 50 120
noncomputable def set_B : List ℕ := primes_in_range 120 250

theorem primes_sum_difference :
  let sum_A := list.sum set_A
  let sum_B := list.sum set_B
  sum_B - sum_A = 2816 :=
by
  sorry

end primes_sum_difference_l376_376173


namespace solution_set_ineq_l376_376725

theorem solution_set_ineq (x : ℝ) : 
  x * (x + 2) > 0 → abs x < 1 → 0 < x ∧ x < 1 := by
sorry

end solution_set_ineq_l376_376725


namespace fraction_product_equivalence_l376_376308

theorem fraction_product_equivalence :
  (1 / 3) * (1 / 2) * (2 / 5) * (3 / 7) = 6 / 35 := 
by 
  sorry

end fraction_product_equivalence_l376_376308


namespace median_of_consecutive_integers_l376_376583

theorem median_of_consecutive_integers (sum_n : ℤ) (n : ℤ) 
  (h1 : sum_n = 6^4) (h2 : n = 36) : 
  (sum_n / n) = 36 :=
by
  sorry

end median_of_consecutive_integers_l376_376583


namespace find_integer_n_l376_376714

theorem find_integer_n (n : ℤ) (h1 : n ≥ 3) (h2 : ∃ k : ℚ, k * k = (n^2 - 5) / (n + 1)) : n = 3 := by
  sorry

end find_integer_n_l376_376714


namespace smallest_number_is_61_point_4_l376_376949

theorem smallest_number_is_61_point_4 (x y z t : ℝ)
  (h1 : y = 2 * x)
  (h2 : z = 4 * y)
  (h3 : t = (y + z) / 3)
  (h4 : (x + y + z + t) / 4 = 220) :
  x = 2640 / 43 :=
by sorry

end smallest_number_is_61_point_4_l376_376949


namespace median_of_consecutive_integers_l376_376586

theorem median_of_consecutive_integers (sum_n : ℤ) (n : ℤ) 
  (h1 : sum_n = 6^4) (h2 : n = 36) : 
  (sum_n / n) = 36 :=
by
  sorry

end median_of_consecutive_integers_l376_376586


namespace fractional_linear_function_solution_l376_376713

-- Defining a fractional-linear function
def f (x : ℝ) : ℝ := x / 2

-- The main theorem to be proven
theorem fractional_linear_function_solution :
  f 0 = 0 ∧ f 1 = 4 ∧ f 4 = 2 :=
by
  sorry

end fractional_linear_function_solution_l376_376713


namespace sum_of_solutions_l376_376764

-- Define the conditions
variables {f : ℝ → ℝ}
axiom fx_symmetry : ∀ x : ℝ, f(1 - x) = f(1 + x)
axiom fx_zero_solutions : ∃ S : finset ℝ, S.card = 2009 ∧ ∀ x ∈ S, f(x) = 0

-- Define the proof goal
theorem sum_of_solutions : 
  ∃ S : finset ℝ, S.card = 2009 ∧ (∀ x ∈ S, f(x) = 0) ∧ S.sum id = 2009 :=
sorry

end sum_of_solutions_l376_376764


namespace distance_from_O_to_plane_ABC_l376_376819

-- Define respective points and conditions
variables {P A B C O : ℝ × ℝ × ℝ}
variable (PA PB PC : ℝ)
variable (PA_eq_1 : PA = 1)

-- Define the mutually perpendicular nature and lengths
def right_triang_pyramid (P A B C : ℝ × ℝ × ℝ) : Prop :=
  let PA := (A.1 - P.1, A.2 - P.2, A.3 - P.3) in
  let PB := (B.1 - P.1, B.2 - P.2, B.3 - P.3) in
  let PC := (C.1 - P.1, C.2 - P.2, C.3 - P.3) in
  (PA.1 * PB.1 + PA.2 * PB.2 + PA.3 * PB.3 = 0) ∧
  (PA.1 * PC.1 + PA.2 * PC.2 + PA.3 * PC.3 = 0) ∧
  (PB.1 * PC.1 + PB.2 * PC.2 + PB.3 * PC.3 = 0) ∧
  (PA.1^2 + PA.2^2 + PA.3^2 = 1) ∧
  (PB.1^2 + PB.2^2 + PB.3^2 = 1) ∧
  (PC.1^2 + PC.2^2 + PC.3^2 = 1)

-- Distance function from point to a plane passing through three points
def distance_from_point_to_plane 
  (O : ℝ × ℝ × ℝ) (A B C : ℝ × ℝ × ℝ) : ℝ :=
  let normal_vec := (
    (B.2 - A.2) * (C.3 - A.3) - (B.3 - A.3) * (C.2 - A.2),
    (B.3 - A.3) * (C.1 - A.1) - (B.1 - A.1) * (C.3 - A.3),
    (B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)
  ) in
  let d := -(normal_vec.1 * A.1 + normal_vec.2 * A.2 + normal_vec.3 * A.3) in
  (abs (normal_vec.1 * O.1 + normal_vec.2 * O.2 + normal_vec.3 * O.3 + d)
       / sqrt (normal_vec.1^2 + normal_vec.2^2 + normal_vec.3^2))

-- The theorem to be proved
theorem distance_from_O_to_plane_ABC
  (h : right_triang_pyramid P A B C)
  (hO : O = 
    ((A.1 + B.1 + C.1) / 2, 
    (A.2 + B.2 + C.2) / 2, 
    (A.3 + B.3 + C.3) / 2)) : 
  distance_from_point_to_plane O A B C = real.sqrt 3 / 6 := sorry

end distance_from_O_to_plane_ABC_l376_376819


namespace rectangle_segment_product_l376_376084

theorem rectangle_segment_product :
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 9) ∧ (5 * (19 + x) = x * (9 - x)) → x = 11.95 :=
by
  intros x h
  cases h with bounds equation
  cases bounds with lower_bound upper_bound
  -- Prove that x must equal 11.95
  sorry

end rectangle_segment_product_l376_376084


namespace determine_d_l376_376133

theorem determine_d (f g : ℝ → ℝ) (c d : ℝ) (h1 : ∀ x, f x = 5 * x + c) (h2 : ∀ x, g x = c * x + 3) (h3 : ∀ x, f (g x) = 15 * x + d) : d = 18 := 
  sorry

end determine_d_l376_376133


namespace parallel_vectors_l376_376431

theorem parallel_vectors (α : ℝ) 
  (h_parallel : ∃ k : ℝ, (sin α, cos α - 2 * sin α) = (k * 1, k * 2)) :
  (1 + 2 * sin α * cos α) / (sin α ^ 2 - cos α ^ 2) = - 5 / 3 := by
  sorry

end parallel_vectors_l376_376431


namespace divisor_of_271_l376_376966

theorem divisor_of_271 {
  (D : ℕ) (h₁ : 271 = D * 9 + 1) :
  D = 30 :=
sorry

end divisor_of_271_l376_376966


namespace max_omega_for_monotonic_sin_l376_376070

noncomputable def f (ω x : ℝ) : ℝ := Math.sin (ω * x)

theorem max_omega_for_monotonic_sin :
  ∀ (ω : ℝ), (∀ x y : ℝ, 
    (-Real.pi / 4) < x ∧ x < Real.pi / 4 → 
    (-Real.pi / 4) < y ∧ y < Real.pi / 4 →
    (x < y → f ω x ≤ f ω y ∨ f ω x ≥ f ω y)) →
    0 < ω → ω ≤ 2 :=
by
  sorry

end max_omega_for_monotonic_sin_l376_376070


namespace count_b_values_l376_376058

noncomputable def b_positive_integers (b : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ b ^ n = 256

theorem count_b_values : (finset.card (finset.filter b_positive_integers (finset.range (256 + 1))) = 4) :=
by {
  -- You can fill in the proof here if desired
  sorry
}

end count_b_values_l376_376058


namespace inequality_solution_l376_376177

theorem inequality_solution (x : ℝ) :
  ((x - 1)^2 + 1 > 0) → ((x - 3 < 0) ↔ (x ∈ Iio 3)) :=
by
  intro h
  exact sorry

end inequality_solution_l376_376177


namespace relationship_between_u_and_v_l376_376829

variables {r u v p : ℝ}
variables (AB G : ℝ)

theorem relationship_between_u_and_v (hAB : AB = 2 * r) (hAG_GF : u = (p^2 / (2 * r)) - p) :
    v^2 = u^3 / (2 * r - u) :=
sorry

end relationship_between_u_and_v_l376_376829


namespace find_max_l376_376348

noncomputable def maximum_expression (x : ℝ) : ℝ :=
  real.sqrt (2 * x + 27) + real.sqrt (17 - x) + real.sqrt (3 * x)

theorem find_max : ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 17 ∧ maximum_expression x = 14.951 :=
by
  use 17
  split
  exact le_refl 17
  split
  exact le_of_lt (by linarith : (17 : ℝ) < 18)
  sorry

end find_max_l376_376348


namespace sequence_sum_l376_376091

open scoped BigOperators

variable {α : Type*} [AddGroup α] [Module ℤ α] [ZMod]

-- Definitions used in conditions
def arithmetic_sequence (a : ℕ → α) : Prop :=
  ∀ n m k : ℕ, m = n + k → a m = a n + k • a 1

axiom seq : ℕ → ℤ  -- The arithmetic sequence
axiom h : seq 3 + seq 4 + seq 5 + seq 6 + seq 7 + seq 8 + seq 9 = 420

-- Problem to prove
theorem sequence_sum : seq 2 + seq 10 = 120 :=
by
  sorry

end sequence_sum_l376_376091


namespace sum_tangents_equal_third_l376_376597

theorem sum_tangents_equal_third
  {S1 S2 S3 : Circle} (h1 : S1.radius = S2.radius)
  (h2 : S2.radius = S3.radius) (P : Point)
  (h3 : P ∈ S1) (T1 T2 T3 : Line)
  (h4 : Tangent S1 T1 P) (h5 : Tangent S2 T2 P)
  (h6 : Tangent S3 T3 P) :
  length(T1) + length(T2) = length(T3) :=
sorry

end sum_tangents_equal_third_l376_376597


namespace M_union_N_eq_M_l376_376860

def M : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ |x * y| = 1 ∧ x > 0}

def N : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ Real.arctan x + Real.arccot y = Real.pi}

theorem M_union_N_eq_M : M ∪ N = M :=
by
  sorry

end M_union_N_eq_M_l376_376860


namespace even_function_expression_l376_376758

theorem even_function_expression (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = x * (2 * x - 1)) :
  ∀ x, x > 0 → f x = x * (2 * x + 1) :=
by 
  sorry

end even_function_expression_l376_376758


namespace angle_EFG_and_angle_H_l376_376440

variables (E F G H : Type) [EuclideanGeometry E] [EuclideanGeometry F] [EuclideanGeometry G] [EuclideanGeometry H]
variables (EFGH : Parallelogram E F G H)
variables (angleF : Angle F = 120)
variables (diagonalEH : Bisects E H)

theorem angle_EFG_and_angle_H :
  (angleEFG E F G H = 30) ∧ (angleH = 120) :=
sorry

end angle_EFG_and_angle_H_l376_376440


namespace trigonometric_identity_l376_376018

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
    Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = -4 / 3 :=
sorry

end trigonometric_identity_l376_376018


namespace sin_double_angle_l376_376064

-- Definition of the conditions given in the problem
def tan_alpha : ℝ := -1 / 3

-- Theorem statement to prove
theorem sin_double_angle : ∀ α : ℝ, real.tan α = tan_alpha → real.sin (2 * α) = -3 / 5 :=
by intro α h; sorry

end sin_double_angle_l376_376064


namespace part1_part2_l376_376745

-- Part (1): Prove k = 3 given x = -1 is a solution
theorem part1 (k : ℝ) (h : k * (-1)^2 + 4 * (-1) + 1 = 0) : k = 3 := 
sorry

-- Part (2): Prove k ≤ 4 and k ≠ 0 for the quadratic equation to have two real roots
theorem part2 (k : ℝ) (h : 16 - 4 * k ≥ 0) : k ≤ 4 ∧ k ≠ 0 :=
sorry

end part1_part2_l376_376745


namespace retirement_year_l376_376980

-- Define the basic conditions
def rule_of_70 (age: ℕ) (years_of_employment: ℕ) : Prop :=
  age + years_of_employment ≥ 70

def age_in_hiring_year : ℕ := 32
def hiring_year : ℕ := 1987

theorem retirement_year : ∃ y: ℕ, rule_of_70 (age_in_hiring_year + y) y ∧ (hiring_year + y = 2006) :=
  sorry

end retirement_year_l376_376980


namespace move_sin_graph_left_l376_376030

theorem move_sin_graph_left {φ : ℝ} (hφ : |φ| < π / 2) (h_symm : ∀ x : ℝ, x = π / 6 → sin (2x + φ) = sin (2 (π / 6) + φ)) :
  ∃ t : ℝ, t = π / 12 ∧ (∀ x : ℝ, sin (2 x + φ) = sin (2 (x + t))) :=
by
  sorry

end move_sin_graph_left_l376_376030


namespace circumscribed_circle_area_ratio_l376_376282

theorem circumscribed_circle_area_ratio (P : ℝ) (hP : P > 0) :
  let
    R1 := (P * Real.sqrt 3) / 9,
    R2 := P / 6,
    A := Real.pi * R1^2,
    B := Real.pi * R2^2
  in
  A / B = 4 / 3 :=
by
  -- Definitions
  let R1 := (P * Real.sqrt 3) / 9,
  let R2 := P / 6,
  let A := Real.pi * R1^2,
  let B := Real.pi * R2^2
  -- Conclusion
  show A / B = 4 / 3 from sorry

end circumscribed_circle_area_ratio_l376_376282


namespace derivative_f_at_0_l376_376255

def f (x : ℝ) : ℝ := if x ≠ 0 then tan (x^3 + x^2 * sin (2/x)) else 0

noncomputable def f_prime_at_0 : ℝ :=
deriv f 0

theorem derivative_f_at_0 : f_prime_at_0 = 0 :=
by
  have h0 : f 0 = 0 := rfl
  have h_diff : differentiable_at ℝ f 0 := 
    sorry
  have h_deriv : deriv f 0 = 0 :=
    sorry
  exact h_deriv

end derivative_f_at_0_l376_376255


namespace asymptote_slope_of_hyperbola_l376_376693

noncomputable def hyperbola_asymptote_slope : Prop :=
  let equation := λ x y : ℝ, x^2 / 16 - y^2 / 25 = 1
  let asymptote_eq := λ m x y : ℝ, y = m * x ∨ y = -m * x
  ∀ x y : ℝ, equation x y → ∃ m : ℝ, m > 0 ∧ asymptote_eq m x y ∧ m = 5 / 4

theorem asymptote_slope_of_hyperbola : hyperbola_asymptote_slope :=
sorry

end asymptote_slope_of_hyperbola_l376_376693


namespace shaded_area_of_rotated_semicircle_l376_376718

theorem shaded_area_of_rotated_semicircle (R : ℝ) :
  let α := 60 * (Real.pi / 180) in -- 60 degrees in radians
  let S0 := (Real.pi * R^2) / 2 in
  let SectorArea := (1 / 2) * (2 * R)^2 * (Real.pi / 3) in
  SectorArea = (2 * Real.pi * R^2) / 3 :=
by
  sorry

end shaded_area_of_rotated_semicircle_l376_376718


namespace correct_calculation_l376_376236

theorem correct_calculation (h1 : sqrt 20 = 2 * sqrt 10)
                           (h2 : sqrt 2 * sqrt 3 = sqrt 6)
                           (h3 : sqrt 4 - sqrt 2 = sqrt 2)
                           (h4 : sqrt ((-3)^2) = -3) : sqrt 2 * sqrt 3 = sqrt 6 :=
by
  sorry

end correct_calculation_l376_376236


namespace area_difference_l376_376827

-- Definitions based on the conditions
variables {A B C D F : Type} [PlaneType : AffinePlane A B C]

def right_angle (α : Angle) : Prop := α = 90
def area_of_triangle (X Y Z : A) : Real :=
  (1/2) * (distance X Y) * (distance Y Z)

-- Specific problem conditions
def conditions : Prop :=
  (right_angle (Angle FAB)) ∧
  (right_angle (Angle ABC)) ∧
  (distance A B = 5) ∧
  (distance B C = 7) ∧
  (distance A F = 9) ∧
  (intersect AC BF D)

-- The target statement to be proven
theorem area_difference:
  conditions → (area_of_triangle A D F - area_of_triangle B D C = 5) :=
by
  sorry

end area_difference_l376_376827


namespace line_intersects_hyperbola_two_distinct_points_l376_376811

theorem line_intersects_hyperbola_two_distinct_points (k : ℝ) :
  (- (Real.sqrt 15) / 3 < k) ∧ (k < -1) ↔
  let discr := (4 * k) ^ 2 - 40 * (k ^ 2 - 1) in
  discr > 0 ∧
  let sum_roots := (- 4 * k / (k ^ 2 - 1)) in
  sum_roots > 0 ∧
  let product_roots := (10 / (k ^ 2 - 1)) in
  product_roots > 0 :=
by sorry

end line_intersects_hyperbola_two_distinct_points_l376_376811


namespace sum_sequence_2012_l376_376190

def a (n : ℕ) : ℝ := n * Real.cos (n * Real.pi / 2) + 1

def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)

theorem sum_sequence_2012 : S 2012 = 2012 := by
  sorry

end sum_sequence_2012_l376_376190


namespace ant_traverses_same_edge_twice_l376_376672

-- Define a dodecahedron face, edge, and vertices
structure Dodecahedron := 
  (vertices : Finset ℕ) 
  (edges : Finset (ℕ × ℕ)) 
  (faces : Finset (Finset ℕ))
  (face_count : faces.card = 12)

-- Define the path of the ant
def ant_path (D : Dodecahedron) : Type := List (ℕ × ℕ)

-- Define a valid ant path (that it never reverses direction)
def valid_ant_path (D : Dodecahedron) (path : ant_path D) : Prop :=
  ∀ e ∈ path, e ∈ D.edges ∧ (e.snd, e.fst) ∉ path

-- The theorem to prove
theorem ant_traverses_same_edge_twice (D : Dodecahedron) (path : ant_path D) (h_path : valid_ant_path D path) :
  ∃ e ∈ path, (path.count e >= 2) :=
sorry

end ant_traverses_same_edge_twice_l376_376672


namespace odd_function_periodic_function_f_on_interval_f_at_neg_9_over_2_l376_376029

noncomputable def f : ℝ → ℝ :=
λ x, if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x) else 0  -- initial definition for 0 ≤ x ≤ 1

theorem odd_function (f : ℝ → ℝ) : ∀ x, f(-x) = -f(x) :=
sorry

theorem periodic_function (f : ℝ → ℝ) : ∀ x, f(x + 2) = f(x) :=
sorry

theorem f_on_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f(x) = 2 * x * (1 - x) :=
sorry

theorem f_at_neg_9_over_2 : f(λ x, if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x) else 0) (-9/2) = -1/2 :=
begin
  have h1 : f(-9/2) = f(-5/2), from periodic_function _ _,
  have h2 : f(-5/2) = f(-1/2), from periodic_function _ _,
  have h3 : f(-1/2) = -f(1/2), from odd_function _ _,
  have h4 : f(1/2) = 2 * 1/2 * (1 - 1/2), from f_on_interval 1/2 (by norm_num),
  rw h4 at h3,
  rw h3,
  simp,
end

end odd_function_periodic_function_f_on_interval_f_at_neg_9_over_2_l376_376029


namespace hyperbola_asymptotes_l376_376540

theorem hyperbola_asymptotes (a : ℝ) (h : a ≠ 0) : 
  ∀ (x y : ℝ), (\frac{x^2}{2 * a} - \frac{y^2}{a} = 1) → (y = \frac{\sqrt{2}}{2} * x ∨ y = - \frac{\sqrt{2}}{2} * x) := 
sorry

end hyperbola_asymptotes_l376_376540


namespace speed_of_second_train_l376_376995

theorem speed_of_second_train (length_first_train : ℝ) (speed_first_train_kmph : ℝ) 
  (crossing_time : ℝ) (length_second_train : ℝ)
  (same_direction : true) : 
  length_first_train = 420 ∧ speed_first_train_kmph = 72 ∧ crossing_time = 105.99152067834574 ∧ length_second_train = 640 → 
  let speed_second_train_kmph := (speed_first_train_kmph * (1000 / 3600) - ( (length_first_train + length_second_train) / crossing_time)) * (3600 / 1000) in
  speed_second_train_kmph = 36 :=
by 
  intros h
  have h1 : length_first_train = 420 ∧ speed_first_train_kmph = 72 ∧ crossing_time = 105.99152067834574 ∧ length_second_train = 640 := h
  sorry

end speed_of_second_train_l376_376995


namespace binary_multiplication_addition_l376_376307

-- Define the binary representation of the given numbers
def b1101 : ℕ := 0b1101
def b111 : ℕ := 0b111
def b1011 : ℕ := 0b1011
def b1011010 : ℕ := 0b1011010

-- State the theorem
theorem binary_multiplication_addition :
  (b1101 * b111 + b1011) = b1011010 := 
sorry

end binary_multiplication_addition_l376_376307


namespace chairs_left_to_move_l376_376312

theorem chairs_left_to_move (total_chairs : ℕ) (carey_chairs : ℕ) (pat_chairs : ℕ) (h1 : total_chairs = 74)
  (h2 : carey_chairs = 28) (h3 : pat_chairs = 29) : total_chairs - carey_chairs - pat_chairs = 17 :=
by 
  sorry

end chairs_left_to_move_l376_376312


namespace trig_identity_l376_376620

theorem trig_identity (x : ℝ) (k : ℤ) (hx1 : cos x ≠ 0) (hx2 : sin x ≠ 0) :
  tan (5 * Real.pi / 2 + x) - 3 * tan x ^ 2 = (cos (2 * x) - 1) / (cos x) ^ 2 ↔
  ∃ k : ℤ, x = Real.pi / 4 * (4 * k - 1) := 
by
  sorry

end trig_identity_l376_376620


namespace intersection_A_B_l376_376491

def A := {x : ℝ | 2 ≤ x ∧ x ≤ 8}
def B := {x : ℝ | x^2 - 3 * x - 4 < 0}
def expected := {x : ℝ | 2 ≤ x ∧ x < 4 }

theorem intersection_A_B : (A ∩ B) = expected := 
by 
  sorry

end intersection_A_B_l376_376491


namespace customer_payment_strawberries_watermelons_max_discount_value_l376_376873

-- Definitions for prices
def price_strawberries : ℕ := 60
def price_jingbai_pears : ℕ := 65
def price_watermelons : ℕ := 80
def price_peaches : ℕ := 90

-- Definition for condition on minimum purchase for promotion
def min_purchase_for_promotion : ℕ := 120

-- Definition for percentage Li Ming receives
def li_ming_percentage : ℕ := 80
def customer_percentage : ℕ := 100

-- Proof problem for part 1
theorem customer_payment_strawberries_watermelons (x : ℕ) (total_price : ℕ) :
  x = 10 →
  total_price = price_strawberries + price_watermelons →
  total_price >= min_purchase_for_promotion →
  total_price - x = 130 :=
  by sorry

-- Proof problem for part 2
theorem max_discount_value (m x : ℕ) :
  m >= min_purchase_for_promotion →
  (m - x) * li_ming_percentage / customer_percentage ≥ m * 7 / 10 →
  x ≤ m / 8 :=
  by sorry

end customer_payment_strawberries_watermelons_max_discount_value_l376_376873


namespace find_a_l376_376075

theorem find_a (a : ℝ) (x₁ x₂ : ℝ) :
  (2 * x₁ + 1 = 3) →
  (2 - (a - x₂) / 3 = 1) →
  (x₁ = x₂) →
  a = 4 :=
by
  intros h₁ h₂ h₃
  sorry

end find_a_l376_376075


namespace geometric_sequence_sum_l376_376446

variable {α : Type*} [LinearOrderedField α]

/-- Given a geometric sequence {a_n}, where a_1 + a_2 + a_3 = 2 and a_3 + a_4 + a_5 = 8,
prove that a_4 + a_5 + a_6 = ±16. -/
theorem geometric_sequence_sum 
  {a : ℕ → α} {q : α} (ha_geometric : ∀ n, a(n+1) = q * a(n))
  (h_sum1 : a(1) + a(2) + a(3) = 2)
  (h_sum2 : a(3) + a(4) + a(5) = 8) :
  a(4) + a(5) + a(6) = 16 ∨ a(4) + a(5) + a(6) = -16 :=
sorry

end geometric_sequence_sum_l376_376446


namespace sum_of_digits_product_l376_376872

-- Define what it means for a number to be good
def isGoodNumber (n : ℕ) : Prop :=
  ∀ d ∈ (Nat.digits 10 n), d = 0 ∨ d = 1

-- Define the sum of the digits of a number
def sumOfDigits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

-- The theorem statement
theorem sum_of_digits_product (A B : ℕ) (hA : isGoodNumber A) (hB : isGoodNumber B) (hAB : isGoodNumber (A * B)) :
  sumOfDigits (A * B) = sumOfDigits A * sumOfDigits B := sorry

end sum_of_digits_product_l376_376872


namespace Rajesh_savings_proof_l376_376168

def Rajesh_spending_food := 0.35
def Rajesh_spending_medicines := 0.20
def Rajesh_spending_rent := 0.10
def Rajesh_spending_transportation := 0.05
def Rajesh_spending_miscellaneous := 0.10
def Rajesh_salary_increase_per_year := 0.08
def Rajesh_current_salary := 15000
def Rajesh_total_spending_percentage := Rajesh_spending_food + Rajesh_spending_medicines + Rajesh_spending_rent + Rajesh_spending_transportation + Rajesh_spending_miscellaneous
def Rajesh_percentage_saved := 1 - Rajesh_total_spending_percentage
def Rajesh_salary_second_year := Rajesh_current_salary * (1 + Rajesh_salary_increase_per_year)
def Rajesh_monthly_savings_second_year := Rajesh_salary_second_year * Rajesh_percentage_saved

theorem Rajesh_savings_proof : Rajesh_monthly_savings_second_year = 3240 := by
  sorry

end Rajesh_savings_proof_l376_376168


namespace median_of_36_consecutive_integers_l376_376565

theorem median_of_36_consecutive_integers (x : ℤ) (sum_eq : (∑ i in finset.range 36, (x + i)) = 6^4) : (17 + 18) / 2 = 36 :=
by
  -- Proof goes here
  sorry

end median_of_36_consecutive_integers_l376_376565


namespace price_after_discounts_l376_376985

theorem price_after_discounts :
  ∃ P X : ℝ, 
  let P_discount := 0.504 * P in
  (X * P = 800) ∧ 
  ((X + 5) * P_discount = 800) ∧ 
  (P_discount = 79.36) := 
sorry

end price_after_discounts_l376_376985


namespace median_of_36_consecutive_integers_l376_376581

theorem median_of_36_consecutive_integers (f : ℕ → ℤ) (h_consecutive : ∀ n : ℕ, f (n + 1) = f n + 1) 
(h_size : ∃ k, f 36 = f 0 + 35) (h_sum : ∑ i in finset.range 36, f i = 6^4) : 
(∃ m, m = f (36 / 2 - 1) ∧ m = 36) :=
by
  sorry

end median_of_36_consecutive_integers_l376_376581


namespace range_of_k_l376_376899

-- Defining the equation to be solved
def equation (k x : ℝ) : ℝ :=
  k * (9^x) - k * (3^(x + 1)) + 6 * (k - 5)

-- The interval for x
def interval_x := set.Icc (0 : ℝ) 2

-- The range of k to be proved
def valid_k (k : ℝ) : Prop :=
  1 / 2 ≤ k ∧ k ≤ 8

theorem range_of_k :
  (∀ k x : ℝ, (equation k x = 0) → x ∈ interval_x → valid_k k) ∧
  (∀ k : ℝ, valid_k k → ∃ x : ℝ, x ∈ interval_x ∧ equation k x = 0) := 
begin
  sorry,
end

end range_of_k_l376_376899


namespace batsman_average_after_11th_inning_l376_376243

theorem batsman_average_after_11th_inning 
  (x : ℝ) 
  (h1 : (10 * x + 95) / 11 = x + 5) : 
  x + 5 = 45 :=
by 
  sorry

end batsman_average_after_11th_inning_l376_376243


namespace calc_expression_solve_equation_l376_376970

-- Problem 1: Calculation

theorem calc_expression : 
  |Real.sqrt 3 - 2| + Real.sqrt 12 - 6 * Real.sin (Real.pi / 6) + (-1/2 : Real)⁻¹ = Real.sqrt 3 - 3 := 
by {
  sorry
}

-- Problem 2: Solve the Equation

theorem solve_equation (x : Real) : 
  x * (x + 6) = -5 ↔ (x = -5 ∨ x = -1) := 
by {
  sorry
}

end calc_expression_solve_equation_l376_376970


namespace min_a_squared_plus_b_squared_l376_376632

theorem min_a_squared_plus_b_squared (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 :=
sorry

end min_a_squared_plus_b_squared_l376_376632


namespace find_area_of_octagon_l376_376324

noncomputable def area_of_octagon
  (circumradius : ℝ)
  (perpendicular_bisectors : ∀ (P Q R : Point), Point)
  (reflection : ∀ (circumcenter side_midpoint : Point), Point)
  (perimeter : ℝ) : ℝ :=
sorry

theorem find_area_of_octagon :
  area_of_octagon 10 
    (λ P Q R => meet_perpendicular_bisector_with_circumcircle P Q R)
    (λ O PQ => reflect_over_side O PQ)
    45 = 225 :=
sorry

end find_area_of_octagon_l376_376324


namespace odd_blue_faces_in_cubes_l376_376667

noncomputable def count_odd_blue_faces (length width height : ℕ) : ℕ :=
if length = 6 ∧ width = 4 ∧ height = 2 then 16 else 0

theorem odd_blue_faces_in_cubes : count_odd_blue_faces 6 4 2 = 16 := 
by
  -- The proof would involve calculating the corners, edges, etc.
  sorry

end odd_blue_faces_in_cubes_l376_376667


namespace monotonic_intervals_find_max_value_l376_376778

open Real

noncomputable def f (x a : ℝ) : ℝ := x^2 * exp (-a * x)

theorem monotonic_intervals (a : ℝ) (ha : 0 < a) :
  (∀ x, x ∈ Ioo 0 (2 / a) → deriv (f x a) > 0) ∧
  (∀ x, x ∈ Ioo (2 / a) (2 / a + 1) → deriv (f x a) < 0) :=
sorry
  
theorem find_max_value (a : ℝ) (ha : 0 < a) :
  let max_on_interval :=
    if 0 < a ∧ a < 1 then 4 * exp (-2 * a)
    else if 1 ≤ a ∧ a ≤ 2 then (4 / a^2) * exp (-2)
    else if a > 2 then exp (-a)
    else 0 in
  ∀ x, x ∈ Icc 1 2 → f x a ≤ max_on_interval :=
sorry

end monotonic_intervals_find_max_value_l376_376778


namespace area_of_APQD_l376_376467

noncomputable def regular_pentagon_area :=
  1 / 2

structure Point (α : Type) :=
(x : α)
(y : α)

structure Pentagon (α : Type) :=
(A B C D E : Point α)

structure Star (α : Type) :=
(A C E B D : Point α)

def compute_intersection (l1 l2 : Line α) : Point α := sorry

structure Quadrilateral (α : Type) :=
(A P Q D : Point α)

def regular_pentagon (A B C D E : Point α) : Prop := sorry -- Regular pentagon condition definition

def star_area (star : Star α) : ℝ := 1

def quad_area (quad : Quadrilateral ℝ) : ℝ := sorry -- Function to compute the area

theorem area_of_APQD (A B C D E P Q : Point ℝ) 
  (pent : regular_pentagon A B C D E)
  (star : star_area (Star.mk A C E B D) = 1)
  (P_intersection : P = compute_intersection (Line.mk A C) (Line.mk B E))
  (Q_intersection : Q = compute_intersection (Line.mk B D) (Line.mk C E)) :
  quad_area (Quadrilateral.mk A P Q D) = regular_pentagon_area := 
begin
  sorry,
end 

end area_of_APQD_l376_376467


namespace increase_by_percentage_l376_376977

def initial_value : ℕ := 550
def percentage_increase : ℚ := 0.35
def final_value : ℚ := 742.5

theorem increase_by_percentage :
  (initial_value : ℚ) * (1 + percentage_increase) = final_value := by
  sorry

end increase_by_percentage_l376_376977


namespace CD_intersect_common_point_l376_376033

open_locale classical

variables (k1 k2 : Circle)
variables (P Q : Point) (e : Line)
variables (A B C D S : Point)

-- Define the conditions
axiom Circle_in_circle (h1 : P ∈ k1) (h2 : Q ∉ k2)
axiom arbitrary_line_through_P_not_Q (h3 : e ∩ₗ P) (h4 : Q ∉ e)

-- Define A, B as intersection points of k1 and line e
axiom intersection_points (h5 : A ∈ (e ∩ₗ k1)) (h6 : B ∈ (e ∩ₗ k1))

-- Define circumcircle of triangle ABQ
noncomputable def circumcircle_ABQ : Circle := circumcircle A B Q

-- Define intersections of circumcircle with k2
axiom intersections_with_k2 (h7 : C ∈ (circumcircle_ABQ ∩ k2)) (h8 : D ∈ (circumcircle_ABQ ∩ k2))

theorem CD_intersect_common_point :
∀ (P Q : Point) (k1 k2 : Circle) (e : Line) (A B C D S : Point),
P ∈ k1 ∧ Q ∉ k2 ∧ e ∩ₗ P ∧ Q ∉ e ∧
(A ∈ (e ∩ₗ k1)) ∧ (B ∈ (e ∩ₗ k1)) ∧
(C ∈ (circumcircle A B Q ∩ k2)) ∧ (D ∈ (circumcircle A B Q ∩ k2)) →
∃ S : Point, ∀ (C D : Point), line_through C D S := 
sorry

end CD_intersect_common_point_l376_376033


namespace sum_of_coefficients_l376_376148

theorem sum_of_coefficients : 
  let α_candidates := { α : ℤ | ∃ a b : ℤ, a < 0 ∧ b < 0 ∧ a ≠ b ∧ a * b = 25 ∧ α = a + b } in
  ∑ α in α_candidates, α = 26 :=
by
  -- Proof omitted
  sorry

end sum_of_coefficients_l376_376148


namespace stone_slab_length_l376_376263

-- Definitions of conditions
def num_stone_slabs := 30
def total_area := 67.5
def area_one_slab := total_area / num_stone_slabs

-- The length of a slab is the square root of the area of one slab
def length_one_slab := real.sqrt area_one_slab

-- The proof objective, proving the length is 1.5 meters
theorem stone_slab_length : length_one_slab = 1.5 := 
by 
  sorry

end stone_slab_length_l376_376263


namespace probability_y_gt_2x_l376_376891

theorem probability_y_gt_2x (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  Pr (y > 2 * x) = 1 / 4 :=
sorry

end probability_y_gt_2x_l376_376891


namespace gcd_fib_consecutive_rel_prime_gcd_fib_l376_376246

/-- 
Prove that the gcd of two consecutive Fibonacci numbers is 1.
-/
theorem gcd_fib_consecutive_rel_prime (n : ℕ) : Nat.gcd (Nat.fib n) (Nat.fib (n + 1)) = 1 :=
sorry

/-- 
Prove that gcd(F_m, F_n) = F_gcd(m, n) where F is Fibonacci sequence.
-/
theorem gcd_fib (m n : ℕ) : Nat.gcd (Nat.fib m) (Nat.fib n) = Nat.fib (Nat.gcd m n) :=
sorry

end gcd_fib_consecutive_rel_prime_gcd_fib_l376_376246


namespace intersection_area_l376_376885

noncomputable def intersecting_triangle_area : ℚ :=
  let A := (0 : ℝ, 0 : ℝ)
  let B := (9 : ℝ, 0 : ℝ)
  let C := (15 : ℝ, 8 : ℝ)
  let D := (-6 : ℝ, 8 : ℝ)
  -- Assuming point P is known as the intersection of AC and BD
  let P := (some_computed_value_x, some_computed_value_y)
  let area := (9 * (12/5) / 2)
  area

theorem intersection_area (m n : ℕ) (hmn : nat.gcd m n = 1) (h_area : intersecting_triangle_area = 54 / 5) : m + n = 59 := 
by {
  -- Given intersection is calculated as area = 54 / 5
  -- which is area in form m/n = 54/5
  have h1 : (54 : ℚ) / (5 : ℚ) = m / n := sorry,
  sorry
  -- Here we ensure m + n is 59
}

end intersection_area_l376_376885


namespace initially_calculated_average_height_l376_376179

theorem initially_calculated_average_height
    (A : ℕ)
    (initial_total_height : ℕ)
    (real_total_height : ℕ)
    (height_error : ℕ := 60)
    (num_boys : ℕ := 35)
    (actual_average_height : ℕ := 183)
    (initial_total_height_eq : initial_total_height = num_boys * A)
    (real_total_height_eq : real_total_height = num_boys * actual_average_height)
    (height_discrepancy : initial_total_height = real_total_height + height_error) :
    A = 181 :=
by
  sorry

end initially_calculated_average_height_l376_376179


namespace part_a_part_b_l376_376085

-- Define the round-robin chess tournament setting and irregular game
def roundRobinTournament (N : ℕ) : Prop :=
  ∀ i j : ℕ, (i < N ∧ j < N) → (i ≠ j → (wins i j ∨ draws i j ∨ wins j i))

def points (player : ℕ) (wins draws : ℕ → ℕ → Prop) : ℕ :=
  ∑ i in (λ i, if wins player i then 1 else if draws player i then 0.5 else 0)

def isIrregularGame (i j : ℕ) (wins : ℕ → ℕ → Prop) (points : ℕ → ℕ) : Prop :=
  wins i j ∧ points i < points j

-- Part (a): Irregular games cannot constitute more than 75%
theorem part_a (N : ℕ) (wins draws : ℕ → ℕ → Prop) (points : ℕ → ℕ → ℕ) :
  (roundRobinTournament N) →
  (sum_irregular_games wins points < 0.75 * total_games N) :=
sorry

-- Part (b): Irregular games can constitute more than 70%
theorem part_b (N : ℕ) (wins draws : ℕ → ℕ → Prop) (points : ℕ → ℕ → ℕ) :
  (roundRobinTournament N) →
  (sum_irregular_games wins points > 0.70 * total_games N) :=
sorry

end part_a_part_b_l376_376085


namespace correct_factorization_l376_376238

-- Definitions from conditions
def A: Prop := ∀ x y: ℝ, x^2 - 4*y^2 = (x + y) * (x - 4*y)
def B: Prop := ∀ x: ℝ, (x + 4) * (x - 4) = x^2 - 16
def C: Prop := ∀ x: ℝ, x^2 - 2*x + 1 = (x - 1)^2
def D: Prop := ∀ x: ℝ, x^2 - 8*x + 9 = (x - 4)^2 - 7

-- Goal is to prove that C is a correct factorization
theorem correct_factorization: C := by
  sorry

end correct_factorization_l376_376238


namespace jeff_maria_debts_settlement_l376_376109

-- Definitions based on the conditions
def jeff_pays : ℝ := 90
def maria_pays : ℝ := 150
def lee_pays : ℝ := 210
def total_paid : ℝ := jeff_pays + maria_pays + lee_pays
def each_should_pay : ℝ := total_paid / 3

-- Define the amounts given after the trip, based on conditions
def j := each_should_pay - jeff_pays
def m := each_should_pay - maria_pays

theorem jeff_maria_debts_settlement : j - m = 60 := by
  sorry

end jeff_maria_debts_settlement_l376_376109


namespace problem_statement_l376_376007

variable {θ₁ θ₂ θ₃ θ₄ : ℝ}

theorem problem_statement 
  (h₁ : 0 < θ₁ ∧ θ₁ < Real.pi / 2)
  (h₂ : 0 < θ₂ ∧ θ₂ < Real.pi / 2)
  (h₃ : 0 < θ₃ ∧ θ₃ < Real.pi / 2)
  (h₄ : 0 < θ₄ ∧ θ₄ < Real.pi / 2)
  (sum_angles : θ₁ + θ₂ + θ₃ + θ₄ = Real.pi) :
  (sqrt 2 * Real.sin θ₁ - 1) / Real.cos θ₁ +
  (sqrt 2 * Real.sin θ₂ - 1) / Real.cos θ₂ +
  (sqrt 2 * Real.sin θ₃ - 1) / Real.cos θ₃ +
  (sqrt 2 * Real.sin θ₄ - 1) / Real.cos θ₄ ≥ 0 :=
sorry

end problem_statement_l376_376007


namespace number_of_zeros_l376_376698

def f (x : ℝ) : ℝ := x^2 - 2^x

theorem number_of_zeros (f : ℝ → ℝ) : 
  (∀ x : ℝ, f x + f (x + 5) = 16) → 
  (∀ x : ℝ, x ∈ Ioc (-1) 4 → f x = x^2 - 2^x) →
  (∀ (a b : ℝ), 0 ≤ a ∧ b ≤ 2013 ∧ b = a + 2013 ∧ ∀ m n : ℤ, f m = 0 ∧ n * 5 ≤ m ∧ m ≤ n * 5 + 4 → ∃ n : ℕ, n = 402) :=
begin
  sorry
end

end number_of_zeros_l376_376698


namespace problem_solution_l376_376028

-- Defining the function f with its properties
variable (f : ℝ → ℝ)

-- The domain of f is ℝ
-- Given conditions:
def odd_function_property := ∀ x : ℝ, f(-2 * x + 1) = -f(2 * x + 1)
def symmetric_property := ∀ x : ℝ, f(x - 1) = f(-x - 1)

-- The statements to prove:
theorem problem_solution 
  (h1 : odd_function_property f)
  (h2 : symmetric_property f) :
  f 1 = 0 ∧ f 5 = 0 := 
sorry -- Proof goes here

end problem_solution_l376_376028


namespace problem_solution_l376_376386

variables {ℝ : Type} [linear_ordered_field ℝ]

def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem problem_solution (f g : ℝ → ℝ) (h_dom : ∀ x, x ∈ ℝ) 
  (h_odd : odd_function g) 
  (h_eq1 : ∀ x, f x + g x = 2)
  (h_eq2 : ∀ x, f x + g (x - 2) = 2) :
  (f 0 = 2) ∧ (∀ n : ℕ, ∑ i in finset.range n, g i = 0) :=
begin
  sorry
end

end problem_solution_l376_376386


namespace negation_of_tan_one_l376_376399

theorem negation_of_tan_one :
  (∃ x : ℝ, Real.tan x = 1) ↔ ¬ (∀ x : ℝ, Real.tan x ≠ 1) :=
by
  sorry

end negation_of_tan_one_l376_376399


namespace lowest_price_per_component_l376_376273

theorem lowest_price_per_component (
  production_cost_per_component : ℕ := 80,
  shipping_cost_per_component : ℕ := 3,
  fixed_cost_per_month : ℕ := 16500,
  number_of_components : ℕ := 150
) : 
  let cost_per_component := production_cost_per_component + shipping_cost_per_component in 
  let total_variable_cost := cost_per_component * number_of_components in
  let total_cost := total_variable_cost + fixed_cost_per_month in
  let lowest_price := total_cost / number_of_components 
  in lowest_price = 193 := 
by
  -- proof placeholder
  sorry

end lowest_price_per_component_l376_376273


namespace median_of_consecutive_integers_sum_eq_6_pow_4_l376_376571

theorem median_of_consecutive_integers_sum_eq_6_pow_4 :
  ∀ (s : ℕ) (n : ℕ), s = 36 → ∑ i in finset.range 36, (n + i) = 6^4 → 36 / 2 = 36 :=
by
  sorry

end median_of_consecutive_integers_sum_eq_6_pow_4_l376_376571


namespace combined_salaries_BCDE_l376_376200

-- Define the given conditions
def salary_A : ℕ := 10000
def average_salary : ℕ := 8400
def num_individuals : ℕ := 5

-- Define the total salary of all individuals
def total_salary_all : ℕ := average_salary * num_individuals

-- Define the proof problem
theorem combined_salaries_BCDE : (total_salary_all - salary_A) = 32000 := by
  sorry

end combined_salaries_BCDE_l376_376200


namespace blue_paint_cans_needed_l376_376160

-- Definitions of the conditions
def blue_to_green_ratio : ℕ × ℕ := (4, 3)
def total_cans : ℕ := 42
def expected_blue_cans : ℕ := 24

-- Proof statement
theorem blue_paint_cans_needed (r : ℕ × ℕ) (total : ℕ) (expected : ℕ) 
  (h1: r = (4, 3)) (h2: total = 42) : expected = 24 :=
by
  sorry

end blue_paint_cans_needed_l376_376160


namespace a_5_is_130_l376_376786

def a (n : ℕ) : ℕ := ∑ i in (Finset.range 20).map (λ x, x + 1), |n - i|

-- Given that n is a positive natural number between 1 and 20
def condition (n : ℕ) : Prop := n > 0 ∧ n ≤ 20

theorem a_5_is_130 (n : ℕ) (h1 : condition n) : a 5 = 130 := 
by
  sorry

end a_5_is_130_l376_376786


namespace determine_function_l376_376700

theorem determine_function (f : ℚ → ℝ) (hf : ∀ x y : ℚ, f x ≠ 0 ∧ f y ≠ 0) 
  (functional_eq : ∀ x y : ℚ, (f x)^2 * f (2 * y) + (f y)^2 * f (2 * x) = 2 * f x * f y * f (x + y)) :
  ∃ b c : ℝ, b ≠ 0 ∧ c > 0 ∧ ∀ x : ℚ, f x = b * c ^ x :=
sorry

end determine_function_l376_376700


namespace unique_functional_equation_l376_376351

theorem unique_functional_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = x + y :=
sorry

end unique_functional_equation_l376_376351


namespace increasing_intervals_range_of_f_l376_376394

def f (x : ℝ) : ℝ := (Real.sin x)^2 + (Real.sqrt 3) * (Real.sin x) * (Real.cos x)

-- Problem statement (1)
theorem increasing_intervals : ∀ (k : ℤ), 
  ∀ (x : ℝ), (k * Real.pi - Real.pi / 6 ≤ x) ∧ (x ≤ k * Real.pi + Real.pi / 3) → f x = sin (2 * x - Real.pi / 6) + 1 / 2
:= sorry

-- Problem statement (2)
theorem range_of_f : ∀ (x : ℝ), 
  (0 ≤ x) ∧ (x ≤ Real.pi / 2) → (0 ≤ f x) ∧ (f x ≤ 3 / 2)
:= sorry

end increasing_intervals_range_of_f_l376_376394


namespace knight_game_winner_l376_376268

theorem knight_game_winner (n : ℕ) (h_n : n > 3):
  let initial_pos := (1, 1)
  let target_pos := (n, n)
  (exists (moves1 moves2 : list (ℕ × ℕ)), 
    -- Player 1's moves: knight move twice
    (∀ i < moves1.length, 
      (i % 2 = 0 → (moves1.nth i = (initial_pos) ∨ (moves1.nth i).is_knight_move (initial_pos))) ∧
    -- moves to diagonal
      (i % 2 = 1 → (moves1.nth i = (target_pos) ∨ (moves1.nth i).is_knight_move (moves1.nth (i-1))) ∧ 
      is_diag(moves1.nth i) ))) ∧
  -- Player 2's moves: extended knight move once
    (∀ i < moves2.length, 
      (i % 2 = 1 → (moves2.nth i).is_extended_knight_move (moves1.nth (i-1))) ∧
    -- moves off diagonal
      not(is_diag (moves2.nth i) ))) →
  Player 1 always wins :=
sorry

-- Auxiliary definitions, you may need something like these

-- Represents if a move is a standard knight move
def ℕ × ℕ).is_knight_move(p : ℕ × ℕ) : Prop := 
  ∃ a b, (a, b) = (p.1 + 2, p.2 + 1) ∨ (a, b) = (p.1 + 1, p.2 + 2) ∨ 
          (a, b) = (p.1 - 2, p.2 + 1) ∨ (a, b) = (p.1 - 1, p.2 + 2) ∨ 
          (a, b) = (p.1 + 2, p.2 - 1) ∨ (a, b) = (p.1 + 1, p.2 - 2) ∨ 
          (a, b) = (p.1 - 2, p.2 - 1) ∨ (a, b) = (p.1 - 1, p.2 - 2)

-- Represents if a move is an extended knight move
def ℕ × ℕ).is_extended_knight_move(p : ℕ × ℕ) : Prop := 
  ∃ a b, (a, b) = (p.1 + 3, p.2 + 1) ∨ (a, b) = (p.1 + 1, p.2 + 3) ∨ 
          (a, b) = (p.1 - 3, p.2 + 1) ∨ (a, b) = (p.1 - 1, p.2 + 3) ∨ 
          (a, b) = (p.1 + 3, p.2 - 1) ∨ (a, b) = (p.1 + 1, p.2 - 3) ∨ 
          (a, b) = (p.1 - 3, p.2 - 1) ∨ (a, b) = (p.1 - 1, p.2 - 3)

-- Diagonal definition
def is_diag : ℕ × ℕ → bool
| (x, y) := x = y

end knight_game_winner_l376_376268


namespace curves_representation_minimum_distance_l376_376905

open Real

/-- Convert the parametric equation of curve C1 to its standard form, and
    the standard form of curve C2 to its parametric form --/
theorem curves_representation :
  (∃ t, (x = -4 + cos t ∧ y = 3 + sin t) ↔ (x + 4)^2 + (y - 3)^2 = 1) ∧
  (∃ θ, (x = 8 * cos θ ∧ y = 3 * sin θ) ↔ (x / 8)^2 + (y / 3)^2 = 1) :=
by sorry

/-- Find the minimum distance from the midpoint M of points P and Q to the line C3 --/
theorem minimum_distance :
  let M (θ : ℝ) := (-2 + 4 * cos θ, 2 + (3 / 2) * sin θ) in
  ∀ θ, ∃ d, d = √5 / 4 * abs (4 * cos θ - 3 * sin θ - 13) →
  (cos θ = 4 / 5 ∧ sin θ = -3 / 5) → d = 8 * √5 / 5 :=
by sorry

end curves_representation_minimum_distance_l376_376905


namespace exists_good_board_l376_376116

theorem exists_good_board (m n : ℕ) (h₁ : m ≥ n) (h₂ : n ≥ 4) :
    (∃ board : list (list ℕ), 
        (∀ i j, i < m ∧ j < n → board.nth_le i sorry.nth_le j sorry ∈ (0,1)) ∧  -- Numbers are 0 or 1
        (¬ (∀ i j, i < m ∧ j < n → board.nth_le i sorry = 0) ∨  -- Not all numbers on the board are 0
           ∀ i j, i < m ∧ j < n → board.nth_le i sorry = 1) ∧  -- Not all numbers on the board are 1
        (∀ i j, 3 ≤ m ∧ 3 ≤ n → let sum_3x3 := 
          ∑ a in (range 3), ∑ b in (range 3), board.nth_le (i + a) sorry.nth_le (j + b) sorry 
          in sum_3x3 = k) ∧  -- Sum of all 3x3 sub-boards is the same
        (∀ i j, 4 ≤ m ∧ 4 ≤ n → let sum_4x4 := 
          ∑ a in (range 4), ∑ b in (range 4), board.nth_le (i + a) sorry.nth_le (j + b) sorry 
          in sum_4x4 = l) -- Sum of all 4x4 sub-boards is the same) 
    ↔ ((∃ k : ℕ, m = 4 + k ∧ n = 4) ∨ (∃ k : ℕ, m = 5 + k ∧ n = 5)) := 
sorry

end exists_good_board_l376_376116


namespace slices_per_person_l376_376662

theorem slices_per_person
  (small_pizza_slices : ℕ)
  (large_pizza_slices : ℕ)
  (small_pizzas_purchased : ℕ)
  (large_pizzas_purchased : ℕ)
  (george_slices : ℕ)
  (bob_extra : ℕ)
  (susie_divisor : ℕ)
  (bill_slices : ℕ)
  (fred_slices : ℕ)
  (mark_slices : ℕ)
  (ann_slices : ℕ)
  (kelly_multiplier : ℕ) :
  small_pizza_slices = 4 →
  large_pizza_slices = 8 →
  small_pizzas_purchased = 4 →
  large_pizzas_purchased = 3 →
  george_slices = 3 →
  bob_extra = 1 →
  susie_divisor = 2 →
  bill_slices = 3 →
  fred_slices = 3 →
  mark_slices = 3 →
  ann_slices = 2 →
  kelly_multiplier = 2 →
  (2 * (small_pizzas_purchased * small_pizza_slices + large_pizzas_purchased * large_pizza_slices -
    (george_slices + (george_slices + bob_extra) + (george_slices + bob_extra) / susie_divisor +
     bill_slices + fred_slices + mark_slices + ann_slices + ann_slices * kelly_multiplier))) =
    (small_pizzas_purchased * small_pizza_slices + large_pizzas_purchased * large_pizza_slices -
    (george_slices + (george_slices + bob_extra) + (george_slices + bob_extra) / susie_divisor +
     bill_slices + fred_slices + mark_slices + ann_slices + ann_slices * kelly_multiplier)) :=
by
  sorry

end slices_per_person_l376_376662


namespace no_repetition_five_digit_count_l376_376224

theorem no_repetition_five_digit_count (digits : Finset ℕ) (count : Nat) :
  digits = {0, 1, 2, 3, 4, 5} →
  (∀ n ∈ digits, 0 ≤ n ∧ n ≤ 5) →
  (∃ numbers : Finset ℕ, 
    (∀ x ∈ numbers, (x / 100) % 10 ≠ 3 ∧ x % 5 = 0 ∧ x < 100000 ∧ x ≥ 10000) ∧
    (numbers.card = count)) →
  count = 174 :=
by
  sorry

end no_repetition_five_digit_count_l376_376224


namespace percentage_less_than_l376_376252

theorem percentage_less_than (p j t : ℝ) (h1 : j = 0.75 * p) (h2 : j = 0.80 * t) : 
  t = (1 - 0.0625) * p := 
by 
  sorry

end percentage_less_than_l376_376252


namespace find_base_of_log_equation_l376_376726

theorem find_base_of_log_equation :
  ∃ b : ℝ, (∀ x : ℝ, (9 : ℝ)^(x + 5) = (5 : ℝ)^x → x = Real.logb b ((9 : ℝ)^5)) ∧ b = 5 / 9 :=
by
  sorry

end find_base_of_log_equation_l376_376726


namespace percentage_of_books_in_english_l376_376102

noncomputable def percentage_of_english_books (total_books english_books_published_outside english_percentage : ℕ) : ℕ :=
  let E := english_books_published_outside / 0.40
  ((E / total_books) * 100)

theorem percentage_of_books_in_english :
  percentage_of_english_books 2300 736 0.60 ≈ 80 :=
sorry

end percentage_of_books_in_english_l376_376102


namespace base6_addition_is_correct_l376_376306

def base6_to_base10 (n : List ℕ) : ℕ := 
  List.foldl (λ acc x => 6 * acc + x) 0 n

def base10_to_base6 (n : ℕ) : List ℕ :=
  if n == 0 then [0]
  else
    let rec convert (k : ℕ) (acc : List ℕ) : List ℕ :=
      if k == 0 then acc
      else convert (k / 6) ((k % 6) :: acc)
    convert n []

theorem base6_addition_is_correct :
  base10_to_base6 (base6_to_base10 [3, 5, 2, 4] + base6_to_base10 [2, 4, 4, 2]) = [1, 0, 4, 1, 0] :=
by 
  sorry

end base6_addition_is_correct_l376_376306


namespace max_reflections_l376_376634

open Real

def angle_CDA : ℝ := 5
def angle_at_B : ℝ := 85

theorem max_reflections (n : ℕ) : angle_CDA * n ≤ angle_at_B → n ≤ 17 :=
by
  assume h : angle_CDA * n ≤ angle_at_B
  have h1 : n ≤ angle_at_B / angle_CDA := by sorry
  exact le_of_le_div angle_CDA angle_at_B_of_angle_at_C angle_at_B / angle_CDA h

end max_reflections_l376_376634


namespace unique_tangent_point_l376_376867

-- Define the set of points \(\mathcal{G}\)
def G (x y : ℝ) : Prop :=
  0 ≤ y ∧ y ≤ 8 ∧ (x - 3)^2 + 31 = (y - 4)^2 + 8 * real.sqrt(y * (8 - y))

-- Define the unique line \(\ell\) that is tangent to \(\mathcal{G}\) and passes through \((0, 4)\)
def line_tangent (m : ℝ) (x y : ℝ) : Prop :=
  y = -m * x + 4

-- Define the conditions for \((\alpha, \beta)\)
def P :=
  (12 / 5, 8 / 5)

-- Prove that \((\alpha, \beta)\) are the coordinates of the unique tangent point \(P\)
theorem unique_tangent_point (α β : ℝ) :
  (G α β ∧ ∃ m, line_tangent m α β) ∧ (G α β ∧ ∃ m, line_tangent m α β ∧ α = 0 ∧ β = 4) -> (α = 12 / 5 ∧ β = 8 / 5) :=
by
  sorry

end unique_tangent_point_l376_376867


namespace store_A_total_cost_store_B_total_cost_cost_effective_store_l376_376097

open Real

def total_cost_store_A (x : ℝ) : ℝ :=
  110 * x + 210 * (100 - x)

def total_cost_store_B (x : ℝ) : ℝ :=
  120 * x + 202 * (100 - x)

theorem store_A_total_cost (x : ℝ) :
  total_cost_store_A x = -100 * x + 21000 :=
by
  sorry

theorem store_B_total_cost (x : ℝ) :
  total_cost_store_B x = -82 * x + 20200 :=
by
  sorry

theorem cost_effective_store (x : ℝ) (h : x = 60) :
  total_cost_store_A x < total_cost_store_B x :=
by
  rw [h]
  sorry

end store_A_total_cost_store_B_total_cost_cost_effective_store_l376_376097


namespace sum_of_cos_sq_irreducible_fractions_l376_376868

open Nat Real

def is_irreducible_proper_fraction (k d : ℕ) : Prop :=
  gcd k d = 1 ∧ k < d

def all_irreducible_proper_fractions (n : ℕ) : List ℚ :=
  List.filter (λ x, is_irreducible_proper_fraction x.num x.denom) (List.range n).map (λ k, (k : ℚ) / 60)

theorem sum_of_cos_sq_irreducible_fractions :
  let fractions := all_irreducible_proper_fractions 60
  fractions.length = totient 60 →
  (∑ f in fractions, (cos (f * π / 2))^2) = 8 :=
by
  intros fractions h_totient
  sorry

end sum_of_cos_sq_irreducible_fractions_l376_376868


namespace units_digit_five_l376_376141

def is_digit (d : ℕ) : Prop := d < 10 ∧ d > 0

def odd_product_digits (n : ℕ) : Prop :=
  (digits 10 n).Prod % 2 = 1

def valid_units_digit (n : ℕ) : Prop := n % 10 = 5

def valid_number (n : ℕ) : Prop :=
  ∃ l : List ℕ, (∀ d ∈ l, is_digit d) ∧ l.prod = n ∧ n > 10

theorem units_digit_five (n : ℕ) 
  (h1 : valid_number n) 
  (h2 : odd_product_digits n) :
  valid_units_digit n :=
sorry

end units_digit_five_l376_376141


namespace total_sheets_of_paper_l376_376877

theorem total_sheets_of_paper :
  (let 
     num_first_classes := 3,
     num_second_classes := 3,
     students_first_classes := 22,
     students_second_classes := 18,
     sheets_per_student_first := 6,
     sheets_per_student_second := 4,
     total_first_classes := num_first_classes * students_first_classes * sheets_per_student_first,
     total_second_classes := num_second_classes * students_second_classes * sheets_per_student_second
   in
   total_first_classes + total_second_classes = 612) :=
by
  -- proof steps can be filled here
  sorry

end total_sheets_of_paper_l376_376877


namespace how_many_years_older_l376_376623

-- Definitions of the conditions
variables (a b c : ℕ)
def b_is_16 : Prop := b = 16
def b_is_twice_c : Prop := b = 2 * c
def sum_is_42 : Prop := a + b + c = 42

-- Statement of the proof problem
theorem how_many_years_older (h1 : b_is_16 b) (h2 : b_is_twice_c b c) (h3 : sum_is_42 a b c) : a - b = 2 :=
by
  sorry

end how_many_years_older_l376_376623


namespace median_of_consecutive_integers_sum_eq_6_pow_4_l376_376569

theorem median_of_consecutive_integers_sum_eq_6_pow_4 :
  ∀ (s : ℕ) (n : ℕ), s = 36 → ∑ i in finset.range 36, (n + i) = 6^4 → 36 / 2 = 36 :=
by
  sorry

end median_of_consecutive_integers_sum_eq_6_pow_4_l376_376569


namespace variance_of_data_l376_376591

-- Definition of the data set
def data : List ℝ := [6, 7, 7, 8, 7]

-- Definition of the mean
def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

-- Definition of the variance
def variance (l : List ℝ) : ℝ :=
  (1 / l.length) * (l.map (λ x => (x - mean l) ^ 2)).sum

-- Statement of the theorem
theorem variance_of_data : variance data = 2 / 5 :=
  by
    sorry

end variance_of_data_l376_376591


namespace line_equation_proof_l376_376011

-- Define the conditions
def point := (1 : ℝ, 2 : ℝ)
def passes_through (p : ℝ × ℝ) (a b c : ℝ) : Prop := (a * p.1 + b * p.2 + c = 0)

noncomputable def line_equation : Prop :=
  ∃ (a b c : ℝ), 
    a ≠ 0 ∧ passes_through point a b c ∧ 
    (∃ (h_intercept v_intercept : ℝ), 
      h_intercept ≠ 0 ∧ 
      (h_intercept, 0) = (-c/a, 0) ∧ 
      (0, v_intercept) = (0, -c/b) ∧ 
      v_intercept = 2 * h_intercept)

-- The theorem to prove the equivalent equation
theorem line_equation_proof : line_equation ∧
  (∀ (b₁ b₂ c₁ c₂ : ℝ), passes_through point b₁ (-1 : ℝ) 0 ∨ passes_through point b₂ (1 : ℝ) (-4 : ℝ)) :=
by
  sorry

end line_equation_proof_l376_376011


namespace missing_number_geometric_sequence_l376_376194

theorem missing_number_geometric_sequence : 
  ∃ (x : ℤ), (x = 162) ∧ 
  (x = 54 * 3 ∧ 
  486 = x * 3 ∧ 
  ∀ a b : ℤ, (b = 2 * 3) ∧ 
              (a = 2 * 3) ∧ 
              (18 = b * 3) ∧ 
              (54 = 18 * 3) ∧ 
              (54 * 3 = x)) := 
by sorry

end missing_number_geometric_sequence_l376_376194


namespace marty_combinations_l376_376496

theorem marty_combinations : 
  let C := 5
  let P := 4
  C * P = 20 :=
by
  sorry

end marty_combinations_l376_376496


namespace odd_integers_count_between_13_div_3_and_43_div_2_l376_376057

theorem odd_integers_count_between_13_div_3_and_43_div_2 : 
  let count := ((5: ℤ) :: [7, 9, 11, 13, 15, 17, 19, 21]).length 
  in count = 9 :=
by
  sorry

end odd_integers_count_between_13_div_3_and_43_div_2_l376_376057


namespace part1_line_intercepts_chord_2_eq_x0_or_y_eq_34x_plus_2_part2_ratio_of_arcs_3_to_1_eq_y_eq_13x_plus_2_or_y_eq_minus3x_plus_2_l376_376742

-- Given definitions and conditions
def O (x y : ℝ) := x^2 + y^2 - 4*x - 2*y = 0
def P : (ℝ × ℝ) := (0, 2)

-- Length of chord intercepted is 2
def length_of_chord (l : ℝ × ℝ → Prop) : Prop :=
  ∃ p1 p2 : ℝ × ℝ, O (p1.fst) (p1.snd) ∧ O (p2.fst) (p2.snd) ∧
  p1 ≠ p2 ∧
  dist p1 p2 = 2 ∧
  l p1 ∧ l p2

-- Ratio of arc lengths intercepted is 3:1
def ratio_of_arc_lengths (l : ℝ × ℝ → Prop) : Prop :=
  ∃ c1 c2 : ℝ, c1 ≠ c2 ∧
  c1 + c2 = 2 * real.pi ∧
  ((c1 = (3/4) * 2 * real.pi ∧ c2 = (1/4) * 2 * real.pi) ∨ 
   (c1 = (1/4) * 2 * real.pi ∧ c2 = (3/4) * 2 * real.pi))

-- Statement for Part (1)
theorem part1_line_intercepts_chord_2_eq_x0_or_y_eq_34x_plus_2 (l : ℝ × ℝ → Prop) :
  (∀ p, l p → O p.fst p.snd) → (P ∈ l) → (length_of_chord l) →
  (l = (λ p, p.1 = 0) ∨ l = (λ p, p.2 = 2 + 3/4 * p.1)) :=
sorry

-- Statement for Part (2)
theorem part2_ratio_of_arcs_3_to_1_eq_y_eq_13x_plus_2_or_y_eq_minus3x_plus_2 (l : ℝ × ℝ → Prop) :
  (∀ p, l p → O p.fst p.snd) → (P ∈ l) → (ratio_of_arc_lengths l) →
  (l = (λ p, p.2 = 2 + 1/3 * p.1) ∨ l = (λ p, p.2 = 2 - 3 * p.1)) :=
sorry

end part1_line_intercepts_chord_2_eq_x0_or_y_eq_34x_plus_2_part2_ratio_of_arcs_3_to_1_eq_y_eq_13x_plus_2_or_y_eq_minus3x_plus_2_l376_376742


namespace magnitude_unique_for_quadratic_l376_376328

theorem magnitude_unique_for_quadratic (z : ℂ) :
  (3 * z ^ 2 - 18 * z + 55 = 0) → (∃! m : ℝ, m = complex.abs z) :=
by
  sorry

end magnitude_unique_for_quadratic_l376_376328


namespace parabola_points_relationship_l376_376397

theorem parabola_points_relationship (c y1 y2 y3 : ℝ)
  (h1 : y1 = -0^2 + 2 * 0 + c)
  (h2 : y2 = -1^2 + 2 * 1 + c)
  (h3 : y3 = -3^2 + 2 * 3 + c) :
  y2 > y1 ∧ y1 > y3 := by
  sorry

end parabola_points_relationship_l376_376397


namespace discount_percentage_calculation_l376_376494

noncomputable def cost_price : ℝ := 540
noncomputable def marked_price : ℝ := cost_price + 0.15 * cost_price
noncomputable def selling_price : ℝ := 457
noncomputable def discount : ℝ := marked_price - selling_price
noncomputable def discount_percentage : ℝ := (discount / marked_price) * 100

theorem discount_percentage_calculation :
  discount_percentage ≈ 26.41 := 
sorry

end discount_percentage_calculation_l376_376494


namespace min_abs_E_value_l376_376233

theorem min_abs_E_value (x E : ℝ) (h : |x - 4| + |E| + |x - 5| = 10) : |E| = 9 :=
sorry

end min_abs_E_value_l376_376233


namespace circumscribed_circle_radius_l376_376459

theorem circumscribed_circle_radius
  (A B C : Point)
  (BC : ℝ)
  (BL : Segment)
  (r : ℝ)
  (hBC : BC = 28)
  (hRatio : ∃ (I : Point), ∃ p q : ℕ, p/q = 4/3 ∧ BL.split_at I = (p, q))
  (hInradius : r = 12) :
  ∃ R : ℝ, R = 50 := 
sorry

end circumscribed_circle_radius_l376_376459


namespace axis_of_symmetry_parabola_l376_376184

theorem axis_of_symmetry_parabola : 
  (∃ a b c : ℝ, ∀ x : ℝ, (y = x^2 + 4 * x - 5) ∧ (a = 1) ∧ (b = 4) → ( x = -b / (2 * a) ) → ( x = -2 ) ) :=
by
  sorry

end axis_of_symmetry_parabola_l376_376184


namespace sum_of_squares_of_perpendiculars_to_bisectors_l376_376733

noncomputable theory

open EuclideanGeometry ProbabilityTheory

theorem sum_of_squares_of_perpendiculars_to_bisectors (A B C : Point) 
  (a b c : ℝ) (hA : ‖A - B‖ = a) (hB : ‖B - C‖ = b) (hC : ‖C - A‖ = c) :
  let p1 := foot (interior_angle_bisector B C A) A,
      p2 := foot (exterior_angle_bisector B C A) A,
      q1 := foot (interior_angle_bisector A C B) B,
      q2 := foot (exterior_angle_bisector A C B) B,
      r1 := foot (interior_angle_bisector A B C) C,
      r2 := foot (exterior_angle_bisector A B C) C,
      sumsquares := norm_squared (A - p1) + norm_squared (A - p2) +
                    norm_squared (B - q1) + norm_squared (B - q2) +
                    norm_squared (C - r1) + norm_squared (C - r2) in
  sumsquares = 2 * (a^2 + b^2 + c^2) :=
sorry

end sum_of_squares_of_perpendiculars_to_bisectors_l376_376733


namespace root_of_function_l376_376762

noncomputable def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

theorem root_of_function (f : ℝ → ℝ) (x₀ : ℝ) (h₀ : odd_function f) (h₁ : f (x₀) = Real.exp (x₀)) :
  (f (-x₀) * Real.exp (-x₀) + 1 = 0) :=
by
  sorry

end root_of_function_l376_376762


namespace find_x_l376_376919

theorem find_x {x : ℕ} :
  (2 * x > 70) ∨ (x > 100) ∨ (3 * x > 25) ∨ (x ≥ 10) ∨ (x > 5) :=
by
  let statements := [2 * x > 70, x > 100, 3 * x > 25, x ≥ 10, x > 5]
  have three_true : (2 * 36 > 70 ∧ 3 * 36 > 25 ∧ 36 ≥ 10) :=
    by
    decide
  have two_false : (¬(36 > 100) ∧ ¬(36 > 5)) :=
    by
    decide
  use
    three_true
    two_false
  sorry

end find_x_l376_376919


namespace libby_igloo_bricks_l376_376493

variable (rows_total : ℕ) (bricks_per_row_bottom : ℕ) (bricks_per_row_top : ℕ)

theorem libby_igloo_bricks (h1 : rows_total = 10) (h2 : bricks_per_row_bottom = 12) (h3 : bricks_per_row_top = 8) : 
  let rows_bottom := rows_total / 2 in
  let rows_top := rows_total / 2 in
  let total_bottom_bricks := rows_bottom * bricks_per_row_bottom in
  let total_top_bricks := rows_top * bricks_per_row_top in
  let total_bricks := total_bottom_bricks + total_top_bricks in
  total_bricks = 100 := 
by
  sorry

end libby_igloo_bricks_l376_376493


namespace shaded_area_equals_l376_376716

noncomputable def area_shaded_figure (R : ℝ) : ℝ :=
  let α := (60 : ℝ) * (Real.pi / 180)
  (2 * Real.pi * R^2) / 3

theorem shaded_area_equals : ∀ R : ℝ, area_shaded_figure R = (2 * Real.pi * R^2) / 3 := sorry

end shaded_area_equals_l376_376716


namespace remainder_of_127_div_25_is_2_l376_376278

theorem remainder_of_127_div_25_is_2 : ∃ r, 127 = 25 * 5 + r ∧ r = 2 := by
  have h1 : 127 = 25 * 5 + (127 - 25 * 5) := by rw [mul_comm 25 5, mul_comm 5 25]
  have h2 : 127 - 25 * 5 = 2 := by norm_num
  exact ⟨127 - 25 * 5, h1, h2⟩

end remainder_of_127_div_25_is_2_l376_376278


namespace divide_not_lose_root_l376_376957

theorem divide_not_lose_root (x : ℂ) (h : x^4 + x^3 + x^2 + x + 1 = 0) : x ≠ 0 :=
by
  intro hx
  simp [hx] at h
  exact h

end divide_not_lose_root_l376_376957


namespace maximum_area_of_triangle_l376_376833

theorem maximum_area_of_triangle 
  (a b c : ℝ) 
  (C : ℝ)
  (htriangle : 0 < C ∧ C < π)
  (hSides : c = 4)
  (hEquation : (a + b - c) * (a + b + c) = 3 * a * b) :
  1 / 2 * a * b * Real.sin(C) ≤ 4 * Real.sqrt 3 := 
sorry

end maximum_area_of_triangle_l376_376833


namespace sample_size_is_150_l376_376876

-- Define the conditions
def total_parents : ℕ := 823
def sampled_parents : ℕ := 150
def negative_attitude_parents : ℕ := 136

-- State the theorem
theorem sample_size_is_150 : sampled_parents = 150 := 
by
  sorry

end sample_size_is_150_l376_376876


namespace milestones_with_two_diff_digits_l376_376258

-- Definitions and Conditions
def distance_A_B : ℕ := 999
def num_milestones : ℕ := 1000
def milestone_pair (d : ℕ) : (ℕ × ℕ) := (d, distance_A_B - d)
def has_two_diff_digits (n : ℕ) : Prop :=
  let digits := (n / 100, (n % 100) / 10, n % 10)
  (digits.fst ≠ digits.snd ∨ digits.snd ≠ digits.thrd ∨ digits.fst ≠ digits.thrd) ∧
  (digits.fst = digits.snd ∨ digits.fst = digits.thrd ∨ digits.snd = digits.thrd)

-- Main Theorem
theorem milestones_with_two_diff_digits : {d : ℕ | d < num_milestones ∧ has_two_diff_digits d}.card = 40 := by
  sorry

end milestones_with_two_diff_digits_l376_376258


namespace final_number_at_least_one_over_n_l376_376881

theorem final_number_at_least_one_over_n (n : ℕ) (h : n > 0) :
  ∀ (initial_numbers : list ℝ), (initial_numbers.length = n ∧ (∀ x ∈ initial_numbers, x = 1)) →
    (∀ final_number, final_number ∈ perform_operations n initial_numbers → final_number ≥ 1 / n) :=
begin
  sorry
end

end final_number_at_least_one_over_n_l376_376881


namespace find_n_l376_376720

theorem find_n (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 10) (h3 : n ≡ -2154 [MOD 7]) : n = 2 :=
sorry

end find_n_l376_376720


namespace bernie_saves_most_using_store_a_l376_376680

def cost_local_store (weeks : ℕ) : ℕ := 6 * weeks

def cost_store_a (weeks : ℕ) : ℕ := 
  let packs := ((2 * weeks + 4) / 5) in -- packs needed to cover chocolates for the weeks
  packs * 10

def cost_store_b (weeks : ℕ) : ℕ := 5 * weeks

def cost_store_c (weeks : ℕ) : ℕ := 
  let packs := ((2 * weeks + 9) / 10) in -- packs needed to cover chocolates for the weeks
  packs * 18

theorem bernie_saves_most_using_store_a : 
  (cost_local_store 13) - (cost_store_a 13) = 28 := 
  by
    sorry

end bernie_saves_most_using_store_a_l376_376680


namespace circleEquation_l376_376189

-- Definition to add the conditions in Lean
def isTangent (center : ℝ × ℝ) (radius : ℝ) (line : ℝ → ℝ) : Prop :=
  let (h, k) := center
  r = (abs (1*h + (-1) + k)) / (Real.sqrt (1 +1))

-- The center of the circle
def center : ℝ × ℝ := (1, 0)

-- The line equation y = x + 1
def line (x : ℝ) : ℝ := x + 1

-- The definition stating that the circle with center (1,0) and the equation holds.
theorem circleEquation : 
  ∃ r : ℝ, isTangent center r line ∧ ((r = Real.sqrt 2) → (∀ x y : ℝ, (x - 1)^2 + y^2 = 2)) :=
by
  sorry

end circleEquation_l376_376189


namespace median_of_36_consecutive_integers_l376_376577

theorem median_of_36_consecutive_integers (sum_of_integers : ℕ) (num_of_integers : ℕ) 
  (h1 : num_of_integers = 36) (h2 : sum_of_integers = 6 ^ 4) : 
  (sum_of_integers / num_of_integers) = 36 := 
by 
  sorry

end median_of_36_consecutive_integers_l376_376577


namespace matrix_hall_property_l376_376463

open Finset

variable {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℕ)

theorem matrix_hall_property (m_ne_n : m ≠ n)
  (h : ∀ (f : Fin m → Fin n) (hf : Function.Injective f), ∃ i, A i (f i) = 0) :
  ∃ S : Finset (Fin m), ∃ T : Finset (Fin n),
    (∀ i ∈ S, ∀ j ∈ T, A i j = 0) ∧ S.card + T.card > n := 
sorry

end matrix_hall_property_l376_376463


namespace sin_eq_sides_eq_n_iff_s_l376_376079

theorem sin_eq_sides_eq_n_iff_s {A B C : Type*} {a b c : ℝ}
  (triangle_ABC : ∀ {A₁ A₂ B₁ B₂ C₁ C₂}, a ≠ b → A = B ∧ B = C → A₁ = B₁ ∧ C₁ = C₂)
  (A B : Angle) :
  (a = b ↔ sin A = sin B) :=
by
  sorry

end sin_eq_sides_eq_n_iff_s_l376_376079


namespace angelaAgeInFiveYears_l376_376299

namespace AgeProblem

variables (A B : ℕ) -- Define Angela's and Beth's current age as natural numbers.

-- Condition 1: Angela is four times as old as Beth.
axiom angelaAge : A = 4 * B

-- Condition 2: Five years ago, the sum of their ages was 45 years.
axiom ageSumFiveYearsAgo : (A - 5) + (B - 5) = 45

-- Theorem: Prove that Angela's age in 5 years will be 49.
theorem angelaAgeInFiveYears : A + 5 = 49 :=
by {
  -- proof goes here
  sorry
}

end AgeProblem

end angelaAgeInFiveYears_l376_376299


namespace checkerboard_matching_number_sum_l376_376982

def checkerboard.sum_matching_numbers : ℕ := 440

theorem checkerboard_matching_number_sum:
  let f (i j : ℕ) := 19 * (i - 1) + j + 2
  let g (i j : ℕ) := 15 * (j - 1) + i + 2
  (∑ i in (Finset.range 16), ∑ j in (Finset.range 20), if f i j = g i j then f i j else 0) = checkerboard.sum_matching_numbers :=
by
  sorry

end checkerboard_matching_number_sum_l376_376982


namespace skew_lines_sufficient_not_necessary_l376_376791

-- Definitions for the conditions
def skew_lines (l1 l2 : Type) : Prop := sorry -- Definition of skew lines
def do_not_intersect (l1 l2 : Type) : Prop := sorry -- Definition of not intersecting

-- The main theorem statement
theorem skew_lines_sufficient_not_necessary (l1 l2 : Type) :
  (skew_lines l1 l2) → (do_not_intersect l1 l2) ∧ ¬ (do_not_intersect l1 l2 → skew_lines l1 l2) :=
by
  sorry

end skew_lines_sufficient_not_necessary_l376_376791


namespace Ariana_running_time_l376_376172

theorem Ariana_running_time
  (time_Sadie : ℝ)
  (speed_Sadie : ℝ)
  (speed_Ariana : ℝ)
  (speed_Sarah : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (distance_Sadie := speed_Sadie * time_Sadie)
  (time_Ariana_Sarah := total_time - time_Sadie)
  (distance_Ariana_Sarah := total_distance - distance_Sadie) :
  (6 * (time_Ariana_Sarah - (11 - 6 * (time_Ariana_Sarah / (speed_Ariana + (4 / speed_Sarah)))))
  = (0.5 : ℝ)) :=
by
  sorry

end Ariana_running_time_l376_376172


namespace difference_of_same_prime_factors_l376_376510

theorem difference_of_same_prime_factors (n : ℕ) :
  ∃ a b : ℕ, a - b = n ∧ (a.primeFactors.card = b.primeFactors.card) :=
by
  sorry

end difference_of_same_prime_factors_l376_376510


namespace n_equals_23_l376_376136

theorem n_equals_23 (k : ℕ) (h_k : k ≥ 6) (n : ℕ) (h_n : n = 2*k - 1) (T : Set (Fin n → Bool)) 
(h_T : ∀ x : Fin n → Bool, x ∈ T) 
(d : (Fin n → Bool) → (Fin n → Bool) → ℕ) 
(h_d : ∀ x y : Fin n → Bool, d x y = ∑ i in Finset.univ, abs (x i - y i)) 
(S : Set (Fin n → Bool)) (h_S_size : S.toFinset.card = 2^k) 
(h_S : ∀ x : Fin n → Bool, ∃! y ∈ S, d x y ≤ 3) : 
n = 23 := 
sorry

end n_equals_23_l376_376136


namespace polyhedron_euler_formula_polyhedron_has_30_faces_polyhedron_edges_polyhedron_value_is_270_l376_376332

noncomputable def polyhedron_value (H T V : ℕ) (t h : ℕ) (E F : ℕ) :=
  100 * H + 10 * T + V

theorem polyhedron_euler_formula (V E F : ℕ) : V - E + F = 2 :=
sorry

theorem polyhedron_has_30_faces : 30 = 8 + 22 :=
sorry

theorem polyhedron_edges (E t h : ℕ): E = (3 * t + 6 * h) / 2 :=
sorry

theorem polyhedron_value_is_270 
  (V E F H T : ℕ)
  (h : ℕ := 22) 
  (t : ℕ := 8)
  (E_eq : E = 78)
  (F_eq : F = 30)
  (H_eq : H = 2)
  (T_eq : T = 2)
  (V_eq : V = 50) :
  polyhedron_value H T V t h E F = 270 :=
by {
  rw [polyhedron_value, H_eq, T_eq, V_eq],
  norm_num,
  solve_by_elim,
}

end polyhedron_euler_formula_polyhedron_has_30_faces_polyhedron_edges_polyhedron_value_is_270_l376_376332


namespace sum_of_tangential_circle_radii_seq_l376_376859

theorem sum_of_tangential_circle_radii_seq 
(A B C : Point) 
(ω : ℕ → Circle) 
(r : ℕ → ℝ) 
(h_angle : angle A C B = 120)
(h_tangent_CA_CB : ∀ n, tangent (ω n) CA ∧ tangent (ω n) CB)
(h_tangent_seq : ∀ n, tangent (ω (n+1)) (ω n))
(h_initial_radius : r 0 = 3)
(h_radius_seq : ∀ n, r (n+1) = (7 - 4 * sqrt 3) * r n) :
  ∑' n, r n = (3 / 2) + sqrt 3 := sorry

end sum_of_tangential_circle_radii_seq_l376_376859


namespace felix_jump_longer_than_betty_step_l376_376681

noncomputable def betty_steps_per_gap : ℕ := 36
noncomputable def felix_jumps_per_gap : ℕ := 9
noncomputable def total_distance : ℕ := 7920

theorem felix_jump_longer_than_betty_step :
  let num_gaps := 50
  let total_betty_steps := betty_steps_per_gap * num_gaps
  let total_felix_jumps := felix_jumps_per_gap * num_gaps
  let betty_step_length := total_distance / total_betty_steps
  let felix_jump_length := total_distance / total_felix_jumps
  felix_jump_length - betty_step_length = 13.2 :=
by
  sorry

end felix_jump_longer_than_betty_step_l376_376681


namespace min_sum_of_a_and_b_l376_376602

theorem min_sum_of_a_and_b (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > 4 * b) : a + b ≥ 6 :=
by
  sorry

end min_sum_of_a_and_b_l376_376602


namespace number_of_students_l376_376113

theorem number_of_students (T : ℕ) (n : ℕ) (h1 : (T + 20) / n = T / n + 1 / 2) : n = 40 :=
  sorry

end number_of_students_l376_376113


namespace find_t_on_line_l376_376783

theorem find_t_on_line : 
  (∀ x1 y1 x2 y2 x3 y3, x1 ≠ x2 → x2 ≠ x3 → x3 ≠ x1 →
  let slope := (y2 - y1) / (x2 - x1) in
  let slope' := (y3 - y2) / (x3 - x2) in
  slope = slope' →
  ∀ t, (y1 - t) = slope * (x1 - 20) → t = 42) :=
sorry

end find_t_on_line_l376_376783


namespace initially_calculated_average_height_l376_376180

theorem initially_calculated_average_height
    (A : ℕ)
    (initial_total_height : ℕ)
    (real_total_height : ℕ)
    (height_error : ℕ := 60)
    (num_boys : ℕ := 35)
    (actual_average_height : ℕ := 183)
    (initial_total_height_eq : initial_total_height = num_boys * A)
    (real_total_height_eq : real_total_height = num_boys * actual_average_height)
    (height_discrepancy : initial_total_height = real_total_height + height_error) :
    A = 181 :=
by
  sorry

end initially_calculated_average_height_l376_376180


namespace sum_of_money_l376_376981

theorem sum_of_money (A B C : ℝ) (hB : B = 0.65 * A) (hC : C = 0.40 * A) (hC_val : C = 56) :
  A + B + C = 287 :=
by {
  sorry
}

end sum_of_money_l376_376981


namespace national_education_fund_expenditure_l376_376178

theorem national_education_fund_expenditure (gdp_2012 : ℝ) (h : gdp_2012 = 43.5 * 10^12) : 
  (0.04 * gdp_2012) = 1.74 * 10^13 := 
by sorry

end national_education_fund_expenditure_l376_376178


namespace draw_three_balls_l376_376988

theorem draw_three_balls (n m : ℕ) (h1 : n = 6) (h2 : m = 2) :
  let total_balls := n + m
  in (total_balls = 8) → (finset.card (finset.choose (total_balls) 3) = nat.choose 8 3) :=
by
  intros,
  sorry

end draw_three_balls_l376_376988


namespace inverse_of_f_is_g_l376_376702

def f (x : ℝ) : ℝ := 3 - 4 * x + x^2

def is_inverse (f g : ℝ → ℝ) := ∀ x, f (g x) = x

theorem inverse_of_f_is_g : is_inverse f (fun x => 2 - sqrt (1 + x)) :=
by
  sorry

end inverse_of_f_is_g_l376_376702


namespace evaluate_expression_correct_l376_376709

noncomputable def evaluate_expression : ℝ :=
  (0.82 ^ 3 - 0.1 ^ 3) / (0.82 ^ 2 + 0.082 + 0.1 ^ 2 + Real.log 0.82 / Real.log 5 - Real.sin (Real.pi / 4)) ^ 2

theorem evaluate_expression_correct : 
  | evaluate_expression - 126.229 | < 1e-3 := 
by 
  sorry

end evaluate_expression_correct_l376_376709


namespace cone_volume_proof_l376_376647

-- Definitions based on given conditions
def radius_of_circle : ℝ := 5
def arc_length (r : ℝ) : ℝ := r * π

-- Base radius of the cone (using arc length of half-sector)
def base_radius (r : ℝ) : ℝ := (arc_length r) / (2 * π)

-- Height of the cone calculated using Pythagorean theorem
def cone_height (r : ℝ) (base_r : ℝ) : ℝ := 
  real.sqrt (r^2 - base_r^2)

-- Volume of the cone
def cone_volume (base_r h : ℝ) : ℝ := 
  (1 / 3) * π * (base_r^2) * h

-- Proof statement: Given radius of circle 5 inches, the volume of the cone 
-- formed by rolling up the half-sector is \( \frac{125\sqrt{3}\pi}{24} \)
theorem cone_volume_proof : 
  cone_volume (base_radius radius_of_circle) (cone_height radius_of_circle (base_radius radius_of_circle)) 
  = (125 * real.sqrt 3 * π) / 24 :=
by
  -- This is just a placeholder. The actual proof steps are omitted according to instructions.
  sorry

end cone_volume_proof_l376_376647


namespace youseff_commute_l376_376628

-- Define distance in blocks and time per block.
def blocks : ℝ := 21
def walk_time_per_block : ℝ := 1
def bike_time_per_block : ℝ := 20 / 60

-- Define the main statement we want to prove.
theorem youseff_commute (x : ℝ) (walk_time : ℝ) (bike_time : ℝ) :
  (walk_time = x * walk_time_per_block) ∧
  (bike_time = x * bike_time_per_block) ∧
  (walk_time = bike_time + 14) → 
  x = 21 := 
by
  intro h
  sorry

end youseff_commute_l376_376628


namespace roots_of_unity_in_polynomial_l376_376703

noncomputable def is_root_of_unity (z : ℂ) (n : ℕ) : Prop :=
z ^ n = 1

noncomputable def is_root_of_polynomial (z : ℂ) (a b : ℤ) : Prop :=
z ^ 3 + (a : ℂ) * z ^ 2 + (b : ℂ) * z + 1 = 0

theorem roots_of_unity_in_polynomial (a b : ℤ) :
  {z : ℂ | is_root_of_unity z 3 ∧ is_root_of_polynomial z a b}.card = 8 := 
sorry

end roots_of_unity_in_polynomial_l376_376703


namespace profit_function_expression_l376_376968

def dailySalesVolume (x : ℝ) : ℝ := 300 + 3 * (99 - x)

def profitPerItem (x : ℝ) : ℝ := x - 50

def dailyProfit (x : ℝ) : ℝ := (x - 50) * (300 + 3 * (99 - x))

theorem profit_function_expression (x : ℝ) :
  dailyProfit x = (x - 50) * dailySalesVolume x :=
by sorry

end profit_function_expression_l376_376968


namespace functional_equation_solution_l376_376325

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ a b c : ℝ, (f a - f b) * (f b - f c) * (f c - f a) = f (a * b^2 + b * c^2 + c * a^2) - f (a^2 * b + b^2 * c + c^2 * a)) →
  ∃ c : ℝ, f = (λ x, c) ∨
           f = (λ x, x + c) ∨
           f = (λ x, -x + c) ∨
           f = (λ x, x^3 + c) ∨
           f = (λ x, -x^3 + c) := sorry

end functional_equation_solution_l376_376325


namespace ratio_pat_to_mark_l376_376506

theorem ratio_pat_to_mark (K P M : ℕ) 
  (h1 : P + K + M = 117) 
  (h2 : P = 2 * K) 
  (h3 : M = K + 65) : 
  P / Nat.gcd P M = 1 ∧ M / Nat.gcd P M = 3 := 
by
  sorry

end ratio_pat_to_mark_l376_376506


namespace square_area_l376_376155

theorem square_area (E F A B C D : Type) (BE EF FD : ℝ) (h1 : BE = 40) (h2 : EF = 20) (h3 : FD = 40) :
  let x := BE + EF + FD in
  x = 100 →
  (x ^ 2) = 10000 :=
by
  intro x hx
  have hx : x = 100 := by rw [h1, h2, h3]
  rw hx
  norm_num
  sorry

end square_area_l376_376155


namespace books_per_author_l376_376838

theorem books_per_author (total_books : ℕ) (num_authors : ℕ) (books_per_author : ℕ) : 
total_books = 198 ∧ num_authors = 6 → books_per_author = 33 :=
begin
  sorry
end

end books_per_author_l376_376838


namespace probability_of_odd_sum_l376_376944

theorem probability_of_odd_sum (P : ℝ → Prop) 
    (P_even_sum : ℝ)
    (P_odd_sum : ℝ)
    (h1 : P_even_sum = 2 * P_odd_sum) 
    (h2 : P_even_sum + P_odd_sum = 1) :
    P_odd_sum = 4/9 := 
sorry

end probability_of_odd_sum_l376_376944


namespace sequence_subsequence_l376_376676

theorem sequence_subsequence :
  ∃ (a : Fin 101 → ℕ), 
  (∀ i, a i = i + 1) ∧ 
  ∃ (b : Fin 11 → ℕ), 
  (b 0 < b 1 ∧ b 1 < b 2 ∧ b 2 < b 3 ∧ b 3 < b 4 ∧ b 4 < b 5 ∧ 
  b 5 < b 6 ∧ b 6 < b 7 ∧ b 7 < b 8 ∧ b 8 < b 9 ∧ b 9 < b 10) ∨ 
  (b 0 > b 1 ∧ b 1 > b 2 ∧ b 2 > b 3 ∧ b 3 > b 4 ∧ b 4 > b 5 ∧ 
  b 5 > b 6 ∧ b 6 > b 7 ∧ b 7 > b 8 ∧ b 8 > b 9 ∧ b 9 > b 10) :=
by {
  sorry
}

end sequence_subsequence_l376_376676


namespace incorrect_statements_count_l376_376486

noncomputable def A := {x : ℝ | ∃ (y : ℝ), y = x^2 - 4}
noncomputable def B := {y : ℝ | ∃ (x : ℝ), y = x^2 - 4}
noncomputable def C := {(x, y) : ℝ × ℝ | y = x^2 - 4}

theorem incorrect_statements_count :
  ({s  | s ∈ [A ∩ C = ∅, A = C, A = B, B = C] ∧ s = false}.card = 3) :=
sorry

end incorrect_statements_count_l376_376486


namespace philanthropist_total_withdrawal_l376_376280

-- The total amount withdrawn given the initial conditions.
theorem philanthropist_total_withdrawal (a r : ℝ) (h_r_pos : r > 0) :
  let amount_withdrawn := a / r * ((1 + r)^8 - (1 + r))
  in amount_withdrawn = a / r * ((1 + r)^8 - (1 + r)) :=
sorry

end philanthropist_total_withdrawal_l376_376280


namespace cash_before_brokerage_correct_l376_376911

variable (realized_cash : ℝ) (brokerage_rate : ℝ)

def cash_before_brokerage (realized_cash brokerage_rate : ℝ) : ℝ :=
  realized_cash / (1 - brokerage_rate)

theorem cash_before_brokerage_correct :
  cash_before_brokerage 109.25 0.0025 = 109.56 :=
by 
  sorry

end cash_before_brokerage_correct_l376_376911


namespace equilateral_triangle_side_length_l376_376296

theorem equilateral_triangle_side_length {DEF : Type*}
  [decidable_eq DEF]
  (Q : DEF -> ℝ) -- Q: function encoding point distance relations
  (DQ EQ FQ : ℝ)
  (h1 : DQ = 2)
  (h2 : EQ = real.sqrt 5)
  (h3 : FQ = 3)
  (h4 : ∀ (P Q R : DEF), equilateral_triangle P Q R) -- encodes that DEF is an equilateral triangle
  (h5 : hexagon_area (reflect Q DEF))     -- Q reflections form a hexagon
  (h6 : triangle_areas_condition Q DEF)   -- area condition for triangles 
  : 
  side_length DEF = 2 * real.sqrt 6 := 
sorry

end equilateral_triangle_side_length_l376_376296


namespace nancy_antacids_per_day_l376_376498

theorem nancy_antacids_per_day 
  (antacids_indian : ℕ)
  (antacids_mexican : ℕ)
  (days_indian_per_week : ℕ)
  (days_mexican_per_week : ℕ)
  (antacids_per_month : ℕ)
  (days_per_month : ℕ) :
  let antacids_other_per_month := antacids_per_month - ((antacids_indian * days_indian_per_week + antacids_mexican * days_mexican_per_week) * (days_per_month / 7)) in
  let days_other_per_month := days_per_month - (days_indian_per_week + days_mexican_per_week) * (days_per_month / 7) in
  antacids_other_per_month / days_other_per_month = 0.8 :=
by
  -- Given conditions
  let antacids_indian := 3 in
  let antacids_mexican := 2 in
  let days_indian_per_week := 3 in
  let days_mexican_per_week := 2 in
  let antacids_per_month := 60 in
  let days_per_month := 30 in
  -- Definitions based on conditions
  let antacids_other_per_month := antacids_per_month - ((antacids_indian * days_indian_per_week + antacids_mexican * days_mexican_per_week) * (days_per_month / 7)) in
  let days_other_per_month := days_per_month - (days_indian_per_week + days_mexican_per_week) * (days_per_month / 7) in
  -- Proof we need to show:
  show antacids_other_per_month / days_other_per_month = 0.8 from sorry

end nancy_antacids_per_day_l376_376498


namespace integer_pairs_count_l376_376045

theorem integer_pairs_count : 
  ∃ (m n : ℕ), (m > 0 ∧ n > 0) ∧ ([1/m, n] ∋ (λ x, |log 2 x|)) ∧ ([0, 2] ∋ (λ x, |log 2 x|)) ∧ 
  (∃ p : ℕ, p = 7) :=
begin
    sorry,
end

end integer_pairs_count_l376_376045


namespace incenter_eq_distance_l376_376217

noncomputable def midpoint (x y : ℝ) : ℝ := (x + y) / 2

theorem incenter_eq_distance :
  ∀ (ω_1 ω_2 : Circle) (A C : ω_1) (B D : ω_2)
    (AB CD : Line) (M X Y I : Point),
  is_tangent AB ω_1 ∧ is_tangent AB ω_2 ∧
  is_tangent CD ω_1 ∧ is_tangent CD ω_2 ∧
  midpoint (AB.A) (AB.B) = M ∧
  tangent_through_point ω_1 M ≠ AB ∧
  tangent_through_point ω_2 M ≠ AB ∧
  intersect (tangent_through_point ω_1 M) CD = X ∧
  intersect (tangent_through_point ω_2 M) CD = Y ∧
  incenter (triangle.mk M X Y) = I →
  dist I C = dist I D :=
by
  sorry

end incenter_eq_distance_l376_376217


namespace friends_count_l376_376842

def bananas_total : ℝ := 63
def bananas_per_friend : ℝ := 21.0

theorem friends_count : bananas_total / bananas_per_friend = 3 := sorry

end friends_count_l376_376842


namespace arithmetic_sequence_geometric_sequence_l376_376260

-- Arithmetic sequence proof
theorem arithmetic_sequence (d n : ℕ) (a_n a_1 : ℤ) (s_n : ℤ) :
  d = 2 → n = 15 → a_n = -10 →
  a_1 = -38 ∧ s_n = -360 :=
sorry

-- Geometric sequence proof
theorem geometric_sequence (a_1 a_4 q s_3 : ℤ) :
  a_1 = -1 → a_4 = 64 →
  q = -4 ∧ s_3 = -13 :=
sorry

end arithmetic_sequence_geometric_sequence_l376_376260


namespace sum_reciprocals_lt_seven_sixths_l376_376398

open scoped BigOperators

variable (a : Fin 1009 → ℕ)
variable (a_sorted : StrictMono a)
variable (a_bounds : ∀ i, 1 < a i ∧ a i < 2018)
variable (lcm_condition : ∀ i j, i ≠ j → Nat.lcm (a i) (a j) > 2018)

theorem sum_reciprocals_lt_seven_sixths :
  ∑ i in Finset.univ, (1 : ℚ) / (a i) < 7 / 6 :=
sorry

end sum_reciprocals_lt_seven_sixths_l376_376398


namespace max_value_of_f_for_x_lt_zero_l376_376193

def f (x : ℝ) : ℝ := (x^2 - 2 * x + 9) / x

theorem max_value_of_f_for_x_lt_zero : ∀ x : ℝ, x < 0 → f x ≤ -8 :=
sorry

end max_value_of_f_for_x_lt_zero_l376_376193


namespace minimum_value_of_x_plus_y_l376_376763

theorem minimum_value_of_x_plus_y
  (x y : ℝ)
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : (1 / y) + (4 / x) = 1) : 
  x + y = 9 :=
sorry

end minimum_value_of_x_plus_y_l376_376763


namespace mean_proportional_l376_376722

theorem mean_proportional (x : ℝ) (h : (72.5:ℝ) = Real.sqrt (x * 81)): x = 64.9 := by
  sorry

end mean_proportional_l376_376722


namespace part_one_part_two_l376_376627

-- Problem setup
def twin_primes (p q : ℕ) : Prop := nat.prime p ∧ nat.prime q ∧ q = p + 2

-- Part (i): Given m = p, prove p = 3
theorem part_one (p q : ℕ) (m n : ℕ) (hpq : twin_primes p q) (hm : m = p) (hn : n ≥ 1) :
  p = 3 :=
sorry

-- Part (ii): Prove there is exactly one solution quadruple (p, q, m, n) 
theorem part_two : ∃! (p q m n : ℕ), twin_primes p q ∧ n! + p * q^2 = (m * p)^2 ∧ 1 ≤ n ∧ 1 ≤ m :=
sorry

end part_one_part_two_l376_376627


namespace remainder_sum_remainders_mod_500_l376_376127

open Nat

/-- Define the set of remainders of 3^n mod 500 for nonnegative integers n -/
def remainders_mod_500 : Set ℕ := {r | ∃ n : ℕ, r = 3^n % 500}

/-- Define the sum of the elements in the set of remainders -/
def S : ℕ := remainders_mod_500.sum (λ x, x)

theorem remainder_sum_remainders_mod_500 (x : ℕ)
  (hx : S % 500 = x) :
  S % 500 = x := by
  sorry

end remainder_sum_remainders_mod_500_l376_376127


namespace prime_product_solution_l376_376846

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_product_solution (p_1 p_2 p_3 p_4 : ℕ) :
  is_prime p_1 ∧ is_prime p_2 ∧ is_prime p_3 ∧ is_prime p_4 ∧ 
  p_1 ≠ p_2 ∧ p_1 ≠ p_3 ∧ p_1 ≠ p_4 ∧ p_2 ≠ p_3 ∧ p_2 ≠ p_4 ∧ p_3 ≠ p_4 ∧
  2 * p_1 + 3 * p_2 + 5 * p_3 + 7 * p_4 = 162 ∧
  11 * p_1 + 7 * p_2 + 5 * p_3 + 4 * p_4 = 162 
  → p_1 * p_2 * p_3 * p_4 = 570 := 
by
  sorry

end prime_product_solution_l376_376846


namespace proof_AD_plus_BD_eq_BC_l376_376092

-- Define the isosceles triangle properties.
variables {A B C D : Type} [EuclideanGeometry]

-- Given 
-- 1. A isosceles triangle ABC with AB = AC,
-- 2. The vertex angle at A equal to 100 degrees,
-- 3. A bisector BD is drawn.
noncomputable def isosceles_triangle_with_bisector (AB AC : ℝ) (angle_A : ℝ) (BD : ℝ) :=
  ∀ (A B C D : Point) (h1 : AB = AC) (h2 : angle A B C = 100) (h3 : isAngleBisector B C D),
  AD + BD = BC

-- Here provides a sorry stub as the proof is not required.
theorem proof_AD_plus_BD_eq_BC : isosceles_triangle_with_bisector AB AC (100 : ℝ) BD :=
  sorry

end proof_AD_plus_BD_eq_BC_l376_376092


namespace negation_proof_l376_376550

open Real

-- Define the condition of the original proposition
def exists_x0_prop : Prop :=
  ∃ x0 : ℝ, 0 < x0 ∧ x0 < real.exp 1 ∧ ln x0 > 3 - x0

-- Define the negation of the original proposition
def neg_exists_x0_prop : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < real.exp 1 → ln x ≤ 3 - x

-- The goal is to prove that neg_exists_x0_prop is the negation of exists_x0_prop
theorem negation_proof : neg_exists_x0_prop ↔ ¬ exists_x0_prop := by
  sorry

end negation_proof_l376_376550


namespace maximize_profit_l376_376665

noncomputable def cost_per_item : ℝ := 10
noncomputable def initial_price : ℝ := 18
noncomputable def initial_sales : ℝ := 60

def profit_function (x : ℝ) : ℝ :=
  if x ≤ 18 then 
    (x - cost_per_item) * (initial_sales + (18 - x) * 10)
  else 
    (x - cost_per_item) * (initial_sales - (x - 18) * 5)

theorem maximize_profit : 
  let max_price := 20
  let max_profit := 500
  profit_function max_price = max_profit := 
by
  -- The main body of the proof will confirm the equivalence of the computed profit
  sorry

end maximize_profit_l376_376665


namespace complement_A_intersect_B_eq_l376_376870

def setA : Set ℝ := { x : ℝ | |x - 2| ≤ 2 }

def setB : Set ℝ := { y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2 }

def A_intersect_B := setA ∩ setB

def complement (A : Set ℝ) : Set ℝ := { x : ℝ | x ∉ A }

theorem complement_A_intersect_B_eq {A : Set ℝ} {B : Set ℝ} 
  (hA : A = { x : ℝ | |x - 2| ≤ 2 })
  (hB : B = { y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2 }) :
  complement (A ∩ B) = { x : ℝ | x ≠ 0 } :=
by
  sorry

end complement_A_intersect_B_eq_l376_376870


namespace colonies_ratio_rounded_l376_376158

def ratio_rounded_to_nearest_tenth (num den : ℕ) : ℝ :=
  let ratio : ℝ := num / den;
  Float.round (ratio * 10) / 10

theorem colonies_ratio_rounded :
  ratio_rounded_to_nearest_tenth 10 15 = 0.7 :=
by
  sorry

end colonies_ratio_rounded_l376_376158


namespace min_tangent_length_l376_376370

noncomputable def circle_symmetric_min_tangent_length (a b : ℝ) (h1 : (x^2 + y^2 + 2*x - 4*y + 3 = 0))
    (h2 : symmetric_line: 2*a*x + b*y + 6 = 0) : ℝ :=
  4

theorem min_tangent_length (a b : ℝ) (h1 : (x^2 + y^2 + 2*x - 4*y + 3 = 0))
    (h2 : (2*a*x + b*y + 6 = 0)) :
    circle_symmetric_min_tangent_length a b h1 h2 = 4 :=
  sorry

end min_tangent_length_l376_376370


namespace find_m_value_l376_376408

-- Define given vectors and condition
def veca : ℝ × ℝ := (1, -2)
def vecb (m : ℝ) : ℝ × ℝ := (2, m)
def condition (m : ℝ) : Prop := vecb m = (2 * (1 : ℝ), 2 * (-2 : ℝ))

-- The theorem to prove
theorem find_m_value : ∃ m : ℝ, condition m ∧ m = -4 :=
by
  use -4
  rw condition
  simp
  sorry

end find_m_value_l376_376408


namespace max_points_with_natural_number_angles_l376_376204

theorem max_points_with_natural_number_angles:
  ∃ N : ℕ, (∀ (points : Finset (ℝ × ℝ)), points.card = N → 
    (∀ (A B C : (ℝ × ℝ)), A ∈ points → B ∈ points → C ∈ points → 
        ∃ α β γ : ℕ, α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0)
    ) ∧ N = 180 :=
begin
  sorry
end

end max_points_with_natural_number_angles_l376_376204


namespace find_larger_number_l376_376157

theorem find_larger_number (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : y = 33 :=
by
  sorry

end find_larger_number_l376_376157


namespace square_triangle_same_area_l376_376100

theorem square_triangle_same_area (perimeter_square height_triangle : ℤ) (same_area : ℚ) 
  (h_perimeter_square : perimeter_square = 64) 
  (h_height_triangle : height_triangle = 64)
  (h_same_area : same_area = 256) :
  ∃ x : ℚ, x = 8 :=
by
  sorry

end square_triangle_same_area_l376_376100


namespace eq_three_div_x_one_of_eq_l376_376060

theorem eq_three_div_x_one_of_eq (x : ℝ) (hx : 1 - 6 / x + 9 / (x ^ 2) = 0) : (3 / x) = 1 :=
sorry

end eq_three_div_x_one_of_eq_l376_376060


namespace impossible_rearrange_reverse_l376_376207

theorem impossible_rearrange_reverse :
  ∀ (tokens : ℕ → ℕ), 
    (∀ i, (i % 2 = 1 ∧ i < 99 → tokens i = tokens (i + 2)) 
      ∧ (i % 2 = 0 ∧ i < 99 → tokens i = tokens (i + 2))) → ¬(∀ i, tokens i = 100 + 1 - tokens (i - 1)) :=
by
  intros tokens h
  sorry

end impossible_rearrange_reverse_l376_376207


namespace find_a_b_c_l376_376601

noncomputable def m_value (a b c: ℕ) : ℝ := a - b * Real.sqrt c

theorem find_a_b_c (a b c : ℕ) (h1 : a = 120) (h2 : b = 60) (h3 : c = 2)
  (m : ℝ) (h4 : m = m_value a b c) (h5 : ¬ ∃ p, p.prime ∧ p^2 ∣ c):
  a + b + c = 182 := by
  sorry

end find_a_b_c_l376_376601


namespace paco_initial_cookies_l376_376159

-- Define the given conditions
def cookies_given : ℕ := 14
def cookies_eaten : ℕ := 10
def cookies_left : ℕ := 12

-- Proposition to prove: Paco initially had 36 cookies
theorem paco_initial_cookies : (cookies_given + cookies_eaten + cookies_left = 36) :=
by
  sorry

end paco_initial_cookies_l376_376159


namespace shortest_distance_l376_376118

theorem shortest_distance :
  let A := (a : ℝ) (a^2 - 4 * a + 4 : ℝ) in
  let B := (x : ℝ) (2 * x - 3 : ℝ) in
  ∃ a : ℝ, dist_to_parabola_line a = (Real.sqrt 10) / 5 :=
sorry

end shortest_distance_l376_376118


namespace median_of_36_consecutive_integers_l376_376579

theorem median_of_36_consecutive_integers (f : ℕ → ℤ) (h_consecutive : ∀ n : ℕ, f (n + 1) = f n + 1) 
(h_size : ∃ k, f 36 = f 0 + 35) (h_sum : ∑ i in finset.range 36, f i = 6^4) : 
(∃ m, m = f (36 / 2 - 1) ∧ m = 36) :=
by
  sorry

end median_of_36_consecutive_integers_l376_376579


namespace problem_solution_l376_376406

def p : Prop := ∀ x : ℝ, cos x = cos (x + 2 * Real.pi)
def q : Prop := ∀ x : ℝ, (x^3 + sin x) = -(x^3 + sin (-x))

theorem problem_solution (Hp : ¬ p) (Hq : q) : p ∨ q :=
by
  exact Or.inr Hq

end problem_solution_l376_376406


namespace books_per_author_l376_376839

theorem books_per_author (total_books : ℕ) (num_authors : ℕ) (books_per_author : ℕ) : 
total_books = 198 ∧ num_authors = 6 → books_per_author = 33 :=
begin
  sorry
end

end books_per_author_l376_376839


namespace minimize_area_quadrilateral_l376_376834

theorem minimize_area_quadrilateral 
  {O A M N : Point} 
  {phi psi beta : Real}
  (h1 : forms_angles OA phi psi)
  (h2 : phi + psi + beta > pi)
  (h3 : quadrilateral_area_minimal O M A N)
  (h4 : phi > 90 - beta / 2)
  (h5 : psi > 90 - beta / 2) 
  : |MA| = |AN| :=
sorry

end minimize_area_quadrilateral_l376_376834


namespace problem_1_problem_2_l376_376311

theorem problem_1 :
  (0.001)^(-1 / 3) + 27^(2 / 3) + (1 / 4)^(-1 / 2) - (1 / 9)^(-1.5) = -6 :=
by
  sorry

theorem problem_2 :
  (1 / 2) * log 10 25 + log 10 2 - log 10 (sqrt 0.1) - log 2 9 * log 3 2 = -1 / 2 :=
by
  sorry

end problem_1_problem_2_l376_376311


namespace monotonic_intervals_l376_376042

open Set

noncomputable def f (a x : ℝ) : ℝ := - (1 / 3) * a * x^3 + x^2 + 1

theorem monotonic_intervals (a : ℝ) (h : a ≤ 0) :
  (a = 0 → (∀ x : ℝ, (x < 0 → deriv (f a) x < 0) ∧ (0 < x → deriv (f a) x > 0))) ∧
  (a < 0 → (∀ x : ℝ, (x < 2 / a → deriv (f a) x > 0 ∨ deriv (f a) x = 0) ∧ 
                     (2 / a < x → deriv (f a) x < 0 ∨ deriv (f a) x = 0))) :=
by
  sorry

end monotonic_intervals_l376_376042


namespace arithmetic_sequence_of_roots_sums_l376_376134

noncomputable def roots_sum (P : Polynomial ℂ) : ℂ :=
  if P.degree = 0 then 0 else - (P.coeff (P.nat_degree - 1) / P.leading_coeff)

theorem arithmetic_sequence_of_roots_sums
  (P : Polynomial ℂ) (h : P ≠ 0) (n : ℕ) (hn : P.nat_degree = n) :
  ∃ d : ℂ, ∀ k : ℕ, k < n → roots_sum (Polynomial.derivative^[k] P) = roots_sum P + d * k :=
sorry

end arithmetic_sequence_of_roots_sums_l376_376134


namespace find_number_l376_376975

axiom division_condition (x : ℝ) : 25.25 / x = 0.016833333333333332

theorem find_number (x : ℝ) (h : division_condition x) : x = 1500 :=
by
  -- proof steps go here
  sorry

end find_number_l376_376975


namespace intersection_of_sets_l376_376401

def setA : Set ℝ := { x : ℝ | 2^(x - 1) > 1 }
def setB : Set ℝ := { x : ℝ | x * (x - 2) < 0 }

theorem intersection_of_sets : (setA ∩ setB) = { x : ℝ | 1 < x ∧ x < 2 } :=
by 
  sorry

end intersection_of_sets_l376_376401


namespace max_sides_of_convex_polygon_l376_376335

theorem max_sides_of_convex_polygon (n : ℕ) 
  (h_convex : n ≥ 3) 
  (h_angles: ∀ (a : Fin 4), (100 : ℝ) ≤ a.val) 
  : n ≤ 8 :=
sorry

end max_sides_of_convex_polygon_l376_376335


namespace largest_valid_n_l376_376721

def is_valid_n (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 10 * a + b ∧ n = a * (a + b)

theorem largest_valid_n : ∀ n : ℕ, is_valid_n n → n ≤ 48 := by sorry

example : is_valid_n 48 := by sorry

end largest_valid_n_l376_376721


namespace books_per_author_l376_376841

theorem books_per_author (total_books : ℕ) (authors : ℕ) (h1 : total_books = 198) (h2 : authors = 6) : total_books / authors = 33 :=
by sorry

end books_per_author_l376_376841


namespace initial_avg_height_l376_376181

-- Lean 4 statement for the given problem
theorem initial_avg_height (A : ℝ) (n : ℕ) (wrong_height correct_height actual_avg init_diff : ℝ)
  (h_class_size : n = 35)
  (h_wrong_height : wrong_height = 166)
  (h_correct_height : correct_height = 106)
  (h_actual_avg : actual_avg = 183)
  (h_init_diff : init_diff = wrong_height - correct_height)
  (h_total_height_actual : n * actual_avg = 35 * 183)
  (h_total_height_wrong : n * A = 35 * actual_avg - init_diff) :
  A = 181 :=
by {
  -- The problem and conditions are correctly stated. The proof is skipped with sorry.
  sorry
}

end initial_avg_height_l376_376181


namespace arrangements_not_next_to_each_other_l376_376595

theorem arrangements_not_next_to_each_other (A B : Type) [fintype A] [decidable_eq B] [fintype B] (h : fintype.card B = 6) :
  ∃ n, n = 480 ∧ ∀ perm : list B → list B, ¬ (person_a_next_to_person_b perm A B) :=
sorry

def person_a_next_to_person_b {B : Type} [decidable_eq B] (perm : list B) (A B : B) : Prop :=
∃ i, (perm[i] = A ∧ perm[i+1] = B) ∨ (perm[i] = B ∧ perm[i+1] = A)

end arrangements_not_next_to_each_other_l376_376595


namespace smallest_n_divisibility_problem_l376_376954

theorem smallest_n_divisibility_problem :
  ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), 1 ≤ k → k ≤ n + 2 → n^3 - n ≠ 0 → (n^3 - n) % k = 0) ∧
    (∃ (k : ℕ), 1 ≤ k → k ≤ n + 2 → k ∣ n^3 - n) ∧
    (∃ (k : ℕ), 1 ≤ k → k ≤ n + 2 → ¬ k ∣ n^3 - n) ∧
    (∀ (m : ℕ), m > 0 ∧ (∀ (k : ℕ), 1 ≤ k → k ≤ m + 2 → m^3 - m ≠ 0 → (m^3 - m) % k = 0) ∧
      (∃ (k : ℕ), 1 ≤ k → k ≤ m + 2 → k ∣ m^3 - m) ∧
      (∃ (k : ℕ), 1 ≤ k → k ≤ m + 2 → ¬ k ∣ m^3 - m) → n ≤ m) :=
sorry

end smallest_n_divisibility_problem_l376_376954


namespace probability_a2_plus_b2_eq_25_probability_isosceles_triangle_l376_376215

-- Part (Ⅰ): Probability that a^2 + b^2 = 25
theorem probability_a2_plus_b2_eq_25 :
  let possible_outcomes := [(a, b) | a ∈ (Finset.range 7).erase 0, b ∈ (Finset.range 7).erase 0],
      favorable_outcomes := possible_outcomes.filter (λ (pair : ℕ × ℕ), pair.1 ^ 2 + pair.2 ^ 2 = 25),
      total_outcomes := 36
  in favorable_outcomes.card / total_outcomes = 1 / 18 :=
sorry

-- Part (Ⅱ): Probability that a, b, 5 form an isosceles triangle
theorem probability_isosceles_triangle :
  let possible_outcomes := [(a, b) | a ∈ (Finset.range 7).erase 0, b ∈ (Finset.range 7).erase 0],
      favorable_outcomes := possible_outcomes.filter
          (λ (pair : ℕ × ℕ), pair.1 = 5 ∨ pair.2 = 5 ∨ pair.1 = pair.2),
      total_outcomes := 36
  in favorable_outcomes.card / total_outcomes = 7 / 18 :=
sorry

end probability_a2_plus_b2_eq_25_probability_isosceles_triangle_l376_376215


namespace median_of_consecutive_integers_l376_376585

theorem median_of_consecutive_integers (sum_n : ℤ) (n : ℤ) 
  (h1 : sum_n = 6^4) (h2 : n = 36) : 
  (sum_n / n) = 36 :=
by
  sorry

end median_of_consecutive_integers_l376_376585


namespace hexagon_count_in_100th_ring_l376_376691

theorem hexagon_count_in_100th_ring :
  (100 * 6) = 600 :=
by
  calculate
  -- Proof for the theorem will be inserted here
  sorry

end hexagon_count_in_100th_ring_l376_376691


namespace find_p_q_pairs_l376_376337

noncomputable def is_integer_root (x : ℤ) (p q : ℤ) :=
  x^4 + 2 * p * x^2 + q * x + p^2 - 36 = 0

theorem find_p_q_pairs :
  ∃ (p q : ℤ), 
    (∀ (a b c d : ℤ), 
     a + b + c + d = 0 ∧
     ab + ac + ad + bc + bd + cd = 2 * p ∧
     abcd = p^2 - 36 ∧
     is_integer_root a p q ∧
     is_integer_root b p q ∧
     is_integer_root c p q ∧
     is_integer_root d p q ∧
     (a, b, c, d)) :=
sorry

end find_p_q_pairs_l376_376337


namespace empty_one_container_l376_376211

theorem empty_one_container (a b c : ℕ) :
  ∃ a' b' c', (a' = 0 ∨ b' = 0 ∨ c' = 0) ∧
    (a' = a ∧ b' = b ∧ c' = c ∨
     (a' ≤ a ∧ b' ≤ b ∧ c' ≤ c ∧ (a + b + c = a' + b' + c')) ∧
     (∀ i j, i ≠ j → (i = 1 ∨ i = 2 ∨ i = 3) →
              (j = 1 ∨ j = 2 ∨ j = 3) →
              (if i = 1 then (if j = 2 then a' = a - a ∨ a' = a else (if j = 3 then a' = a - a ∨ a' = a else false))
               else if i = 2 then (if j = 1 then b' = b - b ∨ b' = b else (if j = 3 then b' = b - b ∨ b' = b else false))
               else (if j = 1 then c' = c - c ∨ c' = c else (if j = 2 then c' = c - c ∨ c' = c else false))))) :=
by
  sorry

end empty_one_container_l376_376211


namespace ratio_of_gaps_is_two_to_one_l376_376690

def older_brother_birth_year := 1932
def older_sister_birth_year := 1936
def grandmother_birth_year := 1944

def gap_older_siblings := older_sister_birth_year - older_brother_birth_year
def gap_grandmother_and_sister := grandmother_birth_year - older_sister_birth_year

theorem ratio_of_gaps_is_two_to_one : 
  (gap_grandmother_and_sister : ℚ) / gap_older_siblings = 2 :=
by
  rw [gap_older_siblings, gap_grandmother_and_sister]
  norm_cast
  exact (8 : ℚ) / 4

end ratio_of_gaps_is_two_to_one_l376_376690


namespace problem_1_problem_2_l376_376521

-- Problem 1: Proof statement
theorem problem_1 : 
  (1 : ℝ) * (0.64) ^ (-1 / 2) + (27 : ℝ)^(2 / 3) - (1 / 4) ^ 0 - (1 / 2) ^ (-3) = 5 / 4 := 
by
  sorry

-- Problem 2: Proof statement
theorem problem_2 :
  2 * Real.log 10 / Real.log 3 + Real.log 0.81 / Real.log 3 = 4 :=
by 
  sorry

end problem_1_problem_2_l376_376521


namespace sequence_bound_l376_376851

noncomputable def bounded_seq (a : ℕ → ℝ) := ∃ D, ∀ n, a n < D

theorem sequence_bound (a : ℕ → ℝ) (H1 : bounded_seq a)
  (H2 : ∀ n, a n < (∑ k in finset.range (2*n + 2007)).filter (λ k, k ≥ n).sum (λ k, a k / (k + 1)) + 1 / (2*n + 2007)) :
  ∀ n, a n < 1 / n :=
sorry

end sequence_bound_l376_376851


namespace inequality_proof_l376_376022

variable {x₁ x₂ x₃ x₄ : ℝ}

theorem inequality_proof
  (h₁ : x₁ ≥ x₂) (h₂ : x₂ ≥ x₃) (h₃ : x₃ ≥ x₄) (h₄ : x₄ ≥ 2)
  (h₅ : x₂ + x₃ + x₄ ≥ x₁) 
  : (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := 
by {
  sorry
}

end inequality_proof_l376_376022


namespace original_number_people_l376_376104

theorem original_number_people (x : ℕ)
    (h1 : ∀ {x}, ∃ y, y = (2 * x) / 3)
    (dancers : ∀ {x}, ∃ z, z = (1 / 4) * ((2 * x) / 3))
    (new_total : ∀ {x}, ∃ w, w = ((2 * x) / 3) + 6)
    (non_dancers : ∀ {x}, ∃ v, v = ((2 * x) / 3) + 6 - ((1 / 4) * ((2 * x) / 3)) = 15) : 
  x = 27 := 
sorry

end original_number_people_l376_376104


namespace cost_of_largest_pot_l376_376144

theorem cost_of_largest_pot
  (n : ℕ) (n = 6) (total_cost : ℝ) (total_cost = 8.25) (increment : ℝ) (increment = 0.3) :
  ∃ (cost_of_smallest : ℝ),
  let largest_pot := (cost_of_smallest + increment * 5) in
  6 * cost_of_smallest + increment * (0 + 1 + 2 + 3 + 4 + 5) = total_cost ∧
  largest_pot = 2.125 :=
begin
  sorry,
end

end cost_of_largest_pot_l376_376144


namespace num_multiples_of_three_in_ap_l376_376548

variable (a : ℕ → ℚ)  -- Defining the arithmetic sequence

def first_term (a1 : ℚ) := a 1 = a1
def eighth_term (a8 : ℚ) := a 8 = a8
def general_term (d : ℚ) := ∀ n : ℕ, a n = 9 + (n - 1) * d
def multiple_of_three (n : ℕ) := ∃ k : ℕ, a n = 3 * k

theorem num_multiples_of_three_in_ap 
  (a : ℕ → ℚ)
  (h1 : first_term a 9)
  (h2 : eighth_term a 12) :
  ∃ n : ℕ, n = 288 ∧ ∃ l : ℕ → Prop, ∀ k : ℕ, l k → multiple_of_three a (k * 7 + 1) :=
sorry

end num_multiples_of_three_in_ap_l376_376548


namespace sum_of_series_l376_376694

noncomputable def T : ℝ :=
  1 - (1/3) - (1/9) + (1/27) - (1/81) - (1/243) + (1/729) - (1/2187) - ...

theorem sum_of_series : T = 27 / 39 :=
  sorry

end sum_of_series_l376_376694


namespace rod_weight_proof_l376_376657

variable {Real : Type} [LinearOrderedField Real]

-- Defining the constants based on the conditions
def length_6m : Real := 6
def weight_6m : Real := 10.8
def length_13m : Real := 13

-- Calculating weight per meter
def weight_per_meter : Real := weight_6m / length_6m

-- Calculating weight for 13 meters long rod
def weight_13m : Real := weight_per_meter * length_13m

theorem rod_weight_proof : weight_13m = 23.4 := 
by 
  sorry

end rod_weight_proof_l376_376657


namespace basketball_tournament_probability_l376_376500

theorem basketball_tournament_probability :
  let A_games := 7
  let win_probability := 0.5
  let A_initial_point := 1
  let A_remaining_wins := ∑ i in {4, 5, 6}, (Nat.choose 7 i) * (win_probability ^ i) * ((1 - win_probability) ^ (A_games - i))
  let total_probability := A_remaining_wins / (win_probability ^ A_games)
  (total_probability : ℚ) = 11 / 32 := 
by
  sorry

end basketball_tournament_probability_l376_376500


namespace correct_answer_l376_376618

def opposite_meanings (condA condB condC condD : Prop) : Prop :=
  condA = (Winning 1 Game ∧ Losing 20000 Dollars) ∧
  condB = (Traveling 5 km East ∧ Traveling 10 km North) ∧
  condC = (Transporting 6 kg Apples ∧ Selling 5 kg Apples) ∧
  condD = (WaterLevel Rising 0.6 meters ∧ WaterLevel Dropping 1 meter)

theorem correct_answer (condA condB condC condD : Prop) :
  opposite_meanings condA condB condC condD → condD :=
by
  sorry

end correct_answer_l376_376618


namespace sacks_after_6_days_l376_376492

theorem sacks_after_6_days (sacks_per_day : ℕ) (days : ℕ) 
  (h1 : sacks_per_day = 83) (h2 : days = 6) : 
  sacks_per_day * days = 498 :=
by
  sorry

end sacks_after_6_days_l376_376492


namespace num_young_employees_l376_376557

-- Suppose the total number of employees be n.
variables (n : ℕ) (r_y r_m r_e : ℕ)
-- Ratio of young, middle-aged, and elderly employees is 10:8:7.
-- It means y:m:e = 10:8:7, so we have r_y = 10, r_m = 8, r_e = 7.
-- Then the total ratio is r_y + r_m + r_e.
def total_ratio := r_y + r_m + r_e

-- Given conditions
variables (selected_employees : ℕ) (prob_selection : ℚ)

-- Assume 200 employees are selected.
-- Assume probability of each being selected is 0.2.
def num_employees (selected_employees : ℕ) (prob_selection : ℚ) :=
  selected_employees / prob_selection

def num_young (n : ℕ) (r_y r_m r_e : ℕ) :=
  (r_y * n) / total_ratio r_y r_m r_e

theorem num_young_employees 
(selected_employees : 200) (prob_selection : 0.2) 
(r_y : 10) (r_m : 8) (r_e : 7) :
(num_young (num_employees selected_employees prob_selection) r_y r_m r_e) = 400 := 
sorry

end num_young_employees_l376_376557


namespace initial_investment_approximation_l376_376727

theorem initial_investment_approximation :
  ∃ (x : ℝ), x ≈ 340.27 ∧ x * 1.08^5 = 500 :=
sorry

end initial_investment_approximation_l376_376727


namespace arab_soldiers_zero_l376_376151

noncomputable def arab_soldiers_taken (x : ℕ) (y : ℕ) : ℕ :=
  let remaining_eskimos := y in
  let missing_indians := x / 3 in
  let total_soldiers_taken := (4 * x) / 3 in
  let total_taken := y + (x - y) + missing_indians in
  total_soldiers_taken - total_taken

theorem arab_soldiers_zero (x y : ℕ) (h1 : y = x / 3) :
  arab_soldiers_taken x y = 0 :=
by
  sorry

end arab_soldiers_zero_l376_376151


namespace roots_cubed_l376_376695

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 + 3 * x - 1
noncomputable def g (x : ℝ) : ℝ := x^3 - 6 * x^2 + 11 * x - 6

theorem roots_cubed (r : ℝ) : f r = 0 → g (r^3) = 0 := by
  intro hr
  -- Assume r is a root of f
  have h1 : r^3 = 2 * r^2 - 3 * r + 1, from sorry
  -- Prove that r^3 is a root of g using h1
  suffices h2 : g (r^3) = (r^3)^3 - 6 * (r^3)^2 + 11 * r^3 - 6, from sorry
  -- Confirm the ordered triple (b, c, d) for g(x) is (-6, 11, -6)
  exact h2

end roots_cubed_l376_376695


namespace larger_integer_value_l376_376925

theorem larger_integer_value 
  (a b : ℤ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a = 7 * k) 
  (h4 : b = 3 * k) 
  (h5 : a * b = 168) : 
  a = 14 * real.sqrt 2 := 
sorry

end larger_integer_value_l376_376925


namespace problem_x2_minus_y2_l376_376805

-- Problem statement: Given the conditions, prove x^2 - y^2 = 5 / 1111
theorem problem_x2_minus_y2 (x y : ℝ) (h1 : x + y = 5 / 11) (h2 : x - y = 1 / 101) :
  x^2 - y^2 = 5 / 1111 :=
by
  sorry

end problem_x2_minus_y2_l376_376805


namespace trigonometric_identity_l376_376590

theorem trigonometric_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (70 * Real.pi / 180) +
   Real.sin (10 * Real.pi / 180) * Real.sin (50 * Real.pi / 180)) = 1 / 4 :=
by sorry

end trigonometric_identity_l376_376590


namespace compare_relative_errors_and_absolute_increase_l376_376298

def first_measurement_error : ℝ := 0.05
def first_line_length : ℝ := 25
def second_measurement_error : ℝ := 0.5
def second_line_length : ℝ := 200

def relative_error (measurement_error : ℝ) (line_length : ℝ) : ℝ :=
  (measurement_error / line_length) * 100

theorem compare_relative_errors_and_absolute_increase :
  relative_error second_measurement_error second_line_length > relative_error first_measurement_error first_line_length
  ∧ second_measurement_error - first_measurement_error = 0.45 :=
by
  sorry

end compare_relative_errors_and_absolute_increase_l376_376298


namespace terminating_decimals_l376_376356

theorem terminating_decimals (k : ℕ) (h₁ : 1 ≤ k) (h₂ : k ≤ 419) : 
  (∃ (n : ℕ), (1 ≤ n ∧ n ≤ 419 ∧ (∃ (m : ℕ), n = 21 * m) ∧ ((420 / (420 / n)).denom.factors ≤ [2, 5])) → k = 19) := sorry

end terminating_decimals_l376_376356


namespace determine_m_find_n_l376_376704

variable {x m n : ℝ}

/- Question (1) -/
def matrix_inequality (m : ℝ) : Prop :=
∀ x : ℝ, (x + m) * x - 2 < 0 → x > -1 ∧ x < 2

theorem determine_m : ∃ m, matrix_inequality m :=
exists.intro (-1)
by 
  sorry

/- Question (2) -/
def quadratic_root_n (m n : ℝ) (x₁ x₂ : ℂ) : Prop :=
x₁ + x₂ = -m ∧ x₁ * x₂ = n

theorem find_n (m : ℝ) : ∃ n, quadratic_root_n m n ((1/2) + (complex.I * sqrt(3)/2)) ((1/2) - (complex.I * sqrt(3)/2)) :=
exists.intro 1
by 
  sorry

end determine_m_find_n_l376_376704


namespace total_hours_at_ballpark_l376_376150

theorem total_hours_at_ballpark :
  let days_in_week := 7
  let weekdays_in_week := 5
  let nathan_days := 2 * days_in_week
  let nathan_hours := 3 * nathan_days
  let tobias_days := 1 * days_in_week
  let tobias_hours := 5 * tobias_days
  let leo_days := 10
  let leo_hours := 2.5 * leo_days
  let maddison_days := 3 * weekdays_in_week
  let maddison_hours := 6 * maddison_days
  nathan_hours + tobias_hours + leo_hours + maddison_hours = 192 :=
by 
  let days_in_week := 7
  let weekdays_in_week := 5
  let nathan_days := 2 * days_in_week
  let nathan_hours := 3 * nathan_days
  let tobias_days := 1 * days_in_week
  let tobias_hours := 5 * tobias_days
  let leo_days := 10
  let leo_hours := 2.5 * leo_days
  let maddison_days := 3 * weekdays_in_week
  let maddison_hours := 6 * maddison_days
  show nathan_hours + tobias_hours + leo_hours + maddison_hours = 192
  sorry -- Proof omitted

end total_hours_at_ballpark_l376_376150


namespace equal_piles_impossible_l376_376209

theorem equal_piles_impossible (initial_stones : ℕ) (move_stones : ℕ → ℕ) 
  (h_initial : initial_stones = 2017)
  (h_move : ∀ i > 0, move_stones i = i)
  (h_properties : ∃ n ≥ 2, 
    ∀ j, ∃ (split_stone_count: ℕ), split_stone_count ∈ (Part.less_than_or_equal initial_stones) → (∃ p1 p2, p1 + move_stones j = split_stone_count ∧ p2 = split_stone_count - p1 ∧ p1 ≠ 0 ∧ p2 ≠ 0 ∧
    ((sum_of_piles_after_moves j initial_stones + (j * (j + 1)) / 2) % ∀ m : ℕ, stocks ∉ (disjoint_collections (has_prime_factors)) 
     })) : False := sorry

end equal_piles_impossible_l376_376209


namespace integer_roots_and_composite_l376_376512

theorem integer_roots_and_composite (a b : ℤ) (h1 : ∃ x1 x2 : ℤ, x1 * x2 = 1 - b ∧ x1 + x2 = -a) (h2 : b ≠ 1) : 
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ m * n = (a^2 + b^2) := 
sorry

end integer_roots_and_composite_l376_376512


namespace milk_production_improved_rate_l376_376529

variable (a b c d e : ℕ)

-- Assume original milk production rate and improved rate
def original_rate_per_day_per_cow := b / (a * c)
def improved_rate_per_day_per_cow := 1.2 * (original_rate_per_day_per_cow a b c)

-- How many gallons of milk will d cows give in e days at the improved rate
theorem milk_production_improved_rate (ha : 0 < a) (hc : 0 < c) (he : 0 < e) :
  d * e * (improved_rate_per_day_per_cow a b c) = (1.2 * b * d * e) / (a * c) := by
  sorry

end milk_production_improved_rate_l376_376529


namespace solve_equation_y_equals_64_l376_376176

theorem solve_equation_y_equals_64 (y : ℝ) :
  (sqrt (2 + sqrt (3 + sqrt y)) = real.sqrt (2 + real.cbrt y) ^ (1/4)) → y = 64 :=
by
  sorry

end solve_equation_y_equals_64_l376_376176


namespace fishing_festival_total_fish_l376_376936

noncomputable def total_fish_caught (y : ℕ) : ℕ  := 6 * y - 107

theorem fishing_festival_total_fish :
  let y := 9 + 5 + 7 + 23 + 5 + 2 + 1 in -- Total number of contestants
  let fish_caught := total_fish_caught y in
  fish_caught = 127 :=
by
  sorry

end fishing_festival_total_fish_l376_376936


namespace sum_eight_l376_376749

variable {a : ℕ → ℚ} {S : ℕ → ℚ}

-- **Conditions**
-- 1. a_3 + a_4 = 18 - a_6 - a_5
def condition1 : Prop := a 3 + a 4 = 18 - (a 6 + a 5)

-- 2. Arithmetic sequence properties and sum of the sequence
def arithmetic_seq (a : ℕ → ℚ) :=
  ∀ n k : ℕ, a (n + k) - a n = a (n + 1) - a n

-- 3. Sum of first n terms of arithmetic sequence S_n
def sum_of_sequence (S : ℕ → ℚ) (a : ℕ → ℚ) :=
  ∀ n : ℕ, S n = (n : ℚ) * (a 1 + a n) / 2

-- **Proof Goal**
theorem sum_eight (h1 : condition1) (h2 : arithmetic_seq a) (h3 : sum_of_sequence S a) : S 8 = 36 := 
sorry

end sum_eight_l376_376749


namespace range_of_a_l376_376777

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then -x^3 + x^2 else (a * Real.log x) / (x * (x + 1))

theorem range_of_a (a : ℝ) (A B : ℝ × ℝ) (OA_perpendicular_OB : angle (0,0) A (0,0) B = π/2)
  (mid_AB_on_y_axis : (A.1 + B.1) / 2 = 0) :
  a ∈ Set.Ici Real.exp :=
sorry

end range_of_a_l376_376777


namespace monotonic_range_of_a_l376_376768

theorem monotonic_range_of_a 
  (a : ℝ)
  (h_mono : ∀ x y : ℝ, (f x a - f y a) * (x - y) ≥ 0)
  : - real.sqrt 3 ≤ a ∧ a ≤ real.sqrt 3 :=
by sorry

def f (x a : ℝ) := -x^3 + a*x^2 - x - 2

end monotonic_range_of_a_l376_376768


namespace least_fraction_proof_l376_376235

noncomputable def least_fraction_added_to_unity :=
  let series_sum := (∑ n in finset.range (21 - 2 + 1), (1 / ((n + 2) * ((n + 2) + 1)))) 
  let one := 1 in
  let seven_over_twenty_two := 7 / 22 in
  let fifteen_over_twenty_two := 15 / 22 in
  thirteen

theorem least_fraction_proof : least_fraction_added_to_unity = 15 / 22 := sorry

end least_fraction_proof_l376_376235


namespace log_equation_solutions_are_irrational_l376_376900

theorem log_equation_solutions_are_irrational :
  ∃ x₁ x₂ : ℝ, (log 2 (3 * x₁^2 - 12 * x₁) = 3 ∧ log 2 (3 * x₂^2 - 12 * x₂) = 3) ∧
              (x₁ = 2 + 2 * real.sqrt 15 / 3 ∧ x₂ = 2 - 2 * real.sqrt 15 / 3) ∧
              irrational x₁ ∧ irrational x₂ :=
by
  sorry

end log_equation_solutions_are_irrational_l376_376900


namespace total_number_of_boys_l376_376249

theorem total_number_of_boys (T : ℝ)
  (h1 : 0.20 * T = number_of_boys_from_school_a)
  (h2 : 0.30 * (0.20 * T) = number_of_boys_from_school_a_study_science)
  (h3 : number_of_boys_from_school_a - number_of_boys_from_school_a_study_science = 21) :
  T = 150 := 
by 
  have h4 : 0.06 * T = number_of_boys_from_school_a_study_science 
    := by rw [h2]

  have h5 : 0.14 * T = 21 
    := by rw [←h3]; rw [h4] at h3 

  have h6 : T = 21 / 0.14 := by linarith

  show T = 150, from sorry

end total_number_of_boys_l376_376249


namespace polygon_symmetric_image_vertex_inside_boundary_l376_376886

noncomputable def centrallySymmetric (M : set (ℝ × ℝ)) (O : ℝ × ℝ) :=
  ∀ (x : ℝ × ℝ), (x ∈ M) → (2 * O - x) ∈ M

theorem polygon_symmetric_image_vertex_inside_boundary
  (T M : set (ℝ × ℝ))
  (P : ℝ × ℝ)
  (h1 : T ⊆ M)
  (h2 : centrallySymmetric M O)
  (h3 : P ∈ T)
  (T' := {x : ℝ × ℝ | ∃ (y ∈ T), x = 2 * P - y}) :
  ∃ v ∈ T', v ∈ M :=
sorry

end polygon_symmetric_image_vertex_inside_boundary_l376_376886


namespace largest_divisor_consecutive_odd_l376_376857

theorem largest_divisor_consecutive_odd (m n : ℤ) (h : ∃ k : ℤ, m = 2 * k + 1 ∧ n = 2 * k - 1) :
  ∃ d : ℤ, d = 8 ∧ ∀ m n : ℤ, (∃ k : ℤ, m = 2 * k + 1 ∧ n = 2 * k - 1) → d ∣ (m^2 - n^2) :=
by
  sorry

end largest_divisor_consecutive_odd_l376_376857


namespace triangle_area_PQR_l376_376456

noncomputable def area_TRIANGLE_PQR (PQ PR PM : ℕ) :=
  let semi_perimeter := (PQ + PR + 24) / 2
  let area := Real.sqrt (semi_perimeter * (semi_perimeter - PQ) * (semi_perimeter - PR) * (semi_perimeter - 24))
  area

theorem triangle_area_PQR
  (PQ PR PM : ℕ)
  (hPQ : PQ = 8)
  (hPR : PR = 18)
  (hPM : PM = 12) :
  area_TRIANGLE_PQR PQ PR PM = Real.sqrt 2975 :=
by
  -- Given conditions
  have h_semi_perimeter : (PQ + PR + 24) / 2 = 25 := sorry
  -- Calculate the area of triangle \(PRS\) and prove it equals \(\sqrt{2975}\)
  have h_area : Real.sqrt (25 * (25 - PQ) * (25 - PR) * (25 - 24)) = Real.sqrt 2975 := sorry
  -- Hence, the area of \(PQR\) is also \(\sqrt{2975}\) due to the congruency of \(PQR\) and \(PRS\)
  exact h_area

end triangle_area_PQR_l376_376456


namespace cosine_of_angle_is_negative_one_l376_376256

def A : ℝ × ℝ × ℝ := (-1, 2, -3)
def B : ℝ × ℝ × ℝ := (0, 1, -2)
def C : ℝ × ℝ × ℝ := (-3, 4, -5)

def vector_sub (p1 p2 : ℝ × ℝ × ℝ) := (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

def AB := vector_sub A B
def AC := vector_sub A C

def dot_product (v1 v2 : ℝ × ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) := real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def cos_angle (v1 v2 : ℝ × ℝ × ℝ) := (dot_product v1 v2) / ((magnitude v1) * (magnitude v2))

theorem cosine_of_angle_is_negative_one : cos_angle AB AC = -1 := by 
  sorry

end cosine_of_angle_is_negative_one_l376_376256


namespace find_a_minus_c_l376_376248

theorem find_a_minus_c (a b c : ℝ) (h1 : (a + b) / 2 = 80) (h2 : (b + c) / 2 = 180) : a - c = -200 :=
by 
  sorry

end find_a_minus_c_l376_376248


namespace sunset_duration_l376_376333

theorem sunset_duration (changes : ℕ) (interval : ℕ) (total_changes : ℕ) (h1 : total_changes = 12) (h2 : interval = 10) : ∃ hours : ℕ, hours = 2 :=
by
  sorry

end sunset_duration_l376_376333


namespace line_perpendicular_planes_l376_376527

open Set Theory

variables {Point Line Plane : Type} [has_coe Line (Set Point)] [has_coe Plane (Set Point)]
variable {l : Line}
variables {alpha beta : Plane}

noncomputable def perpendicular (l : Line) (p : Plane) : Prop := ∀ (A B : Point), A ∈ l → B ∈ l → A ≠ B → A ∈ p ∧ B ∈ p
noncomputable def parallel (p1 p2 : Plane) : Prop := ∀ (A B : Point), A ∈ p1 ∧ B ∈ p2 → A ∈ p2 ∨ B ∈ p1

theorem line_perpendicular_planes (h1 : perpendicular l alpha) (h2 : parallel alpha beta) : perpendicular l beta :=
sorry

end line_perpendicular_planes_l376_376527


namespace defective_pens_l376_376432

theorem defective_pens {D N : ℕ} (h1 : D + N = 10) 
  (h2 : (N / 10) * ((N - 1) / 9) = 0.4666666666666666) : D = 3 :=
sorry

end defective_pens_l376_376432


namespace find_three_digit_number_l376_376339

theorem find_three_digit_number (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) 
  : 100 * a + 10 * b + c = 5 * a * b * c → a = 1 ∧ b = 7 ∧ c = 5 :=
by
  sorry

end find_three_digit_number_l376_376339


namespace groups_of_people_l376_376439

theorem groups_of_people (men women : ℕ) (group_size : ℕ) :
  men = 4 → women = 5 → group_size = 3 →
  (∃ g1 g2 g3,
    (g1.card = group_size ∧ g2.card = group_size ∧ g3.card = group_size) ∧
    (∃ mana womena, g1 ≠ g2 ∧ g2 ≠ g3 ∧ g1 ≠ g3 ∧
      1 ≤ mana ∧ 1 ≤ womena ∧
      (mana + womena = group_size)
    ) ∧
    4.choose 2 * 5.choose 1 * 2.choose 1 * 4.choose 2 * 1 * 1 = 360
  ) :=
begin
  intros,
  sorry
end

end groups_of_people_l376_376439


namespace max_a_value_l376_376078

theorem max_a_value :
  ∀ (a x : ℝ), 
  (x - 1) * x - (a - 2) * (a + 1) ≥ 1 → a ≤ 3 / 2 := sorry

end max_a_value_l376_376078


namespace fraction_of_allowance_l376_376837

-- Definitions from conditions
def initial_amount : ℕ := 43
def allowance_per_week : ℕ := 10
def weeks : ℕ := 8
def final_amount : ℕ := 83
def amount_saved : ℕ := final_amount - initial_amount
def total_allowance : ℕ := allowance_per_week * weeks

-- The fraction of allowance that Jack puts into his piggy bank every week
variable (f : ℚ)

-- The main statement to be proved
theorem fraction_of_allowance :
  f = amount_saved / total_allowance :=
by
  have h1 : amount_saved = 40 := by norm_num
  have h2 : total_allowance = 80 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end fraction_of_allowance_l376_376837


namespace log_a_2_m_log_a_3_n_l376_376363

theorem log_a_2_m_log_a_3_n (a m n : ℝ) (h1 : log a 2 = m) (h2 : log a 3 = n) : a^(2*m + n) = 12 :=
sorry

end log_a_2_m_log_a_3_n_l376_376363


namespace product_of_local_and_absolute_value_l376_376247

theorem product_of_local_and_absolute_value :
  ∀ (n : ℕ), 
    (∀ (d : ℕ), (d = 4) → (564823 = n) → (local_value d n = 40000)) →
    (∀ (d : ℤ), (d = 4) → (abs d = 4)) →
    product (local_value 4 564823) (abs 4) = 160000 := by
  sorry

def local_value (d : ℕ) (n : ℕ) : ℕ :=
  if n = 564823 ∧ d = 4 then 40000 else 0

def product (x y : ℕ) : ℕ := x * y

end product_of_local_and_absolute_value_l376_376247


namespace centers_lie_on_circle_l376_376967

theorem centers_lie_on_circle {n : ℕ} (A : Fin n → Point) (O : Point)
  (C : Fin n → Point) (hC : ∀ i, C i = center_circle_through (O, A i, A ((i + 1) % n)))
  (polygon_inscribed : is_circumscribed_polygon A O)
  (polygon_circumscribed : is_inscribed_polygon A O) :
  ∃ K : Point, ∀ i, dist K (C i) = dist K (C 0) :=
sorry

end centers_lie_on_circle_l376_376967


namespace no_positive_real_roots_l376_376933

theorem no_positive_real_roots (x : ℝ) : (x^3 + 6 * x^2 + 11 * x + 6 = 0) → x < 0 :=
sorry

end no_positive_real_roots_l376_376933


namespace product_of_real_parts_of_roots_l376_376474

def i := Complex.I

theorem product_of_real_parts_of_roots (z : ℂ) : 
  (z^2 + 3 * z + (7 - 2 * i) = 0) →
  let root1 := (-3 + Complex.cosh (-19 + 8 * i)) / 2 in
  let root2 := (-3 - Complex.cosh (-19 + 8 * i)) / 2 in
  (root1.re * root2.re = 2) := 
by 
  sorry

end product_of_real_parts_of_roots_l376_376474


namespace white_cubes_position_invariant_l376_376593

theorem white_cubes_position_invariant : 
  ∀ (cubes1 cubes2 : ℕ → Prop), 
    (∀ l x y, x < 100 ∧ y < 100 ∧ x + l ≤ 100 ∧ y + l ≤ 100 →
      abs (count_set (cubes1 ∘ (λ n, x + n)) {b | b = 1}) - 
          count_set (cubes1 ∘ (λ n, y + n)) {b | b = 1}) ≤ 1) ∧
    (∀ l x, x + l ≤ 100 → count_set (cubes1 ∘ (λ n, n)) {b | b = 1} ≤ 
          count_set (cubes1 ∘ (λ n, x + n)) {b | b = 1}) ∧
    (∀ l y, y + l ≤ 100 → count_set (cubes1 ∘ (λ n, 100 - l + n)) {b | b = 1} ≥ 
          count_set (cubes1 ∘ (λ n, y + n)) {b | b = 1}) →
    (count_set (cubes1 ∘ (λ n, n)) {w | w = 0}) = 23 ∧
    (count_set (cubes2 ∘ (λ n, n)) {w | w = 0}) = 23 →
    ∀ n, cubes1 n = 0 ↔ cubes2 n = 0 :=
sorry

end white_cubes_position_invariant_l376_376593


namespace painted_cubes_count_l376_376279

-- Given conditions
variables (m n k : ℕ) (h : k ≤ n ∧ n ≤ m) 

-- Definition of the problem
def total_cubes := m * n * k
def unpainted_cubes := (m - 1) * (n - 1) * (k - 1)

-- Half the cubes have at least one painted face
def half_painted_condition := 
  (total_cubes m n k) / 2 = total_cubes m n k - unpainted_cubes m n k

-- The set of possible painted cube counts
def possible_painted_counts := {60, 72, 84, 90, 120}

-- The proof statement
theorem painted_cubes_count (m n k : ℕ) (h : k ≤ n ∧ n ≤ m) :
  (total_cubes m n k - unpainted_cubes m n k) ∈ possible_painted_counts := 
sorry

end painted_cubes_count_l376_376279


namespace pairs_A_not_equal_pairs_B_not_equal_pairs_C_equal_pairs_D_equal_l376_376619

-- Define the functions in each pair

-- Pair A
def fA (x : ℝ) : ℝ := real.sqrt (x^2)
def gA (x : ℝ) : ℝ := (real.sqrt x)^2

-- Pair B
def fB (x : ℝ) : ℝ := 1
def gB (x : ℝ) : ℝ := x^0

-- Pair C
def fC (x : ℝ) : ℝ := if x >= 0 then x else -x
def gC (x : ℝ) : ℝ := real.sqrt (x^2)

-- Pair D
def fD (x : ℝ) : ℝ := x
def gD (x : ℝ) : ℝ := real.cbrt (x^3)

theorem pairs_A_not_equal : ∃ x : ℝ, fA x ≠ gA x := by
  sorry

theorem pairs_B_not_equal : ∃ x : ℝ, fB x ≠ gB x := by
  sorry

theorem pairs_C_equal : ∀ x : ℝ, fC x = gC x := by
  sorry

theorem pairs_D_equal : ∀ x : ℝ, fD x = gD x := by
  sorry

end pairs_A_not_equal_pairs_B_not_equal_pairs_C_equal_pairs_D_equal_l376_376619


namespace kate_saved_in_march_l376_376115

theorem kate_saved_in_march :
  ∃ M : ℤ, (∀ (AprilSavings MaySavings KeyboardCost MouseCost RemainingMoney : ℤ),
    AprilSavings = 13 → MaySavings = 28 → KeyboardCost = 49 → MouseCost = 5 → RemainingMoney = 14 →
    M + AprilSavings + MaySavings - KeyboardCost - MouseCost = RemainingMoney) → M = 27 :=
by
  use 27
  intros AprilSavings MaySavings KeyboardCost MouseCost RemainingMoney
  intros hApr hMay hKeyb hMouse hRem
  linarith

end kate_saved_in_march_l376_376115


namespace smallest_possible_angle_l376_376866

variables {ℝ : Type*} [inner_product_space ℝ E] {E : Type*} [inner_product_space ℝ E]
variables (a b c : E)
noncomputable theory

def vector_magnitudes_and_cross (a b c : E) : (∥a∥ = 1) ∧ (∥b∥ = 1) ∧ (∥c∥ = 3) ∧ (a × (a × c) + 2 • b = 0) := sorry

def angle_between (x y : E) : ℝ := (real.arccos ((inner_product_space.inner x y) / (∥x∥ * ∥y∥)))

theorem smallest_possible_angle (a b c : E)
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = 1)
  (h3 : ∥c∥ = 3)
  (h4 : a × (a × c) + 2 • b = 0)
  : angle_between a c = 53.13 := 
sorry

end smallest_possible_angle_l376_376866


namespace arithmetic_sequence_properties_l376_376377

-- Definition of the arithmetic sequence conditions
variables {a : ℕ → ℤ} (a1 : ℤ) (d : ℤ)
hypothesis seq_def : ∀ n, a n = a1 + (n - 1) * d
hypothesis a11 : a 11 = -26
hypothesis a51 : a 51 = 54

-- The Lean statement for the proof
theorem arithmetic_sequence_properties (h1 : a1 + 10 * d = -26)
                                       (h2 : a1 + 50 * d = 54) :
  a 14 = -20 ∧ ∀ n, (n ≥ 25) → (a n ≥ 0) :=
by
  -- Solution steps are omitted
  sorry

end arithmetic_sequence_properties_l376_376377


namespace problem_statement_l376_376259

variable (F : ℕ → Prop)

theorem problem_statement (h1 : ∀ k : ℕ, F k → F (k + 1)) (h2 : ¬F 7) : ¬F 6 ∧ ¬F 5 := by
  sorry

end problem_statement_l376_376259


namespace find_m_l376_376755

variables (O A B C : Type) [instance : AddCommGroup O] 
variables [module ℝ O] [innerProductSpace ℝ O]

variable (θ : ℝ)
variables (cos sin : ℝ → ℝ)
variables (overrightarrow : O → O → O)
variable (m : ℝ)
variables (a b c : ℝ)
variables (AB AC AO OB OC : O)

-- Given conditions
axiom cond1 : θ = a
axiom cond2 : overrightarrow A B = AB
axiom cond3 : overrightarrow A C = AC
axiom cond4 : overrightarrow A O = AO
axiom vector_eq : (cos b / sin c) • AB + (cos c / sin b) • AC = 2 * m • AO

-- The theorem to prove
theorem find_m (h : ∀ B C : ℝ, B + C = π - θ) (hAB : ∥AO∥ = 1)
  : m = sin θ :=
sorry

end find_m_l376_376755


namespace problem_l376_376015

theorem problem (
  a b : ℝ
  (h : 1 / a^2 + 1 / b^2 = 4 / (a^2 + b^2))
) :
  (b / a) ^ 2022 - (a / b) ^ 2021 = 0 ∨ (b / a) ^ 2022 - (a / b) ^ 2021 = 2 :=
  sorry

end problem_l376_376015


namespace points_distribution_consistent_l376_376895

variables (total_points : ℝ) (starting_points_per_game : ℝ) (reserve_points_per_game : ℝ) 
(starting_players : ℕ) (reserve_players : ℕ) (rookie_players_20 : ℕ) 
(rookie_players_10 : ℕ) (rookie_players_5 : ℕ ) 
(extra_points_20 : ℝ) (extra_points_10 : ℝ)

def starting_total_points := (starting_players * 20 * starting_points_per_game)
def reserve_total_points := (reserve_players * 15 * reserve_points_per_game)
def rookie_total_points :=
  total_points - (starting_total_points + reserve_total_points)

def total_points_from_rookies_with_conditions :=
  (rookie_players_20 * (20 + extra_points_20)) +
  (rookie_players_10 * (10 + extra_points_10)) +
  (rookie_players_5 * 5)

theorem points_distribution_consistent :
  starting_total_points + reserve_total_points + rookie_total_points = total_points :=
sorry

end points_distribution_consistent_l376_376895


namespace O1O2_parallel_BC_l376_376437

variables {A B C D E F K M N O1 O2 : Type*}
variables [triangle ABC] [acute_triangle ABC] (H1 : AB ≠ AC)
variables (H2 : midpoint K (median AD)) (H3 : perpendicular_foot D AB E) (H4 : perpendicular_foot D AC F)
variables (H5 : intersects KE BC M) (H6 : intersects KF BC N)
variables (H7 : circumcenter △(DEM) O1) (H8 : circumcenter △(DFN) O2)

theorem O1O2_parallel_BC : parallel O1O2 BC := 
by
  sorry

end O1O2_parallel_BC_l376_376437


namespace dagger_simplified_l376_376953

def dagger (m n p q : ℚ) : ℚ := (m^2) * p * (q / n)

theorem dagger_simplified :
  dagger (5:ℚ) (9:ℚ) (4:ℚ) (6:ℚ) = (200:ℚ) / (3:ℚ) :=
by
  sorry

end dagger_simplified_l376_376953


namespace hitting_target_at_least_twice_l376_376653

theorem hitting_target_at_least_twice (p : ℝ) (n : ℕ) (h_p : p = 0.6) (h_n : n = 3) : 
  (∑ k in {2, 3}, (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)) = 81 / 125 :=
by
  sorry

end hitting_target_at_least_twice_l376_376653


namespace limit_ax_eq_one_l376_376163

noncomputable def limit_ax (a : ℝ) (h : 0 < a) : ℝ :=
  sorry

theorem limit_ax_eq_one (a : ℝ) (h : 0 < a) : tendsto (λ x : ℚ, a ^ (x : ℝ)) (nhds 0) (nhds 1) :=
  sorry

end limit_ax_eq_one_l376_376163


namespace Jed_cards_after_4_weeks_l376_376107

theorem Jed_cards_after_4_weeks :
  (∀ n: ℕ, (if n % 2 = 0 then 20 + 4*n - 2*n else 20 + 4*n - 2*(n-1)) = 40) :=
by {
  sorry
}

end Jed_cards_after_4_weeks_l376_376107


namespace identify_linear_function_l376_376294

def is_linear_function (f : ℝ → ℝ) : Prop :=
  ∃ (m : ℝ) (b : ℝ), ∀ x, f x = m * x + b

def f_A (x : ℝ) : ℝ := 2 * x^2
def f_B (x : ℝ) : ℝ := 2 / x
def f_C (x : ℝ) : ℝ := -2^x - 3
def f_D (x : ℝ) : ℝ := (1/3) * x

theorem identify_linear_function : is_linear_function f_D ∧ 
  ¬ is_linear_function f_A ∧ ¬ is_linear_function f_B ∧ ¬ is_linear_function f_C :=
by
  sorry

end identify_linear_function_l376_376294


namespace consecutive_arithmetic_sequence_l376_376729

theorem consecutive_arithmetic_sequence (a b c : ℝ) 
  (h : (2 * b - a)^2 + (2 * b - c)^2 = 2 * (2 * b^2 - a * c)) : 
  2 * b = a + c :=
by
  sorry

end consecutive_arithmetic_sequence_l376_376729


namespace find_x_in_list_l376_376355

theorem find_x_in_list :
  ∃ x : ℕ, x > 0 ∧ x ≤ 120 ∧ (45 + 76 + 110 + x + x) / 5 = 2 * x ∧ x = 29 :=
by
  sorry

end find_x_in_list_l376_376355


namespace sum_of_values_c_l376_376323

noncomputable def f (x : ℝ) : ℝ := ((x - 4) * (x - 2) * x * (x + 2) * (x + 4)) / 120 - 2

theorem sum_of_values_c : 
    ∑ c in {c : ℝ | ∃ S : finset ℝ, S.card = 4 ∧ ∀ x ∈ S, f x = c}, true = -3.8 := 
by
  sorry

end sum_of_values_c_l376_376323


namespace correct_number_of_students_answered_both_l376_376880

def students_enrolled := 25
def answered_q1_correctly := 22
def answered_q2_correctly := 20
def not_taken_test := 3

def students_answered_both_questions_correctly : Nat :=
  let students_took_test := students_enrolled - not_taken_test
  let b := answered_q2_correctly
  b

theorem correct_number_of_students_answered_both :
  students_answered_both_questions_correctly = answered_q2_correctly :=
by {
  -- this space is for the proof, we are currently not required to provide it
  sorry
}

end correct_number_of_students_answered_both_l376_376880


namespace irreducible_polynomial_l376_376863

noncomputable def P (a : ℕ → ℤ) (n : ℕ) : Polynomial ℤ :=
  ∏ i in Finset.range n, (Polynomial.X - Polynomial.C (a i)) + 1

theorem irreducible_polynomial (n : ℕ) (a : ℕ → ℤ) (h : n ≥ 10)
  (distinct : Function.Injective a) :
  Irreducible (P a n) := 
sorry

end irreducible_polynomial_l376_376863


namespace solution_set_of_inequality_l376_376856

theorem solution_set_of_inequality (f : ℝ → ℝ) (b : ℝ) 
  (H1 : Even f) 
  (H2 : ∀ x ∈ Icc (-2*b) 0, ∃ y ∈ Icc x 0, f(x) ≤ f(y)) 
  (H3 : ∀ x ∈ Icc (-2*b) (3+b), True)
  : {x : ℝ | (f (x-1)) ≥ f 3} = Icc (-2) 4 :=
by
  sorry

end solution_set_of_inequality_l376_376856


namespace opposite_of_one_half_l376_376920

theorem opposite_of_one_half : -((1:ℚ)/2) = -1/2 := by
  -- Skipping the proof using sorry
  sorry

end opposite_of_one_half_l376_376920


namespace part_one_part_two_l376_376748

noncomputable def a_seq : ℕ → ℚ
| 1 := 1
| (n+1) := 1 / 16 * (1 + 4 * (a_seq n) + sqrt (1 + 24 * (a_seq n)))

def b_seq (n: ℕ) : ℚ := sqrt (1 + 24 * a_seq n)

theorem part_one :
  ∀ n, (b_seq (n+1) - 3) = (b_seq n - 3) / 2 := 
sorry

theorem part_two : 
  ∀ n, a_seq n = 2 / 3 * (1 / 4^n) + (1 / 2^n) + 1 / 3 :=
sorry

end part_one_part_two_l376_376748


namespace find_a6_a7_a8_l376_376935

variable {a : ℕ → ℝ}
variable {a1 : ℝ}
variable {d : ℝ}

-- Define the arithmetic sequence
def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

-- Define the sum of the first 13 terms
def sum_first_13_terms (a1 d : ℝ) : ℝ :=
  13 * a1 + (13 * (13 - 1) / 2) * d

-- Given: sum of the first 13 terms is 39
axiom sum_condition : sum_first_13_terms a1 d = 39

-- Prove that a6 + a7 + a8 = 9
theorem find_a6_a7_a8 (h : a1 + 6 * d = 3) :
  arithmetic_sequence a1 d 6 + arithmetic_sequence a1 d 7 + arithmetic_sequence a1 d 8 = 9 :=
sorry

end find_a6_a7_a8_l376_376935


namespace terminating_decimals_count_l376_376359

theorem terminating_decimals_count :
  let n := {n : ℕ | 1 ≤ n ∧ n ≤ 419 ∧ ∃ k : ℕ, n = 21 * k}
  n.card = 19 :=
by
  sorry

end terminating_decimals_count_l376_376359


namespace total_simple_interest_is_correct_l376_376992

noncomputable def principal : ℝ := 15041.875
noncomputable def rate : ℝ := 8
noncomputable def time : ℝ := 5
noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem total_simple_interest_is_correct :
  simple_interest principal rate time = 6016.75 := 
sorry

end total_simple_interest_is_correct_l376_376992


namespace kitty_cleaning_weeks_l376_376156

def time_spent_per_week (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust_furniture: ℕ) : ℕ :=
  pick_up + vacuum + clean_windows + dust_furniture

def total_weeks (total_time: ℕ) (time_per_week: ℕ) : ℕ :=
  total_time / time_per_week

theorem kitty_cleaning_weeks
  (pick_up_time : ℕ := 5)
  (vacuum_time : ℕ := 20)
  (clean_windows_time : ℕ := 15)
  (dust_furniture_time : ℕ := 10)
  (total_cleaning_time : ℕ := 200)
  : total_weeks total_cleaning_time (time_spent_per_week pick_up_time vacuum_time clean_windows_time dust_furniture_time) = 4 :=
by
  sorry

end kitty_cleaning_weeks_l376_376156


namespace cross_section_is_rectangle_l376_376612

def RegularTetrahedron : Type := sorry

def Plane : Type := sorry

variable (T : RegularTetrahedron) (P : Plane)

-- Conditions
axiom regular_tetrahedron (T : RegularTetrahedron) : Prop
axiom plane_intersects_tetrahedron (P : Plane) (T : RegularTetrahedron) : Prop
axiom plane_parallel_opposite_edges (P : Plane) (T : RegularTetrahedron) : Prop

-- The cross-section formed by intersecting a regular tetrahedron with a plane
-- that is parallel to two opposite edges is a rectangle.
theorem cross_section_is_rectangle (T : RegularTetrahedron) (P : Plane) 
  (hT : regular_tetrahedron T) 
  (hI : plane_intersects_tetrahedron P T) 
  (hP : plane_parallel_opposite_edges P T) :
  ∃ (shape : Type), shape = Rectangle := 
  sorry

end cross_section_is_rectangle_l376_376612


namespace total_spending_l376_376902

theorem total_spending :
  let price_per_pencil := 0.20
  let tolu_pencils := 3
  let robert_pencils := 5
  let melissa_pencils := 2
  let tolu_cost := tolu_pencils * price_per_pencil
  let robert_cost := robert_pencils * price_per_pencil
  let melissa_cost := melissa_pencils * price_per_pencil
  let total_cost := tolu_cost + robert_cost + melissa_cost
  total_cost = 2.00 := by
  sorry

end total_spending_l376_376902


namespace part_a_part_b_l376_376825

section
variables {A B C H M : Type} [triangle ABC]
variable {AH BM CD : ℝ}
variable {angle_ABC : ℝ}

-- Given an acute-angled triangle ABC, AH is the longest altitude with H lying on BC,
-- M is the midpoint of AC, and CD is an angle bisector with D on AB

-- (a) Prove that if AH <= BM, then angle ABC <= 60°
theorem part_a (h_acute: acute_angle_tr ABC) (h_altitude: longest_altitude AH H BC)
  (h_midpoint: midpoint M AC) (h_bisector: angle_bisector CD D AB)
  (h_condition: AH ≤ BM) : angle_ABC ≤ 60 :=
sorry

-- (b) Prove that if AH = BM = CD, then triangle ABC is equilateral
theorem part_b (h_acute: acute_angle_tr ABC) (h_altitude: longest_altitude AH H BC)
  (h_midpoint: midpoint M AC) (h_bisector: angle_bisector CD D AB)
  (h_condition: AH = BM ∧ BM = CD) : equilateral ABC :=
sorry
end

end part_a_part_b_l376_376825


namespace four_digit_integers_count_l376_376055

theorem four_digit_integers_count :
  let digits := [2, 2, 2, 9]
  let n := digits.length
  let n2 := digits.count 2
  let n9 := digits.count 9
  n = 4 ∧ n2 = 3 ∧ n9 = 1 →
  nat.factorial n / (nat.factorial n2 * nat.factorial n9) = 4 :=
by {
  sorry
}

end four_digit_integers_count_l376_376055


namespace number_of_integer_pairs_l376_376972

theorem number_of_integer_pairs :
  (∃ x y : ℕ, x ≤ y ∧ real.sqrt 1992 = real.sqrt x + real.sqrt y) → 2 = 2 := 
by
  sorry

end number_of_integer_pairs_l376_376972


namespace desargues_theorem_l376_376603

-- Definitions of Points, Lines, and Collinearity
structure Point :=
(x : ℝ) (y : ℝ)

structure Line :=
(p1 p2 : Point)
 
def is_collinear (p q r : Point) : Prop :=
  ∃ a b c, a * p.x + b * p.y + c = 0 ∧
           a * q.x + b * q.y + c = 0 ∧
           a * r.x + b * r.y + c = 0

axiom two_triangles_perspective_from_point 
  (A1 B1 C1 A2 B2 C2 O : Point)
  (h1 : ∃ l : Line, l.p1 = A1 ∧ l.p2 = A2 ∧ 
                    ∃ m : Line, m.p1 = B1 ∧ m.p2 = B2 ∧
                    ∃ n : Line, n.p1 = C1 ∧ n.p2 = C2 ∧ 
                    ∃ p : Line, p.p1 = O ∧ p.p2 = O)
  (h2 : intersects A1 A2 O ∧ intersects B1 B2 O ∧ intersects C1 C2 O) 
  : Prop

-- The statement of Desargues' Theorem in Lean
theorem desargues_theorem 
  (A1 B1 C1 A2 B2 C2 O P Q R : Point)
  (h1 : two_triangles_perspective_from_point A1 B1 C1 A2 B2 C2 O)
  (P_intersection : Line.mk A1 B1 ∩ Line.mk A2 B2 = P)
  (Q_intersection : Line.mk B1 C1 ∩ Line.mk B2 C2 = Q)
  (R_intersection : Line.mk C1 A1 ∩ Line.mk C2 A2 = R) :
  is_collinear P Q R := sorry

end desargues_theorem_l376_376603


namespace find_R_value_l376_376083

-- Define R such that R(1+R)^5 = 1
def satisfies_condition (R : ℝ) : Prop :=
  R * (1 + R)^5 = 1

-- The final proof statement
theorem find_R_value : ∃ R : ℝ, satisfies_condition R ∧ 
  ((R^(R^(R^3 + R^(-2)) + R^(-1)) + R^(-1)) = 2) :=
by
  use 0.618 -- R is approximately 0.618 from the context
  have h1 : satisfies_condition 0.618 := by
    sorry -- Proof that 0.618 satisfies the given condition
  have h2: (0.618^(0.618^(0.618^3 + 0.618^(-2)) + 0.618^(-1)) + 0.618^(-1)) = 2 := by
    sorry -- Proof of the final expression evaluation
  exact ⟨ 0.618, h1, h2 ⟩

end find_R_value_l376_376083


namespace sum_of_x_given_gx_eq_6_l376_376485

def g (x : ℝ) : ℝ :=
if x < 0 then 15 * x + 25 else 3 * x - 9

theorem sum_of_x_given_gx_eq_6 : 
  g x = 6 → (x = -19/15 ∨ x = 5) ∧ (-19/15 + 5 = 56/15) := 
by
  sorry

end sum_of_x_given_gx_eq_6_l376_376485


namespace sufficient_but_not_necessary_condition_l376_376913

theorem sufficient_but_not_necessary_condition 
    (a : ℝ) (h_pos : a > 0)
    (h_line : ∀ x y, 2 * a * x - y + 2 * a^2 = 0)
    (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / 4 = 1) :
    (a ≥ 2) → 
    (∀ x y, ¬ (2 * a * x - y + 2 * a^2 = 0 ∧ x^2 / a^2 - y^2 / 4 = 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l376_376913


namespace construct_point_M_l376_376468

variables (A B C P Q M : Point)
variables (hABC : acute_angled_triangle A B C)
variables (hAP : is_altitude A P C B)
variables (hBQ : is_altitude B Q A C)
variables (hM : M ∈ segment A B)

theorem construct_point_M : ∃ M ∈ segment A B, ∠A Q M = ∠B P M := by
  sorry

end construct_point_M_l376_376468


namespace set_intersection_subset_condition_l376_376402

-- Define the sets A and B
def A (x : ℝ) : Prop := 1 < x - 1 ∧ x - 1 ≤ 4
def B (a : ℝ) (x : ℝ) : Prop := x < a

-- First proof problem: A ∩ B = {x | 2 < x < 3}
theorem set_intersection (a : ℝ) (x : ℝ) (h_a : a = 3) :
  A x ∧ B a x ↔ 2 < x ∧ x < 3 :=
by
  sorry

-- Second proof problem: a > 5 given A ⊆ B
theorem subset_condition (a : ℝ) :
  (∀ x, A x → B a x) ↔ a > 5 :=
by
  sorry

end set_intersection_subset_condition_l376_376402


namespace rational_solutions_zero_l376_376513

theorem rational_solutions_zero (x y z : ℚ) (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by 
  sorry

end rational_solutions_zero_l376_376513


namespace area_F1PF2Q_l376_376824

noncomputable def hyperbola_directrix (x : ℝ) := (x = 3/2)
noncomputable def hyperbola_asymptote (x y : ℝ) := (y = (sqrt 3 / 3) * x) ∨ (y = -(sqrt 3 / 3) * x)

def point_P := (3/2, sqrt 3 / 2)
def point_Q := (3/2, -sqrt 3 / 2)
def foci_F1 := (-2, 0)
def foci_F2 := (2, 0)

theorem area_F1PF2Q : 
  let P := point_P
  let Q := point_Q
  let F1 := foci_F1
  let F2 := foci_F2
  hyperbola_directrix P.1 → (hyperbola_asymptote P.1 P.2 ∧ hyperbola_asymptote Q.1 Q.2) →
  (∃ area : ℝ, area = 2*sqrt 3 ∧
  quadrilateral_area P Q F1 F2 = area) :=
by
  sorry

end area_F1PF2Q_l376_376824


namespace minimize_function_l376_376350

noncomputable def f (x : ℝ) : ℝ := x - 4 + 9 / (x + 1)

theorem minimize_function : 
  (∀ x : ℝ, x > -1 → f x ≥ 1) ∧ (f 2 = 1) :=
by 
  sorry

end minimize_function_l376_376350


namespace running_speed_l376_376604

theorem running_speed
  (walking_speed : Float)
  (walking_time : Float)
  (running_time : Float)
  (distance : Float) :
  walking_speed = 8 → walking_time = 3 → running_time = 1.5 → distance = walking_speed * walking_time → 
  (distance / running_time) = 16 :=
by
  intros h_walking_speed h_walking_time h_running_time h_distance
  sorry

end running_speed_l376_376604


namespace expand_polynomial_l376_376336

variable (x : ℝ)

theorem expand_polynomial :
  (7 * x - 3) * (2 * x ^ 3 + 5 * x ^ 2 - 4) = 14 * x ^ 4 + 29 * x ^ 3 - 15 * x ^ 2 - 28 * x + 12 := by
  sorry

end expand_polynomial_l376_376336


namespace find_a_of_max_value_l376_376810

theorem find_a_of_max_value (a : ℝ) 
  (h₀ : ∀ x ∈ set.Icc (0 : ℝ) 2, f x ≤ 1) 
  (h₁ : x = 0 ∨ x = 2 → f x = 1) :
  a = 1 :=
sorry

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x - a

end find_a_of_max_value_l376_376810


namespace sum_first_ten_terms_arithmetic_sequence_l376_376611

theorem sum_first_ten_terms_arithmetic_sequence (a₁ d : ℤ) (h₁ : a₁ = -3) (h₂ : d = 4) : 
  let a₁₀ := a₁ + (9 * d)
  let S := ((a₁ + a₁₀) / 2) * 10
  S = 150 :=
by
  subst h₁
  subst h₂
  let a₁₀ := -3 + (9 * 4)
  let S := ((-3 + a₁₀) / 2) * 10
  sorry

end sum_first_ten_terms_arithmetic_sequence_l376_376611


namespace average_minutes_heard_l376_376663

theorem average_minutes_heard :
  ∀ (total_duration : ℕ) (total_audience : ℕ)
    (pct_heard_entire : ℕ) (pct_slept : ℕ) 
    (pct_heard_quarter : ℕ) (pct_heard_three_quarters : ℕ),
  total_duration = 90 → total_audience = 100 →
  pct_heard_entire = 30 → pct_slept = 15 →
  pct_heard_quarter = 50 / 2 → pct_heard_three_quarters = 50 / 2 →
  let heard_entire := pct_heard_entire * total_audience / 100 * total_duration in
  let slept_through := pct_slept * total_audience / 100 * 0 in
  let heard_one_quarter := pct_heard_quarter * total_audience / 100 * (total_duration / 4) in
  let heard_three_quarters := pct_heard_three_quarters * total_audience / 100 * (total_duration * 3 / 4) in
  let total_heard := heard_entire + slept_through + heard_one_quarter + heard_three_quarters in
  let average_heard := total_heard / total_audience in
  average_heard = 52 :=
begin
  assume total_duration total_audience
    pct_heard_entire pct_slept 
    pct_heard_quarter pct_heard_three_quarters,
  intro h_duration h_audience
    h_pct_heard_entire h_pct_slept
    h_pct_heard_quarter h_pct_heard_three_quarters,
  let heard_entire := pct_heard_entire * total_audience / 100 * total_duration,
  let slept_through := pct_slept * total_audience / 100 * 0,
  let heard_one_quarter := pct_heard_quarter * total_audience / 100 * (total_duration / 4),
  let heard_three_quarters := pct_heard_three_quarters * total_audience / 100 * (total_duration * 3 / 4),
  let total_heard := heard_entire + slept_through + heard_one_quarter + heard_three_quarters,
  let average_heard := total_heard / total_audience,
  have h_heard_entire : heard_entire = 2700, from sorry,
  have h_slept_through : slept_through = 0, from sorry,
  have h_heard_one_quarter : heard_one_quarter = 630, from sorry,
  have h_heard_three_quarters : heard_three_quarters = 1822.5, from sorry,
  have h_total_heard : total_heard = 5152.5, from sorry,
  have h_average_heard : average_heard = 51.525, from sorry,
  exact sorry
end

end average_minutes_heard_l376_376663


namespace Mike_ride_distance_l376_376147

theorem Mike_ride_distance 
  (M : ℕ)
  (total_cost_Mike : ℝ)
  (total_cost_Annie : ℝ)
  (h1 : total_cost_Mike = 4.50 + 0.30 * M)
  (h2: total_cost_Annie = 15.00)
  (h3: total_cost_Mike = total_cost_Annie) : 
  M = 35 := 
by
  sorry

end Mike_ride_distance_l376_376147


namespace local_minimum_at_one_l376_376809

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 2 * x^2 + a^2 * x

theorem local_minimum_at_one (a : ℝ) (hfmin : ∀ x : ℝ, deriv (f a) x = 3 * a * x^2 - 4 * x + a^2) (h1 : f a 1 = f a 1) : a = 1 :=
sorry

end local_minimum_at_one_l376_376809


namespace joined_in_fifth_verse_l376_376269

theorem joined_in_fifth_verse (choir_size : ℕ) (half_sang_first : ℕ) (third_sang_second : ℕ) (quarter_sang_third : ℕ) (fifth_sang_fourth : ℕ) (remain_fifth : ℕ) :
  choir_size = 60 →
  half_sang_first = choir_size / 2 →
  third_sang_second = (choir_size - half_sang_first) / 3 →
  quarter_sang_third = (choir_size - half_sang_first - third_sang_second) / 4 →
  fifth_sang_fourth = (choir_size - half_sang_first - third_sang_second - quarter_sang_third) / 5 →
  remain_fifth = (choir_size - half_sang_first - third_sang_second - quarter_sang_third - fifth_sang_fourth) →
  remain_fifth = 12 :=
by {
  intros,
  sorry
}

end joined_in_fifth_verse_l376_376269


namespace sequence_4951_l376_376451

theorem sequence_4951 :
  (∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, 0 < n → a (n + 1) = a n + n) ∧ a 100 = 4951) :=
sorry

end sequence_4951_l376_376451


namespace hyperbola_asymptote_angle_proof_l376_376046

noncomputable def hyperbola_asymptote_angle (a b : ℝ) (h : a > b) (eccentricity : ℝ) 
  (eccentricity_eq : eccentricity = (2 * Real.sqrt 3) / 3) : Prop :=
let e := eccentricity in
let lnhyp := λ x y : ℝ, (x^2)/(a^2) - (y^2)/(b^2) = 1 in
let angle := Real.arctan (Real.sqrt 3) in
lnhyp (a * cos angle) (b * sin angle) ∧ e = (2 * Real.sqrt 3) / 3 ∧ angle = π / 3

theorem hyperbola_asymptote_angle_proof (a b : ℝ) (h : a > b) 
  (eccentricity : ℝ) (eccentricity_eq : eccentricity = (2 * Real.sqrt 3) / 3) 
  : hyperbola_asymptote_angle a b h eccentricity eccentricity_eq :=
by 
  sorry

end hyperbola_asymptote_angle_proof_l376_376046


namespace not_relevant_line_2x_minus_y_plus_1_l376_376744

noncomputable def distance_point_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  |A * p.1 + B * p.2 + C| / real.sqrt (A ^ 2 + B ^ 2)

def is_relevant_line (M : ℝ × ℝ) (l : ℝ × ℝ × ℝ) : Prop :=
  distance_point_line M l.1 l.2 l.3 ≤ 4

theorem not_relevant_line_2x_minus_y_plus_1 (M : ℝ × ℝ) :
  M = (5, 0) → ¬ is_relevant_line M (2, -1, 1) :=
by {
  intro hM,
  rw is_relevant_line,
  unfold distance_point_line,
  simp [hM],
  have : |2 * 5 + (-1) * 0 + 1| = 11, by norm_num,
  have : real.sqrt (2 ^ 2 + (-1) ^ 2) = real.sqrt 5, by norm_num,
  rw [this, ←div_lt_iff],
  { norm_num at *,
    exact_mod_cast (4 < 11 / real.sqrt 5), sorry },
  { apply real.sqrt_pos.2,
    norm_num, },
}

end not_relevant_line_2x_minus_y_plus_1_l376_376744


namespace problem_statement_l376_376415

open Complex

noncomputable def count_satisfying_n : ℕ :=
  (Finset.range 2012).filter (λ n => (1 + I)^(2 * n) = (2 : ℂ)^ n * I).card

theorem problem_statement :
  count_satisfying_n = 503 :=
  sorry

end problem_statement_l376_376415


namespace sin_405_eq_sqrt2_div2_l376_376313

theorem sin_405_eq_sqrt2_div2 : Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_405_eq_sqrt2_div2_l376_376313


namespace find_min_value_l376_376349

noncomputable def f : ℝ × ℝ → ℝ := λ p, 3 * p.1^2 + 6 * p.1 * p.2 + 5 * p.2^2 - 10 * p.1 - 8 * p.2

theorem find_min_value : ∃ x y : ℝ, (f (x, y) = -41 / 3) :=
by
  sorry

end find_min_value_l376_376349


namespace trig_product_l376_376689

theorem trig_product : 
  let cos_π6 := Real.cos (Real.pi / 6)
  let cos_π3 := Real.cos (Real.pi / 3)
  let sin_π6 := Real.sin (Real.pi / 6)
  let sin_π3 := Real.sin (Real.pi / 3)
  (1 + cos_π6) * (1 + cos_π3) * (1 + -cos_π3) * (1 + -cos_π6) = 3 / 16 :=
by 
  -- We introduce the conditions in the problem
  have h1 : cos_π6 = Real.cos (Real.pi / 6) := rfl
  have h2 : cos_π3 = Real.cos (Real.pi / 3) := rfl
  have h3 : cos (5 * Real.pi / 6) = -cos (Real.pi / 6) := by sorry
  have h4 : cos (2 * Real.pi / 3) = -cos (Real.pi / 3) := by sorry 
  have h5 : sin (Real.pi / 6) = 1 / 2 := by sorry
  have h6 : sin (Real.pi / 3) = math.sqrt 3 / 2 := by sorry
  -- The rest of the proof
  sorry

end trig_product_l376_376689


namespace minimal_positive_period_f_max_value_g_on_interval_l376_376041

def f (x : ℝ) : ℝ := sin x * cos x - sqrt 3 * cos (x + real.pi) * cos x

theorem minimal_positive_period_f :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ real.pi) :=
sorry

def g (x : ℝ) : ℝ := (λ x, sin (2 * (x - real.pi / 4) - real.pi / 6) + sqrt 3) x

theorem max_value_g_on_interval :
  ∀ x, 0 ≤ x ∧ x ≤ real.pi / 4 → g x ≤ sqrt 3 + sqrt 3 / 2 ∧ 
  (∃ max_x, 0 ≤ max_x ∧ max_x ≤ real.pi / 4 ∧ g max_x = sqrt 3 + sqrt 3 / 2) :=
sorry

end minimal_positive_period_f_max_value_g_on_interval_l376_376041


namespace incorrect_statement_C_l376_376670

theorem incorrect_statement_C
  (hA : ∀ metals (M : metals), M.conducts_electricity → iron(M.conducts_electricity))
  (hB : ∀ data (x : data), variance(x) = 4 → standard_deviation(−3 * x + 2015) = 6)
  (hD : ∀ vars (y : vars) (x : vars), correlation_coefficient(y, x) = -0.9362 → strong_linear_correlation(y, x)) :
  ∀ regression_model (R2 : regression_model), better_fitting_effect(smaller(R2)) → incorrect(C) :=
by
  sorry

end incorrect_statement_C_l376_376670


namespace find_f_prime_at_2_l376_376487

noncomputable def f (x : ℝ) := x^2 + 2 * x * f' 1
noncomputable def f' (x : ℝ) := deriv f x

theorem find_f_prime_at_2 : f' 2 = 0 :=
by
  -- Proof will be inserted here
  sorry

end find_f_prime_at_2_l376_376487


namespace smallest_n_l376_376026

theorem smallest_n (m n : ℕ) (r : ℝ) (h1 : (real.sqrt (real.sqrt m)) = n + r)
  (h2 : n > 0)
  (h3 : 0 < r ∧ r < 1 / 1000) : 
  n = 19 := 
sorry

end smallest_n_l376_376026


namespace prob_in_ellipse_l376_376904

/-- Define base-5 representation conditions for 2014 --/
def is_base_5_representation (n : ℕ) (digits : list ℕ) : Prop :=
  list.sum (list.map_with_index (λ i a, a * 5^i) digits.reverse) = n

/-- Identify the given digits for the number 2014 converted to base-5 --/
def digits_2014 : list ℕ := [4, 2, 0, 1, 3]

lemma base_5_rep_2014 : is_base_5_representation 2014 digits_2014 :=
by {
  rw [is_base_5_representation, digits_2014],
  norm_num,
}

/-- Define the ellipse equation as a predicate --/
def ellipse (a b : ℕ) : Prop :=
  (a : ℚ)^2 / 16 + (b : ℚ)^2 / 9 ≤ 1

/-- Define the total possible points from the digits --/
def points := list.product digits_2014 digits_2014

/-- We calculate the number of points within the ellipse --/
def points_in_ellipse := (points.filter (λ p, ellipse p.1 p.2)).length

/-- We prove the probability of selecting a point within the ellipse is 11/25 --/
theorem prob_in_ellipse : (points_in_ellipse : ℚ) / points.length = 11 / 25 :=
by {
  sorry
}

end prob_in_ellipse_l376_376904


namespace batting_difference_l376_376537

theorem batting_difference (avg : ℕ) (n : ℕ) (avg_excl: ℕ) (n_excl : ℕ) (highest : ℕ) (lowest : ℕ) :
  (avg = 60) -> (n = 46) -> (avg_excl = 58) -> (n_excl = 44) -> (highest = 199) -> 
  let total_runs := avg * n in
  let total_runs_excl := avg_excl * n_excl in
  let sum_high_low := total_runs - total_runs_excl in
  lowest = sum_high_low - highest ->
  (highest - lowest = 190) :=
by
  intros
  let total_runs := avg * n
  let total_runs_excl := avg_excl * n_excl
  let sum_high_low := total_runs - total_runs_excl
  have h_l := sum_high_low - highest
  sorry

end batting_difference_l376_376537


namespace count_symmetrical_numbers_l376_376152

def is_valid_digit (d : Nat) : Prop := d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 6 ∨ d = 9

def is_symmetrical (n : List Nat) : Prop :=
  n.length = 9 ∧
  is_valid_digit n.head ∧
  n.head ≠ 0 ∧
  n.head = n.reverse.head ∧
  n[1] = n.reverse[1] ∧
  n[2] = n.reverse[2] ∧
  n[3] = n.reverse[3] ∧
  n[4] = n[4] ∧
  n.all is_valid_digit 

theorem count_symmetrical_numbers : ∃ n, n = 1500 ∧
  n = List.length (List.filter is_symmetrical (List.range (10^9))) := 
sorry

end count_symmetrical_numbers_l376_376152


namespace green_paint_quarts_l376_376360

theorem green_paint_quarts (blue_green_ratio : ℕ × ℕ) (blue_quarts : ℕ) 
  (h_ratio : blue_green_ratio = (5, 3)) (h_blue_quarts : blue_quarts = 10) : 
  let parts_per_quart := blue_quarts / blue_green_ratio.1 in
  blue_green_ratio.2 * parts_per_quart = 6 :=
by
  sorry

end green_paint_quarts_l376_376360


namespace cubes_painted_prob_l376_376274

variable {n : ℕ} (n = 5)

/- Define the cube and properties about painted faces -/
def cube_volume := n^3
def three_painted_faces := 8
def one_painted_face_edges := 27
def total_combinations := Nat.choose 125 2
def successful_combinations := 8 * 27
def desired_probability := successful_combinations / total_combinations

theorem cubes_painted_prob :
  desired_probability = (24 / 875) := 
sorry

end cubes_painted_prob_l376_376274


namespace no_such_p_l376_376422

theorem no_such_p : ¬ ∃ p : ℕ, p > 0 ∧ (∃ k : ℤ, 4 * p + 35 = k * (3 * p - 7)) :=
by
  sorry

end no_such_p_l376_376422


namespace y_intercept_of_line_l376_376606

theorem y_intercept_of_line : 
  ∀ (x y : ℝ), 3 * x - 5 * y = 7 → y = -7 / 5 :=
by
  intro x y h
  sorry

end y_intercept_of_line_l376_376606


namespace b_should_pay_360_l376_376621

theorem b_should_pay_360 :
  let total_cost : ℝ := 870
  let a_horses  : ℝ := 12
  let a_months  : ℝ := 8
  let b_horses  : ℝ := 16
  let b_months  : ℝ := 9
  let c_horses  : ℝ := 18
  let c_months  : ℝ := 6
  let a_horse_months := a_horses * a_months
  let b_horse_months := b_horses * b_months
  let c_horse_months := c_horses * c_months
  let total_horse_months := a_horse_months + b_horse_months + c_horse_months
  let cost_per_horse_month := total_cost / total_horse_months
  let b_cost := b_horse_months * cost_per_horse_month
  b_cost = 360 :=
by sorry

end b_should_pay_360_l376_376621


namespace presidency_meeting_ways_l376_376272

def num_ways_arrange_meeting (club_members : ℕ) (schools : ℕ) (chosen_reps_host_school : ℕ) (chosen_reps_other_schools : ℕ) : ℕ :=
  (schools * (choose club_members chosen_reps_host_school) * (choose club_members chosen_reps_other_schools) * (choose club_members chosen_reps_other_schools))

theorem presidency_meeting_ways :
  num_ways_arrange_meeting 6 3 3 1 = 2160 := by 
  sorry

end presidency_meeting_ways_l376_376272


namespace polar_not_one_to_one_correspondence_l376_376614

theorem polar_not_one_to_one_correspondence :
  ¬ ∃ f : ℝ × ℝ → ℝ × ℝ, (∀ p1 p2 : ℝ × ℝ, f p1 = f p2 → p1 = p2) ∧
  (∀ q : ℝ × ℝ, ∃ p : ℝ × ℝ, q = f p) :=
by
  sorry

end polar_not_one_to_one_correspondence_l376_376614


namespace additional_bags_at_max_weight_l376_376594

/-
Constants representing the problem conditions.
-/
def num_people : Nat := 6
def bags_per_person : Nat := 5
def max_weight_per_bag : Nat := 50
def total_weight_capacity : Nat := 6000

/-
Calculate the total existing luggage weight.
-/
def total_existing_bags : Nat := num_people * bags_per_person
def total_existing_weight : Nat := total_existing_bags * max_weight_per_bag
def remaining_weight_capacity : Nat := total_weight_capacity - total_existing_weight

/-
The proof statement asserting that given the conditions, 
the airplane can hold 90 more bags at maximum weight.
-/
theorem additional_bags_at_max_weight : remaining_weight_capacity / max_weight_per_bag = 90 := by
  sorry

end additional_bags_at_max_weight_l376_376594


namespace circumcircle_through_center_l376_376371

open EuclideanGeometry

def is_circumcircle (A B C O : Point) : Prop :=
  ∃ k : circle, circle.contains k O ∧ triangle_circumcircle k A B C

theorem circumcircle_through_center
  (O A B C : Point)
  (k : circle)
  (hO : k.center = O)
  (hA : k.contains A)
  (hB : k.contains B)
  (l : Line)
  (h_perpendicular : perpendicular l (Line.mk O A))
  (hA_on_l : A ∈ l)
  (B' : Point)
  (hB'_reflection : reflection B l B')
  (ray_AB' : Ray A B')
  (hC : C ∈ k ∧ C ∈ ray_AB') :
  is_circumcircle A B C O :=
sorry

end circumcircle_through_center_l376_376371


namespace max_intersection_points_of_polynomials_l376_376607

def p (x : ℝ) : ℝ := sorry  -- Placeholder for a 4th degree polynomial function
def q (x : ℝ) : ℝ := sorry  -- Placeholder for a 5th degree polynomial function

theorem max_intersection_points_of_polynomials : 
  (leading_coeffs p = 1 ∧ degree p = 4) ∧ (leading_coeffs q = 1 ∧ degree q = 5) →
  ∃ k : ℕ, k ≤ 5 ∧ 
  ∀ x : ℝ, p x = q x → x = k := 
sorry

end max_intersection_points_of_polynomials_l376_376607


namespace coprime_set_existence_l376_376741

-- Definitions based on the conditions and questions
def pairwise_coprime (a : List ℕ) : Prop :=
  ∀ i j : ℕ, i < a.length → j < a.length → i ≠ j → Nat.gcd (a.get i) (a.get j) = 1

-- Lean statement of the problem
theorem coprime_set_existence (n : ℕ) (a : List ℕ) 
  (h_n_pos : 3 ≤ n) (h_length : a.length = n) (h_pairwise : pairwise_coprime a) :
  (∃ (signs : List ℤ), signs.length = n ∧ signs.sum = 0 ∧ signs.forall (λ x => x = 1 ∨ x = -1)) →
  (n = 3 ↔ ∃ (b : List ℕ), b.length = n ∧ 
    ∀ k : ℕ, pairwise_coprime (List.map (λ (i : ℕ) => (b.get i) + k * (a.get i)) (List.finRange n))) :=
by
  sorry

end coprime_set_existence_l376_376741


namespace candidate_B_votes_l376_376271

-- Given conditions
variables (A_votes B_votes C_votes D_votes E_votes : ℕ)
variables (total_votes : ℕ)
variables (same_CD_votes : C_votes = D_votes)
variables (highest_A_votes : A_votes = 25)
variables (fewest_E_votes : E_votes = 4)
variables (total_students : total_votes = 46)
variables (second_highest_B_votes : B_votes > C_votes)

-- Proof problem: Prove that B received 7 votes given the conditions
theorem candidate_B_votes :
  total_votes = A_votes + B_votes + C_votes + D_votes + E_votes →
  same_CD_votes →
  highest_A_votes →
  fewest_E_votes →
  B_votes > C_votes →
  B_votes = 7 :=
by
  sorry

end candidate_B_votes_l376_376271


namespace least_integer_k_for_bk_is_integer_l376_376930

theorem least_integer_k_for_bk_is_integer :
  ∃ k : ℕ, k > 1 ∧
  (∀ (n : ℕ) (hn: 1 ≤ n),
    let b : ℕ → ℝ := λ n, if n = 1 then 2 else b 1 + real.log (∏ i in finset.range (n - 1), ((2*i+3)/(2*i+2))) / real.log 7 
    in b k ∈ ℤ) ∧ k = 23 :=
by
  sorry

end least_integer_k_for_bk_is_integer_l376_376930


namespace find_value_of_f_at_2_l376_376012

def power_func (f : ℝ → ℝ) :=
  ∃ α : ℝ, ∀ x : ℝ, f(x) = x^α

theorem find_value_of_f_at_2 (f : ℝ → ℝ) 
   (h1 : power_func f) 
   (h2 : f (real.sqrt 2) = 2 * real.sqrt 2)
   : f 2 = 8 := 
sorry

end find_value_of_f_at_2_l376_376012


namespace point_m_coordinates_l376_376497

def move_left (x : ℤ) (n : ℤ) : ℤ := x - n
def move_up (y : ℤ) (n : ℤ) : ℤ := y + n

theorem point_m_coordinates :
  let P := (-1, 2) in
  let moved_left := (move_left P.1 2, P.2) in
  let M := (moved_left.1, move_up moved_left.2 1) in
  M = (-3, 3) :=
by
  let P := (-1, 2)
  let moved_left := (move_left P.1 2, P.2)
  let M := (moved_left.1, move_up moved_left.2 1)
  show M = (-3, 3)
  sorry

end point_m_coordinates_l376_376497


namespace set_elements_l376_376947

def is_divisor (a b : ℤ) : Prop := ∃ k : ℤ, b = k * a

theorem set_elements:
  {x : ℤ | ∃ d : ℤ, is_divisor d 12 ∧ d = 6 - x ∧ x ≥ 0} = 
  {0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 18} :=
by {
  sorry
}

end set_elements_l376_376947


namespace find_a_l376_376366

noncomputable def f (x : ℝ) := x + 100 / x

theorem find_a (a : ℝ) (h_positive : a > 0) 
  (m1 : ℝ) (m2 : ℝ)
  (h_m1 : ∀ x ∈ Set.Ioc 0 a, f x ≥ m1 ∧ ∃ y ∈ Set.Ioc 0 a, f y = m1)
  (h_m2 : ∀ x ∈ Set.Ico a (Real.infinity), f x ≥ m2 ∧ ∃ y ∈ Set.Ico a (Real.infinity), f y = m2)
  (h_product : m1 * m2 = 2020) : a = 1 ∨ a = 100 := 
sorry

end find_a_l376_376366


namespace function_is_quadratic_l376_376241

-- Definitions for the conditions
def is_quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0) ∧ ∀ (x : ℝ), f x = a * x^2 + b * x + c

-- The function to be proved as a quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

-- The theorem statement: f must be a quadratic function
theorem function_is_quadratic : is_quadratic_function f :=
  sorry

end function_is_quadratic_l376_376241


namespace projection_matrix_correct_l376_376119

-- Define a type for 3D vectors
def vec3 := ℝ × ℝ × ℝ

-- Define the dot product for 3D vectors
def dot_product (v1 v2 : vec3) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the normal vector
def n : vec3 := (2, -1, 1)

-- Define the projection matrix
noncomputable def Q : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.vecℚ
    [[1 / 3, 1 / 6, -1 / 6],
     [1 / 3, 5 / 6, 1 / 6],
     [-1 / 3, 5 / 6, 5 / 6]]

-- Define the projection of any vector v onto the plane defined by n
def projection (v : vec3) : vec3 :=
  -- Compute the dot products
  let num := dot_product v n
  let denom := dot_product n n
  -- Compute the projection onto the normal vector
  let proj_n := (num / denom) *: n
  -- Subtract this from the original vector to get the projection onto the plane
  (v.1 - proj_n.1, v.2 - proj_n.2, v.3 - proj_n.3)

-- Main theorem statement
theorem projection_matrix_correct :
  ∀v : vec3, Matrix.mulVec Q v = projection v :=
sorry

end projection_matrix_correct_l376_376119


namespace sum_odd_multiples_of_5_from_1_to_60_l376_376626

def isOdd (n : ℤ) : Prop := ¬ Even n

def isMultipleOf5 (n : ℤ) : Prop := ∃ k : ℤ, n = 5 * k

def oddMultiplesOf5 (n : ℤ) : Prop := isOdd n ∧ isMultipleOf5 n

def filterOddMultiplesOf5UpTo60 : List ℤ :=
  List.filter oddMultiplesOf5 (List.range' 1 60)

theorem sum_odd_multiples_of_5_from_1_to_60 :
  ∑ n in filterOddMultiplesOf5UpTo60, n = 180 := by
  sorry

end sum_odd_multiples_of_5_from_1_to_60_l376_376626


namespace asymptotes_of_hyperbola_l376_376541

theorem asymptotes_of_hyperbola : 
  (∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1 → y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intro x y h
  sorry

end asymptotes_of_hyperbola_l376_376541


namespace polynomial_has_n_roots_in_circle_l376_376862

open Complex

noncomputable def f (z : ℂ) (n : ℕ) (a : Fin n → ℂ) : ℂ :=
  z^n + ∑ i in Finset.range n, a ⟨i, by linarith⟩ * z^(n-1-i)

def max_modulus (n : ℕ) (a : Fin n → ℂ) : ℝ :=
  Finset.univ.sup (λ i, complex.abs (a i))

theorem polynomial_has_n_roots_in_circle (n : ℕ) (a : Fin (n+1) → ℂ) :
  ∃ roots : Multiset ℂ, (roots.card = n) ∧ ∀ z ∈ roots, ∣z∣ < 1 + max_modulus (n + 1) a :=
sorry

end polynomial_has_n_roots_in_circle_l376_376862


namespace circle_geometry_l376_376061

theorem circle_geometry (Q P A C : Point) (AB CD : Line) (h1 : Circle Q) 
  (h2 : Diameter AB) (h3 : Diameter CD) (h4 : Perpendicular AB CD)
  (h5 : OnPoint P (line_through Q A)) (h6 : ∠ Q P C = 45°) :
  (length_segment P Q) / (length_segment A Q) = (√2 / 2) := 
sorry

end circle_geometry_l376_376061


namespace f_decreasing_on_positive_reals_f_max_min_on_interval_l376_376393

-- Definition of the function f
def f (x : ℝ) : ℝ := 2 / (2^x - 1)

-- Theorem statements
theorem f_decreasing_on_positive_reals : ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 < x2 → f(x1) > f(x2) :=
by sorry

theorem f_max_min_on_interval :
  (f 1 = 2) ∧ (f (log 2 6) = 2 / 5) :=
by sorry

end f_decreasing_on_positive_reals_f_max_min_on_interval_l376_376393


namespace AC_is_7_07_l376_376889

namespace Mathlib

noncomputable def AC_length (AD BC: ℝ) (BAC_deg ADB_deg: ℝ) : ℝ :=
  let BAC := BAC_deg * real.pi / 180
  let ADB := ADB_deg * real.pi / 180
  let ABC := real.pi - (BAC + ADB)
  real.sqrt (AD^2 + BC^2 - 2 * AD * BC * real.cos ABC)

theorem AC_is_7_07
  (AD BC : ℕ) (BAC_deg ADB_deg : ℕ)
  (hAD : AD = 5)
  (hBC : BC = 7)
  (hBAC : BAC_deg = 60)
  (hADB : ADB_deg = 50) :
  AC_length AD BC BAC_deg ADB_deg = real.sqrt 50.06 :=
by
  rw [hAD, hBC, hBAC, hADB]
  norm_num
  sorry

end Mathlib

end AC_is_7_07_l376_376889


namespace gcd_of_gx_and_x_l376_376479

theorem gcd_of_gx_and_x (x : ℕ) (h : 7200 ∣ x) : Nat.gcd ((5 * x + 3) * (11 * x + 2) * (17 * x + 5) * (4 * x + 7)) x = 30 := 
by 
  sorry

end gcd_of_gx_and_x_l376_376479


namespace general_formula_sum_Tn_less_half_l376_376038

-- Defining the arithmetic sequence with an undefined common difference and initial term
def arithmetic_seq (a d : ℤ) : ℕ → ℤ
| 0 => a
| (n+1) => (arithmetic_seq a d n) + d

-- Defining the sum of the first n terms of the arithmetic sequence
def S_n (a d : ℤ) (n : ℕ) : ℤ :=
match n with
| 0 => 0
| (n+1) => (n.succ : ℤ) * a + ((n : ℤ * (n.succ : ℤ) / 2) * d)

-- Defining the sequence for the reciprocal of the product of consecutive terms
def reciprocal_product_seq (a d: ℤ) (n: ℕ) : ℚ :=
1 / ((arithmetic_seq a d n) * (arithmetic_seq a d (n + 1)))

-- Defining the sum of the first n terms of the reciprocal product sequence
def T_n (a d : ℤ) (n : ℕ) : ℚ :=
(nat_range n).sum (reciprocal_product_seq a d)

theorem general_formula 
  (a d : ℤ) 
  (h1 : arithmetic_seq a d 4 = 7)
  (h2 : arithmetic_seq a d 1 * arithmetic_seq a d 5 = (arithmetic_seq a d 2)^2) :
  ∀ n, arithmetic_seq a d n = 2 * n - 1 :=
sorry

theorem sum_Tn_less_half
  (a d : ℤ)
  (h1 : arithmetic_seq a d 4 = 7)
  (h2 : arithmetic_seq a d 1 * arithmetic_seq a d 5 = (arithmetic_seq a d 2)^2):
  ∀ n, T_n a d n < 1 / 2 :=
sorry

end general_formula_sum_Tn_less_half_l376_376038


namespace green_balls_count_l376_376927

theorem green_balls_count (b g : ℕ) (h1 : b = 15) (h2 : 5 * g = 3 * b) : g = 9 :=
by
  sorry

end green_balls_count_l376_376927


namespace find_smallest_natural_number_l376_376724

theorem find_smallest_natural_number :
  ∃ x : ℕ, (2 * x = b^2 ∧ 3 * x = c^3) ∧ (∀ y : ℕ, (2 * y = d^2 ∧ 3 * y = e^3) → x ≤ y) := by
  sorry

end find_smallest_natural_number_l376_376724


namespace intersection_lines_equal_angles_iff_l376_376598

variable {A B C D : Type} [euclidean_space ℝ A] [euclidean_space ℝ B] [euclidean_space ℝ C] [euclidean_space ℝ D]

def tetrahedron (A B C D : Type) : Prop := 
is_tetrahedron A B C D  -- Assuming is_tetrahedron is appropriately defined within the Euclidean context.

noncomputable def are_lines_of_intersection_forming_six_equal_angles (A B C D : Type)
(plane_tangent_to_sphere_at_A : Type) : Prop := sorry -- Placeholder to define the lines of intersection forming six equal angles.

noncomputable def condition_for_equal_angles (A B C D : Type) : Prop := 
(distance A B) * (distance C D) = (distance A C) * (distance B D) ∧ (distance A C) * (distance B D) = (distance A D) * (distance B C)

theorem intersection_lines_equal_angles_iff 
{A B C D : Type} [euclidean_space ℝ A] [euclidean_space ℝ B] [euclidean_space ℝ C] [euclidean_space ℝ D]
(h_tetrahedron : tetrahedron A B C D)
(plane_tangent_to_sphere_at_A : Type) :
  are_lines_of_intersection_forming_six_equal_angles A B C D plane_tangent_to_sphere_at_A ↔
  condition_for_equal_angles A B C D :=
by
  sorry

end intersection_lines_equal_angles_iff_l376_376598


namespace avg_values_l376_376343

theorem avg_values (z : ℝ) : (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end avg_values_l376_376343


namespace debt_compounding_interest_l376_376843

theorem debt_compounding_interest (t : ℕ) (h : ∀ n : ℕ, n < t → 1.06^n ≤ 3) : t = 19 :=
by
  sorry

end debt_compounding_interest_l376_376843


namespace find_ff_neg2_l376_376391

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else 1 - Real.log x / Real.log 2

theorem find_ff_neg2 : f (f (-2)) = 3 := 
by
  sorry

end find_ff_neg2_l376_376391


namespace trigonometric_identity_l376_376898

theorem trigonometric_identity :
  sin 21 * cos 81 - cos 21 * sin 81 = - (sqrt 3 / 2) :=
by
  sorry

end trigonometric_identity_l376_376898


namespace arithmetic_sequence_multiples_l376_376546

theorem arithmetic_sequence_multiples (a1 a8 : ℤ) (n : ℕ) (f : ℤ → ℤ) (d : ℤ) :
  a1 = 9 →
  a8 = 12 →
  ∀ n, f n = a1 + (n - 1) * d →
  ∃ k, ∀ m, (1 ≤ m ∧ m ≤ 2015) → f m = 3 * k ∧ k ≥ 0 ∧ k ≤ 287 →
  count_multiples_3 (first_2015_terms (f)) = 288 :=
by
  sorry

end arithmetic_sequence_multiples_l376_376546


namespace circumscribedCircleDiameter_is_10sqrt2_l376_376286

noncomputable def circumscribedCircleDiameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem circumscribedCircleDiameter_is_10sqrt2 :
  circumscribedCircleDiameter 10 (Real.pi / 4) = 10 * Real.sqrt 2 :=
by
  sorry

end circumscribedCircleDiameter_is_10sqrt2_l376_376286


namespace transformation_correct_l376_376943

def initial_function (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)
def transformed_function (x : ℝ) : ℝ := Real.sin (2 * x + 3 * Real.pi / 4) + 2

theorem transformation_correct : 
  ∀ x : ℝ, transformed_function x = Real.sin (2 * x + 3 * Real.pi / 4) + 2 := 
by
  intro x
  simp [transformed_function, Real.sin]
  sorry

end transformation_correct_l376_376943


namespace steven_seeds_l376_376525

def average_seeds (fruit: String) : Nat :=
  match fruit with
  | "apple" => 6
  | "pear" => 2
  | "grape" => 3
  | "orange" => 10
  | "watermelon" => 300
  | _ => 0

def fruits := [("apple", 2), ("pear", 3), ("grape", 5), ("orange", 1), ("watermelon", 2)]

def required_seeds := 420

def total_seeds (fruit_list : List (String × Nat)) : Nat :=
  fruit_list.foldr (fun (fruit_qty : String × Nat) acc =>
    acc + (average_seeds fruit_qty.fst) * fruit_qty.snd) 0

theorem steven_seeds : total_seeds fruits - required_seeds = 223 := by
  sorry

end steven_seeds_l376_376525


namespace sin_alpha_terminal_side_l376_376772

theorem sin_alpha_terminal_side (alpha : ℝ) (x y r : ℝ)
    (h1 : x = -2) (h2 : y = 4) (h3 : r = real.sqrt (x^2 + y^2)) :
    real.sin (-y / r) = 2 * real.sqrt 5 / 5 :=
by 
  sorry

end sin_alpha_terminal_side_l376_376772


namespace truck_filling_rate_l376_376903

-- Defining the conditions and the final statement to be proven.
theorem truck_filling_rate :
  ∃ (R : ℕ), 
    let stella_twinkle_hours : ℕ := 4,
        additional_workers : ℕ := 6,
        total_blocks : ℕ := 6000,
        total_hours : ℕ := 6 in
    let initial_blocks_filled := 2 * R * stella_twinkle_hours,
        additional_blocks_filled := (2 + additional_workers) * R * (total_hours - stella_twinkle_hours),
        total_blocks_filled := initial_blocks_filled + additional_blocks_filled in
    total_blocks_filled = total_blocks → R = 250 :=
begin
  sorry
end

end truck_filling_rate_l376_376903


namespace probability_of_different_parity_l376_376732

noncomputable def probability_different_parity : ℚ :=
  let total_cards := finset.range 9 -- cards numbered 1 through 9
  let total_pairs := total_cards.card.choose 2 -- total ways to choose 2 cards
  let odd_cards := {1, 3, 5, 7, 9} -- set of odd cards
  let even_cards := {2, 4, 6, 8} -- set of even cards
  let odd_even_pairs := odd_cards.card * even_cards.card -- ways to choose one odd and one even card
  (odd_even_pairs : ℚ) / (total_pairs : ℚ) -- probability

theorem probability_of_different_parity :
  probability_different_parity = 5 / 9 := sorry

end probability_of_different_parity_l376_376732


namespace max_subjects_per_teacher_l376_376658

theorem max_subjects_per_teacher (math_teachers physics_teachers chemistry_teachers min_teachers : ℕ)
  (h_math : math_teachers = 4)
  (h_physics : physics_teachers = 3)
  (h_chemistry : chemistry_teachers = 3)
  (h_min_teachers : min_teachers = 5) :
  (math_teachers + physics_teachers + chemistry_teachers) / min_teachers = 2 :=
by
  sorry

end max_subjects_per_teacher_l376_376658


namespace seats_in_stadium_l376_376976

theorem seats_in_stadium : 
  ∀ (children adults empty_seats : ℕ), 
  children = 52 → adults = 29 → empty_seats = 14 → 
  children + adults + empty_seats = 95 :=
by
  intros children adults empty_seats h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end seats_in_stadium_l376_376976


namespace common_chord_equation_l376_376775

-- Definition of the first circle
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

-- Definition of the second circle
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0

-- Proposition stating we need to prove the line equation
theorem common_chord_equation (x y : ℝ) : circle1 x y → circle2 x y → x - y = 0 :=
by
  intros h1 h2
  sorry

end common_chord_equation_l376_376775


namespace find_X_l376_376242

theorem find_X :
  (15.2 * 0.25 - 48.51 / 14.7) / X = ((13 / 44 - 2 / 11 - 5 / 66) / (5 / 2) * (6 / 5)) / (3.2 + 0.8 * (5.5 - 3.25)) ->
  X = 137.5 :=
by
  intro h
  sorry

end find_X_l376_376242


namespace intersection_size_divisor_order_l376_376101

variables {G : Type*} [Group G] [Fintype G] {H : Subgroup G} [Fintype H] {a b : G}

theorem intersection_size_divisor_order (H_le_G : H ≤ G) (a b : G) : 
  ∃ d : ℕ, (|aH ∩ Hb| = 0 ∨ d ∣ |H|) :=
sorry

end intersection_size_divisor_order_l376_376101


namespace median_of_36_consecutive_integers_l376_376578

theorem median_of_36_consecutive_integers (f : ℕ → ℤ) (h_consecutive : ∀ n : ℕ, f (n + 1) = f n + 1) 
(h_size : ∃ k, f 36 = f 0 + 35) (h_sum : ∑ i in finset.range 36, f i = 6^4) : 
(∃ m, m = f (36 / 2 - 1) ∧ m = 36) :=
by
  sorry

end median_of_36_consecutive_integers_l376_376578


namespace divides_expression_l376_376423

theorem divides_expression (x : ℕ) (hx : Even x) : 90 ∣ (15 * x + 3) * (15 * x + 9) * (5 * x + 10) :=
sorry

end divides_expression_l376_376423


namespace combined_weight_of_jake_and_sister_l376_376425

theorem combined_weight_of_jake_and_sister (j s : ℕ) (h1 : j = 188) (h2 : j - 8 = 2 * s) : j + s = 278 :=
sorry

end combined_weight_of_jake_and_sister_l376_376425


namespace general_term_sum_inequality_l376_376387

variable {α : Type*}

variables (a : ℕ → ℕ) (S : ℕ → ℕ)

axiom is_arithmetic_sequence : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Using the conditions directly
axiom a_3_eq_7 : a 3 = 7
axiom sum_S : ∀ n : ℕ, S n = ∑ i in finset.range n, a (i + 1)

-- Proving the general term of arithmetic sequence
theorem general_term : ∀ n : ℕ, a n = 2 * n + 1 := sorry

-- Prove the inequality
theorem sum_inequality (p q : ℕ) (hp : p > 0) (hq : q > 0) (hpq : p ≠ q) : 
  S (p + q) < (S (2 * p) + S (2 * q)) / 2 := sorry

end general_term_sum_inequality_l376_376387


namespace jill_paid_more_than_jack_l376_376836

-- Definitions of the problem conditions
def num_slices : ℕ := 12
def cost_plain_pizza : ℕ := 12
def cost_pepperoni_extra : ℕ := 3
def slices_pepperoni : ℕ := num_slices / 3
def slices_jill_plain : ℕ := 3
def slices_jack_plain : ℕ := num_slices - (slices_pepperoni + slices_jill_plain)

-- Cost calculations
def total_cost_pizza : ℝ := cost_plain_pizza + cost_pepperoni_extra
def cost_per_slice : ℝ := total_cost_pizza / num_slices
def jill_slices : ℕ := slices_pepperoni + slices_jill_plain
def jack_slices : ℕ := slices_jack_plain

-- Payments
def cost_jack : ℝ := jack_slices
def cost_jill : ℝ := total_cost_pizza - cost_jack

-- Conjecture to prove
theorem jill_paid_more_than_jack : cost_jill - cost_jack = 5 := by
  sorry

end jill_paid_more_than_jack_l376_376836


namespace total_annual_interest_l376_376283

def total_amount : ℝ := 4000
def P1 : ℝ := 2800
def Rate1 : ℝ := 0.03
def Rate2 : ℝ := 0.05

def P2 : ℝ := total_amount - P1
def I1 : ℝ := P1 * Rate1
def I2 : ℝ := P2 * Rate2
def I_total : ℝ := I1 + I2

theorem total_annual_interest : I_total = 144 := by
  sorry

end total_annual_interest_l376_376283


namespace goods_train_speed_l376_376644

theorem goods_train_speed
  (length_train : ℝ)
  (length_platform : ℝ)
  (time_taken : ℝ)
  (speed_kmph : ℝ)
  (h1 : length_train = 240.0416)
  (h2 : length_platform = 280)
  (h3 : time_taken = 26)
  (h4 : speed_kmph = 72.00576) :
  speed_kmph = ((length_train + length_platform) / time_taken) * 3.6 := sorry

end goods_train_speed_l376_376644


namespace terminating_decimals_count_l376_376358

theorem terminating_decimals_count :
  let n := {n : ℕ | 1 ≤ n ∧ n ≤ 419 ∧ ∃ k : ℕ, n = 21 * k}
  n.card = 19 :=
by
  sorry

end terminating_decimals_count_l376_376358


namespace number_of_B_eq_l376_376196

variable (a b : ℝ)
variable (B : ℝ)

theorem number_of_B_eq : 3 * B = a + b → B = (a + b) / 3 :=
by sorry

end number_of_B_eq_l376_376196


namespace hexagon_infinite_solutions_l376_376812

theorem hexagon_infinite_solutions :
  ∃ (a_1 a_2 a_3 a_4 a_5 a_6 : ℤ), (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 20) ∧
  (∀ (i j k : ℕ), 1 ≤ i → i < j → j < k → k ≤ 6 → a_i + a_j ≤ a_k) :=
sorry

end hexagon_infinite_solutions_l376_376812


namespace monthly_interest_rate_l376_376460

-- Define the principal amount (initial amount).
def principal : ℝ := 200

-- Define the final amount after 2 months (A).
def amount_after_two_months : ℝ := 222

-- Define the number of months (n).
def months : ℕ := 2

-- Define the monthly interest rate (r) we need to prove.
def interest_rate : ℝ := 0.053

-- Main statement to prove
theorem monthly_interest_rate :
  amount_after_two_months = principal * (1 + interest_rate)^months :=
sorry

end monthly_interest_rate_l376_376460


namespace factorization_l376_376710

theorem factorization (x : ℝ) : 
  x^6 - 4 * x^4 + 6 * x^2 - 4 = (x^2 - 1) * (x^4 - 2 * x^2 + 2) :=
by
  sorry

end factorization_l376_376710


namespace find_a_l376_376756

theorem find_a (x y : ℝ) (a : ℝ) (h1 : x = 3) (h2 : y = 2) (h3 : a * x + 2 * y = 1) : a = -1 := by
  sorry

end find_a_l376_376756


namespace miles_driven_l376_376145

theorem miles_driven (rental_fee charge_per_mile total_amount_paid : ℝ) (h₁ : rental_fee = 20.99) (h₂ : charge_per_mile = 0.25) (h₃ : total_amount_paid = 95.74) :
  (total_amount_paid - rental_fee) / charge_per_mile = 299 :=
by
  -- Placeholder for proof
  sorry

end miles_driven_l376_376145


namespace percentage_less_than_l376_376251

theorem percentage_less_than (p j t : ℝ) (h1 : j = 0.75 * p) (h2 : j = 0.80 * t) : 
  t = (1 - 0.0625) * p := 
by 
  sorry

end percentage_less_than_l376_376251


namespace ramsey_example_l376_376165

theorem ramsey_example (P : Fin 10 → Fin 10 → Prop) :
  (∀ (i j k : Fin 10), i ≠ j → i ≠ k → j ≠ k → ¬(¬P i j ∧ ¬P j k ∧ ¬P k i))
  ∨ (∀ (i j k : Fin 10), i ≠ j → i ≠ k → j ≠ k → ¬(P i j ∧ P j k ∧ P k i)) →
  (∃ (i j k l : Fin 10), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ (P i j ∧ P j k ∧ P k l ∧ P i k ∧ P j l ∧ P i l))
  ∨ (∃ (i j k l : Fin 10), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ (¬P i j ∧ ¬P j k ∧ ¬P k l ∧ ¬P i k ∧ ¬P j l ∧ ¬P i l)) :=
by
  sorry

end ramsey_example_l376_376165


namespace boys_without_calculators_l376_376879

theorem boys_without_calculators (total_boys total_students students_with_calculators girls_with_calculators : ℕ) 
    (h1 : total_boys = 20) 
    (h2 : total_students = 40) 
    (h3 : students_with_calculators = 30) 
    (h4 : girls_with_calculators = 18) : 
    (total_boys - (students_with_calculators - girls_with_calculators)) = 8 :=
by
  sorry

end boys_without_calculators_l376_376879


namespace wire_cut_square_octagon_area_l376_376291

theorem wire_cut_square_octagon_area (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (equal_area : (a / 4)^2 = (2 * (b / 8)^2 * (1 + Real.sqrt 2))) : 
  a / b = Real.sqrt ((1 + Real.sqrt 2) / 2) := 
  sorry

end wire_cut_square_octagon_area_l376_376291


namespace arithmetic_sequence_multiples_l376_376545

theorem arithmetic_sequence_multiples (a1 a8 : ℤ) (n : ℕ) (f : ℤ → ℤ) (d : ℤ) :
  a1 = 9 →
  a8 = 12 →
  ∀ n, f n = a1 + (n - 1) * d →
  ∃ k, ∀ m, (1 ≤ m ∧ m ≤ 2015) → f m = 3 * k ∧ k ≥ 0 ∧ k ≤ 287 →
  count_multiples_3 (first_2015_terms (f)) = 288 :=
by
  sorry

end arithmetic_sequence_multiples_l376_376545


namespace union_complement_eq_l376_376404

open Set

variable (U A B : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

theorem union_complement_eq (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) :
  (complement U A) ∪ B = {0, 2, 4} :=
by
  rw [hU, hA, hB]
  sorry

end union_complement_eq_l376_376404


namespace area_triangle_PQR_l376_376454

noncomputable def length_of_median {a b c : ℕ} (length_side_a : a) (length_side_b : b) (length_side_c : c) : ℕ :=
  sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)

theorem area_triangle_PQR : 
  (PQ PR PM : ℕ) 
  (PQ = 8) 
  (PR = 18) 
  (PM = 12) 
  :
  area_of_triangle PQ PR PM = 72 * sqrt 5 :=
sorry

end area_triangle_PQR_l376_376454


namespace max_gcd_consecutive_terms_seq_l376_376232

def b (n : ℕ) := (2 * n)! + n^2

theorem max_gcd_consecutive_terms_seq (n : ℕ) (hn : n ≥ 1) :
  ∃ m : ℕ, ∀ i : ℕ, i ≥ n → gcd (b i) (b (i + 1)) = m ∧ m = 1 :=
by
  sorry

end max_gcd_consecutive_terms_seq_l376_376232


namespace meat_needed_for_30_hamburgers_l376_376890

-- Definitions based on the problem conditions
def meat_per_10_hamburgers := 5  -- pounds
def waste_percentage := 20       -- percent
def hamburgers_made := 10        -- from 5 pounds of meat
def hamburgers_needed := 30

-- Calculation definitions
def effective_meat_used := meat_per_10_hamburgers * (1 - waste_percentage / 100.0)
def effective_meat_per_hamburger := effective_meat_used / hamburgers_made
def total_effective_meat_needed := effective_meat_per_hamburger * hamburgers_needed
def total_raw_meat_needed := total_effective_meat_needed / (1 - waste_percentage / 100.0)

-- Proof problem statement
theorem meat_needed_for_30_hamburgers : total_raw_meat_needed = 15 :=
by
  sorry

end meat_needed_for_30_hamburgers_l376_376890


namespace net_progress_l376_376622

-- Definitions based on conditions in the problem
def loss := 5
def gain := 9

-- Theorem: Proving the team's net progress
theorem net_progress : (gain - loss) = 4 :=
by
  -- Placeholder for proof
  sorry

end net_progress_l376_376622


namespace abigail_total_fences_l376_376999

variable (initial : ℕ) (first_rate : ℕ) (first_quicker : ℚ) 
          (second_rate : ℕ) (second_slower : ℚ) (usual_rate : ℕ)
          (first_hours : ℕ) (second_hours : ℕ) (total_hours : ℕ)
          (first_fences : ℕ) (second_fences : ℕ) (remaining_fences : ℕ)

noncomputable def abigail_fences : ℕ :=
  let first_period := (first_hours * 60) / (first_rate * ((100 - first_quicker) / 100))
  let break1 := 45
  let second_period := (second_hours * 60) / (second_rate * ((100 + second_slower) / 100))
  let break2 := 30
  let remaining_period := (total_hours * 60) - (first_hours * 60 + break1 + second_hours * 60 + break2)
  let usual_period := remaining_period / usual_rate
  initial + first_period.toNat + second_period.toNat + usual_period.toNat

theorem abigail_total_fences :
  abigail_fences 10 30 15 30 25 30 3 2 8 7 3 3 = 23 := 
  by
  simp [abigail_fences]
  sorry

end abigail_total_fences_l376_376999


namespace radius_B_eq_8_div_9_l376_376688

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Given conditions
variable (A B C D : Circle)
variable (h1 : A.radius = 1)
variable (h2 : A.radius + A.radius = D.radius)
variable (h3 : B.radius = C.radius)
variable (h4 : (A.center.1 - B.center.1)^2 + (A.center.2 - B.center.2)^2 = (A.radius + B.radius)^2)
variable (h5 : (A.center.1 - C.center.1)^2 + (A.center.2 - C.center.2)^2 = (A.radius + C.radius)^2)
variable (h6 : (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 = (B.radius + C.radius)^2)
variable (h7 : (D.center.1 - A.center.1)^2 + (D.center.2 - A.center.2)^2 = D.radius^2)

-- Prove the radius of circle B is 8/9
theorem radius_B_eq_8_div_9 : B.radius = 8 / 9 := 
by
  sorry

end radius_B_eq_8_div_9_l376_376688


namespace constant_term_binomial_expansion_l376_376000

noncomputable def a : ℝ := (2 / Real.pi) * ∫ x in -1..1, (Real.sqrt (1 - x^2) + Real.sin x)

theorem constant_term_binomial_expansion : a = 1 → 
  let f := (fun x : ℝ => (x - (a / x^2))^9) in 
  (∃ C : ℝ, ∃ r : ℕ, (C * (x ^ (9 - 3 * r)) = f x) ∧ r = 3 ∧ C = -84) :=
by
  intro ha
  let f := (fun x : ℝ => (x - (1 / x^2))^9)
  have hr : 9 - 3 * 3 = 0 := by norm_num
  use -84
  use 3
  split
  have : f = (fun x => (-1)^3 * Mathlib.Mathbin.Combinatorics.Basic.binom 9 3 * x^0) := by
    sorry
  exact (this x)
  split
  exact hr
  norm_num
  sorry

end constant_term_binomial_expansion_l376_376000


namespace distinct_four_digit_integers_with_digit_product_eight_l376_376056

theorem distinct_four_digit_integers_with_digit_product_eight : 
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (∀ (a b c d : ℕ), 10 > a ∧ 10 > b ∧ 10 > c ∧ 10 > d ∧ n = 1000 * a + 100 * b + 10 * c + d ∧ a * b * c * d = 8) ∧ (∃ (count : ℕ), count = 20 ) :=
sorry

end distinct_four_digit_integers_with_digit_product_eight_l376_376056


namespace sum_of_sides_l376_376850

open EuclideanGeometry

-- Definitions of the points and proofs
variables {A B C D O : Point}

-- Conditions
variables (h_incenter : is_incenter O A B C)
          (h_midpoint : is_midpoint D A B)
          (h_angle : ∠A O D = 90)

theorem sum_of_sides (h_incenter : is_incenter O A B C)
                     (h_midpoint : is_midpoint D A B)
                     (h_angle : ∠A O D = 90) :
  dist A B + dist B C = 3 * dist A C :=
begin
  sorry
end

end sum_of_sides_l376_376850


namespace sin_405_eq_sqrt2_div2_l376_376315

theorem sin_405_eq_sqrt2_div2 : Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_405_eq_sqrt2_div2_l376_376315


namespace point_p_locus_equation_l376_376588

noncomputable def locus_point_p (x y : ℝ) : Prop :=
  ∀ (k b x1 y1 x2 y2 : ℝ), 
  (x1^2 + y1^2 = 1) ∧ 
  (x2^2 + y2^2 = 1) ∧ 
  (3 * x1 * x + 4 * y1 * y = 12) ∧ 
  (3 * x2 * x + 4 * y2 * y = 12) ∧ 
  (1 + k^2 = b^2) ∧ 
  (y = 3 / b) ∧ 
  (x = -4 * k / (3 * b)) → 
  x^2 / 16 + y^2 / 9 = 1

theorem point_p_locus_equation :
  ∀ (x y : ℝ), locus_point_p x y → (x^2 / 16 + y^2 / 9 = 1) :=
by
  intros x y h
  sorry

end point_p_locus_equation_l376_376588


namespace decreasing_function_b_range_l376_376065

   noncomputable def f (x b : ℝ) : ℝ := - (1 / 2) * x^2 + b * Real.log (x + 2)

   theorem decreasing_function_b_range :
     ∀ b : ℝ, (∀ x : ℝ, x > -1 → deriv (λ x : ℝ, f x b) x ≤ 0) ↔ b ≤ -1 := 
   by
     sorry
   
end decreasing_function_b_range_l376_376065


namespace maximize_profit_l376_376993

-- Definitions from the conditions
def cost_price : ℝ := 16
def initial_selling_price : ℝ := 20
def initial_sales_volume : ℝ := 80
def price_decrease_per_step : ℝ := 0.5
def sales_increase_per_step : ℝ := 20

def functional_relationship (x : ℝ) : ℝ := -40 * x + 880

-- The main theorem we need to prove
theorem maximize_profit :
  (∀ x, 16 ≤ x → x ≤ 20 → functional_relationship x = -40 * x + 880) ∧
  (∃ x, 16 ≤ x ∧ x ≤ 20 ∧ (∀ y, 16 ≤ y → y ≤ 20 → 
    ((-40 * x + 880) * (x - cost_price) ≥ (-40 * y + 880) * (y - cost_price)) ∧
    (-40 * x + 880) * (x - cost_price) = 360 ∧ x = 19)) :=
by
  sorry

end maximize_profit_l376_376993


namespace find_three_digit_number_l376_376338

theorem find_three_digit_number (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) 
  : 100 * a + 10 * b + c = 5 * a * b * c → a = 1 ∧ b = 7 ∧ c = 5 :=
by
  sorry

end find_three_digit_number_l376_376338


namespace angle_CED_l376_376470

theorem angle_CED (O A B E C D : Type) [incircle : Circle O 1] 
  (diameter : diameter AB = 2 * radius O)
  (point_on_circle : E ∈ Circle O 1)
  (tangent_B : is_tangent B)
  (tangent_E : is_tangent E)
  (A_on_circle : A ∈ Circle O 1)
  (B_on_circle : B ∈ Circle O 1)
  (circle_center : center O)
  (t1 : tangent_bisect B C)
  (t2 : tangent_bisect E C)
  (point_on_tangent : tangent_intersect A E D)
  (angle_BAE : angle BAE = 43) :
  angle CED = 47 :=
sorry

end angle_CED_l376_376470


namespace power_of_two_equivalence_l376_376418

theorem power_of_two_equivalence (x : ℝ) (h : 128^7 = 16^x) : 2^-x = 1/2^(49/4) :=
by
  -- Proof goes here
  sorry

end power_of_two_equivalence_l376_376418


namespace fraction_used_to_buy_clothes_l376_376735

theorem fraction_used_to_buy_clothes :
  ∀ (total_money mom_fraction charity_fraction kept_money spent_fraction: ℝ),
  total_money = 400 →
  mom_fraction = 1 / 4 →
  charity_fraction = 1 / 5 →
  kept_money = 170 →
  spent_fraction = 1 / 8 →
  let
    money_given_to_mom := total_money * mom_fraction,
    money_left_after_mom := total_money - money_given_to_mom,
    money_given_to_charity := total_money * charity_fraction,
    money_left_after_charity := money_left_after_mom - money_given_to_charity,
    money_spent_on_clothes := money_left_after_charity - kept_money
  in
  money_spent_on_clothes / total_money = spent_fraction :=
by
  intros total_money mom_fraction charity_fraction kept_money spent_fraction
  assume 
    h1 : total_money = 400,
    h2 : mom_fraction = 1 / 4,
    h3 : charity_fraction = 1 / 5,
    h4 : kept_money = 170,
    h5 : spent_fraction = 1 / 8

  -- Definitions based on conditions
  let money_given_to_mom := total_money * mom_fraction
  let money_left_after_mom := total_money - money_given_to_mom
  let money_given_to_charity := total_money * charity_fraction
  let money_left_after_charity := money_left_after_mom - money_given_to_charity
  let money_spent_on_clothes := money_left_after_charity - kept_money

  -- Skipping the proof
  sorry

end fraction_used_to_buy_clothes_l376_376735


namespace relationship_xyz_l376_376063

variables {a b c x y z : ℝ}

-- Conditions
def log_a_b : Prop := Real.log a b = x
def log_b_c : Prop := Real.log b c = y
def log_c_a : Prop := Real.log c a = z

-- Proof problem statement
theorem relationship_xyz (h1 : log_a_b) (h2 : log_b_c) (h3 : log_c_a) : x * y * z = 1 :=
sorry

end relationship_xyz_l376_376063


namespace const_term_in_expansion_is_40_l376_376186

def binom_expansion_constant_term (x : ℝ) : ℝ :=
  have h1 : (√x + 3) * (√x - 2/x)^5 = (√x + 3) * Finset.range(6).sum (λ k, (Nat.choose 5 k) * (√x)^(5 - k) * (-2/x)^k) := by sorry 
  have h2 : (√x - 2/x)^5 = Finset.range(6).sum (λ k, (Nat.choose 5 k) * (√x)^(5 - k) * (-2/x)^k) := by sorry 
  Finset.range(6).sum (λ k, (Nat.choose 5 k) * (√x)^(5 - k) * (-2/x)^k)

theorem const_term_in_expansion_is_40 (x : ℝ) (hx : x ≠ 0) : binom_expansion_constant_term x = 40 := by
  sorry

end const_term_in_expansion_is_40_l376_376186


namespace value_of_m_l376_376802

theorem value_of_m (x m : ℝ) (h : 2 * x + m - 6 = 0) (hx : x = 1) : m = 4 :=
by
  sorry

end value_of_m_l376_376802


namespace number_of_triangles_with_area_1_l376_376032

-- Given conditions as definitions in Lean
def points_l1 := {A : Point, B : Point, C : Point}
def points_l2 := {D : Point, E : Point, F : Point, G : Point, H : Point}

def distance_AB : Real := 1
def distance_BC : Real := 2
def distance_DE : Real := 1
def distance_EF : Real := 1
def distance_FG : Real := 1
def distance_GH : Real := 1

def line_parallel (l1 l2 : Line) : Prop := ∃ (d : Real), d > 0 ∧ l1.distance_to(l2) = d
def l1_parallel_l2 : Prop := line_parallel l1 l2

def distance_l1_l2 : Real := 1

-- Define the theorem to prove
theorem number_of_triangles_with_area_1 : (number_of_triangles (points_l1, points_l2, distance_AB, distance_BC, distance_DE, distance_EF, distance_FG, distance_GH, l1_parallel_l2, distance_l1_l2) = 14) :=
by {
  sorry
}

end number_of_triangles_with_area_1_l376_376032


namespace domain_of_arcsin_function_l376_376556

noncomputable def f (x : ℝ) : ℝ := 2 * Real.arcsin (x - 2)

theorem domain_of_arcsin_function (x : ℝ) : 
  (∀ y, y ∈ set.Icc (-(Real.pi / 3)) Real.pi → y = f x) → 
  x ∈ set.Icc (3 / 2) 3 :=
by
  sorry

end domain_of_arcsin_function_l376_376556


namespace median_of_consecutive_integers_sum_eq_6_pow_4_l376_376572

theorem median_of_consecutive_integers_sum_eq_6_pow_4 :
  ∀ (s : ℕ) (n : ℕ), s = 36 → ∑ i in finset.range 36, (n + i) = 6^4 → 36 / 2 = 36 :=
by
  sorry

end median_of_consecutive_integers_sum_eq_6_pow_4_l376_376572


namespace minimum_rows_required_l376_376640

theorem minimum_rows_required (n : ℕ) (c : Fin n → ℕ)
  (h1 : ∀ i, 1 ≤ c i ∧ c i ≤ 39)
  (h2 : (∑ i, c i) = 1990) : 
  ∃ r, r ≤ 12 ∧ (∀ i, ∃ j < r, ∑ (k : Fin n) in (finset.filter (λ k, c k ≤ 199 - ∑ m in (set.range j), c m) finset.univ), c k = 199) ∧ (∀ j < r, ∑ i, c i ≤ 199) :=
sorry

end minimum_rows_required_l376_376640


namespace ellipse_eq_l376_376427

theorem ellipse_eq (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : a^2 - b^2 = 4)
  (h4 : ∃ (line_eq : ℝ → ℝ), ∀ (x : ℝ), line_eq x = 3 * x + 7)
  (h5 : ∃ (mid_y : ℝ), mid_y = 1 ∧ ∃ (x1 y1 x2 y2 : ℝ), 
    ((y1 = 3 * x1 + 7) ∧ (y2 = 3 * x2 + 7)) ∧ 
    (y1 + y2) / 2 = mid_y): 
  (∀ x y : ℝ, (y^2 / (a^2 - 4) + x^2 / b^2 = 1) ↔ 
  (x^2 / 8 + y^2 / 12 = 1)) :=
by { sorry }

end ellipse_eq_l376_376427


namespace cos_2θ_sin_α_plus_β_l376_376823

variable (θ α β : ℝ)

-- Conditions
axiom point_P : ∃ θ : ℝ, ∃ α : ℝ, (1 / 2, Real.cos θ ^ 2)
axiom point_Q : ∃ θ : ℝ, ∃ β : ℝ, (Real.sin θ ^ 2, -1)
axiom dot_product_cond : ∃ θ : ℝ, (1 / 2 * (Real.sin θ) ^ 2 - (Real.cos θ) ^ 2 = -1 / 2)

-- Proof problems
theorem cos_2θ (θ : ℝ) (h : 1 / 2 * (Real.sin θ) ^ 2 - (Real.cos θ) ^ 2 = -1 / 2) :
  Real.cos (2 * θ) = 1 / 3 := sorry

theorem sin_α_plus_β (θ α β: ℝ)
  (h1 : 1 / 2 * (Real.sin θ) ^ 2 - (Real.cos θ) ^ 2 = -1 / 2)
  (h2 : Real.cos (2 * θ) = 1 / 3)
  (h3 : (1 / 2, Real.cos θ ^ 2))
  (h4 : (Real.sin θ ^ 2, -1)) :
  Real.sin (α + β) = -Real.sqrt 10 / 10 := sorry

end cos_2θ_sin_α_plus_β_l376_376823


namespace problem1_problem2_problem3_l376_376381

def arithmetic_sequence (a : ℝ) (seq : List ℝ) (d : ℝ) : Prop :=
  match seq with
  | [] => True
  | x :: xs => (x = a + d) ∧ ∀ n, (n < xs.length → xs.nth n = some (a + (n + 1) * d))

def geometric_sequence (a : ℝ) (seq : List ℝ) (r : ℝ) : Prop :=
  match seq with
  | [] => True
  | x :: xs => (x = a * r) ∧ ∀ n, (n < xs.length → xs.nth n = some (a * r^(n + 1)))

theorem problem1 (a b : ℝ) (a_seq b_seq : List ℝ) 
  (h_distinct : a ≠ b) (h_pos : 0 < a ∧ 0 < b) 
  (h_arith : arithmetic_sequence a a_seq ((b - a) / 6)) 
  (h_geom : geometric_sequence a b_seq (Real.root 6 (b / a))) 
  (h_a_seq_len : a_seq.length = 5) 
  (h_b_seq_len : b_seq.length = 5) 
  (h_condition : a_seq.nth 2 = some ((a + b) / 2) ∧ b_seq.nth 2 = some (Real.sqrt (a * b)) ∧ ((a + b) / 2) / (Real.sqrt (a * b)) = 5 / 4): 
  b / a = 4 := 
  sorry

theorem problem2 (a : ℝ) (m : ℕ) (λ : ℕ) (n : ℕ) 
  (h_pos : 0 < a)
  (h_conditions : 2 ≤ λ ∧ (6 ≤ n ∧ n ≤ m) ∧ λ ∈ ℕ^* ∧ b = λ * a ∧ ∃ (a_seq b_seq : List ℝ), a_seq.nth (n-5) = b_seq.nth n ∧ 
    arithmetic_sequence a a_seq ((λ - 1) / (m + 1)) ∧ geometric_sequence a b_seq (λ^(1/(m+1)))) :
  λ = 4 ∧ m = 29 :=
  sorry

theorem problem3 (a b : ℝ) (m : ℕ) (a_seq b_seq : List ℝ) 
  (h_pos : 0 < a ∧ 0 < b) 
  (h_arith : arithmetic_sequence a a_seq ((b - a) / (m + 1))) 
  (h_geom : geometric_sequence a b_seq (Real.root (m+1) (b / a))) 
  (h_length : a_seq.length = m ∧ b_seq.length = m)
  (h_distinct : a ≠ b) :
  ∀ n, (n ∈ (Fin m)) → a_seq.nth n > b_seq.nth n :=
  sorry

end problem1_problem2_problem3_l376_376381


namespace area_triangle_less_than_one_l376_376135

-- Define the necessary parameters and relationships
variables {A B C O L : Type} [triangle ABC]
variables {h_a l_a h_b l_b h_c l_c : ℝ}

-- Given conditions
axiom ha_lt_la : h_a < l_a
axiom la_lt_1  : l_a < 1
axiom hb_lt_lb : h_b < l_b
axiom lb_lt_1  : l_b < 1
axiom hc_lt_lc : h_c < l_c
axiom lc_lt_1  : l_c < 1

-- To prove: the area of the triangle ABC is less than 1
theorem area_triangle_less_than_one (h_a_lt_1 : h_a < 1) 
                                    (h_b_lt_1 : h_b < 1) 
                                    (h_c_lt_1 : h_c < 1) : 
                                    area ABC < 1 :=
begin
  sorry
end

end area_triangle_less_than_one_l376_376135


namespace set_B_equals_l376_376766

-- Given Definitions and Conditions
variable (U A B : Set ℕ) -- Sets U, A, B
variable (U_def : U = {1, 3, 5, 7, 9}) -- Definition of U as a set of natural numbers
variable (A_sub_U : A ⊆ U) (B_sub_U : B ⊆ U)
variable (A_cap_B : A ∩ B = {1, 3}) -- Intersection of A and B
variable (CU_A : U \ A) -- Complement of A in U
variable (CU_A_cap_B : CU_A ∩ B = {5}) -- Intersection of complement of A in U with B

-- Prove
theorem set_B_equals : B = {1, 3, 5} := by
  sorry

end set_B_equals_l376_376766


namespace concurrency_of_cevians_l376_376203

open EuclideanGeometry

/-- Given a triangle ABC, intersected by a line e at points A', B', C'.
    The lines AA', BB', and CC' form the triangle A''B''C''. 
    We need to prove that AA'', BB'', and CC'' meet at a common point E. -/
theorem concurrency_of_cevians
  {A B C A' B' C'' A'' B'' C'' E : Point}
  (h1 : Collinear A' B' C')  -- A', B', C' are collinear on line e
  (h2 : Intersect (Side A B) (LineThroughPoints A A')) -- AA' intersects side AB
  (h3 : Intersect (Side B C) (LineThroughPoints B B')) -- BB' intersects side BC
  (h4 : Intersect (Side C A) (LineThroughPoints C C')) -- CC' intersects side CA
  (h5 : FormsTriangle A A' C' B' B'' C'' A'' E)  -- Definition of triangle
  : Concurrent (LineThroughPoints A A'') (LineThroughPoints B B'') (LineThroughPoints C C'') :=
sorry

end concurrency_of_cevians_l376_376203


namespace max_abs_x2_is_2_l376_376785

noncomputable def max_abs_x2 {x₁ x₂ x₃ : ℝ} (h : x₁^2 + x₂^2 + x₃^2 + x₁ * x₂ + x₂ * x₃ = 2) : ℝ :=
2

theorem max_abs_x2_is_2 {x₁ x₂ x₃ : ℝ} (h : x₁^2 + x₂^2 + x₃^2 + x₁ * x₂ + x₂ * x₃ = 2) :
  max_abs_x2 h = 2 := 
sorry

end max_abs_x2_is_2_l376_376785


namespace roots_of_equation_l376_376928

theorem roots_of_equation : ∀ x : ℝ, x^2 - 3 * x = 0 ↔ x = 0 ∨ x = 3 :=
by sorry

end roots_of_equation_l376_376928


namespace three_digit_number_five_times_product_of_digits_l376_376341

theorem three_digit_number_five_times_product_of_digits :
  ∃ (a b c : ℕ), a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ (100 * a + 10 * b + c = 5 * a * b * c) ∧ (100 * a + 10 * b + c = 175) := 
begin
  existsi 1,
  existsi 7,
  existsi 5,
  split, { norm_num }, -- a > 0
  split, { norm_num }, -- a < 10
  split, { norm_num }, -- b < 10
  split, { norm_num }, -- c < 10
  split,
  { calc 100 * 1 + 10 * 7 + 5 = 100 + 70 + 5 : by norm_num
                        ... = 175 : by norm_num
                        ... = 5 * 1 * 7 * 5 : by norm_num [1*7*5] },
  { norm_num }
end

end three_digit_number_five_times_product_of_digits_l376_376341


namespace sin_405_eq_sqrt2_div2_l376_376314

theorem sin_405_eq_sqrt2_div2 : Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_405_eq_sqrt2_div2_l376_376314


namespace sum_first_13_terms_l376_376934

theorem sum_first_13_terms
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h₁ : a 4 + a 10 - (a 7)^2 + 15 = 0)
  (h₂ : ∀ n : ℕ, a n > 0) :
  S 13 = 65 :=
sorry

end sum_first_13_terms_l376_376934


namespace range_of_m_l376_376472

-- Given functions f and g and the interval [2, 3]
def f (x : ℝ) : ℝ := x^3 - 2*x + 7
def g (x m : ℝ) : ℝ := x + m
def intimate_interval (a b : ℝ) (f g : ℝ → ℝ) : Prop := ∀ x ∈ set.Icc a b, abs (f x - g x) ≤ 10

-- Statement to prove
theorem range_of_m :
  intimate_interval 2 3 f (g m) → 15 ≤ m ∧ m ≤ 19 :=
by sorry

end range_of_m_l376_376472


namespace base_ten_to_six_l376_376231

def convert_base_six (n : ℕ) : ℕ :=
  let c3 := n / 216
  let r3 := n % 216
  let c2 := r3 / 36
  let r2 := r3 % 36
  let c1 := r2 / 6
  let r1 := r2 % 6
  let c0 := r1
  1000*c3 + 100*c2 + 10*c1 + c0

theorem base_ten_to_six (n : ℕ) (h : n = 231) : convert_base_six n = 1023 :=
by
  rw [h]
  unfold convert_base_six
  simp
  sorry

end base_ten_to_six_l376_376231


namespace water_speed_l376_376652

theorem water_speed (swimmer_speed still_water : ℝ) (distance time : ℝ) (h1 : swimmer_speed = 12) (h2 : distance = 12) (h3 : time = 6) :
  ∃ v : ℝ, v = 10 ∧ distance = (swimmer_speed - v) * time :=
by { sorry }

end water_speed_l376_376652


namespace find_angle_B_find_sin_BAC_l376_376081

open Real

noncomputable def triangle_condition (a b c A B C : ℝ) := 
  2 * b * sin (C + π / 6) = a + c

noncomputable def midpoint_condition (a b c : ℝ) (AM AC : ℝ) :=
  AM = AC ∧ AM = a / 2 ∧ AC = sqrt((a^2 + c^2 - a * c))

theorem find_angle_B (a b c A B C : ℝ) (h : triangle_condition a b c A B C) :
  B = π / 3 :=
sorry

theorem find_sin_BAC (a b c A B C : ℝ) (AM AC : ℝ) (h1 : triangle_condition a b c A B C) (h2 : midpoint_condition a b c AM AC) :
  sin A = sqrt(21) / 7 :=
sorry

end find_angle_B_find_sin_BAC_l376_376081


namespace interior_angle_trig_negatives_l376_376798

theorem interior_angle_trig_negatives (α : ℝ) (h₁ : 0 < α) (h₂ : α < π) : 
  (∃ α, 0 < α ∧ α < π ∧ cos α < 0) ∧ (∃ α, 0 < α ∧ α < π ∧ tan α < 0) :=
by
  sorry

end interior_angle_trig_negatives_l376_376798


namespace train_length_is_correct_l376_376285

noncomputable def speed_kmph_to_mps (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

noncomputable def distance_crossed (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

noncomputable def train_length (speed_kmph crossing_time bridge_length : ℝ) : ℝ :=
  distance_crossed (speed_kmph_to_mps speed_kmph) crossing_time - bridge_length

theorem train_length_is_correct :
  ∀ (crossing_time bridge_length speed_kmph : ℝ),
    crossing_time = 26.997840172786177 →
    bridge_length = 150 →
    speed_kmph = 36 →
    train_length speed_kmph crossing_time bridge_length = 119.97840172786177 :=
by
  intros crossing_time bridge_length speed_kmph h1 h2 h3
  rw [h1, h2, h3]
  simp only [speed_kmph_to_mps, distance_crossed, train_length]
  sorry

end train_length_is_correct_l376_376285


namespace complex_number_properties_l376_376372

theorem complex_number_properties :
  let z : ℂ := (1 + 5*complex.I) / (1 - complex.I)
  in (conj z = -2 - 3*complex.I) ∧ (z - 3*complex.I ∈ ℝ) ∧ (z + 2).im = 3 ∧ (z + 2).re = 0 :=
by
  let z : ℂ := (1 + 5*complex.I) / (1 - complex.I)
  sorry

end complex_number_properties_l376_376372


namespace non_empty_subsets_of_even_set_l376_376412

def even_set : Set ℕ := {2, 4, 6, 8, 10}

theorem non_empty_subsets_of_even_set :
  (even_set.powerset.erase ∅).card = 31 :=
by
  sorry

end non_empty_subsets_of_even_set_l376_376412


namespace finite_solutions_l376_376174

theorem finite_solutions : 
  { (a, b, c) : ℕ × ℕ × ℕ | 0 < a ∧ 0 < b ∧ 0 < c ∧ (1 / a + 1 / b + 1 / c = 1 / 1000) }.finite :=
begin
  sorry
end

end finite_solutions_l376_376174


namespace andrea_meets_lauren_in_17_5_minutes_l376_376675

/-
Question: How many total minutes from the start does it take for Andrea to reach Lauren?
Conditions: 
  Andrea and Lauren start from two points that are 30 kilometers apart.
  Andrea travels at twice the speed of Lauren.
  Initially, the distance between them decreases at a rate of 2 kilometers per minute.
  After biking for 10 minutes, Lauren gets a flat tire and stops biking.
Answer: 17.5 minutes
-/
open_locale real

theorem andrea_meets_lauren_in_17_5_minutes
  (distance_initial : ℝ := 30)
  (time_together : ℝ := 10)
  (decrease_rate : ℝ := 2)
  (v_L : ℝ)
  (v_A : ℝ := 2 * v_L)
  (initial_speed_sum : v_A + v_L = decrease_rate) :
  v_L = 2 / 3 → (time_together + 10 / (4 / 3)) = 17.5 := 
by
  sorry

end andrea_meets_lauren_in_17_5_minutes_l376_376675


namespace range_of_a_l376_376137

def f (a x : ℝ) := x * Real.log x - a * x^2 + (2 * a - 1) * x

def g (a x : ℝ) := (f a x).derivative

def local_max_at_one (a : ℝ) : Prop := 
  f a 1 > f a (1 - ε) ∧ f a 1 > f a (1 + ε) ∧ (∀ ε, 0 < ε → (f a (1 + ε) < f a 1 ∧ f a (1 - ε) < f a 1))

theorem range_of_a (a : ℝ) : local_max_at_one a → a > 1/2 := 
by
  sorry

end range_of_a_l376_376137


namespace percentage_spent_on_household_items_l376_376669

theorem percentage_spent_on_household_items (monthly_income : ℝ) (savings : ℝ) (clothes_percentage : ℝ) (medicines_percentage : ℝ) (household_spent : ℝ) : 
  monthly_income = 40000 ∧ 
  savings = 9000 ∧ 
  clothes_percentage = 0.25 ∧ 
  medicines_percentage = 0.075 ∧ 
  household_spent = monthly_income - (clothes_percentage * monthly_income + medicines_percentage * monthly_income + savings)
  → (household_spent / monthly_income) * 100 = 45 :=
by
  intro h
  cases' h with h1 h_rest
  cases' h_rest with h2 h_rest
  cases' h_rest with h3 h_rest
  cases' h_rest with h4 h5
  have h_clothes := h3
  have h_medicines := h4
  have h_savings := h2
  have h_income := h1
  have h_household := h5
  sorry

end percentage_spent_on_household_items_l376_376669


namespace fraction_power_multiplication_l376_376605

theorem fraction_power_multiplication :
  ( (8 / 9)^3 * (5 / 3)^3 ) = (64000 / 19683) :=
by
  sorry

end fraction_power_multiplication_l376_376605


namespace projection_matrix_correct_l376_376120

-- Define a type for 3D vectors
def vec3 := ℝ × ℝ × ℝ

-- Define the dot product for 3D vectors
def dot_product (v1 v2 : vec3) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the normal vector
def n : vec3 := (2, -1, 1)

-- Define the projection matrix
noncomputable def Q : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.vecℚ
    [[1 / 3, 1 / 6, -1 / 6],
     [1 / 3, 5 / 6, 1 / 6],
     [-1 / 3, 5 / 6, 5 / 6]]

-- Define the projection of any vector v onto the plane defined by n
def projection (v : vec3) : vec3 :=
  -- Compute the dot products
  let num := dot_product v n
  let denom := dot_product n n
  -- Compute the projection onto the normal vector
  let proj_n := (num / denom) *: n
  -- Subtract this from the original vector to get the projection onto the plane
  (v.1 - proj_n.1, v.2 - proj_n.2, v.3 - proj_n.3)

-- Main theorem statement
theorem projection_matrix_correct :
  ∀v : vec3, Matrix.mulVec Q v = projection v :=
sorry

end projection_matrix_correct_l376_376120


namespace find_quadratic_function_l376_376746

-- Define the quadratic function f(x) and the given conditions
def quadratic_function (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f(x + 1) - f(x) = 2 * x) ∧ (f 0 = 1)

theorem find_quadratic_function (f : ℝ → ℝ) :
  quadratic_function f →
  (∀ x : ℝ, f(x) = x^2 - x + 1) ∧ (∃ x : ℝ, x ∈ set.Icc (-1 : ℝ) (1 : ℝ) ∧ f(x) = (3/4 : ℝ)) ∧ (∃ x : ℝ, x ∈ set.Icc (-1 : ℝ) (1 : ℝ) ∧ f(x) = 3) := by
  sorry

end find_quadratic_function_l376_376746


namespace candle_burning_time_l376_376600

theorem candle_burning_time :
  ∃ T : ℝ, 
    (∀ T, 0 ≤ T ∧ T ≤ 4 → thin_candle_length = 24 - 6 * T) ∧
    (∀ T, 0 ≤ T ∧ T ≤ 6 → thick_candle_length = 24 - 4 * T) ∧
    (2 * (24 - 6 * T) = 24 - 4 * T) →
    T = 3 :=
by
  sorry

end candle_burning_time_l376_376600


namespace Tetrahedron_equality_l376_376918

noncomputable def midpoint (A B : Point) : Point :=
  -- Definition of midpoint (not implemented here)
  sorry

structure Tetrahedron :=
  (S A B C : Point)
  (SH : Line)
  (midSA : midpoint S A)
  (distance_condition : dist (midpoint S A) S = dist (midpoint S A) A ∧ dist (midpoint S A) A = dist (midpoint S A) C)

theorem Tetrahedron_equality (tet : Tetrahedron) :
  let M := tet.midSA in
  let H := foot_of_perpendicular tet.S tet.ABC in
  dist tet.B tet.A ^ 2 + dist tet.B H ^ 2 = dist tet.C tet.A ^ 2 + dist tet.C H ^ 2 :=
sorry

end Tetrahedron_equality_l376_376918


namespace triangle_eq_segments_l376_376948

/-- Given a triangle ABC with ∠BAC < 90°, points D and E such that 
|AD| = |BD| and ∠ADB = 90°, and |AE| = |CE| and ∠AEC = 90°, and a point X such that
quadrilateral ADXE is a parallelogram, prove that |BX| = |CX|. -/
theorem triangle_eq_segments
  (A B C D E X : EuclideanGeometry.Point ℝ)
  (hBAC : EuclideanGeometry.angle A B C < 90)
  (hAD_eq_BD : EuclideanGeometry.distance A D = EuclideanGeometry.distance B D)
  (hADB : EuclideanGeometry.angle A D B = 90)
  (hAE_eq_CE : EuclideanGeometry.distance A E = EuclideanGeometry.distance C E)
  (hAEC : EuclideanGeometry.angle A E C = 90)
  (hParallelogram : EuclideanGeometry.is_parallelogram A D X E) :
  EuclideanGeometry.distance B X = EuclideanGeometry.distance C X :=
sorry

end triangle_eq_segments_l376_376948


namespace day_of_week_2003_l376_376806

/-- 
  If the 25th day of the year 2003 falls on a Saturday,
  and 2003 is not a leap year (hence, it has 365 days),
  prove that the 365th day of the year 2003 falls on a Wednesday.
-/
theorem day_of_week_2003 :
  (day_of_week 2003 25 = saturday) ∧ (¬ (is_leap_year 2003)) → (day_of_week 2003 365 = wednesday) :=
by
  sorry

end day_of_week_2003_l376_376806


namespace common_difference_of_arithmetic_seq_l376_376388

variable (a_1 d : ℤ) (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Condition 1: definition of general term in an arithmetic sequence
axiom general_term_arith_sequence (n : ℕ) : a n = a_1 + (n - 1) * d

-- Condition 2: sum of the first n terms of the arithmetic sequence
axiom sum_first_n_arith_sequence (n : ℕ) : S n = n * (2 * a_1 + (n - 1) * d) / 2

-- Condition 3: given condition S_4 = 3 * S_2
axiom S4_eq_3S2 : S 4 = 3 * S 2

-- Condition 4: given condition a_7 = 15
axiom a7_eq_15 : a 7 = 15

-- Goal: prove that the common difference d is 2
theorem common_difference_of_arithmetic_seq : d = 2 := by
  sorry

end common_difference_of_arithmetic_seq_l376_376388


namespace angle_B_measure_max_area_triangle_l376_376376

variable {A B C : ℝ} -- Angles of the triangle (in radians)
variable {a b c : ℝ} -- Sides opposite these angles

-- Define the initial condition
axiom h1 : b * Real.sin A + a * Real.cos B = 0

-- First goal: Prove that angle B is 3π/4
theorem angle_B_measure (h1 : b * Real.sin A + a * Real.cos B = 0) : B = 3 * Real.pi / 4 := 
  sorry

-- Second goal: Calculate the maximum area of the triangle when b = 2
theorem max_area_triangle (h1 : b * Real.sin A + a * Real.cos B = 0) (hb : b = 2) : 
  let S := 0.5 * a * c * Real.sin B in 
  S ≤ 2 - Real.sqrt 2 :=
  sorry

end angle_B_measure_max_area_triangle_l376_376376


namespace inequality_proof_l376_376006

theorem inequality_proof
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 ≥ x2)
  (h2 : x2 ≥ x3)
  (h3 : x3 ≥ x4)
  (h4 : x2 + x3 + x4 ≥ x1) :
  (x1 + x2 + x3 + x4)^2 ≤ 4 * x1 * x2 * x3 * x4 := 
by
  sorry

end inequality_proof_l376_376006


namespace extreme_points_count_l376_376198

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x

theorem extreme_points_count : 
  (set.count {x : ℝ | ∃ l, has_deriv_at f l x ∧ has_deriv_at f 0 x}) = 0 :=
by
  sorry

end extreme_points_count_l376_376198


namespace relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0_sufficiency_x_lt_1_necessity_x_lt_1_l376_376960

theorem relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0 (x : ℝ) :
  (x < 1) → (x^2 - 4 * x + 3 > 0) :=
by sorry

-- Define the sufficiency part
theorem sufficiency_x_lt_1 (x : ℝ) :
  (x < 1) → (x^2 - 4 * x + 3 > 0) :=
by sorry

-- Define the necessity part
theorem necessity_x_lt_1 (x : ℝ) :
  (x^2 - 4 * x + 3 > 0) → (x < 1 ∨ x > 3) :=
by sorry

end relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0_sufficiency_x_lt_1_necessity_x_lt_1_l376_376960


namespace radiator_capacity_l376_376637

theorem radiator_capacity :
  ∀ (C : ℝ), 
  (0.40 * C - 0.40 + 1 = 0.50 * C) →
  (C = 6) :=
begin
  intros C h,
  sorry,
end

end radiator_capacity_l376_376637


namespace area_of_S3_l376_376664

theorem area_of_S3 (area_S1 : ℝ)
  (h1 : area_S1 = 36) :
  let side_S1 := real.sqrt area_S1,
      diagonal_S1 := side_S1 * real.sqrt 2,
      side_S2 := diagonal_S1 / 3,
      area_S2 := side_S2 ^ 2,
      diagonal_S2 := side_S2 * real.sqrt 2,
      side_S3 := diagonal_S2 / 3,
      area_S3 := side_S3 ^ 2
  in area_S3 = 16 / 9 :=
by
  sorry

end area_of_S3_l376_376664


namespace correct_factorization_l376_376237

-- Definitions from conditions
def A: Prop := ∀ x y: ℝ, x^2 - 4*y^2 = (x + y) * (x - 4*y)
def B: Prop := ∀ x: ℝ, (x + 4) * (x - 4) = x^2 - 16
def C: Prop := ∀ x: ℝ, x^2 - 2*x + 1 = (x - 1)^2
def D: Prop := ∀ x: ℝ, x^2 - 8*x + 9 = (x - 4)^2 - 7

-- Goal is to prove that C is a correct factorization
theorem correct_factorization: C := by
  sorry

end correct_factorization_l376_376237


namespace geometric_sequence_formula_l376_376090

variable {q : ℝ} -- Common ratio
variable {m n : ℕ} -- Positive natural numbers
variable {b : ℕ → ℝ} -- Geometric sequence

-- This is only necessary if importing Mathlib didn't bring it in
noncomputable def geom_sequence (m n : ℕ) (b : ℕ → ℝ) (q : ℝ) : Prop :=
  b n = b m * q^(n - m)

theorem geometric_sequence_formula (q : ℝ) (m n : ℕ) (b : ℕ → ℝ) 
  (hmn : 0 < m ∧ 0 < n) :
  geom_sequence m n b q :=
by sorry

end geometric_sequence_formula_l376_376090


namespace tunnel_length_correct_l376_376624

noncomputable def length_of_tunnel
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time_min : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let crossing_time_s := crossing_time_min * 60
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem tunnel_length_correct :
  length_of_tunnel 800 78 1 = 500.2 :=
by
  -- The proof will be filled later.
  sorry

end tunnel_length_correct_l376_376624


namespace negation_proof_l376_376551

theorem negation_proof :
  (¬ ∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) :=
by
  sorry

end negation_proof_l376_376551


namespace percentage_decrease_mpg_l376_376685

theorem percentage_decrease_mpg :
  ∀ (mpg_45 mph mpg_60mph : ℝ),
  mpg_45mph = 45 → mpg_60mph = (360 / 10) → 
  ((mpg_45mph - mpg_60mph) / mpg_45mph) * 100 = 20 :=
by
  intros mpg_45mph mpg_60mph h₁ h₂
  rw [h₁, h₂]
  -- Proof omitted
  sorry

end percentage_decrease_mpg_l376_376685


namespace simplify_and_find_fraction_l376_376520

theorem simplify_and_find_fraction (m : ℤ) : 
  (∃ c d : ℤ, ∀ x : ℤ, (6 * x + 12) / 3 = c * x + d ∧ c / d = 1 / 2) := 
by
  use 2
  use 4
  intro x
  split
  calc 
    (6 * x + 12) / 3 = (6 * (x + 2)) / 3 := by sorry
                 ... = 6 / 3 * (x + 2) := by sorry
                 ... = 2 * (x + 2) := by sorry
                 ... = 2 * x + 4 := by sorry
  show 2 / 4 = 1 / 2 from by sorry

end simplify_and_find_fraction_l376_376520


namespace median_of_consecutive_integers_l376_376584

theorem median_of_consecutive_integers (sum_n : ℤ) (n : ℤ) 
  (h1 : sum_n = 6^4) (h2 : n = 36) : 
  (sum_n / n) = 36 :=
by
  sorry

end median_of_consecutive_integers_l376_376584


namespace find_n_lines_l376_376434

theorem find_n_lines (n : ℕ) (h₀ : 0 < n) (h₁ : ∀ i j : ℕ, i < j → i < n → j < n → ∃ p : ℝ × ℝ, ∀ k < n, k ≠ i ∧ k ≠ j → ¬ p ∈ (line_k : ℝ × ℝ → Prop))  
                    (h₂ : ¬ ∃ (p : ℝ × ℝ), ∀ (i, j, k : ℕ), i < n → j < n → k < n → i < j → j < k → p ∈ (line_i ∩ line_j ∩ line_k : ℝ × ℝ → Prop))
                    (h₃ : 16 = (n * (n - 1)) / 2 - 6) 
                    (h₄ : 6 = number_of_points_three_lines_intersect) : 
    n = 8 := 
by 
suffices (n * (n - 1)) / 2 = 28 from 
  have h : n * (n - 1) = 56, 
     show n = 8,
     from sorry,
    sorry

end find_n_lines_l376_376434


namespace distinct_arrangements_STARS_l376_376416

def num_letters : ℕ := 5
def freq_S : ℕ := 2

theorem distinct_arrangements_STARS :
  (num_letters.factorial / freq_S.factorial) = 60 := 
by
  sorry

end distinct_arrangements_STARS_l376_376416


namespace sum_of_adjacent_to_7_l376_376924

/-- Define the divisors of 245, excluding 1 -/
def divisors245 : Set ℕ := {5, 7, 35, 49, 245}

/-- Define the adjacency condition to ensure every pair of adjacent integers has a common factor greater than 1 -/
def adjacency_condition (a b : ℕ) : Prop := (a ≠ b) ∨ (Nat.gcd a b > 1)

/-- Prove the sum of the two integers adjacent to 7 in the given condition is 294. -/
theorem sum_of_adjacent_to_7 (d1 d2 : ℕ) (h1 : d1 ∈ divisors245) (h2 : d2 ∈ divisors245) 
    (adj1 : adjacency_condition 7 d1) (adj2 : adjacency_condition 7 d2) : 
    d1 + d2 = 294 := 
sorry

end sum_of_adjacent_to_7_l376_376924


namespace triangle_angle_not_greater_than_60_l376_376514

theorem triangle_angle_not_greater_than_60 (A B C : ℝ) (h : A + B + C = 180) : 
  A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 :=
begin
  sorry
end

end triangle_angle_not_greater_than_60_l376_376514


namespace jonathan_saved_money_l376_376114

theorem jonathan_saved_money :
  let cost_dictionary := 11
  let cost_dinosaur_book := 19
  let cost_childrens_cookbook := 7
  let amount_needed_more := 29
  let total_cost := cost_dictionary + cost_dinosaur_book + cost_childrens_cookbook
  let saved_money := total_cost - amount_needed_more
  saved_money = 8 :=
by
  -- Definitions
  let cost_dictionary := 11
  let cost_dinosaur_book := 19
  let cost_childrens_cookbook := 7
  let amount_needed_more := 29
  -- Calculation
  let total_cost := cost_dictionary + cost_dinosaur_book + cost_childrens_cookbook
  let saved_money := total_cost - amount_needed_more
  -- Proof
  have h : saved_money = 8, by sorry -- to be proved
  exact h

end jonathan_saved_money_l376_376114


namespace sum_derivatives_positive_l376_376039

noncomputable def f (x : ℝ) : ℝ := -x^2 - x^4 - x^6
noncomputable def f' (x : ℝ) : ℝ := -2*x - 4*x^3 - 6*x^5

theorem sum_derivatives_positive (x1 x2 x3 : ℝ) (h1 : x1 + x2 < 0) (h2 : x2 + x3 < 0) (h3 : x3 + x1 < 0) :
  f' x1 + f' x2 + f' x3 > 0 := 
sorry

end sum_derivatives_positive_l376_376039


namespace angle_between_vectors_l376_376912

-- Defining the vectors based on the given conditions
noncomputable def vector_u := (1 : ℝ, 0 : ℝ, real.sqrt 2)
noncomputable def vector_v_neigh := (0 : ℝ, 1 : ℝ, real.sqrt 2)
noncomputable def vector_v_opp := (-1 : ℝ, 0 : ℝ, real.sqrt 2)

-- Definitions for the problem
def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def angle_cosine (a b : ℝ × ℝ × ℝ) : ℝ :=
  dot_product a b / (magnitude a * magnitude b)

-- Problem statement
theorem angle_between_vectors :
  let phi := real.arccos (angle_cosine vector_u vector_v_neigh),
      phi' := real.arccos (angle_cosine vector_u vector_v_opp)
  in (phi ≈ real.div_pi 3 * 160 / 9.0 ∧ -- approx 48 degrees 11' 23''
      phi' ≈ real.div_pi 3 * 275 / 8.5) -- approx 70 degrees 31' 44''
  :=
sorry

end angle_between_vectors_l376_376912


namespace standard_equation_of_parabola_intersection_of_line_and_parabola_l376_376743

section PartI

def point_A := (1/2 : ℝ, -Real.sqrt 2)
def parabola_curve (p : ℝ) (x : ℝ) (y : ℝ) := y^2 = 2 * p * x

theorem standard_equation_of_parabola :
  ∃ p, parabola_curve p (point_A.1) (point_A.2) ∧ (parabola_curve p = λ x y, y^2 = 4 * x) :=
by
  sorry 
-- Proof to be completed such that we prove the parabola curve meets the conditions and returns y^2 = 4x

end PartI

section PartII

def point_P := (-2 : ℝ, 1 : ℝ)
def line_eq (k : ℝ) (x : ℝ) := k * x + (2 * k + 1)
def line_intersects_parabola_twice (k : ℝ) := 
  ∃ p : ℝ, parabola_curve 2 = (λ x y, y^2 = 4 * x) ∧
  (k ≠ 0 ∧ -1 < k ∧ k < 1/2)

theorem intersection_of_line_and_parabola :
  ∀ k, line_intersects_parabola_twice k ↔ (k ≠ 0 ∧ -1 < k ∧ k < 1/2) :=
by
  sorry 
-- Proof to be completed to show under which conditions the line intersects the parabola twice

end PartII

end standard_equation_of_parabola_intersection_of_line_and_parabola_l376_376743


namespace area_of_triangle_PKF_l376_376047

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

noncomputable def distance (A B : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  dist A B

noncomputable def area_of_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  let a := distance B C
  let b := distance A C
  let c := distance A B
  (1/2) * abs (A.x * B.y + B.x * C.y + C.x * A.y - A.y * B.x - B.y * C.x - C.y * A.x)

noncomputable def K : EuclideanSpace ℝ (Fin 2) := ![-2, 0]
noncomputable def F : EuclideanSpace ℝ (Fin 2) := ![2, 0]

theorem area_of_triangle_PKF (x y : ℝ)
  (hP_on_parabola : point_on_parabola x y)
  (hPK_eq_sqrt2_PF : distance ![x, y] K = real.sqrt 2 * distance ![x, y] F):
  area_of_triangle ![x, y] K F = 8 :=
sorry

end area_of_triangle_PKF_l376_376047


namespace num_non_integer_angles_l376_376478

def interior_angle (n : ℕ) : ℚ :=
  180 * (n - 2) / n

theorem num_non_integer_angles :
  ∃ S : Finset ℕ, S.card = 2 ∧
  ∀ n ∈ S, 3 ≤ n ∧ n ≤ 12 ∧ ¬ (interior_angle n).denom = 1 :=
by
  sorry

end num_non_integer_angles_l376_376478


namespace odd_n_proof_l376_376715

theorem odd_n_proof (n : ℕ) (hn : n ≥ 3) (odd_n : n % 2 = 1) : 
  ∀ (a b : Fin n → ℝ), (∀ k, |a k| + |b k| = 1) → 
  ∃ x : Fin n → ℤ, (∀ k, x k = -1 ∨ x k = 1) ∧ 
  |∑ k in Finset.range n, (x k : ℝ) * a k| + |∑ k in Finset.range n, (x k : ℝ) * b k| ≤ 1 := sorry

end odd_n_proof_l376_376715


namespace speed_of_goods_train_is_72_kmph_l376_376642

-- Definitions for conditions
def length_of_train : ℝ := 240.0416
def length_of_platform : ℝ := 280
def time_to_cross : ℝ := 26

-- Distance covered by the train while crossing the platform
def total_distance : ℝ := length_of_train + length_of_platform

-- Speed calculation in meters per second
def speed_mps : ℝ := total_distance / time_to_cross

-- Speed conversion from meters per second to kilometers per hour
def speed_kmph : ℝ := speed_mps * 3.6

-- Proof statement
theorem speed_of_goods_train_is_72_kmph : speed_kmph = 72 := 
by
  sorry

end speed_of_goods_train_is_72_kmph_l376_376642


namespace triangles_with_equal_area_l376_376822

variable (A B C D P Q : Type) [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B]
          [AddCommGroup C] [Module ℝ C] [AddCommGroup D] [Module ℝ D]
          [AddCommGroup P] [Module ℝ P] [AddCommGroup Q] [Module ℝ Q]

-- Define a parallelogram ABCD
def is_parallelogram (A B C D : A) : Prop :=
  let AB := B - A
  let AD := D - A
  let BC := C - B
  let CD := D - C
  AB + BC = AD + CD

-- Define midpoint P of BC
def is_midpoint (P B C : P) : Prop :=
  2 • P = B + C

-- Define Q as the intersection point on CD when a line through P parallel to BD intersects CD
def is_parallel_through_midpoint (P B D Q C : Q) : Prop :=
  let BD := D - B
  let PQ := Q - P
  BD = PQ

-- Define the main proof problem
theorem triangles_with_equal_area (A B C D P Q : Type) 
          [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B]
          [AddCommGroup C] [Module ℝ C] [AddCommGroup D] [Module ℝ D]
          [AddCommGroup P] [Module ℝ P] [AddCommGroup Q] [Module ℝ Q]
          (h1 : is_parallelogram A B C D)
          (h2 : is_midpoint P B C)
          (h3 : is_parallel_through_midpoint P B D Q C) :
          ∃ (T : Set (Set A)), T.card = 4 ∧ (∀ t ∈ T, ∃ (a b c : A), t = {a, b, c} ∧ area_eq_triangle (triangle A B P) (triangle a b c)) :=
sorry

end triangles_with_equal_area_l376_376822


namespace trigonometric_expression_l376_376059

open Real

theorem trigonometric_expression (α β : ℝ) (h : cos α ^ 2 = cos β ^ 2) :
  (sin β ^ 2 / sin α + cos β ^ 2 / cos α = sin α + cos α ∨ sin β ^ 2 / sin α + cos β ^ 2 / cos α = -sin α + cos α) :=
sorry

end trigonometric_expression_l376_376059


namespace median_of_36_consecutive_integers_l376_376575

theorem median_of_36_consecutive_integers (sum_of_integers : ℕ) (num_of_integers : ℕ) 
  (h1 : num_of_integers = 36) (h2 : sum_of_integers = 6 ^ 4) : 
  (sum_of_integers / num_of_integers) = 36 := 
by 
  sorry

end median_of_36_consecutive_integers_l376_376575


namespace find_weight_of_second_square_l376_376994

-- Define given conditions
def side_length1 : ℝ := 4
def weight1 : ℝ := 16
def side_length2 : ℝ := 6

-- Define the uniform density and thickness condition
def uniform_density (a₁ a₂ : ℝ) (w₁ w₂ : ℝ) : Prop :=
  (a₁ * w₂ = a₂ * w₁)

-- Problem statement:
theorem find_weight_of_second_square : 
  uniform_density (side_length1 ^ 2) (side_length2 ^ 2) weight1 w₂ → 
  w₂ = 36 :=
by
  sorry

end find_weight_of_second_square_l376_376994


namespace mark_money_l376_376874

theorem mark_money (M : ℝ) 
  (h1 : (1 / 2) * M + 14 + (1 / 3) * M + 16 + (1 / 4) * M + 18 = M) : 
  M = 576 := 
sorry

end mark_money_l376_376874


namespace triangle_side_length_l376_376639

open Real

/-- Given a triangle ABC with the incircle touching side AB at point D,
where AD = 5 and DB = 3, and given that the angle A is 60 degrees,
prove that the length of side BC is 13. -/
theorem triangle_side_length
  (A B C D : Point)
  (AD DB : ℝ)
  (hAD : AD = 5)
  (hDB : DB = 3)
  (angleA : Real)
  (hangleA : angleA = π / 3) : 
  ∃ BC : ℝ, BC = 13 :=
sorry

end triangle_side_length_l376_376639


namespace find_common_remainder_l376_376956

theorem find_common_remainder :
  ∃ (d : ℕ), 100 ≤ d ∧ d ≤ 999 ∧ (312837 % d = 96) ∧ (310650 % d = 96) :=
sorry

end find_common_remainder_l376_376956


namespace part1_part2_l376_376117

/-
  Given:
  A = { x ∈ ℤ | |x| ≤ 6 }
  B = {1, 2, 3}
  C = {3, 4, 5, 6}
-/

def A := {x : ℤ | |x| ≤ 6}
def B := {1, 2, 3}
def C := {3, 4, 5, 6}

theorem part1 : A ∩ (B ∩ C) = {3} := by
  sorry

theorem part2 : A ∩ A - (B ∪ C) = {-6, -5, -4, -3, -2, -1, 0} := by
  sorry

end part1_part2_l376_376117


namespace LukaNeeds24CupsOfWater_l376_376142

theorem LukaNeeds24CupsOfWater
  (L S W : ℕ)
  (h1 : S = 2 * L)
  (h2 : W = 4 * S)
  (h3 : L = 3) :
  W = 24 := by
  sorry

end LukaNeeds24CupsOfWater_l376_376142


namespace smallest_value_z_minus_x_l376_376940

theorem smallest_value_z_minus_x 
  (x y z : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (hmul : x * y * z = 5040) 
  (hxy : x < y) 
  (hyz : y < z) : 
  z - x = 9 := 
  sorry

end smallest_value_z_minus_x_l376_376940


namespace parametric_to_cartesian_intersection_product_range_l376_376782

theorem parametric_to_cartesian (a b : ℝ) (α : ℝ) (h1 : a * (Real.cos (π / 4)) = 1)
  (h2 : b * (Real.sin (π / 4)) = sqrt 2 / 2) :
  (∃ a b : ℝ, parametric_eq_C := (λ α, (a * Real.cos α, b * Real.sin α))
    ∧ curve_C_eq : ∀ x y, (∃ α, (x, y) = parametric_eq_C α) → (x^2/2 + y^2 = 1)) :=
by
  sorry

theorem intersection_product_range (θ α : ℝ) (α : ℝ) (h1 : ∃ a b : ℝ, 
  line_l := (λ t, (t * Real.cos θ, sqrt 2 + t * Real.sin θ))
  ∧ curve_C := (λ α, (sqrt 2 * Real.cos α, Real.sin α))
  ∃ x y, (x, y) in line_l ∩ curve_C) :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ (1 ≤ t1 * t2) ∧ (t1 * t2 ≤ 2) :=
by
  sorry

end parametric_to_cartesian_intersection_product_range_l376_376782


namespace csc_neg_330_eq_2_l376_376712

theorem csc_neg_330_eq_2 : 
  (∀ θ, csc θ = 1 / sin θ) →
  (∀ θ, sin (θ + 360) = sin θ) →
  sin 30 = 1 / 2 →
  csc (-330) = 2 := by
  sorry

end csc_neg_330_eq_2_l376_376712


namespace general_formula_of_geometric_seq_term_in_arithmetic_seq_l376_376447

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Condition: Geometric sequence {a_n} with a_1 = 2 and a_4 = 16
def geometric_seq (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n, a (n + 1) = a n * q

-- General formula for the sequence {a_n}
theorem general_formula_of_geometric_seq 
  (ha : geometric_seq a) (h1 : a 1 = 2) (h4 : a 4 = 16) :
  ∀ n, a n = 2^n :=
sorry

-- Condition: Arithmetic sequence {b_n} with b_3 = a_3 and b_5 = a_5
def arithmetic_seq (b : ℕ → ℝ) := ∃ d : ℝ, ∀ n, b (n + 1) = b n + d

-- Check if a_9 is a term in the sequence {b_n} and find its term number
theorem term_in_arithmetic_seq 
  (ha : geometric_seq a) (hb : arithmetic_seq b)
  (h1 : a 1 = 2) (h4 : a 4 = 16)
  (hb3 : b 3 = a 3) (hb5 : b 5 = a 5) :
  ∃ n, b n = a 9 ∧ n = 45 :=
sorry

end general_formula_of_geometric_seq_term_in_arithmetic_seq_l376_376447


namespace flour_requirement_for_160_cookies_l376_376989

def flour_needed (recipe_cookies : ℕ) (recipe_flour : ℕ) (desired_cookies : ℕ) : ℕ :=
  (desired_cookies / recipe_cookies) * recipe_flour

theorem flour_requirement_for_160_cookies :
  ∀ (recipe_cookies recipe_flour desired_cookies : ℕ),
    recipe_cookies = 40 →
    recipe_flour = 3 →
    desired_cookies = 160 →
    flour_needed recipe_cookies recipe_flour desired_cookies = 12 :=
by
  intros recipe_cookies recipe_flour desired_cookies h1 h2 h3
  simp [flour_needed, h1, h2, h3]
  sorry

end flour_requirement_for_160_cookies_l376_376989


namespace min_nS_n_eq_neg32_l376_376750

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (d : ℤ) (a_1 : ℤ)

-- Conditions
axiom arithmetic_sequence_def : ∀ n : ℕ, a n = a_1 + (n - 1) * d
axiom sum_first_n_def : ∀ n : ℕ, S n = n * a_1 + (n * (n - 1) / 2) * d

axiom a5_eq_3 : a 5 = 3
axiom S10_eq_40 : S 10 = 40

theorem min_nS_n_eq_neg32 : ∃ n : ℕ, n * S n = -32 :=
sorry

end min_nS_n_eq_neg32_l376_376750


namespace arsh_eq_arch_pos_eq_arch_neg_eq_arth_eq_l376_376164

noncomputable def arsh (x : ℝ) := Real.log (x + Real.sqrt (x^2 + 1))
noncomputable def arch_pos (x : ℝ) := Real.log (x + Real.sqrt (x^2 - 1))
noncomputable def arch_neg (x : ℝ) := Real.log (x - Real.sqrt (x^2 - 1))
noncomputable def arth (x : ℝ) := (1 / 2) * Real.log ((1 + x) / (1 - x))

theorem arsh_eq (x : ℝ) : arsh x = Real.log (x + Real.sqrt (x^2 + 1)) := by
  sorry

theorem arch_pos_eq (x : ℝ) : arch_pos x = Real.log (x + Real.sqrt (x^2 - 1)) := by
  sorry

theorem arch_neg_eq (x : ℝ) : arch_neg x = Real.log (x - Real.sqrt (x^2 - 1)) := by
  sorry

theorem arth_eq (x : ℝ) : arth x = (1 / 2) * Real.log ((1 + x) / (1 - x)) := by
  sorry

end arsh_eq_arch_pos_eq_arch_neg_eq_arth_eq_l376_376164


namespace proj_matrix_eq_Q_l376_376121

noncomputable def Q : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    [1/3, 1/3, 2/3],
    [2/3, 2/3, 1/3],
    [1/3, 1/3, 2/3]
  ]

def normal_vector : Fin 3 → ℝ := ![
  2, -1, 1
]

def proj_matrix_correct (v : Fin 3 → ℝ) : Prop :=
  Q.mulVec v = v - (1 / 6) * (v.dotProduct normal_vector) • normal_vector

theorem proj_matrix_eq_Q (v : Fin 3 → ℝ) : proj_matrix_correct v :=
by
  sorry

end proj_matrix_eq_Q_l376_376121


namespace find_alpha_l376_376139

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else x^2

theorem find_alpha (α : ℝ) (h : f α = 9) : α = -9 ∨ α = 3 :=
by
  -- Proof goes here
  sorry

end find_alpha_l376_376139


namespace points_form_convex_n_gon_l376_376206

theorem points_form_convex_n_gon (n : ℕ) (points : Fin n → (ℝ × ℝ))
  (h : ∀ (A B C D : Fin n), A ≠ B → B ≠ C → C ≠ D → A ≠ C → A ≠ D → B ≠ D →
    ∃ [Q : convex_hull (set.range points)], convex_hull.is_convex_quadrilateral Q A B C D) :
  ∃ P : finset (ℝ × ℝ), P.card = n ∧ convex_hull.is_convex_ngon P :=
sorry

end points_form_convex_n_gon_l376_376206


namespace team_total_points_l376_376444

-- Definitions based on conditions
def chandra_points (akiko_points : ℕ) := 2 * akiko_points
def akiko_points (michiko_points : ℕ) := michiko_points + 4
def michiko_points (bailey_points : ℕ) := bailey_points / 2
def bailey_points := 14

-- Total points scored by the team
def total_points :=
  let michiko := michiko_points bailey_points
  let akiko := akiko_points michiko
  let chandra := chandra_points akiko
  bailey_points + michiko + akiko + chandra

theorem team_total_points : total_points = 54 := by
  sorry

end team_total_points_l376_376444


namespace soda_total_l376_376105

namespace SodaProof

variable (x : ℚ) -- Define x as a rational number for precision

-- Conditions
def jack_soda := x
def jill_soda := 1.25 * x
def jordan_soda := 1.5 * x

def jack_drunk := 0.6 * x
def jill_drunk := 0.75 * x
def jordan_drunk := 0.9 * x

def jack_remain := 0.4 * x
def jill_remain := 0.5 * x
def jordan_remain := 0.6 * x

def jill_to_jordan := 0.25 * jill_remain + 3
def jack_to_jordan := (1/3) * jack_remain

def new_jack_total := jack_drunk - jack_to_jordan
def new_jill_total := jill_drunk - jill_to_jordan
def new_jordan_total := jordan_drunk + jill_to_jordan + jack_to_jordan

theorem soda_total (x : ℚ):
  (new_jack_total x = new_jill_total x) →
  (new_jack_total x = new_jordan_total x) →
  jack_soda x + jill_soda x + jordan_soda x = 150 :=
by
  intros h1 h2
  sorry

end SodaProof

end soda_total_l376_376105


namespace minkowski_inequality_l376_376066

variable {n m : ℕ}
variable {a : Fin n.succ → Fin m.succ → ℝ}
variable {p : ℝ}

theorem minkowski_inequality
  (ha : ∀ i j, a i j > 0)
  (hp : p > 1) :
  (∑ i : Fin n.succ, (∑ j : Fin m.succ, a i j) ^ p) ^ (1 / p) ≤
  ∑ j : Fin m.succ, (∑ i : Fin n.succ, a i j ^ p) ^ (1 / p) :=
by
  sorry

end minkowski_inequality_l376_376066


namespace triangle_congruence_l376_376848

-- Define the necessary elements (triangle, point, circle, angles, tangents)
variables (A B C M N P : Point)
variables (K : Circle)
variables (L : Point → Point → Point → Prop) -- Incircle predicate (ABC and incenter I)
variables (onArcNotContaining : Point → Point → Point → Prop) -- Predicate for point being on the arc not containing another point
variables (tangentIntersect : Point → Point → Point → Circle → Point → Prop) -- Tangent intersection predicate

-- Define the conditions
-- 1. A triangle ABC inscribed in circle K
def inscribed_in_circle (A B C : Point) (K : Circle) : Prop := sorry
-- 2. M is a point on the arc BC not containing A
def on_arc_BC_not_containing_A (M B C A : Point) (K : Circle) : Prop := onArcNotContaining M B C
-- 3. Tangents from M to the incircle intersect K at points N and P
def tangents_intersect_K_at_NP (M N P : Point) (K : Circle) : Prop := tangentIntersect M N P K L

-- Main theorem stating the congruence given the conditions
theorem triangle_congruence
  (h1 : inscribed_in_circle A B C K)
  (h2 : on_arc_BC_not_containing_A M B C A K)
  (h3 : tangents_intersect_K_at_NP M N P K)
  (h4 : ∠ BAC = ∠ NMP) :
  congruent_triangles (A B C) (M N P) :=
sorry

end triangle_congruence_l376_376848


namespace problem_l376_376062

theorem problem (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) 
: (5 * m * r - 2 * n * t) / (7 * n * t - 10 * m * r) = -31 / 56 := 
sorry

end problem_l376_376062


namespace bugs_diagonally_at_least_9_unoccupied_l376_376504

theorem bugs_diagonally_at_least_9_unoccupied (bugs : ℕ × ℕ → Prop) :
  let board_size := 9
  let cells := (board_size * board_size)
  let black_cells := 45
  let white_cells := 36
  ∃ unoccupied_cells ≥ 9, true := 
sorry

end bugs_diagonally_at_least_9_unoccupied_l376_376504


namespace conic_section_is_ellipse_l376_376706

theorem conic_section_is_ellipse :
  (∀ x y : ℝ, 
    sqrt (x^2 + (y - 2)^2) + sqrt ((x - 6)^2 + (y + 4)^2) = 14) →
  (∃ f1 f2 : ℝ × ℝ, f1 = (0, 2) ∧ f2 = (6, -4) ∧ (∀ x y : ℝ,
    sqrt (x^2 + (y - f1.snd)^2) + sqrt ((x - f2.fst)^2 + (y - f2.snd)^2) = 14)) →
  True := 
begin
  sorry -- Proof not required
end

end conic_section_is_ellipse_l376_376706


namespace both_nilpotent_l376_376464

-- We start by defining the given conditions as Lean statements
variables {n : ℕ}
variables (A B : Matrix (Fin n) (Fin n) ℝ)
variables (t : Fin (n+1) → ℝ)
variables (h_diff : Function.Injective t) -- ensures t_i are n+1 distinct real numbers
variables (h_nil : ∀ i : Fin (n+1), Matrix.nilpotent (A + t i • B)) -- ensures A + t_i B are nilpotent matrices

-- Now we state the theorem we need to prove
theorem both_nilpotent : Matrix.nilpotent A ∧ Matrix.nilpotent B := 
by
  sorry

end both_nilpotent_l376_376464


namespace find_59th_digit_in_1_div_17_l376_376229

theorem find_59th_digit_in_1_div_17 : (decimal_expansion 59 (1 / 17)) = 4 := 
by 
  -- Given the repeating cycle of length 16 for the decimal representation of 1/17
  have cycle_length : nat := 16
  -- Check the 59th digit after the decimal point
  sorry

end find_59th_digit_in_1_div_17_l376_376229


namespace f_ge_three_inequality_given_condition_l376_376043

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Proving the minimum value of the function f(x) is 3
theorem f_ge_three (x : ℝ) : f x ≥ 3 := sorry

-- Defining the given condition a^2 + b^2 = 3 with a, b > 0
variables (a b : ℝ)
include a b

-- Proving the inequality given a^2 + b^2 = 3
theorem inequality_given_condition (h : a > 0 ∧ b > 0 ∧ a^2 + b^2 = 3) :
  2 / (1/a + 1/b) ≤ sqrt 6 / 2 := sorry

end f_ge_three_inequality_given_condition_l376_376043


namespace complex_square_l376_376738

theorem complex_square (a b : ℝ) (i : ℂ) (h1 : a + b * i - 2 * i = 2 - b * i) : 
  (a + b * i) ^ 2 = 3 + 4 * i := 
by {
  -- Proof steps skipped (using sorry to indicate proof is required)
  sorry
}

end complex_square_l376_376738


namespace algebraic_expression_value_l376_376003

theorem algebraic_expression_value {x : ℝ} (h : x * (x + 2) = 2023) : 2 * (x + 3) * (x - 1) - 2018 = 2022 := 
by 
  sorry

end algebraic_expression_value_l376_376003


namespace exists_point_not_on_polyline_l376_376906

-- Definitions as per the conditions
def is_square_grid (n : ℕ) : Prop := n = 100

def polyline {n : ℕ} (grid : fin n × fin n → bool) : Prop :=
  ∀ i j, grid (i, j) = true → -- grid point (i, j) is part of a polyline

  -- Each polyline is non-self-intersecting
  ∀ p1 p2, grid p1 = true ∧ grid p2 = true → p1 ≠ p2

  -- Polylines do not intersect each other
  ∀ p1 p2, grid p1 = true → grid p2 = true → p1 ≠ p2

  -- Endpoints are on the edges of the square
  ∀ i j b, i = 0 ∨ i = n-1 ∨ j = 0 ∨ j = n-1 → grid (i, j) = b →

  -- Polylines are strictly contained within the interior of the square
  ∀ i j, 0 < i → i < n-1 → 0 < j → j < n-1 → grid (i, j) = false

-- Main theorem to be proven
theorem exists_point_not_on_polyline :
  ∀ (n : ℕ) (grid : fin n × fin n → bool),
  is_square_grid n →
  polyline grid →
  ∃ i j, (0 < i ∧ i < n-1 ∧ 0 < j ∧ j < n-1) ∧ grid (i, j) = false ∧
         i ≠ 0 ∧ i ≠ n-1 ∧ j ≠ 0 ∧ j ≠ n-1 :=
sorry

end exists_point_not_on_polyline_l376_376906


namespace new_average_weight_l376_376253

theorem new_average_weight (original_players : ℕ) (new_players : ℕ) 
  (average_weight_original : ℝ) (weight_new_player1 : ℝ) (weight_new_player2 : ℝ) : 
  original_players = 7 → 
  new_players = 2 →
  average_weight_original = 76 → 
  weight_new_player1 = 110 → 
  weight_new_player2 = 60 → 
  (original_players * average_weight_original + weight_new_player1 + weight_new_player2) / (original_players + new_players) = 78 :=
by 
  intros h1 h2 h3 h4 h5;
  sorry

end new_average_weight_l376_376253


namespace bisection_method_root_exists_bisection_method_next_calculation_l376_376973

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem bisection_method_root_exists :
  (f 0 < 0) → (f 0.5 > 0) → ∃ x0 : ℝ, 0 < x0 ∧ x0 < 0.5 ∧ f x0 = 0 :=
by
  intro h0 h05
  sorry

theorem bisection_method_next_calculation :
  f 0.25 = (0.25)^3 + 3 * 0.25 - 1 :=
by
  calc
    f 0.25 = 0.25^3 + 3 * 0.25 - 1 := rfl

end bisection_method_root_exists_bisection_method_next_calculation_l376_376973


namespace value_of_y_when_x_is_8_l376_376803

theorem value_of_y_when_x_is_8 
  (k : ℝ) 
  (y : ℝ) 
  (h1 : y = k * (64:ℝ)^(1/3)) 
  (h2 : y = 3 * real.sqrt 2) : 
  ∃ k, y = k * (8:ℝ)^(1/3) ∧ y = (3 * real.sqrt 2) / 2 := 
by
  use (3 * real.sqrt 2 / 4)
  split
  . sorry
  . sorry

end value_of_y_when_x_is_8_l376_376803


namespace factorization_correct_l376_376959

theorem factorization_correct :
  (x: ℝ) → (y: ℝ) → (A: x * (x + 1) = x^2 + x)
    ∧ (B: x^2 + x + 1 = x * (x + 1) + 1)
    ∧ (C: x^2 - x = x * (x - 1))
    ∧ (D: 2 * x * (y - 1) = 2 * x * y - 2 * x) → C = true :=
by
  sorry

end factorization_correct_l376_376959


namespace tricycle_total_spokes_l376_376671

noncomputable def front : ℕ := 20
noncomputable def middle : ℕ := 2 * front
noncomputable def back : ℝ := 20 * Real.sqrt 2
noncomputable def total_spokes : ℝ := front + middle + back

theorem tricycle_total_spokes : total_spokes = 88 :=
by
  sorry

end tricycle_total_spokes_l376_376671


namespace assembly_line_arrangement_l376_376965

theorem assembly_line_arrangement (A B C D E F : {X // ∀ i j : Fin 6, i ≠ j → X i ≠ X j}) :
  ∃ (y : ℕ), 
  (axles_before_wheels A B) ∧ (tasks_ordered C D E F) → 
  y = 120 :=
sorry

def axles_before_wheels (A B : ℕ) : Prop := A < B

def tasks_ordered (C D E F : ℕ) : Prop := 
  (C < D) ∧ (D < E) ∧ (E < F)

end assembly_line_arrangement_l376_376965


namespace b_range_given_conditions_l376_376037

theorem b_range_given_conditions 
    (b c : ℝ)
    (roots_in_interval : ∀ x, x^2 + b * x + c = 0 → -1 ≤ x ∧ x ≤ 1)
    (ineq : 0 ≤ 3 * b + c ∧ 3 * b + c ≤ 3) :
    0 ≤ b ∧ b ≤ 2 :=
sorry

end b_range_given_conditions_l376_376037


namespace diagonal_ratio_of_regular_octagon_l376_376539

noncomputable def diagonal_ratio : ℝ :=
  let a := 1 in  -- Assume the side length of the octagon is 1 (scaling factor doesn't matter)
  let length_skipping_two_sides := a * Real.sqrt 2 in
  let length_skipping_three_sides := a in
  length_skipping_two_sides / length_skipping_three_sides

theorem diagonal_ratio_of_regular_octagon :
  diagonal_ratio = Real.sqrt 2 :=
by
  sorry

end diagonal_ratio_of_regular_octagon_l376_376539


namespace set_intersection_eq_l376_376077

noncomputable def U : Set ℝ := Set.univ

noncomputable def A : Set ℝ := { x | 2 < x ∧ x ≤ 4 }

noncomputable def B : Set ℤ := { x | x^2 - 7*x + 10 < 0 }

theorem set_intersection_eq :
  A ∩ (U \ ↑B) = { x | 2 < x ∧ x < 3 } ∪ { x | 3 < x ∧ x ≤ 4 } := 
by
  sorry

end set_intersection_eq_l376_376077


namespace sector_perimeter_l376_376384

theorem sector_perimeter (S : ℝ) (r : ℝ) (h1 : S = 8) (h2 : r = 2) : S = 8 ∧ r = 2 → let l := 8 in let perimeter := l + 2 * r in perimeter = 12 :=
by
  intro h
  let l := 8
  let perimeter := l + 2 * r
  have h_perimeter : perimeter = 12 := sorry
  exact h_perimeter

end sector_perimeter_l376_376384


namespace kite_parabolas_l376_376921

-- Definitions of the parabolas
def parabola1 (a x : ℝ) : ℝ := a * x^2 + 4
def parabola2 (b x : ℝ) : ℝ := 6 - b * x^2

-- Definition that the parabolas intersect the coordinate axes in exactly four points, forming vertices of a kite with given area
def formsKiteWithArea (a b : ℝ) (area : ℝ) :=
  area = 18 ∧
  by {
    have h1 : ∃ x, parabola1 a x = 0,
    have h2 : ∃ y, parabola2 b y = 0,
    sorry
  }

-- Statement of the theorem
theorem kite_parabolas (a b : ℝ) (h : formsKiteWithArea a b 18) : a + b = 4 / 45 :=
sorry

end kite_parabolas_l376_376921


namespace nina_spends_70_l376_376499

-- Definitions of the quantities and prices
def toys := 3
def toy_price := 10
def basketball_cards := 2
def card_price := 5
def shirts := 5
def shirt_price := 6

-- Calculate the total amount spent
def total_spent := (toys * toy_price) + (basketball_cards * card_price) + (shirts * shirt_price)

-- Problem statement: Prove that the total amount spent is $70
theorem nina_spends_70 : total_spent = 70 := by
  sorry

end nina_spends_70_l376_376499


namespace factorize_expression_l376_376711

variable {x y : ℝ}

theorem factorize_expression (x y : ℝ) :
  x^2 * y - y^3 = y * (x + y) * (x - y) :=
sorry

end factorize_expression_l376_376711


namespace specific_divisors_count_l376_376864

-- Declare the value of n
def n : ℕ := (2^40) * (3^25) * (5^10)

-- Definition to count the number of positive divisors of a number less than n that don't divide n.
def count_specific_divisors (n : ℕ) : ℕ :=
sorry  -- This would be the function implementation

-- Lean statement to assert the number of such divisors
theorem specific_divisors_count : 
  count_specific_divisors n = 31514 :=
sorry

end specific_divisors_count_l376_376864


namespace comparison_BO_BK_l376_376471

noncomputable def O := sorry  -- Center of equilateral triangle
def A := sorry  -- Vertex A of triangle ABC
def B := sorry  -- Vertex B of triangle ABC
def C := sorry  -- Vertex C of triangle ABC
def M := sorry  -- Midpoint of AC
def K := sorry  -- Point dividing BM in ratio 3:1 from B

-- Side length of triangle ABC
def side_length : ℝ := 10

-- Length of median BD in an equilateral triangle with side length 10
def BD : ℝ := (sqrt 3) * side_length / 2

-- Length of BO
def BO : ℝ := (2 / 3) * BD

-- Length of BM using cosine rule
def BM : ℝ := sqrt ((BO^2 + (BO/2)^2 - BO * (BO/2) * -1) : ℝ)

-- Length of BK as K divides BM in ratio 3:1 from B
def BK : ℝ := (3 / 4) * BM

theorem comparison_BO_BK : BO > BK := by
  unfold BO BD BM BK side_length
  sorry

end comparison_BO_BK_l376_376471


namespace problem1_problem2_l376_376309

theorem problem1 : ((-1 : ℤ) ^ 3 + abs (1 - real.sqrt 2) + real.cbrt 8) = real.sqrt 2 := 
by sorry

theorem problem2 : ((-2 : ℤ) ^ 3 + real.sqrt ((-3 : ℤ) ^ 2) + 3 * real.cbrt (1 / 27) + abs (real.sqrt 3 - 4)) = -real.sqrt 3 := 
by sorry

end problem1_problem2_l376_376309


namespace investment_growth_l376_376270

variable (x : ℝ)

theorem investment_growth (h₁ : 1500 * (1 + x)^2 = 4250) : x = sqrt (4250 / 1500) - 1 ∨ x = - sqrt (4250 / 1500) - 1 :=
by
-- Proof would go here, but it is replaced by sorry.
sorry

end investment_growth_l376_376270


namespace k_eq_2_is_sufficient_but_not_necessary_l376_376430

def is_sufficient_but_not_necessary (M : Set ℝ) (C_RM : ℝ → Set ℝ) (k : ℝ) :=
  (k = 2) → (2 ∈ C_RM 2) ∧ ¬ ∀ k, (2 ∈ C_RM 2) → (k = 2)

theorem k_eq_2_is_sufficient_but_not_necessary :
  is_sufficient_but_not_necessary ({x : ℝ | |x| > 2}) (λ x, {y : ℝ | |y| ≤ |x| ∧ y ≠ x}) 2 :=
sorry

end k_eq_2_is_sufficient_but_not_necessary_l376_376430


namespace julia_used_16_cans_l376_376844
noncomputable theory
open Classical

-- Define the initial condition: the total number of rooms initially coverable
def initial_rooms := 45

-- Define the number of cans lost
def lost_cans := 4

-- Define the reduced number of rooms after losing paint
def remaining_rooms := 36

-- Reduced capacity implies cans covering fewer rooms
def rooms_lost_per_can := (initial_rooms - remaining_rooms) / lost_cans

-- Number of cans needed to cover 36 rooms
def cans_needed := remaining_rooms / rooms_lost_per_can

-- Theorem: Prove that the number of cans used to paint 36 rooms is 16
theorem julia_used_16_cans :
  cans_needed = 16 :=
by
  simp [cans_needed, remaining_rooms, rooms_lost_per_can, initial_rooms, lost_cans]
  sorry

end julia_used_16_cans_l376_376844


namespace number_of_scooters_l376_376818

theorem number_of_scooters (b t s : ℕ) (h1 : b + t + s = 10) (h2 : 2 * b + 3 * t + 2 * s = 26) : s = 2 := 
by sorry

end number_of_scooters_l376_376818


namespace wire_cut_square_octagon_area_l376_376290

theorem wire_cut_square_octagon_area (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (equal_area : (a / 4)^2 = (2 * (b / 8)^2 * (1 + Real.sqrt 2))) : 
  a / b = Real.sqrt ((1 + Real.sqrt 2) / 2) := 
  sorry

end wire_cut_square_octagon_area_l376_376290


namespace median_of_consecutive_integers_l376_376587

theorem median_of_consecutive_integers (sum_n : ℤ) (n : ℤ) 
  (h1 : sum_n = 6^4) (h2 : n = 36) : 
  (sum_n / n) = 36 :=
by
  sorry

end median_of_consecutive_integers_l376_376587


namespace first_train_cross_post_time_l376_376222

-- Definitions for the conditions
def length_of_train : ℝ := 120
def time_second_train_cross_post : ℝ := 12
def time_trains_cross_each_other : ℝ := 10.909090909090908
def speed_second_train : ℝ := length_of_train / time_second_train_cross_post
def relative_speed_when_crossing : ℝ := length_of_train * 2 / time_trains_cross_each_other
def speed_first_train : ℝ := relative_speed_when_crossing - speed_second_train

-- The statement we want to prove
theorem first_train_cross_post_time : (length_of_train / speed_first_train) = 10 := by
  sorry

end first_train_cross_post_time_l376_376222


namespace students_in_ms_delmont_class_l376_376167

-- Let us define the necessary conditions

def total_cupcakes : Nat := 40
def students_mrs_donnelly_class : Nat := 16
def adults_count : Nat := 4 -- Ms. Delmont, Mrs. Donnelly, the school nurse, and the school principal
def leftover_cupcakes : Nat := 2

-- Define the number of students in Ms. Delmont's class
def students_ms_delmont_class : Nat := 18

-- The statement to prove
theorem students_in_ms_delmont_class :
  total_cupcakes - adults_count - students_mrs_donnelly_class - leftover_cupcakes = students_ms_delmont_class :=
by
  sorry

end students_in_ms_delmont_class_l376_376167


namespace maximum_Q_l376_376528

noncomputable def Q (b : ℝ) : ℝ :=
  ∫ x in 0..b, ∫ y in 0..1, if cos(π * x) ^ 2 + cos(π * y) ^ 2 > 3 / 2 then 1 else 0

theorem maximum_Q : ∀ b : ℝ, 0 ≤ b → b ≤ 1/2 → Q(b) = 1/3 :=
by
  intro b hb1 hb2
  sorry

end maximum_Q_l376_376528


namespace triangle_isosceles_or_right_l376_376757

theorem triangle_isosceles_or_right (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_side_constraint : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_condition: a^2 * c^2 - b^2 * c^2 = a^4 - b^4) :
  (a = b) ∨ (a^2 + b^2 = c^2) :=
by {
  sorry
}

end triangle_isosceles_or_right_l376_376757


namespace quadrilateral_angle_inequality_l376_376865

variables {A B C D A1 B1 C1 D1 : Type}
variables (ABCD : ConvexQuadrilateral A B C D) (A1B1C1D1 : ConvexQuadrilateral A1 B1 C1 D1)
variables (Angles : ℝ)
variables (a_angle : Angles)
variables (a1_angle : Angles)
variables (b_angle : Angles)
variables (b1_angle : Angles)
variables (c_angle : Angles)
variables (c1_angle : Angles)
variables (d_angle : Angles)
variables (d1_angle : Angles)

-- The side length conditions as equality of sides for corresponding sides of quadrilaterals.
variable (eq_sides : Equiv (Sides ABCD) (Sides A1B1C1D1))

-- The given angle condition.
variable (angle_cond : a_angle > a1_angle)

-- The conclusions to prove.
theorem quadrilateral_angle_inequality :
  a_angle > a1_angle →
  b_angle < b1_angle ∧
  c_angle > c1_angle ∧
  d_angle < d1_angle :=
sorry

end quadrilateral_angle_inequality_l376_376865


namespace probability_age_less_than_20_l376_376250

theorem probability_age_less_than_20 (total_people : ℕ) (over_30_years : ℕ) 
  (less_than_20_years : ℕ) (h1 : total_people = 120) (h2 : over_30_years = 90) 
  (h3 : less_than_20_years = total_people - over_30_years) : 
  (less_than_20_years : ℚ) / total_people = 1 / 4 :=
by {
  sorry
}

end probability_age_less_than_20_l376_376250


namespace cos_double_angle_value_l376_376017

variable {θ : Real}
hypothesis (h1 : 3 * sin (2 * θ) = 4 * tan θ)
hypothesis (h2 : θ ≠ k * Real.pi ∀ k : ℤ)

theorem cos_double_angle_value : cos (2 * θ) = 1 / 3 :=
by sorry

end cos_double_angle_value_l376_376017


namespace remainder_of_S_mod_500_eq_zero_l376_376124

open Function

def R : Set ℕ := { r | ∃ n : ℕ, r = (3^n % 500) }

def S : ℕ := ∑ r in R.toFinset, r

theorem remainder_of_S_mod_500_eq_zero :
  (S % 500) = 0 := by
  sorry

end remainder_of_S_mod_500_eq_zero_l376_376124


namespace triangle_angle_B_max_dot_product_k_l376_376080

-- Problem 1: Proving the magnitude of angle B
theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ)
  (h1 : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h2 : A + B + C = Real.pi)
  (h3 : 0 < A) (h4 : A < Real.pi) : B = Real.pi / 3 :=
by sorry

-- Problem 2: Proving the value of k such that the max value of dot product is 5
theorem max_dot_product_k (k : ℝ) (A : ℝ)
  (h1 : k > 1)
  (h2 : ∀ t (h_t : 0 < t ∧ t ≤ 1), 
    4 * k * Real.sin A + Real.cos (2 * A) ≤ 5)
  (h3 : (∃ t (h_t : 0 < t ∧ t ≤ 1), 
    4 * k * t + Real.cos (2 * A)) = 5) : k = 3 / 2 :=
by sorry

end triangle_angle_B_max_dot_product_k_l376_376080


namespace determine_k_l376_376986

theorem determine_k (k : ℝ) :
  ∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
  (x₁, y₁) = (6, 10) →
  (x₂, y₂) = (-2, k) →
  (x₃, y₃) = (-10, 6) →
  ((y₂ - y₁) / (x₂ - x₁) = (y₃ - y₂) / (x₃ - x₂)) →
  k = 8 :=
by
  intros x₁ y₁ x₂ y₂ x₃ y₃ h₁ h₂ h₃ h₄
  subst h₁
  subst h₂
  subst h₃
  rw [←(show y₂ = k by rfl), ←(show y₃ - y₂ = 6 - k by rfl), ←(show x₃ - x₂ = -10 + 2 by rfl)] at h₄
  dsimp at h₄
  sorry

end determine_k_l376_376986


namespace largest_distance_max_l376_376480

noncomputable def largest_distance (z : ℂ) (hz : |z| = 2) : ℝ :=
  |(5 + 2 * complex.I) * z^3 - z^4|

theorem largest_distance_max (z : ℂ) (hz : |z| = 2) :
  largest_distance z hz = 24 * Real.sqrt 29 :=
sorry

end largest_distance_max_l376_376480


namespace find_y_equal_9_l376_376549

theorem find_y_equal_9 (y : ℕ) :
  let lst := [1, 2, 3, 4, 4, 5, y]
  let mean := (list.sum lst) / (list.length lst)
  let median := list.nth (list.drop ((list.length lst - 1) / 2) (list.sort compare lst)) 0
  let mode := list.head (list.maximumBy (λ l a b, compare (list.length a) (list.length b)) (list.group lst))
  mean = median ∧ median = mode → y = 9 :=
by sorry

end find_y_equal_9_l376_376549


namespace shape_reflection_correct_l376_376616

-- Example condition definitions:
structure Shape where
  points : Finset (ℤ × ℤ)  -- For simplicity, let's assume shapes are sets of grid points.

def original_L : Shape :=
  Shape.mk {(0, 0), (0, 1), (1, 1), (2, 1)}

def reflect_diagonal (s : Shape) : Shape :=
  Shape.mk (s.points.map (λ ⟨x, y⟩, (y, x)))

def option_A : Shape :=
  Shape.mk {(0, 1), (0, 2), (1, 0), (1, 1)}

def option_B : Shape :=
  Shape.mk {(0, 1), (1, 1), (2, 1), (2, 0)}

def option_C : Shape :=
  Shape.mk {(0, 0), (0, 1), (1, 0), (2, 0)}

def option_D : Shape :=
  Shape.mk {(1, 0), (1, 1), (1, 2), (2, 2)}

def option_E : Shape :=
  Shape.mk {(0, 0), (1, 0), (1, 1), (1, 2)}

-- Stating the theorem to prove:
theorem shape_reflection_correct :
  reflect_diagonal original_L = option_D :=
by
  sorry

end shape_reflection_correct_l376_376616


namespace solve_for_x_l376_376952

theorem solve_for_x :
  ∃ x : ℝ, 40 + (5 * x) / (180 / 3) = 41 ∧ x = 12 :=
by
  sorry

end solve_for_x_l376_376952


namespace no_roots_greater_than_sqrt29_over_2_l376_376901

noncomputable def eq1_roots : set ℝ := {x | 5 * x^2 + 3 = 53}

noncomputable def eq2_roots : set ℝ := {x | (3 * x - 1)^2 = (x - 2)^2}

noncomputable def eq3_solution : set ℝ := {x | (x^2 - 9) ≥ (x - 2) ∧ x^2 - x - 7 ≥ 0}

theorem no_roots_greater_than_sqrt29_over_2 :
  (∀ x ∈ eq1_roots, x ≤ sqrt 29 / 2) ∧ 
  (∀ x ∈ eq2_roots, x ≤ sqrt 29 / 2) ∧
  (∀ x ∈ eq3_solution, x ≤ sqrt 29 / 2) :=
by sorry

end no_roots_greater_than_sqrt29_over_2_l376_376901


namespace rectangular_prism_diagonals_l376_376990

theorem rectangular_prism_diagonals (prism : Type)
  (h_faces : ∃ (F : prism → prism → prism → Prop), ∀ a b c, F a b c → id (HasSixFaces a b c prism))
  (h_edges : ∃ (E : prism → prism → Prop), ∀ a b, E a b → id (HasTwelveEdges a b prism))
  (h_diff_dims : MeasuresDifferently prism)
  (h_face_diag : ∀ (R : prism → prism → Prop) (a b : prism), IsFaceRectangle a b R → HasTwoDiagonals a b prism)
  (h_space_diag : ∃ (S : prism → prism → Prop), ∀ a, S a → HasFourSpaceDiagonals prism):
  TotalDiagonals prism = 16 := sorry

end rectangular_prism_diagonals_l376_376990


namespace terminating_decimals_l376_376357

theorem terminating_decimals (k : ℕ) (h₁ : 1 ≤ k) (h₂ : k ≤ 419) : 
  (∃ (n : ℕ), (1 ≤ n ∧ n ≤ 419 ∧ (∃ (m : ℕ), n = 21 * m) ∧ ((420 / (420 / n)).denom.factors ≤ [2, 5])) → k = 19) := sorry

end terminating_decimals_l376_376357


namespace number_of_lines_l376_376277

def Point := (ℝ × ℝ)
def Origin : Point := (0, 0)
def M : Point := (2, 1)

def line_through (p : Point) (slope : ℝ) : set Point :=
  {q | q.2 - p.2 = slope * (q.1 - p.1)}

def intersects_x_axis (l : set Point) : Point :=
  let x := classical.some (exists_mem l (λ q, q.2 = 0)) in (x.fst, 0)

def intersects_y_axis (l : set Point) : Point :=
  let y := classical.some (exists_mem l (λ q, q.1 = 0)) in (0, y.snd)

def triangle_area (a b c : Point) : ℝ :=
  |(b.1 - a.1) * (c.2 - a.2) - (b.2 - a.2) * (c.1 - a.1)| / 2

theorem number_of_lines : 
  let P := intersects_x_axis (line_through M)
  let Q := intersects_y_axis (line_through M)
  (triangle_area Origin P Q = 4) → (∃! l, line_through M l ∧ triangle_area Origin P Q = 4)
  :=
sorry

end number_of_lines_l376_376277


namespace locus_area_l376_376466

-- Define the conditions
def square (A B C D : ℝ × ℝ) : Prop :=
  (dist A B = 1) ∧ (dist B C = 1) ∧ (dist C D = 1) ∧ (dist D A = 1) ∧
  (dist A C = sqrt 2) ∧ (dist B D = sqrt 2) ∧
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ A)

def angles_90 (P1 P2 P3 : ℝ × ℝ) : Prop :=
  angle P1 P2 P3 = π / 2

def different_sides (P1 P2 P3 A B C D : ℝ × ℝ) : Prop :=
  ∃ (s1 s2 s3 : ℝ × ℝ), s1 ≠ s2 ∧ s2 ≠ s3 ∧
  s1 ≠ s3 ∧
  ((P1 = s1 ∧ P2 = s2 ∧ P3 = s3) ∨
   (P1 = s1 ∧ P3 = s2 ∧ P2 = s3) ∨
   (P2 = s1 ∧ P1 = s2 ∧ P3 = s3) ∨
   (P2 = s1 ∧ P3 = s2 ∧ P1 = s3) ∨
   (P3 = s1 ∧ P1 = s2 ∧ P2 = s3) ∨
   (P3 = s1 ∧ P2 = s2 ∧ P1 = s3))

-- Define the locus region area proof
theorem locus_area
  {A B C D P1 P2 P3 : ℝ × ℝ}
  (h_square : square A B C D)
  (h_angles : angles_90 P1 P2 P3)
  (h_diff_sides : different_sides P1 P2 P3 A B C D) :
  ∃ R : ℝ, R = (1/3) * (23 - 16 * sqrt 2) :=
sorry

end locus_area_l376_376466


namespace second_greatest_divisor_180_n_l376_376218

noncomputable def second_greatest_divisor (p q : ℕ) : ℕ :=
  let common_divisors := {1, p, q, p * q : ℕ} in
  common_divisors.filter (λ d, d < p * q).max' sorry

theorem second_greatest_divisor_180_n (n : ℕ) (h : ∀ d, d ∣ 180 ∧ d ∣ n → d ∈ {1, 2, 3, 6}) :
  second_greatest_divisor 2 3 = 3 :=
by
  unfold second_greatest_divisor
  sorry

end second_greatest_divisor_180_n_l376_376218


namespace problem_I_problem_II_l376_376094

noncomputable def C1_polar_equation (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

noncomputable def l_param_eqn (t : ℝ) : ℝ × ℝ :=
  (1 - Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t)

noncomputable def C2_param_eqn (α : ℝ) : ℝ × ℝ :=
  (3 + Real.sqrt 2 * Real.cos α, 4 + Real.sqrt 2 * Real.sin α)

theorem problem_I :
  let A := (-1 / Real.sqrt 2, 1 / Real.sqrt 2)
  let B := (1 + Real.sqrt 2, -1 - Real.sqrt 2)
  let dist := (A, B)
  dist = Real.sqrt 6 :=
sorry

theorem problem_II :
  let P (α : ℝ) := C2_param_eqn α
  let area (α : ℝ) : ℝ := 
    let A := (-1 / Real.sqrt 2, 1 / Real.sqrt 2)
    let B := (1 + Real.sqrt 2, -1 - Real.sqrt 2)
    let PA := P α - A
    let PB := P α - B
    (PA.1 * PB.2 - PA.2 * PB.1) / 2
  ∀ α : ℝ, area α ≥ 2 * Real.sqrt 3 :=
sorry

end problem_I_problem_II_l376_376094


namespace ellipse_equation_l376_376448

theorem ellipse_equation {a b : ℝ} 
  (center_origin : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → x + y = 0)
  (foci_on_x : ∀ c : ℝ, c = a / 2)
  (perimeter_triangle : ∀ A B : ℝ, A + B + 2 * c = 16) :
  a = 4 ∧ b^2 = 12 → (∀ x y : ℝ, x^2/16 + y^2/12 = 1) :=
by
  sorry

end ellipse_equation_l376_376448


namespace total_triangles_in_square_configuration_l376_376322

/-- Given a square with vertices A, B, C, D; the diagonals AC and BD; 
the midpoints of each side (M, N, P, Q for sides AB, BC, CD, DA respectively); 
and the center of the square (O), proving that the total number of triangles 
formed by the new figures (including lines OA, OB, OC, OD, OM, ON, OP, OQ) is 40. -/
theorem total_triangles_in_square_configuration (A B C D O M N P Q : Point)
  (h_square : square A B C D)
  (h_center : center_of_square O A B C D)
  (h_midpoints : 
    midpoint M A B ∧ midpoint N B C ∧ midpoint P C D ∧ midpoint Q D A)
  (h_lines : 
    line O A ∧ line O B ∧ line O C ∧ line O D ∧ 
    line O M ∧ line O N ∧ line O P ∧ line O Q) : 
  count_triangles (configuration A B C D O M N P Q) = 40 :=
sorry

end total_triangles_in_square_configuration_l376_376322


namespace general_term_formula_l376_376400

def seq (n : ℕ) : ℤ :=
  match n with
  | 1     => 2
  | 2     => -6
  | 3     => 12
  | 4     => -20
  | 5     => 30
  | 6     => -42
  | _     => 0 -- We match only the first few elements as given

theorem general_term_formula (n : ℕ) :
  seq n = (-1)^(n+1) * n * (n + 1) := by
  sorry

end general_term_formula_l376_376400


namespace exists_sum_of_two_squares_l376_376511

theorem exists_sum_of_two_squares (n : ℤ) (h : n > 10000) : ∃ m : ℤ, (∃ a b : ℤ, m = a^2 + b^2) ∧ 0 < m - n ∧ m - n < 3 * n^(1/4) :=
by
  sorry

end exists_sum_of_two_squares_l376_376511


namespace exceptional_numbers_count_l376_376963

-- Definitions for iterative rules and the concept of exceptional numbers
def apply_rule (n : ℕ) : ℕ :=
  if n ≤ 30 then 3 * n else n - 15

def is_exceptional (F : ℕ) : Prop :=
  ¬ ∃ (n : ℕ), (apply_rule^[n] F) = 21

/-- 
  There are 7 exceptional numbers between 1 and 100.
-/
theorem exceptional_numbers_count : 
  (finset.filter is_exceptional (finset.range 101)).card = 7 :=
sorry

end exceptional_numbers_count_l376_376963


namespace sum_of_powers_modulo_l376_376130

theorem sum_of_powers_modulo (R : Finset ℕ) (S : ℕ) :
  (∀ n < 100, ∃ r, r ∈ R ∧ r = 3^n % 500) →
  S = R.sum id →
  (S % 500) = 0 :=
by {
  -- Proof would go here
  sorry
}

end sum_of_powers_modulo_l376_376130


namespace find_equation_of_ellipse_find_equation_of_line_l376_376036

-- Define the given conditions
variables (a b : ℝ)
def ellipse (a : ℝ) (b : ℝ) := { p : ℝ × ℝ // (p.1 ^ 2) / a ^ 2 + (p.2 ^ 2) / b ^ 2 = 1 }

-- The given points are the foci of the ellipse
def F1 := (-1, 0) : ℝ × ℝ
def F2 := (1, 0) : ℝ × ℝ

-- The perimeter of triangle ABF1
def perimeter_ΔABF1 := 4 * sqrt 3

-- Define a line l passing through F2
variable (l : ℝ × ℝ → Prop)
def passes_through_F2 : Prop := l F2

-- Define the orthogonality condition
def OA_orthogonal_OB (A B : ℝ × ℝ) : Prop := 
  (A.1 * B.1 + A.2 * B.2 = 0)

-- Statements
theorem find_equation_of_ellipse (h1 : a > b) (h2 : b > 0) (h3 : a = sqrt 3) (h4 : b = sqrt 2) :
  ∀ x y, ((x ^ 2) / 3 + (y ^ 2) / 2 = 1) → 
  ellipse a b = { p : ℝ × ℝ // (p.1 ^ 2) / 3 + (p.2 ^ 2) / 2 = 1 } :=
sorry

theorem find_equation_of_line (A B : ℝ × ℝ) 
  (h1 : passes_through_F2 l) 
  (h2 : OA_orthogonal_OB A B)
  (h3 : ellipse a b) :
  l = λ p, p.2 = sqrt 2 * (p.1 - 1) ∨ p.2 = -sqrt 2 * (p.1 - 1) :=
sorry

end find_equation_of_ellipse_find_equation_of_line_l376_376036


namespace expr_containing_x_to_y_l376_376893

theorem expr_containing_x_to_y (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  -- proof steps would be here
  sorry

end expr_containing_x_to_y_l376_376893


namespace midpoint_intersection_of_common_tangent_and_radical_axis_l376_376871

theorem midpoint_intersection_of_common_tangent_and_radical_axis
  (Γ1 Γ2 : Circle) (A B P Q : Point)
  (h₁ : intersects Γ1 Γ2 A B)
  (h₂ : tangent_to_circle P Γ1)
  (h₃ : tangent_to_circle Q Γ2)
  (h₄ : common_tangent_line_tangent_to P Q Γ1 Γ2) :
  midpoint (intersection_segment (line_through_points A B) (line_segment P Q))
  = intersection_segment (line_through_points A B) (line_segment P Q) :=
sorry

end midpoint_intersection_of_common_tangent_and_radical_axis_l376_376871


namespace max_and_min_value_of_divided_number_l376_376651

theorem max_and_min_value_of_divided_number :
  ∀ (divisor quotient : ℕ) (remainder_max remainder_min : ℕ),
  divisor = 17 → quotient = 25 → remainder_max = 16 → remainder_min = 1 →
  (divisor * quotient + remainder_max = 441) ∧ (divisor * quotient + remainder_min = 426) :=
by
  intros divisor quotient remainder_max remainder_min Hd Hq Hr_max Hr_min
  rw [Hd, Hq, Hr_max, Hr_min]
  split
  · simp
  · simp
  sorry

end max_and_min_value_of_divided_number_l376_376651


namespace pairing_number_with_13_l376_376552

theorem pairing_number_with_13 :
  ∃ (A B C D E F : ℕ), {41, 35, 19, 9, 26, 45, 28} = {A, B, C, D, E, F} ∧
                      (A + B = 54 ∧ C + D = 54 ∧ E + F = 54) ∧ 13 + 41 = 54 := sorry

end pairing_number_with_13_l376_376552


namespace intersection_eq_l376_376490

noncomputable def U : Set ℝ := Set.univ
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 3}

-- Complement of B in U
def complement_B : Set ℝ := {x | x < 2 ∨ x ≥ 3}

-- Intersection of A and complement of B
def intersection : Set ℕ := {x ∈ A | ↑x < 2 ∨ ↑x ≥ 3}

theorem intersection_eq : intersection = {1, 3, 4} :=
by
  sorry

end intersection_eq_l376_376490


namespace correct_statements_l376_376295

-- Define the conditions as boolean propositions
def C1 : Prop := ∀ (c : Concentration), promotes_cell_elongation c
def C2 : Prop := ∀ (h₁ h₂ : Hormone), interacts h₁ h₂
def C3 : Prop := ∀ (c₁ c₂ : Concentration), ⟨roots_eq c₁ c₂⟩
def C4 : Prop := ∀ (p : Plant), microgravity_space p → ¬polar_transport_hormones p

-- Define the set of true conditions
def true_conditions : set Prop := {C1, C2}

-- The problem is equivalent to proving the set of conditions matches true_conditions
theorem correct_statements : {C1, C2} = true_conditions := by sorry

end correct_statements_l376_376295


namespace problem_solution_l376_376004

theorem problem_solution (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : a + b = 1) :
  (a + 1 / b) ^ 2 + (b + 1 / a) ^ 2 ≥ 25 / 2 :=
sorry

end problem_solution_l376_376004


namespace triangle_longest_side_l376_376087

open Real

theorem triangle_longest_side (a b c : ℝ) (r : ℝ) (x : ℝ) (hx : 16.5 + 1.5 * x = 21) 
  (h1 : a = 21) (h2 : b = 2 * x) (h3 : c = x + 12)
  (h4 : r = 5) : 
  max (max a b) c = 21 := 
by
  -- Geometry conditions
  have s := (a + b + c) / 2
  have s_eq : s = 16.5 + 1.5 * x, from hx

  -- Use Herons formula to calculate area
  let delta := sqrt (s * (s - a) * (s - b) * (s - c))
  
  -- Use area and radius relation
  have area_radius_relation : r = delta / s, from h4
  
  -- Given lengths satisfy the equation derived in solution
  have length_eq : a + b = 21 ∧ b = 2 * x ∧ c = x + 12, from ⟨h1, h2, h3⟩ 
  
  -- Therefore, the longest side is 21 units 
  exact sorry -- Proof steps omitted

end triangle_longest_side_l376_376087


namespace win_rate_product_correct_l376_376170

noncomputable def product_of_win_rates : ℚ := 
  (∏ n in Finset.range 2015 \ {0}, (n : ℚ) / (n + 1))

theorem win_rate_product_correct : 
  product_of_win_rates = 1 / 2015 := 
by
  sorry

end win_rate_product_correct_l376_376170


namespace find_nm_l376_376853

def vec_u (n : ℚ) : ℚ × ℚ × ℚ := (4, n, -2)
def vec_v (m : ℚ) : ℚ × ℚ × ℚ := (1, 2, m)

def orthogonal (u v : ℚ × ℚ × ℚ) : Prop :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0

theorem find_nm (n m : ℚ) :
  orthogonal (vec_u n) (vec_v m) →
  n = 11/4 ∧ m = 19/4 :=
  sorry

end find_nm_l376_376853


namespace mandy_total_cost_after_discount_l376_376143

-- Define the conditions
def packs_black_shirts : ℕ := 6
def packs_yellow_shirts : ℕ := 8
def packs_green_socks : ℕ := 5

def items_per_pack_black_shirts : ℕ := 7
def items_per_pack_yellow_shirts : ℕ := 4
def items_per_pack_green_socks : ℕ := 5

def cost_per_pack_black_shirts : ℕ := 25
def cost_per_pack_yellow_shirts : ℕ := 15
def cost_per_pack_green_socks : ℕ := 10

def discount_rate : ℚ := 0.10

-- Calculate the total number of each type of item
def total_black_shirts : ℕ := packs_black_shirts * items_per_pack_black_shirts
def total_yellow_shirts : ℕ := packs_yellow_shirts * items_per_pack_yellow_shirts
def total_green_socks : ℕ := packs_green_socks * items_per_pack_green_socks

-- Calculate the total cost before discount
def total_cost_before_discount : ℕ :=
  (packs_black_shirts * cost_per_pack_black_shirts) +
  (packs_yellow_shirts * cost_per_pack_yellow_shirts) +
  (packs_green_socks * cost_per_pack_green_socks)

-- Calculate the total cost after discount
def discount_amount : ℚ := discount_rate * total_cost_before_discount
def total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount

-- Problem to prove: Total cost after discount is $288
theorem mandy_total_cost_after_discount : total_cost_after_discount = 288 := by
  sorry

end mandy_total_cost_after_discount_l376_376143


namespace probability_blue_or_purple_l376_376979

def total_jelly_beans : ℕ := 35
def blue_jelly_beans : ℕ := 7
def purple_jelly_beans : ℕ := 10

theorem probability_blue_or_purple : (blue_jelly_beans + purple_jelly_beans: ℚ) / total_jelly_beans = 17 / 35 := 
by sorry

end probability_blue_or_purple_l376_376979


namespace probability_point_in_smaller_ellipse_l376_376883

namespace ProbabilityOfPointInEllipse

def area_of_ellipse (a b : ℝ) : ℝ := π * a * b

theorem probability_point_in_smaller_ellipse :
  let a_G := 7 in
  let b_G := 4 in
  let a_g := 5 in
  let b_g := 3 in 
  let S_G := area_of_ellipse a_G b_G in
  let S_g := area_of_ellipse a_g b_g in
  S_g / S_G = (3:ℝ)/(7:ℝ) :=
by
  trivial  -- This line is used as a placeholder for the proof
sorry

end ProbabilityOfPointInEllipse

end probability_point_in_smaller_ellipse_l376_376883


namespace probability_both_selected_l376_376234

theorem probability_both_selected 
  (p_jamie : ℚ) (p_tom : ℚ) 
  (h1 : p_jamie = 2/3) 
  (h2 : p_tom = 5/7) : 
  (p_jamie * p_tom = 10/21) :=
by
  sorry

end probability_both_selected_l376_376234


namespace area_of_intersection_l376_376945

theorem area_of_intersection (XY YE XF EX FY : ℤ)
  (h1 : XY = 15)
  (h2 : YE = 12)
  (h3 : XF = 12)
  (h4 : EX = 13)
  (h5 : FY = 13)
  (h6 : ∆ XYE ≅ ∆ XYF) :
  ∃ (p q : ℕ), p + q = 75 :=
by
  sorry

end area_of_intersection_l376_376945


namespace total_students_l376_376082

variable A : ℕ
variable glasses : ℕ

-- Conditions
axiom poor_vision_proportion : Float
axiom glasses_proportion : Float
axiom glasses_students : ℕ
axiom glasses_condition : 0.28 * A = 21

-- Proof statement
theorem total_students (h1 : poor_vision_proportion = 0.4)
                      (h2 : glasses_proportion = 0.7)
                      (h3 : glasses_students = 21)
                      (h4 : glasses_condition) :
                      A = 75 :=
sorry

end total_students_l376_376082


namespace mod_equiv_example_l376_376526

theorem mod_equiv_example : (185 * 944) % 60 = 40 := by
  sorry

end mod_equiv_example_l376_376526


namespace probability_sum_greater_than_six_l376_376093

variable (A : Finset ℕ) (B : Finset ℕ)
variable (balls_in_A : A = {1, 2}) (balls_in_B : B = {3, 4, 5, 6})

theorem probability_sum_greater_than_six : 
  (∃ selected_pair ∈ (A.product B), selected_pair.1 + selected_pair.2 > 6) →
  (Finset.filter (λ pair => pair.1 + pair.2 > 6) (A.product B)).card / 
  (A.product B).card = 3 / 8 := sorry

end probability_sum_greater_than_six_l376_376093


namespace tangent_line_eq_l376_376543

theorem tangent_line_eq {f : ℝ → ℝ} (h_f : ∀ x, f x = x * real.log x) :
  ∀ x y, (x = 1) ∧ (y = 0) → x - y - 1 = 0 :=
by
  intros x y h
  cases h with hx hy
  rw [hx, hy]
  exact eq.refl 0

end tangent_line_eq_l376_376543


namespace prove_simplification_l376_376897

def simplification_problem : Prop :=
  (sqrt (32 ^ (1 / 5)) - sqrt 7) ^ 2 = 11 - 4 * sqrt 7

theorem prove_simplification : simplification_problem :=
by
  sorry

end prove_simplification_l376_376897


namespace f_comp_f3_eq_11_l376_376392

def f (x : ℝ) : ℝ :=
  if x < 4 then 2^(x-1) else 2*x + 3

theorem f_comp_f3_eq_11 : f (f 3) = 11 :=
by
  sorry

end f_comp_f3_eq_11_l376_376392


namespace quadratic_function_conclusions_correct_count_l376_376554

/-- Given the quadratic function y = ax^2 + bx + c, where a ≠ 0, and the table of values:
    x |  -3  |-2  |-1  | 0  | 1  | 2  | 3  | 4  | 5
    y |  12  | 5  | 0  |-3 | -4 | -3 | 0  | 5  | 12
  Prove that the number of correct conclusions among the following is 2:
  1. The quadratic function y = ax^2 + bx + c has a minimum value of -3.
  2. When -1/2 < x < 2, y < 0.
  3. The graph of the quadratic function intersects the x-axis at two points, and they are on opposite sides of the y-axis.
-/
theorem quadratic_function_conclusions_correct_count 
  (a b c : ℝ) 
  (h_a : a ≠ 0) 
  (values : ∀ (x : ℝ), x ∈ {-3, -2, -1, 0, 1, 2, 3, 4, 5} → y = a * x^2 + b * x + c)
  (concl_1 : (∀ (x : ℝ), x ∈ {1} → y != -3))
  (concl_2 : (∀ (x : ℝ), -1/2 < x ∧ x < 2 → y < 0))
  (concl_3 : (∀ (x1 x2 : ℝ), x1 = -1 ∧ x2 = 3 → x1 ≠ x2 ∧ (a * x1^2 + b * x1 + c = 0) ∧ (a * x2^2 + b * x2 + c = 0)))
  : (concl_1 = false) + (concl_2 = true) + (concl_3 = true) = 2 :=
sorry

end quadratic_function_conclusions_correct_count_l376_376554


namespace reflection_correct_l376_376352

open Matrix

def vector_1 : Vector ℝ 2 := ![1, 5]
def vector_2 : Vector ℝ 2 := ![-2, 4]

def dot_product (v1 v2 : Vector ℝ 2) : ℝ := (v1 0) * (v2 0) + (v1 1) * (v2 1)

-- Definition of the projection of vector_1 onto vector_2
def projection (v1 v2 : Vector ℝ 2) : Vector ℝ 2 :=
  let scalar := (dot_product v1 v2) / (dot_product v2 v2)
  ![(scalar * (v2 0)), (scalar * (v2 1))]

-- Definition of the reflection of vector_1 over vector_2
def reflection : Vector ℝ 2 :=
  let p := projection vector_1 vector_2
  ![2 * (p 0) - (vector_1 0), 2 * (p 1) - (vector_1 1)]

theorem reflection_correct : reflection = ![-23/5, 11/5] := by
  sorry

end reflection_correct_l376_376352


namespace median_of_consecutive_integers_sum_eq_6_pow_4_l376_376568

theorem median_of_consecutive_integers_sum_eq_6_pow_4 :
  ∀ (s : ℕ) (n : ℕ), s = 36 → ∑ i in finset.range 36, (n + i) = 6^4 → 36 / 2 = 36 :=
by
  sorry

end median_of_consecutive_integers_sum_eq_6_pow_4_l376_376568


namespace only_option_A_is_quadratic_l376_376617

-- Define what it means to be a quadratic equation in one variable
def is_quadratic_in_one_variable (eq : ℤ → Prop) : Prop :=
  ∃ a b c : ℤ, (a ≠ 0) ∧ (∀ x : ℤ, eq x ↔ a * x^2 + b * x + c = 0)

-- Define the four options
def option_A : ℤ → Prop := λ x, x^2 + 1 = 0
def option_B : ℤ × ℤ → Prop := λ p, let (x, y) := p in x^2 + y = 5
def option_C : ℤ → Prop := λ x, x^(-2) + x = 0
def option_D : ℤ → Prop := λ x, x^2 - x = 3 + x^2

-- Prove that only option A is a quadratic equation in one variable
theorem only_option_A_is_quadratic : 
  (is_quadratic_in_one_variable option_A) ∧ 
  ¬(is_quadratic_in_one_variable (λ x, ∃ y, option_B (x, y))) ∧ 
  ¬(is_quadratic_in_one_variable option_C) ∧ 
  ¬(is_quadratic_in_one_variable option_D) :=
by
  sorry

end only_option_A_is_quadratic_l376_376617


namespace large_pyramid_tiers_l376_376589

def surface_area_pyramid (n : ℕ) : ℕ :=
  4 * n^2 + 2 * n

theorem large_pyramid_tiers :
  (∃ n : ℕ, surface_area_pyramid n = 42) →
  (∃ n : ℕ, surface_area_pyramid n = 2352) →
  ∃ n : ℕ, surface_area_pyramid n = 2352 ∧ n = 24 :=
by
  sorry

end large_pyramid_tiers_l376_376589


namespace algebraic_expression_value_l376_376002

theorem algebraic_expression_value {x : ℝ} (h : x * (x + 2) = 2023) : 2 * (x + 3) * (x - 1) - 2018 = 2022 := 
by 
  sorry

end algebraic_expression_value_l376_376002


namespace four_boxes_l376_376202

universe u

constant Box : Type u
constants (Card AceOfHearts AceOfClubs AceOfDiamonds AceOfSpades : Box)

axiom h1 : ∀ (x : Box), x = AceOfHearts ∨ x = AceOfClubs ∨ x = AceOfDiamonds ∨ x = AceOfSpades
  
def guesses (x1 x2 x3 x4 : Box) :=
  ((x1 = AceOfClubs ∧ x3 ≠ AceOfDiamonds) ∨ (x1 ≠ AceOfClubs ∧ x3 = AceOfDiamonds)) ∧
  ((x2 = AceOfClubs ∧ x3 ≠ AceOfSpades) ∨ (x2 ≠ AceOfClubs ∧ x3 = AceOfSpades)) ∧
  ((x4 = AceOfSpades ∧ x2 ≠ AceOfDiamonds) ∨ (x4 ≠ AceOfSpades ∧ x2 = AceOfDiamonds)) ∧
  ((x4 = AceOfHearts ∧ x3 ≠ AceOfDiamonds) ∨ (x4 ≠ AceOfHearts ∧ x3 = AceOfDiamonds))

-- Given the guesses are half correct, prove that the 4th box contains Ace of Hearts or Ace of Spades
theorem four_boxes (x1 x2 x3 x4 : Box) (h : guesses x1 x2 x3 x4) : x4 = AceOfHearts ∨ x4 = AceOfSpades :=
by
  sorry

end four_boxes_l376_376202


namespace minimum_distance_proof_l376_376024

noncomputable def minimum_distance_AB : ℝ :=
  let f (x : ℝ) := x^2 - Real.log x
  let x_min := Real.sqrt 2 / 2
  let min_dist := (5 + Real.log 2) / 4
  min_dist

theorem minimum_distance_proof :
  ∃ a : ℝ, a = minimum_distance_AB :=
by
  use (5 + Real.log 2) / 4
  sorry

end minimum_distance_proof_l376_376024


namespace coloring_ways_l376_376633

theorem coloring_ways :
  ∃ n : ℕ, n = 4^2 ∧ (∀ i j : fin 4, ∃! x : fin 2, x = 2) → n = 90 :=
begin
  sorry
end

end coloring_ways_l376_376633


namespace ratio_of_wire_pieces_l376_376289

theorem ratio_of_wire_pieces (a b : ℝ) (h_equal_areas : (a / 4) ^ 2 = 2 * (1 + real.sqrt 2) * (b / 8) ^ 2) :
  a / b = real.sqrt (2 + real.sqrt 2) / 2 := 
by
  sorry

end ratio_of_wire_pieces_l376_376289


namespace abs_x_plus_7_eq_0_has_no_solution_l376_376615

theorem abs_x_plus_7_eq_0_has_no_solution : ¬∃ x : ℝ, |x| + 7 = 0 :=
by
  sorry

end abs_x_plus_7_eq_0_has_no_solution_l376_376615


namespace cylinder_ratio_l376_376373

theorem cylinder_ratio (V : ℝ) (π : ℝ) (H R : ℝ) (A : ℝ)
  (hV : V = 500 * π)
  (hA : A = 2 * π * R^2 + 2 * π * R * H)
  (h_min_A : ∀ (r h : ℝ), (hV : V = π * r^2 * h) → 
                              (surface_area : surface_area = 2 * π * r^2 + 2 * π * r * h) → 
                              A ≤ surface_area) :
  H / R = 2 := 
sorry

end cylinder_ratio_l376_376373


namespace sally_cards_l376_376518

variable (initial_amount : Int)

theorem sally_cards (sold : Int) (given : Int) (bought : Int) : 
  sold = 27 → given = 41 → bought = 20 → 
  initial_amount - sold + given + bought = initial_amount + 34 := 
by
  intros hs hg hb
  rw [hs, hg, hb]
  calc 
    initial_amount - 27 + 41 + 20 
      = initial_amount + (-27 + 61) : by sorry
      = initial_amount + 34 : by sorry

end sally_cards_l376_376518


namespace cannot_take_value_l376_376730

theorem cannot_take_value (x y : ℝ) (h : |x| + |y| = 13) : 
  ∀ (v : ℝ), x^2 + 7*x - 3*y + y^2 = v → (0 ≤ v ∧ v ≤ 260) := 
by
  sorry

end cannot_take_value_l376_376730


namespace collinear_and_parallel_lines_l376_376832

open_locale classical

variables {A B C A₀ B₀ A₁ B₁ Mₐ Mᵦ Mₓ : Type}
variables [geometry_type A B C A₀ B₀ A₁ B₁ Mₐ Mᵦ Mₓ]

/-- Define medians and altitudes --/
def is_median (A₀ A B C : A) : Prop := sorry
def is_median (B₀ A B C : B) : Prop := sorry
def is_altitude (A₁ A B C : A) : Prop := sorry
def is_altitude (B₁ A B C : B) : Prop := sorry

/-- Define circumcircle intersections --/
def circumcircle_intersections (C A₀ B₀ A₁ B₁ Mₓ : A) : Prop :=
  circumcircle (triangle.of_points C A₀ B₀) Mₓ ∧ circumcircle (triangle.of_points C A₁ B₁) Mₓ

/- Main proof statement that we need to prove -/
theorem collinear_and_parallel_lines :
  is_median A A₀ B C →
  is_median B B₀ A C →
  is_altitude A A₁ B C →
  is_altitude B B₁ A C →
  circumcircle_intersections C A₀ B₀ A₁ B₁ Mₐ →
  circumcircle_intersections B A₀ B₀ A₁ B₁ Mᵦ →
  circumcircle_intersections A A₀ B₀ A₁ B₁ Mₓ →
  (collinear Mₐ Mᵦ Mₓ) ∧ (parallel (line_through A Mₐ) (line_through B Mᵦ)) ∧ (parallel (line_through B Mᵦ) (line_through C Mₓ)) :=
by
  intros
  sorry

end collinear_and_parallel_lines_l376_376832


namespace range_of_a_l376_376374

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h_increasing : ∀ n : ℕ+, f n < f (n + 1))
  (f : ℝ → ℝ := λ x, if x ≤ 2 then (3 - a) * x + 2 else a ^ (2 * x ^ 2 - 9 * x + 11)) :
  2 < a ∧ a < 3 :=
sorry

end range_of_a_l376_376374


namespace ordering_of_abc_l376_376739

open Real

noncomputable def a := (ln 2) / 2
noncomputable def b := (ln 3) / 3
noncomputable def c := 1 / exp 1

theorem ordering_of_abc : a < b ∧ b < c := by
  -- sorry is used to skip the proof part as instructed
  sorry

end ordering_of_abc_l376_376739


namespace min_value_at_2_l376_376971

noncomputable def min_value (x : ℝ) := x + 4 / x + 5

theorem min_value_at_2 (x : ℝ) (h : x > 0) : min_value x ≥ 9 :=
sorry

end min_value_at_2_l376_376971


namespace shari_effective_distance_l376_376896

-- Define the given conditions
def constant_rate : ℝ := 4 -- miles per hour
def wind_resistance : ℝ := 0.5 -- miles per hour
def walking_time : ℝ := 2 -- hours

-- Define the effective walking speed considering wind resistance
def effective_speed : ℝ := constant_rate - wind_resistance

-- Define the effective walking distance
def effective_distance : ℝ := effective_speed * walking_time

-- State that Shari effectively walks 7.0 miles
theorem shari_effective_distance :
  effective_distance = 7.0 :=
by
  sorry

end shari_effective_distance_l376_376896


namespace sin_405_eq_sqrt_2_div_2_l376_376318

theorem sin_405_eq_sqrt_2_div_2 : sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_eq_sqrt_2_div_2_l376_376318


namespace geometric_sequence_a2_l376_376009

theorem geometric_sequence_a2 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h1 : a 1 = 1/4) 
  (h3_h5 : a 3 * a 5 = 4 * (a 4 - 1)) 
  (h_seq : ∀ n : ℕ, a n = a 1 * q ^ (n - 1)) :
  a 2 = 1/2 :=
sorry

end geometric_sequence_a2_l376_376009


namespace second_pumpkin_weight_l376_376495

variable weight_pumpkin1 : ℝ := 4
variable total_weight : ℝ := 12.7 
variable weight_pumpkin2: ℝ := 8.7

theorem second_pumpkin_weight : (total_weight - weight_pumpkin1) = weight_pumpkin2 := 
by 
  sorry

end second_pumpkin_weight_l376_376495


namespace area_of_triangle_ABC_l376_376884

noncomputable def distance (a b : ℝ × ℝ) : ℝ := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

theorem area_of_triangle_ABC (A B C O : ℝ × ℝ)
  (h_isosceles_right : ∃ d: ℝ, distance A B = d ∧ distance A C = d ∧ distance B C = Real.sqrt (2 * d^2))
  (h_A_right : A = (0, 0))
  (h_OA : distance O A = 5)
  (h_OB : distance O B = 7)
  (h_OC : distance O C = 3) :
  ∃ S : ℝ, S = (29 / 2) + (5 / 2) * Real.sqrt 17 :=
sorry

end area_of_triangle_ABC_l376_376884


namespace median_of_36_consecutive_integers_l376_376582

theorem median_of_36_consecutive_integers (f : ℕ → ℤ) (h_consecutive : ∀ n : ℕ, f (n + 1) = f n + 1) 
(h_size : ∃ k, f 36 = f 0 + 35) (h_sum : ∑ i in finset.range 36, f i = 6^4) : 
(∃ m, m = f (36 / 2 - 1) ∧ m = 36) :=
by
  sorry

end median_of_36_consecutive_integers_l376_376582


namespace area_of_region_is_correct_l376_376201

noncomputable def area_of_region : ℝ :=
  let x := (3.5 : ℝ)
  let y := (1.5 : ℝ)
  let base := Real.sqrt ((3.5 - 2) ^ 2 + (1.5 - 0) ^ 2)
  let height := (5 : ℝ) - (Real.abs (0 - 2))
  (1 / 2) * base * height

theorem area_of_region_is_correct :
  (∃ x y : ℝ, (|x - 2| ≤ y) ∧ (y ≤ 5 - |x|)) →
  area_of_region = 2.25 * Real.sqrt 2 :=
by
  sorry

end area_of_region_is_correct_l376_376201


namespace participants_won_more_than_lost_l376_376558

-- Define the conditions given in the problem
def total_participants := 64
def rounds := 6

-- Define a function that calculates the number of participants reaching a given round
def participants_after_round (n : Nat) (r : Nat) : Nat :=
  n / (2 ^ r)

-- The theorem we need to prove
theorem participants_won_more_than_lost :
  participants_after_round total_participants 2 = 16 :=
by 
  -- Provide a placeholder for the proof
  sorry

end participants_won_more_than_lost_l376_376558


namespace triangle_altitude_l376_376067

theorem triangle_altitude
  (base : ℝ) (height : ℝ) (side : ℝ)
  (h_base : base = 6)
  (h_side : side = 6)
  (area_triangle : ℝ) (area_square : ℝ)
  (h_area_square : area_square = side ^ 2)
  (h_area_equal : area_triangle = area_square)
  (h_area_triangle : area_triangle = (base * height) / 2) :
  height = 12 := 
by
  sorry

end triangle_altitude_l376_376067


namespace water_added_to_solution_l376_376978

-- Provide the main statement to be proven
theorem water_added_to_solution (V_initial : ℝ) (P_initial : ℝ) (A_added : ℝ) (P_final : ℝ) (W : ℝ) :
  V_initial = 40 ∧ P_initial = 0.05 ∧ A_added = 3.5 ∧ P_final = 0.11 ∧
  5.5 = 0.11 * (43.5 + W) →
  W ≈ 6.5 :=
begin
  sorry
end

end water_added_to_solution_l376_376978


namespace sum_of_all_possible_values_of_g_11_l376_376855

def f (x : ℝ) : ℝ := x^2 - 6 * x + 14

def g (x : ℝ) : ℝ := 3 * x + 4

theorem sum_of_all_possible_values_of_g_11 :
  (∀ x : ℝ, f x = 11 → g x = 13 ∨ g x = 7) →
  (13 + 7 = 20) := by
  intros h
  sorry

end sum_of_all_possible_values_of_g_11_l376_376855


namespace rhombus_diagonal_length_l376_376187

theorem rhombus_diagonal_length 
  (d1 : ℝ) (d2 : ℝ) (p : ℝ) (s : ℝ) : 
  d1 = 24 → 
  p = 52 → 
  s = p / 4 → 
  (d1 / 2)^2 + (d2 / 2)^2 = s^2 → 
  d2 = 10 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end rhombus_diagonal_length_l376_376187


namespace radius_of_circle_l376_376560

theorem radius_of_circle (ρ θ : ℝ) (φ : ℝ) (h₁ : ρ = 2) (h₂ : φ = π / 4) : 
  sqrt ((ρ * sin φ) ^ 2) = sqrt 2 := 
by {
  sorry
}

end radius_of_circle_l376_376560


namespace greatest_integer_not_exceeding_a_l376_376389

theorem greatest_integer_not_exceeding_a (a : ℝ) (h : 3^a + a^3 = 123) : ⌊a⌋ = 4 :=
sorry

end greatest_integer_not_exceeding_a_l376_376389


namespace extremum_at_one_determines_a_l376_376914

theorem extremum_at_one_determines_a (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = x^3 - a*x) ∧ (f' 1 = 0)) → a = 3 :=
by
  sorry

end extremum_at_one_determines_a_l376_376914


namespace perimeter_difference_of_octagon_l376_376828

noncomputable def sqrt2 := Real.sqrt 2

theorem perimeter_difference_of_octagon :
  (∃ (octagon : Fin 8 → ℝ × ℝ), 
    -- The octagon is named COMPUTER with all internal angles being either 90° or 270°
    (∀ i, 
      let θ := (octagon i).2 - (octagon (i + 7) % 8).2
      in θ = 90 ∨ θ = 270) ∧
    -- Given side lengths CO = OM = MP = PU = UT = TE = sqrt(2)
    (octagon 1) - (octagon 0) = (sqrt2,0) ∧
    (octagon 2) - (octagon 1) = (sqrt2,0) ∧
    (octagon 3) - (octagon 2) = (sqrt2,0) ∧
    (octagon 4) - (octagon 3) = (sqrt2,0) ∧
    (octagon 5) - (octagon 4) = (sqrt2,0) ∧
    (octagon 6) - (octagon 5) = (sqrt2,0) ∧
    (octagon 7) - (octagon 6) = (sqrt2,0) ∧
    -- Points A and B divide the octagon into two regions of equal area
    -- We consider these points to equidivide the octagon by area
    -- We need the difference in perimeter of the regions 
    difference_in_perimeters (octagon, A, B) = 2 * sqrt2) :=
sorry

end perimeter_difference_of_octagon_l376_376828


namespace exists_two_distinct_nice_subsets_and_const_BA_BA_l376_376849

-- Define the points B and C
def B : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (1, 0)

-- Define what it means for a subset of the plane to be nice
def isNice (S : set (ℝ × ℝ)) : Prop :=
  (∃ T ∈ S, ∀ Q ∈ S, segment T Q ⊆ S) ∧
  (∀ (P1 P2 P3 : ℝ × ℝ), ∃! (A : ℝ × ℝ) (σ : Perm (Fin 3)), 
    A ∈ S ∧ Similar (triangleA : triangles ABC) (triangleP : triangles (Pσ 0) (Pσ 1) (Pσ 2)))

-- The main theorem
theorem exists_two_distinct_nice_subsets_and_const_BA_BA' :
  ∃ (S S' : set (ℝ × ℝ)), S ≠ S' ∧ isNice S ∧ isNice S' ∧ ∀ (A : ℝ × ℝ) (A' : ℝ × ℝ),
    (A ∈ S) → (A' ∈ S') → (∀ (P1 P2 P3 : ℝ × ℝ), 
      unique_choice_in_condition_ii A S P1 P2 P3 ∧ 
      unique_choice_in_condition_ii A' S' P1 P2 P3) →
    dist B A * dist B A' = const :=
sorry

end exists_two_distinct_nice_subsets_and_const_BA_BA_l376_376849


namespace median_of_consecutive_integers_sum_eq_6_pow_4_l376_376570

theorem median_of_consecutive_integers_sum_eq_6_pow_4 :
  ∀ (s : ℕ) (n : ℕ), s = 36 → ∑ i in finset.range 36, (n + i) = 6^4 → 36 / 2 = 36 :=
by
  sorry

end median_of_consecutive_integers_sum_eq_6_pow_4_l376_376570


namespace at_least_one_angle_ge_60_l376_376212

theorem at_least_one_angle_ge_60 (A B C : ℝ) (hA : A < 60) (hB : B < 60) (hC : C < 60) (h_sum : A + B + C = 180) : false :=
sorry

end at_least_one_angle_ge_60_l376_376212


namespace fairness_of_game_l376_376262

noncomputable def is_game_unfair : Prop :=
  let deck := [3, 4, 5, 6]
  let possible_numbers := (deck.product deck).map (λ (x : ℕ × ℕ), x.1 * 10 + x.2)
  let wins_for_A := possible_numbers.count (λ n, n < 45)
  let wins_for_B := possible_numbers.count (λ n, n ≥ 45)
  wins_for_A ≠ wins_for_B

theorem fairness_of_game : is_game_unfair :=
  sorry

end fairness_of_game_l376_376262


namespace total_points_scored_l376_376442

theorem total_points_scored
    (Bailey_points Chandra_points Akiko_points Michiko_points : ℕ)
    (h1 : Bailey_points = 14)
    (h2 : Michiko_points = Bailey_points / 2)
    (h3 : Akiko_points = Michiko_points + 4)
    (h4 : Chandra_points = 2 * Akiko_points) :
  Bailey_points + Michiko_points + Akiko_points + Chandra_points = 54 := by
  sorry

end total_points_scored_l376_376442


namespace book_pairs_count_l376_376797

theorem book_pairs_count :
  let mystery_books := 4
  let science_fiction_books := 4
  let historical_books := 4
  (mystery_books + science_fiction_books + historical_books) = 12 ∧ 
  (mystery_books = 4 ∧ science_fiction_books = 4 ∧ historical_books = 4) →
  let genres := 3
  ∃ pairs, pairs = 48 :=
by
  sorry

end book_pairs_count_l376_376797


namespace team_total_points_l376_376443

-- Definitions based on conditions
def chandra_points (akiko_points : ℕ) := 2 * akiko_points
def akiko_points (michiko_points : ℕ) := michiko_points + 4
def michiko_points (bailey_points : ℕ) := bailey_points / 2
def bailey_points := 14

-- Total points scored by the team
def total_points :=
  let michiko := michiko_points bailey_points
  let akiko := akiko_points michiko
  let chandra := chandra_points akiko
  bailey_points + michiko + akiko + chandra

theorem team_total_points : total_points = 54 := by
  sorry

end team_total_points_l376_376443


namespace solve_equation_l376_376523

open Real

noncomputable def f (x : ℝ) := 2017 * x ^ 2017 - 2017 + x
noncomputable def g (x : ℝ) := (2018 - 2017 * x) ^ (1 / 2017 : ℝ)

theorem solve_equation :
  ∀ x : ℝ, 2017 * x ^ 2017 - 2017 + x = (2018 - 2017 * x) ^ (1 / 2017 : ℝ) → x = 1 :=
by
  sorry

end solve_equation_l376_376523


namespace jury_concludes_you_are_not_guilty_l376_376907

def criminal_is_a_liar : Prop := sorry -- The criminal is a liar, known.
def you_are_a_liar : Prop := sorry -- You are a liar, unknown.
def you_are_not_guilty : Prop := sorry -- You are not guilty.

theorem jury_concludes_you_are_not_guilty :
  criminal_is_a_liar → you_are_a_liar → you_are_not_guilty → "I am guilty" = "You are not guilty" :=
by
  -- Proof construct omitted as per problem requirements
  sorry

end jury_concludes_you_are_not_guilty_l376_376907


namespace f_increasing_solve_inequality_l376_376382

-- Given
variables {f : ℝ → ℝ}
hypothesis odd_f : ∀ x, f(-x) = -f(x)
hypothesis f_one : f(1) = 1
hypothesis pos_quot : ∀ a b : ℝ, a ∈ Icc (-1) 1 → b ∈ Icc (-1) 1 → a + b ≠ 0 → (f(a) + f(b)) / (a + b) > 0

-- To prove: Part (1)
theorem f_increasing : ∀ x1 x2 ∈ Icc (-1) 1, x1 < x2 → f(x1) < f(x2) :=
sorry

-- To prove: Part (2)
theorem solve_inequality : {x : ℝ // x ∈ Icc 0 (1/4) ∧ f(x + 1/2) < f(1 - x)} :=
sorry

end f_increasing_solve_inequality_l376_376382


namespace median_of_36_consecutive_integers_l376_376563

theorem median_of_36_consecutive_integers (x : ℤ) (sum_eq : (∑ i in finset.range 36, (x + i)) = 6^4) : (17 + 18) / 2 = 36 :=
by
  -- Proof goes here
  sorry

end median_of_36_consecutive_integers_l376_376563


namespace initial_speed_of_car_l376_376983

-- Definition of conditions
def distance_from_A_to_B := 100  -- km
def time_remaining_first_reduction := 30 / 60  -- hours
def speed_reduction_first := 10  -- km/h
def time_remaining_second_reduction := 20 / 60  -- hours
def speed_reduction_second := 10  -- km/h
def additional_time_reduced_speeds := 5 / 60  -- hours

-- Variables for initial speed and intermediate distances
variables (v x : ℝ)

-- Proposition to prove the initial speed
theorem initial_speed_of_car :
  (100 - (v / 2 + x + 20)) / v + 
  (v / 2) / (v - 10) + 
  20 / (v - 20) - 
  20 / (v - 10) 
  = 5 / 60 →
  v = 100 :=
by
  sorry

end initial_speed_of_car_l376_376983


namespace corrected_average_l376_376183

theorem corrected_average (incorrect_avg : ℕ) (correct_val incorrect_val number_of_values : ℕ) (avg := 17) (n := 10) (inc := 26) (cor := 56) :
  incorrect_avg = 17 →
  number_of_values = 10 →
  correct_val = 56 →
  incorrect_val = 26 →
  correct_avg = (incorrect_avg * number_of_values + (correct_val - incorrect_val)) / number_of_values →
  correct_avg = 20 := by
  sorry

end corrected_average_l376_376183


namespace sin_405_eq_sqrt2_div_2_l376_376321

theorem sin_405_eq_sqrt2_div_2 :
  Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_eq_sqrt2_div_2_l376_376321


namespace pi_irrational_l376_376961

theorem pi_irrational :
  ∀ (rational : ℚ → Prop) (irrational : ℝ → Prop),
    (∀ x : ℚ, rational x) →
    (rational 0) →
    (rational (22 / 7)) →
    (irrational π) →
    (rational (∛8)) →
    irrational π :=
by
  intros rational irrational h_rational h0 h_rat_22_7 h_irrat_pi h_rat_cbrt8
  exact h_irrat_pi

end pi_irrational_l376_376961


namespace z_conj_in_fourth_quadrant_l376_376034

-- Given a complex number z = 3 + i
def z : ℂ := 3 + complex.i

-- Define the conjugate of z
def z_conj : ℂ := complex.conj z

-- Prove that the point corresponding to z_conj lies in the fourth quadrant
theorem z_conj_in_fourth_quadrant : 
  (z_conj.re > 0) ∧ (z_conj.im < 0) :=
  by
    -- Proof will be developed here
    sorry

end z_conj_in_fourth_quadrant_l376_376034


namespace fraction_of_income_from_tips_l376_376998

variable (S T I : ℝ)
variable (h : T = (5 / 4) * S)

theorem fraction_of_income_from_tips (h : T = (5 / 4) * S) (I : ℝ) (w : I = S + T) : (T / I) = 5 / 9 :=
by
  -- The proof goes here
  sorry

end fraction_of_income_from_tips_l376_376998


namespace shaded_area_l376_376228

theorem shaded_area 
  (side_length : ℝ) -- The square's side length
  (n : ℕ) -- Number of circles per row (and column)
  (h1 : side_length = 20) -- Given square's side length is 20 inches
  (h2 : n = 3) -- Given 3 circles per row and column
  (tangent_to_sides : ∀ (i j : ℕ), i < n → j < n → True) -- Each circle is tangent to sides and each other
: 
  let radius := side_length / (2 * n),
      area_square := side_length ^ 2,
      area_circle := π * radius ^ 2,
      total_circle_area := n^2 * area_circle,
      shaded_area := area_square - total_circle_area 
  in shaded_area = 400 - 100 * π := 
by
  have radius_eq : radius = 10/3 := by
    rw [h1, h2],
    calc 20 / (2 * 3) = 20 / 6 : by norm_num
                ... = 10 / 3 : by norm_num,
  have area_square_eq : area_square = 400 := by
    rw [h1],
    calc 20 ^ 2 = 400 : by norm_num,
  have area_circle_eq : area_circle = 100 * π / 9 := by
    rw [radius_eq],
    calc π * (10 / 3) ^ 2 = π * (100 / 9) : by norm_num (10 / 3) ^ 2
                      ... = 100 * π / 9 : by ring,
  have total_circle_area_eq : total_circle_area = 100 * π := by
    rw [h2, area_circle_eq],
    calc 9 * (100 * π / 9) = 100 * π : by ring,
  have shaded_area_eq : shaded_area = 400 - 100 * π := by
    rw [total_circle_area_eq, area_square_eq],
    calc 400 - 100 * π = 400 - 100 * π : by ring,
  exact shaded_area_eq

end shaded_area_l376_376228


namespace median_of_36_consecutive_integers_l376_376576

theorem median_of_36_consecutive_integers (sum_of_integers : ℕ) (num_of_integers : ℕ) 
  (h1 : num_of_integers = 36) (h2 : sum_of_integers = 6 ^ 4) : 
  (sum_of_integers / num_of_integers) = 36 := 
by 
  sorry

end median_of_36_consecutive_integers_l376_376576


namespace right_triangle_square_area_and_triangle_area_l376_376221

theorem right_triangle_square_area_and_triangle_area
  {AB BC CD AD AC : ℝ} :
  -- Right triangles
  (AB^2 + BC^2 = AC^2) ∧
  (AC^2 + CD^2 = AD^2) ∧
  -- Given square areas
  (AB^2 = 25) ∧
  (BC^2 = 36) ∧
  (CD^2 = 49) →
  -- Prove the area of the fourth square and the area of triangle ACD
  (AD^2 = 110) ∧
  (∃ AC CD : ℝ, AC = real.sqrt 61 ∧ CD = 7 ∧ (1/2) * AC * CD = (7 * real.sqrt 61) / 2) :=
by
  intros h
  sorry

end right_triangle_square_area_and_triangle_area_l376_376221


namespace b_in_terms_of_a_l376_376132

noncomputable def a (k : ℝ) : ℝ := 3 + 3^k
noncomputable def b (k : ℝ) : ℝ := 3 + 3^(-k)

theorem b_in_terms_of_a (k : ℝ) :
  b k = (3 * (a k) - 8) / ((a k) - 3) := 
sorry

end b_in_terms_of_a_l376_376132


namespace mul_lt_one_l376_376631

theorem mul_lt_one (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |log a| > |log b|) : a * b < 1 :=
sorry

end mul_lt_one_l376_376631


namespace find_legs_of_right_triangle_l376_376185

theorem find_legs_of_right_triangle 
  (O A B C : Point) 
  (d1 d2 : ℝ) 
  (h_right : right_triangle A B C)
  (h_incenter : is_incenter O A B C)
  (h1 : dist O A = d1) 
  (h2 : dist O B = d2) 
  (h_d1 : d1 = real.sqrt 5) 
  (h_d2 : d2 = real.sqrt 10) : 
  dist A C = 3 ∧ dist B C = 4 := 
sorry

end find_legs_of_right_triangle_l376_376185


namespace interval_of_decrease_l376_376330

noncomputable def f : ℝ → ℝ := fun x => x^2 - 2 * x

theorem interval_of_decrease : 
  ∃ a b : ℝ, a = -2 ∧ b = 1 ∧ ∀ x1 x2 : ℝ, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 ≥ f x2 :=
by 
  use -2, 1
  sorry

end interval_of_decrease_l376_376330


namespace total_collisions_l376_376974

def Ball := ℕ

constants (mass speed : ℝ) (wall_pos : ℝ) (balls_red balls_blue : List Ball)
constants (initial_position_red initial_position_blue : Ball → ℝ)

-- Initial condition constraints
axiom balls_red_count : balls_red.length = 20
axiom balls_blue_count : balls_blue.length = 16
axiom balls_equal_mass : ∀ (b1 b2 : Ball), b1 ≠ b2 → mass = mass
axiom balls_equal_speed : ∀ (b1 b2 : Ball), b1 ≠ b2 → speed = speed
axiom balls_scatter_same_speed : ∀ (b1 b2 : Ball), (b1 ∈ balls_red ∨ b1 ∈ balls_blue) → (b2 ∈ balls_red ∨ b2 ∈ balls_blue) → speed = speed
axiom balls_bounce_same_speed : ∀ (b1 : Ball), (b1 ∈ balls_red ∨ b1 ∈ balls_blue) → speed = speed
axiom balls_move_trough : ∀ (b1 : Ball), (b1 ∈ balls_red ∨ b1 ∈ balls_blue) → initial_position_red b1 ≠ wall_pos ∧ initial_position_blue b1 ≠ wall_pos

theorem total_collisions : 20 * 16 + (20 * (20 - 1)) / 2 = 510 := by
  sorry

end total_collisions_l376_376974


namespace steps_in_staircase_l376_376505

theorem steps_in_staircase (h1 : 120 / 20 = 6) (h2 : 180 / 6 = 30) : 
  ∃ n : ℕ, n = 30 :=
by
  -- the proof is omitted
  sorry

end steps_in_staircase_l376_376505


namespace median_of_36_consecutive_integers_l376_376580

theorem median_of_36_consecutive_integers (f : ℕ → ℤ) (h_consecutive : ∀ n : ℕ, f (n + 1) = f n + 1) 
(h_size : ∃ k, f 36 = f 0 + 35) (h_sum : ∑ i in finset.range 36, f i = 6^4) : 
(∃ m, m = f (36 / 2 - 1) ∧ m = 36) :=
by
  sorry

end median_of_36_consecutive_integers_l376_376580


namespace price_per_glass_first_day_l376_376882

theorem price_per_glass_first_day
    (O W : ℝ) (P1 P2 : ℝ)
    (h1 : O = W)
    (h2 : P2 = 0.40)
    (h3 : 2 * O * P1 = 3 * O * P2) :
    P1 = 0.60 :=
by
    sorry

end price_per_glass_first_day_l376_376882


namespace consecutive_probability_is_correct_l376_376596

def cards : Finset ℕ := {1, 2, 3, 4, 5}

noncomputable def probability_consecutive (c : Finset ℕ) : ℚ :=
let pairs := c.powerset.filter (λ s, s.card = 2) in
let consecutive (s : Finset ℕ) : Prop := ∃ a b, s = {a, b} ∧ (a + 1 = b ∨ b + 1 = a) in
let consecutive_pairs := pairs.filter consecutive in
(consecutive_pairs.card : ℚ) / (pairs.card : ℚ)

theorem consecutive_probability_is_correct :
  probability_consecutive cards = 0.4 :=
by
  sorry

end consecutive_probability_is_correct_l376_376596


namespace correct_factorization_l376_376240

theorem correct_factorization :
  (x^2 - 2 * x + 1 = (x - 1)^2) ∧ 
  (¬ (x^2 - 4 * y^2 = (x + y) * (x - 4 * y))) ∧ 
  (¬ ((x + 4) * (x - 4) = x^2 - 16)) ∧ 
  (¬ (x^2 - 8 * x + 9 = (x - 4)^2 - 7)) :=
by
  sorry

end correct_factorization_l376_376240


namespace total_tower_surface_area_is_correct_l376_376175

-- Define the volumes of the cubes
def volumes : List ℕ := [1, 8, 27, 64, 125, 216]

-- Function to calculate the side length of a cube given its volume
def side_length (v : ℕ) : ℕ := v^(1 / 3).to_nat

-- Function to calculate the surface area of a cube given its side length
def surface_area (s : ℕ) : ℕ := 6 * s * s

-- Function to calculate the total surface area of the stacked cubes
def total_surface_area (volumes : List ℕ) : ℕ :=
  let areas := volumes.map (λ v => surface_area (side_length v))
  -- Adjust for overlapping faces, assuming perfect overlap from second cube onwards
  let adjusted_areas := List.zipWith (λ a s => a - s^2) areas (List.tail! volumes.map (side_length))
  List.sum adjusted_areas + volumes.head!.surface_area

-- The proof statement
theorem total_tower_surface_area_is_correct : total_surface_area volumes = 456 :=
by
  sorry

end total_tower_surface_area_is_correct_l376_376175


namespace cannot_ensure_daily_duty_l376_376436

theorem cannot_ensure_daily_duty
  (guards : ℕ → ℕ)
  (H₁ : ∀ i, guards (i + 1) ≤ guards i / 3) :
  ¬ ∃ (duty_schedule : ℕ → ℕ → bool),
    (∀ n, ∃ i, duty_schedule i n = tt) ∧
    (∀ i n, duty_schedule i n = tt ↔ n % (2 * guards i) < guards i) := 
sorry

end cannot_ensure_daily_duty_l376_376436


namespace ratio_of_wire_pieces_l376_376288

theorem ratio_of_wire_pieces (a b : ℝ) (h_equal_areas : (a / 4) ^ 2 = 2 * (1 + real.sqrt 2) * (b / 8) ^ 2) :
  a / b = real.sqrt (2 + real.sqrt 2) / 2 := 
by
  sorry

end ratio_of_wire_pieces_l376_376288


namespace polygons_with_A1_more_than_without_A1_l376_376753

theorem polygons_with_A1_more_than_without_A1 (A : Finset Point) (hA : A.card = 16) :
  let polygons_with_A1 := ∑ k in Finset.range (15 - 1 + 1), (A.erase A1).card.choose k
  let polygons_without_A1 := ∑ k in Finset.range (15 - 2 + 1), (A.erase A1).card.choose k
  polygons_with_A1 > polygons_without_A1 :=
by
  let polygons_with_A1 := ∑ k in Finset.range (15 - 1 + 1), (15).choose k
  let polygons_without_A1 := ∑ k in Finset.range (15 - 2 + 1), (15).choose k
  sorry

end polygons_with_A1_more_than_without_A1_l376_376753


namespace OD_expression_l376_376354

noncomputable def OD_calc (t1 : ℝ) : ℝ :=
  2 * t1 / (t1 + 1)

theorem OD_expression (θ : ℝ) (t1 : ℝ) (t2 : ℝ)
  (h1 : t1 = Real.sin (2 * θ))
  (h2 : t2 = Real.cos (2 * θ)) :
  let r := 2 in
  let O := (0, 0) in
  let A := (r * t2, r * t1) in
  let B := (r * t2, - r * t1) in -- Assuming B is along the line where AB is tangent to the circle
  OD_calc t1 = 2 * t1 / (t1 + 1) :=
by 
  sorry

end OD_expression_l376_376354


namespace hyperbola_equation_l376_376010

open Real

-- Definitions based on conditions
def is_hyperbola_centered_at_origin (h : ℝ × ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), h (x, y) → h (-x, y) ∧ h (x, -y)

def has_asymptote (h : ℝ × ℝ → Prop) (line : ℝ × ℝ → Prop) : Prop :=
  ∃ (A B : ℝ), ∀ (x y : ℝ), line (x, y) ∧ abs (h(x, y) / x) = A ∧ abs (x / y) = B

def passes_through (h : ℝ × ℝ → Prop) (p : ℝ × ℝ) : Prop :=
  h p

-- Specific functions as per given problem
def hyperbola (p : ℝ × ℝ) := (p.snd ^ 2 / 4) - (p.fst ^ 2) - 1

def asymptote_line (p : ℝ × ℝ) := p.fst - 2 * p.snd

-- Stating the theorem
theorem hyperbola_equation :
  ∃ h : ℝ × ℝ → Prop,
    is_hyperbola_centered_at_origin h ∧
    has_asymptote h asymptote_line ∧
    passes_through h (√ (5 / 2), 3) ∧
    (∀ x y, h (x, y) ↔ (y^2 / 4 - x^2 = 1)) :=
by {
  sorry
}

end hyperbola_equation_l376_376010


namespace value_of_a_l376_376048

-- Definitions based on conditions
def A (a : ℝ) : Set ℝ := {1, 2, a}
def B : Set ℝ := {1, 7}

-- Theorem statement
theorem value_of_a (a : ℝ) (h : B ⊆ A a) : a = 7 :=
sorry

end value_of_a_l376_376048


namespace expression_evaluation_l376_376629

variable (a b : ℝ)

theorem expression_evaluation (h : a + b = 1) :
  a^3 + b^3 + 3 * (a^3 * b + a * b^3) + 6 * (a^3 * b^2 + a^2 * b^3) = 1 :=
by
  sorry

end expression_evaluation_l376_376629


namespace equal_angles_PAB_QAC_l376_376482

-- Definitions of points and lines based on given conditions
variables (A B C M N P Q : Type)
variables (TriangleABC : Triangle A B C)
variables (LineBC : LineSegment B C)
variables (LineAM : LineSegment A M)
variables (LineAN : LineSegment A N)
variables (LineBN : Line B N)
variables (LineCM : Line C M)
variables (CircumcircleBMP : CircleThrough B M P)
variables (CircumcircleCNP : CircleThrough C N P)

-- Conditions
variables (MN_parallel_BC : Parallel (LineParallel M N) (LineSegment B C))
variables (IntersectsLineBN_CM_in_P : IntersectsAt LineBN LineCM P)
variables (CircumcirclesIntersect : Intersects (CircumcircleBMP) (CircumcircleCNP) Q)

-- Theorem statement to prove
theorem equal_angles_PAB_QAC : ∀ (A B C M N P Q TriangleABC LineBC LineAM LineAN LineBN LineCM CircumcircleBMP CircumcircleCNP), 
  (MN_parallel_BC) → (IntersectsLineBN_CM_in_P) → (CircumcirclesIntersect) → 
  (∠P A B = ∠Q A C) :=
by 
  intros A B C M N P Q TriangleABC LineBC LineAM LineAN LineBN LineCM CircumcircleBMP CircumcircleCNP MN_parallel_BC IntersectsLineBN_CM_in_P CircumcirclesIntersect,
  -- Proof will be provided here
  sorry

end equal_angles_PAB_QAC_l376_376482


namespace general_formula_for_a_sum_of_first_n_terms_of_b_l376_376378

noncomputable def a (n : ℕ) : ℤ := 2 * n - 1
noncomputable def b (n : ℕ) : ℤ := 2^(n-1) * (a 5)

theorem general_formula_for_a :
  (a 2 = 3) ∧ (a 4 + a 6 = 18) → (∀ n : ℕ, n > 0 → a n = 2 * n - 1) :=
by 
  intros h n hn;
  sorry

theorem sum_of_first_n_terms_of_b (n : ℕ) :
  let b1 := a 5 in 
  (b (n + 1) = 2 * b n) ∧ (b 0 = b1) → (∀ n : ℕ, n ≠ 0 → ∑ i in finset.range n, b i = 9 * (2^n - 1)) :=
by 
  intros h n hn;
  sorry

end general_formula_for_a_sum_of_first_n_terms_of_b_l376_376378


namespace hyperbola_focal_length_l376_376396

theorem hyperbola_focal_length (x y : ℝ) : 
  (∃ h : x^2 / 9 - y^2 / 4 = 1, 
   ∀ a b : ℝ, a^2 = 9 → b^2 = 4 → 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 13) :=
by sorry

end hyperbola_focal_length_l376_376396


namespace speed_of_goods_train_is_72_kmph_l376_376643

-- Definitions for conditions
def length_of_train : ℝ := 240.0416
def length_of_platform : ℝ := 280
def time_to_cross : ℝ := 26

-- Distance covered by the train while crossing the platform
def total_distance : ℝ := length_of_train + length_of_platform

-- Speed calculation in meters per second
def speed_mps : ℝ := total_distance / time_to_cross

-- Speed conversion from meters per second to kilometers per hour
def speed_kmph : ℝ := speed_mps * 3.6

-- Proof statement
theorem speed_of_goods_train_is_72_kmph : speed_kmph = 72 := 
by
  sorry

end speed_of_goods_train_is_72_kmph_l376_376643


namespace marks_in_social_studies_l376_376519

def shekar_marks : ℕ := 82

theorem marks_in_social_studies 
  (marks_math : ℕ := 76)
  (marks_science : ℕ := 65)
  (marks_english : ℕ := 67)
  (marks_biology : ℕ := 55)
  (average_marks : ℕ := 69)
  (num_subjects : ℕ := 5) :
  marks_math + marks_science + marks_english + marks_biology + shekar_marks = average_marks * num_subjects :=
by
  sorry

end marks_in_social_studies_l376_376519


namespace modulus_of_z_l376_376759

variable (a : ℝ)
variable (z : ℂ)

-- Define the properties assumed in the problem
def pure_imaginary (w : ℂ) : Prop := w.re = 0

-- Assumptions in the problem
axiom h1 : ∀ a : ℝ, pure_imaginary ( (2 - complex.I) / (a + complex.I) )
axiom h2 : z = 4 * a + complex.sqrt 2 * complex.I

-- Statement of what needs to be proved
theorem modulus_of_z : ∀ a : ℝ, (pure_imaginary ((2 - complex.I) / (a + complex.I)) → complex.abs (4 * a + complex.sqrt 2 * complex.I) = complex.sqrt 6) :=
by
  intros a ha
  sorry

end modulus_of_z_l376_376759


namespace correct_proposition_l376_376379

variable {R : Type*} [real_field R]

def f (x : R) : R

def even_function (f : R → R) :=
∀ x : R, f (-x) = f (x)

def periodicity (f : R → R) (a b c : R) :=
∀ x : R, f (x + a) = f (x) + b

def monotonic_decreasing (f : R → R) (a b : R) :=
∀ x y : R, (a ≤ x ∧ x ≤ y ∧ y ≤ b) → f y ≤ f x

noncomputable def proposition_4 (f : R → R) :=
∀ m : R, ∀ x1 x2 : R, (x1 ≤ -2 ∧ x1 ≥ -6 ∧ x2 ≤ -2 ∧ x2 ≥ -6) ∧ (f x1 = m ∧ f x2 = m) → x1 + x2 = -8

theorem correct_proposition :
  even_function f →
  periodicity f 4 (f 2) →
  monotonic_decreasing f 0 2 →
  proposition_4 f :=
by
  intros h_even h_periodic h_mono x m x1 x2 h x_eq
  sorry

end correct_proposition_l376_376379


namespace orchard_yield_correct_l376_376098

-- Definitions for conditions
def gala3YrTreesYield : ℕ := 10 * 120
def gala2YrTreesYield : ℕ := 10 * 150
def galaTotalYield : ℕ := gala3YrTreesYield + gala2YrTreesYield

def fuji4YrTreesYield : ℕ := 5 * 180
def fuji5YrTreesYield : ℕ := 5 * 200
def fujiTotalYield : ℕ := fuji4YrTreesYield + fuji5YrTreesYield

def redhaven6YrTreesYield : ℕ := 15 * 50
def redhaven4YrTreesYield : ℕ := 15 * 60
def redhavenTotalYield : ℕ := redhaven6YrTreesYield + redhaven4YrTreesYield

def elberta2YrTreesYield : ℕ := 5 * 70
def elberta3YrTreesYield : ℕ := 5 * 75
def elberta5YrTreesYield : ℕ := 5 * 80
def elbertaTotalYield : ℕ := elberta2YrTreesYield + elberta3YrTreesYield + elberta5YrTreesYield

def appleTotalYield : ℕ := galaTotalYield + fujiTotalYield
def peachTotalYield : ℕ := redhavenTotalYield + elbertaTotalYield
def orchardTotalYield : ℕ := appleTotalYield + peachTotalYield

-- Theorem to prove
theorem orchard_yield_correct : orchardTotalYield = 7375 := 
by sorry

end orchard_yield_correct_l376_376098


namespace solve_log_equation_l376_376522

theorem solve_log_equation :
  ∃ x : ℝ, log x - 3 * log 4 = -3 ∧ x = 0.064 :=
begin
  sorry
end

end solve_log_equation_l376_376522


namespace length_of_AB_l376_376383

open Real

noncomputable def parabola : Set (ℝ × ℝ) := { p | p.2 ^ 2 = 4 * p.1 }

variable (A : ℝ × ℝ) (F : ℝ × ℝ)
variable (condition_A : A = (4, 4))
variable (condition_F : F = (1, 0))

theorem length_of_AB (hA : A ∈ parabola) :
  ∃ B : ℝ × ℝ, B ≠ A ∧ B ∈ parabola ∧ 
  let l := line_through A F in
  B ∈ l ∧ 
  dist A B = 25 / 4 :=
begin
  sorry
end

def line_through (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  { p | ∃ m b, p.2 = m * p.1 + b ∧ ∀ q ∈ {A, B}, q.2 = m * q.1 + b }

def dist (p1 p2 : ℝ × ℝ) : ℝ := 
  sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

end length_of_AB_l376_376383


namespace max_m_value_l376_376095

theorem max_m_value (m : ℝ) (A B : ℝ × ℝ)
  (hAB_on_line : A.1 + A.2 = m ∧ B.1 + B.2 = m)
  (hAB_distance : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 100) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi →
    let P := (Real.cos θ, Real.sin θ) in
    (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0) →
  m = 4 * Real.sqrt 2 :=
begin
  sorry
end

end max_m_value_l376_376095


namespace complement_A_inter_B_l376_376789

def U : Set ℕ := {x | 0 ≤ x ∧ x ≤ 4}
def A : Set ℕ := {-1, 2, 3}
def B : Set ℕ := {2, 3}
def A_inter_B : Set ℕ := A ∩ B
def complement_U (S : Set ℕ) : Set ℕ := {x ∈ U | x ∉ S}

theorem complement_A_inter_B :
  complement_U A_inter_B = {0, 1, 4} :=
by {
  -- proof is omitted
  sorry
}

end complement_A_inter_B_l376_376789


namespace jessica_flour_count_l376_376110

def flour_initially_put_in (total_flour required_flour additional_flour : ℕ) : ℕ :=
  required_flour - additional_flour

theorem jessica_flour_count :
  flour_initially_put_in 8 8 4 = 4 :=
by
  simp [flour_initially_put_in]
  sorry

end jessica_flour_count_l376_376110


namespace trailing_zeros_500_factorial_l376_376683

theorem trailing_zeros_500_factorial : 
  let count_multiples (n k : ℕ) := n / k
  let factors_of_5 := count_multiples 500 5
  let factors_of_25 := count_multiples 500 25
  let factors_of_125 := count_multiples 500 125
  let factors_of_625 := count_multiples 500 625
  factors_of_5 + factors_of_25 + factors_of_125 + factors_of_625 = 124 := 
by
  let count_multiples (n k : ℕ) := n / k
  let factors_of_5 := count_multiples 500 5
  let factors_of_25 := count_multiples 500 25
  let factors_of_125 := count_multiples 500 125
  let factors_of_625 := count_multiples 500 625
  have h1 : factors_of_5 = 100 := rfl
  have h2 : factors_of_25 = 20 := rfl
  have h3 : factors_of_125 = 4 := rfl
  have h4 : factors_of_625 = 0 := rfl
  show factors_of_5 + factors_of_25 + factors_of_125 + factors_of_625 = 124
  from by {
    rw [h1, h2, h3, h4],
    rfl,
  }
  sorry

end trailing_zeros_500_factorial_l376_376683


namespace average_of_values_l376_376344

theorem average_of_values (z : ℝ) : 
  (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end average_of_values_l376_376344


namespace intervals_of_monotonicity_relationship_a_b_l376_376779

section PartI

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + c

theorem intervals_of_monotonicity :
  ∀ c : ℝ, ∃ I1 I2 I3 : set ℝ,
    I1 = {x | x ∈ set.Iic (-1)} ∧
    I2 = {x | x ∈ set.Ioc (-1) 3} ∧
    I3 = {x | x ∈ set.Ioi (3)} ∧
    ∀ x, (x ∈ I1 → f' x > 0) ∧ (x ∈ I2 → f' x < 0) ∧ (x ∈ I3 → f' x > 0) :=
begin
  sorry
end

end PartI

section PartII

def g (x a b c : ℝ) : ℝ := x^3 - a * x^2 + b * x + c

theorem relationship_a_b (a b c : ℝ) :
  (∃ x, g' x a b c = 0) → (a^2 ≥ 3 * b) :=
begin
  sorry
end

end PartII

end intervals_of_monotonicity_relationship_a_b_l376_376779


namespace village_population_l376_376264

theorem village_population (P : ℝ) (h : 0.9 * P = 45000) : P = 50000 :=
by
  sorry

end village_population_l376_376264


namespace amount_after_two_years_l376_376334

theorem amount_after_two_years (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ)
  (hP : P = 64000) (hr : r = 1 / 6) (hn : n = 2) : 
  A = P * (1 + r) ^ n := by
  sorry

end amount_after_two_years_l376_376334


namespace least_value_expression_l376_376951

-- Definition of the expression
def expression (x y : ℝ) := (x * y - 2) ^ 2 + (x - 1 + y) ^ 2

-- Statement to prove the least possible value of the expression
theorem least_value_expression : ∃ x y : ℝ, expression x y = 2 := 
sorry

end least_value_expression_l376_376951


namespace balls_in_boxes_l376_376937

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 3
  let ways := ∑ (g : Multiset (Fin 5)), (g.card = 3) → (∏ x in g, ∑ y in (Multiset.erase g x), y < boxes)
  ways = 150 :=
by sorry

end balls_in_boxes_l376_376937


namespace find_b_if_continuous_l376_376484

def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 3 then 3 * x ^ 2 - 5 else b * x + 7

theorem find_b_if_continuous : 
(∀ x : ℝ, continuous (f x)) → (∃ b : ℝ, f 3 b = f 3 b) := by
  sorry

end find_b_if_continuous_l376_376484


namespace real_value_of_m_pure_imaginary_value_of_m_l376_376740

open Complex

-- Given condition
def z (m : ℝ) : ℂ := (m^2 - m : ℂ) - (m^2 - 1 : ℂ) * I

-- Part (I)
theorem real_value_of_m (m : ℝ) (h : im (z m) = 0) : m = 1 ∨ m = -1 := by
  sorry

-- Part (II)
theorem pure_imaginary_value_of_m (m : ℝ) (h1 : re (z m) = 0) (h2 : im (z m) ≠ 0) : m = 0 := by
  sorry

end real_value_of_m_pure_imaginary_value_of_m_l376_376740


namespace sin_405_eq_sqrt_2_div_2_l376_376316

theorem sin_405_eq_sqrt_2_div_2 : sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_eq_sqrt_2_div_2_l376_376316


namespace distinct_tangent_line_counts_l376_376154

theorem distinct_tangent_line_counts (r1 r2 : ℝ) (h1 : r1 = 5) (h2 : r2 = 7) :
  ∃ k : ℕ, k = 4 :=
by
  use 4
  sorry

end distinct_tangent_line_counts_l376_376154


namespace pizza_pepperoni_fraction_l376_376331

theorem pizza_pepperoni_fraction :
  (∀ {r : ℝ}, 8 * r = 6 → ∃ f : ℝ, f = 32 * (π * r^2) / (π * 6^2) ∧ f = 1 / 2) :=
by
  intro r hr
  use 32 * (π * r^2) / (π * 6^2)
  split 
  { sorry },
  { sorry }

end pizza_pepperoni_fraction_l376_376331


namespace num_possible_sets_l376_376197

def condition (B : Set ℕ) : Prop :=
  {3, 5} ∪ B = {3, 5, 7}

theorem num_possible_sets : {B : Set ℕ // condition B}.card = 4 :=
  sorry

end num_possible_sets_l376_376197


namespace number_of_divisors_of_8m3_l376_376858

variables {m : ℕ}
variables (hm₁ : odd m) (hm₂ : ∃ q : ℕ, prime q ∧ m = q ^ 6)

theorem number_of_divisors_of_8m3 (hm₁ : odd m) (hm₂ : ∃ q : ℕ, prime q ∧ m = q ^ 6) : 
  nat.divisors_count (8 * m ^ 3) = 76 :=
sorry

end number_of_divisors_of_8m3_l376_376858


namespace find_a100_l376_376452

noncomputable def sequence : ℕ → ℕ
| 0     := 1
| (n+1) := sequence n + n

theorem find_a100 : sequence 100 = 4951 :=
by
  sorry

end find_a100_l376_376452


namespace additional_cards_added_l376_376304

theorem additional_cards_added :
  ∀ (initial total_per_player players final : ℕ),
    initial = 52 →
    total_per_player = 18 →
    players = 3 →
    final = total_per_player * players →
    final - initial = 2 :=
by
  intros initial total_per_player players final h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
sorry

end additional_cards_added_l376_376304


namespace perpendicular_lines_1_and_4_l376_376776

-- Define the equations
def equation1 (x y : ℝ) : Prop := 4 * y - 3 * x = 15
def equation2 (x y : ℝ) : Prop := -3 * x - 4 * y = 12
def equation3 (x y : ℝ) : Prop := 4 * y + 3 * x = 15
def equation4 (x y : ℝ) : Prop := 3 * y + 4 * x = 12

-- Define the slopes from the equations
def slope1 : ℝ := 3 / 4
def slope2 : ℝ := -3 / 4
def slope3 : ℝ := -3 / 4
def slope4 : ℝ := -4 / 3

-- Define the predicate for perpendicular slopes
def perpendicular_slopes (m1 m2 : ℝ) : Prop := (m1 * m2 = -1)

-- Prove that (1) and (4) have perpendicular slopes
theorem perpendicular_lines_1_and_4 : perpendicular_slopes slope1 slope4 :=
by {
  dsimp [perpendicular_slopes, slope1, slope4],
  sorry
}

end perpendicular_lines_1_and_4_l376_376776


namespace circumcenter_condition_l376_376469

-- Definitions for conditions
def is_equilateral (P1 P2 P3 : Point) : Prop := -- definition of an equilateral triangle
sorry

def is_circumcenter (P : Point) (Δ : Triangle) : Prop := -- definition of circumcenter of a triangle
sorry

def is_orthocenter (P : Point) (Δ : Triangle) : Prop := -- definition of orthocenter of a triangle
sorry

-- Main statement
theorem circumcenter_condition (P1 P2 P3 : Point) (Pn : ℕ → Point) 
    (h_eq : is_equilateral P1 P2 P3)
    (h_Pn : ∀ n ≥ 4, Pn n = if (n % 4) = 0 then P1 else P2) 
    : ∀ n ≥ 4, (is_circumcenter (Pn n) (Triangle.mk P1 P2 P3)) ↔ (n % 4 = 0) :=
sorry

end circumcenter_condition_l376_376469


namespace num_true_subsets_of_my_set_l376_376559

-- Define Universal Set
def my_set : Set Int := {-1, 0, 1}

-- Define the number of elements in the set
def num_elements (s : Set α) : Nat := s.toFinset.card

-- State the theorem
theorem num_true_subsets_of_my_set : num_elements my_set = 3 → (2^3) - 1 = 7 := by
  intros h
  rw ← h
  simp
  sorry

end num_true_subsets_of_my_set_l376_376559


namespace incircle_equal_segments_l376_376964

theorem incircle_equal_segments 
  (A B C D O : Point) 
  (hO : is_incenter O A B C)
  (hD : on_circumcircle D A B C)
  (hD_ne : D ≠ A)
  (hD_lies : lies_on_line D A O) :
  dist B D = dist C D ∧ dist D O = dist D C :=
sorry

end incircle_equal_segments_l376_376964


namespace count_multiples_of_12_l376_376411

theorem count_multiples_of_12 (low high : ℕ) (n : ℕ) (h₁ : low = 30) (h₂ : high = 200) (h₃ : n = 12) :
  (∑ k in finset.Icc (low / n + 1) (high / n), 1) = 14 :=
by
  sorry

end count_multiples_of_12_l376_376411


namespace num_ways_to_stand_l376_376205

-- Define the number of people
def num_people : ℕ := 7

-- Define the key condition: exactly 2 people between person A and person B
def condition (a b : ℕ) (l : List ℕ) : Prop :=
  ∃ (a_idx b_idx : ℕ), a_idx < b_idx ∧ (b_idx - a_idx = 3) ∧ l.get? a_idx = some a ∧ l.get? b_idx = some b

-- Define the main theorem: calculating the number of ways to arrange people 
theorem num_ways_to_stand (A B : ℕ) :
  (A ≠ B) →
  num_people = 7 →
  (∃ n, num_ways_to_stand_f A B n) →
  n = 960
:= by
  sorry

end num_ways_to_stand_l376_376205


namespace breadth_of_hall_l376_376648

theorem breadth_of_hall (length_hall : ℝ) (stone_length_dm stone_breadth_dm : ℝ) (num_stones : ℕ) 
  (length_hall_eq : length_hall = 36) 
  (stone_length_eq : stone_length_dm = 6) 
  (stone_breadth_eq : stone_breadth_dm = 5) 
  (num_stones_eq : num_stones = 1800) : 
  ∃ (breadth_hall : ℝ), breadth_hall = 15 :=
by
  -- Given data
  let stone_area_dm2 := stone_length_dm * stone_breadth_dm
  let stone_area_m2 := stone_area_dm2 * 0.1^2
  let total_area := stone_area_m2 * num_stones.toFloat
  have length_hall : length_hall = 36 := length_hall_eq
  have total_area := stone_area_m2 * num_stones.toFloat
  let breadth_hall := total_area / length_hall
  use breadth_hall
  sorry

end breadth_of_hall_l376_376648


namespace smallest_value_l376_376804

theorem smallest_value (y : ℝ) (hy : 0 < y ∧ y < 1) :
  y^3 < y^2 ∧ y^3 < 3*y ∧ y^3 < (y)^(1/3:ℝ) ∧ y^3 < (1/y) :=
sorry

end smallest_value_l376_376804


namespace olivia_choc_chip_cookies_l376_376502

noncomputable def oatmealCookies : ℕ := Float.ceil (3.111111111 * 9.0)
noncomputable def totalCookies : ℕ := 41
noncomputable def chocChipCookies : ℕ := totalCookies - oatmealCookies

theorem olivia_choc_chip_cookies : chocChipCookies = 13 := by
  sorry

end olivia_choc_chip_cookies_l376_376502


namespace alternating_binomial_sum_l376_376736

theorem alternating_binomial_sum (i : ℂ) (h : i * i = -1) :
  (1 + i)^2010 =  1 + ∑ k in finset.range 2010, binomial 2010 k * (i^k) →
  (∑ k in finset.range 1005, (-1)^k * binomial 2010 (2*k + 1)) = 2^1005 := 
by {
  sorry
}

end alternating_binomial_sum_l376_376736


namespace minimum_S_n_at_7_l376_376488

noncomputable def a (n : ℕ) : ℝ := sorry -- arithmetic sequence definition

theorem minimum_S_n_at_7 
  (a₁ : a 1 < 0)
  (a₇_a₈_neg : a 7 * a 8 < 0)
  (Sn : ℕ → ℝ := λ n, ∑ i in range n, a i) :
  ∃ n, Sn n = Sn 7 ∧ ∀ m, Sn m ≥ Sn 7 :=
by sorry

end minimum_S_n_at_7_l376_376488


namespace books_sold_correct_l376_376111

-- Definitions of the conditions
def initial_books : ℕ := 33
def remaining_books : ℕ := 7
def books_sold : ℕ := initial_books - remaining_books

-- The statement to be proven (with proof omitted)
theorem books_sold_correct : books_sold = 26 := by
  -- Proof omitted
  sorry

end books_sold_correct_l376_376111


namespace m_value_if_Q_subset_P_l376_376052

noncomputable def P : Set ℝ := {x | x^2 = 1}
def Q (m : ℝ) : Set ℝ := {x | m * x = 1}
def m_values (m : ℝ) : Prop := Q m ⊆ P → m = 0 ∨ m = 1 ∨ m = -1

theorem m_value_if_Q_subset_P (m : ℝ) : m_values m :=
sorry

end m_value_if_Q_subset_P_l376_376052


namespace sum_of_adjacent_to_7_l376_376923

/-- Define the divisors of 245, excluding 1 -/
def divisors245 : Set ℕ := {5, 7, 35, 49, 245}

/-- Define the adjacency condition to ensure every pair of adjacent integers has a common factor greater than 1 -/
def adjacency_condition (a b : ℕ) : Prop := (a ≠ b) ∨ (Nat.gcd a b > 1)

/-- Prove the sum of the two integers adjacent to 7 in the given condition is 294. -/
theorem sum_of_adjacent_to_7 (d1 d2 : ℕ) (h1 : d1 ∈ divisors245) (h2 : d2 ∈ divisors245) 
    (adj1 : adjacency_condition 7 d1) (adj2 : adjacency_condition 7 d2) : 
    d1 + d2 = 294 := 
sorry

end sum_of_adjacent_to_7_l376_376923


namespace remainder_sum_remainders_mod_500_l376_376126

open Nat

/-- Define the set of remainders of 3^n mod 500 for nonnegative integers n -/
def remainders_mod_500 : Set ℕ := {r | ∃ n : ℕ, r = 3^n % 500}

/-- Define the sum of the elements in the set of remainders -/
def S : ℕ := remainders_mod_500.sum (λ x, x)

theorem remainder_sum_remainders_mod_500 (x : ℕ)
  (hx : S % 500 = x) :
  S % 500 = x := by
  sorry

end remainder_sum_remainders_mod_500_l376_376126


namespace jerry_less_study_time_l376_376646

theorem jerry_less_study_time 
  (d : ℕ → ℤ) 
  (days : Fin 6) 
  (differ_factors : ∀ i, d i = [15, -5, 25, 0, -15, 10].nth i.getD 0) 
  (tom_additional_study : ∀ i, d i > 0 → ∃ extra, extra = 20): 
  (∑ i in days, if d i > 0 then d i + 20 else d i) / 6 = 15 := 
by 
  sorry

end jerry_less_study_time_l376_376646


namespace geometric_monotonic_condition_l376_376769

-- Definition of a geometrically increasing sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Definition of a monotonically increasing sequence
def monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

-- The theorem statement
theorem geometric_monotonic_condition (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (a 1 < a 2 ∧ a 2 < a 3) ↔ monotonically_increasing a :=
sorry

end geometric_monotonic_condition_l376_376769


namespace z_in_fourth_quadrant_l376_376068

-- Define the complex number z
def z : ℂ := (2 + I) / (I^5)

-- Statement: Prove that z lies in the fourth quadrant
theorem z_in_fourth_quadrant : (z.re > 0) ∧ (z.im < 0) :=
sorry

end z_in_fourth_quadrant_l376_376068


namespace mean_of_points_scored_l376_376734

def mean (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

theorem mean_of_points_scored (lst : List ℕ)
  (h1 : lst = [81, 73, 83, 86, 73]) : 
  mean lst = 79.2 :=
by
  rw [h1, mean]
  sorry

end mean_of_points_scored_l376_376734


namespace train_lengths_are_65_meters_l376_376254

/-- Two trains of equal length running on parallel lines in the same direction with
the faster train at 49 km/hr and the slower train at 36 km/hr.
The faster train passes the slower train in 36 seconds.
Prove that the length of each train is 65 meters. -/
theorem train_lengths_are_65_meters :
  ∃ (L : ℝ), (L > 0) ∧ let relative_speed := (49 * 1000 / 3600) - (36 * 1000 / 3600) in
  let distance_covered := relative_speed * 36 in
  2 * L = distance_covered ∧ L ≈ 65 :=
sorry

end train_lengths_are_65_meters_l376_376254


namespace max_tickets_l376_376353

theorem max_tickets (ticket_price normal_discounted_price budget : ℕ) (h1 : ticket_price = 15) (h2 : normal_discounted_price = 13) (h3 : budget = 180) :
  ∃ n : ℕ, ((n ≤ 10 → ticket_price * n ≤ budget) ∧ (n > 10 → normal_discounted_price * n ≤ budget)) ∧ ∀ m : ℕ, ((m ≤ 10 → ticket_price * m ≤ budget) ∧ (m > 10 → normal_discounted_price * m ≤ budget)) → m ≤ 13 :=
by
  sorry

end max_tickets_l376_376353


namespace angle_is_pi_over_4_l376_376424

variables (a b : ℝ^3) (θ : ℝ)
def norm (v : ℝ^3) := real.sqrt (v.1^2 + v.2^2 + v.3^2)
noncomputable def dot_product (u v : ℝ^3) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
def perpendicular (u v : ℝ^3) := dot_product u v = 0
def angle_between_vectors (u v : ℝ^3) := real.acos (dot_product u v / (norm u * norm v))

axiom norm_a : norm a = 1
axiom norm_b : norm b = real.sqrt 2
axiom perp_condition : perpendicular (a - b) a

theorem angle_is_pi_over_4 : angle_between_vectors a b = real.pi / 4 :=
sorry

end angle_is_pi_over_4_l376_376424


namespace regression_pos_correlation_predict_savings_for_monthly_income_l376_376654

def monthly_income_sum := ∑ i in range(10), x_i
def monthly_savings_sum := ∑ i in range(10), y_i
def monthly_income_savings_product_sum := ∑ i in range(10), x_i * y_i
def monthly_income_squared_sum := ∑ i in range(10), x_i^2

def b_hat :=
  (monthly_income_savings_product_sum - (1 / 10) * monthly_income_sum * monthly_savings_sum) /
  (monthly_income_squared_sum - (1 / 10) * monthly_income_sum^2)

def a_hat :=
  (1 / 10) * monthly_savings_sum - b_hat * (1 / 10) * monthly_income_sum

def regression_equation (x : ℝ) :=
  b_hat * x + a_hat

theorem regression_pos_correlation : b_hat > 0 -> true :=
begin
  sorry
end

theorem predict_savings_for_monthly_income : regression_equation 7 = 1.7 :=
begin
  sorry
end

-- Given conditions
variables (x_i y_i : ℝ) (i : fin 10)
#check monthly_income_sum = 80
#check monthly_savings_sum = 20
#check monthly_income_savings_product_sum = 184
#check monthly_income_squared_sum = 720

end regression_pos_correlation_predict_savings_for_monthly_income_l376_376654


namespace distance_between_A_and_B_l376_376508

theorem distance_between_A_and_B
  (vA vB D : ℝ)
  (hvB : vB = (3/2) * vA)
  (second_meeting_distance : 20 = D * 2 / 5) : 
  D = 50 := 
by
  sorry

end distance_between_A_and_B_l376_376508


namespace prob_chocolate_milk_l376_376894

theorem prob_chocolate_milk (visits : ℕ) (p_chocolate : ℚ) (p_regular : ℚ) :
  visits = 7 ∧ p_chocolate = 3/4 ∧ p_regular = 1/4 →
  (∑ k in (finset.range visits.succ).filter (λ k, k = 5 ∧ visits - k = 2),
    (nat.choose visits k) * (p_chocolate^k) * (p_regular^(visits - k))) = 5103 / 16384 :=
by
  intro h,
  cases h with h1 rest,
  cases rest with h2 h3,
  simp [h1, h2, h3],
  sorry

end prob_chocolate_milk_l376_376894


namespace adjacent_squares_difference_l376_376561

theorem adjacent_squares_difference
  (n : ℕ)
  (chesboard : Fin (n * n) → Fin (n * n))
  (h_unique : Function.Injective chesboard) :
  ∃ (i j : Fin n) (i' j' : Fin n), (|chesboard ⟨i.val * n + j.val, sorry⟩ - chesboard ⟨i'.val * n + j'.val, sorry⟩| ≥ n) ∧ 
  ((|i.val - i'.val| = 1 ∧ j = j') ∨ (|j.val - j'.val| = 1 ∧ i = i')) :=
begin
  sorry
end

end adjacent_squares_difference_l376_376561


namespace tom_helicopter_hours_l376_376213

theorem tom_helicopter_hours (total_cost : ℤ) (cost_per_hour : ℤ) (days : ℤ) (h : total_cost = 450) (c : cost_per_hour = 75) (d : days = 3) :
  total_cost / cost_per_hour / days = 2 := by
  -- Proof goes here
  sorry

end tom_helicopter_hours_l376_376213


namespace three_digit_number_five_times_product_of_digits_l376_376340

theorem three_digit_number_five_times_product_of_digits :
  ∃ (a b c : ℕ), a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ (100 * a + 10 * b + c = 5 * a * b * c) ∧ (100 * a + 10 * b + c = 175) := 
begin
  existsi 1,
  existsi 7,
  existsi 5,
  split, { norm_num }, -- a > 0
  split, { norm_num }, -- a < 10
  split, { norm_num }, -- b < 10
  split, { norm_num }, -- c < 10
  split,
  { calc 100 * 1 + 10 * 7 + 5 = 100 + 70 + 5 : by norm_num
                        ... = 175 : by norm_num
                        ... = 5 * 1 * 7 * 5 : by norm_num [1*7*5] },
  { norm_num }
end

end three_digit_number_five_times_product_of_digits_l376_376340


namespace patricia_candies_l376_376507

theorem patricia_candies (initial_candies : ℕ) (taken_away : ℕ) :
  initial_candies = 76 ∧ taken_away = 5 → initial_candies - taken_away = 71 :=
by
  intros h
  cases h with h_initial h_taken
  rw [h_initial, h_taken]
  exact rfl

end patricia_candies_l376_376507


namespace cube_add_constant_135002_l376_376950

theorem cube_add_constant_135002 (n : ℤ) : 
  (∃ m : ℤ, m = n + 1 ∧ m^3 - n^3 = 135002) →
  (n = 149 ∨ n = -151) :=
by
  -- This is where the proof should go
  sorry

end cube_add_constant_135002_l376_376950


namespace find_value_a_pow_2m_plus_n_l376_376364

theorem find_value_a_pow_2m_plus_n (a : ℝ) (m n : ℝ) (h1 : log a 2 = m) (h2 : log a 3 = n) : a^(2 * m + n) = 12 :=
by 
  sorry

end find_value_a_pow_2m_plus_n_l376_376364


namespace maximum_y1_minus_y2_l376_376677

-- Definitions of the conditions from the problem
def ellipse_eqn (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

def on_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def foc1 : ℝ × ℝ := (-1, 0)
def foc2 : ℝ × ℝ := (1, 0)

def point_P_on_ellipse (P : ℝ × ℝ) : Prop := 
  (ellipse_eqn P.1 P.2) ∧ (on_first_quadrant P.1 P.2)

def line_eqn_PF1 (P Q : ℝ × ℝ) : Prop :=
  Q.1 = (P.1 + 1) / P.2 * Q.2 - 1

def line_eqn_PF2 (P Q : ℝ × ℝ) : Prop :=
  Q.1 = (P.1 - 1) / P.2 * Q.2 + 1

def intersects_ellipse (Q : ℝ × ℝ) : Prop := ellipse_eqn Q.1 Q.2

-- Main statement to prove
theorem maximum_y1_minus_y2 
  (P Q1 Q2 : ℝ × ℝ) 
  (hp : point_P_on_ellipse P) 
  (hq1 : intersects_ellipse Q1) 
  (hq2 : intersects_ellipse Q2) 
  (hline1 : line_eqn_PF1 P Q1) 
  (hline2 : line_eqn_PF2 P Q2) 
  : y1 - y2 ≤ (2 * real.sqrt 2) / 3 :=
sorry  -- The proof is omitted since it's not required.

end maximum_y1_minus_y2_l376_376677


namespace quadratic_sum_solutions_l376_376705

theorem quadratic_sum_solutions : 
  (let f : ℝ → ℝ := λ x => x^2 - 6 * x + 5 - 2 * x + 8 in
    ∀ x : ℝ, f x = 0 → x ∈ {4 + real.sqrt 3, 4 - real.sqrt 3} ∧ (4 + real.sqrt 3) + (4 - real.sqrt 3) = 8) :=
by
  sorry

end quadratic_sum_solutions_l376_376705


namespace eccentricity_of_hyperbola_range_l376_376073

def hyperbola_eccentricity_range (a b: ℝ) (h_a: a > 0) (h_b: b > 0) 
  (h_intersect: ∀ x: ℝ, (3 * x = y) → (x^2 / a^2 - y^2 / b^2 = 1)) 
  : set ℝ :=
  { e : ℝ | e > sqrt 10 }

theorem eccentricity_of_hyperbola_range (a b: ℝ) (h_a: a > 0) (h_b: b > 0) 
  (h_intersect: (b / a) > 3):
  hyperbola_eccentricity_range a b h_a h_b h_intersect = { e : ℝ | e > sqrt 10 } :=
sorry

end eccentricity_of_hyperbola_range_l376_376073


namespace geometric_sequence_increasing_condition_l376_376420

noncomputable def is_geometric (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (h_geo : is_geometric a) (h_cond : a 0 < a 1 ∧ a 1 < a 2) :
  ¬(∀ n : ℕ, a n < a (n + 1)) → (a 0 < a 1 ∧ a 1 < a 2) :=
sorry

end geometric_sequence_increasing_condition_l376_376420


namespace charlie_rope_first_post_l376_376687

theorem charlie_rope_first_post (X : ℕ) (h : X + 20 + 14 + 12 = 70) : X = 24 :=
sorry

end charlie_rope_first_post_l376_376687


namespace cosine_of_angle_l376_376793

open Real

noncomputable def vector_a := (1, -1)
noncomputable def vector_b (t : ℝ) := (-2, t)
def perpendicular_condition (t : ℝ) := (1, -1) • ((1, -1) - (-2, t)) = 0

theorem cosine_of_angle {t : ℝ} (h : perpendicular_condition t) :
  let a := vector_a
  let b := vector_b t in
  t = -4 → cos (arccos ((1, -1) • (-2, -4) / (sqrt (1^2 + (-1)^2) * sqrt ((-2)^2 + (-4)^2)))) = sqrt (10) / 10 :=
by
  sorry

end cosine_of_angle_l376_376793


namespace decimal_sum_difference_l376_376684

theorem decimal_sum_difference :
  (0.5 - 0.03 + 0.007 + 0.0008 = 0.4778) :=
by
  sorry

end decimal_sum_difference_l376_376684


namespace functions_correct_l376_376638

-- Define the conditions
def fixed_cost : ℝ := 200
def variable_cost_per_unit : ℝ := 0.003
def selling_price_per_unit : ℝ := 0.005

-- Define the functions
def total_cost (X : ℝ) : ℝ := fixed_cost + variable_cost_per_unit * X
def unit_cost (X : ℝ) : ℝ := total_cost(X) / X
def sales_revenue (X : ℝ) : ℝ := selling_price_per_unit * X
def profit (X : ℝ) : ℝ := sales_revenue(X) - total_cost(X)

-- The theorem to prove
theorem functions_correct (X : ℝ) (hX : X > 0) :
  total_cost(X) = 200 + 0.003 * X ∧
  unit_cost(X) = 200 / X + 0.003 ∧
  sales_revenue(X) = 0.005 * X ∧
  profit(X) = 0.002 * X - 200 := by
  sorry

end functions_correct_l376_376638


namespace monotonically_increasing_intervals_find_b_plus_c_l376_376390
-- Note: Mathlib should cover the necessary mathematical concepts.

noncomputable def f (x : ℝ) : ℝ := sin (x + π / 6) + 2 * sin (x / 2)^2

-- First problem (I)
theorem monotonically_increasing_intervals (k : ℤ) : 
  ∀ x, (2 * k * π - π / 3 ≤ x) ∧ (x ≤ 2 * k * π + 2 * π / 3) → monotone_incr_on f x :=
sorry

-- Second problem (II)
theorem find_b_plus_c (A : ℝ) (b c : ℝ) (S : ℝ) (a : ℝ) : 
  f A = 3 / 2 ∧ S = sqrt 3 / 2 ∧ a = sqrt 3 → b + c = 3 := 
sorry

end monotonically_increasing_intervals_find_b_plus_c_l376_376390


namespace find_distance_and_sum_l376_376199

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Sphere where
  center : Point3D
  radius : ℝ

structure Triangle3D where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D

def distance (a b : Point3D) : ℝ := 
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2 + (a.z - b.z)^2)

def is_right_triangle (t : Triangle3D) : Prop :=
  let a := distance t.p1 t.p2
  let b := distance t.p2 t.p3
  let c := distance t.p1 t.p3
  a^2 + b^2 = c^2 ∨ 
  a^2 + c^2 = b^2 ∨
  b^2 + c^2 = a^2

def distance_from_point_to_plane (s : Sphere) (t : Triangle3D) : ℝ :=
  let p := t.p1
  let q := t.p2
  let r := t.p3
  let d := (p.x*q.y*r.z + p.y*q.z*r.x + p.z*q.x*r.y - p.x*q.z*r.y - p.y*q.x*r.z - p.z*q.y*r.x) / 
           Real.sqrt (((q.y - p.y)*(r.z - p.z) - (q.z - p.z)*(r.y - p.y))^2 +
                      ((q.z - p.z)*(r.x - p.x) - (q.x - p.x)*(r.z - p.z))^2 +
                      ((q.x - p.x)*(r.y - p.y) - (q.y - p.y)*(r.x - p.x))^2)
  Real.abs (distance s.center t.p1 - d)

theorem find_distance_and_sum (P Q R S : Point3D) 
  (h1 : distance P Q = 15)
  (h2 : distance Q R = 20)
  (h3 : distance P R = 25)
  (hs : Sphere S 18)
  (ht : Triangle3D P Q R)
  (h_right : is_right_triangle ht) :
  let distance := distance_from_point_to_plane hs ht in
  let a := 5
  let b := 671
  let c := 4
  distance = 5 * Real.sqrt 671 / 4 ∧ a + b + c = 680 :=
by skip

#eval 1 -- Ensuring successful Lean code build

end find_distance_and_sum_l376_376199


namespace probability_three_dice_same_number_is_1_div_36_l376_376609

noncomputable def probability_same_number_three_dice : ℚ :=
  let first_die := 1
  let second_die := 1 / 6
  let third_die := 1 / 6
  first_die * second_die * third_die

theorem probability_three_dice_same_number_is_1_div_36 : probability_same_number_three_dice = 1 / 36 :=
  sorry

end probability_three_dice_same_number_is_1_div_36_l376_376609


namespace range_of_t_l376_376780

noncomputable def f (x : ℝ) := 1 / (x + 2)

def θ (n : ℕ) := π / 2 - atan (n / f n)

def cos_θ_over_sin_θ (n : ℕ) := cos (θ n) / sin (θ n)

def sum_cos_θ_over_sin_θ (n : ℕ) := ∑ k in Finset.range (n + 1), cos_θ_over_sin_θ k

theorem range_of_t (t : ℝ) : (∀ n : ℕ, sum_cos_θ_over_sin_θ n < t) ↔ t ≥ 3 / 4 :=
sorry

end range_of_t_l376_376780


namespace negate_universal_proposition_l376_376195

theorem negate_universal_proposition : 
  (¬ (∀ x : ℝ, x^2 - 2 * x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) :=
by sorry

end negate_universal_proposition_l376_376195


namespace fraction_of_quarters_l376_376515
/-
Roger collects the first 18 U.S. state quarters. Five states joined the union during the decade 1790 through 1799.
Prove that the fraction of Roger's quarters representing states that joined the union during this decade is 5/18.
-/
theorem fraction_of_quarters (total_quarters : ℕ) (states_joined_1790s : ℕ)
  (h1 : total_quarters = 18) (h2 : states_joined_1790s = 5) :
  states_joined_1790s / total_quarters = 5 / 18 :=
by {
  rw [h1, h2],
  norm_num,
}

end fraction_of_quarters_l376_376515


namespace angle_E_is_100_degrees_l376_376166

theorem angle_E_is_100_degrees (EFGH : Parallelogram) (exterior_angle_F : ∀ F : Point, measure_angle (exterior_angle_at F) = 80) :
  measure_angle E = 100 :=
sorry

end angle_E_is_100_degrees_l376_376166


namespace no_consecutive_natural_numbers_after_transformation_l376_376153

/-- Initial sequence on the board -/
def initial_sequence : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

/-- Transformation sequence: replacing each element with 1, 2, ..., 10 -/
def transformation_sequence : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

/-- Condition for a sequence to be 10 consecutive natural numbers -/
def is_consecutive (l : List ℕ) : Prop :=
  ∃ (n : ℕ), l = List.range 10 |>.map (λ i, n + i)

theorem no_consecutive_natural_numbers_after_transformation :
  ∀ l : List ℕ, 
  ((∀ i, i < 10 → l.nth i = (initial_sequence.nth i >>= (λ x, transformation_sequence.nth (x - 1)))) → 
  ¬ is_consecutive l) :=
by {
  intros l h,
  sorry -- Proof goes here
}

end no_consecutive_natural_numbers_after_transformation_l376_376153


namespace smallest_n_exists_l376_376723

theorem smallest_n_exists :
  ∃ n : ℕ, (∀ s : finset ℕ, s.card = n → 
            ∃ a b ∈ s, a ≠ b ∧ (a^2 - b^2) % 2004 = 0) ∧
            (∀ m < n, ∃ s : finset ℕ, s.card = m ∧ 
                       ∀ a b ∈ s, a ≠ b → (a^2 - b^2) % 2004 ≠ 0) :=
begin
  -- The correct smallest n is known to be 337
  use 337,
  sorry -- Proof is omitted
end

end smallest_n_exists_l376_376723


namespace min_a_for_increasing_interval_l376_376781

def f (x a : ℝ) : ℝ := x^2 + (a - 2) * x - 1

theorem min_a_for_increasing_interval (a : ℝ) : (∀ x : ℝ, x ≥ 2 → f x a ≤ f (x + 1) a) ↔ a ≥ -2 :=
sorry

end min_a_for_increasing_interval_l376_376781


namespace shifted_parabola_expression_l376_376532

theorem shifted_parabola_expression (x y x' y' : ℝ) 
  (h_initial : y = (x + 2)^2 + 3)
  (h_shift_right : x' = x - 3)
  (h_shift_down : y' = y - 2)
  : y' = (x' - 1)^2 + 1 := 
sorry

end shifted_parabola_expression_l376_376532


namespace general_term_formula_sum_reciprocal_less_three_fourths_l376_376385

variable {d : ℕ} (h_d : d ≠ 0)
variable {a : ℕ → ℕ} (S : ℕ → ℕ)
variable (hS9 : S 9 = 99)
variable (h_geom : a 7 ^ 2 = a 4 * a 12)
variable (a_def : ∀ n, a n = d * n + (a 1))

-- Proof Problem 1: General term of the arithmetic sequence
theorem general_term_formula (h2 : a 4 = a 1 + 3 * d)
    (h3 : a 7 = a 1 + 6 * d)
    (h4 : a 12 = a 1 + 11 * d) :
    ∀ n, a n = 2 * n + 1 := 
sorry

-- Proof Problem 2: Prove T_n < 3/4
theorem sum_reciprocal_less_three_fourths (T : ℕ → ℝ)
    (hT : ∀ n, T n = ∑ k in finset.range n, 1 / S (k + 1)) :
    ∀ n, T n < 3 / 4 := 
sorry

end general_term_formula_sum_reciprocal_less_three_fourths_l376_376385


namespace line_through_fixed_point_and_circle_l376_376035

/-- Given point P and a circle with parametric equation, prove the straight line equations -/
theorem line_through_fixed_point_and_circle 
  (P : ℝ × ℝ) (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) 
  (A B : ℝ × ℝ) (OP : ℝ → ℝ → ℝ) (AB : ℝ → ℝ → Prop) : 
  (P = (-3, -3 / 2)) ∧ 
  (C x y ↔ x^2 + y^2 = 25 ∧ ∃ θ : ℝ, x = 5 * cos θ ∧ y = 5 * sin θ) ∧ 
  (∀ A B : ℝ × ℝ, AB A B ↔ C (A.1) (A.2) ∧ C (B.1) (B.2) ∧ (real.dist A B = 8)) ∧ 
  (by cases A, B with A B; exact P = midpoint ℝ A B) → 
  (l x y ↔ (3 * x + 4 * y + 15 = 0) ∨ (x = -3)) ∧ 
  (∀ A B : ℝ × ℝ, OP A B P → AB x y ↔ (4 * x + 2 * y + 15 = 0)) := sorry

end line_through_fixed_point_and_circle_l376_376035


namespace train_speed_l376_376245

/-- Let a train 140 meters long pass by a man running at 6 km/h 
    in the direction opposite to that of the train. 
    Given that the train passes the man in 6 seconds, 
    prove that the speed of the train is approximately 78 km/h. -/
theorem train_speed (train_length : ℝ) (man_speed_kmh : ℝ) (time_seconds : ℝ) (train_speed_kmh : ℝ) 
  (h_train_length : train_length = 140)
  (h_man_speed_kmh : man_speed_kmh = 6)
  (h_time_seconds : time_seconds = 6)
  (h_train_speed : (train_length / time_seconds + man_speed_kmh * 1000 / 3600) * 3600 / 1000 = train_speed_kmh) :
  train_speed_kmh ≈ 78 :=
sorry

end train_speed_l376_376245


namespace total_population_of_maplefield_l376_376592

theorem total_population_of_maplefield : 
  let num_towns := 25
  let avg_population := (4800 + 5300) / 2
  num_towns * avg_population = 126250 :=
by
  let num_towns := 25
  let avg_population := (4800 + 5300) / 2
  rw [avg_population]
  have h : (4800 + 5300) / 2 = 5050 := by norm_num
  rw [h]
  sorry

end total_population_of_maplefield_l376_376592


namespace cubic_sum_of_roots_l376_376020

theorem cubic_sum_of_roots (r s a b : ℝ) (h1 : r + s = a) (h2 : r * s = b) : 
  r^3 + s^3 = a^3 - 3 * a * b :=
by
  sorry

end cubic_sum_of_roots_l376_376020


namespace find_59th_digit_in_1_div_17_l376_376230

theorem find_59th_digit_in_1_div_17 : (decimal_expansion 59 (1 / 17)) = 4 := 
by 
  -- Given the repeating cycle of length 16 for the decimal representation of 1/17
  have cycle_length : nat := 16
  -- Check the 59th digit after the decimal point
  sorry

end find_59th_digit_in_1_div_17_l376_376230


namespace calculate_f2_f_l376_376071

variable {f : ℝ → ℝ}

-- Definition of the conditions
def tangent_line_at_x2 (f : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ → ℝ), (∀ x, L x = -x + 1) ∧ (∀ x, f x = L x + (f x - L 2))

theorem calculate_f2_f'2 (h : tangent_line_at_x2 f) :
  f 2 + deriv f 2 = -2 :=
sorry

end calculate_f2_f_l376_376071


namespace pq_necessary_not_sufficient_l376_376969

theorem pq_necessary_not_sufficient (p q : Prop) : (p ∨ q) → (p ∧ q) ↔ false :=
by sorry

end pq_necessary_not_sufficient_l376_376969


namespace shaded_area_of_rotated_semicircle_l376_376719

theorem shaded_area_of_rotated_semicircle (R : ℝ) :
  let α := 60 * (Real.pi / 180) in -- 60 degrees in radians
  let S0 := (Real.pi * R^2) / 2 in
  let SectorArea := (1 / 2) * (2 * R)^2 * (Real.pi / 3) in
  SectorArea = (2 * Real.pi * R^2) / 3 :=
by
  sorry

end shaded_area_of_rotated_semicircle_l376_376719


namespace surface_area_formulation_l376_376752

noncomputable def surface_area_sphere (O : Type) [MetricSpace O] [nondescript_sphere : Sphere O] 
  (A B C : O) 
  (angle_BAC : ℝ) (BC : ℝ) (distance_O_to_plane : ℝ) : ℝ :=
if h₀ : angle_BAC = 2 * Real.pi / 3 
  ∧ BC = 4 * Real.sqrt 3 
  ∧ distance_O_to_plane = 3 
then 4 * Real.pi * 25
else 0

theorem surface_area_formulation (O : Type) [MetricSpace O] [nondescript_sphere : Sphere O]
  (A B C : O)
  (h₀ : angle_BAC = 2*Real.pi / 3)
  (h₁ : BC = 4*Real.sqrt 3)
  (h₂ : distance_O_to_plane = 3) :
  surface_area_sphere O A B C (2*Real.pi / 3) (4*Real.sqrt 3) 3 = 100 * Real.pi :=
  by sorry

end surface_area_formulation_l376_376752


namespace output_y_for_x_eq_5_l376_376892

def compute_y (x : Int) : Int :=
  if x > 0 then 3 * x + 1 else -2 * x + 3

theorem output_y_for_x_eq_5 : compute_y 5 = 16 := by
  sorry

end output_y_for_x_eq_5_l376_376892


namespace point_line_plane_relation_l376_376426

-- Definitions for conditions
variable (M m α : Type)
variable [HasMem M m] [HasSubset m α] -- Membership and subset relationships
variable (h1 : M ∈ m) (h2 : m ⊆ α)

-- Statement of the theorem
theorem point_line_plane_relation : M ∈ m ∧ m ⊆ α := 
by
  sorry

end point_line_plane_relation_l376_376426


namespace num_terms_added_l376_376946

theorem num_terms_added {k : ℕ} (hk : 1 < k) :
  let lhs := (1 + ∑ i in (finset.range (2 ^ k - 1)), 1 / (i + 1)) in
  let lhs' := (1 + ∑ i in (finset.range (2 ^ (k + 1) - 1)), 1 / (i + 1)) in
  (lhs' - lhs) = 2 ^ k := 
sorry

end num_terms_added_l376_376946


namespace goods_train_speed_l376_376244

/-- Define the speed of the man's train in km/h -/
def man_speed_kmph : ℝ := 40

/-- Define the length of the goods train in meters -/
def goods_train_length_m : ℝ := 280

/-- Define the time it takes for the goods train to pass the man in seconds -/
def passing_time_s : ℝ := 9

/-- Define the conversion from km/h to m/s -/
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

/-- Define the speed of the man's train in m/s -/
def man_speed_mps : ℝ := kmph_to_mps man_speed_kmph

/-- Define the relative speed in m/s -/
def relative_speed_mps : ℝ := goods_train_length_m / passing_time_s

/-- Define the speed of the goods train in m/s -/
def goods_train_speed_mps : ℝ := relative_speed_mps - man_speed_mps

/-- Define the conversion from m/s to km/h -/
def mps_to_kmph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

/-- Define the speed of the goods train in km/h -/
def goods_train_speed_kmph : ℝ := mps_to_kmph goods_train_speed_mps

/-- The proof statement that the speed of the goods train is 72 km/h -/
theorem goods_train_speed : goods_train_speed_kmph = 72 := by
  sorry

end goods_train_speed_l376_376244


namespace units_digit_3m_squared_plus_2m_l376_376476

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_3m_squared_plus_2m : (3 * (m^2 + 2^m)) % 10 = 9 := by
  sorry

end units_digit_3m_squared_plus_2m_l376_376476


namespace num_multiples_of_three_in_ap_l376_376547

variable (a : ℕ → ℚ)  -- Defining the arithmetic sequence

def first_term (a1 : ℚ) := a 1 = a1
def eighth_term (a8 : ℚ) := a 8 = a8
def general_term (d : ℚ) := ∀ n : ℕ, a n = 9 + (n - 1) * d
def multiple_of_three (n : ℕ) := ∃ k : ℕ, a n = 3 * k

theorem num_multiples_of_three_in_ap 
  (a : ℕ → ℚ)
  (h1 : first_term a 9)
  (h2 : eighth_term a 12) :
  ∃ n : ℕ, n = 288 ∧ ∃ l : ℕ → Prop, ∀ k : ℕ, l k → multiple_of_three a (k * 7 + 1) :=
sorry

end num_multiples_of_three_in_ap_l376_376547


namespace book_prices_purchasing_plans_l376_376503

theorem book_prices (x y : ℕ) (h1 : 20 * x + 40 * y = 1600) (h2 : 20 * x = 30 * y + 200) : x = 40 ∧ y = 20 :=
by
  sorry

theorem purchasing_plans (m : ℕ) (h3 : 2 * m + 20 ≥ 70) (h4 : 40 * m + 20 * (m + 20) ≤ 2000) :
  (m = 25 ∧ m + 20 = 45) ∨ (m = 26 ∧ m + 20 = 46) :=
by
  -- proof steps
  sorry

end book_prices_purchasing_plans_l376_376503


namespace number_of_mappings_eq_eight_l376_376795

open Function Fintype

theorem number_of_mappings_eq_eight :
  let A := {a, b, c}
  let B := {1, 2}
  (card (B → A)) = 8 := by
{
  let A := {a, b, c}
  let B := {1, 2}
  have h₁ : Fintype.card B = 2, by
  {
    -- Here we will formally prove that |B| = 2
    sorry
  },
  have h₂ : Fintype.card A = 3, by
  {
    -- Here we will formally prove that |A| = 3
    sorry
  },
  have h₃ : card (B → A) = 8, by
  {
    -- Here we use the combinatorial rule to count the mappings
    sorry
  },
  exact h₃,
}

end number_of_mappings_eq_eight_l376_376795


namespace books_per_author_l376_376840

theorem books_per_author (total_books : ℕ) (authors : ℕ) (h1 : total_books = 198) (h2 : authors = 6) : total_books / authors = 33 :=
by sorry

end books_per_author_l376_376840


namespace sugar_percentage_in_new_solution_l376_376266

open Real

noncomputable def original_volume : ℝ := 450
noncomputable def original_sugar_percentage : ℝ := 20 / 100
noncomputable def added_sugar : ℝ := 7.5
noncomputable def added_water : ℝ := 20
noncomputable def added_kola : ℝ := 8.1
noncomputable def added_flavoring : ℝ := 2.3

noncomputable def original_sugar_amount : ℝ := original_volume * original_sugar_percentage
noncomputable def total_sugar_amount : ℝ := original_sugar_amount + added_sugar
noncomputable def new_total_volume : ℝ := original_volume + added_water + added_kola + added_flavoring + added_sugar
noncomputable def new_sugar_percentage : ℝ := (total_sugar_amount / new_total_volume) * 100

theorem sugar_percentage_in_new_solution : abs (new_sugar_percentage - 19.97) < 0.01 := sorry

end sugar_percentage_in_new_solution_l376_376266


namespace probability_two_red_faces_l376_376641

-- Define that the original cube is cut into 512 smaller cubes
def original_cube_num_cubes : ℕ := 512

-- Define that the smaller cubes are mixed and placed into a bag
def cubes_mixed_in_bag : Prop := True  -- Placeholder for the condition that cubes are mixed.

-- The main theorem about the probability of picking a small cube with exactly two red faces
theorem probability_two_red_faces :
  (* Proving that the probability is 9/64 when picking a random cube with two faces painted red *)
  (let num_edges := 12 in
   let cubes_per_edge := 8 - 2 in
   let num_desired := num_edges * cubes_per_edge in
   let total_cubes := original_cube_num_cubes in
   (num_desired / total_cubes : ℚ) = 9 / 64) :=
by
  sorry

end probability_two_red_faces_l376_376641


namespace sum_of_digits_of_n_hexadecimal_l376_376410

theorem sum_of_digits_of_n_hexadecimal :
  let n := ∑ i in (Finset.range 1000).filter (λ x, ∀ c in x.digits 16, c < 10), x in
  n.digits 10.sum = 21 :=
by
  sorry

end sum_of_digits_of_n_hexadecimal_l376_376410


namespace hari_joins_l376_376161

theorem hari_joins {x : ℕ} :
  let praveen_start := 3500
  let hari_start := 9000
  let total_months := 12
  (praveen_start * total_months) * 3 = (hari_start * (total_months - x)) * 2
  → x = 5 :=
by
  intros
  sorry

end hari_joins_l376_376161


namespace find_x2009_l376_376013

def sequence (x : ℕ → ℝ) := ∀ n : ℕ, (n + 1) * x (n + 1) = x n + n

def initial_condition (x : ℕ → ℝ) := x 1 = 2

theorem find_x2009 {x : ℕ → ℝ} (h_seq : sequence x) (h_init : initial_condition x) :
  x 2009 = (2009.factorial + 1) / 2009.factorial :=
sorry

end find_x2009_l376_376013


namespace Ben_shirts_is_15_l376_376292

variable (Alex_shirts Joe_shirts Ben_shirts : Nat)

def Alex_has_4 : Alex_shirts = 4 := by sorry

def Joe_has_more_than_Alex : Joe_shirts = Alex_shirts + 3 := by sorry

def Ben_has_more_than_Joe : Ben_shirts = Joe_shirts + 8 := by sorry

theorem Ben_shirts_is_15 (h1 : Alex_shirts = 4) (h2 : Joe_shirts = Alex_shirts + 3) (h3 : Ben_shirts = Joe_shirts + 8) : Ben_shirts = 15 := by
  sorry

end Ben_shirts_is_15_l376_376292


namespace correct_factorization_l376_376239

theorem correct_factorization :
  (x^2 - 2 * x + 1 = (x - 1)^2) ∧ 
  (¬ (x^2 - 4 * y^2 = (x + y) * (x - 4 * y))) ∧ 
  (¬ ((x + 4) * (x - 4) = x^2 - 16)) ∧ 
  (¬ (x^2 - 8 * x + 9 = (x - 4)^2 - 7)) :=
by
  sorry

end correct_factorization_l376_376239


namespace proj_matrix_eq_Q_l376_376122

noncomputable def Q : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    [1/3, 1/3, 2/3],
    [2/3, 2/3, 1/3],
    [1/3, 1/3, 2/3]
  ]

def normal_vector : Fin 3 → ℝ := ![
  2, -1, 1
]

def proj_matrix_correct (v : Fin 3 → ℝ) : Prop :=
  Q.mulVec v = v - (1 / 6) * (v.dotProduct normal_vector) • normal_vector

theorem proj_matrix_eq_Q (v : Fin 3 → ℝ) : proj_matrix_correct v :=
by
  sorry

end proj_matrix_eq_Q_l376_376122


namespace shortest_side_of_triangle_l376_376815

theorem shortest_side_of_triangle
  (A B C D E : Type)
  [has_dist A]
  [has_dist B]
  [has_dist C]
  [has_dist D]
  [has_dist E]
  (AD AE : ∀ (a b : Type), Prop)
  (bd de ec : ℝ)
  (angle_bisectors : ∀ (x y : Type), Prop)
  (hx : AD A D ∧ AD D C ∧ AE A E ∧ AE B E ∧ angle_bisectors B D E) :
  bd = 3 → de = 4 → ec = 5 → ∃ (y : ℝ), y = (2 * real.sqrt 55) / (real.sqrt 73) :=
begin
  intros h1 h2 h3,
  sorry
end

end shortest_side_of_triangle_l376_376815


namespace markdown_percent_l376_376991

theorem markdown_percent (P : ℝ) (hp : 0 < P) : 
  ∃ X : ℝ, X = 10 ∧ 0.63 * P = 0.70 * P - (X / 100 * 0.70 * P) :=
by
  use 10
  field_simp
  linarith

end markdown_percent_l376_376991


namespace solve_for_b_l376_376799

theorem solve_for_b (b : ℚ) (h : b - b / 4 = 5 / 2) : b = 10 / 3 :=
by 
  sorry

end solve_for_b_l376_376799


namespace xyz_logarithm_sum_l376_376544

theorem xyz_logarithm_sum :
  ∃ (X Y Z : ℕ), X > 0 ∧ Y > 0 ∧ Z > 0 ∧
  Nat.gcd X (Nat.gcd Y Z) = 1 ∧ 
  (↑X * Real.log 3 / Real.log 180 + ↑Y * Real.log 5 / Real.log 180 = ↑Z) ∧ 
  (X + Y + Z = 4) :=
by
  sorry

end xyz_logarithm_sum_l376_376544


namespace binomial_arithmetic_sequence_l376_376069

theorem binomial_arithmetic_sequence (n : ℕ) :
  let Cnk (k : ℕ) := Nat.choose n k in
  2 * Cnk 2 = Cnk 1 + Cnk 3 ∧ (∃ k, Cnk k * x^(7 - 2 * k) = 1) ↔ n = 7 :=
by
  sorry

end binomial_arithmetic_sequence_l376_376069


namespace remainder_of_S_mod_500_eq_zero_l376_376123

open Function

def R : Set ℕ := { r | ∃ n : ℕ, r = (3^n % 500) }

def S : ℕ := ∑ r in R.toFinset, r

theorem remainder_of_S_mod_500_eq_zero :
  (S % 500) = 0 := by
  sorry

end remainder_of_S_mod_500_eq_zero_l376_376123


namespace cost_of_23_days_l376_376807

-- Define the rates and the days
def cost_per_day_first_week : ℕ → ℝ := λ d, 18.00 * d
def cost_per_day_additional_weeks : ℕ → ℝ := λ d, 13.00 * d

-- Total cost function for 23 days
def total_cost (d : ℕ) : ℝ :=
  if d <= 7 then cost_per_day_first_week d
  else cost_per_day_first_week 7 + cost_per_day_additional_weeks (d - 7)

-- The proof statement
theorem cost_of_23_days : total_cost 23 = 334.00 := sorry

end cost_of_23_days_l376_376807


namespace abc_plus_one_gt_3a_l376_376784

theorem abc_plus_one_gt_3a (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a ≤ b) (h5 : b ≤ c) (h6 : a^2 + b^2 + c^2 = 9) : abc + 1 > 3a :=
sorry

end abc_plus_one_gt_3a_l376_376784


namespace length_of_train_l376_376284

theorem length_of_train (speed_km_hr : ℝ) (time_first_post : ℝ) (time_second_post : ℝ) (distance_between_posts : ℝ) : 
  speed_km_hr = 40 → time_first_post = 17.1 → time_second_post = 25.2 → distance_between_posts = 190 → (let speed_m_s := (speed_km_hr * 1000) / 3600 in let L := speed_m_s * time_first_post in L + distance_between_posts = speed_m_s * time_second_post → L = 90) :=
begin
  intros h1 h2 h3 h4,
  let speed_m_s := (speed_km_hr * 1000) / 3600,
  let L := speed_m_s * time_first_post,
  rw [h1, h2, h3, h4] at *,
  sorry -- proof omitted
end

end length_of_train_l376_376284


namespace smallest_positive_integer_form_l376_376610

theorem smallest_positive_integer_form (m n : ℤ) : 
  ∃ (x : ℤ), x > 0 ∧ x = gcd 3030 50505 ∧ ∃ (m n : ℤ), x = 3030 * m + 50505 * n := 
begin
  sorry
end

end smallest_positive_integer_form_l376_376610


namespace find_train_length_l376_376666

noncomputable def length_of_train (speed : Float) (time : Float) (bridge_length : Float) : Float :=
  let train_speed_ms := speed * 1000 / 3600
  let total_distance := train_speed_ms * time
  total_distance - bridge_length

theorem find_train_length : length_of_train 50 26.64 130 ≈ 239.9912 :=
by
  sorry

end find_train_length_l376_376666


namespace number_of_zeros_in_interval_l376_376697

noncomputable def f (x : ℝ) : ℝ :=
if hx : -1 < x ∧ x ≤ 4 then
  x^2 - 2^x
else
  sorry -- Placeholder for the extended definition using functional equation

theorem number_of_zeros_in_interval :
  (∀ x : ℝ, f x + f (x + 5) = 16) →
  (∀ x : ℝ, x ∈ Icc (-1 : ℝ) 4 → f x = x^2 - 2^x) →
  (∃ n : ℕ, n = 604 ∧
    ∀ x : ℝ, x ∈ Ioc 0 2013 → (f x = 0 ↔ x ∈ Finset.range 604)) :=
begin
  -- Placeholder for the proof
  sorry 
end

end number_of_zeros_in_interval_l376_376697


namespace derivative_ln_div_x_l376_376538

noncomputable def f (x : ℝ) := (Real.log x) / x

theorem derivative_ln_div_x (x : ℝ) (h : x ≠ 0) : deriv f x = (1 - Real.log x) / (x^2) :=
by
  sorry

end derivative_ln_div_x_l376_376538


namespace ln_inequality_complex_ln_inequality_l376_376138

noncomputable def C (α : ℝ) := 1 / α

theorem ln_inequality (α : ℝ) (x : ℝ)
  (hα : 0 < α ∧ α ≤ 1) (hx : 0 ≤ x) :
  Real.log (1 + x) ≤ C(α) * x^α :=
sorry

theorem complex_ln_inequality (α : ℝ) (z1 z2 : ℂ)
  (hα : 0 < α ∧ α ≤ 1) (hz1 : z1 ≠ 0) (hz2 : z2 ≠ 0) :
  Complex.abs (Complex.log (Complex.abs (z1 / z2))) ≤
    C(α) * ((Complex.abs ((z1 - z2) / z2))^α + (Complex.abs ((z2 - z1) / z1))^α) :=
sorry

end ln_inequality_complex_ln_inequality_l376_376138


namespace find_f_and_maximum_value_l376_376261

-- Define the function f such that f(x+1) = x^2 - 1
axiom f : ℝ → ℝ
axiom f_eq : ∀ x : ℝ, f (x + 1) = x^2 - 1

-- Main statement to prove
theorem find_f_and_maximum_value :
  (∀ x : ℝ, f x = x^2 - 2x) ∧ (f 1 = -1) := by
  sorry

end find_f_and_maximum_value_l376_376261


namespace number_of_terms_in_arithmetic_sequence_l376_376413

-- Define the necessary conditions
def a := 2
def d := 5
def l := 1007  -- last term

-- Prove the number of terms in the sequence
theorem number_of_terms_in_arithmetic_sequence : 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 202 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l376_376413


namespace angle_C_is_120_l376_376821

theorem angle_C_is_120 (C L U A : ℝ)
  (H1 : C = L)
  (H2 : L = U)
  (H3 : A = L)
  (H4 : A + L = 180)
  (H5 : 6 * C = 720) : C = 120 :=
by
  sorry

end angle_C_is_120_l376_376821


namespace union_complement_A_when_a_eq_1_A_cap_B_eq_A_range_of_a_l376_376016

def setA (a : ℝ) : Set ℝ := { x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3 }
def setB : Set ℝ := { x | -1 / 2 < x ∧ x < 2 }
def complementB : Set ℝ := { x | x ≤ -1 / 2 ∨ x ≥ 2 }

theorem union_complement_A_when_a_eq_1 :
  (complementB ∪ setA 1) = { x | x ≤ 1 ∨ x ≥ 2 } :=
by
  sorry

theorem A_cap_B_eq_A_range_of_a (a : ℝ) :
  (setA a ∩ setB = setA a) → (-1 < a ∧ a ≤ 1) :=
by
  sorry

end union_complement_A_when_a_eq_1_A_cap_B_eq_A_range_of_a_l376_376016


namespace sequence_4951_l376_376450

theorem sequence_4951 :
  (∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, 0 < n → a (n + 1) = a n + n) ∧ a 100 = 4951) :=
sorry

end sequence_4951_l376_376450


namespace A_days_to_complete_alone_l376_376636

theorem A_days_to_complete_alone
  (work_left : ℝ := 0.41666666666666663)
  (B_days : ℝ := 20)
  (combined_days : ℝ := 5)
  : ∃ (A_days : ℝ), A_days = 15 := 
by
  sorry

end A_days_to_complete_alone_l376_376636


namespace area_of_AF1F2_is_7sqrt5_over_2_l376_376761

noncomputable def calculate_area_of_triangle
  (F1 F2 A : ℝ × ℝ)
  (hF1 : F1 = (-Real.sqrt 2, 0))
  (hF2 : F2 = (Real.sqrt 2, 0))
  (hEllipse : ∃ θ : ℝ, A = (3 * Real.cos θ, Real.sqrt 7 * Real.sin θ))
  (hAngle : ∠A F1 F2 = 45) :
  Real :=
  let AF1 := (A.1 + Real.sqrt 2)^2 + A.2^2
  let AF2 := (A.1 - Real.sqrt 2)^2 + A.2^2
  1 / (2 * Real.sqrt 2) * Real.sqrt AF1 * Real.sqrt AF2

theorem area_of_AF1F2_is_7sqrt5_over_2 :
  ∃ F1 F2 A : ℝ × ℝ,
  (F1 = (-Real.sqrt 2, 0)) ∧
  (F2 = (Real.sqrt 2, 0)) ∧
  (∃ θ : ℝ, A = (3 * Real.cos θ, Real.sqrt 7 * Real.sin θ)) ∧
  (∠A F1 F2 = 45) ∧
  calculate_area_of_triangle F1 F2 A (-Real.sqrt 2, 0) (Real.sqrt 2, 0)
    (∃ θ : ℝ, A = (3 * Real.cos θ, Real.sqrt 7 * Real.sin θ))
    (∠A F1 F2 = 45) =
  7 * Real.sqrt 5 / 2 :=
sorry

end area_of_AF1F2_is_7sqrt5_over_2_l376_376761


namespace line_y_intercept_l376_376303

theorem line_y_intercept (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, 9)) (h2 : (x2, y2) = (5, 21)) :
    ∃ b : ℝ, (∀ x : ℝ, y = 4 * x + b) ∧ (b = 1) :=
by
  use 1
  sorry

end line_y_intercept_l376_376303


namespace jasmine_ribbon_length_l376_376106

theorem jasmine_ribbon_length :
  ∃ (l : ℕ), 
    let ribbon_length := 10 in
    let janice_length := 2 in
    (ribbon_length % janice_length = 0) ∧
    (ribbon_length % l = 0) ∧
    l ≠ janice_length ∧
    l ≠ 1 ∧
    l = 5 :=
sorry

end jasmine_ribbon_length_l376_376106


namespace unique_peg_placement_l376_376208

theorem unique_peg_placement :
  ∃! (f : ℕ → ℕ × ℕ), 
    (∀ c r : ℕ, (c, r) ∈ range 6 →
      ((color_of_pegs (c, r) = purple) → ∃! col : ℕ, f purple col = (c, r)) ∧
      ((color_of_pegs (c, r) = yellow) → ∃! col : ℕ, f yellow col = (c, r)) ∧
      ((color_of_pegs (c, r) = red) → ∃! col : ℕ, f red col = (c, r)) ∧
      ((color_of_pegs (c, r) = green) → ∃! col : ℕ, f green col = (c, r)) ∧
      ((color_of_pegs (c, r) = blue) → ∃! col : ℕ, f blue col = (c, r)))
    sorry

end unique_peg_placement_l376_376208


namespace pyramid_volume_of_unit_cube_l376_376997

noncomputable def volume_of_pyramid : ℝ :=
  let s := (Real.sqrt 2) / 2
  let base_area := (Real.sqrt 3) / 8
  let height := 1
  (1 / 3) * base_area * height

theorem pyramid_volume_of_unit_cube :
  volume_of_pyramid = (Real.sqrt 3) / 24 := by
  sorry

end pyramid_volume_of_unit_cube_l376_376997


namespace parallel_through_point_perpendicular_through_point_l376_376908

-- Problem (a) Definitions

variables {A B O P Q R S : Type}

noncomputable def is_midpoint (O A B : Type) := dist A O = dist O B
noncomputable def not_on_line (P A B : Type) := ¬ collinear P A B
noncomputable def line (X Y : Type) := set_of_points_on_line X Y
noncomputable def parallel (X Y W Z : Type) := ∃u v w z, ( (X - Y) * u + (W - Z) * v = 0)

theorem parallel_through_point
  (AB_midpoint : is_midpoint O A B)
  (P_not_on_AB : not_on_line P A B)
  (parallel_line : parallel (line A B) (line P S)) :
parallel (line A B) (line P S) := sorry

-- Problem (b) Definitions

noncomputable def circle (O radius : Type) := set_of_points_equal_distance_from_O O radius
noncomputable def perpendicular (X Y W Z : Type) :=
  ∃ a b c d, (X - Y) * a + (W - Z) * b = 0

theorem perpendicular_through_point
  (AB_midpoint : is_midpoint O A B)
  (circle_OA : circle O (dist O A))
  (P_outside_circle : P ∉ (circle O (dist O A)))
  (perpendicular_line : perpendicular (line A B) (line P S)) :
perpendicular (line A B) (line P S) := sorry

end parallel_through_point_perpendicular_through_point_l376_376908


namespace find_S30_l376_376931

-- Define that a sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i

-- Problem statement with given conditions
theorem find_S30 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_S10 : S 10 = 10)
  (h_S20 : S 20 = 30)
  (h_sum : ∀ n, S n = sum_first_n_terms a n) : 
  S 30 = 60 := 
sorry

end find_S30_l376_376931


namespace cookie_price_l376_376536

-- Define the conditions from the problem
def total_money_from_cupcakes (cupcakes : Nat) (price_per_cupcake : Nat) : Nat :=
  cupcakes * price_per_cupcake

def total_money_from_cookies (cookies : Nat) (price_per_cookie : Nat) : Nat :=
  cookies * price_per_cookie

def total_cost (items : List Nat) : Nat :=
  items.foldl (· + ·) 0

-- Assume numbers involved in the problem
def num_cupcakes := 50
def price_per_cupcake := 2
def num_cookies := 40
noncomputable def price_per_cookie := 0.5 -- the unknown we need to prove
def num_basketballs := 2
def price_per_basketball := 40
def num_energy_drinks := 20
def price_per_energy_drink := 2

-- Define the main proof problem
theorem cookie_price :
  let total_money_made := total_money_from_cupcakes num_cupcakes price_per_cupcake
                         + total_money_from_cookies num_cookies price_per_cookie in
  let total_money_spent := total_cost [num_basketballs * price_per_basketball,
                                        num_energy_drinks * price_per_energy_drink] in
  total_money_made = total_money_spent → price_per_cookie = 0.5 :=
by
  let total_money_made := total_money_from_cupcakes num_cupcakes price_per_cupcake
                         + total_money_from_cookies num_cookies price_per_cookie
  let total_money_spent := total_cost [
                            num_basketballs * price_per_basketball,
                            num_energy_drinks * price_per_energy_drink]
  intro h
  sorry

end cookie_price_l376_376536


namespace triangle_perimeter_is_43_triangle_area_is_7446_l376_376996

def triangle_sides : ℝ × ℝ × ℝ := (10, 15, 18)

def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_perimeter_is_43 :
  perimeter (triangle_sides.1) (triangle_sides.2) (triangle_sides.3) = 43 :=
sorry

theorem triangle_area_is_7446 :
  heron_area (triangle_sides.1) (triangle_sides.2) (triangle_sides.3) ≈ 74.46 :=
sorry

end triangle_perimeter_is_43_triangle_area_is_7446_l376_376996


namespace probability_A_not_selected_l376_376613

theorem probability_A_not_selected :
  let students := {'A', 'B', 'C', 'D', 'E'}
  let total_pairs := ((students.to_finset.powerset.filter (λ s, s.card = 2)).card : ℚ)
  let favorable_pairs := ((students.to_finset.erase 'A').powerset.filter (λ s, s.card = 2)).card
  total_pairs = 10 →
  favorable_pairs = 6 →
  favorable_pairs / total_pairs = 3 / 5 :=
by intros; sorry

end probability_A_not_selected_l376_376613


namespace kolya_prevent_divisible_by_nine_l376_376845

theorem kolya_prevent_divisible_by_nine :
  ∃ (digits : Fin 20 → Fin 5) (is_divisible_by_nine : ℕ → Bool),
    (∀ n, digits n ∈ Finset.range 1 6) ∧
    (∀ n, (n % 2 = 0 → ∃ k, digits n = k) ∧ (n % 2 = 1 → ∃ v, digits n = v)) ∧
    (∀ m : ℕ, m = (∑ i in Finset.range 20, digits i).val → is_divisible_by_nine m = false) :=
begin
  sorry
end

end kolya_prevent_divisible_by_nine_l376_376845


namespace plane_equation_l376_376701

/-- Given a plane parametrically as:
    v = [1 + s - t, 2 - 2s + t, 3 - 3s + 3t],
    the required plane in the form Ax + By + Cz + D = 0 is
    equivalently 3x + z - 6 = 0. -/
theorem plane_equation (s t x y z : ℝ) :
  (∃ A B C D : ℤ, 
  (A : ℝ) * x + (B : ℝ) * y + (C : ℝ) * z + (D : ℝ) = 0 ∧
  gcd A B C D = 1) →
  (x = 1 + s - t) →
  (y = 2 - 2s + t) →
  (z = 3 - 3s + 3t) →
  3 * x + z - 6 = 0 :=
by
  sorry

end plane_equation_l376_376701


namespace max_sum_in_8x10_table_l376_376088

def black_cells (row : list (option ℕ)) : ℕ :=
  row.foldr (λ cell acc => match cell with | some _ => acc + 1 | none => acc) 0

def row_sum (row : list (option ℕ)) : ℕ :=
  let x := black_cells row
  let white_cells := row.length - x
  x * white_cells

-- Given the table is 8 rows and 10 columns
def max_table_sum (table : list (list (option ℕ))) : Prop :=
  table.length = 8 ∧ (∀ row, row.length = 10) →
  (∑ row in table, row_sum row) ≤ 200

theorem max_sum_in_8x10_table : max_table_sum :=
sorry

end max_sum_in_8x10_table_l376_376088


namespace area_of_triangle_is_3_l376_376210

noncomputable def area_of_triangle_ABC (A B C : ℝ × ℝ) : ℝ :=
1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_is_3 : 
  ∀ (A B C : ℝ × ℝ), 
  A = (-5, -2) → 
  B = (0, 0) → 
  C = (7, -4) →
  area_of_triangle_ABC A B C = 3 :=
by
  intros A B C hA hB hC
  rw [hA, hB, hC]
  sorry

end area_of_triangle_is_3_l376_376210


namespace sin_405_eq_sqrt2_div_2_l376_376320

theorem sin_405_eq_sqrt2_div_2 :
  Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_eq_sqrt2_div_2_l376_376320


namespace fish_life_span_in_months_fish_life_span_correct_l376_376171

theorem fish_life_span_in_months 
  (breed_a_life : ℕ) (breed_a_popularity : ℝ)
  (breed_b_life : ℕ) (breed_b_popularity : ℝ)
  (breed_c_life : ℕ) (breed_c_popularity : ℝ)
  (breed_d_life : ℕ) (breed_d_popularity : ℝ)
  (years_to_months : ℕ) : ℕ :=
  (breed_a_life * breed_a_popularity + breed_b_life * breed_b_popularity + breed_c_life * breed_c_popularity + breed_d_life * breed_d_popularity) / 4 + (2 * years_to_months)

/-
Given
  breed_a_life = 10
  breed_a_popularity = 0.4
  breed_b_life = 12
  breed_b_popularity = 0.3
  breed_c_life = 14
  breed_c_popularity = 0.2
  breed_d_life = 16
  breed_d_popularity = 0.1
  years_to_months = 12
Show
  weighted average fish life-span in months = 168
-/
theorem fish_life_span_correct : fish_life_span_in_months 10 0.4 12 0.3 14 0.2 16 0.1 12 = 168 := 
sorry 

end fish_life_span_in_months_fish_life_span_correct_l376_376171


namespace rectangular_plot_area_l376_376192

theorem rectangular_plot_area (Breadth Length Area : ℕ): 
  (Length = 3 * Breadth) → 
  (Breadth = 30) → 
  (Area = Length * Breadth) → 
  Area = 2700 :=
by 
  intros h_length h_breadth h_area
  rw [h_breadth] at h_length
  rw [h_length, h_breadth] at h_area
  exact h_area

end rectangular_plot_area_l376_376192


namespace prob_diff_fruit_correct_l376_376302

noncomputable def prob_same_all_apple : ℝ := (0.4)^3
noncomputable def prob_same_all_orange : ℝ := (0.3)^3
noncomputable def prob_same_all_banana : ℝ := (0.2)^3
noncomputable def prob_same_all_grape : ℝ := (0.1)^3

noncomputable def prob_same_fruit_all_day : ℝ := 
  prob_same_all_apple + prob_same_all_orange + prob_same_all_banana + prob_same_all_grape

noncomputable def prob_diff_fruit (prob_same : ℝ) : ℝ := 1 - prob_same

theorem prob_diff_fruit_correct :
  prob_diff_fruit prob_same_fruit_all_day = 0.9 :=
by
  sorry

end prob_diff_fruit_correct_l376_376302


namespace remainder_sum_remainders_mod_500_l376_376128

open Nat

/-- Define the set of remainders of 3^n mod 500 for nonnegative integers n -/
def remainders_mod_500 : Set ℕ := {r | ∃ n : ℕ, r = 3^n % 500}

/-- Define the sum of the elements in the set of remainders -/
def S : ℕ := remainders_mod_500.sum (λ x, x)

theorem remainder_sum_remainders_mod_500 (x : ℕ)
  (hx : S % 500 = x) :
  S % 500 = x := by
  sorry

end remainder_sum_remainders_mod_500_l376_376128


namespace efficiency_ratio_A_B_is_half_l376_376276

noncomputable def workRate (days : ℕ) : ℚ := 1 / days

-- Conditions
def B_work_rate : ℚ := workRate 15
def combined_work_rate : ℚ := workRate 10

-- Let x be the efficiency ratio of A's work rate to B's work rate
def A_to_B_efficiency_ratio (x : ℚ) : Prop :=
  x * B_work_rate + B_work_rate = combined_work_rate

-- The theorem
theorem efficiency_ratio_A_B_is_half (x : ℚ) (h : A_to_B_efficiency_ratio x) : x = 1 / 2 :=
by
  unfold A_to_B_efficiency_ratio at h
  rw [B_work_rate, combined_work_rate] at h
  linarith

end efficiency_ratio_A_B_is_half_l376_376276


namespace goods_train_speed_l376_376645

theorem goods_train_speed
  (length_train : ℝ)
  (length_platform : ℝ)
  (time_taken : ℝ)
  (speed_kmph : ℝ)
  (h1 : length_train = 240.0416)
  (h2 : length_platform = 280)
  (h3 : time_taken = 26)
  (h4 : speed_kmph = 72.00576) :
  speed_kmph = ((length_train + length_platform) / time_taken) * 3.6 := sorry

end goods_train_speed_l376_376645


namespace max_value_of_expression_l376_376031

open probability_theory

noncomputable def normal_max_expression (σ a x : ℝ) : ℝ :=
  (1 / (1 + a * x)) - (1 / (1 + 3 * x))

theorem max_value_of_expression (σ : ℝ) (x : ℝ) (h_pos_x : x > 0) :
  ∀ (ξ : ℝ →ₐ measure_theory.probability_measure ℝ),
  (ξ ∘ measure_theory.probability.spec_of (normal_distribution 2 σ)) →
  (probability P(λ ξ, ξ ≤ 1) = P(λ ξ, ξ ≥ a + 2)) →
  ∃ (a : ℝ), normal_max_expression σ a x ≤ 2 - sqrt 3 := by
sorry

end max_value_of_expression_l376_376031


namespace x_intercept_l376_376830

theorem x_intercept (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -8) (hy2 : y2 = -6) : 
  ∃ x : ℝ, (y = 0) ∧ (∃ m : ℝ, m = (y2 - y1) / (x2 - x1) ∧ y1 - y = m * (x1 - x)) ∧ x = 4 :=
sorry

end x_intercept_l376_376830


namespace local_minimum_at_one_iff_l376_376044

open Real

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x => x^3 - 2*a*x^2 + a^2*x + 1

theorem local_minimum_at_one_iff (a : ℝ) : (∀ f', f' = deriv (f a) → f' 1 = 0 ∧ (forall x, f' x = 3*x^2 - 4*a*x + a^2) → (∀ x, 1 < x → f' x > 0) ∧ (∀ x, x < 1 → f' x < 0)) ↔ a = 1 :=
by
  sorry

end local_minimum_at_one_iff_l376_376044


namespace projection_onto_plane_l376_376852

open Matrix

def normal_vector : Vector ℝ 3 := ![1, -2, 1]

def proj_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![5/6, 1/3, -1/6],
    ![1/3, 1/3, 1/3],
    ![-1/6, 1/3, 5/6]]

def projection (v : Vector ℝ 3) : Vector ℝ 3 :=
  proj_matrix.mulVec v

theorem projection_onto_plane (v : Vector ℝ 3) :
  projection v = proj_matrix.mulVec v := by
  sorry

end projection_onto_plane_l376_376852


namespace maximize_revenue_l376_376267

noncomputable def revenue (p : ℝ) : ℝ :=
  p * (150 - 6 * p)

theorem maximize_revenue : ∃ (p : ℝ), p = 12.5 ∧ p ≤ 30 ∧ ∀ q ≤ 30, revenue q ≤ revenue 12.5 := by 
  sorry

end maximize_revenue_l376_376267


namespace carrie_mom_money_l376_376686

theorem carrie_mom_money :
  ∀ (sweater_cost t_shirt_cost shoes_cost left_money total_money : ℕ),
  sweater_cost = 24 →
  t_shirt_cost = 6 →
  shoes_cost = 11 →
  left_money = 50 →
  total_money = sweater_cost + t_shirt_cost + shoes_cost + left_money →
  total_money = 91 :=
sorry

end carrie_mom_money_l376_376686


namespace high_jump_mode_median_l376_376300

theorem high_jump_mode_median (heights : List ℝ) (counts : List ℕ)
  (h_data : heights = [1.50, 1.60, 1.65, 1.70, 1.75]) 
  (h_counts : counts = [1, 2, 3, 5, 2]) :
  let data := heights.zip counts
  let mode := 1.70
  let median := 1.70
  (∃ h c, c = 5 ∧ (h, c) ∈ data ∧ h = mode) ∧
  (∃ m, (m = 1.70) ∧ (∃ i, i < 13 ∧ (List.replicate (counts[0]!) 1.50 ++ List.replicate (counts[1]!) 1.60 ++ List.replicate (counts[2]!) 1.65 ++ List.replicate (counts[3]!) 1.70 ++ List.replicate (counts[4]!) 1.75).nth! i = m)) :=
by
  sorry

end high_jump_mode_median_l376_376300


namespace check_f_properties_l376_376023

variable {ℝ : Type*} [Nonempty ℝ]
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

theorem check_f_properties (h_deriv : ∀ x : ℝ, f' x < f x) :
  f 2 < Real.exp 2 * f 0 ∧ f 2018 < Real.exp 2018 * f 0 := 
sorry

end check_f_properties_l376_376023


namespace product_of_chickens_and_pigs_l376_376938

variable (P C : ℕ)

-- Conditions
def condition1 := C = P + 12
def condition2 := C + P = 52

-- Question combined with answer in terms of a proof goal
theorem product_of_chickens_and_pigs (h1 : condition1) (h2 : condition2) : P * C = 640 := 
  sorry

end product_of_chickens_and_pigs_l376_376938


namespace find_x_for_dot_product_l376_376407

theorem find_x_for_dot_product :
  let a : (ℝ × ℝ) := (1, -1)
  let b : (ℝ × ℝ) := (2, x)
  (a.1 * b.1 + a.2 * b.2 = 1) ↔ x = 1 :=
by
  sorry

end find_x_for_dot_product_l376_376407


namespace discount_price_l376_376987

theorem discount_price (original_price : ℕ) (discount_rate : ℝ) : 
  original_price = 120 → discount_rate = 0.8 → 
  original_price * discount_rate = 96 :=
by
  intros h1 h2
  have h3 := congr_arg (λ x, x * discount_rate) h1
  rw h2 at h3
  exact eq.trans h3 (by norm_num)

end discount_price_l376_376987


namespace count_valid_dates_l376_376169

/-
  To prove: There are exactly 6 other days in 2018 where each digit in the date (DD-MM-YYYY)
  appears exactly twice, besides the given example of 12-08-2018.
-/

structure Date :=
(day : ℕ)
(month : ℕ)
(year : ℕ)

def digits (n : ℕ) : list ℕ :=
  (to_string n).to_list.map (λ c, c.to_nat - '0'.to_nat)

def date_digits (d : Date) : list ℕ :=
  digits d.day ++ digits d.month ++ digits d.year

def valid_date (d : Date) : Prop :=
  d.year = 2018 ∧ (date_digits d).count_occ (0) = 2 ∧
  (date_digits d).count_occ (1) = 2 ∧ (date_digits d).count_occ (2) = 2 ∧
  (date_digits d).count_occ (8) = 2

/-- 
 There are exactly 6 other days in 2018 where each digit in the date appears exactly twice 
 besides 12-08-2018.
-/
theorem count_valid_dates : (finset.filter valid_date ({x | x.year = 2018}.to_finset)).card = 7 :=
by
  -- August 12 (12-08-2018) is one valid date.
  -- So we verify there are exactly 6 additional valid dates in this set.
  sorry

end count_valid_dates_l376_376169


namespace append_nine_to_two_digit_number_l376_376808

theorem append_nine_to_two_digit_number (t u : ℕ) (ht : t < 10) (hu : u < 10) : 
  let n := 10 * t + u in 
  (n * 10 + 9) = 100 * t + 10 * u + 9 :=
by
  sorry

end append_nine_to_two_digit_number_l376_376808


namespace abs_neg_three_l376_376310

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l376_376310


namespace union_P_Q_l376_376050

-- Definition of sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 4 }
def Q : Set ℝ := { x | -3 < x ∧ x < 3 }

-- Statement to prove
theorem union_P_Q :
  P ∪ Q = { x : ℝ | -3 < x ∧ x ≤ 4 } :=
sorry

end union_P_Q_l376_376050


namespace not_odd_function_l376_376869

def floor (x : ℝ) : ℤ := ⌊x⌋

def f (x : ℝ) : ℝ := x - floor x 

theorem not_odd_function :
  ¬ (∀ x, f (-x) = - (f x)) := 
sorry

end not_odd_function_l376_376869


namespace value_a7_l376_376089

variables {a : ℕ → ℝ}

-- Condition 1: Arithmetic sequence where each term is non-zero
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variable (h1 : arithmetic_sequence a)
-- Condition 2: 2a_3 - a_1^2 + 2a_11 = 0
variable (h2 : 2 * a 3 - (a 1)^2 + 2 * a 11 = 0)
-- Condition 3: a_3 + a_11 = 2a_7
variable (h3 : a 3 + a 11 = 2 * a 7)

theorem value_a7 : a 7 = 4 := by
  sorry

end value_a7_l376_376089


namespace quadratic_rewrite_sum_l376_376555

theorem quadratic_rewrite_sum :
  let a : ℝ := -3
  let b : ℝ := 9 / 2
  let c : ℝ := 567 / 4
  a + b + c = 143.25 :=
by 
  let a : ℝ := -3
  let b : ℝ := 9 / 2
  let c : ℝ := 567 / 4
  sorry

end quadratic_rewrite_sum_l376_376555


namespace distance_interval_l376_376293

def distance_to_town (d : ℝ) : Prop :=
  ¬(d ≥ 8) ∧ ¬(d ≤ 7) ∧ ¬(d ≤ 6) ∧ ¬(d ≥ 9)

theorem distance_interval (d : ℝ) : distance_to_town d → d ∈ Set.Ioo 7 8 :=
by
  intro h
  have h1 : d < 8 := by sorry
  have h2 : d > 7 := by sorry
  rw [Set.mem_Ioo]
  exact ⟨h2, h1⟩

end distance_interval_l376_376293


namespace circle_center_radius_l376_376188

theorem circle_center_radius (x y : ℝ) :
  (x ^ 2 + y ^ 2 + 2 * x - 4 * y - 6 = 0) →
  ((x + 1) ^ 2 + (y - 2) ^ 2 = 11) :=
by sorry

end circle_center_radius_l376_376188


namespace count_valid_boards_l376_376820

/-- Define the set of values that can be placed in each cell -/
def cell_values : set ℕ := {2, 3, 4, 5}

/-- Define the size of the board -/
def board_size : ℕ := 5

/-- Define a function that checks if a list of numbers has an even sum -/
def even_sum (l : list ℕ) : Prop := l.sum % 2 = 0

/-- Define the properties of a valid board -
A board is represented as a list of lists of numbers -/
def is_valid_board (board : list (list ℕ)) : Prop :=
  (∀ row ∈ board, even_sum row) ∧
  (∀ col in (list.map (λ i, list.map (λ row, row.nth_le i sorry) board) (list.range board_size)), even_sum col) ∧
  -- Here you would also check all diagonal sums.

-- Finally, the theorem statement asserting the number of valid boards
theorem count_valid_boards : 
  (∃ boards : list (list (list ℕ)), 
    (∀ board ∈ boards, is_valid_board board) 
    ∧ boards.length = 2^39) :=
sorry

end count_valid_boards_l376_376820


namespace correct_statements_is_1_l376_376475

-- Define the entities: lines and planes
variables {l1 l2 : Line} {α β : Plane}

-- Define conditions
-- 1. Lines l1, l2 are distinct
axiom lines_distinct : l1 ≠ l2

-- 2. Planes α, β are distinct
axiom planes_distinct : α ≠ β

-- Statements
-- 1. If l1 in α and l1 parallel to β, and l2 in β and l2 parallel to α, then α parallel to β
def statement1 : Prop := (l1 ∈ α) ∧ (l1 ∥ β) ∧ (l2 ∈ β) ∧ (l2 ∥ α) → (α ∥ β)

-- 2. If l1 perpendicular to α and l2 perpendicular to α, then l1 parallel to l2
def statement2 : Prop := (l1 ⊥ α) ∧ (l2 ⊥ α) → (l1 ∥ l2)

-- 3. If l1 perpendicular to α and l1 perpendicular to l2, then l2 parallel to α
def statement3 : Prop := (l1 ⊥ α) ∧ (l1 ⊥ l2) → (l2 ∥ α)

-- 4. If α perpendicular to β and l1 in α, then l1 perpendicular to β
def statement4 : Prop := (α ⊥ β) ∧ (l1 ∈ α) → (l1 ⊥ β)

-- Define the overall correctness checking function
def number_of_correct_statements : Nat :=
  (if statement1 then 1 else 0) +
  (if statement2 then 1 else 0) +
  (if statement3 then 1 else 0) +
  (if statement4 then 1 else 0)

-- Prove that the number of correct statements is 1
theorem correct_statements_is_1 : number_of_correct_statements = 1 := by
  sorry

end correct_statements_is_1_l376_376475


namespace log_a_2_m_log_a_3_n_l376_376362

theorem log_a_2_m_log_a_3_n (a m n : ℝ) (h1 : log a 2 = m) (h2 : log a 3 = n) : a^(2*m + n) = 12 :=
sorry

end log_a_2_m_log_a_3_n_l376_376362


namespace e_max_value_l376_376861

noncomputable def b (n : ℕ) := (15^n - 1) / 14
def e (n : ℕ) := Nat.gcd (b n) (b (n+1))

theorem e_max_value : ∀ n : ℕ, e n = 1 :=
by sorry

end e_max_value_l376_376861


namespace shaded_area_equals_l376_376717

noncomputable def area_shaded_figure (R : ℝ) : ℝ :=
  let α := (60 : ℝ) * (Real.pi / 180)
  (2 * Real.pi * R^2) / 3

theorem shaded_area_equals : ∀ R : ℝ, area_shaded_figure R = (2 * Real.pi * R^2) / 3 := sorry

end shaded_area_equals_l376_376717


namespace sum_of_squares_l376_376449

theorem sum_of_squares (n : ℕ) (a : ℕ → ℕ) 
  (h : ∀ n, (finset.range n).sum (λ i, a (i + 1)) = 2 ^ n - 1) : 
  (finset.range n).sum (λ i, (a (i + 1))^2) = (4^n - 1) / 3 :=
by
  sorry

end sum_of_squares_l376_376449


namespace sqrt_product_eq_sixty_sqrt_two_l376_376682

theorem sqrt_product_eq_sixty_sqrt_two : (Real.sqrt 50) * (Real.sqrt 18) * (Real.sqrt 8) = 60 * (Real.sqrt 2) := 
by 
  sorry

end sqrt_product_eq_sixty_sqrt_two_l376_376682


namespace a_plus_b_eq_half_l376_376728

def f (a b : ℝ) (x : ℝ) : ℝ :=
if x < 1 then a * x + b
else if x < 3 then 2 * x - 1
else 10 - 4 * x

theorem a_plus_b_eq_half (a b : ℝ) (h : ∀ x : ℝ, f a b (f a b x) = x) : a + b = 1/2 :=
sorry

end a_plus_b_eq_half_l376_376728


namespace sum_of_f_l376_376984

-- Define the periodic function f
def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x < -1 then -(x + 2)^2
  else if -1 ≤ x ∧ x < 3 then x
  else f (x - 6)

theorem sum_of_f (n : ℕ) (hn : n = 2012) : 
  (finset.range (n + 1)).sum f = 338 :=
by
  sorry

end sum_of_f_l376_376984


namespace transformed_parabola_correct_l376_376534

def f (x : ℝ) : ℝ := (x + 2)^2 + 3
def g (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem transformed_parabola_correct :
  ∀ x : ℝ, g x = f (x - 3) - 2 := by
  sorry

end transformed_parabola_correct_l376_376534


namespace complement_of_P_in_U_l376_376788

def universal_set : Set ℝ := Set.univ
def set_P : Set ℝ := { x | x^2 - 5 * x - 6 ≥ 0 }
def complement_in_U (U : Set ℝ) (P : Set ℝ) : Set ℝ := U \ P

theorem complement_of_P_in_U :
  complement_in_U universal_set set_P = { x | -1 < x ∧ x < 6 } :=
by
  sorry

end complement_of_P_in_U_l376_376788


namespace derivative_at_pi_div_two_l376_376395

noncomputable def f (x : ℝ) : ℝ :=
  x * Real.cos x - Real.sin x + Real.pi / 2

theorem derivative_at_pi_div_two : 
  Deriv f (Real.pi / 2) = - (Real.pi / 2) :=
by
  sorry

end derivative_at_pi_div_two_l376_376395


namespace area_triangle_PQR_l376_376455

noncomputable def length_of_median {a b c : ℕ} (length_side_a : a) (length_side_b : b) (length_side_c : c) : ℕ :=
  sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)

theorem area_triangle_PQR : 
  (PQ PR PM : ℕ) 
  (PQ = 8) 
  (PR = 18) 
  (PM = 12) 
  :
  area_of_triangle PQ PR PM = 72 * sqrt 5 :=
sorry

end area_triangle_PQR_l376_376455


namespace find_k_l376_376792

-- Definitions of the vectors and condition about perpendicularity
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (-2, k)
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- The theorem that states if vector_a is perpendicular to (2 * vector_a - vector_b), then k = 14
theorem find_k (k : ℝ) (h : perpendicular vector_a (2 • vector_a - vector_b k)) : k = 14 := sorry

end find_k_l376_376792


namespace oliver_boxes_l376_376501

def boxes_morning : ℕ := 8
def boxes_afternoon : ℕ := 3 * boxes_morning
def total_boxes : ℕ := boxes_morning + boxes_afternoon
def boxes_given_away : ℕ := (0.25 * total_boxes.toFloat()).to_nat  -- Convert to floating point for 25% calculation then back to ℕ
def boxes_remaining : ℕ := total_boxes - boxes_given_away

theorem oliver_boxes : boxes_remaining = 24 := by
  sorry

end oliver_boxes_l376_376501


namespace median_of_36_consecutive_integers_l376_376574

theorem median_of_36_consecutive_integers (sum_of_integers : ℕ) (num_of_integers : ℕ) 
  (h1 : num_of_integers = 36) (h2 : sum_of_integers = 6 ^ 4) : 
  (sum_of_integers / num_of_integers) = 36 := 
by 
  sorry

end median_of_36_consecutive_integers_l376_376574


namespace tangent_circle_eqn_correct_l376_376562

theorem tangent_circle_eqn_correct :
  ∃ center : ℝ × ℝ, ∃ r : ℝ, r = 2 ∧ (0 < r) ∧ 
  (center = (2, 2)) ∧ 
  ((∀ x : ℝ, ((x, 0) - center).dist center = r) ∨ 
   (∀ y : ℝ, ((0, y) - center).dist center = r)) ∧ 
  ∀ (x y : ℝ), ((x - 2)^2 + (y - 2)^2 = r^2) := 
sorry

end tangent_circle_eqn_correct_l376_376562


namespace tom_initial_foreign_exchange_l376_376214

theorem tom_initial_foreign_exchange (x : ℝ) (y₀ y₁ y₂ y₃ y₄ : ℝ) :
  y₀ = x / 2 - 5 ∧
  y₁ = y₀ / 2 - 5 ∧
  y₂ = y₁ / 2 - 5 ∧
  y₃ = y₂ / 2 - 5 ∧
  y₄ = y₃ / 2 - 5 ∧
  y₄ - 5 = 100
  → x = 3355 :=
by
  intro h
  sorry

end tom_initial_foreign_exchange_l376_376214


namespace sin_tan_value_l376_376814

theorem sin_tan_value (α : ℝ) (h₁ : cos α = 3 / 5) (h₂ : sin α = - 4 / 5) : sin α * tan α = 16 / 15 := 
by
  sorry

end sin_tan_value_l376_376814


namespace number_of_zeros_l376_376699

def f (x : ℝ) : ℝ := x^2 - 2^x

theorem number_of_zeros (f : ℝ → ℝ) : 
  (∀ x : ℝ, f x + f (x + 5) = 16) → 
  (∀ x : ℝ, x ∈ Ioc (-1) 4 → f x = x^2 - 2^x) →
  (∀ (a b : ℝ), 0 ≤ a ∧ b ≤ 2013 ∧ b = a + 2013 ∧ ∀ m n : ℤ, f m = 0 ∧ n * 5 ≤ m ∧ m ≤ n * 5 + 4 → ∃ n : ℕ, n = 402) :=
begin
  sorry
end

end number_of_zeros_l376_376699


namespace extra_mangoes_l376_376878

theorem extra_mangoes (original_price_total : ℝ) (num_mangoes : ℝ) (amount_spent : ℝ) (price_reduction : ℝ) :
  original_price_total = 416.67 →
  num_mangoes = 125 →
  amount_spent = 360 →
  price_reduction = 0.1 →
  (let original_price_per_mango := original_price_total / num_mangoes,
       new_price_per_mango := original_price_per_mango * (1 - price_reduction),
       mangoes_original := amount_spent / original_price_per_mango,
       mangoes_new := amount_spent / new_price_per_mango
   in mangoes_new - mangoes_original = 12) :=
begin
  -- Given conditions
  intros h1 h2 h3 h4,
  -- Calculations
  let original_price_per_mango := 416.67 / 125,
  let new_price_per_mango := original_price_per_mango * (1 - 0.1),
  let mangoes_original := 360 / original_price_per_mango,
  let mangoes_new := 360 / new_price_per_mango,
  -- Proof skipped
  sorry
end

end extra_mangoes_l376_376878


namespace domain_of_f_parity_of_f_l376_376040

noncomputable def f (x : ℝ) : ℝ := log (x + 1) - log (1 - x)

theorem domain_of_f :
  {x : ℝ | x + 1 > 0 ∧ 1 - x > 0} = set.Ioo (-1 : ℝ) 1 :=
by
  sorry

theorem parity_of_f : ∀ x, f (-x) = -f x :=
by
  sorry

end domain_of_f_parity_of_f_l376_376040


namespace bernardo_vs_silvia_l376_376679

theorem bernardo_vs_silvia :
  let bernardo_set := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let silvia_set := {1, 2, 3, 4, 5, 6, 7, 8}
  let bernardo_picks : ℕ → Set ℕ := λ k, {s | s ⊆ bernardo_set ∧ s.card = k}
  let silvia_picks : ℕ → Set ℕ := λ k, {s | s ⊆ silvia_set ∧ s.card = k}
  let bernardo_numbers := {n | ∃ s ∈ bernardo_picks 4, n = list.reverse s.to_list}
  let silvia_numbers := {n | ∃ s ∈ silvia_picks 4, n = list.reverse s.to_list}
  let probability_larger := 
    ((84/210) + ((126-56)/126)*(3/5)*(4/9)) =
    (2/3) 
  in
  probability_larger :=
sorry

end bernardo_vs_silvia_l376_376679


namespace incorrect_statement_D_l376_376477

noncomputable theory

-- Define lines m and n, and planes alpha and beta
variables (m n : Line) (α β : Plane)

-- Preconditions based on problem conditions
axiom parallel_m_n : parallel m n
axiom different_planes : distinct α β

-- Statements as hypotheses
axiom A : ∀ m n α β, (m ⊥ α) → (m ∥ n) → (n ∥ β) → (α ⊥ β)
axiom B : ∀ m α β, (α ⊥ β) → (m ∉ α) → (m ⊥ β) → (m ∥ α)
axiom C : ∀ m α β, (m ⊥ β) → (m ∈ α) → (α ⊥ β)
axiom D : ∀ m n α β, (α ⊥ β) → (m ∈ α) → (n ∈ β) → (m ⊥ n)

-- The theorem to be proven
theorem incorrect_statement_D : (α ⊥ β) → (m ⊥ β) → (m ∉ α) → ¬ ((m ∥ α) → (m ⊥ n)) :=
begin
  sorry,
end

end incorrect_statement_D_l376_376477


namespace shop_owner_cheat_percentage_l376_376661

theorem shop_owner_cheat_percentage (CP SP : ℝ) (cheat_buy_percent profit_percent cheat_sell_percent : ℝ) (hCP : CP = 100) (hCheat_buy : cheat_buy_percent = 14) (hProfit : profit_percent = 42.5)
    (hSP : SP = CP + CP * (profit_percent / 100))
    (hCheat_formula : SP * (1 - cheat_sell_percent / 100) = CP) : 
  cheat_sell_percent ≈ 29.82 := by
    -- steps of proof will go here
    sorry

end shop_owner_cheat_percentage_l376_376661


namespace perpendicular_relation_l376_376021

-- Given conditions as definitions
variable (A B C D E F G H : Type)
variable [MetricSpace A]
variable [MetricSpace B]
variable [MetricSpace C]
variable [MetricSpace D]
variable [MetricSpace E]
variable [MetricSpace F]
variable [MetricSpace G]
variable [MetricSpace H]
variable (p q r m n : ℝ)

-- Additional given conditions and setup.
variable (AC AB BD AD BC EF : ℝ)
variable [AC_perp_AB : AC ⊥ AB]
variable [BD_perp_AB : BD ⊥ AB]
variable [AD_intersect_BC_E : Exists (intersects AD BC E)]
variable [EF_perp_AB_at_F : Exists (is_perp EF AB F)]
variable [AC_eq_p : AC = p]
variable [BD_eq_q : BD = q]
variable [EF_eq_r : EF = r]
variable [AF_eq_m : AF = m]
variable [FB_eq_n : FB = n]

theorem perpendicular_relation:
  (1/p) + (1/q) = 1/r := 
sorry

end perpendicular_relation_l376_376021


namespace arriving_time_later_l376_376226

noncomputable def usual_time : ℝ := 24
noncomputable def slow_factor : ℝ := 4 / 5

def new_time : ℝ := usual_time / slow_factor
def extra_time : ℝ := new_time - usual_time

theorem arriving_time_later : extra_time = 6 := by
  sorry

end arriving_time_later_l376_376226


namespace find_a_l376_376096

noncomputable def curve_equation (a : ℝ) : set (ℝ × ℝ) :=
{ p | (p.1 - a)^2 + (p.2 - a)^2 = 2 * a^2 }

def parametric_line (t : ℝ) : ℝ × ℝ :=
(-1 + 1/2 * t, (√3/2) * t)

theorem find_a (a t : ℝ) (P M N : ℝ × ℝ) (hP : P = (-1, 0)) (hl : ∃ t, parametric_line t = P) 
  (hPM : ∃ t₁ t₂, parametric_line t₁ = M ∧ parametric_line t₂ = N ∧ abs (dist P M) + abs (dist P N) = 5)
  (h_curve : M ∈ curve_equation a ∧ N ∈ curve_equation a) :
  a = 2 * √3 - 2 :=
by
  sorry

end find_a_l376_376096


namespace num_powers_of_3_not_27_less_than_500000_l376_376796

theorem num_powers_of_3_not_27_less_than_500000 : 
  (count (λ (n : ℕ), n < 500000 ∧ ∃ (k : ℕ), n = 3^k ∧ (¬ ∃ (m : ℕ), n = 27^m))) = 8 :=
by
  sorry

end num_powers_of_3_not_27_less_than_500000_l376_376796


namespace crescent_moon_area_l376_376932

theorem crescent_moon_area :
  let big_quarter_circle := (4 * 4 * Real.pi) / 4
  let small_semi_circle := (2 * 2 * Real.pi) / 2
  let crescent_area := big_quarter_circle - small_semi_circle
  crescent_area = 2 * Real.pi :=
by
  let big_quarter_circle := (4 * 4 * Real.pi) / 4
  let small_semi_circle := (2 * 2 * Real.pi) / 2
  let crescent_area := big_quarter_circle - small_semi_circle
  have h_bqc : big_quarter_circle = 4 * Real.pi := by
    sorry
  have h_ssc : small_semi_circle = 2 * Real.pi := by
    sorry
  have h_ca : crescent_area = 2 * Real.pi := by
    sorry
  exact h_ca

end crescent_moon_area_l376_376932


namespace even_n_non_regular_equal_angles_l376_376438

-- Definition 1: Conditions of the regular polygon
def regular_polygon (n : ℕ) (polygon : list (ℝ × ℝ)) : Prop :=
  ∃ (a : ℝ), (∀ i : ℕ, i < n → dist (polygon.nth i) (polygon.nth ((i + 1) % n)) = a) ∧
             (∀ i j : ℕ, i < n → j < n → i ≠ j → ∠ polygon.nth i polygon.nth ((i + 1) % n) polygon.nth ((i + 2) % n) =
                                                            ∠ polygon.nth j polygon.nth ((j + 1) % n) polygon.nth ((j + 2) % n))

-- Definition 2: Conditions of the new polygon with chosen points
def new_polygon (n : ℕ) (original new_polygon : list (ℝ × ℝ)) : Prop :=
  new_polygon.length = n ∧
  ∀ i : ℕ, i < n → ∃ λ : ℝ, 0 < λ ∧ λ < 1 ∧
                            dist (new_polygon.nth i) (original.nth ((i + 1) % n)) = λ * dist (original.nth i) (original.nth ((i + 1) % n))

-- Statement to prove: For even n, non-regular polygon with equal internal angles exists
theorem even_n_non_regular_equal_angles (n : ℕ) (original new_polygon : list (ℝ × ℝ)) (hreg : regular_polygon n original)
  (hnew : new_polygon n original new_polygon) :
  (∃ i : ℕ, i < n → dist (new_polygon.nth i) (new_polygon.nth ((i + 1) % n)) ≠ dist (new_polygon.nth ((i + 1) % n)) (new_polygon.nth ((i + 2) % n))) :=
  sorry

end even_n_non_regular_equal_angles_l376_376438


namespace median_of_36_consecutive_integers_l376_376564

theorem median_of_36_consecutive_integers (x : ℤ) (sum_eq : (∑ i in finset.range 36, (x + i)) = 6^4) : (17 + 18) / 2 = 36 :=
by
  -- Proof goes here
  sorry

end median_of_36_consecutive_integers_l376_376564
