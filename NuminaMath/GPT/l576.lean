import Mathlib

namespace mingyu_change_l576_576883

theorem mingyu_change :
  let eraser_cost := 350
  let pencil_cost := 180
  let erasers_count := 3
  let pencils_count := 2
  let payment := 2000
  let total_eraser_cost := erasers_count * eraser_cost
  let total_pencil_cost := pencils_count * pencil_cost
  let total_cost := total_eraser_cost + total_pencil_cost
  let change := payment - total_cost
  change = 590 := 
by
  -- The proof will go here
  sorry

end mingyu_change_l576_576883


namespace sum_ratio_l576_576724

noncomputable def geometric_sequence_sum (a1 q : ℝ) (n : ℕ) : ℝ := 
  a1 * (1 - q^n) / (1 - q)

theorem sum_ratio (a1 q : ℝ) 
  (h : 8 * (a1 * q) + (a1 * q^4) = 0) :
  geometric_sequence_sum a1 q 6 / geometric_sequence_sum a1 q 3 = -7 := 
by
  sorry

end sum_ratio_l576_576724


namespace ellipse_locus_l576_576265

theorem ellipse_locus (F : ℂ) (A B : ℂ) (h_eq1 : abs (B + 2) + abs (B - 2) = 6) (h_eq2 : abs (B - 2) * complex.exp (complex.I * (π / 3)) = A) (h_eq3 : abs (B + 2) * (1 + √3 * complex.I) / 2 = A - 2√3 * complex.I) :
  abs (A - 2) + abs (A - 2√3 * complex.I) = 6 :=
sorry

end ellipse_locus_l576_576265


namespace fraction_to_decimal_l576_576188

theorem fraction_to_decimal : 
  let decimal_eq := (7 / 12 : ℝ)
  decimal_eq ≈ 0.58333 :=
by
  sorry

end fraction_to_decimal_l576_576188


namespace percentage_change_is_neg_4_l576_576322

-- Definitions based on the conditions
variable (E₀ : ℝ) -- initial cost of a dozen eggs
variable (A₀ : ℝ) -- initial cost of 10 apples
variable (h : E₀ = A₀) -- initial costs are the same

def E₁ := (E₀ * 0.90) -- new price of eggs after 10% decrease
def A₁ := (E₀ * 1.02) -- new price of apples after 2% increase

-- Total initial cost
def C₀ := (E₀ + A₀)

-- Total new cost
def C₁ := (E₁ + A₁)

-- Show that the percentage change in total cost is -4%
theorem percentage_change_is_neg_4 :
  (E₀ = A₀) →
  ((C₁ - C₀) / C₀ * 100) = -4 := by
  sorry

end percentage_change_is_neg_4_l576_576322


namespace simplest_radical_is_sqrt7_l576_576985

def simplest_quadratic_radical (x : Real) : Prop :=
  x = Real.sqrt 7

theorem simplest_radical_is_sqrt7 :
  simplest_quadratic_radical (Real.sqrt 7) ∧ 
  ¬ simplest_quadratic_radical (3 * Real.sqrt 2) ∧ 
  ¬ simplest_quadratic_radical (2 * Real.sqrt 5) ∧ 
  ¬ simplest_quadratic_radical (Real.sqrt 3 / 3) := 
by 
  split
  . refl
  . split
    . intro h
      cases h
    . split
      . intro h
        cases h
      . intro h
        cases h

end simplest_radical_is_sqrt7_l576_576985


namespace find_q_l576_576242

noncomputable def arith_geo_sequence (d q : ℚ) (h_pos : 0 < q) (h_q : q < 1) (h_d : d ≠ 0) : Prop :=
  let a1 := d
  let a2 := a1 + d
  let a3 := a1 + 2 * d
  let b1 := d^2
  let b2 := b1 * q
  let b3 := b1 * (q^2)
  let sum_a := a1^2 + a2^2 + a3^2
  let sum_b := b1 + b2 + b3
  (sum_a / sum_b) ∈ ℤ

theorem find_q (d q : ℚ) (h_pos : 0 < q) (h_q : q < 1) (h_d : d ≠ 0) :
  arith_geo_sequence d q h_pos h_q h_d → q = 1/2 :=
by
  sorry

end find_q_l576_576242


namespace remainder_when_divided_by_20_l576_576376

theorem remainder_when_divided_by_20
  (a b : ℤ) 
  (h1 : a % 60 = 49)
  (h2 : b % 40 = 29) :
  (a + b) % 20 = 18 :=
by
  sorry

end remainder_when_divided_by_20_l576_576376


namespace min_troublemakers_l576_576513

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l576_576513


namespace jaime_can_buy_five_apples_l576_576326

theorem jaime_can_buy_five_apples :
  ∀ (L M : ℝ),
  (L = M / 2 + 1 / 2) →
  (M / 3 = L / 4 + 1 / 2) →
  (15 / M = 5) :=
by
  intros L M h1 h2
  sorry

end jaime_can_buy_five_apples_l576_576326


namespace prove_existence_and_pd_eq_r_l576_576090

variables {A B C D P : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace P]
variables {r : ℝ}

def quadrilateral_inscribed (A B C D : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] 
  (radius : ℝ) : Prop :=
  ∃ circle : Interval A B C D, circle.radius = radius

def point_on_cd (P CD : Type*) [MetricSpace P] [MetricSpace CD] : Prop :=
  ∃ point : P, point ∈ CD 

axiom conditions:
  ∃ A B C D P : Type*, 
  (quadrilateral_inscribed A B C D r) ∧
  (point_on_cd P CD) ∧
  (distance C B = distance B P) ∧
  (distance B P = distance P A) ∧
  (distance P A = distance A B)

theorem prove_existence_and_pd_eq_r :
  ∃ A B C D P : Type*, 
  (quadrilateral_inscribed A B C D r) ∧
  (point_on_cd P CD) ∧
  (distance C B = distance B P) ∧
  (distance B P = distance P A) ∧
  (distance P A = distance A B) ∧
  (distance P D = r) :=
sorry

end prove_existence_and_pd_eq_r_l576_576090


namespace proof_a_l576_576694

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ (y - 3) / (x - 2) = 3}
def N (a : ℝ) : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ a * x + 2 * y + a = 0}

-- Given conditions that M ∩ N = ∅, prove that a = -6 or a = -2
theorem proof_a (h : ∃ a : ℝ, (N a ∩ M = ∅)) : ∃ a : ℝ, a = -6 ∨ a = -2 :=
  sorry

end proof_a_l576_576694


namespace common_difference_range_l576_576335

theorem common_difference_range (a : ℕ → ℝ) (d : ℝ) (h : a 3 = 2) (h_pos : ∀ n, a n > 0) (h_arith : ∀ n, a (n + 1) = a n + d) : 0 ≤ d ∧ d < 1 :=
by
  sorry

end common_difference_range_l576_576335


namespace colin_average_mile_time_l576_576636

theorem colin_average_mile_time :
  let first_mile_time := 6
  let next_two_miles_total_time := 5 + 5
  let fourth_mile_time := 4
  let total_time := first_mile_time + next_two_miles_total_time + fourth_mile_time
  let number_of_miles := 4
  (total_time / number_of_miles) = 5 := by
    let first_mile_time := 6
    let next_two_miles_total_time := 5 + 5
    let fourth_mile_time := 4
    let total_time := first_mile_time + next_two_miles_total_time + fourth_mile_time
    let number_of_miles := 4
    have h1 : total_time = 20 := by sorry
    have h2 : total_time / number_of_miles = 20 / 4 := by sorry
    have h3 : 20 / 4 = 5 := by sorry
    exact Eq.trans (Eq.trans h2 h3) h1.symm

end colin_average_mile_time_l576_576636


namespace sin_C_eq_1_l576_576319

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
axiom triangle_angles_arith_seq : A < B ∧ B < C ∧ (B - A = C - B)
axiom sides_given : a = 1 ∧ b = √3
axiom triangle_sum : A + B + C = π

theorem sin_C_eq_1 : (sin C = 1) :=
by
  -- The proof is to be filled
  sorry

end sin_C_eq_1_l576_576319


namespace find_a_max_OMON_l576_576721

-- Definitions for the conditions provided in the problem
def parametric_circle (a θ : ℝ) : ℝ × ℝ := (a + a * Real.cos θ, a * Real.sin θ)

def polar_line (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + π / 4) = 2 * Real.sqrt 2

-- Main theorem statements
theorem find_a (θ : ℝ) (h_param : ∀ θ : ℝ, θ ∈ Ioo 0 π → parametric_circle a θ)
              (h_polar : ∀ ρ θ : ℝ, polar_line ρ θ)
              (h_intersect : |(parametric_circle a θ.1).fst - (parametric_circle a θ.2).fst| +
                             |(parametric_circle a θ.1).snd - (parametric_circle a θ.2).snd| = 2 * Real.sqrt 2)
              (h_a : 0 < a ∧ a < 5) : a = 2 := sorry

theorem max_OMON (ρ θ : ℝ)
                 (h_param : ∀ θ : ℝ, θ ∈ Ioo 0 π → parametric_circle ρ θ)
                 (h_angle : ∀ θ1 θ2 : ℝ, θ2 = θ1 + π / 3 → ∃ M N : ℝ, M = ρ * Real.cos θ1 ∧ N = ρ * Real.cos θ2 ∧ atan2 M N = π / 3)
                 (h_max : ∃ θ : ℝ, θ ∈ Ioo (-π/2) (π/6) ∧ (ρ * Real.cos θ + ρ * Real.cos (θ + π / 3)) = 4 * Real.sqrt 3) : 
                 (∀ θ : ℝ, θ ∈ Ioo (-π/2) (π/6) → ρ * Real.cos θ + ρ * Real.cos (θ + π / 3) ≤ 4 * Real.sqrt 3) :=
  sorry

end find_a_max_OMON_l576_576721


namespace vasya_solved_13_problems_first_9_days_at_least_one_each_day_cloudy_rule_sunny_rule_day_10_cloudy_max_problems_day_15_l576_576970

-- Define the number of problems solved on each day
def problems_solved : ℕ → ℕ := sorry

-- Hypothesis: Over the first 9 days, Vasya solved 13 problems 
def sum_of_first_9_days : ℕ := (0 to 8).sum problems_solved
theorem vasya_solved_13_problems_first_9_days : sum_of_first_9_days = 13 := sorry

-- Hypothesis: At least one problem each day
theorem at_least_one_each_day (i : ℕ) : 1 ≤ problems_solved i := sorry

-- Hypothesis: If cloudy on day i+1, solved one more problem than the previous day
def is_cloudy : ℕ → Prop := sorry

theorem cloudy_rule (i : ℕ) : is_cloudy (i+1) → problems_solved (i+1) = problems_solved i + 1 := sorry

-- Hypothesis: If sunny on day i+1, solved one less problem than the previous day
def is_sunny : ℕ → Prop := sorry

theorem sunny_rule (i : ℕ) : is_sunny (i+1) → problems_solved (i+1) = problems_solved i - 1 := sorry

-- Conclusion for Part (a): The 10th day was cloudy, Vasya solved 2 problems
theorem day_10_cloudy : is_cloudy 10 ∧ problems_solved 10 = 2 := sorry

-- Conclusion for Part (b): The maximum number of problems on Day 15 is 7
theorem max_problems_day_15 : problems_solved 15 ≤ 7 := sorry

end vasya_solved_13_problems_first_9_days_at_least_one_each_day_cloudy_rule_sunny_rule_day_10_cloudy_max_problems_day_15_l576_576970


namespace probability_PA_lt_half_l576_576131

theorem probability_PA_lt_half (P : Type) [point : field P] (ABC_is_equilateral : ∀ (A B C : P), equilateral_triangle A B C 1) :
  let probability := calc_area (set_of (λ p : P, dist p A < 1 / 2)) / calc_area (triangle P 1)
  in probability = (sqrt 3 * pi) / 18 := by sorry

end probability_PA_lt_half_l576_576131


namespace length_of_three_same_length_ropes_l576_576523

theorem length_of_three_same_length_ropes :
  ∃ (x : ℝ), (let r1 := 8 in 
              let r2 := 20 in 
              let r3 := 7 in 
              let knots := 5 * 1.2 in 
              let total_length_after_tying := 35 in 
              35 + 6 = r1 + r2 + r3 + 3 * x
             ) ∧ x = 2 :=
by {
  let r1 := 8,
  let r2 := 20,
  let r3 := 7,
  let knots := 5 * 1.2,
  let total_length_after_tying := 35,
  use 2, 
  simp,
  sorry
}

end length_of_three_same_length_ropes_l576_576523


namespace find_fx0_l576_576117

noncomputable def f (x : ℝ) : ℝ := Real.ln x + 2 * x

def slope_at (x0 : ℝ) : ℝ := (deriv f) x0

theorem find_fx0 : 
  ∃ x0 : ℝ, slope_at x0 = 3 ∧ f x0 = 2 := 
by
  sorry

end find_fx0_l576_576117


namespace length_of_conjugate_axis_l576_576278

-- Define the hyperbola and the given conditions
def hyperbola_eq (x y b : ℝ) : Prop := (x^2 / 4) - (y^2 / b^2) = 1

def distance_focus_asymptote (b : ℝ) : Prop := 
  let focus_distance : ℝ := b * Real.sqrt(4 + b^2) / Real.sqrt(4 + b^2)
  focus_distance = 3

-- Target statement: The length of the conjugate axis of the hyperbola is 6
theorem length_of_conjugate_axis (b : ℝ) (h1 : b > 0) (h2 : distance_focus_asymptote b) : 2 * b = 6 := 
sorry

end length_of_conjugate_axis_l576_576278


namespace probability_of_hitting_target_l576_576607

theorem probability_of_hitting_target:
  (∑ k in set.range(4), if k = 2 then nat.choose 3 k * (0.6 ^ k) * (0.4 ^ (3 - k)) else if k = 3 then nat.choose 3 k * (0.6 ^ k) * (0.4 ^ (3 - k)) else 0) 
  = 81 / 125 :=
by sorry

end probability_of_hitting_target_l576_576607


namespace grid_sums_correct_l576_576395

open Nat

def digitListToInt (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem grid_sums_correct :
  let grid := [[1, 5, 7], [4, 8, 2], [6, 3, 9]] in
  let rowSum (i : ℕ) := digitListToInt (grid.getD i []) in
  let colSum (i : ℕ) := digitListToInt (List.map (fun row => row.getD i 0) grid) in
  rowSum 0 + rowSum 1 = rowSum 2 ∧
  colSum 0 + colSum 1 = colSum 2 :=
by
  sorry

end grid_sums_correct_l576_576395


namespace Pete_vs_Polly_l576_576161

noncomputable def original_price : ℝ := 120
noncomputable def tax_rate : ℝ := 0.08
noncomputable def discount_rate : ℝ := 0.25

noncomputable def Pete_total : ℝ := (original_price * (1 + tax_rate)) * (1 - discount_rate)
noncomputable def Polly_total : ℝ := (original_price * (1 - discount_rate)) * (1 + tax_rate)

theorem Pete_vs_Polly : Pete_total = Polly_total :=
by
  calc
    Pete_total = (original_price * (1 + tax_rate)) * (1 - discount_rate) : by rfl
            ... = (120 * 1.08) * 0.75                      : by rfl
            ... = 129.60 * 0.75                            : by norm_num
            ... = 97.20                                    : by norm_num
            ... = (120 * 0.75) * 1.08                      : by norm_num
            ... = 90.00 * 1.08                             : by norm_num
            ... = 97.20                                    : by norm_num
            ... = Polly_total                              : by rfl

#eval Pete_vs_Polly

end Pete_vs_Polly_l576_576161


namespace proof_problem_l576_576251

noncomputable def a {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def b {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def c {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def d {α : Type*} [LinearOrderedField α] : α := sorry

theorem proof_problem (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
(hprod : a * b * c * d = 1) : 
a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end proof_problem_l576_576251


namespace part_a_part_b_l576_576350

variables {A B C D E K G H I M N P Q O : Type*}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
          [Inhabited K] [Inhabited G] [Inhabited H] [Inhabited I] [Inhabited M]
          [Inhabited N] [Inhabited P] [Inhabited Q] [Inhabited O]

-- Defining the main setup of the problem
structure ProblemSetup :=
  (ABCD_inscribed_in_O : A ∈ O ∧ B ∈ O ∧ C ∈ O ∧ D ∈ O)
  (AB_CD_intersect_at_E : ∃ E, line A B ∧ line C D)
  (AC_BD_intersect_at_K : ∃ K, line A C ∧ line B D)
  (O_not_on_KE : ¬ point_on_line O (line K E))
  (G_H_midpoints : midpoint G A B ∧ midpoint H C D)
  (circumcircle_GKH : exists_circle_through_points G K H I)
  (MN_intersect_in_GH : M ∈ I ∧ N ∈ I ∧ convex_quadrilateral M G H N)
  (P_intersection_MG_HN : intersection P (line M G) (line H N))
  (Q_intersection_MN_GH : intersection Q (line M N) (line G H))

-- Prove that IK is parallel to OE
theorem part_a (setup : ProblemSetup) : parallel (line I K) (line O E) :=
by sorry

-- Prove that PK is perpendicular to IQ
theorem part_b (setup : ProblemSetup) : perpendicular (line P K) (line I Q) :=
by sorry

end part_a_part_b_l576_576350


namespace equal_segments_HE_HF_l576_576114

-- Define the geometric entities and their properties
variables {A B C H D E F : Point}
variable [triangle ABC]

-- Define H as the orthocenter of triangle ABC
axiom H_is_orthocenter : orthocenter H A B C

-- Define D as the midpoint of AC
axiom D_is_midpoint_of_AC : midpoint D A C

-- Define perpendicularity condition: A line through H perpendicular to DH intersects AB and BC at E and F
axiom Line_through_H_perpendicular_DH_intersects_EF : 
  ∃ l : Line, (passing_through H l) ∧ (perpendicular l (line D H)) ∧ (∃ E F, intersection (line A B) l E ∧ intersection (line B C) l F)

-- The problem statement that we need to prove
theorem equal_segments_HE_HF : distance H E = distance H F := 
by
  sorry

end equal_segments_HE_HF_l576_576114


namespace minimize_distance_l576_576697

theorem minimize_distance
  (a b c d : ℝ)
  (h1 : (a + 3 * Real.log a) / b = 1)
  (h2 : (d - 3) / (2 * c) = 1) :
  (a - c)^2 + (b - d)^2 = (9 / 5) * (Real.log (Real.exp 1 / 3))^2 :=
by sorry

end minimize_distance_l576_576697


namespace binomial_expansion_sum_l576_576718

theorem binomial_expansion_sum (n : ℕ) (hn : n = 9) :
  (∑ k in range (n + 1), (binomial n k) * (-2)^k) = -1 :=
by {
  sorry
}

end binomial_expansion_sum_l576_576718


namespace find_extremum_find_common_points_find_k_range_l576_576191

noncomputable def operation (p q : ℝ) (b c : ℝ) : ℝ := - (1 / 3) * (p - c) * (q - b) + 4 * b * c

def f_1 (x c : ℝ) : ℝ := x^2 - 2 * c
def f_2 (x b : ℝ) : ℝ := x - 2 * b
def f (x b c : ℝ) : ℝ := f_1 x c * f_2 x b

theorem find_extremum (b c : ℝ) : b = -1 ∧ c = 3 ↔ has_extremum (f 1 b c) (-4 / 3) 1 :=
sorry

theorem find_common_points (b c : ℝ) (hb : b ≠ 0) :
    (0, b * c) ∈ curve_points ∧ (3 * b, 4 * b * c) ∈ curve_points ∧
    (2 * b, (4 / 3) * b^3 + 3 * b * c) ∈ curve_points ∧
    (-b, (4 / 3) * b^3) ∈ curve_points :=
sorry

theorem find_k_range (k : ℝ) : k ≤ 1 / 2 :=
sorry

end find_extremum_find_common_points_find_k_range_l576_576191


namespace digit_205_of_14_div_360_l576_576529

noncomputable def decimal_expansion_of_fraction (n d : ℕ) : ℕ → ℕ := sorry

theorem digit_205_of_14_div_360 : 
  decimal_expansion_of_fraction 14 360 205 = 8 :=
sorry

end digit_205_of_14_div_360_l576_576529


namespace initial_percentage_of_water_l576_576576

theorem initial_percentage_of_water (P : ℕ) : 
  (P / 100) * 120 + 54 = (3 / 4) * 120 → P = 30 :=
by 
  intro h
  sorry

end initial_percentage_of_water_l576_576576


namespace john_has_remaining_money_l576_576849

-- Definitions of conditions
def johns_initial_money : ℕ := 100
def fraction_given_to_jenna : ℚ := 1/4
def grocery_cost : ℕ := 40

-- Expected Result
def johns_remaining_money (initial : ℕ) (fraction : ℚ) (grocery : ℕ) : ℕ :=
  initial - (initial * fraction).natAbs - grocery

theorem john_has_remaining_money : johns_remaining_money johns_initial_money fraction_given_to_jenna grocery_cost = 35 :=
by
  sorry

end john_has_remaining_money_l576_576849


namespace pow_evaluation_l576_576198

theorem pow_evaluation (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end pow_evaluation_l576_576198


namespace w_share_is_625_l576_576163

noncomputable def w_share (S : ℝ) : ℝ := (25/605) * S

theorem w_share_is_625 {S : ℝ} 
  (h1 : (210/605) * S = 2100)
  (h2 : (60/605) * S = 600)
  (h3 : (25/605) * S + (210/605) * S + (60/605) * S + (160/605) * S + (150/605) * S ≤ 25000) :
  w_share S = 625 :=
begin
  sorry
end

end w_share_is_625_l576_576163


namespace total_dividend_received_l576_576600

noncomputable def investmentAmount : Nat := 14400
noncomputable def faceValue : Nat := 100
noncomputable def premium : Real := 0.20
noncomputable def declaredDividend : Real := 0.07

theorem total_dividend_received :
  let cost_per_share := faceValue * (1 + premium)
  let number_of_shares := investmentAmount / cost_per_share
  let dividend_per_share := faceValue * declaredDividend
  let total_dividend := number_of_shares * dividend_per_share
  total_dividend = 840 := 
by 
  sorry

end total_dividend_received_l576_576600


namespace rectangle_area_l576_576157

theorem rectangle_area (square_area : ℝ) (rectangle_length_ratio : ℝ) (square_area_eq : square_area = 36)
  (rectangle_length_ratio_eq : rectangle_length_ratio = 3) :
  ∃ (rectangle_area : ℝ), rectangle_area = 108 :=
by
  -- Extract the side length of the square from its area
  let side_length := real.sqrt square_area
  have side_length_eq : side_length = 6, from calc
    side_length = real.sqrt 36 : by rw [square_area_eq]
    ... = 6 : real.sqrt_eq 6 (by norm_num)
  -- Calculate the rectangle's width
  let width := side_length
  -- Calculate the rectangle's length
  let length := rectangle_length_ratio * width
  -- Calculate the area of the rectangle
  let area := width * length
  -- Prove the area is 108
  use area
  calc area
    = 6 * (3 * 6) : by rw [side_length_eq, rectangle_length_ratio_eq]
    ... = 108 : by norm_num

end rectangle_area_l576_576157


namespace K9_monochromatic_triangle_l576_576702

-- Conditions: 9 points in space, no four coplanar, complete graph K_9
theorem K9_monochromatic_triangle (V : fin 9) (E : set (V × V))
  (hE : ∀ (v1 v2 : V), v1 ≠ v2 → (v1, v2) ∈ E)
  (colored_edges : E → Prop) :
  (∃ triangle : set (V × V), triangle ⊆ E ∧ 
  (∀ e ∈ triangle, colored_edges e ∧ (∀ e1 e2 ∈ triangle, e1 ≠ e2 → (e1.2 = e2.1) ∨ (e1.1 = e2.2))) ∧
  (∃ color : bool, ∀ e ∈ triangle, colored_edges e = color)) :=
by
  sorry

end K9_monochromatic_triangle_l576_576702


namespace calculate_circumradius_of_triangle_ABC_l576_576526

noncomputable def circumradius (A B C : EuclideanSpace ℝ 3) : ℝ := sorry

theorem calculate_circumradius_of_triangle_ABC
  (A B C : EuclideanSpace ℝ 3)
  (r1 r2 : ℝ)
  (h1 : r1 + r2 = 13)
  (h2 : dist (center_of_sphere_through A r1) (center_of_sphere_through B r2) = sqrt 505)
  (h3 : dist C (tangent_point_sphere A r1) = 8)
  (h4 : dist C (tangent_point_sphere B r2) = 8) :
  circumradius A B C = 2 * sqrt 21 := 
sorry

end calculate_circumradius_of_triangle_ABC_l576_576526


namespace units_digit_of_m_power2_plus_power_of_2_l576_576358

theorem units_digit_of_m_power2_plus_power_of_2 (m : ℕ) (h : m = 2023^2 + 2^2023) : 
  (m^2 + 2^m) % 10 = 7 :=
by
  -- Enter the proof here
  sorry

end units_digit_of_m_power2_plus_power_of_2_l576_576358


namespace constant_term_expansion_l576_576425

/-- The constant term in the expansion of (√x + 1/√x)^4 is 6. -/
theorem constant_term_expansion : 
  (∃ k : ℕ, 
    k = Nat.choose 4 2 ∧ 
    (∀ (x : ℚ), 
      ((√x + 1/√x)^4).nth k = 6)) :=
by
  sorry

end constant_term_expansion_l576_576425


namespace exists_point_P_l576_576349

open Int

-- Define the property of being an integral point
structure Point :=
  (x : Int)
  (y : Int)

-- Define A, B, C as distinct points with integral coordinates
variables (A B C : Point)
(h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)

-- Define a function that checks if the interior of a segment contains integral points
def interior_contains_integral_point (P1 P2 : Point) : Prop := 
  ∃ t : ℚ, 0 < t ∧ t < 1 ∧ ∀ k : ℚ, (t = k → k ∉ ℤ) ∧ ∃ (x' y' : ℤ), 
    x' = (P1.x * (1 - t) + P2.x * t) ∧ y' = (P1.y * (1 - t) + P2.y * t)

-- Prove the existence of the required point P
theorem exists_point_P (A B C : Point) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C):
  ∃ (P : Point), P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ 
  ¬ interior_contains_integral_point P A ∧ 
  ¬ interior_contains_integral_point P B ∧ 
  ¬ interior_contains_integral_point P C :=
sorry

end exists_point_P_l576_576349


namespace contestant_wins_probability_l576_576604

section RadioProgramQuiz
  -- Defining the conditions
  def number_of_questions : ℕ := 4
  def number_of_choices_per_question : ℕ := 3
  def probability_of_correct_answer : ℚ := 1 / 3
  
  -- Defining the target probability
  def winning_probability : ℚ := 1 / 9

  -- The theorem
  theorem contestant_wins_probability :
    (let p := probability_of_correct_answer
     let p_correct_all := p^4
     let p_correct_three :=
       4 * (p^3 * (1 - p))
     p_correct_all + p_correct_three = winning_probability) :=
    sorry
end RadioProgramQuiz

end contestant_wins_probability_l576_576604


namespace triangle_circumcenter_altitudes_l576_576340

theorem triangle_circumcenter_altitudes :
  ∀ (A B C O : Type)
    [metric_space A] [metric_space B]
    [metric_space C] [metric_space O]
    (angle_A : real)
    (AB AC : real)
    (M N H BE CF : A)
    (BM CN MH NH OH : real),
    angle_A = 60 ∧ AB > AC ∧ -- Given conditions
    BE = altitude B ∧ CF = altitude C ∧
    H = intersection BE CF ∧
    M ∈ BH ∧ N ∈ HF ∧
    BM = CN
    → MH + NH = √3 * OH := 
begin
  sorry
end

end triangle_circumcenter_altitudes_l576_576340


namespace min_trees_include_three_types_l576_576820

noncomputable def minNumTrees (T : Type) (tree_counts : T → ℕ) :=
  ∀ (total_trees : ℕ) (at_least : ℕ → Prop),
    total_trees = 100 →
    (∀ S : Finset T, S.card = 85 → (∃ t ∈ S, at_least (tree_counts t))) →
    (at_least ThreeTypes ↔ ∃ (S : Finset T), S.card = 69)

def ThreeTypes (tree_counts : T → ℕ) := 
  ∀ (birches spruces pines aspens : ℕ),
    birches + spruces + pines + aspens = 100 →
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ pines ≥ 1) ∨ 
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ aspens ≥ 1) ∨ 
    (birches ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1) ∨ 
    (spruces ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1)

theorem min_trees_include_three_types :
  ∃ (T : Type) (tree_counts : T → ℕ), minNumTrees T tree_counts := 
sorry

end min_trees_include_three_types_l576_576820


namespace count_possible_starting_days_l576_576603

def month_has_equal_sundays_and_tuesdays (first_day : Nat) : Prop :=
  let extra_days := first_day % 7 in
  let days_with_extra_in_30 := if extra_days = 6 then 5 else 4 in
  days_with_extra_in_30 = 4

theorem count_possible_starting_days : 
  ∃ count, count = 6 ∧ 
    ∀ d, d < 7 → (month_has_equal_sundays_and_tuesdays d ↔ (d ≠ 6)) :=
by
  sorry

end count_possible_starting_days_l576_576603


namespace setA_not_determined_l576_576984

-- Define the set of points more than 1 unit away from the origin
def setB : Set ℝ := {x | x > 1 ∨ x < -1}

-- Define the set of prime numbers less than 100
def setC : Set ℕ := {n | Prime n ∧ n < 100}

-- Define the set of solutions for the quadratic equation x^2 + 2x + 7 = 0
def setD : Set ℝ := {x | x^2 + 2 * x + 7 = 0}

-- The main theorem to prove
theorem setA_not_determined (A B C D : Set α) :
  ¬ (∃ (h : α → Prop), h = λ x, "x is a tall student at Chongqing No.1 Middle School") := 
sorry

end setA_not_determined_l576_576984


namespace minimum_trees_l576_576792

variable (Trees : Type) [Fintype Trees] [DecidableEq Trees]

def trees_in_grove : Nat := 100

def tree_type := {birches, spruces, pines, aspens} : Set Trees

def condition (s : Finset Trees) : Prop := 
  s.card > 85 → tree_type ⊆ s

theorem minimum_trees (s : Finset Trees) (H : condition s) : 
  ∃ (n : Nat), n ≤ trees_in_grove ∧ n ≥ 69 → 
  ∃ t ⊆ s, t.card = n ∧ (|t ∩ tree_type| >= 3) :=
sorry

end minimum_trees_l576_576792


namespace count_divisors_38_412_l576_576751

theorem count_divisors_38_412 : 
  let N := 38412 in
  (finset.filter (λ d, N % d = 0) (finset.range 10)).card = 6 :=
by
  sorry

end count_divisors_38_412_l576_576751


namespace minimum_number_of_troublemakers_l576_576485

theorem minimum_number_of_troublemakers (n m : ℕ) (students : Fin n → Prop) 
  (truthtellers : Fin n → Prop) (liars : Fin n → Prop) (round_table : List (Fin n))
  (h_all_students : n = 29)
  (h_truth_or_liar : ∀ s, students s → (truthtellers s ∨ liars s))
  (h_truth_tellers_conditions : ∀ s, truthtellers s → 
    (count_lies_next_to s = 1 ∨ count_lies_next_to s = 2))
  (h_liars_conditions : ∀ s, liars s → 
    (count_lies_next_to s ≠ 1 ∧ count_lies_next_to s ≠ 2))
  (h_round_table_conditions : round_table.length = n 
    ∧ (∀ i, i < n → round_table.nth i ≠ none)):
  m = 10 :=
by
  sorry

end minimum_number_of_troublemakers_l576_576485


namespace cervical_spine_work_relation_l576_576522

theorem cervical_spine_work_relation :
  let total_population : ℕ := 50 in
  let cervical_spine_prob : ℚ := 3 / 5 in
  let white_collar_no_disease : ℕ := 5 in
  let blue_collar_with_disease : ℕ := 10 in
  let expected_with_disease := total_population * cervical_spine_prob in
  let expected_without_disease := total_population - expected_with_disease in
  let white_collar_with_disease := expected_with_disease - blue_collar_with_disease in
  let blue_collar_no_disease := expected_without_disease - white_collar_no_disease in
  let total_white_collar := white_collar_with_disease + white_collar_no_disease in
  let total_blue_collar := blue_collar_with_disease + blue_collar_no_disease in
  let observed_values := [ [20, 5], [10, 15] ] in -- From the completed table
  let chi_square_value : ℚ := (50 * (20 * 15 - 5 * 10)^2) / (25 * 25 * 30 * 20) in
  chi_square_value > 7.879 := 
    by
      -- Proof will be here
      sorry

end cervical_spine_work_relation_l576_576522


namespace problem_statement_l576_576030

variable {n : ℕ}
variable {x : Fin n → ℝ}

theorem problem_statement (h : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  (∑ i, x i + 1) ^ 2 ≥ 4 * (∑ i, (x i ^ 2)) :=
by {
  sorry
}

end problem_statement_l576_576030


namespace difference_in_mileage_l576_576027

-- Define the conditions
def advertised_mpg : ℝ := 35
def tank_capacity : ℝ := 12
def regular_gasoline_mpg : ℝ := 30
def premium_gasoline_mpg : ℝ := 40
def diesel_mpg : ℝ := 32
def fuel_proportion : ℝ := 1 / 3

-- Define the weighted average function
def weighted_average_mpg (mpg1 mpg2 mpg3 : ℝ) (proportion : ℝ) : ℝ :=
  (mpg1 * proportion) + (mpg2 * proportion) + (mpg3 * proportion)

-- Proof
theorem difference_in_mileage :
  advertised_mpg - weighted_average_mpg regular_gasoline_mpg premium_gasoline_mpg diesel_mpg fuel_proportion = 1 := by
  sorry

end difference_in_mileage_l576_576027


namespace all_roots_non_zero_l576_576650

noncomputable def eq1 : Prop := ∃ x : ℝ, 4 * x^2 - 6 = 34
noncomputable def eq2 : Prop := ∃ x : ℝ, (3 * x - 1)^2 = (x + 2)^2
noncomputable def eq3 : Prop := ∃ x : ℝ, real.sqrt (x^2 - 4) = real.sqrt (x + 3)

theorem all_roots_non_zero :
  (∀ x, 4 * x^2 - 6 = 34 → x ≠ 0) ∧
  (∀ x, (3 * x - 1)^2 = (x + 2)^2 → x ≠ 0) ∧
  (∀ x, real.sqrt (x^2 - 4) = real.sqrt (x + 3) → x ≠ 0) :=
by
  sorry

end all_roots_non_zero_l576_576650


namespace boys_meeting_count_l576_576960

noncomputable def meetings_count (speed1 speed2 speed3 : ℕ) (lap_time : ℕ) : ℕ := 
  (lap_time / (speed1 + speed2)) + 
  (lap_time / (abs (Int.ofNat speed1 - Int.ofNat speed3)))

theorem boys_meeting_count (speed1 speed2 speed3 lap_time : ℕ) :
  speed1 = 6 →
  speed2 = 10 →
  speed3 = 4 →
  lap_time = 60 →
  (meetings_count speed1 speed2 speed3 lap_time = 3) := 
by
  intros h1 h2 h3 h4
  simp [meetings_count, h1, h2, h3, h4]
  sorry

end boys_meeting_count_l576_576960


namespace smiths_bakery_multiple_l576_576409

theorem smiths_bakery_multiple (x : ℤ) (mcgee_pies : ℤ) (smith_pies : ℤ) 
  (h1 : smith_pies = x * mcgee_pies + 6)
  (h2 : mcgee_pies = 16)
  (h3 : smith_pies = 70) : x = 4 :=
by
  sorry

end smiths_bakery_multiple_l576_576409


namespace digit_swap_problem_l576_576615

theorem digit_swap_problem :
  ∃! (ab_pair : ℕ × ℕ), (ab_pair.1 ∈ {1, 2, 3, 4} ∧ ab_pair.2 = 2 * ab_pair.1) :=
begin
  sorry
end

-- Explanation of the goal:
-- We seek to find exactly four (a, b) pairs such that when digits a and b are swapped, the resulting
-- number is 4/7 the original number and fits the form specified in the steps above.

end digit_swap_problem_l576_576615


namespace three_types_in_69_trees_l576_576787

variable (birches spruces pines aspens : ℕ)
variable (total_trees : ℕ := 100)
variable (all_trees : list (string × ℕ))

-- We assert that there are 100 trees in total, and our list of trees represents this
axiom h_total : ∑ t in all_trees, t.2 = total_trees

-- Among any 85 trees, there must be at least one of each type
axiom h_85_trees_all_types : ∀ (s : list (string × ℕ)), s.card = 85 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0 ∧ a > 0 ∧ 
  b + s + p + a = 85 ∧ (("birches", b) ∈ s ∧ ("spruces", s) ∈ s ∧ ("pines", p) ∈ s ∧ ("aspens", a) ∈ s))

-- We need to prove that any subset of 69 or more trees contains at least three different types.
theorem three_types_in_69_trees :
  ∀ (s : list (string × ℕ)), s.card = 69 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0) ∨ (b > 0 ∧ s > 0 ∧ a > 0) ∨ (b > 0 ∧ p > 0 ∧ a > 0) ∨ (s > 0 ∧ p > 0 ∧ a > 0) :=
begin
  sorry
end

end three_types_in_69_trees_l576_576787


namespace hcf_of_12_and_15_l576_576417

-- Definitions of LCM and HCF
def LCM (a b : ℕ) : ℕ := sorry  -- Placeholder for actual LCM definition
def HCF (a b : ℕ) : ℕ := sorry  -- Placeholder for actual HCF definition

theorem hcf_of_12_and_15 :
  LCM 12 15 = 60 → HCF 12 15 = 3 :=
by
  sorry

end hcf_of_12_and_15_l576_576417


namespace AE_bisects_BD_l576_576865

open_locale classical

variables {Γ : Type*} [incidence_plane Γ] 
variables {A B C D E : Γ}
variables (h_circle : circle A B C) (h_AB_BC : dist A B = dist B C)
variables (h_tangents : tangent A D Γ ∧ tangent B D Γ)
variables (h_intersection : ∃ E, line_through D C ∩ circle A B C = {E})

theorem AE_bisects_BD : midpoint A E ∈ segment B D :=
sorry

end AE_bisects_BD_l576_576865


namespace HCF_of_two_numbers_l576_576083

theorem HCF_of_two_numbers (a b : ℕ) (h1 : a * b = 2562) (h2 : Nat.lcm a b = 183) : Nat.gcd a b = 14 := 
by
  sorry

end HCF_of_two_numbers_l576_576083


namespace maryann_time_spent_calling_clients_l576_576378

theorem maryann_time_spent_calling_clients (a c : ℕ) 
  (h1 : a + c = 560) 
  (h2 : a = 7 * c) : c = 70 := 
by 
  sorry

end maryann_time_spent_calling_clients_l576_576378


namespace village_of_Roche_has_group_of_22_with_same_age_l576_576070

theorem village_of_Roche_has_group_of_22_with_same_age :
  ∀ (n : ℕ) (h_n : n = 2020)
    (villagers : Fin n → ℕ)
    (h1 : ∀ i, ∃ j, villagers i = villagers j)
    (h2 : ∀ (s : Finset (Fin n)), s.card = 192 → ∃ i j k, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ villagers i = villagers j ∧ villagers j = villagers k),
  ∃ s : Finset (Fin n), s.card = 22 ∧ ∀ i j ∈ s, villagers i = villagers j :=
by
  sorry

end village_of_Roche_has_group_of_22_with_same_age_l576_576070


namespace part_I_part_II_l576_576064

noncomputable def seq_a : ℕ → ℝ 
| 0       => 1   -- Normally, we start with n = 1, so we set a_0 to some default value.
| (n+1)   => (1 + 1 / (n^2 + n)) * seq_a n + 1 / (2^n)

theorem part_I (n : ℕ) (h: n ≥ 2) : seq_a n ≥ 2 :=
sorry

theorem part_II (n : ℕ) : seq_a n < Real.exp 2 :=
sorry

-- Assumption: ln(1 + x) < x for all x > 0
axiom ln_ineq (x : ℝ) (hx : 0 < x) : Real.log (1 + x) < x

end part_I_part_II_l576_576064


namespace problem_statement_l576_576974

def base9_to_base10 (n : ℕ) : ℕ := 
  8 * 9^0 + 6 * 9^1 + 4 * 9^2 + 2 * 9^3

def base5_to_base10 (n : ℕ) : ℕ := 
  0 * 5^0 + 0 * 5^1 + 2 * 5^2

def base8_to_base10 (n : ℕ) : ℕ := 
  6 * 8^0 + 5 * 8^1 + 4 * 8^2 + 3 * 8^3

def base9_2_to_base10 (n : ℕ) : ℕ := 
  0 * 9^0 + 9 * 9^1 + 8 * 9^2 + 7 * 9^3

theorem problem_statement : 
  (base9_to_base10 2468 / base5_to_base10 200 - base8_to_base10 3456 + base9_2_to_base10 7890).toInt.floordiv 1 = 4030 :=
by
  sorry

end problem_statement_l576_576974


namespace loss_of_30_yuan_is_minus_30_yuan_l576_576305

def profit (p : ℤ) : Prop := p = 20
def loss (l : ℤ) : Prop := l = -30

theorem loss_of_30_yuan_is_minus_30_yuan (p : ℤ) (l : ℤ) (h : profit p) : loss l :=
by
  sorry

end loss_of_30_yuan_is_minus_30_yuan_l576_576305


namespace find_b_vector_l576_576858

-- Definitions of vectors a and b
def a : ℝ^3 := ![2, 1, 5]
def b : ℝ^3 := ![-1, 3, 2]

-- Conditions given in the problem
def dot_product_condition : Prop := (a ⬝ b) = 11
def cross_product_condition : Prop := (a × b) = ![-13, -9, 7]

-- The theorem that needs to be proven
theorem find_b_vector : dot_product_condition ∧ cross_product_condition → b = ![-1, 3, 2] :=
by
  sorry

end find_b_vector_l576_576858


namespace find_f_zero_l576_576735

theorem find_f_zero : 
  ∀ (f : ℤ → ℤ), 
  (∀ x : ℤ, f (x - 2) = 2^x - x + 3) → 
  f 0 = 5 :=
by
  intros f h
  sorry

end find_f_zero_l576_576735


namespace min_trees_include_three_types_l576_576821

noncomputable def minNumTrees (T : Type) (tree_counts : T → ℕ) :=
  ∀ (total_trees : ℕ) (at_least : ℕ → Prop),
    total_trees = 100 →
    (∀ S : Finset T, S.card = 85 → (∃ t ∈ S, at_least (tree_counts t))) →
    (at_least ThreeTypes ↔ ∃ (S : Finset T), S.card = 69)

def ThreeTypes (tree_counts : T → ℕ) := 
  ∀ (birches spruces pines aspens : ℕ),
    birches + spruces + pines + aspens = 100 →
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ pines ≥ 1) ∨ 
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ aspens ≥ 1) ∨ 
    (birches ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1) ∨ 
    (spruces ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1)

theorem min_trees_include_three_types :
  ∃ (T : Type) (tree_counts : T → ℕ), minNumTrees T tree_counts := 
sorry

end min_trees_include_three_types_l576_576821


namespace correct_formula_l576_576921

-- Given conditions
def table : List (ℕ × ℕ) := [(2, 0), (3, 2), (4, 6), (5, 12), (6, 20)]

-- Candidate formulas
def formulaA (x : ℕ) : ℕ := 2 * x - 4
def formulaB (x : ℕ) : ℕ := x^2 - 3 * x + 2
def formulaC (x : ℕ) : ℕ := x^3 - 3 * x^2 + 2 * x
def formulaD (x : ℕ) : ℕ := x^2 - 4 * x
def formulaE (x : ℕ) : ℕ := x^2 - 4

-- The statement to be proven
theorem correct_formula : ∀ (x y : ℕ), (x, y) ∈ table → y = formulaB x :=
by
  sorry

end correct_formula_l576_576921


namespace min_value_of_f_inequality_with_conditions_l576_576277

-- Definition of f
def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (5 - x)

-- Minimum value of f
theorem min_value_of_f : ∃ m, m = (9:ℝ) / 2 ∧ ∀ x, f(x) ≥ m :=
by
  let m := (9:ℝ) / 2
  use m
  split
  . rfl -- This establishes that the value is indeed 9/2
  . sorry -- Here we would prove the actual inequality for all x

-- Given constraints, prove the inequality
theorem inequality_with_conditions (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 3) :
  1 / (a + 1) + 1 / (b + 2) ≥ 2 / 3 :=
by
  have h_sum : (a + 1) + (b + 2) = 6 := by
    linarith -- Simplify the condition
  sorry -- Here we would complete the proof using properties of reciprocals and inequalities

end min_value_of_f_inequality_with_conditions_l576_576277


namespace min_trees_include_three_types_l576_576817

noncomputable def minNumTrees (T : Type) (tree_counts : T → ℕ) :=
  ∀ (total_trees : ℕ) (at_least : ℕ → Prop),
    total_trees = 100 →
    (∀ S : Finset T, S.card = 85 → (∃ t ∈ S, at_least (tree_counts t))) →
    (at_least ThreeTypes ↔ ∃ (S : Finset T), S.card = 69)

def ThreeTypes (tree_counts : T → ℕ) := 
  ∀ (birches spruces pines aspens : ℕ),
    birches + spruces + pines + aspens = 100 →
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ pines ≥ 1) ∨ 
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ aspens ≥ 1) ∨ 
    (birches ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1) ∨ 
    (spruces ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1)

theorem min_trees_include_three_types :
  ∃ (T : Type) (tree_counts : T → ℕ), minNumTrees T tree_counts := 
sorry

end min_trees_include_three_types_l576_576817


namespace basketball_team_count_l576_576183

theorem basketball_team_count :
  (∃ n : ℕ, n = (Nat.choose 13 4) ∧ n = 715) :=
by
  sorry

end basketball_team_count_l576_576183


namespace number_of_students_l576_576296

theorem number_of_students (groups : ℕ) (students_per_group : ℕ) (minutes_per_student : ℕ) (minutes_per_group : ℕ) :
    groups = 3 →
    minutes_per_student = 4 →
    minutes_per_group = 24 →
    minutes_per_group = students_per_group * minutes_per_student →
    18 = groups * students_per_group :=
by
  intros h_groups h_minutes_per_student h_minutes_per_group h_relation
  sorry

end number_of_students_l576_576296


namespace find_extrema_of_S_l576_576234

theorem find_extrema_of_S (x y z : ℚ) (h1 : 3 * x + 2 * y + z = 5) (h2 : x + y - z = 2) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  2 ≤ 2 * x + y - z ∧ 2 * x + y - z ≤ 3 :=
by
  sorry

end find_extrema_of_S_l576_576234


namespace general_term_formula_sum_of_sequence_l576_576283

-- Problem 1
theorem general_term_formula (a : ℕ → ℕ) (S : ℕ → ℕ):
  (a 1 = 2) ∧ (a 2 = 4) ∧ (∀ n > 1, S (n + 1) + S (n - 1) = 2 * (S n + 1)) →
  ∀ n : ℕ, n ≥ 1 → a n = 2 * n :=
sorry

-- Problem 2
theorem sum_of_sequence (a : ℕ → ℕ) (b : ℕ → ℝ):
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n) →
  (∀ n : ℕ, b n = n / (4:ℝ)^a n) →
  (∀ T : ℕ → ℝ, T n = ∑ i in finset.range n, b (i + 1)) →
  ∀ n : ℕ, T n = (4 / 9) - (4 + n) / (9 * (4:ℝ)^n) :=
sorry

end general_term_formula_sum_of_sequence_l576_576283


namespace range_of_a_l576_576374

def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x - (Real.log x) / x + a

theorem range_of_a (a : ℝ) :
  (∃ x > 0, f x a = 0) → a ∈ set.Iic (Real.exp 2 + 1 / Real.exp 1) :=
by
  sorry

end range_of_a_l576_576374


namespace collinear_A0_B0_C0_l576_576844

-- Define the scenario described in the problem
variables {A B C M : Type*}

-- Assume necessary properties of the triangle and its orthocenter
variables [Triangle ABC] (orthocenter M ABC)

-- Define the line non-parallel to any sides of the triangle
variable (e : Line)
variable (h_parallel : ¬e ∥ (line AC) ∧ ¬e ∥ (line AB) ∧ ¬e ∥ (line BC))

-- Define the points A0, B0, C0 per the construction in the problem
variables (A0 B0 C0 : Point)
variables [DrawParallelsThroughVertices A B C e A0 B0 C0]

-- Prove that A0, B0, and C0 are collinear
theorem collinear_A0_B0_C0 : collinear {A0, B0, C0} :=
sorry

end collinear_A0_B0_C0_l576_576844


namespace extremum_at_x1_l576_576273

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_at_x1 (a b : ℝ) (h1 : (3*1^2 + 2*a*1 + b) = 0) (h2 : 1^3 + a*1^2 + b*1 + a^2 = 10) :
  a = 4 :=
by
  sorry

end extremum_at_x1_l576_576273


namespace range_of_a_for_two_extreme_points_l576_576271

def f (x : ℝ) (a : ℝ) : ℝ := log x + (1 / 2) * a * x^2 - 2 * x

theorem range_of_a_for_two_extreme_points :
  (∀ x : ℝ, x > 0 → ∃ a : ℝ, 0 < a ∧ a < 1) → 
  (∃ (a : ℝ), 0 < a ∧ a < 1) :=
begin
  sorry -- proof to be provided
end

end range_of_a_for_two_extreme_points_l576_576271


namespace packaging_combinations_l576_576122

theorem packaging_combinations :
  ∀ (types_of_wrapping_papers types_of_ribbons types_of_cards types_of_boxes : ℕ),
    types_of_wrapping_papers = 10 →
    types_of_ribbons = 3 →
    types_of_cards = 4 →
    types_of_boxes = 5 →
    types_of_wrapping_papers * types_of_ribbons * types_of_cards * types_of_boxes = 600 :=
by
  intros types_of_wrapping_papers types_of_ribbons types_of_cards types_of_boxes
  intros h_wrapping h_ribbon h_card h_box
  simp [h_wrapping, h_ribbon, h_card, h_box]
  sorry

end packaging_combinations_l576_576122


namespace log_inverse_eq_two_l576_576758

theorem log_inverse_eq_two (x : ℝ) (h : log 25 (x - 4) = 1 / 2) : 1 / log x 3 = 2 :=
by
  sorry

end log_inverse_eq_two_l576_576758


namespace print_time_ratio_l576_576552

-- Defining the hours each printer takes to complete the job
def hours_x : ℝ := 12
def hours_y : ℝ := 10
def hours_z : ℝ := 20
def hours_w : ℝ := 15

-- Defining the rates at which each printer works
def rate_x : ℝ := 1 / hours_x
def rate_y : ℝ := 1 / hours_y
def rate_z : ℝ := 1 / hours_z
def rate_w : ℝ := 1 / hours_w

-- Combined rate of printers y, z, and w
def combined_rate_yzw : ℝ := rate_y + rate_z + rate_w

-- Time taken by printers y, z, and w to complete the job together
def time_yzw : ℝ := 1 / combined_rate_yzw

-- The ratio of the time taken by printer x to the time taken by printers y, z, and w together
def time_ratio : ℝ := hours_x / time_yzw

theorem print_time_ratio :
  time_ratio = 2.6 := by
  sorry

end print_time_ratio_l576_576552


namespace difference_between_advertised_and_actual_mileage_l576_576026

def advertised_mileage : ℕ := 35

def city_mileage_regular : ℕ := 30
def highway_mileage_premium : ℕ := 40
def traffic_mileage_diesel : ℕ := 32

def gallons_regular : ℕ := 4
def gallons_premium : ℕ := 4
def gallons_diesel : ℕ := 4

def total_miles_driven : ℕ :=
  (gallons_regular * city_mileage_regular) + 
  (gallons_premium * highway_mileage_premium) + 
  (gallons_diesel * traffic_mileage_diesel)

def total_gallons_used : ℕ :=
  gallons_regular + gallons_premium + gallons_diesel

def weighted_average_mpg : ℤ :=
  total_miles_driven / total_gallons_used

theorem difference_between_advertised_and_actual_mileage :
  advertised_mileage - weighted_average_mpg = 1 :=
by
  -- proof to be filled in
  sorry

end difference_between_advertised_and_actual_mileage_l576_576026


namespace exist_a_b_not_triangle_l576_576197

theorem exist_a_b_not_triangle (h₁ : ∀ a b : ℕ, (a > 1000) → (b > 1000) →
  ∃ c : ℕ, (∃ (k : ℕ), c = k * k) →
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  ∃ (a b : ℕ), (a > 1000 ∧ b > 1000) ∧ 
  ∀ c : ℕ, (∃ (k : ℕ), c = k * k) →
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
sorry

end exist_a_b_not_triangle_l576_576197


namespace sum_of_cube_faces_l576_576408

theorem sum_of_cube_faces (a d b e c f : ℕ) (h1: a > 0) (h2: d > 0) (h3: b > 0) (h4: e > 0) (h5: c > 0) (h6: f > 0)
(h7 : a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 1491) :
  a + d + b + e + c + f = 41 := 
sorry

end sum_of_cube_faces_l576_576408


namespace three_types_in_69_trees_l576_576788

variable (birches spruces pines aspens : ℕ)
variable (total_trees : ℕ := 100)
variable (all_trees : list (string × ℕ))

-- We assert that there are 100 trees in total, and our list of trees represents this
axiom h_total : ∑ t in all_trees, t.2 = total_trees

-- Among any 85 trees, there must be at least one of each type
axiom h_85_trees_all_types : ∀ (s : list (string × ℕ)), s.card = 85 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0 ∧ a > 0 ∧ 
  b + s + p + a = 85 ∧ (("birches", b) ∈ s ∧ ("spruces", s) ∈ s ∧ ("pines", p) ∈ s ∧ ("aspens", a) ∈ s))

-- We need to prove that any subset of 69 or more trees contains at least three different types.
theorem three_types_in_69_trees :
  ∀ (s : list (string × ℕ)), s.card = 69 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0) ∨ (b > 0 ∧ s > 0 ∧ a > 0) ∨ (b > 0 ∧ p > 0 ∧ a > 0) ∨ (s > 0 ∧ p > 0 ∧ a > 0) :=
begin
  sorry
end

end three_types_in_69_trees_l576_576788


namespace complex_product_identity_l576_576298

def Q := 3 + 4 * Complex.i
def E := 2 * Complex.i
def D := 3 - 4 * Complex.i

theorem complex_product_identity : 2 * Q * E * D = 100 * Complex.i := by
  sorry

end complex_product_identity_l576_576298


namespace cost_of_450_candies_l576_576583

theorem cost_of_450_candies (box_cost : ℝ) (box_candies : ℕ) (total_candies : ℕ) 
  (h1 : box_cost = 7.50) (h2 : box_candies = 30) (h3 : total_candies = 450) : 
  (total_candies / box_candies) * box_cost = 112.50 :=
by
  sorry

end cost_of_450_candies_l576_576583


namespace problem_l576_576253

theorem problem (x : ℝ) (h : x + 1/x = 10) :
  (x^2 + 1/x^2 = 98) ∧ (x^3 + 1/x^3 = 970) :=
by
  sorry

end problem_l576_576253


namespace determine_xyz_l576_576194

variables {x y z : ℝ}

theorem determine_xyz (h : (x - y - 3)^2 + (y - z)^2 + (x - z)^2 = 3) : 
  x = z + 1 ∧ y = z - 1 := 
sorry

end determine_xyz_l576_576194


namespace product_of_roots_l576_576216

theorem product_of_roots :
  let p1 := 3 * X^3 - 2 * X^2 + 9 * X - 15
  let p2 := 4 * X^4 + 3 * X^3 - 10 * X^2 + 7
  let p := p1 * p2
  let product_of_roots := (-constant_term p) / (leading_coefficient p)
  product_of_roots = 35 / 4 := 
by 
  sorry

end product_of_roots_l576_576216


namespace food_donation_problem_l576_576226

theorem food_donation_problem :
  let foster_chickens := 45
  let american_summits_water := 2 * foster_chickens
  let hormel_chickens := 3 * foster_chickens
  let boudin_chickens := hormel_chickens / 3
  let delmonte_water := american_summits_water - 30
  let total_boudin_delmonte := boudin_chickens + delmonte_water
  let total_donations := foster_chickens + american_summits_water + hormel_chickens + boudin_chickens + delmonte_water
  (total_boudin_delmonte % 7 = 0) ∧
  (foster_chickens = (hormel_chickens + boudin_chickens) / 2)
  → total_donations = 375 :=
by
  let foster_chickens := 45
  let american_summits_water := 2 * foster_chickens
  let hormel_chickens := 3 * foster_chickens
  let boudin_chickens := hormel_chickens / 3
  let delmonte_water := american_summits_water - 30
  let total_boudin_delmonte := boudin_chickens + delmonte_water
  let total_donations := foster_chickens + american_summits_water + hormel_chickens + boudin_chickens + delmonte_water
  have hboudin_delmonte : total_boudin_delmonte % 7 = 0 := by sorry
  have hfoster_eq : foster_chickens = (hormel_chickens + boudin_chickens) / 2 := by sorry
  exact 375

end food_donation_problem_l576_576226


namespace least_number_to_add_l576_576924

theorem least_number_to_add (x : ℕ) (h1 : (1789 + x) % 6 = 0) (h2 : (1789 + x) % 4 = 0) (h3 : (1789 + x) % 3 = 0) : x = 7 := 
sorry

end least_number_to_add_l576_576924


namespace colin_average_time_per_mile_l576_576643

theorem colin_average_time_per_mile :
  (let first_mile := 6
   let second_mile := 5
   let third_mile := 5
   let fourth_mile := 4
   let total_miles := 4
   let total_time := first_mile + second_mile + third_mile + fourth_mile
   let average_time := total_time / total_miles
   average_time = 5) :=
begin
  let first_mile := 6,
  let second_mile := 5,
  let third_mile := 5,
  let fourth_mile := 4,
  let total_miles := 4,
  let total_time := first_mile + second_mile + third_mile + fourth_mile,
  let average_time := total_time / total_miles,
  show average_time = 5,
  sorry,
end

end colin_average_time_per_mile_l576_576643


namespace chess_tournament_participants_l576_576996

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 120) : n = 16 :=
sorry

end chess_tournament_participants_l576_576996


namespace kolya_cannot_descend_without_repeating_l576_576598

theorem kolya_cannot_descend_without_repeating (s : ℕ) : ∀ (s : ℕ), s = 100 →
  ∀ (jump_sizes : list ℕ), jump_sizes = [6, 7, 8] →
  ∀ (seq : ℕ → ℕ), 
    (∀ n, seq 0 = s ∧ (∀ n, seq (n + 1) = seq n - (jump_sizes.nth (n % 3)).get_or_else 0)) →
    ∀ m n, m ≠ n → seq m ≠ seq n :=
begin
  -- To be filled with a proof in Lean
  sorry
end

end kolya_cannot_descend_without_repeating_l576_576598


namespace grace_marks_per_student_l576_576047

theorem grace_marks_per_student (n : ℕ) (old_avg new_avg g : ℕ) (h_n : n = 35) (h_old_avg : old_avg = 37) (h_new_avg : new_avg = 40) :
  g = (n * new_avg - n * old_avg) / n := by
  have h1 : n * old_avg = 35 * 37 := by rw [h_n, h_old_avg]
  have h2 : n * new_avg = 35 * 40 := by rw [h_n, h_new_avg]
  have h3 : g = (35 * 40 - 35 * 37) / 35 := by rw [h1, h2]
  rw [h_n, h_old_avg, h_new_avg] at h3
  exact h3

end grace_marks_per_student_l576_576047


namespace AE_length_l576_576051

-- Definition of points A, B, C, D, and E
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (7, 0)
def D : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (5, 3)

noncomputable def E : ℝ × ℝ :=
  let x := ((A.1 * (B.2 - D.2) - A.2 * (B.1 - D.1) - D.1 * C.2 + D.2 * C.1) /
           ((B.2 - D.2) - (A.2 * (B.1 - D.1) / (A.1 - B.1)))) in
  let y := (B.2 - D.2) / (B.1 - D.1) * (x - A.1) + A.2 in
  (x, y)

-- Function to compute the Euclidean distance between two points
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Statement to be proved
theorem AE_length :
  euclidean_distance A E = (4 * sqrt 65) / 7 :=
  sorry

end AE_length_l576_576051


namespace p_completes_alone_in_80_days_l576_576087

theorem p_completes_alone_in_80_days:
  ∀ (W Rp Rq : ℕ),
  (Rq = W / 48) →
  (16 * Rp + 24 * Rp + 24 * Rq = W) →
  (40 * Rp = W / 2) →
  (Dp = 40 * 2) →
  Dp = 80 :=
begin
  intros W Rp Rq Rq_def work_eq rate_eq Dp_eq,
  subst Rq_def,
  subst work_eq,
  subst rate_eq,
  subst Dp_eq,
  exact rfl,
end

end p_completes_alone_in_80_days_l576_576087


namespace eq_a_n_compare_T_n_S_n_eq_sum_c_n_l576_576720

variable {f : ℝ → ℝ}

-- Define that f satisfies the equation f(x) + f(1-x) = 1/2 for any x in ℝ
axiom f_property : ∀ x : ℝ, f(x) + f(1-x) = 1/2

-- Define sequence a_n
def a_n (n : ℕ) : ℝ := 
  (1 / n) * (Finset.sum (Finset.range n) (λ k, f ((k : ℕ) / n))) + f 1

-- Define the equation to prove in 1)
theorem eq_a_n (n : ℕ) : a_n n = (n + 1) / 4 := sorry

-- Define sequence b_n based on a_n
def b_n (n : ℕ) : ℝ := 4 / (4 * a_n n - 1)

-- Define the sum T_n based on b_n
def T_n (n : ℕ) : ℝ := Finset.sum (Finset.range n) (λ k, (b_n (k + 1)) ^ 2)

-- Define S_n
def S_n (n : ℕ) : ℝ := 32 - (16 / n)

-- Define the comparison theorem for 2)
theorem compare_T_n_S_n (n : ℕ) : T_n n ≥ S_n n := sorry

-- Define sequence c_n based on b_n and q
variable {q : ℝ}
def c_n (n : ℕ) : ℝ := b_n n * q ^ (n - 1)

-- Define the sum of the first n terms of c_n
def sum_c_n (n : ℕ) : ℝ :=
  if q = 1 then
    n * (n + 1) / 2
  else
    (1 - q ^ n) / (1 - q) - (1 + n - n * q) * q ^ n / (1 - q) ^ 2

-- Define the theorem for the sum of c_n in 3)
theorem eq_sum_c_n (n : ℕ) : 
  (Finset.sum (Finset.range n) (λ k, c_n (k + 1))) = sum_c_n n := sorry

end eq_a_n_compare_T_n_S_n_eq_sum_c_n_l576_576720


namespace minimum_liars_l576_576479

noncomputable def minimum_troublemakers (n : ℕ) := 10

theorem minimum_liars (n : ℕ) (students : ℕ → bool) 
  (seated_at_round_table : ∀ i, students (i % 29) = tt ∨ students (i % 29) = ff)
  (one_liar_next_to : ∀ i, students i = tt → (students ((i + 1) % 29) = ff ∨ students ((i + 28) % 29) = ff))
  (two_liars_next_to : ∀ i, students i = ff → ((students ((i + 1) % 29) = ff) ∧ students ((i + 28) % 29) = ff)) :
  ∃ m, m = minimum_troublemakers 29 ∧ (m ≤ n) :=
by
  let liars_count := 10
  have : liars_count = 10 := rfl
  use liars_count
  exact ⟨rfl, sorry⟩

end minimum_liars_l576_576479


namespace find_width_of_garden_l576_576044

noncomputable theory

def width_of_garden (Area Length : ℝ) : ℝ := Area / Length

theorem find_width_of_garden :
  width_of_garden 48.6 5.4 = 9 := by
  sorry

end find_width_of_garden_l576_576044


namespace rectangle_area_from_square_l576_576148

theorem rectangle_area_from_square {s a : ℝ} (h1 : s^2 = 36) (h2 : a = 3 * s) :
    s * a = 108 :=
by
  -- The proof goes here
  sorry

end rectangle_area_from_square_l576_576148


namespace y_difference_positive_l576_576712

theorem y_difference_positive (a c y1 y2 : ℝ) (h1 : a < 0)
  (h2 : y1 = a * 1^2 + 2 * a * 1 + c)
  (h3 : y2 = a * 2^2 + 2 * a * 2 + c) : y1 - y2 > 0 := 
sorry

end y_difference_positive_l576_576712


namespace knights_in_company_l576_576324

theorem knights_in_company :
  ∃ k : ℕ, (k = 0 ∨ k = 6) ∧ k ≤ 39 ∧
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 39) →
    (∃ i : ℕ, (1 ≤ i ∧ i ≤ 39) ∧ n * k = 1 + (i - 1) * k) →
    ∃ i : ℕ, ∃ nk : ℕ, (nk = i * k ∧ nk ≤ 39 ∧ (nk ∣ k → i = 1 + (i - 1))) :=
by
  sorry

end knights_in_company_l576_576324


namespace math_problem_l576_576168

-- Define the conditions for the ellipses and circle
def ellipse_c1 (x y a b : ℝ) := (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1
def circle_c2 (x y : ℝ) := (x^2 + y^2 = 4)

-- Define known properties
def major_axis_eq_diameter (a : ℝ) := 2 * a = 4
def eccentricity (a c : ℝ) := c / a = 1 / 2
def b_value (a c : ℝ) := c = 1 → b = Real.sqrt (a^2 - c^2)

-- Define the given points and lines
def perpendicular_lines (l1 l2 : ℝ → ℝ) (M : ℝ × ℝ) := l1 1 = 0 ∧ l2 M.1 = 0 ∧ ∀ x, l1 x = -(l2 x)

-- Define slope of line l1
def slope_of_l1_is_k (k : ℝ) := k > 0

-- Define the area of quadrilateral and the implication for the slope
def area_quad (a b c d : ℝ × ℝ) := 1/2 * (a.1 * b.2 + b.1 * c.2 + c.1 * d.2 + d.1 * a.2 - a.2 * b.1 - b.2 * c.1 - c.2 * d.1 - d.2 * a.1) = (12 : ℝ) / 7 * Real.sqrt 14 

theorem math_problem :
  ∀ (a b c : ℝ) (l1 l2 : ℝ → ℝ) (k : ℝ) (M A B C D : ℝ × ℝ),
  major_axis_eq_diameter a →
  eccentricity a c →
  b_value a c →
  ellipse_c1 M.1 M.2 a b →
  circle_c2 M.1 M.2 →
  perpendicular_lines l1 l2 M →
  (slope_of_l1_is_k k ∧ 
   area_quad A B C D →
   k = 1) := 
by 
  sorry

end math_problem_l576_576168


namespace determine_a_values_l576_576306

theorem determine_a_values (a : ℝ) (A : Set ℝ) (B : Set ℝ)
  (hA : A = { x | abs x = 1 }) 
  (hB : B = { x | a * x = 1 }) 
  (h_superset : A ⊇ B) :
  a = -1 ∨ a = 0 ∨ a = 1 :=
sorry

end determine_a_values_l576_576306


namespace range_of_quadratic_l576_576939

noncomputable def quadratic (x : ℝ) : ℝ := x^2 - 4*x + 1

theorem range_of_quadratic : 
  set.image quadratic (set.Icc 2 5) = set.Icc (-3) 6 :=
by 
  sorry

end range_of_quadratic_l576_576939


namespace angle_bisectors_l576_576554

/- Define the lines -/
def line1 (x y : ℝ) : Prop := 4 * x - 3 * y = 7
def line2 (x y : ℝ) : Prop := 12 * x + 5 * y = 7

/- Define a function that checks if a line is the angle bisector of two given lines -/
def is_angle_bisector (bisector : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, line1 x y ∨ line2 x y → bisector x y

/- Prove the given equations are the angle bisectors -/
theorem angle_bisectors :
  is_angle_bisector (λ x y, 8 * x - y = 9) ∧ is_angle_bisector (λ x y, x + 8 * y = 7) :=
by
  sorry

end angle_bisectors_l576_576554


namespace rectangle_area_from_square_l576_576147

theorem rectangle_area_from_square {s a : ℝ} (h1 : s^2 = 36) (h2 : a = 3 * s) :
    s * a = 108 :=
by
  -- The proof goes here
  sorry

end rectangle_area_from_square_l576_576147


namespace scalene_polygon_exists_l576_576884

theorem scalene_polygon_exists (n: ℕ) (a: Fin n → ℝ) (h: ∀ i, 1 ≤ a i ∧ a i ≤ 2013) (h_geq: n ≥ 13):
  ∃ (A B C : Fin n), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ a A + a B > a C ∧ a A + a C > a B ∧ a B + a C > a A :=
sorry

end scalene_polygon_exists_l576_576884


namespace base_h_equation_l576_576686

theorem base_h_equation (h : ℕ) : 
  (5 * h^3 + 7 * h^2 + 3 * h + 4) + (6 * h^3 + 4 * h^2 + 2 * h + 1) = 
  1 * h^4 + 4 * h^3 + 1 * h^2 + 5 * h + 5 → 
  h = 10 := 
sorry

end base_h_equation_l576_576686


namespace remainder_f_600_mod_3_l576_576366

def f : ℕ → ℕ
| 0     := 3
| 1     := 5
| (n+2) := f(n+1) + f(n)

theorem remainder_f_600_mod_3 : (f 599) % 3 = 2 :=
sorry

end remainder_f_600_mod_3_l576_576366


namespace initial_white_lights_eq_51_l576_576057

/-
The lights in Malcolm's house are flickering, and he hopes that replacing all of his white lights with colored lights will make it stop. He decides to buy 16 red lights, 4 yellow lights and twice as many blue lights as yellow lights. He also buys 8 green lights and 3 purple lights. Finally, Malcolm realizes he still needs to buy 25% more blue lights and 10 more red lights to fully replace all the white lights. Prove that the total number of white lights that Malcolm had initially is equal to 51.
-/
theorem initial_white_lights_eq_51 : 
  let red_initial := 16 in
  let yellow := 4 in
  let blue_initial := 2 * yellow in
  let green := 8 in
  let purple := 3 in
  let total_initial := red_initial + yellow + blue_initial + green + purple in
  let blue_additional := 0.25 * blue_initial in
  let red_additional := 10 in
  let total_blue := blue_initial + blue_additional in
  let total_red := red_initial + red_additional in
  let total_final := total_red + yellow + total_blue + green + purple in
  total_final = 51 :=
by {
  sorry
}

end initial_white_lights_eq_51_l576_576057


namespace sum_of_valid_two_digit_integers_is_zero_l576_576979

noncomputable theory
open_locale classical

def two_digit_integer (x : ℕ) : Prop :=
  10 ≤ x ∧ x < 100

def sum_of_digits (x : ℕ) : ℕ :=
  let a := x / 10 in
  let b := x % 10 in
  a + b

def product_of_digits (x : ℕ) : ℕ :=
  let a := x / 10 in
  let b := x % 10 in
  a * b

def divisible_by (x y : ℕ) : Prop :=
  y ∣ x

def valid_two_digit_integer (x : ℕ) : Prop :=
  two_digit_integer x ∧
    divisible_by x (sum_of_digits x + 1) ∧
    divisible_by x (product_of_digits x + 1)

theorem sum_of_valid_two_digit_integers_is_zero :
  ∑ x in finset.filter valid_two_digit_integer (finset.range 100), x = 0 :=
by
  sorry

end sum_of_valid_two_digit_integers_is_zero_l576_576979


namespace amount_of_second_alloy_used_l576_576330

variable (x : ℝ)

-- Conditions
def chromium_in_first_alloy : ℝ := 0.10 * 15
def chromium_in_second_alloy (x : ℝ) : ℝ := 0.06 * x
def total_weight (x : ℝ) : ℝ := 15 + x
def chromium_in_third_alloy (x : ℝ) : ℝ := 0.072 * (15 + x)

-- Proof statement
theorem amount_of_second_alloy_used :
  1.5 + 0.06 * x = 0.072 * (15 + x) → x = 35 := by
  sorry

end amount_of_second_alloy_used_l576_576330


namespace determinant_of_2x2_matrix_l576_576645

theorem determinant_of_2x2_matrix (a b c d : ℝ) (h_a : a = 7) (h_b : b = -2) (h_c : c = -3) (h_d : d = 6) :
  determinant ![![a, b], ![c, d]] = 36 :=
by 
  rw [h_a, h_b, h_c, h_d]
  dsimp
  norm_num
  sorry

end determinant_of_2x2_matrix_l576_576645


namespace min_troublemakers_l576_576453

theorem min_troublemakers (students : Finset ℕ) (table : List ℕ) (is_liar : ℕ → Bool) :
  students.card = 29 ∧
  (∀ s ∈ students, 
    (∃ i j : ℕ, 
      table.nth i = some s ∧ table.nth ((i + 1) % 29) = some j ∧
      (∃ k : ℕ, 
        (is_liar s = false → count (/=is_liar k liar) = 0 ∧
        is_liar s = true → count (/=is_liar k liar) = 2))) → 
  (∃! l ∈ students, count (/=is_liar l liar) = 1))) → 
  ∃ t : ℕ, t = 10 :=
sorry

end min_troublemakers_l576_576453


namespace scientific_notation_n_l576_576553

theorem scientific_notation_n (a n : ℝ) (h1 : 0.0813 = a * 10^n) (h2 : 1 ≤ a) (h3 : a < 10) (h4 : n ∈ Int) : n = -2 :=
sorry

end scientific_notation_n_l576_576553


namespace year_2024_AD_representation_l576_576634

def year_representation (y: Int) : Int :=
  if y > 0 then y else -y

theorem year_2024_AD_representation : year_representation 2024 = 2024 :=
by sorry

end year_2024_AD_representation_l576_576634


namespace alice_wins_with_optimal_strategy_l576_576167

theorem alice_wins_with_optimal_strategy :
  (∀ (N : ℕ) (X Y : ℕ), N = 270000 → N = X * Y → gcd X Y ≠ 1 → 
    (∃ (alice : ℕ → ℕ → Prop), ∀ N, ∃ (X Y : ℕ), alice N (X * Y) → gcd X Y ≠ 1) ∧
    (∀ (bob : ℕ → ℕ → ℕ → Prop), ∀ N X Y, bob N X Y → gcd X Y ≠ 1)) →
  (N : ℕ) → N = 270000 → gcd N 1 ≠ 1 :=
by
  sorry

end alice_wins_with_optimal_strategy_l576_576167


namespace even_function_properties_l576_576621

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem even_function_properties
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h1 : ∀ x, f (x + 1) = -f x)
  (h2 : ∀ x (hx : -1 ≤ x ∧ x ≤ 0), f x ≤ f (x + 1)) :
  (\Sigma i in (1 : set ℕ), i = 1) ∧
  (\Sigma i in (4 : set ℕ), i = 4) :=
sorry

end even_function_properties_l576_576621


namespace average_speed_boat_still_water_l576_576579

noncomputable def boat_speed_in_still_water
  (t_AB_with_current : ℕ)
  (t_BA_against_current : ℕ)
  (current_speed : ℕ)
  (h : ∀ (x : ℝ), 2 * (x + current_speed) = 2.5 * (x - current_speed)) : ℝ :=
  27

theorem average_speed_boat_still_water : boat_speed_in_still_water 2 2.5 3 sorry = 27 := by
  sorry

end average_speed_boat_still_water_l576_576579


namespace compound_interest_rate_l576_576091

theorem compound_interest_rate
  (P : ℝ) (t : ℕ) (A : ℝ) (interest : ℝ)
  (hP : P = 6000)
  (ht : t = 2)
  (hA : A = 7260)
  (hInterest : interest = 1260.000000000001)
  (hA_eq : A = P + interest) :
  ∃ r : ℝ, (1 + r)^(t : ℝ) = A / P ∧ r = 0.1 :=
by
  sorry

end compound_interest_rate_l576_576091


namespace subset_strict_M_P_l576_576000

-- Define the set M
def M : Set ℕ := {x | ∃ a : ℕ, a > 0 ∧ x = a^2 + 1}

-- Define the set P
def P : Set ℕ := {y | ∃ b : ℕ, b > 0 ∧ y = b^2 - 4*b + 5}

-- Prove that M is strictly a subset of P
theorem subset_strict_M_P : M ⊆ P ∧ ∃ x ∈ P, x ∉ M :=
by
  sorry

end subset_strict_M_P_l576_576000


namespace sufficient_but_not_necessary_l576_576050

noncomputable def line1 (m : ℝ) : ℝ × ℝ → ℝ := λ (p : ℝ × ℝ), 2 * (m + 1) * p.1 + (m - 3) * p.2 + 7 - 5 * m
noncomputable def line2 (m : ℝ) : ℝ × ℝ → ℝ := λ (p : ℝ × ℝ), (m - 3) * p.1 + 2 * p.2 - 5

lemma perpendicular_condition {m : ℝ} : 
  (2 * (m + 1) * (m - 3) + 2 * (m - 3) = 0) → m = 3 ∨ m = -2 :=
sorry

theorem sufficient_but_not_necessary :
  (∃ m, m = 3 ∧ (∀ p, line1 m p = 0 → line2 m p = 0)) →
  (∃ m, m ≠ 3 ∧ (∀ p, line1 m p = 0 → line2 m p = 0)) :=
sorry

end sufficient_but_not_necessary_l576_576050


namespace minimum_number_of_troublemakers_l576_576488

theorem minimum_number_of_troublemakers (n m : ℕ) (students : Fin n → Prop) 
  (truthtellers : Fin n → Prop) (liars : Fin n → Prop) (round_table : List (Fin n))
  (h_all_students : n = 29)
  (h_truth_or_liar : ∀ s, students s → (truthtellers s ∨ liars s))
  (h_truth_tellers_conditions : ∀ s, truthtellers s → 
    (count_lies_next_to s = 1 ∨ count_lies_next_to s = 2))
  (h_liars_conditions : ∀ s, liars s → 
    (count_lies_next_to s ≠ 1 ∧ count_lies_next_to s ≠ 2))
  (h_round_table_conditions : round_table.length = n 
    ∧ (∀ i, i < n → round_table.nth i ≠ none)):
  m = 10 :=
by
  sorry

end minimum_number_of_troublemakers_l576_576488


namespace prisoner_hat_strategy_l576_576327

theorem prisoner_hat_strategy (n : ℕ) :
  ∃ (strategy : fin n → (fin n → bool) → bool),
    (∃ (S : set (fin n → bool)), 
      ∀ (f : fin n → bool), (∑ (i : fin n), if strategy i (λ j, if j = i then f j else f j) = f i then 1 else 0) = n / 2) :=
sorry

end prisoner_hat_strategy_l576_576327


namespace not_k_type_3_minus_4_div_x_three_type_neg_half_x_squared_plus_x_two_type_abs_3_pow_x_minus_1_max_diff_n_minus_m_l576_576357

-- Prove that f(x) = 3 - 4/x is not a k-type function.
theorem not_k_type_3_minus_4_div_x (k : ℝ) : ¬k_type_function (λ x, 3 - 4 / x) := sorry

-- Prove that if y = -1/2 * x^2 + x is a 3-type function, then m = -4 and n = 0.
theorem three_type_neg_half_x_squared_plus_x (m n : ℝ) :
  k_type_function (λ x, -1/2 * x^2 + x) 3 → m = -4 ∧ n = 0 := sorry

-- Prove that if f(x) = |3^x - 1| is a 2-type function, then m + n = 1.
theorem two_type_abs_3_pow_x_minus_1 (m n : ℝ) :
  k_type_function (λ x, abs (3 ^ x - 1)) 2 → m + n = 1 := sorry

-- Prove that if y = ((a^2 + a)x - 1)/(a^2x) is a 1-type function, then the maximum value of n - m is 2√3 / 3.
theorem max_diff_n_minus_m (a m n : ℝ) (ha : a ≠ 0) :
  k_type_function (λ x, ((a^2 + a) * x - 1) / (a^2 * x)) 1 → n - m ≤ 2 * real.sqrt 3 / 3 := sorry

end not_k_type_3_minus_4_div_x_three_type_neg_half_x_squared_plus_x_two_type_abs_3_pow_x_minus_1_max_diff_n_minus_m_l576_576357


namespace count_4_digit_numbers_l576_576292

/-- 
Prove that the number of 4-digit positive integers with four different digits, 
where the leading digit is not zero, the integer is a multiple of 8, 
and the largest digit is 8, is equal to 21.
-/
theorem count_4_digit_numbers : 
  (finset.range 9000).filter (λ n, 
    let digits := n.digits 10 in 
      n > 999 ∧ 
      digits.nodup ∧ 
      digits.head ≠ 0 ∧ 
      digits.max' (by sorry) = 8 ∧ 
      n % 8 = 0
  ).card = 21 := 
    sorry

end count_4_digit_numbers_l576_576292


namespace rectangle_perimeter_l576_576435

theorem rectangle_perimeter (length : ℕ) (width_cm : ℕ) : length = 6 → width_cm = 20 → 2 * (length + width_cm / 10) = 16 := by
  intros h1 h2
  rw [h1, h2]
  sorry

end rectangle_perimeter_l576_576435


namespace find_m_l576_576445

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the line equation
def line (x m : ℝ) : ℝ := 2 * x + m

-- Define the distance from point (a, a^2) to the line 2a - a^2 + m = 0
def distance (a m : ℝ) : ℝ := (2 * a - a^2 + m) / real.sqrt 5

-- Given condition: the shortest distance is sqrt(5)
def given_distance (a m : ℝ) : ℝ := sqrt 5

-- The main theorem to prove: if the shortest distance from a point on the parabola to the line is sqrt(5), then m should be -6
theorem find_m (a m : ℝ) (h : distance a m = sqrt 5) : m = -6 :=
sorry

end find_m_l576_576445


namespace fraction_inequality_l576_576760

theorem fraction_inequality (a b c d : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) :
  (a / d) < (b / c) :=
sorry

end fraction_inequality_l576_576760


namespace intersection_S_T_eq_interval_l576_576287

-- Define the sets S and T
def S : Set ℝ := {x | x ≥ 2}
def T : Set ℝ := {x | x ≤ 5}

-- Prove the intersection of S and T is [2, 5]
theorem intersection_S_T_eq_interval : S ∩ T = {x | 2 ≤ x ∧ x ≤ 5} :=
by
  sorry

end intersection_S_T_eq_interval_l576_576287


namespace min_trees_include_three_types_l576_576819

noncomputable def minNumTrees (T : Type) (tree_counts : T → ℕ) :=
  ∀ (total_trees : ℕ) (at_least : ℕ → Prop),
    total_trees = 100 →
    (∀ S : Finset T, S.card = 85 → (∃ t ∈ S, at_least (tree_counts t))) →
    (at_least ThreeTypes ↔ ∃ (S : Finset T), S.card = 69)

def ThreeTypes (tree_counts : T → ℕ) := 
  ∀ (birches spruces pines aspens : ℕ),
    birches + spruces + pines + aspens = 100 →
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ pines ≥ 1) ∨ 
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ aspens ≥ 1) ∨ 
    (birches ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1) ∨ 
    (spruces ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1)

theorem min_trees_include_three_types :
  ∃ (T : Type) (tree_counts : T → ℕ), minNumTrees T tree_counts := 
sorry

end min_trees_include_three_types_l576_576819


namespace trigonometric_identity_proof_l576_576558

theorem trigonometric_identity_proof (α : ℝ) :
  3.3998 * (Real.cos α) ^ 4 - 4 * (Real.cos α) ^ 3 - 8 * (Real.cos α) ^ 2 + 3 * Real.cos α + 1 =
  -2 * Real.sin (7 * α / 2) * Real.sin (α / 2) :=
by
  sorry

end trigonometric_identity_proof_l576_576558


namespace complex_on_imaginary_axis_l576_576304

noncomputable def isImaginary (z : ℂ) : Prop := z.re = 0

theorem complex_on_imaginary_axis (z : ℂ) (h : |z - 1| = |z + 1|) : isImaginary z :=
by sorry

end complex_on_imaginary_axis_l576_576304


namespace cost_of_game_l576_576402

theorem cost_of_game
  (number_of_ice_creams : ℕ) 
  (price_per_ice_cream : ℕ)
  (total_sold : number_of_ice_creams = 24)
  (price : price_per_ice_cream = 5) :
  (number_of_ice_creams * price_per_ice_cream) / 2 = 60 :=
by
  sorry

end cost_of_game_l576_576402


namespace rearrangement_inequality_positive_numbers_l576_576369

theorem rearrangement_inequality_positive_numbers
  (n : ℕ) (x : Fin n → ℝ) (hpos : ∀ i, 0 < x i) :
  ∑ i : Fin n, (x i)^2 ≥ ∑ i : Fin n, x i * x ((i + 1) %
    n) := by sorry

end rearrangement_inequality_positive_numbers_l576_576369


namespace pentagonal_pyramid_layer_15_prism_layer_15_pentagonal_to_triangular_l576_576174

-- Definitions and conditions

def pentagonal_number (k : ℕ) : ℕ :=
  (1 / 2 : ℚ) * k * (3 * k - 1)

def total_pentagonal_pyramid (l : ℕ) : ℕ :=
  (l * (3 * l^2 - l)) / 2

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Part (a)
theorem pentagonal_pyramid_layer_15 : pentagonal_number 15 = 330 := 
by sorry

-- Part (b)
theorem prism_layer_15 : (total_pentagonal_pyramid 15) / 15 = 330 := 
by sorry

-- Part (c)
theorem pentagonal_to_triangular (l : ℕ) (h : l ≥ 2) :
  ∃ n : ℕ, l * triangular_number n = total_pentagonal_pyramid l :=
by sorry

end pentagonal_pyramid_layer_15_prism_layer_15_pentagonal_to_triangular_l576_576174


namespace triangle_area_l576_576963

theorem triangle_area (r R : ℝ) (h_r : r = 4) (h_R : R = 9) (h_trig : 2 * Real.cos A = Real.cos B + Real.cos C) :
  ∃ a b c : ℕ, a ≠ 0 ∧ c ≠ 0 ∧ Nat.gcd a c = 1 ∧ ¬ ∃ d : ℕ, d * d = b ∧ 4 * Real.sqrt b = 8 * Real.sqrt 181 ∧
   6 * 4 * (1 + 4 / 9) * (1 + 4 / 9) = ∃ (T : ℝ), T = 8 *Real.sqrt 181 := sorry

end triangle_area_l576_576963


namespace sarah_photos_l576_576189

theorem sarah_photos (photos_Cristina photos_John photos_Clarissa total_slots : ℕ)
  (hCristina : photos_Cristina = 7)
  (hJohn : photos_John = 10)
  (hClarissa : photos_Clarissa = 14)
  (hTotal : total_slots = 40) :
  ∃ photos_Sarah, photos_Sarah = total_slots - (photos_Cristina + photos_John + photos_Clarissa) ∧ photos_Sarah = 9 :=
by
  sorry

end sarah_photos_l576_576189


namespace vector_dot_product_solution_l576_576717

noncomputable def vector_dot_product_problem : Prop :=
  ∀ (a b : EuclideanSpace ℝ (Fin 2)), 
  ‖a‖ = 3 → ‖b‖ = 6 → angle a b = real.pi / 3 → 
  dot_product a (a + b) = 18

theorem vector_dot_product_solution : vector_dot_product_problem := 
by
  intros a b ha hb hab 
  -- Proof omitted
  sorry

end vector_dot_product_solution_l576_576717


namespace divisible_by_5_l576_576973

theorem divisible_by_5 (A : ℕ) (h : A ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) : (A * 10000 + 835) % 5 = 0 :=
by sorry

end divisible_by_5_l576_576973


namespace min_value_5_5_l576_576863

noncomputable def given_expression (x y z : ℝ) : ℝ :=
  (6 * z) / (2 * x + y) + (6 * x) / (y + 2 * z) + (4 * y) / (x + z)

theorem min_value_5_5 (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z = 1) :
  given_expression x y z ≥ 5.5 :=
sorry

end min_value_5_5_l576_576863


namespace distance_between_foci_of_hyperbola_is_correct_l576_576673

noncomputable def distance_between_foci_of_hyperbola : ℝ := 
  let a_sq := 50
  let b_sq := 8
  let c_sq := a_sq + b_sq
  let c := Real.sqrt c_sq
  2 * c

theorem distance_between_foci_of_hyperbola_is_correct :
  distance_between_foci_of_hyperbola = 2 * Real.sqrt 58 :=
by
  sorry

end distance_between_foci_of_hyperbola_is_correct_l576_576673


namespace relationship_a_c_b_l576_576701

variable (x : ℝ)
noncomputable def a := Real.log x
noncomputable def b := (1/2)^(Real.log x)
noncomputable def c := Real.exp (Real.log x)

theorem relationship_a_c_b (h : x ∈ Set.Ioo (Real.exp (-1)) 1) :
    a x < c x ∧ c x < b x :=
sorry

end relationship_a_c_b_l576_576701


namespace quadrilateral_sum_reciprocal_power_zero_l576_576866

-- Define the vertices of the quadrilateral
variables {A₁ A₂ A₃ A₄ : Type}

-- Define non-cyclic quadrilateral
axiom non_cyclic_quad : ¬ cyclic {A₁, A₂, A₃, A₄}

-- Define the centers of the circumcircles of the triangles
variables (O₁ O₂ O₃ O₄ : Type)

-- Define the radii of the circumcircles of the triangles
variables (r₁ r₂ r₃ r₄ : ℝ)
 
-- Define the distances from the centers to the points
variables (d₁ : ℝ := (O₁ A₁) ^ 2) (d₂ : ℝ := (O₂ A₂) ^ 2) (d₃ : ℝ := (O₃ A₃) ^ 2) (d₄ : ℝ := (O₄ A₄) ^ 2)

theorem quadrilateral_sum_reciprocal_power_zero 
  (h₁ : d₁ - r₁^2 ≠ 0)
  (h₂ : d₂ - r₂^2 ≠ 0)
  (h₃ : d₃ - r₃^2 ≠ 0)
  (h₄ : d₄ - r₄^2 ≠ 0) :
  1 / (d₁ - r₁^2) + 
  1 / (d₂ - r₂^2) + 
  1 / (d₃ - r₃^2) + 
  1 / (d₄ - r₄^2) = 0 :=
by sorry

end quadrilateral_sum_reciprocal_power_zero_l576_576866


namespace intersection_complement_N_l576_576756

open Set

def U := {1, 2, 3, 4, 5}
def M := {1, 2, 3}
def N := {x | 4 < x ∧ x ≤ 6}

theorem intersection_complement_N : (U \ M) ∩ N = {5} :=
by 
  sorry

end intersection_complement_N_l576_576756


namespace solve_inequality_l576_576738

def f (x : ℝ) : ℝ := if x ≥ 0 then 1 else -1

theorem solve_inequality :
  {x : ℝ | x + (x+2) * f (x+2) ≤ 5} = {x : ℝ | x ≤ (3 / 2)} :=
by
  sorry

end solve_inequality_l576_576738


namespace correct_option_A_l576_576549

def compute_sqrt_division : Prop := (sqrt 27 / sqrt 3 = 3)
def sqrt_sum : Prop := (sqrt 2 + sqrt 5 ≠ sqrt 7)
def sqrt_multiplication : Prop := (sqrt 8 / sqrt 2 ≠ 4) -- redefining as the simplest comparison
def sqrt_negative : Prop := (sqrt ((-3)^2) ≠ -3)

theorem correct_option_A 
  (h1 : compute_sqrt_division) 
  (h2 : sqrt_sum) 
  (h3 : sqrt_multiplication) 
  (h4 : sqrt_negative) : 
  true := 
by 
  exact trivial

end correct_option_A_l576_576549


namespace problem_statement_l576_576284

noncomputable def a : ℕ → ℤ
| 0 => 0       -- These offsets account for our sequence being 1-indexed in the problem
| 1 => 1
| 2 => 2
| (n+3) => a (n+2) - a (n+1)

def periodicity (a : ℕ → ℤ) (T : ℕ) : Prop :=
  ∀ n, a (n + T) = a n

def S : ℕ → ℤ
| 0 => 0
| (n + 1) => S n + a (n + 1)

theorem problem_statement :
  a 3 = 1 ∧ a 4 = -1 ∧ a 5 = -2 ∧ periodicity a 6 ∧ S 6 = 0 ∧
  (∀ n : ℕ, S (6 * n) = 0) ∧
  (∀ n r : ℕ, n = 6 * (n / 6) + r → r < 6 → S n = S r) :=
by
  sorry

end problem_statement_l576_576284


namespace seating_arrangements_l576_576958

def valid_seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  if total_seats = 8 ∧ people = 3 then 12 else 0

theorem seating_arrangements (total_seats people : ℕ) (h1 : total_seats = 8) (h2 : people = 3) :
  valid_seating_arrangements total_seats people = 12 :=
by
  rw [valid_seating_arrangements, h1, h2]
  simp
  done

end seating_arrangements_l576_576958


namespace base3_addition_correct_l576_576617

theorem base3_addition_correct :
  show_nat (1012_3 + 2021_3 + 11021_3 + 12012_3) 3 = 212002_3 := by
  sorry

end base3_addition_correct_l576_576617


namespace prob_students_both_days_l576_576961

def num_scenarios (students : ℕ) (choices : ℕ) : ℕ :=
  choices ^ students

def scenarios_sat_sun (total_scenarios : ℕ) (both_days_empty : ℕ) : ℕ :=
  total_scenarios - both_days_empty

theorem prob_students_both_days :
  let students := 3
  let choices := 2
  let total_scenarios := num_scenarios students choices
  let both_days_empty := 2 -- When all choose Saturday or all choose Sunday
  let scenarios_both := scenarios_sat_sun total_scenarios both_days_empty
  let probability := scenarios_both / total_scenarios
  probability = 3 / 4 :=
by
  sorry

end prob_students_both_days_l576_576961


namespace value_of_expression_l576_576299

theorem value_of_expression (x y : ℝ) (hy : y > 0) (h : x = 3 * y) :
  (x^y * y^x) / (y^y * x^x) = 3^(-2 * y) := by
  sorry

end value_of_expression_l576_576299


namespace domain_of_f_max_a_for_inequality_l576_576729

-- Conditions
def f (x a : ℝ) : ℝ := Real.log (2, abs (x + 1) + abs (x - 1) - a)

-- Problem 1: Domain of f(x) when a = 3
theorem domain_of_f (x : ℝ) : (f x 3 > 0) ↔ (x < -3/2 ∨ x > 3/2) :=
sorry

-- Problem 2: Maximum value of a for inequality f(x) ≥ 2 holding ∀ x ∈ ℝ
theorem max_a_for_inequality (a : ℝ) : (∀ x : ℝ, f x a ≥ 2) ↔ a ≤ -2 :=
sorry

end domain_of_f_max_a_for_inequality_l576_576729


namespace number_of_students_l576_576770

-- Given conditions
def average_score_boys (x : ℕ) : ℝ := 4
def average_score_girls (y : ℕ) : ℝ := 3.25
def overall_average (x y : ℕ) : ℝ := 3.6
def student_range (x y : ℕ) : Prop := 30 < x + y ∧ x + y < 50

-- Proof Statement
theorem number_of_students (x y : ℕ) (hx : average_score_boys x) (hy : average_score_girls y) (hxy : overall_average x y) (h_range : student_range x y) :
  x = 21 ∧ y = 24 :=
sorry

end number_of_students_l576_576770


namespace shaded_area_circle_area_ratio_l576_576557

noncomputable theory

def AB := 6 * r
def AC := 4 * r
def CB := 2 * r
def AE := (1 / 3) * AB
def CD := sqrt (28) * r

def shaded_area := (1/2) * π * (3 * r)^2 - ( (1/2) * π * (2 * r)^2 + (1/2) * π * r^2 + (1/2) * π * (r / 3)^2 )

def circle_area := π * (CD)^2

def ratio := shaded_area / circle_area

theorem shaded_area_circle_area_ratio (r : ℝ) : ratio =  35 / 252 :=
by
  sorry

end shaded_area_circle_area_ratio_l576_576557


namespace find_S9_l576_576068

variable {a : ℕ → ℝ} -- Define a as a function from natural numbers to reals representing the sequence

-- Conditions: arithmetic sequence with given sum condition
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0  -- Definition of an arithmetic sequence

def sum_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 5 + a 8 = 15

-- Define Sn
def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  n * (a 1 + a n) / 2

-- Goal
theorem find_S9 (a : ℕ → ℝ) (h1: arithmetic_seq a) (h2: sum_condition a) : S 9 a = 45 :=
by
  sorry

end find_S9_l576_576068


namespace monotonic_decreasing_interval_l576_576926

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem monotonic_decreasing_interval :
  ∃ a b : ℝ, (∀ x > 1, f x = x / Real.log x) ∧ (∀ x > 1, f' x = (Real.log x - 1) / (Real.log x)^2) ∧ (∀ x, 1 < x ∧ x < Real.exp 1 → f' x < 0) ∧ (∀ x, 1 < x ∧ x < Real.exp 1 → ∀ y, 1 < y ∧ y < x → f x > f y) :=
by
  sorry

end monotonic_decreasing_interval_l576_576926


namespace colin_average_mile_time_l576_576637

theorem colin_average_mile_time :
  let first_mile_time := 6
  let next_two_miles_total_time := 5 + 5
  let fourth_mile_time := 4
  let total_time := first_mile_time + next_two_miles_total_time + fourth_mile_time
  let number_of_miles := 4
  (total_time / number_of_miles) = 5 := by
    let first_mile_time := 6
    let next_two_miles_total_time := 5 + 5
    let fourth_mile_time := 4
    let total_time := first_mile_time + next_two_miles_total_time + fourth_mile_time
    let number_of_miles := 4
    have h1 : total_time = 20 := by sorry
    have h2 : total_time / number_of_miles = 20 / 4 := by sorry
    have h3 : 20 / 4 = 5 := by sorry
    exact Eq.trans (Eq.trans h2 h3) h1.symm

end colin_average_mile_time_l576_576637


namespace fraction_meaningful_l576_576424

theorem fraction_meaningful (x : ℝ) : (x + 2 ≠ 0) ↔ x ≠ -2 := by
  sorry

end fraction_meaningful_l576_576424


namespace min_trees_for_three_types_l576_576801

-- Define types and trees
inductive TreeType
| birch | spruce | pine | aspen
deriving Inhabited, DecidableEq

-- A grove with 100 trees of any of the four types.
structure Tree :=
(type : TreeType)

constant grove : List Tree
axiom grove_size : grove.length = 100

-- Condition: Among any 85 trees, there are trees of all four types.
axiom all_types_in_any_85 : ∀ (s : Finset Tree), s.card = 85 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, ∃ t4 ∈ s, 
      t1.type = TreeType.birch ∧
      t2.type = TreeType.spruce ∧
      t3.type = TreeType.pine ∧
      t4.type = TreeType.aspen

-- We need to show that at least 69 trees are needed to ensure at least 3 types.
theorem min_trees_for_three_types : 
  ∀ (s : Finset Tree), s.card = 69 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, 
      t1.type ≠ t2.type ∧ t2.type ≠ t3.type ∧ t1.type ≠ t3.type := 
sorry

end min_trees_for_three_types_l576_576801


namespace min_liars_needed_l576_576473

namespace Problem

/-
  We introduce our problem using Lean definitions and statements.
-/

def students_sitting_at_table : ℕ := 29

structure Student :=
(is_liar : Bool)

def student_truth_statement (s: ℕ) (next1: Student) (next2: Student) : Bool :=
  if s then (next1.is_liar && !next2.is_liar) || (!next1.is_liar && next2.is_liar)
  else next1.is_liar && next2.is_liar

def minimum_liars_class (sitting_students: ℕ) : ℕ :=
  sitting_students / 3 + (if sitting_students % 3 = 0 then 0 else 1)

theorem min_liars_needed (students : List Student) (H_total : students.length = students_sitting_at_table)
  (H_truth_teller_claim : ∀ (i : ℕ), i < students.length → student_truth_statement i (students.get ⟨(i+1) % students.length, sorry⟩) (students.get ⟨(i+2) % students.length, sorry⟩)) :
  (students.count (λ s => s.is_liar)) ≥ minimum_liars_class students_sitting_at_table :=
sorry

end Problem

end min_liars_needed_l576_576473


namespace general_solution_correct_l576_576207

noncomputable def general_solution_equation (C1 C2 : ℝ) (x : ℝ) : ℝ :=
  (C1 + C2 * x + 7 * x^2) * real.exp (-3 * x)

theorem general_solution_correct (C1 C2 : ℝ) (x : ℝ) :
  ∃ (y : ℝ → ℝ), (∀ x, 
  has_deriv_at y (-3 * exp (-3 * x) * (C1 + C2 * x + 7 * x^2) + exp (-3 * x) * (C2 + 14 * x))
  x ∧ 
  has_deriv_at (has_deriv_at y x) (-3 * exp (-3 * x) * (C1 + C2 * x + 7 * x^2) 
  + 2 * exp (-3 * x) * (C2 + 14 * x))
  x ∧ y'' + 6 * y' + 9 * y = 14 * exp (-3 * x)) :=
sorry

end general_solution_correct_l576_576207


namespace problem_l576_576698

def f (x : ℝ) := x^3 + 2 * x

theorem problem (a : ℝ) : f(a) + f(-a) = 0 :=
by
  -- Proof goes here
  sorry

end problem_l576_576698


namespace value_of_expression_l576_576447

theorem value_of_expression : (5^2 - 4^2 + 3^2) = 18 := 
by
  sorry

end value_of_expression_l576_576447


namespace exists_comb_of_signs_divisible_by_7_exists_comb_of_signs_divisible_by_p_l576_576351

/-- Specific case for p = 7 -/
theorem exists_comb_of_signs_divisible_by_7 (x : Fin 6 → ℤ) (h : ∀ i : Fin 6, ¬ (7 ∣ x i)) :
  ∃ ε : Fin 6 → ℤ, (∀ i : Fin 6, ε i ∈ ({-1, 1} : Set ℤ)) ∧ (∑ i, ε i * x i) % 7 = 0 :=
sorry

/-- Generalization to any odd prime p -/
theorem exists_comb_of_signs_divisible_by_p (p : ℕ) (hp : Nat.Prime p) (hodd : p % 2 = 1)
  (x : Fin (p - 1) → ℤ) (h : ∀ i : Fin (p - 1), ¬ (p ∣ x i)) :
  ∃ ε : Fin (p - 1) → ℤ, (∀ i : Fin (p - 1), ε i ∈ ({-1, 1} : Set ℤ)) ∧ (∑ i, ε i * x i) % p = 0 :=
sorry

end exists_comb_of_signs_divisible_by_7_exists_comb_of_signs_divisible_by_p_l576_576351


namespace problem_solution_l576_576171
noncomputable def count_distinct_results : ℕ :=
-- Define the set of distinct results obtained by arranging 2, 0, 1, 3 with "+" and "-" operators
let results := {6, 2, 4, 0, -4, -2, -6} in 
results.card

theorem problem_solution : count_distinct_results = 7 := 
by
  -- Skip the proof, the theorem statement verifies that there are exactly 7 distinct results.
  sorry

end problem_solution_l576_576171


namespace sheep_daddy_input_l576_576054

-- Conditions for black box transformations
def black_box (k : ℕ) : ℕ :=
  if k % 2 = 1 then 4 * k + 1 else k / 2

-- The transformation chain with three black boxes
def black_box_chain (k : ℕ) : ℕ :=
  black_box (black_box (black_box k))

-- Theorem statement capturing the problem:
-- Final output m is 2, and the largest input leading to this is 64.
theorem sheep_daddy_input : ∃ k : ℕ, ∀ (k1 k2 k3 k4 : ℕ), 
  black_box_chain k1 = 2 ∧ 
  black_box_chain k2 = 2 ∧ 
  black_box_chain k3 = 2 ∧ 
  black_box_chain k4 = 2 ∧ 
  k1 ≠ k2 ∧ k2 ≠ k3 ∧ k3 ≠ k4 ∧ k4 ≠ k1 ∧ 
  k = max k1 (max k2 (max k3 k4)) → k = 64 :=
sorry  -- Proof is not required

end sheep_daddy_input_l576_576054


namespace ping_pong_property_l576_576031

variable (G : Type) [Fintype G] (match_result : G → G → Prop) [DecidableRel match_result]

def member_exists (A B C : G) : Prop :=
  (match_result A B ∨ ∃ C : G, match_result A C ∧ match_result C B)

theorem ping_pong_property :
  ∃ A : G, ∀ B : G, member_exists G match_result A B :=
sorry

end ping_pong_property_l576_576031


namespace modulus_solution_eq_one_l576_576764

noncomputable def modulus_of_solution (z : ℂ) (h : z^2 - z + 1 = 0) : ℝ :=
|z|

theorem modulus_solution_eq_one (z : ℂ) (h : z^2 - z + 1 = 0) : modulus_of_solution z h = 1 :=
sorry

end modulus_solution_eq_one_l576_576764


namespace find_coefficients_l576_576285

theorem find_coefficients
  (a b c : ℝ)
  (hA : ∀ x : ℝ, (x = -3 ∨ x = 4) ↔ (x^2 + a * x - 12 = 0))
  (hB : ∀ x : ℝ, (x = -3 ∨ x = 1) ↔ (x^2 + b * x + c = 0))
  (hAnotB : ¬ (∀ x, (x^2 + a * x - 12 = 0) ↔ (x^2 + b * x + c = 0)))
  (hA_inter_B : ∀ x, x = -3 ↔ (x^2 + a * x - 12 = 0) ∧ (x^2 + b * x + c = 0))
  (hA_union_B : ∀ x, (x = -3 ∨ x = 1 ∨ x = 4) ↔ (x^2 + a * x - 12 = 0) ∨ (x^2 + b * x + c = 0)):
  a = -1 ∧ b = 2 ∧ c = -3 :=
sorry

end find_coefficients_l576_576285


namespace correct_choice_is_C_l576_576097

-- Define the proposition C.
def prop_C : Prop := ∃ x : ℝ, |x - 1| < 0

-- The problem statement in Lean 4.
theorem correct_choice_is_C : ¬ prop_C :=
by
  sorry

end correct_choice_is_C_l576_576097


namespace find_GR_l576_576845

open Real -- Use Real for calculations

-- Given triangle XYZ with specified side lengths
variables {X Y Z G R : Type}
variables (XY XZ YZ : ℝ)

-- Medians intersect at the centroid G
variables [G_centroid : ∀ (G : Type), is_centroid G X Y Z]

-- The foot of the altitude from G to YZ is R
variables (R_foot : ∀ (R : Type), is_foot_of_altitude R G Y Z)

-- Given conditions:
def side_XY := (XY = 14 : ℝ)
def side_XZ := (XZ = 15 : ℝ)
def side_YZ := (YZ = 21 : ℝ)

-- Declaration about the sides of the triangle
noncomputable def down to Lean proof : (GR : ℝ) :=
begin
  sorry
end

theorem find_GR :
  (GR = (200 / 63) : ℝ) :=
begin
  -- Use given side lengths and geometric properties to prove the length of GR
  work_on_goal
  sorry
end

end find_GR_l576_576845


namespace max_log_sum_l576_576361

open Real

theorem max_log_sum (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 4 * y = 40) :
  log 10 x + log 10 y ≤ 2 :=
by 
  sorry

end max_log_sum_l576_576361


namespace smallest_trees_in_three_types_l576_576777

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end smallest_trees_in_three_types_l576_576777


namespace karlson_word_count_l576_576346

def single_word_count : Nat := 9
def ten_to_nineteen_count : Nat := 10
def two_word_count (num_tens_units : Nat) : Nat := 2 * num_tens_units

def count_words_1_to_99 : Nat :=
  let single_word := single_word_count + ten_to_nineteen_count
  let two_word := two_word_count (99 - (single_word_count + ten_to_nineteen_count))
  single_word + two_word

def prefix_hundred (count_1_to_99 : Nat) : Nat := 9 * count_1_to_99
def extra_prefix (num_two_word_transformed : Nat) : Nat := 9 * num_two_word_transformed

def total_words : Nat :=
  let first_99 := count_words_1_to_99
  let nine_hundreds := prefix_hundred count_words_1_to_99 + extra_prefix 72
  first_99 + nine_hundreds + 37

theorem karlson_word_count : total_words = 2611 :=
  by
    sorry

end karlson_word_count_l576_576346


namespace angle_P_A₂_A₃_l576_576912

-- Define points and the assumptions on their configuration
variable {Point : Type} [EuclideanGeometry Point] 
variable (A₁ A₂ A₃ A₄ P : Point) 

-- Assumptions based on the problem statement
def regular_hexagon_consecutive_vertices (A₁ A₂ A₃ A₄ : Point) : Prop := 
  ∃ (hexagon : List Point), 
    hexagon.Nodup ∧ -- The points are distinct
    hexagon.Cycle ∧ -- They form a cyclic order like a hexagon
    hexagon = [A₁, A₂, A₃, A₄] ∨ hexagon = [A₂, A₃, A₄, A₁] ∨ 
    hexagon = [A₃, A₄, A₁, A₂] ∨ hexagon = [A₄, A₁, A₂, A₃]

-- Angle measures provided in the problem
def angle_P_A₁_A₂_is_40 (A₁ A₂ P : Point) : Prop := 
  ∃ (angle : Angle), angle.measure = 40 ∧ angle = Angle.mk P A₁ A₂

def angle_P_A₃_A₄_is_120 (A₃ A₄ P : Point) : Prop := 
  ∃ (angle : Angle), angle.measure = 120 ∧ angle = Angle.mk P A₃ A₄

-- The theorem to prove that angle seen at A₂A₃ from point P is 37.36°
theorem angle_P_A₂_A₃ {A₁ A₂ A₃ A₄ P : Point} 
  (h_hex : regular_hexagon_consecutive_vertices A₁ A₂ A₃ A₄)
  (h_angle_40 : angle_P_A₁_A₂_is_40 A₁ A₂ P)
  (h_angle_120 : angle_P_A₃_A₄_is_120 A₃ A₄ P) : 
  ∃ (angle : Angle), angle.measure = 37.36 ∧ angle = Angle.mk P A₂ A₃ := 
sorry

end angle_P_A₂_A₃_l576_576912


namespace find_second_number_l576_576118

theorem find_second_number (A B : ℝ) (h1 : A = 6400) (h2 : 0.05 * A = 0.2 * B + 190) : B = 650 :=
by
  sorry

end find_second_number_l576_576118


namespace range_of_m_for_distinct_real_roots_of_quadratic_l576_576312

theorem range_of_m_for_distinct_real_roots_of_quadratic (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 4*x1 - m = 0 ∧ x2^2 + 4*x2 - m = 0) ↔ m > -4 :=
by
  sorry

end range_of_m_for_distinct_real_roots_of_quadratic_l576_576312


namespace basketball_team_win_requirement_l576_576575

theorem basketball_team_win_requirement :
  ∀ (games_won_first_60 : ℕ) (total_games : ℕ) (win_percentage : ℚ) (remaining_games : ℕ),
    games_won_first_60 = 45 →
    total_games = 110 →
    win_percentage = 0.75 →
    remaining_games = 50 →
    ∃ games_won_remaining, games_won_remaining = 38 ∧
    (games_won_first_60 + games_won_remaining) / total_games = win_percentage :=
by
  intros
  sorry

end basketball_team_win_requirement_l576_576575


namespace change_received_after_purchase_l576_576034

def football_price : ℝ := 9.14
def baseball_price : ℝ := 6.81
def jump_rope_price : ℝ := 3.55
def frisbee_price : ℝ := 2.50
def jump_rope_discount : ℝ := 0.15
def frisbee_discount : ℝ := 0.10
def amount_paid : ℝ := 20 + 3 + 0.50  -- Sum of $20 bill, three $1 coins, and one 50-cent coin

theorem change_received_after_purchase : 
  let total_before_discounts := football_price + baseball_price + jump_rope_price + frisbee_price,
      jump_rope_discount_amount := jump_rope_discount * jump_rope_price,
      frisbee_discount_amount := frisbee_discount * frisbee_price,
      total_discounts := jump_rope_discount_amount + frisbee_discount_amount,
      total_after_discounts := total_before_discounts - total_discounts,
      rounded_total := Real.round (total_after_discounts * 100) / 100,
      change := amount_paid - rounded_total
  in
  change = 2.28 := by
  sorry

end change_received_after_purchase_l576_576034


namespace minimum_circles_to_cover_minimum_circles_to_cover_with_radius_lt_sqrt3_div2_minimum_circles_to_cover_with_radius_lt_sqrt2_div2_l576_576372

def circle := { radius : ℝ }

theorem minimum_circles_to_cover (K : circle) (K_i : ℕ → circle) :
  (K.radius = 1) →
  (∀ i, K_i i.radius < 1) →
  (∃ n, (∀ i < n, circle K_i i ⊆ circle K) ∧ ¬ ∃ m, (m < n ∧ ∀ i < m, circle K_i i ⊆ circle K)) →
  ∃ m, m ≥ 3 :=
sorry

theorem minimum_circles_to_cover_with_radius_lt_sqrt3_div2 (K : circle) (K_i : ℕ → circle) :
  (K.radius = 1) →
  (∀ i, K_i i.radius < (Math.sqrt 3) / 2) →
  (∃ n, (∀ i < n, circle K_i i ⊆ circle K) ∧ ¬ ∃ m, (m < n ∧ ∀ i < m, circle K_i i ⊆ circle K)) →
  ∃ m, m ≥ 4 :=
sorry

theorem minimum_circles_to_cover_with_radius_lt_sqrt2_div2 (K : circle) (K_i : ℕ → circle) :
  (K.radius = 1) →
  (∀ i, K_i i.radius < (Math.sqrt 2) / 2) →
  (∃ n, (∀ i < n, circle K_i i ⊆ circle K) ∧ ¬ ∃ m, (m < n ∧ ∀ i < m, circle K_i i ⊆ circle K)) →
  ∃ m, m ≥ 5 :=
sorry

end minimum_circles_to_cover_minimum_circles_to_cover_with_radius_lt_sqrt3_div2_minimum_circles_to_cover_with_radius_lt_sqrt2_div2_l576_576372


namespace smallest_value_n_l576_576071

theorem smallest_value_n :
  ∃ (n : ℕ), ∀ (lines : list (set ℝ)), 
    (∃ l₁, list.countp (λ l, l ≠ l₁ ∧ l₁ ∩ l ≠ ∅) lines = 5) ∧
    (∃ l₂, list.countp (λ l, l ≠ l₂ ∧ l₂ ∩ l ≠ ∅) lines = 9) ∧
    (∃ l₃, list.countp (λ l, l ≠ l₃ ∧ l₃ ∩ l ≠ ∅) lines = 11) → 
  n = 12 :=
sorry

end smallest_value_n_l576_576071


namespace horizontal_asymptote_crossing_value_l576_576229

def g (x : ℝ) : ℝ := (3 * x^2 - 7 * x - 6) / (2 * x^2 - 5 * x + 2)

theorem horizontal_asymptote_crossing_value :
  ∃ x : ℝ, g x = 3 / 2 ∧ x = 18 / 7 :=
by
  use 18 / 7
  split
  · -- to prove g(18 / 7) = 3 / 2
    have h : 3 * (18 / 7)^2 - 7 * (18 / 7) - 6 = (3/2) * (2 * (18 / 7)^2 - 5 * (18 / 7) + 2), { sorry }
    rw [←h],
    exact rfl,
  · rfl


end horizontal_asymptote_crossing_value_l576_576229


namespace passes_through_fixed_point_l576_576428

theorem passes_through_fixed_point (a : ℝ) (ha_pos : a > 0) (ha_ne : a ≠ 1) :
    (2 * a^(|1 - 1|) - 1 = 1) :=
by
  sorry

end passes_through_fixed_point_l576_576428


namespace find_s_of_3_l576_576862

noncomputable def t (x : ℝ) : ℝ := 4 * x - 5
noncomputable def s (y : ℝ) : ℝ := y^2 + 4 * y - 1

theorem find_s_of_3 : s(3) = 11 := by
  sorry

end find_s_of_3_l576_576862


namespace min_troublemakers_in_class_l576_576461

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l576_576461


namespace complex_i_power_l576_576870

theorem complex_i_power (n : ℕ) (x : Fin n → ℤ)
  (h1 : ∀ i, x i = 1 ∨ x i = -1)
  (h2 : (Fin n).sum (λ i, x i * x ((i + 1) % n)) = 0) : Complex.I ^ n = 1 :=
sorry

end complex_i_power_l576_576870


namespace minimum_trees_l576_576790

variable (Trees : Type) [Fintype Trees] [DecidableEq Trees]

def trees_in_grove : Nat := 100

def tree_type := {birches, spruces, pines, aspens} : Set Trees

def condition (s : Finset Trees) : Prop := 
  s.card > 85 → tree_type ⊆ s

theorem minimum_trees (s : Finset Trees) (H : condition s) : 
  ∃ (n : Nat), n ≤ trees_in_grove ∧ n ≥ 69 → 
  ∃ t ⊆ s, t.card = n ∧ (|t ∩ tree_type| >= 3) :=
sorry

end minimum_trees_l576_576790


namespace distribute_seedlings_l576_576693

noncomputable def box_contents : List ℕ := [28, 51, 135, 67, 123, 29, 56, 38, 79]

def total_seedlings (contents : List ℕ) : ℕ := contents.sum

def obtainable_by_sigmas (contents : List ℕ) (σs : List ℕ) : Prop :=
  ∃ groups : List (List ℕ),
    (groups.length = σs.length) ∧
    (∀ g ∈ groups, contents.contains g.sum) ∧
    (∀ g, g ∈ groups → g.sum ∈ σs)

theorem distribute_seedlings : 
  total_seedlings box_contents = 606 →
  obtainable_by_sigmas box_contents [202, 202, 202] ∧
  ∃ way1 way2 : List (List ℕ),
    (way1 ≠ way2) ∧
    (obtainable_by_sigmas box_contents [202, 202, 202]) :=
by
  sorry

end distribute_seedlings_l576_576693


namespace unique_solution_l576_576203

theorem unique_solution (x : ℝ) : (3 : ℝ)^x + (4 : ℝ)^x + (5 : ℝ)^x = (6 : ℝ)^x ↔ x = 3 := by
  sorry

end unique_solution_l576_576203


namespace shopkeeper_percentage_above_cost_l576_576135

theorem shopkeeper_percentage_above_cost (CP MP SP : ℚ) 
  (h1 : CP = 100) 
  (h2 : SP = CP * 1.02)
  (h3 : SP = MP * 0.85) : 
  (MP - CP) / CP * 100 = 20 :=
by sorry

end shopkeeper_percentage_above_cost_l576_576135


namespace inequality_lemma_l576_576877

variable {ℝ : Type*}
variables (f : ℝ → ℝ) [OrderedSemiring ℝ]

-- Condition: f satisfies f(x * y) ≤ f(x) * f(y) for any positive x, y
axiom hf : ∀ x y : ℝ, 0 < x → 0 < y → f(x * y) ≤ f(x) * f(y)

-- Prove that: f(x^n) ≤ product of (f(x^k))^(1/k) for k=1 to n
theorem inequality_lemma (x : ℝ) (n : ℕ) (hx : 0 < x) : 
  f(x ^ n) ≤ ∏ k in Finset.range (n + 1) \ {0}, f(x ^ k) ^ (1 / k) := by
  sorry

end inequality_lemma_l576_576877


namespace crit_point_at_zero_zero_in_intervals_iff_range_of_a_l576_576733

-- Given conditions
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * Real.sin x - a * Real.log (x + 1)

def givenData : Prop := (Real.exp (π / 4) * Real.sqrt 2 / 2) > 1

-- Part (Ⅰ)
theorem crit_point_at_zero {a : ℝ} (h : (deriv (f a) 0 = 0)) : a = 1 := by sorry

-- Part (Ⅱ)
theorem zero_in_intervals_iff_range_of_a (a : ℝ) (h1 : ∃ x ∈ Ioo (-1 : ℝ) 0, f a x = 0)
  (h2 : ∃ x ∈ Ioo (π / 4) π, f a x = 0) (h3 : givenData) : 0 < a ∧ a < 1 := by sorry

end crit_point_at_zero_zero_in_intervals_iff_range_of_a_l576_576733


namespace ratio_of_x_and_y_l576_576291

theorem ratio_of_x_and_y {x y a b : ℝ} (h1 : (2 * a - x) / (3 * b - y) = 3) (h2 : a / b = 4.5) : x / y = 3 :=
sorry

end ratio_of_x_and_y_l576_576291


namespace prime_pair_problem_l576_576753

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def count_prime_pairs : ℕ :=
  Finset.card {
    (p, q) | is_prime p ∧ is_prime q ∧ is_prime (p^q - p * q) }.to_Finset

theorem prime_pair_problem :
  count_prime_pairs = 2 :=
sorry

end prime_pair_problem_l576_576753


namespace ellas_10th_roll_is_last_l576_576302

def prob_last_roll_is_10th (rolls : List ℕ) : ℚ :=
  if rolls.length = 10 ∧ List.pairwise (λ x y => x ≠ y) rolls.tail.heads.prefix 9 ++ [true] then
    (5/6)^8 * (1/6)
  else
    0

theorem ellas_10th_roll_is_last :
  ∀ rolls : List ℕ, is_die_rolls rolls → ∃ p : ℚ, prob_last_roll_is_10th rolls = 0.039 :=
by
  sorry

end ellas_10th_roll_is_last_l576_576302


namespace equiangular_iff_regular_hexagon_l576_576594

-- Definition of a hexagon being equiangular
def is_equiangular_hexagon (P : Type) [polygon P] : Prop :=
  (num_sides P = 6) ∧ (∀ angles ∈ (interior_angles P), angles = 120)

-- Definition of a hexagon being regular
def is_regular_hexagon (P : Type) [polygon P] : Prop :=
  (is_equilateral P) ∧ (is_equiangular_hexagon P)

-- Theorem statement: A polygon is an equiangular hexagon if and only if it is a regular hexagon
theorem equiangular_iff_regular_hexagon (P : Type) [polygon P] : 
  is_equiangular_hexagon P ↔ is_regular_hexagon P :=
by
  sorry -- Proof omission

end equiangular_iff_regular_hexagon_l576_576594


namespace least_three_digit_multiple_of_9_eq_108_l576_576537

/--
What is the least positive three-digit multiple of 9?
-/
theorem least_three_digit_multiple_of_9_eq_108 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 9 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 9 = 0 → n ≤ m :=
  ∃ n : ℕ, n = 108 :=
begin
  sorry
end

end least_three_digit_multiple_of_9_eq_108_l576_576537


namespace inequality_sum_le_one_l576_576399

-- Given non-negative real numbers whose average is 1
variable {a : Fin 2017 → ℝ}
variable h_nonneg : ∀ i, 0 ≤ a i
variable h_avg : (∑ i, a i) = 2017

-- Prove the inequality
theorem inequality_sum_le_one : 
  (∑ i, a i / (a i ^ 2018 + ∑ j, if i ≠ j then a j else 0)) ≤ 1 := by
  sorry

end inequality_sum_le_one_l576_576399


namespace midpoint_of_IL1_or_IL_on_Omega_l576_576353

open EuclideanGeometry

variable {A B C I D M K S N L L1 : Point}

theorem midpoint_of_IL1_or_IL_on_Omega
  (hABC : scalene_triangle A B C)
  (hOmega : is_circumcircle A B C)
  (hI : incenter I A B C)
  (hAI_bc : ∃ M, ray AI ∩ circumcircle Ω = {M})
  (hD : ∃ D, line_segment I D)
  (hK : ∃ K, diameter_circle D M ∩ circumcircle Ω = {K})
  (hS : ∃ S, line MK ∩ line BC = {S})
  (hN : midpoint N I S)
  (hL1L : intersects_circles (triangle K I D) (triangle M A N) L1 L) :
  passes_midpoint_IL1_or_IL (circumcircle Ω) L1 L :=
sorry

end midpoint_of_IL1_or_IL_on_Omega_l576_576353


namespace minimum_trees_with_at_least_three_types_l576_576813

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end minimum_trees_with_at_least_three_types_l576_576813


namespace min_trees_include_three_types_l576_576818

noncomputable def minNumTrees (T : Type) (tree_counts : T → ℕ) :=
  ∀ (total_trees : ℕ) (at_least : ℕ → Prop),
    total_trees = 100 →
    (∀ S : Finset T, S.card = 85 → (∃ t ∈ S, at_least (tree_counts t))) →
    (at_least ThreeTypes ↔ ∃ (S : Finset T), S.card = 69)

def ThreeTypes (tree_counts : T → ℕ) := 
  ∀ (birches spruces pines aspens : ℕ),
    birches + spruces + pines + aspens = 100 →
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ pines ≥ 1) ∨ 
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ aspens ≥ 1) ∨ 
    (birches ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1) ∨ 
    (spruces ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1)

theorem min_trees_include_three_types :
  ∃ (T : Type) (tree_counts : T → ℕ), minNumTrees T tree_counts := 
sorry

end min_trees_include_three_types_l576_576818


namespace water_consumption_eq_l576_576668

-- Define all conditions
variables (x : ℝ) (improvement : ℝ := 0.8) (water : ℝ := 80) (days_difference : ℝ := 5)

-- State the theorem
theorem water_consumption_eq (h : improvement = 0.8) (initial_water := 80) (difference := 5) : 
  initial_water / x - (initial_water * improvement) / x = difference :=
sorry

end water_consumption_eq_l576_576668


namespace horner_operations_l576_576968

-- Definition of the polynomial using coefficients list
def poly_coeffs : List ℝ := [3, 4, 5, 6, 7, 8, 1]

-- Horner's method applied to polynomial with given x
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
coeffs.foldr (λ a acc, a + x * acc) 0

-- Assertion of number of operations
theorem horner_operations (x : ℝ) :
  let f := horner_eval poly_coeffs x in
  count_add_mult poly_coeffs = (6, 6) :=
sorry

end horner_operations_l576_576968


namespace john_initial_clean_jerk_weight_l576_576345

def initial_snatch_weight : ℝ := 50
def increase_rate : ℝ := 1.8
def total_new_lifting_capacity : ℝ := 250

theorem john_initial_clean_jerk_weight :
  ∃ (C : ℝ), 2 * C + (increase_rate * initial_snatch_weight) = total_new_lifting_capacity ∧ C = 80 := by
  sorry

end john_initial_clean_jerk_weight_l576_576345


namespace find_length_of_10_songs_l576_576411

def length_of_10_songs (x : ℕ) : Prop :=
  let length_of_2min_songs := 15 * 2 in     -- Total length of 15 2-minute songs
  let total_current_length := length_of_2min_songs + 10 * x in -- Total current length
  total_current_length = 100 - 40           -- Equation from the conditions

theorem find_length_of_10_songs (x : ℕ) (h : length_of_10_songs x) : x = 3 :=
  sorry

end find_length_of_10_songs_l576_576411


namespace range_of_a_l576_576268

noncomputable def f (x a : ℝ) : ℝ := (2 * x - 3) * Real.exp x + a / x

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, f x a = 0) ∧ (∃ x ≠ 0, f x a = 0) ∧ (∃ x ∈ Iio (-3/2), f x a = 0) → -9 * Real.exp (-3 / 2) < a ∧ a < 0 := by
  sorry

end range_of_a_l576_576268


namespace expected_convergence_l576_576003

-- Conditions
variable {α : Type*} [ProbabilityMeasure α]
variables (ξ η ζ : α → ℝ) (ξn ηn ζn : ℕ → α → ℝ)
variable [∀ n, ProbabilityMeasureSpace α (ξn n)]
variable [∀ n, ProbabilityMeasureSpace α (ηn n)]
variable [∀ n, ProbabilityMeasureSpace α (ζn n)]
variable (h1 : ∀ n, ∀ ω, ηn n ω ≤ ξn n ω ∧ ξn n ω ≤ ζn n ω)
variable (h2a : ∀ n, tendsto (ξn n) (𝓝[support α] ξ) (𝓝[support α] 0))
variable (h2b : ∀ n, tendsto (ηn n) (𝓝[support α] η) (𝓝[support α] 0))
variable (h2c : ∀ n, tendsto (ζn n) (𝓝[support α] ζ) (𝓝[support α] 0))
variable (h3a : tendsto (λ n, ∫ ω, ηn n ω ∂ volume) (𝓝[support α]) (integral volume η))
variable (h3b : tendsto (λ n, ∫ ω, ζn n ω ∂ volume) (𝓝[support α]) (integral volume ζ))
variable (finite_exp_ξ : ∫ ω, |ξ ω| ∂ volume < ∞)
variable (finite_exp_η : ∫ ω, |η ω| ∂ volume < ∞)
variable (finite_exp_ζ : ∫ ω, |ζ ω| ∂ volume < ∞)

-- Goal
theorem expected_convergence
: (tendsto (λ n, ∫ ω, ξn n ω ∂ volume) (𝓝[support α]) (integral volume ξ)) ∧
  (tendsto (λ n, ∫ ω, |ξn n ω - ξ ω| ∂ volume) (𝓝[support α]) 0 ↔
   ((tendsto (λ n, ∫ ω, |ηn n ω - η ω| ∂ volume) (𝓝[support α]) 0) ∧
    (tendsto (λ n, ∫ ω, |ζn n ω - ζ ω| ∂ volume) (𝓝[support α]) 0))) :=
sorry

end expected_convergence_l576_576003


namespace sandbox_volume_l576_576611

def length : ℕ := 312
def width : ℕ := 146
def depth : ℕ := 75
def volume (l w d : ℕ) : ℕ := l * w * d

theorem sandbox_volume : volume length width depth = 3429000 := by
  sorry

end sandbox_volume_l576_576611


namespace min_troublemakers_l576_576452

theorem min_troublemakers (students : Finset ℕ) (table : List ℕ) (is_liar : ℕ → Bool) :
  students.card = 29 ∧
  (∀ s ∈ students, 
    (∃ i j : ℕ, 
      table.nth i = some s ∧ table.nth ((i + 1) % 29) = some j ∧
      (∃ k : ℕ, 
        (is_liar s = false → count (/=is_liar k liar) = 0 ∧
        is_liar s = true → count (/=is_liar k liar) = 2))) → 
  (∃! l ∈ students, count (/=is_liar l liar) = 1))) → 
  ∃ t : ℕ, t = 10 :=
sorry

end min_troublemakers_l576_576452


namespace problem_statement_l576_576719

noncomputable def ellipse_equation (t : ℝ) (ht : t > 0) : String :=
  if h : t = 2 then "x^2/9 + y^2/2 = 1"
  else "invalid equation"

theorem problem_statement (m : ℝ) (t : ℝ) (ht : t > 0) (ha : t = 2) 
  (A E F B : ℝ × ℝ) (hA : A = (-3, 0)) (hB : B = (1, 0))
  (hl : ∀ x y, x = m * y + 1) (area : ℝ) (har : area = 16/3) :
  ((ellipse_equation t ht) = "x^2/9 + y^2/2 = 1") ∧
  (∃ M N : ℝ × ℝ, 
    (M.1 = 3 ∧ N.1 = 3) ∧
    ((M.1 - B.1) * (N.1 - B.1) + (M.2 - B.2) * (N.2 - B.2) = 0)) := 
sorry

end problem_statement_l576_576719


namespace mary_max_earnings_l576_576103

noncomputable theory

def max_hours : ℕ := 50
def regular_hours : ℕ := 20
def regular_rate : ℝ := 8
def overtime_rate : ℝ := regular_rate * 1.25

def calculate_earnings : ℝ :=
  let regular_earnings := regular_rate * regular_hours
  let overtime_hours := max_hours - regular_hours
  let overtime_earnings := overtime_rate * overtime_hours
  regular_earnings + overtime_earnings

theorem mary_max_earnings : calculate_earnings = 460 :=
by
  sorry

end mary_max_earnings_l576_576103


namespace simplify_expression_l576_576041

theorem simplify_expression (x y : ℝ) (m : ℤ) : 
  ((x + y)^(2 * m + 1) / (x + y)^(m - 1) = (x + y)^(m + 2)) :=
by sorry

end simplify_expression_l576_576041


namespace remainder_when_a_is_divided_by_nine_l576_576359

theorem remainder_when_a_is_divided_by_nine (n : ℕ) (hn : n > 0) (a : ℤ) 
  (h : a ≡ (3^(2*n) + 4)⁻¹ [ZMOD 9]) : a ≡ 7 [ZMOD 9] :=
sorry

end remainder_when_a_is_divided_by_nine_l576_576359


namespace g_analytical_expression_h_range_l576_576272

section problem1

variable (t : ℝ) (ht : t > 0)

/-- Function f with given properties -/
def f (x : ℝ) : ℝ := x + t / x

/-- Given function g -/
def g (y : ℝ) : ℝ := y^2 - 2

/-- Verifying the expression g(y) for given conditions -/
theorem g_analytical_expression (x : ℝ) (hx : x ≠ 0) : g (x + 1 / x) = x^2 + 1 / x^2 - 2 := by
  sorry

end problem1

section problem2

/-- Given function h and its domain restriction -/
def h (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : ℝ := (4 * x^2 - 12 * x - 3) / (2 * x + 1)

/-- Function f adjusted for h given the range of transformation -/
def f (t : ℝ) : ℝ := t + 4 / t - 8

/-- Verifying the range of h(x) given the properties of f -/
theorem h_range (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : ∃ y, y ∈ set.Icc (-4 : ℝ) (-3) ∧ h x hx = y := by
  sorry

end problem2

end g_analytical_expression_h_range_l576_576272


namespace minimum_trees_with_at_least_three_types_l576_576810

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end minimum_trees_with_at_least_three_types_l576_576810


namespace number_of_correct_conclusions_l576_576688

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def f (x : ℝ) : ℝ := x - floor x

theorem number_of_correct_conclusions : 
  ∃ n, n = 3 ∧ 
  (0 ≤ f 0) ∧ 
  (∀ x : ℝ, 0 ≤ f x) ∧ 
  (∀ x : ℝ, f x < 1) ∧ 
  (∀ x : ℝ, f (x + 1) = f x) ∧ 
  ¬ (∀ x : ℝ, f (-x) = f x) := 
sorry

end number_of_correct_conclusions_l576_576688


namespace sufficient_but_not_necessary_condition_l576_576743

theorem sufficient_but_not_necessary_condition (x : ℝ) : 
  x^2 - 4 * x + 3 < 0 → (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 3) :=
by
  intros h
  have : x < 3 ∧ x > 1,
    sorry
  cases this,
  by_cases (x < 2),
    left,
    split,
    exact this.1,
    exact h_1,
  right,
  split,
  exact h_1,
  exact this.2

end sufficient_but_not_necessary_condition_l576_576743


namespace colin_avg_time_l576_576639

def totalTime (a b c d : ℕ) : ℕ := a + b + c + d

def averageTime (total_time miles : ℕ) : ℕ := total_time / miles

theorem colin_avg_time :
  let first_mile := 6
  let second_mile := 5
  let third_mile := 5
  let fourth_mile := 4
  let total_time := totalTime first_mile second_mile third_mile fourth_mile
  4 > 0 -> averageTime total_time 4 = 5 :=
by
  intros
  -- proof goes here
  sorry

end colin_avg_time_l576_576639


namespace minimum_average_speed_minimize_waiting_time_difference_l576_576584

-- Problem 1
theorem minimum_average_speed 
  (L : ℝ)           -- Length of the loop (30 km)
  (n : ℕ)           -- Number of trains (9)
  (max_wait_time : ℝ) : -- Maximum waiting time (10 minutes)
  L = 30 → n = 9 → max_wait_time = 10 →
  ∃ v : ℝ, v ≥ 20 :=
by 
  intros hL hn hmax_wait
  use 20
  sorry

-- Problem 2
theorem minimize_waiting_time_difference
  (L : ℝ)           -- Length of each loop (30 km)
  (v_in : ℝ)        -- Speed of the inner loop (25 km/h)
  (v_out : ℝ)       -- Speed of the outer loop (30 km/h)
  (n : ℕ) :         -- Total number of trains (18)
  L = 30 → v_in = 25 → v_out = 30 → n = 18 →
  ∃ x : ℕ, ∃ y : ℕ, x = 10 ∧ y = 8 :=
by 
  intros hL hvin hvout hn
  use 10
  use 8
  sorry

end minimum_average_speed_minimize_waiting_time_difference_l576_576584


namespace sum_of_interior_angles_of_regular_polygon_l576_576058

-- Define the condition and prove the theorem
theorem sum_of_interior_angles_of_regular_polygon (h : ∀ (n : ℕ), (n > 2) → 360 / n = 40) : 
  (180 * (9 - 2)) = 1260 :=
by
  -- Given that the measure of each exterior angle of the polygon is 40 degrees, n = 9.
  have n_pos : 9 > 2 := by norm_num
  specialize h 9 n_pos
  -- Sum of interior angles of a polygon
  have sum_interior := 180 * (9 - 2)
  rw h
  -- Conclude the proof with the calculated sum
  norm_num
  sorry

end sum_of_interior_angles_of_regular_polygon_l576_576058


namespace num_ways_to_assign_students_l576_576521

theorem num_ways_to_assign_students : 
    let students := 5
    let courses := 3
    (students ≥ courses) → 
    (∃! n : ℕ, n = 150) :=
by
    exists 150
    sorry

end num_ways_to_assign_students_l576_576521


namespace min_troublemakers_l576_576457

theorem min_troublemakers (students : Finset ℕ) (table : List ℕ) (is_liar : ℕ → Bool) :
  students.card = 29 ∧
  (∀ s ∈ students, 
    (∃ i j : ℕ, 
      table.nth i = some s ∧ table.nth ((i + 1) % 29) = some j ∧
      (∃ k : ℕ, 
        (is_liar s = false → count (/=is_liar k liar) = 0 ∧
        is_liar s = true → count (/=is_liar k liar) = 2))) → 
  (∃! l ∈ students, count (/=is_liar l liar) = 1))) → 
  ∃ t : ℕ, t = 10 :=
sorry

end min_troublemakers_l576_576457


namespace gummy_vitamin_discount_l576_576347

theorem gummy_vitamin_discount :
  ∀ (usual_price_per_bottle : ℕ) (num_bottles : ℕ) (num_coupons : ℕ) (coupon_value : ℕ) (total_cost_with_sale : ℕ),
  usual_price_per_bottle = 15 →
  num_bottles = 3 →
  num_coupons = 3 →
  coupon_value = 2 →
  total_cost_with_sale = 30 →
  100 * (usual_price_per_bottle * num_bottles - total_cost_with_sale - num_coupons * coupon_value) / (usual_price_per_bottle * num_bottles) = 20 :=
by
  intros usual_price_per_bottle num_bottles num_coupons coupon_value total_cost_with_sale
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  dsimp
  norm_num
  sorry

end gummy_vitamin_discount_l576_576347


namespace range_of_b_l576_576739

noncomputable def f (x : ℝ) : ℝ := log x - (1 / 4) * x + (3 / (4 * x)) - 1
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := x ^ 2 - 2 * b * x + 4

theorem range_of_b {b : ℝ} :
  (∀ x1 : ℝ, 0 < x1 ∧ x1 < 2 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 ≥ g x2 b) ↔
  b ≥ 17 / 8 :=
sorry

end range_of_b_l576_576739


namespace payment_difference_is_565_l576_576406

def interest_rate := 0.1
def principal := 15000
def years := 10

-- Plan 1 specifics
def plan1_first_payment_time := 5
def plan1_interest_compounds_per_year := 2
def plan1_fraction_paid_first := 1 / 3

-- Plan 2 specifics
def plan2_first_payment_time := 5
def plan2_interest_compounds_per_year := 1
def plan2_fraction_paid_first := 1 / 4

-- Total payments
noncomputable def total_payment_plan1 : ℝ :=
  let a_5 := principal * (1 + interest_rate / plan1_interest_compounds_per_year) ^ (plan1_interest_compounds_per_year * plan1_first_payment_time)
  let first_payment := a_5 * plan1_fraction_paid_first
  let remaining := a_5 * (1 - plan1_fraction_paid_first)
  let a_10 := remaining * (1 + interest_rate / plan1_interest_compounds_per_year) ^ (plan1_interest_compounds_per_year * (years - plan1_first_payment_time))
  first_payment + a_10

noncomputable def total_payment_plan2 : ℝ :=
  let a_5 := principal * (1 + interest_rate) ^ plan2_first_payment_time
  let first_payment := a_5 * plan2_fraction_paid_first
  let remaining := a_5 * (1 - plan2_fraction_paid_first)
  let a_10 := remaining * (1 + interest_rate) ^ (years - plan2_first_payment_time)
  first_payment + a_10

noncomputable def payment_difference := abs (total_payment_plan2 - total_payment_plan1)

theorem payment_difference_is_565 :
  payment_difference = 565 := sorry

end payment_difference_is_565_l576_576406


namespace shopkeeper_loss_l576_576990

def initial_value : ℝ := 100
def profit_percentage : ℝ := 0.10
def theft_percentage : ℝ := 0.30

def loss_percentage (initial_value profit_percentage theft_percentage : ℝ) : ℝ :=
  let goods_after_theft := initial_value * (1 - theft_percentage)
  let selling_price := goods_after_theft * (1 + profit_percentage)
  let loss := initial_value - selling_price
  (loss / initial_value) * 100

theorem shopkeeper_loss :
  loss_percentage initial_value profit_percentage theft_percentage = 23 := by
  sorry

end shopkeeper_loss_l576_576990


namespace units_digit_3542_pow_876_l576_576632

theorem units_digit_3542_pow_876 : (3542 ^ 876) % 10 = 6 := by 
  sorry

end units_digit_3542_pow_876_l576_576632


namespace sufficient_but_not_necessary_condition_for_prop_l576_576060

theorem sufficient_but_not_necessary_condition_for_prop :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) → a ≥ 5 :=
sorry

end sufficient_but_not_necessary_condition_for_prop_l576_576060


namespace mike_land_purchase_l576_576381

variable (A : ℕ)
variable (cost_per_acre buy_price_per_acre sell_price_per_acre profit revenue cost : ℝ)

-- Given conditions
def buy_land (A : ℕ) (cost_per_acre : ℝ) : ℝ :=
  A * cost_per_acre

def sell_land (A : ℕ) (sell_price_per_acre : ℝ) : ℝ :=
  (A / 2) * sell_price_per_acre

def profit_eq (revenue cost : ℝ) : ℝ :=
  revenue - cost

theorem mike_land_purchase :
  buy_price_per_acre = 70 → sell_price_per_acre = 200 → profit = 6000 →
  profit_eq (sell_land A sell_price_per_acre) (buy_land A buy_price_per_acre) = profit →
  A = 200 := by
  sorry

end mike_land_purchase_l576_576381


namespace min_max_expr_l576_576195

noncomputable def expr (a b c : ℝ) : ℝ :=
  (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) *
  (a^2 / (a^2 + 1) + b^2 / (b^2 + 1) + c^2 / (c^2 + 1))

theorem min_max_expr (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h_cond : a * b + b * c + c * a = 1) :
  27 / 16 ≤ expr a b c ∧ expr a b c ≤ 2 :=
sorry

end min_max_expr_l576_576195


namespace sector_area_l576_576932

theorem sector_area (r : ℝ) : (2 * r + 2 * r = 16) → (1/2 * r^2 * 2 = 16) :=
by
  intro h1
  sorry

end sector_area_l576_576932


namespace permutation_of_subsets_exists_l576_576398

theorem permutation_of_subsets_exists (n : ℕ) (h : n ≥ 4) : 
  ∃ (P : Fin (2^n - n - 1) → Finset (Fin n)), 
  (∀ i : Fin (2^n - n - 2), 
     ((P i).card ≥ 2) ∧ ((P i) ∩ (P i.succ)).card = 2) :=
sorry

end permutation_of_subsets_exists_l576_576398


namespace meet_at_start_l576_576560

noncomputable def track_length := 600 -- meters
noncomputable def speed_A_kmph := 30 -- kmph
noncomputable def speed_B_kmph := 60 -- kmph

noncomputable def speed_A_mps := speed_A_kmph * (1000 / 3600)
noncomputable def speed_B_mps := speed_B_kmph * (1000 / 3600)

noncomputable def time_A_lap := track_length / speed_A_mps
noncomputable def time_B_lap := track_length / speed_B_mps

noncomputable def lcm_time_A_B := Int.lcm (Int.ofNat time_A_lap.toInt) (Int.ofNat time_B_lap.toInt)

theorem meet_at_start (h1 : track_length = 600)
                      (h2 : speed_A_kmph = 30)
                      (h3 : speed_B_kmph = 60) : 
                      lcm_time_A_B = 72 :=
by
  sorry

end meet_at_start_l576_576560


namespace initial_number_of_friends_l576_576075

theorem initial_number_of_friends (F : ℕ) (h : 6 * (F + 2) = 60) : F = 8 :=
by {
  sorry
}

end initial_number_of_friends_l576_576075


namespace triangle_perimeter_from_medians_l576_576263

-- Given the lengths of the medians of the triangle
def medians_of_triangle := (3 : ℝ, 4 : ℝ, 6 : ℝ)

-- Let's state the theorem we need to prove
theorem triangle_perimeter_from_medians (m1 m2 m3 : ℝ) 
  (h1 : m1 = 3) (h2 : m2 = 4) (h3 : m3 = 6) : 
  ∃ a b c : ℝ, a + b + c = 26 :=
by
  let a := 6
  let b := 8
  let c := 12
  use a, b, c
  have hap : a = 6 := rfl
  have hbp : b = 8 := rfl
  have hcp : c = 12 := rfl
  rw [hap, hbp, hcp]
  exact ⟨rfl⟩

end triangle_perimeter_from_medians_l576_576263


namespace number_of_possible_values_of_k_l576_576629

theorem number_of_possible_values_of_k :
  (∃ p q : ℕ, prime p ∧ prime q ∧ p + q = 73 ∧ k = p * q ) → (∃ k : ℕ, unique (λ k, ∃ p q : ℕ, prime p ∧ prime q ∧ p + q = 73 ∧ k = p * q)) :=
by
  sorry

end number_of_possible_values_of_k_l576_576629


namespace service_cost_is_2_10_l576_576771

-- Defining the given conditions
def fuel_cost_per_liter : ℝ := 0.70
def cost_per_vehicle : ℝ := 347.2
def mini_van_tank : ℝ := 65
def truck_tank_size_bigger_percentage : ℝ := 1.2
def num_mini_vans : ℝ := 3
def num_trucks : ℝ := 2

-- Calculate the truck tank capacity
def truck_tank : ℝ := mini_van_tank * (1 + truck_tank_size_bigger_percentage)

-- Calculate the total number of liters needed
def total_liters : ℝ := (num_mini_vans * mini_van_tank) + (num_trucks * truck_tank)

-- Calculate the total fuel cost
def total_fuel_cost : ℝ := total_liters * fuel_cost_per_liter

-- Calculate the total service cost
def total_service_cost : ℝ := cost_per_vehicle - total_fuel_cost

-- Calculate the service cost per vehicle
def service_cost_per_vehicle : ℝ := total_service_cost / (num_mini_vans + num_trucks)

theorem service_cost_is_2_10 :
    service_cost_per_vehicle = 2.10 := by
  sorry

end service_cost_is_2_10_l576_576771


namespace geometric_seq_general_term_l576_576703

variable {α : Type*}

def geometric_seq (a q : α) [Semigroup α] (n : Nat) : α :=
  a * q^(n - 1)

theorem geometric_seq_general_term (a q : α) [Semigroup α] (n : Nat) :
  geometric_seq a q n = a * q^(n - 1) :=
sorry

end geometric_seq_general_term_l576_576703


namespace hiker_total_distance_l576_576561

theorem hiker_total_distance :
  let day1_distance := 18
  let day1_speed := 3
  let day2_speed := day1_speed + 1
  let day1_time := day1_distance / day1_speed
  let day2_time := day1_time - 1
  let day2_distance := day2_speed * day2_time
  let day3_speed := 5
  let day3_time := 3
  let day3_distance := day3_speed * day3_time
  let total_distance := day1_distance + day2_distance + day3_distance
  total_distance = 53 :=
by
  sorry

end hiker_total_distance_l576_576561


namespace measure_of_angle_y_l576_576334

theorem measure_of_angle_y (m n : ℝ) (A B C D F G H : ℝ) :
  (m = n) → (A = 40) → (B = 90) → (B = 40) → (y = 80) :=
by
  -- proof steps to be filled in
  sorry

end measure_of_angle_y_l576_576334


namespace selling_price_of_radio_l576_576609

-- Definitions based on given conditions
def purchase_price : ℝ := 232
def overhead_expenses : ℝ := 15
def profit_percent : ℝ := 21.457489878542503

-- Lean statement for the proof that selling price is 300 Rs
theorem selling_price_of_radio : 
  let total_cost_price := purchase_price + overhead_expenses in
  let profit_amount := (profit_percent / 100) * total_cost_price in
  let selling_price := total_cost_price + profit_amount in
  selling_price = 300 :=
by
  sorry

end selling_price_of_radio_l576_576609


namespace motorcycle_speeds_l576_576923

noncomputable def speed_m = 18
noncomputable def total_timeAtoB = 67
noncomputable def total_timeBtoA = 76
noncomputable def flat_distance = 12
noncomputable def up_distance = 3
noncomputable def down_distance = 6

theorem motorcycle_speeds (v1 v2 : ℝ) :
  (1/5 : ℝ) = v1 ∧ (1/2 : ℝ) = v2 ↔
  (3 / v1 + 6 / v2 = (27 : ℝ)) ∧
  (3 / v2 + 6 / v1 = (36 : ℝ)) ∧
  v1 = 12 ∧ v2 = 30 :=
by sorry

end motorcycle_speeds_l576_576923


namespace initial_selling_rate_l576_576601

/-- A man lost 10% by selling oranges at a certain rate per rupee. 
To gain 32%, he must sell them at the rate of 12.272727272727273 a rupee. 
Prove that the initial selling rate of oranges was approximately 8.37 oranges per rupee. --/
theorem initial_selling_rate (rate_gain : ℝ) :
  rate_gain = 12.272727272727273 →
  ∃ rate_loss : ℝ, rate_loss ≈ 8.37 := 
sorry

end initial_selling_rate_l576_576601


namespace beats_log_partition_l576_576988

noncomputable def log_base_2_10 : ℝ := real.log 10 / real.log 2
noncomputable def log_base_5_10 : ℝ := real.log 10 / real.log 5

theorem beats_log_partition (n : ℕ) (h : n > 1) :
  ∃ k : ℕ, (⌊(k * log_base_2_10 : ℝ)⌋ + 1 = n) ∨ (⌊(k * log_base_5_10 : ℝ)⌋ + 1 = n) :=
by sorry

end beats_log_partition_l576_576988


namespace max_tan_y_l576_576368

noncomputable def tan_y_upper_bound (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) 
    (h : Real.sin y = 2005 * Real.cos (x + y) * Real.sin x) : Real :=
  Real.tan y

theorem max_tan_y (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) 
    (h : Real.sin y = 2005 * Real.cos (x + y) * Real.sin x) : 
    tan_y_upper_bound x y hx hy h = 2005 * Real.sqrt 2006 / 4012 := 
by 
  sorry

end max_tan_y_l576_576368


namespace sum_of_perimeters_l576_576043

theorem sum_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 41) : 
  4 * (Real.sqrt 63 + Real.sqrt 22) = 4 * x + 4 * y := by
  sorry

end sum_of_perimeters_l576_576043


namespace average_speed_boat_in_still_water_l576_576578

variables (x : ℝ) -- average speed of the boat in still water

-- Conditions
def speed_with_current := x + 3
def speed_against_current := x - 3
def time_with_current := 2
def time_against_current := 2.5

-- Distances in both directions should be equal
def distance_with_current := time_with_current * speed_with_current
def distance_against_current := time_against_current * speed_against_current

theorem average_speed_boat_in_still_water : distance_with_current = distance_against_current → x = 27 :=
by
  sorry

end average_speed_boat_in_still_water_l576_576578


namespace determine_durations_l576_576830

def simultaneous_surgeries (dur : Fin 4 → ℕ) : Prop :=
  let total_duration := 127
  let ongoing_18min_before := 60
  let ongoing_33min_before := 25
  ∃ d1 d2 d3 d4 : ℕ, 
    (d1 + d2 + d3 + d4 = total_duration) ∧ 
    (d1 + d2 + d3 + 18 = ongoing_18min_before) ∧
    (d1 + d2 + 33 = ongoing_33min_before) ∧
    ((d4 = 13) ∧ (d1 ≠ x ∨ d2 ≠ y ∨ d3 ≠ z))

theorem determine_durations (d : Fin 4 → ℕ) : 
  simultaneous_surgeries d →
  (d 3 = 13 ∧ (d 0 = x ∨ d 1 = y ∨ d 2 = z)) :=
by {
  sorry
}

end determine_durations_l576_576830


namespace min_value_of_f_inequality_with_conditions_l576_576276

-- Definition of f
def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (5 - x)

-- Minimum value of f
theorem min_value_of_f : ∃ m, m = (9:ℝ) / 2 ∧ ∀ x, f(x) ≥ m :=
by
  let m := (9:ℝ) / 2
  use m
  split
  . rfl -- This establishes that the value is indeed 9/2
  . sorry -- Here we would prove the actual inequality for all x

-- Given constraints, prove the inequality
theorem inequality_with_conditions (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 3) :
  1 / (a + 1) + 1 / (b + 2) ≥ 2 / 3 :=
by
  have h_sum : (a + 1) + (b + 2) = 6 := by
    linarith -- Simplify the condition
  sorry -- Here we would complete the proof using properties of reciprocals and inequalities

end min_value_of_f_inequality_with_conditions_l576_576276


namespace deborah_finishes_first_l576_576878

-- Define the areas of lawns
def jonathan_lawn_area (z : ℝ) : ℝ := z
def deborah_lawn_area (z : ℝ) : ℝ := z / 3
def ezekiel_lawn_area (z : ℝ) : ℝ := z / 4

-- Define the mowing rates
def jonathan_mower_rate (r : ℝ) : ℝ := r
def deborah_mower_rate (r : ℝ) : ℝ := r / 4
def ezekiel_mower_rate (r : ℝ) : ℝ := r / 6

-- Calculate the mowing times
def jonathan_mowing_time (z r : ℝ) : ℝ := jonathan_lawn_area z / jonathan_mower_rate r
def deborah_mowing_time (z r : ℝ) : ℝ := deborah_lawn_area z / deborah_mower_rate r
def ezekiel_mowing_time (z r : ℝ) : ℝ := ezekiel_lawn_area z / ezekiel_mower_rate r

theorem deborah_finishes_first (z r : ℝ) (hz : 0 < z) (hr : 0 < r) :
  deborah_mowing_time z r < jonathan_mowing_time z r ∧
  deborah_mowing_time z r < ezekiel_mowing_time z r := by
  sorry

end deborah_finishes_first_l576_576878


namespace determinant_of_2x2_matrix_l576_576646

theorem determinant_of_2x2_matrix (a b c d : ℝ) (h_a : a = 7) (h_b : b = -2) (h_c : c = -3) (h_d : d = 6) :
  determinant ![![a, b], ![c, d]] = 36 :=
by 
  rw [h_a, h_b, h_c, h_d]
  dsimp
  norm_num
  sorry

end determinant_of_2x2_matrix_l576_576646


namespace sum_of_intervals_l576_576689

def floor (x : ℝ) : ℤ := sorry

def f (x : ℝ) : ℝ := floor x * (2014 ^ (x - floor x) - 1)

theorem sum_of_intervals : 
  (∑ k in finset.range 2013, real.log (k + 1) / real.log 2014 - real.log k / real.log 2014) = 1 := 
sorry

end sum_of_intervals_l576_576689


namespace min_trees_include_three_types_l576_576815

noncomputable def minNumTrees (T : Type) (tree_counts : T → ℕ) :=
  ∀ (total_trees : ℕ) (at_least : ℕ → Prop),
    total_trees = 100 →
    (∀ S : Finset T, S.card = 85 → (∃ t ∈ S, at_least (tree_counts t))) →
    (at_least ThreeTypes ↔ ∃ (S : Finset T), S.card = 69)

def ThreeTypes (tree_counts : T → ℕ) := 
  ∀ (birches spruces pines aspens : ℕ),
    birches + spruces + pines + aspens = 100 →
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ pines ≥ 1) ∨ 
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ aspens ≥ 1) ∨ 
    (birches ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1) ∨ 
    (spruces ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1)

theorem min_trees_include_three_types :
  ∃ (T : Type) (tree_counts : T → ℕ), minNumTrees T tree_counts := 
sorry

end min_trees_include_three_types_l576_576815


namespace part1_part2_part3_l576_576732

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + cos x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := deriv (f a) x
noncomputable def y (a : ℝ) (x : ℝ) : ℝ := f a x - x * log x

theorem part1 : g (1/2) x + (1 - cos x) ≥ 0 := sorry

theorem part2 (a : ℝ) :
  (∃ x : ℝ, deriv (f a) x = 0) ↔ (a ≥ 1/2) := sorry

theorem part3 (a : ℝ) (D : set ℝ) :
  (∀ x : ℝ, (x ∈ (m, +∈) → (deriv (y a) x > 0))) ∧ (m) ⊆ D → is_monotonic_on (D) := sorry

end part1_part2_part3_l576_576732


namespace five_n_is_odd_perfect_l576_576873

def sigma (n : ℕ) : ℕ := (Nat.divisors n).sum

def is_perfect (n : ℕ) : Prop := sigma n = 2 * n

theorem five_n_is_odd_perfect (n : ℕ) (h : 3 * sigma n = 5 * n) : is_perfect (5 * n) ∧ n % 2 = 1 :=
by {
  sorry
}

end five_n_is_odd_perfect_l576_576873


namespace complex_solution_count_l576_576212

theorem complex_solution_count : 
  ∃ (s : Finset ℂ), (∀ z ∈ s, (z^3 - 8) / (z^2 - 3 * z + 2) = 0) ∧ s.card = 2 := 
by
  sorry

end complex_solution_count_l576_576212


namespace n_times_s_eq_4_l576_576010

def f (n : ℕ) : ℕ

axiom f_prop : ∀ a b : ℕ, 3 * f (a^2 + b^2) = 2 * (f a)^2 + 2 * (f b)^2 - (f a) * (f b)

theorem n_times_s_eq_4 (f : ℕ → ℕ) (h : ∀ a b : ℕ, 3 * f (a^2 + b^2) = 2 * (f a)^2 + 2 * (f b)^2 - (f a) * (f b)) : 4 :=
begin
    have h0 : f 0 = 0 ∨ f 0 = 1,  -- Derived using the condition with a=0, b=0
    sorry,
    have h1 : f 16 = 0 ∨ f 16 = 2,  -- Derived through further analysis
    sorry,
    let n := 2,   -- Number of possible values for f(16)
    let s := 2,   -- Sum of the possible values for f(16)
    exact n * s,  -- n * s = 4
end

end n_times_s_eq_4_l576_576010


namespace smallest_trees_in_three_types_l576_576778

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end smallest_trees_in_three_types_l576_576778


namespace find_roots_of_equation_l576_576217

theorem find_roots_of_equation :
  ∀ (x : ℝ), x ≠ 3 ∧ x ≠ -3 → (18 / (x^2 - 9) - 3 / (x - 3) = 2 ↔ x = 3 ∨ x = -4.5) :=
by
  intros x h,
  sorry

end find_roots_of_equation_l576_576217


namespace probability_of_different_colors_l576_576957

theorem probability_of_different_colors :
  let total_chips := 12
  let prob_blue_then_yellow_red := ((6 / total_chips) * ((4 + 2) / total_chips))
  let prob_yellow_then_blue_red := ((4 / total_chips) * ((6 + 2) / total_chips))
  let prob_red_then_blue_yellow := ((2 / total_chips) * ((6 + 4) / total_chips))
  prob_blue_then_yellow_red + prob_yellow_then_blue_red + prob_red_then_blue_yellow = 11 / 18 := by
    sorry

end probability_of_different_colors_l576_576957


namespace sum_fraction_equals_two_l576_576005

theorem sum_fraction_equals_two
  (a b c d : ℝ) (h₁ : a ≠ -1) (h₂ : b ≠ -1) (h₃ : c ≠ -1) (h₄ : d ≠ -1)
  (ω : ℂ) (h₅ : ω^4 = 1) (h₆ : ω ≠ 1)
  (h₇ : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = (4 / (ω^2))) 
  (h₈ : a + b + c + d = a * b * c * d)
  (h₉ : a * b + a * c + a * d + b * c + b * d + c * d = a * b * c + a * b * d + a * c * d + b * c * d) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := 
sorry

end sum_fraction_equals_two_l576_576005


namespace angle_of_inclination_of_line_AB_l576_576767

-- Definition of points A and B based on conditions
def pointA := (1 : ℝ, 1 : ℝ)
def pointB := (-2 : ℝ, 1 + Real.sqrt 3)

-- The slope of the line passing through points A and B
def slope := (pointB.2 - pointA.2) / (pointB.1 - pointA.1)

-- The angle of inclination of the line with the given slope
def angleOfInclination := Real.arctan slope

theorem angle_of_inclination_of_line_AB : Real.arctan (-Real.sqrt 3 / 3) = 5 * Real.pi / 6 := by
  sorry

end angle_of_inclination_of_line_AB_l576_576767


namespace probability_of_y_gt_2x_l576_576182

noncomputable def probability_y_gt_2x : ℝ := 
  (∫ x in (0:ℝ)..(1000:ℝ), ∫ y in (2*x)..(2000:ℝ), (1 / (1000 * 2000) : ℝ)) * (1000 * 2000)

theorem probability_of_y_gt_2x : probability_y_gt_2x = 0.5 := sorry

end probability_of_y_gt_2x_l576_576182


namespace polar_to_cartesian_eq_polar_circle_area_l576_576933

theorem polar_to_cartesian_eq (p θ x y : ℝ) (h : p = 2 * Real.cos θ)
  (hx : x = p * Real.cos θ) (hy : y = p * Real.sin θ) :
  x^2 - 2 * x + y^2 = 0 := sorry

theorem polar_circle_area (p θ : ℝ) (h : p = 2 * Real.cos θ) :
  Real.pi = Real.pi := (by ring)


end polar_to_cartesian_eq_polar_circle_area_l576_576933


namespace calculate_power_expr_l576_576179

theorem calculate_power_expr :
  let a := (-8 : ℝ)
  let b := (0.125 : ℝ)
  a^2023 * b^2024 = -0.125 :=
by
  sorry

end calculate_power_expr_l576_576179


namespace perpendicular_planes_l576_576289

-- Definitions for lines and planes and their relationships
variable {a b : Line}
variable {α β : Plane}

-- Given conditions for the problem
axiom line_perpendicular (l1 l2 : Line) : Prop -- l1 ⊥ l2
axiom line_parallel (l1 l2 : Line) : Prop -- l1 ∥ l2
axiom line_plane_perpendicular (l : Line) (p : Plane) : Prop -- l ⊥ p
axiom line_plane_parallel (l : Line) (p : Plane) : Prop -- l ∥ p
axiom plane_perpendicular (p1 p2 : Plane) : Prop -- p1 ⊥ p2

-- Problem statement
theorem perpendicular_planes (h1 : line_perpendicular a b)
                            (h2 : line_plane_perpendicular a α)
                            (h3 : line_plane_perpendicular b β) :
                            plane_perpendicular α β :=
sorry

end perpendicular_planes_l576_576289


namespace min_troublemakers_29_l576_576502

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l576_576502


namespace minimum_number_of_troublemakers_l576_576487

theorem minimum_number_of_troublemakers (n m : ℕ) (students : Fin n → Prop) 
  (truthtellers : Fin n → Prop) (liars : Fin n → Prop) (round_table : List (Fin n))
  (h_all_students : n = 29)
  (h_truth_or_liar : ∀ s, students s → (truthtellers s ∨ liars s))
  (h_truth_tellers_conditions : ∀ s, truthtellers s → 
    (count_lies_next_to s = 1 ∨ count_lies_next_to s = 2))
  (h_liars_conditions : ∀ s, liars s → 
    (count_lies_next_to s ≠ 1 ∧ count_lies_next_to s ≠ 2))
  (h_round_table_conditions : round_table.length = n 
    ∧ (∀ i, i < n → round_table.nth i ≠ none)):
  m = 10 :=
by
  sorry

end minimum_number_of_troublemakers_l576_576487


namespace polynomial_properties_l576_576441

theorem polynomial_properties :
  let p := (3 * x^2 * y - x * y^2 - 3 * x * y^3 + x^5 - 1) in
  (degree p = 5) ∧
  (num_terms p = 5) ∧
  (arrange_desc_y p = (- 3 * x * y^3 - x * y^2 + 3 * x^2 * y + x^5 - 1)) :=
sorry

end polynomial_properties_l576_576441


namespace faster_train_passes_slower_l576_576085

noncomputable def time_to_pass (L V_faster V_slower : ℝ) : ℝ :=
  let relative_speed := (V_faster - V_slower) * (5 / 18) -- Convert km/hr to m/s
  let total_distance := 2 * L
  total_distance / relative_speed

theorem faster_train_passes_slower :
  ∀ (L V_faster V_slower : ℝ),
  L = 100 ∧ V_faster = 46 ∧ V_slower = 36 →
  abs (time_to_pass L V_faster V_slower - 72) < 1 :=
by
  intros L V_faster V_slower h
  cases h with hL hVS
  cases hVS with hVf hVsl
  simp [hL, hVf, hVsl, time_to_pass]
  -- calculation verification is omitted 
  sorry

end faster_train_passes_slower_l576_576085


namespace son_work_time_l576_576991

theorem son_work_time :
  let M := (1 : ℚ) / 7
  let combined_rate := (1 : ℚ) / 3
  let S := combined_rate - M
  1 / S = 5.25 :=  
by
  sorry

end son_work_time_l576_576991


namespace petya_time_less_than_two_l576_576891

noncomputable def is_distance_less_than_two (d v : ℝ) : Prop :=
  (d / v) < 2

theorem petya_time_less_than_two {d v : ℝ} 
  (same_distance : d = d)
  (vasya_speed : 2 * v)
  (started_late : 1)
  (petya_first : (d / v) < ((d / (2 * v)) + 1)) :
  (d / v) < 2 := 
sorry

end petya_time_less_than_two_l576_576891


namespace machines_work_together_l576_576079

theorem machines_work_together (x : ℝ) (h_pos : 0 < x) :
  (1 / (x + 2) + 1 / (x + 3) + 1 / (x + 1) = 1 / x) → x = 1 :=
by
  sorry

end machines_work_together_l576_576079


namespace smallest_trees_in_three_types_l576_576775

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end smallest_trees_in_three_types_l576_576775


namespace find_cost_price_l576_576104

variables (SP_d SP SP_wd CP : ℝ)

-- Conditions extracted from the problem statement
def condition1 := SP_wd = 27000
def condition2 := SP_d = 0.9 * SP_wd
def condition3 := SP_d = CP * 1.08

-- Statement to be proven
theorem find_cost_price (h1 : condition1) (h2 : condition2) (h3 : condition3) : CP = 22500 :=
by
  sorry

end find_cost_price_l576_576104


namespace corn_plants_multiple_of_nine_l576_576297

theorem corn_plants_multiple_of_nine 
  (num_sunflowers : ℕ) (num_tomatoes : ℕ) (num_corn : ℕ) (max_plants_per_row : ℕ)
  (h1 : num_sunflowers = 45) (h2 : num_tomatoes = 63) (h3 : max_plants_per_row = 9)
  : ∃ k : ℕ, num_corn = 9 * k :=
by
  sorry

end corn_plants_multiple_of_nine_l576_576297


namespace distance_between_foci_of_hyperbola_l576_576206

theorem distance_between_foci_of_hyperbola :
  ∀ x y : ℝ, (x^2 - 8 * x - 16 * y^2 - 16 * y = 48) → (∃ c : ℝ, 2 * c = 2 * Real.sqrt 63.75) :=
by
  sorry

end distance_between_foci_of_hyperbola_l576_576206


namespace height_of_platform_l576_576527

variable (h l w : ℕ)

-- Define the conditions as hypotheses
def measured_length_first_configuration : Prop := l + h - w = 40
def measured_length_second_configuration : Prop := w + h - l = 34

-- The goal is to prove that the height is 37 inches
theorem height_of_platform
  (h l w : ℕ)
  (config1 : measured_length_first_configuration h l w)
  (config2 : measured_length_second_configuration h l w) : 
  h = 37 := 
sorry

end height_of_platform_l576_576527


namespace sum_integer_solutions_l576_576906

noncomputable def isValidSolution (x : ℝ) : Prop :=
  12 * ((| x + 10 | - | x - 20 |) / (| 4 * x - 25 | - | 4 * x - 15 |)) -
  ((| x + 10 | + | x - 20 |) / (| 4 * x - 25 | + | 4 * x - 15 |)) ≥ -6

theorem sum_integer_solutions : ∑ x in finset.filter (λ i : ℤ, -100 < i ∧ i < 100 ∧ isValidSolution (i : ℝ)) (finset.range 200), x = 4 := 
by {
  sorry
}

end sum_integer_solutions_l576_576906


namespace rectangle_area_l576_576140

theorem rectangle_area (a w l : ℝ) (h_square_area : a = 36) 
    (h_rect_width : w * w = a) 
    (h_rect_length : l = 3 * w) : w * l = 108 := 
sorry

end rectangle_area_l576_576140


namespace min_value_f_l576_576699

-- Definition of the function f
def f (x : ℝ) : ℝ := x^3 - (1/2)*x^2 - 2*x + 6

-- The statement that we need to prove
theorem min_value_f : min (f (1 : ℝ)) (min (f (-1)) (f 2)) = (9/2 : ℝ) := by
  sorry

end min_value_f_l576_576699


namespace set_intersection_l576_576747

def U : Set ℝ := Set.univ
def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x ≥ 2}
def C_U_B : Set ℝ := {x | x < 2}

theorem set_intersection :
  A ∩ C_U_B = {-1, 0, 1} :=
sorry

end set_intersection_l576_576747


namespace upper_bound_expression_l576_576691

theorem upper_bound_expression (n : ℤ) (U : ℤ) :
  (∀ n, 4 * n + 7 > 1 ∧ 4 * n + 7 < U → ∃ k : ℤ, k = 50) →
  U = 204 :=
by
  sorry

end upper_bound_expression_l576_576691


namespace colin_avg_time_l576_576641

def totalTime (a b c d : ℕ) : ℕ := a + b + c + d

def averageTime (total_time miles : ℕ) : ℕ := total_time / miles

theorem colin_avg_time :
  let first_mile := 6
  let second_mile := 5
  let third_mile := 5
  let fourth_mile := 4
  let total_time := totalTime first_mile second_mile third_mile fourth_mile
  4 > 0 -> averageTime total_time 4 = 5 :=
by
  intros
  -- proof goes here
  sorry

end colin_avg_time_l576_576641


namespace ultra_squarish_count_l576_576134

def is_perfect_square (n : ℕ) : Prop := ∃ k, k * k = n

def is_ultra_squarish (n : ℕ) : Prop :=
  let digits := n.digits 9 in
  digits.length = 7 ∧ 
  (∀ d ∈ digits, d ≠ 0) ∧
  is_perfect_square n ∧
  is_perfect_square (digits.take 3).foldl (λ acc d => 9 * acc + d) 0 ∧
  is_perfect_square (digits.drop 2).take 3.foldl (λ acc d => 9 * acc + d) 0 ∧
  is_perfect_square (digits.drop 4).foldl (λ acc d => 9 * acc + d) 0

theorem ultra_squarish_count : ∃ n, is_ultra_squarish n ∧ (ultra_squarish_numbers (9^6) (9^7 - 1)).count = 2 :=
sorry

end ultra_squarish_count_l576_576134


namespace no_real_solutions_l576_576763

theorem no_real_solutions (x : ℝ) : 
  (log (x + 4) + log (x - 2) = log (x^2 - 6 * x + 8)) → false := 
by
  sorry

end no_real_solutions_l576_576763


namespace part1_eq_of_circle_part2_l_intersects_C_l576_576843

def O := (0 : Real, 0 : Real)
def A := (6 : Real, Real.pi / 2)
def B := (6 * Real.sqrt 2, Real.pi / 4)

def lineL (t α : Real) := (-1 + t * Real.cos α, 2 + t * Real.sin α)

def P := (-1 : Real, 2 : Real)

theorem part1_eq_of_circle {θ : Real} :
  ∀ θ, ∃ ρ, ρ = 6 * (Real.cos θ + Real.sin θ) := sorry

theorem part2_l_intersects_C (α t : Real) :
  ∃ (M N : (Real × Real)), (M ≠ N) ∧ 
  (M.1 = -1 + t * Real.cos α) ∧ (M.2 = 2 + t * Real.sin α) ∧
  (N.1 = -1 + t * Real.cos α) ∧ (N.2 = 2 + t * Real.sin α) ∧
  |M - P| * |N - P| = 1 := sorry

end part1_eq_of_circle_part2_l_intersects_C_l576_576843


namespace arithmetic_seq_property_l576_576832

theorem arithmetic_seq_property (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_cond : a 1 + 3 * a 8 + a 15 = 120) :
  2 * a 9 - a 10 = 24 := 
sorry

end arithmetic_seq_property_l576_576832


namespace min_troublemakers_l576_576514

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l576_576514


namespace final_expression_simplified_l576_576755

variable (b : ℝ)

theorem final_expression_simplified :
  ((3 * b + 6 - 5 * b) / 3) = (-2 / 3) * b + 2 := by
  sorry

end final_expression_simplified_l576_576755


namespace octagon_area_l576_576169

theorem octagon_area (P: ℕ) (hP: P = 108) : 
    let side := P / 4 in
    let trisected_segment := side / 3 in
    let area_square := side ^ 2 in
    let area_triangle := (trisected_segment * trisected_segment) / 2 in
    let total_area_triangles := 4 * area_triangle in
    let area_octagon := area_square - total_area_triangles in
    area_octagon = 567 := 
by
    cut (side = 27)
    cut (trisected_segment = 9)
    cut (area_square = 729)
    cut (area_triangle = 40.5)
    cut (total_area_triangles = 162)
    cut (area_octagon = 567)
    sorry

end octagon_area_l576_576169


namespace min_value_of_b_over_a_l576_576279

theorem min_value_of_b_over_a (a b : ℝ) (h : ∀ x : ℝ, x > -1 → ln (x + 1) - 1 ≤ a * x + b) :
  1 - e ≤ b / a :=
sorry

end min_value_of_b_over_a_l576_576279


namespace relationship_l576_576084

-- Given definitions
def S : ℕ := 31
def L : ℕ := 124 - S

-- Proving the relationship
theorem relationship: S + L = 124 ∧ S = 31 → L = S + 62 := by
  sorry

end relationship_l576_576084


namespace necessary_not_sufficient_condition_l576_576716

def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0

lemma purely_imaginary_iff_conjugate_eq_zero {z : ℂ} :
  is_purely_imaginary z ↔ z + conj(z) = 0 :=
begin
  sorry
end

theorem necessary_not_sufficient_condition {z : ℂ} :
  (z + conj(z) = 0) → is_purely_imaginary z ∨ z = 0 :=
begin
  sorry
end

end necessary_not_sufficient_condition_l576_576716


namespace tangent_line_at_1_l576_576918

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6

theorem tangent_line_at_1 :
  let x := 1 in
  let y := f x in
  let slope := deriv f 1 in
  3 * (1 : ℝ) + y = 7 :=
by
  let x := (1 : ℝ)
  let y := f x
  let slope := deriv f 1
  have h1 : f 1 = 4 := by
    simp [f]
  have h2 : deriv f 1 = -3 := by
    simp [f]
  rw [h1, ←h2]
  simp
  sorry

end tangent_line_at_1_l576_576918


namespace find_ab_value_l576_576433

-- Definitions from the conditions
def ellipse_eq (a b : ℝ) : Prop := b^2 - a^2 = 25
def hyperbola_eq (a b : ℝ) : Prop := a^2 + b^2 = 49

-- Main theorem statement
theorem find_ab_value {a b : ℝ} (h_ellipse : ellipse_eq a b) (h_hyperbola : hyperbola_eq a b) : 
  |a * b| = 2 * Real.sqrt 111 :=
by
  -- Proof goes here
  sorry

end find_ab_value_l576_576433


namespace centroid_quad_area_ratio_l576_576857

variables {P Q R S : Type} [AddCommGroup P] [Module ℝ P] [AddCommGroup Q] [Module ℝ Q] [AddCommGroup R] [Module ℝ R] [AddCommGroup S] [Module ℝ S]
variables {PQRS : ConvexQuadrilateral P Q R S}

-- Definitions of centroids
def G_P (Q R S : P) : P := (Q + R + S) / 3
def G_Q (P R S : Q) : Q := (P + R + S) / 3
def G_R (P Q S : R) : R := (P + Q + S) / 3
def G_S (P Q R : S) : S := (P + Q + R) / 3

-- Area ratio proof statement
theorem centroid_quad_area_ratio (PQRS : ConvexQuadrilateral P Q R S) :
  ([G_P Q R S, G_Q P R S, G_R P Q S, G_S P Q R] : set P) / ([P, Q, R, S] : set P) = 1 / 9 :=
sorry

end centroid_quad_area_ratio_l576_576857


namespace weight_loss_challenge_l576_576099

noncomputable def percentage_weight_loss (W : ℝ) : ℝ :=
  ((W - (0.918 * W)) / W) * 100

theorem weight_loss_challenge (W : ℝ) (h : W > 0) :
  percentage_weight_loss W = 8.2 :=
by
  sorry

end weight_loss_challenge_l576_576099


namespace ryegrass_percentage_in_X_l576_576897

-- Definitions of conditions
def P : ℝ := 46.67 / 100
def Q : ℝ := 100 - 46.67
def ryegrass_Y : ℝ := 25 / 100
def ryegrass_mixture : ℝ := 32 / 100

-- The percentage of ryegrass in mixture X we need to find
noncomputable def ryegrass_X : ℝ := 40 / 100

-- The main theorem we need to prove
theorem ryegrass_percentage_in_X : 
  (P * (ryegrass_X) + Q * ryegrass_Y) = ryegrass_mixture := 
by
  -- This is where the proof will go
  sorry

end ryegrass_percentage_in_X_l576_576897


namespace minjeong_soohyeok_equal_day_l576_576017

def equal_money_day (d : ℕ) : Prop :=
  let minjeong_amount := 8000 + 300 * d
  let soohyeok_amount := 5000 + 500 * d
  minjeong_amount = soohyeok_amount

theorem minjeong_soohyeok_equal_day : ∃ d : ℕ, equal_money_day d ∧ d = 15 :=
by
  use 15
  split
  · unfold equal_money_day
    rfl
  · rfl
  sorry

end minjeong_soohyeok_equal_day_l576_576017


namespace smallest_b_value_l576_576412

theorem smallest_b_value (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a - b = 7) 
    (h₄ : (Nat.gcd ((a^3 + b^3) / (a + b)) (a^2 * b)) = 12) : b = 6 :=
by
    -- proof goes here
    sorry

end smallest_b_value_l576_576412


namespace colin_average_time_per_mile_l576_576644

theorem colin_average_time_per_mile :
  (let first_mile := 6
   let second_mile := 5
   let third_mile := 5
   let fourth_mile := 4
   let total_miles := 4
   let total_time := first_mile + second_mile + third_mile + fourth_mile
   let average_time := total_time / total_miles
   average_time = 5) :=
begin
  let first_mile := 6,
  let second_mile := 5,
  let third_mile := 5,
  let fourth_mile := 4,
  let total_miles := 4,
  let total_time := first_mile + second_mile + third_mile + fourth_mile,
  let average_time := total_time / total_miles,
  show average_time = 5,
  sorry,
end

end colin_average_time_per_mile_l576_576644


namespace total_time_to_row_l576_576998

/-- 
  Given:
    speed_boat_still_water : Float := 16
    speed_stream : Float := 2
    distance_to_place : Float := 7380
  
  Prove:
    total_time_taken = 937.14
-/
theorem total_time_to_row (speed_boat_still_water : Float) (speed_stream : Float) (distance_to_place : Float) : 
  speed_boat_still_water = 16 ∧ speed_stream = 2 ∧ distance_to_place = 7380 → 
  let downstream_speed := speed_boat_still_water + speed_stream,
      upstream_speed := speed_boat_still_water - speed_stream,
      time_downstream := distance_to_place / downstream_speed,
      time_upstream := distance_to_place / upstream_speed in
  (time_downstream + time_upstream).toRound 2 = 937.14 :=
by
  intros h
  cases h with hs h
  obtain ⟨hb, hc⟩ := h
  sorry

end total_time_to_row_l576_576998


namespace domain_f_a_5_abs_inequality_ab_l576_576429

-- Definition for the domain of f(x) when a=5
def domain_of_f_a_5 (x : ℝ) : Prop := |x + 1| + |x + 2| - 5 ≥ 0

-- The theorem to find the domain A of the function f(x) when a=5.
theorem domain_f_a_5 (x : ℝ) : domain_of_f_a_5 x ↔ (x ≤ -4 ∨ x ≥ 1) :=
by
  sorry

-- Theorem to prove the inequality for a, b ∈ (-1, 1)
theorem abs_inequality_ab (a b : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) :
  |a + b| / 2 < |1 + a * b / 4| :=
by
  sorry

end domain_f_a_5_abs_inequality_ab_l576_576429


namespace cube_of_final_result_l576_576300

-- Conditions
variable {N : ℕ}
variable (h1 : 0.05 * N = 220)

-- Problem statement
theorem cube_of_final_result (h1 : 0.05 * N = 220) : (286 ^ 3) = 23393616 :=
by
  sorry

end cube_of_final_result_l576_576300


namespace max_t_value_l576_576833

noncomputable def max_t (x : ℝ) : ℝ :=
  1/2 * (x - 1/x)

theorem max_t_value :
  (∃ t : ℝ, (∀ m > 0, t = 1/2 * (2 * m + (log m / m) - m * log m)) ∧ 
  ↑t = 1/2 * (real.exp 1 - 1 / real.exp 1)) :=
begin
  sorry

end max_t_value_l576_576833


namespace algebra_problem_l576_576700

noncomputable def problem_statement (x : ℝ) : Prop :=
  x + x⁻¹ = 3 → x ^ (3/2) + x ^ (-3/2) = real.sqrt 5

-- The main theorem we'll state
theorem algebra_problem (x : ℝ) : problem_statement x :=
by
  -- let the proof be completed later
  sorry

end algebra_problem_l576_576700


namespace three_types_in_69_trees_l576_576789

variable (birches spruces pines aspens : ℕ)
variable (total_trees : ℕ := 100)
variable (all_trees : list (string × ℕ))

-- We assert that there are 100 trees in total, and our list of trees represents this
axiom h_total : ∑ t in all_trees, t.2 = total_trees

-- Among any 85 trees, there must be at least one of each type
axiom h_85_trees_all_types : ∀ (s : list (string × ℕ)), s.card = 85 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0 ∧ a > 0 ∧ 
  b + s + p + a = 85 ∧ (("birches", b) ∈ s ∧ ("spruces", s) ∈ s ∧ ("pines", p) ∈ s ∧ ("aspens", a) ∈ s))

-- We need to prove that any subset of 69 or more trees contains at least three different types.
theorem three_types_in_69_trees :
  ∀ (s : list (string × ℕ)), s.card = 69 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0) ∨ (b > 0 ∧ s > 0 ∧ a > 0) ∨ (b > 0 ∧ p > 0 ∧ a > 0) ∨ (s > 0 ∧ p > 0 ∧ a > 0) :=
begin
  sorry
end

end three_types_in_69_trees_l576_576789


namespace min_troublemakers_l576_576496

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l576_576496


namespace index_card_area_l576_576882

theorem index_card_area
  (L W : ℕ)
  (h1 : L = 4)
  (h2 : W = 6)
  (h3 : (L - 1) * W = 18) :
  (L * (W - 1) = 20) :=
by
  sorry

end index_card_area_l576_576882


namespace least_positive_three_digit_multiple_of_9_l576_576533

theorem least_positive_three_digit_multiple_of_9 : 
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ n % 9 = 0 ∧ ∀ m : ℕ, (m >= 100 ∧ m < 1000 ∧ m % 9 = 0) → n ≤ m :=
begin
  use 108,
  split,
  { exact nat.le_refl 108 },
  split,
  { exact nat.lt_of_lt_of_le (nat.succ_pos 9) (nat.succ_le_succ (nat.le_refl 99)) },
  split,
  { exact nat.mod_eq_zero_of_mk (nat.zero_of_succ_pos 12) },
  { intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    exact nat.le_of_eq ((nat.mod_eq_zero_of_dvd (nat.gcd_eq_gcd_ab (12) (8) (1)))),
  },
  sorry
end

end least_positive_three_digit_multiple_of_9_l576_576533


namespace January_1st_is_Tuesday_l576_576342

-- Let us define the context:
def January_has_31_days : Prop := true

def January_has_four_Fridays : Prop := true

def January_has_four_Mondays : Prop := true

-- The main theorem we want to prove:
theorem January_1st_is_Tuesday
  (h1 : January_has_31_days)
  (h2 : January_has_four_Fridays)
  (h3 : January_has_four_Mondays) :
  ∃ (d : string), d = "Tuesday" :=
sorry

end January_1st_is_Tuesday_l576_576342


namespace area_of_rectangle_l576_576153

--- Define the problem's conditions
def square_area : ℕ := 36
def rectangle_width := (square_side : ℕ) (h : square_area = square_side * square_side) : ℕ := square_side
def rectangle_length := (width : ℕ) : ℕ := 3 * width

--- State the theorem using the defined conditions
theorem area_of_rectangle (square_side : ℕ) 
  (h1 : square_area = square_side * square_side)
  (width := rectangle_width square_side h1)
  (length := rectangle_length width) :
  width * length = 108 := by
    sorry

end area_of_rectangle_l576_576153


namespace min_troublemakers_29_l576_576504

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l576_576504


namespace min_troublemakers_l576_576511

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l576_576511


namespace product_probability_l576_576414

/-- The set of integers from 1 to 29. -/
def S : Finset ℕ := Finset.range 30 \ {0}

/-- Two distinct integers x, y chosen from S should satisfy that their product is neither a multiple of 2 nor 3. -/
def neither_multiple_of_2_nor_3 (x y : ℕ) : Prop := x ≠ y ∧ (x * y) % 2 ≠ 0 ∧ (x * y) % 3 ≠ 0

/-- The probability that the product of two distinct integers chosen randomly from {1, 2, ..., 29} is neither a multiple of 2 nor 3 is 45/406. -/
theorem product_probability :
  (Finset.card (S.filter (λ x, ∀ y ∈ S, neither_multiple_of_2_nor_3 x y)).powerset.card : ℚ) / (S.card.choose 2) = 45 / 406 := 
sorry

end product_probability_l576_576414


namespace rectangle_area_l576_576156

theorem rectangle_area (square_area : ℝ) (rectangle_length_ratio : ℝ) (square_area_eq : square_area = 36)
  (rectangle_length_ratio_eq : rectangle_length_ratio = 3) :
  ∃ (rectangle_area : ℝ), rectangle_area = 108 :=
by
  -- Extract the side length of the square from its area
  let side_length := real.sqrt square_area
  have side_length_eq : side_length = 6, from calc
    side_length = real.sqrt 36 : by rw [square_area_eq]
    ... = 6 : real.sqrt_eq 6 (by norm_num)
  -- Calculate the rectangle's width
  let width := side_length
  -- Calculate the rectangle's length
  let length := rectangle_length_ratio * width
  -- Calculate the area of the rectangle
  let area := width * length
  -- Prove the area is 108
  use area
  calc area
    = 6 * (3 * 6) : by rw [side_length_eq, rectangle_length_ratio_eq]
    ... = 108 : by norm_num

end rectangle_area_l576_576156


namespace area_of_rectangle_l576_576152

--- Define the problem's conditions
def square_area : ℕ := 36
def rectangle_width := (square_side : ℕ) (h : square_area = square_side * square_side) : ℕ := square_side
def rectangle_length := (width : ℕ) : ℕ := 3 * width

--- State the theorem using the defined conditions
theorem area_of_rectangle (square_side : ℕ) 
  (h1 : square_area = square_side * square_side)
  (width := rectangle_width square_side h1)
  (length := rectangle_length width) :
  width * length = 108 := by
    sorry

end area_of_rectangle_l576_576152


namespace count_squares_3x3_grid_count_squares_5x5_grid_l576_576108

/-- Define a mathematical problem: 
  Prove that the number of squares with all four vertices on the dots in a 3x3 grid is 4.
  Prove that the number of squares with all four vertices on the dots in a 5x5 grid is 50.
-/

def num_squares_3x3 : Nat := 4
def num_squares_5x5 : Nat := 50

theorem count_squares_3x3_grid : 
  ∀ (grid_size : Nat), grid_size = 3 → (∃ (dots_on_square : Bool), ∀ (distance_between_dots : Real), (dots_on_square = true → num_squares_3x3 = 4)) := 
by 
  intros grid_size h1
  exists true
  intros distance_between_dots 
  sorry

theorem count_squares_5x5_grid : 
  ∀ (grid_size : Nat), grid_size = 5 → (∃ (dots_on_square : Bool), ∀ (distance_between_dots : Real), (dots_on_square = true → num_squares_5x5 = 50)) :=
by 
  intros grid_size h1
  exists true
  intros distance_between_dots 
  sorry

end count_squares_3x3_grid_count_squares_5x5_grid_l576_576108


namespace mother_age_twice_xiaoming_in_18_years_l576_576556

-- Definitions based on conditions
def xiaoming_age_now : ℕ := 6
def mother_age_now : ℕ := 30

theorem mother_age_twice_xiaoming_in_18_years : 
    ∀ (n : ℕ), xiaoming_age_now + n = 24 → mother_age_now + n = 2 * (xiaoming_age_now + n) → n = 18 :=
by
  intro n hn hm
  sorry

end mother_age_twice_xiaoming_in_18_years_l576_576556


namespace area_of_rectangle_is_108_l576_576143

theorem area_of_rectangle_is_108 (s w l : ℕ) (h₁ : s * s = 36) (h₂ : w = s) (h₃ : l = 3 * w) : w * l = 108 :=
by
  -- This is a placeholder for a detailed proof.
  sorry

end area_of_rectangle_is_108_l576_576143


namespace quadratic_root_ratio_l576_576765

theorem quadratic_root_ratio (x y : ℂ) (h1 : x^2 + (1 + complex.i) * x - 6 + 3 * complex.i = 0)
  (h2 : y^2 + (1 + complex.i) * y - 6 + 3 * complex.i = 0) (hx_real : x.im = 0) :
  (x / y) = (-6 / 5) - (3 / 5) * complex.i :=
by
  sorry

end quadratic_root_ratio_l576_576765


namespace player_holds_13_cards_l576_576608

theorem player_holds_13_cards :
  ∃ (S : ℕ),
    (S + 6 = 7) ∧
    (2 * S + 4 * S = 6) ∧
    (S + 2 * S + 4 * S + 6 = 13) :=
by
  use 1
  sorry

end player_holds_13_cards_l576_576608


namespace constant_term_in_binomial_expansion_const_l576_576426

theorem constant_term_in_binomial_expansion_const (x : ℝ) : 
  let A := x^(1/3)
      B := x^(-1/2)
  in ∃ k : ℕ, (by have h : (15 - k) * (1/3 : ℝ) + k * (-1/2 : ℝ) = (0 : ℝ) := by normalize; exact h) ∧ k + 1 = 7 := sorry

end constant_term_in_binomial_expansion_const_l576_576426


namespace mina_pass_mark_l576_576022

theorem mina_pass_mark (total_problems : ℕ) (arithmetic_problems : ℕ) (algebra_problems : ℕ) (trigonometry_problems : ℕ)
  (correct_arithmetic_percentage : ℚ) (correct_algebra_percentage : ℚ) (correct_trigonometry_percentage : ℚ)
  (pass_percentage : ℚ)
  (total_correct : ℕ) (additional_for_pass : ℕ) :
  total_problems = 80 ∧
  arithmetic_problems = 20 ∧
  algebra_problems = 35 ∧
  trigonometry_problems = 25 ∧
  correct_arithmetic_percentage = 0.75 ∧
  correct_algebra_percentage = 0.30 ∧
  correct_trigonometry_percentage = 0.70 ∧
  pass_percentage = 0.55 ∧
  total_correct = (correct_arithmetic_percentage * arithmetic_problems).to_nat + (correct_algebra_percentage * algebra_problems).to_nat + (correct_trigonometry_percentage * trigonometry_problems).to_nat ∧
  additional_for_pass = ((pass_percentage * total_problems).to_nat + 1) - total_correct
  → additional_for_pass = 1 :=
by
  sorry

end mina_pass_mark_l576_576022


namespace F_monotonicity_g_condition_l576_576728

noncomputable def f (x : ℝ) : ℝ := Real.ln x
noncomputable def F (a x : ℝ) : ℝ := (3 / 2) * x^2 - (6 + a) * x + 2 * a * f x
noncomputable def g (x : ℝ) : ℝ := f x / (f x).derivative

variables {a x k x1 x2 : ℝ}

theorem F_monotonicity (a_pos : a > 0) :
  (∀ x, if a > 6 then (x < 2 ∨ x > a / 3) → 0 < x → (F a x) = F a x 
      else if a = 6 then 0 < x → (F a x) = F a x 
      else if 0 < a < 6 then 
        (if x < a / 3 ∨ x > 2 then 0 < x → (F a x) = F a x
         else a / 3 < x < 2 → (F a x) = F a x)) := sorry

theorem g_condition {k_pos : 0 < k} (intersection1 intersection2 : g x1 = y1 ∧ g x2 = y2) 
  (lt_x1x2 : x1 < x2): x1 < 1 / k ∧ 1 / k < x2 := sorry

end F_monotonicity_g_condition_l576_576728


namespace min_troublemakers_in_class_l576_576459

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l576_576459


namespace train_cross_platform_length_l576_576992

open Real

noncomputable def speed_kmph_to_mps (kmph : ℝ) : ℝ := kmph * 1000 / 3600

theorem train_cross_platform_length (v_kmph : ℝ) (t_platform t_man : ℝ) :
  v_kmph = 72 → t_platform = 32 → t_man = 18 →
  let v_mps := speed_kmph_to_mps v_kmph in
  let L := v_mps * t_man in
  (v_mps * t_platform) - L = 280 :=
by {
  sorry
}

end train_cross_platform_length_l576_576992


namespace quadratic_root_nature_real_and_equal_l576_576186

theorem quadratic_root_nature_real_and_equal :
  ∀ (x : ℝ), x ^ 2 + 4 * x * real.sqrt 2 + 8 = 0 → 
    (∃ r : ℝ, (x = r ∧ r ^ 2 + 4 * r * real.sqrt 2 + 8 = 0)) :=
by
  sorry

end quadratic_root_nature_real_and_equal_l576_576186


namespace measure_angle_PSR_is_40_l576_576622

noncomputable def isosceles_triangle (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : Triangle := sorry
noncomputable def square (D R S T : Point) : Square := sorry
noncomputable def angle (A B C : Point) (θ : ℝ) : Prop := sorry

def angle_PQR (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : ℝ := sorry
def angle_PRQ (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : ℝ := sorry

theorem measure_angle_PSR_is_40
  (P Q R S T D : Point)
  (PQ PR : ℝ)
  (hPQ_PR : PQ = PR)
  (hQ_eq_D : Q = D)
  (hQPS : angle P Q S 100)
  (hDRST_square : square D R S T) : angle P S R 40 :=
by
  -- Proof omitted for brevity
  sorry

end measure_angle_PSR_is_40_l576_576622


namespace enclosed_region_area_correct_l576_576204

noncomputable def area_of_enclosed_region : ℝ :=
  let f (x y : ℝ) := abs (x - 70) + abs y - abs (x / 5)
  let region := { p : ℝ × ℝ | f p.1 p.2 = 0 }
  -- Calculate the area of the closed region bounded by the graph
  let area := sorry -- Here we will need to calculate the area.
  in area

theorem enclosed_region_area_correct : area_of_enclosed_region = 340.4199 :=
sorry -- This is where the actual proof would go.

end enclosed_region_area_correct_l576_576204


namespace students_without_glasses_l576_576956

theorem students_without_glasses (total_students : ℕ) (perc_glasses : ℕ) (students_with_glasses students_without_glasses : ℕ) 
  (h1 : total_students = 325) (h2 : perc_glasses = 40) (h3 : students_with_glasses = perc_glasses * total_students / 100)
  (h4 : students_without_glasses = total_students - students_with_glasses) : students_without_glasses = 195 := 
by
  sorry

end students_without_glasses_l576_576956


namespace A_wins_probability_is_3_over_4_l576_576119

def parity (n : ℕ) : Bool := n % 2 == 0

def number_of_dice_outcomes : ℕ := 36

def same_parity_outcome : ℕ := 18

def probability_A_wins : ℕ → ℕ → ℕ → ℚ
| total_outcomes, same_parity, different_parity =>
  (same_parity / total_outcomes : ℚ) * 1 + (different_parity / total_outcomes : ℚ) * (1 / 2)

theorem A_wins_probability_is_3_over_4 :
  probability_A_wins number_of_dice_outcomes same_parity_outcome (number_of_dice_outcomes - same_parity_outcome) = 3/4 :=
by
  sorry

end A_wins_probability_is_3_over_4_l576_576119


namespace intersection_M_N_l576_576286

-- Definitions for the sets M and N based on the given conditions
def M : Set ℝ := { x | x < 2 }
def N : Set ℝ := { x | x^2 - 3 * x - 4 ≤ 0 }

-- The statement we need to prove
theorem intersection_M_N : M ∩ N = { x | -1 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end intersection_M_N_l576_576286


namespace tan_alpha_eq_plus_minus_one_l576_576237

variable (m : ℝ)

theorem tan_alpha_eq_plus_minus_one 
  (hP : ∃ α, P = (-sqrt 3, m) ∧ sin α = (sqrt 2 / 4) * m) : 
  ∃ α, tan α = 1 ∨ tan α = -1 :=
by
  sorry

end tan_alpha_eq_plus_minus_one_l576_576237


namespace range_of_a_l576_576734

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

theorem range_of_a (a : ℝ) : 
  (∀ x, deriv (f a) x ≠ 0) ↔ -3 ≤ a ∧ a ≤ 6 :=
by
  have h : ∀ x, deriv (f a) x = 3 * x^2 + 2 * a * x + (a + 6) :=
    by funext; simp [f, deriv];
  sorry

end range_of_a_l576_576734


namespace tan_tan_solution_count_l576_576754

noncomputable def arctan_942 := Real.arctan 942

theorem tan_tan_solution_count : 
  ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ arctan_942 → ∃! x : ℝ, 0 ≤ x ∧ x ≤ arctan_942 ∧ tan x = tan (tan x) ∧ x ≠ 0 :=
sorry

end tan_tan_solution_count_l576_576754


namespace find_plane_Q_l576_576245

open Real

noncomputable def plane_equation (a b c d : ℝ) : ℝ × ℝ × ℝ → ℝ :=
  λ p, a * p.1 + b * p.2 + c * p.3 + d

def distance_point_to_plane (plane : ℝ × ℝ × ℝ → ℝ) (p : ℝ × ℝ × ℝ) : ℝ :=
  abs (plane p) / sqrt ((plane (1, 0, 0))^2 + (plane (0, 1, 0))^2 + (plane (0, 0, 1))^2)

theorem find_plane_Q :
  (∃ a b c d : ℝ, 
    (∀ x y z : ℝ, 3 * x - y + 2 * z = 6 → a * x + b * y + c * z + d = 0) ∧
    (∀ x y z : ℝ, x + 3 * y - z = 2 → a * x + b * y + c * z + d = 0) ∧
    distance_point_to_plane (plane_equation a b c d) (1, -1, 2) = 3 / sqrt 2 ∧
    distance_point_to_plane (plane_equation a b c d) (0, 1, 0) = 1 / sqrt 6 ∧
    plane_equation a b c d = λ p, (p.1 - 7 * p.2 + 4 * p.3 - 2) :=
  sorry

end find_plane_Q_l576_576245


namespace minimum_trees_with_at_least_three_types_l576_576808

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end minimum_trees_with_at_least_three_types_l576_576808


namespace roots_sum_eq_coprime_187_53_value_of_m_plus_n_l576_576868

-- Define the polynomial P(x)
def P (x : ℂ) := x^20 - 7*x^3 + 1

-- State the main theorem
theorem roots_sum_eq : 
  let r : Fin 20 → ℂ := λ i, ( (Complex.solve_deg20 P).val i ) in
  ( ∑ i : Fin 20, 1 / (r i)^2 + 1 ) = (187 : ℚ) / (53 : ℚ) :=
sorry

-- Define the coprimality function for completeness
def coprime (a b : ℕ) := ∀ d : ℕ, d ∣ a ∧ d ∣ b → d = 1

theorem coprime_187_53 : coprime 187 53 :=
sorry

theorem value_of_m_plus_n : 
  let m := 187 in
  let n := 53 in
  coprime m n → m + n = 240 :=
by
  intros h
  rw [Nat.add_comm]
  exact h

end roots_sum_eq_coprime_187_53_value_of_m_plus_n_l576_576868


namespace distance_between_parallel_lines_l576_576261

noncomputable def line1 (x y : ℝ) : Prop := 2 * x + y - 2 = 0
noncomputable def line2 (m x y : ℝ) : Prop := 4 * x + m * y + 6 = 0

theorem distance_between_parallel_lines (m : ℝ) (h : 2 * m - 4 = 0) : 
    let p := Real.sqrt (4 + 1) in 
    abs (-2 - 3) / p = Real.sqrt 5 :=
by
  -- Given m = 2 from 2m - 4 = 0
  have h_m : m = 2 := by linarith [h]
  -- Rewrite the second line using the value of m
  let line2' := 2 * x + y + 3
  -- Check that the two lines are parallel
  have parallel_check : (2 * x + y - 2 = 0) = (2 * x + y + 3 = 0) := by sorry
  -- Compute the distance
  let distance := abs (-2 - 3) / Real.sqrt (2^2 + 1^2)
  have distance_calc : distance = Real.sqrt 5 := by 
    calc
      abs (-2 - 3) / Real.sqrt (2^2 + 1^2) = abs (-5) / Real.sqrt 5 : by sorry
      ... = 5 / Real.sqrt 5 : by sorry
      ... = Real.sqrt 5 : by sorry
  -- Conclude the theorem
  exact distance_calc

end distance_between_parallel_lines_l576_576261


namespace line_has_one_common_point_with_parabola_l576_576917

theorem line_has_one_common_point_with_parabola :
  (∀ x y, (y = -6 * x - 7 ∨ x = -1) ↔ (y = 8 * x^2 + 10 * x + 1) → y = -1) :=
begin
  sorry,
end

end line_has_one_common_point_with_parabola_l576_576917


namespace min_trees_for_three_types_l576_576800

-- Define types and trees
inductive TreeType
| birch | spruce | pine | aspen
deriving Inhabited, DecidableEq

-- A grove with 100 trees of any of the four types.
structure Tree :=
(type : TreeType)

constant grove : List Tree
axiom grove_size : grove.length = 100

-- Condition: Among any 85 trees, there are trees of all four types.
axiom all_types_in_any_85 : ∀ (s : Finset Tree), s.card = 85 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, ∃ t4 ∈ s, 
      t1.type = TreeType.birch ∧
      t2.type = TreeType.spruce ∧
      t3.type = TreeType.pine ∧
      t4.type = TreeType.aspen

-- We need to show that at least 69 trees are needed to ensure at least 3 types.
theorem min_trees_for_three_types : 
  ∀ (s : Finset Tree), s.card = 69 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, 
      t1.type ≠ t2.type ∧ t2.type ≠ t3.type ∧ t1.type ≠ t3.type := 
sorry

end min_trees_for_three_types_l576_576800


namespace purely_imaginary_number_eq_l576_576007

theorem purely_imaginary_number_eq (z : ℂ) (a : ℝ) (i : ℂ) (h_imag : z.im = 0 ∧ z = 0 ∧ (3 - i) * z = a + i + i) :
  a = 1 / 3 :=
  sorry

end purely_imaginary_number_eq_l576_576007


namespace min_troublemakers_in_class_l576_576465

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l576_576465


namespace imag_part_of_z_l576_576726

def z : ℂ := (1 + 2 * complex.I) / complex.I

theorem imag_part_of_z : z.im = -1 :=
by
  sorry

end imag_part_of_z_l576_576726


namespace sequence_is_increasing_and_natural_l576_576920

noncomputable def sequence (a : ℕ) (q : ℝ) (h : q > 1) : ℕ → ℝ
| 0     := a
| 1     := a * q
| 2     := a * q^2
| (n+3) := if (n+3) % 2 = 0 then 2 * sequence (n+2) - sequence (n+1) 
                              else (sequence (n+2) * sequence (n)) / sequence (n+1)

theorem sequence_is_increasing_and_natural (a : ℕ) (q : ℝ) (h : q > 1) :
  ∀ n, n ≥ 3 → ((sequence a q h) n > (sequence a q h) (n-1)) ∧ 
               (sequence a q h) n ∈ ℕ :=
sorry

end sequence_is_increasing_and_natural_l576_576920


namespace min_troublemakers_in_class_l576_576464

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l576_576464


namespace smallest_positive_t_l576_576219

theorem smallest_positive_t (x_1 x_2 x_3 x_4 x_5 t : ℝ) :
  (x_1 + x_3 = 2 * t * x_2) →
  (x_2 + x_4 = 2 * t * x_3) →
  (x_3 + x_5 = 2 * t * x_4) →
  (0 ≤ x_1) →
  (0 ≤ x_2) →
  (0 ≤ x_3) →
  (0 ≤ x_4) →
  (0 ≤ x_5) →
  (x_1 ≠ 0 ∨ x_2 ≠ 0 ∨ x_3 ≠ 0 ∨ x_4 ≠ 0 ∨ x_5 ≠ 0) →
  t = 1 / Real.sqrt 2 → 
  ∃ t, (0 < t) ∧ (x_1 + x_3 = 2 * t * x_2) ∧ (x_2 + x_4 = 2 * t * x_3) ∧ (x_3 + x_5 = 2 * t * x_4)
:=
sorry

end smallest_positive_t_l576_576219


namespace existence_of_solution_l576_576670

theorem existence_of_solution (a : ℝ) :
  a ∈ Icc (-18 : ℝ) 18 → ∃ (x y b : ℝ), 
    (x^2 + y^2 + 2 * a * (a - x - y) = 64) ∧ 
    (y = 8 * sin (x - 2 * b) - 6 * cos (x - 2 * b)) :=
by
  intros ha
  sorry

end existence_of_solution_l576_576670


namespace solution_set_of_inequality_l576_576945

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 2) : 
  ((2 * x) / (x - 2) ≤ 1) ↔ (-2 ≤ x ∧ x < 2) :=
sorry

end solution_set_of_inequality_l576_576945


namespace proof_inequality_l576_576250

noncomputable def problem (a b c d : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1 → a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d

theorem proof_inequality (a b c d : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end proof_inequality_l576_576250


namespace rhombus_side_length_l576_576940
-- Importing the necessary libraries

-- Defining the problem conditions as hypotheses
theorem rhombus_side_length :
  (∃(S : Type)(A B C D M N K L F P R Q : S),
    ∃(d1 d2 h : ℝ),
    (∀(x : S), {x = midpoint A B} ∨ {x = midpoint B C} ∨ 
               {x = midpoint C D} ∨ {x = midpoint D A}) ∧
    (angle_between_plane_and_face (face_S_A_BC) (plane_A_B_C_D) = 60) ∧
    (radius_inscribed_circle (rhombus A B C D) = 2.4) ∧
    (volume_parallelepiped (rectangle M N K L) = 12 * sqrt 3)
  ) →
  (side_length (rhombus A B C D) = 5) :=
begin
  sorry
end

end rhombus_side_length_l576_576940


namespace measure_of_angle_CE_length_of_AB_l576_576210

axiom A : ℝ := 1
axiom theta : ℝ := Real.pi / 10
axiom B_midpoint_of_AE : ∃ (B E : ℝ), B = 1/2 * E
axiom angle_ECB_eq_angle_CAB : ∃ (angle_ECB angle_CAB : ℝ), angle_ECB = angle_CAB
axiom angle_C_eq_angle_B : ∃ (angle_C angle_B : ℝ), angle_C = angle_B
axiom C_tangent_to_circle : ∃ (C radius : ℝ), radius > 0

theorem measure_of_angle_CE : ∃ (angle_CE : ℝ), angle_CE = 2 * theta :=
by
  use 2 * theta
  rw [theta]
  simp
  sorry

theorem length_of_AB : ∃ (AB : ℝ), AB = (Real.sqrt 5 - 1) / 2 :=
by
  use (Real.sqrt 5 - 1) / 2
  sorry

end measure_of_angle_CE_length_of_AB_l576_576210


namespace weight_of_each_bag_l576_576519

theorem weight_of_each_bag 
  (bags_per_trip : ℕ)
  (number_of_trips : ℕ)
  (total_weight : ℕ)
  (h_bags_per_trip : bags_per_trip = 10)
  (h_number_of_trips : number_of_trips = 20)
  (h_total_weight : total_weight = 10000) : 
  total_weight / (bags_per_trip * number_of_trips) = 50 :=
by
  -- We assume the conditions given directly
  rw [h_bags_per_trip, h_number_of_trips, h_total_weight]
  simp
  exact Nat.div_eq_of_eq_mul (Nat.eq_of_mul_eq_mul_right _ _ _ (by norm_num))

end weight_of_each_bag_l576_576519


namespace tank_fill_time_l576_576565
noncomputable def time_to_fill_tank (rate_A rate_B : ℝ) : ℝ :=
  let R_A := rate_A
  let R_B := rate_B
  let R_combined := R_A + R_B
  1 / R_combined

theorem tank_fill_time (h1 : ∀ t : ℝ, t > 0 → pipe_fill_rate t = 1 / t) 
  (h2 : pipe_fill_rate 6 = 1 / 6) : time_to_fill_tank (pipe_fill_rate 6) (pipe_fill_rate 3) = 2 := by
  sorry

end tank_fill_time_l576_576565


namespace mutually_exclusive_B_C_l576_576230

-- Define the events A, B, C
def event_A (x y : ℕ → Bool) : Prop := ¬(x 1 = false ∨ x 2 = false)
def event_B (x y : ℕ → Bool) : Prop := x 1 = false ∧ x 2 = false
def event_C (x y : ℕ → Bool) : Prop := ¬(x 1 = false ∧ x 2 = false)

-- Prove that event B and event C are mutually exclusive
theorem mutually_exclusive_B_C (x y : ℕ → Bool) :
  (event_B x y ∧ event_C x y) ↔ false := sorry

end mutually_exclusive_B_C_l576_576230


namespace min_troublemakers_in_class_l576_576466

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l576_576466


namespace find_x_value_l576_576102

theorem find_x_value :
  let a := 0.47 * 1442
  let b := 0.36 * 1412
  (a - b) + 63 = 232.42 :=
by 
  let a := 0.47 * 1442
  let b := 0.36 * 1412
  have ha : a = 677.74 := sorry,
  have hb : b = 508.32 := sorry,
  have h1 : a - b = 169.42 := sorry,
  have h2 : 169.42 + 63 = 232.42 := sorry,
  exact h2

end find_x_value_l576_576102


namespace rhombus_diagonals_perpendicular_l576_576032

theorem rhombus_diagonals_perpendicular (Q : Type) [Quadrilateral Q] [Rhombus Q] : ArePerpendicular (diagonals Q).fst (diagonals Q).snd :=
by
  sorry

end rhombus_diagonals_perpendicular_l576_576032


namespace part_a_exists_eight_consecutive_two_lucky_part_b_exists_twelve_consecutive_none_lucky_part_c_for_all_thirteen_consecutive_at_least_one_lucky_l576_576129

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_lucky (n : ℕ) : Prop :=
  digit_sum n % 7 = 0

theorem part_a_exists_eight_consecutive_two_lucky :
  ∃ (a : ℕ), ∀ i : ℕ, i < 8 → 
    is_lucky (a + i) ↔ (i = 0 ∨ i = 7) := 
sorry

theorem part_b_exists_twelve_consecutive_none_lucky :
  ∃ (a : ℕ), ∀ i : ℕ, i < 12 → 
    ¬ is_lucky (a + i) := 
sorry

theorem part_c_for_all_thirteen_consecutive_at_least_one_lucky :
  ∀ (a : ℕ), ∃ (i : ℕ), i < 13 ∧ 
    is_lucky (a + i) := 
sorry

end part_a_exists_eight_consecutive_two_lucky_part_b_exists_twelve_consecutive_none_lucky_part_c_for_all_thirteen_consecutive_at_least_one_lucky_l576_576129


namespace sum_of_x_coordinates_of_intersection_l576_576651

theorem sum_of_x_coordinates_of_intersection :
  let f := fun x : ℝ => x ^ 2
  let g := fun x : ℝ => 2 * x + 3
  let intersections := {x | f x = g x}
  (∑ x in intersections, x) = 2 :=
by
  sorry

end sum_of_x_coordinates_of_intersection_l576_576651


namespace bacteria_population_final_value_l576_576934

noncomputable def doubling_time := 6 -- minutes
noncomputable def total_time := 53.794705707972525 -- minutes
noncomputable def initial_population := 1000 -- bacteria

def calculate_final_population 
  (initial_population : ℕ) 
  (doubling_time : ℝ) 
  (total_time : ℝ) : ℝ :=
  initial_population * (2 ^ (total_time / doubling_time))

theorem bacteria_population_final_value : 
  calculate_final_population initial_population doubling_time total_time = 495451 := 
  sorry

end bacteria_population_final_value_l576_576934


namespace calculate_total_income_l576_576606

/-- Total income calculation proof for a person with given distributions and remaining amount -/
theorem calculate_total_income
  (I : ℝ) -- total income
  (leftover : ℝ := 40000) -- leftover amount after distribution and donation
  (c1_percentage : ℝ := 3 * 0.15) -- percentage given to children
  (c2_percentage : ℝ := 0.30) -- percentage given to wife
  (c3_percentage : ℝ := 0.05) -- percentage donated to orphan house
  (remaining_percentage : ℝ := 1 - (c1_percentage + c2_percentage)) -- remaining percentage after children and wife
  (R : ℝ := remaining_percentage * I) -- remaining amount after children and wife
  (donation : ℝ := c3_percentage * R) -- amount donated to orphan house)
  (left_amount : ℝ := R - donation) -- final remaining amount
  (income : ℝ := (leftover / (1 - remaining_percentage * (1 - c3_percentage)))) -- calculation of the actual income
  : I = income := sorry

end calculate_total_income_l576_576606


namespace exist_consecutive_nums_with_exact_primes_l576_576660

-- Define the interval size
def intervalSize := 2016

-- Define function to count the number of primes within a given interval
def S (n : ℕ) : ℕ :=
  ((n : ℕ) to (n + intervalSize - 1)).count prime

-- Statement of the problem
theorem exist_consecutive_nums_with_exact_primes : ∃ m : ℕ, S m = 16 := 
begin
  sorry
end

end exist_consecutive_nums_with_exact_primes_l576_576660


namespace regular_pentagonal_prism_diagonal_count_l576_576572

noncomputable def diagonal_count (n : ℕ) : ℕ := 
  if n = 5 then 10 else 0

theorem regular_pentagonal_prism_diagonal_count :
  diagonal_count 5 = 10 := 
  by
    sorry

end regular_pentagonal_prism_diagonal_count_l576_576572


namespace matrix_solution_property_l576_576656

theorem matrix_solution_property (N : Matrix (Fin 2) (Fin 2) ℝ) 
    (h : N = Matrix.of ![![2, 4], ![1, 4]]) :
    N ^ 4 - 5 * N ^ 3 + 9 * N ^ 2 - 5 * N = Matrix.of ![![6, 12], ![3, 6]] :=
by 
  sorry

end matrix_solution_property_l576_576656


namespace number_of_subsets_l576_576696

theorem number_of_subsets (a : ℝ) :
  ∃ M : set ℝ, (M = {x | x^2 - 3 * x - a^2 + 2 = 0}) ∧ fintype.card (set.powerset M) = 4 := by
  sorry

end number_of_subsets_l576_576696


namespace minimum_liars_l576_576476

noncomputable def minimum_troublemakers (n : ℕ) := 10

theorem minimum_liars (n : ℕ) (students : ℕ → bool) 
  (seated_at_round_table : ∀ i, students (i % 29) = tt ∨ students (i % 29) = ff)
  (one_liar_next_to : ∀ i, students i = tt → (students ((i + 1) % 29) = ff ∨ students ((i + 28) % 29) = ff))
  (two_liars_next_to : ∀ i, students i = ff → ((students ((i + 1) % 29) = ff) ∧ students ((i + 28) % 29) = ff)) :
  ∃ m, m = minimum_troublemakers 29 ∧ (m ≤ n) :=
by
  let liars_count := 10
  have : liars_count = 10 := rfl
  use liars_count
  exact ⟨rfl, sorry⟩

end minimum_liars_l576_576476


namespace problem1_l576_576568

theorem problem1 : (π - 2023)^0 - 3 * Real.tan(π/6) + |1 - Real.sqrt 3| = 0 := 
by
  have h1 : (π - 2023) ^ 0 = 1 := by sorry
  have h2 : Real.tan (π / 6) = Real.sqrt 3 / 3 := by sorry
  have h3 : |1 - Real.sqrt 3| = Real.sqrt 3 - 1 := by sorry
  sorry

end problem1_l576_576568


namespace acronym_XYZ_length_l576_576910

theorem acronym_XYZ_length :
  let X_segments := 2 * real.sqrt 2
  let Y_segments := 4
  let Z_segments := 2 + real.sqrt 2
  let circular_hole := real.pi
  (X_segments + Y_segments + Z_segments + circular_hole) = 6 + 3 * real.sqrt 2 + real.pi :=
by
  sorry

end acronym_XYZ_length_l576_576910


namespace number_of_disjoint_subsets_mod_1000_l576_576002

theorem number_of_disjoint_subsets_mod_1000 :
  let T := {1, 2, 3, ..., 15}
  let m := (4^15 - 3 * 3^15 + 3 * 2^15 - 1) / 6
  m % 1000 = 567 :=
  by
    let T := {1, 2, 3, ..., 15}
    let m := (4^15 - 3 * 3^15 + 3 * 2^15 - 1) / 6
    show m % 1000 = 567, by sorry

end number_of_disjoint_subsets_mod_1000_l576_576002


namespace geom_seq_factorial_divisible_by_one_billion_l576_576185

theorem geom_seq_factorial_divisible_by_one_billion (n : ℕ) (a1 a2 : ℚ) (h_a1 : a1 = 5/8) (h_a2 : a2 = 25) :
  (∃ n, (let r := (25 / (5/8)) in (r^(n-1) * (5 / 8)) * n!) ∣ 10^9) ∧ n = 10 :=
sorry

end geom_seq_factorial_divisible_by_one_billion_l576_576185


namespace angle_MHB_l576_576318

noncomputable def geometry_problem (A B C H M : Point) : Prop :=
  ∠A = 80 ∧ ∠B = 70 ∧ ∠C = 30 ∧
  is_altitude B H A C ∧
  is_median_and_angle_bisector C M A B

theorem angle_MHB (A B C H M : Point) (h : geometry_problem A B C H M) : ∠MHB = 35 :=
by sorry

end angle_MHB_l576_576318


namespace pickles_per_jar_correct_l576_576394

variable (jars : ℕ) (cucumbers : ℕ) (vinegar_initial : ℕ) (vinegar_left : ℕ)
variable (pickles_per_cucumber : ℕ) (vinegar_per_jar : ℕ) (total_pickles : ℕ) (pickles_per_jar : ℕ)

-- Conditions
axiom cond1 : jars = 4
axiom cond2 : cucumbers = 10
axiom cond3 : vinegar_initial = 100
axiom cond4 : vinegar_left = 60
axiom cond5 : pickles_per_cucumber = 6
axiom cond6 : vinegar_per_jar = 10

-- Derived values based on conditions
def vinegar_used := vinegar_initial - vinegar_left
def total_jars_filled := vinegar_used / vinegar_per_jar
def total_pickles := cucumbers * pickles_per_cucumber
def pickles_per_jar := total_pickles / jars

-- Theorem statement
theorem pickles_per_jar_correct : pickles_per_jar = 15 :=
by
  sorry

end pickles_per_jar_correct_l576_576394


namespace sum_of_geometric_series_l576_576370

theorem sum_of_geometric_series :
  let z : ℕ → ℂ := sorry,
  (z 0 = 1) →
  (z 2013 = (1 / 2013 ^ 2013)) →
  (∀ r : ℂ, (r ^ 2013 = (1 / 2013 ^ 2013)) →
    ((∑ k in finset.range 2013, 1 / (1 - (r * complex.exp (complex.I * (2 * k * complex.pi / 2013))))) 
    = (2013 ^ 2014 / (2013 ^ 2013 - 1)))) :=
sorry

end sum_of_geometric_series_l576_576370


namespace f_2002_l576_576567

noncomputable def f : ℕ → ℝ := sorry

axiom f_condition (n : ℕ) (h : n > 1) : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

axiom f_2001 : f 2001 = 1

theorem f_2002 : f 2002 = 2 :=
  sorry

end f_2002_l576_576567


namespace minimum_trees_with_at_least_three_types_l576_576811

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end minimum_trees_with_at_least_three_types_l576_576811


namespace least_positive_three_digit_multiple_of_9_l576_576530

   theorem least_positive_three_digit_multiple_of_9 : ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ 9 ∣ n ∧ n = 108 :=
   by
     sorry
   
end least_positive_three_digit_multiple_of_9_l576_576530


namespace min_liars_needed_l576_576469

namespace Problem

/-
  We introduce our problem using Lean definitions and statements.
-/

def students_sitting_at_table : ℕ := 29

structure Student :=
(is_liar : Bool)

def student_truth_statement (s: ℕ) (next1: Student) (next2: Student) : Bool :=
  if s then (next1.is_liar && !next2.is_liar) || (!next1.is_liar && next2.is_liar)
  else next1.is_liar && next2.is_liar

def minimum_liars_class (sitting_students: ℕ) : ℕ :=
  sitting_students / 3 + (if sitting_students % 3 = 0 then 0 else 1)

theorem min_liars_needed (students : List Student) (H_total : students.length = students_sitting_at_table)
  (H_truth_teller_claim : ∀ (i : ℕ), i < students.length → student_truth_statement i (students.get ⟨(i+1) % students.length, sorry⟩) (students.get ⟨(i+2) % students.length, sorry⟩)) :
  (students.count (λ s => s.is_liar)) ≥ minimum_liars_class students_sitting_at_table :=
sorry

end Problem

end min_liars_needed_l576_576469


namespace min_liars_needed_l576_576468

namespace Problem

/-
  We introduce our problem using Lean definitions and statements.
-/

def students_sitting_at_table : ℕ := 29

structure Student :=
(is_liar : Bool)

def student_truth_statement (s: ℕ) (next1: Student) (next2: Student) : Bool :=
  if s then (next1.is_liar && !next2.is_liar) || (!next1.is_liar && next2.is_liar)
  else next1.is_liar && next2.is_liar

def minimum_liars_class (sitting_students: ℕ) : ℕ :=
  sitting_students / 3 + (if sitting_students % 3 = 0 then 0 else 1)

theorem min_liars_needed (students : List Student) (H_total : students.length = students_sitting_at_table)
  (H_truth_teller_claim : ∀ (i : ℕ), i < students.length → student_truth_statement i (students.get ⟨(i+1) % students.length, sorry⟩) (students.get ⟨(i+2) % students.length, sorry⟩)) :
  (students.count (λ s => s.is_liar)) ≥ minimum_liars_class students_sitting_at_table :=
sorry

end Problem

end min_liars_needed_l576_576468


namespace angle_BAH_eq_OAC_l576_576971

-- Point definitions
variables {A B C O H : Type}

-- Triangle ABC is acute-angled
variable [triangle ABC]

-- O is the center of the circumcircle of triangle ABC
def center_circumcircle (A B C O : Type) : Prop :=
  true -- placeholder, define the circumcenter property.

-- AH is the altitude from A to BC
def is_altitude (A H B C : Type) : Prop :=
  true -- placeholder, define the altitude property.

-- Angle definition using general terms.
def angle (x y z : Type) : Type :=
  sorry -- placeholder, define angle measure.

-- Proving the required equality
theorem angle_BAH_eq_OAC (A B C O H : Type)
  [h₁ : triangle ABC]
  [h₂ : center_circumcircle A B C O]
  [h₃ : is_altitude A H B C] :
  angle B A H = angle O A C :=
sorry

end angle_BAH_eq_OAC_l576_576971


namespace value_of_a_l576_576307

noncomputable def A : Set ℝ := { x | abs x = 1 }
def B (a : ℝ) : Set ℝ := { x | a * x = 1 }
def is_superset (A B : Set ℝ) : Prop := ∀ x, x ∈ B → x ∈ A

theorem value_of_a (a : ℝ) (h : is_superset A (B a)) : a = 1 ∨ a = 0 ∨ a = -1 :=
  sorry

end value_of_a_l576_576307


namespace inradius_of_triangle_l576_576440

theorem inradius_of_triangle (p A : ℝ) (h₁ : p = 32) (h₂ : A = 56) : 
  ∃ r : ℝ, r = 3.5 ∧ r * p / 2 = A :=
by
  use 3.5
  split
  · -- r = 3.5
    refl
  · --  r * p / 2 = A
    rw [h₁, h₂]
    have : 3.5 * 32 / 2 = 56 := by norm_num
    exact this

end inradius_of_triangle_l576_576440


namespace sectors_containing_all_numbers_l576_576589

theorem sectors_containing_all_numbers (n : ℕ) (h : 0 < n) :
  ∃ (s : Finset (Fin (2 * n))), (s.card = n) ∧ (∀ i : Fin n, ∃ j : Fin (2 * n), j ∈ s ∧ (j.val % n) + 1 = i.val) :=
  sorry

end sectors_containing_all_numbers_l576_576589


namespace find_AH_AD_plus_BH_BE_plus_CH_CF_l576_576241

variables (A B C H D E F : Point) 
variables (a b c : ℝ)
variables [Triangle Δ : Triangle A B C]
variables [AcuteTriangle : isAcute Δ]
variables [AD BE CF : Altitude Δ]
variables [InterH : isOrthocenter H A B C]
variables [EqualSides : (BC = a) (CA = b) (AB = c)]

theorem find_AH_AD_plus_BH_BE_plus_CH_CF :
  let AH := distance A H,
      BH := distance B H,
      CH := distance C H,
      AD := distance A D,
      BE := distance B E,
      CF := distance C F
  in AH * AD + BH * BE + CH * CF = (a^2 + b^2 + c^2) / 2 :=
sorry

end find_AH_AD_plus_BH_BE_plus_CH_CF_l576_576241


namespace possible_values_of_a_l576_576518

theorem possible_values_of_a :
  ∃ (a b c : ℤ), ∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c) → a = 3 ∨ a = 7 :=
by
  sorry

end possible_values_of_a_l576_576518


namespace avg_two_ab_l576_576049

-- Defining the weights and conditions
variables (A B C : ℕ)

-- The conditions provided in the problem
def avg_three (A B C : ℕ) := (A + B + C) / 3 = 45
def avg_two_bc (B C : ℕ) := (B + C) / 2 = 43
def weight_b (B : ℕ) := B = 35

-- The target proof statement
theorem avg_two_ab (A B C : ℕ) (h1 : avg_three A B C) (h2 : avg_two_bc B C) (h3 : weight_b B) : (A + B) / 2 = 42 := 
sorry

end avg_two_ab_l576_576049


namespace magazines_cover_area_l576_576952

theorem magazines_cover_area (S : ℝ) (n : ℕ) (h_n_15 : n = 15) (h_cover : ∀ m ≤ n, ∃(Sm:ℝ), (Sm ≥ (m : ℝ) / n * S) ) :
  ∃ k : ℕ, k = n - 7 ∧ ∃ (Sk : ℝ), (Sk ≥ 8/15 * S) := 
by
  sorry

end magazines_cover_area_l576_576952


namespace initial_chocolate_amount_l576_576088

-- Define the problem conditions

def initial_dough (d : ℕ) := d = 36
def left_over_chocolate (lo_choc : ℕ) := lo_choc = 4
def chocolate_percentage (p : ℚ) := p = 0.20
def total_weight (d : ℕ) (c_choc : ℕ) := d + c_choc - 4
def chocolate_used (c_choc : ℕ) (lo_choc : ℕ) := c_choc - lo_choc

-- The main proof goal
theorem initial_chocolate_amount (d : ℕ) (lo_choc : ℕ) (p : ℚ) (C : ℕ) :
  initial_dough d → left_over_chocolate lo_choc → chocolate_percentage p →
  p * (total_weight d C) = chocolate_used C lo_choc → C = 13 :=
by
  intros hd hlc hp h
  sorry

end initial_chocolate_amount_l576_576088


namespace number_of_subsets_summing_to_2008_l576_576214

def subsets_with_sum (s : Finset ℕ) (n : ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ t, t.sum id = n)

theorem number_of_subsets_summing_to_2008 :
  (subsets_with_sum (Finset.range 64) 2008).card = 6 := sorry

end number_of_subsets_summing_to_2008_l576_576214


namespace determinant_example_l576_576647

def det_2x2 (a b c d : ℤ) : ℤ := a * d - b * c

theorem determinant_example : det_2x2 7 (-2) (-3) 6 = 36 := by
  sorry

end determinant_example_l576_576647


namespace rigid_motion_transformations_l576_576919

-- Define a structure to represent the line and the figure.
structure Line (α : Type) :=
  (l : α) -- Line l
  (figure : α → α) -- Function describing the figure on line l

def transforms_onto_itself {α : Type} (t : α → α) (figure : α → α) : Prop :=
  ∀ x, figure (t x) = figure x

theorem rigid_motion_transformations (α : Type) 
(line_figure : Line α)
(rotation : α → α)
(translation : α → α)
(reflection_across_line : α → α)
(reflection_perpendicular_line : α → α) :
transforms_onto_itself rotation line_figure.figure ∧
transforms_onto_itself translation line_figure.figure ∧
¬transforms_onto_itself reflection_across_line line_figure.figure ∧
¬transforms_onto_itself reflection_perpendicular_line line_figure.figure →
2 = (if transforms_onto_itself rotation line_figure.figure then 1 else 0) +
    (if transforms_onto_itself translation line_figure.figure then 1 else 0) +
    (if transforms_onto_itself reflection_across_line line_figure.figure then 1 else 0) +
    (if transforms_onto_itself reflection_perpendicular_line line_figure.figure then 1 else 0) :=
sorry -- Proof not required

end rigid_motion_transformations_l576_576919


namespace solution_set_of_quadratic_inequality_l576_576446

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 3} :=
sorry

end solution_set_of_quadratic_inequality_l576_576446


namespace rectangle_area_l576_576158

theorem rectangle_area (square_area : ℝ) (rectangle_length_ratio : ℝ) (square_area_eq : square_area = 36)
  (rectangle_length_ratio_eq : rectangle_length_ratio = 3) :
  ∃ (rectangle_area : ℝ), rectangle_area = 108 :=
by
  -- Extract the side length of the square from its area
  let side_length := real.sqrt square_area
  have side_length_eq : side_length = 6, from calc
    side_length = real.sqrt 36 : by rw [square_area_eq]
    ... = 6 : real.sqrt_eq 6 (by norm_num)
  -- Calculate the rectangle's width
  let width := side_length
  -- Calculate the rectangle's length
  let length := rectangle_length_ratio * width
  -- Calculate the area of the rectangle
  let area := width * length
  -- Prove the area is 108
  use area
  calc area
    = 6 * (3 * 6) : by rw [side_length_eq, rectangle_length_ratio_eq]
    ... = 108 : by norm_num

end rectangle_area_l576_576158


namespace one_half_percent_as_decimal_l576_576750

def percent_to_decimal (x : ℚ) := x / 100

theorem one_half_percent_as_decimal : percent_to_decimal (1 / 2) = 0.005 := 
by
  sorry

end one_half_percent_as_decimal_l576_576750


namespace min_troublemakers_l576_576510

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l576_576510


namespace sum_of_thousand_numbers_is_one_l576_576951

def sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, n > 0 → (n + 1) < 1000 → a n = (a (n - 1)) + (a (n + 1))

def first_two_numbers_are_one (a : ℕ → ℤ) : Prop :=
a 0 = 1 ∧ a 1 = 1

theorem sum_of_thousand_numbers_is_one (a : ℕ → ℤ) (hseq : sequence a) (hstart : first_two_numbers_are_one a) :
  (Finset.range 1000).sum a = 1 :=
sorry 

end sum_of_thousand_numbers_is_one_l576_576951


namespace ratio_of_ages_l576_576098

variables (X Y : ℕ)

theorem ratio_of_ages (h1 : X - 6 = 24) (h2 : X + Y = 36) : X / Y = 2 :=
by 
  have h3 : X = 30 - 6 := by sorry
  have h4 : X = 24 := by sorry
  have h5 : X + Y = 36 := by sorry
  have h6 : Y = 12 := by sorry
  have h7 : X / Y = 2 := by sorry
  exact h7

end ratio_of_ages_l576_576098


namespace ratio_of_speeds_is_6_over_7_l576_576569

-- Definitions of speeds and initial conditions
variables (v_A v_B : ℝ)
variables (A B O : Point) -- Assuming Points are 2D points in ℝ²

def initial_conditions :=
  (A = (0, 0)) ∧ (B = (0, -700))

def position_after_three_minutes :=
  A = (3 * v_A, 0) ∧ B = (0, -700 + 3 * v_B)

def equidistant_after_three_minutes :=
  ∃ v_A v_B : ℝ, abs (3 * v_A) = abs (3 * v_B - 700)

def position_after_twelve_minutes :=
  A = (12 * v_A, 0) ∧ B = (0, -700 + 12 * v_B)

def equidistant_after_twelve_minutes :=
  abs (12 * v_A) = abs (12 * v_B - 700)

-- The proof goal
theorem ratio_of_speeds_is_6_over_7 (A B O : Point) :
  initial_conditions A B O →
  position_after_three_minutes A B v_A v_B →
  equidistant_after_three_minutes v_A v_B →
  position_after_twelve_minutes A B v_A v_B →
  equidistant_after_twelve_minutes v_A v_B →
  (v_A / v_B = 6 / 7) :=
begin
  sorry -- Proof would go here
end

end ratio_of_speeds_is_6_over_7_l576_576569


namespace minutes_spent_calling_clients_l576_576379

theorem minutes_spent_calling_clients
    (C : ℕ)
    (H1 : 7 * C + C = 560) :
    C = 70 :=
sorry

end minutes_spent_calling_clients_l576_576379


namespace minutes_spent_calling_clients_l576_576380

theorem minutes_spent_calling_clients
    (C : ℕ)
    (H1 : 7 * C + C = 560) :
    C = 70 :=
sorry

end minutes_spent_calling_clients_l576_576380


namespace part1_part2_l576_576355

-- Definitions for set A and set B
def A : Set ℝ := { x | (x - 4) * (x + 1) < 0 }
def B (a : ℝ) : Set ℝ := { x | x^2 - 2 * a * x + a^2 - 1 < 0 }

-- Part 1: Proving A ∪ B when a = 4
theorem part1 : A ∪ B 4 = { x | -1 < x ∧ x < 5 } := 
by sorry

-- Part 2: Proving the range of a such that x ∈ Aᶜ ↔ x ∈ B(a)ᶜ is a ∈ [0, 3]
theorem part2 : { a | ∀ x, x ∈ (Aᶜ : Set ℝ) ↔ x ∈ (B a)ᶜ } = Icc (0 : ℝ) 3 := 
by sorry

end part1_part2_l576_576355


namespace minimum_number_of_troublemakers_l576_576484

theorem minimum_number_of_troublemakers (n m : ℕ) (students : Fin n → Prop) 
  (truthtellers : Fin n → Prop) (liars : Fin n → Prop) (round_table : List (Fin n))
  (h_all_students : n = 29)
  (h_truth_or_liar : ∀ s, students s → (truthtellers s ∨ liars s))
  (h_truth_tellers_conditions : ∀ s, truthtellers s → 
    (count_lies_next_to s = 1 ∨ count_lies_next_to s = 2))
  (h_liars_conditions : ∀ s, liars s → 
    (count_lies_next_to s ≠ 1 ∧ count_lies_next_to s ≠ 2))
  (h_round_table_conditions : round_table.length = n 
    ∧ (∀ i, i < n → round_table.nth i ≠ none)):
  m = 10 :=
by
  sorry

end minimum_number_of_troublemakers_l576_576484


namespace ellipse_hyperbola_foci_l576_576430

theorem ellipse_hyperbola_foci (a b : ℝ) 
    (h1 : b^2 - a^2 = 25) 
    (h2 : a^2 + b^2 = 49) : 
    |a * b| = 2 * Real.sqrt 111 := 
by 
  -- proof omitted 
  sorry

end ellipse_hyperbola_foci_l576_576430


namespace range_of_positive_integers_in_set_j_l576_576997

def consecutive_evens (start : Int) (n : Nat) : List Int :=
  List.map (λ i => start + 2 * i) (List.range n)

def positive_integers (l : List Int) : List Int :=
  l.filter (λ x => x > 0)

def range (l : List Int) : Int :=
  match l with
  | [] => 0
  | _ => l.foldl Int.min l.head! - l.foldl Int.max l.head!

theorem range_of_positive_integers_in_set_j : range (positive_integers (consecutive_evens (-12) 18)) = 20 :=
by
  -- proof to be filled
  sorry

end range_of_positive_integers_in_set_j_l576_576997


namespace simplify_expression_l576_576039

theorem simplify_expression (x y : ℝ) :
  (3 * x^2 * y)^3 + (4 * x * y) * y^4 = 27 * x^6 * y^3 + 4 * x * y^5 :=
by 
  sorry

end simplify_expression_l576_576039


namespace least_number_to_add_l576_576977

theorem least_number_to_add (a b n : ℕ) (h₁ : a = 1056) (h₂ : b = 29) (h₃ : (a + n) % b = 0) : n = 17 :=
sorry

end least_number_to_add_l576_576977


namespace num_ways_to_place_balls_l576_576072

theorem num_ways_to_place_balls : 
  let balls := {1, 2, 3, 4, 5}
  let boxes := {1, 2, 3, 4, 5}
  ∃ n : ℕ, (n = 1200) ∧
    (let ways_to_select_two_balls := Nat.choose 5 2 in
    let ways_to_permute_five_objects := Nat.factorial 5 in
    n = ways_to_select_two_balls * ways_to_permute_five_objects) := sorry

end num_ways_to_place_balls_l576_576072


namespace property_of_isosceles_not_right_triangle_l576_576618

-- Definitions of basic properties of triangles
structure Triangle :=
(A B C : Point)
(angle : Angle)
(sides : ℝ)

-- Definitions of isosceles and right triangles
def IsoscelesTriangle (T : Triangle) : Prop :=
  ∃ A B C, T.A = A ∧ T.B = B ∧ T.C = C ∧ (T.sides A B = T.sides A C ∨ T.sides B C = T.sides B A ∨ T.sides C A = T.sides C B)

def RightTriangle (T : Triangle) : Prop :=
  ∃ A B C, T.A = A ∧ T.B = B ∧ T.C = C ∧ T.angle = 90°

-- Property of interest
def AngleBisectorPerpendicular (T : Triangle) : Prop :=
  ∃ A B C, T.A = A ∧ T.B = B ∧ T.C = C ∧ (∃ M : Point, T.Bangle = 90° ∧ T.sides A M = T.sides C M)

theorem property_of_isosceles_not_right_triangle :
  ∀ (T : Triangle), IsoscelesTriangle T → ¬ RightTriangle T → AngleBisectorPerpendicular T :=
by
  sorry

end property_of_isosceles_not_right_triangle_l576_576618


namespace george_coin_distribution_l576_576231

theorem george_coin_distribution (a b c : ℕ) (h₁ : a = 1050) (h₂ : b = 1260) (h₃ : c = 210) :
  Nat.gcd (Nat.gcd a b) c = 210 :=
by
  sorry

end george_coin_distribution_l576_576231


namespace sequence_finite_values_l576_576363

def P (n : ℕ) : ℕ := (n.digits 10).prod

def sequence (n : ℕ) : ℕ → ℕ
| 0     := n
| (k+1) := let nk := sequence k in nk + P nk

theorem sequence_finite_values (n : ℕ) : ∃ N : ℕ, ∀ k ≥ N, sequence n k = sequence n N := 
sorry

end sequence_finite_values_l576_576363


namespace minimum_trees_l576_576793

variable (Trees : Type) [Fintype Trees] [DecidableEq Trees]

def trees_in_grove : Nat := 100

def tree_type := {birches, spruces, pines, aspens} : Set Trees

def condition (s : Finset Trees) : Prop := 
  s.card > 85 → tree_type ⊆ s

theorem minimum_trees (s : Finset Trees) (H : condition s) : 
  ∃ (n : Nat), n ≤ trees_in_grove ∧ n ≥ 69 → 
  ∃ t ⊆ s, t.card = n ∧ (|t ∩ tree_type| >= 3) :=
sorry

end minimum_trees_l576_576793


namespace find_h_l576_576356

-- Define the polynomial f(x)
def f (x : ℤ) := x^4 - 2 * x^3 + x - 1

-- Define the condition that f(x) + h(x) = 3x^2 + 5x - 4
def condition (f h : ℤ → ℤ) := ∀ x, f x + h x = 3 * x^2 + 5 * x - 4

-- Define the solution for h(x) to be proved
def h_solution (x : ℤ) := -x^4 + 2 * x^3 + 3 * x^2 + 4 * x - 3

-- State the theorem to be proved
theorem find_h (h : ℤ → ℤ) (H : condition f h) : h = h_solution :=
by
  sorry

end find_h_l576_576356


namespace find_b_minus_2a_l576_576056

-- Define the function f, with parameters a and b.
def f (a b : ℝ) (x : ℝ) : ℝ := (a * x - b) / (4 - x^2)

-- Define the condition that f is an odd function.
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f(x)

-- Define the given problem conditions.
theorem find_b_minus_2a :
  ∃ a b : ℝ, is_odd_function (f a b)
  ∧ f a b 1 = 1 / 3
  ∧ b - 2 * a = -2 := sorry

end find_b_minus_2a_l576_576056


namespace line_equation_l576_576705

theorem line_equation (P : ℝ × ℝ) (hP : P = (1, 5)) (h1 : ∃ a, a ≠ 0 ∧ (P.1 + P.2 = a)) (h2 : x_intercept = y_intercept) : 
  (∃ a, a ≠ 0 ∧ P = (a, 0) ∧ P = (0, a) → x + y - 6 = 0) ∨ (5*P.1 - P.2 = 0) :=
by
  sorry

end line_equation_l576_576705


namespace root_in_interval_implies_a_in_range_l576_576766

theorem root_in_interval_implies_a_in_range {a : ℝ} (h : ∃ x : ℝ, x ≤ 1 ∧ 2^x - a^2 - a = 0) : 0 < a ∧ a ≤ 1 := sorry

end root_in_interval_implies_a_in_range_l576_576766


namespace min_value_of_expression_l576_576710

noncomputable def a (n : ℕ) : ℕ := 3 * n + 2

noncomputable def S (n : ℕ) : ℕ := (3 * n^2 + 7 * n) / 2

theorem min_value_of_expression :
  (∀ (d > 0) (a_1 = 5) (S_n = (3 * n^2 + 7 * n) / 2) 
    (a_n = 3 * n + 2), 
    ∃ (m : ℚ), m = (3 * n^2 + 8 * n + 32) / (3 * n + 3) ∧ m = 20 / 3) :=
sorry

end min_value_of_expression_l576_576710


namespace shortest_distance_circle_to_line_l576_576280

theorem shortest_distance_circle_to_line :
  let C1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 7}
  let C2 := {p : ℝ × ℝ | p.1 + p.2 = 4}
  let dist := (x y : ℝ) -> abs (x + y - 4) / sqrt (1 + 1)
  let shortest_dist := dist 0 0 - sqrt 7 
  shortest_dist = 2 * sqrt 2 - sqrt 7 := by
    sorry

end shortest_distance_circle_to_line_l576_576280


namespace area_of_rectangle_is_108_l576_576144

theorem area_of_rectangle_is_108 (s w l : ℕ) (h₁ : s * s = 36) (h₂ : w = s) (h₃ : l = 3 * w) : w * l = 108 :=
by
  -- This is a placeholder for a detailed proof.
  sorry

end area_of_rectangle_is_108_l576_576144


namespace period_of_tan2x_plus_cot2x_l576_576657

noncomputable def function := λ x : ℝ, Real.tan (2 * x) + Real.cot (2 * x)

theorem period_of_tan2x_plus_cot2x : function(x) = function(x + (π / 2)) :=
by sorry

end period_of_tan2x_plus_cot2x_l576_576657


namespace maximal_length_word_number_of_words_max_length_l576_576709

-- Given n letters, the maximal possible length of a word is 3n
theorem maximal_length_word (n : ℕ) : ∃ L, L = 3 * n := by
  use 3 * n
  trivial

-- Given n letters, the number of words of maximal length is n! * 2^(n-1)
theorem number_of_words_max_length (n : ℕ) : ∃ N, N = nat.factorial n * 2^(n-1) := by
  use (nat.factorial n * 2^(n-1))
  trivial

end maximal_length_word_number_of_words_max_length_l576_576709


namespace DVDs_per_season_l576_576385

theorem DVDs_per_season (total_DVDs : ℕ) (seasons : ℕ) (h1 : total_DVDs = 40) (h2 : seasons = 5) : total_DVDs / seasons = 8 :=
by
  sorry

end DVDs_per_season_l576_576385


namespace y_value_when_x_is_3_l576_576949

theorem y_value_when_x_is_3 :
  (x + y = 30) → (x - y = 12) → (x * y = 189) → (x = 3) → y = 63 :=
by 
  intros h1 h2 h3 h4
  sorry

end y_value_when_x_is_3_l576_576949


namespace fractions_product_l576_576178

theorem fractions_product :
  (8 / 4) * (10 / 25) * (20 / 10) * (15 / 45) * (40 / 20) * (24 / 8) * (30 / 15) * (35 / 7) = 64 := by
  sorry

end fractions_product_l576_576178


namespace star_value_l576_576190

-- Defining the star operation
def star (A B : ℝ) : ℝ := (A + B) / 2

-- Theorem to be proved
theorem star_value : star (star 3 15) 6 = 7.5 :=
by
  sorry

end star_value_l576_576190


namespace area_of_rectangle_is_108_l576_576145

theorem area_of_rectangle_is_108 (s w l : ℕ) (h₁ : s * s = 36) (h₂ : w = s) (h₃ : l = 3 * w) : w * l = 108 :=
by
  -- This is a placeholder for a detailed proof.
  sorry

end area_of_rectangle_is_108_l576_576145


namespace students_in_cars_l576_576885

theorem students_in_cars (total_students : ℕ := 396) (buses : ℕ := 7) (students_per_bus : ℕ := 56) :
  total_students - (buses * students_per_bus) = 4 := by
  sorry

end students_in_cars_l576_576885


namespace ratio_of_fifth_to_second_l576_576076

-- Definitions based on the conditions
def first_stack := 7
def second_stack := first_stack + 3
def third_stack := second_stack - 6
def fourth_stack := third_stack + 10

def total_blocks := 55

-- The number of blocks in the fifth stack
def fifth_stack := total_blocks - (first_stack + second_stack + third_stack + fourth_stack)

-- The ratio of the fifth stack to the second stack
def ratio := fifth_stack / second_stack

-- The theorem we want to prove
theorem ratio_of_fifth_to_second: ratio = 2 := by
  sorry

end ratio_of_fifth_to_second_l576_576076


namespace min_trees_for_three_types_l576_576799

-- Define types and trees
inductive TreeType
| birch | spruce | pine | aspen
deriving Inhabited, DecidableEq

-- A grove with 100 trees of any of the four types.
structure Tree :=
(type : TreeType)

constant grove : List Tree
axiom grove_size : grove.length = 100

-- Condition: Among any 85 trees, there are trees of all four types.
axiom all_types_in_any_85 : ∀ (s : Finset Tree), s.card = 85 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, ∃ t4 ∈ s, 
      t1.type = TreeType.birch ∧
      t2.type = TreeType.spruce ∧
      t3.type = TreeType.pine ∧
      t4.type = TreeType.aspen

-- We need to show that at least 69 trees are needed to ensure at least 3 types.
theorem min_trees_for_three_types : 
  ∀ (s : Finset Tree), s.card = 69 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, 
      t1.type ≠ t2.type ∧ t2.type ≠ t3.type ∧ t1.type ≠ t3.type := 
sorry

end min_trees_for_three_types_l576_576799


namespace length_segment_D_D_l576_576964

def point := {x : ℝ, y : ℝ}

def reflect_y (p : point) : point :=
  {x := -p.x, y := p.y}

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def D : point := {x := 2, y := -4}

def D' : point := reflect_y D

theorem length_segment_D_D' : dist D D' = 4 :=
  by sorry

end length_segment_D_D_l576_576964


namespace trig_simplification_l576_576407

noncomputable def simplify_trig_expression (x : ℝ) : Prop :=
  (sin x + (3 * sin x - 4 * (sin x) ^ 3)) / 
  (1 + cos x + (4 * (cos x) ^ 3 - 3 * cos x)) = 
  4 * (sin x - (sin x) ^ 3) / (1 - 2 * cos x + 4 * (cos x) ^ 3)

theorem trig_simplification (x : ℝ) : simplify_trig_expression x :=
by
  sorry

end trig_simplification_l576_576407


namespace card_sum_cases_l576_576983

theorem card_sum_cases : 
  let cards := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∃ (pairs : set (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ pairs → a ∈ cards ∧ b ∈ cards ∧ a ≠ b ∧ (a + b = 3 ∨ a + b = 14)) ∧ 
    pairs.card = 3 :=
by
  let cards := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let pairs := { (1, 2), (5, 9), (6, 8) }
  have : Π p ∈ pairs, p = (1, 2) ∨ p = (5, 9) ∨ p = (6, 8) := sorry
  have : ∀ (a b : ℕ), (a, b) ∈ pairs → a ∈ cards ∧ b ∈ cards ∧ a ≠ b ∧ (a + b = 3 ∨ a + b = 14) := sorry
  have : pairs.card = 3 := sorry
  exact ⟨pairs, this, sorry⟩

end card_sum_cases_l576_576983


namespace doughnut_problem_l576_576018

theorem doughnut_problem :
  ∀ (total_doughnuts first_two_box_doughnuts boxes : ℕ),
  total_doughnuts = 72 →
  first_two_box_doughnuts = 12 →
  boxes = 4 →
  (total_doughnuts - 2 * first_two_box_doughnuts) / boxes = 12 :=
by
  intros total_doughnuts first_two_box_doughnuts boxes ht12 hb12 b4
  sorry

end doughnut_problem_l576_576018


namespace sequence_real_roots_l576_576202

def n : ℕ := sorry -- Define n ∈ ℕ
def a : Fin n.succ → ℝ := sorry -- Define the sequence a_i of length n+1
def x : Fin n.succ → ℝ := sorry -- Define the sequence x_i of length n+1

theorem sequence_real_roots 
  (hn : a (Fin.last n) ≠ 0)
  (hsorted : ∀ i j : Fin n.succ, i < j → x i < x j)
  (hzeros : ∀ i : Fin n.succ, f (x i) = 0)
  (f : ℝ → ℝ) 
  (hf_differentiable : ∀ k ≤ n, Differentiable ℝ (f^[k])) :
  (∃ h ∈ Set.Ioo (x Fin.val 0) (x Fin.val n), 
    (Finset.range n.succ).sum (λ i, a ⟨i, sorry⟩ * deriv^[i] f h) = 0) 
  ↔ 
  (∀ (P : ℝ → ℝ) (hP : P = ∑ i in Finset.range n.succ, a ⟨i, sorry⟩ * x^i), 
    ∀ c ∈ P.roots, ∃ hc : ℝ, c = hc) := 
sorry

end sequence_real_roots_l576_576202


namespace ratio_of_kids_l576_576320

theorem ratio_of_kids (k2004 k2005 k2006 : ℕ) 
  (h2004: k2004 = 60) 
  (h2005: k2005 = k2004 / 2)
  (h2006: k2006 = 20) :
  (k2006 : ℚ) / k2005 = 2 / 3 :=
by
  sorry

end ratio_of_kids_l576_576320


namespace price_of_each_hot_dog_l576_576966

variable (x : ℝ) -- price of each hot dog

-- Definition of the conditions
def five_hot_dogs_cost := 5 * x
def three_salads_cost := 3 * 2.50
def total_paid := 2 * 10
def change_received := 5
def actual_amount_spent := total_paid - change_received

-- The proof statement
theorem price_of_each_hot_dog : 
  five_hot_dogs_cost + three_salads_cost = actual_amount_spent → x = 1.50 := by
  sorry

end price_of_each_hot_dog_l576_576966


namespace probability_of_odd_product_lt_25_l576_576900

def ball_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7]

noncomputable def choices : ℕ × ℕ → ℕ :=
λ (x : ℕ × ℕ), (x.1 * x.2)

noncomputable def is_odd : ℕ → Bool := λ n, (n % 2 = 1)

noncomputable def valid_pair (x : ℕ × ℕ) : Prop :=
is_odd x.1 ∧ is_odd x.2 ∧ x.1 * x.2 < 25

noncomputable def probability_valid_product_lt_25 : ℚ :=
12 / 49

theorem probability_of_odd_product_lt_25 :
  (∑ i in ball_numbers.product ball_numbers, if valid_pair i then 1 else 0 : ℚ) / (∑ i in ball_numbers.product ball_numbers, 1 : ℚ) = probability_valid_product_lt_25 :=
by {
  sorry
}

end probability_of_odd_product_lt_25_l576_576900


namespace find_modulus_of_Z_l576_576740

def determinant (a b c d : ℂ) : ℂ := a * d - b * c

def modulus (z : ℂ) : ℝ := Complex.abs z

theorem find_modulus_of_Z (z : ℂ) : 
  determinant z Complex.i 1 Complex.i = (1 : ℂ) + Complex.i → 
  modulus z = Real.sqrt 5 := 
by
  intro h
  have hz : z = (2 : ℂ) - Complex.i := 
    by
      sorry
  rw hz
  simp [modulus]
  -- Final result to be shown
  sorry

end find_modulus_of_Z_l576_576740


namespace max_points_in_plane_max_points_in_space_l576_576092

noncomputable theory

def max_points_plane_no_obtuse : ℕ := 4

def max_points_space_no_obtuse : ℕ := 8

theorem max_points_in_plane (P : set (ℝ × ℝ)) (h_no_collinear : ∀ A B C : P, ¬collinear P A B C) (h_no_obtuse : ∀ A B C : P, ¬obtuse_triangle A B C) :
  ∃ (n : ℕ), n ≤ max_points_plane_no_obtuse :=
begin
  use 4,
  sorry
end

theorem max_points_in_space (P : set (ℝ × ℝ × ℝ)) (h_convex_hull : ∀ x : ℝ × ℝ × ℝ, x ∈ P → ∃ v w, is_convex_hull P v w x) (h_no_obtuse : ∀ A B C : P, ¬obtuse_triangle A B C) :
  ∃ (n : ℕ), n ≤ max_points_space_no_obtuse :=
begin
  use 8,
  sorry
end

end max_points_in_plane_max_points_in_space_l576_576092


namespace min_troublemakers_in_class_l576_576462

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l576_576462


namespace max_basketballs_l576_576962

theorem max_basketballs (total_balls : ℕ) (budget : ℕ) (cost_basketball : ℕ) (cost_soccerball : ℕ)
  (h_total : total_balls = 50) (h_budget : budget = 3000) (h_cost_basketball : cost_basketball = 80)
  (h_cost_soccerball : cost_soccerball = 50) :
  ∃ (max_m : ℕ), max_m = 16 ∧ (max_m ≤ 50) ∧ (cost_basketball * max_m + cost_soccerball * (50 - max_m) ≤ budget) :=
by {
  use 16,
  split,
  refl,
  split,
  norm_num,
  rw [h_budget, h_cost_basketball, h_cost_soccerball],
  norm_num,
  sorry
}

end max_basketballs_l576_576962


namespace sum_of_x_for_ggg_eq_neg2_l576_576132

noncomputable def g (x : ℝ) := (x^2) / 3 + x - 2

theorem sum_of_x_for_ggg_eq_neg2 : (∃ x1 x2 : ℝ, (g (g (g x1)) = -2 ∧ g (g (g x2)) = -2 ∧ x1 ≠ x2)) ∧ (x1 + x2 = 0) :=
by
  sorry

end sum_of_x_for_ggg_eq_neg2_l576_576132


namespace concyclic_points_l576_576587

open EuclideanGeometry

variables {A B C K L P Q S T : Point} {l : Line} {ω : Circle}

-- Conditions
axiom tangent_to_angle (h : Tangent ω (Line.mk A B) ∧ Tangent ω (Line.mk A C) → TangentPoints ω = (B, C))
axiom intersects_segments (h : Intersects l (Segment.mk A B) K ∧ Intersects l (Segment.mk A C) L)
axiom circle_intersects_line (h : Intersects ω l P ∧ Intersects ω l Q)
axiom ks_parallel_ac (h : Parallel (Line.mk K S) (Line.mk A C))
axiom lt_parallel_ab (h : Parallel (Line.mk L T) (Line.mk A B))
axiom points_on_bc (hS : On S (Segment.mk B C) ∧ hT : On T (Segment.mk B C))

-- To Prove
theorem concyclic_points : Concyclic P Q S T :=
sorry

end concyclic_points_l576_576587


namespace fraction_sum_l576_576515

theorem fraction_sum (n : ℕ) (a : ℚ) (sum_fraction : a = 1/12) (number_of_fractions : n = 450) : 
  ∀ (f : ℚ), (n * f = a) → (f = 1/5400) :=
by
  intros f H
  sorry

end fraction_sum_l576_576515


namespace frog_climbing_time_is_correct_l576_576126

noncomputable def frog_climb_out_time : Nat :=
  let well_depth := 12
  let climb_up := 3
  let slip_down := 1
  let net_gain := climb_up - slip_down
  let total_cycles := (well_depth - 3) / net_gain + 1
  let total_time := total_cycles * 3
  let extra_time := 6
  total_time + extra_time

theorem frog_climbing_time_is_correct :
  frog_climb_out_time = 22 := by
  sorry

end frog_climbing_time_is_correct_l576_576126


namespace circles_intersect_l576_576935

structure Circle :=
  center : (ℝ × ℝ)
  radius : ℝ

noncomputable def circle1 : Circle := 
  { center := (-1, -4),
    radius := 5 }

noncomputable def circle2 : Circle :=
  { center := (2, 2),
    radius := 3 }

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def intersecting (C1 C2 : Circle) : Prop :=
  let d := distance C1.center C2.center in
  abs (C1.radius - C2.radius) < d ∧ d < C1.radius + C2.radius

theorem circles_intersect : intersecting circle1 circle2 :=
  by sorry

end circles_intersect_l576_576935


namespace quadratic_trinomial_no_real_roots_l576_576427

variable (a b d : ℝ) (h : a ≠ 0) (h_bd : b ≠ d)

def f (x : ℝ) := a * x + b
def g (x : ℝ) := a * x + d

theorem quadratic_trinomial_no_real_roots :
  ∀ x : ℝ, (b - d) * ((f a b d x)^2 + (f a b d x) * (g a b d x) + (g a b d x)^2) ≠ 0 :=
  by
  intros
  sorry

end quadratic_trinomial_no_real_roots_l576_576427


namespace find_line_and_intersection_l576_576332

def direct_proportion_function (k : ℝ) (x : ℝ) : ℝ :=
  k * x

def shifted_function (k : ℝ) (x b : ℝ) : ℝ :=
  k * x + b

theorem find_line_and_intersection
  (k : ℝ) (b : ℝ) (h₀ : direct_proportion_function k 1 = 2) (h₁ : b = 5) :
  (shifted_function k 1 b = 7) ∧ (shifted_function k (-5/2) b = 0) :=
by
  -- This is just a placeholder to indicate where the proof would go
  sorry

end find_line_and_intersection_l576_576332


namespace count_true_propositions_l576_576442

-- Definitions based on provided conditions
def proposition_original (A B C : Point) : Prop := (A.distance B = A.distance C) → is_isosceles_triangle A B C
def proposition_converse (A B C : Point) : Prop := is_isosceles_triangle A B C → (A.distance B = A.distance C)
def proposition_inverse (A B C : Point) : Prop := (A.distance B ≠ A.distance C) → ¬ is_isosceles_triangle A B C
def proposition_contrapositive (A B C : Point) : Prop := ¬ is_isosceles_triangle A B C → (A.distance B ≠ A.distance C)

-- Statement to prove the count of true propositions
theorem count_true_propositions (A B C : Point) :
  (nat_val (bounded_array.mk [proposition_converse A B C, proposition_inverse A B C, proposition_contrapositive A B C] 3 (by simp [proposition_converse, proposition_inverse, proposition_contrapositive]))) = 1 :=
by sorry

end count_true_propositions_l576_576442


namespace greatest_five_digit_multiple_of_30_l576_576967

-- The digits to be used
def digits : List ℕ := [0, 3, 5, 7, 9]

-- Define a function that checks if a number is a multiple of 30
def isMultipleOf30 (n : ℕ) : Prop :=
  n % 30 = 0

-- Define a function that checks if a number can be formed using each digit exactly once
def isValidNumber (n : ℕ) : Prop :=
  let ds := Int.toDigits 10 n
  (ds.length = 5) ∧ (List.PerM digits ds)

-- The statement we need to prove:
theorem greatest_five_digit_multiple_of_30 : ∃ n : ℕ, isValidNumber n ∧ isMultipleOf30 n ∧ n = 97530 := by
  sorry

end greatest_five_digit_multiple_of_30_l576_576967


namespace find_rate_l576_576175

def principal : ℝ := 750
def amount : ℝ := 900
def time : ℝ := 16

def simple_interest_formula (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem find_rate : ∃ R : ℝ, amount = principal + simple_interest_formula principal R time ∧ R = 1.25 :=
by
  sorry

end find_rate_l576_576175


namespace wheel_radius_increase_l576_576663

theorem wheel_radius_increase :
  let r := 18
  let distance_AB := 600   -- distance from A to B in miles
  let distance_BA := 582   -- distance from B to A in miles
  let circumference_orig := 2 * Real.pi * r
  let dist_per_rotation_orig := circumference_orig / 63360
  let rotations_orig := distance_AB / dist_per_rotation_orig
  let r' := ((distance_BA * dist_per_rotation_orig * 63360) / (2 * Real.pi * rotations_orig))
  ((r' - r) : ℝ) = 0.34 := by
  sorry

end wheel_radius_increase_l576_576663


namespace problem1_problem2_problem3_problem4_l576_576616

def GanToShar (g: ℚ) : ℚ := g * 1000

theorem problem1 : GanToShar ((1/3 : ℚ) + (2/18 : ℚ)) = 444.44 := 
by sorry

theorem problem2 : GanToShar ((1/3 : ℚ) + (3/18 : ℚ)) = 500 := 
by sorry

theorem problem3 : GanToShar (1 - (1/36 : ℚ)) = 972.22 := 
by sorry

theorem problem4 : 
  let diff := 20 - (1 + (1/3 : ℚ)) in
  let whole_part := (diff : ℚ).floor in
  let frac_part := GanToShar ((diff - whole_part) : ℚ) in
  whole_part = 18 ∧ frac_part = 666.67 := 
by sorry

end problem1_problem2_problem3_problem4_l576_576616


namespace maryann_time_spent_calling_clients_l576_576377

theorem maryann_time_spent_calling_clients (a c : ℕ) 
  (h1 : a + c = 560) 
  (h2 : a = 7 * c) : c = 70 := 
by 
  sorry

end maryann_time_spent_calling_clients_l576_576377


namespace deepak_age_l576_576106

-- Defining the problem with the given conditions in Lean:
theorem deepak_age (x : ℕ) (rahul_current : ℕ := 4 * x) (deepak_current : ℕ := 3 * x) :
  (rahul_current + 6 = 38) → (deepak_current = 24) :=
by
  sorry

end deepak_age_l576_576106


namespace sum_of_integer_solutions_l576_576903

noncomputable def absolute_value_inequality (x : ℝ) : Prop :=
  12 * (|x + 10| - |x - 20|) / (|4 * x - 25| - |4 * x - 15|) -
  (|x + 10| + |x - 20|) / (|4 * x - 25| + |4 * x - 15|) ≥ -6

theorem sum_of_integer_solutions :
  (∑ x in (Finset.filter (λ x, |x| < 100) (Finset.range 201)).filter (λ x, absolute_value_inequality x), x) = 4 :=
by
  sorry

end sum_of_integer_solutions_l576_576903


namespace unique_plants_count_l576_576078

open Finset

variable (A B C : Finset ℕ)

def card_A : ℕ := 600
def card_B : ℕ := 550
def card_C : ℕ := 400
def card_AB : ℕ := 60
def card_AC : ℕ := 110
def card_BC : ℕ := 90
def card_ABC : ℕ := 30

theorem unique_plants_count :  
  ∀ A B C : Finset ℕ,  
  A.card = card_A ∧ 
  B.card = card_B ∧ 
  C.card = card_C ∧ 
  (A ∩ B).card = card_AB ∧ 
  (A ∩ C).card = card_AC ∧ 
  (B ∩ C).card = card_BC ∧ 
  (A ∩ B ∩ C).card = card_ABC → 
  (A ∪ B ∪ C).card = 1320 := 
by sorry

end unique_plants_count_l576_576078


namespace simplify_expression_l576_576902

theorem simplify_expression (a : ℝ) (h : a ≠ 0) : (a^2 / (sqrt(a) * (3 * a^2))) = (1 / sqrt(a)) :=
by sorry

end simplify_expression_l576_576902


namespace lollipop_distribution_l576_576773

theorem lollipop_distribution :
  let n := 42
  let initial_lollipops := 650
  let required_lollipops := n * (n + 1) / 2
  (required_lollipops - initial_lollipops) = 253 :=
by
  let n := 42
  let initial_lollipops := 650
  let required_lollipops := n * (n + 1) / 2
  have h : required_lollipops = 903 := by norm_num
  have h2 : (required_lollipops - initial_lollipops) = 253 := by norm_num
  exact h2

end lollipop_distribution_l576_576773


namespace length_of_CD_l576_576062

theorem length_of_CD (L : ℝ) (r : ℝ) (V_total : ℝ) (cylinder_vol : ℝ) (hemisphere_vol : ℝ) : 
  r = 5 ∧ V_total = 900 * Real.pi ∧ cylinder_vol = Real.pi * r^2 * L ∧ hemisphere_vol = (2/3) *Real.pi * r^3 → 
  V_total = cylinder_vol + 2 * hemisphere_vol → 
  L = 88 / 3 := 
by
  sorry

end length_of_CD_l576_576062


namespace boys_not_in_clubs_l576_576890

-- Definitions based on conditions
def total_students : ℕ := 150
def percentage_girls : ℝ := 0.6
def percentage_boys : ℝ := 1 - percentage_girls
def boys_in_total : ℕ := (percentage_boys * total_students).toNat
def fraction_boys_in_clubs : ℝ := 1 / 3
def boys_in_clubs : ℕ := (fraction_boys_in_clubs * boys_in_total).toNat

-- Statement to prove
theorem boys_not_in_clubs : boys_in_total - boys_in_clubs = 40 :=
by
  sorry

end boys_not_in_clubs_l576_576890


namespace a_and_c_can_complete_in_20_days_l576_576559

-- Define the work rates for the pairs given in the conditions.
variables {A B C : ℚ}

-- a and b together can complete the work in 12 days
axiom H1 : A + B = 1 / 12

-- b and c together can complete the work in 15 days
axiom H2 : B + C = 1 / 15

-- a, b, and c together can complete the work in 10 days
axiom H3 : A + B + C = 1 / 10

-- We aim to prove that a and c together can complete the work in 20 days,
-- hence their combined work rate should be 1 / 20.
theorem a_and_c_can_complete_in_20_days : A + C = 1 / 20 :=
by
  -- sorry will be used to skip the proof
  sorry

end a_and_c_can_complete_in_20_days_l576_576559


namespace sqrt_20n_integer_exists_l576_576759

theorem sqrt_20n_integer_exists : 
  ∃ n : ℤ, 0 ≤ n ∧ ∃ k : ℤ, k * k = 20 * n :=
sorry

end sqrt_20n_integer_exists_l576_576759


namespace count_integers_with_zero_l576_576752

/-- There are 740 positive integers less than or equal to 3017 that contain the digit 0. -/
theorem count_integers_with_zero (n : ℕ) (h : n ≤ 3017) : 
  (∃ k : ℕ, k ≤ 3017 ∧ ∃ d : ℕ, d < 10 ∧ d ≠ 0 ∧ k / 10 ^ d % 10 = 0) ↔ n = 740 :=
by sorry

end count_integers_with_zero_l576_576752


namespace second_group_men_count_l576_576301

-- Define the conditions given in the problem
def men1 := 8
def days1 := 80
def days2 := 32

-- The question we need to answer
theorem second_group_men_count : 
  ∃ (men2 : ℕ), men1 * days1 = men2 * days2 ∧ men2 = 20 :=
by
  sorry

end second_group_men_count_l576_576301


namespace math_problem_l576_576631

def pow (a b : ℕ) : ℕ := a ^ b
def div_pow (a b c : ℕ) : ℕ := a ^ b / a ^ c
def mul (a b : ℕ) : ℕ := a * b

-- Here we're stating the theorem to prove that (8^3 / 8^2) * 3^3 = 216
theorem math_problem : mul (div_pow 8 3 2) (pow 3 3) = 216 :=
by
  sorry

end math_problem_l576_576631


namespace average_number_of_problems_per_day_l576_576415

theorem average_number_of_problems_per_day (P D : ℕ) (hP : P = 161) (hD : D = 7) : (P / D) = 23 :=
  by sorry

end average_number_of_problems_per_day_l576_576415


namespace sum_even_odd_diff_l576_576975

theorem sum_even_odd_diff (n : ℕ) (h : n = 1500) : 
  let S_odd := n / 2 * (1 + (1 + (n - 1) * 2))
  let S_even := n / 2 * (2 + (2 + (n - 1) * 2))
  (S_even - S_odd) = n :=
by
  sorry

end sum_even_odd_diff_l576_576975


namespace min_trees_include_three_types_l576_576814

noncomputable def minNumTrees (T : Type) (tree_counts : T → ℕ) :=
  ∀ (total_trees : ℕ) (at_least : ℕ → Prop),
    total_trees = 100 →
    (∀ S : Finset T, S.card = 85 → (∃ t ∈ S, at_least (tree_counts t))) →
    (at_least ThreeTypes ↔ ∃ (S : Finset T), S.card = 69)

def ThreeTypes (tree_counts : T → ℕ) := 
  ∀ (birches spruces pines aspens : ℕ),
    birches + spruces + pines + aspens = 100 →
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ pines ≥ 1) ∨ 
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ aspens ≥ 1) ∨ 
    (birches ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1) ∨ 
    (spruces ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1)

theorem min_trees_include_three_types :
  ∃ (T : Type) (tree_counts : T → ℕ), minNumTrees T tree_counts := 
sorry

end min_trees_include_three_types_l576_576814


namespace power_of_three_ends_in_0001_l576_576847

theorem power_of_three_ends_in_0001 : ∃ n : ℕ, 3^n % 10000 = 1 := 
by
  let a := 3
  let n := 10000
  let φ := (λ n, n * (1 - 1/2)) * (λ n, n * (1 - 1/5)) in
  have coprime : Nat.gcd a n = 1 := by sorry,
  have euler_theorem := Nat.pow_totient_mod n 3 (by rw coprime) φ,
  have 3 ^ (φ 10000) = 1 % 10000 := by rw euler_theorem; apply Nat.mod_eq_of_lt; sorry,
  exact ⟨4000, by rw [this, pow_mul, pow_one]⟩

end power_of_three_ends_in_0001_l576_576847


namespace num_permutations_with_P_gt_without_P_l576_576861

noncomputable def permutations_with_property_P (n : ℕ) : Finset (Finset (Fin n)) :=
{ perm | ∃ i, (1 ≤ i ∧ i < 2 * n) ∧ (|perm[i] - perm[i+1]| = n) }

noncomputable def permutations_without_property_P (n : ℕ) : Finset (Finset (Fin n)) :=
{ perm | ¬ ∃ i, (1 ≤ i ∧ i < 2 * n) ∧ (|perm[i] - perm[i+1]| = n) }

theorem num_permutations_with_P_gt_without_P (n : ℕ) :
  (permutations_with_property_P n).card > (permutations_without_property_P n).card :=
sorry

end num_permutations_with_P_gt_without_P_l576_576861


namespace curve_transformation_l576_576336

theorem curve_transformation :
  (∀ (x y : ℝ), 2 * (5 * x)^2 + 8 * (3 * y)^2 = 1) → (∀ (x y : ℝ), 50 * x^2 + 72 * y^2 = 1) :=
by
  intros h x y
  have h1 : 2 * (5 * x)^2 + 8 * (3 * y)^2 = 1 := h x y
  sorry

end curve_transformation_l576_576336


namespace min_employees_to_hire_l576_576590

-- Definitions of the given conditions
def employees_cust_service : ℕ := 95
def employees_tech_support : ℕ := 80
def employees_both : ℕ := 30

-- The theorem stating the minimum number of new employees to hire
theorem min_employees_to_hire (n : ℕ) :
  n = (employees_cust_service - employees_both) 
    + (employees_tech_support - employees_both) 
    + employees_both → 
  n = 145 := sorry

end min_employees_to_hire_l576_576590


namespace difference_between_advertised_and_actual_mileage_l576_576025

def advertised_mileage : ℕ := 35

def city_mileage_regular : ℕ := 30
def highway_mileage_premium : ℕ := 40
def traffic_mileage_diesel : ℕ := 32

def gallons_regular : ℕ := 4
def gallons_premium : ℕ := 4
def gallons_diesel : ℕ := 4

def total_miles_driven : ℕ :=
  (gallons_regular * city_mileage_regular) + 
  (gallons_premium * highway_mileage_premium) + 
  (gallons_diesel * traffic_mileage_diesel)

def total_gallons_used : ℕ :=
  gallons_regular + gallons_premium + gallons_diesel

def weighted_average_mpg : ℤ :=
  total_miles_driven / total_gallons_used

theorem difference_between_advertised_and_actual_mileage :
  advertised_mileage - weighted_average_mpg = 1 :=
by
  -- proof to be filled in
  sorry

end difference_between_advertised_and_actual_mileage_l576_576025


namespace probability_divisible_by_4_l576_576396

theorem probability_divisible_by_4 :
  let S := {1, 2, ..., 2007}
  let count_divisible_by_4 := 501
  let prob_divisible_by_4 := count_divisible_by_4 / 2007
  let remaining_prob := 1506 / 2007
  let p := 1 / 2
  let final_prob := prob_divisible_by_4 + remaining_prob * p
  final_prob = 1254 / 2007 := 
sorry

end probability_divisible_by_4_l576_576396


namespace exactly_two_segments_longer_than_one_l576_576895

noncomputable def probability_of_two_segments_longer_than_one : ℚ :=
let total_area : ℚ := 9 / 2 in
let event_area : ℚ := 3 / 2 in
event_area / total_area

theorem exactly_two_segments_longer_than_one (x y : ℚ) (h1 : 0 < x) (h2 : x < 3) (h3 : 0 < y) (h4 : y < 3) (h5 : x + y < 3) :
  probability_of_two_segments_longer_than_one = 1 / 3 :=
by
  sorry

end exactly_two_segments_longer_than_one_l576_576895


namespace range_of_b_l576_576260

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 
  if x ∈ Ioo 0 2 then log (x^2 - x + b) else 0

theorem range_of_b
  (h_odd : ∀ x : ℝ, f(-x) b = -f(x) b)
  (h_periodic : ∀ x : ℝ, f(x + 4) b = f(x) b)
  (h_zeros : ∃ a b c d e : ℝ, list.in (a :: b :: c :: d :: e :: []), 
    a ∈ Icc (-2 : ℝ) 2 ∧ b ∈ Icc (-2 : ℝ) 2 ∧ c ∈ Icc (-2 : ℝ) 2 ∧ d ∈ Icc (-2 : ℝ) 2 ∧ e ∈ Icc (-2 : ℝ) 2 ∧ 
    f (a) b = 0 ∧ f (b) b = 0 ∧ f (c) b = 0 ∧ f (d) b = 0 ∧ f (e) b = 0 )
  (h_func_cond : ∀ x, x ∈ Ioo (0 : ℝ) 2 → x^2 - x + b > 0) :
  (1 / 4 < b ∧ b ≤ 1) ∨ b = 5 / 4 :=
sorry

end range_of_b_l576_576260


namespace distance_between_foci_hyperbola_l576_576676

theorem distance_between_foci_hyperbola :
  (let a^2 := 50 in
   let b^2 := 8 in
   let c^2 := a^2 + b^2 in
   2 * Real.sqrt c^2 = 2 * Real.sqrt 58) :=
by
  sorry

end distance_between_foci_hyperbola_l576_576676


namespace base_b_digit_sum_l576_576972

theorem base_b_digit_sum :
  ∃ (b : ℕ), ((b^2 / 2 + b / 2) % b = 2) ∧ (b = 8) :=
by
  sorry

end base_b_digit_sum_l576_576972


namespace min_troublemakers_in_class_l576_576460

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l576_576460


namespace find_lowest_score_l576_576420

theorem find_lowest_score (s : Fin 15 → ℕ) 
  (mean_15 := (∑ i, s i) / 15 = 90)
  (mean_13 := (∑ i in Finset.erase (Finset.univ) (0 : Fin 15), if i = 0 then 100 else s i) / 13 = 92) 
  (max_score := s 0 = 100) :
  ∃ low_score, low_score = 54 := 
sorry

end find_lowest_score_l576_576420


namespace prime_divisors_of_expression_l576_576401

theorem prime_divisors_of_expression (a : ℤ) (h : a > 1) (p : ℤ) (hp : p.prime) (hdiv : p ∣ (5 * a ^ 4 - 5 * a ^ 2 + 1)) :
  ∃ k : ℤ, p = 20 * k + 1 ∨ p = 20 * k - 1 :=
by
  sorry

end prime_divisors_of_expression_l576_576401


namespace largest_solution_l576_576680

theorem largest_solution (h₁ : (abs (2 * sin x - 1) + abs (2 * cos (2 * x) - 1)) = 0)
  (h₂ : 0 ≤ x) (h₃ : x ≤ 10 * Real.pi) : 
  x ≈ 27.7 :=
sorry

end largest_solution_l576_576680


namespace minimum_trees_l576_576797

variable (Trees : Type) [Fintype Trees] [DecidableEq Trees]

def trees_in_grove : Nat := 100

def tree_type := {birches, spruces, pines, aspens} : Set Trees

def condition (s : Finset Trees) : Prop := 
  s.card > 85 → tree_type ⊆ s

theorem minimum_trees (s : Finset Trees) (H : condition s) : 
  ∃ (n : Nat), n ≤ trees_in_grove ∧ n ≥ 69 → 
  ∃ t ⊆ s, t.card = n ∧ (|t ∩ tree_type| >= 3) :=
sorry

end minimum_trees_l576_576797


namespace locus_of_intersection_points_l576_576110

noncomputable theory

variables {Γ1 Γ2 Γ3 : Circle} {r1 r2 r3 : ℝ}
variables (A : Point) (ℓ : Line) (ω : Circle)

-- Given Conditions
def gamma1_inscribed_in_corner : Prop := (Γ1.r = r1) ∧ (r1 < r2) ∧ (r1 < r3)
def gamma2_inscribed_in_corner : Prop := (Γ2.r = r2)
def gamma3_inscribed_in_corner: Prop := (Γ3.r = r3)
def tangent_at_A : Prop := is_tangent(ℓ, Γ1, A)
def circle_omega_tangent_to_gamma1_and_ℓ : Prop := is_tangent(ω, Γ1) ∧ is_tangent(ω, ℓ)

-- Conclusion: locus of points of intersection
def locus_of_points_of_intersection_of_internal_tangents (H : Point) : Prop := 
  ∃ ω, is_tangent(ω, Γ1) ∧ is_tangent(ω, ℓ) ∧ H ∈ (intersection_points_of_internal_tangents(ω, Γ3))

-- Theorem statement
theorem locus_of_intersection_points (H : Point) :
  gamma1_inscribed_in_corner ∧
  gamma2_inscribed_in_corner ∧
  gamma3_inscribed_in_corner ∧
  tangent_at_A ∧ 
  circle_omega_tangent_to_gamma1_and_ℓ → 
  (locus_of_points_of_intersection_of_internal_tangents H → is_point_on_interval(H, A, B)) :=
sorry


end locus_of_intersection_points_l576_576110


namespace composite_numbers_l576_576438

theorem composite_numbers (n : ℕ) (hn : n > 0) :
  (∃ p q, p > 1 ∧ q > 1 ∧ 2 * 2^(2^n) + 1 = p * q) ∧ 
  (∃ p q, p > 1 ∧ q > 1 ∧ 3 * 2^(2*n) + 1 = p * q) :=
sorry

end composite_numbers_l576_576438


namespace polynomial_simplification_l576_576555

variable (x : ℝ)

theorem polynomial_simplification :
  (3 * x^2 + 5 * x + 9) * (x + 2) - (x + 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x + 2) * (x + 4) =
  6 * x^3 - 28 * x^2 - 59 * x + 42 :=
by
  sorry

end polynomial_simplification_l576_576555


namespace unique_function_l576_576669

noncomputable def f : ℝ → ℝ := sorry

theorem unique_function 
  (h_f : ∀ x > 0, ∀ y > 0, f x * f y = 2 * f (x + y * f x)) : ∀ x > 0, f x = 2 :=
by
  sorry

end unique_function_l576_576669


namespace parabola_midpoint_locus_minimum_slope_difference_exists_l576_576711

open Real

def parabola_locus (x y : ℝ) : Prop :=
  x^2 = 4 * y

def slope_difference_condition (x1 x2 k1 k2 : ℝ) : Prop :=
  |k1 - k2| = 1

theorem parabola_midpoint_locus :
  ∀ (x y : ℝ), parabola_locus x y :=
by
  intros x y
  apply sorry

theorem minimum_slope_difference_exists :
  ∀ {x1 y1 x2 y2 k1 k2 : ℝ},
  slope_difference_condition x1 x2 k1 k2 :=
by
  intros x1 y1 x2 y2 k1 k2
  apply sorry

end parabola_midpoint_locus_minimum_slope_difference_exists_l576_576711


namespace min_troublemakers_l576_576456

theorem min_troublemakers (students : Finset ℕ) (table : List ℕ) (is_liar : ℕ → Bool) :
  students.card = 29 ∧
  (∀ s ∈ students, 
    (∃ i j : ℕ, 
      table.nth i = some s ∧ table.nth ((i + 1) % 29) = some j ∧
      (∃ k : ℕ, 
        (is_liar s = false → count (/=is_liar k liar) = 0 ∧
        is_liar s = true → count (/=is_liar k liar) = 2))) → 
  (∃! l ∈ students, count (/=is_liar l liar) = 1))) → 
  ∃ t : ℕ, t = 10 :=
sorry

end min_troublemakers_l576_576456


namespace sum_integer_solutions_l576_576905

noncomputable def isValidSolution (x : ℝ) : Prop :=
  12 * ((| x + 10 | - | x - 20 |) / (| 4 * x - 25 | - | 4 * x - 15 |)) -
  ((| x + 10 | + | x - 20 |) / (| 4 * x - 25 | + | 4 * x - 15 |)) ≥ -6

theorem sum_integer_solutions : ∑ x in finset.filter (λ i : ℤ, -100 < i ∧ i < 100 ∧ isValidSolution (i : ℝ)) (finset.range 200), x = 4 := 
by {
  sorry
}

end sum_integer_solutions_l576_576905


namespace neg_of_all_men_are_honest_l576_576927

variable {α : Type} (man honest : α → Prop)

theorem neg_of_all_men_are_honest :
  ¬ (∀ x, man x → honest x) ↔ ∃ x, man x ∧ ¬ honest x :=
by
  sorry

end neg_of_all_men_are_honest_l576_576927


namespace smallest_lambda_l576_576367

theorem smallest_lambda (n : ℕ) (h : n ≥ 2) :
  ∃ (λ : ℝ) (λ_eq : λ = n / (n + 1 : ℕ)), ∀ (x : fin n → ℝ) (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1), 
  ∃ (ε : fin n → ℕ) (hε : ∀ i, ε i = 0 ∨ ε i = 1), 
  ∀ (i j : fin n) (hij : i ≤ j), 
  abs (∑ k in finset.Ico i j.succ, (ε k : ℝ) - x k) ≤ λ :=
by 
  sorry

end smallest_lambda_l576_576367


namespace matrix_characteristic_eq_l576_576852

noncomputable def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1, 2, 2], ![2, 1, 2], ![2, 2, 1]]

theorem matrix_characteristic_eq :
  ∃ (a b c : ℚ), a = -6 ∧ b = -12 ∧ c = -18 ∧ 
  (B ^ 3 + a • (B ^ 2) + b • B + c • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0) :=
by
  sorry

end matrix_characteristic_eq_l576_576852


namespace trail_length_proof_l576_576383

theorem trail_length_proof (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x1 + x2 = 28)
  (h2 : x2 + x3 = 30)
  (h3 : x3 + x4 + x5 = 42)
  (h4 : x1 + x4 = 30) :
  x1 + x2 + x3 + x4 + x5 = 70 := by
  sorry

end trail_length_proof_l576_576383


namespace distance_in_scientific_notation_l576_576052

theorem distance_in_scientific_notation :
  ∃ a n : ℝ, 1 ≤ |a| ∧ |a| < 10 ∧ n = 4 ∧ 38000 = a * 10^n ∧ a = 3.8 :=
by
  sorry

end distance_in_scientific_notation_l576_576052


namespace vasya_guarantee_l576_576023

-- Define the variables and their constraints
variables {x : ℕ → ℝ} -- x is a mapping from natural numbers to real numbers
variables {Pcards Vcards : set (set ℕ)} -- Sets of sets of natural numbers representing Petya's and Vasya's cards

-- Define the assignment constraint
axiom sorted_variables : ∀ i j, i ≤ j → x i ≤ x j
axiom value_range : ∀ i, 0 ≤ x i

-- Define the game rules: validity of the card sets
axiom valid_game : 
  (∀ c ∈ Pcards, c.card = 5 ∧ c ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) ∧
  (∀ c ∈ Vcards, c.card = 5 ∧ c ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) ∧
  (Pcards ∪ Vcards).card = 252 -- Number of combinations of 10 choose 5

-- Define the product for a given card
def card_product (c : set ℕ) := ∏ i in c, x i

-- Define the sum of the products of cards for Petya and Vasya
def sum_products (cards : set (set ℕ)) := ∑ c in cards, card_product c

-- The final theorem stating Vasya can always guarantee a higher sum
theorem vasya_guarantee : 
  ∃ (Vassign : ℕ → ℝ), 
  (∀ i, 0 ≤ Vassign i) ∧ 
  (∀ i j, i ≤ j → Vassign i ≤ Vassign j) → 
  (Pcards.card + Vcards.card = 252) → 
  sum_products Vcards > sum_products Pcards :=
sorry

end vasya_guarantee_l576_576023


namespace cookies_jim_ate_l576_576879

def MattBatch : Nat := 12
def FlourPerBatch : Nat := 2
def Bags : Nat := 4
def FlourPerBag : Nat := 5
def CookiesLeft : Nat := 105

theorem cookies_jim_ate (MattBatch = 12) (FlourPerBatch = 2) (Bags = 4) (FlourPerBag = 5) (CookiesLeft = 105) : 
  15 = (Bags * FlourPerBag / FlourPerBatch * MattBatch - CookiesLeft) :=
sorry

end cookies_jim_ate_l576_576879


namespace m_plus_n_is_23_l576_576848

noncomputable def find_m_plus_n : ℕ := 
  let A := 12
  let B := 4
  let C := 3
  let D := 3

  -- Declare the radius of E
  let radius_E : ℚ := (21 / 2)
  
  -- Let radius_E be written as m / n where m and n are relatively prime
  let (m : ℕ) := 21
  let (n : ℕ) := 2

  -- Calculate m + n
  m + n

theorem m_plus_n_is_23 : find_m_plus_n = 23 :=
by
  -- Proof is omitted
  sorry

end m_plus_n_is_23_l576_576848


namespace megan_initial_acorns_l576_576880

def initial_acorns (given_away left: ℕ) : ℕ := 
  given_away + left

theorem megan_initial_acorns :
  initial_acorns 7 9 = 16 := 
by 
  unfold initial_acorns
  rfl

end megan_initial_acorns_l576_576880


namespace equal_angles_BKM_CAM_l576_576893

/-- Let \( M \) be the midpoint of side \( BC \) in triangle \( ABC \). Let \( CL \) be perpendicular to \( AM \) with \( L \) lying between \( A \) and \( M \), and let \( K \) be a point on \( AM \) such that \( AK = 2LM \). Prove that \( \angle BKM = \angle CAM \). -/
theorem equal_angles_BKM_CAM (A B C M L K : Point) (ABC : Triangle A B C) 
  (hM : M = midpoint B C)
  (hCL_perp : perp (line C L) (line A M))
  (hL_between : between A L M)
  (hK_condition : dist A K = 2 * dist L M) : 
  angle B K M = angle C A M := 
sorry

end equal_angles_BKM_CAM_l576_576893


namespace amount_left_after_spending_l576_576089

-- Define the initial amount and percentage spent
def initial_amount : ℝ := 500
def percentage_spent : ℝ := 0.30

-- Define the proof statement that the amount left is 350
theorem amount_left_after_spending : 
  (initial_amount - (percentage_spent * initial_amount)) = 350 :=
by
  sorry

end amount_left_after_spending_l576_576089


namespace greatest_possible_value_of_x_l576_576413

theorem greatest_possible_value_of_x (x : ℕ) (h₁ : x % 4 = 0) (h₂ : x > 0) (h₃ : x^3 < 8000) :
  x ≤ 16 := by
  apply sorry

end greatest_possible_value_of_x_l576_576413


namespace geometric_seq_of_sqrt_value_of_t_values_of_a1_l576_576715

variable {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ} {t : ℝ}

-- Conditions
def pos_int (n : ℕ) := 0 < n
def seq_a_pos (n : ℕ) := a n > 0
def seq_a_condition (n : ℕ) := 4 * (n + 1 : ℝ) * (a n)^2 - (n : ℝ) * (a (n + 1))^2 = 0
def seq_b (n : ℕ) := b n = (a n)^2 / t^n

-- Theorem statement
theorem geometric_seq_of_sqrt (n : ℕ) (h₁ : pos_int n) (h₂ : seq_a_pos n) (h₃ : seq_a_condition n) :
  ∃ r, ∀ m, a (m+1) / (sqrt (m+1)) = r * (a m / (sqrt m)) := sorry

theorem value_of_t (h₁ : ∀ n, seq_a_pos n) (h₂ : ∀ n, seq_a_condition n) (h₃ : ∀ n, seq_b n)
  (h₄ : ∃d, ∀ m n : ℕ, b m - b n = d * (m - n)) :
  t = 4 := sorry

theorem values_of_a1 (h₁ : ∀ n, seq_a_pos n) (h₂ : ∀ n, seq_a_condition n)
  (h₃ : ∀ n, seq_b n) (h₄ : ∀ n, ∃d, ∀ m, b m - b n = d * (m - n))
  (h₅ : ∃ m (n : ℕ), 8 * a 1^2 * (S n) - (a 1)^4 * (n : ℝ)^2 = 16 * b m) :
  ∃ k : ℕ, a 1 = 2 * k := sorry

end geometric_seq_of_sqrt_value_of_t_values_of_a1_l576_576715


namespace minimum_trees_with_at_least_three_types_l576_576806

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end minimum_trees_with_at_least_three_types_l576_576806


namespace min_troublemakers_l576_576455

theorem min_troublemakers (students : Finset ℕ) (table : List ℕ) (is_liar : ℕ → Bool) :
  students.card = 29 ∧
  (∀ s ∈ students, 
    (∃ i j : ℕ, 
      table.nth i = some s ∧ table.nth ((i + 1) % 29) = some j ∧
      (∃ k : ℕ, 
        (is_liar s = false → count (/=is_liar k liar) = 0 ∧
        is_liar s = true → count (/=is_liar k liar) = 2))) → 
  (∃! l ∈ students, count (/=is_liar l liar) = 1))) → 
  ∃ t : ℕ, t = 10 :=
sorry

end min_troublemakers_l576_576455


namespace minimum_AP_BP_l576_576352

theorem minimum_AP_BP (A B : ℝ × ℝ) (P : ℝ × ℝ) 
  (hA : A = (1, 0)) 
  (hB : B = (5, 4)) 
  (hP : P.2 ^ 2 = 4 * P.1) : 
  ∃ (min_value : ℝ), min_value = 6 ∧ 
    ∀ (P : ℝ × ℝ), P.2 ^ 2 = 4 * P.1 → dist A P + dist B P ≥ min_value :=
by
  have h := Min_eq_6_of_AP_BP A B P hA hB hP
  exact ⟨6, h⟩
sorry

end minimum_AP_BP_l576_576352


namespace seeds_in_second_plot_l576_576227

theorem seeds_in_second_plot (S : ℕ) (h1 : 0.25 * 300 = 75)
  (h2 : ∀ S, 0.30 * S = 0.3 * S)
  (h3 : 0.27 * (300 + S) = 0.27 * (300 + S))
  (h4 : 75 + 0.3 * S = 0.27 * (300 + S)) : 
  S = 200 := sorry

end seeds_in_second_plot_l576_576227


namespace num_factors_P_l576_576008

def P : ℕ := 54^6 + 6 * 54^5 + 15 * 54^4 + 20 * 54^3 + 15 * 54^2 + 6 * 54 + 1

theorem num_factors_P : Nat.factors P = 49 := 
sorry

end num_factors_P_l576_576008


namespace geometric_mean_concave_l576_576127

variables {I : Type*} [linear_ordered_field I] {R : Type*} [linear_ordered_field R]
variables (f : I → R) (f1 f2 ... fn : I → R)
variables (n : ℕ)

-- A function is concave
def concave (f : I → R) : Prop :=
  ∀ (x y : I) (θ : R), 0 ≤ θ → θ ≤ 1 → f (θ * x + (1 - θ) * y) ≥ θ * f x + (1 - θ) * f y

-- Functions f1, ..., fn are all concave and have nonnegative values
variables (concave_f1 : concave f1)
variables (concave_f2 : concave f2)
-- Add similar definitions for all f1 to fn
variables (concave_fn : concave fn)

def all_nonnegative (f : I → R) : Prop := ∀ x : I, 0 ≤ f x
variables (nonnegative_f1 : all_nonnegative f1)
variables (nonnegative_f2 : all_nonnegative f2)
-- Add similar definitions for all f1 to fn
variables (nonnegative_fn : all_nonnegative fn)

noncomputable def geometric_mean (f1 f2 ... fn : I → R) (x : I) : R := 
  (f1 x * f2 x * ... * fn x) ^ (1 / n)

theorem geometric_mean_concave : 
  ∀ (x y : I) (θ : R), 0 ≤ θ → θ ≤ 1 →
    geometric_mean f1 f2 ... fn (θ * x + (1 - θ) * y) ≥ θ * geometric_mean f1 f2 ... fn x + (1 - θ) * geometric_mean f1 f2 ... fn y :=
sorry

end geometric_mean_concave_l576_576127


namespace min_troublemakers_29_l576_576503

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l576_576503


namespace sum_of_natural_numbers_eq_4005_l576_576177

theorem sum_of_natural_numbers_eq_4005 :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 4005 ∧ n = 89 :=
by
  sorry

end sum_of_natural_numbers_eq_4005_l576_576177


namespace integer_division_l576_576200

theorem integer_division (n : ℤ) (h : n ≥ 1) : (n ^ 2 ∣ 2 ^ n + 1 ↔ n = 1 ∨ n = 3) :=
sorry

end integer_division_l576_576200


namespace find_x_value_l576_576009

theorem find_x_value (b x : ℝ) (hb : b > 1) (hx : x > 0) 
    (heq: (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) : 
    x = 1 / 5 := 
sorry

end find_x_value_l576_576009


namespace evaluate_f_at_25pi_div_6_l576_576731

def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 3)

theorem evaluate_f_at_25pi_div_6 : f (25 * Real.pi / 6) = -Real.sqrt 3 := by
  sorry

end evaluate_f_at_25pi_div_6_l576_576731


namespace bicycle_price_after_discounts_l576_576386

noncomputable def originalPrice : ℝ := 200
noncomputable def firstDiscountRate : ℝ := 0.4
noncomputable def secondDiscountRate : ℝ := 0.25

theorem bicycle_price_after_discounts :
  let firstDiscount := originalPrice * firstDiscountRate;
      priceAfterFirstDiscount := originalPrice - firstDiscount; 
      secondDiscount := priceAfterFirstDiscount * secondDiscountRate;
      finalPrice := priceAfterFirstDiscount - secondDiscount
  in finalPrice = 90 := by
  let firstDiscount := originalPrice * firstDiscountRate
  let priceAfterFirstDiscount := originalPrice - firstDiscount
  let secondDiscount := priceAfterFirstDiscount * secondDiscountRate
  let finalPrice := priceAfterFirstDiscount - secondDiscount
  show finalPrice = 90 from by
  sorry

end bicycle_price_after_discounts_l576_576386


namespace find_ab_value_l576_576432

-- Definitions from the conditions
def ellipse_eq (a b : ℝ) : Prop := b^2 - a^2 = 25
def hyperbola_eq (a b : ℝ) : Prop := a^2 + b^2 = 49

-- Main theorem statement
theorem find_ab_value {a b : ℝ} (h_ellipse : ellipse_eq a b) (h_hyperbola : hyperbola_eq a b) : 
  |a * b| = 2 * Real.sqrt 111 :=
by
  -- Proof goes here
  sorry

end find_ab_value_l576_576432


namespace garden_perimeter_is_24_l576_576033

def perimeter_of_garden(a b c x: ℕ) (h1: a + b + c = 3) : ℕ :=
  3 + 5 + a + x + b + 4 + c + 4 + 5 - x

theorem garden_perimeter_is_24 (a b c x : ℕ) (h1 : a + b + c = 3) :
  perimeter_of_garden a b c x h1 = 24 :=
  by
  sorry

end garden_perimeter_is_24_l576_576033


namespace triangle_perimeter_is_720_l576_576067

-- Definitions corresponding to conditions
variables (x : ℕ)
noncomputable def shortest_side := 5 * x
noncomputable def middle_side := 6 * x
noncomputable def longest_side := 7 * x

-- Given the length of the longest side is 280 cm
axiom longest_side_eq : longest_side x = 280

-- Prove that the perimeter of the triangle is 720 cm
theorem triangle_perimeter_is_720 : 
  shortest_side x + middle_side x + longest_side x = 720 :=
by
  sorry

end triangle_perimeter_is_720_l576_576067


namespace min_liars_needed_l576_576472

namespace Problem

/-
  We introduce our problem using Lean definitions and statements.
-/

def students_sitting_at_table : ℕ := 29

structure Student :=
(is_liar : Bool)

def student_truth_statement (s: ℕ) (next1: Student) (next2: Student) : Bool :=
  if s then (next1.is_liar && !next2.is_liar) || (!next1.is_liar && next2.is_liar)
  else next1.is_liar && next2.is_liar

def minimum_liars_class (sitting_students: ℕ) : ℕ :=
  sitting_students / 3 + (if sitting_students % 3 = 0 then 0 else 1)

theorem min_liars_needed (students : List Student) (H_total : students.length = students_sitting_at_table)
  (H_truth_teller_claim : ∀ (i : ℕ), i < students.length → student_truth_statement i (students.get ⟨(i+1) % students.length, sorry⟩) (students.get ⟨(i+2) % students.length, sorry⟩)) :
  (students.count (λ s => s.is_liar)) ≥ minimum_liars_class students_sitting_at_table :=
sorry

end Problem

end min_liars_needed_l576_576472


namespace customer_paid_correct_amount_l576_576439

-- Define the cost price of the computer table
def cost_price : ℝ := 5000

-- Define the additional percentage charged by the owner
def additional_percentage : ℝ := 0.15

-- Define the total amount paid by the customer
def total_amount_paid := cost_price * (1 + additional_percentage)

theorem customer_paid_correct_amount :
  total_amount_paid = 5750 := 
by
  unfold total_amount_paid
  unfold cost_price
  unfold additional_percentage
  norm_num -- simplifies the calculation
  rfl -- goal is trivially true

end customer_paid_correct_amount_l576_576439


namespace total_exercise_time_l576_576627

-- Define the daily exercise structure
structure Exercise := 
  (jogging : Nat)
  (swimming : Nat)
  (kickboxing : Nat)

-- Define the initial conditions
def day1 : Exercise := { jogging := 30, swimming := 0, kickboxing := 0 }
def day2 : Exercise := { jogging := 35, swimming := 10, kickboxing := 0 }
def day3 : Exercise := { jogging := 30, swimming := 20, kickboxing := 0 }
def day4 : Exercise := { jogging := 35, swimming := 30, kickboxing := 0 }
def day5 : Exercise := { jogging := 40, swimming := 40, kickboxing := 80 }
def day6 : Exercise := { jogging := 45, swimming := 50, kickboxing := 0 }
def day7 : Exercise := { jogging := 50, swimming := 60, kickboxing := 0 }

-- Define the total time calculation in minutes
def total_time (days : List Exercise) : Nat :=
  days.foldl (λ acc day => acc + day.jogging + day.swimming + day.kickboxing) 0

-- Convert minutes to hours
def minutes_to_hours (minutes : Nat) : Float :=
  minutes / 60.0

-- List of exercises over the week
def weekly_exercises : List Exercise :=
  [day1, day2, day3, day4, day5, day6, day7]

-- Proven statement with sorry for proof placeholder
theorem total_exercise_time :
  minutes_to_hours (total_time weekly_exercises) = 9.25 :=
by {
  sorry
}

end total_exercise_time_l576_576627


namespace polynomial_coeffs_l576_576695

theorem polynomial_coeffs (a b c d e f : ℤ) :
  (((2 : ℤ) * x - 1) ^ 5 = a * x ^ 5 + b * x ^ 4 + c * x ^ 3 + d * x ^ 2 + e * x + f) →
  (a + b + c + d + e + f = 1) ∧ 
  (b + c + d + e = -30) ∧
  (a + c + e = 122) :=
by
  intro h
  sorry  -- Proof omitted

end polynomial_coeffs_l576_576695


namespace three_types_in_69_trees_l576_576784

variable (birches spruces pines aspens : ℕ)
variable (total_trees : ℕ := 100)
variable (all_trees : list (string × ℕ))

-- We assert that there are 100 trees in total, and our list of trees represents this
axiom h_total : ∑ t in all_trees, t.2 = total_trees

-- Among any 85 trees, there must be at least one of each type
axiom h_85_trees_all_types : ∀ (s : list (string × ℕ)), s.card = 85 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0 ∧ a > 0 ∧ 
  b + s + p + a = 85 ∧ (("birches", b) ∈ s ∧ ("spruces", s) ∈ s ∧ ("pines", p) ∈ s ∧ ("aspens", a) ∈ s))

-- We need to prove that any subset of 69 or more trees contains at least three different types.
theorem three_types_in_69_trees :
  ∀ (s : list (string × ℕ)), s.card = 69 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0) ∨ (b > 0 ∧ s > 0 ∧ a > 0) ∨ (b > 0 ∧ p > 0 ∧ a > 0) ∨ (s > 0 ∧ p > 0 ∧ a > 0) :=
begin
  sorry
end

end three_types_in_69_trees_l576_576784


namespace find_other_outlet_rate_l576_576614

noncomputable def other_outlet_rate (tank_volume_cubft : ℕ) (inlet_rate : ℕ) (outlet1_rate : ℕ) (emptying_time: ℕ) : ℕ :=
  let tank_volume_cubin := tank_volume_cubft * 1728
  let net_emptying_rate := (tank_volume_cubin / emptying_time) + inlet_rate
  net_emptying_rate - outlet1_rate

theorem find_other_outlet_rate :
  other_outlet_rate 30 3 6 3456 = 12 :=
by
  simp [other_outlet_rate]
  sorry

end find_other_outlet_rate_l576_576614


namespace complement_event_l576_576635

def total_students : ℕ := 4
def males : ℕ := 2
def females : ℕ := 2
def choose2 (n : ℕ) := n * (n - 1) / 2

noncomputable def eventA : ℕ := males * females
noncomputable def eventB : ℕ := choose2 males
noncomputable def eventC : ℕ := choose2 females

theorem complement_event {total_students males females : ℕ}
  (h_total : total_students = 4)
  (h_males : males = 2)
  (h_females : females = 2) :
  (total_students.choose 2 - (eventB + eventC)) / total_students.choose 2 = 1 / 3 :=
by
  sorry

end complement_event_l576_576635


namespace position_of_99_l576_576172

-- Define a function that describes the position of an odd number in the 5-column table.
def position_in_columns (n : ℕ) : ℕ := sorry  -- position in columns is defined by some rule

-- Now, state the theorem regarding the position of 99.
theorem position_of_99 : position_in_columns 99 = 3 := 
by 
  sorry  -- Proof goes here

end position_of_99_l576_576172


namespace mass_percentage_N_in_NH4I_l576_576209

/-- The molar mass of nitrogen in g/mol -/ 
def molar_mass_N : ℝ := 14.01

/-- The molar mass of hydrogen in g/mol -/
def molar_mass_H : ℝ := 1.01

/-- The molar mass of iodine in g/mol -/
def molar_mass_I : ℝ := 126.90

/-- The total molar mass of NH4I in g/mol -/
def molar_mass_NH4I : ℝ := molar_mass_N + 4 * molar_mass_H + molar_mass_I

/-- Assertion that the mass percentage of Nitrogen (N) in ammonium iodide (NH4I) is 9.66%. -/
theorem mass_percentage_N_in_NH4I : (molar_mass_N / molar_mass_NH4I) * 100 ≈ 9.66 := by
  sorry

end mass_percentage_N_in_NH4I_l576_576209


namespace subsets_of_012_correct_proper_subsets_of_012_correct_l576_576014

-- Define the set {0, 1, 2}
def set_012 : Set ℕ := {0, 1, 2}

-- Definition of subsets and proper subsets
def all_subsets_012 : Set (Set ℕ) := { 
  ∅, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2} 
}

def proper_subsets_012 : Set (Set ℕ) := { 
  ∅, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2} 
}

-- Theorem statements
theorem subsets_of_012_correct : 
  (∀ s : Set ℕ, s ∈ all_subsets_012) ↔ 
  (s = ∅ ∨ s = {0} ∨ s = {1} ∨ s = {2} ∨ s = {0, 1} ∨ s = {0, 2} ∨ s = {1, 2} ∨ s = {0, 1, 2}) := 
sorry

theorem proper_subsets_of_012_correct : 
  (∀ s : Set ℕ, s ∈ proper_subsets_012) ↔ 
  (s = ∅ ∨ s = {0} ∨ s = {1} ∨ s = {2} ∨ s = {0, 1} ∨ s = {0, 2} ∨ s = {1, 2}) := 
sorry

end subsets_of_012_correct_proper_subsets_of_012_correct_l576_576014


namespace cos_alpha_of_point_P_l576_576725

theorem cos_alpha_of_point_P :
  let α := real.arccos (1 / 2) in
  (x = 1 ∧ y = - real.sqrt 3 ∧ r = real.sqrt (x^2 + y^2) ∧ P = (1, - real.sqrt 3)) →
  real.cos α = 1 / 2 :=
by
  intro h
  sorry

end cos_alpha_of_point_P_l576_576725


namespace quadratic_roots_form_l576_576443

theorem quadratic_roots_form (d : ℝ) (h : (λ x, x^2 - 7 * x + d = 0) has roots of the form (λ x, x = (7 + real.sqrt(1 + d)) / 2) ∨ (λ x, x = (7 - real.sqrt(1 + d)) / 2)) : 
    d = 48 / 5 := 
sorry

end quadratic_roots_form_l576_576443


namespace guests_count_l576_576021

noncomputable def total_guests_at_start (G : ℕ) (women men children : ℕ) : Prop :=
  -- Conditions
  -- 1. Half of the guests are women.
  women = G / 2 ∧
  -- 2. 15 of the guests are men.
  men = 15 ∧
  -- 3. The rest of the guests are children.
  children = G - (women + men) ∧
  -- 4. 1/3 of the men and 5 children left in the middle of the celebration.
  let men_left := men / 3 in
  let men_stayed := men - men_left in
  let children_left := 5 in
  let children_stayed := children - children_left in
  -- 5. 50 people stayed and enjoyed the birthday celebration.
  50 = women + men_stayed + children_stayed

theorem guests_count (G women men children : ℕ) (h : total_guests_at_start G women men children) :
  G = 60 :=
by
  sorry

end guests_count_l576_576021


namespace minimum_trees_l576_576791

variable (Trees : Type) [Fintype Trees] [DecidableEq Trees]

def trees_in_grove : Nat := 100

def tree_type := {birches, spruces, pines, aspens} : Set Trees

def condition (s : Finset Trees) : Prop := 
  s.card > 85 → tree_type ⊆ s

theorem minimum_trees (s : Finset Trees) (H : condition s) : 
  ∃ (n : Nat), n ≤ trees_in_grove ∧ n ≥ 69 → 
  ∃ t ⊆ s, t.card = n ∧ (|t ∩ tree_type| >= 3) :=
sorry

end minimum_trees_l576_576791


namespace number_of_factors_of_n_l576_576006

-- Define n as provided in the conditions
def n : ℕ := 2^5 * 3^3 * 5^2 * 6^4

-- Main theorem to prove the number of natural-number factors of n is 240
theorem number_of_factors_of_n : nat.totient n = 240 := 
  sorry

end number_of_factors_of_n_l576_576006


namespace parallel_vectors_l576_576748

theorem parallel_vectors {m : ℝ} 
  (h : (2 * m + 1) / 2 = 3 / m): m = 3 / 2 ∨ m = -2 :=
by
  sorry

end parallel_vectors_l576_576748


namespace num_positive_integers_m_l576_576690

theorem num_positive_integers_m (h : ∀ m : ℕ, ∃ d : ℕ, 3087 = d ∧ m^2 = d + 3) :
  ∃! m : ℕ, 0 < m ∧ (3087 % (m^2 - 3) = 0) := by
  sorry

end num_positive_integers_m_l576_576690


namespace minimum_liars_l576_576477

noncomputable def minimum_troublemakers (n : ℕ) := 10

theorem minimum_liars (n : ℕ) (students : ℕ → bool) 
  (seated_at_round_table : ∀ i, students (i % 29) = tt ∨ students (i % 29) = ff)
  (one_liar_next_to : ∀ i, students i = tt → (students ((i + 1) % 29) = ff ∨ students ((i + 28) % 29) = ff))
  (two_liars_next_to : ∀ i, students i = ff → ((students ((i + 1) % 29) = ff) ∧ students ((i + 28) % 29) = ff)) :
  ∃ m, m = minimum_troublemakers 29 ∧ (m ≤ n) :=
by
  let liars_count := 10
  have : liars_count = 10 := rfl
  use liars_count
  exact ⟨rfl, sorry⟩

end minimum_liars_l576_576477


namespace part1_part2_l576_576274

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |5 - x|

theorem part1 : ∃ m, m = 9 / 2 ∧ ∀ x, f x ≥ m :=
sorry

theorem part2 (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 3) : 
  (1 / (a + 1) + 1 / (b + 2)) ≥ 2 / 3 :=
sorry

end part1_part2_l576_576274


namespace ratio_of_r_to_pq_l576_576105

theorem ratio_of_r_to_pq (p q r : ℕ) (h₁ : p + q + r = 7000) (h₂ : r = 2800) :
  r / (p + q) = 2 / 3 :=
by sorry

end ratio_of_r_to_pq_l576_576105


namespace inequality_of_thirds_of_ordered_triples_l576_576713

variable (a1 a2 a3 b1 b2 b3 : ℝ)

theorem inequality_of_thirds_of_ordered_triples 
  (h1 : a1 ≤ a2) 
  (h2 : a2 ≤ a3) 
  (h3 : b1 ≤ b2)
  (h4 : b2 ≤ b3)
  (h5 : a1 + a2 + a3 = b1 + b2 + b3)
  (h6 : a1 * a2 + a2 * a3 + a1 * a3 = b1 * b2 + b2 * b3 + b1 * b3)
  (h7 : a1 ≤ b1) : 
  a3 ≤ b3 := 
by 
  sorry

end inequality_of_thirds_of_ordered_triples_l576_576713


namespace range_of_f_l576_576270

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem range_of_f :
  set.range (λ x, f x) = set.Icc 1 5 :=
by {
  sorry
}

end range_of_f_l576_576270


namespace g_triple_evaluation_l576_576874

def g (x : ℤ) : ℤ := 
if x < 8 then x ^ 2 - 6 
else x - 15

theorem g_triple_evaluation :
  g (g (g 20)) = 4 :=
by sorry

end g_triple_evaluation_l576_576874


namespace y_is_multiple_of_16_y_is_multiple_of_8_y_is_multiple_of_4_y_is_multiple_of_2_l576_576856

def y : ℕ := 112 + 160 + 272 + 432 + 1040 + 1264 + 4256

theorem y_is_multiple_of_16 : y % 16 = 0 :=
sorry

theorem y_is_multiple_of_8 : y % 8 = 0 :=
sorry

theorem y_is_multiple_of_4 : y % 4 = 0 :=
sorry

theorem y_is_multiple_of_2 : y % 2 = 0 :=
sorry

end y_is_multiple_of_16_y_is_multiple_of_8_y_is_multiple_of_4_y_is_multiple_of_2_l576_576856


namespace min_troublemakers_l576_576491

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l576_576491


namespace hall_emptying_time_with_third_escalator_l576_576392

noncomputable def time_to_empty_hall (
  people_enter_uniformly: Prop,
  hall_initially_empty: Prop,
  hall_half_full_one_escalator: Prop,
  hall_full_two_escalators: Prop,
): ℕ :=
  60

theorem hall_emptying_time_with_third_escalator 
  (people_enter_uniformly: Prop)
  (hall_initially_empty: Prop)
  (hall_half_full_one_escalator: Prop)
  (hall_full_two_escalators: Prop)
: time_to_empty_hall people_enter_uniformly hall_initially_empty hall_half_full_one_escalator hall_full_two_escalators = 60 := by
  sorry

end hall_emptying_time_with_third_escalator_l576_576392


namespace find_radius_l576_576037

theorem find_radius :
  ∃ (r : ℝ), 
  (∀ (x : ℝ), y = x^2 + r) ∧ 
  (∀ (x : ℝ), y = x) ∧ 
  (∀ (x : ℝ), x^2 + r = x) ∧ 
  (∀ (x : ℝ), x^2 - x + r = 0 → (-1)^2 - 4 * 1 * r = 0) → 
  r = 1 / 4 :=
by
  sorry

end find_radius_l576_576037


namespace min_troublemakers_29_l576_576506

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l576_576506


namespace find_BC_l576_576419

variable (S α b : ℝ)

noncomputable def BC (S α b : ℝ) : ℝ :=
  sqrt ((4 * S ^ 2) / (b ^ 2 * (sin α) ^ 2) + b ^ 2 - 4 * S * cot α)

theorem find_BC (S α b : ℝ) (h1 : S > 0) (h2 : 0 < α < π) (h3 : b > 0) :
  BC S α b = sqrt ((4 * S ^ 2) / (b ^ 2 * (sin α) ^ 2) + b ^ 2 - 4 * S * cot α) :=
by
  sorry

end find_BC_l576_576419


namespace probability_of_blue_buttons_l576_576343

theorem probability_of_blue_buttons
  (orig_red_A : ℕ) (orig_blue_A : ℕ)
  (removed_red : ℕ) (removed_blue : ℕ)
  (target_ratio : ℚ)
  (final_red_A : ℕ) (final_blue_A : ℕ)
  (final_red_B : ℕ) (final_blue_B : ℕ)
  (orig_buttons_A : orig_red_A + orig_blue_A = 16)
  (removed_buttons : removed_red = 3 ∧ removed_blue = 5)
  (final_buttons_A : final_red_A + final_blue_A = 8)
  (buttons_ratio : target_ratio = 2 / 3)
  (final_ratio_A : final_red_A + final_blue_A = target_ratio * 16)
  (red_in_A : final_red_A = orig_red_A - removed_red)
  (blue_in_A : final_blue_A = orig_blue_A - removed_blue)
  (red_in_B : final_red_B = removed_red)
  (blue_in_B : final_blue_B = removed_blue):
  (final_blue_A / (final_red_A + final_blue_A)) * (final_blue_B / (final_red_B + final_blue_B)) = 25 / 64 := 
by
  sorry

end probability_of_blue_buttons_l576_576343


namespace square_side_length_eq_8_over_pi_l576_576928

noncomputable def side_length_square : ℝ := 8 / Real.pi

theorem square_side_length_eq_8_over_pi :
  ∀ (s : ℝ),
  (4 * s = (Real.pi * (s / Real.sqrt 2) ^ 2) / 2) →
  s = side_length_square :=
by
  intro s h
  sorry

end square_side_length_eq_8_over_pi_l576_576928


namespace strictly_increasing_sequences_exists_l576_576941

theorem strictly_increasing_sequences_exists (n : ℕ) (a : ℕ → ℕ) 
  (h : ∀ k : ℕ, count (fun m => a m = k) (list.range (2 * n)) ≤ n) : 
  ∃ (b c : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ n → b i < c i) ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ n → a (b i) ≠ a (c i)) ∧ ∀ i : ℕ, i < 2 * n → (∃ j, b j = i ∨ c j = i) := 
sorry

end strictly_increasing_sequences_exists_l576_576941


namespace remainder_when_added_then_divided_l576_576541

def num1 : ℕ := 2058167
def num2 : ℕ := 934
def divisor : ℕ := 8

theorem remainder_when_added_then_divided :
  (num1 + num2) % divisor = 5 := 
sorry

end remainder_when_added_then_divided_l576_576541


namespace problem_condition_sufficient_not_necessary_l576_576111

open real

theorem problem_condition_sufficient_not_necessary :
  (∃ φ, φ = π ∧ ∀ x, sin (2 * x + φ) = 0 → x = 0) ∧ (∃ φ, φ ≠ π ∧ ∀ x, sin (2 * x + φ) = 0 → x = 0) :=
by
  sorry

end problem_condition_sufficient_not_necessary_l576_576111


namespace parabola_direction_l576_576931

noncomputable def question_conditions_answer (a : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x₁ x₂ : ℝ, 4 * a * x₁^2 + 4 * (a + 1) * x₁ + (a^2 + a + 3 + 1/a) = 0 ∧
                    4 * a * x₂^2 + 4 * (a + 1) * x₂ + (a^2 + a + 3 + 1/a) = 0) :
  Prop :=
4 * a < 0

theorem parabola_direction (a : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x₁ x₂ : ℝ, 4 * a * x₁^2 + 4 * (a + 1) * x₁ + (a^2 + a + 3 + 1/a) = 0 ∧
                    4 * a * x₂^2 + 4 * (a + 1) * x₂ + (a^2 + a + 3 + 1/a) = 0) :
  question_conditions_answer a h₁ h₂ :=
sorry

end parabola_direction_l576_576931


namespace books_sold_in_february_l576_576120

theorem books_sold_in_february (F : ℕ) 
  (h_avg : (15 + F + 17) / 3 = 16): 
  F = 16 := 
by 
  sorry

end books_sold_in_february_l576_576120


namespace min_liars_needed_l576_576467

namespace Problem

/-
  We introduce our problem using Lean definitions and statements.
-/

def students_sitting_at_table : ℕ := 29

structure Student :=
(is_liar : Bool)

def student_truth_statement (s: ℕ) (next1: Student) (next2: Student) : Bool :=
  if s then (next1.is_liar && !next2.is_liar) || (!next1.is_liar && next2.is_liar)
  else next1.is_liar && next2.is_liar

def minimum_liars_class (sitting_students: ℕ) : ℕ :=
  sitting_students / 3 + (if sitting_students % 3 = 0 then 0 else 1)

theorem min_liars_needed (students : List Student) (H_total : students.length = students_sitting_at_table)
  (H_truth_teller_claim : ∀ (i : ℕ), i < students.length → student_truth_statement i (students.get ⟨(i+1) % students.length, sorry⟩) (students.get ⟨(i+2) % students.length, sorry⟩)) :
  (students.count (λ s => s.is_liar)) ≥ minimum_liars_class students_sitting_at_table :=
sorry

end Problem

end min_liars_needed_l576_576467


namespace products_arrangement_count_l576_576959

/--
There are five different products: A, B, C, D, and E arranged in a row on a shelf.
- Products A and B must be adjacent.
- Products C and D must not be adjacent.
Prove that there are a total of 24 distinct valid arrangements under these conditions.
-/
theorem products_arrangement_count : 
  ∃ (n : ℕ), 
  (∀ (A B C D E : Type), n = 24 ∧
  ∀ l : List (Type), l = [A, B, C, D, E] ∧
  -- A and B must be adjacent
  (∀ p : List (Type), p = [A, B] ∨ p = [B, A]) ∧
  -- C and D must not be adjacent
  ¬ (∀ q : List (Type), q = [C, D] ∨ q = [D, C])) :=
sorry

end products_arrangement_count_l576_576959


namespace buckets_required_l576_576080

theorem buckets_required (C : ℚ) (N : ℕ) (h : 250 * (4/5 : ℚ) * C = N * C) : N = 200 :=
by
  sorry

end buckets_required_l576_576080


namespace minimum_liars_l576_576482

noncomputable def minimum_troublemakers (n : ℕ) := 10

theorem minimum_liars (n : ℕ) (students : ℕ → bool) 
  (seated_at_round_table : ∀ i, students (i % 29) = tt ∨ students (i % 29) = ff)
  (one_liar_next_to : ∀ i, students i = tt → (students ((i + 1) % 29) = ff ∨ students ((i + 28) % 29) = ff))
  (two_liars_next_to : ∀ i, students i = ff → ((students ((i + 1) % 29) = ff) ∧ students ((i + 28) % 29) = ff)) :
  ∃ m, m = minimum_troublemakers 29 ∧ (m ≤ n) :=
by
  let liars_count := 10
  have : liars_count = 10 := rfl
  use liars_count
  exact ⟨rfl, sorry⟩

end minimum_liars_l576_576482


namespace find_multiple_of_y_l576_576317

noncomputable def multiple_of_y (q m : ℝ) : Prop :=
  ∀ x y : ℝ, (x = 5 - q) → (y = m * q - 1) → (q = 1) → (x = 3 * y) → (m = 7 / 3)

theorem find_multiple_of_y :
  multiple_of_y 1 (7 / 3) :=
by
  sorry

end find_multiple_of_y_l576_576317


namespace prob_consecutive_without_replacement_prob_consecutive_with_replacement_l576_576582

-- Define the set of tags
def tags : Set ℕ := {1, 2, 3, 4, 5}

-- Define a function to check if two numbers are consecutive
def consecutive (a b : ℕ) : Prop := (a + 1 = b) ∨ (b + 1 = a)

-- Probability calculation without replacement
theorem prob_consecutive_without_replacement : 
  (n : ℕ) (h1 : n = 8) (total : ℕ) (h2 : total = 20), n / total = 2 / 5 :=
by
  sorry

-- Probability calculation with replacement
theorem prob_consecutive_with_replacement : 
  (n : ℕ) (h1 : n = 8) (total : ℕ) (h2 : total = 25), n / total = 8 / 25 :=
by
  sorry

end prob_consecutive_without_replacement_prob_consecutive_with_replacement_l576_576582


namespace min_troublemakers_l576_576493

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l576_576493


namespace line_passes_through_fixed_point_l576_576835

theorem line_passes_through_fixed_point {m n k : ℝ} (hmn : m * n = 3) 
  (hM : ∀ x y : ℝ, (x / 2) ^ 2 + (y / √3) ^ 2 = 1 → x ≠ 2 ∧ x ≠ -2)
  (hF2 : F2 = (1, 0))
  (hl : ∀ x y : ℝ, y = k * x + m)
  (hPQ : ∀ x1 x2 y1 y2 : ℝ, (x1 ^ 2 / 4 + y1 ^ 2 / 3 = 1) ∧ (x2 ^ 2 / 4 + y2 ^ 2 / 3 = 1) ∧ ((k * x1 + m) / (x1 - 1) + (k * x2 + m) / (x2 - 1) = 0))
  (halpha_beta_pi : α + β = real.pi) : ∃ x y : ℝ, l = (x, y) ∧ x = 4 ∧ y = 0 := 
sorry

end line_passes_through_fixed_point_l576_576835


namespace main_proof_l576_576550

-- Definitions and conditions
def hyperbola_foci (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2)
def ellipse_foci (a b : ℝ) : ℝ := real.sqrt (a^2 - b^2)
def check_asymptotic_lines (a b : ℝ) : Prop := ∃ d, d = b / a 
def is_ellipse_with_foci_on_x_axis (k : ℝ) : Prop := 1 < k ∧ k < 2

-- Main theorem statements
theorem main_proof : 
  (hyperbola_foci 15 1 ≠ ellipse_foci 25 9) = false ∧
  (∀ (F1 F2 : ℝ), F1 = -1 ∧ F2 = 1 → ∀ P : ℝ, ∃ y_P, (P = (0, y_P) → abs y_P ≠ 2)) ∧
  check_asymptotic_lines 2 (2*real.sqrt 3) ∧
  is_ellipse_with_foci_on_x_axis k
:= 
by 
  sorry

end main_proof_l576_576550


namespace xiaoyu_reading_days_l576_576987

theorem xiaoyu_reading_days
  (h1 : ∀ (p d : ℕ), p = 15 → d = 24 → p * d = 360)
  (h2 : ∀ (p t : ℕ), t = 360 → p = 18 → t / p = 20) :
  ∀ d : ℕ, d = 20 :=
by
  sorry

end xiaoyu_reading_days_l576_576987


namespace hyperbola_point_properties_l576_576001

theorem hyperbola_point_properties (x y : ℝ) (λ₁ λ₂ : ℝ) (k : ℝ) :
    (∀ (x y : ℝ), (x^2 / 3) - y^2 = 1 → x > 0 → y > 0)
    → (exists P : ℝ × ℝ, 
        P.1 = x ∧ P.2 = y ∧ (x^2 / 3) - y^2 = 1 ∧ x > 0 ∧ y > 0) 
    → 
    (λ₁ + λ₂ = -14) 
    ∧ 
    (∃ k : ℝ, (∀ (k : ℝ), k = -(x / (21 * y)) 
        ∧ x / y ∈ (sqrt 3, ⊤)
        → k ∈ (-∞, - (sqrt 3 / 21)))) := 
begin
  sorry
end

end hyperbola_point_properties_l576_576001


namespace area_of_rhombus_l576_576328

-- Define the vertices of the rhombus
def A := (1.2, 4.1)
def B := (7.3, 2.5)
def C := (1.2, -2.8)
def D := (-4.9, 2.5)

-- Define the vectors formed by the vertices
def vector1 := (B.1 - A.1, B.2 - A.2)
def vector2 := (D.1 - A.1, D.2 - A.2)

-- Area of rhombus proof statement
theorem area_of_rhombus :
  abs ((vector1.1 * vector2.2) - (vector1.2 * vector2.1)) = 19.52 := 
sorry

end area_of_rhombus_l576_576328


namespace min_positive_numbers_in_circle_l576_576323

theorem min_positive_numbers_in_circle (nums : Fin 101 → ℤ) 
  (h : ∀ i : Fin 101, 2 ≤ ((Finset.range 5).image (λ k => nums ((i + k) % 101))).filter (λ x => 0 < x).card) : 
  41 ≤ (Finset.univ.filter (λ i => 0 < nums i)).card :=
sorry

end min_positive_numbers_in_circle_l576_576323


namespace number_of_pairs_l576_576256

open Nat

theorem number_of_pairs :
  ∃ n, n = 9 ∧
    (∃ x y : ℕ,
      x > 0 ∧ y > 0 ∧
      x + y = 150 ∧
      x % 3 = 0 ∧
      y % 5 = 0 ∧
      (∃! (x y : ℕ), x + y = 150 ∧ x % 3 = 0 ∧ y % 5 = 0 ∧ x > 0 ∧ y > 0)) := sorry

end number_of_pairs_l576_576256


namespace greatest_solution_l576_576682

open Real

noncomputable def max_solution_in_interval : ℝ :=
  let sol1 := (π / 6) in
  let sol2 := (π / 6 + π) in
  let sol3 := (π / 6 + 2 * π) in
  let sol4 := (π / 6 + 3 * π) in
  let sol5 := (π / 6 + 4 * π) in
  let sol6 := (π / 6 + 5 * π) in
  max 0 (max sol1 (max sol2 (max sol3 (max sol4 (max sol5 sol6)))))

theorem greatest_solution : (|2 * sin max_solution_in_interval - 1| + |2 * cos (2 * max_solution_in_interval) - 1| = 0) ∧ (round (max_solution_in_interval * 10^3) / 10^3 = 27.7) :=
by
  let sols := [π / 6, 13 * π / 6, 25 * π / 6, 37 * π / 6, 49 * π / 6, 61 * π / 6]
  have h_solutions : ∀ x ∈ sols, (|2 * sin x - 1| + |2 * cos (2 * x) - 1| = 0) := sorry
  have h_max : max_solution_in_interval = 61 * π / 6 := sorry
  simp [max_solution_in_interval, h_max]
  have h_rounded : round (61 * π / 6 * 10^3) / 10^3 = 27.7 := sorry
  exact ⟨h_solutions _, h_rounded⟩

end greatest_solution_l576_576682


namespace part1_part2_l576_576275

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |5 - x|

theorem part1 : ∃ m, m = 9 / 2 ∧ ∀ x, f x ≥ m :=
sorry

theorem part2 (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 3) : 
  (1 / (a + 1) + 1 / (b + 2)) ≥ 2 / 3 :=
sorry

end part1_part2_l576_576275


namespace seventh_observation_is_five_l576_576107

theorem seventh_observation_is_five (avg6 : ℕ) (new_avg : ℕ) (sum6 sum7 : ℕ) (seventh_obs : ℕ) :
  avg6 = 12 → new_avg = 11 → sum6 = 6 * avg6 → sum7 = 7 * new_avg → seventh_obs = sum7 - sum6 → seventh_obs = 5 :=
begin
  sorry
end

end seventh_observation_is_five_l576_576107


namespace trader_gain_pens_l576_576630

theorem trader_gain_pens (C S : ℝ) (h1 : S = 1.25 * C) 
                         (h2 : 80 * S = 100 * C) : S - C = 0.25 * C :=
by
  have h3 : S = 1.25 * C := h1
  have h4 : 80 * S = 100 * C := h2
  sorry

end trader_gain_pens_l576_576630


namespace savings_if_together_l576_576162

def price_per_window : ℕ := 150

def discount_offer (n : ℕ) : ℕ := n - n / 7

def cost (n : ℕ) : ℕ := price_per_window * discount_offer n

def alice_windows : ℕ := 9
def bob_windows : ℕ := 10

def separate_cost : ℕ := cost alice_windows + cost bob_windows

def total_windows : ℕ := alice_windows + bob_windows

def together_cost : ℕ := cost total_windows

def savings : ℕ := separate_cost - together_cost

theorem savings_if_together : savings = 150 := by
  sorry

end savings_if_together_l576_576162


namespace operation_on_original_number_l576_576605

theorem operation_on_original_number (f : ℕ → ℕ) (x : ℕ) (h : 3 * (f x + 9) = 51) (hx : x = 4) :
  f x = 2 * x :=
by
  sorry

end operation_on_original_number_l576_576605


namespace B_work_time_l576_576570

theorem B_work_time :
  (∀ A_efficiency : ℝ, A_efficiency = 1 / 12 → ∀ B_efficiency : ℝ, B_efficiency = A_efficiency * 1.2 → (1 / B_efficiency = 10)) :=
by
  intros A_efficiency A_efficiency_eq B_efficiency B_efficiency_eq
  sorry

end B_work_time_l576_576570


namespace rectangle_area_l576_576160

theorem rectangle_area (square_area : ℝ) (rectangle_length_ratio : ℝ) (square_area_eq : square_area = 36)
  (rectangle_length_ratio_eq : rectangle_length_ratio = 3) :
  ∃ (rectangle_area : ℝ), rectangle_area = 108 :=
by
  -- Extract the side length of the square from its area
  let side_length := real.sqrt square_area
  have side_length_eq : side_length = 6, from calc
    side_length = real.sqrt 36 : by rw [square_area_eq]
    ... = 6 : real.sqrt_eq 6 (by norm_num)
  -- Calculate the rectangle's width
  let width := side_length
  -- Calculate the rectangle's length
  let length := rectangle_length_ratio * width
  -- Calculate the area of the rectangle
  let area := width * length
  -- Prove the area is 108
  use area
  calc area
    = 6 * (3 * 6) : by rw [side_length_eq, rectangle_length_ratio_eq]
    ... = 108 : by norm_num

end rectangle_area_l576_576160


namespace midpoints_form_equilateral_triangle_l576_576913

variables {A B C D O M N P : Type*} [AddGroup A] [AddGroup B] [AddGroup C]
           [AddGroup D] [AddGroup O] [AddGroup M] [AddGroup N] [AddGroup P]
           [Ring A] [Ring B] [Ring C] [Ring D] [Ring O] [Ring M] [Ring N] [Ring P]

-- The conditions: A geometric description of the isosceles trapezoid ABCD
def isosceles_trapezoid (A B C D : Type*) : Prop :=
  -- State that the sides AB and CD are parallel
  (parallel A B C D) ∧ 
  -- State the diagonals intersect at an angle of 60 degrees
  ∃ O, angle A O C = 60 ∧ angle B O D = 60

-- Midpoints definitions
def midpoint (x y : Type*) : Type* := sorry

-- Defining the midpoints M, N of OA and OD, and P of BC
def midpoint_OA := midpoint O A
def midpoint_OD := midpoint O D
def midpoint_BC := midpoint B C

-- Now to prove the final statement
theorem midpoints_form_equilateral_triangle
  (h_trap : isosceles_trapezoid A B C D) :
  equilateral_triangle (midpoint_OA) (midpoint_OD) (midpoint_BC) :=
sorry

end midpoints_form_equilateral_triangle_l576_576913


namespace least_three_digit_multiple_of_9_eq_108_l576_576536

/--
What is the least positive three-digit multiple of 9?
-/
theorem least_three_digit_multiple_of_9_eq_108 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 9 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 9 = 0 → n ≤ m :=
  ∃ n : ℕ, n = 108 :=
begin
  sorry
end

end least_three_digit_multiple_of_9_eq_108_l576_576536


namespace possible_digits_l576_576390

theorem possible_digits (a b c : ℕ) (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : (∀ (x y z : ℕ), (a b c ∈ {x,y,z}) → even (10 * x + y + 100 * z))
  (h3 : (∃ (x y z : ℕ), (a b c ∈ {x,y,z}) ∧ ((x + y + z) % 3 = 0) ∧ (alternating_sum x y z % 11 = 0)) :
  {a, b, c} = {2, 4, 6} :=
begin
  sorry
end

def alternating_sum (x y z : ℕ) : ℕ :=
  (x - y + z) % 11

end possible_digits_l576_576390


namespace rectangle_area_l576_576136

theorem rectangle_area (a w l : ℝ) (h_square_area : a = 36) 
    (h_rect_width : w * w = a) 
    (h_rect_length : l = 3 * w) : w * l = 108 := 
sorry

end rectangle_area_l576_576136


namespace partition_squares_l576_576388

theorem partition_squares (squares : Finset Square) (k : ℕ) 
  (h_intersecting : ∀ (sq : Finset Square), sq.card = k + 1 → ∃ (sq1 sq2 : Square), sq1 ∈ sq ∧ sq2 ∈ sq ∧ sq1 ≠ sq2 ∧ sq1.intersects sq2) :
  ∃ (partition : Finset (Finset Square)), 
  partition.card ≤ 2 * k - 1 ∧ 
  ∀ subset ∈ partition, ∃ (common_point : Point), ∀ square ∈ subset, common_point ∈ square :=
sorry

end partition_squares_l576_576388


namespace probability_below_8_l576_576612

def prob_hit_10 := 0.20
def prob_hit_9 := 0.30
def prob_hit_8 := 0.10

theorem probability_below_8 : (1 - (prob_hit_10 + prob_hit_9 + prob_hit_8) = 0.40) :=
by
  sorry

end probability_below_8_l576_576612


namespace problem_solution_l576_576255

theorem problem_solution:
  (∀ x : ℝ, f (-x) = -f x) →                 -- f is an odd function
  (∀ x : ℝ, f (x + 1) = f (- (x + 1))) →     -- f(x + 1) is an even function
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 0 → f x = x^3) →    -- f(x) = x^3 for -1 ≤ x ≤ 0
  (∀ x : ℝ, f (x + 4) = f x) →               -- f is periodic with period 4
  f (9/2) = (1/8) :=                         -- the goal to prove
by
  sorry

end problem_solution_l576_576255


namespace divisibility_sum_of_powers_l576_576029

theorem divisibility_sum_of_powers (S T : ℕ) (n : ℕ) (h1 : S = ∑ k in Finset.range n, k.succ ^ n)
  (h2 : T = ∑ k in Finset.range n, k.succ) : 
  (T ∣ S) := by
sorry

end divisibility_sum_of_powers_l576_576029


namespace trigonometric_identity_l576_576196

theorem trigonometric_identity :
  sin 18 * sin 78 - cos 162 * cos 78 = 1 / 2 := 
by sorry

end trigonometric_identity_l576_576196


namespace min_trees_for_three_types_l576_576798

-- Define types and trees
inductive TreeType
| birch | spruce | pine | aspen
deriving Inhabited, DecidableEq

-- A grove with 100 trees of any of the four types.
structure Tree :=
(type : TreeType)

constant grove : List Tree
axiom grove_size : grove.length = 100

-- Condition: Among any 85 trees, there are trees of all four types.
axiom all_types_in_any_85 : ∀ (s : Finset Tree), s.card = 85 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, ∃ t4 ∈ s, 
      t1.type = TreeType.birch ∧
      t2.type = TreeType.spruce ∧
      t3.type = TreeType.pine ∧
      t4.type = TreeType.aspen

-- We need to show that at least 69 trees are needed to ensure at least 3 types.
theorem min_trees_for_three_types : 
  ∀ (s : Finset Tree), s.card = 69 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, 
      t1.type ≠ t2.type ∧ t2.type ≠ t3.type ∧ t1.type ≠ t3.type := 
sorry

end min_trees_for_three_types_l576_576798


namespace correct_propositions_l576_576267

theorem correct_propositions 
  (conjugate_real : ∀ (x : ℝ), x.conj = x)
  (ellipse_trajectory : ∀ (z : ℂ), (|z - Complex.I| + |z + Complex.I| = 2) → False)
  (summation_zero : ∀ (m : ℤ), (Complex.I ^ m + Complex.I ^ (m + 1) + Complex.I ^ (m + 2) + Complex.I ^ (m + 3) = 0))
  (imaginary_part : ∀ (a b : ℝ), ∃ (Z : ℂ), (Z = a + b * Complex.I) ∧ (imaginaryPart Z = b) ∧ (false → imaginaryPart Z = Complex.I)) : 
  (conjugate_real ∧ summation_zero) ∧ (¬ellipse_trajectory ∧ ¬imaginary_part) :=
by
  sorry

end correct_propositions_l576_576267


namespace distance_between_foci_of_hyperbola_is_correct_l576_576674

noncomputable def distance_between_foci_of_hyperbola : ℝ := 
  let a_sq := 50
  let b_sq := 8
  let c_sq := a_sq + b_sq
  let c := Real.sqrt c_sq
  2 * c

theorem distance_between_foci_of_hyperbola_is_correct :
  distance_between_foci_of_hyperbola = 2 * Real.sqrt 58 :=
by
  sorry

end distance_between_foci_of_hyperbola_is_correct_l576_576674


namespace smallest_A_for_quadratic_poly_l576_576220

theorem smallest_A_for_quadratic_poly (f : ℝ → ℝ) (h_quad : ∀ x : ℝ, f x = a*x^2 + b*x + c) 
(h_bound : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) : 
∃ A, (∀ (a b c : ℝ), f'(0) ≤ A) ∧ A = 8 := sorry

end smallest_A_for_quadratic_poly_l576_576220


namespace average_remaining_two_numbers_l576_576422

theorem average_remaining_two_numbers (numbers : Fin 6 → ℝ)
  (h_avg_total : (∑ i, numbers i) / 6 = 3.9)
  (h_avg_first_two : (numbers 0 + numbers 1) / 2 = 3.4)
  (h_avg_second_two : (numbers 2 + numbers 3) / 2 = 3.85) : 
  (numbers 4 + numbers 5) / 2 = 4.45 :=
sorry

end average_remaining_two_numbers_l576_576422


namespace athlete_stops_at_Q_l576_576619

/-- Definition for the dimensions of the rectangular park --/
def park_dimensions : ℕ × ℕ := (900, 600)

/-- Definition for the total running distance in meters --/
def total_running_distance : ℕ := 15500

/-- Definition for the starting point distance from one vertex --/
def starting_point_distance : ℕ := 550

/-- Definition for the stopping point --/
def stopping_point (d : ℕ × ℕ) (trd spd : ℕ) (P Q : ℕ) : Prop :=
  let perimeter := 2 * d.1 + 2 * d.2 in
  let complete_laps := trd / perimeter in
  let remaining_distance := trd - complete_laps * perimeter in
  let total_initial_distance := spd + remaining_distance in
  Q = total_initial_distance%perimeter

theorem athlete_stops_at_Q
  (d : ℕ × ℕ := park_dimensions)
  (trd : ℕ := total_running_distance)
  (spd : ℕ := starting_point_distance) :
  ∃ Q : ℕ, stopping_point d trd spd 150 :=
sorry

end athlete_stops_at_Q_l576_576619


namespace triangle_angle_EDL_right_l576_576649

theorem triangle_angle_EDL_right (A B C M L D E : Type) [triangle ABC] (midpoint_M_AC : midpoint M A C) 
  (foot_L_bisector_B : foot L B) (line_parallel_AB_M : is_parallel (line AB) (line_through M parallel_to AB)) 
  (intersection_BL_D : intersects (line_through BL) (line_through M parallel_to AB) D)
  (intersection_BM_E : intersects (line_through BM) (line_through L parallel_to BC) E) 
  (AB_gt_BC : length_AB > length_BC) : 
  angle_EDL = 90 :=
sorry

end triangle_angle_EDL_right_l576_576649


namespace grove_tree_selection_l576_576829

theorem grove_tree_selection (birches spruces pines aspens : ℕ) :
  birches + spruces + pines + aspens = 100 →
  (∀ s : set ℕ, s.card = 85 → (birches ∈ s ∧ spruces ∈ s ∧ pines ∈ s ∧ aspens ∈ s)) →
  ∀ t : set ℕ, t.card = 69 → (birches ∈ t ∧ spruces ∈ t) ∨ (birches ∈ t ∧ pines ∈ t) ∨ (birches ∈ t ∧ aspens ∈ t) ∨ (spruces ∈ t ∧ pines ∈ t) ∨ (spruces ∈ t ∧ aspens ∈ t) ∨ (pines ∈ t ∧ aspens ∈ t) :=
sorry

end grove_tree_selection_l576_576829


namespace scientific_notation_conversion_l576_576664

theorem scientific_notation_conversion : 
  ∀ (n : ℝ), n = 1.8 * 10^8 → n = 180000000 :=
by
  intros n h
  sorry

end scientific_notation_conversion_l576_576664


namespace min_troublemakers_l576_576454

theorem min_troublemakers (students : Finset ℕ) (table : List ℕ) (is_liar : ℕ → Bool) :
  students.card = 29 ∧
  (∀ s ∈ students, 
    (∃ i j : ℕ, 
      table.nth i = some s ∧ table.nth ((i + 1) % 29) = some j ∧
      (∃ k : ℕ, 
        (is_liar s = false → count (/=is_liar k liar) = 0 ∧
        is_liar s = true → count (/=is_liar k liar) = 2))) → 
  (∃! l ∈ students, count (/=is_liar l liar) = 1))) → 
  ∃ t : ℕ, t = 10 :=
sorry

end min_troublemakers_l576_576454


namespace arithmetic_sequence_value_l576_576259

theorem arithmetic_sequence_value (a : ℝ) 
  (h1 : 2 * (2 * a + 1) = (a - 1) + (a + 4)) : a = 1 / 2 := 
by 
  sorry

end arithmetic_sequence_value_l576_576259


namespace minimum_number_of_troublemakers_l576_576483

theorem minimum_number_of_troublemakers (n m : ℕ) (students : Fin n → Prop) 
  (truthtellers : Fin n → Prop) (liars : Fin n → Prop) (round_table : List (Fin n))
  (h_all_students : n = 29)
  (h_truth_or_liar : ∀ s, students s → (truthtellers s ∨ liars s))
  (h_truth_tellers_conditions : ∀ s, truthtellers s → 
    (count_lies_next_to s = 1 ∨ count_lies_next_to s = 2))
  (h_liars_conditions : ∀ s, liars s → 
    (count_lies_next_to s ≠ 1 ∧ count_lies_next_to s ≠ 2))
  (h_round_table_conditions : round_table.length = n 
    ∧ (∀ i, i < n → round_table.nth i ≠ none)):
  m = 10 :=
by
  sorry

end minimum_number_of_troublemakers_l576_576483


namespace number_of_warehouses_on_straight_road_l576_576888

theorem number_of_warehouses_on_straight_road
  (distance_between_warehouses : ℕ = 1)
  (load_capacity : ℕ = 8)
  (goods_per_warehouse : ℕ = 8)
  (odd_number_of_warehouses : ∃ n : ℕ, n % 2 = 1)
  (optimal_route_travel_kilometers : ℕ = 300) :
  ∃ n : ℕ, n = 25 :=
begin
  -- We will prove this theorem by verifying the given conditions,
  -- using them to derive the number of warehouses as 25.
  sorry
end

end number_of_warehouses_on_straight_road_l576_576888


namespace Hay_s_Linens_sales_l576_576749

theorem Hay_s_Linens_sales :
  ∃ (n : ℕ), 500 ≤ 52 * n ∧ 52 * n ≤ 700 ∧
             ∀ m, (500 ≤ 52 * m ∧ 52 * m ≤ 700) → n ≤ m :=
sorry

end Hay_s_Linens_sales_l576_576749


namespace least_positive_three_digit_multiple_of_9_l576_576534

theorem least_positive_three_digit_multiple_of_9 : 
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ n % 9 = 0 ∧ ∀ m : ℕ, (m >= 100 ∧ m < 1000 ∧ m % 9 = 0) → n ≤ m :=
begin
  use 108,
  split,
  { exact nat.le_refl 108 },
  split,
  { exact nat.lt_of_lt_of_le (nat.succ_pos 9) (nat.succ_le_succ (nat.le_refl 99)) },
  split,
  { exact nat.mod_eq_zero_of_mk (nat.zero_of_succ_pos 12) },
  { intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    exact nat.le_of_eq ((nat.mod_eq_zero_of_dvd (nat.gcd_eq_gcd_ab (12) (8) (1)))),
  },
  sorry
end

end least_positive_three_digit_multiple_of_9_l576_576534


namespace average_speed_boat_in_still_water_l576_576577

variables (x : ℝ) -- average speed of the boat in still water

-- Conditions
def speed_with_current := x + 3
def speed_against_current := x - 3
def time_with_current := 2
def time_against_current := 2.5

-- Distances in both directions should be equal
def distance_with_current := time_with_current * speed_with_current
def distance_against_current := time_against_current * speed_against_current

theorem average_speed_boat_in_still_water : distance_with_current = distance_against_current → x = 27 :=
by
  sorry

end average_speed_boat_in_still_water_l576_576577


namespace area_of_rectangle_l576_576151

--- Define the problem's conditions
def square_area : ℕ := 36
def rectangle_width := (square_side : ℕ) (h : square_area = square_side * square_side) : ℕ := square_side
def rectangle_length := (width : ℕ) : ℕ := 3 * width

--- State the theorem using the defined conditions
theorem area_of_rectangle (square_side : ℕ) 
  (h1 : square_area = square_side * square_side)
  (width := rectangle_width square_side h1)
  (length := rectangle_length width) :
  width * length = 108 := by
    sorry

end area_of_rectangle_l576_576151


namespace area_of_triangle_CM_and_N_proof_l576_576836

/-- A square ABCD with area 1 square inch --/
structure Square :=
  (A B C D : Point)
  (side : ℝ)
  (area_sq : side ^ 2 = 1)

/-- A right triangle CMN with legs MN, CN, and hypotenuse CM --/
structure RightTriangle :=
  (C M N : Point)
  (leg1 leg2 hypotenuse : ℝ)
  (pythagorean : hypotenuse ^ 2 = leg1 ^ 2 + leg2 ^ 2)

noncomputable def area_of_triangle_CM_and_N : Square → RightTriangle → ℝ
| ⟨_, _, _, _, _, _⟩ ⟨_, _, _, MN, CN, _⟩ := 
  (1 / 2) * MN * CN

theorem area_of_triangle_CM_and_N_proof :
  ∀ (sq : Square) (rt : RightTriangle), 
    sq.side ^ 2 = 1 → rt.leg1 = sq.side → rt.leg2 = 1 - rt.leg1 → area_of_triangle_CM_and_N sq rt = 1 / 8 :=
by 
  sorry

end area_of_triangle_CM_and_N_proof_l576_576836


namespace students_without_glasses_l576_576955

theorem students_without_glasses (total_students : ℕ) (perc_glasses : ℕ) (students_with_glasses students_without_glasses : ℕ) 
  (h1 : total_students = 325) (h2 : perc_glasses = 40) (h3 : students_with_glasses = perc_glasses * total_students / 100)
  (h4 : students_without_glasses = total_students - students_with_glasses) : students_without_glasses = 195 := 
by
  sorry

end students_without_glasses_l576_576955


namespace correct_equation_solves_time_l576_576633

noncomputable def solve_time_before_stop (t : ℝ) : Prop :=
  let total_trip_time := 4 -- total trip time in hours including stop
  let stop_time := 0.5 -- stop time in hours
  let total_distance := 180 -- total distance in km
  let speed_before_stop := 60 -- speed before stop in km/h
  let speed_after_stop := 80 -- speed after stop in km/h
  let time_after_stop := total_trip_time - stop_time - t -- time after the stop in hours
  speed_before_stop * t + speed_after_stop * time_after_stop = total_distance -- distance equation

-- The theorem states that the equation is valid for solving t
theorem correct_equation_solves_time :
  solve_time_before_stop t = (60 * t + 80 * (7/2 - t) = 180) :=
sorry -- proof not required

end correct_equation_solves_time_l576_576633


namespace proof_problem_l576_576252

noncomputable def a {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def b {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def c {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def d {α : Type*} [LinearOrderedField α] : α := sorry

theorem proof_problem (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
(hprod : a * b * c * d = 1) : 
a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end proof_problem_l576_576252


namespace find_counterfeit_coin_l576_576449

/-- 
There are 12 coins, one of which is counterfeit and lighter. 
There is a balance scale that is either regular (heavier side outweighs lighter side)
or magical (lighter side outweighs heavier side, but equality is shown correctly).
Prove that it is possible to find the counterfeit coin in 3 weighings. 
-/
theorem find_counterfeit_coin (coins : Fin₄ 12) (is_counterfeit : coins → Prop)
  (is_lighter : ∀ c, is_counterfeit c → ∀ other, ¬ is_counterfeit other → c < other)
  (scale : ∀ (coin1 coin2 : coins), {outcome : ℤ} →
          (outcome = 1 → coin1 > coin2) ∧ 
          (outcome = -1 → coin2 > coin1) ∧
          (outcome = 0 → coin1 = coin2)) :
  ∃ (weighings : (fin 3) → (coins × coins × coins × coins × coins × coins × coins × coins)) 
  (result : weighings → Fin₄ 12 → Prop), 
  ∀ (w : weighings), 
    (result w 0 = coins → result w 1 = coins → result w 2 = is_counterfeit). 
sorry

end find_counterfeit_coin_l576_576449


namespace finite_distinct_values_l576_576011

def g (x : ℝ) : ℝ := 5 * x - x ^ 2

def sequence (x₀ : ℝ) : ℕ → ℝ
| 0     := x₀
| (n+1) := g (sequence n)

theorem finite_distinct_values :
  {x₀ : ℝ | ∃ (N : ℕ), ∀ n m ≥ N, sequence x₀ n = sequence x₀ m}.finite.card = 2 :=
sorry

end finite_distinct_values_l576_576011


namespace total_surface_area_of_reg_triangular_pyramid_l576_576066

-- We use noncomputable here because the calculations involve square roots.
noncomputable def total_surface_area_of_pyramid 
  (a m n : ℝ) 
  (h_a_pos : 0 < a)
  (h_m_pos : 0 < m) 
  (h_n_pos : 0 < n) : ℝ :=
  (a^2 * real.sqrt 3 / 4) * (1 + real.sqrt (3 * (m + 2 * n) / m))

-- Statement of the problem
theorem total_surface_area_of_reg_triangular_pyramid 
  (a m n : ℝ) 
  (h_a_pos : 0 < a)
  (h_m_pos : 0 < m) 
  (h_n_pos : 0 < n) : 
  total_surface_area_of_pyramid a m n h_a_pos h_m_pos h_n_pos = 
    (a^2 * real.sqrt 3 / 4) * (1 + real.sqrt (3 * (m + 2 * n) / m)) :=
by
  sorry

end total_surface_area_of_reg_triangular_pyramid_l576_576066


namespace question_I_question_II_l576_576876

def f (x a : ℝ) : ℝ := |x - a| + 3 * x

theorem question_I (a : ℝ) (h_pos : a > 0) : 
  (f 1 x ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) := by sorry

theorem question_II (a : ℝ) (h_pos : a > 0) : 
  (- (a / 2) = -1) ↔ (a = 2) := by sorry

end question_I_question_II_l576_576876


namespace sum_geometric_sequence_l576_576704

theorem sum_geometric_sequence (a_n : ℕ → ℝ) (S_n T_n : ℕ → ℝ) (n : ℕ) :
  (∀ n : ℕ, S_n = ∑ i in finset.range n, a_n i) ∧
  a_n 4 = 2 * a_n 5 ∧
  S_n 6 = 63 / 64 →
  S_n n = 1 - (1/2)^n ∧
  (∀ n : ℕ, let b_n := (2^n * a_n n) / (n^2 + n) in T_n = ∑ i in finset.range n, b_n i) →
  T_n n = n / (n + 1) :=
by
  sorry

end sum_geometric_sequence_l576_576704


namespace grove_tree_selection_l576_576826

theorem grove_tree_selection (birches spruces pines aspens : ℕ) :
  birches + spruces + pines + aspens = 100 →
  (∀ s : set ℕ, s.card = 85 → (birches ∈ s ∧ spruces ∈ s ∧ pines ∈ s ∧ aspens ∈ s)) →
  ∀ t : set ℕ, t.card = 69 → (birches ∈ t ∧ spruces ∈ t) ∨ (birches ∈ t ∧ pines ∈ t) ∨ (birches ∈ t ∧ aspens ∈ t) ∨ (spruces ∈ t ∧ pines ∈ t) ∨ (spruces ∈ t ∧ aspens ∈ t) ∨ (pines ∈ t ∧ aspens ∈ t) :=
sorry

end grove_tree_selection_l576_576826


namespace ellipse_and_line_equation_l576_576244

-- Definition of the ellipse and its conditions
def ellipse_equation (a b : ℝ) : Prop := 
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

-- Given conditions
def eccentricity (a c : ℝ) : Prop :=
  c / a = Real.sqrt 2 / 2

def point_on_ellipse (x y a b : ℝ) : Prop :=
  (x, y) = (2, 1) → (x^2 / a^2) + (y^2 / b^2) = 1

def foot_of_perpendicular (Q : ℝ × ℝ) : Prop :=
  Q = (2, 0)

-- Prove the ellipse equation and the line equation
theorem ellipse_and_line_equation : 
  ∃ (a b : ℝ), 
  ellipse_equation a b ∧ 
  eccentricity a (Real.sqrt(a^2 - b^2)) ∧ 
  point_on_ellipse 2 1 a b ∧ 
  foot_of_perpendicular (2, 0) ∧
  ∃ (t : ℝ), t ≠ 0 ∧ 
  (3 * (a - 2) + (b - 2) = 0) → 
  (b = t * a) → 
  (b = (Real.sqrt 10 / 2) * (a - 2)) :=
sorry  

end ellipse_and_line_equation_l576_576244


namespace smallest_k_property_l576_576187

theorem smallest_k_property (n : ℕ) : 
  ∃ k : ℕ, (∀ (S : Finset ℕ), S ⊆ Finset.range (n + 1) → S.card = k → 
          ∃ a b ∈ S, a ≠ b ∧ (a ∣ b ∨ b ∣ a)) ∧ 
         k = nat.ceil (n / 2) + 1 :=
by
  sorry

end smallest_k_property_l576_576187


namespace arcsin_sin_eq_x_over_2_solutions_l576_576410

theorem arcsin_sin_eq_x_over_2_solutions :
  (∀ x : ℝ, -π ≤ x ∧ x ≤ π → arcsin (sin x) = x / 2 ↔ x = -2 * π / 3 ∨ x = 0 ∨ x = 2 * π / 3) := 
by 
  sorry

end arcsin_sin_eq_x_over_2_solutions_l576_576410


namespace three_types_in_69_trees_l576_576782

variable (birches spruces pines aspens : ℕ)
variable (total_trees : ℕ := 100)
variable (all_trees : list (string × ℕ))

-- We assert that there are 100 trees in total, and our list of trees represents this
axiom h_total : ∑ t in all_trees, t.2 = total_trees

-- Among any 85 trees, there must be at least one of each type
axiom h_85_trees_all_types : ∀ (s : list (string × ℕ)), s.card = 85 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0 ∧ a > 0 ∧ 
  b + s + p + a = 85 ∧ (("birches", b) ∈ s ∧ ("spruces", s) ∈ s ∧ ("pines", p) ∈ s ∧ ("aspens", a) ∈ s))

-- We need to prove that any subset of 69 or more trees contains at least three different types.
theorem three_types_in_69_trees :
  ∀ (s : list (string × ℕ)), s.card = 69 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0) ∨ (b > 0 ∧ s > 0 ∧ a > 0) ∨ (b > 0 ∧ p > 0 ∧ a > 0) ∨ (s > 0 ∧ p > 0 ∧ a > 0) :=
begin
  sorry
end

end three_types_in_69_trees_l576_576782


namespace num_pieces_cut_l576_576308

theorem num_pieces_cut (original_length_meters remaining_length_meters length_piece_centimeters : ℕ) 
    (h1 : original_length_meters = 51) 
    (h2 : remaining_length_meters = 36)
    (h3 : length_piece_centimeters = 15) :
    let length_cut_meters := original_length_meters - remaining_length_meters in
    let length_cut_centimeters := length_cut_meters * 100 in
    length_cut_centimeters / length_piece_centimeters = 100 :=
by {
  sorry
}

end num_pieces_cut_l576_576308


namespace area_of_rectangle_is_108_l576_576141

theorem area_of_rectangle_is_108 (s w l : ℕ) (h₁ : s * s = 36) (h₂ : w = s) (h₃ : l = 3 * w) : w * l = 108 :=
by
  -- This is a placeholder for a detailed proof.
  sorry

end area_of_rectangle_is_108_l576_576141


namespace find_positive_real_solution_l576_576671

theorem find_positive_real_solution (x : ℝ) (h : 0 < x) :
  (1 / 3) * (4 * x ^ 2 - 3) = (x ^ 2 - 75 * x - 15) * (x ^ 2 + 40 * x + 8) →
  x = (75 + Real.sqrt (75 ^ 2 + 4 * 13)) / 2 ∨ x = (-40 + Real.sqrt (40 ^ 2 - 4 * 7)) / 2 :=
by
  sorry

end find_positive_real_solution_l576_576671


namespace simplified_fraction_sum_l576_576982

theorem simplified_fraction_sum : 
  let f1 := 56 
  let f2 := 98 
  let f3 := 1 
  let f4 := 14 
  let simplified_result := (f1 / f2) - (f3 / f4) 
  let numerator := numer simplified_result -- Pseudo-code to get the numerator
  let denominator := denom simplified_result -- Pseudo-code to get the denominator
  (numerator + denominator = 3) :=
sorry

end simplified_fraction_sum_l576_576982


namespace sum_over_positive_reals_nonnegative_l576_576235

theorem sum_over_positive_reals_nonnegative (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (b + c - 2 * a) / (a^2 + b * c) + 
  (c + a - 2 * b) / (b^2 + c * a) + 
  (a + b - 2 * c) / (c^2 + a * b) ≥ 0 :=
sorry

end sum_over_positive_reals_nonnegative_l576_576235


namespace integral_evaluation_l576_576666

noncomputable def integral_value : ℝ :=
  ∫ x in 0..1, (2 + sqrt (1 - x^2))

theorem integral_evaluation :
  integral_value = 2 + π / 4 :=
by
  sorry

end integral_evaluation_l576_576666


namespace calculate_circumradius_of_triangle_ABC_l576_576525

noncomputable def circumradius (A B C : EuclideanSpace ℝ 3) : ℝ := sorry

theorem calculate_circumradius_of_triangle_ABC
  (A B C : EuclideanSpace ℝ 3)
  (r1 r2 : ℝ)
  (h1 : r1 + r2 = 13)
  (h2 : dist (center_of_sphere_through A r1) (center_of_sphere_through B r2) = sqrt 505)
  (h3 : dist C (tangent_point_sphere A r1) = 8)
  (h4 : dist C (tangent_point_sphere B r2) = 8) :
  circumradius A B C = 2 * sqrt 21 := 
sorry

end calculate_circumradius_of_triangle_ABC_l576_576525


namespace minimum_number_of_troublemakers_l576_576486

theorem minimum_number_of_troublemakers (n m : ℕ) (students : Fin n → Prop) 
  (truthtellers : Fin n → Prop) (liars : Fin n → Prop) (round_table : List (Fin n))
  (h_all_students : n = 29)
  (h_truth_or_liar : ∀ s, students s → (truthtellers s ∨ liars s))
  (h_truth_tellers_conditions : ∀ s, truthtellers s → 
    (count_lies_next_to s = 1 ∨ count_lies_next_to s = 2))
  (h_liars_conditions : ∀ s, liars s → 
    (count_lies_next_to s ≠ 1 ∧ count_lies_next_to s ≠ 2))
  (h_round_table_conditions : round_table.length = n 
    ∧ (∀ i, i < n → round_table.nth i ≠ none)):
  m = 10 :=
by
  sorry

end minimum_number_of_troublemakers_l576_576486


namespace crayons_per_box_l576_576375

-- Define the conditions
def crayons : ℕ := 80
def boxes : ℕ := 10

-- State the proof problem
theorem crayons_per_box : (crayons / boxes) = 8 := by
  sorry

end crayons_per_box_l576_576375


namespace triangle_ceva_isosceles_l576_576563

theorem triangle_ceva_isosceles 
  {A B C A1 B1 C1 : Point}
  (isosceles : Triangle ABC)
  (base : A ≠ B)
  (C1_on_AB : C1 ∈ Line.segment A B)
  (A1_on_BC : A1 ∈ Line.segment B C)
  (B1_on_CA : B1 ∈ Line.segment C A)
  (intersecting : Line.intersect (Line.through A A1) (Line.through B B1) = Line.intersect (Line.through A A1) (Line.through C C1)) :
  ∃ ratio: ℝ, 
  ratio = (AC1_over_C1B : ℝ),
  AC1_over_C1B = (sin_angle_ABB1 * sin_angle_CAA1) / (sin_angle_BAA1 * sin_angle_CBB1) := by sorry


end triangle_ceva_isosceles_l576_576563


namespace rational_if_vn_le_n_plus_8_l576_576842

-- Definitions based on the conditions
variable (a : ℝ)
def v_a (n : ℕ) : ℕ := sorry -- Number of different digit sequences of length n in the decimal expansion of a.

-- The theorem statement we need to prove
theorem rational_if_vn_le_n_plus_8 (n : ℕ) (h : v_a a n ≤ n + 8) : ∃ q : ℚ, a = q :=
sorry

end rational_if_vn_le_n_plus_8_l576_576842


namespace geometric_sequence_solution_l576_576706

theorem geometric_sequence_solution
  (a_n : ℕ → ℝ)
  (h1 : ∀ n, a_n > 0)
  (h2 : ∃ k > 0, ∀ n, a_n = a_1 * k^n) 
  (h3 : 2 * a_2 = a_3 + 6 + a_1)
  (h4 : a_4^2 = 9 * a_1 * a_5) :
  (∀ n, a_n = 3^n) ∧
  (∀ b_n T_n : ℕ → ℝ,
    (∀ n, b_n n = (log (a_n n) / log (sqrt 3) + 1) * a_n n) →
    (∀ n, T_n n = n * 3^(n + 1))) :=
by sorry

end geometric_sequence_solution_l576_576706


namespace total_number_of_water_filled_jars_l576_576564

theorem total_number_of_water_filled_jars : 
  ∃ (x : ℕ), 28 = x * (1/4 + 1/2 + 1) ∧ 3 * x = 48 :=
by
  sorry

end total_number_of_water_filled_jars_l576_576564


namespace pear_sales_ratio_l576_576610

theorem pear_sales_ratio : 
  ∀ (total_sold afternoon_sold morning_sold : ℕ), 
  total_sold = 420 ∧ afternoon_sold = 280 ∧ total_sold = afternoon_sold + morning_sold 
  → afternoon_sold / morning_sold = 2 :=
by 
  intros total_sold afternoon_sold morning_sold 
  intro h 
  have h_total : total_sold = 420 := h.1 
  have h_afternoon : afternoon_sold = 280 := h.2.1 
  have h_morning : total_sold = afternoon_sold + morning_sold := h.2.2
  sorry

end pear_sales_ratio_l576_576610


namespace range_of_g_l576_576193

-- Define the function g
def g (x : ℝ) : ℝ := x / (x^2 - 2*x + 2)

-- State the main theorem
theorem range_of_g : ∀ y, (∃ x, g x = y) ↔ (0 ≤ y ∧ y ≤ 1/2) := 
sorry

end range_of_g_l576_576193


namespace minimum_trees_with_at_least_three_types_l576_576807

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end minimum_trees_with_at_least_three_types_l576_576807


namespace instantaneous_rate_of_change_at_1_l576_576683

theorem instantaneous_rate_of_change_at_1 (x : ℝ) (h : x = 1) : 
  (deriv (λ x : ℝ, x * exp x)) 1 = 2 * exp 1 :=
by
  sorry

end instantaneous_rate_of_change_at_1_l576_576683


namespace point_inside_region_point_outside_region_l576_576994

-- Lean statement for Proof Problem 1 (Part a)
theorem point_inside_region (P : Type) (path : Type) (closed_path : Bool) (non_intersecting : Bool) (intersections : Nat) 
  (h_closed : closed_path = true) (h_non_intersecting : non_intersecting = true) (h_intersections : intersections = 3) : 
  P ∈ path :=
by
  -- This is stated to show the point is inside
  sorry

-- Lean statement for Proof Problem 2 (Part b)
theorem point_outside_region (Q : Type) (path : Type) (closed_path : Bool) (non_intersecting : Bool) (intersections : Nat) 
  (h_closed : closed_path = true) (h_non_intersecting : non_intersecting = true) (h_intersections : intersections = 6) : 
  Q ∉ path :=
by
  -- This is stated to show the point is outside
  sorry

end point_inside_region_point_outside_region_l576_576994


namespace rectangle_area_l576_576159

theorem rectangle_area (square_area : ℝ) (rectangle_length_ratio : ℝ) (square_area_eq : square_area = 36)
  (rectangle_length_ratio_eq : rectangle_length_ratio = 3) :
  ∃ (rectangle_area : ℝ), rectangle_area = 108 :=
by
  -- Extract the side length of the square from its area
  let side_length := real.sqrt square_area
  have side_length_eq : side_length = 6, from calc
    side_length = real.sqrt 36 : by rw [square_area_eq]
    ... = 6 : real.sqrt_eq 6 (by norm_num)
  -- Calculate the rectangle's width
  let width := side_length
  -- Calculate the rectangle's length
  let length := rectangle_length_ratio * width
  -- Calculate the area of the rectangle
  let area := width * length
  -- Prove the area is 108
  use area
  calc area
    = 6 * (3 * 6) : by rw [side_length_eq, rectangle_length_ratio_eq]
    ... = 108 : by norm_num

end rectangle_area_l576_576159


namespace tan_sum_angle_l576_576262

theorem tan_sum_angle 
  (x : ℝ)
  (θ : ℝ)
  (hP : terminal_side θ = ⟨-x, -6⟩)
  (hcos : cos θ = -5/13) : 
  tan (θ + π / 4) = -17/7 :=
sorry

end tan_sum_angle_l576_576262


namespace num_zeros_1_div_15_pow_15_l576_576684

theorem num_zeros_1_div_15_pow_15 : (number_of_zeros 1 (15^15)) = 17 :=
  sorry

end num_zeros_1_div_15_pow_15_l576_576684


namespace exterior_angle_BAC_l576_576613

open Real

-- Definitions based on the given conditions
def is_coplanar (s1 s2 : Set Point) : Prop := ∃ p : ℜ², s1 ⊆ {p} ∧ s2 ⊆ {p}
def common_side (s1 s2 : Set Segment) (side : Segment) : Prop := side ∈ s1 ∧ side ∈ s2

-- Given conditions
def A : Point := sorry
def D : Point := sorry
def S : Set Point := {A, sorry, sorry, D} -- Square vertices
def O : Set Point := sorry -- Octagon vertices
def AD : Segment := {A, D}

axiom coplanar_SO : is_coplanar S O
axiom common_AD : common_side S O AD

-- The theorem stating the desired proof problem
theorem exterior_angle_BAC : ∃ B C : Point, exterior_angle B A C = 135 :=
by
  sorry

end exterior_angle_BAC_l576_576613


namespace problem1_problem2_l576_576112

-- Problem 1
theorem problem1 : (-1 : ℤ) ^ 2024 + (1 / 3 : ℝ) ^ (-2 : ℤ) - (3.14 - Real.pi) ^ 0 = 9 := 
sorry

-- Problem 2
theorem problem2 (x : ℤ) (y : ℤ) (hx : x = 2) (hy : y = 3) : 
  x * (x + 2 * y) - (x + 1) ^ 2 + 2 * x = 11 :=
sorry

end problem1_problem2_l576_576112


namespace compute_ratio_d_e_l576_576055

open Polynomial

noncomputable def quartic_polynomial (a b c d e : ℚ) : Polynomial ℚ := 
  C a * X^4 + C b * X^3 + C c * X^2 + C d * X + C e

def roots_of_quartic (a b c d e: ℚ) : Prop :=
  (quartic_polynomial a b c d e).roots = {1, 2, 3, 5}

theorem compute_ratio_d_e (a b c d e : ℚ) 
    (h : roots_of_quartic a b c d e) :
    d / e = -61 / 30 :=
  sorry

end compute_ratio_d_e_l576_576055


namespace complement_eq_target_l576_576746

namespace ComplementProof

-- Define the universal set U
def U : Set ℕ := {2, 4, 6, 8, 10}

-- Define the set A
def A : Set ℕ := {2, 6, 8}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x ∈ U | x ∉ A}

-- Define the target set
def target_set : Set ℕ := {4, 10}

-- Prove that the complement of A with respect to U is equal to {4, 10}
theorem complement_eq_target :
  complement_U_A = target_set := by sorry

end ComplementProof

end complement_eq_target_l576_576746


namespace rectangle_area_from_square_l576_576150

theorem rectangle_area_from_square {s a : ℝ} (h1 : s^2 = 36) (h2 : a = 3 * s) :
    s * a = 108 :=
by
  -- The proof goes here
  sorry

end rectangle_area_from_square_l576_576150


namespace part1_part2_l576_576665

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x + 3|

theorem part1 (x : ℝ) : f x ≥ 6 ↔ x ≥ 1 ∨ x ≤ -2 := by
  sorry

theorem part2 (a b : ℝ) (m : ℝ) 
  (a_pos : a > 0) (b_pos : b > 0) 
  (fmin : m = 4) 
  (condition : 2 * a * b + a + 2 * b = m) : 
  a + 2 * b = 2 * Real.sqrt 5 - 2 := by
  sorry

end part1_part2_l576_576665


namespace find_k_l576_576281

-- Define the vectors a, b, and c
def vecA : ℝ × ℝ := (2, -1)
def vecB : ℝ × ℝ := (1, 1)
def vecC : ℝ × ℝ := (-5, 1)

-- Define the condition for two vectors being parallel
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 - u.2 * v.1 = 0

-- Define the target statement to be proven
theorem find_k : ∃ k : ℝ, parallel (vecA.1 + k * vecB.1, vecA.2 + k * vecB.2) vecC ∧ k = 1/2 := 
sorry

end find_k_l576_576281


namespace permutation_combination_example_l576_576113

-- Definition of permutation (A) and combination (C) in Lean
def permutation (n k : ℕ): ℕ := Nat.factorial n / Nat.factorial (n - k)
def combination (n k : ℕ): ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The Lean statement of the proof problem
theorem permutation_combination_example : 
3 * permutation 3 2 + 2 * combination 4 2 = 30 := 
by 
  sorry

end permutation_combination_example_l576_576113


namespace circumference_of_tangent_circle_l576_576772

noncomputable def tangent_circle_circumference (arc1_length: ℝ) (arc2_angle_deg: ℝ) (triangle : Type) :=
  let arc1 := arc1_length
  let arc2_angle_rad := arc2_angle_deg * Real.pi / 180
  let A B C : triangle
  let r₁ := arc1 / ((2 * Real.pi / 6) : ℝ)
  let r₂ := arc2_angle_rad / ((2 * Real.pi / 75) : ℝ)
  let r := 30 / Real.pi
  in 2 * Real.pi * r

theorem circumference_of_tangent_circle 
  (arc1_length: ℝ) (arc2_angle_deg: ℝ) (triangle : Type)
  (abc_equilateral: ∀ (A B C : triangle), (side_length A B = side_length B C) ∧ (side_length B C = side_length C A))
  (arc1: length_of_arc arc1_length)
  (arc2: angle_of_arc arc2_angle_deg = 75) :
  tangent_circle_circumference arc1_length arc2_angle_deg triangle = 60 :=
sorry

end circumference_of_tangent_circle_l576_576772


namespace complex_quadrant_problem_l576_576722

open Complex

-- Define the conditions
def z : ℂ := -1 + 2 * I

-- Define the required property to prove
def quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "Fourth quadrant"
  else "On an axis"

noncomputable def complex_problem : String :=
  quadrant ((5 * I) / z)

-- The expected answer to prove
theorem complex_quadrant_problem : complex_problem = "Third quadrant" :=
  sorry

end complex_quadrant_problem_l576_576722


namespace compute_g2_l576_576909

variables {α β : Type}

noncomputable def f : α → β := sorry
noncomputable def g : α → β := sorry

axiom fg_eq_x3 (x : α) (h : x ≥ 1) : f (g x) = x^3
axiom gf_eq_x4 (x : α) (h : x ≥ 1) : g (f x) = x^4
axiom g_16_eq_8 : g 16 = 8

theorem compute_g2 : (g 2)^4 = 8 := 
sorry

end compute_g2_l576_576909


namespace sum_binom_p_l576_576012

noncomputable def p (k : ℕ) (x : ℝ) : ℝ :=
  (Finset.range k).sum (λ i => x ^ i)

theorem sum_binom_p (n : ℕ) (x : ℝ) (h : n > 0) :
  ∑ k in Finset.range n.succ \ {0}, Nat.choose n k * p k x = 2^(n-1) * p n ((1 + x) / 2) := 
sorry

end sum_binom_p_l576_576012


namespace arithmetic_sequence_sum_10_l576_576243

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

noncomputable def a_n (a1 d : α) (n : ℕ) : α :=
a1 + (n - 1) • d

def sequence_sum (a1 d : α) (n : ℕ) : α :=
n • a1 + (n • (n - 1) / 2) • d

theorem arithmetic_sequence_sum_10 
  (a1 d : ℤ)
  (h1 : a_n a1 d 2 + a_n a1 d 4 = 4)
  (h2 : a_n a1 d 3 + a_n a1 d 5 = 10) :
  sequence_sum a1 d 10 = 95 :=
by
  sorry

end arithmetic_sequence_sum_10_l576_576243


namespace max_sum_l576_576901

def nat := ℕ  -- we'll use natural numbers

theorem max_sum :
  ∀ (n : ℕ → ℕ),
    (∑ i in finset.range 2003, (i + 1) * (n i)) = 2003 →
    (∑ i in finset.range 2003, i * (n (i + 1))) ≤ 2001 :=
by
  intro n h
  let N := ∑ i in finset.range 2003, n i
  have H : 2003 - N ≤ 2001 := sorry
  exact H

end max_sum_l576_576901


namespace Seojun_apples_decimal_l576_576036

theorem Seojun_apples_decimal :
  let total_apples := 100
  let seojun_apples := 11
  seojun_apples / total_apples = 0.11 :=
by
  let total_apples := 100
  let seojun_apples := 11
  sorry

end Seojun_apples_decimal_l576_576036


namespace a_seq_is_2_pow_T_seq_bound_l576_576745

open BigOperators -- for sequences sums operators like ∑

noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2^n

noncomputable def b_seq (n : ℕ) : ℝ :=
  1 / ((real.log 2 (a_seq n) * real.log 2 (a_seq (n + 2))))

noncomputable def T_seq (n : ℕ) : ℝ :=
  ∑ i in finset.range n, b_seq (i + 1)

theorem a_seq_is_2_pow (n : ℕ) (h : n > 0) :
  a_seq n = 2^n := 
sorry

theorem T_seq_bound (n : ℕ) (h : n > 0) :
  T_seq n < 3 / 4 := 
sorry

end a_seq_is_2_pow_T_seq_bound_l576_576745


namespace square_divided_into_equal_right_triangles_l576_576993

theorem square_divided_into_equal_right_triangles
  (a : ℕ)
  (h : ∃ (n : ℕ), a^2 = 6 * n) :
  ∃ (k : ℕ), even k :=
by
  sorry

end square_divided_into_equal_right_triangles_l576_576993


namespace elephants_at_WePreserveForFuture_l576_576436

theorem elephants_at_WePreserveForFuture (E : ℕ) 
  (h1 : ∀ gest : ℕ, gest = 3 * E)
  (h2 : ∀ total : ℕ, total = E + 3 * E) 
  (h3 : total = 280) : 
  E = 70 := 
by
  sorry

end elephants_at_WePreserveForFuture_l576_576436


namespace no_robbery_l576_576019

-- Define the suspects and their guilt status
variables (A B C : Prop)

-- Define the conditions
def cond1 : Prop := (A ∨ B ∨ C) → true
def cond2 : Prop := A → (B ∧ ¬C) ∨ (¬B ∧ C)
def cond3 : Prop := ¬B → ¬C
def cond4 : Prop := (B ∧ C) → A
def cond5 : Prop := ¬C → ¬B

-- The proof problem statement
theorem no_robbery
  (h1 : cond1)
  (h2 : cond2)
  (h3 : cond3)
  (h4 : cond4)
  (h5 : cond5) :
  ¬(A ∨ B ∨ C) :=
by {
  sorry
}

end no_robbery_l576_576019


namespace necessary_but_not_sufficient_l576_576232

variables {a b : ℝ}

theorem necessary_but_not_sufficient (h₁ : a > 0) (h₂ : b ∈ ℝ) :
  (a > |b| → a + b > 0) ∧ ¬(a + b > 0 → a > |b|) :=
begin
  -- part 1: prove necessary condition
  split,
  { intros h,
    apply lt_trans h₁,
    linarith, },
  -- part 2: prove not sufficient condition
  { intro h,
    linarith, },
end

end necessary_but_not_sufficient_l576_576232


namespace dance_group_selection_at_Hilltop_High_l576_576943

noncomputable def numberOfWays : ℕ :=
  nat.choose 4 3 * nat.choose 6 4

theorem dance_group_selection_at_Hilltop_High : numberOfWays = 60 := by
  sorry

end dance_group_selection_at_Hilltop_High_l576_576943


namespace continuous_integral_equal_l576_576860

variable {α : Type*} [LinearOrder α] [TopologicalSpace α] [TopologicalRing α] 

theorem continuous_integral_equal {f : α → α} {a b : α} (hcont : ContinuousOn f (Set.Icc a b)) (n : ℕ) (ξ : Fin n → α) 
  (hdiv : ∀ i, ξ i ∈ Set.Icc a b) : 
  ∫ x in Set.Icc a b, f x = (∀ i, (n:α)) :=
sorry

end continuous_integral_equal_l576_576860


namespace eventually_constant_l576_576867

theorem eventually_constant (a : ℕ → ℕ) (N : ℕ) (hN : N > 1)
  (h : ∀ (n : ℕ), n ≥ N → (∑ i in finset.range n, a i / a (i + 1)) + (a n / a 1) ∈ ℤ) :
  ∃ M : ℕ, ∀ m : ℕ, m ≥ M → a m = a (m+1) :=
sorry

end eventually_constant_l576_576867


namespace factorize_correct_l576_576199
noncomputable def factorize_expression (a b : ℝ) : ℝ :=
  (a - b)^4 + (a + b)^4 + (a + b)^2 * (a - b)^2

theorem factorize_correct (a b : ℝ) :
  factorize_expression a b = (3 * a^2 + b^2) * (a^2 + 3 * b^2) :=
by
  sorry

end factorize_correct_l576_576199


namespace length_of_RU_l576_576339

theorem length_of_RU
  (P Q R S T U : Type)
  (PQ QR PR : ℝ)
  (PQ_eq : PQ = 13)
  (QR_eq : QR = 26)
  (PR_eq : PR = 24)
  (angle_bisector_PQR : line P Q = same_line P Q U)
  (S_on_PR : S ∈ line P R)
  (T_on_circumcircle_PQR : T ≠ Q ∧ (T ∈ circumcircle(P, Q, R)))
  (U_on_circumcircle_PTS : U ≠ P ∧ (U ∈ circumcircle(P, T, S))) :
  length(P R) - length(P U) = 15 :=
sorry

end length_of_RU_l576_576339


namespace exists_valid_sequence_l576_576623

def valid_sequence (l : List ℕ) : Prop :=
  l.head? = some 101 ∧
  l.nodup ∧
  l.sorted (≤) ∧
  (∀ (n : ℕ) (h1 : n + 1 < l.length),
    let diff := l.nthLe (n + 1) h1 - l.nthLe n (Nat.le_of_lt (Nat.lt_add_one_iff.mpr (Nat.lt_of_lt_pred h1))) in
    diff = 2 ∨ diff = 5)

theorem exists_valid_sequence : ∃ l : List ℕ, valid_sequence l :=
  sorry

end exists_valid_sequence_l576_576623


namespace circular_permutation_divisible_41_l576_576595

theorem circular_permutation_divisible_41 (N : ℤ) (a b c d e : ℤ) (h : N = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e)
  (h41 : 41 ∣ N) :
  ∀ (k : ℕ), 41 ∣ (10^((k % 5) * (4 - (k / 5))) * a + 10^((k % 5) * 3 + (k / 5) * 4) * b + 10^((k % 5) * 2 + (k / 5) * 3) * c + 10^((k % 5) + (k / 5) * 2) * d + 10^(k / 5) * e) :=
sorry

end circular_permutation_divisible_41_l576_576595


namespace exists_example_sum_diff_one_no_example_sum_diff_four_point_five_l576_576109

def median_of_set (s : List ℝ) : ℝ :=
  if s.length % 2 = 1 then
    (s.sorted.nth (s.length / 2)).get_or_else 0
  else 
    ((s.sorted.nth (s.length / 2 - 1)).get_or_else 0 + (s.sorted.nth (s.length / 2)).get_or_else 0 ) / 2

def direct_sum (X Y : List ℝ) : List ℝ :=
  List.bind X (λ x, Y.map (λ y, x + y))

-- part(a)
theorem exists_example_sum_diff_one :
  ∃ (X Y : List ℝ), 
    median_of_set X = 0 ∧
    median_of_set Y = 0 ∧
    median_of_set (direct_sum X Y) = 1 := by 
  sorry

-- part(b)
theorem no_example_sum_diff_four_point_five :
  ∀ (Y : List ℝ), (1 ≤ List.minimum Y.get_or_else 0 ∧ List.maximum Y.get_or_else 0 ≤ 5) →
  ¬ ∃ (X Y : List ℝ),
    median_of_set (direct_sum X Y) - (median_of_set X + median_of_set Y) = 4.5 := by
  sorry

end exists_example_sum_diff_one_no_example_sum_diff_four_point_five_l576_576109


namespace Dean_handled_100_transactions_l576_576387

-- Definitions for the given conditions
def Mabel_transactions : ℕ := 90
def Anthony_transactions : ℕ := (9 * Mabel_transactions) / 10 + Mabel_transactions
def Cal_transactions : ℕ := (2 * Anthony_transactions) / 3
def Jade_transactions : ℕ := Cal_transactions + 14
def Dean_transactions : ℕ := (Jade_transactions * 25) / 100 + Jade_transactions

-- Define the theorem we need to prove
theorem Dean_handled_100_transactions : Dean_transactions = 100 :=
by
  -- Statement to skip the actual proof
  sorry

end Dean_handled_100_transactions_l576_576387


namespace cos_product_equals_one_over_128_l576_576224

theorem cos_product_equals_one_over_128 :
  (Real.cos (Real.pi / 15)) *
  (Real.cos (2 * Real.pi / 15)) *
  (Real.cos (3 * Real.pi / 15)) *
  (Real.cos (4 * Real.pi / 15)) *
  (Real.cos (5 * Real.pi / 15)) *
  (Real.cos (6 * Real.pi / 15)) *
  (Real.cos (7 * Real.pi / 15))
  = 1 / 128 := 
sorry

end cos_product_equals_one_over_128_l576_576224


namespace number_of_solutions_l576_576213

theorem number_of_solutions (x : ℕ) :
  (∃ n r, 0 ≤ r ∧ r ≤ 10 ∧ x = 11 * n + r) →
  (∃ n, 0 ≤ x ∧ x = 11 * n ∧ 1 + n = Int.floor (x / 10)) →
  (∃! x, ˙floor (x / 10) = Int.floor (x / 11) + 1) = 110 := by
  sorry

end number_of_solutions_l576_576213


namespace find_d_squared_plus_e_squared_l576_576894

theorem find_d_squared_plus_e_squared {a b c d e : ℕ} 
  (h1 : (a + 1) * (3 * b * c + 1) = d + 3 * e + 1)
  (h2 : (b + 1) * (3 * c * a + 1) = 3 * d + e + 13)
  (h3 : (c + 1) * (3 * a * b + 1) = 4 * (26 - d - e) - 1)
  : d ^ 2 + e ^ 2 = 146 := 
sorry

end find_d_squared_plus_e_squared_l576_576894


namespace min_troublemakers_l576_576451

theorem min_troublemakers (students : Finset ℕ) (table : List ℕ) (is_liar : ℕ → Bool) :
  students.card = 29 ∧
  (∀ s ∈ students, 
    (∃ i j : ℕ, 
      table.nth i = some s ∧ table.nth ((i + 1) % 29) = some j ∧
      (∃ k : ℕ, 
        (is_liar s = false → count (/=is_liar k liar) = 0 ∧
        is_liar s = true → count (/=is_liar k liar) = 2))) → 
  (∃! l ∈ students, count (/=is_liar l liar) = 1))) → 
  ∃ t : ℕ, t = 10 :=
sorry

end min_troublemakers_l576_576451


namespace value_of_expression_l576_576545

theorem value_of_expression : (15 + 5)^2 - (15^2 + 5^2) = 150 := by
  sorry

end value_of_expression_l576_576545


namespace math_problem_l576_576269

def f (x a b : ℝ) : ℝ := (a - 3 * b + 9) * Real.log (x + 3) + (1 / 2) * x ^ 2 + (b - 3) * x

theorem math_problem (a b : ℝ) (h_a_pos : a > 0) (h_a_ne_1 : a ≠ 1)
  (h_f_prime_1 : deriv (λ x, f x a b) 1 = 0)
  (h_f_prime_3_le : deriv (λ x, f x a b) 3 ≤ (1 / 6))
  (h_f_prime_ge : ∀ x, |x| ≥ 2 → deriv (λ x, f x a b) x ≥ 0) :
  (b = -a - 1) ∧ (∀ x, f x a b = 25 * Real.log (x + 3) + (1 / 2) * x ^ 2 - 7 * x) ∧
  ((∃ x, x ∈ Ioo (-3 : ℝ) 2 ∧ (f x a b = deriv (λ x, f x a b) x) ∧ (x = -2) ∧ (f -2 a b = 16))) :=
by
  sorry

end math_problem_l576_576269


namespace min_value_expression_l576_576981

theorem min_value_expression (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) : 
  (∃ θ, 0 < θ ∧ θ < π / 2) → 
  ∃ θ, 5 * Real.sqrt 5 = inf {y | ∃ θ, y = (8 / Real.cos θ) + (1 / Real.sin θ) ∧ 0 < θ ∧ θ < π / 2 } :=
sorry

end min_value_expression_l576_576981


namespace three_types_in_69_trees_l576_576786

variable (birches spruces pines aspens : ℕ)
variable (total_trees : ℕ := 100)
variable (all_trees : list (string × ℕ))

-- We assert that there are 100 trees in total, and our list of trees represents this
axiom h_total : ∑ t in all_trees, t.2 = total_trees

-- Among any 85 trees, there must be at least one of each type
axiom h_85_trees_all_types : ∀ (s : list (string × ℕ)), s.card = 85 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0 ∧ a > 0 ∧ 
  b + s + p + a = 85 ∧ (("birches", b) ∈ s ∧ ("spruces", s) ∈ s ∧ ("pines", p) ∈ s ∧ ("aspens", a) ∈ s))

-- We need to prove that any subset of 69 or more trees contains at least three different types.
theorem three_types_in_69_trees :
  ∀ (s : list (string × ℕ)), s.card = 69 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0) ∨ (b > 0 ∧ s > 0 ∧ a > 0) ∨ (b > 0 ∧ p > 0 ∧ a > 0) ∨ (s > 0 ∧ p > 0 ∧ a > 0) :=
begin
  sorry
end

end three_types_in_69_trees_l576_576786


namespace kocourkov_coins_l576_576321

theorem kocourkov_coins :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 
  (∀ n > 53, ∃ x y : ℕ, n = x * a + y * b) ∧ 
  ¬ (∃ x y : ℕ, 53 = x * a + y * b) ∧
  ((a = 2 ∧ b = 55) ∨ (a = 3 ∧ b = 28)) :=
by {
  sorry
}

end kocourkov_coins_l576_576321


namespace convex_polyhedron_faces_same_edges_l576_576400

theorem convex_polyhedron_faces_same_edges (n : ℕ) (f : Fin n → ℕ) 
  (n_ge_4 : 4 ≤ n)
  (h : ∀ i : Fin n, 3 ≤ f i ∧ f i ≤ n - 1) : 
  ∃ (i j : Fin n), i ≠ j ∧ f i = f j := 
by
  sorry

end convex_polyhedron_faces_same_edges_l576_576400


namespace vasya_solved_13_problems_first_9_days_at_least_one_each_day_cloudy_rule_sunny_rule_day_10_cloudy_max_problems_day_15_l576_576969

-- Define the number of problems solved on each day
def problems_solved : ℕ → ℕ := sorry

-- Hypothesis: Over the first 9 days, Vasya solved 13 problems 
def sum_of_first_9_days : ℕ := (0 to 8).sum problems_solved
theorem vasya_solved_13_problems_first_9_days : sum_of_first_9_days = 13 := sorry

-- Hypothesis: At least one problem each day
theorem at_least_one_each_day (i : ℕ) : 1 ≤ problems_solved i := sorry

-- Hypothesis: If cloudy on day i+1, solved one more problem than the previous day
def is_cloudy : ℕ → Prop := sorry

theorem cloudy_rule (i : ℕ) : is_cloudy (i+1) → problems_solved (i+1) = problems_solved i + 1 := sorry

-- Hypothesis: If sunny on day i+1, solved one less problem than the previous day
def is_sunny : ℕ → Prop := sorry

theorem sunny_rule (i : ℕ) : is_sunny (i+1) → problems_solved (i+1) = problems_solved i - 1 := sorry

-- Conclusion for Part (a): The 10th day was cloudy, Vasya solved 2 problems
theorem day_10_cloudy : is_cloudy 10 ∧ problems_solved 10 = 2 := sorry

-- Conclusion for Part (b): The maximum number of problems on Day 15 is 7
theorem max_problems_day_15 : problems_solved 15 ≤ 7 := sorry

end vasya_solved_13_problems_first_9_days_at_least_one_each_day_cloudy_rule_sunny_rule_day_10_cloudy_max_problems_day_15_l576_576969


namespace speed_against_current_l576_576562

variables (v_speed_with_current v_speed_current : ℝ)

-- Conditions: 
def speed_with_current : Prop := v_speed_with_current = 22
def current_speed : Prop := v_speed_current = 5

-- Define the man's speed in still water
def still_water_speed : ℝ := v_speed_with_current - v_speed_current

theorem speed_against_current (h1 : speed_with_current) (h2 : current_speed) :
  still_water_speed v_speed_with_current v_speed_current - v_speed_current = 12 :=
by sorry

end speed_against_current_l576_576562


namespace exists_x0_in_interval_l576_576859

noncomputable def f (a b x : ℝ) : ℝ := a * x + b + 9 / x

theorem exists_x0_in_interval (a b : ℝ) : ∃ x0 ∈ set.Icc 1 9, |f a b x0| ≥ 2 :=
by
  sorry

end exists_x0_in_interval_l576_576859


namespace complex_magnitude_l576_576236

open Complex

theorem complex_magnitude (z : ℂ) (h : z * Complex.I + 3 = 2 * Complex.I) : Complex.abs (conj z - Complex.I) = 2 * Real.sqrt 5 := by
  sorry

end complex_magnitude_l576_576236


namespace arithmetic_progression_complete_iff_divides_l576_576133

-- Definitions from the conditions
def complete_sequence (s : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, s n ≠ 0) ∧ (∀ m : ℤ, m ≠ 0 → ∃ n : ℕ, s n = m)

-- Arithmetic progression definition
def arithmetic_progression (a r : ℤ) (n : ℕ) : ℤ :=
  a + n * r

-- Lean theorem statement
theorem arithmetic_progression_complete_iff_divides (a r : ℤ) :
  (complete_sequence (arithmetic_progression a r)) ↔ (r ∣ a) := by
  sorry

end arithmetic_progression_complete_iff_divides_l576_576133


namespace cylindrical_container_price_l576_576593

noncomputable def volume (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def price (v : ℝ) (p_per_v : ℝ) : ℝ := (v * p_per_v)

theorem cylindrical_container_price :
  let V1 := volume 2.5 10
  let V2 := volume 5 15
  let P1 := 2.50
  let price_per_volume := P1 / V1
  price V2 price_per_volume = 15 :=
by
  sorry

end cylindrical_container_price_l576_576593


namespace minimum_liars_l576_576475

noncomputable def minimum_troublemakers (n : ℕ) := 10

theorem minimum_liars (n : ℕ) (students : ℕ → bool) 
  (seated_at_round_table : ∀ i, students (i % 29) = tt ∨ students (i % 29) = ff)
  (one_liar_next_to : ∀ i, students i = tt → (students ((i + 1) % 29) = ff ∨ students ((i + 28) % 29) = ff))
  (two_liars_next_to : ∀ i, students i = ff → ((students ((i + 1) % 29) = ff) ∧ students ((i + 28) % 29) = ff)) :
  ∃ m, m = minimum_troublemakers 29 ∧ (m ≤ n) :=
by
  let liars_count := 10
  have : liars_count = 10 := rfl
  use liars_count
  exact ⟨rfl, sorry⟩

end minimum_liars_l576_576475


namespace log_b_n_l576_576708

variable {α : Type*} [OrderedSemiring α] [Real α]

-- Definitions as given in conditions
def S (n : ℕ) : ℕ := n^2 - n

def a (n : ℕ) : ℕ := S n - S (n - 1)

def b : ℕ → ℝ
| 0     := 0  -- Not used
| 1     := 0  -- Not used
| 2     := a 3
| (n+3) := (4 * b n ^ 2) / b (n-1)

-- Required theorem
theorem log_b_n (n : ℕ) (h : b 2 = a 3) (h_rec : ∀ (k : ℕ), k ≥ 2 → b (k + 3) * b (k - 1) = 4 * b k ^ 2) :
  log 2 (b n) = n := by
  sorry

end log_b_n_l576_576708


namespace constant_in_denominator_l576_576948

theorem constant_in_denominator (x y z : ℝ) (some_constant : ℝ)
  (h : ((x - y)^3 + (y - z)^3 + (z - x)^3) / (some_constant * (x - y) * (y - z) * (z - x)) = 0.2) :
  some_constant = 15 := 
sorry

end constant_in_denominator_l576_576948


namespace cuberoot_sum_l576_576869

theorem cuberoot_sum (x : ℝ) (h1 : x > 0) (h2 : x + 1 / x = 50) :
  real.cbrt x + real.cbrt (1 / x) = real.cbrt 53 :=
by
  -- we omit the proof here
  sorry

end cuberoot_sum_l576_576869


namespace intersection_A_complementB_l576_576288

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1, 3, 5, 7}
def complementB := U \ B

theorem intersection_A_complementB :
  A ∩ complementB = {2, 4, 6} := 
by
  sorry

end intersection_A_complementB_l576_576288


namespace max_prime_factor_arithmetic_sequence_l576_576309

theorem max_prime_factor_arithmetic_sequence (a b c : ℕ) (h_digit_range_a : 1 ≤ a ∧ a ≤ 9) (h_digit_range_b : 0 ≤ b ∧ b ≤ 9) (h_digit_range_c : 0 ≤ c ∧ c ≤ 9)
    (h_sequence : ∃ d, b = a + d ∧ c = a + 2 * d) :
    ∃ (d : ℕ), let n := 100 * a + 10 * b + c in prime 317 ∧ 317 ∣ n :=
begin
  sorry
end

end max_prime_factor_arithmetic_sequence_l576_576309


namespace length_of_AE_l576_576898

/-
 Segment AB is both a diameter of a circle of radius √3 and a side of an equilateral triangle ABC.
 The circle intersects AC and BC at points D and E respectively.
 Calculate the length of AE.
-/

/-- Given a circle with radius √3 and diameter AB which is also the side of an equilateral triangle ABC,
    and the circle intersects AC and BC at points D and E respectively,
    prove that the length of AE is 3. -/
theorem length_of_AE (A B C D E : Type*) [euclidean_space ℝ A B C D E]
  (r : ℝ) (h_circle_radius : r = real.sqrt 3)
  (h_diameter_AB : dist A B = 2 * real.sqrt 3)
  (h_equilateral_triangle : is_equilateral A B C)
  (h_intersect_AC : circle_intersects A C D)
  (h_intersect_BC : circle_intersects B C E)
  : dist A E = 3 :=
sorry

end length_of_AE_l576_576898


namespace tim_income_percentage_less_l576_576016

theorem tim_income_percentage_less {M T J : ℝ} 
  (h1 : M = 1.6 * T) 
  (h2 : M = 1.28 * J) : T = 0.8 * J :=
by
  have h : 1.6 * T = 1.28 * J := by rw [h1, h2]
  calc
    T = 1.28 * J / 1.6 : by rw [← h, div_eq_mul_inv]
    ... = 0.8 * J      : by norm_num

end tim_income_percentage_less_l576_576016


namespace function_machine_output_l576_576841

def function_machine (x : ℕ) : ℕ :=
  let y_1 := x * 3 in
  let y_2 := if y_1 > 30 then y_1 - 7 else y_1 + 10 in
  y_2 * 2

theorem function_machine_output :
  function_machine 15 = 76 :=
by
  sorry

end function_machine_output_l576_576841


namespace minimum_number_of_troublemakers_l576_576489

theorem minimum_number_of_troublemakers (n m : ℕ) (students : Fin n → Prop) 
  (truthtellers : Fin n → Prop) (liars : Fin n → Prop) (round_table : List (Fin n))
  (h_all_students : n = 29)
  (h_truth_or_liar : ∀ s, students s → (truthtellers s ∨ liars s))
  (h_truth_tellers_conditions : ∀ s, truthtellers s → 
    (count_lies_next_to s = 1 ∨ count_lies_next_to s = 2))
  (h_liars_conditions : ∀ s, liars s → 
    (count_lies_next_to s ≠ 1 ∧ count_lies_next_to s ≠ 2))
  (h_round_table_conditions : round_table.length = n 
    ∧ (∀ i, i < n → round_table.nth i ≠ none)):
  m = 10 :=
by
  sorry

end minimum_number_of_troublemakers_l576_576489


namespace methane_cos_theta_l576_576723

noncomputable def methane_molecule := 
  -- Define the structure of a methane molecule (central carbon atom and 4 hydrogen atoms in a regular tetrahedron).
  let central_atom := (0, 0, 0)
  let h1 := (1, 1, 1)
  let h2 := (1, -1, -1)
  let h3 := (-1, 1, -1)
  let h4 := (-1, -1, 1)
  [central_atom, h1, h2, h3, h4]

theorem methane_cos_theta :
  let θ := angle (coordinate (methane_molecule 1)) (coordinate (methane_molecule 2))
  cos θ = -1 / 3 :=
by
  sorry

end methane_cos_theta_l576_576723


namespace triangle_identity_converse_not_necessarily_hold_l576_576341

-- Lean statement for the first part: proving the given condition
theorem triangle_identity (a b c : ℝ) (A B : ℝ)
  (cond1 : ∠A = 3 * ∠B)
  (cond2 : side1 = BC) (cond3 : side2 = CA) (cond4 : side3 = AB) :
  (a^2 - b^2) * (a - b) = b * c^2 :=
sorry

-- Lean statement for the second part: proving the converse does not necessarily hold
theorem converse_not_necessarily_hold (a b c : ℝ) (A B : ℝ) :
  ((a^2 - b^2) * (a - b) = b * c^2) → (∠A = 3 * ∠B) :=
sorry

end triangle_identity_converse_not_necessarily_hold_l576_576341


namespace product_of_solutions_of_x_squared_eq_49_l576_576215

theorem product_of_solutions_of_x_squared_eq_49 : 
  (∀ x, x^2 = 49 → x = 7 ∨ x = -7) → (7 * (-7) = -49) :=
by
  intros
  sorry

end product_of_solutions_of_x_squared_eq_49_l576_576215


namespace slope_problem_l576_576762

theorem slope_problem (m : ℝ) (h₀ : m > 0) (h₁ : (3 - m) = m * (1 - m)) : m = Real.sqrt 3 := by
  sorry

end slope_problem_l576_576762


namespace sum_of_interchanged_primes_l576_576093

open Nat

def digits_interchanged (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  10 * units + tens

noncomputable def valid_two_digit_primes := 
  [23, 29, 31, 37, 53, 59, 71, 73, 79].filter (λ p, p.prime ∧ (digits_interchanged p).prime)

noncomputable def interchanged_primes_sum : ℕ :=
  (valid_two_digit_primes.filter (λ p, (digits_interchanged p).prime)).sum

theorem sum_of_interchanged_primes : 
  interchanged_primes_sum = 418 := by 
  sorry

end sum_of_interchanged_primes_l576_576093


namespace age_of_new_person_l576_576045

theorem age_of_new_person (T : ℝ) (A : ℝ) (h : T / 20 - 4 = (T - 60 + A) / 20) : A = 40 :=
sorry

end age_of_new_person_l576_576045


namespace max_adjacent_distinct_pairs_l576_576950

theorem max_adjacent_distinct_pairs (n : ℕ) (h : n = 100) : 
  ∃ (a : ℕ), a = 50 := 
by 
  -- Here we use the provided constraints and game scenario to state the theorem formally.
  sorry

end max_adjacent_distinct_pairs_l576_576950


namespace negation_of_existence_l576_576938

theorem negation_of_existence :
  ¬ (∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ ∀ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
sorry

end negation_of_existence_l576_576938


namespace pieces_per_serving_l576_576073

-- Definitions based on conditions
def jaredPopcorn : Nat := 90
def friendPopcorn : Nat := 60
def numberOfFriends : Nat := 3
def totalServings : Nat := 9

-- Statement to verify
theorem pieces_per_serving : 
  ((jaredPopcorn + numberOfFriends * friendPopcorn) / totalServings) = 30 :=
by
  sorry

end pieces_per_serving_l576_576073


namespace parallelogram_area_l576_576192

/-- Define the vertices of the parallelogram -/
def vertex1 : ℝ × ℝ := (0, 0)
def vertex2 : ℝ × ℝ := (3, 0)
def vertex3 : ℝ × ℝ := (2, 5)
def vertex4 : ℝ × ℝ := (5, 5)

/-- The base of the parallelogram is given by the horizontal distance between vertex1 and vertex2 -/
def base : ℝ := dist vertex1 vertex2

/-- The height of the parallelogram is given by the vertical distance between the line containing vertex3 and vertex4 and the x-axis -/
def height : ℝ := vertex3.2

/-- The area of the parallelogram -/
def area_parallelogram (base height : ℝ) : ℝ := base * height

theorem parallelogram_area : area_parallelogram base height = 15 := by
  sorry

end parallelogram_area_l576_576192


namespace matrix_determinant_zero_l576_576667

theorem matrix_determinant_zero (a b c : ℝ) : 
  Matrix.det (Matrix.of ![![1, a + b, b + c], ![1, a + 2 * b, b + 2 * c], ![1, a + 3 * b, b + 3 * c]]) = 0 := 
by
  sorry

end matrix_determinant_zero_l576_576667


namespace distance_between_parallel_lines_l576_576915

theorem distance_between_parallel_lines :
  let a := 3
  let b := 4
  let c₁ := -3
  let c₂ := 7
  let D := (abs (c₁ - c₂)) / (Real.sqrt (a ^ 2 + b ^ 2))
  D = 2 :=
by
  dsimp [abs, (/)]; dsimp [funext, pow_succ, Real.sqrt]; sorry

end distance_between_parallel_lines_l576_576915


namespace athlete_runs_entire_track_in_44_seconds_l576_576450

noncomputable def time_to_complete_track (flags : ℕ) (time_to_4th_flag : ℕ) : ℕ :=
  let distances_between_flags := flags - 1
  let distances_to_4th_flag := 4 - 1
  let time_per_distance := time_to_4th_flag / distances_to_4th_flag
  distances_between_flags * time_per_distance

theorem athlete_runs_entire_track_in_44_seconds :
  time_to_complete_track 12 12 = 44 :=
by
  sorry

end athlete_runs_entire_track_in_44_seconds_l576_576450


namespace fill_in_the_blanks_correctly_l576_576899

theorem fill_in_the_blanks_correctly :
  ∀ (subject_phenomena : String) (verb1 : String) (verb2 : String),
  subject_phenomena = "such phenomena" →
  (verb1 = "are" ∨ verb1 = "is") →
  (verb2 = "seem" ∨ verb2 = "seems") →
  (verb1 = "are" ∧ verb2 = "seem") :=
begin
  intros subject_phenomena verb1 verb2 h_subj h_verb1 h_verb2,
  split;
  { sorry },
end

end fill_in_the_blanks_correctly_l576_576899


namespace colin_avg_time_l576_576640

def totalTime (a b c d : ℕ) : ℕ := a + b + c + d

def averageTime (total_time miles : ℕ) : ℕ := total_time / miles

theorem colin_avg_time :
  let first_mile := 6
  let second_mile := 5
  let third_mile := 5
  let fourth_mile := 4
  let total_time := totalTime first_mile second_mile third_mile fourth_mile
  4 > 0 -> averageTime total_time 4 = 5 :=
by
  intros
  -- proof goes here
  sorry

end colin_avg_time_l576_576640


namespace seq_b_is_5th_order_repeatable_seq_c_is_not_5th_order_repeatable_min_m_for_3rd_order_repeatable_find_last_term_value_l576_576239

-- Definition of a kth-order repeatable sequence
def is_kth_order_repeatable (k : Nat) (seq : List (ℕ)) : Prop :=
  ∃ i j, 2 ≤ k ∧ k ≤ seq.length - 1 ∧ i ≠ j ∧ List.take k (List.drop i seq) = List.take k (List.drop j seq)

-- Problem a: Verify sequences for 5th-order repeatable
def seq_b : List ℕ := [0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
def seq_c : List ℕ := [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]

theorem seq_b_is_5th_order_repeatable :
  is_kth_order_repeatable 5 seq_b := sorry

theorem seq_c_is_not_5th_order_repeatable :
  ¬ is_kth_order_repeatable 5 seq_c := sorry

-- Problem b: Minimum value of m for any sequence to be 3rd-order repeatable
theorem min_m_for_3rd_order_repeatable :
  ∀ seq, seq.length = 11 → is_kth_order_repeatable 3 seq := sorry

-- Problem c: Last term modification to make 5th-order repeatable sequence
def a : List ℕ → ℕ := λ seq => seq.ilast -- Extract last element from sequence

theorem find_last_term_value (a : List ℕ) (h : ∀ x ∈ a, x = 0 ∨ x = 1) :
  ¬ is_kth_order_repeatable 5 a →
    is_kth_order_repeatable 5 (a ++ [0]) ∨ is_kth_order_repeatable 5 (a ++ [1]) →
    (a.head! ≠ 1) → (a.ilast ≠ 1) := sorry

end seq_b_is_5th_order_repeatable_seq_c_is_not_5th_order_repeatable_min_m_for_3rd_order_repeatable_find_last_term_value_l576_576239


namespace distinct_solutions_diff_l576_576360

theorem distinct_solutions_diff (r s : ℝ) 
  (h1 : r ≠ s) 
  (h2 : (5*r - 15)/(r^2 + 3*r - 18) = r + 3) 
  (h3 : (5*s - 15)/(s^2 + 3*s - 18) = s + 3) 
  (h4 : r > s) : 
  r - s = 13 :=
sorry

end distinct_solutions_diff_l576_576360


namespace lines_are_perpendicular_l576_576936

-- Define the two given lines
def line1 (c : ℝ) : set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 + p.2 + c = 0}
def line2 : set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - 2 * p.2 + 1 = 0}

-- Define the slopes
def slope1 (c : ℝ) : ℝ := -2
def slope2 : ℝ := 1 / 2

-- Define a predicate for lines being perpendicular
def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem asserting the lines are perpendicular
theorem lines_are_perpendicular (c : ℝ) : 
  are_perpendicular (slope1 c) slope2 :=
by
  unfold slope1 slope2 are_perpendicular
  sorry  -- proof to be filled in

end lines_are_perpendicular_l576_576936


namespace factorization_problem_l576_576907

theorem factorization_problem 
    (a m n b : ℝ)
    (h1 : (x + 2) * (x + 4) = x^2 + a * x + m)
    (h2 : (x + 1) * (x + 9) = x^2 + n * x + b) :
    (x + 3) * (x + 3) = x^2 + a * x + b :=
by
  sorry

end factorization_problem_l576_576907


namespace quadratic_expression_l576_576707

noncomputable def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_expression (a b c : ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h_leading_coeff : a < 0)
  (h_inequality : ∀ x, f x > -2*x ↔ (1 < x) ∧ (x < 3))
  (h_equal_roots : ∃ x, f x + 6*a = 0 ∧ ∀ y, f y + 6 * a = 0 → y = x)
  : f x = -x^2 - x - 3/5 :=
begin
  sorry
end

end quadratic_expression_l576_576707


namespace min_liars_needed_l576_576474

namespace Problem

/-
  We introduce our problem using Lean definitions and statements.
-/

def students_sitting_at_table : ℕ := 29

structure Student :=
(is_liar : Bool)

def student_truth_statement (s: ℕ) (next1: Student) (next2: Student) : Bool :=
  if s then (next1.is_liar && !next2.is_liar) || (!next1.is_liar && next2.is_liar)
  else next1.is_liar && next2.is_liar

def minimum_liars_class (sitting_students: ℕ) : ℕ :=
  sitting_students / 3 + (if sitting_students % 3 = 0 then 0 else 1)

theorem min_liars_needed (students : List Student) (H_total : students.length = students_sitting_at_table)
  (H_truth_teller_claim : ∀ (i : ℕ), i < students.length → student_truth_statement i (students.get ⟨(i+1) % students.length, sorry⟩) (students.get ⟨(i+2) % students.length, sorry⟩)) :
  (students.count (λ s => s.is_liar)) ≥ minimum_liars_class students_sitting_at_table :=
sorry

end Problem

end min_liars_needed_l576_576474


namespace time_difference_l576_576100

/-- Define the times it takes for Danny and Steve to reach each other's houses -/
def T_d := 25
def T_s := 2 * T_d

/-- Define the times it takes for Danny and Steve to reach the halfway point -/
def T_dH := T_d / 2
def T_sH := T_s / 2

/-- The time difference theorem to be proved -/
theorem time_difference : T_sH - T_dH = 12.5 :=
by
  sorry

end time_difference_l576_576100


namespace min_troublemakers_l576_576512

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l576_576512


namespace part1_part2_l576_576246

-- Define the complex numbers and conditions
variable {a b : ℝ}
noncomputable def z1 := complex.mk a (-5)
noncomputable def z2 := complex.mk (-2) b

-- Define the problem
theorem part1 (h : z1 = complex.conj z2) : -2 + 5 * complex.I ∈ {z : complex | z.re < 0 ∧ 0 < z.im} :=
by {
  have a_eq_neg2 : a = -2, from sorry,
  have b_eq_5 : b = 5, from sorry,
  rw [a_eq_neg2, b_eq_5],
  exact ⟨by norm_num, by norm_num⟩,
  sorry
}

theorem part2 (h1 : z1 = complex.conj z2) (h2 : z2 = -2 + 5 * complex.I) :
  ∀ z : complex, z^2 + -2*z + 5 = 0 → (z = 1 + 2 * complex.I ∨ z = 1 - 2 * complex.I) :=
by {
  intro z,
  -- Use quadratic formula for the solution
  have discriminant : (-2) ^ 2 - 4 * 1 * 5 = -16, by norm_num,
  rw discriminant at h2,
  unfold complex.number,
  sorry
}

end part1_part2_l576_576246


namespace rectangle_area_increase_l576_576925

theorem rectangle_area_increase :
  let l := 33.333333333333336
  let b := l / 2
  let A_original := l * b
  let l_new := l - 5
  let b_new := b + 4
  let A_new := l_new * b_new
  A_new - A_original = 30 :=
by
  sorry

end rectangle_area_increase_l576_576925


namespace Minerva_stamps_l576_576015

variable (Lizette Minerva : ℕ)

-- Condition 1: Lizette has 813 stamps
def Lizette_stamps : Lizette = 813 := by rfl

-- Condition 2: Lizette has 125 more stamps than Minerva
def Lizette_more_than_Minerva : Lizette = Minerva + 125 :=
  by rfl

-- Theorem: Minerva has 688 stamps
theorem Minerva_stamps (h1: Lizette = 813) (h2: Lizette = Minerva + 125) : Minerva = 688 :=
  by
    sorry

end Minerva_stamps_l576_576015


namespace circle_radius_tangent_to_ellipse_l576_576620

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | (p.1^2) / 36 + (p.2^2) / 9 = 1 }

noncomputable def focus : ℝ × ℝ :=
  (-3 * Real.sqrt 3, 0)

noncomputable def circle (r : ℝ) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | (p.1 + 3 * Real.sqrt 3)^2 + p.2^2 = r^2 }

theorem circle_radius_tangent_to_ellipse :
  ∃ r : ℝ, r = 3 ∧ circle r ⊆ ellipse :=
  sorry

end circle_radius_tangent_to_ellipse_l576_576620


namespace passes_through_fixed_point_l576_576922

theorem passes_through_fixed_point : ∃ (x y : ℝ), (y = 2^(x-4) + 3) ∧ (x = 4) ∧ (y = 4) := by
  use 4, 4
  split
  · rfl
  split
  · rfl
  sorry

end passes_through_fixed_point_l576_576922


namespace yunas_math_score_l576_576989

theorem yunas_math_score (K E M : ℕ) 
  (h1 : (K + E) / 2 = 92) 
  (h2 : (K + E + M) / 3 = 94) : 
  M = 98 :=
sorry

end yunas_math_score_l576_576989


namespace tickets_per_candy_l576_576986

theorem tickets_per_candy (tickets_whack_a_mole : ℕ) (tickets_skee_ball : ℕ) (candies_bought : ℕ)
    (h1 : tickets_whack_a_mole = 26) (h2 : tickets_skee_ball = 19) (h3 : candies_bought = 5) :
    (tickets_whack_a_mole + tickets_skee_ball) / candies_bought = 9 := by
  sorry

end tickets_per_candy_l576_576986


namespace average_speed_boat_still_water_l576_576580

noncomputable def boat_speed_in_still_water
  (t_AB_with_current : ℕ)
  (t_BA_against_current : ℕ)
  (current_speed : ℕ)
  (h : ∀ (x : ℝ), 2 * (x + current_speed) = 2.5 * (x - current_speed)) : ℝ :=
  27

theorem average_speed_boat_still_water : boat_speed_in_still_water 2 2.5 3 sorry = 27 := by
  sorry

end average_speed_boat_still_water_l576_576580


namespace S_2016_eq_1008_l576_576444

def sequence (a : ℕ → ℚ) : ℕ → ℚ
| 0 => 3/5
| (n+1) => if 0 ≤ a n ∧ a n ≤ 1/2 then 2 * a n else 2 * a n - 1

def sum_first_n {a : ℕ → ℚ} (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) a

theorem S_2016_eq_1008 : sum_first_n 2016 (sequence (λ n, n)) = 1008 := 
sorry

end S_2016_eq_1008_l576_576444


namespace collinear_A_Y_M_l576_576872

theorem collinear_A_Y_M
  (ABC : Triangle)
  (M : Point)
  (is_midpoint : midpoint BC M)
  (A : Point)
  (D : Point)
  (tangent_to_circumcircle : tangent (circumcircle ABC) A D BC)
  (Γ : Circle)
  (centre_D : Γ.center = D)
  (radius_AD : Γ.radius = dist A D)
  (E : Point)
  (E_on_circumcircle : on_circle (circumcircle ABC) E)
  (E_on_Γ : on_circle Γ E)
  (X : Point)
  (BE_line : line B E)
  (X_on_BE : on_line BE_line X)
  (X_on_Γ : on_circle Γ X)
  (Y : Point)
  (CX_line : line C X)
  (Y_on_CX : on_line CX_line Y)
  (Y_on_Γ : on_circle Γ Y) :
  collinear [A, Y, M] := sorry

end collinear_A_Y_M_l576_576872


namespace area_of_rectangle_is_108_l576_576142

theorem area_of_rectangle_is_108 (s w l : ℕ) (h₁ : s * s = 36) (h₂ : w = s) (h₃ : l = 3 * w) : w * l = 108 :=
by
  -- This is a placeholder for a detailed proof.
  sorry

end area_of_rectangle_is_108_l576_576142


namespace least_three_digit_multiple_of_9_eq_108_l576_576538

/--
What is the least positive three-digit multiple of 9?
-/
theorem least_three_digit_multiple_of_9_eq_108 :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 9 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 9 = 0 → n ≤ m :=
  ∃ n : ℕ, n = 108 :=
begin
  sorry
end

end least_three_digit_multiple_of_9_eq_108_l576_576538


namespace min_troublemakers_29_l576_576499

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l576_576499


namespace bakery_customers_l576_576591

theorem bakery_customers 
    (total_pastries : ℕ) 
    (regular_customers : ℕ) 
    (regular_share : ℕ) 
    (reduction : ℕ) 
    (new_customers : ℕ) 
    (total_shared_pastries : ℕ) 
    (equal_division : total_pastries = regular_customers * regular_share)
    (new_share : regular_share - reduction = total_shared_pastries / new_customers)
    (total_pastries_eq : new_customers * total_shared_pastries = total_pastries) :
    new_customers = 49 := 
by
  have h1 : regular_share = 14 :=
    calc 
      regular_share = total_pastries / regular_customers : by sorry
      ... = 14 : by sorry
  have h2 : total_shared_pastries / new_customers = 8 :=
    calc 
      total_shared_pastries / new_customers = regular_share - reduction : by sorry
      ... = 14 - 6 : by congr; exact h1
      ... = 8 : by linarith
  have h3 : new_customers = total_pastries / 8 :=
    calc 
      new_customers = total_pastries / total_shared_pastries : by sorry
      ... = 392 / 8 : by sorry
      ... = 49 : by norm_num
  exact h3

end bakery_customers_l576_576591


namespace min_b_n_S_n_l576_576065

theorem min_b_n_S_n : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → (m - 8) * (m / (m + 1) : ℚ) ≥ -4) ∧ 
  ((n - 8) * (n / (n + 1) : ℚ) = -4) :=
by 
  use 2
  split
  · exact Nat.le_refl 2
  split
  · intro m Hm
    by_cases hm1 : m = 2
    · rw [hm1]
      norm_cast
      linarith
    by_cases hm2 : m = 4
    · rw [hm2]
      norm_cast
      linarith only [show (4 : ℚ) = 4, from rfl, show (4 + 1 : ℚ) = 5, from rfl]
    sorry
  · 
    norm_cast 
    linarith only [show (2 - 8 : ℚ) = -6, from rfl, show (2 + 1 : ℚ) = 3, from rfl]

end min_b_n_S_n_l576_576065


namespace range_of_a_for_monotonic_decreasing_l576_576736

noncomputable def f (a x : ℝ) : ℝ := a * x * |x - a|

def is_monotonic_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

theorem range_of_a_for_monotonic_decreasing :
  (a : ℝ) → Prop :=
  a ∈ (-∞, 0) ∪ [3/2, 2] ↔ 
  is_monotonic_decreasing (f a) (Set.Ioo 1 (3/2)) := sorry

end range_of_a_for_monotonic_decreasing_l576_576736


namespace students_without_glasses_l576_576954

theorem students_without_glasses (total_students: ℕ) (percentage_with_glasses: ℕ) (p: percentage_with_glasses = 40) (t: total_students = 325) : ∃ x : ℕ, x = (total_students * (100 - percentage_with_glasses)) / 100 ∧ x = 195 :=
by
  have total_students := 325
  have percentage_with_glasses := 40
  have percentage_without_glasses := 100 - percentage_with_glasses
  have number_without_glasses := (total_students * percentage_without_glasses) / 100
  exact ⟨number_without_glasses, number_without_glasses, rfl⟩

end students_without_glasses_l576_576954


namespace exists_points_no_three_collinear_integer_distances_l576_576397

theorem exists_points_no_three_collinear_integer_distances (n : ℕ) (h : n ≥ 2) : 
  ∃ (points : Fin n → ℝ × ℝ), 
    (∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → ¬ collinear (points i) (points j) (points k)) ∧
    (∀ i j : Fin n, i ≠ j → dist (points i) (points j) ∈ ℤ) :=
sorry

end exists_points_no_three_collinear_integer_distances_l576_576397


namespace correct_propositions_l576_576293

-- Definitions of the conditions in the Math problem

variable (triangle_outside_plane : Prop)
variable (triangle_side_intersections_collinear : Prop)
variable (parallel_lines_coplanar : Prop)
variable (noncoplanar_points_planes : Prop)

-- Math proof problem statement
theorem correct_propositions :
  (triangle_outside_plane ∧ 
   parallel_lines_coplanar ∧ 
   ¬noncoplanar_points_planes) →
  2 = 2 :=
by
  sorry

end correct_propositions_l576_576293


namespace quadratic_real_roots_implies_k_range_l576_576316

theorem quadratic_real_roots_implies_k_range (k : ℝ) 
  (h : ∃ x : ℝ, k * x^2 + 2 * x - 1 = 0)
  (hk : k ≠ 0) : k ≥ -1 ∧ k ≠ 0 :=
sorry

end quadratic_real_roots_implies_k_range_l576_576316


namespace grove_tree_selection_l576_576827

theorem grove_tree_selection (birches spruces pines aspens : ℕ) :
  birches + spruces + pines + aspens = 100 →
  (∀ s : set ℕ, s.card = 85 → (birches ∈ s ∧ spruces ∈ s ∧ pines ∈ s ∧ aspens ∈ s)) →
  ∀ t : set ℕ, t.card = 69 → (birches ∈ t ∧ spruces ∈ t) ∨ (birches ∈ t ∧ pines ∈ t) ∨ (birches ∈ t ∧ aspens ∈ t) ∨ (spruces ∈ t ∧ pines ∈ t) ∨ (spruces ∈ t ∧ aspens ∈ t) ∨ (pines ∈ t ∧ aspens ∈ t) :=
sorry

end grove_tree_selection_l576_576827


namespace range_of_f_l576_576061

def f (x : ℝ) : ℝ := 4^x - 2^(x + 2) - 3

theorem range_of_f : Set.range f = Set.Ici (-7) :=
sorry

end range_of_f_l576_576061


namespace sum_of_cubes_closest_to_2008_l576_576946

-- Define the integers whose cubes are closest to 2008
def nearest_cubes (n : ℤ) : ℤ × ℤ :=
  if n > 12 then (12, 13) else (13, 12)

-- Define the cubes of the nearest integers
def cube (n : ℤ) : ℤ := n * n * n

-- Define the sum of the two nearest cubes
def sum_of_nearest_cubes (n : ℤ) : ℤ :=
  let (a, b) := nearest_cubes (Int.to_nat (Real.cbrt n))
  in cube a + cube b

-- Define the main theorem to be proved
theorem sum_of_cubes_closest_to_2008 : sum_of_nearest_cubes 2008 = 3925 :=
by
  sorry

end sum_of_cubes_closest_to_2008_l576_576946


namespace fraction_of_students_playing_woodwind_or_brass_this_year_l576_576101

   variable (x : ℕ) -- The number of students in the school

   /- Conditions -/
   def woodwindLastYear := (1 / 2 : ℚ) * x
   def brassLastYear := (2 / 5 : ℚ) * x
   def percussionLastYear := (1 : ℚ) - (1 / 2) - (2 / 5)
   def woodwindLeftThisYear := (1 / 2 : ℚ) * woodwindLastYear
   def brassLeftThisYear := (1 / 4 : ℚ) * brassLastYear

   /- Fraction of students who play either woodwind or brass this year -/
   def woodwindThisYear := woodwindLastYear - woodwindLeftThisYear
   def brassThisYear := brassLastYear - brassLeftThisYear
   def studentsPlayingWoodwindOrBrassThisYear := woodwindThisYear + brassThisYear

   theorem fraction_of_students_playing_woodwind_or_brass_this_year :
     studentsPlayingWoodwindOrBrassThisYear / x = (11 / 20 : ℚ) :=
   by
     sorry
   
end fraction_of_students_playing_woodwind_or_brass_this_year_l576_576101


namespace min_unsuccessful_placements_l576_576837

-- Define the board as an 8x8 matrix of integers where each value is either 1 or -1.
def board : Type := matrix (fin 8) (fin 8) ℤ

-- Define a condition that each cell contains either 1 or -1
def valid_board (b : board) : Prop := ∀ i j, b i j = 1 ∨ b i j = -1

-- Define the positions relative to the center of the cross configuration
def cross_positions (i j : fin 8) : list (fin 8 × fin 8) :=
  [(i, j), (i.pred.pred, j), (i.succ.succ, j), (i, j.pred.pred), (i, j.succ.succ)]

-- Define the sum of cells in a cross configuration
def cross_sum (b : board) (i j : fin 8) : ℤ :=
  (cross_positions i j).sum (λ ⟨x, y⟩, b x y)

-- Define an unsuccessful cross placement
def unsuccessful_cross (b : board) (i j : fin 8) : Prop :=
  cross_sum b i j ≠ 0

-- Calculate the number of unsuccessful placements on the board
def count_unsuccessful_placements (b : board) : ℕ :=
  (finset.univ.product finset.univ).card (λ ⟨i, j⟩, unsuccessful_cross b i j)

-- The main theorem to prove
theorem min_unsuccessful_placements :
  ∃ b, valid_board b ∧ count_unsuccessful_placements b = 36 :=
sorry

end min_unsuccessful_placements_l576_576837


namespace exists_x0_lt_l576_576853

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d
noncomputable def Q (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem exists_x0_lt {a b c d p q r s : ℝ} (h1 : r < s) (h2 : s - r > 2)
  (h3 : ∀ x, r < x ∧ x < s → P x a b c d < 0 ∧ Q x p q < 0)
  (h4 : ∀ x, x < r ∨ x > s → P x a b c d >= 0 ∧ Q x p q >= 0) :
  ∃ x0, r < x0 ∧ x0 < s ∧ P x0 a b c d < Q x0 p q :=
sorry

end exists_x0_lt_l576_576853


namespace apples_juice_amount_l576_576418

/-- The total amount of apples produced per year in million tons --/
def total_apples : ℝ := 6.5

/-- The percentage of apples mixed with other products --/
def percentage_mixed : ℝ := 0.25

/-- The percentage of the remaining apples used for juice --/
def percentage_for_juice : ℝ := 0.60

/-- Calculation of apples for juice, rounded to the nearest tenth --/
noncomputable def apples_for_juice : ℝ :=
  let remaining := (1 - percentage_mixed) * total_apples
  let for_juice := percentage_for_juice * remaining
  (Real.toRat for_juice).num.toNat * 0.1

/-- Final theorem statement: the apples used for juice should equal 2.9 million tons --/
theorem apples_juice_amount : apples_for_juice = 2.9 := by
  sorry

end apples_juice_amount_l576_576418


namespace positive_integers_satisfy_condition_l576_576914

theorem positive_integers_satisfy_condition :
  ∃ n, ∀ (n : ℕ), (6 * n + 25 < 40) ↔ (n = 1 ∨ n = 2) :=
by
  sorry

end positive_integers_satisfy_condition_l576_576914


namespace infinite_series_sum_l576_576222

theorem infinite_series_sum :
  (∑' n : ℕ, n * (1/5)^n) = 5/16 :=
by sorry

end infinite_series_sum_l576_576222


namespace value_of_x_l576_576692

theorem value_of_x (x : ℝ) : 
  10^(2*x) * 1000^x = 10000^3 ↔ x = 12 / 5 := 
by sorry

end value_of_x_l576_576692


namespace find_cos_beta_l576_576258

-- Declare the variables and hypotheses
variables {α β : ℝ}
-- α and β are acute angles
hypothesis (h₁ : 0 < α ∧ α < π / 2)
hypothesis (h₂ : 0 < β ∧ β < π / 2)
-- cosα = 4/5
hypothesis (h₃ : cos α = 4 / 5)
-- tan(α - β) = -1/3
hypothesis (h₄ : tan (α - β) = -1 / 3)

theorem find_cos_beta : cos β = 9 * sqrt 10 / 50 :=
by
  -- The proof would go here, but we put sorry to skip the proof step.
  sorry

end find_cos_beta_l576_576258


namespace polynomial_is_constant_l576_576854

theorem polynomial_is_constant (p : ℕ) (P : Polynomial ℝ) (h_prime : Nat.Prime p) (h_degree : P.degree < p - 1)
    (h_values : ∀ i : ℕ, 1 ≤ i ∧ i ≤ p → |P.eval i| = |P.eval 1|) : ∃ c : ℝ, P = Polynomial.C c :=
by
  sorry

end polynomial_is_constant_l576_576854


namespace min_troublemakers_l576_576498

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l576_576498


namespace minimum_liars_l576_576481

noncomputable def minimum_troublemakers (n : ℕ) := 10

theorem minimum_liars (n : ℕ) (students : ℕ → bool) 
  (seated_at_round_table : ∀ i, students (i % 29) = tt ∨ students (i % 29) = ff)
  (one_liar_next_to : ∀ i, students i = tt → (students ((i + 1) % 29) = ff ∨ students ((i + 28) % 29) = ff))
  (two_liars_next_to : ∀ i, students i = ff → ((students ((i + 1) % 29) = ff) ∧ students ((i + 28) % 29) = ff)) :
  ∃ m, m = minimum_troublemakers 29 ∧ (m ≤ n) :=
by
  let liars_count := 10
  have : liars_count = 10 := rfl
  use liars_count
  exact ⟨rfl, sorry⟩

end minimum_liars_l576_576481


namespace count_unit_squares_within_circle_l576_576405

def in_circle (x y : ℤ) : Prop := x^2 + y^2 < 9

def unit_squares (r : ℕ) (origin : ℤ × ℤ) : ℕ :=
  let (ox, oy) := origin
  (Finset.filter 
    (λ (p : ℤ × ℤ), let (x, y) := p in in_circle (x - ox) (y - oy))
    (Finset.product (Finset.Icc (-r) r) (Finset.Icc (-r) r))).card

theorem count_unit_squares_within_circle : unit_squares 3 (0, 0) = 21 :=
sorry

end count_unit_squares_within_circle_l576_576405


namespace birds_each_monkey_ate_l576_576180

/-- Initial number of monkeys -/
def M : ℕ := 6

/-- Initial number of birds -/
def B : ℕ := 6

/-- Number of birds each monkey eats -/
def x : ℕ

/-- Total number of animals after eating -/
def total_animals_after_eating : ℕ := M + (B - 2 * x)

/-- The fraction of animals that are monkeys after eating -/
def monkey_fraction_after_eating : ℝ := 0.6

theorem birds_each_monkey_ate :
  M = monkey_fraction_after_eating * total_animals_after_eating → x = 1 :=
by
  sorry

end birds_each_monkey_ate_l576_576180


namespace minimum_trees_l576_576796

variable (Trees : Type) [Fintype Trees] [DecidableEq Trees]

def trees_in_grove : Nat := 100

def tree_type := {birches, spruces, pines, aspens} : Set Trees

def condition (s : Finset Trees) : Prop := 
  s.card > 85 → tree_type ⊆ s

theorem minimum_trees (s : Finset Trees) (H : condition s) : 
  ∃ (n : Nat), n ≤ trees_in_grove ∧ n ≥ 69 → 
  ∃ t ⊆ s, t.card = n ∧ (|t ∩ tree_type| >= 3) :=
sorry

end minimum_trees_l576_576796


namespace distance_per_trip_l576_576382

--  Define the conditions as assumptions
variables (total_distance : ℝ) (num_trips : ℝ)
axiom h_total_distance : total_distance = 120
axiom h_num_trips : num_trips = 4

-- Define the question converted into a statement to be proven
theorem distance_per_trip : total_distance / num_trips = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end distance_per_trip_l576_576382


namespace rectangle_area_l576_576138

theorem rectangle_area (a w l : ℝ) (h_square_area : a = 36) 
    (h_rect_width : w * w = a) 
    (h_rect_length : l = 3 * w) : w * l = 108 := 
sorry

end rectangle_area_l576_576138


namespace min_troublemakers_l576_576458

theorem min_troublemakers (students : Finset ℕ) (table : List ℕ) (is_liar : ℕ → Bool) :
  students.card = 29 ∧
  (∀ s ∈ students, 
    (∃ i j : ℕ, 
      table.nth i = some s ∧ table.nth ((i + 1) % 29) = some j ∧
      (∃ k : ℕ, 
        (is_liar s = false → count (/=is_liar k liar) = 0 ∧
        is_liar s = true → count (/=is_liar k liar) = 2))) → 
  (∃! l ∈ students, count (/=is_liar l liar) = 1))) → 
  ∃ t : ℕ, t = 10 :=
sorry

end min_troublemakers_l576_576458


namespace minimum_liars_l576_576478

noncomputable def minimum_troublemakers (n : ℕ) := 10

theorem minimum_liars (n : ℕ) (students : ℕ → bool) 
  (seated_at_round_table : ∀ i, students (i % 29) = tt ∨ students (i % 29) = ff)
  (one_liar_next_to : ∀ i, students i = tt → (students ((i + 1) % 29) = ff ∨ students ((i + 28) % 29) = ff))
  (two_liars_next_to : ∀ i, students i = ff → ((students ((i + 1) % 29) = ff) ∧ students ((i + 28) % 29) = ff)) :
  ∃ m, m = minimum_troublemakers 29 ∧ (m ≤ n) :=
by
  let liars_count := 10
  have : liars_count = 10 := rfl
  use liars_count
  exact ⟨rfl, sorry⟩

end minimum_liars_l576_576478


namespace min_troublemakers_l576_576495

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l576_576495


namespace sachin_age_l576_576404

theorem sachin_age {Sachin_age Rahul_age : ℕ} (h1 : Sachin_age + 14 = Rahul_age) (h2 : Sachin_age * 9 = Rahul_age * 7) : Sachin_age = 49 := by
sorry

end sachin_age_l576_576404


namespace min_troublemakers_29_l576_576501

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l576_576501


namespace temperature_on_Friday_l576_576048

variable {M T W Th F : ℝ}

theorem temperature_on_Friday
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (hM : M = 41) :
  F = 33 :=
by
  -- Proof goes here
  sorry

end temperature_on_Friday_l576_576048


namespace percentage_loss_l576_576599

variable (CP SP : ℝ) (Loss : ℝ := CP - SP) (Percentage_of_Loss : ℝ := (Loss / CP) * 100)

theorem percentage_loss (h1: CP = 1600) (h2: SP = 1440) : Percentage_of_Loss = 10 := by
  sorry

end percentage_loss_l576_576599


namespace polynomial_positive_coeffs_of_shift_l576_576364

-- Initial definitions and conditions
variable {R : Type} [LinearOrderedField R]
variable {P : R[X]} (a : (Fin n → R)) (a_n : R) (m : R)

-- Definition of the polynomial P
def polynomial_P (a_n : R) (a : Fin n → R) : R[X] :=
  a_n * X ^ n + ∑ i in Finset.range n, (a i) * X ^ i

-- Conditions
variable (h_a_n : a_n ≥ 1)
variable (h_m : ∀ i, m > |a i| + 1)

-- Definition of the polynomial Q
def polynomial_Q (a_n : R) (a : Fin n → R) (m : R) : R[X] :=
  polynomial_P a_n a >>= (λ p, p.comp (X + C m))

-- The statement to prove
theorem polynomial_positive_coeffs_of_shift (a_n : R) (a : Fin n → R) (m : R)
  (h_a_n : a_n ≥ 1) (h_m : ∀ i, m > |a i| + 1) :
  ∀ i, (polynomial_Q a_n a m).coeff i > 0 := 
  sorry

end polynomial_positive_coeffs_of_shift_l576_576364


namespace min_troublemakers_in_class_l576_576463

noncomputable def min_troublemakers : ℕ :=
  10

theorem min_troublemakers_in_class :
  (∃ t l : ℕ, t + l = 29 ∧ t + l - 1 = 29 ∧
   (∀ i : ℕ, i < 29 → (i % 3 = 0 → ∃ t : ℕ, t = 1) ∧ 
   (i % 3 ≠ 0 → ∃ t : ℕ, t = 2))) →
   min_troublemakers = 10 :=
by
  sorry

end min_troublemakers_in_class_l576_576463


namespace value_of_fraction_power_series_l576_576095

theorem value_of_fraction_power_series (x : ℕ) (h : x = 3) :
  (x^3 * x^5 * x^7 * x^9 * x^11 * x^13 * x^15 * x^17 * x^19 * x^21) /
  (x^4 * x^8 * x^12 * x^16 * x^20 * x^24) = 3^36 :=
by
  subst h
  sorry

end value_of_fraction_power_series_l576_576095


namespace oranges_left_after_taking_48_percent_l576_576516

theorem oranges_left_after_taking_48_percent :
  ∀ (total_oranges : ℕ) (percentage_taken : ℕ),
    total_oranges = 96 →
    percentage_taken = 48 →
    let taken_oranges := (percentage_taken * total_oranges) / 100 in
    total_oranges - taken_oranges = 50 :=
by
  intros total_oranges percentage_taken h_total h_percentage
  let taken_oranges := (percentage_taken * total_oranges) / 100
  have h1 : taken_oranges = 46 :=
    by
      calc
        taken_oranges = (48 * 96) / 100 : by rw [h_total, h_percentage]
                    ... = 4608 / 100   : by norm_num
                    ... = 46           : by norm_num
  show total_oranges - taken_oranges = 50
  calc
    total_oranges - taken_oranges = 96 - 46 : by rw [h_total, h1]
                          ... = 50        : by norm_num

end oranges_left_after_taking_48_percent_l576_576516


namespace smallest_trees_in_three_types_l576_576781

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end smallest_trees_in_three_types_l576_576781


namespace exists_tetrahedron_with_square_sections_l576_576661

theorem exists_tetrahedron_with_square_sections :
  ∀ a b : ℝ, a > 0 ∧ b > 0 → ∃ tetrahedron, 
  (∃ s₁ s₂ : ℝ, s₁ = 100 ∧ s₂ = 1 ∧ 
    ∃ plane₁ plane₂ : set ℝ, 
      (tetrahedron ∩ plane₁ = square(s₁) ∧ tetrahedron ∩ plane₂ = square(s₂))) :=
by sorry

-- Definitions of tetrahedron and square can be appropriately filled in.

end exists_tetrahedron_with_square_sections_l576_576661


namespace find_omega_max_value_and_set_l576_576233

-- Definitions based on conditions
def a (ω x : ℝ) := (Real.cos (ω * x), Real.sin (ω * x))
def b (ω x : ℝ) := (2 * Real.cos (ω * x) + Real.sin (ω * x), Real.cos (ω * x))
def f (ω x : ℝ) := (a ω x).1 * (b ω x).1 + (a ω x).2 * (b ω x).2

-- Problem 1: Given conditions, prove ω = 4
theorem find_omega (ω : ℝ) (h : ω > 0) (hf : ∀ x, f ω x = f ω (x + π / 4)) : ω = 4 := sorry

-- Problem 2: Given ω = 4, find the maximum value and the set of x for which f(x) attains this maximum value
theorem max_value_and_set (x : ℝ) :
  ∃ k : ℤ, x = π / 32 + k * π / 4 ∧ f 4 x = 1 + Real.sqrt 2 := sorry

end find_omega_max_value_and_set_l576_576233


namespace correct_statement_C_l576_576228

-- Definitions for lines and planes
variables (Line Plane : Type) [has_subset Line Plane] [has_parallel Line Line] [has_parallel Line Plane]
variables {m n : Line} {α β : Plane}

-- Conditions
def condition_C1 (m : Line) (α : Plane) : Prop := m ⊆ α
def condition_C2 (n : Line) (α : Plane) : Prop := ∃ p, n ∥ α
def condition_C3 (m n : Line) (β : Plane) : Prop := ∀ p. collinear p m n

-- Theorem
theorem correct_statement_C (h1 : condition_C1 m α) (h2 : condition_C2 n α) (h3 : condition_C3 m n β) : m ∥ n := sorry

end correct_statement_C_l576_576228


namespace count_odd_three_digit_numbers_l576_576035

-- Define the set of digits
def digits := {0, 1, 2, 3, 4}

-- Definition stating a digit belongs to the defined set of digits
def is_digit (d : ℕ) : Prop := d ∈ digits

-- Definition stating the number is a three-digit number
def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Definition stating the number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Definition stating the digits in the number are different
def has_unique_digits (n : ℕ) : Prop :=
  let d1 := n / 100,
      d2 := (n / 10) % 10,
      d3 := n % 10
  in d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

-- The main statement that we want to prove
theorem count_odd_three_digit_numbers : 
  ∃ count : ℕ, count = 18 ∧ 
              count = (finset.filter 
                        (λ n, is_three_digit_number n ∧ 
                              is_odd n ∧ 
                              has_unique_digits n)
                        (finset.range 1000)).card :=
by sorry

end count_odd_three_digit_numbers_l576_576035


namespace smallest_trees_in_three_types_l576_576780

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end smallest_trees_in_three_types_l576_576780


namespace slope_of_tangent_line_at_x_2_l576_576218

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3*x

theorem slope_of_tangent_line_at_x_2 : (deriv curve 2) = 7 := by
  sorry

end slope_of_tangent_line_at_x_2_l576_576218


namespace coefficient_of_x5_l576_576672

theorem coefficient_of_x5 :
  let expr := 2 * (x^3 - x^5 + 2 * x^2) + 4 * (x^4 + 3 * x^5 - x^3 + 2 * x^2) - 3 * (2 + 2 * x^2 - 3 * x^5 - x^4)
  coeff_of_x5 (expr) = 19 := by
  sorry

end coefficient_of_x5_l576_576672


namespace least_positive_integer_l576_576978

theorem least_positive_integer (n : ℕ) :
  (∃ n : ℕ, 25^n + 16^n ≡ 1 [MOD 121] ∧ ∀ m : ℕ, (m < n ∧ 25^m + 16^m ≡ 1 [MOD 121]) → false) ↔ n = 32 :=
sorry

end least_positive_integer_l576_576978


namespace division_remainder_l576_576389

theorem division_remainder (dividend quotient divisor remainder : ℕ) (h1 : dividend = 109)
    (h2 : quotient = 9) (h3 : divisor = 12) (h4 : dividend = divisor * quotient + remainder) :
    remainder = 1 :=
by
  subst h1
  subst h2
  subst h3
  simp at h4
  exact h4

end division_remainder_l576_576389


namespace clock_hand_overlaps_in_24_hours_l576_576295

-- Define the number of revolutions of the hour hand in 24 hours.
def hour_hand_revolutions_24_hours : ℕ := 2

-- Define the number of revolutions of the minute hand in 24 hours.
def minute_hand_revolutions_24_hours : ℕ := 24

-- Define the number of overlaps as a constant.
def number_of_overlaps (hour_rev : ℕ) (minute_rev : ℕ) : ℕ :=
  minute_rev - hour_rev

-- The theorem we want to prove:
theorem clock_hand_overlaps_in_24_hours :
  number_of_overlaps hour_hand_revolutions_24_hours minute_hand_revolutions_24_hours = 22 :=
sorry

end clock_hand_overlaps_in_24_hours_l576_576295


namespace acid_solution_mix_l576_576520

theorem acid_solution_mix (x : ℝ) (h₁ : 0.2 * x + 50 = 0.35 * (100 + x)) : x = 100 :=
by
  sorry

end acid_solution_mix_l576_576520


namespace symmetrical_triangle_l576_576911

variables {ABC : Type} [triangle ABC]
variables (A B C O H A_1 B_1 C_1 : ABC)
variables [is_circumcenter A B C O]

-- Define reflections
variables (hA1 : A_1 = reflect_over O (side B C))
variables (hB1 : B_1 = reflect_over O (side A C))
variables (hC1 : C_1 = reflect_over O (side A B))

theorem symmetrical_triangle :
  (is_circumcenter A B C O) ∧
  (A_1 = reflect_over O (side B C)) ∧
  (B_1 = reflect_over O (side A C)) ∧
  (C_1 = reflect_over O (side A B)) →
  -- Centrally Symmetric Triangle
  triangle_centrally_symmetric (triangle A B C) (triangle A_1 B_1 C_1) ∧
  -- O bisects OH
  midpoint O H = O ∧
  -- O is the orthocenter of A_1 B_1 C_1
  is_orthocenter_of O (triangle A_1 B_1 C_1) :=
begin
  sorry,
end

end symmetrical_triangle_l576_576911


namespace find_diameter_endpoint_l576_576325

def circle_center : ℝ × ℝ := (4, 1)
def diameter_endpoint_1 : ℝ × ℝ := (1, 5)

theorem find_diameter_endpoint :
  let (h, k) := circle_center
  let (x1, y1) := diameter_endpoint_1
  (2 * h - x1, 2 * k - y1) = (7, -3) :=
by
  let (h, k) := circle_center
  let (x1, y1) := diameter_endpoint_1
  sorry

end find_diameter_endpoint_l576_576325


namespace minimum_trees_l576_576795

variable (Trees : Type) [Fintype Trees] [DecidableEq Trees]

def trees_in_grove : Nat := 100

def tree_type := {birches, spruces, pines, aspens} : Set Trees

def condition (s : Finset Trees) : Prop := 
  s.card > 85 → tree_type ⊆ s

theorem minimum_trees (s : Finset Trees) (H : condition s) : 
  ∃ (n : Nat), n ≤ trees_in_grove ∧ n ≥ 69 → 
  ∃ t ⊆ s, t.card = n ∧ (|t ∩ tree_type| >= 3) :=
sorry

end minimum_trees_l576_576795


namespace find_plane_equation_l576_576677

noncomputable def plane_equation : ℕ × ℕ × ℕ × ℤ :=
let a := (-3, 4, -2),
    b := (1, 4, 0),
    c := (3, 2, -1),
    direction1 := (b.1 - a.1, b.2 - a.2, b.3 - a.3),
    direction2 := (c.1 - a.1, c.2 - a.2, c.3 - a.3),
    cross_product := (direction1.2 * direction2.3 - direction1.3 * direction2.2,
                      direction1.3 * direction2.1 - direction1.1 * direction2.3,
                      direction1.1 * direction2.2 - direction1.2 * direction2.1),
    gcd_all := Int.gcd (Int.gcd cross_product.1 cross_product.2) cross_product.3,
    n := (cross_product.1 / gcd_all, cross_product.2 / gcd_all, cross_product.3 / gcd_all),
    A := n.1,
    B := n.2,
    C := n.3,
    D := -(A * a.1 + B * a.2 + C * a.3)
in (A, B, C, D)

theorem find_plane_equation : plane_equation = (1, 2, -2, -9) :=
sorry

end find_plane_equation_l576_576677


namespace proof_problem_l576_576013

noncomputable def triangle := Type
noncomputable def point := Type

variables (A B C I D M E : point)
variables (ABC : triangle)
variables (incircle_center : ∀ (ABC : triangle), point)
variables (touches_side : ∀ (I : point) (ABC : triangle) (BC : point), point)

-- Defining the conditions from part a
def conditions (ABC : triangle) (A B C I D M E : point) :=
  incircle_center ABC = I ∧
  touches_side I ABC BC = D ∧
  midpoint (altitude A BC) = M ∧
  intersects (line_through M I) BC = E

-- Statement of the theorem
theorem proof_problem (ABC : triangle) (A B C I D M E : point) :
  conditions ABC A B C I D M E → (length_segment B D = length_segment C E) :=
begin
  sorry -- proof to be written
end

end proof_problem_l576_576013


namespace total_path_length_A_l576_576624

theorem total_path_length_A (α : ℝ) (hα : 0 < α ∧ α < π / 3) : 
  ∃ B A C : ℝ → ℝ, 
  (B = 0) ∧ (|A| = 1) ∧ (|C| = 1) ∧ 
  angle (B - A) (C - B) = 2 * α ∧ 
  (total_length (rotate_around A, rotate_around B, rotate_around C) 100 A = 22 * π * (1 + sin α) - 66 * α) :=
sorry

end total_path_length_A_l576_576624


namespace count_double_seven_side_by_side_correct_l576_576448

def has_double_seven_side_by_side (n : ℕ) : Bool :=
  toString n |> fun s => s.contains "77"

def count_double_seven_side_by_side : ℕ :=
  (List.range' 1 1001).filter has_double_seven_side_by_side |>.length

theorem count_double_seven_side_by_side_correct : count_double_seven_side_by_side = 19 := by
  sorry

end count_double_seven_side_by_side_correct_l576_576448


namespace quadratic_real_roots_l576_576313

theorem quadratic_real_roots (k : ℝ) (h1 : k ≠ 0) : (4 + 4 * k) ≥ 0 ↔ k ≥ -1 := 
by 
  sorry

end quadratic_real_roots_l576_576313


namespace coefficient_of_linear_term_l576_576423

theorem coefficient_of_linear_term : ∀ (a b c x : Real), a = 1 ∧ b = 3 ∧ c = -1 → (a * x^2 + b * x + c = 0 → b = 3) :=
by
  intros a b c x h
  cases h with ha hb
  cases hb with hb hc
  simp [ha, hb, hc]
  sorry

end coefficient_of_linear_term_l576_576423


namespace exists_f_add_not_exists_f_sub_l576_576846

noncomputable theory

open set

-- The function f(x) is defined on the entire real line.
variable (f : ℝ → ℝ)

-- Proof of existence for part a)
theorem exists_f_add : (∃ f : ℝ → ℝ, (∀ x : ℝ, f (sin x) + f (cos x) = 1) ∧ (∀ x y : ℝ, x ≠ y → f x ≠ f y)) :=
by {
  -- statement only, proof not required
  sorry
}

-- Proof of non-existence for part b)
theorem not_exists_f_sub : ¬ (∃ f : ℝ → ℝ, (∀ x : ℝ, f (sin x) - f (cos x) = 1) ∧ (∀ x y : ℝ, x ≠ y → f x ≠ f y)) :=
by {
  -- statement only, proof not required
  sorry
}

end exists_f_add_not_exists_f_sub_l576_576846


namespace problem_l576_576337

def sequence (a : ℕ → ℚ) :=
  (a 1 = 1) ∧ ∀ n, n ≥ 2 → a n = 3 * a (n - 1) / (a (n - 1) + 3)

theorem problem (a : ℕ → ℚ) (h_seq : sequence a) :
  a 2 = 3 / 4 ∧ a 3 = 3 / 5 ∧ a 4 = 1 / 2 ∧ ∀ n ≥ 1, a n = 3 / (n + 2) :=
begin
  sorry
end

end problem_l576_576337


namespace evaluate_functions_l576_576303

def f (x : ℝ) := x + 2
def g (x : ℝ) := 2 * x^2 - 4
def h (x : ℝ) := x + 1

theorem evaluate_functions : f (g (h 3)) = 30 := by
  sorry

end evaluate_functions_l576_576303


namespace mode_combined_sets_l576_576524

theorem mode_combined_sets 
  (m n : ℤ)
  (h₁ : (-2 + m + 2 * n + 9 + 12) / 5 = 5) 
  (h₂ : (3 * m + 7 + n) / 3 = 5) 
  (hm : m = 2) 
  (hn : n = 2) :
  mode [-2, 2 * n, 9, 12, 3 * m, 7, n, -2, m, 2, 9, 12].head = 2 := 
by
  sorry

end mode_combined_sets_l576_576524


namespace index_card_area_l576_576881

theorem index_card_area
  (L W : ℕ)
  (h1 : L = 4)
  (h2 : W = 6)
  (h3 : (L - 1) * W = 18) :
  (L * (W - 1) = 20) :=
by
  sorry

end index_card_area_l576_576881


namespace quadratic_square_binomial_l576_576761

theorem quadratic_square_binomial (a : ℝ) :
  (∃ d : ℝ, 9 * x ^ 2 - 18 * x + a = (3 * x + d) ^ 2) → a = 9 :=
by
  intro h
  match h with
  | ⟨d, h_eq⟩ => sorry

end quadratic_square_binomial_l576_576761


namespace rectangle_area_l576_576139

theorem rectangle_area (a w l : ℝ) (h_square_area : a = 36) 
    (h_rect_width : w * w = a) 
    (h_rect_length : l = 3 * w) : w * l = 108 := 
sorry

end rectangle_area_l576_576139


namespace colin_average_mile_time_l576_576638

theorem colin_average_mile_time :
  let first_mile_time := 6
  let next_two_miles_total_time := 5 + 5
  let fourth_mile_time := 4
  let total_time := first_mile_time + next_two_miles_total_time + fourth_mile_time
  let number_of_miles := 4
  (total_time / number_of_miles) = 5 := by
    let first_mile_time := 6
    let next_two_miles_total_time := 5 + 5
    let fourth_mile_time := 4
    let total_time := first_mile_time + next_two_miles_total_time + fourth_mile_time
    let number_of_miles := 4
    have h1 : total_time = 20 := by sorry
    have h2 : total_time / number_of_miles = 20 / 4 := by sorry
    have h3 : 20 / 4 = 5 := by sorry
    exact Eq.trans (Eq.trans h2 h3) h1.symm

end colin_average_mile_time_l576_576638


namespace vector_norm_difference_l576_576290

variable {V : Type} [inner_product_space ℝ V]

variables (a b : V)
variables (h1 : ∥a∥ = 2*real.sqrt 2)
variables (h2 : ∥b∥ = real.sqrt 2)
variables (h3 : inner_product_space.inner a b = 1)

theorem vector_norm_difference :
  ∥a - 2 • b∥ = 2 * real.sqrt 3 :=
by
  sorry

end vector_norm_difference_l576_576290


namespace sum_of_digits_of_even_1_to_12000_l576_576685

def is_even (n : ℕ) : Prop := n % 2 = 0

def digits_sum (n : ℕ) : ℕ := n.digits.sum

def sum_of_digits_of_even_numbers (n : ℕ) : ℕ :=
  (list.range n).filter is_even |> list.map digits_sum |> list.sum

theorem sum_of_digits_of_even_1_to_12000 :
  sum_of_digits_of_even_numbers 12001 = 129348 :=
sorry

end sum_of_digits_of_even_1_to_12000_l576_576685


namespace rectangle_area_from_square_l576_576149

theorem rectangle_area_from_square {s a : ℝ} (h1 : s^2 = 36) (h2 : a = 3 * s) :
    s * a = 108 :=
by
  -- The proof goes here
  sorry

end rectangle_area_from_square_l576_576149


namespace minimum_trees_with_at_least_three_types_l576_576812

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end minimum_trees_with_at_least_three_types_l576_576812


namespace Elaine_rent_increase_l576_576851

noncomputable def Elaine_rent_percent (E: ℝ) : ℝ :=
  let last_year_rent := 0.20 * E
  let this_year_earnings := 1.25 * E
  let this_year_rent := 0.30 * this_year_earnings
  let ratio := (this_year_rent / last_year_rent) * 100
  ratio

theorem Elaine_rent_increase (E : ℝ) : Elaine_rent_percent E = 187.5 :=
by 
  -- The proof would go here.
  sorry

end Elaine_rent_increase_l576_576851


namespace rectangle_has_unique_diagonal_property_l576_576937

def is_parallelogram (s : Type) [IsParallelogram s] : Prop := true
def is_rectangle (s : Type) [IsRectangle s] : Prop := true
def is_rhombus (s : Type) [IsRhombus s] : Prop := true
def diagonals_equal (s : Type) [HasDiagonals s] : Prop := true

theorem rectangle_has_unique_diagonal_property (s : Type) [IsRectangle s]:
  is_rectangle s →
  is_parallelogram s →
  (∀ s : Type, is_parallelogram s → is_rhombus s → ¬ diagonals_equal s) →
  diagonals_equal s :=
by
  sorry

end rectangle_has_unique_diagonal_property_l576_576937


namespace mean_score_seniors_138_l576_576082

def total_students : ℕ := 200
def mean_score_all : ℕ := 120

variable (s n : ℕ) -- number of seniors and non-seniors
variable (ms mn : ℚ) -- mean score of seniors and non-seniors

def non_seniors_twice_seniors := n = 2 * s
def mean_score_non_seniors := mn = 0.8 * ms
def total_students_eq := s + n = total_students

def total_score := (s : ℚ) * ms + (n : ℚ) * mn = (total_students : ℚ) * mean_score_all

theorem mean_score_seniors_138 :
  ∃ s n ms mn,
    non_seniors_twice_seniors s n ∧
    mean_score_non_seniors ms mn ∧
    total_students_eq s n ∧
    total_score s n ms mn → 
    ms = 138 :=
sorry

end mean_score_seniors_138_l576_576082


namespace colin_average_time_per_mile_l576_576642

theorem colin_average_time_per_mile :
  (let first_mile := 6
   let second_mile := 5
   let third_mile := 5
   let fourth_mile := 4
   let total_miles := 4
   let total_time := first_mile + second_mile + third_mile + fourth_mile
   let average_time := total_time / total_miles
   average_time = 5) :=
begin
  let first_mile := 6,
  let second_mile := 5,
  let third_mile := 5,
  let fourth_mile := 4,
  let total_miles := 4,
  let total_time := first_mile + second_mile + third_mile + fourth_mile,
  let average_time := total_time / total_miles,
  show average_time = 5,
  sorry,
end

end colin_average_time_per_mile_l576_576642


namespace family_of_functions_count_l576_576240

theorem family_of_functions_count : 
  (∀ f : ℝ → ℝ, (∀ x, f x = x^2) ∧ (∀ y, y ∈ set.range f → y = 1 ∨ y = 2) → 
  ∃ D : set ℝ, f '' D = {1, 2} ∧ set.finite D ∧ set.card D = 9) :=
begin
  sorry
end

end family_of_functions_count_l576_576240


namespace equilateral_sector_area_l576_576310

noncomputable def area_of_equilateral_sector (r : ℝ) : ℝ :=
  if h : r = r then (1/2) * r^2 * 1 else 0

theorem equilateral_sector_area (r : ℝ) : r = 2 → area_of_equilateral_sector r = 2 :=
by
  intros hr
  rw [hr]
  unfold area_of_equilateral_sector
  split_ifs
  · norm_num
  · contradiction

end equilateral_sector_area_l576_576310


namespace tax_rate_equals_65_l576_576586

def tax_rate_percentage := 65
def tax_rate_per_dollars (rate_percentage : ℕ) : ℕ :=
  (rate_percentage / 100) * 100

theorem tax_rate_equals_65 :
  tax_rate_per_dollars tax_rate_percentage = 65 := by
  sorry

end tax_rate_equals_65_l576_576586


namespace solution_set_exponential_inequality_l576_576944

theorem solution_set_exponential_inequality :
  { x : ℝ | 2^(x + 2) > 8 } = { x : ℝ | 1 < x } :=
by
  sorry

end solution_set_exponential_inequality_l576_576944


namespace grove_tree_selection_l576_576822

theorem grove_tree_selection (birches spruces pines aspens : ℕ) :
  birches + spruces + pines + aspens = 100 →
  (∀ s : set ℕ, s.card = 85 → (birches ∈ s ∧ spruces ∈ s ∧ pines ∈ s ∧ aspens ∈ s)) →
  ∀ t : set ℕ, t.card = 69 → (birches ∈ t ∧ spruces ∈ t) ∨ (birches ∈ t ∧ pines ∈ t) ∨ (birches ∈ t ∧ aspens ∈ t) ∨ (spruces ∈ t ∧ pines ∈ t) ∨ (spruces ∈ t ∧ aspens ∈ t) ∨ (pines ∈ t ∧ aspens ∈ t) :=
sorry

end grove_tree_selection_l576_576822


namespace find_x_l576_576757

theorem find_x (x : ℝ) (h1 : log 10 (x^2 - 5 * x + 10) = 2) (h2 : x > 0) : x = 15 :=
by
  sorry

end find_x_l576_576757


namespace samara_oil_spent_l576_576166

theorem samara_oil_spent (O : ℕ) (A_total : ℕ) (S_tires : ℕ) (S_detailing : ℕ) (diff : ℕ) (S_total : ℕ) :
  A_total = 2457 →
  S_tires = 467 →
  S_detailing = 79 →
  diff = 1886 →
  S_total = O + S_tires + S_detailing →
  A_total = S_total + diff →
  O = 25 :=
by
  sorry

end samara_oil_spent_l576_576166


namespace average_weight_of_remaining_students_l576_576769

theorem average_weight_of_remaining_students
  (M F M' F' : ℝ) (A A' : ℝ)
  (h1 : M + F = 60 * A)
  (h2 : M' + F' = 59 * A')
  (h3 : A' = A + 0.2)
  (h4 : M' = M - 45):
  A' = 57 :=
by
  sorry

end average_weight_of_remaining_students_l576_576769


namespace difference_in_mileage_l576_576028

-- Define the conditions
def advertised_mpg : ℝ := 35
def tank_capacity : ℝ := 12
def regular_gasoline_mpg : ℝ := 30
def premium_gasoline_mpg : ℝ := 40
def diesel_mpg : ℝ := 32
def fuel_proportion : ℝ := 1 / 3

-- Define the weighted average function
def weighted_average_mpg (mpg1 mpg2 mpg3 : ℝ) (proportion : ℝ) : ℝ :=
  (mpg1 * proportion) + (mpg2 * proportion) + (mpg3 * proportion)

-- Proof
theorem difference_in_mileage :
  advertised_mpg - weighted_average_mpg regular_gasoline_mpg premium_gasoline_mpg diesel_mpg fuel_proportion = 1 := by
  sorry

end difference_in_mileage_l576_576028


namespace triangle_angles_21_equal_triangles_around_square_l576_576348

theorem triangle_angles_21_equal_triangles_around_square
    (theta alpha beta gamma : ℝ)
    (h1 : 4 * theta + 90 = 360)
    (h2 : alpha + beta + 90 = 180)
    (h3 : alpha + beta + gamma = 180)
    (h4 : gamma + 90 = 180)
    : theta = 67.5 ∧ alpha = 67.5 ∧ beta = 22.5 ∧ gamma = 90 :=
by
  sorry

end triangle_angles_21_equal_triangles_around_square_l576_576348


namespace area_of_rectangle_l576_576154

--- Define the problem's conditions
def square_area : ℕ := 36
def rectangle_width := (square_side : ℕ) (h : square_area = square_side * square_side) : ℕ := square_side
def rectangle_length := (width : ℕ) : ℕ := 3 * width

--- State the theorem using the defined conditions
theorem area_of_rectangle (square_side : ℕ) 
  (h1 : square_area = square_side * square_side)
  (width := rectangle_width square_side h1)
  (length := rectangle_length width) :
  width * length = 108 := by
    sorry

end area_of_rectangle_l576_576154


namespace derivative_at_zero_l576_576566

noncomputable def f : ℝ → ℝ 
| 0       := 0
| (x : ℝ) := Real.arctan ((3 * x / 2) - x^2 * Real.sin (1 / x))

theorem derivative_at_zero : D f 0 = 3/2 :=
sorry

end derivative_at_zero_l576_576566


namespace num_pages_with_same_units_digit_l576_576581

-- Definitions based on the conditions in the problem statement
def is_valid_page (x : ℕ) : Prop := x ≤ 63

def has_same_units_digit (x y : ℕ) : Prop :=
  x % 10 = y % 10

-- The statement of the proof problem
theorem num_pages_with_same_units_digit : 
  {x : ℕ // is_valid_page x ∧ has_same_units_digit x (64 - x)}.card = 13 :=
by
  sorry

end num_pages_with_same_units_digit_l576_576581


namespace spring_extension_l576_576130

noncomputable theory

variables (k m g : ℝ)

def displacement (x : ℝ) : Prop :=
  (x^2 = (3 * m * g) / (2 * k))

theorem spring_extension (k m g : ℝ) (h₀ : k > 0) (h₁ : m > 0) (h₂ : g > 0) :
  ∃ x : ℝ, displacement k m g x :=
by {
  use real.sqrt ((3 * m * g) / (2 * k)),
  unfold displacement,
  field_simp [h₀.ne.symm],
  ring,
  sorry
}

end spring_extension_l576_576130


namespace probability_same_heads_l576_576850

theorem probability_same_heads :
  let outcomes := [(1, 3)] in
  let total_outcomes := 16 in
  let favorable_outcomes := 4 in
  favorable_outcomes / total_outcomes = 1 / 4 :=
by sorry

end probability_same_heads_l576_576850


namespace modulus_z_eq_sqrt_10_l576_576311

noncomputable def z : ℂ := (1 + 7 * Complex.I) / (2 + Complex.I)

theorem modulus_z_eq_sqrt_10 : Complex.abs z = Real.sqrt 10 := sorry

end modulus_z_eq_sqrt_10_l576_576311


namespace determinant_example_l576_576648

def det_2x2 (a b c d : ℤ) : ℤ := a * d - b * c

theorem determinant_example : det_2x2 7 (-2) (-3) 6 = 36 := by
  sorry

end determinant_example_l576_576648


namespace number_of_bags_bruced_can_buy_l576_576176

noncomputable def crayon_cost := 5
noncomputable def book_cost := 5
noncomputable def calculator_cost := 5
noncomputable def num_packs_of_crayons := 5
noncomputable def num_books := 10
noncomputable def num_calculators := 3
noncomputable def book_discount := 0.20
noncomputable def sales_tax := 0.05
noncomputable def bruce_money := 200
noncomputable def bag_cost := 10

theorem number_of_bags_bruced_can_buy :
  let crayon_total := crayon_cost * num_packs_of_crayons,
      book_total := book_cost * num_books,
      calculator_total := calculator_cost * num_calculators,
      total_before_discount := crayon_total + book_total + calculator_total,
      book_discount_amount := book_discount * book_total,
      new_book_total := book_total - book_discount_amount,
      new_total := crayon_total + new_book_total + calculator_total,
      tax := sales_tax * new_total,
      final_total := new_total + tax,
      change := bruce_money - final_total,
      num_bags := (change / bag_cost).floor in
  num_bags = 11 :=
by
  sorry

end number_of_bags_bruced_can_buy_l576_576176


namespace log_sqrt2_bounds_l576_576551

theorem log_sqrt2_bounds :
  10^3 = 1000 →
  10^4 = 10000 →
  2^11 = 2048 →
  2^12 = 4096 →
  2^13 = 8192 →
  2^14 = 16384 →
  3 / 22 < Real.log 2 / Real.log 10 / 2 ∧ Real.log 2 / Real.log 10 / 2 < 1 / 7 :=
by
  sorry

end log_sqrt2_bounds_l576_576551


namespace find_actual_number_l576_576421

theorem find_actual_number (S : ℕ → ℤ) (x : ℤ) (h1 : 10 * 16 = (∑ i in finset.range 10, S i) - x + 25)
  (h2 : 10 * 17 = (∑ i in finset.range 10, S i)) : x = 15 := by
  sorry

end find_actual_number_l576_576421


namespace point_B_coordinates_l576_576165

def move_up (x y : Int) (units : Int) : Int := y + units
def move_left (x y : Int) (units : Int) : Int := x - units

theorem point_B_coordinates :
  let A : Int × Int := (1, -1)
  let B : Int × Int := (move_left A.1 A.2 3, move_up A.1 A.2 2)
  B = (-2, 1) := 
by
  -- This is where the proof would go, but we omit it with "sorry"
  sorry

end point_B_coordinates_l576_576165


namespace quadratic_real_roots_l576_576314

theorem quadratic_real_roots (k : ℝ) (h1 : k ≠ 0) : (4 + 4 * k) ≥ 0 ↔ k ≥ -1 := 
by 
  sorry

end quadratic_real_roots_l576_576314


namespace oranges_for_friend_l576_576123

theorem oranges_for_friend (initial_oranges : ℕ)
                           (portion_to_brother : ℚ)
                           (portion_to_friend : ℚ) :
  initial_oranges = 12 →
  portion_to_brother = 1/3 →
  portion_to_friend = 1/4 →
  let oranges_brother := portion_to_brother * initial_oranges,
      oranges_remaining := initial_oranges - oranges_brother,
      oranges_friend := portion_to_friend * oranges_remaining in
  oranges_friend = 2 :=
by
  intros initial_oranges_eq portion_to_brother_eq portion_to_friend_eq
  have h1 : initial_oranges = 12 := initial_oranges_eq
  have h2 : portion_to_brother = 1/3 := portion_to_brother_eq
  have h3 : portion_to_friend = 1/4 := portion_to_friend_eq
  let oranges_brother := portion_to_brother * initial_oranges
  let oranges_remaining := initial_oranges - oranges_brother
  let oranges_friend := portion_to_friend * oranges_remaining
  sorry

end oranges_for_friend_l576_576123


namespace distinct_intersections_l576_576930

theorem distinct_intersections :
  let eq1 := (λ x y : ℝ, 2 * x - y + 3)
  let eq2 := (λ x y : ℝ, 4 * x + 2 * y - 5)
  let eq3 := (λ x y : ℝ, x - 2 * y - 1)
  let eq4 := (λ x y : ℝ, 3 * x - 4 * y + 6)
  (∃ p1 p2 p3 p4 : ℝ × ℝ,
    (eq1 p1.1 p1.2 = 0 ∧ eq3 p1.1 p1.2 = 0) ∧
    (eq1 p2.1 p2.2 = 0 ∧ eq4 p2.1 p2.2 = 0) ∧
    (eq2 p3.1 p3.2 = 0 ∧ eq3 p3.1 p3.2 = 0) ∧
    (eq2 p4.1 p4.2 = 0 ∧ eq4 p4.1 p4.2 = 0) ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧
    p2 ≠ p3 ∧ p2 ≠ p4 ∧
    p3 ≠ p4) :=
sorry

end distinct_intersections_l576_576930


namespace greatest_solution_l576_576681

open Real

noncomputable def max_solution_in_interval : ℝ :=
  let sol1 := (π / 6) in
  let sol2 := (π / 6 + π) in
  let sol3 := (π / 6 + 2 * π) in
  let sol4 := (π / 6 + 3 * π) in
  let sol5 := (π / 6 + 4 * π) in
  let sol6 := (π / 6 + 5 * π) in
  max 0 (max sol1 (max sol2 (max sol3 (max sol4 (max sol5 sol6)))))

theorem greatest_solution : (|2 * sin max_solution_in_interval - 1| + |2 * cos (2 * max_solution_in_interval) - 1| = 0) ∧ (round (max_solution_in_interval * 10^3) / 10^3 = 27.7) :=
by
  let sols := [π / 6, 13 * π / 6, 25 * π / 6, 37 * π / 6, 49 * π / 6, 61 * π / 6]
  have h_solutions : ∀ x ∈ sols, (|2 * sin x - 1| + |2 * cos (2 * x) - 1| = 0) := sorry
  have h_max : max_solution_in_interval = 61 * π / 6 := sorry
  simp [max_solution_in_interval, h_max]
  have h_rounded : round (61 * π / 6 * 10^3) / 10^3 = 27.7 := sorry
  exact ⟨h_solutions _, h_rounded⟩

end greatest_solution_l576_576681


namespace solution_l576_576744

def sequence (a : ℕ → ℤ) : Prop := ∀ n, n ≥ 2 → a n - (-1 : ℤ)^n * a (n - 1) = n

def Sn (a : ℕ → ℤ) (n : ℕ) : ℤ := ∑ i in finset.range (n + 1), a i

theorem solution (a : ℕ → ℤ) (h : sequence a) : Sn a 40 = 440 :=
sorry

end solution_l576_576744


namespace angles_sum_l576_576840

theorem angles_sum :
  ∀ (A B D x z : ℝ), 
  A = 30 ∧ 
  B = 70 ∧ 
  D = 29 → 
  x + z = 129 :=
by {
  intros A B D x z h,
  sorry
}

end angles_sum_l576_576840


namespace range_of_k_equation_of_line_exist_point_C_l576_576834

-- The equation of the circle
def circle (x y : ℝ) := x^2 + y^2 - 12 * x + 32 = 0

-- The line equation passing through point (0,2) with slope k
def line (x k : ℝ) := k * x + 2

-- Problem 1: Proving the range of k
theorem range_of_k (k : ℝ) : 
  (∃x y : ℝ, line x k = y ∧ circle x y) ↔ (-3 / 4 < k ∧ k < 0) := 
  sorry

-- Problem 2: Proving the equation of the line given a condition on dot product
theorem equation_of_line (k : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, line x1 k = y1 ∧ line x2 k = y2 ∧ circle x1 y1 ∧ circle x2 y2 ∧ (x1 * x2 + y1 * y2) = 28) → 
  (update line equation to be y = (-3 + sqrt(6)) * x + 2 := 
  sorry

-- Problem 3: Existence of point C on the y-axis
theorem exist_point_C (k : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, line x1 k = y1 ∧ line x2 k = y2 ∧ circle x1 y1 ∧ circle x2 y2 ∧ (∃ a : ℝ, (0, a) ∈ y_axis ∧ 
  (x1 * x2 + (y1 - a) * (y2 - a)) = 36) :=
  sorry

end range_of_k_equation_of_line_exist_point_C_l576_576834


namespace min_troublemakers_l576_576497

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l576_576497


namespace find_annual_interest_rate_l576_576205

noncomputable def compound_interest_problem : Prop :=
  ∃ (r : ℝ),
    let P := 8000
    let CI := 3109
    let t := 2.3333
    let A := 11109
    let n := 1
    A = P * (1 + r/n)^(n*t) ∧ r = 0.1505

theorem find_annual_interest_rate : compound_interest_problem :=
by sorry

end find_annual_interest_rate_l576_576205


namespace no_such_function_exists_l576_576038

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f^[n] n = n + 1 :=
by
  sorry

end no_such_function_exists_l576_576038


namespace correct_option_l576_576096

-- Definition of given options
def option_a : Prop := (2023 : ℝ)^0 = 1
def option_b : Prop := sqrt 7 + sqrt 6 = sqrt 13
def option_c (a : ℝ) : Prop := (-3 * a^2)^3 = -27 * a^3
def option_d (a : ℝ) : Prop := a^7 / a^3 = a^4

-- The main theorem to state the correct option
theorem correct_option : (∀ a : ℝ, ¬ option_a) ∧ (¬ option_b) ∧ (∀ a : ℝ, ¬ option_c a) ∧ (∀ a : ℝ, option_d a) :=
by
  sorry

end correct_option_l576_576096


namespace sum_of_values_satisfying_eq_l576_576094

theorem sum_of_values_satisfying_eq (x : ℝ) :
  (x^2 - 5 * x + 5 = 16) → ∀ r s : ℝ, (r + s = 5) :=
by
  sorry  -- Proof is omitted, looking to verify the structure only.

end sum_of_values_satisfying_eq_l576_576094


namespace purely_imaginary_simplify_fraction_l576_576264

-- Define the given complex number z
def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * complex.I

-- Statement Part I: Prove z is purely imaginary if and only if m = -1/2
theorem purely_imaginary (m : ℝ) : z m.Im = 0 ↔ m = -1 / 2 :=
by
  rw [z, complex.im_add, complex.im_of_real, add_zero, complex.im_smul, complex.of_real_eq_zero]
  apply sorry

-- Define z when m = 0
def z_m0 := z 0

-- Statement Part II: Simplify z^2 / (z + 5 + 2i) when m = 0
theorem simplify_fraction : z_m0^2 / (z_m0 + 5 + (2:ℂ) * complex.I) = -32 / 25 - 24 / 25 * complex.I :=
by
  rw [z_m0, complex.add, complex.mul, complex.of_real, complex.pow_two]
  apply sorry

end purely_imaginary_simplify_fraction_l576_576264


namespace CK_equals_24_over_5_angle_ACB_equality_l576_576838

noncomputable def CK_length (CK1 BK2 BC : ℕ) (R r : ℝ) : ℝ :=
  let CK := 3 * (R / r)
  CK

noncomputable def angle_ACB (r : ℝ) : ℝ :=
  2 * Real.arcsin (3 / 5) = Real.arccos (7 / 25)

theorem CK_equals_24_over_5 :
  let CK1 := 3
  let BK2 := 7
  let BC := 16
  ∃ (CK : ℝ) (R r : ℝ), 
    (R = (8 * r) / 5) ∧
    CK = 3 * (R / r) ∧
    CK1 + BK2 = BC → CK = 24 / 5 :=
by sorry

theorem angle_ACB_equality :
  ∃ (r : ℝ),
  2 * Real.arcsin (3 / 5) = Real.arccos (7 / 25) :=
by sorry

end CK_equals_24_over_5_angle_ACB_equality_l576_576838


namespace projections_of_perpendiculars_intersect_l576_576393

theorem projections_of_perpendiculars_intersect (A B C D A' B' C' D' : Point) (MediansIntersect : ∀ (X : Triangle), intersection (median (face ABCD A) Medians) = A' ∧ intersection (median (face ABCD B) Medians) = B' ∧ intersection (median (face ABCD C) Medians)) (Perpendiculars : ∀ (X : Point), perpendicular (to face (tetrahedron ABCD) at X)) : 
  ∃ (P : Point), ∀ (X Y Z : Point), Z ≠ X ∧ X ≠ Y ∧ Z ≠ Y → projections_intersect (projection (Perpendiculars A') (face ABCD)) (projection (Perpendiculars B') (face ABCD)) (projection (Perpendiculars C') (face ABCD)) P := sorry

end projections_of_perpendiculars_intersect_l576_576393


namespace equivalent_math_problem_l576_576331

noncomputable def parametric_curve_eq (α : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos α, sin α)

noncomputable def polar_line_eq (θ : ℝ) : ℝ := 
  (sqrt 2 / 2) * (real.cos (θ + real.pi / 4))

def cartesian_circle_eq (x y : ℝ) : Prop :=
  (x^2 / 3) + y^2 = 1

def cartesian_line_eq (x y : ℝ) : Prop :=
  x - y + 2 = 0

def cartesian_line1_eq (t : ℝ) : ℝ × ℝ :=
  ( -1 + (sqrt 2 / 2) * t, (sqrt 2 / 2) * t)

def quadratic_eq (t : ℝ) : ℝ :=
  2 * t^2 - sqrt 2 * t - 2

theorem equivalent_math_problem
  (α θ t : ℝ) :
  ∃ (C : ℝ × ℝ) (M A B : ℝ × ℝ), 
    cartesian_circle_eq C.1 C.2 ∧
    cartesian_line_eq (polar_line_eq θ) (polar_line_eq θ) = -1 ∧
    cartesian_line_eq (cartesian_line1_eq t).1 (cartesian_line1_eq t).2 ∧
    (quadratic_eq t = 0) ∧
    let A := cartesian_line1_eq t in
    let B := cartesian_line1_eq t in
    |A.1 * B.1 + A.2 * B.2| = 1 :=
sorry

end equivalent_math_problem_l576_576331


namespace minimum_liars_l576_576480

noncomputable def minimum_troublemakers (n : ℕ) := 10

theorem minimum_liars (n : ℕ) (students : ℕ → bool) 
  (seated_at_round_table : ∀ i, students (i % 29) = tt ∨ students (i % 29) = ff)
  (one_liar_next_to : ∀ i, students i = tt → (students ((i + 1) % 29) = ff ∨ students ((i + 28) % 29) = ff))
  (two_liars_next_to : ∀ i, students i = ff → ((students ((i + 1) % 29) = ff) ∧ students ((i + 28) % 29) = ff)) :
  ∃ m, m = minimum_troublemakers 29 ∧ (m ≤ n) :=
by
  let liars_count := 10
  have : liars_count = 10 := rfl
  use liars_count
  exact ⟨rfl, sorry⟩

end minimum_liars_l576_576480


namespace incorrect_statement_d_l576_576248

variable {a b : Plane}
variable {m n : Line}
 
theorem incorrect_statement_d (h1 : a ≠ b) 
  (h2 : m ∩ a ≠ ∅) 
  (h3 : a ∥ b → m ⊥ a → m ⊥ b) 
  (h4 : m ∥ n → m ⊥ a → n ⊥ a) 
  (h5 : m ⊥ a → m ∩ b ≠ ∅ → a ⊥ b) 
  (h6 : a ⊥ b → m ∩ a ≠ ∅ → m ⊥ b → False) : False := 
sorry

end incorrect_statement_d_l576_576248


namespace donovan_correct_answers_l576_576662

variable (C : ℝ)
variable (incorrectAnswers : ℝ := 13)
variable (percentageCorrect : ℝ := 0.7292)

theorem donovan_correct_answers :
  (C / (C + incorrectAnswers)) = percentageCorrect → C = 35 := by
  sorry

end donovan_correct_answers_l576_576662


namespace carrot_serving_problem_l576_576042

theorem carrot_serving_problem (total_carrots : ℕ) (uneaten_carrots : ℕ) (h_eq : total_carrots = 74) (h_uneaten : uneaten_carrots = 2) :
  ∃ n : ℕ, n ∣ (total_carrots - uneaten_carrots) ∧ 1 < n ∧ n < (total_carrots - uneaten_carrots) := 
by {
  sorry,
}

end carrot_serving_problem_l576_576042


namespace solution_conditions_l576_576678

variable (α β : Real)
noncomputable def solution_α (n : ℤ) := (2 * Real.pi / 3) * n
noncomputable def solution_β (k : ℤ) := (Real.pi / 4) + (k * Real.pi / 2)

theorem solution_conditions :
  ∃ (n k : ℤ), 
  (cos (2 * (α + (Real.pi * n)))) = cos (solution_α n) ∧ 
  (cos (2 * (β + (Real.pi * k)))) = cos (solution_β k) :=
by 
  sorry

end solution_conditions_l576_576678


namespace domain_of_f_l576_576184

noncomputable def f (x y : ℝ) : ℝ := real.sqrt (y - real.sqrt (8 - real.sqrt x))

theorem domain_of_f :
  {p : ℝ × ℝ | ∃ z, f p.1 p.2 = z} = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 64 ∧ 0 ≤ p.2} := 
by sorry

end domain_of_f_l576_576184


namespace range_of_a_for_monotonic_decrease_l576_576658

theorem range_of_a_for_monotonic_decrease (a : ℝ) :
    (∀ x ∈ set.Iic (a / 3), deriv (λ x, a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x) x ≤ 0) ↔ -1 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_for_monotonic_decrease_l576_576658


namespace min_liars_needed_l576_576471

namespace Problem

/-
  We introduce our problem using Lean definitions and statements.
-/

def students_sitting_at_table : ℕ := 29

structure Student :=
(is_liar : Bool)

def student_truth_statement (s: ℕ) (next1: Student) (next2: Student) : Bool :=
  if s then (next1.is_liar && !next2.is_liar) || (!next1.is_liar && next2.is_liar)
  else next1.is_liar && next2.is_liar

def minimum_liars_class (sitting_students: ℕ) : ℕ :=
  sitting_students / 3 + (if sitting_students % 3 = 0 then 0 else 1)

theorem min_liars_needed (students : List Student) (H_total : students.length = students_sitting_at_table)
  (H_truth_teller_claim : ∀ (i : ℕ), i < students.length → student_truth_statement i (students.get ⟨(i+1) % students.length, sorry⟩) (students.get ⟨(i+2) % students.length, sorry⟩)) :
  (students.count (λ s => s.is_liar)) ≥ minimum_liars_class students_sitting_at_table :=
sorry

end Problem

end min_liars_needed_l576_576471


namespace min_troublemakers_l576_576509

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l576_576509


namespace optimal_kn_value_l576_576238

theorem optimal_kn_value (n : ℕ) (a : Fin 3n → ℝ) 
  (h1 : ∀ i j : Fin 3n, i ≤ j → a i ≤ a j) 
  (h2 : n > 0 ∧ ∀ i : Fin 3n, 0 ≤ a i) : 
  (∑ i, a i) ^ 3 ≥ 3 * ∑ i in Finset.range n, a i * a (Fin n + i) * a (Fin 2n + i) := 
sorry

end optimal_kn_value_l576_576238


namespace hyperbola_equation_l576_576892

-- Define the condition: asymptotes of the hyperbola
def asymptote_eqn (x y : ℝ) : Prop := y = sqrt 3 * x ∨ y = -sqrt 3 * x

-- State the problem: Prove the equation of the hyperbola given the asymptotes
theorem hyperbola_equation (a b : ℝ) (h : ∀ x y, asymptote_eqn x y → y = sqrt 3 * x ∨ y = -sqrt 3 * x) : 
  ∃ x y, x^2 - y^2 / 3 = 1 :=
sorry

end hyperbola_equation_l576_576892


namespace dilute_solution_l576_576929

theorem dilute_solution :
  ∃ N : ℕ, let original_volume := 12
           let original_alcohol := 0.40 * original_volume
           let target_concentration := 0.25
           let required_volume := original_volume + N
           let new_alcohol_concentration := original_alcohol / required_volume
           new_alcohol_concentration = target_concentration ∧ N = 7 :=
by
  exists 7
  let original_volume := 12
  let original_alcohol := 0.40 * original_volume
  let target_concentration := 0.25
  let required_volume := original_volume + 7
  have new_alcohol_concentration : new_alcohol_concentration = original_alcohol / required_volume
  show new_alcohol_concentration = target_concentration
  sorry

end dilute_solution_l576_576929


namespace prove_A_share_of_profit_l576_576128

noncomputable def investment_over_time (amount : ℕ) (months : ℕ) : ℕ := amount * months
noncomputable def total_investment_over_time (a_invest_time : ℕ) (b_invest_time : ℕ) : ℕ := a_invest_time + b_invest_time
noncomputable def proportion (part : ℕ) (whole : ℕ) : ℚ := part / whole
noncomputable def profit_share (total_profit : ℚ) (ratio : ℚ) : ℚ := total_profit * ratio

theorem prove_A_share_of_profit :
  let a_invest := 300 in
  let b_invest := 200 in
  let a_months := 12 in
  let b_months := 6 in
  let total_profit := 100 in
  let a_invest_time := investment_over_time a_invest a_months in
  let b_invest_time := investment_over_time b_invest b_months in
  let total_time := total_investment_over_time a_invest_time b_invest_time in
  let a_ratio := proportion a_invest_time total_time in
  profit_share total_profit a_ratio = 75 :=
by
  sorry

end prove_A_share_of_profit_l576_576128


namespace arithmetic_sequence_20th_term_l576_576416

theorem arithmetic_sequence_20th_term (a₁ d : ℤ) (n t : ℤ) (h₁ : a₁ = 8) (h₂ : d = -3) (h₃ : n = 20) : 
  t = a₁ + (n - 1) * d → t = -49 := by
  intros h
  rw [h₁, h₂, h₃, h]
  sorry

end arithmetic_sequence_20th_term_l576_576416


namespace average_ge_neg_half_l576_576942

variable (n : ℕ) (a : Fin n → ℤ)
variable (h1 : a 0 = 0)
variable (h2 : ∀ k : Fin (n - 1), |a (k + 1)| = |a k + 1|)

theorem average_ge_neg_half (n : ℕ) (a : Fin n → ℤ)
  (h1 : a 0 = 0) 
  (h2 : ∀ k : Fin (n - 1), |a (k + 1)| = |a k + 1|) :
  (∑ i, a i : ℤ) / n ≥ -1 / 2 := 
sorry

end average_ge_neg_half_l576_576942


namespace remainder_M_1200_l576_576354

def binaryWithExactly9Ones (n : ℕ) : Prop :=
  nat.popcount n = 9

def sequenceT : ℕ → ℕ
| 0 => nat.find (λ n, binaryWithExactly9Ones n)
| (k+1) => nat.find (λ n, binaryWithExactly9Ones n ∧ n > sequenceT k)

def M : ℕ := sequenceT 1199

theorem remainder_M_1200 : M % 1200 = 88 := by
  sorry

end remainder_M_1200_l576_576354


namespace quadratic_real_roots_implies_k_range_l576_576315

theorem quadratic_real_roots_implies_k_range (k : ℝ) 
  (h : ∃ x : ℝ, k * x^2 + 2 * x - 1 = 0)
  (hk : k ≠ 0) : k ≥ -1 ∧ k ≠ 0 :=
sorry

end quadratic_real_roots_implies_k_range_l576_576315


namespace decompose_96_l576_576653

theorem decompose_96 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 96) (h4 : a^2 + b^2 = 208) : 
  (a = 8 ∧ b = 12) ∨ (a = 12 ∧ b = 8) :=
sorry

end decompose_96_l576_576653


namespace prize_winner_is_C_l576_576626

def entries : Type := {A, B, C, D}

def student_prediction_results (winning_entry : entries) : Prop := 
  let A_statement := (winning_entry = 'A')
  let B_statement := (winning_entry = 'C')
  let C_statement := (winning_entry ≠ 'B' ∧ winning_entry ≠ 'D')
  let D_statement := (winning_entry = 'A' ∨ winning_entry = 'D')
  (A_statement.to_bool[int] + B_statement.to_bool[int] + C_statement.to_bool[int] + D_statement.to_bool[int] = 2)

theorem prize_winner_is_C : ∃ X : entries, (student_prediction_results X) ∧ X = 'C' := sorry

end prize_winner_is_C_l576_576626


namespace weeks_in_a_month_l576_576181

theorem weeks_in_a_month (W : ℕ) (P : W * 20 * 2 + 20 = 180) : W = 4 :=
by {
  exact sorry,
}

end weeks_in_a_month_l576_576181


namespace fifth_number_sequence_l576_576329

theorem fifth_number_sequence : (2::16::4::6::14::12::8::[]) !! 4 = 6 :=
by
  sorry

end fifth_number_sequence_l576_576329


namespace min_troublemakers_29_l576_576505

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l576_576505


namespace meeting_probability_l576_576384

theorem meeting_probability :
  let steps := 8
  let total_paths := 2^steps
  let intersection_count := ∑ i in (Finset.range (steps + 1)), (Nat.choose steps i) ^ 2
  (intersection_count / total_paths ^ 2) = (6435 : ℚ) / 65536 :=
by
  sorry

end meeting_probability_l576_576384


namespace train_speed_l576_576081

theorem train_speed (d t : ℝ) (vR vH : ℝ) (h1 : d = 100) (h2 : vR = 45) (h3 : t = 4 / 3) :
  vH * t + vR * t = d → vH = 30 :=
by
  intros h
  field_simp [h, h1, h2, h3]
  ring
  sorry

end train_speed_l576_576081


namespace upstream_distance_l576_576602

theorem upstream_distance (v : ℝ) 
  (H1 : ∀ d : ℝ, (10 + v) * 2 = 28) 
  (H2 : (10 - v) * 2 = d) : d = 12 := by
  sorry

end upstream_distance_l576_576602


namespace pos_diff_squares_primes_l576_576539

def sumSquares (n: ℕ): ℕ := (List.range (n + 1)).map (λ i, i ^ 2).sum

def primeNumbersBetween (a: ℕ) (b: ℕ): List ℕ :=
  List.filter Nat.Prime (List.range' a (b + 1))

def sumList (lst : List ℕ) : ℕ := lst.foldl (λ acc x => acc + x) 0

theorem pos_diff_squares_primes :
  (sumSquares 6 - sumList (primeNumbersBetween 2 16)) = 50 := by
  sorry

end pos_diff_squares_primes_l576_576539


namespace sqrt_sum_inequality_l576_576871

theorem sqrt_sum_inequality (n : ℕ) (h : n ≥ 2)
  (x : Fin n → ℝ) (hx : ∀ i, 0 < x i ∧ x i < 1) :
  (∑ i, (sqrt (1 - x i)) / (x i)) < (sqrt (n - 1)) / (∏ i, (x i)) :=
by
  sorry

end sqrt_sum_inequality_l576_576871


namespace range_of_k_for_real_roots_roots_when_k_max_integer_l576_576266

-- (1) Prove k's range for real roots
theorem range_of_k_for_real_roots (k : ℝ) : 
  (∀ x : ℝ, (k-2) * x^2 - 2 * x + 1 = 0) → (k ≤ 3 ∧ k ≠ 2) :=
by
  sorry

-- (2) Prove roots when k is max integer value in the range
theorem roots_when_k_max_integer : 
  (x : ℝ) (h : (3-2) * x^2 - 2 * x + 1 = 0) → x = 1 :=
by
  sorry

end range_of_k_for_real_roots_roots_when_k_max_integer_l576_576266


namespace min_troublemakers_l576_576494

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l576_576494


namespace area_of_rectangle_l576_576155

--- Define the problem's conditions
def square_area : ℕ := 36
def rectangle_width := (square_side : ℕ) (h : square_area = square_side * square_side) : ℕ := square_side
def rectangle_length := (width : ℕ) : ℕ := 3 * width

--- State the theorem using the defined conditions
theorem area_of_rectangle (square_side : ℕ) 
  (h1 : square_area = square_side * square_side)
  (width := rectangle_width square_side h1)
  (length := rectangle_length width) :
  width * length = 108 := by
    sorry

end area_of_rectangle_l576_576155


namespace smallest_trees_in_three_types_l576_576779

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end smallest_trees_in_three_types_l576_576779


namespace measure_of_largest_angle_proof_l576_576125

-- defining the sum of interior angles of a pentagon
def sum_of_interior_angles_of_pentagon := 540

-- defining the angles as functions of x
def angle1 (x : ℚ) := x + 2
def angle2 (x : ℚ) := 2 * x + 3
def angle3 (x : ℚ) := 3 * x
def angle4 (x : ℚ) := 4 * x - 4
def angle5 (x : ℚ) := 5 * x - 2

-- x that satisfies the sum condition
def x_value := 541 // 15

-- defining the largest angle
def largest_angle (x : ℚ) := 5 * x - 2

theorem measure_of_largest_angle_proof (x : ℚ) (h : (angle1 x) + (angle2 x) + (angle3 x) + (angle4 x) + (angle5 x) = sum_of_interior_angles_of_pentagon) : 
  largest_angle x = 178 + 1/3 := 
  sorry

end measure_of_largest_angle_proof_l576_576125


namespace cot_inequality_l576_576257

theorem cot_inequality (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2)
  (h : cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 = 1) : 
  cot β * cot γ + cot γ * cot α + cot α * cot β ≤ 3 / 2 :=
sorry

end cot_inequality_l576_576257


namespace transformed_z_correct_l576_576086

-- Define complex number
def z : ℂ := -3 - 8 * Complex.I

-- Define the rotation by 45 degrees counter-clockwise using cis (cos + i sin)
def rotation_factor : ℂ := Complex.exp(Complex.I * Real.pi / 4)

-- Define the dilation scaling factor
def dilation_factor : ℂ := Real.sqrt 2

-- Combine the rotation and dilation as one operation
def transformation : ℂ := rotation_factor * dilation_factor

-- Apply the transformation to the complex number
def transformed_z : ℂ := z * transformation

-- The theorem we want to prove
theorem transformed_z_correct: transformed_z = 5 - 11 * Complex.I :=
by
  sorry

end transformed_z_correct_l576_576086


namespace joe_saves_6000_l576_576344

-- Definitions based on the conditions
def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000
def money_left : ℕ := 1000

-- Total expenses
def total_expenses : ℕ := flight_cost + hotel_cost + food_cost

-- Total savings
def total_savings : ℕ := total_expenses + money_left

-- The proof statement
theorem joe_saves_6000 : total_savings = 6000 := by
  -- Proof goes here
  sorry

end joe_saves_6000_l576_576344


namespace time_to_cross_platform_l576_576574

-- Definitions based on the given conditions
def train_length : ℝ := 300
def time_to_cross_pole : ℝ := 18
def platform_length : ℝ := 350

-- The question reformulated as a theorem in Lean 4
theorem time_to_cross_platform 
  (l_train : ℝ := train_length)
  (t_pole_cross : ℝ := time_to_cross_pole)
  (l_platform : ℝ := platform_length) :
  (l_train / t_pole_cross * (l_train + l_platform) = 39) :=
sorry

end time_to_cross_platform_l576_576574


namespace imaginary_part_abcd_zero_l576_576371

open Complex

theorem imaginary_part_abcd_zero (a b c d : ℂ)
  (h1 : ∃ θ : ℝ, (a / ∥a∥) = exp (I * θ) * (b / ∥b∥) ∧ (c / ∥c∥) = exp (I * θ) * (d / ∥d∥)) :
  im (a * b * c * d) = 0 :=
by
  sorry

end imaginary_part_abcd_zero_l576_576371


namespace min_troublemakers_29_l576_576500

noncomputable def min_troublemakers (n : ℕ) : ℕ :=
sorry

theorem min_troublemakers_29 : min_troublemakers 29 = 10 := 
sorry

end min_troublemakers_29_l576_576500


namespace smallest_trees_in_three_types_l576_576774

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end smallest_trees_in_three_types_l576_576774


namespace sum_of_digits_11_pow_2017_mod_9_l576_576365

theorem sum_of_digits_11_pow_2017_mod_9 : 
  (nat.sum_of_digits (11 ^ 2017)) % 9 = 2 :=
sorry

end sum_of_digits_11_pow_2017_mod_9_l576_576365


namespace ellipse_hyperbola_foci_l576_576431

theorem ellipse_hyperbola_foci (a b : ℝ) 
    (h1 : b^2 - a^2 = 25) 
    (h2 : a^2 + b^2 = 49) : 
    |a * b| = 2 * Real.sqrt 111 := 
by 
  -- proof omitted 
  sorry

end ellipse_hyperbola_foci_l576_576431


namespace larger_number_is_165_l576_576434

-- Define the conditions
def hcf : ℕ := 15
def factor1 : ℕ := 11
def factor2 : ℕ := 15
def lcm := hcf * factor1 * factor2

-- Define the number A as the larger number and B as the smaller number
def is_larger (A B : ℕ) := A ≤ B → false

-- State the theorem: Given the conditions, the larger number is 165
theorem larger_number_is_165 (A B : ℕ) (hcf_ab : hcf = 15) (factor1_ab : factor1 = 11) (factor2_ab : factor2 = 15)
  (lcm_ab : lcm = 15 * 11 * 15) (product : A * B = lcm * hcf) : is_larger A B → A = 165 :=
by sorry

end larger_number_is_165_l576_576434


namespace evaluate_expression_l576_576655

noncomputable def a_seq : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 3
| (n + 1) := a_seq (n) * a_seq (n - 1) * a_seq (n - 2) - 1

noncomputable def b_seq : ℕ → ℕ
| 1 := 1
| 2 := 4
| 3 := 9
| (n + 1) := b_seq (n) + 3 * (n + 1) - 1

def product_a (n : ℕ) : ℕ :=
(list.range n).map (λ i, a_seq (i + 1)).prod

def sum_b (n : ℕ) : ℕ :=
(list.range n).map (λ i, b_seq (i + 1)).sum

theorem evaluate_expression : 
  product_a 100 - sum_b 100 = exact_value_based_on_recursive_evaluation :=
sorry

end evaluate_expression_l576_576655


namespace y1_y2_positive_l576_576247

theorem y1_y2_positive 
  (x1 x2 x3 : ℝ)
  (y1 y2 y3 : ℝ)
  (h_line1 : y1 = -2 * x1 + 3)
  (h_line2 : y2 = -2 * x2 + 3)
  (h_line3 : y3 = -2 * x3 + 3)
  (h_order : x1 < x2 ∧ x2 < x3)
  (h_product_neg : x2 * x3 < 0) :
  y1 * y2 > 0 :=
by
  sorry

end y1_y2_positive_l576_576247


namespace case_m_eq_neg_1_case_m_gt_neg_1_case_m_lt_neg_1_l576_576221

noncomputable def solution_set (m x : ℝ) : Prop :=
  x^2 + (m-1) * x - m > 0

theorem case_m_eq_neg_1 (x : ℝ) :
  solution_set (-1) x ↔ x ≠ 1 :=
sorry

theorem case_m_gt_neg_1 (m x : ℝ) (hm : m > -1) :
  solution_set m x ↔ (x < -m ∨ x > 1) :=
sorry

theorem case_m_lt_neg_1 (m x : ℝ) (hm : m < -1) :
  solution_set m x ↔ (x < 1 ∨ x > -m) :=
sorry

end case_m_eq_neg_1_case_m_gt_neg_1_case_m_lt_neg_1_l576_576221


namespace logarithmic_equation_solution_l576_576571

noncomputable def satisfies_logarithmic_equation (y : ℝ) : Prop :=
  ∃ y, y > 1 ∧ (\(log (y - 1) - log y = log (2 * y - 2) - log (y + 2)))

theorem logarithmic_equation_solution :
  satisfies_logarithmic_equation 2 :=
by
  sorry

end logarithmic_equation_solution_l576_576571


namespace pancakes_left_l576_576628

def pancakes_per_batch := 21
def batches_made := 3
def doubled_batch := 2
def bobby_ate := 5
def dog_ate := 7
def friends_ate := 22

theorem pancakes_left : 
  let total_pancakes := pancakes_per_batch * 2 * doubled_batch + pancakes_per_batch * (batches_made - doubled_batch + 1) in
  let total_eaten := bobby_ate + dog_ate + friends_ate in
  total_pancakes - total_eaten = 50 :=
  sorry

end pancakes_left_l576_576628


namespace sum_of_integer_solutions_l576_576904

noncomputable def absolute_value_inequality (x : ℝ) : Prop :=
  12 * (|x + 10| - |x - 20|) / (|4 * x - 25| - |4 * x - 15|) -
  (|x + 10| + |x - 20|) / (|4 * x - 25| + |4 * x - 15|) ≥ -6

theorem sum_of_integer_solutions :
  (∑ x in (Finset.filter (λ x, |x| < 100) (Finset.range 201)).filter (λ x, absolute_value_inequality x), x) = 4 :=
by
  sorry

end sum_of_integer_solutions_l576_576904


namespace increased_hypotenuse_length_l576_576839

theorem increased_hypotenuse_length :
  let AB := 24
  let BC := 10
  let AB' := AB + 6
  let BC' := BC + 6
  let AC := Real.sqrt (AB^2 + BC^2)
  let AC' := Real.sqrt (AB'^2 + BC'^2)
  AC' - AC = 8 := by
  sorry

end increased_hypotenuse_length_l576_576839


namespace max_distinct_dance_counts_l576_576116

theorem max_distinct_dance_counts (B G : ℕ) (hB : B = 29) (hG : G = 15) 
  (dance_with : ℕ → ℕ → Prop)
  (h_dance_limit : ∀ b g, dance_with b g → b ≤ B ∧ g ≤ G) :
  ∃ max_counts : ℕ, max_counts = 29 :=
by
  -- The statement of the theorem. Proof is omitted.
  sorry

end max_distinct_dance_counts_l576_576116


namespace distance_between_foci_hyperbola_l576_576675

theorem distance_between_foci_hyperbola :
  (let a^2 := 50 in
   let b^2 := 8 in
   let c^2 := a^2 + b^2 in
   2 * Real.sqrt c^2 = 2 * Real.sqrt 58) :=
by
  sorry

end distance_between_foci_hyperbola_l576_576675


namespace three_types_in_69_trees_l576_576785

variable (birches spruces pines aspens : ℕ)
variable (total_trees : ℕ := 100)
variable (all_trees : list (string × ℕ))

-- We assert that there are 100 trees in total, and our list of trees represents this
axiom h_total : ∑ t in all_trees, t.2 = total_trees

-- Among any 85 trees, there must be at least one of each type
axiom h_85_trees_all_types : ∀ (s : list (string × ℕ)), s.card = 85 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0 ∧ a > 0 ∧ 
  b + s + p + a = 85 ∧ (("birches", b) ∈ s ∧ ("spruces", s) ∈ s ∧ ("pines", p) ∈ s ∧ ("aspens", a) ∈ s))

-- We need to prove that any subset of 69 or more trees contains at least three different types.
theorem three_types_in_69_trees :
  ∀ (s : list (string × ℕ)), s.card = 69 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0) ∨ (b > 0 ∧ s > 0 ∧ a > 0) ∨ (b > 0 ∧ p > 0 ∧ a > 0) ∨ (s > 0 ∧ p > 0 ∧ a > 0) :=
begin
  sorry
end

end three_types_in_69_trees_l576_576785


namespace second_digit_of_n_l576_576059

theorem second_digit_of_n (n : ℕ) :
  (10^99 ≤ 8 * n ∧ 8 * n < 10^100) ∧ (10^101 ≤ 81 * n ∧ 81 * n < 10^102) →
  ∃ d : ℕ, d = 2 ∧ second_digit (n : ℕ) = d :=
by
  intros h
  sorry

def second_digit (n : ℕ) : ℕ := 
  -- This function definition is just a placeholder. You need to implement it.
  sorry

end second_digit_of_n_l576_576059


namespace least_positive_three_digit_multiple_of_9_l576_576531

   theorem least_positive_three_digit_multiple_of_9 : ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ 9 ∣ n ∧ n = 108 :=
   by
     sorry
   
end least_positive_three_digit_multiple_of_9_l576_576531


namespace angle_AF_CE_angle_CD_plane_BCD_l576_576831

variable {a : ℝ}

-- Conditions
def is_midpoint (p1 p2 midpoint : ℝ×ℝ×ℝ) : Prop :=
  midpoint = ((p1 + p2) / 2)

def lengths_equal (a b c d e : ℝ) : Prop := a = b ∧ a = c ∧ a = d ∧ a = e 

-- Problem (I): Angle between lines AF and CE
theorem angle_AF_CE (A B C D E F : ℝ×ℝ×ℝ) 
  (H1 : is_midpoint A D E) (H2 : is_midpoint B C F)
  (H3 : lengths_equal (dist A B) (dist B C) (dist C D) (dist D A) a) 
  : 
  ∃ θ, θ = arccos (2 / 3) := sorry

-- Problem (II): Angle between CD and plane BCD
theorem angle_CD_plane_BCD (A B C D E F : ℝ×ℝ×ℝ) 
  (H1 : is_midpoint A D E) (H2 : is_midpoint B C F)
  (H3 : lengths_equal (dist A B) (dist B C) (dist C D) (dist D A) a) 
  : 
  ∃ φ, φ = arcsin ((sqrt 2) / 3) := sorry

end angle_AF_CE_angle_CD_plane_BCD_l576_576831


namespace num_pairs_satisfying_eq_l576_576437

theorem num_pairs_satisfying_eq : 
  ∃ (n : ℕ), n = 3 ∧ (∀ (a b : ℕ), (0 ≤ a ∧ 0 ≤ b) → (|a - b| + a * b = 1) → ((a,b) = (0,1) ∨ (a,b) = (1,0) ∨ (a,b) = (1,1))) := by
{
  -- Here we need to prove the existence and uniqueness of 3 such pairs
  sorry
}

end num_pairs_satisfying_eq_l576_576437


namespace maximize_revenue_l576_576121

theorem maximize_revenue (p : ℝ) (hp : p ≤ 30) :
  (p = 12 ∨ p = 13) → (∀ p : ℤ, p ≤ 30 → 200 * p - 8 * p * p ≤ 1248) :=
by
  intros h1 h2
  sorry

end maximize_revenue_l576_576121


namespace max_volume_is_correct_l576_576333

noncomputable def max_volume_of_inscribed_sphere (AB BC AA₁ : ℝ) (h₁ : AB = 6) (h₂ : BC = 8) (h₃ : AA₁ = 3) : ℝ :=
  let AC := Real.sqrt ((6 : ℝ) ^ 2 + (8 : ℝ) ^ 2)
  let r := (AB + BC - AC) / 2
  let sphere_radius := AA₁ / 2
  (4/3) * Real.pi * sphere_radius ^ 3

theorem max_volume_is_correct : max_volume_of_inscribed_sphere 6 8 3 (by rfl) (by rfl) (by rfl) = 9 * Real.pi / 2 := by
  sorry

end max_volume_is_correct_l576_576333


namespace find_f_84_l576_576596

def f : ℤ → ℤ
| n := if n >= 1000 then n - 3 else f (f (n + 5))

theorem find_f_84 : f 84 = 997 :=
by
  simp [f]
  sorry

end find_f_84_l576_576596


namespace minimum_trees_l576_576794

variable (Trees : Type) [Fintype Trees] [DecidableEq Trees]

def trees_in_grove : Nat := 100

def tree_type := {birches, spruces, pines, aspens} : Set Trees

def condition (s : Finset Trees) : Prop := 
  s.card > 85 → tree_type ⊆ s

theorem minimum_trees (s : Finset Trees) (H : condition s) : 
  ∃ (n : Nat), n ≤ trees_in_grove ∧ n ≥ 69 → 
  ∃ t ⊆ s, t.card = n ∧ (|t ∩ tree_type| >= 3) :=
sorry

end minimum_trees_l576_576794


namespace B_and_D_know_their_grades_l576_576164

-- Define the students and their respective grades
inductive Grade : Type
| excellent : Grade
| good : Grade

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define the information given in the problem regarding which student sees whose grade
def sees (s1 s2 : Student) : Prop :=
  (s1 = Student.A ∧ (s2 = Student.B ∨ s2 = Student.C)) ∨
  (s1 = Student.B ∧ s2 = Student.C) ∨
  (s1 = Student.D ∧ s2 = Student.A)

-- Define the condition that there are 2 excellent and 2 good grades
def grade_distribution (gA gB gC gD : Grade) : Prop :=
  gA ≠ gB → (gC = gA ∨ gC = gB) ∧ (gD = gA ∨ gD = gB) ∧
  (gA = Grade.excellent ∧ (gB = Grade.good ∨ gC = Grade.good ∨ gD = Grade.good)) ∧
  (gA = Grade.good ∧ (gB = Grade.excellent ∨ gC = Grade.excellent ∨ gD = Grade.excellent))

-- Student A's statement after seeing B and C's grades
def A_statement (gA gB gC : Grade) : Prop :=
  (gB = gA ∨ gC = gA) ∨ (gB ≠ gA ∧ gC ≠ gA)

-- Formal proof goal: Prove that B and D can know their own grades based on the information provided
theorem B_and_D_know_their_grades (gA gB gC gD : Grade)
  (h1 : grade_distribution gA gB gC gD)
  (h2 : A_statement gA gB gC)
  (h3 : sees Student.A Student.B)
  (h4 : sees Student.A Student.C)
  (h5 : sees Student.B Student.C)
  (h6 : sees Student.D Student.A) :
  (gB ≠ Grade.excellent ∨ gB ≠ Grade.good) ∧ (gD ≠ Grade.excellent ∨ gD ≠ Grade.good) :=
by sorry

end B_and_D_know_their_grades_l576_576164


namespace range_of_t_and_min_c_l576_576741

noncomputable def parabola_line_intersect (y_parabola y_line: ℝ → ℝ): Prop :=
  ∃ x1 x2 t c, 
    y_parabola x1 = x1 ^ 2 ∧ 
    y_line x1 = (2 * t - 1) * x1 - c ∧ 
    y_parabola x2 = x2 ^ 2 ∧ 
    y_line x2 = (2 * t - 1) * x2 - c ∧ 
    x1^2 + x2^2 = t^2 + 2 * t - 3

theorem range_of_t_and_min_c : 
  (∀ (y_parabola y_line : ℝ → ℝ), parabola_line_intersect y_parabola y_line → 
    ∃ t c_min,
      (2 - real.sqrt 2 ≤ t ∧ t ≤ 2 + real.sqrt 2) ∧
      (t = 2 - real.sqrt 2 → c_min = (11 - 6 * real.sqrt 2) / 4)) :=
sorry

end range_of_t_and_min_c_l576_576741


namespace triangle_side_length_l576_576338

variable (A C : ℝ) (a c b : ℝ)

theorem triangle_side_length (h1 : c = 48) (h2 : a = 27) (h3 : C = 3 * A) : b = 35 := by
  sorry

end triangle_side_length_l576_576338


namespace dice_probability_l576_576547

theorem dice_probability :
  ∃ (p : ℚ), 
    (∀ (die1 die2 : ℕ), die1 ∈ finset.Icc 1 6 → 
                        die2 ∈ finset.Icc 1 6 → 
                        die1 ≠ die2 → 
                        (die1 = 3 ∨ die2 = 3) → 
                        p = 1/3) :=
begin
  sorry
end

end dice_probability_l576_576547


namespace remainder_mod7_eq_l576_576540

theorem remainder_mod7_eq :
  (9^4 + 8^5 + 7^6) % 7 = 3 :=
by
  have h9 : 9 % 7 = 2 := by norm_num
  have h8 : 8 % 7 = 1 := by norm_num
  have h7 : 7 % 7 = 0 := by norm_num
  have h2 : (2^4) % 7 = 2 := by norm_num
  have h1 : (1^5) % 7 = 1 := by norm_num
  have h0 : (0^6) % 7 = 0 := by norm_num
  calc
    (9^4 + 8^5 + 7^6) % 7
        = ((9 % 7)^4 + (8 % 7)^5 + (7 % 7)^6) % 7 := by congr; apply h9; apply h8; apply h7
    ... = (2^4 + 1^5 + 0^6) % 7 := by congr; apply pow_congr h9; apply pow_congr h8; apply pow_congr h7
    ... = (2 + 1 + 0) % 7 := by congr; apply h2; apply h1; apply h0
    ... = 3 % 7 := by norm_num
    ... = 3 := by norm_num

end remainder_mod7_eq_l576_576540


namespace min_trees_for_three_types_l576_576804

-- Define types and trees
inductive TreeType
| birch | spruce | pine | aspen
deriving Inhabited, DecidableEq

-- A grove with 100 trees of any of the four types.
structure Tree :=
(type : TreeType)

constant grove : List Tree
axiom grove_size : grove.length = 100

-- Condition: Among any 85 trees, there are trees of all four types.
axiom all_types_in_any_85 : ∀ (s : Finset Tree), s.card = 85 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, ∃ t4 ∈ s, 
      t1.type = TreeType.birch ∧
      t2.type = TreeType.spruce ∧
      t3.type = TreeType.pine ∧
      t4.type = TreeType.aspen

-- We need to show that at least 69 trees are needed to ensure at least 3 types.
theorem min_trees_for_three_types : 
  ∀ (s : Finset Tree), s.card = 69 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, 
      t1.type ≠ t2.type ∧ t2.type ≠ t3.type ∧ t1.type ≠ t3.type := 
sorry

end min_trees_for_three_types_l576_576804


namespace three_types_in_69_trees_l576_576783

variable (birches spruces pines aspens : ℕ)
variable (total_trees : ℕ := 100)
variable (all_trees : list (string × ℕ))

-- We assert that there are 100 trees in total, and our list of trees represents this
axiom h_total : ∑ t in all_trees, t.2 = total_trees

-- Among any 85 trees, there must be at least one of each type
axiom h_85_trees_all_types : ∀ (s : list (string × ℕ)), s.card = 85 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0 ∧ a > 0 ∧ 
  b + s + p + a = 85 ∧ (("birches", b) ∈ s ∧ ("spruces", s) ∈ s ∧ ("pines", p) ∈ s ∧ ("aspens", a) ∈ s))

-- We need to prove that any subset of 69 or more trees contains at least three different types.
theorem three_types_in_69_trees :
  ∀ (s : list (string × ℕ)), s.card = 69 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0) ∨ (b > 0 ∧ s > 0 ∧ a > 0) ∨ (b > 0 ∧ p > 0 ∧ a > 0) ∨ (s > 0 ∧ p > 0 ∧ a > 0) :=
begin
  sorry
end

end three_types_in_69_trees_l576_576783


namespace min_trees_for_three_types_l576_576803

-- Define types and trees
inductive TreeType
| birch | spruce | pine | aspen
deriving Inhabited, DecidableEq

-- A grove with 100 trees of any of the four types.
structure Tree :=
(type : TreeType)

constant grove : List Tree
axiom grove_size : grove.length = 100

-- Condition: Among any 85 trees, there are trees of all four types.
axiom all_types_in_any_85 : ∀ (s : Finset Tree), s.card = 85 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, ∃ t4 ∈ s, 
      t1.type = TreeType.birch ∧
      t2.type = TreeType.spruce ∧
      t3.type = TreeType.pine ∧
      t4.type = TreeType.aspen

-- We need to show that at least 69 trees are needed to ensure at least 3 types.
theorem min_trees_for_three_types : 
  ∀ (s : Finset Tree), s.card = 69 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, 
      t1.type ≠ t2.type ∧ t2.type ≠ t3.type ∧ t1.type ≠ t3.type := 
sorry

end min_trees_for_three_types_l576_576803


namespace find_k_zero_point_l576_576864

noncomputable def f (x : ℝ) := 2^x + x

theorem find_k_zero_point :
  (∃ x₀ : ℝ, f x₀ = 0 ∧ x₀ ∈ (k, k+1)) → (∃ k : ℤ, k = -1) :=
begin
  sorry
end

end find_k_zero_point_l576_576864


namespace three_digit_numbers_no_789_l576_576294

theorem three_digit_numbers_no_789 : 
  let hundreds_choices := 6 
      tens_choices := 7 
      units_choices := 7 
  in hundreds_choices * tens_choices * units_choices = 294 := 
by
  sorry

end three_digit_numbers_no_789_l576_576294


namespace robin_gum_packages_l576_576896

theorem robin_gum_packages (P : ℕ) (h1 : 7 * P + 6 = 41) : P = 5 :=
by
  sorry

end robin_gum_packages_l576_576896


namespace least_positive_three_digit_multiple_of_9_l576_576535

theorem least_positive_three_digit_multiple_of_9 : 
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ n % 9 = 0 ∧ ∀ m : ℕ, (m >= 100 ∧ m < 1000 ∧ m % 9 = 0) → n ≤ m :=
begin
  use 108,
  split,
  { exact nat.le_refl 108 },
  split,
  { exact nat.lt_of_lt_of_le (nat.succ_pos 9) (nat.succ_le_succ (nat.le_refl 99)) },
  split,
  { exact nat.mod_eq_zero_of_mk (nat.zero_of_succ_pos 12) },
  { intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm3 hm4,
    exact nat.le_of_eq ((nat.mod_eq_zero_of_dvd (nat.gcd_eq_gcd_ab (12) (8) (1)))),
  },
  sorry
end

end least_positive_three_digit_multiple_of_9_l576_576535


namespace sum_of_sides_l576_576980

theorem sum_of_sides (h : ℕ := 6) (t : ℕ := 3) (q : ℕ := 4) : h + t + q = 13 := 
by
  -- hexagon has 6 sides
  -- triangle has 3 sides
  -- quadrilateral has 4 sides
  rfl

end sum_of_sides_l576_576980


namespace ratio_u_v_l576_576588

variables {u v : ℝ}
variables (u_lt_v : u < v)
variables (h_triangle : triangle 15 12 9)
variables (inscribed_circle : is_inscribed_circle 15 12 9 u v)

theorem ratio_u_v : u / v = 1 / 2 :=
sorry

end ratio_u_v_l576_576588


namespace f_value_at_5_l576_576714

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 / 2 then 2 * x^2 else sorry

theorem f_value_at_5 (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_periodic : periodic_function f 3)
  (h_definition : ∀ x, 0 ≤ x ∧ x ≤ 3 / 2 → f x = 2 * x^2) :
  f 5 = 2 :=
by
  sorry

end f_value_at_5_l576_576714


namespace potatoes_cost_l576_576020

-- Definitions of variables
def N : ℕ := 3
def C : ℕ := 3
def T : ℕ := 15

-- Definition of the cost of potatoes
def P : ℕ := T - (N * C)

-- Theorem statement that specifies the value of P
theorem potatoes_cost : P = 6 := by
  -- Substitute the definitions and perform the calculation
  delta P N C T
  simp
  sorry

end potatoes_cost_l576_576020


namespace min_value_xlny_l576_576254

noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x

theorem min_value_xlny (x y : ℝ) (h1 : Real.log x y + Real.log y x = 5 / 2) (h2 : Real.log x y > 1) :
    (∃ x : ℝ, ∃ y : ℝ, f x = -2 / Real.exp 1) :=
by
  sorry

end min_value_xlny_l576_576254


namespace initial_distance_proof_l576_576965

noncomputable def initial_distance (V_A V_B T : ℝ) : ℝ :=
  (V_A * T) + (V_B * T)

theorem initial_distance_proof 
  (V_A V_B : ℝ) 
  (T : ℝ) 
  (h1 : V_A / V_B = 5 / 6)
  (h2 : V_B = 90)
  (h3 : T = 8 / 15) :
  initial_distance V_A V_B T = 88 := 
by
  -- proof goes here
  sorry

end initial_distance_proof_l576_576965


namespace cylinder_volume_triple_l576_576592

def volume_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

theorem cylinder_volume_triple (r h : ℝ) (r_B h_B: ℝ) :
  r = 5 → h = 12 →
  r_B = 15 → h_B = 4 →
  volume_cylinder r_B h_B = 3 * volume_cylinder r h :=
by 
  intros hr hh hrB hhB;
  rw [hr, hh, hrB, hhB];
  sorry

end cylinder_volume_triple_l576_576592


namespace asia_paid_140_l576_576173

noncomputable def original_price : ℝ := 350
noncomputable def discount_percentage : ℝ := 0.60
noncomputable def discount_amount : ℝ := original_price * discount_percentage
noncomputable def final_price : ℝ := original_price - discount_amount

theorem asia_paid_140 : final_price = 140 := by
  unfold final_price
  unfold discount_amount
  unfold original_price
  unfold discount_percentage
  sorry

end asia_paid_140_l576_576173


namespace min_troublemakers_l576_576507

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l576_576507


namespace f_geq_1_range_of_x_l576_576730

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem 1: Prove that f(x) >= 1 for all real x
theorem f_geq_1 (x : ℝ) : f x ≥ 1 := by
  sorry

-- Claim that f(x) is equal to the given expression
def g (a : ℝ) : ℝ := (a^2 + 2) / real.sqrt (a^2 + 1)

-- Theorem 2: If g(a) has a solution, find the range of x
theorem range_of_x (a : ℝ) (h : ∃ x, f x = g a) : 
  (a = 0) ∨ (∀ x : ℝ, x ∈ set.Iic (1/2) ∪ set.Ici (5/2)) := 
by 
  sorry

end f_geq_1_range_of_x_l576_576730


namespace fibonacci_factorial_sum_l576_576544

def factorial_last_two_digits(n: ℕ) : ℕ :=
  if n > 10 then 0 else 
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 24
  | 5 => 120 % 100
  | 6 => 720 % 100
  | 7 => 5040 % 100
  | 8 => 40320 % 100
  | 9 => 362880 % 100
  | 10 => 3628800 % 100
  | _ => 0

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 144]

noncomputable def sum_last_two_digits (l: List ℕ) : ℕ :=
  l.map factorial_last_two_digits |>.sum

theorem fibonacci_factorial_sum:
  sum_last_two_digits fibonacci_factorial_series = 50 := by
  sorry

end fibonacci_factorial_sum_l576_576544


namespace find_n_l576_576053

theorem find_n (n : ℕ) (hn : 9! / (9 - n)! = 504) : n = 3 := 
by
  sorry

end find_n_l576_576053


namespace right_triangle_congruence_two_legs_right_triangle_congruence_leg_angle_adjacent_l576_576115

-- (a) Criterion for Equality of Right Triangles "By Two Legs"
theorem right_triangle_congruence_two_legs
  (A B C A' B' C' : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space A'] [metric_space B'] [metric_space C']
  (triangle_ABC : triangle A B C) (triangle_A'B'C' : triangle A' B' C')
  (right_angle_B : ∠ B = 90) (right_angle_B' : ∠ B' = 90)
  (AB_eq : dist A B = dist A' B') (BC_eq : dist B C = dist B' C') :
  triangle ABC ≅ triangle A'B'C' :=
sorry

-- (b) Criterion for Equality of Right Triangles by a Leg and Adjacent Acute Angle
theorem right_triangle_congruence_leg_angle_adjacent
  (A B C A' B' C' : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space A'] [metric_space B'] [metric_space C']
  (triangle_ABC : triangle A B C) (triangle_A'B'C' : triangle A' B' C')
  (right_angle_B : ∠ B = 90) (right_angle_B' : ∠ B' = 90)
  (AB_eq : dist A B = dist A' B') (angle_BAC_eq : ∠ A = ∠ A') :
  triangle ABC ≅ triangle A'B'C' :=
sorry

end right_triangle_congruence_two_legs_right_triangle_congruence_leg_angle_adjacent_l576_576115


namespace min_people_like_mozart_bach_not_beethoven_l576_576391

-- Define the initial conditions
variables {n a b c : ℕ}
variables (total_people := 150)
variables (likes_mozart := 120)
variables (likes_bach := 105)
variables (likes_beethoven := 45)

theorem min_people_like_mozart_bach_not_beethoven : 
  ∃ (x : ℕ), 
    total_people = 150 ∧ 
    likes_mozart = 120 ∧ 
    likes_bach = 105 ∧ 
    likes_beethoven = 45 ∧ 
    x = (likes_mozart + likes_bach - total_people) := 
    sorry

end min_people_like_mozart_bach_not_beethoven_l576_576391


namespace number_of_pairs_l576_576528

noncomputable def total_ordered_pairs : ℕ := 198

theorem number_of_pairs:
  ∃ (a : ℝ) (b : ℕ), 
  (a > 0) ∧ (b % 2 = 0) ∧ (4 ≤ b ∧ b ≤ 200) ∧ ((Real.log a / Real.log b)^2 = Real.log (a^2) / Real.log b) ∧ 
  let total_pairs := 2 * (200 / 2 - 4 / 2 + 1) in
  total_pairs = total_ordered_pairs :=
sorry

end number_of_pairs_l576_576528


namespace parallelogram_area_l576_576995

theorem parallelogram_area (base height : ℝ) (h_base : base = 25) (h_height : height = 15) :
  base * height = 375 :=
by
  subst h_base
  subst h_height
  sorry

end parallelogram_area_l576_576995


namespace four_numbers_perfect_square_l576_576742

theorem four_numbers_perfect_square
  (N : Finset ℕ)
  (hN : N.card = 48)
  (h : ∃ p : Finset ℕ, p.card = 10 ∧ ∀ n ∈ N, ∀ q ∈ p, q.prime ∧ q ∣ n)
  : ∃ a b c d ∈ N, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ is_square (a * b * c * d) :=
by
  sorry

end four_numbers_perfect_square_l576_576742


namespace point_Q_converges_to_specific_triangle_l576_576024

variable {A B C : Point}
variable {S : ℝ}

def moves_in_specific_pattern (Q : Point) : Prop :=
  -- Define the specific motion pattern of Q as described in the problem
  sorry

theorem point_Q_converges_to_specific_triangle (area_ABC : ℝ) (h_area : area_ABC = S) :
  ∃ (T : Triangle), (tends_to Q T) ∧ (area T = area_ABC / 7) :=
by
  sorry

end point_Q_converges_to_specific_triangle_l576_576024


namespace min_liars_needed_l576_576470

namespace Problem

/-
  We introduce our problem using Lean definitions and statements.
-/

def students_sitting_at_table : ℕ := 29

structure Student :=
(is_liar : Bool)

def student_truth_statement (s: ℕ) (next1: Student) (next2: Student) : Bool :=
  if s then (next1.is_liar && !next2.is_liar) || (!next1.is_liar && next2.is_liar)
  else next1.is_liar && next2.is_liar

def minimum_liars_class (sitting_students: ℕ) : ℕ :=
  sitting_students / 3 + (if sitting_students % 3 = 0 then 0 else 1)

theorem min_liars_needed (students : List Student) (H_total : students.length = students_sitting_at_table)
  (H_truth_teller_claim : ∀ (i : ℕ), i < students.length → student_truth_statement i (students.get ⟨(i+1) % students.length, sorry⟩) (students.get ⟨(i+2) % students.length, sorry⟩)) :
  (students.count (λ s => s.is_liar)) ≥ minimum_liars_class students_sitting_at_table :=
sorry

end Problem

end min_liars_needed_l576_576470


namespace smallest_x_for_cubic_1890_l576_576543

theorem smallest_x_for_cubic_1890 (x : ℕ) (N : ℕ) (hx : 1890 * x = N ^ 3) : x = 4900 :=
sorry

end smallest_x_for_cubic_1890_l576_576543


namespace variance_poisson_l576_576225

variables {λ : ℝ} {X : ℕ → ℝ}

-- Definition for Poisson PMF
def poisson_pmf (k : ℕ) : ℝ := (λ^k * real.exp (-λ)) / (nat.factorial k)

-- Expected value of a Poisson-distributed random variable with parameter λ
def expected_value (f : ℕ → ℝ) : ℝ := ∑' k, f k * poisson_pmf k

-- Variance of a Poisson-distributed random variable
def variance (X : ℕ → ℝ) : ℝ := expected_value (λ x, x^2) - (expected_value (λ x, x))^2

-- The goal: Proving the variance of a Poisson random variable is λ
theorem variance_poisson (λ : ℝ) (h : 0 ≤ λ) : variance id = λ := 
by 
  sorry

end variance_poisson_l576_576225


namespace min_trees_for_three_types_l576_576805

-- Define types and trees
inductive TreeType
| birch | spruce | pine | aspen
deriving Inhabited, DecidableEq

-- A grove with 100 trees of any of the four types.
structure Tree :=
(type : TreeType)

constant grove : List Tree
axiom grove_size : grove.length = 100

-- Condition: Among any 85 trees, there are trees of all four types.
axiom all_types_in_any_85 : ∀ (s : Finset Tree), s.card = 85 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, ∃ t4 ∈ s, 
      t1.type = TreeType.birch ∧
      t2.type = TreeType.spruce ∧
      t3.type = TreeType.pine ∧
      t4.type = TreeType.aspen

-- We need to show that at least 69 trees are needed to ensure at least 3 types.
theorem min_trees_for_three_types : 
  ∀ (s : Finset Tree), s.card = 69 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, 
      t1.type ≠ t2.type ∧ t2.type ≠ t3.type ∧ t1.type ≠ t3.type := 
sorry

end min_trees_for_three_types_l576_576805


namespace least_positive_three_digit_multiple_of_9_l576_576532

   theorem least_positive_three_digit_multiple_of_9 : ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ 9 ∣ n ∧ n = 108 :=
   by
     sorry
   
end least_positive_three_digit_multiple_of_9_l576_576532


namespace mass_percentage_Cr_in_dichromate_is_48_15_l576_576208

-- Definition of the molar masses
def molar_mass_Cr : ℝ := 52.00
def molar_mass_O : ℝ := 16.00

-- Definition of the mass percentage function
def mass_percentage_of_Cr_in_dichromate (molar_mass_Cr molar_mass_O : ℝ) : ℝ :=
  let total_mass_dichromate := (2 * molar_mass_Cr) + (7 * molar_mass_O)
  let total_mass_Cr := 2 * molar_mass_Cr
  (total_mass_Cr / total_mass_dichromate) * 100

-- Theorem stating the final answer
theorem mass_percentage_Cr_in_dichromate_is_48_15 :
  mass_percentage_of_Cr_in_dichromate molar_mass_Cr molar_mass_O = 48.15 := 
  by
  -- Skipping the proof since it's not required as per instructions
  sorry

end mass_percentage_Cr_in_dichromate_is_48_15_l576_576208


namespace largest_solution_l576_576679

theorem largest_solution (h₁ : (abs (2 * sin x - 1) + abs (2 * cos (2 * x) - 1)) = 0)
  (h₂ : 0 ≤ x) (h₃ : x ≤ 10 * Real.pi) : 
  x ≈ 27.7 :=
sorry

end largest_solution_l576_576679


namespace grove_tree_selection_l576_576828

theorem grove_tree_selection (birches spruces pines aspens : ℕ) :
  birches + spruces + pines + aspens = 100 →
  (∀ s : set ℕ, s.card = 85 → (birches ∈ s ∧ spruces ∈ s ∧ pines ∈ s ∧ aspens ∈ s)) →
  ∀ t : set ℕ, t.card = 69 → (birches ∈ t ∧ spruces ∈ t) ∨ (birches ∈ t ∧ pines ∈ t) ∨ (birches ∈ t ∧ aspens ∈ t) ∨ (spruces ∈ t ∧ pines ∈ t) ∨ (spruces ∈ t ∧ aspens ∈ t) ∨ (pines ∈ t ∧ aspens ∈ t) :=
sorry

end grove_tree_selection_l576_576828


namespace suraj_new_average_is_42_l576_576999

noncomputable def suraj_average_after_ninth_inning (A : ℝ) : ℝ :=
  let total_runs_first_8_innings : ℝ := 8 * A
  let total_runs_after_9th_inning : ℝ := total_runs_first_8_innings + 90
  let new_average : ℝ := A + 6
  9 * new_average = total_runs_after_9th_inning

theorem suraj_new_average_is_42 (A : ℝ) (h : suraj_average_after_ninth_inning A) : A + 6 = 42 := by
  sorry

end suraj_new_average_is_42_l576_576999


namespace min_trees_include_three_types_l576_576816

noncomputable def minNumTrees (T : Type) (tree_counts : T → ℕ) :=
  ∀ (total_trees : ℕ) (at_least : ℕ → Prop),
    total_trees = 100 →
    (∀ S : Finset T, S.card = 85 → (∃ t ∈ S, at_least (tree_counts t))) →
    (at_least ThreeTypes ↔ ∃ (S : Finset T), S.card = 69)

def ThreeTypes (tree_counts : T → ℕ) := 
  ∀ (birches spruces pines aspens : ℕ),
    birches + spruces + pines + aspens = 100 →
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ pines ≥ 1) ∨ 
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ aspens ≥ 1) ∨ 
    (birches ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1) ∨ 
    (spruces ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1)

theorem min_trees_include_three_types :
  ∃ (T : Type) (tree_counts : T → ℕ), minNumTrees T tree_counts := 
sorry

end min_trees_include_three_types_l576_576816


namespace proof_inequality_l576_576249

noncomputable def problem (a b c d : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1 → a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d

theorem proof_inequality (a b c d : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end proof_inequality_l576_576249


namespace rose_can_afford_l576_576403

noncomputable def total_cost_before_discount : ℝ :=
  2.40 + 9.20 + 6.50 + 12.25 + 4.75

noncomputable def discount : ℝ :=
  0.15 * total_cost_before_discount

noncomputable def total_cost_after_discount : ℝ :=
  total_cost_before_discount - discount

noncomputable def budget : ℝ :=
  30.00

noncomputable def remaining_budget : ℝ :=
  budget - total_cost_after_discount

theorem rose_can_afford :
  remaining_budget = 0.165 :=
by
  -- proof goes here
  sorry

end rose_can_afford_l576_576403


namespace original_amount_of_water_l576_576597

variable {W : ℝ} -- Assume W is a real number representing the original amount of water

theorem original_amount_of_water (h1 : 30 * 0.02 = 0.6) (h2 : 0.6 = 0.06 * W) : W = 10 :=
by
  sorry

end original_amount_of_water_l576_576597


namespace original_proposition_true_converse_proposition_false_inverse_proposition_false_contrapositive_proposition_true_count_true_propositions_l576_576282

theorem original_proposition_true (a b : ℝ) (h : a > b ∧ b > 0) : log (1/2) a < log (1/2) b + 1 :=
sorry

theorem converse_proposition_false (a b : ℝ) (h : log (1/2) a < log (1/2) b + 1) : ¬ (a > b ∧ b > 0) :=
sorry

theorem inverse_proposition_false (a b : ℝ) (h : ¬ (a > b ∧ b > 0)) : ¬ (log (1/2) a < log (1/2) b + 1) :=
sorry

theorem contrapositive_proposition_true (a b : ℝ) (h : ¬ (log (1/2) a < log (1/2) b + 1)) : ¬ (a > b ∧ b > 0) :=
sorry

theorem count_true_propositions : 2 =
if original_proposition_true_2 → contrapositive_proposition_true_2 then 2 else 0 :=
sorry

end original_proposition_true_converse_proposition_false_inverse_proposition_false_contrapositive_proposition_true_count_true_propositions_l576_576282


namespace problem_statement_l576_576659

theorem problem_statement (x : ℝ) (a b c : ℕ) (h : a * b * c ≠ 0) 
  (h_eq : cos(x)^2 + cos(2 * x)^2 + cos(3 * x)^2 + cos(4 * x)^2 = 2) :
  (sin(a * x) * sin(b * x) * sin(c * x) = 0) ∧ (a + b + c = 8) :=
sorry

end problem_statement_l576_576659


namespace no_solution_inequality_l576_576768

theorem no_solution_inequality (m x : ℝ) (h1 : x - 2 * m < 0) (h2 : x + m > 2) : m ≤ 2 / 3 :=
  sorry

end no_solution_inequality_l576_576768


namespace jill_spent_50_percent_on_clothing_l576_576887

theorem jill_spent_50_percent_on_clothing (
  T : ℝ) (hT : T ≠ 0)
  (h : 0.05 * T * C + 0.10 * 0.30 * T = 0.055 * T):
  C = 0.5 :=
by
  sorry

end jill_spent_50_percent_on_clothing_l576_576887


namespace minimum_trees_with_at_least_three_types_l576_576809

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end minimum_trees_with_at_least_three_types_l576_576809


namespace eccentricity_is_sqrt_2_l576_576916

-- Define the condition of the hyperbola
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / 9) = 1

-- Define the constant a squared and b squared from the given equation
def a_squared : ℝ := 9
def b_squared : ℝ := 9

-- Define the resulting c squared from the formula c^2 = a^2 + b^2
def c_squared : ℝ := a_squared + b_squared

-- Define the eccentricity e
def eccentricity : ℝ := Real.sqrt c_squared / Real.sqrt a_squared

-- Prove that the eccentricity is indeed √2
theorem eccentricity_is_sqrt_2 : eccentricity = Real.sqrt 2 :=
by
  have h1 : c_squared = 18 := by sorry
  have h2 : Real.sqrt c_squared = Real.sqrt 18 := by sorry
  have h3 : Real.sqrt a_squared = 3 := by sorry
  have h4 : eccentricity = Real.sqrt 18 / 3 := by sorry
  have h5 : Real.sqrt 18 = Real.sqrt (9 * 2) := by sorry
  have h6 : Real.sqrt (9 * 2) = 3 * Real.sqrt 2 := by sorry
  have h7 : Real.sqrt 18 / 3 = (3 * Real.sqrt 2) / 3 := by sorry
  have h8 : (3 * Real.sqrt 2) / 3 = Real.sqrt 2 := by sorry
  exact h8

end eccentricity_is_sqrt_2_l576_576916


namespace solve_for_x_l576_576223

theorem solve_for_x : ∃ x : ℚ, (x = -21/4) ∧ ((sqrt (4 * x + 7)) / (sqrt (8 * x + 10)) = sqrt 7 / 4) :=
by
  use -21/4
  sorry

end solve_for_x_l576_576223


namespace cube_root_1728000_l576_576040

theorem cube_root_1728000 :
  (∛1728000 : ℝ) = 120 := 
by
  sorry

end cube_root_1728000_l576_576040


namespace students_without_glasses_l576_576953

theorem students_without_glasses (total_students: ℕ) (percentage_with_glasses: ℕ) (p: percentage_with_glasses = 40) (t: total_students = 325) : ∃ x : ℕ, x = (total_students * (100 - percentage_with_glasses)) / 100 ∧ x = 195 :=
by
  have total_students := 325
  have percentage_with_glasses := 40
  have percentage_without_glasses := 100 - percentage_with_glasses
  have number_without_glasses := (total_students * percentage_without_glasses) / 100
  exact ⟨number_without_glasses, number_without_glasses, rfl⟩

end students_without_glasses_l576_576953


namespace rectangle_area_l576_576137

theorem rectangle_area (a w l : ℝ) (h_square_area : a = 36) 
    (h_rect_width : w * w = a) 
    (h_rect_length : l = 3 * w) : w * l = 108 := 
sorry

end rectangle_area_l576_576137


namespace min_troublemakers_l576_576508

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l576_576508


namespace parallel_line_slope_l576_576542

theorem parallel_line_slope (x y : ℝ) : (∃ (c : ℝ), 3 * x - 6 * y = c) → (1 / 2) = 1 / 2 :=
by sorry

end parallel_line_slope_l576_576542


namespace sequence_formula_l576_576063

theorem sequence_formula (a : ℕ → ℚ) (h₁ : a 1 = 0)
  (h : ∀ n, a (n + 1) = 1 / (2 - a n)) :
  ∀ n, a n = (n - 1) / n :=
sorry

end sequence_formula_l576_576063


namespace problem_proof_l576_576855

variable {x y z : ℝ}

theorem problem_proof (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x^2 + y^2 + z^2 + 2 * x * y * z = 1) : 2 * (x + y + z) ≤ 3 := 
sorry

end problem_proof_l576_576855


namespace grove_tree_selection_l576_576825

theorem grove_tree_selection (birches spruces pines aspens : ℕ) :
  birches + spruces + pines + aspens = 100 →
  (∀ s : set ℕ, s.card = 85 → (birches ∈ s ∧ spruces ∈ s ∧ pines ∈ s ∧ aspens ∈ s)) →
  ∀ t : set ℕ, t.card = 69 → (birches ∈ t ∧ spruces ∈ t) ∨ (birches ∈ t ∧ pines ∈ t) ∨ (birches ∈ t ∧ aspens ∈ t) ∨ (spruces ∈ t ∧ pines ∈ t) ∨ (spruces ∈ t ∧ aspens ∈ t) ∨ (pines ∈ t ∧ aspens ∈ t) :=
sorry

end grove_tree_selection_l576_576825


namespace rectangle_area_from_square_l576_576146

theorem rectangle_area_from_square {s a : ℝ} (h1 : s^2 = 36) (h2 : a = 3 * s) :
    s * a = 108 :=
by
  -- The proof goes here
  sorry

end rectangle_area_from_square_l576_576146


namespace triangle_at_most_one_obtuse_l576_576548

theorem triangle_at_most_one_obtuse (T : Triangle) : 
  (∃ a1 a2 a3 : Angle, T.has_angles a1 a2 a3 ∧ a1 > 90 ∧ a2 > 90) → False := 
by
  sorry

end triangle_at_most_one_obtuse_l576_576548


namespace smallest_trees_in_three_types_l576_576776

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end smallest_trees_in_three_types_l576_576776


namespace division_quotient_l576_576889

theorem division_quotient (dividend divisor remainder quotient : ℕ)
  (H1 : dividend = 190)
  (H2 : divisor = 21)
  (H3 : remainder = 1)
  (H4 : dividend = divisor * quotient + remainder) : quotient = 9 :=
by {
  sorry
}

end division_quotient_l576_576889


namespace polynomial_bounded_a_l576_576362

-- Definition of the set M of polynomials and the condition P(x) in [-1, 1]
def is_bounded_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, (∀ x : ℝ, |P x| ≤ 1) ∧ (P = λ x, a * x^3 + b * x^2 + c * x + d)

-- Main theorem stating the existence of a constant k such that |a| ≤ k for polynomials in M.
theorem polynomial_bounded_a : ∃ k : ℝ, ∀ (P : ℝ → ℝ), is_bounded_polynomial P → 
  (∃ a b c d : ℝ, P = λ x, a * x^3 + b * x^2 + c * x + d ∧ |a| ≤ k) :=
sorry

end polynomial_bounded_a_l576_576362


namespace dave_added_apps_l576_576652

-- Define the conditions as a set of given facts
def initial_apps : Nat := 10
def deleted_apps : Nat := 17
def remaining_apps : Nat := 4

-- The statement to prove
theorem dave_added_apps : ∃ x : Nat, initial_apps + x - deleted_apps = remaining_apps ∧ x = 11 :=
by
  use 11
  sorry

end dave_added_apps_l576_576652


namespace difference_between_3rd_and_2nd_smallest_l576_576517

-- Define the three numbers as a set
def numbers : Set ℕ := {10, 11, 12}

-- Define the function to find the nth smallest number in a finite set of numbers
def nth_smallest (s : Set ℕ) (n : ℕ) : ℕ :=
  (Finset.sort (· < ·) s.to_finset).nth_le (n - 1) sorry

-- Define what we aim to prove
theorem difference_between_3rd_and_2nd_smallest :
  nth_smallest numbers 3 - nth_smallest numbers 2 = 1 :=
by
  sorry

end difference_between_3rd_and_2nd_smallest_l576_576517


namespace problem1_problem2_l576_576875

-- Define the function f(x) = x^2 - ax + b
def f (x a b : ℝ) := x^2 - a * x + b

-- Problem (1): Prove the solution set of 6x^2 - 5x + 1 > 0
theorem problem1 : ( ∀ x : ℝ, 2 < x ∧ x < 3 → f x 5 6 < 0 ) →
  ( ∀ x : ℝ, 6 * x^2 - 5 * x + 1 > 0 ↔ x < 1 / 3 ∨ x > 1 / 2 ) := sorry

-- Problem (2): Prove the range of a when b = 3 - a
theorem problem2 (a : ℝ) : ( ∀ x : ℝ, x ∈ Icc (-1) 0 → f x a (3 - a) ≥ 0 ) → a ≤ 3 := sorry

end problem1_problem2_l576_576875


namespace union_of_sets_l576_576373

theorem union_of_sets :
  let A := {1, 2, 3}
  let B := {x : ℤ | -1 < x ∧ x < 2}
  A ∪ B = {0, 1, 2, 3} := 
by
  let A := {1, 2, 3}
  let B := {x : ℤ | -1 < x ∧ x < 2}
  show A ∪ B = {0, 1, 2, 3} 
  from sorry

end union_of_sets_l576_576373


namespace sum_of_ages_five_years_ago_l576_576170

-- Definitions from the conditions
variables (A B : ℕ) -- Angela's current age and Beth's current age

-- Conditions
def angela_is_four_times_as_old_as_beth := A = 4 * B
def angela_will_be_44_in_five_years := A + 5 = 44

-- Theorem statement to prove the sum of their ages five years ago
theorem sum_of_ages_five_years_ago (h1 : angela_is_four_times_as_old_as_beth A B) (h2 : angela_will_be_44_in_five_years A) : 
  (A - 5) + (B - 5) = 39 :=
by sorry

end sum_of_ages_five_years_ago_l576_576170


namespace find_sum_of_numbers_l576_576947

theorem find_sum_of_numbers 
  (a b : ℕ)
  (h₁ : a.gcd b = 5)
  (h₂ : a * b / a.gcd b = 120)
  (h₃ : (1 : ℚ) / a + 1 / b = 0.09166666666666666) :
  a + b = 55 := 
sorry

end find_sum_of_numbers_l576_576947


namespace find_n_for_constant_term_l576_576687

theorem find_n_for_constant_term :
  ∃ n : ℕ, (binom n 4) * 15 = 15 := sorry

end find_n_for_constant_term_l576_576687


namespace range_of_g_l576_576201

def g (x : ℝ) : ℝ := arctan x + arctan((2 - x) / (2 + x))

theorem range_of_g :
  ∀ y ∈ set.range g, y = π / 12 ∨ y = -π / 12 :=
by {
  sorry
}

end range_of_g_l576_576201


namespace ratio_of_red_to_total_simplified_l576_576124

def number_of_red_haired_children := 9
def total_number_of_children := 48

theorem ratio_of_red_to_total_simplified:
  (number_of_red_haired_children: ℚ) / (total_number_of_children: ℚ) = (3 : ℚ) / (16 : ℚ) := 
by
  sorry

end ratio_of_red_to_total_simplified_l576_576124


namespace part1_solution_part2_solution_l576_576585

open Real

noncomputable theory

def daily_transport_A_machine (x : ℝ) : ℝ := x
def daily_transport_B_machine (x : ℝ) : ℝ := x + 10

def transport_equation (x : ℝ) : Prop := 
  450 / x = 500 / (x + 10)

def min_cost_purchasing_plan (m : ℕ) : ℝ :=
  if (10 ≤ m ∧ m ≤ 12) ∧ (90 * m + 100 * (30 - m) ≥ 2880) ∧ (1.5 * m + 2 * (30 - m) ≤ 55) then
    1.5 * m + 2 * (30 - m)
  else
    ∞

theorem part1_solution :
  ∃ x : ℝ, transport_equation x ∧ daily_transport_A_machine x = 90 ∧ daily_transport_B_machine x = 100 :=
begin
  sorry
end

theorem part2_solution :
  ∃ m : ℕ, min_cost_purchasing_plan m = 54 :=
begin
  sorry
end

end part1_solution_part2_solution_l576_576585


namespace smallest_possible_overlap_l576_576573

/-!
# Problem Statement
85% of adults drink coffee and 80% drink tea. Prove that the smallest possible percent of adults who drink both coffee and tea is 65%.
-/

-- Define the percentages
def Pc : ℝ := 0.85
def Pt : ℝ := 0.80

-- Let Pct be the percentage of adults who drink both coffee and tea
def smallest_possible_Pct := 0.65

-- The Lean statement that needs to be proved
theorem smallest_possible_overlap : ∃ (Pct : ℝ), Pc + Pt - 1 = Pct ∧ Pct ≥ smallest_possible_Pct := 
sorry

end smallest_possible_overlap_l576_576573


namespace problem_statement_l576_576004

theorem problem_statement {a b c A B C : ℝ} (h₀ : a ≠ c)
  (h₁ : log (sin A), log (sin B), log (sin C) form an arithmetic sequence)
  : (cos A * cos C + cos B^2) * (1 + cos B) - sin A * sin C = 0 →
  perpendicular_line (cos A * cos C + cos B^2) (-sin A) a (1 + cos B) sin C (-c) :=
sorry

end problem_statement_l576_576004


namespace max_interval_length_a_eq_3_l576_576654

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  ((a^2 + a) * x - 1) / (a^2 * x)

-- Definitions for conditions
def interval_length (x1 x2 : ℝ) := x2 - x1

def domain_of_f (a : ℝ) := {x : ℝ | x ≠ 0}

-- Main question prove statement
theorem max_interval_length_a_eq_3 (a : ℝ) (m n : ℝ) (h1 : a ≠ 0)
  (h2 : (domain_of_f a).nonempty)
  (h3 : ∀ x, x ∈ (m, n) → f a x ∈ (m, n)) :
  a = 3 :=
sorry

end max_interval_length_a_eq_3_l576_576654


namespace f_three_equals_322_l576_576908

def f (z : ℝ) : ℝ := (z^2 - 2) * ((z^2 - 2)^2 - 3)

theorem f_three_equals_322 :
  f 3 = 322 :=
by
  -- Proof steps (left out intentionally as per instructions)
  sorry

end f_three_equals_322_l576_576908


namespace ruler_with_marks_l576_576886

def can_measure_all_lengths (total_length : ℕ) (marks : List ℕ) : Prop :=
  ∀ length : ℕ, 1 ≤ length ∧ length ≤ total_length →
  ∃ (slices : List ℕ), (∀ s in slices, s ∈ marks ∨ s = total_length - m for some m in marks ∨ s = marks[i] - marks[j] for some i and j) ∧ slices.sum = length

theorem ruler_with_marks :
  can_measure_all_lengths 9 [1, 4, 7] :=
sorry

end ruler_with_marks_l576_576886


namespace min_value_expression_l576_576211

theorem min_value_expression : 
  ∃ x : ℝ, (∀ y : ℝ, 2 * sin y ^ 4 + 4 * cos y ^ 4 ≥ 2 * sin x ^ 4 + 4 * cos x ^ 4) ∧ (2 * sin x ^ 4 + 4 * cos x ^ 4 = 1 / 3) :=
sorry

end min_value_expression_l576_576211


namespace beetle_centroid_coincides_l576_576077

variable {A B C : Point} (triangle : Triangle A B C)

def centroid (t : Triangle A B C) : Point := sorry

def beetle_movement (p : Point) : Point := sorry

theorem beetle_centroid_coincides (G : Point) 
  (Hcentroid : G = centroid triangle)
  (Hstationary : ∀ t' : Triangle, centroid t' = G) 
  (Hmovement : ∀ cycle_t : Triangle, beetle_movement (centroid cycle_t) = centroid triangle):
  ∀ t'' : Triangle, beetle_movement (centroid t'') = G :=
by {
  sorry
}

end beetle_centroid_coincides_l576_576077


namespace minimum_number_of_troublemakers_l576_576490

theorem minimum_number_of_troublemakers (n m : ℕ) (students : Fin n → Prop) 
  (truthtellers : Fin n → Prop) (liars : Fin n → Prop) (round_table : List (Fin n))
  (h_all_students : n = 29)
  (h_truth_or_liar : ∀ s, students s → (truthtellers s ∨ liars s))
  (h_truth_tellers_conditions : ∀ s, truthtellers s → 
    (count_lies_next_to s = 1 ∨ count_lies_next_to s = 2))
  (h_liars_conditions : ∀ s, liars s → 
    (count_lies_next_to s ≠ 1 ∧ count_lies_next_to s ≠ 2))
  (h_round_table_conditions : round_table.length = n 
    ∧ (∀ i, i < n → round_table.nth i ≠ none)):
  m = 10 :=
by
  sorry

end minimum_number_of_troublemakers_l576_576490


namespace find_m_plus_n_l576_576069

variable (U : Set ℝ) (A : Set ℝ) (CUA : Set ℝ) (m n : ℝ)
  -- Condition 1: The universal set U is the set of all real numbers
  (hU : U = Set.univ)
  -- Condition 2: A is defined as the set of all x such that (x - 1)(x - m) > 0
  (hA : A = { x : ℝ | (x - 1) * (x - m) > 0 })
  -- Condition 3: The complement of A in U is [-1, -n]
  (hCUA : CUA = { x : ℝ | x ∈ U ∧ x ∉ A } ∧ CUA = Icc (-1) (-n))

theorem find_m_plus_n : m + n = -2 :=
  sorry 

end find_m_plus_n_l576_576069


namespace min_trees_for_three_types_l576_576802

-- Define types and trees
inductive TreeType
| birch | spruce | pine | aspen
deriving Inhabited, DecidableEq

-- A grove with 100 trees of any of the four types.
structure Tree :=
(type : TreeType)

constant grove : List Tree
axiom grove_size : grove.length = 100

-- Condition: Among any 85 trees, there are trees of all four types.
axiom all_types_in_any_85 : ∀ (s : Finset Tree), s.card = 85 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, ∃ t4 ∈ s, 
      t1.type = TreeType.birch ∧
      t2.type = TreeType.spruce ∧
      t3.type = TreeType.pine ∧
      t4.type = TreeType.aspen

-- We need to show that at least 69 trees are needed to ensure at least 3 types.
theorem min_trees_for_three_types : 
  ∀ (s : Finset Tree), s.card = 69 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, 
      t1.type ≠ t2.type ∧ t2.type ≠ t3.type ∧ t1.type ≠ t3.type := 
sorry

end min_trees_for_three_types_l576_576802


namespace grove_tree_selection_l576_576823

theorem grove_tree_selection (birches spruces pines aspens : ℕ) :
  birches + spruces + pines + aspens = 100 →
  (∀ s : set ℕ, s.card = 85 → (birches ∈ s ∧ spruces ∈ s ∧ pines ∈ s ∧ aspens ∈ s)) →
  ∀ t : set ℕ, t.card = 69 → (birches ∈ t ∧ spruces ∈ t) ∨ (birches ∈ t ∧ pines ∈ t) ∨ (birches ∈ t ∧ aspens ∈ t) ∨ (spruces ∈ t ∧ pines ∈ t) ∨ (spruces ∈ t ∧ aspens ∈ t) ∨ (pines ∈ t ∧ aspens ∈ t) :=
sorry

end grove_tree_selection_l576_576823


namespace min_troublemakers_l576_576492

theorem min_troublemakers (n : ℕ) (truth_teller liar : ℕ → Prop) 
  (total_students : n = 29)
  (exists_one_liar : ∀ i : ℕ, liar i = true → ((liar (i - 1) = true ∨ liar (i + 1) = true))) -- assume cyclic seating
  (exists_two_liar : ∀ i : ℕ, truth_teller i = true → (liar (i - 1) = true ∧ liar (i + 1) = true)) :
  (∃ k : ℕ, k = 10 ∧ (∀ i : ℕ, i < 29 -> liar i = true -> k ≥ i)) :=
sorry

end min_troublemakers_l576_576492


namespace grove_tree_selection_l576_576824

theorem grove_tree_selection (birches spruces pines aspens : ℕ) :
  birches + spruces + pines + aspens = 100 →
  (∀ s : set ℕ, s.card = 85 → (birches ∈ s ∧ spruces ∈ s ∧ pines ∈ s ∧ aspens ∈ s)) →
  ∀ t : set ℕ, t.card = 69 → (birches ∈ t ∧ spruces ∈ t) ∨ (birches ∈ t ∧ pines ∈ t) ∨ (birches ∈ t ∧ aspens ∈ t) ∨ (spruces ∈ t ∧ pines ∈ t) ∨ (spruces ∈ t ∧ aspens ∈ t) ∨ (pines ∈ t ∧ aspens ∈ t) :=
sorry

end grove_tree_selection_l576_576824


namespace tangent_line_through_origin_l576_576727

theorem tangent_line_through_origin (x : ℝ) (h₁ : 0 < x) (h₂ : ∀ x, ∃ y, y = 2 * Real.log x) (h₃ : ∀ x, y = 2 * Real.log x) :
  x = Real.exp 1 :=
sorry

end tangent_line_through_origin_l576_576727


namespace foil_covered_prism_width_l576_576074

def inner_prism_length (l : ℝ) := l
def inner_prism_width (l : ℝ) := 2 * l
def inner_prism_height (l : ℝ) := l
def inner_prism_volume (l : ℝ) := l * (2 * l) * l

theorem foil_covered_prism_width :
  (∃ l : ℝ, inner_prism_volume l = 128) → (inner_prism_width l + 2 = 8) := by
sorry

end foil_covered_prism_width_l576_576074


namespace daily_profit_140_implies_price_maximize_profit_l576_576625

-- Definitions extracted from the problem
def cost_price := 5
def base_sell_price := 9
def base_sales := 32
def sales_decrease_per_increase := 4

-- Definitions for the first part
def profit_per_item (selling_price : ℕ) : ℕ := selling_price - cost_price
def daily_sales (selling_price : ℕ) : ℕ := base_sales - sales_decrease_per_increase * (selling_price - base_sell_price)
def daily_profit (selling_price : ℕ) : ℕ := profit_per_item selling_price * daily_sales selling_price

-- Definitions for the second part
def profit_function (selling_price : ℕ) : ℕ := 
  -4 * (selling_price * selling_price) + 88 * selling_price - 340

-- Part 1: Prove that for a daily profit of 140 yuan, the selling price is 12 yuan or 10 yuan
theorem daily_profit_140_implies_price (x : ℕ) : daily_profit x = 140 → (x = 12 ∨ x = 10) :=
by
  sorry

-- Part 2: Prove that to maximize profit, the selling price should be 11 yuan and the maximum profit is 144 yuan
theorem maximize_profit (x : ℕ) : 
  (∀ y : ℕ, profit_function x ≥ profit_function y) → 
  (x = 11 ∧ profit_function x = 144) :=
by
  sorry

end daily_profit_140_implies_price_maximize_profit_l576_576625


namespace repeating_decimal_as_fraction_l576_576546

theorem repeating_decimal_as_fraction :
  (∃ y : ℚ, y = 737910 ∧ 0.73 + 864 / 999900 = y / 999900) :=
by
  -- proof omitted
  sorry

end repeating_decimal_as_fraction_l576_576546


namespace students_exam_mark_l576_576046

theorem students_exam_mark
  (N T : ℕ)
  (h1 : T = N * 80)
  (h2 : ∀ m, m = N - 8 → h3 : T = 160 + m * 90) :
  N = 56 :=
by
  sorry

end students_exam_mark_l576_576046


namespace max_value_of_function_l576_576737

theorem max_value_of_function 
  (a : ℝ)
  (h_a : ∀ x ∈ set.Icc (0:ℝ) (4:ℝ), x = 3 → 3 * a * (3:ℝ)^2 - 30 * (3:ℝ) + 36 = 0)
  : ∃ x ∈ set.Icc (0:ℝ) (4:ℝ), (2 * x^3 - 15 * x^2 + 36 * x - 24 = 8) :=
sorry

end max_value_of_function_l576_576737


namespace first_quartile_correct_l576_576976

-- Define the list of numbers
def numbers : List ℝ := [42.6, -24.1, 30, 22, -26.5, 27.8, 33, 35, -42, 24.3, 30.5, -22.7, 26.2, -27.9, 33.1, -35.2]

-- Definition of the first quartile
def first_quartile (lst : List ℝ) : ℝ :=
  let sorted_lst := lst.qsort (· ≤ ·)
  let N := sorted_lst.length
  let pos := ((N + 1 : ℝ) / 4)
  let lower_idx := pos.toInt -- integer part of position
  let fractional_part := pos - lower_idx
  if fractional_part = 0 then
    sorted_lst.nth lower_idx.succ.pred.getD (0 : ℝ)
  else
    let value1 := sorted_lst.nth lower_idx.succ.pred.getD (0 : ℝ)
    let value2 := sorted_lst.nth lower_idx.succ.getD (0 : ℝ)
    value1 + fractional_part * (value2 - value1)

-- Statement of the theorem
theorem first_quartile_correct : first_quartile numbers = -25.9 := by
  sorry

end first_quartile_correct_l576_576976
