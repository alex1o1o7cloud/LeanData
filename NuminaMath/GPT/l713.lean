import Mathlib

namespace sum_angles_tangent_circles_l713_713034

/-- Let there be three circles each passing through a common point P. Each pair 
of circles intersects at a unique point different from P. Is it true that the 
sum of the three angles formed by the tangents drawn at these points of 
intersection, other than P, is 180 degrees? --/
theorem sum_angles_tangent_circles
  {c1 c2 c3 : ℝ → ℝ → Prop} {P : ℝ × ℝ}
  (h1 : ∀ x y, c1 x y → c2 x y → x = P.1 ∧ y = P.2)
  (h2 : ∀ x y, c2 x y → c3 x y → x = P.1 ∧ y = P.2)
  (h3 : ∀ x y, c3 x y → c1 x y → x = P.1 ∧ y = P.2) :
  (¬ ∑ θ in {θ1, θ2, θ3}, θ = 180) :=
sorry

end sum_angles_tangent_circles_l713_713034


namespace find_m_l713_713081

theorem find_m (m : ℕ) : 5 ^ m = 5 * 25 ^ 2 * 125 ^ 3 ↔ m = 14 := by
  sorry

end find_m_l713_713081


namespace largest_square_multiple_of_18_under_500_l713_713054

theorem largest_square_multiple_of_18_under_500 : 
  ∃ n : ℕ, n * n < 500 ∧ n * n % 18 = 0 ∧ (∀ m : ℕ, m * m < 500 ∧ m * m % 18 = 0 → m * m ≤ n * n) → 
  n * n = 324 :=
by
  sorry

end largest_square_multiple_of_18_under_500_l713_713054


namespace sasha_max_quarters_l713_713343

theorem sasha_max_quarters
  (h1 : ∃ q : ℕ, 0.35 * q = 4.50)
  (h2 : ∀ q : ℕ, ∃ d : ℕ, d = q) : 
  ∃ q : ℕ, (0.35 * q = 4.50) ∧ (q ≤ 12) :=
by
  -- The details of the proof are omitted
  sorry

end sasha_max_quarters_l713_713343


namespace hens_not_laying_eggs_l713_713043

def chickens_on_farm := 440
def number_of_roosters := 39
def total_eggs := 1158
def eggs_per_hen := 3

theorem hens_not_laying_eggs :
  (chickens_on_farm - number_of_roosters) - (total_eggs / eggs_per_hen) = 15 :=
by
  sorry

end hens_not_laying_eggs_l713_713043


namespace cos_120_eq_neg_half_l713_713990

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l713_713990


namespace number_of_true_propositions_is_three_l713_713598

-- Given conditions about three distinct planes 
variables (α β γ : Plane)

-- Propositions definitions
def prop1 : Prop := (α ∥ β ∧ β ∥ γ) → α ∥ γ
def prop2 : Prop := (α ∥ β ∧ ∃ a b, α ∩ γ = a ∧ β ∩ γ = b) → ∀ a b, α ∩ γ = a ∧ β ∩ γ = b → a ∥ b
def prop3 : Prop := (α ∥ β ∧ β ⊥ γ) → α ⊥ γ
def prop4 : Prop := (α ⊥ β ∧ β ⊥ γ) → α ⊥ γ

-- Proof Problem Statement
theorem number_of_true_propositions_is_three :
  (nat.count (λ p, p = true) [prop1, prop2, prop3, prop4]) = 3 := sorry

end number_of_true_propositions_is_three_l713_713598


namespace cos_120_eq_neg_half_l713_713978

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713978


namespace probability_xy_minus_x_minus_y_even_l713_713409

theorem probability_xy_minus_x_minus_y_even :
  (∑ x in ({1,2,3,4,5,6,7,8,9,10} : Finset ℕ), 
  ∑ y in ({1,2,3,4,5,6,7,8,9,10} : Finset ℕ), 
  if x ≠ y ∧ (x - 1) % 2 = 0 ∧ (y - 1) % 2 = 0 then 1 else 0)
  / (∑ x in ({1,2,3,4,5,6,7,8,9,10} : Finset ℕ), 
  ∑ y in ({1,2,3,4,5,6,7,8,9,10} : Finset ℕ), 
  if x ≠ y then 1 else 0)
  = 2 / 9 :=
by sorry

end probability_xy_minus_x_minus_y_even_l713_713409


namespace inverse_proportion_quadrants_l713_713246

-- Define the inverse proportion function
def inverse_proportion_function (k : ℝ) : ℝ → ℝ := λ x, k / x

-- Hypotheses: 
variables (k : ℝ) (H : inverse_proportion_function k (-2) = -1)

-- The function is symmetric with respect to the origin.
def symmetric_origin (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- The Lean statement
theorem inverse_proportion_quadrants : (symmetric_origin (inverse_proportion_function k) ∧ inverse_proportion_function k (-2) = -1) →
  ∃ x₁ x₂, (x₁ < 0 ∧ inverse_proportion_function k x₁ < 0) ∧ (x₂ > 0 ∧ inverse_proportion_function k x₂ > 0) :=
by
  intro h
  sorry

end inverse_proportion_quadrants_l713_713246


namespace sum_of_squares_101_to_200_l713_713166

theorem sum_of_squares_101_to_200 : 
  let sum_squares (n : ℕ) := n * (n + 1) * (2 * n + 1) / 6 in
  (sum_squares 200 - sum_squares 100) = 2348350 := by
  sorry

end sum_of_squares_101_to_200_l713_713166


namespace general_term_of_seq_l713_713628

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n > 0, a (n + 1) = real.sqrt (2 + a n)

theorem general_term_of_seq (a : ℕ → ℝ) :
  seq a →
  ∀ n > 0, a n = 2 * real.cos (real.pi / (3 * 2^(n - 1))) :=
by sorry

end general_term_of_seq_l713_713628


namespace B_gain_l713_713484

-- Problem statement and conditions
def principalA : ℝ := 3500
def rateA : ℝ := 0.10
def periodA : ℕ := 2
def principalB : ℝ := 3500
def rateB : ℝ := 0.14
def periodB : ℕ := 3

-- Calculate amount A will receive from B after 2 years
noncomputable def amountA := principalA * (1 + rateA / 1) ^ periodA

-- Calculate amount B will receive from C after 3 years
noncomputable def amountB := principalB * (1 + rateB / 2) ^ (2 * periodB)

-- Calculate B's gain
noncomputable def gainB := amountB - amountA

-- The theorem to prove
theorem B_gain : gainB = 1019.20 := by
  sorry

end B_gain_l713_713484


namespace test_end_time_l713_713881

def start_time := 12 * 60 + 35  -- 12 hours 35 minutes in minutes
def duration := 4 * 60 + 50     -- 4 hours 50 minutes in minutes

theorem test_end_time : (start_time + duration) = 17 * 60 + 25 := by
  sorry

end test_end_time_l713_713881


namespace circles_externally_tangent_l713_713023

theorem circles_externally_tangent (r1 r2 d : ℝ) (h1 : r1 = 2) (h2 : r2 = 3) (h3 : d = 5) :
  d = r1 + r2 :=
by
  rw [h1, h2, h3]
  norm_num
  exact rfl

end circles_externally_tangent_l713_713023


namespace hourly_wage_increase_l713_713884

theorem hourly_wage_increase (W H W' H' : ℝ) (h1 : W * H = W' * H') 
    (h2 : H' = H * 0.8695652173913043) : 
    (W' / W - 1) * 100 ≈ 15 :=
by
  sorry

end hourly_wage_increase_l713_713884


namespace banker_l713_713356

-- Define the given conditions
def present_worth : ℝ := 400
def interest_rate : ℝ := 0.10
def time_period : ℕ := 3

-- Define the amount due in the future
def amount_due (PW : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  PW * (1 + r) ^ n

-- Define the banker's gain
def bankers_gain (A PW : ℝ) : ℝ :=
  A - PW

-- State the theorem we need to prove
theorem banker's_gain_is_correct :
  bankers_gain (amount_due present_worth interest_rate time_period) present_worth = 132.4 :=
by sorry

end banker_l713_713356


namespace equal_perimeters_of_triangles_with_equal_angle_and_excircle_radius_l713_713412

variables {α : Type} [OrdinaryEuclideanGeometry α]
open OrdinaryEuclideanGeometry.Real3

-- Define the triangles and their properties
def Triangle1 (ABC : Triangle α) : Prop :=
  ∃ A B C : Point α, A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 
  let ⦃A B C⦄ := ABC in ∃ r1 : ℝ, 
  exists_equilateral_triangle_with_excircle_radius A B C r1

def Triangle2 (A'B'C' : Triangle α) : Prop :=
  ∃ A' B' C' : Point α, A' ≠ B' ∧ B' ≠ C' ∧ C' ≠ A' ∧ 
  let ⦃A' B' C'⦄ := A'B'C' in ∃ r2 : ℝ, 
  exists_equilateral_triangle_with_excircle_radius A' B' C' r2

-- Define the theorem to prove equivalence of perimeters
theorem equal_perimeters_of_triangles_with_equal_angle_and_excircle_radius (ABC A'B'C' : Triangle α) 
  (h1 : Triangle1 ABC) (h2 : Triangle2 A'B'C') 
  (equal_Angle : angle A B C = angle A' B' C') 
  (equal_radii : excircle_radius ABC = excircle_radius A'B'C') : 
  perimeter ABC = perimeter A'B'C' := sorry

end equal_perimeters_of_triangles_with_equal_angle_and_excircle_radius_l713_713412


namespace B_time_proof_l713_713469

-- Definitions based on given conditions
def A_work : ℝ := 1 / 4
def AC_work : ℝ := 1 / 2
def BC_work : ℝ := 1 / 3

-- Definition we aim to prove
def B_time : ℝ := 12

theorem B_time_proof (C_work B_work : ℝ) (h1 : A_work + C_work = AC_work) (h2 : B_work + C_work = BC_work) 
    (h3 : B_work = 1 / B_time) : B_time = 12 :=
by
  sorry

end B_time_proof_l713_713469


namespace area_of_rhombus_l713_713049

theorem area_of_rhombus (x y : ℝ) (h : |2 * x| + |3 * y| = 12) : 
  let d1 := 12 in
  let d2 := 8 in
  0.5 * d1 * d2 = 48 := 
by
  sorry

end area_of_rhombus_l713_713049


namespace folded_triangle_segment_length_squared_l713_713100

noncomputable def equilateral_triangle {p q : ℝ} (a b c : Type*) (side_length : ℝ) :=
∀ (x y z : a), dist x y = dist y z ∧ dist y z = dist z x ∧ dist z x = side_length

noncomputable def point_on_side {a b : Type*} (A P : a) (B C : b) (distance_from_b : ℝ) :=
dist A P = distance_from_b ∧ dist P B = distance_from_b

noncomputable def folded_equilateral_triangle {a b : Type*} (A B C Ap Bp : a) (P Q : b) : Prop :=
equilateral_triangle A B C 15 ∧
point_on_side A P B (15 - 10) ∧
point_on_side B Q C (15 - 3) ∧
  let AB := dist A B,
      PB := dist P B,
      AQ := dist A Q,
      BQ := dist B Q in
           PB = 5 ∧ BQ = 12 ∧ PQ := √(2 * 5^2 * (1 - (√((15^2 + 5^2 - 5^2)/(2 * 15 * 5)) * √((15^2 + 3^2 - 12^2)/(2 * 15 * 3))))) in
PQ^2 = (5^2 + 5^2 - 2 * 5 * 5 * (√((15^2 + 5^2 - 5^2)/(2 * 15 * 5)) * √((15^2 + 3^2 - 12^2)/(2 * 15 * 3))))  

theorem folded_triangle_segment_length_squared {A B C Ap Bp P Q : ℝ} :
folded_equilateral_triangle A B C Ap Bp P Q →
(PQ)^2 = 50 :=
by {
  sorry
}

end folded_triangle_segment_length_squared_l713_713100


namespace square_mod_3_remainders_square_mod_4_remainders_square_mod_5_remainders_square_mod_8_remainders_l713_713829

theorem square_mod_3_remainders (n : ℤ) : ∃ r : ℤ, r ∈ {0, 1} ∧ (n^2 ≡ r [ZMOD 3]) := sorry

theorem square_mod_4_remainders (n : ℤ) : ∃ r : ℤ, r ∈ {0, 1} ∧ (n^2 ≡ r [ZMOD 4]) := sorry

theorem square_mod_5_remainders (n : ℤ) : ∃ r : ℤ, r ∈ {0, 1, 4} ∧ (n^2 ≡ r [ZMOD 5]) := sorry

theorem square_mod_8_remainders (n : ℤ) : ∃ r : ℤ, r ∈ {0, 1, 4} ∧ (n^2 ≡ r [ZMOD 8]) := sorry

end square_mod_3_remainders_square_mod_4_remainders_square_mod_5_remainders_square_mod_8_remainders_l713_713829


namespace remainder_div1_remainder_div2_l713_713825

open Polynomial

noncomputable def polynomial1 := X^1001 - 1
noncomputable def divisor1 := X^4 + X^3 + 2 * X^2 + X + 1
noncomputable def divisor2 := X^8 + X^6 + 2 * X^4 + X^2 + 1

theorem remainder_div1 :
  (polynomial1 % divisor1) = X^2 * (1 - X) :=
sorry

theorem remainder_div2 :
  (polynomial1 % divisor2) = -2 * X^7 - X^5 - 2 * X^3 - 1 :=
sorry

end remainder_div1_remainder_div2_l713_713825


namespace equal_playtime_l713_713407

theorem equal_playtime (children : ℕ) (total_minutes : ℕ) (simultaneous_players : ℕ) (equal_playtime_per_child : ℕ)
  (h1 : children = 12) (h2 : total_minutes = 120) (h3 : simultaneous_players = 2) (h4 : equal_playtime_per_child = (simultaneous_players * total_minutes) / children) :
  equal_playtime_per_child = 20 := 
by sorry

end equal_playtime_l713_713407


namespace cos_120_degrees_eq_l713_713946

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l713_713946


namespace smallest_value_of_a_squared_plus_b_l713_713299

theorem smallest_value_of_a_squared_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → a * x^3 + b * y^2 ≥ x * y - 1) :
    a^2 + b = 2 / (3 * Real.sqrt 3) :=
by
  sorry

end smallest_value_of_a_squared_plus_b_l713_713299


namespace abs_value_x_minus_2_plus_x_plus_3_ge_4_l713_713384

theorem abs_value_x_minus_2_plus_x_plus_3_ge_4 :
  ∀ x : ℝ, (|x - 2| + |x + 3| ≥ 4) ↔ (x ≤ - (5 / 2)) := 
sorry

end abs_value_x_minus_2_plus_x_plus_3_ge_4_l713_713384


namespace min_flips_to_all_down_l713_713395

theorem min_flips_to_all_down : ∃ n : ℕ, n > 0 ∧ (∀ (c₁ c₂ c₃ c₄ : bool), 
  (c₁ = true) ∧ (c₂ = true) ∧ (c₃ = true) ∧ (c₄ = true) →
  (∃ f : ℕ → bool × bool × bool × bool, (∀ k, 
    f (k + 1) = match f k with
                 | (a, b, c, d) => (bnot a, bnot b, bnot c, d)
                 end ∨
                 f (k + 1) = match f k with
                             | (a, b, c, d) => (a, bnot b, bnot c, bnot d)
                             end ∨
                             f (k + 1) = match f k with
                                         | (a, b, c, d) => (bnot a, b, bnot c, bnot d)
                                         end ∨
                                         f (k + 1) = match f k with
                                                     | (a, b, c, d) => (bnot a, bnot b, c, bnot d)
                                                     end) 
    (f 0 = (true, true, true, true)) →
    (f n = (false, false, false, false)) ∧ n = 4
  )) 
sorry

end min_flips_to_all_down_l713_713395


namespace average_tickets_per_day_l713_713128

def total_revenue : ℕ := 960
def price_per_ticket : ℕ := 4
def number_of_days : ℕ := 3

theorem average_tickets_per_day :
  (total_revenue / price_per_ticket) / number_of_days = 80 := 
sorry

end average_tickets_per_day_l713_713128


namespace convex_quadrilateral_exists_l713_713578

-- Define the notion of points being collinear
def no_three_collinear (points : List (ℝ × ℝ)) : Prop :=
  ∀ (p₁ p₂ p₃ : ℝ × ℝ), p₁ ∈ points → p₂ ∈ points → p₃ ∈ points →
    p₁ ≠ p₂ → p₂ ≠ p₃ → p₁ ≠ p₃ →
    ¬(∃ a b c : ℝ, a ≠ 0 ∨ b ≠ 0 ∧
      a * p₁.1 + b * p₁.2 + c = 0 ∧
      a * p₂.1 + b * p₂.2 + c = 0 ∧
      a * p₃.1 + b * p₃.2 + c = 0)

-- Define the notion of a convex quadrilateral
def is_convex_quadrilateral (p₁ p₂ p₃ p₄ : ℝ × ℝ) : Prop :=
  ∀ (x₁ x₂ x₃ x₄ : ℝ × ℝ), 
    set.pairwise {p₁, p₂, p₃, p₄} (λ p q, p ≠ q) →
    set.pairwise {p₁, p₂, p₃, p₄} (λ p q, ¬ collinear ℝ ({p, q, x₁} : set _)) → 
    convex_hull ℝ ({p₁, p₂, p₃, p₄} : set _) = 
    conv_hull ℝ({p₁, p₂, p₃, p₄} : set _)

theorem convex_quadrilateral_exists {A B C D E : ℝ × ℝ} (h : no_three_collinear [A, B, C, D, E]) :
  ∃ (p₁ p₂ p₃ p₄ : ℝ × ℝ), 
    set.pairwise {p₁, p₂, p₃, p₄} (λ p₁ p₂, ¬ collinear ℝ ({p₁, p₂, E} : set _)) ∧ 
    is_convex_quadrilateral p₁ p₂ p₃ p₄ := 
sorry

end convex_quadrilateral_exists_l713_713578


namespace cos_120_eq_neg_half_l713_713979

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713979


namespace choir_students_min_l713_713867

/-- 
  Prove that the minimum number of students in the choir, where the number 
  of students must be a multiple of 9, 10, and 11, is 990. 
-/
theorem choir_students_min (n : ℕ) :
  (∃ n, n > 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) ∧ (∀ m, m > 0 ∧ m % 9 = 0 ∧ m % 10 = 0 ∧ m % 11 = 0 → n ≤ m) → n = 990 :=
by
  sorry

end choir_students_min_l713_713867


namespace find_n_for_tn_l713_713531

-- Define the sequence recursively
def t : ℕ → ℚ
| 1 := 1
| n := if n % 2 = 0 then 1 + t (n / 2) else (t (n - 1)) ^ 2

-- Problem statement: Find n such that t_n = 16 / 81
theorem find_n_for_tn : ∃ n, t n = 16 / 81 ∧ n = 26 := by
  use 26
  simp only [t]
  have h1 : t (26 - 1) = 4 / 9 := sorry
  have h2 : t ((26 - 2) / 2) = 5 / 4 := sorry
  -- Continuing backwards from the solution steps to reach t(1)
  -- Adding the specifics of these steps:
  sorry

end find_n_for_tn_l713_713531


namespace ProblemStatement_l713_713119

noncomputable def PropositionA : Prop :=
  ∃ (m : ℝ), (m ≠ 2) ∧ (∀ x > 0, (m-1) * x^(m^2 - 4*m + 3) < 0)

noncomputable def PropositionB : Prop :=
  ∀ (a : ℝ), a > 0 → ∃ (x : ℝ), x > 0 ∧ ln x^2 + ln x - a = 0

noncomputable def PropositionC : Prop :=
  ∃ (α β : ℝ), cos(α + β) = cos α + sin β

noncomputable def PropositionD : Prop :=
  ∀ (φ : ℝ), ¬∀ (x : ℝ), x ≥ 0 → sin(2x + φ) = sin(-2x - φ)

theorem ProblemStatement :
  PropositionA ∧ PropositionB ∧ PropositionC → ¬PropositionD :=
by
  sorry

end ProblemStatement_l713_713119


namespace sequence_formula_and_sum_l713_713185

def arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) :=
  ∀ m n k, m < n → n < k → a n^2 = a m * a k

def Sn (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sequence_formula_and_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (arithmetic_geometric_sequence a 0 ∧ a 1 = 2 ∧ geometric_sequence a → ∀ n, a n = 2) ∧
  (arithmetic_geometric_sequence a 4 ∧ a 1 = 2 ∧ geometric_sequence a → ∀ n, a n = 4 * n - 2) ∧
  (arithmetic_geometric_sequence a 4 ∧ a 1 = 2 ∧ (∀ n, S n = (n * (4 * n)) / 2) → ∃ n > 0, S n > 60 * n + 800 ∧ n = 41) ∧
  (arithmetic_geometric_sequence a 0 ∧ a 1 = 2 ∧ (∀ n, S n = 2 * n) → ∀ n > 0, ¬ (S n > 60 * n + 800)) :=
by sorry

end sequence_formula_and_sum_l713_713185


namespace exponent_problem_l713_713239

variable (x y : ℤ)
variable (cond : 3 * y - 2 * x = 4)

theorem exponent_problem (x y : ℤ) (cond : 3 * y - 2 * x = 4) :
  (16^(x + 1)) / (8 ^ (2 * y - 1)) = 1 / 2 := 
by
  sorry

end exponent_problem_l713_713239


namespace exists_polynomials_l713_713725

noncomputable def P_n (n : ℕ) (x : ℕ) : ℕ :=
  ∑ i in finset.range n, x ^ (n - 1 - i)

theorem exists_polynomials (a : ℕ) : ∃ (n : ℕ) (R Q : polynomial ℤ), P_n n x = (1 + a * (polynomial.X) + (polynomial.X)^2 * R) * Q :=
sorry

end exists_polynomials_l713_713725


namespace solution_set_of_inequality_l713_713027

theorem solution_set_of_inequality (x : ℝ) : 
  x^2 - 7*x + 12 < 0 ↔ 3 < x ∧ x < 4 :=
by {
  sorry
}

end solution_set_of_inequality_l713_713027


namespace selling_price_proof_l713_713640

variable (total_cost : ℝ) (cost_first_book : ℝ) (loss_percentage_first_book : ℝ) 
           (gain_percentage_second_book : ℝ)

theorem selling_price_proof 
  (H_total : total_cost = 450)
  (H_first : cost_first_book = 262.5)
  (H_loss : loss_percentage_first_book = 0.15)
  (H_gain : gain_percentage_second_book = 0.19) :
  let selling_price_first_book := cost_first_book - (loss_percentage_first_book * cost_first_book) in
  let cost_second_book := total_cost - cost_first_book in
  let selling_price_second_book := cost_second_book + (gain_percentage_second_book * cost_second_book) in
  selling_price_first_book + selling_price_second_book = 446.25 :=
by
  sorry

end selling_price_proof_l713_713640


namespace starting_time_is_noon_l713_713778

-- Define the necessary variables
def degrees_moved : ℝ := 74.99999999999999
def degrees_per_hour : ℕ := 30
def end_time : ℕ := 14 * 60 + 30  -- 2:30 PM in minutes

-- Calculate the number of minutes the hour hand moves
def minutes_moved : ℝ := (degrees_moved / degrees_per_hour) * 60

-- Define a function to calculate the start time in minutes from midnight
def start_time : ℝ := end_time - minutes_moved

-- The main theorem: the start time in minutes should be 720 minutes (which is 12:00 PM)
theorem starting_time_is_noon : start_time = 720 :=
by
  sorry

end starting_time_is_noon_l713_713778


namespace min_neighbor_pairs_l713_713746

theorem min_neighbor_pairs (n : ℕ) (h : n = 2005) :
  ∃ (pairs : ℕ), pairs = 56430 :=
by
  sorry

end min_neighbor_pairs_l713_713746


namespace cos_120_degrees_eq_l713_713951

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l713_713951


namespace marked_points_form_rectangle_l713_713398

-- marking these definitions noncomputable as we do not need to compute with them
noncomputable def O : Type := _
noncomputable def R : Type := _
noncomputable def O_i : Type := _
noncomputable def R_i : Type := _

-- conditions
variable (O : O)
variable (R : R)
variable (O1 O2 O3 : O_i)
variable (R1 R2 R3 : R_i)

-- Given
axiom touch_each_other_externally : ∀ i ≠ j, ((i = 1 ∧ j = 2) ∨ (i = 2 ∧ j = 3) ∨ (i = 1 ∧ j = 3)) → (O_i O_j = R_i + R_j)
axiom touch_internally : ∀ i, (i = 1 ∨ i = 2 ∨ i = 3) → (O O_i = R - R_i)

-- Prove
theorem marked_points_form_rectangle : ∀ (O O1 O2 O3 : O_i) (R R1 R2 R3 : R_i),
  (touch_each_other_externally O1 O2 O3 R1 R2 R3) ∧ (touch_internally O O1 O2 O3 R R1 R2 R3) →
  (is_rectangle O O1 O2 O3) :=
sorry

end marked_points_form_rectangle_l713_713398


namespace time_to_cross_person_l713_713904

-- Necessary constants and conversion factors
def train_length : ℝ := 500 -- meters
def train_speed : ℝ := 120 * 1000 / 3600 -- kmph to m/s
def person_speed : ℝ := 20 * 1000 / 3600 -- kmph to m/s
def expected_time : ℝ := 18 -- seconds

-- Calculation of the relative speed and time needed for the train to pass the person
noncomputable def relative_speed : ℝ := train_speed - person_speed
noncomputable def time_to_cross : ℝ := train_length / relative_speed

-- Proof statement: Show that the time to cross is approximately 18 seconds
theorem time_to_cross_person : abs (time_to_cross - expected_time) < 1 := by
  sorry

end time_to_cross_person_l713_713904


namespace sufficient_but_not_necessary_condition_l713_713726

theorem sufficient_but_not_necessary_condition (a1 d : ℝ) : 
  (2 * a1 + 11 * d > 0) → (2 * a1 + 11 * d ≥ 0) :=
by
  intro h
  apply le_of_lt
  exact h

end sufficient_but_not_necessary_condition_l713_713726


namespace prob_abs_ξ_minus_10_lt_0_1_l713_713307

-- Conditions
variables (ξ : ℝ → ℝ) (φ : ℝ → ℝ)
noncomputable def normal_distribution := sorry  -- Define the normal distribution function
noncomputable def cdf_standard_normal := φ -- The standard normal distribution CDF

-- Hypothesis that ξ follows a normal distribution N(10, 0.1^2)
axiom ξ_normal : ξ = normal_distribution 10 (0.1^2)

-- Proof statement (omitting proof for now)
theorem prob_abs_ξ_minus_10_lt_0_1 :
  P (|ξ - 10| < 0.1) = φ 10.1 - φ 9.9 :=
sorry

end prob_abs_ξ_minus_10_lt_0_1_l713_713307


namespace nat_solution_unique_l713_713155

theorem nat_solution_unique (n : ℕ) (h : 2 * n - 1 / n^5 = 3 - 2 / n) : 
  n = 1 :=
sorry

end nat_solution_unique_l713_713155


namespace right_triangle_perimeter_5_shortest_altitude_1_l713_713503

-- Definition of a right-angled triangle's sides with given perimeter and altitude
def right_angled_triangle (a b c : ℚ) : Prop :=
a^2 + b^2 = c^2 ∧ a + b + c = 5 ∧ a * b = c

-- Statement of the theorem to prove the side lengths of the triangle
theorem right_triangle_perimeter_5_shortest_altitude_1 :
  ∃ (a b c : ℚ), right_angled_triangle a b c ∧ (a = 5 / 3 ∧ b = 5 / 4 ∧ c = 25 / 12) ∨ (a = 5 / 4 ∧ b = 5 / 3 ∧ c = 25 / 12) :=
by
  sorry

end right_triangle_perimeter_5_shortest_altitude_1_l713_713503


namespace trig_identity_l713_713642

theorem trig_identity (A : ℝ) (h : Real.cos (π + A) = -1/2) : Real.sin (π / 2 + A) = 1/2 :=
by 
sorry

end trig_identity_l713_713642


namespace number_corresponding_to_8_minutes_l713_713466

theorem number_corresponding_to_8_minutes (x : ℕ) : 
  (12 / 6 = x / 480) → x = 960 :=
by
  sorry

end number_corresponding_to_8_minutes_l713_713466


namespace system_solution_exists_l713_713919

theorem system_solution_exists (x y z : ℤ) : 
  (z + 2) * 7 ^ (Int.natAbs y - 1) = 4 ^ (x^2 + 2 * x * y + 1) ∧
  sin (3 * Real.pi * (z : ℝ) / 2) = 1 →
  (x = 1 ∧ y = -1 ∧ z = -1) ∨ (x = -1 ∧ y = 1 ∧ z = -1) :=
by
  intros h
  sorry

end system_solution_exists_l713_713919


namespace sword_length_difference_l713_713132

def christopher_sword := 15.0
def jameson_sword := 2 * christopher_sword + 3
def june_sword := jameson_sword + 5
def average_length := (christopher_sword + jameson_sword + june_sword) / 3
def laura_sword := average_length - 0.1 * average_length
def difference := june_sword - laura_sword

theorem sword_length_difference :
  difference = 12.197 := 
sorry

end sword_length_difference_l713_713132


namespace jony_speed_l713_713694

theorem jony_speed :
  let start_block := 10 in
  let end_block := 90 in
  let turnaround_block := 70 in
  let block_distance := 40 in
  let start_time := 0 in  -- in minutes, 07:00 is taken as 0
  let end_time := 40 in   -- in minutes, 07:40 is 40 minutes
  let total_blocks := (end_block - start_block) + (end_block - turnaround_block) in
  let total_distance := total_blocks * block_distance in
  let total_time := end_time - start_time in
  let speed := total_distance / total_time in
  speed = 100 :=
by
  -- placeholders for actual proof steps
  let start_block := 10
  let end_block := 90
  let turnaround_block := 70
  let block_distance := 40
  let start_time := 0
  let end_time := 40
  let total_blocks := (end_block - start_block) + (end_block - turnaround_block)
  let total_distance := total_blocks * block_distance
  let total_time := end_time - start_time
  let speed := total_distance / total_time
  have h1 : total_blocks = 100 := by sorry
  have h2 : total_distance = 4000 := by sorry
  have h3 : total_time = 40 := by sorry
  have h4 : speed = total_distance / total_time := by sorry
  exact sorry

end jony_speed_l713_713694


namespace seven_line_intersections_twenty_one_line_intersections_l713_713148

-- Problem 1

theorem seven_line_intersections : 
  let k_values : List ℝ := [0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9]
  (lines : Set (ℝ → ℝ)) := 
  {f | ∃ k ∈ k_values, ∀ x, f x = - k * x - k^3} 
  (intersections : Set (ℝ × ℝ)) :=
  {p | ∃ l1 l2 ∈ lines, l1 ≠ l2 ∧ ∃ x, p = (x, l1 x) ∧ p = (x, l2 x)} 
  (intersections ∧ ∀ p1 p2 ∈ intersections, p1 = p2) →
  intersections.card = 11 :=
sorry

-- Problem 2

theorem twenty_one_line_intersections : 
  let k_values : List ℝ := [0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 
                            0.6, -0.6, 0.7, -0.7, 0.8, -0.8, 0.9, -0.9, 1.0, -1.0]
  (lines : Set (ℝ → ℝ)) := 
  {f | ∃ k ∈ k_values, ∀ x, f x = - k * x - k^3} 
  (intersections : Set (ℝ × ℝ)) :=
  {p | ∃ l1 l2 ∈ lines, l1 ≠ l2 ∧ ∃ x, p = (x, l1 x) ∧ p = (x, l2 x)} 
  (intersections ∧ ∀ p1 p2 ∈ intersections, p1 = p2) →
  intersections.card = 110 :=
sorry

end seven_line_intersections_twenty_one_line_intersections_l713_713148


namespace whales_last_year_eq_4000_l713_713389

variable (W : ℕ) (last_year this_year next_year : ℕ)

theorem whales_last_year_eq_4000
    (h1 : this_year = 2 * last_year)
    (h2 : next_year = this_year + 800)
    (h3 : next_year = 8800) :
    last_year = 4000 := by
  sorry

end whales_last_year_eq_4000_l713_713389


namespace not_a_factorization_method_l713_713835

def factorization_methods : Set String := 
  {"Taking out the common factor", "Cross multiplication method", "Formula method", "Group factorization"}

theorem not_a_factorization_method : 
  ¬ ("Addition and subtraction elimination method" ∈ factorization_methods) :=
sorry

end not_a_factorization_method_l713_713835


namespace log_expression_identity_l713_713930

theorem log_expression_identity :
  log10 ((4 * Real.sqrt 2) / 7) - log10 (2 / 3) + log10 (7 * Real.sqrt 5) = log10 6 + 1/2 :=
sorry

end log_expression_identity_l713_713930


namespace children_ages_l713_713486

theorem children_ages (parent_age : ℕ) (a d : ℕ) :
  parent_age^2 = (a^2 + (a+d)^2 + (a+2*d)^2 + (a+3*d)^2 + (a+4*d)^2 + (a+5*d)^2 + (a+6*d)^2 + (a+7*d)^2 + (a+8*d)^2) →
  a = 2 → d = 3 → parent_age = 48 :=
begin
  intros h_eq h_a h_d,
  sorry
end

end children_ages_l713_713486


namespace factorial_division_l713_713519

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division (n : ℕ) (h : n > 0) : factorial (n) / factorial (n - 1) = n :=
by
  sorry

example : factorial 12 / factorial 11 = 12 :=
by
  exact factorial_division 12 (by norm_num)

end factorial_division_l713_713519


namespace custom_operation_correct_l713_713020

def custom_operation (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem custom_operation_correct : custom_operation 6 3 = 27 :=
by {
  sorry
}

end custom_operation_correct_l713_713020


namespace simplest_quadratic_radical_correct_l713_713436

noncomputable def simplest_quadratic_radical (A B C D: ℝ) : ℝ := B

theorem simplest_quadratic_radical_correct (A B C D : ℝ) 
  (hA : A = Real.sqrt 9) 
  (hB : B = Real.sqrt 7) 
  (hC : C = Real.sqrt 20) 
  (hD : D = Real.sqrt (1/3)) : 
  simplest_quadratic_radical A B C D = Real.sqrt 7 := 
by 
  rw simplest_quadratic_radical 
  rw hB
  sorry

end simplest_quadratic_radical_correct_l713_713436


namespace rectangle_side_significant_digits_l713_713377

theorem rectangle_side_significant_digits :
  ∀ (area : ℝ), area = 0.07344 → (∃ (side : ℝ), significant_digits side = 3) :=
by
  sorry

end rectangle_side_significant_digits_l713_713377


namespace shorter_side_of_rectangle_l713_713789

variable (R : Type) [LinearOrderedField R]

noncomputable def findShorterSide (a b : R) : Prop :=
  let d : R := real.sqrt (a^2 + b^2)
  let x : R := a / 3
  let y : R := b / 4
  a = 3 * x ∧ b = 4 * x ∧ (d = 9) →

  (3 * (9 / 5) = 5.4)

theorem shorter_side_of_rectangle : ∀ (a b : R), a = 3 * (9 / 5) → findShorterSide a b :=
by
  sorry

end shorter_side_of_rectangle_l713_713789


namespace cos_120_eq_neg_half_l713_713982

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713982


namespace second_player_wins_l713_713033

theorem second_player_wins :
  ∃ (win_strategy : ∀ (n : ℕ), n > 0 → n % 4 ≠ 0 → ℕ), 
  (∃ (turns : ℕ → ℕ), 
    (turns 0 = 100) ∧ 
    (∀ (i : ℕ), turns (i + 1) = win_strategy (turns i)) ∧ 
    (∃ (i : ℕ), turns i = 0 ∧ ∀ (j : ℕ), j < i → turns j % 4 ≠ 0)) :=
sorry

end second_player_wins_l713_713033


namespace correct_quadratic_equation_l713_713813

theorem correct_quadratic_equation :
  (∃ (a b c : ℤ), a = 1 ∧ 
   (5 + 3 = -b) ∧ 
   (-6 * -4 = c) ∧ 
   (1 * x^2 + b * x + c = 0)) →
  x^2 - 8x + 24 = 0 :=
by
  sorry

end correct_quadratic_equation_l713_713813


namespace katya_minimum_problems_l713_713286

-- Defining the conditions
def katya_probability_solve : ℚ := 4 / 5
def pen_probability_solve : ℚ := 1 / 2
def total_problems : ℕ := 20
def minimum_correct_for_good_grade : ℚ := 13

-- The Lean statement to prove
theorem katya_minimum_problems (x : ℕ) :
  x * katya_probability_solve + (total_problems - x) * pen_probability_solve ≥ minimum_correct_for_good_grade → x ≥ 10 :=
sorry

end katya_minimum_problems_l713_713286


namespace inscribed_circle_radius_one_third_altitude_l713_713269

theorem inscribed_circle_radius_one_third_altitude (a b c : ℝ)
  (h1 : b = (a + c) / 2) :
  let s := (a + b + c) / 2,
      Δ := 1 / 2 * a * (2 * Δ / b),
      r := Δ / s,
      h_B := (2 * Δ) / b
  in r = h_B / 3 :=
by
  sorry

end inscribed_circle_radius_one_third_altitude_l713_713269


namespace find_term_gt_10000_l713_713362

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n > 0, a (n + 1) = (∑ i in Finset.range n, a (i + 1)) + a n

theorem find_term_gt_10000 (a : ℕ → ℕ) (h : sequence a) : ∃ n, a n > 10000 :=
by
  have h_base : a 1 = 2 := h.1
  have h_ind : ∀ n > 0, a (n + 1) = (∑ i in Finset.range n, a (i + 1)) + a n := h.2
  -- Proceed in proof with calculated or hypothesized steps
  sorry

end find_term_gt_10000_l713_713362


namespace cos_120_eq_neg_half_l713_713987

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l713_713987


namespace surface_area_of_cube_l713_713799

-- Define the condition: volume of the cube is 1728 cubic centimeters
def volume_cube (s : ℝ) : ℝ := s^3
def given_volume : ℝ := 1728

-- Define the question: surface area of the cube
def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

-- The statement that needs to be proved
theorem surface_area_of_cube :
  ∃ s : ℝ, volume_cube s = given_volume → surface_area_cube s = 864 :=
by
  sorry

end surface_area_of_cube_l713_713799


namespace new_volume_proof_l713_713024

variable (r h : ℝ)
variable (π : ℝ := Real.pi) -- Lean's notation for π
variable (original_volume : ℝ := 15) -- given original volume

-- Define original volume of the cylinder
def V := π * r^2 * h

-- Define new volume of the cylinder using new dimensions
def new_V := π * (3 * r)^2 * (2 * h)

-- Prove that new_V is 270 when V = 15
theorem new_volume_proof (hV : V = 15) : new_V = 270 :=
by
  -- Proof will go here
  sorry

end new_volume_proof_l713_713024


namespace range_of_approximate_number_l713_713352

theorem range_of_approximate_number (a : ℝ) (h : a ≈ 270000) : 265000 ≤ a ∧ a < 275000 :=
sorry

end range_of_approximate_number_l713_713352


namespace triangular_array_distribution_count_l713_713109

noncomputable def triangular_array_valid_distributions : ℕ :=
  let n := 12 in
  let valid_distributions := (finset.range 2^n).filter (λ dist, 
    let x := λ i, (dist / 2^i) % 2 in
    (fin.sum univ (λ k, x k * (finset.choose (n-1) k % 4)) ) % 4 = 0
  ) in
  valid_distributions.card

theorem triangular_array_distribution_count : triangular_array_valid_distributions = 1280 :=
sorry

end triangular_array_distribution_count_l713_713109


namespace shaded_area_first_quadrant_l713_713408

/-- Two concentric circles with radii 15 and 9. Prove that the area of the shaded region 
within the first quadrant is 36π. -/
theorem shaded_area_first_quadrant (r_big r_small : ℝ) (h_big : r_big = 15) (h_small : r_small = 9) : 
  (π * (r_big ^ 2 - r_small ^ 2)) / 4 = 36 * π := 
by
  sorry

end shaded_area_first_quadrant_l713_713408


namespace simplify_fraction_expression_l713_713069

variable (d : ℤ)

theorem simplify_fraction_expression : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := by
  sorry

end simplify_fraction_expression_l713_713069


namespace work_completed_in_30_days_l713_713843

namespace WorkProblem

variables (W : ℝ) (A B : ℝ)

-- Condition: A can finish the work alone in 60 days
def rate_A : ℝ := W / 60

-- Condition: A and B together worked for 20 days, then A alone worked 20 days.
noncomputable def rate_AB (x : ℝ) := (W / 60) + (W / x)

theorem work_completed_in_30_days (htogether : 20 * rate_AB W 60 + 20 * (W / 60) = W) : 
  20 * (1 / 60 + 1 / 60) + 20 * (1 / 60) = 1 →
  1 / (1 / 60 + 1 / 60) = 30 :=
by
  sorry

end WorkProblem

end work_completed_in_30_days_l713_713843


namespace cos_120_eq_neg_half_l713_713936

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713936


namespace cos_120_eq_neg_half_l713_713973

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713973


namespace area_of_plot_l713_713775

def cm_to_miles (a : ℕ) : ℕ := a * 9

def miles_to_acres (b : ℕ) : ℕ := b * 640

theorem area_of_plot :
  let bottom := 12
  let top := 18
  let height := 10
  let area_cm2 := ((bottom + top) * height) / 2
  let area_miles2 := cm_to_miles area_cm2
  let area_acres := miles_to_acres area_miles2
  area_acres = 864000 :=
by
  sorry

end area_of_plot_l713_713775


namespace sickness_temperature_increase_l713_713039

theorem sickness_temperature_increase :
  ∀ (normal_temp fever_threshold current_temp : ℕ), normal_temp = 95 → fever_threshold = 100 →
  current_temp = fever_threshold + 5 → (current_temp - normal_temp = 10) :=
by
  intros normal_temp fever_threshold current_temp h1 h2 h3
  sorry

end sickness_temperature_increase_l713_713039


namespace integer_solutions_of_inequality_l713_713014

theorem integer_solutions_of_inequality :
  {x : ℤ | 3 ≤ 5 - 2 * x ∧ 5 - 2 * x ≤ 9} = {-2, -1, 0, 1} :=
by
  sorry

end integer_solutions_of_inequality_l713_713014


namespace _l713_713458

noncomputable def limit_theorem (f : ℝ → ℝ) (a L : ℝ) : Prop :=
∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - a) < δ → abs (f x - L) < ε

example : limit_theorem (λ x, (x^3 + 7*x^2 + 15*x + 9) / (x^3 + 8*x^2 + 21*x + 18)) (-3) 2 :=
by {
  sorry
}

end _l713_713458


namespace ap_sum_proof_l713_713076

noncomputable def AP_sum_first_15 (a11 a7 : ℝ) (d a1 : ℝ) (h1 : a11 = a1 + 10 * d) (h2 : a7 = a1 + 6 * d) (n : ℕ) (h : n = 15) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem ap_sum_proof (a11 a7 : ℝ) (h1 : a11 = 5.25) (h2 : a7 = 3.25) :
  AP_sum_first_15 a11 a7 0.5 0.25 5.25 3.25 15 rfl = 56.25 :=
sorry

end ap_sum_proof_l713_713076


namespace first_nonzero_digit_one_over_157_l713_713419

theorem first_nonzero_digit_one_over_157 : 
  (∃ d : ℕ, d < 10 ∧ (∃ n : ℕ, (1000 / 157 : ℝ) = (6 + d * 10^-1 + n * 10^-2) * 157)) →
  d = 3 :=
by sorry

end first_nonzero_digit_one_over_157_l713_713419


namespace prob_sum_zero_prob_unique_solution_l713_713596

open Finset

-- Definitions of sets A and B
def A : Finset ℤ := {-1, 0, 1}
def B : Finset ℤ := {-2, -1, 0, 1, 2}

-- First proof statement: Probability that the sum of selected numbers is 0
theorem prob_sum_zero (a ∈ A) (b ∈ B) : (card { (a, b) ∈ A.product(B) | a + b = 0 }) / (card (A.product B)) = 1 / 5 := sorry

-- Second proof statement: Probability that the system of equations has only one solution
theorem prob_unique_solution (a ∈ A) (b ∈ B) : (card { (a, b) ∈ A.product(B) | b = 2 * a }) / (card (A.product B)) = 3 / 15 → 
(card { (a, b) ∈ A.product(B) | ¬(b = 2 * a) }) / (card (A.product B)) = 4 / 5 := sorry

end prob_sum_zero_prob_unique_solution_l713_713596


namespace b_is_square_of_positive_integer_l713_713747

theorem b_is_square_of_positive_integer 
  (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (h : b^2 = a^2 + ab + b) : 
  ∃ k : ℕ, b = k^2 := 
by 
  sorry

end b_is_square_of_positive_integer_l713_713747


namespace midpoint_trajectory_l713_713248

theorem midpoint_trajectory (x y : ℝ) (h : ∃ (xₚ yₚ : ℝ), yₚ = 2 * xₚ^2 + 1 ∧ y = 4 * (xₚ / 2) ^ 2) : y = 4 * x ^ 2 :=
sorry

end midpoint_trajectory_l713_713248


namespace cos_120_eq_neg_half_l713_713964

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l713_713964


namespace triangular_array_distribution_count_l713_713108

noncomputable def triangular_array_valid_distributions : ℕ :=
  let n := 12 in
  let valid_distributions := (finset.range 2^n).filter (λ dist, 
    let x := λ i, (dist / 2^i) % 2 in
    (fin.sum univ (λ k, x k * (finset.choose (n-1) k % 4)) ) % 4 = 0
  ) in
  valid_distributions.card

theorem triangular_array_distribution_count : triangular_array_valid_distributions = 1280 :=
sorry

end triangular_array_distribution_count_l713_713108


namespace total_pages_read_l713_713689

variable (Jairus_pages : ℕ)
variable (Arniel_pages : ℕ)
variable (J_total : Jairus_pages = 20)
variable (A_total : Arniel_pages = 2 + 2 * Jairus_pages)

theorem total_pages_read : Jairus_pages + Arniel_pages = 62 := by
  rw [J_total, A_total]
  sorry

end total_pages_read_l713_713689


namespace cos_120_eq_neg_half_l713_713977

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713977


namespace better_value_price_l713_713336

def price_per_ounce (cost : ℕ) (weight : ℕ) : ℕ :=
  cost / weight

theorem better_value_price (
  (large_box_cost : ℕ) (large_box_weight : ℕ)
  (small_box_cost : ℕ) (small_box_weight : ℕ)
  (h_large_cost : large_box_cost = 480)
  (h_large_weight : large_box_weight = 30)
  (h_small_cost : small_box_cost = 340)
  (h_small_weight : small_box_weight = 20) :
  min (price_per_ounce large_box_cost large_box_weight) (price_per_ounce small_box_cost small_box_weight) = 16 := by
  sorry

end better_value_price_l713_713336


namespace round_45_67893_to_tenth_l713_713752

def round_to_tenth (x : ℝ) : ℝ :=
  let tenths : ℝ := (Real.floor (x * 10) : ℝ) / 10
  let hundredths : ℝ := x * 100 - (Real.floor (x * 10) : ℝ * 10)
  if hundredths >= 5 then tenths + 0.1 else tenths

theorem round_45_67893_to_tenth : round_to_tenth 45.67893 = 45.7 :=
by
  sorry

end round_45_67893_to_tenth_l713_713752


namespace simplest_form_fraction_l713_713833

theorem simplest_form_fraction 
  (m n a : ℤ) (h_f1 : (2 * m) / (10 * m * n) = 1 / (5 * n))
  (h_f2 : (m^2 - n^2) / (m + n) = (m - n))
  (h_f3 : (2 * a) / (a^2) = 2 / a) : 
  ∀ (f : ℤ), f = (m^2 + n^2) / (m + n) → 
    (∀ (k : ℤ), k ≠ 1 → (m^2 + n^2) / (m + n) ≠ k * f) :=
by
  intros f h_eq k h_kneq1
  sorry

end simplest_form_fraction_l713_713833


namespace no_three_consecutive_odd_sums_of_two_squares_l713_713759

theorem no_three_consecutive_odd_sums_of_two_squares (k : ℤ):
  ¬ (∃ (a b c : ℤ), (∀ n ∈ {a, b, c}, (∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ n = x^2 + y^2))
    ∧ {a, b, c} = {4*k + 1, 4*k + 3, 4*k + 5}) :=
by {
  sorry
}

end no_three_consecutive_odd_sums_of_two_squares_l713_713759


namespace unique_id_function_l713_713858

open Function

noncomputable def strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ ⦃a b⦄, a < b → f a < f b

theorem unique_id_function (f : ℕ → ℕ) 
  (h_mono : strictly_increasing f)
  (h_func_eq : ∀ m n : ℕ, Nat.coprime m n → f (m * n) = f m * f n)
  (h_f2 : f 2 = 2) :
  ∀ n : ℕ, f n = n :=
by {
  sorry
}

end unique_id_function_l713_713858


namespace jezebel_total_cost_l713_713274

theorem jezebel_total_cost :
  let red_rose_count := 2 * 12,
      sunflower_count := 3,
      red_rose_cost := 1.50,
      sunflower_cost := 3,
      total_cost := (red_rose_count * red_rose_cost) + (sunflower_count * sunflower_cost)
  in
  total_cost = 45 := 
by
  sorry

end jezebel_total_cost_l713_713274


namespace rectangle_shorter_side_l713_713788

theorem rectangle_shorter_side
  (x : ℝ)
  (a b d : ℝ)
  (h₁ : a = 3 * x)
  (h₂ : b = 4 * x)
  (h₃ : d = 9) :
  a = 5.4 := 
by
  sorry

end rectangle_shorter_side_l713_713788


namespace find_k_l713_713241

theorem find_k (k : ℕ) (h1 : 12^k ∣ 856736) : 3^k - k^3 = 1 :=
by {
  have h2 : k = 0, {
    -- Translate divisibility condition logic (already proven through factorization steps)
    sorry,
  },
  rw h2,
  norm_num,
}

end find_k_l713_713241


namespace partnership_total_profit_l713_713113

theorem partnership_total_profit
  (total_capital : ℝ)
  (A_share : ℝ := 1/3)
  (B_share : ℝ := 1/4)
  (C_share : ℝ := 1/5)
  (D_share : ℝ := 1 - (A_share + B_share + C_share))
  (A_profit : ℝ := 805)
  (A_capital : ℝ := total_capital * A_share)
  (total_capital_positive : 0 < total_capital)
  (shares_add_up : A_share + B_share + C_share + D_share = 1) :
  (A_profit / (total_capital * A_share)) * total_capital = 2415 :=
by
  -- Proof will go here.
  sorry

end partnership_total_profit_l713_713113


namespace Pyarelal_loss_is_1800_l713_713918

noncomputable def Ashok_and_Pyarelal_loss (P L : ℝ) : Prop :=
  let Ashok_cap := (1 / 9) * P
  let total_cap := P + Ashok_cap
  let Pyarelal_ratio := P / total_cap
  let total_loss := 2000
  let Pyarelal_loss := Pyarelal_ratio * total_loss
  Pyarelal_loss = 1800

theorem Pyarelal_loss_is_1800 (P : ℝ) (h1 : P > 0) (h2 : L = 2000) :
  Ashok_and_Pyarelal_loss P L := sorry

end Pyarelal_loss_is_1800_l713_713918


namespace pattern_fraction_pattern_double_fraction_problem_1_problem_2_problem_3_l713_713342

-- Definition of the pattern: 1/(n*(n+1)) = 1/n - 1/(n+1)
theorem pattern_fraction (n : Nat) (h : 0 < n) : (1 : ℚ) / (n * (n + 1)) = 1 / n - 1 / (n + 1) := by sorry

-- Definition of the pattern: 2/(n*(n+2)) = 1/n - 1/(n+2)
theorem pattern_double_fraction (n : Nat) (h : 0 < n) : (2 : ℚ) / (n * (n + 2)) = 1 / n - 1 / (n + 2) := by sorry

-- Problem 1: Prove the specific instance of the pattern
theorem problem_1 : (1 : ℚ) / (2022 * 2023) = 1 / 2022 - 1 / 2023 := by
  exact pattern_fraction 2022 (Nat.pos_of_ne_zero (by decide))

-- Problem 2: Calculate the value of the given expression
theorem problem_2 : 1 - 1 / 2 - 1 / 6 - 1 / 12 - 1 / 20 - 1 / 30 = (1 : ℚ) / 6 := by sorry

-- Problem 3: Sum a series using the pattern and calculate
theorem problem_3 : 
  (Finset.range 1011).sum (λ k, (2 : ℚ) / (2 * (k + 1) * (2 * (k + 1) + 2))) = 1011 / 2024 := 
by sorry

end pattern_fraction_pattern_double_fraction_problem_1_problem_2_problem_3_l713_713342


namespace eddys_climbing_rate_l713_713638

def base_camp_ft := 5000
def departure_time := 6 -- in hours: 6:00 AM
def hillary_climbing_rate := 800 -- ft/hr
def stopping_distance_ft := 1000 -- ft short of summit
def hillary_descending_rate := 1000 -- ft/hr
def passing_time := 12 -- in hours: 12:00 PM

theorem eddys_climbing_rate :
  ∀ (base_ft departure hillary_rate stop_dist descend_rate pass_time : ℕ),
    base_ft = base_camp_ft →
    departure = departure_time →
    hillary_rate = hillary_climbing_rate →
    stop_dist = stopping_distance_ft →
    descend_rate = hillary_descending_rate →
    pass_time = passing_time →
    (pass_time - departure) * hillary_rate - descend_rate * (pass_time - (departure + (base_ft - stop_dist) / hillary_rate)) = 6 * 500 :=
by
  intros
  sorry

end eddys_climbing_rate_l713_713638


namespace four_machines_save_11_hours_l713_713467

-- Define the conditions
def three_machines_complete_order_in_44_hours := 3 * (1 / (3 * 44)) * 44 = 1

def additional_machine_reduces_time (T : ℝ) := 4 * (1 / (3 * 44)) * T = 1

-- Define the theorem to prove the number of hours saved
theorem four_machines_save_11_hours : 
  (∃ T : ℝ, additional_machine_reduces_time T ∧ three_machines_complete_order_in_44_hours) → 
  44 - 33 = 11 :=
by
  sorry

end four_machines_save_11_hours_l713_713467


namespace avg_velocity_within_first_second_avg_velocity_between_one_and_two_seconds_instantaneous_velocity_at_one_second_l713_713504

noncomputable def s (t : ℝ) : ℝ :=
  -3 * t^3 + t^2 + 20

theorem avg_velocity_within_first_second :
  ∀ t ∈ Icc (0 : ℝ) (1 : ℝ), 
  (s 1 - s 0) / (1 - 0) = -2 := 
by
  sorry

theorem avg_velocity_between_one_and_two_seconds :
  ∀ t ∈ Icc (1 : ℝ) (2 : ℝ), 
  (s 2 - s 1) / (2 - 1) = -18 := 
by
  sorry

noncomputable def v' (t : ℝ) : ℝ :=
  -9 * t^2 + 2 * t

theorem instantaneous_velocity_at_one_second :
  v' 1 = -7 :=
by
  sorry

end avg_velocity_within_first_second_avg_velocity_between_one_and_two_seconds_instantaneous_velocity_at_one_second_l713_713504


namespace hexagon_area_of_parallels_tangent_incircle_l713_713040

def triangle (α β γ : ℝ) := α + β + γ = 180

theorem hexagon_area_of_parallels_tangent_incircle
  (a b c : ℝ)
  (ha : a = 5)
  (hb : b = 7)
  (hc : c = 8)
  (h_inc : incircle_tangent (triangle 5 7 8)) :
  hexagon_area_of_tangents (triangle 5 7 8) = (31 / 5) * real.sqrt 3 :=
by
  -- All the conditions are provided in the assumptions.
  sorry

end hexagon_area_of_parallels_tangent_incircle_l713_713040


namespace exists_plane_γ_l713_713021

variables (α : Type*) [affine_space α] (β γ : affine_subspace α) 
variables (B P : α) (trace_l : set α) (trace_l' : set α) (point_E point_F : α)
variables (proj_P : α) (proj_BE proj_BF : set α)

-- Conditions
-- Plane β is given by its trace and point B
def plane_β (β : affine_subspace α) := ∀ x ∈ trace_l, x ∈ β ∧ B ∈ β

-- Point P is given with its image and projection
def point_P_with_proj (P proj_P : α) := ∃ proj_P, P ∈ proj_P

-- Parallel condition between projections
def parallel_proj (e' f' proj_BE proj_BF : set α) := ∀ e' f', e' ∥ proj_BE ∧ f' ∥ proj_BF

-- Main Theorem
theorem exists_plane_γ (β γ : affine_subspace α) (B P : α) :
  (plane_β β → point_P_with_proj P proj_P → parallel_proj proj_BE proj_BF) →
  ∃ γ, γ ∥ β ∧ P ∈ γ :=
sorry

end exists_plane_γ_l713_713021


namespace min_value_l713_713649

theorem min_value : ∀ (a b : ℝ), a + b^2 = 2 → (∀ x y : ℝ, x = a^2 + 6 * y^2 → y = b) → (∃ c : ℝ, c = 3) :=
by
  intros a b h₁ h₂
  sorry

end min_value_l713_713649


namespace count_funny_vs_happy_l713_713330

-- Define the properties of digits
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def odd_digit (n : ℕ) : Prop := n ∈ {1, 3, 5, 7, 9}
def even_digit (n : ℕ) : Prop := n ∈ {0, 2, 4, 6, 8}

-- Define the properties of "funny" and "happy" numbers
def funny_number (n : ℕ) : Prop := 
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  in odd_digit d1 ∧ even_digit d2 ∧ odd_digit d3

def happy_number (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  in d1 ≠ 0 ∧ even_digit d1 ∧ odd_digit d2 ∧ even_digit d3

-- State the theorem
theorem count_funny_vs_happy : 
  let funny_count := (filter funny_number (range 100 1000)).length
  let happy_count := (filter happy_number (range 100 1000)).length
  funny_count - happy_count = 25 := sorry

end count_funny_vs_happy_l713_713330


namespace ellipse_equation_circle_intersection_l713_713610

-- Proof Problem (Ⅰ)
theorem ellipse_equation (a b c : ℝ) 
  (h1 : a > b ∧ b > 0) 
  (h2 : c / a = sqrt 3 / 2) 
  (h3 : a^2 = b^2 + c^2) 
  (h4 : 1 / a^2 + 3 / (4 * b^2) = 1) : 
  (a = 2 ∧ b = 1 ∧ c = sqrt 3) ∧ (∀ x y, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}) :=
sorry

-- Proof Problem (Ⅱ)
theorem circle_intersection (P1 P2 : ℝ × ℝ) 
  (l : ℝ → ℝ) 
  (h1 : ∀ x y, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1})
  (h2 : ∃! p, p ∈ {q : ℝ × ℝ | q.1^2 / 4 + q.2^2 = 1} ∧ p ∈ {r : ℝ × ℝ | l r.1 = r.2}) :
  ∃ r : ℝ, r > 0 ∧ (∀ p : ℝ × ℝ, p ∈ {q : ℝ × ℝ | q.1^2 + q.2^2 = r^2} → p ∈ {q : ℝ × ℝ | q.1^2 / 4 + q.2^2 = 1} ∨ p ∈ {q : ℝ × ℝ | l q.1 = q.2}) ∧ r = sqrt 5 :=
sorry

end ellipse_equation_circle_intersection_l713_713610


namespace S8_equals_85_l713_713597

-- Definitions based on conditions
def geometric_sum (terms : ℕ → ℕ) (n : ℕ) : ℕ := (List.iota n).sum (terms)

noncomputable def terms : ℕ → ℕ := λ n, n  -- Placeholder for the actual sequence terms

noncomputable def S (n : ℕ) : ℕ := geometric_sum terms n

-- Conditions from the problem
axiom S2 : S 2 = 1
axiom S4 : S 4 = 5

-- Prove the final goal
theorem S8_equals_85 : S 8 = 85 :=
by
  sorry

end S8_equals_85_l713_713597


namespace students_unable_to_partner_l713_713401

/-- 
Three different 6th grade classes are combining for a square dancing unit. The first class has 17 males and 13 females, while the second class has 
14 males and 18 females, and the third class has 15 males and 17 females. Prove that the number of students who cannot partner with a student of the 
opposite gender is 2.
-/
theorem students_unable_to_partner : 
  let males1 := 17
  let females1 := 13
  let males2 := 14
  let females2 := 18
  let males3 := 15
  let females3 := 17
  let total_males := males1 + males2 + males3
  let total_females := females1 + females2 + females3
  total_females - total_males = 2 := 
by
  let males1 := 17
  let females1 := 13
  let males2 := 14
  let females2 := 18
  let males3 := 15
  let females3 := 17
  let total_males := males1 + males2 + males3
  let total_females := females1 + females2 + females3
  have h1 : total_males = 46 := by sorry
  have h2 : total_females = 48 := by sorry
  show total_females - total_males = 2 from
    calc
      total_females - total_males = 48 - 46 : by rw [h1, h2]
      ... = 2 : by norm_num

end students_unable_to_partner_l713_713401


namespace ds_equals_2pl_l713_713869

noncomputable def circle_in_angle (O P Q R L : Point) : Prop :=
  circle O ∧ inscribed_in_angle Q P R ∧ touches_side_at O PR L

noncomputable def tangent_parallel (O P Q R L S D : Point) : Prop :=
  tangent_to_circle_parallel_to PO S ∧ intersects_ray S PQ ∧ intersects_ray D LP

theorem ds_equals_2pl
  (O P Q R L S D : Point)
  (circ_in_ang : circle_in_angle O P Q R L)
  (tang_par : tangent_parallel O P Q R L S D) :
  DS = 2 * PL := 
sorry

end ds_equals_2pl_l713_713869


namespace ratio_of_combined_area_to_combined_perimeter_l713_713410

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ :=
  (s^2 * Real.sqrt 3) / 4

noncomputable def equilateral_triangle_perimeter (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_combined_area_to_combined_perimeter :
  (equilateral_triangle_area 6 + equilateral_triangle_area 8) / 
  (equilateral_triangle_perimeter 6 + equilateral_triangle_perimeter 8) = (25 * Real.sqrt 3) / 42 :=
by
  sorry

end ratio_of_combined_area_to_combined_perimeter_l713_713410


namespace correct_statement_l713_713840

-- Define each statement as a predicate
def statement_A (r: ℝ) : Prop :=
  abs r = r ∧ r > 0

def statement_B : Prop :=
  ∀ RSS : ℝ, RSS < 0 → RSS < 1

def statement_C (R2: ℝ) : Prop :=
  R2 < 0.5

def statement_D : Prop :=
  ∀ model, ¬linear model

-- Theorem stating which statement is correct
theorem correct_statement : statement_B := 
by 
  sorry

end correct_statement_l713_713840


namespace neg_exists_is_forall_l713_713376

theorem neg_exists_is_forall: 
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
by
  sorry

end neg_exists_is_forall_l713_713376


namespace cos_120_eq_neg_half_l713_713939

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713939


namespace first_nonzero_digit_one_div_157_l713_713424

theorem first_nonzero_digit_one_div_157 : 
  ∃ d : ℕ, d = 6 ∧ (∃ n : ℕ, n ≥ 1 ∧ n * d = (1000 * 1 / 157).floor) := 
by
  sorry

end first_nonzero_digit_one_div_157_l713_713424


namespace math_problem_l713_713593

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def g' : ℝ → ℝ := sorry

def condition1 (x : ℝ) : Prop := f (x + 3) = g (-x) + 4
def condition2 (x : ℝ) : Prop := f' x + g' (1 + x) = 0
def even_function (x : ℝ) : Prop := g (2 * x + 1) = g (- (2 * x + 1))

theorem math_problem (x : ℝ) :
  (∀ x, condition1 x) →
  (∀ x, condition2 x) →
  (∀ x, even_function x) →
  (g' 1 = 0) ∧
  (∀ x, f (1 - x) = f (x + 3)) ∧
  (∀ x, f' x = f' (-x + 2)) :=
by
  intros
  sorry

end math_problem_l713_713593


namespace find_m_for_min_value_l713_713576

theorem find_m_for_min_value :
  ∃ (m : ℝ), ( ∀ x : ℝ, (y : ℝ) = m * x^2 - 4 * x + 1 → (∃ x_min : ℝ, (∀ x : ℝ, (m * x_min^2 - 4 * x_min + 1 ≤ m * x^2 - 4 * x + 1) → y = -3))) :=
sorry

end find_m_for_min_value_l713_713576


namespace cosine_120_eq_negative_half_l713_713962

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l713_713962


namespace fruit_basket_combinations_l713_713641

theorem fruit_basket_combinations :
  let apples := 3
  let oranges := 7
  let bananas := 4
  let total_combinations := (apples+1) * (oranges+1) * (bananas+1)
  let empty_basket := 1
  total_combinations - empty_basket = 159 :=
by
  let apples := 3
  let oranges := 7
  let bananas := 4
  let total_combinations := (apples + 1) * (oranges + 1) * (bananas + 1)
  let empty_basket := 1
  have h_total_combinations : total_combinations = 4 * 8 * 5 := by sorry
  have h_empty_basket : empty_basket = 1 := by sorry
  have h_subtract : 4 * 8 * 5 - 1 = 159 := by sorry
  exact h_subtract

end fruit_basket_combinations_l713_713641


namespace chinese_chess_draw_probability_l713_713442

theorem chinese_chess_draw_probability (pMingNotLosing : ℚ) (pDongLosing : ℚ) : 
    pMingNotLosing = 3/4 → 
    pDongLosing = 1/2 → 
    (pMingNotLosing - (1 - pDongLosing)) = 1/4 :=
by
  intros
  sorry

end chinese_chess_draw_probability_l713_713442


namespace one_yard_equals_28_45_strides_l713_713348

noncomputable def steps_to_strides (s : ℚ) : ℚ := s * (4 / 3)
noncomputable def leaps_to_steps (l : ℚ) : ℚ := l * (2 / 5)
noncomputable def leaps_to_yards (l : ℚ) : ℚ := l * (6 / 7)
noncomputable def yards_to_leaps (y : ℚ) : ℚ := y * (7 / 6)
noncomputable def steps_to_yards (s : ℚ) : ℚ := leaps_to_yards (steps_to_leaps s)
noncomputable def strides_to_yards (st : ℚ) : ℚ := steps_to_yards (strides_to_steps st)

theorem one_yard_equals_28_45_strides : 
  let y_in_steps := (7 / 15 : ℚ)
    let y_in_strides := y_in_steps * (4 / 3)
  in y_in_strides = (28 / 45 : ℚ) :=
by
  let y_in_steps := (7 / 15 : ℚ)
  let y_in_strides := y_in_steps * (4 / 3)
  show y_in_strides = (28 / 45 : ℚ), from sorry

end one_yard_equals_28_45_strides_l713_713348


namespace xy_largest_l713_713704

variable {α : Type} [LinearOrderedField α]

-- Define the variables for side lengths
variables (WZ YZ XZ ZW XY : α)

-- Given conditions
def cyclic_quadrilateral : Prop := WZ * YZ = XZ * ZW

-- Law of cosines expressions for XY
def law_of_cosines_WZY (cos_WZY : α) : α :=
  WZ^2 + ZW^2 - 2 * WZ * ZW * cos_WZY

def law_of_cosines_YZX (cos_YZX : α) : α :=
  YZ^2 + XZ^2 - 2 * YZ * XZ * cos_YZX

-- Largest possible value for XY condition
def largest_xy (XY : α) : Prop :=
  XY^2 = 355

-- Final statement combining all conditions and desired result
theorem xy_largest 
  (h : cyclic_quadrilateral) 
  (h_law_wzy : ∃ cos_WZY, law_of_cosines_WZY cos_WZY = XY^2) 
  (h_law_yzx : ∃ cos_YZX, law_of_cosines_YZX cos_YZX = XY^2)
  (h_cos_relation : ∀ cos_WZY cos_YZX, cos_WZY = -cos_YZX) :
  largest_xy XY := 
sorry

end xy_largest_l713_713704


namespace magnitude_of_b_l713_713219

open Real

noncomputable def a : ℝ × ℝ := (-sqrt 3, 1)

theorem magnitude_of_b (b : ℝ × ℝ)
    (h1 : (a.1 + 2 * b.1, a.2 + 2 * b.2) = (a.1, a.2))
    (h2 : (a.1 + b.1, a.2 + b.2) = (b.1, b.2)) :
    sqrt (b.1 ^ 2 + b.2 ^ 2) = sqrt 2 :=
sorry

end magnitude_of_b_l713_713219


namespace volume_fraction_of_A_l713_713507

theorem volume_fraction_of_A (V : ℝ) (V > 0) :
  (1 / 2) * (2 / 3) * V = (1 / 3) * V :=
by sorry

end volume_fraction_of_A_l713_713507


namespace number_is_two_l713_713648

theorem number_is_two 
  (N : ℝ)
  (h1 : N = 4 * 1 / 2)
  (h2 : (1 / 2) * N = 1) :
  N = 2 :=
sorry

end number_is_two_l713_713648


namespace maximum_area_of_triangle_l713_713707

theorem maximum_area_of_triangle (A B C : ℝ) (a b c : ℝ) (hC : C = π / 6) (hSum : a + b = 12) :
  ∃ (S : ℝ), S = 9 ∧ ∀ S', S' ≤ S := 
sorry

end maximum_area_of_triangle_l713_713707


namespace cypress_tree_count_l713_713636

-- Definitions for the conditions
def priceCabin : ℕ := 129000
def cashInHand : ℕ := 150
def cashLeft : ℕ := 350
def numPineTrees : ℕ := 600
def numMapleTrees : ℕ := 24
def priceCypressTree : ℕ := 100
def priceMapleTree : ℕ := 300
def pricePineTree : ℕ := 200

-- Statement to prove
theorem cypress_tree_count (C : ℕ) :
  let totalAmountRaised := priceCabin + cashLeft - cashInHand
      amountFromPines := numPineTrees * pricePineTree
      amountFromMaples := numMapleTrees * priceMapleTree
      amountFromCypress := totalAmountRaised - amountFromPines - amountFromMaples
      numCypress := amountFromCypress / priceCypressTree
  in numCypress = 20 := by
  sorry

end cypress_tree_count_l713_713636


namespace shorter_side_of_rectangle_l713_713790

variable (R : Type) [LinearOrderedField R]

noncomputable def findShorterSide (a b : R) : Prop :=
  let d : R := real.sqrt (a^2 + b^2)
  let x : R := a / 3
  let y : R := b / 4
  a = 3 * x ∧ b = 4 * x ∧ (d = 9) →

  (3 * (9 / 5) = 5.4)

theorem shorter_side_of_rectangle : ∀ (a b : R), a = 3 * (9 / 5) → findShorterSide a b :=
by
  sorry

end shorter_side_of_rectangle_l713_713790


namespace value_of_a_l713_713600

theorem value_of_a (x y a : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : 3 * x - a * y = 1) : a = 1 := by
  sorry

end value_of_a_l713_713600


namespace polynomial_divisible_2520_l713_713758

theorem polynomial_divisible_2520 (n : ℕ) : (n^7 - 14 * n^5 + 49 * n^3 - 36 * n) % 2520 = 0 := 
sorry

end polynomial_divisible_2520_l713_713758


namespace cos_120_eq_neg_half_l713_713970

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l713_713970


namespace length_of_bridge_l713_713455

-- Conditions: defining constants
def length_of_train : ℝ := 150 -- in meters
def speed_of_train_km_per_hr : ℝ := 45 -- in km/h
def time_to_cross_bridge : ℝ := 30 -- in seconds

-- Conversion factor from km/hr to m/s
def speed_of_train_m_per_s : ℝ := speed_of_train_km_per_hr * 1000 / 3600

-- Statement: the length of the bridge can be proven to be 225 meters
theorem length_of_bridge : 
  let total_distance := speed_of_train_m_per_s * time_to_cross_bridge in
  total_distance - length_of_train = 225 := 
by
  sorry

end length_of_bridge_l713_713455


namespace option_a_option_b_option_c_l713_713592

open Real

-- Define the functions f and g
variable {f g : ℝ → ℝ}

-- Given conditions
axiom cond1 : ∀ x : ℝ, f(x + 3) = g(-x) + 4
axiom cond2 : ∀ x : ℝ, deriv f x + deriv g (1 + x) = 0
axiom cond3 : ∀ x : ℝ, g(2*x + 1) = g(-(2*x) + 1)

-- Prove the statements
theorem option_a : deriv g 1 = 0 :=
sorry

theorem option_b : ∀ x : ℝ, f(x + 4) = f(4 - x) :=
sorry

theorem option_c : ∀ x : ℝ, deriv f (x + 1) = deriv f (1 - x) :=
sorry

end option_a_option_b_option_c_l713_713592


namespace tom_sends_168_roses_in_week_l713_713807

theorem tom_sends_168_roses_in_week : 
  (let number_of_roses_in_a_dozen := 12
       number_of_dozens_sent_per_day := 2
       number_of_days_in_week := 7
       daily_roses := number_of_roses_in_a_dozen * number_of_dozens_sent_per_day
       weekly_roses := daily_roses * number_of_days_in_week
   in weekly_roses) = 168 := 
by 
  -- proof goes here
  sorry

end tom_sends_168_roses_in_week_l713_713807


namespace find_constants_l713_713157

theorem find_constants (P Q R : ℤ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (2 * x^2 - 5 * x + 6) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1)) →
  P = -6 ∧ Q = 8 ∧ R = -5 :=
by
  sorry

end find_constants_l713_713157


namespace vec_AB_eq_l713_713595

def point := (ℝ × ℝ × ℝ)

def vec_sub (A B : point) : point :=
  (B.1 - A.1, B.2 - A.2, B.3 - A.3)

theorem vec_AB_eq :
  let A := (3, 1, 2) : point
  let B := (4, -2, -2) : point
  vec_sub A B = (1, -3, -4) :=
by
  sorry

end vec_AB_eq_l713_713595


namespace number_of_dispatch_plans_l713_713106

def teacher := unit -- Placeholder type for teachers
def school := unit -- Placeholder type for schools

noncomputable def total_dispatch_plans : ℕ := 30

theorem number_of_dispatch_plans : 
  let teachers : list teacher := [unit.star, unit.star, unit.star, unit.star, unit.star] in
  let A := teachers.head in
  let B := teachers.nth_le 1 (by norm_num) in
  let C := teachers.nth_le 2 (by norm_num) in
  let D := teachers.nth_le 3 (by norm_num) in
  let E := teachers.nth_le 4 (by norm_num) in
  exists (plan : list (list teacher)),
    plan.length = 3 ∧
    (∀ s, s ∈ plan → ¬s.empty) ∧
    (A ∈ plan.head ∧ B ∈ plan.head) ∧ 
    (∀ s, C ∈ s → A ∉ s) ∧
    (∀ t, t ∈ teachers → t ∈ plan.join) ∧
    (plan.join ~ teachers) ∧ -- permutation of lists
    plan.countp (λ s, s.length = 2) = 2 ∧
    plan.countp (λ s, s.length = 3) = 1 ∧
    plan.countp (λ s, s.length = 1) = 3 ∧
    plan.countp (λ s, s.length = 1) + 
      plan.countp (λ s, s.length = 2) + 
      plan.countp (λ s, s.length = 3) = 30 := sorry

end number_of_dispatch_plans_l713_713106


namespace sweets_ratio_l713_713485

theorem sweets_ratio (total_sweets : ℕ) (mother_ratio : ℚ) (eldest_sweets second_sweets : ℕ)
  (h1 : total_sweets = 27) (h2 : mother_ratio = 1 / 3) (h3 : eldest_sweets = 8) (h4 : second_sweets = 6) :
  let mother_sweets := mother_ratio * total_sweets
  let remaining_sweets := total_sweets - mother_sweets
  let other_sweets := eldest_sweets + second_sweets
  let youngest_sweets := remaining_sweets - other_sweets
  youngest_sweets / eldest_sweets = 1 / 2 :=
by
  sorry

end sweets_ratio_l713_713485


namespace dalton_glasses_l713_713138

def minutes_in_hours (h : ℕ) : ℕ := h * 60

def total_minutes (hours : ℕ) (minutes : ℕ) : ℕ := minutes_in_hours hours + minutes

def glasses_of_lemonade (total_minutes : ℕ) (rate_minutes : ℕ) : ℕ := total_minutes / rate_minutes

theorem dalton_glasses (hours : ℕ) (minutes : ℕ) (rate_minutes : ℕ) : hours = 5 → minutes = 50 → rate_minutes = 20 → glasses_of_lemonade (total_minutes hours minutes) rate_minutes = 17 :=
by {
  intros h_eq m_eq r_eq,
  unfold total_minutes,
  rw [h_eq, m_eq, minutes_in_hours],
  change 5 * 60 + 50 with 350,
  unfold glasses_of_lemonade,
  change 350 / 20 with 17,
  exact rfl,
}

end dalton_glasses_l713_713138


namespace max_pos_numbers_circle_l713_713853

theorem max_pos_numbers_circle
  (a : Fin 100 → ℤ)
  (h : ∀ i : Fin 100, a i > a ((i + 1) % 100) + a ((i + 2) % 100)) :
  ∃ POS : ℕ, POS ≤ 49 ∧ POS = (Finset.univ.filter (λ i, 0 < a i)).card :=
sorry

end max_pos_numbers_circle_l713_713853


namespace slope_product_is_neg_one_l713_713774

noncomputable def slope_product (m n : ℝ) : ℝ := m * n

theorem slope_product_is_neg_one 
  (m n : ℝ)
  (eqn1 : ∀ x, ∃ y, y = m * x)
  (eqn2 : ∀ x, ∃ y, y = n * x)
  (angle : ∃ θ1 θ2 : ℝ, θ1 = θ2 + π / 4)
  (neg_reciprocal : m = -1 / n):
  slope_product m n = -1 := 
sorry

end slope_product_is_neg_one_l713_713774


namespace number_of_integers_in_1000_that_can_be_expressed_l713_713116

noncomputable def f (x : ℝ) : ℕ := Nat.floor (2 * x) + Nat.floor (4 * x) + Nat.floor (6 * x) + Nat.floor (8 * x)

theorem number_of_integers_in_1000_that_can_be_expressed : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 1000 ∧ ∃ x : ℝ, f x = n}.card = 600 := 
sorry

end number_of_integers_in_1000_that_can_be_expressed_l713_713116


namespace hyperbola_equation_l713_713483

theorem hyperbola_equation (foci_eq : (0, ± 4 * Real.sqrt 3)) (ecc_ellipse : Real := Real.sqrt 3 / 2)
  (eq_rec_ecce : Real := 2 / Real.sqrt 3) (a : ℝ := 6) (b : ℝ := 2 * Real.sqrt 3):
  (4 * (x : ℝ)^2) + y^2 = 64 → 
  ecc_ellipse * eq_rec_ecce = 1 →
  foci_eq = (0, ± 4 * Real.sqrt 3) →
  (y^2) / 36 - (x^2) / 12 = 1 :=
by
  intros h_eq_ellipse h_ecc h_foci
  sorry

end hyperbola_equation_l713_713483


namespace min_digits_representation_l713_713360

theorem min_digits_representation (N : ℕ) (h1 : ∀ d ∈ to_digits N, d = 1 ∨ d = 2)
  (h2 : ∀ n ∈ (list.Ico 0 10000),
    let target := (list.repeat 1 9999).insert_nth n 2 in
    target ⊆ (to_digits N)) :
  ∃ n, to_digits N = list.repeat 1 99 ++ list.repeat 2 100 ++ list.repeat 1 99 ∧
       n = 10198 :=
by
  sorry

end min_digits_representation_l713_713360


namespace cos_120_eq_neg_half_l713_713942

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713942


namespace factorize_expression_l713_713556

theorem factorize_expression (x y : ℝ) :
  9 * x^2 - y^2 - 4 * y - 4 = (3 * x + y + 2) * (3 * x - y - 2) :=
by
  sorry

end factorize_expression_l713_713556


namespace cos_120_degrees_l713_713997

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l713_713997


namespace min_value_of_sum_log_cond_l713_713233

open Real

theorem min_value_of_sum_log_cond (a b : ℝ) (h1 : log 4 (a + b) = log 2 (sqrt (a * b))) (h2 : a > 0) (h3 : b > 0) :
  (a + b) ≥ 4 :=
sorry

end min_value_of_sum_log_cond_l713_713233


namespace min_distance_is_9_div_5_l713_713508

-- Define the function that represents the graph
def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + 3 * Real.log x

-- Define the line equation
def g (x : ℝ) : ℝ := 2 * x + (1 / 2)

-- Define the minimum distance function
def min_distance (x m y n : ℝ) : ℝ := (x - m)^2 + (y - n)^2

theorem min_distance_is_9_div_5 :
  ∃ x m y n, 
    (∃ (hx : x > 0), y = f x) ∧ (∃ m n, n = g m) ∧
    min_distance x m y n = 9 / 5 :=
by
  sorry

end min_distance_is_9_div_5_l713_713508


namespace perpendicular_lines_sin2theta_l713_713247

theorem perpendicular_lines_sin2theta (θ : ℝ)
  (h1 : ∀ x y : ℝ, x * cos θ + 2 * y = 0 → y = -((1 / 2) * x * cos θ))
  (h2 : ∀ x y : ℝ, 3 * x + y * sin θ + 3 = 0 → y = -(3 * x + 3) * sin θ)
  (h_perpendicular : (∃ x1 y1 x2 y2 : ℝ, (y1 = -((1 / 2) * x1 * cos θ)) ∧ (y2 = -(3 * x2 + 3) * sin θ) ∧ (-((1 / 2) * cos θ) * -(3 * sin θ) = -1))) :
  sin (2 * θ) = -12 / 13 :=
by
  sorry

end perpendicular_lines_sin2theta_l713_713247


namespace man_speed_approx_l713_713913

noncomputable def speed_of_man :=
  let speed_of_train_kmh := 63
  let length_of_train := 800
  let crossing_time := 47.99616030717543
  let speed_of_train_ms := speed_of_train_kmh * 1000 / 3600
  let relative_speed := length_of_train / crossing_time
  speed_of_train_ms - relative_speed

theorem man_speed_approx :
  abs (speed_of_man - 0.832) < 0.001 := sorry

end man_speed_approx_l713_713913


namespace probability_student_gets_at_least_12_correct_l713_713900

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

noncomputable def prob_at_least_12_correct : ℚ :=
  (finset.range 21).filter (≥ 12)
  .sum (λ k, binomial_prob 20 k (1/2 : ℚ))

theorem probability_student_gets_at_least_12_correct :
  prob_at_least_12_correct = 160466 / 1048576 :=
begin
  sorry
end

end probability_student_gets_at_least_12_correct_l713_713900


namespace range_of_a_l713_713220

noncomputable def quadratic_has_two_negative_roots (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧
               x1 + x2 = 2a ∧
               x1 * x2 = |a| ∧
               (4a^2 - 4*|a|) ≥ 0

theorem range_of_a (a : ℝ) (h : quadratic_has_two_negative_roots a) : 
  ∃ l u : ℝ, l ≤ a ∧ a ≤ u :=
sorry

end range_of_a_l713_713220


namespace proof_l713_713310

noncomputable def f (a b x : ℝ) := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem proof :
  ∀ (a b : ℝ),
    a ≠ 0 → b ≠ 0 →
    (∀ x : ℝ, f a b x ≤ |f a b (π / 6)|) →
    (f a b (11 * π / 12) = 0) ∧ 
    (|f a b (7 * π / 12)| < |f a b (π / 5)|) ∧ 
    (¬ (∀ x : ℝ, f a b (-x) = f a b x) ∧ ¬ (∀ x : ℝ, f a b (-x) = -f a b x)) ∧
    (∀ k : ℤ, ¬ (monotonic (λ x, f a b x) (k * π + π / 6) (k * π + 2 * π / 3))) ∧
    (∀ m n : ℝ, ∃ x : ℝ, f a b x = m * x + n) :=
by
  intros a b ha hb hf
  split
  { -- Proof for statement 1
    sorry },
  { -- Proof for statement 2
    sorry },
  split
  { -- Proof for statement 3
    sorry },
  { -- Proof for statement 4
    sorry },
  { -- Proof for statement 5
    sorry }

end proof_l713_713310


namespace proof_tan_alpha_proof_exp_l713_713195

-- Given conditions
variables (α : ℝ) (h_condition1 : Real.tan (α + Real.pi / 4) = - 1 / 2) (h_condition2 : Real.pi / 2 < α ∧ α < Real.pi)

-- To prove
theorem proof_tan_alpha :
  Real.tan α = -3 :=
sorry -- proof goes here

theorem proof_exp :
  (Real.sin (2 * α) - 2 * Real.cos α ^ 2) / Real.sin (α - Real.pi / 4) = - 2 * Real.sqrt 5 / 5 :=
sorry -- proof goes here

end proof_tan_alpha_proof_exp_l713_713195


namespace red_tiles_count_l713_713487

theorem red_tiles_count (blue_tiles : ℕ) (total_needed : ℕ) (additional_needed : ℕ)
  (h1 : blue_tiles = 48) (h2 : total_needed = 100) (h3 : additional_needed = 20) :
  total_needed - additional_needed - blue_tiles = 32 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end red_tiles_count_l713_713487


namespace probability_of_rolling_2_4_6_l713_713063

theorem probability_of_rolling_2_4_6 : 
  let outcomes := {1, 2, 3, 4, 5, 6}
  let favorable := {2, 4, 6}
  -- Probability is the ratio of the size of the favorable outcomes set to the size of the total outcomes set
  (favorable.to_finset.card * 1.0) / (outcomes.to_finset.card * 1.0) = 1 / 2 :=
by
  sorry

end probability_of_rolling_2_4_6_l713_713063


namespace sum_of_squares_of_digits_l713_713457

-- Definitions to represent the conditions
def jump_sequence : ℕ × ℕ → ℕ × ℕ
| (0, m) => (1, m + 1)  -- 1 cm east
| (1, m) => (m, m + 2)  -- 2 cm north
| (2, m) => (m - 3, m + 3)  -- 3 cm west
| (3, m) => (m - 4, m + 4)  -- 4 cm south

-- The total number of jumps needed
def total_jumps (n : ℕ) : ℕ := n

-- Prove the sum of the squares of the digits of n
theorem sum_of_squares_of_digits (n : ℕ) (h1 : total_jumps n = 323) : 
  let digits := (n / 100, (n / 10) % 10, n % 10)
  let squares_sum := digits.1^2 + digits.2^2 + digits.3^2
  squares_sum = 22 :=
by 
  -- Use sorry to mark the place where proof should be implemented.
  sorry

end sum_of_squares_of_digits_l713_713457


namespace triangle_inequality_l713_713301

theorem triangle_inequality (a b c : ℝ) (n : ℕ) (h1 : 2 ≤ n) (h2 : a + b + c = 1) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (Real.root (↑n) (a^n + b^n) + Real.root (↑n) (b^n + c^n) + Real.root (↑n) (c^n + a^n)) < 1 + (Real.root (↑n) 2 / 2) :=
by
  sorry

end triangle_inequality_l713_713301


namespace length_AD_l713_713809

-- Definitions of given conditions
variables (O B D A : Point)
variables (r1 r2 : ℝ)
variables [Circle ω] [Circle Ω]
variables (tangent_BD : Tangent B D ω)

-- Specific values for the radii
def radius_omega : real := 8
def radius_Omega : real := 13

-- AB is a diameter of Ω
axiom AB_diameter_Omega : Diameter A B Ω

-- Tangent from B to ω touches ω at D
axiom tangent_from_B : Tangent B D ω

-- Prove that the length of AD is 19
theorem length_AD (h_radius_omega : radius ω = 8)
                  (h_radius_Omega : radius Ω = 13)
                  (h_AB_diameter : AB_diameter_Omega)
                  (h_tangent : tangent_from_B) :
    distance A D = 19 :=
by 
    sorry

end length_AD_l713_713809


namespace angle_PAC_half_pi_l713_713842

theorem angle_PAC_half_pi (A B C P Q : Point) (hABC : Triangle A B C) 
  (hAngleA : Angle A > pi / 2) (hP_on_BC : OnLine P (LineThrough B C)) 
  (hQ_on_BC : OnLine Q (LineThrough B C)) (hDistinct_PQ : P ≠ Q) 
  (hAngleBAP_PAQ : AngleBAP = AnglePAQ) 
  (hBP_CQ_eq_BC_PQ : dist B P * dist C Q = dist B C * dist P Q) : 
  AnglePAC = pi / 2 :=
by
  sorry

end angle_PAC_half_pi_l713_713842


namespace bushels_needed_l713_713538

theorem bushels_needed (cows : ℕ) (bushels_per_cow : ℕ)
                       (sheep : ℕ) (bushels_per_sheep : ℕ)
                       (chickens : ℕ) (bushels_per_chicken : ℕ) :
  (cows = 4) → (bushels_per_cow = 2) →
  (sheep = 3) → (bushels_per_sheep = 2) →
  (chickens = 7) → (bushels_per_chicken = 3) →
  (cows * bushels_per_cow + sheep * bushels_per_sheep + chickens * bushels_per_chicken = 35) :=
begin
  sorry
end

end bushels_needed_l713_713538


namespace base7_subtraction_l713_713564

def base7 : Type := Fin 7

def to_base7 (n : Nat) : list base7 := [2, 3, 3, 3]
def to_base7_minuend : list base7 := [1, 1, 1, 1]
def to_base7_result : list base7 := [1, 2, 2, 2]

theorem base7_subtraction :
  (to_base7 2333 - to_base7_minuend 1111 = to_base7_result 1222) := by
    sorry

end base7_subtraction_l713_713564


namespace rate_for_paving_the_floor_l713_713373

theorem rate_for_paving_the_floor :
  ∀ (length width : ℝ)(total_cost : ℝ),
    length = 5.5 →
    width = 3.75 →
    total_cost = 8250 →
    (total_cost / (length * width)) = 400 := by
  intros length width total_cost h_length h_width h_cost
  rw [h_length, h_width, h_cost]
  norm_num
  sorry -- Assuming norm_num suffices to compute this.

end rate_for_paving_the_floor_l713_713373


namespace shortest_segment_l713_713681

/-- Given a quadrilateral with vertices A, B, C, D and the angles:
  - ∠ABD = 35°
  - ∠ADB = 60°
  - ∠CBD = 70°
  - ∠BDC = 65°
   
  Prove that the segment AD is the shortest side in the quadrilateral. --/
theorem shortest_segment
  (A B C D : Type)
  (triangle_ABD : ∠ A B D = 35 ∧ ∠ A D B = 60)
  (triangle_BCD : ∠ B C D = 70 ∧ ∠ B D C = 65) :
  (AD < AB ∧ AB < BD ∧ BD < BC ∧ BC < CD) → AD < all_segments :=
begin
  sorry, -- Placeholder for the actual proof
end

end shortest_segment_l713_713681


namespace shaded_areas_are_different_l713_713136

theorem shaded_areas_are_different (s : ℝ) (hs : s > 0) :
  let shaded_area_fig1 := s^2 * (1 - π / 4),
      shaded_area_fig2 := s^2 / 2,
      shaded_area_fig3 := (3 * s^2) / 4
  in shaded_area_fig1 ≠ shaded_area_fig2 ∧ shaded_area_fig1 ≠ shaded_area_fig3 ∧ shaded_area_fig2 ≠ shaded_area_fig3 :=
by
  sorry

end shaded_areas_are_different_l713_713136


namespace probability_of_extracting_bacterium_l713_713861

theorem probability_of_extracting_bacterium :
  ∀ (total_volume extracted_volume : ℕ) (bacterium_in_total_volume : Bool),
  total_volume = 100 ∧ extracted_volume = 20 ∧ bacterium_in_total_volume = true →
  let p := (extracted_volume : ℚ) / (total_volume : ℚ) in
  p = 1 / 5 :=
by
  intros total_volume extracted_volume bacterium_in_total_volume h
  cases h with h_total h_rest
  cases h_rest with h_extracted h_bacterium
  rw [h_total, h_extracted]
  let p := (20 : ℚ) / (100 : ℚ)
  calc
    p = (20 : ℚ) / (100 : ℚ) : by rfl
    ... = 1 / 5 : by norm_num

end probability_of_extracting_bacterium_l713_713861


namespace part1_part2_l713_713208

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + 2 * sin x

theorem part1 : f (-π / 6) = -1 / 2 := 
  by sorry

theorem part2 : set.range f = set.Icc (-3 : ℝ) (3 / 2) :=
  by sorry

end part1_part2_l713_713208


namespace max_bishops_on_chessboard_l713_713056

theorem max_bishops_on_chessboard (N : ℕ) (N_pos: 0 < N) : 
  ∃ max_number : ℕ, max_number = 2 * N - 2 :=
sorry

end max_bishops_on_chessboard_l713_713056


namespace domain_and_symmetry_l713_713209

-- Definitions based on conditions:
def f (a x : ℝ) : ℝ := (a * x) / (x + 1) - a

-- Theorem statement based on the equivalent mathematical proof problem:
theorem domain_and_symmetry (a : ℝ) (h : a ≠ 0) :
  (∀ x, x ≠ -1 → ∃ y, f a x = y) ∧ 
  (∀ x, f a (-1 - x) = f a (x - 1) → (1, 0) = (-1, 0)) :=
by
  sorry

end domain_and_symmetry_l713_713209


namespace cosine_120_eq_negative_half_l713_713954

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l713_713954


namespace inequality_solution_set_l713_713385

statement
theorem inequality_solution_set (x : ℝ) : 8 - x^2 > 2x ↔ -4 < x ∧ x < 2 :=
by
  sorry

end inequality_solution_set_l713_713385


namespace coefficient_of_x3_in_expansion_l713_713765

noncomputable def binomial_expansion_coefficient : ℤ :=
let polynomial := (1 - X^2 + 2 * X⁻¹)^7 in
polynomial.coeff (3 : ℤ) -- Coefficient of x^3 in the polynomial expansion

theorem coefficient_of_x3_in_expansion :
  binomial_expansion_coefficient = -910 :=
by sorry

end coefficient_of_x3_in_expansion_l713_713765


namespace paul_Aplus_l713_713745

variable (B+_reward : ℕ) (A_reward : ℕ) (A+_reward : ℕ) (courses : ℕ) (max_reward : ℕ)

def calculate_rewards (x : ℕ) : ℕ :=
  A+_reward * x + 2 * A_reward * (courses - x)

theorem paul_Aplus (B+_reward : ℕ) (A_reward : ℕ) (A+_reward : ℕ) (courses : ℕ) (max_reward : ℕ) :
  B+_reward = 5 → A_reward = 10 → A+_reward = 15 → courses = 10 → max_reward = 190 →
  ∃ (x : ℕ), calculate_rewards B+_reward A_reward A+_reward courses x = max_reward ∧ x = 2 :=
by
  intros
  sorry

end paul_Aplus_l713_713745


namespace boat_speed_in_still_water_l713_713257

theorem boat_speed_in_still_water (B S : ℝ) (h1 : B + S = 13) (h2 : B - S = 9) : B = 11 :=
by
  sorry

end boat_speed_in_still_water_l713_713257


namespace equation_two_solutions_l713_713018

theorem equation_two_solutions (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ = Real.log a (-x₁^2 + 2 * x₁ + a)) ∧ (x₂ = Real.log a (-x₂^2 + 2 * x₂ + a)) :=
sorry

end equation_two_solutions_l713_713018


namespace time_AB_is_3_l713_713443

-- Define the conditions
def distance_AB := 180
def time_BA := 2.5
def saved_time := 0.5
def avg_speed := 80

-- Define the assertion to prove
theorem time_AB_is_3 (t : ℝ) : 
  (360 / (t - saved_time + (time_BA - saved_time)) = avg_speed) → t = 3 := 
by
  sorry

end time_AB_is_3_l713_713443


namespace correctly_calculated_value_l713_713230

theorem correctly_calculated_value : 
  ∃ x : ℝ, (x + 4 = 40) ∧ (x / 4 = 9) :=
sorry

end correctly_calculated_value_l713_713230


namespace complex_product_cis_example_l713_713126

def complex_cis (r : ℝ) (theta : ℝ) : Complex := Complex.polar r (theta * Real.pi / 180)

theorem complex_product_cis_example :
  complex_cis 4 30 * complex_cis 5 (-45) = complex_cis 20 345 :=
sorry

end complex_product_cis_example_l713_713126


namespace find_quotient_l713_713770

theorem find_quotient
  (larger smaller : ℕ)
  (h1 : larger - smaller = 1200)
  (h2 : larger = 1495)
  (rem : ℕ := 4)
  (h3 : larger % smaller = rem) :
  larger / smaller = 5 := 
by 
  sorry

end find_quotient_l713_713770


namespace someone_received_grade_D_or_F_l713_713255

theorem someone_received_grade_D_or_F (m x : ℕ) (hboys : ∃ n : ℕ, n = m + 3) 
  (hgrades_B : ∃ k : ℕ, k = x + 2) (hgrades_C : ∃ l : ℕ, l = 2 * (x + 2)) :
  ∃ p : ℕ, p = 1 ∨ p = 2 :=
by
  sorry

end someone_received_grade_D_or_F_l713_713255


namespace first_two_tanks_need_8_gallons_l713_713733

def fish_tanks_amount (X : ℕ) : Prop :=
  -- conditions:
  -- In one week:
  let amount_week := 2 * X + 2 * (X - 2) in
  -- In four weeks, it amounts to 112 gallons:
  4 * amount_week = 112

theorem first_two_tanks_need_8_gallons:
  ∃ X : ℕ, fish_tanks_amount X ∧ X = 8 :=
by
  sorry

end first_two_tanks_need_8_gallons_l713_713733


namespace greatest_common_divisor_l713_713411

theorem greatest_common_divisor (n : ℕ) : 
  (∀ d, d ∣ 120 ∧ d ∣ n → d = 1 ∨ d = 2 ∨ d = 4 ∨ d = 8) → 
  let common_divisors : finset ℕ := {1, 2, 4, 8}
  in common_divisors.max = some 8 := by
  sorry

end greatest_common_divisor_l713_713411


namespace solve_cubic_equation_l713_713760

theorem solve_cubic_equation (z : ℂ) (a b : ℝ) (h1 : z = a + b * Complex.i)
  (h2 : z * (z + 1) * (z + 3) = 2018) : 
  ∃ (a : ℝ), ∃ (b : ℝ), 
  a ∈ {10, 12, 15, 18, 20} ∧
  (a * (a^2 + 4 * a + 3 - b^2) = 2018) ∧
  (a * (2 * a + 4) * b = 0) := 
sorry

end solve_cubic_equation_l713_713760


namespace cone_generatrix_l713_713841

theorem cone_generatrix (cone : Type) (vertex : cone) (base : set cone)
(line_segment : cone → cone → cone) :
(∀ p : cone, p ∈ base → ∃ l : cone, l = line_segment vertex p) :=
sorry

end cone_generatrix_l713_713841


namespace inverse_exponential_l713_713199

theorem inverse_exponential :
  ∀ (x : ℝ), x > 0 → (∃ f : ℝ → ℝ, ∀ y : ℝ, y = e^x ↔ x = f y ∧ y > 0) → (f x = log x) :=
begin
  sorry
end

end inverse_exponential_l713_713199


namespace find_integers_l713_713806

theorem find_integers (a b c : ℤ) (h1 : {a, b, c} = {a - 1, b + 1, c * c}) (h2 : a + b + c = 2013) :
  (a = 1007 ∧ b = 1006 ∧ c = 0) ∨ (a = 1006 ∧ b = 1007 ∧ c = 0) :=
by
  sorry

end find_integers_l713_713806


namespace min_guests_l713_713923

/-- Problem statement:
Given:
1. The total food consumed by all guests is 319 pounds.
2. Each guest consumes no more than 1.5 pounds of meat, 0.3 pounds of vegetables, and 0.2 pounds of dessert.
3. Each guest has equal proportions of meat, vegetables, and dessert.

Prove:
The minimum number of guests such that the total food consumed is less than or equal to 319 pounds is 160.
-/
theorem min_guests (total_food : ℝ) (meat_per_guest : ℝ) (veg_per_guest : ℝ) (dessert_per_guest : ℝ) (G : ℕ) :
  total_food = 319 ∧ meat_per_guest ≤ 1.5 ∧ veg_per_guest ≤ 0.3 ∧ dessert_per_guest ≤ 0.2 ∧
  (meat_per_guest + veg_per_guest + dessert_per_guest = 2.0) →
  G = 160 :=
by
  intros h
  sorry

end min_guests_l713_713923


namespace adam_action_figures_l713_713906

theorem adam_action_figures :
  ∀ (shelf1 shelf2 shelf3 new_shelf : ℕ),
  shelf1 = 9 →
  shelf2 = 14 →
  shelf3 = 7 →
  new_shelf = 11 →
  3 * new_shelf + (shelf1 + shelf2 + shelf3) = 52 :=
by
  intros shelf1 shelf2 shelf3 new_shelf
  intro hs1
  intro hs2
  intro hs3
  intro hns
  rw [hs1, hs2, hs3, hns]
  calc
    9 + 14 + 7 = 30 := by norm_num
    2 * 11 = 22 := by norm_num
    30 + 22 = 52 := by norm_num
  sorry

end adam_action_figures_l713_713906


namespace alfred_gain_percent_l713_713448

-- Definitions based on the conditions
def purchase_price : ℝ := 4700
def repair_costs : ℝ := 800
def selling_price : ℝ := 6000

-- Lean statement to prove gain percent
theorem alfred_gain_percent :
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 9.09 := by
  sorry

end alfred_gain_percent_l713_713448


namespace remainder_zero_l713_713163

-- Definition of polynomials
def p1 := X^55 + X^44 + X^33 + X^22 + X^11 + 1
def p2 := X^5 + X^4 + X^3 + X^2 + X + 1

-- The statement of the theorem
theorem remainder_zero : p1 % p2 = 0 := 
by {
  sorry
}

end remainder_zero_l713_713163


namespace relationship_among_a_b_c_l713_713686

-- Define y = f x as an odd function and introduce the given conditions
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f(-x) = -f(x))
variable (h_ineq : ∀ x, x < 0 → f(x) + x * (deriv f x) < 0)

-- Define a = 3 * f(3), b = -2 * f(-2), and c = f(1)
def a := 3 * f 3
def b := -2 * f (-2)
def c := f 1

-- Prove the correct relationship among a, b, and c
theorem relationship_among_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_among_a_b_c_l713_713686


namespace geometric_sequence_root_product_l713_713661

theorem geometric_sequence_root_product
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (a1_pos : 0 < a 1)
  (a19_root : a 1 * r^18 = (1 : ℝ))
  (h_poly : ∀ x, x^2 - 10 * x + 16 = 0) :
  a 8 * a 12 = 16  :=
sorry

end geometric_sequence_root_product_l713_713661


namespace infinite_primes_satisfying_condition_l713_713805

theorem infinite_primes_satisfying_condition :
  ∃ (S : Set ℕ), (∀ (n : ℕ), n ∈ S → Prime n) ∧
  (∀ (p : ℕ), p ∈ S → ∃ n : ℕ, sqrt (p + n) + sqrt (n) ∈ ℕ) ∧
  Infinite S :=
by
  sorry

end infinite_primes_satisfying_condition_l713_713805


namespace total_pages_read_l713_713688

-- Definitions of the conditions
def pages_read_by_jairus : ℕ := 20

def pages_read_by_arniel : ℕ := 2 + 2 * pages_read_by_jairus

-- The statement to prove the total number of pages read by both is 62
theorem total_pages_read : pages_read_by_jairus + pages_read_by_arniel = 62 := by
  sorry

end total_pages_read_l713_713688


namespace probability_of_x_plus_1_lt_y_l713_713103

open Set Function

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

-- Define the event where x + 1 < y
def event_x_plus_1_lt_y : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + 1 < p.2}

-- Calculate the area of the rectangle
def area_rectangle := 12

-- Calculate the area of the triangle within the rectangle where x + 1 < y
def area_triangle := 1 / 2

-- Define probability as the ratio of the areas
def probability_event := area_triangle / area_rectangle

-- Theorem statement
theorem probability_of_x_plus_1_lt_y : 
  ∀ (x y : ℝ), (x, y) ∈ rectangle → (x, y) ∈ event_x_plus_1_lt_y = 
  probability_event = 1 / 24 :=
by
  sorry

end probability_of_x_plus_1_lt_y_l713_713103


namespace value_of_m_l713_713082

theorem value_of_m :
  ∃ m : ℕ, 5 ^ m = 5 * 25 ^ 2 * 125 ^ 3 ∧ m = 14 :=
begin
  -- Solution steps would be here
  sorry
end

end value_of_m_l713_713082


namespace ratio_of_liquid_level_rises_l713_713042

theorem ratio_of_liquid_level_rises (r1 r2 : ℝ) (h1 h2 : ℝ) (V : ℝ) (marble_vol : ℝ) :
  r1 = 5 ∧ r2 = 10 ∧ V = (25 / 3) * π * h1 ∧ V = (100 / 3) * π * h2 ∧ marble_vol = (32 / 3) * π → 
  h1 / h2 = 4 →
  let Δh1 := (marble_vol * 3 / (25 * π))
  let Δh2 := (marble_vol * 3 / (100 * π)) in
  Δh1 / Δh2 = 4 := 
by
  intros
  simp only [*, mul_eq_mul_right_iff, eq_self_iff_true, true_and]
  field_simp
  norm_num
  field_simp
  norm_num
  done

end ratio_of_liquid_level_rises_l713_713042


namespace product_of_x1_to_x13_is_zero_l713_713368

theorem product_of_x1_to_x13_is_zero
  (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ)
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 :=
sorry

end product_of_x1_to_x13_is_zero_l713_713368


namespace hershel_remaining_fish_l713_713637

def hershel_initial_betta : ℕ := 10
def hershel_initial_gold : ℕ := 15
def bexley_betta_ratio : ℚ := 2 / 5
def bexley_gold_ratio : ℚ := 1 / 3
def gifting_fraction : ℚ := 1 / 2

theorem hershel_remaining_fish :
  let betta_brought := (bexley_betta_ratio * hershel_initial_betta).to_nat,
      gold_brought := (bexley_gold_ratio * hershel_initial_gold).to_nat,
      total_betta := hershel_initial_betta + betta_brought,
      total_gold := hershel_initial_gold + gold_brought,
      total_fish := total_betta + total_gold,
      gifting_num := (gifting_fraction * total_fish).to_nat,
      remaining_fish := total_fish - gifting_num
  in remaining_fish = 17 := by
  -- Proof goes here
  sorry

end hershel_remaining_fish_l713_713637


namespace part1_union_part1_complement_part2_intersect_l713_713193

namespace MathProof

open Set Real

def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }
def R : Set ℝ := univ  -- the set of all real numbers

theorem part1_union :
  A ∪ B = { x | 1 ≤ x ∧ x < 10 } :=
sorry

theorem part1_complement :
  R \ B = { x | x ≤ 2 ∨ x ≥ 10 } :=
sorry

theorem part2_intersect (a : ℝ) :
  (A ∩ C a ≠ ∅) → a > 1 :=
sorry

end MathProof

end part1_union_part1_complement_part2_intersect_l713_713193


namespace number_of_smaller_cubes_l713_713479

theorem number_of_smaller_cubes (l : ℕ) (h_l : l = 4) (h1 : ∀ n : ℕ, n > 0 → n ≤ l → (∃ m : ℕ, l % m = 0)) :
  let N := 57 in N = 57 :=
by
  sorry

end number_of_smaller_cubes_l713_713479


namespace BF_perp_KL_l713_713298

-- We declare all given points, lines, and geometric properties using variables and lemmas.

variables (A B C D K L F : Type) [rectABCD : is_rectangle A B C D]
variables (p1 p2 : line) (h1 : passes_through B p1) (h2 : passes_through B p2)
variables (p1_perp_p2 : is_perpendicular p1 p2)
variables (K_on_AD : lies_on_segment K A D p1)
variables (L_on_CDExt : lies_on_extension L C D p2)
variables (F_inter_KL_AC : intersection_point F KL AC)

-- Statement of the problem to prove BF perpendicular to KL

theorem BF_perp_KL : is_perpendicular (line_through B F) (line_through K L) := 
sorry

end BF_perp_KL_l713_713298


namespace certain_number_minus_15_l713_713433

theorem certain_number_minus_15 (n : ℕ) (h : n / 10 = 6) : n - 15 = 45 :=
sorry

end certain_number_minus_15_l713_713433


namespace cos_120_degrees_eq_l713_713948

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l713_713948


namespace symmetric_point_xOz_l713_713388

def symmetric_point (p : ℝ × ℝ × ℝ) (plane : String) : ℝ × ℝ × ℝ :=
  match plane with
  | "xOz" => (p.1, -p.2, p.3)
  | _ => p

theorem symmetric_point_xOz (x y z : ℝ) (h : (x, y, z) = (2, 3, 4)) : symmetric_point (x, y, z) "xOz" = (2, -3, 4) :=
by
  -- Proof omitted.
  sorry

end symmetric_point_xOz_l713_713388


namespace find_rate_percent_l713_713058

-- Given conditions as definitions
def SI : ℕ := 128
def P : ℕ := 800
def T : ℕ := 4

-- Define the formula for Simple Interest
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Define the rate percent we need to prove
def rate_percent : ℕ := 4

-- The theorem statement we need to prove
theorem find_rate_percent (h1 : simple_interest P rate_percent T = SI) : rate_percent = 4 := 
by sorry

end find_rate_percent_l713_713058


namespace first_nonzero_digit_one_over_157_l713_713420

theorem first_nonzero_digit_one_over_157 : 
  (∃ d : ℕ, d < 10 ∧ (∃ n : ℕ, (1000 / 157 : ℝ) = (6 + d * 10^-1 + n * 10^-2) * 157)) →
  d = 3 :=
by sorry

end first_nonzero_digit_one_over_157_l713_713420


namespace floor_square_minus_square_floor_l713_713551

theorem floor_square_minus_square_floor (x : ℝ) (h : x = 13.2) :
  (⌊ x^2 ⌋ - (⌊ x ⌋ * ⌊ x ⌋) = 5) :=
by
  have h1 : x^2 = 174.24 := by sorry
  have h2 : ⌊ x^2 ⌋ = 174 := by sorry
  have h3 : ⌊ x ⌋ = 13 := by sorry
  have h4 : ⌊ x ⌋ * ⌊ x ⌋ = 13 * 13 := by sorry
  show 174 - 169 = 5, from sorry

end floor_square_minus_square_floor_l713_713551


namespace central_angle_of_unfolded_cone_l713_713855

theorem central_angle_of_unfolded_cone
  (r : ℝ) 
  (h1 : r > 0)
  (l : ℝ) 
  (h2 : l = real.sqrt 2 * r) :
  ∃ (α : ℝ), α = real.sqrt 2 * real.pi :=
by 
  sorry

end central_angle_of_unfolded_cone_l713_713855


namespace find_f_neg_a_l713_713615

def f (x : ℝ) := log 2 (2 * (1 + x) / (x - 1))

theorem find_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end find_f_neg_a_l713_713615


namespace smallest_possible_sum_of_squares_l713_713007

theorem smallest_possible_sum_of_squares : 
  ∃ (x y : ℕ), (x^2 - y^2 = 221) ∧ (∀(a b : ℕ), (a^2 - b^2 = 221) → (x^2 + y^2 ≤ a^2 + b^2)) ∧ (x^2 + y^2 = 229) :=
begin
  sorry, -- Proof omitted as per instructions
end

end smallest_possible_sum_of_squares_l713_713007


namespace max_smallest_barrel_capacity_l713_713088

theorem max_smallest_barrel_capacity (A B C D : ℕ) (h : [A, B, C, D].sorted) (avg_capacity : A + B + C + D = 32) (median_capacity : B + C = 20) : A ≤ 2 :=
by {
  sorry
}

end max_smallest_barrel_capacity_l713_713088


namespace probability_odd_product_in_each_row_l713_713089

-- Define the set of numbers and the grid
def numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def grid := {g : (Fin 3 × Fin 3) → ℕ // (∀ i j, g ⟨i, j⟩ ∈ numbers) ∧ (∃ f: ℕ → ℕ → Prop, bijective f)}

-- Prove that the probability that the product of numbers in each row is odd is zero
theorem probability_odd_product_in_each_row : 
  (probability (λ (g : grid), ∀ i, (g.val ⟨i, 0⟩ * g.val ⟨i, 1⟩ * g.val ⟨i, 2⟩) % 2 = 1) = 0) := 
sorry

end probability_odd_product_in_each_row_l713_713089


namespace C_plus_D_l713_713305

theorem C_plus_D (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) : 
  C + D = -10 := by
  sorry

end C_plus_D_l713_713305


namespace max_points_in_grid_l713_713133

noncomputable def max_points (n : ℕ) : ℕ := 3 * n - 3

theorem max_points_in_grid (n : ℕ) (h : n = 2011) :
  ∃ (points : ℕ), points = max_points n :=
by
  use max_points n
  simp [max_points, h]
  sorry

end max_points_in_grid_l713_713133


namespace katya_solves_enough_l713_713292

theorem katya_solves_enough (x : ℕ) :
  (0 ≤ x ∧ x ≤ 20) → -- x should be within the valid range of problems
  (4 / 5) * x + (1 / 2) * (20 - x) ≥ 13 → 
  x ≥ 10 :=
by 
  intros h₁ h₂
  -- Formalize the expected value equation and the inequality transformations
  sorry

end katya_solves_enough_l713_713292


namespace quadrilateral_is_rhombus_if_perpendicular_diagonals_l713_713434

variable (A B C D : Type) [EuclideanGeometry A B C D]
variables (AC BD : A B C D → ℝ) (AB BC : A B → ℝ)
variables (AD BD : A D → ℝ)

def is_parallelogram (ABCD : Quadrilateral A B C D) : Prop :=
  -- Definitions related to parallelogram properties would go here
  sorry

def is_rhombus (ABCD : Quadrilateral A B C D) : Prop :=
  -- Definitions related to rhombus properties would go here
  sorry

theorem quadrilateral_is_rhombus_if_perpendicular_diagonals
  (ABCD : Quadrilateral A B C D) (H_parallelogram : is_parallelogram ABCD) 
  (H_perpendicular_diagonals : perpendicular AC BD) : is_rhombus ABCD := by
  sorry

end quadrilateral_is_rhombus_if_perpendicular_diagonals_l713_713434


namespace birch_trees_probability_l713_713873

theorem birch_trees_probability (m n : ℕ) 
  (h : (m, n).gcd = 1) 
  (p : (m : ℚ) / n = 56 / 792) : 
  m + n = 106 :=
by
  have h1 : 56 / 792 = 7 / 99, by norm_num,
  rw ← h1 at p,
  have h2 : (7, 99).gcd = 1, by norm_num,
  sorry

end birch_trees_probability_l713_713873


namespace time_to_cross_bridge_l713_713222

def length_of_train : ℕ := 110
def speed_km_per_hr : ℕ := 90
def length_of_bridge : ℕ := 132

noncomputable def speed_m_per_s : ℚ := (speed_km_per_hr * 1000) / 3600
noncomputable def total_distance : ℕ := length_of_train + length_of_bridge

theorem time_to_cross_bridge :
  (total_distance : ℚ) / speed_m_per_s ≈ 9.68 := 
sorry

end time_to_cross_bridge_l713_713222


namespace almostPalindromeCount_l713_713512

def isAlmostPalindrome (s : List Char) : Prop :=
  let rev := s.reverse
  (List.foldl (fun acc (x : ℕ × Char) => if x.2 = rev.nthLe x.1 (by { simp [List.length_reverse] }) then acc else acc + 1) 0 (List.zip (List.range s.length) s)) = 2

def countAlmostPalindromes (letters : List Char) (target : ℕ) : Prop :=
  ∃ l : List (List Char), (∀ x ∈ l, x.perm letters ∧ isAlmostPalindrome x) ∧ l.length = target

theorem almostPalindromeCount :
  countAlmostPalindromes ['H', 'M', 'M', 'T', 'T', 'H', 'E', 'M', 'E', 'T', 'E', 'A', 'M'] 2160 :=
sorry

end almostPalindromeCount_l713_713512


namespace _l713_713092

noncomputable def circle_centered_at_A_with_radius_1 : Prop := 
∀ (A : Type) [MetricSpace A], ∃ (C : A), dist C A = 1

noncomputable def circle_centered_at_B_with_radius_4 : Prop := 
∀ (B : Type) [MetricSpace B], ∃ (D : B), dist D B = 4

noncomputable def circles_A_and_B_are_externally_tangent : Prop := 
∀ (A B : Type) [MetricSpace A] [MetricSpace B], 
∃ (C D: Type), dist C A + dist D B = 5

noncomputable def third_circle_is_tangent_to_first_two_and_common_external_tangent : Prop :=
∀ (A B: Type) [MetricSpace A] [MetricSpace B], 
∃ (E: Type), (dist E A = x + 1) ∧ (dist E B = x + 4) ∧ 
(dist (common_external_point E) = x + (x + 4) + x)

noncomputable theorem radius_of_third_circle 
  (A B: Type) [MetricSpace A] [MetricSpace B] 
  (h1: circle_centered_at_A_with_radius_1) 
  (h2: circle_centered_at_B_with_radius_4) 
  (h3: circles_A_and_B_are_externally_tangent A B)
  (h4: third_circle_is_tangent_to_first_two_and_common_external_tangent A B) : 
  ∃ (x : ℝ), x = 4 / 9 :=
sorry

end _l713_713092


namespace exist_n_pairs_l713_713804

theorem exist_n_pairs 
  (n : ℕ)
  (p : ℝ → ℝ)
  (h_cont : ContinuousOn p (Icc 0 n))
  (h_boundary : p 0 = p n) : 
  ∃ S : Finset ((ℝ × ℝ)), S.card = n ∧ ∀ (a b : ℝ), (a, b) ∈ S → p a = p b ∧ ∃ k : ℕ, k > 0 ∧ b - a = k :=
by
  sorry

end exist_n_pairs_l713_713804


namespace toy_poodle_height_l713_713386

noncomputable def height_of_toy_poodle (std_ht mini_ht toy_ht : ℝ) : ℝ :=
  let mini_ht := toy_ht + 6.25
  let std_ht := mini_ht + 8.5
  toy_ht

theorem toy_poodle_height (std_ht : ℝ) (h1 : std_ht = 28) (h2 : ∀ (toy_ht : ℝ), std_ht = toy_ht + 6.25 + 8.5) : ∃ (toy_ht : ℝ), toy_ht = 13.25 :=
  by
  exists 13.25
  rw [h1, h2]
  linarith

end toy_poodle_height_l713_713386


namespace minimum_edges_to_form_triangle_l713_713577

theorem minimum_edges_to_form_triangle (n : ℕ) (h : n = 2014) : 
  ∃ G : SimpleGraph (Fin n), 
  (∀ v w : Fin n, v ≠ w → ∃ u : Fin n, u ≠ v ∧ u ≠ w ∧ G.Adj u v ∧ G.Adj u w) ∧ 
  G.edgeCount = n - 1 := by
    sorry

end minimum_edges_to_form_triangle_l713_713577


namespace destiny_cookies_divisible_l713_713141

theorem destiny_cookies_divisible (C : ℕ) (h : C % 6 = 0) : ∃ k : ℕ, C = 6 * k :=
by {
  sorry
}

end destiny_cookies_divisible_l713_713141


namespace smallest_positive_x_for_palindrome_l713_713826

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem smallest_positive_x_for_palindrome :
  ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 8901) ∧ ∀ y : ℕ, y > 0 ∧ is_palindrome (y + 8901) → x ≤ y :=
  by
  let x := 108
  existsi x
  split
  · exact nat.lt_succ_self 0
  split
  · sorry -- Proof that 9009 is a palindrome
  · intros y hy1 hy2
    sorry -- Proof that 108 is the smallest such x

end smallest_positive_x_for_palindrome_l713_713826


namespace cos_120_eq_neg_half_l713_713967

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l713_713967


namespace count_integers_of_form_ab40_divisible_by_40_l713_713228

theorem count_integers_of_form_ab40_divisible_by_40 :
  let count := (λ (n : ℕ), n >= 1000 ∧ n < 10000 ∧ n % 100 = 40 ∧ n % 40 = 0) in
  (Finset.filter count (Finset.range 10000)).card = 11 := 
by
  sorry

end count_integers_of_form_ab40_divisible_by_40_l713_713228


namespace equivalent_area_l713_713405

-- Define the points A, B, C, D, and E on a plane
variables {A B C D E F : Type} [Point A B C D E F]

-- Define the pentagon ABCDE
def pentagon : Polygon := {vertices := [A, B, C, D, E], sides := sorry}

-- Define the diagonal EC
def diagonal_EC : Line := Line.mk E C

-- Define the line FD which is parallel to EC through point D
def line_FD : Line := Line.parallel_through_point diagonal_EC D

-- Define the intersection F as the point where line FD intersects the extension of AE
def point_F : Point := Line.intersect (Line.extend (Line.mk A E)) line_FD

-- Define the quadrilateral ABCF that is supposedly equal in area to pentagon ABCDE
def quadrilateral : Set Point := {A, B, C, F}

-- The theorem we want to prove: quadrilateral ABCF has the same area as pentagon ABCDE
theorem equivalent_area (A B C D E F : Point) 
  (pent := pentagon)
  (diag_EC := diagonal_EC)
  (parallel_FD := line_FD)
  (intersection_F := point_F)
  (quad := quadrilateral):
  area quad = area pent := 
sorry

end equivalent_area_l713_713405


namespace initial_average_customers_l713_713124

theorem initial_average_customers (x A : ℕ) (h1 : x = 1) (h2 : (A + 120) / 2 = 90) : A = 60 := by
  sorry

end initial_average_customers_l713_713124


namespace relatively_prime_sequence_l713_713719

noncomputable def f : ℤ → ℤ := λ x, x^2 - x + 1

def a_seq (m : ℤ) (n : ℕ) : ℤ :=
  Nat.recOn n m (λ n a_n_1, f a_n_1)

theorem relatively_prime_sequence (m : ℤ) (h : m > 1) :
  ∀ i j : ℕ, i ≠ j → Nat.gcd (a_seq m i).natAbs (a_seq m j).natAbs = 1 := 
sorry

end relatively_prime_sequence_l713_713719


namespace walking_speed_ratio_l713_713555

def conditions (v_walk v_car : ℝ) : Prop :=
  let d := v_walk * 60 in
  d = v_car * 5

theorem walking_speed_ratio (v_walk v_car : ℝ) (h : conditions v_walk v_car) : v_car / v_walk = 12 :=
by
  have eq1 : v_walk * 60 = v_car * 5 := h
  have eq2 : v_car = v_walk * (60 / 5) := by
    linarith
  have eq3 : v_car = v_walk * 12 := by
    norm_num at eq2
    exact eq2
  have ratio : v_car / v_walk = 12 := by
    rw [eq3]
    field_simp
  exact ratio

end walking_speed_ratio_l713_713555


namespace area_of_rhombus_l713_713050

theorem area_of_rhombus (x y : ℝ) (h : |2 * x| + |3 * y| = 12) : 
  let d1 := 12 in
  let d2 := 8 in
  0.5 * d1 * d2 = 48 := 
by
  sorry

end area_of_rhombus_l713_713050


namespace total_pages_read_l713_713687

-- Definitions of the conditions
def pages_read_by_jairus : ℕ := 20

def pages_read_by_arniel : ℕ := 2 + 2 * pages_read_by_jairus

-- The statement to prove the total number of pages read by both is 62
theorem total_pages_read : pages_read_by_jairus + pages_read_by_arniel = 62 := by
  sorry

end total_pages_read_l713_713687


namespace broadcast_sequence_count_l713_713862

-- Define the conditions as a set of assumptions.
section
variables
  (ads: Finset String) -- Set of advertisements
  (comm: Finset String) -- Set of commercial ads
  (olym: Finset String) -- Set of Olympic promotional ads
  (psa: String) -- Public service ad
  (seqs: Finset (List String)) -- Set of valid sequences

  (h_ads : ads = {'ad1', 'ad2', 'ad3', 'ad4', 'ad5', 'ad6})
  (h_comm : comm = {'c1', 'c2', 'c3})
  (h_olym : olym = {'o1', 'o2})
  (h_psa : psa = 'p1')
  (h_last_not_comm : ∀ seq ∈ seqs, ¬(List.last seq = 'c1' ∨ List.last seq = 'c2' ∨ List.last seq = 'c3'))
  (h_olym_not_consec : ∀ seq ∈ seqs, ¬(List.inits seq)).any (λ l, List.take 2 l ∈ List.permutations ['o1', 'o2'])
  (h_olym_psa_not_consec : ∀ seq ∈ seqs, ¬(List.inits seq)).any (λ l, List.take 2 l ∈ List.permutations ['o1', 'p1'] ∨ List.take 2 l ∈ List.permutations ['o2', 'p1'])

-- State the main theorem
theorem broadcast_sequence_count : Finset.card seqs = 99 := 
  sorry
end

end broadcast_sequence_count_l713_713862


namespace cos_120_degrees_l713_713993

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l713_713993


namespace cost_price_of_article_l713_713329

variable (CP : ℝ)

def MP : ℝ := 1.15 * CP
def SP : ℝ := 0.7407 * MP

theorem cost_price_of_article
  (h1 : SP = 460) : CP = 460 / (0.7407 * 1.15) :=
sorry

end cost_price_of_article_l713_713329


namespace sum_of_squares_eq_229_l713_713000

-- The conditions
variables (x y : ℤ)
axiom diff_eq_221 : x^2 - y^2 = 221

-- The proof goal
theorem sum_of_squares_eq_229 : x^2 - y^2 = 221 → ∃ x y : ℤ, x^2 + y^2 = 229 :=
by
  sorry

end sum_of_squares_eq_229_l713_713000


namespace systematic_sampling_method_l713_713492

-- Define the condition stating the auditorium has 25 rows and 20 seats per row.
def auditorium_rows : ℕ := 25
def seats_per_row : ℕ := 20

-- Define the condition that the auditorium was filled with students.
def auditorium_filled : Prop := true

-- Define the condition that 25 students with seat number 15 were kept for testing.
def seats_for_testing : (ℕ × ℕ) → Prop := λ seat, seat.1 ≤ 25 ∧ seat.2 = 15

theorem systematic_sampling_method :
  ∃ (sampling_method : String),
  sampling_method = "Systematic sampling method" ∧
  auditorium_rows = 25 ∧ seats_per_row = 20 ∧ 
  auditorium_filled ∧
  (∀ seat, seats_for_testing seat → seat.2 = 15) :=
sorry

end systematic_sampling_method_l713_713492


namespace money_needed_to_finish_collection_l713_713692

-- Define the conditions
def initial_action_figures : ℕ := 9
def total_action_figures_needed : ℕ := 27
def cost_per_action_figure : ℕ := 12

-- Define the goal
theorem money_needed_to_finish_collection 
  (initial : ℕ) (total_needed : ℕ) (cost_per : ℕ) 
  (h1 : initial = initial_action_figures)
  (h2 : total_needed = total_action_figures_needed)
  (h3 : cost_per = cost_per_action_figure) :
  ((total_needed - initial) * cost_per = 216) := 
by
  sorry

end money_needed_to_finish_collection_l713_713692


namespace arithmetic_sum_cos_sequence_l713_713187

theorem arithmetic_sum_cos_sequence (a : ℕ → ℝ) (cos : ℕ → ℝ) (n : ℕ) 
  (h1 : a 1 + a 8 = 2 * a 5 - 2)
  (h2 : a 3 + a 11 = 26)
  (hcos1 : cos = λ n, if n % 2 = 0 then 1 else -1) :
  ∑ i in Finset.range 2022, a (i + 1) * cos (i + 1) = 2022 :=
sorry

end arithmetic_sum_cos_sequence_l713_713187


namespace cistern_problem_l713_713093

theorem cistern_problem (x : ℝ) (h1 : 7 ≠ 0) (h2 : 31.5 ≠ 0)
  (fill_rate_first : 1 / 7) (combined_rate : 1 / 31.5) :
  (fill_rate_first - (1 / x) = combined_rate) → x = 9 := by
  sorry

end cistern_problem_l713_713093


namespace desired_cost_per_pound_l713_713870
-- Importing the necessary library

-- Defining the candy weights and their costs per pound
def weight1 : ℝ := 20
def cost_per_pound1 : ℝ := 8
def weight2 : ℝ := 40
def cost_per_pound2 : ℝ := 5

-- Defining the proof statement
theorem desired_cost_per_pound :
  let total_cost := (weight1 * cost_per_pound1 + weight2 * cost_per_pound2)
  let total_weight := (weight1 + weight2)
  let desired_cost := total_cost / total_weight
  desired_cost = 6 := sorry

end desired_cost_per_pound_l713_713870


namespace select_people_english_japanese_l713_713481

-- Definitions based on conditions
def total_people : ℕ := 9
def english_speakers : ℕ := 7
def japanese_speakers : ℕ := 3

-- Theorem statement
theorem select_people_english_japanese (h1 : total_people = 9) 
                                      (h2 : english_speakers = 7) 
                                      (h3 : japanese_speakers = 3) :
  ∃ n, n = 20 :=
by {
  sorry
}

end select_people_english_japanese_l713_713481


namespace floor_square_minus_square_floor_l713_713552

theorem floor_square_minus_square_floor (x : ℝ) (h : x = 13.2) :
  (⌊ x^2 ⌋ - (⌊ x ⌋ * ⌊ x ⌋) = 5) :=
by
  have h1 : x^2 = 174.24 := by sorry
  have h2 : ⌊ x^2 ⌋ = 174 := by sorry
  have h3 : ⌊ x ⌋ = 13 := by sorry
  have h4 : ⌊ x ⌋ * ⌊ x ⌋ = 13 * 13 := by sorry
  show 174 - 169 = 5, from sorry

end floor_square_minus_square_floor_l713_713552


namespace minimum_value_of_f_l713_713375

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then x^2 - 1 else x + 6 / x - 7

theorem minimum_value_of_f : (x : ℝ) (h_x : x > 0) ∃ x₀, f x₀ = 2 * Real.sqrt 6 - 7 :=
by
  sorry

end minimum_value_of_f_l713_713375


namespace part1_part2_l713_713207

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * (Real.sin x) ^ 2

theorem part1 : f (Real.pi / 6) = 1 / 2 :=
by
  sorry

theorem part2 : 
  ∃ (M m : ℝ), 
  (∀ x ∈ set.Icc (-Real.pi / 4) (Real.pi / 6), f x ≤ M) ∧
  (∀ x ∈ set.Icc (-Real.pi / 4) (Real.pi / 6), f x ≥ m) ∧
  (∀ x ∈ set.Icc (-Real.pi / 4) (Real.pi / 6), M = f x → x = 0) ∧
  (∀ x ∈ set.Icc (-Real.pi / 4) (Real.pi / 6), m = f x → x = -Real.pi / 4) :=
by
  sorry

end part1_part2_l713_713207


namespace an_formula_Tn_formula_l713_713894

variables {a : ℕ → ℝ} {b : ℕ → ℝ} {c : ℕ → ℝ}

-- Definitions given in the problem
def Sn (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def bn (n : ℕ) : ℝ := ∑ i in finset.range n, real.log (a (i + 1)) / real.log 2

-- Conditions from the problem
axiom a_eq : ∀ n ≥ 2, a n = 2 * a (n - 1)
axiom arith_seq : 2 * (a 3 + 1) = a 1 + a 4
axiom a1_val : a 1 = 2

-- The sequence c_n
def cn (n : ℕ) : ℝ := 1 / bn n + (-1)^n * a n

-- The sum of the first n terms of c_n
def Tn (n : ℕ) : ℝ := ∑ i in finset.range n, cn (i + 1)

-- Proof that a_n = 2^n
theorem an_formula (n : ℕ) : a n = 2^n :=
sorry

-- Proof for T_n = 4/3 - 2/(n+1) - (1/3)(-2)^(n+1)
theorem Tn_formula (n : ℕ) : Tn n = 4/3 - 2/(n + 1) - (1/3) * ((-2)^(n + 1)) :=
sorry

end an_formula_Tn_formula_l713_713894


namespace relationship_of_points_l713_713652

variable (y k b x : ℝ)
variable (y1 y2 : ℝ)

noncomputable def linear_func (x : ℝ) : ℝ := k * x - b

theorem relationship_of_points
  (h_pos_k : k > 0)
  (h_point1 : linear_func k b (-1) = y1)
  (h_point2 : linear_func k b 2 = y2):
  y1 < y2 := 
sorry

end relationship_of_points_l713_713652


namespace coefficient_x2_l713_713541

theorem coefficient_x2 : 
  let expr := 5 * (λ x => x - 2 * x ^ 3) 
          - 4 * (λ x => 2 * x ^ 2 - 3 * x ^ 3 + x ^ 6) 
          + 3 * (λ x => 3 * x ^ 2 - x ^ 10) 
  in
  (expr 0).coeff (2 : ℕ) = 1 :=
by 
  let term1 := 5 * (λ x => x - 2 * x ^ 3)
  let term2 := -4 * (λ x => 2 * x ^ 2 - 3 * x ^ 3 + x ^ 6)
  let term3 := 3 * (λ x => 3 * x ^ 2 - x ^ 10)
  let coeff_x2 := term1.coeff 2 + term2.coeff 2 + term3.coeff 2
  have h1 : term1.coeff 2 = 0 := by sorry
  have h2 : term2.coeff 2 = -8 := by sorry
  have h3 : term3.coeff 2 = 9 := by sorry
  show coeff_x2 = 0 + (-8) + 9 => by sorry

end coefficient_x2_l713_713541


namespace area_of_rhombus_l713_713048

theorem area_of_rhombus (x y : ℝ) (h : |2 * x| + |3 * y| = 12) : 
  let d1 := 12 in
  let d2 := 8 in
  0.5 * d1 * d2 = 48 := 
by
  sorry

end area_of_rhombus_l713_713048


namespace greatest_s_property_l713_713699

noncomputable def find_greatest_s (m n : ℕ) (p : ℕ) [Fact (Nat.Prime p)] : ℕ :=
if h : m > 0 ∧ n > 0 then m else 0

theorem greatest_s_property (m n : ℕ) (p : ℕ) [Fact (Nat.Prime p)] (H : 0 < m) (H1 : 0 < n)  :
  ∃ s, (s = find_greatest_s m n p) ∧ s * n * p ≤ m * n * p :=
by 
  sorry

end greatest_s_property_l713_713699


namespace taps_fill_sink_time_l713_713450

theorem taps_fill_sink_time (t1 t2 t_comb : ℝ) (r1 r2 r_comb : ℝ) :
  t1 = 210 ∧ t2 = 214 ∧ r1 = 1 / t1 ∧ r2 = 1 / t2 ∧ r_comb = r1 + r2 ∧ t_comb = 1 / r_comb →
  t_comb ≈ 106 :=
by {
  sorry -- Steps for computational proof would go here
}

end taps_fill_sink_time_l713_713450


namespace cos_120_eq_neg_half_l713_713969

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l713_713969


namespace john_finishes_ahead_l713_713279

noncomputable def InitialDistanceBehind : ℝ := 12
noncomputable def JohnSpeed : ℝ := 4.2
noncomputable def SteveSpeed : ℝ := 3.7
noncomputable def PushTime : ℝ := 28

theorem john_finishes_ahead :
  (JohnSpeed * PushTime - InitialDistanceBehind) - (SteveSpeed * PushTime) = 2 := by
  sorry

end john_finishes_ahead_l713_713279


namespace mass_percentage_oxygen_in_mixture_l713_713143

def mass_percentage_oxygen 
  (mol_HBrO3 : ℕ := 1)
  (mol_H2SO3 : ℕ := 2)
  (molar_mass_H : ℝ := 1.01)
  (molar_mass_Br : ℝ := 79.90)
  (molar_mass_S : ℝ := 32.07)
  (molar_mass_O : ℝ := 16.00)
  : ℝ :=
  let molar_mass_HBrO3 := molar_mass_H + molar_mass_Br + 3 * molar_mass_O
  let molar_mass_H2SO3 := 2 * molar_mass_H + molar_mass_S + 3 * molar_mass_O
  let total_mass_oxygen := (3 * molar_mass_O * mol_HBrO3) + (3 * molar_mass_O * mol_H2SO3)
  let total_mass_mixture := (molar_mass_HBrO3 * mol_HBrO3) + (molar_mass_H2SO3 * mol_H2SO3)
  (total_mass_oxygen / total_mass_mixture) * 100

theorem mass_percentage_oxygen_in_mixture
  (h : mass_percentage_oxygen ≈ 49.13) : 
  True := sorry

end mass_percentage_oxygen_in_mixture_l713_713143


namespace circle_fill_unique_differences_l713_713260

theorem circle_fill_unique_differences :
  ∃ (perm : Finset ℕ), perm.card = 8 ∧ (∀ (a b : ℕ), 
  (a ∈ perm ∧ b ∈ perm ∧ a ≠ b) -> 
  abs (a - b) ∈ {1, 2, 3, 4, 5, 6, 7}) :=
by
  -- We state the existence of such a permutation without providing the exact solution
  sorry

-- We define a correct tuple for specific values according to the problem
def circle_fill_example : list ℕ := [8, 7, 6, 5, 4, 3, 2, 1]

end circle_fill_unique_differences_l713_713260


namespace calculate_l713_713723

def q (x y : ℤ) : ℤ :=
  if x > 0 ∧ y ≥ 0 then x + 2*y
  else if x < 0 ∧ y ≤ 0 then x - 3*y
  else 4*x + 2*y

theorem calculate : q (q 2 (-2)) (q (-3) 1) = -4 := 
  by
    sorry

end calculate_l713_713723


namespace compute_expression_l713_713311

def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := 2 * x
noncomputable def f_inv (x : ℝ) : ℝ := x - 3
noncomputable def g_inv (x : ℝ) : ℝ := x / 2

theorem compute_expression : 
  f (g_inv (f_inv (f_inv (g (f 15))))) = 18 := by
  sorry

end compute_expression_l713_713311


namespace range_of_a_l713_713213

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Icc (-real.sqrt 2) (real.sqrt 2), 
    x^2 - real.sqrt 2 * x - a = 0 ∧ 
    (∃ y₁ ≠ y₂ ∈ Icc (-real.sqrt 2) (real.sqrt 2), 
      x = y₁ ∧ x = y₂)) ↔ a ∈ Icc (-1/2) 4 :=
by sorry

end range_of_a_l713_713213


namespace num_points_satisfy_conditions_l713_713630

def P_subset_Q (x y : ℕ) : Prop := {x, 1} ⊆ {y, 1, 2}

theorem num_points_satisfy_conditions : 
  (∑ x in finset.range 10, ∑ y in finset.range 10, if P_subset_Q x y then 1 else 0) = 9 := 
by sorry

end num_points_satisfy_conditions_l713_713630


namespace cos_120_eq_neg_half_l713_713992

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l713_713992


namespace function_increasing_interval_l713_713781

theorem function_increasing_interval :
  ∀ (x : ℝ), (x > 1 / 2) → (differentiable_at ℝ (λ x, 4 * x ^ 2 + 1 / x) x) → (8 * x - 1 / x ^ 2 > 0) := by
  sorry

end function_increasing_interval_l713_713781


namespace angle_BPC_in_isosceles_triangle_l713_713340

theorem angle_BPC_in_isosceles_triangle (A B C P: Type) [Isosceles_Triangle ABC B AB BC]
  (h1 : ∠ B A C = 80)
  (h2 : ∠ P A C = 40)
  (h3 : ∠ A C P = 30) :
  ∠ B P C = 100 :=
sorry

end angle_BPC_in_isosceles_triangle_l713_713340


namespace twelve_factorial_div_eleven_factorial_eq_twelve_l713_713524

theorem twelve_factorial_div_eleven_factorial_eq_twelve :
  12! / 11! = 12 :=
by
  sorry

end twelve_factorial_div_eleven_factorial_eq_twelve_l713_713524


namespace construct_parallelogram_l713_713817

-- Define the points and their relation
variables {A C M B D : Type}
variables (α : ℝ) (AC_length d2 : ℝ)

-- Assume the points and given conditions
axiom point_A_C : A ≠ C
axiom angle_α_exists : ∃ (angle : ℝ), angle = α
axiom midpoint_M : exists (M : A → C → Type), ∀ (A C: A → C → Type), Midpoint A C = M
axiom second_diagonal_length : exists (d2 : ℝ), d2 = length BD

-- Proving the construction of parallelogram
theorem construct_parallelogram (h1 : Diagonal AC_length) (h2 : Angle α) (h3 : Midpoint M) (h4 : Length_of_Second_Diagonal d2):
  ∃ (A B C D : Type), parallelogram A B C D ∧ Diagonal(A, C) = AC_length ∧ Angle(α) = angle_α_exists ∧ Diagonal(B, D) = d2 :=
by
  sorry

end construct_parallelogram_l713_713817


namespace first_nonzero_digit_of_one_over_157_l713_713417

theorem first_nonzero_digit_of_one_over_157 : 
  (∀ n, 157 * n < 1000 → (n > 0 ∧ (1000 * 1) / (157 * n) > 10^floor(log10( (1000 of 157 * n))) - 10^(1) = 3) := 
by sorry

end first_nonzero_digit_of_one_over_157_l713_713417


namespace shares_distribution_l713_713150

open Nat

-- Define the main statement for the problem at hand
theorem shares_distribution (E J P S : ℕ) :
  E > 0 ∧ J > 0 ∧ P > 0 ∧ S > 0 ∧ E + J + P + S = 16 ∧ E ≤ J + P + S ∧ J ≤ E + P + S ∧ P ≤ E + J + S ∧ S ≤ E + J + P →
  ((∑ x in (finset.range 4), if x < 4 then 1 else 0) = 315) :=
  by sorry

end shares_distribution_l713_713150


namespace stream_current_speed_l713_713893

theorem stream_current_speed (r w : ℝ) (h1 : 18 / (r + w) + 4 = 18 / (r - w))
  (h2 : 18 / (1.5 * r + w) + 2 = 18 / (1.5 * r - w)) : w = 2.5 :=
by
  -- Translate the equations from the problem conditions directly.
  sorry

end stream_current_speed_l713_713893


namespace cos_120_eq_neg_half_l713_713974

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713974


namespace min_value_pm_pn_l713_713568

theorem min_value_pm_pn (x y : ℝ)
  (h : x ^ 2 - y ^ 2 / 3 = 1) 
  (hx : 1 ≤ x) : (8 * x - 3) = 5 :=
sorry

end min_value_pm_pn_l713_713568


namespace finite_solutions_of_quadratic_not_perfect_square_l713_713698

theorem finite_solutions_of_quadratic_not_perfect_square 
  (a b : ℤ) 
  (h : ¬ ∃ z : ℤ, z^2 = b) : 
  {x : ℤ | ∃ y : ℤ, x^2 + a * x + b = y^2}.finite :=
sorry

end finite_solutions_of_quadratic_not_perfect_square_l713_713698


namespace proportionalities_l713_713445

noncomputable def find_x (x y z : ℝ) (m n : ℝ) : ℝ := m * (n / (z^.sqrt))^3

theorem proportionalities :
  ∃ m n : ℝ, 
  let x1 := 8 in
  let z1 := 9 in
  let z2 := 64 in
  let x2 := find_x x1 ((n / (9:^.sqrt)) : ℝ) 9 m n in
  x2 = 8 →
  find_x x2 ((n / (64:^.sqrt)) : ℝ) 64 m n = 27/64 :=
by
  sorry

end proportionalities_l713_713445


namespace find_min_n_minus_m_l713_713617

def f (x : ℝ) : ℝ := 2^|x| + 2 * |x|

theorem find_min_n_minus_m : (∃ m n : ℝ, m ≤ ∀ x ∈ set.Icc (-1 : ℝ) 1, f x ∧ (∀ x ∈ set.Icc (-1 : ℝ) 1, f x ≤ n) ∧ (n - m = 3)) :=
by
  sorry

end find_min_n_minus_m_l713_713617


namespace find_A_l713_713830

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) (h : 100 * A + 78 - (200 + B) = 364) : A = 5 :=
by
  sorry

end find_A_l713_713830


namespace covering_remainder_contains_all_l713_713046

variable (n k p : ℕ)
variable (d : Fin n → ℕ) (a : Fin n → ℕ)
variable (h_short_covering : is_short_covering_sequence d)
variable (h_covering : is_covering_sequence_with_arith_progressions d a)
variable (h_prime : Nat.Prime p)
variable (h_div : ∀ i, i < k → p ∣ d i)
variable (h_not_div : ∀ i, k ≤ i ∧ i < n → ¬ (p ∣ d i))

theorem covering_remainder_contains_all (h_short_covering : is_short_covering_sequence d)
    (h_covering : is_covering_sequence_with_arith_progressions d a)
    (h_prime : Nat.Prime p)
    (h_div : ∀ i, i < k → p ∣ d i)
    (h_not_div : ∀ i, k ≤ i ∧ i < n → ¬ (p ∣ d i)) :
    { x : ℕ // x < p } ⊆ { y : ℕ // y < p ∧ ∃ i, i < k ∧ (a i) % p = y } :=
sorry

end covering_remainder_contains_all_l713_713046


namespace mass_of_sand_pile_l713_713471

noncomputable theory

variables (base_area height density : ℝ)

def volume_of_cone (base_area height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

def mass_of_sand (volume density : ℝ) : ℝ :=
  volume * density

theorem mass_of_sand_pile (h_base_area : base_area = 62.8) (h_height : height = 3) (h_density : density = 1.5) :
  mass_of_sand (volume_of_cone base_area height) density = 94.2 :=
by {
  rw [h_base_area, h_height, h_density],
  calc
    mass_of_sand (volume_of_cone 62.8 3) 1.5
        = mass_of_sand ((1 / 3) * 62.8 * 3) 1.5 : rfl
    ... = mass_of_sand 62.8 1.5 : by norm_num
    ... = 62.8 * 1.5 : rfl
    ... = 94.2 : by norm_num
}

end mass_of_sand_pile_l713_713471


namespace sum_cos_eq_neg1_l713_713151

theorem sum_cos_eq_neg1 : ∑ k in Finset.range 18, cos (2 * k * Real.pi / 18) = -1 :=
sorry

end sum_cos_eq_neg1_l713_713151


namespace probability_both_red_l713_713272

def initial_red_buttons : ℕ := 5
def initial_blue_buttons : ℕ := 10
def total_initial_buttons : ℕ := initial_red_buttons + initial_blue_buttons

def final_buttons_in_A : ℕ := (3 * total_initial_buttons) / 5

def removed_buttons : ℕ := total_initial_buttons - final_buttons_in_A

-- Carla removes the same number of red and blue buttons
axiom removed_red_buttons : ℕ 
axiom removed_blue_buttons : ℕ 

-- After removal, Jar A has the remaining buttons
def remaining_red_buttons_in_A : ℕ := initial_red_buttons - removed_red_buttons
def remaining_blue_buttons_in_A : ℕ := initial_blue_buttons - removed_blue_buttons
def total_remaining_buttons_in_A : ℕ := remaining_red_buttons_in_A + remaining_blue_buttons_in_A

-- Jar B would contain the removed buttons
def red_buttons_in_B : ℕ := removed_red_buttons
def blue_buttons_in_B : ℕ := removed_blue_buttons
def total_buttons_in_B : ℕ := red_buttons_in_B + blue_buttons_in_B

-- Probabilities of drawing red buttons
def prob_red_A : ℚ := remaining_red_buttons_in_A / total_remaining_buttons_in_A
def prob_red_B : ℚ := red_buttons_in_B / total_buttons_in_B

-- Assertion that the final probability that both selected buttons are red is 1/9
theorem probability_both_red :
  total_initial_buttons = 15 → 
  final_buttons_in_A = 9 → 
  total_remaining_buttons_in_A = 9 → 
  removed_buttons = 6 → 
  red_buttons_in_B = removed_red_buttons → 
  blue_buttons_in_B = removed_blue_buttons → 
  removed_red_buttons + removed_blue_buttons = 6 → 
  prob_red_A = 1 / 3 → 
  prob_red_B = 1 / 3 → 
  prob_red_A * prob_red_B = 1 / 9 :=
  by
    intros,
    sorry

end probability_both_red_l713_713272


namespace calculate_x_l713_713317

variable (a b x : ℝ)
variable (h1 : r = (3 * a) ^ (3 * b))
variable (h2 : r = a ^ b * x ^ b)
variable (h3 : x > 0)

theorem calculate_x (a b x : ℝ) (h1 : r = (3 * a) ^ (3 * b)) (h2 : r = a ^ b * x ^ b) (h3 : x > 0) : x = 27 * a ^ 2 := by
  sorry

end calculate_x_l713_713317


namespace num_positive_solutions_eq_32_l713_713145

theorem num_positive_solutions_eq_32 : 
  ∃ n : ℕ, (∀ x y : ℕ, 4 * x + 7 * y = 888 → x > 0 ∧ y > 0) ∧ n = 32 :=
sorry

end num_positive_solutions_eq_32_l713_713145


namespace point_on_hyperbola_l713_713609

theorem point_on_hyperbola (p r : ℝ) (h1 : p > 0) (h2 : r > 0)
  (h_el : ∀ (x y : ℝ), x^2 / 4 + y^2 / 2 = 1)
  (h_par : ∀ (x y : ℝ), y^2 = 2 * p * x)
  (h_circum : ∀ (a b c : ℝ), a = 2 * r - 2 * p) :
  r^2 - p^2 = 1 := sorry

end point_on_hyperbola_l713_713609


namespace find_m_l713_713182

noncomputable def polynomial (x : ℝ) (m : ℝ) := 4 * x^2 - 3 * x + 5 - 2 * m * x^2 - x + 1

theorem find_m (m : ℝ) : 
  ∀ x : ℝ, (4 * x^2 - 3 * x + 5 - 2 * m * x^2 - x + 1 = (4 - 2 * m) * x^2 - 4 * x + 6)
  → (4 - 2 * m = 0) → (m = 2) :=
by
  intros x h1 h2
  sorry

end find_m_l713_713182


namespace distance_X_to_CD_l713_713271

theorem distance_X_to_CD (s : ℝ) (A B C D X : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (s, 0))
  (hC : C = (s, s))
  (hD : D = (0, s))
  (h_arc_A : ∀ (x y : ℝ), (x^2 + y^2 = (s/2)^2) → A = (x, y))
  (h_arc_B : ∀ (x y : ℝ), ((x - s)^2 + y^2 = (s/2)^2) → B = (x, y))
  (h_intersect : X = (s / 2, 0)) :
  ∃ d : ℝ, d = s :=
begin
  sorry
end

end distance_X_to_CD_l713_713271


namespace Liam_commute_distance_l713_713727

variables {D T T' : ℝ}

theorem Liam_commute_distance (D : ℝ) (S : ℝ) (T T' : ℝ) (commute_time : T = D / S) 
  (slower_speed_commute_time : T' = D / (S - 5)) (time_diff : T' = T + 4 / 60) 
  (speed : S = 60) : D = 44 :=
by
  have commute_time_eq : T = D / 60 := by rw [commute_time, speed]
  have slower_speed_commute_time_eq : T' = D / 55 := by rw [slower_speed_commute_time, speed, sub_eq_add_neg, add_neg_cancel_comm]
  have time_diff_eq : D / 55 = D / 60 + 4 / 60 := by rw [slower_speed_commute_time_eq, commute_time_eq, time_diff]
  sorry

end Liam_commute_distance_l713_713727


namespace inverse_prop_l713_713371

theorem inverse_prop (x : ℝ) : x < 0 → x^2 > 0 :=
by
  sorry

end inverse_prop_l713_713371


namespace determine_x_l713_713215

/-
  Determine \( x \) when \( y = 19 \)
  given the ratio of \( 5x - 3 \) to \( y + 10 \) is constant,
  and when \( x = 3 \), \( y = 4 \).
-/

theorem determine_x (x y k : ℚ) (h1 : ∀ x y, (5 * x - 3) / (y + 10) = k)
  (h2 : 5 * 3 - 3 / (4 + 10) = k) : x = 39 / 7 :=
sorry

end determine_x_l713_713215


namespace cos_120_eq_neg_half_l713_713935

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713935


namespace cos_120_eq_neg_half_l713_713986

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l713_713986


namespace distance_from_origin_l713_713666

def point := (Int × Int) -- defining a point as a pair of integers

def distance (p₁ p₂ : point) : Float :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  Math.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2).toFloat

theorem distance_from_origin :
  distance (0, 0) (12, -5) = 13 :=
by
  sorry -- proof omitted

end distance_from_origin_l713_713666


namespace arithmetic_series_sum_l713_713860

theorem arithmetic_series_sum : 
  ∑ i in finset.range 100 \ {0}, (100 * i + (100 - i)) = 499950 := by
  sorry

end arithmetic_series_sum_l713_713860


namespace base_of_parallelogram_l713_713101

theorem base_of_parallelogram 
  (Area : ℝ) (Height : ℝ) (h₁ : Area = 44) (h₂ : Height = 11) : 
  (Base : ℝ) (H : Area = Base * Height) : Base = 4 :=
by
  sorry

end base_of_parallelogram_l713_713101


namespace factorial_ratio_l713_713515

theorem factorial_ratio : 12! / 11! = 12 := by
  sorry

end factorial_ratio_l713_713515


namespace katya_needs_at_least_ten_l713_713288

variable (x : ℕ)

def prob_katya : ℚ := 4 / 5
def prob_pen  : ℚ := 1 / 2

def expected_correct (x : ℕ) : ℚ := x * prob_katya + (20 - x) * prob_pen

theorem katya_needs_at_least_ten :
  expected_correct x ≥ 13 → x ≥ 10 :=
  by sorry

end katya_needs_at_least_ten_l713_713288


namespace geom_series_even_odd_sum_ratio_l713_713750

theorem geom_series_even_odd_sum_ratio 
  (a : ℝ) (r : ℝ) (n : ℕ) (h_pos : 0 < 2 * n) (h_r : r ≠ 1) :
  let S_even := a * r * (1 - r^(2*n)) / (1 - r^2),
      S_odd := a * (1 - r^(2*n)) / (1 - r^2)
  in S_even / S_odd = r :=
by
  sorry

end geom_series_even_odd_sum_ratio_l713_713750


namespace percent_increase_l713_713898

theorem percent_increase (new_value old_value : ℕ) (h_new : new_value = 480) (h_old : old_value = 320) :
  ((new_value - old_value) / old_value) * 100 = 50 := by
  sorry

end percent_increase_l713_713898


namespace find_x_y_sum_l713_713635

-- Definitions of the vectors
def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b (x y : ℝ) : ℝ × ℝ × ℝ := (2, x, y)

-- Definition of parallel vectors
def are_parallel (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v1 = (k * v2.1, k * v2.2, k * v2.3)

-- Theorem to prove the required condition
theorem find_x_y_sum (x y : ℝ) (h : are_parallel a (b x y)) : x + y = -2 :=
by
  sorry

end find_x_y_sum_l713_713635


namespace original_num_of_men_l713_713091

-- Definitions of the conditions
def work {M W : ℕ} : Prop :=
  let work_rate_orig := W / 60 
  let work_rate_more := W / 50
  M * work_rate_orig = (M + 8) * work_rate_more

-- Problem statement (theorem)
theorem original_num_of_men (M W : ℕ) (h_work : work) : M = 48 :=
by
  sorry

end original_num_of_men_l713_713091


namespace twelve_factorial_div_eleven_factorial_eq_twelve_l713_713526

theorem twelve_factorial_div_eleven_factorial_eq_twelve :
  12! / 11! = 12 :=
by
  sorry

end twelve_factorial_div_eleven_factorial_eq_twelve_l713_713526


namespace log_sqrt_abs_sin_range_is_non_positive_l713_713426

theorem log_sqrt_abs_sin_range_is_non_positive :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 180 →
  ∃ y : ℝ, y = real.log 2 (real.sqrt (|real.sin x|)) ∧ y ≤ 0 :=
by
  sorry

end log_sqrt_abs_sin_range_is_non_positive_l713_713426


namespace exists_points_irrational_distance_rational_area_l713_713171

noncomputable def points := λ (n : ℕ), fin n → (ℝ × ℝ)

theorem exists_points_irrational_distance_rational_area (n : ℕ) (hn : 3 ≤ n) :
  ∃ (P : points n), 
    (∀ i j, i ≠ j → irrational (dist (P i) (P j))) ∧ 
    (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ∃ (a : ℚ), area (P i) (P j) (P k) = a) :=
sorry

end exists_points_irrational_distance_rational_area_l713_713171


namespace sum_of_squares_eq_229_l713_713003

-- The conditions
variables (x y : ℤ)
axiom diff_eq_221 : x^2 - y^2 = 221

-- The proof goal
theorem sum_of_squares_eq_229 : x^2 - y^2 = 221 → ∃ x y : ℤ, x^2 + y^2 = 229 :=
by
  sorry

end sum_of_squares_eq_229_l713_713003


namespace WI_parallel_to_Euler_line_of_ABC_l713_713696

-- Definitions of the main geometrical objects and conditions
variables 
  (A B C D E F W W_a W_b W_c N I : Point)
  [IsAcuteScaleneTriangle ABC]
  [FeetOfAltitudes ABC D E F]
  [ReflectionsOverSides ABC W W_a W_b W_c]
  [Circumcenter W_aW_bW_c N]
  [Incenter W_aW_bW_c I]
  [NinePointCenter DEF N]

-- Statement of the problem
theorem WI_parallel_to_Euler_line_of_ABC :
  Parallel (LineThroughPoints W I) (EulerLine ABC) :=
sorry

end WI_parallel_to_Euler_line_of_ABC_l713_713696


namespace find_e_of_polynomial_properties_l713_713786

theorem find_e_of_polynomial_properties :
  ∃ (d e f : ℤ), (∀ (Q : ℤ → ℤ),
  (Q = (λ x, 3 * x^3 + d * x^2 + e * x + f)) ∧
  (Q(0) = 9) ∧
  (let zeros_mean := -3,
       zeros_product := -3, 
       sum_coeff := 3 + d + e + f in
  zeros_mean = -3 ∧
  zeros_product = -3 ∧ 
  sum_coeff = -3
  )) → e = -42 :=
by
  sorry

end find_e_of_polynomial_properties_l713_713786


namespace palindromes_between_2005_and_3000_l713_713488

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

def count_palindromes (start : ℕ) (end : ℕ) : ℕ :=
  (List.range' start (end - start)).countp is_palindrome

theorem palindromes_between_2005_and_3000 : count_palindromes 2005 3000 = 9 :=
by
  sorry -- Proof goes here

end palindromes_between_2005_and_3000_l713_713488


namespace min_area_triangle_ABC_l713_713653

theorem min_area_triangle_ABC (k : ℝ) : 
  let p := (-((k - 1) : ℝ), (-k - 1) : ℝ)
  let q := (vertex_y := -(1 / 4) * (k ^ 2 + 2 * k + 5) : ℝ)
  let distance_ab := (k^2 + 2*k + 5).sqrt
  let area_abc := (1 / 2) * distance_ab * vertex_y
  ∃ (k : ℝ), area_abc = 1 :=
begin
  sorry
end

end min_area_triangle_ABC_l713_713653


namespace positive_odd_divisible_l713_713413

theorem positive_odd_divisible (x y : ℤ) (n : ℕ) (k : ℕ) (h1 : n = 2 * k - 1) (h2 : 0 < k) :
  (x^n + y^n) mod (x + y) = 0 := 
sorry

end positive_odd_divisible_l713_713413


namespace geometric_sequence_root_product_l713_713662

theorem geometric_sequence_root_product
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (a1_pos : 0 < a 1)
  (a19_root : a 1 * r^18 = (1 : ℝ))
  (h_poly : ∀ x, x^2 - 10 * x + 16 = 0) :
  a 8 * a 12 = 16  :=
sorry

end geometric_sequence_root_product_l713_713662


namespace interval_of_monotonic_decrease_minimum_value_of_a_l713_713210

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - 1/2 * a * x^2 + x

theorem interval_of_monotonic_decrease (a : ℝ) (h : f 1 a = 0) :
  { x : ℝ | x > 1 } ⊆ { x : ℝ | ∃ a, f 1 a = 0 ∧ (f'' x a) < 0 } :=
sorry

theorem minimum_value_of_a (a : ℝ) :
  (∀ x, f x a ≤ a * x - 1) → a ≥ 2 :=
sorry

end interval_of_monotonic_decrease_minimum_value_of_a_l713_713210


namespace prob_chemistry_prob_biology_union_history_prob_chemistry_union_geography_l713_713090

variables (students : Type) [fintype students] [decidable_eq students]
variables (physics chemistry biology politics history geography : set students)
variables (n : ℕ) (total_students := 1000)

-- Students counts based on conditions
variables (h_total : fintype.card students = total_students)
variables (h_physics : fintype.card physics = 300)
variables (h_chemistry : fintype.card chemistry = 200)
variables (h_biology : fintype.card biology = 100)
variables (h_politics : fintype.card politics = 200)
variables (h_history : fintype.card history = 100)
variables (h_geography : fintype.card geography = 100)

namespace high_school_probabilities

-- Probability of selecting a student from a set
def P (s : set students) : ℚ := fintype.card s / total_students

-- Problem 1: Prove P(B) = 1/5
theorem prob_chemistry : P chemistry = 1/5 :=
sorry

-- Problem 2: Prove P(C ∪ E) = 1/5
theorem prob_biology_union_history : P (biology ∪ history) = 1/5 :=
sorry

-- Problem 3: Prove P(B ∪ F) = 3/10
theorem prob_chemistry_union_geography : P (chemistry ∪ geography) = 3/10 :=
sorry

end high_school_probabilities

end prob_chemistry_prob_biology_union_history_prob_chemistry_union_geography_l713_713090


namespace athletes_teams_l713_713673

-- Define the type representing athletes
def Athlete : Type := Fin 10

-- Define the two specific athletes A and B
def A : Athlete := 0
def B : Athlete := 1

-- The theorem we want to prove
theorem athletes_teams (hAB : A ≠ B) : 
  ∃ (teams : Finset (Fin 10) × Finset (Fin 10)), 
    (teams.1.card = 5 ∧ teams.2.card = 5 ∧
     teams.1 ∩ teams.2 = ∅ ∧ teams.1 ∪ teams.2 = Finset.univ ∧
     (A ∈ teams.1 ∧ B ∈ teams.2) ∨ (A ∈ teams.2 ∧ B ∈ teams.1) ∧
     teams.1.choose 4 = 70) := sorry

end athletes_teams_l713_713673


namespace range_of_a_l713_713211

variable (a : ℝ)

def f (x : ℝ) := Real.logBase (1/5) (2 * a * x - 5)

theorem range_of_a (h : ∀ x1 x2 : ℝ, (2 < x1) → (2 < x2) → (x1 ≠ x2) → ((f a x1) - (f a x2)) / (x1 - x2) < 0) : a ≥ 5 / 4 :=
sorry

end range_of_a_l713_713211


namespace number_of_ways_to_choose_team_l713_713743

-- Define the conditions
def total_players : ℕ := 16
def quadruplets : Finset ℕ := {1, 2, 3, 4}
def other_players : Finset ℕ := (Finset.range 16) \ quadruplets
def choose_three_quadruplets : ℕ := (quadruplets.card.choose 3)
def choose_two_others : ℕ := ((Finset.range 16).card - quadruplets.card).choose 2

-- Prove the number of ways to choose the team
theorem number_of_ways_to_choose_team : choose_three_quadruplets * choose_two_others = 264 := by
  have h1 : quadruplets.card = 4 := by simp [quadruplets]
  have h2 : (Finset.range 16).card = 16 := by simp
  have h3 : other_players.card = 12 := by 
    simp [other_players, quadruplets]
    exact Nat.sub_eq_of_eq_add h1.symm
  simp [h1, h3, choose_three_quadruplets, choose_two_others]
  norm_num
  sorry

end number_of_ways_to_choose_team_l713_713743


namespace minimum_perimeter_triangle_area_l713_713105

variable (A B C D S M : Type)
variable (AB AD SA : ℕ)
variable [One AB] [Ten AD] [Four SA]
variables [Rectangle ABCD] [Perpendicular SA ABCD]

theorem minimum_perimeter_triangle_area (M : A → B → Prop) 
  (h1 : AB = 1) (h2 : AD = 10) (h3 : SA = 4) 
  : area_triangle_with_minimum_perimeter S M C = 6 :=
begin
  sorry
end

end minimum_perimeter_triangle_area_l713_713105


namespace solution_proof_l713_713463

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(-x)

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f(x) ≤ f(y)

def problem_statement : Prop :=
  is_even_function (fun x => abs x + 1) ∧
  is_monotonically_increasing (fun x => abs x + 1) (set.Ioi 0)

theorem solution_proof : problem_statement :=
  by
    have h_even : is_even_function (fun x => abs x + 1),
    { sorry },
    have h_monotonic : is_monotonically_increasing (fun x => abs x + 1) (set.Ioi 0),
    { sorry },
    exact ⟨h_even, h_monotonic⟩

end solution_proof_l713_713463


namespace sales_tax_on_taxable_purchases_l713_713139

-- Daniel's total expenditure
def total_expenditure : ℝ := 25

-- Cost of tax-free items
def tax_free_items_cost : ℝ := 19.7

-- Tax rate
def tax_rate : ℝ := 0.06

-- Sales tax on taxable purchases
theorem sales_tax_on_taxable_purchases :
  ∃ (X : ℝ), tax_free_items_cost + X + tax_rate * X = total_expenditure ∧ tax_rate * X = 0.3 :=
begin
  sorry
end

end sales_tax_on_taxable_purchases_l713_713139


namespace ratio_hunter_to_sisters_l713_713332

theorem ratio_hunter_to_sisters (Ella_hotdogs Emma_hotdogs Matthew_total: ℕ) (Luke_factor Hunter_total: ℕ) : 
  Ella_hotdogs = 2 → 
  Emma_hotdogs = 2 → 
  Luke_factor = 2 → 
  Matthew_total = 14 → 
  Hunter_total = Matthew_total - (Ella_hotdogs + Emma_hotdogs + Luke_factor * (Ella_hotdogs + Emma_hotdogs)) → 
  (Hunter_total : (Ella_hotdogs + Emma_hotdogs)) = (1 : 2) :=
by
  sorry

end ratio_hunter_to_sisters_l713_713332


namespace f_is_odd_f_is_monotonically_increasing_solve_inequality_l713_713614

-- Define the function f(x)
def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

-- Problem 1: Prove that f(x) is odd
theorem f_is_odd : ∀ x : ℝ, f (-x) = - f x :=
by
  intro x
  sorry

-- Problem 2: Prove that f(x) is monotonically increasing
theorem f_is_monotonically_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
by
  intros x1 x2 h
  sorry

-- Problem 3: Prove the solution set of the inequality
theorem solve_inequality : ∀ m : ℝ, f (1 - m) + f (1 - m^2) < 0 ↔ m ∈ Ioo (-∞) (-2) ∪ Ioo 1 ∞ :=
by
  intro m
  sorry

end f_is_odd_f_is_monotonically_increasing_solve_inequality_l713_713614


namespace piecewise_function_not_composed_of_multiple_functions_l713_713547

theorem piecewise_function_not_composed_of_multiple_functions :
  ∀ (f : ℝ → ℝ), (∃ (I : ℝ → Prop) (f₁ f₂ : ℝ → ℝ),
    (∀ x, I x → f x = f₁ x) ∧ (∀ x, ¬I x → f x = f₂ x)) →
    ¬(∃ (g₁ g₂ : ℝ → ℝ), (∀ x, f x = g₁ x ∨ f x = g₂ x)) :=
by
  sorry

end piecewise_function_not_composed_of_multiple_functions_l713_713547


namespace parabola_directrix_eq_l713_713565

variable (x : ℝ)

def parabola : ℝ → ℝ := λ x, 3 * x^2 - 12 * x + 15

def directrix (a h k : ℝ) : ℝ := k - (1 / (4 * a))

theorem parabola_directrix_eq :
  directrix 3 2 3 = 35 / 12 :=
by sorry

end parabola_directrix_eq_l713_713565


namespace angle_equivalence_l713_713672

noncomputable theory

variables {A B C N M : Type}
variables [plane_geometry]


-- Define an acute-angled triangle
def acute_triangle (A B C : triangle) : Prop := A.angle < 90 ∧ B.angle < 90 ∧ C.angle < 90

-- Midpoint definition
def midpoint (M B C : point) : Prop := dist B M = dist M C

-- Tangent Intersection definition
def tangent_intersection (N B C : point) : circle → Prop := 
λ (circ : circle), tangent B circ = N ∧ tangent C circ = N 

theorem angle_equivalence {A B C: point} (circ : circle) (N M : point)
  (h1: circumcircle A B C circ)
  (h2: acute_triangle A B C)
  (h3: midpoint M B C)
  (h4: tangent_intersection N B C circ):
  ∠A B M = ∠A C N := 
sorry

end angle_equivalence_l713_713672


namespace b_eq_6_l713_713714

theorem b_eq_6 (a b : ℤ) (h₁ : |a| = 1) (h₂ : ∀ x : ℝ, a * x^2 - 2 * x - b + 5 = 0 → x < 0) : b = 6 := 
by
  sorry

end b_eq_6_l713_713714


namespace soccer_balls_donated_l713_713887

def num_classes_per_school (elem_classes mid_classes : ℕ) : ℕ :=
  elem_classes + mid_classes

def total_classes (num_schools : ℕ) (classes_per_school : ℕ) : ℕ :=
  num_schools * classes_per_school

def total_soccer_balls (num_classes : ℕ) (balls_per_class : ℕ) : ℕ :=
  num_classes * balls_per_class

theorem soccer_balls_donated 
  (elem_classes mid_classes num_schools balls_per_class : ℕ) 
  (h_elem_classes : elem_classes = 4) 
  (h_mid_classes : mid_classes = 5) 
  (h_num_schools : num_schools = 2) 
  (h_balls_per_class : balls_per_class = 5) :
  total_soccer_balls (total_classes num_schools (num_classes_per_school elem_classes mid_classes)) balls_per_class = 90 :=
by
  sorry

end soccer_balls_donated_l713_713887


namespace compute_fraction_l713_713323

noncomputable def distinct_and_sum_zero (w x y z : ℝ) : Prop :=
w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ w + x + y + z = 0

theorem compute_fraction (w x y z : ℝ) (h : distinct_and_sum_zero w x y z) :
  (w * y + x * z) / (w^2 + x^2 + y^2 + z^2) = -1 / 2 :=
sorry

end compute_fraction_l713_713323


namespace first_nonzero_digit_one_over_157_l713_713421

theorem first_nonzero_digit_one_over_157 : 
  (∃ d : ℕ, d < 10 ∧ (∃ n : ℕ, (1000 / 157 : ℝ) = (6 + d * 10^-1 + n * 10^-2) * 157)) →
  d = 3 :=
by sorry

end first_nonzero_digit_one_over_157_l713_713421


namespace smallest_possible_sum_of_squares_l713_713005

theorem smallest_possible_sum_of_squares : 
  ∃ (x y : ℕ), (x^2 - y^2 = 221) ∧ (∀(a b : ℕ), (a^2 - b^2 = 221) → (x^2 + y^2 ≤ a^2 + b^2)) ∧ (x^2 + y^2 = 229) :=
begin
  sorry, -- Proof omitted as per instructions
end

end smallest_possible_sum_of_squares_l713_713005


namespace contrapositive_even_addition_l713_713768

theorem contrapositive_even_addition (a b : ℕ) :
  (¬((a % 2 = 0) ∧ (b % 2 = 0)) → (a + b) % 2 ≠ 0) :=
sorry

end contrapositive_even_addition_l713_713768


namespace math_problem_l713_713140

noncomputable def f (x : ℝ) : ℝ := sorry

theorem math_problem (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (1 - x) = f (1 + x))
  (h2 : ∀ x : ℝ, f (-x) = -f x)
  (h3 : ∀ {x y : ℝ}, (0 ≤ x → x < y → y ≤ 1 → f x < f y)) :
  (f 0 = 0) ∧ 
  (∀ x : ℝ, f (x + 2) = f (-x)) ∧ 
  (∀ x : ℝ, x = -1 ∨ ∀ ε > 0, ε ≠ (x + 1))
:= sorry

end math_problem_l713_713140


namespace complement_subset_l713_713702

variable {α : Type*} [PartialOrder α]

def P (x : α) : Prop := x < 1

def Q (x : α) : Prop := x > -1

def CRP (x : α) : Prop := x >= 1

theorem complement_subset {x : α} (h : CRP x) : Q x :=
sorry

end complement_subset_l713_713702


namespace probability_of_first_good_product_on_third_try_l713_713122

-- Define the problem parameters
def pass_rate : ℚ := 3 / 4
def failure_rate : ℚ := 1 / 4
def epsilon := 3

-- The target probability statement
theorem probability_of_first_good_product_on_third_try :
  (failure_rate * failure_rate * pass_rate) = ((1 / 4) ^ 2 * (3 / 4)) :=
by
  sorry

end probability_of_first_good_product_on_third_try_l713_713122


namespace find_valid_start_l713_713916

-- Define the room types
inductive RoomType
| black
| hatched
| gray
| white

-- Define the room labels
inductive RoomLabel
| A | B | C | D | E | F | G
| others : Nat → RoomLabel

-- Define a structure for the room
structure Room where
  label : RoomLabel
  rtype : RoomType

-- Define the maze as a set of interconnected rooms and list of all rooms
structure Maze where
  rooms : List Room
  adj : Room → Room → Bool -- represents adjacency

-- Condition: Huahua can leave the maze if she visits all the gray rooms and then enters the hatched room
def canExitMaze (maze : Maze) (start : Room) : Bool :=
  sorry

-- Huahua can only start from one of the rooms represented by "A, B, C, D, E, F, G"
def validStartingRooms : List RoomLabel := [RoomLabel.A, RoomLabel.B, RoomLabel.C, RoomLabel.D, RoomLabel.E, RoomLabel.F, RoomLabel.G]

def isValidStart (maze : Maze) (start : Room) : Bool :=
  validStartingRooms.contains start.label ∧ canExitMaze maze start

-- The proof problem
theorem find_valid_start (mz : Maze) (ra : Room) (rd : Room) :
  ra.label = RoomLabel.A → rd.label = RoomLabel.D → 
  (isValidStart mz ra ∨ isValidStart mz rd) :=
sorry

end find_valid_start_l713_713916


namespace average_price_returned_packets_l713_713728

theorem average_price_returned_packets :
  ∀ (avg_price_initial : ℕ) (total_packets : ℕ) (returned_packets : ℕ) (avg_price_remaining : ℕ) (remaining_packets : ℕ),
  avg_price_initial = 20 →
  total_packets = 5 →
  returned_packets = 2 →
  avg_price_remaining = 12 →
  remaining_packets = 3 →
  let total_cost_initial := total_packets * avg_price_initial in
  let total_cost_remaining := remaining_packets * avg_price_remaining in
  let total_cost_returned := total_cost_initial - total_cost_remaining in
  total_cost_returned / returned_packets = 32 :=
by
  intros avg_price_initial total_packets returned_packets avg_price_remaining remaining_packets
  intros h1 h2 h3 h4 h5
  let total_cost_initial := total_packets * avg_price_initial
  let total_cost_remaining := remaining_packets * avg_price_remaining
  let total_cost_returned := total_cost_initial - total_cost_remaining
  show total_cost_returned / returned_packets = 32
  sorry

end average_price_returned_packets_l713_713728


namespace triangle_area_is_correct_l713_713268

noncomputable def area_triangle
  (AB BC : ℝ) (angle_A : ℝ) : ℝ :=
  0.5 * AB * BC * real.sin angle_A

theorem triangle_area_is_correct : 
  ∀ (AB BC : ℝ) (angle_A : ℝ), AB = real.sqrt 6 → BC = 2 → angle_A = real.pi / 4 → 
  area_triangle AB BC angle_A = (3 + real.sqrt 3) / 2 :=
by
  intros AB BC angle_A h_AB h_BC h_A
  simp only [area_triangle]
  sorry

end triangle_area_is_correct_l713_713268


namespace integer_solutions_l713_713142

theorem integer_solutions (a b c : ℤ) :
  a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  intros h
  sorry

end integer_solutions_l713_713142


namespace degree_measure_supplement_complement_l713_713820

noncomputable def supp_degree_complement (α : ℕ) := 180 - (90 - α)

theorem degree_measure_supplement_complement : 
  supp_degree_complement 36 = 126 :=
by sorry

end degree_measure_supplement_complement_l713_713820


namespace find_value_of_expression_l713_713180

variables (a b c : ℝ)

theorem find_value_of_expression
  (h1 : a ^ 4 * b ^ 3 * c ^ 5 = 18)
  (h2 : a ^ 3 * b ^ 5 * c ^ 4 = 8) :
  a ^ 5 * b * c ^ 6 = 81 / 2 :=
sorry

end find_value_of_expression_l713_713180


namespace washing_machine_time_l713_713497

theorem washing_machine_time (shirts pants sweaters jeans : ℕ) (max_items_per_cycle minutes_per_cycle : ℕ)
    (h_shirts : shirts = 18) (h_pants : pants = 12) (h_sweaters : sweaters = 17) (h_jeans : jeans = 13)
    (h_max_items_per_cycle : max_items_per_cycle = 15) (h_minutes_per_cycle : minutes_per_cycle = 45) :
    (shirts + pants + sweaters + jeans) / max_items_per_cycle * minutes_per_cycle / 60 = 3 := by
  -- Total number of items calculation
  have total_items : shirts + pants + sweaters + jeans = 18 + 12 + 17 + 13 := by
    rw [h_shirts, h_pants, h_sweaters, h_jeans]
    norm_num
    
  -- Number of cycles calculation
  have cycles : (shirts + pants + sweaters + jeans) / max_items_per_cycle = 60 / 15 := by
    rw [total_items, h_max_items_per_cycle]
    norm_num

  -- Total time in minutes calculation
  have total_time_minutes : (shirts + pants + sweaters + jeans) / max_items_per_cycle * minutes_per_cycle = 4 * 45 := by
    rw [cycles, h_minutes_per_cycle]
    norm_num

  -- Time conversion from minutes to hours
  show 4 * 45 / 60 = 3
  norm_num

end washing_machine_time_l713_713497


namespace total_amount_spent_l713_713736

theorem total_amount_spent : 
  let value_half_dollar := 0.50
  let wednesday_spending := 4 * value_half_dollar
  let next_day_spending := 14 * value_half_dollar
  wednesday_spending + next_day_spending = 9.00 :=
by
  let value_half_dollar := 0.50
  let wednesday_spending := 4 * value_half_dollar
  let next_day_spending := 14 * value_half_dollar
  show _ 
  sorry

end total_amount_spent_l713_713736


namespace range_of_a_l713_713545

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ∀ θ : ℝ, θ ∈ Icc 0 (π / 2) →
    (x + 3 + 2 * sin θ * cos θ)^2 + (x + a * sin θ + a * cos θ)^2 ≥ 1 / 8) ↔
  (a ≥ 7 / 2 ∨ a ≤ real.sqrt 6) :=
by
  sorry

end range_of_a_l713_713545


namespace max_marked_cells_no_distance_15_l713_713327

def distance (x1 y1 x2 y2 : ℕ) : ℕ := 
  max (int.natAbs (x1 - x2)) (int.natAbs (y1 - y2))

def is_valid_board (board : Finset (ℕ × ℕ)) : Prop :=
  ∀ (x1 y1 x2 y2 : ℕ), (x1, y1) ∈ board → (x2, y2) ∈ board → 
  (x1 ≠ x2 ∨ y1 ≠ y2) → distance x1 y1 x2 y2 ≠ 15

theorem max_marked_cells_no_distance_15 :
  ∃ (board : Finset (ℕ × ℕ)), board.card = 3025 ∧ 
  is_valid_board board :=
sorry

end max_marked_cells_no_distance_15_l713_713327


namespace proof_problem_l713_713192

def P : Prop := ∀ (a b c : ℝ), b^2 = a * c → (∃ r : ℝ, b = a * r ∧ c = b * r)
def Q : Prop := ∀ x : ℝ, cos (π / 2 + x) = -sin x

theorem proof_problem : ¬P ∧ Q → (P ∨ Q) :=
by
  intro h,
  cases h with hP hQ,
  right,
  exact hQ,
  sorry

end proof_problem_l713_713192


namespace max_area_inscribed_quadrilateral_max_area_inscribed_ngon_l713_713073

/-- For any convex quadrilateral with given angles and perimeter, 
    the inscribed quadrilateral has the largest area. -/
theorem max_area_inscribed_quadrilateral 
  (α β γ δ : ℝ) 
  (P : ℝ)
  (convex : α + β + γ + δ = 2 * π) 
  (positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < δ) 
  (positive_perimeter : P > 0) :
  ∃ Q : Quadrilateral, (inscribed Q) ∧ (∀ Q' : Quadrilateral, area Q ≤ area Q') :=
sorry

/-- For any convex n-gon with given angles and perimeter, 
    the inscribed n-gon has the largest area. -/
theorem max_area_inscribed_ngon 
  (n : ℕ) 
  (angles : Fin n → ℝ) 
  (P : ℝ)
  (convex : ∑ i, angles i = (n - 2) * π) 
  (positive_angles : ∀ i, 0 < angles i) 
  (positive_perimeter : P > 0) :
  ∃ ng : NGon, (inscribed ng) ∧ (∀ ng' : NGon, area ng ≤ area ng') :=
sorry

end max_area_inscribed_quadrilateral_max_area_inscribed_ngon_l713_713073


namespace interest_difference_l713_713454

noncomputable def difference_between_interest (P : ℝ) (R : ℝ) (T : ℝ) (n : ℕ) : ℝ :=
  let SI := P * R * T / 100
  let CI := P * (1 + (R / (n*100)))^(n * T) - P
  CI - SI

theorem interest_difference (P : ℝ) (R : ℝ) (T : ℝ) (n : ℕ) (hP : P = 1200) (hR : R = 10) (hT : T = 1) (hn : n = 2) :
  difference_between_interest P R T n = -59.25 := by
  sorry

end interest_difference_l713_713454


namespace dan_gave_cards_l713_713753

noncomputable def sally_initial_cards : ℕ := 27
noncomputable def sally_total_cards : ℕ := 88
noncomputable def cards_bought : ℕ := 20

-- Define the required proof
theorem dan_gave_cards : ∃ d : ℕ, sally_total_cards = sally_initial_cards + d + cards_bought ∧ d = 41 :=
by {
  use 41,
  simp only [sally_initial_cards, sally_total_cards, cards_bought],
  exact nat.add_right_cancel_iff.mp 
    (nat.add_left_cancel_iff.mp rfl)
}

end dan_gave_cards_l713_713753


namespace point_transformation_l713_713022

theorem point_transformation :
  let init_point := (2, 2, 2)
  let after_z_rotation_90 := (-2, 2, 2)
  let after_xz_reflection := (-2, -2, 2)
  let after_x_rotation_90 := (-2, 2, -2)
  let after_yz_reflection := (2, 2, -2)
  let final_point := (2, -2, -2)
  final_point = transform_point init_point := sorry

/--
  Given a point $(2,2,2)$ and a series of transformations:
  1. Rotate $90^\circ$ about the $z$-axis
  2. Reflect through the $xz$-plane
  3. Rotate $90^\circ$ about the $x$-axis
  4. Reflect through the $yz$-plane
  5. Rotate $90^\circ$ about the $z$-axis

  We need to show that the final coordinates of the point are $(2, -2, -2)$.
--/

noncomputable def transform_point (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let p1 := ( -p.2,  p.1, p.3 ) -- Rotation 90 degrees about the z-axis
  let p2 := ( p1.1, -p1.2, p1.3 ) -- Reflection through the xz-plane
  let p3 := ( p2.1,  p2.3, -p2.2 ) -- Rotation 90 degrees about the x-axis
  let p4 := (-p3.1,  p3.2, p3.3 ) -- Reflection through the yz-plane
  let p5 := ( p4.2, -p4.1, p4.3 ) -- Rotation 90 degrees about the z-axis
  p5

end point_transformation_l713_713022


namespace correct_value_wrongly_copied_l713_713015

theorem correct_value_wrongly_copied 
  (mean_initial : ℕ)
  (mean_correct : ℕ)
  (wrong_value : ℕ) 
  (n : ℕ) 
  (initial_mean : mean_initial = 250)
  (correct_mean : mean_correct = 251)
  (wrongly_copied : wrong_value = 135)
  (number_of_values : n = 30) : 
  ∃ x : ℕ, x = 165 := 
by
  use (wrong_value + (mean_correct - mean_initial) * n / n)
  sorry

end correct_value_wrongly_copied_l713_713015


namespace train_crosses_bridge_in_time_l713_713451

noncomputable def speed_kmph_to_mps (speed_kmph : ℝ) : ℝ := (speed_kmph * 1000) / 3600

theorem train_crosses_bridge_in_time:
  (length_train length_bridge : ℝ) (speed_train_kmph : ℝ) 
  (h_train : length_train = 165) (h_bridge : length_bridge = 660) (h_speed : speed_train_kmph = 36) :
  let total_distance := length_train + length_bridge in
  let speed_mps := speed_kmph_to_mps speed_train_kmph in
  let time := total_distance / speed_mps in
  time = 82.5 :=
by
  -- Code for automatic proof can be added here
  sorry

end train_crosses_bridge_in_time_l713_713451


namespace evaluate_expression_l713_713553

/-- 
Define the floor function and variable x
-/
def floor (n : ℝ) : ℤ := Int.floor n
def x : ℝ := 13.2

/-- 
Proof problem: Prove that the given expression evaluates to 5
-/
theorem evaluate_expression : 
  floor (x^2) - floor x * floor x = 5 := by
  sorry

end evaluate_expression_l713_713553


namespace square_pyramid_properties_l713_713828

-- Definitions for the square pyramid with a square base
def square_pyramid_faces : Nat := 4 + 1
def square_pyramid_edges : Nat := 4 + 4
def square_pyramid_vertices : Nat := 4 + 1

-- Definition for the number of diagonals in a square
def diagonals_in_square_base (n : Nat) : Nat := n * (n - 3) / 2

-- Theorem statement
theorem square_pyramid_properties :
  (square_pyramid_faces + square_pyramid_edges + square_pyramid_vertices = 18) ∧ (diagonals_in_square_base 4 = 2) :=
by
  sorry

end square_pyramid_properties_l713_713828


namespace smallest_possible_sum_of_squares_l713_713006

theorem smallest_possible_sum_of_squares : 
  ∃ (x y : ℕ), (x^2 - y^2 = 221) ∧ (∀(a b : ℕ), (a^2 - b^2 = 221) → (x^2 + y^2 ≤ a^2 + b^2)) ∧ (x^2 + y^2 = 229) :=
begin
  sorry, -- Proof omitted as per instructions
end

end smallest_possible_sum_of_squares_l713_713006


namespace isosceles_triangle_perimeter_l713_713249

-- Define the isosceles triangle conditions
variables {a b : ℝ} (is_isosceles : a = b ∨ b = 7 ∨ a = 7)
variable (triangle_sides : fin 3 → ℝ) -- Representing the sides of the triangle

def is_triangle (sides : fin 3 → ℝ) : Prop :=
sides 0 + sides 1 > sides 2 ∧ sides 1 + sides 2 > sides 0 ∧ sides 2 + sides 0 > sides 1

-- The given side lengths
axiom side_lengths : triangle_sides 0 = 3 ∨ triangle_sides 1 = 3 ∨ triangle_sides 2 = 3
axiom side_lengths_7 : triangle_sides 0 = 7 ∨ triangle_sides 1 = 7 ∨ triangle_sides 2 = 7

-- The perimeter of a triangle with the sides lengths a, b and c
def triangle_perimeter (sides : fin 3 → ℝ) : ℝ :=
  sides 0 + sides 1 + sides 2

-- The main theorem stating the perimeter of the triangle
theorem isosceles_triangle_perimeter {triangle_sides : fin 3 → ℝ} 
(is_isosceles : let t := list.sort (≤) [triangle_sides 0, triangle_sides 1, triangle_sides 2] in t.head! = 7 ∧ (t.nth 1 = some 7 ∨ t.last! = 7) ∧ t.head! ≤ t.nth 1 ∧ t.head! ≤ t.last!)
(side_lengths :  ∃ x, ∃ y, ∃ z, x = 3 ∧ y = 7 ∧ z = 7 ∧ let t := list.sort (≤) [x, y, z] in t.head! + t.nth_le 1 3 + t.last! > t.nth 2 ∧ t.nth 1 + t.nth_le 1 3 + t.head! > t.head! ∧ t.last! + t.head! > t.nth_le 1 3)
: triangle_perimeter triangle_sides = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l713_713249


namespace cylinder_volume_ratio_l713_713075

theorem cylinder_volume_ratio
  (h : ℝ)
  (r1 : ℝ)
  (r3 : ℝ := 3 * r1)
  (V1 : ℝ := 40) :
  let V2 := π * r3^2 * h
  (π * r1^2 * h = V1) → 
  V2 = 360 := by
{
  sorry
}

end cylinder_volume_ratio_l713_713075


namespace min_area_containing_squares_l713_713827

def area_rectangle_containing_squares (α : ℝ) : Prop :=
  ∀ (x : ℝ), (1 / √ 2) ≤ x ∧ x ≤ 1 →
  x^2 + x * (√ (1 - x^2)) ≤ α

theorem min_area_containing_squares :
  ∃ α : ℝ, area_rectangle_containing_squares α ∧ 
  ∀ β : ℝ, area_rectangle_containing_squares β → α ≤ β :=
  ∃ α = (1 / 2) * (1 + √ 2), 
  area_rectangle_containing_squares α ∧ α ≤ (1 / 2) * (1 + √ 2) := 
  by sorry

end min_area_containing_squares_l713_713827


namespace intersection_of_lines_l713_713380

noncomputable def points_collinear (A B C D : Type*) [MetricSpace A] (f : CircleMap A C) (g : CircleMap B D)
:= sorry

theorem intersection_of_lines 
  (A M1 M2 C E F B D : Point)
  (k1 : Circle M1 A)
  (k2 : Circle M2 C)
  (tangent_common : (is_tangent k1 k2 B D))
  (collinear_conds: Collinear [A, M1, M2, C])
  (intersection_conds : Intersect k1 k2 = {E, F}) :
  ∃ X : Point, lies_on X (Line_through A B) ∧ lies_on X (Line_through C D) ∧ lies_on X (Line_through E F) :=
sorry

end intersection_of_lines_l713_713380


namespace can_form_triangle_l713_713838

-- Define the lengths of the segments
def a := 1
def b := 2
def c := 2

-- State the theorem
theorem can_form_triangle : a + b > c ∧ a + c > b ∧ b + c > a :=
by
  -- Using the definitions of a, b, c given above
  unfold a b c
  -- Using the triangle inequality
  split_ifs
  -- First inequality
  linarith
  -- Second inequality
  linarith
  -- Third inequality
  linarith
  sorry

end can_form_triangle_l713_713838


namespace solution_l713_713237

theorem solution (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) 
    (h₄ : d ≠ 0) (h₅ : ∀ x, dx + b ≠ 0) (h₆ : ∀ x, 
    (let f := λ x, (a * x^2 + b * x + c) / (d * x + b) in 
    f (f x)) = x) : a + d = 2 := 
by
  sorry

end solution_l713_713237


namespace cos_120_degrees_eq_l713_713943

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l713_713943


namespace true_proposition_l713_713624

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, 3^(2 * x + 1) > 0

def q : Prop := (∀ x : ℝ, log x / log 2 < 1) ↔ (0 < x ∧ x < 2)

theorem true_proposition : p ∧ ¬q := by
  sorry

end true_proposition_l713_713624


namespace smallest_solution_l713_713164

theorem smallest_solution (x : ℝ) (h : x * |x| = 2 * x + 1) : x = -1 := 
by
  sorry

end smallest_solution_l713_713164


namespace teacher_to_gerald_ratio_l713_713172

variables (bar_squares : ℕ) (gerald_bars : ℕ) (students : ℕ) (chocolate_per_student : ℕ)

def total_squares_needed := students * chocolate_per_student
def gerald_squares := gerald_bars * bar_squares
def teacher_squares := total_squares_needed - gerald_squares
def teacher_bars := teacher_squares / bar_squares
def ratio := teacher_bars / gerald_bars

theorem teacher_to_gerald_ratio 
  (h1 : bar_squares = 8) 
  (h2 : gerald_bars = 7) 
  (h3 : students = 24) 
  (h4 : chocolate_per_student = 7) 
  : ratio = 2 :=
by {
  unfold ratio teacher_bars teacher_squares gerald_squares total_squares_needed,
  rw [h1, h2, h3, h4],
  norm_num,
  sorry
}

end teacher_to_gerald_ratio_l713_713172


namespace olympic_partition_of_nonnegatives_l713_713300

variable (a b : ℕ)
variable (ha : 0 < a)
variable (hb : a < b)

def is_olympic (s : Set ℕ) : Prop :=
  ∃ x y z, s = {x, y, z} ∧ x < y ∧ y < z ∧ {z - y, y - x} = {a, b}

def disjoint_sets (s1 s2 : Set ℕ) : Prop :=
  s1 ∩ s2 = ∅

def all_nonnegative_partition (olympic_sets : Set (Set ℕ)) : Prop :=
  (∀ s ∈ olympic_sets, is_olympic a b s) ∧ 
  (∀ s1 s2 ∈ olympic_sets, s1 ≠ s2 → disjoint_sets s1 s2) ∧
  (⋃₀ olympic_sets = Set.univ)

theorem olympic_partition_of_nonnegatives :
  ∃ olympic_sets : Set (Set ℕ), all_nonnegative_partition a b olympic_sets :=
sorry

end olympic_partition_of_nonnegatives_l713_713300


namespace propositional_logic_problem_l713_713191

theorem propositional_logic_problem 
  (p : Prop) (q : Prop) 
  (hp : ¬ (∃ x : ℝ, sin x = sqrt 5 / 2)) 
  (hq : ∀ x : ℝ, x^2 + x + 1 > 0) : 
  (¬ p ∧ q) ∧ ¬ (p ∨ ¬ q) :=
by
  split
  { split
    { exact hp },
    { exact hq }
  },
  { intro h,
    cases h,
    { exact hp h },
    { apply False.elim,
      apply hq,
      exact h }
  }

end propositional_logic_problem_l713_713191


namespace four_pairwise_distinct_sums_l713_713700

theorem four_pairwise_distinct_sums (A : Finset ℤ) (hA_card : A.card = 12) 
  (hA_range : ∀ a ∈ A, 1 ≤ a ∧ a ≤ 30) : 
  ∃ a b c d ∈ A, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b = c + d := 
by sorry

end four_pairwise_distinct_sums_l713_713700


namespace tangent_spheres_count_equivalence_l713_713590

noncomputable def num_tangent_spheres (s1 s2 s3 s4 : Sphere) : ℕ :=
  if s1.r = s2.r ∧ s2.r = s3.r ∧ s3.r = s4.r ∧ s1.center_plane ⊥ s2.center_plane ∧ s2.center_plane ⊥ s3.center_plane ∧ s3.center_plane ⊥ s4.center_plane then
    16
  else
    0

theorem tangent_spheres_count_equivalence
  (s1 s2 s3 s4 : Sphere)
  (h1 : s1.radius = s2.radius)
  (h2 : s2.radius = s3.radius)
  (h3 : s3.radius = s4.radius)
  (h4 : ¬ collinear {s1.center, s2.center, s3.center, s4.center})
  (h5 : ∀ (i j : ℕ), i ≠ j → distance (si.center, sj.center) ≠ 0) :
  (num_tangent_spheres s1 s2 s3 s4 = 16) :=
begin
  sorry
end

end tangent_spheres_count_equivalence_l713_713590


namespace people_per_column_l713_713253

theorem people_per_column :
  ∃ (P x : ℕ), (P = 16 * x ∧ P = 40 * 12 ∧ x = 30) :=
by
  use 480 30
  simp
  sorry

end people_per_column_l713_713253


namespace curve1_cartesian_curve2_cartesian_find_cos_alpha_l713_713675

   -- Conditions
   variable (α : ℝ) (t : ℝ)
   variable (x y theta ρ : ℝ)
   def curve1_param_x : ℝ := 1 + t * cos α
   def curve1_param_y : ℝ := t * sin α
   def polar_curve2 : ℝ := (ρ^2 = 12 / (3 + sin(theta)^2))

   -- Theorem statements
   theorem curve1_cartesian (h : 0 ≤ α ∧ α < real.pi) : 
     y = tan α * (x - 1) :=
   sorry

   theorem curve2_cartesian (h : ρ^2 = 12 / (3 + sin(theta)^2)) : 
     (x^2 / 4) + (y^2 / 3) = 1 :=
   sorry

   theorem find_cos_alpha (h₁ : 0 ≤ α ∧ α < real.pi) 
   (h₂ : (1 + t * cos α)^2 / 4 + (t * sin α)^2 / 3 = 1)
   (h3 : ∀ t1 t2, |t1 * t1 + t1 * t2| = 2 * |t1 * t2| → t1 + t2 = -6 * cos α / (3 + sin α^2)) : 
     cos α = 2 / 3 :=
   sorry
   
end curve1_cartesian_curve2_cartesian_find_cos_alpha_l713_713675


namespace correct_operation_l713_713066

theorem correct_operation (x : ℝ) : (-x^3)^2 = x^6 :=
by sorry

end correct_operation_l713_713066


namespace cubic_yard_to_cubic_meter_l713_713223

/-- Define the conversion from yards to meters. -/
def yard_to_meter : ℝ := 0.9144

/-- Theorem stating how many cubic meters are in one cubic yard. -/
theorem cubic_yard_to_cubic_meter :
  (yard_to_meter ^ 3 : ℝ) = 0.7636 :=
by
  sorry

end cubic_yard_to_cubic_meter_l713_713223


namespace sum_of_squares_eq_229_l713_713001

-- The conditions
variables (x y : ℤ)
axiom diff_eq_221 : x^2 - y^2 = 221

-- The proof goal
theorem sum_of_squares_eq_229 : x^2 - y^2 = 221 → ∃ x y : ℤ, x^2 + y^2 = 229 :=
by
  sorry

end sum_of_squares_eq_229_l713_713001


namespace fixed_point_of_line_l713_713201

theorem fixed_point_of_line (a : ℝ) (x y : ℝ)
  (h : ∀ a : ℝ, a * x + y + 1 = 0) :
  x = 0 ∧ y = -1 := 
by
  sorry

end fixed_point_of_line_l713_713201


namespace find_angle_C_and_range_k_l713_713184

noncomputable def triangle_angles_and_sides (A B C : ℝ) (a b c : ℝ) 
  (sin_A sin_B sin_C : ℝ) (b_c: ℝ) (a_b: ℝ) (λ: ℝ) : Prop := 
  ∃ (λ : ℝ), 
  (sin_A = λ * (sin_C - sin_B)) ∧
  (b_c = λ * a_b)

noncomputable def measure_angle_C (A B C : ℝ) (a b c : ℝ)
 (sin_A sin_B sin_C : ℝ) (b_c a_b : ℝ) (λ : ℝ) 
 (h1 : triangle_angles_and_sides A B C a b c sin_A sin_B sin_C b_c a_b λ) : Prop :=
  C = π / 3

noncomputable def range_of_k (A B C : ℝ) (a b c : ℝ)
 (sin_A sin_B sin_C : ℝ) (b_c a_b : ℝ) (λ : ℝ)  
 (k : ℝ) 
 (h1 : triangle_angles_and_sides A B C a b c sin_A sin_B sin_C b_c a_b λ) 
 (h2 : a + b = k * c) : Prop :=
  1 < k ∧ k ≤ 2

theorem find_angle_C_and_range_k (A B C a b c sin_A sin_B sin_C b_c a_b λ : ℝ) :
  triangle_angles_and_sides A B C a b c sin_A sin_B sin_C b_c a_b λ →
  measure_angle_C A B C a b c sin_A sin_B sin_C b_c a_b λ →
  ∃ k, range_of_k A B C a b c sin_A sin_B sin_C b_c a_b λ k :=
sorry

end find_angle_C_and_range_k_l713_713184


namespace technician_trip_type_l713_713901

theorem technician_trip_type (to_center return_trip : ℝ) (total_trip : ℝ)
  (H1 : return_trip = to_center) 
  (H2 : total_trip = to_center + return_trip) 
  (H3 : to_center = 0.5 * total_trip)
  (H4 : technician_completed : ℝ)
  (H5 : technician_completed = to_center + 0.4 * return_trip) : 
  technician_completed = 0.7 * total_trip :=
by 
  rw [return_trip, total_trip, H3, H4] 
  sorry

end technician_trip_type_l713_713901


namespace distance_from_origin_to_p_l713_713663

-- Define the two-dimensional points
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance function between two points
def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the origin point and the given point (12, -5)
def origin : Point2D := { x := 0, y := 0 }
def p : Point2D := { x := 12, y := -5 }

-- State the proof problem: the distance from origin to point p is 13
theorem distance_from_origin_to_p : distance origin p = 13 := by
  sorry

end distance_from_origin_to_p_l713_713663


namespace dalton_uncles_money_l713_713535

theorem dalton_uncles_money (cost_jump_rope : ℕ) (cost_board_game : ℕ) (cost_ball : ℕ) 
    (saved_money : ℕ) (still_needed : ℕ) (uncle_money : ℕ) :
  cost_jump_rope = 7 →
  cost_board_game = 12 →
  cost_ball = 4 →
  saved_money = 6 →
  still_needed = 4 →
  uncle_money = 13 :=
by
  intros h1 h2 h3 h4 h5
  let total_cost := cost_jump_rope + cost_board_game + cost_ball
  have ha : total_cost = 23 := by rw [h1, h2, h3]; norm_num
  let current_amount := total_cost - still_needed
  have hb : current_amount = 19 := by rw [ha, h5]; norm_num
  let uncle_money := current_amount - saved_money
  have hc : uncle_money = 13 := by rw [hb, h4]; norm_num
  exact hc

end dalton_uncles_money_l713_713535


namespace cos_120_eq_neg_half_l713_713975

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713975


namespace binomial_coeff_not_arithmetic_progression_l713_713078

theorem binomial_coeff_not_arithmetic_progression 
  (n k : ℤ)
  (hk : 0 ≤ k) 
  (hnk : k ≤ n - 3) :
  ¬ (∃ d : ℤ, 
    (choose n (k + 1) - choose n k = d ∧ 
    choose n (k + 2) - choose n (k + 1) = d ∧ 
    choose n (k + 3) - choose n (k + 2) = d)) :=
by {
  sorry
}

end binomial_coeff_not_arithmetic_progression_l713_713078


namespace units_digit_base7_sum_l713_713167

theorem units_digit_base7_sum (a b : ℕ) (ha : a = 52) (hb : b = 62) :
  ((a + b) % 7) = 4 :=
by
  have ha_units : a % 7 = 2, by sorry
  have hb_units : b % 7 = 2, by sorry
  have sum_units : (a + b) % 7 = (2 + 2) % 7, by sorry
  simp at sum_units,
  exact sum_units

end units_digit_base7_sum_l713_713167


namespace katya_solves_enough_l713_713295

theorem katya_solves_enough (x : ℕ) :
  (0 ≤ x ∧ x ≤ 20) → -- x should be within the valid range of problems
  (4 / 5) * x + (1 / 2) * (20 - x) ≥ 13 → 
  x ≥ 10 :=
by 
  intros h₁ h₂
  -- Formalize the expected value equation and the inequality transformations
  sorry

end katya_solves_enough_l713_713295


namespace sqrt_div_simp_l713_713438

theorem sqrt_div_simp : sqrt 50 / sqrt 2 = 5 := 
by 
  sorry

end sqrt_div_simp_l713_713438


namespace score_order_l713_713658

theorem score_order (a b c d : ℕ) 
  (h1 : b + d = a + c)
  (h2 : a + b > c + d)
  (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c := 
by
  sorry

end score_order_l713_713658


namespace closest_to_1994_in_sequence_l713_713025

-- Definition of the sequence and conditions
def sequence : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 4
| _ := sorry  -- Detailed construction skipped for brevity

-- Problem statement: Prove that 1993 or 1995 is closest to 1994 in the given sequence.
theorem closest_to_1994_in_sequence :
  ∃ n m : ℕ, sequence n = 1993 ∧ sequence m = 1995 :=
sorry

end closest_to_1994_in_sequence_l713_713025


namespace relationship_among_a_b_c_l713_713200

noncomputable def f : ℝ → ℝ := sorry  -- The function f

def a : ℝ := 4 ^ 0.2
def b : ℝ := (Real.log 3 / Real.log 4) * f (Real.log 3 / Real.log 4)
def c : ℝ := (Real.log (1 / 16) / Real.log 4) * (Real.log (1 / 16) / Real.log 4)

axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_condition (x : ℝ) (h : x > 0) : f x + x * deriv f x > 0

theorem relationship_among_a_b_c : c > a ∧ a > b := sorry

end relationship_among_a_b_c_l713_713200


namespace distance_from_A_to_focus_l713_713245

-- Definitions
def parabola : Type := {x : ℝ // ∃ y : ℝ, y^2 = 8 * x}
def focus_of_parabola : ℝ := 2
def point_A : parabola := ⟨4, ⟨12, by norm_num⟩⟩

-- The distance formula
def distance (x1 x2 : ℝ) : ℝ := abs (x1 - x2)

-- Math proof problem statement
theorem distance_from_A_to_focus : distance point_A.val focus_of_parabola = 6 := 
sorry

end distance_from_A_to_focus_l713_713245


namespace probability_of_winning_at_least_10_rubles_l713_713031

-- Definitions based on conditions
def total_tickets : ℕ := 100
def win_20_rubles_tickets : ℕ := 5
def win_15_rubles_tickets : ℕ := 10
def win_10_rubles_tickets : ℕ := 15
def win_2_rubles_tickets : ℕ := 25
def win_nothing_tickets : ℕ := total_tickets - (win_20_rubles_tickets + win_15_rubles_tickets + win_10_rubles_tickets + win_2_rubles_tickets)

-- Probability calculations
def prob_win_20_rubles : ℚ := win_20_rubles_tickets / total_tickets
def prob_win_15_rubles : ℚ := win_15_rubles_tickets / total_tickets
def prob_win_10_rubles : ℚ := win_10_rubles_tickets / total_tickets

-- Prove the probability of winning at least 10 rubles
theorem probability_of_winning_at_least_10_rubles : 
  prob_win_20_rubles + prob_win_15_rubles + prob_win_10_rubles = 0.30 := by
  sorry

end probability_of_winning_at_least_10_rubles_l713_713031


namespace parallelogram_of_isosceles_l713_713533

-- Define the conditions of the problem.
variables {A B C X Y Z : Type*}
variables (ABC : Type*) [plane_geometry ABC] [triangle ABC A B C]
variables (isosceles_triangle_ABX : isosceles_triangle ABC A B X)
variables (isosceles_triangle_BCY : isosceles_triangle ABC B C Y)
variables (isosceles_triangle_CAZ : isosceles_triangle ABC C A Z)
variables (non_collinear : ¬collinear X Y Z C)

-- Main statement to prove that the points X, Y, Z, and C form a parallelogram.
theorem parallelogram_of_isosceles {
  isosceles_triangle_ABX : isosceles_triangle ABC A B X
  isosceles_triangle_BCY : isosceles_triangle ABC B C Y
  isosceles_triangle_CAZ : isosceles_triangle ABC C A Z
  (non_collinear : ¬collinear X Y Z C)
}: parallelogram ABC X Y Z C :=
sorry

end parallelogram_of_isosceles_l713_713533


namespace arithmetic_expr_eval_l713_713464

/-- A proof that the arithmetic expression (80 + (5 * 12) / (180 / 3)) ^ 2 * (7 - (3^2)) evaluates to -13122. -/
theorem arithmetic_expr_eval : (80 + (5 * 12) / (180 / 3)) ^ 2 * (7 - (3^2)) = -13122 :=
by
  sorry

end arithmetic_expr_eval_l713_713464


namespace sum_of_common_ratios_l713_713711

variable (m x y : ℝ)
variable (h₁ : x ≠ y)
variable (h₂ : a2 = m * x)
variable (h₃ : a3 = m * x^2)
variable (h₄ : b2 = m * y)
variable (h₅ : b3 = m * y^2)
variable (h₆ : a3 - b3 = 3 * (a2 - b2))

theorem sum_of_common_ratios : x + y = 3 :=
by
  sorry

end sum_of_common_ratios_l713_713711


namespace trig_identity_proof_l713_713511

noncomputable def sin_30 : Real := 1 / 2
noncomputable def cos_120 : Real := -1 / 2
noncomputable def cos_45 : Real := Real.sqrt 2 / 2
noncomputable def tan_30 : Real := Real.sqrt 3 / 3

theorem trig_identity_proof : 
  sin_30 + cos_120 + 2 * cos_45 - Real.sqrt 3 * tan_30 = Real.sqrt 2 - 1 := 
by
  sorry

end trig_identity_proof_l713_713511


namespace polynomials_equal_at_all_x_l713_713718

variable {R : Type} [CommRing R]

def f (a_5 a_4 a_3 a_2 a_1 a_0 : R) (x : R) := a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0
def g (b_3 b_2 b_1 b_0 : R) (x : R) := b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0
def h (c_2 c_1 c_0 : R) (x : R) := c_2 * x^2 + c_1 * x + c_0

theorem polynomials_equal_at_all_x 
    (a_5 a_4 a_3 a_2 a_1 a_0 b_3 b_2 b_1 b_0 c_2 c_1 c_0 : ℤ)
    (bound_a : ∀ i ∈ [a_5, a_4, a_3, a_2, a_1, a_0], |i| ≤ 4)
    (bound_b : ∀ i ∈ [b_3, b_2, b_1, b_0], |i| ≤ 1)
    (bound_c : ∀ i ∈ [c_2, c_1, c_0], |i| ≤ 1)
    (H : f a_5 a_4 a_3 a_2 a_1 a_0 10 = g b_3 b_2 b_1 b_0 10 * h c_2 c_1 c_0 10) :
    ∀ x, f a_5 a_4 a_3 a_2 a_1 a_0 x = g b_3 b_2 b_1 b_0 x * h c_2 c_1 c_0 x := by
  sorry

end polynomials_equal_at_all_x_l713_713718


namespace factorial_division_l713_713523

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division (n : ℕ) (h : n > 0) : factorial (n) / factorial (n - 1) = n :=
by
  sorry

example : factorial 12 / factorial 11 = 12 :=
by
  exact factorial_division 12 (by norm_num)

end factorial_division_l713_713523


namespace perpendicular_segments_l713_713072

open EuclideanGeometry

-- Define the initial conditions of the problem
variable (A B C B1 C1 B2 C2 O : Point)
variable (hABC : IsTriangle A B C)
variable (hAeq : angle A B C = 30 * degree)
variable (hB1Alt : IsAltitude B1 A B C)
variable (hC1Alt : IsAltitude C1 A B C)
variable (hB2Mid : Midpoint B2 A C)
variable (hC2Mid : Midpoint C2 A B)

-- Statement that we need to prove
theorem perpendicular_segments :
  ∃ O, Collinear B1 C2 O ∧ Collinear B2 C1 O ∧ angle B2 O C2 = 90 * degree :=
by
  sorry

end perpendicular_segments_l713_713072


namespace infinite_solutions_l713_713085

theorem infinite_solutions (a b : ℕ) (h : a^2 > b) : 
  ∃ infinitely_many x : ℕ, ⌊real.sqrt (x^2 + 2*a*x + b)⌋ = x + a - 1 := 
sorry

end infinite_solutions_l713_713085


namespace factorial_division_l713_713522

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division (n : ℕ) (h : n > 0) : factorial (n) / factorial (n - 1) = n :=
by
  sorry

example : factorial 12 / factorial 11 = 12 :=
by
  exact factorial_division 12 (by norm_num)

end factorial_division_l713_713522


namespace sum_of_squares_eq_229_l713_713002

-- The conditions
variables (x y : ℤ)
axiom diff_eq_221 : x^2 - y^2 = 221

-- The proof goal
theorem sum_of_squares_eq_229 : x^2 - y^2 = 221 → ∃ x y : ℤ, x^2 + y^2 = 229 :=
by
  sorry

end sum_of_squares_eq_229_l713_713002


namespace log2_y_equals_1_l713_713645

theorem log2_y_equals_1
  (h : let y := (\log 16 / \log 4)^((\log 4 / \log 16) * (\log 4 / \log 2)) in y = 2^1) :
  \log 2^y = 1 :=
by
  sorry

end log2_y_equals_1_l713_713645


namespace screw_revolutions_screw_revolutions_correct_l713_713406

theorem screw_revolutions (depth_per_quarter_turn : ℝ) (target_depth : ℝ) : 
  (1 / 4) * depth_per_quarter_turn * 4 * 3 = target_depth :=
by
  have h : (1 / 4) * depth_per_quarter_turn * 4 = 3 * 3 := sorry
  exact h

# Tactic is missing here to be directly verifiable 
-- To match the structure: depth_per_quarter_turn = 3 and target_depth = 36
theorem screw_revolutions_correct : screw_revolutions 3 36 := sorry

end screw_revolutions_screw_revolutions_correct_l713_713406


namespace lcm_fractions_l713_713927

open Nat

-- Definitions of the fractions
def num1 := 7
def num2 := 8
def num3 := 3
def num4 := 5

def den1 := 10
def den2 := 9
def den3 := 8
def den4 := 12

-- Definition of LCM of a list of numbers
def lcm_list (l : List Nat) : Nat :=
  l.foldr lcm 1

-- Definition of GCD of a list of numbers
def gcd_list (l : List Nat) : Nat :=
  l.foldr gcd 0

-- Assertion statement
theorem lcm_fractions : 
  lcm_list [num1, num2, num3, num4] / gcd_list [den1, den2, den3, den4] = 840 := by
  sorry

end lcm_fractions_l713_713927


namespace exists_2_room_partition_l713_713874

open Classical

variables {V : Type} (G : Type)

structure Graph (V : Type) :=
  (adj : V → V → Prop)

variable [Nonempty V]

def out_degree (G : Graph V) (v : V) : ℕ :=
  (Set.toFinset (SetOf (G.adj v))).Card

def in_degree (G : Graph V) (v : V) : ℕ :=
  (Set.toFinset (SetOf (λ u, G.adj u v))).Card

def k_content (G : Graph V) (v : V) (k : ℕ) : Prop :=
  out_degree G v ≥ k ∨ in_degree G v ≥ k

def k_digraph (G : Graph V) (k : ℕ) : Prop :=
  ∀ v : V, out_degree G v ≥ k ∨ in_degree G v ≥ k

def three_k_plus_one_digraph (G : Graph V) (k : ℕ) : Prop :=
  ∀ v : V, out_degree G v ≥ 3 * k + 1 ∨ in_degree G v ≥ 3 * k + 1

theorem exists_2_room_partition {G : Graph V} (k : ℕ) 
  (H : three_k_plus_one_digraph G k) :
  ∃ (A B : Set V), A ≠ ∅ ∧ B ≠ ∅ ∧ 
  (∀ v ∈ A, k_content (Graph.mk (λ u₁ u₂, G.adj u₁ u₂ ∧ u₁ ∈ A ∧ u₂ ∈ A)) v k) ∧ 
  (∀ v ∈ B, k_content (Graph.mk (λ u₁ u₂, G.adj u₁ u₂ ∧ u₁ ∈ B ∧ u₂ ∈ B)) v k) :=
sorry

end exists_2_room_partition_l713_713874


namespace value_of_a_l713_713602

variable (x y a : ℝ)

-- Conditions
def condition1 : Prop := (x = 1)
def condition2 : Prop := (y = 2)
def condition3 : Prop := (3 * x - a * y = 1)

-- Theorem stating the equivalence between the conditions and the value of 'a'
theorem value_of_a (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 x y a) : a = 1 :=
by
  -- Insert proof here
  sorry

end value_of_a_l713_713602


namespace bob_repay_l713_713909

theorem bob_repay {x : ℕ} (h : 50 + 10 * x >= 150) : x >= 10 :=
by
  sorry

end bob_repay_l713_713909


namespace maximum_value_of_T_l713_713605

variable (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
variable [fact (q = 2)] [n ≠ 0]

noncomputable def S (n : ℕ) : ℝ :=
a 0 * (1 - (q ^ n)) / (1 - q)

noncomputable def T (n : ℕ) : ℝ :=
(9 * S a q n - S a q (2 * n)) / (a (n + 1))

theorem maximum_value_of_T :
  ∃ n, 0 < n ∧ T a q n = 3 := sorry

end maximum_value_of_T_l713_713605


namespace equivalent_operation_l713_713437

theorem equivalent_operation (x : ℚ) : 
  (x * (5 / 6) / (2 / 7)) = x * (35 / 12) :=
by
  sorry

end equivalent_operation_l713_713437


namespace log_quadratic_irrational_roots_l713_713543

theorem log_quadratic_irrational_roots :
  ∀ x : ℝ, log 10 (x^2 - 20 * x + 36) = 1 ↔ ∃ a b : ℝ, irrational a ∧ irrational b ∧ a ≠ b ∧ (x = a ∨ x = b) :=
by
  sorry

end log_quadratic_irrational_roots_l713_713543


namespace max_min_S_value_l713_713734

noncomputable def S (x y z : ℝ) : ℝ := 
  2*x^2*y^2 + 2*x^2*z^2 + 2*y^2*z^2 - x^4 - y^4 - z^4

theorem max_min_S_value (x y z : ℝ) (h1 : 5 ≤ x) (h2 : x ≤ 8) (h3 : 5 ≤ y) (h4 : y ≤ 8) (h5 : 5 ≤ z) (h6 : z ≤ 8) :
  minimum (S x y z) = 1875 ∧ maximum (S x y z) = 31488 :=
sorry

end max_min_S_value_l713_713734


namespace find_a_l713_713391

noncomputable def f (x : ℝ) := -x^2 + 2*x + 15

noncomputable def g (a : ℝ) (x : ℝ) := (2 - 2 * a) * x - f x

theorem find_a (a : ℝ) (h : ∀ x ∈ set.Icc (0 : ℝ) 2, g a x ≤ 5) : a = -4 := 
by
  sorry

end find_a_l713_713391


namespace eval_nested_expression_l713_713429

theorem eval_nested_expression : (2 * (2 * (2 * (2 * (2 * (2 + 1) + 1) + 1) + 1) + 1) + 1) = 127 :=
by
  rw [Nat.mul_add, Nat.mul_add, Nat.mul_add, Nat.mul_add, Nat.mul_add]
  rw [Nat.add_comm, Nat.succ_eq_add_one, Nat.succ_eq_add_one, Nat.succ_eq_add_one, Nat.succ_eq_add_one]
  sorry

end eval_nested_expression_l713_713429


namespace permutation_mod_l713_713701

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def count_permutations : ℕ :=
  (∑ k in Finset.range 4, binomial 5 (k + 1) * binomial 6 k * binomial 7 (k + 2))

theorem permutation_mod :
  count_permutations % 1000 = 555 :=
by
  sorry

end permutation_mod_l713_713701


namespace cos_120_degrees_eq_l713_713944

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l713_713944


namespace sum_of_three_distinct_roots_l713_713784

theorem sum_of_three_distinct_roots (a b : ℝ) 
  (h1 : a^2 - 4 * b > 0) 
  (h2 : b^2 - 4 * a > 0) 
  (h3 : ∃ r1 r2 r3 : ℝ, 
      ∀ x : ℝ, (x^2 + a * x + b) * (x^2 + b * x + a) = 0 → (x = r1 ∨ x = r2 ∨ x = r3) 
        ∧ (r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)) : 
  r1 + r2 + r3 = 0 := 
begin
  sorry
end

end sum_of_three_distinct_roots_l713_713784


namespace smallest_possible_value_l713_713074

-- Definitions and conditions provided
def x_plus_4_y_minus_4_eq_zero (x y : ℝ) : Prop := (x + 4) * (y - 4) = 0

-- Main theorem to state
theorem smallest_possible_value (x y : ℝ) (h : x_plus_4_y_minus_4_eq_zero x y) : x^2 + y^2 = 32 :=
sorry

end smallest_possible_value_l713_713074


namespace find_polynomial_l713_713708

noncomputable def P : Polynomial ℝ :=
  16 * X^3 + 64 * X^2 + 90 * X + 140

theorem find_polynomial
  (a b c : ℝ)
  (h_roots : ∀ x, x^3 + 4 * x^2 + 6 * x + 9 = 0 → (x = a ∨ x = b ∨ x = c))
  (h_sum : a + b + c = -4)
  (h_P_eq1 : P.eval a = b + c)
  (h_P_eq2 : P.eval b = a + c)
  (h_P_eq3 : P.eval c = a + b)
  (h_P_neg4 : P.eval (-4) = -20) :
  P = 16 * X^3 + 64 * X^2 + 90 * X + 140 :=
sorry

end find_polynomial_l713_713708


namespace jezebel_total_flower_cost_l713_713275

theorem jezebel_total_flower_cost :
  let red_rose_count := 2 * 12
  let red_rose_cost := 1.5
  let sunflower_count := 3
  let sunflower_cost := 3
  (red_rose_count * red_rose_cost + sunflower_count * sunflower_cost = 45) :=
by
  let red_rose_count := 2 * 12
  let red_rose_cost := 1.5
  let sunflower_count := 3
  let sunflower_cost := 3
  sorry

end jezebel_total_flower_cost_l713_713275


namespace find_f_for_negative_x_l713_713363

noncomputable def f : ℝ → ℝ :=
  λ x, if x > 0 then x - Real.log (|x|) else - ( - x - Real.log (| - x|))

theorem find_f_for_negative_x (x : ℝ) (h : x < 0) :
  f x = x + Real.log (|x|) := by
sorry

end find_f_for_negative_x_l713_713363


namespace range_of_m_l713_713190

open Real

theorem range_of_m (m : ℝ) 
  (p : m + 1 ≤ 0) 
  (q : ∀ x : ℝ, x^2 + m * x + 1 > 0) 
  (h : ¬ (p ∧ q)) :
  m ≤ -2 ∨ m > -1 :=
sorry

end range_of_m_l713_713190


namespace vec_perpendicular_parallel_magnitude_l713_713314

open Real

def vec (a b c : ℝ) := (a, b, c)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.2.2 * v.2.2

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2 + v.2.2 ^ 2)

def is_perpendicular (a c : ℝ × ℝ × ℝ) : Prop :=
  dot_product a c = 0

def is_parallel (b c : ℝ × ℝ × ℝ) : Prop :=
  b.1 / c.1 = b.2 / c.2 ∧ b.2 / c.2 = b.2.2 / c.2.2

theorem vec_perpendicular_parallel_magnitude :
  ∀ (x y : ℝ),
  let a := vec x 1 1
  let b := vec 1 y 1
  let c := vec 2 (-4) 2
  is_perpendicular a c →
  is_parallel b c →
  magnitude (a.1 + b.1, a.2 + b.2, a.2.2 + b.2.2) = 3 :=
by
  sorry

end vec_perpendicular_parallel_magnitude_l713_713314


namespace q_zero_is_neg_two_l713_713318

open Polynomial

noncomputable def p : Polynomial ℝ := ...
noncomputable def q : Polynomial ℝ := ...
noncomputable def r : Polynomial ℝ := p * q

-- Conditions
def const_term_p := constantCoeff p = 5
def const_term_r := constantCoeff r = -10

-- Theorem Statement
theorem q_zero_is_neg_two (h1 : const_term_p) (h2 : const_term_r) : q.eval 0 = -2 :=
sorry

end q_zero_is_neg_two_l713_713318


namespace total_trees_after_planting_l713_713801

def initial_trees : ℕ := 39
def trees_planted_today : ℕ := 41
def trees_planted_tomorrow : ℕ := 20

theorem total_trees_after_planting :
  initial_trees + trees_planted_today + trees_planted_tomorrow = 100 := 
by sorry

end total_trees_after_planting_l713_713801


namespace triangle_area_ratio_l713_713779

theorem triangle_area_ratio (A B C D E F K L M N: Point) :
  -- Incircle touches BC at D, AC at E, AB at F
  incircle_touches A B C D E F →
  -- Perpendicular feet from F and E to BC are K and L
  perpendicular F K D ∧ perpendicular E L D →
  -- Second intersections of these perpendiculars with the incircle are M and N
  second_intersection_with_incircle F K D M ∧ second_intersection_with_incircle E L D N →
  -- The areas ratio of triangles is the ratio of segments as stated
  area_ratio_triangle B M D C N D = segment_ratio D K D L :=
sorry

end triangle_area_ratio_l713_713779


namespace power_of_two_grows_faster_l713_713574

theorem power_of_two_grows_faster (n : ℕ) (h : n ≥ 5) : 2^n > n^2 :=
sorry

end power_of_two_grows_faster_l713_713574


namespace point_inside_circle_of_distance_lt_radius_l713_713202

/--
Suppose a circle with center O has a radius of 6 cm and point P is 4 cm from O.
Prove that point P is inside the circle.
-/
theorem point_inside_circle_of_distance_lt_radius (O P : Type*) (r d : ℝ) :
  r = 6 ∧ d = 4 → d < r :=
begin
  intros h,
  cases h with hr hd,
  rw [hr, hd],
  exact lt_add_one 5,
end

end point_inside_circle_of_distance_lt_radius_l713_713202


namespace math_books_together_l713_713831

theorem math_books_together (math_books english_books : ℕ) (h_math_books : math_books = 2) (h_english_books : english_books = 2) : 
  ∃ ways, ways = 12 := by
  sorry

end math_books_together_l713_713831


namespace Austin_started_with_l713_713510

def initial_robot_cost : ℝ := 8.75
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.08
def total_tax_paid : ℝ := 7.22
def change_left : ℝ := 11.53

def num_friends_with_one_robot : ℝ := 2
def num_friends_with_two_robots : ℝ := 3
def num_friends_with_three_robots : ℝ := 2

def robots_per_friend (one two three : ℝ) : ℝ := (one * 1) + (two * 2) + (three * 3)

theorem Austin_started_with :
    ∀ (initial_robot_cost discount_rate tax_rate total_tax_paid change_left : ℝ)
    (num_friends_with_one_robot num_friends_with_two_robots num_friends_with_three_robots : ℝ),
    let total_robots := robots_per_friend num_friends_with_one_robot num_friends_with_two_robots num_friends_with_three_robots in
    let initial_total_cost := total_robots * initial_robot_cost in
    let total_discount := initial_total_cost * discount_rate in
    let discounted_cost := initial_total_cost - total_discount in
    let cost_before_tax := total_tax_paid / tax_rate in
    let total_paid := cost_before_tax + total_tax_paid in
    let initial_amount := total_paid + change_left in
    initial_amount = 109 :=
begin
  sorry
end

end Austin_started_with_l713_713510


namespace number_of_roots_l713_713012

-- Definitions for the conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_monotonic_in_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → y ≤ a → f x ≤ f y

-- Main theorem to prove
theorem number_of_roots (f : ℝ → ℝ) (a : ℝ) (h1 : 0 < a) 
  (h2 : is_even_function f) (h3 : is_monotonic_in_interval f a) 
  (h4 : f 0 * f a < 0) : ∃ x0 > 0, f x0 = 0 ∧ ∃ x1 < 0, f x1 = 0 :=
sorry

end number_of_roots_l713_713012


namespace geometric_sequence_a3_l713_713261

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a3 
  (a : ℕ → ℝ) (h1 : a 1 = -2) (h5 : a 5 = -8)
  (h : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) : 
  a 3 = -4 :=
sorry

end geometric_sequence_a3_l713_713261


namespace find_a_given_symmetric_function_l713_713619

def function_symmetric_at_point (f : ℝ → ℝ) (p : ℝ × ℝ) := 
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f(x)

theorem find_a_given_symmetric_function :
  ∀ (a : ℝ), function_symmetric_at_point (λ x, (1 + a^2 * x) / (a^2 - x)) (1, -1) → (a = 1 ∨ a = -1) :=
by
  intros a h
  sorry

end find_a_given_symmetric_function_l713_713619


namespace part_1_solution_set_part_2_inequality_l713_713206

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|

theorem part_1_solution_set (x : ℝ) : f(x) < |x| + 1 ↔ 0 < x ∧ x < 2 := sorry

theorem part_2_inequality (x y : ℝ) (h1 : |x - y - 1| ≤ 1 / 3) (h2 : |2 * y + 1| ≤ 1 / 6) : f(x) < 1 := sorry

end part_1_solution_set_part_2_inequality_l713_713206


namespace problem_statement_l713_713629

def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x <= -3}
def R (S : Set ℝ) : Set ℝ := {x | ∃ y ∈ S, x = y}

theorem problem_statement : R (M ∪ N) = {x | x >= 1} :=
by
  sorry

end problem_statement_l713_713629


namespace cosine_120_eq_negative_half_l713_713961

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l713_713961


namespace problem_statement_l713_713644

theorem problem_statement (x y : ℕ) (h1 : x = 3) (h2 :y = 5) :
  (x^5 + 2*y^2 - 15) / 7 = 39 + 5 / 7 := 
by 
  sorry

end problem_statement_l713_713644


namespace find_height_l713_713890

-- Defining the known conditions
def length : ℝ := 3
def width : ℝ := 5
def cost_per_sqft : ℝ := 20
def total_cost : ℝ := 1240

-- Defining the unknown dimension as a variable
variable (height : ℝ)

-- Surface area formula for a rectangular tank
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

-- Given statement to prove that the height is 2 feet.
theorem find_height : surface_area length width height = total_cost / cost_per_sqft → height = 2 := by
  sorry

end find_height_l713_713890


namespace better_value_ounces_l713_713335

-- Definitions of the conditions
def largerBoxWeight : ℕ := 30
def largerBoxCost : ℝ := 4.80
def smallerBoxWeight : ℕ := 20
def smallerBoxCost : ℝ := 3.40

-- Definition to convert dollars to cents
def toCents (dollars : ℝ) : ℕ := (dollars * 100).round.to_nat

-- Proof statement (with 'sorry' indicating the proof is not provided)
theorem better_value_ounces :
  let largerPricePerOunce := largerBoxCost / largerBoxWeight
  let smallerPricePerOunce := smallerBoxCost / smallerBoxWeight
  let largerPricePerOunceInCents := toCents largerPricePerOunce
  let smallerPricePerOunceInCents := toCents smallerPricePerOunce in
  largerPricePerOunceInCents = 16 ∧ largerPricePerOunceInCents < smallerPricePerOunceInCents :=
by
  sorry

end better_value_ounces_l713_713335


namespace probability_bounds_l713_713174

variable (X : ℝ → Prop)
variable (x1 x2 : ℝ) (α β : ℝ)
variable (P : (ℝ → Prop) → ℝ)

theorem probability_bounds (h1 : P(λ X, X >= x1) = 1 - α)
                           (h2 : P(λ X, X <= x2) = 1 - β)
                           (h3 : x1 < x2) :
  P(λ X, x1 ≤ X ∧ X ≤ x2) = 1 - (α + β) :=
by
  sorry

end probability_bounds_l713_713174


namespace percentage_reduction_l713_713846

-- Definitions from the conditions
def area_of_regular_hexagon (a : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * a^2
def area_of_smaller_hexagon (a : ℝ) : ℝ := (3 * Real.sqrt 3 / 8) * a^2

-- The proof statement we need to express
theorem percentage_reduction (a : ℝ) (h : a > 0) : 
  let reduction := ((area_of_regular_hexagon a) - (area_of_smaller_hexagon a)) / (area_of_regular_hexagon a) * 100 in
  reduction = 75 :=
by
  sorry

end percentage_reduction_l713_713846


namespace percentage_difference_l713_713452

def percentage (p : ℝ) (x : ℝ) : ℝ := (p / 100) * x

theorem percentage_difference : 
  percentage 60 50 - percentage 42 30 = 17.4 :=
by
  sorry

end percentage_difference_l713_713452


namespace high_school_students_l713_713474

theorem high_school_students
    (music_only : ℕ)
    (art_only : ℕ)
    (both_music_art : ℕ)
    (neither_music_nor_art : ℕ)
    (total_students : ℕ)
    (h_music : music_only = 40)
    (h_art : art_only = 20)
    (h_both : both_music_art = 10)
    (h_neither : neither_music_nor_art = 450)
    (h_total : total_students = music_only + art_only + both_music_art + neither_music_nor_art):
    total_students = 500 :=
by
    have h_music_only : music_only - both_music_art = 30, from sorry,
    have h_art_only : art_only - both_music_art = 10, from sorry,
    exact sorry

end high_school_students_l713_713474


namespace part1_part2_part3_l713_713326

open Real

def f (a : ℝ) (x : ℝ) : ℝ :=
  a - 2 / (2^x + 1)

theorem part1 {a : ℝ} (h_odd : ∀ x, f a (-x) = -f a x) : a = 1 :=
sorry

theorem part2 {a : ℝ} : ∀ (x1 x2 : ℝ), x1 < x2 → f a x1 < f a x2 :=
sorry

theorem part3 (h_odd : ∀ x, f 1 (-x) = -f 1 x) (h_ineq : ∀ x, f 1 (k * 3^x) + f 1 (3^x - 9^x - 2) < 0) :
  k < -1 + 2 * Real.sqrt 2 :=
sorry

end part1_part2_part3_l713_713326


namespace expected_value_binomial_l713_713214

-- Assuming the random variable follows a binomial distribution with given parameters
variable (X : ℕ → ℕ) -- Assuming X is a binomial random variable function

-- Condition: X ∼ B(6, 1/2)
def binomial_X := true -- Placeholder for actual binomial distribution assumption

-- Expected value for binomial distribution
theorem expected_value_binomial : (n : ℕ) → (p : ℚ) → binomial_X → E(X) = n * p :=
  by
  intros n p h,
  -- Prove that with n = 6 and p = 1/2, E(X) = 3, assuming binomial_X holds
  specialize expected_value_binomial 6 (1/2) (binomial_X), -- Specializing the parameters
  have : E(X) = 6 * (1/2),
  exact sorry

end expected_value_binomial_l713_713214


namespace reflect_y_axis_l713_713266

theorem reflect_y_axis (x y z : ℝ) : (x, y, z) = (1, -2, 3) → (-x, y, -z) = (-1, -2, -3) :=
by
  intros
  sorry

end reflect_y_axis_l713_713266


namespace cosine_120_eq_negative_half_l713_713955

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l713_713955


namespace sum_of_real_solutions_abs_quadratic_eq_three_l713_713060

theorem sum_of_real_solutions_abs_quadratic_eq_three :
  (∑ x in {x : ℝ | |x^2 - 10x + 28| = 3}.toFinset, x) = 5 := 
sorry

end sum_of_real_solutions_abs_quadratic_eq_three_l713_713060


namespace ax_product_zero_l713_713370

theorem ax_product_zero 
  {a x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ x₁₃ : ℤ} 
  (h1 : a = (1 + x₁) * (1 + x₂) * (1 + x₃) * (1 + x₄) * (1 + x₅) * (1 + x₆) * (1 + x₇) *
           (1 + x₈) * (1 + x₉) * (1 + x₁₀) * (1 + x₁₁) * (1 + x₁₂) * (1 + x₁₃))
  (h2 : a = (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄) * (1 - x₅) * (1 - x₆) * (1 - x₇) *
           (1 - x₈) * (1 - x₉) * (1 - x₁₀) * (1 - x₁₁) * (1 - x₁₂) * (1 - x₁₃)) :
  a * x₁ * x₂ * x₃ * x₄ * x₅ * x₆ * x₇ * x₈ * x₉ * x₁₀ * x₁₁ * x₁₂ * x₁₃ = 0 := 
sorry

end ax_product_zero_l713_713370


namespace trig_identity_simplification_l713_713550

theorem trig_identity_simplification :
  (sin (40 * pi / 180) - cos (10 * pi / 180)) / 
  (sin (10 * pi / 180) - cos (40 * pi / 180)) = 
  sqrt 3 / 3 :=
by
  sorry

end trig_identity_simplification_l713_713550


namespace john_total_time_spent_l713_713693

-- Definitions based on conditions
def years_to_months (years : ℕ) : ℕ := years * 12

def time_spent_south_america_exploring : ℕ := years_to_months 3
def time_spent_africa_exploring : ℕ := years_to_months 2

def time_spent_south_america_writing_notes : ℕ := years_to_months (3 / 2)
def time_spent_africa_writing_notes : ℕ := 8 -- 8 months

def time_spent_compiling_notes : ℕ := 8 -- 8 months
def time_spent_finalizing_book_with_editor : ℕ := 6 -- 6 months

-- The main statement to prove
theorem john_total_time_spent : 
  time_spent_south_america_exploring +
  time_spent_africa_exploring +
  time_spent_south_america_writing_notes +
  time_spent_africa_writing_notes +
  time_spent_compiling_notes +
  time_spent_finalizing_book_with_editor = 100 :=
begin
  -- Insert the proof here
  sorry
end

end john_total_time_spent_l713_713693


namespace calculate_total_travel_time_l713_713797

/-- The total travel time, including stops, from the first station to the last station. -/
def total_travel_time (d1 d2 d3 : ℕ) (s1 s2 s3 : ℕ) (t1 t2 : ℕ) : ℚ :=
  let leg1_time := d1 / s1
  let stop1_time := t1 / 60
  let leg2_time := d2 / s2
  let stop2_time := t2 / 60
  let leg3_time := d3 / s3
  leg1_time + stop1_time + leg2_time + stop2_time + leg3_time

/-- Proof that total travel time is 2 hours and 22.5 minutes. -/
theorem calculate_total_travel_time :
  total_travel_time 30 40 50 60 40 80 10 5 = 2.375 :=
by
  sorry

end calculate_total_travel_time_l713_713797


namespace geometric_sequence_product_l713_713660

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r)
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_roots : (∃ a₁ a₁₉ : ℝ, (a₁ + a₁₉ = 10) ∧ (a₁ * a₁₉ = 16) ∧ a 1 = a₁ ∧ a 19 = a₁₉)) :
  a 8 * a 12 = 16 := 
sorry

end geometric_sequence_product_l713_713660


namespace vertex_closer_to_Q_than_P_l713_713270

open Metric

theorem vertex_closer_to_Q_than_P
  {α : Type*} [MetricSpace α]
  {polygon : Set α} (h_convex : Convex ℝ polygon)
  {P Q : α} (hP_in : P ∈ polygon) (hQ_in : Q ∈ polygon) :
  ∃ (V ∈ polygon), dist V Q < dist V P := 
sorry

end vertex_closer_to_Q_than_P_l713_713270


namespace twelve_factorial_div_eleven_factorial_eq_twelve_l713_713528

theorem twelve_factorial_div_eleven_factorial_eq_twelve :
  12! / 11! = 12 :=
by
  sorry

end twelve_factorial_div_eleven_factorial_eq_twelve_l713_713528


namespace common_line_planes_l713_713740

-- Definitions of the conditions in the problem
variables {p q : set (ℝ × ℝ × ℝ)} [line p] [line q]
variable A : ℝ × ℝ × ℝ
variables (M : ℝ × ℝ × ℝ) (moves_on_p : p M)

-- A projection function that projects point M onto line q
noncomputable def projection (M : ℝ × ℝ × ℝ) : q :=
  -- placeholder definition of projection; the actual function would depend on the geometry library in use
  sorry

variable N : ℝ × ℝ × ℝ
-- Define N as the projection of M onto line q
axiom proj_def : N = projection M

-- Now we state the theorem
theorem common_line_planes (p_intersects_q : intersects p q)
  (common_perpendicular_exists : ∃ P Q, common_perpendicular PQ p q A)
  (N_proj : N = projection M) :
  ∃ l : set (ℝ × ℝ × ℝ), ∀ M : ℝ × ℝ × ℝ, moves_on_p M → ∀ N : ℝ × ℝ × ℝ, N = projection M → ∀ Aₘ Nₘ : set (ℝ × ℝ × ℝ), (A ∈ Aₘ) ∧ (M ∈ Aₘ) ∧ (N ∈ Aₘ) ∧ plane Aₘ ∧ plane Nₘ → (l ⊆ Aₘ) ∧ (l ⊆ Nₘ) :=
sorry

end common_line_planes_l713_713740


namespace average_difference_is_five_l713_713353

-- Definitions and conditions based on the problem description.
def set1 := [10, 20, 60]
def set2 := [10, 40, 25]

def average (s : list ℤ) : ℤ :=
  s.sum / s.length

theorem average_difference_is_five :
  average set1 - average set2 = 5 := 
by
  sorry

end average_difference_is_five_l713_713353


namespace cos_120_degrees_l713_713998

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l713_713998


namespace system_solution_exists_l713_713920

theorem system_solution_exists (x y z : ℤ) : 
  (z + 2) * 7 ^ (Int.natAbs y - 1) = 4 ^ (x^2 + 2 * x * y + 1) ∧
  sin (3 * Real.pi * (z : ℝ) / 2) = 1 →
  (x = 1 ∧ y = -1 ∧ z = -1) ∨ (x = -1 ∧ y = 1 ∧ z = -1) :=
by
  intros h
  sorry

end system_solution_exists_l713_713920


namespace green_duck_percentage_l713_713850

noncomputable def smaller_pond_ducks : ℕ := 45
noncomputable def larger_pond_ducks : ℕ := 55
noncomputable def green_percentage_small_pond : ℝ := 0.20
noncomputable def green_percentage_large_pond : ℝ := 0.40

theorem green_duck_percentage :
  let total_ducks := smaller_pond_ducks + larger_pond_ducks
  let green_ducks_smaller := green_percentage_small_pond * (smaller_pond_ducks : ℝ)
  let green_ducks_larger := green_percentage_large_pond * (larger_pond_ducks : ℝ)
  let total_green_ducks := green_ducks_smaller + green_ducks_larger
  (total_green_ducks / total_ducks) * 100 = 31 :=
by {
  -- The proof is omitted.
  sorry
}

end green_duck_percentage_l713_713850


namespace boat_upstream_speed_l713_713456

variable (Vb Vc : ℕ)

def boat_speed_upstream (Vb Vc : ℕ) : ℕ := Vb - Vc

theorem boat_upstream_speed (hVb : Vb = 50) (hVc : Vc = 20) : boat_speed_upstream Vb Vc = 30 :=
by sorry

end boat_upstream_speed_l713_713456


namespace charlie_made_additional_shots_l713_713131

-- Define initial conditions
def initial_made_shots : ℕ := 18
def initial_total_shots : ℕ := 45
def additional_shots : ℕ := 15
def new_shooting_percentage : ℚ := 0.45

-- Define the proof statement
theorem charlie_made_additional_shots (made_initial : initial_made_shots = 18)
  (total_initial : initial_total_shots = 45)
  (additional : additional_shots = 15)
  (average : new_shooting_percentage = 0.45) :
  ∃ k, k = 9 :=
by
  use 9
  sorry

end charlie_made_additional_shots_l713_713131


namespace cricket_bat_cost_l713_713491

noncomputable def CP_A_sol : ℝ := 444.96 / 1.95

theorem cricket_bat_cost (CP_A : ℝ) (SP_B : ℝ) (SP_C : ℝ) (SP_D : ℝ) :
  (SP_B = 1.20 * CP_A) →
  (SP_C = 1.25 * SP_B) →
  (SP_D = 1.30 * SP_C) →
  (SP_D = 444.96) →
  CP_A = CP_A_sol :=
by
  intros h1 h2 h3 h4
  sorry

end cricket_bat_cost_l713_713491


namespace num_divisors_m2_less_than_m_not_divide_m_l713_713720

namespace MathProof

def m : ℕ := 2^20 * 3^15 * 5^6

theorem num_divisors_m2_less_than_m_not_divide_m :
  let m2 := m ^ 2
  let total_divisors_m2 := 41 * 31 * 13
  let total_divisors_m := 21 * 16 * 7
  let divisors_m2_less_than_m := (total_divisors_m2 - 1) / 2
  divisors_m2_less_than_m - total_divisors_m = 5924 :=
by sorry

end MathProof

end num_divisors_m2_less_than_m_not_divide_m_l713_713720


namespace solution_exists_l713_713560

theorem solution_exists (x y : ℝ) :
  (x^2 + y = 12) ∧ (y^2 + x = 12) →
  (x = 3 ∧ y = 3) ∨ (x = -4 ∧ y = -4) ∨
  (x = (1 + 3 * Real.sqrt 5) / 2 ∧ y = (1 - 3 * Real.sqrt 5) / 2) ∨
  (x = (1 - 3 * Real.sqrt. 5) / 2 ∧ y = (1 + 3 * Real.sqrt. 5) / 2) :=
by { sorry }

end solution_exists_l713_713560


namespace sum_of_first_and_fourth_l713_713029

theorem sum_of_first_and_fourth (x : ℤ) (h : x + (x + 6) = 156) : (x + 2) = 77 :=
by {
  -- This block represents the assumptions and goal as expressed above,
  -- but the proof steps are omitted.
  sorry
}

end sum_of_first_and_fourth_l713_713029


namespace speed_difference_l713_713907

noncomputable def park_distance : ℝ := 10
noncomputable def kevin_time_hours : ℝ := 1 / 4
noncomputable def joel_time_hours : ℝ := 2

theorem speed_difference : (10 / kevin_time_hours) - (10 / joel_time_hours) = 35 := by
  sorry

end speed_difference_l713_713907


namespace only_five_regular_polyhedra_l713_713751

theorem only_five_regular_polyhedra (n m : ℕ) (P B G : ℕ) (regular_polyhedron : Type)
  (h1 : regular_polyhedron.faces_are_reg_ngons n)
  (h2 : ∀ v ∈ regular_polyhedron.vertices, v.faces_count = m)
  (h3 : 2 * P = m * B)
  (h4 : 2 * P = n * G)
  (h5 : B - P + G = 2) :
  (m, n) ∈ [(3, 3), (3, 4), (4, 3), (3, 5), (5, 3)] :=
sorry

end only_five_regular_polyhedra_l713_713751


namespace lambda_range_monotonically_increasing_seq_l713_713236

theorem lambda_range_monotonically_increasing_seq (λ : ℝ):
  (∀ n : ℕ, 0 < n → let a := n^2 + λ * n + 3 in let b := (n + 1)^2 + λ * (n + 1) + 3 in a ≤ b) ↔ -3 < λ := 
sorry

end lambda_range_monotonically_increasing_seq_l713_713236


namespace number_of_memorable_phone_numbers_l713_713107

def is_memorable (d : Fin 10 → ℕ) : Prop :=
(d 0 = d 3 ∧ d 1 = d 4 ∧ d 2 = d 5 ∨ 
 d 0 = d 4 ∧ d 1 = d 5 ∧ d 2 = d 6 ∨
 d 0 = d 3 ∧ d 1 = d 4 ∧ d 2 = d 5 ∧ d 0 = d 4 ∧ d 1 = d 5 ∧ d 2 = d 6)

theorem number_of_memorable_phone_numbers : 
  let phone_numbers := {f : Fin 10 → ℕ // ∀i, f i < 10} in
  ∃ (n : ℕ), n = 19990 ∧ 
  n = card {f ∈ phone_numbers | is_memorable f} := sorry

end number_of_memorable_phone_numbers_l713_713107


namespace eccentricity_of_hyperbola_l713_713622

variables (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)

def hyperbola_eccentricity (a b : ℝ) : ℝ := (real.sqrt (a ^ 2 + b ^ 2)) / a

theorem eccentricity_of_hyperbola :
  ∀ (a b : ℝ), a > 0 → b > 0 → ∠MAN = 60 ° →
  hyperbola_eccentricity a b = 2 * real.sqrt 3 / 3 :=
by
  sorry

end eccentricity_of_hyperbola_l713_713622


namespace range_of_omega_l713_713612

theorem range_of_omega (ω : ℝ) : ∀ x : ℝ, ∃ x : ℝ, 2 * sin (ω * x) = -2 :=
by
  sorry

end range_of_omega_l713_713612


namespace prism_lateral_area_l713_713204

-- Define the prism and its properties
structure Prism :=
  (base_side : ℝ)
  (height : ℝ)

def circumscribed_sphere_volume (prism : Prism) : ℝ :=
  (4 / 3) * Real.pi

def lateral_area (prism : Prism) : ℝ :=
  4 * prism.base_side * prism.height

-- Define the given conditions
def conditions (prism : Prism) : Prop :=
  prism.base_side = 1 ∧ circumscribed_sphere_volume prism = (4 / 3) * Real.pi

-- State the theorem
theorem prism_lateral_area (prism : Prism) (h_cond : conditions prism) :
  lateral_area prism = 4 * Real.sqrt 2 := by
  sorry

end prism_lateral_area_l713_713204


namespace axis_of_symmetry_is_left_of_y_axis_l713_713065

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ :=
  (1/2) * x^2 + 2 * x + 3

-- Define the axis of symmetry
def axis_of_symmetry (qf : ℝ → ℝ) : ℝ :=
  -2

theorem axis_of_symmetry_is_left_of_y_axis :
  axis_of_symmetry quadratic_function < 0 :=
by {
  dsimp [axis_of_symmetry, quadratic_function],
  linarith,
}

end axis_of_symmetry_is_left_of_y_axis_l713_713065


namespace gym_guest_count_l713_713925

theorem gym_guest_count (G : ℕ) (H1 : ∀ G, 0 < G → ∀ G, G * 5.7 = 285 ∧ G = 50) : G = 50 :=
by
  sorry

end gym_guest_count_l713_713925


namespace simplify_expression_l713_713321

theorem simplify_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let x := (b / c + c / b) ^ 2,
      y := (a / c + c / a) ^ 2,
      z := (a / b + b / a) ^ 2 in
  x * y + y * z + z * x - Real.sqrt (x ^ 2 * y ^ 2 * z ^ 2) = 
  -- Here we expect the simplified result expression
  sorry

end simplify_expression_l713_713321


namespace probability_rain_one_day_at_least_l713_713396

open ProbabilityTheory

-- Define the given conditions
variables {P_A P_B P_B_given_A P_B_given_not_A : ℚ}
variables (h1 : P_A = 0.4)
variables (h2 : P_B = 0.3)
variables (h3 : P_B_given_A = 2 * P_B_given_not_A)

-- Define the statement that needs to be proven
theorem probability_rain_one_day_at_least (h1 : P_A = 0.4) (h2 : P_B = 0.3) (h3 : P_B_given_A = 2 * P_B_given_not_A) :
  let P_not_A := 1 - P_A,
      P_B_given_not_A := P_B / (P_A * 2 + (1 - P_A)),
      P_not_B_given_A := 1 - P_B_given_A,
      P_not_B_given_not_A := 1 - P_B_given_not_A,
      P_not_A_and_not_B := P_not_A * P_not_B_given_not_A in
  1 - P_not_A_and_not_B = 37 / 70 := by
  let P_not_A := 1 - 0.4
  let P_B_given_not_A := 0.3 / (0.4 * 2 + (1 - 0.4))
  let P_not_B_given_A := 1 - 2 * P_B_given_not_A
  let P_not_B_given_not_A := 1 - P_B_given_not_A
  let P_not_A_and_not_B := P_not_A * P_not_B_given_not_A
  show 1 - P_not_A_and_not_B = 37 / 70

end probability_rain_one_day_at_least_l713_713396


namespace angle_PBA_eq_angle_PCA_l713_713676

universe u
variables {α : Type u} [LinearOrderedField α]

-- Definitions based on given conditions
def acute_triangle (A B C : α × α) : Prop := 
  ∃ (a b c : α), a + b + c = 180 ∧ a < 90 ∧ b < 90 ∧ c < 90

def point_on_segment (D B C : α × α) : Prop := 
  D.1 > min B.1 C.1 ∧ D.1 < max B.1 C.1 ∧ D.2 >= min B.2 C.2 ∧ D.2 <= max B.2 C.2

-- Define conditions and problem
variables (A B C D E P : α × α)
variable (BD CE : α)

variables (AB_lt_AC : A.1 * B.1 + A.2 * B.2 < A.1 * C.1 + A.2 * C.2)
variables (BD_CE_eq : BD = CE)
variables (BD_eq_CE_on_BC : BD = sqrt((D.1 - B.1)^2 + (D.2 - B.2)^2) ∧ CE = sqrt((E.1 - C.1)^2 + (E.2 - C.2)^2))
variables (P_internal : P.1 > min A.1 B.1 ∧ P.1 < max A.1 B.1 ∧ P.2 > min A.2 B.2 ∧ P.2 < max A.2 B.2)
variables (PD_parallel_AE : (P.2 - D.2) / (P.1 - D.1) = (A.2 - E.2) / (A.1 - E.1))
variables (angle_PAB_eq_angle_EAC : atan2 (P.2 - A.2) (P.1 - A.1) = atan2 (E.2 - A.2) (E.1 - A.1))

-- Statement to prove
theorem angle_PBA_eq_angle_PCA : acute_triangle A B C ∧
    point_on_segment D B C ∧
    point_on_segment E B C ∧
    BD_CE_eq ∧
    PD_parallel_AE ∧
    angle_PAB_eq_angle_EAC →
    angle (P.1 - B.1) (P.2 - B.2) = angle (P.1 - C.1) (P.2 - C.2) :=
by
  sorry

end angle_PBA_eq_angle_PCA_l713_713676


namespace finite_A_n_iff_f_inequality_l713_713169

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def A_n (n : ℕ) : Set ℕ := { p : ℕ | ∃ (a b : ℕ), is_coprime (a + b) p ∧ is_coprime (a^n + b^n) (p^2) }

def f (n : ℕ) : ℕ := if n = 2 then 0 else Set.card (A_n n)

theorem finite_A_n_iff {n : ℕ} : (Set.finite (A_n n)) ↔ n ≠ 2 := 
sorry

theorem f_inequality (k m : ℕ) (hk : k % 2 = 1) (hm : m % 2 = 1) (d : ℕ) (hd : d = Nat.gcd k m) :
  f(d) ≤ f(k) + f(m) - f(km) ∧ f(k) + f(m) - f(km) ≤ 2 * f(d) :=
sorry

end finite_A_n_iff_f_inequality_l713_713169


namespace total_amount_spent_l713_713735

theorem total_amount_spent : 
  let value_half_dollar := 0.50
  let wednesday_spending := 4 * value_half_dollar
  let next_day_spending := 14 * value_half_dollar
  wednesday_spending + next_day_spending = 9.00 :=
by
  let value_half_dollar := 0.50
  let wednesday_spending := 4 * value_half_dollar
  let next_day_spending := 14 * value_half_dollar
  show _ 
  sorry

end total_amount_spent_l713_713735


namespace trajectory_is_ellipse_l713_713584

theorem trajectory_is_ellipse (M : ℝ × ℝ) (F : ℝ × ℝ) (line_y : ℝ) (ratio : ℝ) (hF : F = ⟨0, 2⟩) (hline : line_y = 8) (hratio : ratio = 1 / 2) :
  (dist M F / real.abs (M.snd - line_y) = ratio) → (M.fst ^ 2 / 12 + M.snd ^ 2 / 16 = 1) :=
by
  sorry

end trajectory_is_ellipse_l713_713584


namespace delphinia_traffic_organization_l713_713677

-- Define the problem statement
def traffic_organization_possible (n : ℕ) : Prop :=
  if n = 2 ∨ n = 4 then False else True

theorem delphinia_traffic_organization (n : ℕ) :
  (¬ (2 ∨ 4 = n)) → traffic_organization_possible n :=
begin
  intro h,
  sorry -- Proof goes here
end

end delphinia_traffic_organization_l713_713677


namespace fraction_planted_of_field_is_correct_l713_713557

/-- Given a right triangle with legs 5 units and 12 units, and a small unplanted square S
at the right-angle vertex such that the shortest distance from S to the hypotenuse is 3 units,
prove that the fraction of the field that is planted is 52761/857430. -/
theorem fraction_planted_of_field_is_correct :
  let area_triangle := (5 * 12) / 2
  let area_square := (180 / 169) ^ 2
  let area_planted := area_triangle - area_square
  let fraction_planted := area_planted / area_triangle
  fraction_planted = 52761 / 857430 :=
sorry

end fraction_planted_of_field_is_correct_l713_713557


namespace equation_represents_circle_m_condition_l713_713772

theorem equation_represents_circle_m_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - x + y + m = 0) → m < 1/2 := 
by
  sorry

end equation_represents_circle_m_condition_l713_713772


namespace factorial_ratio_l713_713514

theorem factorial_ratio : 12! / 11! = 12 := by
  sorry

end factorial_ratio_l713_713514


namespace factorial_ratio_l713_713517

theorem factorial_ratio : 12! / 11! = 12 := by
  sorry

end factorial_ratio_l713_713517


namespace bushels_needed_l713_713536

theorem bushels_needed (cows sheep chickens : ℕ) (cows_eat sheep_eat chickens_eat : ℕ) :
  cows = 4 → cows_eat = 2 →
  sheep = 3 → sheep_eat = 2 →
  chickens = 7 → chickens_eat = 3 →
  4 * 2 + 3 * 2 + 7 * 3 = 35 := 
by
  intros hc hec hs hes hch hech
  sorry

end bushels_needed_l713_713536


namespace simplified_value_l713_713430

theorem simplified_value :
  (245^2 - 205^2) / 40 = 450 := by
  sorry

end simplified_value_l713_713430


namespace product_of_x1_to_x13_is_zero_l713_713367

theorem product_of_x1_to_x13_is_zero
  (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ)
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 :=
sorry

end product_of_x1_to_x13_is_zero_l713_713367


namespace frustum_volume_frustum_lateral_surface_area_l713_713585

noncomputable def base_areas_frustum (A₁ A₂ : ℝ) (height : ℝ) : ℝ :=
1/3 * (A₁ + A₂ + Real.sqrt (A₁ * A₂)) * height

noncomputable def lateral_surface_area_frustum (r R l : ℝ) : ℝ :=
π * r * l + π * R * l

open Real

theorem frustum_volume (h₁ h₂ : ℝ) : base_areas_frustum (4 * π) (25 * π) 4 = 42 * π := by
  unfold base_areas_frustum
  sorry

theorem frustum_lateral_surface_area (h₁ h₂ : ℝ) : lateral_surface_area_frustum 2 5 5 = 35 * π := by
  unfold lateral_surface_area_frustum
  sorry

end frustum_volume_frustum_lateral_surface_area_l713_713585


namespace geometric_sequence_product_l713_713659

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r)
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_roots : (∃ a₁ a₁₉ : ℝ, (a₁ + a₁₉ = 10) ∧ (a₁ * a₁₉ = 16) ∧ a 1 = a₁ ∧ a 19 = a₁₉)) :
  a 8 * a 12 = 16 := 
sorry

end geometric_sequence_product_l713_713659


namespace sum_first_1060_terms_l713_713792

def sequence : List ℕ := [
  1, 2, 2, 1, 2, 2, 3, 1, 2, 2, 3, 4, 1, 2, 2, 3, 4, 5, 1, 2, 2, 3, 4, 5, 6, 1, -- continues
]

def sequence_sum (n : ℕ) : ℕ :=
  -- function to generate the sequence upto the n-th term and compute its sum
  (sequence.take n).sum 

theorem sum_first_1060_terms : sequence_sum 1060 = 9870 := 
  sorry

end sum_first_1060_terms_l713_713792


namespace sin_B_over_sin_A_area_of_triangle_l713_713655

variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
axiom triangle_conditions :
  a = 1 ∧ b = 2 ∧ c = 2 ∧ cos C = 1 / 4 ∧ (cos A - 2 * cos B) / cos C = (2 * b - a) / c

-- The first part of the problem: prove that sin B / sin A = 2
theorem sin_B_over_sin_A (h : triangle_conditions) : sin B / sin A = 2 :=
sorry

-- The second part of the problem: find the area S of the triangle
theorem area_of_triangle (h : triangle_conditions) : 
  let S := 1 / 2 * a * b * sin C in
  S = sqrt 15 / 4 :=
sorry

end sin_B_over_sin_A_area_of_triangle_l713_713655


namespace shift_right_pi_six_l713_713038

theorem shift_right_pi_six (x : ℝ) : 
  (2 * sin (2 * (x - (π / 6)))) = 2 * sin (2 * x - π / 3) :=
by
  sorry

end shift_right_pi_six_l713_713038


namespace sum_is_neg_24_l713_713776

-- Define the arithmetic sequence with first term a₁ = 1 and common difference d where d ≠ 0.
def a (n : ℕ) : ℤ :=
1 + (n - 1) * d

-- Define that a₂, a₃, and a₆ form a geometric sequence.
lemma a_geom_seq (h : d ≠ 0) : (a 3) ^ 2 = (a 2) * (a 6) :=
begin
  -- Substituting the expressions for a₂, a₃, and a₆ from step 1
  have a_2 := 1 + d,
  have a_3 := 1 + 2 * d,
  have a_6 := 1 + 5 * d,
  have h_geom : (1 + 2 * d) ^ 2 = (1 + d) * (1 + 5 * d),
  exact h_geom,
end

-- Define the formula for the sum of the first 6 terms of the arithmetic sequence
def sum_first_6_terms (h : d ≠ 0) : ℤ :=
3 * (2 * 1 + 5 * d)

-- Prove that the sum of the first 6 terms is -24, given the conditions.
theorem sum_is_neg_24 (h : d ≠ 0) : sum_first_6_terms h = -24 :=
begin
  -- Solve for the value of d using the equation from the geometric sequence
  have d_eq := -2,
  have sum := 3 * (2 * 1 + 5 * d_eq),
  exact sum
end

end sum_is_neg_24_l713_713776


namespace min_product_is_neg_480_l713_713824

-- Defining the set of numbers
def nums : Set ℤ := {-10, -7, -3, 1, 4, 6, 8}

-- Defining a proposition for the minimum product
def min_product_of_three (s : Set ℤ) := min (s.prod (λ (x : ℤ), x))

theorem min_product_is_neg_480 : min_product_of_three nums = -480 :=
sorry

end min_product_is_neg_480_l713_713824


namespace distance_from_origin_l713_713665

def point := (Int × Int) -- defining a point as a pair of integers

def distance (p₁ p₂ : point) : Float :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  Math.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2).toFloat

theorem distance_from_origin :
  distance (0, 0) (12, -5) = 13 :=
by
  sorry -- proof omitted

end distance_from_origin_l713_713665


namespace equilateral_triangle_has_nine_line_segments_isosceles_triangle_has_seven_line_segments_right_triangle_has_seven_line_segments_l713_713256

-- Let's define the necessary structures representing the conditions
structure Triangle where
  A : Point
  B : Point
  C : Point

structure LineSegment where
  start : Point
  end : Point

def isAltitude (t : Triangle) (l : LineSegment) : Prop := sorry
def isMedian (t : Triangle) (l : LineSegment) : Prop := sorry
def isAngleBisector (t : Triangle) (l : LineSegment) : Prop := sorry

-- Define the specific triangle types
def isEquilateralTriangle (t : Triangle) : Prop := sorry
def isIsoscelesTriangle (t : Triangle) : Prop := sorry
def isRightTriangle (t : Triangle) : Prop := sorry

-- The main equivalences we need to prove
theorem equilateral_triangle_has_nine_line_segments (t : Triangle) (h : isEquilateralTriangle t) :
  ∃ (lines : List LineSegment), (lines.filter (isAltitude t)).length + 
                               (lines.filter (isMedian t)).length + 
                               (lines.filter (isAngleBisector t)).length = 9 :=
sorry

theorem isosceles_triangle_has_seven_line_segments (t : Triangle) (h : isIsoscelesTriangle t) :
  ∃ (lines : List LineSegment), (lines.filter (isAltitude t)).length + 
                               (lines.filter (isMedian t)).length + 
                               (lines.filter (isAngleBisector t)).length = 7 :=
sorry
                               
theorem right_triangle_has_seven_line_segments (t : Triangle) (h : isRightTriangle t) :
  ∃ (lines : List LineSegment), (lines.filter (isAltitude t)).length + 
                               (lines.filter (isMedian t)).length + 
                               (lines.filter (isAngleBisector t)).length = 7 :=
sorry

end equilateral_triangle_has_nine_line_segments_isosceles_triangle_has_seven_line_segments_right_triangle_has_seven_line_segments_l713_713256


namespace solve_equation_l713_713761

theorem solve_equation (x : ℂ) :
  (x^3 + 4 * x^2 * complex.sqrt(3) + 12 * x + 4 * complex.sqrt(3) + x^2 - 1 = 0) ↔
  (x = 0 ∨ x = -complex.sqrt(3) ∨ x = (-complex.sqrt(3) + complex.i) / 2 ∨ x = (-complex.sqrt(3) - complex.i) / 2) :=
sorry

end solve_equation_l713_713761


namespace work_completion_days_l713_713482

noncomputable def original_number_of_men : ℕ := 42

noncomputable def days_to_complete_with_original_men : ℕ := 17

noncomputable def men_absent : ℕ := 8

noncomputable def remaining_number_of_men : ℕ := original_number_of_men - men_absent

theorem work_completion_days :
  ∃ X : ℕ, 42 * 17 = 34 * X :=
begin
  use 21,
  sorry
end

end work_completion_days_l713_713482


namespace probability_of_multiple_of_4_or_6_l713_713549

-- Condition: 80 ping-pong balls numbered from 1 to 80
-- Condition: A ball is chosen at random
-- Goal: Prove that the probability of selecting a ball that is a multiple of 4 or 6 or both is 27/80

theorem probability_of_multiple_of_4_or_6 :
  let balls := Finset.range 80 in
  let multiples_of_4 := balls.filter (λ n, (n + 1) % 4 = 0) in
  let multiples_of_6 := balls.filter (λ n, (n + 1) % 6 = 0) in
  let multiples_of_both := balls.filter (λ n, (n + 1) % 12 = 0) in
  let favorable_outcomes := multiples_of_4.card + multiples_of_6.card - multiples_of_both.card in
  let total_outcomes := balls.card in
  (favorable_outcomes / total_outcomes : ℚ) = 27 / 80 :=
by
  sorry

end probability_of_multiple_of_4_or_6_l713_713549


namespace sum_of_squares_of_perpendicular_segments_equals_6_times_sum_of_squares_from_centroid_l713_713703

open EuclideanGeometry

noncomputable def centroid (A B C : Point) : Point :=
(Point.baricenter [A, B, C])


theorem sum_of_squares_of_perpendicular_segments_equals_6_times_sum_of_squares_from_centroid 
  (A B C : Point)
  (S : Point := centroid A B C) :
  ∀ (p : Triangle.PerpendicularSegment A B C S), 
  ∑ (P₁ : Triangle.PerpendicularSegment A B C S), P₁^2 = 6 (dist S A)^2 + (dist S B)^2 + (dist S C)^2 := 
by sorry

end sum_of_squares_of_perpendicular_segments_equals_6_times_sum_of_squares_from_centroid_l713_713703


namespace vector_parallel_x_is_neg1_l713_713856

variables (a b : ℝ × ℝ)
variable (x : ℝ)

def vectors_parallel : Prop := 
  (a = (1, -1)) ∧ (b = (x, 1)) ∧ (a.1 * b.2 - a.2 * b.1 = 0)

theorem vector_parallel_x_is_neg1 (h : vectors_parallel a b x) : x = -1 :=
sorry

end vector_parallel_x_is_neg1_l713_713856


namespace a6_value_b_seq_term_T_n_sum_l713_713203

-- Noncomputable, need LEM (Law of Excluded Middle) and EM (excluded middle axiom)
noncomputable theory

-- Define the sequences and function
def a_seq (n : ℕ) (a2 d : ℝ) : ℝ := a2 + (n - 2) * d
def b_seq (n : ℕ) (b1 r : ℝ) : ℝ := b1 * r^(n - 1)
def f (x b1 b2 b3 : ℝ) := b1 * x^2 + b2 * x + b3

-- Conditions
axiom a_seq_arith : ∀ n : ℕ, 1 ≤ n → ∃ d : ℝ, a_seq n (-7/2) d = (if n = 2 then -7/2 else x) ∧ d ≠ 0
axiom b_seq_geom : ∀ n : ℕ, ∃ r : ℝ, b_seq n b1 r * b_seq n b1 r = b_seq n b1 r
axiom b3_neg4 : b3 = -4
axiom f_max_value : ∀ (b1 b2 b3 : ℝ) (a6 : ℝ), f (-(b2 / (2 * b1))) b1 b2 b3 = a6 - 7/2
axiom f_equal : ∀ (b1 b2 b3 : ℝ) (a2 a8 a3 a11 : ℝ), f (a2 + a8) b1 b2 b3 = f (a3 + a11) b1 b2 b3
axiom a2_neg7_2 : a_seq 2 (-7/2) d = -7/2 

-- Proof stubs (with goals)
theorem a6_value : ∃ a6 : ℝ, a6 = 1/2 := sorry

theorem b_seq_term : ∃ r : ℝ, ∀ n : ℕ, b_seq n b1 r = -(-2)^(n-1) := sorry

theorem T_n_sum (n : ℕ) : ∃ T_n : ℝ, T_n = (-4 * n) / (9 * (2 * n - 9)) := sorry

end a6_value_b_seq_term_T_n_sum_l713_713203


namespace arithmetic_properties_l713_713186

noncomputable def arithmetic_sequence (a₁ d : ℕ) : ℕ → ℕ
| n := a₁ + (n - 1) * d

noncomputable def sum_of_arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ :=
n * (2 * a₁ + (n - 1) * d) / 2

noncomputable def b_n (n : ℕ) : Rat :=
1 / ((arithmetic_sequence 3 2 n)^2 - 1 : ℝ)

noncomputable def sum_of_b_n (n : ℕ) : Rat :=
∑ i in Finset.range n, b_n i

theorem arithmetic_properties :
  (arithmetic_sequence 3 2 3 = 7) →
  (arithmetic_sequence 3 2 5 + arithmetic_sequence 3 2 7 = 26) →
  (∀ n, arithmetic_sequence 3 2 n = 2 * n + 1) ∧
  (∀ n, sum_of_arithmetic_sequence 3 2 n = n^2 + 2 * n) ∧
  (∀ n, sum_of_b_n n = n / (4 * (n + 1))) :=
by
  intros h1 h2
  have a₁   : ℕ := 3
  have d    : ℕ := 2
  have a_n_def   : ∀ n, arithmetic_sequence a₁ d n = 2 * n + 1 :=
    by sorry
  have S_n_def   : ∀ n, sum_of_arithmetic_sequence a₁ d n = n^2 + 2 * n :=
    by sorry
  have T_n_def   : ∀ n, sum_of_b_n n = n / (4 * (n + 1)) :=
    by sorry
  exact ⟨a_n_def, S_n_def, T_n_def⟩

end arithmetic_properties_l713_713186


namespace perpendicular_circumcenters_l713_713885

noncomputable def circumcenter (A B C : Point) : Point := sorry

def is_perpendicular_bisector (P Q : Point) : Prop := sorry

theorem perpendicular_circumcenters
  (A B C A1 C1 O O1 : Point)
  (h1 : triangle A B C)
  (h2 : is_perpendicular_bisector A C A1 C1)
  (h3 : O = circumcenter A B C)
  (h4 : O1 = circumcenter A1 B C1) :
  Line.through C1 O1 ⊥ Line.through A O := sorry

end perpendicular_circumcenters_l713_713885


namespace polynomial_coeff_sums_l713_713382

theorem polynomial_coeff_sums (g h : ℤ) (d : ℤ) :
  (7 * d^2 - 3 * d + g) * (3 * d^2 + h * d - 8) = 21 * d^4 - 44 * d^3 - 35 * d^2 + 14 * d - 16 →
  g + h = -3 :=
by
  sorry

end polynomial_coeff_sums_l713_713382


namespace better_value_price_l713_713337

def price_per_ounce (cost : ℕ) (weight : ℕ) : ℕ :=
  cost / weight

theorem better_value_price (
  (large_box_cost : ℕ) (large_box_weight : ℕ)
  (small_box_cost : ℕ) (small_box_weight : ℕ)
  (h_large_cost : large_box_cost = 480)
  (h_large_weight : large_box_weight = 30)
  (h_small_cost : small_box_cost = 340)
  (h_small_weight : small_box_weight = 20) :
  min (price_per_ounce large_box_cost large_box_weight) (price_per_ounce small_box_cost small_box_weight) = 16 := by
  sorry

end better_value_price_l713_713337


namespace sum_of_prime_factors_of_expr_l713_713572

theorem sum_of_prime_factors_of_expr :
  let expr := 7^7 - 7^4 + 2^2,
      prime_factors := {2, 3, 7, 19},
      sum_prime_factors := 2 + 3 + 7 + 19
  in prime_factors.sum = sum_prime_factors :=
by
  let expr := 7^7 - 7^4 + 2^2
  let prime_factors := {2, 3, 7, 19}
  let sum_prime_factors := 2 + 3 + 7 + 19
  sorry

end sum_of_prime_factors_of_expr_l713_713572


namespace complex_conjugate_imaginary_l713_713650

theorem complex_conjugate_imaginary (m : ℝ) (h : ∃ (m : ℝ), z = m^2 + m + (m + 1) * complex.i ∧ z.im ≠ 0 ∧ z.re = 0) : complex.conj (m^2 + m + (m + 1) * complex.i) = -complex.i := 
sorry

end complex_conjugate_imaginary_l713_713650


namespace cos_120_degrees_eq_l713_713949

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l713_713949


namespace solve_system_l713_713922

noncomputable def solution_existence (x y z : ℤ) : Prop :=
  4^(x^2 + 2*x*y + 1) = (z + 2) * 7^(abs y - 1) ∧
  sin (3 * Real.pi * z / 2) = 1

theorem solve_system : ∀ x y z : ℤ, 
  solution_existence x y z → (x, y, z) = (1, -1, -1) ∨ (x, y, z) = (-1, 1, -1) :=
by
  intros
  sorry

end solve_system_l713_713922


namespace three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3_l713_713062

theorem three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3 :
  ∃ x1 x2 x3 : ℕ, ((x1 = 414 ∧ x2 = 444 ∧ x3 = 474) ∧ 
  (∀ n, (100 * 4 + 10 * n + 4 = x1 ∨ 100 * 4 + 10 * n + 4 = x2 ∨ 100 * 4 + 10 * n + 4 = x3) 
  → (100 * 4 + 10 * n + 4) % 3 = 0)) :=
by
  sorry

end three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3_l713_713062


namespace num_ways_to_remove_blocks_l713_713331

-- Definitions based on the problem conditions
def stack_blocks := 85
def block_layers := [1, 4, 16, 64]

-- Theorem statement
theorem num_ways_to_remove_blocks : 
  (∃ f : (ℕ → ℕ), 
    (∀ n, f n = if n = 0 then 1 else if n ≤ 4 then n * f (n - 1) + 3 * (f (n - 1) - 1) else 4^3 * 16) ∧ 
    f 5 = 3384) := sorry

end num_ways_to_remove_blocks_l713_713331


namespace roots_eqn_values_l713_713712

theorem roots_eqn_values : 
  ∀ (x1 x2 : ℝ), (x1^2 + x1 - 4 = 0) ∧ (x2^2 + x2 - 4 = 0) ∧ (x1 + x2 = -1)
  → (x1^3 - 5 * x2^2 + 10 = -19) := 
by
  intros x1 x2
  intros h
  sorry

end roots_eqn_values_l713_713712


namespace part1_expectation_is_9_over_2_part2_employees_and_bonus_l713_713478

noncomputable def lottery_expectation (red_balls_initial white_balls_initial: ℕ) : ℚ :=
  let total_balls := red_balls_initial + white_balls_initial
  let p_X3 := 1 / (total_balls.choose 3)
  let p_X4 := (red_balls_initial.choose 1 * white_balls_initial.choose 2) / (total_balls.choose 3)
  let p_X5 := (red_balls_initial.choose 2 * white_balls_initial.choose 1) / (total_balls.choose 3)
  let p_X6 := 1 / (total_balls.choose 3)
  3 * p_X3 + 4 * p_X4 + 5 * p_X5 + 6 * p_X6

theorem part1_expectation_is_9_over_2 (red_balls_initial white_balls_initial: ℕ) 
  (h1: red_balls_initial = 3) (h2: white_balls_initial = 3) : 
  lottery_expectation red_balls_initial white_balls_initial = 9 / 2 :=
by
  have total_balls := red_balls_initial + white_balls_initial
  rw [h1, h2]
  simp [lottery_expectation, Nat.choose]
  sorry

def normal_distribution_exceeds_probability (mean variance threshold : ℚ) : ℚ :=
  let σ := sqrt variance
  let probability := (1 - 0.6827) / 2 -- Given from reference data
  if threshold > mean + σ then probability else 0

theorem part2_employees_and_bonus (employees : ℕ) (mean variance threshold : ℚ)
  (h_mean: mean = 100) (h_variance: variance = 225) (h_threshold: threshold = 115) : 
  ∃ (num_eligible_employees : ℕ) (average_bonus : ℚ),
  num_eligible_employees = 159 ∧ 
  average_bonus = 4500 / 159 :=
by
  have σ := sqrt variance
  have probability_exceeds := normal_distribution_exceeds_probability mean variance threshold
  rw [h_mean, h_variance, h_threshold]
  have num_eligible_employees := round (employees * probability_exceeds)
  have total_bonus := 4.5 * 1000 -- From Part 1 result
  have average_bonus := total_bonus / num_eligible_employees
  exists num_eligible_employees, average_bonus
  sorry

end part1_expectation_is_9_over_2_part2_employees_and_bonus_l713_713478


namespace cos_120_eq_neg_half_l713_713989

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l713_713989


namespace length_segment_ab_l713_713041

theorem length_segment_ab (A B C D E : Point) (rA rB : ℝ)
  (hA : rA = 5) (hB : rB = 3)
  (tangent_at_C : externally_tangent A B C rA rB)
  (tangent_DE : is_tangent_common_external D E A B rA rB C) :
  distance A B = 8 := by
  sorry

end length_segment_ab_l713_713041


namespace inequality_satisfaction_l713_713235

theorem inequality_satisfaction (a b : ℝ) (h : a < 0) : (a < b) ∧ (a^2 + b^2 > 2) :=
by
  sorry

end inequality_satisfaction_l713_713235


namespace polygon_sides_eq_five_l713_713017

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem polygon_sides_eq_five (n : ℕ) (h : n - number_of_diagonals n = 0) : n = 5 :=
by
  sorry

end polygon_sides_eq_five_l713_713017


namespace cos_120_eq_neg_half_l713_713937

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713937


namespace min_AP_l713_713303

noncomputable def A : ℝ × ℝ := (2, 0)
noncomputable def B' : ℝ × ℝ := (8, 6)
def parabola (P' : ℝ × ℝ) : Prop := P'.2^2 = 8 * P'.1

theorem min_AP'_plus_BP' : 
  ∃ P' : ℝ × ℝ, parabola P' ∧ (dist A P' + dist B' P' = 12) := 
sorry

end min_AP_l713_713303


namespace max_pages_min_left_over_l713_713149

-- Definitions based on given conditions
def printing_letter (x : ℕ) : ℕ := if x = 0 then 1 else if x < 10 then 1 else if x < 100 then 2 else 3

-- Condition: To print twelve pages of a book, the required digits are as follows: 1,2,3,4,5,6,7,8,9,1,0,1,1,1,2.
def twelve_pages_digits : list ℕ := [1,2,3,4,5,6,7,8,9,1,0,1,1,1,2]

-- Condition: 2011 printing letters are taken from the storeroom.
def total_lead_types : ℕ := 2011

-- Goal: Proving maximum number of pages and minimum number of printing letters left over.
theorem max_pages_min_left_over (total_lead_types : ℕ := 2011) : 
  exists (pages : ℕ) (left_over : ℕ), pages = 706 ∧ left_over = 1 :=
by
  sorry

end max_pages_min_left_over_l713_713149


namespace find_lengths_of_segments_l713_713567

variable (b c : ℝ)

theorem find_lengths_of_segments (CK AK AB CT AC AT : ℝ)
  (h1 : CK = AK + AB)
  (h2 : CK = (b + c) / 2)
  (h3 : CT = AC - AT)
  (h4 : AC = b) :
  AT = (b + c) / 2 ∧ CT = (b - c) / 2 := 
sorry

end find_lengths_of_segments_l713_713567


namespace find_m_l713_713080

theorem find_m (m : ℕ) : 5 ^ m = 5 * 25 ^ 2 * 125 ^ 3 ↔ m = 14 := by
  sorry

end find_m_l713_713080


namespace cos_120_eq_neg_half_l713_713972

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l713_713972


namespace correct_operation_l713_713439

theorem correct_operation (a : ℕ) :
  (a^2 * a^3 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(a^6 / a^2 = a^3) ∧ ¬(3 * a^2 - 2 * a = a^2) :=
by
  sorry

end correct_operation_l713_713439


namespace orthocenter_on_circumcircle_l713_713358

open EuclideanGeometry

noncomputable def circumscribed_circle (A B C : Point) := sorry
noncomputable def extension (A B : Point) : Point := sorry
noncomputable def angle_bisector (A B C : Point) : Line := sorry
noncomputable def orthocenter (A B C : Point) : Point := sorry

theorem orthocenter_on_circumcircle
  {A B C W Ω: Point}
  (h₀ : circumscribed_circle A B C = Ω)
  (B₁ : Point)
  (h₁ : line_through A B₁ = extension A B)
  (h₂ : dist A B₁ = dist A C)
  (h₃ : angle_bisector A B C = line_through A W)
  (h₄ : W ∈ Ω) :
  orthocenter A W B₁ ∈ Ω :=
sorry

end orthocenter_on_circumcircle_l713_713358


namespace jogger_distance_ahead_l713_713876

-- Define the jogger's speed in km/hr
def jogger_speed_kmh : ℝ := 9

-- Define the train's speed in km/hr
def train_speed_kmh : ℝ := 45

-- Convert speeds from km/hr to m/s
def convert_kmh_to_mps (speed_kmh: ℝ) : ℝ := speed_kmh * 1000 / 3600

def jogger_speed_mps : ℝ := convert_kmh_to_mps jogger_speed_kmh

def train_speed_mps : ℝ := convert_kmh_to_mps train_speed_kmh

-- Define the length of the train in meters
def train_length_m : ℝ := 210

-- Define the time taken for the train to pass the jogger in seconds
def time_to_pass_s : ℝ := 45

-- Calculate the relative speed in m/s
def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps

-- Define the proof statement
theorem jogger_distance_ahead : 
  (relative_speed_mps * time_to_pass_s - train_length_m) = 240 :=
by
  -- Lean does not compute the proof, so it should be manually proven
  sorry

end jogger_distance_ahead_l713_713876


namespace sam_pennies_l713_713754

def pennies_from_washing_clothes (total_money_cents : ℤ) (quarters : ℤ) : ℤ :=
  total_money_cents - (quarters * 25)

theorem sam_pennies :
  pennies_from_washing_clothes 184 7 = 9 :=
by
  sorry

end sam_pennies_l713_713754


namespace marble_probability_l713_713097

theorem marble_probability :
  let total_marbles := 4 + 5 + 11
  let prob_red := 4 / total_marbles
  let prob_green := 5 / (total_marbles - 1)
  let prob_white := 11 / (total_marbles - 2)
  (prob_red * prob_green * prob_white) = 11 / 342 :=
by {
  let total_marbles := 4 + 5 + 11
  let prob_red := 4 / total_marbles
  let prob_green := 5 / (total_marbles - 1)
  let prob_white := 11 / (total_marbles - 2)
  rw [show total_marbles = 20 from rfl],
  rw [show prob_red = 4 / 20 from rfl],
  rw [show prob_green = 5 / 19 from rfl],
  rw [show prob_white = 11 / 18 from rfl],
  norm_num,
  ring,
  norm_num,
  exact rfl
}

end marble_probability_l713_713097


namespace angle_DHQ_of_triangle_DEF_altitudes_l713_713267

theorem angle_DHQ_of_triangle_DEF_altitudes 
  (DEF : Triangle ℝ) 
  (D E F H P Q : Point ℝ) 
  (h1: altitude D P DEF) 
  (h2: altitude E Q DEF) 
  (h3: orthocenter H DEF) 
  (h4: angle DEF = 48) 
  (h5: angle DFE = 85) : 
  angle DHQ = 85 :=
sorry

end angle_DHQ_of_triangle_DEF_altitudes_l713_713267


namespace unique_measures_of_A_l713_713374

theorem unique_measures_of_A : 
  ∃ n : ℕ, n = 17 ∧ 
    (∀ A B : ℕ, 
      (A > 0) ∧ (B > 0) ∧ (A + B = 180) ∧ (∃ k : ℕ, A = k * B) → 
      ∃! A : ℕ, A > 0 ∧ (A + B = 180)) :=
sorry

end unique_measures_of_A_l713_713374


namespace jacket_sale_savings_l713_713924

theorem jacket_sale_savings :
  let regular_price : ℝ := 80
  let regular_total : ℝ := 3 * regular_price
  let discounted_total : ℝ := regular_price 
                          + (regular_price - 0.25 * regular_price) 
                          + (regular_price - 0.60 * regular_price)
  let savings : ℝ := regular_total - discounted_total
  let savings_percentage : ℝ := (savings / regular_total) * 100
  savings_percentage = 28 :=
by 
  sorry

end jacket_sale_savings_l713_713924


namespace inequality_1_inequality_2_l713_713715

variables {A1 A2 A3 A4 : Type*} [metric_space A1] [metric_space A2] [metric_space A3] [metric_space A4]
variables (G A1' A2' A3' A4' G_A1 G_A2 G_A3 G_A4 : ℝ)
variables (G_A1' G_A2' G_A3' G_A4' : ℝ)
variables (circumradius : ℝ)
variables (centroid : ℝ)

-- We assume the metric space on vertices and all points are provided
-- The variables G_A1, G_A2, ... represent distances between the centroid and vertices or intersection points

theorem inequality_1
  (h : G_A1 + G_A2 + G_A3 + G_A4 = 4 * centroid) : 
  G_A1 * G_A2 + G_A3 + G_A4 ≤ G_A1' + G_A2' + G_A3' + G_A4' := sorry

theorem inequality_2 
  ( h_circumsphere : G_A1' = circumradius ∧ 
                      G_A2' = circumradius ∧ 
                      G_A3' = circumradius ∧ 
                      G_A4' = circumradius) :
  1 / G_A1' + 1 / G_A2' + 1 / G_A3' + 1 / G_A4' ≤ 
  1 / G_A1 + 1 / G_A2 + 1 / G_A3 + 1 / G_A4 := sorry

end inequality_1_inequality_2_l713_713715


namespace parabola_circle_tangent_l713_713623

theorem parabola_circle_tangent :
  (∀ t : ℝ, ∃ x y : ℝ, x = 8 * t^2 ∧ y = 8 * t) →
  (∃ line : ℝ → ℝ, (∀ t : ℝ, line t = t - 2) ∧ ∃ (focus_x focus_y : ℝ), focus_x = 2 ∧ focus_y = 0 ∧ 
  (line focus_x = focus_y) ∧
  ∀ r : ℝ, r > 0 → ∃ circle_center_x circle_center_y : ℝ, circle_center_x = 4 ∧
  circle_center_y = 0 ∧ (line circle_center_x = circle_center_y) ∧
  (x - 4) ^ 2 + y ^ 2 = r ^ 2 → r = sqrt 2) sorry

end parabola_circle_tangent_l713_713623


namespace felicity_collecting_weeks_l713_713152

-- Define the conditions
def fort_total_sticks : ℕ := 400
def fort_completion_percent : ℝ := 0.60
def store_visits_per_week : ℕ := 3

-- Define the proof problem
theorem felicity_collecting_weeks :
  let collected_sticks := (fort_completion_percent * fort_total_sticks).to_nat
  in collected_sticks / store_visits_per_week = 80 := by
  -- This will be proven in the proof section, currently left as sorry
  sorry

end felicity_collecting_weeks_l713_713152


namespace books_left_after_garage_sale_l713_713339

theorem books_left_after_garage_sale
  (total_books : ℕ)
  (fantasy_novels : ℕ)
  (romance_novels : ℕ)
  (science_fiction_novels : ℕ)
  (sell_percent_fantasy : ℚ)
  (sell_percent_romance : ℚ)
  (sell_percent_scifi : ℚ) :
  total_books = 250 →
  fantasy_novels = 80 →
  romance_novels = 120 →
  science_fiction_novels = 50 →
  sell_percent_fantasy = 0.35 →
  sell_percent_romance = 0.50 →
  sell_percent_scifi = 0 →
  (fantasy_novels - (fantasy_novels * sell_percent_fantasy).natFloor +
   romance_novels - (romance_novels * sell_percent_romance).natFloor +
   science_fiction_novels - (science_fiction_novels * sell_percent_scifi).natFloor) = 162 :=
by
  intros h_total_books h_fantasy_novels h_romance_novels h_science_fiction_novels h_sell_percent_fantasy h_sell_percent_romance h_sell_percent_scifi
  sorry

end books_left_after_garage_sale_l713_713339


namespace polynomial_evaluation_l713_713716

theorem polynomial_evaluation (Q : ℝ → ℝ) (a : ℕ → ℕ) (n : ℕ)
  (hQ : ∀ x, Q x = ∑ i in finset.range (n + 1), a i * x^i)
  (h_coeff : ∀ i, 0 ≤ a i ∧ a i < 3)
  (h_eval : Q (real.sqrt 3) = 20 + 17 * real.sqrt 3) :
  Q 2 = 86 :=
by
  -- placeholders to match the required structure
  sorry

end polynomial_evaluation_l713_713716


namespace volleyball_last_place_score_l713_713670

theorem volleyball_last_place_score (n : ℕ) (h : n ≥ 2) 
  (points : Fin n → ℕ) 
  (h_arith_prog : ∃ a d, ∀ i : Fin n, points i = a + (i : ℕ) * d)
  (h_total_points : ∑ i in Finset.finRange n, points ⟨i, sorry⟩ = n * (n - 1) / 2) :
  ∃ i : Fin n, points i = 0 :=
by 
  use 0
  sorry

end volleyball_last_place_score_l713_713670


namespace probability_of_theta_is_correct_l713_713810

def is_in_interval (θ : ℝ) : Prop := θ ∈ Ioc 0 (π / 2)

noncomputable def probability_theta_in_interval : ℚ :=
  let outcomes := { (m, n) : ℕ × ℕ // 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6 }
  let favorable := { (m, n) ∈ outcomes | m ≥ n }
  favorable.card / outcomes.card

theorem probability_of_theta_is_correct :
  probability_theta_in_interval = 7 / 12 := by sorry

end probability_of_theta_is_correct_l713_713810


namespace surface_area_sphere_of_prism_l713_713892

-- Definitions and conditions based on the problem statement
def triangular_prism (V: ℝ) (AB: ℝ) :=
  V = 3 * real.sqrt 3 ∧ AB = real.sqrt 3

-- Main theorem to prove the sphere's surface area
theorem surface_area_sphere_of_prism 
  (V: ℝ) (AB: ℝ) (AA1: ℝ) (R: ℝ)
  (h: triangular_prism V AB)
  (h_height: AA1 = 4)
  (h_radius: R = real.sqrt (13/3)):
  4 * real.pi * R^2 = 52 * real.pi / 3 :=
by
  sorry

end surface_area_sphere_of_prism_l713_713892


namespace soccer_team_matches_l713_713125

noncomputable def number_of_matches (y x : ℕ) : ℕ :=
let x := 0.40 * y in
let total_before := y in
let total_after := y + 12 in
if (x + 8) / total_after = 0.55 then
    total_after
else
    0  -- Since we know the conditions satisfy the problem, this will not be the actual case of zero

theorem soccer_team_matches (total_matches : ℕ) :
  ∃ y x, (0.40 * y = x) ∧ ((x + 8) / (y + 12) = 0.55) ∧ (total_matches = y + 12) :=
sorry

end soccer_team_matches_l713_713125


namespace construct_segment_a_minus_c_l713_713669

noncomputable def lengths_of_triangle (a b c : ℝ) : Prop :=
  ∃ (A B C I : Point) (D E F : Point),
    is_triangle A B C ∧
    side_lengths A B C = (a, b, c) ∧
    incenter A B C = I ∧
    tangent_points I A B C D E F ∧
    segment_length A C (a - c)

theorem construct_segment_a_minus_c (a b c : ℝ) (A B C I D E F : Point) :
  lengths_of_triangle a b c →
  ∃ G H K : Point, 
    construct_with_ruler (G H K : Point) 3 (segment_length_true G H (a - c)).

end construct_segment_a_minus_c_l713_713669


namespace find_large_number_l713_713162

theorem find_large_number (L S : ℕ) 
  (h1 : L - S = 50000) 
  (h2 : L = 13 * S + 317) : 
  L = 54140 := 
sorry

end find_large_number_l713_713162


namespace slower_speed_is_10_l713_713886

-- Define the problem conditions
def walked_distance (faster_speed slower_speed actual_distance extra_distance : ℕ) : Prop :=
  actual_distance / slower_speed = (actual_distance + extra_distance) / faster_speed

-- Define main statement to prove
theorem slower_speed_is_10 (actual_distance : ℕ) (extra_distance : ℕ) (faster_speed : ℕ) (slower_speed : ℕ) :
  walked_distance faster_speed slower_speed actual_distance extra_distance ∧ 
  faster_speed = 15 ∧ extra_distance = 15 ∧ actual_distance = 30 → slower_speed = 10 :=
by
  intro h
  sorry

end slower_speed_is_10_l713_713886


namespace sdr_count_eq_fibonacci_l713_713589

noncomputable def fibonacci (n : ℕ) : ℝ := 
  (1 / Real.sqrt 5) * ( ( (1 + Real.sqrt 5) / 2 ) ^ n ) - (1 / Real.sqrt 5) * ( ( (1 - Real.sqrt 5) / 2 ) ^ n )

theorem sdr_count_eq_fibonacci (n : ℕ) (h : n ≥ 2) : 
  let u : ℕ → ℝ := λ n, match n with
                        | 0 => 1
                        | 1 => 1
                        | k+2 => u k + u (k+1)
                        end
  in u n = fibonacci (n + 1) :=
by
  sorry

end sdr_count_eq_fibonacci_l713_713589


namespace expression_equals_25_l713_713250

-- Define the conditions
variables (x y : ℝ)
hypotheses (h1 : x + y = 25) (h2 : x * y = 1)

-- Statement we need to prove
theorem expression_equals_25 (x y: ℝ) (h1: x + y = 25) (h2: x * y = 1) : x + y = 25 :=
by sorry

end expression_equals_25_l713_713250


namespace kiwi_lemon_relationship_l713_713763

open Nat

-- Define the conditions
def total_fruits : ℕ := 58
def mangoes : ℕ := 18
def pears : ℕ := 10
def pawpaws : ℕ := 12
def lemons_in_last_two_baskets : ℕ := 9

-- Define the question and the proof goal
theorem kiwi_lemon_relationship :
  ∃ (kiwis lemons : ℕ), 
  kiwis = lemons_in_last_two_baskets ∧ 
  lemons = lemons_in_last_two_baskets ∧ 
  kiwis + lemons = total_fruits - (mangoes + pears + pawpaws) :=
sorry

end kiwi_lemon_relationship_l713_713763


namespace sqrt_mean_square_geq_mean_l713_713045

theorem sqrt_mean_square_geq_mean (n : ℕ) (a : Fin n → ℝ) :
  sqrt (∑ i : Fin n, a i ^ 2 / n) ≥ (∑ i : Fin n, a i) / n := by
  sorry

end sqrt_mean_square_geq_mean_l713_713045


namespace cosine_120_eq_negative_half_l713_713957

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l713_713957


namespace treasure_chest_coins_l713_713496

theorem treasure_chest_coins :
  ∃ n : ℕ, (n % 8 = 6) ∧ (n % 9 = 5) ∧ (n ≥ 0) ∧
  (∀ m : ℕ, (m % 8 = 6) ∧ (m % 9 = 5) → m ≥ 0 → n ≤ m) ∧
  (∃ r : ℕ, n = 11 * (n / 11) + r ∧ r = 3) :=
by
  sorry

end treasure_chest_coins_l713_713496


namespace twelve_factorial_div_eleven_factorial_eq_twelve_l713_713527

theorem twelve_factorial_div_eleven_factorial_eq_twelve :
  12! / 11! = 12 :=
by
  sorry

end twelve_factorial_div_eleven_factorial_eq_twelve_l713_713527


namespace matrix_inverse_l713_713161

variable {R : Type*} [CommRing R]

def matrix_2x2 := Matrix (Fin 2) (Fin 2) R

def A : matrix_2x2 := ![
  ![5, -3],
  ![3, -2]
]

def inv_A : matrix_2x2 := ![
  ![2, -3],
  ![3, -5]
]

theorem matrix_inverse :
  A * inv_A = 1 ∧ inv_A * A = 1 := 
sorry

end matrix_inverse_l713_713161


namespace triangle_partition_l713_713338

noncomputable def point_on_segment (A B : ℝ × ℝ) (k : ℝ) : ℝ × ℝ :=
  ((1 - k) * A.1 + k * B.1, (1 - k) * A.2 + k * B.2)

theorem triangle_partition (A B C : ℝ × ℝ) :
  ∃ (D E F G H : ℝ × ℝ),
    point_on_segment A B (1/6) = D ∧
    point_on_segment B C (1/6) = E ∧
    point_on_segment C A (1/6) = F ∧
    point_on_segment A C (2 / 5) = G ∧
    point_on_segment D E (1 / 2) = H ∧
    (∃ ℓ1 ℓ2 : line, line.through D E ℓ1 ∧ line.through D F ℓ2 ∧ (∃ ℓ3 : line, line.through G H ℓ3 ∧
    (is_similar_triangle_ABC D E F) ∧ (is_similar_triangle_ABC G H E))) :=
sorry

end triangle_partition_l713_713338


namespace find_WH_l713_713674

-- Define the problem conditions using Lean statements
variable (EFGH : Rectangle) 
variable (J : Point) (FG VW : Line) (FJ JV : Length) (EJH : Angle)
variable (JH : Line) (K : Point) (L : Point) (JKE : Triangle) (JE EK JK : Length)
variable (wh : Length)

-- Conditions
axiom angle_EJH_90 : EJH = 90
axiom VW_perpendicular_FG : Perpendicular VW FG
axiom FJ_eq_JV : FJ = JV
axiom JH_intersects_VW_at_K : JH.intersects VW = K
axiom LE_passes_through_K : LE.intersects K → LE.passes K
axiom lengths_in_triangle_JKE : (JE = 15) ∧ (EK = 20) ∧ (JK = 25)

-- The question rewritten as a theorem to be proven
theorem find_WH : wh = 15 / 7 := by
  sorry

end find_WH_l713_713674


namespace probability_carol_first_six_l713_713501

theorem probability_carol_first_six : 
  ∀ (toss_probability : ℚ),
    toss_probability = 1 / 6 → 
    ∃ (probability_carol_first : ℚ),
      probability_carol_first = 125 / 671 := 
by
  intro toss_probability h_toss_probability
  use 125 / 671
  sorry

end probability_carol_first_six_l713_713501


namespace fraction_of_price_l713_713480

theorem fraction_of_price (d : ℝ) : d * 0.65 * 0.70 = d * 0.455 :=
by
  sorry

end fraction_of_price_l713_713480


namespace range_of_b_div_a_l713_713403

theorem range_of_b_div_a (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
(h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  (2 / 3 : ℝ) ≤ b / a ∧ b / a ≤ (3 / 2 : ℝ) :=
sorry

end range_of_b_div_a_l713_713403


namespace cos_120_eq_neg_half_l713_713991

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l713_713991


namespace cos_120_degrees_eq_l713_713950

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l713_713950


namespace cos_120_eq_neg_half_l713_713988

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l713_713988


namespace solution1_solution2_l713_713086

section Problem1

variable {x : ℝ}

def inequality1 := -2 * x^2 - x + 6 ≥ 0

theorem solution1 : { x : ℝ | -2 * x^2 - x + 6 ≥ 0 } = set.Icc (-2 : ℝ) (3/2 : ℝ) :=
sorry

end Problem1

section Problem2

variable {x k : ℝ}

def inequality2 := x^2 - 2 * x + k^2 - 1 > 0

theorem solution2 : (∀ x : ℝ, x^2 - 2 * x + k^2 - 1 > 0) ↔ (k > Real.sqrt 2 ∨ k < - Real.sqrt 2) :=
sorry

end Problem2

end solution1_solution2_l713_713086


namespace actual_price_of_food_before_tax_and_tip_l713_713871

theorem actual_price_of_food_before_tax_and_tip 
  (total_paid : ℝ)
  (tip_percentage : ℝ)
  (tax_percentage : ℝ)
  (pre_tax_food_price : ℝ)
  (h1 : total_paid = 132)
  (h2 : tip_percentage = 0.20)
  (h3 : tax_percentage = 0.10)
  (h4 : total_paid = (1 + tip_percentage) * (1 + tax_percentage) * pre_tax_food_price) :
  pre_tax_food_price = 100 :=
by sorry

end actual_price_of_food_before_tax_and_tip_l713_713871


namespace fraction_of_water_is_one_fourth_l713_713472

-- Definitions for the problem
def total_weight : ℝ := 23.999999999999996
def sand_weight : ℝ := (1/3) * total_weight
def gravel_weight : ℝ := 10
def water_weight : ℝ := total_weight - sand_weight - gravel_weight

-- The fraction of water in the mixture
def fraction_water : ℝ := water_weight / total_weight

-- The statement we want to prove
theorem fraction_of_water_is_one_fourth :
  fraction_water = 1 / 4 :=
by
  -- Here we would insert the proof steps, but we use sorry to skip the proof.
  sorry

end fraction_of_water_is_one_fourth_l713_713472


namespace percent_of_a_is_b_l713_713243

variable {a b c : ℝ}

theorem percent_of_a_is_b (h1 : c = 0.25 * a) (h2 : c = 0.10 * b) : b = 2.5 * a :=
by sorry

end percent_of_a_is_b_l713_713243


namespace sequence_count_l713_713077

def a (n : ℕ) : ℕ
def b (n : ℕ) : ℕ
def c (n : ℕ) : ℕ

axiom base_cases : a 1 = 1 ∧ b 1 = 1 ∧ c 1 = 1
axiom recurrence_relations :
  (∀ n, a (n + 1) = a n + b n) ∧
  (∀ n, b (n + 1) = a n + b n + c n) ∧
  (∀ n, c (n + 1) = b n + c n)

theorem sequence_count (n : ℕ) :
  a n + b n + c n = (1 / 2 : ℝ) * ((1 + real.sqrt 2) ^ (n + 1) + (1 - real.sqrt 2) ^ (n + 1)) :=
by
  sorry

end sequence_count_l713_713077


namespace three_digit_not_multiple_of_5_or_9_l713_713229

theorem three_digit_not_multiple_of_5_or_9 : 
  let three_digit_numbers := {n | 100 ≤ n ∧ n ≤ 999},
      multiples_of_5 := {n ∈ three_digit_numbers | n % 5 = 0},
      multiples_of_9 := {n ∈ three_digit_numbers | n % 9 = 0},
      multiples_of_45 := {n ∈ three_digit_numbers | n % 45 = 0},
      not_multiple_of_5_or_9 := three_digit_numbers \ (multiples_of_5 ∪ multiples_of_9)
  in 
  (set.card not_multiple_of_5_or_9 = 640) :=
by
  sorry

end three_digit_not_multiple_of_5_or_9_l713_713229


namespace students_unable_to_partner_l713_713402

/-- 
Three different 6th grade classes are combining for a square dancing unit. The first class has 17 males and 13 females, while the second class has 
14 males and 18 females, and the third class has 15 males and 17 females. Prove that the number of students who cannot partner with a student of the 
opposite gender is 2.
-/
theorem students_unable_to_partner : 
  let males1 := 17
  let females1 := 13
  let males2 := 14
  let females2 := 18
  let males3 := 15
  let females3 := 17
  let total_males := males1 + males2 + males3
  let total_females := females1 + females2 + females3
  total_females - total_males = 2 := 
by
  let males1 := 17
  let females1 := 13
  let males2 := 14
  let females2 := 18
  let males3 := 15
  let females3 := 17
  let total_males := males1 + males2 + males3
  let total_females := females1 + females2 + females3
  have h1 : total_males = 46 := by sorry
  have h2 : total_females = 48 := by sorry
  show total_females - total_males = 2 from
    calc
      total_females - total_males = 48 - 46 : by rw [h1, h2]
      ... = 2 : by norm_num

end students_unable_to_partner_l713_713402


namespace det_projection_matrix_l713_713306

noncomputable def projection_matrix (v : ℝ × ℝ) : ℝ × ℝ → ℝ × ℝ :=
  let θ := real.atan2 v.2 v.1 in
  λ p, (real.cos θ ^ 2 * p.1 + real.cos θ * real.sin θ * p.2, real.cos θ * real.sin θ * p.1 + real.sin θ ^ 2 * p.2)

def det (A : ℝ × ℝ → ℝ × ℝ) := (λ p, p.1) A (1, 0) * (λ p, p.2) A (0, 1) - (λ p, p.2) A (1, 0) * (λ p, p.1) A (0, 1)

theorem det_projection_matrix : 
  let P := projection_matrix (4, -7) in det P = 0 :=
by 
  let P := projection_matrix (4, -7) 
  have h1 : (λ p, p.1) P (1, 0) = real.cos (real.atan2 (-7) 4) ^ 2 := sorry
  have h2 : (λ p, p.2) P (0, 1) = real.sin (real.atan2 (-7) 4) ^ 2 := sorry
  have h3 : (λ p, p.2) P (1, 0) = real.cos (real.atan2 (-7) 4) * real.sin (real.atan2 (-7) 4) := sorry
  have h4 : (λ p, p.1) P (0, 1) = real.cos (real.atan2 (-7) 4) * real.sin (real.atan2 (-7) 4) := sorry
  rw [det, h1, h2, h3, h4]
  simp only [mul_eq_zero, sub_self]
  exact add_zero 0

end det_projection_matrix_l713_713306


namespace walking_speed_back_l713_713880

theorem walking_speed_back (d : ℝ) (to_speed : ℝ) (total_time : ℝ) (w_speed : ℝ) :
  d = 19.999999999999996 →
  to_speed = 25 →
  total_time = 5 + 48 / 60 →
  ((2 * d) / to_speed + d / w_speed = total_time) →
  w_speed = 4 :=
by
  intro h1 h2 h3 h4
  have h5 : 19.999999999999996 = 20 := sorry
  rw [h1, h5] at h4
  sorry

end walking_speed_back_l713_713880


namespace card_picking_ways_l713_713328

theorem card_picking_ways (left_cards : ℕ) (right_cards : ℕ)
  (h_left : left_cards = 15) (h_right : right_cards = 20) : left_cards + right_cards = 35 :=
by {
  rw [h_left, h_right],
  exact rfl,
}

end card_picking_ways_l713_713328


namespace smallest_possible_sum_of_squares_l713_713008

theorem smallest_possible_sum_of_squares : 
  ∃ (x y : ℕ), (x^2 - y^2 = 221) ∧ (∀(a b : ℕ), (a^2 - b^2 = 221) → (x^2 + y^2 ≤ a^2 + b^2)) ∧ (x^2 + y^2 = 229) :=
begin
  sorry, -- Proof omitted as per instructions
end

end smallest_possible_sum_of_squares_l713_713008


namespace parabola_directrix_l713_713361

theorem parabola_directrix (y : ℝ) : 
  x = -((1:ℝ)/4)*y^2 → x = 1 :=
by 
  sorry

end parabola_directrix_l713_713361


namespace count_divisible_by_four_in_64_rows_Pascal_l713_713558

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Condition to check if the binomial coefficient is divisible by four
def divisible_by_four (n k : ℕ) : Prop := (binom n k) % 4 = 0

-- The main theorem
theorem count_divisible_by_four_in_64_rows_Pascal: 
  (List.range 65).sum (λ n => (List.range (n+1)).count (divisible_by_four n)) = 946 := 
sorry

end count_divisible_by_four_in_64_rows_Pascal_l713_713558


namespace num_valid_distributions_for_top_square_multiple_of_4_l713_713111

theorem num_valid_distributions_for_top_square_multiple_of_4 
  (x : Fin 12 -> Fin 2) 
  (k : ℕ) 
  (h : 1 ≤ k ∧ k ≤ 12)
  : 
  ( ∑ i in fin_range 12, nat.choose 11 i * x i.val ) % 4 = 0 
  -> 16 * 2^9 = 8192 :=
by
  sorry

end num_valid_distributions_for_top_square_multiple_of_4_l713_713111


namespace inclination_angle_range_l713_713383

theorem inclination_angle_range (α : ℝ) (h : -1 ≤ sin α ∧ sin α ≤ 1) :
  ∃ θ : set ℝ, θ = {θ | θ ∈ [0, π) ∧ (θ ∈ [0, π/4] ∨ θ ∈ [3*π/4, π))} :=
by
  sorry

end inclination_angle_range_l713_713383


namespace sequence_properties_l713_713216

theorem sequence_properties :
  (∀ n, S n = (1/2) * a (n + 1) - 2^n + (3/2)) ∧
  (a 1, a 2 + 1, a 3 in arithmetic_sequence) →
  (a 1 = -1) ∧ (a 2 = -1) ∧ (a 3 = 1) ∧ 
  (∀ n, a n = 3^(n-1) - 2^n) :=
by
  sorry

end sequence_properties_l713_713216


namespace midpoint_equidistant_from_AB_and_CD_l713_713357

-- Define the Trapezoid ABCD with bases BC and AD
structure Trapezoid (A B C D : Type) := 
  (BC AD : line) -- AD and BC are the bases
  (AB : line)
  (CD : line)

-- Define the points M and N as the intersections of the specified angle bisectors
structure Points (M N K : Type) := 
  (M_on_bisector_C_ext_A : M)
  (N_on_bisector_B_ext_D : N)
  (midpoint_MN : K = (M + N) / 2)

-- Define the equidistance property of midpoint K from lines AB and CD
def Equidistant {K : Type} (AB CD : line) (K : K) :=
  distance K AB = distance K CD

theorem midpoint_equidistant_from_AB_and_CD
  {A B C D M N K : Type}
  (AB : line) (BC : line) (CD : line)
  (trapezoid : Trapezoid A B C D)
  (points : Points M N K)
  (midpoint_property : Equidistant AB CD points.midpoint_MN) :
  Equidistant AB CD points.midpoint_MN := 
sorry

end midpoint_equidistant_from_AB_and_CD_l713_713357


namespace bushels_needed_l713_713539

theorem bushels_needed (cows : ℕ) (bushels_per_cow : ℕ)
                       (sheep : ℕ) (bushels_per_sheep : ℕ)
                       (chickens : ℕ) (bushels_per_chicken : ℕ) :
  (cows = 4) → (bushels_per_cow = 2) →
  (sheep = 3) → (bushels_per_sheep = 2) →
  (chickens = 7) → (bushels_per_chicken = 3) →
  (cows * bushels_per_cow + sheep * bushels_per_sheep + chickens * bushels_per_chicken = 35) :=
begin
  sorry
end

end bushels_needed_l713_713539


namespace cos_120_eq_neg_half_l713_713940

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713940


namespace find_f1_l713_713175

noncomputable def f (x : ℝ) : ℝ :=
sorry  -- This will be defined within the proof using conditions

theorem find_f1 (f' : ℝ → ℝ) (f'_1 : ℝ) (h1 : ∀ x, f x = f'_1 * x^2 + real.exp x) (h2 : f'_1 = -(real.exp 1)) :
  f 1 = 0 :=
sorry

end find_f1_l713_713175


namespace quadrupled_container_volume_l713_713495

theorem quadrupled_container_volume (r h : ℝ) (V : ℝ) (pi_pos : Real.pi > 0) (three_gallon_vol : V = Real.pi * r ^ 2 * h) :
  (Real.pi * (4 * r) ^ 2 * (4 * h)) = 192 :=
by
  -- Conditions
  have original_volume : V = 3 := by
    -- These assumptions are part of our conditions, so we skip the proof here.
    sorry

  have new_radius : 4 * r = 4 * r := by
    -- Trivial equality
    refl

  have new_height : 4 * h = 4 * h := by
    -- Trivial equality
    refl
  
  -- However, for a complete theorem, we will assume a few things:
  -- - r and h are strictly positive
  -- - The calculations are performed correctly, and we skip intermediate steps
  let new_volume : ℝ := Real.pi * (4 * r) ^ 2 * (4 * h)
  suffices h1 : new_volume = 64 * Real.pi * r ^ 2 * h, from
    calc new_volume = 64 * Real.pi * r ^ 2 * h : by exact h1
                       ... = 64 * V                : by rw [three_gallon_vol]
                       ... = 64 * 3                : by rw [original_volume]
                       ... = 192                   : by norm_num
  sorry

end quadrupled_container_volume_l713_713495


namespace P_is_neg5_or_2_l713_713741

-- Define points A and B
def A : ℝ := 1
def B : ℝ := -4

-- Define the function that calculates the distance between two points on the number line
def dist (x y : ℝ) : ℝ := abs (x - y)

-- Define the condition for point P
def condition (P : ℝ) : Prop :=
  dist P A + dist P B = 7

-- The statement to be proved: 
theorem P_is_neg5_or_2 (P : ℝ) (h : condition P) : P = -5 ∨ P = 2 :=
by
  sorry

end P_is_neg5_or_2_l713_713741


namespace smallest_possible_norm_l713_713706

-- Defining the vector \begin{pmatrix} -2 \\ 4 \end{pmatrix}
def vec_a : ℝ × ℝ := (-2, 4)

-- Condition: the norm of \mathbf{v} + \begin{pmatrix} -2 \\ 4 \end{pmatrix} = 10
def satisfies_condition (v : ℝ × ℝ) : Prop :=
  (Real.sqrt ((v.1 + vec_a.1) ^ 2 + (v.2 + vec_a.2) ^ 2)) = 10

noncomputable def smallest_norm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem smallest_possible_norm (v : ℝ × ℝ) (h : satisfies_condition v) : smallest_norm v = 10 - 2 * Real.sqrt 5 := by
  sorry

end smallest_possible_norm_l713_713706


namespace area_leq_semiperimeter_l713_713684

-- Definitions
def convex_figure (Φ : set (ℝ × ℝ)) : Prop := 
  convex ℝ Φ ∧ measurable_set Φ

def semiperimeter (Φ : set (ℝ × ℝ)) : ℝ := sorry  -- Needs proper definition for semiperimeter

-- Statement
theorem area_leq_semiperimeter {Φ : set (ℝ × ℝ)} (h1 : convex_figure Φ) 
    (h2 : ∀ (x y : ℤ), ¬ ((↑x, ↑y) ∈ Φ)) :
  measure_theory.measure.volume Φ ≤ semiperimeter Φ := 
sorry

end area_leq_semiperimeter_l713_713684


namespace length_of_wire_l713_713845

-- Define the problem conditions
def radius_sphere : ℝ := 12
def radius_wire : ℝ := 8
def volume_sphere (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius ^ 3
def volume_wire (radius length : ℝ) : ℝ := Real.pi * radius ^ 2 * length

-- State the theorem
theorem length_of_wire : 
  let V_s := volume_sphere radius_sphere in
  let V_w (h : ℝ) := volume_wire radius_wire h in
  ∀ h : ℝ, V_s = V_w h → h = 36 := 
by
  intros h V_s_eq_V_w_h
  sorry

end length_of_wire_l713_713845


namespace problem_7_sqrt_13_l713_713604

theorem problem_7_sqrt_13 : 
  let m := Int.floor (Real.sqrt 13)
  let n := 10 - Real.sqrt 13 - Int.floor (10 - Real.sqrt 13)
  m + n = 7 - Real.sqrt 13 :=
by
  sorry

end problem_7_sqrt_13_l713_713604


namespace find_principal_l713_713447

/-- Given a simple interest of Rs. 4016.25, a rate of 1% per annum, and a time period of 5 years, 
    we need to find the principal amount. -/
theorem find_principal (SI R T : ℝ) (h1 : SI = 4016.25) (h2 : R = 1) (h3 : T = 5) :
  let P := SI * 100 / (R * T) in P = 80325 :=
by
  -- Define principal P as given in the condition
  let P := SI * 100 / (R * T)
  -- State that this P equals 80325
  have : P = 80325 := sorry
  exact this

end find_principal_l713_713447


namespace calc3aMinus4b_l713_713631

theorem calc3aMinus4b (a b : ℤ) (h1 : a * 1 - b * 2 = -1) (h2 : a * 1 + b * 2 = 7) : 3 * a - 4 * b = 1 :=
by
  /- Proof goes here -/
  sorry

end calc3aMinus4b_l713_713631


namespace children_play_time_equal_l713_713254

-- Definitions based on the conditions in the problem
def totalChildren := 7
def totalPlayingTime := 140
def playersAtATime := 2

-- The statement to be proved
theorem children_play_time_equal :
  (playersAtATime * totalPlayingTime) / totalChildren = 40 := by
sorry

end children_play_time_equal_l713_713254


namespace second_train_length_l713_713815

theorem second_train_length:
  ∀ (speed1_kmph speed2_kmph : ℝ) (time_seconds length1_meters : ℝ),
  speed1_kmph = 42 →
  speed2_kmph = 30 →
  time_seconds = 21.998240140788738 →
  length1_meters = 120 →
  let speed1_mps := speed1_kmph * (5 / 18),
      speed2_mps := speed2_kmph * (5 / 18),
      relative_speed_mps := speed1_mps + speed2_mps,
      total_distance := relative_speed_mps * time_seconds,
      length2_meters := total_distance - length1_meters
  in length2_meters ≈ 319.96480281577476 :=
by
  intros speed1_kmph speed2_kmph time_seconds length1_meters speed1_eq speed2_eq time_eq len1_eq,
  let speed1_mps := speed1_kmph * (5 / 18),
  let speed2_mps := speed2_kmph * (5 / 18),
  let relative_speed_mps := speed1_mps + speed2_mps,
  let total_distance := relative_speed_mps * time_seconds,
  let length2_meters := total_distance - length1_meters,
  have approx_eq : length2_meters ≈ 319.96480281577476 := sorry,
  exact approx_eq

end second_train_length_l713_713815


namespace smallest_value_among_options_l713_713135

def Q (x : ℝ) : ℝ := x^4 - (1/2) * x^3 + (3/2) * x^2 - 5 * x + 6

theorem smallest_value_among_options :
  let Q0 := Q 0,
      sum_coeffs := 1 - (1 / 2) + (3 / 2) - 5 + 6,
      prod_zeros := 6,
      sum_real_zeros := 6,  -- placeholder, more detailed calculation needed
      prod_non_real_zeros := 1 in -- placeholder, more detailed calculation needed
  min Q0 (min sum_coeffs (min prod_zeros (min sum_real_zeros prod_non_real_zeros))) = 
    prod_non_real_zeros := sorry

end smallest_value_among_options_l713_713135


namespace radio_tube_probability_l713_713104

section probability

variables (H1 H2 H3 : Prop)
variables (P : Prop → ℝ)
variables (A : Prop)
variables (p1 p2 p3 : ℝ) (pA_H1 pA_H2 pA_H3 : ℝ)

noncomputable def P_A : ℝ :=
  p1 * pA_H1 + p2 * pA_H2 + p3 * pA_H3

theorem radio_tube_probability
  (h1 : P(H1) = p1)
  (h2 : P(H2) = p2)
  (h3 : P(H3) = p3)
  (hA_H1 : P(A ∧ H1) / P(H1) = pA_H1)
  (hA_H2 : P(A ∧ H2) / P(H2) = pA_H2)
  (hA_H3 : P(A ∧ H3) / P(H3) = pA_H3) :
  P_A p1 p2 p3 0.9 0.8 0.7 = 0.77 :=
begin
  -- Proof goes here
  sorry
end

end probability

end radio_tube_probability_l713_713104


namespace cos_120_eq_neg_half_l713_713934

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713934


namespace average_weight_of_24_boys_l713_713394

theorem average_weight_of_24_boys (A : ℝ) : 
  (24 * A + 8 * 45.15) / 32 = 48.975 → A = 50.25 :=
by
  intro h
  sorry

end average_weight_of_24_boys_l713_713394


namespace right_angle_triangle_of_p_eq_2R_plus_r_varphi_is_angle_of_triangle_l713_713849

-- Part (a)
theorem right_angle_triangle_of_p_eq_2R_plus_r (R r p : ℝ) (h : p = 2 * R + r) : 
  ∃ (a b c : ℝ), a ^ 2 + b ^ 2 = c ^ 2 ∧ (a + b + c) / 2 = R ∧ (b * r) / (a + b - r) = r := 
sorry

-- Part (b)
theorem varphi_is_angle_of_triangle (R r p : ℝ) (varphi : ℝ) (h1 : p = 2 * R * sin varphi 
+ r * cot (varphi / 2)) (h2 : 0 < varphi) (h3 : varphi < π) :
  ∃ (A B C : ℝ), A + B + C = π ∧ (sin A = sin varphi ∨ sin B = sin varphi ∨ sin C = sin varphi) := 
sorry

end right_angle_triangle_of_p_eq_2R_plus_r_varphi_is_angle_of_triangle_l713_713849


namespace g_extreme_values_l713_713462

-- Definitions based on the conditions
def f (x : ℝ) := x^3 - 2 * x^2 + x
def g (x : ℝ) := f x + 1

-- Theorem statement
theorem g_extreme_values : 
  (g (1/3) = 31/27) ∧ (g 1 = 1) := sorry

end g_extreme_values_l713_713462


namespace right_triangle_unique_hypotenuse_l713_713490

theorem right_triangle_unique_hypotenuse (p q : ℕ) (m n k : ℕ) 
  (hp_prime : p.prime)
  (hq_prime : q.prime)
  (hm_pos : m > 0)
  (hn_pos : n > 0)
  (hk_pos : k > 0)
  (a : ℕ := p^m)
  (b : ℕ := q^n)
  (c : ℕ := 2 * k + 1)
  (pythagorean : p^2 * m + q^2 * n = c^2) :
  (c = 5 ∧ ((a = 4 ∧ b = 3) ∨ (a = 3 ∧ b = 4))) :=
by 
  sorry

end right_triangle_unique_hypotenuse_l713_713490


namespace cos_120_eq_neg_half_l713_713981

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713981


namespace orthic_triangle_angles_and_orthocenter_l713_713390

-- Assuming we have a triangle ABC with vertices A, B, and C.
variables {A B C D E F H : Type}

-- Define the orthic triangle DEF as the feet of the altitudes from vertices A, B, and C.
def orthic_triangle (ABC : triangle A B C) : Prop :=
  foot_of_altitude A B C D ∧ foot_of_altitude B A C E ∧ foot_of_altitude C A B F

-- Define the cyclic quadrilaterals involving the orthic triangle.
def cyclic_quadrilaterals (ABC : triangle A B C) : Prop :=
  is_cyclic {A, D, E, F} ∧ is_cyclic {B, D, E, F} ∧ is_cyclic {C, D, E, F}

-- Define the angle conditions based on the properties of cyclic quadrilaterals.
def angle_properties_of_cyclic_quadrilaterals (ABC : triangle A B C) (DEF : triangle D E F) : Prop :=
  angle DFE + angle B = 90 ∧
  angle DEF + angle C = 90 ∧
  angle EFD + angle A = 90

-- Define the orthocenter condition.
def orthocenter_condition (ABC : triangle A B C) (DEF : triangle D E F) : Prop :=
  orthocenter ABC H ∧ orthocenter DEF H

-- Problem statement: Prove the angles of the orthic triangle DEF and the position of the orthocenter H.
theorem orthic_triangle_angles_and_orthocenter
  (ABC : triangle A B C)
  (H : orthocenter_condition ABC (orthic_triangle ABC)) :
  ∀ (DEF : triangle D E F), 
    orthic_triangle ABC → cyclic_quadrilaterals ABC → angle_properties_of_cyclic_quadrilaterals ABC DEF ∧ orthocenter_condition ABC DEF :=
by sorry

end orthic_triangle_angles_and_orthocenter_l713_713390


namespace johns_new_weekly_earnings_l713_713280

-- Define the original weekly earnings and the percentage increase as given conditions:
def original_weekly_earnings : ℕ := 60
def percentage_increase : ℕ := 50

-- Prove that John's new weekly earnings after the raise is 90 dollars:
theorem johns_new_weekly_earnings : original_weekly_earnings + (percentage_increase * original_weekly_earnings / 100) = 90 := by
sorry

end johns_new_weekly_earnings_l713_713280


namespace cos_120_degrees_l713_713994

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l713_713994


namespace first_nonzero_digit_of_one_over_157_l713_713416

theorem first_nonzero_digit_of_one_over_157 : 
  (∀ n, 157 * n < 1000 → (n > 0 ∧ (1000 * 1) / (157 * n) > 10^floor(log10( (1000 of 157 * n))) - 10^(1) = 3) := 
by sorry

end first_nonzero_digit_of_one_over_157_l713_713416


namespace cosine_120_eq_negative_half_l713_713956

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l713_713956


namespace average_tickets_per_day_l713_713129

def total_revenue : ℕ := 960
def price_per_ticket : ℕ := 4
def number_of_days : ℕ := 3

theorem average_tickets_per_day :
  (total_revenue / price_per_ticket) / number_of_days = 80 := 
sorry

end average_tickets_per_day_l713_713129


namespace area_enclosed_by_abs_graph_l713_713052

theorem area_enclosed_by_abs_graph : 
  (∃ (x y : ℝ), abs (2 * x) + abs (3 * y) = 12) →
  ∑ (x y : ℝ), abs (2 * x) + abs (3* y) = 48 := by
  sorry

end area_enclosed_by_abs_graph_l713_713052


namespace system_of_equations_solution_l713_713159

/-- Integer solutions to the system of equations:
    \begin{cases}
        xz - 2yt = 3 \\
        xt + yz = 1
    \end{cases}
-/
theorem system_of_equations_solution :
  ∃ (x y z t : ℤ), 
    x * z - 2 * y * t = 3 ∧ 
    x * t + y * z = 1 ∧
    ((x = 1 ∧ y = 0 ∧ z = 3 ∧ t = 1) ∨
     (x = -1 ∧ y = 0 ∧ z = -3 ∧ t = -1) ∨
     (x = 3 ∧ y = 1 ∧ z = 1 ∧ t = 0) ∨
     (x = -3 ∧ y = -1 ∧ z = -1 ∧ t = 0)) :=
by {
  sorry
}

end system_of_equations_solution_l713_713159


namespace percentage_of_uninsured_part_time_l713_713668

noncomputable def number_of_employees := 330
noncomputable def uninsured_employees := 104
noncomputable def part_time_employees := 54
noncomputable def probability_neither := 0.5606060606060606

theorem percentage_of_uninsured_part_time:
  (13 / 104) * 100 = 12.5 := 
by 
  -- Here you can assume proof steps would occur/assertions to align with the solution found
  sorry

end percentage_of_uninsured_part_time_l713_713668


namespace weight_of_b_l713_713851

theorem weight_of_b (a b c : ℝ) (h1 : a + b + c = 135) (h2 : a + b = 80) (h3 : b + c = 82) : b = 27 :=
by
  sorry

end weight_of_b_l713_713851


namespace tiffany_bags_total_l713_713037

-- Given conditions:
variables (n : ℕ)

-- Prove the total number of bags Tiffany had altogether
theorem tiffany_bags_total (n : ℕ) : n + 4 = n + 4 :=
begin
  sorry
end

end tiffany_bags_total_l713_713037


namespace exists_not_prime_in_sequence_l713_713749

open Nat

theorem exists_not_prime_in_sequence (a b x0 : ℕ) (h_a : 0 < a) (h_b : 0 < b) (h_x0 : 0 < x0) :
    ∃ i ≥ 1, ¬ prime (nat.iterate (λ x, a * x + b) (i-1) x0) :=
sorry

end exists_not_prime_in_sequence_l713_713749


namespace bushels_needed_l713_713537

theorem bushels_needed (cows sheep chickens : ℕ) (cows_eat sheep_eat chickens_eat : ℕ) :
  cows = 4 → cows_eat = 2 →
  sheep = 3 → sheep_eat = 2 →
  chickens = 7 → chickens_eat = 3 →
  4 * 2 + 3 * 2 + 7 * 3 = 35 := 
by
  intros hc hec hs hes hch hech
  sorry

end bushels_needed_l713_713537


namespace circumcircle_radius_l713_713372

variables (a b : ℝ)

-- Defining the isosceles trapezoid with given conditions
def is_isosceles_trapezoid (a b : ℝ) := 
  ∃ A B C D : ℝ,
  ∃ θ : ℝ,
  θ = 30 ∧ 
  A = 0 ∧ 
  D = 1 ∧ 
  B = a ∧ 
  (C - D)/2 = a / 2 ∧ 
  (C - A)/2 = b/2 ∧
  θ = 30

-- Theorem to be proven: radius of the circumcircle equals sqrt(b^2 + (a^2 / 4))
theorem circumcircle_radius (h : is_isosceles_trapezoid a b) :
  ∃ R : ℝ, R = sqrt(b^2 + a^2 / 4) :=
sorry

end circumcircle_radius_l713_713372


namespace find_radius_of_circle_l713_713079

noncomputable def radius_of_circle (DE EB EC : ℝ) (O E C : Point) (r : ℝ) : Prop :=
DE = 3 ∧ EB = 5 ∧ EC = 1 ∧ (O = center) ∧ (E : Point) ∧ (C : Point) ∧ 
(power_of_point_theorem DE EB OE EC (r - 1) 1)

theorem find_radius_of_circle (O E C : Point) (r : ℝ) (DE EB EC : ℝ) :
  DE = 3 → 
  EB = 5 → 
  EC = 1 → 
  r = 16 :=
begin
  -- Here the proof steps should be included, but we are skipping with a placeholder.
  sorry
end

end find_radius_of_circle_l713_713079


namespace cos_120_eq_neg_half_l713_713941

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713941


namespace sufficient_but_not_necessary_condition_l713_713194

variable (k : ℤ)

theorem sufficient_but_not_necessary_condition (α : ℝ) :
  (∃ k : ℤ, α = 2 * k * π - π / 4) →
  (cos α =  √2 / 2) ∧ (¬∀ k : ℤ, α = 2 * k * π ± π / 4) :=
by
  sorry

end sufficient_but_not_necessary_condition_l713_713194


namespace crackers_shared_equally_l713_713729

theorem crackers_shared_equally : ∀ (matthew_crackers friends_crackers left_crackers friends : ℕ),
  matthew_crackers = 23 →
  left_crackers = 11 →
  friends = 2 →
  matthew_crackers - left_crackers = friends_crackers →
  friends_crackers = friends * 6 :=
by
  intro matthew_crackers friends_crackers left_crackers friends
  sorry

end crackers_shared_equally_l713_713729


namespace total_cost_alex_had_to_pay_l713_713470

def baseCost : ℝ := 30
def costPerText : ℝ := 0.04 -- 4 cents in dollars
def textsSent : ℕ := 150
def costPerMinuteOverLimit : ℝ := 0.15 -- 15 cents in dollars
def hoursUsed : ℝ := 26
def freeHours : ℝ := 25

def totalCost : ℝ :=
  baseCost + (costPerText * textsSent) + (costPerMinuteOverLimit * (hoursUsed - freeHours) * 60)

theorem total_cost_alex_had_to_pay :
  totalCost = 45 := by
  sorry

end total_cost_alex_had_to_pay_l713_713470


namespace M_inter_N_eq_l713_713084

def M := {-1, 0, 1}
def N := {x : ℝ | x * x ≤ x}
def M_inter_N := M ∩ N

theorem M_inter_N_eq : M_inter_N = {0, 1} :=
by
sory

end M_inter_N_eq_l713_713084


namespace regular_polygon_sides_and_angles_l713_713891

theorem regular_polygon_sides_and_angles
  (perimeter : ℕ)
  (side_length : ℕ)
  (H1 : perimeter = 180)
  (H2 : side_length = 15) :
  ∃ n : ℕ, n = perimeter / side_length ∧ (n = 12 ∧ (n - 2) * 180 / n = 150) :=
by
  use 12
  split
  . rw [H1, H2]
    norm_num
  . split
    . rfl
    . norm_num
      done

end regular_polygon_sides_and_angles_l713_713891


namespace larger_number_is_1634_l713_713771

theorem larger_number_is_1634 (L S : ℤ) (h1 : L - S = 1365) (h2 : L = 6 * S + 20) : L = 1634 := 
sorry

end larger_number_is_1634_l713_713771


namespace least_multiple_25_gt_500_l713_713823

theorem least_multiple_25_gt_500 : ∃ (k : ℕ), 25 * k > 500 ∧ (∀ m : ℕ, (25 * m > 500 → 25 * k ≤ 25 * m)) :=
by
  use 21
  sorry

end least_multiple_25_gt_500_l713_713823


namespace smallest_possible_sum_of_squares_l713_713009

theorem smallest_possible_sum_of_squares : 
  ∃ (x y : ℕ), (x^2 - y^2 = 221) ∧ (∀(a b : ℕ), (a^2 - b^2 = 221) → (x^2 + y^2 ≤ a^2 + b^2)) ∧ (x^2 + y^2 = 229) :=
begin
  sorry, -- Proof omitted as per instructions
end

end smallest_possible_sum_of_squares_l713_713009


namespace f_2002_value_l713_713302

noncomputable def f : ℕ → ℝ := sorry

axiom f_condition :
  ∀ n > 1, ∃ p : ℕ, nat.prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

axiom f_2001 : f 2001 = 1

theorem f_2002_value : f 2002 = 2 :=
  by sorry

end f_2002_value_l713_713302


namespace percentage_reduction_is_approx_13_l713_713895

noncomputable def original_employees : ℝ := 302.3
noncomputable def reduced_employees : ℝ := 263.0

def percent_reduction (O R : ℝ) : ℝ :=
  ((O - R) / O) * 100

theorem percentage_reduction_is_approx_13 :
  percent_reduction original_employees reduced_employees ≈ 13 := by
  sorry

end percentage_reduction_is_approx_13_l713_713895


namespace rhombus_projection_rectangle_l713_713530

/-- A rhombus has a point inside, projections on sides, and specific intersecting perpendicular bisectors.
    Prove the points P, S, T, and O form a rectangle. -/
theorem rhombus_projection_rectangle (ABCD : rhombus)
  (O : Point) (P : Point) (not_on_diagonals : ¬P ∈ (diagonals ABCD))
  (M N Q R : Point)
  (P_on_ABCD_sides : projections P ABCD M N Q R)
  (S : Point) (T : Point)
  (S_intersection : S ∈ (perpendicular_bisector (M, N)) ∧ S ∈ (perpendicular_bisector (R, Q)))
  (T_intersection : T ∈ (perpendicular_bisector (N, Q)) ∧ T ∈ (perpendicular_bisector (M, R))) : 
  is_rectangle P S T O :=
sorry

end rhombus_projection_rectangle_l713_713530


namespace triangle_options_correct_l713_713671

theorem triangle_options_correct 
  (A B C a b c : ℝ)
  (h_triangle : ∀ x, 0 < x ∧ x < π / 2)
  (h_side_lengths : a > 0 ∧ b > 0 ∧ c > 0)
  (h_cosC : cos C = b / (2 * a) - 1 / 2)
  (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  : (b > a) ∧ (C = 2 * A) ∧ (tan C > sqrt 3) :=
sorry

end triangle_options_correct_l713_713671


namespace is_divisible_by_six_l713_713848

/-- A stingy knight keeps gold coins in six chests. Given that he can evenly distribute the coins by opening any
two chests, any three chests, any four chests, or any five chests, prove that the total number of coins can be 
evenly distributed among all six chests. -/
theorem is_divisible_by_six (n : ℕ) 
  (h2 : ∀ (a b : ℕ), a + b = n → (a % 2 = 0 ∧ b % 2 = 0))
  (h3 : ∀ (a b c : ℕ), a + b + c = n → (a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0)) 
  (h4 : ∀ (a b c d : ℕ), a + b + c + d = n → (a % 4 = 0 ∧ b % 4 = 0 ∧ c % 4 = 0 ∧ d % 4 = 0))
  (h5 : ∀ (a b c d e : ℕ), a + b + c + d + e = n → (a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 ∧ e % 5 = 0)) :
  n % 6 = 0 :=
sorry

end is_divisible_by_six_l713_713848


namespace arithmetic_sum_l713_713625

theorem arithmetic_sum (n : ℕ) :  
  let x : ℕ → ℚ := λ k, 2 + (k - 1) * (1 / 3)
  in (finset.range n).sum x = n * (n + 11) / 6 :=
by
  sorry

end arithmetic_sum_l713_713625


namespace find_multiplier_l713_713899

theorem find_multiplier (x : ℤ) : 
  30 * x - 138 = 102 ↔ x = 8 := 
by
  sorry

end find_multiplier_l713_713899


namespace production_value_decreased_by_10_percent_l713_713473

variable (a : ℝ)

def production_value_in_January : ℝ := a

def production_value_in_February (a : ℝ) : ℝ := 0.9 * a

theorem production_value_decreased_by_10_percent (a : ℝ) :
  production_value_in_February a = 0.9 * production_value_in_January a := 
by
  sorry

end production_value_decreased_by_10_percent_l713_713473


namespace john_finish_work_alone_in_48_days_l713_713278

variable {J R : ℝ}

theorem john_finish_work_alone_in_48_days
  (h1 : J + R = 1 / 24)
  (h2 : 16 * (J + R) = 2 / 3)
  (h3 : 16 * J = 1 / 3) :
  1 / J = 48 := 
by
  sorry

end john_finish_work_alone_in_48_days_l713_713278


namespace lateral_surface_area_of_cone_l713_713181

-- Definitions based on given conditions:
variable (S A B : Point)
variable (cos_ASB : ℝ)
variable (angle_SA_base : ℝ)
variable (area_SAB : ℝ)

-- Conditions provided in the problem:
def cone_conditions : Prop :=
  cos_ASB = 7 / 8 ∧
  angle_SA_base = π / 4 ∧
  area_SAB = 5 * Real.sqrt 15

-- The main statement to prove:
theorem lateral_surface_area_of_cone :
  cone_conditions S A B cos_ASB angle_SA_base area_SAB →
  lateral_surface_area S A B = 40 * Real.sqrt 2 * π :=
sorry

end lateral_surface_area_of_cone_l713_713181


namespace smallest_distance_l713_713713

open Complex

variable (z w : ℂ)

def a : ℂ := -2 - 4 * I
def b : ℂ := 5 + 6 * I

-- Conditions
def cond1 : Prop := abs (z + 2 + 4 * I) = 2
def cond2 : Prop := abs (w - 5 - 6 * I) = 4

-- Problem
theorem smallest_distance (h1 : cond1 z) (h2 : cond2 w) : abs (z - w) = Real.sqrt 149 - 6 :=
sorry

end smallest_distance_l713_713713


namespace find_BM_length_l713_713263

variables (A M C : Type) (MA CA BC BM : ℝ) (x d h : ℝ)

-- Conditions
def condition1 : Prop := MA + (BC - BM) = 2 * CA
def condition2 : Prop := MA = x
def condition3 : Prop := CA = d
def condition4 : Prop := BC = h

-- The proof problem statement
theorem find_BM_length (A M C : Type) (MA CA BC BM : ℝ) (x d h : ℝ)
  (h1 : condition1 MA CA BC BM)
  (h2 : condition2 MA x)
  (h3 : condition3 CA d)
  (h4 : condition4 BC h) :
  BM = 2 * d :=
sorry

end find_BM_length_l713_713263


namespace grey_eyed_black_haired_students_l713_713656

theorem grey_eyed_black_haired_students (total_students black_haired green_eyed_red_haired grey_eyed : ℕ) 
(h_total : total_students = 60) 
(h_black_haired : black_haired = 35) 
(h_green_eyed_red_haired : green_eyed_red_haired = 20) 
(h_grey_eyed : grey_eyed = 25) : 
grey_eyed - (total_students - black_haired - green_eyed_red_haired) = 20 :=
by
  sorry

end grey_eyed_black_haired_students_l713_713656


namespace complement_of_A_in_U_l713_713857

-- Define the universal set U
def U : Set ℕ := {2, 3, 4}

-- Define set A
def A : Set ℕ := {2, 3}

-- Define the complement of A in U
def C_U_A : Set ℕ := U \ A

-- Prove the complement of A in U is {4}
theorem complement_of_A_in_U : C_U_A = {4} := 
  by 
  sorry

end complement_of_A_in_U_l713_713857


namespace estimated_probability_of_white_ball_l713_713120

noncomputable def frequencies : List ℝ := [0.580, 0.640, 0.580, 0.596, 0.590, 0.604, 0.601]

noncomputable def mean (lst : List ℝ) : ℝ :=
  (lst.sum / lst.length)

-- The main theorem to prove that the estimated probability is approximately 0.60
theorem estimated_probability_of_white_ball : abs (mean frequencies - 0.60) < 0.01 :=
by
  sorry

end estimated_probability_of_white_ball_l713_713120


namespace count_integers_of_form_ab40_divisible_by_40_l713_713227

theorem count_integers_of_form_ab40_divisible_by_40 :
  let count := (λ (n : ℕ), n >= 1000 ∧ n < 10000 ∧ n % 100 = 40 ∧ n % 40 = 0) in
  (Finset.filter count (Finset.range 10000)).card = 11 := 
by
  sorry

end count_integers_of_form_ab40_divisible_by_40_l713_713227


namespace evaluate_expression_l713_713554

/-- 
Define the floor function and variable x
-/
def floor (n : ℝ) : ℤ := Int.floor n
def x : ℝ := 13.2

/-- 
Proof problem: Prove that the given expression evaluates to 5
-/
theorem evaluate_expression : 
  floor (x^2) - floor x * floor x = 5 := by
  sorry

end evaluate_expression_l713_713554


namespace acres_for_corn_l713_713844

theorem acres_for_corn (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ)
  (total_ratio : beans_ratio + wheat_ratio + corn_ratio = 11)
  (land_parts : total_land / 11 = 94)
  : (corn_ratio = 4) → (total_land = 1034) → 4 * 94 = 376 :=
by
  intros
  sorry

end acres_for_corn_l713_713844


namespace karlson_wins_with_optimal_play_l713_713509

def game_win_optimal_play: Prop :=
  ∀ (total_moves: ℕ), 
  (total_moves % 2 = 1) 

theorem karlson_wins_with_optimal_play: game_win_optimal_play :=
by sorry

end karlson_wins_with_optimal_play_l713_713509


namespace n_in_S_implies_n_squared_in_S_l713_713717

-- Definition of the set S
def S : Set ℕ := {n | ∃ a b c d e f : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ 
                      n - 1 = a^2 + b^2 ∧ n = c^2 + d^2 ∧ n + 1 = e^2 + f^2}

-- The proof goal
theorem n_in_S_implies_n_squared_in_S (n : ℕ) (h : n ∈ S) : n^2 ∈ S :=
by
  sorry

end n_in_S_implies_n_squared_in_S_l713_713717


namespace complementary_angles_not_obtuse_l713_713387

-- Define the concept of complementary angles.
def is_complementary (a b : ℝ) : Prop :=
  a + b = 90

-- Define that neither angle should be obtuse.
def not_obtuse (a b : ℝ) : Prop :=
  a < 90 ∧ b < 90

-- Proof problem statement
theorem complementary_angles_not_obtuse (a b : ℝ) (ha : a < 90) (hb : b < 90) (h_comp : is_complementary a b) : 
  not_obtuse a b :=
by
  sorry

end complementary_angles_not_obtuse_l713_713387


namespace factor_1000000000001_l713_713047

theorem factor_1000000000001 : ∃ a b c : ℕ, 1000000000001 = a * b * c ∧ a = 73 ∧ b = 137 ∧ c = 99990001 :=
by {
  sorry
}

end factor_1000000000001_l713_713047


namespace value_of_a_l713_713601

variable (x y a : ℝ)

-- Conditions
def condition1 : Prop := (x = 1)
def condition2 : Prop := (y = 2)
def condition3 : Prop := (3 * x - a * y = 1)

-- Theorem stating the equivalence between the conditions and the value of 'a'
theorem value_of_a (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 x y a) : a = 1 :=
by
  -- Insert proof here
  sorry

end value_of_a_l713_713601


namespace total_cats_and_kittens_received_l713_713695

open Real

section
variable (adult_cats : ℕ)
variable (female_cats_fraction female_cats_with_kittens_fraction average_kittens_per_litter : ℝ)

theorem total_cats_and_kittens_received (adult_cats_eq : adult_cats = 120)
    (female_cats_fraction_eq : female_cats_fraction = 1 / 3)
    (female_cats_calculated : ∀ (x : ℕ), x = adult_cats * female_cats_fraction → x = 40)
    (female_cats_with_kittens_fraction_eq : female_cats_with_kittens_fraction = 0.4)
    (average_kittens_per_litter_eq : average_kittens_per_litter = 5)
    : (adult_cats + (40 * 0.4 * 5)) = 200 :=
by
    intro
    simp [adult_cats_eq, female_cats_fraction_eq, female_cats_calculated, female_cats_with_kittens_fraction_eq, average_kittens_per_litter_eq]
    sorry
end

end total_cats_and_kittens_received_l713_713695


namespace not_factorization_method_l713_713836

theorem not_factorization_method {A B C D : Type} 
  (taking_out_common_factor : A)
  (cross_multiplication_method : B)
  (formula_method : C)
  (addition_subtraction_elimination_method : D) :
  ¬(D) := 
sorry

end not_factorization_method_l713_713836


namespace ratio_of_diagonal_to_perimeter_l713_713096

theorem ratio_of_diagonal_to_perimeter (x y : ℝ) (h1 : x = y / 3) : 
  (sqrt (x^2 + y^2)) / (2 * x + 2 * y) = sqrt 10 / 8 :=
by
  -- The proof will go here
  sorry

end ratio_of_diagonal_to_perimeter_l713_713096


namespace distance_between_first_and_last_pots_l713_713548

theorem distance_between_first_and_last_pots (n : ℕ) (d : ℕ) 
  (h₁ : n = 8) 
  (h₂ : d = 100) : 
  ∃ total_distance : ℕ, total_distance = 175 := 
by 
  sorry

end distance_between_first_and_last_pots_l713_713548


namespace problem_statement_l713_713197

def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
def g (x : ℝ) : ℝ := 2 * x + 1

theorem problem_statement : f (g 5) - g (f 5) = 63 :=
by
  sorry

end problem_statement_l713_713197


namespace new_stationary_points_relation_l713_713791

noncomputable def g (x : ℝ) := x^3 - 1
noncomputable def h (x : ℝ) := 2 * x
noncomputable def φ (x : ℝ) := Real.log (x + 1)

noncomputable def α : ℝ :=
  Classical.choose (Exists.intro (1 : ℝ) 
    (by {
      let x := 1 in
      have hx : g x = g' x := rfl,
      exact hx
    }))

noncomputable def β : ℝ :=
  Classical.choose (Exists.intro (1 : ℝ)
    (by {
      let x := 1 in
      have hx : h x = h' x := rfl,
      exact hx
    }))

noncomputable def γ : ℝ :=
  Classical.choose (Exists.intro (0 : ℝ)
    (by {
      let x := 0 in
      have hx : φ x = φ' x := rfl,
      exact hx
    }))

theorem new_stationary_points_relation :
  α > β ∧ β > γ :=
sorry

end new_stationary_points_relation_l713_713791


namespace find_n_such_that_l713_713154

theorem find_n_such_that :
  ∃ n : ℕ, 2^6 * 3^3 * 5^1 * n = fact 10 ∧ n = 420 :=
by
  use 420
  split
  · sorry
  · rfl

end find_n_such_that_l713_713154


namespace symmetrical_point_A_wrt_xOz_l713_713265

def Point := ℝ × ℝ × ℝ
def symmetrical_wrt_xOz (p: Point) : Point := (p.1, -p.2, p.3)

theorem symmetrical_point_A_wrt_xOz :
  symmetrical_wrt_xOz (-3, 2, -4) = (-3, -2, -4) :=
by
  sorry

end symmetrical_point_A_wrt_xOz_l713_713265


namespace f_at_3_l713_713709

noncomputable def f : ℝ → ℝ := sorry

lemma periodic (f : ℝ → ℝ) : ∀ x : ℝ, f (x + 4) = f x := sorry

lemma odd_function (f : ℝ → ℝ) : ∀ x : ℝ, f (-x) + f x = 0 := sorry

lemma given_interval (f : ℝ → ℝ) : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = (x - 1)^2 := sorry

theorem f_at_3 : f 3 = 0 := 
by
  sorry

end f_at_3_l713_713709


namespace complex_division_l713_713766

theorem complex_division : ( -1 + 3 * complex.i ) / ( 1 + complex.i ) = 1 + 2 * complex.i := 
by {
    sorry
}

end complex_division_l713_713766


namespace cosine_120_eq_negative_half_l713_713959

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l713_713959


namespace f_2023_l713_713095

noncomputable def f : ℕ → ℤ
| 1       := 2
| 2       := 3
| (n + 3) := f (n + 2) - f (n + 1) + 2 * (n + 3)

theorem f_2023 : f 2023 = 4046 :=
by {
  -- This is where the proof would go, using the recursive definition of f.
  sorry
}

end f_2023_l713_713095


namespace find_z_when_y_is_6_l713_713350

variable {y z : ℚ}

/-- Condition: y^4 varies inversely with √[4]{z}. -/
def inverse_variation (k : ℚ) (y z : ℚ) : Prop :=
  y^4 * z^(1/4) = k

/-- Given constant k based on y = 3 and z = 16. -/
def k_value : ℚ := 162

theorem find_z_when_y_is_6
  (h_inv : inverse_variation k_value 3 16)
  (h_y : y = 6) :
  z = 1 / 4096 := 
sorry

end find_z_when_y_is_6_l713_713350


namespace interest_rate_correct_l713_713427

theorem interest_rate_correct :
  let SI := 155
  let P := 810
  let T := 4
  let R := SI * 100 / (P * T)
  R = 155 * 100 / (810 * 4) := 
sorry

end interest_rate_correct_l713_713427


namespace area_enclosed_by_abs_graph_l713_713053

theorem area_enclosed_by_abs_graph : 
  (∃ (x y : ℝ), abs (2 * x) + abs (3 * y) = 12) →
  ∑ (x y : ℝ), abs (2 * x) + abs (3* y) = 48 := by
  sorry

end area_enclosed_by_abs_graph_l713_713053


namespace num_valid_distributions_for_top_square_multiple_of_4_l713_713110

theorem num_valid_distributions_for_top_square_multiple_of_4 
  (x : Fin 12 -> Fin 2) 
  (k : ℕ) 
  (h : 1 ≤ k ∧ k ≤ 12)
  : 
  ( ∑ i in fin_range 12, nat.choose 11 i * x i.val ) % 4 = 0 
  -> 16 * 2^9 = 8192 :=
by
  sorry

end num_valid_distributions_for_top_square_multiple_of_4_l713_713110


namespace value_of_a_l713_713599

theorem value_of_a (x y a : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : 3 * x - a * y = 1) : a = 1 := by
  sorry

end value_of_a_l713_713599


namespace equivalence_of_statements_l713_713493

variable (P Q : Prop)

theorem equivalence_of_statements (h : P → Q) :
  (P → Q) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q) := by
  sorry

end equivalence_of_statements_l713_713493


namespace equal_area_iff_K_on_MN_l713_713742

-- Definitions based on problem
variables {A B C D M N E F K : Point}
variables (parallelogram : Parallelogram A B C D)
variables (M_on_BC : Point_on_segment M B C)
variables (N_on_CD : Point_on_segment N C D)
variables (E_on_AM : Point_on_line E (line_through A M))
variables (F_on_AN : Point_on_line F (line_through A N))
variables (intersect_E : intersect (line_through B D) (line_through A M) E)
variables (intersect_F : intersect (line_through B D) (line_through A N) F)
variables (EK_parallel_AD : parallel (line_through E K) (line_through A D))
variables (FK_parallel_AB : parallel (line_through F K) (line_through A B))

-- Goals to prove: equality of areas and location of point K on segment MN
theorem equal_area_iff_K_on_MN : 
  (area (triangle A E B) + area (triangle A F D) = area (triangle M E N) + area (triangle M F N)) ↔ 
  Point_on_segment K M N := by
    sorry

end equal_area_iff_K_on_MN_l713_713742


namespace tangent_line_to_curve_at_pi_l713_713566

noncomputable def tangent_line_equation (f : ℝ → ℝ) (x₀ y₀ : ℝ) (f' : ℝ → ℝ) (m : ℝ) (b : ℝ) : Prop :=
  ∀ x, f x₀ = y₀ → f' x₀ = m → f x = m * (x - x₀) + b

theorem tangent_line_to_curve_at_pi :
  tangent_line_equation
    (λ x => x * sin x)     -- the function f(x) = x * sin(x)
    π                      -- x₀ = π
    0                      -- y₀ = 0
    (λ x => sin x + x * cos x)  -- the derivative f'(x) = sin(x) + x * cos(x)
    (-π)                   -- the slope m = -π
    (π^2)                  -- the y-intercept b = π^2
:= sorry

end tangent_line_to_curve_at_pi_l713_713566


namespace jezebel_total_cost_l713_713273

theorem jezebel_total_cost :
  let red_rose_count := 2 * 12,
      sunflower_count := 3,
      red_rose_cost := 1.50,
      sunflower_cost := 3,
      total_cost := (red_rose_count * red_rose_cost) + (sunflower_count * sunflower_cost)
  in
  total_cost = 45 := 
by
  sorry

end jezebel_total_cost_l713_713273


namespace parallelogram_altitude_length_l713_713258

theorem parallelogram_altitude_length
  (A B C D E F : Type) [parallelogram ABCD]
  (de_alt : is_altitude DE.to AB)
  (df_alt : is_altitude DF.to BC)
  (DC_length : DC = 15)
  (EB_length : EB = 5)
  (DE_length : DE = 7.5) :
  DF = 5 :=
sorry

end parallelogram_altitude_length_l713_713258


namespace modulus_of_z_plus_1_l713_713928

def z : ℂ := (1 - 3 * complex.I) / (1 + complex.I)

theorem modulus_of_z_plus_1 : complex.abs (z + 1) = 2 :=
by
  sorry

end modulus_of_z_plus_1_l713_713928


namespace spinner_prime_probability_l713_713134

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def spinner_sections : List ℕ := [2, 3, 5, 7, 4, 6, 8, 9]

def no_of_prime_numbers : ℕ := (spinner_sections.filter is_prime).length

def probability_prime : ℚ := (no_of_prime_numbers : ℚ) / (spinner_sections.length : ℚ)

theorem spinner_prime_probability : probability_prime = 1 / 2 := by
  rw [probability_prime, no_of_prime_numbers]
  simp only [spinner_sections, List.filter_eq_self.mpr, List.length]
  norm_num
  sorry

end spinner_prime_probability_l713_713134


namespace find_number_l713_713865

-- Define the given condition
def number_div_property (num : ℝ) : Prop :=
  num / 0.3 = 7.3500000000000005

-- State the theorem to prove
theorem find_number (num : ℝ) (h : number_div_property num) : num = 2.205 :=
by sorry

end find_number_l713_713865


namespace greatest_gcd_6Pn_n_plus_1_l713_713170

open Nat

def pentagonal (n : ℕ) : ℕ := n * (3 * n - 1) / 2

theorem greatest_gcd_6Pn_n_plus_1 (n : ℕ) (hn : n > 0) :
  gcd (6 * (pentagonal n)) (n + 1) = 3 :=
sorry

end greatest_gcd_6Pn_n_plus_1_l713_713170


namespace washing_time_l713_713499

theorem washing_time (shirts pants sweaters jeans : ℕ) (items_per_cycle minutes_per_cycle : ℕ) :
  shirts = 18 → pants = 12 → sweaters = 17 → jeans = 13 → 
  items_per_cycle = 15 → minutes_per_cycle = 45 → 
  (shirts + pants + sweaters + jeans) / items_per_cycle * minutes_per_cycle / 60 = 3 :=
by
  intros h_shirts h_pants h_sweaters h_jeans h_cycle_max h_cycle_time
  calc
    (shirts + pants + sweaters + jeans) / items_per_cycle * minutes_per_cycle / 60
    = (18 + 12 + 17 + 13) / 15 * 45 / 60 : by rw [h_shirts, h_pants, h_sweaters, h_jeans, h_cycle_max, h_cycle_time]
    ... = 60 / 15 * 45 / 60 : by norm_num
    ... = 4 * 45 / 60 : by norm_num
    ... = 180 / 60 : by norm_num
    ... = 3 : by norm_num

end washing_time_l713_713499


namespace first_nonzero_digit_one_div_157_l713_713423

theorem first_nonzero_digit_one_div_157 : 
  ∃ d : ℕ, d = 6 ∧ (∃ n : ℕ, n ≥ 1 ∧ n * d = (1000 * 1 / 157).floor) := 
by
  sorry

end first_nonzero_digit_one_div_157_l713_713423


namespace largest_postal_code_l713_713926

def phone_number_digits : List ℕ := [3, 4, 6, 2, 7, 8, 9]

def sum_digits (l : List ℕ) : ℕ := l.foldl (· + ·) 0

def is_distinct (l : List ℕ) := (l.nodup)

def is_five_digit (n : ℕ) := 10000 ≤ n ∧ n < 100000

theorem largest_postal_code :
  ∃ (postal_code : ℕ), is_five_digit postal_code ∧ is_distinct (postal_code.digits 10) ∧ sum_digits (postal_code.digits 10) = 39 ∧ postal_code = 98765 :=
by
  sorry

end largest_postal_code_l713_713926


namespace ellipse_equation_parabola_equation_area_triangle_range_ratio_value_const_l713_713852

noncomputable def point := ℝ × ℝ

-- Given conditions
def F₁ : point := (x₁, y₁)  -- Define coordinates for F₁ based on given problem context
def F₂ : point := (1, 0)
def O : point := (0, 0)
def A : point := (3/2, sqrt 6)

-- Conditions for curve C₁ (Ellipse) and C₂ (Parabola)
def isEllipse (p : point) : Prop := (p.1 ^ 2) / 9 + (p.2 ^ 2) / 8 = 1
def isParabola (p : point) : Prop := p.2 ^ 2 = 4 * p.1

-- Intersection requirement
axiom A_intersection : isEllipse A ∧ isParabola A

-- Definitions based on given problem
def line_through (p: point) (m: ℝ) : (ℝ → ℝ) := λ y, m * y + fst p   -- Equation of line through point with slope m
def intersects (curve: point → Prop) (line: ℝ → ℝ) : Prop := ∃ y, curve (line y, y)

-- Midpoints G and H
def midpoint (p₁ p₂ : point) : point := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)

-- Conditions translated
def CDF1_area_condition (m : ℝ) : Prop := ∃ c1 c2 y1 y2, 
    c1 ≠ c2 ∧ 
    isEllipse c1 ∧ isEllipse c2 ∧ 
    intersects isEllipse (line_through F₂ m) ∧ 
    distance y1 y2 = sqrt (m ^ 2 + 1) ∧
    4 ≤ 1/2 * abs ( fst c1 * (y2 - y1) )  -- Calculated area of triangle

def ratio_constant : Prop :=
  ∃ B C D E G H m, 
    intersects isEllipse (line_through F₂ m) ∧ intersects isParabola (line_through F₂ m) ∧
    midpoint C D = G ∧ midpoint B E = H ∧ 
    (abs (distance B E) * abs (distance G F₂)) / (abs (distance C D) * abs (distance H F₂)) = 3

-- Proof statements
theorem ellipse_equation : isEllipse A := sorry
theorem parabola_equation : isParabola A := sorry
theorem area_triangle_range (m : ℝ) : CDF1_area_condition m := sorry
theorem ratio_value_const : ratio_constant := sorry

end ellipse_equation_parabola_equation_area_triangle_range_ratio_value_const_l713_713852


namespace find_BP_l713_713036

variable {A B C O P Q : Type*}

variables (a c p BP : ℝ) (h₁ : ∃ triangle ABC, acute_angle ABC)
variables (h₂ : circumcenter O ABC)
variables (h₃ : intersects_perpendicular_and_extension O P Q (BO) (AB c) (BC a) P Q)
variables (h₄ : length BQ = p)

theorem find_BP (h₅ : similar_triangles ABC QBP) : BP = a * p / c := sorry

end find_BP_l713_713036


namespace find_z_l713_713315

noncomputable def z (a b : ℝ) := a + b * complex.I

theorem find_z (a b : ℝ) (h₁ : a^2 + b^2 = 1) (h₂ : 3 * a - 4 * b = 0) : 
  z a b = (4/5 : ℝ) - (3/5 : ℝ) * complex.I ∨ z a b = -(4/5 : ℝ) + (3/5 : ℝ) * complex.I :=
by {
  sorry
}

end find_z_l713_713315


namespace alpha_beta_range_l713_713506

theorem alpha_beta_range (α β : ℝ) (P : ℝ × ℝ)
  (h1 : α > 0) 
  (h2 : β > 0) 
  (h3 : P = (α, 3 * β))
  (circle_eq : (α - 1)^2 + 9 * (β^2) = 1) :
  1 < α + β ∧ α + β < 5 / 3 :=
sorry

end alpha_beta_range_l713_713506


namespace builders_count_l713_713397

theorem builders_count (daily_wage : ℕ) (days_per_floor : ℕ) (num_builders_per_floor : ℕ) (num_houses : ℕ) (floors_per_house : ℕ) (total_cost : ℕ) :
  daily_wage = 100 → days_per_floor = 30 → num_builders_per_floor = 3 → num_houses = 5 → floors_per_house = 6 → total_cost = 270000 →
  (total_cost / (num_builders_per_floor * daily_wage * days_per_floor)) = 30 :=
begin
  sorry
end

end builders_count_l713_713397


namespace profit_equation_l713_713359

theorem profit_equation (x : ℝ) :
  let cost_price := 40
  let initial_selling_price := 60
  let initial_quantity := 200
  let quantity_increase_per_unit_reduction := 8
  let total_profit_target := 8450
  let new_selling_price := initial_selling_price - x
  let new_quantity_sold := initial_quantity + quantity_increase_per_unit_reduction * x
  let profit_per_unit := new_selling_price - cost_price
  (profit_per_unit * new_quantity_sold = total_profit_target) :=
begin
  sorry
end

end profit_equation_l713_713359


namespace misha_erase_half_proper_divisors_l713_713731

open Nat

theorem misha_erase_half_proper_divisors (a b : ℕ) 
  (ha : 2 ≤ a) (hb : 2 ≤ b) 
  (h_composite_a : ¬ prime a) (h_composite_b : ¬ prime b) : 
  ∃ Da Db : Finset ℕ, 
    Da ⊆ (Finset.filter (λ x, 1 < x ∧ x < a ∧ a % x = 0) (Finset.range (a + 1))) ∧
    Db ⊆ (Finset.filter (λ x, 1 < x ∧ x < b ∧ b % x = 0) (Finset.range (b + 1))) ∧
    (Da.card ≤ (Finset.filter (λ x, 1 < x ∧ x < a ∧ a % x = 0) (Finset.range (a + 1))).card / 2) ∧ 
    (Db.card ≤ (Finset.filter (λ x, 1 < x ∧ x < b ∧ b % x = 0) (Finset.range (b + 1))).card / 2) ∧
    (∀ x ∈ Da, ∀ y ∈ Db, ¬ (a + b) % (x + y) = 0) :=
sorry

end misha_erase_half_proper_divisors_l713_713731


namespace fill_venn_diagram_l713_713917

namespace VennCircles

variables (A₁ A₂ A₃ A₄ A₅ A₆ A₇ A₈ A₉ : ℕ)

def prefilled_A₂ : A₂ = 5 := rfl
def prefilled_A₈ : A₈ = 9 := rfl

def available_numbers := [2, 3, 4, 6, 7, 8, 10]

def sums_to_15 (circle : List ℕ) : Prop :=
  circle.sum = 15

theorem fill_venn_diagram :
  ∃ (A₁ A₃ A₄ A₅ A₆ A₇ A₉ : ℕ), 
    A₂ = 5 ∧
    A₈ = 9 ∧
    multiset.of_list [A₁, A₃, A₄, A₅, A₆, A₇, A₉] = multiset.of_list available_numbers ∧
    sums_to_15 [A₁, A₂, A₃] ∧
    sums_to_15 [A₄, A₅, A₆] ∧
    sums_to_15 [A₇, A₈, A₉] ∧
    -- Add similar conditions for the additional circles
    sorry :=
begin
  use [10, 7, 3, 8, 4, 2, 6],
  split, exact prefilled_A₂,
  split, exact prefilled_A₈,
  split,
  {
    -- check the numbers match the available numbers
    rfl,
  },
  split,
  {
    -- Circle sum condition: Circle 1
    have h : [10, 5, 7].sum = 15, by norm_num,
    exact h,
  },
  split,
  {
    -- Circle sum condition: Circle 2
    have h : [3, 8, 4].sum = 15, by norm_num,
    exact h,
  },
  split,
  {
    -- Circle sum condition: Circle 3
    have h : [2, 9, 6].sum = 15, by norm_num,
    exact h,
  },
  sorry, -- Further formulations for other circles
end

end VennCircles

end fill_venn_diagram_l713_713917


namespace square_dance_partners_l713_713399

theorem square_dance_partners (m1 f1 m2 f2 m3 f3 : ℕ) 
                              (h1 : m1 = 17) (h2 : f1 = 13) 
                              (h3 : m2 = 14) (h4 : f2 = 18) 
                              (h5 : m3 = 15) (h6 : f3 = 17) : 
                              (m1 + m2 + m3 < f1 + f2 + f3) 
                              → (f1 + f2 + f3 - (m1 + m2 + m3) = 2) :=
by {
  rw [h1, h2, h3, h4, h5, h6],
  norm_num,
  intro h,
  exact h,
}

end square_dance_partners_l713_713399


namespace tiling_possible_if_and_only_if_one_dimension_is_integer_l713_713819

-- Define our conditions: a, b are dimensions of the board and t is the positive dimension of the small rectangles
variable (a b : ℝ) (t : ℝ)

-- Define corresponding properties for these variables
axiom pos_t : t > 0

-- Theorem stating the condition for tiling
theorem tiling_possible_if_and_only_if_one_dimension_is_integer (a_non_int : ¬ ∃ z : ℤ, a = z) (b_non_int : ¬ ∃ z : ℤ, b = z) :
  ∃ n m : ℕ, n * 1 + m * t = a * b :=
sorry

end tiling_possible_if_and_only_if_one_dimension_is_integer_l713_713819


namespace tan_alpha_plus_pi_over_4_l713_713603

noncomputable def tan_sum_formula (a b : Real) : Real :=
  (Math.tan a + Math.tan b) / (1 - Math.tan a * Math.tan b)

theorem tan_alpha_plus_pi_over_4 (α : Real) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = Real.sqrt 65 / 65) :
  tan_sum_formula α (π / 4) = -9 / 7 :=
sorry

end tan_alpha_plus_pi_over_4_l713_713603


namespace katya_minimum_problems_l713_713283

-- Defining the conditions
def katya_probability_solve : ℚ := 4 / 5
def pen_probability_solve : ℚ := 1 / 2
def total_problems : ℕ := 20
def minimum_correct_for_good_grade : ℚ := 13

-- The Lean statement to prove
theorem katya_minimum_problems (x : ℕ) :
  x * katya_probability_solve + (total_problems - x) * pen_probability_solve ≥ minimum_correct_for_good_grade → x ≥ 10 :=
sorry

end katya_minimum_problems_l713_713283


namespace g3_values_l713_713322

theorem g3_values (g : ℝ → ℝ) (h : ∀ x y, g(x) * g(y) - g(x * y) = x^2 + y^2) :
  (let m := (λ s, 1) (SetOf (λ z, g 3 = z)).card in
   let t := (λ s, ∑' z in SetOf (λ z, g 3 = z), z) 1 Set.empty in
   m * t = 10) :=
by
  sorry

end g3_values_l713_713322


namespace twelve_factorial_div_eleven_factorial_eq_twelve_l713_713525

theorem twelve_factorial_div_eleven_factorial_eq_twelve :
  12! / 11! = 12 :=
by
  sorry

end twelve_factorial_div_eleven_factorial_eq_twelve_l713_713525


namespace trapezoid_area_148_l713_713016

-- Definitions for conditions
variables {a b m : ℝ}

-- Condition 1: height of the trapezoid
def height := 2

-- Condition 2: The difference in the perimeters of the quadrilaterals
def diff_perimeters := (λ a b m : ℝ, (b + m + height + height) - (a + m + height + height)) = 24

-- Condition 3: The ratio of their areas
def area_ratio := (λ a b m : ℝ, (3 * b + a) / (3 * a + b)) = 20 / 17

-- Condition: Midsegment definition
def midsegment := (λ a b : ℝ, m = (a + b) / 2)

-- Question: Calculate the area of the trapezoid
def area_trapezoid (a b : ℝ) := (a + b) / 2 * height

-- The theorem to prove
theorem trapezoid_area_148 (a b m : ℝ):
  height = 2 ∧
  diff_perimeters a b m ∧
  area_ratio a b m ∧
  midsegment a b →
  area_trapezoid a b = 148 :=
sorry

end trapezoid_area_148_l713_713016


namespace angle_RPQ_eq_72_l713_713678

-- Definitions and conditions
variables {P Q R S : Type*}
variable {y : ℝ}

-- Conditions given in the problem
variables (is_on_RS : ∃ (P : P), true)  -- P is on line RS
variables (bisection : ∃ (QP : ℝ), QP = (angle R Q P) / 2)  -- QP bisects angle SQR
variables (PQ_eq_PR : PQ = PR)  -- PQ = PR
variables (angle_RSQ : angle R S Q = 3 * y)  -- ∠RSQ = 3y°
variables (angle_RPQ : angle R P Q = 2 * y)  -- ∠RPQ = 2y°

-- The theorem to be proved
theorem angle_RPQ_eq_72 (h1 : is_on_RS) (h2 : bisection y) (h3 : PQ_eq_PR) (h4 : angle_RSQ y = 3 * y) (h5 : angle_RPQ y = 2 * y) : 
  angle R P Q = 72 :=
begin
  sorry
end

end angle_RPQ_eq_72_l713_713678


namespace odds_against_C_l713_713221

-- Definitions based on the conditions
def odds_against (p : ℚ) (q : ℚ) : ℝ := p/q

def A_prob : ℚ := 2 / (5 + 2)
def B_prob : ℚ := 4 / (7 + 4)
def C_prob : ℚ := 1 - (2 / 7 + 4 / 11)

theorem odds_against_C : odds_against (1 - C_prob) C_prob = 50 / 27 := by
  sorry

end odds_against_C_l713_713221


namespace katya_minimum_problems_l713_713285

-- Defining the conditions
def katya_probability_solve : ℚ := 4 / 5
def pen_probability_solve : ℚ := 1 / 2
def total_problems : ℕ := 20
def minimum_correct_for_good_grade : ℚ := 13

-- The Lean statement to prove
theorem katya_minimum_problems (x : ℕ) :
  x * katya_probability_solve + (total_problems - x) * pen_probability_solve ≥ minimum_correct_for_good_grade → x ≥ 10 :=
sorry

end katya_minimum_problems_l713_713285


namespace reflection_matrix_condition_l713_713147

noncomputable def reflection_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, b], ![-(3/4 : ℝ), 1/4]]

noncomputable def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], ![0, 1]]

theorem reflection_matrix_condition (a b : ℝ) :
  (reflection_matrix a b)^2 = identity_matrix ↔ a = -(1/4) ∧ b = -(3/4) :=
  by
  sorry

end reflection_matrix_condition_l713_713147


namespace dropped_student_score_l713_713354

theorem dropped_student_score (total_students : ℕ) (remaining_students : ℕ) (initial_average : ℝ) (new_average : ℝ) (x : ℝ) 
  (h1 : total_students = 16) 
  (h2 : remaining_students = 15) 
  (h3 : initial_average = 62.5) 
  (h4 : new_average = 63.0) 
  (h5 : total_students * initial_average - remaining_students * new_average = x) : 
  x = 55 := 
sorry

end dropped_student_score_l713_713354


namespace intersections_sum_l713_713130

theorem intersections_sum (A B : ℕ) (C D : ℝ) (r : ℝ) (s : ℝ) : 
  let start := (0, 0)
  let end := (A, B)
  A = 1001 →
  B = 429 →
  r = 1 / 10 →
  s = 1 / 5 →
  let line := λ x : ℝ, (C / D) * x
  C = 3 →
  D = 7 →
  (count_intersections line ((0, 0), (A, B)) r s) = 862 := 
by
  sorry

end intersections_sum_l713_713130


namespace trains_crossing_time_l713_713814

noncomputable def time_to_cross_each_other (length1 length2 speed1_kmh speed2_kmh : ℕ) : ℝ :=
  let speed1_ms := speed1_kmh * 5 / 18
  let speed2_ms := speed2_kmh * 5 / 18
  let relative_speed := speed1_ms + speed2_ms
  let total_distance := (length1 + length2) : ℕ
  total_distance / relative_speed

theorem trains_crossing_time : 
  time_to_cross_each_other 500 750 80 65 ≈ 31.03 :=
sorry

end trains_crossing_time_l713_713814


namespace square_dance_partners_l713_713400

theorem square_dance_partners (m1 f1 m2 f2 m3 f3 : ℕ) 
                              (h1 : m1 = 17) (h2 : f1 = 13) 
                              (h3 : m2 = 14) (h4 : f2 = 18) 
                              (h5 : m3 = 15) (h6 : f3 = 17) : 
                              (m1 + m2 + m3 < f1 + f2 + f3) 
                              → (f1 + f2 + f3 - (m1 + m2 + m3) = 2) :=
by {
  rw [h1, h2, h3, h4, h5, h6],
  norm_num,
  intro h,
  exact h,
}

end square_dance_partners_l713_713400


namespace total_trees_after_planting_l713_713800

def initial_trees : ℕ := 39
def trees_planted_today : ℕ := 41
def trees_planted_tomorrow : ℕ := 20

theorem total_trees_after_planting :
  initial_trees + trees_planted_today + trees_planted_tomorrow = 100 := 
by sorry

end total_trees_after_planting_l713_713800


namespace periodic_function_is_a_of_value_l713_713319

theorem periodic_function_is_a_of_value (a : ℝ) (m : ℝ) (h_m : m ≠ 0) :
  (t_1 t_2 : ℝ) 
  (h_t1_t2 : t_1 ≠ t_2)
  (h_roots : t_1^2 - (5 * a - 2) * t_1 - 3 * a^2 - 7 * a + 1 = 0
    ∧ t_2^2 - (5 * a - 2) * t_2 - 3 * a^2 - 7 * a + 1 = 0) :
  (a = 2 / 5) :=
by {
  sorry
}

end periodic_function_is_a_of_value_l713_713319


namespace opposite_of_number_is_5_l713_713019

-- Define the condition
def opposite (x : ℝ) : ℝ := -x

-- Construct the statement to prove
theorem opposite_of_number_is_5 : ∃ x : ℝ, opposite x = 5 ∧ x = -5 :=
by
  use -5
  split
  · exact rfl
  · exact rfl

end opposite_of_number_is_5_l713_713019


namespace smallest_sum_214_l713_713783

theorem smallest_sum_214 :
  ∃ (a b c : Fin 9 → ℕ) (perm : ∀ i : Fin 3, a i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
                                         b i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
                                         c i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) 
            (distinct : List.nodup (a 0 :: a 1 :: a 2 :: b 0 :: b 1 :: b 2 :: c 0 :: c 1 :: c 2 :: [])) 
            (a_has_even : ∃ i : Fin 3, a i % 2 = 0) (a_has_odd : ∃ i : Fin 3, a i % 2 = 1)
            (b_has_even : ∃ i : Fin 3, b i % 2 = 0) (b_has_odd : ∃ i : Fin 3, b i % 2 = 1)
            (c_has_even : ∃ i : Fin 3, c i % 2 = 0) (c_has_odd : ∃ i : Fin 3, c i % 2 = 1), 
  a 0 * a 1 * a 2 + b 0 * b 1 * b 2 + c 0 * c 1 * c 2 = 214 :=
by
  sorry

end smallest_sum_214_l713_713783


namespace quadratic_polynomial_example_l713_713570

theorem quadratic_polynomial_example :
  ∃ (P : ℝ[X]), P.coeff 2 = 3 ∧ P.coeff 1 = -6 ∧ P.coeff 0 = -15 ∧
  (Polynomial.eval (-1 - 2 * Complex.I) P = 0) :=
by
  let P := Polynomial.C (-15) + Polynomial.X * Polynomial.C (-6) + Polynomial.X^2 * Polynomial.C 3
  use P
  split
  { -- Verifying the coefficient of x^2
    show P.coeff 2 = 3, sorry },
  split
  { -- Verifying the coefficient of x
    show P.coeff 1 = -6, sorry },
  split
  { -- Verifying the constant term
    show P.coeff 0 = -15, sorry },
  { -- Verifying the root condition
    show Polynomial.eval (-1 - 2 * Complex.I) P = 0, sorry }

end quadratic_polynomial_example_l713_713570


namespace find_required_water_amount_l713_713540

-- Definitions based on the conditions
def sanitizer_volume : ℝ := 12
def initial_alcohol_concentration : ℝ := 0.60
def desired_alcohol_concentration : ℝ := 0.40

-- Statement of the proof problem
theorem find_required_water_amount : 
  ∃ (x : ℝ), x = 6 ∧ sanitizer_volume * initial_alcohol_concentration = desired_alcohol_concentration * (sanitizer_volume + x) :=
sorry

end find_required_water_amount_l713_713540


namespace simplify_fraction_l713_713070

variable (d : ℤ)

theorem simplify_fraction (d : ℤ) : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := 
by 
  sorry

end simplify_fraction_l713_713070


namespace count_four_digit_integers_l713_713226

theorem count_four_digit_integers (n : ℕ) (h : 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 40 ∧ n % 40 = 0) :
  (finset.filter (λ n, 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 40 ∧ n % 40 = 0) (finset.range 10000)).card = 5 :=
sorry

end count_four_digit_integers_l713_713226


namespace Carol_rectangle_length_l713_713513

theorem Carol_rectangle_length :
  (∃ (L : ℕ), (L * 15 = 4 * 30) → L = 8) :=
by
  sorry

end Carol_rectangle_length_l713_713513


namespace rectangle_length_l713_713446

theorem rectangle_length (P W : ℝ) (hP : P = 40) (hW : W = 8) : ∃ L : ℝ, 2 * (L + W) = P ∧ L = 12 := 
by 
  sorry

end rectangle_length_l713_713446


namespace segment_distance_sum_l713_713262

theorem segment_distance_sum
  (AB_len : ℝ) (A'B'_len : ℝ) (D_midpoint : AB_len / 2 = 4)
  (D'_midpoint : A'B'_len / 2 = 6) (x : ℝ) (y : ℝ)
  (x_val : x = 3) :
  x + y = 10 :=
by sorry

end segment_distance_sum_l713_713262


namespace length_bd_divided_segment_l713_713098

noncomputable def prove_length_bd (CB : ℝ) (AC : ℝ) (BD : ℝ) : Prop :=
  ∃ (tangent : ℝ),
    AC = 4 * CB ∧
    ∃ (circle_ac_radius circle_cb_radius : ℝ),
      circle_ac_radius = AC / 2 ∧
      circle_cb_radius = CB / 2 ∧
      tangent = circle_cb_radius ∧
      BD = tangent

theorem length_bd_divided_segment (CB x : ℝ) (hCB : CB = x) :
  prove_length_bd CB (4 * CB) (CB / 2) :=
by {
  use (CB / 2),
  split,
  { rw hCB, ring, },
  use [(2 * CB), (CB / 2)],
  split,
  { field_simp, ring, },
  split,
  { field_simp, },
  split,
  { field_simp, },
  { refl, },
}

end length_bd_divided_segment_l713_713098


namespace myFunction_isQuadratic_l713_713651

-- Define what it means for a function to be quadratic
def isQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ f = λ x, a * x^2 + b * x + c

-- Define the specific function in question
def myFunction (a : ℝ) : ℝ → ℝ :=
  λ x, (a + 1) * x^2 - 2 * x + 3

-- Statement: Prove that the function is quadratic if and only if a ≠ -1
theorem myFunction_isQuadratic (a : ℝ) : isQuadratic (myFunction a) ↔ a ≠ -1 :=
by
  sorry

end myFunction_isQuadratic_l713_713651


namespace problem_1_problem_2_l713_713234

-- Definition for Problem 1
def log_cond (S : ℝ) : Prop := real.logb 9 S = 3 / 2

-- Problem 1: Prove S = 27
theorem problem_1 (S : ℝ) (h : log_cond S) : S = 27 := sorry

-- Definition for Problem 2
def line1 (x y : ℝ) : Prop := x + 5 * y = 0
def line2 (T S x y : ℝ) : Prop := T * x - S * y = 0
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Slope calculation
def slope1 : ℝ := -1 / 5
def slope2 (T S : ℝ) : ℝ := T / S

-- Problem 2: Prove T = 135
theorem problem_2 (T S : ℝ) (hS : S = 27)
    (h1 : ∀ x y, line1 x y)
    (h2 : ∀ x y, line2 T S x y)
    (h_perp : perpendicular slope1 (slope2 T S)) : T = 135 := sorry

end problem_1_problem_2_l713_713234


namespace remainder_when_divided_by_x_minus_3_l713_713571

noncomputable def polynomial_f : ℤ[X] := X^4 - 4 * X^2 + 7 * X - 1

theorem remainder_when_divided_by_x_minus_3 : polynomial.eval 3 polynomial_f = 65 :=
by {
  sorry -- proof will be written here
}

end remainder_when_divided_by_x_minus_3_l713_713571


namespace socks_distribution_l713_713441

structure SockProblem where
  total_socks : ℕ := 9
  at_least_two_in_four : ∀ {socks : Finset ℕ}, socks.card = 4 → ∃ a b, a ∈ socks ∧ b ∈ socks ∧ a ≠ b
  at_most_three_in_five : ∀ {socks : Finset ℕ}, socks.card = 5 → ∀ s, s ∈ socks → socks.filter (λ x, x = s).card ≤ 3

theorem socks_distribution {k s : ℕ} (h : SockProblem) : k = 3 ∧ s = 3 :=
sorry

end socks_distribution_l713_713441


namespace unique_solution_values_l713_713217

theorem unique_solution_values (a : ℝ) :
  (∃! x : ℝ, a * x^2 - x + 1 = 0) ↔ (a = 0 ∨ a = 1 / 4) :=
by
  sorry

end unique_solution_values_l713_713217


namespace gas_equilibrium_l713_713459

variable (v₁ v₂ : ℝ)
variable (T₁ T₂ : ℝ)

-- Conditions
def condition1 : Prop := v₁ + v₂ = 4
def condition2 : Prop := v₂ - v₁ = 0.62
def T1_value : Prop := T₁ = 373

-- Define the quantities to be proven
def v1_value : Prop := v₁ = 1.69
def v2_value : Prop := v₂ = 2.31
def T2_value : Prop := T₂ = 272.9

-- Final theorem combining all conditions and results
theorem gas_equilibrium
  (h1 : condition1)
  (h2 : condition2)
  (hT1 : T1_value)
  : v1_value ∧ v2_value ∧ T2_value :=
by {
  sorry
}

end gas_equilibrium_l713_713459


namespace knight_probability_l713_713808

theorem knight_probability :
  let Q := 1 - ((binom 16 3) / (binom 20 4)) in
  let simplified_Q := (66 : ℚ) / 75 in
  let sum_nd := 66 + 75 in
  Q = simplified_Q ∧ sum_nd = 141 :=
by
  sorry

end knight_probability_l713_713808


namespace sum_of_areas_squares_l713_713803

theorem sum_of_areas_squares (a : ℕ) (h1 : (a + 4)^2 - a^2 = 80) : a^2 + (a + 4)^2 = 208 := by
  sorry

end sum_of_areas_squares_l713_713803


namespace combined_length_of_legs_is_ten_l713_713379

-- Define the conditions given in the problem.
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = a * Real.sqrt 2

def hypotenuse_length (c : ℝ) : Prop :=
  c = 7.0710678118654755

def perimeter_condition (a b c perimeter : ℝ) : Prop :=
  perimeter = a + b + c ∧ perimeter = 10 + c

-- Prove the combined length of the two legs is 10.
theorem combined_length_of_legs_is_ten :
  ∃ (a b c : ℝ), is_isosceles_right_triangle a b c →
  hypotenuse_length c →
  ∀ perimeter : ℝ, perimeter_condition a b c perimeter →
  2 * a = 10 :=
by
  sorry

end combined_length_of_legs_is_ten_l713_713379


namespace sum_of_triangle_perimeters_l713_713505

noncomputable def geometric_series_sum (a r : ℝ) (hr : |r| < 1) : ℝ :=
  a / (1 - r)

theorem sum_of_triangle_perimeters (a : ℝ) (ha : 0 < a) : 
  let first_perimeter := 3 * a,
      common_ratio := 1 / 2 in
  geometric_series_sum first_perimeter common_ratio (by norm_num : |common_ratio| < 1) = 6 * a := 
by
  sorry

end sum_of_triangle_perimeters_l713_713505


namespace find_g3_value_l713_713325

def g (n : ℕ) : ℕ :=
  if n < 5 then 2 * n ^ 2 + 3 else 4 * n + 1

theorem find_g3_value : g (g (g 3)) = 341 := by
  sorry

end find_g3_value_l713_713325


namespace calculate_expression_l713_713320

theorem calculate_expression : 
  let x := -2023
  in abs (abs (abs x - x) + abs x) - x = 8092 := 
by
  sorry

end calculate_expression_l713_713320


namespace parabola_chord_length_eight_l713_713878

noncomputable def parabola_intersection_length (x1 x2: ℝ) (y1 y2: ℝ) : ℝ :=
  if x1 + x2 = 6 ∧ y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 then
    let A := (x1, y1)
    let B := (x2, y2)
    dist A B
  else
    0

theorem parabola_chord_length_eight :
  ∀ (x1 x2 y1 y2 : ℝ), (x1 + x2 = 6) → (y1^2 = 4 * x1) → (y2^2 = 4 * x2) →
  parabola_intersection_length x1 x2 y1 y2 = 8 :=
by
  -- proof goes here
  sorry

end parabola_chord_length_eight_l713_713878


namespace range_of_A_l713_713618

noncomputable def f : ℝ → ℝ
| x := if x ≤ 1 then -x^2 + 2 * x - 5 / 4 else log (1 / 3) - 1 / 4

noncomputable def g (A : ℝ) (x : ℝ) : ℝ :=
  abs (A - 2) * sin x

theorem range_of_A (A : ℝ) :
  (∀ x1 x2 : ℝ, f x1 ≤ g A x2) ↔ (7 / 4 ≤ A ∧ A ≤ 9 / 4) :=
begin
  -- proof omitted
  sorry
end

end range_of_A_l713_713618


namespace probability_divisible_by_11_l713_713242

theorem probability_divisible_by_11 :
  let five_digit_numbers : Finset (Fin 100000) := 
        {n | n < 100000 ∧ n ≥ 10000 ∧ (List.sum n.digits 10 = 44)}
  let divisible_by_11 : Finset (Fin 100000) := 
        {n | n ∈ five_digit_numbers ∧ n % 11 = 0}
  (divisible_by_11.card : ℚ) / (five_digit_numbers.card : ℚ) = 2 / 15 :=
by { sorry }

end probability_divisible_by_11_l713_713242


namespace cos_120_eq_neg_half_l713_713985

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l713_713985


namespace pyramids_to_cube_l713_713802

noncomputable def side_length : ℝ := 1 / Real.sqrt 2

theorem pyramids_to_cube (tetrahedron quadrangular_pyramid : Set ℝ^3) 
  (edge_length : ℝ) (h_edge : edge_length = 1) 
  (h_tetra : (∀ v ∈ tetrahedron, (∃ x y z: ℝ, v = (x, y, z) ∧ (x^2 + y^2 + z^2 = edge_length^2)))
    ∧ (∀ (u v ∈ tetrahedron), (u = v) ∨ (u - v).norm = edge_length))
  (h_quad : (∀ v ∈ quadrangular_pyramid, (∃ x y z: ℝ, v = (x, y, z) ∧ (x^2 + y^2 + z^2 = edge_length^2)))
    ∧ (∀ (u v ∈ quadrangular_pyramid), (u = v) ∨ (u - v).norm = edge_length)) :
  ∃ cube : Set ℝ^3, 
    (∀ v ∈ cube, (∃ x y z: ℝ, v = (x, y, z) ∧ (x^2 + y^2 + z^2 = side_length^2)))
    ∧ (∀ (u v ∈ cube), (u = v) ∨ (u - v).norm = side_length) :=
sorry

end pyramids_to_cube_l713_713802


namespace f_transform_l713_713646

noncomputable def f (x : ℝ) : ℝ :=
log ((1 + x) / (1 - x))

theorem f_transform (x : ℝ) (h : -1 < x ∧ x < 1) :
  f ( (5 * x + x^5) / (1 + 5 * x^4) ) = 5 * f x :=
sorry

end f_transform_l713_713646


namespace sum_of_odd_numbers_l713_713165

theorem sum_of_odd_numbers (n : ℕ) : ∑ k in finset.range n, (2 * (k + 1) - 1) = n^2 :=
sorry

end sum_of_odd_numbers_l713_713165


namespace time_to_fill_drum_l713_713685

variable (rain_rate : ℝ) -- Rain rate in cm/hour
variable (depth : ℝ) -- Depth in cm
variable (base_area : ℝ) -- Base area in cm²

theorem time_to_fill_drum (h1 : rain_rate = 5) (h2 : depth = 15) (h3 : base_area = 300) :
  depth * base_area / (rain_rate * base_area) = 3 :=
by
  calc
    depth * base_area / (rain_rate * base_area) = (15 * 300) / (5 * 300) : by rw [h1, h2, h3]
                                              ... = 4500 / 1500 : by norm_num
                                              ... = 3 : by norm_num

end time_to_fill_drum_l713_713685


namespace find_k_l713_713156

-- Auxiliary function to calculate the product of the digits of a number
def productOfDigits (n : ℕ) : ℕ :=
  (n.digits 10).foldl (λ acc d => acc * d) 1

theorem find_k (k : ℕ) (h1 : 0 < k) (h2 : productOfDigits k = (25 * k) / 8 - 211) : 
  k = 72 ∨ k = 88 :=
by
  sorry

end find_k_l713_713156


namespace pennies_for_washing_clothes_l713_713756

theorem pennies_for_washing_clothes (total_money_cents : ℕ) (num_quarters : ℕ) (value_quarter_cents : ℕ) :
  total_money_cents = 184 → num_quarters = 7 → value_quarter_cents = 25 → (total_money_cents - num_quarters * value_quarter_cents) = 9 :=
by
  intros htm hq hvq
  rw [htm, hq, hvq]
  linarith

end pennies_for_washing_clothes_l713_713756


namespace katya_needs_at_least_ten_l713_713290

variable (x : ℕ)

def prob_katya : ℚ := 4 / 5
def prob_pen  : ℚ := 1 / 2

def expected_correct (x : ℕ) : ℚ := x * prob_katya + (20 - x) * prob_pen

theorem katya_needs_at_least_ten :
  expected_correct x ≥ 13 → x ≥ 10 :=
  by sorry

end katya_needs_at_least_ten_l713_713290


namespace cannot_form_figureB_l713_713777

-- Define the pieces as terms
inductive Piece
| square : Piece
| rectangle : Π (h w : ℕ), Piece   -- h: height, w: width

-- Define the available pieces in a list (assuming these are predefined somewhere)
def pieces : List Piece := [Piece.rectangle 2 1, Piece.rectangle 1 2, Piece.square, Piece.square, Piece.square, 
                            Piece.square, Piece.square, Piece.square, Piece.square, Piece.square]

-- Define the figures that can be formed
def figureA : List Piece := [Piece.square, Piece.square, Piece.square, Piece.square, Piece.square, 
                            Piece.square, Piece.square, Piece.square, Piece.square, Piece.square]

def figureC : List Piece := [Piece.rectangle 2 1, Piece.rectangle 1 2, Piece.square, Piece.square, 
                             Piece.square, Piece.square]

def figureD : List Piece := [Piece.rectangle 2 2, Piece.square, Piece.square, Piece.square,
                              Piece.square]

def figureE : List Piece := [Piece.rectangle 3 1, Piece.square, Piece.square, Piece.square]

-- Define the figure B that we need to prove cannot be formed
def figureB : List Piece := [Piece.rectangle 5 1, Piece.square, Piece.square, Piece.square,
                              Piece.square]

theorem cannot_form_figureB :
  ¬(∃ arrangement : List Piece, arrangement ⊆ pieces ∧ arrangement = figureB) :=
sorry

end cannot_form_figureB_l713_713777


namespace percent_decrease_in_square_area_l713_713679

-- Definitions based on the problem conditions
def area_triangle_I := 50 * Real.sqrt 3
def area_triangle_III := 18 * Real.sqrt 3
def area_square_II := 50
def side_length_AD_decreased := 0.8 * Real.sqrt (2 * area_square_II)

-- Statement to prove the question equals the correct answer
theorem percent_decrease_in_square_area 
  (h1 : area_triangle_I = 50 * Real.sqrt 3)
  (h2 : area_triangle_III = 18 * Real.sqrt 3)
  (h3 : area_square_II = 50)
  (decrease_AD : side_length_AD_decreased = 0.8 * Real.sqrt (2 * area_square_II)) : 
  ((area_square_II - (side_length_AD_decreased ^ 2)) / area_square_II) * 100 = 36 := 
by
  sorry

end percent_decrease_in_square_area_l713_713679


namespace coin_order_is_correct_l713_713392

def Coin := {F, B, C, D, E, A}

-- Conditions as definitions
def covers (x y : Coin) : Prop := 
  (x = 'F' ∧ y ∈ { 'B', 'C', 'D', 'E', 'A' }) ∨
  (x = 'B' ∧ y ∈ { 'C', 'D', 'E' }) ∨
  (x = 'C' ∧ y ∈ { 'A', 'E' }) ∨
  (x = 'D' ∧ y = 'A') ∨
  (x = 'E' ∧ y = 'A')

-- Define the ordering of coins
def coin_order : list Coin := ['F', 'B', 'C', 'D', 'E', 'A']

noncomputable def correct_order : Prop :=
  (∀ (x y : Coin), x ∈ coin_order ∧ y ∈ coin_order → 
    covers x y → list.index_of x coin_order < list.index_of y coin_order)

theorem coin_order_is_correct : correct_order := sorry

end coin_order_is_correct_l713_713392


namespace fish_swim_eastward_l713_713859

-- Define the conditions
variables (E : ℕ)
variable (total_fish_left : ℕ := 2870)
variable (fish_westward : ℕ := 1800)
variable (fish_north : ℕ := 500)
variable (fishwestward_not_caught : ℕ := fish_westward / 4)
variable (fishnorth_not_caught : ℕ := fish_north)
variable (fish_tobe_left_after_caught : ℕ := total_fish_left - fishwestward_not_caught - fishnorth_not_caught)

-- Define the theorem to prove
theorem fish_swim_eastward (h : 3 / 5 * E = fish_tobe_left_after_caught) : E = 3200 := 
by
  sorry

end fish_swim_eastward_l713_713859


namespace first_day_price_l713_713494

theorem first_day_price (x n: ℝ) :
  n * x = (n + 100) * (x - 1) ∧ 
  n * x = (n - 200) * (x + 2) → 
  x = 4 :=
by
  sorry

end first_day_price_l713_713494


namespace problem_statement_l713_713428

-- Define what it means for a number's tens and ones digits to have a sum of 13
def sum_of_tens_and_ones_equals (n : ℕ) (s : ℕ) : Prop :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit = s

-- State the theorem with the given conditions and correct answer
theorem problem_statement : sum_of_tens_and_ones_equals (6^11) 13 :=
sorry

end problem_statement_l713_713428


namespace no_real_solutions_l713_713144

theorem no_real_solutions (x : ℝ) (h_nonzero : x ≠ 0) (h_pos : 0 < x):
  (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 12 * x^9 :=
by
-- Proof will go here.
sorry

end no_real_solutions_l713_713144


namespace sum_of_quotient_and_remainder_is_184_l713_713432

theorem sum_of_quotient_and_remainder_is_184 
  (q r : ℕ)
  (h1 : 23 * 17 + 19 = q)
  (h2 : q * 10 = r)
  (h3 : r / 23 = 178)
  (h4 : r % 23 = 6) :
  178 + 6 = 184 :=
by
  -- Inform Lean that we are skipping the proof
  sorry

end sum_of_quotient_and_remainder_is_184_l713_713432


namespace triangle_circle_intersection_points_l713_713198

noncomputable def possibleCommonPoints (triangle_sides : Set ℝ) (circle_radius : ℝ) : Set ℕ :=
  { pts | ∃ (T : Triangle) (C : Circle), T.sides = triangle_sides ∧ C.radius = circle_radius ∧ pts ∈ {0, 1, 2, 4} }

theorem triangle_circle_intersection_points :
  possibleCommonPoints {3, 4, 5} 1 = {0, 1, 2, 4} :=
by
  sorry

end triangle_circle_intersection_points_l713_713198


namespace derivative_log2_l713_713616

noncomputable def f (x : ℝ) : ℝ := log x / log 2

theorem derivative_log2 (x : ℝ) (hx : 0 < x) : deriv f x = 1 / (x * log 2) :=
by
  -- Skip the actual proof
  sorry

end derivative_log2_l713_713616


namespace sum_of_sequence_l713_713794

def sequence (n : ℕ) : ℚ := n + 1/(2^n)

def sum_sequence (n : ℕ) : ℚ := ∑ i in Finset.range n, sequence (i + 1)

theorem sum_of_sequence (n : ℕ) :
  sum_sequence n = (n * (n + 1) / 2) + 1 - 1/(2^n) :=
sorry

end sum_of_sequence_l713_713794


namespace rods_in_one_mile_l713_713196

theorem rods_in_one_mile (mile_to_furlong : ℕ) (furlong_to_rod : ℕ) (mile_eq : 1 = 8 * mile_to_furlong) (furlong_eq: 1 = 50 * furlong_to_rod) : 
  (1 * 8 * 50 = 400) :=
by
  sorry

end rods_in_one_mile_l713_713196


namespace find_number_l713_713346

theorem find_number (x : ℝ) (h : (x - 5) / 3 = 4) : x = 17 :=
by {
  sorry
}

end find_number_l713_713346


namespace find_change_l713_713281

def initial_amount : ℝ := 1.80
def cost_of_candy_bar : ℝ := 0.45
def change : ℝ := 1.35

theorem find_change : initial_amount - cost_of_candy_bar = change :=
by sorry

end find_change_l713_713281


namespace ratio_of_arithmetic_sums_l713_713929

theorem ratio_of_arithmetic_sums : 
  let a1 := 4
  let d1 := 4
  let l1 := 48
  let a2 := 2
  let d2 := 3
  let l2 := 35
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let S1 := n1 * (a1 + l1) / 2
  let S2 := n2 * (a2 + l2) / 2
  let ratio := S1 / S2
  ratio = 52 / 37 := by sorry

end ratio_of_arithmetic_sums_l713_713929


namespace sum_series_l713_713333

noncomputable def a (i : ℕ) : ℝ := 1 / (Real.sqrt i + Real.sqrt (i + 1))

theorem sum_series (n : ℕ) :
  (∑ i in Finset.range n, a (i + 1)) = Real.sqrt (n + 1) - 1 := by
sorry

end sum_series_l713_713333


namespace cos_120_eq_neg_half_l713_713966

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l713_713966


namespace sin_add_pi_over_3_and_cos_beta_l713_713682

theorem sin_add_pi_over_3_and_cos_beta (α β : ℝ)
  (h1 : sin (α + β) = 5 / 13)
  (h2 : cos α = -3 / 5)
  (h3 : sin α = -4 / 5) : 
  sin (α + π / 3) = - (4 + 3 * sqrt 3) / 10 ∧ 
  (cos β = -56 / 65 ∨ cos β = 16 / 65) := 
by 
  sorry

end sin_add_pi_over_3_and_cos_beta_l713_713682


namespace min_radius_circumscribed_sphere_l713_713264

theorem min_radius_circumscribed_sphere (x y : ℝ) (h₁ : ∠ BAC = 90) (h₂ : 4 * x * y = 16) :
    ∃ r, r = 2 * sqrt 2 := by
  sorry

end min_radius_circumscribed_sphere_l713_713264


namespace tim_younger_than_jenny_l713_713404

variable (Tim_age Rommel_age Jenny_age Uncle_age Aunt_age : ℝ)
variable (condition1 : Tim_age = 5)
variable (condition2 : Rommel_age = 3 * Tim_age)
variable (condition3 : Jenny_age = Rommel_age + 2)
variable (condition4 : Uncle_age = Real.sqrt (Rommel_age * Jenny_age))
variable (condition5 : Aunt_age = Real.sqrt (Uncle_age * Jenny_age))

theorem tim_younger_than_jenny : (Jenny_age - Tim_age) = 12 :=
by
  unfold Tim_age Rommel_age Jenny_age Uncle_age Aunt_age
  rw [condition1, condition2, condition3, condition4, condition5]
  sorry

end tim_younger_than_jenny_l713_713404


namespace roll_four_fair_dice_l713_713057
noncomputable def roll_four_fair_dice_prob : ℚ :=
  let total_outcomes : ℚ := 6^4
  let favorable_outcomes : ℚ := 6
  let prob_all_same : ℚ := favorable_outcomes / total_outcomes
  let prob_not_all_same : ℚ := 1 - prob_all_same
  prob_not_all_same

theorem roll_four_fair_dice :
  roll_four_fair_dice_prob = 215 / 216 :=
by
  sorry

end roll_four_fair_dice_l713_713057


namespace cos_120_eq_neg_half_l713_713976

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713976


namespace composite_quotient_l713_713127

def first_eight_composites := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites := [16, 18, 20, 21, 22, 24, 25, 26]

def product (l : List ℕ) := l.foldl (· * ·) 1

theorem composite_quotient :
  let numerator := product first_eight_composites
  let denominator := product next_eight_composites
  numerator / denominator = (1 : ℚ)/(1430 : ℚ) :=
by
  sorry

end composite_quotient_l713_713127


namespace least_number_to_divisible_by_30_l713_713055

theorem least_number_to_divisible_by_30 : 
  ∃ x : ℕ, (929 + x) % 30 = 0 ∧ ∀ y : ℕ, (929 + y) % 30 = 0 → x ≤ y :=
begin
  sorry
end

end least_number_to_divisible_by_30_l713_713055


namespace jake_wins_probability_l713_713691

noncomputable def probability_ellie_heads : ℝ := 1 / 3
noncomputable def probability_jake_heads : ℝ := 1 / 4

theorem jake_wins_probability (Jake_first: Prop) (Jake_heads_prob: real := 1 / 4) (Ellie_heads_prob: real := 1 / 3) : 
  ∃ (P : ℝ), P = 1 / 2 :=
by
  have Jake_wins_first_turn := 1 / 4
  have both_tails := (3 / 4) * (2 / 3)
  have Jake_wins_follow := both_tails * (1 / 4)
  let series_sum := Jake_wins_first_turn + Jake_wins_follow * (1 / (1 - both_tails))
  use [series_sum]
  sorry

end jake_wins_probability_l713_713691


namespace new_average_weight_l713_713863

def original_players : ℕ := 8
def original_average_weight : ℚ := 105.5
def additional_player1_weight : ℚ := 110.3
def additional_player2_weight : ℚ := 99.7
def additional_player3_weight : ℚ := 103.2
def additional_player4_weight : ℚ := 115.4

theorem new_average_weight :
  let total_weight := original_players * original_average_weight + 
                      additional_player1_weight + 
                      additional_player2_weight + 
                      additional_player3_weight + 
                      additional_player4_weight in
  let total_players := original_players + 4 in
  total_weight / total_players = 106.05 := sorry

end new_average_weight_l713_713863


namespace degree_measure_supplement_complement_l713_713821

noncomputable def supp_degree_complement (α : ℕ) := 180 - (90 - α)

theorem degree_measure_supplement_complement : 
  supp_degree_complement 36 = 126 :=
by sorry

end degree_measure_supplement_complement_l713_713821


namespace count_satisfying_numbers_l713_713224

theorem count_satisfying_numbers :
  let numbers := [-5, -4, -3, -2, -1, 0, 1, 2, 3]
  in card { x | x ∈ numbers ∧ -3 * x^2 < -14 } = 4 :=
by {
  let numbers := [-5, -4, -3, -2, -1, 0, 1, 2, 3],

  have h1: ∀ x, x ∈ numbers → -3 * x^2 < -14 → x^2 > 14 / 3 := sorry,

  sorry
}

end count_satisfying_numbers_l713_713224


namespace value_of_adams_collection_l713_713114

theorem value_of_adams_collection (num_coins : ℕ) (coins_value : ℕ) (total_value_4coins : ℕ) (h1 : num_coins = 20) (h2 : total_value_4coins = 16) (h3 : ∀ k, k = 4 → coins_value = total_value_4coins / k) : 
  num_coins * coins_value = 80 := 
by {
  sorry
}

end value_of_adams_collection_l713_713114


namespace odd_f_even_g_fg_eq_g_increasing_min_g_sum_l713_713212

noncomputable def f (x : ℝ) : ℝ := (0.5) * ((2:ℝ)^x - (2:ℝ)^(-x))
noncomputable def g (x : ℝ) : ℝ := (0.5) * ((2:ℝ)^x + (2:ℝ)^(-x))

theorem odd_f (x : ℝ) : f (-x) = -f (x) := sorry
theorem even_g (x : ℝ) : g (-x) = g (x) := sorry
theorem fg_eq (x : ℝ) : f (x) + g (x) = (2:ℝ)^x := sorry
theorem g_increasing (x : ℝ) : x ≥ 0 → ∀ y, 0 ≤ y ∧ y < x → g y < g x := sorry
theorem min_g_sum (x : ℝ) : ∃ t, t ≥ 2 ∧ (g x + g (2 * x) = 2) := sorry

end odd_f_even_g_fg_eq_g_increasing_min_g_sum_l713_713212


namespace f_neg_one_f_log_two_three_l713_713620

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then 2^x else f (x - 2)

theorem f_neg_one : f (-1) = 1 / 2 :=
by sorry

theorem f_log_two_three : f (Real.log 3 / Real.log 2) = 3 / 4 :=
by sorry

end f_neg_one_f_log_two_three_l713_713620


namespace incorrect_major_premise_l713_713816

theorem incorrect_major_premise (a : ℝ) : ¬ (∀ x : ℝ, x^2 > 0) :=
by {
  use 0,
  simp,
}

end incorrect_major_premise_l713_713816


namespace tangent_lines_between_circles_l713_713780

noncomputable def num_tangent_lines : ℝ :=
  let C1 := (x^2 + y^2 - 6x + 4y + 12 = 0)
  let C2 := (x^2 + y^2 - 14x - 2y + 14 = 0)
  let c1_center := (3, -2)
  let c2_center := (7, 1)
  let r1 := 1
  let r2 := 6
  let dist_centers := ℝ.sqrt ((7 - 3)^2 + (1 + 2)^2)
  if dist_centers = 5 then 3 else sorry

theorem tangent_lines_between_circles :
  let C1 := (x^2 + y^2 - 6x + 4y + 12 = 0)
  let C2 := (x^2 + y^2 - 14x - 2y + 14 = 0)
  num_tangent_lines = 3 :=
begin
  sorry
end

end tangent_lines_between_circles_l713_713780


namespace not_factorization_method_l713_713837

theorem not_factorization_method {A B C D : Type} 
  (taking_out_common_factor : A)
  (cross_multiplication_method : B)
  (formula_method : C)
  (addition_subtraction_elimination_method : D) :
  ¬(D) := 
sorry

end not_factorization_method_l713_713837


namespace job_completion_l713_713864

theorem job_completion (x y z : ℝ) 
  (h1 : 1/x + 1/y = 1/2) 
  (h2 : 1/y + 1/z = 1/4) 
  (h3 : 1/z + 1/x = 1/2.4) 
  (h4 : 1/x + 1/y + 1/z = 7/12) : 
  x = 3 := 
sorry

end job_completion_l713_713864


namespace first_nonzero_digit_of_one_over_157_l713_713418

theorem first_nonzero_digit_of_one_over_157 : 
  (∀ n, 157 * n < 1000 → (n > 0 ∧ (1000 * 1) / (157 * n) > 10^floor(log10( (1000 of 157 * n))) - 10^(1) = 3) := 
by sorry

end first_nonzero_digit_of_one_over_157_l713_713418


namespace count_H_functions_l713_713240

def is_H_function (f : ℝ → ℝ) : Prop :=
∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (x₁ * f x₁ + x₂ * f x₂) ≥ (x₁ * f x₂ + x₂ * f x₁)

noncomputable def f1 (x : ℝ) : ℝ := -x^3 + x + l
noncomputable def f2 (x : ℝ) : ℝ := 3*x - 2*(Real.sin x - Real.cos x)
noncomputable def f3 (x : ℝ) : ℝ := l - e*x
noncomputable def f4 : ℝ → ℝ := λ x, if x < 1 then 0 else Real.log x

def number_of_H_functions : Nat :=
  [f1, f2, f3, f4].filter is_H_function |>.length

theorem count_H_functions : number_of_H_functions = 2 := 
sorry

end count_H_functions_l713_713240


namespace max_min_of_f_find_a_and_theta_l713_713621

noncomputable def f (x θ a : ℝ) : ℝ :=
  Real.sin (x + θ) + a * Real.cos (x + 2 * θ)

theorem max_min_of_f (a θ : ℝ) (h1 : a = Real.sqrt 2) (h2 : θ = π / 4) :
  (∀ x ∈ Set.Icc 0 π, -1 ≤ f x θ a ∧ f x θ a ≤ (Real.sqrt 2) / 2) := sorry

theorem find_a_and_theta (a θ : ℝ) (h1 : f (π / 2) θ a = 0) (h2 : f π θ a = 1) :
  a = -1 ∧ θ = -π / 6 := sorry

end max_min_of_f_find_a_and_theta_l713_713621


namespace anne_distance_diff_l713_713123

def track_length := 300
def min_distance := 100

-- Define distances functions as described
def distance_AB (t : ℝ) : ℝ := sorry  -- Distance function between Anne and Beth over time 
def distance_AC (t : ℝ) : ℝ := sorry  -- Distance function between Anne and Carmen over time 

theorem anne_distance_diff (Anne_speed Beth_speed Carmen_speed : ℝ) 
  (hneA : Anne_speed ≠ Beth_speed)
  (hneC : Anne_speed ≠ Carmen_speed) :
  ∃ α ≥ 0, min_distance ≤ distance_AB α ∧ min_distance ≤ distance_AC α :=
sorry

end anne_distance_diff_l713_713123


namespace felicity_collecting_weeks_l713_713153

-- Define the conditions
def fort_total_sticks : ℕ := 400
def fort_completion_percent : ℝ := 0.60
def store_visits_per_week : ℕ := 3

-- Define the proof problem
theorem felicity_collecting_weeks :
  let collected_sticks := (fort_completion_percent * fort_total_sticks).to_nat
  in collected_sticks / store_visits_per_week = 80 := by
  -- This will be proven in the proof section, currently left as sorry
  sorry

end felicity_collecting_weeks_l713_713153


namespace min_value_f_D_is_2_l713_713117

def f_A (x : ℝ) : ℝ := x + 4 / x
def f_B (x : ℝ) : ℝ := log x + 1 / log x
def f_C (x : ℝ) : ℝ := real.sqrt (x^2 + 1) + 1 / real.sqrt (x^2 + 1)
def f_D (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem min_value_f_D_is_2 (x : ℝ) (hx : x > 0) :
  ∀ x > 0, ∃ y, (f_D x) = y ∧ y = 2 := sorry

end min_value_f_D_is_2_l713_713117


namespace find_flat_fee_l713_713875

def flat_fee_exists (f n : ℝ) : Prop :=
  f + n = 120 ∧ f + 4 * n = 255

theorem find_flat_fee : ∃ f n, flat_fee_exists f n ∧ f = 75 := by
  sorry

end find_flat_fee_l713_713875


namespace prob_both_standard_prob_only_one_standard_l713_713489

-- Given conditions
axiom prob_A1 : ℝ
axiom prob_A2 : ℝ
axiom prob_A1_std : prob_A1 = 0.95
axiom prob_A2_std : prob_A2 = 0.95
axiom prob_not_A1 : ℝ
axiom prob_not_A2 : ℝ
axiom prob_not_A1_std : prob_not_A1 = 0.05
axiom prob_not_A2_std : prob_not_A2 = 0.05
axiom independent_A1_A2 : prob_A1 * prob_A2 = prob_A1 * prob_A2

-- Definitions of events
def event_A1 := true -- Event that the first product is standard
def event_A2 := true -- Event that the second product is standard
def event_not_A1 := not event_A1
def event_not_A2 := not event_A2

-- Proof problems
theorem prob_both_standard :
  prob_A1 * prob_A2 = 0.9025 := by sorry

theorem prob_only_one_standard :
  (prob_A1 * prob_not_A2) + (prob_not_A1 * prob_A2) = 0.095 := by sorry

end prob_both_standard_prob_only_one_standard_l713_713489


namespace first_nonzero_digit_one_div_157_l713_713422

theorem first_nonzero_digit_one_div_157 : 
  ∃ d : ℕ, d = 6 ∧ (∃ n : ℕ, n ≥ 1 ∧ n * d = (1000 * 1 / 157).floor) := 
by
  sorry

end first_nonzero_digit_one_div_157_l713_713422


namespace only_zero_makes_rational_l713_713546

-- Define the given expression
def expr (x : ℝ) : ℝ :=
  x + sqrt (2 * x^2 + 1) - 2 / (x + sqrt (2 * x^2 + 1))

-- Define a predicate to check if the given expression is rational
def is_rational (a : ℝ) : Prop := 
  ∃ (q : ℚ), a = (q : ℝ)

-- The main theorem stating the equivalence of the rational expression and x = 0
theorem only_zero_makes_rational : ∀ (x : ℝ), is_rational (expr x) ↔ x = 0 :=
by
  sorry

end only_zero_makes_rational_l713_713546


namespace horner_eval_operations_l713_713931

noncomputable def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
coeffs.foldr (λ a acc, a + x * acc) 0

theorem horner_eval_operations :
 let coeffs := [1, 8, 7, 6, 5, 4, 3] in
 let x := 0.4 in
 (coeffs.length - 1, coeffs.length - 1) = (6, 6) :=
by
  let coeffs := [1, 8, 7, 6, 5, 4, 3]
  let x := 0.4
  have num_additions : coeffs.length - 1 = 6 := by sorry
  have num_multiplications : coeffs.length - 1 = 6 := by sorry
  exact ⟨num_additions, num_multiplications⟩

end horner_eval_operations_l713_713931


namespace max_smoothie_servings_l713_713888

theorem max_smoothie_servings :
  ∀ (bananas strawberries yogurt bananas_per_serving strawberries_per_serving yogurt_per_serving servings_available),
    bananas = 10 →
    strawberries = 3 →
    yogurt = 12 →
    bananas_per_serving = 3 →
    strawberries_per_serving = 1 →
    yogurt_per_serving = 2 →
    servings_available = 4 →
    min (bananas / bananas_per_serving * servings_available) (min (strawberries / strawberries_per_serving * servings_available) (yogurt / yogurt_per_serving * servings_available)) = 12 :=
by
  intros bananas strawberries yogurt bananas_per_serving strawberries_per_serving yogurt_per_serving servings_available
  assume h_bananas : bananas = 10
  assume h_strawberries : strawberries = 3
  assume h_yogurt : yogurt = 12
  assume h_bananas_per_serving : bananas_per_serving = 3
  assume h_strawberries_per_serving : strawberries_per_serving = 1
  assume h_yogurt_per_serving : yogurt_per_serving = 2
  assume h_servings_available : servings_available = 4
  sorry

end max_smoothie_servings_l713_713888


namespace ax_product_zero_l713_713369

theorem ax_product_zero 
  {a x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ x₁₃ : ℤ} 
  (h1 : a = (1 + x₁) * (1 + x₂) * (1 + x₃) * (1 + x₄) * (1 + x₅) * (1 + x₆) * (1 + x₇) *
           (1 + x₈) * (1 + x₉) * (1 + x₁₀) * (1 + x₁₁) * (1 + x₁₂) * (1 + x₁₃))
  (h2 : a = (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄) * (1 - x₅) * (1 - x₆) * (1 - x₇) *
           (1 - x₈) * (1 - x₉) * (1 - x₁₀) * (1 - x₁₁) * (1 - x₁₂) * (1 - x₁₃)) :
  a * x₁ * x₂ * x₃ * x₄ * x₅ * x₆ * x₇ * x₈ * x₉ * x₁₀ * x₁₁ * x₁₂ * x₁₃ = 0 := 
sorry

end ax_product_zero_l713_713369


namespace first_player_cannot_win_with_odd_piles_l713_713431

theorem first_player_cannot_win_with_odd_piles 
    (n m : ℕ) 
    (is_odd_n : n % 2 = 1) 
    (is_odd_m : m % 2 = 1) 
    : ∃ turns : ℕ, 
      ∀ (removes : ℕ × ℕ) (h : removes.1 ≤ n ∧ removes.2 ≤ m ∧ (removes.1 > 0 ∨ removes.2 > 0)), 
        (turns % 2 = 0 → (n - removes.1 = 0 ∧ m - removes.2 > 0) 
        ∨ (n - removes.1 > 0 ∧ m - removes.2 = 0)) → 
        ((n - removes.1 ≠ m - removes.2) →
        first_player_loses (n - removes.1) (m - removes.2)) :=
sorry

end first_player_cannot_win_with_odd_piles_l713_713431


namespace principal_amount_borrowed_l713_713099

theorem principal_amount_borrowed 
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) 
  (h1 : SI = 9000) 
  (h2 : R = 0.12) 
  (h3 : T = 3) 
  (h4 : SI = P * R * T) : 
  P = 25000 :=
sorry

end principal_amount_borrowed_l713_713099


namespace paul_daily_pencil_production_l713_713744

-- Definitions from the problem conditions.
def startStock : ℕ := 80
def soldPencils : ℕ := 350
def endStock : ℕ := 230
def daysInWeek : ℕ := 5

-- The main statement to be proved.
theorem paul_daily_pencil_production : 
  (let totalPencilsBeforeSelling := endStock + soldPencils in
   let totalPencilsMade := totalPencilsBeforeSelling - startStock in
   totalPencilsMade / daysInWeek = 100) :=
by 
  sorry

end paul_daily_pencil_production_l713_713744


namespace spell_AMC10_paths_l713_713667

noncomputable def number_of_paths_AMC10 : ℕ :=
  12

theorem spell_AMC10_paths :
  let start_at_A := true -- we start at A
  let move_adjacent := true -- can only move to adjacent letters
  let external_block_10 := true -- consider '10' as a single block adjacent to 'C'
  start_at_A ∧ move_adjacent ∧ external_block_10 → number_of_paths_AMC10 = 12 :=
by
  intros _ _ _
  exact rfl

end spell_AMC10_paths_l713_713667


namespace a1_arithmetic_sequence_exist_constants_a_b_l713_713608

def Sn (n : ℕ) : ℕ := 3 * n^2 + 5 * n
def a_n (n : ℕ) : ℕ := Sn n - Sn (n - 1)

def b_seq : ℕ → ℤ
| 1       := 80
| (n + 1) := b_seq n / 64

theorem a1_arithmetic_sequence (n : ℕ) : 
  ∃ d : ℕ, ∀ k : ℕ, a_n (k + 1) - a_n k = d :=
sorry

theorem exist_constants_a_b :
  ∃ a b : ℝ, ∀ n : ℕ, (n > 0) →
  a_n(n) = log a (b_seq n) + b :=
sorry

end a1_arithmetic_sequence_exist_constants_a_b_l713_713608


namespace jogger_distance_ahead_l713_713877

theorem jogger_distance_ahead (v_j v_t : ℕ) (train_length time : ℕ)
    (hj : v_j = 9) (ht : v_t = 45) (hk : train_length = 120) (htime : time = 35) :
    let relative_speed : ℕ := (v_t - v_j) * 1000 / 3600 in
    let distance_covered := relative_speed * time in
    let distance_ahead := distance_covered - train_length in
    distance_ahead = 230 := by
  let relative_speed := (v_t - v_j) * 1000 / 3600
  let distance_covered := relative_speed * time
  let distance_ahead := distance_covered - train_length
  sorry

end jogger_distance_ahead_l713_713877


namespace pi_bounds_l713_713341

theorem pi_bounds : 
  3.14 < Real.pi ∧ Real.pi < 3.142 ∧
  9.86 < Real.pi ^ 2 ∧ Real.pi ^ 2 < 9.87 := sorry

end pi_bounds_l713_713341


namespace katya_needs_at_least_ten_l713_713291

variable (x : ℕ)

def prob_katya : ℚ := 4 / 5
def prob_pen  : ℚ := 1 / 2

def expected_correct (x : ℕ) : ℚ := x * prob_katya + (20 - x) * prob_pen

theorem katya_needs_at_least_ten :
  expected_correct x ≥ 13 → x ≥ 10 :=
  by sorry

end katya_needs_at_least_ten_l713_713291


namespace rectangle_width_l713_713889

theorem rectangle_width (L : ℚ) (A : ℚ) (W : ℚ) (hL : L = 3 / 5) (hA : A = 1 / 3) (hAW : A = L * W) : W = 5 / 9 :=
by
  subst hL
  subst hA
  simp at hAW
  sorry

end rectangle_width_l713_713889


namespace solve_system_of_inequalities_l713_713762

theorem solve_system_of_inequalities 
  (x : ℝ) 
  (h1 : x - 3 * (x - 2) ≥ 4)
  (h2 : (1 + 2 * x) / 3 > x - 1) : 
  x ≤ 1 := 
sorry

end solve_system_of_inequalities_l713_713762


namespace total_amount_spent_l713_713737

theorem total_amount_spent (half_dollar_value : ℝ) (wednesday_spend : ℕ) (thursday_spend : ℕ) : 
  wednesday_spend = 4 → thursday_spend = 14 → half_dollar_value = 0.5 → (wednesday_spend + thursday_spend) * half_dollar_value = 9 :=
by
  intros wednesday_cond thursday_cond half_dollar_cond
  rw [wednesday_cond, thursday_cond, half_dollar_cond]
  norm_num
  sorry

end total_amount_spent_l713_713737


namespace cosine_120_eq_negative_half_l713_713958

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l713_713958


namespace solve_triples_l713_713563

theorem solve_triples 
  (a b c : ℕ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_expr_int : ∃ k : ℤ, (a + b) ^ 4 / c + (b + c) ^ 4 / a + (c + a) ^ 4 / b = k) 
  (h_prime : Nat.Prime (a + b + c)) : 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 2 ∧ b = 2 ∧ c = 1) ∨ 
  (a = 6 ∧ b = 3 ∧ c = 2) ∨ 
  (a = 6 ∧ b = 2 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 6 ∧ c = 3) ∨ 
  (a = 3 ∧ b = 6 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 1 ∧ c = 2) ∨ 
  (a = 1 ∧ b = 2 ∧ c = 2) ∨ 
  (a = 1 ∧ b = 1 ∧ c = 1) := sorry

end solve_triples_l713_713563


namespace cosine_120_eq_negative_half_l713_713953

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l713_713953


namespace total_wheels_l713_713032

theorem total_wheels (n_bicycles n_tricycles n_unicycles n_four_wheelers : ℕ)
                     (w_bicycle w_tricycle w_unicycle w_four_wheeler : ℕ)
                     (h1 : n_bicycles = 16)
                     (h2 : n_tricycles = 7)
                     (h3 : n_unicycles = 10)
                     (h4 : n_four_wheelers = 5)
                     (h5 : w_bicycle = 2)
                     (h6 : w_tricycle = 3)
                     (h7 : w_unicycle = 1)
                     (h8 : w_four_wheeler = 4)
  : (n_bicycles * w_bicycle + n_tricycles * w_tricycle
     + n_unicycles * w_unicycle + n_four_wheelers * w_four_wheeler) = 83 := by
  sorry

end total_wheels_l713_713032


namespace cos_120_eq_neg_half_l713_713980

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713980


namespace product_of_ab_l713_713643

theorem product_of_ab (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 7) : a * b = -10 :=
by
  sorry

end product_of_ab_l713_713643


namespace confectioner_customers_l713_713094

theorem confectioner_customers (x : ℕ) (h : 0 < x) :
  (49 * (392 / x - 6) = 392) → x = 28 :=
by
sorry

end confectioner_customers_l713_713094


namespace total_percent_exceeding_speed_limit_l713_713739

theorem total_percent_exceeding_speed_limit (pA pB pC : ℝ) (rA rB rC : ℝ) :
  pA = 0.18 → pB = 0.09 → pC = 0.03 → rA = 0.30 → rB = 0.60 → rC = 0.90 →
  (pA + pB + pC = 0.30) ∧
  (rA * pA = 0.054) ∧
  (rB * pB = 0.054) ∧
  (rC * pC = 0.027) :=
by introv hA hB hC rA_rate rB_rate rC_rate
   split
   { rw [hA, hB, hC], norm_num }
   { split
     { rw [hA, rA_rate], norm_num }
     { split
       { rw [hB, rB_rate], norm_num }
       { rw [hC, rC_rate], norm_num } } }

end total_percent_exceeding_speed_limit_l713_713739


namespace triangle_H1H2H3_is_right_l713_713297

noncomputable def is_right_triangle (H1 H2 H3 : Complex) : Prop :=
(arg (H2 - H1)).val - (arg (H3 - H1)).val = π / 2

theorem triangle_H1H2H3_is_right
  (O A B C D E : Complex)
  (H1 H2 H3 : Complex)
  (hA : abs (A - O) = 1)
  (hB : abs (B - O) = 1)
  (hC : abs (C - O) = 1)
  (hD : abs (D - O) = 1)
  (hE : abs (E - O) = 1)
  (hAC : abs (A - C) = abs (D - O))
  (hBD : abs (B - D) = abs (D - O))
  (hCE : abs (C - E) = abs (D - O))
  (orthoACD : is_orthocenter H1 A D C)
  (orthoBCD : is_orthocenter H2 B D C)
  (orthoBCE : is_orthocenter H3 B C E) :
  is_right_triangle H1 H2 H3 := sorry

end triangle_H1H2H3_is_right_l713_713297


namespace perimeter_of_square_III_l713_713035

theorem perimeter_of_square_III :
  ∀ (P₁ P₂ : ℕ), P₁ = 20 → P₂ = 16 →
  4 * (P₁ / 4 - P₂ / 4) = 4 :=
by
  intros P₁ P₂ hP₁ hP₂
  rw [hP₁, hP₂]
  sorry

end perimeter_of_square_III_l713_713035


namespace cos_120_eq_neg_half_l713_713965

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l713_713965


namespace smallest_k_is_two_l713_713705

open Nat

def matches_in_position (a b : ℕ) (pos : Fin 5) : Prop :=
(a / 10^pos) % 10 = (b / 10^pos) % 10

def satisfies_condition (N : Fin 2 → ℕ) (A : ℕ) : Prop :=
∃ i : Fin 2, ∃ pos : Fin 5, matches_in_position (N i) A pos

theorem smallest_k_is_two :
  ∃ (N : Fin 2 → ℕ),
  (∀ A, (∀ i : Fin 4, A / 10^i % 10 ≤ A / 10^(i+1) % 10) →
   satisfies_condition N A) ∧
  (∀ N' : Fin 1 → ℕ, ∃ A, (∀ i : Fin 4, A / 10^i % 10 ≤ A / 10^(i+1) % 10) ∧ ¬ satisfies_condition N' A) :=
sorry

end smallest_k_is_two_l713_713705


namespace max_distance_l713_713847

def avg_mpg_highway : ℝ := 12.2
def gallons : ℝ := 23
def expected_distance : ℝ := 280.6

theorem max_distance (avg_mpg_highway = 12.2) (gallons = 23) : avg_mpg_highway * gallons = expected_distance := 
by
  simp
  sorry

end max_distance_l713_713847


namespace conjugate_of_complex_number_l713_713767

open Complex

theorem conjugate_of_complex_number : conj ((1 + 2 * I) * (1 + 2 * I)) = -3 - 4 * I := 
by
  sorry

end conjugate_of_complex_number_l713_713767


namespace sequence_inequality_l713_713626

theorem sequence_inequality {a : ℕ → ℝ} (h₁ : a 1 > 1)
  (h₂ : ∀ n : ℕ, a (n+1) = (a n ^ 2 + 1) / (2 * a n)) :
  ∀ n : ℕ, (∑ i in Finset.range n, a (i + 1)) < n + 2 * (a 1 - 1) := by
  sorry

end sequence_inequality_l713_713626


namespace katya_minimum_problems_l713_713282

-- Defining the conditions
def katya_probability_solve : ℚ := 4 / 5
def pen_probability_solve : ℚ := 1 / 2
def total_problems : ℕ := 20
def minimum_correct_for_good_grade : ℚ := 13

-- The Lean statement to prove
theorem katya_minimum_problems (x : ℕ) :
  x * katya_probability_solve + (total_problems - x) * pen_probability_solve ≥ minimum_correct_for_good_grade → x ≥ 10 :=
sorry

end katya_minimum_problems_l713_713282


namespace problem_inequality_l713_713580

theorem problem_inequality 
  (m n : ℝ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : m + n = 1) : 
  (m + 1 / m) * (n + 1 / n) ≥ 25 / 4 := 
sorry

end problem_inequality_l713_713580


namespace cos_120_eq_neg_half_l713_713938

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713938


namespace disapprove_estimate_l713_713251

theorem disapprove_estimate :
  let sample_size := 50
  let approval_count := 14
  let disapproval_ratio := (sample_size - approval_count) / sample_size
  let total_population := 9600
  let estimated_disapproval := total_population * disapproval_ratio
  estimated_disapproval ≈ 6912 := 
by
  let sample_size := 50
  let approval_count := 14
  let disapproval_ratio := (sample_size - approval_count) / sample_size.toRat
  let total_population := 9600
  let estimated_disapproval := total_population.toRat * disapproval_ratio
  exact (estimated_disapproval - 6912).abs < 1

end disapprove_estimate_l713_713251


namespace arithmetic_sum_S9_l713_713308

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable (S : ℕ → ℝ) -- Define the sum of the first n terms
variable (d : ℝ) -- Define the common difference
variable (a_1 : ℝ) -- Define the first term of the sequence

-- Assume the arithmetic sequence properties
axiom arith_seq_def : ∀ n, a (n + 1) = a_1 + n * d

-- Define the sum of the first n terms
axiom sum_first_n_terms : ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom given_condition : a 1 + a 7 = 15 - a 4

theorem arithmetic_sum_S9 : S 9 = 45 :=
by
  -- Proof omitted
  sorry

end arithmetic_sum_S9_l713_713308


namespace min_a_n_l713_713627

def a_n (n : ℕ) : ℤ := n^2 - 8 * n + 5

theorem min_a_n : ∃ n : ℕ, ∀ m : ℕ, a_n n ≤ a_n m ∧ a_n n = -11 :=
by
  sorry

end min_a_n_l713_713627


namespace katya_solves_enough_l713_713296

theorem katya_solves_enough (x : ℕ) :
  (0 ≤ x ∧ x ≤ 20) → -- x should be within the valid range of problems
  (4 / 5) * x + (1 / 2) * (20 - x) ≥ 13 → 
  x ≥ 10 :=
by 
  intros h₁ h₂
  -- Formalize the expected value equation and the inequality transformations
  sorry

end katya_solves_enough_l713_713296


namespace max_volume_pyramid_l713_713587

noncomputable def max_volume_of_pyramid (a : ℝ) : ℝ :=
  let base_area := (sqrt 3 / 4) * (sqrt(3 * a^2) ^ 2) in
  let height := sqrt(a^2 - (a / sqrt 3)^2) in
  (1 / 3) * base_area * height

theorem max_volume_pyramid {a : ℝ} (h_a_pos : 0 < a) :
  max_volume_of_pyramid a = a^3 / 6 :=
sorry

end max_volume_pyramid_l713_713587


namespace problem_l713_713573

noncomputable def f_alpha (α : ℝ) (x : ℝ) : ℤ := ⌊α * x + 1/2⌋

theorem problem {α : ℝ} (hα : 1 < α) (n : ℕ) :
  let β := 1 / α in
  f_alpha β (f_alpha α n) = n := by
  sorry

end problem_l713_713573


namespace cos_120_eq_neg_half_l713_713933

-- Definitions based on the given conditions
def unit_circle := true  -- Trivially true because we just need the unit circle concept.

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ ≤ 180

def cosine_identity_second_quadrant (θ : ℝ) (hθ : second_quadrant θ) : ℝ :=
  -cos (180 - θ)

-- Given condition for known trigonometric value
def cos_60 : ℝ := 1 / 2

-- Final proof statement to show that cos 120∘ equals -1/2
theorem cos_120_eq_neg_half : cos 120 = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l713_713933


namespace prove_condition_l713_713710

variable {α β : Type} [plane α] [plane β]
variable {m n : Type} [line m] [line n]
variable [rel_planes : rel α β]
variable [rel_lines : rel m β] [rel_lines : rel n β]

-- Conditions
def lines_in_plane (m n : Type) [line m] [line n] [plane α] := ∀ (m n : Type), m, n ⊆ α
def planes_disjoint (α β : Type) [plane α] [plane β] := α ≠ β
def planes_parallel (α β : Type) [plane α] [plane β] := ∀ (α β : Type), α ∥ β

-- The math proof problem
theorem prove_condition : 
  (planes_parallel α β → (lines_in_plane m n α → lines_in_plane m n β)) ∧ 
  (¬((lines_in_plane m n β) → (planes_parallel α β))) := sorry

end prove_condition_l713_713710


namespace washing_time_l713_713500

theorem washing_time (shirts pants sweaters jeans : ℕ) (items_per_cycle minutes_per_cycle : ℕ) :
  shirts = 18 → pants = 12 → sweaters = 17 → jeans = 13 → 
  items_per_cycle = 15 → minutes_per_cycle = 45 → 
  (shirts + pants + sweaters + jeans) / items_per_cycle * minutes_per_cycle / 60 = 3 :=
by
  intros h_shirts h_pants h_sweaters h_jeans h_cycle_max h_cycle_time
  calc
    (shirts + pants + sweaters + jeans) / items_per_cycle * minutes_per_cycle / 60
    = (18 + 12 + 17 + 13) / 15 * 45 / 60 : by rw [h_shirts, h_pants, h_sweaters, h_jeans, h_cycle_max, h_cycle_time]
    ... = 60 / 15 * 45 / 60 : by norm_num
    ... = 4 * 45 / 60 : by norm_num
    ... = 180 / 60 : by norm_num
    ... = 3 : by norm_num

end washing_time_l713_713500


namespace min_value_4m_2n_l713_713582

theorem min_value_4m_2n (a m n : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) 
  (hx : ∀ x, f x = log a (x - 1) + 1)
  (hf_A : f 2 = 1)
  (h_line : ∀ x y, A = (2,1) ∧ (mx - y + n = 0)) :
  4 ^ m + 2 ^ n ≥ 2 * Real.sqrt 2 :=
sorry

end min_value_4m_2n_l713_713582


namespace sum_first_1000_b_n_l713_713793

noncomputable def a_n : ℕ → ℕ
| 1 := 1
| n := n

def b_n (n : ℕ) : ℕ :=
  ⌊ log (a_n n) ⌋₊

theorem sum_first_1000_b_n :
  ∑ n in Finset.range 1000, b_n (n + 1) = 1893 :=
by {
  sorry
}

end sum_first_1000_b_n_l713_713793


namespace sum_of_squares_eq_229_l713_713004

-- The conditions
variables (x y : ℤ)
axiom diff_eq_221 : x^2 - y^2 = 221

-- The proof goal
theorem sum_of_squares_eq_229 : x^2 - y^2 = 221 → ∃ x y : ℤ, x^2 + y^2 = 229 :=
by
  sorry

end sum_of_squares_eq_229_l713_713004


namespace find_k_perpendicular_l713_713168

noncomputable def k_perpendicular_condition (k : ℝ) : Prop :=
  let v1 := (k, 1, 2)
  let v2 := (-4, 2, -1)
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

theorem find_k_perpendicular : ∃ k : ℝ, k_perpendicular_condition k ∧ k = 0 :=
by {
  use 0,
  unfold k_perpendicular_condition,
  simp,
  done
}

end find_k_perpendicular_l713_713168


namespace proof_problem_correct_l713_713721

noncomputable def proof_problem (r : ℂ) : Prop :=
  r^4 = 1 ∧ r ≠ 1 → (r - 1) * (r^2 - 1) * (r^3 - 1) = 0

theorem proof_problem_correct (r : ℂ) : proof_problem r :=
by {
  intro h,
  obtain ⟨h1, h2⟩ := h,
  sorry
}

end proof_problem_correct_l713_713721


namespace jezebel_total_flower_cost_l713_713276

theorem jezebel_total_flower_cost :
  let red_rose_count := 2 * 12
  let red_rose_cost := 1.5
  let sunflower_count := 3
  let sunflower_cost := 3
  (red_rose_count * red_rose_cost + sunflower_count * sunflower_cost = 45) :=
by
  let red_rose_count := 2 * 12
  let red_rose_cost := 1.5
  let sunflower_count := 3
  let sunflower_cost := 3
  sorry

end jezebel_total_flower_cost_l713_713276


namespace hyperbola_asymptotes_l713_713011

-- Defining constants a and b
def a := 3
def b := 4

-- Defining the asymptote function based on the hyperbola equation
noncomputable def asymptote (x : ℝ) : (ℝ → Prop) :=
  λ y, y = (b / a) * x ∨ y = -(b / a) * x

-- Stating the theorem that proves the asymptotes
theorem hyperbola_asymptotes (x y : ℝ) : 
  (y = (b / a) * x ∨ y = -(b / a) * x) ↔ asymptote x y :=
by sorry

end hyperbola_asymptotes_l713_713011


namespace modulusOfComplexNumber_proof_l713_713205

noncomputable def complexNumber {a : ℝ} (h : (a - 1) + 1 * Complex.I = (0 : ℂ)) : ℂ :=
  (2 + Real.sqrt 2 * Complex.I) / (a - Complex.I)

theorem modulusOfComplexNumber_proof (a : ℝ) (h : (a - 1) + 1 * Complex.I = (0 : ℂ)) : Complex.abs (complexNumber h) = Real.sqrt 3 := by
  sorry

end modulusOfComplexNumber_proof_l713_713205


namespace solve_system_l713_713921

noncomputable def solution_existence (x y z : ℤ) : Prop :=
  4^(x^2 + 2*x*y + 1) = (z + 2) * 7^(abs y - 1) ∧
  sin (3 * Real.pi * z / 2) = 1

theorem solve_system : ∀ x y z : ℤ, 
  solution_existence x y z → (x, y, z) = (1, -1, -1) ∨ (x, y, z) = (-1, 1, -1) :=
by
  intros
  sorry

end solve_system_l713_713921


namespace katya_solves_enough_l713_713294

theorem katya_solves_enough (x : ℕ) :
  (0 ≤ x ∧ x ≤ 20) → -- x should be within the valid range of problems
  (4 / 5) * x + (1 / 2) * (20 - x) ≥ 13 → 
  x ≥ 10 :=
by 
  intros h₁ h₂
  -- Formalize the expected value equation and the inequality transformations
  sorry

end katya_solves_enough_l713_713294


namespace selling_price_correct_l713_713902

def meters_of_cloth : ℕ := 45
def profit_per_meter : ℝ := 12
def cost_price_per_meter : ℝ := 88
def total_selling_price : ℝ := 4500

theorem selling_price_correct :
  (cost_price_per_meter * meters_of_cloth) + (profit_per_meter * meters_of_cloth) = total_selling_price :=
by
  sorry

end selling_price_correct_l713_713902


namespace common_area_of_rotated_squares_l713_713137

theorem common_area_of_rotated_squares (theta : ℝ) (cos_theta : ℝ) (sin_theta : ℝ)
  (hcos : cos_theta = 3 / 5) (hsin : sin_theta = 4 / 5) :
  let side_len := 2 in
  let area_common := side_len * side_len * sin_theta * cos_theta in
  area_common = 48 / 25 :=
by
  sorry

end common_area_of_rotated_squares_l713_713137


namespace tangent_perpendicular_l713_713365

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := exp x - m * x

theorem tangent_perpendicular (m : ℝ) (x : ℝ) (h : f' x = -2) : m > 2 :=
by
  have h_derivative : f' x = exp x - m := by
    sorry  -- Derivative computation goes here.
  rw h at h_derivative
  have : -2 = exp x - m := h_derivative
  have : m = 2 + exp x := by
    linarith
  exact h
  sorry

end tangent_perpendicular_l713_713365


namespace area_enclosed_by_trajectory_of_P_l713_713304

-- Definitions of points
structure Point where
  x : ℝ
  y : ℝ

-- Definition of fixed points A and B
def A : Point := { x := -3, y := 0 }
def B : Point := { x := 3, y := 0 }

-- Condition for the ratio of distances
def ratio_condition (P : Point) : Prop :=
  ((P.x + 3)^2 + P.y^2) / ((P.x - 3)^2 + P.y^2) = 1 / 4

-- Definition of a circle based on the derived condition in the solution
def circle_eq (P : Point) : Prop :=
  (P.x + 5)^2 + P.y^2 = 16

-- Theorem stating the area enclosed by the trajectory of point P is 16π
theorem area_enclosed_by_trajectory_of_P : 
  (∀ P : Point, ratio_condition P → circle_eq P) →
  ∃ A : ℝ, A = 16 * Real.pi :=
by
  sorry

end area_enclosed_by_trajectory_of_P_l713_713304


namespace symmetric_function_zero_l713_713061

variables {α : Type*} (f : α → α → Prop)

theorem symmetric_function_zero (x y : α) (h : f x y ↔ f y x) : f x y = 0 :=
by sorry

end symmetric_function_zero_l713_713061


namespace expected_interval_trains_l713_713764

noncomputable def expected_interval_between_trains 
  (northern_route_time southern_route_time : ℕ)
  (avg_diff : ℚ)
  (home_to_work_diff : ℚ)
  (northern_prob : ℚ) 
  (expected_interval : ℚ) : Prop :=
  let p := northern_prob in
  let E1 := southern_route_time * p + northern_route_time * (1 - p) in
  let E2 := northern_route_time * p + southern_route_time * (1 - p) in
  home_to_work_diff = E2 - E1 ∧
  expected_interval = 5 / (4 * (1 - p)) 

theorem expected_interval_trains:
  expected_interval_between_trains 17 11 (5 / 4) 1 (7 / 12) 3 :=
sorry

end expected_interval_trains_l713_713764


namespace fair_die_odd_prob_l713_713064

theorem fair_die_odd_prob : 
  let outcomes := {1, 2, 3, 4, 5, 6}
  let favorable_outcomes := {x : ℕ | x ∈ outcomes ∧ x % 2 = 1}
  (favorable_outcomes.card : ℝ) / (outcomes.card : ℝ) = 1 / 2 :=
by
  let outcomes := {1, 2, 3, 4, 5, 6}
  let favorable_outcomes := {x : ℕ | x ∈ outcomes ∧ x % 2 = 1}
  let total_number := outcomes.card
  let favorable_number := favorable_outcomes.card
  have h1 : favorable_number = 3 := by sorry
  have h2 : total_number = 6 := by sorry
  have h3 : (favorable_number : ℝ) / (total_number : ℝ) = 1 / 2 := by
    rw [h1, h2]
    norm_num
  exact h3

end fair_die_odd_prob_l713_713064


namespace inequality_solution_l713_713562

theorem inequality_solution (x : ℝ) : 
  (3 / 16 + | x - 5 / 32 | < 7 / 32) ↔ (1 / 8 < x ∧ x < 3 / 16) :=
by
  sorry

end inequality_solution_l713_713562


namespace simplify_trig_expression_l713_713344

-- Definitions for trigonometric identities
def cot (B : ℝ) : ℝ := cos B / sin B
def csc (B : ℝ) : ℝ := 1 / sin B
def tan (B : ℝ) : ℝ := sin B / cos B
def sec (B : ℝ) : ℝ := 1 / cos B

-- Main theorem to be proved
theorem simplify_trig_expression (B : ℝ) :
  (1 - cot B + csc B) * (1 - tan B - sec B) = csc B * sec B * sin(B - π / 2) := 
by 
  sorry

end simplify_trig_expression_l713_713344


namespace butterfly_1023_distance_l713_713468

noncomputable def omega : Complex := Complex.exp (Complex.I * Real.pi / 4)

noncomputable def Q (n : ℕ) : Complex :=
  match n with
  | 0     => 0
  | k + 1 => Q k + (k + 1) * omega ^ k

noncomputable def butterfly_distance (n : ℕ) : ℝ := Complex.abs (Q n)

theorem butterfly_1023_distance : butterfly_distance 1023 = 511 * Real.sqrt (2 + Real.sqrt 2) :=
  sorry

end butterfly_1023_distance_l713_713468


namespace range_of_a_l713_713378

theorem range_of_a (a : ℝ) :
  (2 * 0 - 0 + a > 0 ∨ 2 * 1 - 1 + a > 0) → a ∈ Icc (-1 : ℝ) 0 :=
by
  intro h
  sorry

end range_of_a_l713_713378


namespace length_AD_l713_713461

-- Define the endpoints of the segments
def A := 0
def D := 4 * x
def B := x
def C := 2 * x

def midpoint (p q : ℝ) : ℝ := (p + q) / 2

-- Given conditions:
-- B and C divide AD into four equal lengths with D being endpoint further from A.
-- M is the midpoint of both AD and AB.
-- Given MC = 10

variable (x : ℝ)
variable (M D : ℝ)

-- Condition: D = 4 * x
def D_def : D = 4 * x := by sorry

-- Condition: M is the midpoint of AD and AB
def M_def1 : M = midpoint A D := by sorry
def M_def2 : M = midpoint A B := by sorry

-- Given: MC = 10
def MC : ℝ := midpoint B C + C - M
axiom MC_value : MC = 10

-- The length of AD is 16
theorem length_AD : D = 16 :=
by sorry

end length_AD_l713_713461


namespace cos_120_eq_neg_half_l713_713968

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l713_713968


namespace cos_120_degrees_eq_l713_713947

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l713_713947


namespace equilibrium_shift_if_K_changes_l713_713839

-- Define the equilibrium constant and its relation to temperature
def equilibrium_constant (T : ℝ) : ℝ := sorry

-- Define the conditions
axiom K_related_to_temp (T₁ T₂ : ℝ) (K₁ K₂ : ℝ) :
  equilibrium_constant T₁ = K₁ ∧ equilibrium_constant T₂ = K₂ → T₁ = T₂ ↔ K₁ = K₂

axiom K_constant_with_concentration_change (T : ℝ) (K : ℝ) (c₁ c₂ : ℝ) :
  equilibrium_constant T = K → equilibrium_constant T = K

axiom K_squared_with_stoichiometric_double (T : ℝ) (K : ℝ) :
  equilibrium_constant (2 * T) = K * K

-- Define the problem to be proved
theorem equilibrium_shift_if_K_changes (T₁ T₂ : ℝ) (K₁ K₂ : ℝ) :
  equilibrium_constant T₁ = K₁ ∧ equilibrium_constant T₂ = K₂ → K₁ ≠ K₂ → T₁ ≠ T₂ := 
sorry

end equilibrium_shift_if_K_changes_l713_713839


namespace cos_120_degrees_l713_713999

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l713_713999


namespace rowing_time_ratio_l713_713796

noncomputable def V_b : ℝ := 18
noncomputable def V_s : ℝ := 6
noncomputable def V_u : ℝ := V_b - V_s
noncomputable def V_d : ℝ := V_b + V_s
noncomputable def T_u : ℝ := 1 / V_u
noncomputable def T_d : ℝ := 1 / V_d

theorem rowing_time_ratio 
  (Vb : ℝ) (Vs : ℝ)
  (hb : Vb = 18) (hs : Vs = 6) :
  (1 / (Vb - Vs)) / (1 / (Vb + Vs)) = 2 := by
{
  rw [hb, hs],
  rw [show 18 - 6 = 12, by norm_num],
  rw [show 18 + 6 = 24, by norm_num],
  norm_num,
}

#eval rowing_time_ratio 18 6 rfl rfl

end rowing_time_ratio_l713_713796


namespace cos_120_eq_neg_half_l713_713963

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l713_713963


namespace solve_system_of_equations_l713_713345

theorem solve_system_of_equations :
  ∃ x y z : ℚ, 
    (y * z = 3 * y + 2 * z - 8) ∧
    (z * x = 4 * z + 3 * x - 8) ∧
    (x * y = 2 * x + y - 1) ∧
    ((x = 2 ∧ y = 3 ∧ z = 1) ∨ 
     (x = 3 ∧ y = 5 / 2 ∧ z = -1)) := 
by
  sorry

end solve_system_of_equations_l713_713345


namespace area_enclosed_by_abs_graph_l713_713051

theorem area_enclosed_by_abs_graph : 
  (∃ (x y : ℝ), abs (2 * x) + abs (3 * y) = 12) →
  ∑ (x y : ℝ), abs (2 * x) + abs (3* y) = 48 := by
  sorry

end area_enclosed_by_abs_graph_l713_713051


namespace given_conditions_l713_713647

theorem given_conditions :
  ∀ (t : ℝ), t > 0 → t ≠ 1 → 
  let x := t^(2/(t-1))
  let y := t^((t+1)/(t-1))
  ¬ ((y * x^(1/y) = x * y^(1/x)) ∨ (y * x^y = x * y^x) ∨ (y^x = x^y) ∨ (x^(x+y) = y^(x+y))) :=
by
  intros t ht_pos ht_ne_1 x_def y_def
  let x := x_def
  let y := y_def
  sorry

end given_conditions_l713_713647


namespace problem_statement_l713_713313

theorem problem_statement (x : ℝ) (h : x^3 - 3 * x = 7) : x^7 + 27 * x^2 = 76 * x^2 + 270 * x + 483 :=
sorry

end problem_statement_l713_713313


namespace part1_part2_l713_713613

-- Definitions for the function f(x)
def f (x : ℝ) : ℝ := 2 * (Real.sqrt 3) * Real.sin x * Real.cos x + 2 * (Real.cos x)^2

-- Statement for the first proof problem.
theorem part1 : f (Real.pi / 24) = Real.sqrt 2 + 1 :=
  sorry

-- Statement for the second proof problem.
theorem part2 (m : ℝ) : (∀ x ∈ Icc (-m) m, f' x > 0) → m ≤ Real.pi / 6 :=
  sorry

end part1_part2_l713_713613


namespace minimize_volume_difference_l713_713183

variable (a b c : ℝ)
variable (h : a < b)
variable (h' : b < c)

def min_edge_length (a b : ℝ) : ℝ :=
  min b (4 * a / 3)

theorem minimize_volume_difference :
  ∀ (a b c : ℝ), (a < b) → (b < c) → 
    ∃ x, x = min_edge_length a b :=
by
  intros a b c h h'
  use min_edge_length a b
  sorry

end minimize_volume_difference_l713_713183


namespace factorial_division_l713_713521

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division (n : ℕ) (h : n > 0) : factorial (n) / factorial (n - 1) = n :=
by
  sorry

example : factorial 12 / factorial 11 = 12 :=
by
  exact factorial_division 12 (by norm_num)

end factorial_division_l713_713521


namespace problem_proof_l713_713176

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2 * x else f (x + 1)

theorem problem_proof : f (4/3) + f (-4/3) = 4 := by
  sorry

end problem_proof_l713_713176


namespace quadratic_vertex_coordinates_l713_713575

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ :=
  -2 * (x + 1)^2 - 4

-- State the main theorem to be proved: The vertex of the quadratic function is at (-1, -4)
theorem quadratic_vertex_coordinates : 
  ∃ h k : ℝ, ∀ x : ℝ, quadratic x = -2 * (x + h)^2 + k ∧ h = -1 ∧ k = -4 := 
by
  -- proof required here
  sorry

end quadratic_vertex_coordinates_l713_713575


namespace trigonometric_identity_l713_713118

theorem trigonometric_identity :
  (tan (real.to_radians 22.5) / (1 - (tan (real.to_radians 22.5))^2)) = 1/2 :=
sorry

end trigonometric_identity_l713_713118


namespace joan_kittens_count_correct_l713_713277

def joan_initial_kittens : Nat := 8
def kittens_from_friends : Nat := 2
def joan_total_kittens (initial: Nat) (added: Nat) : Nat := initial + added

theorem joan_kittens_count_correct : joan_total_kittens joan_initial_kittens kittens_from_friends = 10 := 
by
  sorry

end joan_kittens_count_correct_l713_713277


namespace trajectory_point_P_quadrilateral_area_constant_l713_713189

-- Define circle F1 and fixed point F2
def circle_F1 (x y : ℝ) := (x + 1)^2 + y^2 = 16
def F2 := (1, 0 : ℝ)

-- Define moving point A on circle F1
def on_circle_F1 (A : ℝ × ℝ) := circle_F1 (A.1) (A.2)

-- Define point P being the intersection of the perpendicular bisector of F2A and radius F1A
def perpendicular_bisector_intersection_point (A P : ℝ × ℝ) :=
  let PA := dist P A
  let PF2 := dist P F2
  PF2 = PA

-- Define trajectory C as an ellipse
def is_ellipse (P : ℝ × ℝ) := (P.1^2 / 4) + (P.2^2 / 3) = 1

-- Theorem for Part I
theorem trajectory_point_P (A P : ℝ × ℝ)
  (hA_on_circle : on_circle_F1 A)
  (h_intersect : perpendicular_bisector_intersection_point A P) :
    is_ellipse P := sorry

-- Define quadrilateral EFGH, its vertices on curve C, its diagonals passing through origin, and slope condition
def is_on_curve_C (E F G H : ℝ × ℝ) := is_ellipse E ∧ is_ellipse F ∧ is_ellipse G ∧ is_ellipse H
def pass_through_origin (E G F H : ℝ × ℝ) :=
  let slope (P1 P2 : ℝ × ℝ) := (P2.2 - P1.2) / (P2.1 - P1.1)
  (origin : ℝ × ℝ) := (0, 0)
  slope E G * slope F H = -3/4

def area_quadrilateral (E F G H : ℝ × ℝ) : ℝ := 
  -- Placeholder for actual area calculation function
  4 * sqrt 3

-- Theorem for Part II
theorem quadrilateral_area_constant (E F G H : ℝ × ℝ)
  (h_vertices_on_curve : is_on_curve_C E F G H)
  (h_diagonal_pass : pass_through_origin E G F H) :
    area_quadrilateral E F G H = 4 * sqrt 3 := sorry

end trajectory_point_P_quadrilateral_area_constant_l713_713189


namespace gcf_252_96_l713_713425

theorem gcf_252_96 : Int.gcd 252 96 = 12 := by
  sorry

end gcf_252_96_l713_713425


namespace simplest_quadratic_radical_correct_l713_713435

noncomputable def simplest_quadratic_radical (A B C D: ℝ) : ℝ := B

theorem simplest_quadratic_radical_correct (A B C D : ℝ) 
  (hA : A = Real.sqrt 9) 
  (hB : B = Real.sqrt 7) 
  (hC : C = Real.sqrt 20) 
  (hD : D = Real.sqrt (1/3)) : 
  simplest_quadratic_radical A B C D = Real.sqrt 7 := 
by 
  rw simplest_quadratic_radical 
  rw hB
  sorry

end simplest_quadratic_radical_correct_l713_713435


namespace polynomial_value_bound_l713_713324

theorem polynomial_value_bound 
  (n : ℕ) 
  (a : Fin n → ℤ)
  (x : Fin (n + 1) → ℤ)
  (h: ∀ i j, i < j → x i < x j) : 
  ∃ i, (| ∑ j in Finset.range (n + 1), P (x j) * ∏ l in Finset.range (n + 1), if l = j then 1 else (x j - x l) | ≥ n.factorial / (2 ^ n) : ℝ :=
begin
   sorry
end

noncomputable def P (x : ℤ) : ℤ :=
  x ^ n + ∑ i in Finset.range n, a i * x ^ i

end polynomial_value_bound_l713_713324


namespace total_pages_read_l713_713690

variable (Jairus_pages : ℕ)
variable (Arniel_pages : ℕ)
variable (J_total : Jairus_pages = 20)
variable (A_total : Arniel_pages = 2 + 2 * Jairus_pages)

theorem total_pages_read : Jairus_pages + Arniel_pages = 62 := by
  rw [J_total, A_total]
  sorry

end total_pages_read_l713_713690


namespace lowest_value_meter_can_record_l713_713475

theorem lowest_value_meter_can_record (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 6) (h2 : A = 2) : A = 2 :=
by sorry

end lowest_value_meter_can_record_l713_713475


namespace polynomial_degree_l713_713542

-- Definitions of the polynomials
def poly1 : Polynomial ℝ := 3 * X^5 + 2 * X^3 - X + 5
def poly2 : Polynomial ℝ := 4 * X^8 - 8 * X^5 + 11 * X^2 + 3
def poly3 : Polynomial ℝ := (2 * X^2 + 3)^6

-- The proof statement
theorem polynomial_degree : (poly1 * poly2 - poly3).degree = 13 :=
by
  sorry

end polynomial_degree_l713_713542


namespace fixed_point_of_symmetric_line_l713_713244

-- Define the symmetric function
def symmetric_point (p : ℝ × ℝ) (c : ℝ × ℝ) : ℝ × ℝ :=
  (2 * c.1 - p.1, 2 * c.2 - p.2)

-- Define the line l_1
def l1 (k : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - 4)}

-- Define a function to check if a point lies on a line
def lies_on_line (p : ℝ × ℝ) (line : set (ℝ × ℝ)) : Prop :=
  p ∈ line

theorem fixed_point_of_symmetric_line 
  (k : ℝ)
  (M : ℝ × ℝ := (4, 0))
  (C : ℝ × ℝ := (2, 1))
  (N : ℝ × ℝ := symmetric_point M C)
  : 
  (∀ p, lies_on_line p (l1 k) ↔ lies_on_line (symmetric_point p C) (l1 k)) → 
  lies_on_line N (l1 k) :=
sorry

end fixed_point_of_symmetric_line_l713_713244


namespace correct_propositions_count_l713_713911

theorem correct_propositions_count :
  let proposition1 := ∃ (p: ℕ → Prop), (∀ i, 1 ≤ i ∧ i ≤ 13 → p i) ∧ 
                      (∀ i, ∀ j, p i = p j → i = j) → False
  let proposition2 := ∃ (students: ℕ) (sampleSize: ℕ), 
                      students > 0 ∧ sampleSize ≤ students ∧ 10 = sampleSize
  let proposition3 := ∃ (shots: ℕ) (successProbability: ℝ), 
                      successProbability = 0.7 ∧ shots = 10 ∧ 
                      ¬ (successProbability * shots = 7)
  let proposition4 := ∃ (draws: ℕ → bool), 
                      (0.6 <= (count_true (λ b, b) draws) / length draws 
                      ∧ 0.6 >= (count_true (λ b, b) draws) / length draws)

  (proposition1 ∧ proposition2 ∧ proposition3 ∧ proposition4)
  → 3 = cardinality (filter id [true, true, false, true]) := 
by
  sorry

end correct_propositions_count_l713_713911


namespace max_odd_integers_chosen_l713_713115

theorem max_odd_integers_chosen (a b c d e f : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) (h_prod_even : a * b * c * d * e * f % 2 = 0) : 
  (∀ n : ℕ, n = 5 → ∃ a b c d e, (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1) ∧ f % 2 = 0) :=
sorry

end max_odd_integers_chosen_l713_713115


namespace max_b_c_value_l713_713465

theorem max_b_c_value (a b c : ℕ) (h1 : a > b) (h2 : a + b = 18) (h3 : c - b = 2) : b + c = 18 :=
sorry

end max_b_c_value_l713_713465


namespace exists_convex_pentagon_l713_713179

theorem exists_convex_pentagon (S : Finset (Fin 9 → ℝ)) (h₀ : ∀ (p1 p2 p3 : (Fin 9 → ℝ)), p1 ≠ p2 → p2 ≠ p3 → p3 ≠ p1 → 
  ¬ (Collinear ℝ ({p1, p2, p3} : Set (Fin 9 → ℝ)))) :
  ∃ (A B C D E : Fin 9 → ℝ),
    A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧ E ∈ S ∧ 
    ¬Collinear ℝ ({A, B, C, D, E} : Set (Fin 9 → ℝ)) ∧
    ConvexHull ℝ ({A, B, C, D, E} : Set (Fin 9 → ℝ)) = 
    ({A, B, C, D, E} : Set (Fin 9 → ℝ)) :=
sorry

end exists_convex_pentagon_l713_713179


namespace perfect_square_condition_l713_713559

def is_perfect_square (x : ℤ) : Prop := ∃ k : ℤ, k^2 = x

noncomputable def a_n (n : ℕ) : ℤ := (10^n - 1) / 9

theorem perfect_square_condition (n b : ℕ) (h1 : 0 < b) (h2 : b < 10) :
  is_perfect_square ((a_n (2 * n)) - b * (a_n n)) ↔ (b = 2 ∨ (b = 7 ∧ n = 1)) := by
  sorry

end perfect_square_condition_l713_713559


namespace parity_FG_even_odd_find_g_ln_ex_1_l713_713188

variable (f : ℝ → ℝ)
def F (x : ℝ) := (f x + f (-x)) / 2
def G (x : ℝ) := (f x - f (-x)) / 2

theorem parity_FG_even_odd : 
  (∀ x, F f (-x) = F f x) ∧ (∀ x, G f (-x) = - (G f x)) :=
by {
  sorry
}

noncomputable def f_ln_ex_1 := λ (x : ℝ), Real.log (Real.exp x + 1)

theorem find_g_ln_ex_1 :
  ∀ x : ℝ, G f_ln_ex_1 x = x / 2 :=
by {
  sorry
}

end parity_FG_even_odd_find_g_ln_ex_1_l713_713188


namespace student_range_exact_student_count_l713_713866

-- Definitions for the conditions
def retail_price (x : ℕ) : ℕ := 240
def wholesale_price (x : ℕ) : ℕ := 260 / (x + 60)

def student_conditions (x : ℕ) : Prop := (x < 250) ∧ (x + 60 ≥ 250)
def wholesale_retail_equation (a : ℕ) : Prop := (240^2 / a) * 240 = (260 / (a+60)) * 288

-- Proofs of the required statements
theorem student_range (x : ℕ) (hc : student_conditions x) : 190 ≤ x ∧ x < 250 :=
by {
  sorry
}

theorem exact_student_count (a : ℕ) (heq : wholesale_retail_equation a) : a = 200 :=
by {
  sorry
}

end student_range_exact_student_count_l713_713866


namespace find_y_l713_713347

theorem find_y (y : ℝ) (h : real.cbrt (1 + real.sqrt (2 * y - 3)) = real.cbrt 27) : y = 339.5 :=
sorry

end find_y_l713_713347


namespace general_term_a_n_sum_sequence_l713_713606

-- Define the sequence sum
def S (n : ℕ) : ℝ := (3 * n / 2) - (n^2 / 2)

-- Given the condition on S_n, prove a_n formula
theorem general_term_a_n (n : ℕ) (h : 0 < n) : 
  ∃ a : ℕ → ℝ, (∀ n : ℕ, 0 < n → a n = S n - S (n - 1)) ∧ a n = 2 - n :=
by
  sorry

-- Given the general term of a_n, prove the sum of the sequence
theorem sum_sequence (n : ℕ) (h : 0 < n) :
  ∃ S_alt : ℕ → ℝ, 
    (∀ n : ℕ, 0 < n → S_alt n = (1 / 2) * ∑ k in Finset.range n, (1 / (2 * k + 1 - 2) - 1 / (2 * k + 1 + 2))) ∧ 
    S_alt n = n / (1 - 2 * n) :=
by
  sorry

end general_term_a_n_sum_sequence_l713_713606


namespace students_behind_l713_713440

theorem students_behind
  (h1 : 7 > 0)
  (h2 : 4 > 0)
  (total_students : 9):
  (let original_place := 7 in
   let pass := 4 in
   let new_place := original_place - pass in
   let students_behind := total_students - new_place in
   students_behind = 6) :=
by {
  sorry
}

end students_behind_l713_713440


namespace sum_of_extreme_values_of_g_l713_713312

noncomputable def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |3 * x - 15|

theorem sum_of_extreme_values_of_g :
  (let min_val := -13 in
  let max_val := -8 in
  min_val + max_val = -21) :=
by
  let min_val := -13
  let max_val := -8
  show min_val + max_val = -21
  sorry

end sum_of_extreme_values_of_g_l713_713312


namespace vectors_not_coplanar_l713_713121

def vector_a : Fin 3 → ℤ := ![1, 5, 2]
def vector_b : Fin 3 → ℤ := ![-1, 1, -1]
def vector_c : Fin 3 → ℤ := ![1, 1, 1]

def scalar_triple_product (a b c : Fin 3 → ℤ) : ℤ :=
  a 0 * (b 1 * c 2 - b 2 * c 1) -
  a 1 * (b 0 * c 2 - b 2 * c 0) +
  a 2 * (b 0 * c 1 - b 1 * c 0)

theorem vectors_not_coplanar :
  scalar_triple_product vector_a vector_b vector_c ≠ 0 :=
by
  sorry

end vectors_not_coplanar_l713_713121


namespace cannot_form_B_l713_713444

def Piece : Type := {s : ℕ × ℕ // s.1 * s.2 > 0}
def pieces : List Piece := [{val := (1,1), property := by decide}, {val := (1,1), property := by decide},
                            {val := (1,1), property := by decide}, {val := (2,1), property := by decide},
                            {val := (2,1), property := by decide}, {val := (3,1), property := by decide}]

inductive Figure
| A : Figure    -- 3x2 rectangle fully filled
| B : Figure    -- 3x3 square with one unit missing
| C : Figure    -- Two layers, top layer 3x1 and bottom layer 2x3
| D : Figure    -- 3x3 rectangle missing a corner
| E : Figure    -- 2x5 rectangle

open Figure

theorem cannot_form_B : ¬(Piece → Figure → Prop) pieces B := by
  sorry

end cannot_form_B_l713_713444


namespace min_value_trig_expression_l713_713569

open Real

theorem min_value_trig_expression :
  ∃ x ∈ ℝ, | sin x + cos x + tan x + cot x + sec x + csc x | = 2 * sqrt 2 - 1 := sorry

end min_value_trig_expression_l713_713569


namespace expected_value_of_dodecahedral_die_is_6_l713_713822

noncomputable def expected_value_dodecahedral_die : ℝ :=
  let outcomes := (list.range 12).map (λ x => x + 1) in
  (outcomes.sum / (list.length outcomes : ℝ))

theorem expected_value_of_dodecahedral_die_is_6.5 :
  expected_value_dodecahedral_die = 6.5 :=
sorry

end expected_value_of_dodecahedral_die_is_6_l713_713822


namespace compare_exponents_l713_713932

noncomputable def two := 2.0
noncomputable def exponent1 := -1.1
noncomputable def exponent2 := -1.2

def incrementing_pow_function (base : ℝ) (exp1 exp2 : ℝ) : Prop :=
  base > 1 ∧ exp1 > exp2 → base^exp1 > base^exp2

theorem compare_exponents :
  incrementing_pow_function two exponent1 exponent2 :=
by
  sorry

end compare_exponents_l713_713932


namespace grey_cats_count_l713_713657

theorem grey_cats_count (initial_cats : ℕ) (white_cats_initial : ℕ) (black_cats_fraction : ℚ) (half_black_cats_left : ℕ) (new_white_cats : ℕ) :
  initial_cats = 16 →
  white_cats_initial = 2 →
  black_cats_fraction = 0.25 →
  half_black_cats_left = (initial_cats * black_cats_fraction) / 2 →
  new_white_cats = 2 →
  (let remaining_black_cats := (initial_cats * black_cats_fraction) - half_black_cats_left in
   let total_white_cats := white_cats_initial + new_white_cats in
   let remaining_cats := initial_cats - half_black_cats_left in
   let grey_cats := remaining_cats - total_white_cats - remaining_black_cats in
   grey_cats = 8) :=
begin
  sorry
end

end grey_cats_count_l713_713657


namespace cos_120_degrees_eq_l713_713945

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l713_713945


namespace convert_base_10_to_12_l713_713534

theorem convert_base_10_to_12 : ∀ n : ℕ, n = 173 → base_conversion n 12 = "B5" := by
  intros n h,
  sorry

end convert_base_10_to_12_l713_713534


namespace range_for_a_l713_713178

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x : ℝ) (a : ℝ) : Prop := x > a

-- Definition of what it means for q to be a sufficient but not necessary condition for p
def sufficient_not_necessary (a : ℝ) : Prop :=
  ∀ x, p x → q x a ∧ ∃ x, p x ∧ ¬ q x a

-- The statement expressing the required range for a
theorem range_for_a (a : ℝ) :
  sufficient_not_necessary a → a ∈ set.Ici (1 : ℝ) := 
sorry -- No proof required

end range_for_a_l713_713178


namespace solve_for_x_l713_713238

theorem solve_for_x (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : 6 * x^3 + 18 * x^2 * y * z = 3 * x^4 + 6 * x^3 * y * z) : 
  x = 2 := 
by sorry

end solve_for_x_l713_713238


namespace correct_algorithm_statements_l713_713502

theorem correct_algorithm_statements :
  (∀ (p : Prop), ¬(unique_algorithm p) ∧ algorithm_stops_finite_steps p ∧
  algorithm_steps_clear p ∧ algorithm_produces_definite_result p) :=
by 
  -- Define what it means for an algorithm to solve a problem
  def algorithm (p : Prop) : Prop :=
    ∃ a, solves a p

  -- Statement ①: Unique algorithm for a problem is false.
  def unique_algorithm (p : Prop) : Prop :=
    ¬∃ a1 a2, (solves a1 p ∧ solves a2 p ∧ a1 ≠ a2)

  -- Statement ②: Algorithm stops after finite steps.
  def algorithm_stops_finite_steps (p : Prop) : Prop :=
    ∃ a, solves a p ∧ stops_in_finite_steps a

  -- Statement ③: Each step of an algorithm is clear.
  def algorithm_steps_clear (p : Prop) : Prop :=
    ∃ a, solves a p ∧ clear_steps a

  -- Statement ④: Algorithm produces a definite result.
  def algorithm_produces_definite_result (p : Prop) : Prop :=
    ∃ a, solves a p ∧ produces_definite_result a

  sorry

end correct_algorithm_statements_l713_713502


namespace factorial_division_l713_713520

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_division (n : ℕ) (h : n > 0) : factorial (n) / factorial (n - 1) = n :=
by
  sorry

example : factorial 12 / factorial 11 = 12 :=
by
  exact factorial_division 12 (by norm_num)

end factorial_division_l713_713520


namespace find_line_AB_l713_713633

noncomputable def equation_of_line_AB : Prop :=
  ∀ (x y : ℝ), ((x-2)^2 + (y-1)^2 = 10) ∧ ((x+6)^2 + (y+3)^2 = 50) → (2*x + y = 0)

theorem find_line_AB : equation_of_line_AB := by
  sorry

end find_line_AB_l713_713633


namespace xiaofen_needs_median_to_determine_top_6_l713_713087

theorem xiaofen_needs_median_to_determine_top_6 (scores : List ℝ) (hscores : scores.length = 12) (h_unique : scores.nodup) (x : ℝ) :
  (∃ (med : ℝ), median scores = med ∧ x > med) ↔ (x ∈ (top_n_scores 6 scores)) :=
by
sorry

end xiaofen_needs_median_to_determine_top_6_l713_713087


namespace katya_needs_at_least_ten_l713_713289

variable (x : ℕ)

def prob_katya : ℚ := 4 / 5
def prob_pen  : ℚ := 1 / 2

def expected_correct (x : ℕ) : ℚ := x * prob_katya + (20 - x) * prob_pen

theorem katya_needs_at_least_ten :
  expected_correct x ≥ 13 → x ≥ 10 :=
  by sorry

end katya_needs_at_least_ten_l713_713289


namespace collinear_reflect_points_l713_713586

/-- Given a triangle ABC and points P and Q on its circumcircle, reflecting P with respect to BC yields P_a. 
    The intersection of line QP_a and line BC is denoted as A'. 
    Points B' and C' are constructed analogously. 
    Prove A', B', and C' are collinear. -/
theorem collinear_reflect_points
  (A B C P Q : Type)
  [linear_order A] [linear_order B] [linear_order C] [linear_order P] [linear_order Q]
  (circumcircle : circle (triangle ABC))
  (P_reflected_BC : reflects P BC)
  (Q_line : line Q)
  (A_prime_defined : ∃ A', intersection (line Q P_reflected_BC) (line BC) = A')
  (B_prime_defined : ∃ B', intersection (line Q B_reflected_CA) (line CA) = B')
  (C_prime_defined : ∃ C', intersection (line Q C_reflected_AB) (line AB) = C') :
  collinear {A', B', C'} :=
by
  sorry

end collinear_reflect_points_l713_713586


namespace average_weight_increase_l713_713355

theorem average_weight_increase
  (A : ℝ) -- Average weight of the two persons
  (w1 : ℝ) (h1 : w1 = 65) -- One person's weight is 65 kg 
  (w2 : ℝ) (h2 : w2 = 74) -- The new person's weight is 74 kg
  :
  ((A * 2 - w1 + w2) / 2 - A = 4.5) :=
by
  simp [h1, h2]
  sorry

end average_weight_increase_l713_713355


namespace travel_times_l713_713883

variable (t v1 v2 : ℝ)

def conditions := 
  (v1 * 2 = v2 * t) ∧ 
  (v2 * 4.5 = v1 * t)

theorem travel_times (h : conditions t v1 v2) : 
  t = 3 ∧ 
  (t + 2 = 5) ∧ 
  (t + 4.5 = 7.5) := by
  sorry

end travel_times_l713_713883


namespace find_m_n_l713_713160

theorem find_m_n : 
  ∃ (m : ℤ) (n : ℤ), 
    (0 ≤ m ∧ m ≤ 17) ∧ 
    (0 ≤ n ∧ n ≤ 13) ∧ 
    (m ≡ 98765 [MOD 18]) ∧ 
    (n ≡ 98765 [MOD 14]) ∧ 
    m = 17 ∧ 
    n = 9 := 
by
  sorry

end find_m_n_l713_713160


namespace sum_of_mobile_numbers_l713_713453

theorem sum_of_mobile_numbers : (Finset.range 10).sum = 45 := 
by {
  sorry
}

end sum_of_mobile_numbers_l713_713453


namespace complement_B_intersection_A_complement_B_l713_713632

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := {x | x < 0}
noncomputable def B : Set ℝ := {x | x > 1}

theorem complement_B :
  (U \ B) = {x | x ≤ 1} := by
  sorry

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x | x < 0} := by
  sorry

end complement_B_intersection_A_complement_B_l713_713632


namespace time_to_install_rest_l713_713460

variable (total_windows installed_windows time_per_window : Nat)

-- Conditions
def total_windows := 14
def installed_windows := 5
def time_per_window := 4

-- Goal
theorem time_to_install_rest : 
  (total_windows - installed_windows) * time_per_window = 36 := by
  sorry

end time_to_install_rest_l713_713460


namespace expected_number_of_matches_l713_713914

def xi (i : ℕ) (σ : list ℕ) : ℕ :=
  if σ.nth (i - 1) = some i then 1 else 0

theorem expected_number_of_matches :
  (1 / 720) * ∑ σ in (list.permutations [1, 2, 3, 4, 5, 6]), ∑ i in [1, 2, 3, 4, 5, 6], xi i σ = 1 :=
sorry

end expected_number_of_matches_l713_713914


namespace powers_of_2_not_4_l713_713639

theorem powers_of_2_not_4 (n : ℕ) (n < 500000) :
  card {m | ∃ k, m = 2^k ∧ 0 < m ∧ m < n ∧ ∀ l, m ≠ 4^l} = 9 :=
sorry

end powers_of_2_not_4_l713_713639


namespace third_quartile_l713_713795

theorem third_quartile (s : List ℕ) (h : s = [13, 11, 12, 15, 16, 18, 21, 17]) : 
  let sorted_s := List.sort (<=) s in
  nth_le sorted_s (sorted_s.length * 3 / 4 - 1) <|
by exact sorry :=
  (sorted_s.nth_le (6 - 1) sorry + sorted_s.nth_le (7 - 1) sorry) / 2 = 17.5 :=
by sorry

end third_quartile_l713_713795


namespace betty_wins_strategy_l713_713908

theorem betty_wins_strategy :
  ∃ (strategy : ℕ × ℕ × Bool → ℕ × ℕ × Bool),
    (∀ (s : ℕ × ℕ × Bool), (betty_mirrors_albert_move s strategy)) → 
    wins(strategy) := 
by
  sorry

end betty_wins_strategy_l713_713908


namespace ellipse_focus_coordinates_l713_713010

theorem ellipse_focus_coordinates (a b c : ℝ) (x1 y1 x2 y2 : ℝ) 
  (major_axis_length : 2 * a = 20) 
  (focal_relationship : c^2 = a^2 - b^2)
  (focus1_location : x1 = 3 ∧ y1 = 4) 
  (focus_c_calculation : c = Real.sqrt (x1^2 + y1^2)) :
  (x2 = -3 ∧ y2 = -4) := by
  sorry

end ellipse_focus_coordinates_l713_713010


namespace sum_of_roots_of_P_l713_713785

open Complex Polynomial

noncomputable def P (x : ℂ) : Polynomial ℂ :=
  (X^2 - 4 * cos (π / 6) * X + 4) * (X^2 - 4 * sin (π / 6) * X + 4)

theorem sum_of_roots_of_P :
  ((2 * cos (π / 6) + 2 * Complex.I * sin (π / 6)) +
   (2 * cos (π / 6) - 2 * Complex.I * sin (π / 6)) +
   (2 * sin (π / 6) + 2 * Complex.I * cos (π / 6)) +
   (2 * sin (π / 6) - 2 * Complex.I * cos (π / 6))) = 4 * sqrt 3 + 4 := 
sorry

end sum_of_roots_of_P_l713_713785


namespace renovation_costs_l713_713897

theorem renovation_costs :
  ∃ (x y : ℝ), 
    8 * x + 8 * y = 3520 ∧
    6 * x + 12 * y = 3480 ∧
    x = 300 ∧
    y = 140 ∧
    300 * 12 > 140 * 24 :=
by sorry

end renovation_costs_l713_713897


namespace count_four_digit_integers_l713_713225

theorem count_four_digit_integers (n : ℕ) (h : 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 40 ∧ n % 40 = 0) :
  (finset.filter (λ n, 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 40 ∧ n % 40 = 0) (finset.range 10000)).card = 5 :=
sorry

end count_four_digit_integers_l713_713225


namespace max_factors_of_x12_minus_1_l713_713381

theorem max_factors_of_x12_minus_1 :
  ∃ m (q: fin m → polynomial ℝ), (x^12 - 1 = ∏ i, q i) ∧ (∀ i, q i.degree > 0) ∧ (∀ i, q i ∈ polynomial ℝ) ∧ m = 7 :=
sorry

end max_factors_of_x12_minus_1_l713_713381


namespace find_ratio_l713_713905

def celsius_to_fahrenheit_ratio (ratio : ℝ) (c f : ℝ) : Prop :=
  f = ratio * c + 32

theorem find_ratio (ratio : ℝ) :
  (∀ c f, celsius_to_fahrenheit_ratio ratio c f ∧ ((f = 58) → (c = 14.444444444444445)) → f = 1.8 * c + 32) ∧ 
  (f - 32 = ratio * (c - 0)) ∧
  (c = 14.444444444444445 → f = 32 + 26) ∧
  (f = 58 → c = 14.444444444444445) ∧ 
  (ratio = 1.8)
  → ratio = 1.8 := 
sorry 


end find_ratio_l713_713905


namespace distance_from_origin_to_p_l713_713664

-- Define the two-dimensional points
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance function between two points
def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the origin point and the given point (12, -5)
def origin : Point2D := { x := 0, y := 0 }
def p : Point2D := { x := 12, y := -5 }

-- State the proof problem: the distance from origin to point p is 13
theorem distance_from_origin_to_p : distance origin p = 13 := by
  sorry

end distance_from_origin_to_p_l713_713664


namespace solution_set_f_x_minus_1_l713_713588

def f (x : ℝ) : ℝ := if x > 0 then 2^x - 4 else 4 - 2^(-x)

lemma odd_function (x : ℝ) : f(-x) = -f(x) :=
by sorry

theorem solution_set_f_x_minus_1 : { x : ℝ | f(x-1) > 0 } = { x : ℝ | (-1 < x ∧ x < 1) ∨ (x > 3) } :=
by sorry

end solution_set_f_x_minus_1_l713_713588


namespace tan_half_sum_of_angles_l713_713309

theorem tan_half_sum_of_angles (a b : ℝ) 
  (hcos : cos a + cos b = 1 / 3) 
  (hsin : sin a + sin b = 4 / 13) : 
  Real.tan ((a + b) / 2) = 12 / 13 :=
sorry

end tan_half_sum_of_angles_l713_713309


namespace probability_sum_multiple_of_3_l713_713218
open Finset

theorem probability_sum_multiple_of_3 :
  let S := {1, 2, 3, 4}
  let events := (S.powerset.filter (λ t => t.card = 2))
  let multiples_of_3 := (events.filter (λ t => ((t.to_finset.sum id) % 3) = 0))
in 
  (multiples_of_3.card : ℚ) / events.card = 1 / 3 :=
by
  sorry

end probability_sum_multiple_of_3_l713_713218


namespace extremum_of_f_when_a_eq_e_zeros_of_f_ineq_e_pow_2x_minus_2_l713_713177

open Real

-- Defining the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * log x - a

-- Given conditions
variables {a : ℝ} (ha : 0 < a)

-- Problem Statements

-- (1) Prove that f(x) has a minimum value of 0 when a = e
theorem extremum_of_f_when_a_eq_e : (∀ x : ℝ, f exp(1 : ℝ) x ≥ 0) := sorry

-- (2) If y = f(x) has two zeros x1 and x2 (0 < x1 < x2), 
-- prove that 1/a < x1 < 1 < x2 < a
theorem zeros_of_f (hzeros : ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ f a x1 = 0 ∧ f a x2 = 0) : 
  ∃ x1 x2 : ℝ, (1 / a) < x1 ∧ x1 < 1 ∧ 1 < x2 ∧ x2 < a := sorry

-- (3) Prove that e^(2x - 2) - e^(x - 1) log x - x ≥ 0
theorem ineq_e_pow_2x_minus_2 (x : ℝ) : 
  exp (2 * x - 2) - exp (x - 1) * log x - x ≥ 0 := sorry

end extremum_of_f_when_a_eq_e_zeros_of_f_ineq_e_pow_2x_minus_2_l713_713177


namespace option_a_option_b_option_c_l713_713591

open Real

-- Define the functions f and g
variable {f g : ℝ → ℝ}

-- Given conditions
axiom cond1 : ∀ x : ℝ, f(x + 3) = g(-x) + 4
axiom cond2 : ∀ x : ℝ, deriv f x + deriv g (1 + x) = 0
axiom cond3 : ∀ x : ℝ, g(2*x + 1) = g(-(2*x) + 1)

-- Prove the statements
theorem option_a : deriv g 1 = 0 :=
sorry

theorem option_b : ∀ x : ℝ, f(x + 4) = f(4 - x) :=
sorry

theorem option_c : ∀ x : ℝ, deriv f (x + 1) = deriv f (1 - x) :=
sorry

end option_a_option_b_option_c_l713_713591


namespace original_pencil_count_l713_713915

-- Defining relevant constants and assumptions based on the problem conditions
def pencilsRemoved : ℕ := 4
def pencilsLeft : ℕ := 83

-- Theorem to prove the original number of pencils is 87
theorem original_pencil_count : pencilsLeft + pencilsRemoved = 87 := by
  sorry

end original_pencil_count_l713_713915


namespace bears_total_l713_713173

-- Define the number of each type of bear
def brown_bears : ℕ := 15
def white_bears : ℕ := 24
def black_bears : ℕ := 27
def polar_bears : ℕ := 12
def grizzly_bears : ℕ := 18

-- Define the total number of bears
def total_bears : ℕ := brown_bears + white_bears + black_bears + polar_bears + grizzly_bears

-- The theorem stating the total number of bears is 96
theorem bears_total : total_bears = 96 :=
by
  -- The proof is omitted here
  sorry

end bears_total_l713_713173


namespace region_division_by_lines_l713_713532

-- Define the plane and lines
def line1 (x : ℝ) : ℝ := 3 * x
def line2 (y : ℝ) : ℝ := (1/3) * y

-- Define the problem statement
theorem region_division_by_lines :
  ∃ regions : ℕ , regions = 4 :=
sorry

end region_division_by_lines_l713_713532


namespace exists_infinite_subset_B_l713_713697

-- Define that P is a finite set of primes
variable (P : Set ℕ) [finite P]

-- Define that A is an infinite set of positive integers
variable (A : Set ℕ) [infinite A]

-- Define the condition that every element of A has a prime factor not in P
def has_prime_factor_not_in_P (n : ℕ) : Prop :=
  ∃ p, p.prime ∧ p ∣ n ∧ p ∉ P

variable (hA : ∀ a ∈ A, has_prime_factor_not_in_P P a)

theorem exists_infinite_subset_B :
  ∃ B ⊆ A, infinite B ∧ ∀ (F : Finset ℕ), (↑F ⊆ B) → has_prime_factor_not_in_P P (F.sum id) :=
sorry

end exists_infinite_subset_B_l713_713697


namespace cube_sequence_count_l713_713872

theorem cube_sequence_count : 
  let numbers := {1, 2, 3, 4, 5, 6},
      opposites : (ℕ × ℕ) := {(1, 6), (2, 5), (3, 4)},
      initial_down := 6,
      n := 6 in 
  (∀ seq : list ℕ, seq.length = n ∧ ∀ x ∈ numbers, x ∈ seq 
    ∧ (∀ i ∈ [1, 2, 3, 4, 5], seq.nth_le i sorry ≠ opposites.prod.snd)
    → count_seq := card {seq | seq ∈ permutations numbers ∧ satisfies_conditions seq opposites initial_down})
    ∧ count_seq = 40.

end cube_sequence_count_l713_713872


namespace katya_minimum_problems_l713_713284

-- Defining the conditions
def katya_probability_solve : ℚ := 4 / 5
def pen_probability_solve : ℚ := 1 / 2
def total_problems : ℕ := 20
def minimum_correct_for_good_grade : ℚ := 13

-- The Lean statement to prove
theorem katya_minimum_problems (x : ℕ) :
  x * katya_probability_solve + (total_problems - x) * pen_probability_solve ≥ minimum_correct_for_good_grade → x ≥ 10 :=
sorry

end katya_minimum_problems_l713_713284


namespace simplify_fraction_expression_l713_713068

variable (d : ℤ)

theorem simplify_fraction_expression : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := by
  sorry

end simplify_fraction_expression_l713_713068


namespace number_of_real_roots_of_cos_eq_sin_plus_x_l713_713782

theorem number_of_real_roots_of_cos_eq_sin_plus_x :
  ∃! x ∈ ℝ, cos x = x + sin x :=
sorry

end number_of_real_roots_of_cos_eq_sin_plus_x_l713_713782


namespace find_tan_alpha_find_alpha_plus_2gamma_l713_713581

noncomputable theory
open_locale real

-- Definitions for the given conditions
def tan_alpha_plus_beta := 9 / 13
def tan_beta_minus_pi_over_4 := -1 / 3
def cos_gamma := (3 * sqrt 10) / 10
def is_acute (x : ℝ) : Prop := 0 < x ∧ x < π / 2

-- Variables representing the angles
variables (α β γ : ℝ)

-- Conditions stating the given tangents and cosine, and that α and γ are acute
axiom tan_alpha_plus_beta_ax : tan (α + β) = tan_alpha_plus_beta
axiom tan_beta_minus_pi_over_4_ax : tan (β - π / 4) = tan_beta_minus_pi_over_4
axiom cos_gamma_ax : cos γ = cos_gamma
axiom alpha_aqute_ax : is_acute α
axiom gamma_acute_ax : is_acute γ

-- Proofs to be provided
theorem find_tan_alpha : tan α = 1 / 7 :=
by sorry

theorem find_alpha_plus_2gamma : α + 2 * γ = π / 4 :=
by sorry

end find_tan_alpha_find_alpha_plus_2gamma_l713_713581


namespace shaded_triangle_probability_l713_713680

theorem shaded_triangle_probability :
  ∃ (triangles : Finset ℕ) (shaded_triangles : Finset ℕ),
    3 ≤ triangles.card ∧  -- There are more than three triangles
    (∀ t ∈ triangles, t ∈ shaded_triangles) ∨ (∀ t ∈ triangles, t ∉ shaded_triangles) ∧ -- Each triangle is either shaded or not shaded
    triangles.card = 5 ∧  -- The total number of triangles is 5
    shaded_triangles.card = 3 ∧ -- The number of shaded triangles is 3
    ∀ (t ∈ triangles), t ∈ shaded_triangles → (shaded_triangles.card : ℚ) / (triangles.card : ℚ) = 3 / 5 := -- Probability calculation
by
  sorry

end shaded_triangle_probability_l713_713680


namespace george_score_l713_713732

theorem george_score (avg_without_george avg_with_george : ℕ) (num_students : ℕ) 
(h1 : avg_without_george = 75) (h2 : avg_with_george = 76) (h3 : num_students = 20) :
  (num_students * avg_with_george) - ((num_students - 1) * avg_without_george) = 95 :=
by 
  sorry

end george_score_l713_713732


namespace math_problem_l713_713594

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def g' : ℝ → ℝ := sorry

def condition1 (x : ℝ) : Prop := f (x + 3) = g (-x) + 4
def condition2 (x : ℝ) : Prop := f' x + g' (1 + x) = 0
def even_function (x : ℝ) : Prop := g (2 * x + 1) = g (- (2 * x + 1))

theorem math_problem (x : ℝ) :
  (∀ x, condition1 x) →
  (∀ x, condition2 x) →
  (∀ x, even_function x) →
  (g' 1 = 0) ∧
  (∀ x, f (1 - x) = f (x + 3)) ∧
  (∀ x, f' x = f' (-x + 2)) :=
by
  intros
  sorry

end math_problem_l713_713594


namespace complement_of_A_l713_713231

theorem complement_of_A :
  let A := {x : ℝ | 3 * x - 7 > 0}
  let complement_A := {x : ℝ | x ≤ 7 / 3}
  (∀ x, x ∈ (set.univ \ A) ↔ x ∈ complement_A) :=
begin
  sorry
end

end complement_of_A_l713_713231


namespace rowing_distance_l713_713879

theorem rowing_distance
  (v_still : ℝ) (v_current : ℝ) (time : ℝ)
  (h1 : v_still = 15) (h2 : v_current = 3) (h3 : time = 17.998560115190784) :
  (v_still + v_current) * 1000 / 3600 * time = 89.99280057595392 :=
by
  rw [h1, h2, h3] -- Apply the given conditions
  -- This will reduce to proving (15 + 3) * 1000 / 3600 * 17.998560115190784 = 89.99280057595392
  sorry

end rowing_distance_l713_713879


namespace shoveling_driveway_time_l713_713067

theorem shoveling_driveway_time (S : ℝ) (Wayne_rate : ℝ) (combined_rate : ℝ) :
  (S = 1 / 7) → (Wayne_rate = 6 * S) → (combined_rate = Wayne_rate + S) → (combined_rate = 1) :=
by { sorry }

end shoveling_driveway_time_l713_713067


namespace not_a_factorization_method_l713_713834

def factorization_methods : Set String := 
  {"Taking out the common factor", "Cross multiplication method", "Formula method", "Group factorization"}

theorem not_a_factorization_method : 
  ¬ ("Addition and subtraction elimination method" ∈ factorization_methods) :=
sorry

end not_a_factorization_method_l713_713834


namespace principal_is_1000_l713_713449

noncomputable def findPrincipal (A : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  A / (1 + (R * T))

theorem principal_is_1000 (h : findPrincipal 1120 0.05 2.4 = 1000) : true :=
by {
  trivial,
}

end principal_is_1000_l713_713449


namespace gcd_108_45_l713_713044

def gcd_euclidean (a b : Nat) : Nat :=
  if b = 0 then a else gcd_euclidean b (a % b)

theorem gcd_108_45 : gcd_euclidean 108 45 = 9 := by
  sorry

end gcd_108_45_l713_713044


namespace cosine_120_eq_negative_half_l713_713960

theorem cosine_120_eq_negative_half :
  let θ := 120 * Real.pi / 180
  let point := (Real.cos θ, Real.sin θ)
  point.1 = -1 / 2 := by
  sorry

end cosine_120_eq_negative_half_l713_713960


namespace odd_decreasing_values_count_l713_713529

def f (α : ℝ) (x : ℝ) := x ^ α

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g (x)

def is_monotonically_decreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → g y < g x

theorem odd_decreasing_values_count :
  {α ∈ {-2, -1, -1/2, 1/2, 1, 2} | is_odd_function (f α) ∧ is_monotonically_decreasing (f α)}.card = 1 := 
by
  sorry

end odd_decreasing_values_count_l713_713529


namespace nickels_remaining_l713_713730

variable (original_nickels borrowed_nickels remaining_nickels : ℕ)

def mike_originally_had : original_nickels = 87 := sorry
def mike_dad_borrowed : borrowed_nickels = 75 := sorry

theorem nickels_remaining (h1 : original_nickels = 87) (h2 : borrowed_nickels = 75) :
  remaining_nickels = original_nickels - borrowed_nickels :=
begin
  rw [h1, h2],
  norm_num,
end

end nickels_remaining_l713_713730


namespace coefficient_d_nonzero_l713_713013

theorem coefficient_d_nonzero (a b c d f p q r : ℝ) (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r)
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) (h_not_one : p ≠ 1 ∧ q ≠ 1 ∧ r ≠ 1)
  (h_intercepts : (x : ℝ) → x * (x - 1) * (x - p) * (x - q) * (x - r) = 0 → x = 0 ∨ x = 1 ∨ x = p ∨ x = q ∨ x = r)
  (h_q0 : Q(0) = 0) (h_q1 : Q(1) = 0) : d ≠ 0 := by
  let Q : ℝ → ℝ := λ x, x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + f
  have h_f : f = 0 := sorry
  have h_sum : a + b + c + d = -1 := sorry
  sorry

end coefficient_d_nonzero_l713_713013


namespace ellipse_focus_on_y_axis_l713_713611

theorem ellipse_focus_on_y_axis (k : ℝ) : 1 < k ∧ k < 2 ↔ 
  (∀ x y : ℝ, (0 < 2 - k) ∧ (0 < 2k - 1) ∧ (2k - 1 > 2 - k) 
  → (x^2 / (2 - k) + y^2 / (2k - 1) = 1 → ∃ a b : ℝ, (a > 0) ∧ (b > 0) ∧
  (x^2 / a^2 + y^2 / b^2 = 1 ∧ (b^2 - a^2 > 0))) ) :=
sorry

end ellipse_focus_on_y_axis_l713_713611


namespace total_amount_spent_l713_713738

theorem total_amount_spent (half_dollar_value : ℝ) (wednesday_spend : ℕ) (thursday_spend : ℕ) : 
  wednesday_spend = 4 → thursday_spend = 14 → half_dollar_value = 0.5 → (wednesday_spend + thursday_spend) * half_dollar_value = 9 :=
by
  intros wednesday_cond thursday_cond half_dollar_cond
  rw [wednesday_cond, thursday_cond, half_dollar_cond]
  norm_num
  sorry

end total_amount_spent_l713_713738


namespace convex_quadrilateral_AE_eq_BF_l713_713583

noncomputable theory
open_locale classical

variables {k : Type*} [euclidean_geometry k]

theorem convex_quadrilateral_AE_eq_BF 
  (P Q R S O E F A B : k)
  (W W1 W2 : circle k)
  (hPQRS_inscribed : W.inscribed (P, Q, R, S))
  (hW1_circumcircle_PQS : W1.circumcircle (P, Q, S))
  (hW2_circumcircle_QOR : W2.circumcircle (Q, O, R))
  (hO_intersection_PR_QS : intersects_at (lineseg P R) (lineseg Q S) O)
  (hEF_through_O : line_through O E F)
  (hEF_intersects_W1_at_E : E ∈ W1 ∧ line_through O E = EF)
  (hEF_intersects_W2_at_F : F ∈ W2 ∧ line_through O F = EF)
  (hEF_intersects_W_at_A_B : A ∈ W ∧ B ∈ W ∧ line_through O A = line_through O B):
  dist A E = dist B F :=
sorry

end convex_quadrilateral_AE_eq_BF_l713_713583


namespace max_min_distance_correct_l713_713259

noncomputable def max_min_distance : ℝ × ℝ :=
let l := {p : ℝ × ℝ | p.1 + p.2 = 4} in
let P (θ : ℝ) := (sqrt 3 * cos θ, sin θ) in
let distance (P : ℝ × ℝ) := (|P.1 + P.2 - 4| / sqrt 2) in
(max distance⟨P θ⟩ for all θ), (min distance⟨P θ⟩ for all θ)

theorem max_min_distance_correct : max_min_distance = (3 * sqrt 2, sqrt 2) :=
sorry

end max_min_distance_correct_l713_713259


namespace rope_cut_prob_l713_713102

theorem rope_cut_prob (x : ℝ) (hx : 0 < x) : 
  (∃ (a b : ℝ), a + b = 1 ∧ min a b ≤ max a b / x) → 
  (1 / (x + 1) * 2) = 2 / (x + 1) :=
sorry

end rope_cut_prob_l713_713102


namespace train_initial_speed_l713_713903

theorem train_initial_speed (x : ℝ) (h : 3 * 25 * (x / V + (2 * x / 20)) = 3 * x) : V = 50 :=
  by
  sorry

end train_initial_speed_l713_713903


namespace gen_terms_a_gen_terms_b_sum_first_n_terms_range_of_k_l713_713607

-- Define the sequences {a_n}
def S (n : ℕ) : ℚ := 2 * a n - 1/2
def a : ℕ → ℚ
| 0     := 1/2 -- Note: n \in \mathbb{N^*} implies n starts from 1
| (n+1) := 2 * a n

-- Define the sequences {b_n}
def b : ℕ → ℚ
| 0     := 1  -- given b_1 = 1
| (n+1) := 2 * b n + 1 -- derived from the problem

-- Define the sequences {c_n}
def c (n : ℕ) : ℚ := (a n) * (b n)

-- Sum of the first n terms of {c_n}
def T (n : ℕ) : ℚ := (finset.range n).sum c

-- Range of k for the given inequalities
def k_range (λ : ℚ) : set ℚ := { k : ℚ | k < 2 * λ + 1 / λ }

theorem gen_terms_a (n : ℕ) (n_pos : 0 < n) :
  a n = 2^(n-2) := sorry

theorem gen_terms_b (n : ℕ) :
  b n = 2 * n - 1 := sorry

theorem sum_first_n_terms (n : ℕ) :
  T n = 3/2 + (2 * n - 3) * 2^(n-1) := sorry

theorem range_of_k (λ : ℚ) (h : 0 < λ) :
  ∀ n, 2 * λ^2 - k * λ + 2 > b n / (a (2 * n)) → k < 2 * λ + 1 / λ := sorry

end gen_terms_a_gen_terms_b_sum_first_n_terms_range_of_k_l713_713607


namespace shop_makes_off_each_jersey_l713_713351

theorem shop_makes_off_each_jersey :
  ∀ (T : ℝ) (jersey_earnings : ℝ),
  (T = 25) →
  (jersey_earnings = T + 90) →
  jersey_earnings = 115 := by
  intros T jersey_earnings ht hj
  sorry

end shop_makes_off_each_jersey_l713_713351


namespace discount_difference_l713_713477

theorem discount_difference (p : ℝ) (single_discount first_discount second_discount : ℝ) :
    p = 12000 →
    single_discount = 0.45 →
    first_discount = 0.35 →
    second_discount = 0.10 →
    (p * (1 - single_discount) - p * (1 - first_discount) * (1 - second_discount) = 420) := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end discount_difference_l713_713477


namespace value_of_expression_l713_713832

theorem value_of_expression :
  let n := 5.5 in
  let x := n / 6 in
  let y := x * 12 in
  y = 11 :=
by
  sorry

end value_of_expression_l713_713832


namespace independent_events_l713_713232

open ProbabilityTheory

variable {Ω : Type*} {P : Measure Ω}

def A : Set Ω := sorry
def B : Set Ω := sorry

noncomputable def P_A : ℝ := 1 - 2/3
noncomputable def P_B : ℝ := 1/3
noncomputable def P_AB : ℝ := 1/9

theorem independent_events :
  P (A ∩ B) = P A * P B :=
by
  have hA : P A = 1 - P (Aᶜ), by sorry
  sorry

end independent_events_l713_713232


namespace angle_between_plane_S_and_base_l713_713028

theorem angle_between_plane_S_and_base (ABCD : Type) [Fintype ABCD] (V : ℝ) :
  (base_is_square : ∀ (A B C D : ABCD), true) →
  (lateral_faces_are_equilateral : ∀ (A B C E : ABCD), true) →
  (plane_S_passes_through_AB : ∀ (S : Set ABCD), true) →
  (plane_S_divides_volume : ∀ (S : Set ABCD), volume S = V/2) →
  angle_between_S_and_base = 18.4625 :=
by
  sorry

end angle_between_plane_S_and_base_l713_713028


namespace area_of_square_with_given_y_coords_l713_713896

-- Define the conditions: the vertices of the square in terms of their y-coordinates.
def square_has_given_y_coordinates (a b c d : ℝ) (square : set (ℝ × ℝ)) :=
  ∃ x1 x2 x3 x4 : ℝ, 
    ((x1, 2) ∈ square) ∧
    ((x2, 3) ∈ square) ∧
    ((x3, 8) ∈ square) ∧
    ((x4, 7) ∈ square) ∧
    (is_square square)

-- Prove that the area of a square with y-coordinates 2, 3, 7, and 8 is 25.
theorem area_of_square_with_given_y_coords : ∀ (square : set (ℝ × ℝ)), 
  square_has_given_y_coordinates 2 3 7 8 square →
  ∃ side : ℝ, side * side = 25 :=
by
  intro square h,
  sorry

end area_of_square_with_given_y_coords_l713_713896


namespace value_of_m_l713_713083

theorem value_of_m :
  ∃ m : ℕ, 5 ^ m = 5 * 25 ^ 2 * 125 ^ 3 ∧ m = 14 :=
begin
  -- Solution steps would be here
  sorry
end

end value_of_m_l713_713083


namespace max_sum_squares_and_products_l713_713798

theorem max_sum_squares_and_products
  (f g h j : ℕ)
  (h1 : f + g + h + j = 20)
  (h2 : f ∈ {2, 4, 6, 8})
  (h3 : g ∈ {2, 4, 6, 8})
  (h4 : h ∈ {2, 4, 6, 8})
  (h5 : j ∈ {2, 4, 6, 8})
  : (f^2 + g^2 + h^2 + j^2 = 120) ∧ (f * g + g * h + h * j + f * j = 100) :=
begin
  sorry
end

end max_sum_squares_and_products_l713_713798


namespace basketball_volleyball_problem_l713_713476

-- Define variables and conditions
variables (x y : ℕ) (m : ℕ)

-- Conditions
def price_conditions : Prop :=
  2 * x + 3 * y = 190 ∧ 3 * x = 5 * y

def price_solutions : Prop :=
  x = 50 ∧ y = 30

def purchase_conditions : Prop :=
  8 ≤ m ∧ m ≤ 10 ∧ 50 * m + 30 * (20 - m) ≤ 800

-- The most cost-effective plan
def cost_efficient_plan : Prop :=
  m = 8 ∧ (20 - m) = 12

-- Conjecture for the problem
theorem basketball_volleyball_problem :
  price_conditions x y ∧ purchase_conditions m →
  price_solutions x y ∧ cost_efficient_plan m :=
by {
  sorry
}

end basketball_volleyball_problem_l713_713476


namespace tangent_line_at_origin_l713_713158

/-- 
The curve is given by y = exp x.
The tangent line to this curve that passes through the origin (0, 0) 
has the equation y = exp 1 * x.
-/
theorem tangent_line_at_origin :
  ∀ (x y : ℝ), y = Real.exp x → (∃ k : ℝ, ∀ x, y = k * x ∧ k = Real.exp 1) :=
by
  sorry

end tangent_line_at_origin_l713_713158


namespace largest_n_crates_same_orange_count_l713_713112

theorem largest_n_crates_same_orange_count :
  ∀ (num_crates : ℕ) (min_oranges max_oranges : ℕ),
    num_crates = 200 →
    min_oranges = 100 →
    max_oranges = 130 →
    (∃ (n : ℕ), n = 7 ∧ (∃ (distribution : ℕ → ℕ), 
      (∀ x, min_oranges ≤ x ∧ x ≤ max_oranges) ∧ 
      (∀ x, distribution x ≤ num_crates ∧ 
          ∃ y, distribution y ≥ n))) := sorry

end largest_n_crates_same_orange_count_l713_713112


namespace parabolas_intersect_with_high_probability_l713_713811

noncomputable def high_probability_of_intersection : Prop :=
  ∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 1 ≤ d ∧ d ≤ 6 →
  (a - c) ^ 2 + 4 * (b - d) >= 0

theorem parabolas_intersect_with_high_probability : high_probability_of_intersection := sorry

end parabolas_intersect_with_high_probability_l713_713811


namespace washing_machine_time_l713_713498

theorem washing_machine_time (shirts pants sweaters jeans : ℕ) (max_items_per_cycle minutes_per_cycle : ℕ)
    (h_shirts : shirts = 18) (h_pants : pants = 12) (h_sweaters : sweaters = 17) (h_jeans : jeans = 13)
    (h_max_items_per_cycle : max_items_per_cycle = 15) (h_minutes_per_cycle : minutes_per_cycle = 45) :
    (shirts + pants + sweaters + jeans) / max_items_per_cycle * minutes_per_cycle / 60 = 3 := by
  -- Total number of items calculation
  have total_items : shirts + pants + sweaters + jeans = 18 + 12 + 17 + 13 := by
    rw [h_shirts, h_pants, h_sweaters, h_jeans]
    norm_num
    
  -- Number of cycles calculation
  have cycles : (shirts + pants + sweaters + jeans) / max_items_per_cycle = 60 / 15 := by
    rw [total_items, h_max_items_per_cycle]
    norm_num

  -- Total time in minutes calculation
  have total_time_minutes : (shirts + pants + sweaters + jeans) / max_items_per_cycle * minutes_per_cycle = 4 * 45 := by
    rw [cycles, h_minutes_per_cycle]
    norm_num

  -- Time conversion from minutes to hours
  show 4 * 45 / 60 = 3
  norm_num

end washing_machine_time_l713_713498


namespace circle_tangent_to_circumcircle_l713_713683

theorem circle_tangent_to_circumcircle 
  (A B C E F O G : Type) [is_midpoint E A C] [is_midpoint F B C] [is_centroid G A B C] 
  (h1 : cyclic (set.insert E (set.insert F (set.insert O (set.insert C {G}))))):
  tangent (circumcircle A B C) (circle E F O C G) :=
sorry

end circle_tangent_to_circumcircle_l713_713683


namespace measure_B_proof_a_plus_c_proof_l713_713654

def triangle_condition (a b c A B C: ℝ) : Prop :=
  ∃ (a b c A B C : ℝ), (sqrt 3) * b * (Finset.sin A) - a * (Finset.cos B) = 2 * a

def triangle_area (a b c B : ℝ) : Prop :=
  ∃ (b: ℝ), b = 7 ∧ (15 * sqrt 3) / 4 = (1 / 2) * a * c * (sin B)

theorem measure_B_proof (a b c A B C : ℝ) (h : triangle_condition a b c A B C) : B = 2 * π / 3 := 
by
  sorry

theorem a_plus_c_proof (a b c B : ℝ) (h_area : triangle_area a b c B) : a + c = 8 := 
by
  sorry

end measure_B_proof_a_plus_c_proof_l713_713654


namespace subset_N_M_l713_713724

def M : Set ℝ := { x | ∃ (k : ℤ), x = k / 2 + 1 / 3 }
def N : Set ℝ := { x | ∃ (k : ℤ), x = k + 1 / 3 }

theorem subset_N_M : N ⊆ M := 
  sorry

end subset_N_M_l713_713724


namespace pennies_for_washing_clothes_l713_713757

theorem pennies_for_washing_clothes (total_money_cents : ℕ) (num_quarters : ℕ) (value_quarter_cents : ℕ) :
  total_money_cents = 184 → num_quarters = 7 → value_quarter_cents = 25 → (total_money_cents - num_quarters * value_quarter_cents) = 9 :=
by
  intros htm hq hvq
  rw [htm, hq, hvq]
  linarith

end pennies_for_washing_clothes_l713_713757


namespace sum_of_f_values_l713_713579

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem sum_of_f_values :
  (∑ k in (Finset.range 2017).map (λ n, if n = 0 then 1 else n) (\lambda n, f (1 / (n : ℝ)) + f (n : ℝ)) + f 1) = 4031 / 2 := 
by
  sorry

end sum_of_f_values_l713_713579


namespace factorial_ratio_l713_713518

theorem factorial_ratio : 12! / 11! = 12 := by
  sorry

end factorial_ratio_l713_713518


namespace ending_number_divisible_by_six_l713_713393

theorem ending_number_divisible_by_six (first_term : ℕ) (n : ℕ) (common_difference : ℕ) (sequence_length : ℕ) 
  (start : first_term = 12) 
  (diff : common_difference = 6)
  (num_terms : sequence_length = 11) :
  first_term + (sequence_length - 1) * common_difference = 72 := by
  sorry

end ending_number_divisible_by_six_l713_713393


namespace product_EP_EF_l713_713868

open EuclideanGeometry

-- Define the circle and the given dimensions
variables {O A B C D E F P : Point}
variables {r : ℝ} -- radius of the circle
variables {AB CD CP EP EF : ℝ} -- lengths involved

-- Introduce the given conditions
def circle_with_diameter_and_chord_intersection : Prop :=
  (AB = 16) ∧
  (CD = 20) ∧
  (is_perpendicular AB CD) ∧
  (CP = 6)

-- The theorem to prove the product of EP and EF
theorem product_EP_EF : circle_with_diameter_and_chord_intersection → EP * EF = 80 :=
by
  intros,
  sorry

end product_EP_EF_l713_713868


namespace sam_pennies_l713_713755

def pennies_from_washing_clothes (total_money_cents : ℤ) (quarters : ℤ) : ℤ :=
  total_money_cents - (quarters * 25)

theorem sam_pennies :
  pennies_from_washing_clothes 184 7 = 9 :=
by
  sorry

end sam_pennies_l713_713755


namespace circle_area_x2_y2_eq_102_l713_713415

theorem circle_area_x2_y2_eq_102 :
  ∀ (x y : ℝ), (x + 9)^2 + (y - 3)^2 = 102 → π * 102 = 102 * π :=
by
  intros
  sorry

end circle_area_x2_y2_eq_102_l713_713415


namespace katya_needs_at_least_ten_l713_713287

variable (x : ℕ)

def prob_katya : ℚ := 4 / 5
def prob_pen  : ℚ := 1 / 2

def expected_correct (x : ℕ) : ℚ := x * prob_katya + (20 - x) * prob_pen

theorem katya_needs_at_least_ten :
  expected_correct x ≥ 13 → x ≥ 10 :=
  by sorry

end katya_needs_at_least_ten_l713_713287


namespace better_value_ounces_l713_713334

-- Definitions of the conditions
def largerBoxWeight : ℕ := 30
def largerBoxCost : ℝ := 4.80
def smallerBoxWeight : ℕ := 20
def smallerBoxCost : ℝ := 3.40

-- Definition to convert dollars to cents
def toCents (dollars : ℝ) : ℕ := (dollars * 100).round.to_nat

-- Proof statement (with 'sorry' indicating the proof is not provided)
theorem better_value_ounces :
  let largerPricePerOunce := largerBoxCost / largerBoxWeight
  let smallerPricePerOunce := smallerBoxCost / smallerBoxWeight
  let largerPricePerOunceInCents := toCents largerPricePerOunce
  let smallerPricePerOunceInCents := toCents smallerPricePerOunce in
  largerPricePerOunceInCents = 16 ∧ largerPricePerOunceInCents < smallerPricePerOunceInCents :=
by
  sorry

end better_value_ounces_l713_713334


namespace eleventh_number_in_list_l713_713910

def digits_sum_to_11 (n : ℕ) : Prop :=
  (digits 10 n).sum = 11

theorem eleventh_number_in_list :
  ∃ (a : ℕ), digits_sum_to_11 a ∧ ∀ n, n < 11 → ¬digits_sum_to_11 n →
  ∃ (b: ℕ), digits_sum_to_11 b ∧ list.nth (list.filter digits_sum_to_11 (list.range (b + 1))) 10 = some a :=
sorry

end eleventh_number_in_list_l713_713910


namespace distance_P1_P2_l713_713634

-- Definitions of the points
def P1 : ℝ × ℝ × ℝ := (-1, 3, 2)
def P2 : ℝ × ℝ × ℝ := (2, 4, -1)

-- Distance formula between two points in 3D space
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

-- The theorem we need to prove
theorem distance_P1_P2 : distance P1 P2 = real.sqrt 19 := by
  sorry

end distance_P1_P2_l713_713634


namespace a_increasing_l713_713026

noncomputable def a : ℕ → ℝ
| 0     := 1 / 5
| (n+1) := 2^n - 3 * a n

theorem a_increasing (n : ℕ) : a (n + 1) > a n := 
by {
  sorry
}

end a_increasing_l713_713026


namespace rotated_log_eq_inv_exp_l713_713364

theorem rotated_log_eq_inv_exp (x : ℝ) (hx : 0 < x) :
  (∃ y : ℝ, y = 10⁻ˣ ∧ y > 0) ↔ ∃ x, x = 10⁻ˣ :=
sorry

end rotated_log_eq_inv_exp_l713_713364


namespace parabola_vertex_l713_713773

theorem parabola_vertex :
  ∃ h k : ℝ, (∀ x y : ℝ, y^2 + 8*y + 4*x + 9 = 0 → x = -1/4 * (y + 4)^2 + 7/4)
  := 
  ⟨7/4, -4, sorry⟩

end parabola_vertex_l713_713773


namespace cos_120_degrees_l713_713995

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l713_713995


namespace smallest_x_satisfying_expression_l713_713059

theorem smallest_x_satisfying_expression :
  ∃ x : ℤ, (∃ k : ℤ, x^2 + x + 7 = k * (x - 2)) ∧ (∀ y : ℤ, (∃ k' : ℤ, y^2 + y + 7 = k' * (y - 2)) → y ≥ x) ∧ x = -11 :=
by
  sorry

end smallest_x_satisfying_expression_l713_713059


namespace cos_120_eq_neg_half_l713_713971

theorem cos_120_eq_neg_half : 
  let P : (ℝ × ℝ) := (-1 / 2, Real.sqrt 3 / 2)
  in (∃θ : ℝ, θ = 120 * (π / 180) ∧
              ∃P : (ℝ × ℝ), P = (-1 / 2, Real.sqrt 3 / 2) ∧
              cos θ = P.1) :=
by
  sorry

end cos_120_eq_neg_half_l713_713971


namespace number_of_solutions_l713_713544

-- Define the function of interest
def equation (x : ℝ) : ℝ := 3 * (Real.cos x)^3 - 9 * (Real.cos x)^2 + 6 * Real.cos x

-- Define the interval of interest
def interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2 * Real.pi

-- Define the property that x is a solution of the equation within the interval
def is_solution (x : ℝ) : Prop := equation x = 0 ∧ interval x

-- The main statement to prove
theorem number_of_solutions : { x : ℝ | is_solution x }.toFinset.card = 4 := 
sorry

end number_of_solutions_l713_713544


namespace factorial_ratio_l713_713516

theorem factorial_ratio : 12! / 11! = 12 := by
  sorry

end factorial_ratio_l713_713516


namespace cos_120_degrees_eq_l713_713952

-- Define the problem conditions:
def angle_120_degrees := 120 * (Real.pi / 180) -- Convert to radians
def point_on_unit_circle := Real.cos angle_120_degrees

-- Theorem statement
theorem cos_120_degrees_eq : point_on_unit_circle = - (Real.sqrt 3) / 2 :=
by sorry

end cos_120_degrees_eq_l713_713952


namespace sum_fraction_145_repeating_l713_713146

theorem sum_fraction_145_repeating : 
    let x := 0.145145145145...
    let num := 145
    let den := 999
    let fract := num / den
    let gcd_num_den := 1
    ∀ (num denom : ℕ), (fract = num / denom) → gcd num denom = 1 → (num + denom) = 1144 := 
by sorry

end sum_fraction_145_repeating_l713_713146


namespace sum_of_coefficients_l713_713366

theorem sum_of_coefficients 
  (A B C : ℤ)
  (h : ∀ x : ℂ, x ∈ {(-1 : ℂ), 3, 4} → x^3 + (A : ℂ) * x^2 + (B : ℂ) * x + (C : ℂ) = 0) : 
  A + B + C = 11 := 
sorry

end sum_of_coefficients_l713_713366


namespace find_b_for_real_root_l713_713561

noncomputable def polynomial_has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^4 + b * x^3 - 2 * x^2 + b * x + 2 = 0

theorem find_b_for_real_root :
  ∀ b : ℝ, polynomial_has_real_root b → b ≤ 0 := by
  sorry

end find_b_for_real_root_l713_713561


namespace cos_120_eq_neg_half_l713_713984

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l713_713984


namespace central_angle_of_unfolded_cone_l713_713854

theorem central_angle_of_unfolded_cone
  (r : ℝ) 
  (h1 : r > 0)
  (l : ℝ) 
  (h2 : l = real.sqrt 2 * r) :
  ∃ (α : ℝ), α = real.sqrt 2 * real.pi :=
by 
  sorry

end central_angle_of_unfolded_cone_l713_713854


namespace cos_120_degrees_l713_713996

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l713_713996


namespace cos_120_eq_neg_half_l713_713983

theorem cos_120_eq_neg_half : 
  let Q := (rotate (real.pi * (2/3))) (1, 0)
  in Q.1 = -1/2 := 
by
  sorry

end cos_120_eq_neg_half_l713_713983


namespace sum_of_distance_fourth_power_constant_l713_713316

-- Given definitions for a regular n-sided polygon inscribed in a circle
def regular_polygon_inscribed_circle (n : ℕ) (r : ℝ) : Prop :=
  ∃ (P : ℕ → ℂ), (∀ k, P k = r * (complex.exp (2 * complex.pi * complex.I * (k / n))) ∧ (∀ i j, i ≠ j → P i ≠ P j))

-- Definition of any point on the circumference of the circle
def point_on_circumference (P : ℂ) (r : ℝ) : Prop :=
  ∃ (θ : ℝ), P = r * complex.exp (complex.I * θ)

-- Full proposition statement
theorem sum_of_distance_fourth_power_constant (n : ℕ) (r : ℝ) (P : ℂ) :
  regular_polygon_inscribed_circle n r →
  point_on_circumference P r →
  (∀ (P1 P2 ... Pn : ℕ → ℂ), ∃ k, ∑ k in finset.range n, (complex.abs (P - P k)) ^ 4 = 6 * n * r^4) := 
sorry

end sum_of_distance_fourth_power_constant_l713_713316


namespace motorboat_dangerous_part_l713_713882

/-!
# Dangerous Part of the Motorboat's Journey

Given the conditions:
1. The speed of the motorboat is three times that of the patrol boat.
2. The patrol boat is located halfway between the motorboat and the point on the shore that the motorboat wants to reach.
3. The motorboat travels to the destination along two sides of a square.

Prove that the motorboat is in danger between \( \frac{1}{2} \) and \( \frac{7}{8} \) of the second leg of its path.
-/

theorem motorboat_dangerous_part
  (v_m v_p : ℝ)
  (h1 : v_m = 3 * v_p)
  (a : ℝ) :
  (1/2 : ℝ) * a ≤ NP ∧ NP ≤ (7/8 : ℝ) * a :=
begin
  sorry,
end

end motorboat_dangerous_part_l713_713882


namespace katya_solves_enough_l713_713293

theorem katya_solves_enough (x : ℕ) :
  (0 ≤ x ∧ x ≤ 20) → -- x should be within the valid range of problems
  (4 / 5) * x + (1 / 2) * (20 - x) ≥ 13 → 
  x ≥ 10 :=
by 
  intros h₁ h₂
  -- Formalize the expected value equation and the inequality transformations
  sorry

end katya_solves_enough_l713_713293


namespace circle_radii_in_square_interval_l713_713414

noncomputable def circle_radii_interval (r : ℝ) : Prop :=
  r ∈ Ioo (1 - real.sqrt 2 / 2) (2 - real.sqrt 2 / 2 - real.sqrt (4 - 2 * real.sqrt 2))

theorem circle_radii_in_square_interval :
  ∀ (k₁ k₂ k₃ k₄ : ℝ) (r : ℝ),
    (∀ i j : ℕ, i ≠ j → k₁ ≠ k₂ → k₁ = k₃ → k₂ = k₄) →
    (k₁ = k₂) →
    (k₃ = k₄) →
    (∀ i, k₁ = i ∧ k₂ = i ∧ k₃ = i ∧ k₄ = i) →
    circle_radii_interval r :=
by
  intros
  sorry

end circle_radii_in_square_interval_l713_713414


namespace least_square_diff_bound_l713_713030

theorem least_square_diff_bound (a : Fin 5 → ℝ) (h : ∑ i, a i ^ 2 = 1) :
  ∃ i j : Fin 5, i ≠ j ∧ (a i - a j)^2 ≤ 1 / 10 :=
by
  sorry

end least_square_diff_bound_l713_713030


namespace price_decrease_in_may_is_26_l713_713252

def price_change (p: ℝ) (r: ℝ) := p * (1 + r / 100)

theorem price_decrease_in_may_is_26 : 
  ∀ (P0 : ℝ), 
  let P1 := price_change P0 30 in
  let P2 := price_change P1 (-10) in
  let P3 := price_change P2 15 in
  let y := 100 * (P3 - P0) / P3 in
  P3 - y / 100 * P3 = P0 →
  int.ceil y = 26 := 
by
  intros P0 P1 P2 P3 y h
  sorry

end price_decrease_in_may_is_26_l713_713252


namespace three_is_primitive_root_l713_713748

-- Problem conditions
def p (n : ℕ) : ℕ := 2^n + 1
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_primitive_root (g p : ℕ) : Prop := Nat.coprime g p ∧ ∀ d : ℕ, d < p - 1 → g^d % p ≠ 1

-- Statement to be proven
theorem three_is_primitive_root (n : ℕ) (h1 : n > 1) (h2 : is_prime (p n)) : is_primitive_root 3 (p n) :=
sorry  -- Proof to be provided

end three_is_primitive_root_l713_713748


namespace max_distance_l713_713722

theorem max_distance (z : ℂ) (hz : |z| = 3) : 
  ∃ w : ℂ, (5 + 6 * Complex.I) * z ^ 4 - z ^ 6 = w ∧ 
           dist (5 + 6 * Complex.I) z ^ 2 = 9 + Real.sqrt 61 ∧ 
           |w| = 729 + 81 * Real.sqrt 61 :=
by
  sorry

end max_distance_l713_713722


namespace find_y_value_l713_713349

def op (a b : ℤ) : ℤ := 4 * a + 2 * b

theorem find_y_value : ∃ y : ℤ, op 3 (op 4 y) = -14 ∧ y = -29 / 2 := sorry

end find_y_value_l713_713349


namespace rectangle_shorter_side_l713_713787

theorem rectangle_shorter_side
  (x : ℝ)
  (a b d : ℝ)
  (h₁ : a = 3 * x)
  (h₂ : b = 4 * x)
  (h₃ : d = 9) :
  a = 5.4 := 
by
  sorry

end rectangle_shorter_side_l713_713787


namespace simplify_fraction_l713_713071

variable (d : ℤ)

theorem simplify_fraction (d : ℤ) : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := 
by 
  sorry

end simplify_fraction_l713_713071


namespace sectors_combined_area_l713_713812

theorem sectors_combined_area (r : ℝ) (θ1 θ2 : ℝ) (h₁ : r = 10) (h₂ : θ1 = π/2) (h₃ : θ2 = π/4) :
  (1/2 * r^2 * θ1 + 1/2 * r^2 * θ2) = 37.5 * Real.pi :=
by 
  -- Using the given conditions to calculate the values
  rw [h₁, h₂, h₃]
  -- Simplifying the expression to match the correct answer
  sorry

end sectors_combined_area_l713_713812


namespace problem_statement_l713_713912

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem problem_statement :
  ¬ is_pythagorean_triple 2 3 4 ∧ 
  is_pythagorean_triple 3 4 5 ∧ 
  is_pythagorean_triple 6 8 10 ∧ 
  is_pythagorean_triple 5 12 13 :=
by 
  constructor
  sorry
  constructor
  sorry
  constructor
  sorry
  sorry

end problem_statement_l713_713912


namespace rectangle_perimeter_l713_713769

theorem rectangle_perimeter (d m n : ℝ) (h1 : d > 0) (h2 : m > 0) (h3 : n > 0) :
  let BC := d * Real.cos (m * Real.pi / (2 * (m + n)))
      DC := d * Real.cos (n * Real.pi / (2 * (m + n)))
      P := 2 * (BC + DC)
  in P = 2 * Real.sqrt 2 * d * Real.cos ((Real.pi * (m - n)) / (4 * (m + n))) :=
by
  sorry

end rectangle_perimeter_l713_713769


namespace exist_k_m_l713_713818

open Nat

noncomputable def permutation_seq_equiv (A B : List ℕ) : Prop :=
  ∀ i, (1 ≤ i ∧ i ≤ A.length) → 503 ∣ (A[i - 1] - B[i - 1])

def power_seq (A : List ℕ) (r : ℕ) : List ℕ :=
  A.map (λ a => a ^ r)

theorem exist_k_m :
  ∃ k m : ℕ, 250 ≤ k ∧ 
    (∃ pi : Fin (k+1) → List (Fin 503), 
      (∀ i : Fin k, permutation_seq_equiv (map coe (pi i)) (map (λ x => x ^ m) (pi i.succ))) ∧ 
      ∀ i j : Fin (k+1), i ≠ j → pi i ≠ pi j) := 
  sorry

end exist_k_m_l713_713818
