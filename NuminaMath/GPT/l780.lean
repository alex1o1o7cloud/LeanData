import Mathlib

namespace intersection_eq_T_l780_780251

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780251


namespace paperclip_day_l780_780419

theorem paperclip_day:
  ∃ k : ℕ, 5 * 3 ^ k > 500 ∧ ∀ m : ℕ, m < k → 5 * 3 ^ m ≤ 500 ∧ k % 7 = 5 :=
sorry

end paperclip_day_l780_780419


namespace intersection_S_T_l780_780309

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780309


namespace probability_recruitment_l780_780087

-- Definitions for conditions
def P_A : ℚ := 2/3
def P_A_not_and_B_not : ℚ := 1/12
def P_B_and_C : ℚ := 3/8

-- Independence of A, B, and C
axiom independence_A_B_C : ∀ {P_A P_B P_C : Prop}, 
  (P_A ∧ P_B ∧ P_C) → (P_A ∧ P_B) ∧ (P_A ∧ P_C) ∧ (P_B ∧ P_C)

-- Definition of probabilities of B and C
def P_B : ℚ := 3/4
def P_C : ℚ := 1/2

-- Main theorem
theorem probability_recruitment : 
  P_A = 2/3 ∧ 
  P_A_not_and_B_not = 1/12 ∧ 
  P_B_and_C = 3/8 ∧ 
  (∀ {P_A P_B P_C : Prop}, 
    (P_A ∧ P_B ∧ P_C) → (P_A ∧ P_B) ∧ (P_A ∧ P_C) ∧ (P_B ∧ P_C)) → 
  (P_B = 3/4 ∧ P_C = 1/2) ∧ 
  (2/3 * 3/4 * 1/2 + 1/3 * 3/4 * 1/2 + 2/3 * 1/4 * 1/2 + 2/3 * 3/4 * 1/2 = 17/24) := 
by sorry

end probability_recruitment_l780_780087


namespace point_outside_circle_l780_780925

theorem point_outside_circle (a b : ℝ) (h_intersect : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ a*x + b*y = 1) : a^2 + b^2 > 1 :=
by
  sorry

end point_outside_circle_l780_780925


namespace sample_size_of_village_A_l780_780870

theorem sample_size_of_village_A
  (ratio_A : ℚ) (ratio_B : ℚ) (ratio_C : ℚ)
  (sample_A : ℕ) (total_sample_ratio : ℚ)
  (ratios_sum : ratio_A + ratio_B + ratio_C = total_sample_ratio)
  (village_A_ratio : ratio_A = 3 / total_sample_ratio)
  (sample_A_val : sample_A = 15) :
  ∃ n : ℕ, n = 70 :=
begin
  use 70,
  sorry
end

end sample_size_of_village_A_l780_780870


namespace integer_sequence_unique_l780_780724

theorem integer_sequence_unique (a : ℕ → ℤ) :
  (∀ n : ℕ, ∃ p q : ℕ, p ≠ q ∧ a p > 0 ∧ a q < 0) ∧
  (∀ n : ℕ, ∀ i j : ℕ, i < n → j < n → i ≠ j → a i % (n : ℤ) ≠ a j % (n : ℤ))
  → ∀ x : ℤ, ∃! i : ℕ, a i = x :=
by
  sorry

end integer_sequence_unique_l780_780724


namespace AG_squared_eq_EG_squared_add_AC_mul_AD_l780_780406

variable (O P : Type) [metric_space O] [metric_space P]
variable (A B C D E F G : O)
variable (circO circP : set O)
variable (lineCD : O → O → Prop)

-- Given conditions
variable (hO : circO = {p | dist p O < rO})
variable (hP : circP = {p | dist p P < rP})
variable (hAB : A ∈ circO ∧ A ∈ circP ∧ B ∈ circO ∧ B ∈ circP)
variable (hCD : lineCD C D ∧ A ∈ C ∧ D ∈ C)
variable (hCBF : B ∈ lineCB ∧ lineCB C B ∧ F ∈ circP)
variable (hDBE : B ∈ lineDB ∧ lineDB D B ∧ E ∈ circO)
variable (hPerp : ⊥ (lineCD A) ∈ G)

-- Proof statement
theorem AG_squared_eq_EG_squared_add_AC_mul_AD :
  dist A G ^ 2 = dist E G ^ 2 + dist A C * dist A D := 
sorry

end AG_squared_eq_EG_squared_add_AC_mul_AD_l780_780406


namespace intersection_of_sets_l780_780171

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780171


namespace correct_calculation_l780_780632

theorem correct_calculation : sqrt 8 / sqrt 2 = 2 :=
by
-- sorry

end correct_calculation_l780_780632


namespace tan_equiv_1230_l780_780759

-- Define the conditions and problem.
theorem tan_equiv_1230 (n : ℤ) (h : -180 < n ∧ n < 180) : 
  (real.tan (n * real.pi / 180) = real.tan (1230 * real.pi / 180)) ↔ (n = 150) :=
by sorry

end tan_equiv_1230_l780_780759


namespace largest_n_binomial_l780_780564

theorem largest_n_binomial :
  (∃n : ℕ, binom 10 4 + binom 10 5 = binom 11 n) → (n ≤ 6) :=
by
  sorry

end largest_n_binomial_l780_780564


namespace intersection_S_T_eq_T_l780_780213

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780213


namespace number_of_triples_l780_780685

theorem number_of_triples : 
  ∃ n : ℕ, 
  n = 2 ∧
  ∀ (a b c : ℕ), 
    (2 ≤ a ∧ a ≤ b ∧ b ≤ c) →
    (a * b * c = 4 * (a * b + b * c + c * a)) →
    n = 2 :=
sorry

end number_of_triples_l780_780685


namespace intersection_with_x_axis_l780_780968

theorem intersection_with_x_axis :
  (∃ x, ∃ y, y = 0 ∧ y = -3 * x + 3 ∧ (x = 1 ∧ y = 0)) :=
by
  -- proof will go here
  sorry

end intersection_with_x_axis_l780_780968


namespace red_before_green_probability_l780_780677

open ProbabilityTheory

def redChips : ℕ := 4
def greenChips : ℕ := 3
def totalChips : ℕ := redChips + greenChips

noncomputable def probability_red_before_green : ℚ :=
  let totalArrangements := Nat.choose totalChips greenChips in
  let favorableArrangements := Nat.choose (totalChips - 1) (greenChips - 1) in
  favorableArrangements / totalArrangements

theorem red_before_green_probability :
  probability_red_before_green = 3 / 7 :=
by
  sorry

end red_before_green_probability_l780_780677


namespace ratio_A_B_l780_780736

/-- Definition of A and B from the problem. -/
def A : ℕ :=
  1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28

def B : ℕ :=
  1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

/-- Proof of the inequality 0 < A / B < 1 given the definitions of A and B. -/
theorem ratio_A_B (hA : A = 1400) (hB : B = 1500) : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 :=
by
  rw [hA, hB]
  norm_num
  sorry

end ratio_A_B_l780_780736


namespace solution_to_problem1_solution_to_problem2_l780_780069

noncomputable def problem1 : ℚ :=
  -54 * (-1/2 + 2/3 - 4/9)

theorem solution_to_problem1 : problem1 = 15 := by
  sorry

noncomputable def problem2 : ℚ :=
  -2 ÷ (4/9) * (-2/3) ^ 2

theorem solution_to_problem2 : problem2 = -2 := by
  sorry

end solution_to_problem1_solution_to_problem2_l780_780069


namespace min_value_problem_l780_780436

noncomputable def min_value (a b c d e f : ℝ) := (2 / a) + (3 / b) + (9 / c) + (16 / d) + (25 / e) + (36 / f)

theorem min_value_problem 
  (a b c d e f : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) 
  (h_sum : a + b + c + d + e + f = 10) : 
  min_value a b c d e f >= (329 + 38 * Real.sqrt 6) / 10 := 
sorry

end min_value_problem_l780_780436


namespace sum_of_squares_of_digits_of_462_l780_780766

-- Define what it means to be the closest perfect square to a given natural number
def closest_perfect_square (n : ℕ) : ℕ :=
  let m := Nat.sqrt n in
  if (m + 1) * (m + 1) - n < n - m * m then (m + 1) * (m + 1) else m * m

-- Given conditions from the problem statement
def f (n : ℕ) (N : ℕ) :=
  closest_perfect_square 91 * closest_perfect_square 120 * closest_perfect_square 143 * closest_perfect_square 180 * closest_perfect_square N  = 91 * 120 * 143 * 180 * N

-- Prove the main statement
theorem sum_of_squares_of_digits_of_462:
  let N := 462 in
  f 91 N →
  (4 ^ 2 + 6 ^ 2 + 2 ^ 2 = 56) :=
by
  intros h
  sorry

end sum_of_squares_of_digits_of_462_l780_780766


namespace second_group_num_persons_l780_780015

def man_hours (num_persons : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  num_persons * days * hours_per_day

theorem second_group_num_persons :
  ∀ (x : ℕ),
    let first_group_man_hours := man_hours 36 12 5
    let second_group_days := 12
    let second_group_hours_per_day := 6
    (first_group_man_hours = man_hours x second_group_days second_group_hours_per_day) →
    x = 30 :=
by
  intros x first_group_man_hours second_group_days second_group_hours_per_day h
  sorry

end second_group_num_persons_l780_780015


namespace original_monthly_bill_l780_780944

-- Define the necessary conditions
def increased_bill (original: ℝ): ℝ := original + 0.3 * original
def total_bill_after_increase : ℝ := 78

-- The proof we need to construct
theorem original_monthly_bill (X : ℝ) (H : increased_bill X = total_bill_after_increase) : X = 60 :=
by {
    sorry -- Proof is not required, only statement
}

end original_monthly_bill_l780_780944


namespace intersection_S_T_l780_780149

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780149


namespace ratio_A_B_l780_780738

/-- Definition of A and B from the problem. -/
def A : ℕ :=
  1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28

def B : ℕ :=
  1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

/-- Proof of the inequality 0 < A / B < 1 given the definitions of A and B. -/
theorem ratio_A_B (hA : A = 1400) (hB : B = 1500) : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 :=
by
  rw [hA, hB]
  norm_num
  sorry

end ratio_A_B_l780_780738


namespace intersection_S_T_l780_780141

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780141


namespace product_of_fractions_l780_780720

theorem product_of_fractions :
  (∏ n in finset.range 5 + 2, (n^2 - 1) / (n^2 + 1)) = 1 / 14 :=
by
  sorry

end product_of_fractions_l780_780720


namespace largest_n_binom_identity_l780_780554

open Nat

theorem largest_n_binom_identity :
  (∃ n, binom 11 n = binom 10 4 + binom 10 5) ∧
  (∀ m, binom 11 m = binom 10 4 + binom 10 5 → m ≤ 6) :=
by
  sorry

end largest_n_binom_identity_l780_780554


namespace least_area_triangle_l780_780998

def isSolution (z : ℂ) : Prop := (z - 4) ^ 10 = 1024

noncomputable def areaOfTriangle (D E F : ℂ) : ℝ :=
  let base := 2 * 2 * Complex.sin (Real.pi / 5)
  let height := 2 * Complex.cos (Real.pi / 5)
  (1 / 2 * base * height : ℝ)

theorem least_area_triangle (D E F : ℂ) :
  isSolution D → isSolution E → isSolution F →
  D ≠ E → E ≠ F → F ≠ D → 
  areaOfTriangle D E F = 2 * Real.sin (2 * Real.pi / 5) :=
  sorry

end least_area_triangle_l780_780998


namespace officer_assignment_count_l780_780789

namespace OfficerAssignment

-- Define four members
inductive Member
| Alice
| Bob
| Carol
| Dave
deriving DecidableEq

-- Define four officer positions
inductive Position
| President
| VicePresident
| Secretary
| Treasurer
deriving DecidableEq

-- The main theorem to prove
theorem officer_assignment_count : 
  let members := [Member.Alice, Member.Bob, Member.Carol, Member.Dave]
  let positions := [Position.President, Position.VicePresident, Position.Secretary, Position.Treasurer]
  (Finset.univ.members).perms positions.card = 24 := 
by
  sorry

end OfficerAssignment

end officer_assignment_count_l780_780789


namespace find_product_of_h_roots_l780_780442

noncomputable def p (y : ℂ) := y^5 - y^3 + 1
noncomputable def h (y : ℂ) := y^2 + 1

theorem find_product_of_h_roots :
  ∀ y1 y2 y3 y4 y5 : ℂ,
    p y1 = 0 →
    p y2 = 0 →
    p y3 = 0 →
    p y4 = 0 →
    p y5 = 0 →
    h y1 * h y2 * h y3 * h y4 * h y5 = complex.i :=
by
  intros y1 y2 y3 y4 y5 h1 h2 h3 h4 h5
  sorry

end find_product_of_h_roots_l780_780442


namespace sin_add_pi_over_4_cos_sub_5pi_over_6_2α_l780_780779

-- Assumptions
variables (α : ℝ)
hypothesis h1 : α ∈ (Set.Ioo (Real.pi / 2) Real.pi)
hypothesis h2 : Real.sin α = (Real.sqrt 5 / 5)

-- Problem 1
theorem sin_add_pi_over_4 :
  Real.sin (Real.pi / 4 + α) = - (Real.sqrt 10 / 10) :=
sorry

-- Problem 2
theorem cos_sub_5pi_over_6_2α :
  Real.cos (5 * Real.pi / 6 - 2 * α) = (-4 - 3 * Real.sqrt 3) / 10 :=
sorry

end sin_add_pi_over_4_cos_sub_5pi_over_6_2α_l780_780779


namespace tangent_line_at_3_monotonicity_of_f_l780_780342

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2

def tangent_line_equation_at_point (f : ℝ → ℝ) (a : ℝ) (x0 : ℝ) : 
    ℝ × ℝ → Prop := 
  ∃ m b : ℝ, ∀ (p : ℝ× ℝ), p.1 = x0 → f x0 = p.2 ∧ p = (x0, f x0) ∧ m = deriv f x0
    ∧ b = p.snd - m * p.fst ∧ p ∈ set.range (λ (x : ℝ), (x, m * x + b))
   
theorem tangent_line_at_3 (a : ℝ) (f : ℝ → ℝ) : 
  f = λ x, (1/3) * x^3 - (1/2) * a * x^2 → a = 2 → 
  tangent_line_equation_at_point (λ x, f x) a 3 (3, 0) := 
  by
    intro hf ha 
    sorry

theorem monotonicity_of_f (a : ℝ) : 
  (∀ x, f x a = (1 / 3) * x ^ 3 - (1 / 2) * a * x ^ 2) → 
    (if a = 0 then ∀ x1 x2, x1 ≤ x2 → f x1 a ≤ f x2 a else 
      if a < 0 then ∀ x, 
        (x < a ∨ 0 < x → deriv f x a > 0) ∧ 
        (a < x → x < 0 → deriv f x a < 0) else 
          if a > 0 then ∀ x, 
            (x < 0 ∨ a < x → deriv f x a > 0) ∧ 
            (0 < x → x < a → deriv f x a < 0) ) := 
  by
    intro hf
    sorry

end tangent_line_at_3_monotonicity_of_f_l780_780342


namespace ABCDisIsoscelesTrapezoid_l780_780013

-- Definitions of the quadrilateral, its inscribed circle, and conditions
variables {A B C D I : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace I]
variables [CircleInscribed A B C D I] -- custom typeclass for stating a circle inscribed in quadrilateral ABCD with center I

-- Specific assumption given in the problem
axiom condition : (dist A I + dist D I) ^ 2 + (dist B I + dist C I) ^ 2 = (dist A B + dist C D) ^ 2

-- Define isosceles trapezoid property
class IsoscelesTrapezoid (A B C D : Type) : Prop :=
  (parallel : parallel A D B C)
  (congruent : dist A B = dist C D)

-- Main theorem stating the problem statement
theorem ABCDisIsoscelesTrapezoid [CircleInscribed A B C D I] 
  (h : (dist A I + dist D I) ^ 2 + (dist B I + dist C I) ^ 2 = (dist A B + dist C D) ^ 2) : IsoscelesTrapezoid A B C D := by
  sorry

end ABCDisIsoscelesTrapezoid_l780_780013


namespace find_integer_pairs_l780_780753

def satisfies_conditions (m n : ℤ) : Prop :=
  m^2 = n^5 + n^4 + 1 ∧ ((m - 7 * n) ∣ (m - 4 * n))

theorem find_integer_pairs :
  ∀ (m n : ℤ), satisfies_conditions m n → (m, n) = (-1, 0) ∨ (m, n) = (1, 0) := by
  sorry

end find_integer_pairs_l780_780753


namespace exists_non_prime_form_l780_780063

theorem exists_non_prime_form (n : ℕ) : ∃ n : ℕ, ¬Nat.Prime (n^2 + n + 41) :=
sorry

end exists_non_prime_form_l780_780063


namespace S_inter_T_eq_T_l780_780303

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780303


namespace cube_distance_l780_780038

noncomputable def distance_from_vertex_A_to_plane : ℚ := (17 - 15 * real.sqrt 450) / 4

theorem cube_distance
  (side_len : ℝ)
  (height1 height2 height3 : ℝ)
  (h1 : side_len = 8)
  (h2 : height1 = 13)
  (h3 : height2 = 15)
  (h4 : height3 = 17) :
  ∃ p q r : ℕ, p + q + r < 2000 ∧ distance_from_vertex_A_to_plane = (17 - 15 * real.sqrt q) / r ∧ p = 17 ∧ q = 450 ∧ r = 4 :=
by
  use [17, 450, 4]
  simp [distance_from_vertex_A_to_plane]
  linarith
  split;
  intros x;
  exact sorry

end cube_distance_l780_780038


namespace negation_of_exists_leq_zero_l780_780497

theorem negation_of_exists_leq_zero (x : ℝ) : ¬(∃ x ≥ 1, 2^x ≤ 0) ↔ ∀ x ≥ 1, 2^x > 0 :=
by
  sorry

end negation_of_exists_leq_zero_l780_780497


namespace midpoint_of_intersection_is_correct_l780_780347

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (1 + (1/2) * t, -3 * Real.sqrt 3 + (Real.sqrt 3) / 2 * t)

def circle (R : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = R^2

def standard_line (x y : ℝ) : Prop :=
  Real.sqrt 3 * x - y - 4 * Real.sqrt 3 = 0

theorem midpoint_of_intersection_is_correct :
  ∀ (R : ℝ) (t₁ t₂ : ℝ),
  circle R (fst (parametric_line t₁)) (snd (parametric_line t₁)) →
  circle R (fst (parametric_line t₂)) (snd (parametric_line t₂)) →
  parametric_line t₁ ≠ parametric_line t₂ →
  let A := parametric_line t₁,
      B := parametric_line t₂,
      midpoint := ((fst A + fst B) / 2, (snd A + snd B) / 2) in
  midpoint = (3, -Real.sqrt 3) :=
begin
  sorry
end

end midpoint_of_intersection_is_correct_l780_780347


namespace simplify_f_value_f_second_quadrant_l780_780774

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (3 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (3 * Real.pi / 2 - α)) / 
  (Real.cos (Real.pi - α) * Real.sin (-Real.pi - α))

theorem simplify_f (α : ℝ) : 
  f α = Real.cos α := 
sorry

theorem value_f_second_quadrant (α : ℝ) (hα : π / 2 < α ∧ α < π) (hcosα : Real.cos (π / 2 + α) = -1 / 3) :
  f α = - (2 * Real.sqrt 2) / 3 := 
sorry

end simplify_f_value_f_second_quadrant_l780_780774


namespace iterated_function_identity_l780_780328

def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

noncomputable def F (a b c d : ℕ → ℝ) (x : ℝ) (n : ℕ) : ℝ :=
(nat.rec_on n x) λ n fh, f (a n) (b n) (c n) (d n) fh

theorem iterated_function_identity (a b c d : ℕ → ℝ) (h₀ : ∀ n, f (a n) (b n) (c n) (d n) 0 ≠ 0) 
  (h₁ : ∀ n, f (a n) (b n) (c n) (d n) (f (a n) (b n) (c n) (d n) 0) ≠ 0) 
  (h₂ : ∀ x, F a b c d x 0 = x) (h₃ : F a b c d 0 n = 0) : 
  ∀ x, F a b c d x n = x :=
begin
  sorry
end

end iterated_function_identity_l780_780328


namespace largest_n_for_binomial_equality_l780_780536

theorem largest_n_for_binomial_equality :
  ∃ n : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 n ) ∧
  ∀ m : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 m ) → m ≤ 6 :=
begin
  sorry
end

end largest_n_for_binomial_equality_l780_780536


namespace circle_area_l780_780404

open Real

noncomputable def radius_square (x : ℝ) (DE : ℝ) (EF : ℝ) : ℝ :=
  let DE_square := DE^2
  let r_square_1 := x^2 + DE_square
  let product_DE_EF := DE * EF
  let r_square_2 := product_DE_EF + x^2
  r_square_2

theorem circle_area (x : ℝ) (h1 : OE = x) (h2 : DE = 8) (h3 : EF = 4) :
  π * radius_square x 8 4 = 96 * π :=
by
  sorry

end circle_area_l780_780404


namespace raindrop_prob_green_slope_l780_780692

-- Define the angles at the apex of the pyramid
variables (α β γ : ℝ)

-- Define that the angles at the apex are right angles
axiom right_angle_sum : α + β + γ = π / 2

-- The main theorem to state the probability result
theorem raindrop_prob_green_slope (h : right_angle_sum α β γ) : 
  (1 - cos α ^ 2 - sin β ^ 2) = cos γ ^ 2 :=
sorry

end raindrop_prob_green_slope_l780_780692


namespace no_quadrilateral_with_given_angle_side_properties_l780_780872

theorem no_quadrilateral_with_given_angle_side_properties:
  ∀ (A B C D : Point) (AB BC CD DA : ℝ) 
    (angleA angleB angleC angleD : ℝ),
  convex_quadrilateral A B C D AB BC CD DA angleA angleB angleC angleD →
  (∀ S : set ℝ, S = {AB, BC, CD, DA} → S.pairwise (≠)) → -- all sides different
  (∀ S : set ℝ, S = {angleA, angleB, angleC, angleD} → S.pairwise (≠)) → -- all angles different
  ¬ (is_greatest_angle_non_adjacent_to_smallest_side A B C D AB BC CD DA angleA angleB angleC angleD ∧
    is_smallest_angle_non_adjacent_to_greatest_side A B C D AB BC CD DA angleA angleB angleC angleD) := sorry

end no_quadrilateral_with_given_angle_side_properties_l780_780872


namespace largest_n_binom_equality_l780_780524

theorem largest_n_binom_equality :
  ∃ n : ℕ, (binomial 10 4 + binomial 10 5 = binomial 11 n) ∧ ∀ m : ℕ, (binomial 11 m = binomial 11 5) → m ≤ 6 :=
by
  sorry

end largest_n_binom_equality_l780_780524


namespace find_a_l780_780819

theorem find_a (a : ℝ) :
  (∃ A B : ℝ × ℝ,
    (A ≠ B) ∧
    ((4 * A.1 - 3 * A.2 + a = 0) ∧ (4 * B.1 - 3 * B.2 + a = 0)) ∧
    ((A.1 * A.1 + A.2 * A.2 + 4 * A.1 = 0) ∧ (B.1 * B.1 + B.2 * B.2 + 4 * B.1 = 0)) ∧
    (∠AOB = 120)) →
  a = 3 :=
sorry

end find_a_l780_780819


namespace tv_sale_percentage_l780_780026

-- Define the original price of the TV
def original_price : ℚ := 450.00

-- Define the first discount rate
def first_discount_rate : ℚ := 0.30

-- Define the second discount rate
def second_discount_rate : ℚ := 0.10

-- Define the final sale price as a percentage of the original price
theorem tv_sale_percentage :
  let first_discounted_price := original_price * (1 - first_discount_rate),
      final_price := first_discounted_price * (1 - second_discount_rate) in
  (final_price / original_price) * 100 = 63 := by
  -- Proof is omitted
  sorry

end tv_sale_percentage_l780_780026


namespace intersection_S_T_eq_T_l780_780131

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780131


namespace functional_equation_unique_solution_l780_780099

theorem functional_equation_unique_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x - f(y)) = f(f(y)) + x * f(y) + f(x) - 1) →
  (∀ z : ℝ, f(z) = 1 - z^2 / 2) :=
by
  intros h z
  sorry

end functional_equation_unique_solution_l780_780099


namespace domain_of_f_l780_780490

theorem domain_of_f :
  {x : ℝ | x > -1 ∧ x ≠ 0 ∧ x ≤ 2} = {x : ℝ | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

end domain_of_f_l780_780490


namespace intersection_S_T_l780_780154

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780154


namespace intersection_S_T_eq_T_l780_780254

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780254


namespace cannot_express_form_ab_l780_780429

theorem cannot_express_form_ab
  (n : ℕ) (hn : 0 < n) :
  ¬ (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a * b = n^2 + 3 * n + 3 ∧ abs (a - b) < 2 * (Real.sqrt (n + 1))) :=
by sorry

end cannot_express_form_ab_l780_780429


namespace largest_n_binomial_l780_780571

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l780_780571


namespace intersection_of_sets_l780_780181

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780181


namespace range_of_a_l780_780345

noncomputable def f (x a : ℝ) : ℝ := x - (a+1) * Real.log x - a / x

noncomputable def g (x : ℝ) : ℝ := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

theorem range_of_a :
  ∀ a : ℝ, (a < 1) →
  (∃ x1 ∈ Icc Real.exp (Real.exp 2), ∀ x2 ∈ Icc (-2) 0, f x1 a < g x2) →
  a ∈ Ioo ((Real.exp 2 - 2 * Real.exp) / (Real.exp + 1)) 1 :=
sorry

end range_of_a_l780_780345


namespace percentage_red_is_40_l780_780457

-- Definitions for the conditions
variables (total_students : ℕ) (blue_percentage : ℝ) (combined_yellow_blue : ℕ)
variables (students_like_blue : ℕ) (students_like_yellow : ℕ) (students_not_like_blue : ℕ)
variables (students_like_red : ℕ) (red_percentage : ℝ)

-- Conditions given in the problem
def conditions :=
  total_students = 200 ∧ 
  blue_percentage = 0.30 ∧ 
  combined_yellow_blue = 144 ∧ 
  students_like_blue = 0.30 * 200 ∧
  students_like_yellow = 144 - students_like_blue ∧
  students_not_like_blue = total_students - students_like_blue ∧
  students_like_red = students_not_like_blue - students_like_yellow

-- The goal: Prove that the percentage of the remaining students who like the color red equals 40%
theorem percentage_red_is_40 (h : conditions total_students blue_percentage combined_yellow_blue students_like_blue students_like_yellow students_not_like_blue students_like_red red_percentage) :
  red_percentage = 40 :=
by { -- Proof goes here
  sorry
}

end percentage_red_is_40_l780_780457


namespace ratio_A_B_l780_780743

noncomputable def A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
noncomputable def B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)

theorem ratio_A_B :
  let r := (A : ℚ) / (B : ℚ) in 0 < r ∧ r < 1 :=
by {
  -- Proof steps are omitted with sorry
  sorry
}

end ratio_A_B_l780_780743


namespace coeff_x4_is_200_l780_780402

open Nat

-- Define the polynomial (1 + x^3)(1 - x)^10
def poly := (1 + X^3) * (1 - X) ^ 10

-- State the goal: to prove that the coefficient of x^4 in "poly" is 200
theorem coeff_x4_is_200 : coeff poly 4 = 200 :=
by
  sorry

end coeff_x4_is_200_l780_780402


namespace intersection_S_T_eq_T_l780_780279

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780279


namespace intersection_of_sets_l780_780173

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780173


namespace num_possible_sums_l780_780446

theorem num_possible_sums (s : Finset ℕ) (hs : s.card = 80) (hsub: s ⊆ Finset.range 121) : 
  ∃ (n : ℕ), (n = 3201) ∧ ∀ U, U = s.sum id → ∃ (U_min U_max : ℕ), U_min = 3240 ∧ U_max = 6440 ∧ (U_min ≤ U ∧ U ≤ U_max) :=
sorry

end num_possible_sums_l780_780446


namespace largest_n_for_binom_equality_l780_780594

theorem largest_n_for_binom_equality : ∃ (n : ℕ), (∑ i : {i // i = 4 ∨ i = 5}, Nat.choose 10 i) = Nat.choose 11 n ∧ n = 6 :=
by
  sorry

end largest_n_for_binom_equality_l780_780594


namespace one_corresponds_to_36_l780_780874

-- Define the given conditions
def corresponds (n : Nat) (s : String) : Prop :=
match n with
| 2  => s = "36"
| 3  => s = "363"
| 4  => s = "364"
| 5  => s = "365"
| 36 => s = "2"
| _  => False

-- Statement for the proof problem: Prove that 1 corresponds to 36
theorem one_corresponds_to_36 : corresponds 1 "36" :=
by
  sorry

end one_corresponds_to_36_l780_780874


namespace ratio_of_areas_l780_780877

-- Define the problem conditions
variables (a b : ℝ) -- lengths of sides AB and AD
variable  (ABCD : Type) [rectangle ABCD] -- ABCD is a rectangle
variable  (M N : Point) -- M and N are points
variables [segment AB AD] [midpoint M AB] [midpoint N BC] -- M is midpoint of AB, N is midpoint of BC

-- Define the areas
def area_triangle_AMN : ℝ := (1 / 2) * (a / 2) * (b / 2)
def area_rectangle_ABCD : ℝ := a * b

-- Prove the ratio of areas is 1/8
theorem ratio_of_areas : (area_triangle_AMN / area_rectangle_ABCD) = (1 / 8) :=
by sorry

end ratio_of_areas_l780_780877


namespace find_AD_length_l780_780799

-- Define the cyclic quadrilateral and its properties.
structure CyclicQuadrilateral (ABCD : Type) :=
  (R : ℝ) -- Circumradius
  (AB BC CD AD : ℝ) -- Side lengths
  (is_cyclic : R > 0) -- Condition for being cyclic
  (AB_eq : AB = 200)
  (BC_eq : BC = 200)
  (CD_eq : CD = 200)
  (R_eq : R = 200 * Real.sqrt 2)

-- Define the theorem we want to prove.
theorem find_AD_length (ABCD : Type) [CyclicQuadrilateral ABCD] : 
  CyclicQuadrilateral.AD ABCD = 200 * Real.sqrt 3 := 
by
  sorry

end find_AD_length_l780_780799


namespace officer_assignment_count_l780_780788

namespace OfficerAssignment

-- Define four members
inductive Member
| Alice
| Bob
| Carol
| Dave
deriving DecidableEq

-- Define four officer positions
inductive Position
| President
| VicePresident
| Secretary
| Treasurer
deriving DecidableEq

-- The main theorem to prove
theorem officer_assignment_count : 
  let members := [Member.Alice, Member.Bob, Member.Carol, Member.Dave]
  let positions := [Position.President, Position.VicePresident, Position.Secretary, Position.Treasurer]
  (Finset.univ.members).perms positions.card = 24 := 
by
  sorry

end OfficerAssignment

end officer_assignment_count_l780_780788


namespace least_possible_value_of_k_l780_780375

theorem least_possible_value_of_k (k : ℤ) (h : 0.00010101 * 10^k > 1000) : k >= 8 := sorry

end least_possible_value_of_k_l780_780375


namespace max_discount_l780_780037

def cost_price := 100
def selling_price := 150
def profit_margin := 0.20

theorem max_discount (x : ℝ) :
  (selling_price - (selling_price * x / 100) - cost_price) / cost_price ≥ profit_margin → x ≤ 20 :=
sorry

end max_discount_l780_780037


namespace boy_reaches_early_l780_780032

theorem boy_reaches_early :
  ∀ (s₁ s₂ D : ℝ) (d : ℕ),
  s₁ = 4 → s₂ = 8 → D = 2 → d = 7 →
  let time₁ := D / s₁ in
  let delay := d / 60 in
  let t_actual := time₁ - delay in
  let time₂ := D / s₂ in
  let Delta_t := t_actual - time₂ in
  Delta_t * 60 = 8 :=
by sorry

end boy_reaches_early_l780_780032


namespace exists_Q_R_nonnegative_l780_780007

-- Condition: P(x) is a polynomial with real coefficients and P(x) > 0 for all x > 0.
variable (P : ℝ[X])
hypothesis (hP : ∀ x > 0, P.eval x > 0)

-- Goal: There exist polynomials Q(x) and R(x) with nonnegative coefficients such that P(x) = Q(x) / R(x).
theorem exists_Q_R_nonnegative (P : ℝ[X]) (hP : ∀ x > 0, P.eval x > 0) :
  ∃ (Q R : ℝ[X]), (∀ i, 0 ≤ Q.coeff i) ∧ (∀ j, 0 ≤ R.coeff j) ∧ (P = Q / R) :=
sorry

end exists_Q_R_nonnegative_l780_780007


namespace smallest_m_n_l780_780727

noncomputable def g (m n : ℕ) (x : ℝ) : ℝ := Real.arccos (Real.log (↑n * x) / Real.log (↑m))

theorem smallest_m_n (m n : ℕ) (h1 : 1 < m) (h2 : ∀ x : ℝ, -1 ≤ Real.log (↑n * x) / Real.log (↑m) ∧
                      Real.log (↑n * x) / Real.log (↑m) ≤ 1 ∧
                      (forall a b : ℝ,  a ≤ x ∧ x ≤ b -> b - a = 1 / 1007)) :
  m + n = 1026 :=
sorry

end smallest_m_n_l780_780727


namespace Captain_Zarnin_staffing_scheme_l780_780717

theorem Captain_Zarnin_staffing_scheme :
  let positions := 6
  let candidates := 15
  (Nat.choose candidates positions) * 
  (Nat.factorial positions) = 3276000 :=
by
  let positions := 6
  let candidates := 15
  let ways_to_choose := Nat.choose candidates positions
  let ways_to_permute := Nat.factorial positions
  have h : (ways_to_choose * ways_to_permute) = 3276000 := sorry
  exact h

end Captain_Zarnin_staffing_scheme_l780_780717


namespace sum_of_numbers_l780_780992

theorem sum_of_numbers (x : ℝ) (hx_sum_squares : x^2 + (2 * x)^2 + (4 * x)^2 = 1701) :=
  x + 2 * x + 4 * x = 63

end sum_of_numbers_l780_780992


namespace super_cool_triangle_area_sum_l780_780050

theorem super_cool_triangle_area_sum : 
  let is_supercool (a b : ℕ) := (a > 0) ∧ (b > 0) ∧ (a * b) / 2 = 3 * (a + b)
  let areas := { ab / 2 | (a, b) in { (a, b) | is_supercool a b }} 
    ∑ area in areas := 219 :=
begin
  sorry,
end

end super_cool_triangle_area_sum_l780_780050


namespace find_circle_equation_find_tangent_lines_l780_780784

-- Definitions for the conditions
def A := (6 : ℝ, 2 * Real.sqrt 3)
def B := (4 : ℝ, 4)
def O := (0 : ℝ, 0)
def P := (0 : ℝ, 4 * Real.sqrt 3)
def circumcircle_center := (4 : ℝ, 3 * Real.sqrt 3 / 2)
def circumcircle_radius := 4 : ℝ

-- Proving the general equation of the circumcircle of triangle OAB
theorem find_circle_equation :
  ∃ x y, (x - 4)^2 + (y - 3 * Real.sqrt 3 / 2)^2 = 16 :=
sorry

-- Proving the equation of line l passing through P and tangent to the circle
theorem find_tangent_lines :
  ∃ (k : ℝ),
    (P.1 + P.2 - 3 * Real.sqrt 3 = 0) ∨ (k = -Real.sqrt 3 / 3 ∧ (P.1 + P.2 * Real.sqrt 3 - 12 = 0)) :=
sorry

end find_circle_equation_find_tangent_lines_l780_780784


namespace largest_n_binom_identity_l780_780550

open Nat

theorem largest_n_binom_identity :
  (∃ n, binom 11 n = binom 10 4 + binom 10 5) ∧
  (∀ m, binom 11 m = binom 10 4 + binom 10 5 → m ≤ 6) :=
by
  sorry

end largest_n_binom_identity_l780_780550


namespace intersection_eq_T_l780_780247

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780247


namespace A_completion_days_l780_780033

-- Definitions based on the conditions
def B_days := 5
def C_days := 7.5
def combined_days := 2

-- Problem statement
theorem A_completion_days (W : ℝ) (A_rate B_rate C_rate : ℝ) :
    B_rate = W / B_days →
    C_rate = W / C_days →
    (A_rate + B_rate + C_rate) = W / combined_days →
    1 / A_rate = 6 :=
by
    intros hB_rate hC_rate hCombined
    sorry

end A_completion_days_l780_780033


namespace vector_sum_length_l780_780116

/- Define the problem statement -/
theorem vector_sum_length (n : ℕ) (v : Fin n → ℝ^2)
  (h_len : ∀ i, ‖v i‖ = 1)
  (h_sum : (∑ i, v i) = 0) :
  ∃ (σ : Fin n → Fin n), ∀ k : Fin n, ‖∑ i in Finset.range (k + 1), v (σ i)‖ ≤ 2 :=
by
  sorry

end vector_sum_length_l780_780116


namespace largest_n_binomial_l780_780570

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l780_780570


namespace incorrect_inequality_l780_780771

theorem incorrect_inequality (a b : ℝ) (h : a < b) : ¬ (-4 * a < -4 * b) :=
by sorry

end incorrect_inequality_l780_780771


namespace intersection_of_sets_l780_780011

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {1, 2, 3}) (hB : B = {3, 4}) :
  A ∩ B = {3} :=
by
  rw [hA, hB]
  exact sorry

end intersection_of_sets_l780_780011


namespace intersection_S_T_eq_T_l780_780211

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780211


namespace largest_n_for_binom_equality_l780_780591

theorem largest_n_for_binom_equality : ∃ (n : ℕ), (∑ i : {i // i = 4 ∨ i = 5}, Nat.choose 10 i) = Nat.choose 11 n ∧ n = 6 :=
by
  sorry

end largest_n_for_binom_equality_l780_780591


namespace mean_temperature_84_l780_780987

-- Define the given temperatures
def temperatures : List ℝ := [82, 85, 83, 86, 84, 87, 82]

-- Define the problem to prove the mean temperature is 84
theorem mean_temperature_84 (temps : List ℝ) (h : temps = [82, 85, 83, 86, 84, 87, 82]) :
  (temps.sum / temps.length).round = 84 :=
by
  -- sorry to skip the proof
  debug_print temperatures
  sorry

end mean_temperature_84_l780_780987


namespace largest_consecutive_integer_sum_l780_780374

theorem largest_consecutive_integer_sum (n : ℕ) : 
  (∑ k in Finset.range (n + 1), k) ≤ 10000000 → n ≤ 14141 :=
begin
  sorry
end

end largest_consecutive_integer_sum_l780_780374


namespace intersection_S_T_l780_780187

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780187


namespace max_positive_factors_l780_780721

theorem max_positive_factors (b n : ℕ) (h_b : 1 ≤ b ∧ b ≤ 15) (h_n : 1 ≤ n ∧ n ≤ 15) 
  (h_b_primes : ∃ p₁ p₂, p₁ < 15 ∧ p₂ < 15 ∧ p₁ ≠ p₂ ∧ b = p₁ * p₂) : 
  (∀ b n, b = 6 ∧ n = 15 → nat.divisors_count (b^n) = 256) := 
begin
  intro b, 
  intro n,
  intro h_def,
  cases h_def with hb hn,
  rw [hb, hn],
  dsimp,
  have h_prime_factors : nat.prime_factors 6 = [2, 3] := sorry,
  have h_natural_definitions : ∀ {a b : ℕ}, nat.prime_factors (a * b) = (nat.prime_factors a) ++ (nat.prime_factors b) := sorry,
  rw [nat.prime_factors_eq_to 6, [2, 3] to prime_exponent_formula, 2 to power_exponent_formula, divisors_count_of_36, factor_count_of_exponents],
  norm_num,
  exact sorry
end

end max_positive_factors_l780_780721


namespace intersection_S_T_l780_780142

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780142


namespace intersection_S_T_eq_T_l780_780206

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780206


namespace problem1_problem2_problem3_l780_780093

section Problem1

variable (a b : ℝ)
variable (h_condition : a^b * b^a + log a b = 0)
variable (ha_pos : 0 < a)
variable (hb_pos : 0 < b)

theorem problem1 : ab + 1 < a + b :=
sorry

end Problem1

section Problem2

variable (b : ℝ)
variable (h_condition : 2^b * b^2 + log 2 b = 0)
variable (hb_pos : 0 < b)

theorem problem2 : 2^b < 1 / b :=
sorry

end Problem2

section Problem3

variable (b : ℝ)
variable (h_condition : 2^b * b^2 + log 2 b = 0)
variable (hb_pos : 0 < b)

theorem problem3 : (2 * b + 1 - real.sqrt 5) * (3 * b - 2) < 0 :=
sorry

end Problem3

end problem1_problem2_problem3_l780_780093


namespace largest_n_binom_identity_l780_780553

open Nat

theorem largest_n_binom_identity :
  (∃ n, binom 11 n = binom 10 4 + binom 10 5) ∧
  (∀ m, binom 11 m = binom 10 4 + binom 10 5 → m ≤ 6) :=
by
  sorry

end largest_n_binom_identity_l780_780553


namespace largest_integer_binom_l780_780618

theorem largest_integer_binom
  (n : ℕ)
  (h₁ : binom 10 4 + binom 10 5 = binom 11 n) :
  n = 6 :=
sorry

end largest_integer_binom_l780_780618


namespace largest_n_binomial_l780_780565

theorem largest_n_binomial :
  (∃n : ℕ, binom 10 4 + binom 10 5 = binom 11 n) → (n ≤ 6) :=
by
  sorry

end largest_n_binomial_l780_780565


namespace evaluate_ff_14_eq_1_9_l780_780340

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≥ 0 then log x / log 2 else 3^x

theorem evaluate_ff_14_eq_1_9 :
  f (f (1 / 4)) = 1 / 9 :=
by  
  sorry

end evaluate_ff_14_eq_1_9_l780_780340


namespace henry_games_l780_780411

theorem henry_games {N H : ℕ} (hN : N = 7) (hH : H = 4 * N) 
    (h_final: H - 6 = 4 * (N + 6)) : H = 58 :=
by
  -- Proof would be inserted here, but skipped using sorry
  sorry

end henry_games_l780_780411


namespace pentagon_edges_same_color_l780_780722

-- Definitions required for the problem
variables {P : Type} [prism P]
variables {A1 A2 A3 A4 A5 B1 B2 B3 B4 B5 : P}

-- Edges and conditions
variable (colored_edge : Π (x y : P), Prop)
variable (red blue : Prop)
variable (color : P → P → Prop)
variables (edges : list (P × P))
variable (is_triangle : P → P → P → Prop)
variable (triangle_property : Π a b c : P, is_triangle a b c → (colored_edge a b ↔ ¬ colored_edge b c))

-- Set of all vertices of the pentagonal prism
variables (verts_top : set P)
variables (verts_bottom : set P)

theorem pentagon_edges_same_color :
  (∀ x y : P, (x, y) ∈ verts_top × verts_top ∨ (x, y) ∈ verts_bottom × verts_bottom →
    colored_edge x y = red ∨ colored_edge x y = blue) →
  (∀ i j : {1, 2, 3, 4, 5}, is_triangle (A1 + i) (A1 + j) (B1 + i)) →
  (∀ i j : {1, 2, 3, 4, 5}, is_triangle (A1 + i) (A1 + j) (B1 + j)) →
  -- Given that each triangle formed by the vertices of the prism and having edges in colored segments has two sides of different colors
  (∀ x y z : P, is_triangle x y z → (colored_edge x y ≠ colored_edge y z ∧ colored_edge y z ≠ colored_edge z x)) →
  -- Conclusion: All ten edges of the top and bottom pentagons of the prism have the same color
  (∀ x y : P, (x, y) ∈ verts_top × verts_top ∨ (x, y) ∈ verts_bottom × verts_bottom →
    colored_edge x y = red ∨ colored_edge x y = blue) ∧ sorry :=
begin
  sorry
end

end pentagon_edges_same_color_l780_780722


namespace power_function_value_l780_780859

theorem power_function_value (α : ℝ) (f : ℝ → ℝ) (h1 : f = λ x, x ^ α) (h2 : f 2 = (Mathlib.sqrt 2) / 2) : f 4 = 1 / 2 :=
by sorry

end power_function_value_l780_780859


namespace patrick_height_100_inches_l780_780025

theorem patrick_height_100_inches 
  (height_light_pole shadow_light_pole : ℝ) 
  (shadow_patrick : ℝ) 
  (h1 : height_light_pole = 60) 
  (h2 : shadow_light_pole = 15) 
  (h3 : shadow_patrick = 25) : 
  let height_patrick := (height_light_pole / shadow_light_pole) * shadow_patrick 
  in height_patrick = 100 :=
by
  let height_patrick := (height_light_pole / shadow_light_pole) * shadow_patrick
  have : height_patrick = (60 / 15) * 25 := by sorry
  have : height_patrick = 4 * 25 := by sorry
  have : height_patrick = 100 := by sorry
  exact this

end patrick_height_100_inches_l780_780025


namespace intersection_S_T_eq_T_l780_780136

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780136


namespace intersection_S_T_eq_T_l780_780271

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780271


namespace intersection_S_T_l780_780148

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780148


namespace shaded_area_value_l780_780868

theorem shaded_area_value :
  let area_grid := 6 * 6 * (1.5 * 1.5),
      area_small_circles := 4 * Real.pi * (0.75 * 0.75),
      area_large_circle := Real.pi * (1.5 * 1.5),
      area_hexagon := (3 * Real.sqrt 3 / 2) * (1.5 * 1.5),
      area_total := area_small_circles + area_large_circle + area_hexagon,
      shaded_area := area_grid - area_total
  in shaded_area = 81 - 4.5 * Real.pi - 3.375 * Real.sqrt 3 ∧ 81 + 4.5 + 3.375 = 88.875 :=
by
  sorry

end shaded_area_value_l780_780868


namespace intersection_of_sets_l780_780180

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780180


namespace estimate_ratio_l780_780734

theorem estimate_ratio (A B : ℕ) (A_def : A = 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28)
  (B_def : B = 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20) : 0 < A / B ∧ A / B < 1 := by
  sorry

end estimate_ratio_l780_780734


namespace theta_in_second_quadrant_l780_780838

open Real

-- Definitions for conditions
def cond1 (θ : ℝ) : Prop := sin θ > cos θ
def cond2 (θ : ℝ) : Prop := tan θ < 0

-- Main theorem statement
theorem theta_in_second_quadrant (θ : ℝ) 
  (h1 : cond1 θ) 
  (h2 : cond2 θ) : 
  θ > π/2 ∧ θ < π :=
sorry

end theta_in_second_quadrant_l780_780838


namespace least_number_subtracted_l780_780623

theorem least_number_subtracted (n m : ℕ) (h₁ : m = 2590) (h₂ : n = 2590 - 16) :
  (n % 9 = 6) ∧ (n % 11 = 6) ∧ (n % 13 = 6) :=
by
  sorry

end least_number_subtracted_l780_780623


namespace tangential_quadrilateral_area_convex_quadrilateral_area_l780_780953

noncomputable def area_tangential_quadrilateral (a b c d A C : ℝ) : ℝ :=
  real.sqrt (a * b * c * d) * real.sin ((A + C) / 2)

noncomputable def area_convex_quadrilateral (a b c d p ϕ: ℝ) : ℝ :=
  real.sqrt (((p - a) * (p - b) * (p - c) * (p - d)) - a * b * c * d * real.cos ϕ ^ 2)

theorem tangential_quadrilateral_area (a b c d A C : ℝ) : 
  area_tangential_quadrilateral a b c d A C = real.sqrt (a * b * c * d) * real.sin ((A + C) / 2) := 
sorry

theorem convex_quadrilateral_area (a b c d p ϕ: ℝ) : 
  area_convex_quadrilateral a b c d p ϕ = 
  real.sqrt (((p - a) * (p - b) * (p - c) * (p - d)) - a * b * c * d * real.cos ϕ ^ 2) :=
sorry

end tangential_quadrilateral_area_convex_quadrilateral_area_l780_780953


namespace correct_calculation_l780_780631

theorem correct_calculation : sqrt 8 / sqrt 2 = 2 :=
by
-- sorry

end correct_calculation_l780_780631


namespace intersection_eq_T_l780_780245

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780245


namespace problem_l780_780776

noncomputable def givenCondition (x : ℝ) := log 5 3 = 2 / x

theorem problem (x : ℝ) (h : givenCondition x) : 3 ^ x * 27 ^ (-x / 2) = 1 / 5 :=
by {
  sorry
}

end problem_l780_780776


namespace ball_never_falls_into_pocket_l780_780684

-- Define the condition of the billiard table dimensions
def table_width : ℝ := 1
def table_height : ℝ := real.sqrt 2

-- Define the initial position of the ball
def initial_position : ℝ × ℝ := (0, 0)

-- Define the shooting angle of the ball
def shooting_angle : ℝ := real.pi / 4

-- Define the pockets positions
def is_pocket (p : ℝ × ℝ) : Prop :=
  (p = (0, 0)) ∨ (p = (1, 0)) ∨ (p = (0, real.sqrt 2)) ∨ (p = (1, real.sqrt 2))

-- Prove that the ball will never fall into a pocket 
theorem ball_never_falls_into_pocket :
  ∀ (ball_position : ℝ × ℝ), ¬ is_pocket ball_position :=
sorry

end ball_never_falls_into_pocket_l780_780684


namespace f_2007_eq_sin_l780_780775

def f : ℕ →  (ℝ → ℝ)
| 0 := λ x, Real.cos x
| (n + 1) := λ x, (f n)' x

theorem f_2007_eq_sin (x : ℝ) : f 2007 x = Real.sin x := by 
  sorry

end f_2007_eq_sin_l780_780775


namespace largest_n_binom_identity_l780_780552

open Nat

theorem largest_n_binom_identity :
  (∃ n, binom 11 n = binom 10 4 + binom 10 5) ∧
  (∀ m, binom 11 m = binom 10 4 + binom 10 5 → m ≤ 6) :=
by
  sorry

end largest_n_binom_identity_l780_780552


namespace sector_central_angle_l780_780119

theorem sector_central_angle (r l α : ℝ) 
  (h1 : 2 * r + l = 6) 
  (h2 : 0.5 * l * r = 2) :
  α = l / r → α = 4 ∨ α = 1 :=
sorry

end sector_central_angle_l780_780119


namespace positive_difference_of_diagonal_sums_l780_780652

noncomputable def initial_matrix : matrix (fin 4) (fin 4) ℕ := λ i j =>
  [[1, 2, 3, 4],
   [8, 9, 10, 11],
   [15, 16, 17, 18],
   [22, 23, 24, 25]].nth i.untyped! j.untyped! .get!

noncomputable def modified_matrix : matrix (fin 4) (fin 4) ℕ := λ i j =>
  [[1, 2, 3, 4],
   [11, 10, 9, 8],
   [15, 16, 17, 18],
   [25, 24, 23, 22]].nth i.untyped! j.untyped! .get!

def diag_sum (m : matrix (fin 4) (fin 4) ℕ) : ℕ :=
  ∑ i, m i i

def secondary_diag_sum (m : matrix (fin 4) (fin 4) ℕ) : ℕ :=
  ∑ i, m i (3 - i.to_nat)

theorem positive_difference_of_diagonal_sums :
  |diag_sum modified_matrix - secondary_diag_sum modified_matrix| = 4 :=
by
  sorry

end positive_difference_of_diagonal_sums_l780_780652


namespace probability_P_eq_one_l780_780507

/-- The vertices of a triangle in the complex plane -/
def vertices : List ℂ := [1 + I, 1 - I, -2]

/-- The probability P = 1 for given condition to be 1 out of 19683 -/
theorem probability_P_eq_one :
  ∃ (a b : ℕ), coprime a b ∧ b = 19683 ∧ a + b = 19684 ∧ (P = 1 → probability P = 1 / 19683) :=
sorry

end probability_P_eq_one_l780_780507


namespace solution_x_volume_required_l780_780959

-- Let Vx be the volume of solution x
def Vx (V : ℕ) : ℝ := 0.10 * V -- volume of alcohol in solution x

-- Define the known values
def solution_y_volume : ℕ := 200
def solution_y_alcohol : ℝ := 0.30 * solution_y_volume
def desired_concentration (total_volume : ℝ) : ℝ := 0.20 * total_volume

-- Proof statement: we need to show that the desired volume V_x is 200 ml to get 20% alcohol by volume in the new solution
theorem solution_x_volume_required : 
  ∀ (V_x : ℕ), (0.10 * V_x + (0.30 * solution_y_volume) = 0.20 * (V_x + solution_y_volume)) → (V_x = 200) :=
begin
  intros V_x h,
  have h1 : 0.10 * V_x + (0.30 * solution_y_volume) = 0.20 * (V_x + solution_y_volume) := h,
  sorry,
end

end solution_x_volume_required_l780_780959


namespace intersection_eq_T_l780_780252

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780252


namespace largest_n_binomial_l780_780566

theorem largest_n_binomial :
  (∃n : ℕ, binom 10 4 + binom 10 5 = binom 11 n) → (n ≤ 6) :=
by
  sorry

end largest_n_binomial_l780_780566


namespace intersection_eq_T_l780_780287

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780287


namespace partition_students_l780_780061

theorem partition_students (
  eng_speakers: Set Student,
  ger_speakers: Set Student,
  fre_speakers: Set Student,
  h_eng: eng_speakers.card = 50,
  h_ger: ger_speakers.card = 50,
  h_fre: fre_speakers.card = 50
) : ∃ (groups : Fin 5 → Set Student), 
    (∀ (i : Fin 5), (groups i).card = 10 ∧ (eng_speakers ∩ groups i).card = 10 ∧ (ger_speakers ∩ groups i).card = 10 ∧ (fre_speakers ∩ groups i).card = 10) :=
begin
  sorry
end

end partition_students_l780_780061


namespace correct_calculation_l780_780627

theorem correct_calculation :
  (2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  (5 * Real.sqrt 3 * 5 * Real.sqrt 2 ≠ 5 * Real.sqrt 6) ∧
  (Real.sqrt (4 + 1/2) ≠ 2 * Real.sqrt (1/2)) :=
by
  sorry

end correct_calculation_l780_780627


namespace sum_of_angles_l780_780941

theorem sum_of_angles (PS QT RU : Line) (O P Q R S T U: Point)
  (h1: PS ∩ QT ∩ RU = {O})
  (h2 : P ∈ PS) (h3 : Q ∈ QT) (h4 : R ∈ RU) 
  (h5 : S ∈ PS) (h6 : T ∈ QT) (h7 : U ∈ RU) 
  (h8 : is_triangle (P, O, R))
  (h9 : is_triangle (Q, O, S))
  (h10 : is_triangle (T, O, U))
  : ∠P + ∠Q + ∠R + ∠S + ∠T + ∠U = 360 := 
sorry

end sum_of_angles_l780_780941


namespace quadratic_root_difference_l780_780485

theorem quadratic_root_difference 
  (a b c : ℕ) (h_a : a ≤ 2019) (h_b : b ≤ 2019) (h_c : c ≤ 2019) :
  let discriminant := b^2 - 4*a*c in
  (discriminant ≥ 0) ∧ ((discriminant > 0) → ((abs ( sqrt discriminant / a)) < 0.01)) → 
  (a = 1001) ∧ (b = 2001) ∧ (c = 1000)
:= 
by
  sorry

end quadratic_root_difference_l780_780485


namespace percentage_within_one_standard_deviation_l780_780036

theorem percentage_within_one_standard_deviation (m d : ℝ) 
  (symmetric_distribution : ∀ x, P (m - x) = P (m + x))
  (percent_less_than_m_plus_d : P (m + d) = 0.80) : 
  P (m - d < x ∧ x < m + d) = 0.60 :=
sorry

end percentage_within_one_standard_deviation_l780_780036


namespace james_daily_soda_consumption_l780_780415

theorem james_daily_soda_consumption
  (N_p : ℕ) -- number of packs
  (S_p : ℕ) -- sodas per pack
  (S_i : ℕ) -- initial sodas
  (D : ℕ)  -- days in a week
  (h1 : N_p = 5)
  (h2 : S_p = 12)
  (h3 : S_i = 10)
  (h4 : D = 7) : 
  (N_p * S_p + S_i) / D = 10 := 
by 
  sorry

end james_daily_soda_consumption_l780_780415


namespace intersection_S_T_l780_780318

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780318


namespace find_pi_l780_780008

def volume_formula (C H V : ℝ) : Prop := 
  V = (1 / 12) * (C^2 * H)

variables (C : ℝ) (H : ℝ) (pi : ℝ) (R : ℝ) 

def cylinder_conditions : Prop := 
  C = 2 * pi * R ∧ volume_formula C H ((1 / 12) * (C^2 * H))

theorem find_pi (h : cylinder_conditions C H (400 / 3)) : pi = 3 :=
sorry

end find_pi_l780_780008


namespace quadratic_equality_l780_780831

theorem quadratic_equality (a_2 : ℝ) (a_1 : ℝ) (a_0 : ℝ) (r : ℝ) (s : ℝ) (x : ℝ)
  (h₁ : a_2 ≠ 0)
  (h₂ : a_0 ≠ 0)
  (h₃ : a_2 * r^2 + a_1 * r + a_0 = 0)
  (h₄ : a_2 * s^2 + a_1 * s + a_0 = 0) :
  a_0 + a_1 * x + a_2 * x^2 = a_0 * (1 - x / r) * (1 - x / s) :=
by
  sorry

end quadratic_equality_l780_780831


namespace polynomial_roots_property_l780_780950

theorem polynomial_roots_property (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let p := λ x => α * x ^ 3 - α * x ^ 2 + β * x + β in
  let roots := Multiset.roots (Polynomial.ofFunction p) in
  roots.card = 3 →
  ∃ x1 x2 x3 : ℝ, 
  Multiset.map (Function.inv (root_1_VEC x1 x2 x3)) roots =
  [x1, x2, x3].map (λ x, 1 / x) ∧
  (x1 + x2 + x3) * (1 / x1 + 1 / x2 + 1 / x3) = -1 :=
by
  sorry

end polynomial_roots_property_l780_780950


namespace hospital_bed_occupancy_l780_780674

theorem hospital_bed_occupancy 
  (x : ℕ)
  (beds_A := x)
  (beds_B := 2 * x)
  (beds_C := 3 * x)
  (occupied_A := (1 / 3) * x)
  (occupied_B := (1 / 2) * (2 * x))
  (occupied_C := (1 / 4) * (3 * x))
  (max_capacity_B := (3 / 4) * (2 * x))
  (max_capacity_C := (5 / 6) * (3 * x)) :
  (4 / 3 * x) / (2 * x) = 2 / 3 ∧ (3 / 4 * x) / (3 * x) = 1 / 4 := 
  sorry

end hospital_bed_occupancy_l780_780674


namespace bus_ride_cost_l780_780053

/-- The cost of a bus ride from town P to town Q, given that the cost of a train ride is $2.35 more 
    than a bus ride, and the combined cost of one train ride and one bus ride is $9.85. -/
theorem bus_ride_cost (B : ℝ) (h1 : ∃T, T = B + 2.35) (h2 : ∃T, T + B = 9.85) : B = 3.75 :=
by
  obtain ⟨T1, hT1⟩ := h1
  obtain ⟨T2, hT2⟩ := h2
  simp only [hT1, add_right_inj] at hT2
  sorry

end bus_ride_cost_l780_780053


namespace DeepakAgeProof_l780_780002

def RahulAgeAfter10Years (RahulAge : ℕ) : Prop := RahulAge + 10 = 26

def DeepakPresentAge (ratioRahul ratioDeepak : ℕ) (RahulAge : ℕ) : ℕ :=
  (2 * RahulAge) / ratioRahul

theorem DeepakAgeProof {DeepakCurrentAge : ℕ}
  (ratioRahul ratioDeepak RahulAge : ℕ)
  (hRatio : ratioRahul = 4)
  (hDeepakRatio : ratioDeepak = 2) :
  RahulAgeAfter10Years RahulAge →
  DeepakCurrentAge = DeepakPresentAge ratioRahul ratioDeepak RahulAge :=
  sorry

end DeepakAgeProof_l780_780002


namespace intersection_of_S_and_T_l780_780236

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780236


namespace xy_sum_l780_780326

noncomputable def f (t : ℝ) : ℝ := t^3 + 2 * t + sin t

theorem xy_sum (x y : ℝ) (h1 : (x - 2)^3 + 2 * x + sin (x - 2) = 2)
    (h2 : (y - 2)^3 + 2 * y + sin (y - 2) = 6) : x + y = 4 := 
by {
    -- Proof steps (skipped with sorry)
    sorry
}

end xy_sum_l780_780326


namespace angle_bisectors_product_not_unique_l780_780060

-- Define the problem context
def cannot_uniquely_determine_triangle (a b c : ℕ) : Prop :=
∃ Δ1 Δ2 : Triangle, Δ1 ≠ Δ2 ∧ Δ1.angle_bisectors_product = Δ2.angle_bisectors_product

-- State the theorem
theorem angle_bisectors_product_not_unique:
  cannot_uniquely_determine_triangle (angle_bisectors_product Δ1) (angle_bisectors_product Δ2) (angle_bisectors_product Δ3) := sorry

end angle_bisectors_product_not_unique_l780_780060


namespace ratio_A_B_l780_780741

noncomputable def A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
noncomputable def B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)

theorem ratio_A_B :
  let r := (A : ℚ) / (B : ℚ) in 0 < r ∧ r < 1 :=
by {
  -- Proof steps are omitted with sorry
  sorry
}

end ratio_A_B_l780_780741


namespace john_total_cost_l780_780034

def base_cost : ℤ := 25
def text_cost_per_message : ℤ := 8
def extra_minute_cost_per_minute : ℤ := 15
def international_minute_cost : ℤ := 100

def texts_sent : ℤ := 200
def total_hours : ℤ := 42
def international_minutes : ℤ := 10

-- Calculate the number of extra minutes
def extra_minutes : ℤ := (total_hours - 40) * 60

noncomputable def total_cost : ℤ :=
  base_cost +
  (texts_sent * text_cost_per_message) / 100 +
  (extra_minutes * extra_minute_cost_per_minute) / 100 +
  international_minutes * (international_minute_cost / 100)

theorem john_total_cost :
  total_cost = 69 := by
    sorry

end john_total_cost_l780_780034


namespace kyuyeon_drank_amount_l780_780426

-- Definitions for the total amount of apple juice and the difference Kyu-yeon drank
def total_juice : ℚ := 12 + (400 / 1000)
def extra_kyuyeon_drink : ℚ := 2 + (600 / 1000)

-- The amount Na-eun drank
def naeun_drink (total extra : ℚ) : ℚ := (total - extra) / 2

-- The amount Kyu-yeon drank
def kyuyeon_drink (total extra : ℚ) : ℚ := naeun_drink(total, extra) + extra

-- Lean 4 Proof Statement
theorem kyuyeon_drank_amount : kyuyeon_drink total_juice extra_kyuyeon_drink = 7.5 :=
by
  sorry

end kyuyeon_drank_amount_l780_780426


namespace distance_between_parallel_lines_l780_780488

/-- Given two parallel lines y=2x and y=2x+5, the distance between them is √5. -/
theorem distance_between_parallel_lines :
  let A := -2
  let B := 1
  let C1 := 0
  let C2 := -5
  let distance := (|C2 - C1|: ℝ) / Real.sqrt (A ^ 2 + B ^ 2)
  distance = Real.sqrt 5 := by
  -- Assuming calculations as done in the original solution
  sorry

end distance_between_parallel_lines_l780_780488


namespace largest_n_for_binomial_equality_l780_780537

theorem largest_n_for_binomial_equality :
  ∃ n : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 n ) ∧
  ∀ m : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 m ) → m ≤ 6 :=
begin
  sorry
end

end largest_n_for_binomial_equality_l780_780537


namespace calculate_exponent_product_l780_780068

theorem calculate_exponent_product :
  (2^0.5) * (2^0.3) * (2^0.2) * (2^0.1) * (2^0.9) = 4 :=
by
  sorry

end calculate_exponent_product_l780_780068


namespace intersection_of_sets_l780_780169

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780169


namespace intersection_S_T_l780_780320

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780320


namespace find_dot_product_l780_780770

variables (a b : ℝ × ℝ)
noncomputable def a' := (2, 1)
noncomputable def ab_relation := a - (2 • b) = (1, 1)
noncomputable def dot_product := a.1 * b.1 + a.2 * b.2

theorem find_dot_product :
  a = a' →
  ab_relation →
  dot_product = 1 :=
by
  sorry

end find_dot_product_l780_780770


namespace intersection_S_T_l780_780183

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780183


namespace largest_n_for_binom_equality_l780_780589

theorem largest_n_for_binom_equality : ∃ (n : ℕ), (∑ i : {i // i = 4 ∨ i = 5}, Nat.choose 10 i) = Nat.choose 11 n ∧ n = 6 :=
by
  sorry

end largest_n_for_binom_equality_l780_780589


namespace fraction_sum_possible_l780_780716

theorem fraction_sum_possible :
  ∃ a b c d e f g h i : ℕ,
    {a, b, c, d, e, f, g, h, i} = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
    g ≠ h ∧ g ≠ i ∧
    h ≠ i ∧
    (a.toRat / b.toRat) + (c.toRat / d.toRat) + (e.toRat / f.toRat) + (g.toRat / h.toRat) = i :=
sorry

end fraction_sum_possible_l780_780716


namespace angle_between_vectors_60_deg_l780_780324

theorem angle_between_vectors_60_deg
  (a b : ℝ)
  (h1 : (a + 3 * b) • (7 * a - 5 * b) = 0)
  (h2 : (a - 4 * b) • (7 * a - 2 * b) = 0) :
  real.angle ⟨a, b⟩ = real.pi / 3 :=
sorry

end angle_between_vectors_60_deg_l780_780324


namespace tangent_line_eqn_monotonicity_of_f_l780_780344

noncomputable def f (a x : ℝ) : ℝ := 1 / 3 * x^3 - 1 / 2 * a * x^2

theorem tangent_line_eqn (a x y : ℝ) (h : a = 2 ∧ x = 3 ∧ y = f 2 3) :
  3 * x - y - 9 = 0 := sorry

theorem monotonicity_of_f (a x : ℝ) :
  let f' := λ x, x * (x - a) in
  if h : a = 0 then 
    ∀ x1 x2, x1 ≤ x2 → f' x1 ≤ f' x2
  else if h : a < 0 then 
    ∀ x1 x2, (x1 ∈ Iio a ∨ x1 ∈ Ioi 0) ∧ (x2 ∈ Iio a ∨ x2 ∈ Ioi 0) ∧ x1 ≤ x2 → f' x1 ≤ f' x2 ∧ 
    ∀ x1 x2, (x1 ∈ Ioo a 0) ∧ (x2 ∈ Ioo a 0) ∧ x1 ≤ x2 → f' x1 > f' x2
  else 
    ∀ x1 x2, (x1 ∈ Iio 0 ∨ x1 ∈ Ioi a) ∧ (x2 ∈ Iio 0 ∨ x2 ∈ Ioi a) ∧ x1 ≤ x2 → f' x1 ≤ f' x2 ∧
    ∀ x1 x2, (x1 ∈ Ioo 0 a) ∧ (x2 ∈ Ioo 0 a) ∧ x1 ≤ x2 → f' x1 > f' x2 := sorry

end tangent_line_eqn_monotonicity_of_f_l780_780344


namespace alternating_squares_sum_l780_780074

theorem alternating_squares_sum :
  let N := ∑ i in range 51, if i % 3 == 0 then (3 * i)^2 else if i % 3 == 1 then -(3 * (50 - i))^2 else 0
  in N = 22950 :=
by
  sorry

end alternating_squares_sum_l780_780074


namespace count_not_divisible_by_3_5_7_l780_780361

theorem count_not_divisible_by_3_5_7 : 
  let A := { n : ℕ | n ≤ 120 ∧ n % 3 = 0 }
  let B := { n : ℕ | n ≤ 120 ∧ n % 5 = 0 }
  let C := { n : ℕ | n ≤ 120 ∧ n % 7 = 0 }
  (finset.range 121).filter (λ n, ¬(n ∈ A ∨ n ∈ B ∨ n ∈ C)).card = 54 :=
by
  sorry

end count_not_divisible_by_3_5_7_l780_780361


namespace largest_n_binom_identity_l780_780556

open Nat

theorem largest_n_binom_identity :
  (∃ n, binom 11 n = binom 10 4 + binom 10 5) ∧
  (∀ m, binom 11 m = binom 10 4 + binom 10 5 → m ≤ 6) :=
by
  sorry

end largest_n_binom_identity_l780_780556


namespace trapezoid_midsegment_inscribed_circle_l780_780498

theorem trapezoid_midsegment_inscribed_circle (P : ℝ) (hP : P = 40) 
    (inscribed : Π (a b c d : ℝ), a + b = c + d) : 
    (∃ (c d : ℝ), (c + d) / 2 = 10) :=
by
  sorry

end trapezoid_midsegment_inscribed_circle_l780_780498


namespace carol_maximizes_chance_of_winning_l780_780058

noncomputable def carol_optimal_choice : ℝ :=
0.725

theorem carol_maximizes_chance_of_winning :
  ∀ (a b d : ℝ), (0 ≤ a ∧ a ≤ 1) ∧ (0.25 ≤ b ∧ b ≤ 0.75) ∧ (0 ≤ d ∧ d ≤ 1) →
  (∃ c : ℝ, (0.6 ≤ c ∧ c ≤ 0.9) ∧ (c = 0.725) ∧
  (c > a ∧ c < b ∨ 
   c < a ∧ c > b) ∧
  (c > d ∨ c < d)) :=
sorry

end carol_maximizes_chance_of_winning_l780_780058


namespace football_team_progress_l780_780671

theorem football_team_progress (lost_yards gained_yards : Int) : lost_yards = -5 → gained_yards = 13 → lost_yards + gained_yards = 8 := 
by
  intros h_lost h_gained
  rw [h_lost, h_gained]
  sorry

end football_team_progress_l780_780671


namespace inequality_proof_l780_780781

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1)
                        (hb : 0 ≤ b) (hb1 : b ≤ 1)
                        (hc : 0 ≤ c) (hc1 : c ≤ 1) :
    (a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1) :=
by 
  sorry

end inequality_proof_l780_780781


namespace intersection_of_sets_l780_780170

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780170


namespace probability_is_one_sixth_l780_780385

noncomputable def probability_on_line : ℚ :=
  let A := {0, 1, 2, 3, 4, 5}
  let total_points := (A.product A).card
  let favorable_points := (A.filter (λ (a : ℕ), a ∈ A ∧ a ∈ A)).card
  favorable_points / total_points

theorem probability_is_one_sixth : probability_on_line = 1 / 6 := by
  sorry

end probability_is_one_sixth_l780_780385


namespace largest_n_binom_l780_780541

theorem largest_n_binom (n : ℕ) (h : 0 ≤ n ∧ n ≤ 11) : 
  (binomial 10 4 + binomial 10 5 = binomial 11 n) → n = 6 :=
by
  sorry

end largest_n_binom_l780_780541


namespace difference_of_numbers_l780_780004

-- Definitions for the digits and the numbers formed
def digits : List ℕ := [5, 3, 1, 4]

def largestNumber : ℕ := 5431
def leastNumber : ℕ := 1345

-- The problem statement
theorem difference_of_numbers (digits : List ℕ) (n_largest n_least : ℕ) :
  n_largest = 5431 ∧ n_least = 1345 → (n_largest - n_least) = 4086 :=
by
  sorry

end difference_of_numbers_l780_780004


namespace concurrency_of_lines_l780_780428

open_locale classical
noncomputable theory

variables {A B C H H_A H_B H_C K M T : Type}
variables (ABC : triangle A B C)
variables (K_on_AH_A : K ∈ line A H_A)    -- K on AH_A
variables (K_on_BM : K ∈ line B M)        -- K on BM
variables (T_on_BC : T ∈ line B C)        -- T on BC
variables (KT_parallel_AC : parallel KT AC) -- KT parallel to AC
variables (H_or : orthocenter H ABC)   -- H is the orthocenter
variables (AB_gt_BC : side_length A B > side_length B C) -- AB > BC

theorem concurrency_of_lines :
  concurrent (line H_C H_A) (line H T) (line A C) :=
sorry

end concurrency_of_lines_l780_780428


namespace largest_n_for_binom_equality_l780_780595

theorem largest_n_for_binom_equality : ∃ (n : ℕ), (∑ i : {i // i = 4 ∨ i = 5}, Nat.choose 10 i) = Nat.choose 11 n ∧ n = 6 :=
by
  sorry

end largest_n_for_binom_equality_l780_780595


namespace inscribed_not_possible_l780_780723

-- Definitions of convex polyhedron, colorings, and conditions
variables {P : Type} [Polyhedron P] (color : Π (f : Face P), Color)
  (adj : Π (e : Edge P), AdjacentFaces P e)
  (adj_blue : ∀ e : Edge P, (color (adj.left e) = Blue) → (color (adj.right e) ≠ Blue) )

-- More than half of faces are colored blue
variable (half_blue : ((countBlueFaces P color) > (numberFaces P / 2)))

theorem inscribed_not_possible (h : InscribedSphere P) : False :=
  sorry

end inscribed_not_possible_l780_780723


namespace james_total_matches_l780_780901

-- Define the conditions
def dozen : Nat := 12
def boxes_per_dozen : Nat := 5
def matches_per_box : Nat := 20

-- Calculate the expected number of matches
def expected_matches : Nat := boxes_per_dozen * dozen * matches_per_box

-- State the theorem to be proved
theorem james_total_matches : expected_matches = 1200 := by
  sorry

end james_total_matches_l780_780901


namespace total_emeralds_l780_780673

theorem total_emeralds (D R E : ℕ) 
  (h1 : 2 * D + 2 * E + 2 * R = 6)
  (h2 : R = D + 15) : 
  E = 12 :=
by
  -- Proof omitted
  sorry

end total_emeralds_l780_780673


namespace inequality_solution_inequality_proof_l780_780447

-- Definitions of the functions f and g
def f (x : ℝ) : ℝ := |x - 3|
def g (x : ℝ) : ℝ := |x - 2|

-- Definitions to be used in the Lean statements based on given conditions and questions

-- Question 1
theorem inequality_solution {x : ℝ} : f x + g x < 2 ↔ x ∈ set.Ioo (3 / 2) 2 ∨ x ∈ set.Icc 2 3 ∨ x ∈ set.Ioo 3 (7 / 2) :=
by 
  simp [f, g]
  sorry

-- Question 2
theorem inequality_proof (x y : ℝ) (h1 : f x ≤ 1) (h2 : g y ≤ 1) : |x - 2 * y + 1| ≤ 3 :=
by 
  simp [f, g]
  sorry

end inequality_solution_inequality_proof_l780_780447


namespace six_digit_ababab_divisible_by_101_l780_780432

theorem six_digit_ababab_divisible_by_101 (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) (h₃ : 0 ≤ b) (h₄ : b ≤ 9) :
  ∃ k : ℕ, 101 * k = 101010 * a + 10101 * b :=
sorry

end six_digit_ababab_divisible_by_101_l780_780432


namespace tangent_line_at_3_monotonicity_of_f_l780_780341

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2

def tangent_line_equation_at_point (f : ℝ → ℝ) (a : ℝ) (x0 : ℝ) : 
    ℝ × ℝ → Prop := 
  ∃ m b : ℝ, ∀ (p : ℝ× ℝ), p.1 = x0 → f x0 = p.2 ∧ p = (x0, f x0) ∧ m = deriv f x0
    ∧ b = p.snd - m * p.fst ∧ p ∈ set.range (λ (x : ℝ), (x, m * x + b))
   
theorem tangent_line_at_3 (a : ℝ) (f : ℝ → ℝ) : 
  f = λ x, (1/3) * x^3 - (1/2) * a * x^2 → a = 2 → 
  tangent_line_equation_at_point (λ x, f x) a 3 (3, 0) := 
  by
    intro hf ha 
    sorry

theorem monotonicity_of_f (a : ℝ) : 
  (∀ x, f x a = (1 / 3) * x ^ 3 - (1 / 2) * a * x ^ 2) → 
    (if a = 0 then ∀ x1 x2, x1 ≤ x2 → f x1 a ≤ f x2 a else 
      if a < 0 then ∀ x, 
        (x < a ∨ 0 < x → deriv f x a > 0) ∧ 
        (a < x → x < 0 → deriv f x a < 0) else 
          if a > 0 then ∀ x, 
            (x < 0 ∨ a < x → deriv f x a > 0) ∧ 
            (0 < x → x < a → deriv f x a < 0) ) := 
  by
    intro hf
    sorry

end tangent_line_at_3_monotonicity_of_f_l780_780341


namespace squares_to_nine_l780_780989

theorem squares_to_nine (x : ℤ) : x^2 = 9 ↔ x = 3 ∨ x = -3 :=
sorry

end squares_to_nine_l780_780989


namespace ratio_of_distances_l780_780806

noncomputable def ellipse := set_of (λ p : ℝ × ℝ, p.1^2 + p.2^2 = 1)
noncomputable def foci_1 := (-1, 0 : ℝ × ℝ)
noncomputable def foci_2 := (1, 0 : ℝ × ℝ)
noncomputable def line_l := set_of (λ p : ℝ × ℝ, p.1 + 2 * real.sqrt 3 * p.2 - 4 * real.sqrt 3 = 0)

theorem ratio_of_distances (P : ℝ × ℝ) (hP : P ∈ line_l) (hP1 : P ∈ ellipse) (hP2 : ∀ P', P' ∈ ellipse → ∠ foci_1 P' foci_2 ≤ ∠ foci_1 P foci_2) :
  dist P foci_1 / dist P foci_2 = real.sqrt 15 / 3 :=
sorry

end ratio_of_distances_l780_780806


namespace intersection_S_T_eq_T_l780_780140

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780140


namespace average_salary_jan_feb_march_april_is_8000_l780_780966

variables {avg_salary feb_march_april_may_avg may_salary jan_salary total_salary_feb_march_april total_salary_jan_feb_march_april : ℝ}

-- Given conditions
def condition1 : avg_salary = 8000 := sorry
def condition2 : feb_march_april_may_avg = 8700 := sorry
def condition3 : may_salary = 6500 := sorry
def condition4 : jan_salary = 3700 := sorry

-- Derived calculations
def total_salary_feb_march_april := feb_march_april_may_avg * 4 - may_salary
def total_salary_jan_feb_march_april := jan_salary + total_salary_feb_march_april
def average_jan_feb_march_april := total_salary_jan_feb_march_april / 4

-- Proof statement
theorem average_salary_jan_feb_march_april_is_8000 :
  average_jan_feb_march_april = 8000 :=
by 
  rw [total_salary_feb_march_april, condition2, condition3],
  rw [total_salary_jan_feb_march_april, condition4],
  sorry

end average_salary_jan_feb_march_april_is_8000_l780_780966


namespace valid_license_plates_possible_l780_780695

def num_letters : ℕ := 26
def num_symbols : ℕ := 5
def num_digits : ℕ := 10

theorem valid_license_plates_possible : num_letters * num_letters * num_symbols * num_digits^4 = 33_800_000 := by
  sorry

end valid_license_plates_possible_l780_780695


namespace intersection_S_T_l780_780184

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780184


namespace intersection_of_S_and_T_l780_780233

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780233


namespace intersection_eq_T_l780_780243

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780243


namespace intersection_S_T_l780_780192

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780192


namespace bridge_length_l780_780980

-- Definitions and conditions
def train_length : ℝ := 170
def train_speed_kmh : ℝ := 45
def crossing_time : ℝ := 30
def km_to_m (km : ℝ) : ℝ := km * 1000
def hr_to_s (hr : ℝ) : ℝ := hr * 3600
def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600

-- Problem statement to prove
theorem bridge_length :
  let total_distance := train_speed_ms * crossing_time in
  let bridge_length := total_distance - train_length in
  bridge_length = 205 := by
  sorry

end bridge_length_l780_780980


namespace largest_integer_comb_l780_780607

theorem largest_integer_comb (n: ℕ) (h1 : n = 6) : 
  (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) :=
by
  rw [h1]
  exact sorry

end largest_integer_comb_l780_780607


namespace expression_value_l780_780114

def quadratic_condition (x : ℝ) : Prop :=
  x^2 - 4 * x - 1 = 0

def original_expression (x : ℝ) : ℝ :=
  (x - 3) / (x - 4) - 1 / x

theorem expression_value
  (x : ℝ) 
  (hx : quadratic_condition x) 
  : original_expression x = 5 :=
sorry

end expression_value_l780_780114


namespace evaluate_expression_l780_780369

theorem evaluate_expression (x : ℝ) (h : x > 2) : |1 - |1 - x|| = x - 2 :=
by
  sorry

end evaluate_expression_l780_780369


namespace darius_initial_buckets_l780_780725

theorem darius_initial_buckets :
  ∃ (b₁ b₂ b₃ b₄ b₅ : ℕ),
  b₁ = 11 ∧ b₂ = 13 ∧ b₃ = 12 ∧ b₄ = 16 ∧ b₅ = 10 ∧
  (b₅ + b₂ = 23) ∧
  (b₁ + b₃ + b₄ = 39) ∧
  (∃ total_buckets : ℕ, total_buckets = 5) :=
begin
  sorry
end

end darius_initial_buckets_l780_780725


namespace AF_passes_through_incenter_l780_780383

variables {A B C D E F L M N P Q : Point} [EuclideanGeometry] [Triangle A B C]
          (mid_BC_L : Midpoint L B C) (mid_CA_M : Midpoint M C A) (mid_AB_N : Midpoint N A B)
          (D_on_BC : OnLine D B C) (E_on_AB : OnLine E A B)
          (AD_perimeter_bisects : Bisects AD (perimeter (Triangle A B C)))
          (CE_perimeter_bisects : Bisects CE (perimeter (Triangle A B C)))
          (P_symmetric : Symmetric P D L) (Q_symmetric : Symmetric Q E N)
          (intersect_PQ_LM_F : IntersectsAt PQ LM F)
          (AB_greater_AC : AB > AC)

theorem AF_passes_through_incenter : (PassesThrough A F (Incenter (Triangle A B C))) :=
by 
  sorry

end AF_passes_through_incenter_l780_780383


namespace intersection_of_sets_l780_780176

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780176


namespace car_speed_after_modifications_l780_780905

theorem car_speed_after_modifications (s : ℕ) (p : ℝ) (w : ℕ) :
  s = 150 →
  p = 0.30 →
  w = 10 →
  s + (p * s) + w = 205 := 
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  norm_num
  done

end car_speed_after_modifications_l780_780905


namespace starting_player_wins_by_taking_2_white_first_l780_780396

-- Define initial setup
def initial_blue_balls : ℕ := 15
def initial_white_balls : ℕ := 12

-- Define conditions of the game
def can_take_blue_balls (n : ℕ) : Prop := n % 3 = 0
def can_take_white_balls (n : ℕ) : Prop := n % 2 = 0
def player_win_condition (blue white : ℕ) : Prop := 
  (blue = 0 ∧ white = 0)

-- Define the game strategy to establish and maintain the ratio 3/2
def maintain_ratio (blue white : ℕ) : Prop := blue * 2 = white * 3

-- Prove that the starting player should take 2 white balls first to ensure winning
theorem starting_player_wins_by_taking_2_white_first :
  (can_take_white_balls 2) →
  maintain_ratio initial_blue_balls (initial_white_balls - 2) →
  ∀ (blue white : ℕ), player_win_condition blue white :=
by
  intros h_take_white h_maintain_ratio blue white
  sorry

end starting_player_wins_by_taking_2_white_first_l780_780396


namespace intersection_of_sets_l780_780179

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780179


namespace james_matches_l780_780898

theorem james_matches (dozen_boxes : ℕ) (matches_per_box : ℕ) (dozen_value : ℕ) 
  (h1 : dozen_boxes = 5) (h2 : matches_per_box = 20) (h3 : dozen_value = 12) :
  dozen_boxes * dozen_value * matches_per_box = 1200 :=
by
  rw [h1, h2, h3]
  calc
    5 * 12 * 20 = 60 * 20 := by norm_num
    ... = 1200 := by norm_num

end james_matches_l780_780898


namespace scientific_notation_of_distance_l780_780865

theorem scientific_notation_of_distance :
  ∃ a n, (1 ≤ a ∧ a < 10) ∧ 384000 = a * 10^n ∧ a = 3.84 ∧ n = 5 :=
by
  sorry

end scientific_notation_of_distance_l780_780865


namespace intersection_S_T_eq_T_l780_780218

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780218


namespace largest_n_binomial_l780_780560

theorem largest_n_binomial :
  (∃n : ℕ, binom 10 4 + binom 10 5 = binom 11 n) → (n ≤ 6) :=
by
  sorry

end largest_n_binomial_l780_780560


namespace problem_statement_l780_780111

-- Define the function f
def f (α : ℝ) : ℝ := (Real.sin (π + α) * Real.sin (α + π/2)) / (Real.cos (α - π/2))

-- Proof statement using Lean
theorem problem_statement (α : ℝ)
  (h1 : π < α ∧ α < 3 * π / 2) 
  (h2 : Real.cos (α + π / 2) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 :=
sorry

end problem_statement_l780_780111


namespace polygon_triangle_division_l780_780951

theorem polygon_triangle_division (n k : ℕ) (h : k * 3 = n * 3 - 6) : k ≥ n - 2 := sorry

end polygon_triangle_division_l780_780951


namespace largest_n_binom_l780_780546

theorem largest_n_binom (n : ℕ) (h : 0 ≤ n ∧ n ≤ 11) : 
  (binomial 10 4 + binomial 10 5 = binomial 11 n) → n = 6 :=
by
  sorry

end largest_n_binom_l780_780546


namespace intersection_of_equilateral_triangles_area_l780_780352

-- Defining two equilateral triangles
variables S : ℝ

-- Area calculation of intersection
noncomputable def intersection_area_of_rotated_equilateral_triangles (area : ℝ) : Prop :=
  ∃ (t1 t2 : triangle),
    is_equilateral t1 ∧ 
    is_equilateral t2 ∧
    area_of_triangle t1 = area ∧
    area_of_triangle t2 = area ∧
    rotate_triangle t1 30 t2 ∧
    area_of_intersection t1 t2 = area * (sqrt 3 - 1)

-- Now, we write the theorem to state the problem
theorem intersection_of_equilateral_triangles_area (S : ℝ) :
  ∃ (t1 t2 : triangle),
    is_equilateral t1 ∧ 
    is_equilateral t2 ∧
    area_of_triangle t1 = S ∧
    area_of_triangle t2 = S ∧
    rotate_triangle t1 30 t2 ∧
    area_of_intersection t1 t2 = S * (sqrt 3 - 1) :=
sorry

end intersection_of_equilateral_triangles_area_l780_780352


namespace cocyclic_points_ratio_l780_780697

open Real EuclideanGeometry

/- Defining the condition of cocyclic points and the intersection property -/
variables {A B C D E : Point} [Cyclic ABCD] (h_intersect : collinear A B E ∧ collinear C D E)

theorem cocyclic_points_ratio (h_cyclic : cyclic ABCD) : 
  (dist A C) / (dist B C) * (dist A D) / (dist B D) = (dist A E) / (dist B E) :=
by 
  sorry

end cocyclic_points_ratio_l780_780697


namespace abs_val_eq_two_l780_780850

theorem abs_val_eq_two (x : ℝ) (h : |x| = 2) : x = 2 ∨ x = -2 := 
sorry

end abs_val_eq_two_l780_780850


namespace simple_interest_rate_b_calc_l780_780830

def total_money := 10000
def money_lent_b : Float := 4000.0000000000005
def money_lent_a := total_money - money_lent_b
def time_years := 2
def simple_interest_rate_a := 0.15
def interest_diff := 360
def simple_interest_a := money_lent_a * simple_interest_rate_a * time_years

noncomputable def simple_interest_rate_b : Float := sorry

axiom si_a_gt_si_b_by : simple_interest_a = (money_lent_b * simple_interest_rate_b * time_years) + interest_diff

theorem simple_interest_rate_b_calc : simple_interest_rate_b = 0.18 := by
  sorry

end simple_interest_rate_b_calc_l780_780830


namespace probability_neither_l780_780371
-- Import the necessary Lean library for probability.

-- Define the events A and B and their probabilities.
variables {Ω : Type} [MeasurableSpace Ω] (μ : Measure Ω)
variables {A B : Set Ω}

-- Given Conditions
#check (μ(A) = 0.70) -- Probability that a student answered the first question correctly
#check (μ(B) = 0.55) -- Probability that a student answered the second question correctly
#check (μ(A ∩ B) = 0.45) -- Probability that a student answered both questions correctly

-- The theorem to prove
theorem probability_neither (hA : μ(A) = 0.70)
                           (hB : μ(B) = 0.55)
                           (hAB : μ(A ∩ B) = 0.45) :
    μ((A ∪ B)ᶜ) = 0.20 := 
by {
    -- The proof is omitted
    sorry
}

end probability_neither_l780_780371


namespace find_integers_l780_780097

theorem find_integers (a b m : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a + b^2) * (b + a^2) = 2^m → a = 1 ∧ b = 1 ∧ m = 2 :=
by
  sorry

end find_integers_l780_780097


namespace angle_DAE_l780_780409

variable (A B C D O E : Point)
variable [circle : Circle O A B C]
variable (hACB : ∠ A C B = 50)
variable (hCBA : ∠ C B A = 70)
variable (hD_perpendicular : ∃ H, ⟂ H D)
variable (hE_diameter : (∀ x, x ∈ diameter O ↔ x = A ∨ x = E))

theorem angle_DAE (A B C D O E : Point) 
  (hACB : ∠ A C B = 50) 
  (hCBA : ∠ C B A = 70) 
  (hD_perpendicular : D = foot_of_perpendicular A B C) 
  (hE_diameter : E = other_end_of_diameter_through A circle): 
  ∠ D A E = 20 := 
sorry

end angle_DAE_l780_780409


namespace determine_central_cell_value_l780_780947

noncomputable def sum (grid : Fin 5 × Fin 5 → ℕ) : ℕ :=
  ∑ i in Finset.univ, ∑ j in Finset.univ, grid (i, j)

theorem determine_central_cell_value (grid : Fin 5 × Fin 5 → ℕ)
  (sum_eq_500 : sum grid = 500)
  (query_sum : (Fin 5 × Fin 5) → ℕ)
  (query_sum_correct : ∀i j, query_sum (i, j) = grid (i, j) + 
    (if i > 0 then grid (i-1, j) else 0) + 
    (if i < 4 then grid (i+1, j) else 0) + 
    (if j > 0 then grid (i, j-1) else 0) + 
    (if j < 4 then grid (i, j+1) else 0)) :
  ∃ c, c = grid (2, 2) :=
sorry

end determine_central_cell_value_l780_780947


namespace inequality_with_means_l780_780935

theorem inequality_with_means (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := 
sorry

end inequality_with_means_l780_780935


namespace valid_votes_other_candidate_l780_780640

theorem valid_votes_other_candidate (total_votes : ℕ) (invalid_percentage : ℕ) (candidate1_percentage : ℕ) (valid_votes_other_candidate : ℕ) : 
  total_votes = 7500 → 
  invalid_percentage = 20 → 
  candidate1_percentage = 55 → 
  valid_votes_other_candidate = 2700 :=
by
  sorry

end valid_votes_other_candidate_l780_780640


namespace largest_integer_binom_l780_780581

theorem largest_integer_binom (n : ℕ) (h : nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) : n = 6 :=
sorry

end largest_integer_binom_l780_780581


namespace find_angle_ABC_l780_780649

noncomputable def point : Type := ℝ × ℝ -- just a simple definition for our purposes

-- Definitions for points A, B, and C
variables (A B C O : point)

-- Angles definition
noncomputable def angle (P Q R : point) : ℝ := sorry

-- Conditions
variable h1: angle B O C = 110
variable h2: angle A O B = 140

-- Theorem statement
theorem find_angle_ABC : ∀ {A B C O : point}, angle B O C = 110 ∧ angle A O B = 140 → angle A B C = 55 := by
  sorry

end find_angle_ABC_l780_780649


namespace train_speed_correct_l780_780693

/-- Define the train's length and the bridge's length and crossing time in seconds. -/
def train_length : ℝ := 120   -- in meters
def bridge_length : ℝ := 255  -- in meters
def crossing_time : ℝ := 30   -- in seconds

/-- Define the total distance traveled when crossing the bridge. -/
def total_distance : ℝ := train_length + bridge_length

/-- Define the speed in meters per second. -/
def speed_mps : ℝ := total_distance / crossing_time

/-- Define the conversion factor from meters per second (m/s) to kilometers per hour (km/hr). -/
def mps_to_kph : ℝ := 3.6

/-- Define the speed in kilometers per hour. -/
def speed_kph : ℝ := speed_mps * mps_to_kph

/-- Theorem: The speed of the train is 45 km/hr given the conditions. -/
theorem train_speed_correct : speed_kph = 45 := by
  sorry

end train_speed_correct_l780_780693


namespace original_speed_l780_780506

-- Given conditions
variables (length_highway length_original : ℝ) (speed_increase speed_original : ℝ) (time_reduction_factor : ℝ)

-- Conditions in the problem
def conditions := 
  length_highway = 200 ∧
  length_original = length_highway + 20 ∧
  speed_increase = 45 ∧
  time_reduction_factor = 0.5

-- The statement of the problem rephrased as a Lean theorem
theorem original_speed (length_highway length_original speed_increase speed_original : ℝ) (time_reduction_factor : ℝ) :
  conditions length_highway length_original speed_increase speed_original time_reduction_factor →
  speed_original = 55 :=
by
  sorry

end original_speed_l780_780506


namespace ratio_A_B_l780_780746

theorem ratio_A_B :
  let A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
  let B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)
  0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 :=
by
  let A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
  let B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)
  have hA : A = 1400 := rfl
  have hB : B = 1500 := rfl
  have h_ratio : A / B = 14 / 15 := by
    rw [hA, hB]
    norm_num
  exact And.intro (by norm_num) (by norm_num)

end ratio_A_B_l780_780746


namespace A_intersection_complement_B_is_empty_l780_780940

noncomputable def set_A : Set ℝ := {x : ℝ | sqrt (x - 2) ≤ 0}
def set_B : Set ℝ := {x : ℝ | 10^2 * 2 = 10^2}
def B_complement : Set ℝ := {x : ℝ | x ≠ -1 ∧ x ≠ 2}
def intersection_A_B_complement : Set ℝ := set_A ∩ B_complement

theorem A_intersection_complement_B_is_empty : intersection_A_B_complement = ∅ :=
sorry

end A_intersection_complement_B_is_empty_l780_780940


namespace S_inter_T_eq_T_l780_780306

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780306


namespace intersection_S_T_l780_780143

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780143


namespace sum_b_eq_4320_l780_780768

-- Define the function b(p) as per the conditions provided.
def b (p : ℕ) : ℕ :=
  let k := Nat.ceil (Real.cbrt (p : ℝ))
  if (k : ℝ) - Real.cbrt p < 0.5 then k else k - 1

-- Define the function T which sums b(p) from p = 1 to 1000
def T : ℕ := (Finset.range 1000).sum (λ p, b (p + 1))

-- State the theorem
theorem sum_b_eq_4320 : T = 4320 := 
by
   sorry

end sum_b_eq_4320_l780_780768


namespace handy_subsets_count_l780_780916

def is_handy (S : Finset ℕ) : Prop := (S.sum id) % 5 = 0

def count_handy_subsets (T : Finset ℕ) : ℕ := (T.powerset.filter is_handy).card

theorem handy_subsets_count :
  let T := (Finset.range 16).erase 0 in
  count_handy_subsets T = 6560 := by
  sorry

end handy_subsets_count_l780_780916


namespace find_a_extreme_value_l780_780438

noncomputable def f (a x : ℝ) : ℝ := ln (x + 1) - x - a * x

theorem find_a_extreme_value :
  (λ x, ln (x + 1) - x - a * x).diff (⟨1, rfl⟩) = 0 ↔ a = -1/2 := by
  sorry

end find_a_extreme_value_l780_780438


namespace largest_n_binom_equality_l780_780526

theorem largest_n_binom_equality :
  ∃ n : ℕ, (binomial 10 4 + binomial 10 5 = binomial 11 n) ∧ ∀ m : ℕ, (binomial 11 m = binomial 11 5) → m ≤ 6 :=
by
  sorry

end largest_n_binom_equality_l780_780526


namespace partition_number_10_elements_l780_780761

open Finset

noncomputable def num_partitions (n : ℕ) : ℕ :=
  (2^n - 2) / 2

theorem partition_number_10_elements :
  num_partitions 10 = 511 := 
by
  sorry

end partition_number_10_elements_l780_780761


namespace distance_from_focus_l780_780449

-- Definitions used in Lean 4 statement based on conditions from the problem
def parabola (x y : ℝ) : Prop := y^2 = 2*x
def point_on_parabola (P : ℝ × ℝ) : Prop := parabola (P.fst) (P.snd) ∧ P.fst = 2
def focus := (1 / 2 : ℝ, 0 : ℝ)

-- Theorem statement
theorem distance_from_focus (P : ℝ × ℝ) (h : point_on_parabola P) : 
  real.dist P focus = 5 / 2 :=
sorry

end distance_from_focus_l780_780449


namespace intersection_S_T_eq_T_l780_780256

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780256


namespace polynomial_inequality_l780_780990

theorem polynomial_inequality
  (a b c : ℝ)
  (hP : ∃ x1 x2 x3 : ℝ, P = λ x, (x - x1) * (x - x2) * (x - x3))
  (hQ_no_real_roots : ∀ x : ℝ, (x^2 + x + 2001 - x1) * (x^2 + x + 2001 - x2) * (x^2 + x + 2001 - x3) ≠ 0) :
  P 2001 > 1 / 64 :=
by
  sorry

end polynomial_inequality_l780_780990


namespace min_total_balls_l780_780510

theorem min_total_balls (R G B : Nat) (hG : G = 12) (hRG : R + G < 24) : 23 ≤ R + G + B :=
by {
  sorry
}

end min_total_balls_l780_780510


namespace beautifulEquations_1_find_n_l780_780379

def isBeautifulEquations (eq1 eq2 : ℝ → Prop) : Prop :=
  ∃ x y : ℝ, eq1 x ∧ eq2 y ∧ x + y = 1

def eq1a (x : ℝ) : Prop := 4 * x - (x + 5) = 1
def eq2a (y : ℝ) : Prop := -2 * y - y = 3

theorem beautifulEquations_1 : isBeautifulEquations eq1a eq2a :=
sorry

def eq1b (x : ℝ) (n : ℝ) : Prop := 2 * x - n + 3 = 0
def eq2b (x : ℝ) (n : ℝ) : Prop := x + 5 * n - 1 = 0

theorem find_n (n : ℝ) : (∀ x1 x2 : ℝ, eq1b x1 n ∧ eq2b x2 n ∧ x1 + x2 = 1) → n = -1 / 3 :=
sorry

end beautifulEquations_1_find_n_l780_780379


namespace problem_f_neg2_equals_2_l780_780801

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem problem_f_neg2_equals_2 (f : ℝ → ℝ) (b : ℝ) 
  (h_odd : is_odd_function f)
  (h_def : ∀ x : ℝ, x ≥ 0 → f x = x^2 - 3 * x + b) 
  (h_b : b = 0) :
  f (-2) = 2 :=
by
  sorry

end problem_f_neg2_equals_2_l780_780801


namespace expression_equals_5776_l780_780643

-- Define constants used in the problem
def a : ℕ := 476
def b : ℕ := 424
def c : ℕ := 4

-- Define the expression using the constants
def expression : ℕ := (a + b) ^ 2 - c * a * b

-- The target proof statement
theorem expression_equals_5776 : expression = 5776 := by
  sorry

end expression_equals_5776_l780_780643


namespace perpendicular_lines_condition_l780_780496

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * (m + 1) * x + (m - 3) * y + 7 - 5 * m = 0) ↔ (∀ x y : ℝ, (m - 3) * x + 2 * y - 5 = 0) →
  (m = 3 ∨ m = -2) :=
sorry

end perpendicular_lines_condition_l780_780496


namespace polynomial_evaluation_l780_780930

theorem polynomial_evaluation (p : Polynomial ℚ) 
  (hdeg : p.degree = 7)
  (h : ∀ n : ℕ, n ≤ 7 → p.eval (2^n) = 1 / 2^(n + 1)) : 
  p.eval 0 = 255 / 2^28 := 
sorry

end polynomial_evaluation_l780_780930


namespace intersection_S_T_l780_780312

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780312


namespace largest_integer_binom_l780_780586

theorem largest_integer_binom (n : ℕ) (h : nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) : n = 6 :=
sorry

end largest_integer_binom_l780_780586


namespace northton_time_capsule_depth_l780_780476

def southton_depth : ℕ := 15

def northton_depth : ℕ := 4 * southton_depth + 12

theorem northton_time_capsule_depth : northton_depth = 72 := by
  sorry

end northton_time_capsule_depth_l780_780476


namespace polynomial_solution_l780_780081

theorem polynomial_solution (P : ℝ → ℝ) :
  (∀ a b c : ℝ, (a * b + b * c + c * a = 0) → 
  (P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c))) →
  ∃ α β : ℝ, ∀ x : ℝ, P x = α * x ^ 4 + β * x ^ 2 :=
by
  sorry

end polynomial_solution_l780_780081


namespace exists_pos_coeff_polynomial_l780_780936

variable (g : Polynomial ℤ)

-- Conditions
def g_in_Zx (g : Polynomial ℤ) : Prop := true -- This is always true because g : Polynomial ℤ

def no_nonneg_real_roots (g : Polynomial ℤ) : Prop :=
  ∀ r : ℝ, Polynomial.eval (r : ℤ) g = 0 → r < 0

-- Main Statement
theorem exists_pos_coeff_polynomial (hg_in_Zx : g_in_Zx g) (hg_no_nonneg_roots : no_nonneg_real_roots g) :
  ∃ h : Polynomial ℤ, ∀ x : ℤ, Polynomial.eval x (g * h) ≥ 0 :=
sorry

end exists_pos_coeff_polynomial_l780_780936


namespace sequence_sum_l780_780121

theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, S n = n^2 * a n) :
  ∀ n : ℕ, S n = 2 * n / (n + 1) := 
by 
  sorry

end sequence_sum_l780_780121


namespace intersection_eq_T_l780_780289

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780289


namespace ellipse_equation_and_area_l780_780787

theorem ellipse_equation_and_area 
  (a b c : ℝ) (D : ℝ × ℝ)
  (h_ellipse_eq : ∀ (x y : ℝ), (x^2)/(a^2) + (y^2)/(b^2) = 1)
  (h_a_gt_b : a > b)
  (h_b_gt_0 : b > 0)
  (h_eccentricity : c = a * (sqrt 3)/2)
  (h_perimeter : ∀ (M : ℝ × ℝ), ∃ F1 F2 : ℝ × ℝ, F1 = (-c, 0) ∧ F2 = (c, 0) ∧ M ∈ {p : ℝ × ℝ | (p.1^2)/(a^2) + (p.2^2)/(b^2) = 1} ∧ p ≠ (a, 0) ∧ p ≠ (-a, 0) → dist M F1 + dist M F2 = 4 + 2 * sqrt 3)
  (H_line : ∀ (k : ℝ), (D.2 = k * D.1 - 2))
  :
  (a = 2) ∧ (b^2 = a^2 - c^2 ∧ b^2 = 1) ∧ (∀ (x y : ℝ), (x^2)/4 + y^2 = 1) ∧
  (∃ (k : ℝ), k^2 > 3/4 ∧ 8 * sqrt ((4 * k^2 - 3)/(1 + 4 * k^2)^2) = 2 ∧ 
  ((H_line k) = (H_line (sqrt 7)/2) ∨ (H_line (- sqrt 7 / 2))))
:= sorry

end ellipse_equation_and_area_l780_780787


namespace platform_length_is_350_l780_780023

variables (L : ℕ)

def train_length := 300
def time_to_cross_pole := 18
def time_to_cross_platform := 39

-- Speed of the train when crossing the pole
def speed_cross_pole : ℚ := train_length / time_to_cross_pole

-- Speed of the train when crossing the platform
def speed_cross_platform (L : ℕ) : ℚ := (train_length + L) / time_to_cross_platform

-- The main goal is to prove that the length of the platform is 350 meters
theorem platform_length_is_350 (L : ℕ) (h : speed_cross_pole = speed_cross_platform L) : L = 350 := sorry

end platform_length_is_350_l780_780023


namespace largest_n_binom_l780_780540

theorem largest_n_binom (n : ℕ) (h : 0 ≤ n ∧ n ≤ 11) : 
  (binomial 10 4 + binomial 10 5 = binomial 11 n) → n = 6 :=
by
  sorry

end largest_n_binom_l780_780540


namespace intersection_S_T_eq_T_l780_780265

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780265


namespace fraction_neg_range_l780_780863

theorem fraction_neg_range (x : ℝ) : (x ≠ 0 ∧ x < 1) ↔ (x - 1 < 0 ∧ x^2 > 0) := by
  sorry

end fraction_neg_range_l780_780863


namespace term_3025_is_37_l780_780975

-- Define a function to calculate the sum of the squares of the digits of a number
def sumOfSquaresOfDigits (n : Nat) : Nat := 
  n.digits.foldr (λ d acc => d * d + acc) 0

-- Define the sequence
noncomputable def sequence : Nat → Nat
| 0 => 3025
| (n + 1) => sumOfSquaresOfDigits (sequence n)

-- Prove that the 3025th term of the sequence is 37
theorem term_3025_is_37 : sequence 3025 = 37 :=
by
  sorry

end term_3025_is_37_l780_780975


namespace coefficient_of_x_term_l780_780840

noncomputable def m : ℝ :=
  ∫ x in 0..(Real.pi/2), (Real.sqrt 2) * Real.sin (x + (Real.pi/4))

theorem coefficient_of_x_term :
  m = 2 → 
  let binom := (fun x : ℝ => (Real.sqrt x - 2 / (Real.sqrt x)) ^ 6) in
  ∃ (c : ℝ), 
    (∃ r, r = 2 ∧ c = (-2)^r * Nat.choose 6 r) ∧
    3 - r = 1 ∧
    binom = (fun x : ℝ => c * x) .

end coefficient_of_x_term_l780_780840


namespace intersection_S_T_eq_T_l780_780270

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780270


namespace intersection_S_T_eq_T_l780_780214

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780214


namespace largest_integer_comb_l780_780601

theorem largest_integer_comb (n: ℕ) (h1 : n = 6) : 
  (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) :=
by
  rw [h1]
  exact sorry

end largest_integer_comb_l780_780601


namespace largest_n_binomial_l780_780568

theorem largest_n_binomial :
  (∃n : ℕ, binom 10 4 + binom 10 5 = binom 11 n) → (n ≤ 6) :=
by
  sorry

end largest_n_binomial_l780_780568


namespace largest_integer_binom_l780_780612

theorem largest_integer_binom
  (n : ℕ)
  (h₁ : binom 10 4 + binom 10 5 = binom 11 n) :
  n = 6 :=
sorry

end largest_integer_binom_l780_780612


namespace ratio_A_B_l780_780739

/-- Definition of A and B from the problem. -/
def A : ℕ :=
  1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28

def B : ℕ :=
  1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

/-- Proof of the inequality 0 < A / B < 1 given the definitions of A and B. -/
theorem ratio_A_B (hA : A = 1400) (hB : B = 1500) : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 :=
by
  rw [hA, hB]
  norm_num
  sorry

end ratio_A_B_l780_780739


namespace most_accurate_value_l780_780388

variable (D : ℝ)
variable (uncertainty : ℝ)

-- Given conditions
def givenD : D = 3.78249 := sorry
def givenUncertainty : uncertainty = 0.00295 := sorry

-- Prove that the most accurate value that can be announced is 3.8
theorem most_accurate_value (D = 3.78249) (uncertainty = 0.00295) : 
  rounded_to_tenth (D + uncertainty) = 3.8 ∧ rounded_to_tenth (D - uncertainty) = 3.8 :=
sorry

end most_accurate_value_l780_780388


namespace cost_price_of_book_l780_780030

-- Variables declaration
variables (C : ℝ) (SP_10 SP_15 : ℝ)

-- Conditions
def SP_10_condition := SP_10 = 1.10 * C
def SP_15_condition := SP_15 = 1.15 * C
def SP_relation := SP_15 = SP_10 + 100

-- Theorem statement
theorem cost_price_of_book (h1 : SP_10_condition C SP_10)
                           (h2 : SP_15_condition C SP_15)
                           (h3 : SP_relation C SP_10 SP_15) 
                           : C = 2000 :=
sorry

end cost_price_of_book_l780_780030


namespace diagonal_difference_is_4_l780_780655

def original_matrix : Matrix (Fin 4) (Fin 4) ℕ := 
  ![![1, 2, 3, 4], 
    ![8, 9, 10, 11], 
    ![15, 16, 17, 18], 
    ![22, 23, 24, 25]]

def reversed_matrix : Matrix (Fin 4) (Fin 4) ℕ := 
  ![![1, 2, 3, 4], 
    ![11, 10, 9, 8], 
    ![15, 16, 17, 18], 
    ![25, 24, 23, 22]]

def diagonal_sum (m : Matrix (Fin 4) (Fin 4) ℕ) : ℕ := 
  m 0 0 + m 1 1 + m 2 2 + m 3 3

def anti_diagonal_sum (m : Matrix (Fin 4) (Fin 4) ℕ) : ℕ := 
  m 0 3 + m 1 2 + m 2 1 + m 3 0

theorem diagonal_difference_is_4 : 
  |(diagonal_sum reversed_matrix) - (anti_diagonal_sum reversed_matrix)| = 4 :=
by
  sorry

end diagonal_difference_is_4_l780_780655


namespace intersection_eq_T_l780_780286

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780286


namespace largest_n_binomial_l780_780575

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l780_780575


namespace min_x_plus_y_l780_780842

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y ≥ 9 := by
  sorry

end min_x_plus_y_l780_780842


namespace find_repair_charge_l780_780073

theorem find_repair_charge
    (cost_oil_change : ℕ)
    (cost_car_wash : ℕ)
    (num_oil_changes : ℕ)
    (num_repairs : ℕ)
    (num_car_washes : ℕ)
    (total_earnings : ℕ)
    (R : ℕ) :
    (cost_oil_change = 20) →
    (cost_car_wash = 5) →
    (num_oil_changes = 5) →
    (num_repairs = 10) →
    (num_car_washes = 15) →
    (total_earnings = 475) →
    5 * cost_oil_change + 10 * R + 15 * cost_car_wash = total_earnings →
    R = 30 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end find_repair_charge_l780_780073


namespace minimize_G_at_half_l780_780364

def F (p q : ℝ) : ℝ :=
  -3 * p * q + 4 * p * (1 - q) + 2 * (1 - p) * q - 5 * (1 - p) * (1 - q)

def G (p : ℝ) : ℝ :=
  max (F p 0) (F p 1)

theorem minimize_G_at_half : ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧ G p = G (1 / 2) :=
begin
  use 1 / 2,
  split ; norm_num,
  split ; linarith,
  sorry
end

end minimize_G_at_half_l780_780364


namespace cookies_with_five_cups_l780_780699

theorem cookies_with_five_cups (cookies_per_four_cups : ℕ) (flour_for_four_cups : ℕ) (flour_for_five_cups : ℕ) (h : 24 / 4 = cookies_per_four_cups / 5) :
  cookies_per_four_cups = 30 :=
by
  sorry

end cookies_with_five_cups_l780_780699


namespace largest_n_for_binom_equality_l780_780592

theorem largest_n_for_binom_equality : ∃ (n : ℕ), (∑ i : {i // i = 4 ∨ i = 5}, Nat.choose 10 i) = Nat.choose 11 n ∧ n = 6 :=
by
  sorry

end largest_n_for_binom_equality_l780_780592


namespace reciprocal_of_complex_on_circle_l780_780885

def reciprocal_in_fourth_quadrant (k : ℂ) (G : ℂ) : Prop :=
  G = k + k * complex.I ∧ complex.abs G = 2 ∧
  (1/G).re > 0 ∧ (1/G).im < 0

theorem reciprocal_of_complex_on_circle {k : ℂ} (h : complex.abs (k + k * complex.I) = 2) :
  reciprocal_in_fourth_quadrant k (k + k * complex.I) :=
by
  sorry

end reciprocal_of_complex_on_circle_l780_780885


namespace quotient_56_div_k_is_4_l780_780762

theorem quotient_56_div_k_is_4 (k : ℕ) (h : k = 14) : 56 / k = 4 :=
by
  rw [h]
  exact Nat.div_eq_of_eq_mul_right (Nat.succ_pos 13) rfl

end quotient_56_div_k_is_4_l780_780762


namespace hyperbola_eccentricity_l780_780327

open Real

variables (a b t : ℝ) (F1 F2 A : Point)
-- Define the hyperbola
def hyperbola (a b : ℝ) (P : Point) : Prop :=
  let (x, y) := P in (x * x) / (a * a) - (y * y) / (b * b) = 1

-- Define points F1 and F2
def foci (F1 F2 : Point) (a : ℝ) : Prop :=
  let (x1, y1) := F1
      (x2, y2) := F2 in
  x1 = -c ∧ x2 = c ∧ y1 = 0 ∧ y2 = 0 where c := ∀ t > 0, sqrt 5  * t / 2 

-- Define angle condition
def angle_condition (F1 F2 A : Point) : Prop :=
  ∠ F1AF2 = 90

-- Define distance condition
def distance_condition (A F1 F2 : Point) : Prop :=
  dist A F1 = 2 * dist A F2

-- Eccentricity
def eccentricity (a c : ℝ) : ℝ := c / a

-- Eccentricity theorem
theorem hyperbola_eccentricity (h1 : hyperbola a b A) 
                               (h2 : foci F1 F2 a)
                               (h3 : angle_condition F1 F2 A)
                               (h4 : distance_condition A F1 F2) : 
  eccentricity a (sqrt 5 * t / 2) = sqrt 5 :=
by
  sorry 

end hyperbola_eccentricity_l780_780327


namespace intersection_S_T_l780_780195

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780195


namespace intersection_S_T_eq_T_l780_780127

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780127


namespace polygon_divided_into_5_triangles_l780_780853

-- Define a theorem to state the problem and expected result
theorem polygon_divided_into_5_triangles (n : ℕ) 
  (h : ∃ v, n-sided_polygon_divided_from_vertex v = 5) : n = 7 :=
sorry

end polygon_divided_into_5_triangles_l780_780853


namespace terminal_side_in_second_quadrant_l780_780796

theorem terminal_side_in_second_quadrant (α : ℝ) 
  (hcos : Real.cos α = -1/5) 
  (hsin : Real.sin α = 2 * Real.sqrt 6 / 5) : 
  (π / 2 < α ∧ α < π) :=
by
  sorry

end terminal_side_in_second_quadrant_l780_780796


namespace six_digit_repetend_of_3_over_13_l780_780755

noncomputable def repetend_of_fraction (n : ℚ) (k : ℕ) : list ℕ := sorry

theorem six_digit_repetend_of_3_over_13 :
  repetend_of_fraction (3 / 13) 6 = [2, 3, 0, 7, 6, 9] :=
sorry

end six_digit_repetend_of_3_over_13_l780_780755


namespace exam_combinations_l780_780867

/-- In the "$3+1+2$" examination plan in Hubei Province, 2021,
there are three compulsory subjects: Chinese, Mathematics, and English.
Candidates must choose one subject from Physics and History.
Candidates must choose two subjects from Chemistry, Biology, Ideological and Political Education, and Geography.
Prove that the total number of different combinations of examination subjects is 12.
-/
theorem exam_combinations : exists n : ℕ, n = 12 :=
by
  have compulsory_choice := 1
  have physics_history_choice := 2
  have remaining_subjects_choice := Nat.choose 4 2
  exact Exists.intro (compulsory_choice * physics_history_choice * remaining_subjects_choice) sorry

end exam_combinations_l780_780867


namespace correct_calculation_l780_780626

theorem correct_calculation : 
  ¬(2 * Real.sqrt 3 + 3 * Real.sqrt 2 = 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  ¬(5 * Real.sqrt 3 * 5 * Real.sqrt 2 = 5 * Real.sqrt 6) ∧
  ¬(Real.sqrt (4 + 1 / 2) = 2 * Real.sqrt (1 / 2)) :=
by {
  -- Using the conditions to prove the correct option B
  sorry
}

end correct_calculation_l780_780626


namespace trapezium_area_correct_l780_780001

def area_of_trapezium (a b h : ℝ) : ℝ :=
  1 / 2 * (a + b) * h

theorem trapezium_area_correct {a b h : ℝ} (ha : a = 20) (hb : b = 15) (hh : h = 14) :
  area_of_trapezium a b h = 245 :=
by
  rw [ha, hb, hh]
  simp [area_of_trapezium]
  norm_num
  sorry

end trapezium_area_correct_l780_780001


namespace part_a_part_b_l780_780952

-- Part (a): Proving that 111...1 (12 consecutive 1s) is divisible by 13
theorem part_a (n : ℕ) (h : n = 12) : 
  (∑ i in finset.range n, 10^i) % 13 = 0 :=
by {
  -- Use the hypothesis that n = 12
  rw h,
  sorry
}

-- Part (b): Proving that 111...1 (16 consecutive 1s) is divisible by 17
theorem part_b (n : ℕ) (h : n = 16) : 
  (∑ i in finset.range n, 10^i) % 17 = 0 :=
by {
  -- Use the hypothesis that n = 16
  rw h,
  sorry
}

end part_a_part_b_l780_780952


namespace S_inter_T_eq_T_l780_780301

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780301


namespace intersection_S_T_l780_780310

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780310


namespace intersection_S_T_eq_T_l780_780217

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780217


namespace intersection_S_T_eq_T_l780_780209

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780209


namespace largest_integer_comb_l780_780608

theorem largest_integer_comb (n: ℕ) (h1 : n = 6) : 
  (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) :=
by
  rw [h1]
  exact sorry

end largest_integer_comb_l780_780608


namespace fraction_meaningful_domain_l780_780856

theorem fraction_meaningful_domain (x : ℝ) : (∃ f : ℝ, f = 1 / (x - 2)) → x ≠ 2 :=
by
  sorry

end fraction_meaningful_domain_l780_780856


namespace ratio_A_B_l780_780737

/-- Definition of A and B from the problem. -/
def A : ℕ :=
  1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28

def B : ℕ :=
  1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

/-- Proof of the inequality 0 < A / B < 1 given the definitions of A and B. -/
theorem ratio_A_B (hA : A = 1400) (hB : B = 1500) : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 :=
by
  rw [hA, hB]
  norm_num
  sorry

end ratio_A_B_l780_780737


namespace largest_n_binom_l780_780544

theorem largest_n_binom (n : ℕ) (h : 0 ≤ n ∧ n ≤ 11) : 
  (binomial 10 4 + binomial 10 5 = binomial 11 n) → n = 6 :=
by
  sorry

end largest_n_binom_l780_780544


namespace celebrity_baby_photo_probability_l780_780667

theorem celebrity_baby_photo_probability : 
  let total_arrangements := Nat.factorial 4
  let correct_arrangements := 1
  let probability := correct_arrangements / total_arrangements
  probability = 1/24 :=
by
  sorry

end celebrity_baby_photo_probability_l780_780667


namespace largest_n_for_binomial_equality_l780_780535

theorem largest_n_for_binomial_equality :
  ∃ n : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 n ) ∧
  ∀ m : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 m ) → m ≤ 6 :=
begin
  sorry
end

end largest_n_for_binomial_equality_l780_780535


namespace S_inter_T_eq_T_l780_780308

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780308


namespace intersection_S_T_eq_T_l780_780278

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780278


namespace james_matches_l780_780896

theorem james_matches (dozen_boxes : ℕ) (matches_per_box : ℕ) (dozen_value : ℕ) 
  (h1 : dozen_boxes = 5) (h2 : matches_per_box = 20) (h3 : dozen_value = 12) :
  dozen_boxes * dozen_value * matches_per_box = 1200 :=
by
  rw [h1, h2, h3]
  calc
    5 * 12 * 20 = 60 * 20 := by norm_num
    ... = 1200 := by norm_num

end james_matches_l780_780896


namespace difference_in_owed_amount_l780_780956

-- We define all conditions as Lean definitions.
def principal : ℝ := 8000
def annual_rate : ℝ := 0.10
def time_years : ℕ := 3

-- Define the semi-annual and annual compounding functions.
def amount_owed (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- Define specific amounts for semi-annual and annual compounding.
def amount_semi_annual : ℝ := amount_owed principal annual_rate 2 time_years
def amount_annual : ℝ := amount_owed principal annual_rate 1 time_years

-- Define the proof problem.
theorem difference_in_owed_amount : amount_semi_annual - amount_annual = 72.79 := by
  sorry

end difference_in_owed_amount_l780_780956


namespace intersection_eq_T_l780_780156

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780156


namespace largest_n_binomial_l780_780559

theorem largest_n_binomial :
  (∃n : ℕ, binom 10 4 + binom 10 5 = binom 11 n) → (n ≤ 6) :=
by
  sorry

end largest_n_binomial_l780_780559


namespace count_satisfying_n_l780_780358

def isPerfectSquare (m : ℤ) : Prop :=
  ∃ k : ℤ, m = k * k

def countNsatisfying (low high : ℤ) (e : ℤ → ℤ) : ℤ :=
  (Finset.range (Int.natAbs (high - low + 1))).count (λ i, isPerfectSquare (e (low + i)))

theorem count_satisfying_n : countNsatisfying 5 15 (λ n, 2 * n^2 + 3 * n + 2) = 1 :=
by
  sorry

end count_satisfying_n_l780_780358


namespace intersection_S_T_l780_780144

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780144


namespace intersection_of_S_and_T_l780_780230

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780230


namespace equilateral_triangle_area_percentage_is_correct_l780_780046

noncomputable def side_length : ℝ := sorry -- side length of both the square and the equilateral triangle

-- Area of the equilateral triangle
noncomputable def area_triangle : ℝ := (sqrt 3 / 4) * side_length^2

-- Area of the square
noncomputable def area_square : ℝ := side_length^2

-- Total area of the pentagon
noncomputable def area_pentagon : ℝ := area_square + area_triangle

-- Percentage of the pentagon's area that is the area of the equilateral triangle
noncomputable def percentage : ℝ := (area_triangle / area_pentagon) * 100

theorem equilateral_triangle_area_percentage_is_correct :
  percentage = (100 * sqrt 3) / (4 + sqrt 3) := sorry

end equilateral_triangle_area_percentage_is_correct_l780_780046


namespace red_paint_cans_l780_780945

theorem red_paint_cans (total_cans : ℕ) (ratio_red_blue : ℕ) (ratio_blue : ℕ) (h_ratio : ratio_red_blue = 4) (h_blue : ratio_blue = 1) (h_total_cans : total_cans = 50) : 
  (total_cans * ratio_red_blue) / (ratio_red_blue + ratio_blue) = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end red_paint_cans_l780_780945


namespace number_of_pairs_l780_780115

theorem number_of_pairs (U : Set ℕ) (A B : Set ℕ) (hU : U = {1, 2, 3}) (hUnion : A ∪ B = U) (hDiff : A ≠ B) :
  @Finset.pair_eq {1, 2, 3} (A \ B) (B \ A) =
  26 :=
sorry

end number_of_pairs_l780_780115


namespace cos_3theta_l780_780835

theorem cos_3theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 := by
  sorry

end cos_3theta_l780_780835


namespace volume_of_one_part_equal_l780_780690

open Real
open Nat

-- Define the circumference condition
def circumference (r : ℝ) : Prop := 2 * π * r = 16 * π

-- Define the volume of a sphere
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

-- Define the volume of one part (one octant)
def volume_of_one_octant (r : ℝ) : ℝ := volume_of_sphere r / 8

-- The theorem to be proven: Given the circumference, find the volume of one octant
theorem volume_of_one_part_equal (r : ℝ) (h : circumference r) : volume_of_one_octant r = (256 / 3) * π :=
  sorry

end volume_of_one_part_equal_l780_780690


namespace max_difference_in_volume_l780_780003

noncomputable def computed_volume (length width height : ℕ) : ℕ :=
  length * width * height

noncomputable def max_possible_volume (length width height : ℕ) (error : ℕ) : ℕ :=
  (length + error) * (width + error) * (height + error)

theorem max_difference_in_volume :
  ∀ (length width height error : ℕ), length = 150 → width = 150 → height = 225 → error = 1 → 
  max_possible_volume length width height error - computed_volume length width height = 90726 :=
by
  intros length width height error h_length h_width h_height h_error
  rw [h_length, h_width, h_height, h_error]
  simp only [computed_volume, max_possible_volume]
  -- Intermediate calculations
  sorry

end max_difference_in_volume_l780_780003


namespace lavender_punch_parts_l780_780915

theorem lavender_punch_parts (total_volume : ℕ) (ratio : ℕ) (additional_juice : ℕ) :
  total_volume = 72 →
  ratio = 5 →
  additional_juice = 18 →
  let parts_of_lemonade := total_volume / (1 + ratio) in
  parts_of_lemonade = 12 :=
begin
  intros hvol hratio hadd,
  set parts_of_lemonade := total_volume / (1 + ratio),
  have h : parts_of_lemonade = 72 / 6, from sorry,
  rw [h],
  norm_num,
end

end lavender_punch_parts_l780_780915


namespace stirling_part_1_stirling_part_2_stirling_part_3_stirling_part_4_stirling_part_5_l780_780954

open Nat

-- Definitions and statements as needed for the Lean proof
def stirling (n k : ℕ) : ℕ := sorry

-- Part (i)
theorem stirling_part_1 (n : ℕ) : stirling n 2 = 2^(n - 1) - 1 := sorry

-- Part (ii)
theorem stirling_part_2 (n : ℕ) : stirling n (⌊π⌋ - 1) = nat.choose n 2 := sorry

-- Part (iii)
theorem stirling_part_3 (n k : ℕ) : stirling (n + 1) k = stirling n (k - 1) + k * stirling n k := sorry

-- Part (iv)
theorem stirling_part_4 (n k : ℕ) : 
  stirling (n + 1) k = ∑ j in finset.range (n + 1), if j ≥ k - 1 then (nat.choose n j) * stirling j (k - 1) else 0 :=
sorry

-- Part (v)
theorem stirling_part_5 (n : ℕ) (hn : 2 ≤ n) : 
  ∑ k in finset.range (n + 1), (-1) ^ (k - 1) * fact k * stirling n k = 0 :=
sorry

end stirling_part_1_stirling_part_2_stirling_part_3_stirling_part_4_stirling_part_5_l780_780954


namespace evaluate_f_2003_l780_780814

def f (x : ℝ) (a α β : ℝ) : ℝ := a * sin (π * x + α) + b * cos (π * x + β)

theorem evaluate_f_2003 {a α β : ℝ}
  (h₁ : f 2002 a α β = 3) :
  f 2003 a α β = -3 := 
sorry

end evaluate_f_2003_l780_780814


namespace intersection_S_T_l780_780319

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780319


namespace karl_total_miles_l780_780424

def car_mileage_per_gallon : ℕ := 30
def full_tank_gallons : ℕ := 14
def initial_drive_miles : ℕ := 300
def gas_bought_gallons : ℕ := 10
def final_tank_fraction : ℚ := 1 / 3

theorem karl_total_miles (initial_fuel : ℕ) :
  initial_fuel = full_tank_gallons →
  (initial_drive_miles / car_mileage_per_gallon + gas_bought_gallons) = initial_fuel - (initial_fuel * final_tank_fraction) / car_mileage_per_gallon + (580 - initial_drive_miles) / car_mileage_per_gallon →
  initial_drive_miles + (initial_fuel - initial_drive_miles / car_mileage_per_gallon + gas_bought_gallons - initial_fuel * final_tank_fraction / car_mileage_per_gallon) * car_mileage_per_gallon = 580 := 
sorry

end karl_total_miles_l780_780424


namespace number_minus_45_l780_780678

theorem number_minus_45 (x : ℕ) (h1 : (x / 2) / 2 = 85 + 45) : x - 45 = 475 := by
  sorry

end number_minus_45_l780_780678


namespace compute_value_condition_l780_780841

theorem compute_value_condition (x : ℝ) (h : x + (1 / x) = 3) :
  (x - 2) ^ 2 + 25 / (x - 2) ^ 2 = -x + 5 := by
  sorry

end compute_value_condition_l780_780841


namespace platform_length_is_350_l780_780022

variables (L : ℕ)

def train_length := 300
def time_to_cross_pole := 18
def time_to_cross_platform := 39

-- Speed of the train when crossing the pole
def speed_cross_pole : ℚ := train_length / time_to_cross_pole

-- Speed of the train when crossing the platform
def speed_cross_platform (L : ℕ) : ℚ := (train_length + L) / time_to_cross_platform

-- The main goal is to prove that the length of the platform is 350 meters
theorem platform_length_is_350 (L : ℕ) (h : speed_cross_pole = speed_cross_platform L) : L = 350 := sorry

end platform_length_is_350_l780_780022


namespace intersection_S_T_l780_780191

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780191


namespace heather_speed_heather_speed_final_l780_780372

variable {distance time speed : ℝ}

-- Conditions
def heather_distance : distance = 40 := by sorry
def heather_time : time = 5 := by sorry

-- Theorem to prove
theorem heather_speed (d t : ℝ) (h1 : d = 40) (h2 : t = 5) : speed = d / t := 
by sorry

-- Assign speed based on provided conditions
def heather_calculated_speed := (heather_speed distance time heather_distance heather_time)

-- Proof of the final answer
theorem heather_speed_final : heather_calculated_speed = 8 := by sorry

end heather_speed_heather_speed_final_l780_780372


namespace patrick_loss_percentage_l780_780459

theorem patrick_loss_percentage :
  let cost_A := 30 * 1,
      cost_B := 40 * 2,
      cost_C := 10 * 3,
      total_cost := cost_A + cost_B + cost_C,
      selling_A := 30 * (1 - 0.50),
      selling_B := 40 * (2 - 1),
      selling_C := 10 * (3 - 1.50),
      total_selling := selling_A + selling_B + selling_C,
      total_loss := total_cost - total_selling in
  (total_loss.to_float / total_cost.to_float) * 100 = 10.71 := sorry

end patrick_loss_percentage_l780_780459


namespace find_phi_l780_780378

open Real

theorem find_phi (φ : ℝ) (hφ : |φ| < π / 2)
  (h_symm : ∀ x, sin (2 * x + φ) = sin (2 * ((2 * π / 3 - x) / 2) + φ)) :
  φ = -π / 6 :=
by
  sorry

end find_phi_l780_780378


namespace area_of_quadrilateral_ABCG_l780_780118

variables {A B C D E F G : Type} [EuclideanGeometry.GEQ2D]
variables (a : ℝ) -- side length of the hexagon

-- Define the regular hexagon with vertices A, B, C, D, E, F
def is_regular_hexagon (A B C D E F : Type) : Prop :=
  (side_length A B = a ∧ side_length B C = a ∧ side_length C D = a ∧ side_length D E = a ∧ side_length E F = a ∧ side_length F A = a) ∧
  (equal_angles A B C ∧ equal_angles B C D ∧ equal_angles C D E ∧ equal_angles D E F ∧ equal_angles E F A ∧ equal_angles F A B)

-- Define point G as the midpoint of CD
def is_midpoint (G C D : Type) : Prop := 
  between C G D ∧ side_length C G = side_length G D

-- Define the area calculation assertion
theorem area_of_quadrilateral_ABCG
  (h_hex : is_regular_hexagon A B C D E F a)
  (h_mid : is_midpoint G C D) :
  area (quadrilateral A B C G) = (a^2 * sqrt 3) / 2 :=
sorry -- Proof to be completed

end area_of_quadrilateral_ABCG_l780_780118


namespace intersection_S_T_l780_780145

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780145


namespace pumps_fill_tank_together_in_approx_0_2857_hours_l780_780689

/--
   Given:
   - The rate of the small pump which takes 2 hours to fill a tank.
   - The rate of the larger pump which takes 1/3 hour to fill the same tank.

   Prove:
   - Both pumps together will fill the tank in 0.2857 hours.
-/
theorem pumps_fill_tank_together_in_approx_0_2857_hours :
  let small_pump_rate := 1 / 2 -- tanks per hour
      large_pump_rate := 3    -- tanks per hour
      combined_rate := small_pump_rate + large_pump_rate -- tanks per hour
  in (1 / combined_rate ≈ 0.2857) :=
by
  sorry

end pumps_fill_tank_together_in_approx_0_2857_hours_l780_780689


namespace ellipse_major_axis_length_l780_780703

theorem ellipse_major_axis_length :
  let F1 := (10, 5) in
  let F2 := (70, 30) in
  -- Reflect F1 over y-axis
  let F1' := (-10, 5) in
  -- Distance calculation
  let dist_F1'_F2 := Real.sqrt ((70 + 10) ^ 2 + (30 - 5) ^ 2) in
  -- Major Axis length
  dist_F1'_F2 = 85 := 
by
  sorry

end ellipse_major_axis_length_l780_780703


namespace find_f_value_l780_780112

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then 1 - x^2 else 2^x

theorem find_f_value :
  f (1 / (f (Real.log 6 / Real.log 2))) = 35 / 36 :=
by
  sorry

end find_f_value_l780_780112


namespace increasing_iff_range_m_l780_780858

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m / x

theorem increasing_iff_range_m (m : ℝ) : 
  (∀ x ∈ Icc (1 : ℝ) 3, differentiableAt ℝ (f m) x ∧ deriv (f m) x ≥ 0) ↔ (m ≥ -1) :=
by
  sorry

end increasing_iff_range_m_l780_780858


namespace largest_n_binomial_l780_780567

theorem largest_n_binomial :
  (∃n : ℕ, binom 10 4 + binom 10 5 = binom 11 n) → (n ≤ 6) :=
by
  sorry

end largest_n_binomial_l780_780567


namespace intersection_A_B_l780_780351

def setA : Set ℝ := {x | 0 < x}
def setB : Set ℝ := {x | -1 < x ∧ x < 3}
def intersectionAB : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_A_B :
  setA ∩ setB = intersectionAB := by
  sorry

end intersection_A_B_l780_780351


namespace shaded_area_correct_l780_780017

-- Definitions of Points
structure Point where
  x : ℝ
  y : ℝ

-- Definitions of the square and points A, B
def square_side : ℝ := 10
def A : Point := ⟨0, (3/4) * square_side⟩
def B : Point := ⟨square_side, square_side / 2⟩

-- The calculation of the area of the shaded region
noncomputable def shaded_area : ℝ := 
  let ab_dist := square_side
  let height := square_side / 2
  4 * (0.5 * ab_dist * height)

-- Statement of the theorem
theorem shaded_area_correct : shaded_area = 100 := by
  sorry

end shaded_area_correct_l780_780017


namespace sixty_percent_of_total_is_960_l780_780969

variable (number_of_boys number_of_girls total_participants : ℕ)

-- Condition 1: The difference between the number of boys and girls is 400.
def difference_condition : Prop := number_of_girls - number_of_boys = 400

-- Condition 2: There are 600 boys.
def boys_condition : Prop := number_of_boys = 600

-- Condition 3: The number of girls is more than the number of boys.
def girls_condition : Prop := number_of_girls > number_of_boys

-- Given conditions
axiom difference_condition_h : difference_condition number_of_boys number_of_girls
axiom boys_condition_h : boys_condition number_of_boys
axiom girls_condition_h : girls_condition number_of_boys number_of_girls

-- Total number of participants
def total_participants : ℕ := number_of_boys + number_of_girls

theorem sixty_percent_of_total_is_960 :
  0.6 * (number_of_boys + number_of_girls) = 960 :=
by 
  sorry

end sixty_percent_of_total_is_960_l780_780969


namespace product_divisibility_l780_780465

theorem product_divisibility {n : ℕ} (h : n > 2) (a : ℕ → ℤ) : 
  (∏ i in finset.range (n - 1), ∏ j in finset.range (i + 1, n), (a j - a i)) ∣ (∏ i in finset.range (n - 1), ∏ j in finset.range (i + 1, n), (j - i)) := 
sorry

end product_divisibility_l780_780465


namespace tangent_at_A_tangent_through_B_l780_780804

open Real

noncomputable def f (x : ℝ) : ℝ := 4 / x
noncomputable def f' (x : ℝ) : ℝ := -4 / (x ^ 2)

def point_A : ℝ × ℝ := (2, 2)
def point_B : ℝ × ℝ := (2, 0)

theorem tangent_at_A :
  let k := f' 2 in
  let line := fun x => k * (x - 2) + 2 in
  ∀ x y, y = f x → (x, y) = point_A → (x + y - 4 = 0) := sorry

theorem tangent_through_B :
  ∃ (m : ℝ), (m ≠ 0) ∧
  let tangent := fun x => -(4 / m^2) * (x - m) + (4 / m) in
  tangent 2 = 0 :=
sorry

end tangent_at_A_tangent_through_B_l780_780804


namespace correct_calculation_l780_780624

theorem correct_calculation : 
  ¬(2 * Real.sqrt 3 + 3 * Real.sqrt 2 = 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  ¬(5 * Real.sqrt 3 * 5 * Real.sqrt 2 = 5 * Real.sqrt 6) ∧
  ¬(Real.sqrt (4 + 1 / 2) = 2 * Real.sqrt (1 / 2)) :=
by {
  -- Using the conditions to prove the correct option B
  sorry
}

end correct_calculation_l780_780624


namespace cyclic_quadrilateral_iff_angle_equality_l780_780786

universe u
variables {A B C P Q D E M : Type u}

noncomputable def midpoint (A B : Type u) [linear_ordered_field A] : A :=
(A + B) / 2

noncomputable def reflection (P M : Type u) : A :=
2 * M - P

theorem cyclic_quadrilateral_iff_angle_equality
  (ABC : Type u) [linear_ordered_field ABC]
  (A B C P Q D E M : ABC)
  (M_midpoint: M = midpoint A B)
  (Q_reflection: Q = reflection P M)
  (D_intersection: ∃ t, D = A + t * (P - A) ∧ D ∈ line_through A P)
  (E_intersection: ∃ t, E = B + t * (P - B) ∧ E ∈ line_through B P) :
  (∃ O, O ∈ circumcircle A B D E) ↔ ∠ A C P = ∠ Q C B := 
sorry

end cyclic_quadrilateral_iff_angle_equality_l780_780786


namespace num_multiples_3003_in_form_10j_sub_10i_l780_780360

theorem num_multiples_3003_in_form_10j_sub_10i :
  ∃ n : ℕ, n = 192 ∧ ∀ i j : ℕ, 0 ≤ i → i < j → j ≤ 50 →
  (3003 ∣ (10^j - 10^i) ↔ ∃ k : ℕ, j - i = 6 * k) :=
by {
  use 192,
  sorry
}

end num_multiples_3003_in_form_10j_sub_10i_l780_780360


namespace intersection_S_T_eq_T_l780_780133

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780133


namespace fraction_inspected_by_Jane_l780_780914

theorem fraction_inspected_by_Jane (P : ℝ) (x y : ℝ) 
    (h1: 0.007 * x * P + 0.008 * y * P = 0.0075 * P) 
    (h2: x + y = 1) : y = 0.5 :=
by sorry

end fraction_inspected_by_Jane_l780_780914


namespace f_f_n_plus_n_eq_n_plus_1_l780_780960

-- Define the function f : ℕ+ → ℕ+ satisfying the given condition
axiom f : ℕ+ → ℕ+

-- Define that for all positive integers n, f satisfies the condition f(f(n)) + f(n+1) = n + 2
axiom f_condition : ∀ n : ℕ+, f (f n) + f (n + 1) = n + 2

-- State that we want to prove that f(f(n) + n) = n + 1 for all positive integers n
theorem f_f_n_plus_n_eq_n_plus_1 : ∀ n : ℕ+, f (f n + n) = n + 1 := 
by sorry

end f_f_n_plus_n_eq_n_plus_1_l780_780960


namespace product_of_series_l780_780715

theorem product_of_series : 
  (∏ n in Finset.range 7, (1 + 1 / (n + 1 : ℝ))) = 8 := 
by 
  sorry

end product_of_series_l780_780715


namespace intersection_S_T_l780_780316

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780316


namespace marathon_fraction_l780_780070

theorem marathon_fraction :
  ∃ (f : ℚ), (2 * 7) = (6 + (6 + 6 * f)) ∧ f = 1 / 3 :=
by 
  sorry

end marathon_fraction_l780_780070


namespace largest_integer_binom_l780_780588

theorem largest_integer_binom (n : ℕ) (h : nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) : n = 6 :=
sorry

end largest_integer_binom_l780_780588


namespace field_length_to_width_ratio_l780_780978

-- Conditions
def length_of_field := 32
def width_of_field := 16
def area_of_pond := 8 * 8
def area_of_field := length_of_field * width_of_field
def pond_area_fraction := 1 / 8

-- Main statement
theorem field_length_to_width_ratio : (length_of_field / width_of_field) = 2 :=
by
  have area_of_pond_correct : area_of_pond = 64,
    sorry
  have field_area_correct : area_of_field = area_of_pond * 8,
    sorry
  have width_of_field_correct : width_of_field = area_of_field / length_of_field,
    sorry
  rw [length_of_field, width_of_field],
  have : width_of_field = 16 := by
  {
    sorry
  },
  sorry

end field_length_to_width_ratio_l780_780978


namespace largest_integer_comb_l780_780603

theorem largest_integer_comb (n: ℕ) (h1 : n = 6) : 
  (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) :=
by
  rw [h1]
  exact sorry

end largest_integer_comb_l780_780603


namespace circle_center_l780_780098

theorem circle_center (x y: ℝ) : 
  (x + 2)^2 + (y + 3)^2 = 29 ↔ (∃ c1 c2 : ℝ, c1 = -2 ∧ c2 = -3) :=
by sorry

end circle_center_l780_780098


namespace find_oysters_first_day_l780_780511

variable (O : ℕ)  -- Number of oysters on the rocks on the first day

def count_crabs_first_day := 72  -- Number of crabs on the beach on the first day

def oysters_second_day := O / 2  -- Number of oysters on the rocks on the second day

def crabs_second_day := (2 / 3) * count_crabs_first_day  -- Number of crabs on the beach on the second day

def total_count := 195  -- Total number of oysters and crabs counted over the two days

theorem find_oysters_first_day (h:  O + oysters_second_day O + count_crabs_first_day + crabs_second_day = total_count) : 
  O = 50 := by
  sorry

end find_oysters_first_day_l780_780511


namespace intersection_S_T_eq_T_l780_780198

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780198


namespace intersection_S_T_eq_T_l780_780262

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780262


namespace collinear_AFE_AC_eq_AM_MC_squared_eq_2MD_MA_l780_780708

-- Define the necessary predicates and conditions
variables {Point Line Circle : Type}
variable tangent_to : Circle → Line → Point → Prop
variable on_line : Point → Line → Prop
variable on_circle : Point → Circle → Prop
variable collinear : Point → Point → Point → Prop
variable perpendicular : Line → Line → Prop
variable intersect : Line → Circle → Point → Prop
variable diameter : Line → Circle → Prop
variable between : Point → Point → Point → Prop

variables A B C D E F M O : Point
variable l_AB : Line
variable l_CD : Line
variable c_omega : Circle
variable sc_AB : Circle

-- The given conditions
axiom h1 : tangent_to c_omega l_AB M
axiom h2 : tangent_to c_omega sc_AB E
axiom h3 : perpendicular l_CD l_AB
axiom h4 : intersect l_CD sc_AB C
axiom h5 : tangent_to c_omega l_CD F
axiom h6 : between A C M

-- Translate the proof questions
-- 1. Points A, F, and E are collinear.
theorem collinear_AFE : collinear A F E := sorry

-- 2. AC = AM
theorem AC_eq_AM : sorry := sorry

-- 3. Computation relation MC² = 2MD × MA
theorem MC_squared_eq_2MD_MA : sorry := sorry

end collinear_AFE_AC_eq_AM_MC_squared_eq_2MD_MA_l780_780708


namespace foodRatio_l780_780089

-- Definitions of initial purchases and final amounts
def initialDogFood := 15
def initialCatFood := 10
def initialBirdFood := 5
def finalDogFood := 40
def finalCatFood := 15
def finalBirdFood := 5

-- Total final amount
def totalFinalFood := finalDogFood + finalCatFood + finalBirdFood

-- The function to calculate original amounts of food
def originalFood (finalAmount initialAmount : Nat) := finalAmount - initialAmount

def originalDogFood := originalFood finalDogFood initialDogFood
def originalCatFood := originalFood finalCatFood initialCatFood
def originalBirdFood := originalFood finalBirdFood initialBirdFood

-- Total collected food 
def totalOriginalFood := originalDogFood + originalCatFood + originalBirdFood + initialDogFood + initialCatFood + initialBirdFood

-- Proving the ratio is 8:3:1 given the conditions
theorem foodRatio :
  totalFinalFood = 60 ∧ (finalDogFood = 8 * 5 ∧ finalCatFood = 3 * 5 ∧ finalBirdFood = 1 * 5) → 
  (originalDogFood, originalCatFood, originalBirdFood) = (25, 5, 0) ∧ (8:3:1) :=
by
  sorry

end foodRatio_l780_780089


namespace largest_n_for_binom_equality_l780_780590

theorem largest_n_for_binom_equality : ∃ (n : ℕ), (∑ i : {i // i = 4 ∨ i = 5}, Nat.choose 10 i) = Nat.choose 11 n ∧ n = 6 :=
by
  sorry

end largest_n_for_binom_equality_l780_780590


namespace incenter_bisects_segment_l780_780503

open Real EuclideanGeometry

variables {a b c : ℝ}
variables (A B C O B1 : Point)

-- The side lengths of triangle ABC form an arithmetic progression a, b, c where a < b < c.
def isArithmeticProgression (a b c : ℝ) := a < b ∧ b < c ∧ b - a = c - b

-- Given a < b < c and a, b, c form an arithmetic progression. Assume triangle ABC with sides a, b, c.
def conditions (a b c : ℝ) (A B C O B1 : Point) : Prop :=
  Triangle A B C ∧
  isArithmeticProgression a b c ∧
  isIncenter O A B C ∧
  isAngleBisector A B C B1

-- Final goal: Prove that the incenter O bisects the segment BB1, that is OB = OB1.
theorem incenter_bisects_segment
  (A B C O B1 : Point) (a b c : ℝ)
  (h_cond : conditions a b c A B C O B1) :
  dist O B = dist O B1 :=
sorry

end incenter_bisects_segment_l780_780503


namespace reciprocal_of_neg_one_fifth_l780_780501

theorem reciprocal_of_neg_one_fifth : (-(1 / 5) : ℚ)⁻¹ = -5 :=
by
  sorry

end reciprocal_of_neg_one_fifth_l780_780501


namespace find_p_q_l780_780052

def valid_sequence (s : List ℕ) : Prop :=
  s.head = 0 ∧ s.last = 0 ∧ ∀ i < s.length - 1, ¬ (s[i] = 1 ∧ s[i+1] = 1)

def sequence_length_12 (s : List ℕ) : Prop :=
  s.length = 12

def gcd (a b : Nat) : Nat :=
  if b = 0 then a else gcd b (a % b)

def relatively_prime (p q : Nat) : Prop :=
  gcd p q = 1

theorem find_p_q (p q : Nat) :
  (∃ (s : List ℕ), valid_sequence s ∧ sequence_length_12 s) →
  relatively_prime p q →
  (p : ℚ) / (q : ℚ) = 5 / 1024 →
  p + q = 1029 :=
by
  sorry

end find_p_q_l780_780052


namespace find_initial_discount_percentage_l780_780455

variable (x : ℝ) -- Let the initial discount percentage be x

def initial_price : ℝ := 125

def mothers_day_discounted_price : ℝ := initial_price - (x / 100) * initial_price

def additional_discounted_price : ℝ := mothers_day_discounted_price - (4 / 100) * mothers_day_discounted_price

-- The final price after all discounts is $108
axiom final_price_condition : additional_discounted_price = 108

theorem find_initial_discount_percentage
    (initial_price_condition : initial_price = 125)
    (final_price_condition : additional_discounted_price = 108) : x = 10 :=
sorry

end find_initial_discount_percentage_l780_780455


namespace largest_n_binomial_l780_780576

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l780_780576


namespace intersection_S_T_eq_T_l780_780273

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780273


namespace xy_eq_five_l780_780113

-- Definitions of the conditions
variables (x y : ℝ)

-- Hypothesis reflecting the given equation
def given_eq := (x / (1 - Complex.i) - y / (1 - 2 * Complex.i) = 5 / (1 - 3 * Complex.i))

-- Statement translating the problem into a proof problem
theorem xy_eq_five (h : given_eq x y) : x * y = 5 := by
  sorry

end xy_eq_five_l780_780113


namespace sqrt_expression_evaluation_l780_780711

theorem sqrt_expression_evaluation :
  (Real.sqrt 48 - 6 * Real.sqrt (1 / 3) - Real.sqrt 18 / Real.sqrt 6) = Real.sqrt 3 :=
by
  sorry

end sqrt_expression_evaluation_l780_780711


namespace Northton_time_capsule_depth_l780_780474

theorem Northton_time_capsule_depth:
  ∀ (d_southton d_northton : ℝ),
  d_southton = 15 →
  d_northton = (4 * d_southton) - 12 →
  d_northton = 48 :=
by
  intros d_southton d_northton h_southton h_northton
  rw [h_southton] at h_northton
  rw [← h_northton]
  sorry

end Northton_time_capsule_depth_l780_780474


namespace find_a_l780_780818

-- Define the hyperbola equation and the asymptote conditions
def hyperbola (a : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / 9) = 1

def asymptote1 (x y : ℝ) : Prop := 3 * x + 2 * y = 0
def asymptote2 (x y : ℝ) : Prop := 3 * x - 2 * y = 0

-- Prove that if asymptote conditions hold, a = 2
theorem find_a (a : ℝ) (ha : a > 0) :
  (∀ x y, asymptote1 x y) ∧ (∀ x y, asymptote2 x y) → a = 2 :=
sorry

end find_a_l780_780818


namespace orthocenters_collinear_l780_780043

open Real EuclideanGeometry

noncomputable def midpoint (A B : Point) : Point := (A + B) / 2

theorem orthocenters_collinear {A B C M N P X Y Z T : Point} :
  line A B ∩ line M N = {M} →
  line A C ∩ line M N = {N} →
  line B C ∩ line M N = {P} →
  midpoint N M = X →
  midpoint M B = Y →
  midpoint B C = Z →
  midpoint C N = T →
  collinear {ortho A M N, ortho A Y T, ortho P B M, ortho P X Z} :=
by
  sorry

end orthocenters_collinear_l780_780043


namespace cos_A_equals_4_5_l780_780751

-- Define the sides of the triangle
def AB : ℝ := 8
def AC : ℝ := 6
def BC : ℝ := 10

-- Define the proof statement to show that cos A = 4/5
theorem cos_A_equals_4_5 (h : AB^2 + AC^2 = BC^2) : real.cos (real.arctan (AC / AB)) = 4 / 5 :=
by
  have hypotenuse_correct : BC = real.sqrt (AB^2 + AC^2) := by
    rw [←real.sqrt_eq_rpow, real.sqrt_eq_rpow, sq, sq, sq_sqrt, add_comm, add_assoc] 
    exact hypotenuse_correct sorry
  sorry

end cos_A_equals_4_5_l780_780751


namespace largest_integer_binom_l780_780582

theorem largest_integer_binom (n : ℕ) (h : nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) : n = 6 :=
sorry

end largest_integer_binom_l780_780582


namespace find_f_f_neg1_l780_780338

def f (x : ℝ) : ℝ :=
  if x >= 2 then real.sqrt x else 3 - x

theorem find_f_f_neg1 : 
  f (f (-1)) = 2 :=
by
  sorry

end find_f_f_neg1_l780_780338


namespace actual_distance_from_base_l780_780641

theorem actual_distance_from_base (distance_map_mountains : ℝ) (actual_distance_mountains : ℝ)
  (distance_map_ram : ℝ) : ℝ :=
  distance_map_mountains = 310 ∧ actual_distance_mountains = 136 ∧ distance_map_ram = 34 →
  (distance_map_ram * (actual_distance_mountains / distance_map_mountains)) = 14.98 :=
begin
  sorry
end

end actual_distance_from_base_l780_780641


namespace _l780_780638

example (length train : ℕ) (time : ℕ) (h_length : length train = 500) (h_time : time = 50) :
  length train / time = 10 := 
by 
-- We state the theorem without proof, i.e., using sorry.
  sorry

end _l780_780638


namespace grazing_area_allocation_l780_780055

-- Definitions of the conditions
def initial_rhinoceroses : ℕ := 8000
def watering_area : ℕ := 10000
def percentage_increase : ℚ := 0.10
def total_preserve_area : ℕ := 890000

-- Calculate the number of rhinoceroses after the increase
def increased_rhinoceroses : ℕ := initial_rhinoceroses + (initial_rhinoceroses * percentage_increase).toNat

-- Calculate the total grazing area
def total_grazing_area : ℕ := total_preserve_area - watering_area

-- Calculate the grazing area per rhinoceros
def grazing_area_per_rhinoceros : ℚ := total_grazing_area / increased_rhinoceroses

-- The theorem to prove 
theorem grazing_area_allocation : grazing_area_per_rhinoceros = 100 := by
  sorry

end grazing_area_allocation_l780_780055


namespace madeline_flower_count_l780_780451

theorem madeline_flower_count 
    (r w : ℕ) 
    (b_percent : ℝ) 
    (total : ℕ) 
    (h_r : r = 4)
    (h_w : w = 2)
    (h_b_percent : b_percent = 0.40)
    (h_total : r + w + (b_percent * total) = total) : 
    total = 10 :=
by 
    sorry

end madeline_flower_count_l780_780451


namespace distance_to_place_l780_780999

-- Define the conditions
def speed_boat_standing_water : ℝ := 16
def speed_stream : ℝ := 2
def total_time_taken : ℝ := 891.4285714285714

-- Define the calculated speeds
def downstream_speed : ℝ := speed_boat_standing_water + speed_stream
def upstream_speed : ℝ := speed_boat_standing_water - speed_stream

-- Define the variable for the distance
variable (D : ℝ)

-- State the theorem to prove
theorem distance_to_place :
  D / downstream_speed + D / upstream_speed = total_time_taken →
  D = 7020 :=
by
  intro h
  sorry

end distance_to_place_l780_780999


namespace intersection_eq_T_l780_780244

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780244


namespace fraction_to_decimal_subtraction_l780_780077

theorem fraction_to_decimal_subtraction 
    (h : (3 : ℚ) / 40 = 0.075) : 
    0.075 - 0.005 = 0.070 := 
by 
    sorry

end fraction_to_decimal_subtraction_l780_780077


namespace intersection_S_T_eq_T_l780_780128

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780128


namespace largest_n_binom_l780_780547

theorem largest_n_binom (n : ℕ) (h : 0 ≤ n ∧ n ≤ 11) : 
  (binomial 10 4 + binomial 10 5 = binomial 11 n) → n = 6 :=
by
  sorry

end largest_n_binom_l780_780547


namespace intersection_eq_T_l780_780157

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780157


namespace sum_first_10_terms_eq_l780_780785

def sequence_a : ℕ → ℚ
| 1     := 1
| (n+1) := sequence_a n + (n + 1)

noncomputable def sum_first_10_terms : ℚ :=
  (Finset.range 10).sum (λ i, 1 / sequence_a (i + 1))

theorem sum_first_10_terms_eq : sum_first_10_terms = 20 / 11 :=
by
  sorry

end sum_first_10_terms_eq_l780_780785


namespace positive_difference_of_diagonal_sums_l780_780653

noncomputable def initial_matrix : matrix (fin 4) (fin 4) ℕ := λ i j =>
  [[1, 2, 3, 4],
   [8, 9, 10, 11],
   [15, 16, 17, 18],
   [22, 23, 24, 25]].nth i.untyped! j.untyped! .get!

noncomputable def modified_matrix : matrix (fin 4) (fin 4) ℕ := λ i j =>
  [[1, 2, 3, 4],
   [11, 10, 9, 8],
   [15, 16, 17, 18],
   [25, 24, 23, 22]].nth i.untyped! j.untyped! .get!

def diag_sum (m : matrix (fin 4) (fin 4) ℕ) : ℕ :=
  ∑ i, m i i

def secondary_diag_sum (m : matrix (fin 4) (fin 4) ℕ) : ℕ :=
  ∑ i, m i (3 - i.to_nat)

theorem positive_difference_of_diagonal_sums :
  |diag_sum modified_matrix - secondary_diag_sum modified_matrix| = 4 :=
by
  sorry

end positive_difference_of_diagonal_sums_l780_780653


namespace geometric_figure_l780_780075

theorem geometric_figure (x y : ℝ) : 
  |x| + |y| ≤ sqrt (2 * (x^2 + y^2)) ∧ sqrt (2 * (x^2 + y^2)) ≤ sqrt 2 * max (|x|) (|y|) → 
  (∃ rect, (is_rectangle rect) ∧ 
           (inscribes_circle rect) ∧ 
           (circumscribes_circle rect)) :=
by
  intros h
  sorry

end geometric_figure_l780_780075


namespace henry_classical_cds_l780_780866

variable (R C : ℕ)

theorem henry_classical_cds :
  (23 - 3 = R) →
  (R = 2 * C) →
  C = 10 :=
by
  intros h1 h2
  sorry

end henry_classical_cds_l780_780866


namespace intersection_S_T_eq_T_l780_780255

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780255


namespace S_inter_T_eq_T_l780_780296

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780296


namespace coffee_shop_spending_l780_780103

variable (R S : ℝ)

theorem coffee_shop_spending (h1 : S = 0.60 * R) (h2 : R = S + 12.50) : R + S = 50 :=
by
  sorry

end coffee_shop_spending_l780_780103


namespace three_planes_divide_into_six_parts_l780_780381

-- Define a type for Plane
axiom Plane : Type

-- Define a predicate that indicates whether three planes divide space into N parts
axiom divides_space_into : Plane → Plane → Plane → ℕ → Prop

-- Define the positional relationships as predicates
axiom collinear : Plane → Plane → Plane → Prop
axiom parallel_two_and_intersect_third : Plane → Plane → Plane → Prop

-- Theorem statement matching the problem given
theorem three_planes_divide_into_six_parts :
  ∀ (P1 P2 P3 : Plane), divides_space_into P1 P2 P3 6 → (collinear P1 P2 P3 ∨ parallel_two_and_intersect_third P1 P2 P3) :=
by
  intros,
  sorry

end three_planes_divide_into_six_parts_l780_780381


namespace find_ellipse_equation_and_range_of_m_l780_780090

theorem find_ellipse_equation_and_range_of_m:
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ), (x^2) / (a^2) + (y^2) / (b^2) = 1 → 
   (∃ (c : ℝ), x^2 + y^2 = c) ∧ 
   ellipse_eccentricity a b = (sqrt 3) / 2 ∧ 
  (a^2 = 4 * b^2)) ∧ 
  (∀ x0, (-sqrt(15) / 2 < x0 ∧ x0 < sqrt(15) / 2 → 
    (∃ m, (-3 * sqrt(15) / 8 < m) ∧ (m < 3 * sqrt(15) / 8) ∧ 
     (∃ x y : ℝ, line_perpendicular_bisector_intersects 
       (midpoint_of_seg_mn x0 (1 / 4)) (line_intersections_with_ellipse a b) m (0))))) :=
sorry

end find_ellipse_equation_and_range_of_m_l780_780090


namespace intersection_S_T_l780_780193

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780193


namespace math_problem_l780_780337

noncomputable def f (a : ℝ) (x : ℝ) := a * log x - x + (1 / x)
noncomputable def g (b : ℝ) (x : ℝ) := x^2 + x - b
noncomputable def h (a b : ℝ) (x : ℝ) := (f a x) / (g b x)

theorem math_problem
    (a b : ℝ)
    (f_def : ∀ x, f a x = a * log x - x + 1 / x)
    (g_def : ∀ x, g b x = x^2 + x - b)
    (P_def : ∀ P : ℝ × ℝ, P = (1, 0))
    (P_on_f_deriv : ∀ x, P_def (1, 0) → (λ x, (f a x)' = 0) 1 = 0)
    (P_on_g : ∀ x, P_def (1, 0) → g b x = 0) :
    a = 2 ∧ b = 2 ∧ ∀ (x : ℝ), x > 0 ∧ x ≠ 1 → h 2 2 x < 0 :=
sorry

end math_problem_l780_780337


namespace find_f_if_f1_eq1_find_range_of_a_l780_780926

-- f(x) = ax^2 + bx + c
variables {a b c : ℝ} {f : ℝ → ℝ}
variables {g : ℝ → ℝ}
variables {t x1 x2 x3 x4 : ℝ}

-- Conditions for f
def condition_f := (f 0 = 1) ∧ (∀ x, f x = f (2 / a - x))

-- Condition for g
def g_def := g x = log 2 (x - 2 * sqrt x + 2)

-- Question (1): Prove f(x) = 2x^2 - 2x + 1 when f(1) = 1
theorem find_f_if_f1_eq1 (h1 : condition_f) (h2 : f 1 = 1) : ∀ x, f x = 2 * x^2 - 2 * x + 1 := by
  sorry

-- Question (2): Prove the range of a for the given condition
theorem find_range_of_a
  (h1 : condition_f) 
  (hg : ∀ x, g x = log 2 (x - 2 * sqrt x + 2))
  (hp : ∀ t, ∃ x1 x2 ∈ set.Icc (t - 1) (t + 1), ∀ x3 x4 ∈ set.Icc (1 / 4) 4, f x1 - f x2 ≥ g x3 - g x4)
  : 1 ≤ a := by
  sorry

end find_f_if_f1_eq1_find_range_of_a_l780_780926


namespace polygons_have_common_point_l780_780390

open Set

theorem polygons_have_common_point
  (square : Set ℝ)
  (polygons : Fin 2024 → Set ℝ)
  (h_square : measure_theory.measure_space.volume square = 1)
  (h_polygons_area : measure_theory.measure_space.volume (⋃ i, polygons i) > 2023) :
  ∃ p ∈ square, ∀ i : Fin 2024, p ∈ polygons i :=
by
  sorry

end polygons_have_common_point_l780_780390


namespace largest_integer_comb_l780_780605

theorem largest_integer_comb (n: ℕ) (h1 : n = 6) : 
  (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) :=
by
  rw [h1]
  exact sorry

end largest_integer_comb_l780_780605


namespace hardware_contract_probability_l780_780666

noncomputable def P_S' : ℚ := 3 / 5
noncomputable def P_at_least_one : ℚ := 5 / 6
noncomputable def P_H_and_S : ℚ := 0.31666666666666654 -- 19 / 60 in fraction form
noncomputable def P_S : ℚ := 1 - P_S'

theorem hardware_contract_probability :
  (P_at_least_one = P_H + P_S - P_H_and_S) →
  P_H = 0.75 :=
by
  sorry

end hardware_contract_probability_l780_780666


namespace allan_balloons_difference_l780_780701

theorem allan_balloons_difference :
  ∀ (Allan_balloons Jake_balloons : ℕ), Allan_balloons = 5 → Jake_balloons = 3 → Allan_balloons - Jake_balloons = 2 :=
by
  intros Allan_balloons Jake_balloons hAllan hJake
  rw [hAllan, hJake]
  norm_num
  sorry

end allan_balloons_difference_l780_780701


namespace intersection_eq_T_l780_780281

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780281


namespace cos_triple_angle_l780_780834

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 :=
by
  sorry

end cos_triple_angle_l780_780834


namespace intersection_S_T_l780_780314

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780314


namespace largest_lambda_inequality_l780_780100

theorem largest_lambda_inequality :
  ∀ (a b c d e : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → 0 ≤ e →
  (a^2 + b^2 + c^2 + d^2 + e^2 ≥ a * b + (5/4) * b * c + c * d + d * e) :=
by
  sorry

end largest_lambda_inequality_l780_780100


namespace remainder_when_divided_by_8_l780_780642

theorem remainder_when_divided_by_8 (k : ℤ) : ((63 * k + 25) % 8) = 1 := 
by sorry

end remainder_when_divided_by_8_l780_780642


namespace total_crayons_lost_or_given_away_l780_780946

/-
Paul gave 52 crayons to his friends.
Paul lost 535 crayons.
Paul had 492 crayons left.
Prove that the total number of crayons lost or given away is 587.
-/
theorem total_crayons_lost_or_given_away
  (crayons_given : ℕ)
  (crayons_lost : ℕ)
  (crayons_left : ℕ)
  (h_crayons_given : crayons_given = 52)
  (h_crayons_lost : crayons_lost = 535)
  (h_crayons_left : crayons_left = 492) :
  crayons_given + crayons_lost = 587 := 
sorry

end total_crayons_lost_or_given_away_l780_780946


namespace cos_triple_angle_l780_780833

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 :=
by
  sorry

end cos_triple_angle_l780_780833


namespace intersection_S_T_eq_T_l780_780200

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780200


namespace angle_bisector_parallel_side_equality_l780_780923

-- Declare the various points and entities involved in the problem.
variables {R S T E D F : Type}
-- Assume certain relationships
variables [IsTriangle R S T]
variables [IsAngleBisector R E T]
variables [IsParallel ED RT]
variables [IntersectionPoint F TD RE]
variables [EqLength SD RT]

-- State the theorem to be proved.
theorem angle_bisector_parallel_side_equality
  (h1: IsAngleBisector R E T)
  (h2: IsParallel ED RT)
  (h3: IntersectionPoint F TD RE)
  (h4: EqLength SD RT)
  : TE = TF := 
sorry -- placeholder for the proof, not needed according to the instructions

end angle_bisector_parallel_side_equality_l780_780923


namespace length_of_bd_equals_three_l780_780704

noncomputable def equilateral_triangle_bd (a b c : ℕ) : ℕ :=
  let side_length : ℕ := 26 in
  let segment1_a : ℕ := 3 in
  let segment2_a : ℕ := 22 in
  let segment1_c : ℕ := 3 in
  let segment2_c : ℕ := 23 in
  let segment_ab : ℕ := 6 in
  let segment_bd := side_length - segment1_a - segment2_c in
  let equation := (segment1_a * segment1_a + segment1_a * segment2_a) = (segment_bd * (segment_bd + segment2_a + segment2_c)) in
  have h_equation : equation, from sorry,
  have h := (segment_bd + segment2_c) = side_length, from sorry,
  segment_bd

theorem length_of_bd_equals_three :
  equilateral_triangle_bd 3 22 23 = 3 :=
begin
  sorry
end

end length_of_bd_equals_three_l780_780704


namespace intersection_eq_T_l780_780158

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780158


namespace vasya_perimeter_ratio_l780_780691

variable (a b : ℝ)
variable (x y : ℝ)
variable (ratio_perimeter : ℝ)

-- Assume the given conditions
-- 1. Areas ratio for Petya’s division
axiom area_ratio_petya : a * x / (a * (b - x)) = 1 / 2
-- 2. Perimeter ratio for Petya’s division
axiom perimeter_ratio_petya : (2 * a + 2 * x) / (2 * a + 2 * (b - x)) = 3 / 5
-- 3. Areas ratio for Vasya's division
axiom area_ratio_vasya : (y * b) / ((a - y) * b) = 1 / 2

-- Goal: Prove ratio of perimeters in Vasya’s division is 20/19
theorem vasya_perimeter_ratio :
  let b := 6 * a in
  let y := a / 3 in
  let perimeter1 := 2 * y + 2 * b in
  let perimeter2 := 2 * (a - y) + 2 * b in
  (perimeter1 / perimeter2 = 20 / 19) :=
by sorry

end vasya_perimeter_ratio_l780_780691


namespace complex_solution_l780_780852

theorem complex_solution (z : ℂ) (h : z * (0 + 1 * I) = (0 + 1 * I) - 1) : z = 1 + I :=
by
  sorry

end complex_solution_l780_780852


namespace quadratic_has_root_in_interval_l780_780807

theorem quadratic_has_root_in_interval (a b c : ℝ)
    (h_nonzero: a ≠ 0)
    (h_two_roots: ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0)
    (h_condition: |a*(b-c)| > |b^2 - a*c| + |c^2 - a*b|):
  ∃ α : ℝ, 0 < α ∧ α < 2 ∧ a*α^2 + b*α + c = 0 :=
by {
    sorry,
}

end quadratic_has_root_in_interval_l780_780807


namespace spherical_to_rectangular_l780_780078

theorem spherical_to_rectangular :
  let ρ := 6
  let θ := 7 * Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-3 * Real.sqrt 6, -3 * Real.sqrt 6, 3) :=
by
  sorry

end spherical_to_rectangular_l780_780078


namespace number_of_solution_pairs_l780_780082

theorem number_of_solution_pairs : 
  ∃ n, (∀ x y : ℕ, 4 * x + 7 * y = 548 → (x > 0 ∧ y > 0) → n = 19) :=
sorry

end number_of_solution_pairs_l780_780082


namespace larger_circle_radius_l780_780993

theorem larger_circle_radius (r : ℝ) 
  (h1 : (r : ℝ) * 2 = 5 / 2 * r) 
  (h2 : AC = r * 5) 
  (h3 : ∃ BC BD DC, BD = r ∧ AC^2 = AB^2 + BC^2 ∧ BC^2 = r^2 + (5r/2 - r)^2) 
  (h4 : AB = 15) : 
  5 * r / 2 = 75 / Real.sqrt 7 := 
by 
  sorry 

end larger_circle_radius_l780_780993


namespace only_rotationally_symmetric_curve_is_circle_l780_780467

-- Definitions related to the problem
variable {O : ℝ × ℝ} -- Point O
variable {K : set (ℝ × ℝ)} -- Curve K

-- Conditions
axiom maps_to_itself_under_120_deg_rotation (p : ℝ × ℝ) : (p ∈ K) → 
    (complex.exp (2 * π * complex.I / 3) * (p.1 + p.2 * complex.I) = fst p + snd p * complex.I)

-- Theorem: Curve must be a circle
theorem only_rotationally_symmetric_curve_is_circle (h : ∃ x ∈ K, true) :
 ∃ r : ℝ, ∀ p ∈ K, (complex.abs (complex.mk p.1 p.2 - complex.mk O.1 O.2) = r) := sorry

end only_rotationally_symmetric_curve_is_circle_l780_780467


namespace largest_integer_binom_l780_780615

theorem largest_integer_binom
  (n : ℕ)
  (h₁ : binom 10 4 + binom 10 5 = binom 11 n) :
  n = 6 :=
sorry

end largest_integer_binom_l780_780615


namespace S_inter_T_eq_T_l780_780307

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780307


namespace intersection_S_T_eq_T_l780_780275

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780275


namespace intersection_S_T_eq_T_l780_780199

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780199


namespace intersection_eq_T_l780_780166

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780166


namespace james_sodas_per_day_l780_780418

theorem james_sodas_per_day : 
  (∃ packs : ℕ, ∃ sodas_per_pack : ℕ, ∃ additional_sodas : ℕ, ∃ days : ℕ,
    packs = 5 ∧ sodas_per_pack = 12 ∧ additional_sodas = 10 ∧ days = 7 ∧
    ((packs * sodas_per_pack + additional_sodas) / days) = 10) :=
by
  use 5, 12, 10, 7
  split; simp; sorry

end james_sodas_per_day_l780_780418


namespace gcd_problem_l780_780918

def gcd3 (x y z : ℕ) : ℕ := Int.gcd x (Int.gcd y z)

theorem gcd_problem (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : gcd3 (a^2 - 1) (b^2 - 1) (c^2 - 1) = 1) :
  gcd3 (a * b + c) (b * c + a) (c * a + b) = gcd3 a b c :=
by
  sorry

end gcd_problem_l780_780918


namespace set_equivalence_l780_780832

variable (M : Set ℕ)

theorem set_equivalence (h : M ∪ {1} = {1, 2, 3}) : M = {1, 2, 3} :=
sorry

end set_equivalence_l780_780832


namespace zero_in_interval_l780_780480

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 3

theorem zero_in_interval : 
    (∀ x y : ℝ, 0 < x → x < y → f x < f y) → 
    (f 1 = -2) →
    (f 2 = Real.log 2 + 5) →
    (∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0) :=
by 
    sorry

end zero_in_interval_l780_780480


namespace most_and_least_l780_780907

variables {Jan Kim Lee Ron Zay : ℝ}

-- Conditions as hypotheses
axiom H1 : Lee < Jan
axiom H2 : Kim < Jan
axiom H3 : Zay < Ron
axiom H4 : Zay < Lee
axiom H5 : Zay < Jan
axiom H6 : Jan < Ron

theorem most_and_least :
  (Ron > Jan) ∧ (Ron > Kim) ∧ (Ron > Lee) ∧ (Ron > Zay) ∧ 
  (Zay < Jan) ∧ (Zay < Kim) ∧ (Zay < Lee) ∧ (Zay < Ron) :=
by {
  -- Proof is omitted
  sorry
}

end most_and_least_l780_780907


namespace largest_n_for_binom_equality_l780_780597

theorem largest_n_for_binom_equality : ∃ (n : ℕ), (∑ i : {i // i = 4 ∨ i = 5}, Nat.choose 10 i) = Nat.choose 11 n ∧ n = 6 :=
by
  sorry

end largest_n_for_binom_equality_l780_780597


namespace sequence_m_l780_780995

noncomputable def a (n : ℕ) : ℕ :=
  if n = 0 then 0  -- We usually start sequences from n = 1; hence, a_0 is irrelevant
  else (n * n) - n + 1

theorem sequence_m (m : ℕ) (h_positive : m > 0) (h_bound : 43 < a m ∧ a m < 73) : m = 8 :=
by {
  sorry
}

end sequence_m_l780_780995


namespace find_operation_l780_780336

theorem find_operation (a b : Int) (h : a + b = 0) : (7 + (-7) = 0) := 
by
  sorry

end find_operation_l780_780336


namespace machine_depreciation_time_l780_780676

-- Define constants and conditions

def V_initial : ℝ := 128000
def r : ℝ := 0.25
def V_final : ℝ := 54000

-- Define the mathematical proof problem
theorem machine_depreciation_time :
  ∃ t : ℝ, V_initial * (1 - r) ^ t = V_final ∧ t = (Real.log(0.421875) / Real.log(0.75)) :=
by
  sorry

end machine_depreciation_time_l780_780676


namespace intersection_S_T_eq_T_l780_780202

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780202


namespace area_inside_rectangle_outside_circles_is_4_l780_780468

-- Specify the problem in Lean 4
theorem area_inside_rectangle_outside_circles_is_4 :
  let CD := 3
  let DA := 5
  let radius_A := 1
  let radius_B := 2
  let radius_C := 3
  let area_rectangle := CD * DA
  let area_circles := (radius_A^2 + radius_B^2 + radius_C^2) * Real.pi / 4
  abs (area_rectangle - area_circles - 4) < 1 :=
by
  repeat { sorry }

end area_inside_rectangle_outside_circles_is_4_l780_780468


namespace convex_pentagon_area_l780_780464

theorem convex_pentagon_area (vertices : Fin 5 → ℤ × ℤ) (h_convex : Convex ℝ (Set.range vertices)) 
                             (h_no_collinear : ∀ (i j k : Fin 5), i ≠ j → j ≠ k → k ≠ i → ¬Collinear ({vertices i, vertices j, vertices k} : Set (ℤ × ℤ))) : 
                            (polygon_area vertices ≥ 5 / 2) :=
begin
  sorry
end

end convex_pentagon_area_l780_780464


namespace largest_integer_binom_l780_780613

theorem largest_integer_binom
  (n : ℕ)
  (h₁ : binom 10 4 + binom 10 5 = binom 11 n) :
  n = 6 :=
sorry

end largest_integer_binom_l780_780613


namespace digit_X_in_24_factorial_l780_780483

theorem digit_X_in_24_factorial : 
  ∃ X : Nat, (24! / 10000) % 10 = X ∧ (Sum (List.map Char.digitToInt ['6', '2', '0', '4', '4', '8', '4', '0', '1', '7', '3', '3', '2', '3', '9', '4', '3', '9', '3', '6']) + X) % 9 = 0 :=
by
  sorry

end digit_X_in_24_factorial_l780_780483


namespace series_ratio_eq_18_over_19_l780_780752

theorem series_ratio_eq_18_over_19 (n : ℕ) (h_pos : 0 < n) :
  (∑ k in finset.range n, (2 * k + 1)) / (∑ k in finset.range n, 2 * (k + 1)) = (18 : ℚ) / (19 : ℚ) → n = 18 := 
sorry

end series_ratio_eq_18_over_19_l780_780752


namespace eccentricity_of_ellipse_l780_780795

theorem eccentricity_of_ellipse (F₁ F₂ A B : Point) (ellipse : Ellipse) :
  -- Assumptions
  ellipse.foci = (F₁, F₂) →
  (∃ line : Line, line.contains F₁ ∧ line ⟂ ellipse.major_axis ∧ line.intersects ellipse = {A, B}) →
  (triangle A B F₂).is_equilateral →
  
  -- Conclusion
  ellipse.eccentricity = Real.sqrt 3 / 3 :=
by
  sorry

end eccentricity_of_ellipse_l780_780795


namespace intersection_of_S_and_T_l780_780232

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780232


namespace largest_n_binom_equality_l780_780528

theorem largest_n_binom_equality :
  ∃ n : ℕ, (binomial 10 4 + binomial 10 5 = binomial 11 n) ∧ ∀ m : ℕ, (binomial 11 m = binomial 11 5) → m ≤ 6 :=
by
  sorry

end largest_n_binom_equality_l780_780528


namespace length_range_of_ON_l780_780881

theorem length_range_of_ON :
  let A := (x : ℝ) in
  let B := (y : ℝ) in
  (A^2 + B^2 = 4) →  -- Equation of circumcircle
  (real.angle (vector.angle A B) = real.pi / 3) →  -- Angle given as π/3
  let M := (A + B) / 2 in
  let N := symmetric_point_about_line_y_eq_x_plus_2 M in  -- Symmetry condition
  (let O := (0, 0) in [2 * real.sqrt 2 - 1, 2 * real.sqrt 2 + 1] = length (segment O N)) :=
sorry

end length_range_of_ON_l780_780881


namespace intersection_S_T_l780_780150

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780150


namespace all_lines_PQ_through_single_point_l780_780933

theorem all_lines_PQ_through_single_point
  {A B C X : Point}
  (hX_AC : OnLine X A C)
  (hX_not_A : X ≠ A)
  (hX_not_C : X ≠ C)
  (h_circle : ∀ (circle : Circle), PassesThrough circle X B)
  (P Q : Point)
  (PQ_intersects_AC : IntersectsLine P Q A C)
  (PQ_intersects_circumcircle : IntersectsLine P Q (Circumcircle A B C)) :
  ∃ S : Point, ∀ (circle : Circle), PassesThrough circle X B → PassesThrough (Line P Q) S :=
sorry

end all_lines_PQ_through_single_point_l780_780933


namespace intersection_S_T_eq_T_l780_780204

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780204


namespace probability_divisible_by_5_is_zero_l780_780848

theorem probability_divisible_by_5_is_zero :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (digit_sum n = 44) → (probability_divisible_by_5 n) = 0 :=
by
  intro n
  sorry

def digit_sum (n : ℕ) : ℕ :=
  let digits := (string n).to_list.map char.digit_to_int
  digits.foldl (λ acc d, acc + d) 0

def probability_divisible_by_5 (n : ℕ) : ℚ :=
  if (n % 5 = 0) then 1 else 0

end probability_divisible_by_5_is_zero_l780_780848


namespace S_inter_T_eq_T_l780_780305

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780305


namespace tangent_line_min_slope_l780_780820

noncomputable def curve (x : ℝ) : ℝ := x^3 + 3*x - 1

noncomputable def curve_derivative (x : ℝ) : ℝ := 3*x^2 + 3

theorem tangent_line_min_slope :
  ∃ k b : ℝ, (∀ x : ℝ, curve_derivative x ≥ 3) ∧ 
             k = 3 ∧ b = 1 ∧
             (∀ x y : ℝ, y = k * x + b ↔ 3 * x - y + 1 = 0) := 
by {
  sorry
}

end tangent_line_min_slope_l780_780820


namespace domain_of_function_range_of_function_l780_780650

-- Problem 1:
theorem domain_of_function :
  { x : ℝ | 3 - x ≥ 0 ∧ x - 1 ≠ 0 } = { x : ℝ | x ≤ 3 ∧ x ≠ 1 } :=
by sorry

-- Problem 2:
theorem range_of_function :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 4 → (-x^2 + 4 * x - 2) ∈ set.Icc (-2 : ℝ) 2 :=
by sorry

end domain_of_function_range_of_function_l780_780650


namespace correct_calculation_l780_780628

theorem correct_calculation :
  (2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  (5 * Real.sqrt 3 * 5 * Real.sqrt 2 ≠ 5 * Real.sqrt 6) ∧
  (Real.sqrt (4 + 1/2) ≠ 2 * Real.sqrt (1/2)) :=
by
  sorry

end correct_calculation_l780_780628


namespace circle_equation_l780_780334

noncomputable def hyperbola := { p : ℝ × ℝ // (p.1^2 / 4) - p.2^2 = 1 }
def right_vertex : ℝ × ℝ := (2, 0)
def asymptote1 (p : ℝ × ℝ) : Prop := p.2 = (1 / 2) * p.1
def asymptote2 (p : ℝ × ℝ) : Prop := p.2 = -(1 / 2) * p.1
def circle (p : ℝ × ℝ) : Prop := (p.1 - 2)^2 + p.2^2 = (1 / real.sqrt 5)^2

theorem circle_equation :
  ∃ k : ℝ, k = 19 / 5 ∧ (∀ p : ℝ × ℝ, circle p ↔ p.1 ^ 2 + p.2 ^ 2 - 4 * p.1 + k = 0) :=
sorry

end circle_equation_l780_780334


namespace correct_calculation_l780_780630

theorem correct_calculation : sqrt 8 / sqrt 2 = 2 :=
by
-- sorry

end correct_calculation_l780_780630


namespace intersection_S_T_eq_T_l780_780269

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780269


namespace largest_n_binomial_l780_780569

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l780_780569


namespace factorize_expression_triangle_is_isosceles_l780_780719

-- Define the first problem: Factorize the expression.
theorem factorize_expression (a b : ℝ) : a^2 - 4 * a - b^2 + 4 = (a + b - 2) * (a - b - 2) := 
by
  sorry

-- Define the second problem: Determine the shape of the triangle.
theorem triangle_is_isosceles (a b c : ℝ) (h : a^2 - ab - ac + bc = 0) : a = b ∨ a = c :=
by
  sorry

end factorize_expression_triangle_is_isosceles_l780_780719


namespace books_combination_l780_780363

theorem books_combination : nat.choose 15 3 = 455 := by
  sorry

end books_combination_l780_780363


namespace S_inter_T_eq_T_l780_780295

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780295


namespace train_travel_times_l780_780679

theorem train_travel_times (V1 V2 : ℝ) (a b c : ℝ) (h_conditions : a > 0 ∧ b > 0 ∧ c > 0 ∧ V1 > 0 ∧ V2 > V1) :
  let tm := (V1 * a) / (V2 - V1) in
  let D := V1 * (tm + b) in
  let D' := V2 * (tm + c) in
  (D = D' ∧ 
  (a / 2 + b + (Real.sqrt (a * a + 4 * b * c)) / 2 = (tm + b)) ∧ 
  (c - a / 2 + (Real.sqrt (a * a + 4 * b * c)) / 2 = (tm + c))) :=
by
  sorry

end train_travel_times_l780_780679


namespace largest_n_binomial_l780_780572

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l780_780572


namespace largest_n_for_binomial_equality_l780_780532

theorem largest_n_for_binomial_equality :
  ∃ n : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 n ) ∧
  ∀ m : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 m ) → m ≤ 6 :=
begin
  sorry
end

end largest_n_for_binomial_equality_l780_780532


namespace tenth_meeting_distance_is_zero_l780_780516

-- Define the initial parameters
noncomputable def track_circumference : ℝ := 8
noncomputable def speed_A : ℝ := 5
noncomputable def speed_B : ℝ := 3

-- Function to compute distance at nth meeting based on given conditions
noncomputable def nth_meeting_distance (n : ℕ) : ℝ :=
  if n % 2 = 0 then 0 else track_circumference / 2

-- Theorem stating the distance at the 10th meeting
theorem tenth_meeting_distance_is_zero :
  nth_meeting_distance 10 = 0 :=
by
  rw [nth_meeting_distance, if_pos (eq.refl 0)] -- Since 10 is even, distance is 0
  sorry

end tenth_meeting_distance_is_zero_l780_780516


namespace find_a_value_l780_780104

theorem find_a_value (a x1 x2 : ℝ) (h1 : a > 0) (h2 : x1 + x2 = 15) 
  (h3 : ∀ x, x^2 - 2 * a * x - 8 * a^2 < 0) : a = 15 / 2 :=
  sorry

end find_a_value_l780_780104


namespace minimum_value_of_f_g_monotonic_l780_780815

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

theorem minimum_value_of_f (a : ℝ) (h : a > 0) : (∃ x ∈ Icc 1 (Real.exp 1), f a x = -2) → a ≥ 1 :=
by
  sorry

noncomputable def g (a x : ℝ) : ℝ := a * x^2 - a * x + Real.log x

theorem g_monotonic (a : ℝ) : (∀ x1 x2, 0 < x1 → x1 < x2 → f a x1 + 2 * x1 < f a x2 + 2 * x2) ↔ (0 ≤ a ∧ a ≤ 8) :=
by
  sorry

end minimum_value_of_f_g_monotonic_l780_780815


namespace intersection_S_T_l780_780146

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780146


namespace intersection_eq_T_l780_780290

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780290


namespace intersection_S_T_eq_T_l780_780260

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780260


namespace instantaneous_velocity_at_3_l780_780495

noncomputable def displacement (t : ℝ) : ℝ := 
  - (1 / 3) * t^3 + 2 * t^2 - 5

theorem instantaneous_velocity_at_3 : 
  (deriv displacement 3 = 3) :=
by
  sorry

end instantaneous_velocity_at_3_l780_780495


namespace tangent_line_at_one_monotonic_intervals_f_less_than_zero_l780_780813

section part_I
variables (a : ℝ)
def f (x : ℝ) := Real.log x - a * x

theorem tangent_line_at_one (h : a = -2) : 
  ∃ (k : ℝ) (b : ℝ), b = -1 ∧ k = 1 ∧ (∀ x : ℝ, f x = -x + 1) := sorry
end part_I

section part_II
variables (a : ℝ)
def f (x : ℝ) := Real.log x - a * x

theorem monotonic_intervals :
  (a ≤ 0 → ∀ x, 0 < x → deriv f x > 0) ∧
  (a > 0 → ∃ (c : ℝ), (∀ x, 0 < x ∧ x < c → deriv f x > 0) ∧ (∀ x, x > c → deriv f x < 0)) := sorry
end part_II

section part_III
variables (a : ℝ)
def f (x : ℝ) := Real.log x - a * x

theorem f_less_than_zero (h : ∀ x, 0 < x → f x < 0) : 
  a > 1/Real.exp 1 := sorry
end part_III

end tangent_line_at_one_monotonic_intervals_f_less_than_zero_l780_780813


namespace count_valid_B_is_6_l780_780350

def S : Finset ℕ := {0, 1, 2, 3, 4, 5}

def is_isolated_element (A : Finset ℕ) (x : ℕ) : Prop :=
  x ∈ A ∧ (x - 1 ∉ A) ∧ (x + 1 ∉ A)

def no_isolated_elements (A : Finset ℕ) : Prop :=
  ∀ x ∈ A, ¬ is_isolated_element A x

def valid_B (B : Finset ℕ) : Prop :=
  B ⊆ S ∧ B.card = 4 ∧ no_isolated_elements B

noncomputable def count_valid_B : ℕ :=
  (S.powerset.filter valid_B).card

theorem count_valid_B_is_6 : count_valid_B = 6 := by
  sorry

end count_valid_B_is_6_l780_780350


namespace R_squared_closer_to_one_l780_780878

-- Given conditions as definitions
def coefficient_of_determination (R : ℝ) : Prop :=
  0 ≤ R ∧ R ≤ 1

def explained_variance (R : ℝ) (explained : ℝ) : Prop :=
  explained = R

-- The theorem to be proven
theorem R_squared_closer_to_one (R : ℝ) (h₁ : coefficient_of_determination R) (h₂ : explained_variance R 1) :
  (R = 1 → 
  (∀ x y : ℝ, stronger_linear_relationship x y) ∧ model_fits_better) :=
by 
  sorry

end R_squared_closer_to_one_l780_780878


namespace f_at_00_l780_780644

-- Define the polynomial f(x, y) and its properties
variable (n : ℕ) (f : ℕ → ℕ → ℚ)
variable [fact (0 < n)]  -- Ensure n is a positive natural number

-- f(x, y) is a polynomial of degree less than n
axiom degree_f_lt_n (x y : ℕ) : polynomial.degree (polynomial.C (f x y)) < n

-- f(x, y) = x/y for positive integers x, y <= n and x + y <= n+1
axiom f_eq_div (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≤ n) (hyy : y ≤ n) (hsum : x + y ≤ n + 1) : f x y = x / y

-- The main conjecture to prove
theorem f_at_00 : f 0 0 = 1 / n := by
  -- Proof will be filled in later
  sorry

end f_at_00_l780_780644


namespace complete_square_sum_l780_780420

theorem complete_square_sum (b c : ℤ) :
  let eq := (λ x : ℝ, x^2 - 10 * x + 15)
  let rewrite := (λ x : ℝ, (x + b)^2 - c)
  (∀ x : ℝ, eq x = 0 ↔ rewrite x = 0) → b + c = 5 :=
by
  sorry

end complete_square_sum_l780_780420


namespace solve_line_through_P_divides_circle_max_area_difference_l780_780044

noncomputable def line_through_P_divides_circle_max_area_difference (P : ℝ × ℝ) (r : ℝ) :=
  ∃ line_coeffs : ℝ × ℝ × ℝ, line_coeffs = (1, 1, -2)

theorem solve_line_through_P_divides_circle_max_area_difference :
  line_through_P_divides_circle_max_area_difference (1, 1) 2 =
  (1, 1, -2) :=
by
  sorry

end solve_line_through_P_divides_circle_max_area_difference_l780_780044


namespace platform_length_l780_780021

theorem platform_length
  (train_length : ℕ)
  (pole_time : ℕ)
  (platform_time : ℕ)
  (h1 : train_length = 300)
  (h2 : pole_time = 18)
  (h3 : platform_time = 39) :
  let speed := train_length / pole_time in
  let total_distance := speed * platform_time in
  total_distance - train_length = 350 :=
by 
  sorry

end platform_length_l780_780021


namespace largest_integer_comb_l780_780600

theorem largest_integer_comb (n: ℕ) (h1 : n = 6) : 
  (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) :=
by
  rw [h1]
  exact sorry

end largest_integer_comb_l780_780600


namespace commission_rate_change_amount_l780_780051

theorem commission_rate_change_amount :
  ∃ X : ℝ, (∀ S : ℝ, ∀ commission : ℝ, S = 15885.42 → commission = (S - 15000) →
  commission = 0.10 * X + 0.05 * (S - X) → X = 1822.98) :=
sorry

end commission_rate_change_amount_l780_780051


namespace point_in_second_quadrant_l780_780400

def point (x y : ℝ) : Type := ℝ × ℝ

def point_coordinates : point := (-3, 2)

def in_second_quadrant (p : point) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem point_in_second_quadrant : in_second_quadrant point_coordinates :=
by
  sorry

end point_in_second_quadrant_l780_780400


namespace arithmetic_seq_general_term_l780_780123

theorem arithmetic_seq_general_term {a : ℕ → ℤ} (h1 : a 2 = 8) (h2 : ∑ i in Finset.range 10, a (i + 1) = 185) :
  a = λ n, 3 * n - 1 :=
by
  -- Insert proof here
  sorry

end arithmetic_seq_general_term_l780_780123


namespace servings_per_pie_l780_780421

theorem servings_per_pie (serving_apples : ℝ) (guests : ℕ) (pies : ℕ) (apples_per_guest : ℝ)
  (H_servings: serving_apples = 1.5) 
  (H_guests: guests = 12)
  (H_pies: pies = 3)
  (H_apples_per_guest: apples_per_guest = 3) :
  (guests * apples_per_guest) / (serving_apples * pies) = 8 :=
by
  rw [H_servings, H_guests, H_pies, H_apples_per_guest]
  sorry

end servings_per_pie_l780_780421


namespace sum_binomial_coefficients_l780_780810

theorem sum_binomial_coefficients (a b : ℕ) (h1 : a = 2^3) (h2 : b = (2 + 1)^3) : a + b = 35 :=
by
  sorry

end sum_binomial_coefficients_l780_780810


namespace problem_statement_l780_780922

-- Define the main problem
def has_equal_ones_and_zeros (n : ℕ) : Prop :=
  let b := n.binary_repr;
  b.count '1' = b.count '0'

def number_of_valid_numbers : ℕ :=
  ∑ i in range 1001, if has_equal_ones_and_zeros i then 1 else 0

theorem problem_statement : number_of_valid_numbers % 100 = 51 :=
by
  -- Proof to be filled in
  sorry

end problem_statement_l780_780922


namespace largest_n_binomial_l780_780562

theorem largest_n_binomial :
  (∃n : ℕ, binom 10 4 + binom 10 5 = binom 11 n) → (n ≤ 6) :=
by
  sorry

end largest_n_binomial_l780_780562


namespace rainfall_ratio_l780_780414

theorem rainfall_ratio (S M T : ℝ) (h1 : M = S + 3) (h2 : S = 4) (h3 : S + M + T = 25) : T / M = 2 :=
by
  sorry

end rainfall_ratio_l780_780414


namespace water_volume_correct_l780_780637

def depth : ℝ := 2 -- Depth of the river in meters
def width : ℝ := 45 -- Width of the river in meters
def flow_rate_kmph : ℝ := 2 -- Flow rate in kilometers per hour

noncomputable def flow_rate_m_per_min : ℝ := flow_rate_kmph * 1000 / 60 -- Convert flow rate to meters per minute
noncomputable def cross_sectional_area : ℝ := depth * width -- Calculate cross-sectional area
noncomputable def volume_per_minute : ℝ := cross_sectional_area * flow_rate_m_per_min -- Calculate volume per minute

theorem water_volume_correct : volume_per_minute ≈ 2999.7 := by
  sorry

end water_volume_correct_l780_780637


namespace find_n_l780_780683

def proper_divisors (n : ℕ) : List ℕ := 
  (List.range n).filter (λ d, d > 0 ∧ n % d = 0)

def f (n : ℕ) : ℕ := 
  if n > 1 then (proper_divisors n).maximum.getD 0 + 1 else 0

theorem find_n (n : ℕ) (h1 : n ≥ 2) (h2 : f (f n) = 2) : 
  (nat.prime n ∨ ∃ q : ℕ, nat.prime q ∧ q > 2 ∧ n = 2 * (q - 1)) := 
sorry

end find_n_l780_780683


namespace hyperbola_standard_equation_l780_780330

/-- Suppose a hyperbola has asymptotes y = ± √3 * x and passes through the point M(-1, 3).
    Prove that the standard equation of the hyperbola is y^2 / 6 - x^2 / 2 = 1. -/
theorem hyperbola_standard_equation :
  (∀ x y : ℝ, (y = sqrt 3 * x ∨ y = -sqrt 3 * x) →
    (∃ (y x : ℝ), x = -1 ∧ y = 3 ∧ (x * x) / 2 - (y * y) / 6 = 1)) :=
sorry

end hyperbola_standard_equation_l780_780330


namespace max_value_of_x3_div_y4_l780_780791

theorem max_value_of_x3_div_y4 (x y : ℝ) (h1 : 3 ≤ x * y^2) (h2 : x * y^2 ≤ 8) (h3 : 4 ≤ x^2 / y) (h4 : x^2 / y ≤ 9) :
  ∃ (k : ℝ), k = 27 ∧ ∀ (z : ℝ), z = x^3 / y^4 → z ≤ k :=
by
  sorry

end max_value_of_x3_div_y4_l780_780791


namespace at_least_one_solves_l780_780656

variable (A B : Ω → Prop)
variable (P : MeasureTheory.ProbabilityMeasure Ω)

-- Given conditions
def probA : ℝ := 0.4
def probB : ℝ := 0.5

-- Main statement to prove
theorem at_least_one_solves :
  P (A ∨ B) = 0.7 :=
by
  have prob_not_A := 1 - probA
  have prob_not_B := 1 - probB
  have prob_neither := prob_not_A * prob_not_B
  have prob_at_least_one := 1 - prob_neither
  sorry

end at_least_one_solves_l780_780656


namespace S_inter_T_eq_T_l780_780302

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780302


namespace intersection_eq_T_l780_780246

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780246


namespace infinite_divisibility_l780_780917

theorem infinite_divisibility (a b c : ℤ) (h : (a + b + c) ∣ (a^2 + b^2 + c^2)) :
  ∃ infinitely_many (n : ℕ), (a + b + c) ∣ (a^n + b^n + c^n) :=
by
  sorry

end infinite_divisibility_l780_780917


namespace root_diff_sum_leq_l780_780006

-- Given conditions
def monic_quadratic_trinomial (f : Polynomial ℝ) : Prop :=
  f.degree = 2 ∧ leadingCoeff f = 1

def has_two_roots (f : Polynomial ℝ) : Prop :=
  ∃ a b, (f = Polynomial.C (a * b) * Polynomial.C (a + b))

def root_difference (f : Polynomial ℝ) : ℝ :=
  let ⟨a, b, h⟩ := exists_pair f in a - b

axiom monic_quadratic_f (f : Polynomial ℝ) : monic_quadratic_trinomial f
axiom monic_quadratic_g (g : Polynomial ℝ) : monic_quadratic_trinomial g
axiom two_root_f (f : Polynomial ℝ) : has_two_roots f
axiom two_root_g (g : Polynomial ℝ) : has_two_roots g
axiom two_root_f_plus_g (f g : Polynomial ℝ) : has_two_roots (f + g)
axiom root_diff_eq (f g : Polynomial ℝ) : root_difference f = root_difference g

-- Statement to prove
theorem root_diff_sum_leq (f g : Polynomial ℝ) :
  root_difference (f + g) ≤ root_difference f :=
by sorry

end root_diff_sum_leq_l780_780006


namespace largest_integer_comb_l780_780606

theorem largest_integer_comb (n: ℕ) (h1 : n = 6) : 
  (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) :=
by
  rw [h1]
  exact sorry

end largest_integer_comb_l780_780606


namespace intersection_S_T_eq_T_l780_780210

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780210


namespace min_cookies_divisible_by_13_l780_780709

theorem min_cookies_divisible_by_13 (a b : ℕ) : ∃ n : ℕ, n > 0 ∧ n % 13 = 0 ∧ (∃ a b : ℕ, n = 10 * a + 21 * b) ∧ n = 52 :=
by
  sorry

end min_cookies_divisible_by_13_l780_780709


namespace midpoints_single_line_l780_780514

-- Define the geometric setting: triangle, orthocenter, and intersecting lines
variables {α : Type*} [Field α]
variables [MetricSize α]

structure triangle (α) :=
(A B C : Euclidean α)
(orthocenter : Euclidean α)

structure lines_through_orthocenter (α) extends triangle α :=
(perpendicular_1 : Line α) 
(perpendicular_2 : Line α)
(perpendicular : perpendicular_1 ⊥ perpendicular_2)

-- Define the bisected segments intersection of triangle sides
def mid_segments_perpendicular_lines (t : lines_through_orthocenter α) :=
{ P1 P2 P3 : Euclidean α // 
  midpoints of the segments cut by these lines on the sides of the triangle }

-- Main theorem statement
theorem midpoints_single_line (t : lines_through_orthocenter α) :
  ∃ (L : Line α), (∀ (P : (mid_segments_perpendicular_lines t)), P ∈ L) :=
sorry

end midpoints_single_line_l780_780514


namespace congruent_triangles_of_equal_inradii_l780_780780

theorem congruent_triangles_of_equal_inradii
  (A B C D : Point)
  (r : ℝ)
  (h_ABC : inradius (triangle A B C) = r)
  (h_ABD : inradius (triangle A B D) = r)
  (h_BCD : inradius (triangle B C D) = r)
  (h_ACD : inradius (triangle A C D) = r)
  : congruent (triangle A B C) (triangle A B D) ∧
    congruent (triangle A B C) (triangle B C D) ∧
    congruent (triangle A B C) (triangle A C D) :=
by
  sorry

end congruent_triangles_of_equal_inradii_l780_780780


namespace intersection_eq_T_l780_780282

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780282


namespace icing_cubes_count_31_l780_780028

def cake_cubed (n : ℕ) := n^3

noncomputable def slabs_with_icing (n : ℕ): ℕ := 
    let num_faces := 3
    let edge_per_face := n - 1
    let edges_with_icing := num_faces * edge_per_face * (n - 2)
    edges_with_icing + (n - 2) * 4 * (n - 2)

theorem icing_cubes_count_31 : ∀ (n : ℕ), n = 5 → slabs_with_icing n = 31 :=
by
  intros n hn
  revert hn
  sorry

end icing_cubes_count_31_l780_780028


namespace intersection_S_T_eq_T_l780_780258

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780258


namespace alice_acorns_purchase_l780_780057

variable (bob_payment : ℕ) (alice_payment_rate : ℕ) (price_per_acorn : ℕ)

-- Given conditions
def bob_paid : Prop := bob_payment = 6000
def alice_paid : Prop := alice_payment_rate = 9
def acorn_price : Prop := price_per_acorn = 15

-- Proof statement
theorem alice_acorns_purchase
  (h1 : bob_paid bob_payment)
  (h2 : alice_paid alice_payment_rate)
  (h3 : acorn_price price_per_acorn) :
  ∃ n : ℕ, n = (alice_payment_rate * bob_payment) / price_per_acorn ∧ n = 3600 := 
by
  sorry

end alice_acorns_purchase_l780_780057


namespace infinite_solutions_diophantine_l780_780448

theorem infinite_solutions_diophantine :
  ∃ (infinitely_many_triples : set (ℕ × ℕ × ℕ)),
  (∀ (x y z : ℕ), (x, y, z) ∈ infinitely_many_triples → x > 2008 ∧ y > 2008 ∧ z > 2008 ∧ x^2 + y^2 + z^2 - x * y * z + 10 = 0) ∧
  set.infinite infinitely_many_triples :=
sorry

end infinite_solutions_diophantine_l780_780448


namespace largest_integer_binom_l780_780587

theorem largest_integer_binom (n : ℕ) (h : nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) : n = 6 :=
sorry

end largest_integer_binom_l780_780587


namespace cos_double_angle_neg_l780_780365

theorem cos_double_angle_neg (α : ℝ) (h : tan (α + π / 4) < 0) : cos (2 * α) < 0 := 
sorry

end cos_double_angle_neg_l780_780365


namespace intersection_S_T_eq_T_l780_780277

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780277


namespace diagonal_difference_is_4_l780_780654

def original_matrix : Matrix (Fin 4) (Fin 4) ℕ := 
  ![![1, 2, 3, 4], 
    ![8, 9, 10, 11], 
    ![15, 16, 17, 18], 
    ![22, 23, 24, 25]]

def reversed_matrix : Matrix (Fin 4) (Fin 4) ℕ := 
  ![![1, 2, 3, 4], 
    ![11, 10, 9, 8], 
    ![15, 16, 17, 18], 
    ![25, 24, 23, 22]]

def diagonal_sum (m : Matrix (Fin 4) (Fin 4) ℕ) : ℕ := 
  m 0 0 + m 1 1 + m 2 2 + m 3 3

def anti_diagonal_sum (m : Matrix (Fin 4) (Fin 4) ℕ) : ℕ := 
  m 0 3 + m 1 2 + m 2 1 + m 3 0

theorem diagonal_difference_is_4 : 
  |(diagonal_sum reversed_matrix) - (anti_diagonal_sum reversed_matrix)| = 4 :=
by
  sorry

end diagonal_difference_is_4_l780_780654


namespace area_of_cosine_curve_l780_780963

noncomputable def area_under_curve_cos : ℝ :=
  ∫ x in (- (Real.pi) / 2) .. Real.pi, Real.cos x

theorem area_of_cosine_curve : area_under_curve_cos = 3 := by
  sorry

end area_of_cosine_curve_l780_780963


namespace intersection_eq_T_l780_780250

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780250


namespace intersection_eq_T_l780_780294

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780294


namespace find_plane_equation_l780_780758

variable {R : Type*} [LinearOrderedField R]

def point1 : R × R × R := (0, 2, 3)
def point2 : R × R × R := (2, 0, 3)
def plane1_normal : R × R × R := (1, -1, 4)

noncomputable def plane_equation (A B C D : ℤ) : Prop :=
  (A ≠ 0 ∧ (∀ k : ℤ, k ≠ 0 → k * A = A') ∧ A > 0 ∧ int.gcd (A.nat_abs) (B.nat_abs) (C.nat_abs) (D.nat_abs) = 1 ∧
  ∀ (x y z : R), (x, y, z) = point1 → A * x + B * y + C * z + D = 0) ∧
  ∀ (x y z : R), (x, y, z) = point2 → A * x + B * y + C * z + D = 0) ∧
  (cross_product 
    (vector_from_points point1 point2) 
    plane1_normal = (A, B, C))

theorem find_plane_equation : 
  ∃ (A B C D : ℤ), 
  plane_equation A B C D ∧ 
  (∀ x y z : R, 2 * x - 2 * y + z + 1 = 0) :=
sorry

end find_plane_equation_l780_780758


namespace hexagon_perimeter_is_60sqrt3_l780_780973

noncomputable def hexagon_perimeter (h : ℝ) (s : ℝ) : ℝ :=
  6 * s

theorem hexagon_perimeter_is_60sqrt3
  (h : ℝ)
  (height_to_side_relation : ∀ s : ℝ, h = (√3 / 2) * s)
  (hexagon_height : h = 15) :
  hexagon_perimeter h (15 * 2 / √3) = 60 * √3 :=
by {
  sorry
}

end hexagon_perimeter_is_60sqrt3_l780_780973


namespace basis_proof_l780_780924

variable {α : Type*} [AddCommGroup α] [VectorSpace ℝ α]

def is_basis (v1 v2 : α) : Prop :=
  ∃ a1 a2 b1 b2 : ℝ, a1 • v1 + a2 • v2 ≠ 0 ∧ b1 • v1 + b2 • v2 ≠ 0

variable (e1 e2 : α)

axiom basis_e1_e2 : is_basis e1 e2

theorem basis_proof :
  is_basis (e1 + e2) (e1 - e2) :=
sorry

end basis_proof_l780_780924


namespace find_n_equals_4072323_l780_780763

theorem find_n_equals_4072323 :
  ∃ n : ℕ, 
    (∑ k in finset.range n, 1 / (real.sqrt k + real.sqrt (k + 1))) = 2017 ∧ 
    n = 4072323 := by
  sorry

end find_n_equals_4072323_l780_780763


namespace largest_integer_binom_l780_780580

theorem largest_integer_binom (n : ℕ) (h : nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) : n = 6 :=
sorry

end largest_integer_binom_l780_780580


namespace part_a_kingdom_of_cities_possible_part_b_kingdom_of_cities_impossible_l780_780887

-- Part (a)
theorem part_a_kingdom_of_cities_possible :
  ∃ (G : SimpleGraph (Fin 16)), (∀ v : Fin 16, G.degree v = 5) ∧ 
  (∀ u v : Fin 16, u ≠ v → ∃ w : Fin 16, (G.adj u v ∨ (G.adj u w ∧ G.adj w v))) :=
sorry

-- Part (b)
theorem part_b_kingdom_of_cities_impossible :
  ¬∃ (G : SimpleGraph (Fin 16)), (∀ v : Fin 16, G.degree v = 4) ∧ 
  (∀ u v : Fin 16, u ≠ v → ∃ w : Fin 16, (G.adj u v ∨ (G.adj u w ∧ G.adj w v))) :=
sorry

end part_a_kingdom_of_cities_possible_part_b_kingdom_of_cities_impossible_l780_780887


namespace count_of_three_digit_integers_in_set_l780_780828

theorem count_of_three_digit_integers_in_set (S : finset ℕ) 
  (hS : S = {3, 3, 4, 4, 4, 7, 8}) :
  (S.card = 7) → 
  (∀ x ∈ S, 3 ≤ x ∧ x ≤ 8) →
  (\sum (a b c : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S), 
    if (a = b ∧ b = c ∧ S.count a ≥ 3) then 1 else 
    if (a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ S.count a ≥ 1 ∧ S.count b ≥ 1 ∧ S.count c ≥ 1) then 1 else 
    if ((a = b ∧ b ≠ c ∧ S.count a ≥ 2 ∧ S.count c ≥ 1) ∨ (a = c ∧ a ≠ b ∧ S.count a ≥ 2 ∧ S.count b ≥ 1) ∨ (b = c ∧ a ≠ b ∧ S.count b ≥ 2 ∧ S.count a ≥ 1)) then 1 else 0) 
  = 43 := by sorry

end count_of_three_digit_integers_in_set_l780_780828


namespace even_function_condition_iff_l780_780777

theorem even_function_condition_iff (m : ℝ) :
    (∀ x : ℝ, (m * 2^x + 2^(-x)) = (m * 2^(-x) + 2^x)) ↔ (m = 1) :=
by
  sorry

end even_function_condition_iff_l780_780777


namespace intersection_of_S_and_T_l780_780235

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780235


namespace james_total_matches_l780_780903

def boxes_count : ℕ := 5 * 12
def matches_per_box : ℕ := 20
def total_matches (boxes : ℕ) (matches_per_box : ℕ) : ℕ := boxes * matches_per_box

theorem james_total_matches : total_matches boxes_count matches_per_box = 1200 :=
by {
  sorry
}

end james_total_matches_l780_780903


namespace log_condition_necessity_log_condition_insufficiency_log_condition_necessity_but_not_sufficient_l780_780647

theorem log_condition_necessity (a b : ℝ) :
  (10^a > 10^b) → (lga > lgb) :=
sorry

theorem log_condition_insufficiency (a b : ℝ) :
  (lga > lgb) ↛ (10^a > 10^b) :=
sorry

theorem log_condition_necessity_but_not_sufficient (a b : ℝ) :
  (10^a > 10^b) ∧ ¬(∀ a b, (lga > lgb) → (10^a > 10^b)) :=
by
  apply And.intro
  · apply log_condition_necessity
  · apply log_condition_insufficiency
  sorry

end log_condition_necessity_log_condition_insufficiency_log_condition_necessity_but_not_sufficient_l780_780647


namespace intersection_S_T_eq_T_l780_780132

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780132


namespace extracurricular_books_counts_l780_780994

theorem extracurricular_books_counts 
  (a b c d : ℕ)
  (h1 : b + c + d = 110)
  (h2 : a + c + d = 108)
  (h3 : a + b + d = 104)
  (h4 : a + b + c = 119) :
  a = 37 ∧ b = 39 ∧ c = 43 ∧ d = 28 :=
by
  sorry

end extracurricular_books_counts_l780_780994


namespace S_inter_T_eq_T_l780_780297

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780297


namespace problem1_problem2_problem3_l780_780094

section Problem1

variable (a b : ℝ)
variable (h_condition : a^b * b^a + log a b = 0)
variable (ha_pos : 0 < a)
variable (hb_pos : 0 < b)

theorem problem1 : ab + 1 < a + b :=
sorry

end Problem1

section Problem2

variable (b : ℝ)
variable (h_condition : 2^b * b^2 + log 2 b = 0)
variable (hb_pos : 0 < b)

theorem problem2 : 2^b < 1 / b :=
sorry

end Problem2

section Problem3

variable (b : ℝ)
variable (h_condition : 2^b * b^2 + log 2 b = 0)
variable (hb_pos : 0 < b)

theorem problem3 : (2 * b + 1 - real.sqrt 5) * (3 * b - 2) < 0 :=
sorry

end Problem3

end problem1_problem2_problem3_l780_780094


namespace intersection_S_T_eq_T_l780_780274

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780274


namespace angle_between_a_and_b_is_60_deg_l780_780837

-- Define the unit vectors e1 and e2, and the angle between them
variables (e1 e2 : EuclideanSpace ℝ (Fin 3))
variable (h_unit_e1 : ∥ e1 ∥ = 1)
variable (h_unit_e2 : ∥ e2 ∥ = 1)
variable (angle_e1_e2 : real_angle e1 e2 = π / 3)

-- Define the vectors a and b based on given conditions
def a : EuclideanSpace ℝ (Fin 3) := e1 + e2
def b : EuclideanSpace ℝ (Fin 3) := -e1 + 2 * e2

-- The theorem we want to prove
theorem angle_between_a_and_b_is_60_deg : real_angle a b = π / 3 :=
sorry

end angle_between_a_and_b_is_60_deg_l780_780837


namespace minimum_frosting_time_l780_780707

def ann_time_per_cake := 8 -- Ann's time per cake in minutes
def bob_time_per_cake := 6 -- Bob's time per cake in minutes
def carol_time_per_cake := 10 -- Carol's time per cake in minutes
def passing_time := 1 -- time to pass a cake from one person to another in minutes
def total_cakes := 10 -- total number of cakes to be frosted

theorem minimum_frosting_time : 
  (ann_time_per_cake + passing_time + bob_time_per_cake + passing_time + carol_time_per_cake) + (total_cakes - 1) * carol_time_per_cake = 116 := 
by 
  sorry

end minimum_frosting_time_l780_780707


namespace james_total_matches_l780_780902

def boxes_count : ℕ := 5 * 12
def matches_per_box : ℕ := 20
def total_matches (boxes : ℕ) (matches_per_box : ℕ) : ℕ := boxes * matches_per_box

theorem james_total_matches : total_matches boxes_count matches_per_box = 1200 :=
by {
  sorry
}

end james_total_matches_l780_780902


namespace isosceles_iff_congruent_bisectors_l780_780397

theorem isosceles_iff_congruent_bisectors (A B C M N : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited M] [inhabited N]
    (triangle_ABC : triangle A B C)
    (BM_is_angle_bisector : is_angle_bisector A B M)
    (CN_is_angle_bisector : is_angle_bisector A C N) :
    (is_isosceles_triangle A B C) ↔ (BM ≃ CN) := 
sorry

end isosceles_iff_congruent_bisectors_l780_780397


namespace find_difference_l780_780376

variable (k1 k2 t1 t2 : ℝ)

theorem find_difference (h1 : t1 = 5 / 9 * (k1 - 32))
                        (h2 : t2 = 5 / 9 * (k2 - 32))
                        (h3 : t1 = 105)
                        (h4 : t2 = 80) :
  k1 - k2 = 45 :=
by
  sorry

end find_difference_l780_780376


namespace intersection_S_T_eq_T_l780_780219

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780219


namespace complex_num_z_imaginary_square_l780_780331

theorem complex_num_z_imaginary_square (z : ℂ) (h1 : z.im ≠ 0) (h2 : z.re = 0) (h3 : ((z + 1) ^ 2).re = 0) :
  z = Complex.I ∨ z = -Complex.I :=
by
  sorry

end complex_num_z_imaginary_square_l780_780331


namespace polar_to_cartesian_parametric_to_cartesian_l780_780882

-- Problem for C1
theorem polar_to_cartesian (rho theta : ℝ) (h : rho = 6 * real.cos theta) :
  (let x := rho * real.cos theta in
   let y := rho * real.sin theta in
   (x - 3)^2 + y^2 = 9) :=
by
  sorry

-- Problem for C2
theorem parametric_to_cartesian (a b φ : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : a = 2) (h4 : b = 1) (x y : ℝ)
  (h : (x = a * real.cos φ) ∧ (y = b * real.sin φ) ∧ (1 = a * real.cos (π/3)) ∧ (sqrt 3 / 2 = b * real.sin (π/3))) :
  x^2 / 4 + y^2 = 1 :=
by
  sorry

end polar_to_cartesian_parametric_to_cartesian_l780_780882


namespace no_perfect_square_in_range_l780_780356

def f (n : ℕ) : ℕ := 2 * n^2 + 3 * n + 2

theorem no_perfect_square_in_range : ∀ (n : ℕ), 5 ≤ n → n ≤ 15 → ¬ ∃ (m : ℕ), f n = m^2 := by
  intros n h1 h2
  sorry

end no_perfect_square_in_range_l780_780356


namespace count_valid_N_l780_780355

def valid_four_digit_numbers : Finset ℕ :=
  (Finset.range (7000 - 4000)).map (Nat.add_right_injective 4000)

def is_multiple_of_5_and_3 (n : ℕ) : Prop :=
  n % 5 = 0 ∧ n % 3 = 0

def valid_b_c_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (λ pair, 2 ≤ pair.1 ∧ pair.1 < pair.2 ∧ pair.2 ≤ 6) $
    Finset.product (Finset.range' 2 5) (Finset.range' 3 4)

theorem count_valid_N : ∃ count : ℕ, count = 30 ∧
  count = (valid_four_digit_numbers.filter (λ N,
    let a := N / 1000;
    let b := N / 100 % 10;
    let c := N / 10 % 10;
    let d := N % 10;
    a ∈ {4, 5, 6} ∧
    d ∈ {0, 5} ∧
    is_multiple_of_5_and_3 N ∧
    (b, c) ∈ valid_b_c_pairs ∧
    (a + b + c + d) % 3 = 0)).card := by
sorry

end count_valid_N_l780_780355


namespace largest_n_for_binomial_equality_l780_780538

theorem largest_n_for_binomial_equality :
  ∃ n : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 n ) ∧
  ∀ m : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 m ) → m ≤ 6 :=
begin
  sorry
end

end largest_n_for_binomial_equality_l780_780538


namespace gcd_16_12_eq_4_l780_780515

theorem gcd_16_12_eq_4 : Nat.gcd 16 12 = 4 := by
  -- Skipping proof using sorry
  sorry

end gcd_16_12_eq_4_l780_780515


namespace ratio_A_B_l780_780740

noncomputable def A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
noncomputable def B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)

theorem ratio_A_B :
  let r := (A : ℚ) / (B : ℚ) in 0 < r ∧ r < 1 :=
by {
  -- Proof steps are omitted with sorry
  sorry
}

end ratio_A_B_l780_780740


namespace largest_n_binom_identity_l780_780557

open Nat

theorem largest_n_binom_identity :
  (∃ n, binom 11 n = binom 10 4 + binom 10 5) ∧
  (∀ m, binom 11 m = binom 10 4 + binom 10 5 → m ≤ 6) :=
by
  sorry

end largest_n_binom_identity_l780_780557


namespace geometric_sequence_derivative_l780_780765

noncomputable def a (n : ℕ) : ℝ := 2^(n-3)

def f (x : ℝ) : ℝ := ∑ n in Finset.range 10, a(n+1) * x^(n+1)

theorem geometric_sequence_derivative :
  f ' (1/2) = 55 / 4 :=
by
  -- Define the individual terms of the series
  have a1 := a 1
  have a7 := a 7

  -- conditions
  have cond1 : a1 * a7 = 4 := by sorry
  have cond2 : a 6 = 8 := by sorry

  -- (additional required steps to reach the final result)
  -- performing the derivative and plugging in 1/2 with the given conditions
  sorry

end geometric_sequence_derivative_l780_780765


namespace quadratic_function_decreasing_l780_780377

theorem quadratic_function_decreasing (a : ℝ) :
  (∀ x : ℝ, x ∈ Iic 1 → (2 * (a + x) ≤ 0)) → a ≤ -1 :=
by 
  intro h
  -- suppose the logical steps and calculations
  have key_condition := h 1
  sorry

end quadratic_function_decreasing_l780_780377


namespace octahedron_walk_probability_is_half_l780_780076

noncomputable def octahedron_walk_probability : ℚ :=
let p_top_bottom := 2/4 in
let p_c_top_bottom := p_top_bottom in
p_c_top_bottom

theorem octahedron_walk_probability_is_half :
  octahedron_walk_probability = 1/2 :=
by
  rw octahedron_walk_probability
  norm_num
  sorry

end octahedron_walk_probability_is_half_l780_780076


namespace combined_annual_income_after_expenses_l780_780710

noncomputable def brady_monthly_incomes : List ℕ := [150, 200, 250, 300, 200, 150, 180, 220, 240, 270, 300, 350]
noncomputable def dwayne_monthly_incomes : List ℕ := [100, 150, 200, 250, 150, 120, 140, 190, 180, 230, 260, 300]
def brady_annual_expense : ℕ := 450
def dwayne_annual_expense : ℕ := 300

def annual_income (monthly_incomes : List ℕ) : ℕ :=
  monthly_incomes.foldr (· + ·) 0

theorem combined_annual_income_after_expenses :
  (annual_income brady_monthly_incomes - brady_annual_expense) +
  (annual_income dwayne_monthly_incomes - dwayne_annual_expense) = 3930 :=
by
  sorry

end combined_annual_income_after_expenses_l780_780710


namespace regular_tetrahedron_pairs_determine_plane_l780_780713

/-- In a regular tetrahedron, the number of unordered pairs of edges that determine a plane is 15. -/
theorem regular_tetrahedron_pairs_determine_plane : 
  let num_edges := 6 in
  let num_pairs := num_edges.choose 2 in
  num_pairs = 15 := 
by
  sorry

end regular_tetrahedron_pairs_determine_plane_l780_780713


namespace cubic_coefficients_integer_extrema_inflection_l780_780729

-- Given condition: the coefficients (a, b, c) must be such that the coordinates of the
-- extreme values and the inflection point are integers.
theorem cubic_coefficients_integer_extrema_inflection 
  (n p c : ℤ) : 
  ∃ (a b : ℤ), 
  a = 3 * n ∧
  b = 3 * (n^2 - p^2) ∧
  ∀ x : ℝ, 
  let y := x^3 + a * x^2 + b * x + c in 
  (∃ xi : ℝ, y'' xi = 0 ∧ xi ∈ ℤ) ∧
  (∃ xe1 xe2 : ℝ, y' xe1 = 0 ∧ y' xe2 = 0 ∧ xe1 ∈ ℤ ∧ xe2 ∈ ℤ) := 
sorry

end cubic_coefficients_integer_extrema_inflection_l780_780729


namespace find_height_l780_780505

-- Definitions from the problem conditions
def Area : ℕ := 442
def width : ℕ := 7
def length : ℕ := 8

-- The statement to prove
theorem find_height (h : ℕ) (H : 2 * length * width + 2 * length * h + 2 * width * h = Area) : h = 11 := 
by
  sorry

end find_height_l780_780505


namespace largest_n_binomial_l780_780561

theorem largest_n_binomial :
  (∃n : ℕ, binom 10 4 + binom 10 5 = binom 11 n) → (n ≤ 6) :=
by
  sorry

end largest_n_binomial_l780_780561


namespace intersection_S_T_eq_T_l780_780261

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780261


namespace find_lambda_l780_780332

theorem find_lambda (λ : ℝ) 
  (h1 : ∃ μ : ℝ, (2, 3, -1) = (μ * 4, μ * λ, μ * -2)) : λ = 6 :=
by
  sorry

end find_lambda_l780_780332


namespace intersection_S_T_eq_T_l780_780129

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780129


namespace contrapositive_of_zero_implication_l780_780500

theorem contrapositive_of_zero_implication (a b : ℝ) :
  (a = 0 ∨ b = 0 → a * b = 0) → (a * b ≠ 0 → (a ≠ 0 ∧ b ≠ 0)) :=
by
  intro h
  sorry

end contrapositive_of_zero_implication_l780_780500


namespace smallest_n_for_even_T_n_l780_780726

theorem smallest_n_for_even_T_n :
  ∃ (n : ℕ), 0 < n ∧ (n * 10^(n-1) * 285) % 2 = 0 :=
by
  have P : ℕ := (\sum i in (finset.range 9).map finset.succ, i^2)
  have P_val : P = 285 := by sorry
  use 1
  constructor
  · exact nat.one_pos
  · rw [nat.mul_mod, nat.mul_mod, pow_one, nat.mul_mod, P_val]
    norm_num
    sorry

end smallest_n_for_even_T_n_l780_780726


namespace solution_system_of_equations_l780_780823

theorem solution_system_of_equations (x y : ℝ) (k1 k2 : ℤ)
  (h1 : √2 * sin x = sin y)
  (h2 : √2 * cos x = √3 * cos y) :
  (x = ± π / 6 + π * k2) ∧ (y = ± π / 4 + π * k1) ∧ (k1 % 2 = k2 % 2) :=
sorry

end solution_system_of_equations_l780_780823


namespace locus_of_A_l780_780443

-- Define the fixed points B and C
variables (B C : Point)
-- Define the roving point A
variable (A : Point)
-- Define the orthocenter H and centroid G of triangle ABC
variables (H G : Point)
-- Introduce the midpoint M of HG lying on BC
variable (M : Point)
variable (a : Real)

-- Given conditions
axiom midpoint_M_on_BC : midpoint HG = M ∧ M ∈ line BC
axiom coordinates_B : B = Point (Real.ofInt (-a)) Real.zero
axiom coordinates_C : C = Point (Real.ofInt a) Real.zero
-- Centroid G of triangle ABC
axiom centroid_G : G = centroid A B C
-- Orthocenter H of triangle ABC satisfying the orthogonality condition
axiom orthocenter_H_at_M : H = orthocenter A B C

-- Statement to be proved: The locus of A is the hyperbola
theorem locus_of_A (x y : Real) :
  (A = Point x y) →
  ( (x - Real.ofInt a)^2 - (y^2 / 3) = a^2 ) :=
begin
  sorry
end

end locus_of_A_l780_780443


namespace estimate_ratio_l780_780732

theorem estimate_ratio (A B : ℕ) (A_def : A = 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28)
  (B_def : B = 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20) : 0 < A / B ∧ A / B < 1 := by
  sorry

end estimate_ratio_l780_780732


namespace intersection_eq_T_l780_780239

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780239


namespace solve_for_question_mark_l780_780010

theorem solve_for_question_mark :
  let question_mark := 4135 / 45
  (45 * question_mark) + (625 / 25) - (300 * 4) = 2950 + (1500 / (75 * 2)) :=
by
  let question_mark := 4135 / 45
  sorry

end solve_for_question_mark_l780_780010


namespace james_total_matches_l780_780904

def boxes_count : ℕ := 5 * 12
def matches_per_box : ℕ := 20
def total_matches (boxes : ℕ) (matches_per_box : ℕ) : ℕ := boxes * matches_per_box

theorem james_total_matches : total_matches boxes_count matches_per_box = 1200 :=
by {
  sorry
}

end james_total_matches_l780_780904


namespace proof_l780_780938

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x ≥ 1
def q : Prop := ∀ x : ℝ, 0 < x → Real.exp x > Real.log x

-- The theorem statement
theorem proof : p ∧ q := by sorry

end proof_l780_780938


namespace smallest_positive_period_of_cosine_l780_780086

theorem smallest_positive_period_of_cosine :
  let y := λ x : ℝ, cos (2 * x - (π / 3))
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, y (x + T) = y x :=
begin
  use π,
  split,
  { norm_num, },
  sorry
end

end smallest_positive_period_of_cosine_l780_780086


namespace find_GQ_in_triangle_XYZ_l780_780890

noncomputable def GQ_length (XY XZ YZ : ℕ) : ℚ :=
  let a := XY
  let b := XZ
  let c := YZ
  let s := (a + b + c) / 2
  let area_XYZ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let height_XR := (2 * area_XYZ) / c
  height_XR / 3

theorem find_GQ_in_triangle_XYZ (XY XZ YZ : ℕ) (hXY : XY = 12) (hXZ : XZ = 15) (hYZ : YZ = 23) : 
  GQ_length XY XZ YZ = 40 / 23 := 
by 
  -- proof steps can go here
  sorry

end find_GQ_in_triangle_XYZ_l780_780890


namespace intersection_eq_T_l780_780242

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780242


namespace intersection_S_T_eq_T_l780_780203

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780203


namespace largest_n_binom_identity_l780_780555

open Nat

theorem largest_n_binom_identity :
  (∃ n, binom 11 n = binom 10 4 + binom 10 5) ∧
  (∀ m, binom 11 m = binom 10 4 + binom 10 5 → m ≤ 6) :=
by
  sorry

end largest_n_binom_identity_l780_780555


namespace largest_integer_binom_l780_780614

theorem largest_integer_binom
  (n : ℕ)
  (h₁ : binom 10 4 + binom 10 5 = binom 11 n) :
  n = 6 :=
sorry

end largest_integer_binom_l780_780614


namespace find_integer_n_l780_780681

theorem find_integer_n :
  ∃ n : ℕ,
    (∃ (divisors : list ℕ), 
      list.sorted (<) divisors ∧
      list.length divisors = 12 ∧
      (∀ d ∈ divisors, n % d = 0) ∧ 
      divisors.head = 1 ∧ 
      divisors.last = some n ∧
      (let m := (divisors.nth_le 3 _ - 1) in
       list.nth_le divisors m _ = ((divisors.head + divisors.nth_le 1 _ + divisors.nth_le 3 _) * divisors.nth_le 7 _)))
    ∧ 
    n = 1989 :=
sorry

end find_integer_n_l780_780681


namespace total_bill_l780_780056

def everyonePaidEqually (x y z : ℕ) (share : ℕ) : Prop :=
  x = share ∧ y = share ∧ z = share

theorem total_bill (x y z : ℕ) (share total : ℕ) (h : everyonePaidEqually x y z share) :
  total = x + y + z → total = 99 :=
by
  intros
  have h1 : x = 33 := h.left
  have h2 : y = 33 := h.right.left
  have h3 : z = 33 := h.right.right
  calc
    total = x + y + z         : by assumption
    ...   = 33 + 33 + 33      : by rw [h1, h2, h3]
    ...   = 99                : by norm_num

end total_bill_l780_780056


namespace hamiltonian_path_exists_l780_780873

open Finset

-- Define a Hamiltonian path
def is_hamiltonian_path {V : Type*} (G : SimpleGraph V) (p : List V) : Prop :=
  p.nodup ∧ ∀ v ∈ p.to_finset, G.degree v > 0 ∧ p.head? = some v ∧ ∀ u ∈ G.neighbor_set v, List.last p = some u ∧ u ∉ p.tail

noncomputable def example_graph : SimpleGraph (Fin n) := sorry -- Define the example graph with 20 vertices and a degree of at least 10 for each vertex.

theorem hamiltonian_path_exists (G : SimpleGraph (Fin 20)) 
  (h1 : ∀ v : Fin 20, G.degree v ≥ 10) : 
  ∃ p : List (Fin 20), is_hamiltonian_path G p :=
begin
  sorry -- The proof itself is omitted as per the instructions.
end

end hamiltonian_path_exists_l780_780873


namespace dimes_given_l780_780955

theorem dimes_given (initial_dimes final_dimes dimes_dad_gave : ℕ)
  (h1 : initial_dimes = 9)
  (h2 : final_dimes = 16)
  (h3 : final_dimes = initial_dimes + dimes_dad_gave) :
  dimes_dad_gave = 7 :=
by
  rw [h1, h2] at h3
  linarith

end dimes_given_l780_780955


namespace determine_value_lg_expression_l780_780728

-- Definition of terms
def lg : ℝ → ℝ := log10

lemma log_properties :
  ∀ (a b : ℝ), lg a + lg b = lg (a * b) :=
begin
  intros a b,
  exact log10_mul a b,
end

theorem determine_value_lg_expression :
  ∃ x, x = 2 → (lg 2) ^ 2 + (lg 2) * (lg 5) + (lg 50) = 2 :=
begin
  let lg2_5 : lg 2 + lg 5 = 1 := sorry,
  let lg100 : lg 100 = 2 := sorry,
  use 2,
  intro h,
  sorry
end

end determine_value_lg_expression_l780_780728


namespace coefficient_a9_l780_780109

theorem coefficient_a9 :
  ∀ (a : ℕ → ℝ), 
  (1 + x)^10 = ∑ i in finset.range(11), a i * (1 - x)^i →
  a 9 = -20 := 
by sorry

end coefficient_a9_l780_780109


namespace largest_n_binomial_l780_780577

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l780_780577


namespace ax_by_powers_l780_780367

theorem ax_by_powers (a b x y : ℝ) (h1 : a * x + b * y = 5) 
                      (h2: a * x^2 + b * y^2 = 11)
                      (h3: a * x^3 + b * y^3 = 25)
                      (h4: a * x^4 + b * y^4 = 59) : 
                      a * x^5 + b * y^5 = 145 := 
by 
  -- Include the proof steps here if needed 
  sorry

end ax_by_powers_l780_780367


namespace intersection_eq_T_l780_780293

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780293


namespace P_and_Q_together_l780_780458

theorem P_and_Q_together (W : ℝ) (H : W > 0) :
  (1 / (1 / 4 + 1 / (1 / 3 * (1 / 4)))) = 3 :=
by
  sorry

end P_and_Q_together_l780_780458


namespace jade_savings_per_month_l780_780844

def jade_monthly_income : ℝ := 1600
def jade_living_expense_rate : ℝ := 0.75
def jade_insurance_rate : ℝ := 0.2

theorem jade_savings_per_month : 
  jade_monthly_income * (1 - jade_living_expense_rate - jade_insurance_rate) = 80 := by
  sorry

end jade_savings_per_month_l780_780844


namespace intersection_eq_T_l780_780284

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780284


namespace units_digit_of_m_squared_plus_3_to_the_m_l780_780440

def m : ℕ := 2021^3 + 3^2021

theorem units_digit_of_m_squared_plus_3_to_the_m 
  (hm : m = 2021^3 + 3^2021) : 
  ((m^2 + 3^m) % 10) = 7 := 
by 
  -- Here you would input the proof steps, however, we skip it now with sorry.
  sorry

end units_digit_of_m_squared_plus_3_to_the_m_l780_780440


namespace geom_seq_log_sum_l780_780117

theorem geom_seq_log_sum (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n : ℕ, a n = r^(n-1))
  (h_pos : ∀ n : ℕ, 0 < a n) (h_eq : a 4 * a 5 + a 1 * a 8 = 18) :
  (List.range 10).sum (λ n, log 3 (a n)) = 10 :=
by sorry

end geom_seq_log_sum_l780_780117


namespace intersection_S_T_eq_T_l780_780263

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780263


namespace range_of_a_l780_780921

-- Definitions and conditions
def f (a x : ℝ) : ℝ := x + a / x
def g (x : ℝ) : ℝ := x - Real.log x

-- The range of possible values of a
theorem range_of_a (a : ℝ) (h₀ : 0 < a) (h₁ : a ≤ 1) 
    (f_ge_g : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ ≤ Real.exp 1 → 1 ≤ x₂ → x₂ ≤ Real.exp 1 → f a x₁ ≥ g x₂) :
    e - 2 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l780_780921


namespace largest_n_binom_l780_780539

theorem largest_n_binom (n : ℕ) (h : 0 ≤ n ∧ n ≤ 11) : 
  (binomial 10 4 + binomial 10 5 = binomial 11 n) → n = 6 :=
by
  sorry

end largest_n_binom_l780_780539


namespace min_value_a_over_b_l780_780772

theorem min_value_a_over_b (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 2 * Real.sqrt a + b = 1) : ∃ c, c = 0 := 
by
  -- We need to show that the minimum value of a / b is 0 
  sorry

end min_value_a_over_b_l780_780772


namespace initial_investment_proof_l780_780706

def compound_interest 
  (initial_investment : ℝ) 
  (annual_rate : ℝ) 
  (monthly_compound : ℕ) 
  (time_years : ℝ) : ℝ := 
initial_investment * (1 + annual_rate / monthly_compound)^(monthly_compound * time_years)

theorem initial_investment_proof (A : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) (interest_accumulated : ℝ) :
  r = 0.08 → n = 12 → t = 1.5 → interest_accumulated = 3542 →
  A = 27913.58 →
  compound_interest A r n t - A = interest_accumulated :=
by
  intros,
  rw [H, H_1, H_3, H_2],
  sorry

end initial_investment_proof_l780_780706


namespace smallest_class_number_l780_780687

theorem smallest_class_number (x : ℕ)
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) + (x + 15) = 57) :
  x = 2 :=
by sorry

end smallest_class_number_l780_780687


namespace monica_milk_l780_780731

theorem monica_milk (don_milk : ℚ) (rachel_fraction : ℚ) (monica_fraction : ℚ) (h_don : don_milk = 3 / 4)
  (h_rachel : rachel_fraction = 1 / 2) (h_monica : monica_fraction = 1 / 3) :
  monica_fraction * (rachel_fraction * don_milk) = 1 / 8 :=
by
  sorry

end monica_milk_l780_780731


namespace answer_l780_780790

def p := ∃ x : ℝ, x - 2 > Real.log x
def q := ∀ x : ℝ, Real.exp x > 1

theorem answer (hp : p) (hq : ¬ q) : p ∧ ¬ q :=
  by
    exact ⟨hp, hq⟩

end answer_l780_780790


namespace min_sum_of_5_numbers_l780_780508

theorem min_sum_of_5_numbers (a b c d e : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
    (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e) (h8 : c ≠ d) (h9 : c ≠ e) (h10 : d ≠ e)
    (h11 : ∀ (x y : ℕ), x ∈ {a, b, c, d, e} → y ∈ {a, b, c, d, e} → x ≠ y → (x * y) % 12 = 0) :
    a + b + c + d + e = 62 :=
sorry

end min_sum_of_5_numbers_l780_780508


namespace polygon_angles_change_l780_780860

theorem polygon_angles_change (n : ℕ) :
  let initial_sum_interior := (n - 2) * 180
  let initial_sum_exterior := 360
  let new_sum_interior := (n + 2 - 2) * 180
  let new_sum_exterior := 360
  new_sum_exterior = initial_sum_exterior ∧ new_sum_interior - initial_sum_interior = 360 :=
by
  sorry

end polygon_angles_change_l780_780860


namespace calculate_highest_score_l780_780482

noncomputable def highest_score (avg_60 : ℕ) (delta_HL : ℕ) (avg_58 : ℕ) : ℕ :=
  let total_60 := 60 * avg_60
  let total_58 := 58 * avg_58
  let sum_HL := total_60 - total_58
  let L := (sum_HL - delta_HL) / 2
  let H := L + delta_HL
  H

theorem calculate_highest_score :
  highest_score 55 200 52 = 242 :=
by
  sorry

end calculate_highest_score_l780_780482


namespace subtraction_to_nearest_thousandth_l780_780477

theorem subtraction_to_nearest_thousandth : 
  (456.789 : ℝ) - (234.567 : ℝ) = 222.222 :=
by
  sorry

end subtraction_to_nearest_thousandth_l780_780477


namespace probability_of_picking_3_red_beans_l780_780031

noncomputable def jellybean_probability : ℚ :=
  let total JBs := 12
  let red JBs := 5
  let blue JBs := 3
  let white JBs := 4
  let picked JBs := 4
  let red picked := 3
  let non_red picked := picked JBs - red picked
  let total_combinations := Nat.choose total JBs picked JBs
  let red_combinations := Nat.choose red JBs red picked
  let non_red_combinations := Nat.choose (total JBs - red JBs) non_red picked
  let successful_combinations := red_combinations * non_red_combinations
  successful_combinations / total_combinations

theorem probability_of_picking_3_red_beans (h1 : total JBs = 12) (h2 : red JBs = 5) (h3 : blue JBs = 3) (h4 : white JBs = 4) (h5 : picked JBs = 4):
  jellybean_probability = 14/99 := by
  sorry

end probability_of_picking_3_red_beans_l780_780031


namespace largest_n_binom_l780_780548

theorem largest_n_binom (n : ℕ) (h : 0 ≤ n ∧ n ≤ 11) : 
  (binomial 10 4 + binomial 10 5 = binomial 11 n) → n = 6 :=
by
  sorry

end largest_n_binom_l780_780548


namespace smallest_palindrome_div_3_5_l780_780621

theorem smallest_palindrome_div_3_5 : ∃ n : ℕ, n = 50205 ∧ 
  (∃ a b c : ℕ, n = 5 * 10^4 + a * 10^3 + b * 10^2 + a * 10 + 5) ∧ 
  n % 5 = 0 ∧ 
  n % 3 = 0 ∧ 
  n ≥ 10000 ∧ 
  n < 100000 :=
by
  sorry

end smallest_palindrome_div_3_5_l780_780621


namespace collinear_vectors_k_eq_one_l780_780826

theorem collinear_vectors_k_eq_one (k λ : ℝ) :
  let a := (k^2 + k - 3, 1 - k)
  let b := (-3, k - 1)
  (∃ λ : ℝ, a = (λ * b.1, λ * b.2)) → k = 1 :=
sorry

end collinear_vectors_k_eq_one_l780_780826


namespace intersection_eq_T_l780_780164

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780164


namespace max_value_of_f_l780_780981

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos x) ^ 2 + (Real.sin x) * (Real.cos x) - 1

theorem max_value_of_f :
  ∃ x : ℝ, ∀ y : ℝ, f(y) ≤ f(x) ∧ f(x) = (Real.sqrt 5) / 2 :=
by
  sorry

end max_value_of_f_l780_780981


namespace largest_integer_binom_l780_780583

theorem largest_integer_binom (n : ℕ) (h : nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) : n = 6 :=
sorry

end largest_integer_binom_l780_780583


namespace number_of_correct_conclusions_in_triangle_l780_780829

theorem number_of_correct_conclusions_in_triangle (A B C r a b : ℝ) (h₁: 0 < A ∧ A < π) (h₂: 0 < B ∧ B < π) (h₃: A + B + C = π) (h₄: a = 2 * r * sin A) (h₅: b = 2 * r * sin B) :
  (A > B ↔ cos A < cos B) ∧
  (A > B ↔ sin A > sin B) ∧
  (A > B ↔ cos (2 * A) < cos (2 * B)) → 
  3 = 3 := 
sorry

end number_of_correct_conclusions_in_triangle_l780_780829


namespace largest_n_binomial_l780_780563

theorem largest_n_binomial :
  (∃n : ℕ, binom 10 4 + binom 10 5 = binom 11 n) → (n ≤ 6) :=
by
  sorry

end largest_n_binomial_l780_780563


namespace tangent_line_eqn_monotonicity_of_f_l780_780343

noncomputable def f (a x : ℝ) : ℝ := 1 / 3 * x^3 - 1 / 2 * a * x^2

theorem tangent_line_eqn (a x y : ℝ) (h : a = 2 ∧ x = 3 ∧ y = f 2 3) :
  3 * x - y - 9 = 0 := sorry

theorem monotonicity_of_f (a x : ℝ) :
  let f' := λ x, x * (x - a) in
  if h : a = 0 then 
    ∀ x1 x2, x1 ≤ x2 → f' x1 ≤ f' x2
  else if h : a < 0 then 
    ∀ x1 x2, (x1 ∈ Iio a ∨ x1 ∈ Ioi 0) ∧ (x2 ∈ Iio a ∨ x2 ∈ Ioi 0) ∧ x1 ≤ x2 → f' x1 ≤ f' x2 ∧ 
    ∀ x1 x2, (x1 ∈ Ioo a 0) ∧ (x2 ∈ Ioo a 0) ∧ x1 ≤ x2 → f' x1 > f' x2
  else 
    ∀ x1 x2, (x1 ∈ Iio 0 ∨ x1 ∈ Ioi a) ∧ (x2 ∈ Iio 0 ∨ x2 ∈ Ioi a) ∧ x1 ≤ x2 → f' x1 ≤ f' x2 ∧
    ∀ x1 x2, (x1 ∈ Ioo 0 a) ∧ (x2 ∈ Ioo 0 a) ∧ x1 ≤ x2 → f' x1 > f' x2 := sorry

end tangent_line_eqn_monotonicity_of_f_l780_780343


namespace total_cost_books_l780_780362

-- Defining the cost price and conditions
def C1 : ℝ := 274.1666666666667
def loss_percent : ℝ := 0.15
def gain_percent : ℝ := 0.19

-- Selling prices with given conditions
def SP1 : ℝ := C1 * (1 - loss_percent)
def SP2 (C2 : ℝ) : ℝ := C2 * (1 + gain_percent)

-- Given condition that both books were sold at the same price
def C2 : ℝ := SP1 / (1 + gain_percent)

-- The total cost price of the two books
def total_cost : ℝ := C1 + C2

-- Theorem statement
theorem total_cost_books : total_cost = 470 :=
by
  -- Define the value of SP1
  let SP1_value : ℝ := C1 * (1 - loss_percent)
  -- Substitute SP1_value into the definition of C2
  let C2_value : ℝ := SP1_value / (1 + gain_percent)
  -- Define the total cost based on C1 and computed C2_value
  let total_cost_value : ℝ := C1 + C2_value
  -- Assert the total cost equals 470
  show total_cost_value = 470
  sorry

end total_cost_books_l780_780362


namespace angle_ACB_90_degrees_l780_780756

theorem angle_ACB_90_degrees 
  (ABC_equilateral : ∀ (△: Triangle), ◇ direction_A(△ angle 30) (ABC_eq_side △) )
  (side_length : ∀ (△ : Triangle),  △ side_length = 1) :
  ∠ ACB = 90 :=
by
  sorry

end angle_ACB_90_degrees_l780_780756


namespace largest_n_binom_l780_780542

theorem largest_n_binom (n : ℕ) (h : 0 ≤ n ∧ n ≤ 11) : 
  (binomial 10 4 + binomial 10 5 = binomial 11 n) → n = 6 :=
by
  sorry

end largest_n_binom_l780_780542


namespace intersection_S_T_eq_T_l780_780266

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780266


namespace intersection_eq_T_l780_780292

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780292


namespace can_determine_mental_state_but_not_species_l780_780478

constant reliable : Type
def human : Type := reliable
def vampire : Type := reliable
inductive mental_state | sane | insane

constant is_sane : reliable → Prop
constant is_human : reliable → Prop
constant is_vampire : reliable → Prop

axiom human_sane_self_assessment (r : reliable) : is_human r → is_sane r = true
axiom human_insane_self_assessment (r : reliable) : is_human r → is_sane r = false
axiom vampire_sane_self_assessment (r : reliable) : is_vampire r → is_sane r = true
axiom vampire_insane_self_assessment (r : reliable) : is_vampire r → is_sane r = false

-- Lean 4 statement
theorem can_determine_mental_state_but_not_species (r : reliable) :
  (is_sane r → "yes" = "yes") →
  (¬ is_sane r → "no" = "no") →
  (∃ (s : String), (s = "yes" ∨ s = "no"))
  ∧ ¬ (is_human r ∨ is_vampire r) :=
by
  sorry

end can_determine_mental_state_but_not_species_l780_780478


namespace largest_n_binom_l780_780545

theorem largest_n_binom (n : ℕ) (h : 0 ≤ n ∧ n ≤ 11) : 
  (binomial 10 4 + binomial 10 5 = binomial 11 n) → n = 6 :=
by
  sorry

end largest_n_binom_l780_780545


namespace range_of_a_l780_780793

theorem range_of_a 
  (h : ¬ ∃ x : ℝ, ∀ a : ℝ, ax^2 + 2 * a * x + 1 ≤ 0) : 
  ∀ a : ℝ, a ∈ Icc 0 1 :=
by
  sorry

end range_of_a_l780_780793


namespace coefficient_x4_l780_780757

def given_polynomial : ℕ → ℤ
| 6 := 6
| 5 := -9
| 4 := -3
| 2 := 3
| _ := 0

theorem coefficient_x4 :
  given_polynomial 4 = -3 :=
by
  -- Proof would go here.
  sorry

end coefficient_x4_l780_780757


namespace triangle_ABC_angles_l780_780891

-- Definitions for the problem's conditions
structure Triangle :=
  (A B C : Type)
  (angle_A : ℝ := 50)
  (altitude_BH : ℝ)
  (BM_BH_equality : BM = altitude_BH)
  (perp_bisector_intersects_AC_at_K : AC = 2 * HK)

-- Lean 4 statement to prove the angles
theorem triangle_ABC_angles (A B C : Triangle) :
  A.angle_A = 50 ∧ ∠ B = 120 ∧ ∠ C = 10 :=
by
  sorry

end triangle_ABC_angles_l780_780891


namespace conjugate_of_complex_l780_780486

open Complex

theorem conjugate_of_complex : conj (5 / (3 + 4 * I) : ℂ) = (3 / 5) + (4 / 5) * I :=
by
  sorry

end conjugate_of_complex_l780_780486


namespace finite_correct_numbers_l780_780783

theorem finite_correct_numbers (k : ℕ) (h_k : k > 1) :
  {n : ℕ | n > 1 ∧ Nat.coprime n k ∧ ∀ d : ℕ, d < n → d ∣ n → ¬ Nat.coprime (d + k) n}.finite :=
by
  sorry

end finite_correct_numbers_l780_780783


namespace mikey_leaves_l780_780454

theorem mikey_leaves (n b : ℕ) (H₁ : n = 356) (H₂ : b = 244) : n - b = 112 := 
by 
  rw [H₁, H₂]
  exact Nat.sub_self 244

end mikey_leaves_l780_780454


namespace intersection_S_T_l780_780194

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780194


namespace S_inter_T_eq_T_l780_780299

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780299


namespace sqrt_expression_evaluation_l780_780712

theorem sqrt_expression_evaluation :
  (Real.sqrt 48 - 6 * Real.sqrt (1 / 3) - Real.sqrt 18 / Real.sqrt 6) = Real.sqrt 3 :=
by
  sorry

end sqrt_expression_evaluation_l780_780712


namespace ratio_of_areas_l780_780665

noncomputable def radius : ℝ := 4
noncomputable def area_circle : ℝ := π * radius^2
noncomputable def num_arcs : ℕ := 6

-- Assumptions based on equivalent mathematical proof:
-- The arcs are rearranged into a triangular figure.
-- The triangular figure has an area that approximates to 24.55 based on the problem.

noncomputable def area_triangle : ℝ := 24.55
noncomputable def ratio := area_triangle / area_circle

theorem ratio_of_areas : ratio = 1.5 / π :=
by
  sorry

end ratio_of_areas_l780_780665


namespace certain_number_is_17_l780_780843

theorem certain_number_is_17 (x : ℤ) (h : 2994 / x = 177) : x = 17 :=
by
  sorry

end certain_number_is_17_l780_780843


namespace intersection_S_T_l780_780190

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780190


namespace roots_product_l780_780329

theorem roots_product (x1 x2 : ℝ) (h : ∀ x : ℝ, x^2 - 4 * x + 1 = 0 → x = x1 ∨ x = x2) : x1 * x2 = 1 :=
sorry

end roots_product_l780_780329


namespace intersection_of_S_and_T_l780_780234

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780234


namespace intersection_S_T_l780_780185

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780185


namespace rectangle_enclosure_probability_sum_l780_780105

theorem rectangle_enclosure_probability_sum {c_1 c_2 d_1 d_2 : ℕ} :
  c_1 ∈ {1, 2, ..., 100} →
  c_2 ∈ {1, 2, ..., 100} →
  d_1 ∈ {1, 2, ..., 100} →
  d_2 ∈ {1, 2, ..., 100} →
  c_1 ≥ c_2 →
  d_1 ≥ d_2 →
  c_1 + c_2 > 150 →
  let q := (number_of_favorable_outcomes c_1 c_2 d_1 d_2) / (total_possible_choices c_1 c_2 d_1 d_2)
  in (numerator q + denominator q = 8) :=
by
  sorry

/-- Placeholder functions -/
noncomputable def number_of_favorable_outcomes (c_1 c_2 d_1 d_2 : ℕ) : ℕ := sorry
noncomputable def total_possible_choices (c_1 c_2 d_1 d_2 : ℕ) : ℕ := sorry
noncomputable def numerator (q : ℚ) : ℕ := sorry
noncomputable def denominator (q : ℚ) : ℕ := sorry


end rectangle_enclosure_probability_sum_l780_780105


namespace find_k_value_l780_780675

theorem find_k_value (k : ℝ) :
  (∀ m₁ m₂ : ℝ, m₁ = m₂ → (∃ k : ℝ, k = (1, -8) → (k, 15)) ↔ 6 * x + 9 * y = -12) →
  k = -33.5 :=
by
  sorry

end find_k_value_l780_780675


namespace ratio_A_B_l780_780745

theorem ratio_A_B :
  let A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
  let B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)
  0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 :=
by
  let A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
  let B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)
  have hA : A = 1400 := rfl
  have hB : B = 1500 := rfl
  have h_ratio : A / B = 14 / 15 := by
    rw [hA, hB]
    norm_num
  exact And.intro (by norm_num) (by norm_num)

end ratio_A_B_l780_780745


namespace equivalence_of_b_and_c_l780_780919

variables {α : Type*} [LinearOrder α] [Add α] [IsTotal α (· ≤ ·)]

noncomputable def b (a : ℕ → α) (n : ℕ) : α :=
if n = 0 then a 0 else
if n = 1 then max (a 0) (a 1) else
list.foldl (λ acc i, if i = 0 then a 0 else
                    if i = 1 then max (a 0) (a 1) else max acc (b a (i - 1) + a i)) (a 0) (list.range n)

noncomputable def c (a : ℕ → α) (n : ℕ) : α :=
if n = 0 then a 0 else
if n = 1 then max (a 0) (a 1) else
list.foldr (λ i acc, if i = n then a n else
                     if i = n - 1 then max (a n) (a (n - 1)) else max acc (c a (i + 1) + a i)) (a n) (list.range n)

theorem equivalence_of_b_and_c (a : ℕ → α) (n : ℕ) (h_n : 0 < n) : b a n = c a 1 := sorry

end equivalence_of_b_and_c_l780_780919


namespace intersection_eq_T_l780_780163

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780163


namespace compare_ab_1_to_a_b_two_pow_b_eq_one_div_b_sign_of_expression_l780_780091

-- Part 1
theorem compare_ab_1_to_a_b {a b : ℝ} (h1 : a^b * b^a + Real.log b / Real.log a = 0) (ha : a > 0) (hb : b > 0) : ab + 1 < a + b := sorry

-- Part 2
theorem two_pow_b_eq_one_div_b {b : ℝ} (h : 2^b * b^2 + Real.log b / Real.log 2 = 0) : 2^b = 1 / b := sorry

-- Part 3
theorem sign_of_expression {b : ℝ} (h : 2^b * b^2 + Real.log b / Real.log 2 = 0) : (2 * b + 1 - Real.sqrt 5) * (3 * b - 2) < 0 := sorry

end compare_ab_1_to_a_b_two_pow_b_eq_one_div_b_sign_of_expression_l780_780091


namespace reach_any_composite_from_4_l780_780412

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

def can_reach (A : ℕ) : Prop :=
  ∀ n : ℕ, is_composite n → ∃ seq : ℕ → ℕ, seq 0 = A ∧ seq (n + 1) - seq n ∣ seq n ∧ seq (n + 1) ≠ seq n ∧ seq (n + 1) ≠ 1 ∧ seq (n + 1) = n

theorem reach_any_composite_from_4 : can_reach 4 :=
  sorry

end reach_any_composite_from_4_l780_780412


namespace final_price_after_discounts_and_tax_l780_780493

theorem final_price_after_discounts_and_tax :
  let original_price := 240
  let first_discount := 30 / 100.0 * original_price
  let reduced_price_1 := original_price - first_discount
  let second_discount := 15 / 100.0 * reduced_price_1
  let reduced_price_2 := reduced_price_1 - second_discount
  let tax := 8 / 100.0 * reduced_price_2
  let final_price := reduced_price_2 + tax
  final_price = 154.22 :=
by
  have original_price := 240 : ℝ
  have first_discount := 30 / 100.0 * original_price
  have reduced_price_1 := original_price - first_discount
  have second_discount := 15 / 100.0 * reduced_price_1
  have reduced_price_2 := reduced_price_1 - second_discount
  have tax := 8 / 100.0 * reduced_price_2
  have final_price := reduced_price_2 + tax
  show final_price = 154.22
  sorry

end final_price_after_discounts_and_tax_l780_780493


namespace coeff_x4_is_200_l780_780403

open Nat

-- Define the polynomial (1 + x^3)(1 - x)^10
def poly := (1 + X^3) * (1 - X) ^ 10

-- State the goal: to prove that the coefficient of x^4 in "poly" is 200
theorem coeff_x4_is_200 : coeff poly 4 = 200 :=
by
  sorry

end coeff_x4_is_200_l780_780403


namespace intersection_S_T_eq_T_l780_780215

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780215


namespace intersection_S_T_eq_T_l780_780222

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780222


namespace largest_fraction_l780_780794

theorem largest_fraction (p q r s : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : q < r) (h4 : r < s) :
  (∃ (x : ℝ), x = (r + s) / (p + q) ∧ 
  (x > (p + s) / (q + r)) ∧ 
  (x > (p + q) / (r + s)) ∧ 
  (x > (q + r) / (p + s)) ∧ 
  (x > (q + s) / (p + r))) :=
sorry

end largest_fraction_l780_780794


namespace polynomial_division_remainder_l780_780620

noncomputable def dividend := 2 * X^4 + 9 * X^3 - 38 * X^2 - 50 * X + 35
noncomputable def divisor := X^2 + 5 * X - 6
noncomputable def remainder := 61 * X - 91

theorem polynomial_division_remainder :
  (dividend % divisor) = remainder :=
sorry

end polynomial_division_remainder_l780_780620


namespace moment_of_inertia_final_l780_780054

variable (R M : ℝ)

def I₀ : ℝ := (1 / 2) * M * R^2
def Iₕ : ℝ := (1 / 32) * M * R^2 + (1 / 16) * M * R^2
def I : ℝ := I₀ - Iₕ

theorem moment_of_inertia_final :
  I = (13 / 32) * M * R^2 := by
  sorry

end moment_of_inertia_final_l780_780054


namespace largest_integer_binom_l780_780584

theorem largest_integer_binom (n : ℕ) (h : nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) : n = 6 :=
sorry

end largest_integer_binom_l780_780584


namespace problem_I_problem_II_l780_780120

noncomputable theory

-- Definitions:
def S (a : ℕ → ℤ) (n : ℕ) : ℤ := ∑ i in finset.range n, a i

-- Problem I: Prove {a_n} is a geometric sequence
theorem problem_I (a : ℕ → ℤ) (h : ∀ n : ℕ, a (n+1) = 2 * S a (n+1) - 1) :
  ∃ r : ℤ, ∃ a₀ : ℤ, ∀ n : ℕ, a n = a₀ * r ^ n :=
sorry

-- Problem II: Find T_n
theorem problem_II (a : ℕ → ℤ) (h : ∀ n : ℕ, a (n+1) = 2 * S a (n+1) - 1)
  (b : ℕ → ℤ) (h_b : ∀ n : ℕ, b n = (2 * n + 1) * a n) :
  ∀ n : ℕ, ( ∑ i in finset.range n, b i ) = 1 - (n+1) * (-1)^n :=
sorry

end problem_I_problem_II_l780_780120


namespace circumscribed_circle_radius_l780_780688

-- Definitions: Conditions from part a)
def radius : ℝ := 8
def theta : ℝ := sorry -- theta is a given obtuse angle

-- Stating the main problem to prove
theorem circumscribed_circle_radius (θ : ℝ) (hθ_obtuse : θ > π / 2 ∧ θ < π) : 
  let R := 8 * (Real.sec (θ / 2)) in R = 8 / (Real.cos (θ / 2)) :=
sorry

end circumscribed_circle_radius_l780_780688


namespace algebra_expression_value_at_0_l780_780472

theorem algebra_expression_value_at_0 :
  (let x := 0 in (2 * x - 1) ^ 2 - (3 * x + 1) * (3 * x - 1) + 5 * x * (x - 1)) = 2 :=
by
  sorry

end algebra_expression_value_at_0_l780_780472


namespace quadratic_function_properties_l780_780333

theorem quadratic_function_properties :
  (∃ a b, (∀ x, f(x, a, b) = x^2 + a * x + b) ∧ 
            homogeneous_symmetric f 1 ∧ 
            f 0 = 0) →
  ((∃ a b, (f x = x^2 - 2*x given_symmetric tq((−a/2 = 1), tq(a = -2)) tqf origin_passing tq((f 0 = 0)) tqa(b = 0]))) ∧ 
   (range f (0, 3] = [-1, 3]))
  sorry -- Proof is not required.

end quadratic_function_properties_l780_780333


namespace probability_of_divisibility_by_3_l780_780849

def is_prime_digit (d : ℕ) : Prop := 
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_valid_two_digit_number (n : ℕ) : Prop := 
  10 ≤ n ∧ n ≤ 99 ∧ is_prime_digit (n / 10) ∧ is_prime_digit (n % 10)

def sum_of_digits (n : ℕ) : ℕ := 
  (n / 10) + (n % 10)

def is_divisible_by_3 (n : ℕ) : Prop := 
  (sum_of_digits n) % 3 = 0

theorem probability_of_divisibility_by_3 :
  let prime_two_digit_numbers := {n : ℕ | is_valid_two_digit_number n},
      favorable_numbers := {n ∈ prime_two_digit_numbers | is_divisible_by_3 n},
      total_two_digit_numbers := (prime_two_digit_numbers.to_finset).card,
      favorable_count := (favorable_numbers.to_finset).card
  in (favorable_count : ℕ) / (total_two_digit_numbers : ℕ) = 5 / 16 :=
begin
  sorry
end

end probability_of_divisibility_by_3_l780_780849


namespace intersection_S_T_eq_T_l780_780223

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780223


namespace fraction_of_cost_due_to_high_octane_is_half_l780_780662

-- Define the units of high octane and regular octane
def highOctaneUnits := 15
def regularOctaneUnits := 45

-- Define the cost relation between high octane and regular octane
def costRatio := 3

-- Assume the cost per unit of regular octane as 1
def costRegularUnit := 1

-- Calculate the cost per unit of high octane
def costHighUnit := costRatio * costRegularUnit

-- Calculate the total cost of high octane and regular octane
def totalCostHigh := highOctaneUnits * costHighUnit
def totalCostRegular := regularOctaneUnits * costRegularUnit

-- Calculate the total cost of the fuel
def totalCostFuel := totalCostHigh + totalCostRegular

-- The fraction of the cost due to high octane
def fractionCostHigh := totalCostHigh / totalCostFuel

theorem fraction_of_cost_due_to_high_octane_is_half :
  fractionCostHigh = 0.5 :=
by
  sorry

end fraction_of_cost_due_to_high_octane_is_half_l780_780662


namespace intersection_S_T_l780_780311

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780311


namespace original_price_of_item_l780_780045

theorem original_price_of_item (P : ℝ) 
(selling_price : ℝ) 
(h1 : 0.9 * P = selling_price) 
(h2 : selling_price = 675) : 
P = 750 := sorry

end original_price_of_item_l780_780045


namespace platform_length_l780_780019

theorem platform_length
  (train_length : ℕ)
  (pole_time : ℕ)
  (platform_time : ℕ)
  (h1 : train_length = 300)
  (h2 : pole_time = 18)
  (h3 : platform_time = 39) :
  let speed := train_length / pole_time in
  let total_distance := speed * platform_time in
  total_distance - train_length = 350 :=
by 
  sorry

end platform_length_l780_780019


namespace fraction_numerator_l780_780487

theorem fraction_numerator (x : ℚ) : 
  (∃ y : ℚ, y = 4 * x + 4 ∧ x / y = 3 / 7) → x = -12 / 5 :=
by
  sorry

end fraction_numerator_l780_780487


namespace estimate_ratio_l780_780735

theorem estimate_ratio (A B : ℕ) (A_def : A = 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28)
  (B_def : B = 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20) : 0 < A / B ∧ A / B < 1 := by
  sorry

end estimate_ratio_l780_780735


namespace Q_A_correct_Q_B_incorrect_Q_C_correct_Q_D_incorrect_main_proof_l780_780876

-- Definitions for tiered water pricing
def price (x : ℕ) : ℕ :=
  if x ≤ 12 then
    3 * x
  else if x ≤ 18 then
    3 * 12 + 6 * (x - 12)
  else
    3 * 12 + 6 * 6 + 9 * (x - 18)

-- Proof Statements for each question
theorem Q_A_correct (h₁ : price 10 = 30) : true := by
  sorry

theorem Q_B_incorrect (h₂ : price 16 ≠ 96) : true := by
  sorry

theorem Q_C_correct (h₃ : ∃ n : ℕ, price n = 54 ∧ n = 15) : true := by
  sorry

theorem Q_D_incorrect (h₄ : ∀ (A B : ℕ), (price A + price B = 93) → (A ≤ 12) → (B ≤ 18) → A ≠ 9) : true := by
  sorry

-- main theorem combining all individual proofs
theorem main_proof : true := by
  have h₁ : price 10 = 30 := rfl
  have h₂ : price 16 ≠ 96 := sorry -- Calculation to be inserted
  have h₃ : ∃ n : ℕ, price n = 54 ∧ n = 15 := sorry -- Calculation to be inserted
  have h₄ : ∀ (A B : ℕ), (price A + price B = 93) → (A ≤ 12) → (B ≤ 18) → A ≠ 9 := sorry -- Calculation to be inserted
  exact ⟨Q_A_correct h₁, Q_B_incorrect h₂, Q_C_correct h₃, Q_D_incorrect h₄⟩

end Q_A_correct_Q_B_incorrect_Q_C_correct_Q_D_incorrect_main_proof_l780_780876


namespace largest_n_binom_identity_l780_780549

open Nat

theorem largest_n_binom_identity :
  (∃ n, binom 11 n = binom 10 4 + binom 10 5) ∧
  (∀ m, binom 11 m = binom 10 4 + binom 10 5 → m ≤ 6) :=
by
  sorry

end largest_n_binom_identity_l780_780549


namespace intersection_of_sets_l780_780172

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780172


namespace intersection_S_T_l780_780153

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780153


namespace solve_a_l780_780792

-- Defining sets A and B
def set_A (a : ℤ) : Set ℤ := {a^2, a + 1, -3}
def set_B (a : ℤ) : Set ℤ := {a - 3, 2 * a - 1, a^2 + 1}

-- Defining the condition of intersection
def intersection_condition (a : ℤ) : Prop :=
  (set_A a) ∩ (set_B a) = {-3}

-- Stating the theorem
theorem solve_a (a : ℤ) (h : intersection_condition a) : a = -1 :=
sorry

end solve_a_l780_780792


namespace factor_210_into_4_factors_l780_780392

theorem factor_210_into_4_factors : ∃ n : ℕ, n = 15 ∧
  ∃ (a b c d : ℕ), a * b * c * d = 210 ∧ (mul_comm a b ∧ mul_comm a c ∧ mul_comm a d ∧ mul_comm b c ∧ mul_comm b d ∧ mul_comm c d) :=
by 
  sorry

end factor_210_into_4_factors_l780_780392


namespace half_height_of_triangular_prism_l780_780481

theorem half_height_of_triangular_prism (volume base_area height : ℝ) 
  (h_volume : volume = 576)
  (h_base_area : base_area = 3)
  (h_prism : volume = base_area * height) :
  height / 2 = 96 :=
by
  have h : height = volume / base_area := by sorry
  rw [h_volume, h_base_area] at h
  have h_height : height = 192 := by sorry
  rw [h_height]
  norm_num

end half_height_of_triangular_prism_l780_780481


namespace largest_n_for_binomial_equality_l780_780534

theorem largest_n_for_binomial_equality :
  ∃ n : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 n ) ∧
  ∀ m : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 m ) → m ≤ 6 :=
begin
  sorry
end

end largest_n_for_binomial_equality_l780_780534


namespace log_sum_value_l780_780405

-- Define a geometric sequence.
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 0 < q ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Conditions provided in the problem.
variables (a : ℕ → ℝ) (h_geom : is_geometric_sequence a)
variable (h_pos : ∀ n : ℕ, 0 < a n)
variable (h_geom_mean : (a 5) * (a 15) = (2 * Real.sqrt 2) ^ 2)

-- The desired proof statement.
theorem log_sum_value : Real.log 2 (a 4) + Real.log 2 (a 16) = 3 := sorry

end log_sum_value_l780_780405


namespace largest_integer_comb_l780_780602

theorem largest_integer_comb (n: ℕ) (h1 : n = 6) : 
  (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) :=
by
  rw [h1]
  exact sorry

end largest_integer_comb_l780_780602


namespace range_of_func_l780_780817

def func (x : Real) := (3 * Real.sin x + 1) / (Real.sin x + 2)

theorem range_of_func :
  ∀ y, (∃ x, y = func x ∧ -1 ≤ Real.sin x ∧ Real.sin x ≤ 1) ↔ (-2 ≤ y ∧ y ≤ 4/3) :=
by
  sorry

end range_of_func_l780_780817


namespace incenter_lies_on_equal_dividing_line_l780_780042

theorem incenter_lies_on_equal_dividing_line
  (A B C K M : Point)
  (line : Line)
  (triangle : Triangle A B C)
  (hline1 : line ∈ (segment B C))
  (hline2 : line.intersects (segment A K))
  (hline3 : line.intersects (segment A M))
  (hline_divides_triangle_into_equal_areas : area (triangle.segment_division line) = (1 / 2) * area triangle)
  (hline_divides_triangle_into_equal_perimeters : perimeter (triangle.segment_division line) = (1 / 2) * perimeter triangle):
  incircle_center triangle ∈ line := sorry

end incenter_lies_on_equal_dividing_line_l780_780042


namespace largest_integer_binom_l780_780610

theorem largest_integer_binom
  (n : ℕ)
  (h₁ : binom 10 4 + binom 10 5 = binom 11 n) :
  n = 6 :=
sorry

end largest_integer_binom_l780_780610


namespace intersection_of_sets_l780_780178

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780178


namespace complex_numbers_count_l780_780101

theorem complex_numbers_count (z: ℂ) (hz: |z| = 1) (hz_condition : ∀ z, (z^720 - z^120 : ℂ) ∈ ℝ) : 
  ∃(n: ℕ), n = 1440 :=
sorry

end complex_numbers_count_l780_780101


namespace equilateral_triangle_area_l780_780479

theorem equilateral_triangle_area (AM : ℝ) (AM_sqrt6 : AM = Real.sqrt 6) :
  let BC := 2 * Real.sqrt 2 in
  let area := (1 / 2) * BC * AM in
  area = 2 * Real.sqrt 3 :=
by
  intro AM AM_sqrt6
  let BM := Real.sqrt 2
  let BC := 2 * BM
  let area := (1 / 2) * BC * AM
  have h1 : Real.sqrt 6 = AM := by rw [AM_sqrt6]
  have h2 : BM = Real.sqrt 6 / Real.sqrt 3 := by rw [BM, Real.sqrt_div]
  rw [h2]
  have h3 : BC = 2 * BM := rfl
  rw [h3]
  have h4 : area = (1 / 2) * (2 * BM) * Real.sqrt 6 := rfl
  sorry

end equilateral_triangle_area_l780_780479


namespace musicians_fee_l780_780513

theorem musicians_fee (r_janek r_mikeš r_vávra : ℕ) (shared_fee_per_part : ℕ)
  (janek_extra_share : ℕ) (mikeš_extra_share : ℕ)
  (difference_parts : ℕ) (difference_czk : ℕ) :
  ((r_janek = 4) ∧ (r_mikeš = 5) ∧ (r_vávra = 6) ∧ (shared_fee_per_part = 40)
   ∧ (janek_extra_share = 9) ∧ (mikeš_extra_share = 10)
   ∧ (difference_parts = mikeš_extra_share - janek_extra_share)
   ∧ (difference_czk = 40)) →
  45 * shared_fee_per_part = 1800 :=
by
  sorry

end musicians_fee_l780_780513


namespace largest_n_binom_equality_l780_780522

theorem largest_n_binom_equality :
  ∃ n : ℕ, (binomial 10 4 + binomial 10 5 = binomial 11 n) ∧ ∀ m : ℕ, (binomial 11 m = binomial 11 5) → m ≤ 6 :=
by
  sorry

end largest_n_binom_equality_l780_780522


namespace intersection_eq_T_l780_780291

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780291


namespace sixty_percent_of_total_is_960_l780_780972

-- Definitions from the conditions
def boys : ℕ := 600
def difference : ℕ := 400
def girls : ℕ := boys + difference
def total : ℕ := boys + girls
def sixty_percent_of_total : ℕ := total * 60 / 100

-- The theorem to prove
theorem sixty_percent_of_total_is_960 :
  sixty_percent_of_total = 960 := 
  sorry

end sixty_percent_of_total_is_960_l780_780972


namespace min_distance_l780_780399

noncomputable def C1 (t : ℝ) : ℝ × ℝ := (-4 + Real.cos t, 3 + Real.sin t)
noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (8 * Real.cos θ, 3 * Real.sin θ)
noncomputable def P := C1 (Real.pi / 2)
noncomputable def Q (θ : ℝ) := C2 θ
noncomputable def M (θ : ℝ) := ((-2 + 4 * Real.cos θ), (2 + (3 / 2) * Real.sin θ))
noncomputable def C3 := { p : ℝ × ℝ | p.1 - 2 * p.2 = 7 }

noncomputable def distance_to_line (M : ℝ × ℝ) : ℝ :=
  Real.abs ((4 * Real.cos (theta) - 3 * Real.sin (theta) - 13) * Real.sqrt(5) / 5)

theorem min_distance : ∃ θ : ℝ, distance_to_line (M θ) = 8 * Real.sqrt 5 / 5 := sorry

end min_distance_l780_780399


namespace probability_of_being_closer_to_origin_l780_780048

noncomputable def probability_closer_to_origin 
  (rect : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2})
  (origin : ℝ × ℝ := (0, 0))
  (point : ℝ × ℝ := (4, 2))
  : ℚ :=
1/3

theorem probability_of_being_closer_to_origin :
  probability_closer_to_origin = 1/3 :=
by sorry

end probability_of_being_closer_to_origin_l780_780048


namespace cos_double_angle_l780_780797

variable (θ : ℝ)
variable (hθ1 : Real.tan θ = -1/3)
variable (hθ2 : θ ∈ Ioo (π / 2) π)

theorem cos_double_angle (hθ1 : Real.tan θ = -1/3) (hθ2 : θ ∈ Ioo (π / 2) π) :
  Real.cos (2 * θ) = 4 / 5 := sorry

end cos_double_angle_l780_780797


namespace A_beats_B_by_40_meters_l780_780387

-- Definitions based on conditions
def distance_A := 1000 -- Distance in meters
def time_A := 240      -- Time in seconds
def time_diff := 10      -- Time difference in seconds

-- Intermediate calculations
def velocity_A : ℚ := distance_A / time_A
def time_B := time_A + time_diff
def velocity_B : ℚ := distance_A / time_B

-- Distance B covers in 240 seconds
def distance_B_in_240 : ℚ := velocity_B * time_A

-- Proof goal
theorem A_beats_B_by_40_meters : (distance_A - distance_B_in_240 = 40) :=
by
  -- Insert actual steps to prove here
  sorry

end A_beats_B_by_40_meters_l780_780387


namespace sum_of_first_2009_terms_l780_780401

variable (a : ℕ → ℝ) (d : ℝ)

-- conditions: arithmetic sequence and specific sum condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def sum_condition (a : ℕ → ℝ) : Prop :=
  a 1004 + a 1005 + a 1006 = 3

-- sum of the first 2009 terms
noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n.succ * (a 0 + a n.succ) / 2)

-- proof problem
theorem sum_of_first_2009_terms (h1 : is_arithmetic_sequence a d) (h2 : sum_condition a) :
  sum_first_n_terms a 2008 = 2009 :=
sorry

end sum_of_first_2009_terms_l780_780401


namespace simplify_expression_l780_780469

variables {K : Type*} [Field K]

theorem simplify_expression (a b c : K) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) : 
    (a^3 - b^3) / (a * b) - (a * b - c * b) / (a * b - a^2) = (a^2 + a * b + b^2 + a * c) / (a * b) :=
by
  sorry

end simplify_expression_l780_780469


namespace largest_n_for_binom_equality_l780_780596

theorem largest_n_for_binom_equality : ∃ (n : ℕ), (∑ i : {i // i = 4 ∨ i = 5}, Nat.choose 10 i) = Nat.choose 11 n ∧ n = 6 :=
by
  sorry

end largest_n_for_binom_equality_l780_780596


namespace simplify_permutation_l780_780958

-- Define the permutation function A
def perm (n m : ℕ) : ℕ := nat.factorial n / nat.factorial (n - m)

-- Lean theorem statement proving the simplification
theorem simplify_permutation (n m : ℕ) (hnm : n ≥ m) :
  (perm n m) / (perm (n - 1) (m - 1)) = n :=
by
  -- Skip the proof for now
  sorry

end simplify_permutation_l780_780958


namespace tan_addition_l780_780323

variable (α β : ℝ)

theorem tan_addition (h : (1 + sqrt 3 * tan α) * (1 + sqrt 3 * tan β) = 4) : 
  tan (α + β) = sqrt 3 :=
sorry

end tan_addition_l780_780323


namespace family_can_purchase_furniture_in_april_l780_780982

noncomputable def monthly_income : ℤ := 150000
noncomputable def monthly_expenses : ℤ := 115000
noncomputable def initial_savings : ℤ := 45000
noncomputable def furniture_cost : ℤ := 127000

theorem family_can_purchase_furniture_in_april : 
  ∃ (months : ℕ), months = 3 ∧ 
  (initial_savings + months * (monthly_income - monthly_expenses) >= furniture_cost) :=
by
  -- proof will be written here
  sorry

end family_can_purchase_furniture_in_april_l780_780982


namespace vector_triangle_rule_l780_780702

-- Definitions of points A, B, C
variables (A B C : Type) [AddGroup A]

-- Definitions of vectors corresponding to points
def vector_from (P Q : A) : A := Q - P

-- The theorem statement based on the conditions and correct answer
theorem vector_triangle_rule (A B C : A) : 
  vector_from A B = vector_from A C + vector_from C B :=
by
  sorry

end vector_triangle_rule_l780_780702


namespace angle_CFD_right_angle_l780_780949

theorem angle_CFD_right_angle {A B C D E F : Type*} 
  [AddGroup A] [VAdd A B] [VSub A B] [Proportionality A B] [LinearPairing A B C] 
  (ABCD : IsParallelogram A B C D) 
  (E_midpoint : IsMidpoint E A B) 
  (F_on_DE : OnSegment F D E) 
  (AD_eq_BF : Distance A D = Distance F B) 
  : MeasureAngle F C D = 90 :=
sorry

end angle_CFD_right_angle_l780_780949


namespace area_of_region_bounded_by_circle_l780_780083

theorem area_of_region_bounded_by_circle :
  (∃ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 9 = 0) →
  ∃ (area : ℝ), area = 4 * Real.pi :=
by
  sorry

end area_of_region_bounded_by_circle_l780_780083


namespace hyperbola_lattice_points_count_l780_780680

def is_lattice_point (x y : ℤ) : Prop :=
  x^2 - 2 * y^2 = 2000^2

def count_lattice_points (points : List (ℤ × ℤ)) : ℕ :=
  points.length

theorem hyperbola_lattice_points_count : count_lattice_points [(2000, 0), (-2000, 0)] = 2 :=
by
  sorry

end hyperbola_lattice_points_count_l780_780680


namespace inequality_of_reals_l780_780767

theorem inequality_of_reals (a b c d : ℝ) : 
  (a + b + c + d) / ((1 + a^2) * (1 + b^2) * (1 + c^2) * (1 + d^2)) < 1 := 
  sorry

end inequality_of_reals_l780_780767


namespace inequality_solution_l780_780997

open Set

theorem inequality_solution :
  {x : ℝ | |x + 1| - 2 > 0} = {x : ℝ | x < -3} ∪ {x : ℝ | x > 1} :=
by
  sorry

end inequality_solution_l780_780997


namespace no_perfect_square_in_range_l780_780357

def f (n : ℕ) : ℕ := 2 * n^2 + 3 * n + 2

theorem no_perfect_square_in_range : ∀ (n : ℕ), 5 ≤ n → n ≤ 15 → ¬ ∃ (m : ℕ), f n = m^2 := by
  intros n h1 h2
  sorry

end no_perfect_square_in_range_l780_780357


namespace odd_function_f_neg_x_l780_780492

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then x^2 - 2*x else
if h : x < 0 then - (x^2 - 2*(-x)) else 0

theorem odd_function_f_neg_x {x : ℝ} (hx : x < 0) : 
  f(x) = -x^2 - 2*x := by
  sorry

end odd_function_f_neg_x_l780_780492


namespace tangent_lines_to_both_circles_l780_780760

theorem tangent_lines_to_both_circles :
  let C₁ := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 + 2 * p.1 - 6 * p.2 - 26 = 0 },
      C₂ := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 4 * p.1 + 2 * p.2 + 4 = 0 } in
  ∃ l : ℝ → ℝ, (∀ p : ℝ × ℝ, (p ∈ C₁ → p.2 = l p.1) ∧ (p ∈ C₂ → p.2 = l p.1)) ∧
                (∀ l' : ℝ → ℝ, (∀ p : ℝ × ℝ, (p ∈ C₁ → p.2 = l' p.1) ∧ (p ∈ C₂ → p.2 = l' p.1)) → l' = l) :=
by
  sorry

end tangent_lines_to_both_circles_l780_780760


namespace shelly_catches_3_swordfish_l780_780471

def fishing_problem (S : ℕ) (S_catches : ℕ) (Sam_catches : ℕ) : Prop :=
  (S_catches = S) ∧ (Sam_catches = S - 1) ∧ (5 * S + 5 * (S - 1) = 25) → (S = 3)

theorem shelly_catches_3_swordfish :
  ∃ (S : ℕ) (S_catches : ℕ) (Sam_catches : ℕ), fishing_problem S S_catches Sam_catches :=
begin
  use 3,
  use 3,
  use 2,
  unfold fishing_problem,
  simp,
  split,
  { refl, },
  split,
  { refl, },
  intros _,
  sorry,  -- The actual proof would be placed here.
end

end shelly_catches_3_swordfish_l780_780471


namespace cos_B_area_of_triangle_l780_780433

variables {A B C : ℝ} {a b c : ℝ}
variables {m n : ℝ × ℝ}

def triangle_conditions (a b c : ℝ) (A B C : ℝ) :=
  let m := (Real.sin B, -2 * Real.sin A) in
  let n := (Real.sin B, Real.sin C) in
  -- Dot product is zero for orthogonality condition
  m.1 * n.1 + m.2 * n.2 = 0

theorem cos_B (a b c : ℝ) (A B C : ℝ)
  (hab : a = b)
  (orthog_cond : triangle_conditions a b c A B C) :
  Real.cos B = 1 / 4 :=
sorry

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ)
  (hb : B = π / 2)
  (ha : a = Real.sqrt 2)
  (orthog_cond : triangle_conditions a b c A B C) :
  let area := 1 / 2 * b * c in
  area = 1 :=
sorry

end cos_B_area_of_triangle_l780_780433


namespace number_of_boys_l780_780875

def initial_girls : ℕ := 706
def new_girls : ℕ := 418
def total_pupils : ℕ := 1346
def total_girls := initial_girls + new_girls

theorem number_of_boys : 
  total_pupils = total_girls + 222 := 
by
  sorry

end number_of_boys_l780_780875


namespace min_sum_distances_l780_780976

structure Point : Type :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨-1, 2⟩
def B : Point := ⟨3, -4⟩
def C : Point := ⟨5, -6⟩
def D : Point := ⟨-2, 8⟩

noncomputable def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

noncomputable def PA_plus_PB_plus_PC_plus_PD (P : Point) : ℝ :=
  distance P A + distance P B + distance P C + distance P D

theorem min_sum_distances :
  ∃ P : Point, PA_plus_PB_plus_PC_plus_PD P = 23 :=
sorry

end min_sum_distances_l780_780976


namespace largest_n_binom_identity_l780_780551

open Nat

theorem largest_n_binom_identity :
  (∃ n, binom 11 n = binom 10 4 + binom 10 5) ∧
  (∀ m, binom 11 m = binom 10 4 + binom 10 5 → m ≤ 6) :=
by
  sorry

end largest_n_binom_identity_l780_780551


namespace find_value_of_expression_l780_780773

theorem find_value_of_expression (a : ℝ) (h : a^2 - a - 1 = 0) : a^3 - a^2 - a + 2023 = 2023 :=
by
  sorry

end find_value_of_expression_l780_780773


namespace trigonometric_identity_l780_780102

-- Given conditions
def angle10 := 10 * Real.pi / 180
def angle20 := 20 * Real.pi / 180
def angle80 := 80 * Real.pi / 180
def cos_angle10 := Real.cos angle10
def cos_angle20 := Real.cos angle20
def cos_angle80 := Real.cos angle80
def sin_angle20 := Real.sin angle20

-- Problem statement
theorem trigonometric_identity :
  cos_angle10 * cos_angle20 - cos_angle80 * sin_angle20 = Real.sqrt 3 / 2 := 
sorry

end trigonometric_identity_l780_780102


namespace inequality_proof_l780_780920

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (real.sqrt ((z + x) * (z + y)) - z) ≥ real.sqrt (x * y) :=
sorry

end inequality_proof_l780_780920


namespace yield_percentage_is_10_l780_780018

-- Define the annual dividend rate
def annual_dividend_rate : ℝ := 0.12

-- Define the face value of the stock
def face_value : ℝ := 100

-- Define the market price of the stock
def market_price : ℝ := 120

-- Calculate the annual dividend based on the face value and annual dividend rate
def annual_dividend : ℝ := annual_dividend_rate * face_value

-- Calculate the yield percentage
def yield_percentage : ℝ := (annual_dividend / market_price) * 100

-- Statement to prove the yield percentage is 10
theorem yield_percentage_is_10 : yield_percentage = 10 := by
  sorry

end yield_percentage_is_10_l780_780018


namespace radius_of_circumcircle_l780_780948

-- Define the given conditions
variables (A L M K F O : Type)
variables (LM_side_of_triangle_KLM : A ∈ [L, M])
variables (angle_K_60 : ∠ K = 60)
variables (radius_of_AF : dist A F = 4)
variables (radius_of_AO : dist A O = 7)

-- Define the required result to prove
def radius_of_circumcircle_FKO : ℝ :=
  sqrt 65

-- Statement of the problem to be proved
theorem radius_of_circumcircle :
  LM_side_of_triangle_KLM → 
  angle_K_60 → 
  radius_of_AF → 
  radius_of_AO → 
  dist (circumcenter F K O) (circle F K O) = radius_of_circumcircle_FKO.
Proof
  sorry -- actual proof steps are not required

end radius_of_circumcircle_l780_780948


namespace overall_loss_percentage_l780_780462

-- Definitions for initial conditions
def cost_price_A : ℝ := 1
def num_A : ℕ := 30
def total_cost_A : ℝ := cost_price_A * num_A

def cost_price_B : ℝ := 2
def num_B : ℕ := 40
def total_cost_B : ℝ := cost_price_B * num_B

def cost_price_C : ℝ := 3
def num_C : ℕ := 10
def total_cost_C : ℝ := cost_price_C * num_C

def total_cost : ℝ := total_cost_A + total_cost_B + total_cost_C

def selling_price_A : ℝ := cost_price_A - 0.5
def total_selling_A : ℝ := selling_price_A * num_A

def selling_price_B : ℝ := cost_price_B - 1
def total_selling_B : ℝ := selling_price_B * num_B

def selling_price_C : ℝ := cost_price_C - 1.5
def total_selling_C : ℝ := selling_price_C * num_C

def total_selling : ℝ := total_selling_A + total_selling_B + total_selling_C

def loss_due_to_A : ℝ := total_selling_A

def total_loss : ℝ := total_cost - total_selling + loss_due_to_A

def loss_percentage : ℝ := (total_loss / total_cost) * 100

-- The theorem to prove
theorem overall_loss_percentage : 
  loss_percentage = 60.71 := 
sorry

end overall_loss_percentage_l780_780462


namespace income_percentage_l780_780943

theorem income_percentage (J T M : ℝ) 
  (h1 : T = 0.5 * J) 
  (h2 : M = 1.6 * T) : 
  M = 0.8 * J :=
by 
  sorry

end income_percentage_l780_780943


namespace intersection_S_T_eq_T_l780_780201

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780201


namespace largest_integer_binom_l780_780611

theorem largest_integer_binom
  (n : ℕ)
  (h₁ : binom 10 4 + binom 10 5 = binom 11 n) :
  n = 6 :=
sorry

end largest_integer_binom_l780_780611


namespace probability_evenly_spaced_bins_l780_780512

open ProbabilityTheory

def dice_outcomes := {1, 4, 9, 16, 25, 36}

def evenly_spaced_triplet (a b c : ℕ) : Prop := 
  abs (b - a) % 4 = 0 ∧ abs (c - b) % 4 = 0

def valid_triplets : set (ℕ × ℕ × ℕ) := 
  {(1, 9, 25), (4, 16, 36)}

def probability (s : set (ℕ × ℕ × ℕ)) (total_outcomes : ℕ) : ℚ := 
  (s.card : ℚ) / total_outcomes

theorem probability_evenly_spaced_bins : 
  probability valid_triplets (dice_outcomes.card ^ 3) = 1 / 108 :=
sorry

end probability_evenly_spaced_bins_l780_780512


namespace intersection_S_T_l780_780188

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780188


namespace vector_dot_product_problem_l780_780803

variables {a b : ℝ}

theorem vector_dot_product_problem (h1 : a + 2 * b = 0) (h2 : (a + b) * a = 2) : a * b = -2 :=
sorry

end vector_dot_product_problem_l780_780803


namespace C_total_days_l780_780661

-- Conditions
def A_days : ℕ := 30
def B_days : ℕ := 30
def A_worked_days : ℕ := 10
def B_worked_days : ℕ := 10
def C_finished_days : ℕ := 10

-- Work rates
def A_work_rate : ℚ := 1 / A_days
def B_work_rate : ℚ := 1 / B_days

-- Work done
def A_work_done : ℚ := A_work_rate * A_worked_days
def B_work_done : ℚ := B_work_rate * B_worked_days
def total_work_done : ℚ := A_work_done + B_work_done
def work_left_for_C : ℚ := 1 - total_work_done
def C_work_rate : ℚ := work_left_for_C / C_finished_days

-- Equivalent proof problem
theorem C_total_days (d : ℕ) (h : C_work_rate = (1 : ℚ) / d) : d = 30 :=
by sorry

end C_total_days_l780_780661


namespace cookies_left_l780_780107

theorem cookies_left (day1_trays day2_trays day3_trays trays_per_day cookie_per_tray frank_eats_day4 ted_eats_day4 jan_eats_day5 ted_eats_day6 total_bakes total_eaten remaining_cookies : ℕ)
  (H1 : day1_trays = 2)
  (H2 : day2_trays = 3)
  (H3 : day3_trays = 4)
  (H4 : cookie_per_tray = 12)
  (H5 : frank_eats_day4 = 1)
  (H6 : ted_eats_day4 = 4)
  (H7 : jan_eats_day5 = 3)
  (H8 : ted_eats_day6 = 5)
  (H9 : total_bakes = 24 + 36 + 48 + 24 + 36 + 48)
  (H10 : total_eaten = 6 * frank_eats_day4 + ted_eats_day4 + jan_eats_day5 + ted_eats_day6)
  (H11 : remaining_cookies = total_bakes - total_eaten)
  : remaining_cookies = 198 :=
begin
  sorry
end

end cookies_left_l780_780107


namespace angle_terminal_side_l780_780962

theorem angle_terminal_side (k : ℤ) : ∃ α : ℝ, α = k * 360 - 30 ∧ 0 ≤ α ∧ α < 360 →
  α = 330 :=
by
  sorry

end angle_terminal_side_l780_780962


namespace intersection_S_T_l780_780321

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780321


namespace problem1_problem2_l780_780816

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 2)

theorem problem1 (m : ℝ) (h₀ : m > 3) (h₁ : ∃ m, (1/2) * (((m - 1) / 2) - (-(m + 1) / 2) + 3) * (m - 3) = 7 / 2) : m = 4 := by
  sorry

theorem problem2 (a : ℝ) (h₂ : ∃ x, (0 ≤ x ∧ x ≤ 2) ∧ f x ≥ abs (a - 3)) : -2 ≤ a ∧ a ≤ 8 := by
  sorry

end problem1_problem2_l780_780816


namespace intersection_S_T_l780_780151

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780151


namespace amber_age_l780_780059

theorem amber_age 
  (a g : ℕ)
  (h1 : g = 15 * a)
  (h2 : g - a = 70) :
  a = 5 :=
by
  sorry

end amber_age_l780_780059


namespace probability_one_blue_one_white_one_red_l780_780657

theorem probability_one_blue_one_white_one_red :
  let total_marbles := 16
  let marbles_to_draw := 13
  let combinations := Nat.choose
  let total_ways := combinations total_marbles marbles_to_draw
  let favorable_ways :=
    (combinations 5 4) * (combinations 7 6) * (combinations 4 3)
  (total_ways > 0) → 
  (favorable_ways / total_ways) = (1 : ℚ) / (8 : ℚ) := 
by
  sorry

end probability_one_blue_one_white_one_red_l780_780657


namespace intersection_S_T_eq_T_l780_780130

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780130


namespace exponential_inequality_l780_780778

theorem exponential_inequality (x y : ℝ) (hx : x > y) (hy : y > 0) : 2^x - 2^y > 0 :=
sorry

end exponential_inequality_l780_780778


namespace john_has_ten_machines_l780_780910

-- Define the usual cost per ball bearing
def regular_price : ℝ := 1

-- Define the sale price per ball bearing
def sale_price : ℝ := 0.75

-- Define the bulk discount percentage
def bulk_discount_percentage : ℝ := 0.20

-- Define the number of ball bearings per machine
def ball_bearings_per_machine : ℕ := 30

-- Define the total savings
def total_savings : ℝ := 120

-- Define the final cost per ball bearing after bulk discount
def final_sale_price : ℝ := sale_price * (1 - bulk_discount_percentage)

-- Define the cost per machine without sale
def cost_per_machine_without_sale : ℝ := ball_bearings_per_machine * regular_price

-- Define the cost per machine during sale after bulk discount
def cost_per_machine_with_sale : ℝ := ball_bearings_per_machine * final_sale_price

-- Define the savings per machine
def savings_per_machine : ℝ := cost_per_machine_without_sale - cost_per_machine_with_sale

-- Define the number of machines
def number_of_machines : ℝ := total_savings / savings_per_machine

-- The statement to prove
theorem john_has_ten_machines : number_of_machines = 10 :=
by
  sorry

end john_has_ten_machines_l780_780910


namespace quadratic_other_root_l780_780798

noncomputable def find_other_root (m : ℝ) : ℝ :=
  if h : m = 0 then 3 else 3

theorem quadratic_other_root (m : ℝ) : 
  ∃ β, (x^2 + m * x - 6 = (x + 2) * (x + β)) ∧ β = 3 :=
begin
  use 3,
  sorry
end

end quadratic_other_root_l780_780798


namespace union_P_complement_Q_l780_780880

open Set

def P : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }
def Q : Set ℝ := { x | x^2 ≥ 4 }
def R : Set ℝ := { x | -2 < x ∧ x < 2 }
def PQ_union : Set ℝ := P ∪ R

theorem union_P_complement_Q : PQ_union = { x | -2 < x ∧ x ≤ 3 } :=
by sorry

end union_P_complement_Q_l780_780880


namespace quadratic_root_d_value_l780_780862

theorem quadratic_root_d_value :
  (∃ d : ℝ, ∀ x : ℝ, (2 * x^2 + 8 * x + d = 0) ↔ (x = (-8 + Real.sqrt 12) / 4) ∨ (x = (-8 - Real.sqrt 12) / 4)) → 
  d = 6.5 :=
by
  sorry

end quadratic_root_d_value_l780_780862


namespace hyperbola_eccentricity_l780_780974

theorem hyperbola_eccentricity (a c : ℝ) (a_pos : 0 < a) (c_eq_5 : c = 5) (a_eq_4 : a = 4) :
  let e := c / a in e = 5 / 4 :=
by sorry

end hyperbola_eccentricity_l780_780974


namespace combined_share_of_a_and_b_l780_780047

theorem combined_share_of_a_and_b (total_money : ℝ) (ratios : List ℝ) (a_ratio b_ratio : ℝ) :
  total_money = 4500 →
  ratios = [2, 4, 5, 4] →
  a_ratio = 2 →
  b_ratio = 4 →
  (a_ratio + b_ratio) / (ratios.sum) * total_money = 1800 := 
by
  intros h_money h_ratios h_a_ratio h_b_ratio
  rw [h_money, h_ratios, h_a_ratio, h_b_ratio]
  have total_parts : ratios.sum = 15 := by simp
  suffices value_per_part : total_money / ratios.sum = 300 by
    calc
      (a_ratio + b_ratio) / (ratios.sum) * total_money = (2 + 4) / 15 * 4500 : by simp
      ... = 6 / 15 * 4500 : by simp
      ... = 1800 : by simp [mul_div_cancel']
  calc
    total_money / ratios.sum = 4500 / 15 : by simp
    ... = 300 : by norm_num

end combined_share_of_a_and_b_l780_780047


namespace determine_x_l780_780931

theorem determine_x (x : Nat) (h1 : x % 9 = 0) (h2 : x^2 > 225) (h3 : x < 30) : x = 18 ∨ x = 27 :=
sorry

end determine_x_l780_780931


namespace find_side_b_l780_780889

theorem find_side_b (a b c : ℝ) (A B C : ℝ) (ha : a = 2) (hc : c = 2 * Real.sqrt 3)
    (hC : C = Real.pi / 3) (hA : A = Real.pi / 6) (hB : B = Real.pi / 2) : b = 4 := by
  sorry

end find_side_b_l780_780889


namespace intersection_eq_T_l780_780288

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780288


namespace cos_3theta_l780_780836

theorem cos_3theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 := by
  sorry

end cos_3theta_l780_780836


namespace percentage_of_students_who_failed_is_correct_l780_780651

variables (n_b n_g : ℕ) (p_b p_g : ℚ)

def total_students := n_b + n_g

def passed_boys := p_b * n_b
def failed_boys := n_b - passed_boys

def passed_girls := p_g * n_g
def failed_girls := n_g - passed_girls

def total_failed := failed_boys + failed_girls

def percentage_failed := (total_failed / total_students) * 100

theorem percentage_of_students_who_failed_is_correct (hb : n_b = 50) (hg : n_g = 100) (hp_b : p_b = 0.50) (hp_g : p_g = 0.40) :
  percentage_failed n_b n_g p_b p_g = 56.67 :=
by sorry

end percentage_of_students_who_failed_is_correct_l780_780651


namespace registration_methods_for_5_students_l780_780016

def number_of_registration_methods (students groups : ℕ) : ℕ :=
  groups ^ students

theorem registration_methods_for_5_students : number_of_registration_methods 5 2 = 32 := by
  sorry

end registration_methods_for_5_students_l780_780016


namespace Jamie_owns_2_Maine_Coons_l780_780906

-- Definitions based on conditions
variables (Jamie_MaineCoons Gordon_MaineCoons Hawkeye_MaineCoons Jamie_Persians Gordon_Persians Hawkeye_Persians : ℕ)

-- Conditions
axiom Jamie_owns_4_Persians : Jamie_Persians = 4
axiom Gordon_owns_half_as_many_Persians_as_Jamie : Gordon_Persians = Jamie_Persians / 2
axiom Gordon_owns_one_more_Maine_Coon_than_Jamie : Gordon_MaineCoons = Jamie_MaineCoons + 1
axiom Hawkeye_owns_one_less_Maine_Coon_than_Gordon : Hawkeye_MaineCoons = Gordon_MaineCoons - 1
axiom Hawkeye_owns_no_Persian_cats : Hawkeye_Persians = 0
axiom total_number_of_cats_is_13 : Jamie_Persians + Jamie_MaineCoons + Gordon_Persians + Gordon_MaineCoons + Hawkeye_Persians + Hawkeye_MaineCoons = 13

-- Theorem statement
theorem Jamie_owns_2_Maine_Coons : Jamie_MaineCoons = 2 :=
by {
  -- Ideally, you would provide the proof here, stepping through algebraically as shown in the solution,
  -- but we are skipping the proof as specified in the instructions.
  sorry
}

end Jamie_owns_2_Maine_Coons_l780_780906


namespace number_of_people_l780_780509

-- Define the total number of candy bars
def total_candy_bars : ℝ := 5.0

-- Define the amount of candy each person gets
def candy_per_person : ℝ := 1.66666666699999

-- Define a theorem to state that dividing the total candy bars by candy per person gives 3 people
theorem number_of_people : total_candy_bars / candy_per_person = 3 :=
  by
  -- Proof omitted
  sorry

end number_of_people_l780_780509


namespace largest_n_for_binom_equality_l780_780598

theorem largest_n_for_binom_equality : ∃ (n : ℕ), (∑ i : {i // i = 4 ∨ i = 5}, Nat.choose 10 i) = Nat.choose 11 n ∧ n = 6 :=
by
  sorry

end largest_n_for_binom_equality_l780_780598


namespace S_inter_T_eq_T_l780_780298

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780298


namespace largest_n_binom_l780_780543

theorem largest_n_binom (n : ℕ) (h : 0 ≤ n ∧ n ≤ 11) : 
  (binomial 10 4 + binomial 10 5 = binomial 11 n) → n = 6 :=
by
  sorry

end largest_n_binom_l780_780543


namespace count_satisfying_n_l780_780359

def isPerfectSquare (m : ℤ) : Prop :=
  ∃ k : ℤ, m = k * k

def countNsatisfying (low high : ℤ) (e : ℤ → ℤ) : ℤ :=
  (Finset.range (Int.natAbs (high - low + 1))).count (λ i, isPerfectSquare (e (low + i)))

theorem count_satisfying_n : countNsatisfying 5 15 (λ n, 2 * n^2 + 3 * n + 2) = 1 :=
by
  sorry

end count_satisfying_n_l780_780359


namespace no_such_integer_exists_l780_780413

def moving_first_digit_to_end (x : ℕ) : ℕ :=
  let n : ℕ := (x.digits.length - 1)
  let a : ℕ := x / (10^n)
  let N : ℕ := x % (10^n)
  10 * N + a

theorem no_such_integer_exists :
  ¬ ∃ x : ℕ, moving_first_digit_to_end x = 2 * x :=
by
  sorry

end no_such_integer_exists_l780_780413


namespace original_price_of_cycle_l780_780669

theorem original_price_of_cycle (SP : ℝ) (gain_percent : ℝ) (original_price : ℝ) 
  (hSP : SP = 1260) (hgain : gain_percent = 0.40) (h_eq : SP = original_price * (1 + gain_percent)) :
  original_price = 900 :=
by
  sorry

end original_price_of_cycle_l780_780669


namespace base_b_when_square_is_1325_l780_780370

theorem base_b_when_square_is_1325 (b : ℕ) : (35_b:square:b) = 3 * b + 5 ∧ 1325_b = b^3 + 3 * b^2 + 2 * b + 5 -> b = 10 :=
by
  sorry

end base_b_when_square_is_1325_l780_780370


namespace intersection_S_T_eq_T_l780_780268

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780268


namespace required_ratio_l780_780869

theorem required_ratio (M N : ℕ) (hM : M = 8) (hN : N = 27) : 8 / 27 < 10 / 27 :=
by { sorry }

end required_ratio_l780_780869


namespace largest_n_binomial_l780_780573

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l780_780573


namespace Inez_initial_money_l780_780410

theorem Inez_initial_money (X : ℝ) (h : X - (X / 2 + 50) = 25) : X = 150 :=
by
  sorry

end Inez_initial_money_l780_780410


namespace part1_part2_l780_780939

-- Define the sum of the first n terms of the sequence {a_n} as S_n.
def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ := (finset.range n).sum (λ i, a (i + 1))

-- Given conditions
theorem part1 (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, 3 - 2 * S n / a n = (1/3)^(n-1)) : 
  ∀ n, a n = 3^(n-1) := sorry

-- Define sequences b_n and T_n
def bn (a : ℕ → ℕ) (n : ℕ) : ℕ := 3^n / ((a (n + 1) - 1) * (a (n + 2) - 1))
def Tn (b : ℕ → ℕ) (n : ℕ) : ℕ := (finset.range n).sum (λ i, b (i + 1))

-- Proving the upper bound on T_n
theorem part2 (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : ∀ n, a n = 3^(n-1))
  (h2 : ∀ n, b n = 3^n / ((a (n + 1) - 1) * (a (n + 2) - 1)))
  (h3 : ∀ n, T n = (finset.range n).sum (λ i, b (i + 1))) : 
  ∀ n, T n < 1/4 := sorry

end part1_part2_l780_780939


namespace distinct_triangles_of_square_pyramid_l780_780354

theorem distinct_triangles_of_square_pyramid : ∀ (V : Finset ℕ), V.card = 5 → (Finset.card (V.powerset.filter (λ s, s.card = 3))) = 10 :=
by
  intros V V_card
  sorry

end distinct_triangles_of_square_pyramid_l780_780354


namespace intersection_of_sets_l780_780177

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780177


namespace find_common_ratio_l780_780434

variable {a : ℕ → ℝ} {q : ℝ}

-- Define that a is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem find_common_ratio
  (h1 : is_geometric_sequence a q)
  (h2 : 0 < q)
  (h3 : a 1 * a 3 = 1)
  (h4 : sum_first_n_terms a 3 = 7) :
  q = 1 / 2 :=
sorry

end find_common_ratio_l780_780434


namespace james_sodas_per_day_l780_780417

theorem james_sodas_per_day : 
  (∃ packs : ℕ, ∃ sodas_per_pack : ℕ, ∃ additional_sodas : ℕ, ∃ days : ℕ,
    packs = 5 ∧ sodas_per_pack = 12 ∧ additional_sodas = 10 ∧ days = 7 ∧
    ((packs * sodas_per_pack + additional_sodas) / days) = 10) :=
by
  use 5, 12, 10, 7
  split; simp; sorry

end james_sodas_per_day_l780_780417


namespace james_matches_l780_780897

theorem james_matches (dozen_boxes : ℕ) (matches_per_box : ℕ) (dozen_value : ℕ) 
  (h1 : dozen_boxes = 5) (h2 : matches_per_box = 20) (h3 : dozen_value = 12) :
  dozen_boxes * dozen_value * matches_per_box = 1200 :=
by
  rw [h1, h2, h3]
  calc
    5 * 12 * 20 = 60 * 20 := by norm_num
    ... = 1200 := by norm_num

end james_matches_l780_780897


namespace problem_mexican_olympiad_1988_l780_780444

-- Define that f is a function from ℕ to ℕ
def f : ℕ → ℕ

-- Define the main theorem
theorem problem_mexican_olympiad_1988 :
  (∀ m n : ℕ, f (f m + f n) = m + n) →
  f 1988 = 1988 :=
by {
  intros h,
  sorry
}

end problem_mexican_olympiad_1988_l780_780444


namespace taqeesha_grade_correct_l780_780871

-- Definitions for conditions
def total_score_of_24_students := 24 * 82
def total_score_of_25_students (T: ℕ) := 25 * 84
def taqeesha_grade := 132

-- Theorem statement forming the proof problem
theorem taqeesha_grade_correct
    (h1: total_score_of_24_students + taqeesha_grade = total_score_of_25_students taqeesha_grade): 
    taqeesha_grade = 132 :=
by
  sorry

end taqeesha_grade_correct_l780_780871


namespace intersection_of_S_and_T_l780_780237

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780237


namespace present_age_of_son_l780_780636

-- Define variables F (father's age) and S (son's age)
variables (F S : ℕ)

-- Given conditions
def condition1 : Prop := F = S + 24
def condition2 : Prop := F + 2 = 2 * (S + 2)

-- Theorem stating that if the conditions hold, then the son's age is 22
theorem present_age_of_son (h1 : condition1) (h2 : condition2) : S = 22 :=
by sorry

end present_age_of_son_l780_780636


namespace phi_value_l780_780977

theorem phi_value (φ : ℝ) (k : ℤ)
  (h1 : ∀ x : ℝ, 3 * sin (2 * x + φ + π / 3) = 3 * sin (-(2 * x + φ + π / 3)))
  (h2 : abs φ ≤ π / 2) : φ = π / 6 :=
sorry

end phi_value_l780_780977


namespace original_number_of_men_is_20_l780_780635

-- Conditions
variables (x : ℕ) -- original number of men
variable (work : ℕ) -- total amount of work in man-days
variable (absent : ℕ) -- number of men absent
variable (days1 : ℕ) -- initial days planned
variable (days2 : ℕ) -- days taken after absence

-- Definitions based on conditions
def group_planned_to_complete_in_20_days : days1 = 20 := sorry
def ten_men_became_absent : absent = 10 := sorry
def rest_completed_in_40_days : days2 = 40 := sorry
def work_calculated : work = x * days1 := sorry
def work_done_by_remaining_men : work = (x - absent) * days2 := sorry

-- Question translated to proof problem
theorem original_number_of_men_is_20 
    (h1 : group_planned_to_complete_in_20_days)
    (h2 : ten_men_became_absent)
    (h3 : rest_completed_in_40_days)
    (h4 : work_calculated)
    (h5 : work_done_by_remaining_men) : x = 20 := 
begin
  -- proof is omitted as per the requirement
  sorry
end

end original_number_of_men_is_20_l780_780635


namespace radius_of_sphere_touching_four_l780_780106

noncomputable def r_sphere_internally_touching_four := Real.sqrt (3 / 2) + 1
noncomputable def r_sphere_externally_touching_four := Real.sqrt (3 / 2) - 1

theorem radius_of_sphere_touching_four (r : ℝ) (R := Real.sqrt (3 / 2)) :
  r = R + 1 ∨ r = R - 1 :=
by
  sorry

end radius_of_sphere_touching_four_l780_780106


namespace largest_n_for_binomial_equality_l780_780529

theorem largest_n_for_binomial_equality :
  ∃ n : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 n ) ∧
  ∀ m : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 m ) → m ≤ 6 :=
begin
  sorry
end

end largest_n_for_binomial_equality_l780_780529


namespace largest_n_binom_identity_l780_780558

open Nat

theorem largest_n_binom_identity :
  (∃ n, binom 11 n = binom 10 4 + binom 10 5) ∧
  (∀ m, binom 11 m = binom 10 4 + binom 10 5 → m ≤ 6) :=
by
  sorry

end largest_n_binom_identity_l780_780558


namespace intersection_S_T_l780_780322

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780322


namespace intersection_S_T_l780_780147

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780147


namespace common_remainder_zero_l780_780502

theorem common_remainder_zero (n r : ℕ) (h1: n > 1) 
(h2 : n % 25 = r) (h3 : n % 7 = r) (h4 : n = 175) : r = 0 :=
by
  sorry

end common_remainder_zero_l780_780502


namespace range_of_m_length_of_chord_l780_780808

-- Definition of Circle C
def CircleC (x y m : ℝ) := x^2 + y^2 - 2*x - 4*y + m = 0

-- Definition of Circle D
def CircleD (x y : ℝ) := (x + 3)^2 + (y + 1)^2 = 16

-- Definition of Line l
def LineL (x y : ℝ) := x + 2*y - 4 = 0

-- Problem 1: Prove range of values for m
theorem range_of_m (m : ℝ) : (∀ x y, CircleC x y m) → m < 5 := by
  sorry

-- Problem 2: Prove length of chord MN
theorem length_of_chord (x y : ℝ) :
  CircleC x y 4 ∧ CircleD x y ∧ LineL x y →
  (∃ MN, MN = (4*Real.sqrt 5) / 5) := by
    sorry

end range_of_m_length_of_chord_l780_780808


namespace intersection_S_T_eq_T_l780_780205

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780205


namespace geometric_progression_common_point_l780_780700

theorem geometric_progression_common_point (a r : ℝ) :
  ∀ x y : ℝ, (a ≠ 0 ∧ x = 1 ∧ y = 0) ↔ (a * x + (a * r) * y = a * r^2) := by
  sorry

end geometric_progression_common_point_l780_780700


namespace kaleb_final_score_l780_780391

variable (score_first_half : ℝ) (bonus_special_q : ℝ) (bonus_streak : ℝ) (score_second_half : ℝ) (penalty_speed_round : ℝ) (penalty_lightning_round : ℝ)

-- Given conditions from the problem statement
def kaleb_initial_scores (score_first_half score_second_half : ℝ) := 
  score_first_half = 43 ∧ score_second_half = 23

def kaleb_bonuses (score_first_half bonus_special_q bonus_streak : ℝ) :=
  bonus_special_q = 0.20 * score_first_half ∧ bonus_streak = 0.05 * score_first_half

def kaleb_penalties (score_second_half penalty_speed_round penalty_lightning_round : ℝ) := 
  penalty_speed_round = 0.10 * score_second_half ∧ penalty_lightning_round = 0.08 * score_second_half

-- The final score adjusted with all bonuses and penalties
def kaleb_adjusted_score (score_first_half score_second_half bonus_special_q bonus_streak penalty_speed_round penalty_lightning_round : ℝ) : ℝ := 
  score_first_half + bonus_special_q + bonus_streak + score_second_half - penalty_speed_round - penalty_lightning_round

theorem kaleb_final_score :
  kaleb_initial_scores score_first_half score_second_half ∧
  kaleb_bonuses score_first_half bonus_special_q bonus_streak ∧
  kaleb_penalties score_second_half penalty_speed_round penalty_lightning_round →
  kaleb_adjusted_score score_first_half score_second_half bonus_special_q bonus_streak penalty_speed_round penalty_lightning_round = 72.61 :=
by
  intros
  sorry

end kaleb_final_score_l780_780391


namespace increasing_intervals_proof_extreme_points_inequality_l780_780811

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ :=
  (x^2 + m) / x - 2 * Real.log x

-- Define the conditions for the intervals where f is monotonically increasing
def increasing_intervals (m : ℝ) : Prop :=
  if m ≤ -1 then
    ∀ x y, 0 < x → x < y → f x m ≤ f y m
  else if -1 < m ∧ m < 0 then
    ∀ x, ((0 < x ∧ x < 1 - Real.sqrt (1 + m)) ∨ (1 + Real.sqrt (1 + m) < x)) → f x m ≤ f (x + 1) m
  else
    ∀ x y, 0 < x → x < y → f x m ≤ f y m

-- Statement for the first problem
theorem increasing_intervals_proof (m : ℝ) : increasing_intervals m := sorry

-- Define the conditions to prove f(x₂) < x₂ - 1
def two_extreme_points (x1 x2 m: ℝ) : Prop :=
  x1 < x2 ∧ x1 = 1 - Real.sqrt (1 + m) ∧ x2 = 1 + Real.sqrt (1 + m) ∧ -1 < m ∧ m < 0

-- Statement for the second problem
theorem extreme_points_inequality (x1 x2 m : ℝ) (h: two_extreme_points x1 x2 m) :
  f x2 m < x2 - 1 := sorry

end increasing_intervals_proof_extreme_points_inequality_l780_780811


namespace intersection_of_S_and_T_l780_780227

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780227


namespace distance_between_parallel_lines_l780_780489

-- Define the lines l1 and l2
def line1 (x y : ℝ) : Prop := 3 * x - 4 * y - 1 = 0
def line2 (x y : ℝ) : Prop := 6 * x - 8 * y - 7 = 0

-- Function to compute the distance between two parallel lines in the form Ax + By + C1 = 0 and Ax + By + C2 = 0
def parallel_lines_distance (A B C1 C2 : ℝ) : ℝ :=
  abs (C1 - C2) / real.sqrt (A^2 + B^2)

-- The theorem stating the distance between the given lines
theorem distance_between_parallel_lines : 
  parallel_lines_distance 6 (-8) (-2) (-7) = 1 / 2 := 
  by sorry

end distance_between_parallel_lines_l780_780489


namespace intersection_S_T_eq_T_l780_780272

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780272


namespace factorization_of_210_l780_780394

theorem factorization_of_210 :
  {l : List ℕ // l.product = 210 ∧ l.length = 4}.quotient.mk.card = 15 :=
sorry

end factorization_of_210_l780_780394


namespace intersection_S_T_l780_780196

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780196


namespace intersection_S_T_eq_T_l780_780267

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780267


namespace prob_A_exactly_2_hits_prob_B_atleast_2_hits_prob_B_exactly_2_more_than_A_l780_780027

-- Defining the probabilities of hitting the target for A and B
def prob_A_hit (p: ℚ) := p = 1 / 2
def prob_B_hit (p: ℚ) := p = 2 / 3

-- Defining the number of shots taken by A and B
def shots_A (n: ℕ) := n = 3
def shots_B (n: ℕ) := n = 3

-- Theorem statements for questions I, II, III
theorem prob_A_exactly_2_hits (p_A: ℚ) (n_A: ℕ): prob_A_hit p_A → shots_A n_A → 
  (∃ P: ℚ, P = (3.choose 2) * (p_A)^2 * (1 - p_A)^(n_A - 2) ∧ P = 3 / 8) := 
by
  intros h_prob_A h_shots_A
  use (3.choose 2) * (p_A)^2 * (1 - p_A)^(n_A - 2)
  split
  · -- The actual probability calculation
    rw [h_prob_A, h_shots_A]
    norm_num
  · -- The given probability
    norm_num
  done

theorem prob_B_atleast_2_hits (p_B: ℚ) (n_B: ℕ): prob_B_hit p_B → shots_B n_B → 
  (∃ P: ℚ, P = (3.choose 2) * (p_B)^2 * (1 - p_B)^(n_B - 2) + (3.choose 3) * (p_B)^3 ∧ P = 20 / 27) :=
by
  intros h_prob_B h_shots_B
  use (3.choose 2) * (p_B)^2 * (1 - p_B)^(n_B - 2) + (3.choose 3) * (p_B)^3
  split
  · -- The actual probability calculation
    rw [h_prob_B, h_shots_B]
    norm_num
  · -- The given probability
    norm_num
  done

theorem prob_B_exactly_2_more_than_A (p_A p_B: ℚ) (n_A n_B: ℕ): prob_A_hit p_A → prob_B_hit p_B → shots_A n_A → shots_B n_B → 
  (∃ P: ℚ, P = (3.choose 2) * (p_B)^2 * (1 - p_B)^(n_B - 2) * (1 - p_A)^n_A + (3.choose 3) * (p_B)^3 * (p_A) * (1 - p_A)^(n_A - 1) ∧ P = 1 / 6) :=
by
  intros h_prob_A h_prob_B h_shots_A h_shots_B
  use (3.choose 2) * (p_B)^2 * (1 - p_B)^(n_B - 2) * (1 - p_A)^n_A + (3.choose 3) * (p_B)^3 * (p_A) * (1 - p_A)^(n_A - 1)
  split
  · -- The actual probability calculation
    rw [h_prob_A, h_prob_B, h_shots_A, h_shots_B]
    norm_num
  · -- The given probability
    norm_num
  done

end prob_A_exactly_2_hits_prob_B_atleast_2_hits_prob_B_exactly_2_more_than_A_l780_780027


namespace hyperbola_is_given_equation_l780_780346

noncomputable def hyperbola_equation : Prop :=
  ∃ a b : ℝ, 
    (a > 0 ∧ b > 0) ∧ 
    (4^2 = a^2 + b^2) ∧ 
    (a = b) ∧ 
    (∀ x y : ℝ, (x^2 / 8 - y^2 / 8 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1))

theorem hyperbola_is_given_equation : hyperbola_equation :=
sorry

end hyperbola_is_given_equation_l780_780346


namespace min_distance_between_graphs_l780_780517

noncomputable def minimum_distance (a : ℝ) (h : 1 < a) : ℝ :=
  if h1 : a ≤ Real.exp (1 / Real.exp 1) then 0
  else Real.sqrt 2 * (1 + Real.log (Real.log a)) / (Real.log a)

theorem min_distance_between_graphs (a : ℝ) (h1 : 1 < a) :
  minimum_distance a h1 = 
  if a ≤ Real.exp (1 / Real.exp 1) then 0
  else Real.sqrt 2 * (1 + Real.log (Real.log a)) / (Real.log a) :=
by
  intros
  sorry

end min_distance_between_graphs_l780_780517


namespace min_students_green_eyes_backpack_no_glasses_l780_780096

theorem min_students_green_eyes_backpack_no_glasses
  (S G B Gl : ℕ)
  (h_S : S = 25)
  (h_G : G = 15)
  (h_B : B = 18)
  (h_Gl : Gl = 6)
  : ∃ x, x ≥ 8 ∧ x + Gl ≤ S ∧ x ≤ min G B :=
sorry

end min_students_green_eyes_backpack_no_glasses_l780_780096


namespace ratio_A_B_l780_780744

theorem ratio_A_B :
  let A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
  let B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)
  0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 :=
by
  let A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
  let B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)
  have hA : A = 1400 := rfl
  have hB : B = 1500 := rfl
  have h_ratio : A / B = 14 / 15 := by
    rw [hA, hB]
    norm_num
  exact And.intro (by norm_num) (by norm_num)

end ratio_A_B_l780_780744


namespace total_lunch_cost_310_56_l780_780698

-- Conditions
def meal_cost_jose := 60
def meal_cost_rick := meal_cost_jose / 1.5
def meal_cost_adam := (meal_cost_rick * 2) / 3
def meal_cost_sophia := meal_cost_rick
def meal_cost_emma_full := meal_cost_jose
def meal_cost_emma := meal_cost_emma_full * (1 - 0.20)
def appetizer_cost := 20
def tax_rate := 0.08

-- Shared appetizer cost per person
def shared_appetizer_cost := appetizer_cost / 5

-- Total cost of meals before taxes and shared appetizer cost
def total_meal_cost_before := meal_cost_adam + meal_cost_rick + meal_cost_jose + meal_cost_sophia + meal_cost_emma

-- Adding shared appetizer cost to each friend's meal cost
def adam_total := meal_cost_adam + shared_appetizer_cost
def rick_total := meal_cost_rick + shared_appetizer_cost
def jose_total := meal_cost_jose + shared_appetizer_cost
def sophia_total := meal_cost_sophia + shared_appetizer_cost
def emma_total := meal_cost_emma + shared_appetizer_cost

def total_cost_including_appetizer := adam_total + rick_total + jose_total + sophia_total + emma_total
def total_cost_including_taxes := total_cost_including_appetizer * (1 + tax_rate)

-- Statement to prove
theorem total_lunch_cost_310_56 : total_cost_including_taxes = 310.56 := by sorry

end total_lunch_cost_310_56_l780_780698


namespace Brenda_bakes_20_cakes_a_day_l780_780066

-- Define the conditions
variables (x : ℕ)

-- Other necessary definitions
def cakes_baked_in_9_days (x : ℕ) : ℕ := 9 * x
def cakes_after_selling_half (total_cakes : ℕ) : ℕ := total_cakes.div2

-- Given condition that Brenda has 90 cakes after selling half
def final_cakes_after_selling : ℕ := 90

-- Mathematical statement we want to prove
theorem Brenda_bakes_20_cakes_a_day (x : ℕ) (h : cakes_after_selling_half (cakes_baked_in_9_days x) = final_cakes_after_selling) : x = 20 :=
by sorry

end Brenda_bakes_20_cakes_a_day_l780_780066


namespace Callum_points_l780_780913

theorem Callum_points
  (points_per_win : ℕ := 10)
  (total_matches : ℕ := 8)
  (krishna_win_fraction : ℚ := 3/4) :
  let callum_win_fraction := 1 - krishna_win_fraction in
  let callum_wins := callum_win_fraction * total_matches in
  let callum_points := callum_wins * points_per_win in
  callum_points = 20 := 
by
  sorry

end Callum_points_l780_780913


namespace intersection_S_T_l780_780189

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780189


namespace intersection_eq_T_l780_780160

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780160


namespace distinct_lines_in_equilateral_triangle_l780_780705

-- Define an equilateral triangle and its properties
structure EquilateralTriangle (A B C : Type) :=
(side_length : ℝ)
(angle_deg : Real := 60)
(altitude : A → B → ℝ)
(median : A → B → ℝ)
(angle_bisector : A → B → ℝ)

-- Define the concept of distinct lines in the context of an equilateral triangle
def distinct_lines (A B C : Type) [et : EquilateralTriangle A B C] : ℕ :=
3

-- Example proof statement
theorem distinct_lines_in_equilateral_triangle (A B C : Type) [et : EquilateralTriangle A B C] :
  distinct_lines A B C = 3 :=
by
  sorry

end distinct_lines_in_equilateral_triangle_l780_780705


namespace find_p_q_r_sum_l780_780929

noncomputable def largest_real_solution :=
  let f (x : ℝ) := (4 / (x - 4)) + (6 / (x - 6)) + (18 / (x - 18)) + (20 / (x - 20)) - (x^2 - 13 * x - 6) in
  ∃ n : ℝ, is_largest_real_solution f n

theorem find_p_q_r_sum (f : ℝ → ℝ) : 
  (∀ x, f x = (4 / (x - 4)) + (6 / (x - 6)) + (18 / (x - 18)) + (20 / (x - 20)) - (x^2 - 13 * x - 6)) →
  let n := 13 + real.sqrt 61 in 
  p = 13 → q = 61 → r = 0 → p + q + r = 74 :=
by
  intros f_eq n_def p_def q_def r_def
  unfold n_def p_def q_def r_def
  sorry

end find_p_q_r_sum_l780_780929


namespace fraction_meaningful_iff_l780_780855

theorem fraction_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_iff_l780_780855


namespace bell_ratio_l780_780453

theorem bell_ratio :
  ∃ (B3 B2 : ℕ), 
  B2 = 2 * 50 ∧ 
  50 + B2 + B3 = 550 ∧ 
  (B3 / B2 = 4) := 
sorry

end bell_ratio_l780_780453


namespace intersection_S_T_eq_T_l780_780207

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780207


namespace find_two_first_digits_fractional_part_l780_780430

noncomputable def minimal_root : ℝ := 2 - Real.sqrt 2

theorem find_two_first_digits_fractional_part :
  let S := (∑ k in Finset.range 20, minimal_root ^ (k + 1))
  let fractional_part := S - S.floor
  0.41 ≤ fractional_part ∧ fractional_part < 0.42 :=
sorry

end find_two_first_digits_fractional_part_l780_780430


namespace CatCafePawRatio_l780_780071

-- Define the context
def CatCafeMeow (P : ℕ) := 3 * P
def CatCafePaw (P : ℕ) := P
def CatCafeCool := 5
def TotalCats (P : ℕ) := CatCafeMeow P + CatCafePaw P

-- State the theorem
theorem CatCafePawRatio (P : ℕ) (n : ℕ) : 
  CatCafeCool = 5 →
  CatCafeMeow P = 3 * CatCafePaw P →
  TotalCats P = 40 →
  P = 10 →
  n * CatCafeCool = P →
  n = 2 :=
by
  intros
  sorry

end CatCafePawRatio_l780_780071


namespace fraction_meaningful_iff_l780_780854

theorem fraction_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_iff_l780_780854


namespace fox_eats_80_candies_fox_cannot_eat_65_candies_l780_780672

-- Conditions
def total_candies : ℕ := 100
def piles : Fin 3 → ℕ

-- Definition of the distribution ensuring the fox eats 80 candies
-- Devise a way for the fox to distribute the candies into piles so that she eats exactly 80 candies
theorem fox_eats_80_candies (h : piles 0 + piles 1 + piles 2 = total_candies) :
  ∃ i : Fin 3, ∀ j : Fin 2, piles i = 80 ∨ (piles (1 - j) + piles (2 - j) = 20 - piles i) :=
sorry

-- Definition to show the impossibility of fox eating 65 given the constraints
-- Can the fox ensure that she eats exactly 65 candies in the end
theorem fox_cannot_eat_65_candies (h : piles 0 + piles 1 + piles 2 = total_candies) :
  ¬ ∃ i : Fin 3, piles i = 65 :=
sorry

end fox_eats_80_candies_fox_cannot_eat_65_candies_l780_780672


namespace lana_and_mike_security_deposit_l780_780427

open Real

noncomputable def security_deposit_cost_in_euros : ℝ :=
let daily_rate := 125
let pet_fee_per_dog := 100
let tourism_tax_rate := 0.10
let service_cleaning_fee_rate := 0.20
let security_deposit_rate := 0.50
let conversion_rate_usd_to_euro := 0.85
let number_of_days := 14
let number_of_dogs := 2 in
let total_rental_cost := daily_rate * number_of_days in
let total_pet_fee := pet_fee_per_dog * number_of_dogs in
let tourism_tax := tourism_tax_rate * total_rental_cost in
let service_cleaning_fee := service_cleaning_fee_rate * total_rental_cost in
let total_cost_before_deposit := total_rental_cost + total_pet_fee + tourism_tax + service_cleaning_fee in
let security_deposit_usd := security_deposit_rate * total_cost_before_deposit in
let security_deposit_euros := security_deposit_usd * conversion_rate_usd_to_euro in
security_deposit_euros

theorem lana_and_mike_security_deposit :
  security_deposit_cost_in_euros ≈ 1051.88 :=
  sorry

end lana_and_mike_security_deposit_l780_780427


namespace no_such_integers_l780_780730

theorem no_such_integers (x y z : ℤ) : ¬ ((x - y)^3 + (y - z)^3 + (z - x)^3 = 2011) :=
sorry

end no_such_integers_l780_780730


namespace jake_present_weight_l780_780845

theorem jake_present_weight :
  ∃ (J S : ℕ), (J - 12 = 2 * S) ∧ (J + S = 156) ∧ (J = 108) :=
begin
  sorry
end

end jake_present_weight_l780_780845


namespace better_offer_saves_800_l780_780660

theorem better_offer_saves_800 :
  let initial_order := 20000
  let discount1 (x : ℝ) := x * 0.70 * 0.90 - 800
  let discount2 (x : ℝ) := x * 0.75 * 0.80 - 1000
  discount1 initial_order - discount2 initial_order = 800 :=
by
  sorry

end better_offer_saves_800_l780_780660


namespace train_speed_is_36_kph_l780_780694

noncomputable def speed_of_train (length_train length_bridge time_to_pass : ℕ) : ℕ :=
  let total_distance := length_train + length_bridge
  let speed_mps := total_distance / time_to_pass
  let speed_kph := speed_mps * 3600 / 1000
  speed_kph

theorem train_speed_is_36_kph :
  speed_of_train 360 140 50 = 36 :=
by
  sorry

end train_speed_is_36_kph_l780_780694


namespace platform_length_is_350_l780_780024

variables (L : ℕ)

def train_length := 300
def time_to_cross_pole := 18
def time_to_cross_platform := 39

-- Speed of the train when crossing the pole
def speed_cross_pole : ℚ := train_length / time_to_cross_pole

-- Speed of the train when crossing the platform
def speed_cross_platform (L : ℕ) : ℚ := (train_length + L) / time_to_cross_platform

-- The main goal is to prove that the length of the platform is 350 meters
theorem platform_length_is_350 (L : ℕ) (h : speed_cross_pole = speed_cross_platform L) : L = 350 := sorry

end platform_length_is_350_l780_780024


namespace smallest_bottles_needed_l780_780072

/-- Christine needs at least 60 fluid ounces of milk, the store sells milk in 250 milliliter bottles,
and there are 32 fluid ounces in 1 liter. The smallest number of bottles Christine should purchase
is 8. -/
theorem smallest_bottles_needed
  (fl_oz_needed : ℕ := 60)
  (ml_per_bottle : ℕ := 250)
  (fl_oz_per_liter : ℕ := 32) :
  let liters_needed := fl_oz_needed / fl_oz_per_liter
  let ml_needed := liters_needed * 1000
  let bottles := (ml_needed + ml_per_bottle - 1) / ml_per_bottle
  bottles = 8 :=
by
  sorry

end smallest_bottles_needed_l780_780072


namespace largest_integer_comb_l780_780599

theorem largest_integer_comb (n: ℕ) (h1 : n = 6) : 
  (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) :=
by
  rw [h1]
  exact sorry

end largest_integer_comb_l780_780599


namespace probability_same_heads_l780_780425

def outcomes_keiko : ℕ := 2
def outcomes_ephraim : ℕ := 3

theorem probability_same_heads : 
  (probability (outcomes_keiko = outcomes_ephraim) = 5/32) :=
sorry

end probability_same_heads_l780_780425


namespace largest_n_binom_equality_l780_780523

theorem largest_n_binom_equality :
  ∃ n : ℕ, (binomial 10 4 + binomial 10 5 = binomial 11 n) ∧ ∀ m : ℕ, (binomial 11 m = binomial 11 5) → m ≤ 6 :=
by
  sorry

end largest_n_binom_equality_l780_780523


namespace problem_l780_780445

noncomputable def α : ℝ := 3
noncomputable def β : ℝ := 3

theorem problem 
  (α β : ℝ)
  (h : ∀ x y z : ℝ, α * (x * y + y * z + z * x) ≤ 
    (real.sqrt (x^2 + x * y + y^2) * real.sqrt (y^2 + y * z + z^2) + 
     real.sqrt (y^2 + y * z + z^2) * real.sqrt (z^2 + z * x + x^2) + 
     real.sqrt (z^2 + z * x + x^2) * real.sqrt (x^2 + x * y + y^2)) ∧ 
     (real.sqrt (x^2 + x * y + y^2) * real.sqrt (y^2 + y * z + z^2) + 
     real.sqrt (y^2 + y * z + z^2) * real.sqrt (z^2 + z * x + x^2) + 
     real.sqrt (z^2 + z * x + x^2) * real.sqrt (x^2 + x * y + y^2)) ≤ β * (x^2 + y^2 + z^2)
  ): α = 3 ∧ β = 3 := sorry

end problem_l780_780445


namespace students_only_biology_students_biology_or_chemistry_but_not_both_l780_780749

def students_enrolled_in_both : ℕ := 15
def total_biology_students : ℕ := 35
def students_only_chemistry : ℕ := 18

theorem students_only_biology (h₀ : students_enrolled_in_both ≤ total_biology_students) :
  total_biology_students - students_enrolled_in_both = 20 := by
  sorry

theorem students_biology_or_chemistry_but_not_both :
  total_biology_students - students_enrolled_in_both + students_only_chemistry = 38 := by
  sorry

end students_only_biology_students_biology_or_chemistry_but_not_both_l780_780749


namespace largest_n_binom_equality_l780_780527

theorem largest_n_binom_equality :
  ∃ n : ℕ, (binomial 10 4 + binomial 10 5 = binomial 11 n) ∧ ∀ m : ℕ, (binomial 11 m = binomial 11 5) → m ≤ 6 :=
by
  sorry

end largest_n_binom_equality_l780_780527


namespace max_n_is_11_l780_780967

noncomputable def max_n (a1 d : ℝ) : ℕ :=
if h : d < 0 then
  11
else
  sorry

theorem max_n_is_11 (d : ℝ) (a1 : ℝ) (c : ℝ) :
  (d / 2) * (22 ^ 2) + (a1 - (d / 2)) * 22 + c ≥ 0 →
  22 = (a1 - (d / 2)) / (- (d / 2)) →
  max_n a1 d = 11 :=
by
  intros h1 h2
  rw [max_n]
  split_ifs
  · exact rfl
  · exact sorry

end max_n_is_11_l780_780967


namespace break_point_height_l780_780670

-- Define the conditions as constants
def height_of_flagpole : ℝ := 8
def distance_from_base : ℝ := 3

-- Define the function to calculate the break height
def break_height (flagpole_height : ℝ) (ground_distance : ℝ) : ℝ :=
  (Real.sqrt (flagpole_height ^ 2 + ground_distance ^ 2)) / 2

theorem break_point_height : break_height height_of_flagpole distance_from_base = 4.27 :=
by
  -- proof omitted
  sorry

end break_point_height_l780_780670


namespace intersection_eq_T_l780_780159

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780159


namespace intersection_S_T_eq_T_l780_780224

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780224


namespace bn_eq_n_add_1_l780_780122

/-- Define the function f(x) = -1 / (x + 2) -/
def f (x : ℝ) : ℝ := -1 / (x + 2)

/-- Define the sequence a_n such that a_{n+1} is on the graph of f(x) and a_1 = f(0) -/
def a : ℕ+ → ℝ
| ⟨1, _⟩ => -1 / 2
| ⟨n+1, _⟩ => f (a ⟨n, Nat.succ_pos n⟩)

/-- Define the sequence b_n such that b_n = 1 / (a_n + 1) -/
def b : ℕ+ → ℝ :=
  λ n => 1 / (a n + 1)

/-- Prove that the sequence b_n = n + 1 -/
theorem bn_eq_n_add_1 (n : ℕ+) : b n = n + 1 := by
  sorry

end bn_eq_n_add_1_l780_780122


namespace find_angle_A_find_triangle_area_l780_780864

noncomputable def triangle_find_angle (a : ℝ) (c : ℝ) (C : ℝ) : ℝ := 
let sin_C := Real.sin C in 
let sin_A := a * sin_C / c in 
Real.arcsin sin_A

noncomputable def triangle_area (a : ℝ) (b : ℝ) (C : ℝ) : ℝ :=
0.5 * a * b * Real.sin C

theorem find_angle_A : 
  ∀ (a c C : ℝ),
  a = 2 * Real.sqrt 3 / 3 →
  c = 2 →
  C = Real.pi / 3 →
  triangle_find_angle a c C = Real.pi / 6 :=
by
  intros
  sorry

theorem find_triangle_area :
  ∀ (a b c C : ℝ),
  c = 2 →
  C = Real.pi / 3 →
  a = 2 * Real.sqrt 3/ 3 →
  b = 2 * a →
  triangle_area a b C = 2 * Real.sqrt 3/ 3 :=
by
  intros
  sorry

end find_angle_A_find_triangle_area_l780_780864


namespace Greg_more_than_Sharon_l780_780353

-- Define the harvest amounts
def Greg_harvest : ℝ := 0.4
def Sharon_harvest : ℝ := 0.1

-- Show that Greg harvested 0.3 more acres than Sharon
theorem Greg_more_than_Sharon : Greg_harvest - Sharon_harvest = 0.3 := by
  sorry

end Greg_more_than_Sharon_l780_780353


namespace check_statements_l780_780883

-- Define the coordinates of points M and N
variables {x1 y1 x2 y2 a b c : ℝ}
-- Define the line l equation
def line_l (x y : ℝ) : ℝ := a * x + b * y + c
-- Define δ
def δ := (a * x1 + b * y1 + c) / (a * x2 + b * y2 + c)

-- Define statements as Lean predicates
def statement_1 : Prop := ∃ δ, line_l x2 y2 = 0
def statement_2 : Prop := δ = 1 → (b * (x2 - x1) = a * (y2 - y1))
def statement_3 : Prop := δ = -1 → line_l ((x1 + x2) / 2) ((y1 + y2) / 2) = 0

-- Define the proof problem
theorem check_statements : ¬ statement_1 ∧ statement_2 ∧ statement_3 :=
begin
  sorry
end

end check_statements_l780_780883


namespace mike_total_hours_worked_l780_780895

-- Define the conditions
def time_to_wash_car := 10
def time_to_change_oil := 15
def time_to_change_tires := 30

def number_of_cars_washed := 9
def number_of_oil_changes := 6
def number_of_tire_changes := 2

-- Define the conversion factor
def minutes_per_hour := 60

-- Prove that the total time worked equals 4 hours
theorem mike_total_hours_worked : 
  (number_of_cars_washed * time_to_wash_car + 
   number_of_oil_changes * time_to_change_oil + 
   number_of_tire_changes * time_to_change_tires) / minutes_per_hour = 4 := by
  sorry

end mike_total_hours_worked_l780_780895


namespace average_speed_car_l780_780504

theorem average_speed_car (speed_first_hour ground_speed_headwind speed_second_hour : ℝ) (time_first_hour time_second_hour : ℝ) (h1 : speed_first_hour = 90) (h2 : ground_speed_headwind = 10) (h3 : speed_second_hour = 55) (h4 : time_first_hour = 1) (h5 : time_second_hour = 1) : 
(speed_first_hour + ground_speed_headwind) * time_first_hour + speed_second_hour * time_second_hour / (time_first_hour + time_second_hour) = 77.5 :=
sorry

end average_speed_car_l780_780504


namespace intersection_S_T_l780_780315

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780315


namespace sector_area_l780_780861

theorem sector_area (r l : ℝ) (h_r : r = 2) (h_l : l = 3) : (1/2) * l * r = 3 :=
by
  rw [h_r, h_l]
  norm_num

end sector_area_l780_780861


namespace train_travel_distance_l780_780039

variables (t_per_m : ℕ) (m : ℕ) (n : ℕ)
variable h : t_per_m = 2
variable t_per_h : ℕ := 60
variable distance : ℕ

theorem train_travel_distance (h1 : t_per_m = 2) (h2 : t_per_h = 60) (hours_travel : ℕ = 3) :
  distance = 90 :=
by
  -- the proof steps are omitted here
  sorry

end train_travel_distance_l780_780039


namespace sixty_percent_of_total_is_960_l780_780971

-- Definitions from the conditions
def boys : ℕ := 600
def difference : ℕ := 400
def girls : ℕ := boys + difference
def total : ℕ := boys + girls
def sixty_percent_of_total : ℕ := total * 60 / 100

-- The theorem to prove
theorem sixty_percent_of_total_is_960 :
  sixty_percent_of_total = 960 := 
  sorry

end sixty_percent_of_total_is_960_l780_780971


namespace direction_vector_of_line_l780_780821

-- Define the matrix of reflection
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![3/5, -4/5], 
    ![-4/5, -3/5]]

-- Prove that the direction vector of the line l is (2, -1) given the conditions
theorem direction_vector_of_line : 
  ∃ (a b : ℤ), reflection_matrix.mul_vec ![a, b] = ![a, b] ∧ 
    a > 0 ∧ Int.gcd a.nat_abs b.nat_abs = 1 ∧ 
    ![a, b] = ![2, -1] :=
sorry

end direction_vector_of_line_l780_780821


namespace max_product_ge_993_squared_l780_780437

theorem max_product_ge_993_squared (a : Fin 1985 → Fin 1985) (hperm : ∀ n : Fin 1985, ∃ k : Fin 1985, a k = n ∧ ∃ m : Fin 1985, a m = n) :
  ∃ k : Fin 1985, a k * k ≥ 993^2 :=
sorry

end max_product_ge_993_squared_l780_780437


namespace problem_statement_l780_780441

-- Given the conditions and the goal
theorem problem_statement (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz_sum : x + y + z = 1) :
  (2 * x^2 / (y + z)) + (2 * y^2 / (z + x)) + (2 * z^2 / (x + y)) ≥ 1 :=
by
  sorry

end problem_statement_l780_780441


namespace find_first_term_of_sequence_l780_780435

theorem find_first_term_of_sequence
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n+1) = a n + d)
  (h2 : a 0 + a 1 + a 2 = 12)
  (h3 : a 0 * a 1 * a 2 = 48)
  (h4 : ∀ n m, n < m → a n ≤ a m) :
  a 0 = 2 :=
sorry

end find_first_term_of_sequence_l780_780435


namespace polyhedron_U_edge_count_l780_780668

noncomputable theory

variables {S : Type*} 

-- Conditions
variable [convex_polyhedron S]
variable (m : ℕ)
variable [has_vertices S m]
variable (edges_S : ℕ) (h_edges_S : edges_S = 150)

-- Condition that planes intersect according to the vertices
variables (T : ℕ → Type*) [has_planes T] [plane_intersects_vertex S T]

-- The resultant polyhedron U
variables (U : Type*) [convex_polyhedron U]

-- The theorem to prove
theorem polyhedron_U_edge_count
    (h_planes_transform : transforms_into U S T) : 
    (number_of_edges U = 450) :=
sorry

end polyhedron_U_edge_count_l780_780668


namespace intersection_S_T_l780_780186

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l780_780186


namespace remainder_11_pow_2023_mod_7_l780_780619

open Nat

theorem remainder_11_pow_2023_mod_7 :
  (11^2023 % 7) = 4 :=
by
  -- Given conditions
  have h1 : 11 % 7 = 4 := by norm_num,
  have h2 : 4^2 % 7 = 2 := by norm_num,
  have h3 : 4^3 % 7 = 1 := by norm_num,
  sorry

end remainder_11_pow_2023_mod_7_l780_780619


namespace intersection_S_T_eq_T_l780_780253

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780253


namespace ratio_A_B_l780_780747

theorem ratio_A_B :
  let A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
  let B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)
  0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 :=
by
  let A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
  let B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)
  have hA : A = 1400 := rfl
  have hB : B = 1500 := rfl
  have h_ratio : A / B = 14 / 15 := by
    rw [hA, hB]
    norm_num
  exact And.intro (by norm_num) (by norm_num)

end ratio_A_B_l780_780747


namespace largest_integer_binom_l780_780585

theorem largest_integer_binom (n : ℕ) (h : nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) : n = 6 :=
sorry

end largest_integer_binom_l780_780585


namespace intersection_S_T_eq_T_l780_780257

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780257


namespace recurring_decimal_sum_l780_780095

theorem recurring_decimal_sum :
  (∀ (a b : ℚ), a = 2 / 9 → b = 4 / 99 → a + b = 26 / 99) :=
by {
  intros a b ha hb,
  rw [ha, hb],
  norm_num,
  exact rat.add_def 2 4 9 99 dec_trivial dec_trivial -- ensures proper fraction addition
}

end recurring_decimal_sum_l780_780095


namespace proof_problem_l780_780822

-- Definitions for the propositions p and q
def p : Prop := ∀ x, x ≠ 0 → y = 1/x → (y is decreasing)
def q : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

-- The statement to prove
theorem proof_problem : p ∨ q :=
by
  have p_false : ¬p := sorry
  have q_true : q := sorry
  exact Or.inr q_true

end proof_problem_l780_780822


namespace northton_time_capsule_depth_l780_780475

def southton_depth : ℕ := 15

def northton_depth : ℕ := 4 * southton_depth + 12

theorem northton_time_capsule_depth : northton_depth = 72 := by
  sorry

end northton_time_capsule_depth_l780_780475


namespace max_food_cost_excluding_drinks_l780_780659

theorem max_food_cost_excluding_drinks :
  ∃ T F D : ℝ,
    (T = 75 / 1.22) ∧
    (D = 0.20 * T) ∧
    (F = T - D) ∧
    (F ≈ 49.18) :=
by
  have T := 75 / 1.22
  have D := 0.20 * T
  have F := T - D
  use [T, F, D]
  split
  . rfl
  . rfl
  . rfl
  . exact (by norm_num : F ≈ 49.18)

end max_food_cost_excluding_drinks_l780_780659


namespace base_of_1987_with_digit_sum_25_l780_780754

theorem base_of_1987_with_digit_sum_25 (b a c : ℕ) (h₀ : a * b^2 + b * b + c = 1987)
(h₁ : a + b + c = 25) (h₂ : 1 ≤ b ∧ b ≤ 45) : b = 19 :=
sorry

end base_of_1987_with_digit_sum_25_l780_780754


namespace cards_probability_l780_780373

-- Definitions based on conditions
def total_cards := 52
def suits := 4
def cards_per_suit := 13

-- Introducing probabilities for the conditions mentioned
def prob_first := 1
def prob_second := 39 / 52
def prob_third := 26 / 52
def prob_fourth := 13 / 52
def prob_fifth := 26 / 52

-- The problem statement
theorem cards_probability :
  (prob_first * prob_second * prob_third * prob_fourth * prob_fifth) = (3 / 64) :=
by
  sorry

end cards_probability_l780_780373


namespace ratio_of_circle_areas_l780_780824

-- Definitions of the problem
variables {r R : ℝ} -- radii of the circles
variable (r_lt_R : r < R) -- condition that r is less than R

-- Additional condition from the problem
variable (arc_ratio : 1 / 6 * 2 * Real.pi = Real.pi / 3) -- derived from the arc ratio 1:5

-- Main theorem to be proved
theorem ratio_of_circle_areas (r R : ℝ) (r_lt_R : r < R) (arc_ratio : 1 / 6 * 2 * Real.pi = Real.pi / 3) :
  (r / R) = Real.sqrt 3 / 2 →
  (r^2 / R^2) = 3 / 4 :=
begin
  sorry
end

end ratio_of_circle_areas_l780_780824


namespace equilateral_triangle_l780_780348

open Classical

def parabola_C (x y : ℝ) : Prop := y^2 = 4 * x

def focus_F : ℝ × ℝ := (1, 0)

def M_on_parabola_C (x : ℝ) : ℝ × ℝ := (x, 2 * real.sqrt x)

def Q_point (x : ℝ) : ℝ × ℝ := (-1, 2 * real.sqrt x)

def MQ_vector (x : ℝ) : ℝ × ℝ := (x + 1, 0)
def FQ_vector (x : ℝ) : ℝ × ℝ := (-2, 2 * real.sqrt x)
def MF_vector (x : ℝ) : ℝ × ℝ := (x - 1, 2 * real.sqrt x)

variable (x : ℝ)

theorem equilateral_triangle 
    (h_parabola : parabola_C x (2 * real.sqrt x))
    (h_x_pos : 0 < x) 
    (h_equilateral : (MQ_vector x).fst^2 + (MQ_vector x).snd^2 = 
                     (FQ_vector x).fst^2 + (FQ_vector x).snd^2 ∧ 
                     (MQ_vector x).fst^2 + (MQ_vector x).snd^2 = 
                     (MF_vector x).fst^2 + (MF_vector x).snd^2) 
    (h_x_3 : x = 3) :
    (FQ_vector x).fst * (MF_vector x).fst + (FQ_vector x).snd * (MF_vector x).snd = 8 :=
sorry

end equilateral_triangle_l780_780348


namespace number_of_solutions_eq_32_l780_780084

theorem number_of_solutions_eq_32 :
  ∃ s : Finset (Fin 9 × Fin 9 × Fin 9 × Fin 9),
    (∀ t ∈ s, let ⟨a, b, c, d⟩ := t in a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
      (2 * a - 1) * (2 * b - 1) * (2 * c - 1) * (2 * d - 1) = 2 * a * b * c * d - 1) ∧
    (s.card = 32) :=
begin
  sorry
end

end number_of_solutions_eq_32_l780_780084


namespace angelas_average_speed_l780_780062

def average_speed (distance time : ℝ) : ℝ := distance / time

def angelas_trip :=
  let distance1 := 50
  let speed1 := 20
  let distance2 := 25
  let speed2 := 25
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  average_speed total_distance total_time = 21.4286

theorem angelas_average_speed : angelas_trip := by
  sorry
  -- We assume that the average_speed function correctly computes the average speed
  -- given the total distance and total time.

end angelas_average_speed_l780_780062


namespace intersection_eq_T_l780_780161

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780161


namespace volume_tetrahedron_BE_GH_eq_l780_780407

noncomputable def volume_tetrahedron (a b c : ℝ) : ℝ :=
  (volume_of_tetrahedron ABD BE .perpendicular AC)
  * sqrt(3) * b^3/(12 * (a + b) * (b + c))

theorem volume_tetrahedron_BE_GH_eq (a b c : ℝ) :
  (A D = a) → (B E = b) → (C F = c) → (A B = A C = B C = 1) →
  volume_tetrahedron a b c = sqrt(3) * b^3 / (12 * (a + b) * (b + c)) :=
begin
  sorry
end

end volume_tetrahedron_BE_GH_eq_l780_780407


namespace simplify_expression_simplify_and_evaluate_evaluate_expression_l780_780961

theorem simplify_expression (a b : ℝ) : 8 * (a + b) + 6 * (a + b) - 2 * (a + b) = 12 * (a + b) := 
by sorry

theorem simplify_and_evaluate (x y : ℝ) (h : x + y = 1/2) : 
  9 * (x + y)^2 + 3 * (x + y) + 7 * (x + y)^2 - 7 * (x + y) = 2 := 
by sorry

theorem evaluate_expression (x y : ℝ) (h : x^2 - 2 * y = 4) : -3 * x^2 + 6 * y + 2 = -10 := 
by sorry

end simplify_expression_simplify_and_evaluate_evaluate_expression_l780_780961


namespace num_noncongruent_triangles_l780_780463

noncomputable def isosceles_triangle (A B C M N O : Type) : Prop :=
  (M ∈ segment A B) ∧ (N ∈ segment A C) ∧ (O ∈ segment B C) ∧
  (dist A B = dist A C) ∧ (midpoint A B = M) ∧ (midpoint A C = N) ∧ (midpoint B C = O)

theorem num_noncongruent_triangles (A B C M N O : Type) (h : isosceles_triangle A B C M N O) :
  ∃ t : Finset (Triangle A B C M N O), t.card = 9 :=
sorry

end num_noncongruent_triangles_l780_780463


namespace dollar_triple_60_l780_780079

-- Define the function $N
def dollar (N : Real) : Real :=
  0.4 * N + 2

-- Proposition proving that $$(($60)) = 6.96
theorem dollar_triple_60 : dollar (dollar (dollar 60)) = 6.96 := by
  sorry

end dollar_triple_60_l780_780079


namespace parallel_lines_implies_value_of_a_l780_780380

theorem parallel_lines_implies_value_of_a (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y = 0 ∧ x + (a-1)*y + (a^2-1) = 0 → 
  (- a / 2) = - (1 / (a-1))) → a = 2 :=
sorry

end parallel_lines_implies_value_of_a_l780_780380


namespace abs_diff_xy_l780_780964

/-- Given two distinct positive integers x and y,
the arithmetic mean of x and y is a three-digit integer,
the geometric mean of x and y is obtained by reversing the digits of the arithmetic mean.
Prove that |x - y| = 66 * sqrt(1111). -/
theorem abs_diff_xy (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0)
  (h4 : let am := (x + y)/2 in am >= 100 ∧ am < 1000)
  (h5 : let gm := ((x * y).sqrt) in gm = reverse_digits ((x + y) / 2)) :
  |x - y| = 66 * real.sqrt 1111 :=
sorry

/-- Function to reverse the digits of a three-digit number n -/
def reverse_digits (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  in c * 100 + b * 10 + a

#eval reverse_digits 321  -- Example to demonstrate reverse_digits function.
-- Expected output: 123

end abs_diff_xy_l780_780964


namespace intersection_eq_T_l780_780240

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780240


namespace terry_mary_same_color_config_l780_780041

def probability_same_color_config : ℚ :=
  -- Conditions
  let total_candies := 40
  let red_candies := 15
  let blue_candies := 15
  let green_candies := 10

  -- Answer calculated as given in the solution
  (⟨579, 8686⟩ : ℚ)

theorem terry_mary_same_color_config :
  probability_same_color_config = (⟨579, 8686⟩ : ℚ) :=
sorry

end terry_mary_same_color_config_l780_780041


namespace correct_option_l780_780633

theorem correct_option :
  (2 * Real.sqrt 5) + (3 * Real.sqrt 5) = 5 * Real.sqrt 5 :=
by sorry

end correct_option_l780_780633


namespace non_neg_ints_less_than_pi_l780_780986

-- Define the condition: non-negative integers with absolute value less than π
def condition (x : ℕ) : Prop := |(x : ℝ)| < Real.pi

-- Prove that the set satisfying the condition is {0, 1, 2, 3}
theorem non_neg_ints_less_than_pi :
  {x : ℕ | condition x} = {0, 1, 2, 3} := by
  sorry

end non_neg_ints_less_than_pi_l780_780986


namespace intersection_S_T_eq_T_l780_780220

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780220


namespace rolls_combinations_l780_780029

theorem rolls_combinations : ∀ (kinds total : ℕ), kinds = 3 → total = 6 → 
  (∃ (combinations : ℕ), combinations = 10) := 
by
  intros kinds total h_kinds h_total
  use 10
  sorry

end rolls_combinations_l780_780029


namespace james_daily_soda_consumption_l780_780416

theorem james_daily_soda_consumption
  (N_p : ℕ) -- number of packs
  (S_p : ℕ) -- sodas per pack
  (S_i : ℕ) -- initial sodas
  (D : ℕ)  -- days in a week
  (h1 : N_p = 5)
  (h2 : S_p = 12)
  (h3 : S_i = 10)
  (h4 : D = 7) : 
  (N_p * S_p + S_i) / D = 10 := 
by 
  sorry

end james_daily_soda_consumption_l780_780416


namespace rancher_total_animals_l780_780049

theorem rancher_total_animals
  (H C : ℕ) (h1 : C = 5 * H) (h2 : C = 140) :
  C + H = 168 := 
sorry

end rancher_total_animals_l780_780049


namespace correct_calculation_l780_780629

theorem correct_calculation :
  (2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  (5 * Real.sqrt 3 * 5 * Real.sqrt 2 ≠ 5 * Real.sqrt 6) ∧
  (Real.sqrt (4 + 1/2) ≠ 2 * Real.sqrt (1/2)) :=
by
  sorry

end correct_calculation_l780_780629


namespace intersection_eq_T_l780_780241

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780241


namespace math_problem_l780_780847

-- Define the given conditions
variables (a b : ℝ)
hypothesis h1 : 100^a = 4
hypothesis h2 : 100^b = 10

-- State the main proof problem
theorem math_problem :
  25^((2 - 2 * a - b) / (3 * (1 - b))) = 3.968 :=
by
  sorry

end math_problem_l780_780847


namespace average_age_of_dance_group_l780_780965

theorem average_age_of_dance_group (S_f S_m : ℕ) (avg_females avg_males : ℕ) 
(hf : avg_females = S_f / 12) (hm : avg_males = S_m / 18) 
(h1 : avg_females = 25) (h2 : avg_males = 40) : 
  (S_f + S_m) / 30 = 34 :=
by
  sorry

end average_age_of_dance_group_l780_780965


namespace largest_n_for_binom_equality_l780_780593

theorem largest_n_for_binom_equality : ∃ (n : ℕ), (∑ i : {i // i = 4 ∨ i = 5}, Nat.choose 10 i) = Nat.choose 11 n ∧ n = 6 :=
by
  sorry

end largest_n_for_binom_equality_l780_780593


namespace probability_even_product_at_least_one_greater_than_four_l780_780108

namespace DieProbability

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_greater_than_four (n : ℕ) : Prop := n > 4

theorem probability_even_product_at_least_one_greater_than_four :
  (count (λ (x y : ℕ), x ∈ list.range 1 9 ∧ y ∈ list.range 1 9 
    ∧ is_even (x * y) 
    ∧ (x > 4 ∨ y > 4))) / 64 = 3 / 16 := sorry

end DieProbability

end probability_even_product_at_least_one_greater_than_four_l780_780108


namespace sixty_percent_of_total_is_960_l780_780970

variable (number_of_boys number_of_girls total_participants : ℕ)

-- Condition 1: The difference between the number of boys and girls is 400.
def difference_condition : Prop := number_of_girls - number_of_boys = 400

-- Condition 2: There are 600 boys.
def boys_condition : Prop := number_of_boys = 600

-- Condition 3: The number of girls is more than the number of boys.
def girls_condition : Prop := number_of_girls > number_of_boys

-- Given conditions
axiom difference_condition_h : difference_condition number_of_boys number_of_girls
axiom boys_condition_h : boys_condition number_of_boys
axiom girls_condition_h : girls_condition number_of_boys number_of_girls

-- Total number of participants
def total_participants : ℕ := number_of_boys + number_of_girls

theorem sixty_percent_of_total_is_960 :
  0.6 * (number_of_boys + number_of_girls) = 960 :=
by 
  sorry

end sixty_percent_of_total_is_960_l780_780970


namespace intersection_S_T_l780_780313

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780313


namespace intersection_of_S_and_T_l780_780231

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780231


namespace purchase_in_april_l780_780984

namespace FamilySavings

def monthly_income : ℕ := 150000
def monthly_expenses : ℕ := 115000
def initial_savings : ℕ := 45000
def furniture_cost : ℕ := 127000

noncomputable def monthly_savings : ℕ := monthly_income - monthly_expenses
noncomputable def additional_amount_needed : ℕ := furniture_cost - initial_savings
noncomputable def months_required : ℕ := (additional_amount_needed + monthly_savings - 1) / monthly_savings  -- ceiling division

theorem purchase_in_april : months_required = 3 :=
by
  -- Proof goes here
  sorry

end FamilySavings

end purchase_in_april_l780_780984


namespace largest_integer_comb_l780_780604

theorem largest_integer_comb (n: ℕ) (h1 : n = 6) : 
  (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) :=
by
  rw [h1]
  exact sorry

end largest_integer_comb_l780_780604


namespace largest_integer_binom_l780_780617

theorem largest_integer_binom
  (n : ℕ)
  (h₁ : binom 10 4 + binom 10 5 = binom 11 n) :
  n = 6 :=
sorry

end largest_integer_binom_l780_780617


namespace figure_100_squares_l780_780750

def f (n : ℕ) : ℕ := n^3 + 2 * n^2 + 2 * n + 1

theorem figure_100_squares : f 100 = 1020201 :=
by
  -- The proof will go here
  sorry

end figure_100_squares_l780_780750


namespace number_partition_l780_780466

theorem number_partition (A B : Finset ℕ) (h_partition : ∀ x ∈ Finset.range 3 (3^5+1), x ∈ A ∨ x ∈ B) 
(h_disjoint : A ∩ B = ∅) :
∃ a b c ∈ (A ∪ B), (a * b = c ∧ ((a ∈ A ∧ b ∈ A ∧ c ∈ A) ∨ (a ∈ B ∧ b ∈ B ∧ c ∈ B))) := sorry

end number_partition_l780_780466


namespace min_abs_sum_value_l780_780622

theorem min_abs_sum_value : ∃ x : ℝ, x = -4 ∧ ∀ z : ℝ, ∑ i in [2, 4, 5], abs (z + i) ≥ 3 :=
by
  sorry

end min_abs_sum_value_l780_780622


namespace chord_length_l780_780126

noncomputable def length_of_chord_AB : ℝ :=
  4

theorem chord_length (
  (x y : ℝ) 
  (h_circ : x^2 + (y + 1)^2 = 5) 
  (h_line : -sqrt 3 * x - y + 1 = 0)) :
  ∃ A B : (ℝ × ℝ), 
  (A ≠ B ∧ (A, B belong to) on_circle_C x y h_circ ∧ 
    (A, B lie on line ) h_line) →
  dist A B = length_of_chord_AB :=
sorry

end chord_length_l780_780126


namespace drums_of_grapes_per_day_l780_780827

-- Definitions derived from conditions
def pickers := 235
def raspberry_drums_per_day := 100
def total_days := 77
def total_drums := 17017

-- Prove the main theorem
theorem drums_of_grapes_per_day : (total_drums - total_days * raspberry_drums_per_day) / total_days = 121 := by
  sorry

end drums_of_grapes_per_day_l780_780827


namespace time_to_pass_bridge_l780_780639

noncomputable def train_length : Real := 357
noncomputable def speed_km_per_hour : Real := 42
noncomputable def bridge_length : Real := 137

noncomputable def speed_m_per_s : Real := speed_km_per_hour * (1000 / 3600)

noncomputable def total_distance : Real := train_length + bridge_length

noncomputable def time_to_pass : Real := total_distance / speed_m_per_s

theorem time_to_pass_bridge : abs (time_to_pass - 42.33) < 0.01 :=
sorry

end time_to_pass_bridge_l780_780639


namespace odd_function_decreasing_on_interval_l780_780125

theorem odd_function_decreasing_on_interval
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_decreasing : ∀ {x₁ x₂}, 1 ≤ x₁ → x₁ ≤ 4 → 1 ≤ x₂ → x₂ ≤ 4 → x₁ < x₂ → f x₁ > f x₂) :
  ∀ {x₁ x₂}, -4 ≤ x₁ → x₁ ≤ -1 → -4 ≤ x₂ → x₂ ≤ -1 → x₁ < x₂ → f x₁ > f x₂ :=
begin
  sorry
end

end odd_function_decreasing_on_interval_l780_780125


namespace intersection_S_T_eq_T_l780_780197

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780197


namespace nm_odd_if_squares_sum_odd_l780_780846

theorem nm_odd_if_squares_sum_odd
  (n m : ℤ)
  (h : (n^2 + m^2) % 2 = 1) :
  (n * m) % 2 = 1 :=
sorry

end nm_odd_if_squares_sum_odd_l780_780846


namespace charge_difference_percentage_l780_780484

variable (R G P : ℝ)

-- Conditions
def condition1 : Prop := P = R - 0.50 * R
def condition2 : Prop := P = G - 0.10 * G

-- Conclusion to prove
def percentage_increase : Prop := (R - G) / G * 100 = 80

theorem charge_difference_percentage
  (h1 : condition1 R G P)
  (h2 : condition2 R G P) :
  percentage_increase R G :=
  by sorry

end charge_difference_percentage_l780_780484


namespace intersection_S_T_l780_780317

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l780_780317


namespace find_AB_length_l780_780886

-- Let h be the height of the trapezoid.
variables {h : ℝ}

-- Each condition in a) should be used as a definition in Lean 4.
def area_ratio (AB CD : ℝ) (h : ℝ) : Prop := (AB * h / 2) / (CD * h / 2) = 4 / 1
def sum_of_segments (AB CD : ℝ) : Prop := AB + CD = 250

-- State the theorem needing proof
theorem find_AB_length (AB CD : ℝ) (h : ℝ) (area_ratio (AB CD : ℝ) (h : ℝ)) (sum_of_segments (AB CD : ℝ)) : AB = 200 :=
sorry

end find_AB_length_l780_780886


namespace inequality_with_means_l780_780934

theorem inequality_with_means (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := 
sorry

end inequality_with_means_l780_780934


namespace find_sum_l780_780996

variable (P T R : ℝ)

-- Given conditions
def simple_interest_condition :=
  (P * T * R) / 100 = 85

def true_discount_condition :=
  (85 * 100) / (100 + (R * T)) = 75

-- Proof statement
theorem find_sum (h₁ : simple_interest_condition P T R) (h₂ : true_discount_condition P T R) :
  P = 637.5 := sorry

end find_sum_l780_780996


namespace intersection_S_T_eq_T_l780_780135

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780135


namespace concyclic_iff_angles_and_ellipse_l780_780064

noncomputable theory

open Real

variable {a b : ℝ} (h_ellipse : ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1))
variable (A B C D : ℝ×ℝ) -- coordinates of points A, B, C, and D
variable (S : ℝ×ℝ) -- coordinates of the intersection point S
variable (h_angle : ∃ θ, ∃ t u, -- there exists an angle θ such that
  ((A.1 = S.1 + t * cos θ ∧ A.2 = S.2 + t * sin θ) ∧ -- A on line with direction θ
  (B.1 = S.1 - t * cos θ ∧ B.2 = S.2 - t * sin θ) ∧ -- B on opposite side of S
  (C.1 = S.1 + u * cos (π - θ) ∧ C.2 = S.2 + u * sin (π - θ)) ∧ -- C on line with direction π - θ
  (D.1 = S.1 - u * cos (π - θ) ∧ D.2 = S.2 - u * sin (π - θ)))) -- D on opposite side of S

theorem concyclic_iff_angles_and_ellipse :
  (A.1^2 / a^2 + A.2^2 / b^2 = 1 ∧ B.1^2 / a^2 + B.2^2 / b^2 = 1 ∧
   C.1^2 / a^2 + C.2^2 / b^2 = 1 ∧ D.1^2 / a^2 + D.2^2 / b^2 = 1 ∧
   (∃ θ, ∃ t u,
    (A.1 = S.1 + t * cos θ ∧ A.2 = S.2 + t * sin θ) ∧
    (B.1 = S.1 - t * cos θ ∧ B.2 = S.2 - t * sin θ) ∧
    (C.1 = S.1 + u * cos (π - θ) ∧ C.2 = S.2 + u * sin (π - θ)) ∧
    (D.1 = S.1 - u * cos (π - θ) ∧ D.2 = S.2 - u * sin (π - θ)))) ↔
  (dist S A * dist S B = dist S C * dist S D) := sorry

end concyclic_iff_angles_and_ellipse_l780_780064


namespace f_range_l780_780932

def P : Set ℕ := { n | n ≥ 3 }

def f (n : ℕ) (h : n ∈ P) : ℕ := Nat.find (λ d => d > 0 ∧ d ∣ n ∧ d ≠ n ∧ ∀ m, 0 < m ∧ m < d → ¬(m ∣ n))

theorem f_range :
  ∃ (M : Set ℕ), (∀ n ∈ P, f n _ ∈ M) ∧ 
  (∀ q ∈ M, ∃ n ∈ P, f n _ = q) ∧
  (∀ q ∈ M, ∃ (p : ℕ) (h_prime : Nat.Prime p) (α : ℕ), q = p ^ α) :=
begin
  sorry
end

end f_range_l780_780932


namespace intersection_S_T_eq_T_l780_780138

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780138


namespace even_minutes_l780_780892

def even_number_of_minutes (n : ℕ) : Prop := n % 2 = 0

def valid_grid_operation (grid : Array (Array ℤ)) (i j : ℕ) : Prop :=
  (i < 5 ∧ j < 4 ∧ grid[i] != grid[i][j + 1]) ∨ (i < 4 ∧ j < 5 ∧ grid[i][j] != grid[i + 1][j])

def initial_grid : Array (Array ℤ) := Array.mkArray 5 (Array.mkArray 5 0)

def final_state (grid : Array (Array ℤ)) : Prop :=
  ∀ (i j : ℕ), (i < 5 ∧ j < 5) → (grid[i][0] = grid[j][0]) ∧ (grid[0][i] = grid[0][j])

theorem even_minutes (n : ℕ) (grid : Array (Array ℤ)) :
  (∀ (m : ℕ), m < n → valid_grid_operation grid) →
  final_state grid →
  even_number_of_minutes n :=
by
  sorry

end even_minutes_l780_780892


namespace area_of_quadrilateral_AEDC_l780_780888

-- Definitions based on the problem conditions
variables (ABC AEDC : Type) [triangle ABC]
variables (AD CE : median) (P : point) (PE PD DE : real)
variable AEDC_area : real

-- Conditions
axiom AD_CE_intersect_P : AD.intersect P ∧ CE.intersect P
axiom PE_val : PE = 2
axiom PD_val : PD = 3
axiom DE_val : DE = 3.5

-- Proof statement
theorem area_of_quadrilateral_AEDC : AEDC_area = 27 := sorry

end area_of_quadrilateral_AEDC_l780_780888


namespace largest_n_binomial_l780_780578

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l780_780578


namespace factor_210_into_4_factors_l780_780393

theorem factor_210_into_4_factors : ∃ n : ℕ, n = 15 ∧
  ∃ (a b c d : ℕ), a * b * c * d = 210 ∧ (mul_comm a b ∧ mul_comm a c ∧ mul_comm a d ∧ mul_comm b c ∧ mul_comm b d ∧ mul_comm c d) :=
by 
  sorry

end factor_210_into_4_factors_l780_780393


namespace intersection_of_S_and_T_l780_780226

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780226


namespace compare_ab_1_to_a_b_two_pow_b_eq_one_div_b_sign_of_expression_l780_780092

-- Part 1
theorem compare_ab_1_to_a_b {a b : ℝ} (h1 : a^b * b^a + Real.log b / Real.log a = 0) (ha : a > 0) (hb : b > 0) : ab + 1 < a + b := sorry

-- Part 2
theorem two_pow_b_eq_one_div_b {b : ℝ} (h : 2^b * b^2 + Real.log b / Real.log 2 = 0) : 2^b = 1 / b := sorry

-- Part 3
theorem sign_of_expression {b : ℝ} (h : 2^b * b^2 + Real.log b / Real.log 2 = 0) : (2 * b + 1 - Real.sqrt 5) * (3 * b - 2) < 0 := sorry

end compare_ab_1_to_a_b_two_pow_b_eq_one_div_b_sign_of_expression_l780_780092


namespace intersection_eq_T_l780_780285

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780285


namespace convex_ngon_diagonals_inequality_l780_780928

theorem convex_ngon_diagonals_inequality 
  (n : ℕ) (d p : ℝ) 
  (h_d : d = sum_of_diagonals_of_convex_ngon n) 
  (h_p : p = perimeter_of_convex_ngon n) :
  n - 3 < (2 * d) / p ∧ (2 * d) / p < (⌊n / 2⌋) * (⌊(n + 1) / 2⌋) - 2 := 
sorry

end convex_ngon_diagonals_inequality_l780_780928


namespace overall_loss_percentage_l780_780461

-- Definitions for initial conditions
def cost_price_A : ℝ := 1
def num_A : ℕ := 30
def total_cost_A : ℝ := cost_price_A * num_A

def cost_price_B : ℝ := 2
def num_B : ℕ := 40
def total_cost_B : ℝ := cost_price_B * num_B

def cost_price_C : ℝ := 3
def num_C : ℕ := 10
def total_cost_C : ℝ := cost_price_C * num_C

def total_cost : ℝ := total_cost_A + total_cost_B + total_cost_C

def selling_price_A : ℝ := cost_price_A - 0.5
def total_selling_A : ℝ := selling_price_A * num_A

def selling_price_B : ℝ := cost_price_B - 1
def total_selling_B : ℝ := selling_price_B * num_B

def selling_price_C : ℝ := cost_price_C - 1.5
def total_selling_C : ℝ := selling_price_C * num_C

def total_selling : ℝ := total_selling_A + total_selling_B + total_selling_C

def loss_due_to_A : ℝ := total_selling_A

def total_loss : ℝ := total_cost - total_selling + loss_due_to_A

def loss_percentage : ℝ := (total_loss / total_cost) * 100

-- The theorem to prove
theorem overall_loss_percentage : 
  loss_percentage = 60.71 := 
sorry

end overall_loss_percentage_l780_780461


namespace platform_length_l780_780020

theorem platform_length
  (train_length : ℕ)
  (pole_time : ℕ)
  (platform_time : ℕ)
  (h1 : train_length = 300)
  (h2 : pole_time = 18)
  (h3 : platform_time = 39) :
  let speed := train_length / pole_time in
  let total_distance := speed * platform_time in
  total_distance - train_length = 350 :=
by 
  sorry

end platform_length_l780_780020


namespace roxy_bought_flowering_plants_l780_780470

-- Definitions based on conditions
def initial_flowering_plants : ℕ := 7
def initial_fruiting_plants : ℕ := 2 * initial_flowering_plants
def plants_after_saturday (F : ℕ) : ℕ := initial_flowering_plants + F + initial_fruiting_plants + 2
def plants_after_sunday (F : ℕ) : ℕ := (initial_flowering_plants + F - 1) + (initial_fruiting_plants + 2 - 4)
def final_plants_in_garden : ℕ := 21

-- The proof statement
theorem roxy_bought_flowering_plants (F : ℕ) :
  plants_after_sunday F = final_plants_in_garden → F = 3 := 
sorry

end roxy_bought_flowering_plants_l780_780470


namespace sin_double_angle_shift_l780_780110

variable (θ : Real)

theorem sin_double_angle_shift (h : Real.cos (θ + Real.pi) = -1 / 3) :
  Real.sin (2 * θ + Real.pi / 2) = -7 / 9 := 
by 
  sorry

end sin_double_angle_shift_l780_780110


namespace intersection_eq_T_l780_780248

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780248


namespace sequence_properties_l780_780884

def sequence : ℕ → ℤ
| 0       := 0  -- Typically, arithmetic sequences are indexed from 1, not 0
| (n + 1) := -2 * (n + 1) + 15

def sum_sequence (n : ℕ) : ℤ :=
  n * (13 - (n - 1))

theorem sequence_properties :
  (sequence 1 = 13) ∧ 
  (3 * sequence 2 = 11 * sequence 6) ∧ 
  (∀ n, sequence n = -2 * n + 15) ∧ 
  (∀ n, sum_sequence n = -n^2 + 14 * n) :=
begin
  split,
  { sorry },  -- Prove sequence 1 = 13
  split,
  { sorry },  -- Prove 3 * sequence 2 = 11 * sequence 6
  split,
  { intros n,
    sorry },  -- Prove ∀ n, sequence n = -2 * n + 15
  { intros n,
    sorry },  -- Prove ∀ n, sum_sequence n = -n^2 + 14 * n
end

end sequence_properties_l780_780884


namespace family_can_purchase_furniture_in_april_l780_780983

noncomputable def monthly_income : ℤ := 150000
noncomputable def monthly_expenses : ℤ := 115000
noncomputable def initial_savings : ℤ := 45000
noncomputable def furniture_cost : ℤ := 127000

theorem family_can_purchase_furniture_in_april : 
  ∃ (months : ℕ), months = 3 ∧ 
  (initial_savings + months * (monthly_income - monthly_expenses) >= furniture_cost) :=
by
  -- proof will be written here
  sorry

end family_can_purchase_furniture_in_april_l780_780983


namespace intersection_S_T_eq_T_l780_780221

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780221


namespace correct_statement_is_B_l780_780634

-- Defining the problem statements as Lean propositions
def StatementA : Prop :=
  ∀ (A B C D : Point), ∃ (plane : Plane), A ∈ plane ∧ B ∈ plane ∧ C ∈ plane ∧ D ∈ plane

def StatementB : Prop :=
  ∀ (m : Line) (α : Plane), (m ⟂ α) → ∀ (l : Line), l ∈ α → (m ⟂ l)

def StatementC : Prop :=
  ∀ (m : Line) (α : Plane), (m ∥ α) → ∀ (l : Line), l ∈ α → (m ∥ l)

def StatementD : Prop :=
  ∀ (m : Line) (α : Plane), (∃^∞ p, p ∈ m ∧ p ∉ α) → (m ∥ α)

-- The main theorem stating that Statement B is the only correct one
theorem correct_statement_is_B : 
  ¬StatementA ∧ StatementB ∧ ¬StatementC ∧ ¬StatementD :=
by
  -- Proof is omitted but should go here
  sorry

end correct_statement_is_B_l780_780634


namespace intersection_S_T_eq_T_l780_780134

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780134


namespace intersection_S_T_eq_T_l780_780264

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780264


namespace jonah_total_raisins_l780_780911

-- Define the amounts of yellow and black raisins added
def yellow_raisins : ℝ := 0.3
def black_raisins : ℝ := 0.4

-- The main statement to be proved
theorem jonah_total_raisins : yellow_raisins + black_raisins = 0.7 :=
by 
  sorry

end jonah_total_raisins_l780_780911


namespace unit_vector_a_is_correct_l780_780825

def vector (T : Type) := (T × T)
def a : vector ℝ := (2, 1)
def b : vector ℝ := (-3, 1)
def unit_vector (v : vector ℝ) : vector ℝ := 
    let magnitude := Math.sqrt (v.1 * v.1 + v.2 * v.2)
    in (v.1 / magnitude, v.2 / magnitude)

theorem unit_vector_a_is_correct : unit_vector a = (2 * Math.sqrt 5 / 5, Math.sqrt 5 / 5) :=
sorry

end unit_vector_a_is_correct_l780_780825


namespace sqrt_product_simplification_l780_780067

theorem sqrt_product_simplification (p : ℝ) :
  (sqrt (40 * p^2) * sqrt (10 * p^3) * sqrt (8 * p^2) = 40 * p^3 * sqrt p) := 
by
  sorry

end sqrt_product_simplification_l780_780067


namespace Karen_packs_piece_of_cake_days_l780_780912

theorem Karen_packs_piece_of_cake_days 
(Total Ham_Days : ℕ) (Ham_probability Cake_probability : ℝ) 
  (H_Total : Total = 5) 
  (H_Ham_Days : Ham_Days = 3) 
  (H_Ham_probability : Ham_probability = (3 / 5)) 
  (H_Cake_probability : Ham_probability * (Cake_probability / 5) = 0.12) : 
  Cake_probability = 1 := 
by
  sorry

end Karen_packs_piece_of_cake_days_l780_780912


namespace intersection_S_T_eq_T_l780_780276

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780276


namespace sum_of_sequences_l780_780450

def a_n (n : ℕ) : ℕ := n + 1
def b_n (n : ℕ) : ℕ := 2^(n-1)
def a_b_n (n : ℕ) : ℕ := a_n (b_n n)

theorem sum_of_sequences : (∑ k in Finset.range 10, a_b_n (k + 1)) = 1033 := by
  sorry

end sum_of_sequences_l780_780450


namespace connected_points_n_l780_780645

-- Let n be a natural number
variable {n : ℕ}

-- Definition for the number of ways to connect n points following the given rules.
def A (n : ℕ) : ℚ := 1 / (n + 1 : ℚ) * binomial (2 * n) n

-- Theorem stating the required result.
theorem connected_points_n {n : ℕ} : 
  let A_n := 1 / (n + 1 : ℚ) * binomial (2 * n) n in
  -- The number of different ways to connect these n + 1 points is A_n
  (A n) = A_n := 
sorry -- Skip the proof.

end connected_points_n_l780_780645


namespace president_vice_president_ways_l780_780456

theorem president_vice_president_ways :
  let boys := 14
  let girls := 10
  let total_boys_ways := boys * (boys - 1)
  let total_girls_ways := girls * (girls - 1)
  total_boys_ways + total_girls_ways = 272 := 
by
  sorry

end president_vice_president_ways_l780_780456


namespace largest_n_binom_equality_l780_780521

theorem largest_n_binom_equality :
  ∃ n : ℕ, (binomial 10 4 + binomial 10 5 = binomial 11 n) ∧ ∀ m : ℕ, (binomial 11 m = binomial 11 5) → m ≤ 6 :=
by
  sorry

end largest_n_binom_equality_l780_780521


namespace min_AP_BP_l780_780431

noncomputable def A : ℝ × ℝ := (1, 0)
noncomputable def B : ℝ × ℝ := (6, 5)

def on_parabola (P : ℝ × ℝ) : Prop :=
  P.2 ^ 2 = 8 * P.1

def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

def AP (P : ℝ × ℝ) : ℝ :=
  dist A P

def BP (P : ℝ × ℝ) : ℝ :=
  dist B P

theorem min_AP_BP :
  ∀ P : ℝ × ℝ, on_parabola P → AP P + BP P ≥ 8 :=
by 
  sorry

end min_AP_BP_l780_780431


namespace number_of_morse_code_symbols_l780_780384

-- Define the number of sequences for different lengths
def sequences_of_length (n : Nat) : Nat :=
  2 ^ n

theorem number_of_morse_code_symbols : 
  (sequences_of_length 1) + (sequences_of_length 2) + (sequences_of_length 3) + (sequences_of_length 4) + (sequences_of_length 5) = 62 := by
  sorry

end number_of_morse_code_symbols_l780_780384


namespace find_initial_volume_l780_780035

noncomputable def initial_volume_of_solution (V : ℝ) : Prop :=
  let initial_jasmine := 0.05 * V
  let added_jasmine := 8
  let added_water := 2
  let new_total_volume := V + added_jasmine + added_water
  let new_jasmine := 0.125 * new_total_volume
  initial_jasmine + added_jasmine = new_jasmine

theorem find_initial_volume : ∃ V : ℝ, initial_volume_of_solution V ∧ V = 90 :=
by
  use 90
  unfold initial_volume_of_solution
  sorry

end find_initial_volume_l780_780035


namespace intersection_of_S_and_T_l780_780238

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780238


namespace cheese_division_non_rational_cheese_division_rational_l780_780646

-- Part (a)
theorem cheese_division_non_rational (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ n : ℕ, (iterate (λ x, (x * (a / (1 + a)))) n 1 = 1 / 2) :=
by sorry

-- Part (b)
theorem cheese_division_rational (a : ℚ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ¬ ∃ n : ℕ, (iterate (λ x, (x * (a / (1 + a)))) n 1 = 1 / 2) :=
by sorry

end cheese_division_non_rational_cheese_division_rational_l780_780646


namespace number_of_pumpkins_l780_780452

theorem number_of_pumpkins (w1 w2 total : ℝ) (h1 : w1 = 4) (h2 : w2 = 8.7) (h3 : w1 + w2 = total) :
  total = 12.7 → w1 = 4 → w2 = 8.7 → 2 = (list.length [w1, w2]) :=
by
  sorry

end number_of_pumpkins_l780_780452


namespace astroid_circle_intersection_radius_l780_780805

theorem astroid_circle_intersection_radius :
  ∃ (r : ℝ), (r = Real.sqrt (2 / 5)) ∧
  (∀ (x y : ℝ), (x ^ (2 / 3) + y ^ (2 / 3) = 1) → (x ^ 2 + y ^ 2 = r ^ 2)) :=
sorry

end astroid_circle_intersection_radius_l780_780805


namespace joe_average_test_score_l780_780422

theorem joe_average_test_score 
  (A B C : ℕ) 
  (Hsum : A + B + C = 135) 
  : (A + B + C + 25) / 4 = 40 :=
by
  sorry

end joe_average_test_score_l780_780422


namespace lines_either_concurrent_or_parallel_l780_780937

open EuclideanGeometry

-- Let A be a point outside the given circle
variable (A : Point)

-- Define the circle and points M, N where tangents from A touch the circle
variable (circle : Circle)
variable (M N : Point)
variable (AM_tangent : Tangent A circle M)
variable (AN_tangent : Tangent A circle N)

-- Define the secants intersecting the circle at P, Q on the first and K, L on the second
variable (P Q K L : Point)
variable (secant1 : Secant A circle P Q)
variable (secant2 : Secant A circle K L)

-- Now we state the theorem
theorem lines_either_concurrent_or_parallel :
  lines_concurrent_or_parallel (line_through P K) (line_through Q L) (line_through M N) :=
sorry

end lines_either_concurrent_or_parallel_l780_780937


namespace intersection_of_S_and_T_l780_780228

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780228


namespace fraction_meaningful_domain_l780_780857

theorem fraction_meaningful_domain (x : ℝ) : (∃ f : ℝ, f = 1 / (x - 2)) → x ≠ 2 :=
by
  sorry

end fraction_meaningful_domain_l780_780857


namespace log_relation_l780_780839

theorem log_relation : (a b : ℝ) (h1 : a = Real.log 256 / Real.log 4) (h2 : b = Real.log 16 / Real.log 2) : a = b :=
by
  sorry

end log_relation_l780_780839


namespace ratio_triangle_square_l780_780398

-- Define the square and its properties
def square_area (s : ℝ) : ℝ := s * s

-- Define the triangle PXZ and calculate its area
def triangle_area (s : ℝ) : ℝ :=
  let P : (ℝ × ℝ) := (0, 0)
  let X : (ℝ × ℝ) := (s / 2, 0)
  let Z : (ℝ × ℝ) := (s / 2, s / 3)
  0.5 * abs (s / 2 * s / 3)

-- Main theorem: ratio of the areas
theorem ratio_triangle_square (s : ℝ) : triangle_area(s) / square_area(s) = 1 / 12 := 
by
  sorry

end ratio_triangle_square_l780_780398


namespace largest_integer_binom_l780_780609

theorem largest_integer_binom
  (n : ℕ)
  (h₁ : binom 10 4 + binom 10 5 = binom 11 n) :
  n = 6 :=
sorry

end largest_integer_binom_l780_780609


namespace atleast_n_pairwise_disjoint_subsets_in_same_class_l780_780012

variable {α : Type*}
variable (S : Set α)
variable (n : ℕ) (h_pos_n : 0 < n)
variable (h_card_S : S.card = n^2 + n - 1)
variable (partition : {R : Set (Set α) | ∀ (s t ∈ R), s ≠ t → Disjoint s t })

theorem atleast_n_pairwise_disjoint_subsets_in_same_class :
  ∃ (R : Set (Set α)) (H : R ⊆ { t | t ∈ S.powerset_len n }) (HR : ¬Disjoint (t₁ ∈ R) t₂), 
  Set.card R ≥ n :=
begin
  sorry -- proof is omitted
end

end atleast_n_pairwise_disjoint_subsets_in_same_class_l780_780012


namespace james_total_matches_l780_780900

-- Define the conditions
def dozen : Nat := 12
def boxes_per_dozen : Nat := 5
def matches_per_box : Nat := 20

-- Calculate the expected number of matches
def expected_matches : Nat := boxes_per_dozen * dozen * matches_per_box

-- State the theorem to be proved
theorem james_total_matches : expected_matches = 1200 := by
  sorry

end james_total_matches_l780_780900


namespace factorization_of_210_l780_780395

theorem factorization_of_210 :
  {l : List ℕ // l.product = 210 ∧ l.length = 4}.quotient.mk.card = 15 :=
sorry

end factorization_of_210_l780_780395


namespace proof_problem_l780_780851

theorem proof_problem (p q r : ℝ) 
  (h1 : p + q = 20)
  (h2 : p * q = 144) 
  (h3 : q + r = 52) 
  (h4 : 4 * (r + p) = r * p) : 
  r - p = 32 := 
sorry

end proof_problem_l780_780851


namespace Joel_possible_card_value_sum_l780_780423

theorem Joel_possible_card_value_sum :
  ∀ (y : ℝ), 0 < y ∧ y < 90 ∧ sin y ≠ cos y ∧ sin y ≠ tan y ∧ cos y ≠ tan y → 
  (sin y = cos y ∨ sin y = tan y ∨ cos y = tan y) → 
  sum_possibilities(sin y, cos y, tan y) = (1 + Real.sqrt 5) / 2 := by
  sorry

-- A helper function to compute the sum of possibilities, defined appropriately in Lean.
noncomputable def sum_possibilities (a b c : ℝ) : ℝ :=
  if a = cos 45 then a 
  else if b = cos 45 then b 
  else if c = cos 45 then c 
  else a + b + c 

end Joel_possible_card_value_sum_l780_780423


namespace min_value_of_f_cos_2alpha_l780_780339

def f (x : Real) : Real := -cos x ^ 2 - sin x + 1

theorem min_value_of_f : ∀ x : Real, f x ≥ -1 / 4 := 
sorry

theorem cos_2alpha (α : Real) (h : f α = 5 / 16) : cos (2 * α) = 7 / 8 :=
sorry

end min_value_of_f_cos_2alpha_l780_780339


namespace largest_integer_binom_l780_780579

theorem largest_integer_binom (n : ℕ) (h : nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) : n = 6 :=
sorry

end largest_integer_binom_l780_780579


namespace largest_n_for_binomial_equality_l780_780533

theorem largest_n_for_binomial_equality :
  ∃ n : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 n ) ∧
  ∀ m : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 m ) → m ≤ 6 :=
begin
  sorry
end

end largest_n_for_binomial_equality_l780_780533


namespace correct_calculation_l780_780625

theorem correct_calculation : 
  ¬(2 * Real.sqrt 3 + 3 * Real.sqrt 2 = 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  ¬(5 * Real.sqrt 3 * 5 * Real.sqrt 2 = 5 * Real.sqrt 6) ∧
  ¬(Real.sqrt (4 + 1 / 2) = 2 * Real.sqrt (1 / 2)) :=
by {
  -- Using the conditions to prove the correct option B
  sorry
}

end correct_calculation_l780_780625


namespace length_MN_l780_780408

/-- Line equation in parametric form. -/
def line_eq (t : ℝ) : ℝ × ℝ := 
  (1 + t, 2 + t)

/-- Circle equation in polar form. -/
def circle_eq (ρ θ : ℝ) : Prop := 
  ρ^2 + 2 * ρ * sin θ = 3

/-- Function to convert polar coordinates to rectangular coordinates. -/
def polar_to_rect (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

theorem length_MN :
  ∀ (M N : ℝ × ℝ), 
  (∃ t_M t_N : ℝ, line_eq t_M = M ∧ line_eq t_N = N ∧ polar_to_rect (fst M) (snd M) = M ∧ polar_to_rect (fst N) (snd N) = N) → 
  (∃ ρ_M θ_M ρ_N θ_N : ℝ, 
    circle_eq ρ_M θ_M ∧ 
    circle_eq ρ_N θ_N ∧ 
    polar_to_rect ρ_M θ_M = M ∧ 
    polar_to_rect ρ_N θ_N = N) → 
  dist M N = 2 * real.sqrt 2 :=
by
  sorry

end length_MN_l780_780408


namespace error_percent_is_correct_l780_780000

-- Define the actual lengths L and W of the rectangle
variables {L W : ℝ} (hL : L > 0) (hW : W > 0)

-- Define the measured lengths L' and W'
def L' := 1.08 * L
def W' := 0.93 * W

-- Define the actual area and the calculated area
def actualArea := L * W
def measuredArea := L' * W'

-- Define the error in the area
def error := measuredArea - actualArea

-- Define the error percent
def errorPercent := (error / actualArea) * 100

-- Theorem statement that the error percent is 0.44%
theorem error_percent_is_correct : errorPercent = 0.44 := by
  sorry

end error_percent_is_correct_l780_780000


namespace locus_is_circle_l780_780494

theorem locus_is_circle (z : ℂ) : |z - complex.I| = |3 - 4 * complex.I| -> 
  exists (c : ℂ) (r : ℝ), r > 0 ∧ (∀ (w : ℂ), |w - c| = r <-> |w - complex.I| = |3 - 4 * complex.I|) :=
sorry

end locus_is_circle_l780_780494


namespace g_infinite_distinct_values_l780_780080

def g (x : ℝ) : ℝ :=
  (∑ k in finset.range 11, (⌊k * x⌋ - (k + 1) * ⌊x⌋)) + 3 * x

theorem g_infinite_distinct_values : ∀ x : ℝ, x ≥ 0 → set.infinite (set.range g) := 
by sorry

end g_infinite_distinct_values_l780_780080


namespace original_price_of_bag_l780_780718

theorem original_price_of_bag (P : ℝ) 
  (h1 : ∀ x, 0 < x → x < 1 → x * 100 = 75)
  (h2 : 2 * (0.25 * P) = 3)
  : P = 6 :=
sorry

end original_price_of_bag_l780_780718


namespace median_free_throws_l780_780658

def free_throws : List ℕ := [8, 6, 18, 15, 14, 19, 12, 20, 19, 15]

def median_of_list (lst : List ℕ) : ℕ :=
  let sorted_lst := lst.qsort (· ≤ ·)
  if h : sorted_lst.length % 2 = 0 then
    let mid := sorted_lst.length / 2
    (sorted_lst.get! (mid - 1) + sorted_lst.get! mid) / 2
  else
    sorted_lst.get! (sorted_lst.length / 2)

theorem median_free_throws : median_of_list free_throws = 15 :=
by
  sorry

end median_free_throws_l780_780658


namespace james_total_matches_l780_780899

-- Define the conditions
def dozen : Nat := 12
def boxes_per_dozen : Nat := 5
def matches_per_box : Nat := 20

-- Calculate the expected number of matches
def expected_matches : Nat := boxes_per_dozen * dozen * matches_per_box

-- State the theorem to be proved
theorem james_total_matches : expected_matches = 1200 := by
  sorry

end james_total_matches_l780_780899


namespace parabola_focus_and_directrix_l780_780349

theorem parabola_focus_and_directrix :
  (∀ x y : ℝ, x^2 = 4 * y → ∃ a b : ℝ, (a, b) = (0, 1) ∧ y = -1) :=
by
  -- Here, we would provide definitions and logical steps if we were completing the proof.
  -- For now, we will leave it unfinished.
  sorry

end parabola_focus_and_directrix_l780_780349


namespace arc_length_solution_l780_780386

variable (r : ℝ) (α : ℝ)

theorem arc_length_solution (h1 : r = 8) (h2 : α = 5 * Real.pi / 3) : 
    r * α = 40 * Real.pi / 3 := 
by 
    sorry

end arc_length_solution_l780_780386


namespace necessary_ab_given_a_b_not_sufficient_ab_given_a_b_l780_780366

theorem necessary_ab_given_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 4) : 
  a + b ≥ 4 :=
sorry

theorem not_sufficient_ab_given_a_b : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b ≥ 4 ∧ a * b < 4 :=
sorry

end necessary_ab_given_a_b_not_sufficient_ab_given_a_b_l780_780366


namespace tan_α_eq_one_half_complex_expression_eq_neg_sqrt5_fifth_l780_780335

-- Define the conditions
def terminal_side_angle_α (α : ℝ) (P : ℝ × ℝ) : Prop :=
  P = (-2, -1)

def cos_alpha (α : ℝ) : Type :=
  ℝ

theorem tan_α_eq_one_half (α : ℝ) (P : ℝ × ℝ) (x : ℝ)
  (h1 : P = (x, -1))
  (h2: x < 0)
  (h3 : cos α = (sqrt 5) / 5 * x) :
  tan α = 1 / 2 :=
sorry

theorem complex_expression_eq_neg_sqrt5_fifth (α : ℝ) (P : ℝ × ℝ) (x : ℝ)
  (h1 : P = (x, -1))
  (h2: x < 0)
  (h3 : cos α = (sqrt 5) / 5 * x) :
  (1 - cos (2 * α)) / (sqrt 2 * cos (α - π / 4) + sin (π + α)) = - (sqrt 5) / 5 :=
sorry

end tan_α_eq_one_half_complex_expression_eq_neg_sqrt5_fifth_l780_780335


namespace Janet_rides_roller_coaster_7_times_l780_780908

def rides (x : ℕ) : Prop :=
  5 * x + 3 * 4 = 47

theorem Janet_rides_roller_coaster_7_times :
  ∃ (x : ℕ), rides x ∧ x = 7 := by
  constructor
  · use 7
  · sorry

end Janet_rides_roller_coaster_7_times_l780_780908


namespace inclination_angle_range_l780_780991

theorem inclination_angle_range (α θ : ℝ) :
  (∃ (theta : ℝ), tan theta = cos α) →
  (cos α ∈ set.Icc (-1:ℝ) (1:ℝ)) →
  θ ∈ (set.Icc 0 (π/4) ∪ set.Ico (3*π/4) π) := by
  sorry

end inclination_angle_range_l780_780991


namespace count_irreducible_fractions_l780_780988

theorem count_irreducible_fractions :
  let fractions := {n : ℕ | 240 < n ∧ n < 2250 ∧ Int.gcd 15 n = 1}
  fractions.card = 8 :=
by 
  sorry

end count_irreducible_fractions_l780_780988


namespace intersection_eq_T_l780_780162

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780162


namespace solve_equation_l780_780648

theorem solve_equation (x : ℝ) : (3 * x - 2 * (10 - x) = 5) → x = 5 :=
by {
  sorry
}

end solve_equation_l780_780648


namespace intersection_of_sets_l780_780175

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780175


namespace total_kids_got_in_equals_148_l780_780748

def total_kids : ℕ := 120 + 90 + 50

def denied_riverside : ℕ := (20 * 120) / 100
def denied_west_side : ℕ := (70 * 90) / 100
def denied_mountaintop : ℕ := 50 / 2

def got_in_riverside : ℕ := 120 - denied_riverside
def got_in_west_side : ℕ := 90 - denied_west_side
def got_in_mountaintop : ℕ := 50 - denied_mountaintop

def total_got_in : ℕ := got_in_riverside + got_in_west_side + got_in_mountaintop

theorem total_kids_got_in_equals_148 :
  total_got_in = 148 := 
by
  unfold total_got_in
  unfold got_in_riverside got_in_west_side got_in_mountaintop
  unfold denied_riverside denied_west_side denied_mountaintop
  sorry

end total_kids_got_in_equals_148_l780_780748


namespace integral_1_integral_2_l780_780714

theorem integral_1: ∫ x in 0..1, (2 * x + 3) = 4 :=
by sorry

theorem integral_2: ∫ x in Real.exp 1..Real.exp 3, (1 / x) = 2 :=
by sorry

end integral_1_integral_2_l780_780714


namespace find_width_of_room_l780_780979

section RoomWidth

variable (l C P A W : ℝ)
variable (h1 : l = 5.5)
variable (h2 : C = 16500)
variable (h3 : P = 750)
variable (h4 : A = C / P)
variable (h5 : A = l * W)

theorem find_width_of_room : W = 4 := by
  sorry

end RoomWidth

end find_width_of_room_l780_780979


namespace tips_collected_l780_780942

-- Definitions based on conditions
def total_collected : ℕ := 240
def hourly_wage : ℕ := 10
def hours_worked : ℕ := 19

-- Correct answer translated into a proof problem
theorem tips_collected : total_collected - (hours_worked * hourly_wage) = 50 := by
  sorry

end tips_collected_l780_780942


namespace intersection_eq_T_l780_780165

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780165


namespace probability_perfect_square_divisor_of_factorial_15_l780_780682

def factorial : Nat → Nat
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def is_perfect_square (m : Nat) : Prop :=
  ∃ k : Nat, k * k = m

def total_divisors (n : Nat) : Nat := sorry -- A function that computes the number of total divisors
def perfect_square_divisors (n : Nat) : Nat := sorry -- A function that computes the number of perfect square divisors

theorem probability_perfect_square_divisor_of_factorial_15 :
  let n := factorial 15 in
  let total_divs := total_divisors n in
  let perfect_square_divs := perfect_square_divisors n in
  (perfect_square_divs / total_divs) = 7 / 504 :=
by
  sorry

end probability_perfect_square_divisor_of_factorial_15_l780_780682


namespace arithmetic_seq_solution_l780_780389

theorem arithmetic_seq_solution (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) 
  (h_nonzero : ∀ n, a n ≠ 0) 
  (h_arith : ∀ n ≥ 2, a (n+1) - a n ^ 2 + a (n-1) = 0) 
  (h_sum : ∀ k, S k = (k * (a 1 + a k)) / 2) :
  S (2 * n - 1) - 4 * n = -2 := 
sorry

end arithmetic_seq_solution_l780_780389


namespace largest_n_binomial_l780_780574

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l780_780574


namespace proof_problem1_proof_problem2_l780_780009

noncomputable def problem1 : ℝ :=
  (2 + 7 / 9) ^ 0.5 + (0.1) ^ (-2) + (2 + 10 / 27) ^ (-2 / 3) - 3 * Real.pi ^ 0 + 37 / 48

theorem proof_problem1 : problem1 = 100 := 
by sorry

variables (a b : ℝ) (h₁ : a ≠0) (h₂ : b ≠0)

noncomputable def problem2 : ℝ :=
  (a ^ (8 / 5) * b ^ (-6 / 5)) ^ (-1 / 2) * 5 * a ^ 4 / (5 * b ^ 3)

theorem proof_problem2 : 
  ∀ (a b : ℝ) (h₁ : a ≠0) (h₂ : b ≠0), 
  problem2 a b h₁ h₂ = 1 := 
by sorry

end proof_problem1_proof_problem2_l780_780009


namespace jason_manager_years_l780_780909

-- Definitions based on the conditions
def jason_bartender_years : ℕ := 9
def jason_total_months : ℕ := 150
def additional_months_excluded : ℕ := 6

-- Conversion from months to years
def total_years := jason_total_months / 12
def excluded_years := additional_months_excluded / 12

-- Lean statement for the proof problem
theorem jason_manager_years :
  total_years - jason_bartender_years - excluded_years = 3 := by
  sorry

end jason_manager_years_l780_780909


namespace sum_g_eq_1000_l780_780439

def g (x : ℝ) : ℝ := 4 / (16^x + 4)

theorem sum_g_eq_1000 :
  ∑ k in finset.range 2000, g ((k + 1) / 2001) = 1000 :=
sorry

end sum_g_eq_1000_l780_780439


namespace intersection_S_T_eq_T_l780_780208

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l780_780208


namespace purchase_in_april_l780_780985

namespace FamilySavings

def monthly_income : ℕ := 150000
def monthly_expenses : ℕ := 115000
def initial_savings : ℕ := 45000
def furniture_cost : ℕ := 127000

noncomputable def monthly_savings : ℕ := monthly_income - monthly_expenses
noncomputable def additional_amount_needed : ℕ := furniture_cost - initial_savings
noncomputable def months_required : ℕ := (additional_amount_needed + monthly_savings - 1) / monthly_savings  -- ceiling division

theorem purchase_in_april : months_required = 3 :=
by
  -- Proof goes here
  sorry

end FamilySavings

end purchase_in_april_l780_780985


namespace exists_square_with_same_color_around_l780_780005

-- Define the large square with dimensions 50x50 and 4 colors
def large_square : Type := array (fin 50) (array (fin 50) (fin 4))

-- Define the property to check if a square has the same color around it
def has_same_color_around (s : large_square) (i j : fin 50) : Prop :=
  ∃ k₁ k₂ : fin 50, k₁ ≠ i ∧ s[i] = s[k₁] ∧ k₂ ≠ j ∧ s[_] = s[k₂]

-- Statement of the theorem
theorem exists_square_with_same_color_around (s : large_square) : 
  ∃ i j : fin 50, has_same_color_around s i j := sorry

end exists_square_with_same_color_around_l780_780005


namespace ratio_A_B_l780_780742

noncomputable def A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
noncomputable def B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)

theorem ratio_A_B :
  let r := (A : ℚ) / (B : ℚ) in 0 < r ∧ r < 1 :=
by {
  -- Proof steps are omitted with sorry
  sorry
}

end ratio_A_B_l780_780742


namespace exists_1987_points_l780_780893

noncomputable def points (n : ℕ) : ℕ → ℝ × ℝ := λ i, (i, i ^ 2)

def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

def triangle_area (p q r : ℝ × ℝ) : ℝ :=
  (0.5 : ℝ) * abs (p.1 * (q.2 - r.2) + q.1 * (r.2 - p.2) + r.1 * (p.2 - q.2))

theorem exists_1987_points :
  ∃ P : ℕ → ℝ × ℝ, (∀ i j, i ≠ j → ¬ (distance (P i) (P j)).is_rational) ∧
    (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → triangle_area (P i) (P j) (P k) ≠ 0 ∧ 
      (triangle_area (P i) (P j) (P k)).is_rational) ∧
    (∀ i, i < 1987 → P i = points 1987 i) :=
begin
  use points 1987,
  split,
  { intros i j hij,
    rw [distance, points, points, Real.sqrt_eq_rpow, ← sq_eq_sq],
    sorry
  },
  split,
  { intros i j k hijk,
    rw [triangle_area, points, points, points],
    sorry
  },
  { intros i hi,
    refl
  }
end

end exists_1987_points_l780_780893


namespace patrick_loss_percentage_l780_780460

theorem patrick_loss_percentage :
  let cost_A := 30 * 1,
      cost_B := 40 * 2,
      cost_C := 10 * 3,
      total_cost := cost_A + cost_B + cost_C,
      selling_A := 30 * (1 - 0.50),
      selling_B := 40 * (2 - 1),
      selling_C := 10 * (3 - 1.50),
      total_selling := selling_A + selling_B + selling_C,
      total_loss := total_cost - total_selling in
  (total_loss.to_float / total_cost.to_float) * 100 = 10.71 := sorry

end patrick_loss_percentage_l780_780460


namespace largest_n_binom_equality_l780_780519

theorem largest_n_binom_equality :
  ∃ n : ℕ, (binomial 10 4 + binomial 10 5 = binomial 11 n) ∧ ∀ m : ℕ, (binomial 11 m = binomial 11 5) → m ≤ 6 :=
by
  sorry

end largest_n_binom_equality_l780_780519


namespace intersection_S_T_eq_T_l780_780212

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780212


namespace S_inter_T_eq_T_l780_780300

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780300


namespace intersection_S_T_eq_T_l780_780137

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780137


namespace intersection_S_T_eq_T_l780_780259

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l780_780259


namespace chord_condition_l780_780764

theorem chord_condition (n : ℕ → ℕ) (h_sum_even : ∃ m : ℕ, (∑ i in finset.range 1991, n i) = 2 * m) 
(h_bound : ∀ i < 1991, n i ≤ m) : 
(∀ (i j : ℕ) (hi : i < 1991) (hj : j < 1991), n i ≠ n j → disjoint (chords i) (chords j) ∧ different_labels (i) (j)) :=
sorry

end chord_condition_l780_780764


namespace largest_n_for_binomial_equality_l780_780530

theorem largest_n_for_binomial_equality :
  ∃ n : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 n ) ∧
  ∀ m : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 m ) → m ≤ 6 :=
begin
  sorry
end

end largest_n_for_binomial_equality_l780_780530


namespace largest_n_binom_equality_l780_780520

theorem largest_n_binom_equality :
  ∃ n : ℕ, (binomial 10 4 + binomial 10 5 = binomial 11 n) ∧ ∀ m : ℕ, (binomial 11 m = binomial 11 5) → m ≤ 6 :=
by
  sorry

end largest_n_binom_equality_l780_780520


namespace deposit_percentage_l780_780663

theorem deposit_percentage (deposit remaining_amount : ℝ) (deposit_eq : deposit = 140) (remaining_amount_eq : remaining_amount = 1260) :
  (deposit / (deposit + remaining_amount)) * 100 = 10 :=
by
  rw [deposit_eq, remaining_amount_eq]
  norm_num
  sorry

end deposit_percentage_l780_780663


namespace intersection_eq_T_l780_780155

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780155


namespace triangle_area_l780_780491

-- Define the sides and angle
def a := 1
def c := 2
def B := Real.pi / 3

-- Define the area using the given formula
def area (a c B : ℝ) := (1 / 2) * a * c * Real.sin B

-- The goal is to prove the area is sqrt(3)/2 given the conditions
theorem triangle_area : area a c B = Real.sqrt 3 / 2 := sorry

end triangle_area_l780_780491


namespace class_7th_grade_1_has_higher_average_score_class_7th_grade_2_has_higher_weighted_score_l780_780664

noncomputable def average_score (costume pitch innovation : ℕ) : ℚ :=
  (costume + pitch + innovation) / 3

noncomputable def weighted_average_score (costume pitch innovation : ℕ) : ℚ :=
  (costume + 7 * pitch + 2 * innovation) / 10

theorem class_7th_grade_1_has_higher_average_score :
  average_score 90 77 85 > average_score 74 95 80 :=
by sorry

theorem class_7th_grade_2_has_higher_weighted_score :
  weighted_average_score 74 95 80 > weighted_average_score 90 77 85 :=
by sorry

end class_7th_grade_1_has_higher_average_score_class_7th_grade_2_has_higher_weighted_score_l780_780664


namespace intersection_S_T_l780_780152

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l780_780152


namespace Northton_time_capsule_depth_l780_780473

theorem Northton_time_capsule_depth:
  ∀ (d_southton d_northton : ℝ),
  d_southton = 15 →
  d_northton = (4 * d_southton) - 12 →
  d_northton = 48 :=
by
  intros d_southton d_northton h_southton h_northton
  rw [h_southton] at h_northton
  rw [← h_northton]
  sorry

end Northton_time_capsule_depth_l780_780473


namespace fencing_cost_l780_780686

theorem fencing_cost
  (width : ℝ)
  (uncovered_side : ℝ)
  (area : ℝ)
  (tilt_angle_deg : ℝ)
  (cost_uncovered : ℝ)
  (cost_adjacent : ℝ)
  (cost_tilted : ℝ)
  (h_area : area = 600)
  (h_uncovered_side : uncovered_side = 30)
  (h_tilt_angle_deg : tilt_angle_deg = 60)
  (h_cost_uncovered : cost_uncovered = 2)
  (h_cost_adjacent : cost_adjacent = 3)
  (h_cost_tilted : cost_tilted = 4)
  (h_width : width = area / uncovered_side):
  let tilted_side := 2 * uncovered_side in
  let total_cost := uncovered_side * cost_uncovered + width * cost_adjacent + tilted_side * cost_tilted in
  total_cost = 360 :=
by
  simp [h_area, h_uncovered_side, h_tilt_angle_deg, h_cost_uncovered, h_cost_adjacent, h_cost_tilted, h_width]
  done

end fencing_cost_l780_780686


namespace highest_place_value_quotient_and_remainder_l780_780518

-- Conditions
def dividend := 438
def divisor := 4

-- Theorem stating that the highest place value of the quotient is the hundreds place, and the remainder is 2
theorem highest_place_value_quotient_and_remainder : 
  (dividend = divisor * (dividend / divisor) + (dividend % divisor)) ∧ 
  ((dividend / divisor) >= 100) ∧ 
  ((dividend % divisor) = 2) :=
by
  sorry

end highest_place_value_quotient_and_remainder_l780_780518


namespace intersection_eq_T_l780_780167

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780167


namespace range_of_m_l780_780368

noncomputable def f (x m : ℝ) : ℝ := (1 / 2) * (x - 2) ^ 2 + m * Real.log x

theorem range_of_m {m : ℝ} :
  (∀ x ∈ Set.Ioo 1 2, deriv (λ x, f x m) x ≤ 0) → m ≤ 0 :=
begin
  -- Proof omitted
  sorry
end

end range_of_m_l780_780368


namespace three_pow_a_sub_b_eq_frac_10_7_l780_780927

noncomputable def a : ℝ := Real.log 10 / Real.log 3
noncomputable def b : ℝ := Real.log 7 / Real.log 3

theorem three_pow_a_sub_b_eq_frac_10_7 :
  3 ^ (a - b) = 10 / 7 :=
by
  sorry

end three_pow_a_sub_b_eq_frac_10_7_l780_780927


namespace bead_probability_l780_780769

open Finset

-- Number of ways to arrange 9 beads with given counts
def total_arrangements : Nat := (factorial 9 / (factorial 4 * factorial 3 * factorial 2))

-- Function to calculate the number of valid arrangements
-- This is a placeholder to represent the condition of valid arrangements.
noncomputable def valid_arrangements : Nat := 80 -- Assume we have calculated this as per the solution's steps

-- Compute the probability of valid arrangements
noncomputable def probability_valid : ℚ := valid_arrangements / total_arrangements

-- Main theorem statement
theorem bead_probability :
  probability_valid = 10 / 157 :=
by
  -- This is where one would provide the detailed proof, but we're just stating the theorem
  sorry

end bead_probability_l780_780769


namespace intersection_eq_T_l780_780283

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l780_780283


namespace largest_n_binom_equality_l780_780525

theorem largest_n_binom_equality :
  ∃ n : ℕ, (binomial 10 4 + binomial 10 5 = binomial 11 n) ∧ ∀ m : ℕ, (binomial 11 m = binomial 11 5) → m ≤ 6 :=
by
  sorry

end largest_n_binom_equality_l780_780525


namespace F_3_f_5_eq_24_l780_780782

def f (a : ℤ) : ℤ := a - 2
def F (a b : ℤ) : ℤ := b^3 - a

theorem F_3_f_5_eq_24 : F 3 (f 5) = 24 := by
  sorry

end F_3_f_5_eq_24_l780_780782


namespace estimate_ratio_l780_780733

theorem estimate_ratio (A B : ℕ) (A_def : A = 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28)
  (B_def : B = 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20) : 0 < A / B ∧ A / B < 1 := by
  sorry

end estimate_ratio_l780_780733


namespace men_at_each_table_l780_780696

theorem men_at_each_table:
  ∀ (num_tables num_women_per_table total_customers : ℕ)
  (h1: num_tables = 5)
  (h2: num_women_per_table = 5)
  (h3: total_customers = 40),
  let total_women := num_tables * num_women_per_table in
  let total_men := total_customers - total_women in
  let men_per_table := total_men / num_tables in
  men_per_table = 3 :=
by
  intros num_tables num_women_per_table total_customers h1 h2 h3
  let total_women := num_tables * num_women_per_table
  let total_men := total_customers - total_women
  let men_per_table := total_men / num_tables
  sorry

end men_at_each_table_l780_780696


namespace real_numbers_int_approximation_l780_780894

theorem real_numbers_int_approximation:
  ∀ (x y : ℝ), ∃ (m n : ℤ),
  (x - m) ^ 2 + (y - n) * (x - m) + (y - n) ^ 2 ≤ (1 / 3) :=
by
  intros x y
  sorry

end real_numbers_int_approximation_l780_780894


namespace largest_integer_binom_l780_780616

theorem largest_integer_binom
  (n : ℕ)
  (h₁ : binom 10 4 + binom 10 5 = binom 11 n) :
  n = 6 :=
sorry

end largest_integer_binom_l780_780616


namespace intersection_S_T_eq_T_l780_780280

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l780_780280


namespace intersection_eq_T_l780_780168

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l780_780168


namespace line_ellipse_relationship_l780_780014

theorem line_ellipse_relationship (m : ℝ) :
  let L := λ x : ℝ, x + m
  let ellipse := 9 * x^2 + 16 * y^2 = 144
  let Δ := -576 * m^2 + 14400
  if Δ > 0 then -5 < m ∧ m < 5
  else if Δ = 0 then m = 5 ∨ m = -5
  else m > 5 ∨ m < -5 :=
by sorry

end line_ellipse_relationship_l780_780014


namespace intersection_of_S_and_T_l780_780229

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780229


namespace veronica_loss_more_than_seth_l780_780957

noncomputable def seth_loss : ℝ := 17.5
noncomputable def jerome_loss : ℝ := 3 * seth_loss
noncomputable def total_loss : ℝ := 89
noncomputable def veronica_loss : ℝ := total_loss - (seth_loss + jerome_loss)

theorem veronica_loss_more_than_seth :
  veronica_loss - seth_loss = 1.5 :=
by
  have h_seth_loss : seth_loss = 17.5 := rfl
  have h_jerome_loss : jerome_loss = 3 * seth_loss := rfl
  have h_total_loss : total_loss = 89 := rfl
  have h_veronica_loss : veronica_loss = total_loss - (seth_loss + jerome_loss) := rfl
  sorry

end veronica_loss_more_than_seth_l780_780957


namespace slope_of_EF_l780_780124

theorem slope_of_EF
  (a b : ℝ) (c : ℝ)
  (h1 : a > b) 
  (h2 : b > 0)
  (e : ℝ)
  (h3 : e = √2 / 2) 
  (h4 : a = √2 * b)
  (h5 : a = √2 * c)
  (E : ℝ × ℝ)
  (h6 : E = (-c, - (√2 / 2) * c))
  (F : ℝ × ℝ)
  (h7 : F = (c, 0))
  : (E.snd - F.snd) / (E.fst - F.fst) = √2 / 4 :=
sorry

end slope_of_EF_l780_780124


namespace projections_have_equal_radii_l780_780499

-- Definitions for the conditions of the problem
def Body : Type := sorry -- A type to represent a body (we assume it's any geometric 3D object)

noncomputable def proj (b : Body) (P : Type) [Plane P] : Set Point := sorry -- Projection of body b onto plane P

noncomputable def is_circle (s : Set Point) : Prop := sorry -- Predicate indicating if a set of points forms a circle

noncomputable def radius (s : Set Point) (h : is_circle s) : ℝ := sorry -- Radius of the circle formed by a set of points

-- Problem Statement
theorem projections_have_equal_radii (b : Body) (P1 P2 : Type) [Plane P1] [Plane P2]
  (hP1 : is_circle (proj b P1)) (hP2 : is_circle (proj b P2)) :
  radius (proj b P1) hP1 = radius (proj b P2) hP2 :=
sorry

end projections_have_equal_radii_l780_780499


namespace problem_statement_l780_780800

variable {α : Type*} [LinearOrderedField α]
variable {f : α → α}

-- Define that the function f is even
def is_even_function (f : α → α) : Prop :=
  ∀ x, f(x) = f(-x)

-- Define that the function f(x - 2) is monotonically decreasing on [0, 2]
def is_monotonically_decreasing_on_interval (f : α → α) (a b : α) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f(x - 2) ≥ f(y - 2)

-- Main theorem
theorem problem_statement (h1 : is_even_function f)
    (h2 : is_monotonically_decreasing_on_interval f (0 : α) (2 : α)) :
    f 0 < f (-1) ∧ f (-1) < f 2 :=
by
  sorry

end problem_statement_l780_780800


namespace sin_A_in_right_triangle_l780_780879

theorem sin_A_in_right_triangle (A B C : ℝ) (h1 : ∠ B = 90) (h2 : 2 * sin A = 3 * cos A) : 
  sin A = 3 * sqrt 13 / 13 := 
  sorry

end sin_A_in_right_triangle_l780_780879


namespace S_inter_T_eq_T_l780_780304

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l780_780304


namespace intersection_S_T_eq_T_l780_780139

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l780_780139


namespace intersection_of_sets_l780_780182

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780182


namespace largest_n_for_binomial_equality_l780_780531

theorem largest_n_for_binomial_equality :
  ∃ n : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 n ) ∧
  ∀ m : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 m ) → m ≤ 6 :=
begin
  sorry
end

end largest_n_for_binomial_equality_l780_780531


namespace intersection_S_T_eq_T_l780_780216

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l780_780216


namespace triangle_area_l780_780802

noncomputable def hyperbola_eq (a : ℝ) : set (ℝ × ℝ) := {p | p.1^2 - p.2^2 = a^2}

noncomputable def circle_eq (c : ℝ) : set (ℝ × ℝ) := {p | (p.1 - c)^2 + p.2^2 = 1}

noncomputable def line_eq : set (ℝ × ℝ) := {p | p.1 = p.2}

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def area (F1 M F2 : ℝ × ℝ) : ℝ := 
  0.5 * abs ((F1.1 - F2.1) * (M.2 - F2.2) - (F1.2 - F2.2) * (M.1 - F2.1))

theorem triangle_area (a : ℝ) (h : a > 0) (c : ℝ := sqrt 2 * a) 
  (F1 : ℝ × ℝ := (-sqrt 2 * a, 0)) (F2 : ℝ × ℝ := (sqrt 2 * a, 0)) 
  (M : ℝ × ℝ := (sqrt(2)/2, sqrt(2)/2)) : 
  hyperbola_eq a F1 ∧ hyperbola_eq a F2 ∧ circle_eq c F2 ∧ line_eq M →
  area F1 M F2 = 1 :=
by
  sorry

end triangle_area_l780_780802


namespace circle_diameter_line_eq_l780_780809

theorem circle_diameter_line_eq (x y : ℝ) :
  x^2 + y^2 - 2*x + 6*y + 8 = 0 → (2 * 1 + (-3) + 1 = 0) :=
by
  sorry

end circle_diameter_line_eq_l780_780809


namespace logarithmic_equation_solution_l780_780085

theorem logarithmic_equation_solution (x : ℝ) 
  (h1 : x > -5) 
  (h2 : x > 1 / 2) : 
  log (x + 5) + log (2 * x - 1) = log (3 * x^2 - 11 * x + 5) ↔ 
  x = 10 + 3 * Real.sqrt 10 ∨ x = 10 - 3 * Real.sqrt 10 :=
sorry

end logarithmic_equation_solution_l780_780085


namespace f_value_at_4_l780_780325

theorem f_value_at_4 (f : ℤ → ℤ) 
  (h1 : ∀ x, f(-x-1) = -f(x-1)) 
  (h2 : ∀ x, f(-x+1) = f(x+1))
  (h3 : f 2008 = 1) : 
  f 4 = -1 := 
sorry

end f_value_at_4_l780_780325


namespace intersection_eq_T_l780_780249

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l780_780249


namespace intersection_of_sets_l780_780174

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l780_780174


namespace problem_conditions_and_solutions_l780_780812

noncomputable def f (a b c x : ℝ) : ℝ := 
  -x^3 + a*x^2 + b*x + c

theorem problem_conditions_and_solutions (a b c : ℝ) (h_decreasing : ∀ x < 0, f a b c x < f a b c (x+1)) 
  (h_increasing : ∀ x (0:ℝ) < x ∧ x < 1, f a b c (x+1) > f a b c x) 
  (h_root : f a b c 1 = 0) (h_zeros : ∃ x₁ x₂ x₃, f a b c x₁ = 0 ∧ f a b c x₂ = 0 ∧ f a b c x₃ = 0) :
  b = 0 ∧ ∃ r : set ℝ, r = {y : ℝ | ...} := -- specify the set for the range of f(2)
begin
  sorry,
end

end problem_conditions_and_solutions_l780_780812


namespace triangle_is_isosceles_l780_780382

theorem triangle_is_isosceles
    (A B C : ℝ)
    (h_angle_sum : A + B + C = 180)
    (h_sinB : Real.sin B = 2 * Real.cos C * Real.sin A)
    : (A = C) := 
by
    sorry

end triangle_is_isosceles_l780_780382


namespace length_of_rectangle_l780_780088

def question_and_conditions (w x : ℝ) (n : ℕ) (total_area : ℝ) : Prop :=
  (n = 8) ∧ (4 * w = 2 * x) ∧ (total_area = 8000)

theorem length_of_rectangle (x : ℝ) (w : ℝ) (n : ℕ) (total_area : ℝ) : question_and_conditions w x n total_area → x = 45 :=
by
  intro h,
  cases h with hn hrest,
  cases hrest with heqn htarea,
  sorry

end length_of_rectangle_l780_780088


namespace intersection_of_S_and_T_l780_780225

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l780_780225


namespace axel_vowels_written_l780_780065

theorem axel_vowels_written (total_alphabets number_of_vowels n : ℕ) (h1 : total_alphabets = 10) (h2 : number_of_vowels = 5) (h3 : total_alphabets = number_of_vowels * n) : n = 2 :=
by
  sorry

end axel_vowels_written_l780_780065


namespace percentage_loss_l780_780040

theorem percentage_loss (selling_price_with_loss : ℝ)
    (desired_selling_price_for_profit : ℝ)
    (profit_percentage : ℝ) (actual_selling_price : ℝ)
    (calculated_loss_percentage : ℝ) :
    selling_price_with_loss = 16 →
    desired_selling_price_for_profit = 21.818181818181817 →
    profit_percentage = 20 →
    actual_selling_price = 18.181818181818182 →
    calculated_loss_percentage = 12 → 
    calculated_loss_percentage = (actual_selling_price - selling_price_with_loss) / actual_selling_price * 100 := 
sorry

end percentage_loss_l780_780040
