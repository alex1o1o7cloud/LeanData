import Mathlib

namespace find_m_n_diff_l717_717422

theorem find_m_n_diff (a : ℝ) (n m: ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1)
  (h_pass : a^(2 * m - 6) + n = 2) :
  m - n = 2 :=
sorry

end find_m_n_diff_l717_717422


namespace find_z_l717_717768

noncomputable def solve_for_z (i : ℂ) (z : ℂ) :=
  (2 - i) * z = i ^ 2021

theorem find_z (i z : ℂ) (h1 : solve_for_z i z) : 
  z = -1/5 + 2/5 * i := 
by 
  sorry

end find_z_l717_717768


namespace parallelepiped_rectangular_if_circumscribed_by_sphere_l717_717918

-- Define a parallelepiped in Lean.
structure Parallelepiped :=
(faces : List (EuclideanGeometry.Parallelogram))

-- Define the condition of a parallelepiped being circumscribed by a sphere
def isCircumscribedBySphere (P : Parallelepiped) : Prop :=
∀ face ∈ P.faces, EuclideanGeometry.isInscribedInCircle face

-- Define a rectangular parallelepiped
def isRectangular (P : Parallelepiped) : Prop :=
∀ face ∈ P.faces, EuclideanGeometry.isRectangle face

-- Statement to prove
theorem parallelepiped_rectangular_if_circumscribed_by_sphere (P : Parallelepiped) :
  isCircumscribedBySphere P → isRectangular P := by
sorry

end parallelepiped_rectangular_if_circumscribed_by_sphere_l717_717918


namespace petya_prevents_vasya_l717_717915

-- Define the nature of fractions and the players' turns
def is_natural_sum (fractions : List ℚ) : Prop :=
  (fractions.sum = ⌊fractions.sum⌋)

def petya_vasya_game_prevent (fractions : List ℚ) : Prop :=
  ∀ k : ℕ, ∀ additional_fractions : List ℚ, 
  (additional_fractions.length = k) →
  ¬ is_natural_sum (fractions ++ additional_fractions)

theorem petya_prevents_vasya : ∀ fractions : List ℚ, petya_vasya_game_prevent fractions :=
by
  sorry

end petya_prevents_vasya_l717_717915


namespace mutual_fund_profit_inconsistency_l717_717318

theorem mutual_fund_profit_inconsistency:
  ∃ (P : ℝ), let total_investment := 1900 in
             let profit_first_fund := 0.09 * 1700 in
             let total_profit := 52 in
             let investment_first_fund := 1700 in
             let investment_second_fund := total_investment - investment_first_fund in
             let profit_second_fund := total_profit - profit_first_fund in
             profit_second_fund ≠ investment_second_fund * (P / 100) :=
begin
  sorry
end

end mutual_fund_profit_inconsistency_l717_717318


namespace on_time_departure_rate_l717_717953

theorem on_time_departure_rate (x : ℕ) : 
  (3 + x > 0.40 * (4 + x)) → x ≥ 1 := 
by {
  sorry
}

end on_time_departure_rate_l717_717953


namespace angle_APB_largest_probability_l717_717994

constant P_in_CD {A B C D P : Type} {x : ℝ} : 2 - real.sqrt 3 ≤ x ∧ x ≤ real.sqrt 3

/-- Given a rectangle ABCD with AB = 2 and BC = 1, and a point P randomly selected on CD,
    the probability that ∠APB is the largest among the three interior angles of ΔPAB is √3 - 1. -/
theorem angle_APB_largest_probability :
  ∀ {A B C D P: Type} {AB BC CD : ℝ} (h_AB : AB = 2) (h_BC : BC = 1) (h_CD : CD = 2),
  (2 - real.sqrt 3 ≤ P ∧ P ≤ real.sqrt 3) →
  let probability : Type := (real.sqrt 3 - 1) / 2
  in probability = (real.sqrt 3 - 1) :=
begin
  sorry
end

end angle_APB_largest_probability_l717_717994


namespace probability_Jane_Albert_same_committee_is_correct_l717_717239

noncomputable def probability_Jane_Albert_same_committee : ℝ :=
  let n : ℕ := 6
  let k : ℕ := 3
  let all_students : Finset ℕ := {0, 1, 2, 3, 4, 5}  -- representing the 6 students
  let Jane : ℕ := 4
  let Albert : ℕ := 5
  let possible_committees := all_students.powerset.filter (λ s, s.card = k)
  let total_committees : ℕ := possible_committees.card
  let favorable_committees := possible_committees.filter (λ s, Jane ∈ s ∧ Albert ∈ s)
  let favorable_count : ℕ := favorable_committees.card
  (favorable_count : ℝ) / (total_committees : ℝ)

theorem probability_Jane_Albert_same_committee_is_correct : probability_Jane_Albert_same_committee = 1 / 5 :=
by
  let n : ℕ := 6
  let k : ℕ := 3
  let all_students : Finset ℕ := {0, 1, 2, 3, 4, 5}  -- representing the 6 students
  let Jane : ℕ := 4
  let Albert : ℕ := 5
  let possible_committees := all_students.powerset.filter (λ s, s.card = k)
  let total_committees := possible_committees.card
  have h_total_committees : total_committees = 20 := by
    -- Proof of total committees being 20
    sorry
  let favorable_committees := possible_committees.filter (λ s, Jane ∈ s ∧ Albert ∈ s)
  let favorable_count := favorable_committees.card
  have h_favorable_count : favorable_count = 4 := by
    -- Proof of favorable committees being 4
    sorry
  calc
    probability_Jane_Albert_same_committee
      = (favorable_committees.card : ℝ) / (possible_committees.card : ℝ) := rfl
  ... = (favorable_count : ℝ) / (total_committees : ℝ) := by congr
  ... = 4 / 20 := by rw [h_favorable_count, h_total_committees]
  ... = 1 / 5 := by norm_num

end probability_Jane_Albert_same_committee_is_correct_l717_717239


namespace f_monotone_f_inequality_solution_l717_717056

noncomputable def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x : ℝ, x > 0 → ∃ y, f y = x
axiom f_at_2: f 2 = 1
axiom f_mul : ∀ x y, f (x * y) = f x + f y
axiom f_positive : ∀ x, x > 1 → f x > 0

theorem f_monotone (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) : x₁ < x₂ → f x₁ < f x₂ :=
sorry

theorem f_inequality_solution (x : ℝ) (hx : x > 2 ∧ x ≤ 4) : f x + f (x - 2) ≤ 3 :=
sorry

end f_monotone_f_inequality_solution_l717_717056


namespace find_rate_percent_l717_717985

def P : ℝ := 800
def SI : ℝ := 200
def T : ℝ := 4

theorem find_rate_percent (R : ℝ) :
  SI = P * R * T / 100 → R = 6.25 :=
by
  sorry

end find_rate_percent_l717_717985


namespace determine_missing_digits_l717_717157

theorem determine_missing_digits :
  (237 * 0.31245 = 7430.65) := 
by 
  sorry

end determine_missing_digits_l717_717157


namespace algebraic_expression_value_l717_717255

-- Definitions based on conditions
def a : ℝ -- real number a
def b : ℝ -- real number b
def h : a + b = 3 -- condition a + b = 3

-- Theorem statement: given the condition, prove the expression equals 2
theorem algebraic_expression_value (a b : ℝ) (h : a + b = 3) : 2 * (a + 2 * b) - (3 * a + 5 * b) + 5 = 2 :=
by
  sorry

end algebraic_expression_value_l717_717255


namespace length_XZ_l717_717166

noncomputable def circle_radius : ℝ := 7
noncomputable def segment_length_XY : ℝ := 8
noncomputable def midpoint_minor_arc := true  -- Placeholder for the condition that Z is the midpoint of the minor arc XY.
noncomputable def midpoint_XZ := true  -- Placeholder for the condition that W is the midpoint of XZ.
noncomputable def length_YW : ℝ := 6

theorem length_XZ:
  (X Y Z : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z] 
  (dist_XY : dist X Y = segment_length_XY)
  (r : ℝ) (r_eq : r = circle_radius)
  (W_mid_XZ : midpoint_XZ)
  (YW_eq : dist Y W = length_YW)
  (Z_mid_arc_minor : midpoint_minor_arc) :
  dist X Z = 8 := sorry

end length_XZ_l717_717166


namespace solve_for_x_l717_717931

variable (x : ℝ)
axiom h : 3 / 4 + 1 / x = 7 / 8

theorem solve_for_x : x = 8 :=
by
  sorry

end solve_for_x_l717_717931


namespace number_of_boys_and_girls_l717_717471

-- Definitions used in Lean 4 statement based directly on the conditions
variable (b g : ℕ) -- number of boys and girls in the class

-- Conditions
variable (students : list (ℕ × ℕ)) 

-- Responses given
def responses := [(13, 11), (17, 11), (14, 14)]

-- Condition 1: Each child wrote one number correctly and made an error by exactly 2 in the other number
def possible_answers (b g : ℕ) : list (ℕ × ℕ) :=
  [(b - 2, g), (b + 2, g), (b, g - 2), (b, g + 2)]

-- A function to check if a given response could be one of the students' answers
def is_possible (response : ℕ × ℕ) : Prop :=
  ∃ b g, possible_answers b g = [response]

-- The final proof statement
theorem number_of_boys_and_girls (b g : ℕ) (students : list (ℕ × ℕ)) :
  students = responses → b = 15 ∧ g = 12 := by
  sorry

end number_of_boys_and_girls_l717_717471


namespace vector_dot_product_l717_717079

-- Define the vectors a, b, and c
def a := (1, -2 : ℤ × ℤ)
def b := (3, 4 : ℤ × ℤ)
def c := (2, -1 : ℤ × ℤ)

-- Define vector addition
def vector_add (u v : ℤ × ℤ) : ℤ × ℤ := (u.1 + v.1, u.2 + v.2)

-- Define dot product
def dot_product (u v : ℤ × ℤ) : ℤ := u.1 * v.1 + u.2 * v.2

-- Define the theorem to prove
theorem vector_dot_product :
  dot_product (vector_add a b) c = 6 :=
by
  sorry

end vector_dot_product_l717_717079


namespace isosceles_triangle_sin_cos_rational_l717_717040

variables {BC AD : ℕ} (h1 : BC > 0) (h2 : AD > 0)

theorem isosceles_triangle_sin_cos_rational (h_isosceles : ∀ A B C : Type, is_isosceles_triangle A B C BC AD) : 
  (∃ k : ℚ, sin A = (2 * k) / (1 + k^2) ∧ cos A = (1 - k^2) / (1 + k^2)) := 
sorry

end isosceles_triangle_sin_cos_rational_l717_717040


namespace find_added_number_l717_717775

theorem find_added_number (a : ℕ → ℝ) (x : ℝ) (h_init : a 1 = 2) (h_a3 : a 3 = 6)
  (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)  -- arithmetic sequence condition
  (h_geom : (a 4 + x)^2 = (a 1 + x) * (a 5 + x)) : 
  x = -11 := 
sorry

end find_added_number_l717_717775


namespace problem1_problem2_l717_717333

-- Proof statement for the first expression
theorem problem1 : (1 * (-3)^2 - (-1)^3 - (-2) - | -12 |) = 0 := by
  sorry

-- Proof statement for the second expression
theorem problem2 : (-2^2 * 3 * (-3/2) / (2/3) - 4 * ((-3/2)^2)) = 18 := by
  sorry

end problem1_problem2_l717_717333


namespace largest_abs_val_among_2_3_neg3_neg4_l717_717316

def abs_val (a : Int) : Nat := a.natAbs

theorem largest_abs_val_among_2_3_neg3_neg4 : 
  ∀ (x : Int), x ∈ [2, 3, -3, -4] → abs_val x ≤ abs_val (-4) := by
  sorry

end largest_abs_val_among_2_3_neg3_neg4_l717_717316


namespace botanical_garden_ratios_l717_717108

-- Define the given conditions
variables (R : ℕ) (C : ℕ) (L : ℕ) (T : ℕ)

-- Total number of roses
def roses_planted : ℕ := 15

-- Relationship conditions
def carnations_ratio : Prop := ∀ (n : ℕ), n = 3 → C = 2 * n * roses_planted / 3
def lilies_ratio : Prop := ∀ (n : ℕ), n = 3 → L = roses_planted / 2
def tulips_ratio : Prop := ∀ (n : ℕ), n = 3 → T = roses_planted

-- Aesthetic ratio theorem
theorem botanical_garden_ratios :
  (roses_planted = 15) ∧
  (carnations_ratio R C) ∧
  (lilies_ratio R L) ∧
  (tulips_ratio R T) →
  (C = 30) ∧ (L = 7 ∨ L = 8) ∧ (T = 15) :=
by
  sorry

end botanical_garden_ratios_l717_717108


namespace milk_remaining_after_process_l717_717286

noncomputable def remaining_milk (initial_volume : ℕ) (removals : list ℕ) : ℝ :=
  removals.foldl
    (λ (milk_left : ℝ) (removal : ℕ), milk_left * (1 - removal / initial_volume.to_real))
    initial_volume.to_real

theorem milk_remaining_after_process :
  remaining_milk 60 [5, 7, 9, 4] ≈ 38.525 :=
by sorry

end milk_remaining_after_process_l717_717286


namespace solve_for_x_l717_717540

theorem solve_for_x (x : ℝ) :
  5 * 3^x + sqrt (81 * 81^x) = 410 ↔ x = log (3 : ℝ) ((-5 + sqrt 14785) / 18) :=
by
  sorry

end solve_for_x_l717_717540


namespace remainder_71_73_div_8_l717_717264

theorem remainder_71_73_div_8 :
  (71 * 73) % 8 = 7 :=
by
  sorry

end remainder_71_73_div_8_l717_717264


namespace length_of_first_leg_of_triangle_l717_717302

theorem length_of_first_leg_of_triangle 
  (a b c : ℝ) 
  (h1 : b = 8) 
  (h2 : c = 10) 
  (h3 : c^2 = a^2 + b^2) : 
  a = 6 :=
by
  sorry

end length_of_first_leg_of_triangle_l717_717302


namespace relationship_among_a_b_c_l717_717889

noncomputable def a : ℝ := 0.8 ^ 0.7
noncomputable def b : ℝ := 0.8 ^ 0.9
noncomputable def c : ℝ := 1.2 ^ 0.8

theorem relationship_among_a_b_c : c > a ∧ a > b :=
by
  sorry

end relationship_among_a_b_c_l717_717889


namespace problem_statement_l717_717063

noncomputable def general_term (a : ℝ) (x : ℝ) (r : ℕ) : ℝ :=
  a^(6-r) * (-1)^r * (Nat.choose 6 r) * x^((3/2 : ℝ) * r - 6)

theorem problem_statement (a : ℝ) (h_a_pos : a > 0) 
(h_constant_term : general_term a 1 4 = 60) :
  a = 2 ∧ general_term 2 1 5 = -12 ∧ general_term 2 1 0 = 128 ∧ general_term 2 1 6 = 64 :=
by
  split
  sorry  -- Prove that a = 2
  split
  sorry  -- Prove that coefficient of the term containing x^(3/2) is -12
  sorry  -- Prove that rational terms are 60 and x^3

end problem_statement_l717_717063


namespace angle_C_is_pi_div_three_l717_717461

-- Definitions for the problem
variables {α : Type} [linear_ordered_field α] {a b c : α} {A B C : ℝ}

-- Assuming we have a proof that (a + c) * (sin A - sin C) = b * (sin A - sin B)
axiom equation : (a + c) * (Real.sin A - Real.sin C) = b * (Real.sin A - Real.sin B)

-- Prove that angle C is π / 3, given the conditions
theorem angle_C_is_pi_div_three (h : (a + c) * (Real.sin A - Real.sin C) = b * (Real.sin A - Real.sin B)) :
  C = π / 3 :=
by 
  sorry -- skip the proof

end angle_C_is_pi_div_three_l717_717461


namespace find_QS_of_triangle_l717_717545

theorem find_QS_of_triangle
  (Q R S : Type)
  [InnerProductSpace ℝ Q]
  [InnerProductSpace ℝ S]
  [InnerProductSpace ℝ R]
  (cosR : ℝ)
  (RS QR QS : ℝ)
  (hcosR : cosR = 3 / 5)
  (hRS : RS = 5)
  (Tr : QR*QR + RS*RS = QS*QS) :
  QS = Real.sqrt 34 :=
by
  have hQR : QR = 3 := by
    sorry
  rw [hRS, hQR] at Tr
  have hQS : QS = Real.sqrt (3*3 + 5*5) := by
    sorry
  simp at hQS
  exact hQS

end find_QS_of_triangle_l717_717545


namespace remainder_of_power_mod_l717_717340

theorem remainder_of_power_mod (a n p : ℕ) (h_prime : Nat.Prime p) (h_a : a < p) :
  (3 : ℕ)^2024 % 17 = 13 :=
by
  sorry

end remainder_of_power_mod_l717_717340


namespace product_has_correct_sign_and_units_digit_l717_717354

noncomputable def product_negative_integers_divisible_by_3_less_than_198 : ℤ :=
  sorry

theorem product_has_correct_sign_and_units_digit :
  product_negative_integers_divisible_by_3_less_than_198 < 0 ∧
  product_negative_integers_divisible_by_3_less_than_198 % 10 = 6 :=
by
  sorry

end product_has_correct_sign_and_units_digit_l717_717354


namespace find_cos_alpha_l717_717791

   variable (α : ℝ)

   -- Assumptions based on given conditions
   def condition1 : Prop := sin (π - α) = - (2 * sqrt 2) / 3
   def condition2 : Prop := α > π ∧ α < 3 * π / 2

   -- Proof goal
   theorem find_cos_alpha (h1 : condition1 α) (h2 : condition2 α) : cos α = - 1 / 3 :=
   by
     -- proof to be completed
     sorry
   
end find_cos_alpha_l717_717791


namespace mean_age_DeBose_family_l717_717549

theorem mean_age_DeBose_family:
  let ages := [8, 8, 16, 18] in
  (List.sum ages / ages.length : ℝ) = 12.5 :=
by
  sorry

end mean_age_DeBose_family_l717_717549


namespace midpoint_concurrence_l717_717385

variables {V : Type*} [inner_product_space ℝ V] 
variables (A B C P : V)

def L : V := C + B - P
def M : V := A + C - P
def N : V := B + A - P

theorem midpoint_concurrence :
  ∃ S : V, 
    S = (A + L) / 2 ∧
    S = (B + M) / 2 ∧
    S = (C + N) / 2 :=
begin
  use (A + B + C - P) / 2,
  split,
  { rw [L, add_comm C B, ←add_assoc, add_sub_cancel], },
  split,
  { rw [M, add_comm A C, add_assoc, symm (sub_add_cancel)], },
  { rw [N, add_comm A B, add_assoc, symm (add_sub_cancel)] }
end

end midpoint_concurrence_l717_717385


namespace smallest_lambda_is_one_over_two_n_l717_717032

theorem smallest_lambda_is_one_over_two_n (n : ℕ) (hn : 0 < n) :
  ∃ (λ : ℝ), λ = 1 / (2 * n) ∧ 
  (∀ (a : Fin n → ℝ), (∀ i, 0 ≤ a i ∧ a i ≤ 1) → 
    ∀ (x : Fin n → ℝ), (∀ i j, i ≤ j → x i ≤ x j) ∧ (∀ i, 0 ≤ x i ∧ x i ≤ 1) →
      (∃ i, |x i - a i| ≤ λ)) :=
by
  sorry

end smallest_lambda_is_one_over_two_n_l717_717032


namespace height_relationship_l717_717981

-- Define the variables representing the radii and heights of the cylinders
variables {r₁ r₂ h₁ h₂ : ℝ}

-- Define the conditions
def cylinder_volumes_equal (r₁ r₂ h₁ h₂ : ℝ) : Prop :=
  π * r₁^2 * h₁ = π * r₂^2 * h₂

def radius_relationship (r₁ r₂ : ℝ) : Prop :=
  r₂ = 1.2 * r₁

-- Define the theorem we want to prove
theorem height_relationship (r₁ r₂ h₁ h₂ : ℝ) 
  (h_volumes_equal : cylinder_volumes_equal r₁ r₂ h₁ h₂)
  (h_radius : radius_relationship r₁ r₂) : 
  h₁ = 1.44 * h₂ :=
begin
  sorry
end

end height_relationship_l717_717981


namespace main_l717_717894

noncomputable def r_diff_s_is_14 : Prop :=
  let f : ℝ → ℝ := λ x, (x-5)*(x+5) - (24*x - 120)
  let solutions := {x : ℝ | f x = 0}
  ∃ r s : ℝ, r ∈ solutions ∧ s ∈ solutions ∧ r ≠ s ∧ r > s ∧ r - s = 14

theorem main : r_diff_s_is_14 :=
sorry

end main_l717_717894


namespace length_of_AB_l717_717837

theorem length_of_AB (A B C : Point) (h₁ : B ≠ C) (h₂ : dist B C = 8) (h₃ : dist A C = 5)
  (h₄ : is_perpendicular (median A B C) (altitude B A C)) : 
  dist A B = 3 := 
sorry

end length_of_AB_l717_717837


namespace find_possible_values_of_p_l717_717885

noncomputable def T (a : Fin 51 → 𝔽_p) [Field 𝔽_p] : 𝔽_p :=
  ∑ i, a i

theorem find_possible_values_of_p {𝔽_p : Type} [Field 𝔽_p] [CharP 𝔽_p p]
  (a : Fin 51 → 𝔽_p) (h_non_zero : ∀ i, a i ≠ 0)
  (h_permutation : ∀ i, ∃ j, j ≠ i ∧ (∑ (k : Fin 51), a k) - a i = a j) :
  p = 2 ∨ p = 7 := 
sorry

end find_possible_values_of_p_l717_717885


namespace smallest_multiple_of_6_and_15_l717_717758

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ c : ℕ, c > 0 ∧ c % 6 = 0 ∧ c % 15 = 0 → c ≥ b := 
begin
  use 30,
  split,
  { exact nat.succ_pos 29, },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 2 3) (dvd_mul_right 3 5)), },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 3 5) (dvd_mul_right 3 2)), },
  { intros c hc1 hc2,
    have hc3 : c % 30 = 0,
    {
      suffices h : c % 6 = 0 ∧ c % 15 = 0 ↔ c % lcm 6 15 = 0,
      { rw ← h, exact ⟨hc1, hc2⟩, },
      exact nat.dvd_iff_mod_eq_zero,
    },
    linarith,
  }
end

end smallest_multiple_of_6_and_15_l717_717758


namespace blake_initial_amount_l717_717329

theorem blake_initial_amount (X : ℝ) (h1 : X > 0) (h2 : 3 * X / 2 = 30000) : X = 20000 :=
sorry

end blake_initial_amount_l717_717329


namespace grasshopper_trap_l717_717156

-- Define what it means for P to be in the form a * 2^(-k)
def suitable_form (P : ℝ) : Prop :=
  ∃ a k : ℕ, 0 < a ∧ a < 2^k ∧ odd a ∧ P = a * (2:ℝ)^(-k)

-- Define the main proposition
theorem grasshopper_trap (P : ℝ) :
  (∀ A : list ℝ, ∀ choose : ℕ → ℝ → bool, ∀ n : ℕ, ∃ i < n, (0 = i ∨ 1 = i)) 
  ↔ suitable_form P :=
sorry

end grasshopper_trap_l717_717156


namespace slope_of_tangent_at_1_l717_717416

variable {f : ℝ → ℝ}

-- Given conditions
axiom has_deriv_at_f : has_deriv_at f (f' : ℝ → ℝ) 1
axiom limit_cond : tendsto (fun (Δx : ℝ) => (f 1 - f (1 + 2 * Δx)) / Δx) (nhds 0) (nhds 2)

-- Proof problem statement
theorem slope_of_tangent_at_1 : deriv f 1 = -1 := 
sorry

end slope_of_tangent_at_1_l717_717416


namespace expression_result_is_seven_l717_717859

theorem expression_result_is_seven : ∃ p : ℕ → ℕ → ℚ, 
  p 10 9 = 10 / 9 ∧
  p (p 8 7) (p 6 (p 5 (p 4 (p 3 (p 2 1))))) = 7 := by
sorry

end expression_result_is_seven_l717_717859


namespace problem_statement_l717_717831

theorem problem_statement :
  ∀ (x y : ℝ), x = -3 → |y| = 5 → (x + y = 2 ∨ x + y = -8) :=
by
  intros x y hx hy
  rw hx at *
  cases abs_eq hy with hy1 hy2
  { sorry }
  { sorry }

end problem_statement_l717_717831


namespace add_base10_and_convert_to_base4_l717_717605

def base10_to_base4 (n : ℕ) : ℕ :=
  let rec convert (n m : ℕ) : ℕ :=
    if m = 0 then 0
    else (n / m) * (10 ^ Mathlib.Nat.log 4 m) + convert (n % m) (m / 4)
  convert n 16

theorem add_base10_and_convert_to_base4 :
  base10_to_base4 (15 + 27) = 222 :=
by
  sorry

end add_base10_and_convert_to_base4_l717_717605


namespace find_x_h_eq_x_l717_717503

def h (x : ℝ) : ℝ := (3 * x + 54) / 5

theorem find_x_h_eq_x (x : ℝ) (h_def : ∀ x, h (5 * x - 3) = 3 * x + 9) (hx : h x = x) : x = 27 :=
sorry

end find_x_h_eq_x_l717_717503


namespace mariana_socks_probability_l717_717510

open Finset

theorem mariana_socks_probability :
  let colors := ({0, 1, 2, 3, 4} : Finset ℕ) in
  let pairs := colors.powerset.filter(λ s, s.card = 3) in
  let favorable_outcomes := pairs.card * (Finset.card (erase (singleton 2))) * 1 * 1 * 2 in
  let total_combinations := choose 10 5 in
  (favorable_outcomes : ℚ / total_combinations : ℚ) = 5 / 21 :=
by
  sorry

end mariana_socks_probability_l717_717510


namespace general_formula_a_sum_first_n_b_terms_l717_717060

-- Given a sequence a_n which is arithmetic and satisfies the conditions:
variable (a : ℕ → ℤ) (b : ℕ → ℤ)

axiom a_is_arithmetic : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d 
axiom a1_plus_a2_eq_neg4 : a 1 + a 2 = -4
axiom a7_minus_a5_eq_4 : a 7 - a 5 = 4

-- Sequence b_n defined as b_n = a_n + 3^n
axiom b_def : ∀ n : ℕ, b n = a n + (3 ^ n)

-- General formula for a_n and the sum of first n terms of b_n
def a_general_formula : ℕ → ℤ := λ n, 2 * n - 5 
def S (n : ℕ) : ℤ := (n ^ 2) - 4 * n + (3 ^ (n + 1)) / 2 - 1.5

-- Proof obligations as Lean theorems:
theorem general_formula_a (n : ℕ) : a n = 2 * n - 5 :=
  sorry

theorem sum_first_n_b_terms (n : ℕ) : ∑ i in finset.range n, b (i + 1) = ((n ^ 2) - 4 * n + (3 ^ (n + 1)) / 2 - 1.5) :=
  sorry

end general_formula_a_sum_first_n_b_terms_l717_717060


namespace parabola_distance_l717_717087

theorem parabola_distance 
(P F : ℝ × ℝ)
(hP : P.2 ^ 2 = 4 * P.1)
(hF : F = (1, 0))
(h_dist : real.sqrt ((P.1 - F.1) ^ 2 + (P.2 - F.2) ^ 2) = 9) : 
  abs P.1 = 8 :=
by 
  sorry

end parabola_distance_l717_717087


namespace count_isosceles_triangles_l717_717106

-- Definitions and conditions
def is_geoboard_6x6 (x y : ℕ) : Prop := x <= 5 ∧ y <= 5

def point_on_horizontal_segment_DE (x y : ℕ) : Prop :=
  (x = 1 ∨ x = 4) ∧ y = 2

def distance (p1 p2 : ℕ × ℕ) : ℕ :=
  nat.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def is_isosceles_triangle (d e f : ℕ × ℕ) : Prop :=
  let d_e := distance d e in
  let d_f := distance d f in
  let e_f := distance e f in
  d_e = d_f ∨ d_f = e_f ∨ e_f = d_e

-- Main theorem statement
theorem count_isosceles_triangles :
  ∃ count : ℕ, count = 4 ∧
  ∀ f, is_geoboard_6x6 f.1 f.2 ∧ ¬ point_on_horizontal_segment_DE f.1 f.2 →
  (is_isosceles_triangle (1, 2) (4, 2) f) →
  ∃ (count : ℕ), count = 4 :=
by
  sorry

end count_isosceles_triangles_l717_717106


namespace range_of_log_function_l717_717962

noncomputable def f (x : ℝ) : ℝ := log (x^2 - 4*x + 5)

theorem range_of_log_function : set.range f = set.Ici 0 := by 
  sorry -- This will include the necessary proof steps, which are omitted.

end range_of_log_function_l717_717962


namespace number_of_red_items_l717_717576

-- Define the mathematics problem
theorem number_of_red_items (R : ℕ) : 
  (23 + 1) + (11 + 1) + R = 66 → 
  R = 30 := 
by 
  intro h
  sorry

end number_of_red_items_l717_717576


namespace mass_of_man_l717_717257

def boat_length : ℝ := 3 -- boat length in meters
def boat_breadth : ℝ := 2 -- boat breadth in meters
def boat_sink_depth : ℝ := 0.01 -- boat sink depth in meters
def water_density : ℝ := 1000 -- density of water in kg/m^3

/- Theorem: The mass of the man is equal to 60 kg given the parameters defined above. -/
theorem mass_of_man : (water_density * (boat_length * boat_breadth * boat_sink_depth)) = 60 :=
by
  simp [boat_length, boat_breadth, boat_sink_depth, water_density]
  sorry

end mass_of_man_l717_717257


namespace area_of_shaded_region_l717_717006

theorem area_of_shaded_region :
  let v1 := (0, 0)
  let v2 := (15, 0)
  let v3 := (45, 30)
  let v4 := (45, 45)
  let v5 := (30, 45)
  let v6 := (0, 15)
  let area_large_rectangle := 45 * 45
  let area_triangle1 := 1 / 2 * 15 * 15
  let area_triangle2 := 1 / 2 * 15 * 15
  let shaded_area := area_large_rectangle - (area_triangle1 + area_triangle2)
  shaded_area = 1800 :=
by
  sorry

end area_of_shaded_region_l717_717006


namespace problem_max_m_and_a_l717_717069

theorem problem_max_m_and_a (f g : ℝ → ℝ) (x y z m t a : ℝ)
    (h_f : ∀ x, f x = |x + 3|)
    (h_g : ∀ x, g x = m - 2 * |x - 11|)
    (h_ineq : ∀ x, 2 * f x ≥ g (x + 4))
    (h_eq : 2 * x^2 + 3 * y^2 + 6 * z^2 = a) 
    (h_max_sum : (x + y + z) ≤ 20) :
    t = 20 ∧ a = 1 :=
begin
  sorry
end

end problem_max_m_and_a_l717_717069


namespace maximum_radius_l717_717435

open Set Real

-- Definitions of sets M, N, and D_r.
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd ≥ 1 / 4 * p.fst^2}

def N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd ≤ -1 / 4 * p.fst^2 + p.fst + 7}

def D_r (x₀ y₀ r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.fst - x₀)^2 + (p.snd - y₀)^2 ≤ r^2}

-- Theorem statement for the largest r
theorem maximum_radius {x₀ y₀ : ℝ} (H : D_r x₀ y₀ r ⊆ M ∩ N) :
  r = sqrt ((25 - 5 * sqrt 5) / 2) :=
sorry

end maximum_radius_l717_717435


namespace modular_inverse_l717_717356

/-- Define the number 89 -/
def a : ℕ := 89

/-- Define the modulus 90 -/
def n : ℕ := 90

/-- The condition given in the problem -/
lemma pow_mod (h : a ≡ -1 [MOD n]) : (a * a) % n = 1 % n := by 
  sorry

/-- The main statement to prove the modular inverse -/
theorem modular_inverse (h : a ≡ -1 [MOD n]) : (a * a) % n = 1 % n → a ≡ a⁻¹ [MOD n] := by
  intro h1
  have h2 : a⁻¹ % n = a % n := by 
    sorry
  exact h2

end modular_inverse_l717_717356


namespace triangle_BC_length_tan_2B_value_l717_717836

noncomputable def triangle_AB_length : Float := 6
noncomputable def triangle_AC_length : Float := 3 * Real.sqrt 2
noncomputable def dot_product_AB_AC : Float := -18

theorem triangle_BC_length
  (AB : Float)
  (AC : Float)
  (dot_product : Float)
  (h_AB : AB = triangle_AB_length)
  (h_AC : AC = triangle_AC_length)
  (h_dot_product : dot_product = dot_product_AB_AC) :
  Real.sqrt (AB^2 + AC^2 - 2 * AB * AC * -((dot_product / (AB * AC)))) = 3 * Real.sqrt 10 := sorry

theorem tan_2B_value
  (AB : Float)
  (AC : Float)
  (dot_product : Float)
  (h_AB : AB = triangle_AB_length)
  (h_AC : AC = triangle_AC_length)
  (h_dot_product : dot_product = dot_product_AB_AC) :
  let A := Real.acos (dot_product / (AB * AC)) in
  let a := Real.sqrt (AB^2 + AC^2 - 2 * AB * AC * -((dot_product / (AB * AC)))) in
  let b := AC in
  let c := AB in
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c) in
  let sin_B := Real.sqrt (1 - cos_B^2) in
  let tan_B := sin_B / cos_B in
  (2 * tan_B) / (1 - tan_B^2) = 3 / 4 := sorry

end triangle_BC_length_tan_2B_value_l717_717836


namespace complex_number_real_complex_number_pure_imaginary_l717_717700

-- Define the given complex number as a function of m
def complex_number (m : ℝ) : ℂ := (m^2 - 5*m + 6) + (m^2 - 3*m) * complex.I

-- Statement for part (I): The complex number is real if and only if m = 0 or m = 3
theorem complex_number_real (m : ℝ) :
  (∃ x : ℝ, complex_number(m) = complex.of_real x) ↔ (m = 0 ∨ m = 3) :=
by sorry

-- Statement for part (II): The complex number is purely imaginary if and only if m = 2
theorem complex_number_pure_imaginary (m : ℝ) :
  (∃ y : ℝ, complex_number(m) = y * complex.I ∧ y ≠ 0) ↔ (m = 2) :=
by sorry

end complex_number_real_complex_number_pure_imaginary_l717_717700


namespace problem_min_a2_area_l717_717478

noncomputable def area (a b c : ℝ) (A B C : ℝ) : ℝ := 
  0.5 * b * c * Real.sin A

noncomputable def min_a2_area (a b c : ℝ) (A B C : ℝ): ℝ := 
  let S := area a b c A B C
  a^2 / S

theorem problem_min_a2_area :
  ∀ (a b c A B C : ℝ), 
    a > 0 → b > 0 → c > 0 → 
    A + B + C = Real.pi →
    a / Real.sin A = b / Real.sin B ∧ a / Real.sin A = c / Real.sin C →
    b * Real.cos C + c * Real.cos B = 3 * a * Real.cos A →
    min_a2_area a b c A B C ≥ 2 * Real.sqrt 2 :=
by
  sorry

end problem_min_a2_area_l717_717478


namespace percent_decrease_l717_717492

-- Definitions based on conditions
def originalPrice : ℝ := 100
def salePrice : ℝ := 10

-- The percentage decrease is the main statement to prove
theorem percent_decrease : ((originalPrice - salePrice) / originalPrice) * 100 = 90 := 
by
  -- Placeholder for proof
  sorry

end percent_decrease_l717_717492


namespace negation_of_proposition_l717_717380

noncomputable def original_proposition :=
  ∀ a b : ℝ, (a * b = 0) → (a = 0)

theorem negation_of_proposition :
  ¬ original_proposition ↔ ∃ a b : ℝ, (a * b = 0) ∧ (a ≠ 0) :=
by
  sorry

end negation_of_proposition_l717_717380


namespace sampling_is_systematic_l717_717293

-- Defining the conditions
def mock_exam (rooms students_per_room seat_selected: ℕ) : Prop :=
  rooms = 80 ∧ students_per_room = 30 ∧ seat_selected = 15

-- Theorem statement
theorem sampling_is_systematic 
  (rooms students_per_room seat_selected: ℕ)
  (h: mock_exam rooms students_per_room seat_selected) : 
  sampling_method = "Systematic sampling" :=
sorry

end sampling_is_systematic_l717_717293


namespace find_cos_alpha_l717_717386

noncomputable def cos_alpha_satisfies_condition (α : ℝ) : Prop :=
  (1 - Real.cos α) / Real.sin α = 3

theorem find_cos_alpha (α : ℝ) 
  (h : cos_alpha_satisfies_condition α) : Real.cos α = -4 / 5 :=
by
  sorry

end find_cos_alpha_l717_717386


namespace angle_ACE_55_l717_717993

variables (A B C D E : Type)
variables [IsConvexQuadrilateral A B C D]
variables (h1 : AB < AD)
variables [ACBisectsBAD A B C D]
variables (h2 : ∠ ABD = 130)
variables [IsPointOnInteriorAD E A D]
variables (h3 : ∠ BAD = 40)
variables (h4 : BC = CD)
variables (h5 : CD = DE)

theorem angle_ACE_55 :
  ∠ ACE = 55 :=
sorry

end angle_ACE_55_l717_717993


namespace sqrt_equation_solution_l717_717448

theorem sqrt_equation_solution (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 := 
by 
  sorry

end sqrt_equation_solution_l717_717448


namespace discriminant_quadratic_eq_l717_717946

theorem discriminant_quadratic_eq : 
  let a := 1
  let b := -7
  let c := 4
  let Δ := b^2 - 4 * a * c
  Δ = 33 :=
by
  let a := 1
  let b := -7
  let c := 4
  let Δ := b^2 - 4 * a * c
  exact sorry

end discriminant_quadratic_eq_l717_717946


namespace field_length_to_width_ratio_l717_717223
-- Import the math library

-- Define the problem conditions and proof goal statement
theorem field_length_to_width_ratio (w : ℝ) (l : ℝ) (area_pond : ℝ) (area_field : ℝ) 
    (h_length : l = 16) (h_area_pond : area_pond = 64) 
    (h_area_relation : area_pond = (1/2) * area_field)
    (h_field_area : area_field = l * w) : l / w = 2 :=
by 
  -- Leaving the proof as an exercise
  sorry

end field_length_to_width_ratio_l717_717223


namespace abdul_largest_number_l717_717309

theorem abdul_largest_number {a b c d : ℕ} 
  (h1 : a + (b + c + d) / 3 = 17)
  (h2 : b + (a + c + d) / 3 = 21)
  (h3 : c + (a + b + d) / 3 = 23)
  (h4 : d + (a + b + c) / 3 = 29) :
  d = 21 :=
by sorry

end abdul_largest_number_l717_717309


namespace area_of_triangle_is_4_l717_717141

def semiMajorAxis : ℝ := 3
def semiMinorAxis : ℝ := 2
def focalDistance : ℝ := Real.sqrt (semiMajorAxis ^ 2 - semiMinorAxis ^ 2)

def F1 : ℝ × ℝ := (focalDistance, 0)
def F2 : ℝ × ℝ := (-focalDistance, 0)

variable (P : ℝ × ℝ) (hP : P.1 ^ 2 / 9 + P.2 ^ 2 / 4 = 1)
variable (hRatio : (Real.dist P F1) / (Real.dist P F2) = 2 / 1)

theorem area_of_triangle_is_4 :
  let PF1 := Real.dist P F1,
      PF2 := Real.dist P F2,
      F1F2 := 2 * focalDistance in
  PF1 + PF2 = 6 →
  PF1 = 4 ∧ PF2 = 2 →
  ∃ (a b : ℝ), a * a + b * b = F1F2 * F1F2 ∧ a = PF1 ∧ b = PF2 →
  1 / 2 * PF1 * PF2 = 4 :=
by {
  sorry
}

end area_of_triangle_is_4_l717_717141


namespace shaded_area_proof_l717_717476

-- Definitions based on the conditions
def pi : ℝ := Real.pi
def R : ℝ := 12  -- from the condition that the area of the largest circle is 144π
def r : ℝ := R / 3
def r_third : ℝ := r / 2

-- Areas of respective circles
def area_largest_circle : ℝ := pi * R^2
def area_smaller_circle : ℝ := pi * r^2
def area_third_circle : ℝ := pi * r_third^2

-- Shaded areas of respective circles
def shaded_area_largest_circle : ℝ := area_largest_circle / 2
def shaded_area_smaller_circle : ℝ := area_smaller_circle / 2
def shaded_area_third_circle : ℝ := area_third_circle / 2

-- Total shaded area
def total_shaded_area : ℝ := shaded_area_largest_circle + shaded_area_smaller_circle + shaded_area_third_circle

theorem shaded_area_proof : total_shaded_area = 82 * pi := by
  -- Proof goes here
  sorry

end shaded_area_proof_l717_717476


namespace fraction_add_eq_l717_717441

theorem fraction_add_eq (x y : ℝ) (hx : y / x = 3 / 7) : (x + y) / x = 10 / 7 :=
by
  sorry

end fraction_add_eq_l717_717441


namespace virginia_taught_fewer_years_l717_717588

-- Definitions based on conditions
variable (V A D : ℕ)

-- Dennis has taught for 34 years
axiom h1 : D = 34

-- Virginia has taught for 9 more years than Adrienne
axiom h2 : V = A + 9

-- Combined total of years taught is 75
axiom h3 : V + A + D = 75

-- Proof statement: Virginia has taught for 9 fewer years than Dennis
theorem virginia_taught_fewer_years : D - V = 9 :=
  sorry

end virginia_taught_fewer_years_l717_717588


namespace domain_of_sqrt_quadratic_l717_717556

open Set

def domain_of_f : Set ℝ := {x : ℝ | 2*x - x^2 ≥ 0}

theorem domain_of_sqrt_quadratic :
  domain_of_f = Icc 0 2 :=
by
  sorry

end domain_of_sqrt_quadratic_l717_717556


namespace units_digit_17_pow_2007_l717_717604

theorem units_digit_17_pow_2007 : (17^2007) % 10 = 3 :=
by sorry

end units_digit_17_pow_2007_l717_717604


namespace ellipse_hyperbola_tangent_l717_717949

variable {x y m : ℝ}

theorem ellipse_hyperbola_tangent (h : ∃ x y, x^2 + 9 * y^2 = 9 ∧ x^2 - m * (y + 1)^2 = 1) : m = 2 := 
by 
  sorry

end ellipse_hyperbola_tangent_l717_717949


namespace total_earnings_correct_l717_717310

-- Define the initial amounts in USD
def Lauryn_USD : ℝ := 2000
def GBP_to_USD : ℝ := 1.33
def JPY_to_USD : ℝ := 0.0090
def CAD_to_USD : ℝ := 0.75

-- Calculate amounts in local currencies
def Aurelia_USD : ℝ := 0.70 * Lauryn_USD
def Aurelia_GBP : ℝ := Aurelia_USD / GBP_to_USD

def Jackson_GBP : ℝ := 1.50 * Aurelia_GBP
def Jackson_JPY : ℝ := Jackson_GBP / GBP_to_USD / JPY_to_USD

def Maya_JPY : ℝ := 0.40 * Jackson_JPY
def Maya_CAD : ℝ := Maya_JPY * JPY_to_USD / CAD_to_USD

-- Convert all earnings to USD for total calculation
def Jackson_USD : ℝ := Jackson_GBP * GBP_to_USD
def Maya_USD : ℝ := Maya_CAD * CAD_to_USD

-- Total earnings in USD
def total_USD : ℝ := Lauryn_USD + Aurelia_USD + Jackson_USD + Maya_USD

-- Prove the total earnings in USD is $6985.47
theorem total_earnings_correct : total_USD = 6985.47 := by
  sorry

end total_earnings_correct_l717_717310


namespace probability_defective_unit_l717_717150

theorem probability_defective_unit (T : ℝ) 
  (P_A : ℝ := 9 / 1000) 
  (P_B : ℝ := 1 / 50) 
  (output_ratio_A : ℝ := 0.4)
  (output_ratio_B : ℝ := 0.6) : 
  (P_A * output_ratio_A + P_B * output_ratio_B) = 0.0156 :=
by
  sorry

end probability_defective_unit_l717_717150


namespace cases_in_1995_l717_717465

theorem cases_in_1995 (x : ℕ) :
  let initial_cases := 600000
  let cases_2000 := 600
  let rate_of_decrease_1970_1990 := (600000 - x) / 20
  let rate_of_decrease_1990_2000 := rate_of_decrease_1970_1990 / 2
  x = 120480 →
  rate_of_decrease_1990_2000 = (600000 - x) / 40 →
  120480 - 5 * ((600000 - 120480) / 40) = 60580 :=
begin
  sorry
end

end cases_in_1995_l717_717465


namespace turtle_reaches_waterhole_in_28_minutes_l717_717614

-- Definitions
constant x : ℝ
constant turtle_speed : ℝ := 1 / 30
constant lion1_time_to_waterhole : ℝ := 5
constant turtle_time_to_waterhole : ℝ := 30

-- Speeds of the Lion Cubs
def lion1_speed := x
def lion2_speed := 1.5 * x

-- Time for the lion cubs to meet
def meeting_time := lion1_time_to_waterhole / (1 + lion2_speed / lion1_speed)

-- Distance traveled by the turtle in the meeting time
def turtle_distance_covered := turtle_speed * meeting_time

-- Remaining distance for the turtle
def remaining_turtle_distance := 1 - turtle_distance_covered

-- Time for the turtle to cover the remaining distance
def turtle_remaining_time := remaining_turtle_distance * 30

-- Prove that the turtle takes 28 minutes after the meeting to reach the waterhole
theorem turtle_reaches_waterhole_in_28_minutes : turtle_remaining_time = 28 :=
by
  -- Placeholder for the actual proof
  sorry

end turtle_reaches_waterhole_in_28_minutes_l717_717614


namespace average_race_time_l717_717338

theorem average_race_time (carlos_time diego_half_time : ℝ) (diego_total_time average_time_seconds : ℝ) 
                          (Diego_finishes : diego_total_time = diego_half_time * 2)
                          (Total_time : carlos_time + diego_total_time = 8)
                          (Average_time : (carlos_time + diego_total_time) / 2 = 4)
                          (Conversion : 4 * 60 = average_time_seconds) :
  average_time_seconds = 240 :=
by
  -- Definitions and conditions
  have H1 : diego_total_time = 2.5 * 2 := Diego_finishes
  have H2 : carlos_time + diego_total_time = 8 := Total_time
  have H3 : (carlos_time + diego_total_time) / 2 = 4 := Average_time
  have H4 : 4 * 60 = average_time_seconds := Conversion
  -- Concluding the proof
  sorry

end average_race_time_l717_717338


namespace op_correct_l717_717960

-- Definition of the operation * for non-zero integers
def op (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 / b)

theorem op_correct (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 12) (h2 : a * b = 32) :
  op a b = 3 / 8 :=
by
  -- Proof, sorry for now
  sorry

end op_correct_l717_717960


namespace ab_cd_eq_30_l717_717450

theorem ab_cd_eq_30 (a b c d : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = 3) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 17) : 
  a * b + c * d = 30 :=
begin
  sorry
end

end ab_cd_eq_30_l717_717450


namespace area_of_circles_l717_717208

theorem area_of_circles (BD AC : ℝ) (hBD : BD = 6) (hAC : AC = 12) : 
  ∃ S : ℝ, S = 225 / 4 * Real.pi :=
by
  sorry

end area_of_circles_l717_717208


namespace tom_ate_half_of_remaining_slices_l717_717978

theorem tom_ate_half_of_remaining_slices : 
  let total_slices := 2 * 8,
      slices_given_to_jerry := 3 / 8 * total_slices,
      remaining_slices := total_slices - slices_given_to_jerry,
      slices_left := 5,
      slices_eaten := remaining_slices - slices_left
  in slices_eaten / remaining_slices = 1 / 2 :=
by
  let total_slices := 2 * 8
  let slices_given_to_jerry := 3 / 8 * total_slices
  let remaining_slices := total_slices - slices_given_to_jerry
  let slices_left := 5
  let slices_eaten := remaining_slices - slices_left
  have h1 : slices_eaten = 5, from sorry
  have h2 : remaining_slices = 10, from sorry
  rw [h1, h2]
  norm_num
  exact eq.refl 1 / 2

end tom_ate_half_of_remaining_slices_l717_717978


namespace closest_to_123_over_0_123_l717_717723

theorem closest_to_123_over_0_123 : (∃ x ∈ {100, 1000, 10000, 100000, 1000000}, abs (x - (123 / 0.123)) = infi (λ y ∈ {100, 1000, 10000, 100000, 1000000}, abs (y - (123 / 0.123)))) :=
by {
  let choices := {100, 1000, 10000, 100000, 1000000},
  let target := 123 / 0.123,
  let closest := 1000,
  use closest,
  simp only [choices, target, closest],
  sorry
}

end closest_to_123_over_0_123_l717_717723


namespace shiela_drawings_l717_717176

theorem shiela_drawings (neighbors : ℕ) (drawings_per_neighbor : ℕ) (total_drawings : ℕ) 
  (h1 : neighbors = 6) (h2 : drawings_per_neighbor = 9) : total_drawings = 54 :=
  by 
    have h : total_drawings = neighbors * drawings_per_neighbor := sorry
    rw [h1, h2] at h
    exact h
    -- Proof skipped with sorry.

end shiela_drawings_l717_717176


namespace balloons_lost_is_correct_l717_717490

def original_balloons : ℕ := 8
def current_balloons : ℕ := 6
def lost_balloons : ℕ := original_balloons - current_balloons

theorem balloons_lost_is_correct : lost_balloons = 2 := by
  sorry

end balloons_lost_is_correct_l717_717490


namespace eric_shares_erasers_l717_717726

theorem eric_shares_erasers (total_erasers : Nat) (erasers_per_friend : Nat) (eric_has_9306 : total_erasers = 9306) (each_friend_gets_94 : erasers_per_friend = 94) :
  total_erasers / erasers_per_friend = 99 :=
by
  rw [eric_has_9306, each_friend_gets_94]
  norm_num
  exact rfl

end eric_shares_erasers_l717_717726


namespace max_T_l717_717777

noncomputable def S (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  if h : 3 = n then 3 * a(3) + 2 * a(2) else sorry

axiom condition_S3 : ∀ a, S 3 a = 3 * a 3 + 2 * a 2
axiom condition_a4 : ∀ a, a 4 = 8

def a (n : ℕ) : ℕ := 8 * (1 / 2)^(n - 4)

def b (n : ℕ) : ℕ := log 2 (a n)

def T (n : ℕ) : ℕ := ∑ i in range n, b i

theorem max_T : ∀ n ≥ 6, n ≤ 7 → T n = T 7 := 
by sorry

end max_T_l717_717777


namespace cos_double_angle_l717_717769

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + 3 * Real.pi / 2) = 1 / 3) : 
  Real.cos (2 * α) = -7 / 9 := 
by 
  sorry

end cos_double_angle_l717_717769


namespace parallelogram_diagonals_congruent_triangles_l717_717555

theorem parallelogram_diagonals_congruent_triangles 
  (A B C D O : Type) [ordered_ring O]
  [is_parallelogram A B C D] 
  (hACBD : diagonal_intersection_point A C B D O) : 
  (triangle_congruent (A O D) (C O B)) ∧ (triangle_congruent (A O B) (C O D)) :=
by
  sorry

end parallelogram_diagonals_congruent_triangles_l717_717555


namespace overall_percentage_decrease_l717_717924

-- Define the initial pay cut percentages as given in the conditions.
def first_pay_cut := 5.25 / 100
def second_pay_cut := 9.75 / 100
def third_pay_cut := 14.6 / 100
def fourth_pay_cut := 12.8 / 100

-- Define the single shot percentage decrease we want to prove.
def single_shot_decrease := 36.73 / 100

-- Calculate the cumulative multiplier from individual pay cuts.
def cumulative_multiplier := 
  (1 - first_pay_cut) * (1 - second_pay_cut) * (1 - third_pay_cut) * (1 - fourth_pay_cut)

-- Statement: Prove the overall percentage decrease using cumulative multiplier is equal to single shot decrease.
theorem overall_percentage_decrease :
  1 - cumulative_multiplier = single_shot_decrease :=
by sorry

end overall_percentage_decrease_l717_717924


namespace range_of_a_l717_717782

def polynomial_solutions (a : ℝ) : Set ℝ := {x | x^2 + a * x + 1 = 0}

def A (a : ℝ) : Set ℝ := polynomial_solutions a

def B : Set ℝ := {1, 2}

theorem range_of_a (a : ℝ) : (A a ∪ B) = B ↔ (a ∈ Ico (-2 : ℝ) 2) :=
begin
  sorry
end

end range_of_a_l717_717782


namespace triangle_area_bounds_l717_717667

theorem triangle_area_bounds (t : ℝ) :
  let parabola := λ x : ℝ, x^2 - 4
  let vertex := (0, -4)
  let intersects := λ t : ℝ, (-(√(t + 4)), t), (√(t + 4), t)
  let base := 2 * (√(t + 4))
  let height := t + 4
  let area := (base * height) / 2
  t ∈ Icc (-4) 2 ↔ area ≤ 36 :=
by
  sorry

end triangle_area_bounds_l717_717667


namespace partA_partB_l717_717298

-- Part (a)
theorem partA (n k : ℕ) (h : n ≥ 2 * k + 2) : ∃ (s1 s2 : ℕ), s1 ≠ s2 ∧ ¬free_call s1 s2 :=
sorry

-- Part (b)
theorem partB (n k : ℕ) (h : n = 2 * k + 1) : ∃ (free_calls : list (ℕ × ℕ)), ∀ i j, (i ≠ j → (i, j) ∈ free_calls) :=
sorry

-- Definitions
def free_call (s1 s2 : ℕ) : Prop :=  -- Dummy definition, should be replaced with an actual one based on conditions
  s1 ≠ s2

end partA_partB_l717_717298


namespace infer_error_probability_l717_717271

variable (X Y : Type) -- Categorical variables X and Y.
variable (K2 : Type) -- Random variable K^2.
variable (k : K2) -- Observed value of random variable K^2.

-- The statement to be proved:
theorem infer_error_probability (h : k < some_threshold) :
  probability_of_error_in_inference "X is related to Y" > some_probability :=
sorry

end infer_error_probability_l717_717271


namespace units_digit_a2019_l717_717037

theorem units_digit_a2019 (a : ℕ → ℝ) (h₁ : ∀ n, a n > 0)
  (h₂ : a 2 ^ 2 + a 4 ^ 2 = 900 - 2 * a 1 * a 5)
  (h₃ : a 5 = 9 * a 3) : (3^(2018) % 10) = 9 := by
  sorry

end units_digit_a2019_l717_717037


namespace exists_triangle_BX_XY_DY_l717_717112

variables {A B C D P Q X Y : Type} [square: is_square A B C D] 
(P_on_BC : P ∈ segment B C) (Q_on_CD : Q ∈ segment C D)
(h_BP_CQ : distance B P = distance C Q)
(X_on_AP : X ∈ segment A P) (Y_on_AQ : Y ∈ segment A Q)
(h_X_ne_Y : X ≠ Y)

theorem exists_triangle_BX_XY_DY :
  ∃ (Δ : Triangle),
    Δ.sides = (distance B X, distance X Y, distance D Y) :=
sorry

end exists_triangle_BX_XY_DY_l717_717112


namespace inscribed_circle_in_quadrilateral_l717_717278

-- Given definitions related to the problem
structure Quadrilateral :=
  (a b c d: ℝ) -- sides of the quadrilateral
  (O: ℝ × ℝ) -- center of the circle
  (R: ℝ) -- radius of the circle
  (a_chord: ℝ) -- length of the chord cut out by the circle on each side of the quadrilateral

-- Problem: Prove that a circle can be inscribed in the quadrilateral given the conditions.
theorem inscribed_circle_in_quadrilateral (Q: Quadrilateral) (h_chords: 
    (∀ (P₁ P₂ P₃ P₄: ℝ × ℝ), 
      ∥P₁ - Q.O∥ = ∥P₂ - Q.O∥ ∧ ∥P₃ - Q.O∥ = ∥P₄ - Q.O∥ → 
      Q.a_chord = dist P₁ P₂ ∧ Q.a_chord = dist P₃ P₄)) : 
  ∃ (I: ℝ × ℝ), ∀ (P: ℝ × ℝ), P ≠ I → dist I P = Q.R := sorry

end inscribed_circle_in_quadrilateral_l717_717278


namespace sum_b_values_for_one_solution_l717_717012

def quadratic_eq_has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c = 0

theorem sum_b_values_for_one_solution :
  let eq := 3*x^2 + b*x + 6*x + 14
  in quadratic_eq_has_one_solution 3 (b + 6) 14 ∧
      (b = -6 + 2*real.sqrt 42 ∨ b = -6 - 2*real.sqrt 42) → 
      ( -6 + 2*(real.sqrt 42) + (-6 - 2*(real.sqrt 42)) = -12 ) :=
by
  sorry

end sum_b_values_for_one_solution_l717_717012


namespace locus_of_X_is_line_CD_l717_717163

open Real EuclideanGeometry

/-- Setting up the problem with given conditions -/
variables (A B C D P Q X : Point)
variables (circ_circle : Circle)
variables (AP_line BD_line AC_line BP_line : Line)
variables (locus_X : Set Point)

/-- Defining the square ABCD and its properties -/
def square (A B C D : Point) : Prop :=
  is_square A B C D

/-- Defining the circumcircle of the square -/
def circumscribed_circle (circ_circle : Circle) (A B C D : Point) : Prop :=
  circ_circle = circumcircle A B C D

/-- Point P moves along the circumcircle of square ABCD -/
def P_moves_along_circumcircle (circ_circle : Circle) (P : Point) : Prop :=
  P ∈ circ_circle

/-- Lines AP and BD intersect at point Q -/
def AP_BD_intersect_at_Q (AP_line BD_line : Line) (Q : Point) : Prop :=
  Q ∈ AP_line ∧ Q ∈ BD_line

/-- Line through Q parallel to AC intersects line BP at point X -/
def Q_parallel_AC_B_intersects_BP_at_X (Q : Point) (AC_line BP_line X : Point) : Prop :=
  parallel (line_through Q AC_line) AC_line ∧ X ∈ BP_line

/-- The locus of point X is the line CD -/
theorem locus_of_X_is_line_CD (A B C D P Q X : Point) (circ_circle : Circle) 
  (AP_line BD_line AC_line BP_line : Line) (locus_X : Set Point) :
  square A B C D →
  circumscribed_circle circ_circle A B C D →
  P_moves_along_circumcircle circ_circle P →
  AP_BD_intersect_at_Q AP_line BD_line Q →
  Q_parallel_AC_B_intersects_BP_at_X Q AC_line BP_line X →
  ∀ X ∈ locus_X, collinear {C, D, X} :=
sorry

end locus_of_X_is_line_CD_l717_717163


namespace euler_family_mean_age_l717_717196

theorem euler_family_mean_age : 
  let girls_ages := [5, 5, 10, 15]
  let boys_ages := [8, 12, 16]
  let children_ages := girls_ages ++ boys_ages
  let total_sum := List.sum children_ages
  let number_of_children := List.length children_ages
  (total_sum : ℚ) / number_of_children = 10.14 := 
by
  sorry

end euler_family_mean_age_l717_717196


namespace bobby_final_paycheck_is_148_l717_717685

def bobby_gross_salary : ℕ := 450
def federal_tax_rate : ℚ := 1 / 3
def state_tax_rate : ℚ := 0.08
def local_tax_rate : ℚ := 0.05
def health_insurance : ℕ := 50
def life_insurance : ℕ := 20
def city_parking_fee : ℕ := 10
def retirement_contribution_rate : ℚ := 0.03

def final_paycheck_amount : ℕ := bobby_gross_salary
                         - (federal_tax_rate * bobby_gross_salary).natAbs
                         - (state_tax_rate * bobby_gross_salary).natAbs
                         - (local_tax_rate * bobby_gross_salary).natAbs
                         - health_insurance
                         - life_insurance
                         - city_parking_fee
                         - (retirement_contribution_rate * bobby_gross_salary).natAbs

theorem bobby_final_paycheck_is_148 : final_paycheck_amount = 148 := sorry

end bobby_final_paycheck_is_148_l717_717685


namespace hawks_score_l717_717840

theorem hawks_score (E H : ℕ) (h1 : E + H = 82) (h2 : E = H + 22) : H = 30 :=
by
  sorry

end hawks_score_l717_717840


namespace skating_rink_visitors_by_noon_l717_717304

-- Defining the initial conditions
def initial_visitors : ℕ := 264
def visitors_left : ℕ := 134
def visitors_arrived : ℕ := 150

-- Theorem to prove the number of people at the skating rink by noon
theorem skating_rink_visitors_by_noon : initial_visitors - visitors_left + visitors_arrived = 280 := 
by 
  sorry

end skating_rink_visitors_by_noon_l717_717304


namespace area_of_triangle_DBC_l717_717110

-- Define the points A, B, C
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

-- Define the point D as the midpoint of AB
def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the point E as one-third of the way along BC from B to C
def E : ℝ × ℝ := (B.1 + (C.1 - B.1) / 3, B.2 + (C.2 - B.2) / 3)

-- Function to calculate the area of a triangle given vertices (x1, y1), (x2, y2), (x3, y3)
def triangle_area ((x1, y1) (x2, y2) (x3, y3) : ℝ × ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- The theorem statement for the proof problem
theorem area_of_triangle_DBC : triangle_area D B C = 20 :=
  sorry

end area_of_triangle_DBC_l717_717110


namespace decreasing_interval_l717_717097

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(abs (2*x - 4))

theorem decreasing_interval (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : f a 1 = 1/9) :
  ∃ (I : set ℝ), I = set.Ici 2 ∧ ∀ x y ∈ I, x ≤ y → f a x ≥ f a y := 
begin
  sorry
end

end decreasing_interval_l717_717097


namespace sum_of_brothers_ages_l717_717585

theorem sum_of_brothers_ages (Bill Eric: ℕ) 
  (h1: 4 = Bill - Eric) 
  (h2: Bill = 16) : 
  Bill + Eric = 28 := 
by 
  sorry

end sum_of_brothers_ages_l717_717585


namespace trapezoid_division_l717_717035

noncomputable def trapezoid_problem 
  (A B C D M E O N : Point)
  (BC AD : Segment)
  (h1 : AD.length = 2 * BC.length)
  (h2 : M = midpoint A D)
  (h3 : E = line_intersection (line A B) (line C D))
  (h4 : O = line_intersection (line B M) (line A C))
  (h5 : N = line_intersection (line E O) (line B C)) : Prop :=
  divides_segment N B C (1:2)

theorem trapezoid_division 
  (A B C D M E O N : Point)
  (BC AD : Segment)
  (h1 : AD.length = 2 * BC.length)
  (h2 : M = midpoint A D)
  (h3 : E = line_intersection (line A B) (line C D))
  (h4 : O = line_intersection (line B M) (line A C))
  (h5 : N = line_intersection (line E O) (line B C)) :
  divides_segment N B C (1:2) :=
sorry

end trapezoid_division_l717_717035


namespace solve_for_x_l717_717929

theorem solve_for_x (x : ℝ) (h : 3 / 4 + 1 / x = 7 / 8) : x = 8 :=
by
  sorry

end solve_for_x_l717_717929


namespace units_digit_of_17_pow_2007_l717_717601

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_2007 : units_digit (17 ^ 2007) = 3 := by
  have h : ∀ n, units_digit (17 ^ n) = units_digit (7 ^ n) := by
    intro n
    sorry  -- Same units digit logic for powers of 17 as for powers of 7.
  have pattern : units_digit (7 ^ 1) = 7 ∧ 
                 units_digit (7 ^ 2) = 9 ∧ 
                 units_digit (7 ^ 3) = 3 ∧ 
                 units_digit (7 ^ 4) = 1 := by
    sorry  -- Units digit pattern for powers of 7.
  have mod_cycle : 2007 % 4 = 3 := by
    sorry  -- Calculation of 2007 mod 4.
  have result : units_digit (7 ^ 2007) = units_digit (7 ^ 3) := by
    rw [← mod_eq_of_lt (by norm_num : 2007 % 4 < 4), mod_cycle]
    exact (and.left (and.right (and.right pattern)))  -- Extract units digit of 7^3 from pattern.
  rw [h]
  exact result

end units_digit_of_17_pow_2007_l717_717601


namespace product_base8_digits_l717_717593

   theorem product_base8_digits (n : ℕ) (h : n = 8927) :
     let base8_digits := [2, 1, 3, 3, 7] in
     base8_digits.prod = 126 :=
   by
     sorry
   
end product_base8_digits_l717_717593


namespace chord_length_correct_l717_717292

noncomputable def parabola_chord_length : ℝ :=
let focus := (0, 1) in
let line_slope := -1 in
let line_y_intercept := 1 in -- Derived from y = mx + c and focus (0, 1)
let line_eqn (x : ℝ) := line_slope * x + line_y_intercept in
let parabola_eqn (x : ℝ) := x^2 / 4 in
let intersection_pts := { x // parabola_eqn x = line_eqn x } in
let y_coords := { y | ∃ x, parabola_eqn x = y ∧ y = line_eqn x } in
let y_sum := (ˢup y_coords).get (set.nonempty_of_mem sorry) + (ˢup (set.diff y_coords {(ˢup y_coords).get (set.nonempty_of_mem sorry)})).get sorry in
y_sum + 2

theorem chord_length_correct : parabola_chord_length = 8 := sorry

end chord_length_correct_l717_717292


namespace inequality_pi_l717_717509

def seq_a (n : ℕ) : ℝ :=
  let a : ℕ → ℝ
  | 0 => real.sqrt 2 / 2
  | n + 1 => (real.sqrt 2 / 2) * real.sqrt (1 - real.sqrt (1 - (a n) ^ 2))
  a n

def seq_b (n : ℕ) : ℝ :=
  let b : ℕ → ℝ
  | 0 => 1
  | n + 1 => (real.sqrt (1 + (b n) ^ 2) - 1) / b n
  b n

theorem inequality_pi (n : ℕ) : 2^(n+2) * seq_a n < real.pi ∧ real.pi < 2^(n+2) * seq_b n :=
by
  sorry

end inequality_pi_l717_717509


namespace triangle_area_l717_717118

-- Define points in polar coordinates and the area of the triangle
def Point := (ℝ × ℝ)
def O : Point := (0, 0)
def A : Point := (3, Real.pi / 3)
def B : Point := (4, 5 * Real.pi / 6)

noncomputable def distance (p1 p2 : Point) : ℝ :=
  (p1.1 * p2.1 + p1.2 * p2.2).sqrt

noncomputable def angle (p1 p2 : Point) : ℝ :=
  p2.2 - p1.2

theorem triangle_area :
  let area := (1 / 2) * (distance O A) * (distance O B) * (Real.sin (angle A B)) in
  area = 6 :=
by
  sorry

end triangle_area_l717_717118


namespace inv_geom_seq_prod_next_geom_seq_l717_717059

variable {a : Nat → ℝ} (q : ℝ) (h_q : q ≠ 0)
variable (h_geom : ∀ n, a (n + 1) = q * a n)

theorem inv_geom_seq :
  ∀ n, ∃ c q_inv, (q_inv ≠ 0) ∧ (1 / a n = c * q_inv ^ n) :=
sorry

theorem prod_next_geom_seq :
  ∀ n, ∃ c q_sq, (q_sq ≠ 0) ∧ (a n * a (n + 1) = c * q_sq ^ n) :=
sorry

end inv_geom_seq_prod_next_geom_seq_l717_717059


namespace solution_to_problem_l717_717832

theorem solution_to_problem (x y : ℕ) (h : (2*x - 5) * (2*y - 5) = 25) : x + y = 10 ∨ x + y = 18 := by
  sorry

end solution_to_problem_l717_717832


namespace sum_of_Jo_numbers_l717_717130

theorem sum_of_Jo_numbers : 
  let seq := fun n => 2 * n - 1 
  let sum := (n: ℕ) → (finset.range n).sum seq
  sum 25 = 625 :=
by
  -- Proof is required here
  sorry

end sum_of_Jo_numbers_l717_717130


namespace susie_golden_comets_value_l717_717191

variables (G : ℕ)
variables (susie_rhode_island_reds susie_golden_comets : ℕ)
variables (britney_rhode_island_reds britney_golden_comets : ℕ)
variables (susie_total_chickens britney_total_chickens : ℕ)

-- Given conditions
def susie_has_11_rhode_island_reds : susie_rhode_island_reds = 11 := sorry
def britney_has_22_rhode_island_reds : britney_rhode_island_reds = 2 * susie_rhode_island_reds := sorry
def britney_has_half_as_many_golden_comets : britney_golden_comets = susie_golden_comets / 2 := sorry
def britney_has_8_more_chickens : britney_total_chickens = susie_total_chickens + 8 := sorry

-- Calculation of total chickens
def susie_total_chickens_def : susie_total_chickens = susie_rhode_island_reds + susie_golden_comets := sorry
def britney_total_chickens_def : britney_total_chickens = britney_rhode_island_reds + britney_golden_comets := sorry

-- The final proof statement that Susie has 6 Golden Comets
theorem susie_golden_comets_value : susie_golden_comets = 6 :=
by
  have h1 : susie_rhode_island_reds = 11 := susie_has_11_rhode_island_reds
  have h2 : britney_rhode_island_reds = 22 := britney_has_22_rhode_island_reds
  have h3 : britney_golden_comets = susie_golden_comets / 2 := britney_has_half_as_many_golden_comets
  have h4 : britney_total_chickens = susie_total_chickens + 8 := britney_has_8_more_chickens
  have stc : susie_total_chickens = susie_rhode_island_reds + susie_golden_comets := susie_total_chickens_def
  have btc : britney_total_chickens = britney_rhode_island_reds + britney_golden_comets := britney_total_chickens_def
  sorry

end susie_golden_comets_value_l717_717191


namespace fundraiser_total_money_l717_717377

def fundraiser_money : ℝ :=
  let brownies_students := 70
  let brownies_each := 20
  let brownies_price := 1.50
  let cookies_students := 40
  let cookies_each := 30
  let cookies_price := 2.25
  let donuts_students := 35
  let donuts_each := 18
  let donuts_price := 3.00
  let cupcakes_students := 25
  let cupcakes_each := 12
  let cupcakes_price := 2.50
  let total_brownies := brownies_students * brownies_each
  let total_cookies := cookies_students * cookies_each
  let total_donuts := donuts_students * donuts_each
  let total_cupcakes := cupcakes_students * cupcakes_each
  let money_brownies := total_brownies * brownies_price
  let money_cookies := total_cookies * cookies_price
  let money_donuts := total_donuts * donuts_price
  let money_cupcakes := total_cupcakes * cupcakes_price
  money_brownies + money_cookies + money_donuts + money_cupcakes

theorem fundraiser_total_money : fundraiser_money = 7440 := sorry

end fundraiser_total_money_l717_717377


namespace multiple_of_babu_l717_717186

theorem multiple_of_babu's_share:
  ∃ k B E : ℕ, 
  (12 * 84 = k * B) ∧
  (k * B = 6 * E) ∧
  (84 + B + E = 378) ∧
  (k = 4) :=
by
  -- Definitions and conditions
  let A := 84
  use 4
  use 1008 / 4
  use 168 / 4
  --Conditions stated explicitly
  have h1 : 12 * A = 4 * (1008 / 4), by simp,
  have h2 : 4 * (1008 / 4) = 6 * (168 / 4), by simp,
  have h3 : 84 + (1008 / 4) + (168 / 4) = 378, by norm_num,
  exact ⟨h1, h2, h3, rfl⟩

end multiple_of_babu_l717_717186


namespace b_is_geometric_sequence_sum_of_a_sequence_l717_717432

section sequence_problems

-- Define the sequence a_n
def a : ℕ+ → ℤ 
| 1 := 1
| (Nat.succ n) := 2 * (a n) + 3

-- Define the sequence b_n such that b_n = a_n + 3
def b (n : ℕ+) := a n + 3

-- The first statement: Prove that b is a geometric sequence
theorem b_is_geometric_sequence : ∀ n : ℕ+, b (Nat.succ n) = 2 * (b n) := 
sorry

-- The second statement: Find the sum of the first n terms of the sequence a_n, denoted as S_n
def S (n : ℕ) : ℤ := ∑ i in Finset.range n, a ⟨i+1, Nat.succ_pos i⟩

theorem sum_of_a_sequence : ∀ n : ℕ, S n = 2^(n+2) - 3 * n - 4 := 
sorry

end sequence_problems

end b_is_geometric_sequence_sum_of_a_sequence_l717_717432


namespace range_of_function_l717_717008

noncomputable def f (x : ℝ) : ℝ := 2 * x - real.sqrt (1 - 2 * x)

theorem range_of_function : set.range f = set.Iic 1 :=
begin
  sorry
end

end range_of_function_l717_717008


namespace volume_of_rotation_l717_717015

def volume_of_revolution :=
  let f (x : ℝ) := real.sqrt (2 * x) in
  π * ∫ x in 0..3, (f x) ^ 2

theorem volume_of_rotation (V : ℝ) :
  V = 9 * π
  :=
  V = volume_of_revolution

end volume_of_rotation_l717_717015


namespace common_chord_of_circles_is_x_eq_y_l717_717736

theorem common_chord_of_circles_is_x_eq_y :
  ∀ x y : ℝ, (x^2 + y^2 - 4 * x - 3 = 0) ∧ (x^2 + y^2 - 4 * y - 3 = 0) → (x = y) :=
by
  sorry

end common_chord_of_circles_is_x_eq_y_l717_717736


namespace part1_part2_l717_717067

def f (x : ℝ) : ℝ := Real.log (1 + x) + x^2 / 2
def g (x : ℝ) : ℝ := Real.cos x + x^2 / 2

theorem part1 (x : ℝ) (hx : 0 ≤ x) : f x ≥ x :=
by
  sorry

theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : f (Real.exp (a / 2)) = g b - 1) : f (b^2) + 1 > g (a + 1) :=
by
  sorry

end part1_part2_l717_717067


namespace function_identity_l717_717500

variables {R : Type*} [LinearOrderedField R]

-- Define real-valued functions f, g, h
variables (f g h : R → R)

-- Define function composition and multiplication
def comp (f g : R → R) (x : R) := f (g x)
def mul (f g : R → R) (x : R) := f x * g x

-- The statement to prove
theorem function_identity (x : R) : 
  comp (mul f g) h x = mul (comp f h) (comp g h) x :=
sorry

end function_identity_l717_717500


namespace most_N_atoms_in_CONH22_l717_717315

noncomputable def Avogadro : ℝ := 6.02 * 10^23

-- Define the number of nitrogen atoms in each substance
def N_NH4Cl (mol : ℝ) : ℝ := mol * Avogadro
def N_NH4NO3 (mol : ℝ) : ℝ := 2 * mol * Avogadro
def N_CONH22 (molecules : ℝ) : ℝ := 2 * molecules
def N_NH3H2O (mol : ℝ) : ℝ := mol * Avogadro

-- Given amounts
def mol_NH4Cl : ℝ := 0.1
def mol_NH4NO3 : ℝ := 0.1
def molecules_CONH22 : ℝ := 1.204 * 10^23
def mol_NH3H2O : ℝ := 0.2

-- Number of nitrogen atoms in each substance
def num_NH4Cl : ℝ := N_NH4Cl mol_NH4Cl
def num_NH4NO3 : ℝ := N_NH4NO3 mol_NH4NO3
def num_CONH22 : ℝ := N_CONH22 molecules_CONH22
def num_NH3H2O : ℝ := N_NH3H2O mol_NH3H2O

-- Theorem stating that CO(NH2)2 has the most nitrogen atoms
theorem most_N_atoms_in_CONH22 :
  num_CONH22 > num_NH4Cl ∧ num_CONH22 > num_NH4NO3 ∧ num_CONH22 > num_NH3H2O :=
by
  sorry

end most_N_atoms_in_CONH22_l717_717315


namespace simplification_identity_l717_717702

theorem simplification_identity :
  ((5:ℕ) ^ 2010) ^ 2 - ((5:ℕ) ^ 2008) ^ 2) / (((5:ℕ) ^ 2009) ^ 2 - ((5:ℕ) ^ 2007) ^ 2) = 25 :=
by
  sorry

end simplification_identity_l717_717702


namespace expected_composite_selection_l717_717923

noncomputable def expected_composite_count : ℚ :=
  let total_numbers := 100
  let composite_numbers := 74
  let p := (composite_numbers : ℚ) / total_numbers
  let n := 5
  n * p

theorem expected_composite_selection :
  expected_composite_count = 37 / 10 := by
  sorry

end expected_composite_selection_l717_717923


namespace find_length_BC_2sqrt2_l717_717122

-- Let’s define the conditions stated in a)
variables {A B C D : Type} [Geometry]
variables {AB BC CD DA : ℝ} 
variables {alpha beta : ℝ}
variables {area_ABCD : ℝ}

-- Assume the given conditions
axiom parallel_AD_BC: ∥(A, D)∥ = ∥(B, C)∥
axiom angle_A_45: α = 45
axiom angle_D_45: β = 45
axiom angle_B_135: α = 135
axiom angle_C_135: β = 135
axiom length_AB_6: length_AB = 6
axiom area_ABCD_30: area_ABCD = 30

-- State the proof problem
theorem find_length_BC_2sqrt2 :
  ∃ (BC : ℝ), BC = 2 * sqrt 2 :=
sorry

end find_length_BC_2sqrt2_l717_717122


namespace shape_formed_is_graph_l717_717527

noncomputable def independent_variable : Type := ℝ
noncomputable def dependent_variable : Type := ℝ

structure function_graph (independent : independent_variable) (dependent : dependent_variable) :=
(points : set (independent × dependent))

theorem shape_formed_is_graph :
  ∀ (independent: independent_variable) (dependent: dependent_variable), 
  (∀ p : function_graph independent dependent, p = p) :=
by sorry

end shape_formed_is_graph_l717_717527


namespace sum_of_corners_9x9_checkerboard_l717_717937

theorem sum_of_corners_9x9_checkerboard : 
  let topLeft := 1 in 
  let topRight := 9 in 
  let bottomRight := 81 in 
  let bottomLeft := 73 in 
  topLeft + topRight + bottomRight + bottomLeft = 164 :=
by
  sorry

end sum_of_corners_9x9_checkerboard_l717_717937


namespace turtle_reaches_waterhole_28_minutes_after_meeting_l717_717611

theorem turtle_reaches_waterhole_28_minutes_after_meeting (x : ℝ) (distance_lion1 : ℝ := 5 * x) 
  (speed_lion2 : ℝ := 1.5 * x) (distance_turtle : ℝ := 30) (speed_turtle : ℝ := 1/30) : 
  ∃ t_meeting : ℝ, t_meeting = 2 ∧ (distance_turtle - speed_turtle * t_meeting) / speed_turtle = 28 :=
by 
  sorry

end turtle_reaches_waterhole_28_minutes_after_meeting_l717_717611


namespace marbles_shared_equally_l717_717977

def initial_marbles_Wolfgang : ℕ := 16
def additional_fraction_Ludo : ℚ := 1/4
def fraction_Michael : ℚ := 2/3

theorem marbles_shared_equally :
  let marbles_Wolfgang := initial_marbles_Wolfgang
  let additional_marbles_Ludo := additional_fraction_Ludo * initial_marbles_Wolfgang
  let marbles_Ludo := initial_marbles_Wolfgang + additional_marbles_Ludo
  let marbles_Wolfgang_Ludo := marbles_Wolfgang + marbles_Ludo
  let marbles_Michael := fraction_Michael * marbles_Wolfgang_Ludo
  let total_marbles := marbles_Wolfgang + marbles_Ludo + marbles_Michael
  let marbles_each := total_marbles / 3
  marbles_each = 20 :=
by
  sorry

end marbles_shared_equally_l717_717977


namespace probability_of_second_less_than_first_is_two_fifths_l717_717633

variable (Ω : Type) [Fintype Ω] [UniformProbabilityMeasure Ω]

-- Define the event spaces
noncomputable def draws : List (ℕ × ℕ) := [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                                            (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
                                            (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
                                            (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
                                            (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]

noncomputable def event_count (event : (ℕ × ℕ) → Prop) : ℕ :=
  (draws.filter event).length

-- Define the specific event
def second_less_than_first (pair: ℕ × ℕ) : Prop := pair.2 < pair.1

-- Calculate the probability
noncomputable def probability_second_less_than_first : ℚ :=
  (event_count Ω second_less_than_first : ℚ) / (Fintype.card Ω * Fintype.card Ω : ℚ)

theorem probability_of_second_less_than_first_is_two_fifths :
  probability_second_less_than_first = 2 / 5 := by
  sorry

end probability_of_second_less_than_first_is_two_fifths_l717_717633


namespace smallest_multiple_l717_717748

theorem smallest_multiple (b : ℕ) (h1 : b % 6 = 0) (h2 : b % 15 = 0) (h3 : ∀ n : ℕ, (n % 6 = 0 ∧ n % 15 = 0) → n ≥ b) : b = 30 :=
sorry

end smallest_multiple_l717_717748


namespace deepak_present_age_l717_717571

-- Let R be Rahul's current age and D be Deepak's current age
variables (R D : ℕ)

-- Given conditions
def ratio_condition : Prop := (4 : ℚ) / 3 = (R : ℚ) / D
def rahul_future_age_condition : Prop := R + 6 = 50

-- Prove Deepak's present age D is 33 years
theorem deepak_present_age : ratio_condition R D ∧ rahul_future_age_condition R → D = 33 := 
sorry

end deepak_present_age_l717_717571


namespace remainder_when_multiplied_mod_500_l717_717250

-- Given conditions as definitions
def cond1 : ℤ := 1502 % 500
def cond2 : ℤ := 2021 % 500
def targetProd : ℤ := (2 * 21) % 500

theorem remainder_when_multiplied_mod_500 :
  cond1 = 2 ∧ cond2 = 21 → (1502 * 2021) % 500 = 42 :=
by {
  -- cond1 and cond2 assumptions
  assume h : cond1 = 2 ∧ cond2 = 21,
  have h1 : 1502 % 500 = 2, from h.left,
  have h2 : 2021 % 500 = 21, from h.right,
  -- sorry placeholder for actual proof
  sorry
}

end remainder_when_multiplied_mod_500_l717_717250


namespace bobs_walking_rate_l717_717256

theorem bobs_walking_rate (distance_XY : ℕ) 
  (yolanda_rate : ℕ) 
  (bob_distance_when_met : ℕ) 
  (yolanda_extra_hour : ℕ)
  (meet_covered_distance : distance_XY = yolanda_rate * (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1 + bob_distance_when_met / bob_distance_when_met)) 
  (yolanda_distance_when_met : yolanda_rate * (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1) + bob_distance_when_met = distance_XY) 
  : 
  (bob_distance_when_met / (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1) = yolanda_rate) :=
  sorry

end bobs_walking_rate_l717_717256


namespace find_angle_4_l717_717024

def angle_sum_180 (α β : ℝ) : Prop := α + β = 180
def angle_equality (γ δ : ℝ) : Prop := γ = δ
def triangle_angle_values (A B : ℝ) : Prop := A = 80 ∧ B = 50

theorem find_angle_4
  (A B : ℝ) (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle_sum_180 angle1 angle2)
  (h2 : angle_equality angle3 angle4)
  (h3 : triangle_angle_values A B)
  (h4 : angle_sum_180 (angle1 + A + B) 180)
  (h5 : angle_sum_180 (angle2 + angle3 + angle4) 180) :
  angle4 = 25 :=
by sorry

end find_angle_4_l717_717024


namespace parabola_intersection_l717_717054

def parabola_focus (a : ℝ) : ℝ × ℝ :=
  (0, a / 4)

def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  2 * P.1 = A.1 + B.1 ∧ 2 * P.2 = A.2 + B.2

theorem parabola_intersection
  (P A B : ℝ × ℝ)
  (hp : P = (2, 1))
  (hAB : A ≠ B ∧ ∃ l : ℝ, quadratic (l, -1, 12) ∧ (A.1 ^ 2 = 12 * A.2) ∧ (B.1 ^ 2 = 12 * B.2))
  (hmid : is_midpoint P A B) :
  |dist (A, parabola_focus 12)| + |dist (B, parabola_focus 12)| = 8 / 3 :=
sorry

end parabola_intersection_l717_717054


namespace A_share_correct_l717_717671

-- Define the investments of A, B, and C, and the total profit
def A_investment : ℝ := 2400
def B_investment : ℝ := 7200
def C_investment : ℝ := 9600
def total_profit : ℝ := 9000

-- Calculate total investment
def total_investment : ℝ := A_investment + B_investment + C_investment

-- Calculate A's share in the profit
def A_share_in_profit : ℝ := (A_investment / total_investment) * total_profit

-- The statement that needs to be proven
theorem A_share_correct : A_share_in_profit = 1125 := by
  sorry

end A_share_correct_l717_717671


namespace total_pencils_is_correct_l717_717440

-- Define the conditions
def num_sets : ℕ := 4
def pencils_per_set : ℕ := 16

-- The goal is to prove that the total number of mechanical pencils is 64
theorem total_pencils_is_correct : num_sets * pencils_per_set = 64 := by
  calc
    num_sets * pencils_per_set = 4 * 16 : by rfl
                           ... = 64     : by norm_num

end total_pencils_is_correct_l717_717440


namespace triangle_solution_l717_717464

variable {a b c : ℝ}
variable {A B C : ℝ} -- angles in triangle

-- Definitions for given conditions
def side_a : ℝ := 2 * Real.sqrt 2
def sin_C_eq : Real.sin C = Real.sqrt 2 * Real.sin A
def cos_C : ℝ := Real.sqrt 2 / 4

-- Problem Statement
theorem triangle_solution :
  (a = side_a) →
  (Real.sin C = Real.sqrt 2 * Real.sin A) →
  (Real.cos C = Real.sqrt 2 / 4) →
  c = 4 ∧
  (1/2 * a * b * Real.sin C = 2 * Real.sqrt 7) :=
by
  sorry

end triangle_solution_l717_717464


namespace total_cost_for_fresh_water_for_family_one_day_l717_717466

def cost_per_gallon : ℝ := 1
def water_needed_per_person_per_day : ℝ := 1 / 2
def family_size : ℕ := 6

theorem total_cost_for_fresh_water_for_family_one_day : 
  (family_size * water_needed_per_person_per_day * cost_per_gallon) = 3 := 
by 
  sorry

end total_cost_for_fresh_water_for_family_one_day_l717_717466


namespace maximum_xyzw_l717_717144

theorem maximum_xyzw (x y z w : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_pos_w : 0 < w)
(h : (x * y * z) + w = (x + w) * (y + w) * (z + w))
(h_sum : x + y + z + w = 1) :
  xyzw = 1 / 256 :=
sorry

end maximum_xyzw_l717_717144


namespace exists_a_b_l717_717917

theorem exists_a_b (n : ℕ) (hn : 0 < n) : ∃ a b : ℤ, (4 * a^2 + 9 * b^2 - 1) % n = 0 := by
  sorry

end exists_a_b_l717_717917


namespace sum_n_binomial_l717_717595

theorem sum_n_binomial :
  let n := (∑ x in Finset.range 28, if (choose 27 14 + choose 27 x = choose 28 15) then x else 0) in
  n = 28 :=
by
  sorry

end sum_n_binomial_l717_717595


namespace combined_snowfall_l717_717868

theorem combined_snowfall (snow_monday : ℝ) (snow_tuesday : ℝ) (h_monday : snow_monday = 0.32) (h_tuesday : snow_tuesday = 0.21) :
  snow_monday + snow_tuesday = 0.53 :=
by
  rw [h_monday, h_tuesday]
  norm_num
  sorry

end combined_snowfall_l717_717868


namespace max_angle_tangents_parabola_circle_l717_717052

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 4 * x
noncomputable def circle (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 2

theorem max_angle_tangents_parabola_circle :
  ∀ (P : ℝ × ℝ), parabola P.1 P.2 → (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ θ * 180 / Real.pi = 60) :=
begin
  sorry
end

end max_angle_tangents_parabola_circle_l717_717052


namespace gcd_of_17420_23826_36654_l717_717372

theorem gcd_of_17420_23826_36654 : Nat.gcd (Nat.gcd 17420 23826) 36654 = 2 := 
by 
  sorry

end gcd_of_17420_23826_36654_l717_717372


namespace find_a_l717_717096

-- Define the necessary polynomials
def polynomial1 := (x : ℝ) ↦ x + 1
def polynomial2 (a : ℝ) := (x : ℝ) ↦ x^2 - 5 * a * x + a

-- Expanded form of the polynomial product
def expanded_polynomial (a : ℝ) (x : ℝ) := x^3 + (1 - 5 * a) * x^2 - 4 * a * x + a

-- Condition: the coefficient of x^2 in the expanded polynomial is 0
def coefficient_x2_zero (a : ℝ) : Prop := 1 - 5 * a = 0

-- Proof statement that we need to show a = 1/5 given the condition
theorem find_a (a : ℝ) (h : coefficient_x2_zero a) : a = 1/5 :=
sorry

end find_a_l717_717096


namespace exists_increasing_sequence_l717_717530

theorem exists_increasing_sequence (a1 : ℕ) (h1 : a1 > 1) :
  ∃ (a : ℕ → ℕ), (strict_mono a) ∧ (∀ k ≥ 1, (∑ i in finset.range k, (a i)^2) % (∑ i in finset.range k, (a i)) = 0) :=
sorry

end exists_increasing_sequence_l717_717530


namespace AC_length_l717_717167

variable (A B C M I O : Type) [OurGeometry A B C M I O]

-- Conditions
variable (AB BC : ℝ)
variable (MO_eq_MI : distance M O = distance M I)
variable (AB_eq : AB = 15) (BC_eq : BC = 7) 

-- Question: Prove AC = 13
theorem AC_length :
  AB = 15 →
  BC = 7 →
  distance M O = distance M I →
  distance A C = 13 :=
by {
  sorry
}

end AC_length_l717_717167


namespace problem1_problem2_l717_717433

open Real

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

def has_one_solution (a : ℝ) : Prop :=
  discriminant a (-2) 1 = 0 ∨ a = 0

def has_at_least_one_solution (a : ℝ) : Prop :=
  discriminant a (-2) 1 ≥ 0

theorem problem1 (a : ℝ) :
  (∃ x : ℝ, ax^2 - 2x + 1 = 0 ∧ (∀ y : ℝ, ax^2 - 2y + 1 = 0 → y = x)) → (a = 0 ∨ a = 1) :=
  sorry

theorem problem2 (a : ℝ) :
  (∃ x : ℝ, ax^2 - 2x + 1 = 0) → a ≤ 1 :=
  sorry

end problem1_problem2_l717_717433


namespace sin_neg_11_sixth_pi_l717_717332

-- Definitions for the conditions
def period_sine (x : ℝ) : Prop := ∀ n : ℤ, sin (x + 2 * n * π) = sin x
def special_angle_sine : sin (π / 6) = 1 / 2

-- The final proof statement
theorem sin_neg_11_sixth_pi : sin (-11 / 6 * π) = 1 / 2 :=
by 
  have h_period: period_sine (-2 * π + π / 6) := by sorry
  have h_special: special_angle_sine := by sorry
  calc
    sin (-11 / 6 * π) = sin (-2 * π + π / 6) : by sorry
    ... = sin (π / 6) : by exact h_period 1
    ... = 1 / 2 : by exact h_special

end sin_neg_11_sixth_pi_l717_717332


namespace no_two_champions_l717_717111

structure Tournament (Team : Type) :=
  (defeats : Team → Team → Prop)  -- Team A defeats Team B

def is_superior {Team : Type} (T : Tournament Team) (A B: Team) : Prop :=
  T.defeats A B ∨ ∃ C, T.defeats A C ∧ T.defeats C B

def is_champion {Team : Type} (T : Tournament Team) (A : Team) : Prop :=
  ∀ B, A ≠ B → is_superior T A B

theorem no_two_champions {Team : Type} (T : Tournament Team) :
  ¬ (∃ A B, A ≠ B ∧ is_champion T A ∧ is_champion T B) :=
sorry

end no_two_champions_l717_717111


namespace initial_weight_of_suitcase_l717_717483

theorem initial_weight_of_suitcase
(perfume_weight : ℕ := 5 * 1.2)
(chocolate_weight : ℕ := 4)
(soap_weight : ℕ := 2 * 5)
(jam_weight : ℕ := 2 * 8)
(total_return_weight : ℕ := 11)
(h16 : 16 ≠ 0) :
  let total_weight_in_pounds := (perfume_weight + soap_weight + jam_weight) / 16 + chocolate_weight in
  total_return_weight - total_weight_in_pounds = 5 :=
by {
  -- Perform necessary conversions and conclude the proof
  sorry
}

end initial_weight_of_suitcase_l717_717483


namespace circle_area_l717_717211

-- Given conditions
variables {BD AC : ℝ} (BD_pos : BD = 6) (AC_pos : AC = 12)
variables {R : ℝ} (R_pos : R = 15 / 2)

-- Prove that the area of the circles is \(\frac{225}{4}\pi\)
theorem circle_area (BD_pos : BD = 6) (AC_pos : AC = 12) (R : ℝ) (R_pos : R = 15 / 2) : 
        ∃ S, S = (225 / 4) * Real.pi := 
by sorry

end circle_area_l717_717211


namespace compare_x_y_l717_717185

theorem compare_x_y : 
  let x := 2007 * 2011 - 2008 * 2010
  let y := 2008 * 2012 - 2009 * 2011
  in x = y :=
by
  sorry

end compare_x_y_l717_717185


namespace notebooks_ratio_l717_717022

variable (C N : Nat)

theorem notebooks_ratio (h1 : 512 = C * N)
  (h2 : 512 = 16 * (C / 2)) :
  N = C / 8 :=
by
  sorry

end notebooks_ratio_l717_717022


namespace hannah_max_pads_l717_717818

theorem hannah_max_pads (i e p : ℕ) (hi : i ≥ 1) (he : e ≥ 1) 
    (eq : 2 * i + 3 * e + 9 * p = 60) : p ≤ 5 :=
begin
  sorry
end

end hannah_max_pads_l717_717818


namespace exists_smallest_n_gt_2_l717_717393

noncomputable def seq (a : ℕ → ℝ) (c : ℝ) (n : ℕ) : ℝ :=
if n = 1 then 1/2 else a (n - 1) + (c / (a (n - 1))^2)

theorem exists_smallest_n_gt_2 :
  ∃ n : ℕ, seq (λ n, seq (λ n, seq (λ n, ...) (1/216) n) (1/216) n) (1/216) n > 2 ∧
  ∀ m < n, seq (λ n, seq (λ n, seq (λ n, ...) (1/216) n) (1/216) n) (1/216) m ≤ 2 :=
sorry

end exists_smallest_n_gt_2_l717_717393


namespace solve_for_a_l717_717095

theorem solve_for_a (a : ℂ) : (a - complex.I) * (1 + a * complex.I) = -4 + 3 * complex.I → a = -2 := by
  sorry

end solve_for_a_l717_717095


namespace mailing_cost_l717_717765

theorem mailing_cost (W : ℝ) : 
  let cost := 8 * ⌈W / 2⌉ in
  cost = 8 * ⌈W / 2⌉ :=
  sorry

end mailing_cost_l717_717765


namespace card_probability_l717_717311

theorem card_probability :
  let cards := Finset.range 151
  let multiples_of_4 := Finset.filter (λ x, x % 4 = 0) cards
  let multiples_of_5 := Finset.filter (λ x, x % 5 = 0) cards
  let multiples_of_7 := Finset.filter (λ x, x % 7 = 0) cards
  let multiples_of_20 := Finset.filter (λ x, x % 20 = 0) cards
  let multiples_of_28 := Finset.filter (λ x, x % 28 = 0) cards
  let multiples_of_35 := Finset.filter (λ x, x % 35 = 0) cards
  let multiples_of_140 := Finset.filter (λ x, x % 140 = 0) cards
  (multiples_of_4.card + multiples_of_5.card + multiples_of_7.card
   - multiples_of_20.card - multiples_of_28.card - multiples_of_35.card
   + multiples_of_140.card) / cards.card = 73 / 150 :=
by sorry

end card_probability_l717_717311


namespace ratio_lcm_gcf_eq_55_l717_717594

theorem ratio_lcm_gcf_eq_55 : 
  ∀ (a b : ℕ), a = 210 → b = 462 →
  (Nat.lcm a b / Nat.gcd a b) = 55 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end ratio_lcm_gcf_eq_55_l717_717594


namespace hanoi_moves_correct_l717_717581

def hanoi_moves (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * hanoi_moves (n - 1) + 1

theorem hanoi_moves_correct (n : ℕ) : hanoi_moves n = 2^n - 1 := by
  sorry

end hanoi_moves_correct_l717_717581


namespace arithmetic_seq_a4_l717_717793

-- Definition of an arithmetic sequence with the first three terms given.
def arithmetic_seq (a : ℕ → ℕ) :=
  a 0 = 2 ∧ a 1 = 4 ∧ a 2 = 6 ∧ ∃ d, ∀ n, a (n + 1) = a n + d

-- The actual proof goal.
theorem arithmetic_seq_a4 : ∃ a : ℕ → ℕ, arithmetic_seq a ∧ a 3 = 8 :=
by
  sorry

end arithmetic_seq_a4_l717_717793


namespace curve_length_ln_cos_y_l717_717688

noncomputable def arc_length_ln_cos_y : ℝ :=
  ∫ y in 0..(Real.pi / 3), Real.sec y

theorem curve_length_ln_cos_y :
  arc_length_ln_cos_y = Real.log (2 + Real.sqrt 3) :=
by
  sorry

end curve_length_ln_cos_y_l717_717688


namespace log_problem_l717_717786

theorem log_problem {a m n : ℝ} (h₁ : log a 2 = m) (h₂ : log a 3 = n) : a^(2*m + n) = 12 :=
sorry

end log_problem_l717_717786


namespace part1_part2_l717_717428

def f (x : ℝ) : ℝ := abs (2 * x - 1)
def g (x : ℝ) (m : ℝ) : ℝ := 1 / (f x + f (x + 1) + m)

theorem part1 (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x ≤ a) →
  a = 1 :=
by
  intro h
  have h1 : f 0 = 1, by simp [f]
  have h2 : f 1 == 1, by simp [f]
  have h3 : 0 ≤ 0, by linarith
  have h4 : 1 ≤ 1, by linarith
  have h5 : 1 = a, by linarith
  exact h5

theorem part2 (m : ℝ) :
  (∀ x : ℝ, f x + f (x + 1) + m ≠ 0) →
  m ≠ 0 :=
by
  intro h
  have hmin : ∃ x, (f x + f (x + 1)) = 0, from by
    use 1 / 2
    simp [f]
    linarith
  cases hmin with x hx
  have hmin_val : 0 + m ≠ 0, from h x
  linarith

end part1_part2_l717_717428


namespace solution_set_of_inequality_l717_717232

theorem solution_set_of_inequality :
  { x : ℝ | 3 ≤ |2 * x - 5| ∧ |2 * x - 5| < 9 } = { x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) } :=
by 
  -- Conditions and steps omitted for the sake of the statement.
  sorry

end solution_set_of_inequality_l717_717232


namespace wendy_accounting_manager_years_l717_717243

theorem wendy_accounting_manager_years
  (years_accountant : ℕ)
  (total_years : ℕ)
  (percentage_accounting : ℕ)
  (percentage_accounting_eq : percentage_accounting = 50)
  (years_accountant_eq : years_accountant = 25)
  (total_years_eq : total_years = 80)
  : (total_years * percentage_accounting / 100 - years_accountant = 15) :=
by
  have h1 : total_years * percentage_accounting / 100 = 40 := by
    rw [percentage_accounting_eq, total_years_eq]
    norm_num
  have h2 : 40 - years_accountant = 15 := by
    rw [years_accountant_eq]
    norm_num
  rw [← h1, h2]
  norm_num

end wendy_accounting_manager_years_l717_717243


namespace tangent_ellipse_hyperbola_l717_717951

theorem tangent_ellipse_hyperbola {m : ℝ} :
    (∀ x y : ℝ, x^2 + 9*y^2 = 9 → x^2 - m*(y + 1)^2 = 1 → false) →
    m = 72 :=
sorry

end tangent_ellipse_hyperbola_l717_717951


namespace arithmetic_seq_sum_zero_l717_717036

noncomputable def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

theorem arithmetic_seq_sum_zero (a d : ℤ) (h : ∑ i in Finset.range 101, arithmetic_seq a d (i + 1) = 0) : 
  arithmetic_seq a d 3 + arithmetic_seq a d 99 = 0 :=
by
  sorry

end arithmetic_seq_sum_zero_l717_717036


namespace r4_plus_inv_r4_l717_717188

theorem r4_plus_inv_r4 (r : ℝ) (h : (r + (1 : ℝ) / r) ^ 2 = 5) : r ^ 4 + (1 : ℝ) / r ^ 4 = 7 := 
by
  -- Proof goes here
  sorry

end r4_plus_inv_r4_l717_717188


namespace perfect_cubes_not_divisible_by_10_l717_717369

-- Definitions based on conditions
def is_divisible_by_10 (n : ℕ) : Prop := 10 ∣ n
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), n = k ^ 3
def erase_last_three_digits (n : ℕ) : ℕ := n / 1000

-- Main statement
theorem perfect_cubes_not_divisible_by_10 (x : ℕ) :
  is_perfect_cube x ∧ ¬ is_divisible_by_10 x ∧ is_perfect_cube (erase_last_three_digits x) →
  x = 1331 ∨ x = 1728 :=
by
  sorry

end perfect_cubes_not_divisible_by_10_l717_717369


namespace max_value_C_triangle_l717_717463

theorem max_value_C_triangle 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : A + B + C = π)
  (h2 : a * cos B + b * cos A = (a + b) / 2)
  (ha : c = √(a^2 + b^2 - 2 * a * b * cos C))
  : C ≤ π / 3 := 
sorry

end max_value_C_triangle_l717_717463


namespace sum_of_x_coordinates_l717_717168

theorem sum_of_x_coordinates (a c : ℕ) (ha : 0 < a) (hc : 0 < c) 
  (h_intersect : ∃ x, ax + 7 = 0 ∧ 4x + c = 0) : 
  let x1 := -7 / a in
  let x2 := -c / 4 in
  ∀ (p : ℕ × ℕ), p ∈ [(1, 28), (2, 14), (4, 7), (7, 4), (14, 2), (28, 1)] →
    (p.1 = a ∧ p.2 = c) →
    (x1 + x2) = -14 :=
begin
  sorry
end

end sum_of_x_coordinates_l717_717168


namespace probability_odd_sum_l717_717548

theorem probability_odd_sum :
  let tiles := Finset.range 10,
      odd_tiles := {1, 3, 5, 7, 9},
      even_tiles := {2, 4, 6, 8, 10},
      total_permutations := (Finset.choose 10 3) * (Finset.choose 7 3),
      favorable_outcomes_scenario1 := (Finset.choose 5 3) * (Finset.choose 4 2) * 2 * (Finset.choose 2 1),
      favorable_outcomes_scenario2 := (Finset.choose 5 1) * (Finset.choose 4 1) * (Finset.choose 3 1) * (Finset.choose 5 2) * (Finset.choose 3 2) * 1,
      favorable_outcomes := favorable_outcomes_scenario1 + favorable_outcomes_scenario2,
      probability := favorable_outcomes.to_rat / total_permutations.to_rat,
      m := favorable_outcomes.to_nat.gcd(total_permutations.to_nat),
      n := total_permutations.to_nat.gcd(favorable_outcomes.to_nat)
  in m + n = 51 :=
  sorry

end probability_odd_sum_l717_717548


namespace max_height_of_basketball_l717_717634

def h (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 2

theorem max_height_of_basketball : ∃ t : ℝ, h t = 127 :=
by
  use 5
  sorry

end max_height_of_basketball_l717_717634


namespace modular_inverse_l717_717355

/-- Define the number 89 -/
def a : ℕ := 89

/-- Define the modulus 90 -/
def n : ℕ := 90

/-- The condition given in the problem -/
lemma pow_mod (h : a ≡ -1 [MOD n]) : (a * a) % n = 1 % n := by 
  sorry

/-- The main statement to prove the modular inverse -/
theorem modular_inverse (h : a ≡ -1 [MOD n]) : (a * a) % n = 1 % n → a ≡ a⁻¹ [MOD n] := by
  intro h1
  have h2 : a⁻¹ % n = a % n := by 
    sorry
  exact h2

end modular_inverse_l717_717355


namespace james_remaining_balance_l717_717487

theorem james_remaining_balance 
  (initial_balance : ℕ := 500) 
  (ticket_1_2_cost : ℕ := 150)
  (ticket_3_cost : ℕ := ticket_1_2_cost / 3)
  (total_cost : ℕ := 2 * ticket_1_2_cost + ticket_3_cost)
  (roommate_share : ℕ := total_cost / 2) :
  initial_balance - roommate_share = 325 := 
by 
  -- By not considering the solution steps, we skip to the proof.
  sorry

end james_remaining_balance_l717_717487


namespace sum_of_squares_first_6_base6_l717_717697

-- Convert base-6 representation to base-10 integer
def base6_to_base10 (n : ℕ) (digits : List ℕ) : ℕ :=
  digits.reverse.enum.map (λ ⟨i, d⟩, d * n^i).sum

def square_in_base10 (n : ℕ) : ℕ := n * n

def sum_of_squares (n : ℕ) : ℕ :=
  (List.range n).map (λ x, square_in_base10 (x + 1)).sum

-- Convert base-10 integer to base-6 representation
def base10_to_base6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec f (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else f (n / 6) ((n % 6) :: acc)
    f n []

theorem sum_of_squares_first_6_base6 :
  let base6_numbers := [1, 2, 3, 4, 5, 10]
  let squares := base6_numbers.map (λ x, square_in_base10 (base6_to_base10 6 [x]))
  let sum := squares.sum
  base10_to_base6 sum = [2, 3, 1] :=
by
  sorry

end sum_of_squares_first_6_base6_l717_717697


namespace multiply_base5_234_75_l717_717955

def to_base5 (n : ℕ) : ℕ := 
  let rec helper (n : ℕ) (acc : ℕ) (multiplier : ℕ) : ℕ := 
    if n = 0 then acc
    else
      let d := n % 5
      let q := n / 5
      helper q (acc + d * multiplier) (multiplier * 10)
  helper n 0 1

def base5_multiplication (a b : ℕ) : ℕ :=
  to_base5 ((a * b : ℕ))

theorem multiply_base5_234_75 : base5_multiplication 234 75 = 450620 := 
  sorry

end multiply_base5_234_75_l717_717955


namespace complex_eq_real_implies_a_eq_2_l717_717419

theorem complex_eq_real_implies_a_eq_2 (a : ℝ) :
  let z := (2 * complex.i - a) / complex.i in
  z.re = z.im → a = 2 :=
by
  intro h
  have : z = 2 + (a : ℂ) * complex.i := by {
    change (2 * complex.i - a : ℂ) / complex.i with (2 * complex.i - (a : ℂ)) / complex.i,
    field_simp,
    ring,
  }
  rw this at h
  simp [complex.re_add_im, add_comm] at h
  exact h.symm

end complex_eq_real_implies_a_eq_2_l717_717419


namespace circles_area_l717_717204

theorem circles_area (BD AC : ℝ) (r : ℝ) (h1 : BD = 6) (h2 : AC = 12)
  (h3 : ∀ (d1 d2 : ℝ), d1 = AC / 2 → d2 = BD / 2 → r^2 = (r - d2)^2 + d1^2) :
  real.pi * r^2 = (225/4) * real.pi :=
by
  -- proof to be filled
  sorry

end circles_area_l717_717204


namespace minute_hand_only_rotates_l717_717679

-- Define what constitutes translation and rotation
def is_translation (motion : ℝ → ℝ → Prop) : Prop :=
  ∀ (p1 p2 : ℝ), motion p1 p2 → (∃ d : ℝ, ∀ t : ℝ, motion (p1 + t) (p2 + t) ∧ |p1 - p2| = d)

def is_rotation (motion : ℝ → ℝ → Prop) : Prop :=
  ∀ (p : ℝ), ∃ c : ℝ, ∃ r : ℝ, (∀ (t : ℝ), |p - c| = r)

-- Define the condition that the minute hand of a clock undergoes a specific motion
def minute_hand_motion (p : ℝ) (t : ℝ) : Prop :=
  -- The exact definition here would involve trigonometric representation
  sorry

-- The main proof statement
theorem minute_hand_only_rotates :
  is_rotation minute_hand_motion ∧ ¬ is_translation minute_hand_motion :=
sorry

end minute_hand_only_rotates_l717_717679


namespace smallest_multiple_of_6_and_15_l717_717756

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ c : ℕ, c > 0 ∧ c % 6 = 0 ∧ c % 15 = 0 → c ≥ b := 
begin
  use 30,
  split,
  { exact nat.succ_pos 29, },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 2 3) (dvd_mul_right 3 5)), },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 3 5) (dvd_mul_right 3 2)), },
  { intros c hc1 hc2,
    have hc3 : c % 30 = 0,
    {
      suffices h : c % 6 = 0 ∧ c % 15 = 0 ↔ c % lcm 6 15 = 0,
      { rw ← h, exact ⟨hc1, hc2⟩, },
      exact nat.dvd_iff_mod_eq_zero,
    },
    linarith,
  }
end

end smallest_multiple_of_6_and_15_l717_717756


namespace boy_initial_oranges_l717_717636

theorem boy_initial_oranges:
  ∃ O: ℕ, 
    ∀ (f b: ℕ), 
    b = O / 3 →
    f = (2 * O / 3) / 4 →
    f = 2 →
    O = 12 :=
begin
  sorry  -- Proof of the theorem
end

end boy_initial_oranges_l717_717636


namespace tenth_observation_value_l717_717197

-- Define the conditions
variable (obs_9_sum : ℝ) (obs_10_sum : ℝ)
variable (neg_sum : ℝ)
variable (avg_9 : ℝ) (avg_10 : ℝ)
variable (n : ℕ)

-- Set the known values
def obs_9_avg : avg_9 = 15.3 := sorry
def num_obs_9 : n = 9 := sorry
def obs_9_sum_calc : obs_9_sum = avg_9 * n := sorry
def neg_obs_sum : neg_sum = -8.6 := sorry
def new_avg : avg_10 = avg_9 - 1.7 := sorry
def new_obs_num : n + 1 = 10 := sorry

noncomputable def obs_10_value : ℝ :=
obs_10_sum - obs_9_sum

theorem tenth_observation_value (obs_10_value = -1.7) : 
    avg_9 = 15.3 ∧ n = 9 ∧ obs_9_sum = avg_9 * n ∧ 
    neg_sum = -8.6 ∧ avg_10 = avg_9 - 1.7 ∧ 
    obs_10_sum = avg_10 * (n + 1) → obs_10_value = -1.7 := by 
    sorry

end tenth_observation_value_l717_717197


namespace sqrt_D_irrational_l717_717887

noncomputable def is_always_irrational (D : ℤ) : Prop :=
  ∃ (a b : ℤ), (∃ k, 2 * k = a) ∧ (∃ m, 2 * m = b) ∧ b = a + 2 ∧ ∃ c, c = a * b ∧ D = a^2 + b^2 + c^2 ∧ irrational (real.sqrt D)

theorem sqrt_D_irrational : is_always_irrational :=
  by {
    sorry
  }

end sqrt_D_irrational_l717_717887


namespace unique_parallel_line_in_beta_l717_717094

-- Define the basic geometrical entities.
axiom Plane : Type
axiom Line : Type
axiom Point : Type

-- Definitions relating entities.
def contains (P : Plane) (l : Line) : Prop := sorry
def parallel (A B : Plane) : Prop := sorry
def in_plane (p : Point) (P : Plane) : Prop := sorry
def parallel_lines (a b : Line) : Prop := sorry

-- Statements derived from the conditions in problem.
variables (α β : Plane) (a : Line) (B : Point)
-- Given conditions
axiom plane_parallel : parallel α β
axiom line_in_plane : contains α a
axiom point_in_plane : in_plane B β

-- The ultimate goal derived from the question.
theorem unique_parallel_line_in_beta : 
  ∃! b : Line, (in_plane B β) ∧ (parallel_lines a b) :=
sorry

end unique_parallel_line_in_beta_l717_717094


namespace reduced_flow_rate_l717_717669

-- Definitions of the conditions
def r0 : ℕ := 5
def r1 := 0.6 * r0 - 1

-- The theorem statement we need to prove
theorem reduced_flow_rate : r1 = 2 :=
by
  simp only [r0, r1]
  norm_num
  sorry

end reduced_flow_rate_l717_717669


namespace persons_in_boat_l717_717551

theorem persons_in_boat (W1 W2 new_person_weight : ℝ) (n : ℕ)
  (hW1 : W1 = 55)
  (h_new_person : new_person_weight = 50)
  (hW2 : W2 = W1 - 5) :
  (n * W1 + new_person_weight) / (n + 1) = W2 → false :=
by
  intros h_eq
  sorry

end persons_in_boat_l717_717551


namespace Steve_takes_13_5_minutes_longer_l717_717706

/-- Danny can reach Steve's house in 27 minutes. -/
def T_D := 27

/-- Steve takes twice the time to reach Danny's house as Danny takes to reach Steve's house. -/
def T_S := 2 * T_D

/-- Time taken by Danny to reach halfway point -/
def T_D_half := T_D / 2

/-- Time taken by Steve to reach halfway point -/
def T_S_half := T_S / 2

/-- Prove that Steve takes 13.5 minutes longer to reach halfway point than Danny -/
theorem Steve_takes_13_5_minutes_longer : T_S_half - T_D_half = 13.5 := by
  sorry

end Steve_takes_13_5_minutes_longer_l717_717706


namespace set_of_points_in_rhombus_l717_717505

-- Define the basic concepts and objects involved.
variable {Point : Type*} [MetricSpace Point] -- Metric space to define distances
variable {A1 A2 B1 B2 O : Point} -- Points as given in the problem
variable (d : MetricSpace.dist _ _) -- Distance function

-- Define the conditions on the points
def is_diameter_endpoints (A1 A2 B1 B2 O : Point) : Prop :=
  d A1 O = d A2 O ∧ d B1 O = d B2 O

def is_inside_rhombus (P : Point) : Prop :=
  -- Assuming hypothetical definitions of rhombus boundaries using distances
  let a1 := MetricSpace.perpendicular_bisector A1 O
  let a2 := MetricSpace.perpendicular_bisector A2 O
  let b1 := MetricSpace.perpendicular_bisector B1 O
  let b2 := MetricSpace.perpendicular_bisector B2 O
  P ∈ a1 ∧ P ∈ a2 ∧ P ∈ b1 ∧ P ∈ b2

-- Define the main theorem
theorem set_of_points_in_rhombus {A1 A2 B1 B2 O : Point} (P : Point)
  (h1 : is_diameter_endpoints A1 A2 B1 B2 O)
  (h2 : d A1 P > d O P) 
  (h3 : d A2 P > d O P) 
  (h4 : d B1 P > d O P) 
  (h5 : d B2 P > d O P) :
  is_inside_rhombus P :=
begin
  sorry -- The proof will demonstrate the geometric properties.
end

end set_of_points_in_rhombus_l717_717505


namespace checkerboard_squares_count_correct_l717_717631

def checkerboard := fin 10 × fin 10 → bool  -- Defines a checkerboard, alternating squares
def alternating (chess : checkerboard) : Prop :=
  ∀ x y, chess (x, y) = bxor (x % 2 = 0) (y % 2 = 0) -- Defines alternating property

def count_squares_with_at_least_5_black
  (chess : checkerboard) (h : alternating chess) : ℕ :=
  let three_by_three := 32
  let four_by_four := 49
  let five_by_five := 36
  let six_by_six := 25
  let seven_by_seven := 16
  let eight_by_eight := 9
  let nine_by_nine := 4
  let ten_by_ten := 1
  three_by_three + four_by_four + five_by_five + six_by_six +
  seven_by_seven + eight_by_eight + nine_by_nine + ten_by_ten

theorem checkerboard_squares_count_correct :
  ∀ chess : checkerboard, alternating chess →
  count_squares_with_at_least_5_black chess = 172 :=
by
  sorry

end checkerboard_squares_count_correct_l717_717631


namespace inscribable_quadrilateral_l717_717280

theorem inscribable_quadrilateral (a R : ℝ) (O : ℝ × ℝ) (Q : list (ℝ × ℝ)) (hchord_length : ∀ side ∈ Q, (dist (side.1) O = a ∧ dist (side.2) O = a)) : 
  ∃ I : ℝ × ℝ, ∀ side ∈ Q, (dist I side.1 = R ∧ dist I side.2 = R) :=
sorry

end inscribable_quadrilateral_l717_717280


namespace sum_of_repeating_decimals_l717_717331

theorem sum_of_repeating_decimals :
  let x : ℝ := 0.3333333333333333 -- This is for representation purposes in Lean
      y : ℝ := 0.6666666666666666 -- This is for representation purposes in Lean
  in (x + y + 1/4 = 5 / 4) :=
by {
  sorry -- Proof goes here
}

end sum_of_repeating_decimals_l717_717331


namespace impossible_tiling_l717_717479

def cell : Type := ℕ × ℕ
def L_shape (a b c : cell) : Prop :=
  (a = (0,0) ∧ b = (0,1) ∧ c = (1,0)) ∨
  (a = (0,0) ∧ b = (0,1) ∧ c = (1,1)) ∨
  (a = (0,0) ∧ b = (1,0) ∧ c = (1,1)) ∨
  (a = (0,1) ∧ b = (1,0) ∧ c = (1,1))

def in_bounds (m n: ℕ) (cell: cell): Prop :=
  cell.1 < m ∧ cell.2 < n

theorem impossible_tiling : ¬ ∃ k: ℕ, ∃ (f: cell → Finset (Finset cell)),
  (∀ xy, xy ∈ (Finset.range 5).product (Finset.range 7) → 
    ∑ S in f (xy.1, xy.2), (if in_bounds 5 7 S x then 1 else 0) = k) ∧
  (∀ xy, xy ∈ (Finset.range 5).product (Finset.range 7) → 
    ∀ S ∈ f xy, L_shape (xy, S.to_finset)) :=
sorry

end impossible_tiling_l717_717479


namespace greatest_alpha_exists_l717_717846

-- Define lattice points in the Cartesian coordinate system
def is_lattice_point (p : ℤ × ℤ) : Prop := true

-- Define the distance between two lattice points
def dist (A B : ℤ × ℤ) : ℝ :=
  real.sqrt (((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2).to_real)

-- Define the function f and the conditions on it
noncomputable def f (A : ℤ × ℤ) : ℕ := sorry

-- The proof problem
theorem greatest_alpha_exists :
  ∃ (α : ℝ), 
    ( ∃ (f : ℤ × ℤ → ℕ), 
      ∃ (c : ℝ), c > 0 ∧ 
      (∀ n : ℕ, card { A : ℤ × ℤ | f A = n } > (c * (n:ℝ)^α)) ∧
      (∀ A B C : ℤ × ℤ, is_lattice_point A → is_lattice_point B → is_lattice_point C →
        (f A > dist B C ∨ f B > dist A C ∨ f C > dist A B))
    ) ∧ ∀ α' : ℝ, 
          ( ∃ f : ℤ × ℤ → ℕ, 
            ∃ c : ℝ, c > 0 ∧ 
            (∀ n : ℕ, card { A : ℤ × ℤ | f A = n } > (c * (n : ℝ)^α')) ∧
            (∀ A B C : ℤ × ℤ, is_lattice_point A → is_lattice_point B → is_lattice_point C →
              (f A > dist B C ∨ f B > dist A C ∨ f C > dist A B))
          )
          → α' ≤ 1 :=
begin
  sorry
end

end greatest_alpha_exists_l717_717846


namespace expression_for_fx_when_x_neg_l717_717409

def f (x : ℝ) : ℝ := if x > 0 then -x * (1 + x) else -x * (1 - x)

theorem expression_for_fx_when_x_neg (x : ℝ) (h1 : ∀ x > 0, f x = -x * (1 + x)) (h2 : ∀ x < 0, f (-x) = -f x) : ∀ x < 0, f x = -x * (1 - x) :=
by
  sorry

end expression_for_fx_when_x_neg_l717_717409


namespace solve_for_a_l717_717103

theorem solve_for_a (a : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x^2 - 5 * a * x + a) = x^3 + (1 - 5 * a) * x^2 - 4 * a * x + a) →
  (1 - 5 * a = 0) →
  a = 1 / 5 := 
by
  intro h₁ h₂
  sorry

end solve_for_a_l717_717103


namespace balloon_descent_rate_l717_717875

theorem balloon_descent_rate (D : ℕ) 
    (rate_of_ascent : ℕ := 50) 
    (time_chain_pulled_1 : ℕ := 15) 
    (time_chain_pulled_2 : ℕ := 15) 
    (time_chain_released_1 : ℕ := 10) 
    (highest_elevation : ℕ := 1400) :
    (time_chain_pulled_1 + time_chain_pulled_2) * rate_of_ascent - time_chain_released_1 * D = highest_elevation 
    → D = 10 := 
by 
  intro h
  sorry

end balloon_descent_rate_l717_717875


namespace triangle_side_ratio_eq_one_l717_717105

theorem triangle_side_ratio_eq_one
    (a b c C : ℝ)
    (h1 : a = 2 * b * Real.cos C)
    (cosine_rule : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
    (b / c = 1) := 
by 
    sorry

end triangle_side_ratio_eq_one_l717_717105


namespace inverse_89_mod_90_l717_717359

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  sorry -- proof goes here

end inverse_89_mod_90_l717_717359


namespace area_of_circles_l717_717207

theorem area_of_circles (BD AC : ℝ) (hBD : BD = 6) (hAC : AC = 12) : 
  ∃ S : ℝ, S = 225 / 4 * Real.pi :=
by
  sorry

end area_of_circles_l717_717207


namespace trigonometric_identity_example_l717_717269

open Real Real.Angle

theorem trigonometric_identity_example :
  sin (34 * π / 180) * sin (26 * π / 180) - cos (34 * π / 180) * cos (26 * π / 180) = - 1 / 2 :=
by
  -- Convert degrees to radians
  -- Proof goes here
  sorry

end trigonometric_identity_example_l717_717269


namespace volume_of_sphere_correct_l717_717764

noncomputable def volume_of_sphere (v : ℝ → ℝ → ℝ → ℝ) : ℝ :=
  let x := v x y z
  let y := v y x z
  let z := v z x y
  if x^2 + y^2 + z^2 = 6 * x - 12 * y + 18 * z then
    (4 / 3) * Real.pi * ((3 * Real.sqrt 14)^3)
  else 0

theorem volume_of_sphere_correct :
  ∀ v : ℝ → ℝ → ℝ → ℝ,
  volume_of_sphere v = 108 * Real.sqrt 14 * Real.pi :=
  sorry

end volume_of_sphere_correct_l717_717764


namespace units_digit_17_pow_2007_l717_717603

theorem units_digit_17_pow_2007 : (17^2007) % 10 = 3 :=
by sorry

end units_digit_17_pow_2007_l717_717603


namespace derek_percentage_difference_l717_717274

-- Definitions and assumptions based on conditions
def average_score_first_test (A : ℝ) : ℝ := A

def derek_score_first_test (D1 : ℝ) (A : ℝ) : Prop := D1 = 0.5 * A

def derek_score_second_test (D2 : ℝ) (D1 : ℝ) : Prop := D2 = 1.5 * D1

-- Theorem statement
theorem derek_percentage_difference (A D1 D2 : ℝ)
  (h1 : derek_score_first_test D1 A)
  (h2 : derek_score_second_test D2 D1) :
  (A - D2) / A * 100 = 25 :=
by
  -- Placeholder for the proof
  sorry

end derek_percentage_difference_l717_717274


namespace conditional_prob_B_given_A_l717_717523

namespace ConditionalProbability

def questions_total := 5
def science_questions := 3
def arts_questions := 2
def draw_first := 1
def draw_second := 2

def event_A (q : ℕ) : Prop := q = draw_first → science_questions
def event_B (q : ℕ) : Prop := q = draw_second → science_questions

theorem conditional_prob_B_given_A :
  P(event_B | event_A) = 1 / 2 := sorry

end ConditionalProbability

end conditional_prob_B_given_A_l717_717523


namespace inverse_89_mod_90_l717_717358

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  sorry -- proof goes here

end inverse_89_mod_90_l717_717358


namespace minimum_value_expression_l717_717787

theorem minimum_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_mean : log (sqrt 2) = (log (4 ^ a) + log (2 ^ b)) / 2) :
  (2 / a + 1 / b) ≥ 9 :=
by sorry

end minimum_value_expression_l717_717787


namespace james_money_left_l717_717485

-- Define the initial conditions
def ticket1_cost : ℕ := 150
def ticket2_cost : ℕ := 150
def ticket3_cost : ℕ := ticket1_cost / 3
def total_money : ℕ := 500
def roommate_share : ℕ := 2

-- Define and prove the theorem
theorem james_money_left : 
  let total_ticket_cost := ticket1_cost + ticket2_cost + ticket3_cost in
  let james_cost := total_ticket_cost / roommate_share in
  total_money - james_cost = 325 :=
by 
  let total_ticket_cost := ticket1_cost + ticket2_cost + ticket3_cost
  let james_cost := total_ticket_cost / roommate_share
  exact eq.refl 325

end james_money_left_l717_717485


namespace last_five_digits_of_large_exponentiation_is_45289_l717_717716

def f : ℕ → ℕ
| 0     := 9
| (n+1) := 9 ^ f n

theorem last_five_digits_of_large_exponentiation_is_45289 :
  let N := f 1000 
  (N % 100000) = 45289 :=
by
  sorry

end last_five_digits_of_large_exponentiation_is_45289_l717_717716


namespace min_real_roots_l717_717139

theorem min_real_roots (g : Polynomial ℝ)
  (deg_g : g.natDegree = 2010)
  (roots_g : ∀ (s : ℝ), g.is_root s → s ∈ (list.range 2010).map (λ i, g.root (complex.of_real i)))
  (distinct_magnitudes : (list.range 2010).map (λ i, complex.abs (g.root (complex.of_real i))).erase_duplicates = list.range 1005.erase_duplicates) :
  ∃ (real_roots_count : ℕ), real_roots_count = 6 :=
by
  sorry

end min_real_roots_l717_717139


namespace rectangular_eq_of_polar_l717_717119

-- Given conditions: polar coordinate equation of curve \( C \)
def polar_eq (ρ θ : ℝ) := ρ * sin (θ - π / 4) = sqrt 2

-- Convert to rectangular coordinates and prove the resulting equation
theorem rectangular_eq_of_polar (ρ θ : ℝ) :
  polar_eq ρ θ → ∃ (x y : ℝ), (y = ρ * sin θ) ∧ (x = ρ * cos θ) ∧ (y - x = 2) :=
by
  intro h
  use ρ * cos θ, ρ * sin θ
  -- This would normally be where the mathematical transformations would be shown
  sorry

end rectangular_eq_of_polar_l717_717119


namespace proof_a_minus_b_l717_717813

def S (a : ℕ) : Set ℕ := {1, 2, a}
def T (b : ℕ) : Set ℕ := {2, 3, 4, b}

theorem proof_a_minus_b (a b : ℕ)
  (hS : S a = {1, 2, a})
  (hT : T b = {2, 3, 4, b})
  (h_intersection : S a ∩ T b = {1, 2, 3}) :
  a - b = 2 := by
  sorry

end proof_a_minus_b_l717_717813


namespace arrangements_APPLE_is_60_l717_717084

-- Definition of the problem statement based on the given conditions
def distinct_arrangements_APPLE : Nat :=
  let n := 5
  let n_A := 1
  let n_P := 2
  let n_L := 1
  let n_E := 1
  (n.factorial / (n_A.factorial * n_P.factorial * n_L.factorial * n_E.factorial))

-- The proof statement (without the proof itself, which is "sorry")
theorem arrangements_APPLE_is_60 : distinct_arrangements_APPLE = 60 := by
  sorry

end arrangements_APPLE_is_60_l717_717084


namespace part_a_part_b_l717_717531

-- Let γ and δ represent acute angles, γ < δ implies γ - sin γ < δ - sin δ 
theorem part_a (alpha beta : ℝ) (h_alpha : 0 < alpha) (h_alpha2 : alpha < π/2) 
  (h_beta : 0 < beta) (h_beta2 : beta < π/2) (h : alpha < beta) : 
  alpha - Real.sin alpha < beta - Real.sin beta := sorry

-- Let γ and δ represent acute angles, γ < δ implies tan γ - γ < tan δ - δ 
theorem part_b (alpha beta : ℝ) (h_alpha : 0 < alpha) (h_alpha2 : alpha < π/2) 
  (h_beta : 0 < beta) (h_beta2 : beta < π/2) (h : alpha < beta) : 
  Real.tan alpha - alpha < Real.tan beta - beta := sorry

end part_a_part_b_l717_717531


namespace reciprocal_arithmetic_sequence_sum_S_less_than_sixth_l717_717391

-- Define the sequence a_n with the given conditions
def sequence_a (a : ℕ → ℝ) : Prop :=
  (a 1 = 1 / 3) ∧ ∀ n : ℕ, n ≥ 2 → a n ≠ 0 ∧ a (n - 1) - a n = 2 * a n * a (n - 1)

-- Prove that the reciprocal of a_n forms an arithmetic sequence
theorem reciprocal_arithmetic_sequence (a : ℕ → ℝ) (h : sequence_a a) :
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, n ≥ 1 → 1 / a n = a₀ + d * (n - 1)) :=
sorry

-- Define the sequence b_n
def sequence_b (a b : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, b n = a n * a (n + 1))

-- Define the sum S_n of the first n terms of b_n
def sum_S (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = ∑ k in Finset.range n, b (k + 1)

-- Prove that S_n < 1 / 6
theorem sum_S_less_than_sixth (a b : ℕ → ℝ) (S : ℕ → ℝ) (ha : sequence_a a) (hb : sequence_b a b) (hS : sum_S b S) :
  ∀ n : ℕ, S n < 1 / 6 :=
sorry

end reciprocal_arithmetic_sequence_sum_S_less_than_sixth_l717_717391


namespace unique_positive_b_for_discriminant_zero_l717_717738

theorem unique_positive_b_for_discriminant_zero (c : ℝ) : 
  (∃! b : ℝ, b > 0 ∧ (b^2 + 1/b^2)^2 - 4 * c = 0) → c = 1 :=
by
  sorry

end unique_positive_b_for_discriminant_zero_l717_717738


namespace sum_of_powers_of_2_divisible_l717_717898

theorem sum_of_powers_of_2_divisible (k n : ℕ) (h_pos_k : 0 < k) (h_pos_n : 0 < n) (h_lt : k < 2^(n + 1) - 1) : 
  ∃ (S : Finset ℕ), S.card = n ∧ k ∣ S.sum (λ i, 2^i) := 
by 
  sorry

end sum_of_powers_of_2_divisible_l717_717898


namespace value_of_m_l717_717620

theorem value_of_m (s : ℝ) (m : ℝ) (h: (2 ^ 16) * (25 ^ s) = 5 * (10 ^ m)) : m = 1 :=
by
sorry

end value_of_m_l717_717620


namespace inscribed_circle_in_quadrilateral_l717_717279

-- Given definitions related to the problem
structure Quadrilateral :=
  (a b c d: ℝ) -- sides of the quadrilateral
  (O: ℝ × ℝ) -- center of the circle
  (R: ℝ) -- radius of the circle
  (a_chord: ℝ) -- length of the chord cut out by the circle on each side of the quadrilateral

-- Problem: Prove that a circle can be inscribed in the quadrilateral given the conditions.
theorem inscribed_circle_in_quadrilateral (Q: Quadrilateral) (h_chords: 
    (∀ (P₁ P₂ P₃ P₄: ℝ × ℝ), 
      ∥P₁ - Q.O∥ = ∥P₂ - Q.O∥ ∧ ∥P₃ - Q.O∥ = ∥P₄ - Q.O∥ → 
      Q.a_chord = dist P₁ P₂ ∧ Q.a_chord = dist P₃ P₄)) : 
  ∃ (I: ℝ × ℝ), ∀ (P: ℝ × ℝ), P ≠ I → dist I P = Q.R := sorry

end inscribed_circle_in_quadrilateral_l717_717279


namespace binomial_expansion_max_term_l717_717431

open Finset Nat

theorem binomial_expansion_max_term (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  let c := (n * a - b) / (a + b) in
  if n - 1 < c then
    max (Finset.image (λ k, (choose n k : ℝ) * a^k * b^(n-k)) (range (n+1))) = a^n
  else if c ≤ 0 then
    max (Finset.image (λ k, (choose n k : ℝ) * a^k * b^(n-k)) (range (n+1))) = b^n
  else
    let k := floor c + 1 in
    max (Finset.image (λ k, (choose n k : ℝ) * a^k * b^(n-k)) (range (n+1))) = (choose n k : ℝ) * a^k * b^(n-k) :=
sorry

end binomial_expansion_max_term_l717_717431


namespace tower_height_greater_than_103_3_l717_717217

theorem tower_height_greater_than_103_3 :
  ∀ (H : ℝ), (∀ (d : ℝ) (θ : ℝ), d = 100 → θ = 46 → H = d * Real.tan (θ * Real.pi / 180)) → H > 103.3 :=
by {
  intro H,
  intro hc,
  have h₁ : H = 100 * Real.tan (46 * Real.pi / 180) := hc 100 46 rfl rfl,
  sorry
}

end tower_height_greater_than_103_3_l717_717217


namespace broken_path_exists_l717_717637

theorem broken_path_exists (N : ℕ) (hN : 0 < N) : ∃ (path : set (ℕ × ℕ)), 
  (∀ x, x ∈ path) → x.fst < N ∧ x.snd < N ∧
  (∃ segments, list.length segments = 2 * N - 2 ∧
    (∀ segment, segment ∈ segments → valid_segment path segment)) :=
sorry

def valid_segment (path : set (ℕ × ℕ)) (segment : (ℕ × ℕ) × (ℕ × ℕ)) : Prop :=
  let ((x1, y1), (x2, y2)) := segment
  in (x1, y1) ∈ path ∧ (x2, y2) ∈ path ∧ -- segment's endpoints are on the grid
  ((x1 = x2 ∧ abs (y1 - y2) = 1) ∨ (y1 = y2 ∧ abs (x1 - x2) = 1)) -- segment is valid if horizontal or vertical adjacent

end broken_path_exists_l717_717637


namespace circles_area_l717_717203

theorem circles_area (BD AC : ℝ) (r : ℝ) (h1 : BD = 6) (h2 : AC = 12)
  (h3 : ∀ (d1 d2 : ℝ), d1 = AC / 2 → d2 = BD / 2 → r^2 = (r - d2)^2 + d1^2) :
  real.pi * r^2 = (225/4) * real.pi :=
by
  -- proof to be filled
  sorry

end circles_area_l717_717203


namespace number_of_children_to_movies_l717_717651

theorem number_of_children_to_movies (adult_ticket_cost child_ticket_cost total_money : ℕ) 
  (h_adult_ticket_cost : adult_ticket_cost = 8) 
  (h_child_ticket_cost : child_ticket_cost = 3) 
  (h_total_money : total_money = 35) : 
  (total_money - adult_ticket_cost) / child_ticket_cost = 9 := 
by 
  have remaining_money : ℕ := total_money - adult_ticket_cost
  have h_remaining_money : remaining_money = 27 := by 
    rw [h_total_money, h_adult_ticket_cost]
    norm_num
  rw [h_remaining_money]
  exact nat.div_eq_of_eq_mul_right (by norm_num)
    (by norm_num)

end number_of_children_to_movies_l717_717651


namespace f_3p_nonnegative_l717_717018

def t (m : ℤ) : ℤ := 
  if (m % 3 = 2) then 1 
  else if (m % 3 = 1) then 2 
  else 3 

def f : ℤ → ℤ 
| -1 := 0 
| 0 := 1 
| 1 := -1 
| (2^n + m) := 
  if 2^n > m then f (2^n - t m) - f m
  else f (2^n + m)

theorem f_3p_nonnegative (p : ℤ) (hp : p ≥ 0) : f (3 * p) ≥ 0 :=
sorry

end f_3p_nonnegative_l717_717018


namespace distance_ratio_l717_717132

variables (KD DM : ℝ)

theorem distance_ratio : 
  KD = 4 ∧ (KD + DM + DM + KD = 12) → (KD / DM = 2) := 
by
  sorry

end distance_ratio_l717_717132


namespace problem_eq_answer_l717_717451

theorem problem_eq_answer (x : ℝ) (h : sqrt (10 + x) + sqrt (30 - x) = 8) : 
  (10 + x) * (30 - x) = 144 :=
sorry

end problem_eq_answer_l717_717451


namespace somu_age_ratio_l717_717187

variable {S : ℕ} {F : ℕ}

def somu_present_age (S : ℕ) : Prop := S = 18
def somu_father_age_relation (S F : ℕ) : Prop := S - 9 = (F - 9) / 5

theorem somu_age_ratio 
  (h1 : somu_present_age S)
  (h2 : somu_father_age_relation S F) :
  S / nat.gcd S F = 1 ∧ F / nat.gcd S F = 3 :=
by
  sorry

end somu_age_ratio_l717_717187


namespace no_solution_to_system_l717_717083

-- Definitions for the conditions
def eq1 (x y z : ℤ) := x^2 - 4*x*y + 3*y^2 - z^2 = 24
def eq2 (x y z : ℤ) := -x^2 + 3*y*z + 5*z^2 = 60
def eq3 (x y z : ℤ) := x^2 + 2*x*y + 5*z^2 = 85

-- The proof statement
theorem no_solution_to_system : ∀ (x y z : ℤ), eq1 x y z ∧ eq2 x y z ∧ eq3 x y z → false :=
begin
  sorry
end

end no_solution_to_system_l717_717083


namespace conditional_probability_event_B_given_A_l717_717284

open ProbabilityTheory

noncomputable def coin_toss_probability := (1 : ℝ) / 2

def event_A : Event :=
{ description := "The first toss results in heads",
  probability := coin_toss_probability }

def event_B : Event :=
{ description := "The second toss results in tails",
  probability := coin_toss_probability }

def event_A_and_B : Event :=
{ description := "The first toss results in heads and the second toss results in tails",
  probability := coin_toss_probability * coin_toss_probability }

theorem conditional_probability_event_B_given_A :
  P(event_B | event_A) = (1 : ℝ) / 2 := by
sorry

end conditional_probability_event_B_given_A_l717_717284


namespace lateral_surface_area_of_pyramid_l717_717562

noncomputable def ABCD := Type
noncomputable def P := Type
def AB : ℝ := 18
def BC : ℝ := 10
def S_ABCD : ℝ := 90

def lateral_surface_area (P : Type) [has_lateral_surface_area P] : ℝ := sorry

theorem lateral_surface_area_of_pyramid
  (P ABCD : Type) [has_base ABCD P] [has_lateral_surface_area P]
  (AB BC S_ABCD : ℝ) (h1 : AB = 18) (h2 : BC = 10) (h3 : S_ABCD = 90) :
  lateral_surface_area P = 192 :=
sorry

end lateral_surface_area_of_pyramid_l717_717562


namespace volume_remaining_convex_polyhedron_l717_717660

/-
  A regular hexahedron with an edge length of 1 is cut by planes passing through the common vertex of three edges
  and their respective midpoints. We remove the 8 resulting triangular pyramids. Prove that the volume of the
  remaining convex polyhedron is 5/6.
-/
theorem volume_remaining_convex_polyhedron : 
  ∀ (edge_length : ℝ), 
  (∀ (v₁ v₂ v₃ : ℝ), v₁ = edge_length / 2 ∧ v₂ = edge_length / 2 ∧ v₃ = edge_length / 2) →
  edge_length = 1 →
  (let hexahedron_volume := edge_length^3 in
   let pyramid_volume := (1 / 3) * v₁ * v₂ * v₃ in
   let total_pyramids_volume := 8 * pyramid_volume in
   let remaining_volume := hexahedron_volume - total_pyramids_volume in
   remaining_volume = 5 / 6) := 
sorry

end volume_remaining_convex_polyhedron_l717_717660


namespace kim_boxes_on_tuesday_l717_717175

theorem kim_boxes_on_tuesday
  (sold_on_thursday : ℕ)
  (sold_on_wednesday : ℕ)
  (sold_on_tuesday : ℕ)
  (h1 : sold_on_thursday = 1200)
  (h2 : sold_on_wednesday = 2 * sold_on_thursday)
  (h3 : sold_on_tuesday = 2 * sold_on_wednesday) :
  sold_on_tuesday = 4800 :=
sorry

end kim_boxes_on_tuesday_l717_717175


namespace find_a7_l717_717712

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem find_a7 (a : ℕ → ℝ) (h_geom : geometric_sequence a)
  (h3 : a 3 = 1)
  (h_det : a 6 * a 8 - 8 * 8 = 0) :
  a 7 = 8 :=
sorry

end find_a7_l717_717712


namespace expected_digits_die_roll_l717_717324

noncomputable def expected_number_of_digits : ℝ :=
  let prob_one_digit := 9 / 20
  let prob_two_digits := 11 / 20
  (prob_one_digit * 1) + (prob_two_digits * 2)

theorem expected_digits_die_roll : expected_number_of_digits = 1.55 :=
by {
  -- Definitions from the conditions
  let prob_one_digit := 9 / 20
  let prob_two_digits := 11 / 20

  -- Calculation
  have h1 : prob_one_digit * 1 = 9 / 20, by norm_num,
  have h2 : prob_two_digits * 2 = 22 / 20, by norm_num,
  have h3 : (9 / 20) + (22 / 20) = 31 / 20, by norm_num,

  -- Simplification to get the final result
  have h4 : 31 / 20 = 1.55, by norm_num,

  -- Combining the steps
  rw [expected_number_of_digits, h1, h2, h3, h4],
  exact h4,
}

end expected_digits_die_roll_l717_717324


namespace abs_sum_min_value_l717_717788

open Real

theorem abs_sum_min_value (a : ℚ) :
  (∃ a : ℚ, (2 ≤ a ∧ a ≤ 3) ∧ (|a - 1| + |a - 2| + |a - 3| + |a - 4| = 4)) :=
begin
  sorry
end

end abs_sum_min_value_l717_717788


namespace min_value_2a_b_c_l717_717089

theorem min_value_2a_b_c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * (a + b + c) + b * c = 4) : 
  2 * a + b + c ≥ 4 :=
sorry

end min_value_2a_b_c_l717_717089


namespace rectangle_area_in_triangle_l717_717173

open Classical

theorem rectangle_area_in_triangle
  {EF : ℝ} {G : ℝ} 
  (F_to_EG : ℝ) 
  (EG_length : ℝ)
  (AD_on_EG : ∀ A B C D, A ∈ Segment F_to_EG ∧ D ∈ Segment F_to_EG)
  (length_segment_AB_AD : ∀ A B D, distance A B = (1 / 3) * distance A D)
  (F_altitude : F_to_EG = 9)
  (EG_equal_to_12 : EG_length = 12):
  ∃ AB AD, 
    let A := (AB: ℝ), D := (AD: ℝ) in 
    (A, D).1 * (A, D).2 = 432 / 49 := 
  by
    sorry 

end rectangle_area_in_triangle_l717_717173


namespace count_invertible_mod12_l717_717717

theorem count_invertible_mod12 : 
  {a : ℕ | a > 0 ∧ a < 12 ∧ ∃ x : ℕ, (a * x) % 12 = 1}.toFinset.card = 4 :=
by
  sorry

end count_invertible_mod12_l717_717717


namespace exists_set_with_subsets_l717_717534

theorem exists_set_with_subsets :
  ∃ (S : Finset ℕ) (A : Finset (Finset ℕ)),
    S.card = 15 ∧ A.card = 15 ∧
    (∀ (B ∈ A), B.card = 6) ∧
    (∀ (B1 B2 ∈ A), B1 ≠ B2 → 
       (B1 ∩ B2).card = 1 ∨ (B1 ∩ B2).card = 3) := 
sorry

end exists_set_with_subsets_l717_717534


namespace sum_4digit_numbers_remainder_3_l717_717252

theorem sum_4digit_numbers_remainder_3
  (LCM : ℕ := 35)
  (is_4digit : ℕ → Prop := λ n, n >= 1000 ∧ n <= 9999)
  (leaves_remainder_3 : ℕ → Prop := λ n, n % LCM = 3)
  (numbers : list ℕ := list.range' 1000 (9999 - 1000 + 1))
  : (numbers.filter (λ n, leaves_remainder_3 n)).sum = 1414773 := by
  sorry

end sum_4digit_numbers_remainder_3_l717_717252


namespace total_dots_is_78_l717_717322

-- Define the conditions as Lean definitions
def ladybugs_monday : ℕ := 8
def ladybugs_tuesday : ℕ := 5
def dots_per_ladybug : ℕ := 6

-- Define the total number of ladybugs
def total_ladybugs : ℕ := ladybugs_monday + ladybugs_tuesday

-- Define the total number of dots
def total_dots : ℕ := total_ladybugs * dots_per_ladybug

-- Theorem stating the problem to solve
theorem total_dots_is_78 : total_dots = 78 := by
  sorry

end total_dots_is_78_l717_717322


namespace tangent_is_simson_line_l717_717532

noncomputable def triangle_orthocenter (A B C: Point) := 
  ∃ H: Point, is_orthocenter H A B C

noncomputable def midpoint (P Q: Point) := 
  (P + Q) / 2

noncomputable def simson_line (Tangent: Line)(Triangle: Triangle) := 
  ∃ A B C: Point, ∃ H: Point, ∃ A1 B1 C1: Point, 
  (is_orthocenter H A B C) ∧ 
  (A1 = midpoint A H) ∧ 
  (B1 = midpoint B H) ∧ 
  (C1 = midpoint C H) 
  ∧ (Tangent = simson A B C)

theorem tangent_is_simson_line (Tangent: Line) (Parabola: Parabola) (A B C: Point) (H: Point) (A1 B1 C1: Point) :
  (is_tangent_line Tangent Parabola (vertex Parabola)) →
  (is_orthocenter H A B C) →
  (A1 = midpoint A H) →
  (B1 = midpoint B H) →
  (C1 = midpoint C H) →
  simson_line Tangent (triangle_orthocenter A B C) :=
by
  sorry

end tangent_is_simson_line_l717_717532


namespace possible_dimensions_of_plot_l717_717301

theorem possible_dimensions_of_plot (x : ℕ) :
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ 1000 * a + 100 * a + 10 * b + b = x * (x + 1)) →
  x = 33 ∨ x = 66 ∨ x = 99 :=
sorry

end possible_dimensions_of_plot_l717_717301


namespace length_after_haircut_l717_717480

-- Definitions
def original_length : ℕ := 18
def cut_length : ℕ := 9

-- Target statement to prove
theorem length_after_haircut : original_length - cut_length = 9 :=
by
  -- Simplification and proof
  sorry

end length_after_haircut_l717_717480


namespace complex_in_fourth_quadrant_l717_717147

def complex_quadrant (z : ℂ) : ℕ := 
if z.re > 0 ∧ z.im > 0 then 1
else if z.re < 0 ∧ z.im > 0 then 2
else if z.re < 0 ∧ z.im < 0 then 3
else if z.re > 0 ∧ z.im < 0 then 4
else 0 -- in case of boundary conditions

theorem complex_in_fourth_quadrant :
  let z := (⟨1, -3⟩ : ℂ) * (⟨2, 1⟩ : ℂ) in 
  complex_quadrant z = 4 :=
by
  sorry


end complex_in_fourth_quadrant_l717_717147


namespace volume_of_rect_prism_l717_717661

theorem volume_of_rect_prism (l w h : ℝ) 
  (h1 : l * w = 15) 
  (h2 : w * h = 10) 
  (h3 : l * h = 30) : 
  l * w * h = 30 * real.sqrt 5 := 
  sorry

end volume_of_rect_prism_l717_717661


namespace min_value_sin_cos_l717_717374

open Real

theorem min_value_sin_cos : ∀ x : ℝ, 
  ∃ (y : ℝ), (∀ x, y ≤ sin x ^ 6 + (5 / 3) * cos x ^ 6) ∧ y = 5 / 8 :=
by
  sorry

end min_value_sin_cos_l717_717374


namespace angle_a_c_is_60_degrees_l717_717043

-- Define the vectors and their magnitudes
variables {α : Type*} [inner_product_space ℝ α]
variables (a b c : α) (t : ℝ)

-- Conditions
def condition1 : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
def condition2 : Prop := ∥a∥ = t ∧ ∥b∥ = t ∧ ∥c∥ = t
def condition3 : Prop := a + b = c

-- Theorem statement
theorem angle_a_c_is_60_degrees (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  real.angle a c = real.pi / 3 :=
sorry

end angle_a_c_is_60_degrees_l717_717043


namespace max_point_of_f_l717_717789

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Define the first derivative of the function
def f_prime (x : ℝ) : ℝ := 3 * x^2 - 12

-- Define the second derivative of the function
def f_double_prime (x : ℝ) : ℝ := 6 * x

-- Prove that a = -2 is the maximum value point of f(x)
theorem max_point_of_f : ∃ a : ℝ, (f_prime a = 0) ∧ (f_double_prime a < 0) ∧ (a = -2) :=
sorry

end max_point_of_f_l717_717789


namespace sin_120_eq_sqrt3_div_2_l717_717268

theorem sin_120_eq_sqrt3_div_2 : sin (120 * Real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end sin_120_eq_sqrt3_div_2_l717_717268


namespace cookie_radius_l717_717938

theorem cookie_radius (x y : ℝ) : x^2 + y^2 + 28 = 6*x + 20*y → ∃ r, r = 9 :=
by
  sorry

end cookie_radius_l717_717938


namespace log_base_ten_five_l717_717404

open Real

noncomputable def solution (a b : ℝ) (h1 : logBase 8 3 = a) (h2 : logBase 3 5 = b) : ℝ :=
(evaluate: ℝ): Prop := logBase 10 5 = (3 * a * b) / (1 + (3 * a * b))

theorem log_base_ten_five (a b : ℝ) (h1 : Real.logBase 8 3 = a) (h2 : Real.logBase 3 5 = b) : 
  Real.logBase 10 5 = (3 * a * b) / (1 + 3 * a * b) := sorry

end log_base_ten_five_l717_717404


namespace find_a7_l717_717707

variable (a : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n+1) = r * a n

axiom a3_eq_1 : a 3 = 1
axiom det_eq_0 : a 6 * a 8 - 8 * 8 = 0

theorem find_a7 (h_geom : geometric_sequence a) : a 7 = 8 :=
  sorry

end find_a7_l717_717707


namespace james_can_make_sushi_rolls_l717_717872

def fish_per_sushi_roll : Nat := 40
def total_fish_bought : Nat := 400
def percentage_bad_fish : Real := 0.20

theorem james_can_make_sushi_rolls : 
  (total_fish_bought - Nat.floor((percentage_bad_fish * total_fish_bought : Real))) / fish_per_sushi_roll = 8 := 
by
  sorry

end james_can_make_sushi_rolls_l717_717872


namespace train_pass_bridge_time_correct_l717_717619

noncomputable def train_pass_bridge_time 
  (length_of_train : ℝ)
  (length_of_bridge : ℝ)
  (speed_of_train_kmh : ℝ) : ℝ :=
  let total_distance := length_of_train + length_of_bridge
  let speed_of_train_ms := speed_of_train_kmh * (1000 / 3600)
  total_distance / speed_of_train_ms

theorem train_pass_bridge_time_correct :
  train_pass_bridge_time 480 260 85 ≈ 31.33 := sorry

end train_pass_bridge_time_correct_l717_717619


namespace triangle_statements_l717_717104

theorem triangle_statements (A B C : ℝ) (h1 : ∠A + ∠B + ∠C = π) (h2 : tan ((A + B) / 2) = sin C) :
  ((1 < sin A + sin B ∧ sin A + sin B ≤ sqrt 2) ∧ (cos^2 A + cos^2 B = sin^2 C)) ∧
  ¬(tan A * cot B = 1) ∧ ¬(sin^2 A + cos^2 B = 1) :=
by sorry

end triangle_statements_l717_717104


namespace axis_symmetry_center_symmetry_inequality_holds_l717_717080

noncomputable def a := (cos x)^2 - (sin x)^2
noncomputable def b := (1 / 2 : ℝ, (sin x)^2 + real.sqrt 3 * (sin x) * (cos x))

noncomputable def f (x : ℝ) := (a, 1 / 2 : ℝ) • (1 / 2 : ℝ, b)

theorem axis_symmetry (k : ℤ) :
  ∃ x, 2 * x + π / 6 = π / 2 + k * π :=
sorry

theorem center_symmetry (k : ℤ) :
  ∃ x, 2 * x + π / 6 = k * π :=
sorry

noncomputable def g (x : ℝ) := 1 / 2 * sin (4 * x + 5 * π / 6) + 1 / 4

theorem inequality_holds (a : ℝ) :
  (a ≥ 2 / 3 ∨ a ≤ -1 / 2) ↔ (∀ x ∈ Icc (-π / 4) (π / 8), 6 * a ^ 2 - a - 5 / 4 ≥ g x) :=
sorry

end axis_symmetry_center_symmetry_inequality_holds_l717_717080


namespace prime_divisor_bound_l717_717143

theorem prime_divisor_bound (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : 2 ≤ n) 
  (hdiv : p^2 ∣ ∏ k in Finset.range n.succ, (k + 1)^3 + 1) : p ≤ 2 * n - 1 := 
sorry

end prime_divisor_bound_l717_717143


namespace f_five_eq_half_l717_717426

def f : ℝ → ℝ :=
λ x, if x ≤ 0 then 2 ^ x else sorry

theorem f_five_eq_half : f 5 = 1 / 2 :=
sorry

end f_five_eq_half_l717_717426


namespace smallest_pos_multiple_6_15_is_30_l717_717753

theorem smallest_pos_multiple_6_15_is_30 :
  ∃ b > 0, 6 ∣ b ∧ 15 ∣ b ∧ (∀ b', b' > 0 ∧ b' < b → ¬ (6 ∣ b' ∧ 15 ∣ b')) :=
by
  -- Implementation to be done
  sorry

end smallest_pos_multiple_6_15_is_30_l717_717753


namespace abs_value_h_l717_717971

theorem abs_value_h (h : ℝ) (r s : ℝ) (H1 : r^2 + s^2 = 18) (H2 : r + s = -2 * h) (H3 : r * s = -8) :
  |h| = real.sqrt (1 / 2) / 2 := 
  sorry

end abs_value_h_l717_717971


namespace prism_lateral_edges_and_faces_l717_717608

/-- Prove that a prism has equal lateral edges and that its lateral faces are parallelograms. --/
theorem prism_lateral_edges_and_faces (P : Prism) :
  (∀ (e1 e2 : Edge), e1 ∈ lateral_edges P → e2 ∈ lateral_edges P → Edge.length e1 = Edge.length e2) ∧
  (∀ (f : Face), f ∈ lateral_faces P → is_parallelogram f) :=
sorry

end prism_lateral_edges_and_faces_l717_717608


namespace incenter_inequality_l717_717895

theorem incenter_inequality (A B C I A' B' C' : Type)
    [h_triangle : triangle ABC]
    (h_incenter : incenter ABC I)
    (h_angle_bisectors : angle_bisectors ABC I A' B' C') :
  (1 : ℝ)/4 ≤ (AI * BI * CI) / (AA' * BB' * CC') ∧ (AI * BI * CI) / (AA' * BB' * CC') ≤ (8 : ℝ)/27 := 
sorry

end incenter_inequality_l717_717895


namespace max_non_divisible_sum_l717_717152

def sum_arith_seq (a : ℕ) (b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def erased_sum_condition (s : ℕ) : Prop :=
  ¬ ∃ (groups : list (list ℕ)), 
    (∀ g ∈ groups, (∀ x ∈ g, 4 ≤ x ∧ x ≤ 16) ∧ (∑ x in g, x) = s / groups.length) ∧ 
    (∑ g in groups, ∑ x in g, x = s)

theorem max_non_divisible_sum :
  erased_sum_condition (sum_arith_seq 4 16 - 9) :=
sorry

end max_non_divisible_sum_l717_717152


namespace gcd_360_504_is_72_l717_717221

theorem gcd_360_504_is_72 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_is_72_l717_717221


namespace petya_can_prevent_natural_sum_l717_717913

def petya_prevents_natural_sum : Prop :=
  ∀ (fractions : List ℚ),
    (∀ f ∈ fractions, ∃ n : ℕ, f = 1 / n) →
    ∃ m : ℕ, let new_fractions := (1 / m) :: fractions in
    ∀ k : ℕ, 
      k > 0 →
      let vasya_fractions := new_fractions.take k in
      ∑ i in vasya_fractions, id i ∉ ℕ

theorem petya_can_prevent_natural_sum : petya_prevents_natural_sum :=
sorry

end petya_can_prevent_natural_sum_l717_717913


namespace find_x_value_l717_717763

theorem find_x_value (x : ℝ) (h : 2^(x+4) = 216) : x = 3.75 :=
sorry

end find_x_value_l717_717763


namespace other_root_of_quadratic_l717_717382

variable {a : ℝ}

theorem other_root_of_quadratic (h : IsRoot (λ x : ℝ, x^2 + x - a) 2) : IsRoot (λ x : ℝ, x^2 + x - a) (-3) :=
sorry

end other_root_of_quadratic_l717_717382


namespace find_g_expression_l717_717064

theorem find_g_expression (f g : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f x = 2 * x + 3)
  (h2 : ∀ x : ℝ, g (x + 2) = f x) :
  ∀ x : ℝ, g x = 2 * x - 1 :=
by
  sorry

end find_g_expression_l717_717064


namespace normal_line_eqn_l717_717169

variables {α β : Type*} [LinearOrderedField α] [TopologicalSpace β]
  {f : α → β} {x₀ : α} {y₀ : β}

-- Conditions: A differentiable function f, a point (x₀, y₀) on the curve, and the derivative of f
theorem normal_line_eqn (hf : DifferentiableAt ℝ f x₀) (hy₀ : f x₀ = y₀) :
  -f'(x₀) * (y - y₀) = x - x₀ := 
sorry

end normal_line_eqn_l717_717169


namespace initial_boys_down_slide_l717_717183

variable (B : Int)

theorem initial_boys_down_slide:
  B + 13 = 35 → B = 22 := by
  sorry

end initial_boys_down_slide_l717_717183


namespace sequence_sum_fraction_l717_717075

theorem sequence_sum_fraction :
  (∀ n: ℕ, a n = if n = 0 then 1 else a (n-1) + n + 1) →
  (∑ n in range 2006, 1 / (a n)) = 4032 / 2017 :=
by
  intro h
  -- Insert the definitions and loop through the sequence
  sorry

end sequence_sum_fraction_l717_717075


namespace simplify_expression_l717_717538

-- Define the statement we want to prove
theorem simplify_expression (s : ℕ) : (105 * s - 63 * s) = 42 * s :=
  by
    -- Placeholder for the proof
    sorry

end simplify_expression_l717_717538


namespace domain_of_function_l717_717734

/-- The function f(x) = sqrt(4 - sqrt(5 - sqrt(6 - x))) is well-defined within the interval [-19, 6] -/
theorem domain_of_function :
  ∀ x, (-19 ≤ x ∧ x ≤ 6) ↔
       (∃ y, y = sqrt(4 - sqrt(5 - sqrt(6 - x)))) :=
by
  sorry

end domain_of_function_l717_717734


namespace sum_of_ages_l717_717275

theorem sum_of_ages (P K : ℕ) (h1 : P - 7 = 3 * (K - 7)) (h2 : P + 2 = 2 * (K + 2)) : P + K = 50 :=
by
  sorry

end sum_of_ages_l717_717275


namespace number_of_children_l717_717653

-- Definitions for the conditions
def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 3
def total_amount : ℕ := 35

-- Theorem stating the proof problem
theorem number_of_children (A C T : ℕ) (hc: A = adult_ticket_cost) (ha: C = child_ticket_cost) (ht: T = total_amount) :
  (T - A) / C = 9 :=
by
  sorry

end number_of_children_l717_717653


namespace rectangle_bisector_bd_eq_third_l717_717535

theorem rectangle_bisector_bd_eq_third (P : ℝ) (hP : 0 ≤ P ∧ P ≤ 3) :
  let A := (0, 0)
  let B := (3, 0)
  let C := (3, 2)
  let D := (0, 2)
  let M := (3, 1)
  let P' := (P, 0)
  (P' <| midpoint A B) → (angle_bisector CD P' M) → 
  (λ (P'' ∈ line_segment A B),  BP_distance := distance_btw B P') = 
  (1/3) := sorry

end rectangle_bisector_bd_eq_third_l717_717535


namespace sugar_amount_l717_717621

variable {S F B : ℝ}

-- Conditions from the problem
def condition1 : Prop := S / F = 5 / 6
def condition2 : Prop := F / B = 10 / 1
def condition3 : Prop := F / (B + 60) = 8 / 1

-- Final statement to prove
theorem sugar_amount (h1 : condition1) (h2 : condition2) (h3 : condition3) : S = 2000 :=
  sorry

end sugar_amount_l717_717621


namespace find_x_in_terms_of_z_l717_717190

variable (z : ℝ)
variable (x y : ℝ)

theorem find_x_in_terms_of_z (h1 : 0.35 * (400 + y) = 0.20 * x) 
                             (h2 : x = 2 * z^2) 
                             (h3 : y = 3 * z - 5) : 
  x = 2 * z^2 :=
by
  exact h2

end find_x_in_terms_of_z_l717_717190


namespace not_prime_for_some_n_l717_717880

theorem not_prime_for_some_n (x y : ℤ) (hx : 2 ≤ x) (hx2 : x ≤ 100) (hy : 2 ≤ y) (hy2 : y ≤ 100) :
  ∃ n : ℕ+, ¬ Nat.Prime (x ^ (2^n) + y ^ (2^n)) :=
sorry

end not_prime_for_some_n_l717_717880


namespace journal_sessions_per_week_l717_717967

/-- Given that each student writes 4 pages in each session and will write 72 journal pages in 6 weeks, prove that there are 3 journal-writing sessions per week.
--/
theorem journal_sessions_per_week (pages_per_session : ℕ) (total_pages : ℕ) (weeks : ℕ) (sessions_per_week : ℕ) :
  pages_per_session = 4 →
  total_pages = 72 →
  weeks = 6 →
  total_pages = pages_per_session * sessions_per_week * weeks →
  sessions_per_week = 3 :=
by
  intros h1 h2 h3 h4
  sorry

end journal_sessions_per_week_l717_717967


namespace distance_walked_is_18_miles_l717_717650

-- Defining the variables for speed, time, and distance
variables (x t d : ℕ)

-- Declaring the conditions given in the problem
def walked_distance_at_usual_rate : Prop :=
  d = x * t

def walked_distance_at_increased_rate : Prop :=
  d = (x + 1) * (3 * t / 4)

def walked_distance_at_decreased_rate : Prop :=
  d = (x - 1) * (t + 3)

-- The proof problem statement to show the distance walked is 18 miles
theorem distance_walked_is_18_miles
  (hx : walked_distance_at_usual_rate x t d)
  (hz : walked_distance_at_increased_rate x t d)
  (hy : walked_distance_at_decreased_rate x t d) :
  d = 18 := by
  sorry

end distance_walked_is_18_miles_l717_717650


namespace solve_equation_l717_717965

theorem solve_equation :
  ∀ x : ℝ, (4 * x - 2 * x + 1 - 3 = 0) ↔ (x = 1 ∨ x = -1) :=
by
  intro x
  sorry

end solve_equation_l717_717965


namespace boxer_weight_l717_717903

noncomputable theory

variable (x : ℕ) -- initial weight of the boxer
variable (y : ℕ) -- weight loss per month

def dietA := 2 -- weight loss for Diet A
def dietB := 3 -- weight loss for Diet B
def dietC := 4 -- weight loss for Diet C

variable (current_weight : ℕ := 97) -- weight of the boxer 4 months from the fight
variable (months : ℕ := 4) -- number of months

theorem boxer_weight :
  (x = current_weight + dietB * months) ∧
  (x - dietA * months = 101) ∧
  (x - dietB * months = 97) ∧
  (x - dietC * months = 93) :=
by
  split; sorry
  split; sorry
  split; sorry
  sorry

end boxer_weight_l717_717903


namespace inequality_solution_l717_717181

theorem inequality_solution (x : ℝ) :
  (\frac{9 * x^2 + 18 * x - 60}{(3 * x - 4) * (x + 5)} < 2) ↔ 
  (x ∈ Set.Ioo (-5 / 3) (4 / 3) ∪ Set.Ioi 4) :=
by
  sorry

end inequality_solution_l717_717181


namespace sum_of_cubes_eq_square_of_sum_l717_717514

theorem sum_of_cubes_eq_square_of_sum (n : ℕ) (hn : n > 0) : (((Finset.range n).map (λ x, x + 1)).sum (λ x, x^3)) = (n * (n + 1) / 2)^2 := by
  sorry

end sum_of_cubes_eq_square_of_sum_l717_717514


namespace cannot_determine_shape_l717_717390

noncomputable def is_point (P : ℝ × ℝ) := ∃ x y : ℝ, P = (x, y)

def rectangle_points : (ℝ × ℝ) → Prop :=
λ A, (A = (0, 0)) ∨ (A = (0, 4)) ∨ (A = (6, 4)) ∨ (A = (6, 0))

def line_eq_from_A_45 (x : ℝ) : ℝ := x
def line_eq_from_A_75 (x : ℝ) : ℝ := real.tan (75 * real.pi / 180) * x

def line_eq_from_B_neg45 (x : ℝ) : ℝ := 4 - x
def line_eq_from_B_neg75 (x : ℝ) : ℝ := 4 - (real.tan (75 * real.pi / 180) * x)

def intersect (f g : ℝ → ℝ) : (ℝ × ℝ) :=
  let x := (4 - (real.tan (75 * real.pi / 180))) / (1 + real.tan (75 * real.pi / 180)) in
  (x, f x)

theorem cannot_determine_shape :
  ¬ ∃ P1 P2 : ℝ × ℝ,
    is_point P1 ∧ is_point P2 ∧
    intersect line_eq_from_A_45 line_eq_from_B_neg45 = P1 ∧
    intersect line_eq_from_A_75 line_eq_from_B_neg75 = P2 ∧
    ∀ P, ¬rectangle_points P :=
sorry

end cannot_determine_shape_l717_717390


namespace length_chord_2sqrt3_l717_717958

noncomputable def length_of_chord (x : ℝ) (y : ℝ) (t : ℝ) : ℝ :=
  let t1 := ... -- solve for t1 using parametric equation and circle equation
  let t2 := ... -- solve for t2 using parametric equation and circle equation
  |t2 - t1|

theorem length_chord_2sqrt3 :
  let x := -4 + (Real.sqrt 3) / 2 * t in
  let y := 1 / 2 * t in
  let circle_eq := x^2 + y^2 = 7 in
  length_of_chord x y t = 2 * Real.sqrt 3 :=
sorry

end length_chord_2sqrt3_l717_717958


namespace least_integer_x_l717_717590

theorem least_integer_x (x : ℤ) :
  (∀ y : ℤ,  |3 * y + 4| ≤ 18 → x ≤ y) ∧ (|3 * x + 4| ≤ 18) → x = -7 :=
by
  have h := sorry
  exact h

end least_integer_x_l717_717590


namespace number_of_children_to_movies_l717_717652

theorem number_of_children_to_movies (adult_ticket_cost child_ticket_cost total_money : ℕ) 
  (h_adult_ticket_cost : adult_ticket_cost = 8) 
  (h_child_ticket_cost : child_ticket_cost = 3) 
  (h_total_money : total_money = 35) : 
  (total_money - adult_ticket_cost) / child_ticket_cost = 9 := 
by 
  have remaining_money : ℕ := total_money - adult_ticket_cost
  have h_remaining_money : remaining_money = 27 := by 
    rw [h_total_money, h_adult_ticket_cost]
    norm_num
  rw [h_remaining_money]
  exact nat.div_eq_of_eq_mul_right (by norm_num)
    (by norm_num)

end number_of_children_to_movies_l717_717652


namespace range_s_l717_717720

open Set

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^3

theorem range_s : range s = univ \ {s 2} :=
by
  rw [univ_eq_of_forall_mem, ←image_univ]
  have : ∀ y, ∃ x (hx : x ≠ 2), s x = y,
  { intro y,
    use [2 + real.cbrt (1/y), by {
        rw [ereal.cbrt_infi_eq_infi_cbrt_to_real hessian_earth_kent],
        exact 2_ne_3]}, sorry] }

  exact eq.symm (range_eq_of_forall_mem this)

end range_s_l717_717720


namespace units_digit_17_pow_2007_l717_717596

theorem units_digit_17_pow_2007 :
  (17 ^ 2007) % 10 = 3 := 
sorry

end units_digit_17_pow_2007_l717_717596


namespace four_digit_prime_factor_property_l717_717366

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def sum_prime_factors_eq_sum_exponents (n : ℕ) : Prop :=
  let primes := n.factors in
  primes.sum = (primes.map (λ p => (p.count)).sum

def valid_numbers (s : List ℕ) : Prop :=
  s = [1792, 2000, 3125, 3840, 5000, 5760, 6272, 8640, 9600]

theorem four_digit_prime_factor_property :
  ∀ n, is_four_digit n ∧ sum_prime_factors_eq_sum_exponents n → n ∈ [1792, 2000, 3125, 3840, 5000, 5760, 6272, 8640, 9600] :=
sorry

end four_digit_prime_factor_property_l717_717366


namespace uncovered_area_frame_l717_717957

def length_frame : ℕ := 40
def width_frame : ℕ := 32
def length_photo : ℕ := 32
def width_photo : ℕ := 28

def area_frame (l_f w_f : ℕ) : ℕ := l_f * w_f
def area_photo (l_p w_p : ℕ) : ℕ := l_p * w_p

theorem uncovered_area_frame :
  area_frame length_frame width_frame - area_photo length_photo width_photo = 384 :=
by
  sorry

end uncovered_area_frame_l717_717957


namespace possible_arrangements_l717_717550

-- The five materials
inductive Material
| metal
| wood
| earth
| water
| fire

open Material

-- The conquering relation
def conquers (a b : Material) : Prop :=
  match a, b with
  | metal, wood => true
  | wood, earth => true
  | earth, water => true
  | water, fire => true
  | fire, metal => true
  | _, _ => false
  end

-- Define the statement
theorem possible_arrangements:
  (∃ l : list Material, l.length = 5 ∧
    ∀ i, i < 4 → ¬ conquers (l.nth_le i (by linarith)) (l.nth_le (i + 1) (by linarith))) :=
    sorry

end possible_arrangements_l717_717550


namespace sum_of_arithmetic_series_l717_717969

variable {α : Type*} [LinearOrderedField α]

theorem sum_of_arithmetic_series (a : ℕ → α) (m : ℕ) (H₁ : ∑ i in finset.range m, a i = 30) 
  (H₂ : ∑ i in finset.range (2 * m), a i = 100) :
  ∑ i in finset.range (3 * m), a i = 210 :=
sorry

end sum_of_arithmetic_series_l717_717969


namespace initial_boys_down_slide_l717_717184

variable (B : Int)

theorem initial_boys_down_slide:
  B + 13 = 35 → B = 22 := by
  sorry

end initial_boys_down_slide_l717_717184


namespace angles_arithmetic_progression_l717_717624

theorem angles_arithmetic_progression (A B C : ℝ) (h_sum : A + B + C = 180) :
  (B = 60) ↔ (A + C = 2 * B) :=
by
  sorry

end angles_arithmetic_progression_l717_717624


namespace max_integer_roots_of_self_centered_polynomial_l717_717657

def is_self_centered (p : ℤ[X]) : Prop :=
  p.coeff 50 = 1 ∧ p.eval 50 = 50

theorem max_integer_roots_of_self_centered_polynomial (p : ℤ[X]) (h_sc : is_self_centered p) :
  (∀ k : ℤ, p.eval k = k^3) -> ∃ (roots : Finset ℤ), roots.card = 8 :=
sorry

end max_integer_roots_of_self_centered_polynomial_l717_717657


namespace sum_of_solutions_g_eq_0_l717_717508

def g (x : ℝ) : ℝ :=
  if x ≤ -2 then 2 * x + 6 else -(x / 3) - 2

theorem sum_of_solutions_g_eq_0 : (∑ x in {x | g x = 0}, x) = -3 :=
  sorry

end sum_of_solutions_g_eq_0_l717_717508


namespace no_term_divisible_by_4_not_prime_minus_22_l717_717231

-- Definition of the sequence (a_n)
def sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 1 ∧ a 1 = 1 ∧ ∀ n, a (n + 2) = a (n + 1) * a n + 1

-- Part a) No term of the sequence is divisible by 4
theorem no_term_divisible_by_4 (a : ℕ → ℕ) (h : sequence a) : ∀ n, 4 ∣ a n := by
  sorry

-- Part b) a_n - 22 is not a prime number for all n > 10
theorem not_prime_minus_22 (a : ℕ → ℕ) (h : sequence a) : ∀ n, n > 10 → ¬ Prime (a n - 22) := by
  sorry

end no_term_divisible_by_4_not_prime_minus_22_l717_717231


namespace prove_p_and_q_l717_717396

def p : Prop := ∀ x : ℝ, (0 < x) → x > Real.sin x

def q : Prop := ∃ x : ℝ, (1/2)^x = Real.log base (1/2) x

theorem prove_p_and_q : p ∧ q :=
by
  sorry

end prove_p_and_q_l717_717396


namespace average_number_of_carnations_l717_717979

-- Define the conditions in Lean
def number_of_bouquet_1 : ℕ := 9
def number_of_bouquet_2 : ℕ := 14
def number_of_bouquet_3 : ℕ := 13
def total_bouquets : ℕ := 3

-- The main statement to be proved
theorem average_number_of_carnations : 
  (number_of_bouquet_1 + number_of_bouquet_2 + number_of_bouquet_3) / total_bouquets = 12 := 
by
  sorry

end average_number_of_carnations_l717_717979


namespace designate_cube_vertices_on_sphere_l717_717034

theorem designate_cube_vertices_on_sphere :
  ∀ (S : Type) [metric_space S] (sphere : S) 
  (cap : S) (markings : ℕ → ℝ) (great_circle : ℝ → S), 
  ∃ (A B A* B* C C* D D* : S),
    A ≠ B ∧
    A = great_circle 0 ∧ 
    B = great_circle 90 ∧ 
    A* = great_circle 180 ∧ 
    B* = great_circle 270 ∧ 
    perpendicular (A, B) ∧
    perpendicular (A, C) ∧
    perpendicular (A, D) ∧
    ... -- continue with the necessary geometric conditions
  sorry

end designate_cube_vertices_on_sphere_l717_717034


namespace units_digit_of_17_pow_2007_l717_717600

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_2007 : units_digit (17 ^ 2007) = 3 := by
  have h : ∀ n, units_digit (17 ^ n) = units_digit (7 ^ n) := by
    intro n
    sorry  -- Same units digit logic for powers of 17 as for powers of 7.
  have pattern : units_digit (7 ^ 1) = 7 ∧ 
                 units_digit (7 ^ 2) = 9 ∧ 
                 units_digit (7 ^ 3) = 3 ∧ 
                 units_digit (7 ^ 4) = 1 := by
    sorry  -- Units digit pattern for powers of 7.
  have mod_cycle : 2007 % 4 = 3 := by
    sorry  -- Calculation of 2007 mod 4.
  have result : units_digit (7 ^ 2007) = units_digit (7 ^ 3) := by
    rw [← mod_eq_of_lt (by norm_num : 2007 % 4 < 4), mod_cycle]
    exact (and.left (and.right (and.right pattern)))  -- Extract units digit of 7^3 from pattern.
  rw [h]
  exact result

end units_digit_of_17_pow_2007_l717_717600


namespace lattice_intersections_l717_717692

theorem lattice_intersections :
  let lattice_points := {(i : ℤ, j : ℤ) | true},
      circles := {(i, j) | (i, j) ∈ lattice_points ∧ radius (circle_center (i, j)) = 1 / 5},
      squares := {(i, j) | (i, j) ∈ lattice_points ∧ side_length (square_center (i, j)) = 1 / 4},
      line_segment := line_segment (0, 0) (703, 301),
      p := count_intersections line_segment squares,
      q := count_intersections line_segment circles
  in p + q = 576 :=
by
  sorry

end lattice_intersections_l717_717692


namespace largest_tile_size_l717_717617

/-- Define the dimensions of the courtyard in centimeters -/
def length_courtyard := 378
def width_courtyard := 525

/-- Define the greatest common divisor function -/
def gcd (a b : ℕ) : ℕ :=
if b = 0 then a else gcd b (a % b)

/-- Statement of the problem to be proven in Lean 4 -/
theorem largest_tile_size :
  gcd length_courtyard width_courtyard = 21 :=
sorry

end largest_tile_size_l717_717617


namespace triangle_area_inradius_l717_717569

theorem triangle_area_inradius
  (perimeter : ℝ) (inradius : ℝ) (area : ℝ)
  (h1 : perimeter = 35)
  (h2 : inradius = 4.5)
  (h3 : area = inradius * (perimeter / 2)) :
  area = 78.75 := by
  sorry

end triangle_area_inradius_l717_717569


namespace percentage_of_filled_seats_l717_717847

theorem percentage_of_filled_seats (total_seats vacant_seats : ℕ) (h_total : total_seats = 600) (h_vacant : vacant_seats = 240) :
  (total_seats - vacant_seats) * 100 / total_seats = 60 :=
by
  sorry

end percentage_of_filled_seats_l717_717847


namespace solve_trig_eq_l717_717995

theorem solve_trig_eq (x : ℝ) :
  (sin x + cos x + sin (2 * x) + sqrt 2 * sin (5 * x) = 2 * cos x * sin x) ↔ 
  (∃ k : ℤ, x = -((π / 24) : ℝ) + (k : ℝ) * (π / 3)) ∨ 
  (∃ n : ℤ, x = (5 * π / 16 : ℝ) + (n : ℝ) * (π / 2)) :=
sorry

end solve_trig_eq_l717_717995


namespace percentage_increase_is_25_percent_l717_717675

theorem percentage_increase_is_25_percent (a : ℝ) (h : a > 0) :
  ∀ x : ℝ, (a * (1 + x) * 0.8 = a) → x = 0.25 :=
by
  assume x,
  intro h_eq,
  sorry

end percentage_increase_is_25_percent_l717_717675


namespace acute_angles_of_right_triangle_l717_717033
open EuclideanGeometry

/-
Given a right-angled triangle \( \Delta ABC \) such that \( \angle ABC = 90^\circ \),
a median \( BD \) drawn from \( B \) to the hypotenuse \( AC \),
\( K \) as the point of tangency of side \( AD \) with the incircle of \( \Delta ABD \),
and that \( K \) bisects \( AD \), we need to prove that the acute angles of \( \Delta ABC \)
are \( 30^\circ \) and \( 60^\circ \).
-/
theorem acute_angles_of_right_triangle (A B C D K : Point)
    (hright : right_angle B A C)
    (hmedian : is_median B D A C)
    (hincenter : is_incenter K (triangle ABD))
    (hbisection : midpoint K A D) :
    ∃ α β : Real, α + β = 90 ∧ (α = 30 ∧ β = 60 ∨ α = 60 ∧ β = 30) := by
  sorry

end acute_angles_of_right_triangle_l717_717033


namespace selling_price_of_cycle_l717_717259

theorem selling_price_of_cycle (original_price : ℝ) (loss_percentage : ℝ) (loss_amount : ℝ) (selling_price : ℝ) :
  original_price = 2000 →
  loss_percentage = 10 →
  loss_amount = (loss_percentage / 100) * original_price →
  selling_price = original_price - loss_amount →
  selling_price = 1800 :=
by
  intros
  sorry

end selling_price_of_cycle_l717_717259


namespace inverse_89_mod_90_l717_717360

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  sorry -- proof goes here

end inverse_89_mod_90_l717_717360


namespace circles_intersection_probability_l717_717240

noncomputable def probability_intersection_of_circles : Prop :=
  let radiusA := 2
  let radiusB := 1.5
  let segmentA_start := (0 : ℝ, 0 : ℝ)
  let segmentA_end := (3 : ℝ, 0 : ℝ)
  let segmentB_start := (1 : ℝ, 2 : ℝ)
  let segmentB_end := (4 : ℝ, 2 : ℝ)
  -- Uniform and independent random selections of A_X and B_X over their respective segments
  let rangeA := segmentA_end.1 - segmentA_start.1 -- length of A's segment
  let rangeB := segmentB_end.1 - segmentB_start.1 -- length of B's segment
  let effective_distance := sqrt 8.25 -- derived from |A_X - B_X| ≤ sqrt{8.25}
  let probability := effective_distance / rangeA -- probability calculation

  probability ≈ 0.96

theorem circles_intersection_probability : probability_intersection_of_circles :=
by
  sorry

end circles_intersection_probability_l717_717240


namespace smallest_multiple_of_6_and_15_l717_717754

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ c : ℕ, c > 0 ∧ c % 6 = 0 ∧ c % 15 = 0 → c ≥ b := 
begin
  use 30,
  split,
  { exact nat.succ_pos 29, },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 2 3) (dvd_mul_right 3 5)), },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 3 5) (dvd_mul_right 3 2)), },
  { intros c hc1 hc2,
    have hc3 : c % 30 = 0,
    {
      suffices h : c % 6 = 0 ∧ c % 15 = 0 ↔ c % lcm 6 15 = 0,
      { rw ← h, exact ⟨hc1, hc2⟩, },
      exact nat.dvd_iff_mod_eq_zero,
    },
    linarith,
  }
end

end smallest_multiple_of_6_and_15_l717_717754


namespace problem_solution_l717_717762

theorem problem_solution (n : ℤ) : 
  (1 / (n + 2) + 3 / (n + 2) + 2 * n / (n + 2) = 4) → (n = -2) :=
by
  intro h
  sorry

end problem_solution_l717_717762


namespace infinite_integers_exist_l717_717921

theorem infinite_integers_exist (n : ℕ) (h : n ≥ 1) :
  ∃ x y z : ℤ, digit_sum (4 * x^4 + y^4 - z^2 + 4 * x * y * z) ≤ 2 := by
  sorry

end infinite_integers_exist_l717_717921


namespace ratio_of_areas_l717_717224

variable (s : ℝ)
def side_length_square := s
def side_length_longer_rect := 1.2 * s
def side_length_shorter_rect := 0.7 * s
def area_square := s^2
def area_rect := (1.2 * s) * (0.7 * s)

theorem ratio_of_areas (h1 : s > 0) :
  area_rect s / area_square s = 21 / 25 :=
by 
  sorry

end ratio_of_areas_l717_717224


namespace minimal_period_divides_all_periods_l717_717155

variable (n : ℕ) (hn : n > 1)
variable (I : Set ℤ) (hI : ∀ a ∈ I, Nat.gcd a n = 1)
variable (f : ℤ → ℕ) (hf : ∀ a b : ℤ, a ∈ I → b ∈ I → n ∣ (a - b) → f a = f b)

/-- 
  If f is n-periodic and p is the minimal period of f, then p divides any other period k.
-/
theorem minimal_period_divides_all_periods {p : ℕ} {k : ℕ}
  (h_min_period : ∀ a b : ℤ, a ∈ I → b ∈ I → p ∣ (a - b) → f a = f b)
  (h_minimal : ∀ q : ℕ, (q < p → ∃ a b : ℤ, a ∈ I ∧ b ∈ I ∧ q ∣ (a - b) ∧ f a ≠ f b))
  (h_k_period : ∀ a b : ℤ, a ∈ I → b ∈ I → k ∣ (a - b) → f a = f b) : p ∣ k :=
sorry

end minimal_period_divides_all_periods_l717_717155


namespace avg_price_six_toys_l717_717351

def avg_price_five_toys : ℝ := 10
def price_sixth_toy : ℝ := 16
def total_toys : ℕ := 5 + 1

theorem avg_price_six_toys (avg_price_five_toys price_sixth_toy : ℝ) (total_toys : ℕ) :
  (avg_price_five_toys * 5 + price_sixth_toy) / total_toys = 11 := by
  sorry

end avg_price_six_toys_l717_717351


namespace function_properties_l717_717314

-- Define the function f(x) = x^{-1}
def f (x : ℝ) : ℝ := x⁻¹

-- The main statement to prove that f is an odd function and monotonically decreasing on (0, +∞).
theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by
  sorry

end function_properties_l717_717314


namespace positive_real_inequality_l717_717146

noncomputable def positive_real_sum_condition (u v w : ℝ) [OrderedRing ℝ] :=
  u + v + w + Real.sqrt (u * v * w) = 4

theorem positive_real_inequality (u v w : ℝ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  positive_real_sum_condition u v w →
  Real.sqrt (v * w / u) + Real.sqrt (u * w / v) + Real.sqrt (u * v / w) ≥ u + v + w :=
by
  sorry

end positive_real_inequality_l717_717146


namespace lucas_game_points_product_l717_717467

theorem lucas_game_points_product :
  let points_first_10 := [5, 2, 6, 3, 10, 1, 3, 3, 4, 2],
      total_first_10 := points_first_10.sum,
      eleventh_game_points := 5,
      twelfth_game_points := 4
  in (total_first_10 + eleventh_game_points) % 11 = 0 ∧
     (total_first_10 + eleventh_game_points + twelfth_game_points) % 12 = 0 ∧
     eleventh_game_points < 10 ∧
     twelfth_game_points < 10
  → eleventh_game_points * twelfth_game_points = 20 := 
by
  sorry

end lucas_game_points_product_l717_717467


namespace proof_problem_l717_717090

variable {R : Type*} [Field R] {x y z w N : R}

theorem proof_problem 
  (h1 : 4 * x * z + y * w = N)
  (h2 : x * w + y * z = 6)
  (h3 : (2 * x + y) * (2 * z + w) = 15) :
  N = 3 :=
by sorry

end proof_problem_l717_717090


namespace tan_alpha_cos2alpha_plus_2sin2alpha_l717_717442

theorem tan_alpha_cos2alpha_plus_2sin2alpha (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end tan_alpha_cos2alpha_plus_2sin2alpha_l717_717442


namespace pencils_in_each_box_l717_717085

theorem pencils_in_each_box (total_pencils : ℕ) (total_boxes : ℕ) (pencils_per_box : ℕ) 
  (h1 : total_pencils = 648) (h2 : total_boxes = 162) : 
  total_pencils / total_boxes = pencils_per_box := 
by
  sorry

end pencils_in_each_box_l717_717085


namespace parabola_vertex_y_l717_717628

theorem parabola_vertex_y (x : ℝ) : (∃ (h k : ℝ), (4 * (x - h)^2 + k = 4 * x^2 + 16 * x + 11) ∧ k = -5) := 
  sorry

end parabola_vertex_y_l717_717628


namespace part_A_I_part_A_II_part_A_III_part_B_I_part_B_II_part_B_III_l717_717308

-- Sequences definition
def a : ℕ → ℝ 
| 0 => 0
| n + 1 => a (n + 1) + (n + 1)

def b (n : ℕ) : ℝ := Real.log (2^n - 1) / Real.log 2 + 1

def R (n : ℕ) : ℝ := (Finset.range n).sum (λ i, (b i + 1) / (2^i))

def c (n : ℕ) : ℝ := 1 / (4 * (Real.log (2^n - 1) / Real.log 2 + 1) - 3)

def S (n : ℕ) : ℝ := (Finset.range n).sum (λ i, c i)

theorem part_A_I :
    ∀ n : ℕ, (a n + 1) / (a (0) + 1) = 2^n :=
by sorry

theorem part_A_II :
    ∀ n : ℕ, R n = 4 - (n + 2) / (2^(n - 1)) :=
by sorry

theorem part_A_III (n : ℕ) :
    R n ≥ (m : ℝ) - 9 / (2 + 2 * a n) → m ≤ 61 / 16 :=
by sorry

theorem part_B_I : 
    ∀ n : ℕ, (a n + 1) / (a (0) + 1) = 2^n :=
by sorry

theorem part_B_II :
    ∀ n : ℕ, R n = 4 - (n + 2) / (2^(n - 1)) :=
by sorry

theorem part_B_III (n : ℕ):
    S (2 * n + 1) - S n ≤ (m : ℝ) / 30 → m ≥ 38 / 3 :=
by sorry

end part_A_I_part_A_II_part_A_III_part_B_I_part_B_II_part_B_III_l717_717308


namespace proof_problem_l717_717074

-- Define the propositions and conditions
def p : Prop := ∀ x > 0, 3^x > 1
def neg_p : Prop := ∃ x > 0, 3^x ≤ 1
def q (a : ℝ) : Prop := a < -2
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

-- The condition that q is a sufficient condition for f(x) to have a zero in [-1,2]
def has_zero_in_interval (a : ℝ) : Prop := 
  (-a + 3) * (2 * a + 3) ≤ 0

-- The proof problem statement
theorem proof_problem (a : ℝ) (P : p) (Q : has_zero_in_interval a) : ¬ p ∧ q a :=
by
  sorry

end proof_problem_l717_717074


namespace day_3280_students_know_secret_l717_717518

def students_knowing_secret (n : ℕ) : ℕ :=
  (3^(n + 1) - 1) / 2

theorem day_3280_students_know_secret :
  ∃ n : ℕ, students_knowing_secret n = 3280 ∧ (n = 7) :=
by
  have h_n : 3^8 = 6561 := by norm_num
  use 7
  simp [students_knowing_secret, h_n]
  rfl

end day_3280_students_know_secret_l717_717518


namespace embankment_construction_l717_717869

theorem embankment_construction :
  (∃ r : ℚ, 0 < r ∧ (1 / 2 = 60 * r * 3)) →
  (∃ t : ℕ, 1 = 45 * 1 / 360 * t) :=
by
  sorry

end embankment_construction_l717_717869


namespace range_of_f_on_interval_l717_717454

open Function

def f (x : ℝ) : ℝ := x^2 + 4*x + 6

theorem range_of_f_on_interval : 
  (x : ℝ) → x ∈ Icc (-3 : ℝ) 0 → x ≠ 0 → 2 ≤ f x ∧ f x < 6 :=
by
  sorry

end range_of_f_on_interval_l717_717454


namespace unique_function_f_l717_717289

noncomputable def f : ℝ → ℝ := λ x => sin (2 * x - π / 6)

theorem unique_function_f : 
  (∀ x : ℝ, f (x + π) = f x) ∧ 
  (f (2 * (x - π / 3)) = f (2 * (π / 3 - x))) ∧ 
  (∀ x : ℝ, -π / 6 ≤ x ∧ x ≤ π / 3 → f x ≤ f (x + ε)) := 
  f = λ x => sin (2 * x - π / 6) :=
begin
  sorry
end

end unique_function_f_l717_717289


namespace intercept_sum_l717_717291

theorem intercept_sum {x y : ℝ} 
  (h : y - 3 = -3 * (x - 5)) 
  (hx : x = 6) 
  (hy : y = 18) 
  (intercept_sum_eq : x + y = 24) : 
  x + y = 24 :=
by
  sorry

end intercept_sum_l717_717291


namespace number_divisible_by_19_l717_717920

theorem number_divisible_by_19 (n : ℕ) : (12000 + 3 * 10^n + 8) % 19 = 0 := 
by sorry

end number_divisible_by_19_l717_717920


namespace smallest_nonprime_no_prime_factor_lt_15_l717_717507

def is_nonprime (n : ℕ) : Prop := ¬prime n ∧ n > 1

theorem smallest_nonprime_no_prime_factor_lt_15 (n : ℕ) (h : is_nonprime n)
  (h_prime_factor : ∀ p : ℕ, prime p → p ∣ n → p ≥ 15) :
  280 < n ∧ n ≤ 290 :=
sorry

end smallest_nonprime_no_prime_factor_lt_15_l717_717507


namespace circles_area_l717_717205

theorem circles_area (BD AC : ℝ) (r : ℝ) (h1 : BD = 6) (h2 : AC = 12)
  (h3 : ∀ (d1 d2 : ℝ), d1 = AC / 2 → d2 = BD / 2 → r^2 = (r - d2)^2 + d1^2) :
  real.pi * r^2 = (225/4) * real.pi :=
by
  -- proof to be filled
  sorry

end circles_area_l717_717205


namespace exists_sphere_touches_all_edges_l717_717159

variables {P : Type} [convex_polyhedron P]
variables {S : Type} [sphere S]

noncomputable def sphere_touches_all_edges (P : convex_polyhedron) (S : sphere) : Prop :=
  ∀ (e : edge P), ∃ (x : S), x ∈ e ∧ divides_into_three_equal_parts x e

theorem exists_sphere_touches_all_edges 
  (P : convex_polyhedron) 
  (S : sphere) 
  (h : ∀ (e : edge P), ∃ (x1 x2 : S), x1 ∈ e ∧ x2 ∈ e ∧ divides_into_three_equal_parts x1 e ∧ divides_into_three_equal_parts x2 e)
  : ∃ T : sphere, ∀ e : edge P, ∃ t : T, t ∈ e :=
sorry

end exists_sphere_touches_all_edges_l717_717159


namespace sum_of_complex_numbers_l717_717800

def complex_sum (z1 z2 : ℂ) : ℂ :=
  z1 + z2

theorem sum_of_complex_numbers : 
  complex_sum (1 + 7 * complex.I) (-2 - 4 * complex.I) = -1 + 3 * complex.I :=
by
  sorry

end sum_of_complex_numbers_l717_717800


namespace stem_and_leaf_does_not_lose_info_l717_717610

-- Definitions of the concepts of statistical charts
def loses_info (chart: String) : Prop :=
  chart = "Bar chart" ∨ chart = "Pie chart" ∨ chart = "Line chart"

def does_not_lose_info (chart: String) : Prop :=
  ¬ loses_info(chart)

-- Statement we want to prove
theorem stem_and_leaf_does_not_lose_info : does_not_lose_info "Stem-and-leaf plot" :=
by
  -- here we would provide the proof in practice
  sorry

end stem_and_leaf_does_not_lose_info_l717_717610


namespace expected_seeds_replanted_l717_717642

open ProbabilityTheory

theorem expected_seeds_replanted (p : ℝ) (n : ℕ) :
  p = 0.9 ∧ n = 1000 →
  E[X] = 200 :=
by
  sorry

end expected_seeds_replanted_l717_717642


namespace tangent_line_ln_curve_l717_717058

theorem tangent_line_ln_curve (a : ℝ) :
  (∃ x y : ℝ, y = Real.log x + a ∧ x - y + 1 = 0 ∧ (∀ t : ℝ, t = x → (t - (Real.log t + a)) = -(1 - a))) → a = 2 :=
by
  sorry

end tangent_line_ln_curve_l717_717058


namespace hyperbola_problem_l717_717807

def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def eccentricity (a c : ℝ) : Prop :=
  c / a = 2 * Real.sqrt 3 / 3

def focal_distance (c a : ℝ) : Prop :=
  2 * a^2 = 3 * c

def point_on_hyperbola (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  hyperbola a b P.1 P.2

def point_satisfies_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = 2

noncomputable def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

theorem hyperbola_problem (a b c : ℝ) (P F1 F2 : ℝ × ℝ) :
  (a > 0 ∧ b > 0) →
  eccentricity a c →
  focal_distance c a →
  point_on_hyperbola P a b →
  point_satisfies_condition P F1 F2 →
  distance F1 F2 = 2 * c →
  (distance P F1) * (distance P F2) = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end hyperbola_problem_l717_717807


namespace max_combined_storage_l717_717966

noncomputable def f (t : ℝ) : ℝ := 2 + Real.sin t
noncomputable def g (t : ℝ) : ℝ := 5 - abs (t - 6)
noncomputable def H (t : ℝ) : ℝ := f t + g t

theorem max_combined_storage :
  ∃ t ∈ Icc (0 : ℝ) 12, H t = 6.721 ∧ (∀ s ∈ Icc (0 : ℝ) 12, H s ≤ H t) := 
begin
  use 6,
  split,
  { exact ⟨le_refl 6, by norm_num⟩ },
  split,
  { have h_sin_6 : Real.sin 6 ≈ -0.279 := by norm_cast,
    have h_H_6: H 6 = 7 + Real.sin 6,
    { unfold H,
      unfold f,
      unfold g,
      ring,
      rw abs_of_nonneg,
        { norm_num,
          rw h_sin_6 }, 
        { norm_num } },
    rw h_H_6,
    rw h_sin_6,
    norm_num,
    exact rfl,},
  { intros s hs,
    cases le_total s 6 with h_le h_ge,
    { have H_nondec : MonotoneOn (λ t, f t + g t) (Icc 0 6) := sorry,
      apply H_nondec,
      exact hs.left,
      exact h_le },
    { have H_noninc : AntitoneOn (λ t, f t + g t) (Ioc 6 12) := sorry,
      apply H_noninc,
      exact Ico_subset_Icc_self hs.left,
      exact h_ge } }
end

end max_combined_storage_l717_717966


namespace number_of_true_propositions_is_3_l717_717421

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (finset.range n).sum a

def proposition_1 (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  arithmetic_sequence a ∧ sum_first_n_terms a S ∧
  arithmetic_sequence (λ n, S (2 * n) - S n) ∧
  arithmetic_sequence (λ n, S (3 * n) - S (2 * n))

def proposition_2 (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  geometric_sequence a ∧ sum_first_n_terms (λ n, (a n : ℤ)) (λ n, (S n : ℤ)) ∧
  geometric_sequence (λ n, S (2 * n) - S n) ∧
  geometric_sequence (λ n, S (3 * n) - S (2 * n))

def proposition_3 (a b : ℕ → ℤ) : Prop :=
  arithmetic_sequence a ∧ arithmetic_sequence b ∧
  arithmetic_sequence (λ n, a n + b n)

def proposition_4 (a b : ℕ → ℝ) : Prop :=
  geometric_sequence a ∧ geometric_sequence b ∧
  geometric_sequence (λ n, a n * b n)

theorem number_of_true_propositions_is_3 :
  ∃ a : ℕ → ℤ, ∃ b : ℕ → ℤ, ∃ c : ℕ → ℤ, ∃ d : ℕ → ℤ, ∃ S₁ S₂: ℕ → ℤ, ∃ S₃ : ℕ → ℝ,
  proposition_1 a S₁ ∧ ¬proposition_2 c S₃ ∧ proposition_3 a b ∧ proposition_4 d b ∧ 
  (3 = nat.pred (nat.succ (nat.succ nat.zero))) :=
sorry

end number_of_true_propositions_is_3_l717_717421


namespace border_material_length_l717_717643

noncomputable def area (r : ℝ) : ℝ := (22 / 7) * r^2

theorem border_material_length (r : ℝ) (C : ℝ) (border : ℝ) : 
  area r = 616 →
  C = 2 * (22 / 7) * r →
  border = C + 3 →
  border = 91 :=
by
  intro h_area h_circumference h_border
  sorry

end border_material_length_l717_717643


namespace locus_of_point_x_l717_717165

theorem locus_of_point_x (P Q X A B C D : Point) (circumcircle : circle A B C D) 
  (hP : P ∈ circumcircle)
  (O : Point) (hO : O = midpoint A C ∧ O = midpoint B D)
  (hPQ : intersect (line_through A P) (line_through B D) = some Q)
  (hline_Q : parallel (line_through Q X) (line_through A C))
  (hX : intersect (line_through B P) (line_through Q X) = some X)
  : X ∈ (line_through C D) :=
sorry

end locus_of_point_x_l717_717165


namespace smallest_multiple_l717_717746

theorem smallest_multiple (b : ℕ) (h1 : b % 6 = 0) (h2 : b % 15 = 0) (h3 : ∀ n : ℕ, (n % 6 = 0 ∧ n % 15 = 0) → n ≥ b) : b = 30 :=
sorry

end smallest_multiple_l717_717746


namespace part1_part2_l717_717812

-- Definitions of sets A and B
def A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | real.sqrt (x - 1) ≥ 1}

-- Statement of the first part of the problem
theorem part1 : {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
sorry

-- Statement of the second part of the problem
theorem part2 (a : ℝ) : {x : ℝ | 2 ≤ x ∧ x ≤ 3} ⊆ {x : ℝ | x ≥ a} → a ∈ set.Iic 2 :=
sorry

end part1_part2_l717_717812


namespace james_can_make_sushi_rolls_l717_717873

def fish_per_sushi_roll : Nat := 40
def total_fish_bought : Nat := 400
def percentage_bad_fish : Real := 0.20

theorem james_can_make_sushi_rolls : 
  (total_fish_bought - Nat.floor((percentage_bad_fish * total_fish_bought : Real))) / fish_per_sushi_roll = 8 := 
by
  sorry

end james_can_make_sushi_rolls_l717_717873


namespace num_three_digit_integers_divisible_by_12_l717_717439

theorem num_three_digit_integers_divisible_by_12 : 
  (∃ (count : ℕ), count = 3 ∧ 
    (∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → 
      (∀ d : ℕ, d ∈ [n / 100, (n / 10) % 10, n % 10] → 4 < d) ∧ 
      n % 12 = 0 → 
      count = count + 1)) := 
sorry

end num_three_digit_integers_divisible_by_12_l717_717439


namespace area_of_triangle_ABC_equation_of_circumcircle_l717_717473

-- Define points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 2 }
def B : Point := { x := 1, y := 3 }
def C : Point := { x := 3, y := 6 }

-- Theorem to prove the area of triangle ABC
theorem area_of_triangle_ABC : 
  let base := |B.y - A.y|
  let height := |C.x - A.x|
  (1/2) * base * height = 1 := sorry

-- Theorem to prove the equation of the circumcircle of triangle ABC
theorem equation_of_circumcircle : 
  let D := -10
  let E := -5
  let F := 15
  ∀ (x y : ℝ), (x - 5)^2 + (y - 5/2)^2 = 65/4 ↔ 
                x^2 + y^2 + D * x + E * y + F = 0 := sorry

end area_of_triangle_ABC_equation_of_circumcircle_l717_717473


namespace find_varphi_eq_five_pi_div_six_l717_717561

theorem find_varphi_eq_five_pi_div_six :
  ∀ (φ : ℝ), -π ≤ φ ∧ φ ≤ π →
  (∀ x : ℝ, (cos (2 * x + φ)) = sin (2 * (x - π/4) + π/3)) →
  φ = 5 * π / 6 :=
by
  sorry

end find_varphi_eq_five_pi_div_six_l717_717561


namespace f_eval_half_l717_717501

variable {X : Type*}
variable {f : X → ℝ}
variable {g : ℝ → ℝ}

-- Conditions
def g_def : g x = 1 - 2 * x := sorry
def f_def : ∀ (x : ℝ), x ≠ 0 → f (g x) = (1 - x^2) / 2 := sorry

-- Proof statement
theorem f_eval_half : f (1 / 2) = 15 / 32 :=
by 
  have g_def : g x = 1 - 2 * x := sorry
  have f_def : ∀ (x : ℝ), x ≠ 0 → f (g x) = (1 - x^2) / 2 := sorry
  sorry

end f_eval_half_l717_717501


namespace length_segment_AB_l717_717071

theorem length_segment_AB 
  (C : set (ℝ × ℝ))
  (h1 : ∀ x y : ℝ, (y^2 = 8 * x) ↔ ((x, y) ∈ C))
  (F : ℝ × ℝ)
  (h2 : F = (2, 0))
  (A B : ℝ × ℝ)
  (h3 : ∀ d : ℝ, (A.1 = 4) ∧ (abs (A.1 + 2) = 6))
  (h4 : ∀ l : set (ℝ × ℝ), l = {p | ∃ k : ℝ, ∃ m : ℝ, p = (m, (k * (m - 2) + 4 * √2))} ∧ ∀ p ∈ l, (p ∈ C))
  (h5 : A ∈ C ∧ B ∈ C)
  (h6 : ∃ x y : ℝ, B = (1, y) ∧ ((A.1 = 4) → (abs (A.1 - B.1) = 3))) :
  abs (A.1 - B.1) + 6 = 9 :=
sorry

end length_segment_AB_l717_717071


namespace revenue_for_recent_quarter_l717_717670

noncomputable def previous_year_revenue : ℝ := 85.0
noncomputable def percentage_fall : ℝ := 43.529411764705884
noncomputable def recent_quarter_revenue : ℝ := previous_year_revenue - (previous_year_revenue * (percentage_fall / 100))

theorem revenue_for_recent_quarter : recent_quarter_revenue = 48.0 := 
by 
  sorry -- Proof is skipped

end revenue_for_recent_quarter_l717_717670


namespace find_k_l717_717258

theorem find_k : ∃ k : ℚ, (k = (k + 4) / 4) ∧ k = 4 / 3 :=
by
  sorry

end find_k_l717_717258


namespace magnitude_of_angle_C_range_of_m_l717_717838

-- Problem 1: Proving the magnitude of angle C
theorem magnitude_of_angle_C 
(a b c : ℝ) (h : a^2 + b^2 - c^2 = real.sqrt 3 * a * b) : 
  ∃ C : ℝ, 0 < C ∧ C < real.pi ∧ real.cos C = real.sqrt 3 / 2 :=
by
sory

-- Problem 2: Proving the range of m
theorem range_of_m 
(A : ℝ) (hA : 0 < A ∧ A ≤ 2 * real.pi / 3) : 
  let m := 2 * real.cos (A / 2) ^ 2 - real.sin (real.pi - A) - 1 in
  C = real.pi / 6 ∧ -1 ≤ m ∧ m < 1 / 2 :=
by
sory

end magnitude_of_angle_C_range_of_m_l717_717838


namespace gcd_360_504_is_72_l717_717220

theorem gcd_360_504_is_72 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_is_72_l717_717220


namespace total_dots_l717_717319

def ladybugs_monday : ℕ := 8
def ladybugs_tuesday : ℕ := 5
def dots_per_ladybug : ℕ := 6

theorem total_dots :
  (ladybugs_monday + ladybugs_tuesday) * dots_per_ladybug = 78 :=
by
  sorry

end total_dots_l717_717319


namespace parabola_equation_maximum_area_of_triangle_l717_717472

-- Definitions of the conditions
def parabola_eq (x y : ℝ) (p : ℝ) : Prop := x^2 = 2 * p * y ∧ p > 0
def distances_equal (AO AF : ℝ) : Prop := AO = 3 / 2 ∧ AF = 3 / 2
def line_eq (x k b y : ℝ) : Prop := y = k * x + b
def midpoint_y (y1 y2 : ℝ) : Prop := (y1 + y2) / 2 = 1

-- Part (I)
theorem parabola_equation (p : ℝ) (x y AO AF : ℝ) (h1 : parabola_eq x y p)
  (h2 : distances_equal AO AF) :
  x^2 = 4 * y :=
sorry

-- Part (II)
theorem maximum_area_of_triangle (p k b AO AF x1 y1 x2 y2 : ℝ)
  (h1 : parabola_eq x1 y1 p) (h2 : parabola_eq x2 y2 p)
  (h3 : distances_equal AO AF) (h4 : line_eq x1 k b y1) 
  (h5 : line_eq x2 k b y2) (h6 : midpoint_y y1 y2)
  : ∃ (area : ℝ), area = 2 :=
sorry

end parabola_equation_maximum_area_of_triangle_l717_717472


namespace discriminant_of_quadratic_l717_717943

theorem discriminant_of_quadratic : 
  let a := 1
  let b := -7
  let c := 4
  Δ = b ^ 2 - 4 * a * c
  (b ^ 2 - 4 * a * c) = 33 :=
by
  -- definitions of a, b, and c
  let a := 1
  let b := -7
  let c := 4
  -- definition of Δ
  let Δ := b ^ 2 - 4 * a * c
  -- given the quadratic equation x^2 - 7x + 4 = 0, prove that Δ = 33
  show (b ^ 2 - 4 * a * c) = 33,
  -- proof is omitted
  sorry

end discriminant_of_quadratic_l717_717943


namespace smallest_integer_with_18_divisors_l717_717988

theorem smallest_integer_with_18_divisors :
  ∃ n : ℕ, (∀ m : ℕ, (m > 0 ∧ (number_of_divisors m) = 18) → n ≤ m) ∧ 
            n = 78732 :=
sorry

end smallest_integer_with_18_divisors_l717_717988


namespace NapoleonTheorem_l717_717270

def Triangle (α : Type*) := {p : α × α × α // p.1 ≠ p.2 ∧ p.1 ≠ p.3 ∧ p.2 ≠ p.3}

structure CenterEquilateral (α : Type*) :=
(A' B' C' : α)
(is_equilateral_A'_BC : ∀ (A B C : α), Triangle α → Triangle α → Triangle α → Triangle α → Triangle α)
(is_equilateral_AB'_C : ∀ (A B C : α), Triangle α → Triangle α → Triangle α → Triangle α → Triangle α)
(is_equilateral_ABC' : ∀ (A B C : α), Triangle α → Triangle α → Triangle α → Triangle α → Triangle α)

structure EquilateralCenters (α : Type*) := 
  (A_1 B_1 C_1 : α)
  (center_of_equilateral_A1 : CenterEquilateral α)
  (center_of_equilateral_B1 : CenterEquilateral α)
  (center_of_equilateral_C1 : CenterEquilateral α)
  
def triangulate (α : Type*) (A B C : α) :=
  ∃ (A' B' C' : α), ∃ (e_centers : EquilateralCenters α), 
    (e_centers.center_of_equilateral_A1 A B C) ∧ 
    (e_centers.center_of_equilateral_B1 A B C) ∧ 
    (e_centers.center_of_equilateral_C1 A B C)

theorem NapoleonTheorem {α : Type*} [metric_space α] (triangle : Triangle α)
  (center_eq : EquilateralCenters α) 
  (A B C : α) (A_1 B_1 C_1 : α) :
  (EquilateralCenters α) → 
  (triangulate α A B C) → 
  is_equilateral α A_1 B_1 C_1 := 
sorry

end NapoleonTheorem_l717_717270


namespace cody_books_reading_l717_717694

theorem cody_books_reading :
  ∀ (total_books first_week_books second_week_books subsequent_week_books : ℕ),
    total_books = 54 →
    first_week_books = 6 →
    second_week_books = 3 →
    subsequent_week_books = 9 →
    (2 + (total_books - (first_week_books + second_week_books)) / subsequent_week_books) = 7 :=
by
  -- Using sorry to mark the proof as incomplete.
  sorry

end cody_books_reading_l717_717694


namespace first_term_geometric_sequence_l717_717559

theorem first_term_geometric_sequence (a5 a6 : ℚ) (h1 : a5 = 48) (h2 : a6 = 64) : 
  ∃ a : ℚ, a = 243 / 16 :=
by
  sorry

end first_term_geometric_sequence_l717_717559


namespace infinite_cube_volume_sum_l717_717552

noncomputable def sum_of_volumes_of_infinite_cubes (a : ℝ) : ℝ :=
  ∑' n, (((a / (3 ^ n))^3))

theorem infinite_cube_volume_sum (a : ℝ) : sum_of_volumes_of_infinite_cubes a = (27 / 26) * a^3 :=
sorry

end infinite_cube_volume_sum_l717_717552


namespace max_edges_triangle_quad_free_graph_l717_717850

open_locale classical

-- Definition of a Graph structure
structure Graph (V : Type*) :=
  (E : set (V × V))
  (symm : ∀ {x y : V}, (x, y) ∈ E → (y, x) ∈ E)
  (irref : ∀ {x : V}, ¬(x, x) ∈ E)

-- Definition of a triangle-free property in a Graph
def is_triangle_free {V : Type*} (G : Graph V) : Prop :=
  ∀ {a b c : V}, (a, b) ∈ G.E → (b, c) ∈ G.E → (c, a) ∈ G.E → false

-- Definition of a quadrilateral-free property in a Graph
def is_quadrilateral_free {V : Type*} (G : Graph V) : Prop :=
  ∀ {a b c d : V}, (a, b) ∈ G.E → (b, c) ∈ G.E → (c, d) ∈ G.E → (d, a) ∈ G.E → false

-- Theorem statement for maximum edges in an 11-vertex triangle-free, quadrilateral-free graph
theorem max_edges_triangle_quad_free_graph (G : Graph (fin 11)) :
  is_triangle_free G → is_quadrilateral_free G → set.finite G.E ∧ G.E.card ≤ 16 :=
by
  sorry

end max_edges_triangle_quad_free_graph_l717_717850


namespace points_four_units_away_l717_717521

theorem points_four_units_away (x : ℤ) : (x - (-1) = 4 ∨ x - (-1) = -4) ↔ (x = 3 ∨ x = -5) :=
by
  sorry

end points_four_units_away_l717_717521


namespace ratio_of_members_l717_717681

theorem ratio_of_members (f m c : ℕ) 
  (h1 : (35 * f + 30 * m + 10 * c) / (f + m + c) = 25) :
  2 * f + m = 3 * c :=
by
  sorry

end ratio_of_members_l717_717681


namespace monotonic_function_range_l717_717455

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then x * |x + a| - 5 else a / x

theorem monotonic_function_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ∨ (∀ x y : ℝ, x ≤ y → f a x ≥ f a y) →
  a ∈ set.Icc (-3:ℝ) (-2:ℝ) :=
sorry

end monotonic_function_range_l717_717455


namespace solve_f_inequality_l717_717802

def f (x : ℝ) : ℝ :=
  if x < (1 : ℝ) / (2 : ℝ) then (1 : ℝ) / (2 : ℝ) * x + 1
  else 3 * x^2 + x

theorem solve_f_inequality : ∀ x : ℝ, 0 < x → x < (2 : ℝ) / (3 : ℝ) → f(x) < 2 :=
by
  intros x hx1 hx2
  -- This is where the proof would go
  sorry

end solve_f_inequality_l717_717802


namespace find_n_l717_717761

theorem find_n : ∀ n : ℚ, (1 / (n + 2) + 2 / (n + 2) + 3 * n / (n + 2) = 5) → (n = -7 / 2) := by
  intro n h
  sorry

end find_n_l717_717761


namespace simplify_factorial_expression_l717_717178

theorem simplify_factorial_expression : 
  (13.factorial / (10.factorial + 3 * 9.factorial) = 1320) := 
by 
  sorry

end simplify_factorial_expression_l717_717178


namespace birds_and_storks_l717_717272

theorem birds_and_storks :
  let initial_birds := 3 in
  let initial_storks := 4 in
  let birds_after_arrival := initial_birds + 2 in
  let birds_final := birds_after_arrival - 1 in
  let storks_final := initial_storks + 3 in
  storks_final - birds_final = 3 :=
by
  sorry

end birds_and_storks_l717_717272


namespace leak_empty_time_l717_717290

section
variable (leak_rate : ℚ) (inlet_rate : ℚ := 4) (tank_capacity : ℚ := 5760) (empty_time_with_inlet : ℚ := 8)

/--
Given:
1. The inlet pipe fills the tank at the rate of 4 liters per minute.
2. When the tank is full, the inlet is opened, and the tank empties in 8 hours due to a leak at the bottom.
3. The capacity of the tank is 5760 liters.

Prove that the leak can empty the tank in 6 hours when the inlet is closed.
-/
theorem leak_empty_time :
  empty_time_with_inlet * 60 * (leak_rate - inlet_rate) - tank_capacity = 0 →
  leak_rate = 16 →
  tank_capacity / leak_rate = 360 →
  360 / 60 = 6 :=
by
  assume h1 h2 h3
  rw [mul_assoc, mul_comm] at h1
  sorry
end

end leak_empty_time_l717_717290


namespace value_subtracted_l717_717101

theorem value_subtracted (x y : ℤ) (h1 : (x - 5) / 7 = 7) (h2 : (x - y) / 13 = 4) : y = 2 :=
sorry

end value_subtracted_l717_717101


namespace part1_part2_l717_717477

-- Definitions and assumptions based on given conditions
variables {A B C : ℝ}

-- Given condition 1
def condition1 := sqrt 2 * sin A = 2 * (cos (A / 2))^2

-- Given condition 2 for the second question
def condition2 := 2 * A - B = π / 2

-- Statement to prove for part (1)
theorem part1 : condition1 → tan A = 2 * sqrt 2 := sorry

-- Statement to prove for part (2)
theorem part2 : condition1 → condition2 → sin (A + C) = 7 / 9 := sorry

end part1_part2_l717_717477


namespace radius_of_cylinder_base_l717_717677

theorem radius_of_cylinder_base (P A B C D : Type) [RegularPyramid P A B C D]
    (h₁ : VerticesOnCylinderLateralSurface P A B C D)
    (h₂ : CylinderAxisPerpendicular P A B)
    (h₃ : AB_length_eq_a P A B)
    : Radius_of_Cylinder_Base = a / (Real.sqrt 3) :=
sorry

end radius_of_cylinder_base_l717_717677


namespace blake_initial_amount_l717_717327

noncomputable def initial_amount_given (amount: ℕ) := 3 * amount / 2

theorem blake_initial_amount (h_given_amount_b_to_c: ℕ) (c_to_b_transfer: ℕ) (h_c_to_b: c_to_b_transfer = 30000) :
  initial_amount_given c_to_b_transfer = h_given_amount_b_to_c → h_given_amount_b_to_c = 20000 :=
by
  intro h
  have calc_initial := calc 
    initial_amount_given 30000
      = 3 * 30000 / 2 : by rfl
      ... = 90000 / 2 : by norm_num
      ... = 45000 : by norm_num
  contradiction

end blake_initial_amount_l717_717327


namespace diameter_large_circle_tangent_l717_717353

noncomputable def diameter_of_large_circle (r : ℝ) (n : ℕ) : ℝ :=
  if n = 8 then 2 * (r * real.sqrt 2 + r) else 0

theorem diameter_large_circle_tangent (r : ℝ) (d : ℝ) :
  r = 4 → d = 8 * real.sqrt 2 + 8 → diameter_of_large_circle r 8 = d :=
by
  intros
  rw diameter_of_large_circle
  simp [*]
  sorry

end diameter_large_circle_tangent_l717_717353


namespace quadrilateral_centroid_area_is_correct_l717_717543

noncomputable def quadrilateral_centroid_area (EFGH : Type) (Q : EFGH) [MetricSpace EFGH] :=
  side : ℝ,
  point_eq : ℝ → Prop,
  point_fq : ℝ → Prop

theorem quadrilateral_centroid_area_is_correct:
  (side = 40) → 
  (point_eq 16) →
  (point_fq 34) →
  quadrilateral_centroid_area EFGH Q = 800 / 9 :=
by
  sorry

end quadrilateral_centroid_area_is_correct_l717_717543


namespace exists_dense_sequence_l717_717170

namespace SequenceDensity

-- Define the sequence a(n)
def a (n : ℕ) : ℝ :=
  if n = 1 ∨ n = 2 then 0 else n * log(log(n : ℝ))

-- Define the condition a(n + m) ≤ a(n) + a(m) + (n + m)/log(n + m)
def sequence_condition (a : ℕ → ℝ) (n m : ℕ) : Prop :=
  a (n + m) ≤ a n + a m + (n + m : ℝ) / log (n + m)

-- Define the density condition for the set {a(n) / n : n ≥ 1}
def density_condition (a : ℕ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ ε > 0, ∀ y : ℝ, abs (x - y) < ε → ∃ n : ℕ, abs (y - a n / n) < ε

-- Main theorem statement
theorem exists_dense_sequence :
  ∃ a : ℕ → ℝ, 
    (∀ n m : ℕ, 1 ≤ n ∧ 1 ≤ m → sequence_condition a n m) ∧
    density_condition a :=
sorry

end SequenceDensity

end exists_dense_sequence_l717_717170


namespace angle_C_eq_pi_over_3_l717_717459

theorem angle_C_eq_pi_over_3 (a b c A B C : ℝ)
  (h : (a + c) * (Real.sin A - Real.sin C) = b * (Real.sin A - Real.sin B)) :
  C = Real.pi / 3 :=
sorry

end angle_C_eq_pi_over_3_l717_717459


namespace quadratic_intersects_x_axis_iff_l717_717456

theorem quadratic_intersects_x_axis_iff (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x - m = 0) ↔ m ≥ -1 := 
by
  sorry

end quadratic_intersects_x_axis_iff_l717_717456


namespace student_arrangement_count_l717_717630

theorem student_arrangement_count :
  ∃(students : Finset (Fin 6)), 
    ∃(A B C : Finset (Fin 6)), 
      A.card = 1 ∧ B.card = 2 ∧ C.card = 3 ∧
      (A ∪ B ∪ C = students) ∧
      (A ∩ B = ∅) ∧ (A ∩ C = ∅) ∧ (B ∩ C = ∅) ∧
      (∃ (arrangements : ℕ), arrangements = (Finset.choose 6 1) * (Finset.choose 5 2) ∧ arrangements = 60) :=
by
  sorry

end student_arrangement_count_l717_717630


namespace experienced_sailors_monthly_earnings_l717_717663

theorem experienced_sailors_monthly_earnings :
  let total_sailors : Nat := 17
  let inexperienced_sailors : Nat := 5
  let hourly_wage_inexperienced : Nat := 10
  let workweek_hours : Nat := 60
  let weeks_in_month : Nat := 4
  let experienced_sailors : Nat := total_sailors - inexperienced_sailors
  let hourly_wage_experienced := hourly_wage_inexperienced + (hourly_wage_inexperienced / 5)
  let weekly_earnings_experienced := hourly_wage_experienced * workweek_hours
  let total_weekly_earnings_experienced := weekly_earnings_experienced * experienced_sailors
  let monthly_earnings_experienced := total_weekly_earnings_experienced * weeks_in_month
  monthly_earnings_experienced = 34560 := by
  sorry

end experienced_sailors_monthly_earnings_l717_717663


namespace smallest_multiple_l717_717745

theorem smallest_multiple (b : ℕ) (h1 : b % 6 = 0) (h2 : b % 15 = 0) (h3 : ∀ n : ℕ, (n % 6 = 0 ∧ n % 15 = 0) → n ≥ b) : b = 30 :=
sorry

end smallest_multiple_l717_717745


namespace james_remaining_balance_l717_717486

theorem james_remaining_balance 
  (initial_balance : ℕ := 500) 
  (ticket_1_2_cost : ℕ := 150)
  (ticket_3_cost : ℕ := ticket_1_2_cost / 3)
  (total_cost : ℕ := 2 * ticket_1_2_cost + ticket_3_cost)
  (roommate_share : ℕ := total_cost / 2) :
  initial_balance - roommate_share = 325 := 
by 
  -- By not considering the solution steps, we skip to the proof.
  sorry

end james_remaining_balance_l717_717486


namespace find_principal_l717_717999

noncomputable def principal (P R T : ℝ) : ℝ := P
noncomputable def SI (P R T : ℝ) : ℝ := (P * R * T) / 100
noncomputable def CI (P R T : ℝ) : ℝ := P * ((1 + R / 100) ^ T) - P

theorem find_principal :
  ∃ P : ℝ, let R := 10 in
  let T := 2 in
  let SI := SI P R T in
  let CI := CI P R T in
  CI - SI = 18 → P = 1800 :=
begin
  sorry,
end

end find_principal_l717_717999


namespace minimal_polynomial_l717_717007

theorem minimal_polynomial :
  ∃ (P : Polynomial ℚ), P.monic ∧
    P.eval (1 + Real.sqrt 2) = 0 ∧
    P.eval (1 + Real.sqrt 5) = 0 ∧
    P.degree = 4 ∧
    P = Polynomial.Coeff (Polynomial.X ^ 4 - 4 * Polynomial.X ^ 3 + 6 * Polynomial.X ^ 2 + 4 * Polynomial.X + 4) :=
by {
  sorry
}

end minimal_polynomial_l717_717007


namespace number_of_bookshelves_l717_717686

-- Definitions based on the conditions
def books_per_shelf : ℕ := 2
def total_books : ℕ := 38

-- Statement to prove
theorem number_of_bookshelves (books_per_shelf total_books : ℕ) : total_books / books_per_shelf = 19 :=
by sorry

end number_of_bookshelves_l717_717686


namespace mn_minus_7_is_negative_one_l717_717824

def opp (x : Int) : Int := -x
def largest_negative_integer : Int := -1
def m := opp (-6)
def n := opp largest_negative_integer

theorem mn_minus_7_is_negative_one : m * n - 7 = -1 := by
  sorry

end mn_minus_7_is_negative_one_l717_717824


namespace solve_for_x_l717_717927

theorem solve_for_x (x : ℝ) (h : 3 / 4 + 1 / x = 7 / 8) : x = 8 :=
by
  sorry

end solve_for_x_l717_717927


namespace total_property_value_l717_717570

-- Define the given conditions
def price_per_sq_ft_condo := 98
def price_per_sq_ft_barn := 84
def price_per_sq_ft_detached := 102
def price_per_sq_ft_garage := 60
def sq_ft_condo := 2400
def sq_ft_barn := 1200
def sq_ft_detached := 3500
def sq_ft_garage := 480

-- Main statement to prove the total value of the property
theorem total_property_value :
  (price_per_sq_ft_condo * sq_ft_condo + 
   price_per_sq_ft_barn * sq_ft_barn + 
   price_per_sq_ft_detached * sq_ft_detached + 
   price_per_sq_ft_garage * sq_ft_garage = 721800) :=
by
  -- Placeholder for the actual proof
  sorry

end total_property_value_l717_717570


namespace acute_angle_between_base_and_median_is_45_l717_717200

theorem acute_angle_between_base_and_median_is_45
  (A B C D : Type) 
  (AC BD : ℝ)
  (angle_BAC : ℝ)
  (condition_AC : AC = 4)
  (condition_BD : BD = real.sqrt 6 - real.sqrt 2)
  (condition_angle_BAC : angle_BAC = 15) :
  ∃ (angle_BDC : ℝ), angle_BDC = 45 := 
sorry

end acute_angle_between_base_and_median_is_45_l717_717200


namespace sqrt_236000_approx_l717_717026

-- Conditions provided in the problem
def sqrt_23_6_approx : ℝ := 4.858
def sqrt_2_36_approx : ℝ := 1.536

-- Prove the equivalent proof problem
theorem sqrt_236000_approx :
  -real.sqrt 236000 ≈ -485.8 := 
  sorry

end sqrt_236000_approx_l717_717026


namespace sum_abs_a_i_l717_717379

def P (x : ℚ) : ℚ := 1 + (1/4) * x - (1/8) * x^2

def Q (x : ℚ) : ℚ := (P x) * (P (x^2)) * (P (x^4)) * (P (x^8)) * (P (x^16))

theorem sum_abs_a_i : ∑ i in finset.range 61, | a i | = (5 / 8) ^ 5 := 
by sorry

end sum_abs_a_i_l717_717379


namespace avg_megabyte_usage_per_hour_l717_717295

theorem avg_megabyte_usage_per_hour (megabytes : ℕ) (days : ℕ) (hours : ℕ) (avg_mbps : ℕ)
  (h1 : megabytes = 27000)
  (h2 : days = 15)
  (h3 : hours = days * 24)
  (h4 : avg_mbps = megabytes / hours) : 
  avg_mbps = 75 := by
  sorry

end avg_megabyte_usage_per_hour_l717_717295


namespace f_4_plus_f_10_eq_2_l717_717415

noncomputable def f : ℝ → ℝ := sorry

axiom domain_of_f : ∀ x : ℝ, f x ∈ ℝ
axiom symmetry1 : ∀ x : ℝ, f (2 - x) = f x
axiom symmetry2 : ∀ x : ℝ, f (4 - x) = f x
axiom f_0_is_1 : f 0 = 1

theorem f_4_plus_f_10_eq_2 : f 4 + f 10 = 2 :=
by
  -- Proof code will go here
  sorry

end f_4_plus_f_10_eq_2_l717_717415


namespace could_all_be_telling_truth_l717_717615

-- Define the grades
def grades := {3, 4, 5}

-- Define the students' grades
structure StudentGrades :=
  (yegor grades: Set Nat)
  (nikita grades: Set Nat)
  (innokenty grades: Set Nat)

-- Define the conditions (test statements) E1, E2, E3
def E1 (g : StudentGrades) : Prop :=
  (5 ∈ g.yegor ∧ 4 ∈ g.yegor) →
  (4 ∈ g.nikita ∧ 3 ∈ g.nikita)

def E2 (g : StudentGrades) : Prop :=
  (5 ∈ g.nikita ∧ 4 ∈ g.nikita) →
  (4 ∈ g.innokenty ∧ 3 ∈ g.innokenty)

def E3 (g : StudentGrades) : Prop :=
  (5 ∈ g.innokenty ∧ 4 ∈ g.innokenty) →
  (5 ∈ g.yegor ∧ 3 ∈ g.yegor)

-- Define the final problem statement
theorem could_all_be_telling_truth (g : StudentGrades) :
  (5 ∈ g.yegor ∧ 4 ∈ g.yegor ∧ 3 ∈ g.yegor) →
  (5 ∈ g.nikita ∧ 4 ∈ g.nikita ∧ 3 ∈ g.nikita) →
  (5 ∈ g.innokenty ∧ 4 ∈ g.innokenty ∧ 3 ∈ g.innokenty) →
  E1 g ∧ E2 g ∧ E3 g :=
  by
  sorry

end could_all_be_telling_truth_l717_717615


namespace marble_combinations_l717_717580

/--
Tom has:
1. 1 red marble
2. 1 blue marble
3. 2 identical green marbles
4. 3 identical yellow marbles

Prove that Tom can create 19 different groups of two marbles.
-/
theorem marble_combinations : 
  let red := 1,
      blue := 1,
      green := 2,
      yellow := 3 in
  (binom green 2 + binom yellow 2 + binom red red * binom blue blue + 
   binom red red * binom green green + binom red red * binom yellow yellow +
   binom blue blue * binom green green + binom blue blue * binom yellow yellow +
   binom green green * binom yellow yellow) = 19 := 
by 
  sorry

end marble_combinations_l717_717580


namespace expression_value_l717_717728

noncomputable def evaluate_expression : ℝ :=
  Real.logb 2 (3 * 11 + Real.exp (4 - 8)) + 3 * Real.sin (Real.pi^2 - Real.sqrt ((6 * 4) / 3 - 4))

theorem expression_value : evaluate_expression = 3.832 := by
  sorry

end expression_value_l717_717728


namespace no_sum_sixteen_l717_717983

theorem no_sum_sixteen (dice : Fin 5 → ℕ) (h_product: (∏ i, dice i) = 72) :
  (∑ i, dice i) ≠ 16 :=
sorry -- proof goes here

end no_sum_sixteen_l717_717983


namespace derivative_of_y_l717_717554

noncomputable def y (x : ℝ) : ℝ := (sin x) / x

theorem derivative_of_y (x : ℝ) : deriv (λ x, (sin x) / x) x = (x * cos x - sin x) / (x^2) :=
by
  sorry

end derivative_of_y_l717_717554


namespace lg5_value_l717_717402

variable {a b : ℝ}

def log8_eq_a (a : ℝ) : Prop := (Real.log 3 / Real.log 8 = a)
def log3_eq_b (b : ℝ) : Prop := (Real.log 5 / Real.log 3 = b)
def lg5 (a b : ℝ) : ℝ := 3 * a * b / (1 + 3 * a * b)

theorem lg5_value (ha : log8_eq_a a) (hb : log3_eq_b b) : Real.log 5 / Real.log 10 = lg5 a b := by
  sorry

end lg5_value_l717_717402


namespace eight_pow_eq_eight_l717_717086

-- Given condition
variable (x : ℝ) (h : 8^(3 * x) = 512)

-- Goal to prove
theorem eight_pow_eq_eight : 8^(3 * x - 2) = 8 :=
by
  sorry

end eight_pow_eq_eight_l717_717086


namespace average_race_time_l717_717339

theorem average_race_time (carlos_time diego_half_time : ℝ) (diego_total_time average_time_seconds : ℝ) 
                          (Diego_finishes : diego_total_time = diego_half_time * 2)
                          (Total_time : carlos_time + diego_total_time = 8)
                          (Average_time : (carlos_time + diego_total_time) / 2 = 4)
                          (Conversion : 4 * 60 = average_time_seconds) :
  average_time_seconds = 240 :=
by
  -- Definitions and conditions
  have H1 : diego_total_time = 2.5 * 2 := Diego_finishes
  have H2 : carlos_time + diego_total_time = 8 := Total_time
  have H3 : (carlos_time + diego_total_time) / 2 = 4 := Average_time
  have H4 : 4 * 60 = average_time_seconds := Conversion
  -- Concluding the proof
  sorry

end average_race_time_l717_717339


namespace sufficient_not_necessary_l717_717770

variable (x : ℝ)

theorem sufficient_not_necessary (h : x^2 - 3 * x + 2 > 0) : x > 2 → (∀ x : ℝ, x^2 - 3 * x + 2 > 0 ↔ x > 2 ∨ x < -1) :=
by
  sorry

end sufficient_not_necessary_l717_717770


namespace no_valid_prime_p_l717_717547

theorem no_valid_prime_p (p : ℕ) (hp : Nat.Prime p) :
    2017_p + 402_p + 114_p + 230_p + 7_p ≠ 301_p + 472_p + 503_p :=
by 
  -- Conversion to decimal and the resulting polynomial equation:
  have eq : 2 * p^3 - 4 * p^2 - 5 * p + 15 = 0 := sorry
  
  -- Actually testing the given equation for possible primes:
  cases' hp with hp_even hp_prime,
  { exact eq.symm.dne (by norm_num) },
  { norm_num at eq, contradiction }
  sorry

end no_valid_prime_p_l717_717547


namespace area_of_quadrilateral_l717_717288

/-- Given a cube with side length 2, and points P, Q, R, S on the cube such that:
- P is at (0,0,0),
- Q is at (1,1,0),
- R is at (0,0,1), and
- S is at (0,0,1.5)
Prove that the area of quadrilateral PQRS is 1/2. -/
theorem area_of_quadrilateral (P Q R S : ℝ × ℝ × ℝ)
    (hP : P = (0, 0, 0))
    (hQ : Q = (1, 1, 0))
    (hR : R = (0, 0, 1))
    (hS : S = (0, 0, 1.5)) :
    area_of_quadrilateral P Q R S = 1/2 := 
sorry

end area_of_quadrilateral_l717_717288


namespace max_distance_curve_line_l717_717801

theorem max_distance_curve_line :
  let l := {P : ℝ × ℝ | P.1 + P.2 = 2}  -- The line equation x + y - 2 = 0
  let C := {P : ℝ × ℝ | P.1^2 + P.2^2 = 1}  -- The curve equation x^2 + y^2 = 1
  ∃ M, (∀ P ∈ C, ∀ Q ∈ l, dist P Q ≤ M) ∧ (∃ P ∈ C, ∃ Q ∈ l, dist P Q = M) ∧ M = 1 + √2 :=
begin
  sorry
end

end max_distance_curve_line_l717_717801


namespace parabola_intersection_l717_717226

theorem parabola_intersection:
  (∀ x y1 y2 : ℝ, (y1 = 3 * x^2 - 6 * x + 6) ∧ (y2 = -2 * x^2 - 4 * x + 6) → y1 = y2 → x = 0 ∨ x = 2 / 5) ∧
  (∀ a c : ℝ, a = 0 ∧ c = 2 / 5 ∧ c ≥ a → c - a = 2 / 5) :=
by sorry

end parabola_intersection_l717_717226


namespace modular_inverse_l717_717357

/-- Define the number 89 -/
def a : ℕ := 89

/-- Define the modulus 90 -/
def n : ℕ := 90

/-- The condition given in the problem -/
lemma pow_mod (h : a ≡ -1 [MOD n]) : (a * a) % n = 1 % n := by 
  sorry

/-- The main statement to prove the modular inverse -/
theorem modular_inverse (h : a ≡ -1 [MOD n]) : (a * a) % n = 1 % n → a ≡ a⁻¹ [MOD n] := by
  intro h1
  have h2 : a⁻¹ % n = a % n := by 
    sorry
  exact h2

end modular_inverse_l717_717357


namespace range_of_s_l717_717719

open Set Real

def s (x : ℝ) : ℝ := 1 / (2 - x) ^ 3

theorem range_of_s :
  (range s) = (Iio 0 ∪ Ioi 0) :=
sorry

end range_of_s_l717_717719


namespace total_population_of_cities_l717_717575

theorem total_population_of_cities 
  (num_cities : ℕ) (avg_population : ℕ) 
  (h1 : num_cities = 25)
  (h2 : avg_population = (4000 + 4500) / 2) 
  : num_cities * avg_population = 106250 := 
by 
  rw [h1, h2]
  norm_num
  sorry

end total_population_of_cities_l717_717575


namespace parabola_focus_distance_l717_717055

-- Definition of the parabola
def parabola (P : ℝ × ℝ) : Prop :=
  let (m, n) := P in n^2 = 8 * m

-- Definitions of focus, directrix, point on parabola, and conditions
def focus : (ℝ × ℝ) := (2, 0)
def directrix (x : ℝ) : Prop := x = -2
def perpendicular_to_directrix (P A : ℝ × ℝ) : Prop :=
  P.1 = A.1 ∧ P.2 = 0

def angle_of_inclination (A F : ℝ × ℝ) (θ : ℝ) : Prop :=
  θ = 2 * Real.pi / 3

-- The proof problem statement
theorem parabola_focus_distance (P F A : ℝ × ℝ) (m n : ℝ) :
  parabola (m, n) →
  F = (2,0) →
  directrix (-2) →
  perpendicular_to_directrix (P, A) →
  angle_of_inclination (A, F) (2 * Real.pi / 3) →
  P = (m, n) →
  |- PF| = 8 →

  abs ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 64 :=
by sorry

end parabola_focus_distance_l717_717055


namespace perpendicular_lines_l717_717013

/-- 
Given the lines l1: (a+2)x + (1-a)y - 1 = 0 and l2: (a-1)x + (2a+3)y + 2 = 0,
prove that the value of a such that l1 is perpendicular to l2 is a = 5/3.
-/
theorem perpendicular_lines (a : ℝ) :
  let m1 := - (a + 2) / (1 - a),
      m2 := - (a - 1) / (2 * a + 3)
  in m1 * m2 = -1 → a = 5 / 3 :=
by
  intros m1 m2 h
  have eq1 : m1 = - (a + 2) / (1 - a),
  have eq2 : m2 = - (a - 1) / (2 * a + 3),
  sorry

end perpendicular_lines_l717_717013


namespace sqrt_mul_sqrt_eq_sqrt_mul_l717_717381

theorem sqrt_mul_sqrt_eq_sqrt_mul (a b : ℝ) : (sqrt a * sqrt b = sqrt (a * b)) → (a ≥ 0 ∧ b ≥ 0) :=
by sorry

end sqrt_mul_sqrt_eq_sqrt_mul_l717_717381


namespace find_f2014_l717_717423

def f : ℕ+ → ℤ :=
  λ n, if n=1 then 2 else if n=2 then -3 else f(n-1) - f(n-2)

theorem find_f2014 :
  f 2014 = -2 := by
  sorry

end find_f2014_l717_717423


namespace option_D_correct_l717_717797

variables {Point Line Plane : Type}
variables (α β : Plane) (a b c : Line)
variables (P : Point)
variables (contains : Line → Plane → Prop)
variables (parallel : Line → Line → Prop)
variables (perpendicular : Line → Line → Prop)
variables (plane_parallel : Plane → Plane → Prop)
variables (line_parallel_plane : Line → Plane → Prop)

-- Given conditions
axiom plane_distinct : α ≠ β
axiom line_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c
axiom a_in_α : contains a α
axiom b_in_α : contains b α
axiom a_b_intersect : ∃ P, contains a α ∧ contains b α
axiom c_perpendicular_a : perpendicular c a
axiom c_perpendicular_b : perpendicular c b
axiom c_parallel_plane_β : line_parallel_plane c β

-- Statement to prove
theorem option_D_correct : plane_parallel α β :=
by
  sorry

end option_D_correct_l717_717797


namespace plane_forms_equal_angles_l717_717674

variables (a b c d e f g h k x y z : ℝ)

def bisector_plane_1 : Prop :=
  a * x + b * y + c * z + d = 0

def bisector_plane_2 : Prop :=
  e * x + f * y + g * z + h = 0

def plane_p : Prop :=
  (a + e) / 2 * x + (b + f) / 2 * y + (c + g) / 2 * z + k = 0

theorem plane_forms_equal_angles 
    (H1 : bisector_plane_1 a b c d x y z)
    (H2 : bisector_plane_2 e f g h x y z) :
    plane_p (a + e) / 2 (b + f) / 2 (c + g) / 2 k x y z := 
sorry

end plane_forms_equal_angles_l717_717674


namespace tangent_line_slope_at_3_l717_717834

noncomputable def f : ℝ → ℝ := sorry

theorem tangent_line_slope_at_3 :
  (∀ x : ℝ, 2 * x + (f x) + 1 = 0) → deriv f 3 = -2 :=
begin
  intro h,
  sorry
end

end tangent_line_slope_at_3_l717_717834


namespace sum_of_digits_unique_n_l717_717376

-- Definition to state the problem.
def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

-- The main theorem statement
theorem sum_of_digits_unique_n :
  ∃! (n : ℕ), log_base 3 (log_base 27 n) = log_base 9 (log_base 3 n)
  → 27 = (sum_digits n) :=
sorry

-- Helper function to compute the sum of digits
def sum_digits (n : ℕ) : ℕ :=
  (n.toString.data.foldl (λ acc c, acc + (c.toNat - '0'.toNat)) 0)

end sum_of_digits_unique_n_l717_717376


namespace scientific_notation_correct_l717_717234

noncomputable def thickness : ℝ := 0.000136

theorem scientific_notation_correct : thickness = 1.36 * 10^(-4) :=
by
  sorry

end scientific_notation_correct_l717_717234


namespace problem_part1_problem_part2_l717_717062

theorem problem_part1 (m : ℝ) (a : Fin 8 → ℝ) (h_expansion : (x - m)^7 = ∑ i in Finset.range 8, a i * x ^ i)
  (h_coeff_x4 : a 4 = -35) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7) = 1 := sorry

theorem problem_part2 (m : ℝ) (a : Fin 8 → ℝ) (h_expansion : (x - m)^7 = ∑ i in Finset.range 8, a i * x ^ i)
  (h_coeff_x4 : a 4 = -35) :
  (a 1 + a 3 + a 5 + a 7) = 64 := sorry

end problem_part1_problem_part2_l717_717062


namespace smallest_b_is_720_l717_717890

noncomputable def smallest_possible_b (P : ℤ[X]) (b : ℤ) : ℤ :=
  if b > 0 ∧ P.eval 1 = b ∧ P.eval 4 = b ∧ P.eval 5 = b ∧ P.eval 8 = b ∧ 
    P.eval 2 = -b ∧ P.eval 3 = -b ∧ P.eval 6 = -b ∧ P.eval 7 = -b then 
    720
  else 
    sorry

theorem smallest_b_is_720 (P : ℤ[X]) (b : ℤ) (h1 : b > 0) 
  (h2 : P.eval 1 = b) (h3 : P.eval 4 = b) (h4 : P.eval 5 = b) 
  (h5 : P.eval 8 = b) (h6 : P.eval 2 = -b) (h7 : P.eval 3 = -b) 
  (h8 : P.eval 6 = -b) (h9 : P.eval 7 = -b) : 
  smallest_possible_b P b = 720 := 
sorry

end smallest_b_is_720_l717_717890


namespace rectangle_fold_segment_EF_length_l717_717341

-- Conditions for the problem
variables (AB BC : ℝ) (AB_length : AB = 5) (BC_length : BC = 14)

-- Proof statement
theorem rectangle_fold_segment_EF_length (h : is_rectangular_folding AB BC AB_length BC_length) :
  ∃ EF : ℝ, EF = 5.0357 := 
sorry

end rectangle_fold_segment_EF_length_l717_717341


namespace find_f1_l717_717411

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x
def odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem find_f1 (f : ℝ → ℝ)
  (h_periodic : periodic f 2)
  (h_odd : odd f) :
  f 1 = 0 :=
sorry

end find_f1_l717_717411


namespace soccer_team_points_l717_717195

theorem soccer_team_points 
  (total_games wins losses draws : ℕ)
  (points_per_win points_per_draw points_per_loss : ℕ)
  (h_total_games : total_games = 20)
  (h_wins : wins = 14)
  (h_losses : losses = 2)
  (h_draws : draws = total_games - (wins + losses))
  (h_points_per_win : points_per_win = 3)
  (h_points_per_draw : points_per_draw = 1)
  (h_points_per_loss : points_per_loss = 0) :
  (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) = 46 :=
by
  -- the actual proof steps will be inserted here
  sorry

end soccer_team_points_l717_717195


namespace percent_3rd_graders_combined_l717_717972

-- Define the given conditions
def maplewood_total : ℕ := 150
def brookside_total : ℕ := 250
def maplewood_percent_3rd : ℕ := 15
def brookside_percent_3rd : ℕ := 18

-- Prove the statement
theorem percent_3rd_graders_combined
  (maplewood_total = 150)
  (brookside_total = 250)
  (maplewood_percent_3rd = 15)
  (brookside_percent_3rd = 18)
  : (maplewood_percent_3rd * maplewood_total + brookside_percent_3rd * brookside_total) / (maplewood_total + brookside_total) = 17 := 
by 
  sorry

end percent_3rd_graders_combined_l717_717972


namespace div_by_7_of_sum_div_by_7_l717_717177

theorem div_by_7_of_sum_div_by_7 (x y z : ℤ) (h : 7 ∣ x^3 + y^3 + z^3) : 7 ∣ x * y * z := by
  sorry

end div_by_7_of_sum_div_by_7_l717_717177


namespace ellipse_hyperbola_tangent_l717_717948

variable {x y m : ℝ}

theorem ellipse_hyperbola_tangent (h : ∃ x y, x^2 + 9 * y^2 = 9 ∧ x^2 - m * (y + 1)^2 = 1) : m = 2 := 
by 
  sorry

end ellipse_hyperbola_tangent_l717_717948


namespace minimum_value_l717_717044

open Real

theorem minimum_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eqn : 2 * x + y = 2) :
    ∃ x y, (0 < x) ∧ (0 < y) ∧ (2 * x + y = 2) ∧ (x + sqrt (x^2 + y^2) = 8 / 5) :=
sorry

end minimum_value_l717_717044


namespace newer_train_distance_and_time_l717_717296

-- Definitions for the given conditions
def older_train_distance := 300
def speed_factor := 1.3
def newer_speed := 120

-- Proof problem statement
theorem newer_train_distance_and_time :
  (let newer_train_distance := speed_factor * older_train_distance in
   newer_train_distance = 390 ∧ (newer_train_distance / newer_speed) = 3.25) :=
by
  sorry

end newer_train_distance_and_time_l717_717296


namespace product_of_differences_divisible_by_12_l717_717395

theorem product_of_differences_divisible_by_12 (a b c d : ℤ) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) := 
begin
  sorry
end

end product_of_differences_divisible_by_12_l717_717395


namespace max_circle_area_l717_717572

noncomputable def circle_max_area (side_length : ℝ) (measurement_error : ℝ) : ℝ := 
  let max_side_length := side_length + measurement_error
  let radius := max_side_length / 2
  real.pi * radius^2

theorem max_circle_area :
  circle_max_area 5 0.2 = 66.769 := 
by 
  -- Calculate the values
  have h1 : 5 + 0.2 = 5.2 := rfl 
  have h2 : 5.2 / 2 = 2.6 := by norm_num
  have radius := 2.6
  -- Simplify the area calculation
  have h3 : real.pi * radius^2 = 21.2372 * real.pi := by norm_num
  have h4 : (21.2372 : ℝ) * real.pi ≈ 66.769 := by norm_num -- Approximation step
  exact h4
sorry

end max_circle_area_l717_717572


namespace original_price_sarees_l717_717963

theorem original_price_sarees
  (P : ℝ)
  (h : 0.90 * 0.85 * P = 378.675) :
  P = 495 :=
sorry

end original_price_sarees_l717_717963


namespace range_of_m_l717_717803

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if h : x > 0 then 2^x - m else -x^2 - 2 * m * x

def g (x : ℝ) (m : ℝ) : ℝ := f x m - m

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, g x m = 0 → x = 0 ∨ x = -2*m ∨ ((x > 0) ∧ (2^x - m = m))) → m > 1 :=
sorry

end range_of_m_l717_717803


namespace number_of_subsets_l717_717810

theorem number_of_subsets (B : Set ℤ) (hB : B = {-1, 0, 1}) : 
  ∃ n : ℕ, (∀ A : Set ℤ, A ⊆ B → Set.finite A) ∧ n = 2 ^ 3 :=
by
  use 8
  sorry

end number_of_subsets_l717_717810


namespace tangent_line_eq_min_value_f_on_interval_range_of_a_l717_717806

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (x a : ℝ) := -x^2 + a * x - 3
noncomputable def t (t : ℝ) := t + 1 / Real.exp 1
noncomputable def sec_f (x : ℝ) := f (t x)

theorem tangent_line_eq (x : ℝ) :
    let fx := f x in
    let fx' := Real.log 1 + 1 in
    let tangent := fun (x : ℝ) => 1 * (x - 1) + 0 in
    let tang_eq := fun (x : ℝ) => x - 1 in
    tangent x = tang_eq x := by
  sorry

theorem min_value_f_on_interval (t : ℝ) (ht : 0 < t) :
    let domain1 := t
    let domain2 := t + 1 / Real.exp 1 in
    if hpos : 0 < t ∧ t < 1 / Real.exp 1 then
      f (1 / Real.exp 1) = -1 / Real.exp 1
    else
      f t = t * Real.log t := by
  sorry

theorem range_of_a (a : ℝ) :
    let h := fun (x : ℝ) => 2 * Real.log x + x + 3 / x
    h 1 = 4 → a ≤ 4 := by
  sorry

end tangent_line_eq_min_value_f_on_interval_range_of_a_l717_717806


namespace thirtieth_term_of_arithmetic_sequence_l717_717254

theorem thirtieth_term_of_arithmetic_sequence :
  ∀ (a d n : ℕ), a = 2 → d = 3 → n = 30 → (a + (n - 1) * d = 89) :=
by
  intros a d n ha hd hn
  rw [ha, hd, hn]
  sorry

end thirtieth_term_of_arithmetic_sequence_l717_717254


namespace calculateRemainingMoney_l717_717766

def initialAmount : ℝ := 100
def actionFiguresCount : ℕ := 3
def actionFigureOriginalPrice : ℝ := 12
def actionFigureDiscount : ℝ := 0.25
def boardGamesCount : ℕ := 2
def boardGamePrice : ℝ := 11
def puzzleSetsCount : ℕ := 4
def puzzleSetPrice : ℝ := 6
def salesTax : ℝ := 0.05

theorem calculateRemainingMoney :
  initialAmount - (
    (actionFigureOriginalPrice * (1 - actionFigureDiscount) * actionFiguresCount) +
    (boardGamePrice * boardGamesCount) +
    (puzzleSetPrice * puzzleSetsCount)
  ) * (1 + salesTax) = 23.35 :=
by
  sorry

end calculateRemainingMoney_l717_717766


namespace greatest_factor_power_greatest_factor_power_max_l717_717589

theorem greatest_factor_power (x : ℕ) (hx : 7 ^ x ∣ 49 ^ 15) : x ≤ 30 :=
by
  -- Proof goes here
  sorry

theorem greatest_factor_power_max : ∃ x : ℕ, 7 ^ x ∣ 49 ^ 15 ∧ ∀ y : ℕ, 7 ^ y ∣ 49 ^ 15 → y ≤ 30 :=
by
  use 30
  split
  -- Proof that 7^30 divides 49^15
  sorry
  -- Proof that no greater power of 7 divides 49^15
  sorry

end greatest_factor_power_greatest_factor_power_max_l717_717589


namespace tower_height_greater_than_103_3_l717_717216

theorem tower_height_greater_than_103_3 :
  ∀ (H : ℝ), (∀ (d : ℝ) (θ : ℝ), d = 100 → θ = 46 → H = d * Real.tan (θ * Real.pi / 180)) → H > 103.3 :=
by {
  intro H,
  intro hc,
  have h₁ : H = 100 * Real.tan (46 * Real.pi / 180) := hc 100 46 rfl rfl,
  sorry
}

end tower_height_greater_than_103_3_l717_717216


namespace discriminant_of_quadratic_l717_717944

theorem discriminant_of_quadratic : 
  let a := 1
  let b := -7
  let c := 4
  Δ = b ^ 2 - 4 * a * c
  (b ^ 2 - 4 * a * c) = 33 :=
by
  -- definitions of a, b, and c
  let a := 1
  let b := -7
  let c := 4
  -- definition of Δ
  let Δ := b ^ 2 - 4 * a * c
  -- given the quadratic equation x^2 - 7x + 4 = 0, prove that Δ = 33
  show (b ^ 2 - 4 * a * c) = 33,
  -- proof is omitted
  sorry

end discriminant_of_quadratic_l717_717944


namespace vector_triple_product_identity_l717_717135

open Real EuclideanSpace FiniteDimensional

-- Definitions for vectors
def u₁ : ℝ^3 := ![1, 1, 0]
def u₂ : ℝ^3 := ![0, 1, 1]
def u₃ : ℝ^3 := ![1, 0, 1]

-- The equivalence proof theorem
theorem vector_triple_product_identity (v : ℝ^3) :
  (u₁ ⨯ (v ⨯ u₁)) + (u₂ ⨯ (v ⨯ u₂)) + (u₃ ⨯ (v ⨯ u₃)) = 2 • v :=
by sorry

end vector_triple_product_identity_l717_717135


namespace generate_13121_not_generate_12131_l717_717567

theorem generate_13121 : ∃ n m : ℕ, 13121 + 1 = 2^n * 3^m := by
  sorry

theorem not_generate_12131 : ¬∃ n m : ℕ, 12131 + 1 = 2^n * 3^m := by
  sorry

end generate_13121_not_generate_12131_l717_717567


namespace find_line_eqn_find_circle_eqn_l717_717430

def parabola (x y : ℝ) : Prop := x^2 = 4 * y

def midpoint_condition (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = 4

def line_containing_chord (x y : ℝ) : Prop :=
  x - 2 * y + 7 = 0

def tangent_condition (x0 : ℝ) : Prop :=
  x0 = 2

def circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 4

theorem find_line_eqn (x1 y1 x2 y2 : ℝ) :
  parabola x1 y1 →
  parabola x2 y2 →
  midpoint_condition x1 y1 x2 y2 →
  line_containing_chord (x1 + x2) (y1 + y2) :=
sorry

theorem find_circle_eqn (x0 y0 : ℝ) :
  parabola x0 y0 →
  tangent_condition x0 →
  circle x0 y0 :=
sorry

end find_line_eqn_find_circle_eqn_l717_717430


namespace last_number_aryana_counts_l717_717936

theorem last_number_aryana_counts (a d : ℤ) (h_start : a = 72) (h_diff : d = -11) :
  ∃ n : ℕ, (a + n * d > 0) ∧ (a + (n + 1) * d ≤ 0) ∧ a + n * d = 6 := by
  sorry

end last_number_aryana_counts_l717_717936


namespace smallest_pos_multiple_6_15_is_30_l717_717750

theorem smallest_pos_multiple_6_15_is_30 :
  ∃ b > 0, 6 ∣ b ∧ 15 ∣ b ∧ (∀ b', b' > 0 ∧ b' < b → ¬ (6 ∣ b' ∧ 15 ∣ b')) :=
by
  -- Implementation to be done
  sorry

end smallest_pos_multiple_6_15_is_30_l717_717750


namespace jane_blouse_cost_l717_717488

theorem jane_blouse_cost :
  ∀ (num_skirts : ℕ) (cost_per_skirt cost_per_blouse num_blouses total_payment change received change_amount total_skirt_cost total_spent total_blouse_cost: ℕ),
      num_skirts = 2 →
      cost_per_skirt = 13 →
      num_blouses = 3 →
      total_payment = 100 →
      received = 56 →
      change_amount = (total_payment - received) →
      total_skirt_cost = (num_skirts * cost_per_skirt) →
      total_spent = (total_payment - received) →
      total_blouse_cost = (total_spent - total_skirt_cost) →
      (total_blouse_cost / num_blouses) = 6 :=
by
  intros num_skirts cost_per_skirt cost_per_blouse num_blouses total_payment received change_amount total_skirt_cost total_spent total_blouse_cost
  intros h_num_skirts h_cost_per_skirt h_num_blouses h_total_payment h_received h_change_amount h_total_skirt_cost h_total_spent h_total_blouse_cost
  rw [h_total_skirt_cost, h_total_spent, h_total_blouse_cost, h_change_amount, h_num_skirts, h_cost_per_skirt, h_num_blouses, h_total_payment, h_received]
  exact sorry

end jane_blouse_cost_l717_717488


namespace distinct_collections_of_letters_l717_717908

open Classical

-- Definitions and setup
def vowels : list Char := ['A', 'E', 'I', 'O', 'O', 'U']
def consonants : list Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

-- The word in question
def word : list Char := ['B', 'I', 'O', 'G', 'R', 'A', 'P', 'H', 'Y']

-- The count of distinct collections given the conditions
noncomputable def count_distinct_collections : ℕ := 
  let vowels_remaining := vowels.filter (λ x, x ∈ word)
  let consonants_remaining := consonants.filter (λ x, x ∈ word)
  let possible_cases := 
    (choose 2 vowels_remaining).length * (choose 3 consonants_remaining).length +
    (choose 1 vowels_remaining).length * (choose 1 consonants_remaining).length
  possible_cases

-- The proof problem statement
theorem distinct_collections_of_letters : count_distinct_collections = 60 := by
  sorry

end distinct_collections_of_letters_l717_717908


namespace units_digit_17_pow_2007_l717_717602

theorem units_digit_17_pow_2007 : (17^2007) % 10 = 3 :=
by sorry

end units_digit_17_pow_2007_l717_717602


namespace loot_box_cost_l717_717876

variable (C : ℝ) -- Declare cost of each loot box as a real number

-- Conditions (average value of items, money spent, loss)
def avg_value : ℝ := 3.5
def money_spent : ℝ := 40
def avg_loss : ℝ := 12

-- Derived equation
def equation := avg_value * (money_spent / C) = money_spent - avg_loss

-- Statement to prove
theorem loot_box_cost : equation C → C = 5 := by
  sorry

end loot_box_cost_l717_717876


namespace trapezoid_circumcircle_equal_distances_l717_717883

open EuclideanGeometry

noncomputable def midpoint (P Q : Point) : Point := Point.midpoint P Q

theorem trapezoid_circumcircle_equal_distances
  {A B C D M N Q R G : Point}
  (h_trapezoid : is_right_trapezoid A B C D)
  (h_M : M = midpoint A C)
  (h_N : N = midpoint B D)
  (h_G : G = midpoint M N)
  (h_circle_ABN : ∃ O₁ r₁, (∀ P, P ∈ circle O₁ r₁ ↔ P = A ∨ P = B ∨ P = N))
  (h_circle_CDM : ∃ O₂ r₂, (∀ P, P ∈ circle O₂ r₂ ↔ P = C ∨ P = D ∨ P = M))
  (h_Q_intersect : Q ∈ (line_through B C) ∧ Q ∈ (circle_center_radius (classical.some h_circle_ABN).1 (classical.some h_circle_ABN).2))
  (h_R_intersect : R ∈ (line_through B C) ∧ R ∈ (circle_center_radius (classical.some h_circle_CDM).1 (classical.some h_circle_CDM).2)) :
  dist Q G = dist R G := sorry

end trapezoid_circumcircle_equal_distances_l717_717883


namespace domain_of_f_l717_717735

noncomputable def f (x : ℝ) : ℝ := log (2 * sin x + 1) + sqrt (2 * cos x - 1)

def isDomain (x : ℝ) : Prop :=
  ∃ k : ℤ, (2 * k * Real.pi - Real.pi / 6 < x) ∧ (x ≤ 2 * k * Real.pi + Real.pi / 3)

theorem domain_of_f :
  { x : ℝ | ∃ k : ℤ, 2 * k * Real.pi - Real.pi / 6 < x ∧ x ≤ 2 * k * Real.pi + Real.pi / 3}
= { x : ℝ | f x = f x } :=
sorry

end domain_of_f_l717_717735


namespace probability_AMC8_l717_717666

-- Definitions based on the conditions
def vowels := {'A', 'E', 'I', 'O', 'U'} : Finset Char
def consonants := Finset.filter (λ c => c ∉ vowels) (Finset.range 26).map (λ n => Char.ofNat (n + 65))
def digits := (Finset.range 10).map Char.ofNat

-- The set of all valid license plates
def valid_license_plates : Finset (Char × Char × Char × Char) :=
  Finset.product vowels (Finset.product
    (Finset.sigma consonants (λ c1 => 
      Finset.filter (λ c2 => c1 ≠ c2) consonants))
    digits)

-- The specific license plate "AMC8"
def AMC8 : Char × Char × Char × Char := ('A', 'M', 'C', '8')

-- Lean 4 statement to prove the probability of selecting "AMC8"
theorem probability_AMC8 :
  (valid_license_plates.card : ℚ) = 21000 →
  1 / (valid_license_plates.card : ℚ) = 1 / 21000 :=
by
  sorry

end probability_AMC8_l717_717666


namespace find_XB_l717_717839

-- Define the vertices of the triangle
variables {F O X B P : Type*}

-- Define the angles and lengths
variables
  (angle_FOX : ℝ) (angle_FXO : ℝ) (FO : ℝ)
  (angle_OXB : ℝ) (angle_XBO : ℝ) (OX : ℝ) (XB : ℝ) (BO : ℝ)

-- Assume the given conditions
axiom given_conditions :
  angle_FOX = 80 ∧ 
  angle_FXO = 30 ∧ 
  FO = 1 ∧ 
  angle_OXB = 30 ∧ 
  OX = OX ∧
  XB = BO

-- State the problem
theorem find_XB (HC : given_conditions) : XB = real.sqrt(3) / 3 :=
sorry

end find_XB_l717_717839


namespace clock_angle_4_oclock_l717_717245

theorem clock_angle_4_oclock :
  let total_degrees := 360
  let hours := 12
  let degree_per_hour := total_degrees / hours
  let hour_position := 4
  let minute_hand_position := 0
  let hour_hand_angle := hour_position * degree_per_hour
  hour_hand_angle = 120 := sorry

end clock_angle_4_oclock_l717_717245


namespace no_winning_strategy_l717_717676

def point := ℝ × ℝ

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def valid_segment (p1 p2 : point) : Prop :=
  dist p1 p2 = 1 ∧ ¬ (p1 = p2)

noncomputable def next_point (A B : point) : point :=
  (B.1 + 1, B.2)

theorem no_winning_strategy :
  (∀ (A B : point), valid_segment A B → ∃ (A' : point), valid_segment B A' ∧ dist A A' = 1) →
  (∀ (A B : point), valid_segment A B → ∃ (B' : point), valid_segment A B' ∧ dist B B' = 1) →
  false := sorry

end no_winning_strategy_l717_717676


namespace problem_statement_l717_717778

theorem problem_statement
  (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_mono_dec : ∀ x y, x ≤ 0 → y ≤ 0 → x < y → f y < f x)
  (log2_3_eq_log4_9 : log 3 / log 2 = log 9 / log 4)
  (h1 : 2 > log 3 / log 2)
  (h2 : log 3 / log 2 > log 5 / log 4)
  (h3 : 2^(3/2) > 2) :
  let a := f (log 3 / log 2),
      b := f (log 5 / log 4),
      c := f (2^(3/2))
  in b < a ∧ a < c :=
by {
  sorry
}

end problem_statement_l717_717778


namespace parametric_curve_C_line_tangent_to_curve_C_l717_717861

open Real

-- Definitions of the curve C and line l
def curve_C (ρ θ : ℝ) : Prop := ρ^2 - 4 * ρ * cos θ + 1 = 0

def line_l (t α x y : ℝ) : Prop := x = 4 + t * sin α ∧ y = t * cos α ∧ 0 ≤ α ∧ α < π

-- Parametric equation of curve C
theorem parametric_curve_C :
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * π →
  ∃ x y : ℝ, (x = 2 + sqrt 3 * cos θ ∧ y = sqrt 3 * sin θ ∧
              curve_C (sqrt (x^2 + y^2)) θ) :=
sorry

-- Tangency condition for line l and curve C
theorem line_tangent_to_curve_C :
  ∀ α : ℝ, 0 ≤ α ∧ α < π →
  (∃ t : ℝ, ∃ x y : ℝ, (line_l t α x y ∧ (x - 2)^2 + y^2 = 3 ∧
                        ((abs (2 * cos α - 4 * cos α) / sqrt (cos α ^ 2 + sin α ^ 2)) = sqrt 3)) →
                       (α = π / 6 ∧ x = 7 / 2 ∧ y = - sqrt 3 / 2)) :=
sorry

end parametric_curve_C_line_tangent_to_curve_C_l717_717861


namespace max_triangle_area_l717_717773

theorem max_triangle_area (a b c : ℝ) (h1 : b + c = 8) (h2 : a + b > c)
  (h3 : a + c > b) (h4 : b + c > a) :
  (a - b + c) * (a + b - c) ≤ 64 / 17 :=
by sorry

end max_triangle_area_l717_717773


namespace prime_divisor_form_l717_717529

theorem prime_divisor_form (n : ℕ) (q : ℕ) (hq : (2^(2^n) + 1) % q = 0) (prime_q : Nat.Prime q) :
  ∃ k : ℕ, q = 2^(n+1) * k + 1 :=
sorry

end prime_divisor_form_l717_717529


namespace inequality_proof_l717_717897

theorem inequality_proof
  (a b x y z : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (x_pos : 0 < x) 
  (y_pos : 0 < y) 
  (z_pos : 0 < z) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ (3 / (a + b)) :=
by
  sorry

end inequality_proof_l717_717897


namespace postcard_width_l717_717227

theorem postcard_width (P : ℝ) (h : ℝ) (w : ℝ) (hP : P = 20) (hh : h = 4) (hp : P = 2 * (w + h)) : w = 6 :=
by
  have h1 : 2 * (w + h) = 20 := by rw [hp, hP]
  rw [<- hh] at h1
  have h2 : 2 * (w + 4) = 20 := h1
  have h3 : w + 4 = 10 := by linarith
  have h4 : w = 6 := by linarith
  exact h4

end postcard_width_l717_717227


namespace total_dots_l717_717320

def ladybugs_monday : ℕ := 8
def ladybugs_tuesday : ℕ := 5
def dots_per_ladybug : ℕ := 6

theorem total_dots :
  (ladybugs_monday + ladybugs_tuesday) * dots_per_ladybug = 78 :=
by
  sorry

end total_dots_l717_717320


namespace solve_equation_l717_717934

theorem solve_equation (x : ℝ) : 2 * (x - 2)^2 = 6 - 3 * x ↔ (x = 2 ∨ x = 1 / 2) :=
by
  sorry

end solve_equation_l717_717934


namespace cube_root_eq_square_root_l717_717553

theorem cube_root_eq_square_root (x : ℝ) (h : real.cbrt x = real.sqrt x) : x = 0 := 
  sorry

end cube_root_eq_square_root_l717_717553


namespace average_weight_of_boys_l717_717199

theorem average_weight_of_boys (n1 n2 : ℕ) (w1 w2 : ℚ) 
  (weight_avg_22_boys : w1 = 50.25) 
  (weight_avg_8_boys : w2 = 45.15) 
  (count_22_boys : n1 = 22) 
  (count_8_boys : n2 = 8) 
  : ((n1 * w1 + n2 * w2) / (n1 + n2) : ℚ) = 48.89 :=
by
  sorry

end average_weight_of_boys_l717_717199


namespace platform_length_is_200_1_l717_717632

-- Define the conditions
def train_length : ℝ := 300
def cross_platform_time : ℝ := 30
def cross_pole_time : ℝ := 18

-- Define the speed of the train
def train_speed : ℝ := train_length / cross_pole_time

-- Define the total distance covered when crossing the platform
def total_distance := train_speed * cross_platform_time

-- Define the length of the platform
def platform_length := total_distance - train_length

-- State the theorem
theorem platform_length_is_200_1 : platform_length = 200.1 :=
by
  -- Correctness follows from definition calculations.
  sorry

end platform_length_is_200_1_l717_717632


namespace basis_transformation_l717_717088

variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

theorem basis_transformation (h_basis : ∀ (v : V), ∃ (x y z : ℝ), v = x • a + y • b + z • c) :
  ∀ (v : V), ∃ (x y z : ℝ), v = x • (a + b) + y • (a - c) + z • b :=
by {
  sorry  -- to skip the proof steps for now
}

end basis_transformation_l717_717088


namespace locus_of_K_is_arcs_union_l717_717683

-- Define the fixed points on a circle.
variable (circle : Type)
variable {B C A K : circle}
variable (isOnCircle : circle → Prop)
variable (midPoint : circle → circle → circle)

-- Conditions
axiom B_on_circle : isOnCircle B
axiom C_on_circle : isOnCircle C
axiom A_moves_on_circle : ∀ A, isOnCircle A

-- K is the midpoint of the bent line CAB
axiom K_is_midpoint : ∀ A, K = midPoint A (midPoint B C)

-- Question (translating geometric locus problem)
def geometric_locus_of_K := {x : circle | ∃ (I₁ I₂ I₃ I₄ : circle), x ∈ {I₁, I₂, I₃, I₄}}

-- Statement to prove
theorem locus_of_K_is_arcs_union :
  ∀ A, K ∈ geometric_locus_of_K :=
sorry

end locus_of_K_is_arcs_union_l717_717683


namespace find_first_discount_percentage_l717_717491

noncomputable def percentage_first_discount (x : ℝ) : Prop :=
  let initial_price := 32
  let final_price := 18
  let second_discount := 0.25
  let discounted_price := initial_price - (x / 100) * initial_price
  let price_after_second_discount := discounted_price - second_discount * discounted_price
  price_after_second_discount = final_price

theorem find_first_discount_percentage :
  (percentage_first_discount 25) :=
by
  let x := 25
  let initial_price := 32
  let final_price := 18
  let second_discount := 0.25
  let discounted_price := initial_price - (x / 100) * initial_price
  let price_after_second_discount := discounted_price - second_discount * discounted_price
  have h : price_after_second_discount = final_price := by
    sorry
  exact h

end find_first_discount_percentage_l717_717491


namespace not_possible_capture_l717_717656

def within_square (x y : ℝ) (side_length : ℝ) :=
  abs x ≤ side_length / 2 ∧ abs y ≤ side_length / 2

def on_square_perimeter (x y : ℝ) (side_length : ℝ) :=
  (abs x = side_length / 2 ∧ abs y ≤ side_length / 2) ∨ 
  (abs y = side_length / 2 ∧ abs x ≤ side_length / 2)

theorem not_possible_capture 
  (u v : ℝ) 
  (side_length : ℝ) 
  (ratio_condition : u = 1 ∧ v = 3 ∧ u / v = 1 / 3) 
  (initial_policeman : ∃ x y, within_square x y side_length) 
  (initial_gangster : ∃ x y, on_square_perimeter x y side_length) :
  ¬(∃ t x_p y_p x_g y_g, (0 ≤ t) ∧ 
    (u * t ≥ distance (x_p, y_p) (x_g, y_g)) ∧ 
    within_square x_p y_p side_length ∧ 
    on_square_perimeter x_g y_g side_length) :=
sorry

end not_possible_capture_l717_717656


namespace taylor_sin_3z_2_correct_taylor_exp_z_correct_taylor_frac_z_correct_taylor_ln_3mz_correct_l717_717729

open Complex

def taylor_sin_3z_2_power_zp1 (n : ℕ) : ℂ :=
  ∑ i in Finset.range n, (3^i / Real.factorial i) * Complex.sin (-1 + (Real.pi * i)/2)

def taylor_exp_z_power_2zp1 (n : ℕ) : ℂ :=
  (1 / Real.exp 1/2) * ∑ i in Finset.range n, ((2 : ℂ))^i * (Complex.ofReal (Real.factorial i))

def taylor_frac_z (n : ℕ) : ℂ :=
  ∑ i in Finset.range n, (-2 : ℂ) * Complex.ofReal i + (3/2) * ((Complex.ofReal i / 2) : ℂ)

def taylor_ln_3mz (n : ℕ) : ℂ :=
  Complex.ln 3 - ∑ i in Finset.range n, Complex.ofReal (1 / 3^i)

theorem taylor_sin_3z_2_correct : 
  ∀ (z : ℂ), ∃ n : ℕ, |taylor_sin_3z_2_power_zp1 n| = +∞ := by
  sorry

theorem taylor_exp_z_correct : 
  ∀ (z : ℂ), ∃ n : ℕ, |taylor_exp_z_power_2zp1 n| = +∞ := by
  sorry

theorem taylor_frac_z_correct : 
  ∀ (z : ℂ), ∃ n : ℕ, |taylor_frac_z n| = 1 := by
  sorry

theorem taylor_ln_3mz_correct :
  ∀ (z : ℂ), ∃ n : ℕ, |taylor_ln_3mz n| = 3 := by
  sorry

end taylor_sin_3z_2_correct_taylor_exp_z_correct_taylor_frac_z_correct_taylor_ln_3mz_correct_l717_717729


namespace part1_part2_part3_l717_717664

-- Definitions from the problem
def initial_cost_per_bottle := 16
def initial_selling_price := 20
def initial_sales_volume := 60
def sales_decrease_per_yuan_increase := 5

def daily_sales_volume (x : ℕ) : ℕ :=
  initial_sales_volume - sales_decrease_per_yuan_increase * x

def profit_per_bottle (x : ℕ) : ℕ :=
  (initial_selling_price - initial_cost_per_bottle) + x

def daily_profit (x : ℕ) : ℕ :=
  daily_sales_volume x * profit_per_bottle x

-- The proofs we need to establish
theorem part1 (x : ℕ) : 
  daily_sales_volume x = 60 - 5 * x ∧ profit_per_bottle x = 4 + x :=
sorry

theorem part2 (x : ℕ) : 
  daily_profit x = 300 → x = 6 ∨ x = 2 :=
sorry

theorem part3 : 
  ∃ x : ℕ, ∀ y : ℕ, (daily_profit x < daily_profit y) → 
              (daily_profit x = 320 ∧ x = 4) :=
sorry

end part1_part2_part3_l717_717664


namespace ordered_subsequence_exists_l717_717684

-- Definitions based on the problem conditions:
def Warrior := ℕ  -- Assume for simplicity that each warrior can be identified with a natural number representing their height
def infinite_sequence (s : ℕ → Warrior) := ∀ n : ℕ, ∃ m > n, s m ≠ s n  -- Definition of an infinite sequence with different warriors

-- The main theorem to be proven
theorem ordered_subsequence_exists (s : ℕ → Warrior) (h_inf: infinite_sequence s) :
  ∃ (t : ℕ → ℕ), (∀ i j : ℕ, i < j → s (t i) < s (t j)) ∨ (∀ i j : ℕ, i < j → s (t i) > s (t j)) :=
sorry

end ordered_subsequence_exists_l717_717684


namespace volume_ratio_l717_717158

noncomputable theory
open_locale classical

structure Tetrahedron (V : Type*) :=
(V₁ V₂ V₃ V₄ : V)

def volume {V : Type*} [normed_group V] [normed_space ℝ V] (T : Tetrahedron V) : ℝ := sorry

theorem volume_ratio {V : Type*} [normed_group V] [normed_space ℝ V] (T : Tetrahedron V) 
  (B : Π {i j : fin 4}, i ≠ j → V):
  (∃ i : fin 4, volume (Tetrahedron.mk (T.V₁) (B i ⟨1⟩) (B i ⟨2⟩) (B i ⟨3⟩)) ≤ (1/8) * volume T) :=
sorry

end volume_ratio_l717_717158


namespace area_of_triangle_l717_717342

-- Define the given square
def Square : Type := {side_length : ℝ // side_length = 10}

-- Define a point P inside the square
structure PointInSquare (sq : Square) :=
  (P : ℝ × ℝ)
  (is_in_square : 0 ≤ P.1 ∧ P.1 ≤ sq.side_length ∧ 0 ≤ P.2 ∧ P.2 ≤ sq.side_length)

-- Define the vertices of the square
def VertexA (sq : Square) : ℝ × ℝ := (0, 0)
def VertexB (sq : Square) : ℝ × ℝ := (sq.side_length, 0)
def VertexC (sq : Square) : ℝ × ℝ := (sq.side_length, sq.side_length)
def VertexD (sq : Square) : ℝ × ℝ := (0, sq.side_length)

-- Define the condition where distances from P to vertices A, B, and D are equal
def equidistant (sq : Square) (P : PointInSquare sq) : Prop :=
  dist P.P (VertexA sq) = dist P.P (VertexB sq) ∧
  dist P.P (VertexA sq) = dist P.P (VertexD sq)

-- Define the condition where segment PC is perpendicular to AB
def perpendicular_to_AB (sq : Square) (P : PointInSquare sq) : Prop :=
  P.P.2 = sq.side_length / 2 ∧ P.P.1 = sq.side_length / 2

-- Lean statement: Prove that given the conditions, the area of triangle APB is 50 square inches
theorem area_of_triangle (sq : Square)
  (P : PointInSquare sq) 
  (h1 : equidistant sq P) 
  (h2 : perpendicular_to_AB sq P) : 
  let A := VertexA sq
  let B := VertexB sq 
  let area := sq.side_length * sq.side_length / 2
  area = 50 := by sorry

end area_of_triangle_l717_717342


namespace total_corn_yield_after_six_months_l717_717131

-- Define the conditions.
def johnson_yield_per_two_months := 80
def smith_hectares := 2
def smith_yield_factor := 2
def brown_hectares := 1.5
def brown_yield_per_three_months := 50
def taylor_hectares := 0.5
def taylor_yield_per_month := 30

-- Calculate number of periods in six months.
def two_month_periods_in_six_months := 3
def three_month_periods_in_six_months := 2
def one_month_periods_in_six_months := 6

-- Calculate yields for each person.
def johnson_total_yield := johnson_yield_per_two_months * two_month_periods_in_six_months
def smith_yield_per_two_months := johnson_yield_per_two_months * smith_yield_factor
def smith_total_yield := smith_yield_per_two_months * smith_hectares * two_month_periods_in_six_months
def brown_total_yield := brown_yield_per_three_months * brown_hectares * three_month_periods_in_six_months
def taylor_total_yield := taylor_yield_per_month * one_month_periods_in_six_months

-- Prove the total yield.
theorem total_corn_yield_after_six_months :
  johnson_total_yield + smith_total_yield + brown_total_yield + taylor_total_yield = 1530 :=
by
  sorry

end total_corn_yield_after_six_months_l717_717131


namespace sum_of_odd_subsets_l717_717901

open Finset

noncomputable def capacity (X : Finset ℕ) : ℕ :=
  X.sum id

noncomputable def even_subsets_capacity (n : ℕ) : ℕ :=
  (powerset (range (n + 1))).filter (λ s, capacity s % 2 = 0).sum capacity

noncomputable def odd_subsets_capacity (n : ℕ) : ℕ :=
  (powerset (range (n + 1))).filter (λ s, capacity s % 2 = 1).sum capacity

theorem sum_of_odd_subsets (n : ℕ) (h : n ≥ 3) :
  odd_subsets_capacity n = 2^(n-3) * n * (n + 1) := 
sorry

end sum_of_odd_subsets_l717_717901


namespace trapezoid_area_relation_l717_717121

variable {A B C D P : Type}
variable [PartialOrder A] [PartialOrder B]

def trapezoid (a b c d : A) : Prop :=
  a ≤ b

def intersect_at (p : A)(a b : A) : Prop :=
  p = a ∧ p = b

variables (S₁ S₂ S₃ S₄ : B)

theorem trapezoid_area_relation
  (h_trapezoid : trapezoid A B C D)
  (h_intersect : intersect_at P A B)
  (parallel : D = A ∧ C = B)
  (S₁_S := S₁ + S₃)
  (S₂_S := S₂ + S₄) :
  S₁_S ≥ S₂_S :=
sorry

end trapezoid_area_relation_l717_717121


namespace find_a5_l717_717120

def seq (a b : ℕ) : ℕ → ℕ
| 0     := a
| 1     := b
| (n+2) := 3 * seq a b (n+1) - 2 * seq a b n

theorem find_a5 : seq 2 3 5 = 17 := by
  sorry

end find_a5_l717_717120


namespace sum_of_solutions_l717_717760

theorem sum_of_solutions (a b c : ℝ) (h₁ : a = -48) (h₂ : b = 120) (h₃ : c = -75) :
  let sum_of_roots := -b / a in
  sum_of_roots = 5 / 2 :=
by
  simp [h₁, h₂, h₃]
  sorry

end sum_of_solutions_l717_717760


namespace problem_part1_problem_part2_l717_717028

/-
Given the function f(x) = 2√3 sin(x) cos(x) - 2 cos(x)² + 1,
and the triangle ABC with angles A, B, C such that f(C) = 2 and side c = √3,
we need to prove:
1. The set of values of x when f(x) reaches its maximum is { x | x = k * π + π/3, k ∈ ℤ }.
2. The maximum area of triangle ABC is 3√3 / 4.
-/

noncomputable def f (x : ℝ) : ℝ :=
  2 * sqrt 3 * real.sin x * real.cos x - 2 * (real.cos x)^2 + 1

theorem problem_part1 :
  {x : ℝ | ∃ k : ℤ, x = k * real.pi + real.pi / 3} = 
  {x | f(x) = 2 * sqrt 3 * real.sin x * real.cos x - 2 * (real.cos x)^2 + 1 ∧
    ∀ y, f(y) ≤ f(x)} :=
sorry

theorem problem_part2 (C : ℝ) (a b : ℝ) (hC : f C = 2) (hc : b = sqrt 3) :
  ∃ (a b : ℝ), let area := 1/2 * a * b * real.sin C in area = 3 * sqrt 3 / 4 :=
sorry

end problem_part1_problem_part2_l717_717028


namespace combined_savings_after_5_years_l717_717091

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + (r / n)) ^ (n * t)

theorem combined_savings_after_5_years :
  let P1 := 600
  let r1 := 0.10
  let n1 := 12
  let t := 5
  let P2 := 400
  let r2 := 0.08
  let n2 := 4
  compound_interest P1 r1 n1 t + compound_interest P2 r2 n2 t = 1554.998 :=
by
  sorry

end combined_savings_after_5_years_l717_717091


namespace time_first_segment_time_second_segment_time_first_2013_segments_l717_717662

-- Define the distance and speed for the first segment
def first_segment_distance : ℝ := 1 / 2
def first_segment_speed : ℝ := 3

-- Define the distance and speed for the second segment
def second_segment_distance : ℝ := 1 / 3
def second_segment_speed : ℝ := 4

-- Define the general distance and speed for the nth segment
def nth_segment_distance (n : ℕ) : ℝ := 1 / (n + 1)
def nth_segment_speed (n : ℕ) : ℝ := n + 2

-- Prove the time for the first segment
theorem time_first_segment : 
  first_segment_distance / first_segment_speed = 1 / 6 := by
  sorry

-- Prove the time for the second segment
theorem time_second_segment : 
  second_segment_distance / second_segment_speed = 1 / 12 := by
  sorry

-- Prove the total time for the first 2013 segments
theorem time_first_2013_segments : 
  (∑ n in Finset.range 2013, nth_segment_distance n / nth_segment_speed n) = 
    (1 / 2 - 1 / 2015) := by
  sorry

end time_first_segment_time_second_segment_time_first_2013_segments_l717_717662


namespace total_people_wearing_hats_l717_717546

variable (total_adults : ℕ) (total_children : ℕ)
variable (half_adults : ℕ) (women : ℕ) (men : ℕ)
variable (women_with_hats : ℕ) (men_with_hats : ℕ)
variable (children_with_hats : ℕ)
variable (total_with_hats : ℕ)

-- Given conditions
def conditions : Prop :=
  total_adults = 1800 ∧
  total_children = 200 ∧
  half_adults = total_adults / 2 ∧
  women = half_adults ∧
  men = half_adults ∧
  women_with_hats = (25 * women) / 100 ∧
  men_with_hats = (12 * men) / 100 ∧
  children_with_hats = (10 * total_children) / 100 ∧
  total_with_hats = women_with_hats + men_with_hats + children_with_hats

-- Proof goal
theorem total_people_wearing_hats : conditions total_adults total_children half_adults women men women_with_hats men_with_hats children_with_hats total_with_hats → total_with_hats = 353 :=
by
  intros h
  sorry

end total_people_wearing_hats_l717_717546


namespace soccer_team_points_l717_717192

theorem soccer_team_points 
  (total_games : ℕ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (points_per_win : ℕ) 
  (points_per_draw : ℕ) 
  (points_per_loss : ℕ) 
  (draws : ℕ := total_games - (wins + losses)) : 
  total_games = 20 →
  wins = 14 →
  losses = 2 →
  points_per_win = 3 →
  points_per_draw = 1 →
  points_per_loss = 0 →
  46 = (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) :=
by sorry

end soccer_team_points_l717_717192


namespace solve_for_x_l717_717933

variable (x : ℝ)
axiom h : 3 / 4 + 1 / x = 7 / 8

theorem solve_for_x : x = 8 :=
by
  sorry

end solve_for_x_l717_717933


namespace third_divisor_is_11_l717_717986

theorem third_divisor_is_11 (n : ℕ) (x : ℕ) : 
  n = 200 ∧ (n - 20) % 15 = 0 ∧ (n - 20) % 30 = 0 ∧ (n - 20) % x = 0 ∧ (n - 20) % 60 = 0 → 
  x = 11 :=
by
  sorry

end third_divisor_is_11_l717_717986


namespace min_distance_and_coordinates_l717_717072

theorem min_distance_and_coordinates
  (P : ℝ × ℝ)
  (Hcurve : P.2 = P.1 ^ 2)  -- P lies on the curve y = x^2
  (Hline : ∀ t : ℝ, (1 + sqrt 2 * t) - (sqrt 2 * t) = 1)  -- parametric equation of the line x-y=1
  : (∃ (x y : ℝ), P = (x, y) ∧ abs ((x - y - 1) / sqrt 2) = 3 * sqrt 2 / 8 ∧ (x, y) = (1/2, 1/4)) := 
sorry

end min_distance_and_coordinates_l717_717072


namespace find_a7_l717_717710

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem find_a7 (a : ℕ → ℝ) (h_geom : geometric_sequence a)
  (h3 : a 3 = 1)
  (h_det : a 6 * a 8 - 8 * 8 = 0) :
  a 7 = 8 :=
sorry

end find_a7_l717_717710


namespace option_C_correct_l717_717991

-- Define the base a and natural numbers m and n for exponents
variables {a : ℕ} {m n : ℕ}

-- Lean statement to prove (a^5)^3 = a^(5 * 3)
theorem option_C_correct : (a^5)^3 = a^(5 * 3) := 
by sorry

end option_C_correct_l717_717991


namespace probability_of_selecting_presidents_l717_717236

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

def club_sizes : List ℕ := [6, 9, 10, 11]
def effective_sizes := club_sizes.map (λ n => n - 1)

def prob_of_presidents (n : ℕ) :=
  (binomial (n - 3) 2) / (binomial (n - 1) 4)

def total_probability : ℚ :=
  (1 / 4) * (
    prob_of_presidents 5 +
    prob_of_presidents 8 +
    prob_of_presidents 9 +
    prob_of_presidents 10
  )

theorem probability_of_selecting_presidents :
  total_probability = 7 / 25 :=
sorry

end probability_of_selecting_presidents_l717_717236


namespace number_of_tables_l717_717941

-- Define conditions
def chairs_in_base5 : ℕ := 310  -- chairs in base-5
def chairs_base10 : ℕ := 3 * 5^2 + 1 * 5^1 + 0 * 5^0  -- conversion to base-10
def people_per_table : ℕ := 3

-- The theorem to prove
theorem number_of_tables : chairs_base10 / people_per_table = 26 := by
  -- include the automatic proof here
  sorry

end number_of_tables_l717_717941


namespace relationship_between_a_b_c_l717_717497

def e := Real.exp 1

def a := (1 / e) - Real.log (1 / e)
def b := (1 / (2 * e)) - Real.log (1 / (2 * e))
def c := (2 / e) - Real.log (2 / e)

theorem relationship_between_a_b_c : b > a ∧ a > c := 
sorry

end relationship_between_a_b_c_l717_717497


namespace volunteer_arrangements_ant_probability_binomial_coefficient_a_range_l717_717629

-- Problem 1
theorem volunteer_arrangements : 
  ∃ n: ℕ, n = 240 ∧ (∀ p : Fin 5 → Fin 4, 
    (∀ i j : Fin 5, i ≠ j → p i ≠ p j) ∧ 
    (∀ k : Fin 4, ∃ i : Fin 5, p i = k)) := sorry

-- Problem 2
theorem ant_probability : 
  ∃ p: ℚ, p = 4/9 ∧ 
    (∀ moves: Fin 3 → Bool, 
    (probability_moves_right moves = 2 / 3) ∧ 
    (probability_moves_left moves = 1 / 3)) := sorry

-- Problem 3
theorem binomial_coefficient : 
  ∃ a2: ℚ, a2 = 34 ∧ 
    (let expansion := (x + 1)^3 + (x - 2)^8 in 
    binomial_coeff expansion (x - 1) 2 = a2) := sorry

-- Problem 4
theorem a_range : 
  ∃ a: ℝ, a ≥ 0 ∧ 
    (∀ x1: ℝ, x1 ∈ Ioo 0 2 → 
    ∃ x2: ℝ, x2 ∈ Icc 1 2 ∧ 
    4 * x1 * log x1 - x1^2 + 3 + 
    4 * x1 * x2^2 + 8 * a * x1 * x2 - 18 * x1 ≥ 0) := sorry

end volunteer_arrangements_ant_probability_binomial_coefficient_a_range_l717_717629


namespace area_AOC_l717_717866

variables (ABC : Type) [triangle ABC] 
variables (A B C O : point ABC) 
variables (p a : ℝ) 
variables (α : ℝ) 

noncomputable def area_triangle_AOC := 
  (1 / 2) * a * (p - a) * Real.tan (α / 2)

theorem area_AOC (h_perimeter : 2 * p = perimeter ABC)
                 (h_angle_ABC : angle B A C = α)
                 (h_side_AC : dist A C = a)
                 (h_incircle : is_incircle ABC O) :
  area A O C = area_triangle_AOC ABC A B C O p a α :=
  sorry

end area_AOC_l717_717866


namespace hyperbola_eccentricity_l717_717414

theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : (dist_point_to_line (-real.sqrt 2) 0 (real.sqrt (a^2 + b^2)) (b/a) = real.sqrt 5 / 5)) :
  (eccentricity a b = real.sqrt 10 / 3) :=
sorry

-- Definitions for distance and eccentricity used in the theorem
noncomputable def dist_point_to_line (x y : ℝ) (sqrt_ab : ℝ) (ratio_ba : ℝ) : ℝ :=
  abs (ratio_ba * x - y) / real.sqrt (ratio_ba^2 + 1)

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  real.sqrt (1 + (b^2 / a^2))

end hyperbola_eccentricity_l717_717414


namespace time_after_seconds_l717_717129

def initial_time : Nat × Nat × Nat := (4, 45, 0)
def seconds_to_add : Nat := 12345
def final_time : Nat × Nat × Nat := (8, 30, 45)

theorem time_after_seconds (h : initial_time = (4, 45, 0) ∧ seconds_to_add = 12345) : 
  ∃ (h' : Nat × Nat × Nat), h' = final_time := by
  sorry

end time_after_seconds_l717_717129


namespace parametric_rep_two_rays_l717_717568

open Real

-- Define the parametric equations
def parametric_eqns (t : ℝ) : ℝ × ℝ := (t + 1 / t, 2)

-- Define the conditions on x based on the parameter t (excluding t = 0)
theorem parametric_rep_two_rays (t : ℝ) (ht : t ≠ 0) :
  parametric_eqns t = if t > 0 then (t + 1 / t, 2) else (t + 1 / t, 2) ∧
  ((t > 0 → t + 1 / t ≥ 2) ∧ (t < 0 → t + 1 / t ≤ -2)) := by
  sorry

end parametric_rep_two_rays_l717_717568


namespace smallest_circle_radius_l717_717975

-- Define the problem setup and known conditions
variable (A B C : Type) [real A B C]

/-- Define the radius of circles around A, B, and C -/
variable (r_A r_B r_C : ℝ)

-- Given conditions from the problem
variable (side_length : ℝ)
variable (hexagon_vertices : A ≃ B ≃ C)

constant hexagon_condition : regular_hexagon_with_side_length vertex_length

-- Define the problem statement
theorem smallest_circle_radius :
  regular_hexagon_with_side_length side_length →
  circle_touch_externally hexagon_vertices side_length →
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) →
  (r_A + r_B = side_length) →
  (r_A + r_C = side_length * sqrt 3) →
  (r_B + r_C = side_length) →
  r smallest = (2 - sqrt 3) := 
sorry 

end smallest_circle_radius_l717_717975


namespace f_shift_l717_717499

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the main theorem
theorem f_shift (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h - 4) :=
by
  sorry

end f_shift_l717_717499


namespace smallest_positive_integer_l717_717987

theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, 3003 * m + 66666 * n = 3 :=
by
  sorry

end smallest_positive_integer_l717_717987


namespace difference_between_second_and_third_levels_l717_717297

def total_parking_spots : ℕ := 400
def first_level_open_spots : ℕ := 58
def second_level_open_spots : ℕ := first_level_open_spots + 2
def fourth_level_open_spots : ℕ := 31
def total_full_spots : ℕ := 186

def total_open_spots : ℕ := total_parking_spots - total_full_spots

def third_level_open_spots : ℕ := 
  total_open_spots - (first_level_open_spots + second_level_open_spots + fourth_level_open_spots)

def difference_open_spots : ℕ := third_level_open_spots - second_level_open_spots

theorem difference_between_second_and_third_levels : difference_open_spots = 5 :=
sorry

end difference_between_second_and_third_levels_l717_717297


namespace time_to_change_tires_l717_717482

variable (wash_time oil_change_time tire_change_time : ℕ)
variable (cars_washed cars_oiled sets_of_tires : ℕ)
variable (total_time_min : ℕ)

-- Define the problem conditions
def time_to_wash_car := 10
def time_to_change_oil := 15
def mike_washes := 9
def mike_changes_oil := 6
def mike_changes_tires := 2
def total_work_time_hours := 4
def total_work_time_min := total_work_time_hours * 60

-- Question: How many minutes does it take to change a set of tires?
theorem time_to_change_tires :
  let remaining_time := total_work_time_min - (mike_washes * time_to_wash_car + mike_changes_oil * time_to_change_oil) in
  remaining_time / mike_changes_tires = 30 := 
by {
  -- Proof steps are omitted
  sorry
}

end time_to_change_tires_l717_717482


namespace discount_savings_l717_717961

theorem discount_savings
  (p : ℝ) (d1 d2 : ℝ)
  (h₀ : p = 50)
  (h₁ : d1 = 5)
  (h₂ : d2 = 0.1) :
  let price1 := (p - d1) * (1 - d2)
  let price2 := (p * (1 - d2)) - d1
  in price1 - price2 = 0.50 :=
by
  sorry

end discount_savings_l717_717961


namespace problem1_problem2_problem3_l717_717699

-- Problem 1
theorem problem1 (a : ℝ) (φ : ℝ → ℝ) (m n : ℝ) 
  (hφ : ∀ x, m ≤ x ∧ x ≤ n → φ x = (x + a) / (Real.exp x))
  (hF : ∃ r, m < r ∧ r < n ∧ 
    (∀ x, m ≤ x ∧ x ≤ r → φ(x) ≤ φ(x + 1)) ∧
    (∀ x, r ≤ x ∧ x ≤ n ∧ x < n - 1 → φ(x) ≥ φ(x + 1))) : 
  a ∈ Ioo (-1 : ℝ) 0 :=
  sorry

-- Problem 2
theorem problem2 (p : ℝ) (φ : ℝ → ℝ) (h_p : p > 0) (m n : ℝ) 
  (hφ : ∀ x, m ≤ x ∧ x ≤ n → φ x = p * x - (x ^ 2 / 2 + x ^ 3 / 3 + x ^ 4 / 4 + p * x ^ 5 / 5))
  (hF : ∃ r, m < r ∧ r < n ∧ 
    (∀ x, m ≤ x ∧ x ≤ r → φ(x) ≤ φ(x + 1)) ∧
    (∀ x, r ≤ x ∧ x ≤ n ∧ x < n - 1 → φ(x) ≥ φ(x + 1))) : 
  True :=
  sorry

-- Problem 3
theorem problem3 (t : ℝ) (φ : ℝ → ℝ) (m n : ℝ) 
  (hφ : ∀ x, m ≤ x ∧ x ≤ n → φ x = (x^2 - x)*(x^2 - x + t))
  (hF : ∃ r, m < r ∧ r < n ∧ 
    (∀ x, m ≤ x ∧ x ≤ r → φ(x) ≤ φ(x + 1)) ∧
    (∀ x, r ≤ x ∧ x ≤ n ∧ x < n - 1 → φ(x) ≥ φ(x + 1))) : 
  t ∈ Iio (1 / 2 : ℝ) :=
  sorry

end problem1_problem2_problem3_l717_717699


namespace equal_areas_of_triangles_l717_717343

-- Define the points A, B, C and the triangle ABC
variables (A B C D E F G H I : Type) [point A] [point B] [point C] [point D] [point E] [point F] [point G] [point H] [point I]

-- Given conditions:
-- D, E, F are points on the sides BC, AC, AB of triangle ABC respectively
-- AD, BE and CF concur at point G
-- The line through G parallel to BC intersects DF at H and DE at I

def points_on_sides (D : point) (E : point) (F : point) (ABC : triangle) :=
  on_side D B C ∧ on_side E A C ∧ on_side F A B

def concurrent_cevians (A : point) (B : point) (C : point) (G : point) (D : point) (E : point) (F : point) :=
  concurrent_lines (line_through A D) (line_through B E) (line_through C F)

def parallel_intersection (G : point) (BC : line) (DF : line) (DE : line) (H : point) (I : point) :=
  parallel (line_through G BC) BC ∧ intersect_line (line_through G BC) DF = H ∧ intersect_line (line_through G BC) DE = I

-- The goal to prove:
-- The areas of triangles AHG and AIG are equal
theorem equal_areas_of_triangles
  (A B C D E F G H I : point) (ABC : triangle)
  (h1 : points_on_sides D E F ABC)
  (h2 : concurrent_cevians A B C G D E F)
  (h3 : parallel_intersection G (line_through B C) (line_through D F) (line_through D E) H I) :
  area_of_triangle A H G = area_of_triangle A I G :=
sorry

end equal_areas_of_triangles_l717_717343


namespace union_M_N_intersection_complementM_N_l717_717429

open Set  -- Open the Set namespace for convenient notation.

noncomputable def funcDomain : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
noncomputable def setN : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}
noncomputable def complementFuncDomain : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 2}

theorem union_M_N :
  (funcDomain ∪ setN) = {x : ℝ | -1 ≤ x ∧ x < 3} :=
by
  sorry

theorem intersection_complementM_N :
  (complementFuncDomain ∩ setN) = {x : ℝ | 2 ≤ x ∧ x < 3} :=
by
  sorry

end union_M_N_intersection_complementM_N_l717_717429


namespace man_completion_time_l717_717618

theorem man_completion_time (w_time : ℕ) (efficiency_increase : ℚ) (m_time : ℕ) :
  w_time = 40 → efficiency_increase = 1.25 → m_time = (w_time : ℚ) / efficiency_increase → m_time = 32 :=
by
  sorry

end man_completion_time_l717_717618


namespace find_n_l717_717368

theorem find_n (n : ℕ) (h : n > 2) : n = 7 ↔ n! ∣ ∏ i in finset.filter (λ p, p.1 < p.2) (finset.sigma (finset.range n.succ) finset.univ), (i.1 + i.2) := by
  sorry

end find_n_l717_717368


namespace angle_of_inclination_of_line_l717_717107

theorem angle_of_inclination_of_line : 
  ∃ (α : ℝ), 0 ≤ α ∧ α < π ∧ tan α = -real.sqrt 3 ∧ α = 2 * real.pi / 3 :=
by
  sorry

end angle_of_inclination_of_line_l717_717107


namespace batsman_average_excluding_highest_and_lowest_l717_717940

theorem batsman_average_excluding_highest_and_lowest (average : ℝ) (innings : ℕ) (highest_score : ℝ) (score_difference : ℝ) :
  average = 63 →
  innings = 46 →
  highest_score = 248 →
  score_difference = 150 →
  (average * innings - highest_score - (highest_score - score_difference)) / (innings - 2) = 58 :=
by
  intros h_average h_innings h_highest h_difference
  simp [h_average, h_innings, h_highest, h_difference]
  -- Here the detailed steps from the solution would come in to verify the simplification
  sorry

end batsman_average_excluding_highest_and_lowest_l717_717940


namespace marbles_shared_equally_l717_717976

def initial_marbles_Wolfgang : ℕ := 16
def additional_fraction_Ludo : ℚ := 1/4
def fraction_Michael : ℚ := 2/3

theorem marbles_shared_equally :
  let marbles_Wolfgang := initial_marbles_Wolfgang
  let additional_marbles_Ludo := additional_fraction_Ludo * initial_marbles_Wolfgang
  let marbles_Ludo := initial_marbles_Wolfgang + additional_marbles_Ludo
  let marbles_Wolfgang_Ludo := marbles_Wolfgang + marbles_Ludo
  let marbles_Michael := fraction_Michael * marbles_Wolfgang_Ludo
  let total_marbles := marbles_Wolfgang + marbles_Ludo + marbles_Michael
  let marbles_each := total_marbles / 3
  marbles_each = 20 :=
by
  sorry

end marbles_shared_equally_l717_717976


namespace fruit_cost_l717_717658

theorem fruit_cost:
  let strawberry_cost := 2.20
  let cherry_cost := 6 * strawberry_cost
  let blueberry_cost := cherry_cost / 2
  let strawberries_count := 3
  let cherries_count := 4.5
  let blueberries_count := 6.2
  let total_cost := (strawberries_count * strawberry_cost) + (cherries_count * cherry_cost) + (blueberries_count * blueberry_cost)
  total_cost = 106.92 :=
by
  sorry

end fruit_cost_l717_717658


namespace trajectory_of_M_exists_fixed_point_N_l717_717910

-- 1. The equation of the trajectory of point M
theorem trajectory_of_M (P : ℝ × ℝ) (M : ℝ × ℝ) (hP_circle : P.1^2 + P.2^2 = 4) (hM_on_PD : ∃ D : ℝ × ℝ, D = (P.1, 0) ∧ ∃ l : ℝ, M = (P.1, l * P.2) ∧ (1 / l = √2)) :
  M.1^2 / 4 + M.2^2 / 2 = 1 ∧ M.1 ≠ 2 ∧ M.1 ≠ -2 :=
sorry

-- 2. Existence of fixed point N(-7/4, 0)
theorem exists_fixed_point_N (C : ℝ × ℝ) (A B : ℝ × ℝ) (M_trajectory : ∀ M : ℝ × ℝ, M.1^2 / 4 + M.2^2 / 2 = 1 ∧ M.1 ≠ 2 ∧ M.1 ≠ -2) (hC : C = (-1, 0)) :
  ∃ N : ℝ × ℝ, N = (-7 / 4, 0) ∧ ∀ A B : ℝ × ℝ, (A ≠ B ∧ A.1^2 / 4 + A.2^2 / 2 = 1 ∧ A.1 ≠ 2 ∧ A.1 ≠ -2 ∧ B.1^2 / 4 + B.2^2 / 2 = 1 ∧ B.1 ≠ 2 ∧ B.1 ≠ -2 ∧ ∃ k : ℝ, A.2 = k * (A.1 + 1) ∧ B.2 = k * (B.1 + 1)) →  (A.1 - N.1, A.2) • (B.1 - N.1, B.2) = (-15 / 16) :=
sorry

end trajectory_of_M_exists_fixed_point_N_l717_717910


namespace probability_no_defective_pens_l717_717841

theorem probability_no_defective_pens :
  (∃ n d : ℕ, n = 16 ∧ d = 3 ∧ 
  let nd := n - d in 
  let p1 := (nd : ℚ) / n in 
  let p2 := (nd - 1 : ℚ) / (n - 1) in
  p1 * p2 = 13 / 20) :=
begin
  sorry
end

end probability_no_defective_pens_l717_717841


namespace least_number_l717_717373

theorem least_number (n : ℕ) (h1 : n % 31 = 3) (h2 : n % 9 = 3) : n = 282 :=
sorry

end least_number_l717_717373


namespace complement_intersect_l717_717436

noncomputable def U := set.univ
def M := {y : ℝ | ∃ x : ℝ, y = 2 ^ x + 1}
def N := {x : ℝ | ∃ y : ℝ, y = real.log (3 - x)}
def C_U_M := {y : ℝ | y < 1}

theorem complement_intersect :
  (C_U_M ∩ {x : ℝ | x < 3}) = {x : ℝ | x < 1} :=
by
  sorry

end complement_intersect_l717_717436


namespace arith_sequence_a5_l717_717776

theorem arith_sequence_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h₁ : S 16 = 12) (h₂ : a 2 = 5)
  (sum_formula : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) : 
  a 5 = -1 := sorry

end arith_sequence_a5_l717_717776


namespace sum_of_variables_l717_717444

theorem sum_of_variables (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) (ha : a = 2 * Real.sqrt 6) (hb : b = 3 * Real.sqrt 6) (hc : c = 6 * Real.sqrt 6) : 
  a + b + c = 11 * Real.sqrt 6 :=
by
  sorry

end sum_of_variables_l717_717444


namespace discriminant_quadratic_eq_l717_717945

theorem discriminant_quadratic_eq : 
  let a := 1
  let b := -7
  let c := 4
  let Δ := b^2 - 4 * a * c
  Δ = 33 :=
by
  let a := 1
  let b := -7
  let c := 4
  let Δ := b^2 - 4 * a * c
  exact sorry

end discriminant_quadratic_eq_l717_717945


namespace train_speed_in_kmph_l717_717306

-- Definitions for the conditions
def length_of_train : ℝ := 200.016 / 1000  -- in kilometers
def time_to_cross_post : ℝ := 18 / 3600  -- in hours

-- Statement to prove
theorem train_speed_in_kmph : length_of_train / time_to_cross_post = 40.0032 :=
by
  -- The proof steps are omitted
  sorry

end train_speed_in_kmph_l717_717306


namespace inequality_solution_l717_717180

theorem inequality_solution (x : ℝ) :
  (\frac{9 * x^2 + 18 * x - 60}{(3 * x - 4) * (x + 5)} < 2) ↔ 
  (x ∈ Set.Ioo (-5 / 3) (4 / 3) ∪ Set.Ioi 4) :=
by
  sorry

end inequality_solution_l717_717180


namespace min_people_with_all_luxuries_l717_717853

theorem min_people_with_all_luxuries : 
  ∀ (total_people : ℕ) 
    (refrigerator_percent television_percent computer_percent air_conditioner_percent : ℕ),
  refrigerator_percent = 75 →
  television_percent = 90 →
  computer_percent = 85 →
  air_conditioner_percent = 75 →
  ∃ (min_people_with_all : ℕ), 
    min_people_with_all = total_people * refrigerator_percent / 100 :=
begin
  intros,
  use total_people * 75 / 100,
  sorry
end

end min_people_with_all_luxuries_l717_717853


namespace min_value_of_z_l717_717799

noncomputable def minimum_value (z : ℂ) : ℝ :=
  |z - 1|

theorem min_value_of_z (z : ℂ) : 
  (|((z^2 + 1) / (z + complex.i))| + |((z^2 + 4 * complex.i - 3) / (z - complex.i + 2))| = 4) →
  minimum_value z = real.sqrt 2 :=
sorry

end min_value_of_z_l717_717799


namespace no_counterexample_for_n_l717_717149

theorem no_counterexample_for_n (n : ℕ) (hn : n = 5 ∨ n = 11 ∨ n = 17 ∨ n = 29 ∨ n = 41) :
  ¬ (Prime n ∧ Prime (n + 4) ∧ ¬ Prime (n + 2)) :=
by
  intro h
  cases hn <;> 
  { cases h with h1 h2,
    contradiction
    sorry }

end no_counterexample_for_n_l717_717149


namespace inscribable_quadrilateral_l717_717281

theorem inscribable_quadrilateral (a R : ℝ) (O : ℝ × ℝ) (Q : list (ℝ × ℝ)) (hchord_length : ∀ side ∈ Q, (dist (side.1) O = a ∧ dist (side.2) O = a)) : 
  ∃ I : ℝ × ℝ, ∀ side ∈ Q, (dist I side.1 = R ∧ dist I side.2 = R) :=
sorry

end inscribable_quadrilateral_l717_717281


namespace solve_exponential_eq_l717_717014

theorem solve_exponential_eq (x : ℝ) :
  3^(x^2 - 6*x + 5) = 3^(x^2 + 8*x - 3) → x = 4 / 7 :=
by
  sorry

end solve_exponential_eq_l717_717014


namespace jill_present_age_l717_717233

-- Define the main proof problem
theorem jill_present_age (H J : ℕ) (h1 : H + J = 33) (h2 : H - 6 = 2 * (J - 6)) : J = 13 :=
by
  sorry

end jill_present_age_l717_717233


namespace x_range_of_point_on_ellipse_l717_717701

theorem x_range_of_point_on_ellipse
  (P : ℝ → ℝ → ℝ → ℝ)
  (h : ∀ x y, (x^2 / 9 + y^2 / 4 = 1) → (P x y (-√5) + P x y √5 < 0))
  (f1 : ∀ (x : ℝ), (-√5, 0))
  (f2 : ∀ (x : ℝ), (√5, 0)) :
  ∀ x y
  (h_ellipse : x^2 / 9 + y^2 / 4 = 1)
  (h_condition : (x + √5)^2 + y^2 + (x - √5)^2 + y^2 < 20),
  -3 * sqrt 5 / 5 < x ∧ x < 3 * sqrt 5 / 5 :=
by
  intros
  sorry

end x_range_of_point_on_ellipse_l717_717701


namespace find_n_in_triangle_l717_717123

theorem find_n_in_triangle (n m : ℕ) (h1 : AB = 33) (h2 : AC = 21) (h3 : n ≥ 7) :
  (∃ D E, D ∈ AB ∧ E ∈ AC ∧ AD = DE ∧ DE = EC ∧ (n = 11 ∨ n = 21)) :=
sorry

end find_n_in_triangle_l717_717123


namespace original_average_is_24_l717_717198

theorem original_average_is_24
  (A : ℝ)
  (h1 : ∀ n : ℕ, n = 7 → 35 * A = 7 * 120) :
  A = 24 :=
by
  sorry

end original_average_is_24_l717_717198


namespace no_integer_roots_l717_717732

theorem no_integer_roots : ∀ x : ℤ, x^3 - 3 * x^2 - 16 * x + 20 ≠ 0 := by
  intro x
  sorry

end no_integer_roots_l717_717732


namespace sum_of_x_values_l717_717990

theorem sum_of_x_values : 
  (∑ x in {x : ℝ | (x^3 - 3*x^2 - 9*x)/(x + 3) = 2 ∧ 3*x - 9 = 0}, x) = 3 :=
by
  sorry

end sum_of_x_values_l717_717990


namespace mack_total_pages_l717_717519

def pages_written (minutes: ℕ) (rate: ℕ) : ℕ := minutes / rate

theorem mack_total_pages :
  let monday_pages := pages_written 60 30,
      tuesday_pages := pages_written 45 15,
      wednesday_pages := 5,
      thursday_pages_first := pages_written 30 10,
      thursday_pages_second := pages_written 60 20,
      thursday_pages := thursday_pages_first + thursday_pages_second,
      total_pages := monday_pages + tuesday_pages + wednesday_pages + thursday_pages
  in total_pages = 16 :=
by
  have monday_pages_eq : monday_pages = 2 := rfl
  have tuesday_pages_eq : tuesday_pages = 3 := rfl
  have wednesday_pages_eq : wednesday_pages = 5 := rfl
  have thursday_pages_first_eq : thursday_pages_first = 3 := rfl
  have thursday_pages_second_eq : thursday_pages_second = 3 := rfl
  have thursday_pages_eq : thursday_pages = 6 := by
    rw [thursday_pages_first_eq, thursday_pages_second_eq]
  have total_pages_eq : total_pages = 16 := by
    rw [monday_pages_eq, tuesday_pages_eq, wednesday_pages_eq, thursday_pages_eq]
  exact total_pages_eq

end mack_total_pages_l717_717519


namespace number_of_ns_with_prime_g_l717_717138

/-- The function g(n) which returns the sum of the positive integer divisors of n --/
def g (n : ℕ) : ℕ :=
List.sum (List.filter (λ d, n % d = 0) (List.range (n + 1)))

/-- A function to check if a number is prime --/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Counting the number of n in the range 1 ≤ n ≤ 30 for which g(n) is prime --/
theorem number_of_ns_with_prime_g : 
  (Finset.filter (λ n, is_prime (g n)) (Finset.range (30 + 1))).card = 5 := 
by
  sorry

end number_of_ns_with_prime_g_l717_717138


namespace geometric_sequence_sum_l717_717070

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

noncomputable def sum_geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum 
  (a₁ : ℝ) (q : ℝ) 
  (h_q : q = 1 / 2) 
  (h_a₂ : geometric_sequence a₁ q 2 = 2) : 
  sum_geometric_sequence a₁ q 6 = 63 / 8 :=
by
  -- The proof is skipped here
  sorry

end geometric_sequence_sum_l717_717070


namespace num_divisors_of_20_l717_717819

theorem num_divisors_of_20 : 
  {n : ℤ | 20 % n = 0}.to_finset.card = 12 :=
by sorry

end num_divisors_of_20_l717_717819


namespace smallest_pos_multiple_6_15_is_30_l717_717749

theorem smallest_pos_multiple_6_15_is_30 :
  ∃ b > 0, 6 ∣ b ∧ 15 ∣ b ∧ (∀ b', b' > 0 ∧ b' < b → ¬ (6 ∣ b' ∧ 15 ∣ b')) :=
by
  -- Implementation to be done
  sorry

end smallest_pos_multiple_6_15_is_30_l717_717749


namespace cecile_apples_l717_717724

theorem cecile_apples (C D : ℕ) (h1 : D = C + 20) (h2 : C + D = 50) : C = 15 :=
by
  -- Proof steps would go here
  sorry

end cecile_apples_l717_717724


namespace inverse_89_mod_90_l717_717361

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  -- Mathematical proof is skipped
  sorry

end inverse_89_mod_90_l717_717361


namespace width_of_crate_l717_717640

theorem width_of_crate
  (r : ℝ) (h : ℝ) (w : ℝ)
  (h_crate : h = 6 ∨ h = 10 ∨ w = 6 ∨ w = 10)
  (r_tank : r = 4)
  (height_longest_crate : h > w)
  (maximize_volume : ∃ d : ℝ, d = 2 * r ∧ w = d) :
  w = 8 := 
sorry

end width_of_crate_l717_717640


namespace find_m_l717_717228

-- Definition of the conditions
def condition1 (m : ℝ) := m^2 - 3 * m + 3 = 1
def condition2 (m : ℝ) := 2^m = 4

-- Definition of the proof problem
theorem find_m (m : ℝ) (h1 : condition1 m) (h2 : condition2 m) : m = 2 := sorry

end find_m_l717_717228


namespace prove_expression_value_l717_717783

theorem prove_expression_value (x y : ℝ) (h1 : 4 * x + y = 18) (h2 : x + 4 * y = 20) :
  20 * x^2 + 16 * x * y + 20 * y^2 = 724 :=
sorry

end prove_expression_value_l717_717783


namespace eval_expr_l717_717727

theorem eval_expr : (Real.cbrt (8 + 3 * Real.sqrt 21) + Real.cbrt (8 - 3 * Real.sqrt 21) = 1) :=
by
  sorry

end eval_expr_l717_717727


namespace units_digit_17_pow_2007_l717_717597

theorem units_digit_17_pow_2007 :
  (17 ^ 2007) % 10 = 3 := 
sorry

end units_digit_17_pow_2007_l717_717597


namespace total_weight_left_after_discarding_damaged_l717_717512

-- Define the given conversion factors
def gram_to_pounds : ℝ := 0.00220462
def ounce_to_pounds : ℝ := 0.0625
def liter_to_pounds : ℝ := 2.20462

-- Define the initial quantities of the items Melanie bought
def brie_ounces : ℝ := 8
def bread_pounds : ℝ := 1
def tomatoes_pounds : ℝ := 1
def zucchini_pounds : ℝ := 2
def chicken_pounds : ℝ := 1.5
def raspberries_ounces : ℝ := 8
def blueberries_ounces : ℝ := 8
def asparagus_grams : ℝ := 500
def oranges_kilograms : ℝ := 1
def olive_oil_milliliters : ℝ := 750

-- Define the percentage of food that is damaged
def damaged_percentage : ℝ := 0.15

-- Define the theorem to prove the total weight of food after discarding damaged portion
theorem total_weight_left_after_discarding_damaged :
  let brie_pounds := brie_ounces * ounce_to_pounds,
      raspberries_pounds := raspberries_ounces * ounce_to_pounds,
      blueberries_pounds := blueberries_ounces * ounce_to_pounds,
      asparagus_pounds := asparagus_grams * gram_to_pounds,
      oranges_pounds := oranges_kilograms * liter_to_pounds,
      olive_oil_pounds := (olive_oil_milliliters / 1000) * liter_to_pounds,
      total_weight := brie_pounds + bread_pounds + tomatoes_pounds + zucchini_pounds +
                      chicken_pounds + raspberries_pounds + blueberries_pounds +
                      asparagus_pounds + oranges_pounds + olive_oil_pounds,
      damaged_weight := total_weight * damaged_percentage,
      food_left_weight := total_weight - damaged_weight
  in 
  food_left_weight = 10.16633575 := by
  sorry

end total_weight_left_after_discarding_damaged_l717_717512


namespace doritos_ratio_l717_717906

noncomputable def bags_of_chips : ℕ := 80
noncomputable def bags_per_pile : ℕ := 5
noncomputable def piles : ℕ := 4

theorem doritos_ratio (D T : ℕ) (h1 : T = bags_of_chips)
  (h2 : D = piles * bags_per_pile) :
  (D : ℚ) / T = 1 / 4 := by
  sorry

end doritos_ratio_l717_717906


namespace black_cards_taken_out_l717_717541

theorem black_cards_taken_out (initial_black : ℕ) (remaining_black : ℕ) (total_cards : ℕ) (black_cards_per_deck : ℕ) :
  total_cards = 52 → black_cards_per_deck = 26 →
  initial_black = black_cards_per_deck → remaining_black = 22 →
  initial_black - remaining_black = 4 := by
  intros
  sorry

end black_cards_taken_out_l717_717541


namespace smallest_common_multiple_l717_717743

theorem smallest_common_multiple (b : ℕ) (hb : b > 0) (h1 : b % 6 = 0) (h2 : b % 15 = 0) :
    b = 30 :=
sorry

end smallest_common_multiple_l717_717743


namespace unique_solution_exists_l717_717349

theorem unique_solution_exists (a b c d : ℝ) :
(ab + c + d = 3) →
(bc + d + a = 5) →
(cd + a + b = 2) →
(da + b + c = 6) →
(a = 2 ∧ b = 0 ∧ c = 0 ∧ d = 3) :=
begin
  intros h1 h2 h3 h4,
  -- proof needs to be filled in here
  sorry
end

end unique_solution_exists_l717_717349


namespace identity_1_over_n_n_plus_1_sum_series_1_over_k_k_plus_1_sum_series_1_over_3k_minus_2_3k_plus_1_l717_717516

-- Question 1: Prove the given identity for 1/(n(n+1))
theorem identity_1_over_n_n_plus_1 (n : ℕ) (hn : n ≠ 0) : 
  (1 : ℚ) / (n * (n + 1)) = (1 : ℚ) / n - (1 : ℚ) / (n + 1) :=
by
  sorry

-- Question 2: Prove the sum of series 1/k(k+1) from k=1 to k=2021
theorem sum_series_1_over_k_k_plus_1 : 
  (Finset.range 2021).sum (λ k => (1 : ℚ) / (k+1) / (k+2)) = 2021 / 2022 :=
by
  sorry

-- Question 3: Prove the sum of series 1/(3k-2)(3k+1) from k=1 to k=673
theorem sum_series_1_over_3k_minus_2_3k_plus_1 : 
  (Finset.range 673).sum (λ k => (1 : ℚ) / ((3 * k + 1 - 2) * (3 * k + 1))) = 674 / 2023 :=
by
  sorry

end identity_1_over_n_n_plus_1_sum_series_1_over_k_k_plus_1_sum_series_1_over_3k_minus_2_3k_plus_1_l717_717516


namespace circle_area_l717_717210

-- Given conditions
variables {BD AC : ℝ} (BD_pos : BD = 6) (AC_pos : AC = 12)
variables {R : ℝ} (R_pos : R = 15 / 2)

-- Prove that the area of the circles is \(\frac{225}{4}\pi\)
theorem circle_area (BD_pos : BD = 6) (AC_pos : AC = 12) (R : ℝ) (R_pos : R = 15 / 2) : 
        ∃ S, S = (225 / 4) * Real.pi := 
by sorry

end circle_area_l717_717210


namespace rhombus_inscribed_circle_radius_correct_l717_717690

noncomputable def rhombus_inscribed_circle_radius (d1 d2 : ℝ) (area : ℝ) : ℝ :=
  let a := (Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)) in
  area / (2 * a)

theorem rhombus_inscribed_circle_radius_correct :
  rhombus_inscribed_circle_radius 8 30 120 = 60 / Real.sqrt 241 := 
by sorry

end rhombus_inscribed_circle_radius_correct_l717_717690


namespace students_both_courses_l717_717842

-- Definitions from conditions
def total_students : ℕ := 87
def students_french : ℕ := 41
def students_german : ℕ := 22
def students_neither : ℕ := 33

-- The statement we need to prove
theorem students_both_courses : (students_french + students_german - 9 + students_neither = total_students) → (9 = 96 - total_students) :=
by
  -- The proof would go here, but we leave it as sorry for now
  sorry

end students_both_courses_l717_717842


namespace part_a_part_b_l717_717909

-- Definition of the problem's conditions
structure Cube (V : Type) :=
  (vertices : Set V)
  (edges : Set (Set V))
  (midpoints : Set V)
  (condition : ∀ e ∈ edges, ∃ m ∈ midpoints, m = midpoint_of e)

-- Proof problem definitions
variables (V : Type) [Fintype V]
variables (cube : Cube V)
variables {Point : Type} [Fintype Point]

-- Part (a)
theorem part_a : ¬ ∀ S : Set Point, S ⊆ cube.midpoints ∧ (Set.card S = 6) → ∃ c : Point, ∀ p ∈ S, distance c p = k :=
sorry

-- Part (b)
theorem part_b : ∀ S : Set Point, S ⊆ cube.midpoints ∧ (Set.card S = 7) → ∃ c : Point, ∀ p ∈ S, distance c p = k :=
sorry

end part_a_part_b_l717_717909


namespace cody_books_reading_l717_717693

theorem cody_books_reading :
  ∀ (total_books first_week_books second_week_books subsequent_week_books : ℕ),
    total_books = 54 →
    first_week_books = 6 →
    second_week_books = 3 →
    subsequent_week_books = 9 →
    (2 + (total_books - (first_week_books + second_week_books)) / subsequent_week_books) = 7 :=
by
  -- Using sorry to mark the proof as incomplete.
  sorry

end cody_books_reading_l717_717693


namespace radius_of_given_circle_is_eight_l717_717051

noncomputable def radius_of_circle (diameter : ℝ) : ℝ := diameter / 2

theorem radius_of_given_circle_is_eight :
  radius_of_circle 16 = 8 :=
by
  sorry

end radius_of_given_circle_is_eight_l717_717051


namespace simplify_expression_and_evaluate_at_zero_l717_717539

theorem simplify_expression_and_evaluate_at_zero :
  ((2 * (0 : ℝ) - 1) / (0 + 1) - 0 + 1) / ((0 - 2) / ((0 ^ 2) + 2 * 0 + 1)) = 0 :=
by
  -- proof omitted
  sorry

end simplify_expression_and_evaluate_at_zero_l717_717539


namespace mass_increase_l717_717714

theorem mass_increase (ρ₁ ρ₂ m₁ m₂ a₁ a₂ : ℝ) (cond1 : ρ₂ = 2 * ρ₁) 
                      (cond2 : a₂ = 2 * a₁) (cond3 : m₁ = ρ₁ * (a₁^3)) 
                      (cond4 : m₂ = ρ₂ * (a₂^3)) : 
                      ((m₂ - m₁) / m₁) * 100 = 1500 := by
  sorry

end mass_increase_l717_717714


namespace find_sum_of_first_six_terms_l717_717392

def sequence (a : ℕ → ℚ) : Prop :=
∀ n, a (n + 1) = 2 * a n - 1

def initial_condition (a : ℕ → ℚ) : Prop :=
a 3 = 2

def sum_of_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
∑ i in Finset.range n, a i

theorem find_sum_of_first_six_terms (a : ℕ → ℚ) 
  (h_seq : sequence a) 
  (h_init : initial_condition a) :
  sum_of_first_n_terms a 6 = 87 / 4 := 
sorry

end find_sum_of_first_six_terms_l717_717392


namespace negation_proposition_l717_717564

theorem negation_proposition : 
  (¬ ∃ x_0 : ℝ, 2 * x_0 - 3 > 1) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) :=
by
  sorry

end negation_proposition_l717_717564


namespace units_digit_of_17_pow_2007_l717_717599

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_2007 : units_digit (17 ^ 2007) = 3 := by
  have h : ∀ n, units_digit (17 ^ n) = units_digit (7 ^ n) := by
    intro n
    sorry  -- Same units digit logic for powers of 17 as for powers of 7.
  have pattern : units_digit (7 ^ 1) = 7 ∧ 
                 units_digit (7 ^ 2) = 9 ∧ 
                 units_digit (7 ^ 3) = 3 ∧ 
                 units_digit (7 ^ 4) = 1 := by
    sorry  -- Units digit pattern for powers of 7.
  have mod_cycle : 2007 % 4 = 3 := by
    sorry  -- Calculation of 2007 mod 4.
  have result : units_digit (7 ^ 2007) = units_digit (7 ^ 3) := by
    rw [← mod_eq_of_lt (by norm_num : 2007 % 4 < 4), mod_cycle]
    exact (and.left (and.right (and.right pattern)))  -- Extract units digit of 7^3 from pattern.
  rw [h]
  exact result

end units_digit_of_17_pow_2007_l717_717599


namespace player_A_elimination_after_third_round_at_least_one_player_passes_all_l717_717641

-- Define probabilities for Player A's success in each round
def P_A1 : ℚ := 4 / 5
def P_A2 : ℚ := 3 / 4
def P_A3 : ℚ := 2 / 3

-- Define probabilities for Player B's success in each round
def P_B1 : ℚ := 2 / 3
def P_B2 : ℚ := 2 / 3
def P_B3 : ℚ := 1 / 2

-- Define theorems
theorem player_A_elimination_after_third_round :
  P_A1 * P_A2 * (1 - P_A3) = 1 / 5 := by
  sorry

theorem at_least_one_player_passes_all :
  1 - ((1 - (P_A1 * P_A2 * P_A3)) * (1 - (P_B1 * P_B2 * P_B3))) = 8 / 15 := by
  sorry


end player_A_elimination_after_third_round_at_least_one_player_passes_all_l717_717641


namespace find_B_period_of_f_l717_717400

-- Defining the angles and vectors
def angles (A B C : ℝ) := A + B + C = π

def m : ℝ × ℝ := (2, -2 * Real.sqrt 3)
def n (B : ℝ) : ℝ × ℝ := (Real.cos B, Real.sin B)

def orthogonal (m n : ℝ × ℝ) : Prop := m.1 * n.1 + m.2 * n.2 = 0

noncomputable def a (x : ℝ) : ℝ × ℝ := (1 + Real.sin (2 * x), Real.cos (2 * x))

noncomputable def f (x B : ℝ) : ℝ := (a x).1 * (n B).1 + (a x).2 * (n B).2

-- Statement for part (1)
theorem find_B (B : ℝ) (h1 : orthogonal m (n B)) : B = π / 6 :=
  sorry

-- Statement for part (2)
theorem period_of_f (x : ℝ) (B : ℝ) (hx : angles π B B) :
  ∃ T > 0, ∀ x, f (x + T) B = f x B :=
sorry

end find_B_period_of_f_l717_717400


namespace tetrahedron_parallel_planes_l717_717312

-- Define points and projections as given in the conditions.
variables {A B C D A1 A2 B1 B2 C1 C2 : Type} 

-- Definitions for projections
variables {proj_BD : A → A1} {proj_DC : A → A2} 
variables {proj_CD : B → B1} {proj_AD : B → B2}
variables {proj_AD' : C → C1} {proj_BD' : C → C2}

-- Main theorem to be proven
theorem tetrahedron_parallel_planes
  (tetrahedron : Tetrahedron A B C D)
  (projA1BD : proj_BD A = A1) (projA2DC : proj_DC A = A2)
  (projB1CD : proj_CD B = B1) (projB2AD : proj_AD B = B2)
  (projC1AD : proj_AD' C = C1) (projC2BD : proj_BD' C = C2)
  (not_right_angle : ∀ angle ∈ dihedralAngles (Tetrahedron A B C D), angle ≠ 90) :
  ∃ (plane : Plane), parallel (A1A2, plane) ∧ parallel (B1B2, plane) ∧ parallel (C1C2, plane) :=
sorry

end tetrahedron_parallel_planes_l717_717312


namespace function_positive_l717_717557

-- Define the function f(x) and its properties
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the differentiability condition
def differentiable (f' : ℝ → ℝ) := ∀ x, has_deriv_at f (f' x) x

-- Given conditions
axiom domain_real : ∀ x : ℝ, x ∈ ℝ
axiom differentiable_f : differentiable f'
axiom inequality_condition : ∀ x : ℝ, 2 * f x + x * f' x > x^2

-- The goal to prove
theorem function_positive : ∀ x : ℝ, f x > 0 := by
  sorry

end function_positive_l717_717557


namespace decreasing_function_in_interval_neg1_1_l717_717313

noncomputable def is_decreasing_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

def f1 (x : ℝ) := 1 / (1 - x)
def f2 (x : ℝ) := Real.cos x
def f3 (x : ℝ) := Real.log (x + 1)
def f4 (x : ℝ) := 2^(-x)

theorem decreasing_function_in_interval_neg1_1 :
  is_decreasing_in_interval f4 (-1) 1 ∧
  ¬ is_decreasing_in_interval f1 (-1) 1 ∧
  ¬ is_decreasing_in_interval f2 (-1) 1 ∧
  ¬ is_decreasing_in_interval f3 (-1) 1 :=
by
  sorry

end decreasing_function_in_interval_neg1_1_l717_717313


namespace range_k_l717_717805

noncomputable def f (k : ℝ) : ℝ → ℝ :=
  λ x, if x > 0 then Real.ln x else k * x

theorem range_k (k : ℝ) :
  (∃ x0 : ℝ, x0 ≠ 0 ∧ f k (-x0) = f k x0) → k ∈ Set.Ici (-1 / Real.exp 1) :=
by
  intros h
  sorry

end range_k_l717_717805


namespace find_special_number_l717_717213

-- Let us define auxilliary functions for checking the divisibility and unique digit usage.
def divisible_by (n d : Nat) : Prop := n % d = 0

def all_unique (lst : List Nat) : Prop := lst.nodup

-- Main theorem statement
theorem find_special_number :
  ∃ abcde, (∃ (digits : List Nat), digits = [1, 2, 3, 4, 5] ∧ 
            all_unique [(abcde / 10000) % 10, (abcde / 1000) % 10, (abcde / 100) % 10, (abcde / 10) % 10, abcde % 10] ∧
            divisible_by ((abcde / 100) % 100) 4 ∧        -- Condition for abc
            divisible_by ((abcde / 10) % 1000) 5 ∧        -- Condition for bcd
            divisible_by (abcde % 1000) 3 ∧               -- Condition for cde
            ∀ d, d ∈ digits ↔ d ∈ [(abcde / 10000) % 10, (abcde / 1000) % 10, (abcde / 100) % 10, (abcde / 10) % 10, abcde % 10]) ∧
  abcde = 12453 := by
  sorry

end find_special_number_l717_717213


namespace smallest_pos_multiple_6_15_is_30_l717_717751

theorem smallest_pos_multiple_6_15_is_30 :
  ∃ b > 0, 6 ∣ b ∧ 15 ∣ b ∧ (∀ b', b' > 0 ∧ b' < b → ¬ (6 ∣ b' ∧ 15 ∣ b')) :=
by
  -- Implementation to be done
  sorry

end smallest_pos_multiple_6_15_is_30_l717_717751


namespace M_on_median_AA1_l717_717126

variables {Point Line : Type}
variables (A B C D B' C' M A1 : Point)
variables (AD AC AB BC B'C' : Line)
variables [geometry : Geometry Point Line]

open Geometry

-- Conditions
axiom angle_bisector_AD : is_angle_bisector AD A B C
axiom perpendicular_DB' : perpendicular_from_point_to_line D B' AC
axiom perpendicular_DC' : perpendicular_from_point_to_line D C' AB
axiom point_M_on_B'C' : point_on_line M B'C'
axiom DM_perpendicular_BC : is_perpendicular DM BC
axiom A1_is_midpoint_BC : is_midpoint A1 B C

-- Question: prove that M lies on the median AA1
theorem M_on_median_AA1 :
  lies_on_median AA1 M :=
sorry

end M_on_median_AA1_l717_717126


namespace trapezoid_area_l717_717862

theorem trapezoid_area (AD BC AC : ℝ) (BD : ℝ) 
  (hAD : AD = 24) 
  (hBC : BC = 8) 
  (hAC : AC = 13) 
  (hBD : BD = 5 * Real.sqrt 17) : 
  (1 / 2 * (AD + BC) * Real.sqrt (AC^2 - (BC + (AD - BC) / 2)^2)) = 80 :=
by
  sorry

end trapezoid_area_l717_717862


namespace smallest_common_multiple_l717_717740

theorem smallest_common_multiple (b : ℕ) (hb : b > 0) (h1 : b % 6 = 0) (h2 : b % 15 = 0) :
    b = 30 :=
sorry

end smallest_common_multiple_l717_717740


namespace decodeMessage_correct_l717_717212

noncomputable def decodedMessage : String :=
  decodeMessage "873146507381"

def condition1 :=
  let groups := divideDigitsIntoGroups (List.range 10)
  nonOverlapping groups

def condition2 (groups : List (List Nat)) :=
  ∀ g ∈ groups, 
    ∀ p ∈ permutations g, usesAllDigitsExactlyOnce p g

def condition3 (groups : List (List Nat)) (encodedMessage : String) :=
  let allNumbers := concatMap permutations groups
  let sortedNumbers := sort allNumbers
  let letterMapping := Map.fromList (sortedNumbers.zip "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ".toList)
  encodedMessage = mapDecodeToMessage "873146507381" letterMapping

theorem decodeMessage_correct :
  let groups := divideDigitsIntoGroups (List.range 10)
  condition1 →
  condition2 groups →
  condition3 groups "873146507381" →
  decodedMessage = "НАУКА" :=
by
  intro h1 h2 h3
  sorry

end decodeMessage_correct_l717_717212


namespace graduates_continued_second_degree_l717_717474

noncomputable theory

def number_of_graduates : ℕ := 73
def graduates_found_job : ℕ := 32
def graduates_did_both : ℕ := 13
def graduates_did_neither : ℕ := 9

theorem graduates_continued_second_degree :
  let total := number_of_graduates in
  let job := graduates_found_job in
  let both := graduates_did_both in
  let neither := graduates_did_neither in
  ∃ s : ℕ, total - neither = 64 ∧ job - both = 19 ∧ 64 - 19 - both = 32 ∧ s = 32 + both ∧ s = 45 :=
begin
  sorry
end

end graduates_continued_second_degree_l717_717474


namespace tim_initial_cans_l717_717238

theorem tim_initial_cans (x : ℕ) 
    (h_taken : Jeff_takes_6_cans : 6)
    (h_buy : Tim_buys_half_left : (x - 6) / 2)
    (h_final : Tim_total_end : x - 6 + (x - 6) / 2 = 24) : x = 22 :=
by
sorry

end tim_initial_cans_l717_717238


namespace unique_n_concat_cubed_fourth_l717_717000

def concat_digits_unique (a b : ℕ) : Prop :=
  let digits := (a.toString ++ b.toString).toList in
  digits.length = 10 ∧ digits.sorted.nodup ∧ digits.sorted = "0123456789".toList

theorem unique_n_concat_cubed_fourth : 
  ∃! (n : ℕ), concat_digits_unique (n^3) (n^4) :=
begin
  existsi 18,
  split,
  { sorry },
  { intros y hy,
    sorry }
end

end unique_n_concat_cubed_fourth_l717_717000


namespace domain_f_monotone_f_inv_l717_717794

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

-- Condition 1: Functional equation
axiom functional_eq (x y : ℝ) : f(x) + f(y) = f((x + y) / (1 + x * y))

-- Condition 2: f has an inverse function
axiom has_inverse (x : ℝ) : f (f_inv x) = x ∧ f_inv (f x) = x

-- Part 1: The domain of f(x) is (-1, 1)
theorem domain_f : ∀ x : ℝ, f_inv x < 1 ∧ f_inv x > -1 := sorry

-- Part 2: Monotonicity of f⁻¹ in the interval (a, b) with 0 < a < b
theorem monotone_f_inv (a b : ℝ) (h0 : 0 < a) (h1 : a < b) 
  (h2 : ∀ x ∈ Ioo a b, 0 < f_inv x) : ∀ x₁ x₂ ∈ Ioo a b, x₁ < x₂ → f_inv x₁ < f_inv x₂ := sorry

end domain_f_monotone_f_inv_l717_717794


namespace cos_angle_F1PF2_eq_3_over_4_l717_717049

noncomputable def hyperbola_equation : Prop := 
  ∃ (x y : ℝ), x^2 - y^2 = 2

noncomputable def foci_distances (PF1 PF2 : ℝ) : Prop :=
  PF1 = 2 * PF2

theorem cos_angle_F1PF2_eq_3_over_4
  (PF1 PF2 : ℝ)
  (h1 : hyperbola_equation)
  (h2 : foci_distances PF1 PF2)
  : cos (angle (F1) P (F2)) = 3 / 4 :=
sorry

end cos_angle_F1PF2_eq_3_over_4_l717_717049


namespace solution_set_of_f_greater_than_2x_plus_4_domain_of_f_value_of_f_at_minus_one_derivative_of_f_greater_than_2_solution_set_correct_l717_717218

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_of_f_greater_than_2x_plus_4 :
  (∀ x : ℝ, f x > 2 * x + 4) := 
begin
  sorry,
end

theorem domain_of_f : 
  ∀ x : ℝ, true := 
begin
  intro x,
  trivial,
end

theorem value_of_f_at_minus_one :
  f (-1) = 2 := 
begin
  sorry,
end

theorem derivative_of_f_greater_than_2 : 
  ∀ x : ℝ, (f' x) > 2 := 
begin
  sorry,
end

theorem solution_set_correct :
  { x : ℝ | f x > 2 * x + 4 } = Ioi (-1) := 
begin
  sorry,
end

end solution_set_of_f_greater_than_2x_plus_4_domain_of_f_value_of_f_at_minus_one_derivative_of_f_greater_than_2_solution_set_correct_l717_717218


namespace evening_temperature_is_correct_l717_717325

-- Define the temperatures at noon and in the evening
def T_noon : ℤ := 3
def T_evening : ℤ := -2

-- State the theorem to prove
theorem evening_temperature_is_correct : T_evening = -2 := by
  sorry

end evening_temperature_is_correct_l717_717325


namespace f_2048_gt_13_over_2_l717_717498

noncomputable def f (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, (((1 : ℚ) / (k + 1 : ℚ))))

theorem f_2048_gt_13_over_2 :
  (f 2 = 3 / 2) ∧
  (f 4 > 2) ∧
  (f 8 > 5 / 2) ∧
  (f 16 > 3) →
  f 2048 > 13 / 2 :=
by
  sorry

end f_2048_gt_13_over_2_l717_717498


namespace find_a_and_b_l717_717300

open Function

theorem find_a_and_b (a b : ℚ) (k : ℚ)  (hA : (6 : ℚ) = k * (-3))
    (hB : (a : ℚ) = k * 2)
    (hC : (-1 : ℚ) = k * b) : 
    a = -4 ∧ b = 1 / 2 :=
by
  sorry

end find_a_and_b_l717_717300


namespace behaviour_on_interval_neg7_neg3_l717_717102

variable (f : ℝ → ℝ)

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

noncomputable def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f (x) < f (y)

noncomputable def maximum_on_interval (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  ∀ x, a ≤ x → x ≤ b → f(x) ≤ m ∧ (∃ y, a ≤ y ∧ y ≤ b ∧ f(y) = m)

-- Given conditions
variable (f_is_odd : is_odd_function f)
variable (f_increasing_3_7 : is_increasing_on_interval f 3 7)
variable (f_maximum_5 : maximum_on_interval f 3 7 5)

-- Required result
theorem behaviour_on_interval_neg7_neg3 :
  is_increasing_on_interval f (-7) (-3) ∧ ( ∀ x, -7 ≤ x → x ≤ -3 → f (x) ≥ -5 ∧ ∃ y, -7 ≤ y ∧ y ≤ -3 ∧ f(y) = -5 ) :=
sorry

end behaviour_on_interval_neg7_neg3_l717_717102


namespace pairs_of_students_l717_717016

theorem pairs_of_students (n : ℕ) (h : n = 12) : (n * (n - 1)) / 2 = 66 :=
by
  rw [h]
  norm_num

end pairs_of_students_l717_717016


namespace hyperbola_focus_l717_717370

theorem hyperbola_focus :
  ∃ (x y : ℝ), 2 * x^2 - y^2 - 8 * x + 4 * y - 4 = 0 ∧ (x, y) = (2 + 2 * Real.sqrt 3, 2) :=
by
  -- The proof would go here
  sorry

end hyperbola_focus_l717_717370


namespace smallest_positive_integer_rel_prime_180_l717_717009

theorem smallest_positive_integer_rel_prime_180 : 
  ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → y ≥ 7 := 
by 
  sorry

end smallest_positive_integer_rel_prime_180_l717_717009


namespace max_value_of_complex_expression_l717_717140

noncomputable def z : ℂ := sorry

theorem max_value_of_complex_expression (h : ∥z∥ = 2) : 
  ∃ (y : ℝ), 
    ∥(z - complex.I)^2 * (z + complex.I)∥ = 5 * real.sqrt 5 := 
begin
  sorry
end

end max_value_of_complex_expression_l717_717140


namespace part1_problem_part2_problem_l717_717042

-- Conditions for Part (1)
def part1_conditions (t : ℝ) (θ : ℝ) : Prop := 
  θ ∈ [0, π] ∧ t^2 - 1 = 0 ∧ 2 * cos θ + 1 = 0 ∧ t > sin θ

theorem part1_problem (t : ℝ) (θ : ℝ) (h : part1_conditions t θ) : t = 1 := 
by sorry

-- Conditions for Part (2)
def part2_conditions (t : ℝ) (θ : ℝ) : Prop := 
  θ ∈ [0, π] ∧ t = sin θ ∧ t^2 - 1 = 2 * cos θ + 1

theorem part2_problem (t : ℝ) (θ : ℝ) (h : part2_conditions t θ) : θ = π := 
by sorry

end part1_problem_part2_problem_l717_717042


namespace nine_points_chords_l717_717513

theorem nine_points_chords : 
  ∀ (points : Finset Point), 
  points.card = 9 → 
  (∀ p1 p2 p3 ∈ points, ¬Collinear ℝ ({p1, p2, p3} : Set Point)) → 
  points.choose 2.card = 36 := 
by
  intro points h1 h2
  sorry

end nine_points_chords_l717_717513


namespace arithmetic_sequence_general_term_and_sum_l717_717774

theorem arithmetic_sequence_general_term_and_sum (d : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h₁ : d > 0)
  (h₂ : a 3 * a 6 = 55)
  (h₃ : a 2 + a 7 = 16)
  (h₄ : ∀ n, b n = 1 / (a n * a (n + 1)))
  (h₅ : ∀ n, T n = (finset.range n).sum (λ i, b (i + 1))) :
  (∀ n, a n = 2 * n - 1) ∧
  (∀ n, T n = n / (2 * n + 1)) ∧
  set.Icc (1/4 : ℝ) (1/2 : ℝ) ⊆ set.Icc (1 / 4 : ℝ) (1 / 2 : ℝ) :=
by
  sorry

end arithmetic_sequence_general_term_and_sum_l717_717774


namespace copy_pages_l717_717128

theorem copy_pages (total_cents : ℕ) (cost_per_page : ℕ) (h1 : total_cents = 1500) (h2 : cost_per_page = 5) : 
  (total_cents / cost_per_page = 300) :=
sorry

end copy_pages_l717_717128


namespace lg5_value_l717_717403

variable {a b : ℝ}

def log8_eq_a (a : ℝ) : Prop := (Real.log 3 / Real.log 8 = a)
def log3_eq_b (b : ℝ) : Prop := (Real.log 5 / Real.log 3 = b)
def lg5 (a b : ℝ) : ℝ := 3 * a * b / (1 + 3 * a * b)

theorem lg5_value (ha : log8_eq_a a) (hb : log3_eq_b b) : Real.log 5 / Real.log 10 = lg5 a b := by
  sorry

end lg5_value_l717_717403


namespace smallest_multiple_l717_717747

theorem smallest_multiple (b : ℕ) (h1 : b % 6 = 0) (h2 : b % 15 = 0) (h3 : ∀ n : ℕ, (n % 6 = 0 ∧ n % 15 = 0) → n ≥ b) : b = 30 :=
sorry

end smallest_multiple_l717_717747


namespace units_digit_17_pow_2007_l717_717598

theorem units_digit_17_pow_2007 :
  (17 ^ 2007) % 10 = 3 := 
sorry

end units_digit_17_pow_2007_l717_717598


namespace circle_area_l717_717209

-- Given conditions
variables {BD AC : ℝ} (BD_pos : BD = 6) (AC_pos : AC = 12)
variables {R : ℝ} (R_pos : R = 15 / 2)

-- Prove that the area of the circles is \(\frac{225}{4}\pi\)
theorem circle_area (BD_pos : BD = 6) (AC_pos : AC = 12) (R : ℝ) (R_pos : R = 15 / 2) : 
        ∃ S, S = (225 / 4) * Real.pi := 
by sorry

end circle_area_l717_717209


namespace scott_bought_5_pounds_of_eggplants_l717_717536

variable (eggplant_pounds zucchini tomatoes onions basil quarts : ℕ)
variable (cost_per_pound_eggplant cost_per_pound_zucchini cost_per_pound_tomatoes cost_per_pound_onions cost_per_half_pound_basil cost_per_quart : ℝ)

#check zucchini = 4
#check cost_per_pound_zucchini = 2.0
#check tomatoes = 4
#check cost_per_pound_tomatoes = 3.5
#check onions = 3
#check cost_per_pound_onions = 1.0
#check basil = 1
#check cost_per_half_pound_basil = 2.5
#check quarts = 4
#check cost_per_quart = 10.0

theorem scott_bought_5_pounds_of_eggplants :
  let cost_other = (zucchini * cost_per_pound_zucchini + tomatoes * cost_per_pound_tomatoes + onions * cost_per_pound_onions + basil * 2 * cost_per_half_pound_basil)
  let total_cost = quarts * cost_per_quart
  let cost_eggplant = total_cost - cost_other
  eggplant_pounds = cost_eggplant / cost_per_pound_eggplant →
  eggplant_pounds = 5 := by
  sorry

end scott_bought_5_pounds_of_eggplants_l717_717536


namespace mixed_number_expression_l717_717731

theorem mixed_number_expression :
  23 * (((1 + 2 / 3: ℚ) + (2 + 1 / 4: ℚ))) / ((1 + 1 / 2: ℚ) + (1 + 1 / 5: ℚ)) = 367 / 108 := by
  sorry

end mixed_number_expression_l717_717731


namespace jessa_cupcakes_l717_717874

-- Define the number of classes and students
def fourth_grade_classes : ℕ := 3
def students_per_fourth_grade_class : ℕ := 30
def pe_classes : ℕ := 1
def students_per_pe_class : ℕ := 50

-- Calculate the total number of cupcakes needed
def total_cupcakes_needed : ℕ :=
  (fourth_grade_classes * students_per_fourth_grade_class) +
  (pe_classes * students_per_pe_class)

-- Statement to prove
theorem jessa_cupcakes : total_cupcakes_needed = 140 :=
by
  sorry

end jessa_cupcakes_l717_717874


namespace area_of_triangle_proof_l717_717849

noncomputable def area_of_acute_triangle 
  (a b c A B C : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) (hC : 0 < C ∧ C < π / 2)
  (hAeq : A + B + C = π)
  (hba : b = 2)
  (hBval : B = π / 3)
  (heq : c * sin A = sqrt 3 * a * cos C) :
  real :=
  1 / 2 * b * c * sin A

theorem area_of_triangle_proof (a b c A B C : ℝ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) (hC : 0 < C ∧ C < π / 2)
  (hAeq : A + B + C = π)
  (hba : b = 2)
  (hBval : B = π / 3)
  (heq : c * sin A = sqrt 3 * a * cos C):
  area_of_acute_triangle a b c A B C ha hb hc hA hB hC hAeq hba hBval heq = sqrt 3 :=
  sorry

end area_of_triangle_proof_l717_717849


namespace smallest_multiple_l717_717744

theorem smallest_multiple (b : ℕ) (h1 : b % 6 = 0) (h2 : b % 15 = 0) (h3 : ∀ n : ℕ, (n % 6 = 0 ∧ n % 15 = 0) → n ≥ b) : b = 30 :=
sorry

end smallest_multiple_l717_717744


namespace correct_number_of_paths_C_O_N_E_l717_717625

def num_paths_C_O_N_E (grid : Array (Array Char)) : ℕ :=
  let paths : List (List (Int × Int)) := 
    -- Define all possible paths satisfy the constraints described
    [ [(0, 0), (0, 1), (0, 2), (0, 3)], -- first row "CONE"
      [(1, 2), (1, 3), (2, 3), (3, 3)]  -- second row "C", third col "O", third col "N", fourth col "E"
    ]
  -- Count valid paths which result in spelling "CONE"
  paths.toArray.count (λ path : List (Int × Int), 
                        List.map (λ p => grid[p.1.toNat][p.2.toNat]) path == ['C', 'O', 'N', 'E'])

theorem correct_number_of_paths_C_O_N_E :
  let grid := #[#['C', 'O', 'N', 'E'], 
                #['O', 'N', 'E', 'C'],
                #['N', 'E', 'C', 'O'],
                #['E', 'C', 'O', 'N']]
  num_paths_C_O_N_E grid = 2 :=
  sorry -- proof goes here

end correct_number_of_paths_C_O_N_E_l717_717625


namespace trig_identity_l717_717825

-- State the problem as a theorem in Lean 4.
theorem trig_identity (α β : ℝ) (h1 : α ∈ Ioo (π / 2) π) (h2 : β ∈ Ioo (π / 2) π)
  (h3 : (1 - Real.cos (2 * α)) * (1 + Real.sin β) = Real.sin (2 * α) * Real.cos β) :
  2 * α + β = (5 * π) / 2 := 
sorry

end trig_identity_l717_717825


namespace find_constant_term_of_polynomial_with_negative_integer_roots_l717_717506

theorem find_constant_term_of_polynomial_with_negative_integer_roots
  (p q r s : ℝ) (t1 t2 t3 t4 : ℝ)
  (h_roots : ∀ {x : ℝ}, x^4 + p*x^3 + q*x^2 + r*x + s = (x + t1)*(x + t2)*(x + t3)*(x + t4))
  (h_neg_int_roots : ∀ {i : ℕ}, i < 4 → t1 = i ∨ t2 = i ∨ t3 = i ∨ t4 = i)
  (h_sum_coeffs : p + q + r + s = 168) :
  s = 144 :=
by
  sorry

end find_constant_term_of_polynomial_with_negative_integer_roots_l717_717506


namespace area_of_circles_l717_717206

theorem area_of_circles (BD AC : ℝ) (hBD : BD = 6) (hAC : AC = 12) : 
  ∃ S : ℝ, S = 225 / 4 * Real.pi :=
by
  sorry

end area_of_circles_l717_717206


namespace f1_neither_even_nor_odd_f2_min_value_l717_717626

noncomputable def f1 (x : ℝ) : ℝ :=
  x^2 + abs (x - 2) - 1

theorem f1_neither_even_nor_odd : ¬(∀ x : ℝ, f1 x = f1 (-x)) ∧ ¬(∀ x : ℝ, f1 x = -f1 (-x)) :=
sorry

noncomputable def f2 (x a : ℝ) : ℝ :=
  x^2 + abs (x - a) + 1

theorem f2_min_value (a : ℝ) :
  (if a < -1/2 then (∃ x, f2 x a = 3/4 - a)
  else if -1/2 ≤ a ∧ a ≤ 1/2 then (∃ x, f2 x a = a^2 + 1)
  else (∃ x, f2 x a = 3/4 + a)) :=
sorry

end f1_neither_even_nor_odd_f2_min_value_l717_717626


namespace smallest_pos_multiple_6_15_is_30_l717_717752

theorem smallest_pos_multiple_6_15_is_30 :
  ∃ b > 0, 6 ∣ b ∧ 15 ∣ b ∧ (∀ b', b' > 0 ∧ b' < b → ¬ (6 ∣ b' ∧ 15 ∣ b')) :=
by
  -- Implementation to be done
  sorry

end smallest_pos_multiple_6_15_is_30_l717_717752


namespace ellipse_solution_l717_717038

noncomputable def ellipse_standard_eq (a b c k m : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ (c/a = sqrt 3 / 3) ∧ (1/2 * b * 2 * c = sqrt 2) ∧
  (a^2 = b^2 + c^2) ∧ 
  (ellipse_eq : (∀ (x y : ℝ), (x^2/a^2 + y^2/b^2 = 1) ↔ ((x^2)/3 + (y^2)/2 = 1))) ∧
  (line_eq : ∀ (x : ℝ), y = k*x + m) ∧ 
  ((2 + 3 * k^2) * x^2 + 6 * k * m * x + 3 * m^2 - 6 = 0) →

  (3 * k^2 - m^2 + 2 > 0) ∧ 
  (m = k + (2 / (3 * k))) ∧ 
  (m ∈ (-∞, -2 * sqrt(6) / 3] ∪ [2 * sqrt(6) / 3, ∞))

-- Example of how these might be combined in some form:
theorem ellipse_solution (a b c k m : ℝ) :
  ellipse_standard_eq a b c k m :=
sorry

end ellipse_solution_l717_717038


namespace sum_of_squares_ge_sum_of_products_l717_717896

theorem sum_of_squares_ge_sum_of_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := by
  sorry

end sum_of_squares_ge_sum_of_products_l717_717896


namespace find_c_d_l717_717698

theorem find_c_d (c d : ℝ) (h_nonzero_c : c ≠ 0) (h_nonzero_d : d ≠ 0)
  (h_roots : ∀ x, 2 * x^2 + c * x + d = 0 ↔ x = 2 * c ∨ x = 2 * d) :
  (c = 1 / 2 ∧ d = -5 / 8) :=
begin
  sorry
end

end find_c_d_l717_717698


namespace distinct_real_roots_l717_717919

theorem distinct_real_roots :
  ∃ (a : Fin 10 → ℝ), (Function.Injective a) ∧
  (polynomial.map polynomial.C (∏ i in Finset.univ, polynomial.X - polynomial.C (a i)))
  =
  (polynomial.map polynomial.C (∏ i in Finset.univ, polynomial.X + polynomial.C (a i)))
  ∧
  {0, 1, -1, 2, -2}.card = 5 :=
by
  use (λ i: Fin 10, i - 7)
  sorry

end distinct_real_roots_l717_717919


namespace bmws_sold_l717_717276

-- Definitions stated by the problem:
def total_cars : ℕ := 300
def percentage_mercedes : ℝ := 0.20
def percentage_toyota : ℝ := 0.25
def percentage_nissan : ℝ := 0.10
def percentage_bmws : ℝ := 1 - (percentage_mercedes + percentage_toyota + percentage_nissan)

-- Statement to prove:
theorem bmws_sold : (total_cars : ℝ) * percentage_bmws = 135 := by
  sorry

end bmws_sold_l717_717276


namespace power_and_greatest_prime_factor_l717_717246

theorem power_and_greatest_prime_factor (n : ℕ) (h₁ : 4 ^ 17 = 2 ^ 34)
  (h₂ : ∃ p : ℕ, prime p ∧ p ∣ 2 ^ 34 - 2 ^ n ∧ (∀ q : ℕ, prime q ∧ q ∣ 2 ^ 34 - 2 ^ n → q ≤ p)) :
  n = 29 :=
by
  sorry

end power_and_greatest_prime_factor_l717_717246


namespace reading_time_difference_is_180_minutes_l717_717877

variable (Jonathan_speed Alice_speed book_length : ℕ)
variable (time_diff : ℕ)

-- Conditions
def Jonathan_speed := 150
def Alice_speed := 75
def book_length := 450

-- Calculate times
def Jonathan_time := book_length / Jonathan_speed
def Alice_time := book_length / Alice_speed

-- Convert time difference from hours to minutes
def time_diff := 60 * (Alice_time - Jonathan_time)

-- Statement to prove
theorem reading_time_difference_is_180_minutes 
  (h1 : Jonathan_speed = 150) 
  (h2 : Alice_speed = 75) 
  (h3 : book_length = 450) 
  (h4 : Jonathan_time = book_length / Jonathan_speed)
  (h5 : Alice_time = book_length / Alice_speed)
  (h6 : time_diff = 60 * (Alice_time - Jonathan_time)) : 
  time_diff = 180 := 
sorry

end reading_time_difference_is_180_minutes_l717_717877


namespace smaller_cube_surface_area_l717_717305

/-- A cube has a surface area of 54 square meters.
    A sphere is inscribed within this cube.
    A second smaller cube is inscribed within this sphere.
    Prove that the surface area of the smaller cube is 18 square meters. -/
theorem smaller_cube_surface_area
  (S₁ : ℝ)
  (h₁ : 6 * S₁^2 = 54)
  (S₂ : ℝ)
  (h₂ : S₂ = S₁ / Math.sqrt 3 * (3 / Math.sqrt 3))
  (h₃ : S₁ = 3)
  : 6 * S₂^2 = 18 :=
sorry

end smaller_cube_surface_area_l717_717305


namespace line_MN_fixed_point_l717_717772

section EllipseProof

variable {a b : ℝ} (h1 : a = 2 * Real.sqrt 2) (h2 : b = 2)

noncomputable def ellipse_eq : Prop := ∀ (x y : ℝ), (x^2 / 8 + y^2 / 4 = 1) ↔ (Real.sqrt (x^2 + y^2) = 1)

def fixed_point (x y : ℝ) : Prop := x = 0 ∧ y = 1

theorem line_MN_fixed_point (x1 y1 x2 y2 : ℝ) (h3 : x1^2 / 8 + y1^2 / 4 = 1) (h4 : x2^2 / 8 + y2^2 / 4 = 1) (h5 : y1 = k * x1 + 1) (h6 : y2 = k * x2 + 1) (h7 : k ≠ 0) : fixed_point 0 1 := sorry

end EllipseProof

end line_MN_fixed_point_l717_717772


namespace tan_monotonic_intervals_theorem_l717_717375

def tan_monotonic_intervals (k : ℤ) : Set ℝ :=
  {x | -π / 12 + k * π / 2 < x ∧ x < 5 * π / 12 + k * π / 2}

theorem tan_monotonic_intervals_theorem :
  ∀ k : ℤ, ∀ x : ℝ, (∃ (y : ℝ), y = (2 * x - π / 3) ∧ (-π / 2 + k * π) < y ∧ y < (π / 2 + k * π)) ↔ x ∈ tan_monotonic_intervals k :=
by
  intros
  split
  · intro h
    obtain ⟨y, hy1, hy2, hy3⟩ := h
    rw hy1 at *
    split
    · linarith
    · linarith
  · intro h
    use (2 * x - π / 3)
    split
    · refl
    · linarith
    · linarith

# Check if the theorem is well-formed.
#print tan_monotonic_intervals_theorem

end tan_monotonic_intervals_theorem_l717_717375


namespace jane_hector_meeting_point_l717_717655

def park_side := 24
def park_perimeter := 4 * park_side
def hector_speed (s : ℝ) := s
def jane_speed_no_wind (s : ℝ) := 2 * s
def jane_speed_with_wind (s : ℝ) := 1.5 * s
def jane_distance_before_wind := 12

theorem jane_hector_meeting_point :
  ∀ (s : ℝ) (HectorSpeed : hector_speed s > 0), 
  let ⟨time_before_wind, remaining_distance, remaining_time, total_time, hector_distance⟩ := 
    (jane_distance_before_wind / jane_speed_no_wind s, 
    park_perimeter - jane_distance_before_wind, 
    (park_perimeter - jane_distance_before_wind) / (hector_speed s + jane_speed_with_wind s), 
    (jane_distance_before_wind / jane_speed_no_wind s) + (park_perimeter - jane_distance_before_wind) / (hector_speed s + jane_speed_with_wind s), 
    (jane_distance_before_wind / jane_speed_no_wind s) + (park_perimeter - jane_distance_before_wind) / (hector_speed s + jane_speed_with_wind s) * hector_speed s) 
  in 
  (10.8 = hector_distance % park_side) :=
sorry

end jane_hector_meeting_point_l717_717655


namespace prob_X_lt_0_l717_717796

noncomputable def normal_cdf (μ σ : ℝ) : ℝ → ℝ := sorry
noncomputable def normal_pdf (μ σ : ℝ) : ℝ → ℝ := sorry

variable {X : ℝ → ℝ}
variable {σ : ℝ}

-- Conditions
axiom normal_dist : ∀ X, X = normal_cdf 1 σ
axiom prob_condition : ∀ P, P (0 < X 1) = 0.4

-- Question to prove
theorem prob_X_lt_0 : P (X 0) = 0.1 :=
by
  sorry

end prob_X_lt_0_l717_717796


namespace necessary_but_not_sufficient_necessity_l717_717048

variables {ℝ : Type*} [Real]

def derivative (f : ℝ → ℝ) (x₀ : ℝ) : ℝ := 
sorry -- Here you can use specific derivative definitions in Lean

def is_extreme_value (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
sorry -- Predicate checking if x₀ is an extreme value of f

theorem necessary_but_not_sufficient
  (f : ℝ → ℝ) (h1 : differentiable ℝ f)
  (x₀ : ℝ) :
  (derivative f x₀ = 0) → (¬is_extreme_value f x₀) := 
sorry

theorem necessity 
  (f : ℝ → ℝ) (h1 : differentiable ℝ f)
  (x₀ : ℝ)
  (h2 : is_extreme_value f x₀) :
  derivative f x₀ = 0 :=
sorry

end necessary_but_not_sufficient_necessity_l717_717048


namespace switcheroo_returns_original_l717_717154

def switcheroo {α : Type} [Inhabited α] [DecidableEq α] (n : ℕ) (s : Array α) : Array α :=
  Array.foldr (λ k acc, let (left, right) := acc.splitAt ((2^k : ℕ) % s.size)
                       in right ++ left) s (List.range n)

theorem switcheroo_returns_original {α : Type} [Inhabited α] [DecidableEq α] :
  ∀ (n : ℕ) (s : Array α), s.size = 2^n → switcheroo n^(2^n) s = s :=
by
  intros n s h
  sorry

end switcheroo_returns_original_l717_717154


namespace scientific_notation_41600_l717_717673

theorem scientific_notation_41600 : (4.16 * 10^4) = 41600 := by
  sorry

end scientific_notation_41600_l717_717673


namespace find_ratio_l717_717389

variable {a : ℕ → ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, a n > 0 ∧ a (n+1) / a n = a 1 / a 0

def forms_arithmetic_sequence (a1 a3_half a2_times_two : ℝ) : Prop :=
  a3_half = (a1 + a2_times_two) / 2

theorem find_ratio (a : ℕ → ℝ) (h_geom : is_geometric_sequence a)
  (h_arith : forms_arithmetic_sequence (a 1) (1/2 * a 3) (2 * a 2)) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 :=
sorry

end find_ratio_l717_717389


namespace probability_two_positive_roots_l717_717093

noncomputable def probability_positive_roots (a b : ℕ) : ℚ := 
  if 4 ≤ a ∧ a ≤ 6 ∧ b ≤ 2 then 1/36 else 0

theorem probability_two_positive_roots :
  let outcomes := (list.range' 1 6).product (list.range' 1 6)
  let favorable := outcomes.filter (fun (a, b) => 4 ≤ a ∧ a ≤ 6 ∧ b ≤ 2)
  favorable.length / outcomes.length = 1/18 := 
by
  sorry

end probability_two_positive_roots_l717_717093


namespace determine_inverses_l717_717081

def graphF_has_inverse : Prop := ∀ y : ℝ, ∃! x : ℝ, (y = m * x + b) where m > 0
def graphG_has_inverse : Prop := ¬(∀ y : ℝ, ∃! x : ℝ, (y = -x^2 + 4))
def graphH_has_inverse : Prop := ¬(∀ y : ℝ, ∃! x : ℝ, (y = 2 ∨ y = -2))
def graphI_has_inverse : Prop := ∀ y : ℝ, ∃! x : ℝ, ((x < 0 → y = 2*x + 4) ∧ (x ≥ 0 → y = 2*x - 4))

theorem determine_inverses : graphF_has_inverse ∧ ¬graphG_has_inverse ∧ ¬graphH_has_inverse ∧ graphI_has_inverse :=
by { sorry }

end determine_inverses_l717_717081


namespace meet_after_5_minutes_l717_717160

theorem meet_after_5_minutes (n : ℕ) (d : ℝ) (a_speed b_initial_speed b_speed_increase : ℝ) 
  (H1 : d = 30) 
  (H2 : a_speed = 3) 
  (H3 : b_initial_speed = 2) 
  (H4 : b_speed_increase = 0.5) 
  (H5 : ∑ i in (Finset.range (n + 1)), a_speed + (b_initial_speed + b_speed_increase * i) = d) :
  n = 5 := 
by
  sorry

end meet_after_5_minutes_l717_717160


namespace inequality_solution_set_l717_717047

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + (a-1)*x^2

theorem inequality_solution_set (a : ℝ) (ha : ∀ x : ℝ, f x a = -f (-x) a) :
  {x : ℝ | f (a*x) a > f (a-x) a} = {x : ℝ | x > 1/2} :=
by
  sorry

end inequality_solution_set_l717_717047


namespace alpha_value_l717_717413

theorem alpha_value (α : ℝ) (h : (sin (5 * Real.pi / 6), cos (5 * Real.pi / 6)) = (sin α, cos α)) : 
  α = 5 * Real.pi / 3 :=
sorry

end alpha_value_l717_717413


namespace angle_B_is_120_l717_717481

variables {α : Type*} [real_linear_ordered_field α]

-- Define the isosceles property and similarity of the two triangles
structure Isosceles (A B C : α) : Prop :=
  (equal_sides : A = C)

structure Similar (A B C A1 B1 C1 : α) : Prop :=
  (similar_ratio : A / A1 = B / B1 ∧ A / A1 = C / C1)

-- Given the ratio AC : A1C1 = 5 : √3
axiom ratio_AC_A1C1 (A C A1 C1 : α) : A / A1 = 5 / real.sqrt 3

-- Vertical and placement conditions
axiom perpendicular_A1B1_BC (A1 B1 C : α) : A1 * C = 0
axiom vertex_places (A1 AC : α) (B1 BC : α) (C1 AB : α) 
  : A1 = AC ∧ B1 = BC ∧ C1 = AB

-- Prove that the angle B is 120 degrees
theorem angle_B_is_120 {A B C A1 B1 C1 : α} :
  Isosceles A B C →
  Isosceles A1 B1 C1 →
  Similar A B C A1 B1 C1 →
  ratio_AC_A1C1 A C A1 C1 →
  perpendicular_A1B1_BC A1 B1 C →
  vertex_places A1 AC B1 BC C1 AB →
  ∠ B = 120 :=
by sorry

end angle_B_is_120_l717_717481


namespace find_integer_n_l717_717004

theorem find_integer_n (n : ℤ) (hn_range : -180 ≤ n ∧ n ≤ 180) :
  (sin (n : ℝ)) = cos (810 : ℝ) → (n = 0 ∨ n = 180 ∨ n = -180) :=
by
  sorry

end find_integer_n_l717_717004


namespace pythagorean_triples_probability_l717_717835

theorem pythagorean_triples_probability :
  let A := {1, 2, 3, 4, 5}
  ∃ (U : finset ℕ) (H : U ⊆ A) (hcard : U.card = 3),
  (∃ (x y z : ℕ) (hx : x ∈ U) (hy : y ∈ U) (hz : z ∈ U), x ^ 2 + y ^ 2 = z ^ 2 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) → 
  1 / U.powerset.card = 1 / 10 :=
by 
  sorry

end pythagorean_triples_probability_l717_717835


namespace cos_angle_AF2F1_l717_717954

open Real

section hyperbola
variables (a b x y : ℝ)
variables (F1 F2 A : EuclideanGeometry.Point ℝ)
variables (AF1 AF2 : ℝ)

/-- The hyperbola C is defined such that x^2/a^2 - y^2/b^2 = 1 with a > 0 and b > 0.
    The point A on the hyperbola satisfies |F1A| = 2|F2A|. 
    One asymptote of the hyperbola is perpendicular to the line x + 2y + 1 = 0.
    We aim to prove that cos ∠AF2F1 = √5/5. -/
theorem cos_angle_AF2F1 (ha : a > 0) (hb : b = 2 * a)
  (h_asymp_perp : ∃ m : ℝ, m = b / a ∧ m * (-1/2) = -1)
  (h_foci : ∀ (A : EuclideanGeometry.Point ℝ), |F1 - A| = 2 * |F2 - A|) :
  let c := sqrt (a^2 + b^2) 
  in cos (angle A F2 F1) = sqrt 5 / 5 :=
begin
  let c := sqrt (a^2 + b^2),
  sorry
end

end hyperbola

end cos_angle_AF2F1_l717_717954


namespace exists_r_not_prime_l717_717771

theorem exists_r_not_prime (n : ℕ) : ∃ r : ℕ, ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ¬ Nat.Prime (r + k) :=
by
  let r := (Nat.factorial (n+1)) + 1
  use r
  intros k hk
  have h_div : (Nat.factorial (n+1)) % (k+1) = 0 := Nat.factorial_dvd (le_add_one hk.left)
  have h_rk_mod : (r + k) % (k+1) = 0 :=
    by
      rw [← Nat.add_mod, Nat.add_comm, Nat.add_mod]
      exact h_div
  rw [Nat.Prime, not_and]
  intro hp
  have h_rk_gt : r + k > 1 := by linarith
  exact not_lt_of_ge (Nat.le_of_dvd h_rk_gt (dvd_of_mod_eq_zero h_rk_mod)) (Nat.Prime.gt_one hp)
  sorry


end exists_r_not_prime_l717_717771


namespace length_of_cube_side_l717_717942

theorem length_of_cube_side
  (cost_per_kg : ℝ)
  (area_per_kg : ℝ)
  (total_cost : ℝ)
  (total_paint_cost : ℝ) :
  (cost_per_kg = 40) →
  (area_per_kg = 20) →
  (total_cost = 1200) →
  (total_paint_cost = 600) →
  let L := real.sqrt (total_paint_cost / 6) in
  L = 10 :=
by 
  intro h1 h2 h3 h4
  let L := real.sqrt (total_paint_cost / 6)
  have h5 : L = real.sqrt 100 := by {
    rw [←h4, ←h1, ←h2, ←h3],
    show (√(total_cost / cost_per_kg * area_per_kg / 6)) = 10,
    calc (√(total_cost / cost_per_kg * area_per_kg / 6))
        = (√(1200 / 40 * 20 / 6)) : by rw [h3, h1, h2]
    ... = (√(30 * 20 / 6))       : by norm_num
    ... = (√(600 / 6))           : by norm_num
    ... = (√100)                 : by norm_num }

  exact h5.trans real.sqrt_eq_rfl

end length_of_cube_side_l717_717942


namespace tv_tower_height_greater_l717_717214

theorem tv_tower_height_greater (d : ℝ) (θ : ℝ) (H : ℝ)
  (h_condition : d = 100)
  (θ_condition : θ = 46) :
  103.3 < 100 * Real.tan (θ * Real.pi / 180) :=
by
  rw [h_condition, θ_condition]
  sorry

end tv_tower_height_greater_l717_717214


namespace find_a7_l717_717708

variable (a : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n+1) = r * a n

axiom a3_eq_1 : a 3 = 1
axiom det_eq_0 : a 6 * a 8 - 8 * 8 = 0

theorem find_a7 (h_geom : geometric_sequence a) : a 7 = 8 :=
  sorry

end find_a7_l717_717708


namespace cheese_partition_proof_l717_717907

variable {α : Type*}

noncomputable def cheese_partition_possible (m : Fin 9 → ℝ) : Prop :=
  ∀ (hm : ∀ i j, i < j → m i < m j), 
    ∃ (p : ℝ) (k : Fin 9), 0 < p < m k ∧ 
    ∃ (P1 P2 : Finset ℝ), 
      P1.card = 5 ∧ P2.card = 5 ∧ 
      (P1.sum id = P2.sum id) ∧ 
      (P1 = (Finset.univ.image m).erase (m k) ∪ {p}) ∧ 
      (P2 = (Finset.univ.image m).erase (m k) ∪ {m k - p})

theorem cheese_partition_proof (m : Fin 9 → ℝ) 
  (hm : ∀ i j, i < j → m i < m j) : cheese_partition_possible m :=
sorry

end cheese_partition_proof_l717_717907


namespace distinct_cube_configurations_l717_717704

theorem distinct_cube_configurations : 
  let cube = (λ (w b : ℕ), w = 3 ∧ b = 5) in
  number_of_distinct_configurations cube  = 19 := 
sorry

end distinct_cube_configurations_l717_717704


namespace range_of_a_l717_717397

theorem range_of_a (a : ℝ) (p q : Prop) 
    (h₀ : (p ↔ (3 - 2 * a > 1))) 
    (h₁ : (q ↔ (-2 < a ∧ a < 2))) 
    (h₂ : (p ∨ q)) 
    (h₃ : ¬ (p ∧ q)) : 
    a ≤ -2 ∨ 1 ≤ a ∧ a < 2 :=
by
  sorry

end range_of_a_l717_717397


namespace calc_result_l717_717335

theorem calc_result : 75 * 1313 - 25 * 1313 = 65750 := 
by 
  sorry

end calc_result_l717_717335


namespace range_of_a_l717_717098

noncomputable def f (a x : ℝ) : ℝ := (4 / 3) * x ^ 3 - 2 * a * x ^ 2 - (a - 2) * x + 5

-- f' is the derivative of f
noncomputable def f' (a x : ℝ) : ℝ := 4 * x ^ 2 - 4 * a * x - (a - 2)

-- Problem statement
theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' a x1 = 0 ∧ f' a x2 = 0) ↔ (a > 1 ∨ a < -2) :=
begin
  sorry
end

end range_of_a_l717_717098


namespace vacuum_tube_pins_and_holes_l717_717668

theorem vacuum_tube_pins_and_holes :
  ∀ (pins holes : Finset ℕ), 
  pins = {1, 2, 3, 4, 5, 6, 7} →
  holes = {1, 2, 3, 4, 5, 6, 7} →
  (∃ (a : ℕ), ∀ k ∈ pins, ∃ b ∈ holes, (2 * k) % 7 = b) := by
  sorry

end vacuum_tube_pins_and_holes_l717_717668


namespace division_remainder_l717_717843

theorem division_remainder (dividend divisor quotient : ℕ) (h1: dividend = 686) (h2: divisor = 36) (h3: quotient = 19) : ∃ remainder, dividend = (divisor * quotient) + remainder ∧ remainder = 2 :=
by {
  exists 2,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end division_remainder_l717_717843


namespace P_implies_Q_Q_not_implies_P_l717_717779

variable (x y : ℝ)

def P := (x - 1)^2 + (y - 1)^2 = 0
def Q := (x - 1) * (y - 1) = 0

theorem P_implies_Q : P → Q := sorry
theorem Q_not_implies_P : ¬ (Q → P) := sorry

end P_implies_Q_Q_not_implies_P_l717_717779


namespace problem_solution_l717_717424

noncomputable def f (x : ℝ) (a : ℝ) := 2 * sqrt 3 * sin x * sin (π / 2 - x) + 2 * cos x^2 + a
noncomputable def g (x : ℝ) := 2 * sin (2*x - π/3) + 1

theorem problem_solution (k : ℤ) :
  (∀ x ∈ set.Icc (-π/3 + k * π) (π/6 + k * π), monotone_on (λ x, f x 0) (set.Icc (-π/3 + k * π) (π/6 + k * π))) ∧
  (∃ x, f x 0 = 3) ∧
  (set.range (g) ∩ set.Icc (1 - sqrt 3) 3 = set.Icc (1 - sqrt 3) 3) :=
begin
  sorry
end

end problem_solution_l717_717424


namespace negation_of_universal_statement_l717_717565

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^4 - x^3 + x^2 + 5 ≤ 0) ↔ (∃ x : ℝ, x^4 - x^3 + x^2 + 5 > 0) :=
by sorry

end negation_of_universal_statement_l717_717565


namespace increasing_function_range_l717_717830

theorem increasing_function_range (m : ℝ) : 
  (∀ x : ℝ, x > 0 → (deriv (λ x, (m + 2) / x)).to_fun x > 0) ↔ m < -2 :=
by
  sorry

end increasing_function_range_l717_717830


namespace min_value_of_m_l717_717449

theorem min_value_of_m (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b + b * c + c * a = -1) (h3 : a * b * c = -m) : 
    m = - (min (-a ^ 3 + a ^ 2 + a ) (- (1 / 27))) := 
sorry

end min_value_of_m_l717_717449


namespace locus_and_tangent_l717_717468

-- Define point A and the line l
def point_A : ℝ × ℝ := (1, 0)
def directrix : set (ℝ × ℝ) := { p | p.1 = -1 }

-- Define the conditions for locus E and point P
def focus : ℝ × ℝ := point_A
def point_P : ℝ × ℝ := (1, 2)

-- Define the parabola and tangent line equations
def parabola_eqn (x y : ℝ) : Prop := y^2 = 4 * x
def tangent_eqn (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem stating the proof problem
theorem locus_and_tangent :
  ∀ (x y : ℝ), 
    (parabola_eqn x y) ∧
    (tangent_eqn 1 2) :=
by 
  sorry

end locus_and_tangent_l717_717468


namespace problem_statement_l717_717964

-- Define the sequences {a_n} and {b_n} with the given properties
def a_seq (n : ℕ) : ℝ :=
  if n = 1 then 1 / 3 else 1 / 3^n

def b_seq (n : ℕ) : ℝ :=
  3^n / n * a_seq n

-- Define the sum function S_n
noncomputable def S (n : ℕ) : ℝ :=
  ∑ i in finset.range n, b_seq (i + 1)

-- Now we need to prove the main statement
theorem problem_statement : S 2018 - 1 < real.log 2018 :=
by {
  sorry
}

end problem_statement_l717_717964


namespace petya_prevents_vasya_l717_717914

-- Define the nature of fractions and the players' turns
def is_natural_sum (fractions : List ℚ) : Prop :=
  (fractions.sum = ⌊fractions.sum⌋)

def petya_vasya_game_prevent (fractions : List ℚ) : Prop :=
  ∀ k : ℕ, ∀ additional_fractions : List ℚ, 
  (additional_fractions.length = k) →
  ¬ is_natural_sum (fractions ++ additional_fractions)

theorem petya_prevents_vasya : ∀ fractions : List ℚ, petya_vasya_game_prevent fractions :=
by
  sorry

end petya_prevents_vasya_l717_717914


namespace turtle_reaches_waterhole_28_minutes_after_meeting_l717_717612

theorem turtle_reaches_waterhole_28_minutes_after_meeting (x : ℝ) (distance_lion1 : ℝ := 5 * x) 
  (speed_lion2 : ℝ := 1.5 * x) (distance_turtle : ℝ := 30) (speed_turtle : ℝ := 1/30) : 
  ∃ t_meeting : ℝ, t_meeting = 2 ∧ (distance_turtle - speed_turtle * t_meeting) / speed_turtle = 28 :=
by 
  sorry

end turtle_reaches_waterhole_28_minutes_after_meeting_l717_717612


namespace midpoint_intersection_of_tangent_line_l717_717496

-- Definitions for the conditions
variables {Γ1 Γ2 : Type} [circle Γ1] [circle Γ2]
variables {A B M N : point}
variables {Δ : line}

-- Conditions
axiom intersect_points (h1 : intersect Γ1 Γ2 A B)
axiom tangent_points (h2 : tangent Δ Γ1 M) (h3 : tangent Δ Γ2 N)

-- The statement to be proved
theorem midpoint_intersection_of_tangent_line :
  midpoint (segment A B) (segment M N) :=
sorry

end midpoint_intersection_of_tangent_line_l717_717496


namespace each_persons_share_l717_717263

def dining_shared_expense (total_bill : ℝ) (percentage_tip : ℝ) (people : ℕ) : ℝ :=
  let tip := total_bill * (percentage_tip / 100)
  let total_with_tip := total_bill + tip
  total_with_tip / people

theorem each_persons_share (total_bill : ℝ) (percentage_tip : ℝ) (people : ℕ) :
  total_bill = 211 → percentage_tip = 15 → people = 6 →
  (dining_shared_expense total_bill percentage_tip people ≈ 40.44) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end each_persons_share_l717_717263


namespace chapters_per_day_l717_717021

theorem chapters_per_day (total_pages : ℕ) (total_chapters : ℕ) (total_days : ℕ)
  (h1 : total_pages = 193)
  (h2 : total_chapters = 15)
  (h3 : total_days = 660) :
  (total_chapters : ℝ) / total_days = 0.0227 :=
by 
  sorry

end chapters_per_day_l717_717021


namespace sum_of_three_distinct_l717_717082

def S : Set ℤ := {2, 5, 8, 11, 14, 17, 20}

theorem sum_of_three_distinct (S : Set ℤ) (h : S = {2, 5, 8, 11, 14, 17, 20}) :
  (∃ n : ℕ, n = 13 ∧ ∀ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    ∃ k : ℕ, a + b + c = 3 * k) := 
by  -- The proof goes here.
  sorry

end sum_of_three_distinct_l717_717082


namespace least_upper_bound_of_ratio_l717_717879

noncomputable theory
open Real

theorem least_upper_bound_of_ratio 
  (T : Type)
  (inscribed : T → (ℝ × ℝ))
  (vertices : T → (ℝ × ℝ × ℝ × ℝ))
  (AB CD : T → (ℝ × ℝ))
  (AB_parallel_CD : ∀ t, AB t.1 = AB t.2 ∧ CD t.1 = CD t.2)
  (s1 s2 d : ℝ)
  (OE_is_intersection : ∀ t, d ≠ 0 → (d = abs (s1 - s2) / (1 - cos (s1 - s2)) )) :
  Sup { (s1 - s2) / d | t ∈ T ∧ d ≠ 0 } = 2 := by
  sorry

end least_upper_bound_of_ratio_l717_717879


namespace sum_of_largest_and_smallest_two_digit_numbers_l717_717011

theorem sum_of_largest_and_smallest_two_digit_numbers : 
  let digits := {3, 5, 7, 8},
      largest := 87,
      smallest := 35
  in largest + smallest = 122 := 
by 
  -- sum the largest and smallest numbers
  let largest := 87
  let smallest := 35
  have sum := largest + smallest
  exact sum = 122

end sum_of_largest_and_smallest_two_digit_numbers_l717_717011


namespace pell_infinite_solutions_l717_717127

theorem pell_infinite_solutions : ∃ m : ℕ, ∃ a b c : ℕ, 
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (∀ n : ℕ, ∃ an bn cn : ℕ, 
    (1 / an + 1 / bn + 1 / cn + 1 / (an * bn * cn) = m / (an + bn + cn))) := 
sorry

end pell_infinite_solutions_l717_717127


namespace ellipse_major_minor_axis_ratio_l717_717412

theorem ellipse_major_minor_axis_ratio
  (a b : ℝ)
  (h₀ : a = 2 * b):
  2 * a = 4 * b :=
by
  sorry

end ellipse_major_minor_axis_ratio_l717_717412


namespace jade_savings_l717_717445

noncomputable def jade_monthly_savings 
    (income : ℝ) 
    (k401_percent : ℝ) 
    (tax_percent : ℝ) 
    (living_expenses_percent : ℝ) 
    (insurance_percent : ℝ) 
    (transportation_percent : ℝ) 
    (utilities_percent : ℝ) : ℝ :=
    let k401_contrib := k401_percent * income
    let tax_deduct := tax_percent * income
    let post_deduct_income := income - k401_contrib - tax_deduct
    let living_expenses := living_expenses_percent * post_deduct_income
    let insurance := insurance_percent * post_deduct_income
    let transportation := transportation_percent * post_deduct_income
    let utilities := utilities_percent * post_deduct_income
    let total_expenses := living_expenses + insurance + transportation + utilities
    post_deduct_income - total_expenses

theorem jade_savings (income : ℝ) 
    (k401_percent : ℝ) 
    (tax_percent : ℝ) 
    (living_expenses_percent : ℝ) 
    (insurance_percent : ℝ) 
    (transportation_percent : ℝ) 
    (utilities_percent : ℝ) 
    (biweekly_salary: ℝ) 
    (h_income := 2800 : income = 2800)
    (h_k401_percent := 0.08 : k401_percent = 0.08)
    (h_tax_percent := 0.10 : tax_percent = 0.10)
    (h_living_expenses_percent := 0.55 : living_expenses_percent = 0.55)
    (h_insurance_percent := 0.20 : insurance_percent = 0.20)
    (h_transportation_percent := 0.12 : transportation_percent = 0.12)
    (h_utilities_percent := 0.08 : utilities_percent = 0.08)
    (h_biweekly_salary := (2800 / 2) : biweekly_salary = 1400) : 
    jade_monthly_savings income k401_percent tax_percent living_expenses_percent insurance_percent transportation_percent utilities_percent = 114.80 :=
by
  rw [h_income, h_k401_percent, h_tax_percent, h_living_expenses_percent, h_insurance_percent, h_transportation_percent, h_utilities_percent, h_biweekly_salary]
  unfold jade_monthly_savings
  norm_num
  sorry

end jade_savings_l717_717445


namespace part_a_part_b_l717_717882

variables {G : Type*} [simple_graph G]
variables {n m k : ℕ} (h_k : 2 ≤ k)
variables [fintype (vertex_set G)] [fin n] [fin m]
variables (G_no_cycles : ∀ c, cycle c G → ¬ (3 ≤ length c ∧ length c ≤ 2 * k))

theorem part_a (G : simple_graph (fin n)) (m : ℕ) :
  ∃ (S : finset (vertex_set G)), S.nonempty ∧ ∀ v ∈ S, (S.filter (λ w, G.adj v w)).card ≥ nat_ceil(↑m / ↑n) :=
sorry

theorem part_b (G S : simple_graph (fin n)) (H : induced_subgraph G S) (v : vertex_set H) :
  ∀ k, ∃ t, BFS_subgraph k G S v = t ∧ t.card ≥ nat_ceil((↑m / ↑n - 1)^k) :=
sorry

end part_a_part_b_l717_717882


namespace regular_ngon_on_parallel_lines_l717_717384

theorem regular_ngon_on_parallel_lines (n : ℕ) : 
  (∃ f : ℝ → ℝ, (∀ m : ℕ, ∃ k : ℕ, f (m * (360 / n)) = k * (360 / n))) ↔
  n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end regular_ngon_on_parallel_lines_l717_717384


namespace jessa_gave_3_bills_l717_717023

variable (J G K : ℕ)
variable (billsGiven : ℕ)

/-- Initial conditions and question for the problem -/
def initial_conditions :=
  G = 16 ∧
  K = J - 2 ∧
  G = 2 * K ∧
  (J - billsGiven = 7)

/-- The theorem to prove: Jessa gave 3 bills to Geric -/
theorem jessa_gave_3_bills (h : initial_conditions J G K billsGiven) : billsGiven = 3 := 
sorry

end jessa_gave_3_bills_l717_717023


namespace find_refreshment_volunteers_l717_717622

theorem find_refreshment_volunteers (T S B : ℕ) (conditionT : T = 84) (conditionS : S = 25) (conditionB : B = 11) (R N : ℕ) (conditionR : R = 1.5 * N) (conditionT_eq : T = S + R + N - B) : R = 42 :=
sorry

end find_refreshment_volunteers_l717_717622


namespace circle_equation_l717_717952

-- Definitions for the given conditions
def A : ℝ × ℝ := (1, -1)
def B : ℝ × ℝ := (-1, 1)
def line (p : ℝ × ℝ) : Prop := p.1 + p.2 - 2 = 0

-- Theorem statement for the proof problem
theorem circle_equation :
  ∃ (h k : ℝ), line (h, k) ∧ (h = 1) ∧ (k = 1) ∧
  ((h - 1)^2 + (k - 1)^2 = 4) :=
sorry

end circle_equation_l717_717952


namespace QZ_length_l717_717858

variables (A B Q Y Z : Type) [MetricSpace A] [MetricSpace B] [MetricSpace Q] [MetricSpace Y] [MetricSpace Z]

def segment_parallel (AB YZ : Prop) := AB = A ∧ B = B ∧ Y = Y ∧ Z = Z ∧ (A * Y) = (B * Z)

def point_distance (p1 p2 : Type) [MetricSpace p1] [MetricSpace p2] : Type := sorry

axiom AZ_length : point_distance A Z = 48
axiom BQ_length : point_distance B Q = 15
axiom QY_length : point_distance Q Y = 30
axiom AB_parallel_YZ : segment_parallel A B Y Z = true

theorem QZ_length : point_distance Q Z = 32 :=
by
  sorry

end QZ_length_l717_717858


namespace sin_cos_sum_l717_717050

variable (θ : ℝ)
hypothesis (h1 : θ ∈ Icc (3 * Real.pi / 2) (2 * Real.pi))
hypothesis (h2 : Real.tan θ = -3 / 4)

theorem sin_cos_sum : Real.sin θ + Real.cos θ = 1 / 5 := by
  sorry

end sin_cos_sum_l717_717050


namespace solve_for_a_b_l717_717136

open Complex

theorem solve_for_a_b (a b : ℝ) (h : (mk 1 2) / (mk a b) = mk 1 1) : 
  a = 3 / 2 ∧ b = 1 / 2 :=
sorry

end solve_for_a_b_l717_717136


namespace percentage_increase_to_330_weekly_salary_l717_717446

noncomputable def sharonWeeklySalary : ℝ := 324 / 1.08

theorem percentage_increase_to_330_weekly_salary (S : ℝ) (X : ℝ) (h1 : S = 324 / 1.08) (h2 : X = (330 - S) / (S / 100)) :
  X = 10 :=
by
  -- Establish the initial condition for Sharon's salary
  rw h1 at h2
  sorry -- To be proven

end percentage_increase_to_330_weekly_salary_l717_717446


namespace max_real_part_of_z_w_l717_717189

noncomputable def z : ℂ := sorry
noncomputable def w : ℂ := sorry

def largest_real_part (z w : ℂ) :=
  ∃ (a b c d : ℝ), z = a + b * I ∧ w = c + d * I ∧
                   abs z = 2 ∧ abs w = 2 ∧
                   (z * conj w + conj z * w).re = 2 ∧
                   (a + c) = sqrt 10

theorem max_real_part_of_z_w :
  largest_real_part z w :=
  sorry  

end max_real_part_of_z_w_l717_717189


namespace smallest_base_10_integer_l717_717722

theorem smallest_base_10_integer (a b : ℕ) (h1 : a > 3) (h2 : b > 3) (h3 : 14 = a * 1 + 4) (h4 : 23 = 2 * b + 3) (h5 : a + 4 = 2 * b + 3) : 11 :=
begin
  sorry
end

end smallest_base_10_integer_l717_717722


namespace three_points_in_circle_of_radius_l717_717115

theorem three_points_in_circle_of_radius {points : set (ℝ × ℝ)}
  (h₁ : ∀ p ∈ points, p.1 ∈ Icc 0 1 ∧ p.2 ∈ Icc 0 1)
  (h₂ : points.to_finset.card = 51) :
  ∃ (c : ℝ × ℝ), ∃ (r : ℝ), r = 1 / 7 ∧ (∃ p₁ p₂ p₃ ∈ points, p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧ 
   dist p₁ c ≤ r ∧ dist p₂ c ≤ r ∧ dist p₃ c ≤ r) :=
by
  sorry

end three_points_in_circle_of_radius_l717_717115


namespace cube_and_reciprocal_l717_717823

theorem cube_and_reciprocal (m : ℝ) (hm : m + 1/m = 10) : m^3 + 1/m^3 = 970 := 
by
  sorry

end cube_and_reciprocal_l717_717823


namespace train_crossing_time_l717_717262

theorem train_crossing_time (length_of_train : ℝ) (speed_of_train : ℝ) (speed_of_man : ℝ) :
  length_of_train = 1500 → speed_of_train = 95 → speed_of_man = 5 → 
  (length_of_train / ((speed_of_train - speed_of_man) * (1000 / 3600))) = 60 :=
by
  intros h1 h2 h3
  have h_rel_speed : ((speed_of_train - speed_of_man) * (1000 / 3600)) = 25 := by
    rw [h2, h3]
    norm_num
  rw [h1, h_rel_speed]
  norm_num

end train_crossing_time_l717_717262


namespace means_square_sum_l717_717939

theorem means_square_sum {a b c : ℝ} 
  (h_arith : (a + b + c) / 3 = 7)
  (h_geom : (abc).nthRoot 3 = 6)
  (h_harm : 3 / ((1 / a) + (1 / b) + (1 / c)) = 5) 
  : a^2 + b^2 + c^2 = 181.8 := 
by {
  -- sorry proof skipped
  sorry
}

end means_square_sum_l717_717939


namespace inverse_89_mod_90_l717_717362

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  -- Mathematical proof is skipped
  sorry

end inverse_89_mod_90_l717_717362


namespace Ceva_theorem_l717_717916

theorem Ceva_theorem {A B C X Y Z : Point} {BC CA AB : Line} 
  (hX : X ∈ BC) (hY : Y ∈ CA) (hZ : Z ∈ AB) 
  (hConcur : concurrent (line A X) (line B Y) (line C Z)) :
  cevianRatio X B C * cevianRatio Y C A * cevianRatio Z A B = -1 :=
sorry

end Ceva_theorem_l717_717916


namespace min_value_achieved_l717_717790

noncomputable def min_x_y_value (x y : ℝ) := x + y

theorem min_value_achieved :
  ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → (x * y = 2 * x + y) → min_x_y_value x y = sqrt 2 + 1 :=
by
  intros x y hx hy hxy
  sorry

end min_value_achieved_l717_717790


namespace football_tournament_impossible_l717_717844

theorem football_tournament_impossible (n : ℕ) (h : n = 17)
  (matches_played : ∀ i : ℕ, i < n → ℕ) 
  (total_matches : ℕ := (n * (n - 1)) / 2) :
  ¬ (∀ i : ℕ, i < n → (let wins := matches_played i,
                              draws := matches_played i in
                         wins = draws ∧ wins + draws + (matches_played i - wins - draws) = n - 1)) :=
by
  sorry

end football_tournament_impossible_l717_717844


namespace multiply_and_add_pattern_l717_717816

theorem multiply_and_add_pattern :
  1_234_567 * 9 + 8 = 11_111_111 := 
sorry

end multiply_and_add_pattern_l717_717816


namespace probability_of_perfect_square_sum_l717_717583

def two_dice_probability_of_perfect_square_sum : ℚ :=
  let totalOutcomes := 12 * 12
  let perfectSquareOutcomes := 3 + 8 + 9 -- ways to get sums 4, 9, and 16
  (perfectSquareOutcomes : ℚ) / (totalOutcomes : ℚ)

theorem probability_of_perfect_square_sum :
  two_dice_probability_of_perfect_square_sum = 5 / 36 :=
by
  sorry

end probability_of_perfect_square_sum_l717_717583


namespace exists_large_enough_n_l717_717922

theorem exists_large_enough_n :
  ∃ n : ℕ, n ≥ 15884836 ∧
  (∀ (points : set (ℕ × ℕ)), points.card = 1993 * n →
    (∀ (p₁ p₂ p₃ : ℕ × ℕ), p₁ ≠ p₂ → p₁ ≠ p₃ → p₂ ≠ p₃ → p₁ ∈ points → p₂ ∈ points → p₃ ∈ points →
      ¬(is_equilateral_triangle p₁ p₂ p₃))) :=
begin
  sorry
end

def is_equilateral_triangle (p1 p2 p3 : (ℕ × ℕ)) : Prop :=
  let d1 := dist p1 p2,
      d2 := dist p2 p3,
      d3 := dist p3 p1 in
   d1 = d2 ∧ d2 = d3

noncomputable def dist (p1 p2 : (ℕ × ℕ)) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

end exists_large_enough_n_l717_717922


namespace smallest_common_multiple_l717_717742

theorem smallest_common_multiple (b : ℕ) (hb : b > 0) (h1 : b % 6 = 0) (h2 : b % 15 = 0) :
    b = 30 :=
sorry

end smallest_common_multiple_l717_717742


namespace taller_tree_height_l717_717573

variables (h : ℕ) (s : ℕ)

def taller_tree_condition_1 (h s : ℕ) : Prop :=
  h = s + 20

def tree_height_ratio (h s : ℕ) : Prop :=
  s * 7 = h * 5

theorem taller_tree_height (h s : ℕ)
  (cond1 : taller_tree_condition_1 h s)
  (cond2 : tree_height_ratio h s) :
  h = 70 :=
begin
  sorry
end

end taller_tree_height_l717_717573


namespace vector_AB_complex_l717_717201

theorem vector_AB_complex 
  (OA OB : ℂ) 
  (hOA : OA = 5 + 10 * complex.I) 
  (hOB : OB = 3 - 4 * complex.I) : 
  OB - OA = -2 - 14 * complex.I :=
by
  rw [hOA, hOB]
  simp

end vector_AB_complex_l717_717201


namespace max_has_38_quarters_l717_717153

open Int

def max_quarters : ℤ :=
  let q := 38 in
  (8 < q) ∧ (q < 60) ∧ ((q % 36 = 2) ∧ (q % 7 = 3))

theorem max_has_38_quarters : ∃ q : ℤ, max_quarters :=
  by
  use 38
  simp [max_quarters]
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  { norm_num }

end max_has_38_quarters_l717_717153


namespace Cody_reads_books_in_7_weeks_l717_717695

noncomputable def CodyReadsBooks : ℕ :=
  let total_books := 54
  let first_week_books := 6
  let second_week_books := 3
  let book_per_week := 9
  let remaining_books := total_books - first_week_books - second_week_books
  let remaining_weeks := remaining_books / book_per_week
  let total_weeks := 1 + 1 + remaining_weeks
  total_weeks

theorem Cody_reads_books_in_7_weeks : CodyReadsBooks = 7 := by
  sorry

end Cody_reads_books_in_7_weeks_l717_717695


namespace savings_increase_l717_717649

variables (I : ℝ) -- Define the income

def S := 0.35 * I -- Savings in the first year
def E := I - S -- Expenditure in the first year
def I_new := 1.35 * I -- Income in the second year
def E_new := E -- Expenditure in the second year remains the same

-- From the condition that the total expenditure in the two years is double the expenditure in the first year.
theorem savings_increase : 
  E + E_new = 2 * E →
  E = 0.65 * I →
  S = 0.35 * I →
  I_new = 1.35 * I →
  ((1.35 * I - E) - S) / S * 100 = 100 :=
by
  intros h1 h2 h3 h4
  sorry

end savings_increase_l717_717649


namespace alcohol_water_ratio_new_mixture_l717_717241

/-- Two jars contain alcohol and water in different ratios. 
 The first jar has a ratio of alcohol to water of 3:1 and contains 4 liters in total. 
 The second jar has a ratio of alcohol to water of 2:1 and contains 6 liters in total. 
 If 1 liter of mixture from the first jar and 2 liters of mixture from the second jar 
 are poured into a new container, what is the ratio of alcohol to water in this new mixture? -/
theorem alcohol_water_ratio_new_mixture :
  let alcohol1 := 3
  let water1 := 1
  let total1 := 4
  let alcohol2 := 2
  let water2 := 1
  let total2 := 6
  let mix1 := 1
  let mix2 := 2
  -- Volume of alcohol and water in each jar
  let a1 := (alcohol1 / (alcohol1 + water1)) * total1
  let w1 := (water1 / (alcohol1 + water1)) * total1
  let a2 := (alcohol2 / (alcohol2 + water2)) * total2
  let w2 := (water2 / (alcohol2 + water2)) * total2
  -- Amount of alcohol and water taken from each jar
  let ta1 := (mix1 * alcohol1) / (alcohol1 + water1)
  let tw1 := (mix1 * water1) / (alcohol1 + water1)
  let ta2 := (mix2 * alcohol2) / (alcohol2 + water2)
  let tw2 := (mix2 * water2) / (alcohol2 + water2)
  -- Total alcohol and water in new mixture
  let total_a := ta1 + ta2
  let total_w := tw1 + tw2
  -- Ratio of alcohol to water in the new mixture
  (total_a / total_w) = (41 / 19) :=
by
  -- Definitions for the components
  have : a1 = 3 := by { sorry }
  have : w1 = 1 := by { sorry }
  have : a2 = 4 := by { sorry }
  have : w2 = 2 := by { sorry }
  have : ta1 = 3 / 4 := by { sorry }
  have : tw1 = 1 / 4 := by { sorry }
  have : ta2 = 8 / 3 := by { sorry }
  have : tw2 = 4 / 3 := by { sorry }
  have : total_a = 41 / 12 := by { sorry }
  have : total_w = 19 / 12 := by { sorry }
  show total_a / total_w = 41 / 19, 
    by { sorry }


end alcohol_water_ratio_new_mixture_l717_717241


namespace probability_of_D_l717_717635

theorem probability_of_D (P : Type) (A B C D : P) 
  (pA pB pC pD : ℚ) 
  (hA : pA = 1/4) 
  (hB : pB = 1/3) 
  (hC : pC = 1/6) 
  (hSum : pA + pB + pC + pD = 1) :
  pD = 1/4 :=
by 
  sorry

end probability_of_D_l717_717635


namespace geometric_progression_odd_last_term_l717_717645

theorem geometric_progression_odd_last_term :
  ∃ (n : ℕ), (∃ (a : ℕ) (r : ℚ), a = 10 ^ 2015 ∧ ∃ k : ℤ, r = k / a ∧ (10 ^ 2015) * r = odd_integer a n) → n = 8 :=
by sorry

def odd_integer (a : ℕ) (n : ℕ) (r : ℕ) : Prop :=
  (∃ (a1 an : ℕ), an = a1 * r^(n-1) ∧ a1 = 10^2015 ∧ (an % 2 = 1))

end geometric_progression_odd_last_term_l717_717645


namespace hyperbola_eccentricity_l717_717401

noncomputable def hyperbola_focus_conditions (a b PF1 PF2 : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 3 * PF1 = 4 * PF2) : Prop :=
  ∃ F1 F2 P, (P ∈ { (x, y) | x^2 / a^2 - y^2 / b^2 = 1}) ∧
  ((P + F2 | ((O F2) • PF2) = 0) ∧
  ((PF1 - PF2) = 2 * a) ∧
  (angle P F1 F2 = 90 * (π / 180)))

theorem hyperbola_eccentricity (a b PF1 PF2 : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 3 * PF1 = 4 * PF2) :
  hyperbola_focus_conditions a b PF1 PF2 h1 h2 h3 →
  let c := 5 * a in
  (c / a = 5) :=
by
  sorry

end hyperbola_eccentricity_l717_717401


namespace smallest_multiple_of_6_and_15_l717_717757

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ c : ℕ, c > 0 ∧ c % 6 = 0 ∧ c % 15 = 0 → c ≥ b := 
begin
  use 30,
  split,
  { exact nat.succ_pos 29, },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 2 3) (dvd_mul_right 3 5)), },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 3 5) (dvd_mul_right 3 2)), },
  { intros c hc1 hc2,
    have hc3 : c % 30 = 0,
    {
      suffices h : c % 6 = 0 ∧ c % 15 = 0 ↔ c % lcm 6 15 = 0,
      { rw ← h, exact ⟨hc1, hc2⟩, },
      exact nat.dvd_iff_mod_eq_zero,
    },
    linarith,
  }
end

end smallest_multiple_of_6_and_15_l717_717757


namespace vector_sum_l717_717437

-- Define the vectors a and b according to the conditions.
def a : (ℝ × ℝ) := (2, 1)
def b : (ℝ × ℝ) := (-3, 4)

-- Prove that the vector sum a + b is (-1, 5).
theorem vector_sum : (a.1 + b.1, a.2 + b.2) = (-1, 5) :=
by
  -- include the proof later
  sorry

end vector_sum_l717_717437


namespace number_of_solid_figures_is_4_l717_717678

def is_solid_figure (shape : String) : Bool :=
  shape = "cone" ∨ shape = "cuboid" ∨ shape = "sphere" ∨ shape = "triangular prism"

def shapes : List String :=
  ["circle", "square", "cone", "cuboid", "line segment", "sphere", "triangular prism", "right-angled triangle"]

def number_of_solid_figures : Nat :=
  (shapes.filter is_solid_figure).length

theorem number_of_solid_figures_is_4 : number_of_solid_figures = 4 :=
  by sorry

end number_of_solid_figures_is_4_l717_717678


namespace total_trail_length_l717_717161

-- Definitions based on conditions
variables (a b c d e : ℕ)

-- Conditions
def condition1 : Prop := a + b + c = 36
def condition2 : Prop := b + c + d = 48
def condition3 : Prop := c + d + e = 45
def condition4 : Prop := a + d = 31

-- Theorem statement
theorem total_trail_length (h1 : condition1 a b c) (h2 : condition2 b c d) (h3 : condition3 c d e) (h4 : condition4 a d) : 
  a + b + c + d + e = 81 :=
by 
  sorry

end total_trail_length_l717_717161


namespace volume_ratio_of_spheres_l717_717388

theorem volume_ratio_of_spheres (a : ℝ) :
  let r_circumscribed := (sqrt 3 / 2) * a,
      r_inscribed := (a / 2),
      V1 := (4 / 3) * π * r_circumscribed ^ 3,
      V2 := (4 / 3) * π * r_inscribed ^ 3
  in V1 / V2 = 3 * sqrt 3 :=
by
  sorry

end volume_ratio_of_spheres_l717_717388


namespace isosceles_triangle_perimeter_l717_717574

theorem isosceles_triangle_perimeter (a b : ℕ)
  (h_eqn : ∀ x : ℕ, (x - 4) * (x - 2) = 0 → x = 4 ∨ x = 2)
  (h_isosceles : ∃ a b : ℕ, (a = 4 ∧ b = 2) ∨ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 4)) :
  a + a + b = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l717_717574


namespace min_area_of_right_triangle_l717_717251

theorem min_area_of_right_triangle : 
  ∃ (a b : ℝ), (a * b) = 8 ∧
  (∃ (k d : ℝ), y = k * x + d ∧ (1, 3) ∈ hypotenuse ∧
  line_y_equals_x ∧ line_y_equals_negative_x)
:= sorry

end min_area_of_right_triangle_l717_717251


namespace find_z_l717_717821

variable (z : Complex)

theorem find_z (h : conj(z) * (1 - Complex.i)^2 = 4 + 2 * Complex.i) : 
  z = -1 - 2 * Complex.i := 
sorry

end find_z_l717_717821


namespace turtle_reaches_waterhole_in_28_minutes_l717_717613

-- Definitions
constant x : ℝ
constant turtle_speed : ℝ := 1 / 30
constant lion1_time_to_waterhole : ℝ := 5
constant turtle_time_to_waterhole : ℝ := 30

-- Speeds of the Lion Cubs
def lion1_speed := x
def lion2_speed := 1.5 * x

-- Time for the lion cubs to meet
def meeting_time := lion1_time_to_waterhole / (1 + lion2_speed / lion1_speed)

-- Distance traveled by the turtle in the meeting time
def turtle_distance_covered := turtle_speed * meeting_time

-- Remaining distance for the turtle
def remaining_turtle_distance := 1 - turtle_distance_covered

-- Time for the turtle to cover the remaining distance
def turtle_remaining_time := remaining_turtle_distance * 30

-- Prove that the turtle takes 28 minutes after the meeting to reach the waterhole
theorem turtle_reaches_waterhole_in_28_minutes : turtle_remaining_time = 28 :=
by
  -- Placeholder for the actual proof
  sorry

end turtle_reaches_waterhole_in_28_minutes_l717_717613


namespace least_integer_value_y_l717_717591

theorem least_integer_value_y (y : ℤ) (h : abs (3 * y - 4) ≤ 25) : y = -7 :=
sorry

end least_integer_value_y_l717_717591


namespace stationery_sales_l717_717287

theorem stationery_sales (total_sales : ℕ) (fabric_fraction jewelry_fraction : ℚ) (h_total : total_sales = 36)
  (h_fabric : fabric_fraction = 1 / 3) (h_jewelry : jewelry_fraction = 1 / 4) : total_sales - (fabric_fraction * total_sales).natAbs - (jewelry_fraction * total_sales).natAbs = 15 := by {
  unfold fabric_fraction jewelry_fraction,
  rw [h_total, h_fabric, h_jewelry],
  norm_num,
  sorry
}

end stationery_sales_l717_717287


namespace intervals_and_minimum_value_of_g_l717_717892

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := f x + (Real.deriv f) x

theorem intervals_and_minimum_value_of_g :
  (∀ x > 0, (0 < x ∧ x < 1 → g x < g 1) ∧ (x > 1 → g x > g 1)) ∧ g 1 = 1 :=
by
  sorry

end intervals_and_minimum_value_of_g_l717_717892


namespace larger_lateral_side_length_l717_717114

theorem larger_lateral_side_length (A B C D : ℝ) (h1 : ∠ A = 45) (h2 : BD = 20) (h3 : DC = 21) : BC = 41 :=
by sorry

end larger_lateral_side_length_l717_717114


namespace well_depth_l717_717283

noncomputable def depth_of_well (diameter volume : ℝ) : ℝ :=
  let radius := diameter / 2
  depth := volume / (Real.pi * radius^2)
  depth

theorem well_depth :
  depth_of_well 4 175.92918860102841 ≈ 14 := by
  sorry

end well_depth_l717_717283


namespace average_speed_l717_717998

/-- Prove that the average speed of a man who travels up and down an altitude of 200 meters,
    with speeds of 16 km/hr up and 30 km/hr down, is approximately 20.86 km/hr. -/
theorem average_speed {distance_up distance_down speed_up speed_down : ℝ}
  (h1 : distance_up = 0.2) (h2 : distance_down = 0.2)
  (h3 : speed_up = 16) (h4 : speed_down = 30) :
  (distance_up + distance_down) / (distance_up / speed_up + distance_down / speed_down) ≈ 20.86 :=
sorry

end average_speed_l717_717998


namespace monotone_if_derivative_pos_l717_717792

theorem monotone_if_derivative_pos (D : Set ℝ) (f g : ℝ → ℝ) (g' : ℝ → ℝ) :
  (∀ x ∈ D, f' x > 0) →
  (∀ x > 0, g' x = 2 * x) →
  (∀ x > 0, g' x > 0) →
  MonotoneOn g (Set.Ioi 0) :=
by
  sorry

end monotone_if_derivative_pos_l717_717792


namespace distance_one_minute_before_collision_l717_717584

-- Define the initial conditions of the problem
def boat1_speed := 4 -- Boat 1 speed in miles/hr
def boat2_speed := 20 -- Boat 2 speed in miles/hr
def initial_distance := 20 -- Initial distance between the boats in miles
def combined_speed := boat1_speed + boat2_speed -- Combined speed in miles/hr
def combined_speed_mpm := combined_speed / 60 -- Convert combined speed to miles/minute
def collision_time := initial_distance / combined_speed_mpm -- Time until collision in minutes

-- The goal is to prove the distance between the boats one minute before they collide is 0.4 miles.
theorem distance_one_minute_before_collision : (initial_distance - (combined_speed_mpm * (collision_time - 1))) = 0.4 := 
by
  sorry

end distance_one_minute_before_collision_l717_717584


namespace simplify_power_of_product_l717_717179

theorem simplify_power_of_product (x : ℝ) : (5 * x^2)^4 = 625 * x^8 :=
by
  sorry

end simplify_power_of_product_l717_717179


namespace vertical_axis_residuals_of_residual_plot_l717_717856

theorem vertical_axis_residuals_of_residual_plot :
  ∀ (vertical_axis : Type), 
  (vertical_axis = Residuals ∨ 
   vertical_axis = SampleNumber ∨ 
   vertical_axis = EstimatedValue) →
  (vertical_axis = Residuals) :=
by
  sorry

end vertical_axis_residuals_of_residual_plot_l717_717856


namespace modulus_power_eight_l717_717365

theorem modulus_power_eight :
  abs ((1 / 2 + (Real.sqrt 3 / 2) * Complex.I) ^ 8) = 1 := 
sorry

end modulus_power_eight_l717_717365


namespace part1_part2_l717_717713

open Real

-- Definitions from problem conditions
def square_recursive_seq (A : ℕ → ℝ) : Prop :=
  ∀ n, A (n+1) = A n ^ 2

def a_n (n : ℕ) : ℝ
| 0       => 2
| (n + 1) => 2 * (a_n n) ^ 2 + 2 * (a_n n)

-- Part 1: Prove {2a_n + 1} is a square recursive sequence and log2(2a_n + 1) is geometric
def b_n (n : ℕ) : ℝ := 2 * a_n n + 1

theorem part1 :
  square_recursive_seq (λ n, 2 * a_n n + 1) ∧
  geometric_seq (λ n, log2 (2 * a_n n + 1)) (2 : ℝ) :=
by
  sorry

-- Part 2: Find the general term of the sequence {a_n} and T_n
noncomputable def a_n_general_term (n : ℕ) : ℝ :=
  1 / 2 * (5 ^ (2 ^ (n - 1)) - 1)

noncomputable def T_n (n : ℕ) : ℝ :=
  ∏ i in finset.range (n + 1), 2 * a_n i + 1

theorem part2 (n : ℕ) :
  a_n n = 1 / 2 * (5 ^ (2 ^ (n - 1)) - 1) ∧ T_n n = 5 ^ (2 ^ n - 1) :=
by
  sorry

end part1_part2_l717_717713


namespace no_valid_n_l717_717019

def isThreeDigit (k : ℤ) : Prop := 100 ≤ k ∧ k ≤ 999

theorem no_valid_n :
  ∀ n : ℕ, (isThreeDigit (n / 4) ∧ isThreeDigit (4 * n)) → False :=
by 
  intro n
  intro h
  obtain ⟨h1, h2⟩ := h
  have h3 : n ≥ 400 := by linarith
  have h4 : n ≤ 249 := by linarith
  exact nat.not_le_of_gt h3 h4

end no_valid_n_l717_717019


namespace find_a_plus_b_plus_c_l717_717854

def is_equilateral (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def trisects (D E : Point) (B C : LineSegment) : Prop :=
  ∃ p q : ℚ, 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p < q ∧
  D = B + p • (C - B) ∧ E = B + q • (C - B)

def sin_angle_DAE (A D E : Point) : ℚ := sorry

theorem find_a_plus_b_plus_c :
  ∀ (A B C D E : Point),
  is_equilateral A B C →
  trisects D E (line_segment B C) →
  ∃ (a b c : ℕ) (sqrtb : ℚ),
  sin_angle_DAE A D E = (a * sqrtb) / c ∧
  a.gcd c = 1 ∧
  sqrtb * sqrtb = b ∧ ¬ ∃ (p : ℕ), p^2 ∣ b ∧ 1 < p ∧
  a + b + c = 20 := 
by 
  sorry

end find_a_plus_b_plus_c_l717_717854


namespace solve_for_k_l717_717447

theorem solve_for_k : 
  ∃ (k : ℕ), k > 0 ∧ k * k = 2012 * 2012 + 2010 * 2011 * 2013 * 2014 ∧ k = 4048142 :=
sorry

end solve_for_k_l717_717447


namespace can_make_all_zeros_l717_717855

-- Define the structure of a board
structure Board where
  width : ℕ
  height : ℕ
  cells : Fin height → Fin width → ℕ

-- Define the allowed operations on the board
inductive Operation
  | doubleRow (row : Fin (Board.height)) : Operation
  | decrementCol (col : Fin (Board.width)) : Operation

-- Define the application of operation on the board
def applyOperation (b : Board) : Operation → Board
  | Operation.doubleRow row =>
      { b with cells := λ i j => if i = row then 2 * b.cells i j else b.cells i j }
  | Operation.decrementCol col =>
      { b with cells := λ i j => if j = col then b.cells i j - 1 else b.cells i j }

-- The theorem we need to prove
theorem can_make_all_zeros (b : Board) : ∃ (ops : List Operation), 
  let final_board := ops.foldl (applyOperation b) in
  (∀ i j, final_board.cells i j = 0) :=
  sorry

end can_make_all_zeros_l717_717855


namespace isosceles_triangle_third_side_13_l717_717470

theorem isosceles_triangle_third_side_13 (a b : ℝ) (h₁ : a = 13 ∨ a = 6) (h₂ : b = 13 ∨ b = 6) (h_iso : h₁ ≠ h₂) (h_iso_triangle : (a = 13 ∧ b = 13 ∨ a = 13 ∧ b = 6 ∨ a = 6 ∧ b = 13)) :
  ∃ c, c = 13 :=
by sorry

end isosceles_triangle_third_side_13_l717_717470


namespace total_dots_is_78_l717_717321

-- Define the conditions as Lean definitions
def ladybugs_monday : ℕ := 8
def ladybugs_tuesday : ℕ := 5
def dots_per_ladybug : ℕ := 6

-- Define the total number of ladybugs
def total_ladybugs : ℕ := ladybugs_monday + ladybugs_tuesday

-- Define the total number of dots
def total_dots : ℕ := total_ladybugs * dots_per_ladybug

-- Theorem stating the problem to solve
theorem total_dots_is_78 : total_dots = 78 := by
  sorry

end total_dots_is_78_l717_717321


namespace distance_from_focus_to_asymptote_l717_717947

def parabola_focus (x : ℝ) : ℝ × ℝ :=
(1, 0)

def hyperbola_asymptote_1 (x : ℝ) : ℝ :=
2 * x

def hyperbola_asymptote_2 (x : ℝ) : ℝ :=
-2 * x

def distance_point_line (point : ℝ × ℝ) (line : ℝ → ℝ) : ℝ :=
  let (x1, y1) := point
  let A := -line 1  -- coefficient of x in line equation
  let B := 1       -- coefficient of y in line equation
  let C := 0       -- constant term in line equation
  abs (A * x1 + B * y1 + C) / sqrt (A^2 + B^2)

theorem distance_from_focus_to_asymptote :
  distance_point_line (parabola_focus 1) hyperbola_asymptote_1 = (2 * sqrt 5) / 5 := by
  sorry

end distance_from_focus_to_asymptote_l717_717947


namespace cannot_make_all_cells_same_color_l717_717623

-- Define the problem conditions
def chessboard_coloring (n : ℕ) : Prop :=
  ∃ color : ℕ × ℕ → bool,
    ∀ x y : ℕ, 1 ≤ x ∧ x ≤ n → 1 ≤ y ∧ y ≤ n →
      color (x+1, y) ≠ color (x, y) ∧ color (x, y+1) ≠ color (x, y)

def toggle_rectangle (color : ℕ × ℕ → bool) : ℕ × ℕ → (ℕ × ℕ → bool) :=
  λ (i, j),
    λ (x, y), if (i ≤ x ∧ x < i+2) ∧ (j ≤ y ∧ y < j+3) then ¬color (x, y) else color (x, y)

-- Prove that all cells cannot be made the same color with the given moves
theorem cannot_make_all_cells_same_color :
  ∀ (n : ℕ), 200 = n →
  chessboard_coloring n →
  ¬ (∃ (color : ℕ × ℕ → bool),
    ∀ (moves : (ℕ × ℕ → bool) → ℕ × ℕ → (ℕ × ℕ → bool)),
      moves (λ _, false) = color ∨ moves (λ _, true) = color) :=
begin
  intros n hn hc,
  rw nat.eq at hn,
  sorry
end

end cannot_make_all_cells_same_color_l717_717623


namespace parabola_focus_distance_eq_18x_l717_717053

noncomputable def parabola_equation (p : ℝ) : Π (x y : ℝ), Prop :=
λ x y, y^2 = 2 * p * x

theorem parabola_focus_distance_eq_18x (p : ℝ) (h : 0 < p) (t : ℝ) :
  (∀ (x y : ℝ), parabola_equation p x y ↔ y^2 = 2 * p * x) →
  ∃ (F M : ℝ × ℝ),
    F = (p / 2, 0) ∧ 
    M = (4, t) ∧
    |fst M - fst F| = (5:ℝ) →
    parabola_equation 9 =
      (λ x y, y^2 = 18 * x) :=
by
  intros parabola_def
  cases F with Fx Fy
  cases M with Mx My
  have : x = 9 := sorry
  cases this

  sorry

end parabola_focus_distance_eq_18x_l717_717053


namespace num_bikes_l717_717845

variable (C B : ℕ)

-- The given conditions
def num_cars : ℕ := 10
def num_wheels_total : ℕ := 44
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

-- The mathematical proof problem statement
theorem num_bikes :
  C = num_cars →
  B = ((num_wheels_total - (C * wheels_per_car)) / wheels_per_bike) →
  B = 2 :=
by
  intros hC hB
  rw [hC] at hB
  sorry

end num_bikes_l717_717845


namespace abc_product_l717_717061

theorem abc_product :
  ∃ (a b c P : ℕ), 
    b + c = 3 ∧ 
    c + a = 6 ∧ 
    a + b = 7 ∧ 
    P = a * b * c ∧ 
    P = 10 :=
by sorry

end abc_product_l717_717061


namespace max_min_diff_f_l717_717065

noncomputable def f (x : ℝ) : ℝ := 3 - Math.sin x - 2 * (Math.cos x)^2

theorem max_min_diff_f :
  let I := set.Icc (Real.pi / 6) (7 * Real.pi / 6) in
  (real_max_on f I - real_min_on f I) = 9 / 8 :=
by
  have h1 : ∃ x ∈ I, ∀ y ∈ I, f y ≤ f x := Exists.intro sorry sorry
  have h2 : ∃ x ∈ I, ∀ y ∈ I, f x ≤ f y := Exists.intro sorry sorry
  let f_max := classical.some h1
  let f_min := classical.some h2
  have : (λ x => 3 - Math.sin x - 2 * (Math.cos x)^2) = 
    (λ x => 2 * (Math.sin x)^2 - Math.sin x + 1) := sorry
  have f_max_val := f f_max
  have f_min_val := f f_min
  have : f_max_val = 2 := sorry
  have : f_min_val = 7 / 8 := sorry
  exact calc
    (f_max_val - f_min_val) = (2 - 7 / 8) : by congr
    ... = 9 / 8 : by norm_num 

end max_min_diff_f_l717_717065


namespace tan_angle_QDE_l717_717528

theorem tan_angle_QDE :
  ∃ (Q : Point) (φ : ℝ),
    (Q ∈ interior (triangle DEF)) ∧
    (DE.length = 10) ∧
    (EF.length = 12) ∧
    (FD.length = 16) ∧
    (∠QDE = φ) ∧
    (∠QEF = φ) ∧
    (∠QFD = φ) ∧
    (tan φ = 48 / 125) :=
sorry

end tan_angle_QDE_l717_717528


namespace length_of_AB_l717_717350

open Real EuclideanGeometry

variables {A B C : Point}
variables (h : Triangle A B C)
variable (h_right : ∠BAC = π / 6)
variable (BC_length : dist B C = 18)

theorem length_of_AB :
  dist A B = 9 :=
by  
  sorry

end length_of_AB_l717_717350


namespace correct_statements_l717_717785

noncomputable def arithmeticSequenceSum (a d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

theorem correct_statements (a d : ℝ) :
  let S := arithmeticSequenceSum a d
  in S 6 > S 7 → S 7 > S 5 → (d < 0 ∧ ¬(S 11 > 0) ∧ ¬(S 12 < 0) ∧ ¬(∀ n : ℕ, S n > 0 → n ≤ 13)) :=
by
  intros h1 h2
  have statement1 : d < 0 := sorry
  have statement2 : ¬(S 11 > 0) := sorry
  have statement3 : ¬(S 12 < 0) := sorry
  have statement4 : ¬(∀ n : ℕ, S n > 0 → n ≤ 13) := sorry
  exact ⟨statement1, statement2, statement3, statement4⟩

end correct_statements_l717_717785


namespace like_terms_exponents_l717_717457

theorem like_terms_exponents (m n : ℤ) 
  (h1 : 3 = m - 2) 
  (h2 : n + 1 = 2) : m - n = 4 := 
by
  sorry

end like_terms_exponents_l717_717457


namespace parameter_values_for_three_distinct_roots_l717_717001

theorem parameter_values_for_three_distinct_roots (a : ℝ) :
  (∀ x : ℝ, (|x^3 - a^3| = x - a) → (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) ↔ 
  (-2 / Real.sqrt 3 < a ∧ a < -1 / Real.sqrt 3) :=
sorry

end parameter_values_for_three_distinct_roots_l717_717001


namespace correct_password_contains_1_and_7_l717_717294

/-- 
A mobile phone user entered a four-digit password incorrectly four times in a row but got it right 
on the fifth attempt. In each of the first four attempts, two digits were correct but were placed 
in the wrong positions. The passwords entered for the first four attempts were 3406, 1630, 7364, 
and 6173 respectively. Prove that the correct password contains the digits 1 and 7.
-/
theorem correct_password_contains_1_and_7 
  (attempt1 attempt2 attempt3 attempt4 : list ℤ)
  (correct : list ℤ)
  (h1 : attempt1 = [3, 4, 0, 6])
  (h2 : attempt2 = [1, 6, 3, 0])
  (h3 : attempt3 = [7, 3, 6, 4])
  (h4 : attempt4 = [6, 1, 7, 3])
  (H : ∀ (a : list ℤ), a ∈ [attempt1, attempt2, attempt3, attempt4] → ∃ i j, i ≠ j ∧
          (a[i] = correct[j] ∧ a[j] = correct[i]) ∧ correct !i ∧ correct !j) :
  1 ∈ correct ∧ 7 ∈ correct := sorry

end correct_password_contains_1_and_7_l717_717294


namespace find_rate_per_kg_mangoes_l717_717330

-- Definitions based on the conditions
def rate_per_kg_grapes : ℕ := 70
def quantity_grapes : ℕ := 8
def total_payment : ℕ := 1000
def quantity_mangoes : ℕ := 8

-- Proposition stating what we want to prove
theorem find_rate_per_kg_mangoes (r : ℕ) (H : total_payment = (rate_per_kg_grapes * quantity_grapes) + (r * quantity_mangoes)) : r = 55 := sorry

end find_rate_per_kg_mangoes_l717_717330


namespace find_a_for_continuity_of_f_l717_717900

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 3 then 3 * x^2 + 2 else a * x - 1

theorem find_a_for_continuity_of_f :
  (∀ a : ℝ, continuous (f x a)) → (a = 10) :=
sorry

end find_a_for_continuity_of_f_l717_717900


namespace part1_part2_l717_717066

def f (x : ℝ) : ℝ := Real.log (1 + x) + x^2 / 2
def g (x : ℝ) : ℝ := Real.cos x + x^2 / 2

theorem part1 (x : ℝ) (hx : 0 ≤ x) : f x ≥ x :=
by
  sorry

theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : f (Real.exp (a / 2)) = g b - 1) : f (b^2) + 1 > g (a + 1) :=
by
  sorry

end part1_part2_l717_717066


namespace angle_ABD_is_90_l717_717582

-- Define the necessary geometric concepts
variables {A B C D O : Type} [InCirleCircle A B C ω] 
[IsTangents ω A C D]
[IsAngle A B C 135]
[Bisector AB CD]

theorem angle_ABD_is_90 :
    ∠ABD = 90 :=
begin
  sorry
end

end angle_ABD_is_90_l717_717582


namespace tetromino_tiling_impossible_l717_717665

-- Define what a tetromino is in informal terms for this statement
def Tetromino : Type := sorry

-- The main proof statement
theorem tetromino_tiling_impossible :
  (∃ m : ℕ, m = 7 ∧
  ∃ T : Set Tetromino, T.card = 7 ∧
  ∀ t ∈ T, is_tetromino t) →
  ¬ (∃ tiling : List (ℕ × ℕ) → Tetromino, (∀ pos, pos ∈ tiling) → covers_tiling (4, 7) tiling) :=
by
  sorry

end tetromino_tiling_impossible_l717_717665


namespace range_of_s_l717_717718

open Set Real

def s (x : ℝ) : ℝ := 1 / (2 - x) ^ 3

theorem range_of_s :
  (range s) = (Iio 0 ∪ Ioi 0) :=
sorry

end range_of_s_l717_717718


namespace find_2017th_term_l717_717323

def ordered_fractions_sequence : List (ℕ × ℕ) :=
  List.join (List.map (λ d, List.zip (List.range d) (List.repeat (d, d-1))) (List.rangeFrom 2))

def a_n (n : ℕ) : ℕ × ℕ :=
  (ordered_fractions_sequence.get? (n - 1)).getOrElse (0, 0)

theorem find_2017th_term :
  a_n 2017 = (1, 65) :=
by
  sorry

end find_2017th_term_l717_717323


namespace no_fraternity_member_is_club_member_thm_l717_717680

-- Definitions from the conditions
variable (Person : Type)
variable (Club : Person → Prop)
variable (Honest : Person → Prop)
variable (Student : Person → Prop)
variable (Fraternity : Person → Prop)

-- Hypotheses from the problem statements
axiom all_club_members_honest (p : Person) : Club p → Honest p
axiom some_students_not_honest : ∃ p : Person, Student p ∧ ¬ Honest p
axiom no_fraternity_member_is_club_member (p : Person) : Fraternity p → ¬ Club p

-- The theorem to be proven
theorem no_fraternity_member_is_club_member_thm : 
  ∀ p : Person, Fraternity p → ¬ Club p := 
by 
  sorry

end no_fraternity_member_is_club_member_thm_l717_717680


namespace maximum_possible_marked_cells_l717_717174

theorem maximum_possible_marked_cells (board_size : ℕ) (valid_marking : (Fin board_size → Fin board_size → Prop))
    (knight_moves : (Fin board_size × Fin board_size) → (Fin board_size × Fin board_size) → Prop) : 
  board_size = 14 →
  (∀ i j k l, i ≠ k ∧ j ≠ l → valid_marking i j → valid_marking k l → 
                knight_moves (i, j) (k, l) ∨ knight_moves (k, l) (i, j)) →
  ∃ (max_marked_cells : ℕ), max_marked_cells = 13 ∧
    (∀ count, (∃ f : Fin count → (Fin board_size × Fin board_size), 
                (∀ i j, i ≠ j → (f i).fst ≠ (f j).fst ∧ (f i).snd ≠ (f j).snd) ∧
                (∀ i j, i ≠ j → (knight_moves (f i) (f j) ∨ knight_moves (f j) (f i))))
              → count ≤ max_marked_cells) :=
by
  intros board_size valid_marking knight_moves board_size_eq move_constraint
  use 13
  split
  sorry
  sorry

end maximum_possible_marked_cells_l717_717174


namespace product_of_diagonals_is_96_l717_717881

structure Quadrilateral (α : Type _) :=
(A B C D : α)

def midpoint {α : Type _} [Add α] [DivisionRing α] (p q : α) : α :=
(p + q) / 2

def isRectangle {α : Type _} [MetricSpace α] (Q : Quadrilateral α) (w l : ℝ) : Prop :=
let d1 := dist Q.A Q.B in
let d2 := dist Q.B Q.C in
(d1, d2) = (w, l) ∨ (d1, d2) = (l, w)

def isMidpointsQuadrilateral {α : Type _} [Add α] [DivisionRing α] [MetricSpace α]
(Q1 Q2 : Quadrilateral α) : Prop :=
(Q2.A = midpoint Q1.A Q1.B) ∧
(Q2.B = midpoint Q1.B Q1.C) ∧
(Q2.C = midpoint Q1.C Q1.D) ∧
(Q2.D = midpoint Q1.D Q1.A)

noncomputable def diagonalProduct {α : Type _} [MetricSpace α] (Q : Quadrilateral α) : ℝ :=
let d1 := dist Q.A Q.C in
let d2 := dist Q.B Q.D in
d1 * d2

theorem product_of_diagonals_is_96
  {α : Type _} [Add α] [DivisionRing α] [MetricSpace α]
  (Q : Quadrilateral α)
  (Q1 : Quadrilateral α)
  (Q2 : Quadrilateral α)
  (h_midpoints_Q1 : isMidpointsQuadrilateral Q Q1)
  (h_midpoints_Q2 : isMidpointsQuadrilateral Q1 Q2)
  (h_rectangle : isRectangle Q2 4 6) :
  diagonalProduct Q = 96 := sorry

end product_of_diagonals_is_96_l717_717881


namespace calculate_f_at_2_l717_717804

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem calculate_f_at_2
  (a b : ℝ)
  (h_extremum : 3 + 2 * a + b = 0)
  (h_f1 : f 1 a b = 10) :
  f 2 a b = 18 :=
sorry

end calculate_f_at_2_l717_717804


namespace intersection_count_l717_717030

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then abs x else abs (x % 2)

def log_base_7 (x : ℝ) : ℝ := log x / log 7

theorem intersection_count :
  (∃ x1 x2 x3 x4 x5 x6 : ℝ, 
     x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x1 ≠ x6 ∧
     x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x2 ≠ x6 ∧
     x3 ≠ x4 ∧ x3 ≠ x5 ∧ x3 ≠ x6 ∧
     x4 ≠ x5 ∧ x4 ≠ x6 ∧
     x5 ≠ x6 ∧
     f x1 = log_base_7 x1 ∧
     f x2 = log_base_7 x2 ∧
     f x3 = log_base_7 x3 ∧
     f x4 = log_base_7 x4 ∧
     f x5 = log_base_7 x5 ∧
     f x6 = log_base_7 x6) :=
sorry

end intersection_count_l717_717030


namespace bacteria_distribution_impossible_l717_717520

theorem bacteria_distribution_impossible
  (B_w : ℕ) (B_b : ℕ) 
  (h_initial : B_w - B_b = 1 ∨ B_w - B_b = -1) 
  (h_change : ∀ n : ℕ, B_w - B_b = 1 + 3 * n ∨ B_w - B_b = -1 + 3 * n) : 
  ¬(B_w = B_b) := 
by 
  sorry

end bacteria_distribution_impossible_l717_717520


namespace sequence_length_arithmetic_sequence_l717_717715

theorem sequence_length_arithmetic_sequence :
  ∀ (a d l n : ℕ), a = 5 → d = 3 → l = 119 → l = a + (n - 1) * d → n = 39 :=
by
  intros a d l n ha hd hl hln
  sorry

end sequence_length_arithmetic_sequence_l717_717715


namespace correct_statements_l717_717171

def time_spent : List ℝ := [2, 5, 7, 10, 12, 13, 14, 17, 20]
def acceptance_ability : List ℝ := [47.8, 53.5, 56.3, 59, 59.8, 59.9, 59.8, 58.3, 55]

theorem correct_statements :
  (∃ x, x ∈ time_spent ∧ 59.8 ∈ acceptance_ability ∧ ((x = 12) ∨ (x = 14))) ∧
  (∀ x y, x ∈ time_spent → y ∈ acceptance_ability → (independent_var x) ∧ (dependent_var y)) ∧
  (∃ x, x = 13 ∧ x ∈ time_spent ∧ acceptance_ability.nth (time_spent.indexOf 13) = some 59.9) ∧
  (∀ x, 2 ≤ x ∧ x ≤ 13 → ∃ y, y ∈ acceptance_ability ∧ (if x1 < x2 then acceptance_ability.nth (time_spent.indexOf x1) ≤ acceptance_ability.nth (time_spent.indexOf x2))) :=
sorry

def independent_var (x : ℝ) : Prop :=
x ∈ time_spent

def dependent_var (y : ℝ) : Prop :=
y ∈ acceptance_ability

end correct_statements_l717_717171


namespace convert_to_polar_coordinates_l717_717705

theorem convert_to_polar_coordinates : 
  let x := 2 * Real.sqrt 3
  let y := -2
  let r := Real.sqrt (x^2 + y^2)
  let theta := if y < 0 then 2 * Real.pi + Real.arctan (y / x) else Real.arctan (y / x)
  r > 0 ∧ 0 ≤ theta ∧ theta < 2 * Real.pi ∧ r = 4 ∧ theta = 11 * Real.pi / 6 := 
by
  let x := 2 * Real.sqrt 3
  let y := -2
  let r := Real.sqrt (x^2 + y^2)
  let theta := if y < 0 then 2 * Real.pi + Real.arctan (y / x) else Real.arctan (y / x)
  have r_pos : r > 0 := by sorry
  have theta_in_range : 0 ≤ theta ∧ theta < 2 * Real.pi := by sorry
  have r_correct : r = 4 := by sorry
  have theta_correct : theta = 11 * Real.pi / 6 := by sorry
  exact ⟨r_pos, theta_in_range.1, theta_in_range.2, r_correct, theta_correct⟩

end convert_to_polar_coordinates_l717_717705


namespace sum_of_first_n_b_terms_l717_717076

-- Define the sequences a_n and b_n according to given conditions
def sequence_a (n : ℕ) : ℕ :=
  if n = 1 then
    ((2 : ℕ) : ℕ)
  else
    2^(n-1)

def sequence_b : ℕ → ℕ
| 1       := 3
| (n + 1) := (sequence_a n) + (sequence_b n)

def sum_b_first_n_terms (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence_b

-- State the theorem to be proved
theorem sum_of_first_n_b_terms (n : ℕ) : sum_b_first_n_terms n = 2^n + 2 * n - 1 :=
sorry

end sum_of_first_n_b_terms_l717_717076


namespace projection_calculation_l717_717703

theorem projection_calculation (u : ℝ × ℝ × ℝ)
    (h : ∃ k : ℝ, k * u = (⟨-2, 1, -1.5⟩ : ℝ × ℝ × ℝ))
    : 
    let v := (⟨1, 4, -3⟩ : ℝ × ℝ × ℝ) in 
    let v_proj_u := (⟨-1.7931, 0.8966, -1.3448⟩ : ℝ × ℝ × ℝ) in 
    (∃ α : ℝ, α * u = v_proj_u) :=
sorry

end projection_calculation_l717_717703


namespace inequality_solution_set_contains_three_integers_l717_717809

theorem inequality_solution_set_contains_three_integers 
  (a : ℝ) 
  (x : ℝ) 
  (h_inequality : (2 * x - 1) ^ 2 < a * x ^ 2) 
  (h_three_integers : {x : ℝ | (2 * x - 1) ^ 2 < a * x ^ 2}.size = 3) :
  a ∈ set.Ioo (25 / 9) (49 / 16 + 1) :=
sorry

end inequality_solution_set_contains_three_integers_l717_717809


namespace solve_for_x_l717_717930

variable (x : ℝ)
axiom h : 3 / 4 + 1 / x = 7 / 8

theorem solve_for_x : x = 8 :=
by
  sorry

end solve_for_x_l717_717930


namespace solve_for_x_l717_717928

theorem solve_for_x (x : ℝ) (h : 3 / 4 + 1 / x = 7 / 8) : x = 8 :=
by
  sorry

end solve_for_x_l717_717928


namespace find_t_l717_717808

noncomputable def hyperbola_foci_positions (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 12 = 1

noncomputable def point_on_right_branch (M : ℝ × ℝ) (F2 : ℝ × ℝ) (O : ℝ × ℝ) : Prop :=
  let OM := (M.1 - O.1, M.2 - O.2)
  let OF2 := (F2.1 - O.1, F2.2 - O.2)
  let F2M := (M.1 - F2.1, M.2 - F2.2)
  ((OM.1 + OF2.1, OM.2 + OF2.2) • (F2M.1, F2M.2)) = 0

noncomputable def distance_relationship (M F1 F2 : ℝ × ℝ) (t : ℝ) : Prop :=
  let F1M := (M.1 - F1.1, M.2 - F1.2)
  let F2M := (M.1 - F2.1, M.2 - F2.2)
  (F1M.1^2 + F1M.2^2) = t * (F2M.1^2 + F2M.2^2)

theorem find_t (M F1 F2 : ℝ × ℝ) (𝓟 : ℝ → Prop) :
  hyperbola_foci_positions (F1.1, F1.2) (M.1, M.2) →
  point_on_right_branch M F2 (0, 0) →
  distance_relationship M F1 F2 𝓟 →
  𝓝 t = 3 := sorry

end find_t_l717_717808


namespace commute_proof_l717_717997

noncomputable def commute_problem : Prop :=
  let d : ℝ := 1.5 -- distance in miles
  let v_w : ℝ := 3 -- walking speed in miles per hour
  let v_t : ℝ := 20 -- train speed in miles per hour
  let walking_minutes : ℝ := (d / v_w) * 60 -- walking time in minutes
  let train_minutes : ℝ := (d / v_t) * 60 -- train time in minutes
  ∃ x : ℝ, walking_minutes = train_minutes + x + 25 ∧ x = 0.5

theorem commute_proof : commute_problem :=
  sorry

end commute_proof_l717_717997


namespace solution_set_of_xf_gt_0_l717_717644

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_ineq : ∀ x : ℝ, x > 0 → f x < x * (deriv f x)
axiom f_at_one : f 1 = 0

theorem solution_set_of_xf_gt_0 : {x : ℝ | x * f x > 0} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end solution_set_of_xf_gt_0_l717_717644


namespace cone_base_radius_l717_717956

theorem cone_base_radius
  (semicircle_radius : ℝ)
  (h₁ : semicircle_radius = 2) :
  let base_circumference := 2 * real.pi in
  let base_radius := 1 in
  2 * real.pi * base_radius = base_circumference :=
by
  let base_radius := 1
  have h₂ : base_circumference = 2 * real.pi, from rfl
  sorry

end cone_base_radius_l717_717956


namespace percentage_failed_in_Hindi_l717_717852

-- Let Hindi_failed denote the percentage of students who failed in Hindi.
-- Let English_failed denote the percentage of students who failed in English.
-- Let Both_failed denote the percentage of students who failed in both Hindi and English.
-- Let Both_passed denote the percentage of students who passed in both subjects.

variables (Hindi_failed English_failed Both_failed Both_passed : ℝ)
  (H_condition1 : English_failed = 44)
  (H_condition2 : Both_failed = 22)
  (H_condition3 : Both_passed = 44)

theorem percentage_failed_in_Hindi:
  Hindi_failed = 34 :=
by 
  -- Proof goes here
  sorry

end percentage_failed_in_Hindi_l717_717852


namespace Cody_reads_books_in_7_weeks_l717_717696

noncomputable def CodyReadsBooks : ℕ :=
  let total_books := 54
  let first_week_books := 6
  let second_week_books := 3
  let book_per_week := 9
  let remaining_books := total_books - first_week_books - second_week_books
  let remaining_weeks := remaining_books / book_per_week
  let total_weeks := 1 + 1 + remaining_weeks
  total_weeks

theorem Cody_reads_books_in_7_weeks : CodyReadsBooks = 7 := by
  sorry

end Cody_reads_books_in_7_weeks_l717_717696


namespace calculation_correct_l717_717687

theorem calculation_correct : (35 / (8 + 3 - 5) - 2) * 4 = 46 / 3 := by
  sorry

end calculation_correct_l717_717687


namespace volume_of_larger_cube_is_125_l717_717974

theorem volume_of_larger_cube_is_125 (ratio : ℝ) (s_vol : ℝ) :
  ratio = 4.999999999999999 → s_vol = 1 →
  let s := real.cbrt s_vol in 
  let larger_edge := ratio * s in
  let larger_vol := larger_edge ^ 3 in
  larger_vol = 125 :=
by
  intros h_ratio h_s_vol
  let s := real.cbrt s_vol
  let larger_edge := ratio * s
  let larger_vol := larger_edge ^ 3
  have h_s := real.cbrt_pow 3 s_vol
  simp [h_s, h_ratio, h_s_vol]
  sorry

end volume_of_larger_cube_is_125_l717_717974


namespace points_A₁_B₁_C₁_collinear_l717_717647

noncomputable theory

variables {A B C C' B' A' A₁ B₁ C₁ : Type*} [circle A B C C' B' A']

def inscribed_hexagon (A B C C' B' A' : Type*) := 
  inscribed_in_circle A B C C' B' A'

def intersection_BC'_B'C_is_A₁ (A : Type*) (B C' B' C A₁ : Type*) :=
  intersects_at (line B C') (line B' C) A₁

def intersection_CA'_C'A_is_B₁ (A : Type*) (C A' C' A B₁ : Type*) :=
  intersects_at (line C A') (line C' A) B₁

def intersection_AB'_A'B_is_C₁ (A : Type*) (A B' A' B C₁ : Type*) :=
  intersects_at (line A B') (line A' B) C₁

theorem points_A₁_B₁_C₁_collinear
  (h : inscribed_hexagon A B C C' B' A')
  (h₁ : intersection_BC'_B'C_is_A₁ A B C' B' C A₁)
  (h₂ : intersection_CA'_C'A_is_B₁ A C A' C' A B₁)
  (h₃ : intersection_AB'_A'B_is_C₁ A A B' A' B C₁) :
  collinear A₁ B₁ C₁ :=
sorry

end points_A₁_B₁_C₁_collinear_l717_717647


namespace parabola_standard_equation_l717_717453

theorem parabola_standard_equation (P : ℝ) : 
  (∃ x, x = -7) → P = 14 → (y : ℝ) : y^2 = 28 * x :=
by sorry

end parabola_standard_equation_l717_717453


namespace find_integer_n_l717_717005

theorem find_integer_n : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ -2023 [ZMOD 12] ∧ n = 9 := 
by
  use 9
  split
  exact le_refl 9
  split
  exact le_of_lt (by norm_num)
  split
  exact int.modeq.intro (by linarith)
  rfl

end find_integer_n_l717_717005


namespace tan_difference_l717_717443

variables {α β : ℝ}
-- Definitions of given conditions
def tan_alpha : ℝ := 3
def tan_beta : ℝ := 4 / 3

-- Main statement to be proved
theorem tan_difference : 
  tan (α - β) = (tan_alpha - tan_beta) / (1 + tan_alpha * tan_beta) :=
sorry

end tan_difference_l717_717443


namespace find_k_l717_717029

theorem find_k 
  (e1 : ℝ × ℝ) (h_e1 : e1 = (1, 0))
  (e2 : ℝ × ℝ) (h_e2 : e2 = (0, 1))
  (a : ℝ × ℝ) (h_a : a = (1, -2))
  (b : ℝ × ℝ) (h_b : b = (k, 1))
  (parallel : ∃ m : ℝ, a = (m * b.1, m * b.2)) : 
  k = -1/2 :=
sorry

end find_k_l717_717029


namespace sum_largest_and_smallest_eight_digit_numbers_l717_717992

theorem sum_largest_and_smallest_eight_digit_numbers : 
  ∃ L S : ℕ, 
  (L = 66442200) ∧ 
  (S = 20024466) ∧ 
  (L + S = 86466666) :=
begin
  use 66442200,
  use 20024466,
  split,
  { refl },
  split,
  { refl },
  { refl },
end

end sum_largest_and_smallest_eight_digit_numbers_l717_717992


namespace sequence_converges_to_2_l717_717344

noncomputable def sequence (u : ℕ → ℝ) : Prop :=
  (u 0 = 3) ∧ (∀ n : ℕ, n > 0 → u n = (u (n-1) + 2 * n^2 - 2) / n^2)

theorem sequence_converges_to_2 (u : ℕ → ℝ) (h : sequence u) : 
  ∃ L : ℝ, L = 2 ∧ tendsto u at_top (𝓝 L) :=
by
  sorry

end sequence_converges_to_2_l717_717344


namespace complex_number_in_fourth_quadrant_l717_717475

-- Define the function to simplify the complex expression
def simplify_complex_expr (z : ℂ) : ℂ := 
  2 / (1 + complex.I)

-- Define the real and imaginary parts of the resulting complex number
def real_part (z : ℂ) : ℝ := z.re
def imag_part (z : ℂ) : ℝ := z.im

-- Define the conditions for the quadrants
def is_first_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im > 0
def is_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0
def is_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0
def is_fourth_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

-- Define the main theorem to prove that the complex number is in the fourth quadrant
theorem complex_number_in_fourth_quadrant : 
  is_fourth_quadrant (simplify_complex_expr (2 / (1 + complex.I))) :=
by 
  -- You can fill in the proof steps here
  sorry

end complex_number_in_fourth_quadrant_l717_717475


namespace find_a7_l717_717709

variable (a : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n+1) = r * a n

axiom a3_eq_1 : a 3 = 1
axiom det_eq_0 : a 6 * a 8 - 8 * 8 = 0

theorem find_a7 (h_geom : geometric_sequence a) : a 7 = 8 :=
  sorry

end find_a7_l717_717709


namespace minimum_value_l717_717027

noncomputable def f (a b x : ℝ) : ℝ := a^x - b
def g (x : ℝ) : ℝ := x + 1

theorem minimum_value (a b : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1)
  (h_real : b ∈ ℝ)
  (h_ineq : ∀ x : ℝ, (f a b x) * (g x) ≤ 0) :
  (1 / a) + (4 / b) = 4 :=
sorry

end minimum_value_l717_717027


namespace percentage_of_40_eq_140_l717_717638

theorem percentage_of_40_eq_140 (p : ℝ) (h : (p / 100) * 40 = 140) : p = 350 :=
sorry

end percentage_of_40_eq_140_l717_717638


namespace num_pairs_satisfying_equation_l717_717827

open Int

theorem num_pairs_satisfying_equation :
  ∃ n : Nat, n = 2 ∧
  {p : ℕ × ℕ | (p.1 > 0) ∧ (p.2 > 0) ∧ (p.1 + p.2 + 3)^2 = 4 * (p.1^2 + p.2^2)}.size = n :=
by
  sorry -- Proof here

end num_pairs_satisfying_equation_l717_717827


namespace blake_initial_amount_l717_717328

theorem blake_initial_amount (X : ℝ) (h1 : X > 0) (h2 : 3 * X / 2 = 30000) : X = 20000 :=
sorry

end blake_initial_amount_l717_717328


namespace distance_AF_l717_717784

noncomputable def parametric_curve (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.sqrt 2 * Real.cos θ, 1 + Real.cos (2 * θ))

def point_A : ℝ × ℝ := (1, 0)

def focus (θ : ℝ) : Prop :=
  ∃ (Fx Fy : ℝ), parametric_curve θ = (Fx, Fy) ∧ x^2 = 4y

theorem distance_AF :
  ∀ θ : ℝ, ∃ F : ℝ × ℝ, parametric_curve θ = F ∧ Mathlib.dist (1, 0) F = Real.sqrt 2 :=
by
  sorry

end distance_AF_l717_717784


namespace area_triangle_XQY_l717_717864

-- Given conditions
variables (X Y Z P Q R : Type)
noncomputable def area_triangle_XYZ : ℝ := 15
noncomputable def XP : ℝ := 3
noncomputable def PY : ℝ := 4
noncomputable def XY : ℝ := XP + PY -- XY length = 7

-- Points P, Q, R lie on sides XY, YZ, and ZX respectively
axiom h1 : P ∈ segment X Y
axiom h2 : Q ∈ segment Y Z
axiom h3 : R ∈ segment Z X

-- Areas of triangle XQY and trapezoid PYQZ are equal
axiom equal_areas : area_triangle_XQY = area_trapezoid_PYQZ

-- Prove that the area of triangle XQY is 240/49
theorem area_triangle_XQY : area_triangle_XQY = 240 / 49 :=
begin
  sorry
end

end area_triangle_XQY_l717_717864


namespace calc_expr_l717_717336

noncomputable def expr_val : ℝ :=
  Real.sqrt 4 - |(-(1 / 4 : ℝ))| + (Real.pi - 2)^0 + 2^(-2 : ℝ)

theorem calc_expr : expr_val = 3 := by
  sorry

end calc_expr_l717_717336


namespace expression_evaluation_l717_717334

noncomputable def expression : ℝ :=
  (4 - Real.pi)^0 - 2 * Real.sin (Real.pi / 3) + abs (3 - Real.sqrt 12) - (1/2)^(-1)

theorem expression_evaluation : expression = Real.sqrt 3 - 4 :=
by
  sorry

end expression_evaluation_l717_717334


namespace max_daily_sales_revenue_l717_717639

noncomputable def f (t : ℕ) : ℝ :=
  if 0 ≤ t ∧ t < 15 
  then (1 / 3) * t + 8
  else if 15 ≤ t ∧ t < 30 
  then -(1 / 3) * t + 18
  else 0

noncomputable def g (t : ℕ) : ℝ :=
  if 0 ≤ t ∧ t ≤ 30
  then -t + 30
  else 0

noncomputable def W (t : ℕ) : ℝ :=
  f t * g t

theorem max_daily_sales_revenue : ∃ t : ℕ, W t = 243 :=
by
  existsi 3
  sorry

end max_daily_sales_revenue_l717_717639


namespace find_arithmetic_sequence_l717_717420

theorem find_arithmetic_sequence :
  ∃ (a d : ℚ), let s := [a - 3 * d, a - d, a + d, a + 3 * d] in 
  (s.sum = 26) ∧ ((a - d) * (a + d) = 40) ∧ 
  (s = [2, 5, 8, 11] ∨ s = [11, 8, 5, 2]) :=
by
  sorry

end find_arithmetic_sequence_l717_717420


namespace mean_temp_is_84_point_2_l717_717566

def temperatures : List ℝ := [79, 81, 83, 85, 84, 86, 88, 87, 85, 84]

def mean_temperature (temps : List ℝ) : ℝ := (temps.foldl (λ acc temp => acc + temp) 0) / temps.length

theorem mean_temp_is_84_point_2 : mean_temperature temperatures = 84.2 := 
by 
  -- Proof omitted
  sorry

end mean_temp_is_84_point_2_l717_717566


namespace largest_of_seven_consecutive_integers_l717_717968

theorem largest_of_seven_consecutive_integers (n : ℕ) (h : n > 0) (h_sum : n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) = 2222) : (n + 6) = 320 :=
by sorry

end largest_of_seven_consecutive_integers_l717_717968


namespace solve_b_constants_l717_717577

theorem solve_b_constants :
  ∃ (b1 b2 b3 b4 b5 b6 : ℝ), 
    (∀ (θ : ℝ), cos θ ^ 6 = 
      b1 * cos θ + b2 * cos (2 * θ) + b3 * cos (3 * θ) +
      b4 * cos (4 * θ) + b5 * cos (5 * θ) + b6 * cos (6 * θ)) ∧
    (b1 ^ 2 + b2 ^ 2 + b3 ^ 2 + b4 ^ 2 + b5 ^ 2 + b6 ^ 2 = 235 / 1024) :=
sorry

end solve_b_constants_l717_717577


namespace at_least_n_black_squares_l717_717493

theorem at_least_n_black_squares
  (n : ℕ) 
  (h_pos : n > 0)
  (M : Matrix (Fin 4) (Fin n) Bool)
  (h : ∀ i j, M i j = false → 
    (i > 0 ∧ M (i - 1) j = true) ∨ 
    (i < 3 ∧ M (i + 1) j = true) ∨ 
    (j > 0 ∧ M i (j - 1) = true) ∨ 
    (j < n-1 ∧ M i (j + 1) = true)) :
  ∑ i j, if M i j then 1 else 0 ≥ n := 
sorry

end at_least_n_black_squares_l717_717493


namespace triangle_side_length_l717_717857

theorem triangle_side_length 
  (P Q R S T : Point)
  (h₁ : P ≠ Q) (h₂ : Q ≠ R) (h₃ : R ≠ P)
  (h₄ : S ∈ Line(Q, R)) (h₅ : T ∈ Line(P, Q))
  (h₆ : Perpendicular(PS, QR)) (h₇ : Perpendicular(RT, PQ))
  (h₈ : distance(P, T) = 1) (h₉ : distance(T, Q) = 4) (h₁₀ : distance(Q, S) = 3) : 
  distance(S, R) = 11 / 3 :=
by
  sorry

end triangle_side_length_l717_717857


namespace g_seven_l717_717822

def g (x : ℚ) : ℚ := (2 * x + 3) / (4 * x - 5)

theorem g_seven : g 7 = 17 / 23 := by
  sorry

end g_seven_l717_717822


namespace equal_chord_lengths_l717_717265

open EuclideanGeometry RegularPentagon

noncomputable def middle_center (ABCDE : RegularPentagon) (D : ABCDE.Points) := CenterSegmentOn D

theorem equal_chord_lengths (ABCDE : RegularPentagon) (D : ABCDE.Points) (P : Point)
  (hP : P ≠ middle_center ABCDE D) (hPQ : Circumcircle ABCDE.P A B P Q) (hR : PerpendicularToThrough P ABCDE.line_D CD R) : 
  Distance P ABCDE.line_D := P ↔ 
  Distance P ABCDE.line_D :
  Distance AR := QR ∧ AR := QR :=
sorry

end equal_chord_lengths_l717_717265


namespace length_of_segment_OS_l717_717282

open Real

/-- Definitions of the conditions in the problem -/
def circle1_center := (O : ℝ × ℝ)
def circle1_radius : ℝ := 8

def circle2_center := (P : ℝ × ℝ)
def circle2_radius : ℝ := 2

def tangent_point : ℝ × ℝ := Q
def tangent_segment := (T S : ℝ × ℝ)

/-- Mathematical statement of the problem -/
theorem length_of_segment_OS
  (O P Q T S : ℝ × ℝ) 
  (h_circles_tangent : dist O P = circle1_radius + circle2_radius)
  (radius_O : dist O Q = circle1_radius)
  (radius_P : dist P Q = circle2_radius)
  (tangent_TS_O : dist O T = circle1_radius)
  (tangent_TS_P : dist P S = circle2_radius)
  (segment_TS_Tangent : T ≠ S ∧ ∀ x, x ∈ line [T, S] → dist O x ≥ circle1_radius ∧ dist P x ≥ circle2_radius) :
  dist O S = 8 * sqrt 2 :=
sorry

end length_of_segment_OS_l717_717282


namespace candy_bar_cost_l717_717579

theorem candy_bar_cost
  (initial_amount : ℕ)
  (change : ℕ)
  (candy_bar_cost : ℕ)
  (initial_amount = 50)
  (change = 5)
  (candy_bar_cost = initial_amount - change) :
  candy_bar_cost = 45 :=
by
  sorry

end candy_bar_cost_l717_717579


namespace unique_committee_l717_717489

-- Define people
inductive Person
| Jane
| Thomas
| A
| B
| C
| D
deriving DecidableEq, Inhabited

-- Define skills
inductive Skill
| PublicSpeaking
| FinancialPlanning
| EventPlanning
| Accounting
| Marketing
| WebDesign
| SponsorshipCoordination
deriving DecidableEq, Inhabited

-- Each person's skills
def skills : Person → List Skill
| Person.Jane := [Skill.PublicSpeaking, Skill.Accounting]
| Person.Thomas := [Skill.Outreach, Skill.FinancialPlanning]
| Person.A := [Skill.EventPlanning]
| Person.B := [Skill.Marketing]
| Person.C := [Skill.WebDesign]
| Person.D := [Skill.SponsorshipCoordination]

-- The requirement function for needed skills in the committee
def skillSet : List Skill := [Skill.PublicSpeaking, Skill.FinancialPlanning, Skill.EventPlanning]

-- The condition that at least one of Jane or Thomas is selected
def includesJaneOrThomas (committee : List Person) : Prop :=
  Person.Jane ∈ committee ∨ Person.Thomas ∈ committee

-- The condition that required skills are covered
def coversSkills (committee : List Person) : Prop :=
  ∀ skill ∈ skillSet, ∃ p ∈ committee, skill ∈ skills p

-- The definition of our specific committee
def committee : List Person := [Person.Jane, Person.Thomas, Person.A]

-- The theorem to prove
theorem unique_committee :
  includesJaneOrThomas committee ∧ coversSkills committee ∧ (∀ c, includesJaneOrThomas c ∧ coversSkills c → c = committee) :=
by
  sorry

end unique_committee_l717_717489


namespace crayons_ratio_l717_717878

theorem crayons_ratio (K B G J : ℕ) 
  (h1 : K = 2 * B)
  (h2 : B = 2 * G)
  (h3 : G = J)
  (h4 : K = 128)
  (h5 : J = 8) : 
  G / J = 4 :=
by
  sorry

end crayons_ratio_l717_717878


namespace race_positions_l717_717904

variable (nabeel marzuq arabi rafsan lian rahul : ℕ)

theorem race_positions :
  (arabi = 6) →
  (arabi = rafsan + 1) →
  (rafsan = rahul + 2) →
  (rahul = nabeel + 1) →
  (nabeel = marzuq + 6) →
  (marzuq = 8) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end race_positions_l717_717904


namespace triangle_inequality_l717_717046

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) 
  (h2 : b + c > a) (h3 : c + a > b) :
  let S := a^2 + b^2 + c^2
  let P := a * b + b * c + c * a
  in P ≤ S ∧ S < 2 * P :=
by
  let S := a^2 + b^2 + c^2
  let P := a * b + b * c + c * a
  sorry

end triangle_inequality_l717_717046


namespace smallest_solution_eq_l717_717010

theorem smallest_solution_eq {
  x : ℝ
} :
  (x * | x | = 3 * x + 4) →
  (∀ y : ℝ, y * |y| = 3 * y + 4 → y ≥ 0 → y = 4) :=
by
  sorry

end smallest_solution_eq_l717_717010


namespace problem_solution_l717_717606

theorem problem_solution : (6 * 7 * 8 * 9 * 10) / (6 + 7 + 8 + 9 + 10) = 756 := by
  sorry

end problem_solution_l717_717606


namespace petya_can_prevent_natural_sum_l717_717912

def petya_prevents_natural_sum : Prop :=
  ∀ (fractions : List ℚ),
    (∀ f ∈ fractions, ∃ n : ℕ, f = 1 / n) →
    ∃ m : ℕ, let new_fractions := (1 / m) :: fractions in
    ∀ k : ℕ, 
      k > 0 →
      let vasya_fractions := new_fractions.take k in
      ∑ i in vasya_fractions, id i ∉ ℕ

theorem petya_can_prevent_natural_sum : petya_prevents_natural_sum :=
sorry

end petya_can_prevent_natural_sum_l717_717912


namespace james_money_left_l717_717484

-- Define the initial conditions
def ticket1_cost : ℕ := 150
def ticket2_cost : ℕ := 150
def ticket3_cost : ℕ := ticket1_cost / 3
def total_money : ℕ := 500
def roommate_share : ℕ := 2

-- Define and prove the theorem
theorem james_money_left : 
  let total_ticket_cost := ticket1_cost + ticket2_cost + ticket3_cost in
  let james_cost := total_ticket_cost / roommate_share in
  total_money - james_cost = 325 :=
by 
  let total_ticket_cost := ticket1_cost + ticket2_cost + ticket3_cost
  let james_cost := total_ticket_cost / roommate_share
  exact eq.refl 325

end james_money_left_l717_717484


namespace matrix_sequence_product_l717_717689

theorem matrix_sequence_product :
  let seq := list.range' 3 50  -- creates the sequence [3, 5, 7, ..., 101]
  seq.sum = 2600 →
  (seq.foldl (λ M k => matrix.mul M (λ (i j : fin 2), if (i, j) = (0, 1) then k else if i = j then 1 else 0)) (λ (i j : fin 2), if i = j then 1 else 0)).get 0 1 = 2600 := 
sorry

end matrix_sequence_product_l717_717689


namespace eleven_hash_five_l717_717017

noncomputable def op (r s : ℕ) : ℕ := 
  match r, s with
  | 0, s     => s
  | r + 1, s => (op r s) + s.factorial + 1
  | r, 0     => r
  | r, s     => op s r

theorem eleven_hash_five : op 11 5 = 1815 := 
  sorry

end eleven_hash_five_l717_717017


namespace trajectory_equation_l717_717371

theorem trajectory_equation (x y : ℝ) 
  (h : abs (x - 25 / 4) / real.sqrt ((x - 4) ^ 2 + y ^ 2) = 5 / 4) : 
  x ^ 2 / 25 + y ^ 2 / 9 = 1 := 
by sorry

end trajectory_equation_l717_717371


namespace train_crossing_time_l717_717261

/-- A train that is 200 meters long, running at a speed of 72 km/hr, takes 16.6 seconds to cross a bridge that is 132 meters long. -/
theorem train_crossing_time : 
  ∀ (length_train length_bridge : ℕ) (speed_kmh : ℝ), 
  length_train = 200 → 
  length_bridge = 132 → 
  speed_kmh = 72 → 
  (length_train + length_bridge) / (speed_kmh * 1000 / 3600) = 16.6 :=
by
  intros length_train length_bridge speed_kmh ht hb hs
  rw [ht, hb, hs]
  norm_num
  sorry

end train_crossing_time_l717_717261


namespace wall_volume_l717_717222

theorem wall_volume (Width : ℝ) (h1 : Width = 4) : 
  let Height := 6 * Width,
      Length := 7 * Height,
      Volume := Length * Width * Height
  in Volume = 16128 :=
by
  sorry

end wall_volume_l717_717222


namespace a_alone_work_days_l717_717267

noncomputable def days_to_complete : ℝ := 36 / 7

def work_rate (total_days : ℝ) : ℝ := 1 / total_days

def r_a : ℝ := work_rate 3 - work_rate 6 / 2
def r_b : ℝ := work_rate 6 - work_rate 3.6 / 2
def r_c : ℝ := work_rate 3.6 - work_rate 3 / 3 * 2

def r_all : ℝ := r_a + r_b + r_c

theorem a_alone_work_days :
  (work_rate 3) + (work_rate 6) + (work_rate 3.6) - (work_rate 2) = days_to_complete :=
by
  sorry

end a_alone_work_days_l717_717267


namespace find_formula_l717_717798

theorem find_formula :
  (∀ (x : ℕ), x = 0 → 150 = -30 * x + 150) ∧
  (∀ (x : ℕ), x = 1 → 120 = -30 * x + 150) ∧
  (∀ (x : ℕ), x = 2 → 90 = -30 * x + 150) ∧
  (∀ (x : ℕ), x = 3 → 60 = -30 * x + 150) ∧
  (∀ (x : ℕ), x = 4 → 30 = -30 * x + 150)
:= sorry

end find_formula_l717_717798


namespace no_valid_n_sum_equals_zero_l717_717989

theorem no_valid_n_sum_equals_zero :
  (∑ n in Finset.filter (λ n : ℕ, Nat.lcm n 150 = Nat.gcd n 150 + 600) (Finset.range 151)) = 0 :=
by
  -- This statement entails that there are no such positive integers n < 151 that fulfill the criteria
  sorry

end no_valid_n_sum_equals_zero_l717_717989


namespace perimeter_of_ABC_l717_717117

variable P Q R A B C : Type
variable [metric_space P] [metric_space Q] [metric_space R] [metric_space A] [metric_space B] [metric_space C]
variable circle_tangent_to_each_other_at_PQR : ∀ (P Q R : Type), tangent PQR
variable circle_tangent_to_sides_of_triangle : ∀ (P Q R A B C : Type), tangent (P, AB) ∧ tangent (Q, AC) ∧ tangent (R, BC)
variable radius_eq_two : ∀ (P Q R : Type), radius P = 2 ∧ radius Q = 2 ∧ radius R = 2
variable PQR_is_equilateral : ∀ (P Q R : Type), equilateral (triangle P Q R)

theorem perimeter_of_ABC : perimeter (triangle A B C) = 24 := by
  sorry

end perimeter_of_ABC_l717_717117


namespace field_division_l717_717648

theorem field_division (A B : ℝ) (h1 : A + B = 700) (h2 : B - A = (1 / 5) * ((A + B) / 2)) : A = 315 :=
by
  sorry

end field_division_l717_717648


namespace distance_to_plane_is_correct_l717_717002

-- Define the points M0, M1, M2, M3
def M0 : ℝ × ℝ × ℝ := (-21, 20, -16)
def M1 : ℝ × ℝ × ℝ := (-2, -1, -1)
def M2 : ℝ × ℝ × ℝ := (0, 3, 2)
def M3 : ℝ × ℝ × ℝ := (3, 1, -4)

-- Define the function to compute the distance from a point to a plane
-- given by the normal vector (A, B, C) and constant D, and the point (x0, y0, z0)
def point_to_plane_distance (A B C D x0 y0 z0 : ℝ) : ℝ :=
  (Abs (A * x0 + B * y0 + C * z0 + D)) / sqrt (A * A + B * B + C * C)

-- Calculating the coefficients of the plane equation passing through M1, M2, M3
-- that it can be proven and then used in further computation.
def plane_coefficients (M1 M2 M3 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (x1, y1, z1) := M1
  let (x2, y2, z2) := M2
  let (x3, y3, z3) := M3
  let A := (y1 - y2) * (z1 - z3) - (z1 - z2) * (y1 - y3)
  let B := (z1 - z2) * (x1 - x3) - (x1 - x2) * (z1 - z3)
  let C := (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
  let D := - (A * x1 + B * y1 + C * z1)
  (A, B, C, D)

-- Using the obtained plane coefficients and the point M0 to compute the distance
def distance_M0_to_plane : ℝ :=
  let (A, B, C, D) := plane_coefficients M1 M2 M3
  point_to_plane_distance A B C D (M0.1) (M0.2) (M0.3)

-- Now we state the theorem that the distance from point M0 to the plane passing through
-- M1, M2, and M3 is (1023 / sqrt 1021)
theorem distance_to_plane_is_correct :
  distance_M0_to_plane = 1023 / sqrt 1021 :=
  sorry

end distance_to_plane_is_correct_l717_717002


namespace sufficient_but_not_necessary_condition_l717_717092

def floor (x : ℝ) : ℤ := Int.floor x

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (floor x = floor y) → |x - y| < 1 ∧ ¬ (|x - y| < 1 → floor x = floor y) :=
by
  sorry

end sufficient_but_not_necessary_condition_l717_717092


namespace tan_11pi_over_6_l717_717364

theorem tan_11pi_over_6 :
  Real.tan (11 * Real.pi / 6) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_11pi_over_6_l717_717364


namespace james_sushi_rolls_l717_717870

def fish_for_sushi : ℕ := 40
def total_fish : ℕ := 400
def bad_fish_percentage : ℕ := 20

theorem james_sushi_rolls :
  let good_fish := total_fish - (bad_fish_percentage * total_fish / 100)
  good_fish / fish_for_sushi = 8 :=
by
  sorry

end james_sushi_rolls_l717_717870


namespace vector_magnitude_is_sqrt_five_l717_717025

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x, 4)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

def magnitude (u : ℝ × ℝ) : ℝ :=
  real.sqrt (u.1^2 + u.2^2)

theorem vector_magnitude_is_sqrt_five (x : ℝ) (h : dot_product a (b x) = 10) :
  magnitude (vector_sub a (b x)) = real.sqrt 5 :=
by
  sorry

end vector_magnitude_is_sqrt_five_l717_717025


namespace compute_p2_q2_compute_p3_q3_l717_717893

variables (p q : ℝ)

theorem compute_p2_q2 (h1 : p * q = 15) (h2 : p + q = 8) : p^2 + q^2 = 34 :=
sorry

theorem compute_p3_q3 (h1 : p * q = 15) (h2 : p + q = 8) : p^3 + q^3 = 152 :=
sorry

end compute_p2_q2_compute_p3_q3_l717_717893


namespace detour_distance_l717_717526

-- Definitions based on conditions:
def D_black : ℕ := sorry -- The original distance along the black route
def D_black_C : ℕ := sorry -- The distance from C to B along the black route
def D_red : ℕ := sorry -- The distance from C to B along the red route

-- Extra distance due to detour calculation
def D_extra := D_red - D_black_C

-- Prove that the extra distance is 14 km
theorem detour_distance : D_extra = 14 := by
  sorry

end detour_distance_l717_717526


namespace smallest_common_multiple_l717_717741

theorem smallest_common_multiple (b : ℕ) (hb : b > 0) (h1 : b % 6 = 0) (h2 : b % 15 = 0) :
    b = 30 :=
sorry

end smallest_common_multiple_l717_717741


namespace max_plus_min_f_eq_two_l717_717891

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_plus_min_f_eq_two :
  let M := RealSup.sup {y | ∃ x, f x = y} -- maximum value
  let m := RealInf.inf {y | ∃ x, f x = y} -- minimum value
  M + m = 2 := sorry

end max_plus_min_f_eq_two_l717_717891


namespace trig_identity_l717_717826

-- State the problem as a theorem in Lean 4.
theorem trig_identity (α β : ℝ) (h1 : α ∈ Ioo (π / 2) π) (h2 : β ∈ Ioo (π / 2) π)
  (h3 : (1 - Real.cos (2 * α)) * (1 + Real.sin β) = Real.sin (2 * α) * Real.cos β) :
  2 * α + β = (5 * π) / 2 := 
sorry

end trig_identity_l717_717826


namespace solve_for_x_l717_717926

theorem solve_for_x (x : ℝ) (h : 3 / 4 + 1 / x = 7 / 8) : x = 8 :=
by
  sorry

end solve_for_x_l717_717926


namespace value_added_to_number_l717_717229

theorem value_added_to_number (x : ℤ) : 
  (150 - 109 = 109 + x) → (x = -68) :=
by
  sorry

end value_added_to_number_l717_717229


namespace ratio_DE_DF_l717_717124

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variables {A B C D E F : V}
variables {a b c d e f : V}

def point_on_AC (A C D : V) (k : ℝ) : Prop := D = (k / (k + 1)) • A + (1 / (k + 1)) • C
def point_on_BC (B C E : V) (m : ℝ) : Prop := E = (m / (m + 1)) • B + (1 / (m + 1)) • C
def line_intersects (D E F A B : V) : Prop := ∃ λ μ : ℝ, F = λ • D + (1 - λ) • E ∧ F = μ • A + (1 - μ) • B

theorem ratio_DE_DF (A B C D E F : V) (ha : 0 < 4) (hb : 0 < 2) :
  point_on_AC A C D 4 → point_on_BC B C E 2 → line_intersects D E F A B → (∥D - E∥ / ∥D - F∥) = 2 :=
by 
  intros h1 h2 h3
  sorry

end ratio_DE_DF_l717_717124


namespace problem_proof_l717_717646

noncomputable def original_number_of_buses_and_total_passengers : Nat × Nat :=
  let k := 24
  let total_passengers := 529
  (k, total_passengers)

theorem problem_proof (k n : Nat) (h₁ : n = 22 + 23 / (k - 1)) (h₂ : 22 * k + 1 = n * (k - 1)) (h₃ : k ≥ 2) (h₄ : n ≤ 32) :
  (k, 22 * k + 1) = original_number_of_buses_and_total_passengers :=
by
  sorry

end problem_proof_l717_717646


namespace female_managers_count_l717_717109

variable (E : ℕ) -- Total number of employees
variable (M : ℕ) -- Total number of male employees
variable (F_female : ℕ := 1000) -- Number of female employees
variable (F_female_managers : ℕ)

theorem female_managers_count (h1 : 2 * E = 5 * (F_female_managers + 2 * M / 5))
                             (h2 : E = M + F_female)
                             (F_female = 1000) :
  F_female_managers = 400 :=
by
  sorry

end female_managers_count_l717_717109


namespace additional_flour_needed_l717_717151

-- Define the given conditions
def original_flour : ℕ := 500
def flour_added : ℕ := 300
def flour_needed (multiplier : ℕ) : ℕ := original_flour * multiplier

-- Define the statement that needs to be proven
theorem additional_flour_needed (multiplier : ℕ) (h : multiplier = 2) : (flour_needed multiplier) - flour_added = 700 :=
by
  rw h
  sorry

end additional_flour_needed_l717_717151


namespace prism_parallelepiped_iff_conditions_l717_717860

-- Definitions based on conditions
structure PlaneQuadrilateral :=
  (a b c d: ℝ) -- dummy values to represent the sides
  (opposite_sides_parallel: (a // b) = (c // d)) 

structure SpaceQuadrilateralPrism :=
  (face_one face_two face_three face_four face_five face_six: ℝ) -- dummy values to represent the faces
  (opposite_faces_parallel_one: face_one = face_four)
  (opposite_faces_parallel_two: face_two = face_five)
  (opposite_faces_parallel_three: face_three = face_six)

-- conditions for a parallelepiped
def is_parallelepiped (prism: SpaceQuadrilateralPrism) : Prop :=
  (prism.opposite_faces_parallel_one ∧ prism.opposite_faces_parallel_two ∧ prism.opposite_faces_parallel_three) ∧
  (face_one = face_four ∧ face_two = face_five ∧ face_three = face_six)

-- Theorem statement
theorem prism_parallelepiped_iff_conditions (prism: SpaceQuadrilateralPrism) :
  is_parallelepiped prism ↔ 
  (prism.opposite_faces_parallel_one ∧ prism.opposite_faces_parallel_two ∧ prism.opposite_faces_parallel_three) ∧
  (face_one = face_four ∧ face_two = face_five ∧ face_three = face_six) := 
sorry

end prism_parallelepiped_iff_conditions_l717_717860


namespace monotonic_intervals_and_solution_range_l717_717427

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := c * log x + (1/2) * x^2 + b * x

theorem monotonic_intervals_and_solution_range {b c : ℝ} (h1 : b + c + 1 = 0) (h2 : c ≠ 0) :
  (∀ x : ℝ, f x b c = c * log x + (1/2) * x^2 + b * x) →
  (x = 1 → f'(1) = 0) → 
  (c > 1 → (∀ x ∈ (0, 1), f' x > 0) ∧ (∀ x ∈ (1, c), f' x < 0) ∧ (∀ x ∈ (c, +∞), f' x > 0)) ∧
  (∀ h3 : ℝ → ℝ, (∃! y, f y b c = 0) → -1/2 < c < 0) :=
sorry

end monotonic_intervals_and_solution_range_l717_717427


namespace exam_questions_count_l717_717524

theorem exam_questions_count (k : ℕ) (h_k : 1 < k)
  (h_prob : (k * ∏ i in finset.range k, (i / (i + 1))^i) = 2018 * ∏ i in finset.range k, (i / (i + 1))^i) :
  (∑ i in finset.range (k + 1), i) = (2018 * 2019) / 2 :=
by sorry

end exam_questions_count_l717_717524


namespace find_a7_l717_717711

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem find_a7 (a : ℕ → ℝ) (h_geom : geometric_sequence a)
  (h3 : a 3 = 1)
  (h_det : a 6 * a 8 - 8 * 8 = 0) :
  a 7 = 8 :=
sorry

end find_a7_l717_717711


namespace OI_parallel_l_l717_717317

-- Define the points and their properties
variables {A B C M N P Q O I S : Point}
variables (AMNPQ_inscribed : EquilateralPentagonInscribedInTriangle A M N P Q B C)
variables (M_on_AB : M ∈ AB)
variables (Q_on_AC : Q ∈ AC)
variables (N_on_BC : N ∈ BC)
variables (P_on_BC : P ∈ BC)
variables (S_is_intersection : S = intersection (MN) (PQ))
variables (l_is_bisector : IsAngleBisector S M Q l)
variables (O_is_circumcenter : IsCircumcenter O A B C)
variables (I_is_incenter : IsIncenter I A B C)

-- Define what needs to be proven
theorem OI_parallel_l : Parallel O I l :=
sorry

end OI_parallel_l_l717_717317


namespace range_of_m_l717_717815

-- Define the vector a as given in the conditions.
def vector_a (m : ℝ) : ℝ × ℝ := (m, 3 * m - 4)

-- Define the vector b as given in the conditions.
def vector_b : ℝ × ℝ := (1, 2)

-- Define the condition for vectors a and b to form a basis (i.e., they are not collinear).
def not_collinear (m : ℝ) : Prop :=
  m / 1 ≠ (3 * m - 4) / 2

-- The proof problem statement
theorem range_of_m (m : ℝ) : not_collinear m ↔ m ∈ Set.Ioo (-∞) 4 ∪ Set.Ioo 4 ∞ :=
by
  sorry

end range_of_m_l717_717815


namespace third_side_triangle_l717_717418

theorem third_side_triangle (x : ℝ) (h1 : 3 < x) (h2 : x < 10) : 4 < x ∧ x < 10 :=
begin
  split,
  { change 4 < x,
    have t1 : 4 = 7 - 3 := rfl,
    rw t1,
    exact h1 },
  { exact h2 }
end

end third_side_triangle_l717_717418


namespace find_x_l717_717410

variable (x : ℝ)

def length := 4 * x
def width := x + 3

def area := length x * width x
def perimeter := 2 * length x + 2 * width x

theorem find_x (h : area x = 3 * perimeter x) : x = 5.342 := by
  sorry

end find_x_l717_717410


namespace zachary_pushups_l717_717616

variable {P : ℕ}
variable {C : ℕ}

theorem zachary_pushups :
  C = 58 → C = P + 12 → P = 46 :=
by 
  intros hC1 hC2
  rw [hC2] at hC1
  linarith

end zachary_pushups_l717_717616


namespace exists_two_points_with_circles_containing_673_points_l717_717266

theorem exists_two_points_with_circles_containing_673_points 
  (S : finset (ℝ × ℝ))
  (h_card : S.card = 2021)
  (h_convex : ∀ (A B C : ℝ × ℝ), A ∈ S → B ∈ S → C ∈ S → convex_hull ℝ ↑S = ⟦A, B, C⟧)
  (h_noncollinear : ∀ (A B C : ℝ × ℝ), A ∈ S → B ∈ S → C ∈ S → A ≠ B → B ≠ C → A ≠ C → ¬ collinear ℝ ({A, B, C} : set (ℝ × ℝ)))
  (h_noncocyclic : ∀ (A B C D : ℝ × ℝ), A ∈ S → B ∈ S → C ∈ S → D ∈ S → A ≠ B → B ≠ C → C ≠ D → A ≠ D → D ≠ B → ¬ concyclic ℝ ({A, B, C, D} : set (ℝ × ℝ))) :
  ∃ (P Q : ℝ × ℝ), P ∈ S ∧ Q ∈ S ∧ P ≠ Q ∧ ∀ (C : circle ((ℝ × ℝ) × ℝ)) (h : P ∈ C) (h2 : Q ∈ C), finset.card ((S.filter (λ x, x ≠ P ∧ x ≠ Q ∧ x ∈ inner C))) ≥ 673 :=
sorry

end exists_two_points_with_circles_containing_673_points_l717_717266


namespace point_in_fourth_quadrant_l717_717407

theorem point_in_fourth_quadrant (θ : ℝ) (h : (π / 2) < θ ∧ θ < π) :
  sin θ > 0 ∧ cos θ < 0 → 
  (sin θ, cos θ).2 < 0 ∧ (sin θ, cos θ).1 > 0 := sorry

end point_in_fourth_quadrant_l717_717407


namespace trigonometric_identity_tangent_line_l717_717458

theorem trigonometric_identity_tangent_line 
  (α : ℝ) 
  (h_tan : Real.tan α = 4) 
  : Real.cos α ^ 2 - Real.sin (2 * α) = - 7 / 17 := 
by sorry

end trigonometric_identity_tangent_line_l717_717458


namespace fraction_of_alvin_age_l717_717725

variable (A E F : ℚ)

-- Conditions
def edwin_older_by_six : Prop := E = A + 6
def total_age : Prop := A + E = 30.99999999
def age_relation_in_two_years : Prop := E + 2 = F * (A + 2) + 20

-- Statement to prove
theorem fraction_of_alvin_age
  (h1 : edwin_older_by_six A E)
  (h2 : total_age A E)
  (h3 : age_relation_in_two_years A E F) :
  F = 1 / 29 :=
sorry

end fraction_of_alvin_age_l717_717725


namespace road_construction_problem_l717_717285

theorem road_construction_problem (x : ℝ) (h₁ : x > 0) :
    1200 / x - 1200 / (1.20 * x) = 2 :=
by
  sorry

end road_construction_problem_l717_717285


namespace insurance_covers_80_percent_l717_717905

def xray_cost : ℕ := 250
def mri_cost : ℕ := 3 * xray_cost
def total_cost : ℕ := xray_cost + mri_cost
def mike_payment : ℕ := 200
def insurance_coverage : ℕ := total_cost - mike_payment
def insurance_percentage : ℕ := (insurance_coverage * 100) / total_cost

theorem insurance_covers_80_percent : insurance_percentage = 80 := by
  -- Carry out the necessary calculations
  sorry

end insurance_covers_80_percent_l717_717905


namespace proof_problem_l717_717031

noncomputable def parabola (x : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 ^ 2 = 4 * p.1}

def point_in_parabola (p : ℝ × ℝ) : Prop :=
  p.2 ^ 2 = 4 * p.1

def line_through_point (p : ℝ × ℝ) (k : ℝ) : Set (ℝ × ℝ) :=
  {q | q.2 = k * (q.1 - p.1)}

def triangle_area (A B : ℝ × ℝ) (O : ℝ × ℝ) : ℝ :=
  1/2 * |A.1 * B.2 + B.1 * O.2 + O.1 * A.2 - (A.2 * B.1 + B.2 * O.1 + O.2 * A.1)|

theorem proof_problem (x1 x2 y1 y2 k : ℝ)
  (h1 : point_in_parabola (x1, y1))
  (h2 : point_in_parabola (x2, y2))
  (intersect_A : y1 = k * (x1 - 9/2))
  (intersect_B : y2 = k * (x2 - 9/2))
  (area_condition : triangle_area (x1, y1) (x2, y2) (0, 0) = 81/4) :
  (y1 * y2 = -18) ∧ (∃ k, (k = 2/3 ∨ k = -2/3) ∧ ((2 * x1 + 3 * y1 = 9) ∨ (2 * x1 - 3 * y1 = 9))) :=
sorry

end proof_problem_l717_717031


namespace percent_increase_l717_717996

theorem percent_increase (x : ℝ) (h : (1 / 2) * x = 1) : ((x - (1 / 2)) / (1 / 2)) * 100 = 300 := by
  sorry

end percent_increase_l717_717996


namespace systematic_sampling_correct_l717_717607

theorem systematic_sampling_correct :
  ∃ (s : Finset ℕ), s = {5, 15, 25, 35, 45} ∧ (∀ x ∈ s, x ≤ 50) ∧
  (∀ x y ∈ s, x ≠ y → (x - y) % 10 = 0) :=
by
  sorry

end systematic_sampling_correct_l717_717607


namespace find_p_l717_717417

-- Definitions for the problem
def parabola_latus_rectum (p : ℝ) : ℝ :=
  -p / 2

def hyperbola_left_latus_rectum (a c : ℝ) : ℝ :=
  -a^2 / c

-- Condition for the problem
axiom latus_rectum_condition (p : ℝ) (a c : ℝ)
  (h_a : a = sqrt(2))
  (h_c : c = 2) :
  parabola_latus_rectum p = hyperbola_left_latus_rectum a c

-- The main theorem to prove
theorem find_p : ∃ p : ℝ, p = 2 :=
by
  use 2
  sorry

end find_p_l717_717417


namespace solve_inequality_l717_717020

open Set Real

theorem solve_inequality (x : ℝ) : { x : ℝ | x^2 - 4 * x > 12 } = {x : ℝ | x < -2} ∪ {x : ℝ | 6 < x} := 
sorry

end solve_inequality_l717_717020


namespace area_of_Phi_distance_from_T_to_Phi_l717_717260

-- Defining the figure Φ by the inequalities
def figure_Phi (x y : ℝ) : Prop :=
  (y^2 - x^2 ≤ 3 * (x + y)) ∧ (x^2 + y^2 ≤ 6 * y - 6 * x - 9)

-- The area of the figure Φ is 9π / 2
theorem area_of_Phi : (9 * Real.pi / 2 = set.univ.measure (set {p : ℝ × ℝ | figure_Phi p.1 p.2})).to_real :=
sorry

-- The shortest distance from point T(-6, 0) to the figure Φ is 3√2 - 3
theorem distance_from_T_to_Phi : ∀ x y, figure_Phi x y → 
  dist (x, y) (-6, 0) = 3 * Real.sqrt 2 - 3 :=
sorry

end area_of_Phi_distance_from_T_to_Phi_l717_717260


namespace imaginary_part_conjugate_l717_717828

-- Define the complex number z
def z := (2 : ℂ) - (complex.I : ℂ)

-- The problem statement: proving the imaginary part of the conjugate of z is 1
theorem imaginary_part_conjugate (z := (2 - complex.I)) : complex.im (complex.conj z) = 1 :=
by
  sorry

end imaginary_part_conjugate_l717_717828


namespace shaded_area_of_triangle_l717_717225

def length_of_hypotenuse : ℝ := 13
def length_of_leg : ℝ := 5
def halved_leg_length : ℝ := (12 / 2) 

theorem shaded_area_of_triangle :
  let original_area := 0.5 * length_of_leg * 12 in
  let smaller_triangle_leg := length_of_leg / 2 in
  let smaller_area := 0.5 * smaller_triangle_leg * length_of_leg in
  original_area - smaller_area = 22.5 :=
by 
  sorry

end shaded_area_of_triangle_l717_717225


namespace water_tower_usage_l717_717307

theorem water_tower_usage :
  let W := 2700
  let U1 := 300
  let U2 := 2 * U1
  let U3 := U2 + 100
  let U4 := 3 * U1
  let U5 := U3 / 2
  let total_usage := U1 + U2 + U3 + U4 + U5
  total_usage > W → total_usage + 0 ≤ W :=
by {
  intros,
  have h : total_usage = U1 + U2 + U3 + U4 + U5 := rfl,
  sorry,
}

end water_tower_usage_l717_717307


namespace naomi_saw_total_wheels_l717_717682

theorem naomi_saw_total_wheels :
  (let regular_bikes := 7 in
   let childrens_bikes := 11 in
   let wheels_regular_bike := 2 in
   let wheels_children_bike := 4 in
   regular_bikes * wheels_regular_bike + childrens_bikes * wheels_children_bike = 58) :=
by 
  sorry

end naomi_saw_total_wheels_l717_717682


namespace intersection_M_N_l717_717134

noncomputable def M : Set ℝ := {x | x^2 + x - 6 < 0}
noncomputable def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N :
  {x : ℝ | M x ∧ N x } = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l717_717134


namespace find_f_of_7_l717_717408

variable {f : ℝ → ℝ}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem find_f_of_7 (h1 : is_odd_function f)
                    (h2 : is_periodic_function f 4)
                    (h3 : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) :
  f 7 = -2 := 
by
  sorry

end find_f_of_7_l717_717408


namespace weekly_business_hours_l717_717100

theorem weekly_business_hours :
  let weekday_hours := 6 * 5
  let weekend_hours := 4 * 2
  let total_hours := weekday_hours + weekend_hours
  total_hours = 38 :=
by
  unfold total_hours weekday_hours weekend_hours
  sorry

end weekly_business_hours_l717_717100


namespace range_of_2a_plus_c_theorem_l717_717125

noncomputable def range_of_2a_plus_c {A B C : ℝ} (a b c : ℝ) (habc : a^2 + c^2 - b^2 = ac) (hb : b = Real.sqrt 3) : set ℝ :=
  {x | x = 2 * a + c}

theorem range_of_2a_plus_c_theorem {a b c : ℝ} (habc : a^2 + c^2 - b^2 = ac) (hb : b = Real.sqrt 3) :
  range_of_2a_plus_c a b c habc hb = set.Ioo (Real.sqrt 3) (2 * Real.sqrt 7) ∪ {2 * Real.sqrt 7} :=
sorry

end range_of_2a_plus_c_theorem_l717_717125


namespace overlap_width_of_semicircles_l717_717982

-- Definitions for conditions
variables (r h : ℕ) 

-- Radius of each semicircle is 5 cm, height half is 4 cm
def radius := 5
def half_height := 4

-- The mathematical statement to be proven
theorem overlap_width_of_semicircles : 2 * (Math.sqrt (radius^2 - half_height^2)) = 6 := sorry

end overlap_width_of_semicircles_l717_717982


namespace length_of_common_chord_l717_717980

-- Problem conditions
variables (r : ℝ) (h : r = 15)

-- Statement to prove
theorem length_of_common_chord : 2 * (r / 2 * Real.sqrt 3) = 15 * Real.sqrt 3 :=
by
  sorry

end length_of_common_chord_l717_717980


namespace infinite_rational_points_l717_717345

/-- The set of points with positive rational coordinates in the $xy$-plane
    that satisfy the inequality \(x + 2y \leq 10\) is infinite. -/
theorem infinite_rational_points 
  : {p : ℚ × ℚ // 0 < p.1 ∧ 0 < p.2 ∧ p.1 + 2 * p.2 ≤ 10}.infinite :=
sorry

end infinite_rational_points_l717_717345


namespace complex_number_location_second_quadrant_l717_717452

theorem complex_number_location_second_quadrant (z : ℂ) (h : z / (1 + I) = I) : z.re < 0 ∧ z.im > 0 :=
by sorry

end complex_number_location_second_quadrant_l717_717452


namespace white_faces_common_sides_at_least_n_minus_2_l717_717542

open Set

noncomputable def polyhedron_faces : Type := sorry

def colored_faces (n : ℕ) (polyhedron : polyhedron_faces) : Prop := sorry
def black_faces_disjoint_vertices (polyhedron : polyhedron_faces) : Prop := sorry
def white_faces_common_sides (polyhedron : polyhedron_faces) : Prop := sorry

theorem white_faces_common_sides_at_least_n_minus_2
  (n : ℕ)
  (polyhedron : polyhedron_faces)
  (h1 : colored_faces n polyhedron)
  (h2 : black_faces_disjoint_vertices polyhedron) :
  ∃ (k : ℕ), (white_faces_common_sides polyhedron) ≥ n - 2 := sorry

end white_faces_common_sides_at_least_n_minus_2_l717_717542


namespace length_of_AE_l717_717494

variables (A B C D E : Type) [Inhabited A] 
variables (dist : A → A → ℝ)

-- Represent lengths in the quadrilateral
variables (AB AC BD CD AE EC : ℝ)
variables (area : A → A → A → ℝ)

-- Given conditions
variables (h1 : dist A B = 8)
variables (h2 : dist C D = 16)
variables (h3 : dist A C = 20)
variables (h4 : ∃ E, true) -- Existence of intersection E
variables (h5 : ∀ E, area A E D = area B E C)

-- Prove the length of AE
theorem length_of_AE : AE = 10 :=
by sorry

end length_of_AE_l717_717494


namespace angle_ABC_is_83_l717_717383

-- Define a structure for the quadrilateral ABCD 
structure Quadrilateral (A B C D : Type) :=
  (angle_BAC : ℝ) -- Measure in degrees
  (angle_CAD : ℝ) -- Measure in degrees
  (angle_ACD : ℝ) -- Measure in degrees
  (side_AB : ℝ) -- Lengths of sides
  (side_AD : ℝ)
  (side_AC : ℝ)

-- Define the conditions from the problem
variable {A B C D : Type}
variable (quad : Quadrilateral A B C D)
variable (h1 : quad.angle_BAC = 60)
variable (h2 : quad.angle_CAD = 60)
variable (h3 : quad.angle_ACD = 23)
variable (h4 : quad.side_AB + quad.side_AD = quad.side_AC)

-- State the theorem to be proved
theorem angle_ABC_is_83 : quad.angle_ACD = 23 → quad.angle_CAD = 60 → 
                           quad.angle_BAC = 60 → quad.side_AB + quad.side_AD = quad.side_AC → 
                           ∃ angle_ABC : ℝ, angle_ABC = 83 := by
  sorry

end angle_ABC_is_83_l717_717383


namespace number_of_children_l717_717654

-- Definitions for the conditions
def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 3
def total_amount : ℕ := 35

-- Theorem stating the proof problem
theorem number_of_children (A C T : ℕ) (hc: A = adult_ticket_cost) (ha: C = child_ticket_cost) (ht: T = total_amount) :
  (T - A) / C = 9 :=
by
  sorry

end number_of_children_l717_717654


namespace _l717_717865

variables {α : Type*} [EuclideanGeometry α]

open EuclideanGeometry

noncomputable def median (A B C M : α) : Prop := 
  midpoint A C M

noncomputable theorem parallelogram_proof 
  (A B C D E M : α)
  (h_triangle : triangle A B C)
  (h_median : median B C A M)
  (h_point_on_median : M ∈ line B M)
  (h_D_on_median : D ∈ line B (M : set α))
  (h_parallel_DE_AB : parallel (line D E) (line A B))
  (h_parallel_EC_BM : parallel (line E C) (line B M)) :
  BE = AD :=
begin
  sorry
end

end _l717_717865


namespace log_base_ten_five_l717_717405

open Real

noncomputable def solution (a b : ℝ) (h1 : logBase 8 3 = a) (h2 : logBase 3 5 = b) : ℝ :=
(evaluate: ℝ): Prop := logBase 10 5 = (3 * a * b) / (1 + (3 * a * b))

theorem log_base_ten_five (a b : ℝ) (h1 : Real.logBase 8 3 = a) (h2 : Real.logBase 3 5 = b) : 
  Real.logBase 10 5 = (3 * a * b) / (1 + 3 * a * b) := sorry

end log_base_ten_five_l717_717405


namespace probability_given_conditions_l717_717899

noncomputable def y_uniform (y : ℝ) : Prop := 50 ≤ y ∧ y ≤ 150

theorem probability_given_conditions : 
  ∀ (y : ℝ), y_uniform y → (⌊real.sqrt (2 * y)⌋ = 16) → (∅ = {y | (⌊real.sqrt (200 * y)⌋ = 180)}) :=
by
  intros y hy h2
  have h2_range : 128 ≤ y ∧ y < 144.5 :=
    begin
      have : 16 ≤ real.sqrt (2 * y) < 17 := h2,
      split,
      { exact le_of_lt (real.sqrt_le_sqrt (by norm_num) (by linarith), by norm_num, real.sqrt_sqr _, norm_num)},
      { exact lt_of_lt_of_le (lt_of_le_of_ne (real.sqrt_lt_sqrt _, by linarith) (real.sqrt_lt_sqrt _, by linarith))}
    end,
  have h200_range : 162 ≤ y ∧ y < 165.045 :=
    begin
      have : 180 ≤ real.sqrt (200 * y) < 181,
      split,
      { exact le_of_lt (real.sqrt_le_sqrt (by norm_num) (by linarith), by norm_num, real.sqrt_sqr _, norm_num)},
      { exact lt_of_lt_of_le (lt_of_le_of_ne (real.sqrt_lt_sqrt _, by linarith) (real.sqrt_lt_sqrt _, by linarith))}
    end,
  rw set.eq_empty_iff_forall_not_mem,
  intros y hy,
  cases h2_range with hy1 hy2,
  cases h200_range with hy3 hy4,
  exact not_and_distrib.mpr (or.inl (not_le.mpr (by linarith)))

end probability_given_conditions_l717_717899


namespace sum_of_coefficients_l717_717253

theorem sum_of_coefficients : 
  (∑ coeff in (x + 2 * y - 1)^6, coeff) = 64 := 
by {
  let polynomial := (x + 2 * y - 1)^6,
  let sum_coeff := polynomial.eval (1, 1),
  have h : sum_coeff = 64, sorry,
  exact h,
}

end sum_of_coefficients_l717_717253


namespace correct_conclusion_l717_717767

-- Given function definition
def f (x : ℝ) (m n : ℝ) : ℝ := log x + m * x + n

-- Assumption that there exists x in (0, +∞) such that f(x) ≥ 0
def exists_x_in_domain (m n : ℝ) : Prop :=
  ∃ x : ℝ, 0 < x ∧ f(x, m, n) ≥ 0

-- Constants m and n with m < 0
variables (m n : ℝ) (h_m_lt_zero : m < 0)

-- The main theorem to be proven
theorem correct_conclusion (h_exists_x : exists_x_in_domain m n) : 
  n - 1 ≥ log (-m) :=
sorry

end correct_conclusion_l717_717767


namespace find_angle_A_find_triangle_area_l717_717902

-- Definition of the problem
variables (A B C : ℝ) (a b c : ℝ)
variables (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
variables (ha : a = Real.sqrt 3)
variables (hc : sin C = (1 + Real.sqrt 3) / 2 * sin B)
variables (eq_cond : b * (sin B - sin C) + (c - a) * (sin A + sin C) = 0)
variable (A_value : A = π / 3)
variable (triangle_area : 1 / 2 * a * b * sin C = (3 + Real.sqrt 3) / 4)

-- Theorem statements
theorem find_angle_A : A = π / 3 :=
sorry

theorem find_triangle_area : 
  (1 / 2 * (Real.sqrt 3) * b * sin ((π / 3 + B) / 2) = (3 + Real.sqrt 3) / 4) :=
sorry

end find_angle_A_find_triangle_area_l717_717902


namespace A_roster_method_l717_717434

open Set

def A : Set ℤ := {x : ℤ | (∃ (n : ℤ), n > 0 ∧ 6 / (5 - x) = n) }

theorem A_roster_method :
  A = {-1, 2, 3, 4} :=
  sorry

end A_roster_method_l717_717434


namespace range_s_l717_717721

open Set

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^3

theorem range_s : range s = univ \ {s 2} :=
by
  rw [univ_eq_of_forall_mem, ←image_univ]
  have : ∀ y, ∃ x (hx : x ≠ 2), s x = y,
  { intro y,
    use [2 + real.cbrt (1/y), by {
        rw [ereal.cbrt_infi_eq_infi_cbrt_to_real hessian_earth_kent],
        exact 2_ne_3]}, sorry] }

  exact eq.symm (range_eq_of_forall_mem this)

end range_s_l717_717721


namespace tangent_ellipse_hyperbola_l717_717950

theorem tangent_ellipse_hyperbola {m : ℝ} :
    (∀ x y : ℝ, x^2 + 9*y^2 = 9 → x^2 - m*(y + 1)^2 = 1 → false) →
    m = 72 :=
sorry

end tangent_ellipse_hyperbola_l717_717950


namespace james_sushi_rolls_l717_717871

def fish_for_sushi : ℕ := 40
def total_fish : ℕ := 400
def bad_fish_percentage : ℕ := 20

theorem james_sushi_rolls :
  let good_fish := total_fish - (bad_fish_percentage * total_fish / 100)
  good_fish / fish_for_sushi = 8 :=
by
  sorry

end james_sushi_rolls_l717_717871


namespace find_number_l717_717759

theorem find_number : ∃ n : ℕ, (∃ x : ℕ, x / 15 = 4 ∧ x^2 = n) ∧ n = 3600 := 
by
  sorry

end find_number_l717_717759


namespace proportion_of_mothers_full_time_jobs_l717_717113

theorem proportion_of_mothers_full_time_jobs
  (P : ℝ) (W : ℝ) (F : ℝ → Prop) (M : ℝ)
  (hwomen : W = 0.4 * P)
  (hfathers_full_time : ∀ p, F p → p = 0.75)
  (hno_full_time : P - (W + 0.75 * (P - W)) = 0.19 * P) :
  M = 0.9 :=
by
  sorry

end proportion_of_mothers_full_time_jobs_l717_717113


namespace papayas_needed_l717_717116

theorem papayas_needed
    (jake_papayas_per_week : ℕ)
    (brother_papayas_per_week : ℕ)
    (father_papayas_per_week : ℕ)
    (weeks : ℕ) :
    jake_papayas_per_week = 3 →
    brother_papayas_per_week = 5 →
    father_papayas_per_week = 4 →
    weeks = 4 →
    (jake_papayas_per_week + brother_papayas_per_week + father_papayas_per_week) * weeks = 48 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end papayas_needed_l717_717116


namespace maximum_k_value_l717_717387

noncomputable def f (x : ℝ) : ℝ := x * (1 + Real.log x)

theorem maximum_k_value : 
  ∃ k : ℤ, (∀ x > 2, (k : ℝ) * (x - 2) < f x) ∧ (∀ k' : ℤ, ((∀ x > 2, (k' : ℝ) * (x - 2) < f x) → k' ≤ k)) :=
begin
  use 4,
  sorry
end

#print maximum_k_value

end maximum_k_value_l717_717387


namespace log_product_eq_k_l717_717867

theorem log_product_eq_k :
  ∃ a : ℝ, (1 ≤ a ∧ a < 10) →
  (∃ k : ℤ, log 10 (2007^2006 * 2006^2007) = a * 10 ^ k ∧ k = 4) :=
sorry

end log_product_eq_k_l717_717867


namespace segments_form_triangle_l717_717495

open Real

theorem segments_form_triangle (P A B C M : Point) (h_proj : IsProjection P M (PlaneContaining A B C))
  (a b c : ℝ) (hP : PA = a ∧ PB = b ∧ PC = c) (h_triangle_PA_PB_PC : a + b > c) :
  let MA := dist M A;
  let MB := dist M B;
  let MC := dist M C;
  √(MA ^ 2 + MB ^ 2 + MC ^ 2) < √(a^2 + dist PM^2) + √(b^2 + dist PM^2) :=
sorry

end segments_form_triangle_l717_717495


namespace total_tickets_sold_l717_717242

def ticket_prices : Nat := 25
def senior_ticket_price : Nat := 15
def total_receipts : Nat := 9745
def senior_tickets_sold : Nat := 348
def adult_tickets_sold : Nat := (total_receipts - senior_ticket_price * senior_tickets_sold) / ticket_prices

theorem total_tickets_sold : adult_tickets_sold + senior_tickets_sold = 529 :=
by
  sorry

end total_tickets_sold_l717_717242


namespace min_value_h15_l717_717502

noncomputable def h : ℕ → ℝ := sorry

theorem min_value_h15 : 
  (∀ x y : ℕ, (0 < x ∧ 0 < y) → h(x) + h(y) > 2 * y^2) →
  (∀ n : ℕ, (1 ≤ n ∧ n ≤ 20) → ∑ i in (finset.range 20).erase 0, h (i + 1) ≥ 4990) →
  h 15 = 328 :=
sorry

end min_value_h15_l717_717502


namespace minimum_value_expression_l717_717248

theorem minimum_value_expression (x : ℝ) : 
  ∃ y : ℝ, (y = (x+1)*(x+2)*(x+3)*(x+4) + 2023) ∧ ∀ z : ℝ, z = (x+1)*(x+2)*(x+3)*(x+4) + 2023 → y ≤ z :=
begin
  sorry
end

end minimum_value_expression_l717_717248


namespace tv_tower_height_greater_l717_717215

theorem tv_tower_height_greater (d : ℝ) (θ : ℝ) (H : ℝ)
  (h_condition : d = 100)
  (θ_condition : θ = 46) :
  103.3 < 100 * Real.tan (θ * Real.pi / 180) :=
by
  rw [h_condition, θ_condition]
  sorry

end tv_tower_height_greater_l717_717215


namespace concentrate_size_408_l717_717672

-- Define the conditions
def can_of_concentrate_to_water_ratio := 1 / 4
def total_servings := 272
def serving_size := 6
def total_volume := total_servings * serving_size
def concentrate_volume (C : ℕ) := 4 * C

-- The theorem to prove that the size of each can of concentrate is 408 ounces
theorem concentrate_size_408 :
  ∃ C : ℕ, concentrate_volume C = total_volume → C = 408 :=
by
  have ht : total_volume = 1632 := by sorry
  have h_cons : concentrate_volume 408 = 1632 := by sorry
  use 408
  intro h
  rw [h_cons] at h
  exact h

end concentrate_size_408_l717_717672


namespace net_profit_calculation_l717_717299

-- Define the conditions as variables and constants.
variable (SP : ℝ) (gross_profit_percent : ℝ) (tax_percent : ℝ) (discount_percent : ℝ)
variable (CP : ℝ) (gross_profit : ℝ)

-- Initialize the conditions
noncomputable def sp_init := 54
noncomputable def gross_profit_percentage := 1.25
noncomputable def tax_percentage := 0.1
noncomputable def discount_percentage := 0.05

-- Define the sales price after discount
noncomputable def sp_after_discount (SP : ℝ) (discount_percent : ℝ) : ℝ :=
  SP * (1 - discount_percent)

-- Define the tax amount
noncomputable def tax (SP : ℝ) (tax_percent : ℝ) : ℝ :=
  SP * tax_percent

-- Define the actual sales price received from the customer after tax
noncomputable def actual_sp (sp_after_discount : ℝ) (tax : ℝ) : ℝ :=
  sp_after_discount + tax

-- Define the cost price using gross profit percentage
noncomputable def cost_price (sp_after_discount : ℝ) (gross_profit_percentage : ℝ) : ℝ :=
  sp_after_discount / gross_profit_percentage

-- Define the net profit
noncomputable def net_profit (actual_sp : ℝ) (CP : ℝ) : ℝ :=
  actual_sp - CP

theorem net_profit_calculation :
  net_profit (actual_sp (sp_after_discount sp_init discount_percentage) (tax sp_init tax_percentage)) 
             (cost_price (sp_after_discount sp_init discount_percentage) gross_profit_percentage)
  = 15.66 :=
by
  sorry

end net_profit_calculation_l717_717299


namespace intersect_at_centroid_l717_717041

-- Define the triangle and centroid
variable (A B C G : Point)
axiom centroid (A B C : Point) : Point

-- Define a line and distances from points to the line
variable (l : Line)
axiom distance (P : Point) (l : Line) : Real

-- Given conditions: distances are defined with the relationship
axiom dA_eq_dB_plus_dC
  (A B C : Point) (l : Line) : distance A l = distance B l + distance C l

-- The theorem to prove: all such lines intersect at the centroid
theorem intersect_at_centroid
  (A B C : Point) (G : Point) (l : Line) (hG : G = centroid A B C) :
  (distance A l = distance B l + distance C l) → Line.intersects l (Line.through G) :=
sorry

end intersect_at_centroid_l717_717041


namespace solve_equation_1_solve_equation_2_l717_717935

theorem solve_equation_1 (x : ℝ) : x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 :=
by sorry

theorem solve_equation_2 (x : ℝ) : 3 * x^2 + 2 * x - 1 = 0 ↔ x = 1 / 3 ∨ x = -1 :=
by sorry

end solve_equation_1_solve_equation_2_l717_717935


namespace part1_part2_l717_717077

section

variable (a : ℝ) (a_seq : ℕ → ℝ)
variable (h_seq : ∀ n, a_seq (n + 1) = (5 * a_seq n - 8) / (a_seq n - 1))
variable (h_initial : a_seq 1 = a)

-- Part 1:
theorem part1 (h_a : a = 3) : 
  ∃ r : ℝ, ∀ n, (a_seq n - 2) / (a_seq n - 4) = r ^ n ∧ a_seq n = (4 * 3 ^ (n - 1) + 2) / (3 ^ (n - 1) + 1) := 
sorry

-- Part 2:
theorem part2 (h_pos : ∀ n, a_seq n > 3) : 3 < a := 
sorry

end

end part1_part2_l717_717077


namespace solve_for_x_l717_717932

variable (x : ℝ)
axiom h : 3 / 4 + 1 / x = 7 / 8

theorem solve_for_x : x = 8 :=
by
  sorry

end solve_for_x_l717_717932


namespace haley_watch_time_l717_717817

def shopping_time : ℝ := 2
def setup_time : ℝ := 0.5
def snack_time : ℝ := 3 * setup_time
def watch_fraction : ℝ := 0.08

theorem haley_watch_time :
  let total_time_before_watching := (shopping_time + setup_time + snack_time) * 60 in
  let watch_time := watch_fraction * (total_time_before_watching / (1 - watch_fraction)) in
  round watch_time = 21 :=
by
  let total_time_before_watching := (shopping_time + setup_time + snack_time) * 60
  let watch_time := watch_fraction * (total_time_before_watching / (1 - watch_fraction))
  have : round watch_time = 21
  sorry

end haley_watch_time_l717_717817


namespace max_value_of_x_plus_reciprocal_of_x_l717_717235

theorem max_value_of_x_plus_reciprocal_of_x
  (n : ℕ)
  (x : ℝ)
  (y : Fin n → ℝ)
  (sum_x_reciprocals : (∑ i, y i) = 3012 - x ∧ (∑ i, (1 / y i)) = 3012 - (1 / x))
  (hx : 0 < x)
  (hy : ∀ i, 0 < y i):
  x + (1 / x) ≤ 12073 / 3012 :=
    sorry

end max_value_of_x_plus_reciprocal_of_x_l717_717235


namespace locus_of_point_x_l717_717164

theorem locus_of_point_x (P Q X A B C D : Point) (circumcircle : circle A B C D) 
  (hP : P ∈ circumcircle)
  (O : Point) (hO : O = midpoint A C ∧ O = midpoint B D)
  (hPQ : intersect (line_through A P) (line_through B D) = some Q)
  (hline_Q : parallel (line_through Q X) (line_through A C))
  (hX : intersect (line_through B P) (line_through Q X) = some X)
  : X ∈ (line_through C D) :=
sorry

end locus_of_point_x_l717_717164


namespace max_contestants_rock_paper_scissors_l717_717848

theorem max_contestants_rock_paper_scissors (contests : Fin 1024 → Fin 10 → ℕ) : 
  (∀ (i j : Fin 1024), ∃ (k : Fin 10), contests i k ≠ contests j k) → 
  Fintype.card (Fin 1024) ≤ 1024 :=
begin
  sorry
end

end max_contestants_rock_paper_scissors_l717_717848


namespace gumball_difference_l717_717691

theorem gumball_difference :
    let c := 17
    let l := 12
    ∃ x : ℕ, 19 ≤ (c + l + x) / 3 ∧ (c + l + x) / 3 ≤ 25 ∧ 
    (46 - 28 = 18) :=
by
    let c := 17
    let l := 12
    use (x : ℕ)
    exact and.intro (57 ≤ 29 + x) (29 + x ≤ 75)
    exact sorry

end gumball_difference_l717_717691


namespace parabola_directrix_circle_l717_717558

theorem parabola_directrix_circle (x y : ℝ) :
  (∀ y, y^2 = 8 - 4 * x ↔ (y^2 = -4 * (x - 2))) →
  (∃ c, directrix_eq c → c = 3) ∧
  (∃ (h k r : ℝ), circle_eq h k r → h = 2 ∧ k = 0 ∧ r = 1) :=
begin
  sorry
end

end parabola_directrix_circle_l717_717558


namespace correct_condition_l717_717560

/-- 
Given a program that initializes i as 12 and s as 1, loops with s = s * i and i = i - 1 until a condition, 
and returns the final product s as 132, prove that the condition must be i < 11. 
-/
theorem correct_condition (i s : ℕ) (h1 : i = 12) (h2 : s = 1) (h3 : ∀ n, s = 132) :
  (condition : ℕ → Prop) := sorry

end correct_condition_l717_717560


namespace find_english_score_l717_717511

-- Define the scores
def M : ℕ := 82
def K : ℕ := M + 5
variable (E : ℕ)

-- The average score condition
axiom avg_condition : (K + E + M) / 3 = 89

-- Our goal is to prove that E = 98
theorem find_english_score : E = 98 :=
by
  -- The proof will go here
  sorry

end find_english_score_l717_717511


namespace product_of_distinct_elements_of_T_l717_717888

def T := { n : ℕ | n > 0 ∧ n ∣ 72000 }

def count_distinct_products (S : set ℕ) : ℕ :=
  S.to_finset.card * (S.to_finset.card - 1) / 2

theorem product_of_distinct_elements_of_T :
  count_distinct_products T = 379 :=
by
  sorry

end product_of_distinct_elements_of_T_l717_717888


namespace greatest_k_subsets_condition_l717_717303

-- Define M as a set with n elements
def M (n : ℕ) : Type := Fin n

-- Define the problem statement as a theorem
theorem greatest_k_subsets_condition (n : ℕ) : ∃ k, (∀ (A : Finset (Fin n)), ∃ (i : Fin n), i ∈ A ∧ (∀ j, j ∈ A → (coefficient i j = 1)) → k = n :=
sorry

end greatest_k_subsets_condition_l717_717303


namespace volume_of_triangular_prism_l717_717533

theorem volume_of_triangular_prism (S_side_face : ℝ) (distance : ℝ) :
  ∃ (Volume_prism : ℝ), Volume_prism = 1/2 * (S_side_face * distance) :=
by sorry

end volume_of_triangular_prism_l717_717533


namespace find_f_4_l717_717057

-- Define the conditions
variables {ℝ : Type*} [linear_ordered_field ℝ]

variables (f : ℝ → ℝ)
variable (x : ℝ)

-- Condition: f(x+1) is odd
def odd_shifted_function : Prop := ∀ x, f(-(x) + 1) = -f(x + 1)

-- Condition: f(x-1) is even
def even_shifted_function : Prop := ∀ x, f(x - 1) = f(-(x) - 1)

-- Condition: f(0) = 2
def initial_value_condition : Prop :=
  f(0) = 2

-- Theorem to prove: f(4) = -2
theorem find_f_4 (h1 : odd_shifted_function f) (h2 : even_shifted_function f) (h3 : initial_value_condition f) : 
  f 4 = -2 :=
sorry

end find_f_4_l717_717057


namespace min_max_abs_expression_l717_717730

theorem min_max_abs_expression : 
    (min (\lam y, (max (\lam x, abs (x^3 - x*y)) (0 : Real) 1)) (0 : Real) 2) = 0 := 
by 
    sorry

end min_max_abs_expression_l717_717730


namespace find_m_l717_717811

open Set

variable (A B : Set ℝ) (m : ℝ)

theorem find_m (h : A = {-1, 2, 2 * m - 1}) (h2 : B = {2, m^2}) (h3 : B ⊆ A) : m = 1 := 
by
  sorry

end find_m_l717_717811


namespace quadruple_lines_intersection_on_circle_n_lines_cyclic_property_l717_717145

-- Problem (a)
theorem quadruple_lines_intersection_on_circle (l1 l2 l3 l4 : Line) (M : Point) :
  ∀ (M1 M2 M3 M4 : Point),
    (on_circle M (circle M1 M2 M)
    ∧ on_circle M (circle M2 M3 M)
    ∧ on_circle M (circle M3 M4 M)
    ∧ on_circle M (circle M4 M1 M))
    → ∃ (A1 A2 A3 A4 : Point),
        (on_circle A1 (circle M2 M3 M)
        ∧ on_circle A2 (circle M3 M4 M)
        ∧ on_circle A3 (circle M4 M1 M)
        ∧ on_circle A4 (circle M1 M2 M)).
sorry

-- Problem (b)
theorem n_lines_cyclic_property (n : ℕ) (h_odd : n % 2 = 1 ∨ n % 2 = 0) (M : Point) :
  ∀ (L : Finₙ (List Line)) (Points : Finₙ (List Point)) (C : Circle),
    (all_points_on_circle L Points C) →
    ∃ (Result : Type), equivalent_cyclic_property Result L Points.
sorry

end quadruple_lines_intersection_on_circle_n_lines_cyclic_property_l717_717145


namespace quadrilateral_centroid_area_is_correct_l717_717544

noncomputable def quadrilateral_centroid_area (EFGH : Type) (Q : EFGH) [MetricSpace EFGH] :=
  side : ℝ,
  point_eq : ℝ → Prop,
  point_fq : ℝ → Prop

theorem quadrilateral_centroid_area_is_correct:
  (side = 40) → 
  (point_eq 16) →
  (point_fq 34) →
  quadrilateral_centroid_area EFGH Q = 800 / 9 :=
by
  sorry

end quadrilateral_centroid_area_is_correct_l717_717544


namespace rational_terms_count_l717_717249

open Nat

theorem rational_terms_count : 
  let m := 1200
  let n := m / 20 + 1  -- number of multiples of 20 from 0 to 1200
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  n = 61 := 
  by
    sorry

end rational_terms_count_l717_717249


namespace composite_n_property_l717_717348

theorem composite_n_property (n : ℕ) (h1 : ∃ k > 1, n = d_k (λ m, ∃ d : ℕ, m = d * d ∧ 1 = d_1 < d_2 < ··· < d_k = m) )
  (h2 : ∀ i : ℕ, 2 ≤ i → i ≤ k → ∃ m : ℤ , m = d_i - d_(i-1) ∧ (d_2 - d_1) : (d_3 - d_2) : ··· : (d_k - d_(k-1)) = 1:2:···:(k-1)) : 
  n = 4 := 
sorry

end composite_n_property_l717_717348


namespace smallest_common_multiple_l717_717739

theorem smallest_common_multiple (b : ℕ) (hb : b > 0) (h1 : b % 6 = 0) (h2 : b % 15 = 0) :
    b = 30 :=
sorry

end smallest_common_multiple_l717_717739


namespace mean_score_of_seniors_l717_717522

theorem mean_score_of_seniors:
  ∀ (s j : ℕ) (m_s m_j : ℝ),
  s + j = 120 ∧
  j = 0.75 * s ∧
  69 * m_s + 51 * m_j = 13200 ∧
  m_s = 1.4 * m_j → m_s = 124 :=
by
  -- the proof is here
  sorry

end mean_score_of_seniors_l717_717522


namespace min_dot_product_l717_717781

noncomputable def vector_dot_product_min (a b e : ℝ × ℝ) : ℝ :=
  if abs (real.norm_sq e - 1) <= 0 ∧
     (a.1 * e.1 + a.2 * e.2 = 2) ∧
     (b.1 * e.1 + b.2 * e.2 = 3) ∧
     (real.norm_sq (a.1 - b.1, a.2 - b.2) = 5) then 
    real.min (a.1 * b.1 + a.2 * b.2) else 
    0

theorem min_dot_product (a b e : ℝ × ℝ)
  (h1 : real.norm_sq e = 1)
  (h2 : a.1 * e.1 + a.2 * e.2 = 2)
  (h3 : b.1 * e.1 + b.2 * e.2 = 3)
  (h4 : real.norm_sq (a.1 - b.1, a.2 - b.2) = 5) : 
  a.1 * b.1 + a.2 * b.2 = 5 := sorry

end min_dot_product_l717_717781


namespace divisors_of_9_factorial_greater_than_8_factorial_l717_717820

theorem divisors_of_9_factorial_greater_than_8_factorial :
  {d // d ∣ fact 9 ∧ d > fact 8}.card = 8 := 
sorry

end divisors_of_9_factorial_greater_than_8_factorial_l717_717820


namespace tetrahedron_planes_l717_717970

theorem tetrahedron_planes {a b c d e f S : ℝ} 
  (hS : S = a^2 + b^2 + c^2 + d^2 + e^2 + f^2) :
  ∃ (x : ℝ), x = sqrt (S / 12) ∧ (∃ (P Q R S : ℝ^3),  -- Assuming coordinates for tetrahedron vertices
  (dist P Q = a ∧ dist P R = b ∧ dist P S = c ∧
   dist Q R = d ∧ dist R S = e ∧ dist S Q = f) ∧
   (∃ (plane1 plane2 : ℝ → ℝ), -- Defining the parallel planes
     (∀ (p1 p2 : ℝ^3), 
      (plane1 p1 = 0 → plane2 p2 = 0 → abs (dist p1 p2) = x)))
  ) :=
sorry

end tetrahedron_planes_l717_717970


namespace am_gm_ineq_equality_case_l717_717045

theorem am_gm_ineq (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : 
  a + b + c >= 3 :=
begin
  sorry
end

theorem equality_case (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : 
  a = 1 ∧ b = 1 ∧ c = 1 ↔ a + b + c = 3 :=
begin
  sorry
end

end am_gm_ineq_equality_case_l717_717045


namespace soccer_team_points_l717_717193

theorem soccer_team_points 
  (total_games : ℕ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (points_per_win : ℕ) 
  (points_per_draw : ℕ) 
  (points_per_loss : ℕ) 
  (draws : ℕ := total_games - (wins + losses)) : 
  total_games = 20 →
  wins = 14 →
  losses = 2 →
  points_per_win = 3 →
  points_per_draw = 1 →
  points_per_loss = 0 →
  46 = (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) :=
by sorry

end soccer_team_points_l717_717193


namespace sum_coordinates_l717_717237

theorem sum_coordinates :
  let points : List (ℝ × ℝ) :=
    [ (7 + Real.sqrt 176, 20), (7 - Real.sqrt 176, 20),
      (7 + Real.sqrt 176, 6), (7 - Real.sqrt 176, 6) ]
  ∑ p in points, p.1 + p.2 = 74 := by
  sorry

end sum_coordinates_l717_717237


namespace n_students_even_l717_717352

-- Given n students where each pair of students have different ages
-- Each student shook hands with at least one student who did not shake hands with anyone younger than the other.
-- We need to prove that the number of students n is even.

theorem n_students_even (n : ℕ)
  (h_age_diff : ∀ i j, i < n → j < n → i ≠ j → (i ≠ j → i ≠ j))
  (h_handshake : ∀ (students : fin n → Type), ∀ i : fin n,
                  ∃ j : fin n, i ≠ j ∧ 
                              (∀ k : fin n, (j, k, i ≠ j) → k = j ∨ k = i)) :
  n % 2 = 0 :=
sorry

end n_students_even_l717_717352


namespace terms_selection_l717_717039

theorem terms_selection (k : ℕ) (h_k : 1 < k) 
  (a : ℕ → ℝ) 
  (h_decreasing : ∀ n, a n ≥ a (n + 1)) 
  (h_sum : ∑' n, a n = 1) 
  (h_first : a 0 = 1 / (2 * k)) : 
  ∃ (b : fin n → ℝ), (∀ i j, b i ≥ b j / 2) ∧ ∀ i < j, b i = a (i + j) :=
sorry

end terms_selection_l717_717039


namespace select_four_such_that_product_is_square_l717_717795

theorem select_four_such_that_product_is_square
  (nums : Fin 48 → ℕ)
  (h : (∀ primes, (∏ i in primes, nums i).prime_factors.card ≤ 10)) :
  ∃ a b c d : Fin 48, ∃ (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ d) (h₄ : d ≠ a),
  is_square (nums a * nums b * nums c * nums d) :=
by { sorry }

end select_four_such_that_product_is_square_l717_717795


namespace determine_q_l717_717347

theorem determine_q (q : ℝ → ℝ) :
  (∀ x, q x = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ k : ℝ, k < 3) ∧ -- indicating degree considerations for asymptotes
  (q 2 = 18) →
  q = (fun x => (-18 / 5) * x ^ 2 + 162 / 5) :=
by
  sorry

end determine_q_l717_717347


namespace probability_f_leq_zero_equal_l717_717068

noncomputable theory

open Set

def f (x : ℝ) : ℝ := x^2 - x - 2

def domain : Set ℝ := Icc (-5) 5

def interval_satisfying : Set ℝ := Icc (-1) 2

theorem probability_f_leq_zero_equal :
  (interval_satisfying ∩ domain).measure / domain.measure = 3 / 10 :=
by
  sorry

end probability_f_leq_zero_equal_l717_717068


namespace paul_bags_on_saturday_l717_717525

-- Definitions and Conditions
def total_cans : ℕ := 72
def cans_per_bag : ℕ := 8
def extra_bags : ℕ := 3

-- Statement of the problem
theorem paul_bags_on_saturday (S : ℕ) :
  S * cans_per_bag = total_cans - (extra_bags * cans_per_bag) →
  S = 6 :=
sorry

end paul_bags_on_saturday_l717_717525


namespace corrected_observations_mean_l717_717563

noncomputable def corrected_mean (mean incorrect correct: ℚ) (n: ℕ) : ℚ :=
  let S_incorrect := mean * n
  let Difference := correct - incorrect
  let S_corrected := S_incorrect + Difference
  S_corrected / n

theorem corrected_observations_mean:
  corrected_mean 36 23 34 50 = 36.22 := by
  sorry

end corrected_observations_mean_l717_717563


namespace least_positive_integer_satisfying_conditions_l717_717247

theorem least_positive_integer_satisfying_conditions :
  ∃ (a : ℕ), a % 3 = 2 ∧ a % 4 = 3 ∧ a % 5 = 4 ∧ a % 6 = 5 ∧ (∀ b : ℕ, (b % 3 = 2 ∧ b % 4 = 3 ∧ b % 5 = 4 ∧ b % 6 = 5) → b ≥ a) ∧ a = 119 :=
begin
  sorry
end

end least_positive_integer_satisfying_conditions_l717_717247


namespace locus_of_X_is_line_CD_l717_717162

open Real EuclideanGeometry

/-- Setting up the problem with given conditions -/
variables (A B C D P Q X : Point)
variables (circ_circle : Circle)
variables (AP_line BD_line AC_line BP_line : Line)
variables (locus_X : Set Point)

/-- Defining the square ABCD and its properties -/
def square (A B C D : Point) : Prop :=
  is_square A B C D

/-- Defining the circumcircle of the square -/
def circumscribed_circle (circ_circle : Circle) (A B C D : Point) : Prop :=
  circ_circle = circumcircle A B C D

/-- Point P moves along the circumcircle of square ABCD -/
def P_moves_along_circumcircle (circ_circle : Circle) (P : Point) : Prop :=
  P ∈ circ_circle

/-- Lines AP and BD intersect at point Q -/
def AP_BD_intersect_at_Q (AP_line BD_line : Line) (Q : Point) : Prop :=
  Q ∈ AP_line ∧ Q ∈ BD_line

/-- Line through Q parallel to AC intersects line BP at point X -/
def Q_parallel_AC_B_intersects_BP_at_X (Q : Point) (AC_line BP_line X : Point) : Prop :=
  parallel (line_through Q AC_line) AC_line ∧ X ∈ BP_line

/-- The locus of point X is the line CD -/
theorem locus_of_X_is_line_CD (A B C D P Q X : Point) (circ_circle : Circle) 
  (AP_line BD_line AC_line BP_line : Line) (locus_X : Set Point) :
  square A B C D →
  circumscribed_circle circ_circle A B C D →
  P_moves_along_circumcircle circ_circle P →
  AP_BD_intersect_at_Q AP_line BD_line Q →
  Q_parallel_AC_B_intersects_BP_at_X Q AC_line BP_line X →
  ∀ X ∈ locus_X, collinear {C, D, X} :=
sorry

end locus_of_X_is_line_CD_l717_717162


namespace find_xy_l717_717886

theorem find_xy (p x y : ℕ) (hp : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  p * (x - y) = x * y ↔ (x, y) = (p^2 - p, p + 1) := by
  sorry

end find_xy_l717_717886


namespace inscribed_circle_area_gt_half_l717_717394

theorem inscribed_circle_area_gt_half (T : Set (Triangle)) 
  (h : ∀ t ∈ T, ∃ (a b c : ℝ), RightTriangle t a b c ∧ altitude_to_hypotenuse t = 1) :
  ∀ t ∈ T, ∃ R : ℝ, inscribed_circle_area t R ∧ R > 0.5 := 
sorry

end inscribed_circle_area_gt_half_l717_717394


namespace least_positive_integer_l717_717592

theorem least_positive_integer (k : ℕ) (h : (528 + k) % 5 = 0) : k = 2 :=
sorry

end least_positive_integer_l717_717592


namespace perfect_square_is_289_l717_717973

/-- The teacher tells a three-digit perfect square number by
revealing the hundreds digit to person A, the tens digit to person B,
and the units digit to person C, and tells them that all three digits
are different from each other. Each person only knows their own digit and
not the others. The three people have the following conversation:

Person A: I don't know what the perfect square number is.  
Person B: You don't need to say; I also know that you don't know.  
Person C: I already know what the number is.  
Person A: After hearing Person C, I also know what the number is.  
Person B: After hearing Person A also knows what the number is.

Given these conditions, the three-digit perfect square number is 289. -/
theorem perfect_square_is_289:
  ∃ n : ℕ, n^2 = 289 := by
  sorry

end perfect_square_is_289_l717_717973


namespace beach_ball_properties_l717_717627

theorem beach_ball_properties :
  let d : ℝ := 18
  let r : ℝ := d / 2
  let surface_area : ℝ := 4 * π * r^2
  let volume : ℝ := (4 / 3) * π * r^3
  surface_area = 324 * π ∧ volume = 972 * π :=
by
  sorry

end beach_ball_properties_l717_717627


namespace ring_arrangements_leftmost_three_digits_l717_717399

-- Definitions based on the given conditions
def total_arrangements (r: ℕ) (c: ℕ) (f: ℕ) := nat.choose r c * nat.choose (c + f - 1) (f - 1) * nat.factorial (c)

-- Theorem statement based on the problem
theorem ring_arrangements_leftmost_three_digits :
  total_arrangements 7 6 4 = 423360 → "423" = "423" :=
begin
  sorry
end

end ring_arrangements_leftmost_three_digits_l717_717399


namespace smallest_positive_period_of_f_min_value_of_f_max_value_of_f_l717_717425

noncomputable def f (x : ℝ) := 2 * sin x * cos x + 2 * sqrt 3 * (cos x)^2

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := 
by {
  sorry
}

theorem min_value_of_f : ∀ x ∈ Icc (-π / 3) (π / 3), f x = 0 := 
by {
  sorry
}

theorem max_value_of_f : ∀ x ∈ Icc (-π / 3) (π / 3), f x = 2 + sqrt 3 := 
by {
  sorry
}

end smallest_positive_period_of_f_min_value_of_f_max_value_of_f_l717_717425


namespace arithmetic_sequences_integer_ratios_l717_717814

theorem arithmetic_sequences_integer_ratios 
  (a b : ℕ → ℕ)
  (S T : ℕ → ℕ)
  (hS : ∀ n, S n = (n + 1) * a (n + 1) / 2)
  (hT : ∀ n, T n = (n + 1) * b (n + 1) / 2)
  (hRatio : ∀ n, S n / T n = 2 * n + 30 / (n + 3)) :
  {n : ℕ // ∃ k : ℕ, k * b (n + 1) = a (n + 1)}.1.card = 4 :=
by sorry

end arithmetic_sequences_integer_ratios_l717_717814


namespace complete_the_square_l717_717182

theorem complete_the_square : ∀ x : ℝ, x^2 - 6 * x + 4 = 0 → (x - 3)^2 = 5 :=
by
  intro x h
  sorry

end complete_the_square_l717_717182


namespace least_positive_n_l717_717586

-- Definitions based on conditions
def num_walnuts : ℕ := 2021
def reorder_bushy : ℕ := 1232

-- Statement to prove
theorem least_positive_n (n : ℕ) : (∀ walnuts : list ℕ, walnuts.length = num_walnuts →
  (∀ b_perm : list ℕ, b_perm.length = reorder_bushy → 
    (∃ j_perm : list ℕ, j_perm.length = n ∧
      sorted walnuts[j_perm])) →
  sorted walnuts) :=
  n = 1234 :=
  sorry

end least_positive_n_l717_717586


namespace find_erased_number_l717_717273

theorem find_erased_number :
  ∀ (n : ℕ) (h36 : n = 36)
  (Σ : ℕ → ℕ → ℕ) -- Function giving the number on the segments connecting circle i to circle j
  (F : ℕ → ℕ) -- Function giving the numbers recorded in the circles
  (hF : ∀ i, F i = Σ i (i+1 mod n) + Σ (i - 1) mod n i)
  (R_w R_b M : ℤ)
  (hRwRb : R_w = R_b + M),
  ∃ (M' : ℤ), M' = R_w - R_b := 
by
  sorry

end find_erased_number_l717_717273


namespace eventually_positive_set_l717_717780

noncomputable def transform (s : List ℝ) : List ℝ :=
  match s with
  | [a, b, c, d] => [a * b, b * c, c * d, d * a]
  | _ => s

def eventually_positive (s : List ℝ) : Prop :=
  ∃ n : ℕ, ∀ i < 4, 0 < Nat.iterate transform n s[i]

theorem eventually_positive_set {a b c d : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  eventually_positive [a, b, c, d] :=
sorry

end eventually_positive_set_l717_717780


namespace area_of_inscribed_rectangle_l717_717659

theorem area_of_inscribed_rectangle
  (s : ℕ) (R_area : ℕ)
  (h1 : s = 4) 
  (h2 : 2 * 4 + 1 * 1 + R_area = s * s) :
  R_area = 7 :=
by
  sorry

end area_of_inscribed_rectangle_l717_717659


namespace sin_cos_condition_l717_717202

theorem sin_cos_condition (x : ℝ) : (sin x * cos x > 0) → (¬ (sin x + cos x > 1)) → False :=
by sorry

end sin_cos_condition_l717_717202


namespace range_of_m_l717_717833

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, x^2 - x - m = 0) : m ≥ -1/4 :=
by
  sorry

end range_of_m_l717_717833


namespace sally_lost_orange_balloons_l717_717925

theorem sally_lost_orange_balloons :
  ∀ (initial_orange_balloons lost_orange_balloons current_orange_balloons : ℕ),
  initial_orange_balloons = 9 →
  current_orange_balloons = 7 →
  lost_orange_balloons = initial_orange_balloons - current_orange_balloons →
  lost_orange_balloons = 2 :=
by
  intros initial_orange_balloons lost_orange_balloons current_orange_balloons
  intros h_init h_current h_lost
  rw [h_init, h_current] at h_lost
  exact h_lost

end sally_lost_orange_balloons_l717_717925


namespace smallest_n_for_cube_root_form_l717_717142

theorem smallest_n_for_cube_root_form
  (m n : ℕ) (r : ℝ)
  (h_pos_n : n > 0)
  (h_pos_r : r > 0)
  (h_r_bound : r < 1/500)
  (h_m : m = (n + r)^3)
  (h_min_m : ∀ k : ℕ, k = (n + r)^3 → k ≥ m) :
  n = 13 :=
by
  -- proof goes here
  sorry

end smallest_n_for_cube_root_form_l717_717142


namespace minimize_payment_l717_717578

theorem minimize_payment :
  ∀ (bd_A td_A bd_B td_B bd_C td_C : ℕ),
    bd_A = 42 → td_A = 36 →
    bd_B = 48 → td_B = 41 →
    bd_C = 54 → td_C = 47 →
    ∃ (S : ℕ), S = 36 ∧ 
      (S = bd_A - (bd_A - td_A)) ∧
      (S < bd_B - (bd_B - td_B)) ∧
      (S < bd_C - (bd_C - td_C)) := 
by {
  sorry
}

end minimize_payment_l717_717578


namespace solution_to_equation_l717_717244

theorem solution_to_equation (x y : ℤ) (h : x^6 - y^2 = 648) : 
  (x = 3 ∧ y = 9) ∨ 
  (x = -3 ∧ y = 9) ∨ 
  (x = 3 ∧ y = -9) ∨ 
  (x = -3 ∧ y = -9) :=
sorry

end solution_to_equation_l717_717244


namespace hyperbola_trilinear_coordinates_l717_717737

noncomputable def is_hyperbola_equation (A B C x y z : ℝ) : Prop :=
  (sin (2 * A) * cos (B - C)) / x + (sin (2 * B) * cos (C - A)) / y + (sin (2 * C) * cos (A - B)) / z = 0

theorem hyperbola_trilinear_coordinates (A B C x y z : ℝ) :
  let centroid := (1 / sin A, 1 / sin B, 1 / sin C)
  let orthocenter := (1 / sin A, 1 / sin B, 1 / sin C)
  let euler_line := sin (2 * A) * cos (B - C) * x + sin (2 * B) * cos (C - A) * y + sin (2 * C) * cos (A - B) * z = 0
  is_hyperbola_equation A B C x y z :=
sorry

end hyperbola_trilinear_coordinates_l717_717737


namespace smallest_multiple_of_6_and_15_l717_717755

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ c : ℕ, c > 0 ∧ c % 6 = 0 ∧ c % 15 = 0 → c ≥ b := 
begin
  use 30,
  split,
  { exact nat.succ_pos 29, },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 2 3) (dvd_mul_right 3 5)), },
  split,
  { exact nat.mod_eq_zero_of_dvd (dvd_trans (dvd_mul_right 3 5) (dvd_mul_right 3 2)), },
  { intros c hc1 hc2,
    have hc3 : c % 30 = 0,
    {
      suffices h : c % 6 = 0 ∧ c % 15 = 0 ↔ c % lcm 6 15 = 0,
      { rw ← h, exact ⟨hc1, hc2⟩, },
      exact nat.dvd_iff_mod_eq_zero,
    },
    linarith,
  }
end

end smallest_multiple_of_6_and_15_l717_717755


namespace dwarves_hat_vs_shoeless_l717_717911

-- Define the total number of dwarves
def total_dwarves : ℕ := 25

-- Define the number of dwarves without hats
def dwarves_without_hats : ℕ := 12

-- Define the number of dwarves who came barefoot
def dwarves_barefoot : ℕ := 5

-- Define the condition that each dwarf who wore a hat also wore shoes
def hat_wearing_implies_shoes_wearing (d : ℕ) (h : d ∈ hats) : d ∈ shoes := sorry

-- The set of dwarves who wore hats and the set of dwarves who wore shoes
def dwarves_with_hats := total_dwarves - dwarves_without_hats
def dwarves_with_shoes := total_dwarves - dwarves_barefoot

-- Calculate the number of dwarves who wore shoes but without hats
def dwarves_with_shoes_without_hats := dwarves_with_shoes - dwarves_with_hats

-- The main theorem to prove
theorem dwarves_hat_vs_shoeless :
  dwarves_with_hats = dwarves_with_shoes_without_hats + 6 :=
by
  sorry

end dwarves_hat_vs_shoeless_l717_717911


namespace inverse_f_at_1_l717_717099

def f (x : ℝ) : ℝ := 3^x - 2

theorem inverse_f_at_1 : f⁻¹ 1 = 1 :=
by
  sorry

end inverse_f_at_1_l717_717099


namespace school_children_count_l717_717517

-- Define the conditions
variable (A P C B G : ℕ)
variable (A_eq : A = 160)
variable (kids_absent : ∀ (present kids absent children : ℕ), present = kids - absent → absent = 160)
variable (bananas_received : ∀ (two_per child kids : ℕ), (2 * kids) + (2 * 160) = 2 * 6400 + (4 * (6400 / 160)))
variable (boys_girls : B = 3 * G)

-- State the theorem
theorem school_children_count (C : ℕ) (A P B G : ℕ) 
  (A_eq : A = 160)
  (kids_absent : P = C - A)
  (bananas_received : (2 * P) + (2 * A) = 2 * P + (4 * (P / A)))
  (boys_girls : B = 3 * G)
  (total_bananas : 2 * P + 4 * (P / A) = 12960) :
  C = 6560 := 
sorry

end school_children_count_l717_717517


namespace cookies_remaining_are_correct_l717_717346

-- Define the initial number of white cookies
def initial_white_cookies : ℕ := 200

-- Define the number of black cookies
def initial_black_cookies : ℕ := initial_white_cookies + 125

-- Define the fraction of black cookies eaten
def fraction_black_eaten : ℚ := 3 / 4

-- Define the fraction of white cookies eaten
def fraction_white_eaten : ℚ := 7 / 8

-- Define the number of black cookies eaten, rounding down to the nearest whole cookie
def black_cookies_eaten : ℕ := Int.natAbs (fraction_black_eaten * initial_black_cookies).toRounded.truncate

-- Define the number of white cookies eaten
def white_cookies_eaten : ℕ := Int.natAbs (fraction_white_eaten * initial_white_cookies).toRounded.truncate

-- Define the number of remaining black cookies
def remaining_black_cookies : ℕ := initial_black_cookies - black_cookies_eaten

-- Define the number of remaining white cookies
def remaining_white_cookies : ℕ := initial_white_cookies - white_cookies_eaten

-- Define the total number of remaining cookies
def total_remaining_cookies : ℕ := remaining_black_cookies + remaining_white_cookies

-- The statement to prove
theorem cookies_remaining_are_correct : total_remaining_cookies = 107 :=
  by
    -- We can leave the proof as 'sorry' as per the instruction
    sorry

end cookies_remaining_are_correct_l717_717346


namespace doubly_oddly_powerful_count_l717_717337

theorem doubly_oddly_powerful_count : 
    {n : ℤ | ∃ (a : ℤ) (b : ℕ), 1 < b ∧ b % 2 = 1 ∧ a > 0 ∧ a^b = n ∧ n < 4010}.card = 20 :=
by
 sorry

end doubly_oddly_powerful_count_l717_717337


namespace total_people_in_line_l717_717587

theorem total_people_in_line (n_front n_behind : ℕ) (hfront : n_front = 11) (hbehind : n_behind = 12) : n_front + n_behind + 1 = 24 := by
  sorry

end total_people_in_line_l717_717587


namespace Roger_already_eligible_for_retirement_l717_717537

variables (R P T Rb M S L J : ℕ)

-- Conditions
def condition1 : Prop := R = P + T + Rb + M + S + L + J
def condition2 : Prop := P = 12
def condition3 : Prop := T = 2 * Rb
def condition4 : Prop := Rb = 12 - 4
def condition5 : Prop := Rb = M + 2
def condition6 : Prop := S = M + 3
def condition7 : Prop := S = T / 2
def condition8 : Prop := L = Rb - M
def condition9 : Prop := R = 52 + J
def condition10 : Prop := R >= 50

-- Theorem to prove
theorem Roger_already_eligible_for_retirement :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ 
  condition5 ∧ condition6 ∧ condition7 ∧ condition8 ∧ 
  condition9 ∧ condition10 → R >= 50 :=
by {
  intros h,
  sorry
}

end Roger_already_eligible_for_retirement_l717_717537


namespace exists_ϕ_x0_l717_717137

noncomputable def f (x ϕ : ℝ) : ℝ := sin x * sin (x + ϕ)

theorem exists_ϕ_x0 (ϕ : ℝ) : 
  (∃ x0 : ℝ, f x0 ϕ = 1) ↔ ϕ = 2 * Real.pi :=
by 
  sorry

end exists_ϕ_x0_l717_717137


namespace unique_real_solution_l717_717378

theorem unique_real_solution :
  ∃! x : ℝ, -((x + 2) ^ 2) ≥ 0 :=
sorry

end unique_real_solution_l717_717378


namespace negation_of_prop_p_l717_717073

theorem negation_of_prop_p (p : Prop) (h : ∀ x: ℝ, 0 < x → x > Real.log x) :
  (¬ (∀ x: ℝ, 0 < x → x > Real.log x)) ↔ (∃ x_0: ℝ, 0 < x_0 ∧ x_0 ≤ Real.log x_0) :=
by sorry

end negation_of_prop_p_l717_717073


namespace fourth_row_sum_l717_717515

def a_n1 (n : ℕ) : ℕ := (n + 1)^3 - n

def sum_consecutive_odds_from (start n : ℕ) : ℕ :=
  (list.range n).map (λ i, start + 2 * i) |> list.sum

theorem fourth_row_sum :
  sum_consecutive_odds_from (a_n1 4) 5 = 54 :=
by
  sorry

end fourth_row_sum_l717_717515


namespace arithmetic_sequence_property_l717_717851

-- Define arithmetic sequence and given condition
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Lean 4 statement
theorem arithmetic_sequence_property {a : ℕ → ℝ} (h : arithmetic_sequence a) (h1 : a 6 = 30) : a 3 + a 9 = 60 :=
by
  sorry

end arithmetic_sequence_property_l717_717851


namespace sum_of_trinomials_eq_zero_l717_717398

-- Define the quadratic trinomials
def f (a b : ℝ) (c : ℕ → ℝ) (i : ℕ) (x : ℝ) := a * x^2 + b * x + c i

-- Define the roots
variable {a b : ℝ}
variable {c : ℕ → ℝ}
variable {roots : ℕ → ℝ}

-- Hypothesis: roots[i] is a root of the corresponding polynomial f_i(x)
axiom h (i : ℕ) : f a b c i (roots i) = 0

-- Prove the sum condition
theorem sum_of_trinomials_eq_zero :
  ∑ i in finset.range 100, f a b c (i % 100 + 1) (roots ((i - 1) % 100 + 1)) = 0 :=
by
  sorry

end sum_of_trinomials_eq_zero_l717_717398


namespace angle_C_is_pi_div_three_l717_717462

-- Definitions for the problem
variables {α : Type} [linear_ordered_field α] {a b c : α} {A B C : ℝ}

-- Assuming we have a proof that (a + c) * (sin A - sin C) = b * (sin A - sin B)
axiom equation : (a + c) * (Real.sin A - Real.sin C) = b * (Real.sin A - Real.sin B)

-- Prove that angle C is π / 3, given the conditions
theorem angle_C_is_pi_div_three (h : (a + c) * (Real.sin A - Real.sin C) = b * (Real.sin A - Real.sin B)) :
  C = π / 3 :=
by 
  sorry -- skip the proof

end angle_C_is_pi_div_three_l717_717462


namespace sequence_gcd_equality_l717_717230

theorem sequence_gcd_equality (a : ℕ → ℕ) 
  (h : ∀ (i j : ℕ), i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) : 
  ∀ i, a i = i := 
sorry

end sequence_gcd_equality_l717_717230


namespace range_of_m_l717_717219

noncomputable def f (x : ℝ) : ℝ := sin(x)^4 - cos(x)^4

theorem range_of_m :
  ∀ x : ℝ, 1 + (2 / 3) * f(x) - m * f(x / 2) ≥ 0 ↔ -1 / 3 ≤ m ∧ m ≤ 1 / 3 := 
sorry

end range_of_m_l717_717219


namespace find_y_coordinate_of_equidistant_point_l717_717984

theorem find_y_coordinate_of_equidistant_point :
  ∃ y : ℚ, (0, y) ∈ Line.axisY ∧ (dist (0, y) (0, 0) = dist (0, y) (2, 3)) ∧ y = 13/6 :=
by
  sorry

end find_y_coordinate_of_equidistant_point_l717_717984


namespace angle_BAC_60_degree_l717_717133

theorem angle_BAC_60_degree
  (A B C E F : Point)
  (Γ : Circle)
  (h1 : is_circumcircle Γ A B C)
  (h2 : is_intersection_bisectors_circumcircle E F (angle_bisector B A C) (angle_bisector C A B) Γ)
  (γ : Circle)
  (h3 : is_incicle γ A B C)
  (h4 : is_tangent γ E F) :
  angle A B C = 60 :=
sorry

end angle_BAC_60_degree_l717_717133


namespace incenter_coordinates_l717_717863

open Real
open EuclideanGeometry

/-- In triangle ABC with side lengths a = 7, b = 9, and c = 4, the incenter I can be expressed 
as a linear combination I = xA + yB + zC, where x + y + z = 1 -/
theorem incenter_coordinates (a b c : ℝ) (A B C I : Point)
  (ha : a = 7) (hb : b = 9) (hc : c = 4)
  (h_incenter : is_incenter I A B C) :
  ∃ x y z : ℝ, x + y + z = 1 ∧ I = x • A + y • B + z • C ∧ (x, y, z) = (7/20 : ℝ, 9/20 : ℝ, 1/5 : ℝ) :=
sorry

end incenter_coordinates_l717_717863


namespace angle_C_eq_pi_over_3_l717_717460

theorem angle_C_eq_pi_over_3 (a b c A B C : ℝ)
  (h : (a + c) * (Real.sin A - Real.sin C) = b * (Real.sin A - Real.sin B)) :
  C = Real.pi / 3 :=
sorry

end angle_C_eq_pi_over_3_l717_717460


namespace inverse_89_mod_90_l717_717363

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  -- Mathematical proof is skipped
  sorry

end inverse_89_mod_90_l717_717363


namespace op_correct_l717_717959

-- Definition of the operation * for non-zero integers
def op (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 / b)

theorem op_correct (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 12) (h2 : a * b = 32) :
  op a b = 3 / 8 :=
by
  -- Proof, sorry for now
  sorry

end op_correct_l717_717959


namespace f_odd_f_decreasing_f_extremum_l717_717148

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_val : f 1 = -2
axiom f_neg : ∀ x > 0, f x < 0

theorem f_odd : ∀ x : ℝ, f (-x) = -f x :=
sorry

theorem f_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ :=
sorry

theorem f_extremum : ∃ (max min : ℝ), max = f (-3) ∧ min = f 3 :=
sorry

end f_odd_f_decreasing_f_extremum_l717_717148


namespace polynomial_unique_form_l717_717367

theorem polynomial_unique_form (P : ℤ[X]) (exists_seq : ∃ (a : ℕ → ℤ), 
  (∀ i j, i ≠ j → a i ≠ a j) ∧ 
  (P.eval (a 1) = 0) ∧
  (∀ k, P.eval (a (k + 1)) = a k)) :
  ∃ C : ℤ, C ≠ 0 ∧ (∀ x : ℤ, P.eval x = x - C) :=
by sorry

end polynomial_unique_form_l717_717367


namespace bus_driver_overtime_pay_rate_increase_l717_717277

theorem bus_driver_overtime_pay_rate_increase 
  (r : ℝ) (h_t : ℝ) (h_r : ℝ) (c_t : ℝ) (p : ℝ)
  (h_reg_rate : r = 12)
  (h_total_hours : h_t = 63.62)
  (h_regular_hours : h_r = 40)
  (h_total_compensation : c_t = 976)
  (h_correct_answer : p = 75) : 
  [( ((c_t - h_r * r) / (h_t - h_r)) - r) / r * 100 = p] :=
by sorry

end bus_driver_overtime_pay_rate_increase_l717_717277


namespace find_circle_center_l717_717733

-- The statement to prove that the center of the given circle equation is (1, -2)
theorem find_circle_center : 
  ∃ (h k : ℝ), 3 * x^2 - 6 * x + 3 * y^2 + 12 * y - 75 = 0 → (h, k) = (1, -2) := 
by
  sorry

end find_circle_center_l717_717733


namespace problem_1_problem_2_problem_3_l717_717078

-- Problem I
def M := {1, 2, 3, 4, 5, 6}

def e_set (a b : ℕ) : ℚ := if (a < b) ∧ (a ∈ M) ∧ (b ∈ M) then (b : ℚ) / (a : ℚ) else 0

def unique_ratios : ℕ := (M.to_finset.pairs \ 
  (λ (a b : ℕ) _ _, if a < b then true else false)).img e_set |>.card

theorem problem_1 : unique_ratios = 11 :=
sorry

-- Problem II
def A : Finset ℚ := -- List of distinct ratios (already given as an example)
  {1/2, 1/3, 1/4, 1/5, 1/6, 2/3, 2/5, 3/4, 3/5, 4/5, 5/6}
def B : Finset ℚ := -- Reciprocals of elements in set A
  {2, 3, 4, 5, 6, 3/2, 5/2, 4/3, 5/3, 5/4, 6/5}

def sum_ei_ne_ej : ℚ := (A.product A).sum 
  (λ x, if x.1 ≠ x.2 then x.1 - B.product B).sum (λ x, x.2 else 0)

theorem problem_2 : sum_ei_ne_ej = 6039 / 40 :=
sorry

-- Problem III
def count_pairs : ℕ := (A.product B).count (λ x, (x.1 + x.2) ∈ ℤ)

def probability : ℚ := (count_pairs : ℚ) / (A.card * B.card)

theorem problem_3 : probability = 6 / 121 :=
sorry

end problem_1_problem_2_problem_3_l717_717078


namespace find_tan_alpha_minus_beta_l717_717406

variables (α β : ℝ)

theorem find_tan_alpha_minus_beta (h1 : real.sin α = 3/5) (h2 : α ∈ set.Ioo (real.pi / 2) real.pi)
  (h3 : real.tan (real.pi - β) = 1/2) :
  real.tan (α - β) = -2/11 :=
sorry

end find_tan_alpha_minus_beta_l717_717406


namespace integral_f_equals_neg_third_l717_717829

def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 2 * c

theorem integral_f_equals_neg_third :
  (∫ x in (0 : ℝ)..(1 : ℝ), f x (∫ t in (0 : ℝ)..(1 : ℝ), f t (∫ t in (0 : ℝ)..(1 : ℝ), f t 0))) = -1/3 :=
by
  sorry

end integral_f_equals_neg_third_l717_717829


namespace zero_is_monomial_l717_717609

theorem zero_is_monomial : is_monomial 0 := 
sorry

end zero_is_monomial_l717_717609


namespace tangent_lines_to_circle_passing_through_origin_l717_717003

/-- Prove that the equations x = 0 and 3x - 4y = 0 represent tangent lines to the circle
    (x-1)^2 + (y-2)^2 = 1, which pass through the origin (0,0). --/
theorem tangent_lines_to_circle_passing_through_origin :
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 1}
  ∧ let tangent1 := {p : ℝ × ℝ | p.1 = 0}
  ∧ let tangent2 := {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 = 0}
  in (∀ p ∈ tangent1, p ∈ circle → p = (0,0))
     ∧ (∀ p ∈ tangent2, p ∈ circle → p = (0,0))
     ∧ (∀ p ∈ circle, p.1 = 0 → p ∈ tangent1)
     ∧ (∀ p ∈ circle, 3 * p.1 - 4 * p.2 = 0 → p ∈ tangent2) :=
by
  sorry

end tangent_lines_to_circle_passing_through_origin_l717_717003


namespace soccer_team_points_l717_717194

theorem soccer_team_points 
  (total_games wins losses draws : ℕ)
  (points_per_win points_per_draw points_per_loss : ℕ)
  (h_total_games : total_games = 20)
  (h_wins : wins = 14)
  (h_losses : losses = 2)
  (h_draws : draws = total_games - (wins + losses))
  (h_points_per_win : points_per_win = 3)
  (h_points_per_draw : points_per_draw = 1)
  (h_points_per_loss : points_per_loss = 0) :
  (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) = 46 :=
by
  -- the actual proof steps will be inserted here
  sorry

end soccer_team_points_l717_717194


namespace number_of_green_beads_l717_717469

theorem number_of_green_beads (total_beads : ℕ) (red_fraction blue_fraction green_fraction : ℚ)
  (yellow_beads : ℕ)
  (h_red_fraction : red_fraction = 3 / 8)
  (h_blue_fraction : blue_fraction = 1 / 4)
  (h_green_fraction : green_fraction = 1 / 8)
  (h_yellow_fraction : 1 - (red_fraction + blue_fraction + green_fraction) = 1 / 4)
  (h_total_from_yellow : yellow_beads * 4 = total_beads) :
  ((green_fraction * total_beads).round : ℕ) = 23 :=
by
  sorry

end number_of_green_beads_l717_717469


namespace hasan_initial_plates_l717_717438

noncomputable def weight_plate := 10 -- weight of each plate in ounces
noncomputable def max_weight_pounds := 20 -- maximum weight in pounds
noncomputable def removed_plates := 6 -- number of plates removed
noncomputable def ounces_per_pound := 16 -- conversion factor from pounds to ounces

def max_weight_ounces : ℕ := max_weight_pounds * ounces_per_pound

-- Define the initial number of plates we need to prove
def initial_plates (weight_plate : ℕ) (max_weight_ounces : ℕ) (removed_plates : ℕ) : ℕ :=
  (max_weight_ounces + (removed_plates * weight_plate)) / weight_plate

theorem hasan_initial_plates :
  initial_plates weight_plate max_weight_ounces removed_plates = 38 :=
by
  -- solution steps skipped
  sorry

end hasan_initial_plates_l717_717438


namespace X_on_CPQ_circumcircle_l717_717504

-- Declare the points A, B, C, P, Q, R, X on the Euclidean plane
variables {A B C P Q R X : Type}

-- Assume A, B, C form a triangle
axiom triangle_ABC : A B C

-- Assume P is a point on line segment BC
axiom P_on_BC : ∃ (P : Type), P ∈ segment B C

-- Assume Q is a point on line segment CA
axiom Q_on_CA : ∃ (Q : Type), Q ∈ segment C A

-- Assume R is a point on line segment AB
axiom R_on_AB : ∃ (R : Type), R ∈ segment A B

-- Assume circumcircles of triangles AQR and BRP intersect at a second point X
axiom AQR_circum_circle : ∃ (X : Type), is_circumcircle A Q R X
axiom BRP_circum_circle : ∃ (X : Type), is_circumcircle B R P X

-- Proof that X lies on the circumcircle of triangle CPQ given the above conditions
theorem X_on_CPQ_circumcircle : ∃ (X : Type), is_circumcircle C P Q X := by
  sorry

end X_on_CPQ_circumcircle_l717_717504


namespace blake_initial_amount_l717_717326

noncomputable def initial_amount_given (amount: ℕ) := 3 * amount / 2

theorem blake_initial_amount (h_given_amount_b_to_c: ℕ) (c_to_b_transfer: ℕ) (h_c_to_b: c_to_b_transfer = 30000) :
  initial_amount_given c_to_b_transfer = h_given_amount_b_to_c → h_given_amount_b_to_c = 20000 :=
by
  intro h
  have calc_initial := calc 
    initial_amount_given 30000
      = 3 * 30000 / 2 : by rfl
      ... = 90000 / 2 : by norm_num
      ... = 45000 : by norm_num
  contradiction

end blake_initial_amount_l717_717326


namespace flowchart_output_l717_717172

def flowchart_program : ℕ × ℕ → ℕ × ℕ
| (i, s) :=
  if i < 5 then
    flowchart_program (i + 1, s - 1)
  else
    (i, s)

theorem flowchart_output :
  let initial_state := (1, 3)
  let result_state := flowchart_program initial_state
  in result_state.2 = 0 :=
by 
  sorry

end flowchart_output_l717_717172


namespace x_42_gt_x_43_l717_717884

def power_tower (a : ℝ) (n : ℕ) : ℝ
| 1       := a
| (i + 1) := a^(power_tower i)

noncomputable def x (k : ℕ) : ℝ :=
  classical.some (exists_unique_of_exists_of_unique (λ y : ℝ, power_tower y k = power_tower 10 (k + 1)))

theorem x_42_gt_x_43 : x 42 > x 43 :=
sorry

end x_42_gt_x_43_l717_717884
