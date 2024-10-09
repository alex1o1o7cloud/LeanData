import Mathlib

namespace tiffany_max_points_l2230_223078

theorem tiffany_max_points : 
  let initial_money := 3
  let cost_per_game := 1
  let points_red_bucket := 2
  let points_green_bucket := 3
  let rings_per_game := 5
  let games_played := 2
  let red_buckets_first_two_games := 4
  let green_buckets_first_two_games := 5
  let remaining_money := initial_money - games_played * cost_per_game
  let remaining_games := remaining_money / cost_per_game
  let points_first_two_games := red_buckets_first_two_games * points_red_bucket + green_buckets_first_two_games * points_green_bucket
  let max_points_third_game := rings_per_game * points_green_bucket
  points_first_two_games + max_points_third_game = 38 := 
by
  sorry

end tiffany_max_points_l2230_223078


namespace circle_radius_l2230_223063

theorem circle_radius (x y : ℝ) :
  y = (x - 2)^2 ∧ x - 3 = (y + 1)^2 →
  (∃ c d r : ℝ, (c, d) = (3/2, -1/2) ∧ r^2 = 25/4) :=
by
  sorry

end circle_radius_l2230_223063


namespace find_f_2008_l2230_223057

noncomputable def f : ℝ → ℝ := sorry

axiom f_zero : f 0 = 2008

axiom f_inequality1 : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2^x
axiom f_inequality2 : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2^x

theorem find_f_2008 : f 2008 = 2^2008 + 2007 :=
sorry

end find_f_2008_l2230_223057


namespace hyperbola_slope_reciprocals_l2230_223027

theorem hyperbola_slope_reciprocals (P : ℝ × ℝ) (t : ℝ) :
  (P.1 = t ∧ P.2 = - (8 / 9) * t ∧ t ≠ 0 ∧  
    ∃ k1 k2: ℝ, k1 = - (8 * t) / (9 * (t + 3)) ∧ k2 = - (8 * t) / (9 * (t - 3)) ∧
    (1 / k1) + (1 / k2) = -9 / 4) ∧
    ((P = (9/5, -(8/5)) ∨ P = (-(9/5), 8/5)) →
        ∃ kOA kOB kOC kOD : ℝ, (kOA + kOB + kOC + kOD = 0)) := 
sorry

end hyperbola_slope_reciprocals_l2230_223027


namespace distinct_x_intercepts_l2230_223048

theorem distinct_x_intercepts : 
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x, (x + 5) * (x^2 + 5 * x - 6) = 0 ↔ x ∈ s :=
by { 
  sorry 
}

end distinct_x_intercepts_l2230_223048


namespace monotonic_range_of_b_l2230_223039

noncomputable def f (b x : ℝ) : ℝ := x^3 - b * x^2 + 3 * x - 5

theorem monotonic_range_of_b (b : ℝ) : (∀ x y: ℝ, (f b x) ≤ (f b y) → x ≤ y) ↔ -3 ≤ b ∧ b ≤ 3 :=
sorry

end monotonic_range_of_b_l2230_223039


namespace binomial_510_510_l2230_223088

-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem binomial_510_510 : binomial 510 510 = 1 :=
  by
    -- Skip the proof with sorry
    sorry

end binomial_510_510_l2230_223088


namespace total_hours_worked_l2230_223031

theorem total_hours_worked (amber_hours : ℕ) (armand_hours : ℕ) (ella_hours : ℕ) 
(h_amber : amber_hours = 12) 
(h_armand : armand_hours = (1 / 3) * amber_hours) 
(h_ella : ella_hours = 2 * amber_hours) :
amber_hours + armand_hours + ella_hours = 40 :=
sorry

end total_hours_worked_l2230_223031


namespace each_persons_tip_l2230_223083

theorem each_persons_tip
  (cost_julie cost_letitia cost_anton : ℕ)
  (H1 : cost_julie = 10)
  (H2 : cost_letitia = 20)
  (H3 : cost_anton = 30)
  (total_people : ℕ)
  (H4 : total_people = 3)
  (tip_percentage : ℝ)
  (H5 : tip_percentage = 0.20) :
  ∃ tip_per_person : ℝ, tip_per_person = 4 := 
by
  sorry

end each_persons_tip_l2230_223083


namespace inequality_solution_l2230_223006

theorem inequality_solution (a b : ℝ)
  (h₁ : ∀ x, - (1 : ℝ) / 2 < x ∧ x < (1 : ℝ) / 3 → ax^2 + bx + (2 : ℝ) > 0)
  (h₂ : - (1 : ℝ) / 2 = -(b / a))
  (h₃ : (- (1 : ℝ) / 6) = 2 / a) :
  a - b = -10 :=
sorry

end inequality_solution_l2230_223006


namespace line_equation_l2230_223045

noncomputable def P (A B C x y : ℝ) := A * x + B * y + C

theorem line_equation {A B C x₁ y₁ x₂ y₂ : ℝ} (h1 : P A B C x₁ y₁ = 0) (h2 : P A B C x₂ y₂ ≠ 0) :
    ∀ (x y : ℝ), P A B C x y - P A B C x₁ y₁ - P A B C x₂ y₂ = 0 ↔ P A B 0 x y = -P A B 0 x₂ y₂ := by
  sorry

end line_equation_l2230_223045


namespace description_of_S_l2230_223056

noncomputable def S := {p : ℝ × ℝ | (3 = (p.1 + 2) ∧ p.2 - 5 ≤ 3) ∨ 
                                      (3 = (p.2 - 5) ∧ p.1 + 2 ≤ 3) ∨ 
                                      (p.1 + 2 = p.2 - 5 ∧ 3 ≤ p.1 + 2 ∧ 3 ≤ p.2 - 5)}

theorem description_of_S :
  S = {p : ℝ × ℝ | (p.1 = 1 ∧ p.2 ≤ 8) ∨ 
                    (p.2 = 8 ∧ p.1 ≤ 1) ∨ 
                    (p.2 = p.1 + 7 ∧ p.1 ≥ 1 ∧ p.2 ≥ 8)} :=
sorry

end description_of_S_l2230_223056


namespace average_of_data_set_is_five_l2230_223090

def data_set : List ℕ := [2, 5, 5, 6, 7]

def sum_of_data_set : ℕ := data_set.sum
def count_of_data_set : ℕ := data_set.length

theorem average_of_data_set_is_five :
  (sum_of_data_set / count_of_data_set) = 5 :=
by
  sorry

end average_of_data_set_is_five_l2230_223090


namespace intersection_M_N_l2230_223020

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 - 3 * x = 0}

theorem intersection_M_N : M ∩ N = {0} :=
by sorry

end intersection_M_N_l2230_223020


namespace solve_quadratic_equation_l2230_223092

theorem solve_quadratic_equation (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 :=
by
  sorry

end solve_quadratic_equation_l2230_223092


namespace field_area_proof_l2230_223069

-- Define the length of the uncovered side
def L : ℕ := 20

-- Define the total amount of fencing used for the other three sides
def total_fence : ℕ := 26

-- Define the field area function
def field_area (length width : ℕ) : ℕ := length * width

-- Statement: Prove that the area of the field is 60 square feet
theorem field_area_proof : 
  ∃ W : ℕ, (2 * W + L = total_fence) ∧ (field_area L W = 60) :=
  sorry

end field_area_proof_l2230_223069


namespace point_in_third_quadrant_l2230_223049

theorem point_in_third_quadrant (x y : ℤ) (hx : x = -8) (hy : y = -3) : (x < 0) ∧ (y < 0) :=
by
  have hx_neg : x < 0 := by rw [hx]; norm_num
  have hy_neg : y < 0 := by rw [hy]; norm_num
  exact ⟨hx_neg, hy_neg⟩

end point_in_third_quadrant_l2230_223049


namespace odd_function_f_l2230_223077

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

noncomputable def f (x: ℝ) := Real.log ((1 - x) / (1 + x))

theorem odd_function_f :
  odd_function f :=
sorry

end odd_function_f_l2230_223077


namespace students_not_enrolled_l2230_223058

theorem students_not_enrolled (total_students : ℕ) (students_french : ℕ) (students_german : ℕ) (students_both : ℕ)
  (h1 : total_students = 94)
  (h2 : students_french = 41)
  (h3 : students_german = 22)
  (h4 : students_both = 9) : 
  ∃ (students_neither : ℕ), students_neither = 40 :=
by
  -- We would show the calculation here in a real proof 
  sorry

end students_not_enrolled_l2230_223058


namespace geometric_sequence_thm_proof_l2230_223019

noncomputable def geometric_sequence_thm (a : ℕ → ℤ) : Prop :=
  (∃ r : ℤ, ∃ a₀ : ℤ, ∀ n : ℕ, a n = a₀ * r ^ n) ∧
  (a 2) * (a 10) = 4 ∧
  (a 2) + (a 10) > 0 →
  (a 6) = 2

theorem geometric_sequence_thm_proof (a : ℕ → ℤ) :
  geometric_sequence_thm a :=
  by
  sorry

end geometric_sequence_thm_proof_l2230_223019


namespace set_intersection_l2230_223070

open Set

/-- Given sets M and N as defined below, we wish to prove that their complements and intersections work as expected. -/
theorem set_intersection (R : Set ℝ)
  (M : Set ℝ := {x | x > 1})
  (N : Set ℝ := {x | abs x ≤ 2})
  (R_universal : R = univ) :
  ((compl M) ∩ N) = Icc (-2 : ℝ) (1 : ℝ) := by
  sorry

end set_intersection_l2230_223070


namespace dwarfs_truthful_count_l2230_223003

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l2230_223003


namespace carla_water_requirement_l2230_223032

theorem carla_water_requirement (h: ℕ) (p: ℕ) (c: ℕ) (gallons_per_pig: ℕ) (horse_factor: ℕ) 
  (num_pigs: ℕ) (num_horses: ℕ) (tank_water: ℕ): 
  num_pigs = 8 ∧ num_horses = 10 ∧ gallons_per_pig = 3 ∧ horse_factor = 2 ∧ tank_water = 30 →
  h = horse_factor * gallons_per_pig ∧ p = num_pigs * gallons_per_pig ∧ c = tank_water →
  h * num_horses + p + c = 114 :=
by
  intro h1 h2
  cases h1
  cases h2
  sorry

end carla_water_requirement_l2230_223032


namespace ratio_AB_CD_lengths_AB_CD_l2230_223079

theorem ratio_AB_CD 
  (AM MD BN NC : ℝ)
  (h_AM : AM = 25 / 7)
  (h_MD : MD = 10 / 7)
  (h_BN : BN = 3 / 2)
  (h_NC : NC = 9 / 2)
  : (AM / MD) / (BN / NC) = 5 / 6 :=
by
  sorry

theorem lengths_AB_CD
  (AM MD BN NC AB CD : ℝ)
  (h_AM : AM = 25 / 7)
  (h_MD : MD = 10 / 7)
  (h_BN : BN = 3 / 2)
  (h_NC : NC = 9 / 2)
  (AB_div_CD : (AM / MD) / (BN / NC) = 5 / 6)
  (h_touch : true)  -- A placeholder condition indicating circles touch each other
  : AB = 5 ∧ CD = 6 :=
by
  sorry

end ratio_AB_CD_lengths_AB_CD_l2230_223079


namespace intersection_is_singleton_l2230_223050

def M : Set ℕ := {1, 2}
def N : Set ℕ := {n | ∃ a ∈ M, n = 2 * a - 1}

theorem intersection_is_singleton : M ∩ N = {1} :=
by sorry

end intersection_is_singleton_l2230_223050


namespace horner_v2_value_l2230_223081

def polynomial : ℤ → ℤ := fun x => 208 + 9 * x^2 + 6 * x^4 + x^6

def horner (x : ℤ) : ℤ :=
  let v0 := 1
  let v1 := v0 * x
  let v2 := v1 * x + 6
  v2

theorem horner_v2_value (x : ℤ) : x = -4 → horner x = 22 :=
by
  intro h
  rw [h]
  rfl

end horner_v2_value_l2230_223081


namespace units_digit_27_64_l2230_223010

/-- 
  Given that the units digit of 27 is 7, 
  and the units digit of 64 is 4, 
  prove that the units digit of 27 * 64 is 8.
-/
theorem units_digit_27_64 : 
  ∀ (n m : ℕ), 
  (n % 10 = 7) → 
  (m % 10 = 4) → 
  ((n * m) % 10 = 8) :=
by
  intros n m h1 h2
  -- Utilize modular arithmetic properties
  sorry

end units_digit_27_64_l2230_223010


namespace ratio_of_triangle_areas_l2230_223007

theorem ratio_of_triangle_areas (kx ky k : ℝ)
(n m : ℕ) (h1 : n > 0) (h2 : m > 0) :
  let A := (1 / 2) * (ky / m) * (kx / 2)
  let B := (1 / 2) * (kx / n) * (ky / 2)
  (A / B) = (n / m) :=
by
  sorry

end ratio_of_triangle_areas_l2230_223007


namespace double_windows_downstairs_eq_twelve_l2230_223016

theorem double_windows_downstairs_eq_twelve
  (D : ℕ)
  (H1 : ∀ d, d = D → 4 * d + 32 = 80) :
  D = 12 :=
by
  sorry

end double_windows_downstairs_eq_twelve_l2230_223016


namespace sum_of_irreducible_fractions_is_integer_iff_same_denominator_l2230_223040

theorem sum_of_irreducible_fractions_is_integer_iff_same_denominator
  (a b c d A : ℤ) (h_irred1 : Int.gcd a b = 1) (h_irred2 : Int.gcd c d = 1) (h_sum : (a : ℚ) / b + (c : ℚ) / d = A) :
  b = d := 
by
  sorry

end sum_of_irreducible_fractions_is_integer_iff_same_denominator_l2230_223040


namespace length_reduction_percentage_to_maintain_area_l2230_223052

theorem length_reduction_percentage_to_maintain_area
  (L W : ℝ)
  (new_width : ℝ := W * (1 + 28.2051282051282 / 100))
  (new_length : ℝ := L * (1 - 21.9512195121951 / 100))
  (original_area : ℝ := L * W) :
  original_area = new_length * new_width := by
  sorry

end length_reduction_percentage_to_maintain_area_l2230_223052


namespace sin_alpha_eq_63_over_65_l2230_223023

open Real

variables {α β : ℝ}

theorem sin_alpha_eq_63_over_65
  (h1 : tan β = 4 / 3)
  (h2 : sin (α + β) = 5 / 13)
  (h3 : 0 < α ∧ α < π)
  (h4 : 0 < β ∧ β < π) :
  sin α = 63 / 65 := 
by
  sorry

end sin_alpha_eq_63_over_65_l2230_223023


namespace seniors_selected_correct_l2230_223085

-- Definitions based on the conditions problem
def total_freshmen : ℕ := 210
def total_sophomores : ℕ := 270
def total_seniors : ℕ := 300
def selected_freshmen : ℕ := 7

-- Problem statement to prove
theorem seniors_selected_correct : 
  (total_seniors / (total_freshmen / selected_freshmen)) = 10 := 
by 
  sorry

end seniors_selected_correct_l2230_223085


namespace labourer_monthly_income_l2230_223037

-- Define the conditions
def total_expense_first_6_months : ℕ := 90 * 6
def total_expense_next_4_months : ℕ := 60 * 4
def debt_cleared_and_savings : ℕ := 30

-- Define the monthly income
def monthly_income : ℕ := 81

-- The statement to be proven
theorem labourer_monthly_income (I D : ℕ) (h1 : 6 * I + D = total_expense_first_6_months) 
                               (h2 : 4 * I - D = total_expense_next_4_months + debt_cleared_and_savings) :
  I = monthly_income :=
by {
  sorry
}

end labourer_monthly_income_l2230_223037


namespace range_of_m_for_subset_l2230_223014

open Set

variable (m : ℝ)

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | (2 * m - 1) ≤ x ∧ x ≤ (m + 1)}

theorem range_of_m_for_subset (m : ℝ) : B m ⊆ A ↔ m ∈ Icc (-(1 / 2) : ℝ) (2 : ℝ) ∨ m > (2 : ℝ) :=
by
  sorry

end range_of_m_for_subset_l2230_223014


namespace expected_ties_after_10_l2230_223044

def binom: ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binom n k + binom n (k+1)

noncomputable def expected_ties : ℕ → ℝ 
| 0 => 0
| n+1 => expected_ties n + (binom (2*(n+1)) (n+1) / 2^(2*(n+1)))

theorem expected_ties_after_10 : expected_ties 5 = 1.707 := 
by 
  -- Placeholder for the actual proof
  sorry

end expected_ties_after_10_l2230_223044


namespace find_k_l2230_223009

variables (a b : ℝ × ℝ)
variables (k : ℝ)

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (2, -1)

def k_a_plus_b (k : ℝ) : ℝ × ℝ := (k * vector_a.1 + vector_b.1, k * vector_a.2 + vector_b.2)
def a_minus_2b : ℝ × ℝ := (vector_a.1 - 2 * vector_b.1, vector_a.2 - 2 * vector_b.2)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_k (k : ℝ) : dot_product (k_a_plus_b k) a_minus_2b = 0 ↔ k = 2 :=
by
  sorry

end find_k_l2230_223009


namespace scholarship_total_l2230_223053

-- Definitions of the money received by Wendy, Kelly, Nina, and Jason based on the given conditions
def wendy_scholarship : ℕ := 20000
def kelly_scholarship : ℕ := 2 * wendy_scholarship
def nina_scholarship : ℕ := kelly_scholarship - 8000
def jason_scholarship : ℕ := (3 * kelly_scholarship) / 4

-- Total amount of scholarships
def total_scholarship : ℕ := wendy_scholarship + kelly_scholarship + nina_scholarship + jason_scholarship

-- The proof statement that needs to be proven
theorem scholarship_total : total_scholarship = 122000 := by
  -- Here we use 'sorry' to indicate that the proof is not provided.
  sorry

end scholarship_total_l2230_223053


namespace rhombus_side_length_l2230_223028

theorem rhombus_side_length (area d1 d2 side : ℝ) (h_area : area = 24)
(h_d1 : d1 = 6) (h_other_diag : d2 * 6 = 48) (h_side : side = Real.sqrt (3^2 + 4^2)) :
  side = 5 :=
by
  -- This is where the proof would go
  sorry

end rhombus_side_length_l2230_223028


namespace translation_of_point_l2230_223068

variable (P : ℝ × ℝ) (xT yT : ℝ)

def translate_x (P : ℝ × ℝ) (xT : ℝ) : ℝ × ℝ :=
    (P.1 + xT, P.2)

def translate_y (P : ℝ × ℝ) (yT : ℝ) : ℝ × ℝ :=
    (P.1, P.2 + yT)

theorem translation_of_point : translate_y (translate_x (-5, 1) 2) (-4) = (-3, -3) :=
by
  sorry

end translation_of_point_l2230_223068


namespace man_salary_l2230_223055

variable (S : ℝ)

theorem man_salary (S : ℝ) (h1 : S - (1/3) * S - (1/4) * S - (1/5) * S = 1760) : S = 8123 := 
by 
  sorry

end man_salary_l2230_223055


namespace base6_sum_eq_10_l2230_223051

theorem base6_sum_eq_10 
  (A B C : ℕ) 
  (hA : 0 < A ∧ A < 6) 
  (hB : 0 < B ∧ B < 6) 
  (hC : 0 < C ∧ C < 6)
  (distinct : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (h_add : A*36 + B*6 + C + B*6 + C = A*36 + C*6 + A) :
  A + B + C = 10 := 
by
  sorry

end base6_sum_eq_10_l2230_223051


namespace find_a_l2230_223094

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) 
  (h : (6 * a * (-1) + 6) = 4) : 
  a = 10 / 3 :=
by {
  sorry
}

end find_a_l2230_223094


namespace simplify_polynomial_sum_l2230_223008

/- Define the given polynomials -/
def polynomial1 (x : ℝ) : ℝ := (5 * x^10 + 8 * x^9 + 3 * x^8)
def polynomial2 (x : ℝ) : ℝ := (2 * x^12 + 3 * x^10 + x^9 + 4 * x^8 + 6 * x^4 + 7 * x^2 + 9)
def resultant_polynomial (x : ℝ) : ℝ := (2 * x^12 + 8 * x^10 + 9 * x^9 + 7 * x^8 + 6 * x^4 + 7 * x^2 + 9)

theorem simplify_polynomial_sum (x : ℝ) :
  polynomial1 x + polynomial2 x = resultant_polynomial x :=
by
  sorry

end simplify_polynomial_sum_l2230_223008


namespace combined_points_correct_l2230_223096

-- Definitions for the points scored by each player
def points_Lemuel := 7 * 2 + 5 * 3 + 4
def points_Marcus := 4 * 2 + 6 * 3 + 7
def points_Kevin := 9 * 2 + 4 * 3 + 5
def points_Olivia := 6 * 2 + 3 * 3 + 6

-- Definition for the combined points scored by both teams
def combined_points := points_Lemuel + points_Marcus + points_Kevin + points_Olivia

-- Theorem statement to prove combined points equals 128
theorem combined_points_correct : combined_points = 128 :=
by
  -- Lean proof goes here
  sorry

end combined_points_correct_l2230_223096


namespace find_a_l2230_223064

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 0 then 1 - x else a * x

theorem find_a (a : ℝ) : f (-1) a = f 1 a → a = 2 := by
  intro h
  sorry

end find_a_l2230_223064


namespace d_is_greatest_l2230_223017

variable (p : ℝ)

def a := p - 1
def b := p + 2
def c := p - 3
def d := p + 4

theorem d_is_greatest : d > b ∧ d > a ∧ d > c := 
by sorry

end d_is_greatest_l2230_223017


namespace positive_difference_perimeters_l2230_223001

theorem positive_difference_perimeters (length width : ℝ) 
    (cut_rectangles : ℕ) 
    (H : length = 6 ∧ width = 9 ∧ cut_rectangles = 4) : 
    ∃ (p1 p2 : ℝ), (p1 = 24 ∧ p2 = 15) ∧ (abs (p1 - p2) = 9) :=
by
  sorry

end positive_difference_perimeters_l2230_223001


namespace sophia_ate_pie_l2230_223076

theorem sophia_ate_pie (weight_fridge weight_total weight_ate : ℕ)
  (h1 : weight_fridge = 1200) 
  (h2 : weight_fridge = 5 * weight_total / 6) :
  weight_ate = weight_total / 6 :=
by
  have weight_total_formula : weight_total = 6 * weight_fridge / 5 := by
    sorry
  have weight_ate_formula : weight_ate = weight_total / 6 := by
    sorry
  sorry

end sophia_ate_pie_l2230_223076


namespace range_of_f_l2230_223099

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem range_of_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  ∃ y, y = f x ∧ (y ≥ -3 / 2 ∧ y ≤ 3) :=
by {
  sorry
}

end range_of_f_l2230_223099


namespace total_vowels_written_l2230_223011

-- Define the vowels and the condition
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def num_vowels : Nat := vowels.length
def times_written : Nat := 2

-- Assert the total number of vowels written
theorem total_vowels_written : (num_vowels * times_written) = 10 := by
  sorry

end total_vowels_written_l2230_223011


namespace medication_price_reduction_l2230_223087

variable (a : ℝ)

theorem medication_price_reduction (h : 0.60 * x = a) : x = 5/3 * a := by
  sorry

end medication_price_reduction_l2230_223087


namespace sum_of_decimals_l2230_223054

theorem sum_of_decimals : (0.305 : ℝ) + (0.089 : ℝ) + (0.007 : ℝ) = 0.401 := by
  sorry

end sum_of_decimals_l2230_223054


namespace max_number_of_circular_triples_l2230_223029

theorem max_number_of_circular_triples (players : Finset ℕ) (game_results : ℕ → ℕ → Prop) (total_players : players.card = 14)
  (each_plays_13_others : ∀ (p : ℕ) (hp : p ∈ players), ∃ wins losses : Finset ℕ, wins.card = 6 ∧ losses.card = 7 ∧
    (∀ w ∈ wins, game_results p w) ∧ (∀ l ∈ losses, game_results l p)) :
  (∃ (circular_triples : Finset (Finset ℕ)), circular_triples.card = 112 ∧
    ∀ t ∈ circular_triples, t.card = 3 ∧
    (∀ x y z : ℕ, x ∈ t ∧ y ∈ t ∧ z ∈ t → game_results x y ∧ game_results y z ∧ game_results z x)) := 
sorry

end max_number_of_circular_triples_l2230_223029


namespace parabola_vertex_l2230_223002

theorem parabola_vertex (x y : ℝ) : 
  (∀ x y, y^2 - 8*y + 4*x = 12 → (x, y) = (7, 4)) :=
by
  intros x y h
  sorry

end parabola_vertex_l2230_223002


namespace value_of_g_800_l2230_223025

noncomputable def g : ℝ → ℝ :=
sorry

theorem value_of_g_800 (g_eq : ∀ (x y : ℝ) (hx : 0 < x) (hy : 0 < y), g (x * y) = g x / (y^2))
  (g_at_1000 : g 1000 = 4) : g 800 = 625 / 2 :=
sorry

end value_of_g_800_l2230_223025


namespace digit_in_thousandths_place_l2230_223065

theorem digit_in_thousandths_place : (3 / 16 : ℚ) = 0.1875 :=
by sorry

end digit_in_thousandths_place_l2230_223065


namespace cost_for_3300_pens_l2230_223047

noncomputable def cost_per_pack (pack_cost : ℝ) (num_pens_per_pack : ℕ) : ℝ :=
  pack_cost / num_pens_per_pack

noncomputable def total_cost (cost_per_pen : ℝ) (num_pens : ℕ) : ℝ :=
  cost_per_pen * num_pens

theorem cost_for_3300_pens (pack_cost : ℝ) (num_pens_per_pack num_pens : ℕ) (h_pack_cost : pack_cost = 45) (h_num_pens_per_pack : num_pens_per_pack = 150) (h_num_pens : num_pens = 3300) :
  total_cost (cost_per_pack pack_cost num_pens_per_pack) num_pens = 990 :=
  by
    sorry

end cost_for_3300_pens_l2230_223047


namespace exponent_sum_l2230_223061

theorem exponent_sum (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^123 + i^223 + i^323 = -3 * i :=
by
  sorry

end exponent_sum_l2230_223061


namespace rainfall_march_l2230_223080

variable (M A : ℝ)
variable (Hm : A = M - 0.35)
variable (Ha : A = 0.46)

theorem rainfall_march : M = 0.81 := by
  sorry

end rainfall_march_l2230_223080


namespace proof_problem_l2230_223036

noncomputable def arithmetic_sequence_sum (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  n * (a 1) + ((n * (n - 1)) / 2) * (a 2 - a 1)

theorem proof_problem
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (d : ℕ)
  (h_d_gt_zero : d > 0)
  (h_a1 : a 1 = 1)
  (h_S : ∀ n, S n = arithmetic_sequence_sum n a)
  (h_S2_S3 : S 2 * S 3 = 36)
  (h_arith_seq : ∀ n, a (n + 1) = a 1 + n * d)
  (m k : ℕ)
  (h_mk_pos : m > 0 ∧ k > 0)
  (sum_condition : (k + 1) * (a m + a (m + k)) / 2 = 65) :
  d = 2 ∧ (∀ n, S n = n * n) ∧ m = 5 ∧ k = 4 :=
by 
  sorry

end proof_problem_l2230_223036


namespace money_put_in_by_A_l2230_223059

theorem money_put_in_by_A 
  (B_capital : ℕ := 25000)
  (total_profit : ℕ := 9600)
  (A_management_fee : ℕ := 10)
  (A_total_received : ℕ := 4200) 
  (A_puts_in : ℕ) :
  (A_management_fee * total_profit / 100 
    + (A_puts_in / (A_puts_in + B_capital)) * (total_profit - A_management_fee * total_profit / 100) = A_total_received)
  → A_puts_in = 15000 :=
  by
    sorry

end money_put_in_by_A_l2230_223059


namespace mark_siblings_l2230_223021

theorem mark_siblings (total_eggs : ℕ) (eggs_per_person : ℕ) (persons_including_mark : ℕ) (h1 : total_eggs = 24) (h2 : eggs_per_person = 6) (h3 : persons_including_mark = total_eggs / eggs_per_person) : persons_including_mark - 1 = 3 :=
by 
  sorry

end mark_siblings_l2230_223021


namespace canonical_equations_of_line_l2230_223012

/-- Given two planes: 
  Plane 1: 4 * x + y + z + 2 = 0
  Plane 2: 2 * x - y - 3 * z - 8 = 0
  Prove that the canonical equations of the line formed by their intersection are:
  (x - 1) / -2 = (y + 6) / 14 = z / -6 -/
theorem canonical_equations_of_line :
  (∃ x y z : ℝ, 4 * x + y + z + 2 = 0 ∧ 2 * x - y - 3 * z - 8 = 0) →
  (∀ x y z : ℝ, ((x - 1) / -2 = (y + 6) / 14) ∧ ((y + 6) / 14 = z / -6)) :=
by
  sorry

end canonical_equations_of_line_l2230_223012


namespace min_value_of_a_l2230_223097

variables (a b c d : ℕ)

-- Conditions
def conditions : Prop :=
  a > b ∧ b > c ∧ c > d ∧
  a + b + c + d = 2004 ∧
  a^2 - b^2 + c^2 - d^2 = 2004

-- Theorem: minimum value of a
theorem min_value_of_a (h : conditions a b c d) : a = 503 :=
sorry

end min_value_of_a_l2230_223097


namespace alternating_colors_probability_l2230_223013

theorem alternating_colors_probability :
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let successful_outcomes : ℕ := 2
  let total_outcomes : ℕ := Nat.choose total_balls white_balls
  (successful_outcomes : ℚ) / (total_outcomes : ℚ) = (1 / 126) := 
by
  let total_balls := 10
  let white_balls := 5
  let black_balls := 5
  let successful_outcomes := 2
  let total_outcomes := Nat.choose total_balls white_balls
  have h_total_outcomes : total_outcomes = 252 := sorry
  have h_probability : (successful_outcomes : ℚ) / (total_outcomes : ℚ) = (1 / 126) := sorry
  exact h_probability

end alternating_colors_probability_l2230_223013


namespace planning_committee_ways_is_20_l2230_223066

-- Define the number of students in the council
def num_students : ℕ := 6

-- Define the ways to choose a 3-person committee from num_students
def committee_ways (x : ℕ) : ℕ := Nat.choose x 3

-- Given condition: number of ways to choose the welcoming committee is 20
axiom welcoming_committee_condition : committee_ways num_students = 20

-- Statement to prove
theorem planning_committee_ways_is_20 : committee_ways num_students = 20 := by
  exact welcoming_committee_condition

end planning_committee_ways_is_20_l2230_223066


namespace physics_teacher_min_count_l2230_223075

theorem physics_teacher_min_count 
  (maths_teachers : ℕ) 
  (chemistry_teachers : ℕ) 
  (max_subjects_per_teacher : ℕ) 
  (min_total_teachers : ℕ) 
  (physics_teachers : ℕ)
  (h1 : maths_teachers = 7)
  (h2 : chemistry_teachers = 5)
  (h3 : max_subjects_per_teacher = 3)
  (h4 : min_total_teachers = 6) 
  (h5 : 7 + physics_teachers + 5 ≤ 6 * 3) :
  0 < physics_teachers :=
  by 
  sorry

end physics_teacher_min_count_l2230_223075


namespace power_multiplication_l2230_223073

variable (x y m n : ℝ)

-- Establishing our initial conditions
axiom h1 : 10^x = m
axiom h2 : 10^y = n

theorem power_multiplication : 10^(2*x + 3*y) = m^2 * n^3 :=
by
  sorry

end power_multiplication_l2230_223073


namespace joe_spent_on_food_l2230_223033

theorem joe_spent_on_food :
  ∀ (initial_savings flight hotel remaining food : ℝ),
    initial_savings = 6000 →
    flight = 1200 →
    hotel = 800 →
    remaining = 1000 →
    food = initial_savings - remaining - (flight + hotel) →
    food = 3000 :=
by
  intros initial_savings flight hotel remaining food h₁ h₂ h₃ h₄ h₅
  sorry

end joe_spent_on_food_l2230_223033


namespace second_class_students_l2230_223018

-- Define the conditions
variables (x : ℕ)
variable (sum_marks_first_class : ℕ := 35 * 40)
variable (sum_marks_second_class : ℕ := x * 60)
variable (total_students : ℕ := 35 + x)
variable (total_marks_all_students : ℕ := total_students * 5125 / 100)

-- The theorem to prove
theorem second_class_students : 
  1400 + (x * 60) = (35 + x) * 5125 / 100 →
  x = 45 :=
by
  sorry

end second_class_students_l2230_223018


namespace greatest_value_of_squares_l2230_223084

-- Given conditions
variables (a b c d : ℝ)
variables (h1 : a + b = 20)
variables (h2 : ab + c + d = 105)
variables (h3 : ad + bc = 225)
variables (h4 : cd = 144)

theorem greatest_value_of_squares : a^2 + b^2 + c^2 + d^2 ≤ 150 := by
  sorry

end greatest_value_of_squares_l2230_223084


namespace unique_z_value_l2230_223038

theorem unique_z_value (x y u z : ℕ) (hx : 0 < x)
    (hy : 0 < y) (hu : 0 < u) (hz : 0 < z)
    (h1 : 3 + x + 21 = y + 25 + z)
    (h2 : 3 + x + 21 = 15 + u + 4)
    (h3 : y + 25 + z = 15 + u + 4)
    (h4 : 3 + y + 15 = x + 25 + u)
    (h5 : 3 + y + 15 = 21 + z + 4)
    (h6 : x + 25 + u = 21 + z + 4):
    z = 20 :=
by
    sorry

end unique_z_value_l2230_223038


namespace q_minus_r_max_value_l2230_223093

theorem q_minus_r_max_value :
  ∃ (q r : ℕ), q > 99 ∧ q < 1000 ∧ r > 99 ∧ r < 1000 ∧ 
    q = 100 * (q / 100) + 10 * ((q / 10) % 10) + (q % 10) ∧ 
    r = 100 * (q % 10) + 10 * ((q / 10) % 10) + (q / 100) ∧ 
    q - r = 297 :=
by sorry

end q_minus_r_max_value_l2230_223093


namespace find_shortage_l2230_223074

def total_capacity (T : ℝ) : Prop :=
  0.70 * T = 14

def normal_level (normal : ℝ) : Prop :=
  normal = 14 / 2

def capacity_shortage (T : ℝ) (normal : ℝ) : Prop :=
  T - normal = 13

theorem find_shortage (T : ℝ) (normal : ℝ) : 
  total_capacity T →
  normal_level normal →
  capacity_shortage T normal :=
by
  sorry

end find_shortage_l2230_223074


namespace raft_travel_distance_l2230_223067

theorem raft_travel_distance (v_b v_s t : ℝ) (h1 : t > 0) 
  (h2 : v_b + v_s = 90 / t) (h3 : v_b - v_s = 70 / t) : 
  v_s * t = 10 := by
  sorry

end raft_travel_distance_l2230_223067


namespace average_income_PQ_l2230_223089

/-
Conditions:
1. The average monthly income of Q and R is Rs. 5250.
2. The average monthly income of P and R is Rs. 6200.
3. The monthly income of P is Rs. 3000.
-/

def avg_income_QR := 5250
def avg_income_PR := 6200
def income_P := 3000

theorem average_income_PQ :
  ∃ (Q R : ℕ), ((Q + R) / 2 = avg_income_QR) ∧ ((income_P + R) / 2 = avg_income_PR) ∧ 
               (∀ (p q : ℕ), p = income_P → q = (Q + income_P) / 2 → q = 2050) :=
by
  sorry

end average_income_PQ_l2230_223089


namespace min_value_of_sum_of_squares_l2230_223046

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : 2 * x + 3 * y + 4 * z = 10) : 
  x^2 + y^2 + z^2 ≥ 100 / 29 :=
sorry

end min_value_of_sum_of_squares_l2230_223046


namespace modulus_z_eq_sqrt_10_l2230_223043

noncomputable def z : ℂ := (1 + 7 * Complex.I) / (2 + Complex.I)

theorem modulus_z_eq_sqrt_10 : Complex.abs z = Real.sqrt 10 := sorry

end modulus_z_eq_sqrt_10_l2230_223043


namespace probability_one_hits_l2230_223071

theorem probability_one_hits 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 1 / 2) (hB : p_B = 1 / 3):
  p_A * (1 - p_B) + (1 - p_A) * p_B = 1 / 2 := by
  sorry

end probability_one_hits_l2230_223071


namespace how_many_times_l2230_223062

theorem how_many_times (a b : ℝ) (h1 : a = 0.5) (h2 : b = 0.01) : a / b = 50 := 
by 
  sorry

end how_many_times_l2230_223062


namespace polynomial_g_correct_l2230_223041

noncomputable def polynomial_g : Polynomial ℚ := 
  Polynomial.C (-41 / 2) + Polynomial.X * 41 / 2 + Polynomial.X ^ 2

theorem polynomial_g_correct
  (f g : Polynomial ℚ)
  (h1 : f ≠ 0)
  (h2 : g ≠ 0)
  (hx : ∀ x, f.eval (g.eval x) = (Polynomial.eval x f) * (Polynomial.eval x g))
  (h3 : Polynomial.eval 3 g = 50) :
  g = polynomial_g :=
sorry

end polynomial_g_correct_l2230_223041


namespace wolf_nobel_laureates_l2230_223005

/-- 31 scientists that attended a certain workshop were Wolf Prize laureates,
and some of them were also Nobel Prize laureates. Of the scientists who attended
that workshop and had not received the Wolf Prize, the number of scientists who had
received the Nobel Prize was 3 more than the number of scientists who had not received
the Nobel Prize. In total, 50 scientists attended that workshop, and 25 of them were
Nobel Prize laureates. Prove that the number of Wolf Prize laureates who were also
Nobel Prize laureates is 3. -/
theorem wolf_nobel_laureates (W N total W' N' W_N : ℕ)  
  (hW : W = 31) (hN : N = 25) (htotal : total = 50) 
  (hW' : W' = total - W) (hN' : N' = total - N) 
  (hcondition : N' - W' = 3) :
  W_N = N - W' :=
by
  sorry

end wolf_nobel_laureates_l2230_223005


namespace cost_of_large_fries_l2230_223086

noncomputable def cost_of_cheeseburger : ℝ := 3.65
noncomputable def cost_of_milkshake : ℝ := 2
noncomputable def cost_of_coke : ℝ := 1
noncomputable def cost_of_cookie : ℝ := 0.5
noncomputable def tax : ℝ := 0.2
noncomputable def toby_initial_amount : ℝ := 15
noncomputable def toby_remaining_amount : ℝ := 7
noncomputable def split_bill : ℝ := 2

theorem cost_of_large_fries : 
  let total_meal_cost := (split_bill * (toby_initial_amount - toby_remaining_amount))
  let total_cost_so_far := (2 * cost_of_cheeseburger) + cost_of_milkshake + cost_of_coke + (3 * cost_of_cookie) + tax
  total_meal_cost - total_cost_so_far = 4 := 
by
  sorry

end cost_of_large_fries_l2230_223086


namespace seed_mixture_percentage_l2230_223082

theorem seed_mixture_percentage (x y : ℝ) 
  (hx : 0.4 * x + 0.25 * y = 30)
  (hxy : x + y = 100) :
  x / 100 = 0.3333 :=
by 
  sorry

end seed_mixture_percentage_l2230_223082


namespace discount_is_one_percent_l2230_223072

/-
  Assuming the following:
  - market_price is the price of one pen in dollars.
  - num_pens is the number of pens bought.
  - cost_price is the total cost price paid by the retailer.
  - profit_percentage is the profit made by the retailer.
  We need to prove that the discount percentage is 1.
-/

noncomputable def discount_percentage
  (market_price : ℝ)
  (num_pens : ℕ)
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (SP_per_pen : ℝ) : ℝ :=
  ((market_price - SP_per_pen) / market_price) * 100

theorem discount_is_one_percent
  (market_price : ℝ)
  (num_pens : ℕ)
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (buying_condition : cost_price = (market_price * num_pens * (36 / 60)))
  (SP : ℝ)
  (selling_condition : SP = cost_price * (1 + profit_percentage / 100))
  (SP_per_pen : ℝ)
  (sp_per_pen_condition : SP_per_pen = SP / num_pens)
  (profit_condition : profit_percentage = 65) :
  discount_percentage market_price num_pens cost_price profit_percentage SP_per_pen = 1 := by
  sorry

end discount_is_one_percent_l2230_223072


namespace min_value_of_w_l2230_223015

noncomputable def w (x y : ℝ) : ℝ := 2 * x^2 + 3 * y^2 + 8 * x - 6 * y + 30

theorem min_value_of_w : ∃ x y : ℝ, ∀ (a b : ℝ), w x y ≤ w a b ∧ w x y = 19 :=
by
  sorry

end min_value_of_w_l2230_223015


namespace range_of_m_l2230_223024

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, 3 * x^2 + 1 ≥ m * x * (x - 1)) : -6 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l2230_223024


namespace total_flowers_is_288_l2230_223030

-- Definitions from the Conditions in a)
def arwen_tulips : ℕ := 20
def arwen_roses : ℕ := 18
def elrond_tulips : ℕ := 2 * arwen_tulips
def elrond_roses : ℕ := 3 * arwen_roses
def galadriel_tulips : ℕ := 3 * elrond_tulips
def galadriel_roses : ℕ := 2 * arwen_roses

-- Total number of tulips
def total_tulips : ℕ := arwen_tulips + elrond_tulips + galadriel_tulips

-- Total number of roses
def total_roses : ℕ := arwen_roses + elrond_roses + galadriel_roses

-- Total number of flowers
def total_flowers : ℕ := total_tulips + total_roses

theorem total_flowers_is_288 : total_flowers = 288 :=
by
  -- Placeholder for proof
  sorry

end total_flowers_is_288_l2230_223030


namespace canoes_more_than_kayaks_l2230_223026

theorem canoes_more_than_kayaks (C K : ℕ)
  (h1 : 14 * C + 15 * K = 288)
  (h2 : C = 3 * K / 2) :
  C - K = 4 :=
sorry

end canoes_more_than_kayaks_l2230_223026


namespace incorrect_option_C_l2230_223000

-- Definitions of increasing and decreasing functions
def increasing (f : ℝ → ℝ) := ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≤ f x₂
def decreasing (f : ℝ → ℝ) := ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≥ f x₂

-- The incorrectness of option C
theorem incorrect_option_C (f g : ℝ → ℝ) 
  (h₁ : increasing f) 
  (h₂ : decreasing g) : ¬ increasing (fun x => f x + g x) := 
sorry

end incorrect_option_C_l2230_223000


namespace base_5_conversion_correct_l2230_223098

def base_5_to_base_10 : ℕ := 2 * 5^2 + 4 * 5^1 + 2 * 5^0

theorem base_5_conversion_correct : base_5_to_base_10 = 72 :=
by {
  -- Proof (not required in the problem statement)
  sorry
}

end base_5_conversion_correct_l2230_223098


namespace remainder_T_2015_mod_10_l2230_223060

-- Define the number of sequences with no more than two consecutive identical letters
noncomputable def T : ℕ → ℕ
| 0 => 0
| 1 => 2
| 2 => 4
| 3 => 6
| n + 1 => (T n + T (n - 1) + T (n - 2) + T (n - 3))  -- hypothetically following initial conditions pattern

theorem remainder_T_2015_mod_10 : T 2015 % 10 = 6 :=
by 
  sorry

end remainder_T_2015_mod_10_l2230_223060


namespace solve_for_a_l2230_223035

theorem solve_for_a (a x y : ℝ) (h1 : x = 1) (h2 : y = -2) (h3 : a * x + y = 3) : a = 5 :=
by
  sorry

end solve_for_a_l2230_223035


namespace solve_for_x_l2230_223034

variable (x : ℝ)

theorem solve_for_x (h : (4 * x + 2) / (5 * x - 5) = 3 / 4) : x = -23 := 
by
  sorry

end solve_for_x_l2230_223034


namespace rain_on_both_days_l2230_223042

-- Define the events probabilities
variables (P_M P_T P_N P_MT : ℝ)

-- Define the initial conditions
axiom h1 : P_M = 0.6
axiom h2 : P_T = 0.55
axiom h3 : P_N = 0.25

-- Define the statement to prove
theorem rain_on_both_days : P_MT = 0.4 :=
by
  -- The proof is omitted for now
  sorry

end rain_on_both_days_l2230_223042


namespace johns_daily_calorie_intake_l2230_223091

variable (breakfast lunch dinner shake : ℕ)
variable (num_shakes meals_per_day : ℕ)
variable (lunch_inc : ℕ)
variable (dinner_mult : ℕ)

-- Define the conditions from the problem
def john_calories_per_day 
  (breakfast := 500)
  (lunch := breakfast + lunch_inc)
  (dinner := lunch * dinner_mult)
  (shake := 300)
  (num_shakes := 3)
  (lunch_inc := breakfast / 4)
  (dinner_mult := 2)
  : ℕ :=
  breakfast + lunch + dinner + (shake * num_shakes)

theorem johns_daily_calorie_intake : john_calories_per_day = 3275 := by
  sorry

end johns_daily_calorie_intake_l2230_223091


namespace number_of_members_l2230_223004

theorem number_of_members (n : ℕ) (h1 : ∀ m : ℕ, m = n → m * m = 1936) : n = 44 :=
by
  -- Proof omitted
  sorry

end number_of_members_l2230_223004


namespace eliot_account_balance_l2230_223095

-- Definitions for the conditions
variables {A E : ℝ}

--- Conditions rephrased into Lean:
-- 1. Al has more money than Eliot.
def al_more_than_eliot (A E : ℝ) : Prop := A > E

-- 2. The difference between their two accounts is 1/12 of the sum of their two accounts.
def difference_condition (A E : ℝ) : Prop := A - E = (1 / 12) * (A + E)

-- 3. If Al's account were to increase by 10% and Eliot's account were to increase by 15%, 
--     then Al would have exactly $22 more than Eliot in his account.
def percentage_increase_condition (A E : ℝ) : Prop := 1.10 * A = 1.15 * E + 22

-- Prove the total statement
theorem eliot_account_balance : 
  ∀ (A E : ℝ), al_more_than_eliot A E → difference_condition A E → percentage_increase_condition A E → E = 146.67 :=
by
  intros A E h1 h2 h3
  sorry

end eliot_account_balance_l2230_223095


namespace parallelogram_area_formula_l2230_223022

noncomputable def parallelogram_area (ha hb : ℝ) (γ : ℝ) : ℝ := 
  ha * hb / Real.sin γ

theorem parallelogram_area_formula (ha hb γ : ℝ) (a b : ℝ) 
  (h₁ : Real.sin γ ≠ 0) :
  (parallelogram_area ha hb γ = ha * hb / Real.sin γ) := by
  sorry

end parallelogram_area_formula_l2230_223022
