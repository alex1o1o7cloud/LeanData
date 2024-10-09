import Mathlib

namespace base_conversion_problem_l2099_209935

theorem base_conversion_problem (b : ℕ) (h : b^2 + 2 * b - 25 = 0) : b = 3 :=
sorry

end base_conversion_problem_l2099_209935


namespace orthocenter_of_ABC_is_correct_l2099_209904

structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

def A : Point3D := {x := 2, y := 3, z := -1}
def B : Point3D := {x := 6, y := -1, z := 2}
def C : Point3D := {x := 4, y := 5, z := 4}

def orthocenter (A B C : Point3D) : Point3D := {
  x := 101 / 33,
  y := 95 / 33,
  z := 47 / 33
}

theorem orthocenter_of_ABC_is_correct : orthocenter A B C = {x := 101 / 33, y := 95 / 33, z := 47 / 33} :=
  sorry

end orthocenter_of_ABC_is_correct_l2099_209904


namespace cows_on_farm_l2099_209921

theorem cows_on_farm (weekly_production_per_6_cows : ℕ) 
                     (production_over_5_weeks : ℕ) 
                     (number_of_weeks : ℕ) 
                     (cows : ℕ) :
  weekly_production_per_6_cows = 108 →
  production_over_5_weeks = 2160 →
  number_of_weeks = 5 →
  (cows * (weekly_production_per_6_cows / 6) * number_of_weeks = production_over_5_weeks) →
  cows = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end cows_on_farm_l2099_209921


namespace remaining_amount_division_l2099_209946

-- Definitions
def total_amount : ℕ := 2100
def number_of_participants : ℕ := 8
def amount_already_raised : ℕ := 150

-- Proof problem statement
theorem remaining_amount_division :
  (total_amount - amount_already_raised) / (number_of_participants - 1) = 279 :=
by
  sorry

end remaining_amount_division_l2099_209946


namespace ones_digit_of_largest_power_of_three_dividing_27_factorial_l2099_209909

theorem ones_digit_of_largest_power_of_three_dividing_27_factorial :
  let k := (27 / 3) + (27 / 9) + (27 / 27)
  let x := 3^k
  (x % 10) = 3 := by
  sorry

end ones_digit_of_largest_power_of_three_dividing_27_factorial_l2099_209909


namespace find_divisible_by_3_l2099_209930

theorem find_divisible_by_3 (n : ℕ) : 
  (∀ k : ℕ, k ≤ 12 → (3 * k + 12) ≤ n) ∧ 
  (∀ m : ℕ, m ≥ 13 → (3 * m + 12) > n) →
  n = 48 :=
by
  sorry

end find_divisible_by_3_l2099_209930


namespace parrots_per_cage_l2099_209911

theorem parrots_per_cage (total_birds : ℕ) (num_cages : ℕ) (parakeets_per_cage : ℕ) (total_parrots : ℕ) :
  total_birds = 48 → num_cages = 6 → parakeets_per_cage = 2 → total_parrots = 36 →
  ∀ P : ℕ, (total_parrots = P * num_cages) → P = 6 :=
by
  intros h1 h2 h3 h4 P h5
  subst h1 h2 h3 h4
  sorry

end parrots_per_cage_l2099_209911


namespace triangle_inequality_l2099_209923

-- Let α, β, γ be the angles of a triangle opposite to its sides with lengths a, b, and c, respectively.
variables (α β γ a b c : ℝ)

-- Assume that α, β, γ are positive.
axiom positive_angles : α > 0 ∧ β > 0 ∧ γ > 0
-- Assume that a, b, c are the sides opposite to angles α, β, γ respectively.
axiom positive_sides : a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_inequality :
  a * (1 / β + 1 / γ) + b * (1 / γ + 1 / α) + c * (1 / α + 1 / β) ≥ 
  2 * (a / α + b / β + c / γ) :=
sorry

end triangle_inequality_l2099_209923


namespace find_m_find_A_inter_CUB_l2099_209983

-- Definitions of sets A and B given m
def A (m : ℤ) : Set ℤ := {-4, 2 * m - 1, m ^ 2}
def B (m : ℤ) : Set ℤ := {9, m - 5, 1 - m}

-- Define the universal set U
def U : Set ℤ := Set.univ

-- First part: Prove that m = -3
theorem find_m (m : ℤ) : A m ∩ B m = {9} → m = -3 := sorry

-- Condition that m = -3 is true
def m_val : ℤ := -3

-- Second part: Prove A ∩ C_U B = {-4, -7}
theorem find_A_inter_CUB: A m_val ∩ (U \ B m_val) = {-4, -7} := sorry

end find_m_find_A_inter_CUB_l2099_209983


namespace bread_last_days_is_3_l2099_209924

-- Define conditions
def num_members : ℕ := 4
def slices_breakfast : ℕ := 3
def slices_snacks : ℕ := 2
def slices_loaf : ℕ := 12
def num_loaves : ℕ := 5

-- Define the problem statement
def bread_last_days : ℕ :=
  (num_loaves * slices_loaf) / (num_members * (slices_breakfast + slices_snacks))

-- State the theorem to be proved
theorem bread_last_days_is_3 : bread_last_days = 3 :=
  sorry

end bread_last_days_is_3_l2099_209924


namespace range_of_c_l2099_209986

noncomputable def p (c : ℝ) : Prop := ∀ x : ℝ, (2 * c - 1) ^ x = (2 * c - 1) ^ x

def q (c : ℝ) : Prop := ∀ x : ℝ, x + |x - 2 * c| > 1

theorem range_of_c (c : ℝ) (h1 : c > 0)
  (h2 : p c ∨ q c) (h3 : ¬ (p c ∧ q c)) : c ≥ 1 :=
sorry

end range_of_c_l2099_209986


namespace geometric_progression_solution_l2099_209945

theorem geometric_progression_solution 
  (b1 q : ℝ)
  (condition1 : (b1^2 / (1 + q + q^2) = 48 / 7))
  (condition2 : (b1^2 / (1 + q^2) = 144 / 17)) 
  : (b1 = 3 ∨ b1 = -3) ∧ q = 1 / 4 :=
by
  sorry

end geometric_progression_solution_l2099_209945


namespace DiagonalsOfShapesBisectEachOther_l2099_209962

structure Shape where
  bisect_diagonals : Prop

def is_parallelogram (s : Shape) : Prop := s.bisect_diagonals
def is_rectangle (s : Shape) : Prop := s.bisect_diagonals
def is_rhombus (s : Shape) : Prop := s.bisect_diagonals
def is_square (s : Shape) : Prop := s.bisect_diagonals

theorem DiagonalsOfShapesBisectEachOther (s : Shape) :
  is_parallelogram s ∨ is_rectangle s ∨ is_rhombus s ∨ is_square s → s.bisect_diagonals := by
  sorry

end DiagonalsOfShapesBisectEachOther_l2099_209962


namespace waiter_income_fraction_l2099_209950

theorem waiter_income_fraction (S T : ℝ) (hT : T = 5/4 * S) :
  T / (S + T) = 5 / 9 :=
by
  sorry

end waiter_income_fraction_l2099_209950


namespace expression_in_terms_of_p_q_l2099_209933

variables {α β γ δ p q : ℝ}

-- Let α and β be the roots of x^2 - 2px + 1 = 0
axiom root_α_β : ∀ x, (x - α) * (x - β) = x^2 - 2 * p * x + 1

-- Let γ and δ be the roots of x^2 + qx + 2 = 0
axiom root_γ_δ : ∀ x, (x - γ) * (x - δ) = x^2 + q * x + 2

-- Expression to be proved
theorem expression_in_terms_of_p_q :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = 2 * (p - q) ^ 2 :=
sorry

end expression_in_terms_of_p_q_l2099_209933


namespace min_value_mn_squared_l2099_209965

theorem min_value_mn_squared (a b c m n : ℝ) 
  (h_triangle: a^2 + b^2 = c^2)
  (h_line: a * m + b * n + 2 * c = 0):
  m^2 + n^2 = 4 :=
by
  sorry

end min_value_mn_squared_l2099_209965


namespace combined_salaries_l2099_209979

theorem combined_salaries (A B C D E : ℝ) 
  (hC : C = 11000) 
  (hAverage : (A + B + C + D + E) / 5 = 8200) : 
  A + B + D + E = 30000 := 
by 
  sorry

end combined_salaries_l2099_209979


namespace average_salary_rest_l2099_209968

theorem average_salary_rest (number_of_workers : ℕ) 
                            (avg_salary_all : ℝ) 
                            (number_of_technicians : ℕ) 
                            (avg_salary_technicians : ℝ) 
                            (rest_workers : ℕ) 
                            (total_salary_all : ℝ) 
                            (total_salary_technicians : ℝ) 
                            (total_salary_rest : ℝ) 
                            (avg_salary_rest : ℝ) 
                            (h1 : number_of_workers = 28)
                            (h2 : avg_salary_all = 8000)
                            (h3 : number_of_technicians = 7)
                            (h4 : avg_salary_technicians = 14000)
                            (h5 : rest_workers = number_of_workers - number_of_technicians)
                            (h6 : total_salary_all = number_of_workers * avg_salary_all)
                            (h7 : total_salary_technicians = number_of_technicians * avg_salary_technicians)
                            (h8 : total_salary_rest = total_salary_all - total_salary_technicians)
                            (h9 : avg_salary_rest = total_salary_rest / rest_workers) :
  avg_salary_rest = 6000 :=
by {
  -- the proof would go here
  sorry
}

end average_salary_rest_l2099_209968


namespace surface_area_of_interior_box_l2099_209929

def original_sheet_width : ℕ := 40
def original_sheet_length : ℕ := 50
def corner_cut_side : ℕ := 8
def corners_count : ℕ := 4

def area_of_original_sheet : ℕ := original_sheet_width * original_sheet_length
def area_of_one_corner_cut : ℕ := corner_cut_side * corner_cut_side
def total_area_removed : ℕ := corners_count * area_of_one_corner_cut
def area_of_remaining_sheet : ℕ := area_of_original_sheet - total_area_removed

theorem surface_area_of_interior_box : area_of_remaining_sheet = 1744 :=
by
  sorry

end surface_area_of_interior_box_l2099_209929


namespace question1_question2_l2099_209944

def f (x : ℝ) : ℝ := |x + 7| + |x - 1|

theorem question1 (x : ℝ) : ∀ m : ℝ, (∀ x : ℝ, f x ≥ m) → m ≤ 8 :=
by sorry

theorem question2 (x : ℝ) : (∀ x : ℝ, |x - 3| - 2 * x ≤ 2 * 8 - 12) ↔ (x ≥ -1/3) :=
by sorry

end question1_question2_l2099_209944


namespace number_of_ordered_pairs_xy_2007_l2099_209947

theorem number_of_ordered_pairs_xy_2007 : 
  ∃ n, n = 6 ∧ (∀ x y : ℕ, x * y = 2007 → x > 0 ∧ y > 0) :=
sorry

end number_of_ordered_pairs_xy_2007_l2099_209947


namespace complement_of_intersection_l2099_209954

open Set

-- Define the universal set U
def U := @univ ℝ
-- Define the sets M and N
def M : Set ℝ := {x | x >= 2}
def N : Set ℝ := {x | 0 <= x ∧ x < 5}

-- Define M ∩ N
def M_inter_N := M ∩ N

-- Define the complement of M ∩ N with respect to U
def C_U (A : Set ℝ) := Aᶜ

theorem complement_of_intersection :
  C_U M_inter_N = {x : ℝ | x < 2 ∨ x ≥ 5} := 
by 
  sorry

end complement_of_intersection_l2099_209954


namespace find_k_l2099_209907

theorem find_k (x y k : ℝ)
  (h1 : 3 * x + 2 * y = k + 1)
  (h2 : 2 * x + 3 * y = k)
  (h3 : x + y = 3) : k = 7 := sorry

end find_k_l2099_209907


namespace area_of_sector_AOB_l2099_209977

-- Definitions for the conditions
def circumference_sector_AOB : Real := 6 -- Circumference of sector AOB
def central_angle_AOB : Real := 1 -- Central angle of sector AOB

-- Theorem stating the area of the sector is 2 cm²
theorem area_of_sector_AOB (C : Real) (θ : Real) (hC : C = circumference_sector_AOB) (hθ : θ = central_angle_AOB) : 
    ∃ S : Real, S = 2 :=
by
  sorry

end area_of_sector_AOB_l2099_209977


namespace arithmetic_sequence_suff_nec_straight_line_l2099_209920

variable (n : ℕ) (P_n : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ m : ℕ, a (m + 1) = a m + d

def lies_on_straight_line (P : ℕ → ℝ) : Prop :=
  ∃ m b, ∀ n, P n = m * n + b

theorem arithmetic_sequence_suff_nec_straight_line
  (h_n : 0 < n)
  (h_arith : arithmetic_sequence P_n) :
  lies_on_straight_line P_n ↔ arithmetic_sequence P_n :=
sorry

end arithmetic_sequence_suff_nec_straight_line_l2099_209920


namespace solve_fraction_equation_l2099_209967

theorem solve_fraction_equation (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ -3) :
  (2 / x + x / (x + 3) = 1) ↔ x = 6 := 
by
  sorry

end solve_fraction_equation_l2099_209967


namespace shortest_path_octahedron_l2099_209939

theorem shortest_path_octahedron 
  (edge_length : ℝ) (h : edge_length = 2) 
  (d : ℝ) : d = 2 :=
by
  sorry

end shortest_path_octahedron_l2099_209939


namespace find_expression_value_l2099_209941

variable (x y z : ℚ)
variable (h1 : x - y + 2 * z = 1)
variable (h2 : x + y + 4 * z = 3)

theorem find_expression_value : x + 2 * y + 5 * z = 4 := 
by {
  sorry
}

end find_expression_value_l2099_209941


namespace points_on_parabola_l2099_209906

theorem points_on_parabola (t : ℝ) : 
  ∃ a b c : ℝ, ∀ (x y: ℝ), (x, y) = (Real.cos t ^ 2, Real.sin (2 * t)) → y^2 = 4 * x - 4 * x^2 := 
by
  sorry

end points_on_parabola_l2099_209906


namespace largest_multiple_of_11_lt_neg150_l2099_209971

theorem largest_multiple_of_11_lt_neg150 : ∃ (x : ℤ), (x % 11 = 0) ∧ (x < -150) ∧ (∀ y : ℤ, y % 11 = 0 → y < -150 → y ≤ x) ∧ x = -154 :=
by
  sorry

end largest_multiple_of_11_lt_neg150_l2099_209971


namespace sum_of_ages_l2099_209940

-- Definitions based on conditions
variables (J S : ℝ) -- J and S are real numbers

-- First condition: Jane is five years older than Sarah
def jane_older_than_sarah := J = S + 5

-- Second condition: Nine years from now, Jane will be three times as old as Sarah was three years ago
def future_condition := J + 9 = 3 * (S - 3)

-- Conclusion to prove
theorem sum_of_ages (h1 : jane_older_than_sarah J S) (h2 : future_condition J S) : J + S = 28 :=
by
  sorry

end sum_of_ages_l2099_209940


namespace polynomial_value_l2099_209913

theorem polynomial_value (a b : ℝ) : 
  (|a - 2| + (b + 1/2)^2 = 0) → (2 * a * b^2 + a^2 * b) - (3 * a * b^2 + a^2 * b - 1) = 1/2 :=
by
  sorry

end polynomial_value_l2099_209913


namespace solve_logarithmic_inequality_l2099_209948

theorem solve_logarithmic_inequality :
  {x : ℝ | 2 * (Real.log x / Real.log 0.5)^2 + 9 * (Real.log x / Real.log 0.5) + 9 ≤ 0} = 
  {x : ℝ | 2 * Real.sqrt 2 ≤ x ∧ x ≤ 8} :=
sorry

end solve_logarithmic_inequality_l2099_209948


namespace original_integer_is_26_l2099_209966

theorem original_integer_is_26 (x y z w : ℕ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : 0 < w)
(h₅ : x ≠ y) (h₆ : x ≠ z) (h₇ : x ≠ w) (h₈ : y ≠ z) (h₉ : y ≠ w) (h₁₀ : z ≠ w)
(h₁₁ : (x + y + z) / 3 + w = 34)
(h₁₂ : (x + y + w) / 3 + z = 22)
(h₁₃ : (x + z + w) / 3 + y = 26)
(h₁₄ : (y + z + w) / 3 + x = 18) :
    w = 26 := 
sorry

end original_integer_is_26_l2099_209966


namespace inequality_solution_exists_l2099_209975

theorem inequality_solution_exists (a : ℝ) : 
  ∃ x : ℝ, x > 2 ∧ x > -1 ∧ x > a := 
by
  sorry

end inequality_solution_exists_l2099_209975


namespace platform_length_calc_l2099_209953

noncomputable def length_of_platform (V : ℝ) (T : ℝ) (L_train : ℝ) : ℝ :=
  (V * 1000 / 3600) * T - L_train

theorem platform_length_calc (speed : ℝ) (time : ℝ) (length_train : ℝ):
  speed = 72 →
  time = 26 →
  length_train = 280.0416 →
  length_of_platform speed time length_train = 239.9584 := by
  intros
  unfold length_of_platform
  sorry

end platform_length_calc_l2099_209953


namespace total_cakes_correct_l2099_209927

-- Define the initial number of full-size cakes
def initial_cakes : ℕ := 350

-- Define the number of additional full-size cakes made
def additional_cakes : ℕ := 125

-- Define the number of half-cakes made
def half_cakes : ℕ := 75

-- Convert half-cakes to full-size cakes, considering only whole cakes
def half_to_full_cakes := (half_cakes / 2)

-- Total full-size cakes calculation
def total_cakes :=
  initial_cakes + additional_cakes + half_to_full_cakes

-- Prove the total number of full-size cakes
theorem total_cakes_correct : total_cakes = 512 :=
by
  -- Skip the proof
  sorry

end total_cakes_correct_l2099_209927


namespace jon_and_mary_frosting_l2099_209916

-- Jon frosts a cupcake every 40 seconds
def jon_frost_rate : ℚ := 1 / 40

-- Mary frosts a cupcake every 24 seconds
def mary_frost_rate : ℚ := 1 / 24

-- Combined frosting rate of Jon and Mary
def combined_frost_rate : ℚ := jon_frost_rate + mary_frost_rate

-- Total time in seconds for 12 minutes
def total_time_seconds : ℕ := 12 * 60

-- Calculate the total number of cupcakes frosted in 12 minutes
def total_cupcakes_frosted (time_seconds : ℕ) (rate : ℚ) : ℚ :=
  time_seconds * rate

theorem jon_and_mary_frosting : total_cupcakes_frosted total_time_seconds combined_frost_rate = 48 := by
  sorry

end jon_and_mary_frosting_l2099_209916


namespace rectangular_container_volume_l2099_209918

theorem rectangular_container_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) : 
  a * b * c = 60 :=
by
  sorry

end rectangular_container_volume_l2099_209918


namespace num_solutions_system_eqns_l2099_209956

theorem num_solutions_system_eqns :
  ∃ (c : ℕ), 
    (∀ (a1 a2 a3 a4 a5 a6 : ℕ), 
       a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6 = 26 ∧ 
       a1 + a2 + a3 + a4 + a5 + a6 = 5 → 
       (a1, a2, a3, a4, a5, a6) ∈ (solutions : Finset (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ))) ∧
    solutions.card = 5 := sorry

end num_solutions_system_eqns_l2099_209956


namespace binomial_coefficient_third_term_l2099_209978

theorem binomial_coefficient_third_term (x a : ℝ) (h : 10 * a^3 * x = 80) : a = 2 :=
by
  sorry

end binomial_coefficient_third_term_l2099_209978


namespace fraction_of_earth_habitable_l2099_209922

theorem fraction_of_earth_habitable :
  ∀ (earth_surface land_area inhabitable_land_area : ℝ),
    land_area = 1 / 3 → 
    inhabitable_land_area = 1 / 4 → 
    (earth_surface * land_area * inhabitable_land_area) = 1 / 12 :=
  by
    intros earth_surface land_area inhabitable_land_area h_land h_inhabitable
    sorry

end fraction_of_earth_habitable_l2099_209922


namespace tan_x_tan_y_relation_l2099_209932

/-- If 
  (sin x / cos y) + (sin y / cos x) = 2 
  and 
  (cos x / sin y) + (cos y / sin x) = 3, 
  then 
  (tan x / tan y) + (tan y / tan x) = 16 / 3.
 -/
theorem tan_x_tan_y_relation (x y : ℝ)
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 16 / 3 :=
sorry

end tan_x_tan_y_relation_l2099_209932


namespace product_of_consecutive_integers_l2099_209914

theorem product_of_consecutive_integers (l : List ℤ) (h1 : l.length = 2019) (h2 : l.sum = 2019) : l.prod = 0 := 
sorry

end product_of_consecutive_integers_l2099_209914


namespace rectangle_circle_ratio_l2099_209999

theorem rectangle_circle_ratio (r s : ℝ) (h : ∀ r s : ℝ, 2 * r * s - π * r^2 = π * r^2) : s / (2 * r) = π / 2 :=
by
  sorry

end rectangle_circle_ratio_l2099_209999


namespace smallest_percent_increase_l2099_209925

-- Define the values of each question.
def value (n : ℕ) : ℕ :=
  match n with
  | 1  => 150
  | 2  => 300
  | 3  => 450
  | 4  => 600
  | 5  => 800
  | 6  => 1500
  | 7  => 3000
  | 8  => 6000
  | 9  => 12000
  | 10 => 24000
  | 11 => 48000
  | 12 => 96000
  | 13 => 192000
  | 14 => 384000
  | 15 => 768000
  | _ => 0

-- Define the percent increase between two values.
def percent_increase (v1 v2 : ℕ) : ℚ :=
  ((v2 - v1 : ℕ) : ℚ) / v1 * 100 

-- Prove that the smallest percent increase is between question 4 and 5.
theorem smallest_percent_increase :
  percent_increase (value 4) (value 5) = 33.33 := 
by
  sorry

end smallest_percent_increase_l2099_209925


namespace simplify_expression_l2099_209903

theorem simplify_expression (x : ℝ) : (3 * x + 30) + (150 * x - 45) = 153 * x - 15 :=
by
  sorry

end simplify_expression_l2099_209903


namespace range_of_k_l2099_209943

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, f (-x^2 + 3 * x) + f (x - 2 * k) ≤ 0) ↔ k ≥ 2 :=
by
  sorry

end range_of_k_l2099_209943


namespace track_circumference_l2099_209957

variable (A B : Nat → ℝ)
variable (speedA speedB : ℝ)
variable (x : ℝ) -- half the circumference of the track
variable (y : ℝ) -- the circumference of the track

theorem track_circumference
  (x_pos : 0 < x)
  (y_def : y = 2 * x)
  (start_opposite : A 0 = 0 ∧ B 0 = x)
  (B_first_meet_150 : ∃ t₁, B t₁ = 150 ∧ A t₁ = x - 150)
  (A_second_meet_90 : ∃ t₂, A t₂ = 2 * x - 90 ∧ B t₂ = x + 90) :
  y = 720 := 
by 
  sorry

end track_circumference_l2099_209957


namespace total_points_sum_l2099_209919

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls := [6, 2, 5, 3, 4]
def carlos_rolls := [3, 2, 2, 6, 1]

def score (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem total_points_sum :
  score allie_rolls + score carlos_rolls = 44 :=
by
  sorry

end total_points_sum_l2099_209919


namespace min_value_expression_l2099_209955

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : y = Real.sqrt x) :
  ∃ c, c = 2 ∧ ∀ u v : ℝ, 0 < u → v = Real.sqrt u → (u^2 + v^4) / (u * v^2) = c :=
by
  sorry

end min_value_expression_l2099_209955


namespace children_exceed_bridge_limit_l2099_209984

theorem children_exceed_bridge_limit :
  ∀ (Kelly_weight : ℕ) (Megan_weight : ℕ) (Mike_weight : ℕ),
  Kelly_weight = 34 ∧
  Kelly_weight = (85 * Megan_weight) / 100 ∧
  Mike_weight = Megan_weight + 5 →
  Kelly_weight + Megan_weight + Mike_weight - 100 = 19 :=
by sorry

end children_exceed_bridge_limit_l2099_209984


namespace inequality_proof_l2099_209952

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a ≥ b) (h5 : b ≥ c) :
  a + b + c ≤ (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ∧
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≤ (a^3 / (b * c)) + (b^3 / (c * a)) + (c^3 / (a * b)) :=
by
  sorry

end inequality_proof_l2099_209952


namespace log_identity_l2099_209961

theorem log_identity (a b : ℝ) (h1 : a = Real.log 144 / Real.log 4) (h2 : b = Real.log 12 / Real.log 2) : a = b := 
by
  sorry

end log_identity_l2099_209961


namespace baron_munchausen_failed_l2099_209931

theorem baron_munchausen_failed : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 → ¬∃ (d1 d2 : ℕ), ∃ (k : ℕ), n * 100 + (d1 * 10 + d2) = k^2 := 
by
  intros n hn
  obtain ⟨h10, h99⟩ := hn
  sorry

end baron_munchausen_failed_l2099_209931


namespace fraction_of_Bhupathi_is_point4_l2099_209942

def abhinav_and_bhupathi_amounts (A B : ℝ) : Prop :=
  A + B = 1210 ∧ B = 484

theorem fraction_of_Bhupathi_is_point4 (A B : ℝ) (x : ℝ) (h : abhinav_and_bhupathi_amounts A B) :
  (4 / 15) * A = x * B → x = 0.4 :=
by
  sorry

end fraction_of_Bhupathi_is_point4_l2099_209942


namespace sam_runs_more_than_sarah_sue_runs_less_than_sarah_l2099_209960

-- Definitions based on the problem conditions
def street_width : ℝ := 25
def block_side_length : ℝ := 500
def sarah_perimeter : ℝ := 4 * block_side_length
def sam_perimeter : ℝ := 4 * (block_side_length + 2 * street_width)
def sue_perimeter : ℝ := 4 * (block_side_length - 2 * street_width)

-- The proof problem statements
theorem sam_runs_more_than_sarah : sam_perimeter - sarah_perimeter = 200 := by
  sorry

theorem sue_runs_less_than_sarah : sarah_perimeter - sue_perimeter = 200 := by
  sorry

end sam_runs_more_than_sarah_sue_runs_less_than_sarah_l2099_209960


namespace find_possible_values_l2099_209994

noncomputable def possible_values (a b : ℝ) : Set ℝ :=
  { x | ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1/a + 1/b) }

theorem find_possible_values :
  (∀ (a b : ℝ), 0 < a → 0 < b → a + b = 2 → (1 / a + 1 / b) ∈ Set.Ici 2) ∧
  (∀ y, y ∈ Set.Ici 2 → ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 2 ∧ y = (1 / a + 1 / b)) :=
by
  sorry

end find_possible_values_l2099_209994


namespace sequence_conjecture_l2099_209934

theorem sequence_conjecture (a : ℕ → ℚ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n / (a n + 1)) :
  ∀ n : ℕ, 0 < n → a n = 1 / n := by
  sorry

end sequence_conjecture_l2099_209934


namespace probability_excluded_probability_selected_l2099_209972

-- Define the population size and the sample size
def population_size : ℕ := 1005
def sample_size : ℕ := 50
def excluded_count : ℕ := 5

-- Use these values within the theorems
theorem probability_excluded : (excluded_count : ℚ) / (population_size : ℚ) = 5 / 1005 :=
by sorry

theorem probability_selected : (sample_size : ℚ) / (population_size : ℚ) = 50 / 1005 :=
by sorry

end probability_excluded_probability_selected_l2099_209972


namespace factorize_expr_l2099_209900

theorem factorize_expr (y : ℝ) : 3 * y ^ 2 - 6 * y + 3 = 3 * (y - 1) ^ 2 :=
by
  sorry

end factorize_expr_l2099_209900


namespace olympic_iberic_sets_containing_33_l2099_209908

/-- A set of positive integers is iberic if it is a subset of {2, 3, ..., 2018},
    and whenever m, n are both in the set, gcd(m, n) is also in the set. -/
def is_iberic_set (X : Set ℕ) : Prop :=
  X ⊆ {n | 2 ≤ n ∧ n ≤ 2018} ∧ ∀ m n, m ∈ X → n ∈ X → Nat.gcd m n ∈ X

/-- An iberic set is olympic if it is not properly contained in any other iberic set. -/
def is_olympic_set (X : Set ℕ) : Prop :=
  is_iberic_set X ∧ ∀ Y, is_iberic_set Y → X ⊂ Y → False

/-- The olympic iberic sets containing 33 are exactly {3, 6, 9, ..., 2016} and {11, 22, 33, ..., 2013}. -/
theorem olympic_iberic_sets_containing_33 :
  ∀ X, is_iberic_set X ∧ 33 ∈ X → X = {n | 3 ∣ n ∧ 2 ≤ n ∧ n ≤ 2016} ∨ X = {n | 11 ∣ n ∧ 11 ≤ n ∧ n ≤ 2013} :=
by
  sorry

end olympic_iberic_sets_containing_33_l2099_209908


namespace jordan_oreos_l2099_209973

def oreos (james jordan total : ℕ) : Prop :=
  james = 2 * jordan + 3 ∧
  jordan + james = total

theorem jordan_oreos (J : ℕ) (h : oreos (2 * J + 3) J 36) : J = 11 :=
by
  sorry

end jordan_oreos_l2099_209973


namespace bonnets_per_orphanage_correct_l2099_209902

-- Definitions for each day's bonnet count
def monday_bonnets := 10
def tuesday_and_wednesday_bonnets := 2 * monday_bonnets
def thursday_bonnets := monday_bonnets + 5
def friday_bonnets := thursday_bonnets - 5
def saturday_bonnets := friday_bonnets - 8
def sunday_bonnets := 3 * saturday_bonnets

-- Total bonnets made in the week
def total_bonnets := 
  monday_bonnets +
  tuesday_and_wednesday_bonnets +
  thursday_bonnets +
  friday_bonnets +
  saturday_bonnets +
  sunday_bonnets

-- The number of orphanages
def orphanages := 10

-- Bonnets sent to each orphanage
def bonnets_per_orphanage := total_bonnets / orphanages

theorem bonnets_per_orphanage_correct :
  bonnets_per_orphanage = 6 :=
by
  sorry

end bonnets_per_orphanage_correct_l2099_209902


namespace geom_series_first_term_l2099_209905

theorem geom_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) 
  (h_r : r = 1 / 4) 
  (h_S : S = 80)
  : a = 60 :=
by
  sorry

end geom_series_first_term_l2099_209905


namespace initial_blue_marbles_l2099_209981

theorem initial_blue_marbles (B R : ℕ) 
    (h1 : 3 * B = 5 * R) 
    (h2 : 4 * (B - 10) = R + 25) : 
    B = 19 := 
sorry

end initial_blue_marbles_l2099_209981


namespace six_x_mod_nine_l2099_209969

theorem six_x_mod_nine (x : ℕ) (k : ℕ) (hx : x = 9 * k + 5) : (6 * x) % 9 = 3 :=
by
  sorry

end six_x_mod_nine_l2099_209969


namespace tangent_line_passes_through_origin_l2099_209951

noncomputable def curve (α : ℝ) (x : ℝ) : ℝ := x^α + 1

theorem tangent_line_passes_through_origin (α : ℝ)
  (h_tangent : ∀ (x : ℝ), curve α 1 + (α * (x - 1)) - 2 = curve α x) :
  α = 2 :=
sorry

end tangent_line_passes_through_origin_l2099_209951


namespace maximum_value_of_x_minus_y_l2099_209989

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l2099_209989


namespace spencer_session_duration_l2099_209912

-- Definitions of the conditions
def jumps_per_minute : ℕ := 4
def sessions_per_day : ℕ := 2
def total_jumps : ℕ := 400
def total_days : ℕ := 5

-- Calculation target: find the duration of each session
def jumps_per_day : ℕ := total_jumps / total_days
def jumps_per_session : ℕ := jumps_per_day / sessions_per_day
def session_duration := jumps_per_session / jumps_per_minute

theorem spencer_session_duration :
  session_duration = 10 := 
sorry

end spencer_session_duration_l2099_209912


namespace p_sufficient_but_not_necessary_for_q_l2099_209959

def p (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 5
def q (x : ℝ) : Prop := (x - 5) * (x + 1) < 0

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ∃ x : ℝ, q x ∧ ¬ p x :=
by
  sorry

end p_sufficient_but_not_necessary_for_q_l2099_209959


namespace sum_of_areas_squares_l2099_209991

theorem sum_of_areas_squares (a : ℝ) : 
  (∑' n : ℕ, (a^2 / 4^n)) = (4 * a^2 / 3) :=
by
  sorry

end sum_of_areas_squares_l2099_209991


namespace no_solutions_l2099_209982

theorem no_solutions : ¬ ∃ x : ℝ, (6 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 4) := by
  sorry

end no_solutions_l2099_209982


namespace solve_for_x_in_equation_l2099_209915

theorem solve_for_x_in_equation (x : ℝ)
  (h : (2 / 7) * (1 / 4) * x = 12) : x = 168 :=
sorry

end solve_for_x_in_equation_l2099_209915


namespace smallest_k_for_divisibility_l2099_209997

theorem smallest_k_for_divisibility (z : ℂ) (hz : z^7 = 1) : ∃ k : ℕ, (∀ m : ℕ, z ^ (m * k) = 1) ∧ k = 84 :=
sorry

end smallest_k_for_divisibility_l2099_209997


namespace extremum_value_and_min_on_interval_l2099_209995

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

theorem extremum_value_and_min_on_interval
  (a b c : ℝ)
  (h1_eq : 12 * a + b = 0)
  (h2_eq : 4 * a + b = -8)
  (h_max : 16 + c = 28) :
  min (min (f a b c (-3)) (f a b c 3)) (f a b c 2) = -4 :=
by sorry

end extremum_value_and_min_on_interval_l2099_209995


namespace sum_xyz_l2099_209990

theorem sum_xyz (x y z : ℝ) (h : (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0) : x + y + z = 11 := 
by
  sorry

end sum_xyz_l2099_209990


namespace trapezium_area_l2099_209970

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) : 
  1/2 * (a + b) * h = 285 :=
by {
  sorry
}

end trapezium_area_l2099_209970


namespace exists_eight_integers_sum_and_product_eight_l2099_209938

theorem exists_eight_integers_sum_and_product_eight :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 : ℤ), 
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 8 ∧ 
  a1 * a2 * a3 * a4 * a5 * a6 * a7 * a8 = 8 :=
by
  -- The existence proof can be constructed here
  sorry

end exists_eight_integers_sum_and_product_eight_l2099_209938


namespace product_is_zero_l2099_209996

def product_series (a : ℤ) : ℤ :=
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * 
  (a - 4) * (a - 3) * (a - 2) * (a - 1) * a

theorem product_is_zero : product_series 3 = 0 :=
by
  sorry

end product_is_zero_l2099_209996


namespace range_of_a_for_monotonicity_l2099_209987

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a_for_monotonicity (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ 2 < a ∧ a ≤ 3 :=
by sorry

end range_of_a_for_monotonicity_l2099_209987


namespace inequality_1_inequality_2_inequality_3_inequality_4_l2099_209936

noncomputable def triangle_angles (a b c : ℝ) : Prop :=
  a + b + c = Real.pi

theorem inequality_1 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.sin a + Real.sin b + Real.sin c ≤ (3 * Real.sqrt 3 / 2) :=
sorry

theorem inequality_2 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.cos (a / 2) + Real.cos (b / 2) + Real.cos (c / 2) ≤ (3 * Real.sqrt 3 / 2) :=
sorry

theorem inequality_3 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.cos a * Real.cos b * Real.cos c ≤ (1 / 8) :=
sorry

theorem inequality_4 (a b c : ℝ) (h : triangle_angles a b c) :
  Real.sin (2 * a) + Real.sin (2 * b) + Real.sin (2 * c) ≤ Real.sin a + Real.sin b + Real.sin c :=
sorry

end inequality_1_inequality_2_inequality_3_inequality_4_l2099_209936


namespace expression_evaluation_valid_l2099_209901

theorem expression_evaluation_valid (a : ℝ) (h1 : a = 4) :
  (1 + (4 / (a ^ 2 - 4))) * ((a + 2) / a) = 2 := by
  sorry

end expression_evaluation_valid_l2099_209901


namespace total_boxes_count_l2099_209926

theorem total_boxes_count 
    (apples_per_crate : ℕ) (apples_crates : ℕ) 
    (oranges_per_crate : ℕ) (oranges_crates : ℕ) 
    (bananas_per_crate : ℕ) (bananas_crates : ℕ) 
    (rotten_apples_percentage : ℝ) (rotten_oranges_percentage : ℝ) (rotten_bananas_percentage : ℝ)
    (apples_per_box : ℕ) (oranges_per_box : ℕ) (bananas_per_box : ℕ) :
    apples_per_crate = 42 → apples_crates = 12 → 
    oranges_per_crate = 36 → oranges_crates = 15 → 
    bananas_per_crate = 30 → bananas_crates = 18 → 
    rotten_apples_percentage = 0.08 → rotten_oranges_percentage = 0.05 → rotten_bananas_percentage = 0.02 →
    apples_per_box = 10 → oranges_per_box = 12 → bananas_per_box = 15 →
    ∃ total_boxes : ℕ, total_boxes = 126 :=
by sorry

end total_boxes_count_l2099_209926


namespace find_z_l2099_209910

variable {x y z : ℝ}

theorem find_z (h : (1/x + 1/y = 1/z)) : z = (x * y) / (x + y) :=
  sorry

end find_z_l2099_209910


namespace fractional_part_tiled_l2099_209980

def room_length : ℕ := 12
def room_width : ℕ := 20
def number_of_tiles : ℕ := 40
def tile_area : ℕ := 1

theorem fractional_part_tiled :
  (number_of_tiles * tile_area : ℚ) / (room_length * room_width) = 1 / 6 :=
by
  sorry

end fractional_part_tiled_l2099_209980


namespace tangent_line_equation_l2099_209937

noncomputable def curve := fun x : ℝ => Real.sin (x + Real.pi / 3)

def tangent_line (x y : ℝ) : Prop :=
  x - 2 * y + Real.sqrt 3 = 0

theorem tangent_line_equation :
  tangent_line 0 (curve 0) := by
  unfold curve tangent_line
  sorry

end tangent_line_equation_l2099_209937


namespace master_codes_count_l2099_209993

def num_colors : ℕ := 7
def num_slots : ℕ := 5

theorem master_codes_count : num_colors ^ num_slots = 16807 := by
  sorry

end master_codes_count_l2099_209993


namespace betty_eggs_per_teaspoon_vanilla_l2099_209917

theorem betty_eggs_per_teaspoon_vanilla
  (sugar_cream_cheese_ratio : ℚ)
  (vanilla_cream_cheese_ratio : ℚ)
  (sugar_in_cups : ℚ)
  (eggs_used : ℕ)
  (expected_ratio : ℚ) :
  sugar_cream_cheese_ratio = 1/4 →
  vanilla_cream_cheese_ratio = 1/2 →
  sugar_in_cups = 2 →
  eggs_used = 8 →
  expected_ratio = 2 →
  (eggs_used / (sugar_in_cups * 4 * vanilla_cream_cheese_ratio)) = expected_ratio :=
by
  intros h1 h2 h3 h4 h5
  sorry

end betty_eggs_per_teaspoon_vanilla_l2099_209917


namespace XiaoMaHu_correct_calculation_l2099_209985

theorem XiaoMaHu_correct_calculation :
  (∃ A B C D : Prop, (A = ((a b : ℝ) → (a - b)^2 = a^2 - b^2)) ∧ 
                   (B = ((a : ℝ) → (-2 * a^3)^2 = 4 * a^6)) ∧ 
                   (C = ((a : ℝ) → a^3 + a^2 = 2 * a^5)) ∧ 
                   (D = ((a : ℝ) → -(a - 1) = -a - 1)) ∧ 
                   (¬A ∧ B ∧ ¬C ∧ ¬D)) :=
sorry

end XiaoMaHu_correct_calculation_l2099_209985


namespace ratio_jl_jm_l2099_209998

-- Define the side length of the square NOPQ as s
variable (s : ℝ)

-- Define the length (l) and width (m) of the rectangle JKLM
variable (l m : ℝ)

-- Conditions given in the problem
variable (area_overlap : ℝ)
variable (area_condition1 : area_overlap = 0.25 * s * s)
variable (area_condition2 : area_overlap = 0.40 * l * m)

theorem ratio_jl_jm (h1 : area_overlap = 0.25 * s * s) (h2 : area_overlap = 0.40 * l * m) : l / m = 2 / 5 :=
by
  sorry

end ratio_jl_jm_l2099_209998


namespace ratio_of_larger_to_smaller_l2099_209958

theorem ratio_of_larger_to_smaller (a b : ℝ) (h : a > 0) (h' : b > 0) (h_sum_diff : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
by
  sorry

end ratio_of_larger_to_smaller_l2099_209958


namespace minimum_area_of_triangle_l2099_209949

def parabola_focus : Prop :=
  ∃ F : ℝ × ℝ, F = (1, 0)

def on_parabola (A B : ℝ × ℝ) : Prop :=
  (A.2 ^ 2 = 4 * A.1 ∧ B.2 ^ 2 = 4 * B.1) ∧ (A.2 * B.2 < 0)

def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = -4

noncomputable def area (A B : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * B.2 - B.1 * A.2)

theorem minimum_area_of_triangle
  (A B : ℝ × ℝ)
  (h_focus : parabola_focus)
  (h_on_parabola : on_parabola A B)
  (h_dot : dot_product_condition A B) :
  ∃ C : ℝ, C = 4 * Real.sqrt 2 ∧ area A B = C :=
by
  sorry

end minimum_area_of_triangle_l2099_209949


namespace find_n_l2099_209976

theorem find_n 
  (a : ℝ := 9 / 15)
  (S1 : ℝ := 15 / (1 - a))
  (b : ℝ := (9 + n) / 15)
  (S2 : ℝ := 3 * S1)
  (hS1 : S1 = 37.5)
  (hS2 : S2 = 112.5)
  (hb : b = 13 / 15)
  (hn : 13 = 9 + n) : 
  n = 4 :=
by
  sorry

end find_n_l2099_209976


namespace factorization_correctness_l2099_209974

theorem factorization_correctness :
  (∀ x, x^2 + 2 * x + 1 = (x + 1)^2) ∧
  ¬ (∀ x, x * (x + 1) = x^2 + x) ∧
  ¬ (∀ x y, x^2 + x * y - 3 = x * (x + y) - 3) ∧
  ¬ (∀ x, x^2 + 6 * x + 4 = (x + 3)^2 - 5) :=
by
  sorry

end factorization_correctness_l2099_209974


namespace apples_per_pie_l2099_209964

theorem apples_per_pie (total_apples handed_out_apples pies made_pies remaining_apples : ℕ) 
  (h_initial : total_apples = 86)
  (h_handout : handed_out_apples = 30)
  (h_made_pies : made_pies = 7)
  (h_remaining : remaining_apples = total_apples - handed_out_apples) :
  remaining_apples / made_pies = 8 :=
by
  sorry

end apples_per_pie_l2099_209964


namespace area_of_rectangle_perimeter_of_rectangle_l2099_209992

-- Define the input conditions
variables (AB AC BC : ℕ)
def is_right_triangle (a b c : ℕ) : Prop := a * a + b * b = c * c
def area_rect (l w : ℕ) : ℕ := l * w
def perimeter_rect (l w : ℕ) : ℕ := 2 * (l + w)

-- Given the conditions for the problem
axiom AB_eq_15 : AB = 15
axiom AC_eq_17 : AC = 17
axiom right_triangle : is_right_triangle AB BC AC

-- Prove the area and perimeter of the rectangle
theorem area_of_rectangle : area_rect AB BC = 120 := by sorry

theorem perimeter_of_rectangle : perimeter_rect AB BC = 46 := by sorry

end area_of_rectangle_perimeter_of_rectangle_l2099_209992


namespace intersection_A_B_l2099_209988

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_A_B : A ∩ B = {2} := by
  -- Proof to be filled
  sorry

end intersection_A_B_l2099_209988


namespace apple_cost_calculation_l2099_209928

theorem apple_cost_calculation
    (original_price : ℝ)
    (price_raise : ℝ)
    (amount_per_person : ℝ)
    (num_people : ℝ) :
  original_price = 1.6 →
  price_raise = 0.25 →
  amount_per_person = 2 →
  num_people = 4 →
  (num_people * amount_per_person * (original_price * (1 + price_raise))) = 16 :=
by
  -- insert the mathematical proof steps/cardinality here
  sorry

end apple_cost_calculation_l2099_209928


namespace no_such_abc_l2099_209963

theorem no_such_abc :
  ¬ ∃ (a b c : ℕ+),
    (∃ k1 : ℕ, a ^ 2 * b * c + 2 = k1 ^ 2) ∧
    (∃ k2 : ℕ, b ^ 2 * c * a + 2 = k2 ^ 2) ∧
    (∃ k3 : ℕ, c ^ 2 * a * b + 2 = k3 ^ 2) := 
sorry

end no_such_abc_l2099_209963
