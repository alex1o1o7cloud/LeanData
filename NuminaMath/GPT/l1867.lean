import Mathlib

namespace NUMINAMATH_GPT_fraction_of_time_spent_covering_initial_distance_l1867_186758

variables (D T : ℝ) (h1 : T = ((2 / 3) * D) / 80 + ((1 / 3) * D) / 40)

theorem fraction_of_time_spent_covering_initial_distance (h1 : T = ((2 / 3) * D) / 80 + ((1 / 3) * D) / 40) :
  ((2 / 3) * D / 80) / T = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_time_spent_covering_initial_distance_l1867_186758


namespace NUMINAMATH_GPT_barbara_removed_total_sheets_l1867_186707

theorem barbara_removed_total_sheets :
  let bundles_colored := 3
  let bunches_white := 2
  let heaps_scrap := 5
  let sheets_per_bunch := 4
  let sheets_per_bundle := 2
  let sheets_per_heap := 20
  bundles_colored * sheets_per_bundle + bunches_white * sheets_per_bunch + heaps_scrap * sheets_per_heap = 114 :=
by
  sorry

end NUMINAMATH_GPT_barbara_removed_total_sheets_l1867_186707


namespace NUMINAMATH_GPT_complement_union_M_N_correct_l1867_186754

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the set M
def M : Set ℕ := {1, 3, 5, 7}

-- Define the set N
def N : Set ℕ := {5, 6, 7}

-- Define the union of M and N
def union_M_N : Set ℕ := M ∪ N

-- Define the complement of the union of M and N in U
def complement_union_M_N : Set ℕ := U \ union_M_N

-- Main theorem statement to prove
theorem complement_union_M_N_correct : complement_union_M_N = {2, 4, 8} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_M_N_correct_l1867_186754


namespace NUMINAMATH_GPT_arithmetic_sequence_first_term_l1867_186720

theorem arithmetic_sequence_first_term (d : ℤ) (a_n a_2 a_9 a_11 : ℤ) 
  (h1 : a_2 = 7) 
  (h2 : a_11 = a_9 + 6)
  (h3 : a_11 = a_n + 10 * d)
  (h4 : a_9 = a_n + 8 * d)
  (h5 : a_2 = a_n + d) :
  a_n = 4 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_first_term_l1867_186720


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1867_186779

variable (a b c : ℝ) (α β : ℝ)

theorem quadratic_inequality_solution_set
  (hαβ : α < β)
  (hα_lt_0 : α < 0) 
  (hβ_lt_0 : β < 0)
  (h_sol_set : ∀ x : ℝ, a * x^2 + b * x + c < 0 ↔ (x < α ∨ x > β)) :
  (∀ x : ℝ, c * x^2 - b * x + a > 0 ↔ (-(1 / α) < x ∧ x < -(1 / β))) :=
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1867_186779


namespace NUMINAMATH_GPT_unique_solution_of_functional_equation_l1867_186789

theorem unique_solution_of_functional_equation
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f (x + y)) = f x + y) :
  ∀ x : ℝ, f x = x := 
sorry

end NUMINAMATH_GPT_unique_solution_of_functional_equation_l1867_186789


namespace NUMINAMATH_GPT_grill_burns_fifteen_coals_in_twenty_minutes_l1867_186786

-- Define the problem conditions
def total_coals (bags : ℕ) (coals_per_bag : ℕ) : ℕ :=
  bags * coals_per_bag

def burning_ratio (total_coals : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / total_coals

-- Given conditions
def bags := 3
def coals_per_bag := 60
def total_minutes := 240
def fifteen_coals := 15

-- Problem statement
theorem grill_burns_fifteen_coals_in_twenty_minutes :
  total_minutes / total_coals bags coals_per_bag * fifteen_coals = 20 :=
by
  sorry

end NUMINAMATH_GPT_grill_burns_fifteen_coals_in_twenty_minutes_l1867_186786


namespace NUMINAMATH_GPT_find_x_intercept_of_line_through_points_l1867_186791

-- Definitions based on the conditions
def point1 : ℝ × ℝ := (-1, 1)
def point2 : ℝ × ℝ := (0, 3)

-- Statement: The x-intercept of the line passing through the given points is -3/2
theorem find_x_intercept_of_line_through_points :
  let x1 := point1.1
  let y1 := point1.2
  let x2 := point2.1
  let y2 := point2.2
  ∃ x_intercept : ℝ, x_intercept = -3 / 2 ∧ 
    (∀ x, ∀ y, (x2 - x1) * (y - y1) = (y2 - y1) * (x - x1) → y = 0 → x = x_intercept) :=
by
  sorry

end NUMINAMATH_GPT_find_x_intercept_of_line_through_points_l1867_186791


namespace NUMINAMATH_GPT_wash_time_difference_l1867_186760

def C := 30
def T := 2 * C
def total_time := 135

theorem wash_time_difference :
  ∃ S, C + T + S = total_time ∧ T - S = 15 :=
by
  sorry

end NUMINAMATH_GPT_wash_time_difference_l1867_186760


namespace NUMINAMATH_GPT_inequality_solution_set_l1867_186773

noncomputable def solution_set := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

theorem inequality_solution_set : {x : ℝ | (x - 1) * (3 - x) ≥ 0} = solution_set := by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1867_186773


namespace NUMINAMATH_GPT_min_convex_cover_area_l1867_186725

-- Define the dimensions of the box and the hole
def box_side := 5
def hole_side := 1

-- Define a function to represent the minimum area convex cover
def min_area_convex_cover (box_side hole_side : ℕ) : ℕ :=
  5 -- As given in the problem, the minimum area is concluded to be 5.

-- Theorem to state that the minimum area of the convex cover is 5
theorem min_convex_cover_area : min_area_convex_cover box_side hole_side = 5 :=
by
  -- Proof of the theorem
  sorry

end NUMINAMATH_GPT_min_convex_cover_area_l1867_186725


namespace NUMINAMATH_GPT_union_sets_eq_l1867_186713

-- Definitions of the given sets
def A : Set ℕ := {0, 1}
def B : Set ℕ := {1, 2}

-- The theorem to prove the union of sets A and B equals \{0, 1, 2\}
theorem union_sets_eq : (A ∪ B) = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_union_sets_eq_l1867_186713


namespace NUMINAMATH_GPT_density_is_not_vector_l1867_186706

/-- Conditions definition -/
def is_vector (quantity : String) : Prop :=
quantity = "Buoyancy" ∨ quantity = "Wind speed" ∨ quantity = "Displacement"

/-- Problem statement -/
theorem density_is_not_vector : ¬ is_vector "Density" := 
by 
sorry

end NUMINAMATH_GPT_density_is_not_vector_l1867_186706


namespace NUMINAMATH_GPT_number_of_zeros_of_quadratic_function_l1867_186715

-- Given the quadratic function y = x^2 + x - 1
def quadratic_function (x : ℝ) : ℝ := x^2 + x - 1

-- Prove that the number of zeros of the quadratic function y = x^2 + x - 1 is 2
theorem number_of_zeros_of_quadratic_function : 
  ∃ x1 x2 : ℝ, quadratic_function x1 = 0 ∧ quadratic_function x2 = 0 ∧ x1 ≠ x2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_zeros_of_quadratic_function_l1867_186715


namespace NUMINAMATH_GPT_correct_statement_B_l1867_186752

def flowchart_start_points : Nat := 1
def flowchart_end_points : Bool := True  -- Represents one or multiple end points (True means multiple possible)

def program_flowchart_start_points : Nat := 1
def program_flowchart_end_points : Nat := 1

def structure_chart_start_points : Nat := 1
def structure_chart_end_points : Bool := True  -- Represents one or multiple end points (True means multiple possible)

theorem correct_statement_B :
  (program_flowchart_start_points = 1 ∧ program_flowchart_end_points = 1) :=
by 
  sorry

end NUMINAMATH_GPT_correct_statement_B_l1867_186752


namespace NUMINAMATH_GPT_difference_of_two_numbers_l1867_186716

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end NUMINAMATH_GPT_difference_of_two_numbers_l1867_186716


namespace NUMINAMATH_GPT_angle_ABC_bisector_l1867_186712

theorem angle_ABC_bisector (θ : ℝ) (h : θ / 2 = (1 / 3) * (180 - θ)) : θ = 72 :=
by
  sorry

end NUMINAMATH_GPT_angle_ABC_bisector_l1867_186712


namespace NUMINAMATH_GPT_project_completion_in_16_days_l1867_186793

noncomputable def a_work_rate : ℚ := 1 / 20
noncomputable def b_work_rate : ℚ := 1 / 30
noncomputable def c_work_rate : ℚ := 1 / 40
noncomputable def days_a_works (X: ℚ) : ℚ := X - 10
noncomputable def days_b_works (X: ℚ) : ℚ := X - 5
noncomputable def days_c_works (X: ℚ) : ℚ := X

noncomputable def total_work (X: ℚ) : ℚ :=
  (a_work_rate * days_a_works X) + (b_work_rate * days_b_works X) + (c_work_rate * days_c_works X)

theorem project_completion_in_16_days : total_work 16 = 1 := by
  sorry

end NUMINAMATH_GPT_project_completion_in_16_days_l1867_186793


namespace NUMINAMATH_GPT_min_value_of_fraction_l1867_186759

theorem min_value_of_fraction (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  (1 / (a + 1) + 4 / (b + 1)) = 9 / 4 := 
sorry

end NUMINAMATH_GPT_min_value_of_fraction_l1867_186759


namespace NUMINAMATH_GPT_sector_central_angle_l1867_186770

theorem sector_central_angle (r l α : ℝ) 
  (h1 : 2 * r + l = 6) 
  (h2 : 0.5 * l * r = 2) :
  α = l / r → α = 4 ∨ α = 1 :=
sorry

end NUMINAMATH_GPT_sector_central_angle_l1867_186770


namespace NUMINAMATH_GPT_elder_person_age_l1867_186746

-- Definitions based on conditions
variables (y e : ℕ) 

-- Given conditions
def condition1 : Prop := e = y + 20
def condition2 : Prop := e - 5 = 5 * (y - 5)

-- Theorem stating the required proof problem
theorem elder_person_age (h1 : condition1 y e) (h2 : condition2 y e) : e = 30 :=
by
  sorry

end NUMINAMATH_GPT_elder_person_age_l1867_186746


namespace NUMINAMATH_GPT_hunter_saw_32_frogs_l1867_186756

noncomputable def total_frogs (g1 : ℕ) (g2 : ℕ) (d : ℕ) : ℕ :=
g1 + g2 + d

theorem hunter_saw_32_frogs :
  total_frogs 5 3 (2 * 12) = 32 := by
  sorry

end NUMINAMATH_GPT_hunter_saw_32_frogs_l1867_186756


namespace NUMINAMATH_GPT_number_of_digits_in_expression_l1867_186718

theorem number_of_digits_in_expression : 
  (Nat.digits 10 (2^12 * 5^8)).length = 10 := 
by
  sorry

end NUMINAMATH_GPT_number_of_digits_in_expression_l1867_186718


namespace NUMINAMATH_GPT_program_output_is_10_l1867_186783

def final_value_of_A : ℤ :=
  let A := 2
  let A := A * 2
  let A := A + 6
  A

theorem program_output_is_10 : final_value_of_A = 10 := by
  sorry

end NUMINAMATH_GPT_program_output_is_10_l1867_186783


namespace NUMINAMATH_GPT_gas_pressure_inversely_proportional_l1867_186730

variable {T : Type} [Nonempty T]

theorem gas_pressure_inversely_proportional
  (P : T → ℝ) (V : T → ℝ)
  (h_inv : ∀ t, P t * V t = 24) -- Given that pressure * volume = k where k = 24
  (t₀ t₁ : T)
  (hV₀ : V t₀ = 3) (hP₀ : P t₀ = 8) -- Initial condition: volume = 3 liters, pressure = 8 kPa
  (hV₁ : V t₁ = 6) -- New condition: volume = 6 liters
  : P t₁ = 4 := -- We need to prove that the new pressure is 4 kPa
by 
  sorry

end NUMINAMATH_GPT_gas_pressure_inversely_proportional_l1867_186730


namespace NUMINAMATH_GPT_min_tiles_l1867_186778

theorem min_tiles (x y : ℕ) (h1 : 25 * x + 9 * y = 2014) (h2 : ∀ a b, 25 * a + 9 * b = 2014 -> (a + b) >= (x + y)) : x + y = 94 :=
  sorry

end NUMINAMATH_GPT_min_tiles_l1867_186778


namespace NUMINAMATH_GPT_odd_checkerboard_cannot_be_covered_by_dominoes_l1867_186735

theorem odd_checkerboard_cannot_be_covered_by_dominoes 
    (m n : ℕ) (h : (m * n) % 2 = 1) :
    ¬ ∃ (dominos : Finset (Fin 2 × Fin 2)),
    ∀ {i j : Fin 2}, (i, j) ∈ dominos → 
    ((i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 0)) ∧ 
    dominos.card = (m * n) / 2 := sorry

end NUMINAMATH_GPT_odd_checkerboard_cannot_be_covered_by_dominoes_l1867_186735


namespace NUMINAMATH_GPT_calculate_v3_l1867_186781

def f (x : ℤ) : ℤ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def v0 : ℤ := 2
def v1 (x : ℤ) : ℤ := v0 * x + 5
def v2 (x : ℤ) : ℤ := v1 x * x + 6
def v3 (x : ℤ) : ℤ := v2 x * x + 23

theorem calculate_v3 : v3 (-4) = -49 :=
by
sorry

end NUMINAMATH_GPT_calculate_v3_l1867_186781


namespace NUMINAMATH_GPT_milk_production_l1867_186741

theorem milk_production (y : ℕ) (hcows : y > 0) (hcans : y + 2 > 0) (hdays : y + 3 > 0) :
  let daily_production_per_cow := (y + 2 : ℕ) / (y * (y + 3) : ℕ)
  let total_daily_production := (y + 4 : ℕ) * daily_production_per_cow
  let required_days := (y + 6 : ℕ) / total_daily_production
  required_days = (y * (y + 3) * (y + 6)) / ((y + 2) * (y + 4)) :=
by
  sorry

end NUMINAMATH_GPT_milk_production_l1867_186741


namespace NUMINAMATH_GPT_intersection_M_N_l1867_186732

def M : Set ℝ := {x | x^2 - 2 * x - 3 = 0}
def N : Set ℝ := {x | -4 < x ∧ x ≤ 2}
def intersection : Set ℝ := {-1}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1867_186732


namespace NUMINAMATH_GPT_complement_of_M_l1867_186700

-- Definitions:
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Assertion:
theorem complement_of_M :
  (U \ M) = {x | x ≤ -1} ∪ {x | 2 < x} :=
by sorry

end NUMINAMATH_GPT_complement_of_M_l1867_186700


namespace NUMINAMATH_GPT_complex_pure_imaginary_is_x_eq_2_l1867_186728

theorem complex_pure_imaginary_is_x_eq_2
  (x : ℝ)
  (z : ℂ)
  (h : z = ⟨x^2 - 3 * x + 2, x - 1⟩)
  (pure_imaginary : z.re = 0) :
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_complex_pure_imaginary_is_x_eq_2_l1867_186728


namespace NUMINAMATH_GPT_ratio_of_incomes_l1867_186798

variable {I1 I2 E1 E2 S1 S2 : ℝ}

theorem ratio_of_incomes
  (h1 : I1 = 4000)
  (h2 : E1 / E2 = 3 / 2)
  (h3 : S1 = 1600)
  (h4 : S2 = 1600)
  (h5 : S1 = I1 - E1)
  (h6 : S2 = I2 - E2) :
  I1 / I2 = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_incomes_l1867_186798


namespace NUMINAMATH_GPT_band_member_earnings_l1867_186772

-- Define conditions
def n_people : ℕ := 500
def p_ticket : ℚ := 30
def r_earnings : ℚ := 0.7
def n_members : ℕ := 4

-- Definition of total earnings and share per band member
def total_earnings : ℚ := n_people * p_ticket
def band_share : ℚ := total_earnings * r_earnings
def amount_per_member : ℚ := band_share / n_members

-- Statement to be proved
theorem band_member_earnings : amount_per_member = 2625 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_band_member_earnings_l1867_186772


namespace NUMINAMATH_GPT_intersection_M_N_l1867_186769

open Set

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | x > 0 ∧ x < 2}

theorem intersection_M_N : M ∩ N = {1} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_M_N_l1867_186769


namespace NUMINAMATH_GPT_largest_integer_condition_l1867_186744

theorem largest_integer_condition (x : ℤ) : (x/3 + 3/4 : ℚ) < 7/3 → x ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_condition_l1867_186744


namespace NUMINAMATH_GPT_trajectory_of_C_l1867_186745

-- Definitions of points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-1, 3)

-- Definition of point C as a linear combination of points A and B
def C (α β : ℝ) : ℝ × ℝ := (α * A.1 + β * B.1, α * A.2 + β * B.2)

-- The main theorem statement to prove the equation of the trajectory of point C
theorem trajectory_of_C (x y α β : ℝ)
  (h_cond : α + β = 1)
  (h_C : (x, y) = C α β) : 
  x + 2*y = 5 := 
sorry -- Proof to be skipped

end NUMINAMATH_GPT_trajectory_of_C_l1867_186745


namespace NUMINAMATH_GPT_total_first_tier_college_applicants_l1867_186719

theorem total_first_tier_college_applicants
  (total_students : ℕ)
  (sample_size : ℕ)
  (sample_applicants : ℕ)
  (total_applicants : ℕ) 
  (h1 : total_students = 1000)
  (h2 : sample_size = 150)
  (h3 : sample_applicants = 60)
  : total_applicants = 400 :=
sorry

end NUMINAMATH_GPT_total_first_tier_college_applicants_l1867_186719


namespace NUMINAMATH_GPT_intersection_M_N_eq_set_l1867_186711

universe u

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {y | ∃ x, x ∈ M ∧ y = 2 * x + 1}

-- Prove the intersection M ∩ N = {-1, 1}
theorem intersection_M_N_eq_set : M ∩ N = {-1, 1} :=
by
  simp [Set.ext_iff, M, N]
  sorry

end NUMINAMATH_GPT_intersection_M_N_eq_set_l1867_186711


namespace NUMINAMATH_GPT_invalid_votes_percentage_l1867_186717

theorem invalid_votes_percentage (total_votes : ℕ) (valid_votes_candidate2 : ℕ) (valid_votes_percentage_candidate1 : ℕ) 
  (h_total_votes : total_votes = 7500) 
  (h_valid_votes_candidate2 : valid_votes_candidate2 = 2700)
  (h_valid_votes_percentage_candidate1 : valid_votes_percentage_candidate1 = 55) :
  ((total_votes - (valid_votes_candidate2 * 100 / (100 - valid_votes_percentage_candidate1))) * 100 / total_votes) = 20 :=
by sorry

end NUMINAMATH_GPT_invalid_votes_percentage_l1867_186717


namespace NUMINAMATH_GPT_lines_intersect_l1867_186774

theorem lines_intersect (a b : ℝ) 
  (h₁ : ∃ y : ℝ, 4 = (3/4) * y + a ∧ y = 3)
  (h₂ : ∃ x : ℝ, 3 = (3/4) * x + b ∧ x = 4) :
  a + b = 7/4 :=
sorry

end NUMINAMATH_GPT_lines_intersect_l1867_186774


namespace NUMINAMATH_GPT_total_money_l1867_186721

theorem total_money (n : ℕ) (h1 : n * 3 = 36) :
  let one_rupee := n * 1
  let five_rupee := n * 5
  let ten_rupee := n * 10
  (one_rupee + five_rupee + ten_rupee) = 192 :=
by
  -- Note: The detailed calculations would go here in the proof
  -- Since we don't need to provide the proof, we add sorry to indicate the omitted part
  sorry

end NUMINAMATH_GPT_total_money_l1867_186721


namespace NUMINAMATH_GPT_polynomial_difference_l1867_186734

theorem polynomial_difference (a : ℝ) :
  (6 * a^2 - 5 * a + 3) - (5 * a^2 + 2 * a - 1) = a^2 - 7 * a + 4 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_difference_l1867_186734


namespace NUMINAMATH_GPT_union_of_sets_l1867_186722

def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {2, 3, 6}

theorem union_of_sets : A ∪ B = {1, 2, 3, 5, 6} :=
by sorry

end NUMINAMATH_GPT_union_of_sets_l1867_186722


namespace NUMINAMATH_GPT_max_value_fraction_l1867_186792

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x - 1 = 0

theorem max_value_fraction (a b : ℝ) (H : circle_eq a b) :
  ∃ t : ℝ, -1/2 ≤ t ∧ t ≤ 1/2 ∧ b = t * (a - 3) ∧ t = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_max_value_fraction_l1867_186792


namespace NUMINAMATH_GPT_weighted_avg_sales_increase_l1867_186705

section SalesIncrease

/-- Define the weightages for each category last year. -/
def w_e : ℝ := 0.4
def w_c : ℝ := 0.3
def w_g : ℝ := 0.3

/-- Define the percent increases for each category this year. -/
def p_e : ℝ := 0.15
def p_c : ℝ := 0.25
def p_g : ℝ := 0.35

/-- Prove that the weighted average percent increase in sales this year is 0.24 or 24%. -/
theorem weighted_avg_sales_increase :
  ((w_e * p_e) + (w_c * p_c) + (w_g * p_g)) / (w_e + w_c + w_g) = 0.24 := 
by
  sorry

end SalesIncrease

end NUMINAMATH_GPT_weighted_avg_sales_increase_l1867_186705


namespace NUMINAMATH_GPT_find_A_l1867_186726

theorem find_A (A B : ℕ) (h1 : 15 = 3 * A) (h2 : 15 = 5 * B) : A = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_A_l1867_186726


namespace NUMINAMATH_GPT_toothpick_250_stage_l1867_186782

-- Define the arithmetic sequence for number of toothpicks at each stage
def toothpicks (n : ℕ) : ℕ := 5 + (n - 1) * 4

-- The proof statement for the 250th stage
theorem toothpick_250_stage : toothpicks 250 = 1001 :=
  by
  sorry

end NUMINAMATH_GPT_toothpick_250_stage_l1867_186782


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1867_186799

def A := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1) / Real.log 2}
def B := {x : ℝ | x < 2}

theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1867_186799


namespace NUMINAMATH_GPT_arithmetic_sequence_abs_sum_l1867_186797

theorem arithmetic_sequence_abs_sum :
  ∀ (a : ℕ → ℤ), (∀ n, a (n + 1) - a n = 2) → a 1 = -5 → 
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 18) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_abs_sum_l1867_186797


namespace NUMINAMATH_GPT_opposite_of_five_l1867_186714

theorem opposite_of_five : ∃ y : ℤ, 5 + y = 0 ∧ y = -5 := by
  use -5
  constructor
  . exact rfl
  . sorry

end NUMINAMATH_GPT_opposite_of_five_l1867_186714


namespace NUMINAMATH_GPT_ratio_doctors_lawyers_l1867_186743

theorem ratio_doctors_lawyers (d l : ℕ) (h1 : (45 * d + 60 * l) / (d + l) = 50) (h2 : d + l = 50) : d = 2 * l :=
by
  sorry

end NUMINAMATH_GPT_ratio_doctors_lawyers_l1867_186743


namespace NUMINAMATH_GPT_tank_capacity_l1867_186795

theorem tank_capacity (T : ℝ) (h1 : T * (4 / 5) - T * (5 / 8) = 15) : T = 86 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l1867_186795


namespace NUMINAMATH_GPT_age_of_other_man_l1867_186740

variables (A M : ℝ)

theorem age_of_other_man 
  (avg_age_of_men : ℝ)
  (replaced_man_age : ℝ)
  (avg_age_of_women : ℝ)
  (total_age_6_men : 6 * avg_age_of_men = 6 * (avg_age_of_men + 3) - replaced_man_age - M + 2 * avg_age_of_women) :
  M = 44 :=
by
  sorry

end NUMINAMATH_GPT_age_of_other_man_l1867_186740


namespace NUMINAMATH_GPT_probability_miss_at_least_once_l1867_186704
-- Importing the entirety of Mathlib

-- Defining the conditions and question
variable (P : ℝ) (hP : 0 ≤ P ∧ P ≤ 1)

-- The main statement for the proof problem
theorem probability_miss_at_least_once (P : ℝ) (hP : 0 ≤ P ∧ P ≤ 1) : P ≤ 1 → 0 ≤ P ∧ 1 - P^3 ≥ 0 := 
by
  sorry

end NUMINAMATH_GPT_probability_miss_at_least_once_l1867_186704


namespace NUMINAMATH_GPT_next_ten_winners_each_receive_160_l1867_186723

def total_prize_money : ℕ := 2400

def first_winner_amount : ℕ := total_prize_money / 3

def remaining_amount : ℕ := total_prize_money - first_winner_amount

def each_of_ten_winners_receive : ℕ := remaining_amount / 10

theorem next_ten_winners_each_receive_160 : each_of_ten_winners_receive = 160 := by
  sorry

end NUMINAMATH_GPT_next_ten_winners_each_receive_160_l1867_186723


namespace NUMINAMATH_GPT_smallest_k_l1867_186727

-- Define p as the largest prime number with 2023 digits
def p : ℕ := sorry -- This represents the largest prime number with 2023 digits

-- Define the target k
def k : ℕ := 1

-- The theorem stating that k is the smallest positive integer such that p^2 - k is divisible by 30
theorem smallest_k (p_largest_prime : ∀ m : ℕ, m ≤ p → Nat.Prime m → m = p) 
  (p_digits : 10^2022 ≤ p ∧ p < 10^2023) : 
  ∀ n : ℕ, n > 0 → (p^2 - n) % 30 = 0 → n = k :=
by 
  sorry

end NUMINAMATH_GPT_smallest_k_l1867_186727


namespace NUMINAMATH_GPT_emily_card_sequence_l1867_186762

/--
Emily orders her playing cards continuously in the following sequence:
A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2, 3, ...

Prove that the 58th card in this sequence is 6.
-/
theorem emily_card_sequence :
  (58 % 13 = 6) := by
  -- The modulo operation determines the position of the card in the cycle
  sorry

end NUMINAMATH_GPT_emily_card_sequence_l1867_186762


namespace NUMINAMATH_GPT_gcd_30_45_is_15_l1867_186766

theorem gcd_30_45_is_15 : Nat.gcd 30 45 = 15 := by
  sorry

end NUMINAMATH_GPT_gcd_30_45_is_15_l1867_186766


namespace NUMINAMATH_GPT_truth_prob_l1867_186763

-- Define the probabilities
def prob_A := 0.80
def prob_B := 0.60
def prob_C := 0.75

-- The problem statement
theorem truth_prob :
  prob_A * prob_B * prob_C = 0.27 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_truth_prob_l1867_186763


namespace NUMINAMATH_GPT_price_of_basketball_l1867_186751

-- Problem definitions based on conditions
def price_of_soccer_ball (x : ℝ) : Prop :=
  let price_of_basketball := 2 * x
  x + price_of_basketball = 186

theorem price_of_basketball (x : ℝ) (h : price_of_soccer_ball x) : 2 * x = 124 :=
by
  sorry

end NUMINAMATH_GPT_price_of_basketball_l1867_186751


namespace NUMINAMATH_GPT_combinations_of_eight_choose_three_is_fifty_six_l1867_186755

theorem combinations_of_eight_choose_three_is_fifty_six :
  (Nat.choose 8 3) = 56 :=
by
  sorry

end NUMINAMATH_GPT_combinations_of_eight_choose_three_is_fifty_six_l1867_186755


namespace NUMINAMATH_GPT_range_of_a_l1867_186702

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 :=
by {
  sorry -- Proof is not required as per instructions.
}

end NUMINAMATH_GPT_range_of_a_l1867_186702


namespace NUMINAMATH_GPT_contrapositive_l1867_186757

theorem contrapositive (p q : Prop) : (p → q) → (¬q → ¬p) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_l1867_186757


namespace NUMINAMATH_GPT_line_does_not_pass_through_third_quadrant_l1867_186777

-- Define the Cartesian equation of the line
def line_eq (x y : ℝ) : Prop :=
  x + 2 * y = 1

-- Define the property that a point (x, y) belongs to the third quadrant
def in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

-- State the theorem
theorem line_does_not_pass_through_third_quadrant :
  ¬ ∃ (x y : ℝ), line_eq x y ∧ in_third_quadrant x y :=
by
  sorry

end NUMINAMATH_GPT_line_does_not_pass_through_third_quadrant_l1867_186777


namespace NUMINAMATH_GPT_perpendicular_line_slope_l1867_186776

theorem perpendicular_line_slope (m : ℝ) 
  (h1 : ∀ x y : ℝ, x - 2 * y + 5 = 0 → x = 2 * y - 5)
  (h2 : ∀ x y : ℝ, 2 * x + m * y - 6 = 0 → y = - (2 / m) * x + 6 / m)
  (h3 : (1 / 2 : ℝ) * - (2 / m) = -1) : m = 1 :=
sorry

end NUMINAMATH_GPT_perpendicular_line_slope_l1867_186776


namespace NUMINAMATH_GPT_find_a_l1867_186787

theorem find_a (a : ℝ) (x₁ x₂ : ℝ) (h1 : a > 0) (h2 : x₁ = -2 * a) (h3 : x₂ = 4 * a) (h4 : x₂ - x₁ = 15) : a = 5 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_l1867_186787


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l1867_186737

def is_isosceles (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ c = a)

theorem isosceles_triangle_base_length
  (x y : ℝ)
  (h1 : 2 * x + 2 * y = 16)
  (h2 : 4^2 + y^2 = x^2)
  (h3 : is_isosceles x x (2 * y) ) :
  2 * y = 6 := 
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l1867_186737


namespace NUMINAMATH_GPT_infection_equation_correct_l1867_186731

theorem infection_equation_correct (x : ℝ) :
  1 + x + x * (x + 1) = 196 :=
sorry

end NUMINAMATH_GPT_infection_equation_correct_l1867_186731


namespace NUMINAMATH_GPT_planeThroughPointAndLine_l1867_186753

theorem planeThroughPointAndLine :
  ∃ A B C D : ℤ, (A = -3 ∧ B = -4 ∧ C = -4 ∧ D = 14) ∧ 
  (∀ x y z : ℝ, x = 2 ∧ y = -3 ∧ z = 5 ∨ (∃ t : ℝ, x = 4 * t + 2 ∧ y = -5 * t - 1 ∧ z = 2 * t + 3) → A * x + B * y + C * z + D = 0) :=
sorry

end NUMINAMATH_GPT_planeThroughPointAndLine_l1867_186753


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1867_186738

def U : Set ℕ := {1, 2, 3, 4}

def satisfies_inequality (x : ℕ) : Prop := x^2 - 5 * x + 4 < 0

def A : Set ℕ := {x | satisfies_inequality x}

theorem complement_of_A_in_U : U \ A = {1, 4} :=
by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1867_186738


namespace NUMINAMATH_GPT_runway_trip_time_l1867_186765

-- Define the conditions
def num_models := 6
def num_bathing_suit_outfits := 2
def num_evening_wear_outfits := 3
def total_time_minutes := 60

-- Calculate the total number of outfits per model
def total_outfits_per_model := num_bathing_suit_outfits + num_evening_wear_outfits

-- Calculate the total number of runway trips
def total_runway_trips := num_models * total_outfits_per_model

-- State the goal: Time per runway trip
def time_per_runway_trip := total_time_minutes / total_runway_trips

theorem runway_trip_time : time_per_runway_trip = 2 := by
  sorry

end NUMINAMATH_GPT_runway_trip_time_l1867_186765


namespace NUMINAMATH_GPT_correct_system_of_equations_l1867_186775

-- Definitions based on the conditions
def rope_exceeds (x y : ℝ) : Prop := x - y = 4.5
def rope_half_falls_short (x y : ℝ) : Prop := (1/2) * x + 1 = y

-- Proof statement
theorem correct_system_of_equations (x y : ℝ) :
  rope_exceeds x y → rope_half_falls_short x y → 
  (x - y = 4.5 ∧ (1/2 * x + 1 = y)) := 
by 
  sorry

end NUMINAMATH_GPT_correct_system_of_equations_l1867_186775


namespace NUMINAMATH_GPT_overtime_pay_rate_ratio_l1867_186736

noncomputable def regular_pay_rate : ℕ := 3
noncomputable def regular_hours : ℕ := 40
noncomputable def total_pay : ℕ := 180
noncomputable def overtime_hours : ℕ := 10

theorem overtime_pay_rate_ratio : 
  (total_pay - (regular_hours * regular_pay_rate)) / overtime_hours / regular_pay_rate = 2 := by
  sorry

end NUMINAMATH_GPT_overtime_pay_rate_ratio_l1867_186736


namespace NUMINAMATH_GPT_find_a_l1867_186761

variable (a x y : ℝ)

theorem find_a (h1 : x / (2 * y) = 3 / 2) (h2 : (a * x + 6 * y) / (x - 2 * y) = 27) : a = 7 :=
sorry

end NUMINAMATH_GPT_find_a_l1867_186761


namespace NUMINAMATH_GPT_range_of_c_extreme_values_l1867_186796

noncomputable def f (c x : ℝ) : ℝ := x^3 - 2 * c * x^2 + x

theorem range_of_c_extreme_values 
  (c : ℝ) 
  (h : ∃ a b : ℝ, a ≠ b ∧ (3 * a^2 - 4 * c * a + 1 = 0) ∧ (3 * b^2 - 4 * c * b + 1 = 0)) :
  c < - (Real.sqrt 3 / 2) ∨ c > (Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_GPT_range_of_c_extreme_values_l1867_186796


namespace NUMINAMATH_GPT_abc_not_all_positive_l1867_186729

theorem abc_not_all_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ac > 0) (h3 : abc > 0) : 
  ¬(a > 0 ∧ b > 0 ∧ c > 0) ↔ (a ≤ 0 ∨ b ≤ 0 ∨ c ≤ 0) := 
by 
sorry

end NUMINAMATH_GPT_abc_not_all_positive_l1867_186729


namespace NUMINAMATH_GPT_problem_statement_l1867_186739

def A : ℕ := 9 * 10 * 10 * 5
def B : ℕ := 9 * 10 * 10 * 2 / 3

theorem problem_statement : A + B = 5100 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1867_186739


namespace NUMINAMATH_GPT_initial_average_weight_l1867_186710

theorem initial_average_weight (a b c d e : ℝ) (A : ℝ) 
    (h1 : (a + b + c) / 3 = A) 
    (h2 : (a + b + c + d) / 4 = 80) 
    (h3 : e = d + 3) 
    (h4 : (b + c + d + e) / 4 = 79) 
    (h5 : a = 75) : A = 84 :=
sorry

end NUMINAMATH_GPT_initial_average_weight_l1867_186710


namespace NUMINAMATH_GPT_smallest_square_area_l1867_186784

theorem smallest_square_area (a b c d : ℕ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 4) (h4 : d = 5) :
  ∃ s : ℕ, s * s = 64 ∧ (a + c <= s ∧ max b d <= s) ∨ (max a c <= s ∧ b + d <= s) :=
sorry

end NUMINAMATH_GPT_smallest_square_area_l1867_186784


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l1867_186768

theorem sum_of_squares_of_roots (a b : ℝ) (x₁ x₂ : ℝ)
  (h₁ : x₁^2 - (3 * a + b) * x₁ + 2 * a^2 + 3 * a * b - 2 * b^2 = 0)
  (h₂ : x₂^2 - (3 * a + b) * x₂ + 2 * a^2 + 3 * a * b - 2 * b^2 = 0) :
  x₁^2 + x₂^2 = 5 * (a^2 + b^2) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l1867_186768


namespace NUMINAMATH_GPT_sqrt_four_eq_plus_minus_two_l1867_186748

theorem sqrt_four_eq_plus_minus_two : ∃ y : ℤ, y^2 = 4 ∧ (y = 2 ∨ y = -2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sqrt_four_eq_plus_minus_two_l1867_186748


namespace NUMINAMATH_GPT_textile_firm_looms_l1867_186780

theorem textile_firm_looms
  (sales_val : ℝ)
  (manu_exp : ℝ)
  (estab_charges : ℝ)
  (profit_decrease : ℝ)
  (L : ℝ)
  (h_sales : sales_val = 500000)
  (h_manu_exp : manu_exp = 150000)
  (h_estab_charges : estab_charges = 75000)
  (h_profit_decrease : profit_decrease = 7000)
  (hem_equal_contrib : ∀ l : ℝ, l > 0 →
    (l = sales_val / (sales_val / L) - manu_exp / (manu_exp / L)))
  : L = 50 := 
by
  sorry

end NUMINAMATH_GPT_textile_firm_looms_l1867_186780


namespace NUMINAMATH_GPT_solve_for_y_l1867_186771

-- Define the condition
def condition (y : ℤ) : Prop := 7 - y = 13

-- Prove that if the condition is met, then y = -6
theorem solve_for_y (y : ℤ) (h : condition y) : y = -6 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_y_l1867_186771


namespace NUMINAMATH_GPT_percentage_increase_l1867_186703

theorem percentage_increase (original final : ℝ) (h1 : original = 90) (h2 : final = 135) : ((final - original) / original) * 100 = 50 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1867_186703


namespace NUMINAMATH_GPT_sum_of_cube_faces_l1867_186750

-- Define the cube numbers as consecutive integers starting from 15.
def cube_faces (faces : List ℕ) : Prop :=
  faces = [15, 16, 17, 18, 19, 20]

-- Define the condition that the sum of numbers on opposite faces is the same.
def opposite_faces_condition (pairs : List (ℕ × ℕ)) : Prop :=
  ∀ (p : ℕ × ℕ) (hp : p ∈ pairs), (p.1 + p.2) = 35

theorem sum_of_cube_faces : ∃ faces : List ℕ, cube_faces faces ∧ (∃ pairs : List (ℕ × ℕ), opposite_faces_condition pairs ∧ faces.sum = 105) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cube_faces_l1867_186750


namespace NUMINAMATH_GPT_projection_of_vector_a_on_b_l1867_186747

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / norm_b

theorem projection_of_vector_a_on_b
  (a b : ℝ × ℝ) 
  (ha : Real.sqrt (a.1^2 + a.2^2) = 1)
  (hb : Real.sqrt (b.1^2 + b.2^2) = 2)
  (theta : ℝ)
  (h_theta : theta = Real.pi * (5/6)) -- 150 degrees in radians
  (h_cos_theta : Real.cos theta = -(Real.sqrt 3 / 2)) :
  vector_projection a b = -Real.sqrt 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_projection_of_vector_a_on_b_l1867_186747


namespace NUMINAMATH_GPT_augmented_matrix_solution_l1867_186701

theorem augmented_matrix_solution (m n : ℝ) (x y : ℝ)
  (h1 : m * x = 6) (h2 : 3 * y = n) (hx : x = -3) (hy : y = 4) :
  m + n = 10 :=
by
  sorry

end NUMINAMATH_GPT_augmented_matrix_solution_l1867_186701


namespace NUMINAMATH_GPT_square_not_covered_by_circles_l1867_186767

noncomputable def area_uncovered_by_circles : Real :=
  let side_length := 2
  let square_area := (side_length^2 : Real)
  let radius := 1
  let circle_area := Real.pi * radius^2
  let quarter_circle_area := circle_area / 4
  let total_circles_area := 4 * quarter_circle_area
  square_area - total_circles_area

theorem square_not_covered_by_circles :
  area_uncovered_by_circles = 4 - Real.pi := sorry

end NUMINAMATH_GPT_square_not_covered_by_circles_l1867_186767


namespace NUMINAMATH_GPT_initial_caterpillars_l1867_186742

theorem initial_caterpillars (C : ℕ) 
    (hatch_eggs : C + 4 - 8 = 10) : C = 14 :=
by
  sorry

end NUMINAMATH_GPT_initial_caterpillars_l1867_186742


namespace NUMINAMATH_GPT_find_triples_l1867_186794

def is_solution (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

theorem find_triples :
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ is_solution a b c :=
sorry

end NUMINAMATH_GPT_find_triples_l1867_186794


namespace NUMINAMATH_GPT_hockey_league_games_l1867_186709

theorem hockey_league_games (n : ℕ) (k : ℕ) (h_n : n = 25) (h_k : k = 15) : 
  (n * (n - 1) / 2) * k = 4500 := by
  sorry

end NUMINAMATH_GPT_hockey_league_games_l1867_186709


namespace NUMINAMATH_GPT_gcd_of_g_and_y_l1867_186764

noncomputable def g (y : ℕ) := (3 * y + 5) * (8 * y + 3) * (16 * y + 9) * (y + 16)

theorem gcd_of_g_and_y (y : ℕ) (hy : y % 46896 = 0) : Nat.gcd (g y) y = 2160 :=
by
  -- Proof to be written here
  sorry

end NUMINAMATH_GPT_gcd_of_g_and_y_l1867_186764


namespace NUMINAMATH_GPT_rita_money_left_l1867_186749

theorem rita_money_left :
  let initial_amount : ℝ := 400
  let cost_short_dresses : ℝ := 5 * (20 - 0.1 * 20)
  let cost_pants : ℝ := 2 * 15
  let cost_jackets : ℝ := 2 * (30 - 0.15 * 30) + 2 * 30
  let cost_skirts : ℝ := 2 * 18 * 0.8
  let cost_tshirts : ℝ := 2 * 8
  let cost_transportation : ℝ := 5
  let total_spent : ℝ := cost_short_dresses + cost_pants + cost_jackets + cost_skirts + cost_tshirts + cost_transportation
  let money_left : ℝ := initial_amount - total_spent
  money_left = 119.2 :=
by 
  sorry

end NUMINAMATH_GPT_rita_money_left_l1867_186749


namespace NUMINAMATH_GPT_ball_bounce_height_l1867_186790

theorem ball_bounce_height :
  ∃ k : ℕ, (500 * (2 / 3:ℝ)^k < 10) ∧ (∀ m : ℕ, m < k → ¬(500 * (2 / 3:ℝ)^m < 10)) :=
sorry

end NUMINAMATH_GPT_ball_bounce_height_l1867_186790


namespace NUMINAMATH_GPT_proportion_of_ones_l1867_186785

theorem proportion_of_ones (m n : ℕ) (h : Nat.gcd m n = 1) : 
  m + n = 275 :=
  sorry

end NUMINAMATH_GPT_proportion_of_ones_l1867_186785


namespace NUMINAMATH_GPT_perimeter_of_triangle_l1867_186788

noncomputable def ellipse_perimeter (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1) : ℝ :=
  let a := 2
  let c := 1
  2 * a + 2 * c

theorem perimeter_of_triangle (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1) :
  ellipse_perimeter x y h = 6 :=
by 
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_l1867_186788


namespace NUMINAMATH_GPT_keiko_speed_calc_l1867_186724

noncomputable def keiko_speed (r : ℝ) (time_diff : ℝ) : ℝ :=
  let circumference_diff := 2 * Real.pi * 8
  circumference_diff / time_diff

theorem keiko_speed_calc (r : ℝ) (time_diff : ℝ) :
  keiko_speed r 48 = Real.pi / 3 := by
  sorry

end NUMINAMATH_GPT_keiko_speed_calc_l1867_186724


namespace NUMINAMATH_GPT_sturdy_square_impossible_l1867_186733

def size : ℕ := 6
def dominos_used : ℕ := 18
def cells_per_domino : ℕ := 2
def total_cells : ℕ := size * size
def dividing_lines : ℕ := 10

def is_sturdy_square (grid_size : ℕ) (domino_count : ℕ) : Prop :=
  grid_size * grid_size = domino_count * cells_per_domino ∧ 
  ∀ line : ℕ, line < dividing_lines → ∃ domino : ℕ, domino < domino_count

theorem sturdy_square_impossible 
    (grid_size : ℕ) (domino_count : ℕ)
    (h1 : grid_size = size) (h2 : domino_count = dominos_used)
    (h3 : cells_per_domino = 2) (h4 : dividing_lines = 10) : 
  ¬ is_sturdy_square grid_size domino_count :=
by
  cases h1
  cases h2
  cases h3
  cases h4
  sorry

end NUMINAMATH_GPT_sturdy_square_impossible_l1867_186733


namespace NUMINAMATH_GPT_solve_expression_l1867_186708

theorem solve_expression :
  2^3 + 2 * 5 - 3 + 6 = 21 :=
by
  sorry

end NUMINAMATH_GPT_solve_expression_l1867_186708
