import Mathlib

namespace NUMINAMATH_GPT_solve_for_a_l2065_206516

theorem solve_for_a (a : ℝ) (h : ∃ x, x = 2 ∧ a * x - 4 * (x - a) = 1) : a = 3 / 2 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l2065_206516


namespace NUMINAMATH_GPT_opposite_of_fraction_l2065_206554

theorem opposite_of_fraction : - (11 / 2022 : ℚ) = -(11 / 2022) := 
by
  sorry

end NUMINAMATH_GPT_opposite_of_fraction_l2065_206554


namespace NUMINAMATH_GPT_linear_function_through_origin_l2065_206599

theorem linear_function_through_origin (m : ℝ) :
  (∀ x y : ℝ, (y = (m - 1) * x + m ^ 2 - 1) → (x = 0 ∧ y = 0) → m = -1) :=
sorry

end NUMINAMATH_GPT_linear_function_through_origin_l2065_206599


namespace NUMINAMATH_GPT_element_of_sequence_l2065_206572

/-
Proving that 63 is an element of the sequence defined by aₙ = n² + 2n.
-/
theorem element_of_sequence (n : ℕ) (h : 63 = n^2 + 2 * n) : ∃ n : ℕ, 63 = n^2 + 2 * n :=
by
  sorry

end NUMINAMATH_GPT_element_of_sequence_l2065_206572


namespace NUMINAMATH_GPT_faces_painted_morning_l2065_206508

def faces_of_cuboid : ℕ := 6
def faces_painted_evening : ℕ := 3

theorem faces_painted_morning : faces_of_cuboid - faces_painted_evening = 3 := 
by 
  sorry

end NUMINAMATH_GPT_faces_painted_morning_l2065_206508


namespace NUMINAMATH_GPT_proof_equivalence_l2065_206522

noncomputable def compute_expression (N : ℕ) (M : ℕ) : ℚ :=
  ((N - 3)^3 + (N - 2)^3 + (N - 1)^3 + N^3 + (N + 1)^3 + (N + 2)^3 + (N + 3)^3) /
  ((M - 3) * (M - 2) + (M - 1) * M + M * (M + 1) + (M + 2) * (M + 3))

theorem proof_equivalence:
  let N := 65536
  let M := 32768
  compute_expression N M = 229376 := 
  by
    sorry

end NUMINAMATH_GPT_proof_equivalence_l2065_206522


namespace NUMINAMATH_GPT_find_a7_l2065_206575

-- Definitions based on given conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n k : ℕ, a (n + k) = a n + k * (a 1 - a 0)

-- Given condition in Lean statement
def sequence_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 11 = 22

-- Proof problem
theorem find_a7 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) (h2 : sequence_condition a) : a 7 = 11 := 
  sorry

end NUMINAMATH_GPT_find_a7_l2065_206575


namespace NUMINAMATH_GPT_f_at_2018_l2065_206505

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_periodic : ∀ x : ℝ, f (x + 6) = f x
axiom f_at_4 : f 4 = 5

theorem f_at_2018 : f 2018 = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_f_at_2018_l2065_206505


namespace NUMINAMATH_GPT_cubic_sum_l2065_206507

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 := by
  -- Using the provided conditions x + y = 5 and x^2 + y^2 = 13
  sorry

end NUMINAMATH_GPT_cubic_sum_l2065_206507


namespace NUMINAMATH_GPT_length_of_train_l2065_206551

theorem length_of_train
  (T_platform : ℕ)
  (T_pole : ℕ)
  (L_platform : ℕ)
  (h1: T_platform = 39)
  (h2: T_pole = 18)
  (h3: L_platform = 350)
  (L : ℕ)
  (h4 : 39 * L = 18 * (L + 350)) :
  L = 300 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l2065_206551


namespace NUMINAMATH_GPT_cos_alpha_add_pi_over_4_l2065_206547

theorem cos_alpha_add_pi_over_4 (x y r : ℝ) (α : ℝ) (h1 : P = (3, -4)) (h2 : r = Real.sqrt (x^2 + y^2)) (h3 : x / r = Real.cos α) (h4 : y / r = Real.sin α) :
  Real.cos (α + Real.pi / 4) = (7 * Real.sqrt 2) / 10 := by
  sorry

end NUMINAMATH_GPT_cos_alpha_add_pi_over_4_l2065_206547


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2065_206517

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

theorem intersection_of_A_and_B : (A ∩ B) = {x | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2065_206517


namespace NUMINAMATH_GPT_angle_sum_at_point_l2065_206595

theorem angle_sum_at_point (x : ℝ) (h : 170 + 3 * x = 360) : x = 190 / 3 :=
by
  sorry

end NUMINAMATH_GPT_angle_sum_at_point_l2065_206595


namespace NUMINAMATH_GPT_coordinates_P_l2065_206502

theorem coordinates_P 
  (P1 P2 P : ℝ × ℝ)
  (hP1 : P1 = (2, -1))
  (hP2 : P2 = (0, 5))
  (h_ext_line : ∃ t : ℝ, P = (P1.1 + t * (P2.1 - P1.1), P1.2 + t * (P2.2 - P1.2)) ∧ t ≠ 1)
  (h_distance : dist P1 P = 2 * dist P P2) :
  P = (-2, 11) := 
by
  sorry

end NUMINAMATH_GPT_coordinates_P_l2065_206502


namespace NUMINAMATH_GPT_quadratic_properties_l2065_206512

open Real

noncomputable section

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Vertex form of the quadratic
def vertexForm (x : ℝ) : ℝ := (x - 2)^2 - 1

-- Axis of symmetry
def axisOfSymmetry : ℝ := 2

-- Vertex of the quadratic
def vertex : ℝ × ℝ := (2, -1)

-- Minimum value of the quadratic
def minimumValue : ℝ := -1

-- Interval where the function decreases
def decreasingInterval (x : ℝ) : Prop := -1 ≤ x ∧ x < 2

-- Range of y in the interval -1 <= x < 3
def rangeOfY (y : ℝ) : Prop := -1 ≤ y ∧ y ≤ 8

-- Main statement
theorem quadratic_properties :
  (∀ x, quadratic x = vertexForm x) ∧
  (∃ x, axisOfSymmetry = x) ∧
  (∃ v, vertex = v) ∧
  (minimumValue = -1) ∧
  (∀ x, -1 ≤ x ∧ x < 2 → quadratic x > quadratic (x + 1)) ∧
  (∀ y, (∃ x, -1 ≤ x ∧ x < 3 ∧ y = quadratic x) → rangeOfY y) :=
sorry

end NUMINAMATH_GPT_quadratic_properties_l2065_206512


namespace NUMINAMATH_GPT_numerical_puzzle_l2065_206556

noncomputable def THETA (T : ℕ) (A : ℕ) : ℕ := 1000 * T + 100 * T + 10 * T + A
noncomputable def BETA (B : ℕ) (T : ℕ) (A : ℕ) : ℕ := 1000 * B + 100 * T + 10 * T + A
noncomputable def GAMMA (Γ : ℕ) (E : ℕ) (M : ℕ) (A : ℕ) : ℕ := 10000 * Γ + 1000 * E + 100 * M + 10 * M + A

theorem numerical_puzzle
  (T : ℕ) (B : ℕ) (E : ℕ) (M : ℕ) (Γ : ℕ) (A : ℕ)
  (h1 : A = 0)
  (h2 : Γ = 1)
  (h3 : T + T = M)
  (h4 : 2 * E = M)
  (h5 : T ≠ B)
  (h6 : B ≠ E)
  (h7 : E ≠ M)
  (h8 : M ≠ Γ)
  (h9 : Γ ≠ T)
  (h10 : Γ ≠ B)
  (h11 : THETA T A + BETA B T A = GAMMA Γ E M A) :
  THETA 4 0 + BETA 5 4 0 = GAMMA 1 9 8 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_numerical_puzzle_l2065_206556


namespace NUMINAMATH_GPT_geometric_seq_a3_equals_3_l2065_206582

variable {a : ℕ → ℝ}
variable (h_geometric : ∀ m n p q, m + n = p + q → a m * a n = a p * a q)
variable (h_pos : ∀ n, n > 0 → a n > 0)
variable (h_cond : a 2 * a 4 = 9)

theorem geometric_seq_a3_equals_3 : a 3 = 3 := by
  sorry

end NUMINAMATH_GPT_geometric_seq_a3_equals_3_l2065_206582


namespace NUMINAMATH_GPT_numberOfBoys_is_50_l2065_206524

-- Define the number of boys and the conditions given.
def numberOfBoys (B G : ℕ) : Prop :=
  B / G = 5 / 13 ∧ G = B + 80

-- The theorem that we need to prove.
theorem numberOfBoys_is_50 (B G : ℕ) (h : numberOfBoys B G) : B = 50 :=
  sorry

end NUMINAMATH_GPT_numberOfBoys_is_50_l2065_206524


namespace NUMINAMATH_GPT_find_length_of_shop_l2065_206580

noncomputable def length_of_shop (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ) : ℕ :=
  (monthly_rent * 12) / annual_rent_per_sqft / width

theorem find_length_of_shop
  (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ)
  (h_monthly_rent : monthly_rent = 3600)
  (h_width : width = 20)
  (h_annual_rent_per_sqft : annual_rent_per_sqft = 120) 
  : length_of_shop monthly_rent width annual_rent_per_sqft = 18 := 
sorry

end NUMINAMATH_GPT_find_length_of_shop_l2065_206580


namespace NUMINAMATH_GPT_harry_fish_count_l2065_206523

theorem harry_fish_count
  (sam_fish : ℕ) (joe_fish : ℕ) (harry_fish : ℕ)
  (h1 : sam_fish = 7)
  (h2 : joe_fish = 8 * sam_fish)
  (h3 : harry_fish = 4 * joe_fish) :
  harry_fish = 224 :=
by
  sorry

end NUMINAMATH_GPT_harry_fish_count_l2065_206523


namespace NUMINAMATH_GPT_regular_tetrahedron_of_angle_l2065_206581

-- Definition and condition from the problem
def angle_between_diagonals (shape : Type _) (adj_sides_diag_angle : ℝ) : Prop :=
  adj_sides_diag_angle = 60

-- Theorem stating the problem in Lean 4
theorem regular_tetrahedron_of_angle (shape : Type _) (adj_sides_diag_angle : ℝ) 
  (h : angle_between_diagonals shape adj_sides_diag_angle) : 
  shape = regular_tetrahedron :=
sorry

end NUMINAMATH_GPT_regular_tetrahedron_of_angle_l2065_206581


namespace NUMINAMATH_GPT_fish_tagging_problem_l2065_206598

theorem fish_tagging_problem
  (N : ℕ) (T : ℕ)
  (h1 : N = 1250)
  (h2 : T = N / 25) :
  T = 50 :=
sorry

end NUMINAMATH_GPT_fish_tagging_problem_l2065_206598


namespace NUMINAMATH_GPT_simplify_expression_eq_l2065_206573

theorem simplify_expression_eq (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2) / x) * ((y^2 + 2) / y) + ((x^2 - 2) / y) * ((y^2 - 2) / x) = 2 * x * y + 8 / (x * y) :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_eq_l2065_206573


namespace NUMINAMATH_GPT_algebra_expression_bound_l2065_206552

theorem algebra_expression_bound (x y m : ℝ) 
  (h1 : x + y + m = 6) 
  (h2 : 3 * x - y + m = 4) : 
  (-2 * x * y + 1) ≤ 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_algebra_expression_bound_l2065_206552


namespace NUMINAMATH_GPT_unique_two_digit_number_l2065_206561

theorem unique_two_digit_number (n : ℕ) (h1 : 10 ≤ n) (h2 : n ≤ 99) : 
  (13 * n) % 100 = 42 → n = 34 :=
by
  sorry

end NUMINAMATH_GPT_unique_two_digit_number_l2065_206561


namespace NUMINAMATH_GPT_chord_length_of_intersection_l2065_206553

theorem chord_length_of_intersection 
  (A B C : ℝ) (x0 y0 r : ℝ)
  (line_eq : A * x0 + B * y0 + C = 0)
  (circle_eq : (x0 - 1)^2 + (y0 - 3)^2 = r^2) 
  (A_line : A = 4) (B_line : B = -3) (C_line : C = 0) 
  (x0_center : x0 = 1) (y0_center : y0 = 3) (r_circle : r^2 = 10) :
  2 * (Real.sqrt (r^2 - ((A * x0 + B * y0 + C) / (Real.sqrt (A^2 + B^2)))^2)) = 6 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_of_intersection_l2065_206553


namespace NUMINAMATH_GPT_line_through_P_with_opposite_sign_intercepts_l2065_206534

theorem line_through_P_with_opposite_sign_intercepts 
  (P : ℝ × ℝ) (hP : P = (3, -2)) 
  (h : ∀ (A B : ℝ), A ≠ 0 → B ≠ 0 → A * B < 0) : 
  (∀ (x y : ℝ), (x = 5 ∧ y = -5) → (5 * x - 5 * y - 25 = 0)) ∨ (∀ (x y : ℝ), (3 * y = -2) → (y = - (2 / 3) * x)) :=
sorry

end NUMINAMATH_GPT_line_through_P_with_opposite_sign_intercepts_l2065_206534


namespace NUMINAMATH_GPT_value_of_a_l2065_206565

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := (2 : ℝ) * x - y - 1 = 0

def line2 (x y a : ℝ) : Prop := (2 : ℝ) * x + (a + 1) * y + 2 = 0

-- Define the condition for parallel lines
def parallel_lines (a : ℝ) : Prop :=
  ∀ x y : ℝ, (line1 x y) → (line2 x y a)

-- The theorem to be proved
theorem value_of_a (a : ℝ) : parallel_lines a → a = -2 :=
sorry

end NUMINAMATH_GPT_value_of_a_l2065_206565


namespace NUMINAMATH_GPT_common_difference_l2065_206560

variable {a : ℕ → ℤ} -- Define the arithmetic sequence

theorem common_difference (h : a 2015 = a 2013 + 6) : 
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 3 := 
by
  use 3
  sorry

end NUMINAMATH_GPT_common_difference_l2065_206560


namespace NUMINAMATH_GPT_sufficient_budget_for_kvass_l2065_206549

variables (x y : ℝ)

theorem sufficient_budget_for_kvass (h1 : x + y = 1) (h2 : 0.6 * x + 1.2 * y = 1) : 
  3 * y ≥ 1.44 * y :=
by
  sorry

end NUMINAMATH_GPT_sufficient_budget_for_kvass_l2065_206549


namespace NUMINAMATH_GPT_alcohol_concentration_l2065_206579

theorem alcohol_concentration (x : ℝ) (initial_volume : ℝ) (initial_concentration : ℝ) (target_concentration : ℝ) :
  initial_volume = 6 →
  initial_concentration = 0.35 →
  target_concentration = 0.50 →
  (2.1 + x) / (6 + x) = target_concentration →
  x = 1.8 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_alcohol_concentration_l2065_206579


namespace NUMINAMATH_GPT_total_legs_l2065_206532

def total_heads : ℕ := 16
def num_cats : ℕ := 7
def cat_legs : ℕ := 4
def captain_legs : ℕ := 1
def human_legs : ℕ := 2

theorem total_legs : (num_cats * cat_legs + (total_heads - num_cats) * human_legs - human_legs + captain_legs) = 45 :=
by 
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_total_legs_l2065_206532


namespace NUMINAMATH_GPT_largest_lcm_l2065_206584

theorem largest_lcm :
  max (max (max (max (max (Nat.lcm 12 2) (Nat.lcm 12 4)) 
                    (Nat.lcm 12 6)) 
                 (Nat.lcm 12 8)) 
            (Nat.lcm 12 10)) 
      (Nat.lcm 12 12) = 60 :=
by sorry

end NUMINAMATH_GPT_largest_lcm_l2065_206584


namespace NUMINAMATH_GPT_Jason_age_l2065_206528

theorem Jason_age : ∃ J K : ℕ, (J = 7 * K) ∧ (J + 4 = 3 * (2 * (K + 2))) ∧ (J = 56) :=
by
  sorry

end NUMINAMATH_GPT_Jason_age_l2065_206528


namespace NUMINAMATH_GPT_ratio_of_numbers_l2065_206567

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l2065_206567


namespace NUMINAMATH_GPT_range_of_m_condition_l2065_206591

theorem range_of_m_condition (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ * x₁ - 2 * m * x₁ + m - 3 = 0) 
  (h₂ : x₂ * x₂ - 2 * m * x₂ + m - 3 = 0)
  (hx₁ : x₁ > -1 ∧ x₁ < 0)
  (hx₂ : x₂ > 3) :
  m > 6 / 5 ∧ m < 3 :=
sorry

end NUMINAMATH_GPT_range_of_m_condition_l2065_206591


namespace NUMINAMATH_GPT_find_initial_quarters_l2065_206592

variables {Q : ℕ} -- Initial number of quarters

def quarters_to_dollars (q : ℕ) : ℝ := q * 0.25

noncomputable def initial_cash : ℝ := 40
noncomputable def cash_given_to_sister : ℝ := 5
noncomputable def quarters_given_to_sister : ℕ := 120
noncomputable def remaining_total : ℝ := 55

theorem find_initial_quarters (Q : ℕ) (h1 : quarters_to_dollars Q + 40 = 90) : Q = 200 :=
by { sorry }

end NUMINAMATH_GPT_find_initial_quarters_l2065_206592


namespace NUMINAMATH_GPT_not_associative_star_l2065_206538

def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - x - y

theorem not_associative_star : ¬ (∀ x y z : ℝ, star (star x y) z = star x (star y z)) :=
by
  sorry

end NUMINAMATH_GPT_not_associative_star_l2065_206538


namespace NUMINAMATH_GPT_bus_departure_l2065_206574

theorem bus_departure (current_people : ℕ) (min_people : ℕ) (required_people : ℕ) 
  (h1 : current_people = 9) (h2 : min_people = 16) : required_people = 7 :=
by 
  sorry

end NUMINAMATH_GPT_bus_departure_l2065_206574


namespace NUMINAMATH_GPT_cheolsu_initial_number_l2065_206571

theorem cheolsu_initial_number (x : ℚ) (h : x + (-5/12) - (-5/2) = 1/3) : x = -7/4 :=
by 
  sorry

end NUMINAMATH_GPT_cheolsu_initial_number_l2065_206571


namespace NUMINAMATH_GPT_amount_for_gifts_and_charitable_causes_l2065_206514

namespace JillExpenses

def net_monthly_salary : ℝ := 3700
def discretionary_income : ℝ := 0.20 * net_monthly_salary -- 1/5 * 3700
def vacation_fund : ℝ := 0.30 * discretionary_income
def savings : ℝ := 0.20 * discretionary_income
def eating_out_and_socializing : ℝ := 0.35 * discretionary_income
def gifts_and_charitable_causes : ℝ := discretionary_income - (vacation_fund + savings + eating_out_and_socializing)

theorem amount_for_gifts_and_charitable_causes : gifts_and_charitable_causes = 111 := sorry

end JillExpenses

end NUMINAMATH_GPT_amount_for_gifts_and_charitable_causes_l2065_206514


namespace NUMINAMATH_GPT_intersection_M_N_l2065_206501

-- Definition of the sets M and N
def M : Set ℝ := {x | 4 < x ∧ x < 8}
def N : Set ℝ := {x | x^2 - 6 * x < 0}

-- Intersection of M and N
def intersection : Set ℝ := {x | 4 < x ∧ x < 6}

-- Theorem statement asserting the equality between the intersection and the desired set
theorem intersection_M_N : ∀ (x : ℝ), x ∈ M ∩ N ↔ x ∈ intersection := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2065_206501


namespace NUMINAMATH_GPT_truck_total_distance_l2065_206504

noncomputable def truck_distance (b t : ℝ) : ℝ :=
  let acceleration := b / 3
  let time_seconds := 300 + t
  let distance_feet := (1 / 2) * (acceleration / t) * time_seconds^2
  distance_feet / 5280

theorem truck_total_distance (b t : ℝ) : 
  truck_distance b t = b * (90000 + 600 * t + t ^ 2) / (31680 * t) :=
by
  sorry

end NUMINAMATH_GPT_truck_total_distance_l2065_206504


namespace NUMINAMATH_GPT_num_pos_cubes_ending_in_5_lt_5000_l2065_206542

theorem num_pos_cubes_ending_in_5_lt_5000 : 
  (∃ (n1 n2 : ℕ), (n1 ≤ 5000 ∧ n2 ≤ 5000) ∧ (n1^3 % 10 = 5 ∧ n2^3 % 10 = 5) ∧ (n1^3 < 5000 ∧ n2^3 < 5000) ∧ n1 ≠ n2 ∧ 
  ∀ n, (n^3 < 5000 ∧ n^3 % 10 = 5) → (n = n1 ∨ n = n2)) :=
sorry

end NUMINAMATH_GPT_num_pos_cubes_ending_in_5_lt_5000_l2065_206542


namespace NUMINAMATH_GPT_sqrt_sum_inequality_l2065_206577

theorem sqrt_sum_inequality (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h_sum : a + b + c = 3) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ a * b + b * c + c * a) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sum_inequality_l2065_206577


namespace NUMINAMATH_GPT_least_possible_square_area_l2065_206526

theorem least_possible_square_area (s : ℝ) (h1 : 4.5 ≤ s) (h2 : s < 5.5) : s * s ≥ 20.25 := by
  sorry

end NUMINAMATH_GPT_least_possible_square_area_l2065_206526


namespace NUMINAMATH_GPT_sum_numbers_l2065_206541

theorem sum_numbers : 3456 + 4563 + 5634 + 6345 = 19998 := by
  sorry

end NUMINAMATH_GPT_sum_numbers_l2065_206541


namespace NUMINAMATH_GPT_ball_distribution_ways_l2065_206506

theorem ball_distribution_ways :
  let R := 5
  let W := 3
  let G := 2
  let total_balls := 10
  let balls_in_first_box := 4
  ∃ (distributions : ℕ), distributions = (Nat.choose total_balls balls_in_first_box) ∧ distributions = 210 :=
by
  sorry

end NUMINAMATH_GPT_ball_distribution_ways_l2065_206506


namespace NUMINAMATH_GPT_half_radius_of_circle_y_l2065_206500

theorem half_radius_of_circle_y
  (r_x r_y : ℝ)
  (hx : π * r_x ^ 2 = π * r_y ^ 2)
  (hc : 2 * π * r_x = 10 * π) :
  r_y / 2 = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_half_radius_of_circle_y_l2065_206500


namespace NUMINAMATH_GPT_total_tickets_sold_l2065_206533

def SeniorPrice : Nat := 10
def RegularPrice : Nat := 15
def TotalSales : Nat := 855
def RegularTicketsSold : Nat := 41

theorem total_tickets_sold : ∃ (S R : Nat), R = RegularTicketsSold ∧ 10 * S + 15 * R = TotalSales ∧ S + R = 65 :=
by
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l2065_206533


namespace NUMINAMATH_GPT_solution_set_of_abs_inequality_is_real_l2065_206518

theorem solution_set_of_abs_inequality_is_real (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| + m - 7 > 0) ↔ m > 4 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_abs_inequality_is_real_l2065_206518


namespace NUMINAMATH_GPT_smallest_positive_integer_l2065_206550

theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, 3003 * m + 60606 * n = 273 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_l2065_206550


namespace NUMINAMATH_GPT_closest_number_l2065_206593

theorem closest_number
  (a b c : ℝ)
  (h₀ : a = Real.sqrt 5)
  (h₁ : b = 3)
  (h₂ : b = (a + c) / 2) :
  abs (c - 3.5) ≤ abs (c - 2) ∧ abs (c - 3.5) ≤ abs (c - 2.5) ∧ abs (c - 3.5) ≤ abs (c - 3)  :=
by
  sorry

end NUMINAMATH_GPT_closest_number_l2065_206593


namespace NUMINAMATH_GPT_find_inverse_of_512_l2065_206576

-- Define the function f with the given properties
def f : ℕ → ℕ := sorry

axiom f_initial : f 5 = 2
axiom f_property : ∀ x, f (2 * x) = 2 * f x

-- State the problem as a theorem
theorem find_inverse_of_512 : ∃ x, f x = 512 ∧ x = 1280 :=
by 
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_find_inverse_of_512_l2065_206576


namespace NUMINAMATH_GPT_salary_increase_l2065_206513

theorem salary_increase (x : ℝ) 
  (h : ∀ s : ℕ, 1 ≤ s ∧ s ≤ 5 → ∃ p : ℝ, p = 7.50 + x * (s - 1))
  (h₁ : ∃ p₁ p₅ : ℝ, 1 ≤ 1 ∧ 5 ≤ 5 ∧ p₅ = p₁ + 1.25) :
  x = 0.3125 := sorry

end NUMINAMATH_GPT_salary_increase_l2065_206513


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l2065_206578

open Set

def M : Set ℝ := { x | x > 3 / 2 }
def N : Set ℝ := { x | x < 1 ∨ x > 3 }
def R := {x : ℝ | 1 ≤ x ∧ x ≤ 3 / 2}

theorem problem1 : M = { x | 2 * x - 3 > 0 } := sorry
theorem problem2 : N = { x | (x - 3) * (x - 1) > 0 } := sorry
theorem problem3 : M ∩ N = { x | x > 3 } := sorry
theorem problem4 : (M ∪ N)ᶜ = R := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l2065_206578


namespace NUMINAMATH_GPT_probability_x_lt_2y_l2065_206510

noncomputable def probability_x_lt_2y_in_rectangle : ℚ :=
  let area_triangle : ℚ := (1/2) * 4 * 2
  let area_rectangle : ℚ := 4 * 2
  (area_triangle / area_rectangle)

theorem probability_x_lt_2y (x y : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 4) (h3 : 0 ≤ y) (h4 : y ≤ 2) :
  probability_x_lt_2y_in_rectangle = 1/2 := by
  sorry

end NUMINAMATH_GPT_probability_x_lt_2y_l2065_206510


namespace NUMINAMATH_GPT_quadratic_real_roots_l2065_206566

-- Define the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ :=
  (a - 1) * x^2 - 2 * x + 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) : ℝ :=
  4 - 4 * (a - 1)

-- The main theorem stating the needed proof problem
theorem quadratic_real_roots (a : ℝ) : (∃ x : ℝ, quadratic_eq a x = 0) ↔ a ≤ 2 := by
  -- Proof will be inserted here
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l2065_206566


namespace NUMINAMATH_GPT_find_x_l2065_206537

theorem find_x (x : ℝ) (h : 2 * x - 3 * x + 5 * x = 80) : x = 20 :=
by 
  -- placeholder for proof
  sorry 

end NUMINAMATH_GPT_find_x_l2065_206537


namespace NUMINAMATH_GPT_number_of_blue_balloons_l2065_206509

def total_balloons : ℕ := 37
def red_balloons : ℕ := 14
def green_balloons : ℕ := 10

theorem number_of_blue_balloons : (total_balloons - red_balloons - green_balloons) = 13 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_number_of_blue_balloons_l2065_206509


namespace NUMINAMATH_GPT_part1_positive_integer_solutions_part2_value_of_m_part3_fixed_solution_l2065_206545

-- Part 1: Proof that the solutions of 2x + y - 6 = 0 under positive integer constraints are (2, 2) and (1, 4)
theorem part1_positive_integer_solutions : 
  (∃ x y : ℤ, 2 * x + y - 6 = 0 ∧ x > 0 ∧ y > 0) → 
  ({(x, y) | 2 * x + y - 6 = 0 ∧ x > 0 ∧ y > 0} = {(2, 2), (1, 4)})
:= sorry

-- Part 2: Proof that if x = y, the value of m that satisfies the system of equations is -4
theorem part2_value_of_m (x y m : ℤ) : 
  x = y → (∃ m, (2 * x + y - 6 = 0 ∧ 2 * x - 2 * y + m * y + 8 = 0)) → m = -4
:= sorry

-- Part 3: Proof that regardless of m, there is a fixed solution (x, y) = (-4, 0) for the equation 2x - 2y + my + 8 = 0
theorem part3_fixed_solution (m : ℤ) : 
  2 * x - 2 * y + m * y + 8 = 0 → (x, y) = (-4, 0)
:= sorry

end NUMINAMATH_GPT_part1_positive_integer_solutions_part2_value_of_m_part3_fixed_solution_l2065_206545


namespace NUMINAMATH_GPT_paint_proof_l2065_206511

/-- 
Suppose Jack's room has 27 square meters of wall and ceiling area. He has three choices for paint:
- Using 1 can of paint leaves 1 liter of paint left over,
- Using 5 gallons of paint leaves 1 liter of paint left over,
- Using 4 gallons and 2.8 liters of paint.

1. Prove: The ratio between the volume of a can and the volume of a gallon is 1:5.
2. Prove: The volume of a gallon is 3.8 liters.
3. Prove: The paint's coverage is 1.5 square meters per liter.
-/
theorem paint_proof (A : ℝ) (C G : ℝ) (R : ℝ):
  ∀ (H1: A = 27) (H2: C - 1 = 27) (H3: 5 * G - 1 = 27) (H4: 4 * G + 2.8 = 27), 
  (C / G = 1 / 5) ∧ (G = 3.8) ∧ ((A / (5 * G - 1)) = 1.5) :=
by
  sorry

end NUMINAMATH_GPT_paint_proof_l2065_206511


namespace NUMINAMATH_GPT_number_symmetry_equation_l2065_206596

theorem number_symmetry_equation (a b : ℕ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10 * a + b) * (100 * b + 10 * (a + b) + a) = (100 * a + 10 * (a + b) + b) * (10 * b + a) :=
by
  sorry

end NUMINAMATH_GPT_number_symmetry_equation_l2065_206596


namespace NUMINAMATH_GPT_quadratic_function_range_l2065_206543

theorem quadratic_function_range (x : ℝ) (h : x ≥ 0) : 
  3 ≤ x^2 + 2 * x + 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_function_range_l2065_206543


namespace NUMINAMATH_GPT_M_inter_N_is_01_l2065_206531

variable (x : ℝ)

def M := { x : ℝ | Real.log (1 - x) < 0 }
def N := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }

theorem M_inter_N_is_01 : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_M_inter_N_is_01_l2065_206531


namespace NUMINAMATH_GPT_proof_problem_l2065_206558

def number := 432

theorem proof_problem (y : ℕ) (n : ℕ) (h1 : y = 36) (h2 : 6^5 * 2 / n = y) : n = number :=
by 
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_proof_problem_l2065_206558


namespace NUMINAMATH_GPT_cubing_identity_l2065_206562

theorem cubing_identity (x : ℂ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := 
  sorry

end NUMINAMATH_GPT_cubing_identity_l2065_206562


namespace NUMINAMATH_GPT_stratified_sampling_model_A_l2065_206555

theorem stratified_sampling_model_A (r_A r_B r_C n x : ℕ) 
  (r_A_eq : r_A = 2) (r_B_eq : r_B = 3) (r_C_eq : r_C = 5) 
  (n_eq : n = 80) : 
  (r_A * n / (r_A + r_B + r_C) = x) -> x = 16 := 
by 
  intros h
  rw [r_A_eq, r_B_eq, r_C_eq, n_eq] at h
  norm_num at h
  exact h.symm

end NUMINAMATH_GPT_stratified_sampling_model_A_l2065_206555


namespace NUMINAMATH_GPT_solution_of_equation_l2065_206590

theorem solution_of_equation (a b c : ℕ) :
    a^(b + 20) * (c - 1) = c^(b + 21) - 1 ↔ 
    (∃ b' : ℕ, b = b' ∧ a = 1 ∧ c = 0) ∨ 
    (∃ a' b' : ℕ, a = a' ∧ b = b' ∧ c = 1) :=
by sorry

end NUMINAMATH_GPT_solution_of_equation_l2065_206590


namespace NUMINAMATH_GPT_intersection_A_B_l2065_206529

def A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem intersection_A_B : (A ∩ B) = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2065_206529


namespace NUMINAMATH_GPT_part_a_part_b_l2065_206536

-- Conditions
def ornament_to_crackers (n : ℕ) : ℕ := n * 2
def sparklers_to_garlands (n : ℕ) : ℕ := (n / 5) * 2
def garlands_to_ornaments (n : ℕ) : ℕ := n * 4

-- Part (a)
theorem part_a (sparklers : ℕ) (h : sparklers = 10) : ornament_to_crackers (garlands_to_ornaments (sparklers_to_garlands sparklers)) = 32 :=
by
  sorry

-- Part (b)
theorem part_b (ornaments : ℕ) (crackers : ℕ) (sparklers : ℕ) (h₁ : ornaments = 5) (h₂ : crackers = 1) (h₃ : sparklers = 2) :
  ornament_to_crackers ornaments + crackers > ornament_to_crackers (garlands_to_ornaments (sparklers_to_garlands sparklers)) :=
by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l2065_206536


namespace NUMINAMATH_GPT_function_monotonically_increasing_l2065_206515

-- The function y = x^2 - 2x + 8
def f (x : ℝ) : ℝ := x^2 - 2 * x + 8

-- The theorem stating the function is monotonically increasing on (1, +∞)
theorem function_monotonically_increasing : ∀ x y : ℝ, (1 < x) → (x < y) → (f x < f y) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_function_monotonically_increasing_l2065_206515


namespace NUMINAMATH_GPT_length_fraction_of_radius_l2065_206539

noncomputable def side_of_square_area (A : ℕ) : ℕ := Nat.sqrt A
noncomputable def radius_of_circle_from_square_area (A : ℕ) : ℕ := side_of_square_area A

noncomputable def length_of_rectangle_from_area_breadth (A b : ℕ) : ℕ := A / b
noncomputable def fraction_of_radius (len rad : ℕ) : ℚ := len / rad

theorem length_fraction_of_radius 
  (A_square A_rect breadth : ℕ) 
  (h_square_area : A_square = 1296)
  (h_rect_area : A_rect = 360)
  (h_breadth : breadth = 10) : 
  fraction_of_radius 
    (length_of_rectangle_from_area_breadth A_rect breadth)
    (radius_of_circle_from_square_area A_square) = 1 := 
by
  sorry

end NUMINAMATH_GPT_length_fraction_of_radius_l2065_206539


namespace NUMINAMATH_GPT_totalCupsOfLiquid_l2065_206568

def amountOfOil : ℝ := 0.17
def amountOfWater : ℝ := 1.17

theorem totalCupsOfLiquid : amountOfOil + amountOfWater = 1.34 := by
  sorry

end NUMINAMATH_GPT_totalCupsOfLiquid_l2065_206568


namespace NUMINAMATH_GPT_socks_choice_count_l2065_206546

variable (white_socks : ℕ) (brown_socks : ℕ) (blue_socks : ℕ) (black_socks : ℕ)

theorem socks_choice_count :
  white_socks = 5 →
  brown_socks = 4 →
  blue_socks = 2 →
  black_socks = 2 →
  (white_socks.choose 2) + (brown_socks.choose 2) + (blue_socks.choose 2) + (black_socks.choose 2) = 18 :=
by
  -- Here the proof would be elaborated
  sorry

end NUMINAMATH_GPT_socks_choice_count_l2065_206546


namespace NUMINAMATH_GPT_cara_arrangements_l2065_206559

theorem cara_arrangements (n : ℕ) (h : n = 7) : ∃ k : ℕ, k = 6 :=
by
  sorry

end NUMINAMATH_GPT_cara_arrangements_l2065_206559


namespace NUMINAMATH_GPT_Ronald_eggs_initially_l2065_206586

def total_eggs_shared (friends eggs_per_friend : Nat) : Nat :=
  friends * eggs_per_friend

theorem Ronald_eggs_initially (eggs : Nat) (candies : Nat) (friends : Nat) (eggs_per_friend : Nat)
  (h1 : friends = 8) (h2 : eggs_per_friend = 2) (h_share : total_eggs_shared friends eggs_per_friend = 16) :
  eggs = 16 := by
  sorry

end NUMINAMATH_GPT_Ronald_eggs_initially_l2065_206586


namespace NUMINAMATH_GPT_mitzi_money_left_l2065_206527

theorem mitzi_money_left :
  let A := 75
  let T := 30
  let F := 13
  let S := 23
  let total_spent := T + F + S
  let money_left := A - total_spent
  money_left = 9 :=
by
  sorry

end NUMINAMATH_GPT_mitzi_money_left_l2065_206527


namespace NUMINAMATH_GPT_paving_stone_length_l2065_206570

theorem paving_stone_length 
  (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (num_stones : ℕ) (stone_width : ℝ) 
  (courtyard_area : ℝ) 
  (total_stones_area : ℝ) 
  (L : ℝ) :
  courtyard_length = 50 →
  courtyard_width = 16.5 →
  num_stones = 165 →
  stone_width = 2 →
  courtyard_area = courtyard_length * courtyard_width →
  total_stones_area = num_stones * stone_width * L →
  courtyard_area = total_stones_area →
  L = 2.5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_paving_stone_length_l2065_206570


namespace NUMINAMATH_GPT_opposite_of_negative_five_l2065_206585

theorem opposite_of_negative_five : -(-5) = 5 := 
by
  sorry

end NUMINAMATH_GPT_opposite_of_negative_five_l2065_206585


namespace NUMINAMATH_GPT_red_blue_beads_ratio_l2065_206588

-- Definitions based on the conditions
def has_red_beads (betty : Type) := betty → ℕ
def has_blue_beads (betty : Type) := betty → ℕ

def betty : Type := Unit

-- Given conditions
def num_red_beads : has_red_beads betty := λ _ => 30
def num_blue_beads : has_blue_beads betty := λ _ => 20
def red_to_blue_ratio := 3 / 2

-- Theorem to prove the ratio
theorem red_blue_beads_ratio (R B: ℕ) (h_red : R = 30) (h_blue : B = 20) :
  (R / gcd R B) / (B / gcd R B ) = red_to_blue_ratio :=
by sorry

end NUMINAMATH_GPT_red_blue_beads_ratio_l2065_206588


namespace NUMINAMATH_GPT_sandra_total_beignets_l2065_206548

variable (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)

def daily_consumption (beignets_per_day : ℕ) := beignets_per_day
def weekly_consumption (beignets_per_day days_per_week : ℕ) := beignets_per_day * days_per_week
def total_consumption (beignets_per_day days_per_week weeks : ℕ) := weekly_consumption beignets_per_day days_per_week * weeks

theorem sandra_total_beignets :
  daily_consumption 3 = 3 →
  days_per_week = 7 →
  weeks = 16 →
  total_consumption 3 7 16 = 336 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_sandra_total_beignets_l2065_206548


namespace NUMINAMATH_GPT_smallest_number_of_weights_l2065_206520

/-- The smallest number of weights in a set that can be divided into 4, 5, and 6 equal piles is 11. -/
theorem smallest_number_of_weights (n : ℕ) (M : ℕ) : (∀ k : ℕ, (k = 4 ∨ k = 5 ∨ k = 6) → M % k = 0) → n = 11 :=
sorry

end NUMINAMATH_GPT_smallest_number_of_weights_l2065_206520


namespace NUMINAMATH_GPT_students_not_making_the_cut_l2065_206589

-- Define the total number of girls, boys, and the number of students called back
def number_of_girls : ℕ := 39
def number_of_boys : ℕ := 4
def students_called_back : ℕ := 26

-- Define the total number of students trying out
def total_students : ℕ := number_of_girls + number_of_boys

-- Formulate the problem statement as a theorem
theorem students_not_making_the_cut : total_students - students_called_back = 17 := 
by 
  -- Omitted proof, just the statement
  sorry

end NUMINAMATH_GPT_students_not_making_the_cut_l2065_206589


namespace NUMINAMATH_GPT_residue_class_equivalence_l2065_206521

variable {a m : ℤ}
variable {b : ℤ}

def residue_class (a m b : ℤ) : Prop := ∃ t : ℤ, b = m * t + a

theorem residue_class_equivalence (m a b : ℤ) :
  (∃ t : ℤ, b = m * t + a) ↔ b % m = a % m :=
by sorry

end NUMINAMATH_GPT_residue_class_equivalence_l2065_206521


namespace NUMINAMATH_GPT_bus_people_difference_l2065_206519

theorem bus_people_difference 
  (initial : ℕ) (got_off : ℕ) (got_on : ℕ) (current : ℕ) 
  (h_initial : initial = 35)
  (h_got_off : got_off = 18)
  (h_got_on : got_on = 15)
  (h_current : current = initial - got_off + got_on) :
  initial - current = 3 := by
  sorry

end NUMINAMATH_GPT_bus_people_difference_l2065_206519


namespace NUMINAMATH_GPT_crayons_count_l2065_206503

theorem crayons_count 
  (initial_crayons erasers : ℕ) 
  (erasers_count end_crayons : ℕ) 
  (initial_erasers : erasers = 38) 
  (end_crayons_more_erasers : end_crayons = erasers + 353) : 
  initial_crayons = end_crayons := 
by 
  sorry

end NUMINAMATH_GPT_crayons_count_l2065_206503


namespace NUMINAMATH_GPT_f_96_value_l2065_206597

noncomputable def f : ℕ → ℕ :=
sorry

axiom condition_1 (a b : ℕ) : 
  f (a * b) = f a + f b

axiom condition_2 (n : ℕ) (hp : Nat.Prime n) (hlt : 10 < n) : 
  f n = 0

axiom condition_3 : 
  f 1 < f 243 ∧ f 243 < f 2 ∧ f 2 < 11

axiom condition_4 : 
  f 2106 < 11

theorem f_96_value :
  f 96 = 31 :=
sorry

end NUMINAMATH_GPT_f_96_value_l2065_206597


namespace NUMINAMATH_GPT_find_slope_intercept_l2065_206535

def line_eqn (x y : ℝ) : Prop :=
  -3 * (x - 5) + 2 * (y + 1) = 0

theorem find_slope_intercept :
  ∃ (m b : ℝ), (∀ x y : ℝ, line_eqn x y → y = m * x + b) ∧ (m = 3/2) ∧ (b = -17/2) := sorry

end NUMINAMATH_GPT_find_slope_intercept_l2065_206535


namespace NUMINAMATH_GPT_percent_of_75_of_125_l2065_206530

theorem percent_of_75_of_125 : (75 / 125) * 100 = 60 := by
  sorry

end NUMINAMATH_GPT_percent_of_75_of_125_l2065_206530


namespace NUMINAMATH_GPT_people_at_first_concert_l2065_206564

def number_of_people_second_concert : ℕ := 66018
def additional_people_second_concert : ℕ := 119

theorem people_at_first_concert :
  number_of_people_second_concert - additional_people_second_concert = 65899 := by
  sorry

end NUMINAMATH_GPT_people_at_first_concert_l2065_206564


namespace NUMINAMATH_GPT_circle_equation_l2065_206587

theorem circle_equation (x y : ℝ) :
  let C := (4, -6)
  let r := 4
  (x - C.1)^2 + (y - C.2)^2 = r^2 →
  (x - 4)^2 + (y + 6)^2 = 16 :=
by
  intros
  sorry

end NUMINAMATH_GPT_circle_equation_l2065_206587


namespace NUMINAMATH_GPT_jenna_peeled_potatoes_l2065_206525

-- Definitions of constants
def initial_potatoes : ℕ := 60
def homer_rate : ℕ := 4
def jenna_rate : ℕ := 6
def combined_rate : ℕ := homer_rate + jenna_rate
def homer_time : ℕ := 6
def remaining_potatoes : ℕ := initial_potatoes - (homer_rate * homer_time)
def combined_time : ℕ := 4 -- Rounded from 3.6

-- Statement to prove
theorem jenna_peeled_potatoes : remaining_potatoes / combined_rate * jenna_rate = 24 :=
by
  sorry

end NUMINAMATH_GPT_jenna_peeled_potatoes_l2065_206525


namespace NUMINAMATH_GPT_smallest_class_size_l2065_206557

theorem smallest_class_size
  (x : ℕ)
  (h1 : ∀ y : ℕ, y = x + 2)
  (total_students : 5 * x + 2 > 40) :
  ∃ (n : ℕ), n = 5 * x + 2 ∧ n = 42 :=
by
  sorry

end NUMINAMATH_GPT_smallest_class_size_l2065_206557


namespace NUMINAMATH_GPT_sequence_length_arithmetic_sequence_l2065_206544

theorem sequence_length_arithmetic_sequence :
  ∀ (a d l n : ℕ), a = 5 → d = 3 → l = 119 → l = a + (n - 1) * d → n = 39 :=
by
  intros a d l n ha hd hl hln
  sorry

end NUMINAMATH_GPT_sequence_length_arithmetic_sequence_l2065_206544


namespace NUMINAMATH_GPT_cos_double_angle_l2065_206594

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l2065_206594


namespace NUMINAMATH_GPT_point_not_on_graph_l2065_206569

theorem point_not_on_graph : ¬ ∃ (x y : ℝ), (y = (x - 1) / (x + 2)) ∧ (x = -2) ∧ (y = 3) :=
by
  sorry

end NUMINAMATH_GPT_point_not_on_graph_l2065_206569


namespace NUMINAMATH_GPT_milk_cartons_total_l2065_206563

theorem milk_cartons_total (regular_milk soy_milk : ℝ) (h1 : regular_milk = 0.5) (h2 : soy_milk = 0.1) :
  regular_milk + soy_milk = 0.6 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_milk_cartons_total_l2065_206563


namespace NUMINAMATH_GPT_sqrt_sq_eq_l2065_206583

theorem sqrt_sq_eq (x : ℝ) : (Real.sqrt x) ^ 2 = x := by
  sorry

end NUMINAMATH_GPT_sqrt_sq_eq_l2065_206583


namespace NUMINAMATH_GPT_total_fraction_inspected_l2065_206540

-- Define the fractions of products inspected by John, Jane, and Roy.
variables (J N R : ℝ)
-- Define the rejection rates for John, Jane, and Roy.
variables (rJ rN rR : ℝ)
-- Define the total rejection rate.
variable (r_total : ℝ)

-- Define the conditions given in the problem.
def conditions : Prop :=
  (rJ = 0.007) ∧ (rN = 0.008) ∧ (rR = 0.01) ∧ (r_total = 0.0085) ∧
  (0.007 * J + 0.008 * N + 0.01 * R = 0.0085)

-- The proof statement that the total fraction of products inspected is 1.
theorem total_fraction_inspected (h : conditions J N R rJ rN rR r_total) : J + N + R = 1 :=
sorry

end NUMINAMATH_GPT_total_fraction_inspected_l2065_206540
