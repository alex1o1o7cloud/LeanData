import Mathlib

namespace fraction_sum_l1490_149064

theorem fraction_sum : (1 / 3 : ℚ) + (5 / 9 : ℚ) = (8 / 9 : ℚ) :=
by
  sorry

end fraction_sum_l1490_149064


namespace correct_option_l1490_149040

-- Define the options as propositions
def OptionA (a : ℕ) := a ^ 3 * a ^ 5 = a ^ 15
def OptionB (a : ℕ) := a ^ 8 / a ^ 2 = a ^ 4
def OptionC (a : ℕ) := a ^ 2 + a ^ 3 = a ^ 5
def OptionD (a : ℕ) := 3 * a - a = 2 * a

-- Prove that Option D is the only correct statement
theorem correct_option (a : ℕ) : OptionD a ∧ ¬OptionA a ∧ ¬OptionB a ∧ ¬OptionC a :=
by
  sorry

end correct_option_l1490_149040


namespace coordinate_plane_line_l1490_149000

theorem coordinate_plane_line (m n p : ℝ) (h1 : m = n / 5 - 2 / 5) (h2 : m + p = (n + 15) / 5 - 2 / 5) : p = 3 := by
  sorry

end coordinate_plane_line_l1490_149000


namespace intersection_of_complements_l1490_149073

-- Define the universal set U as a natural set with numbers <= 8
def U : Set ℕ := { x | x ≤ 8 }

-- Define the set A
def A : Set ℕ := { 1, 3, 7 }

-- Define the set B
def B : Set ℕ := { 2, 3, 8 }

-- Prove the statement for the intersection of the complements of A and B with respect to U
theorem intersection_of_complements : 
  ((U \ A) ∩ (U \ B)) = ({ 0, 4, 5, 6 } : Set ℕ) :=
by
  sorry

end intersection_of_complements_l1490_149073


namespace return_speed_is_33_33_l1490_149075

noncomputable def return_speed (d: ℝ) (speed_to_b: ℝ) (avg_speed: ℝ): ℝ :=
  d / (3 + (d / avg_speed))

-- Conditions
def distance := 150
def speed_to_b := 50
def avg_speed := 40

-- Prove that the return speed is 33.33 miles per hour
theorem return_speed_is_33_33:
  return_speed distance speed_to_b avg_speed = 33.33 :=
by
  unfold return_speed
  sorry

end return_speed_is_33_33_l1490_149075


namespace projectile_reaches_35m_first_at_10_over_7_l1490_149033

theorem projectile_reaches_35m_first_at_10_over_7 :
  ∃ (t : ℝ), (y : ℝ) = -4.9 * t^2 + 30 * t ∧ y = 35 ∧ t = 10 / 7 :=
by
  sorry

end projectile_reaches_35m_first_at_10_over_7_l1490_149033


namespace golden_section_AP_l1490_149070

-- Definitions of the golden ratio and its reciprocal
noncomputable def phi := (1 + Real.sqrt 5) / 2
noncomputable def phi_inv := (Real.sqrt 5 - 1) / 2

-- Conditions of the problem
def isGoldenSectionPoint (A B P : ℝ) := ∃ AP BP AB, AP < BP ∧ BP = 10 ∧ P = AB ∧ AP = BP * phi_inv

theorem golden_section_AP (A B P : ℝ) (h1 : isGoldenSectionPoint A B P) : 
  ∃ AP, AP = 5 * Real.sqrt 5 - 5 :=
by
  sorry

end golden_section_AP_l1490_149070


namespace Josanna_min_avg_score_l1490_149026

theorem Josanna_min_avg_score (scores : List ℕ) (cur_avg target_avg : ℚ)
  (next_test_bonus : ℚ) (additional_avg_points : ℚ) : ℚ :=
  let cur_avg := (92 + 81 + 75 + 65 + 88) / 5
  let target_avg := cur_avg + 6
  let needed_total := target_avg * 7
  let additional_points := 401 + 5
  let needed_sum := needed_total - additional_points
  needed_sum / 2

noncomputable def min_avg_score : ℚ :=
  Josanna_min_avg_score [92, 81, 75, 65, 88] 80.2 86.2 5 6

example : min_avg_score = 99 :=
by
  sorry

end Josanna_min_avg_score_l1490_149026


namespace function_periodicity_even_l1490_149056

theorem function_periodicity_even (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_period : ∀ x : ℝ, x ≥ 0 → f (x + 2) = -f x)
  (h_def : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2^x - 1) :
  f (-2017) + f 2018 = 1 :=
sorry

end function_periodicity_even_l1490_149056


namespace incorrect_statement_B_l1490_149079

def two_times_root_equation (a b c x1 x2 : ℝ) : Prop :=
  a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ (x1 = 2 * x2 ∨ x2 = 2 * x1)

theorem incorrect_statement_B (m n : ℝ) (h : (x - 2) * (m * x + n) = 0) :
  ¬(two_times_root_equation 1 (-m+n) (-mn) 2 (-n / m) -> m + n = 0) :=
sorry

end incorrect_statement_B_l1490_149079


namespace two_digit_number_multiple_l1490_149076

theorem two_digit_number_multiple (x : ℕ) (h1 : x ≥ 10) (h2 : x < 100) 
(h3 : ∃ k : ℕ, x + 1 = 3 * k) 
(h4 : ∃ k : ℕ, x + 1 = 4 * k) 
(h5 : ∃ k : ℕ, x + 1 = 5 * k) 
(h6 : ∃ k : ℕ, x + 1 = 7 * k) 
: x = 83 := 
sorry

end two_digit_number_multiple_l1490_149076


namespace range_of_m_n_l1490_149090

noncomputable def tangent_condition (m n : ℝ) : Prop :=
  ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1 → (m + 1) * x + (n + 1) * y - 2 = 0

theorem range_of_m_n (m n : ℝ) :
  tangent_condition m n →
  (m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2) :=
sorry

end range_of_m_n_l1490_149090


namespace average_community_age_l1490_149013

variable (num_women num_men : Nat)
variable (avg_age_women avg_age_men : Nat)

def ratio_women_men := num_women = 7 * num_men / 8
def average_age_women := avg_age_women = 30
def average_age_men := avg_age_men = 35

theorem average_community_age (k : Nat) 
  (h_ratio : ratio_women_men (7 * k) (8 * k)) 
  (h_avg_women : average_age_women 30)
  (h_avg_men : average_age_men 35) : 
  (30 * (7 * k) + 35 * (8 * k)) / (15 * k) = 32 + (2 / 3) := 
sorry

end average_community_age_l1490_149013


namespace expression_meaningful_range_l1490_149051

theorem expression_meaningful_range (a : ℝ) : (∃ x, x = (a + 3) ^ (1/2) / (a - 1)) ↔ (a ≥ -3 ∧ a ≠ 1) :=
by
  sorry

end expression_meaningful_range_l1490_149051


namespace tan_expression_l1490_149015

theorem tan_expression (a : ℝ) (h₀ : 45 = 2 * a) (h₁ : Real.tan 45 = 1) 
  (h₂ : Real.tan (2 * a) = 2 * Real.tan a / (1 - Real.tan a * Real.tan a)) :
  Real.tan a / (1 - Real.tan a * Real.tan a) = 1 / 2 :=
by 
  sorry

end tan_expression_l1490_149015


namespace find_intersection_A_B_find_range_t_l1490_149016

-- Define sets A, B, C
def A : Set ℝ := {y | ∃ x, (1 ≤ x ∧ x ≤ 2) ∧ y = 2^x}
def B : Set ℝ := {x | 0 < Real.log x ∧ Real.log x < 1}
def C (t : ℝ) : Set ℝ := {x | t + 1 < x ∧ x < 2 * t}

-- Theorem 1: Finding A ∩ B
theorem find_intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < Real.exp 1} := 
by
  sorry

-- Theorem 2: If A ∩ C = C, find the range of values for t
theorem find_range_t (t : ℝ) (h : A ∩ C t = C t) : t ≤ 2 :=
by
  sorry

end find_intersection_A_B_find_range_t_l1490_149016


namespace common_face_sum_is_9_l1490_149019

noncomputable def common_sum (vertices : Fin 9 → ℕ) : ℕ :=
  let total_sum := (Finset.sum (Finset.univ : Finset (Fin 9)) vertices)
  let additional_sum := 9
  let total_with_addition := total_sum + additional_sum
  total_with_addition / 6

theorem common_face_sum_is_9 :
  ∀ (vertices : Fin 9 → ℕ), (∀ i, 1 ≤ vertices i ∧ vertices i ≤ 9) →
  Finset.sum (Finset.univ : Finset (Fin 9)) vertices = 45 →
  common_sum vertices = 9 := 
by
  intros vertices h1 h_sum
  unfold common_sum
  sorry

end common_face_sum_is_9_l1490_149019


namespace matrix_eq_value_satisfied_for_two_values_l1490_149097

variable (a b c d x : ℝ)

def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

-- Define the specific instance for the given matrix problem
def matrix_eq_value (x : ℝ) : Prop :=
  matrix_value (2 * x) x 1 x = 3

-- Prove that the equation is satisfied for exactly two values of x
theorem matrix_eq_value_satisfied_for_two_values :
  (∃! (x : ℝ), matrix_value (2 * x) x 1 x = 3) :=
sorry

end matrix_eq_value_satisfied_for_two_values_l1490_149097


namespace alpha_sin_beta_lt_beta_sin_alpha_l1490_149020

variable {α β : ℝ}

theorem alpha_sin_beta_lt_beta_sin_alpha (h1 : 0 < α) (h2 : α < β) (h3 : β < Real.pi / 2) : 
  α * Real.sin β < β * Real.sin α := 
by
  sorry

end alpha_sin_beta_lt_beta_sin_alpha_l1490_149020


namespace calculate_expression_l1490_149030

theorem calculate_expression : 15 * 30 + 45 * 15 + 90 = 1215 := 
by 
  sorry

end calculate_expression_l1490_149030


namespace fifth_flower_is_e_l1490_149066

def flowers : List String := ["a", "b", "c", "d", "e", "f", "g"]

theorem fifth_flower_is_e : flowers.get! 4 = "e" := sorry

end fifth_flower_is_e_l1490_149066


namespace sum_of_cubes_of_ages_l1490_149022

noncomputable def dick_age : ℕ := 2
noncomputable def tom_age : ℕ := 5
noncomputable def harry_age : ℕ := 6

theorem sum_of_cubes_of_ages :
  4 * dick_age + 2 * tom_age = 3 * harry_age ∧ 
  3 * harry_age^2 = 2 * dick_age^2 + 4 * tom_age^2 ∧ 
  Nat.gcd (Nat.gcd dick_age tom_age) harry_age = 1 → 
  dick_age^3 + tom_age^3 + harry_age^3 = 349 :=
by
  intros h
  sorry

end sum_of_cubes_of_ages_l1490_149022


namespace ramola_rank_from_first_l1490_149055

-- Conditions definitions
def total_students : ℕ := 26
def ramola_rank_from_last : ℕ := 13

-- Theorem statement
theorem ramola_rank_from_first : total_students - (ramola_rank_from_last - 1) = 14 := 
by 
-- We use 'by' to begin the proof block
sorry 
-- We use 'sorry' to indicate the proof is omitted

end ramola_rank_from_first_l1490_149055


namespace least_number_subtracted_l1490_149034

theorem least_number_subtracted (a b c : ℕ) (h1 : a = 7) (h2 : b = 9) (h3 : c = 11) :
  ∃ x, 0 ≤ x ∧ x < 1398 ∧ (1398 - x) % a = 5 ∧ (1398 - x) % b = 5 ∧ (1398 - x) % c = 5 ∧ x = 22 :=
by {
  sorry
}

end least_number_subtracted_l1490_149034


namespace tetrahedron_face_area_inequality_l1490_149053

theorem tetrahedron_face_area_inequality
  (T_ABC T_ABD T_ACD T_BCD : ℝ)
  (h : T_ABC ≥ 0 ∧ T_ABD ≥ 0 ∧ T_ACD ≥ 0 ∧ T_BCD ≥ 0) :
  T_ABC < T_ABD + T_ACD + T_BCD :=
sorry

end tetrahedron_face_area_inequality_l1490_149053


namespace ratio_of_division_of_chord_l1490_149041

theorem ratio_of_division_of_chord (R AP PB O: ℝ) (radius_given: R = 11) (chord_length: AP + PB = 18) (point_distance: O = 7) : 
  (AP / PB = 2 ∨ PB / AP = 2) :=
by 
  -- Proof goes here, to be filled in later
  sorry

end ratio_of_division_of_chord_l1490_149041


namespace sum_of_areas_is_72_l1490_149025

def base : ℕ := 2
def length1 : ℕ := 1
def length2 : ℕ := 8
def length3 : ℕ := 27

theorem sum_of_areas_is_72 : base * length1 + base * length2 + base * length3 = 72 :=
by
  sorry

end sum_of_areas_is_72_l1490_149025


namespace sufficient_but_not_necessary_condition_l1490_149050

variable (x : ℝ)

def p := x > 2
def q := x^2 > 4

theorem sufficient_but_not_necessary_condition : (p x) → (q x) ∧ ¬((q x) → (p x)) := 
by
  sorry

end sufficient_but_not_necessary_condition_l1490_149050


namespace geometric_reasoning_l1490_149077

-- Definitions of relationships between geometric objects
inductive GeometricObject
  | Line
  | Plane

open GeometricObject

def perpendicular (a b : GeometricObject) : Prop := 
  match a, b with
  | Plane, Plane => True  -- Planes can be perpendicular
  | Line, Plane => True   -- Lines can be perpendicular to planes
  | Plane, Line => True   -- Planes can be perpendicular to lines
  | Line, Line => True    -- Lines can be perpendicular to lines (though normally in a 3D space specific context)

def parallel (a b : GeometricObject) : Prop := 
  match a, b with
  | Plane, Plane => True  -- Planes can be parallel
  | Line, Plane => True   -- Lines can be parallel to planes under certain interpretation
  | Plane, Line => True
  | Line, Line => True    -- Lines can be parallel

axiom x : GeometricObject
axiom y : GeometricObject
axiom z : GeometricObject

-- Main theorem statement
theorem geometric_reasoning (hx : perpendicular x y) (hy : parallel y z) 
  : ¬ (perpendicular x z) → (x = Plane ∧ y = Plane ∧ z = Line) :=
  sorry

end geometric_reasoning_l1490_149077


namespace roots_of_polynomial_fraction_l1490_149084

theorem roots_of_polynomial_fraction (a b c : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a * b + a * c + b * c = 11)
  (h3 : a * b * c = 6) :
  a / (b * c + 2) + b / (a * c + 2) + c / (a * b + 2) = 3 / 2 := 
by
  sorry

end roots_of_polynomial_fraction_l1490_149084


namespace vehicles_with_at_least_80_kmh_equal_50_l1490_149086

variable (num_vehicles_80_to_89 : ℕ := 15)
variable (num_vehicles_90_to_99 : ℕ := 30)
variable (num_vehicles_100_to_109 : ℕ := 5)

theorem vehicles_with_at_least_80_kmh_equal_50 :
  num_vehicles_80_to_89 + num_vehicles_90_to_99 + num_vehicles_100_to_109 = 50 := by
  sorry

end vehicles_with_at_least_80_kmh_equal_50_l1490_149086


namespace sequence_properties_l1490_149008

noncomputable def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
a1 + d * (n - 1)

theorem sequence_properties (d a1 : ℤ) (h_d_ne_zero : d ≠ 0)
(h1 : arithmetic_sequence a1 d 2 + arithmetic_sequence a1 d 4 = 10)
(h2 : (arithmetic_sequence a1 d 2)^2 = (arithmetic_sequence a1 d 1) * (arithmetic_sequence a1 d 5)) :
a1 = 1 ∧ ∀ n : ℕ, n > 0 → arithmetic_sequence 1 2 n = 2 * n - 1 :=
by
  sorry

end sequence_properties_l1490_149008


namespace inequality_preservation_l1490_149042

theorem inequality_preservation (a b x : ℝ) (h : a > b) : a * 2^x > b * 2^x :=
sorry

end inequality_preservation_l1490_149042


namespace exists_sum_or_diff_divisible_by_1000_l1490_149046

theorem exists_sum_or_diff_divisible_by_1000 (nums : Fin 502 → Nat) :
  ∃ a b : Nat, (∃ i j : Fin 502, nums i = a ∧ nums j = b ∧ i ≠ j) ∧
  (a - b) % 1000 = 0 ∨ (a + b) % 1000 = 0 :=
by
  sorry

end exists_sum_or_diff_divisible_by_1000_l1490_149046


namespace range_of_x_l1490_149014

theorem range_of_x (x : ℝ) (h : Real.log (x - 1) < 1) : 1 < x ∧ x < Real.exp 1 + 1 :=
by
  sorry

end range_of_x_l1490_149014


namespace hockey_league_teams_l1490_149001

theorem hockey_league_teams (n : ℕ) (h : (n * (n - 1) * 10) / 2 = 1710) : n = 19 :=
by {
  sorry
}

end hockey_league_teams_l1490_149001


namespace min_value_expression_l1490_149006

theorem min_value_expression : 
  ∀ (x y : ℝ), (3 * x * x + 4 * x * y + 4 * y * y - 12 * x - 8 * y ≥ -28) ∧ 
  (3 * ((8:ℝ)/3) * ((8:ℝ)/3) + 4 * ((8:ℝ)/3) * -1 + 4 * -1 * -1 - 12 * ((8:ℝ)/3) - 8 * -1 = -28) := 
by sorry

end min_value_expression_l1490_149006


namespace no_sol_n4_minus_m4_eq_42_l1490_149078

theorem no_sol_n4_minus_m4_eq_42 :
  ¬ ∃ (n m : ℕ), 0 < n ∧ 0 < m ∧ n^4 - m^4 = 42 :=
by
  sorry

end no_sol_n4_minus_m4_eq_42_l1490_149078


namespace krishan_money_l1490_149021

theorem krishan_money 
  (x y : ℝ)
  (hx1 : 7 * x * 1.185 = 699.8)
  (hx2 : 10 * x * 0.8 = 800)
  (hy : 17 * x = 8 * y) : 
  16 * y = 3400 := 
by
  -- It's acceptable to leave the proof incomplete due to the focus being on the statement.
  sorry

end krishan_money_l1490_149021


namespace non_congruent_rectangles_unique_l1490_149038

theorem non_congruent_rectangles_unique (P : ℕ) (w : ℕ) (h : ℕ) :
  P = 72 ∧ w = 14 ∧ 2 * (w + h) = P → 
  (∃ h, w = 14 ∧ 2 * (w + h) = 72 ∧ 
  ∀ w' h', w' = w → 2 * (w' + h') = 72 → (h' = h)) :=
by
  sorry

end non_congruent_rectangles_unique_l1490_149038


namespace determine_q_l1490_149060

-- Define the polynomial p(x) and its square
def p (x : ℝ) : ℝ := x^2 + x + 1
def p_squared (x : ℝ) : ℝ := (x^2 + x + 1)^2

-- Define the identity condition
def identity_condition (x : ℝ) (q : ℝ → ℝ) : Prop := 
  p_squared x - 2 * p x * q x + (q x)^2 - 4 * p x + 3 * q x + 3 = 0

-- Ellaboration on the required solution
def correct_q (q : ℝ → ℝ) : Prop :=
  (∀ x, q x = x^2 + 2 * x) ∨ (∀ x, q x = x^2 - 1)

-- The theorem statement
theorem determine_q :
  ∀ q : ℝ → ℝ, (∀ x : ℝ, identity_condition x q) → correct_q q :=
by
  intros
  sorry

end determine_q_l1490_149060


namespace articles_produced_l1490_149093

theorem articles_produced (a b c d f p q r g : ℕ) :
  (a * b * c = d) → 
  ((p * q * r * d * g) / (a * b * c * f) = pqr * d * g / (abc * f)) :=
by
  sorry

end articles_produced_l1490_149093


namespace rachel_homework_l1490_149012

theorem rachel_homework : 5 + 2 = 7 := by
  sorry

end rachel_homework_l1490_149012


namespace cos_C_value_l1490_149027

namespace Triangle

theorem cos_C_value (A B C : ℝ)
  (h_triangle : A + B + C = Real.pi)
  (sin_A : Real.sin A = 2/3)
  (cos_B : Real.cos B = 1/2) :
  Real.cos C = (2 * Real.sqrt 3 - Real.sqrt 5) / 6 := 
sorry

end Triangle

end cos_C_value_l1490_149027


namespace quadratic_condition_not_necessary_and_sufficient_l1490_149028

theorem quadratic_condition_not_necessary_and_sufficient (a b c : ℝ) :
  ¬((∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ (b^2 - 4 * a * c < 0)) :=
sorry

end quadratic_condition_not_necessary_and_sufficient_l1490_149028


namespace proof_problem_l1490_149098

theorem proof_problem (s t: ℤ) (h : 514 - s = 600 - t) : s < t ∧ t - s = 86 :=
by
  sorry

end proof_problem_l1490_149098


namespace integer_not_in_range_of_f_l1490_149062

noncomputable def f (x : ℝ) : ℤ :=
  if x > -1 then ⌈1 / (x + 1)⌉ else ⌊1 / (x + 1)⌋

theorem integer_not_in_range_of_f :
  ¬ ∃ x : ℝ, x ≠ -1 ∧ f x = 0 :=
by
  sorry

end integer_not_in_range_of_f_l1490_149062


namespace find_length_of_other_diagonal_l1490_149092

theorem find_length_of_other_diagonal
  (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h1: area = 75)
  (h2: d1 = 10) :
  d2 = 15 :=
by 
  sorry

end find_length_of_other_diagonal_l1490_149092


namespace solve_equation_frac_l1490_149074

theorem solve_equation_frac (x : ℝ) (h : x ≠ 2) : (3 / (x - 2) = 1) ↔ (x = 5) :=
by
  sorry -- proof is to be constructed

end solve_equation_frac_l1490_149074


namespace number_of_sets_without_perfect_squares_l1490_149054

/-- Define the set T_i of all integers n such that 200i ≤ n < 200(i + 1). -/
def T (i : ℕ) : Set ℕ := {n | 200 * i ≤ n ∧ n < 200 * (i + 1)}

/-- The total number of sets T_i from T_0 to T_{499}. -/
def total_sets : ℕ := 500

/-- The number of sets from T_0 to T_{499} that contain at least one perfect square. -/
def sets_with_perfect_squares : ℕ := 317

/-- The number of sets from T_0 to T_{499} that do not contain any perfect squares. -/
def sets_without_perfect_squares : ℕ := total_sets - sets_with_perfect_squares

/-- Proof that the number of sets T_0, T_1, T_2, ..., T_{499} that do not contain a perfect square is 183. -/
theorem number_of_sets_without_perfect_squares : sets_without_perfect_squares = 183 :=
by
  sorry

end number_of_sets_without_perfect_squares_l1490_149054


namespace value_of_a_l1490_149072

theorem value_of_a (a b c : ℂ) (h_real : a.im = 0)
  (h1 : a + b + c = 5) 
  (h2 : a * b + b * c + c * a = 7) 
  (h3 : a * b * c = 2) : a = 2 := by
  sorry

end value_of_a_l1490_149072


namespace factorize_expression_l1490_149029

-- Define the variables m and n
variables (m n : ℝ)

-- The statement to prove
theorem factorize_expression : -8 * m^2 + 2 * m * n = -2 * m * (4 * m - n) :=
sorry

end factorize_expression_l1490_149029


namespace power_of_three_divides_an_l1490_149069

theorem power_of_three_divides_an (a : ℕ → ℕ) (k : ℕ) (h1 : a 1 = 3)
  (h2 : ∀ n, a (n + 1) = ((3 * (a n)^2 + 1) / 2) - a n)
  (h3 : ∃ m, n = 3^m) :
  3^(k + 1) ∣ a (3^k) :=
sorry

end power_of_three_divides_an_l1490_149069


namespace combined_list_correct_l1490_149071

def james_friends : ℕ := 75
def john_friends : ℕ := 3 * james_friends
def shared_friends : ℕ := 25

def combined_list : ℕ :=
  james_friends + john_friends - shared_friends

theorem combined_list_correct : combined_list = 275 := by
  unfold combined_list
  unfold james_friends
  unfold john_friends
  unfold shared_friends
  sorry

end combined_list_correct_l1490_149071


namespace arctan_sum_is_pi_over_4_l1490_149039

open Real

theorem arctan_sum_is_pi_over_4 (a b c : ℝ) (h1 : b = c) (h2 : c / (a + b) + a / (b + c) = 1) :
  arctan (c / (a + b)) + arctan (a / (b + c)) = π / 4 :=
by 
  sorry

end arctan_sum_is_pi_over_4_l1490_149039


namespace cara_total_debt_l1490_149094

def simple_interest (P R T : ℝ) : ℝ := P * R * T

theorem cara_total_debt :
  let P := 54
  let R := 0.05
  let T := 1
  let I := simple_interest P R T
  let total := P + I
  total = 56.7 :=
by
  sorry

end cara_total_debt_l1490_149094


namespace slope_of_parallel_line_l1490_149032

-- Given condition: the equation of the line
def line_equation (x y : ℝ) : Prop := 2 * x - 4 * y = 9

-- Goal: the slope of any line parallel to the given line is 1/2
theorem slope_of_parallel_line (x y : ℝ) (m : ℝ) :
  (∀ x y, line_equation x y) → m = 1 / 2 := by
  sorry

end slope_of_parallel_line_l1490_149032


namespace alcohol_percentage_calculation_l1490_149057

-- Define the conditions as hypothesis
variables (original_solution_volume : ℝ) (original_alcohol_percent : ℝ)
          (added_alcohol_volume : ℝ) (added_water_volume : ℝ)

-- Assume the given values in the problem
variables (h1 : original_solution_volume = 40) (h2 : original_alcohol_percent = 5)
          (h3 : added_alcohol_volume = 2.5) (h4 : added_water_volume = 7.5)

-- Define the proof goal
theorem alcohol_percentage_calculation :
  let original_alcohol_volume := original_solution_volume * (original_alcohol_percent / 100)
  let total_alcohol_volume := original_alcohol_volume + added_alcohol_volume
  let total_solution_volume := original_solution_volume + added_alcohol_volume + added_water_volume
  let new_alcohol_percent := (total_alcohol_volume / total_solution_volume) * 100
  new_alcohol_percent = 9 :=
by {
  sorry
}

end alcohol_percentage_calculation_l1490_149057


namespace certain_number_l1490_149018

theorem certain_number (p q : ℝ) (h1 : 3 / p = 6) (h2 : p - q = 0.3) : 3 / q = 15 :=
by
  sorry

end certain_number_l1490_149018


namespace number_of_floors_l1490_149023

-- Definitions
def height_regular_floor : ℝ := 3
def height_last_floor : ℝ := 3.5
def total_height : ℝ := 61

-- Theorem statement
theorem number_of_floors (n : ℕ) : 
  (n ≥ 2) →
  (2 * height_last_floor + (n - 2) * height_regular_floor = total_height) →
  n = 20 :=
sorry

end number_of_floors_l1490_149023


namespace abc_sum_leq_three_l1490_149044

open Real

theorem abc_sum_leq_three {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 + a * b * c = 4) :
  a + b + c ≤ 3 :=
sorry

end abc_sum_leq_three_l1490_149044


namespace div_by_9_implies_not_div_by_9_l1490_149099

/-- If 9 divides 10^n + 1, then it also divides 10^(n+1) + 1 -/
theorem div_by_9_implies:
  ∀ n: ℕ, (9 ∣ (10^n + 1)) → (9 ∣ (10^(n + 1) + 1)) :=
by
  intro n
  intro h
  sorry

/-- 9 does not divide 10^1 + 1 -/
theorem not_div_by_9:
  ¬(9 ∣ (10^1 + 1)) :=
by 
  sorry

end div_by_9_implies_not_div_by_9_l1490_149099


namespace imaginary_part_of_z_is_1_l1490_149067

def z := Complex.ofReal 0 + Complex.ofReal 1 * Complex.I * (Complex.ofReal 1 + Complex.ofReal 2 * Complex.I)
theorem imaginary_part_of_z_is_1 : z.im = 1 := by
  sorry

end imaginary_part_of_z_is_1_l1490_149067


namespace roots_numerically_equal_but_opposite_signs_l1490_149081

noncomputable def value_of_m (a b c : ℝ) : ℝ := (a - b) / (a + b)

theorem roots_numerically_equal_but_opposite_signs
  (a b c m : ℝ)
  (h : ∀ x : ℝ, (a ≠ 0 ∧ a + b ≠ 0) ∧ (x^2 - b*x = (ax - c) * (m - 1) / (m + 1))) 
  (root_condition : ∃ x₁ x₂ : ℝ, x₁ = -x₂ ∧ x₁ * x₂ != 0) :
  m = value_of_m a b c :=
by
  sorry

end roots_numerically_equal_but_opposite_signs_l1490_149081


namespace exists_perfect_square_sum_l1490_149035

theorem exists_perfect_square_sum (n : ℕ) (h : n > 2) : ∃ m : ℕ, ∃ k : ℕ, n^2 + m^2 = k^2 :=
by
  sorry

end exists_perfect_square_sum_l1490_149035


namespace question_1_solution_question_2_solution_l1490_149004

def f (m x : ℝ) := m*x^2 - (m^2 + 1)*x + m

theorem question_1_solution (x : ℝ) :
  (f 2 x ≤ 0) ↔ (1 / 2 ≤ x ∧ x ≤ 2) :=
sorry

theorem question_2_solution (x m : ℝ) :
  (m > 0) → 
  ((0 < m ∧ m < 1 → f m x > 0 ↔ x < m ∨ x > 1 / m) ∧
  (m = 1 → f m x > 0 ↔ x ≠ 1) ∧
  (m > 1 → f m x > 0 ↔ x < 1 / m ∨ x > m)) :=
sorry

end question_1_solution_question_2_solution_l1490_149004


namespace scout_troop_profit_l1490_149061

theorem scout_troop_profit :
  let bars_bought := 1200
  let cost_per_bar := 1 / 3
  let bars_per_dollar := 3
  let total_cost := bars_bought * cost_per_bar
  let selling_price_per_bar := 3 / 5
  let bars_per_three_dollars := 5
  let total_revenue := bars_bought * selling_price_per_bar
  let profit := total_revenue - total_cost
  profit = 320 := by
  let bars_bought := 1200
  let cost_per_bar := 1 / 3
  let total_cost := bars_bought * cost_per_bar
  let selling_price_per_bar := 3 / 5
  let total_revenue := bars_bought * selling_price_per_bar
  let profit := total_revenue - total_cost
  sorry

end scout_troop_profit_l1490_149061


namespace union_of_M_N_l1490_149024

def M : Set ℝ := { x | x^2 + 2*x = 0 }

def N : Set ℝ := { x | x^2 - 2*x = 0 }

theorem union_of_M_N : M ∪ N = {0, -2, 2} := sorry

end union_of_M_N_l1490_149024


namespace parabola_focus_l1490_149080

theorem parabola_focus (p : ℝ) (hp : ∃ (p : ℝ), ∀ x y : ℝ, x^2 = 2 * p * y) : (∀ (hf : (0, 2) = (0, p / 2)), p = 4) :=
sorry

end parabola_focus_l1490_149080


namespace sin_minus_cos_eq_l1490_149002

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l1490_149002


namespace stamps_on_last_page_l1490_149043

theorem stamps_on_last_page
  (B : ℕ) (P_b : ℕ) (S_p : ℕ) (S_p_star : ℕ) 
  (B_comp : ℕ) (P_last : ℕ) 
  (stamps_total : ℕ := B * P_b * S_p) 
  (pages_total : ℕ := stamps_total / S_p_star)
  (pages_comp : ℕ := B_comp * P_b)
  (pages_filled : ℕ := pages_total - pages_comp) :
  stamps_total - (pages_total - 1) * S_p_star = 8 :=
by
  -- Proof steps would follow here.
  sorry

end stamps_on_last_page_l1490_149043


namespace polynomial_satisfies_conditions_l1490_149049

noncomputable def f (x y z : ℝ) : ℝ :=
  (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

theorem polynomial_satisfies_conditions :
  (f x (z^2) y + f x (y^2) z = 0) ∧ (f (z^3) y x + f (x^3) y z = 0) :=
by
  sorry

end polynomial_satisfies_conditions_l1490_149049


namespace least_integer_greater_than_sqrt_500_l1490_149009

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l1490_149009


namespace always_positive_sum_reciprocal_inequality_l1490_149096

-- Problem 1
theorem always_positive (x : ℝ) : x^6 - x^3 + x^2 - x + 1 > 0 :=
sorry

-- Problem 2
theorem sum_reciprocal_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  1/a + 1/b + 1/c ≥ 9 :=
sorry

end always_positive_sum_reciprocal_inequality_l1490_149096


namespace picnic_adults_children_difference_l1490_149045

theorem picnic_adults_children_difference :
  ∃ (M W A C : ℕ),
    (M = 65) ∧
    (M = W + 20) ∧
    (A = M + W) ∧
    (C = 200 - A) ∧
    ((A - C) = 20) :=
by
  sorry

end picnic_adults_children_difference_l1490_149045


namespace heidi_zoe_paint_fraction_l1490_149059

theorem heidi_zoe_paint_fraction (H_period : ℝ) (HZ_period : ℝ) :
  (H_period = 60 → HZ_period = 40 → (8 / 40) = (1 / 5)) :=
by intros H_period_eq HZ_period_eq
   sorry

end heidi_zoe_paint_fraction_l1490_149059


namespace train_length_l1490_149011

theorem train_length (L V : ℝ) 
  (h1 : V = L / 10) 
  (h2 : V = (L + 870) / 39) 
  : L = 300 :=
by
  sorry

end train_length_l1490_149011


namespace perpendicular_lines_iff_l1490_149065

theorem perpendicular_lines_iff (a : ℝ) : 
  (∀ b₁ b₂ : ℝ, b₁ ≠ b₂ → ¬ (∀ x : ℝ, a * x + b₁ = (a - 2) * x + b₂) ∧ 
   (a * (a - 2) = -1)) ↔ a = 1 :=
by
  sorry

end perpendicular_lines_iff_l1490_149065


namespace original_amount_of_solution_y_l1490_149089

theorem original_amount_of_solution_y (Y : ℝ) 
  (h1 : 0 < Y) -- We assume Y > 0 
  (h2 : 0.3 * (Y - 4) + 1.2 = 0.45 * Y) :
  Y = 8 := 
sorry

end original_amount_of_solution_y_l1490_149089


namespace chess_club_boys_count_l1490_149063

theorem chess_club_boys_count (B G : ℕ) 
  (h1 : B + G = 30)
  (h2 : (2/3 : ℝ) * G + B = 18) : 
  B = 6 :=
by
  sorry

end chess_club_boys_count_l1490_149063


namespace smaller_rectangle_area_l1490_149085

theorem smaller_rectangle_area (L_h S_h : ℝ) (L_v S_v : ℝ) 
  (ratio_h : L_h = (8 / 7) * S_h) 
  (ratio_v : L_v = (9 / 4) * S_v) 
  (area_large : L_h * L_v = 108) :
  S_h * S_v = 42 :=
sorry

end smaller_rectangle_area_l1490_149085


namespace diff_of_two_numbers_l1490_149091

theorem diff_of_two_numbers :
  ∃ D S : ℕ, (1650 = 5 * S + 5) ∧ (D = 1650 - S) ∧ (D = 1321) :=
sorry

end diff_of_two_numbers_l1490_149091


namespace right_triangle_area_semi_perimeter_inequality_l1490_149031

theorem right_triangle_area_semi_perimeter_inequality 
  (x y : ℝ) (h : x > 0 ∧ y > 0) 
  (p : ℝ := (x + y + Real.sqrt (x^2 + y^2)) / 2)
  (S : ℝ := x * y / 2) 
  (hypotenuse : ℝ := Real.sqrt (x^2 + y^2)) 
  (right_triangle : hypotenuse ^ 2 = x ^ 2 + y ^ 2) : 
  S <= p^2 / 5.5 := 
sorry

end right_triangle_area_semi_perimeter_inequality_l1490_149031


namespace ceil_evaluation_l1490_149087

theorem ceil_evaluation : 
  (Int.ceil (((-7 : ℚ) / 4) ^ 2 - (1 / 8)) = 3) :=
sorry

end ceil_evaluation_l1490_149087


namespace infinite_series_sum_l1490_149017

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l1490_149017


namespace power_mod_remainder_l1490_149088

theorem power_mod_remainder (a b c : ℕ) (h1 : 7^40 % 500 = 1) (h2 : 7^4 % 40 = 1) : (7^(7^25) % 500 = 43) :=
sorry

end power_mod_remainder_l1490_149088


namespace distinct_patterns_4x4_3_shaded_l1490_149007

def num_distinct_patterns (n : ℕ) (shading : ℕ) : ℕ :=
  if n = 4 ∧ shading = 3 then 15
  else 0 -- Placeholder for other cases, not relevant for our problem

theorem distinct_patterns_4x4_3_shaded :
  num_distinct_patterns 4 3 = 15 :=
by {
  -- The proof would go here
  sorry
}

end distinct_patterns_4x4_3_shaded_l1490_149007


namespace real_roots_of_polynomial_l1490_149010

theorem real_roots_of_polynomial :
  (∀ x : ℝ, (x^10 + 36 * x^6 + 13 * x^2 = 13 * x^8 + x^4 + 36) ↔ 
    (x = 1 ∨ x = -1 ∨ x = 3 ∨ x = -3 ∨ x = 2 ∨ x = -2)) :=
by 
  sorry

end real_roots_of_polynomial_l1490_149010


namespace solution_set_equivalence_l1490_149047

theorem solution_set_equivalence (a : ℝ) : 
    (-1 < a ∧ a < 1) ∧ (3 * a^2 - 2 * a - 5 < 0) → 
    (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) :=
by
    sorry

end solution_set_equivalence_l1490_149047


namespace inequality_transform_l1490_149003

theorem inequality_transform {a b c d e : ℝ} (hab : a > b) (hb0 : b > 0) 
  (hcd : c < d) (hd0 : d < 0) (he : e < 0) : 
  e / (a - c)^2 > e / (b - d)^2 :=
by 
  sorry

end inequality_transform_l1490_149003


namespace g_of_5_l1490_149037

variable {g : ℝ → ℝ}
variable (h1 : ∀ x y : ℝ, 2 * x * g y = 3 * y * g x)
variable (h2 : g 10 = 15)

theorem g_of_5 : g 5 = 45 / 4 :=
  sorry

end g_of_5_l1490_149037


namespace det_matrix_4x4_l1490_149036

def matrix_4x4 : Matrix (Fin 4) (Fin 4) ℤ :=
  ![
    ![3, 0, 2, 0],
    ![2, 3, -1, 4],
    ![0, 4, -2, 3],
    ![5, 2, 0, 1]
  ]

theorem det_matrix_4x4 : Matrix.det matrix_4x4 = -84 :=
by
  sorry

end det_matrix_4x4_l1490_149036


namespace cellphone_gifting_l1490_149095

theorem cellphone_gifting (n m : ℕ) (h1 : n = 20) (h2 : m = 3) : 
    (Finset.range n).card * (Finset.range (n - 1)).card * (Finset.range (n - 2)).card = 6840 := by
  sorry

end cellphone_gifting_l1490_149095


namespace team_testing_equation_l1490_149048

variable (x : ℝ)

theorem team_testing_equation (h : x > 15) : (600 / x = 500 / (x - 15) * 0.9) :=
sorry

end team_testing_equation_l1490_149048


namespace Paco_cookies_left_l1490_149005

/-
Problem: Paco had 36 cookies. He gave 14 cookies to his friend and ate 10 cookies. How many cookies did Paco have left?
Solution: Paco has 12 cookies left.

To formally state this in Lean:
-/

def initial_cookies := 36
def cookies_given_away := 14
def cookies_eaten := 10

theorem Paco_cookies_left : initial_cookies - (cookies_given_away + cookies_eaten) = 12 :=
by
  sorry

/-
This theorem states that Paco has 12 cookies left given initial conditions.
-/

end Paco_cookies_left_l1490_149005


namespace outlet_pipe_empties_2_over_3_in_16_min_l1490_149083

def outlet_pipe_part_empty_in_t (t : ℕ) (part_per_8_min : ℚ) : ℚ :=
  (part_per_8_min / 8) * t

theorem outlet_pipe_empties_2_over_3_in_16_min (
  part_per_8_min : ℚ := 1/3
) : outlet_pipe_part_empty_in_t 16 part_per_8_min = 2/3 :=
by
  sorry

end outlet_pipe_empties_2_over_3_in_16_min_l1490_149083


namespace max_magnitude_z3_plus_3z_plus_2i_l1490_149082

open Complex

theorem max_magnitude_z3_plus_3z_plus_2i (z : ℂ) (h : Complex.abs z = 1) :
  ∃ M, M = 3 * Real.sqrt 3 ∧ ∀ (z : ℂ), Complex.abs z = 1 → Complex.abs (z^3 + 3 * z + 2 * Complex.I) ≤ M :=
by
  sorry

end max_magnitude_z3_plus_3z_plus_2i_l1490_149082


namespace investment_value_after_five_years_l1490_149052

theorem investment_value_after_five_years :
  let initial_investment := 10000
  let year1 := initial_investment * (1 - 0.05) * (1 + 0.02)
  let year2 := year1 * (1 + 0.10) * (1 + 0.02)
  let year3 := year2 * (1 + 0.04) * (1 + 0.02)
  let year4 := year3 * (1 - 0.03) * (1 + 0.02)
  let year5 := year4 * (1 + 0.08) * (1 + 0.02)
  year5 = 12570.99 :=
  sorry

end investment_value_after_five_years_l1490_149052


namespace find_xy_l1490_149068

-- Define the conditions as constants for clarity
def condition1 (x : ℝ) : Prop := 0.60 / x = 6 / 2
def condition2 (x y : ℝ) : Prop := x / y = 8 / 12

theorem find_xy (x y : ℝ) (hx : condition1 x) (hy : condition2 x y) : 
  x = 0.20 ∧ y = 0.30 :=
by
  sorry

end find_xy_l1490_149068


namespace ratio_of_rectangles_l1490_149058

theorem ratio_of_rectangles (p q : ℝ) (h1 : q ≠ 0) 
    (h2 : q^2 = 1/4 * (2 * p * q  - q^2)) : p / q = 5 / 2 := 
sorry

end ratio_of_rectangles_l1490_149058
