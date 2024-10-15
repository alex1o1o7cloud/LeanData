import Mathlib

namespace NUMINAMATH_GPT_tribe_leadership_choices_l288_28824

theorem tribe_leadership_choices :
  let members := 15
  let ways_to_choose_chief := members
  let remaining_after_chief := members - 1
  let ways_to_choose_supporting_chiefs := Nat.choose remaining_after_chief 2
  let remaining_after_supporting_chiefs := remaining_after_chief - 2
  let ways_to_choose_officers_A := Nat.choose remaining_after_supporting_chiefs 2
  let remaining_for_assistants_A := remaining_after_supporting_chiefs - 2
  let ways_to_choose_assistants_A := Nat.choose remaining_for_assistants_A 2 * Nat.choose (remaining_for_assistants_A - 2) 2
  let remaining_after_A := remaining_for_assistants_A - 2
  let ways_to_choose_officers_B := Nat.choose remaining_after_A 2
  let remaining_for_assistants_B := remaining_after_A - 2
  let ways_to_choose_assistants_B := Nat.choose remaining_for_assistants_B 2 * Nat.choose (remaining_for_assistants_B - 2) 2
  (ways_to_choose_chief * ways_to_choose_supporting_chiefs *
  ways_to_choose_officers_A * ways_to_choose_assistants_A *
  ways_to_choose_officers_B * ways_to_choose_assistants_B = 400762320000) := by
  sorry

end NUMINAMATH_GPT_tribe_leadership_choices_l288_28824


namespace NUMINAMATH_GPT_geometric_sequence_second_term_l288_28869

theorem geometric_sequence_second_term
  (first_term : ℕ) (fourth_term : ℕ) (r : ℕ)
  (h1 : first_term = 6)
  (h2 : first_term * r^3 = fourth_term)
  (h3 : fourth_term = 768) :
  first_term * r = 24 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_second_term_l288_28869


namespace NUMINAMATH_GPT_min_value_y_l288_28897

theorem min_value_y : ∀ (x : ℝ), ∃ y_min : ℝ, y_min = (x^2 + 16 * x + 10) ∧ ∀ (x' : ℝ), (x'^2 + 16 * x' + 10) ≥ y_min := 
by 
  sorry

end NUMINAMATH_GPT_min_value_y_l288_28897


namespace NUMINAMATH_GPT_votes_for_candidate_a_l288_28866

theorem votes_for_candidate_a :
  let total_votes : ℝ := 560000
  let percentage_invalid : ℝ := 0.15
  let percentage_candidate_a : ℝ := 0.85
  let valid_votes := (1 - percentage_invalid) * total_votes
  let votes_candidate_a := percentage_candidate_a * valid_votes
  votes_candidate_a = 404600 :=
by
  sorry

end NUMINAMATH_GPT_votes_for_candidate_a_l288_28866


namespace NUMINAMATH_GPT_ratio_boys_girls_l288_28886

variable (S G : ℕ)

theorem ratio_boys_girls (h : (2 / 3 : ℚ) * G = (1 / 5 : ℚ) * S) :
  (S - G) * 3 = 7 * G := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_boys_girls_l288_28886


namespace NUMINAMATH_GPT_months_in_season_l288_28809

/-- Definitions for conditions in the problem --/
def total_games_per_month : ℝ := 323.0
def total_games_season : ℝ := 5491.0

/-- The statement to be proven: The number of months in the season --/
theorem months_in_season (x : ℝ) (h : x = total_games_season / total_games_per_month) : x = 17.0 := by
  sorry

end NUMINAMATH_GPT_months_in_season_l288_28809


namespace NUMINAMATH_GPT_P_subset_Q_l288_28810

-- Define the set P
def P := {x : ℝ | 0 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 1}

-- Define the set Q
def Q := {x : ℝ | x ≤ 2}

-- Prove P ⊆ Q
theorem P_subset_Q : P ⊆ Q :=
by
  sorry

end NUMINAMATH_GPT_P_subset_Q_l288_28810


namespace NUMINAMATH_GPT_sum_of_cubes_l288_28887

theorem sum_of_cubes (a b t : ℝ) (h : a + b = t^2) : 2 * (a^3 + b^3) = (a * t)^2 + (b * t)^2 + (a * t - b * t)^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l288_28887


namespace NUMINAMATH_GPT_quotient_of_polynomial_l288_28837

theorem quotient_of_polynomial (x : ℤ) :
  (x^6 + 8) = (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) + 9 :=
by { sorry }

end NUMINAMATH_GPT_quotient_of_polynomial_l288_28837


namespace NUMINAMATH_GPT_measure_of_angle_Q_in_hexagon_l288_28836

theorem measure_of_angle_Q_in_hexagon :
  ∀ (Q : ℝ),
    (∃ (angles : List ℝ),
      angles = [134, 108, 122, 99, 87] ∧ angles.sum = 550) →
    180 * (6 - 2) - (134 + 108 + 122 + 99 + 87) = 170 → Q = 170 := by
  sorry

end NUMINAMATH_GPT_measure_of_angle_Q_in_hexagon_l288_28836


namespace NUMINAMATH_GPT_binomial_identity_l288_28825

theorem binomial_identity :
  (Nat.choose 16 6 = 8008) → (Nat.choose 16 7 = 11440) → (Nat.choose 16 8 = 12870) →
  Nat.choose 18 8 = 43758 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_binomial_identity_l288_28825


namespace NUMINAMATH_GPT_largest_integer_a_can_be_less_than_l288_28807

theorem largest_integer_a_can_be_less_than (a b : ℕ) (h1 : 9 < a) (h2 : 19 < b) (h3 : b < 31) (h4 : a / b = 2 / 3) :
  a < 21 :=
sorry

end NUMINAMATH_GPT_largest_integer_a_can_be_less_than_l288_28807


namespace NUMINAMATH_GPT_max_area_triangle_after_t_seconds_l288_28801

-- Define the problem conditions and question
def second_hand_rotation_rate : ℝ := 6 -- degrees per second
def minute_hand_rotation_rate : ℝ := 0.1 -- degrees per second
def perpendicular_angle : ℝ := 90 -- degrees

theorem max_area_triangle_after_t_seconds : 
  ∃ (t : ℝ), (second_hand_rotation_rate - minute_hand_rotation_rate) * t = perpendicular_angle ∧ t = 15 + 15 / 59 :=
by
  -- This is a statement of the proof problem; the proof itself is omitted.
  sorry

end NUMINAMATH_GPT_max_area_triangle_after_t_seconds_l288_28801


namespace NUMINAMATH_GPT_distance_in_scientific_notation_l288_28851

-- Definition for the number to be expressed in scientific notation
def distance : ℝ := 55000000

-- Expressing the number in scientific notation
def scientific_notation : ℝ := 5.5 * (10 ^ 7)

-- Theorem statement asserting the equality
theorem distance_in_scientific_notation : distance = scientific_notation :=
  by
  -- Proof not required here, so we leave it as sorry
  sorry

end NUMINAMATH_GPT_distance_in_scientific_notation_l288_28851


namespace NUMINAMATH_GPT_find_y_l288_28800

-- Given conditions
def x : Int := 129
def student_operation (y : Int) : Int := x * y - 148
def result : Int := 110

-- The theorem statement
theorem find_y :
  ∃ y : Int, student_operation y = result ∧ y = 2 := 
sorry

end NUMINAMATH_GPT_find_y_l288_28800


namespace NUMINAMATH_GPT_inscribed_cone_volume_l288_28889

theorem inscribed_cone_volume
  (H : ℝ) 
  (α : ℝ)
  (h_pos : 0 < H)
  (α_pos : 0 < α ∧ α < π / 2) :
  (1 / 12) * π * H ^ 3 * (Real.sin α) ^ 2 * (Real.sin (2 * α)) ^ 2 = 
  (1 / 3) * π * ((H * Real.sin α * Real.cos α / 2) ^ 2) * (H * (Real.sin α) ^ 2) :=
by sorry

end NUMINAMATH_GPT_inscribed_cone_volume_l288_28889


namespace NUMINAMATH_GPT_find_divisor_l288_28803

-- Define the problem specifications
def divisor_problem (D Q R d : ℕ) : Prop :=
  D = d * Q + R

-- The specific instance with given values
theorem find_divisor :
  divisor_problem 15968 89 37 179 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_divisor_l288_28803


namespace NUMINAMATH_GPT_bus_minibus_seats_l288_28806

theorem bus_minibus_seats (x y : ℕ) 
    (h1 : x = y + 20) 
    (h2 : 5 * x + 5 * y = 300) : 
    x = 40 ∧ y = 20 := 
by
  sorry

end NUMINAMATH_GPT_bus_minibus_seats_l288_28806


namespace NUMINAMATH_GPT_area_of_square_with_diagonal_30_l288_28835

theorem area_of_square_with_diagonal_30 :
  ∀ (d : ℝ), d = 30 → (d * d / 2) = 450 := 
by
  intros d h
  rw [h]
  sorry

end NUMINAMATH_GPT_area_of_square_with_diagonal_30_l288_28835


namespace NUMINAMATH_GPT_perimeter_regular_polygon_l288_28839

-- Definitions of the conditions
def side_length : ℕ := 8
def exterior_angle : ℕ := 72
def sum_of_exterior_angles : ℕ := 360

-- Number of sides calculation
def num_sides : ℕ := sum_of_exterior_angles / exterior_angle

-- Perimeter calculation
def perimeter (n : ℕ) (l : ℕ) : ℕ := n * l

-- Theorem statement
theorem perimeter_regular_polygon : perimeter num_sides side_length = 40 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_regular_polygon_l288_28839


namespace NUMINAMATH_GPT_markers_multiple_of_4_l288_28821

-- Definitions corresponding to conditions
def Lisa_has_12_coloring_books := 12
def Lisa_has_36_crayons := 36
def greatest_number_baskets := 4

-- Theorem statement
theorem markers_multiple_of_4
    (h1 : Lisa_has_12_coloring_books = 12)
    (h2 : Lisa_has_36_crayons = 36)
    (h3 : greatest_number_baskets = 4) :
    ∃ (M : ℕ), M % 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_markers_multiple_of_4_l288_28821


namespace NUMINAMATH_GPT_parallel_lines_perpendicular_lines_l288_28893

-- Define the lines
def l₁ (a : ℝ) (x y : ℝ) := (a - 1) * x + 2 * y + 1 = 0
def l₂ (a : ℝ) (x y : ℝ) := x + a * y + 3 = 0

-- The first proof statement: lines l₁ and l₂ are parallel
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → (a * (a - 1) - 2 = 0)) → (a = 2 ∨ a = -1) :=
by
  sorry

-- The second proof statement: lines l₁ and l₂ are perpendicular
theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → ((a - 1) * 1 + 2 * a = 0)) → (a = -1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_perpendicular_lines_l288_28893


namespace NUMINAMATH_GPT_price_difference_is_7_42_l288_28828

def total_cost : ℝ := 80.34
def shirt_price : ℝ := 36.46
def sweater_price : ℝ := total_cost - shirt_price
def price_difference : ℝ := sweater_price - shirt_price

theorem price_difference_is_7_42 : price_difference = 7.42 :=
  by
    sorry

end NUMINAMATH_GPT_price_difference_is_7_42_l288_28828


namespace NUMINAMATH_GPT_quadratic_root_sum_l288_28820

theorem quadratic_root_sum (k : ℝ) (h : k ≤ 1 / 2) : 
  ∃ (α β : ℝ), (α + β = 2 - 2 * k) ∧ (α^2 - 2 * (1 - k) * α + k^2 = 0) ∧ (β^2 - 2 * (1 - k) * β + k^2 = 0) ∧ (α + β ≥ 1) :=
sorry

end NUMINAMATH_GPT_quadratic_root_sum_l288_28820


namespace NUMINAMATH_GPT_silverware_probability_l288_28861

-- Defining the number of each type of silverware
def num_forks : ℕ := 8
def num_spoons : ℕ := 10
def num_knives : ℕ := 4
def total_silverware : ℕ := num_forks + num_spoons + num_knives
def num_remove : ℕ := 4

-- Proving the probability calculation
theorem silverware_probability :
  -- Calculation of the total number of ways to choose 4 pieces from 22
  let total_ways := Nat.choose total_silverware num_remove
  -- Calculation of ways to choose 2 forks from 8
  let ways_to_choose_forks := Nat.choose num_forks 2
  -- Calculation of ways to choose 1 spoon from 10
  let ways_to_choose_spoon := Nat.choose num_spoons 1
  -- Calculation of ways to choose 1 knife from 4
  let ways_to_choose_knife := Nat.choose num_knives 1
  -- Calculation of the number of favorable outcomes
  let favorable_outcomes := ways_to_choose_forks * ways_to_choose_spoon * ways_to_choose_knife
  -- Probability in simplified form
  let probability := (favorable_outcomes : ℚ) / total_ways
  probability = (32 : ℚ) / 209 :=
by
  sorry

end NUMINAMATH_GPT_silverware_probability_l288_28861


namespace NUMINAMATH_GPT_quadratic_roots_correct_l288_28899

def quadratic (b c : ℝ) (x : ℝ) : ℝ := x^2 + b * x + c

theorem quadratic_roots_correct (b c : ℝ) 
  (h₀ : quadratic b c (-2) = 5)
  (h₁ : quadratic b c (-1) = 0)
  (h₂ : quadratic b c 0 = -3)
  (h₃ : quadratic b c 1 = -4)
  (h₄ : quadratic b c 2 = -3)
  (h₅ : quadratic b c 4 = 5)
  : (quadratic b c (-1) = 0) ∧ (quadratic b c 3 = 0) :=
sorry

end NUMINAMATH_GPT_quadratic_roots_correct_l288_28899


namespace NUMINAMATH_GPT_M_inter_N_l288_28805

def M : Set ℝ := {y | ∃ x : ℝ, y = 2^(-x)}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

theorem M_inter_N : M ∩ N = {y | 0 < y ∧ y ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_M_inter_N_l288_28805


namespace NUMINAMATH_GPT_chord_length_intercepted_by_curve_l288_28873

theorem chord_length_intercepted_by_curve
(param_eqns : ∀ θ : ℝ, (x = 2 * Real.cos θ ∧ y = 1 + 2 * Real.sin θ))
(line_eqn : 3 * x - 4 * y - 1 = 0) :
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_chord_length_intercepted_by_curve_l288_28873


namespace NUMINAMATH_GPT_range_of_a_l288_28857

theorem range_of_a 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : r > 0) 
  (cos_le_zero : (3 * a - 9) / r ≤ 0) 
  (sin_gt_zero : (a + 2) / r > 0) : 
  -2 < a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l288_28857


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l288_28814

variables (a b : Line) (α β : Plane)

def Line : Type := sorry
def Plane : Type := sorry

-- Conditions: a and b are different lines, α and β are different planes
axiom diff_lines : a ≠ b
axiom diff_planes : α ≠ β

-- Perpendicular and parallel definitions
def perp (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry

-- Sufficient but not necessary condition
theorem sufficient_not_necessary_condition
  (h1 : perp a β)
  (h2 : parallel α β) :
  perp a α :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l288_28814


namespace NUMINAMATH_GPT_students_agreed_total_l288_28843

theorem students_agreed_total :
  let third_grade_agreed : ℕ := 154
  let fourth_grade_agreed : ℕ := 237
  third_grade_agreed + fourth_grade_agreed = 391 := 
by
  let third_grade_agreed : ℕ := 154
  let fourth_grade_agreed : ℕ := 237
  show third_grade_agreed + fourth_grade_agreed = 391
  sorry

end NUMINAMATH_GPT_students_agreed_total_l288_28843


namespace NUMINAMATH_GPT_car_truck_ratio_l288_28812

theorem car_truck_ratio (total_vehicles trucks cars : ℕ)
  (h1 : total_vehicles = 300)
  (h2 : trucks = 100)
  (h3 : cars + trucks = total_vehicles)
  (h4 : ∃ (k : ℕ), cars = k * trucks) : 
  cars / trucks = 2 :=
by
  sorry

end NUMINAMATH_GPT_car_truck_ratio_l288_28812


namespace NUMINAMATH_GPT_voice_of_china_signup_ways_l288_28845

theorem voice_of_china_signup_ways : 
  (2 * 2 * 2 = 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_voice_of_china_signup_ways_l288_28845


namespace NUMINAMATH_GPT_rationalized_expression_correct_A_B_C_D_E_sum_correct_l288_28871

noncomputable def A : ℤ := -18
noncomputable def B : ℤ := 2
noncomputable def C : ℤ := 30
noncomputable def D : ℤ := 5
noncomputable def E : ℤ := 428
noncomputable def expression := 3 / (2 * Real.sqrt 18 + 5 * Real.sqrt 20)
noncomputable def rationalized_form := (A * Real.sqrt B + C * Real.sqrt D) / E

theorem rationalized_expression_correct :
  rationalized_form = (18 * Real.sqrt 2 - 30 * Real.sqrt 5) / -428 :=
by
  sorry

theorem A_B_C_D_E_sum_correct :
  A + B + C + D + E = 447 :=
by
  sorry

end NUMINAMATH_GPT_rationalized_expression_correct_A_B_C_D_E_sum_correct_l288_28871


namespace NUMINAMATH_GPT_largest_number_l288_28867

theorem largest_number (a b c : ℤ) 
  (h_sum : a + b + c = 67)
  (h_diff1 : c - b = 7)
  (h_diff2 : b - a = 3)
  : c = 28 :=
sorry

end NUMINAMATH_GPT_largest_number_l288_28867


namespace NUMINAMATH_GPT_integer_solution_count_l288_28813

theorem integer_solution_count :
  ∃ n : ℕ, n = 10 ∧
  ∃ (x1 x2 x3 : ℕ), x1 + x2 + x3 = 15 ∧ (0 ≤ x1 ∧ x1 ≤ 5) ∧ (0 ≤ x2 ∧ x2 ≤ 6) ∧ (0 ≤ x3 ∧ x3 ≤ 7) := 
sorry

end NUMINAMATH_GPT_integer_solution_count_l288_28813


namespace NUMINAMATH_GPT_q_simplification_l288_28884

noncomputable def q (x a b c D : ℝ) : ℝ :=
  (x + a)^2 / ((a - b) * (a - c)) + 
  (x + b)^2 / ((b - a) * (b - c)) + 
  (x + c)^2 / ((c - a) * (c - b))

theorem q_simplification (a b c D x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  q x a b c D = a + b + c + 2 * x + 3 * D / (a + b + c) :=
by
  sorry

end NUMINAMATH_GPT_q_simplification_l288_28884


namespace NUMINAMATH_GPT_repeating_decimal_as_fraction_l288_28808

-- Define the repeating decimal x as .overline{37}
def x : ℚ := 37 / 99

-- The theorem we need to prove
theorem repeating_decimal_as_fraction : x = 37 / 99 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_as_fraction_l288_28808


namespace NUMINAMATH_GPT_range_of_a_l288_28811

theorem range_of_a (a : ℝ) :
  let A := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}
  let B := {x : ℝ | 5 < x}
  (A ∩ B = ∅) ↔ a ∈ {a : ℝ | a ≤ 2 ∨ a > 3} :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l288_28811


namespace NUMINAMATH_GPT_ice_cost_l288_28802

def people : Nat := 15
def ice_needed_per_person : Nat := 2
def pack_size : Nat := 10
def cost_per_pack : Nat := 3

theorem ice_cost : 
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  total_ice_needed = 30 ∧ number_of_packs = 3 ∧ number_of_packs * cost_per_pack = 9 :=
by
  let total_ice_needed := people * ice_needed_per_person
  let number_of_packs := total_ice_needed / pack_size
  have h1 : total_ice_needed = 30 := by sorry
  have h2 : number_of_packs = 3 := by sorry
  have h3 : number_of_packs * cost_per_pack = 9 := by sorry
  exact And.intro h1 (And.intro h2 h3)

end NUMINAMATH_GPT_ice_cost_l288_28802


namespace NUMINAMATH_GPT_angle_C_is_70_l288_28875

namespace TriangleAngleSum

def angle_sum_in_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

def sum_of_two_angles (A B : ℝ) : Prop :=
  A + B = 110

theorem angle_C_is_70 {A B C : ℝ} (h1 : angle_sum_in_triangle A B C) (h2 : sum_of_two_angles A B) : C = 70 :=
by
  sorry

end TriangleAngleSum

end NUMINAMATH_GPT_angle_C_is_70_l288_28875


namespace NUMINAMATH_GPT_heather_start_time_later_than_stacy_l288_28826

theorem heather_start_time_later_than_stacy :
  ∀ (distance_initial : ℝ) (H_speed : ℝ) (S_speed : ℝ) (H_distance_when_meet : ℝ),
    distance_initial = 5 ∧
    H_speed = 5 ∧
    S_speed = 6 ∧
    H_distance_when_meet = 1.1818181818181817 →
    ∃ (Δt : ℝ), Δt = 24 / 60 :=
by
  sorry

end NUMINAMATH_GPT_heather_start_time_later_than_stacy_l288_28826


namespace NUMINAMATH_GPT_product_of_roots_eq_neg35_l288_28834

theorem product_of_roots_eq_neg35 (x : ℝ) : 
  (x + 3) * (x - 5) = 20 → ∃ a b c : ℝ, a ≠ 0 ∧ a * x^2 + b * x + c = 0 ∧ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1 * x2 = -35 := 
by
  sorry

end NUMINAMATH_GPT_product_of_roots_eq_neg35_l288_28834


namespace NUMINAMATH_GPT_xy_value_l288_28853

theorem xy_value (x y : ℝ) (h : x * (x + y) = x ^ 2 + 12) : x * y = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_xy_value_l288_28853


namespace NUMINAMATH_GPT_three_digit_number_proof_l288_28849

noncomputable def is_prime (n : ℕ) : Prop := (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem three_digit_number_proof (H T U : ℕ) (h1 : H = 2 * T)
  (h2 : U = 2 * T^3)
  (h3 : is_prime (H + T + U))
  (h_digits : H < 10 ∧ T < 10 ∧ U < 10)
  (h_nonzero : T > 0) : H * 100 + T * 10 + U = 212 := 
by
  sorry

end NUMINAMATH_GPT_three_digit_number_proof_l288_28849


namespace NUMINAMATH_GPT_sequence_geometric_condition_l288_28894

theorem sequence_geometric_condition
  (a : ℕ → ℤ)
  (p q : ℤ)
  (h1 : a 1 = -1)
  (h2 : ∀ n, a (n + 1) = 2 * (a n - n + 3))
  (h3 : ∀ n, (a (n + 1) - p * (n + 1) + q) = 2 * (a n - p * n + q)) :
  a (Int.natAbs (p + q)) = 40 :=
sorry

end NUMINAMATH_GPT_sequence_geometric_condition_l288_28894


namespace NUMINAMATH_GPT_expected_messages_xiaoli_l288_28878

noncomputable def expected_greeting_messages (probs : List ℝ) (counts : List ℕ) : ℝ :=
  List.sum (List.zipWith (λ p c => p * c) probs counts)

theorem expected_messages_xiaoli :
  expected_greeting_messages [1, 0.8, 0.5, 0] [8, 15, 14, 3] = 27 :=
by
  -- The proof will use the expected value formula
  sorry

end NUMINAMATH_GPT_expected_messages_xiaoli_l288_28878


namespace NUMINAMATH_GPT_problem1_l288_28863

theorem problem1 (a : ℝ) (m n : ℕ) (h1 : a^m = 10) (h2 : a^n = 2) : a^(m - 2 * n) = 2.5 := by
  sorry

end NUMINAMATH_GPT_problem1_l288_28863


namespace NUMINAMATH_GPT_tom_rope_stories_l288_28872

/-- Define the conditions given in the problem. --/
def story_length : ℝ := 10
def rope_length : ℝ := 20
def loss_percentage : ℝ := 0.25
def pieces_of_rope : ℕ := 4

/-- Theorem to prove the number of stories Tom can lower the rope down. --/
theorem tom_rope_stories (story_length rope_length loss_percentage : ℝ) (pieces_of_rope : ℕ) : 
    story_length = 10 → 
    rope_length = 20 →
    loss_percentage = 0.25 →
    pieces_of_rope = 4 →
    pieces_of_rope * rope_length * (1 - loss_percentage) / story_length = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_tom_rope_stories_l288_28872


namespace NUMINAMATH_GPT_find_nat_pair_l288_28859

theorem find_nat_pair (a b : ℕ) (h₁ : a > 1) (h₂ : b > 1) (h₃ : a = 2^155) (h₄ : b = 3^65) : a^13 * b^31 = 6^2015 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_nat_pair_l288_28859


namespace NUMINAMATH_GPT_stock_price_is_500_l288_28879

-- Conditions
def income : ℝ := 1000
def dividend_rate : ℝ := 0.50
def investment : ℝ := 10000
def face_value : ℝ := 100

-- Theorem Statement
theorem stock_price_is_500 : 
  (dividend_rate * face_value / (investment / 1000)) = 500 := by
  sorry

end NUMINAMATH_GPT_stock_price_is_500_l288_28879


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_bn_sequence_sum_l288_28883

/-- 
  In an arithmetic sequence {a_n}, a_2 = 5 and a_6 = 21. 
  Prove the general formula for the nth term a_n and the sum of the first n terms S_n. 
-/
theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : a 2 = 5) (h2 : a 6 = 21) : 
  (∀ n, a n = 4 * n - 3) ∧ (∀ n, S n = n * (2 * n - 1)) := 
sorry

/--
  Given b_n = 2 / (S_n + 5 * n), prove the sum of the first n terms T_n for the sequence {b_n}.
-/
theorem bn_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℚ) (T : ℕ → ℚ) 
  (h1 : a 2 = 5) (h2 : a 6 = 21) 
  (ha : ∀ n, a n = 4 * n - 3) (hS : ∀ n, S n = n * (2 * n - 1)) 
  (hb : ∀ n, b n = 2 / (S n + 5 * n)) : 
  (∀ n, T n = 3 / 4 - 1 / (2 * (n + 1)) - 1 / (2 * (n + 2))) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_bn_sequence_sum_l288_28883


namespace NUMINAMATH_GPT_evaluate_expression_l288_28885

variable (b : ℝ) -- assuming b is a real number, (if b should be of different type, modify accordingly)

theorem evaluate_expression (y : ℝ) (h : y = b + 9) : y - b + 5 = 14 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l288_28885


namespace NUMINAMATH_GPT_children_ticket_cost_is_8_l288_28816

-- Defining the costs of different tickets
def adult_ticket_cost : ℕ := 11
def senior_ticket_cost : ℕ := 9
def total_tickets_cost : ℕ := 64

-- Number of tickets needed
def number_of_adult_tickets : ℕ := 2
def number_of_senior_tickets : ℕ := 2
def number_of_children_tickets : ℕ := 3

-- Defining the total cost equation using the price of children's tickets (C)
def total_cost (children_ticket_cost : ℕ) : ℕ :=
  number_of_adult_tickets * adult_ticket_cost +
  number_of_senior_tickets * senior_ticket_cost +
  number_of_children_tickets * children_ticket_cost

-- Statement to prove that the children's ticket cost is $8
theorem children_ticket_cost_is_8 : (C : ℕ) → total_cost C = total_tickets_cost → C = 8 :=
by
  intro C h
  sorry

end NUMINAMATH_GPT_children_ticket_cost_is_8_l288_28816


namespace NUMINAMATH_GPT_find_X_eq_A_l288_28898

variable {α : Type*}
variable (A X : Set α)

theorem find_X_eq_A (h : X ∩ A = X ∪ A) : X = A := by
  sorry

end NUMINAMATH_GPT_find_X_eq_A_l288_28898


namespace NUMINAMATH_GPT_mean_of_combined_set_l288_28830

theorem mean_of_combined_set
  (mean1 : ℕ → ℝ)
  (n1 : ℕ)
  (mean2 : ℕ → ℝ)
  (n2 : ℕ)
  (h1 : ∀ n1, mean1 n1 = 15)
  (h2 : ∀ n2, mean2 n2 = 26) :
  (n1 + n2) = 15 → 
  ((n1 * 15 + n2 * 26) / (n1 + n2)) = (313/15) :=
by
  sorry

end NUMINAMATH_GPT_mean_of_combined_set_l288_28830


namespace NUMINAMATH_GPT_prob_white_first_yellow_second_l288_28804

-- Defining the number of yellow and white balls
def yellow_balls : ℕ := 6
def white_balls : ℕ := 4

-- Defining the total number of balls
def total_balls : ℕ := yellow_balls + white_balls

-- Define the events A and B
def event_A : Prop := true -- event A: drawing a white ball first
def event_B : Prop := true -- event B: drawing a yellow ball second

-- Conditional probability P(B|A)
def prob_B_given_A : ℚ := 6 / (total_balls - 1)

-- Main theorem stating the proof problem
theorem prob_white_first_yellow_second : prob_B_given_A = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_prob_white_first_yellow_second_l288_28804


namespace NUMINAMATH_GPT_football_goals_in_fifth_match_l288_28877

theorem football_goals_in_fifth_match (G : ℕ) (h1 : (4 / 5 : ℝ) = (4 - G) / 4 + 0.3) : G = 2 :=
by
  sorry

end NUMINAMATH_GPT_football_goals_in_fifth_match_l288_28877


namespace NUMINAMATH_GPT_range_f_log_l288_28876

noncomputable def f : ℝ → ℝ := sorry

axiom f_even (x : ℝ) : f x = f (-x)
axiom f_increasing (x y : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ y) : f x ≤ f y
axiom f_at_1 : f 1 = 0

theorem range_f_log (x : ℝ) : f (Real.log x / Real.log (1 / 2)) > 0 ↔ (0 < x ∧ x < 1 / 2) ∨ (2 < x) :=
by
  sorry

end NUMINAMATH_GPT_range_f_log_l288_28876


namespace NUMINAMATH_GPT_modulo_sum_remainder_l288_28832

theorem modulo_sum_remainder (a b: ℤ) (k j: ℤ) 
  (h1 : a = 84 * k + 77) 
  (h2 : b = 120 * j + 113) :
  (a + b) % 42 = 22 := by
  sorry

end NUMINAMATH_GPT_modulo_sum_remainder_l288_28832


namespace NUMINAMATH_GPT_equivalent_after_eliminating_denominators_l288_28822

theorem equivalent_after_eliminating_denominators (x : ℝ) (h : 1 + 2 / (x - 1) = (x - 5) / (x - 3)) :
  (x - 1) * (x - 3) + 2 * (x - 3) = (x - 5) * (x - 1) :=
sorry

end NUMINAMATH_GPT_equivalent_after_eliminating_denominators_l288_28822


namespace NUMINAMATH_GPT_cost_of_paving_is_correct_l288_28823

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_metre : ℝ := 400
def area_of_rectangle (l: ℝ) (w: ℝ) : ℝ := l * w
def cost_of_paving_floor (area: ℝ) (rate: ℝ) : ℝ := area * rate

theorem cost_of_paving_is_correct
  (h_length: length = 5.5)
  (h_width: width = 3.75)
  (h_rate: rate_per_sq_metre = 400):
  cost_of_paving_floor (area_of_rectangle length width) rate_per_sq_metre = 8250 :=
  by {
    sorry
  }

end NUMINAMATH_GPT_cost_of_paving_is_correct_l288_28823


namespace NUMINAMATH_GPT_dig_time_comparison_l288_28838

open Nat

theorem dig_time_comparison :
  (3 * 420 / 9) - (5 * 40 / 2) = 40 :=
by
  sorry

end NUMINAMATH_GPT_dig_time_comparison_l288_28838


namespace NUMINAMATH_GPT_fraction_is_irreducible_l288_28864

theorem fraction_is_irreducible :
  (1 * 2 * 4 + 2 * 4 * 8 + 3 * 6 * 12 + 4 * 8 * 16 : ℚ) / 
   (1 * 3 * 9 + 2 * 6 * 18 + 3 * 9 * 27 + 4 * 12 * 36) = 8 / 27 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_is_irreducible_l288_28864


namespace NUMINAMATH_GPT_notebook_cost_l288_28846

theorem notebook_cost (s n c : ℕ) (h1 : s ≥ 19) (h2 : n > 2) (h3 : c > n) (h4 : s * c * n = 3969) : c = 27 :=
sorry

end NUMINAMATH_GPT_notebook_cost_l288_28846


namespace NUMINAMATH_GPT_area_difference_l288_28888

theorem area_difference (A B a b : ℝ) : (A * b) - (a * B) = A * b - a * B :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_area_difference_l288_28888


namespace NUMINAMATH_GPT_gardener_cabbages_increased_by_197_l288_28858

theorem gardener_cabbages_increased_by_197 (x : ℕ) (last_year_cabbages : ℕ := x^2) (increase : ℕ := 197) :
  (x + 1)^2 = x^2 + increase → (x + 1)^2 = 9801 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_gardener_cabbages_increased_by_197_l288_28858


namespace NUMINAMATH_GPT_one_of_a_b_c_is_one_l288_28833

theorem one_of_a_b_c_is_one (a b c : ℝ) (h1 : a * b * c = 1) (h2 : a + b + c = (1 / a) + (1 / b) + (1 / c)) :
  a = 1 ∨ b = 1 ∨ c = 1 :=
by
  sorry -- proof to be filled in

end NUMINAMATH_GPT_one_of_a_b_c_is_one_l288_28833


namespace NUMINAMATH_GPT_triangle_inequality_range_l288_28882

theorem triangle_inequality_range (x : ℝ) (h1 : 4 + 5 > x) (h2 : 4 + x > 5) (h3 : 5 + x > 4) :
  1 < x ∧ x < 9 := 
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_range_l288_28882


namespace NUMINAMATH_GPT_real_part_of_z_is_neg3_l288_28827

noncomputable def z : ℂ := (1 + 2 * Complex.I) ^ 2

theorem real_part_of_z_is_neg3 : z.re = -3 := by
  sorry

end NUMINAMATH_GPT_real_part_of_z_is_neg3_l288_28827


namespace NUMINAMATH_GPT_TeresaTotalMarks_l288_28874

/-- Teresa's scores in various subjects as given conditions -/
def ScienceScore := 70
def MusicScore := 80
def SocialStudiesScore := 85
def PhysicsScore := 1 / 2 * MusicScore

/-- Total marks Teresa scored in all the subjects -/
def TotalMarks := ScienceScore + MusicScore + SocialStudiesScore + PhysicsScore

/-- Proof statement: The total marks scored by Teresa in all subjects is 275. -/
theorem TeresaTotalMarks : TotalMarks = 275 := by
  sorry

end NUMINAMATH_GPT_TeresaTotalMarks_l288_28874


namespace NUMINAMATH_GPT_who_is_next_to_Boris_l288_28831

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end NUMINAMATH_GPT_who_is_next_to_Boris_l288_28831


namespace NUMINAMATH_GPT_positions_after_317_moves_l288_28870

-- Define positions for the cat and dog
inductive ArchPosition
| North | East | South | West
deriving DecidableEq

inductive PathPosition
| North | Northeast | East | Southeast | South | Southwest
deriving DecidableEq

-- Define the movement function for cat and dog
def cat_position (n : Nat) : ArchPosition :=
  match n % 4 with
  | 0 => ArchPosition.North
  | 1 => ArchPosition.East
  | 2 => ArchPosition.South
  | _ => ArchPosition.West

def dog_position (n : Nat) : PathPosition :=
  match n % 6 with
  | 0 => PathPosition.North
  | 1 => PathPosition.Northeast
  | 2 => PathPosition.East
  | 3 => PathPosition.Southeast
  | 4 => PathPosition.South
  | _ => PathPosition.Southwest

-- Theorem statement to prove the positions after 317 moves
theorem positions_after_317_moves :
  cat_position 317 = ArchPosition.North ∧
  dog_position 317 = PathPosition.South :=
by
  sorry

end NUMINAMATH_GPT_positions_after_317_moves_l288_28870


namespace NUMINAMATH_GPT_rosa_parks_food_drive_l288_28860

theorem rosa_parks_food_drive :
  ∀ (total_students students_collected_12_cans students_collected_none students_remaining total_cans cans_collected_first_group total_cans_first_group total_cans_last_group cans_per_student_last_group : ℕ),
    total_students = 30 →
    students_collected_12_cans = 15 →
    students_collected_none = 2 →
    students_remaining = total_students - students_collected_12_cans - students_collected_none →
    total_cans = 232 →
    cans_collected_first_group = 12 →
    total_cans_first_group = students_collected_12_cans * cans_collected_first_group →
    total_cans_last_group = total_cans - total_cans_first_group →
    cans_per_student_last_group = total_cans_last_group / students_remaining →
    cans_per_student_last_group = 4 :=
by
  intros total_students students_collected_12_cans students_collected_none students_remaining total_cans cans_collected_first_group total_cans_first_group total_cans_last_group cans_per_student_last_group
  sorry

end NUMINAMATH_GPT_rosa_parks_food_drive_l288_28860


namespace NUMINAMATH_GPT_instantaneous_velocity_at_3_l288_28829

noncomputable def displacement (t : ℝ) : ℝ := 
  - (1 / 3) * t^3 + 2 * t^2 - 5

theorem instantaneous_velocity_at_3 : 
  (deriv displacement 3 = 3) :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_3_l288_28829


namespace NUMINAMATH_GPT_evaluate_expression_l288_28892

def f (x : ℕ) : ℕ := 4 * x + 2
def g (x : ℕ) : ℕ := 3 * x + 4

theorem evaluate_expression : f (g (f 3)) = 186 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l288_28892


namespace NUMINAMATH_GPT_cd_total_l288_28855

theorem cd_total :
  ∀ (Kristine Dawn Mark Alice : ℕ),
  Dawn = 10 →
  Kristine = Dawn + 7 →
  Mark = 2 * Kristine →
  Alice = (Kristine + Mark) - 5 →
  (Dawn + Kristine + Mark + Alice) = 107 :=
by
  intros Kristine Dawn Mark Alice hDawn hKristine hMark hAlice
  rw [hDawn, hKristine, hMark, hAlice]
  sorry

end NUMINAMATH_GPT_cd_total_l288_28855


namespace NUMINAMATH_GPT_speed_of_faster_train_l288_28818

-- Definitions based on the conditions.
def length_train_1 : ℝ := 180
def length_train_2 : ℝ := 360
def time_to_cross : ℝ := 21.598272138228943
def speed_slow_train_kmph : ℝ := 30
def speed_fast_train_kmph : ℝ := 60

-- The theorem that needs to be proven.
theorem speed_of_faster_train :
  (length_train_1 + length_train_2) / time_to_cross * 3.6 = speed_slow_train_kmph + speed_fast_train_kmph :=
sorry

end NUMINAMATH_GPT_speed_of_faster_train_l288_28818


namespace NUMINAMATH_GPT_problem_inequality_l288_28854

theorem problem_inequality 
  (a b c d : ℝ)
  (h1 : d > 0)
  (h2 : a ≥ b)
  (h3 : b ≥ c)
  (h4 : c ≥ d)
  (h5 : a * b * c * d = 1) : 
  (1 / (1 + a)) + (1 / (1 + b)) + (1 / (1 + c)) ≥ 3 / (1 + (a * b * c) ^ (1 / 3)) :=
sorry

end NUMINAMATH_GPT_problem_inequality_l288_28854


namespace NUMINAMATH_GPT_tan_double_angle_difference_l288_28819

variable {α β : Real}

theorem tan_double_angle_difference (h1 : Real.tan α = 1 / 2) (h2 : Real.tan (α - β) = 1 / 5) :
  Real.tan (2 * α - β) = 7 / 9 := 
sorry

end NUMINAMATH_GPT_tan_double_angle_difference_l288_28819


namespace NUMINAMATH_GPT_ratio_6_3_to_percent_l288_28847

theorem ratio_6_3_to_percent : (6 / 3) * 100 = 200 := by
  sorry

end NUMINAMATH_GPT_ratio_6_3_to_percent_l288_28847


namespace NUMINAMATH_GPT_cottonwood_fiber_diameter_in_scientific_notation_l288_28891

theorem cottonwood_fiber_diameter_in_scientific_notation:
  (∃ (a : ℝ) (n : ℤ), 0.0000108 = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10) → (0.0000108 = 1.08 * 10 ^ (-5)) :=
by
  sorry

end NUMINAMATH_GPT_cottonwood_fiber_diameter_in_scientific_notation_l288_28891


namespace NUMINAMATH_GPT_magnitude_of_T_l288_28856

theorem magnitude_of_T : 
  let i := Complex.I
  let T := 3 * ((1 + i) ^ 15 - (1 - i) ^ 15)
  Complex.abs T = 768 := by
  sorry

end NUMINAMATH_GPT_magnitude_of_T_l288_28856


namespace NUMINAMATH_GPT_combined_distance_l288_28848

-- Definitions based on the conditions
def JulienDailyDistance := 50
def SarahDailyDistance := 2 * JulienDailyDistance
def JamirDailyDistance := SarahDailyDistance + 20
def Days := 7

-- Combined weekly distances
def JulienWeeklyDistance := JulienDailyDistance * Days
def SarahWeeklyDistance := SarahDailyDistance * Days
def JamirWeeklyDistance := JamirDailyDistance * Days

-- Theorem statement with the combined distance
theorem combined_distance :
  JulienWeeklyDistance + SarahWeeklyDistance + JamirWeeklyDistance = 1890 := by
  sorry

end NUMINAMATH_GPT_combined_distance_l288_28848


namespace NUMINAMATH_GPT_original_couch_price_l288_28842

def chair_price : ℝ := sorry
def table_price := 3 * chair_price
def couch_price := 5 * table_price
def bookshelf_price := 0.5 * couch_price

def discounted_chair_price := 0.8 * chair_price
def discounted_couch_price := 0.9 * couch_price
def total_price_before_tax := discounted_chair_price + table_price + discounted_couch_price + bookshelf_price
def total_price_after_tax := total_price_before_tax * 1.08

theorem original_couch_price (budget : ℝ) (h_budget : budget = 900) : 
  total_price_after_tax = budget → couch_price = 503.85 :=
by
  sorry

end NUMINAMATH_GPT_original_couch_price_l288_28842


namespace NUMINAMATH_GPT_binary_representation_88_l288_28862

def binary_representation (n : Nat) : String := sorry

theorem binary_representation_88 : binary_representation 88 = "1011000" := sorry

end NUMINAMATH_GPT_binary_representation_88_l288_28862


namespace NUMINAMATH_GPT_max_cables_cut_l288_28865

theorem max_cables_cut (computers cables clusters : ℕ) (h_computers : computers = 200) (h_cables : cables = 345) (h_clusters : clusters = 8) :
  ∃ k : ℕ, k = cables - (computers - clusters + 1) ∧ k = 153 :=
by
  sorry

end NUMINAMATH_GPT_max_cables_cut_l288_28865


namespace NUMINAMATH_GPT_complement_of_intersection_l288_28852

-- Definitions of the sets M and N
def M : Set ℝ := { x | x ≥ 2 }
def N : Set ℝ := { x | x < 3 }

-- Definition of the intersection of M and N
def M_inter_N : Set ℝ := { x | 2 ≤ x ∧ x < 3 }

-- Definition of the complement of M ∩ N in ℝ
def complement_M_inter_N : Set ℝ := { x | x < 2 ∨ x ≥ 3 }

-- The theorem to be proved
theorem complement_of_intersection :
  (M_inter_Nᶜ) = complement_M_inter_N :=
by sorry

end NUMINAMATH_GPT_complement_of_intersection_l288_28852


namespace NUMINAMATH_GPT_gcd_of_36_and_60_is_12_l288_28881

theorem gcd_of_36_and_60_is_12 :
  Nat.gcd 36 60 = 12 :=
sorry

end NUMINAMATH_GPT_gcd_of_36_and_60_is_12_l288_28881


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l288_28868

variable (a : ℝ)

theorem sufficient_not_necessary_condition (h1 : a > 2) : (1 / a < 1 / 2) ↔ (a > 2 ∨ a < 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l288_28868


namespace NUMINAMATH_GPT_total_students_l288_28850

theorem total_students (initial_candies leftover_candies girls boys : ℕ) (h1 : initial_candies = 484)
  (h2 : leftover_candies = 4) (h3 : boys = girls + 3) (h4 : (2 * girls + boys) * (2 * girls + boys) = initial_candies - leftover_candies) :
  2 * girls + boys = 43 :=
  sorry

end NUMINAMATH_GPT_total_students_l288_28850


namespace NUMINAMATH_GPT_precious_stones_l288_28840

variable (total_amount : ℕ) (price_per_stone : ℕ) (number_of_stones : ℕ)

theorem precious_stones (h1 : total_amount = 14280) (h2 : price_per_stone = 1785) : number_of_stones = 8 :=
by
  sorry

end NUMINAMATH_GPT_precious_stones_l288_28840


namespace NUMINAMATH_GPT_karl_total_income_is_53_l288_28895

noncomputable def compute_income (tshirt_price pant_price skirt_price sold_tshirts sold_pants sold_skirts sold_refurbished_tshirts: ℕ) : ℝ :=
  let tshirt_income := 2 * tshirt_price
  let pant_income := sold_pants * pant_price
  let skirt_income := sold_skirts * skirt_price
  let refurbished_tshirt_price := (tshirt_price : ℝ) / 2
  let refurbished_tshirt_income := sold_refurbished_tshirts * refurbished_tshirt_price
  tshirt_income + pant_income + skirt_income + refurbished_tshirt_income

theorem karl_total_income_is_53 : compute_income 5 4 6 2 1 4 6 = 53 := by
  sorry

end NUMINAMATH_GPT_karl_total_income_is_53_l288_28895


namespace NUMINAMATH_GPT_observation_count_l288_28817

theorem observation_count (mean_before mean_after : ℝ) 
  (wrong_value : ℝ) (correct_value : ℝ) (n : ℝ) :
  mean_before = 36 →
  correct_value = 60 →
  wrong_value = 23 →
  mean_after = 36.5 →
  n = 74 :=
by
  intros h_mean_before h_correct_value h_wrong_value h_mean_after
  sorry

end NUMINAMATH_GPT_observation_count_l288_28817


namespace NUMINAMATH_GPT_probability_drop_l288_28890

open Real

noncomputable def probability_of_oil_drop_falling_in_hole (c : ℝ) : ℝ :=
  (0.25 * c^2) / (π * (c^2 / 4))

theorem probability_drop (c : ℝ) (hc : c > 0) : 
  probability_of_oil_drop_falling_in_hole c = 0.25 / π :=
by
  sorry

end NUMINAMATH_GPT_probability_drop_l288_28890


namespace NUMINAMATH_GPT_no_m_for_necessary_and_sufficient_condition_m_geq_3_for_necessary_condition_l288_28815

def P (x : ℝ) : Prop := x ^ 2 - 8 * x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem no_m_for_necessary_and_sufficient_condition :
  ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S x m :=
by sorry

theorem m_geq_3_for_necessary_condition :
  ∃ m : ℝ, (m ≥ 3) ∧ ∀ x : ℝ, S x m → P x :=
by sorry

end NUMINAMATH_GPT_no_m_for_necessary_and_sufficient_condition_m_geq_3_for_necessary_condition_l288_28815


namespace NUMINAMATH_GPT_arccos_gt_arctan_l288_28841

theorem arccos_gt_arctan (x : ℝ) (h : -1 ≤ x ∧ x < 1/2) : Real.arccos x > Real.arctan x :=
sorry

end NUMINAMATH_GPT_arccos_gt_arctan_l288_28841


namespace NUMINAMATH_GPT_old_toilet_water_per_flush_correct_l288_28880

noncomputable def old_toilet_water_per_flush (water_saved : ℕ) (flushes_per_day : ℕ) (days_in_june : ℕ) (reduction_percentage : ℚ) : ℚ :=
  let total_flushes := flushes_per_day * days_in_june
  let water_saved_per_flush := water_saved / total_flushes
  let reduction_factor := reduction_percentage
  let original_water_per_flush := water_saved_per_flush / (1 - reduction_factor)
  original_water_per_flush

theorem old_toilet_water_per_flush_correct :
  old_toilet_water_per_flush 1800 15 30 (80 / 100) = 5 := by
  sorry

end NUMINAMATH_GPT_old_toilet_water_per_flush_correct_l288_28880


namespace NUMINAMATH_GPT_find_a_l288_28896

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + a

theorem find_a :
  (∀ x : ℝ, 0 ≤ f x a) ∧ (∀ y : ℝ, ∃ x : ℝ, y = f x a) ↔ a = 1 := by
  sorry

end NUMINAMATH_GPT_find_a_l288_28896


namespace NUMINAMATH_GPT_more_candidates_selected_l288_28844

theorem more_candidates_selected (n : ℕ) (pA pB : ℝ) 
  (hA : pA = 0.06) (hB : pB = 0.07) (hN : n = 8200) :
  (pB * n - pA * n) = 82 :=
by
  sorry

end NUMINAMATH_GPT_more_candidates_selected_l288_28844
