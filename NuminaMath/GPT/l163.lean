import Mathlib

namespace range_of_a_l163_163606

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ 1 → (x^2 + a * x + 9) ≥ 0) : a ≥ -6 := 
sorry

end range_of_a_l163_163606


namespace ivan_running_distance_l163_163530

theorem ivan_running_distance (x MondayDistance TuesdayDistance WednesdayDistance ThursdayDistance FridayDistance : ℝ) 
  (h1 : MondayDistance = x)
  (h2 : TuesdayDistance = 2 * x)
  (h3 : WednesdayDistance = x)
  (h4 : ThursdayDistance = (1 / 2) * x)
  (h5 : FridayDistance = x)
  (hShortest : ThursdayDistance = 5) :
  MondayDistance + TuesdayDistance + WednesdayDistance + ThursdayDistance + FridayDistance = 55 :=
by
  sorry

end ivan_running_distance_l163_163530


namespace tangent_circles_BC_length_l163_163107

theorem tangent_circles_BC_length
  (rA rB : ℝ) (A B C : ℝ × ℝ) (distAB distAC : ℝ) 
  (hAB : rA + rB = distAB)
  (hAC : distAB + 2 = distAC) 
  (h_sim : ∀ AD BE BC AC : ℝ, AD / BE = rA / rB → BC / AC = rB / rA) :
  BC = 52 / 7 := sorry

end tangent_circles_BC_length_l163_163107


namespace inspection_arrangements_l163_163837

-- Definitions based on conditions
def liberal_arts_classes : ℕ := 2
def science_classes : ℕ := 3
def num_students (classes : ℕ) : ℕ := classes

-- Main theorem statement
theorem inspection_arrangements (liberal_arts_classes science_classes : ℕ)
  (h1: liberal_arts_classes = 2) (h2: science_classes = 3) : 
  num_students liberal_arts_classes * num_students science_classes = 24 :=
by {
  -- Given there are 2 liberal arts classes and 3 science classes,
  -- there are exactly 24 ways to arrange the inspections as per the conditions provided.
  sorry
}

end inspection_arrangements_l163_163837


namespace carmen_parsley_left_l163_163358

theorem carmen_parsley_left (plates_whole_sprig : ℕ) (plates_half_sprig : ℕ) (initial_sprigs : ℕ) :
  plates_whole_sprig = 8 →
  plates_half_sprig = 12 →
  initial_sprigs = 25 →
  initial_sprigs - (plates_whole_sprig + plates_half_sprig / 2) = 11 := by
  intros
  sorry

end carmen_parsley_left_l163_163358


namespace solve_for_x_l163_163066

variable {x y : ℝ}

theorem solve_for_x (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x) : x = 3 / 2 := by
  sorry

end solve_for_x_l163_163066


namespace average_of_four_numbers_l163_163443

theorem average_of_four_numbers (a b c d : ℝ) 
  (h1 : b + c + d = 24) (h2 : a + c + d = 36)
  (h3 : a + b + d = 28) (h4 : a + b + c = 32) :
  (a + b + c + d) / 4 = 10 := 
sorry

end average_of_four_numbers_l163_163443


namespace wall_length_l163_163686

theorem wall_length
    (brick_length brick_width brick_height : ℝ)
    (wall_height wall_width : ℝ)
    (num_bricks : ℕ)
    (wall_length_cm : ℝ)
    (h_brick_volume : brick_length * brick_width * brick_height = 1687.5)
    (h_wall_volume :
        wall_length_cm * wall_height * wall_width
        = (brick_length * brick_width * brick_height) * num_bricks)
    (h_wall_height : wall_height = 600)
    (h_wall_width : wall_width = 22.5)
    (h_num_bricks : num_bricks = 7200) :
    wall_length_cm / 100 = 9 := 
by
  sorry

end wall_length_l163_163686


namespace matchstick_triangles_l163_163842

/-- Using 12 equal-length matchsticks, it is possible to form an isosceles triangle, an equilateral triangle, and a right-angled triangle without breaking or overlapping the matchsticks. --/
theorem matchstick_triangles :
  ∃ a b c : ℕ, a + b + c = 12 ∧ (a = b ∨ b = c ∨ a = c) ∧ (a * a + b * b = c * c ∨ a = b ∧ b = c) :=
by
  sorry

end matchstick_triangles_l163_163842


namespace problem_part1_problem_part2_l163_163298

variable (α : Real)
variable (h : Real.tan α = 1 / 2)

theorem problem_part1 : 
  (2 * Real.cos α - 3 * Real.sin α) / (3 * Real.cos α + 4 * Real.sin α) = 1 / 10 := sorry

theorem problem_part2 : 
  Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 11 / 5 := sorry

end problem_part1_problem_part2_l163_163298


namespace cube_union_volume_is_correct_cube_union_surface_area_is_correct_l163_163676

noncomputable def cubeUnionVolume : ℝ :=
  let cubeVolume := 1
  let intersectionVolume := 1 / 4
  cubeVolume * 2 - intersectionVolume

theorem cube_union_volume_is_correct :
  cubeUnionVolume = 5 / 4 := sorry

noncomputable def cubeUnionSurfaceArea : ℝ :=
  2 * (6 * (1 / 4) + 6 * (1 / 4 / 4))

theorem cube_union_surface_area_is_correct :
  cubeUnionSurfaceArea = 15 / 2 := sorry

end cube_union_volume_is_correct_cube_union_surface_area_is_correct_l163_163676


namespace fifteen_percent_of_x_is_ninety_l163_163790

theorem fifteen_percent_of_x_is_ninety (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end fifteen_percent_of_x_is_ninety_l163_163790


namespace coefficient_of_pi_x_over_5_l163_163249

-- Definition of the function where we find the coefficient
def coefficient_of_fraction (expr : ℝ) : ℝ := sorry

-- Statement with proof obligation
theorem coefficient_of_pi_x_over_5 :
  coefficient_of_fraction (π * x / 5) = π / 5 :=
sorry

end coefficient_of_pi_x_over_5_l163_163249


namespace determine_g_l163_163084

variable (g : ℕ → ℕ)

theorem determine_g (h : ∀ x, g (x + 1) = 2 * x + 3) : ∀ x, g x = 2 * x + 1 :=
by
  sorry

end determine_g_l163_163084


namespace johns_beef_order_l163_163621

theorem johns_beef_order (B : ℕ)
  (h1 : 8 * B + 6 * (2 * B) = 14000) :
  B = 1000 :=
by
  sorry

end johns_beef_order_l163_163621


namespace election_votes_l163_163275

theorem election_votes (T : ℝ) (Vf Va Vn : ℝ)
  (h1 : Va = 0.375 * T)
  (h2 : Vn = 0.125 * T)
  (h3 : Vf = Va + 78)
  (h4 : T = Vf + Va + Vn) :
  T = 624 :=
by
  sorry

end election_votes_l163_163275


namespace find_roots_l163_163133

theorem find_roots (a b c d x : ℝ) (h₁ : a + d = 2015) (h₂ : b + c = 2015) (h₃ : a ≠ c) :
  (x - a) * (x - b) = (x - c) * (x - d) → x = 0 := 
sorry

end find_roots_l163_163133


namespace part_I_intersection_part_I_union_complements_part_II_range_l163_163844

namespace MathProof

-- Definitions of the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a - 1}

-- Prove that the intersection of A and B is {x | 3 < x ∧ x < 6}
theorem part_I_intersection : A ∩ B = {x | 3 < x ∧ x < 6} := sorry

-- Prove that the union of the complements of A and B is {x | x ≤ 3 ∨ x ≥ 6}
theorem part_I_union_complements : (Aᶜ ∪ Bᶜ) = {x | x ≤ 3 ∨ x ≥ 6} := sorry

-- Prove the range of a such that C is a subset of B and B union C equals B
theorem part_II_range (a : ℝ) : B ∪ C a = B → (a ≤ 1 ∨ 2 ≤ a ∧ a ≤ 5) := sorry

end MathProof

end part_I_intersection_part_I_union_complements_part_II_range_l163_163844


namespace rectangle_area_12_l163_163025

theorem rectangle_area_12
  (L W : ℝ)
  (h1 : L + W = 7)
  (h2 : L^2 + W^2 = 25) :
  L * W = 12 :=
by
  sorry

end rectangle_area_12_l163_163025


namespace trihedral_sum_of_angles_le_sum_of_plane_angles_trihedral_sum_of_angles_ge_half_sum_of_plane_angles_l163_163729

-- Part a
theorem trihedral_sum_of_angles_le_sum_of_plane_angles
  (α β γ : ℝ) (ASB BSC CSA : ℝ)
  (h1 : α ≤ ASB)
  (h2 : β ≤ BSC)
  (h3 : γ ≤ CSA) :
  α + β + γ ≤ ASB + BSC + CSA :=
sorry

-- Part b
theorem trihedral_sum_of_angles_ge_half_sum_of_plane_angles
  (α_S β_S γ_S : ℝ) (ASB BSC CSA : ℝ) 
  (h_acute : ASB < (π / 2) ∧ BSC < (π / 2) ∧ CSA < (π / 2))
  (h1 : α_S ≥ (1/2) * ASB)
  (h2 : β_S ≥ (1/2) * BSC)
  (h3 : γ_S ≥ (1/2) * CSA) :
  α_S + β_S + γ_S ≥ (1/2) * (ASB + BSC + CSA) :=
sorry

end trihedral_sum_of_angles_le_sum_of_plane_angles_trihedral_sum_of_angles_ge_half_sum_of_plane_angles_l163_163729


namespace problem_solution_l163_163411

theorem problem_solution (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 6 = 6 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x ^ Real.sqrt 3 + y ^ Real.sqrt 3 = 189 :=
sorry

end problem_solution_l163_163411


namespace meaningful_range_l163_163518

noncomputable def isMeaningful (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (x - 2 ≠ 0)

theorem meaningful_range (x : ℝ) : isMeaningful x ↔ (x ≥ -1) ∧ (x ≠ 2) :=
by
  sorry

end meaningful_range_l163_163518


namespace quadratic_range_extrema_l163_163975

def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 2

theorem quadratic_range_extrema :
  let y := quadratic
  ∃ x_max x_min,
    (x_min = -2 ∧ y x_min = -2) ∧
    (x_max = -2 ∧ y x_max = 14 ∨ x_max = 5 ∧ y x_max = 7) := 
by
  sorry

end quadratic_range_extrema_l163_163975


namespace theater_rows_l163_163452

theorem theater_rows (R : ℕ) (h1 : R < 30 → ∃ r : ℕ, r < R ∧ r * 2 ≥ 30) (h2 : R ≥ 29 → 26 + 3 ≤ R) : R = 29 :=
by
  sorry

end theater_rows_l163_163452


namespace integer_solution_pair_l163_163241

theorem integer_solution_pair (x y : ℤ) (h : x^2 + x * y = y^2) : (x = 0 ∧ y = 0) :=
by
  sorry

end integer_solution_pair_l163_163241


namespace possible_values_x_l163_163470

variable (a b x : ℕ)

theorem possible_values_x (h1 : a + b = 20)
                          (h2 : a * x + b * 3 = 109) :
    x = 10 ∨ x = 52 :=
sorry

end possible_values_x_l163_163470


namespace find_other_number_l163_163362

theorem find_other_number (A : ℕ) (hcf_cond : Nat.gcd A 48 = 12) (lcm_cond : Nat.lcm A 48 = 396) : A = 99 := by
    sorry

end find_other_number_l163_163362


namespace total_students_left_l163_163269

-- Definitions for given conditions
def initialBoys := 14
def initialGirls := 10
def boysDropOut := 4
def girlsDropOut := 3

-- The proof problem statement
theorem total_students_left : 
  initialBoys - boysDropOut + (initialGirls - girlsDropOut) = 17 := 
by
  sorry

end total_students_left_l163_163269


namespace newspaper_target_l163_163816

theorem newspaper_target (total_collected_2_weeks : Nat) (needed_more : Nat) (sections : Nat) (kilos_per_section_2_weeks : Nat)
  (h1 : sections = 6)
  (h2 : kilos_per_section_2_weeks = 280)
  (h3 : total_collected_2_weeks = sections * kilos_per_section_2_weeks)
  (h4 : needed_more = 320)
  : total_collected_2_weeks + needed_more = 2000 :=
by
  sorry

end newspaper_target_l163_163816


namespace solve_quadratics_l163_163079

theorem solve_quadratics :
  (∃ x : ℝ, x^2 + 5 * x - 24 = 0) ∧ (∃ y, y^2 + 5 * y - 24 = 0) ∧
  (∃ z : ℝ, 3 * z^2 + 2 * z - 4 = 0) ∧ (∃ w, 3 * w^2 + 2 * w - 4 = 0) :=
by {
  sorry
}

end solve_quadratics_l163_163079


namespace max_repeating_sequence_length_l163_163828

theorem max_repeating_sequence_length (p q n α β d : ℕ) (h_prime: Nat.gcd p q = 1)
  (hq : q = (2 ^ α) * (5 ^ β) * d) (hd_coprime: Nat.gcd d 10 = 1) (h_repeat: 10 ^ n ≡ 1 [MOD d]) :
  ∃ s, s ≤ n * (10 ^ n - 1) ∧ (10 ^ s ≡ 1 [MOD d^2]) :=
by
  sorry

end max_repeating_sequence_length_l163_163828


namespace value_of_m_l163_163972

theorem value_of_m (a b c : ℤ) (m : ℤ) (h1 : 0 ≤ m) (h2 : m ≤ 26) 
  (h3 : (a + b + c) % 27 = m) (h4 : ((a - b) * (b - c) * (c - a)) % 27 = m) : 
  m = 0 :=
  by
  -- Proof is to be filled in
  sorry

end value_of_m_l163_163972


namespace find_n_l163_163270

variable {a_n : ℕ → ℤ}
variable (a2 : ℤ) (an : ℤ) (d : ℤ) (n : ℕ)

def arithmetic_sequence (a2 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a2 + (n - 2) * d

theorem find_n 
  (h1 : a2 = 12)
  (h2 : an = -20)
  (h3 : d = -2)
  : n = 18 := by
  sorry

end find_n_l163_163270


namespace percentage_of_students_owning_only_cats_is_10_percent_l163_163231

def total_students : ℕ := 500
def cat_owners : ℕ := 75
def dog_owners : ℕ := 150
def both_cat_and_dog_owners : ℕ := 25
def only_cat_owners : ℕ := cat_owners - both_cat_and_dog_owners
def percent_owning_only_cats : ℚ := (only_cat_owners * 100) / total_students

theorem percentage_of_students_owning_only_cats_is_10_percent : percent_owning_only_cats = 10 := by
  sorry

end percentage_of_students_owning_only_cats_is_10_percent_l163_163231


namespace find_c_plus_d_l163_163850

theorem find_c_plus_d (c d : ℝ) (h1 : 2 * c = 6) (h2 : c^2 - d = 4) : c + d = 8 := by
  sorry

end find_c_plus_d_l163_163850


namespace orange_face_probability_correct_l163_163439

-- Define the number of faces
def total_faces : ℕ := 12
def green_faces : ℕ := 5
def orange_faces : ℕ := 4
def purple_faces : ℕ := 3

-- Define the probability of rolling an orange face
def probability_of_orange_face : ℚ := orange_faces / total_faces

-- Statement of the theorem
theorem orange_face_probability_correct :
  probability_of_orange_face = 1 / 3 :=
by
  sorry

end orange_face_probability_correct_l163_163439


namespace sam_possible_lunches_without_violation_l163_163893

def main_dishes := ["Burger", "Fish and Chips", "Pasta", "Vegetable Salad"]
def beverages := ["Soda", "Juice"]
def snacks := ["Apple Pie", "Chocolate Cake"]

def valid_combinations := 
  (main_dishes.length * beverages.length * snacks.length) - 
  ((1 * if "Fish and Chips" ∈ main_dishes then 1 else 0) * if "Soda" ∈ beverages then 1 else 0 * snacks.length)

theorem sam_possible_lunches_without_violation : valid_combinations = 14 := by
  sorry

end sam_possible_lunches_without_violation_l163_163893


namespace tetrahedron_cd_length_l163_163327

theorem tetrahedron_cd_length (a b c d : Type) [MetricSpace a] [MetricSpace b] [MetricSpace c] [MetricSpace d] :
  let ab := 53
  let edge_lengths := [17, 23, 29, 39, 46, 53]
  ∃ cd, cd = 17 :=
by
  sorry

end tetrahedron_cd_length_l163_163327


namespace arithmetic_sequence_third_term_l163_163063

theorem arithmetic_sequence_third_term (a d : ℤ) (h : a + (a + 4 * d) = 14) : a + 2 * d = 7 := by
  -- We assume the sum of the first and fifth term is 14 and prove that the third term is 7.
  sorry

end arithmetic_sequence_third_term_l163_163063


namespace blind_box_problem_l163_163253

theorem blind_box_problem (x y : ℕ) :
  x + y = 135 ∧ 2 * x = 3 * y :=
sorry

end blind_box_problem_l163_163253


namespace find_common_ratio_l163_163985

noncomputable def a_n (n : ℕ) (q : ℚ) : ℚ :=
  if n = 1 then 1 / 8 else (q^(n - 1)) * (1 / 8)

theorem find_common_ratio (q : ℚ) :
  (a_n 4 q = -1) ↔ (q = -2) :=
by
  sorry

end find_common_ratio_l163_163985


namespace luther_latest_line_count_l163_163978

theorem luther_latest_line_count :
  let silk := 10
  let cashmere := silk / 2
  let blended := 2
  silk + cashmere + blended = 17 :=
by
  sorry

end luther_latest_line_count_l163_163978


namespace eggs_left_on_shelf_l163_163612

-- Define the conditions as variables in the Lean statement
variables (x y z : ℝ)

-- Define the final theorem statement
theorem eggs_left_on_shelf (hx : 0 ≤ x) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) :
  x * (1 - y) - z = (x - y * x) - z :=
by
  sorry

end eggs_left_on_shelf_l163_163612


namespace rhombus_diagonal_length_l163_163289

theorem rhombus_diagonal_length (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = 25) (h2 : A = 250) (h3 : A = (d1 * d2) / 2) : d2 = 20 := 
by
  rw [h1, h2] at h3
  sorry

end rhombus_diagonal_length_l163_163289


namespace eval_g_at_neg2_l163_163183

def g (x : ℝ) : ℝ := 5 * x + 2

theorem eval_g_at_neg2 : g (-2) = -8 := by
  sorry

end eval_g_at_neg2_l163_163183


namespace area_of_triangle_OPF_l163_163044

theorem area_of_triangle_OPF (O : ℝ × ℝ) (F : ℝ × ℝ) (P : ℝ × ℝ)
  (hO : O = (0, 0)) (hF : F = (1, 0)) (hP_on_parabola : P.2 ^ 2 = 4 * P.1)
  (hPF : dist P F = 3) : Real.sqrt 2 = 1 / 2 * abs (F.1 - O.1) * (2 * Real.sqrt 2) := 
sorry

end area_of_triangle_OPF_l163_163044


namespace graph_is_two_lines_l163_163489

theorem graph_is_two_lines : ∀ (x y : ℝ), (x ^ 2 - 25 * y ^ 2 - 20 * x + 100 = 0) ↔ (x = 10 + 5 * y ∨ x = 10 - 5 * y) := 
by 
  intro x y
  sorry

end graph_is_two_lines_l163_163489


namespace initial_paintings_l163_163996

theorem initial_paintings (paintings_per_day : ℕ) (days : ℕ) (total_paintings : ℕ) (initial_paintings : ℕ) 
  (h1 : paintings_per_day = 2) 
  (h2 : days = 30) 
  (h3 : total_paintings = 80) 
  (h4 : total_paintings = initial_paintings + paintings_per_day * days) : 
  initial_paintings = 20 := by
  sorry

end initial_paintings_l163_163996


namespace Jackie_exercise_hours_l163_163954

variable (work_hours : ℕ) (sleep_hours : ℕ) (free_time_hours : ℕ) (total_hours_in_day : ℕ)
variable (time_for_exercise : ℕ)

noncomputable def prove_hours_exercising (work_hours sleep_hours free_time_hours total_hours_in_day : ℕ) : Prop :=
  work_hours = 8 ∧
  sleep_hours = 8 ∧
  free_time_hours = 5 ∧
  total_hours_in_day = 24 → 
  time_for_exercise = total_hours_in_day - (work_hours + sleep_hours + free_time_hours)

theorem Jackie_exercise_hours :
  prove_hours_exercising 8 8 5 24 3 :=
by
  -- Proof is omitted as per instruction
  sorry

end Jackie_exercise_hours_l163_163954


namespace combined_function_is_linear_l163_163274

def original_parabola (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 5

def reflected_parabola (x : ℝ) : ℝ := -original_parabola x

def translated_original_parabola (x : ℝ) : ℝ := 3 * (x - 4)^2 + 4 * (x - 4) - 5

def translated_reflected_parabola (x : ℝ) : ℝ := -3 * (x + 6)^2 - 4 * (x + 6) + 5

def combined_function (x : ℝ) : ℝ := translated_original_parabola x + translated_reflected_parabola x

theorem combined_function_is_linear : ∃ (a b : ℝ), ∀ x : ℝ, combined_function x = a * x + b := by
  sorry

end combined_function_is_linear_l163_163274


namespace find_f_l163_163331

theorem find_f {f : ℝ → ℝ} (h : ∀ x, f (1/x) = x / (1 - x)) : ∀ x, f x = 1 / (x - 1) :=
by
  sorry

end find_f_l163_163331


namespace probability_two_red_balls_l163_163661

open Nat

theorem probability_two_red_balls (total_balls red_balls blue_balls green_balls balls_picked : Nat) 
  (total_eq : total_balls = red_balls + blue_balls + green_balls) 
  (red_eq : red_balls = 7) 
  (blue_eq : blue_balls = 5) 
  (green_eq : green_balls = 4) 
  (picked_eq : balls_picked = 2) :
  (choose red_balls balls_picked) / (choose total_balls balls_picked) = 7 / 40 :=
by
  sorry

end probability_two_red_balls_l163_163661


namespace extreme_value_result_l163_163146

open Real

-- Conditions
def function_has_extreme_value_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop := 
  deriv f x₀ = 0

-- The given function
noncomputable def f (x : ℝ) : ℝ := x * sin x

-- The problem statement (to prove)
theorem extreme_value_result (x₀ : ℝ) 
  (h : function_has_extreme_value_at f x₀) :
  (1 + x₀^2) * (1 + cos (2 * x₀)) = 2 :=
sorry

end extreme_value_result_l163_163146


namespace no_solution_condition_l163_163423

theorem no_solution_condition (n : ℝ) : ¬(∃ x y z : ℝ, n^2 * x + y = 1 ∧ n * y + z = 1 ∧ x + n^2 * z = 1) ↔ n = -1 := 
by {
    sorry
}

end no_solution_condition_l163_163423


namespace min_sticks_cover_200cm_l163_163679

def length_covered (n6 n7 : ℕ) : ℕ :=
  6 * n6 + 7 * n7

theorem min_sticks_cover_200cm :
  ∃ (n6 n7 : ℕ), length_covered n6 n7 = 200 ∧ (∀ (m6 m7 : ℕ), (length_covered m6 m7 = 200 → m6 + m7 ≥ n6 + n7)) ∧ (n6 + n7 = 29) :=
sorry

end min_sticks_cover_200cm_l163_163679


namespace quadratic_rewrite_correct_a_b_c_l163_163261

noncomputable def quadratic_rewrite (x : ℝ) : ℝ := -6*x^2 + 36*x + 216

theorem quadratic_rewrite_correct_a_b_c :
  ∃ a b c : ℝ, quadratic_rewrite x = a * (x + b)^2 + c ∧ a + b + c = 261 :=
by
  sorry

end quadratic_rewrite_correct_a_b_c_l163_163261


namespace linear_function_not_in_second_quadrant_l163_163007

theorem linear_function_not_in_second_quadrant (m : ℤ) (h1 : m + 4 > 0) (h2 : m + 2 ≤ 0) : 
  m = -3 ∨ m = -2 := 
sorry

end linear_function_not_in_second_quadrant_l163_163007


namespace total_blocks_traveled_l163_163126

-- Given conditions as definitions
def annie_walked_blocks : ℕ := 5
def annie_rode_blocks : ℕ := 7

-- The total blocks Annie traveled
theorem total_blocks_traveled : annie_walked_blocks + annie_rode_blocks + (annie_walked_blocks + annie_rode_blocks) = 24 := by
  sorry

end total_blocks_traveled_l163_163126


namespace students_in_line_l163_163637

theorem students_in_line (between : ℕ) (Yoojung Eunji : ℕ) (h1 : Yoojung = 1) (h2 : Eunji = 1) : 
  between + Yoojung + Eunji = 16 :=
  sorry

end students_in_line_l163_163637


namespace necessary_but_not_sufficient_condition_proof_l163_163161

noncomputable def necessary_but_not_sufficient_condition (x : ℝ) : Prop :=
  2 * x ^ 2 - 5 * x - 3 ≥ 0

theorem necessary_but_not_sufficient_condition_proof (x : ℝ) :
  (x < 0 ∨ x > 2) → necessary_but_not_sufficient_condition x :=
  sorry

end necessary_but_not_sufficient_condition_proof_l163_163161


namespace age_of_James_when_Thomas_reaches_current_age_l163_163683
    
theorem age_of_James_when_Thomas_reaches_current_age
  (T S J : ℕ)
  (h1 : T = 6)
  (h2 : S = T + 13)
  (h3 : S = J - 5) :
  J + (S - T) = 37 := 
by
  sorry

end age_of_James_when_Thomas_reaches_current_age_l163_163683


namespace price_of_davids_toy_l163_163616

theorem price_of_davids_toy :
  ∀ (n : ℕ) (avg_before : ℕ) (avg_after : ℕ) (total_toys_after : ℕ), 
    n = 5 →
    avg_before = 10 →
    avg_after = 11 →
    total_toys_after = 6 →
  (total_toys_after * avg_after - n * avg_before = 16) :=
by
  intros n avg_before avg_after total_toys_after h_n h_avg_before h_avg_after h_total_toys_after
  sorry

end price_of_davids_toy_l163_163616


namespace totalWatermelons_l163_163895

def initialWatermelons : ℕ := 4
def additionalWatermelons : ℕ := 3

theorem totalWatermelons : initialWatermelons + additionalWatermelons = 7 := by
  sorry

end totalWatermelons_l163_163895


namespace number_of_correct_propositions_is_one_l163_163053

def obtuse_angle_is_second_quadrant (θ : ℝ) : Prop :=
  θ > 90 ∧ θ < 180

def acute_angle (θ : ℝ) : Prop :=
  θ < 90

def first_quadrant_not_negative (θ : ℝ) : Prop :=
  θ > 0 ∧ θ < 90

def second_quadrant_greater_first (θ₁ θ₂ : ℝ) : Prop :=
  (θ₁ > 90 ∧ θ₁ < 180) → (θ₂ > 0 ∧ θ₂ < 90) → θ₁ > θ₂

theorem number_of_correct_propositions_is_one :
  (¬ ∀ θ, obtuse_angle_is_second_quadrant θ) ∧
  (∀ θ, acute_angle θ → θ < 90) ∧
  (¬ ∀ θ, first_quadrant_not_negative θ) ∧
  (¬ ∀ θ₁ θ₂, second_quadrant_greater_first θ₁ θ₂) →
  1 = 1 :=
by
  sorry

end number_of_correct_propositions_is_one_l163_163053


namespace jensen_miles_city_l163_163579

theorem jensen_miles_city (total_gallons : ℕ) (highway_miles : ℕ) (highway_mpg : ℕ)
  (city_mpg : ℕ) (highway_gallons : ℕ) (city_gallons : ℕ) (city_miles : ℕ) :
  total_gallons = 9 ∧ highway_miles = 210 ∧ highway_mpg = 35 ∧ city_mpg = 18 ∧
  highway_gallons = highway_miles / highway_mpg ∧
  city_gallons = total_gallons - highway_gallons ∧
  city_miles = city_gallons * city_mpg → city_miles = 54 :=
by
  sorry

end jensen_miles_city_l163_163579


namespace total_cost_correct_l163_163984

def cost_barette : ℕ := 3
def cost_comb : ℕ := 1

def kristine_barrettes : ℕ := 1
def kristine_combs : ℕ := 1

def crystal_barrettes : ℕ := 3
def crystal_combs : ℕ := 1

def total_spent (cost_barette : ℕ) (cost_comb : ℕ) 
  (kristine_barrettes : ℕ) (kristine_combs : ℕ) 
  (crystal_barrettes : ℕ) (crystal_combs : ℕ) : ℕ :=
  (kristine_barrettes * cost_barette + kristine_combs * cost_comb) + 
  (crystal_barrettes * cost_barette + crystal_combs * cost_comb)

theorem total_cost_correct :
  total_spent cost_barette cost_comb kristine_barrettes kristine_combs crystal_barrettes crystal_combs = 14 :=
by
  sorry

end total_cost_correct_l163_163984


namespace polynomial_form_l163_163024

theorem polynomial_form (P : Polynomial ℝ) (hP : P ≠ 0)
    (h : ∀ x : ℝ, P.eval x * P.eval (2 * x^2) = P.eval (2 * x^3 + x)) :
    ∃ k : ℕ, k > 0 ∧ P = (X^2 + 1) ^ k :=
by sorry

end polynomial_form_l163_163024


namespace different_total_scores_l163_163760

noncomputable def basket_scores (x y z : ℕ) : ℕ := x + 2 * y + 3 * z

def total_baskets := 7
def score_range := {n | 7 ≤ n ∧ n ≤ 21}

theorem different_total_scores : 
  ∃ (count : ℕ), count = 15 ∧ 
  ∀ n ∈ score_range, ∃ (x y z : ℕ), x + y + z = total_baskets ∧ basket_scores x y z = n :=
sorry

end different_total_scores_l163_163760


namespace totalPeaches_l163_163911

-- Define the number of red, yellow, and green peaches
def redPeaches := 7
def yellowPeaches := 15
def greenPeaches := 8

-- Define the total number of peaches and the proof statement
theorem totalPeaches : redPeaches + yellowPeaches + greenPeaches = 30 := by
  sorry

end totalPeaches_l163_163911


namespace sum_of_remainders_l163_163334

theorem sum_of_remainders 
  (a b c : ℕ) 
  (h1 : a % 53 = 37) 
  (h2 : b % 53 = 14) 
  (h3 : c % 53 = 7) : 
  (a + b + c) % 53 = 5 := 
by 
  sorry

end sum_of_remainders_l163_163334


namespace cosine_of_angle_between_tangents_l163_163582

-- Definitions based on the conditions given in a)
def circle_eq (x y : ℝ) : Prop := x^2 - 2 * x + y^2 - 2 * y + 1 = 0
def P : ℝ × ℝ := (3, 2)

-- The main theorem to be proved
theorem cosine_of_angle_between_tangents (x y : ℝ)
  (hx : circle_eq x y) : 
  cos_angle_between_tangents := 
  sorry

end cosine_of_angle_between_tangents_l163_163582


namespace distribution_ways_l163_163326

theorem distribution_ways (books students : ℕ) (h_books : books = 6) (h_students : students = 6) :
  ∃ ways : ℕ, ways = 6 * 5^6 ∧ ways = 93750 :=
by
  sorry

end distribution_ways_l163_163326


namespace pythagorean_triple_l163_163769

theorem pythagorean_triple {a b c : ℕ} (h : a * a + b * b = c * c) (gcd_abc : Nat.gcd (Nat.gcd a b) c = 1) :
  ∃ m n : ℕ, a = 2 * m * n ∧ b = m * m - n * n ∧ c = m * m + n * n :=
sorry

end pythagorean_triple_l163_163769


namespace solve_for_y_l163_163538

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 40

theorem solve_for_y 
  (h₁ : 1.5 * x = 0.75 * y) 
  (h₂ : x = 20) : 
  y = 40 :=
by
  sorry

end solve_for_y_l163_163538


namespace line_intersection_l163_163129

/-- Prove the intersection of the lines given by the equations
    8x - 5y = 10 and 3x + 2y = 1 is (25/31, -22/31) -/
theorem line_intersection :
  ∃ (x y : ℚ), 8 * x - 5 * y = 10 ∧ 3 * x + 2 * y = 1 ∧ x = 25 / 31 ∧ y = -22 / 31 :=
by
  sorry

end line_intersection_l163_163129


namespace sequence_first_number_l163_163634

theorem sequence_first_number (a: ℕ → ℕ) (h1: a 7 = 14) (h2: a 8 = 19) (h3: a 9 = 33) :
  (∀ n, n ≥ 2 → a (n+1) = a n + a (n-1)) → a 1 = 30 :=
by
  sorry

end sequence_first_number_l163_163634


namespace singles_percentage_l163_163035

-- Definitions based on conditions
def total_hits : ℕ := 50
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 7
def non_single_hits : ℕ := home_runs + triples + doubles
def singles : ℕ := total_hits - non_single_hits

-- Theorem based on the proof problem
theorem singles_percentage :
  singles = 38 ∧ (singles / total_hits : ℚ) * 100 = 76 := 
  by
    sorry

end singles_percentage_l163_163035


namespace area_of_30_60_90_triangle_l163_163855

theorem area_of_30_60_90_triangle (altitude : ℝ) (h : altitude = 3) : 
  ∃ (area : ℝ), area = 6 * Real.sqrt 3 := 
sorry

end area_of_30_60_90_triangle_l163_163855


namespace reciprocal_of_neg_eight_l163_163065

theorem reciprocal_of_neg_eight : (1 / (-8 : ℝ)) = -1 / 8 := sorry

end reciprocal_of_neg_eight_l163_163065


namespace descent_phase_duration_l163_163914

noncomputable def start_time_in_seconds : ℕ := 45 * 60 + 39
noncomputable def end_time_in_seconds : ℕ := 47 * 60 + 33

theorem descent_phase_duration :
  end_time_in_seconds - start_time_in_seconds = 114 := by
  sorry

end descent_phase_duration_l163_163914


namespace total_population_l163_163014

theorem total_population (n : ℕ) (avg_population : ℕ) (h1 : n = 20) (h2 : avg_population = 4750) :
  n * avg_population = 95000 := by
  subst_vars
  sorry

end total_population_l163_163014


namespace triangle_area_correct_l163_163688

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_area_correct : 
  area_of_triangle (0, 0) (2, 0) (2, 3) = 3 :=
by
  sorry

end triangle_area_correct_l163_163688


namespace find_numbers_l163_163043

-- Define the conditions
def geometric_mean_condition (a b : ℝ) : Prop :=
  a * b = 3

def harmonic_mean_condition (a b : ℝ) : Prop :=
  2 / (1 / a + 1 / b) = 3 / 2

-- State the theorem to be proven
theorem find_numbers (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  geometric_mean_condition a b ∧ harmonic_mean_condition a b → (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 1) := 
by 
  sorry

end find_numbers_l163_163043


namespace average_num_divisors_2019_l163_163664

def num_divisors (n : ℕ) : ℕ :=
  (n.divisors).card

theorem average_num_divisors_2019 :
  1 / 2019 * (Finset.sum (Finset.range 2020) num_divisors) = 15682 / 2019 :=
by
  sorry

end average_num_divisors_2019_l163_163664


namespace negation_of_prop_l163_163135

theorem negation_of_prop :
  (¬ ∀ x : ℝ, x^2 > x - 1) ↔ ∃ x : ℝ, x^2 ≤ x - 1 :=
sorry

end negation_of_prop_l163_163135


namespace fly_least_distance_l163_163781

noncomputable def leastDistance (r : ℝ) (h : ℝ) (start_dist : ℝ) (end_dist : ℝ) : ℝ := 
  let C := 2 * Real.pi * r
  let R := Real.sqrt (r^2 + h^2)
  let θ := C / R
  let A := (start_dist, 0)
  let B := (Real.cos (θ / 2) * end_dist, Real.sin (θ / 2) * end_dist)
  Real.sqrt ((B.fst - A.fst)^2 + (B.snd - A.snd)^2)

theorem fly_least_distance : 
  leastDistance 600 (200 * Real.sqrt 7) 125 (375 * Real.sqrt 2) = 625 := 
sorry

end fly_least_distance_l163_163781


namespace correct_factorization_option_A_l163_163182

variable (x y : ℝ)

theorem correct_factorization_option_A :
  (2 * x^2 + 3 * x + 1 = (2 * x + 1) * (x + 1)) :=
by {
  sorry
}

end correct_factorization_option_A_l163_163182


namespace intersection_complement_eq_l163_163118

def setA : Set ℝ := { x | (x - 6) * (x + 1) ≤ 0 }
def setB : Set ℝ := { x | x ≥ 2 }

theorem intersection_complement_eq :
  setA ∩ (Set.univ \ setB) = { x | -1 ≤ x ∧ x < 2 } := 
by 
  sorry

end intersection_complement_eq_l163_163118


namespace total_sample_size_is_72_l163_163943

-- Definitions based on the given conditions:
def production_A : ℕ := 600
def production_B : ℕ := 1200
def production_C : ℕ := 1800
def total_production : ℕ := production_A + production_B + production_C
def sampled_B : ℕ := 2

-- Main theorem to prove the sample size:
theorem total_sample_size_is_72 : 
  ∃ (n : ℕ), 
    (∃ s_A s_B s_C, 
      s_A = (production_A * sampled_B * total_production) / production_B^2 ∧ 
      s_B = sampled_B ∧ 
      s_C = (production_C * sampled_B * total_production) / production_B^2 ∧
      n = s_A + s_B + s_C) ∧ 
  (n = 72) :=
sorry

end total_sample_size_is_72_l163_163943


namespace pear_weight_l163_163287

theorem pear_weight
  (w_apple : ℕ)
  (p_weight_relation : 12 * w_apple = 8 * P + 5400)
  (apple_weight : w_apple = 530) :
  P = 120 :=
by
  -- sorry, proof is omitted as per instructions
  sorry

end pear_weight_l163_163287


namespace usual_time_72_l163_163668

namespace TypicalTimeProof

variables (S T : ℝ) 

theorem usual_time_72 (h : T ≠ 0) (h2 : 0.75 * S ≠ 0) (h3 : 4 * T = 3 * (T + 24)) : T = 72 := by
  sorry

end TypicalTimeProof

end usual_time_72_l163_163668


namespace elena_allowance_fraction_l163_163120

variable {A m s : ℝ}

theorem elena_allowance_fraction {A : ℝ} (h1 : m = 0.25 * (A - s)) (h2 : s = 0.10 * (A - m)) : m + s = (4 / 13) * A :=
by
  sorry

end elena_allowance_fraction_l163_163120


namespace number_of_diet_soda_bottles_l163_163886

theorem number_of_diet_soda_bottles (apples regular_soda total_bottles diet_soda : ℕ)
    (h_apples : apples = 36)
    (h_regular_soda : regular_soda = 80)
    (h_total_bottles : total_bottles = apples + 98)
    (h_diet_soda_eq : total_bottles = regular_soda + diet_soda) :
    diet_soda = 54 := by
  sorry

end number_of_diet_soda_bottles_l163_163886


namespace jana_winning_strategy_l163_163691

theorem jana_winning_strategy (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  (m + n) % 2 = 1 ∨ m = 1 ∨ n = 1 := sorry

end jana_winning_strategy_l163_163691


namespace hyperbola_asymptotes_l163_163332

theorem hyperbola_asymptotes (a b c : ℝ) (h : a > 0) (h_b_gt_0: b > 0) 
  (eqn1 : b = 2 * Real.sqrt 2 * a)
  (focal_distance : 2 * a = (2 * c)/3)
  (focal_length : c = 3 * a) : 
  (∀ x : ℝ, ∀ y : ℝ, (y = (2 * Real.sqrt 2) * x) ∨ (y = -(2 * Real.sqrt 2) * x)) := by
  sorry

end hyperbola_asymptotes_l163_163332


namespace volume_in_region_l163_163733

def satisfies_conditions (x y : ℝ) : Prop :=
  |8 - x| + y ≤ 10 ∧ 3 * y - x ≥ 15

def in_region (x y : ℝ) : Prop :=
  satisfies_conditions x y

theorem volume_in_region (x y p m n : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (hn : n ≠ 0) (V : ℝ) 
  (hvol : V = (m * Real.pi) / (n * Real.sqrt p))
  (hprime : m.gcd n = 1 ∧ ¬(∃ k : ℕ, k^2 ∣ p ∧ k ≥ 2)) 
  (hpoints : ∀ (x y : ℝ), in_region x y → 3 * y - x = 15) : 
  m + n + p = 365 := 
sorry

end volume_in_region_l163_163733


namespace geometric_seq_prod_l163_163311

-- Conditions: Geometric sequence and given value of a_1 * a_7 * a_13
variables {a : ℕ → ℝ}
variable (r : ℝ)

-- Definition of a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- The proof problem
theorem geometric_seq_prod (h_geo : geometric_sequence a r) (h_prod : a 1 * a 7 * a 13 = 8) :
  a 3 * a 11 = 4 :=
sorry

end geometric_seq_prod_l163_163311


namespace time_for_b_and_d_together_l163_163092

theorem time_for_b_and_d_together :
  let A_rate := 1 / 3
  let D_rate := 1 / 4
  (∃ B_rate C_rate : ℚ,
    B_rate + C_rate = 1 / 3 ∧
    A_rate + C_rate = 1 / 2 ∧
    1 / (B_rate + D_rate) = 2.4) :=
  
by
  let A_rate := 1 / 3
  let D_rate := 1 / 4
  use 1 / 6, 1 / 6
  sorry

end time_for_b_and_d_together_l163_163092


namespace Malou_third_quiz_score_l163_163366

theorem Malou_third_quiz_score (q1 q2 q3 : ℕ) (avg_score : ℕ) (total_quizzes : ℕ) : 
  q1 = 91 ∧ q2 = 90 ∧ avg_score = 91 ∧ total_quizzes = 3 → q3 = 92 :=
by
  intro h
  sorry

end Malou_third_quiz_score_l163_163366


namespace train_speed_correct_l163_163115

noncomputable def train_speed (length_meters : ℕ) (time_seconds : ℕ) : ℝ :=
  (length_meters : ℝ) / 1000 / (time_seconds / 3600)

theorem train_speed_correct :
  train_speed 2500 50 = 180 := 
by
  -- We leave the proof as sorry, the statement is sufficient
  sorry

end train_speed_correct_l163_163115


namespace library_wall_length_l163_163644

theorem library_wall_length 
  (D B : ℕ) 
  (h1: D = B) 
  (desk_length bookshelf_length leftover_space : ℝ) 
  (h2: desk_length = 2) 
  (h3: bookshelf_length = 1.5) 
  (h4: leftover_space = 1) : 
  3.5 * D + leftover_space = 8 :=
by { sorry }

end library_wall_length_l163_163644


namespace find_m_l163_163755

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

-- Define the function to calculate m * a + b
def m_a_plus_b (m : ℝ) : ℝ × ℝ := (2 * m - 1, 3 * m + 2)

-- Define the vector a - 2 * b
def a_minus_2b : ℝ × ℝ := (4, -1)

-- Define the condition for parallelism
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

-- The theorem that states the equivalence
theorem find_m (m : ℝ) (H : parallel (m_a_plus_b m) a_minus_2b) : m = -1/2 :=
by
  sorry

end find_m_l163_163755


namespace sum_of_imaginary_parts_l163_163012

theorem sum_of_imaginary_parts (x y u v w z : ℝ) (h1 : y = 5) 
  (h2 : w = -x - u) (h3 : (x + y * I) + (u + v * I) + (w + z * I) = 4 * I) :
  v + z = -1 :=
by
  sorry

end sum_of_imaginary_parts_l163_163012


namespace average_value_is_2020_l163_163772

namespace CardsAverage

theorem average_value_is_2020 (n : ℕ) (h : (2020 * 3 * ((n * (n + 1)) + 2) = n * (n + 1) * (2 * n + 1) + 6 * (n + 1))) : n = 3015 := 
by
  sorry

end CardsAverage

end average_value_is_2020_l163_163772


namespace partial_fraction_decomposition_l163_163706

noncomputable def partial_fraction_product (A B C : ℤ) : ℤ :=
  A * B * C

theorem partial_fraction_decomposition:
  ∃ A B C : ℤ, 
  (∀ x : ℤ, (x^2 - 19 = A * (x + 2) * (x - 3) 
                    + B * (x - 1) * (x - 3) 
                    + C * (x - 1) * (x + 2) )) 
  → partial_fraction_product A B C = 3 :=
by
  sorry

end partial_fraction_decomposition_l163_163706


namespace find_a_mul_b_l163_163282

theorem find_a_mul_b (x y z a b : ℝ)
  (h1 : a = x)
  (h2 : b = y)
  (h3 : x + x = y * x)
  (h4 : b = z)
  (h5 : x + x = z * z)
  (h6 : y = 3)
  : a * b = 4 := by
  sorry

end find_a_mul_b_l163_163282


namespace part1_part2_l163_163266

-- Definitions and assumptions based on the problem
def f (x a : ℝ) : ℝ := abs (x - a)

-- Condition (1) with given function and inequality solution set
theorem part1 (a : ℝ) :
  (∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  sorry

-- Condition (2) with the range of m under the previously found value of a
theorem part2 (m : ℝ) :
  (∃ x, f x 2 + f (x + 5) 2 < m) → m > 5 :=
by
  sorry

end part1_part2_l163_163266


namespace parabola_axis_l163_163138

theorem parabola_axis (p : ℝ) (h_parabola : ∀ x : ℝ, y = x^2 → x^2 = y) : (y = - p / 2) :=
by
  sorry

end parabola_axis_l163_163138


namespace sum_of_coordinates_of_intersection_l163_163669

theorem sum_of_coordinates_of_intersection :
  let A := (0, 4)
  let B := (6, 0)
  let C := (9, 3)
  let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let line_AE := (fun x : ℚ => (-1/3) * x + 4)
  let line_CD := (fun x : ℚ => (1/6) * x + 1/2)
  let F_x := (21 : ℚ) / 3
  let F_y := line_AE F_x
  F_x + F_y = 26 / 3 := sorry

end sum_of_coordinates_of_intersection_l163_163669


namespace distance_from_A_to_y_axis_l163_163490

-- Define the coordinates of point A
def point_A : ℝ × ℝ := (-3, 4)

-- Define the distance function from a point to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

-- State the theorem
theorem distance_from_A_to_y_axis :
  distance_to_y_axis point_A = 3 :=
  by
    -- This part will contain the proof, but we omit it with 'sorry' for now.
    sorry

end distance_from_A_to_y_axis_l163_163490


namespace value_of_w_l163_163839

-- Define the positivity of w
def positive_integer (w : ℕ) := w > 0

-- Define the sum of the digits
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Define the function which encapsulates the problem
def problem_condition (w : ℕ) := sum_of_digits (10^w - 74)

-- The main proof problem
theorem value_of_w (w : ℕ) (h : positive_integer w) : problem_condition w = 17 :=
by
  sorry

end value_of_w_l163_163839


namespace sum_xyz_zero_l163_163598

theorem sum_xyz_zero 
  (x y z : ℝ)
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : y = 6 * z) : 
  x + y + z = 0 := by
  sorry

end sum_xyz_zero_l163_163598


namespace richard_cleans_in_45_minutes_l163_163195
noncomputable def richard_time (R : ℝ) := 
  let cory_time := R + 3
  let blake_time := (R + 3) - 4
  (R + cory_time + blake_time = 136) -> R = 45

theorem richard_cleans_in_45_minutes : 
  ∃ R : ℝ, richard_time R := 
sorry

end richard_cleans_in_45_minutes_l163_163195


namespace count_integers_expression_negative_l163_163545

theorem count_integers_expression_negative :
  ∃ n : ℕ, n = 4 ∧ 
  ∀ x : ℤ, x^4 - 60 * x^2 + 144 < 0 → n = 4 := by
  -- Placeholder for the proof
  sorry

end count_integers_expression_negative_l163_163545


namespace solve_for_x_l163_163352

theorem solve_for_x (x : ℝ) (h : -200 * x = 1600) : x = -8 :=
sorry

end solve_for_x_l163_163352


namespace value_of_expression_l163_163752

theorem value_of_expression : 
  103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end value_of_expression_l163_163752


namespace f_2014_odd_f_2014_not_even_l163_163777

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 1 / x
| (n + 1), x => 1 / (x + f n x)

theorem f_2014_odd :
  ∀ x : ℝ, f 2014 x = - f 2014 (-x) :=
sorry

theorem f_2014_not_even :
  ∃ x : ℝ, f 2014 x ≠ f 2014 (-x) :=
sorry

end f_2014_odd_f_2014_not_even_l163_163777


namespace geometric_series_common_ratio_l163_163555

theorem geometric_series_common_ratio (a r : ℝ) (n : ℕ) 
(h1 : a = 7 / 3) 
(h2 : r = 49 / 21)
(h3 : r = 343 / 147):
  r = 7 / 3 :=
by
  sorry

end geometric_series_common_ratio_l163_163555


namespace problem_i_problem_ii_l163_163498

noncomputable def f (m x : ℝ) := (Real.log x / Real.log m) ^ 2 + 2 * (Real.log x / Real.log m) - 3

theorem problem_i (x : ℝ) : f 2 x < 0 ↔ (1 / 8) < x ∧ x < 2 :=
by sorry

theorem problem_ii (m : ℝ) (H : ∀ x, 2 ≤ x ∧ x ≤ 4 → f m x < 0) : 
  (0 < m ∧ m < 4^(1/3)) ∨ (4 < m) :=
by sorry

end problem_i_problem_ii_l163_163498


namespace largest_side_of_rectangle_l163_163482

theorem largest_side_of_rectangle :
  ∃ (l w : ℝ), (2 * l + 2 * w = 240) ∧ (l * w = 12 * 240) ∧ (l = 86.835 ∨ w = 86.835) :=
by
  -- Actual proof would be here
  sorry

end largest_side_of_rectangle_l163_163482


namespace height_ratio_l163_163575

theorem height_ratio (C : ℝ) (h_o : ℝ) (V_s : ℝ) (h_s : ℝ) (r : ℝ) :
  C = 18 * π →
  h_o = 20 →
  V_s = 270 * π →
  C = 2 * π * r →
  V_s = 1 / 3 * π * r^2 * h_s →
  h_s / h_o = 1 / 2 :=
by
  sorry

end height_ratio_l163_163575


namespace inscribed_circle_radius_l163_163715

theorem inscribed_circle_radius (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 30) : 
  ∃ r : ℝ, r = (105 * Real.sqrt 274) / 274 := 
by 
  sorry

end inscribed_circle_radius_l163_163715


namespace distance_between_points_l163_163157

theorem distance_between_points (points : Fin 7 → ℝ × ℝ) (diameter : ℝ)
  (h_diameter : diameter = 1)
  (h_points_in_circle : ∀ i : Fin 7, (points i).fst^2 + (points i).snd^2 ≤ (diameter / 2)^2) :
  ∃ (i j : Fin 7), i ≠ j ∧ (dist (points i) (points j) ≤ 1 / 2) := 
by
  sorry

end distance_between_points_l163_163157


namespace pow_mod_eq_l163_163959

theorem pow_mod_eq (n : ℕ) : 
  (3^n % 5 = 3 % 5) → 
  (3^(n+1) % 5 = (3 * 3^n) % 5) → 
  (3^(n+2) % 5 = (3 * 3^(n+1)) % 5) → 
  (3^(n+3) % 5 = (3 * 3^(n+2)) % 5) → 
  (3^4 % 5 = 1 % 5) → 
  (2023 % 4 = 3) → 
  (3^2023 % 5 = 2 % 5) :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end pow_mod_eq_l163_163959


namespace range_of_m_l163_163560

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, abs (x - m) < 1 ↔ (1/3 < x ∧ x < 1/2)) ↔ (-1/2 ≤ m ∧ m ≤ 4/3) :=
by
  sorry

end range_of_m_l163_163560


namespace maria_cookies_left_l163_163783

theorem maria_cookies_left
    (total_cookies : ℕ) -- Maria has 60 cookies
    (friend_share : ℕ) -- 20% of the initial cookies goes to the friend
    (family_share : ℕ) -- 1/3 of the remaining cookies goes to the family
    (eaten_cookies : ℕ) -- Maria eats 4 cookies
    (neighbor_share : ℕ) -- Maria gives 1/6 of the remaining cookies to neighbor
    (initial_cookies : total_cookies = 60)
    (friend_fraction : friend_share = total_cookies * 20 / 100)
    (remaining_after_friend : ℕ := total_cookies - friend_share)
    (family_fraction : family_share = remaining_after_friend / 3)
    (remaining_after_family : ℕ := remaining_after_friend - family_share)
    (eaten : eaten_cookies = 4)
    (remaining_after_eating : ℕ := remaining_after_family - eaten_cookies)
    (neighbor_fraction : neighbor_share = remaining_after_eating / 6)
    (neighbor_integerized : neighbor_share = 4) -- assumed whole number for neighbor's share
    (remaining_after_neighbor : ℕ := remaining_after_eating - neighbor_share) : 
    remaining_after_neighbor = 24 :=
sorry  -- The statement matches the problem, proof is left out

end maria_cookies_left_l163_163783


namespace alyssa_puppies_left_l163_163719

def initial_puppies : Nat := 7
def puppies_per_puppy : Nat := 4
def given_away : Nat := 15

theorem alyssa_puppies_left :
  (initial_puppies + initial_puppies * puppies_per_puppy) - given_away = 20 := 
  by
    sorry

end alyssa_puppies_left_l163_163719


namespace unoccupied_garden_area_is_correct_l163_163444

noncomputable def area_unoccupied_by_pond_trees_bench (π : ℝ) : ℝ :=
  let garden_area := 144
  let pond_area_rectangle := 6
  let pond_area_semi_circle := 2 * π
  let trees_area := 3
  let bench_area := 3
  garden_area - (pond_area_rectangle + pond_area_semi_circle + trees_area + bench_area)

theorem unoccupied_garden_area_is_correct : 
  area_unoccupied_by_pond_trees_bench Real.pi = 132 - 2 * Real.pi :=
by
  sorry

end unoccupied_garden_area_is_correct_l163_163444


namespace minimum_squares_required_l163_163101

theorem minimum_squares_required (length : ℚ) (width : ℚ) (M N : ℕ) :
  (length = 121 / 2) → (width = 143 / 3) → (M / N = 33 / 26) → (M * N = 858) :=
by
  intros hL hW hMN
  -- Proof skipped
  sorry

end minimum_squares_required_l163_163101


namespace sin_150_eq_half_l163_163888

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l163_163888


namespace estimated_probability_l163_163654

noncomputable def needle_intersection_probability : ℝ := 0.4

structure NeedleExperimentData :=
(distance_between_lines : ℝ)
(length_of_needle : ℝ)
(num_trials_intersections : List (ℕ × ℕ))
(intersection_frequencies : List ℝ)

def experiment_data : NeedleExperimentData :=
{ distance_between_lines := 5,
  length_of_needle := 3,
  num_trials_intersections := [(50, 23), (100, 48), (200, 83), (500, 207), (1000, 404), (2000, 802)],
  intersection_frequencies := [0.460, 0.480, 0.415, 0.414, 0.404, 0.401] }

theorem estimated_probability (data : NeedleExperimentData) :
  ∀ P : ℝ, (∀ n m, (n, m) ∈ data.num_trials_intersections → abs (m / n - P) < 0.1) → P = needle_intersection_probability :=
by
  intro P hP
  sorry

end estimated_probability_l163_163654


namespace customer_saves_7_906304_percent_l163_163528

variable {P : ℝ} -- Define the base retail price as a variable

-- Define the percentage reductions and additions
def reduced_price (P : ℝ) : ℝ := 0.88 * P
def further_discount_price (P : ℝ) : ℝ := reduced_price P * 0.95
def price_with_dealers_fee (P : ℝ) : ℝ := further_discount_price P * 1.02
def final_price (P : ℝ) : ℝ := price_with_dealers_fee P * 1.08

-- Define the final price factor
def final_price_factor : ℝ := 0.88 * 0.95 * 1.02 * 1.08

noncomputable def total_savings (P : ℝ) : ℝ :=
  P - (final_price_factor * P)

theorem customer_saves_7_906304_percent (P : ℝ) :
  total_savings P = P * 0.07906304 := by
  sorry -- Proof to be added

end customer_saves_7_906304_percent_l163_163528


namespace sum_quotient_product_diff_l163_163909

theorem sum_quotient_product_diff (x y : ℚ) (h₁ : x + y = 6) (h₂ : x / y = 6) : 
  (x * y) - (x - y) = 6 / 49 :=
  sorry

end sum_quotient_product_diff_l163_163909


namespace sum_of_series_equals_negative_682_l163_163108

noncomputable def geometric_sum : ℤ :=
  let a := 2
  let r := -2
  let n := 10
  (a * (r ^ n - 1)) / (r - 1)

theorem sum_of_series_equals_negative_682 : geometric_sum = -682 := 
by sorry

end sum_of_series_equals_negative_682_l163_163108


namespace how_fast_is_a_l163_163503

variable (a b : ℝ) (k : ℝ)

theorem how_fast_is_a (h1 : a = k * b) (h2 : a + b = 1 / 30) (h3 : a = 1 / 40) : k = 3 := sorry

end how_fast_is_a_l163_163503


namespace joe_fruit_probability_l163_163764

theorem joe_fruit_probability :
  let prob_same := (1 / 4) ^ 3
  let total_prob_same := 4 * prob_same
  let prob_diff := 1 - total_prob_same
  prob_diff = 15 / 16 :=
by
  sorry

end joe_fruit_probability_l163_163764


namespace expression_evaluation_l163_163375

theorem expression_evaluation (a b : ℤ) (h : a - 2 * b = 4) : 3 - a + 2 * b = -1 :=
by
  sorry

end expression_evaluation_l163_163375


namespace rectangle_inscribed_circle_circumference_l163_163860

/-- A 9 cm by 12 cm rectangle is inscribed in a circle. The circumference of the circle is 15π cm. -/
theorem rectangle_inscribed_circle_circumference :
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  circumference = 15 * Real.pi :=
by
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  have h_diameter : diameter = 15 := by
    sorry
  have h_circumference : circumference = 15 * Real.pi := by
    sorry
  exact h_circumference

end rectangle_inscribed_circle_circumference_l163_163860


namespace log_inequalities_l163_163660

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := Real.log (Real.log 2 / Real.log 3) / Real.log 2

theorem log_inequalities : c < b ∧ b < a :=
  sorry

end log_inequalities_l163_163660


namespace greatest_integer_part_expected_winnings_l163_163728

noncomputable def expected_winnings_one_envelope : ℝ := 500

noncomputable def expected_winnings_two_envelopes : ℝ := 625

noncomputable def expected_winnings_three_envelopes : ℝ := 695.3125

theorem greatest_integer_part_expected_winnings :
  ⌊expected_winnings_three_envelopes⌋ = 695 :=
by 
  sorry

end greatest_integer_part_expected_winnings_l163_163728


namespace range_of_a_l163_163300

theorem range_of_a (A B C : Set ℝ) (a : ℝ) :
  A = { x | -1 < x ∧ x < 4 } →
  B = { x | -5 < x ∧ x < (3 / 2) } →
  C = { x | (1 - 2 * a) < x ∧ x < (2 * a) } →
  (C ⊆ (A ∩ B)) →
  a ≤ (3 / 4) :=
by
  intros hA hB hC hSubset
  sorry

end range_of_a_l163_163300


namespace min_distance_curves_l163_163345

theorem min_distance_curves (P Q : ℝ × ℝ) (h1 : P.2 = (1/3) * Real.exp P.1) (h2 : Q.2 = Real.log (3 * Q.1)) :
  ∃ d : ℝ, d = Real.sqrt 2 * (Real.log 3 - 1) ∧ d = |P.1 - Q.1| := sorry

end min_distance_curves_l163_163345


namespace lesser_fraction_l163_163273

theorem lesser_fraction (x y : ℚ) (h1 : x + y = 10 / 11) (h2 : x * y = 1 / 8) : min x y = (80 - 2 * Real.sqrt 632) / 176 := 
by sorry

end lesser_fraction_l163_163273


namespace schoolchildren_number_l163_163305

theorem schoolchildren_number (n m S : ℕ) 
  (h1 : S = 22 * n + 3)
  (h2 : S = (n - 1) * m)
  (h3 : n ≤ 18)
  (h4 : m ≤ 36) : 
  S = 135 := 
sorry

end schoolchildren_number_l163_163305


namespace prime_odd_sum_l163_163558

theorem prime_odd_sum (a b : ℕ) (h1 : Prime a) (h2 : Odd b) (h3 : a^2 + b = 2001) : a + b = 1999 :=
sorry

end prime_odd_sum_l163_163558


namespace max_new_cars_l163_163130

theorem max_new_cars (b₁ : ℕ) (r : ℝ) (M : ℕ) (L : ℕ) (x : ℝ) (h₀ : b₁ = 30) (h₁ : r = 0.94) (h₂ : M = 600000) (h₃ : L = 300000) :
  x ≤ (3.6 * 10^4) :=
sorry

end max_new_cars_l163_163130


namespace range_of_t_for_obtuse_triangle_l163_163811

def is_obtuse_triangle (a b c : ℝ) : Prop := ∃t : ℝ, a = t - 1 ∧ b = t + 1 ∧ c = t + 3

theorem range_of_t_for_obtuse_triangle :
  ∀ t : ℝ, is_obtuse_triangle (t-1) (t+1) (t+3) → (3 < t ∧ t < 7) :=
by
  intros t ht
  sorry

end range_of_t_for_obtuse_triangle_l163_163811


namespace function_positivity_range_l163_163906

theorem function_positivity_range (m x : ℝ): 
  (∀ x, (2 * x^2 + (4 - m) * x + 4 - m > 0) ∨ (m * x > 0)) ↔ m < 4 :=
sorry

end function_positivity_range_l163_163906


namespace expression_evaluation_l163_163394

theorem expression_evaluation :
  (4 * 6 / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) - 1 = 0) :=
by sorry

end expression_evaluation_l163_163394


namespace quadrilateral_angle_l163_163884

theorem quadrilateral_angle (x y : ℝ) (h1 : 3 * x ^ 2 - x + 4 = 5) (h2 : x ^ 2 + y ^ 2 = 9) :
  x = (1 + Real.sqrt 13) / 6 :=
by
  sorry

end quadrilateral_angle_l163_163884


namespace total_hours_worked_l163_163208

-- Definition of the given conditions.
def hours_software : ℕ := 24
def hours_help_user : ℕ := 17
def percentage_other_services : ℚ := 0.4

-- Statement to prove.
theorem total_hours_worked : ∃ (T : ℕ), hours_software + hours_help_user + percentage_other_services * T = T ∧ T = 68 :=
by {
  -- The proof will go here.
  sorry
}

end total_hours_worked_l163_163208


namespace inequality_holds_l163_163655

variable (a t1 t2 t3 t4 : ℝ)

theorem inequality_holds
  (a_pos : 0 < a)
  (h_a_le : a ≤ 7/9)
  (t1_pos : 0 < t1)
  (t2_pos : 0 < t2)
  (t3_pos : 0 < t3)
  (t4_pos : 0 < t4)
  (h_prod : t1 * t2 * t3 * t4 = a^4) :
  (1 / Real.sqrt (1 + t1) + 1 / Real.sqrt (1 + t2) + 1 / Real.sqrt (1 + t3) + 1 / Real.sqrt (1 + t4)) ≤ (4 / Real.sqrt (1 + a)) :=
by
  sorry 

end inequality_holds_l163_163655


namespace reflect_over_x_axis_l163_163546

def coords (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem reflect_over_x_axis :
  coords (-6, -9) = (-6, 9) :=
by
  sorry

end reflect_over_x_axis_l163_163546


namespace smallest_value_y_l163_163800

theorem smallest_value_y (y : ℝ) : (|y - 8| = 15) → y = -7 :=
by
  sorry

end smallest_value_y_l163_163800


namespace yellow_marbles_at_least_zero_l163_163493

noncomputable def total_marbles := 30
def blue_marbles (n : ℕ) := n / 3
def red_marbles (n : ℕ) := n / 3
def green_marbles := 10
def yellow_marbles (n : ℕ) := n - ((2 * n) / 3 + 10)

-- Conditions
axiom h1 : total_marbles % 3 = 0
axiom h2 : total_marbles = 30

-- Prove the smallest number of yellow marbles is 0
theorem yellow_marbles_at_least_zero : yellow_marbles total_marbles = 0 := by
  sorry

end yellow_marbles_at_least_zero_l163_163493


namespace rectangle_sides_equal_perimeter_and_area_l163_163721

theorem rectangle_sides_equal_perimeter_and_area (x y : ℕ) (h : 2 * x + 2 * y = x * y) : 
    (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 6) ∨ (x = 4 ∧ y = 4) :=
by sorry

end rectangle_sides_equal_perimeter_and_area_l163_163721


namespace find_slope_of_line_l_l163_163285

-- Define the vectors OA and OB
def OA : ℝ × ℝ := (4, 1)
def OB : ℝ × ℝ := (2, -3)

-- The slope k is such that the lengths of projections of OA and OB on line l are equal
theorem find_slope_of_line_l (k : ℝ) :
  (|4 + k| = |2 - 3 * k|) → (k = 3 ∨ k = -1/2) :=
by
  -- Intentionally leave the proof out
  sorry

end find_slope_of_line_l_l163_163285


namespace missing_number_in_proportion_l163_163221

theorem missing_number_in_proportion (x : ℝ) :
  (2 / x) = ((4 / 3) / (10 / 3)) → x = 5 :=
by sorry

end missing_number_in_proportion_l163_163221


namespace fruit_seller_price_l163_163317

theorem fruit_seller_price (C : ℝ) (h1 : 1.05 * C = 14.823529411764707) : 
  0.85 * C = 12 := 
sorry

end fruit_seller_price_l163_163317


namespace walkway_area_l163_163222

/--
Tara has four rows of three 8-feet by 3-feet flower beds in her garden. The beds are separated
and surrounded by 2-feet-wide walkways. Prove that the total area of the walkways is 416 square feet.
-/
theorem walkway_area :
  let flower_bed_width := 8
  let flower_bed_height := 3
  let num_rows := 4
  let num_columns := 3
  let walkway_width := 2
  let total_width := (num_columns * flower_bed_width) + (num_columns + 1) * walkway_width
  let total_height := (num_rows * flower_bed_height) + (num_rows + 1) * walkway_width
  let total_garden_area := total_width * total_height
  let flower_bed_area := flower_bed_width * flower_bed_height * num_rows * num_columns
  total_garden_area - flower_bed_area = 416 :=
by
  -- Proof omitted
  sorry

end walkway_area_l163_163222


namespace dogs_not_liking_any_l163_163920

variables (totalDogs : ℕ) (dogsLikeWatermelon : ℕ) (dogsLikeSalmon : ℕ) (dogsLikeBothSalmonWatermelon : ℕ)
          (dogsLikeChicken : ℕ) (dogsLikeWatermelonNotSalmon : ℕ) (dogsLikeSalmonChickenNotWatermelon : ℕ)

theorem dogs_not_liking_any : totalDogs = 80 → dogsLikeWatermelon = 21 → dogsLikeSalmon = 58 →
  dogsLikeBothSalmonWatermelon = 12 → dogsLikeChicken = 15 →
  dogsLikeWatermelonNotSalmon = 7 → dogsLikeSalmonChickenNotWatermelon = 10 →
  (totalDogs - ((dogsLikeSalmon - (dogsLikeBothSalmonWatermelon + dogsLikeSalmonChickenNotWatermelon)) +
                (dogsLikeWatermelon - (dogsLikeBothSalmonWatermelon + dogsLikeWatermelonNotSalmon)) +
                (dogsLikeChicken - (dogsLikeWatermelonNotSalmon + dogsLikeSalmonChickenNotWatermelon)) +
                dogsLikeBothSalmonWatermelon + dogsLikeWatermelonNotSalmon + dogsLikeSalmonChickenNotWatermelon)) = 13 :=
by
  intros h_totalDogs h_dogsLikeWatermelon h_dogsLikeSalmon h_dogsLikeBothSalmonWatermelon 
         h_dogsLikeChicken h_dogsLikeWatermelonNotSalmon h_dogsLikeSalmonChickenNotWatermelon
  sorry

end dogs_not_liking_any_l163_163920


namespace value_of_y_l163_163662

theorem value_of_y (x y : ℝ) (h1 : x ^ (2 * y) = 81) (h2 : x = 9) : y = 1 :=
sorry

end value_of_y_l163_163662


namespace sum_of_cubes_l163_163592

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end sum_of_cubes_l163_163592


namespace ratio_thursday_to_wednesday_l163_163456

variables (T : ℕ)

def time_studied_wednesday : ℕ := 2
def time_studied_thursday : ℕ := T
def time_studied_friday : ℕ := T / 2
def time_studied_weekend : ℕ := 2 + T + T / 2
def total_time_studied : ℕ := 22

theorem ratio_thursday_to_wednesday (h : 
  time_studied_wednesday + time_studied_thursday + time_studied_friday + time_studied_weekend = total_time_studied
) : (T : ℚ) / time_studied_wednesday = 3 := by
  sorry

end ratio_thursday_to_wednesday_l163_163456


namespace sum_of_three_numbers_l163_163045

theorem sum_of_three_numbers (S F T : ℕ) (h1 : S = 150) (h2 : F = 2 * S) (h3 : T = F / 3) :
  F + S + T = 550 :=
by
  sorry

end sum_of_three_numbers_l163_163045


namespace lada_vs_elevator_l163_163414

def Lada_speed_ratio (V U : ℝ) (S : ℝ) : Prop :=
  (∃ t_wait t_wait' : ℝ,
  ((t_wait = 3*S/U - 3*S/V) ∧ (t_wait' = 7*S/(2*U) - 7*S/V)) ∧
   (t_wait' = 3 * t_wait)) →
  U = 11/4 * V

theorem lada_vs_elevator (V U : ℝ) (S : ℝ) : Lada_speed_ratio V U S :=
sorry

end lada_vs_elevator_l163_163414


namespace cheaper_module_cost_l163_163059

theorem cheaper_module_cost (x : ℝ) :
  (21 * x + 10 = 62.50) → (x = 2.50) :=
by
  intro h
  sorry

end cheaper_module_cost_l163_163059


namespace centroid_inverse_square_sum_l163_163942

theorem centroid_inverse_square_sum
  (α β γ p q r : ℝ)
  (h1 : 1/α^2 + 1/β^2 + 1/γ^2 = 1)
  (hp : p = α / 3)
  (hq : q = β / 3)
  (hr : r = γ / 3) :
  (1/p^2 + 1/q^2 + 1/r^2 = 9) :=
sorry

end centroid_inverse_square_sum_l163_163942


namespace total_whales_correct_l163_163052

def first_trip_male_whales : ℕ := 28
def first_trip_female_whales : ℕ := 2 * first_trip_male_whales
def first_trip_total_whales : ℕ := first_trip_male_whales + first_trip_female_whales

def second_trip_baby_whales : ℕ := 8
def second_trip_parent_whales : ℕ := 2 * second_trip_baby_whales
def second_trip_total_whales : ℕ := second_trip_baby_whales + second_trip_parent_whales

def third_trip_male_whales : ℕ := first_trip_male_whales / 2
def third_trip_female_whales : ℕ := first_trip_female_whales
def third_trip_total_whales : ℕ := third_trip_male_whales + third_trip_female_whales

def total_whales_seen : ℕ :=
  first_trip_total_whales + second_trip_total_whales + third_trip_total_whales

theorem total_whales_correct : total_whales_seen = 178 := by
  sorry

end total_whales_correct_l163_163052


namespace original_number_of_girls_l163_163170

theorem original_number_of_girls (b g : ℕ) (h1 : b = g)
                                (h2 : 3 * (g - 25) = b)
                                (h3 : 6 * (b - 60) = g - 25) :
  g = 67 :=
by sorry

end original_number_of_girls_l163_163170


namespace children_tickets_sold_l163_163047

theorem children_tickets_sold {A C : ℕ} (h1 : 6 * A + 4 * C = 104) (h2 : A + C = 21) : C = 11 :=
by
  sorry

end children_tickets_sold_l163_163047


namespace largest_possible_markers_in_package_l163_163405

theorem largest_possible_markers_in_package (alex_markers jordan_markers : ℕ) 
  (h1 : alex_markers = 56)
  (h2 : jordan_markers = 42) :
  Nat.gcd alex_markers jordan_markers = 14 :=
by
  sorry

end largest_possible_markers_in_package_l163_163405


namespace solve_for_x_l163_163246

-- Define the conditions
def percentage15_of_25 : ℝ := 0.15 * 25
def percentage12 (x : ℝ) : ℝ := 0.12 * x
def condition (x : ℝ) : Prop := percentage15_of_25 + percentage12 x = 9.15

-- The target statement to prove
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 45 :=
by 
  -- The proof is omitted
  sorry

end solve_for_x_l163_163246


namespace age_of_new_teacher_l163_163670

-- Definitions of conditions
def avg_age_20_teachers (sum_of_ages : ℕ) : Prop :=
  sum_of_ages = 49 * 20

def avg_age_21_teachers (sum_of_ages : ℕ) : Prop :=
  sum_of_ages = 48 * 21

-- The proof goal
theorem age_of_new_teacher (sum_age_20 : ℕ) (sum_age_21 : ℕ) (h1 : avg_age_20_teachers sum_age_20) (h2 : avg_age_21_teachers sum_age_21) : 
  sum_age_21 - sum_age_20 = 28 :=
sorry

end age_of_new_teacher_l163_163670


namespace polynomial_sum_of_squares_l163_163949

theorem polynomial_sum_of_squares (P : Polynomial ℝ) 
  (hP : ∀ x : ℝ, 0 ≤ P.eval x) : 
  ∃ (f g : Polynomial ℝ), P = f * f + g * g := 
sorry

end polynomial_sum_of_squares_l163_163949


namespace geometric_sum_3030_l163_163645

theorem geometric_sum_3030 {a r : ℝ}
  (h1 : a * (1 - r ^ 1010) / (1 - r) = 300)
  (h2 : a * (1 - r ^ 2020) / (1 - r) = 540) :
  a * (1 - r ^ 3030) / (1 - r) = 732 :=
sorry

end geometric_sum_3030_l163_163645


namespace chicken_problem_l163_163382

theorem chicken_problem (x y z : ℕ) :
  x + y + z = 100 ∧ 5 * x + 3 * y + z / 3 = 100 → 
  (x = 0 ∧ y = 25 ∧ z = 75) ∨ 
  (x = 12 ∧ y = 4 ∧ z = 84) ∨ 
  (x = 8 ∧ y = 11 ∧ z = 81) ∨ 
  (x = 4 ∧ y = 18 ∧ z = 78) := 
sorry

end chicken_problem_l163_163382


namespace hydrogen_to_oxygen_ratio_l163_163290

theorem hydrogen_to_oxygen_ratio (total_mass_water mass_hydrogen mass_oxygen : ℝ) 
(h1 : total_mass_water = 117)
(h2 : mass_hydrogen = 13)
(h3 : mass_oxygen = total_mass_water - mass_hydrogen) :
(mass_hydrogen / mass_oxygen) = 1 / 8 := 
sorry

end hydrogen_to_oxygen_ratio_l163_163290


namespace albert_large_pizzas_l163_163062

-- Define the conditions
def large_pizza_slices : ℕ := 16
def small_pizza_slices : ℕ := 8
def num_small_pizzas : ℕ := 2
def total_slices_eaten : ℕ := 48

-- Define the question and requirement to prove
def number_of_large_pizzas (L : ℕ) : Prop :=
  large_pizza_slices * L + small_pizza_slices * num_small_pizzas = total_slices_eaten

theorem albert_large_pizzas :
  number_of_large_pizzas 2 :=
by
  sorry

end albert_large_pizzas_l163_163062


namespace log_inequality_solution_l163_163445

variable {a x : ℝ}

theorem log_inequality_solution (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  (1 + Real.log (a ^ x - 1) / Real.log 2 ≤ Real.log (4 - a ^ x) / Real.log 2) →
  ((1 < a ∧ x ≤ Real.log (7 / 4) / Real.log a) ∨ (0 < a ∧ a < 1 ∧ x ≥ Real.log (7 / 4) / Real.log a)) :=
sorry

end log_inequality_solution_l163_163445


namespace value_of_expression_l163_163643

theorem value_of_expression : 10^2 + 10 + 1 = 111 :=
by
  sorry

end value_of_expression_l163_163643


namespace right_triangle_medians_right_triangle_l163_163551

theorem right_triangle_medians_right_triangle (a b c s_a s_b s_c : ℝ)
  (hyp_a_lt_b : a < b) (hyp_b_lt_c : b < c)
  (h_c_hypotenuse : c = Real.sqrt (a^2 + b^2))
  (h_sa : s_a^2 = b^2 + (a / 2)^2)
  (h_sb : s_b^2 = a^2 + (b / 2)^2)
  (h_sc : s_c^2 = (a^2 + b^2) / 4) :
  b = a * Real.sqrt 2 :=
by
  sorry

end right_triangle_medians_right_triangle_l163_163551


namespace prob_A_exactly_once_l163_163865

theorem prob_A_exactly_once (P : ℚ) (h : 1 - (1 - P)^3 = 63 / 64) : 
  (3 * P * (1 - P)^2 = 9 / 64) :=
by
  sorry

end prob_A_exactly_once_l163_163865


namespace simplify_and_evaluate_l163_163319

noncomputable def a := 3

theorem simplify_and_evaluate : (a^2 / (a + 1) - 1 / (a + 1)) = 2 := by
  sorry

end simplify_and_evaluate_l163_163319


namespace larger_integer_of_two_integers_diff_8_prod_120_l163_163036

noncomputable def larger_integer (a b : ℕ) : ℕ :=
if a > b then a else b

theorem larger_integer_of_two_integers_diff_8_prod_120 (a b : ℕ) 
  (h_diff : a - b = 8) 
  (h_product : a * b = 120) 
  (h_positive_a : 0 < a) 
  (h_positive_b : 0 < b) : larger_integer a b = 20 := by
  sorry

end larger_integer_of_two_integers_diff_8_prod_120_l163_163036


namespace how_many_unanswered_l163_163929

theorem how_many_unanswered (c w u : ℕ) (h1 : 25 + 5 * c - 2 * w = 95)
                            (h2 : 6 * c + u = 110) (h3 : c + w + u = 30) : u = 10 :=
by
  sorry

end how_many_unanswered_l163_163929


namespace total_interest_proof_l163_163001

open Real

def initial_investment : ℝ := 10000
def interest_6_months : ℝ := 0.02 * initial_investment
def reinvested_amount_6_months : ℝ := initial_investment + interest_6_months
def interest_10_months : ℝ := 0.03 * reinvested_amount_6_months
def reinvested_amount_10_months : ℝ := reinvested_amount_6_months + interest_10_months
def interest_18_months : ℝ := 0.04 * reinvested_amount_10_months

def total_interest : ℝ := interest_6_months + interest_10_months + interest_18_months

theorem total_interest_proof : total_interest = 926.24 := by
    sorry

end total_interest_proof_l163_163001


namespace no_positive_integer_solutions_l163_163367

theorem no_positive_integer_solutions:
    ∀ x y : ℕ, x > 0 → y > 0 → x^2 + 2 * y^2 = 2 * x^3 - x → false :=
by
  sorry

end no_positive_integer_solutions_l163_163367


namespace find_window_cost_l163_163243

-- Definitions (conditions)
def total_damages : ℕ := 1450
def cost_of_tire : ℕ := 250
def number_of_tires : ℕ := 3
def cost_of_tires := number_of_tires * cost_of_tire

-- The cost of the window that needs to be proven
def window_cost := total_damages - cost_of_tires

-- We state the theorem that the window costs $700 and provide a sorry as placeholder for its proof
theorem find_window_cost : window_cost = 700 :=
by sorry

end find_window_cost_l163_163243


namespace polynomial_value_l163_163868

theorem polynomial_value 
  (x : ℝ) 
  (h1 : x = (1 + (1994 : ℝ).sqrt) / 2) : 
  (4 * x ^ 3 - 1997 * x - 1994) ^ 20001 = -1 := 
  sorry

end polynomial_value_l163_163868


namespace find_k_l163_163297

-- Define the vectors
def e1 : ℝ × ℝ := (1, 0)
def e2 : ℝ × ℝ := (0, 1)

def a : ℝ × ℝ := (e1.1 - 2 * e2.1, e1.2 - 2 * e2.2)
def b (k : ℝ) : ℝ × ℝ := (k * e1.1 + e2.1, k * e1.2 + e2.2)

-- Define the parallel condition
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the problem translated to a Lean theorem
theorem find_k (k : ℝ) : 
  parallel a (b k) -> k = -1 / 2 := by
  sorry

end find_k_l163_163297


namespace sin_585_eq_neg_sqrt_two_div_two_l163_163478

theorem sin_585_eq_neg_sqrt_two_div_two : Real.sin (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_585_eq_neg_sqrt_two_div_two_l163_163478


namespace smallest_multiple_of_6_and_9_l163_163628

theorem smallest_multiple_of_6_and_9 : ∃ n : ℕ, n > 0 ∧ (n % 6 = 0) ∧ (n % 9 = 0) ∧ ∀ m : ℕ, m > 0 ∧ (m % 6 = 0) ∧ (m % 9 = 0) → n ≤ m :=
  by
    sorry

end smallest_multiple_of_6_and_9_l163_163628


namespace pyramid_surface_area_l163_163708

theorem pyramid_surface_area
  (base_side_length : ℝ)
  (peak_height : ℝ)
  (base_area : ℝ)
  (slant_height : ℝ)
  (triangular_face_area : ℝ)
  (total_surface_area : ℝ)
  (h1 : base_side_length = 10)
  (h2 : peak_height = 12)
  (h3 : base_area = base_side_length ^ 2)
  (h4 : slant_height = Real.sqrt (peak_height ^ 2 + (base_side_length / 2) ^ 2))
  (h5 : triangular_face_area = 0.5 * base_side_length * slant_height)
  (h6 : total_surface_area = base_area + 4 * triangular_face_area)
  : total_surface_area = 360 := 
sorry

end pyramid_surface_area_l163_163708


namespace extremum_at_one_and_value_at_two_l163_163537

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_at_one_and_value_at_two (a b : ℝ) (h_deriv : 3 + 2*a + b = 0) (h_value : 1 + a + b + a^2 = 10) : 
  f 2 a b = 18 := 
by 
  sorry

end extremum_at_one_and_value_at_two_l163_163537


namespace no_real_number_pairs_satisfy_equation_l163_163934

theorem no_real_number_pairs_satisfy_equation :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
  ¬ (1 / a + 1 / b = 1 / (2 * a + 3 * b)) :=
by
  intros a b ha hb
  sorry

end no_real_number_pairs_satisfy_equation_l163_163934


namespace FerrisWheelCostIsTwo_l163_163960

noncomputable def costFerrisWheel (rollerCoasterCost multipleRideDiscount coupon totalTicketsBought : ℝ) : ℝ :=
  totalTicketsBought + multipleRideDiscount + coupon - rollerCoasterCost

theorem FerrisWheelCostIsTwo :
  let rollerCoasterCost := 7.0
  let multipleRideDiscount := 1.0
  let coupon := 1.0
  let totalTicketsBought := 7.0
  costFerrisWheel rollerCoasterCost multipleRideDiscount coupon totalTicketsBought = 2.0 :=
by
  sorry

end FerrisWheelCostIsTwo_l163_163960


namespace possible_second_game_scores_count_l163_163428

theorem possible_second_game_scores_count :
  ∃ (A1 A3 B2 : ℕ),
  (A1 + A3 = 22) ∧ (B2 = 11) ∧ (A1 < 11) ∧ (A3 < 11) ∧ ((B2 - A2 = 2) ∨ (B2 >= A2 + 2)) ∧ (A1 + B1 + A2 + B2 + A3 + B3 = 62) :=
  sorry

end possible_second_game_scores_count_l163_163428


namespace sum_real_imag_l163_163727

theorem sum_real_imag (z : ℂ) (hz : z = 3 - 4 * I) : z.re + z.im = -1 :=
by {
  -- Because the task asks for no proof, we're leaving it with 'sorry'.
  sorry
}

end sum_real_imag_l163_163727


namespace red_paint_intensity_l163_163315

variable (I : ℝ) -- Intensity of the original paint
variable (P : ℝ) -- Volume of the original paint
variable (fraction_replaced : ℝ := 1) -- Fraction of original paint replaced
variable (new_intensity : ℝ := 20) -- New paint intensity
variable (replacement_intensity : ℝ := 20) -- Replacement paint intensity

theorem red_paint_intensity : new_intensity = replacement_intensity :=
by
  -- Placeholder for the actual proof
  sorry

end red_paint_intensity_l163_163315


namespace number_of_spinsters_l163_163822

-- Given conditions
variables (S C : ℕ)
axiom ratio_condition : S / C = 2 / 9
axiom difference_condition : C = S + 63

-- Theorem to prove
theorem number_of_spinsters : S = 18 :=
sorry

end number_of_spinsters_l163_163822


namespace scientific_notation_of_120000_l163_163350

theorem scientific_notation_of_120000 : 
  (120000 : ℝ) = 1.2 * 10^5 := 
by 
  sorry

end scientific_notation_of_120000_l163_163350


namespace part1_part2_l163_163998

noncomputable def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - Real.log x + x - a

theorem part1 (x : ℝ) (a : ℝ) :
    (∀ x > 0, f x a ≥ 0) → a ≤ Real.exp 1 + 1 :=
sorry

theorem part2 (x1 x2 : ℝ) (a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 < 1 → 1 < x2 → x1 * x2 < 1 :=
sorry

end part1_part2_l163_163998


namespace expected_value_coins_heads_l163_163335

noncomputable def expected_value_cents : ℝ :=
  let values := [1, 5, 10, 25, 50, 100]
  let probability_heads := 1 / 2
  probability_heads * (values.sum : ℝ)

theorem expected_value_coins_heads : expected_value_cents = 95.5 := by
  sorry

end expected_value_coins_heads_l163_163335


namespace cannot_be_expressed_as_difference_of_squares_l163_163114

theorem cannot_be_expressed_as_difference_of_squares : 
  ¬ ∃ (a b : ℤ), 2006 = a^2 - b^2 :=
sorry

end cannot_be_expressed_as_difference_of_squares_l163_163114


namespace quadratic_has_distinct_real_roots_l163_163709

theorem quadratic_has_distinct_real_roots :
  let a := 5
  let b := 14
  let c := 5
  let discriminant := b^2 - 4 * a * c
  discriminant > 0 := 
by
  sorry

end quadratic_has_distinct_real_roots_l163_163709


namespace valid_three_digit_numbers_count_l163_163885

noncomputable def count_valid_three_digit_numbers : ℕ :=
  let total_three_digit_numbers := 900
  let excluded_numbers := 81 + 72
  total_three_digit_numbers - excluded_numbers

theorem valid_three_digit_numbers_count :
  count_valid_three_digit_numbers = 747 :=
by
  sorry

end valid_three_digit_numbers_count_l163_163885


namespace length_of_second_train_is_229_95_l163_163422

noncomputable def length_of_second_train (length_first_train : ℝ) 
                                          (speed_first_train : ℝ) 
                                          (speed_second_train : ℝ) 
                                          (time_to_cross : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train * (1000 / 3600)
  let speed_second_train_mps := speed_second_train * (1000 / 3600)
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance_covered := relative_speed * time_to_cross
  total_distance_covered - length_first_train

theorem length_of_second_train_is_229_95 :
  length_of_second_train 270 120 80 9 = 229.95 :=
by
  sorry

end length_of_second_train_is_229_95_l163_163422


namespace find_e1_l163_163608

-- Definitions related to the problem statement
variable (P F1 F2 : Type)
variable (cos_angle : ℝ)
variable (e1 e2 : ℝ)

-- Conditions
def cosine_angle_condition := cos_angle = 3 / 5
def eccentricity_relation := e2 = 2 * e1

-- Theorem that needs to be proved
theorem find_e1 (h_cos : cosine_angle_condition cos_angle)
                (h_ecc_rel : eccentricity_relation e1 e2) :
  e1 = Real.sqrt 10 / 5 :=
by
  sorry

end find_e1_l163_163608


namespace simplify_expression_l163_163726

theorem simplify_expression (x : ℝ) : 
  (4 * x + 6 * x^3 + 8 - (3 - 6 * x^3 - 4 * x)) = 12 * x^3 + 8 * x + 5 := 
by
  sorry

end simplify_expression_l163_163726


namespace prob_win_all_6_games_prob_win_exactly_5_out_of_6_games_l163_163520

noncomputable def prob_win_single_game : ℚ := 7 / 10
noncomputable def prob_lose_single_game : ℚ := 3 / 10

theorem prob_win_all_6_games : (prob_win_single_game ^ 6) = 117649 / 1000000 :=
by
  sorry

theorem prob_win_exactly_5_out_of_6_games : (6 * (prob_win_single_game ^ 5) * prob_lose_single_game) = 302526 / 1000000 :=
by
  sorry

end prob_win_all_6_games_prob_win_exactly_5_out_of_6_games_l163_163520


namespace train_speed_approx_l163_163873

noncomputable def man_speed_kmh : ℝ := 3
noncomputable def man_speed_ms : ℝ := (man_speed_kmh * 1000) / 3600
noncomputable def train_length : ℝ := 900
noncomputable def time_to_cross : ℝ := 53.99568034557235
noncomputable def train_speed_ms := (train_length / time_to_cross) + man_speed_ms
noncomputable def train_speed_kmh := (train_speed_ms * 3600) / 1000

theorem train_speed_approx :
  abs (train_speed_kmh - 63.009972) < 1e-5 := sorry

end train_speed_approx_l163_163873


namespace mod_calculation_l163_163002

theorem mod_calculation : (9^7 + 8^8 + 7^9) % 5 = 2 := by
  sorry

end mod_calculation_l163_163002


namespace intersection_of_A_and_B_l163_163055

def I := {x : ℝ | true}
def A := {x : ℝ | x * (x - 1) ≥ 0}
def B := {x : ℝ | x > 1}
def C := {x : ℝ | x > 1}

theorem intersection_of_A_and_B : A ∩ B = C := by
  sorry

end intersection_of_A_and_B_l163_163055


namespace number_of_flute_players_l163_163584

theorem number_of_flute_players (F T B D C H : ℕ)
  (hT : T = 3 * F)
  (hB : B = T - 8)
  (hD : D = B + 11)
  (hC : C = 2 * F)
  (hH : H = B + 3)
  (h_total : F + T + B + D + C + H = 65) :
  F = 6 :=
by
  sorry

end number_of_flute_players_l163_163584


namespace magic_square_d_e_sum_l163_163607

theorem magic_square_d_e_sum 
  (S : ℕ)
  (a b c d e : ℕ)
  (h1 : S = 45 + d)
  (h2 : S = 51 + e) :
  d + e = 57 :=
by
  sorry

end magic_square_d_e_sum_l163_163607


namespace calculate_T6_l163_163986

noncomputable def T (y : ℝ) (m : ℕ) : ℝ := y^m + 1 / y^m

theorem calculate_T6 (y : ℝ) (h : y + 1 / y = 5) : T y 6 = 12098 := 
by
  sorry

end calculate_T6_l163_163986


namespace student_D_most_stable_l163_163689

-- Define the variances for students A, B, C, and D
def SA_squared : ℝ := 2.1
def SB_squared : ℝ := 3.5
def SC_squared : ℝ := 9
def SD_squared : ℝ := 0.7

-- Theorem stating that student D has the most stable performance
theorem student_D_most_stable :
  SD_squared < SA_squared ∧ SD_squared < SB_squared ∧ SD_squared < SC_squared := by
  sorry

end student_D_most_stable_l163_163689


namespace simplified_result_l163_163082

theorem simplified_result (a b M : ℝ) (h1 : (2 * a) / (a ^ 2 - b ^ 2) - 1 / M = 1 / (a - b))
  (h2 : M - (a - b) = 2 * b) : (2 * a) / (a ^ 2 - b ^ 2) - 1 / (a - b) = 1 / (a + b) :=
by
  sorry

end simplified_result_l163_163082


namespace piles_3_stones_impossible_l163_163508

theorem piles_3_stones_impossible :
  ∀ n : ℕ, ∀ piles : ℕ → ℕ,
  (piles 0 = 1001) →
  (∀ k : ℕ, k > 0 → ∃ i j : ℕ, piles (k-1) > 1 → piles k = i + j ∧ i > 0 ∧ j > 0) →
  ¬ (∀ m : ℕ, piles m ≠ 3) :=
by
  sorry

end piles_3_stones_impossible_l163_163508


namespace mutually_exclusive_probability_zero_l163_163505

theorem mutually_exclusive_probability_zero {A B : Prop} (p1 p2 : ℝ) 
  (hA : 0 ≤ p1 ∧ p1 ≤ 1) 
  (hB : 0 ≤ p2 ∧ p2 ≤ 1) 
  (hAB : A ∧ B → False) : 
  (A ∧ B) = False :=
by
  sorry

end mutually_exclusive_probability_zero_l163_163505


namespace average_members_remaining_l163_163301

theorem average_members_remaining :
  let initial_members := [7, 8, 10, 13, 6, 10, 12, 9]
  let members_leaving := [1, 2, 1, 2, 1, 2, 1, 2]
  let remaining_members := List.map (λ (x, y) => x - y) (List.zip initial_members members_leaving)
  let total_remaining := List.foldl Nat.add 0 remaining_members
  let num_families := initial_members.length
  total_remaining / num_families = 63 / 8 := by
    sorry

end average_members_remaining_l163_163301


namespace solve_inequality_l163_163753

theorem solve_inequality : 
  {x : ℝ | -4 * x^2 + 7 * x + 2 < 0} = {x : ℝ | x < -1/4} ∪ {x : ℝ | 2 < x} :=
by
  sorry

end solve_inequality_l163_163753


namespace max_value_of_expression_l163_163741

-- Define the variables and condition.
variable (x y z : ℝ)
variable (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1)

-- State the theorem.
theorem max_value_of_expression :
  (8 * x + 5 * y + 15 * z) ≤ 4.54 :=
sorry

end max_value_of_expression_l163_163741


namespace a5_equals_2_l163_163005

variable {a : ℕ → ℝ}  -- a_n represents the nth term of the arithmetic sequence

-- Define the arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n m : ℕ, a (n + 1) = a 1 + n * d 

-- Given condition
axiom arithmetic_condition (h : is_arithmetic_sequence a) : a 1 + a 5 + a 9 = 6

-- The goal is to prove a_5 = 2
theorem a5_equals_2 (h : is_arithmetic_sequence a) (h_cond : a 1 + a 5 + a 9 = 6) : a 5 = 2 := 
by 
  sorry

end a5_equals_2_l163_163005


namespace digit_difference_l163_163581

theorem digit_difference (X Y : ℕ) (h_digits : 0 ≤ X ∧ X < 10 ∧ 0 ≤ Y ∧ Y < 10) (h_diff :  (10 * X + Y) - (10 * Y + X) = 45) : X - Y = 5 :=
sorry

end digit_difference_l163_163581


namespace no_possible_arrangement_l163_163181

theorem no_possible_arrangement :
  ¬ ∃ (a : Fin 9 → ℕ),
    (∀ i, 1 ≤ a i ∧ a i ≤ 9) ∧
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    (∀ i, (a i + a ((i + 1) % 9) + a ((i + 2) % 9)) % 3 = 0) ∧
    (∀ i, (a i + a ((i + 1) % 9) + a ((i + 2) % 9)) > 12) :=
  sorry

end no_possible_arrangement_l163_163181


namespace ratio_is_one_to_five_l163_163476

def ratio_of_minutes_to_hour (twelve_minutes : ℕ) (one_hour : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd twelve_minutes one_hour
  (twelve_minutes / gcd, one_hour / gcd)

theorem ratio_is_one_to_five : ratio_of_minutes_to_hour 12 60 = (1, 5) := 
by 
  sorry

end ratio_is_one_to_five_l163_163476


namespace bob_shucks_240_oysters_in_2_hours_l163_163440

-- Definitions based on conditions provided:
def oysters_per_minute (oysters : ℕ) (minutes : ℕ) : ℕ :=
  oysters / minutes

def minutes_in_hour : ℕ :=
  60

def oysters_in_two_hours (oysters_per_minute : ℕ) (hours : ℕ) : ℕ :=
  oysters_per_minute * (hours * minutes_in_hour)

-- Parameters given in the problem:
def initial_oysters : ℕ := 10
def initial_minutes : ℕ := 5
def hours : ℕ := 2

-- The main theorem we need to prove:
theorem bob_shucks_240_oysters_in_2_hours :
  oysters_in_two_hours (oysters_per_minute initial_oysters initial_minutes) hours = 240 :=
by
  sorry

end bob_shucks_240_oysters_in_2_hours_l163_163440


namespace number_of_numbers_l163_163060

theorem number_of_numbers (n : ℕ) (S : ℕ) 
  (h1 : (S + 26) / n = 16) 
  (h2 : (S + 46) / n = 18) : 
  n = 10 := 
by 
  -- placeholder for the proof
  sorry

end number_of_numbers_l163_163060


namespace average_weight_men_women_l163_163145

theorem average_weight_men_women (n_men n_women : ℕ) (avg_weight_men avg_weight_women : ℚ)
  (h_men : n_men = 8) (h_women : n_women = 6) (h_avg_weight_men : avg_weight_men = 190)
  (h_avg_weight_women : avg_weight_women = 120) :
  (n_men * avg_weight_men + n_women * avg_weight_women) / (n_men + n_women) = 160 := 
by
  sorry

end average_weight_men_women_l163_163145


namespace sum_of_digits_nine_ab_l163_163553

noncomputable def sum_digits_base_10 (n : ℕ) : ℕ :=
-- Function to compute the sum of digits of a number in base 10
sorry

def a : ℕ := 6 * ((10^1500 - 1) / 9)

def b : ℕ := 3 * ((10^1500 - 1) / 9)

def nine_ab : ℕ := 9 * a * b

theorem sum_of_digits_nine_ab :
  sum_digits_base_10 nine_ab = 13501 :=
sorry

end sum_of_digits_nine_ab_l163_163553


namespace even_odd_difference_l163_163937

def even_sum_n (n : ℕ) : ℕ := (n * (n + 1))
def odd_sum_n (n : ℕ) : ℕ := n * n

theorem even_odd_difference : even_sum_n 100 - odd_sum_n 100 = 100 := by
  -- The proof goes here
  sorry

end even_odd_difference_l163_163937


namespace no_integer_solutions_l163_163442

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), (x ≠ 1 ∧ (x^7 - 1) / (x - 1) = (y^5 - 1)) :=
sorry

end no_integer_solutions_l163_163442


namespace molecular_weight_of_one_mole_l163_163922

theorem molecular_weight_of_one_mole (total_weight : ℝ) (number_of_moles : ℕ) 
    (h : total_weight = 204) (n : number_of_moles = 3) : 
    (total_weight / number_of_moles) = 68 :=
by
  have h_weight : total_weight = 204 := h
  have h_moles : number_of_moles = 3 := n
  rw [h_weight, h_moles]
  norm_num

end molecular_weight_of_one_mole_l163_163922


namespace total_amount_l163_163376

theorem total_amount (a b c total first : ℕ)
  (h1 : a = 1 / 2) (h2 : b = 2 / 3) (h3 : c = 3 / 4)
  (h4 : first = 204)
  (ratio_sum : a * 12 + b * 12 + c * 12 = 23)
  (first_ratio : a * 12 = 6) :
  total = 23 * (first / 6) → total = 782 :=
by 
  sorry

end total_amount_l163_163376


namespace right_angled_triangle_l163_163652

theorem right_angled_triangle (a b c : ℕ) (h₀ : a = 7) (h₁ : b = 9) (h₂ : c = 13) :
  a^2 + b^2 ≠ c^2 :=
by
  sorry

end right_angled_triangle_l163_163652


namespace donovan_points_needed_l163_163431

-- Definitions based on conditions
def average_points := 26
def games_played := 15
def total_games := 20
def goal_average := 30

-- Assertion
theorem donovan_points_needed :
  let total_points_needed := goal_average * total_games
  let points_already_scored := average_points * games_played
  let remaining_games := total_games - games_played
  let remaining_points_needed := total_points_needed - points_already_scored
  let points_per_game_needed := remaining_points_needed / remaining_games
  points_per_game_needed = 42 :=
  by
    -- Proof skipped
    sorry

end donovan_points_needed_l163_163431


namespace trajectory_of_A_l163_163379

theorem trajectory_of_A (A B C : (ℝ × ℝ)) (x y : ℝ) : 
  B = (-2, 0) ∧ C = (2, 0) ∧ (dist A (0, 0) = 3) → 
  (x, y) = A → 
  x^2 + y^2 = 9 ∧ y ≠ 0 := 
sorry

end trajectory_of_A_l163_163379


namespace tom_split_number_of_apples_l163_163087

theorem tom_split_number_of_apples
    (S : ℕ)
    (h1 : S = 8 * A)
    (h2 : A * 5 / 8 / 2 = 5) :
    A = 2 :=
by
  sorry

end tom_split_number_of_apples_l163_163087


namespace total_surface_area_calc_l163_163745

/-- Given a cube with a total volume of 1 cubic foot, cut into four pieces by three parallel cuts:
1) The first cut is 0.4 feet from the top.
2) The second cut is 0.3 feet below the first.
3) The third cut is 0.1 feet below the second.
Prove that the total surface area of the new solid is 6 square feet. -/
theorem total_surface_area_calc :
  ∀ (A B C D : ℝ), 
    A = 0.4 → 
    B = 0.3 → 
    C = 0.1 → 
    D = 1 - (A + B + C) → 
    (6 : ℝ) = 6 := 
by 
  intros A B C D hA hB hC hD 
  sorry

end total_surface_area_calc_l163_163745


namespace heejin_most_balls_is_volleyballs_l163_163687

def heejin_basketballs : ℕ := 3
def heejin_volleyballs : ℕ := 5
def heejin_baseballs : ℕ := 1

theorem heejin_most_balls_is_volleyballs :
  heejin_volleyballs > heejin_basketballs ∧ heejin_volleyballs > heejin_baseballs :=
by
  sorry

end heejin_most_balls_is_volleyballs_l163_163687


namespace determine_d_l163_163784

def Q (x d : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 8

theorem determine_d (d : ℝ) : (∃ d, Q (-2) d = 0) → d = -14 := by
  sorry

end determine_d_l163_163784


namespace negation_of_universal_proposition_l163_163509

def P (x : ℝ) : Prop := x^3 + 2 * x ≥ 0

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 0 ≤ x → P x) ↔ (∃ x : ℝ, 0 ≤ x ∧ ¬ P x) :=
by
  sorry

end negation_of_universal_proposition_l163_163509


namespace range_of_a_l163_163464

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 - 2 * x ≤ a^2 - a - 3) ↔ (-1 < a ∧ a < 2) :=
by 
  sorry

end range_of_a_l163_163464


namespace find_X_l163_163980

variable (E X : ℕ)

-- Theorem statement
theorem find_X (hE : E = 9)
              (hSum : E * 100 + E * 10 + E + E * 100 + E * 10 + E = 1798) :
              X = 7 :=
sorry

end find_X_l163_163980


namespace coffee_price_increase_l163_163597

theorem coffee_price_increase (price_first_quarter price_fourth_quarter : ℕ) 
  (h_first : price_first_quarter = 40) (h_fourth : price_fourth_quarter = 60) : 
  ((price_fourth_quarter - price_first_quarter) * 100) / price_first_quarter = 50 := 
by
  -- proof would proceed here
  sorry

end coffee_price_increase_l163_163597


namespace inequality_solution_l163_163451

theorem inequality_solution (x : ℝ) :
  (0 < x ∧ x ≤ 5 / 6 ∨ 2 < x) ↔ 
  ((2 * x) / (x - 2) + (x - 3) / (3 * x) ≥ 2) :=
by
  sorry

end inequality_solution_l163_163451


namespace kola_age_l163_163496

variables (x y : ℕ)

-- Condition 1: Kolya is twice as old as Olya was when Kolya was as old as Olya is now
def condition1 : Prop := x = 2 * (2 * y - x)

-- Condition 2: When Olya is as old as Kolya is now, their combined age will be 36 years.
def condition2 : Prop := (3 * x - y = 36)

theorem kola_age : condition1 x y → condition2 x y → x = 16 :=
by { sorry }

end kola_age_l163_163496


namespace vector_sum_to_zero_l163_163441

variable {V : Type}
variable [AddCommGroup V] [Module ℝ V] {A B C : V}

theorem vector_sum_to_zero (AB BC CA : V) (hAB : AB = B - A) (hBC : BC = C - B) (hCA : CA = A - C) :
  AB + BC + CA = 0 := by
  sorry

end vector_sum_to_zero_l163_163441


namespace digit_7_occurrences_in_range_1_to_2017_l163_163415

-- Define the predicate that checks if a digit appears in a number
def digit_occurrences (d n : Nat) : Nat :=
  Nat.digits 10 n |>.count d

-- Define the range of numbers we are interested in
def range := (List.range' 1 2017)

-- Sum up the occurrences of digit 7 in the defined range
def total_occurrences (d : Nat) (range : List Nat) : Nat :=
  range.foldr (λ n acc => digit_occurrences d n + acc) 0

-- The main theorem to prove
theorem digit_7_occurrences_in_range_1_to_2017 : total_occurrences 7 range = 602 := by
  -- The proof should go here, but we only need to define the statement.
  sorry

end digit_7_occurrences_in_range_1_to_2017_l163_163415


namespace train_speed_is_correct_l163_163951

noncomputable def speed_of_train (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed_is_correct :
  speed_of_train 200 19.99840012798976 = 36.00287976960864 :=
by
  sorry

end train_speed_is_correct_l163_163951


namespace clea_ride_time_l163_163731

-- Definitions: Let c be Clea's walking speed without the bag and s be the speed of the escalator

variables (c s : ℝ)

-- Conditions translated into equations
def distance_without_bag := 80 * c
def distance_with_bag_and_escalator := 38 * (0.7 * c + s)

-- The problem: Prove that the time t for Clea to ride down the escalator while just standing on it with the bag is 57 seconds.
theorem clea_ride_time :
  (38 * (0.7 * c + s) = 80 * c) ->
  (t = 80 * (38 / 53.4)) ->
  t = 57 :=
sorry

end clea_ride_time_l163_163731


namespace horizontal_distance_l163_163788

theorem horizontal_distance (elev_initial elev_final v_ratio h_ratio distance: ℝ)
  (h1 : elev_initial = 100)
  (h2 : elev_final = 1450)
  (h3 : v_ratio = 1)
  (h4 : h_ratio = 2)
  (h5 : distance = 1350) :
  distance * h_ratio = 2700 := by
  sorry

end horizontal_distance_l163_163788


namespace nonagon_diagonals_l163_163143

-- Define the number of sides of the polygon (nonagon)
def num_sides : ℕ := 9

-- Define the formula for the number of diagonals in a convex n-sided polygon
def number_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem
theorem nonagon_diagonals : number_diagonals num_sides = 27 := 
by
--placeholder for the proof
sorry

end nonagon_diagonals_l163_163143


namespace candy_from_sister_is_5_l163_163197

noncomputable def candy_received_from_sister (candy_from_neighbors : ℝ) (pieces_per_day : ℝ) (days : ℕ) : ℝ :=
  pieces_per_day * days - candy_from_neighbors

theorem candy_from_sister_is_5 :
  candy_received_from_sister 11.0 8.0 2 = 5.0 :=
by
  sorry

end candy_from_sister_is_5_l163_163197


namespace ellipse_slope_product_l163_163982

variables {a b x1 y1 x2 y2 : ℝ} (h₁ : a > b) (h₂ : b > 0) (h₃ : (b^2 * x1^2 + a^2 * y1^2 = a^2 * b^2) ∧ (b^2 * x2^2 + a^2 * y2^2 = a^2 * b^2))

theorem ellipse_slope_product : 
  (a > b) → (b > 0) → (b^2 * x1^2 + a^2 * y1^2 = a^2 * b^2) → 
  (b^2 * x2^2 + a^2 * y2^2 = a^2 * b^2) → 
  ( (y1 + y2)/(x1 + x2) ) * ( (y1 - y2)/(x1 - x2) ) = - (b^2 / a^2) :=
by
  intros ha hb hxy1 hxy2
  sorry

end ellipse_slope_product_l163_163982


namespace a_alone_can_finish_job_l163_163254

def work_in_one_day (A B : ℕ) : Prop := 1/A + 1/B = 1/40

theorem a_alone_can_finish_job (A B : ℕ)
  (work_rate : work_in_one_day A B) 
  (together_10_days : 10 * (1/A + 1/B) = 1/4) 
  (a_21_days : 21 * (1/A) = 3/4) : 
  A = 28 := 
sorry

end a_alone_can_finish_job_l163_163254


namespace smallest_n_divisible_by_13_l163_163200

theorem smallest_n_divisible_by_13 : ∃ (n : ℕ), 5^n + n^5 ≡ 0 [MOD 13] ∧ ∀ (m : ℕ), m < n → ¬(5^m + m^5 ≡ 0 [MOD 13]) :=
sorry

end smallest_n_divisible_by_13_l163_163200


namespace find_solutions_l163_163408

theorem find_solutions (x y z : ℝ) :
  (x = 5 / 3 ∧ y = -4 / 3 ∧ z = -4 / 3) ∨
  (x = 4 / 3 ∧ y = 4 / 3 ∧ z = -5 / 3) →
  (x^2 - y * z = abs (y - z) + 1) ∧ 
  (y^2 - z * x = abs (z - x) + 1) ∧ 
  (z^2 - x * y = abs (x - y) + 1) :=
by
  sorry

end find_solutions_l163_163408


namespace greatest_third_side_l163_163852

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end greatest_third_side_l163_163852


namespace average_sales_l163_163589

theorem average_sales (jan feb mar apr : ℝ) (h_jan : jan = 100) (h_feb : feb = 60) (h_mar : mar = 40) (h_apr : apr = 120) : 
  (jan + feb + mar + apr) / 4 = 80 :=
by {
  sorry
}

end average_sales_l163_163589


namespace gcd_linear_combination_l163_163347

theorem gcd_linear_combination (a b : ℤ) : 
  Int.gcd (5 * a + 3 * b) (13 * a + 8 * b) = Int.gcd a b := 
sorry

end gcd_linear_combination_l163_163347


namespace largest_a_value_l163_163946

theorem largest_a_value (a b c : ℝ) (h1 : a + b + c = 7) (h2 : ab + ac + bc = 12) : 
  a ≤ (7 + Real.sqrt 46) / 3 :=
sorry

end largest_a_value_l163_163946


namespace minimum_additional_marbles_l163_163722

-- Definitions corresponding to the conditions
def friends := 12
def initial_marbles := 40

-- Sum of the first n natural numbers definition
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Prove the necessary number of additional marbles
theorem minimum_additional_marbles (h1 : friends = 12) (h2 : initial_marbles = 40) : 
  ∃ additional_marbles, additional_marbles = sum_first_n friends - initial_marbles := by
  sorry

end minimum_additional_marbles_l163_163722


namespace solve_equation_l163_163872

theorem solve_equation : ∃ x : ℝ, (x^3 - ⌊x⌋ = 3) := 
sorry

end solve_equation_l163_163872


namespace number_of_piles_l163_163354

theorem number_of_piles (n : ℕ) (h₁ : 1000 < n) (h₂ : n < 2000)
  (h3 : n % 2 = 1) (h4 : n % 3 = 1) (h5 : n % 4 = 1) 
  (h6 : n % 5 = 1) (h7 : n % 6 = 1) (h8 : n % 7 = 1) (h9 : n % 8 = 1) : 
  ∃ p, p ≠ 1 ∧ p ≠ n ∧ (n % p = 0) ∧ p = 41 :=
by
  sorry

end number_of_piles_l163_163354


namespace find_B_squared_l163_163372

noncomputable def g (x : ℝ) : ℝ :=
  Real.sqrt 23 + 105 / x

theorem find_B_squared :
  ∃ B : ℝ, (B = (Real.sqrt 443)) ∧ (B^2 = 443) :=
by
  sorry

end find_B_squared_l163_163372


namespace tennis_racket_weight_l163_163564

theorem tennis_racket_weight 
  (r b : ℝ)
  (h1 : 10 * r = 8 * b)
  (h2 : 4 * b = 120) :
  r = 24 :=
by
  sorry

end tennis_racket_weight_l163_163564


namespace sum_of_even_sequence_is_194_l163_163866

theorem sum_of_even_sequence_is_194
  (a b c d : ℕ) 
  (even_a : a % 2 = 0) 
  (even_b : b % 2 = 0) 
  (even_c : c % 2 = 0) 
  (even_d : d % 2 = 0)
  (a_lt_b : a < b) 
  (b_lt_c : b < c) 
  (c_lt_d : c < d)
  (diff_da : d - a = 90)
  (arith_ab_c : 2 * b = a + c)
  (geo_bc_d : c^2 = b * d)
  : a + b + c + d = 194 := 
sorry

end sum_of_even_sequence_is_194_l163_163866


namespace sum_of_positive_integers_n_l163_163656

theorem sum_of_positive_integers_n
  (n : ℕ) (h1: n > 0)
  (h2 : Nat.lcm n 100 = Nat.gcd n 100 + 300) :
  n = 350 :=
sorry

end sum_of_positive_integers_n_l163_163656


namespace find_m_range_l163_163743

-- Definitions
def p (x : ℝ) : Prop := abs (2 * x + 1) ≤ 3
def q (x m : ℝ) (h : m > 0) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0

-- Problem Statement
theorem find_m_range : 
  (∀ (x : ℝ) (h : m > 0), (¬ (p x)) → (¬ (q x m h))) ∧ 
  (∃ (x : ℝ), ¬ (p x) ∧ ¬ (q x m h)) → 
  ∃ (m : ℝ), m ≥ 3 := 
sorry

end find_m_range_l163_163743


namespace find_two_digit_number_l163_163494

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def product_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

theorem find_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ is_odd N ∧ is_multiple_of_9 N ∧ is_perfect_square (product_of_digits N) ∧ N = 99 :=
sorry

end find_two_digit_number_l163_163494


namespace bacteria_original_count_l163_163770

theorem bacteria_original_count (current: ℕ) (increase: ℕ) (hc: current = 8917) (hi: increase = 8317) : current - increase = 600 :=
by
  sorry

end bacteria_original_count_l163_163770


namespace find_a_range_empty_solution_set_l163_163117

theorem find_a_range_empty_solution_set :
  ∀ a : ℝ, (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0 → false) ↔ (-2 ≤ a ∧ a < 6 / 5) :=
by sorry

end find_a_range_empty_solution_set_l163_163117


namespace problem_solution_l163_163346

-- Define the operation otimes
def otimes (x y : ℚ) : ℚ := (x * y) / (x + y / 3)

-- Define the specific values x and y
def x : ℚ := 4
def y : ℚ := 3/2 -- 1.5 in fraction form

-- Prove the mathematical statement
theorem problem_solution : (0.36 : ℚ) * (otimes x y) = 12 / 25 := by
  sorry

end problem_solution_l163_163346


namespace equation_of_circle_correct_l163_163163

open Real

def equation_of_circle_through_points (x y : ℝ) :=
  x^2 + y^2 - 4 * x - 6 * y

theorem equation_of_circle_correct :
  ∀ (x y : ℝ),
    (equation_of_circle_through_points (0 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (4 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (-1 : ℝ) (1 : ℝ) = 0) →
    (equation_of_circle_through_points x y = 0) :=
by 
  sorry

end equation_of_circle_correct_l163_163163


namespace MrsHilt_money_left_l163_163421

theorem MrsHilt_money_left (initial_amount pencil_cost remaining_amount : ℕ) 
  (h_initial : initial_amount = 15) 
  (h_cost : pencil_cost = 11) 
  (h_remaining : remaining_amount = 4) : 
  initial_amount - pencil_cost = remaining_amount := 
by 
  sorry

end MrsHilt_money_left_l163_163421


namespace janice_time_left_l163_163995

def time_before_movie : ℕ := 2 * 60
def homework_time : ℕ := 30
def cleaning_time : ℕ := homework_time / 2
def walking_dog_time : ℕ := homework_time + 5
def taking_trash_time : ℕ := homework_time * 1 / 6

theorem janice_time_left : time_before_movie - (homework_time + cleaning_time + walking_dog_time + taking_trash_time) = 35 :=
by
  sorry

end janice_time_left_l163_163995


namespace minimum_a_l163_163756

noncomputable def f (x : ℝ) := x - Real.exp (x - Real.exp 1)

theorem minimum_a (a : ℝ) (x1 x2 : ℝ) (hx : x2 - x1 ≥ Real.exp 1)
  (hy : Real.exp x1 = 1 + Real.log (x2 - a)) : a ≥ Real.exp 1 - 1 :=
by
  sorry

end minimum_a_l163_163756


namespace martin_total_waste_is_10_l163_163430

def martinWastesTrafficTime : Nat := 2
def martinWastesFreewayTime : Nat := 4 * martinWastesTrafficTime
def totalTimeWasted : Nat := martinWastesTrafficTime + martinWastesFreewayTime

theorem martin_total_waste_is_10 : totalTimeWasted = 10 := 
by 
  sorry

end martin_total_waste_is_10_l163_163430


namespace james_problem_l163_163672

def probability_at_least_two_green_apples (total: ℕ) (red: ℕ) (green: ℕ) (yellow: ℕ) (choices: ℕ) : ℚ :=
  let favorable_outcomes := (Nat.choose green 2) * (Nat.choose (total - green) 1) + (Nat.choose green 3)
  let total_outcomes := Nat.choose total choices
  favorable_outcomes / total_outcomes

theorem james_problem : probability_at_least_two_green_apples 10 5 3 2 3 = 11 / 60 :=
by sorry

end james_problem_l163_163672


namespace average_of_b_and_c_l163_163701

theorem average_of_b_and_c (a b c : ℝ) 
  (h₁ : (a + b) / 2 = 50) 
  (h₂ : c - a = 40) : 
  (b + c) / 2 = 70 := 
by
  sorry

end average_of_b_and_c_l163_163701


namespace kennedy_distance_to_school_l163_163713

def miles_per_gallon : ℕ := 19
def initial_gallons : ℕ := 2
def distance_softball_park : ℕ := 6
def distance_burger_restaurant : ℕ := 2
def distance_friends_house : ℕ := 4
def distance_home : ℕ := 11

def total_distance_possible : ℕ := miles_per_gallon * initial_gallons
def distance_after_school : ℕ := distance_softball_park + distance_burger_restaurant + distance_friends_house + distance_home
def distance_to_school : ℕ := total_distance_possible - distance_after_school

theorem kennedy_distance_to_school :
  distance_to_school = 15 :=
by
  sorry

end kennedy_distance_to_school_l163_163713


namespace greatest_servings_l163_163510

def servings (ingredient_amount recipe_amount: ℚ) (recipe_servings: ℕ) : ℚ :=
  (ingredient_amount / recipe_amount) * recipe_servings

theorem greatest_servings (chocolate_new_recipe sugar_new_recipe water_new_recipe milk_new_recipe : ℚ)
                         (servings_new_recipe : ℕ)
                         (chocolate_jordan sugar_jordan milk_jordan : ℚ)
                         (lots_of_water : Prop) :
  chocolate_new_recipe = 3 ∧ sugar_new_recipe = 1/3 ∧ water_new_recipe = 1.5 ∧ milk_new_recipe = 5 ∧
  servings_new_recipe = 6 ∧ chocolate_jordan = 8 ∧ sugar_jordan = 3 ∧ milk_jordan = 12 ∧ lots_of_water →
  max (servings chocolate_jordan chocolate_new_recipe servings_new_recipe)
      (max (servings sugar_jordan sugar_new_recipe servings_new_recipe)
           (servings milk_jordan milk_new_recipe servings_new_recipe)) = 16 :=
by
  sorry

end greatest_servings_l163_163510


namespace frank_money_l163_163294

theorem frank_money (X : ℝ) (h1 : (3/4) * (4/5) * X = 360) : X = 600 :=
sorry

end frank_money_l163_163294


namespace postit_notes_area_l163_163549

theorem postit_notes_area (length width adhesive_len : ℝ) (num_notes : ℕ)
  (h_length : length = 9.4) (h_width : width = 3.7) (h_adh_len : adhesive_len = 0.6) (h_num_notes : num_notes = 15) :
  (length + (length - adhesive_len) * (num_notes - 1)) * width = 490.62 :=
by
  rw [h_length, h_width, h_adh_len, h_num_notes]
  sorry

end postit_notes_area_l163_163549


namespace general_term_of_sequence_l163_163956

theorem general_term_of_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h1 : ∀ n, S n = 2 * a n - 1) 
    (a₁ : a 1 = 1) :
  ∀ n, a n = 2^(n - 1) := 
sorry

end general_term_of_sequence_l163_163956


namespace ones_digit_of_11_pow_46_l163_163316

theorem ones_digit_of_11_pow_46 : (11 ^ 46) % 10 = 1 :=
by sorry

end ones_digit_of_11_pow_46_l163_163316


namespace remaining_trees_correct_l163_163600

def initial_oak_trees := 57
def initial_maple_trees := 43

def full_cut_oak := 13
def full_cut_maple := 8

def partial_cut_oak := 2.5
def partial_cut_maple := 1.5

def remaining_oak_trees := initial_oak_trees - full_cut_oak
def remaining_maple_trees := initial_maple_trees - full_cut_maple

def total_remaining_trees := remaining_oak_trees + remaining_maple_trees

theorem remaining_trees_correct : remaining_oak_trees = 44 ∧ remaining_maple_trees = 35 ∧ total_remaining_trees = 79 :=
by
  sorry

end remaining_trees_correct_l163_163600


namespace weaving_increase_is_sixteen_over_twentynine_l163_163953

-- Conditions for the problem as definitions
def first_day_weaving := 5
def total_days := 30
def total_weaving := 390

-- The arithmetic series sum formula for 30 days
def sum_arithmetic_series (a d : ℚ) (n : ℕ) := n * a + (n * (n-1) / 2) * d

-- The question is to prove the increase in chi per day is 16/29
theorem weaving_increase_is_sixteen_over_twentynine
  (d : ℚ)
  (h : sum_arithmetic_series first_day_weaving d total_days = total_weaving) :
  d = 16 / 29 :=
sorry

end weaving_increase_is_sixteen_over_twentynine_l163_163953


namespace find_points_on_number_line_l163_163218

noncomputable def numbers_are_opposite (x y : ℝ) : Prop :=
  x = -y

theorem find_points_on_number_line (A B : ℝ) 
  (h1 : numbers_are_opposite A B) 
  (h2 : |A - B| = 8) 
  (h3 : A < B) : 
  (A = -4 ∧ B = 4) :=
by
  sorry

end find_points_on_number_line_l163_163218


namespace number_of_routes_jack_to_jill_l163_163667

def num_routes_avoiding (start goal avoid : ℕ × ℕ) : ℕ := sorry

theorem number_of_routes_jack_to_jill : 
  num_routes_avoiding (0,0) (3,2) (1,1) = 4 :=
sorry

end number_of_routes_jack_to_jill_l163_163667


namespace maximum_pairwise_sum_is_maximal_l163_163614

noncomputable def maximum_pairwise_sum (set_sums : List ℝ) (x y z w : ℝ) : Prop :=
  ∃ (a b c d e : ℝ), set_sums = [400, 500, 600, 700, 800, 900, x, y, z, w] ∧  
  ((2 / 5) * (400 + 500 + 600 + 700 + 800 + 900 + x + y + z + w)) = 
    (a + b + c + d + e) ∧ 
  5 * (a + b + c + d + e) - (400 + 500 + 600 + 700 + 800 + 900) = 1966.67

theorem maximum_pairwise_sum_is_maximal :
  maximum_pairwise_sum [400, 500, 600, 700, 800, 900] 1966.67 (1966.67 / 4) 
(1966.67 / 3) (1966.67 / 2) :=
sorry

end maximum_pairwise_sum_is_maximal_l163_163614


namespace intersection_eq_l163_163359

def M : Set (ℝ × ℝ) := { p | ∃ x, p.2 = x^2 }
def N : Set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = 2 }
def Intersect : Set (ℝ × ℝ) := { p | (M p) ∧ (N p)}

theorem intersection_eq : Intersect = { p : ℝ × ℝ | p = (1,1) ∨ p = (-1, 1) } :=
  sorry

end intersection_eq_l163_163359


namespace perfect_square_trinomial_l163_163806

theorem perfect_square_trinomial (m : ℤ) (h : ∃ b : ℤ, (x : ℤ) → x^2 - 10 * x + m = (x + b)^2) : m = 25 :=
sorry

end perfect_square_trinomial_l163_163806


namespace doctor_visit_cost_l163_163979

theorem doctor_visit_cost (cast_cost : ℝ) (insurance_coverage : ℝ) (out_of_pocket : ℝ) (visit_cost : ℝ) :
  cast_cost = 200 → insurance_coverage = 0.60 → out_of_pocket = 200 → 0.40 * (visit_cost + cast_cost) = out_of_pocket → visit_cost = 300 :=
by
  intros h_cast h_insurance h_out_of_pocket h_equation
  sorry

end doctor_visit_cost_l163_163979


namespace geometric_sum_common_ratios_l163_163627

theorem geometric_sum_common_ratios (k p r : ℝ) 
  (hp : p ≠ r) (h_seq : p ≠ 1 ∧ r ≠ 1 ∧ p ≠ 0 ∧ r ≠ 0) 
  (h : k * p^4 - k * r^4 = 4 * (k * p^2 - k * r^2)) : 
  p + r = 3 :=
by
  -- Details omitted as requested
  sorry

end geometric_sum_common_ratios_l163_163627


namespace remainder_when_3m_div_by_5_l163_163067

variable (m k : ℤ)

theorem remainder_when_3m_div_by_5 (h : m % 5 = 2) : (3 * m) % 5 = 1 := by
  sorry

end remainder_when_3m_div_by_5_l163_163067


namespace motorcycles_meet_after_54_minutes_l163_163889

noncomputable def motorcycles_meet_time : ℕ := sorry

theorem motorcycles_meet_after_54_minutes :
  motorcycles_meet_time = 54 := sorry

end motorcycles_meet_after_54_minutes_l163_163889


namespace solution_set_l163_163090

def f (x : ℝ) : ℝ := sorry

axiom ax1 : ∀ a b : ℝ, f (a + b) = f a + f b - 1
axiom ax2 : ∀ x : ℝ, x > 0 → f x > 1
axiom ax3 : f 4 = 5

theorem solution_set (x : ℝ) : f (3 * x^2 - x - 2) < 3 ↔ (-1 < x ∧ x < 4 / 3) :=
by
  sorry

end solution_set_l163_163090


namespace next_hexagon_dots_l163_163717

theorem next_hexagon_dots (base_dots : ℕ) (increment : ℕ) : base_dots = 2 → increment = 2 → 
  (2 + 6*2) + 6*(2*2) + 6*(3*2) + 6*(4*2) = 122 := 
by
  intros hbd hi
  sorry

end next_hexagon_dots_l163_163717


namespace part1_part2_l163_163534

variables (α β : Real)

theorem part1 (h1 : Real.cos (α + β) = 1 / 3) (h2 : Real.sin α * Real.sin β = 1 / 4) :
  Real.cos α * Real.cos β = 7 / 12 := 
sorry

theorem part2 (h1 : Real.cos (α + β) = 1 / 3) (h2 : Real.sin α * Real.sin β = 1 / 4) :
  Real.cos (2 * α - 2 * β) = 7 / 18 := 
sorry

end part1_part2_l163_163534


namespace train_crossing_tree_time_l163_163846

noncomputable def time_to_cross_platform (train_length : ℕ) (platform_length : ℕ) (time_to_cross_platform : ℕ) : ℕ :=
  (train_length + platform_length) / time_to_cross_platform

noncomputable def time_to_cross_tree (train_length : ℕ) (speed : ℕ) : ℕ :=
  train_length / speed

theorem train_crossing_tree_time :
  ∀ (train_length platform_length time platform_time speed : ℕ),
  train_length = 1200 →
  platform_length = 900 →
  platform_time = 210 →
  speed = (train_length + platform_length) / platform_time →
  time = train_length / speed →
  time = 120 :=
by
  intros train_length platform_length time platform_time speed h_train_length h_platform_length h_platform_time h_speed h_time
  sorry

end train_crossing_tree_time_l163_163846


namespace true_propositions_l163_163858

theorem true_propositions : 
  (∀ x : ℝ, x^3 < 1 → x^2 + 1 > 0) ∧ (∀ x : ℚ, x^2 = 2 → false) ∧ 
  (∀ x : ℕ, x^3 > x^2 → false) ∧ (∀ x : ℝ, x^2 + 1 > 0) :=
by 
  -- proof goes here
  sorry

end true_propositions_l163_163858


namespace max_elephants_l163_163299

def union_members : ℕ := 28
def non_union_members : ℕ := 37

/-- Given 28 union members and 37 non-union members, where elephants are distributed equally among
each group and each person initially receives at least one elephant, and considering 
the unique distribution constraint, the maximum number of elephants is 2072. -/
theorem max_elephants (n : ℕ) 
  (h1 : n % union_members = 0)
  (h2 : n % non_union_members = 0)
  (h3 : n ≥ union_members * non_union_members) :
  n = 2072 :=
by sorry

end max_elephants_l163_163299


namespace seat_number_X_l163_163962

theorem seat_number_X (X : ℕ) (h1 : 42 - 30 = X - 6) : X = 18 :=
by
  sorry

end seat_number_X_l163_163962


namespace percentage_of_profits_l163_163057

variable (R P : ℝ) -- Let R be the revenues and P be the profits in the previous year
variable (H1 : (P/R) * 100 = 10) -- The condition we want to prove
variable (H2 : 0.95 * R) -- Revenues in 2009 are 0.95R
variable (H3 : 0.1 * 0.95 * R) -- Profits in 2009 are 0.1 * 0.95R = 0.095R
variable (H4 : 0.095 * R = 0.95 * P) -- The given relation between profits in 2009 and previous year

theorem percentage_of_profits (H1 : (P/R) * 100 = 10) 
  (H2 : ∀ (R : ℝ),  ∃ ρ, ρ = 0.95 * R)
  (H3 : ∀ (R : ℝ),  ∃ π, π = 0.10 * (0.95 * R))
  (H4 : ∀ (R P : ℝ), 0.095 * R = 0.95 * P) :
  ∀ (P R : ℝ), (P/R) * 100 = 10 := 
by
  sorry

end percentage_of_profits_l163_163057


namespace find_principal_amount_l163_163343

theorem find_principal_amount (A R T : ℝ) (P : ℝ) : 
  A = 1680 → R = 0.05 → T = 2.4 → 1.12 * P = 1680 → P = 1500 :=
by
  intros hA hR hT hEq
  sorry

end find_principal_amount_l163_163343


namespace chord_ratio_l163_163541

theorem chord_ratio (EQ GQ HQ FQ : ℝ) (h1 : EQ = 5) (h2 : GQ = 12) (h3 : HQ = 3) (h4 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 12 / 5 := by
  sorry

end chord_ratio_l163_163541


namespace average_score_is_correct_l163_163517

-- Define the given conditions
def numbers_of_students : List ℕ := [12, 28, 40, 35, 20, 10, 5]
def scores : List ℕ := [95, 85, 75, 65, 55, 45, 35]

-- Function to calculate the total score
def total_score (scores numbers : List ℕ) : ℕ :=
  List.sum (List.zipWith (λ a b => a * b) scores numbers)

-- Calculate the average percent score
def average_percent_score (total number_of_students : ℕ) : ℕ :=
  total / number_of_students

-- Prove that the average percentage score is 70
theorem average_score_is_correct :
  average_percent_score (total_score scores numbers_of_students) 150 = 70 :=
by
  sorry

end average_score_is_correct_l163_163517


namespace correct_formula_for_xy_l163_163747

theorem correct_formula_for_xy :
  (∀ x y, (x = 1 ∧ y = 3) ∨ (x = 2 ∧ y = 7) ∨ (x = 3 ∧ y = 13) ∨ (x = 4 ∧ y = 21) ∨ (x = 5 ∧ y = 31) →
    y = x^2 + x + 1) :=
sorry

end correct_formula_for_xy_l163_163747


namespace polynomial_expansion_abs_sum_l163_163220

theorem polynomial_expansion_abs_sum :
  let a_0 := 1
  let a_1 := -8
  let a_2 := 24
  let a_3 := -32
  let a_4 := 16
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| = 81 :=
by
  sorry

end polynomial_expansion_abs_sum_l163_163220


namespace highest_car_color_is_blue_l163_163704

def total_cars : ℕ := 24
def red_cars : ℕ := total_cars / 4
def blue_cars : ℕ := red_cars + 6
def yellow_cars : ℕ := total_cars - (red_cars + blue_cars)

theorem highest_car_color_is_blue :
  blue_cars > red_cars ∧ blue_cars > yellow_cars :=
by sorry

end highest_car_color_is_blue_l163_163704


namespace shirley_cases_l163_163196

-- Given conditions
def T : ℕ := 54  -- boxes of Trefoils sold
def S : ℕ := 36  -- boxes of Samoas sold
def M : ℕ := 48  -- boxes of Thin Mints sold
def t_per_case : ℕ := 4  -- boxes of Trefoils per case
def s_per_case : ℕ := 3  -- boxes of Samoas per case
def m_per_case : ℕ := 5  -- boxes of Thin Mints per case

-- Amount of boxes delivered per case should meet the required demand
theorem shirley_cases : ∃ (n_cases : ℕ), 
  n_cases * t_per_case ≥ T ∧ 
  n_cases * s_per_case ≥ S ∧ 
  n_cases * m_per_case ≥ M :=
by
  use 14
  sorry

end shirley_cases_l163_163196


namespace monotonic_f_inequality_f_over_h_l163_163214

noncomputable def f (x : ℝ) : ℝ := 1 + (1 / x) + Real.log x + (Real.log x / x)

theorem monotonic_f :
  ∀ x : ℝ, x > 0 → ∃ I : Set ℝ, (I = Set.Ioo 0 x ∨ I = Set.Icc 0 x) ∧ (∀ y ∈ I, y > 0 → f y = f x) :=
by
  sorry

theorem inequality_f_over_h :
  ∀ x : ℝ, x > 1 → (f x) / (Real.exp 1 + 1) > (2 * Real.exp (x - 1)) / (x * Real.exp x + 1) :=
by
  sorry

end monotonic_f_inequality_f_over_h_l163_163214


namespace intersection_points_vary_with_a_l163_163086

-- Define the lines
def line1 (x : ℝ) : ℝ := x + 1
def line2 (a x : ℝ) : ℝ := a * x + 1

-- Prove that the number of intersection points varies with a
theorem intersection_points_vary_with_a (a : ℝ) : 
  (∃ x : ℝ, line1 x = line2 a x) ↔ 
    (if a = 1 then true else true) :=
by 
  sorry

end intersection_points_vary_with_a_l163_163086


namespace discarded_number_l163_163160

theorem discarded_number (S S_48 : ℝ) (h1 : S = 1000) (h2 : S_48 = 900) (h3 : ∃ x : ℝ, S - S_48 = 45 + x): 
  ∃ x : ℝ, x = 55 :=
by {
  -- Using the conditions provided to derive the theorem.
  sorry 
}

end discarded_number_l163_163160


namespace solve_system_correct_l163_163037

noncomputable def solve_system (a b c d e : ℝ) : Prop :=
  3 * a = (b + c + d) ^ 3 ∧ 
  3 * b = (c + d + e) ^ 3 ∧ 
  3 * c = (d + e + a) ^ 3 ∧ 
  3 * d = (e + a + b) ^ 3 ∧ 
  3 * e = (a + b + c) ^ 3

theorem solve_system_correct :
  ∀ (a b c d e : ℝ), solve_system a b c d e → 
    (a = 1/3 ∧ b = 1/3 ∧ c = 1/3 ∧ d = 1/3 ∧ e = 1/3) ∨ 
    (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 ∧ e = 0) ∨ 
    (a = -1/3 ∧ b = -1/3 ∧ c = -1/3 ∧ d = -1/3 ∧ e = -1/3) :=
by
  sorry

end solve_system_correct_l163_163037


namespace gcd_324_243_135_l163_163671

theorem gcd_324_243_135 : Nat.gcd (Nat.gcd 324 243) 135 = 27 := by
  sorry

end gcd_324_243_135_l163_163671


namespace smallest_possible_time_for_travel_l163_163271

theorem smallest_possible_time_for_travel :
  ∃ t : ℝ, (∀ D M P : ℝ, D = 6 → M = 6 → P = 6 → 
    ∀ motorcycle_speed distance : ℝ, motorcycle_speed = 90 → distance = 135 → 
    t < 3.9) :=
  sorry

end smallest_possible_time_for_travel_l163_163271


namespace exponentiation_identity_l163_163990

theorem exponentiation_identity (x : ℝ) : (-x^7)^4 = x^28 := 
sorry

end exponentiation_identity_l163_163990


namespace packages_eq_nine_l163_163899

-- Definitions of the given conditions
def x : ℕ := 50
def y : ℕ := 5
def z : ℕ := 5

-- Statement: Prove that the number of packages Amy could make equals 9
theorem packages_eq_nine : (x - y) / z = 9 :=
by
  sorry

end packages_eq_nine_l163_163899


namespace chairs_to_exclude_l163_163561

theorem chairs_to_exclude (chairs : ℕ) (h : chairs = 765) : 
  ∃ n, n^2 ≤ chairs ∧ chairs - n^2 = 36 := 
by 
  sorry

end chairs_to_exclude_l163_163561


namespace smallest_integer_among_three_l163_163122

theorem smallest_integer_among_three 
  (x y z : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (hxy : y - x ≤ 6)
  (hxz : z - x ≤ 6) 
  (hprod : x * y * z = 2808) : 
  x = 12 := 
sorry

end smallest_integer_among_three_l163_163122


namespace boundary_length_of_pattern_l163_163648

theorem boundary_length_of_pattern (area : ℝ) (num_points : ℕ) 
(points_per_side : ℕ) : 
area = 144 → num_points = 4 → points_per_side = 4 →
∃ length : ℝ, length = 92.5 :=
by
  intros
  sorry

end boundary_length_of_pattern_l163_163648


namespace eval_arith_expression_l163_163158

theorem eval_arith_expression : 2 + 3^2 * 4 - 5 + 6 / 2 = 36 := 
by sorry

end eval_arith_expression_l163_163158


namespace top_square_is_9_l163_163523

def initial_grid : List (List ℕ) := 
  [[1, 2, 3],
   [4, 5, 6],
   [7, 8, 9]]

def fold_step_1 (grid : List (List ℕ)) : List (List ℕ) :=
  let col1 := grid.map (fun row => row.get! 0)
  let col3 := grid.map (fun row => row.get! 2)
  let col2 := grid.map (fun row => row.get! 1)
  [[col1.get! 0, col3.get! 0, col2.get! 0],
   [col1.get! 1, col3.get! 1, col2.get! 1],
   [col1.get! 2, col3.get! 2, col2.get! 2]]

def fold_step_2 (grid : List (List ℕ)) : List (List ℕ) :=
  let col1 := grid.map (fun row => row.get! 0)
  let col2 := grid.map (fun row => row.get! 1)
  let col3 := grid.map (fun row => row.get! 2)
  [[col2.get! 0, col1.get! 0, col3.get! 0],
   [col2.get! 1, col1.get! 1, col3.get! 1],
   [col2.get! 2, col1.get! 2, col3.get! 2]]

def fold_step_3 (grid : List (List ℕ)) : List (List ℕ) :=
  let row1 := grid.get! 0
  let row2 := grid.get! 1
  let row3 := grid.get! 2
  [row3, row2, row1]

def folded_grid : List (List ℕ) :=
  fold_step_3 (fold_step_2 (fold_step_1 initial_grid))

theorem top_square_is_9 : folded_grid.get! 0 = [9, 7, 8] :=
  sorry

end top_square_is_9_l163_163523


namespace part_a_part_b_part_c_l163_163516

-- Part (a)
theorem part_a : ∃ a b, a * b = 80 ∧ (a = 8 ∨ a = 4) ∧ (b = 10 ∨ b = 5) :=
by sorry

-- Part (b)
theorem part_b : ∃ a b c, (a * b) / c = 50 ∧ (a = 10 ∨ a = 5) ∧ (b = 10 ∨ b = 5) ∧ (c = 2 ∨ c = 1) :=
by sorry

-- Part (c)
theorem part_c : ∃ n, n = 4 ∧ ∀ a b c, (a + b) / c = 23 :=
by sorry

end part_a_part_b_part_c_l163_163516


namespace largest_divisor_n4_n2_l163_163883

theorem largest_divisor_n4_n2 (n : ℤ) : (6 : ℤ) ∣ (n^4 - n^2) :=
sorry

end largest_divisor_n4_n2_l163_163883


namespace find_triples_l163_163076

theorem find_triples (x y z : ℝ) :
  (x + 1)^2 = x + y + 2 ∧
  (y + 1)^2 = y + z + 2 ∧
  (z + 1)^2 = z + x + 2 ↔ (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 ∧ y = -1 ∧ z = -1) :=
by
  sorry

end find_triples_l163_163076


namespace ratio_a_to_c_l163_163437

variable (a b c : ℕ)

theorem ratio_a_to_c (h1 : a / b = 8 / 3) (h2 : b / c = 1 / 5) : a / c = 8 / 15 := 
sorry

end ratio_a_to_c_l163_163437


namespace B_lap_time_l163_163484

-- Definitions based on given conditions.
def time_to_complete_lap_A := 40
def meeting_interval := 15

-- The theorem states that given the conditions, B takes 24 seconds to complete the track.
theorem B_lap_time (l : ℝ) (t : ℝ) (h1 : t = 24)
                    (h2 : l / time_to_complete_lap_A + l / t = l / meeting_interval):
  t = 24 := by sorry

end B_lap_time_l163_163484


namespace circle_radius_l163_163588

theorem circle_radius (r : ℝ) (x y : ℝ) :
  x = π * r^2 ∧ y = 2 * π * r ∧ x + y = 100 * π → r = 10 := 
  by
  sorry

end circle_radius_l163_163588


namespace total_toys_l163_163547

theorem total_toys (K A L : ℕ) (h1 : A = K + 30) (h2 : L = 2 * K) (h3 : K + A = 160) : 
    K + A + L = 290 :=
by
  sorry

end total_toys_l163_163547


namespace cube_surface_area_l163_163033

theorem cube_surface_area (Q : ℝ) (a : ℝ) (H : (3 * a^2 * Real.sqrt 3) / 2 = Q) :
    (6 * (a * Real.sqrt 2) ^ 2) = (8 * Q * Real.sqrt 3) / 3 :=
by
  sorry

end cube_surface_area_l163_163033


namespace sqrt_abc_sum_l163_163565

variable (a b c : ℝ)

theorem sqrt_abc_sum (h1 : b + c = 17) (h2 : c + a = 20) (h3 : a + b = 23) :
  Real.sqrt (a * b * c * (a + b + c)) = 10 * Real.sqrt 273 := by
  sorry

end sqrt_abc_sum_l163_163565


namespace determinant_scaled_matrix_l163_163356

-- Definitions based on the conditions given in the problem.
def determinant2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

variable (a b c d : ℝ)
variable (h : determinant2x2 a b c d = 5)

-- The proof statement to be filled, proving the correct answer.
theorem determinant_scaled_matrix :
  determinant2x2 (2 * a) (2 * b) (2 * c) (2 * d) = 20 :=
by
  sorry

end determinant_scaled_matrix_l163_163356


namespace sector_arc_length_circumference_ratio_l163_163355

theorem sector_arc_length_circumference_ratio
  {r : ℝ}
  (h_radius : ∀ (sector_radius : ℝ), sector_radius = 2/3 * r)
  (h_area : ∀ (sector_area circle_area : ℝ), sector_area / circle_area = 5/27) :
  ∀ (l C : ℝ), l / C = 5 / 18 :=
by
  -- Prove the theorem using the given hypothesis.
  -- Construction of the detailed proof will go here.
  sorry

end sector_arc_length_circumference_ratio_l163_163355


namespace find_fraction_of_difference_eq_halves_l163_163121

theorem find_fraction_of_difference_eq_halves (x : ℚ) (h : 9 - x = 2.25) : x = 27 / 4 :=
by sorry

end find_fraction_of_difference_eq_halves_l163_163121


namespace parallel_lines_l163_163891

theorem parallel_lines (a : ℝ) :
  ((3 * a + 2) * x + a * y + 6 = 0) ↔
  (a * x - y + 3 = 0) →
  a = -1 :=
by sorry

end parallel_lines_l163_163891


namespace pencil_sharpening_and_breaking_l163_163127

/-- Isha's pencil initially has a length of 31 inches. After sharpening, it has a length of 14 inches.
Prove that:
1. The pencil was shortened by 17 inches.
2. Each half of the pencil, after being broken in half, is 7 inches long. -/
theorem pencil_sharpening_and_breaking 
  (initial_length : ℕ) 
  (length_after_sharpening : ℕ) 
  (sharpened_length : ℕ) 
  (half_length : ℕ) 
  (h1 : initial_length = 31) 
  (h2 : length_after_sharpening = 14) 
  (h3 : sharpened_length = initial_length - length_after_sharpening) 
  (h4 : half_length = length_after_sharpening / 2) : 
  sharpened_length = 17 ∧ half_length = 7 := 
by {
  sorry
}

end pencil_sharpening_and_breaking_l163_163127


namespace calculate_bmw_sales_and_revenue_l163_163123

variable (total_cars : ℕ) (percentage_ford percentage_toyota percentage_nissan percentage_audi : ℕ) (avg_price_bmw : ℕ)
variable (h_total_cars : total_cars = 300) (h_percentage_ford : percentage_ford = 10)
variable (h_percentage_toyota : percentage_toyota = 25) (h_percentage_nissan : percentage_nissan = 20)
variable (h_percentage_audi : percentage_audi = 15) (h_avg_price_bmw : avg_price_bmw = 35000)

theorem calculate_bmw_sales_and_revenue :
  let percentage_non_bmw := percentage_ford + percentage_toyota + percentage_nissan + percentage_audi
  let percentage_bmw := 100 - percentage_non_bmw
  let number_bmw := total_cars * percentage_bmw / 100
  let total_revenue := number_bmw * avg_price_bmw
  (number_bmw = 90) ∧ (total_revenue = 3150000) := by
  -- Definitions are taken from conditions and used directly in the theorem statement
  sorry

end calculate_bmw_sales_and_revenue_l163_163123


namespace average_annual_population_increase_l163_163795

theorem average_annual_population_increase 
    (initial_population : ℝ) 
    (final_population : ℝ) 
    (years : ℝ) 
    (initial_population_pos : initial_population > 0) 
    (years_pos : years > 0)
    (initial_population_eq : initial_population = 175000) 
    (final_population_eq : final_population = 297500) 
    (years_eq : years = 10) : 
    (final_population - initial_population) / initial_population / years * 100 = 7 :=
by
    sorry

end average_annual_population_increase_l163_163795


namespace max_possible_scores_l163_163928

theorem max_possible_scores (num_questions : ℕ) (points_correct : ℤ) (points_incorrect : ℤ) (points_unanswered : ℤ) :
  num_questions = 10 →
  points_correct = 4 →
  points_incorrect = -1 →
  points_unanswered = 0 →
  ∃ n, n = 45 :=
by
  sorry

end max_possible_scores_l163_163928


namespace union_of_subsets_l163_163016

open Set

variable (A B : Set ℕ)

theorem union_of_subsets (m : ℕ) (hA : A = {1, 3}) (hB : B = {1, 2, m}) (hSubset : A ⊆ B) :
    A ∪ B = {1, 2, 3} :=
  sorry

end union_of_subsets_l163_163016


namespace tan_of_angle_l163_163487

noncomputable def tan_val (α : ℝ) : ℝ := Real.tan α

theorem tan_of_angle (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) (h2 : Real.cos (2 * α) = -3 / 5) :
  tan_val α = -2 := by
  sorry

end tan_of_angle_l163_163487


namespace more_red_flowers_than_white_l163_163292

-- Definitions based on given conditions
def yellow_and_white := 13
def red_and_yellow := 17
def red_and_white := 14
def blue_and_yellow := 16

-- Definitions based on the requirements of the problem
def red_flowers := red_and_yellow + red_and_white
def white_flowers := yellow_and_white + red_and_white

-- Theorem to prove the number of more flowers containing red than white
theorem more_red_flowers_than_white : red_flowers - white_flowers = 4 := by
  sorry

end more_red_flowers_than_white_l163_163292


namespace harkamal_total_amount_l163_163519

-- Conditions
def cost_grapes : ℝ := 8 * 80
def cost_mangoes : ℝ := 9 * 55
def cost_apples_before_discount : ℝ := 6 * 120
def cost_oranges : ℝ := 4 * 75
def discount_apples : ℝ := 0.10 * cost_apples_before_discount
def cost_apples_after_discount : ℝ := cost_apples_before_discount - discount_apples

def total_cost_before_tax : ℝ :=
  cost_grapes + cost_mangoes + cost_apples_after_discount + cost_oranges

def sales_tax : ℝ := 0.05 * total_cost_before_tax

def total_amount_paid : ℝ := total_cost_before_tax + sales_tax

-- Question translated into a Lean statement
theorem harkamal_total_amount:
  total_amount_paid = 2187.15 := 
sorry

end harkamal_total_amount_l163_163519


namespace isosceles_triangle_k_value_l163_163771

theorem isosceles_triangle_k_value 
(side1 : ℝ)
(side2 side3 : ℝ)
(k : ℝ)
(h1 : side1 = 3 ∨ side2 = 3 ∨ side3 = 3)
(h2 : side1 = side2 ∨ side1 = side3 ∨ side2 = side3)
(h3 : Polynomial.eval side1 (Polynomial.C k + Polynomial.X ^ 2) = 0 
    ∨ Polynomial.eval side2 (Polynomial.C k + Polynomial.X ^ 2) = 0 
    ∨ Polynomial.eval side3 (Polynomial.C k + Polynomial.X ^ 2) = 0) :
k = 3 ∨ k = 4 :=
sorry

end isosceles_triangle_k_value_l163_163771


namespace jebb_expense_l163_163665

-- Define the costs
def seafood_platter := 45.0
def rib_eye_steak := 38.0
def vintage_wine_glass := 18.0
def chocolate_dessert := 12.0

-- Define the rules and discounts
def discount_percentage := 0.10
def service_fee_12 := 0.12
def service_fee_15 := 0.15
def tip_percentage := 0.20

-- Total food and wine cost
def total_food_and_wine_cost := 
  seafood_platter + rib_eye_steak + (2 * vintage_wine_glass) + chocolate_dessert

-- Total food cost excluding wine
def food_cost_excluding_wine := 
  seafood_platter + rib_eye_steak + chocolate_dessert

-- 10% discount on food cost excluding wine
def discount_amount := discount_percentage * food_cost_excluding_wine
def reduced_food_cost := food_cost_excluding_wine - discount_amount

-- New total cost before applying the service fee
def total_cost_before_service_fee := reduced_food_cost + (2 * vintage_wine_glass)

-- Determine the service fee based on cost
def service_fee := 
  if total_cost_before_service_fee > 80.0 then 
    service_fee_15 * total_cost_before_service_fee 
  else if total_cost_before_service_fee >= 50.0 then 
    service_fee_12 * total_cost_before_service_fee 
  else 
    0.0

-- Total cost after discount and service fee
def total_cost_after_service_fee := total_cost_before_service_fee + service_fee

-- Tip amount (20% of total cost after discount and service fee)
def tip_amount := tip_percentage * total_cost_after_service_fee

-- Total amount Jebb spent
def total_amount_spent := total_cost_after_service_fee + tip_amount

-- Lean theorem statement
theorem jebb_expense :
  total_amount_spent = 167.67 :=
by
  -- prove the theorem here
  sorry

end jebb_expense_l163_163665


namespace new_roots_quadratic_l163_163363

variable {p q : ℝ}

theorem new_roots_quadratic :
  (∀ (r₁ r₂ : ℝ), r₁ + r₂ = -p ∧ r₁ * r₂ = q → 
  (x : ℝ) → x^2 + ((p^2 - 2 * q)^2 - 2 * q^2) * x + q^4 = 0) :=
by 
  intros r₁ r₂ h x
  have : r₁ + r₂ = -p := h.1
  have : r₁ * r₂ = q := h.2
  sorry

end new_roots_quadratic_l163_163363


namespace sum_f_values_l163_163194

noncomputable def f : ℝ → ℝ := sorry

axiom odd_property (x : ℝ) : f (-x) = -f (x)
axiom periodicity (x : ℝ) : f (x) = f (x + 4)
axiom f1 : f 1 = -1

theorem sum_f_values : f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry

end sum_f_values_l163_163194


namespace investment_calculation_l163_163263

theorem investment_calculation
    (R Trishul Vishal Alok Harshit : ℝ)
    (hTrishul : Trishul = 0.9 * R)
    (hVishal : Vishal = 0.99 * R)
    (hAlok : Alok = 1.035 * Trishul)
    (hHarshit : Harshit = 0.95 * Vishal)
    (hTotal : R + Trishul + Vishal + Alok + Harshit = 22000) :
  R = 22000 / 3.8655 ∧
  Trishul = 0.9 * R ∧
  Vishal = 0.99 * R ∧
  Alok = 1.035 * Trishul ∧
  Harshit = 0.95 * Vishal ∧
  R + Trishul + Vishal + Alok + Harshit = 22000 :=
sorry

end investment_calculation_l163_163263


namespace sticker_price_l163_163642

theorem sticker_price (y : ℝ) (h1 : ∀ (p : ℝ), p = 0.8 * y - 60 → p ≤ y)
  (h2 : ∀ (q : ℝ), q = 0.7 * y → q ≤ y)
  (h3 : (0.8 * y - 60) + 20 = 0.7 * y) :
  y = 400 :=
by
  sorry

end sticker_price_l163_163642


namespace length_of_square_cut_off_l163_163244

theorem length_of_square_cut_off 
  (x : ℝ) 
  (h_eq : (48 - 2 * x) * (36 - 2 * x) * x = 5120) : 
  x = 8 := 
sorry

end length_of_square_cut_off_l163_163244


namespace henry_twice_jill_l163_163543

-- Conditions
def Henry := 29
def Jill := 19
def sum_ages : Nat := Henry + Jill

-- Prove the statement
theorem henry_twice_jill (Y : Nat) (H J : Nat) (h_sum : H + J = 48) (h_H : H = 29) (h_J : J = 19) :
  H - Y = 2 * (J - Y) ↔ Y = 9 :=
by {
  -- Here, we would provide the proof, but we'll skip that with sorry.
  sorry
}

end henry_twice_jill_l163_163543


namespace total_cost_l163_163260

variable (a b : ℝ)

def tomato_cost (a : ℝ) := 30 * a
def cabbage_cost (b : ℝ) := 50 * b

theorem total_cost (a b : ℝ) : 
  tomato_cost a + cabbage_cost b = 30 * a + 50 * b := 
by 
  unfold tomato_cost cabbage_cost
  sorry

end total_cost_l163_163260


namespace minimum_bailing_rate_l163_163272

theorem minimum_bailing_rate (distance_to_shore : ℝ) (row_speed : ℝ) (leak_rate : ℝ) (max_water_intake : ℝ)
  (time_to_shore : ℝ := distance_to_shore / row_speed * 60) (total_water_intake : ℝ := time_to_shore * leak_rate) :
  distance_to_shore = 1.5 → row_speed = 3 → leak_rate = 10 → max_water_intake = 40 →
  ∃ (bail_rate : ℝ), bail_rate ≥ 9 :=
by
  sorry

end minimum_bailing_rate_l163_163272


namespace constant_term_exists_l163_163803

theorem constant_term_exists (n : ℕ) (h : n = 6) : 
  (∃ r : ℕ, 2 * n - 3 * r = 0) ∧ 
  (∃ n' r' : ℕ, n' ≠ 6 ∧ 2 * n' - 3 * r' = 0) := by
  sorry

end constant_term_exists_l163_163803


namespace find_f_13_l163_163022

noncomputable def f : ℝ → ℝ := sorry

axiom f_property : ∀ x, f (x + f x) = 3 * f x
axiom f_of_1 : f 1 = 3

theorem find_f_13 : f 13 = 27 :=
by
  have hf := f_property
  have hf1 := f_of_1
  sorry

end find_f_13_l163_163022


namespace no_rotation_of_11_gears_l163_163137

theorem no_rotation_of_11_gears :
  ∀ (gears : Fin 11 → ℕ → Prop), 
    (∀ i, gears i 0 ∧ gears (i + 1) 1 → gears i 0 = ¬gears (i + 1) 1) →
    gears 10 0 = gears 0 0 →
    False :=
by
  sorry

end no_rotation_of_11_gears_l163_163137


namespace different_quantifiers_not_equiv_l163_163404

theorem different_quantifiers_not_equiv {x₀ : ℝ} :
  (∃ x₀ : ℝ, x₀^2 > 3) ↔ ¬ (∀ x₀ : ℝ, x₀^2 > 3) :=
by
  sorry

end different_quantifiers_not_equiv_l163_163404


namespace part1_part2_l163_163944

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

theorem part1 (a : ℝ) (h : A ∩ B a = {2}) : a = -1 ∨ a = -3 := by
  sorry

theorem part2 (a : ℝ) (h : A ∪ B a = A) : a ≤ -3 := by
  sorry

end part1_part2_l163_163944


namespace apple_eating_contest_l163_163791

theorem apple_eating_contest (a z : ℕ) (h_most : a = 8) (h_fewest : z = 1) : a - z = 7 :=
by
  sorry

end apple_eating_contest_l163_163791


namespace least_product_of_distinct_primes_gt_30_l163_163370

theorem least_product_of_distinct_primes_gt_30 :
  ∃ p q : ℕ, p > 30 ∧ q > 30 ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_gt_30_l163_163370


namespace odd_squarefree_integers_1_to_199_l163_163833

noncomputable def count_squarefree_odd_integers (n : ℕ) :=
  n - List.sum [
    n / 18,   -- for 3^2 = 9
    n / 50,   -- for 5^2 = 25
    n / 98,   -- for 7^2 = 49
    n / 162,  -- for 9^2 = 81
    n / 242,  -- for 11^2 = 121
    n / 338   -- for 13^2 = 169
  ]

theorem odd_squarefree_integers_1_to_199 : count_squarefree_odd_integers 198 = 79 := 
by
  sorry

end odd_squarefree_integers_1_to_199_l163_163833


namespace find_num_female_workers_l163_163455

-- Defining the given constants and equations
def num_male_workers : Nat := 20
def num_child_workers : Nat := 5
def wage_male_worker : Nat := 35
def wage_female_worker : Nat := 20
def wage_child_worker : Nat := 8
def avg_wage_paid : Nat := 26

-- Defining the total number of workers and total daily wage
def total_workers (num_female_workers : Nat) : Nat := 
  num_male_workers + num_female_workers + num_child_workers

def total_wage (num_female_workers : Nat) : Nat :=
  (num_male_workers * wage_male_worker) + (num_female_workers * wage_female_worker) + (num_child_workers * wage_child_worker)

-- Proving the number of female workers given the average wage
theorem find_num_female_workers (F : Nat) 
  (h : avg_wage_paid * total_workers F = total_wage F) : 
  F = 15 :=
by
  sorry

end find_num_female_workers_l163_163455


namespace inequality_proof_l163_163365

theorem inequality_proof
  (x1 x2 x3 x4 x5 : ℝ)
  (hx1 : 0 < x1)
  (hx2 : 0 < x2)
  (hx3 : 0 < x3)
  (hx4 : 0 < x4)
  (hx5 : 0 < x5) :
  x1^2 + x2^2 + x3^2 + x4^2 + x5^2 ≥ x1 * (x2 + x3 + x4 + x5) :=
by
  sorry

end inequality_proof_l163_163365


namespace fraction_of_girls_is_half_l163_163225

variables (T G B : ℝ)
def fraction_x_of_girls (x : ℝ) : Prop :=
  x * G = (1/5) * T ∧ B / G = 1.5 ∧ T = B + G

theorem fraction_of_girls_is_half (x : ℝ) (h : fraction_x_of_girls T G B x) : x = 0.5 :=
sorry

end fraction_of_girls_is_half_l163_163225


namespace sum_of_digits_of_gcd_l163_163336

def gcd_of_differences : ℕ := Int.gcd (Int.gcd 3360 2240) 5600

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem sum_of_digits_of_gcd :
  sum_of_digits gcd_of_differences = 4 :=
by
  sorry

end sum_of_digits_of_gcd_l163_163336


namespace tiles_finite_initial_segment_l163_163808

theorem tiles_finite_initial_segment (S : ℕ → Prop) (hTiling : ∀ n : ℕ, ∃ m : ℕ, m ≥ n ∧ S m) :
  ∃ k : ℕ, ∀ n : ℕ, n ≤ k → S n :=
by
  sorry

end tiles_finite_initial_segment_l163_163808


namespace price_of_whole_pizza_l163_163501

theorem price_of_whole_pizza
    (price_per_slice : ℕ)
    (num_slices_sold : ℕ)
    (num_whole_pizzas_sold : ℕ)
    (total_revenue : ℕ) 
    (H : price_per_slice * num_slices_sold + num_whole_pizzas_sold * P = total_revenue) : 
    P = 15 :=
by
  let price_per_slice := 3
  let num_slices_sold := 24
  let num_whole_pizzas_sold := 3
  let total_revenue := 117
  sorry

end price_of_whole_pizza_l163_163501


namespace motorcycle_time_l163_163754

theorem motorcycle_time (v_m v_b d t_m : ℝ) 
  (h1 : 12 * v_m + 9 * v_b = d)
  (h2 : 21 * v_b + 8 * v_m = d)
  (h3 : v_m = 3 * v_b) :
  t_m = 15 :=
by
  sorry

end motorcycle_time_l163_163754


namespace Jason_reroll_probability_optimal_l163_163435

/-- Represents the action of rerolling dice to achieve a sum of 9 when
    the player optimizes their strategy. The probability 
    that the player chooses to reroll exactly two dice.
 -/
noncomputable def probability_reroll_two_dice : ℚ :=
  13 / 72

/-- Prove that the probability Jason chooses to reroll exactly two
    dice to achieve a sum of 9, given the optimal strategy, is 13/72.
 -/
theorem Jason_reroll_probability_optimal :
  probability_reroll_two_dice = 13 / 72 :=
sorry

end Jason_reroll_probability_optimal_l163_163435


namespace gym_distance_l163_163217

def distance_to_work : ℕ := 10
def distance_to_gym (dist : ℕ) : ℕ := (dist / 2) + 2

theorem gym_distance :
  distance_to_gym distance_to_work = 7 :=
sorry

end gym_distance_l163_163217


namespace combined_avg_of_remaining_two_subjects_l163_163154

noncomputable def avg (scores : List ℝ) : ℝ :=
  scores.foldl (· + ·) 0 / scores.length

theorem combined_avg_of_remaining_two_subjects 
  (S1_avg S2_part_avg all_avg : ℝ)
  (S1_count S2_part_count S2_total_count : ℕ)
  (h1 : S1_avg = 85) 
  (h2 : S2_part_avg = 78) 
  (h3 : all_avg = 80) 
  (h4 : S1_count = 3)
  (h5 : S2_part_count = 5)
  (h6 : S2_total_count = 7) :
  avg [all_avg * (S1_count + S2_total_count) 
       - S1_count * S1_avg 
       - S2_part_count * S2_part_avg] / (S2_total_count - S2_part_count)
  = 77.5 := by
  sorry

end combined_avg_of_remaining_two_subjects_l163_163154


namespace dimension_sum_l163_163147

-- Define the dimensions A, B, C and areas AB, AC, BC
variables (A B C : ℝ) (AB AC BC : ℝ)

-- Conditions
def conditions := AB = 40 ∧ AC = 90 ∧ BC = 100 ∧ A * B = AB ∧ A * C = AC ∧ B * C = BC

-- Theorem statement
theorem dimension_sum : conditions A B C AB AC BC → A + B + C = (83 : ℝ) / 3 :=
by
  intro h
  sorry

end dimension_sum_l163_163147


namespace probability_kyle_catherine_not_david_l163_163991

/--
Kyle, David, and Catherine each try independently to solve a problem. 
Their individual probabilities for success are 1/3, 2/7, and 5/9.
Prove that the probability that Kyle and Catherine, but not David, will solve the problem is 25/189.
-/
theorem probability_kyle_catherine_not_david :
  let P_K := 1 / 3
  let P_D := 2 / 7
  let P_C := 5 / 9
  let P_D_c := 1 - P_D
  P_K * P_C * P_D_c = 25 / 189 :=
by
  sorry

end probability_kyle_catherine_not_david_l163_163991


namespace minimum_value_of_x_plus_2y_l163_163344

-- Definitions for the problem conditions
def isPositive (z : ℝ) : Prop := z > 0

def condition (x y : ℝ) : Prop := 
  isPositive x ∧ isPositive y ∧ (x + 2*y + 2*x*y = 8) 

-- Statement of the problem
theorem minimum_value_of_x_plus_2y (x y : ℝ) (h : condition x y) : x + 2 * y ≥ 4 :=
sorry

end minimum_value_of_x_plus_2y_l163_163344


namespace snowball_total_distance_l163_163457

noncomputable def total_distance (a1 d n : ℕ) : ℕ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem snowball_total_distance :
  total_distance 6 5 25 = 1650 := by
  sorry

end snowball_total_distance_l163_163457


namespace problem_I_problem_II_l163_163976

-- Problem (I)
def A : Set ℝ := { x | x > 2 ∨ x < -1 }
def B : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }
def A_inter_B : Set ℝ := { x | (-3 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 3) }

theorem problem_I : A ∩ B = A_inter_B :=
by
  sorry

-- Problem (II)
def C (m : ℝ) : Set ℝ := { x | 2 * m - 1 < x ∧ x < m + 1 }

theorem problem_II (m : ℝ) : (C m ⊆ B) → m ≥ -1 :=
by
  sorry

end problem_I_problem_II_l163_163976


namespace y_share_per_rupee_l163_163277

theorem y_share_per_rupee (a p : ℝ) (h1 : a * p = 18)
                            (h2 : p + a * p + 0.30 * p = 70) :
    a = 0.45 :=
by 
  sorry

end y_share_per_rupee_l163_163277


namespace train_speed_l163_163091

/-- 
Given:
- Length of train L is 390 meters (0.39 km)
- Speed of man Vm is 2 km/h
- Time to cross man T is 52 seconds

Prove:
- The speed of the train Vt is 25 km/h
--/
theorem train_speed 
  (L : ℝ) (Vm : ℝ) (T : ℝ) (Vt : ℝ)
  (h1 : L = 0.39) 
  (h2 : Vm = 2) 
  (h3 : T = 52 / 3600) 
  (h4 : Vt + Vm = L / T) :
  Vt = 25 :=
by sorry

end train_speed_l163_163091


namespace corvette_trip_average_rate_l163_163468

theorem corvette_trip_average_rate (total_distance : ℕ) (first_half_distance : ℕ)
  (first_half_rate : ℕ) (second_half_time_multiplier : ℕ) (total_time : ℕ) :
  total_distance = 640 →
  first_half_distance = total_distance / 2 →
  first_half_rate = 80 →
  second_half_time_multiplier = 3 →
  total_time = (first_half_distance / first_half_rate) + (second_half_time_multiplier * (first_half_distance / first_half_rate)) →
  (total_distance / total_time) = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end corvette_trip_average_rate_l163_163468


namespace hexagon_diagonals_sum_correct_l163_163099

noncomputable def hexagon_diagonals_sum : ℝ :=
  let AB := 40
  let S := 100
  let AC := 140
  let AD := 240
  let AE := 340
  AC + AD + AE

theorem hexagon_diagonals_sum_correct : hexagon_diagonals_sum = 720 :=
  by
  show hexagon_diagonals_sum = 720
  sorry

end hexagon_diagonals_sum_correct_l163_163099


namespace longest_segment_is_CD_l163_163262

-- Define points A, B, C, D
def A := (-3, 0)
def B := (0, 2)
def C := (3, 0)
def D := (0, -1)

-- Angles in triangle ABD
def angle_ABD := 35
def angle_BAD := 95
def angle_ADB := 50

-- Angles in triangle BCD
def angle_BCD := 55
def angle_BDC := 60
def angle_CBD := 65

-- Length comparison conclusion from triangle ABD
axiom compare_lengths_ABD : ∀ (AD AB BD : ℝ), AD < AB ∧ AB < BD

-- Length comparison conclusion from triangle BCD
axiom compare_lengths_BCD : ∀ (BC BD CD : ℝ), BC < BD ∧ BD < CD

-- Combine results
theorem longest_segment_is_CD : ∀ (AD AB BD BC CD : ℝ), AD < AB → AB < BD → BC < BD → BD < CD → CD ≥ AD ∧ CD ≥ AB ∧ CD ≥ BD ∧ CD ≥ BC :=
by
  intros AD AB BD BC CD h1 h2 h3 h4
  sorry

end longest_segment_is_CD_l163_163262


namespace evaluate_expression_l163_163006

noncomputable def log_4_8 : ℝ := Real.log 8 / Real.log 4
noncomputable def log_8_16 : ℝ := Real.log 16 / Real.log 8

theorem evaluate_expression : Real.sqrt (log_4_8 * log_8_16) = Real.sqrt 2 :=
by
  sorry

end evaluate_expression_l163_163006


namespace monthly_rent_l163_163281

-- Definitions based on the given conditions
def length_ft : ℕ := 360
def width_ft : ℕ := 1210
def sq_feet_per_acre : ℕ := 43560
def cost_per_acre_per_month : ℕ := 60

-- Statement of the problem
theorem monthly_rent : (length_ft * width_ft / sq_feet_per_acre) * cost_per_acre_per_month = 600 := sorry

end monthly_rent_l163_163281


namespace toll_for_18_wheel_truck_l163_163967

-- Definitions based on conditions
def num_axles (total_wheels : ℕ) (wheels_front_axle : ℕ) (wheels_per_other_axle : ℕ) : ℕ :=
  1 + (total_wheels - wheels_front_axle) / wheels_per_other_axle

def toll (x : ℕ) : ℝ :=
  0.50 + 0.50 * (x - 2)

-- The problem statement to prove
theorem toll_for_18_wheel_truck : toll (num_axles 18 2 4) = 2.00 := by
  sorry

end toll_for_18_wheel_truck_l163_163967


namespace harmonic_mean_of_1_3_1_div_2_l163_163386

noncomputable def harmonicMean (a b c : ℝ) : ℝ :=
  let reciprocals := [1 / a, 1 / b, 1 / c]
  (reciprocals.sum) / reciprocals.length

theorem harmonic_mean_of_1_3_1_div_2 : harmonicMean 1 3 (1 / 2) = 9 / 10 :=
  sorry

end harmonic_mean_of_1_3_1_div_2_l163_163386


namespace equation_solution_l163_163936

open Real

theorem equation_solution (x : ℝ) : 
  (x = 4 ∨ x = -1 → 3 * (2 * x - 5) ≠ (2 * x - 5) ^ 2) ∧
  (3 * (2 * x - 5) = (2 * x - 5) ^ 2 → x = 5 / 2 ∨ x = 4) :=
by
  sorry

end equation_solution_l163_163936


namespace spending_on_gifts_l163_163859

-- Defining the conditions as Lean statements
def num_sons_teachers : ℕ := 3
def num_daughters_teachers : ℕ := 4
def cost_per_gift : ℕ := 10

-- The total number of teachers
def total_teachers : ℕ := num_sons_teachers + num_daughters_teachers

-- Proving that the total spending on gifts is $70
theorem spending_on_gifts : total_teachers * cost_per_gift = 70 :=
by
  -- proof goes here
  sorry

end spending_on_gifts_l163_163859


namespace subsequence_sum_q_l163_163198

theorem subsequence_sum_q (S : Fin 1995 → ℕ) (m : ℕ) (hS_pos : ∀ i : Fin 1995, 0 < S i)
  (hS_sum : (Finset.univ : Finset (Fin 1995)).sum S = m) (h_m_lt : m < 3990) :
  ∀ q : ℕ, 1 ≤ q → q ≤ m → ∃ (I : Finset (Fin 1995)), I.sum S = q := 
sorry

end subsequence_sum_q_l163_163198


namespace fabulous_integers_l163_163433

def is_fabulous (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ a : ℕ, 2 ≤ a ∧ a ≤ n - 1 ∧ (a^n - a) % n = 0

theorem fabulous_integers (n : ℕ) : is_fabulous n ↔ ¬(∃ k : ℕ, n = 2^k ∧ k ≥ 1) := 
sorry

end fabulous_integers_l163_163433


namespace mass_percentage_Al_in_mixture_l163_163692

/-- Define molar masses for the respective compounds -/
def molar_mass_AlCl3 : ℝ := 133.33
def molar_mass_Al2SO4_3 : ℝ := 342.17
def molar_mass_AlOH3 : ℝ := 78.01

/-- Define masses of respective compounds given in grams -/
def mass_AlCl3 : ℝ := 50
def mass_Al2SO4_3 : ℝ := 70
def mass_AlOH3 : ℝ := 40

/-- Define molar mass of Al -/
def molar_mass_Al : ℝ := 26.98

theorem mass_percentage_Al_in_mixture :
  (mass_AlCl3 / molar_mass_AlCl3 * molar_mass_Al +
   mass_Al2SO4_3 / molar_mass_Al2SO4_3 * (2 * molar_mass_Al) +
   mass_AlOH3 / molar_mass_AlOH3 * molar_mass_Al) / 
  (mass_AlCl3 + mass_Al2SO4_3 + mass_AlOH3) * 100 
  = 21.87 := by
  sorry

end mass_percentage_Al_in_mixture_l163_163692


namespace sufficient_but_not_necessary_condition_l163_163485

def P (x : ℝ) : Prop := 0 < x ∧ x < 5
def Q (x : ℝ) : Prop := |x - 2| < 3

theorem sufficient_but_not_necessary_condition
  (x : ℝ) : (P x → Q x) ∧ ¬(Q x → P x) :=
by
  sorry

end sufficient_but_not_necessary_condition_l163_163485


namespace circles_intersect_l163_163840

def circle_eq1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 3 = 0
def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y + 4 = 0

theorem circles_intersect :
  (∃ (x y : ℝ), circle_eq1 x y ∧ circle_eq2 x y) :=
sorry

end circles_intersect_l163_163840


namespace real_and_equal_roots_condition_l163_163930

theorem real_and_equal_roots_condition (k : ℝ) : 
  ∀ k : ℝ, (∃ (x : ℝ), 3 * x^2 + 6 * k * x + 9 = 0) ↔ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
by
  sorry

end real_and_equal_roots_condition_l163_163930


namespace ben_david_bagel_cost_l163_163902

theorem ben_david_bagel_cost (B D : ℝ)
  (h1 : D = 0.5 * B)
  (h2 : B = D + 16) :
  B + D = 48 := 
sorry

end ben_david_bagel_cost_l163_163902


namespace volume_region_between_concentric_spheres_l163_163295

open Real

theorem volume_region_between_concentric_spheres (r1 r2 : ℝ) (h_r1 : r1 = 4) (h_r2 : r2 = 8) :
  (4 / 3 * π * r2^3 - 4 / 3 * π * r1^3) = 1792 / 3 * π :=
by
  sorry

end volume_region_between_concentric_spheres_l163_163295


namespace find_n_mod_10_l163_163215

theorem find_n_mod_10 :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n % 10 = (-2023) % 10 ∧ n = 7 :=
sorry

end find_n_mod_10_l163_163215


namespace alice_probability_l163_163779

noncomputable def probability_picking_exactly_three_green_marbles : ℚ :=
  let binom : ℚ := 35 -- binomial coefficient (7 choose 3)
  let prob_green : ℚ := 8 / 15 -- probability of picking a green marble
  let prob_purple : ℚ := 7 / 15 -- probability of picking a purple marble
  binom * (prob_green ^ 3) * (prob_purple ^ 4)

theorem alice_probability :
  probability_picking_exactly_three_green_marbles = 34454336 / 136687500 := by
  sorry

end alice_probability_l163_163779


namespace quadratic_function_properties_l163_163502

-- We define the primary conditions
def axis_of_symmetry (f : ℝ → ℝ) (x_sym : ℝ) : Prop := 
  ∀ x, f x = f (2 * x_sym - x)

def minimum_value (f : ℝ → ℝ) (y_min : ℝ) (x_min : ℝ) : Prop := 
  ∀ x, f x_min ≤ f x

def passes_through (f : ℝ → ℝ) (pt : ℝ × ℝ) : Prop := 
  f pt.1 = pt.2

-- We need to prove that a quadratic function exists with the given properties and find intersections
theorem quadratic_function_properties :
  ∃ f : ℝ → ℝ,
    axis_of_symmetry f (-1) ∧
    minimum_value f (-4) (-1) ∧
    passes_through f (-2, 5) ∧
    (∀ y : ℝ, f 0 = y → y = 5) ∧
    (∀ x : ℝ, f x = 0 → (x = -5/3 ∨ x = -1/3)) :=
sorry

end quadratic_function_properties_l163_163502


namespace power_function_const_coeff_l163_163525

theorem power_function_const_coeff (m : ℝ) (h1 : m^2 + 2 * m - 2 = 1) (h2 : m ≠ 1) : m = -3 :=
  sorry

end power_function_const_coeff_l163_163525


namespace fraction_allocated_for_school_l163_163768

-- Conditions
def days_per_week : ℕ := 5
def hours_per_day : ℕ := 4
def earnings_per_hour : ℕ := 5
def allocation_for_school : ℕ := 75

-- Proof statement
theorem fraction_allocated_for_school :
  let weekly_hours := days_per_week * hours_per_day
  let weekly_earnings := weekly_hours * earnings_per_hour
  allocation_for_school / weekly_earnings = 3 / 4 := 
by
  sorry

end fraction_allocated_for_school_l163_163768


namespace allocation_schemes_for_5_teachers_to_3_buses_l163_163397

noncomputable def number_of_allocation_schemes (teachers : ℕ) (buses : ℕ) : ℕ :=
  if buses = 3 ∧ teachers = 5 then 150 else 0

theorem allocation_schemes_for_5_teachers_to_3_buses : 
  number_of_allocation_schemes 5 3 = 150 := 
by
  sorry

end allocation_schemes_for_5_teachers_to_3_buses_l163_163397


namespace b6_b8_value_l163_163736

def arithmetic_seq (a : ℕ → ℕ) := ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d
def nonzero_sequence (a : ℕ → ℕ) := ∀ n : ℕ, a n ≠ 0
def geometric_seq (b : ℕ → ℕ) := ∃ r : ℕ, ∀ n : ℕ, b (n + 1) = b n * r

theorem b6_b8_value (a b : ℕ → ℕ) (d : ℕ) 
  (h_arith : arithmetic_seq a) 
  (h_nonzero : nonzero_sequence a) 
  (h_cond1 : 2 * a 3 = a 1^2) 
  (h_cond2 : a 1 = d)
  (h_geo : geometric_seq b)
  (h_b13 : b 13 = a 2)
  (h_b1 : b 1 = a 1) :
  b 6 * b 8 = 72 := 
sorry

end b6_b8_value_l163_163736


namespace gcd_exponentiation_l163_163831

def m : ℕ := 2^2050 - 1
def n : ℕ := 2^2040 - 1

theorem gcd_exponentiation : Nat.gcd m n = 1023 := by
  sorry

end gcd_exponentiation_l163_163831


namespace expand_binomial_l163_163618

theorem expand_binomial (x : ℝ) : (x + 3) * (x + 8) = x^2 + 11 * x + 24 :=
by sorry

end expand_binomial_l163_163618


namespace digits_sum_is_23_l163_163835

/-
Juan chooses a five-digit positive integer.
Maria erases the ones digit and gets a four-digit number.
The sum of this four-digit number and the original five-digit number is 52,713.
What can the sum of the five digits of the original number be?
-/

theorem digits_sum_is_23 (x y : ℕ) (h1 : 1000 ≤ x) (h2 : x ≤ 9999) (h3 : y ≤ 9) (h4 : 11 * x + y = 52713) :
  (x / 1000) + (x / 100 % 10) + (x / 10 % 10) + (x % 10) + y = 23 :=
by {
  sorry -- Proof goes here.
}

end digits_sum_is_23_l163_163835


namespace probability_red_and_at_least_one_even_l163_163997

-- Definitions based on conditions
def total_balls : ℕ := 12
def red_balls : Finset ℕ := {1, 2, 3, 4, 5, 6}
def black_balls : Finset ℕ := {7, 8, 9, 10, 11, 12}

-- Condition to check if a ball is red
def is_red (n : ℕ) : Prop := n ∈ red_balls

-- Condition to check if a ball has an even number
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Total number of ways to draw two balls with replacement
def total_ways : ℕ := total_balls * total_balls

-- Number of ways to draw both red balls
def red_red_ways : ℕ := Finset.card red_balls * Finset.card red_balls

-- Number of ways to draw both red balls with none even
def red_odd_numbers : Finset ℕ := {1, 3, 5}
def red_red_odd_ways : ℕ := Finset.card red_odd_numbers * Finset.card red_odd_numbers

-- Number of ways to draw both red balls with at least one even
def desired_outcomes : ℕ := red_red_ways - red_red_odd_ways

-- The probability
def probability : ℚ := desired_outcomes / total_ways

theorem probability_red_and_at_least_one_even :
  probability = 3 / 16 :=
by
  sorry

end probability_red_and_at_least_one_even_l163_163997


namespace one_third_sugar_l163_163854

theorem one_third_sugar (sugar : ℚ) (h : sugar = 3 + 3 / 4) : sugar / 3 = 1 + 1 / 4 :=
by sorry

end one_third_sugar_l163_163854


namespace f_relation_l163_163176

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem f_relation :
  f (-Real.pi / 3) > f 1 ∧ f 1 > f (Real.pi / 5) :=
by
  sorry

end f_relation_l163_163176


namespace cora_cookies_per_day_l163_163360

theorem cora_cookies_per_day :
  (∀ (day : ℕ), day ∈ (Finset.range 30) →
    ∃ cookies_per_day : ℕ,
    cookies_per_day * 30 = 1620 / 18) →
  cookies_per_day = 3 := by
  sorry

end cora_cookies_per_day_l163_163360


namespace librarian_donated_200_books_this_year_l163_163139

noncomputable def total_books_five_years_ago : ℕ := 500
noncomputable def books_bought_two_years_ago : ℕ := 300
noncomputable def books_bought_last_year : ℕ := books_bought_two_years_ago + 100
noncomputable def total_books_current : ℕ := 1000

-- The Lean statement to prove the librarian donated 200 old books this year
theorem librarian_donated_200_books_this_year :
  total_books_five_years_ago + books_bought_two_years_ago + books_bought_last_year - total_books_current = 200 :=
by sorry

end librarian_donated_200_books_this_year_l163_163139


namespace pages_to_read_tomorrow_l163_163248

-- Define the conditions
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Lean statement for the proof problem
theorem pages_to_read_tomorrow : (total_pages - (pages_yesterday + pages_today) = 35) :=
by
  let yesterday := pages_yesterday
  let today := pages_today
  let read_so_far := yesterday + today
  have read_so_far_eq : yesterday + today = 65 := by sorry
  have total_eq : total_pages - read_so_far = 35 := by sorry
  exact total_eq

end pages_to_read_tomorrow_l163_163248


namespace event_A_probability_l163_163862

theorem event_A_probability (n : ℕ) (m₀ : ℕ) (H_n : n = 120) (H_m₀ : m₀ = 32) (p : ℝ) :
  (n * p - (1 - p) ≤ m₀) ∧ (n * p + p ≥ m₀) → 
  (32 / 121 : ℝ) ≤ p ∧ p ≤ (33 / 121 : ℝ) :=
sorry

end event_A_probability_l163_163862


namespace derivative_at_1_l163_163388

def f (x : ℝ) : ℝ := (1 - 2 * x^3) ^ 10

theorem derivative_at_1 : deriv f 1 = 60 :=
by
  sorry

end derivative_at_1_l163_163388


namespace gcd_possible_values_count_l163_163857

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end gcd_possible_values_count_l163_163857


namespace coins_difference_l163_163533

theorem coins_difference (p n d : ℕ) (h1 : p + n + d = 3030)
  (h2 : 1 ≤ p) (h3 : 1 ≤ n) (h4 : 1 ≤ d) (h5 : p ≤ 3029) (h6 : n ≤ 3029) (h7 : d ≤ 3029) :
  (max (p + 5 * n + 10 * d) (max (p + 5 * n + 10 * (3030 - p - n)) (3030 - n - d + 5 * d + 10 * p))) - 
  (min (p + 5 * n + 10 * d) (min (p + 5 * n + 10 * (3030 - p - n)) (3030 - n - d + 5 * d + 10 * p))) = 27243 := 
sorry

end coins_difference_l163_163533


namespace number_of_players_l163_163744

theorem number_of_players (n : ℕ) (G : ℕ) (h : G = 2 * n * (n - 1)) : n = 19 :=
by {
  sorry
}

end number_of_players_l163_163744


namespace tom_sleep_hours_l163_163307

-- Define initial sleep hours and increase fraction
def initial_sleep_hours : ℕ := 6
def increase_fraction : ℚ := 1 / 3

-- Define the function to calculate increased sleep
def increased_sleep_hours (initial : ℕ) (fraction : ℚ) : ℚ :=
  initial * fraction

-- Define the function to calculate total sleep hours
def total_sleep_hours (initial : ℕ) (increased : ℚ) : ℚ :=
  initial + increased

-- Theorem stating Tom's total sleep hours per night after the increase
theorem tom_sleep_hours (initial : ℕ) (fraction : ℚ) (increased : ℚ) (total : ℚ) :
  initial = initial_sleep_hours →
  fraction = increase_fraction →
  increased = increased_sleep_hours initial fraction →
  total = total_sleep_hours initial increased →
  total = 8 :=
by
  intros h_init h_frac h_incr h_total
  rw [h_init, h_frac] at h_incr
  rw [h_init, h_incr] at h_total
  sorry

end tom_sleep_hours_l163_163307


namespace vanessa_savings_weeks_l163_163623

-- Define the conditions as constants
def dress_cost : ℕ := 120
def initial_savings : ℕ := 25
def weekly_allowance : ℕ := 30
def weekly_arcade_spending : ℕ := 15
def weekly_snack_spending : ℕ := 5

-- The theorem statement based on the problem
theorem vanessa_savings_weeks : 
  ∃ (n : ℕ), (n * (weekly_allowance - weekly_arcade_spending - weekly_snack_spending) + initial_savings) ≥ dress_cost ∧ 
             (n - 1) * (weekly_allowance - weekly_arcade_spending - weekly_snack_spending) + initial_savings < dress_cost := by
  sorry

end vanessa_savings_weeks_l163_163623


namespace smallest_value_floor_l163_163237

theorem smallest_value_floor (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) :
  (⌊(2 * a + b) / c⌋ + ⌊(b + 2 * c) / a⌋ + ⌊(c + 2 * a) / b⌋) = 9 :=
sorry

end smallest_value_floor_l163_163237


namespace Q3_x_coords_sum_eq_Q1_x_coords_sum_l163_163030

-- Define a 40-gon and its x-coordinates sum
def Q1_x_coords_sum : ℝ := 120

-- Statement to prove
theorem Q3_x_coords_sum_eq_Q1_x_coords_sum (Q1_x_coords_sum: ℝ) (h: Q1_x_coords_sum = 120) : 
  (Q3_x_coords_sum: ℝ) = Q1_x_coords_sum :=
sorry

end Q3_x_coords_sum_eq_Q1_x_coords_sum_l163_163030


namespace yoongi_age_l163_163535

theorem yoongi_age (H Yoongi : ℕ) : H = Yoongi + 2 ∧ H + Yoongi = 18 → Yoongi = 8 :=
by
  sorry

end yoongi_age_l163_163535


namespace ryan_more_hours_english_than_chinese_l163_163166

-- Definitions for the time Ryan spends on subjects
def weekday_hours_english : ℕ := 6 * 5
def weekend_hours_english : ℕ := 2 * 2
def total_hours_english : ℕ := weekday_hours_english + weekend_hours_english

def weekday_hours_chinese : ℕ := 3 * 5
def weekend_hours_chinese : ℕ := 1 * 2
def total_hours_chinese : ℕ := weekday_hours_chinese + weekend_hours_chinese

-- Theorem stating the difference in hours spent on English vs Chinese
theorem ryan_more_hours_english_than_chinese :
  (total_hours_english - total_hours_chinese) = 17 := by
  sorry

end ryan_more_hours_english_than_chinese_l163_163166


namespace michael_choose_classes_l163_163449

-- Michael's scenario setup
def total_classes : ℕ := 10
def compulsory_class : ℕ := 1
def remaining_classes : ℕ := total_classes - compulsory_class
def total_to_choose : ℕ := 4
def additional_to_choose : ℕ := total_to_choose - compulsory_class

-- Correct answer based on the conditions
def correct_answer : ℕ := 84

-- Function to compute the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem to prove the number of ways Michael can choose his classes
theorem michael_choose_classes : binomial 9 3 = correct_answer := by
  rw [binomial, Nat.factorial]
  sorry

end michael_choose_classes_l163_163449


namespace find_x_l163_163429

theorem find_x (n x : ℚ) (h1 : 3 * n + x = 6 * n - 10) (h2 : n = 25 / 3) : x = 15 :=
by
  sorry

end find_x_l163_163429


namespace product_and_sum_of_roots_l163_163351

theorem product_and_sum_of_roots :
  let a := 24
  let b := 60
  let c := -600
  (c / a = -25) ∧ (-b / a = -2.5) := 
by
  sorry

end product_and_sum_of_roots_l163_163351


namespace find_middle_number_l163_163622

theorem find_middle_number (x y z : ℤ) (h1 : x + y = 22) (h2 : x + z = 29) (h3 : y + z = 37) (h4 : x < y) (h5 : y < z) : y = 15 :=
by
  sorry

end find_middle_number_l163_163622


namespace inequality1_inequality2_l163_163941

theorem inequality1 (x : ℝ) : x ≠ 2 → (x + 1)/(x - 2) ≥ 3 → 2 < x ∧ x ≤ 7/2 :=
sorry

theorem inequality2 (x a : ℝ) : 
  (x^2 - a * x - 2 * a^2 ≤ 0) → 
  (a = 0 → x = 0) ∧ 
  (a > 0 → -a ≤ x ∧ x ≤ 2 * a) ∧ 
  (a < 0 → 2 * a ≤ x ∧ x ≤ -a) :=
sorry

end inequality1_inequality2_l163_163941


namespace find_age_of_mother_l163_163064

def Grace_age := 60
def ratio_GM_Grace := 3 / 8
def ratio_GM_Mother := 2

theorem find_age_of_mother (G M GM : ℕ) (h1 : G = ratio_GM_Grace * GM) 
                           (h2 : GM = ratio_GM_Mother * M) (h3 : G = Grace_age) : 
  M = 80 :=
by
  sorry

end find_age_of_mother_l163_163064


namespace find_first_term_l163_163475

def geom_seq (a r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem find_first_term (a r : ℝ) (h1 : r = 2/3) (h2 : geom_seq a r 3 = 18) (h3 : geom_seq a r 4 = 12) : a = 40.5 := 
by sorry

end find_first_term_l163_163475


namespace Julie_can_print_complete_newspapers_l163_163235

def sheets_in_box_A : ℕ := 4 * 200
def sheets_in_box_B : ℕ := 3 * 350
def total_sheets : ℕ := sheets_in_box_A + sheets_in_box_B

def front_section_sheets : ℕ := 10
def sports_section_sheets : ℕ := 7
def arts_section_sheets : ℕ := 5
def events_section_sheets : ℕ := 3

def sheets_per_newspaper : ℕ := front_section_sheets + sports_section_sheets + arts_section_sheets + events_section_sheets

theorem Julie_can_print_complete_newspapers : total_sheets / sheets_per_newspaper = 74 := by
  sorry

end Julie_can_print_complete_newspapers_l163_163235


namespace min_value_of_y_l163_163765

noncomputable def y (x : ℝ) : ℝ := x^2 + 26 * x + 7

theorem min_value_of_y : ∃ x : ℝ, y x = -162 :=
by
  use -13
  sorry

end min_value_of_y_l163_163765


namespace prove_real_roots_and_find_m_l163_163094

-- Condition: The quadratic equation
def quadratic_eq (m x : ℝ) : Prop := x^2 - (m-1)*x + m-2 = 0

-- Condition: Discriminant
def discriminant (m : ℝ) : ℝ := (m-3)^2

-- Define the problem as a proposition
theorem prove_real_roots_and_find_m (m : ℝ) :
  (discriminant m ≥ 0) ∧ 
  (|3 - m| = 3 → (m = 0 ∨ m = 6)) :=
by
  sorry

end prove_real_roots_and_find_m_l163_163094


namespace smallest_natural_number_divisible_l163_163187

theorem smallest_natural_number_divisible :
  ∃ n : ℕ, (n^2 + 14 * n + 13) % 68 = 0 ∧ 
          ∀ m : ℕ, (m^2 + 14 * m + 13) % 68 = 0 → 21 ≤ m :=
by 
  sorry

end smallest_natural_number_divisible_l163_163187


namespace survey_pop_and_coke_l163_163276

theorem survey_pop_and_coke (total_people : ℕ) (angle_pop angle_coke : ℕ) 
  (h_total : total_people = 500) (h_angle_pop : angle_pop = 240) (h_angle_coke : angle_coke = 90) :
  ∃ (pop_people coke_people : ℕ), pop_people = 333 ∧ coke_people = 125 :=
by 
  sorry

end survey_pop_and_coke_l163_163276


namespace triangle_perimeter_l163_163324

theorem triangle_perimeter (A r p : ℝ) (hA : A = 75) (hr : r = 2.5) :
  A = r * (p / 2) → p = 60 := by
  intros
  sorry

end triangle_perimeter_l163_163324


namespace infinite_series_sum_l163_163380

theorem infinite_series_sum :
  (∑' n : ℕ, (3^n) / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = 1 / 4 :=
by
  sorry

end infinite_series_sum_l163_163380


namespace inequality_for_real_numbers_l163_163657

theorem inequality_for_real_numbers (x y z : ℝ) : 
  - (3 / 2) * (x^2 + y^2 + 2 * z^2) ≤ 3 * x * y + y * z + z * x ∧ 
  3 * x * y + y * z + z * x ≤ (3 + Real.sqrt 13) / 4 * (x^2 + y^2 + 2 * z^2) :=
by
  sorry

end inequality_for_real_numbers_l163_163657


namespace students_in_both_math_and_chem_l163_163912

theorem students_in_both_math_and_chem (students total math physics chem math_physics physics_chem : ℕ) :
  total = 36 →
  students ≤ 2 →
  math = 26 →
  physics = 15 →
  chem = 13 →
  math_physics = 6 →
  physics_chem = 4 →
  math + physics + chem - math_physics - physics_chem - students = total →
  students = 8 := by
  intros h_total h_students h_math h_physics h_chem h_math_physics h_physics_chem h_equation
  sorry

end students_in_both_math_and_chem_l163_163912


namespace find_a_purely_imaginary_z1_z2_l163_163774

noncomputable def z1 (a : ℝ) : ℂ := ⟨a^2 - 3, a + 5⟩
noncomputable def z2 (a : ℝ) : ℂ := ⟨a - 1, a^2 + 2 * a - 1⟩

theorem find_a_purely_imaginary_z1_z2 (a : ℝ)
    (h_imaginary : ∃ b : ℝ, z2 a - z1 a = ⟨0, b⟩) : 
    a = -1 :=
sorry

end find_a_purely_imaginary_z1_z2_l163_163774


namespace boards_cannot_be_covered_by_dominos_l163_163762

-- Definitions of the boards
def board_6x4 := (6 : ℕ) * (4 : ℕ)
def board_5x5 := (5 : ℕ) * (5 : ℕ)
def board_L_shaped := (5 : ℕ) * (5 : ℕ) - (2 : ℕ) * (2 : ℕ)
def board_3x7 := (3 : ℕ) * (7 : ℕ)
def board_plus_shaped := (3 : ℕ) * (3 : ℕ) + (1 : ℕ) * (3 : ℕ)

-- Definition to check if a board can't be covered by dominoes
def cannot_be_covered_by_dominos (n : ℕ) : Prop := n % 2 = 1

-- Theorem stating which specific boards cannot be covered by dominoes
theorem boards_cannot_be_covered_by_dominos :
  cannot_be_covered_by_dominos board_5x5 ∧
  cannot_be_covered_by_dominos board_L_shaped ∧
  cannot_be_covered_by_dominos board_3x7 :=
by
  -- Proof here
  sorry

end boards_cannot_be_covered_by_dominos_l163_163762


namespace cookies_in_one_row_l163_163460

theorem cookies_in_one_row
  (num_trays : ℕ) (rows_per_tray : ℕ) (total_cookies : ℕ)
  (h_trays : num_trays = 4) (h_rows : rows_per_tray = 5) (h_cookies : total_cookies = 120) :
  total_cookies / (num_trays * rows_per_tray) = 6 := by
  sorry

end cookies_in_one_row_l163_163460


namespace right_triangle_height_l163_163735

theorem right_triangle_height
  (h : ℕ)
  (base : ℕ)
  (rectangle_area : ℕ)
  (same_area : (1 / 2 : ℚ) * base * h = rectangle_area)
  (base_eq_width : base = 5)
  (rectangle_area_eq : rectangle_area = 45) :
  h = 18 :=
by
  sorry

end right_triangle_height_l163_163735


namespace percentage_speaking_both_langs_l163_163542

def diplomats_total : ℕ := 100
def diplomats_french : ℕ := 22
def diplomats_not_russian : ℕ := 32
def diplomats_neither : ℕ := 20

theorem percentage_speaking_both_langs
  (h1 : 20% diplomats_total = diplomats_neither)
  (h2 : diplomats_total - diplomats_not_russian = 68)
  (h3 : diplomats_total ≠ 0) :
  (22 + 68 - 80) / diplomats_total * 100 = 10 :=
by
  sorry

end percentage_speaking_both_langs_l163_163542


namespace max_sum_of_arithmetic_seq_l163_163341

theorem max_sum_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h₁ : a 1 = 11) (h₂ : a 5 = -1) 
  (h₃ : ∀ n, a n = 14 - 3 * (n - 1)) 
  : ∀ n, (S n = (n * (a 1 + a n) / 2)) → max (S n) = 26 :=
sorry

end max_sum_of_arithmetic_seq_l163_163341


namespace square_area_with_circles_l163_163540

theorem square_area_with_circles (r : ℝ) (h : r = 8) : (2 * (2 * r))^2 = 1024 := 
by 
  sorry

end square_area_with_circles_l163_163540


namespace red_to_blue_l163_163970

def is_red (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 2020

def is_blue (n : ℕ) : Prop :=
  ¬ is_red n ∧ ∃ m : ℕ, n = m ^ 2019

theorem red_to_blue (n : ℕ) (hn : n > 10^100000000) (hnred : is_red n) 
    (hn1red : is_red (n+1)) :
    ∃ (k : ℕ), 1 ≤ k ∧ k ≤ 2019 ∧ is_blue (n + k) :=
sorry

end red_to_blue_l163_163970


namespace capsule_depth_equation_l163_163685

theorem capsule_depth_equation (x y z : ℝ) (h : y = 4 * x + z) : y = 4 * x + z := 
by 
  exact h

end capsule_depth_equation_l163_163685


namespace comm_add_comm_mul_distrib_l163_163615

variable {α : Type*} [AddCommMonoid α] [Mul α] [Distrib α]

theorem comm_add (a b : α) : a + b = b + a :=
by sorry

theorem comm_mul (a b : α) : a * b = b * a :=
by sorry

theorem distrib (a b c : α) : (a + b) * c = a * c + b * c :=
by sorry

end comm_add_comm_mul_distrib_l163_163615


namespace find_a_for_exactly_two_solutions_l163_163102

theorem find_a_for_exactly_two_solutions :
  ∃ a : ℝ, (∀ x : ℝ, (|x + a| = 1/x) ↔ (a = -2) ∧ (x ≠ 0)) ∧ ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (|x1 + a| = 1/x1) ∧ (|x2 + a| = 1/x2) :=
sorry

end find_a_for_exactly_two_solutions_l163_163102


namespace person_next_to_Boris_arkady_galya_l163_163318

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l163_163318


namespace range_of_a_l163_163751

open Real

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ , x^2 + a * x + 1 < 0) ↔ (-2 : ℝ) ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l163_163751


namespace num_valid_arrangements_without_A_at_start_and_B_at_end_l163_163068

-- Define a predicate for person A being at the beginning
def A_at_beginning (arrangement : List ℕ) : Prop :=
  arrangement.head! = 1

-- Define a predicate for person B being at the end
def B_at_end (arrangement : List ℕ) : Prop :=
  arrangement.getLast! = 2

-- Main theorem stating the number of valid arrangements
theorem num_valid_arrangements_without_A_at_start_and_B_at_end : ∃ (count : ℕ), count = 78 :=
by
  have total_arrangements := Nat.factorial 5
  have A_at_start_arrangements := Nat.factorial 4
  have B_at_end_arrangements := Nat.factorial 4
  have both_A_and_B_arrangements := Nat.factorial 3
  let valid_arrangements := total_arrangements - 2 * A_at_start_arrangements + both_A_and_B_arrangements
  use valid_arrangements
  sorry

end num_valid_arrangements_without_A_at_start_and_B_at_end_l163_163068


namespace g_symmetric_l163_163720

theorem g_symmetric (g : ℝ → ℝ) (h₀ : ∀ x, x ≠ 0 → (g x + 3 * g (1 / x) = 4 * x ^ 2)) : 
  ∀ x : ℝ, x ≠ 0 → g x = g (-x) :=
by 
  sorry

end g_symmetric_l163_163720


namespace shaded_region_area_l163_163023

noncomputable def radius1 := 4
noncomputable def radius2 := 5
noncomputable def distance := radius1 + radius2
noncomputable def large_radius := radius2 + distance / 2

theorem shaded_region_area :
  ∃ (A : ℝ), A = (π * large_radius ^ 2) - (π * radius1 ^ 2) - (π * radius2 ^ 2) ∧
  A = 49.25 * π :=
by
  sorry

end shaded_region_area_l163_163023


namespace box_volume_increase_l163_163552

theorem box_volume_increase (l w h : ℝ) 
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end box_volume_increase_l163_163552


namespace intersection_of_A_and_B_l163_163963

-- Define the sets A and B based on the given conditions
def A := {x : ℝ | x > 1}
def B := {x : ℝ | x ≤ 3}

-- Lean statement to prove the intersection of A and B matches the correct answer
theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 3} :=
by {
  sorry
}

end intersection_of_A_and_B_l163_163963


namespace pow_two_ge_square_l163_163185

theorem pow_two_ge_square {n : ℕ} (hn : n ≥ 4) : 2^n ≥ n^2 :=
sorry

end pow_two_ge_square_l163_163185


namespace joe_time_to_school_l163_163279

theorem joe_time_to_school
    (r_w : ℝ) -- Joe's walking speed
    (t_w : ℝ) -- Time to walk halfway
    (t_stop : ℝ) -- Time stopped at the store
    (r_running_factor : ℝ) -- Factor by which running speed is faster than walking speed
    (initial_walk_time_halfway : t_w = 10)
    (store_stop_time : t_stop = 3)
    (running_speed_factor : r_running_factor = 4) :
    t_w + t_stop + t_w / r_running_factor = 15.5 :=
by
    -- Implementation skipped, just verifying statement is correctly captured
    sorry

end joe_time_to_school_l163_163279


namespace polynomial_expansion_l163_163103

theorem polynomial_expansion :
  ∃ A B C D : ℝ, (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) 
  ∧ (A + B + C + D = 36) :=
by {
  sorry
}

end polynomial_expansion_l163_163103


namespace required_connections_l163_163465

theorem required_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 := by
  sorry

end required_connections_l163_163465


namespace percent_of_liquidX_in_solutionB_l163_163572

theorem percent_of_liquidX_in_solutionB (P : ℝ) (h₁ : 0.8 / 100 = 0.008) 
(h₂ : 1.5 / 100 = 0.015) 
(h₃ : 300 * 0.008 = 2.4) 
(h₄ : 1000 * 0.015 = 15) 
(h₅ : 15 - 2.4 = 12.6) 
(h₆ : 12.6 / 700 = P) : 
P * 100 = 1.8 :=
by sorry

end percent_of_liquidX_in_solutionB_l163_163572


namespace probability_Q_within_three_units_of_origin_l163_163339

noncomputable def probability_within_three_units_of_origin :=
  let radius := 3
  let square_side := 10
  let circle_area := Real.pi * radius^2
  let square_area := square_side^2
  circle_area / square_area

theorem probability_Q_within_three_units_of_origin :
  probability_within_three_units_of_origin = 9 * Real.pi / 100 :=
by
  -- Since this proof is not required, we skip it with sorry.
  sorry

end probability_Q_within_three_units_of_origin_l163_163339


namespace units_digit_of_fraction_l163_163821

-- Define the problem
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_fraction :
  units_digit ((30 * 31 * 32 * 33 * 34 * 35) / 2500) = 2 := by
  sorry

end units_digit_of_fraction_l163_163821


namespace additional_flour_minus_salt_l163_163265

structure CakeRecipe where
  flour    : ℕ
  sugar    : ℕ
  salt     : ℕ

def MaryHasAdded (cups_flour : ℕ) (cups_sugar : ℕ) (cups_salt : ℕ) : Prop :=
  cups_flour = 2 ∧ cups_sugar = 0 ∧ cups_salt = 0

variable (r : CakeRecipe)

theorem additional_flour_minus_salt (H : MaryHasAdded 2 0 0) : 
  (r.flour - 2) - r.salt = 3 :=
sorry

end additional_flour_minus_salt_l163_163265


namespace evaluate_g_neg_1_l163_163306

noncomputable def g (x : ℝ) : ℝ := -2 * x^2 + 5 * x - 7

theorem evaluate_g_neg_1 : g (-1) = -14 := 
by
  sorry

end evaluate_g_neg_1_l163_163306


namespace opposite_of_2023_l163_163054

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l163_163054


namespace time_to_cross_bridge_l163_163961

def train_length : ℕ := 600  -- train length in meters
def bridge_length : ℕ := 100  -- overbridge length in meters
def speed_km_per_hr : ℕ := 36  -- speed of the train in kilometers per hour

-- Convert speed from km/h to m/s
def speed_m_per_s : ℕ := speed_km_per_hr * 1000 / 3600

-- Compute the total distance
def total_distance : ℕ := train_length + bridge_length

-- Prove the time to cross the overbridge
theorem time_to_cross_bridge : total_distance / speed_m_per_s = 70 := by
  sorry

end time_to_cross_bridge_l163_163961


namespace part1_part2_part3_l163_163095

section Part1
variables {a b : ℝ}

theorem part1 (h1 : a + b = 3) (h2 : a * b = 2) : a^2 + b^2 = 5 := 
sorry
end Part1

section Part2
variables {a b c : ℝ}

theorem part2 (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) : a^2 + b^2 + c^2 = 14 := 
sorry
end Part2

section Part3
variables {a b c : ℝ}

theorem part3 (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : a^4 + b^4 + c^4 = 18 :=
sorry
end Part3

end part1_part2_part3_l163_163095


namespace egyptian_fraction_decomposition_l163_163981

theorem egyptian_fraction_decomposition (n : ℕ) (hn : 0 < n) : 
  (2 : ℚ) / (2 * n + 1) = (1 : ℚ) / (n + 1) + (1 : ℚ) / ((n + 1) * (2 * n + 1)) := 
by {
  sorry
}

end egyptian_fraction_decomposition_l163_163981


namespace solve_quadratic_equation_l163_163239

theorem solve_quadratic_equation (x : ℝ) : 
  (2 * x^2 - 4 * x = 6 - 3 * x) ↔ 
  (x = -3/2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_equation_l163_163239


namespace distance_origin_to_point_l163_163049

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_origin_to_point :
  distance (0, 0) (-15, 8) = 17 :=
by 
  sorry

end distance_origin_to_point_l163_163049


namespace penguins_seals_ratio_l163_163869

theorem penguins_seals_ratio (t_total t_seals t_elephants t_penguins : ℕ) 
    (h1 : t_total = 130) 
    (h2 : t_seals = 13) 
    (h3 : t_elephants = 13) 
    (h4 : t_penguins = t_total - t_seals - t_elephants) : 
    (t_penguins / t_seals = 8) := by
  sorry

end penguins_seals_ratio_l163_163869


namespace complex_modulus_square_l163_163823

open Complex

theorem complex_modulus_square (a b : ℝ) (h : 5 * (a + b * I) + 3 * Complex.abs (a + b * I) = 15 - 16 * I) :
  (Complex.abs (a + b * I))^2 = 256 / 25 :=
by sorry

end complex_modulus_square_l163_163823


namespace gcd_polynomial_l163_163763

theorem gcd_polynomial (n : ℕ) (h : n > 2^5) :
  Nat.gcd (n^5 + 125) (n + 5) = if n % 5 = 0 then 5 else 1 :=
by
  sorry

end gcd_polynomial_l163_163763


namespace triangle_is_right_l163_163848

-- Definitions based on the conditions given in the problem
variables {a b c A B C : ℝ}

-- Introduction of the conditions in Lean
def is_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)

def given_condition (A b c : ℝ) : Prop :=
  (Real.cos (A / 2))^2 = (b + c) / (2 * c)

-- Theorem statement to prove the conclusion based on given conditions
theorem triangle_is_right (a b c A B C : ℝ) 
  (h_triangle : is_triangle a b c A B C)
  (h_given : given_condition A b c) :
  A = 90 := sorry

end triangle_is_right_l163_163848


namespace always_positive_iff_k_gt_half_l163_163284

theorem always_positive_iff_k_gt_half (k : ℝ) :
  (∀ x : ℝ, k * x^2 + x + k > 0) ↔ k > 0.5 :=
sorry

end always_positive_iff_k_gt_half_l163_163284


namespace maximum_sum_set_l163_163018

def no_two_disjoint_subsets_have_equal_sums (S : Finset ℕ) : Prop :=
  ∀ (A B : Finset ℕ), A ≠ B ∧ A ∩ B = ∅ → (A.sum id) ≠ (B.sum id)

theorem maximum_sum_set (S : Finset ℕ) (h : ∀ x ∈ S, x ≤ 15) (h_subset_sum : no_two_disjoint_subsets_have_equal_sums S) : S.sum id = 61 :=
sorry

end maximum_sum_set_l163_163018


namespace kindergarten_students_percentage_is_correct_l163_163463

-- Definitions based on conditions
def total_students_annville : ℕ := 150
def total_students_cleona : ℕ := 250
def percent_kindergarten_annville : ℕ := 14
def percent_kindergarten_cleona : ℕ := 10

-- Calculation of number of kindergarten students
def kindergarten_students_annville : ℕ := total_students_annville * percent_kindergarten_annville / 100
def kindergarten_students_cleona : ℕ := total_students_cleona * percent_kindergarten_cleona / 100
def total_kindergarten_students : ℕ := kindergarten_students_annville + kindergarten_students_cleona
def total_students : ℕ := total_students_annville + total_students_cleona
def kindergarten_percentage : ℚ := (total_kindergarten_students * 100) / total_students

-- The theorem to be proved using the conditions
theorem kindergarten_students_percentage_is_correct : kindergarten_percentage = 11.5 := by
  sorry

end kindergarten_students_percentage_is_correct_l163_163463


namespace candy_bar_cost_l163_163716

def initial_amount : ℕ := 4
def remaining_amount : ℕ := 3
def cost_of_candy_bar : ℕ := initial_amount - remaining_amount

theorem candy_bar_cost : cost_of_candy_bar = 1 := by
  sorry

end candy_bar_cost_l163_163716


namespace arithmetic_sequence_value_y_l163_163849

theorem arithmetic_sequence_value_y :
  ∀ (a₁ a₃ y : ℤ), 
  a₁ = 3 ^ 3 →
  a₃ = 5 ^ 3 →
  y = (a₁ + a₃) / 2 →
  y = 76 :=
by 
  intros a₁ a₃ y h₁ h₃ hy 
  sorry

end arithmetic_sequence_value_y_l163_163849


namespace bc_guilty_l163_163881

-- Definition of guilty status of defendants
variables (A B C : Prop)

-- Conditions
axiom condition1 : A ∨ B ∨ C
axiom condition2 : A → ¬B → ¬C

-- Theorem stating that one of B or C is guilty
theorem bc_guilty : B ∨ C :=
by {
  -- Proof goes here
  sorry
}

end bc_guilty_l163_163881


namespace greatest_multiple_of_5_and_6_less_than_1000_l163_163724

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l163_163724


namespace cricketer_initial_average_l163_163646

def initial_bowling_average
  (runs_for_last_5_wickets : ℝ)
  (decreased_average : ℝ)
  (final_wickets : ℝ)
  (initial_wickets : ℝ)
  (initial_average : ℝ) : Prop :=
  (initial_average * initial_wickets + runs_for_last_5_wickets) / final_wickets =
    initial_average - decreased_average

theorem cricketer_initial_average :
  initial_bowling_average 26 0.4 85 80 12 :=
by
  unfold initial_bowling_average
  sorry

end cricketer_initial_average_l163_163646


namespace pleasant_goat_paths_l163_163201

-- Define the grid points A, B, and C
structure Point :=
  (x : ℕ)
  (y : ℕ)

def A : Point := { x := 0, y := 0 }
def C : Point := { x := 3, y := 3 }  -- assuming some grid layout
def B : Point := { x := 1, y := 1 }

-- Define a statement to count the number of shortest paths
def shortest_paths_count (A B C : Point) : ℕ := sorry

-- Proving the shortest paths from A to C avoiding B is 81
theorem pleasant_goat_paths : shortest_paths_count A B C = 81 := 
sorry

end pleasant_goat_paths_l163_163201


namespace two_digit_integer_plus_LCM_of_3_4_5_l163_163164

theorem two_digit_integer_plus_LCM_of_3_4_5 (x : ℕ) (h1 : 9 < x) (h2 : x < 100) (h3 : ∃ k, x = 60 * k + 2) :
  x = 62 :=
by {
  sorry
}

end two_digit_integer_plus_LCM_of_3_4_5_l163_163164


namespace blocks_total_l163_163993

theorem blocks_total (blocks_initial : ℕ) (blocks_added : ℕ) (total_blocks : ℕ) 
  (h1 : blocks_initial = 86) (h2 : blocks_added = 9) : total_blocks = 95 :=
by
  sorry

end blocks_total_l163_163993


namespace could_be_simple_random_sampling_l163_163915

-- Conditions
def boys : Nat := 20
def girls : Nat := 30
def total_students : Nat := boys + girls
def sample_size : Nat := 10
def boys_in_sample : Nat := 4
def girls_in_sample : Nat := 6

-- Theorem Statement
theorem could_be_simple_random_sampling :
  boys = 20 ∧ girls = 30 ∧ sample_size = 10 ∧ boys_in_sample = 4 ∧ girls_in_sample = 6 →
  (∃ (sample_method : String), sample_method = "simple random sampling"):=
by 
  sorry

end could_be_simple_random_sampling_l163_163915


namespace product_value_l163_163705

-- Definitions of each term
def term (n : Nat) : Rat :=
  1 + 1 / (n^2 : ℚ)

-- Define the product of these terms
def product : Rat :=
  term 1 * term 2 * term 3 * term 4 * term 5 * term 6

-- The proof problem statement that needs to be verified
theorem product_value :
  product = 16661 / 3240 :=
sorry

end product_value_l163_163705


namespace exterior_angle_regular_octagon_l163_163613

-- Definition and proof statement
theorem exterior_angle_regular_octagon :
  let n := 8 -- The number of sides of the polygon (octagon)
  let interior_angle_sum := 180 * (n - 2) 
  let interior_angle := interior_angle_sum / n
  let exterior_angle := 180 - interior_angle
  exterior_angle = 45 :=
by
  let n := 8
  let interior_angle_sum := 180 * (n - 2) 
  let interior_angle := interior_angle_sum / n
  let exterior_angle := 180 - interior_angle
  sorry

end exterior_angle_regular_octagon_l163_163613


namespace solution_set_absolute_value_inequality_l163_163389

theorem solution_set_absolute_value_inequality (x : ℝ) :
  (|x-3| + |x-5| ≥ 4) ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end solution_set_absolute_value_inequality_l163_163389


namespace three_angles_difference_is_2pi_over_3_l163_163153

theorem three_angles_difference_is_2pi_over_3 (α β γ : ℝ) 
    (h1 : 0 ≤ α) (h2 : α ≤ β) (h3 : β < γ) (h4 : γ ≤ 2 * Real.pi)
    (h5 : Real.cos α + Real.cos β + Real.cos γ = 0)
    (h6 : Real.sin α + Real.sin β + Real.sin γ = 0) :
    β - α = 2 * Real.pi / 3 :=
sorry

end three_angles_difference_is_2pi_over_3_l163_163153


namespace amy_total_tickets_l163_163048

theorem amy_total_tickets (initial_tickets additional_tickets : ℕ) (h_initial : initial_tickets = 33) (h_additional : additional_tickets = 21) : 
  initial_tickets + additional_tickets = 54 := 
by 
  sorry

end amy_total_tickets_l163_163048


namespace negation_q_sufficient_not_necessary_negation_p_l163_163801

theorem negation_q_sufficient_not_necessary_negation_p :
  (∃ x : ℝ, (∃ p : 16 - x^2 < 0, (x ∈ [-4, 4]))) →
  (∃ x : ℝ, (∃ q : x^2 + x - 6 > 0, (x ∈ [-3, 2]))) :=
sorry

end negation_q_sufficient_not_necessary_negation_p_l163_163801


namespace problem1_l163_163539

theorem problem1 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) : 
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) :=
sorry

end problem1_l163_163539


namespace tennis_tournament_l163_163031

theorem tennis_tournament (n x : ℕ) 
    (p : ℕ := 4 * n) 
    (m : ℕ := (p * (p - 1)) / 2) 
    (r_women : ℕ := 3 * x) 
    (r_men : ℕ := 2 * x) 
    (total_wins : ℕ := r_women + r_men) 
    (h_matches : m = total_wins) 
    (h_ratio : r_women = 3 * x ∧ r_men = 2 * x ∧ 4 * n * (4 * n - 1) = 10 * x): 
    n = 4 :=
by
  sorry

end tennis_tournament_l163_163031


namespace three_brothers_pizza_slices_l163_163512

theorem three_brothers_pizza_slices :
  let large_pizza_slices := 14
  let small_pizza_slices := 8
  let num_brothers := 3
  let total_slices := small_pizza_slices + 2 * large_pizza_slices
  total_slices / num_brothers = 12 := by
  sorry

end three_brothers_pizza_slices_l163_163512


namespace possible_values_of_a_l163_163453

theorem possible_values_of_a :
  ∃ a b c : ℤ, 
    (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) ↔ 
    (a = 3 ∨ a = 7) :=
by
  sorry

end possible_values_of_a_l163_163453


namespace sin_thirty_deg_l163_163499

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l163_163499


namespace find_number_l163_163140

theorem find_number (x : ℝ) (h : 120 = 1.5 * x) : x = 80 :=
by
  sorry

end find_number_l163_163140


namespace correct_answer_l163_163330

theorem correct_answer (a b c : ℝ) : a - (b + c) = a - b - c :=
by sorry

end correct_answer_l163_163330


namespace jar_ratios_l163_163901

theorem jar_ratios (C_X C_Y : ℝ) 
  (h1 : 0 < C_X) 
  (h2 : 0 < C_Y)
  (h3 : (1/2) * C_X + (1/2) * C_Y = (3/4) * C_X) : 
  C_Y = (1/2) * C_X := 
sorry

end jar_ratios_l163_163901


namespace intersection_of_A_and_B_l163_163186

def A : Set ℝ := { x | -2 < x ∧ x < 2 }
def B : Set ℝ := { x | x ≤ 1 ∨ x ≥ 3 }

theorem intersection_of_A_and_B : 
  (A ∩ B) = { x : ℝ | -2 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_A_and_B_l163_163186


namespace circles_positional_relationship_l163_163742

theorem circles_positional_relationship
  (r1 r2 d : ℝ)
  (h1 : r1 = 1)
  (h2 : r2 = 5)
  (h3 : d = 3) :
  d < r2 - r1 := 
by
  sorry

end circles_positional_relationship_l163_163742


namespace Christine_distance_went_l163_163945

-- Definitions from conditions
def Speed : ℝ := 20 -- miles per hour
def Time : ℝ := 4  -- hours

-- Statement of the problem
def Distance_went : ℝ := Speed * Time

-- The theorem we need to prove
theorem Christine_distance_went : Distance_went = 80 :=
by
  sorry

end Christine_distance_went_l163_163945


namespace tim_income_percent_less_than_juan_l163_163206

theorem tim_income_percent_less_than_juan (T M J : ℝ) (h1 : M = 1.5 * T) (h2 : M = 0.9 * J) :
  (J - T) / J = 0.4 :=
by
  sorry

end tim_income_percent_less_than_juan_l163_163206


namespace not_divisible_by_4_8_16_32_l163_163070

def x := 80 + 112 + 144 + 176 + 304 + 368 + 3248 + 17

theorem not_divisible_by_4_8_16_32 : 
  ¬ (x % 4 = 0) ∧ ¬ (x % 8 = 0) ∧ ¬ (x % 16 = 0) ∧ ¬ (x % 32 = 0) := 
by 
  sorry

end not_divisible_by_4_8_16_32_l163_163070


namespace total_votes_cast_l163_163796

theorem total_votes_cast (V : ℝ) (h1 : ∃ x : ℝ, x = 0.31 * V) (h2 : ∃ y : ℝ, y = x + 2451) :
  V = 6450 :=
by
  sorry

end total_votes_cast_l163_163796


namespace total_vegetables_l163_163559

-- Define the initial conditions
def potatoes : Nat := 560
def cucumbers : Nat := potatoes - 132
def tomatoes : Nat := 3 * cucumbers
def peppers : Nat := tomatoes / 2
def carrots : Nat := cucumbers + tomatoes

-- State the theorem to prove the total number of vegetables
theorem total_vegetables :
  560 + (560 - 132) + (3 * (560 - 132)) + ((3 * (560 - 132)) / 2) + ((560 - 132) + (3 * (560 - 132))) = 4626 := by
  sorry

end total_vegetables_l163_163559


namespace chocolates_eaten_by_robert_l163_163814

theorem chocolates_eaten_by_robert (nickel_ate : ℕ) (robert_ate_more : ℕ) (H1 : nickel_ate = 3) (H2 : robert_ate_more = 4) :
  nickel_ate + robert_ate_more = 7 :=
by {
  sorry
}

end chocolates_eaten_by_robert_l163_163814


namespace max_voters_is_five_l163_163131

noncomputable def max_voters_after_T (x : ℕ) : ℕ :=
if h : 0 ≤ (x - 11) then x - 11 else 0

theorem max_voters_is_five (x : ℕ) (hx : 0 ≤ x ∧ x ≤ 10) :
  max_voters_after_T x = 5 :=
by
  sorry

end max_voters_is_five_l163_163131


namespace abc_plus_2_gt_a_plus_b_plus_c_l163_163513

theorem abc_plus_2_gt_a_plus_b_plus_c (a b c : ℝ) (ha : -1 < a) (ha' : a < 1) (hb : -1 < b) (hb' : b < 1) (hc : -1 < c) (hc' : c < 1) :
  a * b * c + 2 > a + b + c :=
sorry

end abc_plus_2_gt_a_plus_b_plus_c_l163_163513


namespace infinitely_many_n_divisible_by_2018_l163_163105

theorem infinitely_many_n_divisible_by_2018 :
  ∃ᶠ n : ℕ in Filter.atTop, 2018 ∣ (1 + 2^n + 3^n + 4^n) :=
sorry

end infinitely_many_n_divisible_by_2018_l163_163105


namespace purely_imaginary_complex_is_two_l163_163504

theorem purely_imaginary_complex_is_two
  (a : ℝ)
  (h_imag : (a^2 - 3 * a + 2) + (a - 1) * I = (a - 1) * I) :
  a = 2 := by
  sorry

end purely_imaginary_complex_is_two_l163_163504


namespace red_grapes_count_l163_163309

-- Definitions of variables and conditions
variables (G R Ra B P : ℕ)
variables (cond1 : R = 3 * G + 7)
variables (cond2 : Ra = G - 5)
variables (cond3 : B = 4 * Ra)
variables (cond4 : P = (1 / 2) * B + 5)
variables (cond5 : G + R + Ra + B + P = 350)

-- Theorem statement
theorem red_grapes_count : R = 100 :=
by sorry

end red_grapes_count_l163_163309


namespace prove_x3_y3_le_2_l163_163189

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

axiom positive_x : 0 < x
axiom positive_y : 0 < y
axiom condition : x^3 + y^4 ≤ x^2 + y^3

theorem prove_x3_y3_le_2 : x^3 + y^3 ≤ 2 := 
by
  sorry

end prove_x3_y3_le_2_l163_163189


namespace smallest_n_l163_163880

theorem smallest_n (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x ∣ y^3) (h2 : y ∣ z^3) (h3 : z ∣ x^3)
  (h4 : x * y * z ∣ (x + y + z)^n) : n = 13 :=
sorry

end smallest_n_l163_163880


namespace find_number_thought_of_l163_163313

theorem find_number_thought_of :
  ∃ x : ℝ, (6 * x^2 - 10) / 3 + 15 = 95 ∧ x = 5 * Real.sqrt 15 / 3 :=
by
  sorry

end find_number_thought_of_l163_163313


namespace ratio_of_second_to_first_l163_163825

noncomputable def building_heights (H1 H2 H3 : ℝ) : Prop :=
  H1 = 600 ∧ H3 = 3 * (H1 + H2) ∧ H1 + H2 + H3 = 7200

theorem ratio_of_second_to_first (H1 H2 H3 : ℝ) (h : building_heights H1 H2 H3) :
  H1 ≠ 0 → (H2 / H1 = 2) :=
by
  unfold building_heights at h
  rcases h with ⟨h1, h3, h_total⟩
  sorry -- Steps of solving are skipped

end ratio_of_second_to_first_l163_163825


namespace tan_addition_identity_l163_163617

theorem tan_addition_identity 
  (tan_30 : Real := Real.tan (Real.pi / 6))
  (tan_15 : Real := 2 - Real.sqrt 3) : 
  tan_15 + tan_30 + tan_15 * tan_30 = 1 := 
by
  have h1 : tan_30 = Real.sqrt 3 / 3 := sorry
  have h2 : tan_15 = 2 - Real.sqrt 3 := sorry
  sorry

end tan_addition_identity_l163_163617


namespace three_digit_integers_product_36_l163_163286

theorem three_digit_integers_product_36 : 
  ∃ (num_digits : ℕ), num_digits = 21 ∧ 
    ∀ (a b c : ℕ), 
      (1 ≤ a ∧ a ≤ 9) ∧ 
      (1 ≤ b ∧ b ≤ 9) ∧ 
      (1 ≤ c ∧ c ≤ 9) ∧ 
      (a * b * c = 36) → 
      num_digits = 21 :=
sorry

end three_digit_integers_product_36_l163_163286


namespace highlighter_total_l163_163636

theorem highlighter_total 
  (pink_highlighters : ℕ)
  (yellow_highlighters : ℕ)
  (blue_highlighters : ℕ)
  (h_pink : pink_highlighters = 4)
  (h_yellow : yellow_highlighters = 2)
  (h_blue : blue_highlighters = 5) :
  pink_highlighters + yellow_highlighters + blue_highlighters = 11 :=
by
  sorry

end highlighter_total_l163_163636


namespace find_ending_number_l163_163046

def ending_number (n : ℕ) : Prop :=
  18 < n ∧ n % 7 = 0 ∧ ((21 + n) / 2 : ℝ) = 38.5

theorem find_ending_number : ending_number 56 :=
by
  unfold ending_number
  sorry

end find_ending_number_l163_163046


namespace middle_part_l163_163438

theorem middle_part (x : ℝ) (h : 2 * x + (2 / 3) * x + (2 / 9) * x = 120) : 
  (2 / 3) * x = 27.6 :=
by
  -- Assuming the given conditions
  sorry

end middle_part_l163_163438


namespace each_student_gets_8_pieces_l163_163296

-- Define the number of pieces of candy
def candy : Nat := 344

-- Define the number of students
def students : Nat := 43

-- Define the number of pieces each student gets, which we need to prove
def pieces_per_student : Nat := candy / students

-- The proof problem statement
theorem each_student_gets_8_pieces : pieces_per_student = 8 :=
by
  -- This proof content is omitted as per instructions
  sorry

end each_student_gets_8_pieces_l163_163296


namespace find_f3_l163_163492

theorem find_f3 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, x * f y = y * f x) (h2 : f 15 = 20) : f 3 = 4 := 
  sorry

end find_f3_l163_163492


namespace triangle_construction_possible_l163_163890

theorem triangle_construction_possible (r l_alpha k_alpha : ℝ) (h1 : r > 0) (h2 : l_alpha > 0) (h3 : k_alpha > 0) :
  l_alpha^2 < (4 * k_alpha^2 * r^2) / (k_alpha^2 + r^2) :=
sorry

end triangle_construction_possible_l163_163890


namespace quad_func_minimum_l163_163342

def quad_func (x : ℝ) : ℝ := x^2 - 8 * x + 5

theorem quad_func_minimum : ∀ x : ℝ, quad_func x ≥ -11 ∧ quad_func 4 = -11 :=
by
  sorry

end quad_func_minimum_l163_163342


namespace linda_five_dollar_bills_l163_163462

theorem linda_five_dollar_bills (x y : ℕ) (h1 : x + y = 12) (h2 : 5 * x + 10 * y = 80) : x = 8 :=
by
  sorry

end linda_five_dollar_bills_l163_163462


namespace arithmetic_sequence_a_value_l163_163042

theorem arithmetic_sequence_a_value :
  ∀ (a : ℤ), (-7) - a = a - 1 → a = -3 :=
by
  intro a
  intro h
  sorry

end arithmetic_sequence_a_value_l163_163042


namespace camille_total_birds_l163_163096

theorem camille_total_birds :
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  let finches := robins / 2
  cardinals + robins + blue_jays + sparrows + pigeons + finches = 55 :=
by
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  let finches := robins / 2
  show cardinals + robins + blue_jays + sparrows + pigeons + finches = 55
  sorry

end camille_total_birds_l163_163096


namespace proposition_B_correct_l163_163851

theorem proposition_B_correct (a b c : ℝ) (hc : c ≠ 0) : ac^2 > b * c^2 → a > b := sorry

end proposition_B_correct_l163_163851


namespace interest_rate_proof_l163_163843

variable (P : ℝ) (n : ℕ) (CI SI : ℝ → ℝ → ℕ → ℝ) (diff : ℝ → ℝ → ℝ)

def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n
def simple_interest (P r : ℝ) (n : ℕ) : ℝ := P * r * n

theorem interest_rate_proof (r : ℝ) :
  diff (compound_interest 5400 r 2) (simple_interest 5400 r 2) = 216 → r = 0.2 :=
by sorry

end interest_rate_proof_l163_163843


namespace internet_bill_proof_l163_163965

variable (current_bill : ℕ)
variable (internet_bill_30Mbps : ℕ)
variable (annual_savings : ℕ)
variable (additional_amount_20Mbps : ℕ)

theorem internet_bill_proof
  (h1 : current_bill = 20)
  (h2 : internet_bill_30Mbps = 40)
  (h3 : annual_savings = 120)
  (monthly_savings : ℕ := annual_savings / 12)
  (h4 : monthly_savings = 10)
  (h5 : internet_bill_30Mbps - (current_bill + additional_amount_20Mbps) = 10) :
  additional_amount_20Mbps = 10 :=
by
  sorry

end internet_bill_proof_l163_163965


namespace balloon_arrangement_count_l163_163740

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l163_163740


namespace train_passes_tree_in_20_seconds_l163_163994

def train_passing_time 
  (length_of_train : ℕ)
  (speed_kmh : ℕ)
  (conversion_factor : ℚ) : ℚ :=
  length_of_train / (speed_kmh * conversion_factor)

theorem train_passes_tree_in_20_seconds 
  (length_of_train : ℕ := 350)
  (speed_kmh : ℕ := 63)
  (conversion_factor : ℚ := 1000 / 3600) : 
  train_passing_time length_of_train speed_kmh conversion_factor = 20 :=
  sorry

end train_passes_tree_in_20_seconds_l163_163994


namespace F_transformed_l163_163229

-- Define the coordinates of point F
def F : ℝ × ℝ := (1, 0)

-- Reflection over the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Reflection over the y-axis
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

-- Reflection over the line y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

-- Point F after all transformations
def F_final : ℝ × ℝ :=
  reflect_y_eq_x (reflect_y (reflect_x F))

-- Statement to prove
theorem F_transformed : F_final = (0, -1) :=
  sorry

end F_transformed_l163_163229


namespace solving_linear_equations_problems_l163_163436

def num_total_math_problems : ℕ := 140
def percent_algebra_problems : ℝ := 0.40
def fraction_solving_linear_equations : ℝ := 0.50

theorem solving_linear_equations_problems :
  let num_algebra_problems := percent_algebra_problems * num_total_math_problems
  let num_solving_linear_equations := fraction_solving_linear_equations * num_algebra_problems
  num_solving_linear_equations = 28 :=
by
  sorry

end solving_linear_equations_problems_l163_163436


namespace initial_amount_l163_163798

theorem initial_amount (cost_bread cost_butter cost_juice total_remain total_amount : ℕ) :
  cost_bread = 2 →
  cost_butter = 3 →
  cost_juice = 2 * cost_bread →
  total_remain = 6 →
  total_amount = cost_bread + cost_butter + cost_juice + total_remain →
  total_amount = 15 := by
  intros h_bread h_butter h_juice h_remain h_total
  sorry

end initial_amount_l163_163798


namespace cost_difference_l163_163245

/-- The selling price and cost of pants -/
def selling_price : ℕ := 34
def store_cost : ℕ := 26

/-- The proof that the store paid 8 dollars less than the selling price -/
theorem cost_difference : selling_price - store_cost = 8 := by
  sorry

end cost_difference_l163_163245


namespace b2009_value_l163_163580

noncomputable def b (n : ℕ) : ℝ := sorry

axiom b_recursion (n : ℕ) (hn : 2 ≤ n) : b n = b (n - 1) * b (n + 1)

axiom b1_value : b 1 = 2 + Real.sqrt 3
axiom b1776_value : b 1776 = 10 + Real.sqrt 3

theorem b2009_value : b 2009 = -4 + 8 * Real.sqrt 3 := 
by sorry

end b2009_value_l163_163580


namespace cakes_baker_made_initially_l163_163257

theorem cakes_baker_made_initially (x : ℕ) (h1 : x - 75 + 76 = 111) : x = 110 :=
by
  sorry

end cakes_baker_made_initially_l163_163257


namespace pencils_total_l163_163759

theorem pencils_total (original_pencils : ℕ) (added_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : original_pencils = 41) 
  (h2 : added_pencils = 30) 
  (h3 : total_pencils = original_pencils + added_pencils) : 
  total_pencils = 71 := 
by
  sorry

end pencils_total_l163_163759


namespace Alyssa_total_spent_l163_163578

/-- Definition of fruit costs -/
def cost_grapes : ℝ := 12.08
def cost_cherries : ℝ := 9.85
def cost_mangoes : ℝ := 7.50
def cost_pineapple : ℝ := 4.25
def cost_starfruit : ℝ := 3.98

/-- Definition of tax and discount -/
def tax_rate : ℝ := 0.10
def discount : ℝ := 3.00

/-- Calculation of the total cost Alyssa spent after applying tax and discount -/
def total_spent : ℝ := 
  let total_cost_before_tax := cost_grapes + cost_cherries + cost_mangoes + cost_pineapple + cost_starfruit
  let tax := tax_rate * total_cost_before_tax
  let total_cost_with_tax := total_cost_before_tax + tax
  total_cost_with_tax - discount

/-- Statement that needs to be proven -/
theorem Alyssa_total_spent : total_spent = 38.43 := by 
  sorry

end Alyssa_total_spent_l163_163578


namespace angle_A_in_triangle_l163_163879

theorem angle_A_in_triangle (a b c : ℝ) (h : a^2 = b^2 + b * c + c^2) : A = 120 :=
sorry

end angle_A_in_triangle_l163_163879


namespace second_number_is_255_l163_163570

theorem second_number_is_255 (x : ℝ) (n : ℝ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 90) 
  (h2 : (128 + n + 511 + 1023 + x) / 5 = 423) : 
  n = 255 :=
sorry

end second_number_is_255_l163_163570


namespace find_value_added_l163_163639

theorem find_value_added :
  ∀ (n x : ℤ), (2 * n + x = 8 * n - 4) → (n = 4) → (x = 20) :=
by
  intros n x h1 h2
  sorry

end find_value_added_l163_163639


namespace correct_arrangements_l163_163152

open Finset Nat

-- Definitions for combinations and powers
def comb (n k : ℕ) : ℕ := choose n k

-- The number of computer rooms
def num_computer_rooms : ℕ := 6

-- The number of arrangements to open at least 2 out of 6 computer rooms
def arrangement_count1 : ℕ := 2^num_computer_rooms - (comb num_computer_rooms 0 + comb num_computer_rooms 1)

-- Another calculation for the number of arrangements to open at least 2 out of 6 computer rooms
def arrangement_count2 : ℕ := comb num_computer_rooms 2 + comb num_computer_rooms 3 + comb num_computer_rooms 4 + 
                               comb num_computer_rooms 5 + comb num_computer_rooms 6

theorem correct_arrangements :
  arrangement_count1 = arrangement_count2 := 
  sorry

end correct_arrangements_l163_163152


namespace sum_of_integers_l163_163190

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 168) : x + y = 32 :=
by
  sorry

end sum_of_integers_l163_163190


namespace part1_part2_l163_163725

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) * Real.exp x

theorem part1 (a x : ℝ) (h : a > 0) : f a x + a / Real.exp 1 > 0 := by
  sorry

theorem part2 (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : f (-1/2) x1 = f (-1/2) x2) : x1 + x2 < 2 := by
  sorry

end part1_part2_l163_163725


namespace tank_ratio_l163_163969

variable (C D : ℝ)
axiom h1 : 3 / 4 * C = 2 / 5 * D

theorem tank_ratio : C / D = 8 / 15 := by
  sorry

end tank_ratio_l163_163969


namespace expression_nonnegative_l163_163916

theorem expression_nonnegative (x : ℝ) : 
  0 ≤ x → x < 3 → 0 ≤ (x - 12 * x^2 + 36 * x^3) / (9 - x^3) :=
  sorry

end expression_nonnegative_l163_163916


namespace arithmetic_sequence_sum_range_l163_163088

noncomputable def S_n (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_range 
  (a d : ℝ)
  (h1 : 1 ≤ a + 3 * d) 
  (h2 : a + 3 * d ≤ 4)
  (h3 : 2 ≤ a + 4 * d)
  (h4 : a + 4 * d ≤ 3) 
  : 0 ≤ S_n a d 6 ∧ S_n a d 6 ≤ 30 := 
sorry

end arithmetic_sequence_sum_range_l163_163088


namespace total_lobster_pounds_l163_163259

theorem total_lobster_pounds
  (combined_other_harbors : ℕ)
  (hooper_bay : ℕ)
  (H1 : combined_other_harbors = 160)
  (H2 : hooper_bay = 2 * combined_other_harbors) :
  combined_other_harbors + hooper_bay = 480 :=
by
  -- proof goes here
  sorry

end total_lobster_pounds_l163_163259


namespace find_roots_of_polynomial_l163_163233

theorem find_roots_of_polynomial :
  ∀ x : ℝ, (3 * x ^ 4 - x ^ 3 - 8 * x ^ 2 - x + 3 = 0) →
    (x = 2 ∨ x = 1/3 ∨ x = -1) :=
by
  intros x h
  sorry

end find_roots_of_polynomial_l163_163233


namespace int_power_sum_is_integer_l163_163081

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem int_power_sum_is_integer {x : ℝ} (h : is_integer (x + 1/x)) (n : ℤ) : is_integer (x^n + 1/x^n) :=
by
  sorry

end int_power_sum_is_integer_l163_163081


namespace calculate_error_percentage_l163_163917

theorem calculate_error_percentage (x : ℝ) (hx : x > 0) (x_eq_9 : x = 9) :
  (abs ((x * (x - 8)) / (8 * x)) * 100) = 12.5 := by
  sorry

end calculate_error_percentage_l163_163917


namespace Mrs_Martin_pays_32_l163_163483

def kiddie_scoop_cost : ℕ := 3
def regular_scoop_cost : ℕ := 4
def double_scoop_cost : ℕ := 6

def num_regular_scoops : ℕ := 2
def num_kiddie_scoops : ℕ := 2
def num_double_scoops : ℕ := 3

def total_cost : ℕ := 
  (num_regular_scoops * regular_scoop_cost) + 
  (num_kiddie_scoops * kiddie_scoop_cost) + 
  (num_double_scoops * double_scoop_cost)

theorem Mrs_Martin_pays_32 :
  total_cost = 32 :=
by
  sorry

end Mrs_Martin_pays_32_l163_163483


namespace statement_C_correct_l163_163238

theorem statement_C_correct (a b : ℝ) (h1 : a < b) (h2 : a * b ≠ 0) : (1 / a) > (1 / b) :=
sorry

end statement_C_correct_l163_163238


namespace true_proposition_among_ABCD_l163_163620

theorem true_proposition_among_ABCD : 
  (∀ x : ℝ, x^2 < x + 1) = false ∧
  (∀ x : ℝ, x^2 ≥ x + 1) = false ∧
  (∃ x : ℝ, ∀ y : ℝ, x * y^2 ≠ y^2) = true ∧
  (∀ x : ℝ, ∃ y : ℝ, x > y^2) = false :=
by 
  sorry

end true_proposition_among_ABCD_l163_163620


namespace anne_distance_l163_163749

-- Definitions based on conditions
def Time : ℕ := 5
def Speed : ℕ := 4
def Distance : ℕ := Speed * Time

-- Proof statement
theorem anne_distance : Distance = 20 := by
  sorry

end anne_distance_l163_163749


namespace ball_beyond_hole_l163_163383

theorem ball_beyond_hole
  (first_turn_distance : ℕ)
  (second_turn_distance : ℕ)
  (total_distance_to_hole : ℕ) :
  first_turn_distance = 180 →
  second_turn_distance = first_turn_distance / 2 →
  total_distance_to_hole = 250 →
  second_turn_distance - (total_distance_to_hole - first_turn_distance) = 20 :=
by
  intros
  -- Proof omitted
  sorry

end ball_beyond_hole_l163_163383


namespace megan_initial_markers_l163_163908

theorem megan_initial_markers (gave : ℕ) (total : ℕ) (initial : ℕ) 
  (h1 : gave = 109) 
  (h2 : total = 326) 
  (h3 : initial + gave = total) : 
  initial = 217 := 
by 
  sorry

end megan_initial_markers_l163_163908


namespace income_after_selling_more_l163_163734

theorem income_after_selling_more (x y : ℝ)
  (h1 : 26 * x + 14 * y = 264) 
  : 39 * x + 21 * y = 396 := 
by 
  sorry

end income_after_selling_more_l163_163734


namespace TrigPowerEqualsOne_l163_163583

theorem TrigPowerEqualsOne : ((Real.cos (160 * Real.pi / 180) + Real.sin (160 * Real.pi / 180) * Complex.I)^36 = 1) :=
by
  sorry

end TrigPowerEqualsOne_l163_163583


namespace network_connections_l163_163469

theorem network_connections (n m : ℕ) (hn : n = 30) (hm : m = 5) 
(h_total_conn : (n * 4) / 2 = 60) : 
60 + m = 65 :=
by
  sorry

end network_connections_l163_163469


namespace min_value_a1_plus_a7_l163_163977

theorem min_value_a1_plus_a7 (a : ℕ → ℝ) (r : ℝ) 
  (h1 : ∀ n, a n > 0) 
  (h2 : ∀ n, a (n+1) = a n * r) 
  (h3 : a 3 * a 5 = 64) : 
  a 1 + a 7 ≥ 16 := 
sorry

end min_value_a1_plus_a7_l163_163977


namespace rs_division_l163_163480

theorem rs_division (a b c : ℝ) 
  (h1 : a = 1 / 2 * b)
  (h2 : b = 1 / 2 * c)
  (h3 : a + b + c = 700) : 
  c = 400 :=
sorry

end rs_division_l163_163480


namespace ticket_price_values_l163_163964

theorem ticket_price_values : 
  ∃ (x_values : Finset ℕ), 
    (∀ x ∈ x_values, x ∣ 60 ∧ x ∣ 80) ∧ 
    x_values.card = 6 :=
by
  sorry

end ticket_price_values_l163_163964


namespace listK_consecutive_integers_count_l163_163184

-- Given conditions
def listK := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] -- A list K consisting of consecutive integers
def leastInt : Int := -5 -- The least integer in list K
def rangePosInt : Nat := 5 -- The range of the positive integers in list K

-- The theorem to prove
theorem listK_consecutive_integers_count : listK.length = 11 := by
  -- skipping the proof
  sorry

end listK_consecutive_integers_count_l163_163184


namespace right_triangle_segments_l163_163028

open Real

theorem right_triangle_segments 
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (h_ab : a > b)
  (P Q : ℝ × ℝ) (P_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (Q_on_ellipse : Q.1^2 / a^2 + Q.2^2 / b^2 = 1)
  (Q_in_first_quad : Q.1 > 0 ∧ Q.2 > 0)
  (OQ_parallel_AP : ∃ k : ℝ, Q.1 = k * P.1 ∧ Q.2 = k * P.2)
  (M : ℝ × ℝ) (M_midpoint : M = ((P.1 + 0) / 2, (P.2 + 0) / 2))
  (R : ℝ × ℝ) (R_on_ellipse : R.1^2 / a^2 + R.2^2 / b^2 = 1)
  (OM_intersects_R : ∃ k : ℝ, R = (k * M.1, k * M.2))
: dist (0,0) Q ≠ 0 →
  dist (0,0) R ≠ 0 →
  dist (Q, R) ≠ 0 →
  dist (0,0) Q ^ 2 + dist (0,0) R ^ 2 = dist ((-a), (b)) ((a), (b)) ^ 2 :=
by
  sorry

end right_triangle_segments_l163_163028


namespace period_tan_half_l163_163041

noncomputable def period_of_tan_half : Real :=
  2 * Real.pi

theorem period_tan_half (f : Real → Real) (h : ∀ x, f x = Real.tan (x / 2)) :
  ∀ x, f (x + period_of_tan_half) = f x := 
by 
  sorry

end period_tan_half_l163_163041


namespace unique_a_b_l163_163322

-- Define the properties of the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * a * x + b else 7 - 2 * x

-- The function satisfies f(f(x)) = x for all x in its domain
theorem unique_a_b (a b : ℝ) (h : ∀ x : ℝ, f a b (f a b x) = x) : a + b = 13 / 4 :=
sorry

end unique_a_b_l163_163322


namespace lcm_18_30_l163_163812

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l163_163812


namespace geom_seq_general_formula_sum_first_n_terms_formula_l163_163793

namespace GeometricArithmeticSequences

def geom_seq_general (a_n : ℕ → ℝ) (n : ℕ) : Prop :=
  a_n 1 = 1 ∧ (2 * a_n 3 = a_n 2) → a_n n = 1 / (2 ^ (n - 1))

def sum_first_n_terms (a_n b_n : ℕ → ℝ) (S_n T_n : ℕ → ℝ) (n : ℕ) : Prop :=
  b_n 1 = 2 ∧ S_n 3 = b_n 2 + 6 → 
  T_n n = 6 - (n + 3) / (2 ^ (n - 1))

theorem geom_seq_general_formula :
  ∀ a_n : ℕ → ℝ, ∀ n : ℕ, geom_seq_general a_n n :=
by sorry

theorem sum_first_n_terms_formula :
  ∀ a_n b_n : ℕ → ℝ, ∀ S_n T_n : ℕ → ℝ, ∀ n : ℕ, sum_first_n_terms a_n b_n S_n T_n n :=
by sorry

end GeometricArithmeticSequences

end geom_seq_general_formula_sum_first_n_terms_formula_l163_163793


namespace john_dimes_l163_163785

theorem john_dimes :
  ∀ (d : ℕ), 
  (4 * 25 + d * 10 + 5) = 135 → (5: ℕ) + (d: ℕ) * 10 + 4 = 4 + 131 + 3*d → d = 3 :=
by
  sorry

end john_dimes_l163_163785


namespace total_selling_price_correct_l163_163446

def meters_sold : ℕ := 85
def cost_price_per_meter : ℕ := 80
def profit_per_meter : ℕ := 25

def selling_price_per_meter : ℕ :=
  cost_price_per_meter + profit_per_meter

def total_selling_price : ℕ :=
  selling_price_per_meter * meters_sold

theorem total_selling_price_correct :
  total_selling_price = 8925 := by
  sorry

end total_selling_price_correct_l163_163446


namespace range_of_x_plus_y_l163_163681

open Real

theorem range_of_x_plus_y (x y : ℝ) (h : x - sqrt (x + 1) = sqrt (y + 1) - y) :
  -sqrt 5 + 1 ≤ x + y ∧ x + y ≤ sqrt 5 + 1 :=
by sorry

end range_of_x_plus_y_l163_163681


namespace fraction_power_multiplication_l163_163882

theorem fraction_power_multiplication :
  ( (8 / 9)^3 * (5 / 3)^3 ) = (64000 / 19683) :=
by
  sorry

end fraction_power_multiplication_l163_163882


namespace certain_event_l163_163711

theorem certain_event (a : ℝ) : a^2 ≥ 0 := 
sorry

end certain_event_l163_163711


namespace union_of_A_and_B_l163_163308

-- Definitions of sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 4}
def B : Set ℝ := {x | -3 ≤ x ∧ x < 1}

-- The theorem we aim to prove
theorem union_of_A_and_B : A ∪ B = { x | -3 ≤ x ∧ x ≤ 4 } :=
sorry

end union_of_A_and_B_l163_163308


namespace value_of_a_l163_163314

noncomputable def A : Set ℝ := {x | x^2 - x - 2 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 5}

theorem value_of_a (a : ℝ) (h : A ⊆ B a) : -3 ≤ a ∧ a ≤ -1 :=
by
  sorry

end value_of_a_l163_163314


namespace inequality_holds_for_all_real_numbers_l163_163677

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l163_163677


namespace probability_of_victory_l163_163180

theorem probability_of_victory (p_A p_B : ℝ) (h_A : p_A = 0.3) (h_B : p_B = 0.6) (independent : true) :
  p_A * p_B = 0.18 :=
by
  -- placeholder for proof
  sorry

end probability_of_victory_l163_163180


namespace ab_bc_ca_plus_one_pos_l163_163072

variable (a b c : ℝ)
variable (h₁ : |a| < 1)
variable (h₂ : |b| < 1)
variable (h₃ : |c| < 1)

theorem ab_bc_ca_plus_one_pos :
  ab + bc + ca + 1 > 0 := sorry

end ab_bc_ca_plus_one_pos_l163_163072


namespace bus_speed_excluding_stoppages_l163_163400

theorem bus_speed_excluding_stoppages 
  (V : ℝ) -- Denote the average speed excluding stoppages as V
  (h1 : 30 / 1 = 30) -- condition 1: average speed including stoppages is 30 km/hr
  (h2 : 1 / 2 = 0.5) -- condition 2: The bus is moving for 0.5 hours per hour due to 30 min stoppage
  (h3 : V = 2 * 30) -- from the condition that the bus must cover the distance in half the time
  : V = 60 :=
by {
  sorry -- proof is not required
}

end bus_speed_excluding_stoppages_l163_163400


namespace score_difference_l163_163310

theorem score_difference (chuck_score red_score : ℕ) (h1 : chuck_score = 95) (h2 : red_score = 76) : chuck_score - red_score = 19 := by
  sorry

end score_difference_l163_163310


namespace axis_of_symmetry_l163_163938

-- Define the given parabolic function
def parabola (x : ℝ) : ℝ := (2 - x) * x

-- Define the axis of symmetry property for the given parabola
theorem axis_of_symmetry : ∀ x : ℝ, ((2 - x) * x) = -((x - 1)^2) + 1 → (∃ x_sym : ℝ, x_sym = 1) :=
by
  sorry

end axis_of_symmetry_l163_163938


namespace maintenance_check_days_l163_163596

theorem maintenance_check_days (x : ℝ) (hx : x + 0.20 * x = 60) : x = 50 :=
by
  -- this is where the proof would go
  sorry

end maintenance_check_days_l163_163596


namespace milly_folds_count_l163_163834

theorem milly_folds_count (mixing_time baking_time total_minutes fold_time rest_time : ℕ) 
  (h : total_minutes = 360)
  (h_mixing_time : mixing_time = 10)
  (h_baking_time : baking_time = 30)
  (h_fold_time : fold_time = 5)
  (h_rest_time : rest_time = 75) : 
  (total_minutes - (mixing_time + baking_time)) / (fold_time + rest_time) = 4 := 
by
  sorry

end milly_folds_count_l163_163834


namespace largest_side_l163_163058

-- Definitions of conditions from part (a)
def perimeter_eq (l w : ℝ) : Prop := 2 * l + 2 * w = 240
def area_eq (l w : ℝ) : Prop := l * w = 2880

-- The main proof statement
theorem largest_side (l w : ℝ) (h1 : perimeter_eq l w) (h2 : area_eq l w) : l = 72 ∨ w = 72 :=
by
  sorry

end largest_side_l163_163058


namespace average_weight_decrease_l163_163172

theorem average_weight_decrease 
  (num_persons : ℕ)
  (avg_weight_initial : ℕ)
  (new_person_weight : ℕ)
  (new_avg_weight : ℚ)
  (weight_decrease : ℚ)
  (h1 : num_persons = 20)
  (h2 : avg_weight_initial = 60)
  (h3 : new_person_weight = 45)
  (h4 : new_avg_weight = (1200 + 45) / 21) : 
  weight_decrease = avg_weight_initial - new_avg_weight :=
by
  sorry

end average_weight_decrease_l163_163172


namespace jackson_souvenirs_total_l163_163766

def jacksons_collections := 
  let hermit_crabs := 120
  let spiral_shells_per_hermit_crab := 8
  let starfish_per_spiral_shell := 5
  let sand_dollars_per_starfish := 3
  let coral_structures_per_sand_dollars := 4
  let spiral_shells := hermit_crabs * spiral_shells_per_hermit_crab
  let starfish := spiral_shells * starfish_per_spiral_shell
  let sand_dollars := starfish * sand_dollars_per_starfish
  let coral_structures := sand_dollars / coral_structures_per_sand_dollars
  hermit_crabs + spiral_shells + starfish + sand_dollars + coral_structures

theorem jackson_souvenirs_total : jacksons_collections = 22880 := by sorry

end jackson_souvenirs_total_l163_163766


namespace blocks_needed_for_wall_l163_163641

theorem blocks_needed_for_wall (length height : ℕ) (block_heights block_lengths : List ℕ)
  (staggered : Bool) (even_ends : Bool)
  (h_length : length = 120)
  (h_height : height = 8)
  (h_block_heights : block_heights = [1])
  (h_block_lengths : block_lengths = [1, 2, 3])
  (h_staggered : staggered = true)
  (h_even_ends : even_ends = true) :
  ∃ (n : ℕ), n = 404 := 
sorry

end blocks_needed_for_wall_l163_163641


namespace blake_change_l163_163787

def cost_oranges : ℕ := 40
def cost_apples : ℕ := 50
def cost_mangoes : ℕ := 60
def initial_money : ℕ := 300

def total_cost : ℕ := cost_oranges + cost_apples + cost_mangoes
def change : ℕ := initial_money - total_cost

theorem blake_change : change = 150 := by
  sorry

end blake_change_l163_163787


namespace starting_lineups_possible_l163_163085

open Nat

theorem starting_lineups_possible (total_players : ℕ) (all_stars : ℕ) (lineup_size : ℕ) 
  (fixed_in_lineup : ℕ) (choose_size : ℕ) 
  (h_fixed : fixed_in_lineup = all_stars)
  (h_remaining : total_players - fixed_in_lineup = choose_size)
  (h_lineup : lineup_size = all_stars + choose_size) :
  (Nat.choose choose_size 3 = 220) :=
by
  sorry

end starting_lineups_possible_l163_163085


namespace jerome_family_members_l163_163864

-- Define the conditions of the problem
variables (C F M T : ℕ)
variables (hC : C = 20) (hF : F = C / 2) (hT : T = 33)

-- Formulate the theorem to prove
theorem jerome_family_members :
  M = T - (C + F) :=
sorry

end jerome_family_members_l163_163864


namespace trigonometric_solution_l163_163167

theorem trigonometric_solution (x : Real) :
  (2 * Real.sin x * Real.cos (3 * Real.pi / 2 + x) 
  - 3 * Real.sin (Real.pi - x) * Real.cos x 
  + Real.sin (Real.pi / 2 + x) * Real.cos x = 0) ↔ 
  (∃ k : Int, x = Real.arctan ((3 + Real.sqrt 17) / -4) + k * Real.pi) ∨ 
  (∃ n : Int, x = Real.arctan ((3 - Real.sqrt 17) / -4) + n * Real.pi) :=
sorry

end trigonometric_solution_l163_163167


namespace slope_of_parallel_line_l163_163635

theorem slope_of_parallel_line (a b c : ℝ) (x y : ℝ) (h : 3 * x + 6 * y = -12):
  (∀ m : ℝ, (∀ (x y : ℝ), (3 * x + 6 * y = -12) → y = m * x + (-(12 / 6) / 6)) → m = -1/2) :=
sorry

end slope_of_parallel_line_l163_163635


namespace spacy_subsets_15_l163_163149

def spacy (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 2
  | 2     => 3
  | n + 3 => spacy n + spacy (n-2)

theorem spacy_subsets_15 : spacy 15 = 406 := 
  sorry

end spacy_subsets_15_l163_163149


namespace find_f_2_l163_163594

variable {f : ℕ → ℤ}

-- Assume the condition given in the problem
axiom h : ∀ x : ℕ, f (x + 1) = x^2 - 1

-- Prove that f(2) = 0
theorem find_f_2 : f 2 = 0 := 
sorry

end find_f_2_l163_163594


namespace sock_pairs_l163_163304

theorem sock_pairs (n : ℕ) (h : ((2 * n) * (2 * n - 1)) / 2 = 90) : n = 10 :=
sorry

end sock_pairs_l163_163304


namespace c_share_l163_163378

theorem c_share (x : ℕ) (a b c d : ℕ) 
  (h1: a = 5 * x)
  (h2: b = 3 * x)
  (h3: c = 2 * x)
  (h4: d = 3 * x)
  (h5: a = b + 1000): 
  c = 1000 := 
by 
  sorry

end c_share_l163_163378


namespace compute_fraction_l163_163368

theorem compute_fraction (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) : 
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end compute_fraction_l163_163368


namespace height_difference_is_9_l163_163682

-- Definitions of the height of Petronas Towers and Empire State Building.
def height_Petronas : ℕ := 452
def height_EmpireState : ℕ := 443

-- Definition stating the height difference.
def height_difference := height_Petronas - height_EmpireState

-- Proving the height difference is 9 meters.
theorem height_difference_is_9 : height_difference = 9 :=
by
  -- the proof goes here
  sorry

end height_difference_is_9_l163_163682


namespace cos_sub_eq_five_over_eight_l163_163931

theorem cos_sub_eq_five_over_eight (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5 / 8 := 
by sorry

end cos_sub_eq_five_over_eight_l163_163931


namespace pi_irrational_l163_163532

theorem pi_irrational :
  ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (π = a / b) :=
by
  sorry

end pi_irrational_l163_163532


namespace find_n_l163_163690

noncomputable def n (n : ℕ) : Prop :=
  lcm n 12 = 42 ∧ gcd n 12 = 6

theorem find_n (n : ℕ) (h : lcm n 12 = 42) (h1 : gcd n 12 = 6) : n = 21 :=
by sorry

end find_n_l163_163690


namespace exists_k_consecutive_squareful_numbers_l163_163845

-- Define what it means for a number to be squareful
def is_squareful (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 1 ∧ m * m ∣ n

-- State the theorem
theorem exists_k_consecutive_squareful_numbers (k : ℕ) : 
  ∃ (a : ℕ), ∀ i, i < k → is_squareful (a + i) :=
sorry

end exists_k_consecutive_squareful_numbers_l163_163845


namespace greatest_integer_solution_l163_163939

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 13 * n + 40 ≤ 0) : n ≤ 8 :=
sorry

end greatest_integer_solution_l163_163939


namespace root_power_sum_eq_l163_163974

open Real

theorem root_power_sum_eq :
  ∀ {a b c : ℝ},
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (a^3 - 3 * a + 1 = 0) → (b^3 - 3 * b + 1 = 0) → (c^3 - 3 * c + 1 = 0) →
  a^8 + b^8 + c^8 = 186 :=
by
  intros a b c h1 h2 h3 ha hb hc
  sorry

end root_power_sum_eq_l163_163974


namespace largest_divisor_69_86_l163_163955

theorem largest_divisor_69_86 (n : ℕ) (h₁ : 69 % n = 5) (h₂ : 86 % n = 6) : n = 16 := by
  sorry

end largest_divisor_69_86_l163_163955


namespace garden_perimeter_is_44_l163_163841

-- Define the original garden's side length given the area
noncomputable def original_side_length (A : ℕ) := Nat.sqrt A

-- Given condition: Area of the original garden is 49 square meters
def original_area := 49

-- Define the new side length after expanding each side by 4 meters
def new_side_length (original_side : ℕ) := original_side + 4

-- Define the perimeter of the new garden given the new side length
def new_perimeter (new_side : ℕ) := 4 * new_side

-- Proof statement: The perimeter of the new garden given the original area is 44 meters
theorem garden_perimeter_is_44 : new_perimeter (new_side_length (original_side_length original_area)) = 44 := by
  -- This is where the proof would go
  sorry

end garden_perimeter_is_44_l163_163841


namespace sum_of_interior_angles_pentagon_l163_163110

theorem sum_of_interior_angles_pentagon : (5 - 2) * 180 = 540 := by
  sorry

end sum_of_interior_angles_pentagon_l163_163110


namespace chess_club_boys_l163_163562

theorem chess_club_boys (G B : ℕ) 
  (h1 : G + B = 30)
  (h2 : (2 / 3) * G + (3 / 4) * B = 18) : B = 24 :=
by
  sorry

end chess_club_boys_l163_163562


namespace quadratic_form_completion_l163_163684

theorem quadratic_form_completion (b c : ℤ)
  (h : ∀ x:ℂ, x^2 + 520*x + 600 = (x+b)^2 + c) :
  c / b = -258 :=
by sorry

end quadratic_form_completion_l163_163684


namespace find_g_expression_l163_163730

theorem find_g_expression (f g : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f x = 2 * x + 3)
  (h2 : ∀ x : ℝ, g (x + 2) = f x) :
  ∀ x : ℝ, g x = 2 * x - 1 :=
by
  sorry

end find_g_expression_l163_163730


namespace remainder_9_plus_y_mod_31_l163_163836

theorem remainder_9_plus_y_mod_31 (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (9 + y) % 31 = 18 :=
sorry

end remainder_9_plus_y_mod_31_l163_163836


namespace ratio_prikya_ladonna_l163_163179

def total_cans : Nat := 85
def ladonna_cans : Nat := 25
def yoki_cans : Nat := 10
def prikya_cans : Nat := total_cans - ladonna_cans - yoki_cans

theorem ratio_prikya_ladonna : prikya_cans.toFloat / ladonna_cans.toFloat = 2 / 1 := 
by sorry

end ratio_prikya_ladonna_l163_163179


namespace minimum_value_proof_l163_163619

noncomputable def min_value (x y : ℝ) : ℝ :=
  (x^2 / (x + 2)) + (y^2 / (y + 1))

theorem minimum_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  min_value x y = 1 / 4 :=
  sorry

end minimum_value_proof_l163_163619


namespace equations_not_equivalent_l163_163903

theorem equations_not_equivalent :
  ∀ x : ℝ, (x + 7 + 10 / (2 * x - 1) = 8 - x + 10 / (2 * x - 1)) ↔ false :=
by
  intro x
  sorry

end equations_not_equivalent_l163_163903


namespace actual_distance_traveled_l163_163847

theorem actual_distance_traveled
  (D : ℝ) 
  (H : ∃ T : ℝ, D = 5 * T ∧ D + 20 = 15 * T) : 
  D = 10 :=
by
  sorry

end actual_distance_traveled_l163_163847


namespace total_students_l163_163050

-- Define the conditions
def chocolates_distributed (y z : ℕ) : ℕ :=
  y * y + z * z

-- Define the main theorem to be proved
theorem total_students (y z : ℕ) (h : z = y + 3) (chocolates_left: ℕ) (initial_chocolates: ℕ)
  (h_chocolates: chocolates_distributed y z = initial_chocolates - chocolates_left) : 
  y + z = 33 :=
by
  sorry

end total_students_l163_163050


namespace largest_number_among_four_l163_163399

theorem largest_number_among_four :
  let a := 0.965
  let b := 0.9687
  let c := 0.9618
  let d := 0.955
  max a (max b (max c d)) = b := 
sorry

end largest_number_among_four_l163_163399


namespace geometric_sequence_product_l163_163625

variable {a b c : ℝ}

theorem geometric_sequence_product (h : ∃ r : ℝ, r ≠ 0 ∧ -4 = c * r ∧ c = b * r ∧ b = a * r ∧ a = -1 * r) (hb : b < 0) : a * b * c = -8 :=
by
  sorry

end geometric_sequence_product_l163_163625


namespace greatest_area_difference_l163_163073

theorem greatest_area_difference :
  ∃ (l1 w1 l2 w2 : ℕ), 2 * l1 + 2 * w1 = 200 ∧ 2 * l2 + 2 * w2 = 200 ∧
  (l1 * w1 - l2 * w2 = 2401) :=
by
  sorry

end greatest_area_difference_l163_163073


namespace eric_has_correct_green_marbles_l163_163697

def total_marbles : ℕ := 20
def white_marbles : ℕ := 12
def blue_marbles : ℕ := 6
def green_marbles : ℕ := total_marbles - (white_marbles + blue_marbles)

theorem eric_has_correct_green_marbles : green_marbles = 2 :=
by
  sorry

end eric_has_correct_green_marbles_l163_163697


namespace trapezoid_height_l163_163966

theorem trapezoid_height (BC AD AB CD h : ℝ) (hBC : BC = 4) (hAD : AD = 25) (hAB : AB = 20) (hCD : CD = 13) :
  h = 12 :=
by
  sorry

end trapezoid_height_l163_163966


namespace baker_cake_count_l163_163224

theorem baker_cake_count :
  let initial_cakes := 62
  let additional_cakes := 149
  let sold_cakes := 144
  initial_cakes + additional_cakes - sold_cakes = 67 :=
by
  sorry

end baker_cake_count_l163_163224


namespace single_elimination_matches_l163_163150

theorem single_elimination_matches (players byes : ℕ)
  (h1 : players = 100)
  (h2 : byes = 28) :
  (players - 1) = 99 :=
by
  -- The proof would go here if it were needed
  sorry

end single_elimination_matches_l163_163150


namespace boy_and_girl_roles_l163_163418

-- Definitions of the conditions
def Sasha_says_boy : Prop := True
def Zhenya_says_girl : Prop := True
def at_least_one_lying (sasha_boy zhenya_girl : Prop) : Prop := 
  (sasha_boy = False) ∨ (zhenya_girl = False)

-- Theorem statement
theorem boy_and_girl_roles (sasha_boy : Prop) (zhenya_girl : Prop) 
  (H1 : Sasha_says_boy) (H2 : Zhenya_says_girl) (H3 : at_least_one_lying sasha_boy zhenya_girl) :
  sasha_boy = False ∧ zhenya_girl = True :=
sorry

end boy_and_girl_roles_l163_163418


namespace john_growth_l163_163210

theorem john_growth 
  (InitialHeight : ℤ)
  (GrowthRate : ℤ)
  (FinalHeight : ℤ)
  (h1 : InitialHeight = 66)
  (h2 : GrowthRate = 2)
  (h3 : FinalHeight = 72) :
  (FinalHeight - InitialHeight) / GrowthRate = 3 :=
by
  sorry

end john_growth_l163_163210


namespace club_positions_l163_163732

def num_ways_to_fill_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5)

theorem club_positions : num_ways_to_fill_positions 12 = 665280 := by 
  sorry

end club_positions_l163_163732


namespace huanhuan_initial_coins_l163_163209

theorem huanhuan_initial_coins :
  ∃ (H L n : ℕ), H = 7 * L ∧ (H + n = 6 * (L + n)) ∧ (H + 2 * n = 5 * (L + 2 * n)) ∧ H = 70 :=
by
  sorry

end huanhuan_initial_coins_l163_163209


namespace n_pow_8_minus_1_divisible_by_480_l163_163968

theorem n_pow_8_minus_1_divisible_by_480 (n : ℤ) (h1 : ¬ (2 ∣ n)) (h2 : ¬ (3 ∣ n)) (h3 : ¬ (5 ∣ n)) : 
  480 ∣ (n^8 - 1) := 
sorry

end n_pow_8_minus_1_divisible_by_480_l163_163968


namespace urn_problem_l163_163605

theorem urn_problem : 
  (5 / 12 * 20 / (20 + M) + 7 / 12 * M / (20 + M) = 0.62) → M = 111 :=
by
  intro h
  sorry

end urn_problem_l163_163605


namespace gretchen_rachelle_ratio_l163_163029

-- Definitions of the conditions
def rachelle_pennies : ℕ := 180
def total_pennies : ℕ := 300
def rocky_pennies (gretchen_pennies : ℕ) : ℕ := gretchen_pennies / 3

-- The Lean 4 theorem statement
theorem gretchen_rachelle_ratio (gretchen_pennies : ℕ) 
    (h_total : rachelle_pennies + gretchen_pennies + rocky_pennies gretchen_pennies = total_pennies) :
    (gretchen_pennies : ℚ) / rachelle_pennies = 1 / 2 :=
sorry

end gretchen_rachelle_ratio_l163_163029


namespace hexagon_cookie_cutters_count_l163_163373

-- Definitions for the conditions
def triangle_side_count := 3
def triangles := 6
def square_side_count := 4
def squares := 4
def total_sides := 46

-- Given conditions translated to Lean 4
def sides_from_triangles := triangles * triangle_side_count
def sides_from_squares := squares * square_side_count
def sides_from_triangles_and_squares := sides_from_triangles + sides_from_squares
def sides_from_hexagons := total_sides - sides_from_triangles_and_squares
def hexagon_side_count := 6

-- Statement to prove that there are 2 hexagon-shaped cookie cutters
theorem hexagon_cookie_cutters_count : sides_from_hexagons / hexagon_side_count = 2 := by
  sorry

end hexagon_cookie_cutters_count_l163_163373


namespace exists_integer_cube_ends_with_2007_ones_l163_163536

theorem exists_integer_cube_ends_with_2007_ones :
  ∃ x : ℕ, x^3 % 10^2007 = 10^2007 - 1 :=
sorry

end exists_integer_cube_ends_with_2007_ones_l163_163536


namespace find_n_l163_163696

theorem find_n
  (c d : ℝ)
  (H1 : 450 * c + 300 * d = 300 * c + 375 * d)
  (H2 : ∃ t1 t2 t3 : ℝ, t1 = 4 ∧ t2 = 1 ∧ t3 = n ∧ 75 * 4 * (c + d) = 900 * c + t3 * d)
  : n = 600 / 7 :=
by
  sorry

end find_n_l163_163696


namespace base_equivalence_l163_163236

theorem base_equivalence :
  let n_7 := 4 * 7 + 3  -- 43 in base 7 expressed in base 10.
  ∃ d : ℕ, (3 * d + 4 = n_7) → d = 9 :=
by
  let n_7 := 31
  sorry

end base_equivalence_l163_163236


namespace max_value_expression_l163_163529

theorem max_value_expression (x : ℝ) : 
  (∃ y : ℝ, y = x^4 / (x^8 + 2 * x^6 + 4 * x^4 + 4 * x^2 + 16) ∧ 
                ∀ z : ℝ, 
                (∃ x : ℝ, z = x^4 / (x^8 + 2 * x^6 + 4 * x^4 + 4 * x^2 + 16)) → 
                y ≥ z) → 
  ∃ y : ℝ, y = 1 / 16 := 
sorry

end max_value_expression_l163_163529


namespace remainder_when_squared_l163_163593

theorem remainder_when_squared (n : ℤ) (h : n % 5 = 3) : (n^2) % 5 = 4 := by
  sorry

end remainder_when_squared_l163_163593


namespace rectangle_side_excess_l163_163252

theorem rectangle_side_excess
  (L W : ℝ)  -- length and width of the rectangle
  (x : ℝ)   -- percentage in excess for the first side
  (h1 : 0.95 * (L * (1 + x / 100) * W) = 1.102 * (L * W)) :
  x = 16 :=
by
  sorry

end rectangle_side_excess_l163_163252


namespace caterpillar_reaches_top_in_16_days_l163_163251

-- Define the constants for the problem
def pole_height : ℕ := 20
def daytime_climb : ℕ := 5
def nighttime_slide : ℕ := 4

-- Define the final result we want to prove
theorem caterpillar_reaches_top_in_16_days :
  ∃ days : ℕ, days = 16 ∧ 
  ((20 - 5) / (daytime_climb - nighttime_slide) + 1) = 16 := by
  sorry

end caterpillar_reaches_top_in_16_days_l163_163251


namespace find_other_side_length_l163_163926

variable (total_shingles : ℕ)
variable (shingles_per_sqft : ℕ)
variable (num_roofs : ℕ)
variable (side_length : ℕ)

theorem find_other_side_length
  (h1 : total_shingles = 38400)
  (h2 : shingles_per_sqft = 8)
  (h3 : num_roofs = 3)
  (h4 : side_length = 20)
  : (total_shingles / shingles_per_sqft / num_roofs / 2) / side_length = 40 :=
by
  sorry

end find_other_side_length_l163_163926


namespace part_I_monotonicity_part_II_value_a_l163_163333

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (x - 1)

def is_monotonic_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x < f y

def is_monotonic_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y < f x

theorem part_I_monotonicity :
  (is_monotonic_increasing f {x | 2 < x}) ∧
  ((is_monotonic_decreasing f {x | x < 1}) ∧ (is_monotonic_decreasing f {x | 1 < x ∧ x < 2})) :=
by
  sorry

theorem part_II_value_a (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → (Real.exp x * (x - 2)) / ((x - 1)^2) ≥ a * (Real.exp x / (x - 1))) → a ∈ Set.Iic 0 :=
by
  sorry

end part_I_monotonicity_part_II_value_a_l163_163333


namespace triangle_angle_and_perimeter_l163_163887

/-
In a triangle ABC, given c * sin B = sqrt 3 * cos C,
prove that angle C equals pi / 3,
and given a + b = 6, find the minimum perimeter of triangle ABC.
-/
theorem triangle_angle_and_perimeter (A B C : ℝ) (a b c : ℝ) 
  (h1 : c * Real.sin B = Real.sqrt 3 * Real.cos C)
  (h2 : a + b = 6) :
  C = Real.pi / 3 ∧ a + b + (Real.sqrt (36 - a * b)) = 9 :=
by
  sorry

end triangle_angle_and_perimeter_l163_163887


namespace find_divisor_l163_163193

theorem find_divisor 
  (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  (dividend = 172) → (quotient = 10) → (remainder = 2) → (dividend = (divisor * quotient) + remainder) → divisor = 17 :=
by 
  sorry

end find_divisor_l163_163193


namespace parabola_vertex_y_coordinate_l163_163610

theorem parabola_vertex_y_coordinate (x y : ℝ) :
  y = 5 * x^2 + 20 * x + 45 ∧ (∃ h k, y = 5 * (x + h)^2 + k ∧ k = 25) :=
by
  sorry

end parabola_vertex_y_coordinate_l163_163610


namespace tan_half_angle_product_l163_163624

theorem tan_half_angle_product (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ x : ℝ, x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = Real.sqrt (26 / 7) ∨ x = -Real.sqrt (26 / 7)) :=
by
  sorry

end tan_half_angle_product_l163_163624


namespace translation_vector_coords_l163_163320

-- Definitions according to the given conditions
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 1
def translated_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- Statement that we need to prove
theorem translation_vector_coords :
  ∃ (a b : ℝ), 
  (∀ x y : ℝ, original_circle x y ↔ translated_circle (x - a) (y - b)) ∧
  (a, b) = (-1, 2) := 
sorry

end translation_vector_coords_l163_163320


namespace train_speed_l163_163371

theorem train_speed
  (train_length : ℝ) (platform_length : ℝ) (time_seconds : ℝ)
  (h_train_length : train_length = 450)
  (h_platform_length : platform_length = 300.06)
  (h_time : time_seconds = 25) :
  (train_length + platform_length) / time_seconds * 3.6 = 108.01 :=
by
  -- skipping the proof with sorry
  sorry

end train_speed_l163_163371


namespace solve_for_c_l163_163112

theorem solve_for_c (a b c d e : ℝ) 
  (h1 : a + b + c = 48)
  (h2 : c + d + e = 78)
  (h3 : a + b + c + d + e = 100) :
  c = 26 :=
by
sorry

end solve_for_c_l163_163112


namespace emily_lives_l163_163390

theorem emily_lives :
  ∃ (lives_gained : ℕ), 
    let initial_lives := 42
    let lives_lost := 25
    let lives_after_loss := initial_lives - lives_lost
    let final_lives := 41
    lives_after_loss + lives_gained = final_lives :=
sorry

end emily_lives_l163_163390


namespace fraction_of_sum_l163_163432

theorem fraction_of_sum (P : ℝ) (R : ℝ) (T : ℝ) (H_R : R = 8.333333333333337) (H_T : T = 2) : 
  let SI := (P * R * T) / 100
  let A := P + SI
  A / P = 1.1666666666666667 :=
by
  sorry

end fraction_of_sum_l163_163432


namespace no_solution_implies_b_positive_l163_163227

theorem no_solution_implies_b_positive (a b : ℝ) :
  (¬ ∃ x y : ℝ, y = x^2 + a * x + b ∧ x = y^2 + a * y + b) → b > 0 :=
by
  sorry

end no_solution_implies_b_positive_l163_163227


namespace inscribed_circle_radius_eq_3_l163_163038

open Real

theorem inscribed_circle_radius_eq_3
  (a : ℝ) (A : ℝ) (p : ℝ) (r : ℝ)
  (h_eq_tri : ∀ (a : ℝ), A = (sqrt 3 / 4) * a^2)
  (h_perim : ∀ (a : ℝ), p = 3 * a)
  (h_area_perim : ∀ (a : ℝ), A = (3 / 2) * p) :
  r = 3 :=
by sorry

end inscribed_circle_radius_eq_3_l163_163038


namespace fraction_given_to_son_l163_163647

theorem fraction_given_to_son : 
  ∀ (blue_apples yellow_apples total_apples remaining_apples given_apples : ℕ),
    blue_apples = 5 →
    yellow_apples = 2 * blue_apples →
    total_apples = blue_apples + yellow_apples →
    remaining_apples = 12 →
    given_apples = total_apples - remaining_apples →
    (given_apples : ℚ) / total_apples = 1 / 5 :=
by
  intros
  sorry

end fraction_given_to_son_l163_163647


namespace order_of_abc_l163_163550

section
variables {a b c : ℝ}

def a_def : a = (1/2) * Real.log 2 := by sorry
def b_def : b = (1/4) * Real.log 16 := by sorry
def c_def : c = (1/6) * Real.log 27 := by sorry

theorem order_of_abc : a < c ∧ c < b :=
by
  have ha : a = (1/2) * Real.log 2 := by sorry
  have hb : b = (1/2) * Real.log 4 := by sorry
  have hc : c = (1/2) * Real.log 3 := by sorry
  sorry
end

end order_of_abc_l163_163550


namespace solution_set_of_inequality_l163_163240

theorem solution_set_of_inequality (x : ℝ) : 
  (x + 3) * (1 - x) ≥ 0 ↔ -3 ≤ x ∧ x ≤ 1 := 
sorry

end solution_set_of_inequality_l163_163240


namespace natasha_avg_speed_climbing_l163_163853

-- Natasha climbs up a hill in 4 hours and descends in 2 hours.
-- Her average speed along the whole journey is 1.5 km/h.
-- Prove that her average speed while climbing to the top is 1.125 km/h.

theorem natasha_avg_speed_climbing (v_up v_down : ℝ) :
  (4 * v_up = 2 * v_down) ∧ (1.5 = (2 * (4 * v_up) / 6)) → v_up = 1.125 :=
by
  -- We provide no proof here; this is just the statement.
  sorry

end natasha_avg_speed_climbing_l163_163853


namespace find_sample_size_l163_163815

theorem find_sample_size : ∃ n : ℕ, n ∣ 36 ∧ (n + 1) ∣ 35 ∧ n = 6 := by
  use 6
  simp
  sorry

end find_sample_size_l163_163815


namespace boxes_needed_l163_163132

-- Let's define the conditions
def total_paper_clips : ℕ := 81
def paper_clips_per_box : ℕ := 9

-- Define the target of our proof, which is that the number of boxes needed is 9
theorem boxes_needed : total_paper_clips / paper_clips_per_box = 9 := by
  sorry

end boxes_needed_l163_163132


namespace kenya_peanuts_correct_l163_163074

def jose_peanuts : ℕ := 85
def kenya_more_peanuts : ℕ := 48

def kenya_peanuts : ℕ := jose_peanuts + kenya_more_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := by
  sorry

end kenya_peanuts_correct_l163_163074


namespace cakes_sold_l163_163291

theorem cakes_sold (total_made : ℕ) (cakes_left : ℕ) (cakes_sold : ℕ) :
  total_made = 217 ∧ cakes_left = 72 → cakes_sold = 145 :=
by
  -- Assuming total_made is 217 and cakes_left is 72, we need to show cakes_sold = 145
  sorry

end cakes_sold_l163_163291


namespace expression_equivalence_l163_163574

theorem expression_equivalence:
  let a := 10006 - 8008
  let b := 10000 - 8002
  a = b :=
by {
  sorry
}

end expression_equivalence_l163_163574


namespace pollen_allergy_expected_count_l163_163591

theorem pollen_allergy_expected_count : 
  ∀ (sample_size : ℕ) (pollen_allergy_ratio : ℚ), 
  pollen_allergy_ratio = 1/4 ∧ sample_size = 400 → sample_size * pollen_allergy_ratio = 100 :=
  by 
    intros
    sorry

end pollen_allergy_expected_count_l163_163591


namespace second_player_wins_for_n_11_l163_163392

theorem second_player_wins_for_n_11 (N : ℕ) (h1 : N = 11) :
  ∃ (list : List ℕ), (∀ x ∈ list, x > 0 ∧ x ≤ 25) ∧
     list.sum ≥ 200 ∧
     (∃ sublist : List ℕ, sublist.sum ≥ 200 - N ∧ sublist.sum ≤ 200 + N) :=
by
  let N := 11
  sorry

end second_player_wins_for_n_11_l163_163392


namespace positive_intervals_of_product_l163_163935

theorem positive_intervals_of_product (x : ℝ) : 
  ((x + 2) * (x - 3) > 0) ↔ (x < -2 ∨ x > 3) := 
sorry

end positive_intervals_of_product_l163_163935


namespace total_cars_parked_l163_163458

theorem total_cars_parked
  (area_a : ℕ) (util_a : ℕ)
  (area_b : ℕ) (util_b : ℕ)
  (area_c : ℕ) (util_c : ℕ)
  (area_d : ℕ) (util_d : ℕ)
  (space_per_car : ℕ) 
  (ha: area_a = 400 * 500)
  (hu_a: util_a = 80)
  (hb: area_b = 600 * 700)
  (hu_b: util_b = 75)
  (hc: area_c = 500 * 800)
  (hu_c: util_c = 65)
  (hd: area_d = 300 * 900)
  (hu_d: util_d = 70)
  (h_sp: space_per_car = 10) :
  (util_a * area_a / 100 / space_per_car + 
   util_b * area_b / 100 / space_per_car + 
   util_c * area_c / 100 / space_per_car + 
   util_d * area_d / 100 / space_per_car) = 92400 :=
by sorry

end total_cars_parked_l163_163458


namespace selection_methods_count_l163_163004

-- Define the number of female students
def num_female_students : ℕ := 3

-- Define the number of male students
def num_male_students : ℕ := 2

-- Define the total number of different selection methods
def total_selection_methods : ℕ := num_female_students + num_male_students

-- Prove that the total number of different selection methods is 5
theorem selection_methods_count : total_selection_methods = 5 := by
  sorry

end selection_methods_count_l163_163004


namespace medians_sum_le_circumradius_l163_163907

-- Definition of the problem
variable (a b c R : ℝ) (m_a m_b m_c : ℝ)

-- Conditions: medians of triangle ABC, and R is the circumradius
def is_median (m : ℝ) (a b c : ℝ) : Prop :=
  m^2 = (2*b^2 + 2*c^2 - a^2) / 4

-- Main theorem to prove
theorem medians_sum_le_circumradius (h_ma : is_median m_a a b c)
  (h_mb : is_median m_b b a c) (h_mc : is_median m_c c a b) 
  (h_R : a^2 + b^2 + c^2 ≤ 9 * R^2) :
  m_a + m_b + m_c ≤ 9 / 2 * R :=
sorry

end medians_sum_le_circumradius_l163_163907


namespace first_three_workers_dig_time_l163_163632

variables 
  (a b c d : ℝ) -- work rates of the four workers
  (hours : ℝ) -- time to dig the trench

def work_together (a b c d hours : ℝ) := (a + b + c + d) * hours = 1

def scenario1 (a b c d : ℝ) := (2 * a + (1/2) * b + c + d) * 6 = 1

def scenario2 (a b c d : ℝ) := (a/2 + 2 * b + c + d) * 4 = 1

theorem first_three_workers_dig_time
  (h1 : work_together a b c d 6)
  (h2 : scenario1 a b c d)
  (h3 : scenario2 a b c d) :
  hours = 6 := 
sorry

end first_three_workers_dig_time_l163_163632


namespace math_problem_l163_163674

theorem math_problem 
  (a b c : ℝ)
  (h1 : a < b)
  (h2 : ∀ x, (x < -2 ∨ |x - 30| ≤ 2) ↔ ( (x - a) * (x - b) / (x - c) ≤ 0 )) :
  a + 2 * b + 3 * c = 86 :=
sorry

end math_problem_l163_163674


namespace carrots_problem_l163_163104

def total_carrots (faye_picked : Nat) (mother_picked : Nat) : Nat :=
  faye_picked + mother_picked

def bad_carrots (total_carrots : Nat) (good_carrots : Nat) : Nat :=
  total_carrots - good_carrots

theorem carrots_problem (faye_picked : Nat) (mother_picked : Nat) (good_carrots : Nat) (bad_carrots : Nat) 
  (h1 : faye_picked = 23) 
  (h2 : mother_picked = 5)
  (h3 : good_carrots = 12) :
  bad_carrots = 16 := sorry

end carrots_problem_l163_163104


namespace more_oaks_than_willows_l163_163039

theorem more_oaks_than_willows (total_trees willows : ℕ) (h1 : total_trees = 83) (h2 : willows = 36) :
  (total_trees - willows) - willows = 11 :=
by
  sorry

end more_oaks_than_willows_l163_163039


namespace abs_neg_five_l163_163629

theorem abs_neg_five : abs (-5) = 5 := 
by 
  sorry

end abs_neg_five_l163_163629


namespace sin_neg_765_eq_neg_sqrt2_div_2_l163_163213

theorem sin_neg_765_eq_neg_sqrt2_div_2 :
  Real.sin (-765 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_765_eq_neg_sqrt2_div_2_l163_163213


namespace metric_regression_equation_l163_163381

noncomputable def predicted_weight_imperial (height : ℝ) : ℝ :=
  4 * height - 130

def inch_to_cm (inch : ℝ) : ℝ := 2.54 * inch
def pound_to_kg (pound : ℝ) : ℝ := 0.45 * pound

theorem metric_regression_equation (height_cm : ℝ) :
  (0.72 * height_cm - 58.5) = 
  (pound_to_kg (predicted_weight_imperial (height_cm / 2.54))) :=
by
  sorry

end metric_regression_equation_l163_163381


namespace part1_part2_l163_163321

variable {m n : ℤ}

theorem part1 (hm : |m| = 1) (hn : |n| = 4) (hprod : m * n < 0) : m + n = -3 ∨ m + n = 3 := sorry

theorem part2 (hm : |m| = 1) (hn : |n| = 4) : ∃ (k : ℤ), k = 5 ∧ ∀ x, x = m - n → x ≤ k := sorry

end part1_part2_l163_163321


namespace sum_of_coefficients_l163_163904

theorem sum_of_coefficients (a₅ a₄ a₃ a₂ a₁ a₀ : ℤ)
  (h₀ : (x - 2)^5 = a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀)
  (h₁ : a₅ + a₄ + a₃ + a₂ + a₁ + a₀ = -1)
  (h₂ : a₀ = -32) :
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 :=
sorry

end sum_of_coefficients_l163_163904


namespace perpendicular_lines_l163_163675

def line1 (a : ℝ) (x y : ℝ) := a * x + 2 * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) := x + (a - 1) * y + a^2 - 1 = 0

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, line1 a x y) ∧ (∀ x y : ℝ, line2 a x y) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, 
    (line1 a x1 y1) ∧ (line2 a x2 y2) → 
    (-a / 2) * (-1 / (a - 1)) = -1) → a = 2 / 3 :=
sorry

end perpendicular_lines_l163_163675


namespace max_marks_l163_163651

theorem max_marks (M : ℝ) :
  (0.33 * M = 125 + 73) → M = 600 := by
  intro h
  sorry

end max_marks_l163_163651


namespace total_population_l163_163567

variable (b g t : ℕ)

-- Conditions: 
axiom boys_to_girls (h1 : b = 4 * g) : Prop
axiom girls_to_teachers (h2 : g = 8 * t) : Prop

theorem total_population (h1 : b = 4 * g) (h2 : g = 8 * t) : b + g + t = 41 * b / 32 :=
sorry

end total_population_l163_163567


namespace first_term_value_l163_163983

noncomputable def find_first_term (a r : ℝ) := a / (1 - r) = 27 ∧ a^2 / (1 - r^2) = 108

theorem first_term_value :
  ∃ (a r : ℝ), find_first_term a r ∧ a = 216 / 31 :=
by
  sorry

end first_term_value_l163_163983


namespace people_owning_only_cats_and_dogs_l163_163019

theorem people_owning_only_cats_and_dogs 
  (total_people : ℕ) 
  (only_dogs : ℕ) 
  (only_cats : ℕ) 
  (cats_dogs_snakes : ℕ) 
  (total_snakes : ℕ) 
  (only_cats_and_dogs : ℕ) 
  (h1 : total_people = 89) 
  (h2 : only_dogs = 15) 
  (h3 : only_cats = 10) 
  (h4 : cats_dogs_snakes = 3) 
  (h5 : total_snakes = 59) 
  (h6 : total_people = only_dogs + only_cats + only_cats_and_dogs + cats_dogs_snakes + (total_snakes - cats_dogs_snakes)) : 
  only_cats_and_dogs = 5 := 
by 
  sorry

end people_owning_only_cats_and_dogs_l163_163019


namespace extracurricular_hours_l163_163230

theorem extracurricular_hours :
  let soccer_hours_per_day := 2
  let soccer_days := 3
  let band_hours_per_day := 1.5
  let band_days := 2
  let total_soccer_hours := soccer_hours_per_day * soccer_days
  let total_band_hours := band_hours_per_day * band_days
  total_soccer_hours + total_band_hours = 9 := 
by
  -- The proof steps go here.
  sorry

end extracurricular_hours_l163_163230


namespace expression_value_l163_163425

   theorem expression_value :
     (20 - (2010 - 201)) + (2010 - (201 - 20)) = 40 := 
   by
     sorry
   
end expression_value_l163_163425


namespace area_PQR_is_4_5_l163_163640

noncomputable def point := (ℝ × ℝ)

def P : point := (2, 1)
def Q : point := (1, 4)
def R_line (x: ℝ) : point := (x, 6 - x)

def area_triangle (A B C : point) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

theorem area_PQR_is_4_5 (x : ℝ) (h : R_line x ∈ {p : point | p.1 + p.2 = 6}) : 
  area_triangle P Q (R_line x) = 4.5 :=
    sorry

end area_PQR_is_4_5_l163_163640


namespace circle_area_increase_l163_163403

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let r_new := 1.5 * r
  let area_original := π * r^2
  let area_new := π * r_new^2
  let increase := area_new - area_original
  let percentage_increase := (increase / area_original) * 100
  percentage_increase = 125 :=
by
  let r_new := 1.5 * r
  let area_original := π * r^2
  let area_new := π * r_new^2
  let increase := area_new - area_original
  let percentage_increase := (increase / area_original) * 100
  sorry

end circle_area_increase_l163_163403


namespace maximum_students_l163_163511

-- Definitions for conditions
def students (n : ℕ) := Fin n → Prop

-- Condition: Among any six students, there are two who are not friends
def not_friend_in_six (n : ℕ) (friend : Fin n → Fin n → Prop) : Prop :=
  ∀ (s : Finset (Fin n)), s.card = 6 → ∃ (a b : Fin n), a ∈ s ∧ b ∈ s ∧ ¬ friend a b

-- Condition: For any pair of students not friends, there is a student who is friends with both
def friend_of_two_not_friends (n : ℕ) (friend : Fin n → Fin n → Prop) : Prop :=
  ∀ (a b : Fin n), ¬ friend a b → ∃ (c : Fin n), c ≠ a ∧ c ≠ b ∧ friend c a ∧ friend c b

-- Theorem stating the main result
theorem maximum_students (n : ℕ) (friend : Fin n → Fin n → Prop) :
  not_friend_in_six n friend ∧ friend_of_two_not_friends n friend → n ≤ 25 := 
sorry

end maximum_students_l163_163511


namespace total_cost_is_correct_l163_163328

noncomputable def nights : ℕ := 3
noncomputable def cost_per_night : ℕ := 250
noncomputable def discount : ℕ := 100

theorem total_cost_is_correct :
  (nights * cost_per_night) - discount = 650 := by
sorry

end total_cost_is_correct_l163_163328


namespace sample_capacity_n_l163_163905

theorem sample_capacity_n
  (n : ℕ) 
  (engineers technicians craftsmen : ℕ) 
  (total_population : ℕ)
  (stratified_interval systematic_interval : ℕ) :
  engineers = 6 →
  technicians = 12 →
  craftsmen = 18 →
  total_population = engineers + technicians + craftsmen →
  total_population = 36 →
  (∃ n : ℕ, n ∣ total_population ∧ 6 ∣ n ∧ 35 % (n + 1) = 0) →
  n = 6 :=
by
  sorry

end sample_capacity_n_l163_163905


namespace expand_polynomial_l163_163136

theorem expand_polynomial :
  (7 * x^2 + 5 * x - 3) * (3 * x^3 + 2 * x^2 - x + 4) = 21 * x^5 + 29 * x^4 - 6 * x^3 + 17 * x^2 + 23 * x - 12 :=
by
  sorry

end expand_polynomial_l163_163136


namespace total_goats_l163_163500

theorem total_goats (W: ℕ) (H_W: W = 180) (H_P: W + 70 = 250) : W + (W + 70) = 430 :=
by
  -- proof goes here
  sorry

end total_goats_l163_163500


namespace line_intersects_circle_l163_163329

variable {a x_0 y_0 : ℝ}

theorem line_intersects_circle (h1: x_0^2 + y_0^2 > a^2) (h2: a > 0) : 
  ∃ (p : ℝ × ℝ), (p.1 ^ 2 + p.2 ^ 2 = a ^ 2) ∧ (x_0 * p.1 + y_0 * p.2 = a ^ 2) :=
sorry

end line_intersects_circle_l163_163329


namespace tan_double_angle_identity_l163_163212

theorem tan_double_angle_identity (θ : ℝ) (h : Real.tan θ = Real.sqrt 3) : 
  Real.sin (2 * θ) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := 
  sorry

end tan_double_angle_identity_l163_163212


namespace greatest_divisor_450_90_l163_163015

open Nat

-- Define a condition for the set of divisors of given numbers which are less than a certain number.
def is_divisor (a : ℕ) (b : ℕ) : Prop := b % a = 0

def is_greatest_divisor (d : ℕ) (n : ℕ) (m : ℕ) (k : ℕ) : Prop :=
  is_divisor m d ∧ d < k ∧ ∀ (x : ℕ), x < k → is_divisor m x → x ≤ d

-- Define the proof problem.
theorem greatest_divisor_450_90 : is_greatest_divisor 18 450 90 30 := 
by
  sorry

end greatest_divisor_450_90_l163_163015


namespace rigged_coin_probability_l163_163585

theorem rigged_coin_probability (p : ℝ) (h1 : p < 1 / 2) (h2 : 20 * (p ^ 3) * ((1 - p) ^ 3) = 1 / 12) :
  p = (1 - Real.sqrt 0.86) / 2 :=
by
  sorry

end rigged_coin_probability_l163_163585


namespace baker_sales_difference_l163_163521

/-!
  Prove that the difference in dollars between the baker's daily average sales and total sales for today is 48 dollars.
-/

theorem baker_sales_difference :
  let price_pastry := 2
  let price_bread := 4
  let avg_pastries := 20
  let avg_bread := 10
  let today_pastries := 14
  let today_bread := 25
  let daily_avg_sales := avg_pastries * price_pastry + avg_bread * price_bread
  let today_sales := today_pastries * price_pastry + today_bread * price_bread
  daily_avg_sales - today_sales = 48 :=
sorry

end baker_sales_difference_l163_163521


namespace coin_flip_prob_nickel_halfdollar_heads_l163_163999

def coin_prob : ℚ :=
  let total_outcomes := 2^5
  let successful_outcomes := 2^3
  successful_outcomes / total_outcomes

theorem coin_flip_prob_nickel_halfdollar_heads :
  coin_prob = 1 / 4 :=
by
  sorry

end coin_flip_prob_nickel_halfdollar_heads_l163_163999


namespace length_of_room_l163_163739

theorem length_of_room 
  (width : ℝ) (cost : ℝ) (rate : ℝ) (area : ℝ) (length : ℝ) 
  (h1 : width = 3.75) 
  (h2 : cost = 24750) 
  (h3 : rate = 1200) 
  (h4 : area = cost / rate) 
  (h5 : area = length * width) : 
  length = 5.5 :=
sorry

end length_of_room_l163_163739


namespace percentage_difference_l163_163128

theorem percentage_difference : (0.5 * 56) - (0.3 * 50) = 13 := by
  sorry

end percentage_difference_l163_163128


namespace longer_side_of_rectangle_l163_163125

theorem longer_side_of_rectangle
  (r : ℝ) (A_rect A_circle L S : ℝ) (h1 : r = 6)
  (h2 : A_circle = π * r^2)
  (h3 : A_rect = 3 * A_circle)
  (h4 : S = 2 * r)
  (h5 : A_rect = S * L) : L = 9 * π :=
by
  sorry

end longer_side_of_rectangle_l163_163125


namespace odd_pos_4_digit_ints_div_5_no_digit_5_l163_163568

open Nat

def is_valid_digit (d : Nat) : Prop :=
  d ≠ 5

def valid_odd_4_digit_ints_count : Nat :=
  let a := 8  -- First digit possibilities: {1, 2, 3, 4, 6, 7, 8, 9}
  let bc := 9  -- Second and third digit possibilities: {0, 1, 2, 3, 4, 6, 7, 8, 9}
  let d := 4  -- Fourth digit possibilities: {1, 3, 7, 9}
  a * bc * bc * d

theorem odd_pos_4_digit_ints_div_5_no_digit_5 : valid_odd_4_digit_ints_count = 2592 := by
  sorry

end odd_pos_4_digit_ints_div_5_no_digit_5_l163_163568


namespace fraction_of_primes_is_prime_l163_163268

theorem fraction_of_primes_is_prime
  (p q r : ℕ) 
  (hp : Nat.Prime p)
  (hq : Nat.Prime q)
  (hr : Nat.Prime r)
  (h : ∃ k : ℕ, p * q * r = k * (p + q + r)) :
  Nat.Prime (p * q * r / (p + q + r)) := 
sorry

end fraction_of_primes_is_prime_l163_163268


namespace hotel_r_charge_percentage_l163_163454

-- Let P, R, and G be the charges for a single room at Hotels P, R, and G respectively
variables (P R G : ℝ)

-- Given conditions:
-- 1. The charge for a single room at Hotel P is 55% less than the charge for a single room at Hotel R.
-- 2. The charge for a single room at Hotel P is 10% less than the charge for a single room at Hotel G.
axiom h1 : P = 0.45 * R
axiom h2 : P = 0.90 * G

-- The charge for a single room at Hotel R is what percent greater than the charge for a single room at Hotel G.
theorem hotel_r_charge_percentage : (R - G) / G * 100 = 100 :=
sorry

end hotel_r_charge_percentage_l163_163454


namespace bn_six_eight_product_l163_163950

noncomputable def sequence_an (n : ℕ) : ℝ := sorry  -- given that an is an arithmetic sequence and an ≠ 0
noncomputable def sequence_bn (n : ℕ) : ℝ := sorry  -- given that bn is a geometric sequence

theorem bn_six_eight_product :
  (∀ n : ℕ, sequence_an n ≠ 0) →
  2 * sequence_an 3 - sequence_an 7 ^ 2 + 2 * sequence_an 11 = 0 →
  sequence_bn 7 = sequence_an 7 →
  sequence_bn 6 * sequence_bn 8 = 16 :=
sorry

end bn_six_eight_product_l163_163950


namespace unique_quantities_not_determinable_l163_163089

noncomputable def impossible_to_determine_unique_quantities 
(x y : ℝ) : Prop :=
  let acid1 := 54 * 0.35
  let acid2 := 48 * 0.25
  ∀ (final_acid : ℝ), ¬(0.35 * x + 0.25 * y = final_acid ∧ final_acid = 0.75 * (x + y))

theorem unique_quantities_not_determinable :
  impossible_to_determine_unique_quantities 54 48 :=
by
  sorry

end unique_quantities_not_determinable_l163_163089


namespace find_x_intercept_of_perpendicular_line_l163_163707

noncomputable def line_y_intercept : ℝ × ℝ := (0, 3)
noncomputable def given_line (x y : ℝ) : Prop := 2 * x + y = 3
noncomputable def x_intercept_of_perpendicular_line : ℝ × ℝ := (-6, 0)

theorem find_x_intercept_of_perpendicular_line :
  (∀ (x y : ℝ), given_line x y → (slope_of_perpendicular_line : ℝ) = 1/2 ∧ 
  ∀ (b : ℝ), line_y_intercept = (0, b) → ∀ (y : ℝ), y = 1/2 * x + b → (x, 0) = x_intercept_of_perpendicular_line) :=
sorry

end find_x_intercept_of_perpendicular_line_l163_163707


namespace area_of_path_correct_l163_163738

noncomputable def area_of_path (length_field : ℝ) (width_field : ℝ) (path_width : ℝ) : ℝ :=
  let length_total := length_field + 2 * path_width
  let width_total := width_field + 2 * path_width
  let area_total := length_total * width_total
  let area_field := length_field * width_field
  area_total - area_field

theorem area_of_path_correct :
  area_of_path 75 55 3.5 = 959 := 
by
  sorry

end area_of_path_correct_l163_163738


namespace structure_burns_in_65_seconds_l163_163826

noncomputable def toothpick_grid_burn_time (m n : ℕ) (toothpicks : ℕ) (burn_time : ℕ) : ℕ :=
  if (m = 3 ∧ n = 5 ∧ toothpicks = 38 ∧ burn_time = 10) then 65 else 0

theorem structure_burns_in_65_seconds : toothpick_grid_burn_time 3 5 38 10 = 65 := by
  sorry

end structure_burns_in_65_seconds_l163_163826


namespace even_sum_probability_l163_163566

theorem even_sum_probability :
  let wheel1 := (2/6, 3/6, 1/6)   -- (probability of even, odd, zero) for the first wheel
  let wheel2 := (2/4, 2/4)        -- (probability of even, odd) for the second wheel
  let both_even := (1/3) * (1/2)  -- probability of both numbers being even
  let both_odd := (1/2) * (1/2)   -- probability of both numbers being odd
  let zero_and_even := (1/6) * (1/2)  -- probability of one number being zero and the other even
  let total_probability := both_even + both_odd + zero_and_even
  total_probability = 1/2 := by sorry

end even_sum_probability_l163_163566


namespace slow_car_speed_l163_163973

theorem slow_car_speed (x : ℝ) (hx : 0 < x) (distance : ℝ) (delay : ℝ) (fast_factor : ℝ) :
  distance = 60 ∧ delay = 0.5 ∧ fast_factor = 1.5 ∧ 
  (distance / x) - (distance / (fast_factor * x)) = delay → 
  x = 40 :=
by
  intros h
  sorry

end slow_car_speed_l163_163973


namespace jackson_saving_l163_163603

theorem jackson_saving (total_amount : ℝ) (months : ℕ) (paychecks_per_month : ℕ) (savings_per_paycheck : ℝ) :
  total_amount = 3000 → months = 15 → paychecks_per_month = 2 →
  savings_per_paycheck = total_amount / months / paychecks_per_month :=
by sorry

end jackson_saving_l163_163603


namespace probability_colored_ball_l163_163723

theorem probability_colored_ball (total_balls blue_balls green_balls white_balls : ℕ)
  (h_total : total_balls = 40)
  (h_blue : blue_balls = 15)
  (h_green : green_balls = 5)
  (h_white : white_balls = 20)
  (h_disjoint : total_balls = blue_balls + green_balls + white_balls) :
  (blue_balls + green_balls) / total_balls = 1 / 2 := by
  -- Proof skipped
  sorry

end probability_colored_ball_l163_163723


namespace solve_for_first_expedition_weeks_l163_163773

-- Define the variables according to the given conditions.
variables (x : ℕ)
variables (days_in_week : ℕ := 7)
variables (total_days_on_island : ℕ := 126)

-- Define the total number of weeks spent on the expeditions.
def total_weeks_on_expeditions (x : ℕ) : ℕ := 
  x + (x + 2) + 2 * (x + 2)

-- Convert total days to weeks.
def total_weeks := total_days_on_island / days_in_week

-- Prove the equation
theorem solve_for_first_expedition_weeks
  (h : total_weeks_on_expeditions x = total_weeks):
  x = 3 :=
by
  sorry

end solve_for_first_expedition_weeks_l163_163773


namespace brass_selling_price_l163_163678

noncomputable def copper_price : ℝ := 0.65
noncomputable def zinc_price : ℝ := 0.30
noncomputable def total_weight_brass : ℝ := 70
noncomputable def weight_copper : ℝ := 30
noncomputable def weight_zinc := total_weight_brass - weight_copper
noncomputable def cost_copper := weight_copper * copper_price
noncomputable def cost_zinc := weight_zinc * zinc_price
noncomputable def total_cost := cost_copper + cost_zinc
noncomputable def selling_price_per_pound := total_cost / total_weight_brass

theorem brass_selling_price :
  selling_price_per_pound = 0.45 :=
by
  sorry

end brass_selling_price_l163_163678


namespace multiple_of_4_multiple_of_8_multiple_of_16_not_multiple_of_32_l163_163748

def y : ℕ := 32 + 48 + 64 + 96 + 200 + 224 + 1600

theorem multiple_of_4 : y % 4 = 0 := by
  -- proof needed
  sorry

theorem multiple_of_8 : y % 8 = 0 := by
  -- proof needed
  sorry

theorem multiple_of_16 : y % 16 = 0 := by
  -- proof needed
  sorry

theorem not_multiple_of_32 : y % 32 ≠ 0 := by
  -- proof needed
  sorry

end multiple_of_4_multiple_of_8_multiple_of_16_not_multiple_of_32_l163_163748


namespace saleswoman_commission_l163_163792

theorem saleswoman_commission (x : ℝ) (h1 : ∀ sale : ℝ, sale = 800) (h2 : (x / 100) * 500 + 0.25 * (800 - 500) = 0.21875 * 800) : x = 20 := by
  sorry

end saleswoman_commission_l163_163792


namespace part_a_value_range_part_b_value_product_l163_163611

-- Define the polynomial 
def P (x y : ℤ) : ℤ := 2 * x^2 - 6 * x * y + 5 * y^2

-- Part (a)
theorem part_a_value_range :
  ∀ (x y : ℤ), (1 ≤ P x y) ∧ (P x y ≤ 100) → ∃ (a b : ℤ), 1 ≤ P a b ∧ P a b ≤ 100 := sorry

-- Part (b)
theorem part_b_value_product :
  ∀ (a b c d : ℤ),
    P a b = r → P c d = s → ∀ (r s : ℤ), (∃ (x y : ℤ), P x y = r) ∧ (∃ (z w : ℤ), P z w = s) → 
    ∃ (u v : ℤ), P u v = r * s := sorry

end part_a_value_range_part_b_value_product_l163_163611


namespace expression_evaluates_to_2023_l163_163009

theorem expression_evaluates_to_2023 : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 :=
by 
  sorry

end expression_evaluates_to_2023_l163_163009


namespace find_n_tan_eq_l163_163071

theorem find_n_tan_eq (n : ℤ) (h1 : -90 < n ∧ n < 90) (h2 : ∀ k : ℤ, 225 - 180 * k = 45) : n = 45 := by
  sorry

end find_n_tan_eq_l163_163071


namespace min_a_value_l163_163177

theorem min_a_value {a b : ℕ} (h : 1998 * a = b^4) : a = 1215672 :=
sorry

end min_a_value_l163_163177


namespace div_binomial_expansion_l163_163775

theorem div_binomial_expansion
  (a n b : Nat)
  (hb : a^n ∣ b) :
  a^(n+1) ∣ (a+1)^b - 1 := by
  sorry

end div_binomial_expansion_l163_163775


namespace solve_abs_equation_l163_163491

theorem solve_abs_equation (x : ℝ) : 2 * |x - 5| = 6 ↔ x = 2 ∨ x = 8 :=
by
  sorry

end solve_abs_equation_l163_163491


namespace new_estimated_y_value_l163_163278

theorem new_estimated_y_value
  (initial_slope : ℝ) (initial_intercept : ℝ) (avg_x_initial : ℝ)
  (datapoints_removed_low_x : ℝ) (datapoints_removed_high_x : ℝ)
  (datapoints_removed_low_y : ℝ) (datapoints_removed_high_y : ℝ)
  (new_slope : ℝ) 
  (x_value : ℝ)
  (estimated_y_new : ℝ) :
  initial_slope = 1.5 →
  initial_intercept = 1 →
  avg_x_initial = 2 →
  datapoints_removed_low_x = 2.6 →
  datapoints_removed_high_x = 1.4 →
  datapoints_removed_low_y = 2.8 →
  datapoints_removed_high_y = 5.2 →
  new_slope = 1.4 →
  x_value = 6 →
  estimated_y_new = new_slope * x_value + (4 - new_slope * avg_x_initial) →
  estimated_y_new = 9.6 := by
  sorry

end new_estimated_y_value_l163_163278


namespace incircle_intersections_equation_l163_163280

-- Assume a triangle ABC with the given configuration
variables {A B C D E F M N : Type}

-- Incircle touches sides CA, AB at points E, F respectively
-- Lines BE and CF intersect the incircle again at points M and N respectively

theorem incircle_intersections_equation
  (triangle_ABC : Type)
  (incircle_I : Type)
  (touch_CA : Type)
  (touch_AB : Type)
  (intersect_BE : Type)
  (intersect_CF : Type)
  (E F : triangle_ABC → incircle_I)
  (M N : intersect_BE → intersect_CF)
  : 
  MN * EF = 3 * MF * NE :=
by 
  -- Sorry as the proof is omitted
  sorry

end incircle_intersections_equation_l163_163280


namespace perfect_square_quadratic_l163_163448

theorem perfect_square_quadratic (a : ℝ) :
  ∃ (b : ℝ), (x : ℝ) → (x^2 - ax + 16) = (x + b)^2 ∨ (x^2 - ax + 16) = (x - b)^2 → a = 8 ∨ a = -8 :=
by
  sorry

end perfect_square_quadratic_l163_163448


namespace gcd_of_840_and_1764_l163_163211

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := 
by {
  sorry
}

end gcd_of_840_and_1764_l163_163211


namespace least_positive_integer_to_add_l163_163877

theorem least_positive_integer_to_add (n : ℕ) (h1 : n > 0) (h2 : (624 + n) % 5 = 0) : n = 1 := 
by
  sorry

end least_positive_integer_to_add_l163_163877


namespace smallest_X_divisible_15_l163_163228

theorem smallest_X_divisible_15 (T X : ℕ) 
  (h1 : T > 0) 
  (h2 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) 
  (h3 : T % 15 = 0) 
  (h4 : X = T / 15) : 
  X = 74 :=
sorry

end smallest_X_divisible_15_l163_163228


namespace sum_of_inner_segments_l163_163032

/-- Given the following conditions:
  1. The sum of the perimeters of the three quadrilaterals is 25 centimeters.
  2. The sum of the perimeters of the four triangles is 20 centimeters.
  3. The perimeter of triangle ABC is 19 centimeters.
Prove that AD + BE + CF = 13 centimeters. -/
theorem sum_of_inner_segments 
  (perimeter_quads : ℝ)
  (perimeter_tris : ℝ)
  (perimeter_ABC : ℝ)
  (hq : perimeter_quads = 25)
  (ht : perimeter_tris = 20)
  (hABC : perimeter_ABC = 19) 
  : AD + BE + CF = 13 :=
by
  sorry

end sum_of_inner_segments_l163_163032


namespace find_original_mean_l163_163364

noncomputable def original_mean (M : ℝ) : Prop :=
  let num_observations := 50
  let decrement := 47
  let updated_mean := 153
  M * num_observations - (num_observations * decrement) = updated_mean * num_observations

theorem find_original_mean : original_mean 200 :=
by
  unfold original_mean
  simp [*, mul_sub_left_distrib] at *
  sorry

end find_original_mean_l163_163364


namespace gcd_k_power_eq_k_minus_one_l163_163113

noncomputable def gcd_k_power (k : ℤ) : ℤ := 
  Int.gcd (k^1024 - 1) (k^1035 - 1)

theorem gcd_k_power_eq_k_minus_one (k : ℤ) : gcd_k_power k = k - 1 := 
  sorry

end gcd_k_power_eq_k_minus_one_l163_163113


namespace area_of_triangle_with_given_medians_l163_163563

noncomputable def area_of_triangle (m1 m2 m3 : ℝ) : ℝ :=
sorry

theorem area_of_triangle_with_given_medians :
    area_of_triangle 3 4 5 = 8 :=
sorry

end area_of_triangle_with_given_medians_l163_163563


namespace loss_per_meter_is_5_l163_163863

-- Define the conditions
def selling_price : ℕ := 18000
def cost_price_per_meter : ℕ := 50
def quantity : ℕ := 400

-- Define the statement to prove (question == answer given conditions)
theorem loss_per_meter_is_5 : 
  ((cost_price_per_meter * quantity - selling_price) / quantity) = 5 := 
by
  sorry

end loss_per_meter_is_5_l163_163863


namespace total_songs_l163_163061

open Nat

/-- Define the overall context and setup for the problem --/
def girls : List String := ["Mary", "Alina", "Tina", "Hanna"]

def hanna_songs : ℕ := 7
def mary_songs : ℕ := 4

def alina_songs (a : ℕ) : Prop := a > mary_songs ∧ a < hanna_songs
def tina_songs (t : ℕ) : Prop := t > mary_songs ∧ t < hanna_songs

theorem total_songs (a t : ℕ) (h_alina : alina_songs a) (h_tina : tina_songs t) : 
  (11 + a + t) % 3 = 0 → (7 + 4 + a + t) / 3 = 7 := by
  sorry

end total_songs_l163_163061


namespace average_marks_l163_163349

noncomputable def TatuyaScore (IvannaScore : ℝ) : ℝ :=
2 * IvannaScore

noncomputable def IvannaScore (DorothyScore : ℝ) : ℝ :=
(3/5) * DorothyScore

noncomputable def DorothyScore : ℝ := 90

noncomputable def XanderScore (TatuyaScore IvannaScore DorothyScore : ℝ) : ℝ :=
((TatuyaScore + IvannaScore + DorothyScore) / 3) + 10

noncomputable def SamScore (IvannaScore : ℝ) : ℝ :=
(3.8 * IvannaScore) + 5.5

noncomputable def OliviaScore (SamScore : ℝ) : ℝ :=
(3/2) * SamScore

theorem average_marks :
  let I := IvannaScore DorothyScore
  let T := TatuyaScore I
  let S := SamScore I
  let O := OliviaScore S
  let X := XanderScore T I DorothyScore
  let total_marks := T + I + DorothyScore + X + O + S
  (total_marks / 6) = 145.458333 := by sorry

end average_marks_l163_163349


namespace product_xyz_42_l163_163577

theorem product_xyz_42 (x y z : ℝ) 
  (h1 : (x - 2)^2 + (y - 3)^2 + (z - 4)^2 = 9)
  (h2 : x + y + z = 12) : x * y * z = 42 :=
by
  sorry

end product_xyz_42_l163_163577


namespace binom_expansion_l163_163093

/-- Given the binomial expansion of (sqrt(x) + 3x)^n for n < 15, 
    with the binomial coefficients of the 9th, 10th, and 11th terms forming an arithmetic sequence,
    we conclude that n must be 14 and describe all the rational terms in the expansion.
-/
theorem binom_expansion (n : ℕ) (h : n < 15)
  (h_seq : Nat.choose n 8 + Nat.choose n 10 = 2 * Nat.choose n 9) :
  n = 14 ∧
  (∃ (t1 t2 t3 : ℕ), 
    (t1 = 1 ∧ (Nat.choose 14 0 : ℕ) * (x ^ 7 : ℤ) = x ^ 7) ∧
    (t2 = 164 ∧ (Nat.choose 14 6 : ℕ) * (x ^ 6 : ℤ) = 164 * x ^ 6) ∧
    (t3 = 91 ∧ (Nat.choose 14 12 : ℕ) * (x ^ 5 : ℤ) = 91 * x ^ 5)) := 
  sorry

end binom_expansion_l163_163093


namespace _l163_163910

noncomputable def charlesPictures : Prop :=
  ∀ (bought : ℕ) (drew_today : ℕ) (drew_yesterday_after_work : ℕ) (left : ℕ),
    (bought = 20) →
    (drew_today = 6) →
    (drew_yesterday_after_work = 6) →
    (left = 2) →
    (bought - left - drew_today - drew_yesterday_after_work = 6)

-- We can use this statement "charlesPictures" to represent the theorem to be proved in Lean 4.

end _l163_163910


namespace prove_frac_addition_l163_163702

def frac_addition_correct : Prop :=
  (3 / 8 + 9 / 12 = 9 / 8)

theorem prove_frac_addition : frac_addition_correct :=
  by
  -- We assume the necessary fractions and their properties.
  sorry

end prove_frac_addition_l163_163702


namespace probability_of_winning_reward_l163_163569

-- Definitions representing the problem conditions
def red_envelopes : ℕ := 4
def card_types : ℕ := 3

-- Theorem statement: Prove the probability of winning the reward is 4/9
theorem probability_of_winning_reward : 
  (∃ (n m : ℕ), n = card_types^red_envelopes ∧ m = (Nat.choose red_envelopes 2) * (Nat.factorial 3)) → 
  (m / n = 4/9) :=
by
  sorry  -- Proof to be filled in

end probability_of_winning_reward_l163_163569


namespace average_minutes_per_day_l163_163892

theorem average_minutes_per_day (e : ℕ) (h_e_pos : 0 < e) : 
  let sixth_grade_minutes := 20
  let seventh_grade_minutes := 18
  let eighth_grade_minutes := 12
  
  let sixth_graders := 3 * e
  let seventh_graders := 4 * e
  let eighth_graders := e
  
  let total_minutes := sixth_grade_minutes * sixth_graders + seventh_grade_minutes * seventh_graders + eighth_grade_minutes * eighth_graders
  let total_students := sixth_graders + seventh_graders + eighth_graders
  
  (total_minutes / total_students) = 18 := by
sorry

end average_minutes_per_day_l163_163892


namespace percentage_born_in_september_l163_163412

theorem percentage_born_in_september (total famous : ℕ) (born_in_september : ℕ) (h1 : total = 150) (h2 : born_in_september = 12) :
  (born_in_september * 100 / total) = 8 :=
by
  sorry

end percentage_born_in_september_l163_163412


namespace part1_part2_l163_163913

-- Part 1
theorem part1 (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := sorry

-- Part 2
theorem part2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a * b + b * c + c * a ≤ 1 / 3 := sorry

end part1_part2_l163_163913


namespace apple_juice_less_than_cherry_punch_l163_163703

def orange_punch : ℝ := 4.5
def total_punch : ℝ := 21
def cherry_punch : ℝ := 2 * orange_punch
def combined_punch : ℝ := orange_punch + cherry_punch
def apple_juice : ℝ := total_punch - combined_punch

theorem apple_juice_less_than_cherry_punch : cherry_punch - apple_juice = 1.5 := by
  sorry

end apple_juice_less_than_cherry_punch_l163_163703


namespace product_of_integers_between_sqrt_115_l163_163757

theorem product_of_integers_between_sqrt_115 :
  ∃ a b : ℕ, 100 < 115 ∧ 115 < 121 ∧ a = 10 ∧ b = 11 ∧ a * b = 110 := by
  sorry

end product_of_integers_between_sqrt_115_l163_163757


namespace solve_for_A_l163_163497

def spadesuit (A B : ℝ) : ℝ := 4*A + 3*B + 6

theorem solve_for_A (A : ℝ) : spadesuit A 5 = 79 → A = 14.5 :=
by
  intros h
  sorry

end solve_for_A_l163_163497


namespace bicycles_difference_on_october_1_l163_163924

def initial_inventory : Nat := 200
def february_decrease : Nat := 4
def march_decrease : Nat := 6
def april_decrease : Nat := 8
def may_decrease : Nat := 10
def june_decrease : Nat := 12
def july_decrease : Nat := 14
def august_decrease : Nat := 16 + 20
def september_decrease : Nat := 18
def shipment : Nat := 50

def total_decrease : Nat := february_decrease + march_decrease + april_decrease + may_decrease + june_decrease + july_decrease + august_decrease + september_decrease
def stock_increase : Nat := shipment
def net_decrease : Nat := total_decrease - stock_increase

theorem bicycles_difference_on_october_1 : initial_inventory - net_decrease = 58 := by
  sorry

end bicycles_difference_on_october_1_l163_163924


namespace new_shoes_last_for_two_years_l163_163027

theorem new_shoes_last_for_two_years :
  let cost_repair := 11.50
  let cost_new := 28.00
  let increase_factor := 1.2173913043478261
  (cost_new / ((increase_factor) * cost_repair)) ≠ 0 :=
by
  sorry

end new_shoes_last_for_two_years_l163_163027


namespace average_xy_l163_163522

theorem average_xy (x y : ℝ) 
  (h : (4 + 6 + 9 + x + y) / 5 = 20) : (x + y) / 2 = 40.5 :=
sorry

end average_xy_l163_163522


namespace jellybean_mass_l163_163192

noncomputable def cost_per_gram : ℚ := 7.50 / 250
noncomputable def mass_for_180_cents : ℚ := 1.80 / cost_per_gram

theorem jellybean_mass :
  mass_for_180_cents = 60 := 
  sorry

end jellybean_mass_l163_163192


namespace problem1_solve_eq_l163_163374

theorem problem1_solve_eq (x : ℝ) : x * (x - 5) = 3 * x - 15 ↔ (x = 5 ∨ x = 3) := by
  sorry

end problem1_solve_eq_l163_163374


namespace jenna_hike_duration_l163_163106

-- Definitions from conditions
def initial_speed : ℝ := 25
def exhausted_speed : ℝ := 10
def total_distance : ℝ := 140
def total_time : ℝ := 8

-- The statement to prove:
theorem jenna_hike_duration : ∃ x : ℝ, 25 * x + 10 * (8 - x) = 140 ∧ x = 4 := by
  sorry

end jenna_hike_duration_l163_163106


namespace tv_cost_l163_163874

theorem tv_cost (savings : ℕ) (fraction_spent_on_furniture : ℚ) (amount_spent_on_furniture : ℚ) (remaining_savings : ℚ) :
  savings = 1000 →
  fraction_spent_on_furniture = 3/5 →
  amount_spent_on_furniture = fraction_spent_on_furniture * savings →
  remaining_savings = savings - amount_spent_on_furniture →
  remaining_savings = 400 :=
by
  sorry

end tv_cost_l163_163874


namespace min_value_of_expression_l163_163988

theorem min_value_of_expression (a b c : ℝ) (hb : b > a) (ha : a > c) (hc : b ≠ 0) :
  ∃ l : ℝ, l = 5.5 ∧ l ≤ (a + b)^2 / b^2 + (b + c)^2 / b^2 + (c + a)^2 / b^2 :=
by
  sorry

end min_value_of_expression_l163_163988


namespace ratio_of_perimeters_l163_163288

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l163_163288


namespace variance_of_heights_l163_163878
-- Importing all necessary libraries

-- Define a list of heights
def heights : List ℕ := [160, 162, 159, 160, 159]

-- Define the function to calculate the mean of a list of natural numbers
def mean (list : List ℕ) : ℚ :=
  list.sum / list.length

-- Define the function to calculate the variance of a list of natural numbers
def variance (list : List ℕ) : ℚ :=
  let μ := mean list
  (list.map (λ x => (x - μ) ^ 2)).sum / list.length

-- The theorem statement that proves the variance is 6/5
theorem variance_of_heights : variance heights = 6 / 5 :=
  sorry

end variance_of_heights_l163_163878


namespace work_completion_by_b_l163_163987

theorem work_completion_by_b (a_days : ℕ) (a_solo_days : ℕ) (a_b_combined_days : ℕ) (b_days : ℕ) :
  a_days = 12 ∧ a_solo_days = 3 ∧ a_b_combined_days = 5 → b_days = 15 :=
by
  sorry

end work_completion_by_b_l163_163987


namespace walked_8_miles_if_pace_4_miles_per_hour_l163_163507

-- Define the conditions
def walked_some_miles_in_2_hours (d : ℝ) : Prop :=
  d = 2

def pace_same_4_miles_per_hour (p : ℝ) : Prop :=
  p = 4

-- Define the proof problem
theorem walked_8_miles_if_pace_4_miles_per_hour :
  ∀ (d p : ℝ), walked_some_miles_in_2_hours d → pace_same_4_miles_per_hour p → (p * d = 8) :=
by
  intros d p h1 h2
  rw [h1, h2]
  exact sorry

end walked_8_miles_if_pace_4_miles_per_hour_l163_163507


namespace complement_event_A_l163_163199

def is_at_least_two_defective (n : ℕ) : Prop :=
  n ≥ 2

def is_at_most_one_defective (n : ℕ) : Prop :=
  n ≤ 1

theorem complement_event_A (n : ℕ) :
  (¬ is_at_least_two_defective n) ↔ is_at_most_one_defective n :=
by
  sorry

end complement_event_A_l163_163199


namespace porch_length_is_6_l163_163258

-- Define the conditions for the house and porch areas
def house_length : ℝ := 20.5
def house_width : ℝ := 10
def porch_width : ℝ := 4.5
def total_shingle_area : ℝ := 232

-- Define the area calculations
def house_area : ℝ := house_length * house_width
def porch_area : ℝ := total_shingle_area - house_area

-- The theorem to prove
theorem porch_length_is_6 : porch_area / porch_width = 6 := by
  sorry

end porch_length_is_6_l163_163258


namespace range_of_x_l163_163556

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2
  else Real.log (-x) / Real.log (1 / 2)

theorem range_of_x (x : ℝ) : f x > f (-x) ↔ (x > 1) ∨ (-1 < x ∧ x < 0) :=
by
  sorry

end range_of_x_l163_163556


namespace find_ordered_pairs_l163_163447

theorem find_ordered_pairs (a b x : ℕ) (h1 : b > a) (h2 : a + b = 15) (h3 : (a - 2 * x) * (b - 2 * x) = 2 * a * b / 3) :
  (a, b) = (8, 7) :=
by
  sorry

end find_ordered_pairs_l163_163447


namespace cost_of_tax_free_items_l163_163694

/-- 
Daniel went to a shop and bought items worth Rs 25, including a 30 paise sales tax on taxable items
with a tax rate of 10%. Prove that the cost of tax-free items is Rs 22.
-/
theorem cost_of_tax_free_items (total_spent taxable_amount sales_tax rate : ℝ)
  (h1 : total_spent = 25)
  (h2 : sales_tax = 0.3)
  (h3 : rate = 0.1)
  (h4 : taxable_amount = sales_tax / rate) :
  (total_spent - taxable_amount = 22) :=
by
  sorry

end cost_of_tax_free_items_l163_163694


namespace central_angle_of_regular_hexagon_l163_163827

-- Define the total degrees in a circle
def total_degrees_in_circle : ℝ := 360

-- Define the number of sides in a regular hexagon
def sides_in_hexagon : ℕ := 6

-- Theorems to prove that the central angle of a regular hexagon is 60°
theorem central_angle_of_regular_hexagon :
  total_degrees_in_circle / sides_in_hexagon = 60 :=
by
  sorry

end central_angle_of_regular_hexagon_l163_163827


namespace material_for_7_quilts_l163_163761

theorem material_for_7_quilts (x : ℕ) (h1 : ∀ y : ℕ, y = 7 * x) (h2 : 36 = 12 * x) : 7 * x = 21 := 
by 
  sorry

end material_for_7_quilts_l163_163761


namespace find_divisor_l163_163416

-- Defining the conditions
def dividend : ℕ := 181
def quotient : ℕ := 9
def remainder : ℕ := 1

-- The statement to prove
theorem find_divisor : ∃ (d : ℕ), dividend = (d * quotient) + remainder ∧ d = 20 := by
  sorry

end find_divisor_l163_163416


namespace programmer_debugging_hours_l163_163746

theorem programmer_debugging_hours 
  (total_hours : ℕ)
  (flow_chart_fraction coding_fraction : ℚ)
  (flow_chart_fraction_eq : flow_chart_fraction = 1/4)
  (coding_fraction_eq : coding_fraction = 3/8)
  (hours_worked : total_hours = 48) :
  ∃ (debugging_hours : ℚ), debugging_hours = 18 := 
by
  sorry

end programmer_debugging_hours_l163_163746


namespace min_value_l163_163875

def f (x y : ℝ) : ℝ := x^2 + 4 * x * y + 5 * y^2 - 10 * x - 6 * y + 3

theorem min_value : ∃ x y : ℝ, (x + y = 2) ∧ (f x y = -(1/7)) :=
by
  sorry

end min_value_l163_163875


namespace speed_of_stream_l163_163303

-- Definitions based on conditions
def boat_speed_still_water : ℕ := 24
def travel_time : ℕ := 4
def downstream_distance : ℕ := 112

-- Theorem statement
theorem speed_of_stream : 
  ∀ (v : ℕ), downstream_distance = travel_time * (boat_speed_still_water + v) → v = 4 :=
by
  intros v h
  -- Proof omitted
  sorry

end speed_of_stream_l163_163303


namespace polar_to_cartesian_parabola_l163_163576

theorem polar_to_cartesian_parabola (r θ : ℝ) (h : r = 1 / (1 - Real.sin θ)) :
  ∃ x y : ℝ, x^2 = 2 * y + 1 :=
by
  sorry

end polar_to_cartesian_parabola_l163_163576


namespace angle_measure_is_fifty_l163_163680

theorem angle_measure_is_fifty (x : ℝ) :
  (90 - x = (1 / 2) * (180 - x) - 25) → x = 50 := by
  intro h
  sorry

end angle_measure_is_fifty_l163_163680


namespace percent_non_unionized_women_is_80_l163_163377

noncomputable def employeeStatistics :=
  let total_employees := 100
  let percent_men := 50
  let percent_unionized := 60
  let percent_unionized_men := 70
  let men := (percent_men / 100) * total_employees
  let unionized := (percent_unionized / 100) * total_employees
  let unionized_men := (percent_unionized_men / 100) * unionized
  let non_unionized_men := men - unionized_men
  let non_unionized := total_employees - unionized
  let non_unionized_women := non_unionized - non_unionized_men
  let percent_non_unionized_women := (non_unionized_women / non_unionized) * 100
  percent_non_unionized_women

theorem percent_non_unionized_women_is_80 :
  employeeStatistics = 80 :=
by
  sorry

end percent_non_unionized_women_is_80_l163_163377


namespace postage_stamp_problem_l163_163466

theorem postage_stamp_problem
  (x y z : ℕ) (h1: y = 10 * x) (h2: x + 2 * y + 5 * z = 100) :
  x = 5 ∧ y = 50 ∧ z = 0 :=
by
  sorry

end postage_stamp_problem_l163_163466


namespace evaluate_expression_l163_163571

theorem evaluate_expression :
  ∀ (a b c : ℚ),
  c = b + 1 →
  b = a + 5 →
  a = 3 →
  (a + 2 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (a + 3) * (b + 1) * (c + 9) / ((a + 2) * (b - 3) * (c + 7)) = 2.43 := 
by
  intros a b c hc hb ha h1 h2 h3
  sorry

end evaluate_expression_l163_163571


namespace transform_equation_to_polynomial_l163_163876

variable (x y : ℝ)

theorem transform_equation_to_polynomial (h : (x^2 + 2) / (x + 1) = y) :
    (x^2 + 2) / (x + 1) + (5 * (x + 1)) / (x^2 + 2) = 6 → y^2 - 6 * y + 5 = 0 :=
by
  intro h_eq
  sorry

end transform_equation_to_polynomial_l163_163876


namespace trapezoid_base_length_l163_163824

-- Definitions from the conditions
def trapezoid_area (a b h : ℕ) : ℕ := (1 / 2) * (a + b) * h

theorem trapezoid_base_length (b : ℕ) (h : ℕ) (a : ℕ) (A : ℕ) (H_area : A = 222) (H_upper_side : a = 23) (H_height : h = 12) :
  A = trapezoid_area a b h ↔ b = 14 :=
by sorry

end trapezoid_base_length_l163_163824


namespace solve_congruence_l163_163406

-- Define the initial condition of the problem
def condition (x : ℤ) : Prop := (15 * x + 3) % 21 = 9 % 21

-- The statement that we want to prove
theorem solve_congruence : ∃ (a m : ℤ), condition a ∧ a % m = 6 % 7 ∧ a < m ∧ a + m = 13 :=
by {
    sorry
}

end solve_congruence_l163_163406


namespace sqrt_domain_l163_163638

theorem sqrt_domain (x : ℝ) : 1 - x ≥ 0 → x ≤ 1 := by
  sorry

end sqrt_domain_l163_163638


namespace geometric_seq_sum_l163_163807

theorem geometric_seq_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, S n = a 0 * (1 - q^n) / (1 - q))
    (h2 : S 10 = 10) (h3 : S 30 = 70) (hq_pos : 0 < q) :
    S 40 = 150 := by
  sorry

end geometric_seq_sum_l163_163807


namespace total_distance_of_race_is_150_l163_163011

variable (D : ℝ)

-- Conditions
def A_covers_distance_in_45_seconds (D : ℝ) : Prop := ∃ A_speed, A_speed = D / 45
def B_covers_distance_in_60_seconds (D : ℝ) : Prop := ∃ B_speed, B_speed = D / 60
def A_beats_B_by_50_meters_in_60_seconds (D : ℝ) : Prop := (D / 45) * 60 = D + 50

theorem total_distance_of_race_is_150 :
  A_covers_distance_in_45_seconds D ∧ 
  B_covers_distance_in_60_seconds D ∧ 
  A_beats_B_by_50_meters_in_60_seconds D → 
  D = 150 :=
by
  sorry

end total_distance_of_race_is_150_l163_163011


namespace minimum_value_of_z_l163_163595

theorem minimum_value_of_z :
  ∀ (x y : ℝ), ∃ z : ℝ, z = 2*x^2 + 3*y^2 + 8*x - 6*y + 35 ∧ z ≥ 24 := by
  sorry

end minimum_value_of_z_l163_163595


namespace min_bounces_for_height_less_than_two_l163_163472

theorem min_bounces_for_height_less_than_two : 
  ∃ (k : ℕ), (20 * (3 / 4 : ℝ)^k < 2 ∧ ∀ n < k, ¬(20 * (3 / 4 : ℝ)^n < 2)) :=
sorry

end min_bounces_for_height_less_than_two_l163_163472


namespace high_fever_temperature_l163_163410

theorem high_fever_temperature (T t : ℝ) (h1 : T = 36) (h2 : t > 13 / 12 * T) : t > 39 :=
by
  sorry

end high_fever_temperature_l163_163410


namespace student_fraction_mistake_l163_163168

theorem student_fraction_mistake (n : ℕ) (h_n : n = 576) 
(h_mistake : ∃ r : ℚ, r * n = (5 / 16) * n + 300) : ∃ r : ℚ, r = 5 / 6 :=
by
  sorry

end student_fraction_mistake_l163_163168


namespace units_digit_of_6_pow_5_l163_163659

theorem units_digit_of_6_pow_5 : (6^5 % 10) = 6 := 
by sorry

end units_digit_of_6_pow_5_l163_163659


namespace trajectory_equation_l163_163992

open Real

-- Define points M and N
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the moving point P
def P (x y : ℝ) : Prop := 
  (4 * Real.sqrt ((x + 2) ^ 2 + y ^ 2) + 4 * (x - 2) = 0) → 
  (y ^ 2 = -8 * x)

-- The theorem stating the desired proof problem
theorem trajectory_equation (x y : ℝ) : P x y :=
sorry

end trajectory_equation_l163_163992


namespace sadies_average_speed_l163_163188

def sadie_time : ℝ := 2
def ariana_speed : ℝ := 6
def ariana_time : ℝ := 0.5
def sarah_speed : ℝ := 4
def total_time : ℝ := 4.5
def total_distance : ℝ := 17

theorem sadies_average_speed :
  ((total_distance - ((ariana_speed * ariana_time) + (sarah_speed * (total_time - sadie_time - ariana_time)))) / sadie_time) = 3 := 
by sorry

end sadies_average_speed_l163_163188


namespace sum_of_integers_is_18_l163_163894

theorem sum_of_integers_is_18 (a b : ℕ) (h1 : b = 2 * a) (h2 : a * b + a + b = 156) (h3 : Nat.gcd a b = 1) (h4 : a < 25) : a + b = 18 :=
by
  sorry

end sum_of_integers_is_18_l163_163894


namespace relationship_coefficients_l163_163918

-- Definitions based directly on the conditions
def has_extrema (a b c : ℝ) : Prop := b^2 - 3 * a * c > 0
def passes_through_origin (x1 x2 y1 y2 : ℝ) : Prop := x1 * y2 = x2 * y1

-- Main statement proving the relationship among the coefficients
theorem relationship_coefficients (a b c d : ℝ) (x1 x2 y1 y2 : ℝ)
  (h_extrema : has_extrema a b c)
  (h_line : passes_through_origin x1 x2 y1 y2)
  (hx1 : x1 ≠ 0) (hx2 : x2 ≠ 0)
  (h_y1 : y1 = a * x1^3 + b * x1^2 + c * x1 + d)
  (h_y2 : y2 = a * x2^3 + b * x2^2 + c * x2 + d) :
  9 * a * d = b * c :=
sorry

end relationship_coefficients_l163_163918


namespace algebraic_expression_evaluation_l163_163040

theorem algebraic_expression_evaluation (x y : ℝ) : 
  3 * (x^2 - 2 * x * y + y^2) - 3 * (x^2 - 2 * x * y + y^2 - 1) = 3 :=
by
  sorry

end algebraic_expression_evaluation_l163_163040


namespace part_a_part_b_l163_163155

-- Part (a)
theorem part_a (students : Fin 67) (answers : Fin 6 → Bool) :
  ∃ (s1 s2 : Fin 67), s1 ≠ s2 ∧ answers s1 = answers s2 := by
  sorry

-- Part (b)
theorem part_b (students : Fin 67) (points : Fin 6 → ℤ)
  (h_points : ∀ k, points k = k ∨ points k = -k) :
  ∃ (scores : Fin 67 → ℤ), ∃ (s1 s2 s3 s4 : Fin 67),
  s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4 ∧
  scores s1 = scores s2 ∧ scores s1 = scores s3 ∧ scores s1 = scores s4 := by
  sorry

end part_a_part_b_l163_163155


namespace sqrt_neg2_sq_l163_163134

theorem sqrt_neg2_sq : Real.sqrt ((-2 : ℝ) ^ 2) = 2 := by
  sorry

end sqrt_neg2_sq_l163_163134


namespace gallons_left_l163_163488

theorem gallons_left (initial_gallons : ℚ) (gallons_given : ℚ) (gallons_left : ℚ) : 
  initial_gallons = 4 ∧ gallons_given = 16/3 → gallons_left = -4/3 :=
by
  sorry

end gallons_left_l163_163488


namespace cristine_initial_lemons_l163_163010

theorem cristine_initial_lemons (L : ℕ) (h : (3 / 4 : ℚ) * L = 9) : L = 12 :=
sorry

end cristine_initial_lemons_l163_163010


namespace find_r_l163_163867

noncomputable def g (x : ℝ) (p q r : ℝ) := x^3 + p * x^2 + q * x + r

theorem find_r 
  (p q r : ℝ) 
  (h1 : ∀ x : ℝ, g x p q r = (x + 100) * (x + 0) * (x + 0))
  (h2 : p + q + r = 100) : 
  r = 0 := 
by
  sorry

end find_r_l163_163867


namespace minimum_value_of_2m_plus_n_solution_set_for_inequality_l163_163514

namespace MathProof

-- Definitions and conditions
def f (x m n : ℝ) : ℝ := |x + m| + |2 * x - n|

-- Part (I)
theorem minimum_value_of_2m_plus_n
  (m n : ℝ)
  (h_mn_pos : m > 0 ∧ n > 0)
  (h_f_nonneg : ∀ x : ℝ, f x m n ≥ 1) :
  2 * m + n ≥ 2 :=
sorry

-- Part (II)
theorem solution_set_for_inequality
  (x : ℝ) :
  (f x 2 3 > 5 ↔ (x < 0 ∨ x > 2)) :=
sorry

end MathProof

end minimum_value_of_2m_plus_n_solution_set_for_inequality_l163_163514


namespace distance_between_places_l163_163531

theorem distance_between_places
  (d : ℝ) -- let d be the distance between A and B
  (v : ℝ) -- let v be the original speed
  (h1 : v * 4 = d) -- initially, speed * time = distance
  (h2 : (v + 20) * 3 = d) -- after speed increase, speed * new time = distance
  : d = 240 :=
sorry

end distance_between_places_l163_163531


namespace dark_squares_more_than_light_l163_163302

/--
A 9 × 9 board is composed of alternating dark and light squares, with the upper-left square being dark.
Prove that there is exactly 1 more dark square than light square.
-/
theorem dark_squares_more_than_light :
  let board_size := 9
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 4 * 5 + 5 * 4
  dark_squares - light_squares = 1 :=
by
  let board_size := 9
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 4 * 5 + 5 * 4
  show dark_squares - light_squares = 1
  sorry

end dark_squares_more_than_light_l163_163302


namespace meaningful_fraction_l163_163069

theorem meaningful_fraction (x : ℝ) : (∃ y, y = (1 / (x - 2))) ↔ x ≠ 2 :=
by
  sorry

end meaningful_fraction_l163_163069


namespace number_of_lockers_l163_163056

-- Problem Conditions
def locker_numbers_consecutive_from_one := ∀ (n : ℕ), n ≥ 1
def cost_per_digit := 0.02
def total_cost := 137.94

-- Theorem Statement
theorem number_of_lockers (h1 : locker_numbers_consecutive_from_one) (h2 : cost_per_digit = 0.02) (h3 : total_cost = 137.94) : ∃ n : ℕ, n = 2001 :=
sorry

end number_of_lockers_l163_163056


namespace min_adults_at_amusement_park_l163_163013

def amusement_park_problem : Prop :=
  ∃ (x y z : ℕ), 
    x + y + z = 100 ∧
    3 * x + 2 * y + (3 / 10) * z = 100 ∧
    (∀ (x' : ℕ), x' < 2 → ¬(∃ (y' z' : ℕ), x' + y' + z' = 100 ∧ 3 * x' + 2 * y' + (3 / 10) * z' = 100))

theorem min_adults_at_amusement_park : amusement_park_problem := sorry

end min_adults_at_amusement_park_l163_163013


namespace ellie_loan_difference_l163_163402

noncomputable def principal : ℝ := 8000
noncomputable def simple_rate : ℝ := 0.10
noncomputable def compound_rate : ℝ := 0.08
noncomputable def time : ℝ := 5
noncomputable def compounding_periods : ℝ := 1

noncomputable def simple_interest_total (P r t : ℝ) : ℝ :=
  P + (P * r * t)

noncomputable def compound_interest_total (P r t n : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem ellie_loan_difference :
  (compound_interest_total principal compound_rate time compounding_periods) -
  (simple_interest_total principal simple_rate time) = -245.36 := 
  by sorry

end ellie_loan_difference_l163_163402


namespace rowing_upstream_speed_l163_163467

theorem rowing_upstream_speed (Vm Vdown : ℝ) (H1 : Vm = 20) (H2 : Vdown = 33) :
  ∃ Vup Vs : ℝ, Vup = Vm - Vs ∧ Vs = Vdown - Vm ∧ Vup = 7 := 
by {
  sorry
}

end rowing_upstream_speed_l163_163467


namespace evaluate_expression_l163_163384

theorem evaluate_expression : (3200 - 3131) ^ 2 / 121 = 36 :=
by
  sorry

end evaluate_expression_l163_163384


namespace interest_equality_l163_163479

-- Definitions based on the conditions
def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

-- Constants for the problem
def P1 : ℝ := 200 -- 200 Rs is the principal of the first case
def r1 : ℝ := 0.1 -- 10% converted to a decimal
def t1 : ℝ := 12 -- 12 years

def P2 : ℝ := 1000 -- Correct answer for the other amount
def r2 : ℝ := 0.12 -- 12% converted to a decimal
def t2 : ℝ := 2 -- 2 years

-- Theorem stating that the interest generated is the same
theorem interest_equality : 
  simple_interest P1 r1 t1 = simple_interest P2 r2 t2 :=
by 
  -- Skip the proof since it is not required
  sorry

end interest_equality_l163_163479


namespace apples_in_box_l163_163142

-- Define the initial conditions
def oranges : ℕ := 12
def removed_oranges : ℕ := 6
def target_percentage : ℚ := 0.70

-- Define the function that models the problem
def fruit_after_removal (apples : ℕ) : ℕ := apples + (oranges - removed_oranges)
def apples_percentage (apples : ℕ) : ℚ := (apples : ℚ) / (fruit_after_removal apples : ℚ)

-- The theorem states the question and expected answer
theorem apples_in_box : ∃ (apples : ℕ), apples_percentage apples = target_percentage ∧ apples = 14 :=
by
  sorry

end apples_in_box_l163_163142


namespace shopkeeper_profit_percentage_l163_163737

theorem shopkeeper_profit_percentage (P : ℝ) : (70 / 100) * (1 + P / 100) = 1 → P = 700 / 3 :=
by
  sorry

end shopkeeper_profit_percentage_l163_163737


namespace trigonometric_identity_l163_163124

variable (α : Real)

theorem trigonometric_identity (h : Real.tan α = Real.sqrt 2) :
  (1/3) * Real.sin α^2 + Real.cos α^2 = 5/9 :=
sorry

end trigonometric_identity_l163_163124


namespace find_integer_a_l163_163426

-- Definitions based on the conditions
def in_ratio (x y z : ℕ) := ∃ k : ℕ, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k
def satisfies_equation (z : ℕ) (a : ℕ) := z = 30 * a - 15

-- The proof problem statement
theorem find_integer_a (x y z : ℕ) (a : ℕ) :
  in_ratio x y z →
  satisfies_equation z a →
  (∃ a : ℕ, a = 4) :=
by
  intros h1 h2
  sorry

end find_integer_a_l163_163426


namespace greatest_mondays_in_45_days_l163_163000

-- Define the days in a week
def days_in_week : ℕ := 7

-- Define the total days being considered
def total_days : ℕ := 45

-- Calculate the complete weeks in the total days
def complete_weeks : ℕ := total_days / days_in_week

-- Calculate the extra days
def extra_days : ℕ := total_days % days_in_week

-- Define that the period starts on Monday (condition)
def starts_on_monday : Bool := true

-- Prove that the greatest number of Mondays in the first 45 days is 7
theorem greatest_mondays_in_45_days (h1 : days_in_week = 7) (h2 : total_days = 45) (h3 : starts_on_monday = true) : 
  (complete_weeks + if starts_on_monday && extra_days >= 1 then 1 else 0) = 7 := 
by
  sorry

end greatest_mondays_in_45_days_l163_163000


namespace similar_triangle_shortest_side_l163_163173

theorem similar_triangle_shortest_side (a b c : ℕ) (H1 : a^2 + b^2 = c^2) (H2 : a = 15) (H3 : c = 34) (H4 : b = Int.sqrt 931) : 
  ∃ d : ℝ, d = 3 * Int.sqrt 931 ∧ d = 102  :=
by
  sorry

end similar_triangle_shortest_side_l163_163173


namespace isosceles_triangle_perimeter_l163_163162

-- Defining the given conditions
def is_isosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ c = a
def triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

-- Stating the problem and goal
theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h_iso: is_isosceles a b c)
  (h_len1: a = 3 ∨ a = 6)
  (h_len2: b = 3 ∨ b = 6)
  (h_triangle: triangle a b c): a + b + c = 15 :=
sorry

end isosceles_triangle_perimeter_l163_163162


namespace john_bought_three_sodas_l163_163919

-- Define the conditions

def cost_per_soda := 2
def total_money_paid := 20
def change_received := 14

-- Definition indicating the number of sodas bought
def num_sodas_bought := (total_money_paid - change_received) / cost_per_soda

-- Question: Prove that John bought 3 sodas given these conditions
theorem john_bought_three_sodas : num_sodas_bought = 3 := by
  -- Proof: This is an example of how you may structure the proof
  sorry

end john_bought_three_sodas_l163_163919


namespace find_initial_volume_l163_163778

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

end find_initial_volume_l163_163778


namespace negation_p_equiv_l163_163477

noncomputable def negation_of_proposition_p : Prop :=
∀ m : ℝ, ¬ ∃ x : ℝ, x^2 + m * x + 1 = 0

theorem negation_p_equiv (p : Prop) (h : p = ∃ m : ℝ, ∃ x : ℝ, x^2 + m * x + 1 = 0) :
  ¬ p ↔ negation_of_proposition_p :=
by {
  sorry
}

end negation_p_equiv_l163_163477


namespace remainder_is_210_l163_163818

-- Define necessary constants and theorems
def x : ℕ := 2^35
def dividend : ℕ := 2^210 + 210
def divisor : ℕ := 2^105 + 2^63 + 1

theorem remainder_is_210 : (dividend % divisor) = 210 :=
by 
  -- Assume the calculation steps in the preceding solution are correct.
  -- No need to manually re-calculate as we've directly taken from the solution.
  sorry

end remainder_is_210_l163_163818


namespace symmetric_angles_l163_163698

theorem symmetric_angles (α β : ℝ) (k : ℤ) (h : α + β = 2 * k * Real.pi) : α = 2 * k * Real.pi - β :=
by
  sorry

end symmetric_angles_l163_163698


namespace AY_is_2_sqrt_55_l163_163700

noncomputable def AY_length : ℝ :=
  let rA := 10
  let rB := 3
  let AB := rA + rB
  let AD := rA - rB
  let BD := Real.sqrt (AB^2 - AD^2)
  2 * Real.sqrt (rA^2 + BD^2)

theorem AY_is_2_sqrt_55 :
  AY_length = 2 * Real.sqrt 55 :=
by
  -- Assuming the given problem's conditions.
  let rA := 10
  let rB := 3
  let AB := rA + rB
  let AD := rA - rB
  let BD := Real.sqrt (AB^2 - AD^2)
  show AY_length = 2 * Real.sqrt 55
  sorry

end AY_is_2_sqrt_55_l163_163700


namespace equation_of_line_AB_l163_163506

-- Definition of the given circle
def circle1 : Type := { p : ℝ × ℝ // p.1^2 + (p.2 - 2)^2 = 4 }

-- Definition of the center and point on the second circle
def center : ℝ × ℝ := (0, 2)
def point : ℝ × ℝ := (-2, 6)

-- Definition of the second circle with diameter endpoints
def circle2_eq (x y : ℝ) : Prop := (x + 1)^2 + (y - 4)^2 = 5

-- Statement to be proved
theorem equation_of_line_AB :
  ∃ x y : ℝ, (x^2 + (y - 2)^2 = 4) ∧ ((x + 1)^2 + (y - 4)^2 = 5) ∧ (x - 2*y + 6 = 0) := 
sorry

end equation_of_line_AB_l163_163506


namespace Ram_Gohul_days_work_together_l163_163395

-- Define the conditions
def Ram_days := 10
def Gohul_days := 15

-- Define the work rates
def Ram_rate := 1 / Ram_days
def Gohul_rate := 1 / Gohul_days

-- Define the combined work rate
def Combined_rate := Ram_rate + Gohul_rate

-- Define the number of days to complete the job together
def Together_days := 1 / Combined_rate

-- State the proof problem
theorem Ram_Gohul_days_work_together : Together_days = 6 := by
  sorry

end Ram_Gohul_days_work_together_l163_163395


namespace ron_spends_on_chocolate_bars_l163_163548

/-- Ron is hosting a camp for 15 scouts where each scout needs 2 s'mores.
    Each chocolate bar costs $1.50 and can be broken into 3 sections to make 3 s'mores.
    A discount of 15% applies if 10 or more chocolate bars are purchased.
    Calculate the total amount Ron will spend on chocolate bars after applying the discount if applicable. -/
theorem ron_spends_on_chocolate_bars :
  let cost_per_bar := 1.5
  let s'mores_per_bar := 3
  let scouts := 15
  let s'mores_per_scout := 2
  let total_s'mores := scouts * s'mores_per_scout
  let bars_needed := total_s'mores / s'mores_per_bar
  let discount := 0.15
  let total_cost := bars_needed * cost_per_bar
  let discount_amount := if bars_needed >= 10 then discount * total_cost else 0
  let final_cost := total_cost - discount_amount
  final_cost = 12.75 := by sorry

end ron_spends_on_chocolate_bars_l163_163548


namespace boys_cannot_score_twice_as_girls_l163_163586

theorem boys_cannot_score_twice_as_girls :
  ∀ (participants : Finset ℕ) (boys girls : ℕ) (points : ℕ → ℝ),
    participants.card = 6 →
    boys = 2 →
    girls = 4 →
    (∀ p, p ∈ participants → points p = 1 ∨ points p = 0.5 ∨ points p = 0) →
    (∀ (p q : ℕ), p ∈ participants → q ∈ participants → p ≠ q → points p + points q = 1) →
    ¬ (∃ (boys_points girls_points : ℝ), 
      (∀ b ∈ (Finset.range 2), boys_points = points b) ∧
      (∀ g ∈ (Finset.range 4), girls_points = points g) ∧
      boys_points = 2 * girls_points) :=
by
  sorry

end boys_cannot_score_twice_as_girls_l163_163586


namespace hyperbola_sum_l163_163250

theorem hyperbola_sum (h k a b : ℝ) (c : ℝ)
  (h_eq : h = 3)
  (k_eq : k = -5)
  (a_eq : a = 5)
  (c_eq : c = 7)
  (c_squared_eq : c^2 = a^2 + b^2) :
  h + k + a + b = 3 + 2 * Real.sqrt 6 :=
by
  rw [h_eq, k_eq, a_eq, c_eq] at *
  sorry

end hyperbola_sum_l163_163250


namespace negation_of_statement_l163_163718

theorem negation_of_statement :
  ¬(∀ x : ℝ, ∃ n : ℕ, 0 < n ∧ n > x^2) ↔ (∃ x : ℝ, ∀ n : ℕ, 0 < n → n < x^2) := by
sorry

end negation_of_statement_l163_163718


namespace geom_sequence_a4_times_a7_l163_163396

theorem geom_sequence_a4_times_a7 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_q : q = 2) 
  (h_a2_a5 : a 2 * a 5 = 32) : 
  a 4 * a 7 = 512 :=
by 
  sorry

end geom_sequence_a4_times_a7_l163_163396


namespace weight_of_NH4I_H2O_l163_163896

noncomputable def total_weight (moles_NH4I : ℕ) (molar_mass_NH4I : ℝ) 
                             (moles_H2O : ℕ) (molar_mass_H2O : ℝ) : ℝ :=
  (moles_NH4I * molar_mass_NH4I) + (moles_H2O * molar_mass_H2O)

theorem weight_of_NH4I_H2O :
  total_weight 15 144.95 7 18.02 = 2300.39 :=
by
  sorry

end weight_of_NH4I_H2O_l163_163896


namespace apples_and_oranges_l163_163398

theorem apples_and_oranges :
  ∃ x y : ℝ, 2 * x + 3 * y = 6 ∧ 4 * x + 7 * y = 13 ∧ (16 * x + 23 * y = 47) :=
by
  sorry

end apples_and_oranges_l163_163398


namespace quadratic_real_roots_condition_l163_163948

theorem quadratic_real_roots_condition (a b c : ℝ) (q : b^2 - 4 * a * c ≥ 0) (h : a ≠ 0) : 
  (b^2 - 4 * a * c ≥ 0 ∧ a ≠ 0) ↔ ((∃ x1 x2 : ℝ, a * x1 ^ 2 + b * x1 + c = 0 ∧ a * x2 ^ 2 + b * x2 + c = 0) ∨ (∃ x : ℝ, a * x ^ 2 + b * x + c = 0)) :=
by
  sorry

end quadratic_real_roots_condition_l163_163948


namespace florist_sold_roses_l163_163838

theorem florist_sold_roses (x : ℕ) (h1 : 5 - x + 34 = 36) : x = 3 :=
by sorry

end florist_sold_roses_l163_163838


namespace original_cost_price_l163_163264

theorem original_cost_price 
  (C SP SP_new C_new : ℝ)
  (h1 : SP = 1.05 * C)
  (h2 : C_new = 0.95 * C)
  (h3 : SP_new = SP - 8)
  (h4 : SP_new = 1.045 * C_new) :
  C = 1600 :=
by
  sorry

end original_cost_price_l163_163264


namespace probability_of_getting_a_prize_l163_163008

theorem probability_of_getting_a_prize {prizes blanks : ℕ} (h_prizes : prizes = 10) (h_blanks : blanks = 25) :
  (prizes / (prizes + blanks) : ℚ) = 2 / 7 :=
by
  sorry

end probability_of_getting_a_prize_l163_163008


namespace complement_union_correct_l163_163789

-- Define the universal set U, set A, and set B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- The complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The union of the complement of A and set B
def union_complement_U_A_B : Set ℕ := complement_U_A ∪ B

-- State the theorem to prove
theorem complement_union_correct : union_complement_U_A_B = {2, 3, 4, 5} := 
by 
  sorry

end complement_union_correct_l163_163789


namespace isosceles_triangle_length_l163_163947

theorem isosceles_triangle_length (a : ℝ) (h_graph_A : ∃ y, (a, y) ∈ {p : ℝ × ℝ | p.snd = -p.fst^2})
  (h_graph_B : ∃ y, (-a, y) ∈ {p : ℝ × ℝ | p.snd = -p.fst^2}) 
  (h_isosceles : ∃ O : ℝ × ℝ, O = (0, 0) ∧ 
    dist (a, -a^2) O = dist (-a, -a^2) O ∧ dist (a, -a^2) (-a, -a^2) = dist (-a, -a^2) O) :
  dist (a, -a^2) (0, 0) = 2 * Real.sqrt 3 := sorry

end isosceles_triangle_length_l163_163947


namespace micah_has_seven_fish_l163_163203

-- Definitions from problem conditions
def micahFish (M : ℕ) : Prop :=
  let kennethFish := 3 * M
  let matthiasFish := kennethFish - 15
  M + kennethFish + matthiasFish = 34

-- Main statement: prove that the number of fish Micah has is 7
theorem micah_has_seven_fish : ∃ M : ℕ, micahFish M ∧ M = 7 :=
by
  sorry

end micah_has_seven_fish_l163_163203


namespace find_y_l163_163151

theorem find_y (x y : ℤ) (h1 : x + y = 250) (h2 : x - y = 200) : y = 25 :=
by
  sorry

end find_y_l163_163151


namespace Robert_photo_count_l163_163171

theorem Robert_photo_count (k : ℕ) (hLisa : ∃ n : ℕ, k = 8 * n) : k = 24 - 16 → k = 24 :=
by
  intro h
  sorry

end Robert_photo_count_l163_163171


namespace value_of_f_at_4_l163_163369

noncomputable def f (α : ℝ) (x : ℝ) := x^α

theorem value_of_f_at_4 : 
  (∃ α : ℝ, f α 2 = (Real.sqrt 2) / 2) → f (-1 / 2) 4 = 1 / 2 :=
by
  intros h
  sorry

end value_of_f_at_4_l163_163369


namespace sum_first_n_terms_arithmetic_sequence_l163_163599

/-- Define the arithmetic sequence with common difference d and a given term a₄. -/
def arithmetic_sequence (n : ℕ) (a₁ d : ℤ) : ℤ :=
  a₁ + (n - 1) * d

/-- Define the sum of the first n terms of an arithmetic sequence. -/
def sum_of_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * ((2 * a₁ + (n - 1) * d) / 2)

theorem sum_first_n_terms_arithmetic_sequence :
  ∀ n : ℕ, 
  ∀ a₁ : ℤ, 
  (∀ d, d = 2 → (∀ a₁, (a₁ + 3 * d = 8) → sum_of_arithmetic_sequence a₁ d n = (n : ℤ) * ((n : ℤ) + 1))) :=
by
  intros n a₁ d hd h₁
  sorry

end sum_first_n_terms_arithmetic_sequence_l163_163599


namespace cartesian_to_polar_curve_C_l163_163767

theorem cartesian_to_polar_curve_C (x y : ℝ) (θ ρ : ℝ) 
  (h1 : x = ρ * Real.cos θ)
  (h2 : y = ρ * Real.sin θ)
  (h3 : x^2 + y^2 - 2 * x = 0) : 
  ρ = 2 * Real.cos θ :=
sorry

end cartesian_to_polar_curve_C_l163_163767


namespace contrapositive_of_lt_l163_163526

theorem contrapositive_of_lt (a b c : ℝ) :
  (a < b → a + c < b + c) → (a + c ≥ b + c → a ≥ b) :=
by
  intro h₀ h₁
  sorry

end contrapositive_of_lt_l163_163526


namespace tan_double_angle_l163_163387

theorem tan_double_angle (α : ℝ) 
  (h : Real.tan α = 1 / 2) : Real.tan (2 * α) = 4 / 3 := 
by
  sorry

end tan_double_angle_l163_163387


namespace range_of_a_l163_163486

noncomputable def p (x : ℝ) : Prop := (3*x - 1)/(x - 2) ≤ 1
noncomputable def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) < 0

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, ¬ q x a) → (¬ ∃ x : ℝ, ¬ p x) → -1/2 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l163_163486


namespace projection_of_A_onto_Oxz_is_B_l163_163111

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def projection_onto_Oxz (A : Point3D) : Point3D :=
  { x := A.x, y := 0, z := A.z }

theorem projection_of_A_onto_Oxz_is_B :
  let A := Point3D.mk 2 3 6
  let B := Point3D.mk 2 0 6
  projection_onto_Oxz A = B :=
by
  let A := Point3D.mk 2 3 6
  let B := Point3D.mk 2 0 6
  have h : projection_onto_Oxz A = B := rfl
  exact h

end projection_of_A_onto_Oxz_is_B_l163_163111


namespace count_triangles_in_figure_l163_163813

-- Define the structure of the grid with the given properties.
def grid_structure : Prop :=
  ∃ (n1 n2 n3 n4 : ℕ), 
  n1 = 3 ∧  -- First row: 3 small triangles
  n2 = 2 ∧  -- Second row: 2 small triangles
  n3 = 1 ∧  -- Third row: 1 small triangle
  n4 = 1    -- 1 large inverted triangle

-- The problem statement
theorem count_triangles_in_figure (h : grid_structure) : 
  ∃ (total_triangles : ℕ), total_triangles = 9 :=
sorry

end count_triangles_in_figure_l163_163813


namespace reciprocal_relationship_l163_163337

theorem reciprocal_relationship (a b : ℚ)
  (h1 : a = (-7 / 8) / (7 / 4 - 7 / 8 - 7 / 12))
  (h2 : b = (7 / 4 - 7 / 8 - 7 / 12) / (-7 / 8)) :
  a = - 1 / b :=
by sorry

end reciprocal_relationship_l163_163337


namespace floor_expression_equals_zero_l163_163017

theorem floor_expression_equals_zero
  (a b c : ℕ)
  (ha : a = 2010)
  (hb : b = 2007)
  (hc : c = 2008) :
  Int.floor ((a^3 : ℚ) / (b * c^2) - (c^3 : ℚ) / (b^2 * a)) = 0 := 
  sorry

end floor_expression_equals_zero_l163_163017


namespace find_ordered_pairs_l163_163799

theorem find_ordered_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (2 * m ∣ 3 * n - 2 ∧ 2 * n ∣ 3 * m - 2) ↔ (m, n) = (2, 2) ∨ (m, n) = (10, 14) ∨ (m, n) = (14, 10) :=
by
  sorry

end find_ordered_pairs_l163_163799


namespace determine_color_sum_or_product_l163_163971

theorem determine_color_sum_or_product {x : ℕ → ℝ} (h_distinct: ∀ i j : ℕ, i < j → x i < x j) (x_pos : ∀ i : ℕ, x i > 0) :
  ∃ c : ℕ → ℝ, (∀ i : ℕ, c i > 0) ∧
  (∀ i j : ℕ, i < j → (∃ r1 r2 : ℕ, (r1 ≠ r2) ∧ (c r1 + c r2 = x₆₄ + x₆₃) ∧ (c r1 * c r2 = x₆₄ * x₆₃))) :=
sorry

end determine_color_sum_or_product_l163_163971


namespace running_speed_l163_163544

theorem running_speed (R : ℝ) (walking_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) (half_distance : ℝ) (walking_time : ℝ) (running_time : ℝ)
  (h1 : walking_speed = 4)
  (h2 : total_distance = 16)
  (h3 : total_time = 3)
  (h4 : half_distance = total_distance / 2)
  (h5 : walking_time = half_distance / walking_speed)
  (h6 : running_time = half_distance / R)
  (h7 : walking_time + running_time = total_time) :
  R = 8 := 
sorry

end running_speed_l163_163544


namespace identity_proof_l163_163353

theorem identity_proof (a b c x y z : ℝ) : 
  (a * x + b * y + c * z) ^ 2 + (b * x + c * y + a * z) ^ 2 + (c * x + a * y + b * z) ^ 2 = 
  (c * x + b * y + a * z) ^ 2 + (b * x + a * y + c * z) ^ 2 + (a * x + c * y + b * z) ^ 2 := 
by
  sorry

end identity_proof_l163_163353


namespace least_upper_bound_neg_expression_l163_163293

noncomputable def least_upper_bound : ℝ :=
  - (9 / 2)

theorem least_upper_bound_neg_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  ∃ M, M = least_upper_bound ∧
  ∀ x, (∀ a b, 0 < a → 0 < b → a + b = 1 → x ≤ - (1 / (2 * a)) - (2 / b)) ↔ x ≤ M :=
sorry

end least_upper_bound_neg_expression_l163_163293


namespace fruit_seller_gain_l163_163573

-- Define necessary variables
variables {C S : ℝ} (G : ℝ)

-- Given conditions
def selling_price_def (C : ℝ) : ℝ := 1.25 * C
def total_cost_price (C : ℝ) : ℝ := 150 * C
def total_selling_price (C : ℝ) : ℝ := 150 * (selling_price_def C)
def gain (C : ℝ) : ℝ := total_selling_price C - total_cost_price C

-- Statement to prove: number of apples' selling price gained by the fruit-seller is 30
theorem fruit_seller_gain : G = 30 ↔ gain C = G * (selling_price_def C) :=
by
  sorry

end fruit_seller_gain_l163_163573


namespace multiplication_problems_l163_163554

theorem multiplication_problems :
  (30 * 30 = 900) ∧
  (30 * 40 = 1200) ∧
  (40 * 70 = 2800) ∧
  (50 * 70 = 3500) ∧
  (60 * 70 = 4200) ∧
  (4 * 90 = 360) :=
by sorry

end multiplication_problems_l163_163554


namespace ellipse_formula_max_area_triangle_l163_163750

-- Definitions for Ellipse part
def ellipse_eq (x y a : ℝ) := (x^2 / a^2) + (y^2 / 3) = 1
def eccentricity (a : ℝ) := (Real.sqrt (a^2 - 3)) / a = 1 / 2

-- Definition for Circle intersection part
def circle_intersection_cond (t : ℝ) := (0 < t) ∧ (t < (2 * Real.sqrt 21) / 7)

-- Main theorem for ellipse equation
theorem ellipse_formula (a : ℝ) (h1 : a > Real.sqrt 3) (h2 : eccentricity a) :
  ellipse_eq x y 2 :=
sorry

-- Main theorem for maximum area of triangle ABC
theorem max_area_triangle (t : ℝ) (h : circle_intersection_cond t) :
  ∃ S, S = (3 * Real.sqrt 7) / 7 :=
sorry

end ellipse_formula_max_area_triangle_l163_163750


namespace part1_part2_l163_163557

def P : Set ℝ := {x | 4 / (x + 2) ≥ 1}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem part1 (h : m = 2) : P ∪ S m = {x | -2 < x ∧ x ≤ 3} :=
  by sorry

theorem part2 (h : ∀ x, x ∈ S m → x ∈ P) : 0 ≤ m ∧ m ≤ 1 :=
  by sorry

end part1_part2_l163_163557


namespace monotonicity_of_f_range_of_a_l163_163361

open Real

noncomputable def f (x a : ℝ) : ℝ := a * exp x + 2 * exp (-x) + (a - 2) * x

noncomputable def f_prime (x a : ℝ) : ℝ := (a * exp (2 * x) + (a - 2) * exp x - 2) / exp x

theorem monotonicity_of_f (a : ℝ) : 
  (∀ x : ℝ, f_prime x a ≤ 0) ↔ (a ≤ 0) :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, f x a ≥ (a + 2) * cos x) ↔ (2 ≤ a) :=
sorry

end monotonicity_of_f_range_of_a_l163_163361


namespace find_side_a_l163_163156

theorem find_side_a (a b c : ℝ) (B : ℝ)
  (h1 : c = Real.sqrt 2)
  (h2 : b = Real.sqrt 6)
  (h3 : B = 120) :
  a = Real.sqrt 2 :=
sorry

end find_side_a_l163_163156


namespace find_X_l163_163267

-- Define the variables for income, tax, and the variable X
def income := 58000
def tax := 8000

-- Define the tax formula as per the problem
def tax_formula (X : ℝ) : ℝ :=
  0.11 * X + 0.20 * (income - X)

-- The theorem we want to prove
theorem find_X :
  ∃ X : ℝ, tax_formula X = tax ∧ X = 40000 :=
sorry

end find_X_l163_163267


namespace amusement_park_weekly_revenue_l163_163712

def ticket_price : ℕ := 3
def visitors_mon_to_fri_per_day : ℕ := 100
def visitors_saturday : ℕ := 200
def visitors_sunday : ℕ := 300

theorem amusement_park_weekly_revenue : 
  let total_visitors_weekdays := visitors_mon_to_fri_per_day * 5
  let total_visitors_weekend := visitors_saturday + visitors_sunday
  let total_visitors := total_visitors_weekdays + total_visitors_weekend
  let total_revenue := total_visitors * ticket_price
  total_revenue = 3000 := by
  sorry

end amusement_park_weekly_revenue_l163_163712


namespace max_min_values_of_x_l163_163312

theorem max_min_values_of_x (x y z : ℝ) (h1 : x + y + z = 0) (h2 : (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ 2) :
  -2/3 ≤ x ∧ x ≤ 2/3 :=
sorry

end max_min_values_of_x_l163_163312


namespace remainder_of_2_pow_30_plus_3_mod_7_l163_163413

theorem remainder_of_2_pow_30_plus_3_mod_7 :
  (2^30 + 3) % 7 = 4 := 
sorry

end remainder_of_2_pow_30_plus_3_mod_7_l163_163413


namespace compute_65_sq_minus_55_sq_l163_163871

theorem compute_65_sq_minus_55_sq : 65^2 - 55^2 = 1200 :=
by
  -- We'll skip the proof here for simplicity
  sorry

end compute_65_sq_minus_55_sq_l163_163871


namespace three_digit_solutions_l163_163601

def three_digit_number (n a x y z : ℕ) : Prop :=
  n = 100 * x + 10 * y + z ∧
  1 ≤ x ∧ x < 10 ∧ 
  0 ≤ y ∧ y < 10 ∧ 
  0 ≤ z ∧ z < 10 ∧ 
  n + (x + y + z) = 111 * a

theorem three_digit_solutions (n : ℕ) (a x y z : ℕ) :
  three_digit_number n a x y z ↔ 
  n = 105 ∨ n = 324 ∨ n = 429 ∨ n = 543 ∨ 
  n = 648 ∨ n = 762 ∨ n = 867 ∨ n = 981 :=
sorry

end three_digit_solutions_l163_163601


namespace find_y_of_arithmetic_mean_l163_163357

theorem find_y_of_arithmetic_mean (y : ℝ) (h: (7 + 12 + 19 + 8 + 10 + y) / 6 = 15) : y = 34 :=
by {
  -- Skipping the proof
  sorry
}

end find_y_of_arithmetic_mean_l163_163357


namespace combined_score_of_three_students_left_l163_163473

variable (T S : ℕ) (avg16 avg13 : ℝ) (N16 N13 : ℕ)

theorem combined_score_of_three_students_left (h_avg16 : avg16 = 62.5) 
  (h_avg13 : avg13 = 62.0) (h_N16 : N16 = 16) (h_N13 : N13 = 13) 
  (h_total16 : T = avg16 * N16) (h_total13 : T - S = avg13 * N13) :
  S = 194 :=
by
  sorry

end combined_score_of_three_students_left_l163_163473


namespace calculate_LN_l163_163216

theorem calculate_LN (sinN : ℝ) (LM LN : ℝ) (h1 : sinN = 4 / 5) (h2 : LM = 20) : LN = 25 :=
by
  sorry

end calculate_LN_l163_163216


namespace strips_overlap_area_l163_163923

theorem strips_overlap_area :
  ∀ (length_left length_right area_only_left area_only_right : ℕ) (S : ℚ),
    length_left = 9 →
    length_right = 7 →
    area_only_left = 27 →
    area_only_right = 18 →
    (area_only_left + S) / (area_only_right + S) = 9 / 7 →
    S = 13.5 :=
by
  intros length_left length_right area_only_left area_only_right S
  intro h1 h2 h3 h4 h5
  sorry

end strips_overlap_area_l163_163923


namespace quadratic_inequality_l163_163653

noncomputable def exists_real_roots (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (a - 1) * x1 + (2 * a - 5) = 0 ∧ x2^2 + (a - 1) * x2 + (2 * a - 5) = 0

noncomputable def valid_values (a : ℝ) : Prop :=
  a > 5 / 2 ∧ a < 10

theorem quadratic_inequality (a : ℝ) 
  (h1 : exists_real_roots a) 
  (h2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (a - 1) * x1 + (2 * a - 5) = 0 ∧ x2^2 + (a - 1) * x2 + (2 * a - 5) = 0 
  → (1 / x1 + 1 / x2 < -3 / 5)) :
  valid_values a :=
sorry

end quadratic_inequality_l163_163653


namespace cricket_initial_matches_l163_163148

theorem cricket_initial_matches (x : ℝ) :
  (0.28 * x + 60 = 0.52 * (x + 60)) → x = 120 :=
by
  sorry

end cricket_initial_matches_l163_163148


namespace general_term_l163_163461

noncomputable def S : ℕ → ℤ
| n => 3 * n ^ 2 - 2 * n + 1

def a : ℕ → ℤ
| 0 => 2  -- Since sequences often start at n=1 and MATLAB indexing starts at 0.
| 1 => 2
| (n+2) => 6 * (n + 2) - 5

theorem general_term (n : ℕ) : 
  a n = if n = 1 then 2 else 6 * n - 5 :=
by sorry

end general_term_l163_163461


namespace marta_candies_received_l163_163202

theorem marta_candies_received:
  ∃ x y : ℕ, x + y = 200 ∧ x < 100 ∧ x > (4 * y) / 5 ∧ (x % 8 = 0) ∧ (y % 8 = 0) ∧ x = 96 ∧ y = 104 := 
sorry

end marta_candies_received_l163_163202


namespace solve_for_N_l163_163474

theorem solve_for_N (N : ℤ) (h : 2 * N^2 + N = 12) (h_neg : N < 0) : N = -3 := 
by 
  sorry

end solve_for_N_l163_163474


namespace units_cost_l163_163417

theorem units_cost (x y z : ℝ) 
  (h1 : 3 * x + 7 * y + z = 3.15)
  (h2 : 4 * x + 10 * y + z = 4.20) : 
  x + y + z = 1.05 :=
by 
  sorry

end units_cost_l163_163417


namespace a_2020_equality_l163_163325

variables (n : ℤ)

def cube (x : ℤ) : ℤ := x * x * x

lemma a_six_n (n : ℤ) :
  cube (n + 1) + cube (n - 1) + cube (-n) + cube (-n) = 6 * n :=
sorry

lemma a_six_n_plus_one (n : ℤ) :
  cube (n + 1) + cube (n - 1) + cube (-n) + cube (-n) + 1 = 6 * n + 1 :=
sorry

lemma a_six_n_minus_one (n : ℤ) :
  cube (n + 1) + cube (n - 1) + cube (-n) + cube (-n) - 1 = 6 * n - 1 :=
sorry

lemma a_six_n_plus_two (n : ℤ) :
  cube n + cube (n - 2) + cube (-n + 1) + cube (-n + 1) + 8 = 6 * n + 2 :=
sorry

lemma a_six_n_minus_two (n : ℤ) :
  cube (n + 2) + cube n + cube (-n - 1) + cube (-n - 1) + (-8) = 6 * n - 2 :=
sorry

lemma a_six_n_plus_three (n : ℤ) :
  cube (n - 3) + cube (n - 5) + cube (-n + 4) + cube (-n + 4) + 27 = 6 * n + 3 :=
sorry

theorem a_2020_equality :
  2020 = cube 339 + cube 337 + cube (-338) + cube (-338) + cube (-2) :=
sorry

end a_2020_equality_l163_163325


namespace selling_price_to_achieve_profit_l163_163098

theorem selling_price_to_achieve_profit :
  ∃ (x : ℝ), let original_price := 210
              let purchase_price := 190
              let avg_sales_initial := 8
              let profit_goal := 280
              (210 - x = 200) ∧
              let profit_per_item := original_price - purchase_price - x
              let avg_sales_quantity := avg_sales_initial + 2 * x
              profit_per_item * avg_sales_quantity = profit_goal := by
  sorry

end selling_price_to_achieve_profit_l163_163098


namespace part1_l163_163424

theorem part1 (a : ℝ) : 
  (∀ x ∈ Set.Ici (1/2 : ℝ), 2 * x + a / (x + 1) ≥ 0) → a ≥ -3 / 2 :=
sorry

end part1_l163_163424


namespace vector_sum_is_correct_l163_163830

-- Define the points A, B, and C
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (0, 1)

-- Define the vectors AB and AC
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vectorAC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- State the theorem
theorem vector_sum_is_correct : vectorAB + vectorAC = (-3, -1) :=
by
  sorry

end vector_sum_is_correct_l163_163830


namespace ram_money_l163_163933

theorem ram_money (R G K : ℝ) (h1 : R / G = 7 / 17) (h2 : G / K = 7 / 17) (h3 : K = 3468) :
  R = 588 := by
  sorry

end ram_money_l163_163933


namespace valid_addends_l163_163075

noncomputable def is_valid_addend (n : ℕ) : Prop :=
  ∃ (X Y : ℕ), (100 * 9 + 10 * X + 4) = n ∧ (30 + Y) ∈ [36, 30, 20, 10]

theorem valid_addends :
  ∀ (n : ℕ),
  is_valid_addend n ↔ (n = 964 ∨ n = 974 ∨ n = 984 ∨ n = 994) :=
by
  sorry

end valid_addends_l163_163075


namespace distance_between_cities_l163_163340

theorem distance_between_cities:
    ∃ (x y : ℝ),
    (x = 135) ∧
    (y = 175) ∧
    (7 / 9 * x = 105) ∧
    (x + 7 / 9 * x + y = 415) ∧
    (x = 27 / 35 * y) :=
by
  sorry

end distance_between_cities_l163_163340


namespace scientific_notation_of_114_trillion_l163_163666

theorem scientific_notation_of_114_trillion :
  (114 : ℝ) * 10^12 = (1.14 : ℝ) * 10^14 :=
by
  sorry

end scientific_notation_of_114_trillion_l163_163666


namespace employee_payment_l163_163804

theorem employee_payment (X Y : ℝ) (h1 : X + Y = 528) (h2 : X = 1.2 * Y) : Y = 240 :=
by
  sorry

end employee_payment_l163_163804


namespace minimum_value_l163_163420

def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 1487

theorem minimum_value : ∃ x : ℝ, f x = 1484 := 
sorry

end minimum_value_l163_163420


namespace beth_total_crayons_l163_163786

theorem beth_total_crayons :
  let packs := 4
  let crayons_per_pack := 10
  let extra_crayons := 6
  packs * crayons_per_pack + extra_crayons = 46 :=
by
  let packs := 4
  let crayons_per_pack := 10
  let extra_crayons := 6
  show packs * crayons_per_pack + extra_crayons = 46
  sorry

end beth_total_crayons_l163_163786


namespace rows_needed_correct_l163_163495

variable (pencils rows_needed : Nat)

def total_pencils : Nat := 35
def pencils_per_row : Nat := 5
def rows_expected : Nat := 7

theorem rows_needed_correct : rows_needed = total_pencils / pencils_per_row →
  rows_needed = rows_expected := by
  sorry

end rows_needed_correct_l163_163495


namespace evaluate_expression_l163_163234

theorem evaluate_expression (a b : ℕ) (h_a : a = 15) (h_b : b = 7) :
  (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 210 :=
by 
  rw [h_a, h_b]
  sorry

end evaluate_expression_l163_163234


namespace speed_of_man_l163_163100

-- Define all given conditions and constants

def trainLength : ℝ := 110 -- in meters
def trainSpeed : ℝ := 40 -- in km/hr
def timeToPass : ℝ := 8.799296056315494 -- in seconds

-- We want to prove that the speed of the man is approximately 4.9968 km/hr
theorem speed_of_man :
  let trainSpeedMS := trainSpeed * (1000 / 3600)
  let relativeSpeed := trainLength / timeToPass
  let manSpeedMS := relativeSpeed - trainSpeedMS
  let manSpeedKMH := manSpeedMS * (3600 / 1000)
  abs (manSpeedKMH - 4.9968) < 0.01 := sorry

end speed_of_man_l163_163100


namespace number_of_red_parrots_l163_163144

-- Defining the conditions from a)
def fraction_yellow_parrots : ℚ := 2 / 3
def total_birds : ℕ := 120

-- Stating the theorem we want to prove
theorem number_of_red_parrots (H1 : fraction_yellow_parrots = 2 / 3) (H2 : total_birds = 120) : 
  (1 - fraction_yellow_parrots) * total_birds = 40 := 
by 
  sorry

end number_of_red_parrots_l163_163144


namespace rope_cut_ratio_l163_163169

theorem rope_cut_ratio (L : ℕ) (a b : ℕ) (hL : L = 40) (ha : a = 2) (hb : b = 3) :
  L / (a + b) * a = 16 :=
by
  sorry

end rope_cut_ratio_l163_163169


namespace weight_difference_l163_163097

open Real

def yellow_weight : ℝ := 0.6
def green_weight : ℝ := 0.4
def red_weight : ℝ := 0.8
def blue_weight : ℝ := 0.5

def weights : List ℝ := [yellow_weight, green_weight, red_weight, blue_weight]

theorem weight_difference : (List.maximum weights).getD 0 - (List.minimum weights).getD 0 = 0.4 :=
by
  sorry

end weight_difference_l163_163097


namespace F_shaped_to_cube_l163_163255

-- Define the problem context in Lean 4
structure F_shaped_figure :=
  (squares : Finset (Fin 5) )

structure additional_squares :=
  (label : String )

def is_valid_configuration (f : F_shaped_figure) (s : additional_squares) : Prop :=
  -- This function should encapsulate the logic for checking the validity of a configuration
  sorry -- Implementation of validity check is omitted (replacing it with sorry)

-- The main theorem statement
theorem F_shaped_to_cube (f : F_shaped_figure) (squares: Finset additional_squares) : 
  ∃ valid_squares : Finset additional_squares, valid_squares.card = 3 ∧ 
    ∀ s ∈ valid_squares, is_valid_configuration f s := 
sorry

end F_shaped_to_cube_l163_163255


namespace selected_student_in_eighteenth_group_l163_163419

def systematic_sampling (first_number common_difference nth_term : ℕ) : ℕ :=
  first_number + (nth_term - 1) * common_difference

theorem selected_student_in_eighteenth_group :
  systematic_sampling 22 50 18 = 872 :=
by
  sorry

end selected_student_in_eighteenth_group_l163_163419


namespace find_coefficient_y_l163_163630

theorem find_coefficient_y (a b c : ℕ) (h1 : 100 * a + 10 * b + c - 7 * (a + b + c) = 100) (h2 : a + b + c ≠ 0) :
  100 * c + 10 * b + a = 43 * (a + b + c) :=
by
  sorry

end find_coefficient_y_l163_163630


namespace greatest_value_a_maximum_value_a_l163_163204

-- Define the quadratic polynomial
def quadratic (a : ℝ) : ℝ := -a^2 + 9 * a - 20

-- The statement to be proven:
theorem greatest_value_a : ∀ a : ℝ, (quadratic a ≥ 0) → a ≤ 5 := 
sorry

theorem maximum_value_a : quadratic 5 = 0 :=
sorry

end greatest_value_a_maximum_value_a_l163_163204


namespace todd_ingredients_l163_163141

variables (B R N : ℕ) (P A : ℝ) (I : ℝ)

def todd_problem (B R N : ℕ) (P A I : ℝ) : Prop := 
  B = 100 ∧ 
  R = 110 ∧ 
  N = 200 ∧ 
  P = 0.75 ∧ 
  A = 65 ∧ 
  I = 25

theorem todd_ingredients :
  todd_problem 100 110 200 0.75 65 25 :=
by sorry

end todd_ingredients_l163_163141


namespace lines_intersect_l163_163809

def line1 (s : ℚ) : ℚ × ℚ :=
  (1 + 2 * s, 4 - 3 * s)

def line2 (v : ℚ) : ℚ × ℚ :=
  (3 + 4 * v, 9 - v)

theorem lines_intersect :
  ∃ s v : ℚ, (line1 s) = (line2 v) ∧ (line1 s) = (-17/5, 53/5) := 
sorry

end lines_intersect_l163_163809


namespace tangent_line_correct_l163_163957

-- Define the curve y = x^3 - 1
def curve (x : ℝ) : ℝ := x^3 - 1

-- Define the derivative of the curve
def derivative_curve (x : ℝ) : ℝ := 3 * x^2

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (1, curve 1)

-- Define the tangent line equation at x = 1
def tangent_line (x : ℝ) : ℝ := 3 * x - 3

-- The formal statement to be proven
theorem tangent_line_correct :
  ∀ x : ℝ, curve x = x^3 - 1 ∧ derivative_curve x = 3 * x^2 ∧ tangent_point = (1, 0) → 
    tangent_line 1 = 3 * 1 - 3 :=
by
  sorry

end tangent_line_correct_l163_163957


namespace samantha_routes_l163_163323

-- Definitions of the conditions
def blocks_west_to_sw_corner := 3
def blocks_south_to_sw_corner := 2
def blocks_east_to_school := 4
def blocks_north_to_school := 3
def ways_house_to_sw_corner : ℕ := Nat.choose (blocks_west_to_sw_corner + blocks_south_to_sw_corner) blocks_south_to_sw_corner
def ways_through_park : ℕ := 2
def ways_ne_corner_to_school : ℕ := Nat.choose (blocks_east_to_school + blocks_north_to_school) blocks_north_to_school

-- The proof statement
theorem samantha_routes : (ways_house_to_sw_corner * ways_through_park * ways_ne_corner_to_school) = 700 :=
by
  -- Using "sorry" as a placeholder for the actual proof
  sorry

end samantha_routes_l163_163323


namespace range_of_m_l163_163051

noncomputable def f (x m : ℝ) : ℝ :=
if x < 0 then (x - m) ^ 2 - 2 else 2 * x ^ 3 - 3 * x ^ 2

theorem range_of_m (m : ℝ) : (∃ x : ℝ, f x m = -1) ↔ m ≥ 1 :=
by
  sorry

end range_of_m_l163_163051


namespace danil_claim_false_l163_163776

theorem danil_claim_false (E O : ℕ) (hE : E % 2 = 0) (hO : O % 2 = 0) (h : O = E + 15) : false :=
by sorry

end danil_claim_false_l163_163776


namespace sum_of_digits_0_to_999_l163_163856

-- Sum of digits from 0 to 9
def sum_of_digits : ℕ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)

-- Sum of digits from 1 to 9
def sum_of_digits_without_zero : ℕ := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)

-- Units place sum
def units_sum : ℕ := sum_of_digits * 100

-- Tens place sum
def tens_sum : ℕ := sum_of_digits * 100

-- Hundreds place sum
def hundreds_sum : ℕ := sum_of_digits_without_zero * 100

-- Total sum
def total_sum : ℕ := units_sum + tens_sum + hundreds_sum

theorem sum_of_digits_0_to_999 : total_sum = 13500 := by
  sorry

end sum_of_digits_0_to_999_l163_163856


namespace solve_for_x_l163_163631

namespace RationalOps

-- Define the custom operation ※ on rational numbers
def star (a b : ℚ) : ℚ := a + b

-- Define the equation involving the custom operation
def equation (x : ℚ) : Prop := star 4 (star x 3) = 1

-- State the theorem to prove the solution
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = -6 := by
  sorry

end solve_for_x_l163_163631


namespace probability_top_red_second_black_l163_163232

def num_red_cards : ℕ := 39
def num_black_cards : ℕ := 39
def total_cards : ℕ := 78

theorem probability_top_red_second_black :
  (num_red_cards * num_black_cards) / (total_cards * (total_cards - 1)) = 507 / 2002 := 
sorry

end probability_top_red_second_black_l163_163232


namespace repeating_decimal_fraction_l163_163527

theorem repeating_decimal_fraction :
  (5 + 341 / 999) = (5336 / 999) :=
by
  sorry

end repeating_decimal_fraction_l163_163527


namespace father_age_l163_163409

theorem father_age : 
  ∀ (S F : ℕ), (S - 5 = 11) ∧ (F - S = S) → F = 32 := 
by
  intros S F h
  -- Use the conditions to derive further equations and steps
  sorry

end father_age_l163_163409


namespace find_number_of_pourings_l163_163083

-- Define the sequence of remaining water after each pouring
def remaining_water (n : ℕ) : ℚ :=
  (2 : ℚ) / (n + 2)

-- The main theorem statement
theorem find_number_of_pourings :
  ∃ n : ℕ, remaining_water n = 1 / 8 :=
by
  sorry

end find_number_of_pourings_l163_163083


namespace game_is_unfair_l163_163900

def pencil_game_unfair : Prop :=
∀ (take1 take2 : ℕ → ℕ),
  take1 1 = 1 ∨ take1 1 = 2 →
  take2 2 = 1 ∨ take2 2 = 2 →
  ∀ n : ℕ,
    n = 5 → (∃ first_move : ℕ, (take1 first_move = 2) ∧ (take2 (take1 first_move) = 1 ∨ take2 (take1 first_move) = 2) ∧ (take1 (take2 (n - take1 first_move)) = 1 ∨ take1 (take2 (n - take1 first_move)) = 2) ∧
    ∀ second_move : ℕ, (second_move = n - first_move - take2 (n - take1 first_move)) → 
    n - first_move - take2 (n - take1 first_move) = 1 ∨ n - first_move - take2 (n - take1 first_move) = 2)

theorem game_is_unfair : pencil_game_unfair := 
sorry

end game_is_unfair_l163_163900


namespace union_M_N_l163_163829

def M : Set ℝ := { x | -3 < x ∧ x ≤ 5 }
def N : Set ℝ := { x | x > 3 }

theorem union_M_N : M ∪ N = { x | x > -3 } :=
by
  sorry

end union_M_N_l163_163829


namespace percent_equivalence_l163_163348

theorem percent_equivalence (y : ℝ) : 0.30 * (0.60 * y) = 0.18 * y :=
by sorry

end percent_equivalence_l163_163348


namespace range_of_a_minus_b_l163_163175

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : 0 < b ∧ b < 1) : -1 < a - b ∧ a - b < 2 := 
by
  sorry

end range_of_a_minus_b_l163_163175


namespace exists_zero_in_interval_l163_163391

open Set Real

theorem exists_zero_in_interval (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Icc a b)) 
  (h_pos : f a * f b > 0) : ∃ c ∈ Ioo a b, f c = 0 := sorry

end exists_zero_in_interval_l163_163391


namespace weight_difference_calc_l163_163673

-- Define the weights in pounds
def Anne_weight : ℕ := 67
def Douglas_weight : ℕ := 52
def Maria_weight : ℕ := 48

-- Define the combined weight of Douglas and Maria
def combined_weight_DM : ℕ := Douglas_weight + Maria_weight

-- Define the weight difference
def weight_difference : ℤ := Anne_weight - combined_weight_DM

-- The theorem stating the difference
theorem weight_difference_calc : weight_difference = -33 := by
  -- The proof will go here
  sorry

end weight_difference_calc_l163_163673


namespace expand_x_plus_3y_squared_expand_2x_plus_3y_squared_expand_m3_plus_n5_squared_expand_5x_minus_3y_squared_expand_3m5_minus_4n2_squared_l163_163832

-- Proof for (x + 3y)^2 = x^2 + 6xy + 9y^2
theorem expand_x_plus_3y_squared (x y : ℝ) : 
  (x + 3 * y) ^ 2 = x ^ 2 + 6 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (2x + 3y)^2 = 4x^2 + 12xy + 9y^2
theorem expand_2x_plus_3y_squared (x y : ℝ) : 
  (2 * x + 3 * y) ^ 2 = 4 * x ^ 2 + 12 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (m^3 + n^5)^2 = m^6 + 2m^3n^5 + n^10
theorem expand_m3_plus_n5_squared (m n : ℝ) : 
  (m ^ 3 + n ^ 5) ^ 2 = m ^ 6 + 2 * m ^ 3 * n ^ 5 + n ^ 10 := 
  sorry

-- Proof for (5x - 3y)^2 = 25x^2 - 30xy + 9y^2
theorem expand_5x_minus_3y_squared (x y : ℝ) : 
  (5 * x - 3 * y) ^ 2 = 25 * x ^ 2 - 30 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (3m^5 - 4n^2)^2 = 9m^10 - 24m^5n^2 + 16n^4
theorem expand_3m5_minus_4n2_squared (m n : ℝ) : 
  (3 * m ^ 5 - 4 * n ^ 2) ^ 2 = 9 * m ^ 10 - 24 * m ^ 5 * n ^ 2 + 16 * n ^ 4 := 
  sorry

end expand_x_plus_3y_squared_expand_2x_plus_3y_squared_expand_m3_plus_n5_squared_expand_5x_minus_3y_squared_expand_3m5_minus_4n2_squared_l163_163832


namespace find_integer_part_of_m_l163_163205

theorem find_integer_part_of_m {m : ℝ} (h_lecture_duration : m > 0) 
    (h_swap_positions : ∃ k : ℤ, 120 + m = 60 + k * 12 * 60 / 13 ∧ (120 + m) % 60 = 60 * (120 + m) / 720) : 
    ⌊m⌋ = 46 :=
by
  sorry

end find_integer_part_of_m_l163_163205


namespace initial_teach_count_l163_163178

theorem initial_teach_count :
  ∃ (x y : ℕ), (x + x * y + (x + x * y) * (y + x * y) = 195) ∧
               (y + x * y + (y + x * y) * (x + x * y) = 192) ∧
               x = 5 ∧ y = 2 :=
by {
  sorry
}

end initial_teach_count_l163_163178


namespace temperature_difference_correct_l163_163650

def refrigerator_temp : ℝ := 3
def freezer_temp : ℝ := -10
def temperature_difference : ℝ := refrigerator_temp - freezer_temp

theorem temperature_difference_correct : temperature_difference = 13 := 
by
  sorry

end temperature_difference_correct_l163_163650


namespace function_correct_max_min_values_l163_163427

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

@[simp]
theorem function_correct : (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 4)) ∧ 
                           (f (3 * Real.pi / 8) = 0) ∧ 
                           (f (Real.pi / 8) = 2) :=
by
  sorry

theorem max_min_values : (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 
                          f x = -2) ∧ 
                         (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 
                          f x = 2) :=
by
  sorry

end function_correct_max_min_values_l163_163427


namespace ratio_of_games_played_to_losses_l163_163219

-- Conditions
def games_played : ℕ := 10
def games_won : ℕ := 5
def games_lost : ℕ := games_played - games_won

-- Prove the ratio of games played to games lost is 2:1
theorem ratio_of_games_played_to_losses
  (h_played : games_played = 10)
  (h_won : games_won = 5) :
  (games_played / Nat.gcd games_played games_lost : ℕ) /
  (games_lost / Nat.gcd games_played games_lost : ℕ) = 2 / 1 :=
by
  sorry

end ratio_of_games_played_to_losses_l163_163219


namespace solution_set_inequality_l163_163401

variable (a x : ℝ)

-- Conditions
theorem solution_set_inequality (h₀ : 0 < a) (h₁ : a < 1) :
  ((a - x) * (x - (1 / a)) > 0) ↔ (a < x ∧ x < 1 / a) := 
by 
  sorry

end solution_set_inequality_l163_163401


namespace fresh_fruit_water_content_l163_163524

theorem fresh_fruit_water_content (W N : ℝ) 
  (fresh_weight_dried: W + N = 50) 
  (dried_weight: (0.80 * 5) = N) : 
  ((W / (W + N)) * 100 = 92) :=
by
  sorry

end fresh_fruit_water_content_l163_163524


namespace evaluate_expression_l163_163940

theorem evaluate_expression : (-3)^7 / 3^5 + 2^5 - 7^2 = -26 := 
by
  sorry

end evaluate_expression_l163_163940


namespace total_rooms_to_paint_l163_163515

theorem total_rooms_to_paint :
  ∀ (hours_per_room hours_remaining rooms_painted : ℕ),
    hours_per_room = 7 →
    hours_remaining = 63 →
    rooms_painted = 2 →
    rooms_painted + hours_remaining / hours_per_room = 11 :=
by
  intros
  sorry

end total_rooms_to_paint_l163_163515


namespace initial_apples_value_l163_163207

-- Definitions for the conditions
def picked_apples : ℤ := 105
def total_apples : ℤ := 161

-- Statement to prove
theorem initial_apples_value : ∀ (initial_apples : ℤ), 
  initial_apples + picked_apples = total_apples → 
  initial_apples = total_apples - picked_apples := 
by 
  sorry

end initial_apples_value_l163_163207


namespace proof_problem_l163_163434

noncomputable def valid_x (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 / 3 ∧ x ≤ 2

theorem proof_problem (x : ℝ) (h : valid_x x) :
  (x^2 + 2 * x^3 - 3 * x^4) / (x + 2 * x^2 - 3 * x^3) ≤ 2 :=
sorry

end proof_problem_l163_163434


namespace students_taking_either_geometry_or_history_but_not_both_l163_163471

theorem students_taking_either_geometry_or_history_but_not_both
    (students_in_both : ℕ)
    (students_in_geometry : ℕ)
    (students_only_in_history : ℕ)
    (students_in_both_cond : students_in_both = 15)
    (students_in_geometry_cond : students_in_geometry = 35)
    (students_only_in_history_cond : students_only_in_history = 18) :
    (students_in_geometry - students_in_both + students_only_in_history = 38) :=
by
  sorry

end students_taking_either_geometry_or_history_but_not_both_l163_163471


namespace plane_speed_east_l163_163921

def plane_travel_problem (v : ℕ) : Prop :=
  let time : ℕ := 35 / 10 
  let distance_east := v * time
  let distance_west := 275 * time
  let total_distance := distance_east + distance_west
  total_distance = 2100

theorem plane_speed_east : ∃ v : ℕ, plane_travel_problem v ∧ v = 325 :=
sorry

end plane_speed_east_l163_163921


namespace pole_intersection_height_l163_163407

theorem pole_intersection_height :
  ∀ (d h1 h2 : ℝ), d = 120 ∧ h1 = 30 ∧ h2 = 90 → 
  ∃ y : ℝ, y = 18 :=
by
  sorry

end pole_intersection_height_l163_163407


namespace polynomial_square_binomial_l163_163817

-- Define the given polynomial and binomial
def polynomial (x : ℚ) (a : ℚ) : ℚ :=
  25 * x^2 + 40 * x + a

def binomial (x b : ℚ) : ℚ :=
  (5 * x + b)^2

-- Theorem to state the problem
theorem polynomial_square_binomial (a : ℚ) : 
  (∃ b, polynomial x a = binomial x b) ↔ a = 16 :=
by
  sorry

end polynomial_square_binomial_l163_163817


namespace gcd_increase_by_9_l163_163797

theorem gcd_increase_by_9 (m n d : ℕ) (h1 : d = Nat.gcd m n) (h2 : 9 * d = Nat.gcd (m + 6) n) : d = 3 ∨ d = 6 :=
by
  sorry

end gcd_increase_by_9_l163_163797


namespace pat_stickers_l163_163283

theorem pat_stickers (stickers_given_away stickers_left : ℝ) 
(h_given_away : stickers_given_away = 22.0)
(h_left : stickers_left = 17.0) : 
(stickers_given_away + stickers_left = 39) :=
by
  sorry

end pat_stickers_l163_163283


namespace negation_of_existence_l163_163385

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem negation_of_existence:
  (∃ x : ℝ, log_base 3 x ≤ 0) ↔ ∀ x : ℝ, log_base 3 x < 0 :=
by
  sorry

end negation_of_existence_l163_163385


namespace g_inv_g_inv_14_l163_163191

noncomputable def g (x : ℝ) := 3 * x - 4

noncomputable def g_inv (x : ℝ) := (x + 4) / 3

theorem g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by sorry

end g_inv_g_inv_14_l163_163191


namespace max_probability_first_black_ace_l163_163794

def probability_first_black_ace(k : ℕ) : ℚ :=
  if 1 ≤ k ∧ k ≤ 51 then (52 - k) / 1326 else 0

theorem max_probability_first_black_ace : 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 51 → probability_first_black_ace k ≤ probability_first_black_ace 1 :=
by
  sorry

end max_probability_first_black_ace_l163_163794


namespace david_produces_8_more_widgets_l163_163925

variable (w t : ℝ)

def widgets_monday (w t : ℝ) : ℝ :=
  w * t

def widgets_tuesday (w t : ℝ) : ℝ :=
  (w + 4) * (t - 2)

theorem david_produces_8_more_widgets (h : w = 2 * t) : 
  widgets_monday w t - widgets_tuesday w t = 8 :=
by
  sorry

end david_produces_8_more_widgets_l163_163925


namespace count_integer_values_l163_163805

-- Statement of the problem in Lean 4
theorem count_integer_values (x : ℤ) : 
  (7 * x^2 + 23 * x + 20 ≤ 30) → 
  ∃ (n : ℕ), n = 6 :=
sorry

end count_integer_values_l163_163805


namespace range_of_2_cos_sq_l163_163782

theorem range_of_2_cos_sq :
  ∀ x : ℝ, 0 ≤ 2 * (Real.cos x) ^ 2 ∧ 2 * (Real.cos x) ^ 2 ≤ 2 :=
by sorry

end range_of_2_cos_sq_l163_163782


namespace steve_cookie_boxes_l163_163080

theorem steve_cookie_boxes (total_spent milk_cost cereal_cost banana_cost apple_cost : ℝ)
  (num_cereals num_bananas num_apples : ℕ) (cookie_cost_multiplier : ℝ) (cookie_cost : ℝ)
  (cookie_boxes : ℕ) :
  total_spent = 25 ∧ milk_cost = 3 ∧ cereal_cost = 3.5 ∧ banana_cost = 0.25 ∧ apple_cost = 0.5 ∧
  cookie_cost_multiplier = 2 ∧ 
  num_cereals = 2 ∧ num_bananas = 4 ∧ num_apples = 4 ∧
  cookie_cost = cookie_cost_multiplier * milk_cost ∧
  total_spent = (milk_cost + num_cereals * cereal_cost + num_bananas * banana_cost + num_apples * apple_cost + cookie_boxes * cookie_cost)
  → cookie_boxes = 2 :=
sorry

end steve_cookie_boxes_l163_163080


namespace difference_of_squares_144_l163_163078

theorem difference_of_squares_144 (n : ℕ) (h : 3 * n + 3 < 150) : (n + 2)^2 - n^2 = 144 :=
by
  -- Given the conditions, we need to show this holds.
  sorry

end difference_of_squares_144_l163_163078


namespace sequence_periodic_l163_163590

def last_digit (n : ℕ) : ℕ := n % 10

noncomputable def a_n (n : ℕ) : ℕ := last_digit (n^(n^n))

theorem sequence_periodic :
  ∃ period : ℕ, period = 20 ∧ ∀ n m : ℕ, n ≡ m [MOD period] → a_n n = a_n m :=
sorry

end sequence_periodic_l163_163590


namespace find_age_of_30th_student_l163_163003

theorem find_age_of_30th_student :
  let avg1 := 23.5
  let n1 := 30
  let avg2 := 21.3
  let n2 := 9
  let avg3 := 19.7
  let n3 := 12
  let avg4 := 24.2
  let n4 := 7
  let avg5 := 35
  let n5 := 1
  let total_age_30 := n1 * avg1
  let total_age_9 := n2 * avg2
  let total_age_12 := n3 * avg3
  let total_age_7 := n4 * avg4
  let total_age_1 := n5 * avg5
  let total_age_29 := total_age_9 + total_age_12 + total_age_7 + total_age_1
  let age_30th := total_age_30 - total_age_29
  age_30th = 72.5 :=
by
  sorry

end find_age_of_30th_student_l163_163003


namespace find_n_in_arithmetic_sequence_l163_163861

theorem find_n_in_arithmetic_sequence 
  (a : ℕ → ℕ)
  (a_1 : ℕ)
  (d : ℕ) 
  (a_n : ℕ) 
  (n : ℕ)
  (h₀ : a_1 = 11)
  (h₁ : d = 2)
  (h₂ : a n = a_1 + (n - 1) * d)
  (h₃ : a n = 2009) :
  n = 1000 := 
by
  -- The proof steps would go here
  sorry

end find_n_in_arithmetic_sequence_l163_163861


namespace quadratic_has_real_roots_iff_l163_163820

theorem quadratic_has_real_roots_iff (a : ℝ) :
  (∃ (x : ℝ), a * x^2 - 4 * x - 2 = 0) ↔ (a ≥ -2 ∧ a ≠ 0) := by
  sorry

end quadratic_has_real_roots_iff_l163_163820


namespace first_discount_percentage_l163_163338

noncomputable def saree_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initial_price * (1 - discount1 / 100) * (1 - discount2 / 100)

theorem first_discount_percentage (x : ℝ) : saree_price 400 x 20 = 240 → x = 25 :=
by sorry

end first_discount_percentage_l163_163338


namespace trapezoid_equilateral_triangle_ratio_l163_163116

theorem trapezoid_equilateral_triangle_ratio (s d : ℝ) (AB CD : ℝ) 
  (h1 : AB = s) 
  (h2 : CD = 2 * d)
  (h3 : d = s) : 
  AB / CD = 1 / 2 := 
by
  sorry

end trapezoid_equilateral_triangle_ratio_l163_163116


namespace smallest_c_l163_163609

variable {f : ℝ → ℝ}

def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 0 = 0) ∧ (f 1 = 1) ∧ (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x) ∧ (∀ x1 x2, 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2)

theorem smallest_c (f : ℝ → ℝ) (h : satisfies_conditions f) : (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ 2 * x) ∧ (∀ c, c < 2 → ∃ x, 0 < x ∧ x ≤ 1 ∧ ¬ (f x ≤ c * x)) :=
by
  sorry

end smallest_c_l163_163609


namespace triangle_side_length_l163_163109

theorem triangle_side_length (A : ℝ) (b : ℝ) (S : ℝ) (hA : A = 120) (hb : b = 4) (hS: S = 2 * Real.sqrt 3) : 
  ∃ c : ℝ, c = 2 := 
by 
  sorry

end triangle_side_length_l163_163109


namespace integer_cube_less_than_triple_l163_163026

theorem integer_cube_less_than_triple (x : ℤ) : x^3 < 3 * x ↔ x = 0 :=
by 
  sorry

end integer_cube_less_than_triple_l163_163026


namespace find_n_l163_163695

theorem find_n (n : ℕ) (hnpos : 0 < n)
  (hsquare : ∃ k : ℕ, k^2 = n^4 + 2*n^3 + 5*n^2 + 12*n + 5) :
  n = 1 ∨ n = 2 := 
sorry

end find_n_l163_163695


namespace Jose_age_proof_l163_163226

-- Definitions based on the conditions
def Inez_age : ℕ := 15
def Zack_age : ℕ := Inez_age + 5
def Jose_age : ℕ := Zack_age - 7

theorem Jose_age_proof : Jose_age = 13 :=
by
  -- Proof omitted
  sorry

end Jose_age_proof_l163_163226


namespace part1_solution_set_eq_part2_a_range_l163_163663

theorem part1_solution_set_eq : {x : ℝ | |2 * x + 1| + |2 * x - 3| ≤ 6} = Set.Icc (-1) 2 :=
by sorry

theorem part2_a_range (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, |2 * x + 1| + |2 * x - 3| < |a - 2|) → 6 < a :=
by sorry

end part1_solution_set_eq_part2_a_range_l163_163663


namespace correct_operation_result_l163_163602

variable (x : ℕ)

theorem correct_operation_result 
  (h : x / 15 = 6) : 15 * x = 1350 :=
sorry

end correct_operation_result_l163_163602


namespace math_problem_l163_163658

noncomputable def proof_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  a^2 + 4 * b^2 + 1 / (a * b) ≥ 4

theorem math_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : proof_problem a b ha hb :=
by
  sorry

end math_problem_l163_163658


namespace find_speed_of_second_car_l163_163393

noncomputable def problem : Prop := 
  let s1 := 1600 -- meters
  let s2 := 800 -- meters
  let v1 := 72 / 3.6 -- converting to meters per second for convenience; 72 km/h = 20 m/s
  let s := 200 -- meters
  let t1 := s1 / v1 -- time taken by the first car to reach the intersection
  let l1 := s2 - s -- scenario 1: second car travels 600 meters
  let l2 := s2 + s -- scenario 2: second car travels 1000 meters
  let v2_1 := l1 / t1 -- speed calculation for scenario 1
  let v2_2 := l2 / t1 -- speed calculation for scenario 2
  v2_1 = 7.5 ∧ v2_2 = 12.5 -- expected speeds in both scenarios

theorem find_speed_of_second_car : problem := sorry

end find_speed_of_second_car_l163_163393


namespace cone_surface_area_l163_163481

theorem cone_surface_area {h : ℝ} {A_base : ℝ} (h_eq : h = 4) (A_base_eq : A_base = 9 * Real.pi) :
  let r := Real.sqrt (A_base / Real.pi)
  let l := Real.sqrt (r^2 + h^2)
  let lateral_area := Real.pi * r * l
  let total_surface_area := lateral_area + A_base
  total_surface_area = 24 * Real.pi :=
by
  sorry

end cone_surface_area_l163_163481


namespace rainfall_november_is_180_l163_163649

-- Defining the conditions
def daily_rainfall_first_15_days := 4 -- inches per day
def days_in_first_period := 15
def total_days_in_november := 30
def multiplier_for_second_period := 2

-- Calculation based on the problem's conditions
def total_rainfall_november := 
  (daily_rainfall_first_15_days * days_in_first_period) + 
  (multiplier_for_second_period * daily_rainfall_first_15_days * (total_days_in_november - days_in_first_period))

-- Prove that the total rainfall in November is 180 inches
theorem rainfall_november_is_180 : total_rainfall_november = 180 :=
by
  -- Proof steps (to be filled in)
  sorry

end rainfall_november_is_180_l163_163649


namespace cos_alpha_plus_beta_l163_163159

variable (α β : ℝ)
variable (hα : Real.sin α = (Real.sqrt 5) / 5)
variable (hβ : Real.sin β = (Real.sqrt 10) / 10)
variable (hα_obtuse : π / 2 < α ∧ α < π)
variable (hβ_obtuse : π / 2 < β ∧ β < π)

theorem cos_alpha_plus_beta : Real.cos (α + β) = Real.sqrt 2 / 2 ∧ α + β = 7 * π / 4 := by
  sorry

end cos_alpha_plus_beta_l163_163159


namespace solve_real_solution_l163_163693

theorem solve_real_solution:
  ∀ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔
           (x = 4 + Real.sqrt 57) ∨ (x = 4 - Real.sqrt 57) :=
by
  sorry

end solve_real_solution_l163_163693


namespace reciprocal_of_neg_two_l163_163710

theorem reciprocal_of_neg_two : ∀ x : ℝ, x = -2 → (1 / x) = -1 / 2 :=
by
  intro x h
  rw [h]
  norm_num

end reciprocal_of_neg_two_l163_163710


namespace count_valid_n_l163_163780

theorem count_valid_n : ∃ (n : ℕ), n < 200 ∧ (∃ (m : ℕ), (m % 4 = 0) ∧ (∃ (k : ℤ), n = 4 * k + 2 ∧ m = 4 * k * (k + 1))) ∧ (∃ k_range : ℕ, k_range = 50) :=
sorry

end count_valid_n_l163_163780


namespace convert_rectangular_to_spherical_l163_163587

theorem convert_rectangular_to_spherical :
  ∀ (x y z : ℝ) (ρ θ φ : ℝ),
    (x, y, z) = (2, -2 * Real.sqrt 2, 2) →
    ρ = Real.sqrt (x^2 + y^2 + z^2) →
    z = ρ * Real.cos φ →
    x = ρ * Real.sin φ * Real.cos θ →
    y = ρ * Real.sin φ * Real.sin θ →
    0 < ρ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi →
    (ρ, θ, φ) = (4, 2 * Real.pi - Real.arcsin (Real.sqrt 6 / 3), Real.pi / 3) :=
by
  intros x y z ρ θ φ H Hρ Hφ Hθ1 Hθ2 Hconditions
  sorry

end convert_rectangular_to_spherical_l163_163587


namespace average_speed_is_70_l163_163242

noncomputable def average_speed (d₁ d₂ d₃ d₄ t₁ t₂ t₃ t₄ : ℝ) : ℝ :=
  (d₁ + d₂ + d₃ + d₄) / (t₁ + t₂ + t₃ + t₄)

theorem average_speed_is_70 :
  let d₁ := 30
  let s₁ := 60
  let t₁ := d₁ / s₁
  let d₂ := 35
  let s₂ := 70
  let t₂ := d₂ / s₂
  let d₃ := 80
  let t₃ := 1
  let s₃ := d₃ / t₃
  let s₄ := 55
  let t₄ := 20/60.0
  let d₄ := s₄ * t₄
  average_speed d₁ d₂ d₃ d₄ t₁ t₂ t₃ t₄ = 70 :=
by
  sorry

end average_speed_is_70_l163_163242


namespace find_f2_l163_163247

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder function definition

theorem find_f2 (h : ∀ x : ℝ, f x + 2 * f (1 - x) = x^3 + 1) : f 2 = -3 :=
by
  -- Lean proof goes here
  sorry

end find_f2_l163_163247


namespace unique_solution_eqn_l163_163633

theorem unique_solution_eqn (a : ℝ) :
  (∃! x : ℝ, 3^(x^2 + 6 * a * x + 9 * a^2) = a * x^2 + 6 * a^2 * x + 9 * a^3 + a^2 - 4 * a + 4) ↔ (a = 1) :=
by
  sorry

end unique_solution_eqn_l163_163633


namespace share_a_is_240_l163_163256

def total_profit : ℕ := 630

def initial_investment_a : ℕ := 3000
def initial_investment_b : ℕ := 4000

def months_a1 : ℕ := 8
def months_a2 : ℕ := 4
def investment_a1 : ℕ := initial_investment_a * months_a1
def investment_a2 : ℕ := (initial_investment_a - 1000) * months_a2
def total_investment_a : ℕ := investment_a1 + investment_a2

def months_b1 : ℕ := 8
def months_b2 : ℕ := 4
def investment_b1 : ℕ := initial_investment_b * months_b1
def investment_b2 : ℕ := (initial_investment_b + 1000) * months_b2
def total_investment_b : ℕ := investment_b1 + investment_b2

def ratio_a : ℕ := 8
def ratio_b : ℕ := 13
def total_ratio : ℕ := ratio_a + ratio_b

noncomputable def share_a (total_profit : ℕ) (ratio_a ratio_total : ℕ) : ℕ :=
  (ratio_a * total_profit) / ratio_total

theorem share_a_is_240 :
  share_a total_profit ratio_a total_ratio = 240 :=
by
  sorry

end share_a_is_240_l163_163256


namespace height_of_brick_l163_163819

-- Definitions of wall dimensions
def L_w : ℝ := 700
def W_w : ℝ := 600
def H_w : ℝ := 22.5

-- Number of bricks
def n : ℝ := 5600

-- Definitions of brick dimensions (length and width)
def L_b : ℝ := 25
def W_b : ℝ := 11.25

-- Main theorem: Prove the height of each brick
theorem height_of_brick : ∃ h : ℝ, h = 6 :=
by
  -- Will add the proof steps here eventually
  sorry

end height_of_brick_l163_163819


namespace compute_combination_l163_163450

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end compute_combination_l163_163450


namespace volume_at_10_l163_163699

noncomputable def gas_volume (T : ℝ) : ℝ :=
  if T = 30 then 40 else 40 - (30 - T) / 5 * 5

theorem volume_at_10 :
  gas_volume 10 = 20 :=
by
  simp [gas_volume]
  sorry

end volume_at_10_l163_163699


namespace range_of_b_l163_163802

theorem range_of_b (b : ℝ) : 
  (¬ (4 ≤ 3 * 3 + b) ∧ (4 ≤ 3 * 4 + b)) ↔ (-8 ≤ b ∧ b < -5) := 
by
  sorry

end range_of_b_l163_163802


namespace magnitude_of_linear_combination_is_sqrt_65_l163_163021

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (2 * m - 1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (-2, 3 * m - 2)
noncomputable def perpendicular (u v : ℝ × ℝ) : Prop := (u.1 * v.1 + u.2 * v.2 = 0)

theorem magnitude_of_linear_combination_is_sqrt_65 (m : ℝ) 
  (h_perpendicular : perpendicular (vector_a m) (vector_b m)) : 
  ‖((2 : ℝ) • (vector_a 1) - (3 : ℝ) • (vector_b 1))‖ = Real.sqrt 65 := 
by
  sorry

end magnitude_of_linear_combination_is_sqrt_65_l163_163021


namespace consecutive_even_product_l163_163927

theorem consecutive_even_product (x : ℤ) (h : x * (x + 2) = 224) : x * (x + 2) = 224 := by
  sorry

end consecutive_even_product_l163_163927


namespace smallest_value_of_n_l163_163119

theorem smallest_value_of_n : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n + 6) % 7 = 0 ∧ (n - 9) % 4 = 0 ∧ n = 113 :=
by
  sorry

end smallest_value_of_n_l163_163119


namespace find_x_l163_163958

variable (x : ℝ)

theorem find_x (h : (15 - 2 + 4 / 1 / 2) * x = 77) : x = 77 / (15 - 2 + 4 / 1 / 2) :=
by sorry

end find_x_l163_163958


namespace dress_design_count_l163_163714

-- Definitions of the given conditions
def number_of_colors : Nat := 4
def number_of_patterns : Nat := 5

-- Statement to prove the total number of unique dress designs
theorem dress_design_count :
  number_of_colors * number_of_patterns = 20 := by
  sorry

end dress_design_count_l163_163714


namespace percentage_greater_l163_163459

theorem percentage_greater (x : ℝ) (h1 : x = 96) (h2 : x > 80) : ((x - 80) / 80) * 100 = 20 :=
by
  sorry

end percentage_greater_l163_163459


namespace radius_of_circle_of_roots_l163_163077

theorem radius_of_circle_of_roots (z : ℂ)
  (h : (z + 2)^6 = 64 * z^6) :
  ∃ r : ℝ, r = 4 / 3 ∧ ∀ z, (z + 2)^6 = 64 * z^6 →
  abs (z + 2) = (4 / 3 : ℝ) * abs z :=
by
  sorry

end radius_of_circle_of_roots_l163_163077


namespace ratio_of_time_spent_l163_163165

theorem ratio_of_time_spent {total_minutes type_a_minutes type_b_minutes : ℝ}
  (h1 : total_minutes = 180)
  (h2 : type_a_minutes = 32.73)
  (h3 : type_b_minutes = total_minutes - type_a_minutes) :
  type_a_minutes / type_a_minutes = 1 ∧ type_b_minutes / type_a_minutes = 4.5 := by
  sorry

end ratio_of_time_spent_l163_163165


namespace shorter_side_length_l163_163932

theorem shorter_side_length (L W : ℝ) (h₁ : L * W = 120) (h₂ : 2 * L + 2 * W = 46) : L = 8 ∨ W = 8 := 
by 
  sorry

end shorter_side_length_l163_163932


namespace leo_average_speed_last_segment_l163_163952

theorem leo_average_speed_last_segment :
  let total_distance := 135
  let total_time_hr := 135 / 60.0
  let segment_time_hr := 45 / 60.0
  let first_segment_distance := 55 * segment_time_hr
  let second_segment_distance := 70 * segment_time_hr
  let last_segment_distance := total_distance - (first_segment_distance + second_segment_distance)
  last_segment_distance / segment_time_hr = 55 :=
by
  sorry

end leo_average_speed_last_segment_l163_163952


namespace linda_total_miles_l163_163898

def calculate_total_miles (x : ℕ) : ℕ :=
  (60 / x) + (60 / (x + 4)) + (60 / (x + 8)) + (60 / (x + 12)) + (60 / (x + 16))

theorem linda_total_miles (x : ℕ) (hx1 : x > 0)
(hdx2 : 60 % x = 0)
(hdx3 : 60 % (x + 4) = 0) 
(hdx4 : 60 % (x + 8) = 0) 
(hdx5 : 60 % (x + 12) = 0) 
(hdx6 : 60 % (x + 16) = 0) :
  calculate_total_miles x = 33 := by
  sorry

end linda_total_miles_l163_163898


namespace dana_hours_sunday_l163_163223

-- Define the constants given in the problem
def hourly_rate : ℝ := 13
def hours_worked_friday : ℝ := 9
def hours_worked_saturday : ℝ := 10
def total_earnings : ℝ := 286

-- Define the function to compute total earnings from worked hours and hourly rate
def earnings (hours : ℝ) (rate : ℝ) : ℝ := hours * rate

-- Define the proof problem to show the number of hours worked on Sunday
theorem dana_hours_sunday (hours_sunday : ℝ) :
  earnings hours_worked_friday hourly_rate
  + earnings hours_worked_saturday hourly_rate
  + earnings hours_sunday hourly_rate = total_earnings ->
  hours_sunday = 3 :=
by
  sorry -- proof to be filled in

end dana_hours_sunday_l163_163223


namespace alex_buys_15_pounds_of_rice_l163_163034

theorem alex_buys_15_pounds_of_rice (r b : ℝ) 
  (h1 : r + b = 30)
  (h2 : 75 * r + 35 * b = 1650) : 
  r = 15.0 := sorry

end alex_buys_15_pounds_of_rice_l163_163034


namespace tan_of_acute_angle_l163_163810

theorem tan_of_acute_angle (α : ℝ) (h1 : α > 0 ∧ α < π / 2) (h2 : Real.cos (π / 2 + α) = -3/5) : Real.tan α = 3 / 4 :=
by
  sorry

end tan_of_acute_angle_l163_163810


namespace copper_needed_l163_163897

theorem copper_needed (T : ℝ) (lead_percentage : ℝ) (lead_weight : ℝ) (copper_percentage : ℝ) 
  (h_lead_percentage : lead_percentage = 0.25)
  (h_lead_weight : lead_weight = 5)
  (h_copper_percentage : copper_percentage = 0.60)
  (h_total_weight : T = lead_weight / lead_percentage) :
  copper_percentage * T = 12 := 
by
  sorry

end copper_needed_l163_163897


namespace block3_reaches_target_l163_163174

-- Type representing the position of a block on a 3x7 grid
structure Position where
  row : Nat
  col : Nat
  deriving DecidableEq, Repr

-- Defining the initial positions of blocks
def Block1Start : Position := ⟨2, 2⟩
def Block2Start : Position := ⟨3, 5⟩
def Block3Start : Position := ⟨1, 4⟩

-- The target position in the center of the board
def TargetPosition : Position := ⟨3, 5⟩

-- A function to represent if blocks collide or not
def canMove (current : Position) (target : Position) (blocks : List Position) : Prop :=
  target.row < 3 ∧ target.col < 7 ∧ ¬(target ∈ blocks)

-- Main theorem stating the goal
theorem block3_reaches_target : ∃ (steps : Nat → Position), steps 0 = Block3Start ∧ steps 7 = TargetPosition :=
  sorry

end block3_reaches_target_l163_163174


namespace points_on_line_relationship_l163_163604

theorem points_on_line_relationship :
  let m := 2 * Real.sqrt 2 + 1
  let n := 4
  m < n :=
by
  sorry

end points_on_line_relationship_l163_163604


namespace product_of_sum_and_reciprocal_ge_four_l163_163020

theorem product_of_sum_and_reciprocal_ge_four (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a + b) * (1 / a + 1 / b) ≥ 4 :=
sorry

end product_of_sum_and_reciprocal_ge_four_l163_163020


namespace jim_can_bake_loaves_l163_163758

-- Define the amounts of flour in different locations
def flour_cupboard : ℕ := 200  -- in grams
def flour_counter : ℕ := 100   -- in grams
def flour_pantry : ℕ := 100    -- in grams

-- Define the amount of flour required for one loaf of bread
def flour_per_loaf : ℕ := 200  -- in grams

-- Total loaves Jim can bake
def loaves_baked (f_c f_k f_p f_r : ℕ) : ℕ :=
  (f_c + f_k + f_p) / f_r

-- Theorem to prove the solution
theorem jim_can_bake_loaves :
  loaves_baked flour_cupboard flour_counter flour_pantry flour_per_loaf = 2 :=
by
  -- Proof is omitted
  sorry

end jim_can_bake_loaves_l163_163758


namespace number_of_students_l163_163989

theorem number_of_students (total_stars : ℕ) (stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : total_stars / stars_per_student = 124 :=
by
  sorry

end number_of_students_l163_163989


namespace fraction_zero_implies_x_is_two_l163_163626

theorem fraction_zero_implies_x_is_two {x : ℝ} (hfrac : (2 - |x|) / (x + 2) = 0) (hdenom : x ≠ -2) : x = 2 :=
by
  sorry

end fraction_zero_implies_x_is_two_l163_163626


namespace number_of_animals_per_aquarium_l163_163870

variable (aq : ℕ) (ani : ℕ) (a : ℕ)

axiom condition1 : aq = 26
axiom condition2 : ani = 52
axiom condition3 : ani = aq * a

theorem number_of_animals_per_aquarium : a = 2 :=
by
  sorry

end number_of_animals_per_aquarium_l163_163870
