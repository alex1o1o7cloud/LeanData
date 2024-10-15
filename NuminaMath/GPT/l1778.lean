import Mathlib

namespace NUMINAMATH_GPT_find_q_l1778_177843

noncomputable def q (x : ℝ) : ℝ := -2 * x^4 + 10 * x^3 - 2 * x^2 + 7 * x + 3

theorem find_q :
  ∀ x : ℝ,
  q x + (2 * x^4 - 5 * x^2 + 8 * x + 3) = (10 * x^3 - 7 * x^2 + 15 * x + 6) :=
by
  intro x
  unfold q
  sorry

end NUMINAMATH_GPT_find_q_l1778_177843


namespace NUMINAMATH_GPT_solve_for_q_l1778_177897

theorem solve_for_q (k l q : ℝ) 
  (h1 : 3 / 4 = k / 48)
  (h2 : 3 / 4 = (k + l) / 56)
  (h3 : 3 / 4 = (q - l) / 160) :
  q = 126 :=
  sorry

end NUMINAMATH_GPT_solve_for_q_l1778_177897


namespace NUMINAMATH_GPT_division_addition_problem_l1778_177801

theorem division_addition_problem :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := by
  sorry

end NUMINAMATH_GPT_division_addition_problem_l1778_177801


namespace NUMINAMATH_GPT_angle_of_inclination_vert_line_l1778_177811

theorem angle_of_inclination_vert_line (x : ℝ) (h : x = -1) : 
  ∃ θ : ℝ, θ = 90 := 
by
  sorry

end NUMINAMATH_GPT_angle_of_inclination_vert_line_l1778_177811


namespace NUMINAMATH_GPT_max_colors_4x4_grid_l1778_177805

def cell := (Fin 4) × (Fin 4)
def color := Fin 8

def valid_coloring (f : cell → color) : Prop :=
∀ c1 c2 : color, (c1 ≠ c2) →
(∃ i : Fin 4, ∃ j1 j2 : Fin 4, j1 ≠ j2 ∧ f (i, j1) = c1 ∧ f (i, j2) = c2) ∨ 
(∃ j : Fin 4, ∃ i1 i2 : Fin 4, i1 ≠ i2 ∧ f (i1, j) = c1 ∧ f (i2, j) = c2)

theorem max_colors_4x4_grid : ∃ (f : cell → color), valid_coloring f :=
sorry

end NUMINAMATH_GPT_max_colors_4x4_grid_l1778_177805


namespace NUMINAMATH_GPT_find_d_l1778_177848

theorem find_d (x y d : ℕ) (h_midpoint : (1 + 5)/2 = 3 ∧ (3 + 11)/2 = 7) 
  : x + y = d ↔ d = 10 := 
sorry

end NUMINAMATH_GPT_find_d_l1778_177848


namespace NUMINAMATH_GPT_standard_equation_of_circle_l1778_177851

/-- A circle with radius 2, center in the fourth quadrant, and tangent to the lines x = 0 and x + y = 2√2 has the standard equation (x - 2)^2 + (y + 2)^2 = 4. -/
theorem standard_equation_of_circle :
  ∃ a, a > 0 ∧ (∀ x y : ℝ, ((x - a)^2 + (y + 2)^2 = 4) ∧ 
                        (a > 0) ∧ 
                        (x = 0 → a = 2) ∧
                        x + y = 2 * Real.sqrt 2 → a = 2) := 
by
  sorry

end NUMINAMATH_GPT_standard_equation_of_circle_l1778_177851


namespace NUMINAMATH_GPT_frank_picked_apples_l1778_177838

theorem frank_picked_apples (F : ℕ) 
  (susan_picked : ℕ := 3 * F) 
  (susan_left : ℕ := susan_picked / 2) 
  (frank_left : ℕ := 2 * F / 3) 
  (total_left : susan_left + frank_left = 78) : 
  F = 36 :=
sorry

end NUMINAMATH_GPT_frank_picked_apples_l1778_177838


namespace NUMINAMATH_GPT_base3_to_base10_conversion_l1778_177854

theorem base3_to_base10_conversion : 
  (1 * 3^4 + 0 * 3^3 + 2 * 3^2 + 0 * 3^1 + 1 * 3^0 = 100) :=
by 
  sorry

end NUMINAMATH_GPT_base3_to_base10_conversion_l1778_177854


namespace NUMINAMATH_GPT_calculate_speed_l1778_177804

variable (time : ℝ) (distance : ℝ)

theorem calculate_speed (h_time : time = 5) (h_distance : distance = 500) : 
  distance / time = 100 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_speed_l1778_177804


namespace NUMINAMATH_GPT_smallest_x_value_l1778_177874

theorem smallest_x_value : 
  ∃ x : ℝ, 
  (∀ y : ℝ, (y^2 - 5 * y - 84) / (y - 9) = 4 / (y + 6) → y >= (x)) ∧ 
  (x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 6) ∧ 
  x = ( - 13 - Real.sqrt 17 ) / 2 := 
sorry

end NUMINAMATH_GPT_smallest_x_value_l1778_177874


namespace NUMINAMATH_GPT_lights_on_fourth_tier_l1778_177817

def number_lights_topmost_tier (total_lights : ℕ) : ℕ :=
  total_lights / 127

def number_lights_tier (tier : ℕ) (lights_topmost : ℕ) : ℕ :=
  2^(tier - 1) * lights_topmost

theorem lights_on_fourth_tier (total_lights : ℕ) (H : total_lights = 381) : number_lights_tier 4 (number_lights_topmost_tier total_lights) = 24 :=
by
  rw [H]
  sorry

end NUMINAMATH_GPT_lights_on_fourth_tier_l1778_177817


namespace NUMINAMATH_GPT_son_present_age_l1778_177820

theorem son_present_age (S F : ℕ) (h1 : F = S + 34) (h2 : F + 2 = 2 * (S + 2)) : S = 32 :=
by
  sorry

end NUMINAMATH_GPT_son_present_age_l1778_177820


namespace NUMINAMATH_GPT_solution_set_empty_iff_a_in_range_l1778_177806

theorem solution_set_empty_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, ¬ (2 * x^2 + a * x + 2 < 0)) ↔ (-4 ≤ a ∧ a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_empty_iff_a_in_range_l1778_177806


namespace NUMINAMATH_GPT_find_y_l1778_177878

variables (ABC ACB BAC : ℝ)
variables (CDE ADE EAD AED DEB y : ℝ)

-- Conditions
axiom angle_ABC : ABC = 45
axiom angle_ACB : ACB = 90
axiom angle_BAC_eq : BAC = 180 - ABC - ACB
axiom angle_CDE : CDE = 72
axiom angle_ADE_eq : ADE = 180 - CDE
axiom angle_EAD : EAD = 45
axiom angle_AED_eq : AED = 180 - ADE - EAD
axiom angle_DEB_eq : DEB = 180 - AED
axiom y_eq : y = DEB

-- Goal
theorem find_y : y = 153 :=
by {
  -- Here we would proceed with the proof using the established axioms.
  sorry
}

end NUMINAMATH_GPT_find_y_l1778_177878


namespace NUMINAMATH_GPT_farm_corn_cobs_l1778_177846

theorem farm_corn_cobs (rows_field1 rows_field2 cobs_per_row : Nat) (h1 : rows_field1 = 13) (h2 : rows_field2 = 16) (h3 : cobs_per_row = 4) : rows_field1 * cobs_per_row + rows_field2 * cobs_per_row = 116 := by
  sorry

end NUMINAMATH_GPT_farm_corn_cobs_l1778_177846


namespace NUMINAMATH_GPT_machine_working_time_l1778_177887

theorem machine_working_time (y : ℝ) :
  (1 / (y + 4) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) → y = 2 :=
by
  sorry

end NUMINAMATH_GPT_machine_working_time_l1778_177887


namespace NUMINAMATH_GPT_second_term_of_geometric_series_l1778_177875

theorem second_term_of_geometric_series (a r S term2 : ℝ) 
  (h1 : r = 1 / 4)
  (h2 : S = 40)
  (h3 : S = a / (1 - r))
  (h4 : term2 = a * r) : 
  term2 = 7.5 := 
  by
  sorry

end NUMINAMATH_GPT_second_term_of_geometric_series_l1778_177875


namespace NUMINAMATH_GPT_melanie_gave_3_plums_to_sam_l1778_177849

theorem melanie_gave_3_plums_to_sam 
  (initial_plums : ℕ) 
  (plums_left : ℕ) 
  (plums_given : ℕ) 
  (h1 : initial_plums = 7) 
  (h2 : plums_left = 4) 
  (h3 : plums_left + plums_given = initial_plums) : 
  plums_given = 3 :=
by 
  sorry

end NUMINAMATH_GPT_melanie_gave_3_plums_to_sam_l1778_177849


namespace NUMINAMATH_GPT_symmetric_line_eq_l1778_177883

theorem symmetric_line_eq (x y : ℝ) :
  (y = 2 * x + 3) → (y - 1 = x + 1) → (x - 2 * y = 0) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_symmetric_line_eq_l1778_177883


namespace NUMINAMATH_GPT_edward_rides_eq_8_l1778_177898

-- Define the initial conditions
def initial_tickets : ℕ := 79
def spent_tickets : ℕ := 23
def cost_per_ride : ℕ := 7

-- Define the remaining tickets after spending at the booth
def remaining_tickets : ℕ := initial_tickets - spent_tickets

-- Define the number of rides Edward could go on
def number_of_rides : ℕ := remaining_tickets / cost_per_ride

-- The goal is to prove that the number of rides is equal to 8.
theorem edward_rides_eq_8 : number_of_rides = 8 := by sorry

end NUMINAMATH_GPT_edward_rides_eq_8_l1778_177898


namespace NUMINAMATH_GPT_fraction_equality_l1778_177884

-- Defining the hypotheses and the goal
theorem fraction_equality (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l1778_177884


namespace NUMINAMATH_GPT_election_winning_votes_l1778_177858

noncomputable def total_votes (x y : ℕ) (p : ℚ) : ℚ := 
  (x + y) / (1 - p)

noncomputable def winning_votes (x y : ℕ) (p : ℚ) : ℚ :=
  p * total_votes x y p

theorem election_winning_votes :
  winning_votes 2136 7636 0.54336448598130836 = 11628 := 
by
  sorry

end NUMINAMATH_GPT_election_winning_votes_l1778_177858


namespace NUMINAMATH_GPT_finite_cuboid_blocks_l1778_177824

/--
Prove that there are only finitely many cuboid blocks with integer dimensions a, b, c
such that abc = 2(a - 2)(b - 2)(c - 2) and c ≤ b ≤ a.
-/
theorem finite_cuboid_blocks :
  ∃ (S : Finset (ℤ × ℤ × ℤ)), ∀ (a b c : ℤ), (abc = 2 * (a - 2) * (b - 2) * (c - 2)) → (c ≤ b) → (b ≤ a) → (a, b, c) ∈ S := 
by
  sorry

end NUMINAMATH_GPT_finite_cuboid_blocks_l1778_177824


namespace NUMINAMATH_GPT_diagonal_of_larger_screen_l1778_177890

theorem diagonal_of_larger_screen (d : ℝ) 
  (h1 : ∃ s : ℝ, s^2 = 20^2 + 42) 
  (h2 : ∀ s, d = s * Real.sqrt 2) : 
  d = Real.sqrt 884 :=
by
  sorry

end NUMINAMATH_GPT_diagonal_of_larger_screen_l1778_177890


namespace NUMINAMATH_GPT_storks_count_l1778_177837

theorem storks_count (B S : ℕ) (h1 : B = 3) (h2 : B + 2 = S + 1) : S = 4 :=
by
  sorry

end NUMINAMATH_GPT_storks_count_l1778_177837


namespace NUMINAMATH_GPT_two_non_coincident_planes_divide_space_l1778_177829

-- Define conditions for non-coincident planes
def non_coincident_planes (P₁ P₂ : Plane) : Prop :=
  ¬(P₁ = P₂)

-- Define the main theorem based on the conditions and the question
theorem two_non_coincident_planes_divide_space (P₁ P₂ : Plane) 
  (h : non_coincident_planes P₁ P₂) :
  ∃ n : ℕ, n = 3 ∨ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_two_non_coincident_planes_divide_space_l1778_177829


namespace NUMINAMATH_GPT_find_larger_number_l1778_177886

theorem find_larger_number (L S : ℤ) (h₁ : L - S = 1000) (h₂ : L = 10 * S + 10) : L = 1110 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l1778_177886


namespace NUMINAMATH_GPT_percentage_increase_on_sale_l1778_177888

theorem percentage_increase_on_sale (P S : ℝ) (hP : P ≠ 0) (hS : S ≠ 0)
  (h_price_reduction : (0.8 : ℝ) * P * S * (1 + (X / 100)) = 1.44 * P * S) :
  X = 80 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_on_sale_l1778_177888


namespace NUMINAMATH_GPT_value_of_5_S_3_l1778_177892

def operation_S (a b : ℝ) : ℝ := 4 * a + 6 * b - 2 * a * b

theorem value_of_5_S_3 : operation_S 5 3 = 8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_5_S_3_l1778_177892


namespace NUMINAMATH_GPT_sum_a_b_c_l1778_177842

theorem sum_a_b_c (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 390) (h2: a * b + b * c + c * a = 5) : a + b + c = 20 ∨ a + b + c = -20 := 
by 
  sorry

end NUMINAMATH_GPT_sum_a_b_c_l1778_177842


namespace NUMINAMATH_GPT_volume_of_cuboid_l1778_177835

variable (a b c : ℝ)

def is_cuboid_adjacent_faces (a b c : ℝ) := a * b = 3 ∧ a * c = 5 ∧ b * c = 15

theorem volume_of_cuboid (a b c : ℝ) (h : is_cuboid_adjacent_faces a b c) :
  a * b * c = 15 := by
  sorry

end NUMINAMATH_GPT_volume_of_cuboid_l1778_177835


namespace NUMINAMATH_GPT_solve_ordered_pairs_l1778_177869

theorem solve_ordered_pairs (a b : ℕ) (h : a^2 + b^2 = ab * (a + b)) : 
  (a, b) = (1, 1) ∨ (a, b) = (1, 1) :=
by 
  sorry

end NUMINAMATH_GPT_solve_ordered_pairs_l1778_177869


namespace NUMINAMATH_GPT_middle_box_label_l1778_177856

/--
Given a sequence of 23 boxes in a row on the table, where each box has a label indicating either
  "There is no prize here" or "The prize is in a neighboring box",
and it is known that exactly one of these statements is true.
Prove that the label on the middle box (the 12th box) says "The prize is in the adjacent box."
-/
theorem middle_box_label :
  ∃ (boxes : Fin 23 → Prop) (labels : Fin 23 → String),
    (∀ i, labels i = "There is no prize here" ∨ labels i = "The prize is in a neighboring box") ∧
    (∃! i : Fin 23, boxes i ∧ (labels i = "The prize is in a neighboring box")) →
    labels ⟨11, sorry⟩ = "The prize is in a neighboring box" :=
sorry

end NUMINAMATH_GPT_middle_box_label_l1778_177856


namespace NUMINAMATH_GPT_slices_per_person_l1778_177807

namespace PizzaProblem

def pizzas : Nat := 3
def slices_per_pizza : Nat := 8
def coworkers : Nat := 12

theorem slices_per_person : (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end PizzaProblem

end NUMINAMATH_GPT_slices_per_person_l1778_177807


namespace NUMINAMATH_GPT_find_numbers_l1778_177840

theorem find_numbers (x y : ℕ) (h1 : x / y = 3) (h2 : (x^2 + y^2) / (x + y) = 5) : 
  x = 6 ∧ y = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_numbers_l1778_177840


namespace NUMINAMATH_GPT_length_of_rect_box_l1778_177852

noncomputable def length_of_box (height : ℝ) (width : ℝ) (volume : ℝ) : ℝ :=
  volume / (width * height)

theorem length_of_rect_box :
  (length_of_box 0.5 25 (6000 / 7.48052)) = 64.1624 :=
by
  unfold length_of_box
  norm_num
  sorry

end NUMINAMATH_GPT_length_of_rect_box_l1778_177852


namespace NUMINAMATH_GPT_solve_for_x_l1778_177828

theorem solve_for_x (x : ℝ) (h₁ : (7 * x) / (x + 4) - 4 / (x + 4) = 2 / (x + 4)) (h₂ : x ≠ -4) : x = 6 / 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1778_177828


namespace NUMINAMATH_GPT_most_likely_event_is_C_l1778_177873

open Classical

noncomputable def total_events : ℕ := 6 * 6

noncomputable def P_A : ℚ := 7 / 36
noncomputable def P_B : ℚ := 18 / 36
noncomputable def P_C : ℚ := 1
noncomputable def P_D : ℚ := 0

theorem most_likely_event_is_C :
  P_C > P_A ∧ P_C > P_B ∧ P_C > P_D := by
  sorry

end NUMINAMATH_GPT_most_likely_event_is_C_l1778_177873


namespace NUMINAMATH_GPT_nba_conferences_division_l1778_177880

theorem nba_conferences_division (teams : ℕ) (games_per_team : ℕ) (E : ℕ) :
  teams = 30 ∧ games_per_team = 82 ∧
  (teams = E + (teams - E)) ∧
  (games_per_team / 2 * E) + (games_per_team / 2 * (teams - E))  ≠ teams * games_per_team / 2 :=
by
  sorry

end NUMINAMATH_GPT_nba_conferences_division_l1778_177880


namespace NUMINAMATH_GPT_find_a_l1778_177870

variable {f : ℝ → ℝ}

-- Conditions
variables (a : ℝ) (domain : Set ℝ := Set.Ioo (3 - 2 * a) (a + 1))
variable (even_f : ∀ x, f (x + 1) = f (- (x + 1)))

-- The theorem stating the problem
theorem find_a (h : ∀ x, x ∈ domain ↔ x ∈ Set.Ioo (3 - 2 * a) (a + 1)) : a = 2 := by
  sorry

end NUMINAMATH_GPT_find_a_l1778_177870


namespace NUMINAMATH_GPT_cubic_sum_l1778_177830

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 :=
  sorry

end NUMINAMATH_GPT_cubic_sum_l1778_177830


namespace NUMINAMATH_GPT_algebraic_expression_value_l1778_177816

theorem algebraic_expression_value (a b : ℝ) (h1 : a + b = 8) (h2 : a * b = 9) : a^2 - 3 * a * b + b^2 = 19 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1778_177816


namespace NUMINAMATH_GPT_equation_for_pears_l1778_177864

-- Define the conditions
def pearDist1 (x : ℕ) : ℕ := 4 * x + 12
def pearDist2 (x : ℕ) : ℕ := 6 * x

-- State the theorem to be proved
theorem equation_for_pears (x : ℕ) : pearDist1 x = pearDist2 x :=
by
  sorry

end NUMINAMATH_GPT_equation_for_pears_l1778_177864


namespace NUMINAMATH_GPT_part1_part2_l1778_177819

-- Definitions of propositions P and q
def P (t : ℝ) : Prop := (4 - t > t - 1 ∧ t - 1 > 0)
def q (a t : ℝ) : Prop := t^2 - (a+3)*t + (a+2) < 0

-- Part 1: If P is true, find the range of t.
theorem part1 (t : ℝ) (hP : P t) : 1 < t ∧ t < 5/2 :=
by sorry

-- Part 2: If P is a sufficient but not necessary condition for q, find the range of a.
theorem part2 (a : ℝ) 
  (hP_q : ∀ t, P t → q a t) 
  (hsubset : ∀ t, 1 < t ∧ t < 5/2 → q a t) 
  : a > 1/2 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1778_177819


namespace NUMINAMATH_GPT_jordan_rectangle_length_l1778_177833

def rectangle_area (length width : ℝ) : ℝ := length * width

theorem jordan_rectangle_length :
  let carol_length := 8
  let carol_width := 15
  let jordan_width := 30
  let carol_area := rectangle_area carol_length carol_width
  ∃ jordan_length, rectangle_area jordan_length jordan_width = carol_area →
  jordan_length = 4 :=
by
  sorry

end NUMINAMATH_GPT_jordan_rectangle_length_l1778_177833


namespace NUMINAMATH_GPT_Katie_average_monthly_balance_l1778_177868

def balances : List ℕ := [120, 240, 180, 180, 240]

def average (l : List ℕ) : ℕ := l.sum / l.length

theorem Katie_average_monthly_balance : average balances = 192 :=
by
  sorry

end NUMINAMATH_GPT_Katie_average_monthly_balance_l1778_177868


namespace NUMINAMATH_GPT_find_average_speed_l1778_177827

theorem find_average_speed :
  ∃ v : ℝ, (880 / v) - (880 / (v + 10)) = 2 ∧ v = 61.5 :=
by
  sorry

end NUMINAMATH_GPT_find_average_speed_l1778_177827


namespace NUMINAMATH_GPT_find_a_l1778_177834

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f : ℝ → ℝ := sorry -- The definition of f is to be handled in the proof

theorem find_a (a : ℝ) (h1 : is_odd_function f)
  (h2 : ∀ x : ℝ, 0 < x → f x = 2^(x - a) - 2 / (x + 1))
  (h3 : f (-1) = 3 / 4) : a = 3 :=
sorry

end NUMINAMATH_GPT_find_a_l1778_177834


namespace NUMINAMATH_GPT_find_k_n_l1778_177850

theorem find_k_n (k n : ℕ) (h_kn_pos : 0 < k ∧ 0 < n) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_k_n_l1778_177850


namespace NUMINAMATH_GPT_circle_radius_correct_l1778_177823

noncomputable def radius_of_circle 
  (side_length : ℝ)
  (angle_tangents : ℝ)
  (sin_18 : ℝ) : ℝ := 
  sorry

theorem circle_radius_correct 
  (side_length : ℝ := 6 + 2 * Real.sqrt 5)
  (angle_tangents : ℝ := 36)
  (sin_18 : ℝ := (Real.sqrt 5 - 1) / 4) :
  radius_of_circle side_length angle_tangents sin_18 = 
  2 * (2 * Real.sqrt 2 + Real.sqrt 5 - 1) :=
sorry

end NUMINAMATH_GPT_circle_radius_correct_l1778_177823


namespace NUMINAMATH_GPT_amount_with_r_l1778_177876

theorem amount_with_r (p q r : ℕ) (h1 : p + q + r = 7000) (h2 : r = (2 * (p + q)) / 3) : r = 2800 :=
sorry

end NUMINAMATH_GPT_amount_with_r_l1778_177876


namespace NUMINAMATH_GPT_height_difference_l1778_177862

def pine_tree_height : ℚ := 12 + 1 / 4
def maple_tree_height : ℚ := 18 + 1 / 2

theorem height_difference :
  maple_tree_height - pine_tree_height = 6 + 1 / 4 :=
by sorry

end NUMINAMATH_GPT_height_difference_l1778_177862


namespace NUMINAMATH_GPT_find_initial_amount_l1778_177802

theorem find_initial_amount
  (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hA : A = 1050)
  (hR : R = 8)
  (hT : T = 5) :
  P = 750 :=
by
  have hSI : P * R * T / 100 = 1050 - P := sorry
  have hFormulaSimplified : P * 0.4 = 1050 - P := sorry
  have hFinal : P * 1.4 = 1050 := sorry
  exact sorry

end NUMINAMATH_GPT_find_initial_amount_l1778_177802


namespace NUMINAMATH_GPT_total_tickets_sales_l1778_177847

theorem total_tickets_sales:
    let student_ticket_price := 6
    let adult_ticket_price := 8
    let number_of_students := 20
    let number_of_adults := 12
    number_of_students * student_ticket_price + number_of_adults * adult_ticket_price = 216 :=
by
    intros
    sorry

end NUMINAMATH_GPT_total_tickets_sales_l1778_177847


namespace NUMINAMATH_GPT_sum_seven_l1778_177871

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

variables (a : ℕ → ℤ) (S : ℕ → ℤ)

axiom a2 : a 2 = 3
axiom a6 : a 6 = 11
axiom arithmetic_seq : arithmetic_sequence a
axiom sum_of_sequence : ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sum_seven : S 7 = 49 :=
sorry

end NUMINAMATH_GPT_sum_seven_l1778_177871


namespace NUMINAMATH_GPT_mean_of_combined_set_is_52_over_3_l1778_177812

noncomputable def mean_combined_set : ℚ := 
  let mean_set1 := 10
  let size_set1 := 4
  let mean_set2 := 21
  let size_set2 := 8
  let sum_set1 := mean_set1 * size_set1
  let sum_set2 := mean_set2 * size_set2
  let total_sum := sum_set1 + sum_set2
  let combined_size := size_set1 + size_set2
  let combined_mean := total_sum / combined_size
  combined_mean

theorem mean_of_combined_set_is_52_over_3 :
  mean_combined_set = 52 / 3 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_combined_set_is_52_over_3_l1778_177812


namespace NUMINAMATH_GPT_total_seashells_after_six_weeks_l1778_177899

theorem total_seashells_after_six_weeks :
  ∀ (a b : ℕ) 
  (initial_a : a = 50) 
  (initial_b : b = 30) 
  (next_a : ∀ k : ℕ, k > 0 → a + 20 = (a + 20) * k) 
  (next_b : ∀ k : ℕ, k > 0 → b * 2 = (b * 2) * k), 
  (a + 20 * 5) + (b * 2 ^ 5) = 1110 :=
by
  intros a b initial_a initial_b next_a next_b
  sorry

end NUMINAMATH_GPT_total_seashells_after_six_weeks_l1778_177899


namespace NUMINAMATH_GPT_elder_age_is_30_l1778_177839

/-- The ages of two persons differ by 16 years, and 6 years ago, the elder one was 3 times as old as the younger one. 
Prove that the present age of the elder person is 30 years. --/
theorem elder_age_is_30 (y e: ℕ) (h₁: e = y + 16) (h₂: e - 6 = 3 * (y - 6)) : e = 30 := 
sorry

end NUMINAMATH_GPT_elder_age_is_30_l1778_177839


namespace NUMINAMATH_GPT_replace_floor_cost_l1778_177821

-- Define the conditions
def floor_removal_cost : ℝ := 50
def new_floor_cost_per_sqft : ℝ := 1.25
def room_length : ℝ := 8
def room_width : ℝ := 7

-- Define the area of the room
def room_area : ℝ := room_length * room_width

-- Define the cost of the new floor
def new_floor_cost : ℝ := room_area * new_floor_cost_per_sqft

-- Define the total cost to replace the floor
def total_cost : ℝ := floor_removal_cost + new_floor_cost

-- State the proof problem
theorem replace_floor_cost : total_cost = 120 := by
  sorry

end NUMINAMATH_GPT_replace_floor_cost_l1778_177821


namespace NUMINAMATH_GPT_simplify_expression_l1778_177825

variable (x : ℝ)

theorem simplify_expression :
  ((3 * x - 2) * (5 * x ^ 12 + 3 * x ^ 11 + 5 * x ^ 10 + 3 * x ^ 9)) =
  (15 * x ^ 13 - x ^ 12 + 9 * x ^ 11 - x ^ 10 - 6 * x ^ 9) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1778_177825


namespace NUMINAMATH_GPT_translate_point_left_l1778_177836

def initial_point : ℝ × ℝ := (-2, -1)
def translation_left (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ := (p.1 - units, p.2)

theorem translate_point_left :
  translation_left initial_point 2 = (-4, -1) :=
by
  -- By definition and calculation
  -- Let p = initial_point
  -- x' = p.1 - 2,
  -- y' = p.2
  -- translation_left (-2, -1) 2 = (-4, -1)
  sorry

end NUMINAMATH_GPT_translate_point_left_l1778_177836


namespace NUMINAMATH_GPT_z_coordinate_of_point_on_line_l1778_177855

theorem z_coordinate_of_point_on_line (t : ℝ)
  (h₁ : (1 + 3 * t, 3 + 2 * t, 2 + 4 * t) = (x, 7, z))
  (h₂ : x = 1 + 3 * t) :
  z = 10 :=
sorry

end NUMINAMATH_GPT_z_coordinate_of_point_on_line_l1778_177855


namespace NUMINAMATH_GPT_sum_of_consecutive_natural_numbers_eq_three_digit_same_digits_l1778_177881

theorem sum_of_consecutive_natural_numbers_eq_three_digit_same_digits :
  ∃ n : ℕ, (1 + n) * n / 2 = 111 * 6 ∧ n = 36 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_natural_numbers_eq_three_digit_same_digits_l1778_177881


namespace NUMINAMATH_GPT_Sarah_copy_total_pages_l1778_177822

theorem Sarah_copy_total_pages (num_people : ℕ) (copies_per_person : ℕ) (pages_per_contract : ℕ)
  (h1 : num_people = 9) (h2 : copies_per_person = 2) (h3 : pages_per_contract = 20) :
  num_people * copies_per_person * pages_per_contract = 360 :=
by
  sorry

end NUMINAMATH_GPT_Sarah_copy_total_pages_l1778_177822


namespace NUMINAMATH_GPT_chess_tournament_distribution_l1778_177808

theorem chess_tournament_distribution 
    (students : List String)
    (games_played : Nat)
    (scores : List ℝ)
    (points_per_game : List ℝ)
    (unique_scores : ∀ (x y : ℝ), x ≠ y → scores.contains x → scores.contains y → x ≠ y)
    (first_place : String)
    (second_place : String)
    (third_place : String)
    (fourth_place : String)
    (andrey_wins_equal_sasha : ℝ)
    (total_points : ℝ)
    : 
    students = ["Andrey", "Vanya", "Dima", "Sasha"] ∧
    games_played = 6 ∧
    points_per_game = [1, 0.5, 0] ∧
    first_place = "Andrey" ∧
    second_place = "Dima" ∧
    third_place = "Vanya" ∧
    fourth_place = "Sasha" ∧
    scores = [4, 3.5, 2.5, 2] ∧
    andrey_wins_equal_sasha = 2 ∧
    total_points = 12 := 
sorry

end NUMINAMATH_GPT_chess_tournament_distribution_l1778_177808


namespace NUMINAMATH_GPT_candy_last_days_l1778_177800

variable (candy_from_neighbors candy_from_sister candy_per_day : ℕ)

theorem candy_last_days
  (h_candy_from_neighbors : candy_from_neighbors = 66)
  (h_candy_from_sister : candy_from_sister = 15)
  (h_candy_per_day : candy_per_day = 9) :
  let total_candy := candy_from_neighbors + candy_from_sister  
  (total_candy / candy_per_day) = 9 := by
  sorry

end NUMINAMATH_GPT_candy_last_days_l1778_177800


namespace NUMINAMATH_GPT_dentist_filling_cost_l1778_177832

variable (F : ℝ)
variable (total_bill : ℝ := 5 * F)
variable (cleaning_cost : ℝ := 70)
variable (extraction_cost : ℝ := 290)
variable (two_fillings_cost : ℝ := 2 * F)

theorem dentist_filling_cost :
  total_bill = cleaning_cost + two_fillings_cost + extraction_cost → 
  F = 120 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_dentist_filling_cost_l1778_177832


namespace NUMINAMATH_GPT_center_of_rotation_l1778_177860

noncomputable def f (z : ℂ) : ℂ := ((-1 - (Complex.I * Real.sqrt 3)) * z + (2 * Real.sqrt 3 - 12 * Complex.I)) / 2

theorem center_of_rotation :
  ∃ c : ℂ, f c = c ∧ c = -5 * Real.sqrt 3 / 2 - 7 / 2 * Complex.I :=
by
  sorry

end NUMINAMATH_GPT_center_of_rotation_l1778_177860


namespace NUMINAMATH_GPT_find_k_l1778_177815

theorem find_k (σ μ : ℝ) (hσ : σ = 2) (hμ : μ = 55) :
  ∃ k : ℝ, μ - k * σ > 48 ∧ k = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1778_177815


namespace NUMINAMATH_GPT_tan_330_eq_neg_sqrt3_div_3_l1778_177861

theorem tan_330_eq_neg_sqrt3_div_3 :
  Real.tan (330 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_GPT_tan_330_eq_neg_sqrt3_div_3_l1778_177861


namespace NUMINAMATH_GPT_range_of_a_l1778_177844

noncomputable def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
noncomputable def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) : a ≤ -1 / 2 ∨ a ≥ 2 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l1778_177844


namespace NUMINAMATH_GPT_chess_team_girls_count_l1778_177841

theorem chess_team_girls_count (B G : ℕ) 
  (h1 : B + G = 26) 
  (h2 : (3 / 4 : ℝ) * B + (1 / 4 : ℝ) * G = 13) : G = 13 := 
sorry

end NUMINAMATH_GPT_chess_team_girls_count_l1778_177841


namespace NUMINAMATH_GPT_parabola_line_unique_eq_l1778_177818

noncomputable def parabola_line_equation : Prop :=
  ∃ (A B : ℝ × ℝ),
    (A.2^2 = 4 * A.1) ∧ (B.2^2 = 4 * B.1) ∧
    ((A.1 + B.1) / 2 = 2) ∧ ((A.2 + B.2) / 2 = 2) ∧
    ∀ x y, (y - 2 = 1 * (x - 2)) → (x - y = 0)

theorem parabola_line_unique_eq : parabola_line_equation :=
  sorry

end NUMINAMATH_GPT_parabola_line_unique_eq_l1778_177818


namespace NUMINAMATH_GPT_problem1_problem2_l1778_177885

-- Problem 1

def a : ℚ := -1 / 2
def b : ℚ := -1

theorem problem1 :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) + a * b^2 = -3 / 4 :=
by
  sorry

-- Problem 2

def x : ℚ := 1 / 2
def y : ℚ := -2 / 3
axiom condition2 : abs (2 * x - 1) + (3 * y + 2)^2 = 0

theorem problem2 :
  5 * x^2 - (2 * x * y - 3 * (x * y / 3 + 2) + 5 * x^2) = 19 / 3 :=
by
  have h : abs (2 * x - 1) + (3 * y + 2)^2 = 0 := condition2
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1778_177885


namespace NUMINAMATH_GPT_present_age_of_son_l1778_177863

theorem present_age_of_son
  (S M : ℕ)
  (h1 : M = S + 24)
  (h2 : M + 2 = 2 * (S + 2)) :
  S = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_present_age_of_son_l1778_177863


namespace NUMINAMATH_GPT_shifted_roots_polynomial_l1778_177891

-- Define the original polynomial
def original_polynomial (x : ℝ) : ℝ :=
  x^3 - 5 * x + 7

-- Define the shifted polynomial
def shifted_polynomial (x : ℝ) : ℝ :=
  x^3 + 9 * x^2 + 22 * x + 19

-- Define the roots condition
def is_root (p : ℝ → ℝ) (r : ℝ) : Prop :=
  p r = 0

-- State the theorem
theorem shifted_roots_polynomial :
  ∀ a b c : ℝ,
    is_root original_polynomial a →
    is_root original_polynomial b →
    is_root original_polynomial c →
    is_root shifted_polynomial (a - 3) ∧
    is_root shifted_polynomial (b - 3) ∧
    is_root shifted_polynomial (c - 3) :=
by
  intros a b c ha hb hc
  sorry

end NUMINAMATH_GPT_shifted_roots_polynomial_l1778_177891


namespace NUMINAMATH_GPT_value_of_expression_l1778_177877

theorem value_of_expression :
  (3150 - 3030)^2 / 144 = 100 :=
by {
  -- This imported module allows us to use basic mathematical functions and properties
  sorry -- We use sorry to skip the actual proof
}

end NUMINAMATH_GPT_value_of_expression_l1778_177877


namespace NUMINAMATH_GPT_sequence_unique_l1778_177859

theorem sequence_unique (n : ℕ) (h1 : n > 1)
  (x : ℕ → ℕ)
  (hx1 : ∀ i j, 1 ≤ i ∧ i < j ∧ j < n → x i < x j)
  (hx2 : ∀ i, 1 ≤ i ∧ i < n → x i + x (n - i) = 2 * n)
  (hx3 : ∀ i j, 1 ≤ i ∧ i < n ∧ 1 ≤ j ∧ j < n ∧ x i + x j < 2 * n →
    ∃ k, 1 ≤ k ∧ k < n ∧ x i + x j = x k) :
  ∀ k, 1 ≤ k ∧ k < n → x k = 2 * k :=
by
  sorry

end NUMINAMATH_GPT_sequence_unique_l1778_177859


namespace NUMINAMATH_GPT_rainy_days_last_week_l1778_177896

theorem rainy_days_last_week (n : ℤ) (R NR : ℕ) (h1 : n * R + 3 * NR = 20)
  (h2 : 3 * NR = n * R + 10) (h3 : R + NR = 7) : R = 2 :=
sorry

end NUMINAMATH_GPT_rainy_days_last_week_l1778_177896


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1778_177866

theorem sufficient_but_not_necessary (a : ℝ) (h : a = 1/4) : 
  (∀ x : ℝ, x > 0 → x + a / x ≥ 1) ∧ ¬(∀ x : ℝ, x > 0 → x + a / x ≥ 1 ↔ a = 1/4) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1778_177866


namespace NUMINAMATH_GPT_tan_theta_neq_2sqrt2_l1778_177803

theorem tan_theta_neq_2sqrt2 (θ : ℝ) (h₀ : 0 < θ ∧ θ < Real.pi) (h₁ : Real.sin θ + Real.cos θ = (2 * Real.sqrt 2 - 1) / 3) : Real.tan θ = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_tan_theta_neq_2sqrt2_l1778_177803


namespace NUMINAMATH_GPT_find_acute_angles_right_triangle_l1778_177872

theorem find_acute_angles_right_triangle (α β : ℝ)
  (h₁ : α + β = π / 2)
  (h₂ : 0 < α ∧ α < π / 2)
  (h₃ : 0 < β ∧ β < π / 2)
  (h4 : Real.tan α + Real.tan β + Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan α ^ 3 + Real.tan β ^ 3 = 70) :
  (α = 75 * (π / 180) ∧ β = 15 * (π / 180)) 
  ∨ (α = 15 * (π / 180) ∧ β = 75 * (π / 180)) := 
sorry

end NUMINAMATH_GPT_find_acute_angles_right_triangle_l1778_177872


namespace NUMINAMATH_GPT_sasha_skated_distance_l1778_177814

theorem sasha_skated_distance (d total_distance v : ℝ)
  (h1 : total_distance = 3300)
  (h2 : v > 0)
  (h3 : d = 3 * v * (total_distance / (3 * v + 2 * v))) :
  d = 1100 :=
by
  sorry

end NUMINAMATH_GPT_sasha_skated_distance_l1778_177814


namespace NUMINAMATH_GPT_multiply_transformed_l1778_177889

theorem multiply_transformed : (268 * 74 = 19832) → (2.68 * 0.74 = 1.9832) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_multiply_transformed_l1778_177889


namespace NUMINAMATH_GPT_range_of_a_l1778_177894

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 - 2 * x + 3 ≤ a^2 - 2 * a - 1)) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1778_177894


namespace NUMINAMATH_GPT_amy_hours_per_week_l1778_177867

theorem amy_hours_per_week {h w summer_salary school_weeks school_salary} 
  (hours_per_week_summer : h = 45)
  (weeks_summer : w = 8)
  (summer_salary_h : summer_salary = 3600)
  (school_weeks_h : school_weeks = 24)
  (school_salary_h : school_salary = 3600) :
  ∃ hours_per_week_school, hours_per_week_school = 15 :=
by
  sorry

end NUMINAMATH_GPT_amy_hours_per_week_l1778_177867


namespace NUMINAMATH_GPT_pet_fee_is_120_l1778_177857

noncomputable def daily_rate : ℝ := 125.00
noncomputable def rental_days : ℕ := 14
noncomputable def service_fee_rate : ℝ := 0.20
noncomputable def security_deposit : ℝ := 1110.00
noncomputable def security_deposit_rate : ℝ := 0.50

theorem pet_fee_is_120 :
  let total_stay_cost := daily_rate * rental_days
  let service_fee := service_fee_rate * total_stay_cost
  let total_before_pet_fee := total_stay_cost + service_fee
  let entire_bill := security_deposit / security_deposit_rate
  let pet_fee := entire_bill - total_before_pet_fee
  pet_fee = 120 := by
  sorry

end NUMINAMATH_GPT_pet_fee_is_120_l1778_177857


namespace NUMINAMATH_GPT_third_number_is_60_l1778_177813

theorem third_number_is_60 (x : ℤ) :
  (20 + 40 + x) / 3 = (10 + 80 + 15) / 3 + 5 → x = 60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_third_number_is_60_l1778_177813


namespace NUMINAMATH_GPT_playground_perimeter_km_l1778_177893

def playground_length : ℕ := 360
def playground_width : ℕ := 480

def perimeter_in_meters (length width : ℕ) : ℕ := 2 * (length + width)

def perimeter_in_kilometers (perimeter_m : ℕ) : ℕ := perimeter_m / 1000

theorem playground_perimeter_km :
  perimeter_in_kilometers (perimeter_in_meters playground_length playground_width) = 168 :=
by
  sorry

end NUMINAMATH_GPT_playground_perimeter_km_l1778_177893


namespace NUMINAMATH_GPT_lines_parallel_l1778_177809

-- Definitions based on conditions
variable (line1 line2 : ℝ → ℝ → Prop) -- Assuming lines as relations for simplicity
variable (plane : ℝ → ℝ → ℝ → Prop) -- Assuming plane as a relation for simplicity

-- Condition: Both lines are perpendicular to the same plane
def perpendicular_to_plane (line : ℝ → ℝ → Prop) (plane : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ (x y z : ℝ), plane x y z → line x y

axiom line1_perpendicular : perpendicular_to_plane line1 plane
axiom line2_perpendicular : perpendicular_to_plane line2 plane

-- Theorem: Both lines are parallel
theorem lines_parallel : ∀ (line1 line2 : ℝ → ℝ → Prop) (plane : ℝ → ℝ → ℝ → Prop),
  (perpendicular_to_plane line1 plane) →
  (perpendicular_to_plane line2 plane) →
  (∀ x y : ℝ, line1 x y → line2 x y) := sorry

end NUMINAMATH_GPT_lines_parallel_l1778_177809


namespace NUMINAMATH_GPT_not_p_sufficient_not_necessary_for_not_q_l1778_177826

theorem not_p_sufficient_not_necessary_for_not_q (p q : Prop) (h1 : q → p) (h2 : ¬ (p → q)) : 
  (¬p → ¬ q) ∧ ¬ (¬ q → ¬ p) :=
sorry

end NUMINAMATH_GPT_not_p_sufficient_not_necessary_for_not_q_l1778_177826


namespace NUMINAMATH_GPT_garden_square_char_l1778_177882

theorem garden_square_char (s q p x : ℕ) (h1 : p = 28) (h2 : q = p + x) (h3 : q = s^2) (h4 : p = 4 * s) : x = 21 :=
by
  sorry

end NUMINAMATH_GPT_garden_square_char_l1778_177882


namespace NUMINAMATH_GPT_yang_hui_rect_eq_l1778_177895

theorem yang_hui_rect_eq (L W x : ℝ) 
  (h1 : L * W = 864)
  (h2 : L + W = 60)
  (h3 : L = W + x) : 
  (60 - x) / 2 * (60 + x) / 2 = 864 :=
by
  sorry

end NUMINAMATH_GPT_yang_hui_rect_eq_l1778_177895


namespace NUMINAMATH_GPT_find_dividend_l1778_177865

def dividend_problem (dividend divisor : ℕ) : Prop :=
  (15 * divisor + 5 = dividend) ∧ (dividend + divisor + 15 + 5 = 2169)

theorem find_dividend : ∃ dividend, ∃ divisor, dividend_problem dividend divisor ∧ dividend = 2015 :=
sorry

end NUMINAMATH_GPT_find_dividend_l1778_177865


namespace NUMINAMATH_GPT_amount_paid_by_customer_l1778_177879

theorem amount_paid_by_customer 
  (cost_price : ℝ)
  (markup_percentage : ℝ)
  (final_price : ℝ)
  (h1 : cost_price = 6681.818181818181)
  (h2 : markup_percentage = 10 / 100)
  (h3 : final_price = cost_price * (1 + markup_percentage)) :
  final_price = 7350 :=
by 
  sorry

end NUMINAMATH_GPT_amount_paid_by_customer_l1778_177879


namespace NUMINAMATH_GPT_probability_of_odd_number_l1778_177831

theorem probability_of_odd_number (total_outcomes : ℕ) (odd_outcomes : ℕ) (h1 : total_outcomes = 6) (h2 : odd_outcomes = 3) : (odd_outcomes / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry 

end NUMINAMATH_GPT_probability_of_odd_number_l1778_177831


namespace NUMINAMATH_GPT_polar_equations_and_ratios_l1778_177853

open Real

theorem polar_equations_and_ratios (α β : ℝ)
    (h_line : ∀ (r θ : ℝ), r * cos θ = 2 ↔ r = 2 / cos θ)
    (h_curve : ∀ (α : ℝ), ∃ (r θ : ℝ), r = 2 * sin θ ∧ θ = β ∧ 0 < β ∧ β < π / 2) :
    ( ∀ (r θ : ℝ), r * cos θ = 2 ↔ r = 2 / cos θ) ∧
    ( ∃ (r θ : ℝ), r = 2 * sin θ ∧ θ = β ∧ 0 < β ∧ β < π / 2 → 
    0 < r * sin 2 * θ / (r / cos θ) ∧ r * sin 2 * θ / (r / cos θ) ≤ 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_polar_equations_and_ratios_l1778_177853


namespace NUMINAMATH_GPT_geometric_sequence_second_term_l1778_177810

theorem geometric_sequence_second_term (a r : ℝ) (h1 : a * r ^ 2 = 5) (h2 : a * r ^ 4 = 45) :
  a * r = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_second_term_l1778_177810


namespace NUMINAMATH_GPT_total_cost_of_purchases_l1778_177845

def cost_cat_toy : ℝ := 10.22
def cost_cage : ℝ := 11.73

theorem total_cost_of_purchases : cost_cat_toy + cost_cage = 21.95 := by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_total_cost_of_purchases_l1778_177845
