import Mathlib

namespace canvas_bag_lower_carbon_solution_l521_521930

theorem canvas_bag_lower_carbon_solution :
  ∀ (canvas_bag_CO2_pounds : ℕ) (plastic_bag_CO2_ounces : ℕ) 
    (plastic_bags_per_trip : ℕ) (ounces_per_pound : ℕ),
    canvas_bag_CO2_pounds = 600 →
    plastic_bag_CO2_ounces = 4 →
    plastic_bags_per_trip = 8 →
    ounces_per_pound = 16 →
    let total_CO2_per_trip := plastic_bags_per_trip * plastic_bag_CO2_ounces / ounces_per_pound in
    (canvas_bag_CO2_pounds / total_CO2_per_trip) = 300 :=
by
  -- Assume the given conditions
  assume canvas_bag_CO2_pounds plastic_bag_CO2_ounces plastic_bags_per_trip ounces_per_pound,
  assume canvas_bag_CO2_pounds_eq plastic_bag_CO2_ounces_eq plastic_bags_per_trip_eq ounces_per_pound_eq,
  -- Introduce the total carbon dioxide per trip
  let total_CO2_per_trip := plastic_bags_per_trip * plastic_bag_CO2_ounces / ounces_per_pound,
  -- Verify that the number of trips is 300
  show ((canvas_bag_CO2_pounds / total_CO2_per_trip) = 300),
  sorry

end canvas_bag_lower_carbon_solution_l521_521930


namespace triangle_area_equivalence_l521_521120

theorem triangle_area_equivalence :
  ∃ (E F : Point2D),
    (E ∈ inside (square 0 0 2 2)) ∧
    (right_triangle (Point2D.mk 0 0) (Point2D.mk 2 0) E) ∧
    (angle (Point2D.mk 2 0) (Point2D.mk 0 0) E = 45) ∧
    (F = intersection (diagonal (Point2D.mk 0 2) (Point2D.mk 2 0)) (line (Point2D.mk 0 0) E)) ∧
    (area_triangle (Point2D.mk 0 0) (Point2D.mk 2 0) F = 1 / 2) :=
sorry

end triangle_area_equivalence_l521_521120


namespace connie_total_markers_l521_521695

def red_markers : ℕ := 5420
def blue_markers : ℕ := 3875
def green_markers : ℕ := 2910
def yellow_markers : ℕ := 6740

def total_markers : ℕ := red_markers + blue_markers + green_markers + yellow_markers

theorem connie_total_markers : total_markers = 18945 := by
  sorry

end connie_total_markers_l521_521695


namespace probability_adjacent_vertices_in_decagon_l521_521557

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l521_521557


namespace four_color_theorem_l521_521707

open Finset

variables {V : Type*} {E : V → V → Prop}

def valid_color_path (G : graph V E) (colors : E → ℕ) : Prop :=
  ∀ (v1 v2 v3 v4 : V), E v1 v2 → E v2 v3 → E v3 v4 → colors (v1, v2) ≠ colors (v3, v4)

def four_colorable (G : graph V E) (colors_by_edge : E → ℕ) : Prop :=
  ∃ color_by_vertex : V → ℕ, 
    (∀ v, color_by_vertex v < 4) ∧ 
    (∀ u v, E u v → color_by_vertex u ≠ color_by_vertex v)

theorem four_color_theorem 
  {V : Type*} {E : V → V → Prop} (G : graph V E) (colors : E → ℕ)
  (h_valid_path : valid_color_path G colors) :
  four_colorable G colors :=
sorry

end four_color_theorem_l521_521707


namespace function_range_l521_521484

noncomputable def range_of_function : set ℝ :=
  { y : ℝ | ∃ x : ℝ, x ≠ -3 ∧ y = (x^2 + 4 * x + 3)/(x + 3) }

theorem function_range :
  range_of_function = {y : ℝ | y ≠ -2} :=
by
  sorry

end function_range_l521_521484


namespace definite_integral_sin_pi_x_plus_x_minus_1_l521_521159

theorem definite_integral_sin_pi_x_plus_x_minus_1 :
  ∫ x in 0..2, (sin (π * x) + x - 1) = 0 :=
sorry

end definite_integral_sin_pi_x_plus_x_minus_1_l521_521159


namespace am_gm_inequality_product_l521_521447

theorem am_gm_inequality_product (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 < a i) (h_prod : (Finset.univ : Finset (Fin n)).prod a = 1) :
    (Finset.univ : Finset (Fin n)).prod (λ i, 1 + a i) ≥ 2^n :=
sorry

end am_gm_inequality_product_l521_521447


namespace product_pattern_l521_521938

theorem product_pattern (m n : ℝ) : 
  m * n = ( ( m + n ) / 2 ) ^ 2 - ( ( m - n ) / 2 ) ^ 2 := 
by 
  sorry

end product_pattern_l521_521938


namespace existence_of_C_l521_521941

theorem existence_of_C (O A B : ℝ × ℝ) (hx : A.2 = 0) (hy : B.2 = 0) (hz : O.2 = 0)
  (OA OB : ℝ) (hOA : OA = real.dist O A) (hOB : OB = real.dist O B) (hOAg0 : hOA > hOB) :
  ∃ C : ℝ × ℝ, C.1 = 0 ∧ ∃ α : ℝ, ∠ (B - C) (A - C) = 2 * α ∧ ∠ (F - C) = α :=
begin
  sorry,
end

end existence_of_C_l521_521941


namespace not_possible_to_create_3_piles_l521_521366

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end not_possible_to_create_3_piles_l521_521366


namespace minimum_voters_needed_l521_521271

-- conditions
def num_voters := 135
def num_districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority_precincts (n : ℕ) := (n + 1) / 2

-- definitions for quantities derived from conditions
def total_precincts := num_districts * precincts_per_district
def majority_districts := majority_precincts num_districts
def precincts_needed_for_district_win := majority_precincts precincts_per_district
def total_precincts_needed_for_win := majority_districts * precincts_needed_for_district_win
def votes_needed_per_precinct := majority_precincts voters_per_precinct

-- main statement
theorem minimum_voters_needed : (votes_needed_per_precinct * total_precincts_needed_for_win = 30) ∧ TallGiraffeWon :=
by sorry

end minimum_voters_needed_l521_521271


namespace expected_value_dodecahedral_die_is_6_5_l521_521066

noncomputable def expected_value_dodecahedral_die : ℝ :=
  let outcomes := finset.range 12 in
  1 / 12 * (finset.sum outcomes (λ i, (i + 1 : ℝ)))

theorem expected_value_dodecahedral_die_is_6_5 :
  expected_value_dodecahedral_die = 6.5 :=
by
  sorry

end expected_value_dodecahedral_die_is_6_5_l521_521066


namespace cardProblem_l521_521294

structure InitialState where
  jimmy_cards : ℕ
  bob_cards : ℕ
  sarah_cards : ℕ

structure UpdatedState where
  jimmy_cards_final : ℕ
  sarah_cards_final : ℕ
  sarahs_friends_cards : ℕ

def cardProblemSolved (init : InitialState) (final : UpdatedState) : Prop :=
  let bob_initial := init.bob_cards + 6
  let bob_to_sarah := bob_initial / 3
  let bob_final := bob_initial - bob_to_sarah
  let sarah_initial := init.sarah_cards + bob_to_sarah
  let sarah_friends := sarah_initial / 3
  let sarah_final := sarah_initial - 3 * sarah_friends
  let mary_cards := 2 * 6
  let jimmy_final := init.jimmy_cards - 6 - mary_cards
  let sarah_to_tim := 0 -- since Sarah can't give fractional cards
  (final.jimmy_cards_final = jimmy_final) ∧ 
  (final.sarah_cards_final = sarah_final - sarah_to_tim) ∧ 
  (final.sarahs_friends_cards = sarah_friends)

theorem cardProblem : 
  cardProblemSolved 
    { jimmy_cards := 68, bob_cards := 5, sarah_cards := 7 }
    { jimmy_cards_final := 50, sarah_cards_final := 1, sarahs_friends_cards := 3 } :=
by 
  sorry

end cardProblem_l521_521294


namespace Morse_code_distinct_symbols_l521_521869

-- Morse code sequences conditions
def MorseCodeSequence (n : ℕ) := n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4

-- Total number of distinct symbols calculation
def total_distinct_symbols : ℕ :=
  2 + 4 + 8 + 16

-- The theorem to prove
theorem Morse_code_distinct_symbols : total_distinct_symbols = 30 := by
  sorry

end Morse_code_distinct_symbols_l521_521869


namespace impossible_to_create_3_piles_l521_521338

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end impossible_to_create_3_piles_l521_521338


namespace triangle_side_lengths_l521_521837

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end triangle_side_lengths_l521_521837


namespace decagon_adjacent_probability_l521_521523

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521523


namespace compound_interest_l521_521865

theorem compound_interest (P R T : ℝ) (SI CI : ℝ)
  (hSI : SI = P * R * T / 100)
  (h_given_SI : SI = 50)
  (h_given_R : R = 5)
  (h_given_T : T = 2)
  (h_compound_interest : CI = P * ((1 + R / 100)^T - 1)) :
  CI = 51.25 :=
by
  -- Since we are only required to state the theorem, we add 'sorry' here.
  sorry

end compound_interest_l521_521865


namespace cost_per_box_l521_521091

-- Definitions of given conditions
def box_length := 20 -- inches
def box_width := 20 -- inches
def box_height := 12 -- inches
def total_volume := 1920000 -- cubic inches
def min_spending := 200 -- dollars

-- Definition and theorem
def volume_of_one_box := box_length * box_width * box_height

-- Define the number of boxes needed
def number_of_boxes := total_volume / volume_of_one_box

-- Theorem to be proven
theorem cost_per_box : (min_spending / number_of_boxes) = 0.50 := by
  sorry

end cost_per_box_l521_521091


namespace sum_of_first_six_terms_l521_521643

noncomputable def sequence : ℕ → ℝ
| 1     := 2
| n + 2 := 1.5 * (n+2)^2 - 1.5 * (n+2) + 2 - sum (λ k, sequence (k + 1)) (finset.range (n + 1)) + sequence 1

theorem sum_of_first_six_terms :
  sequence 1 + sequence 2 + sequence 3 + sequence 4 + sequence 5 + sequence 6 = 117 :=
sorry

end sum_of_first_six_terms_l521_521643


namespace vector_calculation_l521_521678

def vector1 : ℝ × ℝ := (4, -9)
def scalar : ℝ := 2
def vector2 : ℝ × ℝ := (-7, 15)
def result : ℝ × ℝ := (1, -3)

theorem vector_calculation :
  let v1 := vector1 in
  let v2 := vector2 in
  let s := scalar in
  let res := result in
  (s * v1.1, s * v1.2) + v2 = res :=
by
  -- proof goes here
  sorry

end vector_calculation_l521_521678


namespace probability_of_adjacent_vertices_in_decagon_l521_521589

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l521_521589


namespace point_A_in_fourth_quadrant_l521_521093

-- Definitions of the points
def P1 := (2, -Real.sqrt 3)
def P2 := (-2, 3)
def P3 := (-Real.sqrt 6, -6)
def P4 := (2, 3)

-- Definition of the fourth quadrant
def inFourthQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Statement that point P1 lies in the fourth quadrant
theorem point_A_in_fourth_quadrant : inFourthQuadrant P1 :=
by
  sorry

end point_A_in_fourth_quadrant_l521_521093


namespace test_point_selection_l521_521883

theorem test_point_selection (x_1 x_2 : ℝ)
    (interval_begin interval_end : ℝ) (h_interval : interval_begin = 2 ∧ interval_end = 4)
    (h_better_result : x_1 < x_2 ∨ x_1 > x_2)
    (h_test_points : (x_1 = interval_begin + 0.618 * (interval_end - interval_begin) ∧ 
                     x_2 = interval_begin + interval_end - x_1) ∨ 
                    (x_1 = interval_begin + interval_end - (interval_begin + 0.618 * (interval_end - interval_begin)) ∧ 
                     x_2 = interval_begin + 0.618 * (interval_end - interval_begin)))
  : ∃ x_3, x_3 = 3.528 ∨ x_3 = 2.472 := by
    sorry

end test_point_selection_l521_521883


namespace length_segment_EF_l521_521329

noncomputable def eq_point (A B : ℝ × ℝ) := (A.1 + B.1) / 2

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def line (A B : ℝ × ℝ) : ℝ → ℝ :=
  let m := (B.2 - A.2) / (B.1 - A.1) in fun x => A.2 + m * (x - A.1)

noncomputable def perpendicular_bisector (A B : ℝ × ℝ) : (ℝ → ℝ) :=
  let mid := midpoint A B in 
  if A.1 = B.1 then fun x => mid.2 -- vertical line
  else fun x => mid.2 + ((A.1 - B.1) / (B.2 - A.2)) * (x - mid.1)

noncomputable def intersection (f g : ℝ → ℝ) (x : ℝ) : ℝ × ℝ :=
  (x, f x)

theorem length_segment_EF :
  ∀ (A B C : ℝ × ℝ),
    A = (0, 0) → B = (1, 0) → C = (0, 2) →
    ∃ E F: ℝ × ℝ,
      E = intersection (line A C) (perpendicular_bisector A B) (1 / 2) ∧
      F = intersection (line A C) (perpendicular_bisector C A) 2 ∧
      (real.sqrt ((F.1 - E.1) ^ 2 + (F.2 - E.2) ^ 2)) = 3 * real.sqrt 5 / 4 :=
  sorry

end length_segment_EF_l521_521329


namespace ratio_w_y_l521_521000

theorem ratio_w_y (w x y z : ℚ) 
  (h1 : w / x = 5 / 2) 
  (h2 : y / z = 5 / 3) 
  (h3 : z / x = 1 / 6) : 
  w / y = 9 := 
by 
  sorry

end ratio_w_y_l521_521000


namespace probability_adjacent_vertices_decagon_l521_521604

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l521_521604


namespace probability_of_adjacent_vertices_in_decagon_l521_521590

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l521_521590


namespace quadratic_inequality_solution_set_l521_521719

theorem quadratic_inequality_solution_set (x : ℝ) : 
  (x^2 - 2 * x < 0) ↔ (0 < x ∧ x < 2) := 
sorry

end quadratic_inequality_solution_set_l521_521719


namespace total_drinks_correct_l521_521460

def drinks_in_section : ℕ := 3 + 2

def sections_per_vending_machine : ℕ := 6

def vending_machines : ℕ := 28

def drinks_per_vending_machine : ℕ :=
  drinks_in_section * sections_per_vending_machine

def total_drinks_in_arcade : ℕ :=
  drinks_per_vending_machine * vending_machines

theorem total_drinks_correct :
  total_drinks_in_arcade = 840 :=
  by
    unfold total_drinks_in_arcade 
    unfold drinks_per_vending_machine 
    unfold sections_per_vending_machine 
    unfold drinks_in_section 
    linarith

end total_drinks_correct_l521_521460


namespace possible_integer_side_lengths_l521_521803

theorem possible_integer_side_lengths : 
  {x : ℤ | x > 3 ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521803


namespace sheets_bought_l521_521616

variable (x y : ℕ)

-- Conditions based on the problem statement
def A_condition (x y : ℕ) : Prop := x + 40 = y
def B_condition (x y : ℕ) : Prop := 3 * x + 40 = y

-- Proven that if these conditions are met, then the number of sheets of stationery bought by A and B is 120
theorem sheets_bought (x y : ℕ) (hA : A_condition x y) (hB : B_condition x y) : y = 120 :=
by
  sorry

end sheets_bought_l521_521616


namespace Sn_equality_l521_521090

-- Define the sum of the geometric sequence
def geometric_sum (x : ℝ) (n : ℕ) (hx : x ≠ 1) :=
  x * (1 - x ^ n) / (1 - x)

-- Define Sn
def Sn (x : ℝ) (n : ℕ) :=
  ∑ k in range n, (k + 1) * x ^ k

-- The theorem to be proved
theorem Sn_equality (x : ℝ) (n : ℕ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) (hn : n > 0) :
  Sn x n = n * (n + 3) * 2^(n - 2) :=
sorry

end Sn_equality_l521_521090


namespace b_more_than_a_by_25_percent_l521_521098

constant B : ℝ
constant A : ℝ
axiom salary_relation : A = B - 0.2 * B

theorem b_more_than_a_by_25_percent : 
  ((B - A) / A) * 100 = 25 :=
  by
  sorry

end b_more_than_a_by_25_percent_l521_521098


namespace smallest_cube_sum_l521_521427

theorem smallest_cube_sum : 
  ∃ (faces : Fin 6 → ℕ), (∀ (i j : Fin 6), (adjacent i j → (abs (faces i - faces j) > 1))) ∧ (∑ (i : Fin 6), faces i = 18) := 
by
  sorry

-- Definitions for adjacency relations on a cube's faces can be adjusted accordingly.
def adjacent : Fin 6 → Fin 6 → Prop
| ⟨0, _⟩, ⟨1, _⟩ => true -- example equivalences, real function should define all adjacency pairs correctly
| ⟨1, _⟩, ⟨0, _⟩ => true
| _ , _ => false -- all other pairs would be set according to cube adjacency

end smallest_cube_sum_l521_521427


namespace translated_parabola_correct_l521_521997

-- Define translations
def translate_right (a : ℕ) (f : ℝ → ℝ) : ℝ → ℝ := λ x, f (x - a)
def translate_up (b : ℝ) (f : ℝ → ℝ) : ℝ → ℝ := λ x, (f x) + b

-- Original parabola
def original_parabola : ℝ → ℝ := λ x, x^2

-- Translated parabola
def translated_parabola : ℝ → ℝ := translate_up 1 (translate_right 2 original_parabola)

-- Prove the result
theorem translated_parabola_correct :
  ∀ x : ℝ, translated_parabola x = (x - 2)^2 + 1 :=
by
  intros x
  sorry

end translated_parabola_correct_l521_521997


namespace largest_possible_value_of_c_l521_521909

theorem largest_possible_value_of_c (c : ℚ) : (3 * c + 4) * (c - 2) = 9 * c → c ≤ 4 :=
by
  intro h
  have : (3 * c + 4) * (c - 2) = 3 * c^2 - 6 * c + 4 * c - 8 := 
    calc 
    (3 * c + 4) * (c - 2) = (3 * c) * (c - 2) + 4 * (c - 2) : by ring
                         ... = (3 * c) * c - (3 * c) * 2 + 4 * c - 4 * 2 : by ring
                         ... = 3 * c^2 - 6 * c + 4 * c - 8 : by ring
  rw this at h
  have h2 : 3 * c^2 - 11 * c - 8 = 0 := by nlinarith
  sorry

end largest_possible_value_of_c_l521_521909


namespace probability_adjacent_vertices_in_decagon_l521_521558

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l521_521558


namespace not_possible_to_create_3_similar_piles_l521_521340

-- Formalize the definition of similar piles
def similar (a b : ℝ) : Prop := a <= sqrt 2 * b ∧ b <= sqrt 2 * a

-- State the main theorem
theorem not_possible_to_create_3_similar_piles (x₀ : ℝ) (h₀ : 0 < x₀) : 
  ¬ ∃ x₁ x₂ x₃ : ℝ, 
    x₁ + x₂ + x₃ = x₀ ∧ 
    similar x₁ x₂ ∧ 
    similar x₂ x₃ ∧ 
    similar x₃ x₁ :=
sorry

end not_possible_to_create_3_similar_piles_l521_521340


namespace false_statement_dot_product_l521_521659

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

/-- Prove that if a dot b = 0, then a ≠ 0 and b ≠ 0 implies the statement is false -/
theorem false_statement_dot_product :
  (a ≠ 0 ∧ b ≠ 0 → a ⬝ b = 0 → false) := 
sorry

end false_statement_dot_product_l521_521659


namespace sum_of_all_x_l521_521915

noncomputable def f (x : ℝ) : ℝ := 12 * x - 4

theorem sum_of_all_x (x : ℝ) :
  (∀ x, f (x) = 12 * x - 4 → f⁻¹(x) = f (3 * x⁻¹)) →
  (∃ s, s = -52) :=
sorry

end sum_of_all_x_l521_521915


namespace decagon_adjacent_probability_l521_521541

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l521_521541


namespace expected_value_fair_dodecahedral_die_l521_521075

theorem expected_value_fair_dodecahedral_die : 
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  let probability := (1 : ℝ) / 12 in
  (probability * faces.sum) = 6.5 :=
by
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let probability := (1 : ℝ) / 12
  have h : faces.sum = 78 := rfl
  rw [h]
  sorry

end expected_value_fair_dodecahedral_die_l521_521075


namespace expected_value_dodecahedral_die_is_6_5_l521_521065

noncomputable def expected_value_dodecahedral_die : ℝ :=
  let outcomes := finset.range 12 in
  1 / 12 * (finset.sum outcomes (λ i, (i + 1 : ℝ)))

theorem expected_value_dodecahedral_die_is_6_5 :
  expected_value_dodecahedral_die = 6.5 :=
by
  sorry

end expected_value_dodecahedral_die_is_6_5_l521_521065


namespace smallest_N_satisfying_frequencies_l521_521154

def percentageA := 1 / 5
def percentageB := 3 / 8
def percentageC := 1 / 4
def percentageD := 1 / 8
def percentageE := 1 / 20

def Divisible (n : ℕ) (d : ℕ) : Prop := ∃ (k : ℕ), n = k * d

theorem smallest_N_satisfying_frequencies :
  ∃ N : ℕ, 
    Divisible N 5 ∧ 
    Divisible N 8 ∧ 
    Divisible N 4 ∧ 
    Divisible N 20 ∧ 
    N = 40 := sorry

end smallest_N_satisfying_frequencies_l521_521154


namespace locus_is_circle_l521_521942

-- Definition of the complex number and its conjugate
def locus_condition (z : ℂ) : Prop :=
  z * conj z + z + conj z = 3

-- Statement of the proof problem
theorem locus_is_circle : ∀ z : ℂ, locus_condition z → (∃ (c : ℂ) (r : ℝ), |z - c| = r) :=
  sorry

end locus_is_circle_l521_521942


namespace minimize_construction_cost_cost_when_x_is_12_l521_521629

-- Define the problem variables and conditions
variables (a b x : ℝ)
-- Let the old wall be used as one side of the rectangular workshop
-- And the area be 126 square meters
def area := 126

-- Define the cost functions for different cases
-- Plan 1: 0 < x < 14
def cost_plan_1 (x : ℝ) : ℝ :=
  let repair_cost := x * (a / 4)
  let demolish_cost := (14 - x) * (a / 2)
  let build_cost := (2 * x + 252 / x - 14) * a
  repair_cost + demolish_cost + build_cost + b
  
-- Plan 2: x ≥ 14
def cost_plan_2 (x : ℝ) : ℝ :=
  let repair_cost := 14 * (a / 4)
  let build_cost := (2 * x + 252 / x - 14) * a
  repair_cost + build_cost + b

-- Define the minimum cost for the given conditions and prove the optimal x
theorem minimize_construction_cost : 
  0 < x ∧ x < 14 → cost_plan_1 x = 7 * a * (2 * real.sqrt ((x / 4) * (36 / x)) - 1) + b → x = 12 :=
by sorry

theorem cost_when_x_is_12 :
  cost_plan_1 12 = 35 * a + b :=
by sorry

end minimize_construction_cost_cost_when_x_is_12_l521_521629


namespace shaded_area_fraction_of_square_l521_521411

theorem shaded_area_fraction_of_square 
  (Points_lie_on_square : ∀ (T U V W X Y : Point) (P Q R S : Square), 
    points_on_square P Q R S T U V W X Y) 
  (Equal_segments : ∀(T U V W X Y : Point) (P Q R S : Square),
    segment_eq P T ∧ segment_eq T U ∧ segment_eq U Q ∧ segment_eq Q V ∧
    segment_eq V W ∧ segment_eq W R ∧ segment_eq X S ∧ segment_eq S Y) :
  fraction_shaded PQRS = 5 / 18 := 
sorry

end shaded_area_fraction_of_square_l521_521411


namespace canvas_bag_lower_carbon_solution_l521_521926

theorem canvas_bag_lower_carbon_solution :
  let canvas_release_oz := 9600
  let plastic_per_trip_oz := 32
  canvas_release_oz / plastic_per_trip_oz = 300 :=
by
  sorry

end canvas_bag_lower_carbon_solution_l521_521926


namespace mark_total_minutes_played_l521_521391

open Nat -- to use natural number arithmetic

theorem mark_total_minutes_played :
  ∀ (songs_per_gig : Nat) (two_songs_duration_each : Nat) (last_song_multiplier : Nat) (days_per_week : Nat) (weeks : Nat) (days_per_gig: Nat),
  songs_per_gig = 3 →
  two_songs_duration_each = 5 →
  last_song_multiplier = 2 →
  days_per_week = 7 →
  weeks = 2 →
  days_per_gig = 2 →
  (weeks * days_per_week / days_per_gig) * (2 * two_songs_duration_each + (two_songs_duration_each * last_song_multiplier)) = 140 :=
by
  intros songs_per_gig two_songs_duration_each last_song_multiplier days_per_week weeks days_per_gig
  intro h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  have gigs := (weeks * days_per_week) / days_per_gig
  have duration_per_gig := 2 * two_songs_duration_each + (two_songs_duration_each * last_song_multiplier)
  show (gigs * duration_per_gig) = 140
  sorry

end mark_total_minutes_played_l521_521391


namespace expected_value_of_dodecahedral_die_is_6_5_l521_521043

noncomputable def expected_value_of_dodecahedral_die : ℝ := 
  ∑ i in Finset.range (12 + 1), (i : ℝ) / 12

theorem expected_value_of_dodecahedral_die_is_6_5 :
  expected_value_of_dodecahedral_die = 6.5 := sorry

end expected_value_of_dodecahedral_die_is_6_5_l521_521043


namespace weight_of_one_watermelon_l521_521132

-- We define the key conditions
variables (W : ℝ) -- The weight of one watermelon
variables (price_per_pound : ℝ) (total_revenue : ℝ) (num_watermelons : ℕ)

-- Given conditions
def condition1 : price_per_pound = 2 := rfl
def condition2 : total_revenue = 828 := rfl
def condition3 : num_watermelons = 18 := rfl
def equation (W : ℝ) : price_per_pound * (num_watermelons * W) = total_revenue := sorry

-- Prove the weight of one watermelon
theorem weight_of_one_watermelon : W = 23 :=
by {
  sorry,
}

end weight_of_one_watermelon_l521_521132


namespace probability_adjacent_vertices_in_decagon_l521_521560

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l521_521560


namespace triangle_side_lengths_l521_521838

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end triangle_side_lengths_l521_521838


namespace evaluate_expression_right_to_left_l521_521262

theorem evaluate_expression_right_to_left : 
  (2 * (3 + (4 / (6 ^ 2)))) = (56 / 9) :=
by
  -- Starting from the right
  have h1 : (6 ^ 2) = 36 := by norm_num,
  have h2 : 4 / 36 = 1 / 9 := by norm_num,
  have h3 : 3 + (1 / 9) = 28 / 9 := by norm_num,
  have h4 : 2 * (28 / 9) = 56 / 9 := by norm_num,
  rw [h1, h2, h3, h4],
  norm_num,
  -- Thus, the theorem is true
  sorry

end evaluate_expression_right_to_left_l521_521262


namespace goose_eggs_count_l521_521137

theorem goose_eggs_count (E : ℕ)
    (hatch_fraction : ℚ := 1/3)
    (first_month_survival : ℚ := 4/5)
    (first_year_survival : ℚ := 2/5)
    (no_migration : ℚ := 3/4)
    (predator_survival : ℚ := 2/3)
    (final_survivors : ℕ := 140) :
    (predator_survival * no_migration * first_year_survival * first_month_survival * hatch_fraction * E : ℚ) = final_survivors → E = 1050 := by
  sorry

end goose_eggs_count_l521_521137


namespace probability_adjacent_vertices_decagon_l521_521600

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l521_521600


namespace find_b_value_l521_521488

theorem find_b_value :
  (∀ x : ℝ, (x < 0 ∨ x > 4) → -x^2 + 4*x - 4 < 0) ↔ b = 4 := by
sorry

end find_b_value_l521_521488


namespace parabola_translation_l521_521995

theorem parabola_translation (x : ℝ) : 
  let y := x^2 in
  let y_translated := (x - 2)^2 + 1 in 
  ∀ x, y_translated = (x - 2)^2 + 1 :=
by
  sorry

end parabola_translation_l521_521995


namespace find_a_l521_521848

noncomputable def f (x a : ℝ) : ℝ := (2 * x + a) ^ 2

theorem find_a (a : ℝ) (h1 : f 2 a = 20) : a = 1 :=
sorry

end find_a_l521_521848


namespace expected_value_fair_dodecahedral_die_l521_521056

theorem expected_value_fair_dodecahedral_die :
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  expected_value = 6.5 :=
by
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  have h1 : list.sum outcomes = 78 := by sorry
  have h2 : (12: ℝ) * (1 / 12: ℝ) = 1 := by sorry
  show expected_value = 6.5
  { sorry }

end expected_value_fair_dodecahedral_die_l521_521056


namespace cos_angle_MPN_l521_521879

theorem cos_angle_MPN (P Q R S M N : ℝ × ℝ)
  (hP : P = (0, 0))
  (hQ : Q = (0, 2))
  (hR : R = (2, 2))
  (hS : S = (2, 0))
  (hM : M = (2, 1))
  (hN : N = (1, 2)) :
  Real.cos (angle M P N) = 4 / 5 :=
sorry

end cos_angle_MPN_l521_521879


namespace wooden_easel_cost_l521_521656

noncomputable def cost_paintbrush : ℝ := 1.5
noncomputable def cost_set_of_paints : ℝ := 4.35
noncomputable def amount_already_have : ℝ := 6.5
noncomputable def additional_amount_needed : ℝ := 12
noncomputable def total_cost_items : ℝ := cost_paintbrush + cost_set_of_paints
noncomputable def total_amount_needed : ℝ := amount_already_have + additional_amount_needed

theorem wooden_easel_cost :
  total_amount_needed - total_cost_items = 12.65 :=
by
  sorry

end wooden_easel_cost_l521_521656


namespace min_voters_tall_giraffe_win_l521_521281

-- Definitions from the problem statement as conditions
def precinct_voters := 3
def precincts_per_district := 9
def districts := 5
def majority_precincts(p : ℕ) := p / 2 + 1  -- Minimum precincts won in a district 
def majority_districts(d : ℕ) := d / 2 + 1  -- Minimum districts won in the final

-- Condition: majority precincts to win a district
def precinct_votes_to_win := majority_precincts precinct_voters

-- Condition: majority districts to win the final
def district_wins_to_win_final := majority_districts districts

-- Minimum precincts the Tall giraffe needs to win overall
def total_precincts_to_win := district_wins_to_win_final * majority_precincts precincts_per_district

-- Proof that the minimum number of voters who could have voted for the Tall giraffe is 30
theorem min_voters_tall_giraffe_win :
  precinct_votes_to_win * total_precincts_to_win = 30 :=
sorry

end min_voters_tall_giraffe_win_l521_521281


namespace selina_money_left_l521_521418

def pants_price : ℕ := 5
def shorts_price : ℕ := 3
def shirts_price : ℕ := 4
def pants_sold : ℕ := 3
def shorts_sold : ℕ := 5
def shirts_sold : ℕ := 5
def shirts_bought : ℕ := 2
def shirts_buy_price : ℕ := 10

theorem selina_money_left :
  let money_earned := (pants_sold * pants_price) + (shorts_sold * shorts_price) + (shirts_sold * shirts_price) in
  let money_spent := shirts_bought * shirts_buy_price in
  money_earned - money_spent = 30 :=
by
  sorry

end selina_money_left_l521_521418


namespace sum_of_reciprocals_squares_l521_521972

theorem sum_of_reciprocals_squares (a b : ℕ) (h : a * b = 17) :
  (1 : ℚ) / (a * a) + 1 / (b * b) = 290 / 289 :=
sorry

end sum_of_reciprocals_squares_l521_521972


namespace probability_adjacent_vertices_of_decagon_l521_521507

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l521_521507


namespace number_of_pencils_l521_521700

theorem number_of_pencils (E P : ℕ) (h1 : E + P = 8) (h2 : 300 * E + 500 * P = 3000) (hE : E ≥ 1) (hP : P ≥ 1) : P = 3 :=
by
  sorry

end number_of_pencils_l521_521700


namespace probability_adjacent_vertices_in_decagon_l521_521566

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l521_521566


namespace possible_integer_side_lengths_l521_521824

theorem possible_integer_side_lengths (a b : ℕ) (h1 : a = 8) (h2 : b = 5) :
  {x : ℕ | 3 < x ∧ x < 13}.finite ∧ {x : ℕ | 3 < x ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521824


namespace length_of_second_platform_l521_521653

/-- 
Let L be the length of the second platform.
A train crosses a platform of 100 m in 15 sec.
The same train crosses another platform in 20 sec.
The length of the train is 350 m.
Prove that the length of the second platform is 250 meters.
-/
theorem length_of_second_platform (L : ℕ) (train_length : ℕ) (platform1_length : ℕ) (time1 : ℕ) (time2 : ℕ):
  train_length = 350 → platform1_length = 100 → time1 = 15 → time2 = 20 → L = 250 :=
by
  sorry

end length_of_second_platform_l521_521653


namespace solution_l521_521746

noncomputable def problem (A B C : ℝ) (m n : ℝ × ℝ) : Prop :=
  let ⟨m1, m2⟩ := m in
  let ⟨n1, n2⟩ := n in
  A + B + C = Real.pi ∧
  m = (-1, Real.sqrt 3) ∧
  n = (Real.cos A, Real.sin A) ∧
  (m1 * n1 + m2 * n2) = 1 ∧
  (1 + Real.sin (2 * B)) / (Real.cos B ^ 2 - Real.sin B ^ 2) = -3 ∧
  Real.cos C = (2 * Real.sqrt 15 - Real.sqrt 5) / 10

theorem solution (A B C : ℝ) (m n : ℝ × ℝ) :
  problem A B C m n :=
sorry

end solution_l521_521746


namespace impossible_to_create_3_piles_l521_521377

def similar_sizes (x y : ℝ) : Prop := x ≤ sqrt 2 * y

theorem impossible_to_create_3_piles (x y : ℝ) (h₁ : similar_sizes x y) :
  ¬ ∃ (x₁ x₂ x₃ : ℝ), x₁ + x₂ + x₃ = x + y ∧ similar_sizes x₁ x₂ ∧ similar_sizes x₂ x₃ ∧ similar_sizes x₁ x₃ :=
by
  sorry

end impossible_to_create_3_piles_l521_521377


namespace decagon_adjacent_probability_l521_521544

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l521_521544


namespace distance_between_circles_l521_521318

theorem distance_between_circles (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB AC BC : Real) (hAB : AB = 17) (hAC : AC = 15) (hBC : BC = 10) :
  let s := (AB + AC + BC) / 2,
      K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)),
      r := K / s,
      AS := s - AB,
      AT := s,
      AI := Real.sqrt (AS^2 + r^2),
      AE := 21 * AI in
  AE - AI = 20 * Real.sqrt (16 + 5544 / 441) := 
  sorry

end distance_between_circles_l521_521318


namespace probability_adjacent_vertices_decagon_l521_521605

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l521_521605


namespace correct_propositions_l521_521760

def prop1 : Prop := (3.14 : ℚ) ∈ ℚ

def prop2 : Prop := ({0} : Set ℕ) = ∅

def prop3 : Prop := (a : Set ℕ) ∈ {a, b}

def prop4 : Prop := (1, 2) ∈ {y : ℝ × ℝ | y.1 + 1 = y.2}

def prop5 : Prop := ({x : ℝ | x^2 + 1 = 0} ⊆ {1})

theorem correct_propositions : prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4 ∧ prop5 :=
by sorry

end correct_propositions_l521_521760


namespace twenty_seven_cubes_volume_l521_521459

def volume_surface_relation (x V S : ℝ) : Prop :=
  V = x^3 ∧ S = 6 * x^2 ∧ V + S = (4 / 3) * (12 * x)

theorem twenty_seven_cubes_volume (x : ℝ) (hx : volume_surface_relation x (x^3) (6 * x^2)) : 
  27 * (x^3) = 216 :=
by
  sorry

end twenty_seven_cubes_volume_l521_521459


namespace sum_is_odd_prob_l521_521293

-- A type representing the spinner results, which can be either 1, 2, 3 or 4.
inductive SpinnerResult
| one : SpinnerResult
| two : SpinnerResult
| three : SpinnerResult
| four : SpinnerResult

open SpinnerResult

-- Function to determine if a spinner result is odd.
def isOdd (r : SpinnerResult) : Bool :=
  match r with
  | one => true
  | three => true
  | two => false
  | four => false

-- Defining the spinners P, Q, R, and S.
noncomputable def P : SpinnerResult := SpinnerResult.one -- example, could vary
noncomputable def Q : SpinnerResult := SpinnerResult.two -- example, could vary
noncomputable def R : SpinnerResult := SpinnerResult.three -- example, could vary
noncomputable def S : SpinnerResult := SpinnerResult.four -- example, could vary

-- Probability calculation function
def probabilityOddSum : ℚ :=
  let probOdd := 1 / 2
  let probEven := 1 / 2
  let scenario1 := 4 * probOdd * probEven^3
  let scenario2 := 4 * probOdd^3 * probEven
  scenario1 + scenario2

-- The theorem to be stated
theorem sum_is_odd_prob :
  probabilityOddSum = 1 / 2 := by
  sorry

end sum_is_odd_prob_l521_521293


namespace solve_trigonometric_eq_l521_521846

theorem solve_trigonometric_eq : 
  ∃ (n : ℕ), n = 18 ∧ 
  ∃ (x : ℝ), 
    -10 < x ∧ x < 50 ∧ 
    cos(x)^2 + 3 * sin(x)^2 = 1.5 := 
sorry

end solve_trigonometric_eq_l521_521846


namespace probability_of_adjacent_vertices_in_decagon_l521_521588

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l521_521588


namespace find_value_l521_521223

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2 else 3^x
  
theorem find_value :
  f (f (1 / 4)) = 1 / 9 := by
  sorry

end find_value_l521_521223


namespace mixed_number_division_l521_521475

theorem mixed_number_division : 
  let a := 9 / 4
  let b := 3 / 5
  a / b = 15 / 4 :=
by
  sorry

end mixed_number_division_l521_521475


namespace logistics_company_freight_l521_521114

theorem logistics_company_freight :
  ∃ (x y : ℕ), 
    50 * x + 30 * y = 9500 ∧
    70 * x + 40 * y = 13000 ∧
    x = 100 ∧
    y = 140 :=
by
  -- The proof is skipped here
  sorry

end logistics_company_freight_l521_521114


namespace triangle_side_lengths_count_l521_521793

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end triangle_side_lengths_count_l521_521793


namespace jason_has_21_toys_l521_521292

-- Definitions based on the conditions
def rachel_toys : ℕ := 1
def john_toys : ℕ := rachel_toys + 6
def jason_toys : ℕ := 3 * john_toys

-- The theorem to prove
theorem jason_has_21_toys : jason_toys = 21 := by
  -- Proof not needed, hence sorry
  sorry

end jason_has_21_toys_l521_521292


namespace decagon_adjacent_probability_l521_521585

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521585


namespace sum_of_reciprocals_eq_six_l521_521984

theorem sum_of_reciprocals_eq_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  (1 / x + 1 / y) = 6 :=
by
  sorry

end sum_of_reciprocals_eq_six_l521_521984


namespace decagon_adjacent_vertex_probability_l521_521555

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l521_521555


namespace xavier_goals_per_2_hours_l521_521491

theorem xavier_goals_per_2_hours :
  (let goals_per_15_minutes := 2 in
   let minutes_per_hour := 60 in
   let duration_hours := 2 in
   let total_minutes := duration_hours * minutes_per_hour in
   let segment_length := 15 in
   let number_of_segments := total_minutes / segment_length in
   let goals_in_2_hours := number_of_segments * goals_per_15_minutes in
   goals_in_2_hours = 16) :=
begin
  rfl,
end

end xavier_goals_per_2_hours_l521_521491


namespace probability_adjacent_vertices_in_decagon_l521_521569

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l521_521569


namespace order_of_a_b_c_l521_521206

noncomputable def f (x : ℝ) : ℝ := sorry

def a := f (4.1^0.2) / 4.1^0.2
def b := f (0.4^2.1) / 0.4^2.1
def c := f (Real.log 4.1 / Real.log 0.2) / (Real.log 4.1 / Real.log 0.2)

axiom odd_f : ∀ x : ℝ, f (-x) = -f(x)
axiom condition : ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 0 < x₁ ∧ 0 < x₂ → (x₂ * f(x₁) - x₁ * f(x₂)) / (x₁ - x₂) < 0

theorem order_of_a_b_c : a < c ∧ c < b :=
by
  sorry

end order_of_a_b_c_l521_521206


namespace possible_integer_side_lengths_l521_521801

theorem possible_integer_side_lengths : 
  {x : ℤ | x > 3 ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521801


namespace artwork_arrangement_l521_521877

noncomputable def arrangement_count (calligraphy1 calligraphy2 painting1 painting2 design : Type) : ℕ :=
  let adj_calligraphies := 2 * factorial 4
  let adj_paintings := 2 * 2 * factorial 3
  adj_calligraphies - adj_paintings

theorem artwork_arrangement (calligraphy1 calligraphy2 painting1 painting2 design : Type) :
  arrangement_count calligraphy1 calligraphy2 painting1 painting2 design = 24 :=
sorry

end artwork_arrangement_l521_521877


namespace coefficient_x3_l521_521269

-- Define the given expression
def expr := (x^2 - 2 * x) * (1 + x)^6

-- State the property to be proved
theorem coefficient_x3 : (coeff (expr) 3) = -24 := by
  sorry

end coefficient_x3_l521_521269


namespace probability_dice_within_circle_l521_521958

open Classical

noncomputable def probability_within_circle (radius : ℝ) (pt : ℤ × ℤ) : Prop :=
  (pt.1 : ℝ)^2 + (pt.2 : ℝ)^2 < radius^2

theorem probability_dice_within_circle : 
  let total_outcomes := finset.product (finset.range 6).succ (finset.range 6).succ,
      favorable_outcomes := finset.filter (λ pt, probability_within_circle 4 pt) total_outcomes in
  (favorable_outcomes.card : ℚ) / total_outcomes.card = 2 / 9 := sorry

end probability_dice_within_circle_l521_521958


namespace net_goals_times_middle_school_l521_521961

theorem net_goals_times_middle_school :
  let first_match := 5 - 3,
      second_match := 2 - 6,
      third_match := 2 - 2
  in first_match + second_match + third_match = -2 :=
by
  -- Conditions and calculation steps are skipped
  sorry

end net_goals_times_middle_school_l521_521961


namespace general_formula_seq_sum_of_squares_inequality_l521_521194

/-- Define the sequence a₁ = 2 and recurrence relation -/
def seq (n : ℕ) : ℕ := if n = 1 then 2 else 2 * n

theorem general_formula_seq (n : ℕ) (hn : n ∈ ℕ) : seq n = 2 * n := 
sorry

theorem sum_of_squares_inequality (n : ℕ) (hn : n ∈ ℕ) : 
  (finset.sum (finset.range n.succ) (λ k, 1 / (seq (k + 1)) ^ 2)) < 7 / 16 :=
sorry

end general_formula_seq_sum_of_squares_inequality_l521_521194


namespace probability_adjacent_vertices_in_decagon_l521_521563

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l521_521563


namespace range_of_a_l521_521755

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x ≥ 4 ∧ y ≥ 4 ∧ x ≤ y → (x^2 + 2*(a-1)*x + 2) ≤ (y^2 + 2*(a-1)*y + 2)) ↔ a ∈ Set.Ici (-3) :=
by
  sorry

end range_of_a_l521_521755


namespace num_possible_triangle_sides_l521_521811

theorem num_possible_triangle_sides (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (a_eq_8 : a = 8) (b_eq_5 : b = 5) : 
  let x := {n : ℕ | 3 < n ∧ n < 13} in
  set.card x = 9 :=
by
  sorry

end num_possible_triangle_sides_l521_521811


namespace rectangle_constant_k_l521_521973

theorem rectangle_constant_k (d : ℝ) (x : ℝ) (h_ratio : 4 * x = (4 / 3) * (3 * x)) (h_diagonal : d^2 = (4 * x)^2 + (3 * x)^2) : 
  ∃ k : ℝ, k = 12 / 25 ∧ (4 * x) * (3 * x) = k * d^2 := 
sorry

end rectangle_constant_k_l521_521973


namespace expected_value_fair_dodecahedral_die_l521_521080

theorem expected_value_fair_dodecahedral_die : 
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  let probability := (1 : ℝ) / 12 in
  (probability * faces.sum) = 6.5 :=
by
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let probability := (1 : ℝ) / 12
  have h : faces.sum = 78 := rfl
  rw [h]
  sorry

end expected_value_fair_dodecahedral_die_l521_521080


namespace probability_adjacent_vertices_decagon_l521_521601

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l521_521601


namespace cost_of_fencing_per_meter_l521_521980

theorem cost_of_fencing_per_meter
  (RatioLengthWidth : ∀ (x : ℝ), (length width : ℝ), length = 3 * x ∧ width = 2 * x)
  (Area : ℝ := 7350)
  (TotalCost : ℝ := 175)
  : (costPerMeterInPaise : ℝ) :=
by
  have h1 : x := sorry,
  have length := 3 * x,
  have width := 2 * x,
  have perimeter := 2 * (length + width),
  have costPerMeter := TotalCost / perimeter,
  have costPerMeterInPaise := costPerMeter * 100,
  exact costPerMeterInPaise

#eval cost_of_fencing_per_meter

end cost_of_fencing_per_meter_l521_521980


namespace largest_divisor_of_five_consecutive_odds_l521_521088

theorem largest_divisor_of_five_consecutive_odds (n : ℕ) (hn : n % 2 = 0) :
    ∃ d, d = 15 ∧ ∀ m, (m = (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11)) → d ∣ m :=
sorry

end largest_divisor_of_five_consecutive_odds_l521_521088


namespace cows_on_farm_l521_521873

noncomputable def num_cows : ℕ :=
c  -- the number of cows

theorem cows_on_farm 
{c h s : ℕ} 
(h1 : 4 * c + 2 * h + 4 * s = 100)
(h2 : 4 * c + 2 * h + 4 * s = 3 * (c + h + s) + 20) :
num_cows = 6 :=
by sorry

end cows_on_farm_l521_521873


namespace problem_conditions_imply_value_of_y_l521_521467

theorem problem_conditions_imply_value_of_y :
  ∀ (A B C D E : Type) (y : ℚ),
  (AD BE DC AC BC : ℚ) 
  (h1 : AD = 7)
  (h2 : DC = 4)
  (h3 : BE = 3)
  (h4 : AC = AD + DC)
  (h5 : AC = 11)
  (h6 : BC = BE + y)
  (h_similarity : AD / BE = AC / BC) :
  y = 12 / 7 := by
  sorry

end problem_conditions_imply_value_of_y_l521_521467


namespace cosine_of_unit_vectors_l521_521316

variables {a b : ℝ^3}
hypothesis (unit_a : ∥a∥ = 1)
hypothesis (unit_b : ∥b∥ = 1)
hypothesis (proj_condition : (a + b) ⬝ b = 2 / 3)

theorem cosine_of_unit_vectors :
  real.cos (real.angle a b) = -1 / 3 :=
sorry

end cosine_of_unit_vectors_l521_521316


namespace jane_buys_4_bagels_l521_521892

theorem jane_buys_4_bagels (b m : ℕ) (h1 : b + m = 7) (h2 : (80 * b + 60 * m) % 100 = 0) : b = 4 := 
by sorry

end jane_buys_4_bagels_l521_521892


namespace train_length_l521_521654

def length_of_train (speed_train speed_man : ℝ) (time : ℝ) :=
  (speed_train + speed_man) * (1000 / 3600) * time

theorem train_length :
  length_of_train 85 5 6 = 150 :=
by
  sorry

end train_length_l521_521654


namespace second_digging_breadth_l521_521623

theorem second_digging_breadth :
  ∀ (A B depth1 length1 breadth1 depth2 length2 : ℕ),
  (A / B) = 1 → -- Assuming equal number of days and people
  depth1 = 100 → length1 = 25 → breadth1 = 30 → 
  depth2 = 75 → length2 = 20 → 
  (A = depth1 * length1 * breadth1) → 
  (B = depth2 * length2 * x) →
  x = 50 :=
by sorry

end second_digging_breadth_l521_521623


namespace ceil_x_cubed_values_l521_521855

theorem ceil_x_cubed_values (x : ℝ) (h : ⌊x⌋ = -5) : 
  (set.Icc (-125 : ℤ) (-64 : ℤ)).card = 62 := 
by sorry

end ceil_x_cubed_values_l521_521855


namespace gavrila_ascent_time_l521_521471
noncomputable def gavrila_time (V U t : ℝ) : ℝ := t

theorem gavrila_ascent_time (V U : ℝ) :
  (1 = V * 60) →
  (1 = (V + U) * 24) →
  (t = 40 → 1 = U * t) :=
by
  intros h1 h2 h3
  -- Using the given equations:
  -- 1 = V * 60
  -- 1 = (V + U) * 24
  -- Solve for V and substitute to find U
  have hV : V = 1 / 60 := by sorry
  have hU : U = 1 / 40 := by sorry
  rw [h3, hU]
  exact rfl

end gavrila_ascent_time_l521_521471


namespace tanya_work_time_l521_521951

theorem tanya_work_time (
    sakshi_work_time : ℝ := 20,
    tanya_efficiency : ℝ := 1.25
) : 
    let sakshi_rate : ℝ := 1 / sakshi_work_time in
    let tanya_rate : ℝ := tanya_efficiency * sakshi_rate in
    let tanya_work_time := 1 / tanya_rate in
    tanya_work_time = 16 :=
by
    let sakshi_rate : ℝ := 1 / sakshi_work_time
    let tanya_rate : ℝ := tanya_efficiency * sakshi_rate
    let tanya_time : ℝ := 1 / tanya_rate
    show tanya_time = 16
    sorry

end tanya_work_time_l521_521951


namespace largest_int_less_than_log_sum_l521_521478

theorem largest_int_less_than_log_sum :
  (⌊∑ k in finset.range 3010, real.log 3 (k + 2) (k + 1)⌋ : ℤ) = 6 :=
begin
  sorry
end

end largest_int_less_than_log_sum_l521_521478


namespace number_of_sides_possibilities_l521_521818

theorem number_of_sides_possibilities (a b : ℕ) (h₁ : a = 8) (h₂ : b = 5) : 
  { x : ℕ // 3 < x ∧ x < 13 }.to_list.length = 9 :=
by
  sorry

end number_of_sides_possibilities_l521_521818


namespace smallest_k_for_sequence_l521_521196

theorem smallest_k_for_sequence {a : ℕ → ℕ} (k : ℕ) (h : ∀ n ≥ 2, a (n + 1) = k * (a n) / (a (n - 1))) 
  (h0 : a 1 = 1) (h1 : a 2018 = 2020) (h2 : ∀ n, a n ∈ ℕ) : k = 2020 :=
sorry

end smallest_k_for_sequence_l521_521196


namespace christopher_strolled_distance_l521_521688

variable speed : ℝ := 4
variable time : ℝ := 1.25

theorem christopher_strolled_distance : speed * time = 5 := by
  sorry

end christopher_strolled_distance_l521_521688


namespace combined_rainfall_is_23_l521_521891

-- Define the conditions
def monday_hours : ℕ := 7
def monday_rate : ℕ := 1
def tuesday_hours : ℕ := 4
def tuesday_rate : ℕ := 2
def wednesday_hours : ℕ := 2
def wednesday_rate (tuesday_rate : ℕ) : ℕ := 2 * tuesday_rate

-- Calculate the rainfalls
def monday_rainfall : ℕ := monday_hours * monday_rate
def tuesday_rainfall : ℕ := tuesday_hours * tuesday_rate
def wednesday_rainfall (wednesday_rate : ℕ) : ℕ := wednesday_hours * wednesday_rate

-- Define the total rainfall
def total_rainfall : ℕ :=
  monday_rainfall + tuesday_rainfall + wednesday_rainfall (wednesday_rate tuesday_rate)

theorem combined_rainfall_is_23 : total_rainfall = 23 := by
  -- Proof to be filled in
  sorry

end combined_rainfall_is_23_l521_521891


namespace triangle_possible_sides_l521_521786

theorem triangle_possible_sides : 
  let unknown_side_lengths := {x : ℤ | 3 < x ∧ x < 13}
  in set.finite unknown_side_lengths ∧ set.card unknown_side_lengths = 9 :=
by
  sorry

end triangle_possible_sides_l521_521786


namespace example_problem_l521_521095

variable (a b c d : ℝ)

theorem example_problem :
  (a + (b + c - d) = a + b + c - d) ∧
  (a - (b - c + d) = a - b + c - d) ∧
  (a - b - (c - d) ≠ a - b - c - d) ∧
  (a + b - (-c - d) = a + b + c + d) :=
by {
  sorry
}

end example_problem_l521_521095


namespace solution_set_of_inequality_l521_521438

def f : ℝ → ℝ := sorry  -- f is an even function

theorem solution_set_of_inequality (h_even : ∀ x : ℝ, f (-x) = f x)
                                    (h_f_neg4 : f (-4) = 0)
                                    (h_f_2 : f 2 = 0)
                                    (h_decreasing_0_3 : ∀ x y : ℝ, 0 ≤ x → x ≤ y → y ≤ 3 → f y ≤ f x)
                                    (h_increasing_3_inf : ∀ x y : ℝ, 3 ≤ x → x ≤ y → f x ≤ f y):
  {x : ℝ | x * f x < 0} = set.Ioo (-∞) (-4) ∪ set.Ioo (-2) 0 ∪ set.Ioo 2 4 := 
sorry

end solution_set_of_inequality_l521_521438


namespace area_quadrilateral_ABCD_l521_521030

-- Define points and lengths based on given conditions
variables (A B D C : Type)
variables [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field D] [linear_ordered_field C]

-- Assume the triangle sides based on given conditions
variables 
  (AB BD BC : ℝ) 
  (h1 : AB = 12)
  (h2 : BD = 15)
  (h3 : BC = 17)

-- Define right triangles based on given conditions
axioms
(ABD_right : triangle_right_angle A B D)
(BCD_right : triangle_right_angle B D C)

-- Define areas of the triangles based on given sides
noncomputable def area_triangle_ABD : ℝ := 0.5 * AB * 9 -- AD inferred from Pythagorean triple
noncomputable def area_triangle_BCD : ℝ := 0.5 * BD * 8 -- CD inferred from Pythagorean triple

-- Calculate total area
noncomputable def total_area_ABCD : ℝ := area_triangle_ABD + area_triangle_BCD

-- State the main theorem
theorem area_quadrilateral_ABCD : total_area_ABCD = 114 :=
by {
  sorry
}

end area_quadrilateral_ABCD_l521_521030


namespace judah_crayons_l521_521898

theorem judah_crayons (karen beatrice gilbert judah : ℕ) 
  (h1 : karen = 128)
  (h2 : karen = 2 * beatrice)
  (h3 : beatrice = 2 * gilbert)
  (h4 : gilbert = 4 * judah) : 
  judah = 8 :=
by
  sorry

end judah_crayons_l521_521898


namespace j_self_inverse_l521_521675

def h (x : ℝ) : ℝ := (x - 2) / (x - 3)

def j (x b : ℝ) : ℝ := h(x + b)

theorem j_self_inverse {b : ℝ} : (∀ x : ℝ, j (j x b) b = x) ↔ b = -1 := by
  sorry

end j_self_inverse_l521_521675


namespace sequence_geometric_and_lambda_range_l521_521738

theorem sequence_geometric_and_lambda_range:
  (∃ (a : ℕ → ℤ), a 1 = 0 ∧ (∀ n : ℕ, n ≥ 1 → a (n+1) - 2 * a n = 2 * n)) →
  (∃ n : ℕ, n ≥ 1 → a n = 2^(n+1) - 2*n - 2) ∧ 
  (∀ λ : ℚ, (∀ n : ℕ, n ≥ 1 → a n > λ * (2*n + 1) * (-1)^(n-1)) →
    -2/5 < λ ∧ λ < 0) :=
by
  sorry

end sequence_geometric_and_lambda_range_l521_521738


namespace solve_quadratic_eq_l521_521955

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem solve_quadratic_eq :
  ∀ a b c x1 x2 : ℝ,
  a = 2 →
  b = -2 →
  c = -1 →
  quadratic_eq a b c x1 ∧ quadratic_eq a b c x2 →
  (x1 = (1 + Real.sqrt 3) / 2 ∧ x2 = (1 - Real.sqrt 3) / 2) :=
by
  intros a b c x1 x2 ha hb hc h
  sorry

end solve_quadratic_eq_l521_521955


namespace decagon_adjacent_vertex_probability_l521_521556

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l521_521556


namespace decagon_adjacent_vertex_probability_l521_521547

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l521_521547


namespace magnitude_vector_calculation_l521_521184

variable (a b c : ℝ × ℝ × ℝ)
#check ⟨a, b, c⟩

theorem magnitude_vector_calculation (ha : a = (1, 0, 1)) 
                                    (hb : b = (-2, -1, 1)) 
                                    (hc : c = (3, 1, 0)) : 
  (∥(a.1 - b.1 + 2 * c.1, a.2 - b.2 + 2 * c.2, a.3 - b.3 + 2 * c.3)∥ = 3 * Real.sqrt 10) :=
by 
  sorry

end magnitude_vector_calculation_l521_521184


namespace triangle_side_lengths_l521_521840

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end triangle_side_lengths_l521_521840


namespace max_smallest_angle_l521_521250

theorem max_smallest_angle (points : List (ℝ × ℝ)) (h_len : points.length = 5) :
  ∃ α, (∀ (p1 p2 p3 : (ℝ × ℝ)), (p1 ∈ points) ∧ (p2 ∈ points) ∧ (p3 ∈ points) ∧ (p1 ≠ p2) ∧ (p1 ≠ p3) ∧ (p2 ≠ p3) → 
  angle p1 p2 p3 ≥ α) ∧ α ≤ 36 :=
sorry

end max_smallest_angle_l521_521250


namespace badgers_at_least_five_wins_l521_521960

open BigOperators

noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem badgers_at_least_five_wins :
  let n := 9
  let p := 0.5
  (∑ k in (finset.range (n + 1)).filter (λ k, 5 ≤ k), binomial_prob n k p) = 1 / 2 :=
by
  sorry

end badgers_at_least_five_wins_l521_521960


namespace rectangle_total_area_l521_521874

-- Let s be the side length of the smaller squares
variable (s : ℕ)

-- Define the areas of the squares
def smaller_square_area := s ^ 2
def larger_square_area := (3 * s) ^ 2

-- Define the total_area
def total_area : ℕ := 2 * smaller_square_area s + larger_square_area s

-- Assert the total area of the rectangle ABCD is 11s^2
theorem rectangle_total_area (s : ℕ) : total_area s = 11 * s ^ 2 := 
by 
  -- the proof is skipped
  sorry

end rectangle_total_area_l521_521874


namespace salary_problem_l521_521977

theorem salary_problem
  (A B : ℝ)
  (h1 : A + B = 3000)
  (h2 : 0.05 * A = 0.15 * B) :
  A = 2250 :=
sorry

end salary_problem_l521_521977


namespace exists_sum_of_two_squares_l521_521413

theorem exists_sum_of_two_squares (n : ℕ) (h₁ : n > 10000) : 
  ∃ m : ℕ, (∃ a b : ℕ, m = a^2 + b^2) ∧ 0 < m - n ∧ m - n < 3 * Real.sqrt n := 
sorry

end exists_sum_of_two_squares_l521_521413


namespace bobby_pays_correct_amount_l521_521140

noncomputable def bobby_total_cost : ℝ := 
  let mold_cost : ℝ := 250
  let material_original_cost : ℝ := 150
  let material_discount : ℝ := 0.20 * material_original_cost
  let material_cost : ℝ := material_original_cost - material_discount
  let hourly_rate_original : ℝ := 75
  let hourly_rate_increased : ℝ := hourly_rate_original + 10
  let work_hours : ℝ := 8
  let work_cost_original : ℝ := work_hours * hourly_rate_increased
  let work_cost_discount : ℝ := 0.80 * work_cost_original
  let cost_before_tax : ℝ := mold_cost + material_cost + work_cost_discount
  let tax : ℝ := 0.10 * cost_before_tax
  cost_before_tax + tax

theorem bobby_pays_correct_amount : bobby_total_cost = 1005.40 := sorry

end bobby_pays_correct_amount_l521_521140


namespace length_of_AE_l521_521880

noncomputable def lengthAE_proof : ℝ :=
  let A := (0, 5) : ℤ × ℤ;
  let B := (5, 0) : ℤ × ℤ;
  let D := (1, 0) : ℤ × ℤ;
  let C := (5, 3) : ℤ × ℤ;
  let E := (5/6, 25/6) : ℚ × ℚ; -- solved using similar triangles from the solution
  let AE := real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2);
  AE

-- Now state the theorem
theorem length_of_AE : lengthAE_proof = 25 * real.sqrt(2) / 6 := sorry

end length_of_AE_l521_521880


namespace apple_juice_percentage_in_blend_l521_521392

-- Conditions
def apples: ℕ := 18
def bananas: ℕ := 18
def apple_juice_per_apple: ℝ := 10 / 5
def banana_juice_per_banana: ℝ := 12 / 4

-- Problem Statement
theorem apple_juice_percentage_in_blend (n: ℕ) (h1: n ≤ apples) (h2: n ≤ bananas) :
  let total_apple_juice := n * apple_juice_per_apple in
  let total_banana_juice := n * banana_juice_per_banana in
  let total_juice := total_apple_juice + total_banana_juice in
  (total_apple_juice / total_juice) * 100 = 40 := 
by
  sorry

end apple_juice_percentage_in_blend_l521_521392


namespace plane_equation_l521_521025

variable (x y z : ℝ)

def line1 := 3 * x - 2 * y + 5 * z + 3 = 0
def line2 := x + 2 * y - 3 * z - 11 = 0
def origin_plane := 18 * x - 8 * y + 23 * z = 0

theorem plane_equation : 
  (∀ x y z, line1 x y z → line2 x y z → origin_plane x y z) :=
by
  sorry

end plane_equation_l521_521025


namespace unique_family_of_quadratic_polynomials_l521_521239

theorem unique_family_of_quadratic_polynomials :
  ∃ (a : ℝ) (b : ℝ) (c : ℝ) (p : ℝ → ℝ),
    (a ≠ 0) ∧ (p = λ x, a * x^2 + b * x + c) ∧
    (∀ r s, p(r) = 0 ∧ p(s) = 0 → r * s = a + b + c) ∧
    (a = 1 ∧ b = -1 ∧ ∀ c : ℝ, ∃ r s : ℝ, p(r) = 0 ∧ p(s) = 0) :=
begin
  sorry
end

end unique_family_of_quadratic_polynomials_l521_521239


namespace not_possible_to_create_3_piles_l521_521364

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end not_possible_to_create_3_piles_l521_521364


namespace shop_weekly_earnings_l521_521671

theorem shop_weekly_earnings
  (price_women: ℕ := 18)
  (price_men: ℕ := 15)
  (time_open_hours: ℕ := 12)
  (minutes_per_hour: ℕ := 60)
  (weekly_days: ℕ := 7)
  (sell_rate_women: ℕ := 30)
  (sell_rate_men: ℕ := 40) :
  (time_open_hours * (minutes_per_hour / sell_rate_women) * price_women +
   time_open_hours * (minutes_per_hour / sell_rate_men) * price_men) * weekly_days = 4914 := 
sorry

end shop_weekly_earnings_l521_521671


namespace find_solutions_l521_521163

theorem find_solutions (x : ℝ) (hx₁ : ∀ y : ℝ, y = Real.root 4 x → 
  y = 15 / (8 - y^2)) :
  x ≈ 5.0625 ∨ x ≈ 39.0625 :=
by
  -- Lean statement for proving the solutions
  sorry

end find_solutions_l521_521163


namespace cosine_angle_between_a_b_coordinates_of_c_l521_521235

-- Define the vectors
def vec_a : ℝ × ℝ := (4, 3)
def vec_b : ℝ × ℝ := (-1, 2)
def norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Problem 1
theorem cosine_angle_between_a_b :
  let cos_angle := dot_product vec_a vec_b / (norm vec_a * norm vec_b)
  cos_angle = 2 * Real.sqrt 5 / 25 := 
by
  sorry

-- Problem 2
def vec_c (λ : ℝ) : ℝ × ℝ := (4 * λ, 3 * λ)

theorem coordinates_of_c :
  let λ := 2
  let c := vec_c λ
  norm c = 10 ∧ c = (8, 6) :=
by
  sorry

end cosine_angle_between_a_b_coordinates_of_c_l521_521235


namespace graph_intersections_between_2_and_40_l521_521116

def parametric_x (t: ℝ) : ℝ := (Real.cos t) + (t / 2) + (Real.sin (2 * t))
def parametric_y (t: ℝ) : ℝ := Real.sin t

theorem graph_intersections_between_2_and_40:
  {t s: ℝ} (t ≠ s) (2 ≤ parametric_x t) (parametric_x t ≤ 40) (2 ≤ parametric_x s) (parametric_x s ≤ 40) :
  ∃ (n: ℤ), 1 ≤ n ∧ n ≤ 12 :=
sorry

end graph_intersections_between_2_and_40_l521_521116


namespace pentagon_inscribed_circle_l521_521638

theorem pentagon_inscribed_circle :
  ∀ (pentagon : Type) (circle : Type)
  (inscribed : ∀ (A B C D E : pentagon), ∃ (O : circle), True)
  (segments_angle : ∀ (A B C D E : pentagon), ℝ),
  (segments_angle A + segments_angle B + segments_angle C + segments_angle D + segments_angle E = 720) :=
sorry

end pentagon_inscribed_circle_l521_521638


namespace impossible_to_form_three_similar_piles_l521_521353

-- Define the concept of similar sizes
def similar_sizes (a b : ℝ) : Prop := a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2

-- State the proof problem
theorem impossible_to_form_three_similar_piles (x : ℝ) (h : 0 < x) : ¬(∃ a b c : ℝ, a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes a c) :=
sorry

end impossible_to_form_three_similar_piles_l521_521353


namespace acute_triangle_area_range_l521_521876

theorem acute_triangle_area_range (A B C : ℝ) (a b c : ℝ) (h_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) 
(h_arith_seq : 2 * B = A + C) (h_sum_angles : A + B + C = π) (h_b : b = sqrt 3) :
  ∃ S : ℝ, (sqrt 3 / 2) < S ∧ S <= (3 * sqrt 3 / 4) :=
sorry

end acute_triangle_area_range_l521_521876


namespace smallest_possible_value_of_N_l521_521013

-- Conditions definition:
variable (a b c d : ℕ)
variable (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
variable (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
variable (gcd_ab : Int.gcd a b = 1)
variable (gcd_ac : Int.gcd a c = 2)
variable (gcd_ad : Int.gcd a d = 4)
variable (gcd_bc : Int.gcd b c = 5)
variable (gcd_bd : Int.gcd b d = 3)
variable (gcd_cd : Int.gcd c d = N)
variable (hN : N > 5)

-- Statement to prove:
theorem smallest_possible_value_of_N : N = 14 := sorry

end smallest_possible_value_of_N_l521_521013


namespace decagon_adjacent_probability_l521_521545

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l521_521545


namespace decagon_adjacent_probability_l521_521533

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l521_521533


namespace good_p_tuples_count_l521_521306

theorem good_p_tuples_count (p : ℕ) (hp : p.prime) (is_odd : p % 2 = 1) :
  ∃ n, n = (p^(p-2)*(p-1)) ∧ 
    ∀ (a : Fin p → ℕ), 
      (∀ i, 0 ≤ a i ∧ a i ≤ p-1) → 
      (∑ i in Finset.range p, a i) % p ≠ 0 → 
      (∑ i in Finset.range p, a i * a ((i + 1) % p)) % p = 0 → 
        n = p^{p-2} * (p-1) :=
sorry

end good_p_tuples_count_l521_521306


namespace PolygonNumberSides_l521_521858

theorem PolygonNumberSides (n : ℕ) (h : n - (1 / 2 : ℝ) * (n * (n - 3)) / 2 = 0) : n = 7 :=
by
  sorry

end PolygonNumberSides_l521_521858


namespace max_dot_product_l521_521313

noncomputable def vec_a : ℝ × ℝ × ℝ := (1, 1, -2)
variables (x y z : ℝ)

def vec_b : ℝ × ℝ × ℝ := (x, y, z)
def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

def norm_squared (b : ℝ × ℝ × ℝ) : ℝ :=
  b.1^2 + b.2^2 + b.3^2

theorem max_dot_product :
  norm_squared (vec_b x y z) = 16 →
  dot_product vec_a (vec_b x y z) ≤ 4 * real.sqrt 6 :=
begin
  sorry
end

end max_dot_product_l521_521313


namespace Linda_oatmeal_raisin_batches_l521_521390

theorem Linda_oatmeal_raisin_batches :
  ∀ (classmates cookies_per_classmate recipe_yield chocolate_chip_batches more_batches: ℕ), 
  classmates = 24 → 
  cookies_per_classmate = 10 → 
  recipe_yield = 4 * 12 → 
  chocolate_chip_batches = 2 → 
  more_batches = 2 → 
  (classmates * cookies_per_classmate - chocolate_chip_batches * recipe_yield) / recipe_yield - more_batches = 1 := 
by
  intros classmates cookies_per_classmate recipe_yield chocolate_chip_batches more_batches
  intros h_classmates h_cookies_per_classmate h_recipe_yield h_chocolate_chip_batches h_more_batches
  let total_cookies := 24 * 10
  let cookies_per_batch := 4 * 12
  let choco_chip_cookies := 2 * 48
  let remaining_cookies := 240 - 96
  let oatmeal_batches_needed := 144 / 48
  let pre_existing_batches := 3 - 2
  have : (classmates * cookies_per_classmate - chocolate_chip_batches * recipe_yield) / recipe_yield - more_batches = 1 := rfl
  exact (eq_of_sub_eq_one _).mpr sorry -- this is placeholder and needs proper computations

end Linda_oatmeal_raisin_batches_l521_521390


namespace find_a_plus_k_l521_521661

-- Define the conditions.
def foci1 : (ℝ × ℝ) := (2, 0)
def foci2 : (ℝ × ℝ) := (2, 4)
def ellipse_point : (ℝ × ℝ) := (7, 2)

-- Statement of the equivalent proof problem.
theorem find_a_plus_k (a b h k : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (∀ x y, ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ↔ (x, y) = ellipse_point) →
  h = 2 → k = 2 → a = 5 →
  a + k = 7 :=
by
  sorry

end find_a_plus_k_l521_521661


namespace circle_center_l521_521167

theorem circle_center (x y : ℝ) :
  (∀ x y, x^2 - 4*x + y^2 - 6*y - 12 = 0) ↔ (x = 2 ∧ y = 3) :=
begin
  sorry
end

end circle_center_l521_521167


namespace pyramid_volume_l521_521962

theorem pyramid_volume (R α β : ℝ) : 
  let V := \frac{4}{3} * R^3 * (Real.sin (2 * β))^2 * (Real.sin β)^2 * Real.sin α in
  V = V := sorry

end pyramid_volume_l521_521962


namespace decagon_adjacent_probability_l521_521522

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521522


namespace expected_value_of_fair_dodecahedral_die_l521_521052

theorem expected_value_of_fair_dodecahedral_die : 
  (1/12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_of_fair_dodecahedral_die_l521_521052


namespace two_digit_multiples_of_30_eq_2_l521_521845

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_multiple_of_30 (n : ℕ) : Prop := n % 30 = 0

theorem two_digit_multiples_of_30_eq_2 :
  {n : ℕ | is_two_digit n ∧ is_multiple_of_30 n}.card = 2 :=
by
  sorry

end two_digit_multiples_of_30_eq_2_l521_521845


namespace number_of_sides_possibilities_l521_521816

theorem number_of_sides_possibilities (a b : ℕ) (h₁ : a = 8) (h₂ : b = 5) : 
  { x : ℕ // 3 < x ∧ x < 13 }.to_list.length = 9 :=
by
  sorry

end number_of_sides_possibilities_l521_521816


namespace meeting_distance_from_A_l521_521402

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (distance_AB distance_BC : ℝ)
variable (cyclist_speed pedestrian_speed : ℝ)
variable (meet_distance : ℝ)

axiom distance_AB_eq_3 : distance_AB = 3
axiom distance_BC_eq_4 : distance_BC = 4
axiom simultaneous_arrival :
  ∀ AC cyclist_speed pedestrian_speed,
    (distance_AB + distance_BC) / cyclist_speed = distance_AB / pedestrian_speed
axiom speed_ratio :
  cyclist_speed / pedestrian_speed = 7 / 3
axiom meeting_point :
  ∃ meet_distance,
    meet_distance / (distance_AB - meet_distance) = 7 / 3

theorem meeting_distance_from_A :
  meet_distance = 2.1 :=
sorry

end meeting_distance_from_A_l521_521402


namespace equal_illumination_points_l521_521397

-- Define the parameters
variables (a b d : ℝ) (h : a ≠ b)

-- Define the two possible solutions for the position x
def x1 : ℝ := (d * real.sqrt a) / (real.sqrt a - real.sqrt b)
def x2 : ℝ := (d * real.sqrt a) / (real.sqrt a + real.sqrt b)

-- Proof statement
theorem equal_illumination_points :
  ∀ (x : ℝ), (x = x1 a b d ∨ x = x2 a b d) ↔ (a / x^2 = b / (d - x)^2) :=
by
  sorry

end equal_illumination_points_l521_521397


namespace crayons_in_judahs_box_l521_521896

theorem crayons_in_judahs_box (karen_crayons beatrice_crayons gilbert_crayons judah_crayons : ℕ)
  (h1 : karen_crayons = 128)
  (h2 : beatrice_crayons = karen_crayons / 2)
  (h3 : gilbert_crayons = beatrice_crayons / 2)
  (h4 : judah_crayons = gilbert_crayons / 4) :
  judah_crayons = 8 :=
by {
  sorry
}

end crayons_in_judahs_box_l521_521896


namespace expected_value_of_dodecahedral_die_is_6_5_l521_521044

noncomputable def expected_value_of_dodecahedral_die : ℝ := 
  ∑ i in Finset.range (12 + 1), (i : ℝ) / 12

theorem expected_value_of_dodecahedral_die_is_6_5 :
  expected_value_of_dodecahedral_die = 6.5 := sorry

end expected_value_of_dodecahedral_die_is_6_5_l521_521044


namespace sum_of_areas_of_rectangles_l521_521431

theorem sum_of_areas_of_rectangles :
  let widths := [2, 2, 2, 2, 2, 2]
  let lengths := [1, 4, 9, 16, 25, 36]
  let areas := List.map2 (λ w l => w * l) widths lengths
  List.sum areas = 182 := by
  sorry

end sum_of_areas_of_rectangles_l521_521431


namespace sin_identity_l521_521420

theorem sin_identity :
  sin (7 * Real.pi / 30) + sin (11 * Real.pi / 30) = sin (Real.pi / 30) + sin (13 * Real.pi / 30) + 1 / 2 :=
by sorry

end sin_identity_l521_521420


namespace number_of_valid_A_l521_521483

-- Define the range of A
def A_in_range (A : ℕ) : Prop := 1 ≤ A ∧ A ≤ 9

-- Condition for A to satisfy A888 < 5001
def A_condition (A : ℕ) : Prop := A < 5

-- Combining the conditions to specify the valid values of A
def valid_A (A : ℕ) : Prop := A_in_range A ∧ A_condition A

-- Main theorem to prove
theorem number_of_valid_A : (finset.filter valid_A (finset.Icc 1 9)).card = 4 :=
by
  sorry

end number_of_valid_A_l521_521483


namespace chip_notebook_packs_l521_521682

theorem chip_notebook_packs (pages_per_day : ℕ) (days_per_week : ℕ) (classes : ℕ) (sheets_per_pack : ℕ) (weeks : ℕ) :
  pages_per_day = 2 → days_per_week = 5 → classes = 5 → sheets_per_pack = 100 → weeks = 6 →
  (classes * pages_per_day * days_per_week * weeks) / sheets_per_pack = 3 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  sorry

end chip_notebook_packs_l521_521682


namespace canvas_bag_lower_carbon_solution_l521_521929

theorem canvas_bag_lower_carbon_solution :
  ∀ (canvas_bag_CO2_pounds : ℕ) (plastic_bag_CO2_ounces : ℕ) 
    (plastic_bags_per_trip : ℕ) (ounces_per_pound : ℕ),
    canvas_bag_CO2_pounds = 600 →
    plastic_bag_CO2_ounces = 4 →
    plastic_bags_per_trip = 8 →
    ounces_per_pound = 16 →
    let total_CO2_per_trip := plastic_bags_per_trip * plastic_bag_CO2_ounces / ounces_per_pound in
    (canvas_bag_CO2_pounds / total_CO2_per_trip) = 300 :=
by
  -- Assume the given conditions
  assume canvas_bag_CO2_pounds plastic_bag_CO2_ounces plastic_bags_per_trip ounces_per_pound,
  assume canvas_bag_CO2_pounds_eq plastic_bag_CO2_ounces_eq plastic_bags_per_trip_eq ounces_per_pound_eq,
  -- Introduce the total carbon dioxide per trip
  let total_CO2_per_trip := plastic_bags_per_trip * plastic_bag_CO2_ounces / ounces_per_pound,
  -- Verify that the number of trips is 300
  show ((canvas_bag_CO2_pounds / total_CO2_per_trip) = 300),
  sorry

end canvas_bag_lower_carbon_solution_l521_521929


namespace make_polynomial_perfect_square_l521_521027

theorem make_polynomial_perfect_square (m : ℝ) :
  m = 196 → ∃ (f : ℝ → ℝ), ∀ x : ℝ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = (f x) ^ 2 :=
by
  sorry

end make_polynomial_perfect_square_l521_521027


namespace minimized_perimeter_area_l521_521747

noncomputable def F : ℝ × ℝ := (2, 0) -- Right focus of the hyperbola

def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 8 = 1

def A : ℝ × ℝ := (0, 6 * Real.sqrt 6) -- Point A given in the problem

variables (P : ℝ × ℝ) (hP : hyperbola P.1 P.2) -- P is a point on the left branch of the hyperbola

theorem minimized_perimeter_area
  (hF : F = (2, 0))
  (hA : A = (0, 6 * Real.sqrt 6))
  (h_perimeter_min : ∀ Q : ℝ × ℝ, hyperbola Q.1 Q.2 → 
    let d1 := Real.dist A Q
    let d2 := Real.dist Q F
    let d3 := Real.dist A F
    d1 + d2 ≤ d1 + d3 + Real.dist A (-Q.1, Q.2)) : -- Condition ensuring perimeter is minimized
  ∃ (P : ℝ × ℝ), hyperbola P.1 P.2 ∧ area_of_triangle A P F = 12 * Real.sqrt 6 :=
sorry -- Proof omitted

end minimized_perimeter_area_l521_521747


namespace expected_value_of_dodecahedral_die_is_6_5_l521_521039

noncomputable def expected_value_of_dodecahedral_die : ℝ := 
  ∑ i in Finset.range (12 + 1), (i : ℝ) / 12

theorem expected_value_of_dodecahedral_die_is_6_5 :
  expected_value_of_dodecahedral_die = 6.5 := sorry

end expected_value_of_dodecahedral_die_is_6_5_l521_521039


namespace not_possible_to_create_3_piles_l521_521367

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end not_possible_to_create_3_piles_l521_521367


namespace no_hexagonal_cross_section_l521_521490

-- Define a quadrilateral-based pyramid
structure Pyramid :=
(base : Type) -- The base is a quadrilateral
(faces : ℕ)   -- The pyramid has 5 faces

-- Assumption of the problem
def quadrilateral_based_pyramid : Pyramid :=
{ base := Quadrilateral,
  faces := 5 }

-- The theorem we want to prove
theorem no_hexagonal_cross_section (p : Pyramid) (hp : p = quadrilateral_based_pyramid) :
  ¬(∃ cross_section : Type, is_hexagon cross_section) :=
by
  sorry

end no_hexagonal_cross_section_l521_521490


namespace continuous_nondecreasing_of_property_F_l521_521631

noncomputable theory
open set

def property_F (f : ℝ → ℝ) : Prop :=
∀ a : ℝ, ∃ b < a, ∀ x ∈ Ioo b a, f x ≤ f a

theorem continuous_nondecreasing_of_property_F
  {f : ℝ → ℝ}
  (h_cont : continuous f)
  (h_F : property_F f) :
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≤ f x2 :=
by sorry

end continuous_nondecreasing_of_property_F_l521_521631


namespace total_preferred_goldfish_l521_521664

theorem total_preferred_goldfish :
  let n_j := 30 in
  let p_j := 1 / 6 in
  let n_f := 45 in
  let p_f := 2 / 3 in
  let n_h := 36 in
  let p_h := 1 / 5 in
  let n_d := 50 in
  let p_d := 3 / 5 in
  let n_n := 25 in
  let p_n := 2 / 5 in
  n_j * p_j + n_f * p_f + n_h * p_h + n_d * p_d + n_n * p_n = 82 := 
by
  sorry


end total_preferred_goldfish_l521_521664


namespace smallest_GCD_value_l521_521008

theorem smallest_GCD_value (a b c d N : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
    (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : N > 5)
    (hc1 : Nat.gcd a b = 1 ∨ Nat.gcd a c = 1 ∨ Nat.gcd a d = 1 ∨ Nat.gcd b c = 1 ∨ Nat.gcd b d = 1 ∨ Nat.gcd c d = 1)
    (hc2 : Nat.gcd a b = 2 ∨ Nat.gcd a c = 2 ∨ Nat.gcd a d = 2 ∨ Nat.gcd b c = 2 ∨ Nat.gcd b d = 2 ∨ Nat.gcd c d = 2)
    (hc3 : Nat.gcd a b = 3 ∨ Nat.gcd a c = 3 ∨ Nat.gcd a d = 3 ∨ Nat.gcd b c = 3 ∨ Nat.gcd b d = 3 ∨ Nat.gcd c d = 3)
    (hc4 : Nat.gcd a b = 4 ∨ Nat.gcd a c = 4 ∨ Nat.gcd a d = 4 ∨ Nat.gcd b c = 4 ∨ Nat.gcd b d = 4 ∨ Nat.gcd c d = 4)
    (hc5 : Nat.gcd a b = 5 ∨ Nat.gcd a c = 5 ∨ Nat.gcd a d = 5 ∨ Nat.gcd b c = 5 ∨ Nat.gcd b d = 5 ∨ Nat.gcd c d = 5)
    (hcN : Nat.gcd a b = N ∨ Nat.gcd a c = N ∨ Nat.gcd a d = N ∨ Nat.gcd b c = N ∨ Nat.gcd b d = N ∨ Nat.gcd c d = N):
    N = 14 :=
sorry

end smallest_GCD_value_l521_521008


namespace calculate_product_l521_521461

theorem calculate_product (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3*x1*y1^2 = 2030)
  (h2 : y1^3 - 3*x1^2*y1 = 2029)
  (h3 : x2^3 - 3*x2*y2^2 = 2030)
  (h4 : y2^3 - 3*x2^2*y2 = 2029)
  (h5 : x3^3 - 3*x3*y3^2 = 2030)
  (h6 : y3^3 - 3*x3^2*y3 = 2029) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = -1 / 1015 :=
sorry

end calculate_product_l521_521461


namespace teachers_neither_condition_percentage_l521_521648

-- Definitions based on the conditions
def total_teachers : ℕ := 150
def high_bp_teachers : ℕ := 80
def heart_trouble_teachers : ℕ := 50
def both_conditions : ℕ := 30

-- Lean 4 proof statement
theorem teachers_neither_condition_percentage : 
  (total_teachers - (high_bp_teachers + heart_trouble_teachers - both_conditions)) * 100 / total_teachers = 33.33 :=
by
  sorry

end teachers_neither_condition_percentage_l521_521648


namespace div_z_x_l521_521852

variables (x y z : ℚ)

theorem div_z_x (h1 : x / y = 3) (h2 : y / z = 5 / 2) : z / x = 2 / 15 :=
sorry

end div_z_x_l521_521852


namespace possible_integer_side_lengths_l521_521826

theorem possible_integer_side_lengths (a b : ℕ) (h1 : a = 8) (h2 : b = 5) :
  {x : ℕ | 3 < x ∧ x < 13}.finite ∧ {x : ℕ | 3 < x ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521826


namespace decagon_adjacent_probability_l521_521584

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521584


namespace probability_adjacent_vertices_of_decagon_l521_521509

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l521_521509


namespace find_f_zero_l521_521211

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_zero (a : ℝ) (h1 : ∀ x : ℝ, f (x - a) = x^3 + 1)
  (h2 : ∀ x : ℝ, f x + f (2 - x) = 2) : 
  f 0 = 0 :=
sorry

end find_f_zero_l521_521211


namespace even_function_maximum_l521_521860

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

noncomputable def has_maximum_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x : ℝ, a ≤ x ∧ x ≤ b ∧ ∀ y : ℝ, a ≤ y ∧ y ≤ b → f y ≤ f x

theorem even_function_maximum 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_max_1_7 : has_maximum_on_interval f 1 7) :
  has_maximum_on_interval f (-7) (-1) :=
sorry

end even_function_maximum_l521_521860


namespace num_possible_triangle_sides_l521_521806

theorem num_possible_triangle_sides (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (a_eq_8 : a = 8) (b_eq_5 : b = 5) : 
  let x := {n : ℕ | 3 < n ∧ n < 13} in
  set.card x = 9 :=
by
  sorry

end num_possible_triangle_sides_l521_521806


namespace not_possible_to_create_3_piles_l521_521370

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end not_possible_to_create_3_piles_l521_521370


namespace abs_div_inequality_l521_521717

theorem abs_div_inequality (x : ℝ) : 
  (|-((x+1)/x)| > (x+1)/x) ↔ (-1 < x ∧ x < 0) :=
sorry

end abs_div_inequality_l521_521717


namespace spherical_to_cartesian_l521_521699

theorem spherical_to_cartesian 
  (ρ θ φ : ℝ)
  (hρ : ρ = 3) 
  (hθ : θ = 7 * Real.pi / 12) 
  (hφ : φ = Real.pi / 4) :
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ) = 
  (3 * Real.sqrt 2 / 2 * Real.cos (7 * Real.pi / 12), 
   3 * Real.sqrt 2 / 2 * Real.sin (7 * Real.pi / 12), 
   3 * Real.sqrt 2 / 2) :=
by
  sorry

end spherical_to_cartesian_l521_521699


namespace revenue_increase_l521_521626

theorem revenue_increase
  (P Q : ℝ)
  (h : 0 < P)
  (hQ : 0 < Q)
  (price_decrease : 0.90 = 0.90)
  (unit_increase : 2 = 2) :
  (0.90 * P) * (2 * Q) = 1.80 * (P * Q) :=
by
  sorry

end revenue_increase_l521_521626


namespace right_triangle_side_length_l521_521124

theorem right_triangle_side_length (hypotenuse : ℝ) (θ : ℝ) (sin_30 : Real.sin 30 = 1 / 2) (h : θ = 30) 
  (hyp_len : hypotenuse = 10) : 
  let opposite_side := hypotenuse * Real.sin θ
  opposite_side = 5 := by
  sorry

end right_triangle_side_length_l521_521124


namespace range_of_a_l521_521252

theorem range_of_a (a : ℝ) (h : a ≠ 0) : 
  (∀ x : ℝ, x > 0 → x < ∞ → 
   ∃ y : ℝ, y > x ∧ y < x + 1) → 
  - (1 / 8) < a ∧ a < 0 :=
sorry

end range_of_a_l521_521252


namespace geom_sixth_term_is_31104_l521_521859

theorem geom_sixth_term_is_31104 :
  ∃ (r : ℝ), 4 * r^8 = 39366 ∧ 4 * r^(6-1) = 31104 :=
by
  sorry

end geom_sixth_term_is_31104_l521_521859


namespace water_formed_main_reaction_l521_521679

-- Define the initial conditions
def initial_moles_hcl : ℕ := 4
def initial_moles_caoh2 : ℕ := 2
def initial_volume_liters : ℝ := 5.0
def initial_temperature_celsius : ℝ := 27.0
def initial_pressure_atm : ℝ := 1.0

-- Define the main reaction
def main_reaction_hcl_consumed : ℕ := 4
def main_reaction_h2o_produced : ℕ := 4

-- Define the statement to prove
theorem water_formed_main_reaction :
  main_reaction_hcl_consumed = initial_moles_hcl →
  2 * initial_moles_caoh2 * 1 = main_reaction_hcl_consumed / 2 →
  main_reaction_h2o_produced = 4 :=
begin
  intros h1 h2,
  sorry,
end

-- This theorem implies that given the initial conditions of the reactants,
-- 4 moles of water are formed as a result of the main reaction.

end water_formed_main_reaction_l521_521679


namespace impossible_to_create_3_piles_l521_521378

def similar_sizes (x y : ℝ) : Prop := x ≤ sqrt 2 * y

theorem impossible_to_create_3_piles (x y : ℝ) (h₁ : similar_sizes x y) :
  ¬ ∃ (x₁ x₂ x₃ : ℝ), x₁ + x₂ + x₃ = x + y ∧ similar_sizes x₁ x₂ ∧ similar_sizes x₂ x₃ ∧ similar_sizes x₁ x₃ :=
by
  sorry

end impossible_to_create_3_piles_l521_521378


namespace decagon_adjacent_probability_l521_521542

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l521_521542


namespace gum_lcm_l521_521925

theorem gum_lcm (strawberry blueberry cherry : ℕ) (h₁ : strawberry = 6) (h₂ : blueberry = 5) (h₃ : cherry = 8) :
  Nat.lcm (Nat.lcm strawberry blueberry) cherry = 120 :=
by
  rw [h₁, h₂, h₃]
  -- LCM(6, 5, 8) = LCM(LCM(6, 5), 8)
  sorry

end gum_lcm_l521_521925


namespace symmetry_axis_sin_eq_pi_over_3_l521_521169

theorem symmetry_axis_sin_eq_pi_over_3 :
  ∃ (x : ℝ), x = π / 3 ∧ (∃ k : ℤ, f(x) = sin ( 2 * x - π / 6)) where 
    f : ℝ → ℝ := λ x, sin (2 * x - π / 6) :=
sorry

end symmetry_axis_sin_eq_pi_over_3_l521_521169


namespace smallest_t_is_10_l521_521979

noncomputable def a_sequence : ℕ → ℝ
| 0     := 1
| n + 1 := 1 / Real.sqrt (1 / (a_sequence n)^2 + 4)

noncomputable def S (n : ℕ) : ℝ :=
∑ i in Finset.range (n + 1), (a_sequence i)^2

def smallest_t (n : ℕ) : ℝ := S (2 * n + 1) - S n

theorem smallest_t_is_10 : (∀ (n : ℕ), S (2 * n + 1) - S n ≤ t / 30) → t = 10 :=
begin
  sorry
end

end smallest_t_is_10_l521_521979


namespace simplify_expression_l521_521426

variable (a b : ℝ)
variable (h₁ : a = 3 + Real.sqrt 5)
variable (h₂ : b = 3 - Real.sqrt 5)

theorem simplify_expression : (a^2 - 2 * a * b + b^2) / (a^2 - b^2) * (a * b) / (a - b) = 2 / 3 := by
  sorry

end simplify_expression_l521_521426


namespace decagon_adjacent_probability_l521_521583

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521583


namespace crayons_in_judahs_box_l521_521895

theorem crayons_in_judahs_box (karen_crayons beatrice_crayons gilbert_crayons judah_crayons : ℕ)
  (h1 : karen_crayons = 128)
  (h2 : beatrice_crayons = karen_crayons / 2)
  (h3 : gilbert_crayons = beatrice_crayons / 2)
  (h4 : judah_crayons = gilbert_crayons / 4) :
  judah_crayons = 8 :=
by {
  sorry
}

end crayons_in_judahs_box_l521_521895


namespace expected_value_dodecahedral_die_l521_521082

noncomputable def dodecahedral_expected_value : ℝ :=
  let outcomes := (list.range (12 + 1)).tail -- List of numbers from 1 to 12
  let n_faces := 12
  let probability := 1 / (n_faces : ℝ)
  let expected_value := probability * outcomes.sum / n_faces
  expected_value

theorem expected_value_dodecahedral_die :
  dodecahedral_expected_value = 6.5 :=
by {
  unfold dodecahedral_expected_value,
  simp,
  sorry,
}

end expected_value_dodecahedral_die_l521_521082


namespace large_ducks_sold_l521_521963

theorem large_ducks_sold (n : ℕ) (price_regular : ℕ) (price_large : ℕ) (qty_regular : ℕ) (total_raised : ℕ) :
  price_regular = 3 →
  price_large = 5 →
  qty_regular = 221 →
  total_raised = 1588 →
  663 + 5 * n = total_raised →
  n = 185 :=
by
  intros h_price_regular h_price_large h_qty_regular h_total_raised h_eq.
  sorry

end large_ducks_sold_l521_521963


namespace impossible_to_create_3_piles_l521_521339

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end impossible_to_create_3_piles_l521_521339


namespace maximum_S_n_l521_521199

noncomputable def a_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  a_1 + (n - 1) * d

noncomputable def S_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a_1 + (n - 1) * d)

theorem maximum_S_n (a_1 : ℝ) (h : a_1 > 0)
  (h_sequence : 3 * a_n a_1 (2 * a_1 / 39) 8 = 5 * a_n a_1 (2 * a_1 / 39) 13)
  : ∀ n : ℕ, S_n a_1 (2 * a_1 / 39) n ≤ S_n a_1 (2 * a_1 / 39) 20 :=
sorry

end maximum_S_n_l521_521199


namespace not_possible_to_create_3_piles_l521_521369

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end not_possible_to_create_3_piles_l521_521369


namespace find_B_l521_521887

noncomputable def triangle_angle_B (a b : ℝ) (A : ℝ) : ℝ :=
  let sin_B := (b * Real.sin (A * Real.pi / 180)) / a
  in if sin_B = Real.sin (60 * Real.pi / 180) then 60 else 120

theorem find_B (a b : ℝ) (A B : ℝ) (h_a : a = 4) (h_b : b = 4 * Real.sqrt 3) (h_A : A = 30) :
  B = 60 ∨ B = 120 := by
  simp [h_a, h_b, h_A]
  have h_sinA : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
    simp [Real.sin_pi_div_two_half]
  have h_sinB := (4 * Real.sqrt 3 * (1 / 2)) / 4
  simp [triangle_angle_B, h_sinA, Real.sin_pi_div_three, Real.sqrt_eq_rpow, Real.rpow_div, Real.rpow_one, Complex.pi_ne_zero, Real.sin_eq, h_sinB ]
  have h₁ : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := Real.sin_pi_div_three
  have h₂ : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 := Real.sin_tpi_div_3_eq_pi_div_3
  simp [h₁, h₂]
  exact Or.inl rfl -- due to the integration range on h₁
  sorry -- skip remaining proof steps

end find_B_l521_521887


namespace coefficient_x3_in_pq_l521_521168

-- Define the two polynomials
def p (x : ℝ) := 3 * x^3 + 2 * x^2 + 4 * x + 5
def q (x : ℝ) := 4 * x^2 + 5 * x + 6

-- The main theorem stating the proof problem
theorem coefficient_x3_in_pq : 
  (∑ i in (Finset.range 7), (Nat.choose 6 i) * (3 * (5-i)) + (2 * i) + (4 * (i-2)) * 
    (4 * (6 - i) + (5 * (i-2)) + (6 * (5-i+1)))) = 44:= by sorry

end coefficient_x3_in_pq_l521_521168


namespace minimum_distance_point_parabola_line_l521_521100

-- Definitions of the parabola and the line.
def parabola_y (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 4
def line_y (x : ℝ) : ℝ := 3 * x - 4

-- The minimum distance between a point on the parabola and a point on the line is a value computation
-- which we are given as √(10).
def distance_formula (x1 y1 a b c : ℝ) : ℝ := abs (a * x1 + b * y1 + c) / sqrt (a^2 + b^2)

theorem minimum_distance_point_parabola_line :
  ∃ x y, parabola_y x = y ∧ distance_formula x y 3 (-1) (-4) = 7 * sqrt 10 / 20 :=
by
  -- Using 'sorry' to skip the proof since we only need the statement.
  sorry

end minimum_distance_point_parabola_line_l521_521100


namespace smallest_GCD_value_l521_521007

theorem smallest_GCD_value (a b c d N : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
    (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : N > 5)
    (hc1 : Nat.gcd a b = 1 ∨ Nat.gcd a c = 1 ∨ Nat.gcd a d = 1 ∨ Nat.gcd b c = 1 ∨ Nat.gcd b d = 1 ∨ Nat.gcd c d = 1)
    (hc2 : Nat.gcd a b = 2 ∨ Nat.gcd a c = 2 ∨ Nat.gcd a d = 2 ∨ Nat.gcd b c = 2 ∨ Nat.gcd b d = 2 ∨ Nat.gcd c d = 2)
    (hc3 : Nat.gcd a b = 3 ∨ Nat.gcd a c = 3 ∨ Nat.gcd a d = 3 ∨ Nat.gcd b c = 3 ∨ Nat.gcd b d = 3 ∨ Nat.gcd c d = 3)
    (hc4 : Nat.gcd a b = 4 ∨ Nat.gcd a c = 4 ∨ Nat.gcd a d = 4 ∨ Nat.gcd b c = 4 ∨ Nat.gcd b d = 4 ∨ Nat.gcd c d = 4)
    (hc5 : Nat.gcd a b = 5 ∨ Nat.gcd a c = 5 ∨ Nat.gcd a d = 5 ∨ Nat.gcd b c = 5 ∨ Nat.gcd b d = 5 ∨ Nat.gcd c d = 5)
    (hcN : Nat.gcd a b = N ∨ Nat.gcd a c = N ∨ Nat.gcd a d = N ∨ Nat.gcd b c = N ∨ Nat.gcd b d = N ∨ Nat.gcd c d = N):
    N = 14 :=
sorry

end smallest_GCD_value_l521_521007


namespace expected_value_fair_dodecahedral_die_l521_521073

theorem expected_value_fair_dodecahedral_die : 
  let n := 12 in (1 / n) * (List.sum (List.range n).map (λ x => (x + 1))) = 6.5 :=
by
  let n := 12
  have h1 : (1 : ℚ) / n = 1 / 12 := by sorry
  have h2 : List.range n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] := by sorry
  have h3 : List.sum (List.range n).map (λ x => (x + 1)) = 78 := by sorry
  have h4 : (1 / 12) * 78 = 6.5 := by sorry
  exact h4

end expected_value_fair_dodecahedral_die_l521_521073


namespace div_z_x_l521_521853

variables (x y z : ℚ)

theorem div_z_x (h1 : x / y = 3) (h2 : y / z = 5 / 2) : z / x = 2 / 15 :=
sorry

end div_z_x_l521_521853


namespace residual_for_data_point_l521_521758

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 0.85 * x - 82.71

-- Given data points
def x_val : ℝ := 160
def actual_y : ℝ := 53

-- Calculate the predicted weight for the given x value
noncomputable def predicted_y : ℝ := regression_eq x_val

-- Define the residual as the actual weight minus the predicted weight
noncomputable def residual : ℝ := actual_y - predicted_y

-- Prove that the residual equals -0.29
theorem residual_for_data_point : residual = -0.29 := by
  sorry

end residual_for_data_point_l521_521758


namespace not_possible_to_create_3_piles_l521_521363

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end not_possible_to_create_3_piles_l521_521363


namespace problem_equivalent_l521_521966
-- Import the entirety of Mathlib to ensure all necessary definitions are included.

-- Define the expression to be proved.
def expression := 3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)

-- Define the expected simplified result.
def expected_result := 3 + (2 * Real.sqrt 3) / 3

-- The Lean statement asserting that the expression is equal to the expected result.
theorem problem_equivalent :
  expression = expected_result :=
  sorry

end problem_equivalent_l521_521966


namespace minimum_distance_point_parabola_line_l521_521101

-- Definitions of the parabola and the line.
def parabola_y (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 4
def line_y (x : ℝ) : ℝ := 3 * x - 4

-- The minimum distance between a point on the parabola and a point on the line is a value computation
-- which we are given as √(10).
def distance_formula (x1 y1 a b c : ℝ) : ℝ := abs (a * x1 + b * y1 + c) / sqrt (a^2 + b^2)

theorem minimum_distance_point_parabola_line :
  ∃ x y, parabola_y x = y ∧ distance_formula x y 3 (-1) (-4) = 7 * sqrt 10 / 20 :=
by
  -- Using 'sorry' to skip the proof since we only need the statement.
  sorry

end minimum_distance_point_parabola_line_l521_521101


namespace solve_equations_l521_521956

theorem solve_equations :
  (∀ x : ℝ, 3 * (x - 1)^2 = 27 → (x = 4 ∨ x = -2)) ∧
  (∀ x : ℝ, x^3 / 8 + 2 = 3 → x = 2) :=
by
  -- First equation: 3(x-1)^2 = 27
  split
  · intros x h_eq
    have : (x - 1)^2 = 9 := by
      linarith
    have h1 : x - 1 = 3 ∨ x - 1 = -3 := by
      rw [sq_eq_sq] at this
      exact this
    cases h1 with h1_pos h1_neg
    · left
      linarith
    · right
      linarith
  -- Second equation: x^3 / 8 + 2 = 3
  · intros x h_eq
    have : x^3 / 8 = 1 := by
      linarith
    have : x^3 = 8 := by
      field_simp
      linarith
    have h_x : x = 2 := by
      apply (eq_or_eq_neg_of_sq_eq_sq (by norm_num : (2 : ℝ)^3 = 8) this)
    exact h_x

end solve_equations_l521_521956


namespace seven_students_arrangement_l521_521419

def total_arrangements (A B C D: Type) (students: List (Type)) : Nat :=
  if [A, B] ∈ students ∧ [C, D] ∉ students → 
  2 * Nat.factorial 4 * Nat.comb 5 2 * Nat.factorial 2 
  else 0

theorem seven_students_arrangement : 
  (total_arrangements (_ : Type) (_ : Type) (_ : Type) (_ : Type) ([]) = 960) := 
sorry

end seven_students_arrangement_l521_521419


namespace domain_size_of_f_l521_521035

def f : ℕ → ℕ
| 7       := 22
| n       := if n % 2 = 0 then n / 2 else 3 * n + 2

theorem domain_size_of_f :
  ∃ S : set ℕ, 7 ∈ S ∧ ∀ x ∈ S, f x ∈ S ∧ set.card S = 8 := sorry

end domain_size_of_f_l521_521035


namespace total_cans_l521_521940

def bag1 := 5
def bag2 := 7
def bag3 := 12
def bag4 := 4
def bag5 := 8
def bag6 := 10

theorem total_cans : bag1 + bag2 + bag3 + bag4 + bag5 + bag6 = 46 := by
  sorry

end total_cans_l521_521940


namespace all_numbers_between_n_and_2n_complexity_not_greater_no_numbers_between_n_and_2n_have_less_complexity_l521_521388

noncomputable def complexity (n : ℕ) : ℕ := n.factorization.card

theorem all_numbers_between_n_and_2n_complexity_not_greater (k : ℕ) (h_k : k > 0) :
  ∀ m, 2^k ≤ m ∧ m ≤ 2^(k+1) → complexity m ≤ k :=
by
  sorry

theorem no_numbers_between_n_and_2n_have_less_complexity (n : ℕ) (h_n : n > 1):
  ∃ m, n ≤ m ∧ m ≤ 2 * n ∧ complexity m < complexity n → False :=
by
  sorry

end all_numbers_between_n_and_2n_complexity_not_greater_no_numbers_between_n_and_2n_have_less_complexity_l521_521388


namespace allocation_schemes_count_l521_521031

-- Definitions
def attending_physicians : Nat := 2
def interns : Nat := 4
def locationA_group_size : Nat := 3
def locationB_group_size : Nat := 3

-- Main theorem statement
theorem allocation_schemes_count :
  (number of allocation schemes where intern A is not in location A) = 6 :=
by
  sorry

end allocation_schemes_count_l521_521031


namespace lim_prob_eq_zero_l521_521327

noncomputable def A_star {Ω : Type*} (A : ℕ → set Ω) : set Ω := 
  set.limsup A

noncomputable def A_star_lower {Ω : Type*} (A : ℕ → set Ω) : set Ω := 
  set.liminf A

theorem lim_prob_eq_zero {Ω : Type*} {P : ProbMeasure Ω} (A : ℕ → set Ω) :
  (tendsto (λ n, P (A n \ A_star A)) at_top (nhds 0)) ∧
  (tendsto (λ n, P (A_star_lower A \ (A n))) at_top (nhds 0)) :=
by sorry

end lim_prob_eq_zero_l521_521327


namespace triangle_side_lengths_l521_521839

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end triangle_side_lengths_l521_521839


namespace find_three_consecutive_odd_numbers_l521_521453

noncomputable def three_consecutive_odd_numbers_sum_to_same_digit_four_digit_number (x : ℤ) (y : ℕ) : Prop :=
  y > 0 ∧ y < 10 ∧ 3 * (x * x) + 8 = 1111 * y

theorem find_three_consecutive_odd_numbers :
  ∃ x : ℤ, ∃ y : ℕ, three_consecutive_odd_numbers_sum_to_same_digit_four_digit_number x y → 
  (x = 43 ∨ x = -43) :=
begin
  sorry
end

end find_three_consecutive_odd_numbers_l521_521453


namespace swimmers_meetings_l521_521871

theorem swimmers_meetings :
  let length_pool : ℝ := 100
  let speed_swimmer1 : ℝ := 4
  let speed_swimmer2 : ℝ := 3
  let duration : ℝ := 20 * 60 -- time in seconds
  let lcm_period: ℝ := Real.lcm (length_pool / speed_swimmer1 * 2) (length_pool / speed_swimmer2 * 2)
  let meetings_per_period: ℝ := length_pool / (length_pool / speed_swimmer1 + length_pool / speed_swimmer2)
  let total_meetings: ℝ := (duration / lcm_period) * meetings_per_period
  in total_meetings = 178 :=
by
  trivial

end swimmers_meetings_l521_521871


namespace probability_of_adjacent_vertices_in_decagon_l521_521596

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l521_521596


namespace fifteenth_digit_of_sum_l521_521477

noncomputable def decimal_1_8 : ℚ := 1 / 8
noncomputable def decimal_1_11 : ℚ := 1 / 11
noncomputable def sum_decimals := decimal_1_8 + decimal_1_11

theorem fifteenth_digit_of_sum :
  (decimal_repr 15 sum_decimals) = 0 :=
sorry

end fifteenth_digit_of_sum_l521_521477


namespace sequence_formula_l521_521923

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = 3 * a n + 2^n

theorem sequence_formula (a : ℕ → ℕ) (h : sequence a) :
  ∀ n : ℕ, n > 0 → a n = 3^n - 2^n :=
sorry

end sequence_formula_l521_521923


namespace train_speed_l521_521130

noncomputable def train_length : ℝ := 120
noncomputable def crossing_time : ℝ := 2.699784017278618

theorem train_speed : (train_length / crossing_time) = 44.448 := by
  sorry

end train_speed_l521_521130


namespace fraction_identity_l521_521749

open Real

theorem fraction_identity
  (p q r : ℝ)
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) = 8) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) = 2.2 :=
  sorry

end fraction_identity_l521_521749


namespace largest_possible_c_l521_521913

theorem largest_possible_c (c : ℝ) (hc : (3 * c + 4) * (c - 2) = 9 * c) : c ≤ 4 :=
sorry

end largest_possible_c_l521_521913


namespace green_paint_amount_l521_521136

-- Definitions for the given conditions
def ratio_blue_green_white := (1, 2, 5)
def total_paint := 24
def total_parts := 1 + 2 + 5
def gallons_per_part := total_paint / total_parts
def parts_green := 2
def green_paint := parts_green * gallons_per_part

-- Theorem to prove the problem statement
theorem green_paint_amount : green_paint = 6 := by
  unfold ratio_blue_green_white total_paint total_parts gallons_per_part parts_green green_paint
  sorry

end green_paint_amount_l521_521136


namespace white_marble_price_l521_521134

theorem white_marble_price
  (total_marbles : ℕ) (white_percent : ℝ) (black_percent : ℝ)
  (black_price : ℝ) (colored_price : ℝ) (total_earnings : ℝ)
  (white_marbles : ℕ) (black_marbles : ℕ) (colored_marbles : ℕ)
  (earnings_black : ℝ) (earnings_colored : ℝ) (earnings_white : ℝ)
  (price_white : ℝ) :
  total_marbles = 100 ∧ white_percent = 0.20 ∧ black_percent = 0.30 ∧
  black_price = 0.10 ∧ colored_price = 0.20 ∧ total_earnings = 14 ∧ 
  white_marbles = (white_percent * total_marbles).to_nat ∧ 
  black_marbles = (black_percent * total_marbles).to_nat ∧ 
  colored_marbles = total_marbles - (white_marbles + black_marbles) ∧ 
  earnings_black = black_marbles * black_price ∧ 
  earnings_colored = colored_marbles * colored_price ∧ 
  earnings_white = total_earnings - (earnings_black + earnings_colored) ∧ 
  price_white = earnings_white / white_marbles 
  → price_white = 0.05 := by sorry

end white_marble_price_l521_521134


namespace partition_into_rectangles_l521_521503

-- Define the problem conditions
def markedCells (n : ℕ) : set (ℕ × ℕ) := sorry -- Set of 2n marked cells on the chessboard
def rookCanMoveThroughAll (cells : set (ℕ × ℕ)) : Prop := sorry -- Rook can move through all the marked cells without jumping over any unmarked cells

-- Define the theorem
theorem partition_into_rectangles (n : ℕ) (h1 : markedCells n) (h2 : rookCanMoveThroughAll (markedCells n)) : ∃ (rectangles : list (set (ℕ × ℕ))), rectangles.length = n ∧ (⋃ r ∈ rectangles, r) = markedCells n ∧ (∀ r ∈ rectangles, is_rectangle r) :=
sorry

end partition_into_rectangles_l521_521503


namespace impossible_to_form_three_similar_piles_l521_521349

-- Define the concept of similar sizes
def similar_sizes (a b : ℝ) : Prop := a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2

-- State the proof problem
theorem impossible_to_form_three_similar_piles (x : ℝ) (h : 0 < x) : ¬(∃ a b c : ℝ, a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes a c) :=
sorry

end impossible_to_form_three_similar_piles_l521_521349


namespace find_n_l521_521182

theorem find_n 
  (f : ℕ → ℝ)
  (a a_1 a_2 : ℕ → ℝ)
  (n : ℕ)
  (h1 : ∀ x, (1+x) + (1+x)^2 + ... + (1+x)^n = f(x))
  (h2 : a_1(1) + a_2(2) + ... + a(n-1)(n-1) + a(n)(n) = 510 - n) :
  n = 8 := by
  sorry

end find_n_l521_521182


namespace impossible_to_create_3_piles_l521_521372

def similar_sizes (x y : ℝ) : Prop := x ≤ sqrt 2 * y

theorem impossible_to_create_3_piles (x y : ℝ) (h₁ : similar_sizes x y) :
  ¬ ∃ (x₁ x₂ x₃ : ℝ), x₁ + x₂ + x₃ = x + y ∧ similar_sizes x₁ x₂ ∧ similar_sizes x₂ x₃ ∧ similar_sizes x₁ x₃ :=
by
  sorry

end impossible_to_create_3_piles_l521_521372


namespace minimum_voters_needed_l521_521270

-- conditions
def num_voters := 135
def num_districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority_precincts (n : ℕ) := (n + 1) / 2

-- definitions for quantities derived from conditions
def total_precincts := num_districts * precincts_per_district
def majority_districts := majority_precincts num_districts
def precincts_needed_for_district_win := majority_precincts precincts_per_district
def total_precincts_needed_for_win := majority_districts * precincts_needed_for_district_win
def votes_needed_per_precinct := majority_precincts voters_per_precinct

-- main statement
theorem minimum_voters_needed : (votes_needed_per_precinct * total_precincts_needed_for_win = 30) ∧ TallGiraffeWon :=
by sorry

end minimum_voters_needed_l521_521270


namespace parabola_equation_line_tangent_to_fixed_circle_l521_521193

open Real

def parabola_vertex_origin_directrix (p : ℝ) : Prop :=
  ∀ x y : ℝ, y^2 = 2 * p * x ↔ x = -2

def point_on_directrix (l: ℝ) (t : ℝ) : Prop :=
  t ≠ 0 ∧ l = 3 * t - 1 / t

def point_on_y_axis (q : ℝ) (t : ℝ) : Prop :=
  q = 2 * t

theorem parabola_equation (p : ℝ) : 
  parabola_vertex_origin_directrix 4 →
  y^2 = 8 * x :=
by
  sorry

theorem line_tangent_to_fixed_circle (t : ℝ) (x0 : ℝ) (r : ℝ) :
  t ≠ 0 →
  point_on_directrix (-2) t →
  point_on_y_axis (2 * t) t →
  (x0 = 2 ∧ r = 2) →
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 :=
by
  sorry

end parabola_equation_line_tangent_to_fixed_circle_l521_521193


namespace num_possible_triangle_sides_l521_521805

theorem num_possible_triangle_sides (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (a_eq_8 : a = 8) (b_eq_5 : b = 5) : 
  let x := {n : ℕ | 3 < n ∧ n < 13} in
  set.card x = 9 :=
by
  sorry

end num_possible_triangle_sides_l521_521805


namespace range_of_a_l521_521228

def f (x : ℝ) : ℝ :=
if x < 1 then -1 else x - 2

theorem range_of_a (a : ℝ) :
  f (5 * a - 2) > f (2 * a ^ 2) ↔ (3 / 5) < a ∧ a < 2 :=
by
  sorry

end range_of_a_l521_521228


namespace marys_mother_paid_correct_total_l521_521935

def mary_and_friends_payment_per_person : ℕ := 1 -- $1 each
def number_of_people : ℕ := 3 -- Mary and two friends

def total_chicken_cost : ℕ := mary_and_friends_payment_per_person * number_of_people -- Total cost of the chicken

def beef_cost_per_pound : ℕ := 4 -- $4 per pound
def total_beef_pounds : ℕ := 3 -- 3 pounds of beef
def total_beef_cost : ℕ := beef_cost_per_pound * total_beef_pounds -- Total cost of the beef

def oil_cost : ℕ := 1 -- $1 for 1 liter of oil

def total_grocery_cost : ℕ := total_chicken_cost + total_beef_cost + oil_cost -- Total grocery cost

theorem marys_mother_paid_correct_total : total_grocery_cost = 16 := by
  -- Here you would normally provide the proof steps which we're skipping per instructions.
  sorry

end marys_mother_paid_correct_total_l521_521935


namespace decagon_adjacent_probability_l521_521535

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l521_521535


namespace expected_value_of_fair_dodecahedral_die_l521_521051

theorem expected_value_of_fair_dodecahedral_die : 
  (1/12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_of_fair_dodecahedral_die_l521_521051


namespace ratio_of_fuji_trees_l521_521112

variable (F T : ℕ) -- Declaring F as number of pure Fuji trees, T as total number of trees
variables (C : ℕ) -- Declaring C as number of cross-pollinated trees 

theorem ratio_of_fuji_trees 
  (h1: 10 * C = T) 
  (h2: F + C = 221) 
  (h3: T = F + 39 + C) : 
  F * 52 = 39 * T := 
sorry

end ratio_of_fuji_trees_l521_521112


namespace area_triangle_parabola_l521_521900

noncomputable def area_of_triangle_ABC (d : ℝ) (x : ℝ) : ℝ :=
  let A := (x, x^2)
  let B := (x + d, (x + d)^2)
  let C := (x + 2 * d, (x + 2 * d)^2)
  1 / 2 * abs (x * ((x + 2 * d)^2 - (x + d)^2) + (x + d) * ((x + 2 * d)^2 - x^2) + (x + 2 * d) * (x^2 - (x + d)^2))

theorem area_triangle_parabola (d : ℝ) (h_d : 0 < d) (x : ℝ) : 
  area_of_triangle_ABC d x = d^2 := sorry

end area_triangle_parabola_l521_521900


namespace probability_of_adjacent_vertices_in_decagon_l521_521595

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l521_521595


namespace triangle_side_lengths_count_l521_521795

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end triangle_side_lengths_count_l521_521795


namespace tanker_fill_rate_l521_521650

theorem tanker_fill_rate :
  let barrels_per_min := 2
  let liters_per_barrel := 159
  let cubic_meters_per_liter := 0.001
  let minutes_per_hour := 60
  let liters_per_min := barrels_per_min * liters_per_barrel
  let liters_per_hour := liters_per_min * minutes_per_hour
  let cubic_meters_per_hour := liters_per_hour * cubic_meters_per_liter
  cubic_meters_per_hour = 19.08 :=
  by {
    sorry
  }

end tanker_fill_rate_l521_521650


namespace sum_of_selected_numbers_l521_521667

theorem sum_of_selected_numbers (n : ℕ) : 
  let a := λ (i j : ℕ), n * (i - 1) + j
  let b := λ (i : ℕ), i - 1
  let c := λ (j : ℕ), j
  ∑ i in finset.range n, ∑ j in finset.range n, (if i = j then n * b (i + 1) + c (j + 1) else 0) = n * (n^2 + 1) / 2 := 
sorry

end sum_of_selected_numbers_l521_521667


namespace goldbach_conjecture_largest_difference_l521_521452

theorem goldbach_conjecture_largest_difference :
  ∃ (p q : ℕ), p ≠ q ∧ p < q ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (p + q = 156) ∧ ((q - p) = 146) :=
begin
  sorry
end

end goldbach_conjecture_largest_difference_l521_521452


namespace probability_adjacent_vertices_of_decagon_l521_521511

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l521_521511


namespace area_square_abcd_l521_521268

/-- 
In the coordinate plane, let B be at (0,0), C at (2,0), and E at (2,1). 
Determine the area of square ABCD if side BC is part of the square and point E 
lies on a line that also intersects another vertex of the square.
-/
theorem area_square_abcd
  (B C E A D : ℝ × ℝ)
  (hB : B = (0, 0))
  (hC : C = (2, 0))
  (hE : E = (2, 1))
  (square_ABCD : B ≠ C ∧ C ≠ A ∧ A ≠ D ∧ D ≠ B ∧ B = (0, 0) ∧ C = (2, 0) ∧ (∃ s : ℝ, s = real.dist B C ∧ ∀ (P : ℝ × ℝ), P = proj s C E ∧ P ∈ set.range (λ x, x + s * C))) :
  let side_length := real.dist B C in
  side_length ^ 2 = 4 :=
by 
  sorry

end area_square_abcd_l521_521268


namespace calculate_force_l521_521142

noncomputable def force_on_dam (ρ g a b h : ℝ) : ℝ :=
  ρ * g * h^2 * (b / 2 - (b - a) / 3)

theorem calculate_force : force_on_dam 1000 10 4.8 7.2 3.0 = 252000 := 
  by 
  sorry

end calculate_force_l521_521142


namespace probability_adjacent_vertices_decagon_l521_521603

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l521_521603


namespace div_z_x_l521_521854

variables (x y z : ℚ)

theorem div_z_x (h1 : x / y = 3) (h2 : y / z = 5 / 2) : z / x = 2 / 15 :=
sorry

end div_z_x_l521_521854


namespace expand_expression_l521_521708

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 * y + 6) = 36 * x + 48 * y + 72 :=
  sorry

end expand_expression_l521_521708


namespace scale_of_diagram_l521_521635

-- Definitions for the given conditions
def length_miniature_component_mm : ℕ := 4
def length_diagram_cm : ℕ := 8
def length_diagram_mm : ℕ := 80  -- Converted length from cm to mm

-- The problem statement
theorem scale_of_diagram :
  (length_diagram_mm : ℕ) / (length_miniature_component_mm : ℕ) = 20 :=
by
  have conversion : length_diagram_mm = length_diagram_cm * 10 := by sorry
  -- conversion states the formula for converting cm to mm
  have ratio : length_diagram_mm / length_miniature_component_mm = 80 / 4 := by sorry
  -- ratio states the initial computed ratio
  exact sorry

end scale_of_diagram_l521_521635


namespace pure_alcohol_added_l521_521613

theorem pure_alcohol_added (x : ℝ) (h1 : 6 * 0.40 = 2.4)
    (h2 : (2.4 + x) / (6 + x) = 0.50) : x = 1.2 :=
by
  sorry

end pure_alcohol_added_l521_521613


namespace question_equals_answer_l521_521703

def heartsuit (a b : ℤ) : ℤ := |a + b|

theorem question_equals_answer : heartsuit (-3) (heartsuit 5 (-8)) = 0 := 
by
  sorry

end question_equals_answer_l521_521703


namespace min_distance_sum_l521_521200

-- Define the parabola equation: y^2 = 4x
def parabola_p (x y : ℝ) := y^2 = 4 * x

-- Define the line equations: l1: 4x - 3y + 6 = 0 and l2: x + 1 = 0
def line_l1 (x y : ℝ) := 4 * x - 3 * y + 6 = 0
def line_l2 (x : ℝ) := x + 1 = 0

-- Define the focus of the parabola
def focus_F := (1 : ℝ, 0 : ℝ)

-- Prove: the minimum sum of distances from any point P (on the parabola) to the lines l1 and l2 is 2
theorem min_distance_sum (P : ℝ × ℝ) (hP : parabola_p P.1 P.2) :
  ∃ P : ℝ × ℝ, parabola_p P.1 P.2 ∧
  (distance P ⟨P.1, focus_F.2⟩ = abs (4 * focus_F.1 - 3 * focus_F.2 + 6) / sqrt(16 + 9)) ∧
  (distance (⟨P.1, focus_F.2⟩ : ℝ × ℝ) ⟨1, 0⟩ + distance P ⟨-1, P.2⟩ = 2) :=
sorry

end min_distance_sum_l521_521200


namespace students_camping_trip_percentage_l521_521247

theorem students_camping_trip_percentage (total_students : ℕ) (students_who_took_more_than_100 : ℕ)
                                          (students_at_trip_percentage_who_took_more_than_100 : ℚ)
                                          (students_at_trip_percentage_who_did_not_take_more_than_100 : ℚ) :
  students_at_trip_percentage_who_did_not_take_more_than_100 = 0.75 →
  students_at_trip_percentage_who_took_more_than_100 = 0.25 →
  students_who_took_more_than_100 = 22 →
  (22 / 0.25 / total_students) = 0.88 := 
by
  sorry

end students_camping_trip_percentage_l521_521247


namespace find_s_l521_521916

variables {f g : ℝ → ℝ} {s : ℝ}
hypothesis hf : f = λ x, (x - (s + 2)) * (x - (s + 5)) * (x - c)
hypothesis hg : g = λ x, (x - (s + 4)) * (x - (s + 8)) * (x - d)
hypothesis hfg : ∀ x : ℝ, f x - g x = 2 * s

theorem find_s : s = 3.6 :=
sorry

end find_s_l521_521916


namespace not_possible_to_create_3_piles_l521_521360

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end not_possible_to_create_3_piles_l521_521360


namespace ceil_property_2_ceil_property_3_l521_521702

def ceil_function (x : ℝ) : ℤ := ⌈x⌉

theorem ceil_property_2 (x₁ x₂ : ℝ) (hx : ceil_function x₁ = ceil_function x₂) : x₁ - x₂ < 1 :=
sorry

theorem ceil_property_3 (x₁ x₂ : ℝ) : ceil_function (x₁ + x₂) ≤ ceil_function x₁ + ceil_function x₂ :=
sorry

end ceil_property_2_ceil_property_3_l521_521702


namespace intersection_complement_B_A_eq_range_of_a_for_subset_l521_521230

-- Given definitions
def U := set ℝ
def A (a : ℝ) := {x : ℝ | (x - 2) / (x - (3 * a + 1)) < 0}
def B (a : ℝ) := {x : ℝ | (x - a^2 - 2) / (x - a) < 0}

-- The first proof problem
theorem intersection_complement_B_A_eq {a : ℝ} (h : a = 1 / 2) :
  (U \ (B a)) ∩ (A a) = {x : ℝ | 9 / 4 ≤ x ∧ x < 5 / 2} := 
by sorry

-- The second proof problem
theorem range_of_a_for_subset :
  {a : ℝ | ∀ x, x ∈ A a → x ∈ B a} = {a : ℝ | -1 / 2 ≤ a ∧ a ≤ (3 - real.sqrt 5) / 2} := 
by sorry

end intersection_complement_B_A_eq_range_of_a_for_subset_l521_521230


namespace impossible_to_create_3_piles_l521_521333

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end impossible_to_create_3_piles_l521_521333


namespace min_distance_l521_521771

theorem min_distance (a b c d : ℝ) (h1 : ln b + 1 + a - 3 * b = 0) 
                     (h2 : 2 * d - c + Real.sqrt 5 = 0) : 
                     (a - c)^2 + (b - d)^2 = 4 / 5 :=
by sorry

end min_distance_l521_521771


namespace not_possible_three_piles_l521_521380

def similar (x y : ℝ) : Prop :=
  y ≤ (real.sqrt 2) * x ∧ x ≤ (real.sqrt 2) * y

theorem not_possible_three_piles (N : ℝ) : 
  ¬ ∃ a b c : ℝ, a + b + c = N ∧ similar a b ∧ similar b c ∧ similar a c := 
by
  sorry

end not_possible_three_piles_l521_521380


namespace number_of_sides_possibilities_l521_521814

theorem number_of_sides_possibilities (a b : ℕ) (h₁ : a = 8) (h₂ : b = 5) : 
  { x : ℕ // 3 < x ∧ x < 13 }.to_list.length = 9 :=
by
  sorry

end number_of_sides_possibilities_l521_521814


namespace hexagon_inequality_l521_521881

variable {Point : Type*} [MetricSpace Point]

-- Define points A1, A2, A3, A4, A5, A6 in a Metric Space
variables (A1 A2 A3 A4 A5 A6 O : Point)

-- Conditions
def angle_condition (O A1 A2 A3 A4 A5 A6 : Point) : Prop :=
  -- Points form a hexagon where each side is visible from O at 60 degrees
  -- We assume MetricSpace has a function measuring angles such as angle O x y = 60
  true -- A simplified condition; the actual angle measurement needs more geometry setup

def distance_condition_odd (O A1 A3 A5 : Point) : Prop := dist O A1 > dist O A3 ∧ dist O A3 > dist O A5
def distance_condition_even (O A2 A4 A6 : Point) : Prop := dist O A2 > dist O A4 ∧ dist O A4 > dist O A6

-- Question to prove
theorem hexagon_inequality 
  (hc : angle_condition O A1 A2 A3 A4 A5 A6) 
  (ho : distance_condition_odd O A1 A3 A5)
  (he : distance_condition_even O A2 A4 A6) : 
  dist A1 A2 + dist A3 A4 + dist A5 A6 < dist A2 A3 + dist A4 A5 + dist A6 A1 := 
sorry

end hexagon_inequality_l521_521881


namespace sum_of_first_40_digits_of_1_over_1111_equals_90_l521_521489

theorem sum_of_first_40_digits_of_1_over_1111_equals_90 :
  ∀ decimal_rep : ℕ → ℕ,
  (decimal_rep 1 = 0) ∧ (decimal_rep 2 = 0) ∧ (decimal_rep 3 = 0) ∧ (decimal_rep 4 = 9)
  → (decimal_rep 5 = decimal_rep 1) ∧ (decimal_rep 6 = decimal_rep 2) ∧ (decimal_rep 7 = decimal_rep 3) ∧ (decimal_rep 8 = decimal_rep 4)
  → (sum (List.range 40).map decimal_rep) = 90 :=
by
  sorry

end sum_of_first_40_digits_of_1_over_1111_equals_90_l521_521489


namespace set_union_inter_proof_l521_521506

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem set_union_inter_proof : A ∪ B = {0, 1, 2, 3} ∧ A ∩ B = {1, 2} := by
  sorry

end set_union_inter_proof_l521_521506


namespace integral_limit_equivalence_l521_521697

noncomputable theory

open Set Filter

theorem integral_limit_equivalence (f : ℝ → ℝ) (g : ℝ → ℝ) (L : ℝ) :
  (∀ x, 0 ≤ x → ContinuousAt f x) →
  ContinuousOn g (Icc 0 1) →
  Tendsto f (atTop) (𝓝 L) →
  (tendsto (λ n:ℕ, (1 / (n:ℝ)) * ∫ x in (0:ℝ)..(n:ℝ), f x * g (x / (n:ℝ))) atTop (𝓝 ((∫ x in (0:ℝ)..(1:ℝ), g x) * L))) :=
by
  sorry

end integral_limit_equivalence_l521_521697


namespace power_division_identity_l521_521476

theorem power_division_identity : (2^12 / 8^3) = 8 := by
  have h1 : 8 = 2^3 := rfl
  rw [h1, pow_succ, mul_assoc]
  sorry

end power_division_identity_l521_521476


namespace decagon_adjacent_probability_l521_521527

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l521_521527


namespace circle_eq_line_tangent_to_circle_l521_521267

def chord_intersect_circle_length := sqrt 6
def ellipse_eq := (2 * x^2 + y^2 = 4)
def line_eq := (x = 2)
def dist_center_to_line := (1 / sqrt 2)
def intercept_chord_length := sqrt 6

theorem circle_eq {x y : ℝ} (r : ℝ) :
  (x^2 + y^2 = r^2) ∧ (r^2 = dist_center_to_line^2 + (intercept_chord_length / 2)^2) →
  (x^2 + y^2 = 2) := by
  sorry

theorem line_tangent_to_circle {x y : ℝ}
  (x0 t y0 : ℝ) (hA : ellipse_eq x0 y0) (hB : line_eq t)
  (h_perp : x0 * t + 2 * y0 = 0) :
    ((x - t) * y = (y0 - 2) * (x - t)) → 
    (distance (0, 0) (line_eq t)) = sqrt 2 → 
    tangent_to_circle x y :=
  by
  sorry

end circle_eq_line_tangent_to_circle_l521_521267


namespace not_possible_three_piles_l521_521383

def similar (x y : ℝ) : Prop :=
  y ≤ (real.sqrt 2) * x ∧ x ≤ (real.sqrt 2) * y

theorem not_possible_three_piles (N : ℝ) : 
  ¬ ∃ a b c : ℝ, a + b + c = N ∧ similar a b ∧ similar b c ∧ similar a c := 
by
  sorry

end not_possible_three_piles_l521_521383


namespace num_possible_triangle_sides_l521_521810

theorem num_possible_triangle_sides (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (a_eq_8 : a = 8) (b_eq_5 : b = 5) : 
  let x := {n : ℕ | 3 < n ∧ n < 13} in
  set.card x = 9 :=
by
  sorry

end num_possible_triangle_sides_l521_521810


namespace smallest_positive_number_is_A_l521_521705

def is_smallest_positive_number (n : ℝ) (a b c d e : ℝ) : Prop :=
  (n > 0) ∧ (n = min (min (12 - 4*Real.sqrt 8) (min (4*Real.sqrt 8 - 12) (min (20 - 6*Real.sqrt 10) (min (60 - 12*Real.sqrt 30) (12*Real.sqrt 30 - 60))))))
  ∧ n ∈ {12 - 4*Real.sqrt 8, 4*Real.sqrt 8 - 12, 20 - 6*Real.sqrt 10, 60 - 12*Real.sqrt 30, 12*Real.sqrt 30 - 60}

theorem smallest_positive_number_is_A :
  is_smallest_positive_number (12 - 4*Real.sqrt 8)
    (12 - 4*Real.sqrt 8)
    (4*Real.sqrt 8 - 12)
    (20 - 6*Real.sqrt 10)
    (60 - 12*Real.sqrt 30)
    (12*Real.sqrt 30 - 60) := sorry

end smallest_positive_number_is_A_l521_521705


namespace decagon_adjacent_probability_l521_521529

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l521_521529


namespace sin_B_of_arithmetic_sequence_angles_l521_521218

theorem sin_B_of_arithmetic_sequence_angles (A B C : ℝ) (h1 : A + C = 2 * B) (h2 : A + B + C = Real.pi) :
  Real.sin B = Real.sqrt 3 / 2 :=
sorry

end sin_B_of_arithmetic_sequence_angles_l521_521218


namespace decagon_adjacent_probability_l521_521543

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l521_521543


namespace minimum_voters_needed_l521_521272

-- conditions
def num_voters := 135
def num_districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority_precincts (n : ℕ) := (n + 1) / 2

-- definitions for quantities derived from conditions
def total_precincts := num_districts * precincts_per_district
def majority_districts := majority_precincts num_districts
def precincts_needed_for_district_win := majority_precincts precincts_per_district
def total_precincts_needed_for_win := majority_districts * precincts_needed_for_district_win
def votes_needed_per_precinct := majority_precincts voters_per_precinct

-- main statement
theorem minimum_voters_needed : (votes_needed_per_precinct * total_precincts_needed_for_win = 30) ∧ TallGiraffeWon :=
by sorry

end minimum_voters_needed_l521_521272


namespace work_required_to_lift_sphere_out_of_water_l521_521126

theorem work_required_to_lift_sphere_out_of_water
  (H : ℝ) (R : ℝ) (δ : ℝ) (π : ℝ)
  (h_given : H = 14)
  (r_given : R = 3)
  (δ_given : δ = 2)
  (pi_def : π = Real.pi) :
  let Q := (4 / 3) * π * R^3 * ((δ - 1) * H + R * (2 * δ - 1))
  in Q = (4 / 3) * π * 3^3 * ((2 - 1) * 14 + 3 * (2 * 2 - 1)) :=
by
  have h : H = 14 := h_given
  have r : R = 3 := r_given
  have δ : δ = 2 := δ_given
  have π := pi_def
  sorry

end work_required_to_lift_sphere_out_of_water_l521_521126


namespace H2O_required_for_NaH_reaction_l521_521166

theorem H2O_required_for_NaH_reaction
  (n_NaH : ℕ) (n_H2O : ℕ) (n_NaOH : ℕ) (n_H2 : ℕ)
  (h_eq : n_NaH = 2) (balanced_eq : n_NaH = n_H2O ∧ n_H2O = n_NaOH ∧ n_NaOH = n_H2) :
  n_H2O = 2 :=
by
  -- The proof is omitted as we only need to declare the statement.
  sorry

end H2O_required_for_NaH_reaction_l521_521166


namespace probability_adjacent_vertices_decagon_l521_521606

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l521_521606


namespace prime_p_impplies_p_eq_3_l521_521249

theorem prime_p_impplies_p_eq_3 (p : ℕ) (hp : Prime p) (hp2 : Prime (p^2 + 2)) : p = 3 :=
sorry

end prime_p_impplies_p_eq_3_l521_521249


namespace sin_squared_alpha_plus_pi_over_4_l521_521183

theorem sin_squared_alpha_plus_pi_over_4 (α : ℝ) (h₀ : 0 < α ∧ α < π / 4) (h₁ : cos (2 * α) = 4 / 5) :
  sin^2 (α + π / 4) = 4 / 5 := 
sorry

end sin_squared_alpha_plus_pi_over_4_l521_521183


namespace probability_adjacent_vertices_of_decagon_l521_521516

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l521_521516


namespace domain_and_range_of_g_l521_521919

noncomputable def f : ℝ → ℝ := sorry-- Given: a function f with domain [0,2] and range [0,1]
noncomputable def g (x : ℝ) := 1 - f (x / 2 + 1)

theorem domain_and_range_of_g :
  let dom_g := { x | -2 ≤ x ∧ x ≤ 2 }
  let range_g := { y | 0 ≤ y ∧ y ≤ 1 }
  ∀ (x : ℝ), (x ∈ dom_g → (g x) ∈ range_g) := 
sorry

end domain_and_range_of_g_l521_521919


namespace train_crossing_tree_time_l521_521611

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

end train_crossing_tree_time_l521_521611


namespace john_final_push_time_l521_521295

theorem john_final_push_time :
  ∃ t : ℝ, (∀ (d_j d_s : ℝ), d_j = 4.2 * t ∧ d_s = 3.7 * t ∧ (d_j = d_s + 14)) → t = 28 :=
by
  sorry

end john_final_push_time_l521_521295


namespace triangle_side_count_l521_521832

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end triangle_side_count_l521_521832


namespace christopher_strolled_5_miles_l521_521687

theorem christopher_strolled_5_miles (s t : ℝ) (hs : s = 4) (ht : t = 1.25) : s * t = 5 :=
by
  rw [hs, ht]
  norm_num

end christopher_strolled_5_miles_l521_521687


namespace smallest_six_digit_divisor_100011_l521_521172

theorem smallest_six_digit_divisor_100011 :
  ∃ n, (n > 0) ∧ n.divides 100011 ∧ 3.divides n :=
sorry

end smallest_six_digit_divisor_100011_l521_521172


namespace decagon_adjacent_probability_l521_521540

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l521_521540


namespace expected_value_dodecahedral_die_is_6_5_l521_521064

noncomputable def expected_value_dodecahedral_die : ℝ :=
  let outcomes := finset.range 12 in
  1 / 12 * (finset.sum outcomes (λ i, (i + 1 : ℝ)))

theorem expected_value_dodecahedral_die_is_6_5 :
  expected_value_dodecahedral_die = 6.5 :=
by
  sorry

end expected_value_dodecahedral_die_is_6_5_l521_521064


namespace green_paint_blue_percentage_is_correct_l521_521622

-- Given conditions
def maroon_paint_blue_percentage : ℝ := 50
def maroon_paint_red_percentage : ℝ := 50
def green_paint_yellow_percentage : ℝ := 70
def brown_paint_blue_percentage : ℝ := 40
def brown_paint_weight : ℝ := 10
def red_pigment_weight : ℝ := 2.5

-- Definitions based on conditions
def maroon_paint_weight_from_red : ℝ := (red_pigment_weight / (maroon_paint_red_percentage / 100))
def maroon_paint_blue_weight : ℝ := (maroon_paint_weight_from_red * (maroon_paint_blue_percentage / 100))
def total_blue_weight_in_brown_paint : ℝ := (brown_paint_blue_percentage / 100) * brown_paint_weight
def green_paint_blue_weight : ℝ := total_blue_weight_in_brown_paint - maroon_paint_blue_weight
def green_paint_weight : ℝ := brown_paint_weight - maroon_paint_weight_from_red
def green_paint_blue_percentage : ℝ := (green_paint_blue_weight / green_paint_weight) * 100

-- Proof statement
theorem green_paint_blue_percentage_is_correct : green_paint_blue_percentage = 30 := by
  sorry

end green_paint_blue_percentage_is_correct_l521_521622


namespace find_k_l521_521777

def a : ℝ × ℝ × ℝ := (1, 1, 0) -- define vector a
def b : ℝ × ℝ × ℝ := (-1, 0, 1) -- define vector b

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_k (k : ℝ) (h : dot_product (k * a.1, k * a.2, k * a.3) b = 0) : 
  k = 1 / 2 :=
  sorry -- proof not provided

end find_k_l521_521777


namespace solve_operation_l521_521245

def operation (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

theorem solve_operation :
  operation (operation (operation 2 3) 4) 5 = 7 / 8 :=
by
  sorry

end solve_operation_l521_521245


namespace medians_squared_sum_eq_four_ninths_edges_squared_sum_l521_521944

/-- Prove that the sum of the squares of the lengths of the medians of a tetrahedron is equal to $4 / 9$ of the sum of the squares of the lengths of its edges. -/
theorem medians_squared_sum_eq_four_ninths_edges_squared_sum
  (A B C D : Point)
  (mA mB mC mD : ℝ)
  (sum_medians_squared_eq_three_fourths_edges_squared : 
    mA^2 + mB^2 + mC^2 + mD^2 = (3/4) * (∑ (AB^2 + AC^2 + AD^2 + BC^2 + BD^2 + CD^2))) :
  (mA^2 + mB^2 + mC^2 + mD^2) = (4/9) * (∑ (AB^2 + AC^2 + AD^2 + BC^2 + BD^2 + CD^2)) :=
by
  sorry

end medians_squared_sum_eq_four_ninths_edges_squared_sum_l521_521944


namespace smallest_positive_period_f_minimum_value_f_range_g_l521_521224

def f (x : ℝ) := (1/2) * Real.sin (2 * x) - Real.sqrt 3 * (Real.cos x)^2
def g (x : ℝ) := Real.sin (x - π/3) - Real.sqrt 3 / 2

-- To prove that the smallest positive period of f(x) is π
theorem smallest_positive_period_f : ∃ (T : ℝ), T > 0 ∧ ∀ x, f (x) = f (x + T) ∧ T = π := sorry

-- To prove that the minimum value of f(x) is - (2 + sqrt 3) / 2
theorem minimum_value_f : ∃ (m : ℝ), ∀ x, f x ≥ m ∧ m = -((2 + Real.sqrt 3) / 2) := sorry

-- To prove that the range of g(x) when x ∈ [π/2, π] is [(1 - sqrt 3) / 2, (2 - sqrt 3) / 2]
theorem range_g : Set.range (λ x, g x) = Set.Icc ((1 - Real.sqrt 3) / 2) ((2 - Real.sqrt 3) / 2) := sorry

end smallest_positive_period_f_minimum_value_f_range_g_l521_521224


namespace compare_abc_l521_521729

noncomputable def a : ℝ := 2^(4/3)
noncomputable def b : ℝ := 4^(2/5)
noncomputable def c : ℝ := 5^(2/3)

theorem compare_abc : c > a ∧ a > b := 
by
  sorry

end compare_abc_l521_521729


namespace gavrila_ascent_time_l521_521472
noncomputable def gavrila_time (V U t : ℝ) : ℝ := t

theorem gavrila_ascent_time (V U : ℝ) :
  (1 = V * 60) →
  (1 = (V + U) * 24) →
  (t = 40 → 1 = U * t) :=
by
  intros h1 h2 h3
  -- Using the given equations:
  -- 1 = V * 60
  -- 1 = (V + U) * 24
  -- Solve for V and substitute to find U
  have hV : V = 1 / 60 := by sorry
  have hU : U = 1 / 40 := by sorry
  rw [h3, hU]
  exact rfl

end gavrila_ascent_time_l521_521472


namespace triangle_possible_sides_l521_521784

theorem triangle_possible_sides : 
  let unknown_side_lengths := {x : ℤ | 3 < x ∧ x < 13}
  in set.finite unknown_side_lengths ∧ set.card unknown_side_lengths = 9 :=
by
  sorry

end triangle_possible_sides_l521_521784


namespace n_minus_k_minus_l_square_number_l521_521036

variable (n k l x : ℕ)

theorem n_minus_k_minus_l_square_number (h1 : x^2 < n)
                                        (h2 : n < (x + 1)^2)
                                        (h3 : n - k = x^2)
                                        (h4 : n + l = (x + 1)^2) :
  ∃ m : ℕ, n - k - l = m ^ 2 :=
by
  sorry

end n_minus_k_minus_l_square_number_l521_521036


namespace triangle_side_count_l521_521831

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end triangle_side_count_l521_521831


namespace initial_savings_l521_521302

-- Define the necessary entities based on the conditions
variables (cost_sneakers : ℕ) 
          (price_per_figure : ℕ)
          (num_figures_sold : ℕ)
          (amount_left_over : ℕ)

-- Set the conditions
def conditions := cost_sneakers = 90 ∧ price_per_figure = 10 ∧ num_figures_sold = 10 ∧ amount_left_over = 25

-- Define the required proof goal
theorem initial_savings (h : conditions) : 
  let money_from_figures := num_figures_sold * price_per_figure,
      total_money_before := cost_sneakers + amount_left_over,
      initial_savings := total_money_before - money_from_figures in
  initial_savings = 15 :=
by 
  -- Here, we're skipping the proof itself
  sorry

end initial_savings_l521_521302


namespace expected_value_fair_dodecahedral_die_l521_521078

theorem expected_value_fair_dodecahedral_die : 
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  let probability := (1 : ℝ) / 12 in
  (probability * faces.sum) = 6.5 :=
by
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let probability := (1 : ℝ) / 12
  have h : faces.sum = 78 := rfl
  rw [h]
  sorry

end expected_value_fair_dodecahedral_die_l521_521078


namespace rainfall_second_week_l521_521494

theorem rainfall_second_week (total_rainfall : ℝ) (ratio : ℝ) (first_week_rainfall : ℝ) (second_week_rainfall : ℝ) :
  total_rainfall = 35 →
  ratio = 1.5 →
  total_rainfall = first_week_rainfall + second_week_rainfall →
  second_week_rainfall = ratio * first_week_rainfall →
  second_week_rainfall = 21 :=
by
  intros
  sorry

end rainfall_second_week_l521_521494


namespace probability_adjacent_vertices_in_decagon_l521_521572

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l521_521572


namespace net_gain_of_transaction_l521_521632

variable (house_cost store_cost : ℝ)

-- Conditions
def house_selling_price (house_cost : ℝ) : ℝ := 0.9 * house_cost
def store_selling_price (store_cost : ℝ) : ℝ := 1.3 * store_cost

-- Given selling prices
axiom house_sold : house_selling_price house_cost = 9000
axiom store_sold : store_selling_price store_cost = 13000

-- Proof statement
theorem net_gain_of_transaction : house_sold → store_sold → (13000 + 9000) - (house_cost + store_cost) = 2000 :=
by
  intro _ _
  sorry

end net_gain_of_transaction_l521_521632


namespace decagon_adjacent_probability_l521_521519

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521519


namespace number_of_medians_is_six_l521_521905

open Set

def has_median_set (s : Set ℤ) : Prop :=
  ∃ l (t : List ℤ), s = t.to_set ∧ l.length = 3 ∧ (t ++ l).to_set = {3, 5, 7, 10, 12, 18, 20, 25} ∧
    (∀ x ∈ t, x < 3) ∧ ∀ x ∈ l, x ∉ {3, 5, 7, 10, 12, 18, 20, 25}                             

def number_of_possible_medians : ℕ :=
  6

theorem number_of_medians_is_six (S : Set ℤ) (h : has_median_set S) : S.median_set_size = number_of_possible_medians :=
  sorry

#where median_set_size is defined to return 6 on the given set

end number_of_medians_is_six_l521_521905


namespace probability_adjacent_vertices_decagon_l521_521602

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l521_521602


namespace pair_cards_l521_521462

-- Given conditions
def even_number_cards (a : ℕ → ℕ) : Prop := ∃ k, ∀ n, ∃ m, a n = 2 * m
def positive_integer_written_on_each_card (a : ℕ → ℕ) : Prop := ∀ n, 0 < a n
def condition (a : ℕ → ℕ) : Prop := ∀ n, a n - a (n-1) + a (n-2) - a (n-3) + ... ≥ 0

-- Main theorem to prove
theorem pair_cards (a : ℕ → ℕ) (h1: even_number_cards a) (h2: positive_integer_written_on_each_card a) (h3: condition a) :
  ∃ p : (ℕ → ℕ) × (ℕ → ℕ), (∀ n, p.1 n ≠ p.2 n) ∧ (∀ n, a n = p.1 n + p.2 n) :=
sorry

end pair_cards_l521_521462


namespace triangle_side_lengths_count_l521_521794

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end triangle_side_lengths_count_l521_521794


namespace largest_possible_value_of_c_l521_521910

theorem largest_possible_value_of_c (c : ℚ) : (3 * c + 4) * (c - 2) = 9 * c → c ≤ 4 :=
by
  intro h
  have : (3 * c + 4) * (c - 2) = 3 * c^2 - 6 * c + 4 * c - 8 := 
    calc 
    (3 * c + 4) * (c - 2) = (3 * c) * (c - 2) + 4 * (c - 2) : by ring
                         ... = (3 * c) * c - (3 * c) * 2 + 4 * c - 4 * 2 : by ring
                         ... = 3 * c^2 - 6 * c + 4 * c - 8 : by ring
  rw this at h
  have h2 : 3 * c^2 - 11 * c - 8 = 0 := by nlinarith
  sorry

end largest_possible_value_of_c_l521_521910


namespace equation_of_parabola_l521_521754

theorem equation_of_parabola (p : ℝ) (h : p > 0) (A B : ℝ × ℝ)
  (hA : A.2 = A.1 - p / 2) (hB : B.2 = B.1 - p / 2)
  (h_parabola_A : A.2 ^ 2 = 2 * p * A.1) (h_parabola_B : B.2 ^ 2 = 2 * p * B.1)
  (h_dist : dist A B = 8) : 
  (∃ y x, y ^ 2 = 4 * x) :=
begin
  sorry
end

end equation_of_parabola_l521_521754


namespace expected_value_of_fair_dodecahedral_die_l521_521046

theorem expected_value_of_fair_dodecahedral_die : 
  (1/12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_of_fair_dodecahedral_die_l521_521046


namespace triangle_side_lengths_count_l521_521789

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end triangle_side_lengths_count_l521_521789


namespace abs_cos_sin_ge_one_or_abs_sin_cos_ge_one_l521_521412

theorem abs_cos_sin_ge_one_or_abs_sin_cos_ge_one (x : ℝ) : 
    (|Real.cos x - Real.sin x| ≥ 1) ∨ (|Real.sin x + Real.cos x| ≥ 1) := 
begin
  sorry
end

end abs_cos_sin_ge_one_or_abs_sin_cos_ge_one_l521_521412


namespace probability_of_three_white_balls_equals_8_over_65_l521_521619

noncomputable def probability_three_white_balls (n_white n_black : ℕ) (draws : ℕ) : ℚ :=
  (Nat.choose n_white draws : ℚ) / Nat.choose (n_white + n_black) draws

theorem probability_of_three_white_balls_equals_8_over_65 :
  probability_three_white_balls 8 7 3 = 8 / 65 :=
by
  sorry

end probability_of_three_white_balls_equals_8_over_65_l521_521619


namespace impossible_to_form_three_similar_piles_l521_521354

-- Define the concept of similar sizes
def similar_sizes (a b : ℝ) : Prop := a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2

-- State the proof problem
theorem impossible_to_form_three_similar_piles (x : ℝ) (h : 0 < x) : ¬(∃ a b c : ℝ, a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes a c) :=
sorry

end impossible_to_form_three_similar_piles_l521_521354


namespace probability_adjacent_vertices_decagon_l521_521597

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l521_521597


namespace sum_first_18_terms_l521_521229

noncomputable def a_n (n : ℕ) : ℚ := (-1 : ℚ)^n * (3 * n + 2) / (n * (n + 1) * 2^(n + 1))

noncomputable def S (n : ℕ) : ℚ := ∑ i in Finset.range n, a_n (i + 1)

theorem sum_first_18_terms :
  S 18 = (1 / (2^19 * 19) - 1 / 2) :=
sorry

end sum_first_18_terms_l521_521229


namespace decagon_adjacent_probability_l521_521538

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l521_521538


namespace possible_integer_side_lengths_l521_521800

theorem possible_integer_side_lengths : 
  {x : ℤ | x > 3 ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521800


namespace find_point_on_line_and_distance_l521_521165

def distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem find_point_on_line_and_distance :
  ∃ P : ℝ × ℝ, (2 * P.1 - 3 * P.2 + 5 = 0) ∧ (distance P (2, 3) = 13) →
  (P = (5, 5) ∨ P = (-1, 1)) :=
by
  sorry

end find_point_on_line_and_distance_l521_521165


namespace problem_l521_521850

variable (x y z : ℚ)

-- Conditions as definitions
def cond1 : Prop := x / y = 3
def cond2 : Prop := y / z = 5 / 2

-- Theorem statement with the final proof goal
theorem problem (h1 : cond1 x y) (h2 : cond2 y z) : z / x = 2 / 15 := 
by 
  sorry

end problem_l521_521850


namespace decagon_adjacent_probability_l521_521539

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l521_521539


namespace area_hexagon_DEFD_EFE_l521_521971

variable (D E F D' E' F' : Type)
variable (perimeter_DEF : ℝ) (radius_circumcircle : ℝ)
variable (area_hexagon : ℝ)

theorem area_hexagon_DEFD_EFE' (h1 : perimeter_DEF = 42)
    (h2 : radius_circumcircle = 7)
    (h_def : area_hexagon = 147) :
  area_hexagon = 147 := 
sorry

end area_hexagon_DEFD_EFE_l521_521971


namespace minimum_voters_for_tall_giraffe_to_win_l521_521275

/-- Conditions -/
def total_voters := 135
def districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority (n : Nat) : Nat := Nat.ceil (n / 2)

/-- Proof Problem -/
theorem minimum_voters_for_tall_giraffe_to_win : 
  let precincts_needed_in_district := majority precincts_per_district,
      districts_needed_to_win := majority districts,
      precincts_needed_in_total := districts_needed_to_win * precincts_needed_in_district,
      majority_in_precinct := majority voters_per_precinct in
  precincts_needed_in_total * majority_in_precinct = 30 :=
by
  -- Proof steps (commented out)
  -- 1. Calculate the number of precincts in total: 5 * 9 = 45
  -- 2. Calculate how many districts the Tall giraffe needs to win: 3
  -- (majority districts)
  -- 3. Calculate how many precincts in a district the Tall giraffe needs to
  -- win: 5 (majority precincts_per_district)
  -- 4. Calculate the total number of precincts the Tall giraffe needs to win:
  -- 3 * 5 = 15
  -- 5. Calculate the majority of votes in a precinct: 2 
  -- (majority voters_per_precinct)
  -- 6. Calculate the minimum number of voters in 15 precincts: 15 * 2 = 30
  sorry

end minimum_voters_for_tall_giraffe_to_win_l521_521275


namespace points_on_ellipse_l521_521220

-- Definitions of the conditions
def ellipse (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

def passes_through_point (a b : ℝ) : Prop :=
  ellipse a b 2 1

-- Target set of points
def target_set (x y : ℝ) : Prop :=
  x^2 + y^2 < 5 ∧ |y| > 1

-- Main theorem to prove
theorem points_on_ellipse (a b x y : ℝ) (h₁ : passes_through_point a b) (h₂ : |y| > 1) :
  ellipse a b x y → target_set x y :=
sorry

end points_on_ellipse_l521_521220


namespace fixed_point_exists_l521_521764

noncomputable def f (a x : ℝ) : ℝ := log a (x - 1) + 2

theorem fixed_point_exists (a x : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) : f a 2 = 2 :=
by
  have hx : x = 2 := by sorry
  have forward1 : log a 1 = 0 := by sorry
  rw [f, hx]
  rw [forward1]
  sorry

end fixed_point_exists_l521_521764


namespace odd_numbers_coprime_l521_521965

-- Definitions for the problem
def is_odd (n : ℤ) : Prop := n % 2 = 1

-- Main statement to prove
theorem odd_numbers_coprime (a b : ℤ) (h1 : is_odd a) (h2 : is_odd b) (h : |a - b| = 16) : Nat.gcd a.nat_abs b.nat_abs = 1 := 
by 
  -- Skipping the actual proof part
  sorry

end odd_numbers_coprime_l521_521965


namespace decagon_adjacent_probability_l521_521525

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521525


namespace minimum_voters_needed_l521_521273

-- conditions
def num_voters := 135
def num_districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority_precincts (n : ℕ) := (n + 1) / 2

-- definitions for quantities derived from conditions
def total_precincts := num_districts * precincts_per_district
def majority_districts := majority_precincts num_districts
def precincts_needed_for_district_win := majority_precincts precincts_per_district
def total_precincts_needed_for_win := majority_districts * precincts_needed_for_district_win
def votes_needed_per_precinct := majority_precincts voters_per_precinct

-- main statement
theorem minimum_voters_needed : (votes_needed_per_precinct * total_precincts_needed_for_win = 30) ∧ TallGiraffeWon :=
by sorry

end minimum_voters_needed_l521_521273


namespace expected_value_fair_dodecahedral_die_l521_521067

theorem expected_value_fair_dodecahedral_die : 
  let n := 12 in (1 / n) * (List.sum (List.range n).map (λ x => (x + 1))) = 6.5 :=
by
  let n := 12
  have h1 : (1 : ℚ) / n = 1 / 12 := by sorry
  have h2 : List.range n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] := by sorry
  have h3 : List.sum (List.range n).map (λ x => (x + 1)) = 78 := by sorry
  have h4 : (1 / 12) * 78 = 6.5 := by sorry
  exact h4

end expected_value_fair_dodecahedral_die_l521_521067


namespace minimum_moves_l521_521238

def sidingCapacityAccommodate : Prop := ∀ (x : Train), (x = Locomotive ∨ x = Wagon) → x ∈ Siding
def noRopesOrFlyingAllowed : Prop := ¬ (existRopes ∨ existFlying)
def movesCounted : Prop := ∀ (move : Move), move ∈ Moves → change_direction(move) → counted_as_one_move(move)

theorem minimum_moves (h1 : sidingCapacityAccommodate) (h2 : noRopesOrFlyingAllowed) (h3 : movesCounted) :
  ∀ (trains : Train) (switch : Switch), minimum_moves_required (trains, switch) = 14 := 
sorry

end minimum_moves_l521_521238


namespace not_possible_to_create_3_piles_l521_521356

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end not_possible_to_create_3_piles_l521_521356


namespace smallest_n_divisibility_l521_521482

theorem smallest_n_divisibility (n : ℕ) (h : 1 ≤ n) :
  (∀ k, 1 ≤ k ∧ k ≤ n → n^3 - n ∣ k) ∨ (∃ k, 1 ≤ k ∧ k ≤ n ∧ ¬ (n^3 - n ∣ k)) :=
sorry

end smallest_n_divisibility_l521_521482


namespace volleyball_teams_l521_521138

theorem volleyball_teams (managers employees teams : ℕ) (h1 : managers = 3) (h2 : employees = 3) (h3 : teams = 3) :
  ((managers + employees) / teams) = 2 :=
by
  sorry

end volleyball_teams_l521_521138


namespace circumscribed_surface_area_right_prism_l521_521265

noncomputable def circumscribed_surface_area (AB BD : ℝ) :=
  4 * π * (1 / 8)

theorem circumscribed_surface_area_right_prism
  (AB BD : ℝ)
  (h1 : AB * BD = 0)
  (h2 : 4 * AB^2 + 2 * BD^2 = 1) :
  circumscribed_surface_area AB BD = π / 2 := 
sorry

end circumscribed_surface_area_right_prism_l521_521265


namespace polynomial_value_at_0_l521_521034

def polynomial (x : ℝ) : ℝ := 3 * x^6 + 4 * x^5 + 5 * x^4 + 6 * x^3 + 7 * x^2 + 8 * x + 1
    
theorem polynomial_value_at_0.4 :
  polynomial 0.4 = 5.885248 :=
by
  have V0 := 3
  have V1 := V0 * 0.4 + 4
  have V2 := V1 * 0.4 + 5
  have V3 := V2 * 0.4 + 6
  have V4 := V3 * 0.4 + 7
  have V5 := V4 * 0.4 + 8
  have V6 := V5 * 0.4 + 1
  rw [V1, V2, V3, V4, V5, V6]
  exact rfl

#check polynomial_value_at_0.4

end polynomial_value_at_0_l521_521034


namespace candidate_lost_by_4000_votes_l521_521620

-- Defining the conditions
def candidate_votes (total_votes : ℕ) (percentage : ℚ) : ℕ := (percentage * total_votes).toNat
def rival_votes (total_votes : ℕ) (candidate_percentage : ℚ) : ℕ := ((1 - candidate_percentage) * total_votes).toNat

-- The actual theorem statement
theorem candidate_lost_by_4000_votes :
  ∀ (total_votes : ℕ) (percentage : ℚ),
    total_votes = 8000 → percentage = 0.25 →
    rival_votes total_votes percentage - candidate_votes total_votes percentage = 4000 :=
by
  intros total_votes percentage h_total_votes h_percentage
  simp [candidate_votes, rival_votes, h_total_votes, h_percentage]
  have c_votes : candidate_votes total_votes percentage = 2000 :=
    by
      simp [candidate_votes, h_total_votes, h_percentage]
      norm_num
  have r_votes : rival_votes total_votes percentage = 6000 :=
    by
      simp [rival_votes, h_total_votes, h_percentage]
      norm_num
  rw [c_votes, r_votes]
  norm_num
  sorry

end candidate_lost_by_4000_votes_l521_521620


namespace intersection_on_diagonal_l521_521264

variables {A B C D K L M N O : Point}
variable {rectangle : Rectangle}
variable {AB BC CD DA : Line}
variable {KL MN KM LN BD : Line}

-- Conditions
axiom points_on_sides :
  (K ∈ AB) ∧ (L ∈ BC) ∧ (M ∈ CD) ∧ (N ∈ DA) ∧
  (K ≠ A) ∧ (K ≠ B) ∧ (L ≠ B) ∧ (L ≠ C) ∧
  (M ≠ C) ∧ (M ≠ D) ∧ (N ≠ D) ∧ (N ≠ A)

axiom parallel_lines : parallel KL MN
axiom perpendicular_lines_at_O : (KM ∩ NL = O) ∧ (KM ⊥ NL)

-- Prove that the intersection point of KM and LN lies on the diagonal BD of the rectangle
theorem intersection_on_diagonal :
  (KM ∩ LN) = O → (O ∈ BD) :=
by sorry

end intersection_on_diagonal_l521_521264


namespace perpendicular_lines_k_value_l521_521443

theorem perpendicular_lines_k_value (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 3 = 0 → x + (2 * k + 3) * y - 2 = 0) →
  k = -3 :=
by
  sorry

end perpendicular_lines_k_value_l521_521443


namespace probability_product_odd_and_div_by_3_l521_521959

theorem probability_product_odd_and_div_by_3 :
  let integers := {i | 4 ≤ i ∧ i ≤ 20}.to_finset in
  let odd_and_div_by_3 := integers.filter (λ i, i % 2 ≠ 0 ∧ i % 3 = 0) in
  fintype.card (pair_combinations integers) = 136 →
  ((fintype.card (pair_combinations odd_and_div_by_3)).to_rat /
    (fintype.card (pair_combinations integers)).to_rat = 1 / 136) :=
by
  let integers := {i | 4 ≤ i ∧ i ≤ 20}.to_finset
  let odd_and_div_by_3 := integers.filter (λ i, i % 2 ≠ 0 ∧ i % 3 = 0)
  let total_pairs := fintype.card (pair_combinations integers)
  let odd_div_by_3_pairs := fintype.card (pair_combinations odd_and_div_by_3)
  suffices total_pairs = 136, by
    suffices (odd_div_by_3_pairs.to_rat / total_pairs.to_rat = 1 / 136), by
    unfold pair_combinations
  sorry

end probability_product_odd_and_div_by_3_l521_521959


namespace impossible_to_form_three_similar_piles_l521_521350

-- Define the concept of similar sizes
def similar_sizes (a b : ℝ) : Prop := a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2

-- State the proof problem
theorem impossible_to_form_three_similar_piles (x : ℝ) (h : 0 < x) : ¬(∃ a b c : ℝ, a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes a c) :=
sorry

end impossible_to_form_three_similar_piles_l521_521350


namespace parabola_translation_l521_521994

theorem parabola_translation (x : ℝ) : 
  let y := x^2 in
  let y_translated := (x - 2)^2 + 1 in 
  ∀ x, y_translated = (x - 2)^2 + 1 :=
by
  sorry

end parabola_translation_l521_521994


namespace expected_value_fair_dodecahedral_die_l521_521055

theorem expected_value_fair_dodecahedral_die :
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  expected_value = 6.5 :=
by
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  have h1 : list.sum outcomes = 78 := by sorry
  have h2 : (12: ℝ) * (1 / 12: ℝ) = 1 := by sorry
  show expected_value = 6.5
  { sorry }

end expected_value_fair_dodecahedral_die_l521_521055


namespace square_area_l521_521266

-- Define the properties of segments and points on the square
variable (PQ RS : Type) [metric_space PQ] [metric_space RS]
variable (P Q G R S H : PQ)

variables (PG GH HS : ℝ)
-- Conditions: lengths of segments
variable (hPG : dist P G = 20)
variable (hGH : dist G H = 20)
variable (hHS : dist H S = 20)

-- Side length of the square
variable (s : ℝ)

-- Condition: Square properties and equal segmentation
variable (hS : s = 3 * 20)

theorem square_area (h : dist P Q = s ∧ dist R S = s) : s^2 = 360 :=
by
  -- Sorry to indicate missing proof steps
  sorry

end square_area_l521_521266


namespace intersection_and_angle_l521_521150

def line_n (x y : ℝ) : Prop := y - 2 * x = 1
def line_p (x y : ℝ) : Prop := 3 * x + y = 6

noncomputable def intersection (x y : ℝ) : Prop := line_n x y ∧ line_p x y

noncomputable def angle_between_lines (m1 m2 : ℝ) : ℝ :=
  Real.arctan ((m1 - m2) / (1 + m1 * m2))

theorem intersection_and_angle :
  ∃ (x y : ℝ), intersection x y ∧ angle_between_lines 2 (-3) = Real.pi / 4 :=
by
  use 1, 3
  split
  · -- prove intersection point
    unfold intersection line_n line_p
    simp
  · -- prove angle is 45 degrees (π/4 radians)
    unfold angle_between_lines
    norm_num
    rw [Real.arctan_one]
    ring
    norm_num
  sorry

end intersection_and_angle_l521_521150


namespace find_m_l521_521496

-- Given the condition
def condition (m : ℕ) := (1 / 5 : ℝ)^m * (1 / 4 : ℝ)^2 = 1 / (10 : ℝ)^4

-- Theorem to prove that m is 4 given the condition
theorem find_m (m : ℕ) (h : condition m) : m = 4 :=
sorry

end find_m_l521_521496


namespace not_valuePreserving_g_range_a_valuePreserving_f_l521_521176

-- Define value-preserving function as per the problem
def valuePreserving (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  (m < n) ∧
  (∀ x ∈ Icc m n, f x ∈ Icc m n) ∧
  (∀ x1 x2 ∈ Icc m n, x1 ≤ x2 → f x1 ≤ f x2 ∨ f x2 ≤ f x1) -- Monotonicity condition (allows non-strict monotonicity)

-- Problem 1: Prove g(x) = x^2 - 2x is not value-preserving on [0, 1]
theorem not_valuePreserving_g : ¬ valuePreserving (λ x : ℝ, x^2 - 2 * x) 0 1 := sorry

-- Problem 2: Given f(x) = 2 + 1/a - 1/(a^2 x) is value-preserving on [m, n], find the range of a
theorem range_a_valuePreserving_f (a m n : ℝ) (h : a ≠ 0) (hp : valuePreserving (λ x : ℝ, 2 + 1/a - 1/(a^2 * x)) m n) :
  a > 1/2 ∨ a < -3/2 := sorry

end not_valuePreserving_g_range_a_valuePreserving_f_l521_521176


namespace sum_of_remainders_is_six_l521_521952

def sum_of_remainders (n : ℕ) : ℕ :=
  n % 4 + (n + 1) % 4 + (n + 2) % 4 + (n + 3) % 4

theorem sum_of_remainders_is_six : ∀ n : ℕ, sum_of_remainders n = 6 :=
by
  intro n
  sorry

end sum_of_remainders_is_six_l521_521952


namespace inequality_ln_x_lt_x_lt_exp_x_l521_521767

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def g (x : ℝ) : ℝ := Real.log x - x

theorem inequality_ln_x_lt_x_lt_exp_x (x : ℝ) (h : x > 0) : Real.log x < x ∧ x < Real.exp x := by
  -- We need to supply the proof here
  sorry

end inequality_ln_x_lt_x_lt_exp_x_l521_521767


namespace expected_value_fair_dodecahedral_die_l521_521076

theorem expected_value_fair_dodecahedral_die : 
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  let probability := (1 : ℝ) / 12 in
  (probability * faces.sum) = 6.5 :=
by
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let probability := (1 : ℝ) / 12
  have h : faces.sum = 78 := rfl
  rw [h]
  sorry

end expected_value_fair_dodecahedral_die_l521_521076


namespace cookies_with_new_flour_l521_521301

-- Definitions for the conditions
def cookies_per_cup (total_cookies : ℕ) (total_flour : ℕ) : ℕ :=
  total_cookies / total_flour

noncomputable def cookies_from_flour (cookies_per_cup : ℕ) (flour : ℕ) : ℕ :=
  cookies_per_cup * flour

-- Given data
def total_cookies := 24
def total_flour := 4
def new_flour := 3

-- Theorem (problem statement)
theorem cookies_with_new_flour : cookies_from_flour (cookies_per_cup total_cookies total_flour) new_flour = 18 :=
by
  sorry

end cookies_with_new_flour_l521_521301


namespace decagon_adjacent_probability_l521_521528

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l521_521528


namespace growth_factor_after_20_years_l521_521128

-- Define the variables and conditions
def rate_per_annum : ℝ := 5 / 100
def time_period : ℕ := 20

-- Define the formula for growth factor
def growth_factor (rate : ℝ) (time : ℕ) : ℝ :=
  1 + (rate * time)

-- Statement to prove
theorem growth_factor_after_20_years :
  growth_factor rate_per_annum time_period = 2 :=
by
  sorry

end growth_factor_after_20_years_l521_521128


namespace cos_inner_product_l521_521315

variables {ℝ : Type*} [inner_product_space ℝ E] 
variables {a b : E}
variables (hab : ‖a‖ = 1 ∧ ‖b‖ = 1) 
variables (proj_condition : (a + b) = (2 / 3) • b)

theorem cos_inner_product (hab : ‖a‖ = 1 ∧ ‖b‖ = 1) 
(proj_condition : (a + b) = (2 / 3) • b) :
cos_angle a b = -1 / 3 :=
sorry

end cos_inner_product_l521_521315


namespace impossible_to_form_three_similar_piles_l521_521348

-- Define the concept of similar sizes
def similar_sizes (a b : ℝ) : Prop := a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2

-- State the proof problem
theorem impossible_to_form_three_similar_piles (x : ℝ) (h : 0 < x) : ¬(∃ a b c : ℝ, a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes a c) :=
sorry

end impossible_to_form_three_similar_piles_l521_521348


namespace probability_of_adjacent_vertices_in_decagon_l521_521594

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l521_521594


namespace eight_pow_m_div_two_pow_n_l521_521246

theorem eight_pow_m_div_two_pow_n {m n : ℤ} (h : 3 * m - n - 4 = 0) : 8^m / 2^n = 16 :=
by
  sorry

end eight_pow_m_div_two_pow_n_l521_521246


namespace impossible_to_form_three_similar_piles_l521_521352

-- Define the concept of similar sizes
def similar_sizes (a b : ℝ) : Prop := a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2

-- State the proof problem
theorem impossible_to_form_three_similar_piles (x : ℝ) (h : 0 < x) : ¬(∃ a b c : ℝ, a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes a c) :=
sorry

end impossible_to_form_three_similar_piles_l521_521352


namespace cricket_running_percentage_l521_521111

theorem cricket_running_percentage :
  let total_runs := 136
  let boundaries := 12
  let sixes := 2
  let boundary_runs := boundaries * 4
  let six_runs := sixes * 6
  let total_boundary_six_runs := boundary_runs + six_runs
  let running_runs := total_runs - total_boundary_six_runs
  (running_runs / total_runs.toFloat) * 100 = 55.88 :=
by
  let total_runs := 136
  let boundaries := 12
  let sixes := 2
  let boundary_runs := boundaries * 4
  let six_runs := sixes * 6
  let total_boundary_six_runs := boundary_runs + six_runs
  let running_runs := total_runs - total_boundary_six_runs
  have fraction := (running_runs.toFloat / total_runs.toFloat)
  show (fraction * 100 = 55.88) from sorry

end cricket_running_percentage_l521_521111


namespace find_D_l521_521733

variable (n : ℕ) (h_n : n > 0)

def D : ℕ :=
  ∑ S in (finset.powerset (finset.range n).erase ∅), (∏ i in S, (1 : ℚ) / (i + 1))

theorem find_D (hn : n ∈ ℕ+ ): D n = n := by
  sorry

end find_D_l521_521733


namespace meeting_distance_from_A_l521_521404

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (distance_AB distance_BC : ℝ)
variable (cyclist_speed pedestrian_speed : ℝ)
variable (meet_distance : ℝ)

axiom distance_AB_eq_3 : distance_AB = 3
axiom distance_BC_eq_4 : distance_BC = 4
axiom simultaneous_arrival :
  ∀ AC cyclist_speed pedestrian_speed,
    (distance_AB + distance_BC) / cyclist_speed = distance_AB / pedestrian_speed
axiom speed_ratio :
  cyclist_speed / pedestrian_speed = 7 / 3
axiom meeting_point :
  ∃ meet_distance,
    meet_distance / (distance_AB - meet_distance) = 7 / 3

theorem meeting_distance_from_A :
  meet_distance = 2.1 :=
sorry

end meeting_distance_from_A_l521_521404


namespace round_down_example_l521_521416

theorem round_down_example : Real.toInt 2357812.4999998 = 2357812 :=
by
  sorry

end round_down_example_l521_521416


namespace six_digit_number_conditions_l521_521468

theorem six_digit_number_conditions :
  ∀ (a b c d e f : ℕ),
  (a ∈ {1, 2, 3, 4, 5, 6}) ∧ 
  (b ∈ {1, 2, 3, 4, 5, 6}) ∧ 
  (c ∈ {1, 2, 3, 4, 5, 6}) ∧ 
  (d ∈ {1, 2, 3, 4, 5, 6}) ∧ 
  (e ∈ {1, 2, 3, 4, 5, 6}) ∧ 
  (f ∈ {1, 2, 3, 4, 5, 6}) ∧ 
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ 
  (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ 
  (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ 
  (d ≠ e) ∧ (d ≠ f) ∧ 
  (e ≠ f) → 
  ((10 * a + b) % 2 = 0) ∧ 
  ((100 * a + 10 * b + c) % 3 = 0) ∧ 
  ((1000 * a + 100 * b + 10 * c + d) % 4 = 0) ∧ 
  ((10000 * a + 1000 * b + 100 * c + 10 * d + e) % 5 = 0) ∧ 
  ((100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) % 6 = 0) → 
  (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 6 ∧ e = 5 ∧ f = 4 ∨ 
   a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 6 ∧ e = 5 ∧ f = 4) := sorry

end six_digit_number_conditions_l521_521468


namespace probability_win_l521_521446

-- Definitions for the conditions
def P_lose : ℚ := 5 / 11 -- probability of losing

-- Define the goal
theorem probability_win : (1 - P_lose) = 6 / 11 := 
by
  rw [P_lose] -- substitute P_lose with 5/11
  norm_num -- normalize the resulting fraction and simplify
  sorry -- skipping the actual computation

end probability_win_l521_521446


namespace min_dist_to_y_axis_symmetric_center_l521_521884

def f (x : ℝ) : ℝ := (Real.sqrt 2 / 4) * Real.sin (x - Real.pi / 4) + (Real.sqrt 6 / 4) * Real.cos (x - Real.pi / 4)

theorem min_dist_to_y_axis_symmetric_center : 
  ∀ (x : ℝ), f(x) = (Real.sqrt 2 / 2) * Real.sin (x + Real.pi / 12) → 
  (∃ k : ℤ, x = k * Real.pi - Real.pi / 12) → 
  ∃ k : ℤ, abs (k * Real.pi - Real.pi / 12) = Real.pi / 12 :=
sorry

end min_dist_to_y_axis_symmetric_center_l521_521884


namespace decagon_adjacent_probability_l521_521546

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l521_521546


namespace geom_sequence_general_formula_lambda_range_l521_521741

variable {a : ℕ → ℤ}

-- Let 'a' be a sequence satisfying the given initial conditions.
def seq_conditions (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n : ℕ, 0 < n → a (n + 1) - 2 * a n = 2 * n

-- Statement 1: Prove that the given sequence adjusted by 2n + 2 forms a geometric sequence.
theorem geom_sequence {a : ℕ → ℤ} (h : seq_conditions a) :
  ∃ r (c : ℕ → ℤ), (r = 2 ∧ c 1 = 4) ∧ (∀ n : ℕ, 0 < n → a (n + 1) + 2 * (n + 1) + 2 = r * (a n + 2 * n + 2)) :=
sorry

-- Statement 2: Find the general formula for the sequence.
theorem general_formula {a : ℕ → ℤ} (h : seq_conditions a) :
  ∀ n : ℕ, 0 < n → a n = 2 ^ (n + 1) - 2 * n - 2 :=
sorry

-- Statement 3: Given the inequality condition involving lambda, find its range.
theorem lambda_range {a : ℕ → ℤ} (h : seq_conditions a) (λ : ℝ) :
  (∀ n : ℕ, 0 < n → a n > λ * (2 * n + 1) * (-1) ^ (n - 1)) ↔ ( -2 / 5 < λ ∧ λ < 0) :=
sorry

end geom_sequence_general_formula_lambda_range_l521_521741


namespace range_of_expression_l521_521205

theorem range_of_expression (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a * b = 2) :
  (a^2 + b^2) / (a - b) ≤ -4 :=
sorry

end range_of_expression_l521_521205


namespace surface_area_comparison_l521_521918

def parabola := {p : ℝ // p > 0}
def chord (p : parabola) := {PQ : ℝ × ℝ // PQ.2 ^ 2 = 2 * p * PQ.1}
def directrix (p : parabola) := λ (x : ℝ), x = -p

noncomputable def surface_area_of_revolution (PQ : ℝ) := π * PQ
noncomputable def surface_area_of_sphere (MN : ℝ) := π * (MN ^ 2)

theorem surface_area_comparison (p : parabola) (PQ : chord p) (MN : ℝ) (PQ_rot : ℝ)
  (MN_proj : ℝ) (S1 := surface_area_of_revolution PQ_rot) (S2 := surface_area_of_sphere MN_proj) :
  S1 ≥ S2 :=
sorry

end surface_area_comparison_l521_521918


namespace percent_women_non_union_employees_is_65_l521_521255

-- Definitions based on the conditions
variables {E : ℝ} -- Denoting the total number of employees as a real number

def percent_men (E : ℝ) : ℝ := 0.56 * E
def percent_union_employees (E : ℝ) : ℝ := 0.60 * E
def percent_non_union_employees (E : ℝ) : ℝ := 0.40 * E
def percent_women_non_union (percent_non_union_employees : ℝ) : ℝ := 0.65 * percent_non_union_employees

-- Theorem statement
theorem percent_women_non_union_employees_is_65 :
  percent_women_non_union (percent_non_union_employees E) / (percent_non_union_employees E) = 0.65 :=
by
  sorry

end percent_women_non_union_employees_is_65_l521_521255


namespace impossible_to_form_three_similar_piles_l521_521351

-- Define the concept of similar sizes
def similar_sizes (a b : ℝ) : Prop := a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2

-- State the proof problem
theorem impossible_to_form_three_similar_piles (x : ℝ) (h : 0 < x) : ¬(∃ a b c : ℝ, a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes a c) :=
sorry

end impossible_to_form_three_similar_piles_l521_521351


namespace expected_value_dodecahedral_die_is_6_5_l521_521062

noncomputable def expected_value_dodecahedral_die : ℝ :=
  let outcomes := finset.range 12 in
  1 / 12 * (finset.sum outcomes (λ i, (i + 1 : ℝ)))

theorem expected_value_dodecahedral_die_is_6_5 :
  expected_value_dodecahedral_die = 6.5 :=
by
  sorry

end expected_value_dodecahedral_die_is_6_5_l521_521062


namespace impossible_to_create_3_piles_l521_521335

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end impossible_to_create_3_piles_l521_521335


namespace total_recess_correct_l521_521454

-- Definitions based on the conditions
def base_recess : Int := 20
def recess_for_A (n : Int) : Int := n * 2
def recess_for_B (n : Int) : Int := n * 1
def recess_for_C (n : Int) : Int := n * 0
def recess_for_D (n : Int) : Int := -n * 1

def total_recess (a b c d : Int) : Int :=
  base_recess + recess_for_A a + recess_for_B b + recess_for_C c + recess_for_D d

-- The proof statement originally there would use these inputs
theorem total_recess_correct : total_recess 10 12 14 5 = 47 := by
  sorry

end total_recess_correct_l521_521454


namespace christopher_strolled_5_miles_l521_521686

theorem christopher_strolled_5_miles (s t : ℝ) (hs : s = 4) (ht : t = 1.25) : s * t = 5 :=
by
  rw [hs, ht]
  norm_num

end christopher_strolled_5_miles_l521_521686


namespace probability_adjacent_vertices_decagon_l521_521599

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l521_521599


namespace max_sum_a_plus_b_theorem_l521_521888

noncomputable def max_sum_a_plus_b (a b c A B C : ℝ)
  (h1 : a ≥ b) 
  (h2 : sin (π / 3 - A) = sin B)
  (h3 : a * sin C = sqrt 3 * sin A)
  (h4 : A + B + C = π) : ℝ :=
  2

theorem max_sum_a_plus_b_theorem (a b c A B C : ℝ)
  (h1 : a ≥ b)
  (h2 : sin (π / 3 - A) = sin B)
  (h3 : a * sin C = sqrt 3 * sin A)
  (h4 : A + B + C = π) :
  max_sum_a_plus_b a b c A B C h1 h2 h3 h4 = 2 :=
sorry

end max_sum_a_plus_b_theorem_l521_521888


namespace add_and_convert_to_base4_l521_521089

theorem add_and_convert_to_base4 :
  ∀ (a b : ℕ), a = 34 → b = 47 → natToBase 4 (a + b) = [1, 1, 0, 1] :=
by
  intros a b ha hb
  rw [ha, hb]
  norm_num
  sorry

end add_and_convert_to_base4_l521_521089


namespace constant_C_is_27_l521_521857

-- Definitions based on conditions
def f (x : ℝ) (C : ℝ) : ℝ := C - x^2 / 2

-- The theorem statement
theorem constant_C_is_27 (C k : ℝ) (h1 : f (2 * k) C = 3 * k) (h2 : k = 3) : C = 27 :=
by
  have h3 : f (6 : ℝ) C = 9 := by rw [h2, ←h1]
  sorry

end constant_C_is_27_l521_521857


namespace max_houses_saved_l521_521875

theorem max_houses_saved (n c : ℕ) (h1 : 1 ≤ c) (h2 : c ≤ n / 2) :
  let max_houses := n * (n - c) + c * (c - 1) in
  ∀ houses : fin n × fin n → Prop, 
    (∀ t, ∃ defended : fin n × fin n → Prop, ∀ t ≤ n,
        (∃ (i j : fin n), houses (i, j) ∧ (|i.1.1 - 1| + |j.1.1 - c| = t) ∧ ¬defended (i, j)) ∧ 
        (defended (def_house: fin n × fin n) → ∀ neighbor, |def_house.fst.1 - neighbor.fst.1| + 
                      |def_house.snd.1 - neighbor.snd.1| = 1 ∧ ¬houses neighbor))  → 
    ∃ m ≤ max_houses, ∀ t ≤ n, ∃ defended : fin n × fin n → Prop,
      ∃ (i : fin n), defended (1, c) 
      (defended (def_house: fin n × fin n) → houses def_house)

end max_houses_saved_l521_521875


namespace triangle_side_lengths_count_l521_521790

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end triangle_side_lengths_count_l521_521790


namespace slope_product_line_fixed_point_l521_521882

variables {a b x y : ℝ}
variables (x1 x2 y1 y2 : ℝ) 
variables (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (l : ℝ → ℝ)
variables (xO yO: ℝ) (P Q M: ℝ → ℝ ) (kx ky: ℝ)

def midpoint (P Q: ℝ → ℝ) : ℝ → ℝ :=
λ (xO yO: ℝ), (P xO, Q y0)

def hyperbola (x y a b: ℝ) : Prop := (x ^ 2) / (a^2) - (y ^ 2) / (b^2) = 1

noncomputable def point_on_hyperbola_1 (x y a b x1 y1: ℝ ) : Prop := 
midpoint (x,y) (x1,y1) ∧
hyperbola(x1 y1 a b)

noncomputable def point_on_hyperbola_2 (x y a b x2 y2: ℝ ) : Prop := 
midpoint (x,y) (x2,y2) ∧
hyperbola(x2 y2 a b)

-- The product of slopes k1 and k2 is b^2 / a^2
theorem slope_product (x1 y1 x2 y2 : ℝ) (a:ℝ): 
  k1 * k2 = b ^ 2 / a ^ 2 
  := sorry

-- The line always passes through the fixed point
theorem line_fixed_point {fixed_point : ℝ → ℝ} : 
  ∃ p : point,
    (fixed_point (a,b) = 
  (\frac{a(a^2+b^2)}{a^2 - b^2},0))


end slope_product_line_fixed_point_l521_521882


namespace reciprocal_of_2016_is_1_div_2016_l521_521449

theorem reciprocal_of_2016_is_1_div_2016 : (2016 * (1 / 2016) = 1) :=
by
  sorry

end reciprocal_of_2016_is_1_div_2016_l521_521449


namespace infinite_distinct_quadruples_l521_521414

theorem infinite_distinct_quadruples (p : ℕ) (hp : Nat.Prime p) :
  ∃ (inf_list : List (ℕ × ℕ × ℕ × ℕ)), ∀ q ∈ inf_list, 
  let ⟨x, y, z, t⟩ := q in
  x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t ∧ 
  x ^ 2 + p * t ^ 2 = y ^ 2 + p * t ^ 2 ∧
  y ^ 2 + p * t ^ 2 = z ^ 2 + p * t ^ 2 ∧
  ∃ n : ℕ, (x ^ 2 + p * t ^ 2) * (y ^ 2 + p * t ^ 2) * (z ^ 2 + p * t ^ 2) = n ^ 2 :=
sorry

end infinite_distinct_quadruples_l521_521414


namespace coeff_x3_in_expansion_l521_521153

theorem coeff_x3_in_expansion :
  let f := (2 * x - 1) * (1 / x + x) ^ 6 in
  (nat_degree f <= 3) →
  (coeff f 3 = 30) :=
by
  -- We will not provide the full proof here, hence sorry
  sorry

end coeff_x3_in_expansion_l521_521153


namespace possible_integer_side_lengths_l521_521825

theorem possible_integer_side_lengths (a b : ℕ) (h1 : a = 8) (h2 : b = 5) :
  {x : ℕ | 3 < x ∧ x < 13}.finite ∧ {x : ℕ | 3 < x ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521825


namespace expected_value_fair_dodecahedral_die_l521_521053

theorem expected_value_fair_dodecahedral_die :
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  expected_value = 6.5 :=
by
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  have h1 : list.sum outcomes = 78 := by sorry
  have h2 : (12: ℝ) * (1 / 12: ℝ) = 1 := by sorry
  show expected_value = 6.5
  { sorry }

end expected_value_fair_dodecahedral_die_l521_521053


namespace not_possible_three_piles_l521_521386

def similar (x y : ℝ) : Prop :=
  y ≤ (real.sqrt 2) * x ∧ x ≤ (real.sqrt 2) * y

theorem not_possible_three_piles (N : ℝ) : 
  ¬ ∃ a b c : ℝ, a + b + c = N ∧ similar a b ∧ similar b c ∧ similar a c := 
by
  sorry

end not_possible_three_piles_l521_521386


namespace bisect_area_and_perimeter_l521_521231

-- Define the triangle and incenter structure
structure Triangle :=
(A B C : Point)
(incenter : Point)

-- Define the semiperimeter function
def semiperimeter (t : Triangle) : ℝ :=
  let sideLength (a b : Point) := dist a b
  in (sideLength t.A t.B + sideLength t.B t.C + sideLength t.C t.A) / 2

-- Lean theorem statement
theorem bisect_area_and_perimeter (t : Triangle) (l : Line) :
  passes_through l t.incenter →
  bisects_area l t →
  bisects_perimeter l t :=
sorry

end bisect_area_and_perimeter_l521_521231


namespace impossible_to_create_3_piles_l521_521336

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end impossible_to_create_3_piles_l521_521336


namespace triangle_possible_sides_l521_521780

theorem triangle_possible_sides : 
  let unknown_side_lengths := {x : ℤ | 3 < x ∧ x < 13}
  in set.finite unknown_side_lengths ∧ set.card unknown_side_lengths = 9 :=
by
  sorry

end triangle_possible_sides_l521_521780


namespace expected_value_fair_dodecahedral_die_l521_521077

theorem expected_value_fair_dodecahedral_die : 
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  let probability := (1 : ℝ) / 12 in
  (probability * faces.sum) = 6.5 :=
by
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let probability := (1 : ℝ) / 12
  have h : faces.sum = 78 := rfl
  rw [h]
  sorry

end expected_value_fair_dodecahedral_die_l521_521077


namespace count_pages_with_digit_5_l521_521106

def pages_containing_digit_5 (n : ℕ) : ℕ :=
  (List.range (n + 1)).countp (λ p, '5' ∈ p.digits 10)

theorem count_pages_with_digit_5 :
  pages_containing_digit_5 100 = 15 := 
sorry

end count_pages_with_digit_5_l521_521106


namespace count_5_in_bus_numbers_l521_521139

def count_digit_5 (start : ℕ) (end : ℕ) : ℕ :=
  (list.range' start (end - start + 1)).countp (λ n, n.digits 10).contains 5

theorem count_5_in_bus_numbers : count_digit_5 1 54 = 10 := by
  sorry

end count_5_in_bus_numbers_l521_521139


namespace mary_flour_amount_l521_521936

noncomputable def cups_of_flour_already_put_in
    (total_flour_needed : ℕ)
    (total_sugar_needed : ℕ)
    (extra_flour_needed : ℕ)
    (flour_to_be_added : ℕ) : ℕ :=
total_flour_needed - (total_sugar_needed + extra_flour_needed)

theorem mary_flour_amount
    (total_flour_needed : ℕ := 9)
    (total_sugar_needed : ℕ := 6)
    (extra_flour_needed : ℕ := 1) :
    cups_of_flour_already_put_in total_flour_needed total_sugar_needed extra_flour_needed (total_sugar_needed + extra_flour_needed) = 2 := by
  sorry

end mary_flour_amount_l521_521936


namespace decagon_adjacent_vertex_probability_l521_521552

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l521_521552


namespace part1_part2_l521_521727

noncomputable def f (x : ℝ) := x / (1 + 2 * x^2)

theorem part1 (α β : ℝ) (h1 : sin (2 * α + β) = 3 * sin β) (hx : tan α = x) (hy : tan β = f x) :
  tan β = f x :=
sorry

theorem part2 (α x : ℝ) (hx : tan α = x)
  (angle_cond : 0 < α ∧ α < π / 3) (x_range : x ∈ Ioo 0 (sqrt 3)) :
  set.range (λ x, f x) = Ioc 0 (sqrt 2 / 4) :=
sorry

end part1_part2_l521_521727


namespace max_f_max_dot_product_l521_521225

noncomputable def f (x : ℝ) : ℝ := 4 * sqrt 3 * sin x * cos x - 4 * sin x ^ 2 + 1

theorem max_f :
  ∃ (k : ℤ), ∀ x : ℝ, f x ≤ 3 ∧ (∃ k : ℤ, x = π / 6 + k * π) → f (π / 6) = 3 :=
by
  sorry

theorem max_dot_product (a b c : ℝ) (A : ℝ) (h1 : a = 2) (h2 : A = π / 6)
  (h3 : b^2 + c^2 - a^2 = 2 * b * c * cos A) :
  ∃ M : ℝ, ∀ b c : ℝ, 
    (A ∈ (0, π)) → b^2 + c^2 ≥ 2 * b * c →
    (√3 / 2 * b * c) ≤ M ∧ M = 6 + 4 * sqrt 3 :=
by
  sorry

end max_f_max_dot_product_l521_521225


namespace find_prime_x_l521_521117

theorem find_prime_x (n : ℕ) :
  let x_n := (10^n - 1) * (10^n + 1) / 99 in
  (∀ n > 2, ¬ prime x_n) ∧ (x_n = 101 ↔ n = 2) :=
by
  sorry

end find_prime_x_l521_521117


namespace expected_value_dodecahedral_die_l521_521087

noncomputable def dodecahedral_expected_value : ℝ :=
  let outcomes := (list.range (12 + 1)).tail -- List of numbers from 1 to 12
  let n_faces := 12
  let probability := 1 / (n_faces : ℝ)
  let expected_value := probability * outcomes.sum / n_faces
  expected_value

theorem expected_value_dodecahedral_die :
  dodecahedral_expected_value = 6.5 :=
by {
  unfold dodecahedral_expected_value,
  simp,
  sorry,
}

end expected_value_dodecahedral_die_l521_521087


namespace ceiling_roll_is_2013_l521_521677

noncomputable def l : ℝ := 1 / 50
def o : ℝ := 2013

def count_o_num : ℕ := 34
def count_l_num : ℕ := 36
def count_o_den : ℕ := 33
def count_l_den : ℕ := 37

def numerator_value : ℝ := o ^ count_o_num * l ^ count_l_num
def denominator_value : ℝ := o ^ count_o_den * l ^ count_l_den
def r : ℝ := numerator_value / denominator_value
def roll_value : ℝ := r * l

theorem ceiling_roll_is_2013 : ⌈ roll_value ⌉ = 2013 := 
sorry

end ceiling_roll_is_2013_l521_521677


namespace seating_arrangement_l521_521022

theorem seating_arrangement : 
  ∃ n : ℕ, n = 120 ∧ ( ∀ (chairs people : ℕ),
  chairs = 8 → people = 4 → no_two_adjacent_people_sit (chairs people)) :=
begin
  sorry
end

end seating_arrangement_l521_521022


namespace Simson_line_parallel_AN_l521_521328

open EuclideanGeometry

variable (A B C M N : Point) (ABC : Triangle A B C)
variable (hM : M ∈ circumcircle ABC)
variable (h_perp : ⟨M, line_through M perpendicular BC⟩ ∈ circumcircle ABC)

theorem Simson_line_parallel_AN (ABC : Triangle A B C) (M N : Point)
  (hM : M ∈ circumcircle ABC)
  (h_perp : ⟨M, line_through M perpendicular BC⟩ ∈ circumcircle ABC) :
  Simson_line M ABC ∥ line_through A N := sorry

end Simson_line_parallel_AN_l521_521328


namespace count_T_le_1999_l521_521177

-- Define the sum of the digits of a positive integer x
def sum_digits (x : ℕ) : ℕ :=
  x.digits 10 |>.sum

-- Define T(x) as the absolute difference between sum of digits of (x + 2) and x
def T (x : ℕ) : ℕ :=
  |sum_digits (x + 2) - sum_digits x|

-- Define the condition that T(x) does not exceed 1999
def T_le_1999 (x : ℕ) : Prop :=
  T x ≤ 1999

-- The theorem statement to check that the number of values T(x) <= 1999 is exactly 222
theorem count_T_le_1999 : 
  (Finset.filter T_le_1999 (Finset.range 223)).card = 222 :=
by
  sorry

end count_T_le_1999_l521_521177


namespace decagon_adjacent_vertex_probability_l521_521553

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l521_521553


namespace lower_seat_tickets_l521_521498

theorem lower_seat_tickets (L U : ℕ) (h1 : L + U = 80) (h2 : 30 * L + 20 * U = 2100) : L = 50 :=
by
  sorry

end lower_seat_tickets_l521_521498


namespace problem_1_problem_2_problem_3_l521_521726

variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π)
variable (h2 : Real.tan α = -2)

theorem problem_1 : Real.sin (α + (π / 6)) = (2 * Real.sqrt 15 - Real.sqrt 5) / 10 := by
  sorry

theorem problem_2 : (2 * Real.cos ((π / 2) + α) - Real.cos (π - α)) / (Real.sin ((π / 2) - α) - 3 * Real.sin (π + α)) = 5 / 7 := by
  sorry

theorem problem_3 : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 11 / 5 := by
  sorry

end problem_1_problem_2_problem_3_l521_521726


namespace possible_integer_side_lengths_l521_521798

theorem possible_integer_side_lengths : 
  {x : ℤ | x > 3 ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521798


namespace integral_sin_cos_l521_521456

theorem integral_sin_cos : ∫ x in -Real.pi / 2 .. Real.pi / 2, (Real.sin x + Real.cos x) = 2 := 
by sorry

end integral_sin_cos_l521_521456


namespace min_value_a_plus_9b_l521_521907

noncomputable theory

open Real

theorem min_value_a_plus_9b 
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : a + b = 10 * a * b) :
  a + 9 * b ≥ 8 / 5 :=
begin
  sorry
end

example : ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ (a + b = 10 * a * b) ∧ (a + 9 * b = 8 / 5) :=
begin
  use [2/5, 2/15],
  split, linarith,
  split, linarith,
  split,
  { field_simp,
    norm_num },
  { simp }
end

end min_value_a_plus_9b_l521_521907


namespace dual_expr_result_solve_sqrt_eq_16_solve_sqrt_rational_eq_4x_l521_521474

-- Question 1
theorem dual_expr_result (m n : ℝ) (h1 : m = 2 - Real.sqrt 3) (h2 : n = 2 + Real.sqrt 3) :
  m * n = 1 :=
sorry

-- Question 2
theorem solve_sqrt_eq_16 (x : ℝ) (h : Real.sqrt (x + 42) + Real.sqrt (x + 10) = 16) :
  x = 39 :=
sorry

-- Question 3
theorem solve_sqrt_rational_eq_4x (x : ℝ) (h : Real.sqrt (4 * x^2 + 6 * x - 5) + Real.sqrt (4 * x^2 - 2 * x - 5) = 4 * x) :
  x = 3 :=
sorry

end dual_expr_result_solve_sqrt_eq_16_solve_sqrt_rational_eq_4x_l521_521474


namespace problem_I_problem_II_l521_521922

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

end problem_I_problem_II_l521_521922


namespace probability_greater_area_l521_521121

variables (ABC: Type ) [ABC : Real] 
variables (A B P: Point ABC)
variables (angle_BAC: 90∘ = Real)(angle_ABC:  90∘  = Real)(angle_BCA: 90∘  = Real)
variables (P: Triangle (A,B,C): Point)

def area_ABC:A: Point * B:Point * C:Point:= (1.0)/2 * (height_Triangle (hypotenous_BC)) 

theorem probability_greater_area (ABC: Triangle) (P: Point_inside_Triangle (ABC) ):calculate_probability ABC (area (ABC:-> area P )) = 1/4 
 Proof by
    eq'd by assumption 1. 

 sorry 

 Ignore_rotation_Of_Right_Angle 1. 
 sorry 

 rotation_angle_assumed triangle ABC  bc==90 ∘
 sorry 
   
calculation_of_probability  3/2  of xyz  ==1/4  

 sorry 

 #QED    


ignore_rotation_Of_Right_Angle:

where,
 height of A,B = Base*height  

end probability_greater_area_l521_521121


namespace odd_function_def_l521_521756

noncomputable def f (x : ℝ) : ℝ :=
if hx : x > 0 then x * (1 - x)
else if -x > 0 then -x * (1 + x)
else 0  -- This part won't actually be used given the domain constraints

theorem odd_function_def {x : ℝ} (h : x < 0) :
  f x = x * (1 + x) :=
by
  have h_neg : -x > 0 := by linarith
  have h_odd : f x = -f (-x) := sorry  -- using the property that f is odd
  rw [←h_odd, f]
  split_ifs
  · sorry  -- this case should handle when x > 0, not applicable here
  · have h_pos : 0 < -x := by linarith
    rw [if_pos h_pos]
    sorry  -- need to conclude f (-x) = -x * (1 + x)
  · exfalso; linarith

end odd_function_def_l521_521756


namespace range_of_first_person_l521_521104

variable (R1 R2 R3 : ℕ)
variable (min_range : ℕ)
variable (condition1 : min_range = 25)
variable (condition2 : R2 = 25)
variable (condition3 : R3 = 30)
variable (condition4 : min_range ≤ R1 ∧ min_range ≤ R2 ∧ min_range ≤ R3)

theorem range_of_first_person : R1 = 25 :=
by
  sorry

end range_of_first_person_l521_521104


namespace QR_distance_l521_521309

noncomputable def AB : ℝ := 15
noncomputable def BC : ℝ := 20
noncomputable def height_P : ℝ := 30
noncomputable def volume_ratio : ℝ := 3

theorem QR_distance 
  (AB_def : AB = 15)
  (BC_def : BC = 20)
  (height_P_def : height_P = 30)
  (volume_ratio_def : volume_ratio = 3) :
  let h_P' := height_P / (volume_ratio ^ (1 / 3))
  let QR := height_P - h_P'
  QR = 9.2 :=
by 
  let h_P' := height_P / (volume_ratio ^ (1 / 3))
  let QR := height_P - h_P'
  have h_P'_approx : h_P' ≈ 20.8 := sorry
  have QR_approx : QR ≈ 9.2 := sorry
  exact QR_approx

end QR_distance_l521_521309


namespace range_of_AC_l521_521866

-- Define the triangle with sides AB, BC, and requiring angle B to be obtuse
variables (A B C : Type) [ordered_field A]
variables (AB AC BC : A)
variables (angle_B : Type) 
variables [lt : has_lt angle_B]

-- Given conditions
def is_obtuse (angle : angle_B) := sorry -- Definition of an obtuse angle
def is_triangle (a b c : A) := (a + b > c) ∧ (b + c > a) ∧ (c + a > b)
def triangle_AB_CB := is_triangle AB AC BC

-- Define given values
def AB := (6 : ℝ)
def BC := (8 : ℝ)
def angle_B_obtuse := is_obtuse angle_B

-- Define target to prove
theorem range_of_AC (AC : A) : (10 < AC) ∧ (AC < 14) :=
by {
  -- Here we will need to prove the necessary inequalities for AC based on the given conditions.
  sorry 
}

end range_of_AC_l521_521866


namespace sum_alternating_series_l521_521691

theorem sum_alternating_series : (Finset.range 200).sum (λ n, if even n then -(n + 1) else n + 1) = 100 :=
by
  -- Proof goes here
  sorry

end sum_alternating_series_l521_521691


namespace odd_function_ab_l521_521750

theorem odd_function_ab {f : ℝ → ℝ} (hf_odd : ∀ x, f (-x) = -f x)
  (hf : ∀ x ≥ 0, f x = 2 * x - x^2)
  (a b : ℝ)
  (h_range : set.Icc (1 / b) (1 / a) = set.range (λ x, f x ∣ set.Icc a b)) :
  a * b = (1 + Real.sqrt 5) / 2 := 
sorry

end odd_function_ab_l521_521750


namespace not_possible_to_create_3_piles_l521_521361

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end not_possible_to_create_3_piles_l521_521361


namespace expected_value_fair_dodecahedral_die_l521_521079

theorem expected_value_fair_dodecahedral_die : 
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  let probability := (1 : ℝ) / 12 in
  (probability * faces.sum) = 6.5 :=
by
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let probability := (1 : ℝ) / 12
  have h : faces.sum = 78 := rfl
  rw [h]
  sorry

end expected_value_fair_dodecahedral_die_l521_521079


namespace probability_adjacent_vertices_decagon_l521_521598

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l521_521598


namespace smallest_possible_sum_l521_521430

-- Define variables and conditions
variable (A B C D E F : ℕ)
variable (adjacent : (ℕ × ℕ → Prop))
variable (cube_faces : List ℕ := [A, B, C, D, E, F])

-- Condition: Natural numbers on the faces of a cube such that the numbers 
-- on adjacent faces differ by more than 1
def valid_faces (n1 n2 : ℕ) : Prop :=
  (n1 ≠ n2) ∧ (abs (n1 - n2) > 1)

-- Condition: Define adjacency relation in cube
def cube_adjacency : (ℕ × ℕ → Prop) := λ (n1 n2 : ℕ), match (n1, n2) with
  | (A, B), (A, C), (A, D), (B, E), (C, F), (D, E), (E, F) => valid_faces n1 n2
  | _ => false

theorem smallest_possible_sum : ∃ face_sum : ℕ, 
  all (λ (pair : ℕ × ℕ), adjacent pair) (filter cube_adjacency (list.nat.pairs cube_faces))
  ∧ face_sum = (A + B + C + D + E + F) ∧ face_sum = 18 :=
sorry

end smallest_possible_sum_l521_521430


namespace juliet_supporters_in_capulet_rate_l521_521608

theorem juliet_supporters_in_capulet_rate :
  ∀ (P : ℕ),
  ((6 / 9) * P) * 0.8 = (16 / 30) * P ∧
  ((1 / 3) * P) * 0.7 = (7 / 30) * P ∧
  ((6 / 9) * P) * 0.2 = (4 / 30) * P ∧
  ((7 / 30) * P + (4 / 30) * P) = (11 / 30) * P →
  ((7 / 30) * P / (11 / 30) * P) * 100 = 64 :=
by
  sorry

end juliet_supporters_in_capulet_rate_l521_521608


namespace same_centroid_l521_521305

variables {A B C H O D P : Type} [EuclideanGeometry A B C H O D P]

-- Given conditions
axiom tri : Triangle A B C
axiom ab_ne_ac : A ≠ B ∧ B ≠ C ∧ A ≠ C
axiom orthocenter : Orthocenter H A B C 
axiom circumcenter : Circumcenter O A B C 
axiom midpoint : Midpoint D B C
axiom extensions_meet : ∃ P, LineThrough H D ∧ LineThrough A O ∧ Intersect (LineThrough H D) (LineThrough A O) P

-- Prove that triangles AHP and ABC have the same centroid
theorem same_centroid : ∃ G, Centroid G A H P ∧ Centroid G A B C :=
by sorry

end same_centroid_l521_521305


namespace correct_choice_of_propositions_l521_521189

variables (m n a b : Line) (α β : Plane)
variables (h1 : m ≠ n) (h2 : α ≠ β)
variables (hp1 : isPerpendicular a α) (hp2 : isPerpendicular b β)

theorem correct_choice_of_propositions : 
  (if (isPerpendicular m α ∧ isParallel n b ∧ isPerpendicular α β) then ¬isParallel m n else true) ∧
  (if (isParallel m a ∧ isParallel n b ∧ isPerpendicular α β) then isPerpendicular m n else true) ∧
  (if (isParallel m α ∧ isParallel n b ∧ isParallel α β) then isPerpendicular m n else true) ∧
  (if (isPerpendicular m α ∧ isPerpendicular n b ∧ isPerpendicular α β) then ¬isParallel m n else true) :=
by 
  sorry

end correct_choice_of_propositions_l521_521189


namespace exist_distinct_indices_l521_521666

theorem exist_distinct_indices (n : ℕ) (h : even (2 * n)) (π : equiv.perm (fin (2 * n))) :
  ∃ (i j : fin (2 * n)), i ≠ j ∧ (|π i - π j| % (2 * n) = |i - j| % (2 * n)) :=
begin
  sorry
end

end exist_distinct_indices_l521_521666


namespace min_sum_on_faces_ge_16_l521_521401

def cube_vertices : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def faces := { {a, b, c, d} : set ℕ | a ∈ cube_vertices ∧ b ∈ cube_vertices ∧ c ∈ cube_vertices ∧ d ∈ cube_vertices ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d }

theorem min_sum_on_faces_ge_16 :
  (∀ F ∈ faces, (∀ a b c ∈ F, a + b + c ≥ 10) → (∀ a b c d ∈ F, a + b + c + d ≥ 16)) :=
by
  intros F hF habc_10 a b c d ha hb hc hd
  -- Additional steps to prove it will be placed here.
  sorry

end min_sum_on_faces_ge_16_l521_521401


namespace circle_radius_squared_l521_521624

theorem circle_radius_squared (r : ℝ) 
  (AB CD: ℝ) 
  (BP angleAPD : ℝ) 
  (P_outside_circle: True) 
  (AB_eq_12 : AB = 12) 
  (CD_eq_9 : CD = 9) 
  (AngleAPD_eq_45 : angleAPD = 45) 
  (BP_eq_10 : BP = 10) : r^2 = 73 :=
sorry

end circle_radius_squared_l521_521624


namespace triangle_possible_sides_l521_521783

theorem triangle_possible_sides : 
  let unknown_side_lengths := {x : ℤ | 3 < x ∧ x < 13}
  in set.finite unknown_side_lengths ∧ set.card unknown_side_lengths = 9 :=
by
  sorry

end triangle_possible_sides_l521_521783


namespace min_value_of_function_on_interval_l521_521715

noncomputable def function_min : ℝ := (-1:ℝ) / 4

theorem min_value_of_function_on_interval :
  ∃ x ∈ set.Icc (0:ℝ) (1:ℝ), ∀ y ∈ set.Icc (0:ℝ) (1:ℝ), (x^2 - x) ≤ y^2 - y :=
begin
  have h : x ∈ set.Icc (0:ℝ) (1:ℝ) → (x^2 - x) ∈ set.Icc (-(1:ℝ)/4) (0:ℝ),
  sorry,
end

end min_value_of_function_on_interval_l521_521715


namespace calculation_correct_l521_521144

theorem calculation_correct : 
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by
  sorry

end calculation_correct_l521_521144


namespace mx_mt_sqrt2_l521_521645

-- Definitions derived directly from conditions
variables {R : ℝ} -- Radius of the circle
def O := (0 : ℝ, 0 : ℝ)  -- Center of the circle
def M := (R/2, R/2)      -- Midpoint of one of the side of the square
def X (α : ℝ) := (R * cos α, R * sin α)  -- Arbitrary point on the circle
def T (α : ℝ) := ((R * cos α) / 2, (R * sin α) / 2)  -- Midpoint of O and X

-- The proof problem statement
theorem mx_mt_sqrt2 (α : ℝ) : 
  let MX := dist M (X α) in
  let MT := dist M (T α) in
  MX / MT = sqrt 2 := 
sorry

end mx_mt_sqrt2_l521_521645


namespace birdhouse_planks_l521_521141

def cost_of_nail : ℝ := 0.05
def cost_of_plank : ℝ := 3
def nails_per_birdhouse : ℕ := 20
def total_cost_of_four_birdhouses : ℝ := 88

def num_planks_per_birdhouse : ℝ := 7 

theorem birdhouse_planks :
  ∃ P : ℝ, P = num_planks_per_birdhouse ∧ 
  (4 * (nails_per_birdhouse * cost_of_nail + P * cost_of_plank) = total_cost_of_four_birdhouses) :=
begin
  use num_planks_per_birdhouse,
  split,
  { refl },
  { sorry }
end

end birdhouse_planks_l521_521141


namespace cubic_roots_fraction_l521_521693

theorem cubic_roots_fraction 
  (a b c d : ℝ)
  (h_eq : ∀ x: ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ (x = -2 ∨ x = 3 ∨ x = 4)) :
  c / d = -1 / 12 :=
by
  sorry

end cubic_roots_fraction_l521_521693


namespace decagon_adjacent_vertex_probability_l521_521554

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l521_521554


namespace burritos_in_each_box_l521_521296

theorem burritos_in_each_box (B : ℕ) (h1 : 3 * B - B - 30 = 10) : B = 20 :=
by
  sorry

end burritos_in_each_box_l521_521296


namespace problem_solution_l521_521937

open Real

-- Define the constants and conditions
variables (a b c : ℝ)
variable ha : 0 < a ∧ a < π / 2 ∧ a = cos a
variable hb : 0 < b ∧ b < π / 2 ∧ b = sin (cos b)
variable hc : 0 < c ∧ c < π / 2 ∧ c = cos (sin c)

-- Statement to prove
theorem problem_solution : b < a ∧ a < c :=
by
  -- Proof omitted
  sorry

end problem_solution_l521_521937


namespace probability_adjacent_vertices_of_decagon_l521_521508

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l521_521508


namespace triangle_side_count_l521_521830

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end triangle_side_count_l521_521830


namespace part_a_part_b_l521_521778

noncomputable def f (x A : ℝ) := (√3 * A * sin x * cos x + (A / 2) * cos (2 * x))

noncomputable def g (x : ℝ) := 6 * sin (4 * x + π / 3)

theorem part_a (A : ℝ) (hA : 0 < A) (hf_max : ∀ x, f x A ≤ 6) : 
  A = 6 ∧
  (∀ k: ℤ, x = π / 6 + k * (π / 2)) ∧
  (∀ k: ℤ, ∃ y, ( x = -π / 12 + k * (π / 2) ∧ y = 0)) :=
begin
  sorry
end

theorem part_b (x : ℝ) : 
  ∀ x ∈ set.Icc 0 (5 * π / 24), -3 ≤ g x ∧ g x ≤ 6 :=
begin 
  sorry 
end

end part_a_part_b_l521_521778


namespace decagon_adjacent_probability_l521_521518

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521518


namespace percent_round_trip_ticket_l521_521395

def total_passengers := 100
def percent_with_round_trip_ticket_and_car := 0.40
def percent_without_car_given_round_trip_ticket := 0.50

theorem percent_round_trip_ticket :
  let passengers_with_car := total_passengers * percent_with_round_trip_ticket_and_car in
  let total_round_trip_passengers := passengers_with_car * 2 in
  total_round_trip_passengers / total_passengers = 0.80 :=
by
  sorry

end percent_round_trip_ticket_l521_521395


namespace circle_intersection_area_l521_521976

theorem circle_intersection_area
  (r : ℝ)
  (θ : ℝ)
  (a b c : ℝ)
  (h_r : r = 5)
  (h_θ : θ = π / 2)
  (h_expr : a * Real.sqrt b + c * π = 5 * 5 * π / 2 - 5 * 5 * Real.sqrt 3 / 2 ) :
  a + b + c = -9.5 :=
by
  sorry

end circle_intersection_area_l521_521976


namespace whatsapp_group_messages_l521_521615

theorem whatsapp_group_messages :
  let messages_monday := 300 in
  let messages_tuesday := 200 in
  let messages_wednesday := messages_tuesday + 300 in
  let messages_thursday := 2 * messages_wednesday in
  messages_monday + messages_tuesday + messages_wednesday + messages_thursday = 2000 :=
by
  -- Definitions from the conditions
  let messages_monday := 300
  let messages_tuesday := 200
  let messages_wednesday := messages_tuesday + 300
  let messages_thursday := 2 * messages_wednesday
  -- Summing the messages
  have total_messages : messages_monday + messages_tuesday + messages_wednesday + messages_thursday = 2000 := by
    rw [messages_wednesday, messages_thursday]
    simp
  exact total_messages

end whatsapp_group_messages_l521_521615


namespace students_taking_either_but_not_both_l521_521710

-- Definitions to encapsulate the conditions
def students_taking_both : ℕ := 15
def students_taking_mathematics : ℕ := 30
def students_taking_history_only : ℕ := 12

-- The goal is to prove the number of students taking mathematics or history but not both
theorem students_taking_either_but_not_both
  (hb : students_taking_both = 15)
  (hm : students_taking_mathematics = 30)
  (hh : students_taking_history_only = 12) : 
  students_taking_mathematics - students_taking_both + students_taking_history_only = 27 :=
by
  sorry

end students_taking_either_but_not_both_l521_521710


namespace find_extrema_l521_521444

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

theorem find_extrema :
  (∀ x, f x ≤ 17) ∧ (∃ x, f x = 17) ∧ (∀ x, f x ≥ -47) ∧ (∃ x, f x = -47) :=
by
  sorry

end find_extrema_l521_521444


namespace incircle_radius_triangle_DEF_l521_521890

theorem incircle_radius_triangle_DEF
  (D E F : Type)
  (angle_D : ∠ D = 90)
  (angle_E : ∠ E = 45)
  (DE : length D E = 14) : 
  incircle_radius (Δ D E F) = 7 * (1 - Real.sqrt 2) := by
  sorry

end incircle_radius_triangle_DEF_l521_521890


namespace hotdogs_left_l521_521779

theorem hotdogs_left (h_hotdogs: ℕ) (d_hotdogs: ℕ) (h_give_frac: ℚ) (d_give_frac: ℚ) (h_left: ℕ) (d_left: ℕ) :
  h_hotdogs = 101 ∧ d_hotdogs = 379 ∧ h_give_frac = 1/3 ∧ d_give_frac = 15/100 ∧ 
  h_left = h_hotdogs - ⌊h_give_frac * h_hotdogs⌋ ∧ d_left = d_hotdogs - ⌊d_give_frac * d_hotdogs⌋ →
  h_left + d_left = 391 :=
begin
  sorry
end

end hotdogs_left_l521_521779


namespace impossible_to_create_3_piles_l521_521376

def similar_sizes (x y : ℝ) : Prop := x ≤ sqrt 2 * y

theorem impossible_to_create_3_piles (x y : ℝ) (h₁ : similar_sizes x y) :
  ¬ ∃ (x₁ x₂ x₃ : ℝ), x₁ + x₂ + x₃ = x + y ∧ similar_sizes x₁ x₂ ∧ similar_sizes x₂ x₃ ∧ similar_sizes x₁ x₃ :=
by
  sorry

end impossible_to_create_3_piles_l521_521376


namespace table_can_be_zeroed_out_l521_521878

open Matrix

-- Define the dimensions of the table
def m := 8
def n := 5

-- Define the operation of doubling all elements in a row
def double_row (table : Matrix (Fin m) (Fin n) ℕ) (i : Fin m) : Matrix (Fin m) (Fin n) ℕ :=
  fun i' j => if i' = i then 2 * table i' j else table i' j

-- Define the operation of subtracting one from all elements in a column
def subtract_one_column (table : Matrix (Fin m) (Fin n) ℕ) (j : Fin n) : Matrix (Fin m) (Fin n) ℕ :=
  fun i j' => if j' = j then table i j' - 1 else table i j'

-- The main theorem stating that it is possible to transform any table to a table of all zeros
theorem table_can_be_zeroed_out (table : Matrix (Fin m) (Fin n) ℕ) : 
  ∃ (ops : List (Matrix (Fin m) (Fin n) ℕ → Matrix (Fin m) (Fin n) ℕ)), 
    (ops.foldl (fun t op => op t) table) = fun _ _ => 0 :=
sorry

end table_can_be_zeroed_out_l521_521878


namespace probability_adjacent_vertices_in_decagon_l521_521567

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l521_521567


namespace decagon_adjacent_probability_l521_521536

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l521_521536


namespace semicircle_in_square_max_diameter_l521_521642

theorem semicircle_in_square_max_diameter :
  ∃ (m n : ℕ), (∃ d : ℝ, d = m - real.sqrt n ∧ (d / 2 * (1 + real.sqrt 2 / 2)) = 8) ∧ (m + n = 544) :=
by
  sorry

end semicircle_in_square_max_diameter_l521_521642


namespace staffing_arrangements_count_l521_521023

theorem staffing_arrangements_count :
  let grades := {10, 11, 12}
  let num_judges_per_grade := 2
  let num_courts := 3
  let different_grades (g1 g2 : ℕ) : Prop := g1 ≠ g2
  -- Computation based on problem constraints and conditions
  num_judges_per_grade ^ num_courts * num_courts.factorial = 48 :=
by
  let grades := {10, 11, 12}
  let num_judges_per_grade := 2
  let num_courts := 3
  let different_grades (g1 g2 : ℕ) : Prop := g1 ≠ g2
  have h1 : num_judges_per_grade ^ num_courts = 8, from by sorry -- placeholder for actual calculation
  have h2 : num_courts.factorial = 6, from by sorry  -- placeholder for actual calculation
  show 8 * 6 = 48, by sorry -- placeholder for arithmetic result

end staffing_arrangements_count_l521_521023


namespace min_ab_value_l521_521192

-- Given conditions
variable (b a : ℝ)
axiom b_gt_zero : b > 0
axiom lines_perpendicular : ∃ a : ℝ, ∀ b > 0, (b^2 + 1) * 1 + a * (-b^2) = 0

-- The theorem statement
theorem min_ab_value : ∃ ab_min, ab_min = 2 ∧ ∀ (a b : ℝ), b > 0 ∧ 
((b^2 + 1) * 1 + a * (-b^2) = 0) → (a * b) ≥ ab_min := 
by 
  use 2
  intros a b h1 h2
  sorry

end min_ab_value_l521_521192


namespace negation_of_exists_l521_521863

theorem negation_of_exists (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l521_521863


namespace triangle_possible_sides_l521_521781

theorem triangle_possible_sides : 
  let unknown_side_lengths := {x : ℤ | 3 < x ∧ x < 13}
  in set.finite unknown_side_lengths ∧ set.card unknown_side_lengths = 9 :=
by
  sorry

end triangle_possible_sides_l521_521781


namespace smallest_n_three_distinct_primes_l521_521486

theorem smallest_n_three_distinct_primes (n p q r : ℕ) (hp : p.prime) (hq : q.prime) (hr : r.prime) 
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) (hn : n = p * q * r) :
  (¬ (∃ k : ℤ, k = (1 + p + q + r + p*q + p*r + q*r + p*q*r) / 8)) → n = 130 := 
sorry

end smallest_n_three_distinct_primes_l521_521486


namespace triangle_angles_l521_521450

-- Define the sides of the triangle
def side1 : ℝ := 3
def side2 : ℝ := 3
def side3 : ℝ := Real.sqrt 8

-- Define the angles of the triangle
axiom angle1 : ℝ
axiom angle2 : ℝ
axiom angle3 : ℝ
axiom angle_sum : angle1 + angle2 + angle3 = 180

noncomputable def cos_angle1 := (side1 ^ 2 + side2 ^ 2 - side3 ^ 2) / (2 * side1 * side2)
noncomputable def angle1_rad := Real.arccos cos_angle1

-- Convert radians to degrees
noncomputable def rad_to_deg (rad : ℝ) : ℝ := rad * (180 / Real.pi)
noncomputable def angle1_deg := rad_to_deg angle1_rad

axiom iso_triangle : side1 = side2
axiom remaining_angles : angle1_deg + 2 * angle2 = 180
axiom angle2_deg : angle2 = angle3
noncomputable def angle2_deg_val := (180 - angle1_deg) / 2

-- Calculate angle values
theorem triangle_angles :
  angle1_deg = (Real.toFloat 56.25) ∧
  angle2_deg_val = (Real.toFloat 61.875) ∧
  angle2_deg = angle2 :=
by
  sorry

end triangle_angles_l521_521450


namespace minimum_pie_pieces_l521_521441

theorem minimum_pie_pieces (p q : ℕ) (h_coprime : Nat.gcd p q = 1) : 
  ∃ n, (∀ k, k = p ∨ k = q → (n ≠ 0 → n % k = 0)) ∧ n = p + q - 1 :=
by {
  sorry
}

end minimum_pie_pieces_l521_521441


namespace innings_count_l521_521617

-- Definitions of the problem conditions
def total_runs (n : ℕ) : ℕ := 63 * n
def highest_score : ℕ := 248
def lowest_score : ℕ := 98

theorem innings_count (n : ℕ) (h : total_runs n - highest_score - lowest_score = 58 * (n - 2)) : n = 46 :=
  sorry

end innings_count_l521_521617


namespace prime_product_inequality_l521_521307

open Nat

theorem prime_product_inequality (n : ℕ) (h : n ≥ 10) : 
    let p := fun i : ℕ => nthPrime i
    π (sqrt (p 1 * p 2 * p 3 * p 4 * p 5 * p 6 * p 7 * p 8 * p 9 * p 10 * ... * p n)) > 2 * n :=
sorry

end prime_product_inequality_l521_521307


namespace probability_both_in_picture_l521_521946

-- Define the conditions
def completes_lap (laps_time: ℕ) (time: ℕ) : ℕ := time / laps_time

def position_into_lap (laps_time: ℕ) (time: ℕ) : ℕ := time % laps_time

-- Define the positions of Rachel and Robert
def rachel_position (time: ℕ) : ℚ :=
  let rachel_lap_time := 100
  let laps_completed := completes_lap rachel_lap_time time
  let time_into_lap := position_into_lap rachel_lap_time time
  (laps_completed : ℚ) + (time_into_lap : ℚ) / rachel_lap_time

def robert_position (time: ℕ) : ℚ :=
  let robert_lap_time := 70
  let laps_completed := completes_lap robert_lap_time time
  let time_into_lap := position_into_lap robert_lap_time time
  (laps_completed : ℚ) + (time_into_lap : ℚ) / robert_lap_time

-- Define the probability that both are in the picture
theorem probability_both_in_picture :
  let rachel_lap_time := 100
  let robert_lap_time := 70
  let start_time := 720
  let end_time := 780
  ∃ (overlap_time: ℚ) (total_time: ℚ),
    overlap_time / total_time = 1 / 16 :=
sorry

end probability_both_in_picture_l521_521946


namespace probability_adjacent_vertices_in_decagon_l521_521561

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l521_521561


namespace cents_saved_l521_521442

def in_store_price := 120.00
def installment_price := 29.75
def num_installments := 4
def installation_fee := 2.00

noncomputable def total_online_price := (installment_price * num_installments) + installation_fee
noncomputable def savings := (in_store_price - total_online_price) * 100

theorem cents_saved : savings = 0 := by
  sorry

end cents_saved_l521_521442


namespace grandma_olga_grandchildren_l521_521236

theorem grandma_olga_grandchildren :
  let daughters := 3 in
  let sons := 3 in
  let grandsons_per_daughter := 6 in
  let granddaughters_per_son := 5 in
  let total_grandsons := daughters * grandsons_per_daughter in
  let total_granddaughters := sons * granddaughters_per_son in
  total_grandsons + total_granddaughters = 33 :=
by
  sorry

end grandma_olga_grandchildren_l521_521236


namespace g_odd_find_a_f_increasing_l521_521769

-- Problem (I): Prove that if g(x) = f(x) - a is an odd function, then a = 1, given f(x) = 1 - 2/x.
theorem g_odd_find_a (f : ℝ → ℝ) (g : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = 1 - (2 / x)) → 
  (∀ x, g x = f x - a) → 
  (∀ x, g (-x) = - g x) → 
  a = 1 := 
  by
  intros h1 h2 h3
  sorry

-- Problem (II): Prove that f(x) is monotonically increasing on (0, +∞),
-- given f(x) = 1 - 2/x.

theorem f_increasing (f : ℝ → ℝ) : 
  (∀ x, f x = 1 - (2 / x)) → 
  ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → f x1 < f x2 := 
  by
  intros h1 x1 x2 hx1 hx12
  sorry

end g_odd_find_a_f_increasing_l521_521769


namespace not_possible_three_piles_l521_521381

def similar (x y : ℝ) : Prop :=
  y ≤ (real.sqrt 2) * x ∧ x ≤ (real.sqrt 2) * y

theorem not_possible_three_piles (N : ℝ) : 
  ¬ ∃ a b c : ℝ, a + b + c = N ∧ similar a b ∧ similar b c ∧ similar a c := 
by
  sorry

end not_possible_three_piles_l521_521381


namespace find_value_of_expression_l521_521214

open Real

theorem find_value_of_expression (x y z w : ℝ) (h1 : x + y + z + w = 0) (h2 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end find_value_of_expression_l521_521214


namespace bisection_program_flowchart_l521_521448

-- Define the problem using bisection method as a condition
constant bisection_method : (ℝ → ℝ) → (ℝ × ℝ) → (ℝ × ℝ)
constant uses_loop_structure : Bool

-- Define the function for the given equation
def f (x : ℝ) : ℝ := x^2 - 2

-- Given that the algorithm uses a loop structure
axiom bisection_uses_loop : uses_loop_structure = true

-- The theorem to prove that the result is a program flowchart
theorem bisection_program_flowchart : 
  (∀ a b : ℝ, uses_loop_structure = true → 
  (bisection_method f (a, b)) represents program_flowchart) := 
by
  sorry

end bisection_program_flowchart_l521_521448


namespace probability_adjacent_vertices_of_decagon_l521_521512

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l521_521512


namespace find_m_l521_521170

theorem find_m (m : ℤ) (h0 : -90 ≤ m) (h1 : m ≤ 90) (h2 : Real.sin (m * Real.pi / 180) = Real.sin (710 * Real.pi / 180)) : m = -10 :=
sorry

end find_m_l521_521170


namespace sequence_geometric_and_lambda_range_l521_521739

theorem sequence_geometric_and_lambda_range:
  (∃ (a : ℕ → ℤ), a 1 = 0 ∧ (∀ n : ℕ, n ≥ 1 → a (n+1) - 2 * a n = 2 * n)) →
  (∃ n : ℕ, n ≥ 1 → a n = 2^(n+1) - 2*n - 2) ∧ 
  (∀ λ : ℚ, (∀ n : ℕ, n ≥ 1 → a n > λ * (2*n + 1) * (-1)^(n-1)) →
    -2/5 < λ ∧ λ < 0) :=
by
  sorry

end sequence_geometric_and_lambda_range_l521_521739


namespace negation_of_universal_quadratic_l521_521445

theorem negation_of_universal_quadratic (P : ∀ a b c : ℝ, a ≠ 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0) :
  ¬(∀ a b c : ℝ, a ≠ 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0) ↔ ∃ a b c : ℝ, a ≠ 0 ∧ ¬(∃ x : ℝ, a * x^2 + b * x + c = 0) :=
by
  sorry

end negation_of_universal_quadratic_l521_521445


namespace decagon_adjacent_vertex_probability_l521_521548

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l521_521548


namespace smallest_GCD_value_l521_521009

theorem smallest_GCD_value (a b c d N : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
    (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : N > 5)
    (hc1 : Nat.gcd a b = 1 ∨ Nat.gcd a c = 1 ∨ Nat.gcd a d = 1 ∨ Nat.gcd b c = 1 ∨ Nat.gcd b d = 1 ∨ Nat.gcd c d = 1)
    (hc2 : Nat.gcd a b = 2 ∨ Nat.gcd a c = 2 ∨ Nat.gcd a d = 2 ∨ Nat.gcd b c = 2 ∨ Nat.gcd b d = 2 ∨ Nat.gcd c d = 2)
    (hc3 : Nat.gcd a b = 3 ∨ Nat.gcd a c = 3 ∨ Nat.gcd a d = 3 ∨ Nat.gcd b c = 3 ∨ Nat.gcd b d = 3 ∨ Nat.gcd c d = 3)
    (hc4 : Nat.gcd a b = 4 ∨ Nat.gcd a c = 4 ∨ Nat.gcd a d = 4 ∨ Nat.gcd b c = 4 ∨ Nat.gcd b d = 4 ∨ Nat.gcd c d = 4)
    (hc5 : Nat.gcd a b = 5 ∨ Nat.gcd a c = 5 ∨ Nat.gcd a d = 5 ∨ Nat.gcd b c = 5 ∨ Nat.gcd b d = 5 ∨ Nat.gcd c d = 5)
    (hcN : Nat.gcd a b = N ∨ Nat.gcd a c = N ∨ Nat.gcd a d = N ∨ Nat.gcd b c = N ∨ Nat.gcd b d = N ∨ Nat.gcd c d = N):
    N = 14 :=
sorry

end smallest_GCD_value_l521_521009


namespace find_h3_l521_521303

def f (x: ℝ) : ℝ := 3 * x + 6
def g (x: ℝ) : ℝ := (Real.sqrt (f x) - 3) ^ 2
def h (x: ℝ) : ℝ := f (g x)

theorem find_h3 : h 3 = 72 - 18 * Real.sqrt 15 := by
  sorry

end find_h3_l521_521303


namespace apex_angle_of_cone_equals_pi_div_3_l521_521861

noncomputable def apex_angle_of_cone (R : ℝ) : ℝ :=
  if h : R > 0 then π / 3 else 0

theorem apex_angle_of_cone_equals_pi_div_3 (R : ℝ) (hR : R > 0) : 
  apex_angle_of_cone R = π / 3 :=
by
  unfold apex_angle_of_cone
  rw if_pos hR
  sorry

end apex_angle_of_cone_equals_pi_div_3_l521_521861


namespace auction_starting_price_l521_521135

-- Defining the conditions
def bid_increment := 5         -- The dollar increment per bid
def bids_per_person := 5       -- Number of bids per person
def total_bidders := 2         -- Number of people bidding
def final_price := 65          -- Final price of the desk after all bids

-- Calculate derived conditions
def total_bids := bids_per_person * total_bidders
def total_increment := total_bids * bid_increment

-- The statement to be proved
theorem auction_starting_price : (final_price - total_increment) = 15 :=
by
  sorry

end auction_starting_price_l521_521135


namespace rectangle_constant_k_l521_521974

theorem rectangle_constant_k (d : ℝ) (x : ℝ) (h_ratio : 4 * x = (4 / 3) * (3 * x)) (h_diagonal : d^2 = (4 * x)^2 + (3 * x)^2) : 
  ∃ k : ℝ, k = 12 / 25 ∧ (4 * x) * (3 * x) = k * d^2 := 
sorry

end rectangle_constant_k_l521_521974


namespace decagon_adjacent_probability_l521_521582

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521582


namespace triangle_possible_sides_l521_521787

theorem triangle_possible_sides : 
  let unknown_side_lengths := {x : ℤ | 3 < x ∧ x < 13}
  in set.finite unknown_side_lengths ∧ set.card unknown_side_lengths = 9 :=
by
  sorry

end triangle_possible_sides_l521_521787


namespace count_odd_k_with_two_solutions_l521_521722

theorem count_odd_k_with_two_solutions :
  let count := finset.filter (λ k, (exists_eq_two_pairs ((2^(4*m^2) + 2^(m^2 - n^2 + 4) = 2^(k+4) + 2^(3*m^2 + n^2 + k)))) (finset.range' 1 100) in
  count.card = 18 :=
by
  sorry

#check exists_eq_two_pairs

end count_odd_k_with_two_solutions_l521_521722


namespace num_possible_triangle_sides_l521_521807

theorem num_possible_triangle_sides (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (a_eq_8 : a = 8) (b_eq_5 : b = 5) : 
  let x := {n : ℕ | 3 < n ∧ n < 13} in
  set.card x = 9 :=
by
  sorry

end num_possible_triangle_sides_l521_521807


namespace ratio_of_substances_l521_521732

noncomputable def amount_of_substance_ratio (H2O NH3 CH4 : ℝ) : (ℝ × ℝ × ℝ) :=
if (2 * H2O = 1) ∧ (3 * NH3 = 1) ∧ (4 * CH4 = 1) then
    (6, 4, 3)
else
    (0, 0, 0)

theorem ratio_of_substances : 
    ∀ (H2O NH3 CH4 : ℝ),
    (2 * H2O = 1) ∧ (3 * NH3 = 1) ∧ (4 * CH4 = 1) →
    amount_of_substance_ratio H2O NH3 CH4 = (6, 4, 3) :=
by
  intros H2O NH3 CH4 h
  rw [← h.1, ← h.2.1, ← h.2.2]
  sorry

end ratio_of_substances_l521_521732


namespace adjugate_matrix_squared_tr_eq_l521_521920

-- Define the problem
theorem adjugate_matrix_squared_tr_eq {n : ℕ} (hn : n ≥ 2) (A : Matrix (Fin n) (Fin n) ℂ) (hdet : A.det = 0) :
  let A_star := A.adjugate in
  (A_star * A_star) = A_star * A_star.trace :=
by
  -- no proof is required
  sorry

end adjugate_matrix_squared_tr_eq_l521_521920


namespace drum_capacity_ratio_l521_521706

variable {C_X C_Y : ℝ}

theorem drum_capacity_ratio (h1 : C_X / 2 + C_Y / 2 = 3 * C_Y / 4) : C_Y / C_X = 2 :=
by
  have h2: C_X / 2 = C_Y / 4 := by
    sorry
  have h3: C_X = C_Y / 2 := by
    sorry
  rw [h3]
  have h4: C_Y / (C_Y / 2) = 2 := by
    sorry
  exact h4

end drum_capacity_ratio_l521_521706


namespace complex_number_expression_l521_521326

noncomputable def compute_expression (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1)

theorem complex_number_expression (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  compute_expression r h1 h2 = 5 :=
sorry

end complex_number_expression_l521_521326


namespace sin_theta_value_l521_521185

theorem sin_theta_value 
  (θ : ℝ)
  (h1 : Real.sin θ + Real.cos θ = 7/5)
  (h2 : Real.tan θ < 1) :
  Real.sin θ = 3/5 :=
sorry

end sin_theta_value_l521_521185


namespace box_dimensions_l521_521640

theorem box_dimensions (a b c : ℕ) (h1 : a * b * c = a' * b' * c' : ℕ)
  (h2 : ∀ k : ℕ, ∃ a' b' c', a' * b' * c' = 2 ^ k / 5 ^ k)
  (h3 : ∀ x : ℕ, x > 8 → (x' * nat.sqrt 2) / x ^ 3  > 0.4)
  (h4 : a = 2 ∨ b = 5 ∨ c = 5) : 
  (a = 2 ∧ b = 3 ∧ c = 5) ∨ (a = 2 ∧ b = 5 ∧ c = 6) :=
sorry

end box_dimensions_l521_521640


namespace problem_statement_l521_521181

open Finset
open BigOperators
open Classical

noncomputable def set_A := ({1, 3, 5, 7, 9, 11} : Finset ℕ)
noncomputable def set_B := ({2, 6, 10} : Finset ℕ)
noncomputable def set_C := ({4, 8} : Finset ℕ)
noncomputable def universal_set := (range 12).filter (λ x, x > 0)

def product_is_divisible_by_4 (a b c : ℕ) : Prop := (a * b * c) % 4 = 0

def probability_of_divisibility_by_4 : ℚ :=
  let total_ways := (filter (λ (t : (ℕ × ℕ × ℕ)), product_is_divisible_by_4 t.1 t.2.1 t.2.2) 
    (univ.product (univ.product univ)).toFinset.filter 
    (λ t, t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2)) in
  (total_ways.card : ℚ) / (univ.card.choose 3)

theorem problem_statement : probability_of_divisibility_by_4 = 20 / 33 := sorry

end problem_statement_l521_521181


namespace smallest_positive_angle_correct_l521_521208

-- Conditions
def point (α : ℝ) : ℝ × ℝ := (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))

-- Question:
def smallest_positive_angle (α : ℝ) : Prop := α = 11 * Real.pi / 6

-- Statement of the proof problem
theorem smallest_positive_angle_correct (α : ℝ) (h : point α = (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))) :
  smallest_positive_angle α :=
by
  sorry

end smallest_positive_angle_correct_l521_521208


namespace coefficient_x_inv_l521_521908

noncomputable def a := ∫ x in 0..(Real.pi / 2), (Real.sin x + Real.cos x)

theorem coefficient_x_inv : 
  let coeff := (-1)^4 * Nat.choose 6 4 * 2^(6 - 4)
  a = 2 → coeff = 60 := by
  sorry

end coefficient_x_inv_l521_521908


namespace height_after_five_years_l521_521636

namespace PapayaTreeGrowth

def growth_first_year := true → ℝ
def growth_second_year (x : ℝ) := 1.5 * x
def growth_third_year (x : ℝ) := 1.5 * growth_second_year x
def growth_fourth_year (x : ℝ) := 2 * growth_third_year x
def growth_fifth_year (x : ℝ) := 0.5 * growth_fourth_year x

def total_growth (x : ℝ) := x + growth_second_year x + growth_third_year x +
                             growth_fourth_year x + growth_fifth_year x

theorem height_after_five_years (x : ℝ) (H : total_growth x = 23) : x = 2 :=
by
  sorry

end PapayaTreeGrowth

end height_after_five_years_l521_521636


namespace probability_adjacent_vertices_in_decagon_l521_521564

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l521_521564


namespace smallest_possible_n_l521_521260

theorem smallest_possible_n : ∃ (n : ℕ), (∀ (r g b : ℕ), 24 * n = 18 * r ∧ 24 * n = 16 * g ∧ 24 * n = 20 * b) ∧ n = 30 :=
by
  -- Sorry, we're skipping the proof, as specified.
  sorry

end smallest_possible_n_l521_521260


namespace trig_identity_l521_521422

theorem trig_identity :
  sin (7 * π / 30) + sin (11 * π / 30) = sin (π / 30) + sin (13 * π / 30) + 1 / 2 :=
by
  sorry

end trig_identity_l521_521422


namespace triangle_side_count_l521_521833

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end triangle_side_count_l521_521833


namespace intersection_is_expected_result_l521_521774

def set_A : Set ℝ := { x | x * (x + 1) > 0 }
def set_B : Set ℝ := { x | ∃ y, y = Real.sqrt (x - 1) }
def expected_result : Set ℝ := { x | x ≥ 1 }

theorem intersection_is_expected_result : set_A ∩ set_B = expected_result := by
  sorry

end intersection_is_expected_result_l521_521774


namespace number_of_weeks_l521_521248

theorem number_of_weeks (janet : ℕ) (belinda : ℕ) (diff_per_week : ℕ) (total_difference : ℕ) : 
  (janet = 80) → 
  (belinda = 30) → 
  (diff_per_week = (janet - belinda) * 7) → 
  (total_difference = 2100) → 
  total_difference / diff_per_week = 6 :=
by 
  intros h_janet h_belinda h_diff_per_week h_total_difference
  rw [h_janet, h_belinda, h_diff_per_week, h_total_difference]
  exact rfl

end number_of_weeks_l521_521248


namespace hyperbola_asymptote_distance_l521_521744

theorem hyperbola_asymptote_distance (x1 y1 : ℝ) 
  (h_point_on_hyperbola : x1^2 / 4 - y1^2 / 12 = 1) :
  let d1 := (abs (sqrt 3 * x1 - y1)) / 2,
      d2 := (abs (sqrt 3 * x1 + y1)) / 2 in
  d1 * d2 = 3 :=
begin
  sorry
end

end hyperbola_asymptote_distance_l521_521744


namespace problem_l521_521321

theorem problem : 
  let b := 2 ^ 51
  let c := 4 ^ 25
  b > c :=
by 
  let b := 2 ^ 51
  let c := 4 ^ 25
  sorry

end problem_l521_521321


namespace largest_int_less_than_log_sum_l521_521479

theorem largest_int_less_than_log_sum :
  (⌊∑ k in finset.range 3010, real.log 3 (k + 2) (k + 1)⌋ : ℤ) = 6 :=
begin
  sorry
end

end largest_int_less_than_log_sum_l521_521479


namespace vegan_menu_fraction_suitable_l521_521241

theorem vegan_menu_fraction_suitable (vegan_dishes total_dishes vegan_dishes_with_gluten_or_dairy : ℕ)
  (h1 : vegan_dishes = 9)
  (h2 : vegan_dishes = 3 * total_dishes / 10)
  (h3 : vegan_dishes_with_gluten_or_dairy = 7) :
  (vegan_dishes - vegan_dishes_with_gluten_or_dairy) / total_dishes = 1 / 15 := by
  sorry

end vegan_menu_fraction_suitable_l521_521241


namespace problem_l521_521851

variable (x y z : ℚ)

-- Conditions as definitions
def cond1 : Prop := x / y = 3
def cond2 : Prop := y / z = 5 / 2

-- Theorem statement with the final proof goal
theorem problem (h1 : cond1 x y) (h2 : cond2 y z) : z / x = 2 / 15 := 
by 
  sorry

end problem_l521_521851


namespace probability_adjacent_vertices_of_decagon_l521_521513

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l521_521513


namespace function_property_product_l521_521906

theorem function_property_product {S : Type*} [has_mem S (set (ℝ+))] (f : ℝ+ → ℝ)
  (h : ∀ x y : ℝ+, f x * f y = f (x * y) + 2023 * (1 / x + 1 / y + 2022)) :
  let n := 1 in
  let s := 4047 / 2 in
  n * s = 4047 / 2 :=
by
  intro n s
  change 1 with 1
  change 4047 / 2 with (4047 / 2 : ℝ)
  sorry

end function_property_product_l521_521906


namespace probability_adjacent_vertices_in_decagon_l521_521565

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l521_521565


namespace max_knights_on_chessboard_l521_521396

/-- On an 8x8 chessboard, the maximum number of knights n (where n > 6) such that any set of 6 knights has at least 2 that attack each other, is 10. -/
theorem max_knights_on_chessboard : 
  ∃ (n : ℕ), n > 6 ∧ 
  (∀ (S : finset (fin 64)), S.card = 6 → ∃ (k1 k2 : fin 64), k1 ∈ S ∧ k2 ∈ S ∧ knights_attack k1 k2) →
  ∀ (T : finset (fin 64)), T.card = 10 ∧ 
  (∀ (S : finset (fin 64)), S ⊆ T → S.card = 6 → ∃ (k1 k2 : fin 64), k1 ∈ S ∧ k2 ∈ S ∧ knights_attack k1 k2) 
:=
10 sorry

/-- Checks if two knights attack each other. -/
def knights_attack (p1 p2 : fin 64) : Prop :=
  let (x1, y1) := (p1.1 % 8, p1.1 / 8) in
  let (x2, y2) := (p2.1 % 8, p2.1 / 8) in
  abs (x1 - x2) = 2 ∧ abs (y1 - y2) = 1 ∨ abs (x1 - x2) = 1 ∧ abs (y1 - y2) = 2

end max_knights_on_chessboard_l521_521396


namespace equilateral_triangle_l521_521864

theorem equilateral_triangle {a b c : ℝ} (h1 : a + b - c = 2) (h2 : 2 * a * b - c^2 = 4) : a = b ∧ b = c :=
by {
  sorry
}

end equilateral_triangle_l521_521864


namespace trig_identity_proof_l521_521145

theorem trig_identity_proof :
  (sin 17 * cos 13 + sin 73 * sin 167 = 1/2) :=
by
  sorry

end trig_identity_proof_l521_521145


namespace M_iff_N_l521_521202

variables {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

def is_nonzero (v : V) : Prop := v ≠ 0
def is_noncollinear (a b : V) : Prop := ∃ c : ℝ, c ≠ 0 ∧ c • a ≠ b
def perpendicular (u v : V) : Prop := ⟪u, v⟫ = 0

-- Condition M: \( \vec{b} \perp (\vec{a} - \vec{b}) \)
def M : Prop := perpendicular b (a - b)

-- Condition N: \( \forall x \in \mathbb{R}, |\vec{a} - x\vec{b}| \geq |\vec{a} - \vec{b}|\)
def N : Prop := ∀ x : ℝ, ∥a - x • b∥ ≥ ∥a - b∥

-- The goal is to prove that M is a necessary and sufficient condition for N
theorem M_iff_N (h_nonzero_a : is_nonzero a) (h_nonzero_b : is_nonzero b) (h_noncollinear_a_b : is_noncollinear a b) : M a b ↔ N a b :=
by sorry

end M_iff_N_l521_521202


namespace salary_percentage_difference_l521_521856

theorem salary_percentage_difference (A B : ℝ) (h : A = 0.8 * B) :
  (B - A) / A * 100 = 25 :=
sorry

end salary_percentage_difference_l521_521856


namespace problem_correct_l521_521775

variables (x y t M : ℝ)

-- Define the system of equations
def system_eq_1 := x - 3*y = 4 - t
def system_eq_2 := x + y = 3*t

-- Range of t
def range_t := -3 ≤ t ∧ t ≤ 1

-- Define the statements to be proved
def statement_1 := (x = 1) ∧ (y = -1)
def statement_2 := (x - y = 3) → (t = -2)
def statement_3 := (M = 2*x - y - t) ∧ (M ≥ -3)

theorem problem_correct : 
  (range_t) →
  (system_eq_1) →
  (system_eq_2) →
  (statement_1) ∧ (¬ statement_2) ∧ (statement_3) :=
sorry

end problem_correct_l521_521775


namespace students_in_largest_class_l521_521258

theorem students_in_largest_class 
    (number_of_classes : ℕ := 5)
    (total_students : ℕ := 120)
    (students_diff : ℕ := 2)
    (class_counts : Fin number_of_classes → ℕ)
    (h1 : class_counts 0 = class_counts 1 + students_diff)
    (h2 : class_counts 1 = class_counts 2 + students_diff)
    (h3 : class_counts 2 = class_counts 3 + students_diff)
    (h4 : class_counts 3 = class_counts 4 + students_diff)
    (h_total : class_counts 0 + class_counts 1 + class_counts 2 + class_counts 3 + class_counts 4 = total_students) :
    class_counts 0 = 28 :=
sorry

end students_in_largest_class_l521_521258


namespace cat_walking_rate_l521_521894

theorem cat_walking_rate :
  let resisting_time := 20 -- minutes
  let total_distance := 64 -- feet
  let total_time := 28 -- minutes
  let walking_time := total_time - resisting_time
  (total_distance / walking_time = 8) :=
by
  let resisting_time := 20
  let total_distance := 64
  let total_time := 28
  let walking_time := total_time - resisting_time
  have : total_distance / walking_time = 8 :=
    by norm_num [total_distance, walking_time]
  exact this

end cat_walking_rate_l521_521894


namespace not_possible_to_create_3_piles_l521_521368

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end not_possible_to_create_3_piles_l521_521368


namespace jims_investment_l521_521298

theorem jims_investment
  {total_investment : ℝ} 
  (h1 : total_investment = 127000)
  {john_ratio : ℕ} 
  (h2 : john_ratio = 8)
  {james_ratio : ℕ} 
  (h3 : james_ratio = 11)
  {jim_ratio : ℕ} 
  (h4 : jim_ratio = 15)
  {jordan_ratio : ℕ} 
  (h5 : jordan_ratio = 19) :
  jim_ratio / (john_ratio + james_ratio + jim_ratio + jordan_ratio) * total_investment = 35943.40 :=
by {
  sorry
}

end jims_investment_l521_521298


namespace expected_value_fair_dodecahedral_die_l521_521074

theorem expected_value_fair_dodecahedral_die : 
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  let probability := (1 : ℝ) / 12 in
  (probability * faces.sum) = 6.5 :=
by
  let faces := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let probability := (1 : ℝ) / 12
  have h : faces.sum = 78 := rfl
  rw [h]
  sorry

end expected_value_fair_dodecahedral_die_l521_521074


namespace man_twice_son_age_l521_521115

theorem man_twice_son_age (S M Y : ℕ) (h1 : S = 27) (h2 : M = S + 29) (h3 : M + Y = 2 * (S + Y)) : Y = 2 := 
by sorry

end man_twice_son_age_l521_521115


namespace tank_fill_time_l521_521129

/-- Given the rates at which pipes fill a tank, prove the total time to fill the tank using all three pipes. --/
theorem tank_fill_time (R_a R_b R_c : ℝ) (T : ℝ)
  (h1 : R_a = 1 / 35)
  (h2 : R_b = 2 * R_a)
  (h3 : R_c = 2 * R_b)
  (h4 : T = 5) :
  1 / (R_a + R_b + R_c) = T := by
  sorry

end tank_fill_time_l521_521129


namespace impossible_to_create_3_piles_l521_521374

def similar_sizes (x y : ℝ) : Prop := x ≤ sqrt 2 * y

theorem impossible_to_create_3_piles (x y : ℝ) (h₁ : similar_sizes x y) :
  ¬ ∃ (x₁ x₂ x₃ : ℝ), x₁ + x₂ + x₃ = x + y ∧ similar_sizes x₁ x₂ ∧ similar_sizes x₂ x₃ ∧ similar_sizes x₁ x₃ :=
by
  sorry

end impossible_to_create_3_piles_l521_521374


namespace second_term_of_geometric_series_l521_521662

noncomputable def geometric_series_second_term (a r : ℝ) (S : ℝ) : ℝ :=
a * r

theorem second_term_of_geometric_series 
  (a r S : ℝ) 
  (h1 : r = 1 / 4) 
  (h2 : S = 10) 
  (h3 : S = a / (1 - r)) 
  : geometric_series_second_term a r S = 1.875 :=
by
  sorry

end second_term_of_geometric_series_l521_521662


namespace triangle_possible_sides_l521_521782

theorem triangle_possible_sides : 
  let unknown_side_lengths := {x : ℤ | 3 < x ∧ x < 13}
  in set.finite unknown_side_lengths ∧ set.card unknown_side_lengths = 9 :=
by
  sorry

end triangle_possible_sides_l521_521782


namespace wire_length_correct_l521_521032

-- Define the radii of the cylindrical poles
def radius1 := 4 -- radius of smaller pole (8 inches diameter)
def radius2 := 10 -- radius of larger pole (20 inches diameter)

-- Compute the required length of wire
noncomputable def wire_length : ℝ :=
  let straight_length := 2 * real.sqrt(((radius1 + radius2) ^ 2) - ((radius2 - radius1) ^ 2))
  let arc_length1 := 2 * real.pi * radius1 / 4
  let arc_length2 := 2 * real.pi * radius2 * 3 / 4
  straight_length + arc_length1 + arc_length2

-- Desired result as per the solution steps
def desired_length : ℝ := 8 * real.sqrt 10 + 17 * real.pi

-- The proof statement
theorem wire_length_correct : wire_length = desired_length := 
  sorry

end wire_length_correct_l521_521032


namespace number_is_correct_l521_521655

/-- 
  Define the two-digit number x which:
  1. A's conditions: The number has an even number of factors and is greater than 50.
  2. B's conditions: The number is odd and is greater than 60.
  3. C's conditions: The number is even and is greater than 70.
  The solution finds x based on one-half being true for each statement.
-/
def has_even_number_of_factors (n : ℕ) : Prop :=
  ∃ k : ℕ, 2 * k = (List.range (n + 1)).filter (λ d, n % d = 0).length

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def fulfills_a_conditions (n : ℕ) : Prop :=
  has_even_number_of_factors n ∧ n > 50

def fulfills_b_conditions (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n > 60

def fulfills_c_conditions (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n > 70

def correct_number : ℕ :=
  64

theorem number_is_correct :
  ∀ n : ℕ, is_two_digit n →
    (fulfills_a_conditions n ∨ fulfills_b_conditions n ∨ fulfills_c_conditions n) →
    n = correct_number :=
sorry

end number_is_correct_l521_521655


namespace probability_to_break_bill_l521_521417

theorem probability_to_break_bill
  (toys : Fin 10 → ℚ)
  (prices_range : ∀ n : Fin 10, 0.5 ≤ toys n ∧ toys n ≤ 2.5)
  (price_increase : ∀ n m : Fin 10, n < m → toys n + 0.25 = toys m)
  (favorite_toy_price : toys (10 - 2) = 2.25)
  (quarters_initial : ℕ := 12)
  (quarter_value : ℚ := 0.25)
  (machine_only_accepts_quarters : ℚ := 1)
  (favorite_toy_cost_in_quarters : toys (10 - 2) / quarter_value ≤ quarters_initial) :
  let total_permutations := Nat.factorial 10
      favorable_first := Nat.factorial 9
      favorable_second := Nat.factorial 8
      probability := (favorable_first + favorable_second) / total_permutations in
  1 - probability = 8 / 9 := 
sorry

end probability_to_break_bill_l521_521417


namespace expected_value_fair_dodecahedral_die_l521_521057

theorem expected_value_fair_dodecahedral_die :
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  expected_value = 6.5 :=
by
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  have h1 : list.sum outcomes = 78 := by sorry
  have h2 : (12: ℝ) * (1 / 12: ℝ) = 1 := by sorry
  show expected_value = 6.5
  { sorry }

end expected_value_fair_dodecahedral_die_l521_521057


namespace Robert_older_than_Elizabeth_l521_521400

-- Define the conditions
def Patrick_half_Robert (Patrick Robert : ℕ) : Prop := Patrick = Robert / 2
def Robert_turn_30_in_2_years (Robert : ℕ) : Prop := Robert + 2 = 30
def Elizabeth_4_years_younger_than_Patrick (Elizabeth Patrick : ℕ) : Prop := Elizabeth = Patrick - 4

-- The theorem we need to prove
theorem Robert_older_than_Elizabeth
  (Patrick Robert Elizabeth : ℕ)
  (h1 : Patrick_half_Robert Patrick Robert)
  (h2 : Robert_turn_30_in_2_years Robert)
  (h3 : Elizabeth_4_years_younger_than_Patrick Elizabeth Patrick) :
  Robert - Elizabeth = 18 :=
sorry

end Robert_older_than_Elizabeth_l521_521400


namespace probability_adjacent_vertices_in_decagon_l521_521559

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l521_521559


namespace decagon_adjacent_vertex_probability_l521_521549

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l521_521549


namespace union_sets_l521_521773

def A : Set ℝ := { x | (2 / x) > 1 }
def B : Set ℝ := { x | Real.log x < 0 }

theorem union_sets : (A ∪ B) = { x : ℝ | 0 < x ∧ x < 2 } := by
  sorry

end union_sets_l521_521773


namespace decagon_adjacent_probability_l521_521517

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521517


namespace simplify_root_fractions_l521_521424

theorem simplify_root_fractions :
  (√648 / √72) - (√294 / √98) = 3 - √3 :=
by
  sorry

end simplify_root_fractions_l521_521424


namespace coloring_methods_l521_521734

-- Define the conditions: m colors and n length
variables (m n : ℕ)

-- Define the function f that gives the number of coloring methods
def f (n m : ℕ) : ℕ :=
  ∑ k in Finset.range(m+1), (-1)^k * Nat.choose (m-2) k * m * (m-1) *
  (m^2 - 2 * m * k - 3 * m + k^2 + 3 * k + 3)^(n-1)

-- Main theorem statement
theorem coloring_methods (m n : ℕ) : 
  f n m = ∑ k in Finset.range(m+1), (-1)^k * Nat.choose (m-2) k * m * (m-1) *
  (m^2 - 2 * m * k - 3 * m + k^2 + 3 * k + 3)^(n-1) :=
sorry

end coloring_methods_l521_521734


namespace difference_of_squares_divisible_by_7_l521_521291

theorem difference_of_squares_divisible_by_7 (a₁ a₂ a₃ a₄ a₅ : ℕ) :
  ∃ i j, i ≠ j ∧ ((a_i ^ 2 - a_j ^ 2) % 7 = 0) :=
by
  sorry

end difference_of_squares_divisible_by_7_l521_521291


namespace decagon_adjacent_probability_l521_521521

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521521


namespace expected_value_of_fair_dodecahedral_die_l521_521050

theorem expected_value_of_fair_dodecahedral_die : 
  (1/12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_of_fair_dodecahedral_die_l521_521050


namespace shaded_area_of_grid_l521_521870

/-- In a 6x6 grid, the area of the shaded figure formed by connecting midpoints of the sides 
to the center of the grid, creating a smaller central square and four triangles in each corner, 
is 22.5. --/
theorem shaded_area_of_grid : 
  let grid_size := 6 in
  let side_midpoints := { (1.5, 3), (3, 1.5), (4.5, 3), (3, 4.5) } in
  let corner_points := { (0, 0), (0, 6), (6, 0), (6, 6) } in
  let center := (3, 3) in
  area_of_shaded_figure grid_size side_midpoints center corner_points = 22.5 := 
sorry

end shaded_area_of_grid_l521_521870


namespace outer_boundary_diameter_eq_52_l521_521696

-- Given conditions
def width_walking_path : ℕ := 6
def width_garden_ring : ℕ := 10
def diameter_fountain : ℕ := 20

-- Derived condition
def radius_fountain : ℕ := diameter_fountain / 2

-- Prove the theorem
theorem outer_boundary_diameter_eq_52 :
  let total_radius := radius_fountain + width_garden_ring + width_walking_path
  in 2 * total_radius = 52 :=
by
  sorry

end outer_boundary_diameter_eq_52_l521_521696


namespace solution_set_inequality_l521_521981

theorem solution_set_inequality (a b : ℝ)
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → x^2 + a * x + b ≤ 0) :
  a * b = 6 :=
by {
  sorry
}

end solution_set_inequality_l521_521981


namespace translated_parabola_expression_l521_521991

-- Original function definition
def original_parabola (x : ℝ) : ℝ := x^2

-- Transformation definitions
def translate_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, f (x - a)
def translate_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x, f x + b

-- Composed transformation function
def translate_right_and_up (f : ℝ → ℝ) (a b : ℝ) : ℝ → ℝ :=
  translate_up (translate_right f a) b

-- Problem statement
theorem translated_parabola_expression :
  translate_right_and_up original_parabola 2 1 = λ x, (x - 2)^2 + 1 :=
by
  sorry

end translated_parabola_expression_l521_521991


namespace maximum_fraction_l521_521607

theorem maximum_fraction (A B : ℕ) (h1 : A ≠ B) (h2 : 0 < A ∧ A < 1000) (h3 : 0 < B ∧ B < 1000) :
  ∃ (A B : ℕ), (A = 500) ∧ (B = 499) ∧ (A ≠ B) ∧ (0 < A ∧ A < 1000) ∧ (0 < B ∧ B < 1000) ∧ (A - B = 1) ∧ (A + B = 999) ∧ (499 / 500 = 0.998) := sorry

end maximum_fraction_l521_521607


namespace seating_problem_l521_521263

def total_seating_arrangements : Nat := 40320

def wilma_paul_together : Nat := 7.factorial * 2.factorial

def adam_eve_together : Nat := 7.factorial * 2.factorial

def both_pairs_together : Nat := 6.factorial * 2.factorial * 2.factorial

def inclusion_exclusion : Nat :=
  wilma_paul_together + adam_eve_together - both_pairs_together

def acceptable_seating_arrangements : Nat :=
  total_seating_arrangements - inclusion_exclusion

theorem seating_problem :
  acceptable_seating_arrangements = 23040 :=
  by
    sorry

end seating_problem_l521_521263


namespace john_finances_l521_521299

theorem john_finances :
  let total_first_year := 10000
  let tuition_percent := 0.4
  let room_board_percent := 0.35
  let textbook_transport_percent := 0.25
  let tuition_increase := 0.06
  let room_board_increase := 0.03
  let aid_first_year := 0.25
  let aid_increase := 0.02

  let tuition_first_year := total_first_year * tuition_percent
  let room_board_first_year := total_first_year * room_board_percent
  let textbook_transport_first_year := total_first_year * textbook_transport_percent

  let tuition_second_year := tuition_first_year * (1 + tuition_increase)
  let room_board_second_year := room_board_first_year * (1 + room_board_increase)
  let financial_aid_second_year := tuition_second_year * (aid_first_year + aid_increase)

  let tuition_third_year := tuition_second_year * (1 + tuition_increase)
  let room_board_third_year := room_board_second_year * (1 + room_board_increase)
  let financial_aid_third_year := tuition_third_year * (aid_first_year + 2 * aid_increase)

  let total_cost_first_year := 
      (tuition_first_year - tuition_first_year * aid_first_year) +
      room_board_first_year + 
      textbook_transport_first_year

  let total_cost_second_year :=
      (tuition_second_year - financial_aid_second_year) +
      room_board_second_year +
      textbook_transport_first_year

  let total_cost_third_year :=
      (tuition_third_year - financial_aid_third_year) +
      room_board_third_year +
      textbook_transport_first_year

  total_cost_first_year = 9000 ∧
  total_cost_second_year = 9200.20 ∧
  total_cost_third_year = 9404.17 := 
by
  sorry

end john_finances_l521_521299


namespace highest_average_speed_is_during_third_interval_l521_521113

-- Define the structure for the problem
structure Interval where
  start_time : ℕ
  end_time : ℕ

-- Define the hour intervals
def intervals : List Interval := [
  {start_time := 0, end_time := 1},
  {start_time := 1, end_time := 2},
  {start_time := 3, end_time := 4},
  {start_time := 5, end_time := 6},
  {start_time := 10, end_time := 11}
]

-- The function to determine the steepest interval
def hasHighestAverageSpeed (interval : Interval) : Prop :=
  interval = {start_time := 3, end_time := 4}

-- Proof Statement
theorem highest_average_speed_is_during_third_interval :
  ∃ i, i ∈ intervals ∧ hasHighestAverageSpeed i :=
by
  use {start_time := 3, end_time := 4}
  split
  · exact List.Mem.tail (List.Mem.tail (List.Mem.head _))
  · refl

end highest_average_speed_is_during_third_interval_l521_521113


namespace complex_ab_value_l521_521735

theorem complex_ab_value (a b : ℝ) (i : ℂ) (h_i : i = Complex.I) (h_z : a + b * i = (4 + 3 * i) * i) : a * b = -12 :=
by {
  sorry
}

end complex_ab_value_l521_521735


namespace decagon_adjacent_vertex_probability_l521_521551

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l521_521551


namespace music_library_space_per_hour_l521_521628

theorem music_library_space_per_hour 
    (days : ℕ)
    (total_space : ℕ)
    (total_hours := days * 24)
    (space_per_hour := total_space / total_hours)
    (rounded_space_per_hour := Int.ceil (space_per_hour : ℚ)) :
  days = 15 →
  total_space = 20000 →
  rounded_space_per_hour = 56 :=
by
  intros h_days h_space
  rw [h_days, h_space]
  have h_total_hours : total_hours = 360 := by
    rw [h_days]
    norm_num
  have h_space_per_hour : space_per_hour = (20000 / 360 : ℚ) := by
    rw [h_total_hours, h_space]
    norm_cast
  have h_rounded_space_per_hour : rounded_space_per_hour = Int.ceil (55.56 : ℚ) := by
    rw [h_space_per_hour]
    norm_num
    linarith
  exact h_rounded_space_per_hour

end music_library_space_per_hour_l521_521628


namespace probability_of_adjacent_vertices_in_decagon_l521_521592

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l521_521592


namespace sum_sq_geq_nsqr_sum_sq_l521_521191

theorem sum_sq_geq_nsqr_sum_sq (n : ℕ) (a : Fin n → ℕ)
  (h1 : 2 ≤ n)
  (h2 : ∀ i : Fin (n-1), 1 ≤ (a i.succ) / (a i) ∧ (a i.succ) / (a i) ≤ 1 + 1/(i + 1)) :
  (∑ i in Finset.range n, (i + 1) * a ⟨i, Nat.lt_of_lt_pred h1 (Finset.mem_range.1 (Finset.mem_range.2 i))⟩) ^ 2 ≥
  n * (n + 1)^2 / 4 * ∑ i in Finset.range n, (a ⟨i, Nat.lt_of_lt_pred h1 (Finset.mem_range.1 (Finset.mem_range.2 i))⟩)^2 :=
by
  sorry

end sum_sq_geq_nsqr_sum_sq_l521_521191


namespace seq_general_term_arithmetic_seq_sum_l521_521147

theorem seq_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) :
  a 1 = 1 ∧ (∀ n ≥ 1, a (n + 1) = 2 * S n + 1) ∧ S = λ n, (finset.range n).sum a → 
  ∀ n, a n = 3 ^ (n - 1) := by
  sorry

theorem arithmetic_seq_sum (a b : ℕ → ℕ) (T : ℕ → ℕ):
  (∀ n, a n = 3 ^ (n - 1)) ∧ 
  T 3 = 15 ∧ 
  (∀ i, b (i + 1) - b i = b 2 - b 1) ∧ 
  (a 1 + b 1) * (a 3 + b 3) = (a 2 + b 2) ^ 2 → 
  ∀ n, T n = n^2 + 2*n := by
  sorry

end seq_general_term_arithmetic_seq_sum_l521_521147


namespace not_infinite_complementary_sequences_1_sum_first_16_terms_general_terms_l521_521232

section InfiniteComplementarySequences

open Set

/-- Definition of infinite complementary sequences --/
def infinite_complementary_sequences (a b : ℕ → ℕ) : Prop :=
  StrictMono a ∧ StrictMono b ∧ (range a ∩ range b = ∅) ∧ (range a ∪ range b = Set.univ)

/-- Problem 1: Prove that the sequences a_n = 2n - 1 and b_n = 4n - 2
    are not infinite complementary sequences. --/
theorem not_infinite_complementary_sequences_1 :
  ¬ infinite_complementary_sequences (λ n, 2 * n - 1) (λ n, 4 * n - 2) :=
sorry

/-- Problem 2: Assuming a_n = 2^n and a_n, b_n are infinite complementary sequences,
    find the sum of the first 16 terms of b_n is 180. --/
theorem sum_first_16_terms (b : ℕ → ℕ) (h : infinite_complementary_sequences (λ n, 2^n) b) :
  (Finset.sum (Finset.range 16) b) = 180 :=
sorry

/-- Problem 3: Given a_16 = 36 and a_n is an arithmetic sequence,
    find the general terms of a_n and b_n such that they are infinite complementary sequences. --/
theorem general_terms (d : ℕ) (h : (λ n, 2 * n + 4)) :
  infinite_complementary_sequences (λ n, 2 * n + 4)
  (λ n, if n ≤ 5 then n else 2 * n - 5) :=
sorry

end InfiniteComplementarySequences

end not_infinite_complementary_sequences_1_sum_first_16_terms_general_terms_l521_521232


namespace expected_value_of_dodecahedral_die_is_6_5_l521_521045

noncomputable def expected_value_of_dodecahedral_die : ℝ := 
  ∑ i in Finset.range (12 + 1), (i : ℝ) / 12

theorem expected_value_of_dodecahedral_die_is_6_5 :
  expected_value_of_dodecahedral_die = 6.5 := sorry

end expected_value_of_dodecahedral_die_is_6_5_l521_521045


namespace possible_integer_side_lengths_l521_521827

theorem possible_integer_side_lengths (a b : ℕ) (h1 : a = 8) (h2 : b = 5) :
  {x : ℕ | 3 < x ∧ x < 13}.finite ∧ {x : ℕ | 3 < x ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521827


namespace exponent_subtraction_l521_521186

theorem exponent_subtraction (a : ℝ) (m n : ℝ) (hm : a^m = 3) (hn : a^n = 5) : a^(m-n) = 3 / 5 := 
  sorry

end exponent_subtraction_l521_521186


namespace probability_adjacent_vertices_of_decagon_l521_521514

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l521_521514


namespace find_fixed_point_l521_521745

-- Define the given points and circle
structure point :=
  (x : ℝ)
  (y : ℝ)

def M := point.mk (real.sqrt 3) 0
def N := point.mk (- (real.sqrt 3)) 0

def on_circle (P : point) : Prop :=
  (P.x + real.sqrt 3)^2 + P.y^2 = 16

-- Define the trajectory C
def C (Q : point) : Prop :=
  (Q.x^2 / 4) + Q.y^2 = 1

-- Define the point D on the positive y-axis
def D : point := point.mk 0 1

-- Define line l
def line_l (k m : ℝ) (p : point) : Prop :=
  p.y = k * p.x + m

-- Define perpendicular condition AD ⊥ BD
def perpendicular_AD_BD (A B D : point) : Prop :=
  let DA := (A.x, A.y - D.y)
  let DB := (B.x, B.y - D.y)
  DA.1 * DB.1 + (DA.2 * DB.2) = 0

theorem find_fixed_point (k m : ℝ) :
  (line_l k m D → (D = point.mk 0 (-3/5))) →
  (on_circle ∘ C) (M) → D ∈ C →
  ∀ A B, A ∈ C → B ∈ C → ¬ line_l k m D → perpendicular_AD_BD A B D →
  line_l k (-3/5) :=
sorry

end find_fixed_point_l521_521745


namespace num_two_digit_factorizations_l521_521240

theorem num_two_digit_factorizations (n : ℕ)
  (h : n = 945)
  (two_digit_factor : ℕ → Prop := λ x, 10 ≤ x ∧ x < 100)
  (unique_factorizations : ∃! (a b : ℕ), two_digit_factor a ∧ two_digit_factor b ∧ a * b = n) :
  ∃! (a b : ℕ), two_digit_factor a ∧ two_digit_factor b ∧ a * b = 945 ∧ a ≤ b := sorry

end num_two_digit_factorizations_l521_521240


namespace four_player_cycle_probability_l521_521720

theorem four_player_cycle_probability :
  ∀ (players : finset ℕ), 
  players.card = 5 → 
  (∀ ⦃x y : ℕ⦄, x ∈ players → y ∈ players → 
                  x ≠ y → (random.bool 0.5 0.5 : Prop))
  → (∃ (P1 P2 P3 P4: ℕ), 
      P1 ≠ P2 ∧ P2 ≠ P3 ∧ P3 ≠ P4 ∧ P4 ≠ P1 ∧ 
      P1 ∈ players ∧ P2 ∈ players ∧ P3 ∈ players ∧ P4 ∈ players ∧ 
      (random.bool 0.5 0.5 : Prop) ∧ 
      (random.bool 0.5 0.5 : Prop) ∧ 
      (random.bool 0.5 0.5 : Prop) ∧ 
      (random.bool 0.5 0.5 : Prop) ∧ 
      (random.bool 0.5 0.5 : Prop) = 49 / 64) :=
begin
  sorry,
end

end four_player_cycle_probability_l521_521720


namespace inscribed_rectangle_l521_521122

theorem inscribed_rectangle (b h : ℝ) : ∃ x : ℝ, 
  (∃ q : ℝ, x = q / 2) → 
  ∃ x : ℝ, 
    (∃ q : ℝ, q = 2 * x ∧ x = h * q / (2 * h + b)) :=
sorry

end inscribed_rectangle_l521_521122


namespace a_gt_one_l521_521251

noncomputable def f (a : ℝ) (x : ℝ) := 2 * a * x^2 - x - 1

theorem a_gt_one (a : ℝ) :
  (∃! x, 0 < x ∧ x < 1 ∧ f a x = 0) → 1 < a :=
by
  sorry

end a_gt_one_l521_521251


namespace geometric_sequence_condition_l521_521757

variable (a_1 : ℝ) (q : ℝ)

noncomputable def geometric_sum (n : ℕ) : ℝ :=
if q = 1 then a_1 * n else a_1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_condition (a_1 : ℝ) (q : ℝ) :
  (a_1 > 0) ↔ (geometric_sum a_1 q 2017 > 0) :=
by sorry

end geometric_sequence_condition_l521_521757


namespace largest_possible_c_l521_521914

theorem largest_possible_c (c : ℝ) (hc : (3 * c + 4) * (c - 2) = 9 * c) : c ≤ 4 :=
sorry

end largest_possible_c_l521_521914


namespace fox_can_always_catch_l521_521119

noncomputable def fox_can_catch_rabbits (v : ℝ) : Prop :=
  ∀ (A B C D : ℝ × ℝ)
  (fox rabbit1 rabbit2 : ℝ × ℝ)
  (burrow : ℝ × ℝ),
  (burrow = (0, 0)) →
  (A = (0, 0)) →
  (B = (1, 0)) →
  (C = (1, 1)) →
  (D = (0, 1)) →
  (fox = (1, 1)) →
  (rabbit1 = (1, 0)) →
  (rabbit2 = (0, 1)) →
  (∀ (t : ℝ), t ≥ 0 → (1 / v) * t ≤ √2 * t) →
  v ≥ 1 + √2
  
theorem fox_can_always_catch 
  (v : ℝ) : 
  v ≥ 1 + √2 → fox_can_catch_rabbits v := 
sorry

end fox_can_always_catch_l521_521119


namespace man_monthly_salary_l521_521634

theorem man_monthly_salary (S : ℝ) :
  (0.35 * S - 0.15 * (0.35 * S) = 250) → S = 840 :=
by
  intro h,
  sorry

end man_monthly_salary_l521_521634


namespace smallest_N_exists_l521_521010

theorem smallest_N_exists (
  a b c d : ℕ := list.perm [1, 2, 3, 4, 5] [gcd a b, gcd a c, gcd a d, gcd b c, gcd b d, gcd c d]
  (h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_N: N > 5) : 
  N = 14 :=
by sorry

end smallest_N_exists_l521_521010


namespace probability_of_adjacent_vertices_in_decagon_l521_521591

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l521_521591


namespace find_n_l521_521210

theorem find_n (n : ℕ) (h : n > 0) :
  (n * (n - 1) * (n - 2)) / (6 * n^3) = 1 / 16 ↔ n = 4 :=
by sorry

end find_n_l521_521210


namespace arithmetic_sequence_common_difference_l521_521505

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)

-- Conditions
def condition1 : Prop := ∀ n, S n = (n * (2*a 1 + (n-1) * d)) / 2
def condition2 : Prop := S 3 = 6
def condition3 : Prop := a 3 = 0

-- Question
def question : ℝ := d

-- Correct Answer
def correct_answer : ℝ := -2

-- Proof Problem Statement
theorem arithmetic_sequence_common_difference : 
  condition1 a S d ∧ condition2 S ∧ condition3 a →
  question d = correct_answer :=
sorry

end arithmetic_sequence_common_difference_l521_521505


namespace part1_part2a_part2b_l521_521311

-- Define the sequence, sum, and conditions.
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {A B C : ℝ} (h1 : ∀ n : ℕ, 2 * a n + S n = A * n^2 + B * n + C)

-- First part
theorem part1 (A B : ℝ) (C : ℝ) (hA : A = 0) (hB : B = 0) (hC : C = 1) :
  ∀ n, a n = (1 / 3) * (2 / 3)^(n - 1) :=
sorry

-- Second part
theorem part2a (A : ℝ) (C : ℝ) (hA : A = 1) (hC : C = -2)
  (arith_seq : ∀ m n, a m - a n = (m - n) * (a 2 - a 1)) :
  ∀ n, a n = 2 * n - 1 :=
sorry

-- Third part
variable {b : ℕ → ℝ}
variable {T : ℕ → ℝ}
variable (m : ℝ)

def b_n (n : ℕ) : ℝ := b n
def T_n (n : ℕ) : ℝ := T n

theorem part2b (h_b : ∀ n, b n = 2^n * a n)
  (h_T : ∀ n, T n = ∑ i in finset.range (n + 1), b i)
  (ineq_holds : ∀ n : ℕ, T_n n - (2 * n + m) * 2^(n+1) > 0) :
  m ≤ -3 :=
sorry

end part1_part2a_part2b_l521_521311


namespace expected_value_dodecahedral_die_l521_521081

noncomputable def dodecahedral_expected_value : ℝ :=
  let outcomes := (list.range (12 + 1)).tail -- List of numbers from 1 to 12
  let n_faces := 12
  let probability := 1 / (n_faces : ℝ)
  let expected_value := probability * outcomes.sum / n_faces
  expected_value

theorem expected_value_dodecahedral_die :
  dodecahedral_expected_value = 6.5 :=
by {
  unfold dodecahedral_expected_value,
  simp,
  sorry,
}

end expected_value_dodecahedral_die_l521_521081


namespace number_of_correct_propositions_l521_521759

-- Define vector operations and properties
variables {V : Type*} [AddGroup V] [Module ℝ V]

-- Define the propositions
def prop1 (A B : V) : Prop := A + B = 0
def prop2 (A B C : V) : Prop := A + B = C
def prop3 (A B C : V) : Prop := A - C = B
def prop4 (A : V) : Prop := 0 • A = 0

-- Define the proof problem statement
theorem number_of_correct_propositions (A B C : V) :
  (prop1 A (-A)) ∧ (prop2 A B (A + B)) ∧ ¬(prop3 A (B - A) A) ∧ ¬(prop4 A) 
  → 2 = 2 :=
by sorry

end number_of_correct_propositions_l521_521759


namespace possible_integer_side_lengths_l521_521820

theorem possible_integer_side_lengths (a b : ℕ) (h1 : a = 8) (h2 : b = 5) :
  {x : ℕ | 3 < x ∧ x < 13}.finite ∧ {x : ℕ | 3 < x ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521820


namespace correct_conclusions_l521_521439

def D (x : ℝ) : ℝ := if x.is_rational then 1 else 0

def L (x : ℝ) : ℝ := if x.is_rational then x else 0

theorem correct_conclusions (c1 : D 1 = L 1 ∧ ¬∀ x, L (-x) = L x ∧ ¬∃ (A B C D : ℝ × ℝ), 
    (A ∈ graph L ∧ B ∈ graph L ∧ C ∈ graph L ∧ D ∈ graph L) ∧ 
    (is_rhombus A B C D) ∧ ∃ (A B C : ℝ × ℝ), 
    (A ∈ graph L ∧ B ∈ graph L ∧ C ∈ graph L) ∧ (is_equilateral_triangle A B C)) : 
     ∃ conclusions, conclusions = [true, false, false, true] 
:= sorry

end correct_conclusions_l521_521439


namespace solve_system_eqs_l521_521957
noncomputable section

theorem solve_system_eqs (x y z : ℝ) :
  (x * y = 5 * (x + y) ∧ x * z = 4 * (x + z) ∧ y * z = 2 * (y + z))
  ↔ (x = 0 ∧ y = 0 ∧ z = 0)
  ∨ (x = -40 ∧ y = 40 / 9 ∧ z = 40 / 11) := sorry

end solve_system_eqs_l521_521957


namespace eccentricity_of_hyperbola_l521_521748

noncomputable def hyperbola_eccentricity : ℝ :=
  let a := sorry in   -- assumption: a > b > 0
  let b := sorry in   -- assumption: b^2 = c^2 - a^2
  let c := sqrt (a^2 + b^2) in   -- by definition of foci
  let e := c / a in   -- by definition of eccentricity
  e

theorem eccentricity_of_hyperbola :
  ∀ (a b : ℝ), a > b → b > 0 →
  (∃ (c : ℝ), c > 0 ∧ b^2 = c^2 - a^2) →
  hyperbola_eccentricity = sqrt 3 + 1 :=
by
  intros a b ha hb hc
  unfold hyperbola_eccentricity
  sorry

end eccentricity_of_hyperbola_l521_521748


namespace product_of_consecutive_even_numbers_divisible_by_24_l521_521499

theorem product_of_consecutive_even_numbers_divisible_by_24 (n : ℕ) :
  (2 * n) * (2 * n + 2) * (2 * n + 4) % 24 = 0 :=
  sorry

end product_of_consecutive_even_numbers_divisible_by_24_l521_521499


namespace probability_of_monochromatic_triangle_l521_521156

-- Definitions:
structure Hexagon :=
  (vertices : Finset ℕ)
  (edges : Finset (ℕ × ℕ))

noncomputable def all_edges := 
  ({(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)} ∪ 
   {(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (1, 5), 
    (2, 4), (2, 5), (3, 5)} : Finset (ℕ × ℕ))

def hex : Hexagon := { vertices := Finset.range 6, edges := all_edges }

-- Question translated into Lean statement:
theorem probability_of_monochromatic_triangle :
  (let non_mono_prob := 3 / 4 in
   let total_triangles := 20 in
   1 - non_mono_prob^total_triangles) ≈ 0.99683 := sorry

end probability_of_monochromatic_triangle_l521_521156


namespace probability_adjacent_vertices_in_decagon_l521_521570

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l521_521570


namespace num_possible_triangle_sides_l521_521809

theorem num_possible_triangle_sides (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (a_eq_8 : a = 8) (b_eq_5 : b = 5) : 
  let x := {n : ℕ | 3 < n ∧ n < 13} in
  set.card x = 9 :=
by
  sorry

end num_possible_triangle_sides_l521_521809


namespace john_bought_items_l521_521646

theorem john_bought_items (n : ℕ) 
  (item_cost : ℕ := 200)
  (discount_rate : ℝ := 0.10)
  (discount_threshold : ℕ := 1000)
  (total_cost_after_discount : ℕ := 1360)
  -- Given conditions:
  (h1 : item_cost = 200)
  (h2 : discount_rate = 0.10)
  (h3 : discount_threshold = 1000)
  (h4 : total_cost_after_discount = 1360) :
  -- Total cost before discount
  let total_cost_before_discount := item_cost * n in
  -- Discount calculation
  let discount_amount := discount_rate * (total_cost_before_discount - discount_threshold) in
  -- Equation for total cost after discount
  total_cost_after_discount = total_cost_before_discount - discount_amount →
  n = 7 := sorry

end john_bought_items_l521_521646


namespace canvas_bag_lower_carbon_solution_l521_521933

def canvas_bag_emission := 600 -- pounds of CO2
def plastic_bag_emission := 4 -- ounces of CO2 per bag
def bags_per_trip := 8 
def ounce_to_pound := 16 -- 16 ounces in a pound
def co2_trip := (plastic_bag_emission * bags_per_trip) / ounce_to_pound -- CO2 emission in pounds per trip

theorem canvas_bag_lower_carbon_solution : 
  co2_trip * 300 >= canvas_bag_emission :=
by
  unfold canvas_bag_emission plastic_bag_emission bags_per_trip ounce_to_pound co2_trip 
  sorry

end canvas_bag_lower_carbon_solution_l521_521933


namespace acute_triangle_angles_l521_521663

theorem acute_triangle_angles (x y z : ℕ) (angle1 angle2 angle3 : ℕ) 
  (h1 : angle1 = 7 * x) 
  (h2 : angle2 = 9 * y) 
  (h3 : angle3 = 11 * z) 
  (h4 : angle1 + angle2 + angle3 = 180)
  (hx : 1 ≤ x ∧ x ≤ 12)
  (hy : 1 ≤ y ∧ y ≤ 9)
  (hz : 1 ≤ z ∧ z ≤ 8)
  (ha1 : angle1 < 90)
  (ha2 : angle2 < 90)
  (ha3 : angle3 < 90)
  : angle1 = 42 ∧ angle2 = 72 ∧ angle3 = 66 
  ∨ angle1 = 49 ∧ angle2 = 54 ∧ angle3 = 77 
  ∨ angle1 = 56 ∧ angle2 = 36 ∧ angle3 = 88 
  ∨ angle1 = 84 ∧ angle2 = 63 ∧ angle3 = 33 :=
sorry

end acute_triangle_angles_l521_521663


namespace probability_of_adjacent_vertices_in_decagon_l521_521593

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l521_521593


namespace centers_collinear_l521_521253

noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def circle_center (circle : Circle) : Point := sorry

theorem centers_collinear {A B C : Point} (K1 K2 K3 K4 : Circle)
  (h1 : K1.radius = K2.radius) (h2 : K2.radius = K3.radius) (h3 : K3.radius = K4.radius)
  (r : ℝ) (h4 : K1.radius = r) (h5 : K1.tangent_to_side A B)
  (h6 : K1.tangent_to_side A C) (h7 : K2.tangent_to_side B C)
  (h8 : K2.tangent_to_side B A) (h9 : K3.tangent_to_side C A)
  (h10 : K3.tangent_to_side C B) (h11 : K4.tangent_to_circle K1)
  (h12 : K4.tangent_to_circle K2) (h13 : K4.tangent_to_circle K3) :
  collinear (incenter A B C) (circumcenter A B C) (circle_center K4) :=
sorry

end centers_collinear_l521_521253


namespace triangle_side_count_l521_521829

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end triangle_side_count_l521_521829


namespace find_g_x_squared_minus_1_l521_521323

noncomputable def g : ℤ[X] := sorry -- polynomial g is defined but its exact form is not needed here as we're only interested in the final statement

theorem find_g_x_squared_minus_1 (x : ℤ) (h : g (x^2 - 2) = x^4 - 6 * x^2 + 8) :
  g (x^2 - 1) = x^4 - 4 * x^2 + 7 :=
by sorry

end find_g_x_squared_minus_1_l521_521323


namespace goblin_treasure_l521_521988

theorem goblin_treasure : 
  (∃ d : ℕ, 8000 + 300 * d = 5000 + 500 * d) ↔ ∃ (d : ℕ), d = 15 :=
by
  sorry

end goblin_treasure_l521_521988


namespace trig_identity_l521_521423

theorem trig_identity :
  sin (7 * π / 30) + sin (11 * π / 30) = sin (π / 30) + sin (13 * π / 30) + 1 / 2 :=
by
  sorry

end trig_identity_l521_521423


namespace line_equation_perpendicular_l521_521127

theorem line_equation_perpendicular
  (p : Point) (l : Line)
  (h₁ : p = ⟨0, 5⟩)
  (h₂ : l = λ ⟨x, y⟩, 2 * x - 5 * y + 1 = 0)
  : ∃ l' : Line, (l' = λ ⟨x, y⟩, 2 * y + 5 * x - 10 = 0) ∧
    (∀ p : Point, p ∈ l' → p = ⟨0, 5⟩ ∧ is_perpendicular l l') :=
sorry

end line_equation_perpendicular_l521_521127


namespace find_a_maximize_profit_l521_521107

theorem find_a (a: ℕ) (h: 600 * (a - 110) = 160 * a) : a = 150 :=
sorry

theorem maximize_profit (x y: ℕ) (a: ℕ) 
  (ha: a = 150)
  (hx: x + 5 * x + 20 ≤ 200) 
  (profit_eq: ∀ x, y = 245 * x + 600):
  x = 30 ∧ y = 7950 :=
sorry

end find_a_maximize_profit_l521_521107


namespace translated_parabola_expression_l521_521993

-- Original function definition
def original_parabola (x : ℝ) : ℝ := x^2

-- Transformation definitions
def translate_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, f (x - a)
def translate_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x, f x + b

-- Composed transformation function
def translate_right_and_up (f : ℝ → ℝ) (a b : ℝ) : ℝ → ℝ :=
  translate_up (translate_right f a) b

-- Problem statement
theorem translated_parabola_expression :
  translate_right_and_up original_parabola 2 1 = λ x, (x - 2)^2 + 1 :=
by
  sorry

end translated_parabola_expression_l521_521993


namespace meeting_point_distance_l521_521406

/-
 Points \( A, B, C \) are situated sequentially, with \( AB = 3 \) km and \( BC = 4 \) km. 
 A cyclist departed from point \( A \) heading towards point \( C \). 
 A pedestrian departed from point \( B \) heading towards point \( A \). 
 Both arrived at points \( A \) and \( C \) simultaneously. 
 Find the distance from point \( A \) at which they met.
-/

def distance_A_B : ℝ := 3
def distance_B_C : ℝ := 4
def distance_A_C : ℝ := distance_A_B + distance_B_C

theorem meeting_point_distance (V_C V_P : ℝ) (h_time_eq : 7 / V_C = 3 / V_P) : 
  ∃ x : ℝ, x = 2.1 :=
begin
  -- Definitions of the known distances
  let AB := distance_A_B,
  let BC := distance_B_C,
  let AC := distance_A_C,

  -- Set up the ratio of their speeds
  let speed_ratio := 7 / 3,

  -- Define distances covered by cyclist and pedestrian
  let x := 2.1, -- the distance we need to prove
  
  -- Check the ratio relationship
  -- Combine the facts to goal, direct straightforward calculation
  
  use x,
  exact rfl,
end

end meeting_point_distance_l521_521406


namespace range_of_x_l521_521286

noncomputable def functionY (x : ℝ) : ℝ := (sqrt (x + 1)) / x

theorem range_of_x {x : ℝ} : (x + 1 ≥ 0) ∧ (x ≠ 0) ↔ (x ≥ -1 ∧ x ≠ 0) :=
by
  split
  case mp => intros h; exact ⟨h.1, h.2⟩
  case mpr => intros h; exact ⟨h.1, h.2⟩

end range_of_x_l521_521286


namespace internal_angles_triangle_ABC_l521_521440

theorem internal_angles_triangle_ABC (α β γ : ℕ) (h₁ : α + β + γ = 180)
  (h₂ : α + γ = 138) (h₃ : β + γ = 108) : (α = 72) ∧ (β = 42) ∧ (γ = 66) :=
by
  sorry

end internal_angles_triangle_ABC_l521_521440


namespace possible_integer_side_lengths_l521_521796

theorem possible_integer_side_lengths : 
  {x : ℤ | x > 3 ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521796


namespace packs_used_after_6_weeks_l521_521684

-- Define the conditions as constants or definitions.
def pages_per_class_per_day : ℕ := 2
def num_classes : ℕ := 5
def days_per_week : ℕ := 5
def weeks : ℕ := 6
def pages_per_pack : ℕ := 100

-- The total number of packs of notebook paper Chip will use after 6 weeks
theorem packs_used_after_6_weeks : (pages_per_class_per_day * num_classes * days_per_week * weeks) / pages_per_pack = 3 := 
by
  -- skip the proof
  sorry

end packs_used_after_6_weeks_l521_521684


namespace solve_for_y_l521_521432

theorem solve_for_y :
  ∃ y, (y: ℝ >= 0) → (15: ℝ) * y + real.cbrt((15: ℝ) * y + 8: ℝ) = 8 -> 
  y = 168 / 5 :=
by
  sorry

end solve_for_y_l521_521432


namespace rope_to_pentagon_l521_521641

theorem rope_to_pentagon (length : ℝ) (segments : ℕ) (segment_length : ℝ) : 
  length = 2 ∧ segments = 5 ∧ segment_length = 2 / 5 → 
  ∀ i, i < segments → (4 * segment_length) > segment_length :=
by
  rintro ⟨h₁, h₂, h₃⟩ i hi
  have h4 : 4 * segment_length = 4 * (2 / 5) := by rw h₃
  have h5 : 4 * (2 / 5) = 8 / 5 := by norm_num
  rw [h4, h5]
  norm_num
  sorry

end rope_to_pentagon_l521_521641


namespace number_of_sides_possibilities_l521_521819

theorem number_of_sides_possibilities (a b : ℕ) (h₁ : a = 8) (h₂ : b = 5) : 
  { x : ℕ // 3 < x ∧ x < 13 }.to_list.length = 9 :=
by
  sorry

end number_of_sides_possibilities_l521_521819


namespace isosceles_trapezoid_area_l521_521038

theorem isosceles_trapezoid_area (b1 b2 l : ℕ) (h : ℕ) 
  (hb1 : b1 = 9) (hb2 : b2 = 15) (hl : l = 5) 
  (hheight : h = 4) :
  (1 / 2 : ℚ) * (b1 + b2) * h = 48 :=
by
  rw [hb1, hb2, hheight]
  norm_num
  sorry

end isosceles_trapezoid_area_l521_521038


namespace judah_crayons_l521_521897

theorem judah_crayons (karen beatrice gilbert judah : ℕ) 
  (h1 : karen = 128)
  (h2 : karen = 2 * beatrice)
  (h3 : beatrice = 2 * gilbert)
  (h4 : gilbert = 4 * judah) : 
  judah = 8 :=
by
  sorry

end judah_crayons_l521_521897


namespace find_a_n_l521_521217

def S (n : ℕ) : ℕ := 2^(n+1) - 1

def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2^n

theorem find_a_n (n : ℕ) : a_n n = if n = 1 then 3 else 2^n :=
  sorry

end find_a_n_l521_521217


namespace impossible_to_create_3_piles_l521_521337

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end impossible_to_create_3_piles_l521_521337


namespace final_value_of_P_l521_521457

theorem final_value_of_P :
  let P := 1;
  let P := (List.range 6).foldl (λ P _, P * 2) P;
  P = 64 :=
by
  sorry

end final_value_of_P_l521_521457


namespace speed_second_train_l521_521105

-- Define the conditions
def length_first_train : ℝ := 260
def speed_first_train : ℝ := 120  -- in kmph
def length_second_train : ℝ := 240.04
def opposite_directions : Prop := true  -- both trains are running in opposite directions
def crossing_time : ℝ := 9  -- in seconds

-- Define the theorem to be proved
theorem speed_second_train :
  let total_distance := length_first_train + length_second_train in
  let relative_speed_mps := total_distance / crossing_time in
  let relative_speed_kmph := relative_speed_mps * 3.6 in
  relative_speed_kmph = speed_first_train + 80.016 :=
by
  -- Proof omitted
  sorry

end speed_second_train_l521_521105


namespace slope_of_line_eq_45_deg_l521_521704

theorem slope_of_line_eq_45_deg (x y : ℝ) : 
  let line_eq := 2 * x - 2 * y + 1 = 0
  line_eq -> ∃ α : ℝ, tan α = 1 ∧ α = 45 :=
by
  sorry

end slope_of_line_eq_45_deg_l521_521704


namespace circumference_difference_is_30pi_l521_521257

-- Define the necessary variables and conditions
variables (r : ℝ)
def width : ℝ := 15
def radius_inner : ℝ := 2 * r
def radius_outer : ℝ := 2 * r + width

-- Define the circumferences for the inner and outer circles
def circumference_inner : ℝ := 2 * Real.pi * radius_inner
def circumference_outer : ℝ := 2 * Real.pi * radius_outer

-- Define the difference in circumferences
def circumference_difference : ℝ := circumference_outer - circumference_inner

-- State the theorem to be proven
theorem circumference_difference_is_30pi : circumference_difference = 30 * Real.pi :=
sorry

end circumference_difference_is_30pi_l521_521257


namespace largest_possible_c_l521_521912

theorem largest_possible_c (c : ℝ) (hc : (3 * c + 4) * (c - 2) = 9 * c) : c ≤ 4 :=
sorry

end largest_possible_c_l521_521912


namespace decagon_adjacent_probability_l521_521530

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l521_521530


namespace find_a_l521_521730

theorem find_a (f : ℝ → ℝ) (a x : ℝ) 
  (h1 : ∀ x, f (1/2 * x - 1) = 2 * x - 5) 
  (h2 : f a = 6) : a = 7 / 4 := 
by
  sorry

end find_a_l521_521730


namespace probability_harry_reaches_B_l521_521237

-- Define the junctions S, V, W, X, Y, Z and their paths explicitly.
inductive Junction
  | S | V | W | X | Y | Z

open Junction

def paths : Junction → List Junction
  | S => [V]
  | V => [X, W, Y]
  | W => [X]
  | Y => [X]
  | X => [B]
  | _ => []

def prob_move_to (from to : Junction) (paths : Junction → List Junction) : ℚ :=
  if to ∈ paths from then 1 / (paths from).length else 0

-- Define the total probability that Harry reaches junction B:
noncomputable def prob_reaching_B : ℚ :=
  prob_move_to V X paths +
  (prob_move_to V W paths * prob_move_to W X paths) +
  (prob_move_to V Y paths * prob_move_to Y X paths)

theorem probability_harry_reaches_B : prob_reaching_B = 11 / 18 :=
by
  sorry

end probability_harry_reaches_B_l521_521237


namespace num_possible_triangle_sides_l521_521804

theorem num_possible_triangle_sides (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (a_eq_8 : a = 8) (b_eq_5 : b = 5) : 
  let x := {n : ℕ | 3 < n ∧ n < 13} in
  set.card x = 9 :=
by
  sorry

end num_possible_triangle_sides_l521_521804


namespace problem_l521_521320

noncomputable def a := Real.log 2 / Real.log 3
noncomputable def b := Real.log (1/8) / Real.log 2
noncomputable def c := Real.sqrt 2

theorem problem : c > a ∧ a > b := 
by
  sorry

end problem_l521_521320


namespace ratio_height_base_l521_521123

-- Assume r and h are real numbers representing the radius and height of the cone, respectively
variables (r h : ℝ)

-- Define the slant height of the cone
def slant_height : ℝ := real.sqrt (r^2 + h^2)

-- Given that the cone makes 19 complete rotations and returns to its initial position
axiom cone_rotations (h r : ℝ) :
  2 * real.pi * slant_height r h = 19 * 2 * real.pi * r

-- Prove that h / r = 6 * real.sqrt 10
theorem ratio_height_base (r h : ℝ) (h_rotations : cone_rotations r h) :
  h / r = 6 * real.sqrt 10 :=
by
  sorry

end ratio_height_base_l521_521123


namespace decagon_adjacent_probability_l521_521577

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521577


namespace nonneg_real_values_of_x_for_given_condition_l521_521178

theorem nonneg_real_values_of_x_for_given_condition :
  ∃ (s : set ℝ), s.card = 14 ∧ ∀ x ∈ s, ∃ n : ℕ, n ≤ 13 ∧ x = (169 - n^2)^3 :=
by sorry

end nonneg_real_values_of_x_for_given_condition_l521_521178


namespace expected_value_dodecahedral_die_l521_521083

noncomputable def dodecahedral_expected_value : ℝ :=
  let outcomes := (list.range (12 + 1)).tail -- List of numbers from 1 to 12
  let n_faces := 12
  let probability := 1 / (n_faces : ℝ)
  let expected_value := probability * outcomes.sum / n_faces
  expected_value

theorem expected_value_dodecahedral_die :
  dodecahedral_expected_value = 6.5 :=
by {
  unfold dodecahedral_expected_value,
  simp,
  sorry,
}

end expected_value_dodecahedral_die_l521_521083


namespace plane_coverage_with_circles_1988_l521_521290

theorem plane_coverage_with_circles_1988 :
  ∃ (S : Set (Set (ℝ × ℝ))), (∀ (C ∈ S), ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ (p : ℝ × ℝ), p ∈ C ↔ dist p center = radius) ∧ (∀ (p : ℝ × ℝ), ∃ (circles : Set (Set (ℝ × ℝ))), circles ⊆ S ∧ circles.card = 1988 ∧ ∀ (C ∈ circles), p ∈ C) :=
sorry

end plane_coverage_with_circles_1988_l521_521290


namespace Dave_won_tickets_l521_521674

theorem Dave_won_tickets :
  ∀ (tickets_toys tickets_clothes total_tickets : ℕ),
  (tickets_toys = 8) →
  (tickets_clothes = 18) →
  (tickets_clothes = tickets_toys + 10) →
  (total_tickets = tickets_toys + tickets_clothes) →
  total_tickets = 26 :=
by
  intros tickets_toys tickets_clothes total_tickets h1 h2 h3 h4
  have h5 : tickets_clothes = 8 + 10 := by sorry
  have h6 : tickets_clothes = 18 := by sorry
  have h7 : tickets_clothes = 18 := by sorry
  exact sorry

end Dave_won_tickets_l521_521674


namespace total_shaded_cubes_is_sixteen_l521_521435

-- Definition of the cube and shading conditions
def large_cube : Type := array (4 × 4 × 4) (cube)

-- Each face has a mirrored shading pattern
def shading_pattern (face : ℕ) : Prop :=
  (face = 0 ∨ face = 1 ∨ face = 2 ∨ face = 3 ∨ face = 4 ∨ face = 5) ∧
  (shading_center face ∧ shading_corner face) 

-- Define the shaded cubes counting condition
def shaded_cubes (cube : large_cube) : ℕ :=
  let center_shaded := 6 * 4 in
  let corners_shaded := 8 / 2 in
  center_shaded + corners_shaded

-- The final theorem statement proving the total unique shaded cubes equals 16
theorem total_shaded_cubes_is_sixteen (cube : large_cube) : shaded_cubes cube = 16 :=
by
  sorry

end total_shaded_cubes_is_sixteen_l521_521435


namespace sum_first_5_terms_geometric_series_l521_521143

theorem sum_first_5_terms_geometric_series :
  (let a : ℚ := 1
   let r : ℚ := 1 / 4
   let n : ℕ := 5
   let Sn : ℚ := a * (1 - r^n) / (1 - r)
   in Sn = 341 / 256) :=
by 
  -- Proof would go here
  sorry

end sum_first_5_terms_geometric_series_l521_521143


namespace polynomial_roots_product_l521_521152

theorem polynomial_roots_product :
  (∃ b c : ℤ, (∀ r : ℝ, r^2 - r - 1 = 0 → r^6 - b * r - c = 0) ∧ b * c = 40) :=
by
  let r := (1 + Real.sqrt 5) / 2
  have root_r : r^2 - r - 1 = 0 := by sorry  -- Proof that r is a root for x^2 - x - 1
  let b := 8
  let c := 5
  have root_poly : r^6 - b * r - c = 0 := by sorry  -- Proof that r satisfies the second polynomial
  existsi [b, c]
  split
  { intro r hr,
    sorry,  -- Full verification that any root of the first polynomial satisfies the second
  }
  { exact Eq.refl (b * c) }

end polynomial_roots_product_l521_521152


namespace sum_of_a_and_b_l521_521244

theorem sum_of_a_and_b (a b : ℝ)
  (hM : set.M = {b / a, 1})
  (hN : set.N = {a, 0})
  (hf : ∀ x ∈ hM, x ∈ hN) :
  a + b = 1 :=
by
  sorry

end sum_of_a_and_b_l521_521244


namespace constant_term_is_neg2_l521_521216

-- Condition: The sum of all coefficients in the expansion of (1 + ax)^4 is 81 and a > 0.
def sum_of_coefficients_eq_81 (a : ℝ) (h : 0 < a) : Prop := (1 + a)^4 = 81

-- Problem: Find the constant term in the expansion of (1 + x)^{2a} * (2 - 1/x).
def find_constant_term (a : ℝ) (h : sum_of_coefficients_eq_81 a (by positivity)) : ℝ :=
  let expansion := (1 + x)^(2*a) * (2 - 1/x),
  -- We need to extract the constant term from this expansion.
  -- Since computations and actual expansion are complex, we use "sorry" for the implementation.
  sorry

-- Assertion: The expected constant term is -2.
theorem constant_term_is_neg2 (a : ℝ) (h : sum_of_coefficients_eq_81 a (by positivity)) : find_constant_term a h = -2 := 
  by
    -- Directly state the result we expect.
    sorry

end constant_term_is_neg2_l521_521216


namespace geom_sequence_general_formula_lambda_range_l521_521740

variable {a : ℕ → ℤ}

-- Let 'a' be a sequence satisfying the given initial conditions.
def seq_conditions (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n : ℕ, 0 < n → a (n + 1) - 2 * a n = 2 * n

-- Statement 1: Prove that the given sequence adjusted by 2n + 2 forms a geometric sequence.
theorem geom_sequence {a : ℕ → ℤ} (h : seq_conditions a) :
  ∃ r (c : ℕ → ℤ), (r = 2 ∧ c 1 = 4) ∧ (∀ n : ℕ, 0 < n → a (n + 1) + 2 * (n + 1) + 2 = r * (a n + 2 * n + 2)) :=
sorry

-- Statement 2: Find the general formula for the sequence.
theorem general_formula {a : ℕ → ℤ} (h : seq_conditions a) :
  ∀ n : ℕ, 0 < n → a n = 2 ^ (n + 1) - 2 * n - 2 :=
sorry

-- Statement 3: Given the inequality condition involving lambda, find its range.
theorem lambda_range {a : ℕ → ℤ} (h : seq_conditions a) (λ : ℝ) :
  (∀ n : ℕ, 0 < n → a n > λ * (2 * n + 1) * (-1) ^ (n - 1)) ↔ ( -2 / 5 < λ ∧ λ < 0) :=
sorry

end geom_sequence_general_formula_lambda_range_l521_521740


namespace arithmetic_mean_of_two_numbers_l521_521939

def is_arithmetic_mean (x y z : ℚ) : Prop :=
  (x + z) / 2 = y

theorem arithmetic_mean_of_two_numbers :
  is_arithmetic_mean (9 / 12) (5 / 6) (7 / 8) :=
by
  sorry

end arithmetic_mean_of_two_numbers_l521_521939


namespace problem1_problem2_l521_521425

-- Prove that the first expression simplifies to 8(π - 3)
theorem problem1 : 
  (2 * Real.sqrt 3) * (Real.cbrt 1.5) * (Real.sqrt (Real.sqrt (12))) * (Real.sqrt ((3 - Real.pi)^2)) = 
  8 * (Real.pi - 3) := 
sorry

-- Prove that the second expression simplifies to 4
theorem problem2 : 
  Real.log10 25 + (2 / 3) * Real.log10 8 + (Real.log10 5) * (Real.log10 20) + (Real.log10 2)^2 = 
  4 := 
sorry

end problem1_problem2_l521_521425


namespace part_a_int_values_part_b_int_values_l521_521724

-- Part (a)
theorem part_a_int_values (n : ℤ) :
  ∃ k : ℤ, (n^4 + 3) = k * (n^2 + n + 1) ↔ n = -3 ∨ n = -1 ∨ n = 0 :=
sorry

-- Part (b)
theorem part_b_int_values (n : ℤ) :
  ∃ m : ℤ, (n^3 + n + 1) = m * (n^2 - n + 1) ↔ n = 0 ∨ n = 1 :=
sorry

end part_a_int_values_part_b_int_values_l521_521724


namespace example_problem_l521_521096

variable (a b c d : ℝ)

theorem example_problem :
  (a + (b + c - d) = a + b + c - d) ∧
  (a - (b - c + d) = a - b + c - d) ∧
  (a - b - (c - d) ≠ a - b - c - d) ∧
  (a + b - (-c - d) = a + b + c + d) :=
by {
  sorry
}

end example_problem_l521_521096


namespace expected_value_fair_dodecahedral_die_l521_521058

theorem expected_value_fair_dodecahedral_die :
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  expected_value = 6.5 :=
by
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  have h1 : list.sum outcomes = 78 := by sorry
  have h2 : (12: ℝ) * (1 / 12: ℝ) = 1 := by sorry
  show expected_value = 6.5
  { sorry }

end expected_value_fair_dodecahedral_die_l521_521058


namespace integral_curve_solution_l521_521171

noncomputable def y (x : ℝ) : ℝ := real.exp (-x) * (real.cos x + 2 * real.sin x)

theorem integral_curve_solution :
  ∀ x, (deriv (deriv y) x + 2 * deriv y x + 2 * y x = 0) ∧ (y 0 = 1) ∧ (deriv y 0 = 1) :=
by
  sorry

end integral_curve_solution_l521_521171


namespace triangle_possible_sides_l521_521785

theorem triangle_possible_sides : 
  let unknown_side_lengths := {x : ℤ | 3 < x ∧ x < 13}
  in set.finite unknown_side_lengths ∧ set.card unknown_side_lengths = 9 :=
by
  sorry

end triangle_possible_sides_l521_521785


namespace factor_expression_l521_521709

theorem factor_expression (x : ℝ) : 46 * x^3 - 115 * x^7 = -23 * x^3 * (5 * x^4 - 2) := 
by
  sorry

end factor_expression_l521_521709


namespace tank_capacity_l521_521649

-- Given conditions
variable (capacity : ℚ) -- the total capacity of the tank

-- The initial volume of water in the tank is 1/8 of its total capacity
def initial_volume := capacity * (1/8 : ℚ)

-- After adding 150 gallons of water, the tank's volume becomes 2/3 of its total capacity
def final_volume := capacity * (2/3 : ℚ)

-- The difference added is 150 gallons
def added_water := 150

-- We state that initially the tank is 1/8 full, after adding 150 gallons it's 2/3 full
def addition_condition := final_volume - initial_volume = added_water

-- The proof goal
theorem tank_capacity : addition_condition capacity → capacity = 277 :=
by
  intro h
  sorry

end tank_capacity_l521_521649


namespace decagon_adjacent_probability_l521_521526

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521526


namespace probability_sum_to_four_l521_521188

theorem probability_sum_to_four :
  let balls := [1, 2, 2, 3, 3, 3]
  let total_pairs := (6.choose 2 : ℕ)
  let favorable_pairs := (1 + 3 : ℕ)
  favorable_pairs / total_pairs = 4/15 := by
  sorry

end probability_sum_to_four_l521_521188


namespace tanya_work_time_l521_521950

theorem tanya_work_time (
    sakshi_work_time : ℝ := 20,
    tanya_efficiency : ℝ := 1.25
) : 
    let sakshi_rate : ℝ := 1 / sakshi_work_time in
    let tanya_rate : ℝ := tanya_efficiency * sakshi_rate in
    let tanya_work_time := 1 / tanya_rate in
    tanya_work_time = 16 :=
by
    let sakshi_rate : ℝ := 1 / sakshi_work_time
    let tanya_rate : ℝ := tanya_efficiency * sakshi_rate
    let tanya_time : ℝ := 1 / tanya_rate
    show tanya_time = 16
    sorry

end tanya_work_time_l521_521950


namespace expected_value_dodecahedral_die_is_6_5_l521_521060

noncomputable def expected_value_dodecahedral_die : ℝ :=
  let outcomes := finset.range 12 in
  1 / 12 * (finset.sum outcomes (λ i, (i + 1 : ℝ)))

theorem expected_value_dodecahedral_die_is_6_5 :
  expected_value_dodecahedral_die = 6.5 :=
by
  sorry

end expected_value_dodecahedral_die_is_6_5_l521_521060


namespace time_to_ascend_non_working_escalator_l521_521469

-- Definitions from the conditions
def length_of_escalator := 1
def time_standing := 1
def time_running := 24 / 60
def escalator_speed := 1 / 60
def gavrila_speed := 1 / 40

-- The proof problem statement 
theorem time_to_ascend_non_working_escalator 
  (length_of_escalator : ℝ)
  (time_standing : ℝ)
  (time_running : ℝ)
  (escalator_speed : ℝ)
  (gavrila_speed : ℝ) :
  time_standing = 1 →
  time_running = 24 / 60 →
  escalator_speed = 1 / 60 →
  gavrila_speed = 1 / 40 →
  length_of_escalator = 1 →
  1 / gavrila_speed = 40 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end time_to_ascend_non_working_escalator_l521_521469


namespace meeting_point_distance_l521_521407

/-
 Points \( A, B, C \) are situated sequentially, with \( AB = 3 \) km and \( BC = 4 \) km. 
 A cyclist departed from point \( A \) heading towards point \( C \). 
 A pedestrian departed from point \( B \) heading towards point \( A \). 
 Both arrived at points \( A \) and \( C \) simultaneously. 
 Find the distance from point \( A \) at which they met.
-/

def distance_A_B : ℝ := 3
def distance_B_C : ℝ := 4
def distance_A_C : ℝ := distance_A_B + distance_B_C

theorem meeting_point_distance (V_C V_P : ℝ) (h_time_eq : 7 / V_C = 3 / V_P) : 
  ∃ x : ℝ, x = 2.1 :=
begin
  -- Definitions of the known distances
  let AB := distance_A_B,
  let BC := distance_B_C,
  let AC := distance_A_C,

  -- Set up the ratio of their speeds
  let speed_ratio := 7 / 3,

  -- Define distances covered by cyclist and pedestrian
  let x := 2.1, -- the distance we need to prove
  
  -- Check the ratio relationship
  -- Combine the facts to goal, direct straightforward calculation
  
  use x,
  exact rfl,
end

end meeting_point_distance_l521_521407


namespace picked_tomatoes_eq_53_l521_521630

-- Definitions based on the conditions
def initial_tomatoes : ℕ := 177
def initial_potatoes : ℕ := 12
def items_left : ℕ := 136

-- Define what we need to prove
theorem picked_tomatoes_eq_53 : initial_tomatoes + initial_potatoes - items_left = 53 :=
by sorry

end picked_tomatoes_eq_53_l521_521630


namespace minimum_directed_edge_chromatic_number_l521_521652

open nat

def tournament (V : Type) := V × V

def proper_directed_edge_coloring (V : Type) (f : tournament V → ℕ) : Prop :=
  ∀ (u v w : V), u ≠ v ∧ v ≠ w ∧ u ≠ w → f (u, v) ≠ f (v, w)

def directed_edge_chromatic_number (V : Type) [fintype V] : ℕ :=
  inf { k : ℕ | ∃ (f : tournament V → ℕ), proper_directed_edge_coloring V f ∧ ∀ e, f e < k }

theorem minimum_directed_edge_chromatic_number (n : ℕ) (hn : n > 0) :
  ∃ V [fintype V] (G : tournament V), fintype.card V = n →
  directed_edge_chromatic_number V = ⌈log 2 n⌉ :=
sorry

end minimum_directed_edge_chromatic_number_l521_521652


namespace not_possible_three_piles_l521_521382

def similar (x y : ℝ) : Prop :=
  y ≤ (real.sqrt 2) * x ∧ x ≤ (real.sqrt 2) * y

theorem not_possible_three_piles (N : ℝ) : 
  ¬ ∃ a b c : ℝ, a + b + c = N ∧ similar a b ∧ similar b c ∧ similar a c := 
by
  sorry

end not_possible_three_piles_l521_521382


namespace chip_notebook_packs_l521_521683

theorem chip_notebook_packs (pages_per_day : ℕ) (days_per_week : ℕ) (classes : ℕ) (sheets_per_pack : ℕ) (weeks : ℕ) :
  pages_per_day = 2 → days_per_week = 5 → classes = 5 → sheets_per_pack = 100 → weeks = 6 →
  (classes * pages_per_day * days_per_week * weeks) / sheets_per_pack = 3 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  sorry

end chip_notebook_packs_l521_521683


namespace no_integer_solutions_l521_521844

theorem no_integer_solutions (x y z : ℤ) :
  x^2 - 3 * x * y + 2 * y^2 - z^2 = 31 ∧
  -x^2 + 6 * y * z + 2 * z^2 = 44 ∧
  x^2 + x * y + 8 * z^2 = 100 →
  false :=
by
  sorry

end no_integer_solutions_l521_521844


namespace temperature_range_l521_521394

theorem temperature_range (t : ℕ) : (21 ≤ t ∧ t ≤ 29) :=
by
  sorry

end temperature_range_l521_521394


namespace range_of_a_l521_521762

noncomputable def f (a x : ℝ) : ℝ :=
if x < 3 then (a + 1) * x - 2 * a else Real.log 3 x

theorem range_of_a (a : ℝ) : (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a > -1 :=
by
  sorry

end range_of_a_l521_521762


namespace possible_integer_side_lengths_l521_521799

theorem possible_integer_side_lengths : 
  {x : ℤ | x > 3 ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521799


namespace decagon_adjacent_probability_l521_521580

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521580


namespace smallest_possible_sum_l521_521429

-- Define variables and conditions
variable (A B C D E F : ℕ)
variable (adjacent : (ℕ × ℕ → Prop))
variable (cube_faces : List ℕ := [A, B, C, D, E, F])

-- Condition: Natural numbers on the faces of a cube such that the numbers 
-- on adjacent faces differ by more than 1
def valid_faces (n1 n2 : ℕ) : Prop :=
  (n1 ≠ n2) ∧ (abs (n1 - n2) > 1)

-- Condition: Define adjacency relation in cube
def cube_adjacency : (ℕ × ℕ → Prop) := λ (n1 n2 : ℕ), match (n1, n2) with
  | (A, B), (A, C), (A, D), (B, E), (C, F), (D, E), (E, F) => valid_faces n1 n2
  | _ => false

theorem smallest_possible_sum : ∃ face_sum : ℕ, 
  all (λ (pair : ℕ × ℕ), adjacent pair) (filter cube_adjacency (list.nat.pairs cube_faces))
  ∧ face_sum = (A + B + C + D + E + F) ∧ face_sum = 18 :=
sorry

end smallest_possible_sum_l521_521429


namespace find_larger_number_l521_521495

variable (L S : ℕ)

theorem find_larger_number 
  (h1 : L - S = 1355) 
  (h2 : L = 6 * S + 15) : 
  L = 1623 := 
sorry

end find_larger_number_l521_521495


namespace alpha_sin_cos_proof_l521_521737

noncomputable def alpha_sin_cos_expr : ℝ :=
  let α := arbitrary ℝ in
  let sin_α := 3 / 5 in
  let cos_α := -4 / 5 in
  (cos (π / 2 + α) * sin (-π - α)) / 
  (cos (11 * π / 2 - α) * sin (9 * π / 2 + α))

theorem alpha_sin_cos_proof :
  (cos (π / 2 + α) * sin (-π - α)) / 
  (cos (11 * π / 2 - α) * sin (9 * π / 2 + α)) = -3 / 4 :=
by 
  let α := arbitrary ℝ
  have sin_α : sin α = 3 / 5 := sorry
  have cos_α : cos α = -4 / 5 := sorry
  sorry

end alpha_sin_cos_proof_l521_521737


namespace impossible_to_create_3_piles_l521_521334

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end impossible_to_create_3_piles_l521_521334


namespace minimum_value_of_my_function_l521_521714

noncomputable def my_function (x : ℝ) : ℝ :=
  (Real.cos x) ^ 2 + (sqrt 3 / 2) * Real.sin (2 * x)

theorem minimum_value_of_my_function (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) :
  ∃ y, y = 0 ∧ my_function x = y :=
sorry

end minimum_value_of_my_function_l521_521714


namespace not_possible_to_create_3_piles_l521_521358

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end not_possible_to_create_3_piles_l521_521358


namespace probability_of_one_blue_l521_521618

noncomputable def jellybean_probability : ℚ :=
  let total_combinations := nat.choose 12 3 in
  let successful_combinations := (nat.choose 2 1) * (nat.choose 10 2) in
  successful_combinations / total_combinations

theorem probability_of_one_blue (total_jellybeans : ℕ) (red : ℕ) (blue : ℕ) (white : ℕ) (picked : ℕ)
  (h1 : total_jellybeans = 12) (h2 : red = 5) (h3 : blue = 2) (h4 : white = 5) (h5 : picked = 3) :
  jellybean_probability = 9 / 22 :=
by
  -- Proof omitted
  sorry

end probability_of_one_blue_l521_521618


namespace clock_bell_rings_8760_times_in_a_year_l521_521625

/-- 
The clock tower bell rings once every hour.
There are 24 hours in a day.
A year is counted as 365 days.
Prove that the clock tower bell rings 8760 times in a year.
-/
theorem clock_bell_rings_8760_times_in_a_year :
  (∃ (rings_per_hour rings_per_day rings_per_year : ℕ),
    rings_per_hour = 1 ∧
    rings_per_day = 24 ∧
    rings_per_year = 365 * rings_per_day ∧
    rings_per_year = 8760) :=
begin
  use 1,
  use 24,
  use 365 * 24,
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  { norm_num }
end

end clock_bell_rings_8760_times_in_a_year_l521_521625


namespace cosine_angle_l521_521234

theorem cosine_angle (t θ : ℝ) (cosθ : ℝ)
  (a : ℝ × ℝ := (1, -2))
  (b : ℝ × ℝ := (3, t))
  (c : ℝ × ℝ := (1, -3))
  (h_perp : b.1 * c.1 + b.2 * c.2 = 0)
  (t_eq : t = 1)
  (magn_a : ℝ := real.sqrt (a.1^2 + a.2^2))
  (magn_b : ℝ := real.sqrt (b.1^2 + b.2^2))
  (dot_ab : ℝ := a.1 * b.1 + a.2 * b.2)
  (h_cos : cosθ = dot_ab / (magn_a * magn_b))
  : cosθ = real.sqrt 2 / 10 := 
sorry

end cosine_angle_l521_521234


namespace triangle_side_lengths_l521_521836

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end triangle_side_lengths_l521_521836


namespace red_pigment_contribution_l521_521621

theorem red_pigment_contribution 
  (blue_pigment_percent_red : blue_pigment_percent_red = 0.40)
  (red_pigment_percent_red : red_pigment_percent_red = 0.60)
  (blue_pigment_percent_green : blue_pigment_percent_green = 0.40)
  (yellow_pigment_percent_green : yellow_pigment_percent_green = 0.50)
  (white_pigment_percent_green : white_pigment_percent_green = 0.10)
  (red_pigment_percent_orange : red_pigment_percent_orange = 0.50)
  (yellow_pigment_percent_orange : yellow_pigment_percent_orange = 0.40)
  (white_pigment_percent_orange : white_pigment_percent_orange = 0.10)
  (blue_pigment_percent_brown : blue_pigment_percent_brown = 0.35)
  (red_pigment_percent_brown : red_pigment_percent_brown = 0.25)
  (yellow_pigment_percent_brown : yellow_pigment_percent_brown = 0.30)
  (white_pigment_percent_brown : white_pigment_percent_brown = 0.10)
  (brown_paint_weight : brown_paint_weight = 20) :
  (red_pigment_percent_brown * brown_paint_weight) = 5 :=
by
  sorry

end red_pigment_contribution_l521_521621


namespace not_possible_to_create_3_similar_piles_l521_521347

-- Formalize the definition of similar piles
def similar (a b : ℝ) : Prop := a <= sqrt 2 * b ∧ b <= sqrt 2 * a

-- State the main theorem
theorem not_possible_to_create_3_similar_piles (x₀ : ℝ) (h₀ : 0 < x₀) : 
  ¬ ∃ x₁ x₂ x₃ : ℝ, 
    x₁ + x₂ + x₃ = x₀ ∧ 
    similar x₁ x₂ ∧ 
    similar x₂ x₃ ∧ 
    similar x₃ x₁ :=
sorry

end not_possible_to_create_3_similar_piles_l521_521347


namespace possible_integer_side_lengths_l521_521822

theorem possible_integer_side_lengths (a b : ℕ) (h1 : a = 8) (h2 : b = 5) :
  {x : ℕ | 3 < x ∧ x < 13}.finite ∧ {x : ℕ | 3 < x ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521822


namespace meeting_distance_from_A_l521_521403

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (distance_AB distance_BC : ℝ)
variable (cyclist_speed pedestrian_speed : ℝ)
variable (meet_distance : ℝ)

axiom distance_AB_eq_3 : distance_AB = 3
axiom distance_BC_eq_4 : distance_BC = 4
axiom simultaneous_arrival :
  ∀ AC cyclist_speed pedestrian_speed,
    (distance_AB + distance_BC) / cyclist_speed = distance_AB / pedestrian_speed
axiom speed_ratio :
  cyclist_speed / pedestrian_speed = 7 / 3
axiom meeting_point :
  ∃ meet_distance,
    meet_distance / (distance_AB - meet_distance) = 7 / 3

theorem meeting_distance_from_A :
  meet_distance = 2.1 :=
sorry

end meeting_distance_from_A_l521_521403


namespace length_BP_l521_521921

noncomputable def equilateral_triangle :=
  ∃ (A B C : ℝ) (side : ℝ), side = 6 ∧ 
  A = 0 ∧ 
  B = side ∧ 
  C = complex.conj B

noncomputable def circle (A B C : ℝ) : Prop :=
  ∃ (center : ℝ) (radius : ℝ), 
    center = complex.conj ((A + B + C) / 3) ∧
    radius = side^2 / (3 * complex.abs3 center)

noncomputable def point_on_arc (A C P : ℝ) : Prop :=
  ∃ (AP PC : ℝ), AP * PC = 10 ∧
  ∃ (θ : ℝ), θ = ∡(A, C) / 2 ∧
  P = complex.polar radius θ

open real

theorem length_BP :
  ∀ (A B C P : ℝ) (side : ℝ),
  equilateral_triangle A B C side → 
  circle A B C →
  point_on_arc A C P → 
  (abs (B - P) = sqrt 26) :=
by
  sorry

end length_BP_l521_521921


namespace decagon_adjacent_probability_l521_521532

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l521_521532


namespace cross_product_magnitude_l521_521233

-- Definitions of the vectors
def a : ℝ × ℝ := (-3, 4)
def b : ℝ × ℝ := (0, 2)

-- Function to calculate magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Function to calculate dot product of two vectors
def dot (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Function to calculate cos(theta)
noncomputable def cos_theta (v1 v2 : ℝ × ℝ) : ℝ :=
  dot v1 v2 / (magnitude v1 * magnitude v2)

-- Function to calculate sin(theta)
noncomputable def sin_theta (v1 v2 : ℝ × ℝ) : ℝ :=
  real.sqrt (1 - (cos_theta v1 v2) ^ 2)

-- The proof statement
theorem cross_product_magnitude : abs (6) = 6 :=
by
  -- Definitions
  have mag_a := magnitude a
  have mag_b := magnitude b
  have sin_theta_ab := sin_theta a b

  -- Compute |a x b|
  let mag_cross_product := mag_a * mag_b * sin_theta_ab

  -- Check the value
  have h : mag_cross_product = 6 := sorry -- to be proven

  done

end cross_product_magnitude_l521_521233


namespace coefficient_a2bc_n_3_expansion_l521_521731

noncomputable def n : ℕ := 5 * (∫ x in 0..real.pi, real.sin x)

theorem coefficient_a2bc_n_3_expansion :
  let n := 5 * (∫ x in 0..real.pi, real.sin x)
  in  (2:ℚ)^2 * (-3:ℚ) * (nat.choose n 2) = -4320 :=
by
  sorry

end coefficient_a2bc_n_3_expansion_l521_521731


namespace second_player_wins_l521_521989

noncomputable def is_winning_position (n : ℕ) : Prop :=
  n % 4 = 0

theorem second_player_wins (n : ℕ) (h : n = 100) :
  ∃ f : ℕ → ℕ, (∀ k, 0 < k → k ≤ n → (k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 5) → is_winning_position (n - k)) ∧ is_winning_position n := 
sorry

end second_player_wins_l521_521989


namespace eldest_age_l521_521500

theorem eldest_age (A B C : ℕ) (x : ℕ) 
  (h1 : A = 5 * x)
  (h2 : B = 7 * x)
  (h3 : C = 8 * x)
  (h4 : (5 * x - 7) + (7 * x - 7) + (8 * x - 7) = 59) :
  C = 32 := 
by 
  sorry

end eldest_age_l521_521500


namespace inequality_solution_set_l521_521005

theorem inequality_solution_set :
  { x : ℝ | (3 * x + 1) / (x - 2) ≤ 0 } = { x : ℝ | -1/3 ≤ x ∧ x < 2 } :=
sorry

end inequality_solution_set_l521_521005


namespace periodic_function_l521_521609

noncomputable def f : ℝ → ℝ := sorry

theorem periodic_function
  (h₁ : ∀ x, continuous_at f x)
  (h₂ : ∀ n : ℕ, 1 ≤ n → ∫ x in 0..n, f x * f (n - x) = ∫ x in 0..n, (f x) ^ 2) :
  ∃ p : ℝ, 0 < p ∧ ∀ x, f x = f (x + p) :=
sorry

end periodic_function_l521_521609


namespace adjusted_ratio_l521_521158

theorem adjusted_ratio :
  (2^2003 * 3^2005) / (6^2004) = 3 / 2 :=
by
  sorry

end adjusted_ratio_l521_521158


namespace not_possible_to_create_3_similar_piles_l521_521345

-- Formalize the definition of similar piles
def similar (a b : ℝ) : Prop := a <= sqrt 2 * b ∧ b <= sqrt 2 * a

-- State the main theorem
theorem not_possible_to_create_3_similar_piles (x₀ : ℝ) (h₀ : 0 < x₀) : 
  ¬ ∃ x₁ x₂ x₃ : ℝ, 
    x₁ + x₂ + x₃ = x₀ ∧ 
    similar x₁ x₂ ∧ 
    similar x₂ x₃ ∧ 
    similar x₃ x₁ :=
sorry

end not_possible_to_create_3_similar_piles_l521_521345


namespace min_voters_tall_giraffe_win_l521_521280

-- Definitions from the problem statement as conditions
def precinct_voters := 3
def precincts_per_district := 9
def districts := 5
def majority_precincts(p : ℕ) := p / 2 + 1  -- Minimum precincts won in a district 
def majority_districts(d : ℕ) := d / 2 + 1  -- Minimum districts won in the final

-- Condition: majority precincts to win a district
def precinct_votes_to_win := majority_precincts precinct_voters

-- Condition: majority districts to win the final
def district_wins_to_win_final := majority_districts districts

-- Minimum precincts the Tall giraffe needs to win overall
def total_precincts_to_win := district_wins_to_win_final * majority_precincts precincts_per_district

-- Proof that the minimum number of voters who could have voted for the Tall giraffe is 30
theorem min_voters_tall_giraffe_win :
  precinct_votes_to_win * total_precincts_to_win = 30 :=
sorry

end min_voters_tall_giraffe_win_l521_521280


namespace translated_parabola_expression_l521_521992

-- Original function definition
def original_parabola (x : ℝ) : ℝ := x^2

-- Transformation definitions
def translate_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, f (x - a)
def translate_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x, f x + b

-- Composed transformation function
def translate_right_and_up (f : ℝ → ℝ) (a b : ℝ) : ℝ → ℝ :=
  translate_up (translate_right f a) b

-- Problem statement
theorem translated_parabola_expression :
  translate_right_and_up original_parabola 2 1 = λ x, (x - 2)^2 + 1 :=
by
  sorry

end translated_parabola_expression_l521_521992


namespace decagon_adjacent_probability_l521_521581

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521581


namespace swimming_pool_width_l521_521018

theorem swimming_pool_width
  (length : ℝ)
  (lowered_height_inches : ℝ)
  (removed_water_gallons : ℝ)
  (gallons_per_cubic_foot : ℝ)
  (volume_for_removal : ℝ)
  (width : ℝ) :
  length = 60 → 
  lowered_height_inches = 6 →
  removed_water_gallons = 4500 →
  gallons_per_cubic_foot = 7.5 →
  volume_for_removal = removed_water_gallons / gallons_per_cubic_foot →
  width = volume_for_removal / (length * (lowered_height_inches / 12)) →
  width = 20 :=
by
  intros h_length h_lowered_height h_removed_water h_gallons_per_cubic_foot h_volume_for_removal h_width
  sorry

end swimming_pool_width_l521_521018


namespace not_possible_to_create_3_similar_piles_l521_521343

-- Formalize the definition of similar piles
def similar (a b : ℝ) : Prop := a <= sqrt 2 * b ∧ b <= sqrt 2 * a

-- State the main theorem
theorem not_possible_to_create_3_similar_piles (x₀ : ℝ) (h₀ : 0 < x₀) : 
  ¬ ∃ x₁ x₂ x₃ : ℝ, 
    x₁ + x₂ + x₃ = x₀ ∧ 
    similar x₁ x₂ ∧ 
    similar x₂ x₃ ∧ 
    similar x₃ x₁ :=
sorry

end not_possible_to_create_3_similar_piles_l521_521343


namespace triangle_side_lengths_count_l521_521791

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end triangle_side_lengths_count_l521_521791


namespace seq_form_l521_521162

-- Define the sequence a as a function from natural numbers to natural numbers
def seq (a : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, 0 < m → 0 < n → ⌊(a m : ℚ) / a n⌋ = ⌊(m : ℚ) / n⌋

-- Define the statement that all sequences satisfying the condition must be of the form k * i
theorem seq_form (a : ℕ → ℕ) : seq a → ∃ k : ℕ, (0 < k) ∧ (∀ n, 0 < n → a n = k * n) := 
by
  intros h
  sorry

end seq_form_l521_521162


namespace warm_pot_production_time_l521_521092

theorem warm_pot_production_time :
  ∀ (t_cold t_prod1 extra_pots t_total t_warm : ℕ),
  t_cold = 6 →
  t_prod1 = 10 →
  extra_pots = 2 →
  t_total = 60 →
  (t_prod1 + extra_pots) * t_warm = t_total →
  t_warm = 5 :=
by
  intros t_cold t_prod1 extra_pots t_total t_warm
  assume h1 h2 h3 h4 h5
  sorry

end warm_pot_production_time_l521_521092


namespace math_problem_solution_part1_math_problem_solution_part2_l521_521904

noncomputable def part1 (A B : set ℝ) (a : ℝ) : Prop :=
  A = {x | x^2 - 3 * x + 2 = 0} ∧ 
  B = {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0} → 
  (A ∩ B = {2} → (a = -1 ∨ a = -3))

noncomputable def part2 (A B U : set ℝ) (a : ℝ) : Prop :=
  U = set.univ ∧ 
  A = {x | x^2 - 3 * x + 2 = 0} ∧ 
  B = {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0} →
  (A ∩ (U \ B) = A → 
    (a < -3 ∨ (-3 < a ∧ a < -1 - real.sqrt 3) ∨ 
    (-1 - real.sqrt 3 < a ∧ a < -1) ∨ 
    (-1 < a ∧ a < -1 + real.sqrt 3) ∨ 
    a > -1 + real.sqrt 3))

theorem math_problem_solution_part1 (A B : set ℝ) (a : ℝ) : part1 A B a :=
  sorry

theorem math_problem_solution_part2 (A B U : set ℝ) (a : ℝ) : part2 A B U a :=
  sorry

end math_problem_solution_part1_math_problem_solution_part2_l521_521904


namespace minimum_voters_for_tall_giraffe_l521_521283

-- Definitions and conditions
def voters : ℕ := 135
def districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3

-- Conditions encoded in the problem statement
def majority_precinct (votes: ℕ) : Prop := votes >= 2
def majority_district (precinct_wins: ℕ) : Prop := precinct_wins >= 5
def majority_winner (district_wins: ℕ) : Prop := district_wins >= 3

-- The problem states that the Tall giraffe won.
axiom tall_giraffe_won : ∃ district_wins : ℕ, 
  majority_winner district_wins ∧ 
  ∀ (d ∈ (finset.range districts)), ∃ precinct_wins : ℕ, 
  majority_district precinct_wins ∧ 
  ∀ (p ∈ (finset.range precincts_per_district)), ∃ votes : ℕ, 
  majority_precinct votes

-- Proof goal
theorem minimum_voters_for_tall_giraffe : ∃ (votes_for_tall : ℕ), votes_for_tall = 30 :=
by {
  -- proof of the theorem will go here
  sorry 
}

end minimum_voters_for_tall_giraffe_l521_521283


namespace weekly_earnings_l521_521672

def shop_opening_hours_per_day := 720 -- in minutes
def women_tshirt_selling_interval := 30 -- in minutes
def price_per_women_tshirt := 18 -- in dollars
def men_tshirt_selling_interval := 40 -- in minutes
def price_per_men_tshirt := 15 -- in dollars
def days_per_week := 7 -- days

theorem weekly_earnings :
  let women_tshirts_sold_per_day := shop_opening_hours_per_day / women_tshirt_selling_interval in
  let daily_earnings_women_tshirts := women_tshirts_sold_per_day * price_per_women_tshirt in
  let men_tshirts_sold_per_day := shop_opening_hours_per_day / men_tshirt_selling_interval in
  let daily_earnings_men_tshirts := men_tshirts_sold_per_day * price_per_men_tshirt in
  let total_daily_earnings := daily_earnings_women_tshirts + daily_earnings_men_tshirts in
  let weekly_earnings := total_daily_earnings * days_per_week in
  weekly_earnings = 4914 := 
by {
  -- Proof steps go here
  sorry
}

end weekly_earnings_l521_521672


namespace weekly_earnings_l521_521673

def shop_opening_hours_per_day := 720 -- in minutes
def women_tshirt_selling_interval := 30 -- in minutes
def price_per_women_tshirt := 18 -- in dollars
def men_tshirt_selling_interval := 40 -- in minutes
def price_per_men_tshirt := 15 -- in dollars
def days_per_week := 7 -- days

theorem weekly_earnings :
  let women_tshirts_sold_per_day := shop_opening_hours_per_day / women_tshirt_selling_interval in
  let daily_earnings_women_tshirts := women_tshirts_sold_per_day * price_per_women_tshirt in
  let men_tshirts_sold_per_day := shop_opening_hours_per_day / men_tshirt_selling_interval in
  let daily_earnings_men_tshirts := men_tshirts_sold_per_day * price_per_men_tshirt in
  let total_daily_earnings := daily_earnings_women_tshirts + daily_earnings_men_tshirts in
  let weekly_earnings := total_daily_earnings * days_per_week in
  weekly_earnings = 4914 := 
by {
  -- Proof steps go here
  sorry
}

end weekly_earnings_l521_521673


namespace indices_exist_l521_521330

open Nat

theorem indices_exist (a : List ℕ) (h_distinct : a.Nodup) (h_pos : ∀ x ∈ a, 0 < x) (h_len : 3 ≤ a.length) :
  ∃ i j, i ≠ j ∧ ¬ ∃ k, k < a.length ∧ (a[i] + a[j]) ∣ (3 * a[k]) :=
by 
  sorry

end indices_exist_l521_521330


namespace decagon_adjacent_probability_l521_521520

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521520


namespace mom_eyes_l521_521893

theorem mom_eyes :
  ∃ (M : ℕ), 
    let dad_eyes := 3 in
    let kid_eyes := 3 * 4 in
    let total_eyes := 16 in
    M + dad_eyes + kid_eyes = total_eyes ↔ M = 1 :=
by
  sorry

end mom_eyes_l521_521893


namespace perpendicular_lines_l521_521213

noncomputable def f : ℝ → ℝ := λ x, x * Real.exp x

noncomputable def tangent_slope_at_P : ℝ := Deriv (λ x, x * Real.exp x) 1

theorem perpendicular_lines (a b : ℝ) (h_perpendicular : (a * 1 - b * Real.exp 1 - 3 = 0) ∧ (tangent_slope_at_P = 2 * Real.exp 1)) :
  a / b = - (1 / (2 * Real.exp 1)) :=
sorry

end perpendicular_lines_l521_521213


namespace not_possible_three_piles_l521_521385

def similar (x y : ℝ) : Prop :=
  y ≤ (real.sqrt 2) * x ∧ x ≤ (real.sqrt 2) * y

theorem not_possible_three_piles (N : ℝ) : 
  ¬ ∃ a b c : ℝ, a + b + c = N ∧ similar a b ∧ similar b c ∧ similar a c := 
by
  sorry

end not_possible_three_piles_l521_521385


namespace range_is_44_l521_521037

-- Conditions: Let n be an integer, and k be the number of even gaps.
def smallest_odd_multiple (n : ℤ) : ℤ := n
def largest_odd_multiple (n : ℤ) (k : ℤ) : ℤ := n + 2 * k

-- Given that the range of the set is 44.
def range_of_set (n : ℤ) (k : ℤ) : ℤ := largest_odd_multiple n k - smallest_odd_multiple n

-- The problem statement: proving the range of the set is 44.
theorem range_is_44 (n k : ℤ) (h : 2 * k = 44) : range_of_set n k = 44 :=
by
  calc
    range_of_set n k
        = largest_odd_multiple n k - smallest_odd_multiple n := rfl
    ... = (n + 2 * k) - n := rfl
    ... = 2 * k := by ring
    ... = 44 := by rw h

end range_is_44_l521_521037


namespace circle_center_radius_l521_521310

theorem circle_center_radius (x y : ℝ) :
  (x^2 + 4 * y - 16 = -y^2 + 24 * x + 16) →
  let c := 12,
      d := -2,
      s := 2 * Real.sqrt 41 in
  c + d + s = 10 + 2 * Real.sqrt 41 := 
by
  intros h
  let c := 12
  let d := -2
  let s := 2 * Real.sqrt 41
  have : (x - 12)^2 + (y + 2)^2 = 164 := sorry
  show c + d + s = 10 + 2 * Real.sqrt 41 from sorry

end circle_center_radius_l521_521310


namespace cyclist_pedestrian_meeting_distance_l521_521409

theorem cyclist_pedestrian_meeting_distance :
  let A := 0 -- Representing point A at 0 km
  let B := 3 -- Representing point B at 3 km since AB = 3 km
  let C := 7 -- Representing point C at 7 km since AC = AB + BC = 3 + 4 = 7 km
  exists (x : ℝ), x > 0 ∧ x < 3 ∧ x = 2.1 :=
sorry

end cyclist_pedestrian_meeting_distance_l521_521409


namespace possible_integer_side_lengths_l521_521802

theorem possible_integer_side_lengths : 
  {x : ℤ | x > 3 ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521802


namespace floor_floor_3x_eq_floor_x_plus_1_l521_521711

theorem floor_floor_3x_eq_floor_x_plus_1 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1⌋ = ⌊x + 1⌋) ↔ (2 / 3 ≤ x ∧ x < 4 / 3) :=
by
  sorry

end floor_floor_3x_eq_floor_x_plus_1_l521_521711


namespace simplify_f_value_of_f_third_quadrant_l521_521187

variable (α : Real)

def f (α : Real) : Real :=
  (sin (π - α) * cos (2 * π - α) * cos (-α + (3 * π) / 2)) / (cos ((π / 2) - α) * sin (-π - α))

theorem simplify_f :
  f α = -cos α := 
  sorry

def inThirdQuadrant (α : Real) : Prop := 
  (3 * π / 2) < α ∧ α < 2 * π

theorem value_of_f_third_quadrant (h1 : inThirdQuadrant α) (h2 : cos (α - (3 * π / 2)) = 1 / 5) :
  f α = 2 * sqrt 6 / 5 := 
  sorry

end simplify_f_value_of_f_third_quadrant_l521_521187


namespace sum_inverse_squares_inequality_l521_521033

theorem sum_inverse_squares_inequality (n : ℕ) (h : n > 0) :
  (∑ i in Finset.range (n + 1), (1 : ℚ) / (i + 2) ^ 2) > (1 / 2) - (1 / (n + 2)) := 
sorry

end sum_inverse_squares_inequality_l521_521033


namespace Kim_min_score_for_target_l521_521899

noncomputable def Kim_exam_scores : List ℚ := [86, 82, 89]

theorem Kim_min_score_for_target :
  ∃ x : ℚ, ↑((Kim_exam_scores.sum + x) / (Kim_exam_scores.length + 1) ≥ (Kim_exam_scores.sum / Kim_exam_scores.length) + 2)
  ∧ x = 94 := sorry

end Kim_min_score_for_target_l521_521899


namespace daily_expenses_increase_l521_521990

theorem daily_expenses_increase 
  (init_students : ℕ) (new_students : ℕ) (diminish_amount : ℝ) (orig_expenditure : ℝ)
  (orig_expenditure_eq : init_students = 35)
  (new_students_eq : new_students = 42)
  (diminish_amount_eq : diminish_amount = 1)
  (orig_expenditure_val : orig_expenditure = 400)
  (orig_average_expenditure : ℝ) (increase_expenditure : ℝ)
  (orig_avg_calc : orig_average_expenditure = orig_expenditure / init_students)
  (new_total_expenditure : ℝ)
  (new_expenditure_eq : new_total_expenditure = orig_expenditure + increase_expenditure) :
  (42 * (orig_average_expenditure - diminish_amount) = new_total_expenditure) → increase_expenditure = 38 := 
by 
  sorry

end daily_expenses_increase_l521_521990


namespace minimized_sum_of_products_l521_521458

noncomputable def min_sum_ab_bc_cd_da : ℝ :=
  let a := {1, 3, 5, 7}
  let b := {1, 3, 5, 7}
  let c := {1, 3, 5, 7}
  let d := {1, 3, 5, 7} 
  -- additional constraints a, b, c, d are distinct
  if h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d then
    let sum_ab_bc_cd_da := λ (a b c d : ℝ), a * b + b * c + c * d + d * a
    let min_value := 48 -- minimum calculated from the solution steps
  min_value

theorem minimized_sum_of_products : min_sum_ab_bc_cd_da = 48 := 
by sorry

end minimized_sum_of_products_l521_521458


namespace hexagon_area_equality_l521_521627

-- Define the hexagon and its properties
structure Hexagon (α : Type*) [AddGroup α] [Module ℝ α] :=
  (A1 A2 A3 A4 A5 A6 : α)
  (is_convex : Convex ℝ ({A1, A2, A3, A4, A5, A6} : Set α))
  (parallel_opposite : (A2 - A1) ∥ (A5 - A4) ∧ (A3 - A2) ∥ (A6 - A5) ∧ (A4 - A3) ∥ (A1 - A6))

-- Define the property to prove
theorem hexagon_area_equality {α : Type*} [AddGroup α] [Module ℝ α] 
  (h : Hexagon α) : 
  let A1 := h.A1, A3 := h.A3, A5 := h.A5, A4 := h.A4, A6 := h.A6 in
  area (A1, A3, A5) = area (A1, A4, A6) :=
sorry

end hexagon_area_equality_l521_521627


namespace boys_to_girls_ratio_l521_521986

theorem boys_to_girls_ratio (total_students girls : ℕ) (h_total_students : total_students = 27) (h_girls : girls = 15) :
  let boys := total_students - girls in
  boys.to_float / girls.to_float = 4 / 5 :=
by
  sorry

end boys_to_girls_ratio_l521_521986


namespace not_possible_to_create_3_piles_l521_521371

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end not_possible_to_create_3_piles_l521_521371


namespace find_points_in_groups_l521_521287

noncomputable def number_of_lines (n : ℕ) : ℕ := n * (n - 1) / 2

theorem find_points_in_groups (n1 n2 : ℕ) :
  number_of_lines n1 = number_of_lines n2 + 27 ∧
  number_of_lines n1 + number_of_lines n2 = 171 →
  ({n1, n2} = {11, 8}) :=
by
  intros h
  sorry

end find_points_in_groups_l521_521287


namespace smallest_k_for_sequence_l521_521195

theorem smallest_k_for_sequence {a : ℕ → ℕ} (k : ℕ) (h : ∀ n ≥ 2, a (n + 1) = k * (a n) / (a (n - 1))) 
  (h0 : a 1 = 1) (h1 : a 2018 = 2020) (h2 : ∀ n, a n ∈ ℕ) : k = 2020 :=
sorry

end smallest_k_for_sequence_l521_521195


namespace math_problem_solution_part1_math_problem_solution_part2_l521_521903

noncomputable def part1 (A B : set ℝ) (a : ℝ) : Prop :=
  A = {x | x^2 - 3 * x + 2 = 0} ∧ 
  B = {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0} → 
  (A ∩ B = {2} → (a = -1 ∨ a = -3))

noncomputable def part2 (A B U : set ℝ) (a : ℝ) : Prop :=
  U = set.univ ∧ 
  A = {x | x^2 - 3 * x + 2 = 0} ∧ 
  B = {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0} →
  (A ∩ (U \ B) = A → 
    (a < -3 ∨ (-3 < a ∧ a < -1 - real.sqrt 3) ∨ 
    (-1 - real.sqrt 3 < a ∧ a < -1) ∨ 
    (-1 < a ∧ a < -1 + real.sqrt 3) ∨ 
    a > -1 + real.sqrt 3))

theorem math_problem_solution_part1 (A B : set ℝ) (a : ℝ) : part1 A B a :=
  sorry

theorem math_problem_solution_part2 (A B U : set ℝ) (a : ℝ) : part2 A B U a :=
  sorry

end math_problem_solution_part1_math_problem_solution_part2_l521_521903


namespace treasure_probability_l521_521118

variable {Island : Type}

-- Define the probabilities.
def prob_treasure : ℚ := 1 / 3
def prob_trap : ℚ := 1 / 6
def prob_neither : ℚ := 1 / 2

-- Define the number of islands.
def num_islands : ℕ := 5

-- Define the probability of encountering exactly 4 islands with treasure and one with neither traps nor treasures.
theorem treasure_probability :
  (num_islands.choose 4) * (prob_ttreasure^4) * (prob_neither^1) = (5 : ℚ) * (1 / 81) * (1 / 2) :=
  by
  sorry

end treasure_probability_l521_521118


namespace meeting_point_distance_l521_521405

/-
 Points \( A, B, C \) are situated sequentially, with \( AB = 3 \) km and \( BC = 4 \) km. 
 A cyclist departed from point \( A \) heading towards point \( C \). 
 A pedestrian departed from point \( B \) heading towards point \( A \). 
 Both arrived at points \( A \) and \( C \) simultaneously. 
 Find the distance from point \( A \) at which they met.
-/

def distance_A_B : ℝ := 3
def distance_B_C : ℝ := 4
def distance_A_C : ℝ := distance_A_B + distance_B_C

theorem meeting_point_distance (V_C V_P : ℝ) (h_time_eq : 7 / V_C = 3 / V_P) : 
  ∃ x : ℝ, x = 2.1 :=
begin
  -- Definitions of the known distances
  let AB := distance_A_B,
  let BC := distance_B_C,
  let AC := distance_A_C,

  -- Set up the ratio of their speeds
  let speed_ratio := 7 / 3,

  -- Define distances covered by cyclist and pedestrian
  let x := 2.1, -- the distance we need to prove
  
  -- Check the ratio relationship
  -- Combine the facts to goal, direct straightforward calculation
  
  use x,
  exact rfl,
end

end meeting_point_distance_l521_521405


namespace packs_used_after_6_weeks_l521_521685

-- Define the conditions as constants or definitions.
def pages_per_class_per_day : ℕ := 2
def num_classes : ℕ := 5
def days_per_week : ℕ := 5
def weeks : ℕ := 6
def pages_per_pack : ℕ := 100

-- The total number of packs of notebook paper Chip will use after 6 weeks
theorem packs_used_after_6_weeks : (pages_per_class_per_day * num_classes * days_per_week * weeks) / pages_per_pack = 3 := 
by
  -- skip the proof
  sorry

end packs_used_after_6_weeks_l521_521685


namespace jeremy_oranges_picked_on_Wednesday_l521_521393

def oranges_picked_on_Monday := 100
def oranges_picked_on_Tuesday := 3 * oranges_picked_on_Monday
def total_oranges_after_Wednesday := 470

theorem jeremy_oranges_picked_on_Wednesday :
  let o_M := oranges_picked_on_Monday
  let o_T := oranges_picked_on_Tuesday
  let o_total := total_oranges_after_Wednesday
  ∃ o_W, o_W = o_total - (o_M + o_T) ∧ o_W = 70 :=
by
  let o_M := oranges_picked_on_Monday
  let o_T := oranges_picked_on_Tuesday
  let o_total := total_oranges_after_Wednesday
  use o_total - (o_M + o_T)
  split
  case left => rfl
  case right => sorry

end jeremy_oranges_picked_on_Wednesday_l521_521393


namespace value_of_expr_l521_521847

theorem value_of_expr (a b c d : ℝ) (h1 : a = 3 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) : 
  a * c / (b * d) = 15 := 
by
  sorry

end value_of_expr_l521_521847


namespace number_of_sides_possibilities_l521_521812

theorem number_of_sides_possibilities (a b : ℕ) (h₁ : a = 8) (h₂ : b = 5) : 
  { x : ℕ // 3 < x ∧ x < 13 }.to_list.length = 9 :=
by
  sorry

end number_of_sides_possibilities_l521_521812


namespace probability_adjacent_vertices_in_decagon_l521_521574

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l521_521574


namespace decagon_adjacent_probability_l521_521578

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521578


namespace expected_value_dodecahedral_die_l521_521084

noncomputable def dodecahedral_expected_value : ℝ :=
  let outcomes := (list.range (12 + 1)).tail -- List of numbers from 1 to 12
  let n_faces := 12
  let probability := 1 / (n_faces : ℝ)
  let expected_value := probability * outcomes.sum / n_faces
  expected_value

theorem expected_value_dodecahedral_die :
  dodecahedral_expected_value = 6.5 :=
by {
  unfold dodecahedral_expected_value,
  simp,
  sorry,
}

end expected_value_dodecahedral_die_l521_521084


namespace additional_ounces_per_day_l521_521026

theorem additional_ounces_per_day :
  (Tim_drinks_2_bottles_each_1.5_quarts_every_day : Prop)
  (Tim_drinks_812_ounces_of_water_every_week : Prop)
  (There_are_32_ounces_in_1_quart : Prop) :
  ∃ (additional_ounces_per_day : ℕ), additional_ounces_per_day = 20 := by
  sorry

end additional_ounces_per_day_l521_521026


namespace probability_of_five_out_of_seven_days_l521_521947

noncomputable def probability_exactly_5_out_of_7 (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * (p^(n-k))

theorem probability_of_five_out_of_seven_days :
  probability_exactly_5_out_of_7 7 5 (1/2) = 21 / 128 := by
  sorry

end probability_of_five_out_of_seven_days_l521_521947


namespace rook_placement_l521_521109

theorem rook_placement : 
  let n := 8
  let k := 6
  let binom := Nat.choose
  binom 8 6 * binom 8 6 * Nat.factorial 6 = 564480 := by
    sorry

end rook_placement_l521_521109


namespace volume_of_cube_is_correct_surface_area_of_cube_is_correct_l521_521614

-- Define the conditions: total edge length of the cube frame
def total_edge_length : ℕ := 60
def number_of_edges : ℕ := 12

-- Define the edge length of the cube
def edge_length (total_edge_length number_of_edges : ℕ) : ℕ := total_edge_length / number_of_edges

-- Define the volume of the cube
def cube_volume (a : ℕ) : ℕ := a ^ 3

-- Define the surface area of the cube
def cube_surface_area (a : ℕ) : ℕ := 6 * (a ^ 2)

-- Volume Proof Statement
theorem volume_of_cube_is_correct : cube_volume (edge_length total_edge_length number_of_edges) = 125 :=
by
  sorry

-- Surface Area Proof Statement
theorem surface_area_of_cube_is_correct : cube_surface_area (edge_length total_edge_length number_of_edges) = 150 :=
by
  sorry

end volume_of_cube_is_correct_surface_area_of_cube_is_correct_l521_521614


namespace not_possible_three_piles_l521_521387

def similar (x y : ℝ) : Prop :=
  y ≤ (real.sqrt 2) * x ∧ x ≤ (real.sqrt 2) * y

theorem not_possible_three_piles (N : ℝ) : 
  ¬ ∃ a b c : ℝ, a + b + c = N ∧ similar a b ∧ similar b c ∧ similar a c := 
by
  sorry

end not_possible_three_piles_l521_521387


namespace cost_of_chocolate_bars_l521_521948

open Real

def chocolate_bar_brand : Type :=
| BrandA
| BrandB
| BrandC

def cost_per_bar (b : chocolate_bar_brand) : ℝ :=
  match b with
  | chocolate_bar_brand.BrandA => 1.50
  | chocolate_bar_brand.BrandB => 2.10
  | chocolate_bar_brand.BrandC => 3.00

def smores_per_bar (b : chocolate_bar_brand) : ℕ :=
  match b with
  | chocolate_bar_brand.BrandA => 3
  | chocolate_bar_brand.BrandB => 4
  | chocolate_bar_brand.BrandC => 6

def bulk_discount_threshold : ℕ := 10

def discount_percent : ℝ := 0.15

def num_scouts : ℕ := 15

def smores_per_scout : ℕ := 2

def total_smores_needed : ℕ := num_scouts * smores_per_scout

def num_bars_needed (b : chocolate_bar_brand) : ℕ :=
  let spb := smores_per_bar b
  (total_smores_needed + spb - 1) / spb -- round up

def total_cost_before_discount (b : chocolate_bar_brand) : ℝ :=
  cost_per_bar b * (num_bars_needed b).toReal

def total_cost_after_discount (b : chocolate_bar_brand) : ℝ :=
  if num_bars_needed b >= bulk_discount_threshold then
    total_cost_before_discount b * (1 - discount_percent)
  else
    total_cost_before_discount b

def min_cost : ℝ :=
  min (total_cost_after_discount chocolate_bar_brand.BrandA)
     (min (total_cost_after_discount chocolate_bar_brand.BrandB)
          (total_cost_after_discount chocolate_bar_brand.BrandC))

theorem cost_of_chocolate_bars : min_cost = 12.75 :=
  sorry

end cost_of_chocolate_bars_l521_521948


namespace numbered_cells_within_distance_l521_521501

open Real

-- Definitions based on conditions
def board_size : ℕ := 2005
def cell_length : ℝ := 1
def max_distance : ℝ := 150
def number_difference : ℕ := 23
def numbering_distance : ℝ := 10

-- Defining the conditions
def valid_position (x y : ℕ) : Prop :=
  x < board_size ∧ y < board_size

def distance (x1 y1 x2 y2 : ℕ) : ℝ :=
  sqrt ((x2 - x1 : ℝ) ^ 2 + (y2 - y1 : ℝ) ^ 2)

def numbered_cells (cells : board_size × board_size → ℕ) : Prop :=
  ∀ (x y : ℕ), valid_position x y →
    (∃ (nx ny : ℕ), valid_position nx ny ∧ distance x y nx ny < numbering_distance)

-- The theorem to prove
theorem numbered_cells_within_distance (cells : board_size × board_size → ℕ)
  (h_num : numbered_cells cells) :
  ∃ (x1 y1 x2 y2 : ℕ), valid_position x1 y1 ∧ valid_position x2 y2 ∧
  distance x1 y1 x2 y2 < max_distance ∧ abs (cells (x1, y1) - cells (x2, y2)) > number_difference :=
sorry

end numbered_cells_within_distance_l521_521501


namespace probability_adjacent_vertices_in_decagon_l521_521576

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l521_521576


namespace half_angle_third_quadrant_l521_521753

theorem half_angle_third_quadrant (α : ℝ) (k : ℤ)
  (h1 : 90 + k * 360 < α ∧ α < 180 + k * 360)
  (h2 : |real.cos (α / 2)| = -real.cos (α / 2)) :
  45 + k * 180 < α / 2 ∧ α / 2 < 90 + k * 180 ∧ real.cos (α / 2) < 0 :=
by sorry

end half_angle_third_quadrant_l521_521753


namespace triangle_side_lengths_l521_521841

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end triangle_side_lengths_l521_521841


namespace casey_hoodies_l521_521173

theorem casey_hoodies (fiona_owns : ℕ) (total_hoodies : ℕ) (fiona_owns_three : fiona_owns = 3) (total_hoodies_eight : total_hoodies = 8) : ∃ casey_owns : ℕ, casey_owns = 5 := 
by 
  use total_hoodies - fiona_owns 
  rw [total_hoodies_eight, fiona_owns_three] 
  exact rfl

end casey_hoodies_l521_521173


namespace num_possible_triangle_sides_l521_521808

theorem num_possible_triangle_sides (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (a_eq_8 : a = 8) (b_eq_5 : b = 5) : 
  let x := {n : ℕ | 3 < n ∧ n < 13} in
  set.card x = 9 :=
by
  sorry

end num_possible_triangle_sides_l521_521808


namespace emily_walks_270_meters_farther_l521_521465

-- Define constants for initial distances from home to school
def troy_home_to_school : ℕ := 75
def emily_home_to_school : ℕ := 98

-- Define the detours they take each day (in meters)
def troy_detours : List ℕ := [15, 20, 10, 0, 5]
def emily_detours : List ℕ := [10, 25, 10, 15, 10]

-- Function to calculate the total distance for a given distance and list of detours
def total_round_trip_distance (home_to_school : ℕ) (detours : List ℕ) : ℕ :=
  detours.map (λ detour, (home_to_school + detour) * 2).sum

-- Calculate the total distances for both Troy and Emily
def troy_total_distance : ℕ := total_round_trip_distance troy_home_to_school troy_detours
def emily_total_distance : ℕ := total_round_trip_distance emily_home_to_school emily_detours

-- Theorem to prove Emily walks 270 meters farther than Troy
theorem emily_walks_270_meters_farther : emily_total_distance - troy_total_distance = 270 := by
  sorry

end emily_walks_270_meters_farther_l521_521465


namespace expected_value_dodecahedral_die_l521_521085

noncomputable def dodecahedral_expected_value : ℝ :=
  let outcomes := (list.range (12 + 1)).tail -- List of numbers from 1 to 12
  let n_faces := 12
  let probability := 1 / (n_faces : ℝ)
  let expected_value := probability * outcomes.sum / n_faces
  expected_value

theorem expected_value_dodecahedral_die :
  dodecahedral_expected_value = 6.5 :=
by {
  unfold dodecahedral_expected_value,
  simp,
  sorry,
}

end expected_value_dodecahedral_die_l521_521085


namespace TV_cost_l521_521389

theorem TV_cost (initial_savings : ℕ) (fraction_on_furniture : ℚ) (fraction_on_tv : ℚ)
  (h_fraction : fraction_on_tv = 1 - fraction_on_furniture)
  (h_savings : initial_savings = 1200)
  (h_fraction_furniture : fraction_on_furniture = 3 / 4): 
  (initial_savings * fraction_on_tv).toRat = 300 := 
by {
  sorry
}

end TV_cost_l521_521389


namespace impossible_to_create_3_piles_l521_521379

def similar_sizes (x y : ℝ) : Prop := x ≤ sqrt 2 * y

theorem impossible_to_create_3_piles (x y : ℝ) (h₁ : similar_sizes x y) :
  ¬ ∃ (x₁ x₂ x₃ : ℝ), x₁ + x₂ + x₃ = x + y ∧ similar_sizes x₁ x₂ ∧ similar_sizes x₂ x₃ ∧ similar_sizes x₁ x₃ :=
by
  sorry

end impossible_to_create_3_piles_l521_521379


namespace sqrt_four_l521_521983

theorem sqrt_four : {x : ℝ | x ^ 2 = 4} = {-2, 2} := by
  sorry

end sqrt_four_l521_521983


namespace not_possible_to_create_3_piles_l521_521357

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end not_possible_to_create_3_piles_l521_521357


namespace problem_l521_521849

variable (x y z : ℚ)

-- Conditions as definitions
def cond1 : Prop := x / y = 3
def cond2 : Prop := y / z = 5 / 2

-- Theorem statement with the final proof goal
theorem problem (h1 : cond1 x y) (h2 : cond2 y z) : z / x = 2 / 15 := 
by 
  sorry

end problem_l521_521849


namespace probability_adjacent_vertices_in_decagon_l521_521568

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l521_521568


namespace cosine_of_unit_vectors_l521_521317

variables {a b : ℝ^3}
hypothesis (unit_a : ∥a∥ = 1)
hypothesis (unit_b : ∥b∥ = 1)
hypothesis (proj_condition : (a + b) ⬝ b = 2 / 3)

theorem cosine_of_unit_vectors :
  real.cos (real.angle a b) = -1 / 3 :=
sorry

end cosine_of_unit_vectors_l521_521317


namespace research_question_correct_survey_method_correct_l521_521028

-- Define the conditions.
def total_students : Nat := 400
def sampled_students : Nat := 80

-- Define the research question.
def research_question : String := "To understand the vision conditions of 400 eighth-grade students in a certain school."

-- Define the survey method.
def survey_method : String := "A sampling survey method was used."

-- Prove the research_question matches the expected question given the conditions.
theorem research_question_correct :
  research_question = "To understand the vision conditions of 400 eighth-grade students in a certain school" := by
  sorry

-- Prove the survey method used matches the expected method given the conditions.
theorem survey_method_correct :
  survey_method = "A sampling survey method was used" := by
  sorry

end research_question_correct_survey_method_correct_l521_521028


namespace math_proof_problem_l521_521261

noncomputable def sum_of_possible_x (x : ℝ) :=
  if (∀ (α β γ : ℝ), α + β + γ = 180 ∧ (β - α = γ - β) ∧ (
        7^2 = 3^2 + x^2 - 2 * 3 * x * (Real.cos (Real.pi / 3)) ∨
        x^2 = 7^2 + 3^2 - 2 * 7 * 3 * (Real.cos (Real.pi / 3)))) 
  then 8 + Real.sqrt 37
  else 0

theorem math_proof_problem : sum_of_possible_x 8 = 45 := by
  sorry

end math_proof_problem_l521_521261


namespace rearrange_viewers_l521_521502

theorem rearrange_viewers (n : ℕ) (f : ℕ → ℕ) (h_bij : bijective f) (h_not_own_seat : ∀ i, i ≤ n → f i ≠ i)
  (swap : ℕ → ℕ → ℕ → ℕ) (h_swap : ∀ i j, i < j → j < n → (f i ≠ i ∧ f (i+1) ≠ i+1) → (swap f i j = swap f j i)) :
  ∃ g : ℕ → ℕ, bijective g ∧ (∀ i, i ≤ n → g i = i) :=
by
  sorry

end rearrange_viewers_l521_521502


namespace canvas_bag_lower_carbon_solution_l521_521927

theorem canvas_bag_lower_carbon_solution :
  let canvas_release_oz := 9600
  let plastic_per_trip_oz := 32
  canvas_release_oz / plastic_per_trip_oz = 300 :=
by
  sorry

end canvas_bag_lower_carbon_solution_l521_521927


namespace non_integer_interior_angle_count_l521_521324

theorem non_integer_interior_angle_count :
  ∃! (n : ℕ), 3 ≤ n ∧ n < 10 ∧ ¬(∃ k : ℕ, 180 * (n - 2) = n * k) :=
by sorry

end non_integer_interior_angle_count_l521_521324


namespace hyperbola_center_l521_521712

theorem hyperbola_center :
  (∃ h k : ℝ,
    (∀ x y : ℝ, ((4 * x - 8) / 9)^2 - ((5 * y + 5) / 7)^2 = 1 ↔ (x - h)^2 / (81 / 16) - (y - k)^2 / (49 / 25) = 1) ∧
    (h = 2) ∧ (k = -1)) :=
sorry

end hyperbola_center_l521_521712


namespace impossible_to_create_3_piles_l521_521373

def similar_sizes (x y : ℝ) : Prop := x ≤ sqrt 2 * y

theorem impossible_to_create_3_piles (x y : ℝ) (h₁ : similar_sizes x y) :
  ¬ ∃ (x₁ x₂ x₃ : ℝ), x₁ + x₂ + x₃ = x + y ∧ similar_sizes x₁ x₂ ∧ similar_sizes x₂ x₃ ∧ similar_sizes x₁ x₃ :=
by
  sorry

end impossible_to_create_3_piles_l521_521373


namespace square_distance_centroid_l521_521924

structure Point4D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (u : ℝ)

def distance (P₁ P₂ : Point4D) : ℝ :=
  real.sqrt ((P₁.x - P₂.x)^2 + (P₁.y - P₂.y)^2 + (P₁.z - P₂.z)^2 + (P₁.u - P₂.u)^2)

def centroid (P₁ P₂ P₃ P₄ : Point4D) : Point4D :=
  { x := (P₁.x + P₂.x + P₃.x + P₄.x) / 4,
    y := (P₁.y + P₂.y + P₃.y + P₄.y) / 4,
    z := (P₁.z + P₂.z + P₃.z + P₄.z) / 4,
    u := (P₁.u + P₂.u + P₃.u + P₄.u) / 4 }

def squared_distance (P₁ P₂ : Point4D) : ℝ :=
  (distance P₁ P₂)^2

theorem square_distance_centroid (P₁ P₂ P₃ P₄ P₅ : Point4D) :
  squared_distance P₅ (centroid P₁ P₂ P₃ P₄) =
  (1/4) * (squared_distance P₅ P₁ + squared_distance P₅ P₂ + squared_distance P₅ P₃ + squared_distance P₅ P₄) -
  (1/16) * (squared_distance P₁ P₂ + squared_distance P₁ P₃ + squared_distance P₁ P₄ + 
            squared_distance P₂ P₃ + squared_distance P₂ P₄ + squared_distance P₃ P₄) :=
sorry

end square_distance_centroid_l521_521924


namespace sum_binom_square_l521_521945

open BigOperators
open Nat

theorem sum_binom_square (n : ℕ) :
  ∑ k in Finset.range (n + 1), (-1) ^ k * (Nat.choose n k) ^ 2 = 
  if even n then (-1) ^ (n / 2) * Nat.choose n (n / 2) else 0 := 
  sorry

end sum_binom_square_l521_521945


namespace sequence_integer_terms_l521_521978

theorem sequence_integer_terms :
  let seq := (fun (n : ℕ) => 9720 / (4 ^ n)) in
  ∃ n : ℕ, ¬int ((seq (n + 1))) ∧ seq 0 = 9720 := sorry

end sequence_integer_terms_l521_521978


namespace not_possible_to_create_3_piles_l521_521365

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end not_possible_to_create_3_piles_l521_521365


namespace find_rosy_age_l521_521701

-- Definitions and conditions
def rosy_current_age (R : ℕ) : Prop :=
  ∃ D : ℕ,
    (D = R + 18) ∧ -- David is 18 years older than Rosy
    (D + 6 = 2 * (R + 6)) -- In 6 years, David will be twice as old as Rosy

-- Proof statement: Rosy's current age is 12
theorem find_rosy_age : rosy_current_age 12 :=
  sorry

end find_rosy_age_l521_521701


namespace probability_of_adjacent_vertices_in_decagon_l521_521587

def decagon_vertices : Finset ℕ := Finset.range 10

def adjacent (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v2 = (v1 + 1) % 10)

noncomputable def probability_adjacent_vertices : ℚ :=
  let total_ways := decagon_vertices.card * (decagon_vertices.card - 1) in
  let favorable_ways := 2 * decagon_vertices.card in
  favorable_ways / total_ways

theorem probability_of_adjacent_vertices_in_decagon :
  probability_adjacent_vertices = 2 / 9 := by
  sorry

end probability_of_adjacent_vertices_in_decagon_l521_521587


namespace find_n_divisible_by_3_l521_521003

def harmonic_sum (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum (λ k, 1 / (k + 1 : ℚ))

def is_rel_prime (a b : ℕ) : Prop :=
  nat.gcd a b = 1

theorem find_n_divisible_by_3 (n : ℕ) (p_n q_n : ℕ) :
  p_n = (harmonic_sum n).num ∧ q_n = (harmonic_sum n).den ∧ is_rel_prime p_n q_n ∧ 3 ∣ p_n
  ↔ n = 2 ∨ n = 7 ∨ n = 22 :=
sorry

end find_n_divisible_by_3_l521_521003


namespace expected_value_of_fair_dodecahedral_die_l521_521048

theorem expected_value_of_fair_dodecahedral_die : 
  (1/12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_of_fair_dodecahedral_die_l521_521048


namespace canvas_bag_lower_carbon_solution_l521_521932

def canvas_bag_emission := 600 -- pounds of CO2
def plastic_bag_emission := 4 -- ounces of CO2 per bag
def bags_per_trip := 8 
def ounce_to_pound := 16 -- 16 ounces in a pound
def co2_trip := (plastic_bag_emission * bags_per_trip) / ounce_to_pound -- CO2 emission in pounds per trip

theorem canvas_bag_lower_carbon_solution : 
  co2_trip * 300 >= canvas_bag_emission :=
by
  unfold canvas_bag_emission plastic_bag_emission bags_per_trip ounce_to_pound co2_trip 
  sorry

end canvas_bag_lower_carbon_solution_l521_521932


namespace smallest_k_for_sequence_l521_521198

theorem smallest_k_for_sequence (a : ℕ → ℕ) (k : ℕ) (h₁ : a 1 = 1) (h₂ : a 2018 = 2020)
  (h₃ : ∀ n, n ≥ 2 → a (n+1) = k * (a n) / (a (n-1))) : k = 2020 :=
sorry

end smallest_k_for_sequence_l521_521198


namespace smallest_k_for_sequence_l521_521197

theorem smallest_k_for_sequence (a : ℕ → ℕ) (k : ℕ) (h₁ : a 1 = 1) (h₂ : a 2018 = 2020)
  (h₃ : ∀ n, n ≥ 2 → a (n+1) = k * (a n) / (a (n-1))) : k = 2020 :=
sorry

end smallest_k_for_sequence_l521_521197


namespace impossible_to_create_3_piles_l521_521332

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end impossible_to_create_3_piles_l521_521332


namespace gasoline_reduction_l521_521099

theorem gasoline_reduction (P Q : ℝ) (hP : P > 0) (hQ : Q > 0) :
  let new_price := 1.2 * P,
      new_total_cost := 1.08 * (P * Q),
      Q' := new_total_cost / new_price in
  (Q' = 0.9 * Q) → (Q - Q') / Q = 0.1 :=
by
  intro h1
  have h2 : Q' = 0.9 * Q
  from h1
  have h3 : (Q - 0.9 * Q) / Q = 0.1
  by
    simp
    ring
  show (Q - Q') / Q = 0.1 
  from h3

sorry  

end gasoline_reduction_l521_521099


namespace smallest_pos_integer_is_1458_l521_521487

def has_seven_odd_divisors (n : Nat) : Prop :=
  ∃ p₁ p₂ e₁ e₂, e₁ + 1 = 7 ∧ n = p₁ ^ e₁ * p₂ ^ e₂ ∧ p₁ = 3 ∧ e₁ = 6

def has_fourteen_even_divisors (n : Nat) : Prop :=
  ∃ e, (e + 1) * 7 = 14 ∧ n = 2^e * 3^6 ∧ e = 1

def smallest_with_property (n : Nat) : Prop :=
  has_seven_odd_divisors n ∧ has_fourteen_even_divisors n

theorem smallest_pos_integer_is_1458 : ∃ n, smallest_with_property n ∧ n = 1458 :=
by
  exists 1458
  constructor
  unfold smallest_with_property
  constructor
  { use 3, use (3:Nat), use (6:Nat), use (0:Nat)
    sorry } -- here we'd typically have the proof that 1458 has exactly 7 odd divisors
  { use 1
    sorry } -- here we'd typically have the proof that 1458 has exactly 14 even divisors
  rfl

end smallest_pos_integer_is_1458_l521_521487


namespace sum_seq_formula_l521_521006

-- Define the sequence
def seq (k : ℕ) : ℕ := (List.sum (List.map (λ (i : ℕ), 2^(i - 1)) (List.range k))) + 1

def sum_first_n_terms (n : ℕ) : ℕ :=
  List.sum (List.map seq (List.range n))

-- Theorem stating the sum of the first n terms is 2^(n+1) - n - 2
theorem sum_seq_formula (n : ℕ) : sum_first_n_terms n = 2^(n + 1) - n - 2 :=
  sorry

end sum_seq_formula_l521_521006


namespace quadratic_root_x_y_z_l521_521975

theorem quadratic_root_x_y_z
  (x y z d : ℝ)
  (h1 : y = x - d)
  (h2 : z = x - 2d)
  (h3 : x ∈ Set.Ici 0)
  (h4 : y ∈ Set.Ici 0)
  (h5 : z ∈ Set.Ici 0)
  (h6 : (x - 2 * d) * x^2 + (x - d) * x + x = 0) :
  x = -3 / 4 := sorry

end quadratic_root_x_y_z_l521_521975


namespace measure_of_angle_A_l521_521889

noncomputable def angle_A_eq_5pi_over_6 (A B C : ℝ) : Prop :=
  A = π * 5 / 6

theorem measure_of_angle_A :
  ∀ (A B C : triangle),
  (sin B ^ 2 + sin C ^ 2 = sin A ^ 2 - sqrt 3 * sin B * sin C) →
  angle_A_eq_5pi_over_6 A B C :=
by
  intros
  sorry

end measure_of_angle_A_l521_521889


namespace conjugate_in_third_quadrant_l521_521752

-- Define the imaginary unit
def i : ℂ := complex.I

-- Given condition: z * i = ((3 - i) / (1 + i))^2
def z (z : ℂ) : Prop :=
  z * i = ((3 - i) / (1 + i))^2

-- Prove that the point corresponding to the conjugate of z is in the third quadrant
theorem conjugate_in_third_quadrant (z : ℂ) (h : z * i = ((3 - i) / (1 + i))^2) :
  z.conj.im < 0 ∧ z.conj.re < 0 :=
sorry

end conjugate_in_third_quadrant_l521_521752


namespace final_value_of_t_is_120_l521_521761

theorem final_value_of_t_is_120 : 
  let initial_t := 1 in 
  let final_t := List.foldl (*) initial_t [2, 3, 4, 5] in
  final_t = 120 :=
by 
  let initial_t := 1
  let final_t := List.foldl (*) initial_t [2, 3, 4, 5]
  -- Do the arithmetic
  have h1 : initial_t * 2 = 2 := by norm_num
  have h2 : 2 * 3 = 6 := by norm_num
  have h3 : 6 * 4 = 24 := by norm_num
  have h4 : 24 * 5 = 120 := by norm_num
  rw [h1, h2, h3, h4]
  -- So final_t = 120 indeed
  exact rfl

end final_value_of_t_is_120_l521_521761


namespace train_speed_l521_521612

theorem train_speed (d t : ℝ) (h1 : d = 500) (h2 : t = 3) : d / t = 166.67 := by
  sorry

end train_speed_l521_521612


namespace solve_eq_l521_521433

theorem solve_eq : ∀ x : ℂ, (x^3 + 3*x^2 + 4*x + 6) / (x + 5) = x^2 + 10 ↔
  x = (-3 + Complex.I * Real.sqrt 79) / 2 ∨ x = (-3 - Complex.I * Real.sqrt 79) / 2 :=
by 
  intro x
  sorry

end solve_eq_l521_521433


namespace local_max_at_one_iff_no_a_such_that_exists_x_in_one_two_f_le_zero_l521_521763

section problem_1

def f (a x : ℝ) : ℝ :=
  (1 / 3) * x ^ 3 + (1 / 2) * (2 * a + 1) * x ^ 2 - 2 * (a + 1) * x

theorem local_max_at_one_iff :
  (∃ a : ℝ, ∀ x : ℝ, f a x = ((1 / 3) * x ^ 3 + (1 / 2) * (2 * a + 1) * x ^ 2 - 2 * (a + 1) * x) ∧
    ∃ (f has local_maximum at x), x = 1) ↔
  ∀ a : ℝ, a < - (3 / 2) :=
by
  sorry

end problem_1

section problem_2

def g (a x : ℝ) : ℝ :=
  (1 / 3) * x ^ 3 + (1 / 2) * (2 * a + 1) * x ^ 2 - 2 * (a + 1) * x

theorem no_a_such_that_exists_x_in_one_two_f_le_zero :
  (¬ ∃ a : ℝ, ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → g a x ≤ 0) :=
by
  sorry

end problem_2

end local_max_at_one_iff_no_a_such_that_exists_x_in_one_two_f_le_zero_l521_521763


namespace all_angles_def_acute_l521_521148

variable {α β : ℝ} 
variable {A B C D E F : Point}
variable {r : ℝ} -- radius of the inscribed circle
variable {triangle_ABC : Triangle} -- denote triangle ABC
variable [Isosceles : triangle_ABC.isosceles B C] -- ABC is isosceles with AB = AC

-- assume circle is inscribed in triangle ABC
axiom in_circle_ABC : ∃ (circ : Circle), circ.inscribed_in triangle_ABC ∧ 
  circ.touches_at_side B C D ∧ circ.touches_at_side C A E ∧ circ.touches_at_side A B F

-- we need to show that the angles of triangle DEF are acute
theorem all_angles_def_acute 
  (h_in_circle : ∃ (circ : Circle), circ.inscribed_in triangle_ABC ∧ 
    circ.touches_at_side B C D ∧ circ.touches_at_side C A E ∧ circ.touches_at_side A B F)
  (h_isosceles : triangle_ABC.isosceles B C) :
  ∀ angle_DEF, angle_DEF ∈ angles_of (Triangle.mk D E F) → angle_DEF < 90 :=
by
  sorry

end all_angles_def_acute_l521_521148


namespace triangle_side_lengths_count_l521_521788

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end triangle_side_lengths_count_l521_521788


namespace largest_possible_value_of_c_l521_521911

theorem largest_possible_value_of_c (c : ℚ) : (3 * c + 4) * (c - 2) = 9 * c → c ≤ 4 :=
by
  intro h
  have : (3 * c + 4) * (c - 2) = 3 * c^2 - 6 * c + 4 * c - 8 := 
    calc 
    (3 * c + 4) * (c - 2) = (3 * c) * (c - 2) + 4 * (c - 2) : by ring
                         ... = (3 * c) * c - (3 * c) * 2 + 4 * c - 4 * 2 : by ring
                         ... = 3 * c^2 - 6 * c + 4 * c - 8 : by ring
  rw this at h
  have h2 : 3 * c^2 - 11 * c - 8 = 0 := by nlinarith
  sorry

end largest_possible_value_of_c_l521_521911


namespace smallest_N_exists_l521_521011

theorem smallest_N_exists (
  a b c d : ℕ := list.perm [1, 2, 3, 4, 5] [gcd a b, gcd a c, gcd a d, gcd b c, gcd b d, gcd c d]
  (h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_N: N > 5) : 
  N = 14 :=
by sorry

end smallest_N_exists_l521_521011


namespace tangent_line_g_properties_l521_521765

noncomputable section

open Real

-- Define the initial function f
def f (x : ℝ) (a : ℝ) : ℝ := (1 / 4) * x^4 - (1 / 2) * a * x^2

-- Question 1: Prove the equation for the tangent line
theorem tangent_line (x : ℝ) (a : ℝ) : 
  a = 1 → 
  (∃ m b, m = f' 2 1 ∧ b = f 2 1 ∧ (∀ x, 6 * x - m * x - b = 0)) :=
begin
  sorry
end

-- Define the function g
def g (x : ℝ) (a : ℝ) : ℝ := (x^2 - 2*x + 2 - a) * exp x - exp 1 * f x a

-- Question 2: Prove the properties of g based on the value of a
theorem g_properties (a : ℝ) :
  (a ≤ 0 → (∀ x y : ℝ, x ≤ y → g x a ≤ g y a)) ∧
  (a > 0 → (∃ x₁ x₂ : ℝ, x₁ = -sqrt a ∧ x₂ = sqrt a ∧ 
  (∀ x, x < x₁ ∨ x > x₂ → g x a < g x₁ a ∧ g x a > g x₂ a))) :=
begin
  sorry
end

end tangent_line_g_properties_l521_521765


namespace expected_value_is_three_l521_521256

/-- 
There are two balls numbered 1 and 2 in a dark box.
One ball is randomly drawn.
If ball number 2 is drawn, 2 points are earned, and the drawing stops.
If ball number 1 is drawn, 1 point is earned, and the ball is returned to the box for another draw.
The random variable X represents the total score when the drawing stops.
Prove that the expected value E(X) is 3.
-/
theorem expected_value_is_three :
  let ball_draw (n : ℕ) := 
    if n = 2 then 2
    else if n = 1 then 1 + ball_draw 2
    else 0,
  let X := fun () => ball_draw 1,
  E(X) = 3 :=
sorry

end expected_value_is_three_l521_521256


namespace sin_angle_ratio_eq_l521_521289

theorem sin_angle_ratio_eq (A B C D : Type)
  [triangle ABC]
  (angle_B : angle ABC = 45)
  (angle_C : angle ACB = 60)
  (ratio : divide_segment BC D = (3:1)) :
  (sin (angle BAD) / sin (angle CAD)) = (3 * sqrt 3 / 2) := 
sorry

end sin_angle_ratio_eq_l521_521289


namespace wheat_problem_l521_521108

noncomputable def total_weight (weights : List ℝ) : ℝ := weights.sum

noncomputable def expected_total_weight (num_bags : ℕ) (weight_per_bag : ℝ) : ℝ := num_bags * weight_per_bag

noncomputable def weight_difference (actual_weight expected_weight : ℝ) : ℝ := actual_weight - expected_weight

noncomputable def total_revenue (total_flour_kg price_per_kg : ℝ) : ℝ := total_flour_kg * price_per_kg

noncomputable def total_cost (num_bags cost_per_bag processing_cost : ℝ) : ℝ := (num_bags * cost_per_bag) + processing_cost

noncomputable def total_profit (revenue cost : ℝ) : ℝ := revenue - cost

theorem wheat_problem :
  let weights := [91, 92, 90, 89, 89, 91.2, 88.9, 91.8, 91.1, 88] in
  let num_bags := 10 in
  let weight_per_bag := 90 in
  let cost_per_bag := 100 in
  let processing_cost := 500 in
  let conversion_rate := 0.7 in
  let price_per_kg := 4 in
  total_weight weights - expected_total_weight num_bags weight_per_bag = 2 ∧
  total_profit 
    (total_revenue (total_weight weights * conversion_rate) price_per_kg) 
    (total_cost num_bags cost_per_bag processing_cost) = 1025.6 :=
by
  sorry

end wheat_problem_l521_521108


namespace decagon_adjacent_probability_l521_521524

noncomputable def probability_adjacent_vertices (total_vertices : ℕ) (adjacent_vertices : ℕ) : ℚ :=
adjacent_vertices / (total_vertices - 1)

theorem decagon_adjacent_probability :
  probability_adjacent_vertices 10 2 = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521524


namespace number_of_sides_possibilities_l521_521817

theorem number_of_sides_possibilities (a b : ℕ) (h₁ : a = 8) (h₂ : b = 5) : 
  { x : ℕ // 3 < x ∧ x < 13 }.to_list.length = 9 :=
by
  sorry

end number_of_sides_possibilities_l521_521817


namespace find_m_b_l521_521633

noncomputable def line_equation (x y : ℝ) :=
  (⟨-1, 4⟩ : ℝ × ℝ) • (⟨x, y⟩ - ⟨3, -5⟩ : ℝ × ℝ) = 0

theorem find_m_b : ∃ m b : ℝ, (∀ (x y : ℝ), line_equation x y → y = m * x + b) ∧ m = 1 / 4 ∧ b = -23 / 4 :=
by
  sorry

end find_m_b_l521_521633


namespace sequence_sum_problem_l521_521772

theorem sequence_sum_problem :
  let S : ℕ → ℕ := λ n, n^2 + 2 * n + 5
  ∃ a : ℕ → ℕ, (a 2 + a 3 + a 4 + a 4 + a 5) = 41 :=
by
  let S := λ n, n^2 + 2 * n + 5
  have h_S4 : S 4 = 29 := by norm_num
  have h_S1 : S 1 = 8 := by norm_num
  have h_S5 : S 5 = 40 := by norm_num
  have h_S3 : S 3 = 20 := by norm_num
  let a : ℕ → ℕ := λ n, if n = 1 then S 1
                       else if n = 2 then S 2 - S 1
                       else if n = 3 then S 3 - S 2
                       else if n = 4 then S 4 - S 3
                       else S n - S (n - 1)
  existsi a
  have h_sum1 : a 2 + a 3 + a 4 = S 4 - S 1 := by
    change a 2 + a 3 + a 4 = 29 - 8
    rw [h_S4, h_S1]
    norm_num
  have h_sum2 : a 4 + a 5 = S 5 - S 3 := by
    change a 4 + a 5 = 40 - 20
    rw [h_S5, h_S3]
    norm_num
  rw [h_sum1, h_sum2]
  norm_num
  exacts (dec_trivial : 21 + 20 = 41)
  sorry

end sequence_sum_problem_l521_521772


namespace minimum_voters_for_tall_giraffe_to_win_l521_521277

/-- Conditions -/
def total_voters := 135
def districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority (n : Nat) : Nat := Nat.ceil (n / 2)

/-- Proof Problem -/
theorem minimum_voters_for_tall_giraffe_to_win : 
  let precincts_needed_in_district := majority precincts_per_district,
      districts_needed_to_win := majority districts,
      precincts_needed_in_total := districts_needed_to_win * precincts_needed_in_district,
      majority_in_precinct := majority voters_per_precinct in
  precincts_needed_in_total * majority_in_precinct = 30 :=
by
  -- Proof steps (commented out)
  -- 1. Calculate the number of precincts in total: 5 * 9 = 45
  -- 2. Calculate how many districts the Tall giraffe needs to win: 3
  -- (majority districts)
  -- 3. Calculate how many precincts in a district the Tall giraffe needs to
  -- win: 5 (majority precincts_per_district)
  -- 4. Calculate the total number of precincts the Tall giraffe needs to win:
  -- 3 * 5 = 15
  -- 5. Calculate the majority of votes in a precinct: 2 
  -- (majority voters_per_precinct)
  -- 6. Calculate the minimum number of voters in 15 precincts: 15 * 2 = 30
  sorry

end minimum_voters_for_tall_giraffe_to_win_l521_521277


namespace rhombus_diagonal_l521_521964

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d1 = 14) (h2 : area = 126) (h3 : area = (d1 * d2) / 2) : d2 = 18 := 
by
  -- h1, h2, and h3 are the conditions
  sorry

end rhombus_diagonal_l521_521964


namespace triangle_side_count_l521_521834

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end triangle_side_count_l521_521834


namespace elena_lace_length_l521_521110

theorem elena_lace_length (area : ℝ) (pi_approx : ℝ) (extra_length : ℝ) 
  (h_area : area = 154) (h_pi : pi_approx = (22 / 7)) (h_extra : extra_length = 5) : 
  let r := Real.sqrt (area * 7 / 22) in
  let circ := 2 * pi_approx * r in
  circ + extra_length = 49 :=
by
  sorry

end elena_lace_length_l521_521110


namespace emily_necklaces_l521_521157

theorem emily_necklaces (total_beads : ℕ) (beads_per_necklace : ℕ) (total_beads = 308) (beads_per_necklace = 28) :
  total_beads / beads_per_necklace = 11 := 
by sorry

end emily_necklaces_l521_521157


namespace bob_can_guess_k_plus_one_correctly_l521_521610

theorem bob_can_guess_k_plus_one_correctly (k : ℕ) (seq : vector bool (2^k)) :
  ∃ s, s = k + 1 ∧ (∀ (revealed : ℕ), revealed < 2^k → ∀(val : bool), 
  ∃ indices : set ℕ, indices.card = k + 1 ∧ (∀ i ∈ indices, seq.get ⟨i, _⟩ = val) ) :=
by
  sorry

end bob_can_guess_k_plus_one_correctly_l521_521610


namespace smallest_constant_N_l521_521718

theorem smallest_constant_N (a : ℝ) (ha : a > 0) : 
  let b := a
  let c := a
  (a = b ∧ b = c) → (a^2 + b^2 + c^2) / (a + b + c) > (0 : ℝ) := 
by
  -- Assuming the proof steps are written here
  sorry

end smallest_constant_N_l521_521718


namespace angle_sum_l521_521204

noncomputable def alpha : Real := sorry
noncomputable def beta : Real := sorry

axiom h1 : 0 < alpha ∧ alpha < π / 2
axiom h2 : 0 < beta ∧ beta < π / 2
axiom h3 : Real.sin alpha = Real.sqrt 5 / 5
axiom h4 : Real.tan beta = 1 / 3

theorem angle_sum : alpha + beta = π / 4 := by
  sorry

end angle_sum_l521_521204


namespace side_length_of_square_with_area_one_l521_521644

theorem side_length_of_square_with_area_one :
  ∃ n : ℝ, n^2 = 1 ∧ n = 1 :=
begin
  sorry
end

end side_length_of_square_with_area_one_l521_521644


namespace expected_value_fair_dodecahedral_die_l521_521068

theorem expected_value_fair_dodecahedral_die : 
  let n := 12 in (1 / n) * (List.sum (List.range n).map (λ x => (x + 1))) = 6.5 :=
by
  let n := 12
  have h1 : (1 : ℚ) / n = 1 / 12 := by sorry
  have h2 : List.range n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] := by sorry
  have h3 : List.sum (List.range n).map (λ x => (x + 1)) = 78 := by sorry
  have h4 : (1 / 12) * 78 = 6.5 := by sorry
  exact h4

end expected_value_fair_dodecahedral_die_l521_521068


namespace boxes_of_bolts_purchased_l521_521721

theorem boxes_of_bolts_purchased 
  (bolts_per_box : ℕ) 
  (nuts_per_box : ℕ) 
  (num_nut_boxes : ℕ) 
  (leftover_bolts : ℕ) 
  (leftover_nuts : ℕ) 
  (total_bolts_nuts_used : ℕ)
  (B : ℕ) :
  bolts_per_box = 11 →
  nuts_per_box = 15 →
  num_nut_boxes = 3 →
  leftover_bolts = 3 →
  leftover_nuts = 6 →
  total_bolts_nuts_used = 113 →
  B = 7 :=
by
  intros
  sorry

end boxes_of_bolts_purchased_l521_521721


namespace factorize_expr1_factorize_expr2_l521_521160

open BigOperators

/-- Given m and n, prove that m^3 n - 9 m n can be factorized as mn(m + 3)(m - 3). -/
theorem factorize_expr1 (m n : ℤ) : m^3 * n - 9 * m * n = n * m * (m + 3) * (m - 3) :=
sorry

/-- Given a, prove that a^3 + a - 2a^2 can be factorized as a(a - 1)^2. -/
theorem factorize_expr2 (a : ℤ) : a^3 + a - 2 * a^2 = a * (a - 1)^2 :=
sorry

end factorize_expr1_factorize_expr2_l521_521160


namespace third_shift_pension_percentage_l521_521155

theorem third_shift_pension_percentage :
  (∀ x : ℕ, x = 60 → 0.2 * x = 12) ∧
  (∀ y : ℕ, y = 50 → 0.4 * y = 20) ∧
  (∀ z : ℕ, z = 40) ∧
  (∀ t : ℕ, t = 150) ∧
  (∀ p : ℕ, p = 36) →
  (4 / 40) * 100 = 10 :=
by sorry

end third_shift_pension_percentage_l521_521155


namespace line_ab_passes_fixed_point_quadrilateral_area_l521_521770

section parabola_problems

open Real

noncomputable def parabola_eq (x y : ℝ) : Prop := x^2 = 4 * y
noncomputable def point_on_line_y_neg1 (t : ℝ) : Prop := t ≠ 0
noncomputable def point_M (t : ℝ) : ℝ × ℝ := (t, -1)

theorem line_ab_passes_fixed_point :
  ∀ (t : ℝ), t ≠ 0 →
  ∀ (x1 y1 x2 y2 : ℝ), 
    parabola_eq x1 y1 → parabola_eq x2 y2 →
    tangent_at_point (t, -1) (x1, y1) →
    tangent_at_point (t, -1) (x2, y2) →
    on_line_ab (x1, y1) (x2, y2) (0, 1) := sorry

theorem quadrilateral_area :
  ∀ (t : ℝ), 
  point_on_line_y_neg1 t →
  ∀ (x1 y1 x2 y2 : ℝ), 
    parabola_eq x1 y1 → parabola_eq x2 y2 →
    tangent_at_point (t, -1) (x1, y1) →
    tangent_at_point (t, -1) (x2, y2) →
    circle_centered_at_tangent (0, 4) (x1, y1) (x2, y2) →
    ∃ S, S = 6 * sqrt 6 := sorry

end parabola_problems

end line_ab_passes_fixed_point_quadrilateral_area_l521_521770


namespace not_possible_to_create_3_piles_l521_521362

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end not_possible_to_create_3_piles_l521_521362


namespace probability_adjacent_vertices_of_decagon_l521_521510

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l521_521510


namespace length_decrease_by_33_percent_l521_521019

theorem length_decrease_by_33_percent (L W L_new : ℝ) 
  (h1 : L * W = L_new * 1.5 * W) : 
  L_new = (2 / 3) * L ∧ ((1 - (2 / 3)) * 100 = 33.33) := 
by
  sorry

end length_decrease_by_33_percent_l521_521019


namespace cyclist_pedestrian_meeting_distance_l521_521408

theorem cyclist_pedestrian_meeting_distance :
  let A := 0 -- Representing point A at 0 km
  let B := 3 -- Representing point B at 3 km since AB = 3 km
  let C := 7 -- Representing point C at 7 km since AC = AB + BC = 3 + 4 = 7 km
  exists (x : ℝ), x > 0 ∧ x < 3 ∧ x = 2.1 :=
sorry

end cyclist_pedestrian_meeting_distance_l521_521408


namespace canvas_bag_lower_carbon_solution_l521_521931

theorem canvas_bag_lower_carbon_solution :
  ∀ (canvas_bag_CO2_pounds : ℕ) (plastic_bag_CO2_ounces : ℕ) 
    (plastic_bags_per_trip : ℕ) (ounces_per_pound : ℕ),
    canvas_bag_CO2_pounds = 600 →
    plastic_bag_CO2_ounces = 4 →
    plastic_bags_per_trip = 8 →
    ounces_per_pound = 16 →
    let total_CO2_per_trip := plastic_bags_per_trip * plastic_bag_CO2_ounces / ounces_per_pound in
    (canvas_bag_CO2_pounds / total_CO2_per_trip) = 300 :=
by
  -- Assume the given conditions
  assume canvas_bag_CO2_pounds plastic_bag_CO2_ounces plastic_bags_per_trip ounces_per_pound,
  assume canvas_bag_CO2_pounds_eq plastic_bag_CO2_ounces_eq plastic_bags_per_trip_eq ounces_per_pound_eq,
  -- Introduce the total carbon dioxide per trip
  let total_CO2_per_trip := plastic_bags_per_trip * plastic_bag_CO2_ounces / ounces_per_pound,
  -- Verify that the number of trips is 300
  show ((canvas_bag_CO2_pounds / total_CO2_per_trip) = 300),
  sorry

end canvas_bag_lower_carbon_solution_l521_521931


namespace frequency_distribution_necessary_l521_521464

/-- Definition of the necessity to use Frequency Distribution to understand 
the proportion of first-year high school students in the city whose height 
falls within a certain range -/
def necessary_for_proportion (A B C D : Prop) : Prop := D

theorem frequency_distribution_necessary (A B C D : Prop) :
  necessary_for_proportion A B C D ↔ D :=
by
  sorry

end frequency_distribution_necessary_l521_521464


namespace staples_left_in_stapler_l521_521021

def initial_staples : ℕ := 50
def reports_stapled : ℕ := 3 * 12
def staples_per_report : ℕ := 1
def remaining_staples : ℕ := initial_staples - (reports_stapled * staples_per_report)

theorem staples_left_in_stapler : remaining_staples = 14 :=
by
  sorry

end staples_left_in_stapler_l521_521021


namespace quadratic_shared_root_product_zero_l521_521639

theorem quadratic_shared_root_product_zero
  (P : ℝ → ℝ)
  (hP_def : ∃ (b c : ℝ), ∀ x, P(x) = x^2 + b * x + c)
  (h_shared_root : ∃ r : ℝ, P(r) = 0 ∧ P(P(P(r))) = 0) :
  P(0) * P(1) = 0 :=
  sorry

end quadratic_shared_root_product_zero_l521_521639


namespace find_minimum_value_l521_521319

-- This definition captures the condition that a, b, c are positive real numbers
def pos_reals := { x : ℝ // 0 < x }

-- The main theorem statement
theorem find_minimum_value (a b c : pos_reals) :
  4 * (a.1 ^ 4) + 8 * (b.1 ^ 4) + 16 * (c.1 ^ 4) + 1 / (a.1 * b.1 * c.1) ≥ 10 :=
by
  -- This is where the proof will go
  sorry

end find_minimum_value_l521_521319


namespace triangle_side_lengths_l521_521842

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end triangle_side_lengths_l521_521842


namespace total_distance_traveled_l521_521493

theorem total_distance_traveled 
  (Vm Vr : ℝ) (T : ℝ) 
  (hVm : Vm = 10) -- Speed of man in still water
  (hVr : Vr = 1.2) -- Speed of the river
  (hT : T = 1) -- Total time for the round trip
  : let V_upstream := Vm - Vr, 
        V_downstream := Vm + Vr,
        D := (11.2 * 4.928 + 8.8 * 4.928) / 98.56,
        total_distance := 2 * D
    in total_distance = 9.856 :=
sorry

end total_distance_traveled_l521_521493


namespace not_possible_to_create_3_similar_piles_l521_521341

-- Formalize the definition of similar piles
def similar (a b : ℝ) : Prop := a <= sqrt 2 * b ∧ b <= sqrt 2 * a

-- State the main theorem
theorem not_possible_to_create_3_similar_piles (x₀ : ℝ) (h₀ : 0 < x₀) : 
  ¬ ∃ x₁ x₂ x₃ : ℝ, 
    x₁ + x₂ + x₃ = x₀ ∧ 
    similar x₁ x₂ ∧ 
    similar x₂ x₃ ∧ 
    similar x₃ x₁ :=
sorry

end not_possible_to_create_3_similar_piles_l521_521341


namespace decagon_adjacent_probability_l521_521531

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l521_521531


namespace spencer_total_distance_l521_521300

def d1 : ℝ := 1.2
def d2 : ℝ := 0.6
def d3 : ℝ := 0.9
def d4 : ℝ := 1.7
def d5 : ℝ := 2.1
def d6 : ℝ := 1.3
def d7 : ℝ := 0.8

theorem spencer_total_distance : d1 + d2 + d3 + d4 + d5 + d6 + d7 = 8.6 :=
by
  sorry

end spencer_total_distance_l521_521300


namespace triangle_centroid_l521_521415

noncomputable def centroid (A B C : Point) : Point :=
  (A + B + C) / 3

def isMidpoint (M A B : Point) : Prop :=
  M = (A + B) / 2

def isMedian (A B C : Point) (M : Point) : Prop :=
  isMidpoint M B C

def centroidDividesMedianInRatio (A B C : Point) (G : Point) : Prop :=
  ∀ (M : Point), 
    (isMedian A B C M) → 
    ∃ (α β : ℝ), 
      α + β = 1 ∧ 
      α / β = 2 ∧ 
      G = α • A + β • M

theorem triangle_centroid (A B C G : Point) :
  G = centroid A B C →
  centroidDividesMedianInRatio A B C G :=
sorry

end triangle_centroid_l521_521415


namespace minimum_voters_for_tall_giraffe_to_win_l521_521274

/-- Conditions -/
def total_voters := 135
def districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority (n : Nat) : Nat := Nat.ceil (n / 2)

/-- Proof Problem -/
theorem minimum_voters_for_tall_giraffe_to_win : 
  let precincts_needed_in_district := majority precincts_per_district,
      districts_needed_to_win := majority districts,
      precincts_needed_in_total := districts_needed_to_win * precincts_needed_in_district,
      majority_in_precinct := majority voters_per_precinct in
  precincts_needed_in_total * majority_in_precinct = 30 :=
by
  -- Proof steps (commented out)
  -- 1. Calculate the number of precincts in total: 5 * 9 = 45
  -- 2. Calculate how many districts the Tall giraffe needs to win: 3
  -- (majority districts)
  -- 3. Calculate how many precincts in a district the Tall giraffe needs to
  -- win: 5 (majority precincts_per_district)
  -- 4. Calculate the total number of precincts the Tall giraffe needs to win:
  -- 3 * 5 = 15
  -- 5. Calculate the majority of votes in a precinct: 2 
  -- (majority voters_per_precinct)
  -- 6. Calculate the minimum number of voters in 15 precincts: 15 * 2 = 30
  sorry

end minimum_voters_for_tall_giraffe_to_win_l521_521274


namespace parking_spot_difference_l521_521637

theorem parking_spot_difference 
    (open_first : ℕ) (open_third_diff : ℕ) (open_fourth : ℕ) (total_spots : ℕ) (full_spots : ℕ) (open_total : ℕ) (x : ℕ)
    (h1 : open_first = 58) 
    (h2 : open_third_diff = 5) 
    (h3 : open_fourth = 31) 
    (h4 : total_spots = 4 * 100) 
    (h5 : full_spots = 186) 
    (h6 : open_total = total_spots - full_spots)
    (h7 : open_total = 214) 
    (h8 : 58 + x + (x + 5) + 31 = open_total) :
  x - open_first = 2 :=
by 
  have total_spots_eq : total_spots = 400 := by rw [h4]
  have open_total_eq : open_total = 214 := by rw [h6, h5, total_spots_eq, h7]
  have : 58 + x + (x + 5) + 31 = open_total := by rw [h8, open_total_eq]
  simp at this
  sorry -- proof omitted

end parking_spot_difference_l521_521637


namespace not_possible_to_create_3_piles_l521_521359

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end not_possible_to_create_3_piles_l521_521359


namespace ways_to_walk_between_floors_l521_521455

theorem ways_to_walk_between_floors (n_floors : ℕ) (n_staircases : ℕ) : 
  n_floors = 4 → n_staircases = 2 → (2 * 2 * 2 = 2^3) :=
by
  intros h1 h2
  have h3 : 3 = 3 := by rfl
  have h4 : 2^3 = 2 * 2 * 2 := by
    calc
      2^3 = 2 * 2^2 : by sorry
      ... = 2 * (2 * 2) : by sorry
      ... = 2 * 2 * 2 : by sorry
  
  exact h4

end ways_to_walk_between_floors_l521_521455


namespace expected_value_of_dodecahedral_die_is_6_5_l521_521041

noncomputable def expected_value_of_dodecahedral_die : ℝ := 
  ∑ i in Finset.range (12 + 1), (i : ℝ) / 12

theorem expected_value_of_dodecahedral_die_is_6_5 :
  expected_value_of_dodecahedral_die = 6.5 := sorry

end expected_value_of_dodecahedral_die_is_6_5_l521_521041


namespace possible_integer_side_lengths_l521_521823

theorem possible_integer_side_lengths (a b : ℕ) (h1 : a = 8) (h2 : b = 5) :
  {x : ℕ | 3 < x ∧ x < 13}.finite ∧ {x : ℕ | 3 < x ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521823


namespace term_150th_of_sequence_is_2280_l521_521151

def isPowerOf3OrSumOfDistinctPowersOf3 (n : ℕ) : Prop :=
  ∃ (coeff : ℕ → ℕ) (k : ℕ), -- there is a finitely supported function whose support is bounded by k
    (∀ i : ℕ, coeff i < 2) ∧           
    (n = ∑ i in Finset.range k, coeff i * 3^i)

noncomputable def sequence : ℕ → ℕ
| 0       := 1
| (n + 1) := (Finset.filter isPowerOf3OrSumOfDistinctPowersOf3 (Finset.range (sequence n + 1))).max' ⟨sequence n, sorry⟩ + 1

theorem term_150th_of_sequence_is_2280 : sequence 149 = 2280 :=
sorry 
end term_150th_of_sequence_is_2280_l521_521151


namespace number_of_sides_possibilities_l521_521813

theorem number_of_sides_possibilities (a b : ℕ) (h₁ : a = 8) (h₂ : b = 5) : 
  { x : ℕ // 3 < x ∧ x < 13 }.to_list.length = 9 :=
by
  sorry

end number_of_sides_possibilities_l521_521813


namespace incenter_tangency_concurrency_and_ratio_l521_521917

theorem incenter_tangency_concurrency_and_ratio 
  (A B C A1 B1 C1 : Point) 
  (incircle : Circle) 
  (A1_tangent : tangent_point incircle (segment B C) A1) 
  (B1_tangent : tangent_point incircle (segment C A) B1) 
  (C1_tangent : tangent_point incircle (segment A B) C1)
  (AA1 BB1 CC1 : Line)
  (AA1_through : passes_through (line_through A A1) AA1)
  (BB1_through : passes_through (line_through B B1) BB1)
  (CC1_through : passes_through (line_through C C1) CC1)
  (incircle_radius : Real)
  (circumcircle_radius : Real) :
  ∃ M : Point, concurrent_lines (AA1, BB1, CC1) ∧ 
  (dist M A1 / dist M A) * (dist M B1 / dist M B) * (dist M C1 / dist M C) = incircle_radius / (4 * circumcircle_radius) :=
sorry

end incenter_tangency_concurrency_and_ratio_l521_521917


namespace probability_adjacent_vertices_of_decagon_l521_521515

theorem probability_adjacent_vertices_of_decagon : 
  let n := 10 in -- the number of vertices in a decagon
  let adjacent_probability := (2: ℚ) / (n - 1) in
  adjacent_probability = 2 / 9 := by
  sorry

end probability_adjacent_vertices_of_decagon_l521_521515


namespace minimum_voters_for_tall_giraffe_to_win_l521_521276

/-- Conditions -/
def total_voters := 135
def districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority (n : Nat) : Nat := Nat.ceil (n / 2)

/-- Proof Problem -/
theorem minimum_voters_for_tall_giraffe_to_win : 
  let precincts_needed_in_district := majority precincts_per_district,
      districts_needed_to_win := majority districts,
      precincts_needed_in_total := districts_needed_to_win * precincts_needed_in_district,
      majority_in_precinct := majority voters_per_precinct in
  precincts_needed_in_total * majority_in_precinct = 30 :=
by
  -- Proof steps (commented out)
  -- 1. Calculate the number of precincts in total: 5 * 9 = 45
  -- 2. Calculate how many districts the Tall giraffe needs to win: 3
  -- (majority districts)
  -- 3. Calculate how many precincts in a district the Tall giraffe needs to
  -- win: 5 (majority precincts_per_district)
  -- 4. Calculate the total number of precincts the Tall giraffe needs to win:
  -- 3 * 5 = 15
  -- 5. Calculate the majority of votes in a precinct: 2 
  -- (majority voters_per_precinct)
  -- 6. Calculate the minimum number of voters in 15 precincts: 15 * 2 = 30
  sorry

end minimum_voters_for_tall_giraffe_to_win_l521_521276


namespace sum_of_solutions_l521_521723

def s (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_solutions (n : ℕ) :
  (∃ n₁ n₂ : ℕ, n₁ - 3 * s n₁ = 2022 ∧ n₂ - 3 * s n₂ = 2022) →
  n = 2040 + 2067 →
  2040 + 2067 = 4107 :=
by
  intro h₁ h₂
  rw h₂
  exact rfl

end sum_of_solutions_l521_521723


namespace lesser_number_is_21_5_l521_521985

theorem lesser_number_is_21_5
  (x y : ℝ)
  (h1 : x + y = 50)
  (h2 : x - y = 7) :
  y = 21.5 :=
by
  sorry

end lesser_number_is_21_5_l521_521985


namespace trig_identity_proof_l521_521647

noncomputable def trig_identity (α : ℝ) : Prop :=
  sin α ^ 2 + cos (30 * real.pi / 180 - α) ^ 2 - sin α * cos (30 * real.pi / 180 - α) = 3 / 4

theorem trig_identity_proof (α : ℝ) : trig_identity α :=
by sorry

end trig_identity_proof_l521_521647


namespace expected_value_fair_dodecahedral_die_l521_521054

theorem expected_value_fair_dodecahedral_die :
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  expected_value = 6.5 :=
by
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  have h1 : list.sum outcomes = 78 := by sorry
  have h2 : (12: ℝ) * (1 / 12: ℝ) = 1 := by sorry
  show expected_value = 6.5
  { sorry }

end expected_value_fair_dodecahedral_die_l521_521054


namespace who_scored_full_marks_l521_521657

-- Define students and their statements
inductive Student
| A | B | C

open Student

def scored_full_marks (s : Student) : Prop :=
  match s with
  | A => true
  | B => true
  | C => true

def statement_A : Prop := scored_full_marks A
def statement_B : Prop := ¬ scored_full_marks C
def statement_C : Prop := statement_B

-- Given conditions
def exactly_one_lied (a b c : Prop) : Prop :=
  (a ∧ ¬ b ∧ ¬ c) ∨ (¬ a ∧ b ∧ ¬ c) ∨ (¬ a ∧ ¬ b ∧ c)

-- Main proof statement: Prove that B scored full marks
theorem who_scored_full_marks (h : exactly_one_lied statement_A statement_B statement_C) : scored_full_marks B :=
sorry

end who_scored_full_marks_l521_521657


namespace minimum_voters_for_tall_giraffe_l521_521284

-- Definitions and conditions
def voters : ℕ := 135
def districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3

-- Conditions encoded in the problem statement
def majority_precinct (votes: ℕ) : Prop := votes >= 2
def majority_district (precinct_wins: ℕ) : Prop := precinct_wins >= 5
def majority_winner (district_wins: ℕ) : Prop := district_wins >= 3

-- The problem states that the Tall giraffe won.
axiom tall_giraffe_won : ∃ district_wins : ℕ, 
  majority_winner district_wins ∧ 
  ∀ (d ∈ (finset.range districts)), ∃ precinct_wins : ℕ, 
  majority_district precinct_wins ∧ 
  ∀ (p ∈ (finset.range precincts_per_district)), ∃ votes : ℕ, 
  majority_precinct votes

-- Proof goal
theorem minimum_voters_for_tall_giraffe : ∃ (votes_for_tall : ℕ), votes_for_tall = 30 :=
by {
  -- proof of the theorem will go here
  sorry 
}

end minimum_voters_for_tall_giraffe_l521_521284


namespace triangle_side_length_l521_521886

theorem triangle_side_length 
  (X Z : ℝ) (x z y : ℝ)
  (h1 : x = 36)
  (h2 : z = 72)
  (h3 : Z = 4 * X) :
  y = 72 := by
  sorry

end triangle_side_length_l521_521886


namespace solution_of_equation_l521_521451

theorem solution_of_equation (x : ℤ) : 7 * x - 5 = 6 * x → x = 5 := by
  intro h
  sorry

end solution_of_equation_l521_521451


namespace correlation_proof_l521_521180

noncomputable def test_correlation (n : ℕ) (r_B : ℝ) (r_rho : ℝ) (alpha : ℝ) : Prop :=
  let T := r_B * (real.sqrt (n - 2)) / (real.sqrt (1 - r_B ^ 2)) in
  let df := n - 2 in
  let critical_value := 1.99 in -- Approximated value for the critical value with df=98 and alpha/2=0.025
  T > critical_value

theorem correlation_proof : 
  test_correlation 100 0.2 0 0.05 → true :=
by
  intros _ -- Note: this is a placeholder for the actual proof logic.
  sorry -- Proof elided.

end correlation_proof_l521_521180


namespace triangle_side_lengths_count_l521_521792

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end triangle_side_lengths_count_l521_521792


namespace decagon_adjacent_vertex_probability_l521_521550

theorem decagon_adjacent_vertex_probability : 
  ∀ (V : Type) (vertices : Finset V), 
    vertices.card = 10 → 
    (∀ v : V, ∃ u₁ u₂ : V, u₁ ≠ v ∧ u₂ ≠ v ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ (u₁ = v + 1 ∨ u₁ = v - 1) ∧ (u₂ = v + 1 ∨ u₂ = v - 1)) →
    (∃ (u₁ u₂ : V), u₁ ≠ u₂ ∧ u₁ ∈ vertices ∧ u₂ ∈ vertices ∧ Prob (u₁, u₂) = 2/9) :=
begin
  sorry
end

end decagon_adjacent_vertex_probability_l521_521550


namespace expected_value_of_fair_dodecahedral_die_l521_521049

theorem expected_value_of_fair_dodecahedral_die : 
  (1/12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_of_fair_dodecahedral_die_l521_521049


namespace f_eq_l521_521222

open Nat

def f (n : ℕ) : ℕ := (range (n + 3)).sum (λ i, 2^(3 * i + 4))

theorem f_eq (n : ℕ) : f n = 16 * (8^(n+3) - 1) / 7 :=
by
  sorry

end f_eq_l521_521222


namespace triangle_side_lengths_l521_521843

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end triangle_side_lengths_l521_521843


namespace function_domain_range_eq_l521_521658

noncomputable def e_ln_domain_and_range : set ℝ := set.Ioi 0

theorem function_domain_range_eq :
  ∀ (f : ℝ → ℝ),
  (∀ x, f x = e ^ (Real.log x) → x ∈ e_ln_domain_and_range → f x ∈ e_ln_domain_and_range) →
  (∀ x, f x = 1 / Real.sqrt x → x ∈ e_ln_domain_and_range → f x ∈ e_ln_domain_and_range) :=
by
  -- domain and range conditions for e_ln
  let e_ln : ℝ → ℝ := λ x, Real.exp (Real.log x)
  have e_ln_dom : ∀ x, x ∈ e_ln_domain_and_range → e_ln x ∈ e_ln_domain_and_range,
  sorry
  
  -- domain and range conditions for 1/sqrt(x)
  let inv_sqrt : ℝ → ℝ := λ x, 1 / Real.sqrt x
  have inv_sqrt_dom : ∀ x, x ∈ e_ln_domain_and_range → inv_sqrt x ∈ e_ln_domain_and_range,
  sorry

  exact ⟨inv_sqrt_dom⟩

end function_domain_range_eq_l521_521658


namespace not_possible_to_create_3_similar_piles_l521_521342

-- Formalize the definition of similar piles
def similar (a b : ℝ) : Prop := a <= sqrt 2 * b ∧ b <= sqrt 2 * a

-- State the main theorem
theorem not_possible_to_create_3_similar_piles (x₀ : ℝ) (h₀ : 0 < x₀) : 
  ¬ ∃ x₁ x₂ x₃ : ℝ, 
    x₁ + x₂ + x₃ = x₀ ∧ 
    similar x₁ x₂ ∧ 
    similar x₂ x₃ ∧ 
    similar x₃ x₁ :=
sorry

end not_possible_to_create_3_similar_piles_l521_521342


namespace min_voters_tall_giraffe_win_l521_521279

-- Definitions from the problem statement as conditions
def precinct_voters := 3
def precincts_per_district := 9
def districts := 5
def majority_precincts(p : ℕ) := p / 2 + 1  -- Minimum precincts won in a district 
def majority_districts(d : ℕ) := d / 2 + 1  -- Minimum districts won in the final

-- Condition: majority precincts to win a district
def precinct_votes_to_win := majority_precincts precinct_voters

-- Condition: majority districts to win the final
def district_wins_to_win_final := majority_districts districts

-- Minimum precincts the Tall giraffe needs to win overall
def total_precincts_to_win := district_wins_to_win_final * majority_precincts precincts_per_district

-- Proof that the minimum number of voters who could have voted for the Tall giraffe is 30
theorem min_voters_tall_giraffe_win :
  precinct_votes_to_win * total_precincts_to_win = 30 :=
sorry

end min_voters_tall_giraffe_win_l521_521279


namespace arithmetic_mean_inequality_l521_521308

theorem arithmetic_mean_inequality {n : ℕ} (x : Fin (2 * n - 1) → ℝ) (A : ℝ) 
  (h1 : ∀ i j, (i : ℕ) ≤ j → x i ≤ x j) 
  (h2 : (∑ i, x i) / (2 * n - 1) = A) :
  2 * (∑ i, (x i - A) ^ 2) ≥ ∑ i, (x i - x (Fin.ofNat n)) ^ 2 := 
sorry

end arithmetic_mean_inequality_l521_521308


namespace find_210th_number_l521_521103

-- Step a): Definitions based on conditions
def cyclic_sequence (n : ℕ) : ℕ → ℝ := λ i, sorry -- Placeholder for the actual sequence

axiom seq_periodicity : ∀ (n : ℕ), cyclic_sequence n = cyclic_sequence (n + 20)

axiom sum_consecutive_20 : ∀ (n : ℕ), (∑ i in finset.range 20, cyclic_sequence (n + i)) = 72

axiom a17 : cyclic_sequence 17 = 3
axiom a83 : cyclic_sequence 83 = 4
axiom a144 : cyclic_sequence 144 = 9

-- Step c): Final proof problem
theorem find_210th_number : cyclic_sequence 210 = -8 / 5 :=
by sorry

end find_210th_number_l521_521103


namespace equilateral_triangle_parabola_hyperbola_l521_521736

open Real

noncomputable def parabola (p : ℝ) := { x : ℝ × ℝ // x.2^2 = 2 * p * x.1 }
noncomputable def hyperbola := { x : ℝ × ℝ // x.2^2 - x.1^2 = 1 }
def focus (p : ℝ) := (0, p / 2)

theorem equilateral_triangle_parabola_hyperbola (p : ℝ) (hp : 0 < p) :
  (∃ A B : ℝ × ℝ, 
    A ∈ hyperbola ∧ 
    B ∈ hyperbola ∧ 
    A.2 = -p / 2 ∧
    B.2 = -p / 2 ∧
    (0, p / 2) = focus p ∧ 
    equilateral_triangle (0, p / 2) A B) ↔ 
  p = 2 * sqrt 3 :=
sorry

end equilateral_triangle_parabola_hyperbola_l521_521736


namespace find_f_x_l521_521149

def tan : ℝ → ℝ := sorry  -- tan function placeholder
def cos : ℝ → ℝ := sorry  -- cos function placeholder
def sin : ℝ → ℝ := sorry  -- sin function placeholder

axiom conditions : 
  tan 45 = 1 ∧
  cos 60 = 2 ∧
  sin 90 = 3 ∧
  cos 180 = 4 ∧
  sin 270 = 5

theorem find_f_x :
  ∃ f x, (f x = 6) ∧ 
  (f = tan ∧ x = 360) := 
sorry

end find_f_x_l521_521149


namespace parabola_translation_l521_521996

theorem parabola_translation (x : ℝ) : 
  let y := x^2 in
  let y_translated := (x - 2)^2 + 1 in 
  ∀ x, y_translated = (x - 2)^2 + 1 :=
by
  sorry

end parabola_translation_l521_521996


namespace expected_value_dodecahedral_die_is_6_5_l521_521063

noncomputable def expected_value_dodecahedral_die : ℝ :=
  let outcomes := finset.range 12 in
  1 / 12 * (finset.sum outcomes (λ i, (i + 1 : ℝ)))

theorem expected_value_dodecahedral_die_is_6_5 :
  expected_value_dodecahedral_die = 6.5 :=
by
  sorry

end expected_value_dodecahedral_die_is_6_5_l521_521063


namespace not_possible_to_create_3_similar_piles_l521_521346

-- Formalize the definition of similar piles
def similar (a b : ℝ) : Prop := a <= sqrt 2 * b ∧ b <= sqrt 2 * a

-- State the main theorem
theorem not_possible_to_create_3_similar_piles (x₀ : ℝ) (h₀ : 0 < x₀) : 
  ¬ ∃ x₁ x₂ x₃ : ℝ, 
    x₁ + x₂ + x₃ = x₀ ∧ 
    similar x₁ x₂ ∧ 
    similar x₂ x₃ ∧ 
    similar x₃ x₁ :=
sorry

end not_possible_to_create_3_similar_piles_l521_521346


namespace impossible_to_create_3_piles_l521_521375

def similar_sizes (x y : ℝ) : Prop := x ≤ sqrt 2 * y

theorem impossible_to_create_3_piles (x y : ℝ) (h₁ : similar_sizes x y) :
  ¬ ∃ (x₁ x₂ x₃ : ℝ), x₁ + x₂ + x₃ = x + y ∧ similar_sizes x₁ x₂ ∧ similar_sizes x₂ x₃ ∧ similar_sizes x₁ x₃ :=
by
  sorry

end impossible_to_create_3_piles_l521_521375


namespace area_of_right_triangle_ABC_is_6_l521_521872

-- Define the points A, B, and C
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (4, 0)
def C : (ℝ × ℝ) := (0, 3)

-- Define the lengths of segments AB and AC
def lengthAB := real.dist A B
def lengthAC := real.dist A C

-- Define the area of the right-angled triangle ABC
def area_ABC := 0.5 * lengthAB * lengthAC

-- Theorem statement: The area of the right-angled triangle ABC is 6 square units
theorem area_of_right_triangle_ABC_is_6 : area_ABC = 6 := 
by {
  -- Calculations can be carried out inside this block if needed
  sorry
}

end area_of_right_triangle_ABC_is_6_l521_521872


namespace sum_possible_values_l521_521325

theorem sum_possible_values (p q r s : ℝ) (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 4) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -1 := 
by
  sorry

end sum_possible_values_l521_521325


namespace a_n_1994_l521_521002

noncomputable def seq (a : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := (seq a n * real.sqrt 3 + 1) / (real.sqrt 3 - seq a n)

theorem a_n_1994 (a : ℝ) : seq a 1994 = (a + real.sqrt 3) / (1 - a * real.sqrt 3) :=
sorry

end a_n_1994_l521_521002


namespace math_problem_l521_521190

def a : ℝ := 1.25 * 10^(-2016)
def b : ℝ := 8 * 10^(-2018)

theorem math_problem :
  let result := a * b + a / b
  let int_part := result.to_real.num
  let dec_part := result - int_part
  int_part = 15 ∧
  ∃ zeros_count, zeros_count = 4031 ∧ (0.5 * 10^(-zeros_count) + 0.01) = dec_part :=
by
  sorry

end math_problem_l521_521190


namespace cos_inner_product_l521_521314

variables {ℝ : Type*} [inner_product_space ℝ E] 
variables {a b : E}
variables (hab : ‖a‖ = 1 ∧ ‖b‖ = 1) 
variables (proj_condition : (a + b) = (2 / 3) • b)

theorem cos_inner_product (hab : ‖a‖ = 1 ∧ ‖b‖ = 1) 
(proj_condition : (a + b) = (2 / 3) • b) :
cos_angle a b = -1 / 3 :=
sorry

end cos_inner_product_l521_521314


namespace amoeba_population_after_ten_days_l521_521466

theorem amoeba_population_after_ten_days
  (P₀ : ℕ) (r : ℕ) (n : ℕ) (h₀ : P₀ = 2) (h₁ : r = 3) (h₂ : n = 10) :
  P₀ * r^n = 118098 :=
by
  rw [h₀, h₁, h₂]
  norm_num
  sorry

end amoeba_population_after_ten_days_l521_521466


namespace expected_value_fair_dodecahedral_die_l521_521069

theorem expected_value_fair_dodecahedral_die : 
  let n := 12 in (1 / n) * (List.sum (List.range n).map (λ x => (x + 1))) = 6.5 :=
by
  let n := 12
  have h1 : (1 : ℚ) / n = 1 / 12 := by sorry
  have h2 : List.range n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] := by sorry
  have h3 : List.sum (List.range n).map (λ x => (x + 1)) = 78 := by sorry
  have h4 : (1 / 12) * 78 = 6.5 := by sorry
  exact h4

end expected_value_fair_dodecahedral_die_l521_521069


namespace werewolf_is_A_l521_521885

def is_liar (x : ℕ) : Prop := sorry
def is_knight (x : ℕ) : Prop := sorry
def is_werewolf (x : ℕ) : Prop := sorry

axiom A : ℕ
axiom B : ℕ
axiom C : ℕ

-- Conditions from the problem
axiom A_statement : is_liar A ∨ is_liar B
axiom B_statement : is_werewolf C
axiom exactly_one_werewolf : 
  (is_werewolf A ∧ ¬ is_werewolf B ∧ ¬ is_werewolf C) ∨
  (is_werewolf B ∧ ¬ is_werewolf A ∧ ¬ is_werewolf C) ∨
  (is_werewolf C ∧ ¬ is_werewolf A ∧ ¬ is_werewolf B)
axiom werewolf_is_knight : ∀ x : ℕ, is_werewolf x → is_knight x

-- Prove the conclusion
theorem werewolf_is_A : 
  is_werewolf A ∧ is_knight A :=
sorry

end werewolf_is_A_l521_521885


namespace triangle_properties_l521_521868

theorem triangle_properties
  (a b : ℝ)
  (C : ℝ)
  (ha : a = 2)
  (hb : b = 3)
  (hC : C = Real.pi / 3)
  :
  let c := Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)
  let area := (1 / 2) * a * b * Real.sin C
  let sin2A := 2 * (a * Real.sin C / c) * Real.sqrt (1 - (a * Real.sin C / c)^2)
  c = Real.sqrt 7 
  ∧ area = (3 * Real.sqrt 3) / 2 
  ∧ sin2A = (4 * Real.sqrt 3) / 7 :=
by
  sorry

end triangle_properties_l521_521868


namespace largest_integer_less_than_log_sum_l521_521481

theorem largest_integer_less_than_log_sum :
  let sum_logs := (finset.range 3010).sum (λ n, real.logb 3 ((n + 3) / (n + 2)))
  floor sum_logs < 7 :=
by
  sorry

end largest_integer_less_than_log_sum_l521_521481


namespace length_of_the_train_l521_521131

-- Define the conditions
def train_speed_kmph : ℝ := 69.994
def man_speed_kmph : ℝ := 5
def time_sec : ℝ := 6

-- Convert speeds from kmph to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

-- Calculate the relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (train_speed_kmph + man_speed_kmph)

-- The length of the train
def train_length : ℝ := relative_speed_mps * time_sec

-- The theorem to prove
theorem length_of_the_train : train_length = 124.9866 :=
by
  -- Skipping the proof
  sorry

end length_of_the_train_l521_521131


namespace number_of_badminton_players_l521_521259

-- Definitions based on the given conditions
variable (Total_members : ℕ := 30)
variable (Tennis_players : ℕ := 19)
variable (No_sport_players : ℕ := 3)
variable (Both_sport_players : ℕ := 9)

-- The goal is to prove the number of badminton players is 17
theorem number_of_badminton_players :
  ∀ (B : ℕ), Total_members = B + Tennis_players - Both_sport_players + No_sport_players → B = 17 :=
by
  intro B
  intro h
  sorry

end number_of_badminton_players_l521_521259


namespace probability_adjacent_vertices_in_decagon_l521_521571

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l521_521571


namespace inequality_solution_range_l521_521017

theorem inequality_solution_range (a : ℝ) : 
  (∀ x ∈ ({1, 2, 3, 4} : set ℝ), 0 ≤ a * x + 5 ∧ a * x + 5 ≤ 4) ↔ (-5 / 4 ≤ a ∧ a < -1) :=
by {
  sorry
}

end inequality_solution_range_l521_521017


namespace at_most_nine_elements_l521_521304

theorem at_most_nine_elements (A : Set ℕ) (hA : ∀ (x y ∈ A), x ≠ y → |x - y| ≥ (x * y / 25)) :
  A.Finite ∧ A.toFinset.card ≤ 9 := by
sorry

end at_most_nine_elements_l521_521304


namespace outermost_rectangle_perimeter_l521_521967

theorem outermost_rectangle_perimeter (z x : ℝ) (h₁: z > x) :
  ∃ P : ℝ, P = 2 * (z - x) :=
by
  use 2 * (z - x)
  field_simp
  sorry

end outermost_rectangle_perimeter_l521_521967


namespace shop_weekly_earnings_l521_521670

theorem shop_weekly_earnings
  (price_women: ℕ := 18)
  (price_men: ℕ := 15)
  (time_open_hours: ℕ := 12)
  (minutes_per_hour: ℕ := 60)
  (weekly_days: ℕ := 7)
  (sell_rate_women: ℕ := 30)
  (sell_rate_men: ℕ := 40) :
  (time_open_hours * (minutes_per_hour / sell_rate_women) * price_women +
   time_open_hours * (minutes_per_hour / sell_rate_men) * price_men) * weekly_days = 4914 := 
sorry

end shop_weekly_earnings_l521_521670


namespace problem_solution_l521_521179

noncomputable def time_min_distance
  (c : ℝ) (α : ℝ) (a : ℝ) : ℝ :=
a * (Real.cos α) / (2 * c * (1 - Real.sin α))

noncomputable def min_distance
  (c : ℝ) (α : ℝ) (a : ℝ) : ℝ :=
a * Real.sqrt ((1 - (Real.sin α)) / 2)

theorem problem_solution (α : ℝ) (c : ℝ) (a : ℝ) 
  (α_30 : α = Real.pi / 6) (c_50 : c = 50) (a_50sqrt3 : a = 50 * Real.sqrt 3) :
  (time_min_distance c α a = 1.5) ∧ (min_distance c α a = 25 * Real.sqrt 3) :=
by
  sorry

end problem_solution_l521_521179


namespace time_to_ascend_non_working_escalator_l521_521470

-- Definitions from the conditions
def length_of_escalator := 1
def time_standing := 1
def time_running := 24 / 60
def escalator_speed := 1 / 60
def gavrila_speed := 1 / 40

-- The proof problem statement 
theorem time_to_ascend_non_working_escalator 
  (length_of_escalator : ℝ)
  (time_standing : ℝ)
  (time_running : ℝ)
  (escalator_speed : ℝ)
  (gavrila_speed : ℝ) :
  time_standing = 1 →
  time_running = 24 / 60 →
  escalator_speed = 1 / 60 →
  gavrila_speed = 1 / 40 →
  length_of_escalator = 1 →
  1 / gavrila_speed = 40 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end time_to_ascend_non_working_escalator_l521_521470


namespace arcsin_arccos_eq_arctan_plus_pi_by_4_l521_521954

theorem arcsin_arccos_eq_arctan_plus_pi_by_4 (x : ℝ) :
  arcsin x + arccos (1 - x) = arctan x + π / 4 → x = 0 :=
begin
  intro h,
  sorry
end

end arcsin_arccos_eq_arctan_plus_pi_by_4_l521_521954


namespace range_of_k_no_third_quadrant_l521_521862

theorem range_of_k_no_third_quadrant (k : ℝ) : ¬(∃ x : ℝ, ∃ y : ℝ, x < 0 ∧ y < 0 ∧ y = k * x + 3) → k ≤ 0 := 
sorry

end range_of_k_no_third_quadrant_l521_521862


namespace car_B_speed_is_50_l521_521681

def car_speeds (v_A v_B : ℕ) (d_init d_ahead t : ℝ) : Prop :=
  v_A * t = v_B * t + d_init + d_ahead

theorem car_B_speed_is_50 :
  car_speeds 58 50 10 8 2.25 :=
by
  sorry

end car_B_speed_is_50_l521_521681


namespace line_circle_relation_l521_521203

def A : Point := (1, 2)
def B : Point := (3, 2)
def AB_midpoint : Point := midpoint A B
def radius : ℝ := distance A B / 2
def center : Point := AB_midpoint
def line_l (x y : ℝ) : Prop := x + y - 3 = 0
def distance_center_to_line : ℝ := abs (center.1 + center.2 - 3) / sqrt 2

theorem line_circle_relation : 
  distance_center_to_line = sqrt 2 / 2 ∧ sqrt 2 / 2 < radius → intersects_not_pass_through center radius line_l := 
by 
  sorry

end line_circle_relation_l521_521203


namespace problem_1_problem_2_l521_521226

noncomputable def f (x a : ℝ) : ℝ := x * abs (x - a)

theorem problem_1 (a : ℝ) (h : f 1 a + f (-1) a > 1) : a ∈ Set.Iio (-1/2) :=
sorry

theorem problem_2 (a : ℝ) (h1 : 0 < a) (h2 : ∀ x y ∈ Set.Iic a, f x a ≤ abs (y + 5 / 4) + abs (y - a)) : a ∈ Set.Ioo 0 5 :=
sorry

end problem_1_problem_2_l521_521226


namespace abs_pi_minus_abs_pi_minus_9_eq_9_minus_2pi_l521_521694

theorem abs_pi_minus_abs_pi_minus_9_eq_9_minus_2pi :
  (|π - |π - 9|| = 9 - 2 * π) :=
sorry

end abs_pi_minus_abs_pi_minus_9_eq_9_minus_2pi_l521_521694


namespace probability_adjacent_vertices_in_decagon_l521_521562

def decagon_adjacent_vertex_probability: ℚ :=
  2 / 9

theorem probability_adjacent_vertices_in_decagon :
  let vertices := 10
  in ∃ (p : ℚ), p = decagon_adjacent_vertex_probability :=
by
  sorry

end probability_adjacent_vertices_in_decagon_l521_521562


namespace solve_inequality_l521_521434

namespace InequalityProof

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem solve_inequality (x : ℝ) : cube_root x + 3 / (cube_root x + 4) ≤ 0 ↔ x ∈ Set.Icc (-27 : ℝ) (-1 : ℝ) :=
by
  have y_eq := cube_root x
  sorry

end InequalityProof

end solve_inequality_l521_521434


namespace eccentricity_of_ellipse_l521_521212

theorem eccentricity_of_ellipse (a b : ℝ) (h : a = 2 * b) : 
  let c := Real.sqrt (a^2 - b^2)
  in (c / a) = (Real.sqrt 3 / 2) :=
by
  sorry

end eccentricity_of_ellipse_l521_521212


namespace expected_value_of_dodecahedral_die_is_6_5_l521_521042

noncomputable def expected_value_of_dodecahedral_die : ℝ := 
  ∑ i in Finset.range (12 + 1), (i : ℝ) / 12

theorem expected_value_of_dodecahedral_die_is_6_5 :
  expected_value_of_dodecahedral_die = 6.5 := sorry

end expected_value_of_dodecahedral_die_is_6_5_l521_521042


namespace isosceles_triangle_perimeter_l521_521016

-- Define the quadratic equation and its roots.
def quadratic_roots (a b c : ℝ) : set ℝ := {x : ℝ | a * x^2 + b * x + c = 0}

-- Define the isosceles triangle sides and the perimeter calculation.
def isosceles_triangle_perimeters (a b c : ℝ) (x1 x2 : ℝ) : set ℝ :=
  let triangle1 := [x1, x1, x2]
  let triangle2 := [x1, x2, x2]
  in {triangle1.sum, triangle2.sum}

-- The main statement.
theorem isosceles_triangle_perimeter :
  quadratic_roots 1 (-7) 12 = {3, 4} →
  isosceles_triangle_perimeters 1 (-7) 12 3 4 = {10, 11} :=
by sorry

end isosceles_triangle_perimeter_l521_521016


namespace expected_value_dodecahedral_die_l521_521086

noncomputable def dodecahedral_expected_value : ℝ :=
  let outcomes := (list.range (12 + 1)).tail -- List of numbers from 1 to 12
  let n_faces := 12
  let probability := 1 / (n_faces : ℝ)
  let expected_value := probability * outcomes.sum / n_faces
  expected_value

theorem expected_value_dodecahedral_die :
  dodecahedral_expected_value = 6.5 :=
by {
  unfold dodecahedral_expected_value,
  simp,
  sorry,
}

end expected_value_dodecahedral_die_l521_521086


namespace compare_constants_l521_521728

-- Define the constants a, b, and c
def a : ℝ := (1.01)^(-100 : ℤ)
def b : ℝ := Real.sin (Real.pi / 10)
def c : ℝ := 1 / Real.pi

-- Prove the inequality a > c > b
theorem compare_constants : a > c ∧ c > b :=
by
  -- It follows from the problem solution that
  -- 1.01^-100 ≈ 1/e and hence it can be used to validate results
  -- plug in the inequalities derived therein
  sorry

end compare_constants_l521_521728


namespace decagon_adjacent_probability_l521_521537

-- Define the vertices of the decagon and adjacency properties.
def decagon := fin 10

-- Function that checks if two vertices are adjacent
def adjacent (v₁ v₂ : decagon) : Prop :=
  (v₁ = (v₂ + 1) % 10) ∨ (v₁ = (v₂ - 1 + 10) % 10)

-- Define the probability of picking two adjacent vertices
def prob_adjacent : ℚ :=
  2 / 9

-- Lean statement to prove the probability of two distinct random vertices being adjacent
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : decagon), v₁ ≠ v₂ → 
  (if adjacent v₁ v₂ then true else false) → 
  (∑ v2 in univ.erase v₁, if adjacent v₁ v₂ then 1 else 0 : ℚ) / 9 = prob_adjacent :=
sorry

end decagon_adjacent_probability_l521_521537


namespace expected_value_fair_dodecahedral_die_l521_521070

theorem expected_value_fair_dodecahedral_die : 
  let n := 12 in (1 / n) * (List.sum (List.range n).map (λ x => (x + 1))) = 6.5 :=
by
  let n := 12
  have h1 : (1 : ℚ) / n = 1 / 12 := by sorry
  have h2 : List.range n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] := by sorry
  have h3 : List.sum (List.range n).map (λ x => (x + 1)) = 78 := by sorry
  have h4 : (1 / 12) * 78 = 6.5 := by sorry
  exact h4

end expected_value_fair_dodecahedral_die_l521_521070


namespace prove_PS_eq_QR_angle_PQT_trisected_l521_521399

variable (P Q R S T : Type) [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited S] [Inhabited T]

-- Given P, Q, R, S are points on a circle in that order
axiom points_on_circle (hP : P) (hQ : Q) (hR : R) (hS : S) : Prop

-- Given lines PQ and SR are parallel
axiom PQ_parallel_SR : Prop

-- Given QR = SR
axiom QR_eq_SR (QR SR : ℝ) : QR = SR

-- Given QT is a tangent at Q and ∠RQT is acute
axiom QT_tangent (QT : ℝ) (angle_RQT : ℝ) (acute_RQT : angle_RQT < 90) : Prop

-- Proving PS = QR
theorem prove_PS_eq_QR (QR : ℝ) : (QR = SR) → (PS = QR) := by
  sorry

-- Proving ∠PQT is trisected by QR and QS
theorem angle_PQT_trisected (PQT PQS QSR : ℝ) : (PQT = 3 * PQS) → (PQS = QSR) → (angle PQT trisected_by QR QS) := by
  sorry

end prove_PS_eq_QR_angle_PQT_trisected_l521_521399


namespace smallest_possible_value_of_N_l521_521014

-- Conditions definition:
variable (a b c d : ℕ)
variable (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
variable (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
variable (gcd_ab : Int.gcd a b = 1)
variable (gcd_ac : Int.gcd a c = 2)
variable (gcd_ad : Int.gcd a d = 4)
variable (gcd_bc : Int.gcd b c = 5)
variable (gcd_bd : Int.gcd b d = 3)
variable (gcd_cd : Int.gcd c d = N)
variable (hN : N > 5)

-- Statement to prove:
theorem smallest_possible_value_of_N : N = 14 := sorry

end smallest_possible_value_of_N_l521_521014


namespace expected_value_fair_dodecahedral_die_l521_521059

theorem expected_value_fair_dodecahedral_die :
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  expected_value = 6.5 :=
by
  let outcomes := list.range (12 + 1) -- List of outcomes from 1 to 12
  let probabilities := list.replicate 12 (1 / 12: ℝ) -- List of probabilities, each being 1/12
  let expected_value := list.sum (list.zip_with (λ x p, x * p) outcomes probabilities) -- Expected value calculation
  have h1 : list.sum outcomes = 78 := by sorry
  have h2 : (12: ℝ) * (1 / 12: ℝ) = 1 := by sorry
  show expected_value = 6.5
  { sorry }

end expected_value_fair_dodecahedral_die_l521_521059


namespace total_students_l521_521029

theorem total_students (f1 f2 f3 total : ℕ)
  (h_ratio : f1 * 2 = f2)
  (h_ratio2 : f1 * 3 = f3)
  (h_f1 : f1 = 6)
  (h_total : total = f1 + f2 + f3) :
  total = 48 :=
by
  sorry

end total_students_l521_521029


namespace internet_plan_comparison_l521_521133

theorem internet_plan_comparison (d : ℕ) :
    3000 + 200 * d > 5000 → d > 10 :=
by
  intro h
  -- Proof will be written here
  sorry

end internet_plan_comparison_l521_521133


namespace expected_value_of_dodecahedral_die_is_6_5_l521_521040

noncomputable def expected_value_of_dodecahedral_die : ℝ := 
  ∑ i in Finset.range (12 + 1), (i : ℝ) / 12

theorem expected_value_of_dodecahedral_die_is_6_5 :
  expected_value_of_dodecahedral_die = 6.5 := sorry

end expected_value_of_dodecahedral_die_is_6_5_l521_521040


namespace cyclist_pedestrian_meeting_distance_l521_521410

theorem cyclist_pedestrian_meeting_distance :
  let A := 0 -- Representing point A at 0 km
  let B := 3 -- Representing point B at 3 km since AB = 3 km
  let C := 7 -- Representing point C at 7 km since AC = AB + BC = 3 + 4 = 7 km
  exists (x : ℝ), x > 0 ∧ x < 3 ∧ x = 2.1 :=
sorry

end cyclist_pedestrian_meeting_distance_l521_521410


namespace plane_equation_l521_521713

open Real

theorem plane_equation (a b c : ℝ × ℝ × ℝ) (ha : a = (0, 3, -1)) (hb : b = (2, 3, 1)) (hc : c = (4, 1, 0)) : 
  ∃ A B C D : ℤ, A > 0 ∧ Int.gcd (Int.gcd A (Int.gcd B C)) D = 1 ∧ 
  (∀ (x y z : ℝ), (A:ℝ) * x + (B:ℝ) * y + (C:ℝ) * z + (D:ℝ) = 0 ↔ 
   x = 0 ∧ y = 3 ∧ z = -1) :=
begin
  sorry
end

end plane_equation_l521_521713


namespace sum_of_first_five_terms_geometric_sequence_l521_521215

noncomputable def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

theorem sum_of_first_five_terms_geometric_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) :
  geometric_sequence a →
  (∀ n : ℕ, S n = (a 0 * (1 - (classical.some (geometric_sequence a)) ^ n)) / (1 - classical.some (geometric_sequence a))) →
  a 1 * a 2 = 2 * a 0 →
  (a 3 + 2 * a 6) / 2 = 5 / 4 →
  S 5 = 31 :=
by
  sorry

end sum_of_first_five_terms_geometric_sequence_l521_521215


namespace fiona_pairs_l521_521174

theorem fiona_pairs :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 15 → 45 ≤ (n * (n - 1) / 2) ∧ (n * (n - 1) / 2) ≤ 105 :=
by
  intro n
  intro h
  have h₁ : n ≥ 10 := h.left
  have h₂ : n ≤ 15 := h.right
  sorry

end fiona_pairs_l521_521174


namespace birds_in_nests_l521_521953

theorem birds_in_nests (birds : Fin 6 → Fin 6) (move : ∀ n : Fin 6, birds n = (birds n + 1) % 6 ∨ birds n = (birds n - 1) % 6) :
  ¬ ∃ n : Fin 6, ∀ m : Fin 6, birds m = n := 
sorry

end birds_in_nests_l521_521953


namespace f_at_pi_f_increasing_intervals_l521_521968

noncomputable def f (x : ℝ) := Real.cos (4 * x - π / 3)

theorem f_at_pi : f π = 1 / 2 := by
  sorry

theorem f_increasing_intervals (k : ℤ) :
  ∃ a b : ℝ, a = (k * π / 2) - (π / 6) ∧ b = (k * π / 2) + (π / 12) ∧ ∀ x : ℝ, a < x ∧ x < b → f' x > 0 := by
  sorry

end f_at_pi_f_increasing_intervals_l521_521968


namespace expected_value_of_fair_dodecahedral_die_l521_521047

theorem expected_value_of_fair_dodecahedral_die : 
  (1/12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_of_fair_dodecahedral_die_l521_521047


namespace expected_value_fair_dodecahedral_die_l521_521072

theorem expected_value_fair_dodecahedral_die : 
  let n := 12 in (1 / n) * (List.sum (List.range n).map (λ x => (x + 1))) = 6.5 :=
by
  let n := 12
  have h1 : (1 : ℚ) / n = 1 / 12 := by sorry
  have h2 : List.range n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] := by sorry
  have h3 : List.sum (List.range n).map (λ x => (x + 1)) = 78 := by sorry
  have h4 : (1 / 12) * 78 = 6.5 := by sorry
  exact h4

end expected_value_fair_dodecahedral_die_l521_521072


namespace problem_solution_l521_521312

open Matrix

variable {α β : Type} [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]
variable (N : Matrix α β ℝ)

-- Define the conditions
def condition1 : Prop := N.mul_vec ![3, -2] = ![4, 1]
def condition2 : Prop := N.mul_vec ![-4, 6] = ![-2, 0]

-- Define the statement to be proven
theorem problem_solution (h1 : condition1 N) (h2 : condition2 N) : 
  N.mul_vec ![7, 0] = ![14, 4.2] :=
sorry

end problem_solution_l521_521312


namespace arranging_numbers_l521_521473

theorem arranging_numbers (squares : list ℕ) (higher_lower : list (ℕ × ℕ)) :
  (∀ (sq ∈ squares), sq ∈ (list.range 6).map (λ n, n + 1)) →
  (∀ (a b : ℕ), (a, b) ∈ higher_lower → a > b) →
  squares.length = 6 →
  higher_lower.length = 5 →
  ∃ (arrangements : ℕ), arrangements = 12 :=
begin
  sorry
end

end arranging_numbers_l521_521473


namespace max_garden_area_l521_521297

-- Definitions of conditions
def shorter_side (s : ℕ) := s
def longer_side (s : ℕ) := 2 * s
def total_perimeter (s : ℕ) := 2 * shorter_side s + 2 * longer_side s 
def garden_area (s : ℕ) := shorter_side s * longer_side s

-- Theorem with given conditions and conclusion to be proven
theorem max_garden_area (s : ℕ) (h_perimeter : total_perimeter s = 480) : garden_area s = 12800 :=
by
  sorry

end max_garden_area_l521_521297


namespace problem_l521_521243

def Y (a b : ℤ) : ℤ := a^2 - 2 * a * b + b^2
def Z (a b : ℤ) : ℤ := a * b + a + b

theorem problem
  : Z (Y 5 3) (Y 2 1) = 9 := by
  sorry

end problem_l521_521243


namespace Z_nonneg_int_of_le_Z_not_int_infinite_a_l521_521201

noncomputable def Z (a b : ℕ) : ℚ :=
  (nat.factorial (3 * a) * nat.factorial (4 * b)) / 
  ((nat.factorial a) ^ 4 * (nat.factorial b) ^ 3)

-- Proving a ≤ b implies Z(a, b) is a non-negative integer.
theorem Z_nonneg_int_of_le (a b : ℕ) (h : a ≤ b) : 
  Z a b ∈ set.Ici (0 : ℚ) :=
sorry

-- Proving there are infinitely many a such that Z(a, b) is not an integer for any non-negative integer b.
theorem Z_not_int_infinite_a (b : ℕ) : 
  ∃^∞ a, Z a b ∉ set.Ici (0 : ℚ) :=
sorry

end Z_nonneg_int_of_le_Z_not_int_infinite_a_l521_521201


namespace canvas_bag_lower_carbon_solution_l521_521928

theorem canvas_bag_lower_carbon_solution :
  let canvas_release_oz := 9600
  let plastic_per_trip_oz := 32
  canvas_release_oz / plastic_per_trip_oz = 300 :=
by
  sorry

end canvas_bag_lower_carbon_solution_l521_521928


namespace probability_quarter_circle_is_pi_div_16_l521_521776

open Real

noncomputable def probability_quarter_circle : ℝ :=
  let side_length := 2
  let total_area := side_length * side_length
  let quarter_circle_area := π / 4
  quarter_circle_area / total_area

theorem probability_quarter_circle_is_pi_div_16 :
  probability_quarter_circle = π / 16 :=
by
  sorry

end probability_quarter_circle_is_pi_div_16_l521_521776


namespace max_k_value_ellipse_l521_521221

theorem max_k_value_ellipse (t : ℝ) (ht : t ≠ 0) :
  let C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 ^ 2) / 4 + p.2 ^ 2 = 1},
      M : ℝ × ℝ := (0, 1),
      N : ℝ × ℝ := (0, -1),
      T : ℝ × ℝ := (t, 2),
      TMN_area := abs t,
      k_max := (4 / 3) in
  t = 2 * sqrt 3 ∨ t = -2 * sqrt 3 →
  (let k := TMN_area / (abs t - abs (24 * t / (t ^ 2 + 36)) + abs (2 * t * (1 + 3 / (t ^ 2 + 36) + 1 / (t ^ 2 + 4)))) in
  k = k_max) := 
sorry

end max_k_value_ellipse_l521_521221


namespace loss_percentage_l521_521436

theorem loss_percentage (CP SP : ℝ) (hCP : CP = 1500) (hSP : SP = 1200) : 
  (CP - SP) / CP * 100 = 20 :=
by
  -- Proof would be provided here
  sorry

end loss_percentage_l521_521436


namespace translated_parabola_correct_l521_521999

-- Define translations
def translate_right (a : ℕ) (f : ℝ → ℝ) : ℝ → ℝ := λ x, f (x - a)
def translate_up (b : ℝ) (f : ℝ → ℝ) : ℝ → ℝ := λ x, (f x) + b

-- Original parabola
def original_parabola : ℝ → ℝ := λ x, x^2

-- Translated parabola
def translated_parabola : ℝ → ℝ := translate_up 1 (translate_right 2 original_parabola)

-- Prove the result
theorem translated_parabola_correct :
  ∀ x : ℝ, translated_parabola x = (x - 2)^2 + 1 :=
by
  intros x
  sorry

end translated_parabola_correct_l521_521999


namespace number_of_lines_that_satisfy_l521_521970

def line_passing_through (x1 y1 : ℝ) (k : ℝ) : (ℝ × ℝ) := (x1 - 1 / k, 0)
def line_passing_through_y (x1 y1 : ℝ) (k : ℝ) : (ℝ × ℝ) := (0, y1 - 2 * k)

def area_triangle (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ := 
  0.5 * abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂))

theorem number_of_lines_that_satisfy : 
  ∃ (k : ℝ), 
    let P := line_passing_through 2 1 k in 
    let Q := line_passing_through_y 2 1 k in 
    area_triangle 0 0 P.fst P.snd Q.fst Q.snd = 4 → 3 :=
by 
  -- Proof omitted
  sorry

end number_of_lines_that_satisfy_l521_521970


namespace triangle_tetrahedron_inequality_l521_521175

noncomputable theory

variables (G : Type) [finite_graph G] 
def f : G → ℕ -- Number of triangles
def g : G → ℕ -- Number of tetrahedra

theorem triangle_tetrahedron_inequality (G : G) : g(G)^3 ≤ (3 / 32) * f(G)^4 := 
sorry

end triangle_tetrahedron_inequality_l521_521175


namespace committee_meeting_l521_521097

theorem committee_meeting : 
  ∃ (A B : ℕ), 2 * A + B = 7 ∧ A + 2 * B = 11 ∧ A + B = 6 :=
by 
  sorry

end committee_meeting_l521_521097


namespace valid_ticket_buyer_lineups_l521_521651

theorem valid_ticket_buyer_lineups (n : ℕ) : 
  let total_paths := nat.choose (2 * n) n,
  let invalid_paths := nat.choose (2 * n) (n - 1)
  in total_paths - invalid_paths = nat.choose (2 * n) n / (n + 1) :=
by
  sorry

end valid_ticket_buyer_lineups_l521_521651


namespace impossible_to_form_three_similar_piles_l521_521355

-- Define the concept of similar sizes
def similar_sizes (a b : ℝ) : Prop := a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2

-- State the proof problem
theorem impossible_to_form_three_similar_piles (x : ℝ) (h : 0 < x) : ¬(∃ a b c : ℝ, a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes a c) :=
sorry

end impossible_to_form_three_similar_piles_l521_521355


namespace minimum_voters_for_tall_giraffe_l521_521285

-- Definitions and conditions
def voters : ℕ := 135
def districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3

-- Conditions encoded in the problem statement
def majority_precinct (votes: ℕ) : Prop := votes >= 2
def majority_district (precinct_wins: ℕ) : Prop := precinct_wins >= 5
def majority_winner (district_wins: ℕ) : Prop := district_wins >= 3

-- The problem states that the Tall giraffe won.
axiom tall_giraffe_won : ∃ district_wins : ℕ, 
  majority_winner district_wins ∧ 
  ∀ (d ∈ (finset.range districts)), ∃ precinct_wins : ℕ, 
  majority_district precinct_wins ∧ 
  ∀ (p ∈ (finset.range precincts_per_district)), ∃ votes : ℕ, 
  majority_precinct votes

-- Proof goal
theorem minimum_voters_for_tall_giraffe : ∃ (votes_for_tall : ℕ), votes_for_tall = 30 :=
by {
  -- proof of the theorem will go here
  sorry 
}

end minimum_voters_for_tall_giraffe_l521_521285


namespace min_voters_tall_giraffe_win_l521_521278

-- Definitions from the problem statement as conditions
def precinct_voters := 3
def precincts_per_district := 9
def districts := 5
def majority_precincts(p : ℕ) := p / 2 + 1  -- Minimum precincts won in a district 
def majority_districts(d : ℕ) := d / 2 + 1  -- Minimum districts won in the final

-- Condition: majority precincts to win a district
def precinct_votes_to_win := majority_precincts precinct_voters

-- Condition: majority districts to win the final
def district_wins_to_win_final := majority_districts districts

-- Minimum precincts the Tall giraffe needs to win overall
def total_precincts_to_win := district_wins_to_win_final * majority_precincts precincts_per_district

-- Proof that the minimum number of voters who could have voted for the Tall giraffe is 30
theorem min_voters_tall_giraffe_win :
  precinct_votes_to_win * total_precincts_to_win = 30 :=
sorry

end min_voters_tall_giraffe_win_l521_521278


namespace function_increasing_range_l521_521227

theorem function_increasing_range (a : ℝ) (f : ℝ → ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x : ℝ, f x = (a / (a^2 - 2)) * (a^x - a^(-x))) : 
  (∀ x y : ℝ, x < y → f x < f y) ↔ (0 < a ∧ a < 1 ∨ a > sqrt 2) :=
sorry

end function_increasing_range_l521_521227


namespace find_6th_result_l521_521497

theorem find_6th_result (results : Fin 11 → ℕ) (h_avg_all : (∑ i, results i) / 11 = 20)
    (h_avg_first_5 : (∑ i in Finset.range 5, results i) / 5 = 15)
    (h_avg_last_5 : (∑ i in Finset.Ico 6 11, results i) / 5 = 22) :
  results 5 = 35 :=
by
  sorry

end find_6th_result_l521_521497


namespace quad_iv_of_complex_solution_l521_521219

theorem quad_iv_of_complex_solution (z : ℂ) (h : (z + complex.i) * (1 - 2 * complex.i) = 2) :
    (complex.re z, complex.im z) = (2/5 : ℚ, -1/5 : ℚ) ∧ 
    (complex.re z > 0) ∧ (complex.im z < 0) :=
sorry

end quad_iv_of_complex_solution_l521_521219


namespace sum_m_n_l521_521768

open Real

noncomputable def f (x : ℝ) : ℝ := |log x / log 2|

theorem sum_m_n (m n : ℝ) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_mn : m < n) 
  (h_f_eq : f m = f n) (h_max_f : ∀ x : ℝ, m^2 ≤ x ∧ x ≤ n → f x ≤ 2) :
  m + n = 5 / 2 :=
sorry

end sum_m_n_l521_521768


namespace number_of_valid_pairings_l521_521020

-- Define a circle of 12 people
def people := Finset.range 12

-- Define a function to determine who each person knows
def knows (i : ℕ) : Finset ℕ :=
  (Finset.image (λ x, (i + x) % 12) (Finset.range 5 \ {2})) ∪
  (Finset.image (λ x, (i + 12 - x) % 12) (Finset.range 2 ∪ {(2 % 12)} \ {0}))

-- Define conditions to check if a set of pairs satisfies the given conditions
def valid_pairs (pairs : Finset (ℕ × ℕ)) : Prop :=
  pairs.card = 6 ∧ 
  (∀ (x y : ℕ × ℕ), x ∈ pairs → y ∈ pairs → (x.fst = y.fst ∨ x.fst = y.snd ∨ x.snd = y.fst ∨ x.snd = y.snd) → 
    (knows x.fst).mem x.snd ∧ (knows y.fst).mem y.snd)

-- Theorem to prove the number of valid pairings
theorem number_of_valid_pairings : ∃! (pairs : Finset (ℕ × ℕ)), valid_pairs pairs :=
by
  sorry

end number_of_valid_pairings_l521_521020


namespace sin_identity_l521_521421

theorem sin_identity :
  sin (7 * Real.pi / 30) + sin (11 * Real.pi / 30) = sin (Real.pi / 30) + sin (13 * Real.pi / 30) + 1 / 2 :=
by sorry

end sin_identity_l521_521421


namespace subsets_with_sum_2023060_l521_521698

open Finset

def B : Finset ℕ := range 2012

theorem subsets_with_sum_2023060 :
  (B.filter (λ s, s.val.sum = 2023060)).card = 4 := sorry

end subsets_with_sum_2023060_l521_521698


namespace volume_of_cylinder_proof_l521_521669

def side_length_of_cube := 20 -- in cm
def height_of_exposed_cylinder := 8 -- in cm
def base_area_of_cube := side_length_of_cube * side_length_of_cube -- in cm²
def base_area_ratio := 1 / 8
def base_area_of_cylinder := base_area_ratio * base_area_of_cube -- in cm²
def total_height_of_cylinder := side_length_of_cube -- in cm

theorem volume_of_cylinder_proof :
  let volume_of_cylinder :=
    base_area_of_cylinder * total_height_of_cylinder in
  volume_of_cylinder = 650 :=
by
  sorry

end volume_of_cylinder_proof_l521_521669


namespace translated_parabola_correct_l521_521998

-- Define translations
def translate_right (a : ℕ) (f : ℝ → ℝ) : ℝ → ℝ := λ x, f (x - a)
def translate_up (b : ℝ) (f : ℝ → ℝ) : ℝ → ℝ := λ x, (f x) + b

-- Original parabola
def original_parabola : ℝ → ℝ := λ x, x^2

-- Translated parabola
def translated_parabola : ℝ → ℝ := translate_up 1 (translate_right 2 original_parabola)

-- Prove the result
theorem translated_parabola_correct :
  ∀ x : ℝ, translated_parabola x = (x - 2)^2 + 1 :=
by
  intros x
  sorry

end translated_parabola_correct_l521_521998


namespace minimum_voters_for_tall_giraffe_l521_521282

-- Definitions and conditions
def voters : ℕ := 135
def districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3

-- Conditions encoded in the problem statement
def majority_precinct (votes: ℕ) : Prop := votes >= 2
def majority_district (precinct_wins: ℕ) : Prop := precinct_wins >= 5
def majority_winner (district_wins: ℕ) : Prop := district_wins >= 3

-- The problem states that the Tall giraffe won.
axiom tall_giraffe_won : ∃ district_wins : ℕ, 
  majority_winner district_wins ∧ 
  ∀ (d ∈ (finset.range districts)), ∃ precinct_wins : ℕ, 
  majority_district precinct_wins ∧ 
  ∀ (p ∈ (finset.range precincts_per_district)), ∃ votes : ℕ, 
  majority_precinct votes

-- Proof goal
theorem minimum_voters_for_tall_giraffe : ∃ (votes_for_tall : ℕ), votes_for_tall = 30 :=
by {
  -- proof of the theorem will go here
  sorry 
}

end minimum_voters_for_tall_giraffe_l521_521282


namespace S_is_line_l521_521242

def complex_plane : Type :=
  {z : ℂ // ∃ x y : ℝ, z = x + y*I}

def set_S (S : set complex_plane) : Prop :=
  S = {z | ∃ x y : ℝ, z.val = x + y * I ∧ (∃ c : ℝ, (2 + 3 * I) * z.val = c)}

theorem S_is_line (S : set complex_plane) (h : set_S S) : 
  ∃ m b : ℝ, ∀ z, z ∈ S → ∃ x y : ℝ, z.val = x + y * I ∧ x = m * y + b := 
sorry

end S_is_line_l521_521242


namespace distance_between_points_A_and_B_l521_521398

variables 
  (A B : Type) 
  (d : ℝ) -- distance between A and B in kilometers
  (vA vB : ℝ) -- initial speeds of Optimus Prime and Bumblebee respectively
  (vA' vB' : ℝ) -- speeds after transformation
  (time_meet : ℝ) -- time taken to meet

-- Condition 1: Their initial speed ratio is 4:3
def initial_speed_ratio : Prop := vA / vB = 4 / 3

-- Condition 2: After meeting, Optimus Prime's speed increases by 25%
def optimus_prime_speed_increase : Prop := vA' = vA * 1.25

-- Condition 3: After meeting, Bumblebee's speed increases by 30%
def bumblebee_speed_increase : Prop := vB' = vB * 1.3

-- Condition 4: When Optimus Prime reaches point B, Bumblebee is still 83 kilometers from point A
def distance_when_optimus_reaches_B : Prop := d / vA' * vB' = d - 83

-- Prove the distance d between points A and B
theorem distance_between_points_A_and_B 
  (h1 : initial_speed_ratio)
  (h2 : optimus_prime_speed_increase)
  (h3 : bumblebee_speed_increase)
  (h4 : distance_when_optimus_reaches_B) : 
  d = 350 :=
sorry

end distance_between_points_A_and_B_l521_521398


namespace solve_equation_l521_521982

theorem solve_equation : ∃ x : ℝ, 9 ^ (-x) - 2 * 3 ^ (1 - x) = 27 ∧ x = -2 :=
by
  -- We state that there exists an x such that our equation holds and x equals -2
  use -2
  split
  -- We provide the equation for the first part
  { sorry }
  -- We provide x = -2 for the second part
  { rfl }

end solve_equation_l521_521982


namespace number_of_valid_elements_l521_521322

def f (x : ℤ) : ℤ := x^2 + 5 * x + 6

def S : Finset ℤ := Finset.range 31

def valid_elements (n : ℤ) : Prop :=
  f(n) % 5 = 0 ∧ f(n) % 3 = 1

theorem number_of_valid_elements :
  Finset.card (S.filter valid_elements) = 4 :=
by
  sorry

end number_of_valid_elements_l521_521322


namespace problem_statement_false_adjacent_complementary_l521_521094

-- Definition of straight angle, supplementary angles, and complementary angles.
def is_straight_angle (θ : ℝ) : Prop := θ = 180
def are_supplementary (θ ψ : ℝ) : Prop := θ + ψ = 180
def are_complementary (θ ψ : ℝ) : Prop := θ + ψ = 90

-- Definition of adjacent angles (for completeness, though we don't use adjacency differently right now)
def are_adjacent (θ ψ : ℝ) : Prop := ∀ x, θ + x + ψ + x = θ + ψ -- Simplified

-- Additional conditions that could be true or false -- we need one of them to be false.
def false_statement_D (θ ψ : ℝ) : Prop :=
  are_complementary θ ψ → are_adjacent θ ψ

theorem problem_statement_false_adjacent_complementary :
  ∃ (θ ψ : ℝ), ¬ false_statement_D θ ψ :=
by
  sorry

end problem_statement_false_adjacent_complementary_l521_521094


namespace num_routes_P_to_Q_l521_521690

def City := String -- Simplified representation of a city as a string

def Road := (City, City) -- A road connects two cities

def roads : List Road := [
  ("P", "Q"), ("P", "R"), ("P", "S"), 
  ("Q", "T"), ("Q", "R"), ("R", "S"), ("S", "T")
]

def num_routes (start finish : City) (rs : List Road) : Nat :=
  -- Function to calculate routes goes here; implementation is not required
  sorry

theorem num_routes_P_to_Q : num_routes "P" "Q" roads = 6 := by
  sorry

end num_routes_P_to_Q_l521_521690


namespace cos_double_C_l521_521867

theorem cos_double_C (a b S : ℝ) (h₁ : a = 8) (h₂ : b = 5) (h₃ : S = 12) : 
  let C := real.arcsin (2 * S / (a * b)) in
  real.cos (2 * C) = 7 / 25 :=
by
  sorry

end cos_double_C_l521_521867


namespace sixth_largest_number_l521_521987

theorem sixth_largest_number (a b c d : ℕ) (h1 : a = 1) (h2 : b = 3) (h3 : c = 0) (h4 : d = 5) : 
  let nums := [a, b, c, d] in
  let permutations := list.permutations nums in
  let four_digit_numbers := permutations.map (λ l, l.head! * 1000 + l.nth! 1 * 100 + l.nth! 2 * 10 + l.nth! 3) in
  let sorted_numbers := four_digit_numbers.qsort (≥) in
  sorted_numbers.nth! 5 = 5013 :=
by {
  sorry
}

end sixth_largest_number_l521_521987


namespace function_satisfies_equation_for_all_integers_l521_521161

theorem function_satisfies_equation_for_all_integers :
  ∀ (a b : ℤ), 
  let f := (λ n : ℤ, -2 * n + 3) in
  f(a + b) + f(a^2 + b^2) = f(a) * f(b) + 2 := 
by
  intro a b
  let f := (λ n : ℤ, -2 * n + 3)
  sorry

end function_satisfies_equation_for_all_integers_l521_521161


namespace triangle_side_count_l521_521835

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end triangle_side_count_l521_521835


namespace geom_seq_triangle_ratio_range_l521_521209

variable {a b c : ℝ}

theorem geom_seq_triangle_ratio_range
  (triangle_ABC : Triangle a b c)
  (b_sq_eq_ac : b^2 = a * c)
  (pos_sides : 0 < a ∧ 0 < b ∧ 0 < c) :
  2 ≤ (b / a + a / b) ∧ (b / a + a / b) < Real.sqrt 5 := 
sorry

end geom_seq_triangle_ratio_range_l521_521209


namespace number_of_cow_herds_l521_521676

theorem number_of_cow_herds 
    (total_cows : ℕ) 
    (cows_per_herd : ℕ) 
    (h1 : total_cows = 320)
    (h2 : cows_per_herd = 40) : 
    total_cows / cows_per_herd = 8 :=
by
  sorry

end number_of_cow_herds_l521_521676


namespace not_possible_to_create_3_similar_piles_l521_521344

-- Formalize the definition of similar piles
def similar (a b : ℝ) : Prop := a <= sqrt 2 * b ∧ b <= sqrt 2 * a

-- State the main theorem
theorem not_possible_to_create_3_similar_piles (x₀ : ℝ) (h₀ : 0 < x₀) : 
  ¬ ∃ x₁ x₂ x₃ : ℝ, 
    x₁ + x₂ + x₃ = x₀ ∧ 
    similar x₁ x₂ ∧ 
    similar x₂ x₃ ∧ 
    similar x₃ x₁ :=
sorry

end not_possible_to_create_3_similar_piles_l521_521344


namespace decreasing_function_unique_in_interval_l521_521660

noncomputable def f1 (x : ℝ) : ℝ := x ^ -1
noncomputable def f2 (x : ℝ) : ℝ := x ^ (1 / 2)
noncomputable def f3 (x : ℝ) : ℝ := x ^ 2
noncomputable def f4 (x : ℝ) : ℝ := x ^ 3

theorem decreasing_function_unique_in_interval : 
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → f1 x1 > f1 x2) ∧
  (∀ x : ℝ, 0 < x → x^(-1) < x^2 ∧ x^(-1) < x^(1/2) ∧ x^(-1) < x^3) := 
by sorry

end decreasing_function_unique_in_interval_l521_521660


namespace part1_part2_l521_521902

noncomputable def A := {x : ℝ | x^2 - 3*x + 2 = 0}
noncomputable def B (a : ℝ) := {x : ℝ | x^2 + 2*(a + 1)*x + (a^2 - 5) = 0}

theorem part1 {a : ℝ} : (A ∩ B a = {2}) → (a = -1 ∨ a = -3) :=
sorry

theorem part2 {a : ℝ} :
  (λ U : set ℝ, U = set.univ) ∧ (A ∩ (compl (B a)) = A)
  → (a < -3 ∨ (-3 < a ∧ a < -1-Real.sqrt 3) ∨ (-1-Real.sqrt 3 < a ∧ a < -1) ∨ (-1 < a ∧ a < -1+Real.sqrt 3) ∨ a > -1+Real.sqrt 3) :=
sorry

end part1_part2_l521_521902


namespace students_on_bus_after_stops_l521_521024

-- Definitions
def initial_students : ℕ := 10
def first_stop_off : ℕ := 3
def first_stop_on : ℕ := 2
def second_stop_off : ℕ := 1
def second_stop_on : ℕ := 4
def third_stop_off : ℕ := 2
def third_stop_on : ℕ := 3

-- Theorem statement
theorem students_on_bus_after_stops :
  let after_first_stop := initial_students - first_stop_off + first_stop_on
  let after_second_stop := after_first_stop - second_stop_off + second_stop_on
  let after_third_stop := after_second_stop - third_stop_off + third_stop_on
  after_third_stop = 13 := 
by
  sorry

end students_on_bus_after_stops_l521_521024


namespace octadecagon_diagonals_l521_521716

def num_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octadecagon_diagonals : num_of_diagonals 18 = 135 := by
  sorry

end octadecagon_diagonals_l521_521716


namespace find_a_l521_521102

theorem find_a (a : ℝ) : -3 ∈ ({a - 3, 2 * a - 1, a^2 + 1} : set ℝ) → (a = 0 ∨ a = -1) :=
sorry

end find_a_l521_521102


namespace units_digit_7_pow_5_pow_3_l521_521680

theorem units_digit_7_pow_5_pow_3 : (7 ^ (5 ^ 3)) % 10 = 7 := by
  sorry

end units_digit_7_pow_5_pow_3_l521_521680


namespace tree_inequality_l521_521742

theorem tree_inequality (n : ℕ) (x : Fin n → ℝ) (edges : List (Fin n × Fin n)) (S : ℝ)
  (h_tree : is_tree edges)
  (h_n_ge_2 : 2 ≤ n)
  (h_S_def : S = (edges.map (λ e, (x e.fst) * (x e.snd))).sum) :
  sqrt (n - 1) * (Finset.univ.sum (λ i, (x i) ^ 2)) ≥ 2 * S :=
sorry

end tree_inequality_l521_521742


namespace seventeen_pow_2003_mod_29_l521_521485

theorem seventeen_pow_2003_mod_29 :
  (17 ^ 2003) % 29 = 26 := 
by {
  have h1 : (17 ^ 1) % 29 = 17 := by norm_num,
  have h2 : (17 ^ 2) % 29 = 24 := by norm_num,
  have h3 : (17 ^ 3) % 29 = 23 := by norm_num,
  have h4 : (17 ^ 4) % 29 = 16 := by norm_num,
  have h5 : (17 ^ 5) % 29 = 26 := by norm_num,
  have h6 : (17 ^ 6) % 29 = 1 := by norm_num,
  sorry
}

end seventeen_pow_2003_mod_29_l521_521485


namespace decagon_adjacent_probability_l521_521579

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521579


namespace special_sum_to_one_l521_521146

def is_special (x : ℝ) : Prop :=
  ∀ d : ℕ, ∃ (k : ℕ), (1 ≤ k ∧ k < 10) ∧ x * 10^d = k * (7 / 10^d) 

theorem special_sum_to_one : 
  ∃ (n : ℕ), 1 ≤ n ∧ (∀ (a : ℕ → ℝ), (∀ i, is_special (a i)) → ( ∑ i in finset.range n, a i ) = 1) :=
 sorry

end special_sum_to_one_l521_521146


namespace not_possible_three_piles_l521_521384

def similar (x y : ℝ) : Prop :=
  y ≤ (real.sqrt 2) * x ∧ x ≤ (real.sqrt 2) * y

theorem not_possible_three_piles (N : ℝ) : 
  ¬ ∃ a b c : ℝ, a + b + c = N ∧ similar a b ∧ similar b c ∧ similar a c := 
by
  sorry

end not_possible_three_piles_l521_521384


namespace prob_eq_l521_521254

-- Condition definitions
def fair_dice (n : ℕ) : Prop := 
  ∀ (d : ℕ), d < n → ∀ (face : ℕ), face < 6 → (face < 3 ∨ face < 6)

def balanced_faces (die : ℕ) : Prop := 
  ∀ (color : ℕ), color = 0 ∨ color = 1

def p (n : ℕ) : ℝ := 
  1 / 2 -- Placeholder for the actual probability definition

def q (n : ℕ) : ℝ := 
  1 / 2 -- Placeholder for the actual probability definition

-- Theorem to prove
theorem prob_eq (n : ℕ) (h1 : fair_dice n) (h2 : balanced_faces n) (h3 : 0 < n) :
  p n + (n - 1) * q n = n / 2 :=
by
  sorry

end prob_eq_l521_521254


namespace red_number_1992_is_2001_l521_521125

-- Define what it means for a number to be composite.
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Define what it means for a number to be colored red.
def is_colored_red (n : ℕ) : Prop := 
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

-- Define a function to find the nth red number.
def nth_red_number (n : ℕ) : ℕ := sorry

-- Prove that the 1992nd red number is 2001.
theorem red_number_1992_is_2001 : nth_red_number 1992 = 2001 := by
  sorry

end red_number_1992_is_2001_l521_521125


namespace smallest_cube_sum_l521_521428

theorem smallest_cube_sum : 
  ∃ (faces : Fin 6 → ℕ), (∀ (i j : Fin 6), (adjacent i j → (abs (faces i - faces j) > 1))) ∧ (∑ (i : Fin 6), faces i = 18) := 
by
  sorry

-- Definitions for adjacency relations on a cube's faces can be adjusted accordingly.
def adjacent : Fin 6 → Fin 6 → Prop
| ⟨0, _⟩, ⟨1, _⟩ => true -- example equivalences, real function should define all adjacency pairs correctly
| ⟨1, _⟩, ⟨0, _⟩ => true
| _ , _ => false -- all other pairs would be set according to cube adjacency

end smallest_cube_sum_l521_521428


namespace cindy_added_pens_l521_521492

-- Definitions based on conditions:
def initial_pens : ℕ := 20
def mike_pens : ℕ := 22
def sharon_pens : ℕ := 19
def final_pens : ℕ := 65

-- Intermediate calculations:
def pens_after_mike : ℕ := initial_pens + mike_pens
def pens_after_sharon : ℕ := pens_after_mike - sharon_pens

-- Proof statement:
theorem cindy_added_pens : pens_after_sharon + 42 = final_pens :=
by
  sorry

end cindy_added_pens_l521_521492


namespace equivalence_sufficient_necessary_l521_521665

-- Definitions for conditions
variables (A B : Prop)

-- Statement to prove
theorem equivalence_sufficient_necessary :
  (A → B) ↔ (¬B → ¬A) :=
by sorry

end equivalence_sufficient_necessary_l521_521665


namespace christopher_strolled_distance_l521_521689

variable speed : ℝ := 4
variable time : ℝ := 1.25

theorem christopher_strolled_distance : speed * time = 5 := by
  sorry

end christopher_strolled_distance_l521_521689


namespace tanya_work_time_l521_521949

theorem tanya_work_time (
    sakshi_work_time : ℝ := 20,
    tanya_efficiency : ℝ := 1.25
) : 
    let sakshi_rate : ℝ := 1 / sakshi_work_time in
    let tanya_rate : ℝ := tanya_efficiency * sakshi_rate in
    let tanya_work_time := 1 / tanya_rate in
    tanya_work_time = 16 :=
by
    let sakshi_rate : ℝ := 1 / sakshi_work_time
    let tanya_rate : ℝ := tanya_efficiency * sakshi_rate
    let tanya_time : ℝ := 1 / tanya_rate
    show tanya_time = 16
    sorry

end tanya_work_time_l521_521949


namespace g_is_periodic_l521_521725

noncomputable def f : ℤ → ℤ := sorry
noncomputable def g : ℤ → ℤ := sorry

-- Given conditions
def functional_equation (x y : ℤ) : Prop := f(g(x) + y) = g(f(y) + x)
def f_bounded : Prop := ∃ B : ℤ, ∀ x : ℤ, abs (f x) ≤ B

-- Main theorem to prove
theorem g_is_periodic (h1 : ∀ x y : ℤ, functional_equation x y) (h2 : f_bounded) : ∃ p : ℕ, p > 0 ∧ ∀ x : ℤ, g(x + p) = g(x) :=
sorry

end g_is_periodic_l521_521725


namespace part1_part2_l521_521901

noncomputable def A := {x : ℝ | x^2 - 3*x + 2 = 0}
noncomputable def B (a : ℝ) := {x : ℝ | x^2 + 2*(a + 1)*x + (a^2 - 5) = 0}

theorem part1 {a : ℝ} : (A ∩ B a = {2}) → (a = -1 ∨ a = -3) :=
sorry

theorem part2 {a : ℝ} :
  (λ U : set ℝ, U = set.univ) ∧ (A ∩ (compl (B a)) = A)
  → (a < -3 ∨ (-3 < a ∧ a < -1-Real.sqrt 3) ∨ (-1-Real.sqrt 3 < a ∧ a < -1) ∨ (-1 < a ∧ a < -1+Real.sqrt 3) ∨ a > -1+Real.sqrt 3) :=
sorry

end part1_part2_l521_521901


namespace ratio_change_factor_is_5_l521_521001

-- Definitions based on problem conditions
def original_bleach : ℕ := 4
def original_detergent : ℕ := 40
def original_water : ℕ := 100

-- Simplified original ratio
def original_bleach_ratio : ℕ := original_bleach / 4
def original_detergent_ratio : ℕ := original_detergent / 4
def original_water_ratio : ℕ := original_water / 4

-- Altered conditions
def altered_detergent : ℕ := 60
def altered_water : ℕ := 300

-- Simplified altered ratio of detergent to water
def altered_detergent_ratio : ℕ := altered_detergent / 60
def altered_water_ratio : ℕ := altered_water / 60

-- Proof that the ratio change factor is 5
theorem ratio_change_factor_is_5 : 
  (original_water_ratio / altered_water_ratio) = 5
  := by
    have original_detergent_ratio : ℕ := 10
    have original_water_ratio : ℕ := 25
    have altered_detergent_ratio : ℕ := 1
    have altered_water_ratio : ℕ := 5
    sorry

end ratio_change_factor_is_5_l521_521001


namespace probability_adjacent_vertices_in_decagon_l521_521573

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l521_521573


namespace jelly_beans_ratio_closest_to_31_l521_521463

theorem jelly_beans_ratio_closest_to_31 :
  let bagA_beans : ℕ := 26 in
  let bagB_beans : ℕ := 28 in
  let bagC_beans : ℕ := 30 in
  let yellowA_percent : ℚ := 0.50 in
  let yellowB_percent : ℚ := 0.25 in
  let yellowC_percent : ℚ := 0.20 in
  let yellow_beans_A := (yellowA_percent * bagA_beans : ℚ).to_nat in
  let yellow_beans_B := (yellowB_percent * bagB_beans : ℚ).to_nat in
  let yellow_beans_C := (yellowC_percent * bagC_beans : ℚ).to_nat in
  let total_yellow_beans := yellow_beans_A + yellow_beans_B + yellow_beans_C in
  let total_beans := bagA_beans + bagB_beans + bagC_beans in
  (total_yellow_beans / total_beans.to_rat * 100).to_int = 31 :=
begin
  sorry
end

end jelly_beans_ratio_closest_to_31_l521_521463


namespace decagon_adjacent_probability_l521_521586

theorem decagon_adjacent_probability : 
  ∀ (V : Finset ℕ), 
  V.card = 10 → 
  (∃ v₁ v₂ : ℕ, v₁ ≠ v₂ ∧ (v₁, v₂) ∈ decagon_adjacent_edges_set V) → 
  ∃ (prob : ℚ), prob = 2 / 9 := 
by
  sorry

end decagon_adjacent_probability_l521_521586


namespace triangle_side_count_l521_521828

theorem triangle_side_count (x : ℤ) (h1 : 3 < x) (h2 : x < 13) : ∃ n, n = 9 := by
  sorry

end triangle_side_count_l521_521828


namespace largest_integer_less_than_log_sum_l521_521480

theorem largest_integer_less_than_log_sum :
  let sum_logs := (finset.range 3010).sum (λ n, real.logb 3 ((n + 3) / (n + 2)))
  floor sum_logs < 7 :=
by
  sorry

end largest_integer_less_than_log_sum_l521_521480


namespace P_is_on_hyperbola_branch_l521_521331

noncomputable def P_trajectory_is_branch_of_hyperbola : Prop :=
  let F1 := (0 : ℝ, -1 : ℝ)
  let F2 := (0 : ℝ, 1 : ℝ)
  ∃ P : ℝ × ℝ, |F1 - P| - |F2 - P| = 1 ∧ -- condition
    ∀ Q : ℝ × ℝ, |F1 - Q| - |F2 - Q| = 1 → -- unique solution
    Q = P ∨ (trajectory_of Q is one branch of hyperbola) -- desired trajectory

theorem P_is_on_hyperbola_branch : P_trajectory_is_branch_of_hyperbola :=
by
  sorry

end P_is_on_hyperbola_branch_l521_521331


namespace inequalities_for_m_gt_n_l521_521943

open Real

theorem inequalities_for_m_gt_n (m n : ℕ) (hmn : m > n) : 
  (1 + 1 / (m : ℝ)) ^ m > (1 + 1 / (n : ℝ)) ^ n ∧ 
  (1 + 1 / (m : ℝ)) ^ (m + 1) < (1 + 1 / (n : ℝ)) ^ (n + 1) := 
by
  sorry

end inequalities_for_m_gt_n_l521_521943


namespace smallest_N_exists_l521_521012

theorem smallest_N_exists (
  a b c d : ℕ := list.perm [1, 2, 3, 4, 5] [gcd a b, gcd a c, gcd a d, gcd b c, gcd b d, gcd c d]
  (h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_N: N > 5) : 
  N = 14 :=
by sorry

end smallest_N_exists_l521_521012


namespace min_value_and_period_find_sides_l521_521766

-- Define the function f
def f (x : ℝ) : ℝ := (sqrt 3 / 2) * sin (2 * x) - (1 + cos (2 * x)) / 2 - 1 / 2

-- Define the minimum value and period
theorem min_value_and_period : (∃ x, f x = -2) ∧ (∀ x, f (x + π) = f x) :=
begin
  sorry
end

-- Define collinearity condition for vectors m and n
def collinear (A B : ℝ) : Prop := sin B = 2 * sin A

-- Define the side lengths conditions and solve for a and b
theorem find_sides (a b : ℝ) (A B C : ℝ)
  (h1 : c = sqrt 3)
  (h2 : f C = 0)
  (h3 : collinear A B)
  (h4 : b = 2 * a)
  (h5 : c^2 = a^2 + b^2 - 2 * a * b * cos (π / 3))
  : a = 1 ∧ b = 2 :=
begin
  sorry
end

end min_value_and_period_find_sides_l521_521766


namespace binom_7_2_minus_3_eq_18_l521_521692

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def binom (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem binom_7_2_minus_3_eq_18 :
  binom 7 2 - 3 = 18 :=
by
  sorry

end binom_7_2_minus_3_eq_18_l521_521692


namespace angle_between_AB1_and_BC1_l521_521288

-- Define the right triangular prism and the given conditions
structure RightTriangularPrism :=
  (A B C A1 B1 C1 : ℝ × ℝ × ℝ)
  (right_angle_prism : ∃ M : ℝ × ℝ × ℝ, (M ∈ plane B B1 B C))
  (AB_eq_sqrt2_BB1: dist A B = Real.sqrt 2 * dist B B1)

-- Define the theorem to prove
theorem angle_between_AB1_and_BC1 {P : RightTriangularPrism}
  (h: RightTriangularPrism) : angle (line_through P.A P.B1) (line_through P.B P.C1) = Real.pi / 2 :=
sorry

end angle_between_AB1_and_BC1_l521_521288


namespace expected_value_fair_dodecahedral_die_l521_521071

theorem expected_value_fair_dodecahedral_die : 
  let n := 12 in (1 / n) * (List.sum (List.range n).map (λ x => (x + 1))) = 6.5 :=
by
  let n := 12
  have h1 : (1 : ℚ) / n = 1 / 12 := by sorry
  have h2 : List.range n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] := by sorry
  have h3 : List.sum (List.range n).map (λ x => (x + 1)) = 78 := by sorry
  have h4 : (1 / 12) * 78 = 6.5 := by sorry
  exact h4

end expected_value_fair_dodecahedral_die_l521_521071


namespace sin_interval_diff_one_range_l521_521751

open Real

theorem sin_interval_diff_one_range (a b : ℝ) (h1 : a < b) 
(h2 : ∀ x ∈ set.Icc a b, sin x ≤ 1) 
(h3 : ∀ x ∈ set.Icc a b, -1 ≤ sin x) 
(h4 : ∃ x y ∈ set.Icc a b, sin x = 1 ∧ sin y = -1) :
  b - a ∈ set.Icc (π / 3) π :=
sorry

end sin_interval_diff_one_range_l521_521751


namespace theta_eq_zero_suff_but_not_nec_for_sin_theta_eq_zero_l521_521504

theorem theta_eq_zero_suff_but_not_nec_for_sin_theta_eq_zero :
  (∀ θ : ℝ, θ = 0 → sin θ = 0) ∧ ¬(∀ θ : ℝ, sin θ = 0 → θ = 0) :=
by
  sorry

end theta_eq_zero_suff_but_not_nec_for_sin_theta_eq_zero_l521_521504


namespace find_angle_between_vectors_l521_521207

-- Definitions for vectors and conditions
variables (a b : EuclideanSpace ℝ 3)
variable (theta : ℝ)

-- Given conditions
def magnitude_a := ∥a∥ = 2
def magnitude_b := ∥b∥ = 4
def orthogonal_cond := a ⬝ (b - a) = 0

-- Main statement
theorem find_angle_between_vectors
  (a b : EuclideanSpace ℝ 3)
  (magnitude_a : ∥a∥ = 2)
  (magnitude_b : ∥b∥ = 4)
  (orthogonal_cond : a ⬝ (b - a) = 0) :
  ∃ theta : ℝ, 0 ≤ theta ∧ theta ≤ π ∧ theta = π / 3 :=
sorry

end find_angle_between_vectors_l521_521207


namespace possible_integer_side_lengths_l521_521821

theorem possible_integer_side_lengths (a b : ℕ) (h1 : a = 8) (h2 : b = 5) :
  {x : ℕ | 3 < x ∧ x < 13}.finite ∧ {x : ℕ | 3 < x ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521821


namespace number_of_sides_possibilities_l521_521815

theorem number_of_sides_possibilities (a b : ℕ) (h₁ : a = 8) (h₂ : b = 5) : 
  { x : ℕ // 3 < x ∧ x < 13 }.to_list.length = 9 :=
by
  sorry

end number_of_sides_possibilities_l521_521815


namespace smallest_possible_value_of_N_l521_521015

-- Conditions definition:
variable (a b c d : ℕ)
variable (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
variable (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
variable (gcd_ab : Int.gcd a b = 1)
variable (gcd_ac : Int.gcd a c = 2)
variable (gcd_ad : Int.gcd a d = 4)
variable (gcd_bc : Int.gcd b c = 5)
variable (gcd_bd : Int.gcd b d = 3)
variable (gcd_cd : Int.gcd c d = N)
variable (hN : N > 5)

-- Statement to prove:
theorem smallest_possible_value_of_N : N = 14 := sorry

end smallest_possible_value_of_N_l521_521015


namespace original_sales_tax_percentage_l521_521004

theorem original_sales_tax_percentage (rate_reduced_tax diff_tax item_cost : ℝ)
  (h_rate_reduced_tax : rate_reduced_tax = 0.04)
  (h_diff_tax : diff_tax = 10)
  (h_item_cost : item_cost = 1000) :
  let original_tax_percent := 5
  in (item_cost * (original_tax_percent / 100) - item_cost * rate_reduced_tax = diff_tax) :=
by
  sorry

end original_sales_tax_percentage_l521_521004


namespace canvas_bag_lower_carbon_solution_l521_521934

def canvas_bag_emission := 600 -- pounds of CO2
def plastic_bag_emission := 4 -- ounces of CO2 per bag
def bags_per_trip := 8 
def ounce_to_pound := 16 -- 16 ounces in a pound
def co2_trip := (plastic_bag_emission * bags_per_trip) / ounce_to_pound -- CO2 emission in pounds per trip

theorem canvas_bag_lower_carbon_solution : 
  co2_trip * 300 >= canvas_bag_emission :=
by
  unfold canvas_bag_emission plastic_bag_emission bags_per_trip ounce_to_pound co2_trip 
  sorry

end canvas_bag_lower_carbon_solution_l521_521934


namespace probability_adjacent_vertices_in_decagon_l521_521575

-- Define the problem space
def decagon_vertices : ℕ := 10

def choose_two_vertices (n : ℕ) : ℕ :=
  n * (n - 1) / 2

def adjacent_outcomes (n : ℕ) : ℕ :=
  n  -- each vertex has 1 pair of adjacent vertices when counted correctly

theorem probability_adjacent_vertices_in_decagon :
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 :=
by
  let total_outcomes := choose_two_vertices decagon_vertices
  let favorable_outcomes := adjacent_outcomes decagon_vertices
  have h_total : total_outcomes = 45 := by sorry
  have h_favorable : favorable_outcomes = 10 := by sorry
  have h_fraction : (favorable_outcomes : ℚ) / total_outcomes = 2 / 9 := by sorry
  exact h_fraction

end probability_adjacent_vertices_in_decagon_l521_521575


namespace floor_sum_equals_floor_sum_l521_521969
noncomputable theory
open Int

def floor_sum (a b : ℕ) : ℕ :=
  (Finset.range b).sum (λ k => k * a / b)

def floor_sum_compare (a b : ℕ) (coprime_ab : Nat.gcd a b = 1) (a_gt_b : a > b) : Prop :=
  floor_sum a b = floor_sum b a

theorem floor_sum_equals_floor_sum (a b : ℕ) (coprime_ab : Nat.gcd a b = 1) (a_gt_b : a > b) : 
  floor_sum_compare a b coprime_ab a_gt_b := 
by sorry

end floor_sum_equals_floor_sum_l521_521969


namespace feuerbach_centers_parallelogram_or_line_l521_521437

theorem feuerbach_centers_parallelogram_or_line {A B C D E F₁ F₂ F₃ F₄ : Type}
  (hE : between A C E ∧ between B D E)
  (h₁: feuerbach_circle (triangle.mk A B E) F₁)
  (h₂: feuerbach_circle (triangle.mk B C E) F₂)
  (h₃: feuerbach_circle (triangle.mk C D E) F₃)
  (h₄: feuerbach_circle (triangle.mk D A E) F₄)
  : (parallelogram F₁ F₂ F₃ F₄ ∨ collinear F₁ F₂ F₃ F₄) :=
sorry

end feuerbach_centers_parallelogram_or_line_l521_521437


namespace four_thirds_of_product_eq_25_div_2_l521_521164

noncomputable def a : ℚ := 15 / 4
noncomputable def b : ℚ := 5 / 2
noncomputable def c : ℚ := 4 / 3
noncomputable def d : ℚ := a * b
noncomputable def e : ℚ := c * d

theorem four_thirds_of_product_eq_25_div_2 : e = 25 / 2 := 
sorry

end four_thirds_of_product_eq_25_div_2_l521_521164


namespace tn_range_l521_521743

noncomputable def a (n : ℕ) : ℚ :=
  (2 * n - 1) / 10

noncomputable def b (n : ℕ) : ℚ :=
  2^(n - 1)

noncomputable def c (n : ℕ) : ℚ :=
  (1 + a n) / (4 * b n)

noncomputable def T (n : ℕ) : ℚ :=
  (1 / 10) * (2 - (n + 2) / (2^n)) + (9 / 20) * (2 - 1 / (2^(n-1)))

theorem tn_range (n : ℕ) : (101 / 400 : ℚ) ≤ T n ∧ T n < (103 / 200 : ℚ) :=
sorry

end tn_range_l521_521743


namespace decagon_adjacent_probability_l521_521534

namespace DecagonProblem

-- Definition: Decagon and adjacent probabilities
def decagon_vertices : Finset (Fin 10) := Finset.univ

def adjacent_vertices (v : Fin 10) : Finset (Fin 10) :=
  {if v = 0 then 1 else if v = 9 then 8 else v + 1, if v = 0 then 9 else if v = 9 then 0 else v - 1}

-- Probability calculation definition
noncomputable def probability_adjacent : ℚ :=
  (2 : ℚ) / (9 : ℚ)

-- Lean statement asserting the proof problem
theorem decagon_adjacent_probability :
  ∀ (v₁ v₂ : Fin 10), v₁ ≠ v₂ → (v₂ ∈ adjacent_vertices v₁ ↔ real.rat_cast (1 / 5) (= probability_adjacent)) :=
  sorry

end DecagonProblem

end decagon_adjacent_probability_l521_521534


namespace expected_value_dodecahedral_die_is_6_5_l521_521061

noncomputable def expected_value_dodecahedral_die : ℝ :=
  let outcomes := finset.range 12 in
  1 / 12 * (finset.sum outcomes (λ i, (i + 1 : ℝ)))

theorem expected_value_dodecahedral_die_is_6_5 :
  expected_value_dodecahedral_die = 6.5 :=
by
  sorry

end expected_value_dodecahedral_die_is_6_5_l521_521061


namespace angle_CTX_eq_2Y_l521_521668

-- Let ABC be a triangle, with points C, A, and B.
variables {A B C E F T I X Y : Point}
-- Defining point relationships and conditions
variable (h_circle : ∃ γ : Circle, γ.TangentAt A∧ γ.TangentAt B∧ γ.TangentAt T)
-- The inscribed circle properties
variable (h_tangents : TangentTo (circumcircle(A, B, C)) T)
variable (h_tangentCA : TangentTo (lineSegment(C, A)) E)
variable (h_tangentCB : TangentTo (lineSegment(C, B)) F)
-- Midpoint I of EF, and projections
variable (h_midpoint : Midpoint I E F)
variable (h_projection : Projection I A B X)
variable (h_intersection: Intersection (line(A, B)) (line(E, F)) =Y)

theorem angle_CTX_eq_2Y :
  ∀ (A B C E F T I X Y : Point), 
    (∃ γ : Circle, γ.TangentAt A∧ γ.TangentAt B∧ γ.TangentAt T) →
    TangentTo (circumcircle(A, B, C)) T →
    TangentTo (lineSegment(C, A)) E →
    TangentTo (lineSegment(C, B)) F →
    Midpoint I E F →
    Projection I A B X →
    Intersection (line(A, B)) (line(E, F)) = Y →
    angle C T X = 2 * angle Y :=
by
  sorry

end angle_CTX_eq_2Y_l521_521668


namespace possible_integer_side_lengths_l521_521797

theorem possible_integer_side_lengths : 
  {x : ℤ | x > 3 ∧ x < 13}.toFinset.card = 9 :=
by
  sorry

end possible_integer_side_lengths_l521_521797
