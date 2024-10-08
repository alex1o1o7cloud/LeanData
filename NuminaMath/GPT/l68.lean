import Mathlib

namespace shaded_region_area_l68_68339

open Real

noncomputable def area_of_shaded_region (r : ℝ) (s : ℝ) (d : ℝ) : ℝ := 
  (1/4) * π * r^2 + (1/2) * (d - s)^2

theorem shaded_region_area :
  let r := 3
  let s := 2
  let d := sqrt 5
  area_of_shaded_region r s d = 9 * π / 4 + (9 - 4 * sqrt 5) / 2 :=
by
  sorry

end shaded_region_area_l68_68339


namespace weight_of_b_l68_68284

theorem weight_of_b (A B C : ℕ) 
  (h1 : A + B + C = 129) 
  (h2 : A + B = 80) 
  (h3 : B + C = 86) : 
  B = 37 := 
by 
  sorry

end weight_of_b_l68_68284


namespace initial_packs_l68_68990

def num_invitations_per_pack := 3
def num_friends := 9
def extra_invitations := 3
def total_invitations := num_friends + extra_invitations

theorem initial_packs (h : total_invitations = 12) : (total_invitations / num_invitations_per_pack) = 4 :=
by
  have h1 : total_invitations = 12 := by exact h
  have h2 : num_invitations_per_pack = 3 := by exact rfl
  have H_pack : total_invitations / num_invitations_per_pack = 4 := by sorry
  exact H_pack

end initial_packs_l68_68990


namespace solve_eq1_solve_eq2_l68_68993

theorem solve_eq1 {x : ℝ} : 2 * x^2 - 1 = 49 ↔ x = 5 ∨ x = -5 := 
  sorry

theorem solve_eq2 {x : ℝ} : (x + 3)^3 = 64 ↔ x = 1 := 
  sorry

end solve_eq1_solve_eq2_l68_68993


namespace average_of_first_5_multiples_of_5_l68_68172

theorem average_of_first_5_multiples_of_5 : 
  (5 + 10 + 15 + 20 + 25) / 5 = 15 :=
by
  sorry

end average_of_first_5_multiples_of_5_l68_68172


namespace set_intersection_l68_68954

def A : Set ℝ := {x | x < 5}
def B : Set ℝ := {x | x < 2}
def B_complement : Set ℝ := {x | x ≥ 2}

theorem set_intersection :
  A ∩ B_complement = {x | 2 ≤ x ∧ x < 5} :=
by 
  sorry

end set_intersection_l68_68954


namespace reduce_to_one_l68_68842

theorem reduce_to_one (n : ℕ) : ∃ k, (k = 1) :=
by
  sorry

end reduce_to_one_l68_68842


namespace S_eq_T_l68_68693

-- Define the sets S and T
def S : Set ℤ := {x | ∃ n : ℕ, x = 3 * n + 1}
def T : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 2}

-- Prove that S = T
theorem S_eq_T : S = T := 
by {
  sorry
}

end S_eq_T_l68_68693


namespace inequality_proof_l68_68933

noncomputable def a := (1.01: ℝ) ^ (0.5: ℝ)
noncomputable def b := (1.01: ℝ) ^ (0.6: ℝ)
noncomputable def c := (0.6: ℝ) ^ (0.5: ℝ)

theorem inequality_proof : b > a ∧ a > c := 
by
  sorry

end inequality_proof_l68_68933


namespace symmetric_to_y_axis_circle_l68_68262

open Real

-- Definition of the original circle's equation
def original_circle (x y : ℝ) : Prop := x^2 - 2 * x + y^2 = 3

-- Definition of the symmetric circle's equation with respect to the y-axis
def symmetric_circle (x y : ℝ) : Prop := x^2 + 2 * x + y^2 = 3

-- Theorem stating that the symmetric circle has the given equation
theorem symmetric_to_y_axis_circle (x y : ℝ) : 
  (symmetric_circle x y) ↔ (original_circle ((-x) - 2) y) :=
sorry

end symmetric_to_y_axis_circle_l68_68262


namespace equivalent_region_l68_68984

def satisfies_conditions (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 2 ∧ -1 ≤ x / (x + y) ∧ x / (x + y) ≤ 1

def region (x y : ℝ) : Prop :=
  y ≥ 0 ∧ y ≥ -2*x ∧ x^2 + y^2 ≤ 2

theorem equivalent_region (x y : ℝ) :
  satisfies_conditions x y = region x y := 
sorry

end equivalent_region_l68_68984


namespace adjacent_complementary_is_complementary_l68_68909

/-- Two angles are complementary if their sum is 90 degrees. -/
def complementary (α β : ℝ) : Prop :=
  α + β = 90

/-- Two angles are adjacent complementary if they are complementary and adjacent. -/
def adjacent_complementary (α β : ℝ) : Prop :=
  complementary α β ∧ α > 0 ∧ β > 0

/-- Prove that adjacent complementary angles are complementary. -/
theorem adjacent_complementary_is_complementary (α β : ℝ) : adjacent_complementary α β → complementary α β :=
by
  sorry

end adjacent_complementary_is_complementary_l68_68909


namespace molecular_weight_correct_l68_68534

-- Definition of atomic weights for the elements
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Number of atoms in Ascorbic acid (C6H8O6)
def count_C : ℕ := 6
def count_H : ℕ := 8
def count_O : ℕ := 6

-- Calculation of molecular weight
def molecular_weight_ascorbic_acid : ℝ :=
  (count_C * atomic_weight_C) +
  (count_H * atomic_weight_H) +
  (count_O * atomic_weight_O)

theorem molecular_weight_correct :
  molecular_weight_ascorbic_acid = 176.124 :=
by sorry


end molecular_weight_correct_l68_68534


namespace initial_black_water_bottles_l68_68246

-- Define the conditions
variables (red black blue taken left total : ℕ)
variables (hred : red = 2) (hblue : blue = 4) (htaken : taken = 5) (hleft : left = 4)

-- State the theorem with the correct answer given the conditions
theorem initial_black_water_bottles : (red + black + blue = taken + left) → black = 3 :=
by
  intros htotal
  rw [hred, hblue, htaken, hleft] at htotal
  sorry

end initial_black_water_bottles_l68_68246


namespace find_cows_l68_68070

-- Define the number of ducks (D) and cows (C)
variables (D C : ℕ)

-- Define the main condition given in the problem
def legs_eq_condition (D C : ℕ) : Prop :=
  2 * D + 4 * C = 2 * (D + C) + 36

-- State the theorem we wish to prove
theorem find_cows (D C : ℕ) (h : legs_eq_condition D C) : C = 18 :=
sorry

end find_cows_l68_68070


namespace negation_even_l68_68129

open Nat

theorem negation_even (x : ℕ) (h : 0 < x) :
  (∀ x : ℕ, 0 < x → Even x) ↔ ¬ (∃ x : ℕ, 0 < x ∧ Odd x) :=
by
  sorry

end negation_even_l68_68129


namespace divisors_of_64n4_l68_68802

theorem divisors_of_64n4 (n : ℕ) (hn : 0 < n) (hdiv : ∃ d, d = (120 * n^3) ∧ d.divisors.card = 120) : (64 * n^4).divisors.card = 375 := 
by 
  sorry

end divisors_of_64n4_l68_68802


namespace mario_pizza_area_l68_68016

theorem mario_pizza_area
  (pizza_area : ℝ)
  (cut_distance : ℝ)
  (largest_piece : ℝ)
  (smallest_piece : ℝ)
  (total_pieces : ℕ)
  (pieces_mario_gets_area : ℝ) :
  pizza_area = 4 →
  cut_distance = 0.5 →
  total_pieces = 4 →
  pieces_mario_gets_area = (pizza_area - (largest_piece + smallest_piece)) / 2 →
  pieces_mario_gets_area = 1.5 :=
sorry

end mario_pizza_area_l68_68016


namespace total_pens_l68_68116

theorem total_pens (black_pens blue_pens : ℕ) (h1 : black_pens = 4) (h2 : blue_pens = 4) : black_pens + blue_pens = 8 :=
by
  sorry

end total_pens_l68_68116


namespace first_new_player_weight_l68_68449

theorem first_new_player_weight (x : ℝ) :
  (7 * 103) + x + 60 = 9 * 99 → 
  x = 110 := by
  sorry

end first_new_player_weight_l68_68449


namespace valid_triangle_side_l68_68833

theorem valid_triangle_side (x : ℕ) (h_pos : 0 < x) (h1 : x + 6 > 15) (h2 : 21 > x) :
  10 ≤ x ∧ x ≤ 20 :=
by {
  sorry
}

end valid_triangle_side_l68_68833


namespace simplify_expression_l68_68409

theorem simplify_expression (a : ℝ) (h₀ : a ≥ 0) (h₁ : a ≠ 1) (h₂ : a ≠ 1 + Real.sqrt 2) (h₃ : a ≠ 1 - Real.sqrt 2) :
  (1 + 2 * a ^ (1 / 4) - a ^ (1 / 2)) / (1 - a + 4 * a ^ (3 / 4) - 4 * a ^ (1 / 2)) +
  (a ^ (1 / 4) - 2) / (a ^ (1 / 4) - 1) ^ 2 = 1 / (a ^ (1 / 4) - 1) :=
by
  sorry

end simplify_expression_l68_68409


namespace mowing_ratio_is_sqrt2_l68_68212

noncomputable def mowing_ratio (s w : ℝ) (hw_half_area : w * (s * Real.sqrt 2) = s^2) : ℝ :=
  s / w

theorem mowing_ratio_is_sqrt2 (s w : ℝ) (hs_positive : s > 0) (hw_positive : w > 0)
  (hw_half_area : w * (s * Real.sqrt 2) = s^2) : mowing_ratio s w hw_half_area = Real.sqrt 2 :=
by
  sorry

end mowing_ratio_is_sqrt2_l68_68212


namespace intersection_correct_l68_68308

def A : Set ℝ := { x | 0 < x ∧ x < 3 }
def B : Set ℝ := { x | x^2 ≥ 4 }
def intersection : Set ℝ := { x | 2 ≤ x ∧ x < 3 }

theorem intersection_correct : A ∩ B = intersection := by
  sorry

end intersection_correct_l68_68308


namespace pow_1999_mod_26_l68_68148

theorem pow_1999_mod_26 (n : ℕ) (h1 : 17^1 % 26 = 17)
  (h2 : 17^2 % 26 = 17) (h3 : 17^3 % 26 = 17) : 17^1999 % 26 = 17 := by
  sorry

end pow_1999_mod_26_l68_68148


namespace meaningful_fraction_condition_l68_68455

theorem meaningful_fraction_condition (x : ℝ) : x - 2 ≠ 0 ↔ x ≠ 2 := 
by 
  sorry

end meaningful_fraction_condition_l68_68455


namespace sam_original_seashells_count_l68_68950

-- Definitions representing the conditions
def seashells_given_to_joan : ℕ := 18
def seashells_sam_has_now : ℕ := 17

-- The question and the answer translated to a proof problem
theorem sam_original_seashells_count :
  seashells_given_to_joan + seashells_sam_has_now = 35 :=
by
  sorry

end sam_original_seashells_count_l68_68950


namespace find_a_b_l68_68397

def satisfies_digit_conditions (n a b : ℕ) : Prop :=
  n = 2000 + 100 * a + 90 + b ∧
  n / 1000 % 10 = 2 ∧
  n / 100 % 10 = a ∧
  n / 10 % 10 = 9 ∧
  n % 10 = b

theorem find_a_b : ∃ (a b : ℕ), 2^a * 9^b = 2000 + 100*a + 90 + b ∧ satisfies_digit_conditions (2^a * 9^b) a b :=
by
  sorry

end find_a_b_l68_68397


namespace triangle_ctg_inequality_l68_68770

noncomputable def ctg (x : Real) := Real.cos x / Real.sin x

theorem triangle_ctg_inequality (α β γ : Real) (h : α + β + γ = Real.pi) :
  ctg α ^ 2 + ctg β ^ 2 + ctg γ ^ 2 ≥ 1 :=
sorry

end triangle_ctg_inequality_l68_68770


namespace intersection_correct_l68_68210

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | x < 2}

theorem intersection_correct : P ∩ Q = {1} :=
by sorry

end intersection_correct_l68_68210


namespace largest_divisible_l68_68642

theorem largest_divisible (n : ℕ) (h1 : n > 0) (h2 : (n^3 + 200) % (n - 8) = 0) : n = 5376 :=
by
  sorry

end largest_divisible_l68_68642


namespace correct_number_of_statements_l68_68959

-- Define the conditions as invalidity of the given statements
def statement_1_invalid : Prop := ¬ (true) -- INPUT a,b,c should use commas
def statement_2_invalid : Prop := ¬ (true) -- INPUT x=, 3 correct format
def statement_3_invalid : Prop := ¬ (true) -- 3=B , left side should be a variable name
def statement_4_invalid : Prop := ¬ (true) -- A=B=2, continuous assignment not allowed

-- Combine conditions
def all_statements_invalid : Prop := statement_1_invalid ∧ statement_2_invalid ∧ statement_3_invalid ∧ statement_4_invalid

-- State the theorem to prove
theorem correct_number_of_statements : all_statements_invalid → 0 = 0 := 
by sorry

end correct_number_of_statements_l68_68959


namespace minimum_notes_to_determine_prize_location_l68_68253

/--
There are 100 boxes, numbered from 1 to 100. A prize is hidden in one of the boxes, 
and the host knows its location. The viewer can send the host a batch of notes 
with questions that require a "yes" or "no" answer. The host shuffles the notes 
in the batch and, without announcing the questions aloud, honestly answers 
all of them. Prove that the minimum number of notes that need to be sent to 
definitely determine where the prize is located is 99.
-/
theorem minimum_notes_to_determine_prize_location : 
  ∀ (boxes : Fin 100 → Prop) (prize_location : ∃ i : Fin 100, boxes i) 
    (batch_size : Nat), 
  (batch_size + 1) ≥ 100 → batch_size = 99 :=
by
  sorry

end minimum_notes_to_determine_prize_location_l68_68253


namespace wally_not_all_numbers_l68_68724

def next_wally_number (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n / 2
  else
    (n + 1001) / 2

def eventually_print(n: ℕ) : Prop :=
  ∃ k: ℕ, (next_wally_number^[k]) 1 = n

theorem wally_not_all_numbers :
  ¬ ∀ n, n ≤ 100 → eventually_print n :=
by
  sorry

end wally_not_all_numbers_l68_68724


namespace initial_average_daily_production_l68_68970

variable (A : ℝ) -- Initial average daily production
variable (n : ℕ) -- Number of days

theorem initial_average_daily_production (n_eq_5 : n = 5) (new_production_eq_90 : 90 = 90) 
  (new_average_eq_65 : (5 * A + 90) / 6 = 65) : A = 60 :=
by
  sorry

end initial_average_daily_production_l68_68970


namespace total_carrots_grown_l68_68151

theorem total_carrots_grown :
  let Sandy := 6.5
  let Sam := 3.25
  let Sophie := 2.75 * Sam
  let Sara := (Sandy + Sam + Sophie) - 7.5
  Sandy + Sam + Sophie + Sara = 29.875 :=
by
  sorry

end total_carrots_grown_l68_68151


namespace range_u_inequality_le_range_k_squared_l68_68672

def D (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = k}

theorem range_u (k : ℝ) (hk : k > 0) :
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k → 0 < x1 * x2 ∧ x1 * x2 ≤ k^2 / 4 :=
sorry

theorem inequality_le (k : ℝ) (hk : k ≥ 1) :
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k →
  (1 / x1 - x1) * (1 / x2 - x2) ≤ (k / 2 - 2 / k)^2 :=
sorry

theorem range_k_squared (k : ℝ) :
  (0 < k^2 ∧ k^2 ≤ 4 * Real.sqrt 5 - 8) ↔
  ∀ (x1 x2 : ℝ), (x1, x2) ∈ D k →
  (1 / x1 - x1) * (1 / x2 - x2) ≥ (k / 2 - 2 / k)^2 :=
sorry

end range_u_inequality_le_range_k_squared_l68_68672


namespace calculate_expr_l68_68784

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end calculate_expr_l68_68784


namespace total_biking_distance_l68_68241

-- Define the problem conditions 
def shelves := 4
def books_per_shelf := 400
def one_way_distance := shelves * books_per_shelf

-- Prove that the total distance for a round trip is 3200 miles
theorem total_biking_distance : 2 * one_way_distance = 3200 :=
by sorry

end total_biking_distance_l68_68241


namespace equation_of_l3_line_l1_through_fixed_point_existence_of_T_l68_68417

-- Question 1: The equation of the line \( l_{3} \)
theorem equation_of_l3 
  (F : ℝ × ℝ) 
  (H_focus : F = (2, 0))
  (k : ℝ) 
  (H_slope : k = 1) : 
  (∀ x y : ℝ, y = k * x + -2 ↔ y = x - 2) := 
sorry

-- Question 2: Line \( l_{1} \) passes through the fixed point (8, 0)
theorem line_l1_through_fixed_point 
  (k m1 : ℝ)
  (H_km1 : k * m1 ≠ 0)
  (H_m1lt : m1 < -t)
  (H_condition : ∃ x y : ℝ, y = k * x + m1 ∧ x^2 + (8/k) * x + (8 * m1 / k) = 0 ∧ ((x, y) = A1 ∨ (x, y) = B1))
  (H_dot_product : (x1 - 0)*(x2 - 0) + (y1 - 0)*(y2 - 0) = 0) : 
  ∀ P : ℝ × ℝ, P = (8, 0) := 
sorry

-- Question 3: Existence of point T such that S_i and d_i form geometric sequences
theorem existence_of_T
  (k : ℝ)
  (H_k : k = 1)
  (m1 m2 m3 : ℝ)
  (H_m_ordered : m1 < m2 ∧ m2 < m3 ∧ m3 < -t)
  (t : ℝ)
  (S1 S2 S3 d1 d2 d3 : ℝ)
  (H_S_geom_seq : S2^2 = S1 * S3)
  (H_d_geom_seq : d2^2 = d1 * d3)
  : ∃ t : ℝ, t = -2 :=
sorry

end equation_of_l3_line_l1_through_fixed_point_existence_of_T_l68_68417


namespace route_C_is_quicker_l68_68287

/-
  Define the conditions based on the problem:
  - Route C: 8 miles at 40 mph.
  - Route D: 5 miles at 35 mph and 2 miles at 25 mph with an additional 3 minutes stop.
-/

def time_route_C : ℚ := (8 : ℚ) / (40 : ℚ) * 60  -- in minutes

def time_route_D : ℚ := ((5 : ℚ) / (35 : ℚ) * 60) + ((2 : ℚ) / (25 : ℚ) * 60) + 3  -- in minutes

def time_difference : ℚ := time_route_D - time_route_C  -- difference in minutes

theorem route_C_is_quicker : time_difference = 4.37 := 
by 
  sorry

end route_C_is_quicker_l68_68287


namespace son_l68_68587

theorem son's_age (S M : ℕ) 
  (h1 : M = S + 35)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 33 := 
by
  sorry

end son_l68_68587


namespace max_value_expression_l68_68810

theorem max_value_expression (a b c : ℝ) 
  (ha : 300 ≤ a ∧ a ≤ 500) 
  (hb : 500 ≤ b ∧ b ≤ 1500) 
  (hc : c = 100) : 
  (∃ M, M = 8 ∧ ∀ x, x = (b + c) / (a - c) → x ≤ M) := 
sorry

end max_value_expression_l68_68810


namespace blonde_hair_count_l68_68447

theorem blonde_hair_count (total_people : ℕ) (percentage_blonde : ℕ) (h_total : total_people = 600) (h_percentage : percentage_blonde = 30) : 
  (percentage_blonde * total_people / 100) = 180 :=
by
  -- Conditions from the problem
  have h1 : total_people = 600 := h_total
  have h2 : percentage_blonde = 30 := h_percentage
  -- Start the proof
  sorry

end blonde_hair_count_l68_68447


namespace shortest_side_of_similar_triangle_l68_68623

theorem shortest_side_of_similar_triangle (a b : ℕ) (c : ℝ) 
  (h1 : a = 24) (h2 : c = 25) (h3 : a^2 + b^2 = c^2)
  (scale_factor : ℝ) (shortest_side_first : ℝ) (hypo_second : ℝ)
  (h4 : scale_factor = 100 / 25) 
  (h5 : hypo_second = 100) 
  (h6 : b = 7) 
  : (shortest_side_first * scale_factor = 28) :=
by
  sorry

end shortest_side_of_similar_triangle_l68_68623


namespace midpoint_sum_coordinates_l68_68625

theorem midpoint_sum_coordinates :
  let p1 : ℝ × ℝ := (8, -4)
  let p2 : ℝ × ℝ := (-2, 16)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1 + midpoint.2) = 9 :=
by
  let p1 : ℝ × ℝ := (8, -4)
  let p2 : ℝ × ℝ := (-2, 16)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  show midpoint.1 + midpoint.2 = 9
  sorry

end midpoint_sum_coordinates_l68_68625


namespace deck_length_is_30_l68_68055

theorem deck_length_is_30
  (x : ℕ)
  (h1 : ∀ a : ℕ, a = 40 * x)
  (h2 : ∀ b : ℕ, b = 3 * a + 1 * a ∧ b = 4800) :
  x = 30 := by
  sorry

end deck_length_is_30_l68_68055


namespace final_grade_calculation_l68_68701

theorem final_grade_calculation
  (exam_score homework_score class_participation_score : ℝ)
  (exam_weight homework_weight participation_weight : ℝ)
  (h_exam_score : exam_score = 90)
  (h_homework_score : homework_score = 85)
  (h_class_participation_score : class_participation_score = 80)
  (h_exam_weight : exam_weight = 3)
  (h_homework_weight : homework_weight = 2)
  (h_participation_weight : participation_weight = 5) :
  (exam_score * exam_weight + homework_score * homework_weight + class_participation_score * participation_weight) /
  (exam_weight + homework_weight + participation_weight) = 84 :=
by
  -- The proof would go here
  sorry

end final_grade_calculation_l68_68701


namespace largest_marbles_l68_68986

theorem largest_marbles {n : ℕ} (h1 : n < 400) (h2 : n % 3 = 1) (h3 : n % 7 = 2) (h4 : n % 5 = 0) : n = 310 :=
  sorry

end largest_marbles_l68_68986


namespace largest_value_l68_68952

-- Define the five expressions as given in the conditions
def exprA : ℕ := 3 + 1 + 2 + 8
def exprB : ℕ := 3 * 1 + 2 + 8
def exprC : ℕ := 3 + 1 * 2 + 8
def exprD : ℕ := 3 + 1 + 2 * 8
def exprE : ℕ := 3 * 1 * 2 * 8

-- Define the theorem stating that exprE is the largest value
theorem largest_value : exprE = 48 ∧ exprE > exprA ∧ exprE > exprB ∧ exprE > exprC ∧ exprE > exprD := by
  sorry

end largest_value_l68_68952


namespace total_blue_marbles_l68_68115

def jason_blue_marbles : Nat := 44
def tom_blue_marbles : Nat := 24

theorem total_blue_marbles : jason_blue_marbles + tom_blue_marbles = 68 := by
  sorry

end total_blue_marbles_l68_68115


namespace Danica_additional_cars_l68_68871

theorem Danica_additional_cars (n : ℕ) (row_size : ℕ) (danica_cars : ℕ) (answer : ℕ) :
  row_size = 8 →
  danica_cars = 37 →
  answer = 3 →
  ∃ k : ℕ, (k + danica_cars) % row_size = 0 ∧ k = answer :=
by
  sorry

end Danica_additional_cars_l68_68871


namespace chemical_reaction_produces_l68_68266

def balanced_equation : Prop :=
  ∀ {CaCO3 HCl CaCl2 CO2 H2O : ℕ},
    (CaCO3 + 2 * HCl = CaCl2 + CO2 + H2O)

def calculate_final_products (initial_CaCO3 initial_HCl final_CaCl2 final_CO2 final_H2O remaining_HCl : ℕ) : Prop :=
  balanced_equation ∧
  initial_CaCO3 = 3 ∧
  initial_HCl = 8 ∧
  final_CaCl2 = 3 ∧
  final_CO2 = 3 ∧
  final_H2O = 3 ∧
  remaining_HCl = 2

theorem chemical_reaction_produces :
  calculate_final_products 3 8 3 3 3 2 :=
by sorry

end chemical_reaction_produces_l68_68266


namespace correct_operation_l68_68066

theorem correct_operation (a m : ℝ) :
  ¬(a^5 / a^10 = a^2) ∧ 
  (-2 * a^3)^2 = 4 * a^6 ∧ 
  ¬((1 / (2 * m)) - (1 / m) = (1 / m)) ∧ 
  ¬(a^4 + a^3 = a^7) :=
by
  sorry

end correct_operation_l68_68066


namespace smallest_multiple_l68_68684

theorem smallest_multiple (b : ℕ) (h1 : b % 6 = 0) (h2 : b % 15 = 0) (h3 : ∀ n : ℕ, (n % 6 = 0 ∧ n % 15 = 0) → n ≥ b) : b = 30 :=
sorry

end smallest_multiple_l68_68684


namespace parallel_vectors_k_eq_neg1_l68_68179

theorem parallel_vectors_k_eq_neg1
  (k : ℤ)
  (a : ℤ × ℤ := (2 * k + 2, 4))
  (b : ℤ × ℤ := (k + 1, 8))
  (h : a.1 * b.2 = a.2 * b.1) :
  k = -1 :=
by
sorry

end parallel_vectors_k_eq_neg1_l68_68179


namespace most_stable_scores_l68_68489

-- Definitions for the variances of students A, B, and C
def s_A_2 : ℝ := 6
def s_B_2 : ℝ := 24
def s_C_2 : ℝ := 50

-- The proof that student A has the most stable math scores
theorem most_stable_scores : 
  s_A_2 < s_B_2 ∧ s_B_2 < s_C_2 → 
  ("Student A has the most stable scores" = "Student A has the most stable scores") :=
by
  intros h
  sorry

end most_stable_scores_l68_68489


namespace solve_for_nabla_l68_68303

theorem solve_for_nabla (nabla : ℤ) (h : 3 * (-2) = nabla + 2) : nabla = -8 := 
by
  sorry

end solve_for_nabla_l68_68303


namespace y_increase_when_x_increases_by_9_units_l68_68903

-- Given condition as a definition: when x increases by 3 units, y increases by 7 units.
def x_increases_y_increases (x_increase y_increase : ℕ) : Prop := 
  (x_increase = 3) → (y_increase = 7)

-- Stating the problem: when x increases by 9 units, y increases by how many units?
theorem y_increase_when_x_increases_by_9_units : 
  ∀ (x_increase y_increase : ℕ), x_increases_y_increases x_increase y_increase → ((x_increase * 3 = 9) → (y_increase * 3 = 21)) :=
by
  intros x_increase y_increase cond h
  sorry

end y_increase_when_x_increases_by_9_units_l68_68903


namespace books_sold_on_monday_75_l68_68930

namespace Bookstore

variables (total_books sold_Monday sold_Tuesday sold_Wednesday sold_Thursday sold_Friday books_not_sold : ℕ)
variable (percent_not_sold : ℝ)

def given_conditions : Prop :=
  total_books = 1200 ∧
  percent_not_sold = 0.665 ∧
  sold_Tuesday = 50 ∧
  sold_Wednesday = 64 ∧
  sold_Thursday = 78 ∧
  sold_Friday = 135 ∧
  books_not_sold = (percent_not_sold * total_books) ∧
  (total_books - books_not_sold) = (sold_Monday + sold_Tuesday + sold_Wednesday + sold_Thursday + sold_Friday)

theorem books_sold_on_monday_75 (h : given_conditions total_books sold_Monday sold_Tuesday sold_Wednesday sold_Thursday sold_Friday books_not_sold percent_not_sold) :
  sold_Monday = 75 :=
sorry

end Bookstore

end books_sold_on_monday_75_l68_68930


namespace find_unique_number_l68_68435

theorem find_unique_number : 
  ∃ X : ℕ, 
    (X % 1000 = 376 ∨ X % 1000 = 625) ∧ 
    (X * (X - 1) % 10000 = 0) ∧ 
    (Nat.gcd X (X - 1) = 1) ∧ 
    ((X % 625 = 0) ∨ ((X - 1) % 625 = 0)) ∧ 
    ((X % 16 = 0) ∨ ((X - 1) % 16 = 0)) ∧ 
    X = 9376 :=
by sorry

end find_unique_number_l68_68435


namespace find_larger_number_l68_68171

theorem find_larger_number (x y : ℝ) (h1 : x - y = 1860) (h2 : 0.075 * x = 0.125 * y) :
  x = 4650 :=
by
  sorry

end find_larger_number_l68_68171


namespace find_number_l68_68353

theorem find_number (x : ℝ) : x + 5 * 12 / (180 / 3) = 61 ↔ x = 60 := by
  sorry

end find_number_l68_68353


namespace yellow_ball_count_l68_68494

def total_balls : ℕ := 500
def red_balls : ℕ := total_balls / 3
def remaining_after_red : ℕ := total_balls - red_balls
def blue_balls : ℕ := remaining_after_red / 5
def remaining_after_blue : ℕ := remaining_after_red - blue_balls
def green_balls : ℕ := remaining_after_blue / 4
def yellow_balls : ℕ := total_balls - (red_balls + blue_balls + green_balls)

theorem yellow_ball_count : yellow_balls = 201 := by
  sorry

end yellow_ball_count_l68_68494


namespace dad_steps_90_l68_68540

/-- 
  Given:
  - When Dad takes 3 steps, Masha takes 5 steps.
  - When Masha takes 3 steps, Yasha takes 5 steps.
  - Masha and Yasha together made a total of 400 steps.

  Prove: 
  The number of steps that Dad took is 90.
-/
theorem dad_steps_90 (total_steps: ℕ) (masha_to_dad_ratio: ℕ) (yasha_to_masha_ratio: ℕ) (steps_masha_yasha: ℕ) (h1: masha_to_dad_ratio = 5) (h2: yasha_to_masha_ratio = 5) (h3: steps_masha_yasha = 400) :
  total_steps = 90 :=
by
  sorry

end dad_steps_90_l68_68540


namespace hyperbola_a_unique_l68_68164

-- Definitions from the conditions
def hyperbola (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 4 = 1
def foci (c : ℝ) : Prop := c = 2 * Real.sqrt 3
def a_positive (a : ℝ) : Prop := a > 0

-- Statement to prove
theorem hyperbola_a_unique (a : ℝ) (h : hyperbola 0 0 a ∧ foci (2 * Real.sqrt 3) ∧ a_positive a) : a = 2 * Real.sqrt 2 := 
sorry

end hyperbola_a_unique_l68_68164


namespace simplify_polynomial_l68_68740

variable (x : ℝ)

theorem simplify_polynomial :
  (6*x^10 + 8*x^9 + 3*x^7) + (2*x^12 + 3*x^10 + x^9 + 5*x^7 + 4*x^4 + 7*x + 6) =
  2*x^12 + 9*x^10 + 9*x^9 + 8*x^7 + 4*x^4 + 7*x + 6 :=
by
  sorry

end simplify_polynomial_l68_68740


namespace find_number_of_10_bills_from_mother_l68_68408

variable (m10 : ℕ)  -- number of $10 bills given by Luke's mother

def mother_total : ℕ := 50 + 2*20 + 10*m10
def father_total : ℕ := 4*50 + 20 + 10
def total : ℕ := mother_total m10 + father_total

theorem find_number_of_10_bills_from_mother
  (fee : ℕ := 350)
  (m10 : ℕ) :
  total m10 = fee → m10 = 3 := 
by
  sorry

end find_number_of_10_bills_from_mother_l68_68408


namespace space_left_each_side_l68_68268

theorem space_left_each_side (wall_width : ℕ) (picture_width : ℕ)
  (picture_centered : wall_width = 2 * ((wall_width - picture_width) / 2) + picture_width) :
  (wall_width - picture_width) / 2 = 9 :=
by
  have h : wall_width = 25 := sorry
  have h2 : picture_width = 7 := sorry
  exact sorry

end space_left_each_side_l68_68268


namespace length_of_train_l68_68356

-- We state the problem as a theorem in Lean
theorem length_of_train (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ)
  (h_bridge_length : bridge_length = 150)
  (h_crossing_time : crossing_time = 32)
  (h_train_speed_kmh : train_speed_kmh = 45) :
  ∃ (train_length : ℝ), train_length = 250 := 
by
  -- We assume the necessary conditions as given
  have train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  have total_distance : ℝ := train_speed_ms * crossing_time
  have train_length : ℝ := total_distance - bridge_length
  -- Conclude the length of the train is 250
  use train_length
  -- The proof steps are skipped using 'sorry'
  sorry

end length_of_train_l68_68356


namespace ordering_of_exponentiations_l68_68023

def a : ℕ := 3 ^ 34
def b : ℕ := 2 ^ 51
def c : ℕ := 4 ^ 25

theorem ordering_of_exponentiations : c < b ∧ b < a := by
  sorry

end ordering_of_exponentiations_l68_68023


namespace farmer_total_acres_l68_68335

/--
A farmer used some acres of land for beans, wheat, and corn in the ratio of 5 : 2 : 4, respectively.
There were 376 acres used for corn. Prove that the farmer used a total of 1034 acres of land.
-/
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) :
    5 * x + 2 * x + 4 * x = 1034 :=
by
  sorry

end farmer_total_acres_l68_68335


namespace remainder_of_14_pow_53_mod_7_l68_68814

theorem remainder_of_14_pow_53_mod_7 : (14 ^ 53) % 7 = 0 := by
  sorry

end remainder_of_14_pow_53_mod_7_l68_68814


namespace little_john_initial_money_l68_68859

theorem little_john_initial_money :
  let spent := 1.05
  let given_to_each_friend := 2 * 1.00
  let total_spent := spent + given_to_each_friend
  let left := 2.05
  total_spent + left = 5.10 :=
by
  let spent := 1.05
  let given_to_each_friend := 2 * 1.00
  let total_spent := spent + given_to_each_friend
  let left := 2.05
  show total_spent + left = 5.10
  sorry

end little_john_initial_money_l68_68859


namespace find_edge_value_l68_68694

theorem find_edge_value (a b c d e_1 e_2 e_3 e_4 : ℕ) 
  (h1 : e_1 = a + b)
  (h2 : e_2 = b + c)
  (h3 : e_3 = c + d)
  (h4 : e_4 = d + a)
  (h5 : e_1 = 8)
  (h6 : e_3 = 13)
  (h7 : e_1 + e_3 = a + b + c + d)
  : e_4 = 12 := 
by sorry

end find_edge_value_l68_68694


namespace average_speed_for_trip_l68_68828

theorem average_speed_for_trip 
  (Speed1 Speed2 : ℝ) 
  (AverageSpeed : ℝ) 
  (h1 : Speed1 = 110) 
  (h2 : Speed2 = 72) 
  (h3 : AverageSpeed = (2 * Speed1 * Speed2) / (Speed1 + Speed2)) :
  AverageSpeed = 87 := 
by
  -- solution steps would go here
  sorry

end average_speed_for_trip_l68_68828


namespace total_action_figures_l68_68607

def jerry_original_count : Nat := 4
def jerry_added_count : Nat := 6

theorem total_action_figures : jerry_original_count + jerry_added_count = 10 :=
by
  sorry

end total_action_figures_l68_68607


namespace inequality_holds_l68_68618

noncomputable def f (x : ℝ) := x^2 + 2 * Real.cos x

theorem inequality_holds (x1 x2 : ℝ) : 
  f x1 > f x2 → x1 > |x2| := 
sorry

end inequality_holds_l68_68618


namespace age_difference_l68_68513

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 18) : C = A - 18 :=
sorry

end age_difference_l68_68513


namespace trigonometric_identity_l68_68712

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 1 / 2) :
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
by
  sorry

end trigonometric_identity_l68_68712


namespace number_of_squares_or_cubes_l68_68431

theorem number_of_squares_or_cubes (h1 : ∃ n, n = 28) (h2 : ∃ m, m = 9) (h3 : ∃ k, k = 2) : 
  ∃ t, t = 35 :=
sorry

end number_of_squares_or_cubes_l68_68431


namespace problem_solution_l68_68545

theorem problem_solution :
  (- (5 : ℚ) / 12) ^ 2023 * (12 / 5) ^ 2023 = -1 := 
by
  sorry

end problem_solution_l68_68545


namespace quadratic_roots_l68_68597

theorem quadratic_roots (k : ℝ) :
  (∃ x : ℝ, x = 2 ∧ 4 * x ^ 2 - k * x + 6 = 0) →
  k = 11 ∧ (∃ x : ℝ, x ≠ 2 ∧ 4 * x ^ 2 - 11 * x + 6 = 0 ∧ x = 3 / 4) := 
by
  sorry

end quadratic_roots_l68_68597


namespace part1_part2_part3_l68_68547

-- Part (1)
theorem part1 (m n : ℤ) (h1 : m - n = -1) : 2 * (m - n)^2 + 18 = 20 := 
sorry

-- Part (2)
theorem part2 (m n : ℤ) (h2 : m^2 + 2 * m * n = 10) (h3 : n^2 + 3 * m * n = 6) : 2 * m^2 + n^2 + 7 * m * n = 26 :=
sorry

-- Part (3)
theorem part3 (a b c m x : ℤ) (h4: ax^5 + bx^3 + cx - 5 = m) (h5: x = -1) : ax^5 + bx^3 + cx - 5 = -m - 10 :=
sorry

end part1_part2_part3_l68_68547


namespace tina_pink_pens_l68_68340

def number_pink_pens (P G B : ℕ) : Prop :=
  G = P - 9 ∧
  B = P - 6 ∧
  P + G + B = 21

theorem tina_pink_pens :
  ∃ (P G B : ℕ), number_pink_pens P G B ∧ P = 12 :=
by
  sorry

end tina_pink_pens_l68_68340


namespace fifth_term_of_geometric_sequence_l68_68749

theorem fifth_term_of_geometric_sequence (a r : ℕ) (a_pos : 0 < a) (r_pos : 0 < r)
  (h_a : a = 5) (h_fourth_term : a * r^3 = 405) :
  a * r^4 = 405 := by
  sorry

end fifth_term_of_geometric_sequence_l68_68749


namespace average_score_after_19_innings_l68_68248

/-
  Problem Statement:
  Prove that the cricketer's average score after 19 innings is 24,
  given that scoring 96 runs in the 19th inning increased his average by 4.
-/

theorem average_score_after_19_innings :
  ∀ A : ℕ,
  (18 * A + 96) / 19 = A + 4 → A + 4 = 24 :=
by
  intros A h
  /- Skipping proof by adding "sorry" -/
  sorry

end average_score_after_19_innings_l68_68248


namespace rectangle_area_l68_68699

variable {x : ℝ} (h : x > 0)

theorem rectangle_area (W : ℝ) (L : ℝ) (hL : L = 3 * W) (h_diag : W^2 + L^2 = x^2) :
  (W * L) = (3 / 10) * x^2 := by
  sorry

end rectangle_area_l68_68699


namespace find_S3_l68_68525

noncomputable def geometric_sum (n : ℕ) : ℕ := sorry  -- Placeholder for the sum function.

theorem find_S3 (S : ℕ → ℕ) (hS6 : S 6 = 30) (hS9 : S 9 = 70) : S 3 = 10 :=
by
  -- Establish the needed conditions and equation 
  have h : (S 6 - S 3) ^ 2 = (S 9 - S 6) * S 3 := sorry
  -- Substitute given S6 and S9 into the equation and solve
  exact sorry

end find_S3_l68_68525


namespace similar_triangle_perimeter_l68_68096

theorem similar_triangle_perimeter
  (a b c : ℕ)
  (h1 : a = 7)
  (h2 : b = 7)
  (h3 : c = 12)
  (similar_triangle_longest_side : ℕ)
  (h4 : similar_triangle_longest_side = 36)
  (h5 : c * similar_triangle_longest_side = 12 * 36) :
  ∃ P : ℕ, P = 78 := by
  sorry

end similar_triangle_perimeter_l68_68096


namespace restore_original_expression_l68_68112

-- Define the altered product and correct restored products
def original_expression_1 := 4 * 5 * 4 * 7 * 4
def original_expression_2 := 4 * 7 * 4 * 5 * 4
def altered_product := 2247
def corrected_product := 2240

-- Statement that proves the corrected restored product given the altered product
theorem restore_original_expression :
  (4 * 5 * 4 * 7 * 4 = corrected_product ∨ 4 * 7 * 4 * 5 * 4 = corrected_product) :=
sorry

end restore_original_expression_l68_68112


namespace notebook_price_l68_68094

theorem notebook_price (x : ℝ) 
  (h1 : 3 * x + 1.50 + 1.70 = 6.80) : 
  x = 1.20 :=
by 
  sorry

end notebook_price_l68_68094


namespace evaluate_expression_l68_68654

theorem evaluate_expression : (-7)^3 / 7^2 - 4^4 + 5^2 = -238 := by
  sorry

end evaluate_expression_l68_68654


namespace shelves_filled_l68_68185

theorem shelves_filled (carvings_per_shelf : ℕ) (total_carvings : ℕ) (h₁ : carvings_per_shelf = 8) (h₂ : total_carvings = 56) :
  total_carvings / carvings_per_shelf = 7 := by
  sorry

end shelves_filled_l68_68185


namespace part_a_int_values_part_b_int_values_l68_68483

-- Part (a)
theorem part_a_int_values (n : ℤ) :
  ∃ k : ℤ, (n^4 + 3) = k * (n^2 + n + 1) ↔ n = -3 ∨ n = -1 ∨ n = 0 :=
sorry

-- Part (b)
theorem part_b_int_values (n : ℤ) :
  ∃ m : ℤ, (n^3 + n + 1) = m * (n^2 - n + 1) ↔ n = 0 ∨ n = 1 :=
sorry

end part_a_int_values_part_b_int_values_l68_68483


namespace complement_union_eq_l68_68218

def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}
def complement (A : Set ℝ) : Set ℝ := {x | x ∉ A}

theorem complement_union_eq :
  complement (M ∪ N) = {x | x ≥ 1} :=
sorry

end complement_union_eq_l68_68218


namespace nate_search_time_l68_68858

def sectionG_rows : ℕ := 15
def sectionG_cars_per_row : ℕ := 10
def sectionH_rows : ℕ := 20
def sectionH_cars_per_row : ℕ := 9
def cars_per_minute : ℕ := 11

theorem nate_search_time :
  (sectionG_rows * sectionG_cars_per_row + sectionH_rows * sectionH_cars_per_row) / cars_per_minute = 30 :=
  by
    sorry

end nate_search_time_l68_68858


namespace costs_equal_when_x_20_l68_68200

noncomputable def costA (x : ℕ) : ℤ := 150 * x + 3300
noncomputable def costB (x : ℕ) : ℤ := 210 * x + 2100

theorem costs_equal_when_x_20 : costA 20 = costB 20 :=
by
  -- Statements representing the costs equal condition
  have ha : costA 20 = 150 * 20 + 3300 := rfl
  have hb : costB 20 = 210 * 20 + 2100 := rfl
  rw [ha, hb]
  -- Simplification steps (represented here in Lean)
  sorry

end costs_equal_when_x_20_l68_68200


namespace solution_A_to_B_ratio_l68_68373

def ratio_solution_A_to_B (V_A V_B : ℝ) : Prop :=
  (21 / 25) * V_A + (2 / 5) * V_B = (3 / 5) * (V_A + V_B) → V_A / V_B = 5 / 6

theorem solution_A_to_B_ratio (V_A V_B : ℝ) (h : (21 / 25) * V_A + (2 / 5) * V_B = (3 / 5) * (V_A + V_B)) :
  V_A / V_B = 5 / 6 :=
sorry

end solution_A_to_B_ratio_l68_68373


namespace regular_polygon_exterior_angle_l68_68390

theorem regular_polygon_exterior_angle (n : ℕ) (h : 72 = 360 / n) : n = 5 :=
by
  sorry

end regular_polygon_exterior_angle_l68_68390


namespace find_a100_l68_68897

noncomputable def arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a n - a (n + 1) = 2

theorem find_a100 (a : ℕ → ℤ) (h1 : arithmetic_sequence 3 a) (h2 : a 3 = 6) :
  a 100 = -188 :=
sorry

end find_a100_l68_68897


namespace students_in_favor_ABC_l68_68670

variables (U A B C : Finset ℕ)

-- Given conditions
axiom total_students : U.card = 300
axiom students_in_favor_A : A.card = 210
axiom students_in_favor_B : B.card = 190
axiom students_in_favor_C : C.card = 160
axiom students_against_all : (U \ (A ∪ B ∪ C)).card = 40

-- Proof goal
theorem students_in_favor_ABC : (A ∩ B ∩ C).card = 80 :=
by {
  sorry
}

end students_in_favor_ABC_l68_68670


namespace larger_number_l68_68509

theorem larger_number (a b : ℕ) (h1 : 5 * b = 7 * a) (h2 : b - a = 10) : b = 35 :=
sorry

end larger_number_l68_68509


namespace minimum_cards_to_ensure_60_of_same_color_l68_68285

-- Define the conditions as Lean definitions
def total_cards : ℕ := 700
def ratio_red_orange_yellow : ℕ × ℕ × ℕ := (1, 3, 4)
def ratio_green_blue_white : ℕ × ℕ × ℕ := (3, 1, 6)
def yellow_more_than_blue : ℕ := 50

-- Define the proof goal
theorem minimum_cards_to_ensure_60_of_same_color :
  ∀ (x y : ℕ),
  (total_cards = (1 * x + 3 * x + 4 * x + 3 * y + y + 6 * y)) ∧
  (4 * x = y + yellow_more_than_blue) →
  min_cards :=
  -- Sorry here to indicate that proof is not provided
  sorry

end minimum_cards_to_ensure_60_of_same_color_l68_68285


namespace tom_age_ratio_l68_68723

-- Define the conditions
variable (T N : ℕ) (ages_of_children_sum : ℕ)

-- Given conditions as definitions
def condition1 : Prop := T = ages_of_children_sum
def condition2 : Prop := (T - N) = 3 * (T - 4 * N)

-- The theorem statement to be proven
theorem tom_age_ratio : condition1 T ages_of_children_sum ∧ condition2 T N → T / N = 11 / 2 :=
by sorry

end tom_age_ratio_l68_68723


namespace max_value_on_interval_l68_68057

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + c) / Real.exp x
noncomputable def f' (a b c : ℝ) (x : ℝ) : ℝ := ((2 * a * x + b) * Real.exp x - (a * x^2 + b * x + c)) / Real.exp (2 * x)

variable (a b c : ℝ)

-- Given conditions
axiom pos_a : a > 0
axiom zero_point_neg3 : f' a b c (-3) = 0
axiom zero_point_0 : f' a b c 0 = 0
axiom min_value_neg3 : f a b c (-3) = -Real.exp 3

-- Goal: Maximum value of f(x) on the interval [-5, ∞) is 5e^5.
theorem max_value_on_interval : ∃ y ∈ Set.Ici (-5), f a b c y = 5 * Real.exp 5 := by
  sorry

end max_value_on_interval_l68_68057


namespace sample_size_correct_l68_68359

-- Definitions following the conditions in the problem
def total_products : Nat := 80
def sample_products : Nat := 10

-- Statement of the proof problem
theorem sample_size_correct : sample_products = 10 :=
by
  -- The proof is replaced with a placeholder sorry to skip the proof step
  sorry

end sample_size_correct_l68_68359


namespace manager_salary_l68_68432

theorem manager_salary (average_salary_employees : ℕ)
    (employee_count : ℕ) (new_average_salary : ℕ)
    (total_salary_before : ℕ)
    (total_salary_after : ℕ)
    (M : ℕ) :
    average_salary_employees = 1500 →
    employee_count = 20 →
    new_average_salary = 1650 →
    total_salary_before = employee_count * average_salary_employees →
    total_salary_after = (employee_count + 1) * new_average_salary →
    M = total_salary_after - total_salary_before →
    M = 4650 := by
    intros h1 h2 h3 h4 h5 h6
    rw [h6]
    sorry -- The proof is not required, so we use 'sorry' here.

end manager_salary_l68_68432


namespace digit_of_fraction_l68_68677

theorem digit_of_fraction (n : ℕ) : (15 / 37 : ℝ) = 0.405 ∧ 415 % 3 = 1 → ∃ d : ℕ, d = 4 :=
by
  sorry

end digit_of_fraction_l68_68677


namespace eq_970299_l68_68307

theorem eq_970299 : 98^3 + 3 * 98^2 + 3 * 98 + 1 = 970299 :=
by
  sorry

end eq_970299_l68_68307


namespace ABD_collinear_l68_68362

noncomputable def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (p2.1 - p1.1) * k = p3.1 - p1.1 ∧ (p2.2 - p1.2) * k = p3.2 - p1.2

noncomputable def vector (x y : ℝ) : ℝ × ℝ := (x, y)

variables {a b : ℝ × ℝ}
variables {A B C D : ℝ × ℝ}

axiom a_ne_zero : a ≠ (0, 0)
axiom b_ne_zero : b ≠ (0, 0)
axiom a_b_not_collinear : ∀ k : ℝ, a ≠ k • b
axiom AB_def : B = (A.1 + a.1 + b.1, A.2 + a.2 + b.2)
axiom BC_def : C = (B.1 + a.1 + 10 * b.1, B.2 + a.2 + 10 * b.2)
axiom CD_def : D = (C.1 + 3 * (a.1 - 2 * b.1), C.2 + 3 * (a.2 - 2 * b.2))

theorem ABD_collinear : collinear A B D :=
by
  sorry

end ABD_collinear_l68_68362


namespace primes_div_order_l68_68517

theorem primes_div_order (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : q ∣ 3^p - 2^p) : p ∣ q - 1 :=
sorry

end primes_div_order_l68_68517


namespace geometric_sequence_values_l68_68242

theorem geometric_sequence_values (l a b c : ℝ) (h : ∃ r : ℝ, a / (-l) = r ∧ b / a = r ∧ c / b = r ∧ (-9) / c = r) : b = -3 ∧ a * c = 9 :=
by
  sorry

end geometric_sequence_values_l68_68242


namespace check_basis_l68_68418

structure Vector2D :=
  (x : ℤ)
  (y : ℤ)

def are_collinear (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y - v2.x * v1.y = 0

def can_be_basis (v1 v2 : Vector2D) : Prop :=
  ¬ are_collinear v1 v2

theorem check_basis :
  can_be_basis ⟨-1, 2⟩ ⟨5, 7⟩ ∧
  ¬ can_be_basis ⟨0, 0⟩ ⟨1, -2⟩ ∧
  ¬ can_be_basis ⟨3, 5⟩ ⟨6, 10⟩ ∧
  ¬ can_be_basis ⟨2, -3⟩ ⟨(1 : ℤ)/2, -(3 : ℤ)/4⟩ :=
by
  sorry

end check_basis_l68_68418


namespace min_fencing_dims_l68_68586

theorem min_fencing_dims (x : ℕ) (h₁ : x * (x + 5) ≥ 600) (h₂ : x = 23) : 
  2 * (x + (x + 5)) = 102 := 
by
  -- Placeholder for the proof
  sorry

end min_fencing_dims_l68_68586


namespace new_boarders_joined_l68_68943

theorem new_boarders_joined (boarders_initial day_students_initial boarders_final x : ℕ)
  (h1 : boarders_initial = 220)
  (h2 : (5:ℕ) * day_students_initial = (12:ℕ) * boarders_initial)
  (h3 : day_students_initial = 528)
  (h4 : (1:ℕ) * day_students_initial = (2:ℕ) * (boarders_initial + x)) :
  x = 44 := by
  sorry

end new_boarders_joined_l68_68943


namespace evaluate_expression_at_minus_half_l68_68325

noncomputable def complex_expression (x : ℚ) : ℚ :=
  (x - 3)^2 + (x + 3) * (x - 3) - 2 * x * (x - 2) + 1

theorem evaluate_expression_at_minus_half :
  complex_expression (-1 / 2) = 2 :=
by
  sorry

end evaluate_expression_at_minus_half_l68_68325


namespace missing_digit_B_l68_68675

theorem missing_digit_B (B : ℕ) (h : 0 ≤ B ∧ B ≤ 9) (h_div : (100 + 10 * B + 3) % 13 = 0) : B = 4 := 
by
  sorry

end missing_digit_B_l68_68675


namespace inequality_proof_l68_68445

theorem inequality_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
(h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d) (h_sum : a + b + c + d ≥ 1) : 
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 := 
by 
  sorry

end inequality_proof_l68_68445


namespace functions_increasing_in_interval_l68_68123

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin x * Real.cos x

theorem functions_increasing_in_interval :
  ∀ x, -Real.pi / 4 < x → x < Real.pi / 4 →
  (f x < f (x + 1e-6)) ∧ (g x < g (x + 1e-6)) :=
sorry

end functions_increasing_in_interval_l68_68123


namespace candies_per_basket_l68_68318

noncomputable def chocolate_bars : ℕ := 5
noncomputable def mms : ℕ := 7 * chocolate_bars
noncomputable def marshmallows : ℕ := 6 * mms
noncomputable def total_candies : ℕ := chocolate_bars + mms + marshmallows
noncomputable def baskets : ℕ := 25

theorem candies_per_basket : total_candies / baskets = 10 :=
by
  sorry

end candies_per_basket_l68_68318


namespace unique_solution_l68_68745

theorem unique_solution :
  ∀ (x y z n : ℕ), n ≥ 2 → z ≤ 5 * 2^(2 * n) → (x^ (2 * n + 1) - y^ (2 * n + 1) = x * y * z + 2^(2 * n + 1)) → (x, y, z, n) = (3, 1, 70, 2) :=
by
  intros x y z n hn hzn hxyz
  sorry

end unique_solution_l68_68745


namespace chord_length_l68_68925

variable (x y : ℝ)

/--
The chord length cut by the line y = 2x - 2 on the circle (x-2)^2 + (y-2)^2 = 25 is 10.
-/
theorem chord_length (h₁ : y = 2 * x - 2) (h₂ : (x - 2)^2 + (y - 2)^2 = 25) : 
  ∃ length : ℝ, length = 10 :=
sorry

end chord_length_l68_68925


namespace bus_speed_l68_68886

theorem bus_speed (d t : ℕ) (h1 : d = 201) (h2 : t = 3) : d / t = 67 :=
by sorry

end bus_speed_l68_68886


namespace Peter_work_rate_l68_68735

theorem Peter_work_rate:
  ∀ (m p j : ℝ),
    (m + p + j) * 20 = 1 →
    (m + p + j) * 10 = 0.5 →
    (p + j) * 10 = 0.5 →
    j * 15 = 0.5 →
    p * 60 = 1 :=
by
  intros m p j h1 h2 h3 h4
  sorry

end Peter_work_rate_l68_68735


namespace min_value_range_l68_68595

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem min_value_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 2 * a^2 - a - 1) → (0 < a ∧ a ≤ 1) :=
by 
  sorry

end min_value_range_l68_68595


namespace determine_a_l68_68556

theorem determine_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (M m : ℝ)
  (hM : M = max (a^1) (a^2))
  (hm : m = min (a^1) (a^2))
  (hM_m : M = 2 * m) :
  a = 1/2 ∨ a = 2 := 
by sorry

end determine_a_l68_68556


namespace unit_prices_min_number_of_A_l68_68686

theorem unit_prices (x y : ℝ)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860) :
  x = 60 ∧ y = 100 :=
by
  sorry

theorem min_number_of_A (x y a : ℝ)
  (x_h : x = 60)
  (y_h : y = 100)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860)
  (trash_can_condition : a + 200 - a = 200)
  (cost_condition : 60 * a + 100 * (200 - a) ≤ 15000) :
  a ≥ 125 :=
by
  sorry

end unit_prices_min_number_of_A_l68_68686


namespace car_speed_ratio_l68_68561

theorem car_speed_ratio (v_A v_B : ℕ) (h1 : v_B = 50) (h2 : 6 * v_A + 2 * v_B = 1000) :
  v_A / v_B = 3 :=
sorry

end car_speed_ratio_l68_68561


namespace initial_cards_collected_l68_68811

  -- Ralph collects some cards.
  variable (initial_cards: ℕ)

  -- Ralph's father gives Ralph 8 more cards.
  variable (added_cards: ℕ := 8)

  -- Now Ralph has 12 cards.
  variable (total_cards: ℕ := 12)

  -- Proof statement: Prove that the initial number of cards Ralph collected plus 8 equals 12.
  theorem initial_cards_collected: initial_cards + added_cards = total_cards := by
    sorry
  
end initial_cards_collected_l68_68811


namespace gas_cost_problem_l68_68874

theorem gas_cost_problem (x : ℝ) (h : x / 4 - 15 = x / 7) : x = 140 :=
sorry

end gas_cost_problem_l68_68874


namespace function_is_zero_l68_68321

theorem function_is_zero (f : ℝ → ℝ)
  (H : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end function_is_zero_l68_68321


namespace fraction_least_l68_68183

noncomputable def solve_fraction_least : Prop :=
  ∃ (x y : ℚ), x + y = 5/6 ∧ x * y = 1/8 ∧ (min x y = 1/6)
  
theorem fraction_least : solve_fraction_least :=
sorry

end fraction_least_l68_68183


namespace Joey_age_is_six_l68_68259

theorem Joey_age_is_six (ages: Finset ℕ) (a1 a2 a3 a4 : ℕ) (h1: ages = {4, 6, 8, 10})
  (h2: a1 + a2 = 14 ∨ a2 + a3 = 14 ∨ a3 + a4 = 14) (h3: a1 > 7 ∨ a2 > 7 ∨ a3 > 7 ∨ a4 > 7)
  (h4: (6 ∈ ages ∧ a1 ∈ ages) ∨ (6 ∈ ages ∧ a2 ∈ ages) ∨ 
      (6 ∈ ages ∧ a3 ∈ ages) ∨ (6 ∈ ages ∧ a4 ∈ ages)): 
  (a1 = 6 ∨ a2 = 6 ∨ a3 = 6 ∨ a4 = 6) :=
by
  sorry

end Joey_age_is_six_l68_68259


namespace find_natural_n_l68_68290

theorem find_natural_n (a : ℂ) (h₀ : a ≠ 0) (h₁ : a ≠ 1) (h₂ : a ≠ -1)
    (h₃ : a ^ 11 + a ^ 7 + a ^ 3 = 1) : a ^ 4 + a ^ 3 = a ^ 15 + 1 :=
sorry

end find_natural_n_l68_68290


namespace neg_eight_degrees_celsius_meaning_l68_68929

-- Define the temperature in degrees Celsius
def temp_in_degrees_celsius (t : Int) : String :=
  if t >= 0 then toString t ++ "°C above zero"
  else toString (abs t) ++ "°C below zero"

-- Define the proof statement
theorem neg_eight_degrees_celsius_meaning :
  temp_in_degrees_celsius (-8) = "8°C below zero" :=
sorry

end neg_eight_degrees_celsius_meaning_l68_68929


namespace unwilted_roses_proof_l68_68306

-- Conditions
def initial_roses : Nat := 2 * 12
def traded_roses : Nat := 12
def first_day_roses (r: Nat) : Nat := r / 2
def second_day_roses (r: Nat) : Nat := r / 2

-- Initial number of roses
def total_roses : Nat := initial_roses + traded_roses

-- Number of unwilted roses after two days
def unwilted_roses : Nat := second_day_roses (first_day_roses total_roses)

-- Formal statement to prove
theorem unwilted_roses_proof : unwilted_roses = 9 := by
  sorry

end unwilted_roses_proof_l68_68306


namespace find_total_cost_l68_68788

-- Define the cost per kg for flour
def F : ℕ := 21

-- Conditions in the problem
axiom cost_eq_mangos_rice (M R : ℕ) : 10 * M = 10 * R
axiom cost_eq_flour_rice (R : ℕ) : 6 * F = 2 * R

-- Define the cost calculations
def total_cost (M R F : ℕ) : ℕ := (4 * M) + (3 * R) + (5 * F)

-- Prove the total cost given the conditions
theorem find_total_cost (M R : ℕ) (h1 : 10 * M = 10 * R) (h2 : 6 * F = 2 * R) : total_cost M R F = 546 :=
sorry

end find_total_cost_l68_68788


namespace james_hives_l68_68854

-- Define all conditions
def hive_honey : ℕ := 20  -- Each hive produces 20 liters of honey
def jar_capacity : ℕ := 1/2  -- Each jar holds 0.5 liters
def jars_needed : ℕ := 100  -- James needs 100 jars for half the honey

-- Translate to Lean statement
theorem james_hives (hive_honey jar_capacity jars_needed : ℕ) :
  (hive_honey = 20) → 
  (jar_capacity = 1 / 2) →
  (jars_needed = 100) →
  (∀ hives : ℕ, (hives * hive_honey = 200) → hives = 5) :=
by
  intros Hhoney Hjar Hjars
  intros hives Hprod
  sorry

end james_hives_l68_68854


namespace taxi_fare_charge_l68_68348

theorem taxi_fare_charge :
  let initial_fee := 2.25
  let total_distance := 3.6
  let total_charge := 4.95
  let increments := total_distance / (2 / 5)
  let distance_charge := total_charge - initial_fee
  let charge_per_increment := distance_charge / increments
  charge_per_increment = 0.30 :=
by
  sorry

end taxi_fare_charge_l68_68348


namespace average_score_in_all_matches_l68_68420

theorem average_score_in_all_matches (runs_match1_match2 : ℤ) (runs_other_matches : ℤ) (total_matches : ℤ) 
  (average1 : ℤ) (average2 : ℤ)
  (h1 : average1 = 40) (h2 : average2 = 10) (h3 : runs_match1_match2 = 2 * average1)
  (h4 : runs_other_matches = 3 * average2) (h5 : total_matches = 5) :
  ((runs_match1_match2 + runs_other_matches) / total_matches) = 22 := 
by
  sorry

end average_score_in_all_matches_l68_68420


namespace arrangement_ways_l68_68624

def num_ways_arrange_boys_girls : Nat :=
  let boys := 2
  let girls := 3
  let ways_girls := Nat.factorial girls
  let ways_boys := Nat.factorial boys
  ways_girls * ways_boys

theorem arrangement_ways : num_ways_arrange_boys_girls = 12 :=
  by
    sorry

end arrangement_ways_l68_68624


namespace students_prefer_mac_l68_68947

-- Define number of students in survey, and let M be the number who prefer Mac to Windows
variables (M E no_pref windows_pref : ℕ)
-- Total number of students surveyed
variable (total_students : ℕ)
-- Define that the total number of students is 210
axiom H_total : total_students = 210
-- Define that one third as many of the students who prefer Mac equally prefer both brands
axiom H_equal_preference : E = M / 3
-- Define that 90 students had no preference
axiom H_no_pref : no_pref = 90
-- Define that 40 students preferred Windows to Mac
axiom H_windows_pref : windows_pref = 40
-- Define that the total number of students is the sum of all groups
axiom H_students_sum : M + E + no_pref + windows_pref = total_students

-- The statement we need to prove
theorem students_prefer_mac :
  M = 60 :=
by sorry

end students_prefer_mac_l68_68947


namespace find_a_l68_68069

theorem find_a (a : ℤ) (h : ∃ x1 x2 : ℤ, (x - x1) * (x - x2) = (x - a) * (x - 8) - 1) : a = 8 :=
sorry

end find_a_l68_68069


namespace ai_eq_i_l68_68730

namespace Problem

def gcd (m n : ℕ) : ℕ := Nat.gcd m n

def sequence_satisfies (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → gcd (a i) (a j) = gcd i j

theorem ai_eq_i (a : ℕ → ℕ) (h : sequence_satisfies a) : ∀ i : ℕ, a i = i :=
by
  sorry

end Problem

end ai_eq_i_l68_68730


namespace exists_finite_group_with_normal_subgroup_GT_Aut_l68_68878

noncomputable def finite_group_G (n : ℕ) : Type := sorry -- Specific construction details omitted
noncomputable def normal_subgroup_H (n : ℕ) : Type := sorry -- Specific construction details omitted

def Aut_G (n : ℕ) : ℕ := sorry -- Number of automorphisms of G
def Aut_H (n : ℕ) : ℕ := sorry -- Number of automorphisms of H

theorem exists_finite_group_with_normal_subgroup_GT_Aut (n : ℕ) :
  ∃ G H, finite_group_G n = G ∧ normal_subgroup_H n = H ∧ Aut_H n > Aut_G n := sorry

end exists_finite_group_with_normal_subgroup_GT_Aut_l68_68878


namespace correct_meiosis_sequence_l68_68835

-- Define the events as types
inductive Event : Type
| Replication : Event
| Synapsis : Event
| Separation : Event
| Division : Event

-- Define options as lists of events
def option_A := [Event.Replication, Event.Synapsis, Event.Separation, Event.Division]
def option_B := [Event.Synapsis, Event.Replication, Event.Separation, Event.Division]
def option_C := [Event.Synapsis, Event.Replication, Event.Division, Event.Separation]
def option_D := [Event.Replication, Event.Separation, Event.Synapsis, Event.Division]

-- Define the theorem to be proved
theorem correct_meiosis_sequence : option_A = [Event.Replication, Event.Synapsis, Event.Separation, Event.Division] :=
by
  sorry

end correct_meiosis_sequence_l68_68835


namespace exists_good_pair_for_each_m_l68_68971

def is_good_pair (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), m * n = a^2 ∧ (m + 1) * (n + 1) = b^2

theorem exists_good_pair_for_each_m : ∀ m : ℕ, ∃ n : ℕ, m < n ∧ is_good_pair m n := by
  intro m
  let n := m * (4 * m + 3)^2
  use n
  have h1 : m < n := sorry -- Proof that m < n
  have h2 : is_good_pair m n := sorry -- Proof that (m, n) is a good pair
  exact ⟨h1, h2⟩

end exists_good_pair_for_each_m_l68_68971


namespace intersection_of_P_and_Q_is_false_iff_union_of_P_and_Q_is_false_l68_68781

variable (P Q : Prop)

theorem intersection_of_P_and_Q_is_false_iff_union_of_P_and_Q_is_false
  (h : P ∧ Q = False) : (P ∨ Q = False) ↔ (P ∧ Q = False) := 
by 
  sorry

end intersection_of_P_and_Q_is_false_iff_union_of_P_and_Q_is_false_l68_68781


namespace corvette_trip_time_percentage_increase_l68_68214

theorem corvette_trip_time_percentage_increase
  (total_distance : ℝ)
  (first_half_speed : ℝ)
  (average_speed : ℝ)
  (first_half_distance second_half_distance first_half_time second_half_time total_time : ℝ)
  (h1 : total_distance = 640)
  (h2 : first_half_speed = 80)
  (h3 : average_speed = 40)
  (h4 : first_half_distance = total_distance / 2)
  (h5 : second_half_distance = total_distance / 2)
  (h6 : first_half_time = first_half_distance / first_half_speed)
  (h7 : total_time = total_distance / average_speed)
  (h8 : second_half_time = total_time - first_half_time) :
  ((second_half_time - first_half_time) / first_half_time) * 100 = 200 := sorry

end corvette_trip_time_percentage_increase_l68_68214


namespace find_value_l68_68088

def equation := ∃ x : ℝ, x^2 - 2 * x - 3 = 0
def expression (x : ℝ) := 2 * x^2 - 4 * x + 12

theorem find_value :
  (∃ x : ℝ, (x^2 - 2 * x - 3 = 0) ∧ (expression x = 18)) :=
by
  sorry

end find_value_l68_68088


namespace minimum_value_of_func_l68_68438

-- Define the circle and the line constraints, and the question
namespace CircleLineProblem

def is_center_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y + 1 = 0

def line_divides_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, is_center_of_circle x y → a * x - b * y + 3 = 0

noncomputable def func_to_minimize (a b : ℝ) : ℝ :=
  (2 / a) + (1 / (b - 1))

theorem minimum_value_of_func :
  ∃ (a b : ℝ), a > 0 ∧ b > 1 ∧ line_divides_circle a b ∧ func_to_minimize a b = 8 :=
by
  sorry

end CircleLineProblem

end minimum_value_of_func_l68_68438


namespace num_letters_dot_not_straight_line_l68_68029

variable (Total : ℕ)
variable (DS : ℕ)
variable (S_only : ℕ)
variable (D_only : ℕ)

theorem num_letters_dot_not_straight_line 
  (h1 : Total = 40) 
  (h2 : DS = 11) 
  (h3 : S_only = 24) 
  (h4 : Total - S_only - DS = D_only) : 
  D_only = 5 := 
by 
  sorry

end num_letters_dot_not_straight_line_l68_68029


namespace ferry_captives_successfully_l68_68019

-- Definition of conditions
def valid_trip_conditions (trips: ℕ) (captives: ℕ) : Prop :=
  captives = 43 ∧
  (∀ k < trips, k % 2 = 0 ∨ k % 2 = 1) ∧     -- Trips done in pairs or singles
  (∀ k < captives, k > 40)                    -- At least 40 other captives known as werewolves

-- Theorem statement to be proved
theorem ferry_captives_successfully (trips : ℕ) (captives : ℕ) (result : Prop) : 
  valid_trip_conditions trips captives → result = true := by sorry

end ferry_captives_successfully_l68_68019


namespace smallest_three_digit_number_with_property_l68_68025

theorem smallest_three_digit_number_with_property : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
  (∀ d, (1 ≤ d ∧ d ≤ 1000) → ((d = n + 1 ∨ d = n - 1) → d % 11 = 0)) ∧ 
  n = 120 :=
by
  sorry

end smallest_three_digit_number_with_property_l68_68025


namespace basketball_tournament_l68_68935

theorem basketball_tournament (teams : Finset ℕ) (games_played : ℕ → ℕ → ℕ) (win_chance : ℕ → ℕ → Prop) 
(points : ℕ → ℕ) (X Y : ℕ) :
  teams.card = 6 → 
  (∀ t₁ t₂, t₁ ≠ t₂ → games_played t₁ t₂ = 1) → 
  (∀ t₁ t₂, t₁ ≠ t₂ → win_chance t₁ t₂ ∨ win_chance t₂ t₁) → 
  (∀ t₁ t₂, t₁ ≠ t₂ → win_chance t₁ t₂ → points t₁ = points t₁ + 1 ∧ points t₂ = points t₂) → 
  win_chance X Y →
  0.5 = 0.5 →
  0.5 * (1 - ((252 : ℚ) / 1024)) = (193 : ℚ) / 512 →
  ((63 : ℚ) / 256) + ((193 : ℚ) / 512) = (319 : ℚ) / 512 :=
by 
  sorry 

end basketball_tournament_l68_68935


namespace guppies_total_l68_68191

theorem guppies_total :
  let haylee := 3 * 12
  let jose := haylee / 2
  let charliz := jose / 3
  let nicolai := charliz * 4
  haylee + jose + charliz + nicolai = 84 :=
by
  sorry

end guppies_total_l68_68191


namespace discriminant_of_given_quadratic_l68_68562

-- define the coefficients a, b, c
def a : ℚ := 2
def b : ℚ := 2 + 1/2
def c : ℚ := 1/2

-- define the discriminant function for a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

-- state the theorem
theorem discriminant_of_given_quadratic : discriminant a b c = 9/4 :=
by
  -- add the proof here
  sorry

end discriminant_of_given_quadratic_l68_68562


namespace quadratic_switch_real_roots_l68_68240

theorem quadratic_switch_real_roots (a b c u v w : ℝ) (ha : a ≠ u)
  (h_root1 : b^2 - 4 * a * c ≥ 0)
  (h_root2 : v^2 - 4 * u * w ≥ 0)
  (hwc : w * c > 0) :
  (b^2 - 4 * u * c ≥ 0) ∨ (v^2 - 4 * a * w ≥ 0) :=
sorry

end quadratic_switch_real_roots_l68_68240


namespace dot_product_a_b_l68_68313

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (-1, 2)

theorem dot_product_a_b : vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 1 := by
  sorry

end dot_product_a_b_l68_68313


namespace number_534n_divisible_by_12_l68_68846

theorem number_534n_divisible_by_12 (n : ℕ) : (5340 + n) % 12 = 0 ↔ n = 0 := by sorry

end number_534n_divisible_by_12_l68_68846


namespace gcd_765432_654321_eq_3_l68_68486

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l68_68486


namespace expected_value_smallest_N_l68_68471
noncomputable def expectedValueN : ℝ := 6.54

def barryPicksPointsInsideUnitCircle (P : ℕ → ℝ × ℝ) : Prop :=
  ∀ n, (P n).fst^2 + (P n).snd^2 ≤ 1

def pointsIndependentAndUniform (P : ℕ → ℝ × ℝ) : Prop :=
  -- This is a placeholder representing the independent and uniform picking which 
  -- would be formally defined using probability measures in an advanced Lean library.
  sorry

theorem expected_value_smallest_N (P : ℕ → ℝ × ℝ)
  (h1 : barryPicksPointsInsideUnitCircle P)
  (h2 : pointsIndependentAndUniform P) :
  ∃ N : ℕ, N = expectedValueN :=
sorry

end expected_value_smallest_N_l68_68471


namespace calc_diagonal_of_rectangle_l68_68819

variable (a : ℕ) (A : ℕ)

theorem calc_diagonal_of_rectangle (h_a : a = 6) (h_A : A = 48) (H : a * a' = A) :
  ∃ d : ℕ, d = 10 :=
by
 sorry

end calc_diagonal_of_rectangle_l68_68819


namespace gain_percent_is_87_point_5_l68_68162

noncomputable def gain_percent (C S : ℝ) : ℝ :=
  ((S - C) / C) * 100

theorem gain_percent_is_87_point_5 {C S : ℝ} (h : 75 * C = 40 * S) :
  gain_percent C S = 87.5 :=
by
  sorry

end gain_percent_is_87_point_5_l68_68162


namespace correct_statements_l68_68232

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

noncomputable def a_n_sequence (n : ℕ) := a n
noncomputable def Sn_sum (n : ℕ) := S n

axiom Sn_2022_lt_zero : S 2022 < 0
axiom Sn_2023_gt_zero : S 2023 > 0

theorem correct_statements :
  (a 1012 > 0) ∧ ( ∀ n, S n >= S 1011 → n = 1011) :=
  sorry

end correct_statements_l68_68232


namespace simplify_expression_l68_68044

theorem simplify_expression : 2 - Real.sqrt 3 + 1 / (2 - Real.sqrt 3) + 1 / (Real.sqrt 3 + 2) = 6 :=
by
  sorry

end simplify_expression_l68_68044


namespace fraction_of_track_in_forest_l68_68920

theorem fraction_of_track_in_forest (n : ℕ) (l : ℝ) (A B C : ℝ) :
  (∃ x, x = 2*l/3 ∨ x = l/3) → (∃ f, 0 < f ∧ f ≤ 1 ∧ (f = 2/3 ∨ f = 1/3)) :=
by
  -- sorry, the proof will go here
  sorry

end fraction_of_track_in_forest_l68_68920


namespace least_integer_condition_l68_68170

theorem least_integer_condition : ∃ x : ℤ, (x^2 = 2 * x + 72) ∧ (x = -6) :=
sorry

end least_integer_condition_l68_68170


namespace root_conditions_l68_68497

theorem root_conditions (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |x1^2 - 5 * x1| = a ∧ |x2^2 - 5 * x2| = a) ↔ (a = 0 ∨ a > 25 / 4) := 
by 
  sorry

end root_conditions_l68_68497


namespace symmetric_periodic_l68_68106

theorem symmetric_periodic
  (f : ℝ → ℝ) (a b : ℝ) (h1 : a ≠ b)
  (h2 : ∀ x : ℝ, f (a - x) = f (a + x))
  (h3 : ∀ x : ℝ, f (b - x) = f (b + x)) :
  ∀ x : ℝ, f x = f (x + 2 * (b - a)) :=
by
  sorry

end symmetric_periodic_l68_68106


namespace mateen_backyard_area_l68_68588

theorem mateen_backyard_area :
  (∀ (L : ℝ), 30 * L = 1200) →
  (∀ (P : ℝ), 12 * P = 1200) →
  (∃ (L W : ℝ), 2 * L + 2 * W = 100 ∧ L * W = 400) := by
  intros hL hP
  use 40
  use 10
  apply And.intro
  sorry
  sorry

end mateen_backyard_area_l68_68588


namespace coin_difference_l68_68247

theorem coin_difference : ∀ (p : ℕ), 1 ≤ p ∧ p ≤ 999 → (10000 - 9 * 1) - (10000 - 9 * 999) = 8982 :=
by
  intro p
  intro hp
  sorry

end coin_difference_l68_68247


namespace solve_for_x_l68_68640

def determinant_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem solve_for_x (x : ℝ) (h : determinant_2x2 (x+1) (x+2) (x-3) (x-1) = 2023) :
  x = 2018 :=
by {
  sorry
}

end solve_for_x_l68_68640


namespace triple_sum_equals_seven_l68_68461

theorem triple_sum_equals_seven {k m n : ℕ} (hk : 0 < k) (hm : 0 < m) (hn : 0 < n)
  (hcoprime : Nat.gcd k m = 1 ∧ Nat.gcd k n = 1 ∧ Nat.gcd m n = 1)
  (hlog : k * Real.log 5 / Real.log 400 + m * Real.log 2 / Real.log 400 = n) :
  k + m + n = 7 := by
  sorry

end triple_sum_equals_seven_l68_68461


namespace cost_of_meal_l68_68764

noncomputable def total_cost (hamburger_cost fry_cost drink_cost : ℕ) (num_hamburgers num_fries num_drinks : ℕ) (discount_rate : ℕ) : ℕ :=
  let initial_cost := (hamburger_cost * num_hamburgers) + (fry_cost * num_fries) + (drink_cost * num_drinks)
  let discount := initial_cost * discount_rate / 100
  initial_cost - discount

theorem cost_of_meal :
  total_cost 5 3 2 3 4 6 10 = 35 := by
  sorry

end cost_of_meal_l68_68764


namespace optionB_is_a9_l68_68329

-- Definitions of the expressions
def optionA (a : ℤ) : ℤ := a^3 + a^6
def optionB (a : ℤ) : ℤ := a^3 * a^6
def optionC (a : ℤ) : ℤ := a^10 - a
def optionD (a α : ℤ) : ℤ := α^12 / a^2

-- Theorem stating which option equals a^9
theorem optionB_is_a9 (a α : ℤ) : optionA a ≠ a^9 ∧ optionB a = a^9 ∧ optionC a ≠ a^9 ∧ optionD a α ≠ a^9 :=
by
  sorry

end optionB_is_a9_l68_68329


namespace speed_of_man_l68_68365

theorem speed_of_man :
  let L := 500 -- Length of the train in meters
  let t := 29.997600191984642 -- Time in seconds
  let V_train_kmh := 63 -- Speed of train in km/hr
  let V_train := (63 * 1000) / 3600 -- Speed of train converted to m/s
  let V_relative := L / t -- Relative speed of train w.r.t man
  
  V_train - V_relative = 0.833 := by
  sorry

end speed_of_man_l68_68365


namespace tan_neg_5pi_over_4_l68_68613

theorem tan_neg_5pi_over_4 : Real.tan (-5 * Real.pi / 4) = -1 :=
by 
  sorry

end tan_neg_5pi_over_4_l68_68613


namespace total_cost_of_roads_l68_68358

/-- A rectangular lawn with dimensions 150 m by 80 m with two roads running 
through the middle, one parallel to the length and one parallel to the breadth. 
The first road has a width of 12 m, a base cost of Rs. 4 per sq m, and an additional section 
through a hill costing 25% more for a section of length 60 m. The second road has a width 
of 8 m and a cost of Rs. 5 per sq m. Prove that the total cost for both roads is Rs. 14000. -/
theorem total_cost_of_roads :
  let lawn_length := 150
  let lawn_breadth := 80
  let road1_width := 12
  let road2_width := 8
  let road1_base_cost := 4
  let road1_hill_length := 60
  let road1_hill_cost := road1_base_cost + (road1_base_cost / 4)
  let road2_cost := 5
  let road1_length := lawn_length
  let road2_length := lawn_breadth

  let road1_area_non_hill := road1_length * road1_width
  let road1_area_hill := road1_hill_length * road1_width
  let road1_cost_non_hill := road1_area_non_hill * road1_base_cost
  let road1_cost_hill := road1_area_hill * road1_hill_cost

  let total_road1_cost := road1_cost_non_hill + road1_cost_hill

  let road2_area := road2_length * road2_width
  let road2_total_cost := road2_area * road2_cost

  let total_cost := total_road1_cost + road2_total_cost

  total_cost = 14000 := by sorry

end total_cost_of_roads_l68_68358


namespace min_friend_pairs_l68_68940

-- Define conditions
def n : ℕ := 2000
def invitations_per_person : ℕ := 1000
def total_invitations : ℕ := n * invitations_per_person

-- Mathematical problem statement
theorem min_friend_pairs : (total_invitations / 2) = 1000000 := 
by sorry

end min_friend_pairs_l68_68940


namespace n_is_power_of_p_l68_68386

-- Given conditions as definitions
variables {x y p n k l : ℕ}
variables (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < p) (h4 : 0 < n) (h5 : 0 < k)
variables (h6 : x^n + y^n = p^k) (h7 : odd n) (h8 : n > 1) (h9 : prime p) (h10 : odd p)

-- The theorem to be proved
theorem n_is_power_of_p : ∃ l : ℕ, n = p^l :=
  sorry

end n_is_power_of_p_l68_68386


namespace multiple_of_5_l68_68893

theorem multiple_of_5 (a : ℤ) (h : ¬ (5 ∣ a)) : 5 ∣ (a^12 - 1) :=
by
  sorry

end multiple_of_5_l68_68893


namespace range_of_m_l68_68402

theorem range_of_m (x y : ℝ) (m : ℝ) (h1 : x^2 + y^2 = 9) (h2 : |x| + |y| ≥ m) :
    m ≤ 3 / 2 := 
sorry

end range_of_m_l68_68402


namespace difference_of_squares_36_l68_68229

theorem difference_of_squares_36 {x y : ℕ} (h₁ : x + y = 18) (h₂ : x * y = 80) (h₃ : x > y) : x^2 - y^2 = 36 :=
by
  sorry

end difference_of_squares_36_l68_68229


namespace system_real_solution_conditions_l68_68848

theorem system_real_solution_conditions (a b c x y z : ℝ) (h1 : a * x + b * y = c * z) (h2 : a * Real.sqrt (1 - x^2) + b * Real.sqrt (1 - y^2) = c * Real.sqrt (1 - z^2)) :
  abs a ≤ abs b + abs c ∧ abs b ≤ abs a + abs c ∧ abs c ≤ abs a + abs b ∧
  (a * b >= 0 ∨ a * c >= 0 ∨ b * c >= 0) :=
sorry

end system_real_solution_conditions_l68_68848


namespace relationship_f_3x_ge_f_2x_l68_68399

/-- Given a quadratic function f(x) = ax^2 + bx + c with a > 0, and
    satisfying the symmetry condition f(1-x) = f(1+x) for any x ∈ ℝ,
    the relationship f(3^x) ≥ f(2^x) holds. -/
theorem relationship_f_3x_ge_f_2x (a b c : ℝ) (h_a : a > 0) (symm_cond : ∀ x : ℝ, (a * (1 - x)^2 + b * (1 - x) + c) = (a * (1 + x)^2 + b * (1 + x) + c)) :
  ∀ x : ℝ, (a * (3^x)^2 + b * 3^x + c) ≥ (a * (2^x)^2 + b * 2^x + c) :=
sorry

end relationship_f_3x_ge_f_2x_l68_68399


namespace canadian_math_olympiad_1992_l68_68705

theorem canadian_math_olympiad_1992
    (n : ℤ) (a : ℕ → ℤ) (k : ℕ)
    (h1 : n ≥ a 1) 
    (h2 : ∀ i, 1 ≤ i → i ≤ k → a i > 0)
    (h3 : ∀ i j, 1 ≤ i → i ≤ k → 1 ≤ j → j ≤ k → n ≥ Int.lcm (a i) (a j))
    (h4 : ∀ i, 1 ≤ i → i < k → a i > a (i + 1)) :
  ∀ i, 1 ≤ i → i ≤ k → i * a i ≤ n :=
sorry

end canadian_math_olympiad_1992_l68_68705


namespace at_least_two_pairs_in_one_drawer_l68_68866

theorem at_least_two_pairs_in_one_drawer (n : ℕ) (hn : n > 0) : 
  ∃ i j : ℕ, i ≠ j ∧ i < n ∧ j < n :=
by {
  sorry
}

end at_least_two_pairs_in_one_drawer_l68_68866


namespace grover_total_profit_l68_68216

theorem grover_total_profit :
  let boxes := 3
  let masks_per_box := 20
  let price_per_mask := 0.50
  let cost := 15
  let total_masks := boxes * masks_per_box
  let total_revenue := total_masks * price_per_mask
  let total_profit := total_revenue - cost
  total_profit = 15 := by
sorry

end grover_total_profit_l68_68216


namespace efficiency_ratio_l68_68249

theorem efficiency_ratio (E_A E_B : ℝ) 
  (h1 : E_B = 1 / 18) 
  (h2 : E_A + E_B = 1 / 6) : 
  E_A / E_B = 2 :=
by {
  sorry
}

end efficiency_ratio_l68_68249


namespace problem_statement_l68_68660

theorem problem_statement (a b c d : ℝ) 
  (hab : a ≤ b)
  (hbc : b ≤ c)
  (hcd : c ≤ d)
  (hsum : a + b + c + d = 0)
  (hinv_sum : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 :=
sorry

end problem_statement_l68_68660


namespace scientific_notation_347000_l68_68235

theorem scientific_notation_347000 :
  347000 = 3.47 * 10^5 :=
by 
  -- Proof will go here
  sorry

end scientific_notation_347000_l68_68235


namespace marble_221_is_green_l68_68354

def marble_sequence_color (n : ℕ) : String :=
  let cycle_length := 15
  let red_count := 6
  let green_start := red_count + 1
  let green_end := red_count + 5
  let position := n % cycle_length
  if position ≠ 0 then
    let cycle_position := position
    if cycle_position <= red_count then "red"
    else if cycle_position <= green_end then "green"
    else "blue"
  else "blue"

theorem marble_221_is_green : marble_sequence_color 221 = "green" :=
by
  -- proof to be filled in
  sorry

end marble_221_is_green_l68_68354


namespace balance_balls_l68_68791

variable (G B Y W R : ℕ)

theorem balance_balls :
  (4 * G = 8 * B) →
  (3 * Y = 7 * B) →
  (8 * B = 5 * W) →
  (2 * R = 6 * B) →
  (5 * G + 3 * Y + 3 * R = 26 * B) :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end balance_balls_l68_68791


namespace plane_hit_probability_l68_68890

theorem plane_hit_probability :
  let P_A : ℝ := 0.3
  let P_B : ℝ := 0.5
  let P_not_A : ℝ := 1 - P_A
  let P_not_B : ℝ := 1 - P_B
  let P_both_miss : ℝ := P_not_A * P_not_B
  let P_plane_hit : ℝ := 1 - P_both_miss
  P_plane_hit = 0.65 :=
by
  sorry

end plane_hit_probability_l68_68890


namespace linear_equation_unique_l68_68415

theorem linear_equation_unique (x y : ℝ) : 
  (3 * x = 2 * y) ∧ 
  ¬(3 * x - 6 = x) ∧ 
  ¬(x - 1 / y = 0) ∧ 
  ¬(2 * x - 3 * y = x * y) :=
by
  sorry

end linear_equation_unique_l68_68415


namespace range_of_x_range_of_a_l68_68223

-- Problem (1) representation
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (m x : ℝ) : Prop := 1 < m ∧ m < 2 ∧ x = (1 / 2)^(m - 1)

theorem range_of_x (x : ℝ) :
  (∀ m, 1 < m ∧ m < 2 → x = (1 / 2)^(m - 1)) ∧ p (1/4) x →
  1/2 < x ∧ x < 3/4 :=
sorry

-- Problem (2) representation
theorem range_of_a (a : ℝ) :
  (∀ m, 1 < m ∧ m < 2 → ∀ x, x = (1 / 2)^(m - 1) → p a x) →
  1/3 ≤ a ∧ a ≤ 1/2 :=
sorry

end range_of_x_range_of_a_l68_68223


namespace least_number_of_groups_l68_68157

def num_students : ℕ := 24
def max_students_per_group : ℕ := 10

theorem least_number_of_groups : ∃ x, ∀ y, y ≤ max_students_per_group ∧ num_students = x * y → x = 3 := by
  sorry

end least_number_of_groups_l68_68157


namespace fencing_required_l68_68294

-- Conditions
def L : ℕ := 20
def A : ℕ := 680

-- Statement to prove
theorem fencing_required : ∃ W : ℕ, A = L * W ∧ 2 * W + L = 88 :=
by
  -- Here you would normally need the logical steps to arrive at the proof
  sorry

end fencing_required_l68_68294


namespace expected_value_is_750_l68_68650

def winnings (roll : ℕ) : ℕ :=
  if roll % 2 = 0 then 3 * roll else 0

def expected_value : ℚ :=
  (winnings 2 / 8) + (winnings 4 / 8) + (winnings 6 / 8) + (winnings 8 / 8)

theorem expected_value_is_750 : expected_value = 7.5 := by
  sorry

end expected_value_is_750_l68_68650


namespace pipe_A_fill_time_l68_68346

theorem pipe_A_fill_time 
  (t : ℝ)
  (ht : (1 / t - 1 / 6) = 4 / 15.000000000000005) : 
  t = 30 / 13 :=  
sorry

end pipe_A_fill_time_l68_68346


namespace intersecting_lines_value_l68_68931

theorem intersecting_lines_value (m b : ℚ)
  (h₁ : 10 = m * 7 + 5)
  (h₂ : 10 = 2 * 7 + b) :
  b + m = - (23 : ℚ) / 7 := 
sorry

end intersecting_lines_value_l68_68931


namespace subset_condition_l68_68127

theorem subset_condition (m : ℝ) (A : Set ℝ) (B : Set ℝ) :
  A = {1, 3} ∧ B = {1, 2, m} ∧ A ⊆ B → m = 3 :=
by
  sorry

end subset_condition_l68_68127


namespace expression_eval_l68_68121

theorem expression_eval : 2 * 3 + 2 * 3 = 12 := by
  sorry

end expression_eval_l68_68121


namespace total_games_in_season_l68_68190

theorem total_games_in_season (n_teams : ℕ) (games_between_each_team : ℕ) (non_conf_games_per_team : ℕ) 
  (h_teams : n_teams = 8) (h_games_between : games_between_each_team = 3) (h_non_conf : non_conf_games_per_team = 3) :
  let games_within_league := (n_teams * (n_teams - 1) / 2) * games_between_each_team
  let games_outside_league := n_teams * non_conf_games_per_team
  games_within_league + games_outside_league = 108 := by
  sorry

end total_games_in_season_l68_68190


namespace evaporation_amount_l68_68755

variable (E : ℝ)

def initial_koolaid_powder : ℝ := 2
def initial_water : ℝ := 16
def final_percentage : ℝ := 0.04

theorem evaporation_amount :
  (initial_koolaid_powder = 2) →
  (initial_water = 16) →
  (0.04 * (initial_koolaid_powder + 4 * (initial_water - E)) = initial_koolaid_powder) →
  E = 4 :=
by
  intros h1 h2 h3
  sorry

end evaporation_amount_l68_68755


namespace sum_greater_than_3_l68_68014

theorem sum_greater_than_3 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b + b * c + c * a > a + b + c) : a + b + c > 3 :=
sorry

end sum_greater_than_3_l68_68014


namespace LitterPatrol_pickup_l68_68849

theorem LitterPatrol_pickup :
  ∃ n : ℕ, n = 10 + 8 :=
sorry

end LitterPatrol_pickup_l68_68849


namespace binom_divisible_by_prime_l68_68064

theorem binom_divisible_by_prime {p k : ℕ} (hp : Nat.Prime p) (hk1 : 1 ≤ k) (hk2 : k ≤ p - 1) : Nat.choose p k % p = 0 := 
  sorry

end binom_divisible_by_prime_l68_68064


namespace inequalities_proof_l68_68156

theorem inequalities_proof (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  (a < (c / 2)) ∧ (b < a + c / 2) ∧ ¬(b < c / 2) :=
by
  constructor
  { sorry }
  { constructor
    { sorry }
    { sorry } }

end inequalities_proof_l68_68156


namespace smallest_positive_period_of_f_f_ge_negative_sqrt_3_in_interval_l68_68273

noncomputable def f (x : Real) : Real :=
  Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x) ^ 2

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, ( ∀ x, f (x + T') = f x) → T ≤ T') := by
  sorry

theorem f_ge_negative_sqrt_3_in_interval :
  ∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 6), f x ≥ -Real.sqrt 3 := by
  sorry

end smallest_positive_period_of_f_f_ge_negative_sqrt_3_in_interval_l68_68273


namespace probability_non_black_ball_l68_68108

/--
Given the odds of drawing a black ball as 5:3,
prove that the probability of drawing a non-black ball from the bag is 3/8.
-/
theorem probability_non_black_ball (n_black n_non_black : ℕ) (h : n_black = 5) (h' : n_non_black = 3) :
  (n_non_black : ℚ) / (n_black + n_non_black) = 3 / 8 :=
by
  -- proof goes here
  sorry

end probability_non_black_ball_l68_68108


namespace cost_keyboard_l68_68671

def num_keyboards : ℕ := 15
def num_printers : ℕ := 25
def total_cost : ℝ := 2050
def cost_printer : ℝ := 70
def total_cost_printers : ℝ := num_printers * cost_printer
def total_cost_keyboards : ℝ := total_cost - total_cost_printers

theorem cost_keyboard : total_cost_keyboards / num_keyboards = 20 := by
  sorry

end cost_keyboard_l68_68671


namespace evaluate_g_at_3_l68_68616

def g (x : ℤ) : ℤ := 3 * x^3 + 5 * x^2 - 2 * x - 7

theorem evaluate_g_at_3 : g 3 = 113 := by
  -- Proof of g(3) = 113 skipped
  sorry

end evaluate_g_at_3_l68_68616


namespace polygon_sides_arithmetic_sequence_l68_68481

theorem polygon_sides_arithmetic_sequence 
  (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : 2 * (180 * (n - 2)) = n * (100 + 140)) :
  n = 6 :=
  sorry

end polygon_sides_arithmetic_sequence_l68_68481


namespace find_m_given_root_of_quadratic_l68_68144

theorem find_m_given_root_of_quadratic (m : ℝ) : (∃ x : ℝ, x = 3 ∧ x^2 - m * x - 6 = 0) → m = 1 := 
by
  sorry

end find_m_given_root_of_quadratic_l68_68144


namespace horner_v1_value_l68_68921

-- Definitions based on the problem conditions
def f (x : ℝ) : ℝ := 4 * x^5 - 12 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

def horner (x : ℝ) (coeffs : List ℝ) : ℝ :=
  coeffs.foldl (fun acc coeff => acc * x + coeff) 0

theorem horner_v1_value :
  let x := 5
  let coeffs := [4, -12, 3.5, -2.6, 1.7, -0.8]
  let v0 := coeffs.head!
  let v1 := v0 * x + coeffs.getD 1 0
  v1 = 8 := by
  -- skip the actual proof steps
  sorry

end horner_v1_value_l68_68921


namespace time_to_cross_platform_is_correct_l68_68739

noncomputable def speed_of_train := 36 -- speed in km/h
noncomputable def time_to_cross_pole := 12 -- time in seconds
noncomputable def time_to_cross_platform := 49.996960243180546 -- time in seconds

theorem time_to_cross_platform_is_correct : time_to_cross_platform = 49.996960243180546 := by
  sorry

end time_to_cross_platform_is_correct_l68_68739


namespace complement_intersection_l68_68272

-- Define the universal set U and sets A and B.
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 4, 6}
def B : Set ℕ := {4, 5, 7}

-- Define the complements of A and B in U.
def C_UA : Set ℕ := U \ A
def C_UB : Set ℕ := U \ B

-- The proof problem: Prove that the intersection of the complements of A and B 
-- in the universal set U equals {2, 3, 8}.
theorem complement_intersection :
  (C_UA ∩ C_UB = {2, 3, 8}) := by
  sorry

end complement_intersection_l68_68272


namespace sequence_formula_l68_68627

theorem sequence_formula (a : ℕ → ℚ) (h1 : a 2 = 3 / 2) (h2 : a 3 = 7 / 3) 
  (h3 : ∀ n : ℕ, ∃ r : ℚ, (∀ m : ℕ, m ≥ 2 → (m * a m + 1) / (n * a n + 1) = r ^ (m - n))) :
  a n = (2^n - 1) / n := 
sorry

end sequence_formula_l68_68627


namespace favorite_numbers_parity_l68_68224

variables (D J A H : ℤ)

def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem favorite_numbers_parity
  (h1 : odd (D + 3 * J))
  (h2 : odd ((A - H) * 5))
  (h3 : even (D * H + 17)) :
  odd D ∧ even J ∧ even A ∧ odd H := 
sorry

end favorite_numbers_parity_l68_68224


namespace transport_equivalence_l68_68251

theorem transport_equivalence (f : ℤ → ℤ) (x y : ℤ) (h : f x = -x) :
  f (-y) = y :=
by
  sorry

end transport_equivalence_l68_68251


namespace flour_amount_l68_68252

theorem flour_amount (a b : ℕ) (h₁ : a = 8) (h₂ : b = 2) : a + b = 10 := by
  sorry

end flour_amount_l68_68252


namespace find_principal_sum_l68_68056

theorem find_principal_sum (P R : ℝ) (SI CI : ℝ) 
  (h1 : SI = 10200) 
  (h2 : CI = 11730) 
  (h3 : SI = P * R * 2 / 100)
  (h4 : CI = P * (1 + R / 100)^2 - P) :
  P = 17000 :=
by
  sorry

end find_principal_sum_l68_68056


namespace range_of_a_l68_68573

theorem range_of_a (x a : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) : -1 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l68_68573


namespace value_of_a_plus_b_l68_68188

noncomputable def f (a b x : ℝ) := x / (a * x + b)

theorem value_of_a_plus_b (a b : ℝ) (h₁: a ≠ 0) (h₂: f a b (-4) = 4)
    (h₃: ∀ x, f a b (f a b x) = x) : a + b = 3 / 2 :=
sorry

end value_of_a_plus_b_l68_68188


namespace fraction_zero_solve_l68_68020

theorem fraction_zero_solve (x : ℝ) (h : (x^2 - 49) / (x + 7) = 0) : x = 7 :=
by
  sorry

end fraction_zero_solve_l68_68020


namespace hyperbola_center_l68_68896

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (h1 : x1 = 6) (h2 : y1 = 3) (h3 : x2 = 10) (h4 : y2 = 7) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (8, 5) :=
by
  rw [h1, h2, h3, h4]
  simp
  -- Proof steps demonstrating the calculation
  -- simplify the arithmetic expressions
  sorry

end hyperbola_center_l68_68896


namespace adele_age_fraction_l68_68657

theorem adele_age_fraction 
  (jackson_age : ℕ) 
  (mandy_age : ℕ) 
  (adele_age_fraction : ℚ) 
  (total_age_10_years : ℕ)
  (H1 : jackson_age = 20)
  (H2 : mandy_age = jackson_age + 10)
  (H3 : total_age_10_years = (jackson_age + 10) + (mandy_age + 10) + (jackson_age * adele_age_fraction + 10))
  (H4 : total_age_10_years = 95) : 
  adele_age_fraction = 3 / 4 := 
sorry

end adele_age_fraction_l68_68657


namespace tournament_players_l68_68227

theorem tournament_players (n : ℕ) (h : n * (n - 1) / 2 = 56) : n = 14 :=
sorry

end tournament_players_l68_68227


namespace perpendicular_vectors_l68_68502

variable {t : ℝ}

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (6, -4)

theorem perpendicular_vectors (ht : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) : t = -5 :=
sorry

end perpendicular_vectors_l68_68502


namespace Maggie_apples_l68_68250

-- Definition of our problem conditions
def K : ℕ := 28 -- Kelsey's apples
def L : ℕ := 22 -- Layla's apples
def avg : ℕ := 30 -- The average number of apples picked

-- Main statement to prove Maggie's apples
theorem Maggie_apples : (A : ℕ) → (A + K + L) / 3 = avg → A = 40 := by
  intros A h
  -- sorry is added to skip the proof since it's not required here.
  sorry

end Maggie_apples_l68_68250


namespace three_digit_number_ends_same_sequence_l68_68421

theorem three_digit_number_ends_same_sequence (N : ℕ) (a b c : ℕ) (h1 : 100 ≤ N ∧ N < 1000)
  (h2 : N % 10 = c)
  (h3 : (N / 10) % 10 = b)
  (h4 : (N / 100) % 10 = a)
  (h5 : a ≠ 0)
  (h6 : N^2 % 1000 = N) :
  N = 127 :=
by
  sorry

end three_digit_number_ends_same_sequence_l68_68421


namespace number_of_workers_l68_68692

theorem number_of_workers (N C : ℕ) 
  (h1 : N * C = 300000) 
  (h2 : N * (C + 50) = 325000) : 
  N = 500 :=
sorry

end number_of_workers_l68_68692


namespace platform_protection_l68_68653

noncomputable def max_distance (r : ℝ) (n : ℕ) : ℝ :=
  if n > 2 then r / (Real.sin (180.0 / n)) else 0

noncomputable def coverage_ring_area (r : ℝ) (w : ℝ) : ℝ :=
  let inner_radius := r * (Real.sin 20.0)
  let outer_radius := inner_radius + w
  Real.pi * (outer_radius^2 - inner_radius^2)

theorem platform_protection :
  let r := 61
  let w := 22
  let n := 9
  max_distance r n = 60 / Real.sin 20.0 ∧
  coverage_ring_area r w = 2640 * Real.pi / Real.tan 20.0 := by
  sorry

end platform_protection_l68_68653


namespace negation_of_universal_l68_68667

open Classical

theorem negation_of_universal (P : ∀ x : ℤ, x^3 < 1) : ∃ x : ℤ, x^3 ≥ 1 :=
by sorry

end negation_of_universal_l68_68667


namespace bush_height_at_2_years_l68_68173

theorem bush_height_at_2_years (H: ℕ → ℕ) 
  (quadruple_height: ∀ (n: ℕ), H (n+1) = 4 * H n)
  (H_4: H 4 = 64) : H 2 = 4 :=
by
  sorry

end bush_height_at_2_years_l68_68173


namespace max_servings_possible_l68_68600

def number_of_servings
  (peppers cucumbers tomatoes : Nat) (brynza : Nat) : Nat :=
  min (peppers) (min (brynza / 75) (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings_possible :
  number_of_servings 60 117 116 4200 = 56 := 
by 
  -- sorry statement allows skipping the proof
  sorry

end max_servings_possible_l68_68600


namespace number_of_arrangements_l68_68327

theorem number_of_arrangements :
  ∃ (n k : ℕ), n = 10 ∧ k = 5 ∧ Nat.choose n k = 252 := by
  sorry

end number_of_arrangements_l68_68327


namespace smallest_four_digit_multiple_of_53_l68_68531

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l68_68531


namespace factored_quadratic_even_b_l68_68333

theorem factored_quadratic_even_b
  (c d e f y : ℤ)
  (h1 : c * e = 45)
  (h2 : d * f = 45) 
  (h3 : ∃ b, 45 * y^2 + b * y + 45 = (c * y + d) * (e * y + f)) :
  ∃ b, (45 * y^2 + b * y + 45 = (c * y + d) * (e * y + f)) ∧ (b % 2 = 0) :=
by
  sorry

end factored_quadratic_even_b_l68_68333


namespace sum_values_of_cubes_eq_l68_68089

theorem sum_values_of_cubes_eq :
  ∀ (a b : ℝ), a^3 + b^3 + 3 * a * b = 1 → a + b = 1 ∨ a + b = -2 :=
by
  intros a b h
  sorry

end sum_values_of_cubes_eq_l68_68089


namespace total_remaining_macaroons_l68_68297

-- Define initial macaroons count
def initial_red_macaroons : ℕ := 50
def initial_green_macaroons : ℕ := 40

-- Define macaroons eaten
def eaten_green_macaroons : ℕ := 15
def eaten_red_macaroons : ℕ := 2 * eaten_green_macaroons

-- Define remaining macaroons
def remaining_red_macaroons : ℕ := initial_red_macaroons - eaten_red_macaroons
def remaining_green_macaroons : ℕ := initial_green_macaroons - eaten_green_macaroons

-- Prove the total remaining macaroons
theorem total_remaining_macaroons : remaining_red_macaroons + remaining_green_macaroons = 45 := 
by
  -- Proof omitted
  sorry

end total_remaining_macaroons_l68_68297


namespace larger_number_is_299_l68_68983

theorem larger_number_is_299 (A B : ℕ) 
  (HCF_AB : Nat.gcd A B = 23) 
  (LCM_12_13 : Nat.lcm A B = 23 * 12 * 13) : 
  max A B = 299 := 
sorry

end larger_number_is_299_l68_68983


namespace female_athletes_drawn_is_7_l68_68655

-- Given conditions as definitions
def male_athletes := 64
def female_athletes := 56
def drawn_male_athletes := 8

-- The function that represents the equation in stratified sampling
def stratified_sampling_eq (x : Nat) : Prop :=
  (drawn_male_athletes : ℚ) / (male_athletes) = (x : ℚ) / (female_athletes)

-- The theorem which states that the solution to the problem is x = 7
theorem female_athletes_drawn_is_7 : ∃ x : Nat, stratified_sampling_eq x ∧ x = 7 :=
by
  sorry

end female_athletes_drawn_is_7_l68_68655


namespace height_of_box_l68_68375

def base_area : ℕ := 20 * 20
def cost_per_box : ℝ := 1.30
def total_volume : ℕ := 3060000
def amount_spent : ℝ := 663

theorem height_of_box : ∃ h : ℕ, 400 * h = total_volume / (amount_spent / cost_per_box) := sorry

end height_of_box_l68_68375


namespace min_value_l68_68197

-- Define the conditions
variables (x y : ℝ)
-- Assume x and y are in the positive real numbers
axiom pos_x : 0 < x
axiom pos_y : 0 < y
-- Given equation
axiom eq1 : x + 2 * y = 2 * x * y

-- The goal is to prove that the minimum value of 3x + 4y is 5 + 2sqrt(6)
theorem min_value (x y : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (eq1 : x + 2 * y = 2 * x * y) : 
  3 * x + 4 * y ≥ 5 + 2 * Real.sqrt 6 := 
sorry

end min_value_l68_68197


namespace min_largest_value_in_set_l68_68058

theorem min_largest_value_in_set (a b : ℕ) (h1 : 0 < a) (h2 : a < b) (h3 : (8:ℚ) / 19 * a * b ≤ (a - 1) * a / 2): a ≥ 13 :=
by
  sorry

end min_largest_value_in_set_l68_68058


namespace relationship_between_sets_l68_68868

def M : Set ℤ := {x | ∃ k : ℤ, x = 5 * k - 2}
def P : Set ℤ := {x | ∃ n : ℤ, x = 5 * n + 3}
def S : Set ℤ := {x | ∃ m : ℤ, x = 10 * m + 3}

theorem relationship_between_sets : S ⊆ P ∧ P = M := by
  sorry

end relationship_between_sets_l68_68868


namespace students_without_scholarships_l68_68091

theorem students_without_scholarships :
  let total_students := 300
  let full_merit_percent := 0.05
  let half_merit_percent := 0.10
  let sports_percent := 0.03
  let need_based_percent := 0.07
  let full_merit_and_sports_percent := 0.01
  let half_merit_and_need_based_percent := 0.02
  let full_merit := full_merit_percent * total_students
  let half_merit := half_merit_percent * total_students
  let sports := sports_percent * total_students
  let need_based := need_based_percent * total_students
  let full_merit_and_sports := full_merit_and_sports_percent * total_students
  let half_merit_and_need_based := half_merit_and_need_based_percent * total_students
  let total_with_scholarships := (full_merit + half_merit + sports + need_based) - (full_merit_and_sports + half_merit_and_need_based)
  let students_without_scholarships := total_students - total_with_scholarships
  students_without_scholarships = 234 := 
by
  sorry

end students_without_scholarships_l68_68091


namespace problem_1_problem_2_l68_68100

open Set

variables (a x : ℝ)

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a ^ 2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

theorem problem_1 (a : ℝ) (ha : a = 1) : 
  {x : ℝ | x^2 - 4 * a * x + 3 * a ^ 2 < 0} ∩ {x : ℝ | (x - 3) / (x - 2) ≤ 0} = Ioo 2 3 :=
sorry

theorem problem_2 (a : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 4 * a * x + 3 * a ^ 2 < 0) → ¬((x - 3) / (x - 2) ≤ 0)) →
  (∃ x : ℝ, ¬((x - 3) / (x - 2) ≤ 0) → ¬(x^2 - 4 * a * x + 3 * a ^ 2 < 0)) →
  1 < a ∧ a ≤ 2 :=
sorry

end problem_1_problem_2_l68_68100


namespace find_a_l68_68544

def A (x : ℝ) := (x^2 - 4 ≤ 0)
def B (x : ℝ) (a : ℝ) := (2 * x + a ≤ 0)
def C (x : ℝ) := (-2 ≤ x ∧ x ≤ 1)

theorem find_a (a : ℝ) : (∀ x : ℝ, A x → B x a → C x) → a = -2 :=
sorry

end find_a_l68_68544


namespace danny_bottle_caps_l68_68032

theorem danny_bottle_caps 
  (wrappers_park : Nat := 46)
  (caps_park : Nat := 50)
  (wrappers_collection : Nat := 52)
  (more_caps_than_wrappers : Nat := 4)
  (h1 : caps_park = wrappers_park + more_caps_than_wrappers)
  (h2 : wrappers_collection = 52) : 
  (∃ initial_caps : Nat, initial_caps + caps_park = wrappers_collection + more_caps_than_wrappers) :=
by 
  use 6
  sorry

end danny_bottle_caps_l68_68032


namespace probability_of_X_conditioned_l68_68872

variables (P_X P_Y P_XY : ℝ)

-- Conditions
def probability_of_Y : Prop := P_Y = 2/5
def probability_of_XY : Prop := P_XY = 0.05714285714285714
def independent_selection : Prop := P_XY = P_X * P_Y

-- Theorem statement
theorem probability_of_X_conditioned (P_X P_Y P_XY : ℝ) 
  (h1 : probability_of_Y P_Y) 
  (h2 : probability_of_XY P_XY) 
  (h3 : independent_selection P_X P_Y P_XY) :
  P_X = 0.14285714285714285 := 
sorry

end probability_of_X_conditioned_l68_68872


namespace triple_composition_l68_68927

def g (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition :
  g (g (g 3)) = 107 :=
by
  sorry

end triple_composition_l68_68927


namespace cos_4theta_l68_68186

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (4 * θ) = 17/81 :=
  sorry

end cos_4theta_l68_68186


namespace square_of_sum_l68_68244

theorem square_of_sum (x y : ℝ) (A B C D : ℝ) :
  A = 2 * x^2 + y^2 →
  B = 2 * (x + y)^2 →
  C = 2 * x + y^2 →
  D = (2 * x + y)^2 →
  D = (2 * x + y)^2 :=
by intros; exact ‹D = (2 * x + y)^2›

end square_of_sum_l68_68244


namespace least_number_of_square_tiles_l68_68994

theorem least_number_of_square_tiles (length : ℕ) (breadth : ℕ) (gcd : ℕ) (area_room : ℕ) (area_tile : ℕ) (num_tiles : ℕ) :
  length = 544 → breadth = 374 → gcd = Nat.gcd length breadth → gcd = 2 →
  area_room = length * breadth → area_tile = gcd * gcd →
  num_tiles = area_room / area_tile → num_tiles = 50864 :=
by
  sorry

end least_number_of_square_tiles_l68_68994


namespace remaining_students_l68_68617

def students_remaining (n1 n2 n_leaving1 n_leaving2 : Nat) : Nat :=
  (n1 * 4 - n_leaving1) + (n2 * 2 - n_leaving2)

theorem remaining_students :
  students_remaining 15 18 8 5 = 83 := 
by
  sorry

end remaining_students_l68_68617


namespace artwork_collection_l68_68072

theorem artwork_collection :
  ∀ (students quarters years artworks_per_student_per_quarter : ℕ), 
  students = 15 → quarters = 4 → years = 2 → artworks_per_student_per_quarter = 2 →
  students * artworks_per_student_per_quarter * quarters * years = 240 :=
by
  intros students quarters years artworks_per_student_per_quarter
  rintro (rfl : students = 15) (rfl : quarters = 4) (rfl : years = 2) (rfl : artworks_per_student_per_quarter = 2)
  sorry

end artwork_collection_l68_68072


namespace triangle_angles_l68_68860

theorem triangle_angles (r_a r_b r_c R : ℝ)
    (h1 : r_a + r_b = 3 * R)
    (h2 : r_b + r_c = 2 * R) :
    ∃ (A B C : ℝ), A = 30 ∧ B = 60 ∧ C = 90 :=
sorry

end triangle_angles_l68_68860


namespace hundred_days_from_friday_is_sunday_l68_68219

def days_from_friday (n : ℕ) : Nat :=
  (n + 5) % 7  -- 0 corresponds to Sunday, starting from Friday (5 + 0 % 7 = 5 which is Friday)

theorem hundred_days_from_friday_is_sunday :
  days_from_friday 100 = 0 := by
  sorry

end hundred_days_from_friday_is_sunday_l68_68219


namespace complex_problem_l68_68490

open Complex

theorem complex_problem
  (α θ β : ℝ)
  (h : exp (i * (α + θ)) + exp (i * (β + θ)) = 1 / 3 + (4 / 9) * i) :
  exp (-i * (α + θ)) + exp (-i * (β + θ)) = 1 / 3 - (4 / 9) * i :=
by
  sorry

end complex_problem_l68_68490


namespace probability_sum_leq_12_l68_68103

theorem probability_sum_leq_12 (dice1 dice2 : ℕ) (h1 : 1 ≤ dice1 ∧ dice1 ≤ 8) (h2 : 1 ≤ dice2 ∧ dice2 ≤ 8) :
  (∃ outcomes : ℕ, (outcomes = 64) ∧ 
   (∃ favorable : ℕ, (favorable = 54) ∧ 
   (favorable / outcomes = 27 / 32))) :=
sorry

end probability_sum_leq_12_l68_68103


namespace find_y_l68_68065

theorem find_y (x y : ℤ) (q : ℤ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x = q * y + 6) (h4 : (x : ℚ) / y = 96.15) : y = 40 :=
sorry

end find_y_l68_68065


namespace can_construct_parallelogram_l68_68473

theorem can_construct_parallelogram {a b d1 d2 : ℝ} :
  (a = 3 ∧ b = 5 ∧ (a = b ∨ (‖a + b‖ ≥ ‖d1‖ ∧ ‖a + d1‖ ≥ ‖b‖ ∧ ‖b + d1‖ ≥ ‖a‖))) ∨
  (a ≠ 3 ∨ b ≠ 5 ∨ (a ≠ b ∧ (‖a + b‖ < ‖d1‖ ∨ ‖a + d1‖ < ‖b‖ ∨ ‖b + d1‖ < ‖a‖ ∨ ‖a + d1‖ < ‖d2‖ ∨ ‖b + d1‖ < ‖d2‖ ∨ ‖a + d2‖ < ‖d1‖ ∨ ‖b + d2‖ < ‖d1‖))) ↔ 
  (a = 3 ∧ b = 5 ∧ d1 = 0) :=
sorry

end can_construct_parallelogram_l68_68473


namespace sqrt_diff_ineq_sum_sq_gt_sum_prod_l68_68665

-- First proof problem: Prove that sqrt(11) - 2 * sqrt(3) > 3 - sqrt(10)
theorem sqrt_diff_ineq : (Real.sqrt 11 - 2 * Real.sqrt 3) > (3 - Real.sqrt 10) := sorry

-- Second proof problem: Prove that a^2 + b^2 + c^2 > ab + bc + ca given a, b, and c are real numbers that are not all equal
theorem sum_sq_gt_sum_prod (a b c : ℝ) (h : ¬ (a = b ∧ b = c ∧ a = c)) : a^2 + b^2 + c^2 > a * b + b * c + c * a := sorry

end sqrt_diff_ineq_sum_sq_gt_sum_prod_l68_68665


namespace eulers_formula_l68_68288

-- Definitions related to simply connected polyhedra
def SimplyConnectedPolyhedron (V E F : ℕ) : Prop := true  -- Genus 0 implies it is simply connected

-- Euler's characteristic property for simply connected polyhedra
theorem eulers_formula (V E F : ℕ) (h : SimplyConnectedPolyhedron V E F) : V - E + F = 2 := 
by
  sorry

end eulers_formula_l68_68288


namespace xyz_sum_l68_68747

theorem xyz_sum (x y z : ℝ) 
  (h1 : y + z = 17 - 2 * x) 
  (h2 : x + z = 1 - 2 * y) 
  (h3 : x + y = 8 - 2 * z) : 
  x + y + z = 6.5 :=
sorry

end xyz_sum_l68_68747


namespace annie_job_time_l68_68495

noncomputable def annie_time : ℝ :=
  let dan_time := 15
  let dan_rate := 1 / dan_time
  let dan_hours := 6
  let fraction_done_by_dan := dan_rate * dan_hours
  let fraction_left_for_annie := 1 - fraction_done_by_dan
  let annie_work_remaining := fraction_left_for_annie
  let annie_hours := 6
  let annie_rate := annie_work_remaining / annie_hours
  let annie_time := 1 / annie_rate 
  annie_time

theorem annie_job_time :
  annie_time = 3.6 := 
sorry

end annie_job_time_l68_68495


namespace expected_value_of_draws_before_stopping_l68_68808

noncomputable def totalBalls := 10
noncomputable def redBalls := 2
noncomputable def whiteBalls := 8

noncomputable def prob_one_draw_white : ℚ := whiteBalls / totalBalls
noncomputable def prob_two_draws_white : ℚ := (redBalls / totalBalls) * (whiteBalls / (totalBalls - 1))
noncomputable def prob_three_draws_white : ℚ := (redBalls / (totalBalls - redBalls + 1)) * ((redBalls - 1) / (totalBalls - 1)) * (whiteBalls / (totalBalls - 2))

noncomputable def expected_draws_before_white : ℚ :=
  1 * prob_one_draw_white + 2 * prob_two_draws_white + 3 * prob_three_draws_white

theorem expected_value_of_draws_before_stopping : expected_draws_before_white = 11 / 9 := by
  sorry

end expected_value_of_draws_before_stopping_l68_68808


namespace f_1996x_eq_1996_f_x_l68_68406

theorem f_1996x_eq_1996_f_x (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
by
  sorry

end f_1996x_eq_1996_f_x_l68_68406


namespace proof_evaluate_expression_l68_68278

def evaluate_expression : Prop :=
  - (18 / 3 * 8 - 72 + 4 * 8) = 8

theorem proof_evaluate_expression : evaluate_expression :=
by 
  sorry

end proof_evaluate_expression_l68_68278


namespace exists_sum_coprime_seventeen_not_sum_coprime_l68_68813

/-- 
 For any integer \( n \) where \( n > 17 \), there exist integers \( a \) and \( b \) 
 such that \( n = a + b \), \( a > 1 \), \( b > 1 \), and \( \gcd(a, b) = 1 \).
 Additionally, the integer 17 does not have this property.
-/
theorem exists_sum_coprime (n : ℤ) (h : n > 17) : 
  ∃ (a b : ℤ), n = a + b ∧ a > 1 ∧ b > 1 ∧ Int.gcd a b = 1 :=
sorry

/-- 
 The integer 17 cannot be expressed as the sum of two integers greater than 1 
 that are coprime.
-/
theorem seventeen_not_sum_coprime : 
  ¬ ∃ (a b : ℤ), 17 = a + b ∧ a > 1 ∧ b > 1 ∧ Int.gcd a b = 1 :=
sorry

end exists_sum_coprime_seventeen_not_sum_coprime_l68_68813


namespace determine_f_5_l68_68326

theorem determine_f_5 (f : ℝ → ℝ) (h1 : f 1 = 3) 
  (h2 : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x + f y)) : f 5 = 45 :=
sorry

end determine_f_5_l68_68326


namespace sum_of_consecutive_integers_l68_68071

theorem sum_of_consecutive_integers (n : ℕ) (h : n*(n + 1) = 2720) : n + (n + 1) = 103 :=
sorry

end sum_of_consecutive_integers_l68_68071


namespace quadratic_inequality_condition_l68_68543

theorem quadratic_inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + bx + c < 0) ↔ (a < 0 ∧ b^2 - 4 * a * c < 0) :=
sorry

end quadratic_inequality_condition_l68_68543


namespace jen_lisa_spent_l68_68820

theorem jen_lisa_spent (J L : ℝ) 
  (h1 : L = 0.8 * J) 
  (h2 : J = L + 15) : 
  J + L = 135 := 
by
  sorry

end jen_lisa_spent_l68_68820


namespace remainder_pow_700_eq_one_l68_68630

theorem remainder_pow_700_eq_one (number : ℤ) (h : number ^ 700 % 100 = 1) : number ^ 700 % 100 = 1 :=
  by
  exact h

end remainder_pow_700_eq_one_l68_68630


namespace weather_forecast_minutes_l68_68852

theorem weather_forecast_minutes 
  (total_duration : ℕ) 
  (national_news : ℕ) 
  (international_news : ℕ) 
  (sports : ℕ) 
  (advertising : ℕ) 
  (wf : ℕ) :
  total_duration = 30 →
  national_news = 12 →
  international_news = 5 →
  sports = 5 →
  advertising = 6 →
  total_duration - (national_news + international_news + sports + advertising) = wf →
  wf = 2 :=
by
  intros
  sorry

end weather_forecast_minutes_l68_68852


namespace div_scaled_result_l68_68465

theorem div_scaled_result :
  (2994 : ℝ) / 14.5 = 171 :=
by
  have cond1 : (29.94 : ℝ) / 1.45 = 17.1 := sorry
  have cond2 : (2994 : ℝ) = 100 * 29.94 := sorry
  have cond3 : (14.5 : ℝ) = 10 * 1.45 := sorry
  sorry

end div_scaled_result_l68_68465


namespace height_difference_l68_68736

def empireStateBuildingHeight : ℕ := 443
def petronasTowersHeight : ℕ := 452

theorem height_difference :
  petronasTowersHeight - empireStateBuildingHeight = 9 := 
sorry

end height_difference_l68_68736


namespace delta_value_l68_68062

noncomputable def delta : ℝ :=
  Real.arccos (
    (Finset.range 3600).sum (fun k => Real.sin ((2539 + k) * Real.pi / 180)) ^ Real.cos (2520 * Real.pi / 180) +
    (Finset.range 3599).sum (fun k => Real.cos ((2521 + k) * Real.pi / 180)) +
    Real.cos (6120 * Real.pi / 180)
  )

theorem delta_value : delta = 71 :=
by
  sorry

end delta_value_l68_68062


namespace exists_nat_concat_is_perfect_square_l68_68355

theorem exists_nat_concat_is_perfect_square :
  ∃ A : ℕ, ∃ n : ℕ, ∃ B : ℕ, (B * B = (10^n + 1) * A) :=
by sorry

end exists_nat_concat_is_perfect_square_l68_68355


namespace total_distance_traveled_l68_68891

-- Definitions of distances in km
def ZX : ℝ := 4000
def XY : ℝ := 5000
def YZ : ℝ := (XY^2 - ZX^2)^(1/2)

-- Prove the total distance traveled
theorem total_distance_traveled :
  XY + YZ + ZX = 11500 := by
  have h1 : ZX = 4000 := rfl
  have h2 : XY = 5000 := rfl
  have h3 : YZ = (5000^2 - 4000^2)^(1/2) := rfl
  -- Continue the proof showing the calculation of each step
  sorry

end total_distance_traveled_l68_68891


namespace cross_product_example_l68_68700

def vector_cross (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (u.2.1 * v.2.2 - u.2.2 * v.2.1, 
   u.2.2 * v.1 - u.1 * v.2.2, 
   u.1 * v.1 - u.2.1 * v.1)
   
theorem cross_product_example : 
  vector_cross (4, 3, -7) (2, 0, 5) = (15, -34, -6) :=
by
  -- The proof will go here
  sorry

end cross_product_example_l68_68700


namespace pairings_equal_l68_68778

-- Definitions for City A
def A_girls (n : ℕ) : Type := Fin n
def A_boys (n : ℕ) : Type := Fin n
def A_knows (n : ℕ) (g : A_girls n) (b : A_boys n) : Prop := True

-- Definitions for City B
def B_girls (n : ℕ) : Type := Fin n
def B_boys (n : ℕ) : Type := Fin (2 * n - 1)
def B_knows (n : ℕ) (i : Fin n) (j : Fin (2 * n - 1)) : Prop :=
  j.val < 2 * (i.val + 1)

-- Function to count the number of ways to pair r girls and r boys in city A
noncomputable def A (n r : ℕ) : ℕ := 
  if h : r ≤ n then 
    Nat.choose n r * Nat.choose n r * (r.factorial)
  else 0

-- Recurrence relation for city B
noncomputable def B (n r : ℕ) : ℕ :=
  if r = 0 then 1 else if r > n then 0 else
  if n < 2 then if r = 1 then (2 - 1) * 2 else 0 else
  B (n - 1) r + (2 * n - r) * B (n - 1) (r - 1)

-- We want to prove that number of pairings in city A equals number of pairings in city B for any r <= n
theorem pairings_equal (n r : ℕ) (h : r ≤ n) : A n r = B n r := sorry

end pairings_equal_l68_68778


namespace find_k_l68_68102

theorem find_k (k : ℝ) (x₁ x₂ : ℝ)
  (h : x₁^2 + (2 * k - 1) * x₁ + k^2 - 1 = 0)
  (h' : x₂^2 + (2 * k - 1) * x₂ + k^2 - 1 = 0)
  (hx : x₁ ≠ x₂)
  (cond : x₁^2 + x₂^2 = 19) : k = -2 :=
sorry

end find_k_l68_68102


namespace sufficient_but_not_necessary_condition_l68_68976

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1 / 8) → (∀ x : ℝ, x > 0 → 2 * x + a / x ≥ 1) :=
by sorry

end sufficient_but_not_necessary_condition_l68_68976


namespace remainder_when_dividing_p_by_g_is_3_l68_68492

noncomputable def p (x : ℤ) : ℤ := x^5 - 2 * x^3 + 4 * x^2 + x + 5
noncomputable def g (x : ℤ) : ℤ := x + 2

theorem remainder_when_dividing_p_by_g_is_3 : p (-2) = 3 :=
by
  sorry

end remainder_when_dividing_p_by_g_is_3_l68_68492


namespace a_2016_mod_2017_l68_68351

-- Defining the sequence
def seq (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧
  a 1 = 2 ∧
  ∀ n, a (n + 2) = 2 * a (n + 1) + 41 * a n

theorem a_2016_mod_2017 (a : ℕ → ℕ) (h : seq a) : 
  a 2016 % 2017 = 0 := 
sorry

end a_2016_mod_2017_l68_68351


namespace simplify_expression_l68_68364

theorem simplify_expression (w : ℝ) : 2 * w + 4 * w + 6 * w + 8 * w + 10 * w + 12 = 30 * w + 12 :=
by
  sorry

end simplify_expression_l68_68364


namespace solve_equation_l68_68304

theorem solve_equation :
  (∀ x : ℝ, x ≠ 2 / 3 → (7 * x + 3) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) → (x = 1 / 3) ∨ (x = -3)) :=
by
  sorry

end solve_equation_l68_68304


namespace initial_trucks_l68_68855

def trucks_given_to_Jeff : ℕ := 13
def trucks_left_with_Sarah : ℕ := 38

theorem initial_trucks (initial_trucks_count : ℕ) :
  initial_trucks_count = trucks_given_to_Jeff + trucks_left_with_Sarah → initial_trucks_count = 51 :=
by
  sorry

end initial_trucks_l68_68855


namespace line_relation_in_perpendicular_planes_l68_68837

-- Let's define the notions of planes and lines being perpendicular/parallel
variables {α β : Plane} {a : Line}

def plane_perpendicular (α β : Plane) : Prop := sorry -- definition of perpendicular planes
def line_perpendicular_plane (a : Line) (β : Plane) : Prop := sorry -- definition of a line being perpendicular to a plane
def line_parallel_plane (a : Line) (α : Plane) : Prop := sorry -- definition of a line being parallel to a plane
def line_in_plane (a : Line) (α : Plane) : Prop := sorry -- definition of a line lying in a plane

-- The theorem stating the relationship given the conditions
theorem line_relation_in_perpendicular_planes 
  (h1 : plane_perpendicular α β) 
  (h2 : line_perpendicular_plane a β) : 
  line_parallel_plane a α ∨ line_in_plane a α :=
sorry

end line_relation_in_perpendicular_planes_l68_68837


namespace range_of_a_l68_68610

theorem range_of_a (a : ℝ) :
  (¬ ∃ t : ℝ, t^2 - 2 * t - a < 0) ↔ a ≤ -1 :=
by sorry

end range_of_a_l68_68610


namespace sum_abc_l68_68869

theorem sum_abc (a b c: ℝ) 
  (h1 : ∃ x: ℝ, x^2 + a * x + 1 = 0 ∧ x^2 + b * x + c = 0)
  (h2 : ∃ x: ℝ, x^2 + x + a = 0 ∧ x^2 + c * x + b = 0) :
  a + b + c = -3 := 
sorry

end sum_abc_l68_68869


namespace rectangular_area_length_width_l68_68456

open Nat

theorem rectangular_area_length_width (lengthInMeters widthInMeters : ℕ) (h1 : lengthInMeters = 500) (h2 : widthInMeters = 60) :
  (lengthInMeters * widthInMeters = 30000) ∧ ((lengthInMeters * widthInMeters) / 10000 = 3) :=
by
  sorry

end rectangular_area_length_width_l68_68456


namespace exponent_multiplication_l68_68908

theorem exponent_multiplication (m n : ℕ) (h : m + n = 3) : 2^m * 2^n = 8 := 
by
  sorry

end exponent_multiplication_l68_68908


namespace solve_system_l68_68292

theorem solve_system : ∀ (a b : ℝ), (∃ (x y : ℝ), x = 5 ∧ y = b ∧ 2 * x + y = a ∧ 2 * x - y = 12) → (a = 8 ∧ b = -2) :=
by
  sorry

end solve_system_l68_68292


namespace victor_percentage_of_marks_l68_68505

theorem victor_percentage_of_marks (marks_obtained max_marks : ℝ) (percentage : ℝ) 
  (h_marks_obtained : marks_obtained = 368) 
  (h_max_marks : max_marks = 400) 
  (h_percentage : percentage = (marks_obtained / max_marks) * 100) : 
  percentage = 92 := by
sorry

end victor_percentage_of_marks_l68_68505


namespace algebraic_expression_value_l68_68581

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 3*x + 1 = 4) : 2*x^2 - 6*x + 5 = 11 :=
by
  sorry

end algebraic_expression_value_l68_68581


namespace y_explicit_and_range_l68_68782

theorem y_explicit_and_range (m : ℝ) (x1 x2 : ℝ) (h1 : x1^2 - 2*(m-1)*x1 + m + 1 = 0) (h2 : x2^2 - 2*(m-1)*x2 + m + 1 = 0) :
  x1 + x2 = 2*(m-1) ∧ x1 * x2 = m + 1 ∧ (x1^2 + x2^2 = 4*m^2 - 10*m + 2) 
  ∧ ∀ (y : ℝ), (∃ m, y = 4*m^2 - 10*m + 2) → y ≥ 6 :=
by
  sorry

end y_explicit_and_range_l68_68782


namespace three_bodies_with_triangle_front_view_l68_68804

def has_triangle_front_view (b : Type) : Prop :=
  -- Placeholder definition for example purposes
  sorry

theorem three_bodies_with_triangle_front_view :
  ∃ (body1 body2 body3 : Type),
  has_triangle_front_view body1 ∧
  has_triangle_front_view body2 ∧
  has_triangle_front_view body3 :=
sorry

end three_bodies_with_triangle_front_view_l68_68804


namespace negation_correct_l68_68046

-- Define the original proposition
def original_proposition (x : ℝ) : Prop :=
  ∀ x > 0, x^2 - 2 * x + 1 ≥ 0

-- Define what it means to negate the proposition
def negated_proposition (x : ℝ) : Prop :=
  ∃ x > 0, x^2 - 2 * x + 1 < 0

-- Main statement: the negation of the original proposition equals the negated proposition
theorem negation_correct : (¬original_proposition x) = (negated_proposition x) :=
  sorry

end negation_correct_l68_68046


namespace find_number_l68_68006

theorem find_number (x : ℕ) (h : x = 4) : x + 1 = 5 :=
by
  sorry

end find_number_l68_68006


namespace friends_can_reach_destinations_l68_68295

/-- The distance between Coco da Selva and Quixajuba is 24 km. 
    The walking speed is 6 km/h and the biking speed is 18 km/h. 
    Show that the friends can proceed to reach their destinations in at most 2 hours 40 minutes, with the bicycle initially in Quixajuba. -/
theorem friends_can_reach_destinations (d q c : ℕ) (vw vb : ℕ) (h1 : d = 24) (h2 : vw = 6) (h3 : vb = 18): 
  (∃ ta tb tc : ℕ, ta ≤ 2 * 60 + 40 ∧ tb ≤ 2 * 60 + 40 ∧ tc ≤ 2 * 60 + 40 ∧ 
     True) :=
sorry

end friends_can_reach_destinations_l68_68295


namespace solve_x_l68_68059

theorem solve_x :
  ∃ x : ℝ, 2.5 * ( ( x * 0.48 * 2.50 ) / ( 0.12 * 0.09 * 0.5 ) ) = 2000.0000000000002 ∧ x = 3.6 :=
by sorry

end solve_x_l68_68059


namespace avg_lottery_draws_eq_5232_l68_68039

def avg_lottery_draws (n m : ℕ) : ℕ :=
  let N := 90 * 89 * 88 * 87 * 86
  let Nk := 25 * 40320
  N / Nk

theorem avg_lottery_draws_eq_5232 : avg_lottery_draws 90 5 = 5232 :=
by 
  unfold avg_lottery_draws
  sorry

end avg_lottery_draws_eq_5232_l68_68039


namespace XiaoMingAgeWhenFathersAgeIsFiveTimes_l68_68831

-- Define the conditions
def XiaoMingAgeCurrent : ℕ := 12
def FatherAgeCurrent : ℕ := 40

-- Prove the question given the conditions
theorem XiaoMingAgeWhenFathersAgeIsFiveTimes : 
  ∃ (x : ℕ), (FatherAgeCurrent - x) = 5 * x - XiaoMingAgeCurrent ∧ x = 7 := 
by
  use 7
  sorry

end XiaoMingAgeWhenFathersAgeIsFiveTimes_l68_68831


namespace equilateral_triangle_ab_l68_68414

noncomputable def a : ℝ := 25 * Real.sqrt 3
noncomputable def b : ℝ := 5 * Real.sqrt 3

theorem equilateral_triangle_ab
  (a_val : a = 25 * Real.sqrt 3)
  (b_val : b = 5 * Real.sqrt 3)
  (h1 : Complex.abs (a + 15 * Complex.I) = 25)
  (h2 : Complex.abs (b + 45 * Complex.I) = 45)
  (h3 : Complex.abs ((a - b) + (15 - 45) * Complex.I) = 30) :
  a * b = 375 := 
sorry

end equilateral_triangle_ab_l68_68414


namespace extra_fruits_l68_68658

theorem extra_fruits (r g s : Nat) (hr : r = 42) (hg : g = 7) (hs : s = 9) : r + g - s = 40 :=
by
  sorry

end extra_fruits_l68_68658


namespace meanScore_is_91_666_l68_68777

-- Define Jane's quiz scores
def janesScores : List ℕ := [85, 88, 90, 92, 95, 100]

-- Define the total sum of Jane's quiz scores
def sumScores (scores : List ℕ) : ℕ := scores.foldl (· + ·) 0

-- The number of Jane's quiz scores
def numberOfScores (scores : List ℕ) : ℕ := scores.length

-- Define the mean of Jane's quiz scores
def meanScore (scores : List ℕ) : ℚ := sumScores scores / numberOfScores scores

-- The theorem to be proven
theorem meanScore_is_91_666 (h : janesScores = [85, 88, 90, 92, 95, 100]) :
  meanScore janesScores = 91.66666666666667 := by 
  sorry

end meanScore_is_91_666_l68_68777


namespace problem_statement_l68_68004

-- Defining the sets U, M, and N
def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

-- Complement of N in U
def complement_U_N : Set ℕ := U \ N

-- Problem statement
theorem problem_statement : M ∩ complement_U_N = {0, 3} :=
by
  sorry

end problem_statement_l68_68004


namespace find_k_l68_68969

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (2, -3)

def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_k (k : ℝ) :
  is_perpendicular (k • vector_a - 2 • vector_b) vector_a ↔ k = -1 :=
sorry

end find_k_l68_68969


namespace cost_per_pound_of_sausages_l68_68818

/-- Jake buys 2-pound packages of sausages. He buys 3 packages. He pays $24. 
To find the cost per pound of sausages. --/
theorem cost_per_pound_of_sausages 
  (pkg_weight : ℕ) 
  (num_pkg : ℕ) 
  (total_cost : ℕ) 
  (cost_per_pound : ℕ) 
  (h_pkg_weight : pkg_weight = 2) 
  (h_num_pkg : num_pkg = 3) 
  (h_total_cost : total_cost = 24) 
  (h_total_weight : num_pkg * pkg_weight = 6) :
  total_cost / (num_pkg * pkg_weight) = cost_per_pound :=
sorry

end cost_per_pound_of_sausages_l68_68818


namespace no_polynomial_transformation_l68_68141

-- Define the problem conditions: initial and target sequences
def initial_seq : List ℤ := [-3, -1, 1, 3]
def target_seq : List ℤ := [-3, -1, -3, 3]

-- State the main theorem to be proved
theorem no_polynomial_transformation :
  ¬ (∃ (P : ℤ → ℤ), ∀ x ∈ initial_seq, P x ∈ target_seq) :=
  sorry

end no_polynomial_transformation_l68_68141


namespace atleast_one_alarm_rings_on_time_l68_68107

def probability_alarm_A_rings := 0.80
def probability_alarm_B_rings := 0.90

def probability_atleast_one_rings := 1 - (1 - probability_alarm_A_rings) * (1 - probability_alarm_B_rings)

theorem atleast_one_alarm_rings_on_time :
  probability_atleast_one_rings = 0.98 :=
sorry

end atleast_one_alarm_rings_on_time_l68_68107


namespace student_incorrect_answer_l68_68028

theorem student_incorrect_answer (D I : ℕ) (h1 : D / 63 = I) (h2 : D / 36 = 42) : I = 24 := by
  sorry

end student_incorrect_answer_l68_68028


namespace gcd_m_n_l68_68086

def m : ℕ := 131^2 + 243^2 + 357^2
def n : ℕ := 130^2 + 242^2 + 358^2

theorem gcd_m_n : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_l68_68086


namespace parallel_lines_sufficient_not_necessary_l68_68255

theorem parallel_lines_sufficient_not_necessary (a : ℝ) :
  ((a = 3) → (∀ x y : ℝ, (a * x + 2 * y + 1 = 0) → (3 * x + (a - 1) * y - 2 = 0)) ∧ 
  (∀ x y : ℝ, (a * x + 2 * y + 1 = 0) ∧ (3 * x + (a - 1) * y - 2 = 0) → (a = 3 ∨ a = -2))) :=
sorry

end parallel_lines_sufficient_not_necessary_l68_68255


namespace smallest_n_for_divisibility_l68_68269

theorem smallest_n_for_divisibility (n : ℕ) : 
  (∀ m, m > 0 → (315^2 - m^2) ∣ (315^3 - m^3) → m ≥ n) → 
  (315^2 - n^2) ∣ (315^3 - n^3) → 
  n = 90 :=
by
  sorry

end smallest_n_for_divisibility_l68_68269


namespace cos_sin_ratio_l68_68721

open Real

-- Given conditions
variables {α β : Real}
axiom tan_alpha_beta : tan (α + β) = 2 / 5
axiom tan_beta_pi_over_4 : tan (β - π / 4) = 1 / 4

-- Theorem to be proven
theorem cos_sin_ratio (hαβ : tan (α + β) = 2 / 5) (hβ : tan (β - π / 4) = 1 / 4) :
  (cos α + sin α) / (cos α - sin α) = 3 / 22 :=
sorry

end cos_sin_ratio_l68_68721


namespace ben_paperclip_day_l68_68049

theorem ben_paperclip_day :
  ∃ k : ℕ, k = 6 ∧ (∀ n : ℕ, n = k → 5 * 3^n > 500) :=
sorry

end ben_paperclip_day_l68_68049


namespace sally_popped_3_balloons_l68_68594

-- Defining the conditions
def joans_initial_balloons : ℕ := 9
def jessicas_balloons : ℕ := 2
def total_balloons_now : ℕ := 6

-- Definition for the number of balloons Sally popped
def sally_balloons_popped : ℕ := joans_initial_balloons - (total_balloons_now - jessicas_balloons)

-- The theorem statement
theorem sally_popped_3_balloons : sally_balloons_popped = 3 := 
by
  -- Proof omitted; use sorry
  sorry

end sally_popped_3_balloons_l68_68594


namespace heart_digit_proof_l68_68372

noncomputable def heart_digit : ℕ := 3

theorem heart_digit_proof (heartsuit : ℕ) (h : heartsuit * 9 + 6 = heartsuit * 10 + 3) : 
  heartsuit = heart_digit := 
by
  sorry

end heart_digit_proof_l68_68372


namespace roots_squared_sum_eq_13_l68_68510

/-- Let p and q be the roots of the quadratic equation x^2 - 5x + 6 = 0. Then the value of p^2 + q^2 is 13. -/
theorem roots_squared_sum_eq_13 (p q : ℝ) (h₁ : p + q = 5) (h₂ : p * q = 6) : p^2 + q^2 = 13 :=
by
  sorry

end roots_squared_sum_eq_13_l68_68510


namespace gcd_1911_1183_l68_68001

theorem gcd_1911_1183 : gcd 1911 1183 = 91 :=
by sorry

end gcd_1911_1183_l68_68001


namespace distinct_integers_sum_to_32_l68_68664

theorem distinct_integers_sum_to_32 
  (p q r s t : ℤ)
  (h_diff : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t)
  (h_eq : (9 - p) * (9 - q) * (9 - r) * (9 - s) * (9 - t) = -120) : 
  p + q + r + s + t = 32 := 
by 
  sorry

end distinct_integers_sum_to_32_l68_68664


namespace total_apples_packed_correct_l68_68999

-- Define the daily production of apples under normal conditions
def apples_per_box := 40
def boxes_per_day := 50
def days_per_week := 7
def apples_per_day := apples_per_box * boxes_per_day

-- Define the change in daily production for the next week
def fewer_apples := 500
def apples_per_day_next_week := apples_per_day - fewer_apples

-- Define the weekly production in normal and next conditions
def apples_first_week := apples_per_day * days_per_week
def apples_second_week := apples_per_day_next_week * days_per_week

-- Define the total apples packed in two weeks
def total_apples_packed := apples_first_week + apples_second_week

-- Prove the total apples packed is 24500
theorem total_apples_packed_correct : total_apples_packed = 24500 := by
  sorry

end total_apples_packed_correct_l68_68999


namespace distribute_pencils_l68_68977

variables {initial_pencils : ℕ} {num_containers : ℕ} {additional_pencils : ℕ}

theorem distribute_pencils (h₁ : initial_pencils = 150) (h₂ : num_containers = 5)
                           (h₃ : additional_pencils = 30) :
  (initial_pencils + additional_pencils) / num_containers = 36 :=
by sorry

end distribute_pencils_l68_68977


namespace Smiths_Backery_Pies_l68_68468

theorem Smiths_Backery_Pies : 
  ∀ (p : ℕ), (∃ (q : ℕ), q = 16 ∧ p = 4 * q + 6) → p = 70 :=
by
  intros p h
  cases' h with q hq
  cases' hq with hq1 hq2
  rw [hq1] at hq2
  have h_eq : p = 4 * 16 + 6 := hq2
  norm_num at h_eq
  exact h_eq

end Smiths_Backery_Pies_l68_68468


namespace merchant_salt_mixture_l68_68870

theorem merchant_salt_mixture (x : ℝ) (h₀ : (0.48 * (40 + x)) = 1.20 * (14 + 0.50 * x)) : x = 0 :=
by
  sorry

end merchant_salt_mixture_l68_68870


namespace intersection_unique_point_l68_68225

def f (x : ℝ) : ℝ := x^3 + 6 * x^2 + 16 * x + 28

theorem intersection_unique_point :
  ∃ a : ℝ, f a = a ∧ a = -4 := sorry

end intersection_unique_point_l68_68225


namespace min_ab_is_2sqrt6_l68_68965

noncomputable def min_ab (a b : ℝ) : ℝ :=
  if h : (a > 0) ∧ (b > 0) ∧ ((2 / a) + (3 / b) = Real.sqrt (a * b)) then
      2 * Real.sqrt 6
  else
      0 -- or any other value, since this case should not occur in the context

theorem min_ab_is_2sqrt6 {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : (2 / a) + (3 / b) = Real.sqrt (a * b)) :
  min_ab a b = 2 * Real.sqrt 6 := 
by
  sorry

end min_ab_is_2sqrt6_l68_68965


namespace simplify_expression_l68_68512

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ :=
by
  sorry

end simplify_expression_l68_68512


namespace smallest_integer_k_l68_68536

theorem smallest_integer_k : ∀ (k : ℕ), (64^k > 4^16) → k ≥ 6 :=
by
  sorry

end smallest_integer_k_l68_68536


namespace batsman_average_after_12th_innings_l68_68136

theorem batsman_average_after_12th_innings 
  (A : ℕ) 
  (total_runs_11_innings : ℕ := 11 * A) 
  (new_average : ℕ := A + 2) 
  (total_runs_12_innings : ℕ := total_runs_11_innings + 92) 
  (increased_average_after_12 : 12 * new_average = total_runs_12_innings) 
  : new_average = 70 := 
by
  -- skipping proof
  sorry

end batsman_average_after_12th_innings_l68_68136


namespace Alpha_Beta_meet_at_Alpha_Beta_meet_again_l68_68209

open Real

-- Definitions and conditions
def A : ℝ := -24
def B : ℝ := -10
def C : ℝ := 10
def Alpha_speed : ℝ := 4
def Beta_speed : ℝ := 6

-- Question 1: Prove that Alpha and Beta meet at -10.4
theorem Alpha_Beta_meet_at : 
  ∃ t : ℝ, (A + Alpha_speed * t = C - Beta_speed * t) ∧ (A + Alpha_speed * t = -10.4) :=
  sorry

-- Question 2: Prove that after reversing at t = 2, Alpha and Beta meet again at -44
theorem Alpha_Beta_meet_again :
  ∃ t z : ℝ, 
    ((t = 2) ∧ (4 * t + (14 - 4 * t) + (14 - 4 * t + 20) = 40) ∧ 
     (A + Alpha_speed * t - Alpha_speed * z = C - Beta_speed * t - Beta_speed * z) ∧ 
     (A + Alpha_speed * t - Alpha_speed * z = -44)) :=
  sorry  

end Alpha_Beta_meet_at_Alpha_Beta_meet_again_l68_68209


namespace last_person_teeth_removed_l68_68511

-- Define the initial conditions
def total_teeth : ℕ := 32
def total_removed : ℕ := 40
def first_person_removed : ℕ := total_teeth * 1 / 4
def second_person_removed : ℕ := total_teeth * 3 / 8
def third_person_removed : ℕ := total_teeth * 1 / 2

-- Express the problem in Lean
theorem last_person_teeth_removed : 
  first_person_removed + second_person_removed + third_person_removed + last_person_removed = total_removed →
  last_person_removed = 4 := 
by
  sorry

end last_person_teeth_removed_l68_68511


namespace parabola_symmetric_points_l68_68416

theorem parabola_symmetric_points (a : ℝ) (h : 0 < a) :
  (∃ (P Q : ℝ × ℝ), (P ≠ Q) ∧ ((P.fst + P.snd = 0) ∧ (Q.fst + Q.snd = 0)) ∧
    (P.snd = a * P.fst ^ 2 - 1) ∧ (Q.snd = a * Q.fst ^ 2 - 1)) ↔ (3 / 4 < a) := 
sorry

end parabola_symmetric_points_l68_68416


namespace latoya_call_duration_l68_68805

theorem latoya_call_duration
  (initial_credit remaining_credit : ℝ) (cost_per_minute : ℝ) (t : ℝ)
  (h1 : initial_credit = 30)
  (h2 : remaining_credit = 26.48)
  (h3 : cost_per_minute = 0.16)
  (h4 : initial_credit - remaining_credit = t * cost_per_minute) :
  t = 22 := 
sorry

end latoya_call_duration_l68_68805


namespace height_of_parallelogram_l68_68270

def area_of_parallelogram (base height : ℝ) : ℝ := base * height

theorem height_of_parallelogram (A B H : ℝ) (hA : A = 33.3) (hB : B = 9) (hAparallelogram : A = area_of_parallelogram B H) :
  H = 3.7 :=
by 
  -- Proof would go here
  sorry

end height_of_parallelogram_l68_68270


namespace positive_number_solution_l68_68385

theorem positive_number_solution (x : ℚ) (hx : 0 < x) (h : x * x^2 * (1 / x) = 100 / 81) : x = 10 / 9 :=
sorry

end positive_number_solution_l68_68385


namespace normal_mean_is_zero_if_symmetric_l68_68133

-- Definition: A normal distribution with mean μ and standard deviation σ.
structure NormalDist where
  μ : ℝ
  σ : ℝ

-- Condition: The normal curve is symmetric about the y-axis.
def symmetric_about_y_axis (nd : NormalDist) : Prop :=
  nd.μ = 0

-- Theorem: If the normal curve is symmetric about the y-axis, then the mean μ of the corresponding normal distribution is 0.
theorem normal_mean_is_zero_if_symmetric (nd : NormalDist) (h : symmetric_about_y_axis nd) : nd.μ = 0 := 
by sorry

end normal_mean_is_zero_if_symmetric_l68_68133


namespace election_winner_votes_difference_l68_68895

theorem election_winner_votes_difference (V : ℝ) (h1 : 0.62 * V = 1054) : 0.24 * V = 408 :=
by
  sorry

end election_winner_votes_difference_l68_68895


namespace sum_of_coefficients_polynomial_expansion_l68_68557

theorem sum_of_coefficients_polynomial_expansion :
  let polynomial := (2 * (1 : ℤ) + 3)^5
  ∃ b_5 b_4 b_3 b_2 b_1 b_0 : ℤ,
  polynomial = b_5 * 1^5 + b_4 * 1^4 + b_3 * 1^3 + b_2 * 1^2 + b_1 * 1 + b_0 ∧
  (b_5 + b_4 + b_3 + b_2 + b_1 + b_0) = 3125 :=
by
  sorry

end sum_of_coefficients_polynomial_expansion_l68_68557


namespace arithmetic_sequence_solution_l68_68662

theorem arithmetic_sequence_solution
  (a b c : ℤ)
  (h1 : a + 1 = b - a)
  (h2 : b - a = c - b)
  (h3 : c - b = -9 - c) :
  b = -5 ∧ a * c = 21 :=
by sorry

end arithmetic_sequence_solution_l68_68662


namespace base_n_representation_l68_68932

theorem base_n_representation 
  (n : ℕ) 
  (hn : n > 0)
  (a b c : ℕ) 
  (ha : 0 ≤ a ∧ a < n)
  (hb : 0 ≤ b ∧ b < n) 
  (hc : 0 ≤ c ∧ c < n) 
  (h_digits_sum : a + b + c = 24)
  (h_base_repr : 1998 = a * n^2 + b * n + c) 
  : n = 15 ∨ n = 22 ∨ n = 43 :=
sorry

end base_n_representation_l68_68932


namespace ratio_Mary_to_Seth_in_a_year_l68_68968

-- Given conditions
def Seth_current_age : ℝ := 3.5
def age_difference : ℝ := 9

-- Definitions derived from conditions
def Mary_current_age : ℝ := Seth_current_age + age_difference
def Seth_age_in_a_year : ℝ := Seth_current_age + 1
def Mary_age_in_a_year : ℝ := Mary_current_age + 1

-- The statement to prove
theorem ratio_Mary_to_Seth_in_a_year : (Mary_age_in_a_year / Seth_age_in_a_year) = 3 := sorry

end ratio_Mary_to_Seth_in_a_year_l68_68968


namespace percent_employed_females_l68_68126

theorem percent_employed_females (h1 : 96 / 100 > 0) (h2 : 24 / 100 > 0) : 
  (96 - 24) / 96 * 100 = 75 := 
by 
  -- Proof to be filled out
  sorry

end percent_employed_females_l68_68126


namespace find_y_l68_68439

theorem find_y (y : ℝ) (h : 2 * y / 3 = 12) : y = 18 :=
by
  sorry

end find_y_l68_68439


namespace savings_percentage_l68_68238

variables (I S : ℝ)
-- Conditions
-- A man saves a certain portion S of his income I during the first year.
-- He spends the remaining portion (I - S) on his personal expenses.
-- In the second year, his income increases by 50%, so his new income is 1.5I.
-- His savings increase by 100%, so his new savings are 2S.
-- His total expenditure in 2 years is double his expenditure in the first year.

def first_year_expenditure (I S : ℝ) : ℝ := I - S
def second_year_income (I : ℝ) : ℝ := 1.5 * I
def second_year_savings (S : ℝ) : ℝ := 2 * S
def second_year_expenditure (I S : ℝ) : ℝ := second_year_income I - second_year_savings S
def total_expenditure (I S : ℝ) : ℝ := first_year_expenditure I S + second_year_expenditure I S

theorem savings_percentage :
  total_expenditure I S = 2 * first_year_expenditure I S → S / I = 0.5 :=
by
  sorry

end savings_percentage_l68_68238


namespace worker_payment_l68_68628

theorem worker_payment (x : ℕ) (daily_return : ℕ) (non_working_days : ℕ) (total_days : ℕ) 
    (net_earning : ℕ) 
    (H1 : daily_return = 25) 
    (H2 : non_working_days = 24) 
    (H3 : total_days = 30) 
    (H4 : net_earning = 0) 
    (H5 : ∀ w, net_earning = w * x - non_working_days * daily_return) : 
  x = 100 :=
by
  sorry

end worker_payment_l68_68628


namespace visible_black_area_ratio_l68_68079

-- Definitions for circle areas as nonnegative real numbers
variables (A_b A_g A_w : ℝ) (hA_b : 0 ≤ A_b) (hA_g : 0 ≤ A_g) (hA_w : 0 ≤ A_w)
-- Condition: Initial visible black area is 7 times the white area
axiom initial_visible_black_area : 7 * A_w = A_b

-- Definition of new visible black area after movement
def new_visible_black_area := A_b - A_w

-- Prove the ratio of the visible black regions before and after moving the circles
theorem visible_black_area_ratio :
  (7 * A_w) / ((7 * A_w) - A_w) = 7 / 6 :=
by { sorry }

end visible_black_area_ratio_l68_68079


namespace mom_twice_alex_l68_68010

-- Definitions based on the conditions
def alex_age_in_2010 : ℕ := 10
def mom_age_in_2010 : ℕ := 5 * alex_age_in_2010
def future_years_after_2010 (x : ℕ) : ℕ := 2010 + x

-- Defining the ages in the future year
def alex_age_future (x : ℕ) : ℕ := alex_age_in_2010 + x
def mom_age_future (x : ℕ) : ℕ := mom_age_in_2010 + x

-- The theorem to prove
theorem mom_twice_alex (x : ℕ) (h : mom_age_future x = 2 * alex_age_future x) : future_years_after_2010 x = 2040 :=
  by
  sorry

end mom_twice_alex_l68_68010


namespace count_library_books_l68_68496

theorem count_library_books (initial_library_books : ℕ) 
  (books_given_away : ℕ) (books_added_from_source : ℕ) (books_donated : ℕ) 
  (h1 : initial_library_books = 125)
  (h2 : books_given_away = 42)
  (h3 : books_added_from_source = 68)
  (h4 : books_donated = 31) : 
  initial_library_books - books_given_away - books_donated = 52 :=
by sorry

end count_library_books_l68_68496


namespace inequality_proof_l68_68575

theorem inequality_proof (a b c x y z : ℝ) (h1 : a > 0) (h2 : b > 0) 
(h3 : c > 0) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) 
(h7 : a * y + b * x = c) (h8 : c * x + a * z = b) 
(h9 : b * z + c * y = a) :
x / (1 - y * z) + y / (1 - z * x) + z / (1 - x * y) ≤ 2 :=
sorry

end inequality_proof_l68_68575


namespace ratio_of_areas_of_two_concentric_circles_l68_68832

theorem ratio_of_areas_of_two_concentric_circles
  (C₁ C₂ : ℝ)
  (h1 : ∀ θ₁ θ₂, θ₁ = 30 ∧ θ₂ = 24 →
      (θ₁ / 360) * C₁ = (θ₂ / 360) * C₂):
  (C₁ / C₂) ^ 2 = (16 / 25) := by
  sorry

end ratio_of_areas_of_two_concentric_circles_l68_68832


namespace solve_for_b_l68_68690

variable (a b c d m : ℝ)

theorem solve_for_b (h : m = cadb / (a - b)) : b = ma / (cad + m) :=
sorry

end solve_for_b_l68_68690


namespace line_does_not_pass_through_fourth_quadrant_l68_68883

-- Definitions of conditions
variables {a b c x y : ℝ}

-- The mathematical statement to be proven
theorem line_does_not_pass_through_fourth_quadrant
  (h1 : a * b < 0) (h2 : b * c < 0) :
  ¬ (∃ x y, x > 0 ∧ y < 0 ∧ a * x + b * y + c = 0) :=
sorry

end line_does_not_pass_through_fourth_quadrant_l68_68883


namespace machine_A_sprockets_per_hour_l68_68403

theorem machine_A_sprockets_per_hour 
  (A T_Q : ℝ)
  (h1 : 550 = 1.1 * A * T_Q)
  (h2 : 550 = A * (T_Q + 10)) 
  : A = 5 :=
by
  sorry

end machine_A_sprockets_per_hour_l68_68403


namespace find_a_l68_68488

theorem find_a
  (a : ℝ)
  (h_perpendicular : ∀ x y : ℝ, ax + 2 * y - 1 = 0 → 3 * x - 6 * y - 1 = 0 → true) :
  a = 4 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end find_a_l68_68488


namespace compound_interest_rate_l68_68717

open Real

theorem compound_interest_rate
  (P : ℝ) (A : ℝ) (t : ℝ) (r : ℝ)
  (h_inv : P = 8000)
  (h_time : t = 2)
  (h_maturity : A = 8820) :
  r = 0.05 :=
by
  sorry

end compound_interest_rate_l68_68717


namespace find_pairs_l68_68794

theorem find_pairs (m n : ℕ) (h : m > 0 ∧ n > 0 ∧ m^2 = n^2 + m + n + 2018) :
  (m, n) = (1010, 1008) ∨ (m, n) = (506, 503) :=
by sorry

end find_pairs_l68_68794


namespace slope_of_chord_l68_68377

theorem slope_of_chord (x y : ℝ) (h : (x^2 / 16) + (y^2 / 9) = 1) (h_midpoint : (x₁ + x₂ = 2) ∧ (y₁ + y₂ = 4)) :
  ∃ k : ℝ, k = -9 / 32 :=
by
  sorry

end slope_of_chord_l68_68377


namespace no_difference_410_l68_68523

theorem no_difference_410 (n : ℕ) (R L a : ℕ) (h1 : R + L = 300)
  (h2 : L = 300 - R)
  (h3 : a ≤ 2 * R)
  (h4 : n = L + a)  :
  ¬ (n = 410) :=
by
  sorry

end no_difference_410_l68_68523


namespace min_value_expression_l68_68163

noncomputable def expression (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2)

theorem min_value_expression : ∃ x : ℝ, expression x = 2 * Real.sqrt 5 :=
by
  sorry

end min_value_expression_l68_68163


namespace general_formula_for_nth_term_exists_c_makes_bn_arithmetic_l68_68342

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Noncomputable sum of the first n terms of an arithmetic sequence
noncomputable def arithmetic_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 0 + a (n - 1))) / 2

variables {a : ℕ → ℤ} {S : ℕ → ℤ} {d : ℤ}
variable (c : ℤ)

axiom h1 : is_arithmetic_sequence a d
axiom h2 : d > 0
axiom h3 : a 1 * a 2 = 45
axiom h4 : a 0 + a 4 = 18

-- General formula for the nth term
theorem general_formula_for_nth_term :
  ∃ a1 d, a 0 = a1 ∧ d > 0 ∧ (∀ n, a n = a1 + n * d) :=
sorry

-- Arithmetic sequence from Sn/(n+c)
theorem exists_c_makes_bn_arithmetic :
  ∃ (c : ℤ), c ≠ 0 ∧ (∀ n, n > 0 → (arithmetic_sum a n) / (n + c) - (arithmetic_sum a (n - 1)) / (n - 1 + c) = d) :=
sorry

end general_formula_for_nth_term_exists_c_makes_bn_arithmetic_l68_68342


namespace mean_combined_l68_68031

-- Definitions for the two sets and their properties
def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

variables (set₁ set₂ : List ℕ)
-- Conditions based on the problem
axiom h₁ : set₁.length = 7
axiom h₂ : mean set₁ = 15
axiom h₃ : set₂.length = 8
axiom h₄ : mean set₂ = 30

-- Prove that the mean of the combined set is 23
theorem mean_combined (h₁ : set₁.length = 7) (h₂ : mean set₁ = 15)
  (h₃ : set₂.length = 8) (h₄ : mean set₂ = 30) : mean (set₁ ++ set₂) = 23 := 
sorry

end mean_combined_l68_68031


namespace pizza_volume_piece_l68_68574

theorem pizza_volume_piece (h : ℝ) (d : ℝ) (n : ℝ) (V_piece : ℝ) 
  (h_eq : h = 1 / 2) (d_eq : d = 16) (n_eq : n = 8) : 
  V_piece = 4 * Real.pi :=
by
  sorry

end pizza_volume_piece_l68_68574


namespace find_purple_balls_count_l68_68762

theorem find_purple_balls_count (k : ℕ) (h : ∃ k > 0, (21 - 3 * k) = (3 / 4) * (7 + k)) : k = 4 :=
sorry

end find_purple_balls_count_l68_68762


namespace evaluate_expression_l68_68582

theorem evaluate_expression : (Real.sqrt (Real.sqrt 5 ^ 4))^3 = 125 := by
  sorry

end evaluate_expression_l68_68582


namespace like_terms_implies_m_minus_n_l68_68152

/-- If 4x^(2m+2)y^(n-1) and -3x^(3m+1)y^(3n-5) are like terms, then m - n = -1. -/
theorem like_terms_implies_m_minus_n
  (m n : ℤ)
  (h1 : 2 * m + 2 = 3 * m + 1)
  (h2 : n - 1 = 3 * n - 5) :
  m - n = -1 :=
by
  sorry

end like_terms_implies_m_minus_n_l68_68152


namespace gdp_scientific_notation_l68_68149

noncomputable def gdp_nanning_2007 : ℝ := 1060 * 10^8

theorem gdp_scientific_notation :
  gdp_nanning_2007 = 1.06 * 10^11 :=
by sorry

end gdp_scientific_notation_l68_68149


namespace cost_of_adult_ticket_l68_68092

theorem cost_of_adult_ticket (c : ℝ) 
  (h₁ : 2 * (c + 6) + 3 * c = 77)
  : c + 6 = 19 :=
sorry

end cost_of_adult_ticket_l68_68092


namespace cos_15_degree_l68_68887

theorem cos_15_degree : 
  let d15 := 15 * Real.pi / 180
  let d45 := 45 * Real.pi / 180
  let d30 := 30 * Real.pi / 180
  Real.cos d15 = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end cos_15_degree_l68_68887


namespace investment_C_l68_68577

theorem investment_C (A_invest B_invest profit_A total_profit C_invest : ℕ)
  (hA_invest : A_invest = 6300) 
  (hB_invest : B_invest = 4200) 
  (h_profit_A : profit_A = 3900) 
  (h_total_profit : total_profit = 13000) 
  (h_proportional : profit_A / total_profit = A_invest / (A_invest + B_invest + C_invest)) :
  C_invest = 10500 := by
  sorry

end investment_C_l68_68577


namespace complex_expression_l68_68428

def imaginary_unit (i : ℂ) : Prop := i^2 = -1

theorem complex_expression (i : ℂ) (h : imaginary_unit i) :
  (1 - i) ^ 2016 + (1 + i) ^ 2016 = 2 ^ 1009 :=
by
  sorry

end complex_expression_l68_68428


namespace intersection_M_N_l68_68161

open Set Real

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - abs x)

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l68_68161


namespace interest_rate_B_to_C_l68_68639

theorem interest_rate_B_to_C
  (P : ℕ)                -- Principal amount
  (r_A : ℚ)              -- Interest rate A charges B per annum
  (t : ℕ)                -- Time period in years
  (gain_B : ℚ)           -- Gain of B in 3 years
  (H_P : P = 3500)
  (H_r_A : r_A = 0.10)
  (H_t : t = 3)
  (H_gain_B : gain_B = 315) :
  ∃ R : ℚ, R = 0.13 := 
by
  sorry

end interest_rate_B_to_C_l68_68639


namespace fuchsia_to_mauve_l68_68961

def fuchsia_to_mauve_amount (F : ℝ) : Prop :=
  let blue_in_fuchsia := (3 / 8) * F
  let red_in_fuchsia := (5 / 8) * F
  blue_in_fuchsia + 14 = 2 * red_in_fuchsia

theorem fuchsia_to_mauve (F : ℝ) (h : fuchsia_to_mauve_amount F) : F = 16 :=
by
  sorry

end fuchsia_to_mauve_l68_68961


namespace find_x_l68_68189

theorem find_x (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 7 * x^2 + 14 * x * y = 2 * x^3 + 4 * x^2 * y + y^3) : 
  x = 7 :=
sorry

end find_x_l68_68189


namespace intersection_set_l68_68742

-- Definition of the sets A and B
def setA : Set ℝ := { x | -2 < x ∧ x < 2 }
def setB : Set ℝ := { x | x < 0.5 }

-- The main theorem: Finding the intersection A ∩ B
theorem intersection_set : { x : ℝ | -2 < x ∧ x < 0.5 } = setA ∩ setB := by
  sorry

end intersection_set_l68_68742


namespace goods_train_length_is_280_l68_68746

noncomputable def length_of_goods_train (passenger_speed passenger_speed_kmh: ℝ) 
                                       (goods_speed goods_speed_kmh: ℝ) 
                                       (time_to_pass: ℝ) : ℝ :=
  let kmh_to_ms := (1000 : ℝ) / (3600 : ℝ)
  let passenger_speed_ms := passenger_speed * kmh_to_ms
  let goods_speed_ms     := goods_speed * kmh_to_ms
  let relative_speed     := passenger_speed_ms + goods_speed_ms
  relative_speed * time_to_pass

theorem goods_train_length_is_280 :
  length_of_goods_train 70 70 42 42 9 = 280 :=
by
  sorry

end goods_train_length_is_280_l68_68746


namespace amount_of_money_C_l68_68673

theorem amount_of_money_C (a b c d : ℤ) 
  (h1 : a + b + c + d = 600)
  (h2 : a + c = 200)
  (h3 : b + c = 350)
  (h4 : a + d = 300)
  (h5 : a ≥ 2 * b) : c = 150 := 
by
  sorry

end amount_of_money_C_l68_68673


namespace law_of_cosines_l68_68478

theorem law_of_cosines (a b c : ℝ) (A : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ A ≥ 0 ∧ A ≤ π) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A :=
sorry

end law_of_cosines_l68_68478


namespace pythagorean_triplets_l68_68021

theorem pythagorean_triplets (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  ∃ d p q : ℤ, a = 2 * d * p * q ∧ b = d * (q^2 - p^2) ∧ c = d * (p^2 + q^2) := sorry

end pythagorean_triplets_l68_68021


namespace min_sin_cos_sixth_power_l68_68905

noncomputable def min_value_sin_cos_expr : ℝ :=
  (3 + 2 * Real.sqrt 2) / 12

theorem min_sin_cos_sixth_power :
  ∃ x : ℝ, (∀ y, (Real.sin y) ^ 6 + 2 * (Real.cos y) ^ 6 ≥ min_value_sin_cos_expr) ∧ 
            ((Real.sin x) ^ 6 + 2 * (Real.cos x) ^ 6 = min_value_sin_cos_expr) :=
sorry

end min_sin_cos_sixth_power_l68_68905


namespace compute_multiplied_difference_l68_68009

theorem compute_multiplied_difference (a b : ℕ) (h_a : a = 25) (h_b : b = 15) :
  3 * ((a + b) ^ 2 - (a - b) ^ 2) = 4500 := by
  sorry

end compute_multiplied_difference_l68_68009


namespace apples_in_bowl_l68_68924

theorem apples_in_bowl (green_plus_red_diff red_count : ℕ) (h1 : green_plus_red_diff = 12) (h2 : red_count = 16) :
  red_count + (red_count + green_plus_red_diff) = 44 :=
by
  sorry

end apples_in_bowl_l68_68924


namespace find_leftover_amount_l68_68457

open Nat

def octal_to_decimal (n : ℕ) : ℕ :=
  let digits := [5, 5, 5, 5]
  List.foldr (λ (d : ℕ) (acc : ℕ) => d + 8 * acc) 0 digits

def expenses_total : ℕ := 1200 + 800 + 400

theorem find_leftover_amount : 
  let initial_amount := octal_to_decimal 5555
  let final_amount := initial_amount - expenses_total
  final_amount = 525 := by
    sorry

end find_leftover_amount_l68_68457


namespace sum_first_10_terms_l68_68002

def a_n (n : ℕ) : ℤ := (-1)^n * (3 * n - 2)

theorem sum_first_10_terms :
  (a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) +
  (a_n 6) + (a_n 7) + (a_n 8) + (a_n 9) + (a_n 10) = 15 :=
by
  sorry

end sum_first_10_terms_l68_68002


namespace volume_of_box_l68_68790

variable (width length height : ℝ)
variable (Volume : ℝ)

-- Given conditions
def w : ℝ := 9
def l : ℝ := 4
def h : ℝ := 7

-- The statement to prove
theorem volume_of_box : Volume = l * w * h := by
  sorry

end volume_of_box_l68_68790


namespace remaining_credit_to_be_paid_l68_68521

-- Define conditions
def total_credit_limit := 100
def amount_paid_tuesday := 15
def amount_paid_thursday := 23

-- Define the main theorem based on the given question and its correct answer
theorem remaining_credit_to_be_paid : 
  total_credit_limit - amount_paid_tuesday - amount_paid_thursday = 62 := 
by 
  -- Proof is omitted
  sorry

end remaining_credit_to_be_paid_l68_68521


namespace cyclist_pedestrian_meeting_distance_l68_68078

theorem cyclist_pedestrian_meeting_distance :
  let A := 0 -- Representing point A at 0 km
  let B := 3 -- Representing point B at 3 km since AB = 3 km
  let C := 7 -- Representing point C at 7 km since AC = AB + BC = 3 + 4 = 7 km
  exists (x : ℝ), x > 0 ∧ x < 3 ∧ x = 2.1 :=
sorry

end cyclist_pedestrian_meeting_distance_l68_68078


namespace bianca_next_day_run_l68_68756

-- Define the conditions
variable (miles_first_day : ℕ) (total_miles : ℕ)

-- Set the conditions for Bianca's run
def conditions := miles_first_day = 8 ∧ total_miles = 12

-- State the proposition we need to prove
def miles_next_day (miles_first_day total_miles : ℕ) : ℕ := total_miles - miles_first_day

-- The theorem stating the problem to prove
theorem bianca_next_day_run (h : conditions 8 12) : miles_next_day 8 12 = 4 := by
  unfold conditions at h
  simp [miles_next_day] at h
  sorry

end bianca_next_day_run_l68_68756


namespace subset_A_l68_68005

open Set

theorem subset_A (A : Set ℝ) (h : A = { x | x > -1 }) : {0} ⊆ A :=
by
  sorry

end subset_A_l68_68005


namespace find_m_l68_68816

theorem find_m (m : ℕ) (h1 : (3 * m - 7) % 2 = 0) (h2 : 3 * m - 7 < 0) : m = 1 := 
by
  sorry

end find_m_l68_68816


namespace union_A_B_complement_intersection_A_B_l68_68382

def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}

def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

theorem union_A_B : A ∪ B = { x | x ≥ 3 } := 
by
  sorry

theorem complement_intersection_A_B : (A ∩ B)ᶜ = { x | x < 4 } ∪ { x | x ≥ 10 } := 
by
  sorry

end union_A_B_complement_intersection_A_B_l68_68382


namespace ratio_copper_to_zinc_l68_68704

theorem ratio_copper_to_zinc (copper zinc : ℝ) (hc : copper = 24) (hz : zinc = 10.67) : (copper / zinc) = 2.25 :=
by
  rw [hc, hz]
  -- Add the arithmetic operation
  sorry

end ratio_copper_to_zinc_l68_68704


namespace elevator_height_after_20_seconds_l68_68334

-- Conditions
def starting_height : ℕ := 120
def descending_speed : ℕ := 4
def time_elapsed : ℕ := 20

-- Statement to prove
theorem elevator_height_after_20_seconds : 
  starting_height - descending_speed * time_elapsed = 40 := 
by 
  sorry

end elevator_height_after_20_seconds_l68_68334


namespace line_circle_relationship_l68_68147

theorem line_circle_relationship (m : ℝ) :
  (∃ x y : ℝ, (mx + y - m - 1 = 0) ∧ (x^2 + y^2 = 2)) ∨ 
  (∃ x : ℝ, (x - 1)^2 + (m*(x - 1) + (1 - 1))^2 = 2) :=
by
  sorry

end line_circle_relationship_l68_68147


namespace sum_of_three_numbers_l68_68718

-- Definitions for the conditions
def mean_condition_1 (x y z : ℤ) := (x + y + z) / 3 = x + 20
def mean_condition_2 (x y z : ℤ) := (x + y + z) / 3 = z - 18
def median_condition (y : ℤ) := y = 9

-- The Lean 4 statement to prove the sum of x, y, and z is 21
theorem sum_of_three_numbers (x y z : ℤ) 
  (h1 : mean_condition_1 x y z) 
  (h2 : mean_condition_2 x y z) 
  (h3 : median_condition y) : 
  x + y + z = 21 := 
  by 
    sorry

end sum_of_three_numbers_l68_68718


namespace exists_rational_non_integer_linear_l68_68737

theorem exists_rational_non_integer_linear (k1 k2 : ℤ) : 
  ∃ (x y : ℚ), x ≠ ⌊x⌋ ∧ y ≠ ⌊y⌋ ∧ 
  19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

end exists_rational_non_integer_linear_l68_68737


namespace only_one_of_A_B_qualifies_at_least_one_qualifies_l68_68707

-- Define the probabilities
def P_A_written : ℚ := 2/3
def P_B_written : ℚ := 1/2
def P_C_written : ℚ := 3/4

def P_A_interview : ℚ := 1/2
def P_B_interview : ℚ := 2/3
def P_C_interview : ℚ := 1/3

-- Calculate the overall probabilities for each student qualifying
def P_A_qualifies : ℚ := P_A_written * P_A_interview
def P_B_qualifies : ℚ := P_B_written * P_B_interview
def P_C_qualifies : ℚ := P_C_written * P_C_interview

-- Part 1: Probability that only one of A or B qualifies
theorem only_one_of_A_B_qualifies :
  P_A_qualifies * (1 - P_B_qualifies) + (1 - P_A_qualifies) * P_B_qualifies = 4/9 :=
by sorry

-- Part 2: Probability that at least one of A, B, or C qualifies
theorem at_least_one_qualifies :
  1 - (1 - P_A_qualifies) * (1 - P_B_qualifies) * (1 - P_C_qualifies) = 2/3 :=
by sorry

end only_one_of_A_B_qualifies_at_least_one_qualifies_l68_68707


namespace distance_covered_downstream_l68_68101

noncomputable def speed_in_still_water := 16 -- km/hr
noncomputable def speed_of_stream := 5 -- km/hr
noncomputable def time_taken := 5 -- hours
noncomputable def effective_speed_downstream := speed_in_still_water + speed_of_stream -- km/hr

theorem distance_covered_downstream :
  (effective_speed_downstream * time_taken = 105) :=
by
  sorry

end distance_covered_downstream_l68_68101


namespace even_function_a_value_l68_68981

theorem even_function_a_value (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x^2 + (a - 2) * x + a^2 - a - 2 = (a + 1) * x^2 - (a - 2) * x + a^2 - a - 2) → a = 2 := 
by sorry

end even_function_a_value_l68_68981


namespace find_y_value_l68_68429

noncomputable def y_value (y : ℝ) :=
  (3 * y)^2 + (7 * y)^2 + (1 / 2) * (3 * y) * (7 * y) = 1200

theorem find_y_value (y : ℝ) (hy : y_value y) : y = 10 :=
by
  sorry

end find_y_value_l68_68429


namespace negation_of_universal_proposition_l68_68537

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 2^x - 1 > 0)) ↔ (∃ x : ℝ, x^2 + 2^x - 1 ≤ 0) :=
by
  sorry

end negation_of_universal_proposition_l68_68537


namespace composite_quadratic_l68_68374

theorem composite_quadratic (a b : Int) (x1 x2 : Int)
  (h1 : x1 + x2 = -a)
  (h2 : x1 * x2 = b)
  (h3 : abs x1 > 2)
  (h4 : abs x2 > 2) :
  ∃ m n : Int, a + b + 1 = m * n ∧ m > 1 ∧ n > 1 :=
by
  sorry

end composite_quadratic_l68_68374


namespace zero_points_of_f_l68_68948

def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem zero_points_of_f : (f (-1/2) = 0) ∧ (f (-1) = 0) :=
by
  sorry

end zero_points_of_f_l68_68948


namespace martin_travel_time_l68_68622

-- Definitions based on the conditions
def distance : ℕ := 12
def speed : ℕ := 2

-- Statement of the problem to be proven
theorem martin_travel_time : (distance / speed) = 6 := by sorry

end martin_travel_time_l68_68622


namespace quadratic_root_condition_l68_68482

theorem quadratic_root_condition (a b : ℝ) (h : (3:ℝ)^2 + 2 * a * 3 + 3 * b = 0) : 2 * a + b = -3 :=
by
  sorry

end quadratic_root_condition_l68_68482


namespace cube_surface_area_150_of_volume_125_l68_68732

def volume (s : ℝ) : ℝ := s^3

def surface_area (s : ℝ) : ℝ := 6 * s^2

theorem cube_surface_area_150_of_volume_125 :
  ∀ (s : ℝ), volume s = 125 → surface_area s = 150 :=
by 
  intros s hs
  sorry

end cube_surface_area_150_of_volume_125_l68_68732


namespace number_of_triples_l68_68090

theorem number_of_triples : 
  ∃ n : ℕ, 
  n = 2 ∧
  ∀ (a b c : ℕ), 
    (2 ≤ a ∧ a ≤ b ∧ b ≤ c) →
    (a * b * c = 4 * (a * b + b * c + c * a)) →
    n = 2 :=
sorry

end number_of_triples_l68_68090


namespace sin_squared_sum_eq_one_l68_68347

theorem sin_squared_sum_eq_one (α β γ : ℝ) 
  (h₁ : 0 ≤ α ∧ α ≤ π/2) 
  (h₂ : 0 ≤ β ∧ β ≤ π/2) 
  (h₃ : 0 ≤ γ ∧ γ ≤ π/2) 
  (h₄ : Real.sin α + Real.sin β + Real.sin γ = 1)
  (h₅ : Real.sin α * Real.cos (2 * α) + Real.sin β * Real.cos (2 * β) + Real.sin γ * Real.cos (2 * γ) = -1) :
  Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1 := 
sorry

end sin_squared_sum_eq_one_l68_68347


namespace parabola_axis_of_symmetry_range_l68_68824

theorem parabola_axis_of_symmetry_range
  (a b c m n t : ℝ)
  (h₀ : 0 < a)
  (h₁ : m = a * 1^2 + b * 1 + c)
  (h₂ : n = a * 3^2 + b * 3 + c)
  (h₃ : m < n)
  (h₄ : n < c)
  (h_t : t = -b / (2 * a)) :
  (3 / 2) < t ∧ t < 2 :=
sorry

end parabola_axis_of_symmetry_range_l68_68824


namespace find_weight_of_a_l68_68975

-- Define the weights
variables (a b c d e : ℝ)

-- Given conditions
def condition1 := (a + b + c) / 3 = 50
def condition2 := (a + b + c + d) / 4 = 53
def condition3 := (b + c + d + e) / 4 = 51
def condition4 := e = d + 3

-- Proof goal
theorem find_weight_of_a : condition1 a b c → condition2 a b c d → condition3 b c d e → condition4 d e → a = 73 :=
by
  intros h1 h2 h3 h4
  sorry

end find_weight_of_a_l68_68975


namespace inequality_proof_l68_68381

theorem inequality_proof (a b : ℝ) (h : a - |b| > 0) : b + a > 0 :=
sorry

end inequality_proof_l68_68381


namespace ball_bounce_height_l68_68674

theorem ball_bounce_height :
  ∃ k : ℕ, 2000 * (2 / 3 : ℝ) ^ k < 2 ∧ ∀ j : ℕ, j < k → 2000 * (2 / 3 : ℝ) ^ j ≥ 2 :=
by {
  sorry
}

end ball_bounce_height_l68_68674


namespace unit_digit_7_pow_2023_l68_68202

theorem unit_digit_7_pow_2023 : (7^2023) % 10 = 3 :=
by
  -- Provide proof here
  sorry

end unit_digit_7_pow_2023_l68_68202


namespace domain_of_f_l68_68087

noncomputable def f (x : ℝ) : ℝ := (5 * x + 2) / Real.sqrt (2 * x - 10)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f y = f x} = {x : ℝ | x > 5} :=
by
  sorry

end domain_of_f_l68_68087


namespace minimal_difference_big_small_sum_l68_68873

theorem minimal_difference_big_small_sum :
  ∀ (N : ℕ), N > 0 → ∃ (S : ℕ), 
  S = (N * (N - 1) * (2 * N + 5)) / 6 :=
  by 
    sorry

end minimal_difference_big_small_sum_l68_68873


namespace least_possible_b_l68_68902

def is_prime (n : Nat) : Prop := Nat.Prime n

theorem least_possible_b (a b : Nat) (h1 : is_prime a) (h2 : is_prime b) (h3 : a + 2 * b = 180) (h4 : a > b) : b = 19 :=
by 
  sorry

end least_possible_b_l68_68902


namespace find_xy_l68_68646

theorem find_xy (x y : ℝ) (h : x * (x + 2 * y) = x^2 + 10) : x * y = 5 :=
by
  sorry

end find_xy_l68_68646


namespace infinite_k_Q_ineq_l68_68361

def Q (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem infinite_k_Q_ineq :
  ∃ᶠ k in at_top, Q (3 ^ k) > Q (3 ^ (k + 1)) := sorry

end infinite_k_Q_ineq_l68_68361


namespace min_disks_required_l68_68371

def num_files : ℕ := 35
def disk_size : ℕ := 2
def file_size_0_9 : ℕ := 4
def file_size_0_8 : ℕ := 15
def file_size_0_5 : ℕ := num_files - file_size_0_9 - file_size_0_8

-- Prove the minimum number of disks required to store all files.
theorem min_disks_required 
  (n : ℕ) 
  (disk_storage : ℕ)
  (num_files_0_9 : ℕ)
  (num_files_0_8 : ℕ)
  (num_files_0_5 : ℕ) :
  n = num_files → disk_storage = disk_size → num_files_0_9 = file_size_0_9 → num_files_0_8 = file_size_0_8 → num_files_0_5 = file_size_0_5 → 
  ∃ (d : ℕ), d = 15 :=
by 
  intros H1 H2 H3 H4 H5
  sorry

end min_disks_required_l68_68371


namespace expand_expression_l68_68560

theorem expand_expression (x y : ℕ) : 
  (x + 15) * (3 * y + 20) = 3 * x * y + 20 * x + 45 * y + 300 :=
by 
  sorry

end expand_expression_l68_68560


namespace solve_abs_ineq_l68_68710

theorem solve_abs_ineq (x : ℝ) (h : x > 0) : |4 * x - 5| < 8 ↔ 0 < x ∧ x < 13 / 4 :=
by
  sorry

end solve_abs_ineq_l68_68710


namespace largest_fraction_among_fractions_l68_68958

theorem largest_fraction_among_fractions :
  let A := (2 : ℚ) / 5
  let B := (3 : ℚ) / 7
  let C := (4 : ℚ) / 9
  let D := (3 : ℚ) / 8
  let E := (9 : ℚ) / 20
  (A < E) ∧ (B < E) ∧ (C < E) ∧ (D < E) :=
by
  let A := (2 : ℚ) / 5
  let B := (3 : ℚ) / 7
  let C := (4 : ℚ) / 9
  let D := (3 : ℚ) / 8
  let E := (9 : ℚ) / 20
  sorry

end largest_fraction_among_fractions_l68_68958


namespace exists_indices_for_sequences_l68_68759

theorem exists_indices_for_sequences 
  (a b c : ℕ → ℕ) :
  ∃ (p q : ℕ), p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
sorry

end exists_indices_for_sequences_l68_68759


namespace sufficient_and_necessary_condition_l68_68850

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log (x + Real.sqrt (x^2 + 1)) / Real.log 2

theorem sufficient_and_necessary_condition {a b : ℝ} (h : a + b ≥ 0) : f a + f b ≥ 0 :=
sorry

end sufficient_and_necessary_condition_l68_68850


namespace sum_of_largest_and_smallest_is_correct_l68_68027

-- Define the set of digits
def digits : Finset ℕ := {2, 0, 4, 1, 5, 8}

-- Define the largest possible number using the digits
def largestNumber : ℕ := 854210

-- Define the smallest possible number using the digits
def smallestNumber : ℕ := 102458

-- Define the sum of largest and smallest possible numbers
def sumOfNumbers : ℕ := largestNumber + smallestNumber

-- Main theorem to prove
theorem sum_of_largest_and_smallest_is_correct : sumOfNumbers = 956668 := by
  sorry

end sum_of_largest_and_smallest_is_correct_l68_68027


namespace scientific_notation_integer_l68_68283

theorem scientific_notation_integer (x : ℝ) (h1 : x > 10) :
  ∃ (A : ℝ) (N : ℤ), (1 ≤ A ∧ A < 10) ∧ x = A * 10^N :=
by
  sorry

end scientific_notation_integer_l68_68283


namespace solve_quadratic_eq_l68_68000

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 2*x + 1 = 0) : x = 1 :=
by
  sorry

end solve_quadratic_eq_l68_68000


namespace over_limit_weight_l68_68097

variable (hc_books : ℕ) (hc_weight : ℕ → ℝ)
variable (tex_books : ℕ) (tex_weight : ℕ → ℝ)
variable (knick_knacks : ℕ) (knick_weight : ℕ → ℝ)
variable (weight_limit : ℝ)

axiom hc_books_value : hc_books = 70
axiom hc_weight_value : hc_weight hc_books = 0.5
axiom tex_books_value : tex_books = 30
axiom tex_weight_value : tex_weight tex_books = 2
axiom knick_knacks_value : knick_knacks = 3
axiom knick_weight_value : knick_weight knick_knacks = 6
axiom weight_limit_value : weight_limit = 80

theorem over_limit_weight : 
  (hc_books * hc_weight hc_books + tex_books * tex_weight tex_books + knick_knacks * knick_weight knick_knacks) - weight_limit = 33 := by
  sorry

end over_limit_weight_l68_68097


namespace solution_set_of_inequality_l68_68634

theorem solution_set_of_inequality (x : ℝ) : x * (x + 3) ≥ 0 ↔ x ≤ -3 ∨ x ≥ 0 :=
by
  sorry

end solution_set_of_inequality_l68_68634


namespace find_number_of_pens_l68_68738

-- Definitions based on the conditions in the problem
def total_utensils (P L : ℕ) : Prop := P + L = 108
def pencils_formula (P L : ℕ) : Prop := L = 5 * P + 12

-- The theorem we need to prove
theorem find_number_of_pens (P L : ℕ) (h1 : total_utensils P L) (h2 : pencils_formula P L) : P = 16 :=
by sorry

end find_number_of_pens_l68_68738


namespace intersection_AB_l68_68962

variable {x : ℝ}

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | x > 0}

theorem intersection_AB : A ∩ B = {x | 0 < x ∧ x < 2} :=
by sorry

end intersection_AB_l68_68962


namespace total_distance_hiked_l68_68383

-- Defining the distances Terrell hiked on Saturday and Sunday
def distance_Saturday : Real := 8.2
def distance_Sunday : Real := 1.6

-- Stating the theorem to prove the total distance
theorem total_distance_hiked : distance_Saturday + distance_Sunday = 9.8 := by
  sorry

end total_distance_hiked_l68_68383


namespace solve_f_1991_2_1990_l68_68208

-- Define the sum of digits function for an integer k
def sum_of_digits (k : ℕ) : ℕ := k.digits 10 |>.sum

-- Define f1(k) as the square of the sum of digits of k
def f1 (k : ℕ) : ℕ := (sum_of_digits k) ^ 2

-- Define the recursive sequence fn as given in the problem
def fn : ℕ → ℕ → ℕ
| 0, k => k
| n + 1, k => f1 (fn n k)

-- Define the specific problem statement
theorem solve_f_1991_2_1990 : fn 1991 (2 ^ 1990) = 4 := sorry

end solve_f_1991_2_1990_l68_68208


namespace unique_solution_7tuples_l68_68301

theorem unique_solution_7tuples : 
  ∃! (x : Fin 7 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1/8 :=
sorry

end unique_solution_7tuples_l68_68301


namespace tips_fraction_l68_68198

theorem tips_fraction {S T I : ℚ} (h1 : T = (7/4) * S) (h2 : I = S + T) : (T / I) = 7 / 11 :=
by
  sorry

end tips_fraction_l68_68198


namespace simplify_polynomial_l68_68131

theorem simplify_polynomial (s : ℝ) :
  (2*s^2 + 5*s - 3) - (2*s^2 + 9*s - 7) = -4*s + 4 :=
by
  sorry

end simplify_polynomial_l68_68131


namespace first_common_digit_three_digit_powers_l68_68130

theorem first_common_digit_three_digit_powers (m n: ℕ) (hm: 100 ≤ 2^m ∧ 2^m < 1000) (hn: 100 ≤ 3^n ∧ 3^n < 1000) :
  (∃ d, (2^m).div 100 = d ∧ (3^n).div 100 = d ∧ d = 2) :=
sorry

end first_common_digit_three_digit_powers_l68_68130


namespace factorization_of_difference_of_squares_l68_68411

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l68_68411


namespace quadratic_solution_transform_l68_68757

theorem quadratic_solution_transform (a b c : ℝ) (hA : 0 = a * (-3)^2 + b * (-3) + c) (hB : 0 = a * 4^2 + b * 4 + c) :
  (∃ x1 x2 : ℝ, a * (x1 - 1)^2 + b * (x1 - 1) + c = 0 ∧ a * (x2 - 1)^2 + b * (x2 - 1) + c = 0 ∧ x1 = -2 ∧ x2 = 5) :=
  sorry

end quadratic_solution_transform_l68_68757


namespace product_is_approximately_9603_l68_68838

noncomputable def smaller_number : ℝ := 97.49871794028884
noncomputable def successive_number : ℝ := smaller_number + 1
noncomputable def product_of_numbers : ℝ := smaller_number * successive_number

theorem product_is_approximately_9603 : abs (product_of_numbers - 9603) < 10e-3 := 
sorry

end product_is_approximately_9603_l68_68838


namespace marble_count_l68_68167

variable (r b g : ℝ)

-- Conditions
def condition1 : b = r / 1.3 := sorry
def condition2 : g = 1.5 * r := sorry

-- Theorem statement
theorem marble_count (h1 : b = r / 1.3) (h2 : g = 1.5 * r) :
  r + b + g = 3.27 * r :=
by sorry

end marble_count_l68_68167


namespace function_even_iff_a_eq_one_l68_68527

theorem function_even_iff_a_eq_one (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = a * (3^x) + 1/(3^x)) → 
  (∀ x : ℝ, f x = f (-x)) ↔ a = 1 :=
by
  sorry

end function_even_iff_a_eq_one_l68_68527


namespace largest_three_digit_congruent_to_twelve_mod_fifteen_l68_68463

theorem largest_three_digit_congruent_to_twelve_mod_fifteen :
  ∃ n : ℕ, 100 ≤ 15 * n + 12 ∧ 15 * n + 12 < 1000 ∧ (15 * n + 12 = 987) :=
sorry

end largest_three_digit_congruent_to_twelve_mod_fifteen_l68_68463


namespace common_chord_of_circles_l68_68143

theorem common_chord_of_circles
  (x y : ℝ)
  (h1 : x^2 + y^2 + 2 * x = 0)
  (h2 : x^2 + y^2 - 4 * y = 0)
  : x + 2 * y = 0 := 
by
  -- Lean will check the logical consistency of the statement.
  sorry

end common_chord_of_circles_l68_68143


namespace box_dimensions_sum_l68_68797

theorem box_dimensions_sum (X Y Z : ℝ) (hXY : X * Y = 18) (hXZ : X * Z = 54) (hYZ : Y * Z = 36) (hX_pos : X > 0) (hY_pos : Y > 0) (hZ_pos : Z > 0) :
  X + Y + Z = 11 := 
sorry

end box_dimensions_sum_l68_68797


namespace correct_calculation_value_l68_68427

theorem correct_calculation_value (x : ℕ) (h : (x * 5) + 7 = 27) : (x + 5) * 7 = 63 :=
by
  -- The conditions are used directly in the definitions
  -- Given the condition (x * 5) + 7 = 27
  let h1 := h
  -- Solve for x and use x in the correct calculation
  sorry

end correct_calculation_value_l68_68427


namespace jessica_deposit_fraction_l68_68324

-- Definitions based on conditions
variable (initial_balance : ℝ)
variable (fraction_withdrawn : ℝ) (withdrawn_amount : ℝ)
variable (final_balance remaining_balance fraction_deposit : ℝ)

-- Conditions
def conditions := 
  fraction_withdrawn = 2 / 5 ∧
  withdrawn_amount = 400 ∧
  remaining_balance = initial_balance - withdrawn_amount ∧
  remaining_balance = initial_balance * (1 - fraction_withdrawn) ∧
  final_balance = 750 ∧
  final_balance = remaining_balance + fraction_deposit * remaining_balance

-- The proof problem
theorem jessica_deposit_fraction : 
  conditions initial_balance fraction_withdrawn withdrawn_amount final_balance remaining_balance fraction_deposit →
  fraction_deposit = 1 / 4 :=
by
  intro h
  sorry

end jessica_deposit_fraction_l68_68324


namespace determine_perimeter_of_fourth_shape_l68_68011

theorem determine_perimeter_of_fourth_shape
  (P_1 P_2 P_3 P_4 : ℝ)
  (h1 : P_1 = 8)
  (h2 : P_2 = 11.4)
  (h3 : P_3 = 14.7)
  (h4 : P_1 + P_2 + P_4 = 2 * P_3) :
  P_4 = 10 := 
by
  -- Proof goes here
  sorry

end determine_perimeter_of_fourth_shape_l68_68011


namespace ellipse_hexagon_proof_l68_68300

noncomputable def m_value : ℝ := 3 + 2 * Real.sqrt 3

theorem ellipse_hexagon_proof (m : ℝ) (k : ℝ) 
  (hk : k ≠ 0) (hm : m > 3) :
  (∀ x y : ℝ, (x / m)^2 + (y / 3)^2 = 1 ∧ (y = k * x ∨ y = -k * x)) →
  k = Real.sqrt 3 →
  (|((4*m)/(m+1)) - (m-3)| = 0) →
  m = m_value :=
by
  sorry

end ellipse_hexagon_proof_l68_68300


namespace count_ordered_triples_lcm_l68_68479

def lcm_of_pair (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem count_ordered_triples_lcm :
  (∃ (count : ℕ), count = 70 ∧
   ∀ (a b c : ℕ), (0 < a) → (0 < b) → (0 < c) →
   lcm_of_pair a b = 1000 → lcm_of_pair b c = 2000 → lcm_of_pair c a = 2000 → count = 70) :=
sorry

end count_ordered_triples_lcm_l68_68479


namespace Jerry_remaining_pages_l68_68596

theorem Jerry_remaining_pages (total_pages : ℕ) (saturday_read : ℕ) (sunday_read : ℕ) (remaining_pages : ℕ) 
  (h1 : total_pages = 93) (h2 : saturday_read = 30) (h3 : sunday_read = 20) (h4 : remaining_pages = 43) : 
  total_pages - saturday_read - sunday_read = remaining_pages := 
by
  sorry

end Jerry_remaining_pages_l68_68596


namespace ratio_both_basketball_volleyball_l68_68281

variable (total_students : ℕ) (play_basketball : ℕ) (play_volleyball : ℕ) (play_neither : ℕ) (play_both : ℕ)

theorem ratio_both_basketball_volleyball (h1 : total_students = 20)
    (h2 : play_basketball = 20 / 2)
    (h3 : play_volleyball = (2 * 20) / 5)
    (h4 : play_neither = 4)
    (h5 : total_students - play_neither = play_basketball + play_volleyball - play_both) :
    play_both / total_students = 1 / 10 :=
by
  sorry

end ratio_both_basketball_volleyball_l68_68281


namespace average_first_15_even_numbers_l68_68750

theorem average_first_15_even_numbers : 
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30) / 15 = 16 :=
by 
  sorry

end average_first_15_even_numbers_l68_68750


namespace find_k_l68_68165

def otimes (a b : ℝ) := a * b + a + b^2

theorem find_k (k : ℝ) (h1 : otimes 1 k = 2) (h2 : 0 < k) :
  k = 1 :=
sorry

end find_k_l68_68165


namespace inequality_bounds_l68_68436

theorem inequality_bounds (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  1 < (a/(a+b) + b/(b+c) + c/(c+d) + d/(d+e) + e/(e+a)) ∧
  (a/(a+b) + b/(b+c) + c/(c+d) + d/(d+e) + e/(e+a)) < 4 :=
sorry

end inequality_bounds_l68_68436


namespace greatest_positive_multiple_of_4_l68_68007

theorem greatest_positive_multiple_of_4 {y : ℕ} (h1 : y % 4 = 0) (h2 : y > 0) (h3 : y^3 < 8000) : y ≤ 16 :=
by {
  -- The proof will go here
  -- Sorry is placed here to skip the proof for now
  sorry
}

end greatest_positive_multiple_of_4_l68_68007


namespace oranges_to_put_back_l68_68901

variables (A O x : ℕ)

theorem oranges_to_put_back
    (h1 : 40 * A + 60 * O = 560)
    (h2 : A + O = 10)
    (h3 : (40 * A + 60 * (O - x)) / (10 - x) = 50) : x = 6 := 
sorry

end oranges_to_put_back_l68_68901


namespace sum_of_roots_equation_l68_68631

noncomputable def sum_of_roots (a b c : ℝ) : ℝ :=
  (-b) / a

theorem sum_of_roots_equation :
  let a := 3
  let b := -15
  let c := 20
  sum_of_roots a b c = 5 := 
  by {
    sorry
  }

end sum_of_roots_equation_l68_68631


namespace last_digit_of_S_l68_68709

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_S : last_digit (54 ^ 2020 + 28 ^ 2022) = 0 :=
by 
  -- The Lean proof steps would go here
  sorry

end last_digit_of_S_l68_68709


namespace theta1_gt_theta2_l68_68998

theorem theta1_gt_theta2 (a : ℝ) (b : ℝ) (θ1 θ2 : ℝ)
  (h_range_θ1 : 0 ≤ θ1 ∧ θ1 ≤ π) (h_range_θ2 : 0 ≤ θ2 ∧ θ2 ≤ π)
  (x1 x2 : ℝ) (hx1 : x1 = a * Real.cos θ1) (hx2 : x2 = a * Real.cos θ2)
  (h_less : x1 < x2) : θ1 > θ2 :=
by
  sorry

end theta1_gt_theta2_l68_68998


namespace quadratic_roots_l68_68360

theorem quadratic_roots (m : ℝ) (h_eq : ∃ α β : ℝ, (α + β = -4) ∧ (α * β = m) ∧ (|α - β| = 2)) : m = 5 :=
sorry

end quadratic_roots_l68_68360


namespace smallest_natur_number_with_units_digit_6_and_transf_is_four_times_l68_68437

open Nat

theorem smallest_natur_number_with_units_digit_6_and_transf_is_four_times (n : ℕ) :
  (n % 10 = 6 ∧ ∃ m, 6 * 10 ^ (m - 1) + n / 10 = 4 * n) → n = 153846 :=
by 
  sorry

end smallest_natur_number_with_units_digit_6_and_transf_is_four_times_l68_68437


namespace theta_range_l68_68787

noncomputable def f (x θ : ℝ) : ℝ := x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ

theorem theta_range (θ : ℝ) (k : ℤ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x θ > 0) →
  θ ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 12) (2 * k * Real.pi + 5 * Real.pi / 12) :=
sorry

end theta_range_l68_68787


namespace solution_system_eq_l68_68302

theorem solution_system_eq (x y : ℝ) :
  (x^2 * y + x * y^2 - 2 * x - 2 * y + 10 = 0) ∧
  (x^3 * y - x * y^3 - 2 * x^2 + 2 * y^2 - 30 = 0) ↔ 
  (x = -4 ∧ y = -1) :=
by sorry

end solution_system_eq_l68_68302


namespace original_group_size_l68_68068

theorem original_group_size (M : ℕ) 
  (h1 : ∀ work_done_by_one, work_done_by_one = 1 / (6 * M))
  (h2 : ∀ work_done_by_one, work_done_by_one = 1 / (12 * (M - 4))) : 
  M = 8 :=
by
  sorry

end original_group_size_l68_68068


namespace Benjie_is_older_by_5_l68_68104

def BenjieAge : ℕ := 6
def MargoFutureAge : ℕ := 4
def YearsToFuture : ℕ := 3

theorem Benjie_is_older_by_5 :
  BenjieAge - (MargoFutureAge - YearsToFuture) = 5 :=
by
  sorry

end Benjie_is_older_by_5_l68_68104


namespace students_per_van_correct_l68_68110

-- Define the conditions.
def num_vans : Nat := 6
def num_minibuses : Nat := 4
def students_per_minibus : Nat := 24
def total_students : Nat := 156

-- Define the number of students on each van is 'V'
def V : Nat := sorry 

-- State the final question/proof.
theorem students_per_van_correct : V = 10 :=
  sorry


end students_per_van_correct_l68_68110


namespace math_problem_l68_68404

theorem math_problem
  (n : ℕ) (d : ℕ)
  (h1 : d ≤ 9)
  (h2 : 3 * n^2 + 2 * n + d = 263)
  (h3 : 3 * n^2 + 2 * n + 4 = 253 + 6 * d) :
  n + d = 11 := 
sorry

end math_problem_l68_68404


namespace probability_of_sum_odd_is_correct_l68_68043

noncomputable def probability_sum_odd : ℚ :=
  let total_balls := 13
  let drawn_balls := 7
  let total_ways := Nat.choose total_balls drawn_balls
  let favorable_ways := 
    Nat.choose 7 5 * Nat.choose 6 2 + 
    Nat.choose 7 3 * Nat.choose 6 4 + 
    Nat.choose 7 1 * Nat.choose 6 6
  favorable_ways / total_ways

theorem probability_of_sum_odd_is_correct :
  probability_sum_odd = 847 / 1716 :=
by
  -- Proof goes here
  sorry

end probability_of_sum_odd_is_correct_l68_68043


namespace max_a_value_l68_68894

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x - 1

theorem max_a_value : 
  ∀ (a : ℝ), 
  (∀ (x : ℝ), x ∈ Set.Icc (1/2) 2 → (a + 1) * x - 1 - Real.log x ≤ 0) → 
  a ≤ 1 - 2 * Real.log 2 := 
by
  sorry

end max_a_value_l68_68894


namespace quadratic_expression_min_value_l68_68680

noncomputable def min_value_quadratic_expression (x y z : ℝ) : ℝ :=
(x + 5) ^ 2 + (y - 1) ^ 2 + (z + 3) ^ 2

theorem quadratic_expression_min_value :
  ∃ x y z : ℝ, x - 2 * y + 2 * z = 5 ∧ min_value_quadratic_expression x y z = 36 :=
sorry

end quadratic_expression_min_value_l68_68680


namespace student_ticket_cost_l68_68620

theorem student_ticket_cost 
  (total_tickets_sold : ℕ) 
  (total_revenue : ℕ) 
  (nonstudent_ticket_cost : ℕ) 
  (student_tickets_sold : ℕ) 
  (cost_per_student_ticket : ℕ) 
  (nonstudent_tickets_sold : ℕ) 
  (H1 : total_tickets_sold = 821) 
  (H2 : total_revenue = 1933)
  (H3 : nonstudent_ticket_cost = 3)
  (H4 : student_tickets_sold = 530) 
  (H5 : nonstudent_tickets_sold = total_tickets_sold - student_tickets_sold)
  (H6 : 530 * cost_per_student_ticket + nonstudent_tickets_sold * 3 = 1933) : 
  cost_per_student_ticket = 2 := 
by
  sorry

end student_ticket_cost_l68_68620


namespace find_first_term_geom_seq_l68_68499

noncomputable def first_term (a r : ℝ) := a

theorem find_first_term_geom_seq 
  (a r : ℝ) 
  (h1 : a * r ^ 3 = 720) 
  (h2 : a * r ^ 6 = 5040) : 
  first_term a r = 720 / 7 := 
sorry

end find_first_term_geom_seq_l68_68499


namespace find_k_l68_68037

theorem find_k (x y k : ℝ) (h1 : x + y = 5 * k) (h2 : x - y = 9 * k) (h3 : x - 2 * y = 22) : k = 2 :=
by
  sorry

end find_k_l68_68037


namespace pythagorean_triplet_unique_solution_l68_68201

-- Define the conditions given in the problem
def is_solution (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧
  Nat.gcd a (Nat.gcd b c) = 1 ∧
  2000 ≤ a ∧ a ≤ 3000 ∧
  2000 ≤ b ∧ b ≤ 3000 ∧
  2000 ≤ c ∧ c ≤ 3000

-- Prove that the only set of integers (a, b, c) meeting the conditions
-- equals the specific tuple (2100, 2059, 2941)
theorem pythagorean_triplet_unique_solution : 
  ∀ a b c : ℕ, is_solution a b c ↔ (a = 2100 ∧ b = 2059 ∧ c = 2941) :=
by
  sorry

end pythagorean_triplet_unique_solution_l68_68201


namespace intersection_P_Q_l68_68619

section set_intersection

variable (x : ℝ)

def P := { x : ℝ | x ≤ 1 }
def Q := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }

theorem intersection_P_Q : { x | x ∈ P ∧ x ∈ Q } = { x | -1 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end set_intersection

end intersection_P_Q_l68_68619


namespace find_int_solutions_l68_68571

theorem find_int_solutions (x : ℤ) :
  (∃ p : ℤ, Prime p ∧ 2*x^2 - x - 36 = p^2) ↔ (x = 5 ∨ x = 13) := 
sorry

end find_int_solutions_l68_68571


namespace best_regression_effect_l68_68565

theorem best_regression_effect (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.36)
  (h2 : R2_2 = 0.95)
  (h3 : R2_3 = 0.74)
  (h4 : R2_4 = 0.81):
  max (max (max R2_1 R2_2) R2_3) R2_4 = 0.95 := by
  sorry

end best_regression_effect_l68_68565


namespace parallelogram_area_l68_68442

theorem parallelogram_area (base height : ℝ) (h_base : base = 24) (h_height : height = 10) :
  base * height = 240 :=
by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l68_68442


namespace total_sums_attempted_l68_68177

-- Define the necessary conditions
def num_sums_right : ℕ := 8
def num_sums_wrong : ℕ := 2 * num_sums_right

-- Define the theorem to prove
theorem total_sums_attempted : num_sums_right + num_sums_wrong = 24 := by
  sorry

end total_sums_attempted_l68_68177


namespace Caleb_pencils_fewer_than_twice_Candy_l68_68889

theorem Caleb_pencils_fewer_than_twice_Candy:
  ∀ (P_Caleb P_Candy: ℕ), 
    P_Candy = 9 → 
    (∃ X, P_Caleb = 2 * P_Candy - X) → 
    P_Caleb + 5 - 10 = 10 → 
    (2 * P_Candy - P_Caleb = 3) :=
by
  intros P_Caleb P_Candy hCandy hCalebLess twCalen
  sorry

end Caleb_pencils_fewer_than_twice_Candy_l68_68889


namespace segment_length_BD_eq_CB_l68_68898

theorem segment_length_BD_eq_CB {AC CB BD x : ℝ}
  (h1 : AC = 4 * CB)
  (h2 : BD = CB)
  (h3 : CB = x) :
  BD = CB := 
by
  -- Proof omitted
  sorry

end segment_length_BD_eq_CB_l68_68898


namespace negation_of_p_implication_q_l68_68013

noncomputable def negation_of_conditions : Prop :=
∀ (a : ℝ), (a > 0 → a^2 > a) ∧ (¬(a > 0) ↔ ¬(a^2 > a)) → ¬(a ≤ 0 → a^2 ≤ a)

theorem negation_of_p_implication_q :
  negation_of_conditions :=
by {
  sorry
}

end negation_of_p_implication_q_l68_68013


namespace fraction_of_budget_is_31_percent_l68_68317

def coffee_pastry_cost (B : ℝ) (c : ℝ) (p : ℝ) :=
  c = 0.25 * (B - p) ∧ p = 0.10 * (B - c)

theorem fraction_of_budget_is_31_percent (B c p : ℝ) (h : coffee_pastry_cost B c p) :
  c + p = 0.31 * B :=
sorry

end fraction_of_budget_is_31_percent_l68_68317


namespace range_of_a_l68_68261

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = Real.sqrt x) :
  (f a < f (a + 1)) ↔ a ∈ Set.Ici (-1) :=
by
  sorry

end range_of_a_l68_68261


namespace solid_is_triangular_prism_l68_68012

-- Given conditions as definitions
def front_view_is_isosceles_triangle (solid : Type) : Prop := 
  -- Define the property that the front view is an isosceles triangle
  sorry

def left_view_is_isosceles_triangle (solid : Type) : Prop := 
  -- Define the property that the left view is an isosceles triangle
  sorry

def top_view_is_circle (solid : Type) : Prop := 
   -- Define the property that the top view is a circle
  sorry

-- Define the property of being a triangular prism
def is_triangular_prism (solid : Type) : Prop :=
  -- Define the property that the solid is a triangular prism
  sorry

-- The main theorem: proving that given the conditions, the solid could be a triangular prism
theorem solid_is_triangular_prism (solid : Type) :
  front_view_is_isosceles_triangle solid ∧ 
  left_view_is_isosceles_triangle solid ∧ 
  top_view_is_circle solid →
  is_triangular_prism solid :=
sorry

end solid_is_triangular_prism_l68_68012


namespace total_boxes_sold_l68_68822

-- Define the number of boxes of plain cookies
def P : ℝ := 793.375

-- Define the combined value of cookies sold
def total_value : ℝ := 1586.75

-- Define the cost per box of each type of cookie
def cost_chocolate_chip : ℝ := 1.25
def cost_plain : ℝ := 0.75

-- State the theorem to prove
theorem total_boxes_sold :
  ∃ C : ℝ, cost_chocolate_chip * C + cost_plain * P = total_value ∧ C + P = 1586.75 :=
by
  sorry

end total_boxes_sold_l68_68822


namespace total_resistance_l68_68493

theorem total_resistance (x y z : ℝ) (R_parallel r : ℝ)
    (hx : x = 3)
    (hy : y = 6)
    (hz : z = 4)
    (hR_parallel : 1 / R_parallel = 1 / x + 1 / y)
    (hr : r = R_parallel + z) :
    r = 6 := by
  sorry

end total_resistance_l68_68493


namespace initial_boys_count_l68_68899

theorem initial_boys_count (b : ℕ) (h1 : b + 10 - 4 - 3 = 17) : b = 14 :=
by
  sorry

end initial_boys_count_l68_68899


namespace quadratic_completing_square_l68_68153

theorem quadratic_completing_square (b c : ℝ) (h : ∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) :
    b + c = -106 :=
sorry

end quadratic_completing_square_l68_68153


namespace find_x_l68_68041

theorem find_x (x y : ℤ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y + x * y = 104) : x = 34 :=
sorry

end find_x_l68_68041


namespace female_democrats_ratio_l68_68601

theorem female_democrats_ratio 
  (M F : ℕ) 
  (H1 : M + F = 660)
  (H2 : (1 / 3 : ℝ) * 660 = 220)
  (H3 : ∃ dem_males : ℕ, dem_males = (1 / 4 : ℝ) * M)
  (H4 : ∃ dem_females : ℕ, dem_females = 110) :
  110 / F = 1 / 2 :=
by
  sorry

end female_democrats_ratio_l68_68601


namespace number_of_players_tournament_l68_68470

theorem number_of_players_tournament (n : ℕ) : 
  (2 * n * (n - 1) = 272) → n = 17 :=
by
  sorry

end number_of_players_tournament_l68_68470


namespace intersection_point_k_value_l68_68132

theorem intersection_point_k_value :
  (∃ (k : ℝ), (∀ (x y : ℝ),
    ((y = 2 * x + 3 ∧ y = k * x + 2) → (x = 1 ∧ y = 5))) → k = 3) :=
sorry

end intersection_point_k_value_l68_68132


namespace ramsey_6_3_3_l68_68222

open Classical

theorem ramsey_6_3_3 (G : SimpleGraph (Fin 6)) :
  ∃ (A : Finset (Fin 6)), A.card = 3 ∧ (∀ (x y : Fin 6), x ∈ A → y ∈ A → x ≠ y → G.Adj x y) ∨ ∃ (B : Finset (Fin 6)), B.card = 3 ∧ (∀ (x y : Fin 6), x ∈ B → y ∈ B → x ≠ y → ¬ G.Adj x y) :=
by
  sorry

end ramsey_6_3_3_l68_68222


namespace ducks_percentage_non_heron_birds_l68_68946

theorem ducks_percentage_non_heron_birds
  (total_birds : ℕ)
  (geese_percent pelicans_percent herons_percent ducks_percent : ℝ)
  (H_geese : geese_percent = 20 / 100)
  (H_pelicans: pelicans_percent = 40 / 100)
  (H_herons : herons_percent = 15 / 100)
  (H_ducks : ducks_percent = 25 / 100)
  (hnz : total_birds ≠ 0) :
  (ducks_percent / (1 - herons_percent)) * 100 = 30 :=
by
  sorry

end ducks_percentage_non_heron_birds_l68_68946


namespace percent_eighth_graders_combined_l68_68911

theorem percent_eighth_graders_combined (p_students : ℕ) (m_students : ℕ)
  (p_grade8_percent : ℚ) (m_grade8_percent : ℚ) :
  p_students = 160 → m_students = 250 →
  p_grade8_percent = 18 / 100 → m_grade8_percent = 22 / 100 →
  100 * (p_grade8_percent * p_students + m_grade8_percent * m_students) / (p_students + m_students) = 20 := 
by
  intros h1 h2 h3 h4
  sorry

end percent_eighth_graders_combined_l68_68911


namespace number_of_teams_l68_68352

theorem number_of_teams (x : ℕ) (h : x * (x - 1) / 2 = 15) : x = 6 :=
sorry

end number_of_teams_l68_68352


namespace fractional_equation_solution_l68_68211

theorem fractional_equation_solution (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (3 / (x + 1) = 2 / (x - 1)) → (x = 5) :=
sorry

end fractional_equation_solution_l68_68211


namespace find_a8_l68_68485

-- Define the arithmetic sequence and the given conditions
variable {α : Type} [AddCommGroup α] [MulAction ℤ α]

def is_arithmetic_sequence (a : ℕ → α) := ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
variables {a : ℕ → ℝ}
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 5 + a 6 = 22
axiom h3 : a 3 = 7

theorem find_a8 : a 8 = 15 :=
by
  -- Proof omitted
  sorry

end find_a8_l68_68485


namespace total_ducks_and_ducklings_l68_68175

theorem total_ducks_and_ducklings :
  let ducks1 := 2
  let ducklings1 := 5
  let ducks2 := 6
  let ducklings2 := 3
  let ducks3 := 9
  let ducklings3 := 6
  (ducks1 + ducks2 + ducks3) + (ducks1 * ducklings1 + ducks2 * ducklings2 + ducks3 * ducklings3) = 99 :=
by
  sorry

end total_ducks_and_ducklings_l68_68175


namespace equation1_equation2_equation3_equation4_l68_68174

theorem equation1 (x : ℝ) : (x - 1) ^ 2 - 5 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
  sorry

theorem equation2 (x : ℝ) : x * (x + 4) = -3 * (x + 4) ↔ x = -4 ∨ x = -3 := by
  sorry

theorem equation3 (y : ℝ) : 2 * y ^ 2 - 5 * y + 2 = 0 ↔ y = 1 / 2 ∨ y = 2 := by
  sorry

theorem equation4 (m : ℝ) : 2 * m ^ 2 - 7 * m - 3 = 0 ↔ m = (7 + Real.sqrt 73) / 4 ∨ m = (7 - Real.sqrt 73) / 4 := by
  sorry

end equation1_equation2_equation3_equation4_l68_68174


namespace ratio_of_costs_l68_68315

-- Definitions based on conditions
def quilt_length : Nat := 16
def quilt_width : Nat := 20
def patch_area : Nat := 4
def first_10_patch_cost : Nat := 10
def total_cost : Nat := 450

-- Theorem we need to prove
theorem ratio_of_costs : (total_cost - 10 * first_10_patch_cost) / (10 * first_10_patch_cost) = 7 / 2 := by
  sorry

end ratio_of_costs_l68_68315


namespace hexagon_perimeter_is_42_l68_68319

-- Define the side length of the hexagon
def side_length : ℕ := 7

-- Define the number of sides of the hexagon
def num_sides : ℕ := 6

-- Define the perimeter of the hexagon
def hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) : ℕ :=
  num_sides * side_length

-- The theorem to prove
theorem hexagon_perimeter_is_42 : hexagon_perimeter side_length num_sides = 42 :=
by
  sorry

end hexagon_perimeter_is_42_l68_68319


namespace verify_monomial_properties_l68_68357

def monomial : ℚ := -3/5 * (1:ℚ)^1 * (2:ℚ)^2

def coefficient (m : ℚ) : ℚ := -3/5  -- The coefficient of the monomial
def degree (m : ℚ) : ℕ := 3          -- The degree of the monomial

theorem verify_monomial_properties :
  coefficient monomial = -3/5 ∧ degree monomial = 3 :=
by
  sorry

end verify_monomial_properties_l68_68357


namespace parcel_post_cost_l68_68305

def indicator (P : ℕ) : ℕ := if P >= 5 then 1 else 0

theorem parcel_post_cost (P : ℕ) : 
  P ≥ 0 →
  (C : ℕ) = 15 + 5 * (P - 1) - 8 * indicator P :=
sorry

end parcel_post_cost_l68_68305


namespace incorrect_inequality_exists_l68_68728

theorem incorrect_inequality_exists :
  ∃ (x y : ℝ), x < y ∧ x^2 ≥ y^2 :=
by {
  sorry
}

end incorrect_inequality_exists_l68_68728


namespace product_of_two_numbers_l68_68124

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 8) (h2 : Nat.lcm a b = 48) : a * b = 384 :=
by
  sorry

end product_of_two_numbers_l68_68124


namespace range_of_k_l68_68944

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k}

-- State the theorem
theorem range_of_k (k : ℝ) : (M ∩ N k).Nonempty ↔ k ∈ Set.Ici (-1) :=
by
  sorry

end range_of_k_l68_68944


namespace joe_height_is_82_l68_68538

-- Given the conditions:
def Sara_height (x : ℝ) : Prop := true

def Joe_height (j : ℝ) (x : ℝ) : Prop := j = 6 + 2 * x

def combined_height (j : ℝ) (x : ℝ) : Prop := j + x = 120

-- We need to prove:
theorem joe_height_is_82 (x j : ℝ) 
  (h1 : combined_height j x)
  (h2 : Joe_height j x) :
  j = 82 := 
by 
  sorry

end joe_height_is_82_l68_68538


namespace slope_of_line_l68_68067

variable (x y : ℝ)

def line_equation : Prop := 4 * y = -5 * x + 8

theorem slope_of_line (h : line_equation x y) :
  ∃ m b, y = m * x + b ∧ m = -5/4 :=
by
  sorry

end slope_of_line_l68_68067


namespace pima_initial_investment_l68_68083

/-- Pima's initial investment in Ethereum. The investment value gained 25% in the first week and 50% of its current value in the second week. The final investment value is $750. -/
theorem pima_initial_investment (I : ℝ) 
  (h1 : 1.25 * I * 1.5 = 750) : I = 400 :=
sorry

end pima_initial_investment_l68_68083


namespace largest_d_l68_68695

variable (a b c d : ℝ)

theorem largest_d (h : a + 1 = b - 2 ∧ b - 2 = c + 3 ∧ c + 3 = d - 4) : 
  d >= a ∧ d >= b ∧ d >= c :=
by
  sorry

end largest_d_l68_68695


namespace total_spokes_in_garage_l68_68040

def bicycle1_front_spokes : ℕ := 16
def bicycle1_back_spokes : ℕ := 18
def bicycle2_front_spokes : ℕ := 20
def bicycle2_back_spokes : ℕ := 22
def bicycle3_front_spokes : ℕ := 24
def bicycle3_back_spokes : ℕ := 26
def bicycle4_front_spokes : ℕ := 28
def bicycle4_back_spokes : ℕ := 30
def tricycle_front_spokes : ℕ := 32
def tricycle_middle_spokes : ℕ := 34
def tricycle_back_spokes : ℕ := 36

theorem total_spokes_in_garage :
  bicycle1_front_spokes + bicycle1_back_spokes +
  bicycle2_front_spokes + bicycle2_back_spokes +
  bicycle3_front_spokes + bicycle3_back_spokes +
  bicycle4_front_spokes + bicycle4_back_spokes +
  tricycle_front_spokes + tricycle_middle_spokes + tricycle_back_spokes = 286 :=
by
  sorry

end total_spokes_in_garage_l68_68040


namespace max_area_cross_section_of_prism_l68_68722

noncomputable def prism_vertex_A : ℝ × ℝ × ℝ := (3, 0, 0)
noncomputable def prism_vertex_B : ℝ × ℝ × ℝ := (-3, 0, 0)
noncomputable def prism_vertex_C : ℝ × ℝ × ℝ := (0, 3 * Real.sqrt 3, 0)
noncomputable def plane_eq (x y z : ℝ) : ℝ := 2 * x - 3 * y + 6 * z

-- Statement
theorem max_area_cross_section_of_prism (h : ℝ) (A B C : ℝ × ℝ × ℝ)
  (plane : ℝ → ℝ → ℝ → ℝ) (cond_h : h = 5)
  (cond_A : A = prism_vertex_A) (cond_B : B = prism_vertex_B) 
  (cond_C : C = prism_vertex_C) (cond_plane : ∀ x y z, plane x y z = 2 * x - 3 * y + 6 * z - 30) : 
  ∃ cross_section : ℝ, cross_section = 0 :=
by
  sorry

end max_area_cross_section_of_prism_l68_68722


namespace probability_second_year_not_science_l68_68768

def total_students := 2000

def first_year := 600
def first_year_science := 300
def first_year_arts := 200
def first_year_engineering := 100

def second_year := 450
def second_year_science := 250
def second_year_arts := 150
def second_year_engineering := 50

def third_year := 550
def third_year_science := 300
def third_year_arts := 200
def third_year_engineering := 50

def postgraduate := 400
def postgraduate_science := 200
def postgraduate_arts := 100
def postgraduate_engineering := 100

def not_third_year_not_science :=
  (first_year_arts + first_year_engineering) +
  (second_year_arts + second_year_engineering) +
  (postgraduate_arts + postgraduate_engineering)

def second_year_not_science := second_year_arts + second_year_engineering

theorem probability_second_year_not_science :
  (second_year_not_science / not_third_year_not_science : ℚ) = (2 / 7 : ℚ) :=
by
  let total := (first_year_arts + first_year_engineering) + (second_year_arts + second_year_engineering) + (postgraduate_arts + postgraduate_engineering)
  have not_third_year_not_science : total = 300 + 200 + 200 := by sorry
  have second_year_not_science_eq : second_year_not_science = 200 := by sorry
  sorry

end probability_second_year_not_science_l68_68768


namespace average_goals_increase_l68_68155

theorem average_goals_increase (A : ℚ) (h1 : 4 * A + 2 = 4) : (4 / 5 - A) = 0.3 := by
  sorry

end average_goals_increase_l68_68155


namespace max_quotient_l68_68779

theorem max_quotient (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) : 
  ∃ max_val, max_val = 225 ∧ ∀ (x y : ℝ), (100 ≤ x ∧ x ≤ 300) ∧ (500 ≤ y ∧ y ≤ 1500) → (y^2 / x^2) ≤ max_val := 
by
  use 225
  sorry

end max_quotient_l68_68779


namespace sin_30_eq_half_l68_68054

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l68_68054


namespace three_digit_number_cubed_sum_l68_68199

theorem three_digit_number_cubed_sum {a b c : ℕ} (h₁ : 1 ≤ a ∧ a ≤ 9)
                                      (h₂ : 0 ≤ b ∧ b ≤ 9)
                                      (h₃ : 0 ≤ c ∧ c ≤ 9) :
  (100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c ≤ 999) →
  (100 * a + 10 * b + c = (a + b + c) ^ 3) →
  (100 * a + 10 * b + c = 512) :=
by
  sorry

end three_digit_number_cubed_sum_l68_68199


namespace find_g_l68_68744

-- Definitions for functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := sorry -- We will define this later in the statement

theorem find_g :
  (∀ x : ℝ, g (x + 2) = f x) →
  (∀ x : ℝ, g x = 2 * x - 1) :=
by
  intros h
  sorry

end find_g_l68_68744


namespace above_line_sign_l68_68237

theorem above_line_sign (A B C x y : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) 
(h_above : ∃ y₁, Ax + By₁ + C = 0 ∧ y > y₁) : 
  (Ax + By + C > 0 ∧ B > 0) ∨ (Ax + By + C < 0 ∧ B < 0) := 
by
  sorry

end above_line_sign_l68_68237


namespace domain_of_function_l68_68084

theorem domain_of_function :
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) ↔ (1 - x ≥ 0 ∧ x ≠ 0) :=
by
  sorry

end domain_of_function_l68_68084


namespace find_y_l68_68389

theorem find_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 2 + 1 / y) (h2 : y = 2 + 1 / x) :
  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2 :=
by
  sorry

end find_y_l68_68389


namespace minimum_value_of_fraction_sum_l68_68345

open Real

theorem minimum_value_of_fraction_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 2) : 
    6 ≤ (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) := by 
  sorry

end minimum_value_of_fraction_sum_l68_68345


namespace part1_l68_68913

-- Define the vectors a and b
def a : ℝ × ℝ := (-3, 4)
def b : ℝ × ℝ := (2, -1)
-- Define the vectors a - x b and a - b
def vec1 (x : ℝ) : ℝ × ℝ := (a.1 - x * b.1, a.2 - x * b.2)
def vec2 : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
-- Define the dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 

-- Main theorem: prove that the vectors being perpendicular implies x = -7/3
theorem part1 (x : ℝ) : dot_product (vec1 x) vec2 = 0 → x = -7 / 3 :=
by
  sorry

end part1_l68_68913


namespace problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_problem_7_l68_68771

theorem problem_1 (x y z : ℝ) (h : z = (x + y) / 2) : z = (x + y) / 2 :=
sorry

theorem problem_2 (x y w : ℝ) (h1 : w = x + y) : w = x + y :=
sorry

theorem problem_3 (x w y : ℝ) (h1 : w = x + y) (h2 : y = w - x) : y = w - x :=
sorry

theorem problem_4 (x z v : ℝ) (h1 : z = (x + y) / 2) (h2 : v = 2 * z) : v = 2 * (x + (x + y) / 2) :=
sorry

theorem problem_5 (x z u : ℝ) (h : u = - (x + z) / 5) : x + z + 5 * u = 0 :=
sorry

theorem problem_6 (y z t : ℝ) (h : t = (6 + y + z) / 2) : t = (6 + y + z) / 2 :=
sorry

theorem problem_7 (y z s : ℝ) (h : y + z + 4 * s - 10 = 0) : y + z + 4 * s - 10 = 0 :=
sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_problem_7_l68_68771


namespace geom_seq_min_value_l68_68719

theorem geom_seq_min_value (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a n = a 1 * q ^ (n - 1))
  (h_condition : a 7 = a 6 + 2 * a 5)
  (h_mult : ∃ m n, m ≠ n ∧ a m * a n = 16 * (a 1) ^ 2) :
  ∃ (m n : ℕ), m ≠ n ∧ m + n = 6 ∧ (1 / m : ℝ) + (4 / n : ℝ) = 3 / 2 :=
by
  sorry

end geom_seq_min_value_l68_68719


namespace f_a_minus_2_lt_0_l68_68395

theorem f_a_minus_2_lt_0 (f : ℝ → ℝ) (m a : ℝ) (h1 : ∀ x, f x = (m + 1 - x) * (x - m + 1)) (h2 : f a > 0) : f (a - 2) < 0 := 
sorry

end f_a_minus_2_lt_0_l68_68395


namespace value_of_expression_l68_68367

-- Let's define the sequences and sums based on the conditions in a)
def sum_of_evens (n : ℕ) : ℕ :=
  n * (n + 1)

def sum_of_multiples_of_three (p : ℕ) : ℕ :=
  3 * (p * (p + 1)) / 2

def sum_of_odds (m : ℕ) : ℕ :=
  m * m

-- Now let's formulate the problem statement as a theorem.
theorem value_of_expression : 
  sum_of_evens 200 - sum_of_multiples_of_three 100 - sum_of_odds 148 = 3146 :=
  by
  sorry

end value_of_expression_l68_68367


namespace geometric_sequence_fourth_term_l68_68074

theorem geometric_sequence_fourth_term (x : ℚ) (r : ℚ)
  (h1 : x ≠ 0)
  (h2 : x ≠ -1)
  (h3 : 3 * x + 3 = r * x)
  (h4 : 5 * x + 5 = r * (3 * x + 3)) :
  r^3 * (5 * x + 5) = -125 / 12 :=
by
  sorry

end geometric_sequence_fourth_term_l68_68074


namespace callie_caught_frogs_l68_68715

theorem callie_caught_frogs (A Q B C : ℝ) 
  (hA : A = 2)
  (hQ : Q = 2 * A)
  (hB : B = 3 * Q)
  (hC : C = (5 / 8) * B) : 
  C = 7.5 := by
  sorry

end callie_caught_frogs_l68_68715


namespace division_of_cookies_l68_68077

theorem division_of_cookies (n p : Nat) (h1 : n = 24) (h2 : p = 6) : n / p = 4 :=
by sorry

end division_of_cookies_l68_68077


namespace soup_problem_l68_68368

def cans_needed_for_children (children : ℕ) (children_per_can : ℕ) : ℕ :=
  children / children_per_can

def remaining_cans (initial_cans used_cans : ℕ) : ℕ :=
  initial_cans - used_cans

def half_cans (cans : ℕ) : ℕ :=
  cans / 2

def adults_fed (cans : ℕ) (adults_per_can : ℕ) : ℕ :=
  cans * adults_per_can

theorem soup_problem
  (initial_cans : ℕ)
  (children_fed : ℕ)
  (children_per_can : ℕ)
  (adults_per_can : ℕ)
  (reserved_fraction : ℕ)
  (hreserved : reserved_fraction = 2)
  (hintial : initial_cans = 8)
  (hchildren : children_fed = 24)
  (hchildren_per_can : children_per_can = 6)
  (hadults_per_can : adults_per_can = 4) :
  adults_fed (half_cans (remaining_cans initial_cans (cans_needed_for_children children_fed children_per_can))) adults_per_can = 8 :=
by
  sorry

end soup_problem_l68_68368


namespace cost_price_of_computer_table_l68_68458

theorem cost_price_of_computer_table (SP : ℝ) (h1 : SP = 1.15 * CP ∧ SP = 6400) : CP = 5565.22 :=
by
  sorry

end cost_price_of_computer_table_l68_68458


namespace total_songs_listened_l68_68476

theorem total_songs_listened (vivian_daily : ℕ) (fewer_songs : ℕ) (days_in_june : ℕ) (weekend_days : ℕ) :
  vivian_daily = 10 →
  fewer_songs = 2 →
  days_in_june = 30 →
  weekend_days = 8 →
  (vivian_daily * (days_in_june - weekend_days)) + ((vivian_daily - fewer_songs) * (days_in_june - weekend_days)) = 396 := 
by
  intros h1 h2 h3 h4
  sorry

end total_songs_listened_l68_68476


namespace compare_sqrt_expression_l68_68060

theorem compare_sqrt_expression : 2 * Real.sqrt 3 < 3 * Real.sqrt 2 := 
sorry

end compare_sqrt_expression_l68_68060


namespace graduates_distribution_l68_68515

theorem graduates_distribution (n : ℕ) (k : ℕ)
    (h_n : n = 5) (h_k : k = 3)
    (h_dist : ∀ e : Fin k, ∃ g : Finset (Fin n), g.card ≥ 1) :
    ∃ d : ℕ, d = 150 :=
by
  have h_distribution := 150
  use h_distribution
  sorry

end graduates_distribution_l68_68515


namespace high_sulfur_oil_samples_l68_68048

/-- The number of high-sulfur oil samples in a container with the given conditions. -/
theorem high_sulfur_oil_samples (total_samples : ℕ) 
    (heavy_oil_freq : ℚ) (light_low_sulfur_freq : ℚ)
    (no_heavy_low_sulfur: true) (almost_full : total_samples = 198)
    (heavy_oil_freq_value : heavy_oil_freq = 1 / 9)
    (light_low_sulfur_freq_value : light_low_sulfur_freq = 11 / 18) :
    (22 + 68) = 90 := 
by
  sorry

end high_sulfur_oil_samples_l68_68048


namespace uniformColorGridPossible_l68_68928

noncomputable def canPaintUniformColor (n : Nat) (G : Matrix (Fin n) (Fin n) (Fin (n - 1))) : Prop :=
  ∀ (row : Fin n), ∃ (c : Fin (n - 1)), ∀ (col : Fin n), G row col = c

theorem uniformColorGridPossible (n : Nat) (G : Matrix (Fin n) (Fin n) (Fin (n - 1))) :
  (∀ r : Fin n, ∃ c₁ c₂ : Fin n, c₁ ≠ c₂ ∧ G r c₁ = G r c₂) ∧
  (∀ c : Fin n, ∃ r₁ r₂ : Fin n, r₁ ≠ r₂ ∧ G r₁ c = G r₂ c) →
  ∃ c : Fin (n - 1), ∀ (row col : Fin n), G row col = c := by
  sorry

end uniformColorGridPossible_l68_68928


namespace greatest_integer_value_x_l68_68484

theorem greatest_integer_value_x :
  ∀ x : ℤ, (∃ k : ℤ, x^2 + 2 * x + 9 = k * (x - 5)) ↔ x ≤ 49 :=
by
  sorry

end greatest_integer_value_x_l68_68484


namespace laura_walk_distance_l68_68798

theorem laura_walk_distance 
  (east_blocks : ℕ) 
  (north_blocks : ℕ) 
  (block_length_miles : ℕ → ℝ) 
  (h_east_blocks : east_blocks = 8) 
  (h_north_blocks : north_blocks = 14) 
  (h_block_length_miles : ∀ b : ℕ, b = 1 → block_length_miles b = 1 / 4) 
  : (east_blocks + north_blocks) * block_length_miles 1 = 5.5 := 
by 
  sorry

end laura_walk_distance_l68_68798


namespace union_of_M_and_N_l68_68257

def M : Set Int := {-1, 0, 1}
def N : Set Int := {0, 1, 2}

theorem union_of_M_and_N :
  M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end union_of_M_and_N_l68_68257


namespace complex_vector_PQ_l68_68614

theorem complex_vector_PQ (P Q : ℂ) (hP : P = 3 + 1 * I) (hQ : Q = 2 + 3 * I) : 
  (Q - P) = -1 + 2 * I :=
by sorry

end complex_vector_PQ_l68_68614


namespace polynomial_sum_at_points_l68_68696

def P (x : ℝ) : ℝ := x^5 - 1.7 * x^3 + 2.5

theorem polynomial_sum_at_points :
  P 19.1 + P (-19.1) = 5 := by
  sorry

end polynomial_sum_at_points_l68_68696


namespace adults_not_wearing_blue_l68_68800

-- Conditions
def children : ℕ := 45
def adults : ℕ := children / 3
def adults_wearing_blue : ℕ := adults / 3

-- Theorem Statement
theorem adults_not_wearing_blue :
  adults - adults_wearing_blue = 10 :=
sorry

end adults_not_wearing_blue_l68_68800


namespace find_angle_l68_68398

-- Given the complement condition
def complement_condition (x : ℝ) : Prop :=
  x + 2 * (4 * x + 10) = 90

-- Proving the degree measure of the angle
theorem find_angle (x : ℝ) : complement_condition x → x = 70 / 9 := by
  intro hc
  sorry

end find_angle_l68_68398


namespace intermediate_root_exists_l68_68474

open Polynomial

theorem intermediate_root_exists
  (a b c x1 x2 : ℝ) 
  (h1 : a * x1^2 + b * x1 + c = 0) 
  (h2 : -a * x2^2 + b * x2 + c = 0) :
  ∃ x3 : ℝ, (a / 2) * x3^2 + b * x3 + c = 0 ∧ (x1 ≤ x3 ∧ x3 ≤ x2 ∨ x1 ≥ x3 ∧ x3 ≥ x2) :=
sorry

end intermediate_root_exists_l68_68474


namespace min_shirts_to_save_money_l68_68985

theorem min_shirts_to_save_money :
  let acme_cost (x : ℕ) := 75 + 12 * x
  let gamma_cost (x : ℕ) := 18 * x
  ∀ x : ℕ, acme_cost x < gamma_cost x → x ≥ 13 := 
by
  intros
  sorry

end min_shirts_to_save_money_l68_68985


namespace club_committee_selections_l68_68030

theorem club_committee_selections : (Nat.choose 18 3) = 816 := by
  sorry

end club_committee_selections_l68_68030


namespace improper_fraction_2012a_div_b_l68_68472

theorem improper_fraction_2012a_div_b
  (a b : ℕ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : (a : ℚ) / b < (a + 1 : ℚ) / (b + 1)) :
  2012 * a > b :=
by 
  sorry

end improper_fraction_2012a_div_b_l68_68472


namespace total_number_of_candles_l68_68659

theorem total_number_of_candles
  (candles_bedroom : ℕ)
  (candles_living_room : ℕ)
  (candles_donovan : ℕ)
  (h1 : candles_bedroom = 20)
  (h2 : candles_bedroom = 2 * candles_living_room)
  (h3 : candles_donovan = 20) :
  candles_bedroom + candles_living_room + candles_donovan = 50 :=
by
  sorry

end total_number_of_candles_l68_68659


namespace books_leftover_l68_68271

theorem books_leftover (boxes : ℕ) (books_per_box : ℕ) (new_box_capacity : ℕ) 
  (h1 : boxes = 1575) (h2 : books_per_box = 45) (h3 : new_box_capacity = 50) :
  ((boxes * books_per_box) % new_box_capacity) = 25 :=
by
  sorry

end books_leftover_l68_68271


namespace min_value_of_function_l68_68841

theorem min_value_of_function (h : 0 < x ∧ x < 1) : 
  ∃ (y : ℝ), (∀ z : ℝ, z = (4 / x + 1 / (1 - x)) → y ≤ z) ∧ y = 9 :=
by
  sorry

end min_value_of_function_l68_68841


namespace expression_simplifies_l68_68443

variable {a b : ℚ}
variable (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b)

theorem expression_simplifies : (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b := by
  -- TODO: Proof goes here
  sorry

end expression_simplifies_l68_68443


namespace find_number_of_girls_in_class_l68_68915

variable (G : ℕ)

def number_of_ways_to_select_two_boys (n : ℕ) : ℕ := Nat.choose n 2

theorem find_number_of_girls_in_class 
  (boys : ℕ := 13) 
  (ways_to_select_students : ℕ := 780) 
  (ways_to_select_two_boys : ℕ := number_of_ways_to_select_two_boys boys) :
  G * ways_to_select_two_boys = ways_to_select_students → G = 10 := 
by
  sorry

end find_number_of_girls_in_class_l68_68915


namespace value_of_fraction_l68_68401

variable {x y : ℝ}

theorem value_of_fraction (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 := 
sorry

end value_of_fraction_l68_68401


namespace num_factors_1728_l68_68507

open Nat

noncomputable def num_factors (n : ℕ) : ℕ :=
  (6 + 1) * (3 + 1)

theorem num_factors_1728 : 
  num_factors 1728 = 28 := by
  sorry

end num_factors_1728_l68_68507


namespace factorize_expression_l68_68602

theorem factorize_expression (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := sorry

end factorize_expression_l68_68602


namespace determine_angle_C_in_DEF_l68_68703

def Triangle := Type

structure TriangleProps (T : Triangle) :=
  (right_angle : Prop)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)

axiom triangle_ABC : Triangle
axiom triangle_DEF : Triangle

axiom ABC_props : TriangleProps triangle_ABC
axiom DEF_props : TriangleProps triangle_DEF

noncomputable def similar (T1 T2 : Triangle) : Prop := sorry

theorem determine_angle_C_in_DEF
  (h1 : ABC_props.right_angle = true)
  (h2 : ABC_props.angle_A = 30)
  (h3 : DEF_props.right_angle = true)
  (h4 : DEF_props.angle_B = 60)
  (h5 : similar triangle_ABC triangle_DEF) :
  DEF_props.angle_C = 30 :=
sorry

end determine_angle_C_in_DEF_l68_68703


namespace banquet_food_consumption_l68_68593

theorem banquet_food_consumption (n : ℕ) (food_per_guest : ℕ) (total_food : ℕ) 
  (h1 : ∀ g : ℕ, g ≤ n -> g * food_per_guest ≤ total_food)
  (h2 : n = 169) 
  (h3 : food_per_guest = 2) :
  total_food = 338 := 
sorry

end banquet_food_consumption_l68_68593


namespace fraction_meaningful_l68_68844

theorem fraction_meaningful (x : ℝ) : (∃ z, z = 3 / (x - 4)) ↔ x ≠ 4 :=
by
  sorry

end fraction_meaningful_l68_68844


namespace dan_present_age_l68_68636

theorem dan_present_age : ∃ x : ℕ, (x + 18 = 8 * (x - 3)) ∧ x = 6 :=
by
  -- We skip the proof steps
  sorry

end dan_present_age_l68_68636


namespace min_rectangles_needed_l68_68535

theorem min_rectangles_needed : ∀ (n : ℕ), n = 12 → (n * n) / (3 * 2) = 24 :=
by sorry

end min_rectangles_needed_l68_68535


namespace charitable_woman_l68_68829

theorem charitable_woman (initial_pennies : ℕ) 
  (farmer_share : ℕ) (beggar_share : ℕ) (boy_share : ℕ) (left_pennies : ℕ) 
  (h1 : initial_pennies = 42)
  (h2 : farmer_share = (initial_pennies / 2 + 1))
  (h3 : beggar_share = ((initial_pennies - farmer_share) / 2 + 2))
  (h4 : boy_share = ((initial_pennies - farmer_share - beggar_share) / 2 + 3))
  (h5 : left_pennies = initial_pennies - farmer_share - beggar_share - boy_share) : 
  left_pennies = 1 :=
by
  sorry

end charitable_woman_l68_68829


namespace train_speed_kmph_l68_68245

def length_of_train : ℝ := 120
def length_of_bridge : ℝ := 255.03
def time_to_cross : ℝ := 30

theorem train_speed_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross * 3.6 = 45.0036 :=
by
  sorry

end train_speed_kmph_l68_68245


namespace g_15_33_eq_165_l68_68578

noncomputable def g : ℕ → ℕ → ℕ := sorry

axiom g_self (x : ℕ) : g x x = x
axiom g_comm (x y : ℕ) : g x y = g y x
axiom g_equation (x y : ℕ) : (x + y) * g x y = y * g x (x + y)

theorem g_15_33_eq_165 : g 15 33 = 165 := by sorry

end g_15_33_eq_165_l68_68578


namespace calculation_result_l68_68789

theorem calculation_result : ((55 * 45 - 37 * 43) - (3 * 221 + 1)) / 22 = 10 := by
  sorry

end calculation_result_l68_68789


namespace product_of_two_numbers_l68_68206

noncomputable def leastCommonMultiple (a b : ℕ) : ℕ :=
  Nat.lcm a b

noncomputable def greatestCommonDivisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem product_of_two_numbers (a b : ℕ) :
  leastCommonMultiple a b = 36 ∧ greatestCommonDivisor a b = 6 → a * b = 216 := by
  sorry

end product_of_two_numbers_l68_68206


namespace technicians_count_l68_68034

-- Define the conditions
def avg_sal_all (total_workers : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 850

def avg_sal_technicians (teches : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 1000

def avg_sal_rest (others : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 780

-- The main theorem to prove
theorem technicians_count (total_workers : ℕ)
  (teches others : ℕ)
  (total_salary : ℕ) :
  total_workers = 22 →
  total_salary = 850 * 22 →
  avg_sal_all total_workers 850 →
  avg_sal_technicians teches 1000 →
  avg_sal_rest others 780 →
  teches + others = total_workers →
  1000 * teches + 780 * others = total_salary →
  teches = 7 :=
by
  intros
  sorry

end technicians_count_l68_68034


namespace common_difference_of_arithmetic_sequence_l68_68679

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ) (h_arith_seq : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h1 : a 7 - 2 * a 4 = -1)
  (h2 : a 3 = 0) :
  (a 2 - a 1) = - 1 / 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l68_68679


namespace point_A_in_first_quadrant_l68_68150

def point_in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem point_A_in_first_quadrant : point_in_first_quadrant 1 2 := by
  sorry

end point_A_in_first_quadrant_l68_68150


namespace simplify_exponent_expression_l68_68949

theorem simplify_exponent_expression : 2000 * (2000 ^ 2000) = 2000 ^ 2001 :=
by sorry

end simplify_exponent_expression_l68_68949


namespace no_representation_of_expr_l68_68125

theorem no_representation_of_expr :
  ¬ ∃ f g : ℝ → ℝ, (∀ x y : ℝ, 1 + x ^ 2016 * y ^ 2016 = f x * g y) :=
by
  sorry

end no_representation_of_expr_l68_68125


namespace incenter_divides_segment_l68_68145

variables (A B C I M : Type) (R r : ℝ)

-- Definitions based on conditions
def is_incenter (I : Type) (A B C : Type) : Prop := sorry
def is_circumcircle (C : Type) : Prop := sorry
def angle_bisector_intersects_at (A B C M : Type) : Prop := sorry
def divides_segment (I M : Type) (a b : ℝ) : Prop := sorry

-- Proof problem statement
theorem incenter_divides_segment (h1 : is_circumcircle C)
                                   (h2 : is_incenter I A B C)
                                   (h3 : angle_bisector_intersects_at A B C M)
                                   (h4 : divides_segment I M a b) :
  a * b = 2 * R * r :=
sorry

end incenter_divides_segment_l68_68145


namespace time_to_empty_tank_by_leakage_l68_68823

theorem time_to_empty_tank_by_leakage (R_t R_l : ℝ) (h1 : R_t = 1 / 12) (h2 : R_t - R_l = 1 / 18) :
  (1 / R_l) = 36 :=
by
  sorry

end time_to_empty_tank_by_leakage_l68_68823


namespace unique_positive_solution_l68_68119

theorem unique_positive_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end unique_positive_solution_l68_68119


namespace correct_calculation_l68_68184

theorem correct_calculation (y : ℤ) (h : (y + 4) * 5 = 140) : 5 * y + 4 = 124 :=
by {
  sorry
}

end correct_calculation_l68_68184


namespace find_a_for_square_binomial_l68_68343

theorem find_a_for_square_binomial (a r s : ℝ) 
  (h1 : ax^2 + 18 * x + 9 = (r * x + s)^2)
  (h2 : a = r^2)
  (h3 : 2 * r * s = 18)
  (h4 : s^2 = 9) : 
  a = 9 := 
by sorry

end find_a_for_square_binomial_l68_68343


namespace fish_population_l68_68598

theorem fish_population (N : ℕ) (h1 : 50 > 0) (h2 : N > 0) 
  (tagged_first_catch : ℕ) (total_first_catch : ℕ)
  (tagged_second_catch : ℕ) (total_second_catch : ℕ)
  (h3 : tagged_first_catch = 50)
  (h4 : total_first_catch = 50)
  (h5 : tagged_second_catch = 2)
  (h6 : total_second_catch = 50)
  (h_percent : (tagged_first_catch : ℝ) / (N : ℝ) = (tagged_second_catch : ℝ) / (total_second_catch : ℝ))
  : N = 1250 := 
  by sorry

end fish_population_l68_68598


namespace sins_prayers_l68_68475

structure Sins :=
  (pride : Nat)
  (slander : Nat)
  (laziness : Nat)
  (adultery : Nat)
  (gluttony : Nat)
  (self_love : Nat)
  (jealousy : Nat)
  (malicious_gossip : Nat)

def prayer_requirements (s : Sins) : Nat × Nat × Nat :=
  ( s.pride + 2 * s.laziness + 10 * s.adultery + s.gluttony,
    2 * s.pride + 2 * s.slander + 10 * s.adultery + 3 * s.self_love + 3 * s.jealousy + 7 * s.malicious_gossip,
    7 * s.slander + 10 * s.adultery + s.self_love + 2 * s.malicious_gossip )

theorem sins_prayers (sins : Sins) :
  sins.pride = 0 ∧
  sins.slander = 1 ∧
  sins.laziness = 0 ∧
  sins.adultery = 0 ∧
  sins.gluttony = 9 ∧
  sins.self_love = 1 ∧
  sins.jealousy = 0 ∧
  sins.malicious_gossip = 2 ∧
  (sins.pride + sins.slander + sins.laziness + sins.adultery + sins.gluttony + sins.self_love + sins.jealousy + sins.malicious_gossip = 12) ∧
  prayer_requirements sins = (9, 12, 10) :=
  by
  sorry

end sins_prayers_l68_68475


namespace rhombus_area_l68_68826

theorem rhombus_area (side diagonal₁ : ℝ) (h_side : side = 20) (h_diagonal₁ : diagonal₁ = 16) : 
  ∃ (diagonal₂ : ℝ), (2 * diagonal₂ * diagonal₂ + 8 * 8 = side * side) ∧ 
  (1 / 2 * diagonal₁ * diagonal₂ = 64 * Real.sqrt 21) := by
  sorry

end rhombus_area_l68_68826


namespace incorrect_conversion_l68_68604

/--
Incorrect conversion of -150° to radians.
-/
theorem incorrect_conversion : (¬(((-150 : ℝ) * (Real.pi / 180)) = (-7 * Real.pi / 6))) :=
by
  sorry

end incorrect_conversion_l68_68604


namespace gcd_is_18_l68_68769

-- Define gcdX that represents the greatest common divisor of X and Y.
noncomputable def gcdX (X Y : ℕ) : ℕ := Nat.gcd X Y

-- Given conditions
def cond_lcm (X Y : ℕ) : Prop := Nat.lcm X Y = 180
def cond_ratio (X Y : ℕ) : Prop := ∃ k : ℕ, X = 2 * k ∧ Y = 5 * k

-- Theorem to prove that the gcd of X and Y is 18
theorem gcd_is_18 {X Y : ℕ} (h1 : cond_lcm X Y) (h2 : cond_ratio X Y) : gcdX X Y = 18 :=
by
  sorry

end gcd_is_18_l68_68769


namespace price_restoration_l68_68563

theorem price_restoration {P : ℝ} (hP : P > 0) :
  (P - 0.85 * P) / (0.85 * P) * 100 = 17.65 :=
by
  sorry

end price_restoration_l68_68563


namespace positive_integer_divisibility_by_3_l68_68584

theorem positive_integer_divisibility_by_3 (n : ℕ) (h : 0 < n) :
  (n * 2^n + 1) % 3 = 0 ↔ n % 6 = 1 ∨ n % 6 = 2 := 
sorry

end positive_integer_divisibility_by_3_l68_68584


namespace stratified_sampling_vision_test_l68_68857

theorem stratified_sampling_vision_test 
  (n_total : ℕ) (n_HS : ℕ) (n_selected : ℕ)
  (h1 : n_total = 165)
  (h2 : n_HS = 66)
  (h3 : n_selected = 15) :
  (n_HS * n_selected / n_total) = 6 := 
by 
  sorry

end stratified_sampling_vision_test_l68_68857


namespace fraction_of_beans_remaining_l68_68454

variables (J B R : ℝ)

-- Given conditions
def condition1 : Prop := J = 0.10 * (J + B)
def condition2 : Prop := J + R = 0.60 * (J + B)

theorem fraction_of_beans_remaining (h1 : condition1 J B) (h2 : condition2 J B R) :
  R / B = 5 / 9 :=
  sorry

end fraction_of_beans_remaining_l68_68454


namespace saltwater_solution_l68_68806

theorem saltwater_solution (x : ℝ) (h1 : ∃ v : ℝ, v = x ∧ v * 0.2 = 0.20 * x)
(h2 : 3 / 4 * x = 3 / 4 * x)
(h3 : ∃ v' : ℝ, v' = 3 / 4 * x + 6 + 12)
(h4 : (0.20 * x + 12) / (3 / 4 * x + 18) = 1 / 3) : x = 120 :=
by 
  sorry

end saltwater_solution_l68_68806


namespace find_z_l68_68487

theorem find_z 
  (m : ℕ)
  (h1 : (1^(m+1) / 5^(m+1)) * (1^18 / z^18) = 1 / (2 * 10^35))
  (hm : m = 34) :
  z = 4 := 
sorry

end find_z_l68_68487


namespace students_multiple_activities_l68_68689

theorem students_multiple_activities (total_students only_debate only_singing only_dance no_activities students_more_than_one : ℕ) 
  (h1 : total_students = 55) 
  (h2 : only_debate = 10) 
  (h3 : only_singing = 18) 
  (h4 : only_dance = 8)
  (h5 : no_activities = 5)
  (h6 : students_more_than_one = total_students - (only_debate + only_singing + only_dance + no_activities)) :
  students_more_than_one = 14 := by
  sorry

end students_multiple_activities_l68_68689


namespace evaluate_expression_l68_68583

theorem evaluate_expression :
  8^6 * 27^6 * 8^15 * 27^15 = 216^21 :=
by
  sorry

end evaluate_expression_l68_68583


namespace transport_cost_l68_68966

-- Define the conditions
def cost_per_kg : ℕ := 15000
def grams_per_kg : ℕ := 1000
def weight_in_grams : ℕ := 500

-- Define the main theorem stating the proof problem
theorem transport_cost
  (c : ℕ := cost_per_kg)
  (gpk : ℕ := grams_per_kg)
  (w : ℕ := weight_in_grams)
  : c * w / gpk = 7500 :=
by
  -- Since we are not required to provide the proof, adding sorry here
  sorry

end transport_cost_l68_68966


namespace problem_statement_l68_68992

variable (x : ℝ) (x₀ : ℝ)

def p : Prop := ∀ x > 0, x + 4 / x ≥ 4

def q : Prop := ∃ x₀ ∈ Set.Ioi (0 : ℝ), 2 * x₀ = 1 / 2

theorem problem_statement : p ∧ ¬q :=
by
  sorry

end problem_statement_l68_68992


namespace problem1_problem2_l68_68668

def a (n : ℕ) : ℕ :=
  if n = 0 then 0  -- We add this case for Lean to handle zero index
  else if n = 1 then 2
  else 2^(n-1)

def S (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) a

theorem problem1 (n : ℕ) :
  a n = 
  if n = 1 then 2
  else 2^(n-1) :=
sorry

theorem problem2 (n : ℕ) :
  S n = 2^n :=
sorry

end problem1_problem2_l68_68668


namespace twice_son_plus_father_is_70_l68_68378

section
variable {s f : ℕ}

-- Conditions
def son_age : ℕ := 15
def father_age : ℕ := 40

-- Statement to prove
theorem twice_son_plus_father_is_70 : (2 * son_age + father_age) = 70 :=
by
  sorry
end

end twice_son_plus_father_is_70_l68_68378


namespace optimal_order_l68_68117

-- Definition of probabilities
variables (p1 p2 p3 : ℝ) (hp1 : p3 < p1) (hp2 : p1 < p2)

-- The statement to prove
theorem optimal_order (h : p2 > p1) :
  (p2 * (p1 + p3 - p1 * p3)) > (p1 * (p2 + p3 - p2 * p3)) :=
sorry

end optimal_order_l68_68117


namespace tournament_chromatic_index_l68_68550

noncomputable def chromaticIndex {n : ℕ} (k : ℕ) (h₁ : 2^(k-1) < n) (h₂ : n ≤ 2^k) : ℕ :=
k

theorem tournament_chromatic_index (n k : ℕ) (h₁ : 2^(k-1) < n) (h₂ : n ≤ 2^k) :
  chromaticIndex k h₁ h₂ = k :=
by sorry

end tournament_chromatic_index_l68_68550


namespace problem_1_problem_2_problem_3_problem_4_l68_68731

theorem problem_1 : (1 * -2.48) + 4.33 + (-7.52) + (-4.33) = -10 := by
  sorry

theorem problem_2 : 2 * (23 / 6 : ℚ) + - (36 / 7 : ℚ) + - (13 / 6 : ℚ) + - (230 / 7 : ℚ) = -(36 + 1 / 3 : ℚ) := by
  sorry

theorem problem_3 : (4 / 5 : ℚ) - (5 / 6 : ℚ) - (3 / 5 : ℚ) + (1 / 6 : ℚ) = - (7 / 15 : ℚ) := by
  sorry

theorem problem_4 : (-1 ^ 4 : ℚ) - (1 / 6) * (2 - (-3) ^ 2) = 1 / 6 := by
  sorry

end problem_1_problem_2_problem_3_problem_4_l68_68731


namespace polygon_interior_angle_l68_68711

theorem polygon_interior_angle (n : ℕ) (h : n ≥ 3) 
  (interior_angle : ∀ i, 1 ≤ i ∧ i ≤ n → interior_angle = 120) :
  n = 6 := by sorry

end polygon_interior_angle_l68_68711


namespace pieces_length_l68_68233

theorem pieces_length :
  let total_length_meters := 29.75
  let number_of_pieces := 35
  let length_per_piece_meters := total_length_meters / number_of_pieces
  let length_per_piece_centimeters := length_per_piece_meters * 100
  length_per_piece_centimeters = 85 :=
by
  sorry

end pieces_length_l68_68233


namespace gain_percent_is_25_l68_68410

theorem gain_percent_is_25 (C S : ℝ) (h : 50 * C = 40 * S) : (S - C) / C * 100 = 25 :=
  sorry

end gain_percent_is_25_l68_68410


namespace Janet_earnings_l68_68633

/--
Janet works as an exterminator and also sells molten metal casts of fire ant nests on the Internet.
She gets paid $70 an hour for exterminator work and makes $20/pound on her ant nest sculptures.
Given that she does 20 hours of exterminator work and sells a 5-pound sculpture and a 7-pound sculpture,
prove that Janet's total earnings are $1640.
-/
theorem Janet_earnings :
  let hourly_rate_exterminator := 70
  let hours_worked := 20
  let rate_per_pound := 20
  let sculpture_one_weight := 5
  let sculpture_two_weight := 7

  let exterminator_earnings := hourly_rate_exterminator * hours_worked
  let total_sculpture_weight := sculpture_one_weight + sculpture_two_weight
  let sculpture_earnings := rate_per_pound * total_sculpture_weight

  let total_earnings := exterminator_earnings + sculpture_earnings
  total_earnings = 1640 := 
by
  sorry

end Janet_earnings_l68_68633


namespace find_second_x_intercept_l68_68193

theorem find_second_x_intercept (a b c : ℝ)
  (h_vertex : ∀ x, y = a * x^2 + b * x + c → x = 5 → y = -3)
  (h_intercept1 : ∀ y, y = a * 1^2 + b * 1 + c → y = 0) :
  ∃ x, y = a * x^2 + b * x + c ∧ y = 0 ∧ x = 9 :=
sorry

end find_second_x_intercept_l68_68193


namespace greatest_integer_gcd_four_l68_68964

theorem greatest_integer_gcd_four {n : ℕ} (h1 : n < 150) (h2 : Nat.gcd n 12 = 4) : n <= 148 :=
by {
  sorry
}

end greatest_integer_gcd_four_l68_68964


namespace radical_conjugate_sum_l68_68433

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l68_68433


namespace polygon_sides_l68_68038

-- Definition of the conditions used in the problem
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Statement of the theorem
theorem polygon_sides (n : ℕ) (h : sum_of_interior_angles n = 1080) : n = 8 :=
by
  sorry  -- Proof placeholder

end polygon_sides_l68_68038


namespace race_time_comparison_l68_68549

noncomputable def townSquare : ℝ := 3 / 4 -- distance of one lap in miles
noncomputable def laps : ℕ := 7 -- number of laps
noncomputable def totalDistance : ℝ := laps * townSquare -- total distance of the race in miles
noncomputable def thisYearTime : ℝ := 42 -- time taken by this year's winner in minutes
noncomputable def lastYearTime : ℝ := 47.25 -- time taken by last year's winner in minutes

noncomputable def thisYearPace : ℝ := thisYearTime / totalDistance -- pace of this year's winner in minutes per mile
noncomputable def lastYearPace : ℝ := lastYearTime / totalDistance -- pace of last year's winner in minutes per mile
noncomputable def timeDifference : ℝ := lastYearPace - thisYearPace -- the difference in pace

theorem race_time_comparison : timeDifference = 1 := by
  sorry

end race_time_comparison_l68_68549


namespace wrongly_entered_mark_l68_68553

theorem wrongly_entered_mark (x : ℝ) : 
  (∀ marks_instead_of_45 number_of_pupils (total_avg_increase : ℝ),
     marks_instead_of_45 = 45 ∧
     number_of_pupils = 44 ∧
     total_avg_increase = 0.5 →
     x = marks_instead_of_45 + total_avg_increase * number_of_pupils) →
  x = 67 :=
by
  intro h
  sorry

end wrongly_entered_mark_l68_68553


namespace onur_biking_distance_l68_68467

-- Definitions based only on given conditions
def Onur_biking_distance_per_day (O : ℕ) := O
def Hanil_biking_distance_per_day (O : ℕ) := O + 40
def biking_days_per_week := 5
def total_distance_per_week := 2700

-- Mathematically equivalent proof problem
theorem onur_biking_distance (O : ℕ) (cond : 5 * (O + (O + 40)) = 2700) : O = 250 := by
  sorry

end onur_biking_distance_l68_68467


namespace max_m_ratio_l68_68231

theorem max_m_ratio (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : ∀ a b, (4 / a + 1 / b) ≥ m / (a + 4 * b)) :
  (m = 16) → (b / a = 1 / 4) :=
by sorry

end max_m_ratio_l68_68231


namespace percentage_decrease_l68_68888

theorem percentage_decrease (a b p : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
    (h_ratio : a / b = 4 / 5) 
    (h_x : ∃ x, x = a * 1.25)
    (h_m : ∃ m, m = b * (1 - p / 100))
    (h_mx : ∃ m x, (m / x = 0.2)) :
        (p = 80) :=
by
  sorry

end percentage_decrease_l68_68888


namespace percent_notebooks_staplers_clips_l68_68621

def percent_not_special (n s c: ℝ) (h_n: n = 25) (h_s: s = 20) (h_c: c = 30) : ℝ :=
  100 - (n + s + c)

theorem percent_notebooks_staplers_clips (n s c: ℝ) (h_n: n = 25) (h_s: s = 20) (h_c: c = 30) :
  percent_not_special n s c h_n h_s h_c = 25 :=
by
  unfold percent_not_special
  rw [h_n, h_s, h_c]
  norm_num

end percent_notebooks_staplers_clips_l68_68621


namespace A_inter_B_eq_l68_68122

-- Define set A based on the condition for different integer k.
def A (k : ℤ) : Set ℝ := {x | 2 * k * Real.pi - Real.pi < x ∧ x < 2 * k * Real.pi}

-- Define set B based on its condition.
def B : Set ℝ := {x | -5 ≤ x ∧ x < 4}

-- The final proof problem to show A ∩ B equals to the given set.
theorem A_inter_B_eq : 
  (⋃ k : ℤ, A k) ∩ B = {x | (-Real.pi < x ∧ x < 0) ∨ (Real.pi < x ∧ x < 4)} :=
by
  sorry

end A_inter_B_eq_l68_68122


namespace giraffes_count_l68_68137

def numZebras : ℕ := 12

def numCamels : ℕ := numZebras / 2

def numMonkeys : ℕ := numCamels * 4

def numGiraffes : ℕ := numMonkeys - 22

theorem giraffes_count :
  numGiraffes = 2 :=
by 
  sorry

end giraffes_count_l68_68137


namespace rebecca_bought_2_more_bottles_of_water_l68_68073

noncomputable def number_of_more_bottles_of_water_than_tent_stakes
  (T D W : ℕ) 
  (hT : T = 4) 
  (hD : D = 3 * T) 
  (hTotal : T + D + W = 22) : Prop :=
  W - T = 2

theorem rebecca_bought_2_more_bottles_of_water
  (T D W : ℕ) 
  (hT : T = 4) 
  (hD : D = 3 * T) 
  (hTotal : T + D + W = 22) : 
  number_of_more_bottles_of_water_than_tent_stakes T D W hT hD hTotal :=
by 
  sorry

end rebecca_bought_2_more_bottles_of_water_l68_68073


namespace min_value_of_quadratic_expression_l68_68045

theorem min_value_of_quadratic_expression (a b c : ℝ) (h : a + 2 * b + 3 * c = 6) : a^2 + 4 * b^2 + 9 * c^2 ≥ 12 :=
by
  sorry

end min_value_of_quadratic_expression_l68_68045


namespace inequality_condition_l68_68643

theorem inequality_condition 
  (a b c : ℝ) : 
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (c > Real.sqrt (a^2 + b^2)) := 
sorry

end inequality_condition_l68_68643


namespace tenth_square_tiles_more_than_ninth_l68_68825

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ := 2 * n - 1

-- Calculate the number of tiles used in the nth square
def tiles_count (n : ℕ) : ℕ := (side_length n) ^ 2

-- State the theorem that the tenth square requires 72 more tiles than the ninth square
theorem tenth_square_tiles_more_than_ninth : tiles_count 10 - tiles_count 9 = 72 :=
by
  sorry

end tenth_square_tiles_more_than_ninth_l68_68825


namespace solve_for_x_l68_68666

theorem solve_for_x:
  ∃ x : ℚ, (2 / 3 - 1 / 4 = 1 / x) ∧ (x = 12 / 5) := by
  sorry

end solve_for_x_l68_68666


namespace valentines_given_l68_68697

theorem valentines_given (x y : ℕ) (h : x * y = x + y + 40) : x * y = 84 :=
by
  -- solving for x, y based on the factors of 41
  sorry

end valentines_given_l68_68697


namespace cost_for_sugar_substitutes_l68_68570

def packets_per_cup : ℕ := 1
def cups_per_day : ℕ := 2
def days : ℕ := 90
def packets_per_box : ℕ := 30
def price_per_box : ℕ := 4

theorem cost_for_sugar_substitutes : 
  (packets_per_cup * cups_per_day * days / packets_per_box) * price_per_box = 24 := by
  sorry

end cost_for_sugar_substitutes_l68_68570


namespace inequality_abc_l68_68328

theorem inequality_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1) (h3 : 0 ≤ b) (h4 : b ≤ 1) (h5 : 0 ≤ c) (h6 : c ≤ 1) :
  (a / (b * c + 1)) + (b / (a * c + 1)) + (c / (a * b + 1)) ≤ 2 := by
  sorry

end inequality_abc_l68_68328


namespace find_f_log_l68_68344

noncomputable def f : ℝ → ℝ := sorry

-- Given Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x
axiom f_def : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2^x - 2

-- Theorem to be proved
theorem find_f_log : f (Real.log 6 / Real.log (1/2)) = 1 / 2 :=
by
  sorry

end find_f_log_l68_68344


namespace factorize_x4_y4_l68_68906

theorem factorize_x4_y4 (x y : ℝ) : x^4 - y^4 = (x^2 + y^2) * (x^2 - y^2) :=
by
  sorry

end factorize_x4_y4_l68_68906


namespace solve_xy_l68_68916

variable (x y : ℝ)

-- Given conditions
def condition1 : Prop := y = (2 / 3) * x
def condition2 : Prop := 0.4 * x = (1 / 3) * y + 110

-- Statement we want to prove
theorem solve_xy (h1 : condition1 x y) (h2 : condition2 x y) : x = 618.75 ∧ y = 412.5 :=
  by sorry

end solve_xy_l68_68916


namespace trivia_game_points_l68_68937

theorem trivia_game_points (first_round_points second_round_points points_lost last_round_points : ℤ) 
    (h1 : first_round_points = 16)
    (h2 : second_round_points = 33)
    (h3 : points_lost = 48) : 
    first_round_points + second_round_points - points_lost = 1 :=
by
    rw [h1, h2, h3]
    rfl

end trivia_game_points_l68_68937


namespace stamps_cost_l68_68466

theorem stamps_cost (cost_one: ℝ) (cost_three: ℝ) (h: cost_one = 0.34) (h1: cost_three = 3 * cost_one) : 
  2 * cost_one = 0.68 := 
by
  sorry

end stamps_cost_l68_68466


namespace triangle_altitude_l68_68008

theorem triangle_altitude (base side : ℝ) (h : ℝ) : 
  side = 6 → base = 6 → 
  (base * h) / 2 = side ^ 2 → 
  h = 12 :=
by
  intros
  sorry

end triangle_altitude_l68_68008


namespace prime_p_satisfies_condition_l68_68758

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_p_satisfies_condition {p : ℕ} (hp : is_prime p) (hp2_8 : is_prime (p^2 + 8)) : p = 3 :=
sorry

end prime_p_satisfies_condition_l68_68758


namespace find_k_common_term_l68_68504

def sequence_a (k : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then 1 
  else if n = 2 then k 
  else if n = 3 then 3*k - 3 
  else if n = 4 then 6*k - 8 
  else (n * (n-1) * (k-2)) / 2 + n

def is_fermat (x : ℕ) : Prop :=
  ∃ m : ℕ, x = 2^(2^m) + 1

theorem find_k_common_term (k : ℕ) :
  k > 2 → ∃ n m : ℕ, sequence_a k n = 2^(2^m) + 1 :=
by
  sorry

end find_k_common_term_l68_68504


namespace final_price_of_purchases_l68_68882

theorem final_price_of_purchases :
  let electronic_discount := 0.20
  let clothing_discount := 0.15
  let bundle_discount := 10
  let voucher_threshold := 200
  let voucher_value := 20
  let voucher_limit := 2
  let delivery_charge := 15
  let tax_rate := 0.08

  let electronic_original_price := 150
  let clothing_original_price := 80
  let num_clothing := 2

  -- Calculate discounts
  let electronic_discount_amount := electronic_original_price * electronic_discount
  let electronic_discount_price := electronic_original_price - electronic_discount_amount
  let clothing_discount_amount := clothing_original_price * clothing_discount
  let clothing_discount_price := clothing_original_price - clothing_discount_amount

  -- Sum of discounted clothing items
  let total_clothing_discount_price := clothing_discount_price * num_clothing

  -- Calculate bundle discount
  let total_before_bundle_discount := electronic_discount_price + total_clothing_discount_price
  let total_after_bundle_discount := total_before_bundle_discount - bundle_discount

  -- Calculate vouchers
  let num_vouchers := if total_after_bundle_discount >= voucher_threshold * 2 then voucher_limit else 
                      if total_after_bundle_discount >= voucher_threshold then 1 else 0
  let total_voucher_amount := num_vouchers * voucher_value
  let total_after_voucher_discount := total_after_bundle_discount - total_voucher_amount

  -- Add delivery charge
  let total_before_tax := total_after_voucher_discount + delivery_charge

  -- Calculate tax
  let tax_amount := total_before_tax * tax_rate
  let final_price := total_before_tax + tax_amount

  final_price = 260.28 :=
by
  -- the actual proof will be included here
  sorry

end final_price_of_purchases_l68_68882


namespace find_a1_l68_68979

variable (a : ℕ → ℕ)
variable (q : ℕ)
variable (h_q_pos : 0 < q)
variable (h_a2a6 : a 2 * a 6 = 8 * a 4)
variable (h_a2 : a 2 = 2)

theorem find_a1 :
  a 1 = 1 :=
by
  sorry

end find_a1_l68_68979


namespace tylenol_pill_mg_l68_68754

noncomputable def tylenol_dose_per_pill : ℕ :=
  let mg_per_dose := 1000
  let hours_per_dose := 6
  let days := 14
  let pills := 112
  let doses_per_day := 24 / hours_per_dose
  let total_doses := doses_per_day * days
  let total_mg := total_doses * mg_per_dose
  total_mg / pills

theorem tylenol_pill_mg :
  tylenol_dose_per_pill = 500 := by
  sorry

end tylenol_pill_mg_l68_68754


namespace calc_value_l68_68967

theorem calc_value (a b x : ℤ) (h₁ : a = 153) (h₂ : b = 147) (h₃ : x = 900) : x^2 / (a^2 - b^2) = 450 :=
by
  rw [h₁, h₂, h₃]
  -- Proof follows from the calculation in the provided steps
  sorry

end calc_value_l68_68967


namespace compute_avg_interest_rate_l68_68405

variable (x : ℝ)

/-- The total amount of investment is $5000 - x at 3% and x at 7%. The incomes are equal 
thus we are asked to compute the average rate of interest -/
def avg_interest_rate : Prop :=
  let i_3 := 0.03 * (5000 - x)
  let i_7 := 0.07 * x
  i_3 = i_7 ∧
  (2 * i_3) / 5000 = 0.042

theorem compute_avg_interest_rate 
  (condition : ∃ x : ℝ, 0.03 * (5000 - x) = 0.07 * x) :
  avg_interest_rate x :=
by
  sorry

end compute_avg_interest_rate_l68_68405


namespace strawberries_final_count_l68_68388

def initial_strawberries := 300
def buckets := 5
def strawberries_per_bucket := initial_strawberries / buckets
def strawberries_removed_per_bucket := 20
def redistributed_in_first_two := 15
def redistributed_in_third := 25

-- Defining the final counts after redistribution
def final_strawberries_first := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_first_two
def final_strawberries_second := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_first_two
def final_strawberries_third := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_third
def final_strawberries_fourth := strawberries_per_bucket - strawberries_removed_per_bucket
def final_strawberries_fifth := strawberries_per_bucket - strawberries_removed_per_bucket

theorem strawberries_final_count :
  final_strawberries_first = 55 ∧
  final_strawberries_second = 55 ∧
  final_strawberries_third = 65 ∧
  final_strawberries_fourth = 40 ∧
  final_strawberries_fifth = 40 := by
  sorry

end strawberries_final_count_l68_68388


namespace rational_iff_arithmetic_progression_l68_68776

theorem rational_iff_arithmetic_progression (x : ℝ) : 
  (∃ (i j k : ℤ), i < j ∧ j < k ∧ (x + i) + (x + k) = 2 * (x + j)) ↔ 
  (∃ n d : ℤ, d ≠ 0 ∧ x = n / d) := 
sorry

end rational_iff_arithmetic_progression_l68_68776


namespace base8_to_base10_l68_68350

theorem base8_to_base10 :
  ∀ (n : ℕ), (n = 2 * 8^2 + 4 * 8^1 + 3 * 8^0) → (n = 163) :=
by
  intros n hn
  sorry

end base8_to_base10_l68_68350


namespace profit_percentage_l68_68687

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 150) (hSP : SP = 216.67) :
  SP = 0.9 * LP ∧ LP = SP / 0.9 ∧ Profit = SP - CP ∧ Profit_Percentage = (Profit / CP) * 100 ∧ Profit_Percentage = 44.44 :=
by
  sorry

end profit_percentage_l68_68687


namespace negation_of_exists_leq_zero_l68_68942

theorem negation_of_exists_leq_zero (x : ℝ) : (¬ ∃ x : ℝ, x^2 + x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + x + 1 > 0) :=
by
  sorry

end negation_of_exists_leq_zero_l68_68942


namespace add_in_base8_l68_68989

def base8_add (a b : ℕ) (n : ℕ): ℕ :=
  a * (8 ^ n) + b

theorem add_in_base8 : base8_add 123 56 0 = 202 := by
  sorry

end add_in_base8_l68_68989


namespace min_x_plus_y_l68_68765

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y ≥ 9 :=
sorry

end min_x_plus_y_l68_68765


namespace correct_number_l68_68635

theorem correct_number : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  -- proof starts here
  sorry

end correct_number_l68_68635


namespace hogwarts_school_students_l68_68379

def total_students_at_school (participants boys : ℕ) (boy_participants girl_non_participants : ℕ) : Prop :=
  participants = 246 ∧ boys = 255 ∧ boy_participants = girl_non_participants + 11 → (boys + (participants - boy_participants + girl_non_participants)) = 490

theorem hogwarts_school_students : total_students_at_school 246 255 (boy_participants) girl_non_participants := 
 sorry

end hogwarts_school_students_l68_68379


namespace seq_ratio_l68_68656

theorem seq_ratio (a : ℕ → ℝ) (h₁ : a 1 = 5) (h₂ : ∀ n, a n * a (n + 1) = 2^n) : 
  a 7 / a 3 = 4 := 
by 
  sorry

end seq_ratio_l68_68656


namespace time_to_fill_cistern_proof_l68_68685

-- Define the filling rate F and emptying rate E
def filling_rate : ℚ := 1 / 3 -- cisterns per hour
def emptying_rate : ℚ := 1 / 6 -- cisterns per hour

-- Define the net rate as the difference between filling and emptying rates
def net_rate : ℚ := filling_rate - emptying_rate

-- Define the time to fill the cistern given the net rate
def time_to_fill_cistern (net_rate : ℚ) : ℚ := 1 / net_rate

-- The proof statement
theorem time_to_fill_cistern_proof : time_to_fill_cistern net_rate = 6 := 
by sorry

end time_to_fill_cistern_proof_l68_68685


namespace complex_z_solution_l68_68644

theorem complex_z_solution (z : ℂ) (i : ℂ) (h : i * z = 1 - i) (hi : i * i = -1) : z = -1 - i :=
by sorry

end complex_z_solution_l68_68644


namespace min_squared_sum_l68_68900

theorem min_squared_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) : 
  x^2 + y^2 + z^2 ≥ 9 := 
sorry

end min_squared_sum_l68_68900


namespace mary_mileage_l68_68678

def base9_to_base10 : Nat :=
  let d0 := 6 * 9^0
  let d1 := 5 * 9^1
  let d2 := 9 * 9^2
  let d3 := 3 * 9^3
  d0 + d1 + d2 + d3 

theorem mary_mileage :
  base9_to_base10 = 2967 :=
by 
  -- Calculation steps are skipped using sorry
  sorry

end mary_mileage_l68_68678


namespace defective_items_count_l68_68734

variables 
  (total_items : ℕ)
  (total_video_games : ℕ)
  (total_DVDs : ℕ)
  (total_books : ℕ)
  (working_video_games : ℕ)
  (working_DVDs : ℕ)

theorem defective_items_count
  (h1 : total_items = 56)
  (h2 : total_video_games = 30)
  (h3 : total_DVDs = 15)
  (h4 : total_books = total_items - total_video_games - total_DVDs)
  (h5 : working_video_games = 20)
  (h6 : working_DVDs = 10)
  : (total_video_games - working_video_games) + (total_DVDs - working_DVDs) = 15 :=
sorry

end defective_items_count_l68_68734


namespace ricky_roses_l68_68945

theorem ricky_roses (initial_roses : ℕ) (stolen_roses : ℕ) (people : ℕ) (remaining_roses : ℕ)
  (h1 : initial_roses = 40)
  (h2 : stolen_roses = 4)
  (h3 : people = 9)
  (h4 : remaining_roses = initial_roses - stolen_roses) :
  remaining_roses / people = 4 :=
by sorry

end ricky_roses_l68_68945


namespace cost_of_plane_ticket_l68_68181

theorem cost_of_plane_ticket 
  (total_cost : ℤ) (hotel_cost_per_day_per_person : ℤ) (num_people : ℤ) (num_days : ℤ) (plane_ticket_cost_per_person : ℤ) :
  total_cost = 120 →
  hotel_cost_per_day_per_person = 12 →
  num_people = 2 →
  num_days = 3 →
  (total_cost - num_people * hotel_cost_per_day_per_person * num_days) = num_people * plane_ticket_cost_per_person →
  plane_ticket_cost_per_person = 24 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof steps would go here
  sorry

end cost_of_plane_ticket_l68_68181


namespace possible_b_value_l68_68099

theorem possible_b_value (a b : ℤ) (h1 : a = 3^20) (h2 : a ≡ b [ZMOD 10]) : b = 2011 :=
by sorry

end possible_b_value_l68_68099


namespace find_circle_equation_l68_68877

noncomputable def circle_equation (D E F : ℝ) : Prop :=
  ∀ (x y : ℝ), (x, y) = (-1, 3) ∨ (x, y) = (0, 0) ∨ (x, y) = (0, 2) →
  x^2 + y^2 + D * x + E * y + F = 0

theorem find_circle_equation :
  ∃ D E F : ℝ, circle_equation D E F ∧
               (∀ x y, x^2 + y^2 + D * x + E * y + F = x^2 + y^2 + 4 * x - 2 * y) :=
sorry

end find_circle_equation_l68_68877


namespace subtraction_base_8_correct_l68_68864

def sub_in_base_8 (a b : Nat) : Nat := sorry

theorem subtraction_base_8_correct : sub_in_base_8 (sub_in_base_8 0o123 0o51) 0o15 = 0o25 :=
sorry

end subtraction_base_8_correct_l68_68864


namespace part1_part2_l68_68978

theorem part1 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (hA : A + B + C = π) 
  (ha : a = 2) 
  (hcosC : Real.cos C = -1 / 4) 
  (hsinA_sinB : Real.sin A = 2 * Real.sin B) : b = 1 ∧ c = Real.sqrt 6 := 
  sorry

theorem part2
  (A B C : ℝ) 
  (a b c : ℝ) 
  (hA : A + B + C = π) 
  (ha : a = 2) 
  (hcosC : Real.cos C = -1 / 4)
  (hcosA_minus_pi_div_4 : Real.cos (A - π / 4) = 4 / 5) : c = 5 * Real.sqrt 30 / 2 := 
  sorry

end part1_part2_l68_68978


namespace carl_owes_15300_l68_68192

def total_property_damage : ℝ := 40000
def total_medical_bills : ℝ := 70000
def insurance_coverage_property_damage : ℝ := 0.80
def insurance_coverage_medical_bills : ℝ := 0.75
def carl_responsibility : ℝ := 0.60

def carl_personally_owes : ℝ :=
  let insurance_paid_property_damage := insurance_coverage_property_damage * total_property_damage
  let insurance_paid_medical_bills := insurance_coverage_medical_bills * total_medical_bills
  let remaining_property_damage := total_property_damage - insurance_paid_property_damage
  let remaining_medical_bills := total_medical_bills - insurance_paid_medical_bills
  let carl_share_property_damage := carl_responsibility * remaining_property_damage
  let carl_share_medical_bills := carl_responsibility * remaining_medical_bills
  carl_share_property_damage + carl_share_medical_bills

theorem carl_owes_15300 :
  carl_personally_owes = 15300 := by
  sorry

end carl_owes_15300_l68_68192


namespace find_sachin_age_l68_68080

-- Define Sachin's and Rahul's ages as variables
variables (S R : ℝ)

-- Define the conditions
def rahul_age := S + 9
def age_ratio := (S / R) = (7 / 9)

-- State the theorem for Sachin's age
theorem find_sachin_age (h1 : R = rahul_age S) (h2 : age_ratio S R) : S = 31.5 :=
by sorry

end find_sachin_age_l68_68080


namespace worth_of_presents_is_33536_36_l68_68282

noncomputable def total_worth_of_presents : ℝ :=
  let ring := 4000
  let car := 2000
  let bracelet := 2 * ring
  let gown := bracelet / 2
  let jewelry := 1.2 * ring
  let painting := 3000 * 1.2
  let honeymoon := 180000 / 110
  let watch := 5500
  ring + car + bracelet + gown + jewelry + painting + honeymoon + watch

theorem worth_of_presents_is_33536_36 : total_worth_of_presents = 33536.36 := by
  sorry

end worth_of_presents_is_33536_36_l68_68282


namespace grade12_students_selected_l68_68714

theorem grade12_students_selected 
    (N : ℕ) (n10 : ℕ) (n12 : ℕ) (k : ℕ) 
    (h1 : N = 1200)
    (h2 : n10 = 240)
    (h3 : 3 * N / (k + 5 + 3) = n12)
    (h4 : k * N / (k + 5 + 3) = n10) :
    n12 = 360 := 
by sorry

end grade12_students_selected_l68_68714


namespace range_of_a_l68_68393

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0

def neg_p : Prop := ∃ x : ℝ, a * x^2 + a * x + 1 < 0

theorem range_of_a (h : neg_p a) : a ∈ Set.Iio 0 ∪ Set.Ioi 4 :=
  sorry

end range_of_a_l68_68393


namespace problem_solution_l68_68286

theorem problem_solution (a b c : ℝ) (h : b^2 = a * c) :
  (a^2 * b^2 * c^2 / (a^3 + b^3 + c^3)) * (1 / a^3 + 1 / b^3 + 1 / c^3) = 1 :=
  by sorry

end problem_solution_l68_68286


namespace sum_of_smallest_and_second_smallest_l68_68158

-- Define the set of numbers
def numbers : Set ℕ := {10, 11, 12, 13}

-- Define the smallest and second smallest numbers
def smallest_number : ℕ := 10
def second_smallest_number : ℕ := 11

-- Prove the sum of the smallest and the second smallest numbers
theorem sum_of_smallest_and_second_smallest : smallest_number + second_smallest_number = 21 := by
  sorry

end sum_of_smallest_and_second_smallest_l68_68158


namespace value_f2_f5_l68_68081

variable {α : Type} [AddGroup α]

noncomputable def f : α → ℤ := sorry

axiom func_eq : ∀ x y, f (x + y) = f x + f y + 7 * x * y + 4

axiom f_one : f 1 = 4

theorem value_f2_f5 :
  f 2 + f 5 = 125 :=
sorry

end value_f2_f5_l68_68081


namespace necessary_and_sufficient_condition_l68_68095

open Classical

noncomputable def f (x a : ℝ) := x + a / x

theorem necessary_and_sufficient_condition
  (a : ℝ) :
  (∀ x : ℝ, x > 0 → f x a ≥ 2) ↔ (a ≥ 1) :=
by
  sorry

end necessary_and_sufficient_condition_l68_68095


namespace find_xy_sum_l68_68698

open Nat

theorem find_xy_sum (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x + y + x * y = 8) 
  (h2 : y + z + y * z = 15) 
  (h3 : z + x + z * x = 35) : 
  x + y + z + x * y = 15 := 
sorry

end find_xy_sum_l68_68698


namespace abs_a_gt_abs_b_l68_68792

variable (a b : Real)

theorem abs_a_gt_abs_b (h1 : a > 0) (h2 : b < 0) (h3 : a + b > 0) : |a| > |b| :=
by
  sorry

end abs_a_gt_abs_b_l68_68792


namespace time_for_a_alone_l68_68904

theorem time_for_a_alone
  (b_work_time : ℕ := 20)
  (c_work_time : ℕ := 45)
  (together_work_time : ℕ := 72 / 10) :
  ∃ (a_work_time : ℕ), a_work_time = 15 :=
by
  sorry

end time_for_a_alone_l68_68904


namespace quadratic_expression_always_positive_l68_68539

theorem quadratic_expression_always_positive (x y : ℝ) : 
  x^2 - 4 * x * y + 6 * y^2 - 4 * y + 3 > 0 :=
by 
  sorry

end quadratic_expression_always_positive_l68_68539


namespace calculation_result_l68_68320

theorem calculation_result : (4^2)^3 - 4 = 4092 :=
by
  sorry

end calculation_result_l68_68320


namespace james_total_oop_correct_l68_68649

-- Define the costs and insurance coverage percentages as given conditions.
def cost_consultation : ℝ := 300
def coverage_consultation : ℝ := 0.80

def cost_xray : ℝ := 150
def coverage_xray : ℝ := 0.70

def cost_prescription : ℝ := 75
def coverage_prescription : ℝ := 0.50

def cost_therapy : ℝ := 120
def coverage_therapy : ℝ := 0.60

-- Define the out-of-pocket calculation for each service
def oop_consultation := cost_consultation * (1 - coverage_consultation)
def oop_xray := cost_xray * (1 - coverage_xray)
def oop_prescription := cost_prescription * (1 - coverage_prescription)
def oop_therapy := cost_therapy * (1 - coverage_therapy)

-- Define the total out-of-pocket cost
def total_oop : ℝ := oop_consultation + oop_xray + oop_prescription + oop_therapy

-- Proof statement
theorem james_total_oop_correct : total_oop = 190.50 := by
  sorry

end james_total_oop_correct_l68_68649


namespace no_roots_less_than_x0_l68_68142

theorem no_roots_less_than_x0
  (x₀ a b c d : ℝ)
  (h₁ : ∀ x ≥ x₀, x^2 + a * x + b > 0)
  (h₂ : ∀ x ≥ x₀, x^2 + c * x + d > 0) :
  ∀ x ≥ x₀, x^2 + ((a + c) / 2) * x + ((b + d) / 2) > 0 := 
by
  sorry

end no_roots_less_than_x0_l68_68142


namespace horror_movie_more_than_triple_romance_l68_68052

-- Definitions and Conditions
def tickets_sold_romance : ℕ := 25
def tickets_sold_horror : ℕ := 93
def triple_tickets_romance := 3 * tickets_sold_romance

-- Theorem Statement
theorem horror_movie_more_than_triple_romance :
  (tickets_sold_horror - triple_tickets_romance) = 18 :=
by
  sorry

end horror_movie_more_than_triple_romance_l68_68052


namespace find_f_of_odd_function_periodic_l68_68567

noncomputable def arctan (x : ℝ) : ℝ := sorry

theorem find_f_of_odd_function_periodic (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_periodic : ∀ x k : ℤ, f x = f (x + 3 * k))
    (α : ℝ) (h_tan : Real.tan α = 3) :
  f (2015 * Real.sin (2 * (arctan 3))) = 0 :=
sorry

end find_f_of_odd_function_periodic_l68_68567


namespace find_number_l68_68265

theorem find_number : ∃ x : ℝ, 0.35 * x = 0.15 * 40 ∧ x = 120 / 7 :=
by
  sorry

end find_number_l68_68265


namespace second_reduction_percentage_is_4_l68_68726

def original_price := 500
def first_reduction_percent := 5 / 100
def total_reduction := 44

def first_reduction := first_reduction_percent * original_price
def price_after_first_reduction := original_price - first_reduction
def second_reduction := total_reduction - first_reduction
def second_reduction_percent := (second_reduction / price_after_first_reduction) * 100

theorem second_reduction_percentage_is_4 :
  second_reduction_percent = 4 := by
  sorry

end second_reduction_percentage_is_4_l68_68726


namespace remainder_1534_base12_div_by_9_l68_68444

noncomputable def base12_to_base10 (n : ℕ) : ℕ :=
  1 * 12^3 + 5 * 12^2 + 3 * 12 + 4

theorem remainder_1534_base12_div_by_9 :
  (base12_to_base10 1534) % 9 = 4 :=
by
  sorry

end remainder_1534_base12_div_by_9_l68_68444


namespace cubed_difference_l68_68113

theorem cubed_difference (x : ℝ) (h : x - 1/x = 3) : (x^3 - 1/x^3 = 36) := 
by
  sorry

end cubed_difference_l68_68113


namespace probability_boarding_251_l68_68651

theorem probability_boarding_251 :
  let interval_152 := 5
  let interval_251 := 7
  let total_events := interval_152 * interval_251
  let favorable_events := (interval_152 * interval_152) / 2
  (favorable_events / total_events : ℚ) = 5 / 14 :=
by 
  sorry

end probability_boarding_251_l68_68651


namespace A_share_in_profit_l68_68881

def investment_A := 6300
def investment_B := 4200
def investment_C := 10500
def total_profit := 12500

def total_investment := investment_A + investment_B + investment_C
def A_ratio := investment_A / total_investment

theorem A_share_in_profit : (total_profit * A_ratio) = 3750 := by
  sorry

end A_share_in_profit_l68_68881


namespace divides_two_pow_n_minus_one_l68_68384

theorem divides_two_pow_n_minus_one {n : ℕ} (h : n > 0) (divides : n ∣ 2^n - 1) : n = 1 :=
sorry

end divides_two_pow_n_minus_one_l68_68384


namespace train_cross_time_l68_68394

noncomputable def train_length : ℕ := 1200 -- length of the train in meters
noncomputable def platform_length : ℕ := train_length -- length of the platform equals the train length
noncomputable def speed_kmh : ℝ := 144 -- speed in km/hr
noncomputable def speed_ms : ℝ := speed_kmh * (1000 / 3600) -- converting speed to m/s

-- the formula to calculate the crossing time
noncomputable def time_to_cross_platform : ℝ := 
  2 * train_length / speed_ms

theorem train_cross_time : time_to_cross_platform = 60 := by
  sorry

end train_cross_time_l68_68394


namespace bowling_tournament_l68_68526

-- Definition of the problem conditions
def playoff (num_bowlers: Nat): Nat := 
  if num_bowlers < 5 then
    0
  else
    2^(num_bowlers - 1)

-- Theorem statement to prove
theorem bowling_tournament (num_bowlers: Nat) (h: num_bowlers = 5): playoff num_bowlers = 16 := by
  sorry

end bowling_tournament_l68_68526


namespace solve_eqs_l68_68082

theorem solve_eqs (x y : ℤ) (h1 : 7 - x = 15) (h2 : y - 3 = 4 + x) : x = -8 ∧ y = -1 := 
by
  sorry

end solve_eqs_l68_68082


namespace square_no_remainder_5_mod_9_l68_68050

theorem square_no_remainder_5_mod_9 (n : ℤ) : (n^2 % 9 ≠ 5) :=
by sorry

end square_no_remainder_5_mod_9_l68_68050


namespace fraction_product_sum_l68_68980

theorem fraction_product_sum :
  (1/3) * (5/6) * (3/7) + (1/4) * (1/8) = 101/672 :=
by
  sorry

end fraction_product_sum_l68_68980


namespace value_is_50_cents_l68_68105

-- Define Leah's total number of coins and the condition on the number of nickels and pennies.
variables (p n : ℕ)

-- Leah has a total of 18 coins
def total_coins : Prop := n + p = 18

-- Condition for nickels and pennies
def condition : Prop := p = n + 2

-- Calculate the total value of Leah's coins and check if it equals 50 cents
def total_value : ℕ := 5 * n + p

-- Proposition stating that under given conditions, total value is 50 cents
theorem value_is_50_cents (h1 : total_coins p n) (h2 : condition p n) :
  total_value p n = 50 := sorry

end value_is_50_cents_l68_68105


namespace value_of_x_yplusz_l68_68591

theorem value_of_x_yplusz (x y z : ℝ) (h : x * (x + y + z) = x^2 + 12) : x * (y + z) = 12 :=
by
  sorry

end value_of_x_yplusz_l68_68591


namespace puppies_per_cage_l68_68267

/-
Theorem: If a pet store had 56 puppies, sold 24 of them, and placed the remaining puppies into 8 cages, then each cage contains 4 puppies.
-/

theorem puppies_per_cage
  (initial_puppies : ℕ)
  (sold_puppies : ℕ)
  (cages : ℕ)
  (remaining_puppies : ℕ)
  (puppies_per_cage : ℕ) :
  initial_puppies = 56 →
  sold_puppies = 24 →
  cages = 8 →
  remaining_puppies = initial_puppies - sold_puppies →
  puppies_per_cage = remaining_puppies / cages →
  puppies_per_cage = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end puppies_per_cage_l68_68267


namespace arithmetic_sequence_sum_l68_68729

open Real

noncomputable def a_n : ℕ → ℝ := sorry -- to represent the arithmetic sequence

theorem arithmetic_sequence_sum :
  (∃ d : ℝ, ∀ (n : ℕ), a_n n = a_n 1 + (n - 1) * d) ∧
  (∃ a1 a2011 : ℝ, (a_n 1 = a1) ∧ (a_n 2011 = a2011) ∧ (a1 ^ 2 - 10 * a1 + 16 = 0) ∧ (a2011 ^ 2 - 10 * a2011 + 16 = 0)) →
  a_n 2 + a_n 1006 + a_n 2010 = 15 :=
by
  sorry

end arithmetic_sequence_sum_l68_68729


namespace determine_height_impossible_l68_68314

-- Definitions used in the conditions
def shadow_length_same (xiao_ming_height xiao_qiang_height xiao_ming_distance xiao_qiang_distance : ℝ) : Prop :=
  xiao_ming_height / xiao_ming_distance = xiao_qiang_height / xiao_qiang_distance

-- The proof problem: given that the shadow lengths are the same under the same street lamp,
-- prove that it is impossible to determine who is taller.
theorem determine_height_impossible (xiao_ming_height xiao_qiang_height xiao_ming_distance xiao_qiang_distance : ℝ) :
  shadow_length_same xiao_ming_height xiao_qiang_height xiao_ming_distance xiao_qiang_distance →
  ¬ (xiao_ming_height ≠ xiao_qiang_height ↔ true) :=
by
  intro h
  sorry -- Proof not required as per instructions

end determine_height_impossible_l68_68314


namespace ants_crushed_l68_68558

theorem ants_crushed {original_ants left_ants crushed_ants : ℕ} 
  (h1 : original_ants = 102) 
  (h2 : left_ants = 42) 
  (h3 : crushed_ants = original_ants - left_ants) : 
  crushed_ants = 60 := 
by
  sorry

end ants_crushed_l68_68558


namespace range_of_a_l68_68995

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = x^2 - 2 * a * x + a^2 - 1) 
(h_sol : ∀ x, f (f x) ≥ 0) : a ≤ -2 :=
sorry

end range_of_a_l68_68995


namespace remainder_3_pow_2000_mod_17_l68_68988

theorem remainder_3_pow_2000_mod_17 : (3^2000 % 17) = 1 := by
  sorry

end remainder_3_pow_2000_mod_17_l68_68988


namespace min_value_a_2b_3c_l68_68459

theorem min_value_a_2b_3c (a b c : ℝ)
  (h : ∀ x y : ℝ, x + 2 * y - 3 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ x + 2 * y + 3) :
  a + 2 * b - 3 * c ≥ -2 :=
sorry

end min_value_a_2b_3c_l68_68459


namespace find_y_l68_68363

def diamond (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_y : ∃ y : ℝ, diamond 4 y = 44 ∧ y = 48 / 7 :=
by
  sorry

end find_y_l68_68363


namespace mass_of_1m3_l68_68341

/-- The volume of 1 gram of the substance in cubic centimeters cms_per_gram is 1.3333333333333335 cm³. -/
def cms_per_gram : ℝ := 1.3333333333333335

/-- There are 1,000,000 cubic centimeters in 1 cubic meter. -/
def cm3_per_m3 : ℕ := 1000000

/-- Given the volume of 1 gram of the substance, find the mass of 1 cubic meter of the substance. -/
theorem mass_of_1m3 (h1 : cms_per_gram = 1.3333333333333335) (h2 : cm3_per_m3 = 1000000) :
  ∃ m : ℝ, m = 750 :=
by
  sorry

end mass_of_1m3_l68_68341


namespace solution_correct_l68_68793

-- Define the conditions
def condition1 (x : ℝ) : Prop := 2 ≤ |x - 3| ∧ |x - 3| ≤ 5
def condition2 (x : ℝ) : Prop := (x - 3) ^ 2 ≤ 16

-- Define the solution set
def solution_set (x : ℝ) : Prop := (-1 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 7)

-- Prove that the solution set is correct given the conditions
theorem solution_correct (x : ℝ) : condition1 x ∧ condition2 x ↔ solution_set x :=
by
  sorry

end solution_correct_l68_68793


namespace find_radius_of_circle_l68_68669

noncomputable def central_angle := 150
noncomputable def arc_length := 5 * Real.pi
noncomputable def arc_length_formula (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 180) * Real.pi * r

theorem find_radius_of_circle :
  (∃ r : ℝ, arc_length_formula central_angle r = arc_length) ↔ 6 = 6 :=
by  
  sorry

end find_radius_of_circle_l68_68669


namespace trapezium_other_side_length_l68_68626

theorem trapezium_other_side_length (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ) :
  side1 = 20 ∧ distance = 15 ∧ area = 285 → side2 = 18 :=
by
  intro h
  sorry

end trapezium_other_side_length_l68_68626


namespace div_by_64_l68_68766

theorem div_by_64 (n : ℕ) (h : n > 0) : 64 ∣ (5^n - 8*n^2 + 4*n - 1) :=
sorry

end div_by_64_l68_68766


namespace arithmetic_seq_sum_l68_68559

theorem arithmetic_seq_sum (a : ℕ → ℤ) (a1 a7 a3 a5 : ℤ) (S7 : ℤ)
  (h1 : a1 = a 1) (h7 : a7 = a 7) (h3 : a3 = a 3) (h5 : a5 = a 5)
  (h_arith : ∀ n m, a (n + m) = a n + a m - a 0)
  (h_S7 : (7 * (a1 + a7)) / 2 = 14) :
  a3 + a5 = 4 :=
sorry

end arithmetic_seq_sum_l68_68559


namespace area_of_rhombus_l68_68258

-- Defining the conditions
def diagonal1 : ℝ := 20
def diagonal2 : ℝ := 30

-- Proving the area of the rhombus
theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = diagonal1) (h2 : d2 = diagonal2) : 
  (d1 * d2 / 2) = 300 := by
  sorry

end area_of_rhombus_l68_68258


namespace journey_possibility_l68_68506

noncomputable def possible_start_cities 
  (routes : List (String × String)) 
  (visited : List String) : List String :=
sorry

theorem journey_possibility :
  possible_start_cities 
    [("Saint Petersburg", "Tver"), 
     ("Yaroslavl", "Nizhny Novgorod"), 
     ("Moscow", "Kazan"), 
     ("Nizhny Novgorod", "Kazan"), 
     ("Moscow", "Tver"), 
     ("Moscow", "Nizhny Novgorod")]
    ["Saint Petersburg", "Tver", "Yaroslavl", "Nizhny Novgorod", "Moscow", "Kazan"] 
  = ["Saint Petersburg", "Yaroslavl"] :=
sorry

end journey_possibility_l68_68506


namespace population_increase_time_l68_68743

theorem population_increase_time (persons_added : ℕ) (time_minutes : ℕ) (seconds_per_minute : ℕ) (total_seconds : ℕ) (time_for_one_person : ℕ) :
  persons_added = 160 →
  time_minutes = 40 →
  seconds_per_minute = 60 →
  total_seconds = time_minutes * seconds_per_minute →
  time_for_one_person = total_seconds / persons_added →
  time_for_one_person = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end population_increase_time_l68_68743


namespace triangle_is_isosceles_l68_68801

variable (a b c : ℝ)
variable (h : a^2 - b * c = a * (b - c))

theorem triangle_is_isosceles (a b c : ℝ) (h : a^2 - b * c = a * (b - c)) : a = b ∨ b = c ∨ c = a := by
  sorry

end triangle_is_isosceles_l68_68801


namespace simplify_fraction_l68_68910

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l68_68910


namespace minimum_workers_required_l68_68503

theorem minimum_workers_required (total_days : ℕ) (days_elapsed : ℕ) (initial_workers : ℕ) (job_fraction_done : ℚ)
  (remaining_work_fraction : job_fraction_done < 1) 
  (worker_productivity_constant : Prop) : 
  total_days = 40 → days_elapsed = 10 → initial_workers = 10 → job_fraction_done = (1/4) →
  (total_days - days_elapsed) * initial_workers * job_fraction_done = (1 - job_fraction_done) →
  job_fraction_done = 1 → initial_workers = 10 :=
by
  intros;
  sorry

end minimum_workers_required_l68_68503


namespace evaluate_expression_l68_68275

-- Definitions based on conditions
variables (b : ℤ) (x : ℤ)
def condition := x = 2 * b + 9

-- Statement of the problem
theorem evaluate_expression (b : ℤ) (x : ℤ) (h : condition b x) : x - 2 * b + 5 = 14 :=
by sorry

end evaluate_expression_l68_68275


namespace repeating_decimal_as_fraction_l68_68160

theorem repeating_decimal_as_fraction :
  (∃ y : ℚ, y = 737910 ∧ 0.73 + 864 / 999900 = y / 999900) :=
by
  -- proof omitted
  sorry

end repeating_decimal_as_fraction_l68_68160


namespace num_six_digit_asc_digits_l68_68311

theorem num_six_digit_asc_digits : 
  ∃ n : ℕ, n = (Nat.choose 9 3) ∧ n = 84 := 
by
  sorry

end num_six_digit_asc_digits_l68_68311


namespace min_value_a_plus_2b_l68_68391

theorem min_value_a_plus_2b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_condition : (a + b) / (a * b) = 1) : a + 2 * b ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_a_plus_2b_l68_68391


namespace albert_earnings_l68_68529

theorem albert_earnings (E E_final : ℝ) : 
  (0.90 * (E * 1.14) = 678) → 
  (E_final = 0.90 * (E * 1.15 * 1.20)) → 
  E_final = 819.72 :=
by
  sorry

end albert_earnings_l68_68529


namespace solve_for_x_l68_68003

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 = 6 - x) : x = 3 :=
by
  sorry

end solve_for_x_l68_68003


namespace find_c_l68_68139

variable {r s b c : ℚ}

-- Conditions based on roots of the original quadratic equation
def roots_of_original_quadratic (r s : ℚ) := 
  (5 * r ^ 2 - 8 * r + 2 = 0) ∧ (5 * s ^ 2 - 8 * s + 2 = 0)

-- New quadratic equation with roots shifted by 3
def new_quadratic_roots (r s b c : ℚ) :=
  (r - 3) + (s - 3) = -b ∧ (r - 3) * (s - 3) = c 

theorem find_c (r s : ℚ) (hb : b = 22/5) : 
  (roots_of_original_quadratic r s) → 
  (new_quadratic_roots r s b c) → 
  c = 23/5 := 
by
  intros h1 h2
  sorry

end find_c_l68_68139


namespace bob_expected_difference_l68_68524

-- Required definitions and conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

def probability_of_event_s : ℚ := 4 / 7
def probability_of_event_u : ℚ := 2 / 7
def probability_of_event_s_and_u : ℚ := 1 / 7
def number_of_days : ℕ := 365

noncomputable def expected_days_sweetened : ℚ :=
   (probability_of_event_s - (1 / 2) * probability_of_event_s_and_u) * number_of_days

noncomputable def expected_days_unsweetened : ℚ :=
   (probability_of_event_u - (1 / 2) * probability_of_event_s_and_u) * number_of_days

noncomputable def expected_difference : ℚ :=
   expected_days_sweetened - expected_days_unsweetened

theorem bob_expected_difference : expected_difference = 135.45 := sorry

end bob_expected_difference_l68_68524


namespace entertainment_expense_percentage_l68_68892

noncomputable def salary : ℝ := 10000
noncomputable def savings : ℝ := 2000
noncomputable def food_expense_percentage : ℝ := 0.40
noncomputable def house_rent_percentage : ℝ := 0.20
noncomputable def conveyance_percentage : ℝ := 0.10

theorem entertainment_expense_percentage :
  let E := (1 - (food_expense_percentage + house_rent_percentage + conveyance_percentage) - (savings / salary))
  E = 0.10 :=
by
  sorry

end entertainment_expense_percentage_l68_68892


namespace part1_part2_l68_68138

noncomputable def U : Set ℝ := Set.univ

noncomputable def A (a: ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
noncomputable def B (a: ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

theorem part1 (a : ℝ) (ha : a = 1/2) :
  (U \ (B a)) ∩ (A a) = {x | 9/4 ≤ x ∧ x < 5/2} :=
sorry

theorem part2 (p q : ℝ → Prop)
  (hp : ∀ x, p x → x ∈ A a) (hq : ∀ x, q x → x ∈ B a)
  (hq_necessary : ∀ x, p x → q x) :
  -1/2 ≤ a ∧ a ≤ (3 - Real.sqrt 5) / 2 :=
sorry

end part1_part2_l68_68138


namespace find_b2_a2_minus_a1_l68_68426

theorem find_b2_a2_minus_a1 
  (a₁ a₂ b₁ b₂ b₃ : ℝ)
  (d r : ℝ)
  (h_arith_seq : a₁ = -9 + d ∧ a₂ = a₁ + d)
  (h_geo_seq : b₁ = -9 * r ∧ b₂ = b₁ * r ∧ b₃ = b₂ * r ∧ (-9) * (-1) = b₁ * b₃)
  (h_d_val : a₂ - a₁ = d)
  (h_b2_val : b₂ = -1) : 
  b₂ * (a₂ - a₁) = -8 :=
sorry

end find_b2_a2_minus_a1_l68_68426


namespace paint_gallons_l68_68955

theorem paint_gallons (W B : ℕ) (h1 : 5 * B = 8 * W) (h2 : W + B = 6689) : B = 4116 :=
by
  sorry

end paint_gallons_l68_68955


namespace inequality_example_l68_68469

theorem inequality_example (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := 
  sorry

end inequality_example_l68_68469


namespace tenth_term_is_19_over_4_l68_68922

def nth_term_arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

theorem tenth_term_is_19_over_4 :
  nth_term_arithmetic_sequence (1/4) (1/2) 10 = 19/4 :=
by
  sorry

end tenth_term_is_19_over_4_l68_68922


namespace term_37_l68_68239

section GeometricSequence

variable {a b : ℕ → ℝ}
variable (q p : ℝ)

-- Definition of geometric sequences
def is_geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop := ∀ n ≥ 1, a (n + 1) = r * a n

-- Given conditions
axiom a1_25 : a 1 = 25
axiom b1_4 : b 1 = 4
axiom a2b2_100 : a 2 * b 2 = 100

-- Assume a and b are geometric sequences
axiom a_geom_seq : is_geometric_seq a q
axiom b_geom_seq : is_geometric_seq b p

-- Main theorem to prove
theorem term_37 (n : ℕ) (hn : n = 37) : (a n * b n) = 100 :=
sorry

end GeometricSequence

end term_37_l68_68239


namespace determine_ABC_l68_68243

noncomputable def digits_are_non_zero_distinct_and_not_larger_than_5 (A B C : ℕ) : Prop :=
  0 < A ∧ A ≤ 5 ∧ 0 < B ∧ B ≤ 5 ∧ 0 < C ∧ C ≤ 5 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

noncomputable def first_condition (A B : ℕ) : Prop :=
  A * 6 + B + A = B * 6 + A -- AB_6 + A_6 = BA_6 condition translated into arithmetics

noncomputable def second_condition (A B C : ℕ) : Prop :=
  A * 6 + B + B = C * 6 + 1 -- AB_6 + B_6 = C1_6 condition translated into arithmetics

theorem determine_ABC (A B C : ℕ) (h1 : digits_are_non_zero_distinct_and_not_larger_than_5 A B C)
    (h2 : first_condition A B) (h3 : second_condition A B C) :
    A * 100 + B * 10 + C = 5 * 100 + 1 * 10 + 5 := -- Final transformation of ABC to 515
  sorry

end determine_ABC_l68_68243


namespace num_fixed_last_two_digits_l68_68748

theorem num_fixed_last_two_digits : 
  ∃ c : ℕ, c = 36 ∧ ∀ (a : ℕ), 2 ≤ a ∧ a ≤ 101 → 
    (∃ N : ℕ, ∀ n : ℕ, n ≥ N → (a^(2^n) % 100 = a^(2^N) % 100)) ↔ (a = c ∨ c ≠ 36) :=
sorry

end num_fixed_last_two_digits_l68_68748


namespace f_2014_value_l68_68941

def f : ℝ → ℝ :=
sorry

lemma f_periodic (x : ℝ) : f (x + 2) = f (x - 2) :=
sorry

lemma f_on_interval (x : ℝ) (hx : 0 ≤ x ∧ x < 4) : f x = x^2 :=
sorry

theorem f_2014_value : f 2014 = 4 :=
by
  -- Insert proof here
  sorry

end f_2014_value_l68_68941


namespace geometric_sequence_S5_l68_68228

noncomputable def S5 (a₁ q : ℝ) : ℝ :=
  a₁ * (1 - q^5) / (1 - q)

theorem geometric_sequence_S5 
  (a₁ q : ℝ) 
  (h₁ : a₁ * (1 + q) = 3 / 4)
  (h₄ : a₁ * q^3 * (1 + q) = 6) :
  S5 a₁ q = 31 / 4 := 
sorry

end geometric_sequence_S5_l68_68228


namespace total_fruits_is_174_l68_68934

def basket1_apples : ℕ := 9
def basket1_oranges : ℕ := 15
def basket1_bananas : ℕ := 14
def basket1_grapes : ℕ := 12

def basket4_apples : ℕ := basket1_apples - 2
def basket4_oranges : ℕ := basket1_oranges - 2
def basket4_bananas : ℕ := basket1_bananas - 2
def basket4_grapes : ℕ := basket1_grapes - 2

def basket5_apples : ℕ := basket1_apples + 3
def basket5_oranges : ℕ := basket1_oranges - 5
def basket5_bananas : ℕ := basket1_bananas
def basket5_grapes : ℕ := basket1_grapes

def basket6_bananas : ℕ := basket1_bananas * 2
def basket6_grapes : ℕ := basket1_grapes / 2

def total_fruits_b1_3 : ℕ := basket1_apples + basket1_oranges + basket1_bananas + basket1_grapes
def total_fruits_b4 : ℕ := basket4_apples + basket4_oranges + basket4_bananas + basket4_grapes
def total_fruits_b5 : ℕ := basket5_apples + basket5_oranges + basket5_bananas + basket5_grapes
def total_fruits_b6 : ℕ := basket6_bananas + basket6_grapes

def total_fruits_all : ℕ := total_fruits_b1_3 + total_fruits_b4 + total_fruits_b5 + total_fruits_b6

theorem total_fruits_is_174 : total_fruits_all = 174 := by
  -- proof will go here
  sorry

end total_fruits_is_174_l68_68934


namespace circle_radius_l68_68834

theorem circle_radius (M N : ℝ) (hM : M = Real.pi * r ^ 2) (hN : N = 2 * Real.pi * r) (h : M / N = 15) : r = 30 := by
  sorry

end circle_radius_l68_68834


namespace members_didnt_show_up_l68_68753

theorem members_didnt_show_up (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) :
  total_members = 14 →
  points_per_member = 5 →
  total_points = 35 →
  total_members - (total_points / points_per_member) = 7 :=
by
  intros
  sorry

end members_didnt_show_up_l68_68753


namespace result_is_0_85_l68_68277

noncomputable def calc_expression := 1.85 - 1.85 / 1.85

theorem result_is_0_85 : calc_expression = 0.85 :=
by 
  sorry

end result_is_0_85_l68_68277


namespace toothpicks_required_l68_68236

noncomputable def total_small_triangles (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def total_initial_toothpicks (n : ℕ) : ℕ :=
  3 * total_small_triangles n

noncomputable def adjusted_toothpicks (n : ℕ) : ℕ :=
  total_initial_toothpicks n / 2

noncomputable def boundary_toothpicks (n : ℕ) : ℕ :=
  2 * n

noncomputable def total_toothpicks (n : ℕ) : ℕ :=
  adjusted_toothpicks n + boundary_toothpicks n

theorem toothpicks_required {n : ℕ} (h : n = 2500) : total_toothpicks n = 4694375 :=
by sorry

end toothpicks_required_l68_68236


namespace cheapest_third_company_l68_68606

theorem cheapest_third_company (x : ℕ) :
  (120 + 18 * x ≥ 150 + 15 * x) ∧ (220 + 13 * x ≥ 150 + 15 * x) → 36 ≤ x :=
by
  intro h
  cases h with
  | intro h1 h2 =>
    sorry

end cheapest_third_company_l68_68606


namespace maximum_value_of_f_over_interval_l68_68215

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 2) / (2 * x - 2)

theorem maximum_value_of_f_over_interval :
  ∀ x : ℝ, -4 < x ∧ x < 1 → ∃ M : ℝ, (∀ y : ℝ, -4 < y ∧ y < 1 → f y ≤ M) ∧ M = -1 :=
by
  sorry

end maximum_value_of_f_over_interval_l68_68215


namespace thirty_six_hundredths_is_decimal_l68_68917

namespace thirty_six_hundredths

-- Define the fraction representation of thirty-six hundredths
def fraction_thirty_six_hundredths : ℚ := 36 / 100

-- The problem is to prove that this fraction is equal to 0.36 in decimal form
theorem thirty_six_hundredths_is_decimal : fraction_thirty_six_hundredths = 0.36 := 
sorry

end thirty_six_hundredths

end thirty_six_hundredths_is_decimal_l68_68917


namespace trajectory_midpoint_of_chord_l68_68603

theorem trajectory_midpoint_of_chord :
  ∀ (M: ℝ × ℝ), (∃ (C D : ℝ × ℝ), (C.1^2 + C.2^2 = 25 ∧ D.1^2 + D.2^2 = 25 ∧ dist C D = 8) ∧ M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
  → M.1^2 + M.2^2 = 9 :=
sorry

end trajectory_midpoint_of_chord_l68_68603


namespace card_distribution_count_l68_68203

def card_distribution_ways : Nat := sorry

theorem card_distribution_count :
  card_distribution_ways = 9 := sorry

end card_distribution_count_l68_68203


namespace no_solution_l68_68207

theorem no_solution (m n : ℕ) : (5 + 3 * Real.sqrt 2) ^ m ≠ (3 + 5 * Real.sqrt 2) ^ n :=
sorry

end no_solution_l68_68207


namespace dealer_purchase_fraction_l68_68847

theorem dealer_purchase_fraction (P C : ℝ) (h1 : ∃ S, S = 1.5 * P) (h2 : ∃ S, S = 2 * C) :
  C / P = 3 / 8 :=
by
  -- The statement of the theorem has been generated based on the problem conditions.
  sorry

end dealer_purchase_fraction_l68_68847


namespace point_on_graph_l68_68551

def lies_on_graph (x y : ℝ) (f : ℝ → ℝ) : Prop :=
  y = f x

theorem point_on_graph :
  lies_on_graph (-2) 0 (λ x => (1 / 2) * x + 1) :=
by
  sorry

end point_on_graph_l68_68551


namespace circle_center_line_distance_l68_68661

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l68_68661


namespace monotonicity_of_f_range_of_a_l68_68168

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  Real.log x - a / x

theorem monotonicity_of_f (a : ℝ) (h : 0 < a) :
  ∀ x y : ℝ, (0 < x) → (0 < y) → (x < y) → (f x a < f y a) :=
by
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → f x a < x ^ 2) ↔ (-1 ≤ a) :=
by
  sorry

end monotonicity_of_f_range_of_a_l68_68168


namespace volume_in_cubic_yards_l68_68093

-- Adding the conditions as definitions
def feet_to_yards : ℝ := 3 -- 3 feet in a yard
def cubic_feet_to_cubic_yards : ℝ := feet_to_yards^3 -- convert to cubic yards
def volume_in_cubic_feet : ℝ := 108 -- volume in cubic feet

-- The theorem to prove the equivalence
theorem volume_in_cubic_yards
  (h1 : feet_to_yards = 3)
  (h2 : volume_in_cubic_feet = 108)
  : (volume_in_cubic_feet / cubic_feet_to_cubic_yards) = 4 := 
sorry

end volume_in_cubic_yards_l68_68093


namespace triangle_is_obtuse_l68_68464

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  B = 2 * A ∧ a = 1 ∧ b = 4 / 3 ∧ (a^2 + b^2 < c^2)

theorem triangle_is_obtuse (A B C : ℝ) (a b c : ℝ) (h : triangle_ABC A B C a b c) : 
  B > π / 2 :=
by
  sorry

end triangle_is_obtuse_l68_68464


namespace arith_prog_sum_eq_l68_68407

variable (a d : ℕ → ℤ)

def S (n : ℕ) : ℤ := (n / 2) * (2 * a 1 + (n - 1) * d 1)

theorem arith_prog_sum_eq (n : ℕ) : 
  S a d (n + 3) - 3 * S a d (n + 2) + 3 * S a d (n + 1) - S a d n = 0 := 
sorry

end arith_prog_sum_eq_l68_68407


namespace unknown_number_value_l68_68996

theorem unknown_number_value (a : ℕ) (n : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 3 = 21 * n * 45 * 49) : 
  n = 75 :=
sorry

end unknown_number_value_l68_68996


namespace calculate_expression_l68_68799

theorem calculate_expression : (-1^4 + |1 - Real.sqrt 2| - (Real.pi - 3.14)^0) = Real.sqrt 2 - 3 :=
by
  sorry

end calculate_expression_l68_68799


namespace average_class_weight_l68_68767

theorem average_class_weight :
  let students_A := 50
  let weight_A := 60
  let students_B := 60
  let weight_B := 80
  let students_C := 70
  let weight_C := 75
  let students_D := 80
  let weight_D := 85
  let total_students := students_A + students_B + students_C + students_D
  let total_weight := students_A * weight_A + students_B * weight_B + students_C * weight_C + students_D * weight_D
  (total_weight / total_students : ℝ) = 76.35 :=
by
  sorry

end average_class_weight_l68_68767


namespace sequence_term_formula_l68_68611

open Real

def sequence_sum_condition (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n : ℕ, n > 0 → S n + a n = 4 - 1 / (2 ^ (n - 2))

theorem sequence_term_formula 
  (S : ℕ → ℝ) (a : ℕ → ℝ) 
  (h : sequence_sum_condition S a) :
  ∀ n : ℕ, n > 0 → a n = n / 2 ^ (n - 1) :=
sorry

end sequence_term_formula_l68_68611


namespace is_prime_if_two_pow_n_minus_one_is_prime_is_power_of_two_if_two_pow_n_plus_one_is_prime_l68_68817

-- Problem 1: If \(2^{n} - 1\) is prime, then \(n\) is prime.
theorem is_prime_if_two_pow_n_minus_one_is_prime (n : ℕ) (hn : Prime (2^n - 1)) : Prime n :=
sorry

-- Problem 2: If \(2^{n} + 1\) is prime, then \(n\) is a power of 2.
theorem is_power_of_two_if_two_pow_n_plus_one_is_prime (n : ℕ) (hn : Prime (2^n + 1)) : ∃ k : ℕ, n = 2^k :=
sorry

end is_prime_if_two_pow_n_minus_one_is_prime_is_power_of_two_if_two_pow_n_plus_one_is_prime_l68_68817


namespace arithmetic_sequence_30th_term_l68_68809

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem arithmetic_sequence_30th_term :
  arithmetic_sequence 3 6 30 = 177 :=
by
  -- Proof steps go here
  sorry

end arithmetic_sequence_30th_term_l68_68809


namespace john_total_spent_l68_68051

noncomputable def computer_cost : ℝ := 1500
noncomputable def peripherals_cost : ℝ := (1 / 4) * computer_cost
noncomputable def base_video_card_cost : ℝ := 300
noncomputable def upgraded_video_card_cost : ℝ := 2.5 * base_video_card_cost
noncomputable def discount_on_video_card : ℝ := 0.12 * upgraded_video_card_cost
noncomputable def video_card_cost_after_discount : ℝ := upgraded_video_card_cost - discount_on_video_card
noncomputable def sales_tax_on_peripherals : ℝ := 0.05 * peripherals_cost
noncomputable def total_spent : ℝ := computer_cost + peripherals_cost + video_card_cost_after_discount + sales_tax_on_peripherals

theorem john_total_spent : total_spent = 2553.75 := by
  sorry

end john_total_spent_l68_68051


namespace min_megabytes_for_plan_Y_more_economical_l68_68453

theorem min_megabytes_for_plan_Y_more_economical :
  ∃ (m : ℕ), 2500 + 10 * m < 15 * m ∧ m = 501 :=
by
  sorry

end min_megabytes_for_plan_Y_more_economical_l68_68453


namespace longest_perimeter_l68_68280

theorem longest_perimeter 
  (x : ℝ) (h : x > 1)
  (pA : ℝ := 4 + 6 * x)
  (pB : ℝ := 2 + 10 * x)
  (pC : ℝ := 7 + 5 * x)
  (pD : ℝ := 6 + 6 * x)
  (pE : ℝ := 1 + 11 * x) :
  pE > pA ∧ pE > pB ∧ pE > pC ∧ pE > pD :=
by
  sorry

end longest_perimeter_l68_68280


namespace craig_apples_total_l68_68533

-- Conditions
def initial_apples := 20.0
def additional_apples := 7.0

-- Question turned into a proof problem
theorem craig_apples_total : initial_apples + additional_apples = 27.0 :=
by
  sorry

end craig_apples_total_l68_68533


namespace expression_value_l68_68568

-- Define the variables and the main statement
variable (w x y z : ℕ)

theorem expression_value :
  2^w * 3^x * 5^y * 11^z = 825 → w + 2 * x + 3 * y + 4 * z = 12 :=
by
  sorry -- Proof omitted

end expression_value_l68_68568


namespace perpendicular_vectors_l68_68751

theorem perpendicular_vectors (x y : ℝ) (a : ℝ × ℝ := (1, 2)) (b : ℝ × ℝ := (2 + x, 1 - y)) 
  (hperp : (a.1 * b.1 + a.2 * b.2 = 0)) : 2 * y - x = 4 :=
sorry

end perpendicular_vectors_l68_68751


namespace total_distance_traveled_l68_68312

def distance_from_earth_to_planet_x : ℝ := 0.5
def distance_from_planet_x_to_planet_y : ℝ := 0.1
def distance_from_planet_y_to_earth : ℝ := 0.1

theorem total_distance_traveled : 
  distance_from_earth_to_planet_x + distance_from_planet_x_to_planet_y + distance_from_planet_y_to_earth = 0.7 :=
by
  sorry

end total_distance_traveled_l68_68312


namespace one_third_eleven_y_plus_three_l68_68337

theorem one_third_eleven_y_plus_three (y : ℝ) : 
  (1/3) * (11 * y + 3) = 11 * y / 3 + 1 :=
by
  sorry

end one_third_eleven_y_plus_three_l68_68337


namespace difference_in_price_l68_68194

-- Definitions based on the given conditions
def price_with_cork : ℝ := 2.10
def price_cork : ℝ := 0.05
def price_without_cork : ℝ := price_with_cork - price_cork

-- The theorem proving the given question and correct answer
theorem difference_in_price : price_with_cork - price_without_cork = price_cork :=
by
  -- Proof can be omitted
  sorry

end difference_in_price_l68_68194


namespace probability_of_individual_selection_l68_68725

theorem probability_of_individual_selection (sample_size : ℕ) (population_size : ℕ)
  (h_sample : sample_size = 10) (h_population : population_size = 42) :
  (sample_size : ℚ) / (population_size : ℚ) = 5 / 21 := 
by {
  sorry
}

end probability_of_individual_selection_l68_68725


namespace intersection_of_sets_l68_68572

noncomputable def setA : Set ℝ := { x | |x - 2| ≤ 3 }
noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = 1 - x^2 }

theorem intersection_of_sets : (setA ∩ { x | 1 - x^2 ∈ setB }) = Set.Icc (-1) 1 :=
by
  sorry

end intersection_of_sets_l68_68572


namespace arrangements_count_correct_l68_68213

def arrangements_total : Nat :=
  let total_with_A_first := (Nat.factorial 5) -- A^5_5 = 120
  let total_with_B_first := (Nat.factorial 4) * 1 -- A^1_4 * A^4_4 = 96
  total_with_A_first + total_with_B_first

theorem arrangements_count_correct : arrangements_total = 216 := 
by
  -- Proof is required here
  sorry

end arrangements_count_correct_l68_68213


namespace band_and_chorus_but_not_orchestra_l68_68963

theorem band_and_chorus_but_not_orchestra (B C O : Finset ℕ)
  (hB : B.card = 100) 
  (hC : C.card = 120) 
  (hO : O.card = 60)
  (hUnion : (B ∪ C ∪ O).card = 200)
  (hIntersection : (B ∩ C ∩ O).card = 10) : 
  ((B ∩ C).card - (B ∩ C ∩ O).card = 30) :=
by sorry

end band_and_chorus_but_not_orchestra_l68_68963


namespace exist_indices_inequalities_l68_68682

open Nat

theorem exist_indices_inequalities (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by
  -- The proof is to be written here
  sorry

end exist_indices_inequalities_l68_68682


namespace fraction_to_decimal_l68_68592

theorem fraction_to_decimal : (47 : ℝ) / 160 = 0.29375 :=
by
  sorry

end fraction_to_decimal_l68_68592


namespace x_finishes_work_alone_in_18_days_l68_68098

theorem x_finishes_work_alone_in_18_days
  (y_days : ℕ) (y_worked : ℕ) (x_remaining_days : ℝ)
  (hy : y_days = 15) (hy_worked : y_worked = 10) 
  (hx_remaining : x_remaining_days = 6.000000000000001) :
  ∃ (x_days : ℝ), x_days = 18 :=
by 
  sorry

end x_finishes_work_alone_in_18_days_l68_68098


namespace positive_sqrt_729_l68_68017

theorem positive_sqrt_729 (x : ℝ) (h_pos : 0 < x) (h_eq : x^2 = 729) : x = 27 :=
by
  sorry

end positive_sqrt_729_l68_68017


namespace hypotenuse_length_l68_68880

theorem hypotenuse_length (a b c : ℕ) (h1 : a = 12) (h2 : b = 5) (h3 : c^2 = a^2 + b^2) : c = 13 := by
  sorry

end hypotenuse_length_l68_68880


namespace probability_interval_l68_68053

theorem probability_interval (P_A P_B p : ℝ) (hP_A : P_A = 2 / 3) (hP_B : P_B = 3 / 5) :
  4 / 15 ≤ p ∧ p ≤ 3 / 5 := sorry

end probability_interval_l68_68053


namespace odd_function_periodicity_l68_68853

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_periodicity (f_odd : ∀ x, f (-x) = -f x)
  (f_periodic : ∀ x, f (x + 2) = -f x) (f_val : f 1 = 2) : f 2011 = -2 :=
by
  sorry

end odd_function_periodicity_l68_68853


namespace sum_of_a_and_b_l68_68522

theorem sum_of_a_and_b (a b : ℝ) (h_neq : a ≠ b) (h_a : a * (a - 4) = 21) (h_b : b * (b - 4) = 21) :
  a + b = 4 :=
by
  sorry

end sum_of_a_and_b_l68_68522


namespace simplify_expression_l68_68997

theorem simplify_expression : 
  1.5 * (Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + Real.sqrt (1+3+5+7+9)) = 22.5 :=
by
  sorry

end simplify_expression_l68_68997


namespace savings_account_amount_l68_68369

theorem savings_account_amount (stimulus : ℕ) (wife_ratio first_son_ratio wife_share first_son_share second_son_share : ℕ) : 
    stimulus = 2000 →
    wife_ratio = 2 / 5 →
    first_son_ratio = 2 / 5 →
    wife_share = wife_ratio * stimulus →
    first_son_share = first_son_ratio * (stimulus - wife_share) →
    second_son_share = 40 / 100 * (stimulus - wife_share - first_son_share) →
    (stimulus - wife_share - first_son_share - second_son_share) = 432 :=
by
  sorry

end savings_account_amount_l68_68369


namespace main_theorem_l68_68645

variables {m n : ℕ} {x : ℝ}
variables {a : ℕ → ℕ}
noncomputable def relatively_prime (a : ℕ → ℕ) (n : ℕ) : Prop :=
∀ i j, i ≠ j → i < n → j < n → Nat.gcd (a i) (a j) = 1

noncomputable def distinct (a : ℕ → ℕ) (n : ℕ) : Prop :=
∀ i j, i ≠ j → i < n → j < n → a i ≠ a j

theorem main_theorem (hm : 1 < m) (hn : 1 < n) (hge : m ≥ n)
  (hrel_prime : relatively_prime a n)
  (hdistinct : distinct a n)
  (hbound : ∀ i, i < n → a i ≤ m)
  : ∃ i, i < n ∧ ‖a i * x‖ ≥ (2 / (m * (m + 1))) * ‖x‖ := 
sorry

end main_theorem_l68_68645


namespace Jane_Hector_meet_point_C_l68_68412

theorem Jane_Hector_meet_point_C (s t : ℝ) (h_start : ℝ) (j_start : ℝ) (loop_length : ℝ) 
  (h_speed : ℝ) (j_speed : ℝ) (h_dest : ℝ) (j_dest : ℝ)
  (h_speed_eq : h_speed = s) (j_speed_eq : j_speed = 3 * s) (loop_len_eq : loop_length = 30)
  (start_point_eq : h_start = 0 ∧ j_start = 0)
  (opposite_directions : h_dest + j_dest = loop_length)
  (meet_time_eq : t = 15 / (2 * s)) :
  h_dest = 7.5 ∧ j_dest = 22.5 → (h_dest = 7.5 ∧ j_dest = 22.5) :=
by
  sorry

end Jane_Hector_meet_point_C_l68_68412


namespace line_tangent_to_circle_l68_68972

theorem line_tangent_to_circle (r : ℝ) :
  (∀ (x y : ℝ), (x + y = 4) → (x - 2)^2 + (y + 1)^2 = r) → r = 9 / 2 :=
sorry

end line_tangent_to_circle_l68_68972


namespace coordinates_P_correct_l68_68803

noncomputable def coordinates_of_P : ℝ × ℝ :=
  let x_distance_to_y_axis : ℝ := 5
  let y_distance_to_x_axis : ℝ := 4
  -- x-coordinate must be negative, y-coordinate must be positive
  let x_coord : ℝ := -x_distance_to_y_axis
  let y_coord : ℝ := y_distance_to_x_axis
  (x_coord, y_coord)

theorem coordinates_P_correct:
  coordinates_of_P = (-5, 4) :=
by
  sorry

end coordinates_P_correct_l68_68803


namespace fraction_addition_l68_68310

theorem fraction_addition : (1 + 3 + 5)/(2 + 4 + 6) + (2 + 4 + 6)/(1 + 3 + 5) = 25/12 := by
  sorry

end fraction_addition_l68_68310


namespace distance_between_cities_l68_68498

theorem distance_between_cities (x : ℝ) (h1 : x ≥ 100) (t : ℝ)
  (A_speed : ℝ := 12) (B_speed : ℝ := 0.05 * x)
  (condition_A : 7 + A_speed * t + B_speed * t = x)
  (condition_B : t = (x - 7) / (A_speed + B_speed)) :
  x = 140 :=
sorry

end distance_between_cities_l68_68498


namespace variance_decreases_l68_68733

def scores_initial := [5, 9, 7, 10, 9] -- Initial 5 shot scores
def additional_shot := 8 -- Additional shot score

-- Given variance of initial scores
def variance_initial : ℝ := 3.2

-- Placeholder function to calculate variance of a list of scores
noncomputable def variance (scores : List ℝ) : ℝ := sorry

-- Definition of the new scores list
def scores_new := scores_initial ++ [additional_shot]

-- Define the proof problem
theorem variance_decreases :
  variance scores_new < variance_initial :=
sorry

end variance_decreases_l68_68733


namespace total_onions_l68_68552

theorem total_onions (S SA F J : ℕ) (h1 : S = 4) (h2 : SA = 5) (h3 : F = 9) (h4 : J = 7) : S + SA + F + J = 25 :=
by {
  sorry
}

end total_onions_l68_68552


namespace find_line_equation_l68_68396

def parameterized_line (t : ℝ) : ℝ × ℝ :=
  (3 * t + 2, 5 * t - 3)

theorem find_line_equation (x y : ℝ) (t : ℝ) (h : parameterized_line t = (x, y)) :
  y = (5 / 3) * x - 19 / 3 :=
sorry

end find_line_equation_l68_68396


namespace triangle_inequality_equality_condition_l68_68109

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c :=
sorry

end triangle_inequality_equality_condition_l68_68109


namespace find_x_l68_68338

variable (x : ℝ)  -- Current distance Teena is behind Loe in miles
variable (t : ℝ) -- Time period in hours
variable (speed_teena : ℝ) -- Speed of Teena in miles per hour
variable (speed_loe : ℝ) -- Speed of Loe in miles per hour
variable (d_ahead : ℝ) -- Distance Teena will be ahead of Loe in 1.5 hours

axiom conditions : speed_teena = 55 ∧ speed_loe = 40 ∧ t = 1.5 ∧ d_ahead = 15

theorem find_x : (speed_teena * t - speed_loe * t = x + d_ahead) → x = 7.5 :=
by
  intro h
  sorry

end find_x_l68_68338


namespace remainder_of_polynomial_division_l68_68953

noncomputable def evaluate_polynomial (x : ℂ) : ℂ :=
  x^100 + x^75 + x^50 + x^25 + 1

noncomputable def divisor_polynomial (x : ℂ) : ℂ :=
  x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_of_polynomial_division : 
  ∀ β : ℂ, divisor_polynomial β = 0 → evaluate_polynomial β = -1 :=
by
  intros β hβ
  sorry

end remainder_of_polynomial_division_l68_68953


namespace man_mass_calculation_l68_68260

/-- A boat has a length of 4 m, a breadth of 2 m, and a weight of 300 kg.
    The density of the water is 1000 kg/m³.
    When the man gets on the boat, it sinks by 1 cm.
    Prove that the mass of the man is 80 kg. -/
theorem man_mass_calculation :
  let length_boat := 4     -- in meters
  let breadth_boat := 2    -- in meters
  let weight_boat := 300   -- in kg
  let density_water := 1000  -- in kg/m³
  let additional_depth := 0.01 -- in meters (1 cm)
  volume_displaced = length_boat * breadth_boat * additional_depth →
  mass_water_displaced = volume_displaced * density_water →
  mass_of_man = mass_water_displaced →
  mass_of_man = 80 :=
by 
  intros length_boat breadth_boat weight_boat density_water additional_depth volume_displaced
  intros mass_water_displaced mass_of_man
  sorry

end man_mass_calculation_l68_68260


namespace binomial_square_expression_l68_68760

theorem binomial_square_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end binomial_square_expression_l68_68760


namespace solution_interval_l68_68085

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 x + x - 2

theorem solution_interval :
  ∃ x, (1 < x ∧ x < 1.5) ∧ f x = 0 :=
sorry

end solution_interval_l68_68085


namespace good_numbers_characterization_l68_68234

def is_good (n : ℕ) : Prop :=
  ∀ d, d ∣ n → d + 1 ∣ n + 1

theorem good_numbers_characterization :
  {n : ℕ | is_good n} = {1} ∪ {p | Nat.Prime p ∧ p % 2 = 1} :=
by 
  sorry

end good_numbers_characterization_l68_68234


namespace min_max_value_l68_68763

-- Definition of the function to be minimized and maximized
def f (x y : ℝ) : ℝ := |x^3 - x * y^2|

-- Conditions
def x_condition (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def y_condition (y : ℝ) : Prop := true

-- Goal: Prove the minimum of the maximum value
theorem min_max_value :
  ∃ y : ℝ, (∀ x : ℝ, x_condition x → f x y ≤ 8) ∧ (∀ y' : ℝ, (∀ x : ℝ, x_condition x → f x y' ≤ 8) → y' = y) :=
sorry

end min_max_value_l68_68763


namespace area_of_common_region_l68_68332

noncomputable def common_area (length : ℝ) (width : ℝ) (radius : ℝ) : ℝ :=
  let pi := Real.pi
  let sector_area := (pi * radius^2 / 4) * 4
  let triangle_area := (1 / 2) * (width / 2) * (length / 2) * 4
  sector_area - triangle_area

theorem area_of_common_region :
  common_area 10 (Real.sqrt 18) 3 = 9 * (Real.pi) - 9 :=
by
  sorry

end area_of_common_region_l68_68332


namespace staff_price_l68_68576

theorem staff_price (d : ℝ) : (d - 0.55 * d) / 2 = 0.225 * d := by
  sorry

end staff_price_l68_68576


namespace trihedral_sphere_radius_l68_68807

noncomputable def sphere_radius 
  (α r : ℝ) 
  (hα : 0 < α ∧ α < (Real.pi / 2)) 
  : ℝ :=
r * Real.sqrt ((4 * (Real.cos (α / 2)) ^ 2 + 3) / 3)

theorem trihedral_sphere_radius 
  (α r R : ℝ) 
  (hα : 0 < α ∧ α < (Real.pi / 2)) 
  (hR : R = sphere_radius α r hα) 
  : R = r * Real.sqrt ((4 * (Real.cos (α / 2)) ^ 2 + 3) / 3) :=
by
  sorry

end trihedral_sphere_radius_l68_68807


namespace election_at_least_one_past_officer_l68_68785

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem election_at_least_one_past_officer : 
  let total_candidates := 16
  let past_officers := 7
  let officer_positions := 5
  choose total_candidates officer_positions - choose (total_candidates - past_officers) officer_positions = 4242 :=
by
  sorry

end election_at_least_one_past_officer_l68_68785


namespace find_functions_l68_68761

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def domain (f g : ℝ → ℝ) : Prop := ∀ x, x ≠ 1 → x ≠ -1 → true

theorem find_functions
  (f g : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_even : is_even g)
  (h_domain : domain f g)
  (h_eq : ∀ x, x ≠ 1 → x ≠ -1 → f x + g x = 1 / (x - 1)) :
  (∀ x, x ≠ 1 → x ≠ -1 → f x = x / (x^2 - 1)) ∧ 
  (∀ x, x ≠ 1 → x ≠ -1 → g x = 1 / (x^2 - 1)) := 
by
  sorry

end find_functions_l68_68761


namespace sandy_comic_books_l68_68840

-- Define Sandy's initial number of comic books
def initial_comic_books : ℕ := 14

-- Define the number of comic books Sandy sold
def sold_comic_books (n : ℕ) : ℕ := n / 2

-- Define the number of comic books Sandy bought
def bought_comic_books : ℕ := 6

-- Define the number of comic books Sandy has now
def final_comic_books (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  initial - sold + bought

-- The theorem statement to prove the final number of comic books
theorem sandy_comic_books : final_comic_books initial_comic_books (sold_comic_books initial_comic_books) bought_comic_books = 13 := by
  sorry

end sandy_comic_books_l68_68840


namespace smallest_value_expression_l68_68843

theorem smallest_value_expression
    (a b c : ℝ) 
    (h1 : c > b)
    (h2 : b > a)
    (h3 : c ≠ 0) : 
    ∃ z : ℝ, z = 0 ∧ z = (a + b)^2 / c^2 + (b - c)^2 / c^2 + (c - b)^2 / c^2 :=
by
  sorry

end smallest_value_expression_l68_68843


namespace class_boys_count_l68_68387

theorem class_boys_count
    (x y : ℕ)
    (h1 : x + y = 20)
    (h2 : (1 / 3 : ℚ) * x = (1 / 2 : ℚ) * y) :
    x = 12 :=
by
  sorry

end class_boys_count_l68_68387


namespace arithmetic_sequence_binomial_l68_68884

theorem arithmetic_sequence_binomial {n k u : ℕ} (h₁ : u ≥ 3)
    (h₂ : n = u^2 - 2)
    (h₃ : k = Nat.choose u 2 - 1 ∨ k = Nat.choose (u + 1) 2 - 1)
    : (Nat.choose n (k - 1)) - 2 * (Nat.choose n k) + (Nat.choose n (k + 1)) = 0 :=
by
  sorry

end arithmetic_sequence_binomial_l68_68884


namespace annie_gives_mary_25_crayons_l68_68291

theorem annie_gives_mary_25_crayons :
  let initial_crayons_given := 21
  let initial_crayons_in_locker := 36
  let bobby_gift := initial_crayons_in_locker / 2
  let total_crayons := initial_crayons_given + initial_crayons_in_locker + bobby_gift
  let mary_share := total_crayons / 3
  mary_share = 25 := 
by
  sorry

end annie_gives_mary_25_crayons_l68_68291


namespace imperative_sentence_structure_l68_68612

theorem imperative_sentence_structure (word : String) (is_base_form : word = "Surround") :
  (word = "Surround" ∨ word = "Surrounding" ∨ word = "Surrounded" ∨ word = "Have surrounded") →
  (∃ sentence : String, sentence = word ++ " yourself with positive people, and you will keep focused on what you can do instead of what you can’t.") →
  word = "Surround" :=
by
  intros H_choice H_sentence
  cases H_choice
  case inl H1 => assumption
  case inr H2_1 =>
    cases H2_1
    case inl H2_1_1 => sorry
    case inr H2_1_2 =>
      cases H2_1_2
      case inl H2_1_2_1 => sorry
      case inr H2_1_2_2 => sorry

end imperative_sentence_structure_l68_68612


namespace alphanumeric_puzzle_l68_68434

/-- Alphanumeric puzzle proof problem -/
theorem alphanumeric_puzzle
  (A B C D E F H J K L : Nat)
  (h1 : A * B = B)
  (h2 : B * C = 10 * A + C)
  (h3 : C * D = 10 * B + C)
  (h4 : D * E = 100 * C + H)
  (h5 : E * F = 10 * D + K)
  (h6 : F * H = 100 * C + J)
  (h7 : H * J = 10 * K + J)
  (h8 : J * K = E)
  (h9 : K * L = L)
  (h10 : A * L = L) :
  A = 1 ∧ B = 3 ∧ C = 5 ∧ D = 7 ∧ E = 8 ∧ F = 9 ∧ H = 6 ∧ J = 4 ∧ K = 2 ∧ L = 0 :=
sorry

end alphanumeric_puzzle_l68_68434


namespace player_A_wins_after_4_rounds_geometric_sequence_differences_find_P_2_l68_68812

-- Define probabilities of shots
def shooting_probability_A : ℝ := 0.5
def shooting_probability_B : ℝ := 0.6

-- Define initial points for questions
def initial_points_question_1 : ℝ := 0
def initial_points_question_2 : ℝ := 2

-- Given initial probabilities
def P_0 : ℝ := 0
def P_4 : ℝ := 1

-- Probability that player A wins after exactly 4 rounds
def probability_A_wins_after_4_rounds : ℝ :=
  let P_A := shooting_probability_A * (1 - shooting_probability_B)
  let P_B := shooting_probability_B * (1 - shooting_probability_A)
  let P_C := 1 - P_A - P_B
  P_A * P_C^2 * P_A + P_A * P_B * P_A^2

-- Define the probabilities P(i) for i=0..4
def P (i : ℕ) : ℝ := sorry -- Placeholder for the function

-- Define the proof problem
theorem player_A_wins_after_4_rounds : probability_A_wins_after_4_rounds = 0.0348 :=
sorry

theorem geometric_sequence_differences :
  ∀ i : ℕ, i < 4 → (P (i + 1) - P i) / (P (i + 2) - P (i + 1)) = 2/3 :=
sorry

theorem find_P_2 : P 2 = 4/13 :=
sorry

end player_A_wins_after_4_rounds_geometric_sequence_differences_find_P_2_l68_68812


namespace find_min_value_c_l68_68974

theorem find_min_value_c (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 2010) :
  (∃ x y : ℤ, 3 * x + y = 3005 ∧ y = abs (x - a) + abs (x - 2 * b) + abs (x - c) ∧
   (∀ x' y' : ℤ, 3 * x' + y' = 3005 → y' = abs (x' - a) + abs (x' - 2 * b) + abs (x' - c) → x = x' ∧ y = y')) →
  c ≥ 1014 :=
by
  sorry

end find_min_value_c_l68_68974


namespace percentage_of_cars_with_no_features_l68_68182

theorem percentage_of_cars_with_no_features (N S W R SW SR WR SWR : ℕ)
  (hN : N = 120)
  (hS : S = 70)
  (hW : W = 40)
  (hR : R = 30)
  (hSW : SW = 20)
  (hSR : SR = 15)
  (hWR : WR = 10)
  (hSWR : SWR = 5) :
  (120 - (S + W + R - SW - SR - WR + SWR)) / (N : ℝ) * 100 = 16.67 :=
by
  sorry

end percentage_of_cars_with_no_features_l68_68182


namespace permutations_of_six_attractions_is_720_l68_68024

-- Define the number of attractions
def num_attractions : ℕ := 6

-- State the theorem to be proven
theorem permutations_of_six_attractions_is_720 : (num_attractions.factorial = 720) :=
by {
  sorry
}

end permutations_of_six_attractions_is_720_l68_68024


namespace rowing_speed_in_still_water_l68_68166

theorem rowing_speed_in_still_water (v c : ℝ) (h1 : c = 1.5) 
(h2 : ∀ t : ℝ, (v + c) * t = (v - c) * 2 * t) : 
  v = 4.5 :=
by
  sorry

end rowing_speed_in_still_water_l68_68166


namespace inequality_solution_l68_68276

theorem inequality_solution (x : ℝ) :
  (x / (x^2 - 4) ≥ 0) ↔ (x ∈ Set.Iio (-2) ∪ Set.Ico 0 2) :=
by sorry

end inequality_solution_l68_68276


namespace ladder_base_distance_l68_68256

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l68_68256


namespace general_term_formula_sum_of_2_pow_an_l68_68741

variable {S : ℕ → ℕ}
variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {T : ℕ → ℕ}

axiom S5_eq_30 : S 5 = 30
axiom a1_a6_eq_14 : a 1 + a 6 = 14

theorem general_term_formula : ∀ n, a n = 2 * n :=
sorry

theorem sum_of_2_pow_an (n : ℕ) : T n = (4^(n + 1)) / 3 - 4 / 3 :=
sorry

end general_term_formula_sum_of_2_pow_an_l68_68741


namespace probability_bons_wins_even_rolls_l68_68579
noncomputable def probability_of_Bons_winning (p6 : ℚ) (p_not6 : ℚ) : ℚ := 
  let r := p_not6^2
  let a := p_not6 * p6
  a / (1 - r)

theorem probability_bons_wins_even_rolls : 
  let p6 := (1 : ℚ) / 6
  let p_not6 := (5 : ℚ) / 6
  probability_of_Bons_winning p6 p_not6 = (5 : ℚ) / 11 := 
  sorry

end probability_bons_wins_even_rolls_l68_68579


namespace hydrogen_atoms_count_l68_68430

-- Definitions for atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Given conditions
def total_molecular_weight : ℝ := 88
def number_of_C_atoms : ℕ := 4
def number_of_O_atoms : ℕ := 2

theorem hydrogen_atoms_count (nh : ℕ) 
  (h_molecular_weight : total_molecular_weight = 88) 
  (h_C_atoms : number_of_C_atoms = 4) 
  (h_O_atoms : number_of_O_atoms = 2) :
  nh = 8 :=
by
  -- skipping proof
  sorry

end hydrogen_atoms_count_l68_68430


namespace find_f_prime_one_l68_68400

theorem find_f_prime_one (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f' x = 2 * f' 1 + 1 / x) (h_fx : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f' 1 = -1 := 
by 
  sorry

end find_f_prime_one_l68_68400


namespace carol_ate_12_cakes_l68_68279

-- Definitions for conditions
def cakes_per_day : ℕ := 10
def days_baking : ℕ := 5
def cans_per_cake : ℕ := 2
def cans_for_remaining_cakes : ℕ := 76

-- Total cakes baked by Sara
def total_cakes_baked (cakes_per_day days_baking : ℕ) : ℕ :=
  cakes_per_day * days_baking

-- Remaining cakes based on frosting cans needed
def remaining_cakes (cans_for_remaining_cakes cans_per_cake : ℕ) : ℕ :=
  cans_for_remaining_cakes / cans_per_cake

-- Cakes Carol ate
def cakes_carol_ate (total_cakes remaining_cakes : ℕ) : ℕ :=
  total_cakes - remaining_cakes

-- Theorem statement
theorem carol_ate_12_cakes :
  cakes_carol_ate (total_cakes_baked cakes_per_day days_baking) (remaining_cakes cans_for_remaining_cakes cans_per_cake) = 12 :=
by
  sorry

end carol_ate_12_cakes_l68_68279


namespace female_sample_count_is_correct_l68_68541

-- Definitions based on the given conditions
def total_students : ℕ := 900
def male_students : ℕ := 500
def sample_size : ℕ := 45
def female_students : ℕ := total_students - male_students
def female_sample_size : ℕ := (female_students * sample_size) / total_students

-- The lean statement to prove
theorem female_sample_count_is_correct : female_sample_size = 20 := 
by 
  -- A placeholder to indicate the proof needs to be filled in
  sorry

end female_sample_count_is_correct_l68_68541


namespace washing_machine_cost_l68_68501

variable (W D : ℝ)
variable (h1 : D = W - 30)
variable (h2 : 0.90 * (W + D) = 153)

theorem washing_machine_cost :
  W = 100 := by
  sorry

end washing_machine_cost_l68_68501


namespace only_integer_triplet_solution_l68_68856

theorem only_integer_triplet_solution 
  (a b c : ℤ) : 5 * a^2 + 9 * b^2 = 13 * c^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by 
  intro h
  sorry

end only_integer_triplet_solution_l68_68856


namespace quadratic_roots_difference_square_l68_68331

theorem quadratic_roots_difference_square (a b : ℝ) (h : 2 * a^2 - 8 * a + 6 = 0 ∧ 2 * b^2 - 8 * b + 6 = 0) :
  (a - b) ^ 2 = 4 :=
sorry

end quadratic_roots_difference_square_l68_68331


namespace train_cross_time_l68_68424

/-- Given the conditions:
1. Two trains run in opposite directions and cross a man in 17 seconds and some unknown time respectively.
2. They cross each other in 22 seconds.
3. The ratio of their speeds is 1 to 1.
Prove the time it takes for the first train to cross the man. -/
theorem train_cross_time (v_1 v_2 L_1 L_2 : ℝ) (t_2 : ℝ) (h1 : t_2 = 17) (h2 : v_1 = v_2)
  (h3 : (L_1 + L_2) / (v_1 + v_2) = 22) : (L_1 / v_1) = 27 := 
by 
  -- The actual proof will go here
  sorry

end train_cross_time_l68_68424


namespace total_cotton_yield_l68_68914

variables {m n a b : ℕ}

theorem total_cotton_yield (m n a b : ℕ) : 
  m * a + n * b = m * a + n * b := by
  sorry

end total_cotton_yield_l68_68914


namespace floor_e_minus_3_eq_negative_one_l68_68380

theorem floor_e_minus_3_eq_negative_one 
  (e : ℝ) 
  (h : 2 < e ∧ e < 3) : 
  (⌊e - 3⌋ = -1) :=
by
  sorry

end floor_e_minus_3_eq_negative_one_l68_68380


namespace repeating_decimals_fraction_l68_68589

theorem repeating_decimals_fraction :
  (0.81:ℚ) / (0.36:ℚ) = 9 / 4 :=
by
  have h₁ : (0.81:ℚ) = 81 / 99 := sorry
  have h₂ : (0.36:ℚ) = 36 / 99 := sorry
  sorry

end repeating_decimals_fraction_l68_68589


namespace painted_cube_ways_l68_68641

theorem painted_cube_ways (b r g : ℕ) (cubes : ℕ) : 
  b = 1 → r = 2 → g = 3 → cubes = 3 := 
by
  intros
  sorry

end painted_cube_ways_l68_68641


namespace train_length_l68_68111

theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) (speed_conversion_factor : ℝ) (approx_length : ℝ) :
  speed_km_hr = 60 → time_seconds = 6 → speed_conversion_factor = (1000 / 3600) → approx_length = 100.02 →
  speed_km_hr * speed_conversion_factor * time_seconds = approx_length :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  sorry

end train_length_l68_68111


namespace roots_numerically_equal_opposite_signs_l68_68263

theorem roots_numerically_equal_opposite_signs
  (a b d: ℝ) 
  (h: ∃ x : ℝ, (x^2 - (a + 1) * x) / ((b + 1) * x - d) = (n - 2) / (n + 2) ∧ x = -x)
  : n = 2 * (b - a) / (a + b + 2) := by
  sorry

end roots_numerically_equal_opposite_signs_l68_68263


namespace cyclist_C_speed_l68_68973

variables (c d : ℕ) -- Speeds of cyclists C and D in mph
variables (d_eq : d = c + 6) -- Cyclist D travels 6 mph faster than cyclist C
variables (h1 : 80 = 65 + 15) -- Total distance from X to Y and back to the meet point
variables (same_time : 65 / c = 95 / d) -- Equating the travel times of both cyclists

theorem cyclist_C_speed : c = 13 :=
by
  sorry -- Proof is omitted

end cyclist_C_speed_l68_68973


namespace paul_work_days_l68_68047

theorem paul_work_days (P : ℕ) (h : 1 / P + 1 / 120 = 1 / 48) : P = 80 := 
by 
  sorry

end paul_work_days_l68_68047


namespace find_asterisk_l68_68230

theorem find_asterisk : ∃ (x : ℕ), (63 / 21) * (x / 189) = 1 ∧ x = 63 :=
by
  sorry

end find_asterisk_l68_68230


namespace face_value_of_ticket_l68_68632
noncomputable def face_value_each_ticket (total_price : ℝ) (group_size : ℕ) (tax_rate : ℝ) : ℝ :=
  total_price / (group_size * (1 + tax_rate))

theorem face_value_of_ticket (total_price : ℝ) (group_size : ℕ) (tax_rate : ℝ) :
  total_price = 945 →
  group_size = 25 →
  tax_rate = 0.05 →
  face_value_each_ticket total_price group_size tax_rate = 36 := 
by
  intros h_total_price h_group_size h_tax_rate
  rw [h_total_price, h_group_size, h_tax_rate]
  simp [face_value_each_ticket]
  sorry

end face_value_of_ticket_l68_68632


namespace probability_product_is_square_l68_68441

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

noncomputable def probability_square_product : ℚ :=
  let total_outcomes   := 10 * 8
  let favorable_outcomes := 
    [(1,1), (1,4), (2,2), (4,1), (3,3), (2,8), (8,2), (5,5), (6,6), (7,7), (8,8)].length
  favorable_outcomes / total_outcomes

theorem probability_product_is_square : 
  probability_square_product = 11 / 80 :=
  sorry

end probability_product_is_square_l68_68441


namespace zoo_children_tuesday_l68_68076

theorem zoo_children_tuesday 
  (x : ℕ) 
  (child_ticket_cost adult_ticket_cost : ℕ) 
  (children_monday adults_monday adults_tuesday : ℕ)
  (total_revenue : ℕ) : 
  child_ticket_cost = 3 → 
  adult_ticket_cost = 4 → 
  children_monday = 7 → 
  adults_monday = 5 → 
  adults_tuesday = 2 → 
  total_revenue = 61 → 
  7 * 3 + 5 * 4 + x * 3 + 2 * 4 = total_revenue → 
  x = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end zoo_children_tuesday_l68_68076


namespace part_I_part_II_l68_68663

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + a * x^2 - (2 * a + 1) * x

theorem part_I (a : ℝ) (ha : a = -2) : 
  (∃ x : ℝ, f a x = 1) ∧ ∀ x : ℝ, f a x ≤ 1 :=
by sorry

theorem part_II (a : ℝ) (ha : a < 1/2) :
  (∃ x : ℝ, 0 < x ∧ x < exp 1 ∧ f a x < 0) → a < (exp 1 - 1) / (exp 1 * (exp 1 - 2)) :=
by sorry

end part_I_part_II_l68_68663


namespace find_difference_l68_68648

theorem find_difference (x0 y0 : ℝ) 
  (h1 : x0^3 - 2023 * x0 = y0^3 - 2023 * y0 + 2020)
  (h2 : x0^2 + x0 * y0 + y0^2 = 2022) : 
  x0 - y0 = -2020 :=
by
  sorry

end find_difference_l68_68648


namespace find_missing_ratio_l68_68205

theorem find_missing_ratio
  (x y : ℕ)
  (h : ((2 / 3 : ℚ) * (x / y : ℚ) * (11 / 2 : ℚ) = 2)) :
  x = 6 ∧ y = 11 :=
sorry

end find_missing_ratio_l68_68205


namespace function_identity_l68_68879

theorem function_identity (f : ℕ → ℕ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ, f (m * n) = f m * f n)
  (h3 : ∀ n : ℕ, f (n + 1) > f n) : 
  ∀ n : ℕ, f n = n :=
sorry

end function_identity_l68_68879


namespace min_value_of_M_l68_68590

noncomputable def min_val (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :=
  max (1/(a*c) + b) (max (1/a + b*c) (a/b + c))

theorem min_value_of_M (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (min_val a b c h1 h2 h3) >= 2 :=
sorry

end min_value_of_M_l68_68590


namespace largest_four_digit_number_last_digit_l68_68796

theorem largest_four_digit_number_last_digit (n : ℕ) (n' : ℕ) (m r a b : ℕ) :
  (1000 * m + 100 * r + 10 * a + b = n) →
  (100 * m + 10 * r + a = n') →
  (n % 9 = 0) →
  (n' % 4 = 0) →
  b = 3 :=
by
  sorry

end largest_four_digit_number_last_digit_l68_68796


namespace largest_base7_three_digit_is_342_l68_68774

-- Definition of the base-7 number 666
def base7_666 : ℕ := 6 * 7^2 + 6 * 7^1 + 6 * 7^0

-- The largest decimal number represented by a three-digit base-7 number is 342
theorem largest_base7_three_digit_is_342 : base7_666 = 342 := by
  sorry

end largest_base7_three_digit_is_342_l68_68774


namespace minimum_value_of_expression_l68_68851

theorem minimum_value_of_expression (x y z w : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) (h : 5 * w = 3 * x ∧ 5 * w = 4 * y ∧ 5 * w = 7 * z) : x - y + z - w = 11 :=
sorry

end minimum_value_of_expression_l68_68851


namespace solution_set_of_inequality_l68_68530

theorem solution_set_of_inequality (x : ℝ) : (∃ x, (0 ≤ x ∧ x < 1) ↔ (x-2)/(x-1) ≥ 2) :=
sorry

end solution_set_of_inequality_l68_68530


namespace shiela_used_seven_colors_l68_68830

theorem shiela_used_seven_colors (total_blocks : ℕ) (blocks_per_color : ℕ)
  (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) : (total_blocks / blocks_per_color) = 7 :=
by
  sorry

end shiela_used_seven_colors_l68_68830


namespace star_five_seven_l68_68926

def star (a b : ℕ) : ℕ := (a + b + 3) ^ 2

theorem star_five_seven : star 5 7 = 225 := by
  sorry

end star_five_seven_l68_68926


namespace inequality_lemma_l68_68146

theorem inequality_lemma (x y z : ℝ) (h1 : 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z)
    (h2 : (1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1)) :
    (1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1) := 
by
  sorry

end inequality_lemma_l68_68146


namespace necessary_but_not_sufficient_l68_68569

theorem necessary_but_not_sufficient (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by {
  sorry
}

end necessary_but_not_sufficient_l68_68569


namespace moles_of_Br2_combined_l68_68480

-- Definition of the reaction relation
def chemical_reaction (CH4 Br2 CH3Br HBr : ℕ) : Prop :=
  CH4 = 1 ∧ HBr = 1

-- Statement of the proof problem
theorem moles_of_Br2_combined (CH4 Br2 CH3Br HBr : ℕ) (h : chemical_reaction CH4 Br2 CH3Br HBr) : Br2 = 1 :=
by
  sorry

end moles_of_Br2_combined_l68_68480


namespace product_y_coordinates_l68_68982

theorem product_y_coordinates : 
  ∀ y : ℝ, (∀ P : ℝ × ℝ, P.1 = -1 ∧ (P.1 - 4)^2 + (P.2 - 3)^2 = 64 → P = (-1, y)) →
  ((3 + Real.sqrt 39) * (3 - Real.sqrt 39) = -30) :=
by
  intros y h
  sorry

end product_y_coordinates_l68_68982


namespace square_of_binomial_l68_68462

theorem square_of_binomial (c : ℝ) (h : ∃ a : ℝ, x^2 + 50 * x + c = (x + a)^2) : c = 625 :=
by
  sorry

end square_of_binomial_l68_68462


namespace find_x_l68_68936

theorem find_x (x : ℕ) : 8000 * 6000 = x * 10^5 → x = 480 := by
  sorry

end find_x_l68_68936


namespace agatha_initial_money_l68_68195

/-
Agatha has some money to spend on a new bike. She spends $15 on the frame, and $25 on the front wheel.
If she has $20 left to spend on a seat and handlebar tape, prove that she had $60 initially.
-/

theorem agatha_initial_money (frame_cost wheel_cost remaining_money initial_money: ℕ) 
  (h1 : frame_cost = 15) 
  (h2 : wheel_cost = 25) 
  (h3 : remaining_money = 20) 
  (h4 : initial_money = frame_cost + wheel_cost + remaining_money) : 
  initial_money = 60 :=
by {
  -- We state explicitly that initial_money should be 60
  sorry
}

end agatha_initial_money_l68_68195


namespace proposition_A_proposition_B_proposition_C_proposition_D_l68_68702

-- Definitions and conditions for proposition A
def propA_conditions (a b : ℝ) : Prop :=
  a > b ∧ (1 / a) > (1 / b)

def propA (a b : ℝ) : Prop :=
  a * b < 0

-- Definitions and conditions for proposition B
def propB_conditions (a b : ℝ) : Prop :=
  a < b ∧ b < 0

def propB (a b : ℝ) : Prop :=
  a^2 < a * b ∧ a * b < b^2

-- Definitions and conditions for proposition C
def propC_conditions (c a b : ℝ) : Prop :=
  c > a ∧ a > b ∧ b > 0

def propC (c a b : ℝ) : Prop :=
  (a / (c - a)) < (b / (c - b))

-- Definitions and conditions for proposition D
def propD_conditions (a b c : ℝ) : Prop :=
  a > b ∧ b > c ∧ c > 0

def propD (a b c : ℝ) : Prop :=
  (a / b) > ((a + c) / (b + c))

-- The propositions
theorem proposition_A (a b : ℝ) (h : propA_conditions a b) : propA a b := 
sorry

theorem proposition_B (a b : ℝ) (h : propB_conditions a b) : ¬ propB a b :=
sorry

theorem proposition_C (c a b : ℝ) (h : propC_conditions c a b) : ¬ propC c a b :=
sorry

theorem proposition_D (a b c : ℝ) (h : propD_conditions a b c) : propD a b c :=
sorry

end proposition_A_proposition_B_proposition_C_proposition_D_l68_68702


namespace units_digit_of_power_l68_68716

theorem units_digit_of_power (base : ℕ) (exp : ℕ) (units_base : ℕ) (units_exp_mod : ℕ) :
  (base % 10 = units_base) → (exp % 2 = units_exp_mod) → (units_base = 9) → (units_exp_mod = 0) →
  (base ^ exp % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l68_68716


namespace factor_expression_l68_68987

theorem factor_expression (x : ℝ) : 
  72 * x^2 + 108 * x + 36 = 36 * (2 * x^2 + 3 * x + 1) :=
sorry

end factor_expression_l68_68987


namespace arithmetic_sequence_common_difference_l68_68783

/--
Given an arithmetic sequence $\{a_n\}$ and $S_n$ being the sum of the first $n$ terms, 
with $a_1=1$ and $S_3=9$, prove that the common difference $d$ is equal to $2$.
-/
theorem arithmetic_sequence_common_difference :
  ∃ (d : ℝ), (∀ (n : ℕ), aₙ = 1 + (n - 1) * d) ∧ S₃ = a₁ + (a₁ + d) + (a₁ + 2 * d) ∧ a₁ = 1 ∧ S₃ = 9 → d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l68_68783


namespace arithmetic_seq_sum_l68_68912

theorem arithmetic_seq_sum {a_n : ℕ → ℤ} {d : ℤ} (S_n : ℕ → ℤ) :
  (∀ n : ℕ, S_n n = -(n * n)) →
  (∃ d, d = -2 ∧ ∀ n, a_n n = -2 * n + 1) :=
by
  -- Assuming that S_n is given as per the condition of the problem
  sorry

end arithmetic_seq_sum_l68_68912


namespace price_of_each_sundae_l68_68477

theorem price_of_each_sundae
  (num_ice_cream_bars : ℕ)
  (num_sundaes : ℕ)
  (total_price : ℝ)
  (price_per_ice_cream_bar : ℝ)
  (total_cost_for_sundaes : ℝ) :
  num_ice_cream_bars = 225 →
  num_sundaes = 125 →
  total_price = 200 →
  price_per_ice_cream_bar = 0.60 →
  total_cost_for_sundaes = total_price - (num_ice_cream_bars * price_per_ice_cream_bar) →
  (total_cost_for_sundaes / num_sundaes) = 0.52 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end price_of_each_sundae_l68_68477


namespace total_rooms_count_l68_68546

noncomputable def apartment_area : ℕ := 160
noncomputable def living_room_area : ℕ := 60
noncomputable def other_room_area : ℕ := 20

theorem total_rooms_count (A : apartment_area = 160) (L : living_room_area = 60) (O : other_room_area = 20) :
  1 + (apartment_area - living_room_area) / other_room_area = 6 :=
by
  sorry

end total_rooms_count_l68_68546


namespace final_value_of_A_l68_68135

-- Define the initial value of A
def initial_value (A : ℤ) : Prop := A = 15

-- Define the reassignment condition
def reassignment_cond (A : ℤ) : Prop := A = -A + 5

-- The theorem stating that given the initial value and reassignment condition, the final value of A is -10
theorem final_value_of_A (A : ℤ) (h1 : initial_value A) (h2 : reassignment_cond A) : A = -10 := by
  sorry

end final_value_of_A_l68_68135


namespace final_ratio_of_milk_to_water_l68_68187

-- Initial conditions definitions
def initial_milk_ratio : ℚ := 5 / 8
def initial_water_ratio : ℚ := 3 / 8
def additional_milk : ℚ := 8
def total_capacity : ℚ := 72

-- Final ratio statement
theorem final_ratio_of_milk_to_water :
  (initial_milk_ratio * (total_capacity - additional_milk) + additional_milk) / (initial_water_ratio * (total_capacity - additional_milk)) = 2 := by
  sorry

end final_ratio_of_milk_to_water_l68_68187


namespace length_of_side_of_regular_tetradecagon_l68_68440

theorem length_of_side_of_regular_tetradecagon (P : ℝ) (n : ℕ) (h₀ : n = 14) (h₁ : P = 154) : P / n = 11 := 
by
  sorry

end length_of_side_of_regular_tetradecagon_l68_68440


namespace total_people_present_l68_68839

theorem total_people_present (A B : ℕ) 
  (h1 : 2 * A + B = 10) 
  (h2 : A + 2 * B = 14) :
  A + B = 8 :=
sorry

end total_people_present_l68_68839


namespace problem_statement_l68_68114

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x + 2)

def S : Set ℝ := {y | ∃ x ≥ 0, y = f x}

theorem problem_statement :
  (∀ y ∈ S, y ≤ 2) ∧ (¬ (2 ∈ S)) ∧ (∀ y ∈ S, y ≥ 3 / 2) ∧ (3 / 2 ∈ S) :=
by
  sorry

end problem_statement_l68_68114


namespace sqrt_a_minus_2_meaningful_l68_68727

theorem sqrt_a_minus_2_meaningful (a : ℝ) (h : 0 ≤ a - 2) : 2 ≤ a :=
by
  sorry

end sqrt_a_minus_2_meaningful_l68_68727


namespace speed_of_river_l68_68296

theorem speed_of_river :
  ∃ v : ℝ, 
    (∀ d : ℝ, (2 * d = 9.856) → 
              (d = 4.928) ∧ 
              (1 = (d / (10 - v) + d / (10 + v)))) 
    → v = 1.2 :=
sorry

end speed_of_river_l68_68296


namespace initial_violet_balloons_l68_68691

-- Defining the conditions
def violet_balloons_given_by_tom : ℕ := 16
def violet_balloons_left_with_tom : ℕ := 14

-- The statement to prove
theorem initial_violet_balloons (initial_balloons : ℕ) :
  initial_balloons = violet_balloons_given_by_tom + violet_balloons_left_with_tom :=
sorry

end initial_violet_balloons_l68_68691


namespace shopkeeper_loss_percentages_l68_68542

theorem shopkeeper_loss_percentages 
  (TypeA : Type) (TypeB : Type) (TypeC : Type)
  (theft_percentage_A : ℝ) (theft_percentage_B : ℝ) (theft_percentage_C : ℝ)
  (hA : theft_percentage_A = 0.20)
  (hB : theft_percentage_B = 0.25)
  (hC : theft_percentage_C = 0.30)
  :
  (theft_percentage_A = 0.20 ∧ theft_percentage_B = 0.25 ∧ theft_percentage_C = 0.30) ∧
  ((theft_percentage_A + theft_percentage_B + theft_percentage_C) / 3 = 0.25) :=
by
  sorry

end shopkeeper_loss_percentages_l68_68542


namespace pirate_coins_l68_68349

theorem pirate_coins (x : ℕ) (hn : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 15 → ∃ y : ℕ, y = (2 * k * x) / 15) : 
  ∃ y : ℕ, y = 630630 :=
by sorry

end pirate_coins_l68_68349


namespace sum_of_integers_l68_68309

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 120) : x + y = 15 :=
by {
    sorry
}

end sum_of_integers_l68_68309


namespace find_h_l68_68815

theorem find_h (x : ℝ) : 
  ∃ a k : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - (-3 / 2))^2 + k :=
sorry

end find_h_l68_68815


namespace miles_traveled_correct_l68_68827

def initial_odometer_reading := 212.3
def odometer_reading_at_lunch := 372.0
def miles_traveled := odometer_reading_at_lunch - initial_odometer_reading

theorem miles_traveled_correct : miles_traveled = 159.7 :=
by
  sorry

end miles_traveled_correct_l68_68827


namespace both_questions_correct_l68_68518

def total_students := 100
def first_question_correct := 75
def second_question_correct := 30
def neither_question_correct := 20

theorem both_questions_correct :
  (first_question_correct + second_question_correct - (total_students - neither_question_correct)) = 25 :=
by
  sorry

end both_questions_correct_l68_68518


namespace original_avg_age_is_fifty_l68_68821

-- Definitions based on conditions
variable (N : ℕ) -- original number of students
variable (A : ℕ) -- original average age
variable (new_students : ℕ) -- number of new students
variable (new_avg_age : ℕ) -- average age of new students
variable (decreased_avg_age : ℕ) -- new average age after new students join

-- Conditions given in the problem
def original_avg_age_condition : Prop := A = 50
def new_students_condition : Prop := new_students = 12
def avg_age_new_students_condition : Prop := new_avg_age = 32
def decreased_avg_age_condition : Prop := decreased_avg_age = 46

-- Final Mathematical Equivalent Proof Problem
theorem original_avg_age_is_fifty
  (h1 : original_avg_age_condition A)
  (h2 : new_students_condition new_students)
  (h3 : avg_age_new_students_condition new_avg_age)
  (h4 : decreased_avg_age_condition decreased_avg_age) :
  A = 50 :=
by sorry

end original_avg_age_is_fifty_l68_68821


namespace knight_king_moves_incompatible_l68_68221

-- Definitions for moves and chessboards
structure Board :=
  (numbering : Fin 64 → Nat)
  (different_board : Prop)

def knights_move (x y : Fin 64) : Prop :=
  (abs (x / 8 - y / 8) = 2 ∧ abs (x % 8 - y % 8) = 1) ∨
  (abs (x / 8 - y / 8) = 1 ∧ abs (x % 8 - y % 8) = 2)

def kings_move (x y : Fin 64) : Prop :=
  abs (x / 8 - y / 8) ≤ 1 ∧ abs (x % 8 - y % 8) ≤ 1 ∧ (x ≠ y)

-- Theorem stating the proof problem
theorem knight_king_moves_incompatible (vlad_board gosha_board : Board) (h_board_diff: vlad_board.different_board):
  ¬ ∀ i j : Fin 64, (knights_move i j ↔ kings_move (vlad_board.numbering i) (vlad_board.numbering j)) :=
by {
  -- Skipping proofs with sorry
  sorry
}

end knight_king_moves_incompatible_l68_68221


namespace solve_inequality_l68_68392

theorem solve_inequality (x : ℝ) : 3 * x^2 + 2 * x - 3 > 12 - 2 * x → x < -3 ∨ x > 5 / 3 :=
sorry

end solve_inequality_l68_68392


namespace proof_problem_l68_68289

-- Definitions of propositions p and q
def p (a b : ℝ) : Prop := a < b → ∀ c : ℝ, c ≠ 0 → a * c^2 < b * c^2
def q : Prop := ∃ x₀ > 0, x₀ - 1 + Real.log x₀ = 0

-- Conditions for the problem
variable (a b : ℝ)
variable (p_false : ¬ p a b)
variable (q_true : q)

-- Proving which compound proposition is true
theorem proof_problem : (¬ p a b) ∧ q := by
  exact ⟨p_false, q_true⟩

end proof_problem_l68_68289


namespace distinct_integer_roots_l68_68376

theorem distinct_integer_roots (a : ℤ) : 
  (∃ u v : ℤ, u ≠ v ∧ (u + v = -a) ∧ (u * v = 2 * a)) ↔ a = -1 ∨ a = 9 :=
by
  sorry

end distinct_integer_roots_l68_68376


namespace line_equation_l68_68460

theorem line_equation (m n : ℝ) (p : ℝ) (h : p = 3) :
  ∃ b : ℝ, ∀ x y : ℝ, (y = n + 21) → (x = m + 3) → y = 7 * x + b ∧ b = n - 7 * m :=
by sorry

end line_equation_l68_68460


namespace kimberly_skittles_proof_l68_68532

variable (SkittlesInitial : ℕ) (SkittlesBought : ℕ) (OrangesBought : ℕ)

/-- Kimberly's initial number of Skittles --/
def kimberly_initial_skittles := SkittlesInitial

/-- Skittles Kimberly buys --/
def kimberly_skittles_bought := SkittlesBought

/-- Oranges Kimbery buys (irrelevant for Skittles count) --/
def kimberly_oranges_bought := OrangesBought

/-- Total Skittles Kimberly has --/
def kimberly_total_skittles (SkittlesInitial SkittlesBought : ℕ) : ℕ :=
  SkittlesInitial + SkittlesBought

/-- Proof statement --/
theorem kimberly_skittles_proof (h1 : SkittlesInitial = 5) (h2 : SkittlesBought = 7) : 
  kimberly_total_skittles SkittlesInitial SkittlesBought = 12 :=
by
  rw [h1, h2]
  exact rfl

end kimberly_skittles_proof_l68_68532


namespace cost_difference_l68_68425

theorem cost_difference (S : ℕ) (h1 : 15 + S = 24) : 15 - S = 6 :=
by
  sorry

end cost_difference_l68_68425


namespace paving_stone_width_l68_68451

theorem paving_stone_width 
    (length_courtyard : ℝ)
    (width_courtyard : ℝ)
    (length_paving_stone : ℝ)
    (num_paving_stones : ℕ)
    (total_area_courtyard : ℝ)
    (total_area_paving_stones : ℝ)
    (width_paving_stone : ℝ)
    (h1 : length_courtyard = 20)
    (h2 : width_courtyard = 16.5)
    (h3 : length_paving_stone = 2.5)
    (h4 : num_paving_stones = 66)
    (h5 : total_area_courtyard = length_courtyard * width_courtyard)
    (h6 : total_area_paving_stones = num_paving_stones * (length_paving_stone * width_paving_stone))
    (h7 : total_area_courtyard = total_area_paving_stones) :
    width_paving_stone = 2 :=
by
  sorry

end paving_stone_width_l68_68451


namespace option_d_true_l68_68134

theorem option_d_true (r p q : ℝ) (hr : r > 0) (hpq : p * q ≠ 0) (hpr_qr : p * r < q * r) : 1 > q / p :=
sorry

end option_d_true_l68_68134


namespace modulo_sum_of_99_plus_5_l68_68780

theorem modulo_sum_of_99_plus_5 : let s_n := (99 / 2) * (2 * 1 + (99 - 1) * 1)
                                 let sum_with_5 := s_n + 5
                                 sum_with_5 % 7 = 6 :=
by
  sorry

end modulo_sum_of_99_plus_5_l68_68780


namespace molecular_weight_of_compound_l68_68795

-- Define the atomic weights for Hydrogen, Chlorine, and Oxygen
def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cl : ℝ := 35.45
def atomic_weight_O : ℝ := 15.999

-- Define the molecular weight of the compound
def molecular_weight (H_weight : ℝ) (Cl_weight : ℝ) (O_weight : ℝ) : ℝ :=
  H_weight + Cl_weight + 2 * O_weight

-- The proof problem statement
theorem molecular_weight_of_compound :
  molecular_weight atomic_weight_H atomic_weight_Cl atomic_weight_O = 68.456 :=
sorry

end molecular_weight_of_compound_l68_68795


namespace molecular_weight_of_compound_l68_68323

theorem molecular_weight_of_compound (C H O n : ℕ) 
    (atomic_weight_C : ℝ) (atomic_weight_H : ℝ) (atomic_weight_O : ℝ) 
    (total_weight : ℝ) 
    (h_C : C = 2) (h_H : H = 4) 
    (h_atomic_weight_C : atomic_weight_C = 12.01) 
    (h_atomic_weight_H : atomic_weight_H = 1.008) 
    (h_atomic_weight_O : atomic_weight_O = 16.00) 
    (h_total_weight : total_weight = 60) : 
    C * atomic_weight_C + H * atomic_weight_H + n * atomic_weight_O = total_weight → 
    n = 2 := 
sorry

end molecular_weight_of_compound_l68_68323


namespace no_natural_m_n_prime_l68_68638

theorem no_natural_m_n_prime (m n : ℕ) : ¬Prime (n^2 + 2018 * m * n + 2019 * m + n - 2019 * m^2) :=
by
  sorry

end no_natural_m_n_prime_l68_68638


namespace solution_set_of_inequality_l68_68688

open Real Set

noncomputable def f (x : ℝ) : ℝ := exp (-x) - exp x - 5 * x

theorem solution_set_of_inequality :
  { x : ℝ | f (x ^ 2) + f (-x - 6) < 0 } = Iio (-2) ∪ Ioi 3 :=
by
  sorry

end solution_set_of_inequality_l68_68688


namespace smallest_fraction_division_l68_68845

theorem smallest_fraction_division (a b : ℕ) (h_coprime : Nat.gcd a b = 1) 
(h1 : ∃ n, (25 * a = n * 21 * b)) (h2 : ∃ m, (15 * a = m * 14 * b)) : (a = 42) ∧ (b = 5) := 
sorry

end smallest_fraction_division_l68_68845


namespace cube_volume_surface_area_l68_68683

variable (x : ℝ)

theorem cube_volume_surface_area (h1 : s^3 = 8 * x) (h2 : 6 * s^2 = 2 * x) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_l68_68683


namespace geometric_sequence_tenth_term_l68_68956

theorem geometric_sequence_tenth_term :
  let a := 5
  let r := 3 / 2
  let a_n (n : ℕ) := a * r ^ (n - 1)
  a_n 10 = 98415 / 512 :=
by
  sorry

end geometric_sequence_tenth_term_l68_68956


namespace colombian_coffee_amount_l68_68204

theorem colombian_coffee_amount 
  (C B : ℝ) 
  (h1 : C + B = 100)
  (h2 : 8.75 * C + 3.75 * B = 635) :
  C = 52 := 
sorry

end colombian_coffee_amount_l68_68204


namespace triangle_equilateral_l68_68075

-- Assume we are given side lengths a, b, and c of a triangle and angles A, B, and C in radians.
variables {a b c : ℝ} {A B C : ℝ}

-- We'll use the assumption that (a + b + c) * (b + c - a) = 3 * b * c and sin A = 2 * sin B * cos C.
axiom triangle_condition1 : (a + b + c) * (b + c - a) = 3 * b * c
axiom triangle_condition2 : Real.sin A = 2 * Real.sin B * Real.cos C

-- We need to prove that the triangle is equilateral.
theorem triangle_equilateral : (a = b) ∧ (b = c) ∧ (c = a) := by
  sorry

end triangle_equilateral_l68_68075


namespace gcd_combination_l68_68885

theorem gcd_combination (a b d : ℕ) (h : d = Nat.gcd a b) : 
  Nat.gcd (5 * a + 3 * b) (13 * a + 8 * b) = d := 
by
  sorry

end gcd_combination_l68_68885


namespace Douglas_won_72_percent_of_votes_in_county_X_l68_68036

/-- Definition of the problem conditions and the goal -/
theorem Douglas_won_72_percent_of_votes_in_county_X
  (V : ℝ)
  (total_votes_ratio : ∀ county_X county_Y, county_X = 2 * county_Y)
  (total_votes_percentage_both_counties : 0.60 = (1.8 * V) / (2 * V + V))
  (votes_percentage_county_Y : 0.36 = (0.36 * V) / V) : 
  ∃ P : ℝ, P = 72 ∧ P = (1.44 * V) / (2 * V) * 100 :=
sorry

end Douglas_won_72_percent_of_votes_in_county_X_l68_68036


namespace total_houses_l68_68957

theorem total_houses (houses_one_side : ℕ) (houses_other_side : ℕ) (h1 : houses_one_side = 40) (h2 : houses_other_side = 3 * houses_one_side) : houses_one_side + houses_other_side = 160 :=
by sorry

end total_houses_l68_68957


namespace gamma_distribution_moments_l68_68120

noncomputable def gamma_density (α β x : ℝ) : ℝ :=
  (1 / (β ^ (α + 1) * Real.Gamma (α + 1))) * x ^ α * Real.exp (-x / β)

open Real

theorem gamma_distribution_moments (α β : ℝ) (x_bar D_B : ℝ) (hα : α > -1) (hβ : β > 0) :
  α = x_bar ^ 2 / D_B - 1 ∧ β = D_B / x_bar :=
by
  sorry

end gamma_distribution_moments_l68_68120


namespace solve_equation_l68_68254

theorem solve_equation (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x + 3 = 18 :=
sorry

end solve_equation_l68_68254


namespace basketball_free_throws_l68_68923

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a) 
  (h2 : x = b) 
  (h3 : 2 * a + 3 * b + x = 73) : 
  x = 10 := 
by 
  sorry -- The actual proof is omitted as per the requirements.

end basketball_free_throws_l68_68923


namespace average_of_remaining_two_l68_68566

theorem average_of_remaining_two (a1 a2 a3 a4 a5 a6 : ℝ)
    (h_avg6 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 3.95)
    (h_avg2_1 : (a1 + a2) / 2 = 3.4)
    (h_avg2_2 : (a3 + a4) / 2 = 3.85) :
    (a5 + a6) / 2 = 4.6 := 
sorry

end average_of_remaining_two_l68_68566


namespace fuel_tank_capacity_l68_68862

theorem fuel_tank_capacity
  (ethanol_A_ethanol : ∀ {x : Float}, x = 0.12 * 49.99999999999999)
  (ethanol_B_ethanol : ∀ {C : Float}, x = 0.16 * (C - 49.99999999999999))
  (total_ethanol : ∀ {C : Float}, 0.12 * 49.99999999999999 + 0.16 * (C - 49.99999999999999) = 30) :
  (C = 162.5) :=
sorry

end fuel_tank_capacity_l68_68862


namespace relationship_y1_y2_y3_l68_68448

variable (y1 y2 y3 : ℝ)

def quadratic_function (x : ℝ) : ℝ := -x^2 + 4 * x - 5

theorem relationship_y1_y2_y3
  (h1 : quadratic_function (-4) = y1)
  (h2 : quadratic_function (-3) = y2)
  (h3 : quadratic_function (1) = y3) :
  y1 < y2 ∧ y2 < y3 :=
sorry

end relationship_y1_y2_y3_l68_68448


namespace avg_height_is_28_l68_68154

-- Define the height relationship between trees
def height_relation (a b : ℕ) := a = 2 * b ∨ a = b / 2

-- Given tree heights (partial information)
def height_tree_2 := 14
def height_tree_5 := 20

-- Define the tree heights variables
variables (height_tree_1 height_tree_3 height_tree_4 height_tree_6 : ℕ)

-- Conditions based on the given data and height relations
axiom h1 : height_relation height_tree_1 height_tree_2
axiom h2 : height_relation height_tree_2 height_tree_3
axiom h3 : height_relation height_tree_3 height_tree_4
axiom h4 : height_relation height_tree_4 height_tree_5
axiom h5 : height_relation height_tree_5 height_tree_6

-- Compute total and average height
def total_height := height_tree_1 + height_tree_2 + height_tree_3 + height_tree_4 + height_tree_5 + height_tree_6
def average_height := total_height / 6

-- Prove the average height is 28 meters
theorem avg_height_is_28 : average_height = 28 := by
  sorry

end avg_height_is_28_l68_68154


namespace P_gt_Q_l68_68042

variable (x : ℝ)

def P := x^2 + 2
def Q := 2 * x

theorem P_gt_Q : P x > Q x := by
  sorry

end P_gt_Q_l68_68042


namespace complete_square_l68_68772

theorem complete_square (x : ℝ) : (x^2 - 2 * x - 2 = 0) → ((x - 1)^2 = 3) :=
by
  intro h
  sorry

end complete_square_l68_68772


namespace f_m_plus_1_positive_l68_68508

def f (a x : ℝ) := x^2 + x + a

theorem f_m_plus_1_positive (a m : ℝ) (ha : a > 0) (hm : f a m < 0) : f a (m + 1) > 0 := 
  sorry

end f_m_plus_1_positive_l68_68508


namespace y_squared_in_range_l68_68370

theorem y_squared_in_range (y : ℝ) 
  (h : (Real.sqrt (Real.sqrt (y + 16)) - Real.sqrt (Real.sqrt (y - 16)) = 2)) :
  270 ≤ y^2 ∧ y^2 ≤ 280 :=
sorry

end y_squared_in_range_l68_68370


namespace range_of_a_l68_68366

noncomputable def A : Set ℝ := { x : ℝ | x > 5 }
noncomputable def B (a : ℝ) : Set ℝ := { x : ℝ | x > a }

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a < 5 :=
  sorry

end range_of_a_l68_68366


namespace divisibility_by_3_l68_68564

theorem divisibility_by_3 (x y z : ℤ) (h : x^3 + y^3 = z^3) : 3 ∣ x ∨ 3 ∣ y ∨ 3 ∣ z := 
sorry

end divisibility_by_3_l68_68564


namespace tickets_left_l68_68939

theorem tickets_left (initial_tickets used_tickets tickets_left : ℕ) 
  (h1 : initial_tickets = 127) 
  (h2 : used_tickets = 84) : 
  tickets_left = initial_tickets - used_tickets := 
by
  sorry

end tickets_left_l68_68939


namespace min_value_three_l68_68876

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (1 / ((1 - x) * (1 - y) * (1 - z))) +
  (1 / ((1 + x) * (1 + y) * (1 + z))) +
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)))

theorem min_value_three (x y z : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) :
  min_value_expression x y z = 3 :=
by
  sorry

end min_value_three_l68_68876


namespace andrew_vacation_days_l68_68169

theorem andrew_vacation_days (days_worked last_year vacation_per_10 worked_days in_march in_september : ℕ)
  (h1 : vacation_per_10 = 10)
  (h2 : days_worked_last_year = 300)
  (h3 : worked_days = days_worked_last_year / vacation_per_10)
  (h4 : in_march = 5)
  (h5 : in_september = 2 * in_march)
  (h6 : days_taken = in_march + in_september)
  (h7 : vacation_days_remaining = worked_days - days_taken) :
  vacation_days_remaining = 15 :=
by
  sorry

end andrew_vacation_days_l68_68169


namespace side_length_of_largest_square_l68_68520

theorem side_length_of_largest_square (A_cross : ℝ) (s : ℝ)
  (h1 : A_cross = 810) : s = 36 :=
  have h_large_squares : 2 * (s / 2)^2 = s^2 / 2 := by sorry
  have h_small_squares : 2 * (s / 4)^2 = s^2 / 8 := by sorry
  have h_combined_area : s^2 / 2 + s^2 / 8 = 810 := by sorry
  have h_final : 5 * s^2 / 8 = 810 := by sorry
  have h_s2 : s^2 = 1296 := by sorry
  have h_s : s = 36 := by sorry
  h_s

end side_length_of_largest_square_l68_68520


namespace carlo_practice_difference_l68_68681

-- Definitions for given conditions
def monday_practice (T : ℕ) : ℕ := 2 * T
def tuesday_practice (T : ℕ) : ℕ := T
def wednesday_practice (thursday_minutes : ℕ) : ℕ := thursday_minutes + 5
def thursday_practice : ℕ := 50
def friday_practice : ℕ := 60
def total_weekly_practice : ℕ := 300

theorem carlo_practice_difference 
  (T : ℕ) 
  (Monday Tuesday Wednesday Thursday Friday : ℕ) 
  (H1 : Monday = monday_practice T)
  (H2 : Tuesday = tuesday_practice T)
  (H3 : Wednesday = wednesday_practice Thursday)
  (H4 : Thursday = thursday_practice)
  (H5 : Friday = friday_practice)
  (H6 : Monday + Tuesday + Wednesday + Thursday + Friday = total_weekly_practice) :
  (Wednesday - Tuesday = 10) :=
by 
  -- Use the provided conditions and derive the required result.
  sorry

end carlo_practice_difference_l68_68681


namespace father_age_l68_68298

variable (F C1 C2 : ℕ)

theorem father_age (h1 : F = 3 * (C1 + C2))
  (h2 : F + 5 = 2 * (C1 + 5 + C2 + 5)) :
  F = 45 := by
  sorry

end father_age_l68_68298


namespace evaluate_expression_l68_68609

theorem evaluate_expression :
  (2:ℝ) ^ ((0:ℝ) ^ (Real.sin (Real.pi / 2)) ^ 2) + ((3:ℝ) ^ 0) ^ 1 ^ 4 = 2 := by
  -- Given conditions
  have h1 : Real.sin (Real.pi / 2) = 1 := by sorry
  have h2 : (3:ℝ) ^ 0 = 1 := by sorry
  have h3 : (0:ℝ) ^ 1 = 0 := by sorry
  -- Proof omitted
  sorry

end evaluate_expression_l68_68609


namespace min_guests_l68_68220

theorem min_guests (total_food : ℕ) (max_food : ℝ) 
  (H1 : total_food = 337) 
  (H2 : max_food = 2) : 
  ∃ n : ℕ, n = ⌈total_food / max_food⌉ ∧ n = 169 :=
by
  sorry

end min_guests_l68_68220


namespace clarissa_copies_needed_l68_68519

-- Define the given conditions
def manuscript_pages : ℕ := 400
def cost_per_page : ℚ := 0.05
def cost_per_binding : ℚ := 5.00
def total_cost : ℚ := 250.00

-- Calculate the total cost for one manuscript
def cost_per_copy_and_bind : ℚ := cost_per_page * manuscript_pages + cost_per_binding

-- Define number of copies needed
def number_of_copies_needed : ℚ := total_cost / cost_per_copy_and_bind

-- Prove number of copies needed is 10
theorem clarissa_copies_needed : number_of_copies_needed = 10 := 
by 
  -- Implementing the proof steps would go here
  sorry

end clarissa_copies_needed_l68_68519


namespace number_of_legs_twice_heads_diff_eq_22_l68_68176

theorem number_of_legs_twice_heads_diff_eq_22 (P H : ℕ) (L : ℤ) (Heads : ℕ) (X : ℤ) (h1 : P = 11)
  (h2 : L = 4 * P + 2 * H) (h3 : Heads = P + H) (h4 : L = 2 * Heads + X) : X = 22 :=
by
  sorry

end number_of_legs_twice_heads_diff_eq_22_l68_68176


namespace distance_walked_is_4_point_6_l68_68599

-- Define the number of blocks Sarah walked in each direction
def blocks_west : ℕ := 8
def blocks_south : ℕ := 15

-- Define the length of each block in miles
def block_length : ℚ := 1 / 5

-- Calculate the total number of blocks
def total_blocks : ℕ := blocks_west + blocks_south

-- Calculate the total distance walked in miles
def total_distance_walked : ℚ := total_blocks * block_length

-- Statement to prove the total distance walked is 4.6 miles
theorem distance_walked_is_4_point_6 : total_distance_walked = 4.6 := sorry

end distance_walked_is_4_point_6_l68_68599


namespace tetrahedron_circumsphere_surface_area_eq_five_pi_l68_68919

noncomputable def rectangle_diagonal (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

noncomputable def circumscribed_sphere_radius (a b : ℝ) : ℝ :=
  rectangle_diagonal a b / 2

noncomputable def circumscribed_sphere_surface_area (a b : ℝ) : ℝ :=
  4 * Real.pi * (circumscribed_sphere_radius a b)^2

theorem tetrahedron_circumsphere_surface_area_eq_five_pi :
  circumscribed_sphere_surface_area 2 1 = 5 * Real.pi := by
  sorry

end tetrahedron_circumsphere_surface_area_eq_five_pi_l68_68919


namespace problem_l68_68293

theorem problem (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -6) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = -150 := 
by sorry

end problem_l68_68293


namespace find_s_t_l68_68647

noncomputable def problem_constants (a b c : ℝ) : Prop :=
  (a^3 + 3 * a^2 + 4 * a - 11 = 0) ∧
  (b^3 + 3 * b^2 + 4 * b - 11 = 0) ∧
  (c^3 + 3 * c^2 + 4 * c - 11 = 0)

theorem find_s_t (a b c s t : ℝ) (h1 : problem_constants a b c) (h2 : (a + b) * (b + c) * (c + a) = -t)
  (h3 : (a + b) * (b + c) + (b + c) * (c + a) + (c + a) * (a + b) = s) :
s = 8 ∧ t = 23 :=
sorry

end find_s_t_l68_68647


namespace car_new_speed_l68_68413

theorem car_new_speed (original_speed : ℝ) (supercharge_percent : ℝ) (weight_cut_speed_increase : ℝ) :
  original_speed = 150 → supercharge_percent = 0.30 → weight_cut_speed_increase = 10 → 
  original_speed * (1 + supercharge_percent) + weight_cut_speed_increase = 205 :=
by
  intros h_orig h_supercharge h_weight
  rw [h_orig, h_supercharge]
  sorry

end car_new_speed_l68_68413


namespace opposite_of_neg_three_fifths_l68_68316

theorem opposite_of_neg_three_fifths :
  -(-3 / 5) = 3 / 5 :=
by
  sorry

end opposite_of_neg_three_fifths_l68_68316


namespace correct_sampling_methods_l68_68775

/-- 
Given:
1. A group of 500 senior year students with the following blood type distribution: 200 with blood type O,
125 with blood type A, 125 with blood type B, and 50 with blood type AB.
2. A task to select a sample of 20 students to study the relationship between blood type and color blindness.
3. A high school soccer team consisting of 11 players, and the need to draw 2 players to investigate their study load.
4. Sampling methods: I. Random sampling, II. Systematic sampling, III. Stratified sampling.

Prove:
The correct sampling methods are: Stratified sampling (III) for the blood type-color blindness study and
Random sampling (I) for the soccer team study.
-/ 

theorem correct_sampling_methods (students : Finset ℕ) (blood_type_O blood_type_A blood_type_B blood_type_AB : ℕ)
  (sample_size_students soccer_team_size draw_size_soccer_team : ℕ)
  (sampling_methods : Finset ℕ) : 
  (students.card = 500) →
  (blood_type_O = 200) →
  (blood_type_A = 125) →
  (blood_type_B = 125) →
  (blood_type_AB = 50) →
  (sample_size_students = 20) →
  (soccer_team_size = 11) →
  (draw_size_soccer_team = 2) →
  (sampling_methods = {1, 2, 3}) →
  (s = (3, 1)) :=
by
  sorry

end correct_sampling_methods_l68_68775


namespace find_number_l68_68118

theorem find_number (x : ℝ) (h : x / 5 + 23 = 42) : x = 95 :=
by
  -- Proof placeholder
  sorry

end find_number_l68_68118


namespace xyz_value_l68_68938

-- Define the basic conditions
variables (x y z : ℝ)
variables (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
variables (h1 : x * y = 40 * (4:ℝ)^(1/3))
variables (h2 : x * z = 56 * (4:ℝ)^(1/3))
variables (h3 : y * z = 32 * (4:ℝ)^(1/3))
variables (h4 : x + y = 18)

-- The target theorem
theorem xyz_value : x * y * z = 16 * (895:ℝ)^(1/2) :=
by
  -- Here goes the proof, but we add 'sorry' to end the theorem placeholder
  sorry

end xyz_value_l68_68938


namespace hiker_walking_speed_l68_68178

theorem hiker_walking_speed (v : ℝ) :
  (∃ (hiker_shares_cyclist_distance : 20 / 60 * v = 25 * (5 / 60)), v = 6.25) :=
by
  sorry

end hiker_walking_speed_l68_68178


namespace ElaCollected13Pounds_l68_68419

def KimberleyCollection : ℕ := 10
def HoustonCollection : ℕ := 12
def TotalCollection : ℕ := 35

def ElaCollection : ℕ := TotalCollection - KimberleyCollection - HoustonCollection

theorem ElaCollected13Pounds : ElaCollection = 13 := sorry

end ElaCollected13Pounds_l68_68419


namespace shaded_square_percentage_l68_68274

theorem shaded_square_percentage (total_squares shaded_squares : ℕ) (h1 : total_squares = 16) (h2 : shaded_squares = 8) : 
  (shaded_squares : ℚ) / total_squares * 100 = 50 :=
by
  sorry

end shaded_square_percentage_l68_68274


namespace solution_set_inequality_l68_68422

theorem solution_set_inequality 
  (a b : ℝ)
  (h1 : ∀ x, a * x^2 + b * x + 3 > 0 ↔ -1 < x ∧ x < 1/2) :
  ((-1:ℝ) < x ∧ x < 2) ↔ 3 * x^2 + b * x + a < 0 :=
by 
  -- Write the proof here
  sorry

end solution_set_inequality_l68_68422


namespace girls_collected_more_mushrooms_l68_68720

variables (N I A V : ℝ)

theorem girls_collected_more_mushrooms 
    (h1 : N > I) 
    (h2 : N > A) 
    (h3 : N > V) 
    (h4 : I ≤ N) 
    (h5 : I ≤ A) 
    (h6 : I ≤ V) 
    (h7 : A > V) : 
    N + I > A + V := 
by {
    sorry
}

end girls_collected_more_mushrooms_l68_68720


namespace problem_statement_l68_68330

theorem problem_statement (a b c : ℝ) (h₀ : 4 * a - 4 * b + c > 0) (h₁ : a + 2 * b + c < 0) : b^2 > a * c :=
sorry

end problem_statement_l68_68330


namespace function_passes_through_fixed_point_l68_68026

variable (a : ℝ)

theorem function_passes_through_fixed_point (h1 : a > 0) (h2 : a ≠ 1) : (1, 2) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1) + 1)} :=
by
  sorry

end function_passes_through_fixed_point_l68_68026


namespace solve_system_l68_68708

noncomputable def solution1 (a b : ℝ) : ℝ × ℝ := 
  ((a + Real.sqrt (a^2 + 4 * b)) / 2, (-a + Real.sqrt (a^2 + 4 * b)) / 2)

noncomputable def solution2 (a b : ℝ) : ℝ × ℝ := 
  ((a - Real.sqrt (a^2 + 4 * b)) / 2, (-a - Real.sqrt (a^2 + 4 * b)) / 2)

theorem solve_system (a b x y : ℝ) : 
  (x - y = a ∧ x * y = b) ↔ ((x, y) = solution1 a b ∨ (x, y) = solution2 a b) := 
by sorry

end solve_system_l68_68708


namespace area_of_region_l68_68423

theorem area_of_region :
  ∀ (x y : ℝ), (|2 * x - 2| + |3 * y - 3| ≤ 30) → (area_of_figure = 300) :=
sorry

end area_of_region_l68_68423


namespace average_growth_rate_l68_68752

theorem average_growth_rate (x : ℝ) (hx : (1 + x)^2 = 1.44) : x < 0.22 :=
sorry

end average_growth_rate_l68_68752


namespace math_problem_l68_68450

noncomputable def proof_problem (a b c : ℝ) : Prop :=
  (0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) →
  (1 / (a * (a + 1)) + 1 / (b * (b + 1)) + 1 / (c * (c + 1)) ≥  3 / 2)

theorem math_problem (a b c : ℝ) :
  proof_problem a b c :=
by
  sorry

end math_problem_l68_68450


namespace product_of_two_digit_numbers_is_not_five_digits_l68_68786

theorem product_of_two_digit_numbers_is_not_five_digits :
  ∀ (a b c d : ℕ), (10 ≤ 10 * a + b) → (10 * a + b ≤ 99) → (10 ≤ 10 * c + d) → (10 * c + d ≤ 99) → 
    (10 * a + b) * (10 * c + d) < 10000 :=
by
  intros a b c d H1 H2 H3 H4
  -- proof steps would go here
  sorry

end product_of_two_digit_numbers_is_not_five_digits_l68_68786


namespace alix_more_chocolates_than_nick_l68_68226

theorem alix_more_chocolates_than_nick :
  let nick_chocolates := 10
  let initial_alix_chocolates := 3 * nick_chocolates
  let after_mom_took_chocolates := initial_alix_chocolates - 5
  after_mom_took_chocolates - nick_chocolates = 15 := by
sorry

end alix_more_chocolates_than_nick_l68_68226


namespace arrange_3x3_grid_l68_68629

-- Define the problem conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := ¬ is_odd n

-- Define the function to count the number of such arrangements
noncomputable def count_arrangements : ℕ :=
  6 * 3^6 * 4^3 + 9 * 3^4 * 4^5 + 4^9

-- State the main theorem
theorem arrange_3x3_grid (nums : ℕ → Prop) (table : ℕ → ℕ → ℕ) (h : ∀ i j, 1 ≤ table i j ∧ table i j ≤ 7) :
  (∀ i, is_odd (table i 0 + table i 1 + table i 2)) ∧ (∀ j, is_odd (table 0 j + table 1 j + table 2 j)) →
  count_arrangements = 6 * 3^6 * 4^3 + 9 * 3^4 * 4^5 + 4^9 :=
by sorry

end arrange_3x3_grid_l68_68629


namespace dan_money_left_l68_68706

def money_left (initial_amount spent_on_candy spent_on_gum : ℝ) : ℝ :=
  initial_amount - (spent_on_candy + spent_on_gum)

theorem dan_money_left :
  money_left 3.75 1.25 0.80 = 1.70 :=
by
  sorry

end dan_money_left_l68_68706


namespace at_least_three_bushes_with_same_number_of_flowers_l68_68555

-- Defining the problem using conditions as definitions.
theorem at_least_three_bushes_with_same_number_of_flowers (n : ℕ) (f : Fin n → ℕ) (h1 : n = 201)
  (h2 : ∀ (i : Fin n), 1 ≤ f i ∧ f i ≤ 100) : 
  ∃ (x : ℕ), (∃ (i1 i2 i3 : Fin n), i1 ≠ i2 ∧ i1 ≠ i3 ∧ i2 ≠ i3 ∧ f i1 = x ∧ f i2 = x ∧ f i3 = x) := 
by
  sorry

end at_least_three_bushes_with_same_number_of_flowers_l68_68555


namespace minimum_value_of_f_l68_68867

def f (x : ℝ) : ℝ := |3 - x| + |x - 7|

theorem minimum_value_of_f : ∃ x : ℝ, f x = 4 ∧ ∀ y : ℝ, f y ≥ 4 :=
by {
  sorry
}

end minimum_value_of_f_l68_68867


namespace sequence_formula_l68_68865

theorem sequence_formula (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + n + 1) :
  (a 1 = 3) ∧ (∀ n, n ≥ 2 → a n = 2 * n) :=
by
  sorry

end sequence_formula_l68_68865


namespace price_of_first_shirt_l68_68022

theorem price_of_first_shirt
  (price1 price2 price3 : ℕ)
  (total_shirts : ℕ)
  (min_avg_price_of_remaining : ℕ)
  (total_avg_price_of_all : ℕ)
  (prices_of_first_3 : price1 = 100 ∧ price2 = 90 ∧ price3 = 82)
  (condition1 : total_shirts = 10)
  (condition2 : min_avg_price_of_remaining = 104)
  (condition3 : total_avg_price_of_all > 100) :
  price1 = 100 :=
by
  sorry

end price_of_first_shirt_l68_68022


namespace triangle_area_calculation_l68_68514

theorem triangle_area_calculation
  (A : ℕ)
  (BC : ℕ)
  (h : ℕ)
  (nine_parallel_lines : Bool)
  (equal_segments : Bool)
  (largest_area_part : ℕ)
  (largest_part_condition : largest_area_part = 38) :
  9 * (BC / 10) * (h / 10) / 2 = 10 * (BC / 2) * A / 19 :=
sorry

end triangle_area_calculation_l68_68514


namespace ratio_of_third_to_second_building_l68_68554

/-
The tallest building in the world is 100 feet tall. The second tallest is half that tall, the third tallest is some 
fraction of the second tallest building's height, and the fourth is one-fifth as tall as the third. All 4 buildings 
put together are 180 feet tall. What is the ratio of the height of the third tallest building to the second tallest building?

Given H1 = 100, H2 = (1 / 2) * H1, H4 = (1 / 5) * H3, 
and H1 + H2 + H3 + H4 = 180, prove that H3 / H2 = 1 / 2.
-/

theorem ratio_of_third_to_second_building :
  ∀ (H1 H2 H3 H4 : ℝ),
  H1 = 100 →
  H2 = (1 / 2) * H1 →
  H4 = (1 / 5) * H3 →
  H1 + H2 + H3 + H4 = 180 →
  (H3 / H2) = (1 / 2) :=
by
  intros H1 H2 H3 H4 h1_eq h2_half_h1 h4_fifth_h3 total_eq
  /- proof steps go here -/
  sorry

end ratio_of_third_to_second_building_l68_68554


namespace manuscript_copy_cost_l68_68580

theorem manuscript_copy_cost (total_cost : ℝ) (binding_cost : ℝ) (num_manuscripts : ℕ) (pages_per_manuscript : ℕ) (x : ℝ) :
  total_cost = 250 ∧ binding_cost = 5 ∧ num_manuscripts = 10 ∧ pages_per_manuscript = 400 →
  x = (total_cost - binding_cost * num_manuscripts) / (num_manuscripts * pages_per_manuscript) →
  x = 0.05 :=
by
  sorry

end manuscript_copy_cost_l68_68580


namespace students_in_class_C_l68_68159

theorem students_in_class_C 
    (total_students : ℕ := 80) 
    (percent_class_A : ℕ := 40) 
    (class_B_difference : ℕ := 21) 
    (h_percent : percent_class_A = 40) 
    (h_class_B_diff : class_B_difference = 21) 
    (h_total_students : total_students = 80) : 
    total_students - ((percent_class_A * total_students) / 100 - class_B_difference + (percent_class_A * total_students) / 100) = 37 := by
    sorry

end students_in_class_C_l68_68159


namespace sequence_x_2022_l68_68608

theorem sequence_x_2022 :
  ∃ (x : ℕ → ℤ), x 1 = 1 ∧ x 2 = 1 ∧ x 3 = -1 ∧
  (∀ n, 4 ≤ n → x n = x (n-1) * x (n-3)) ∧ x 2022 = 1 := by
  sorry

end sequence_x_2022_l68_68608


namespace conditions_for_star_commute_l68_68336

-- Define the operation star
def star (a b : ℝ) : ℝ := a^3 * b^2 - a * b^3

-- Theorem stating the equivalence
theorem conditions_for_star_commute :
  ∀ (x y : ℝ), (star x y = star y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
sorry

end conditions_for_star_commute_l68_68336


namespace arithmetic_sequence_sum_l68_68713

noncomputable def isArithmeticSeq (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_first_n (a : ℕ → ℤ) (n : ℕ) := (n + 1) * (a 0 + a n) / 2

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h_legal_seq : isArithmeticSeq a) (h_sum : sum_first_n a 9 = 120) : 
  a 1 + a 8 = 24 := by
  sorry

end arithmetic_sequence_sum_l68_68713


namespace center_of_symmetry_l68_68063

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * Real.tan (-7 * x + (Real.pi / 3))

theorem center_of_symmetry : f (Real.pi / 21) = 0 :=
by
  -- Mathematical proof goes here, skipping with sorry.
  sorry

end center_of_symmetry_l68_68063


namespace largest_possible_3_digit_sum_l68_68773

theorem largest_possible_3_digit_sum (X Y Z : ℕ) (h_diff : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) 
(h_digit_X : 0 ≤ X ∧ X ≤ 9) (h_digit_Y : 0 ≤ Y ∧ Y ≤ 9) (h_digit_Z : 0 ≤ Z ∧ Z ≤ 9) :
  (100 * X + 10 * X + X) + (10 * Y + X) + X = 994 → (X, Y, Z) = (8, 9, 0) := by
  sorry

end largest_possible_3_digit_sum_l68_68773


namespace determine_a_value_l68_68605

theorem determine_a_value (a : ℤ) (h : ∀ x : ℝ, x^2 + 2 * (a:ℝ) * x + 1 > 0) : a = 0 := 
sorry

end determine_a_value_l68_68605


namespace contestant_wins_probability_l68_68836

section RadioProgramQuiz
  -- Defining the conditions
  def number_of_questions : ℕ := 4
  def number_of_choices_per_question : ℕ := 3
  def probability_of_correct_answer : ℚ := 1 / 3
  
  -- Defining the target probability
  def winning_probability : ℚ := 1 / 9

  -- The theorem
  theorem contestant_wins_probability :
    (let p := probability_of_correct_answer
     let p_correct_all := p^4
     let p_correct_three :=
       4 * (p^3 * (1 - p))
     p_correct_all + p_correct_three = winning_probability) :=
    sorry
end RadioProgramQuiz

end contestant_wins_probability_l68_68836


namespace Amy_crumbs_l68_68015

variable (z : ℕ)

theorem Amy_crumbs (T C : ℕ) (h1 : T * C = z)
  (h2 : ∃ T_A : ℕ, T_A = 2 * T)
  (h3 : ∃ C_A : ℕ, C_A = (3 * C) / 2) :
  ∃ z_A : ℕ, z_A = 3 * z :=
by
  sorry

end Amy_crumbs_l68_68015


namespace smallest_m_l68_68299

-- Let n be a positive integer and r be a positive real number less than 1/5000
def valid_r (r : ℝ) : Prop := 0 < r ∧ r < 1 / 5000

def m (n : ℕ) (r : ℝ) := (n + r)^3

theorem smallest_m : (∃ (n : ℕ) (r : ℝ), valid_r r ∧ n ≥ 41 ∧ m n r = 68922) :=
by
  sorry

end smallest_m_l68_68299


namespace shared_earnings_eq_27_l68_68491

theorem shared_earnings_eq_27
    (shoes_pairs : ℤ) (shoes_cost : ℤ) (shirts : ℤ) (shirts_cost : ℤ)
    (h1 : shoes_pairs = 6) (h2 : shoes_cost = 3)
    (h3 : shirts = 18) (h4 : shirts_cost = 2) :
    (shoes_pairs * shoes_cost + shirts * shirts_cost) / 2 = 27 := by
  sorry

end shared_earnings_eq_27_l68_68491


namespace johnny_marbles_combination_l68_68018

theorem johnny_marbles_combination : @Nat.choose 9 4 = 126 := by
  sorry

end johnny_marbles_combination_l68_68018


namespace total_wheels_l68_68061

theorem total_wheels (n_bicycles n_tricycles n_unicycles n_four_wheelers : ℕ)
                     (w_bicycle w_tricycle w_unicycle w_four_wheeler : ℕ)
                     (h1 : n_bicycles = 16)
                     (h2 : n_tricycles = 7)
                     (h3 : n_unicycles = 10)
                     (h4 : n_four_wheelers = 5)
                     (h5 : w_bicycle = 2)
                     (h6 : w_tricycle = 3)
                     (h7 : w_unicycle = 1)
                     (h8 : w_four_wheeler = 4)
  : (n_bicycles * w_bicycle + n_tricycles * w_tricycle
     + n_unicycles * w_unicycle + n_four_wheelers * w_four_wheeler) = 83 := by
  sorry

end total_wheels_l68_68061


namespace second_offset_length_l68_68217

noncomputable def quadrilateral_area (d o1 o2 : ℝ) : ℝ :=
  (1 / 2) * d * (o1 + o2)

theorem second_offset_length (d o1 A : ℝ) (h_d : d = 22) (h_o1 : o1 = 9) (h_A : A = 165) :
  ∃ o2, quadrilateral_area d o1 o2 = A ∧ o2 = 6 := by
  sorry

end second_offset_length_l68_68217


namespace fruiting_plants_given_away_l68_68180

noncomputable def roxy_fruiting_plants_given_away 
  (N_f : ℕ) -- initial flowering plants
  (N_ft : ℕ) -- initial fruiting plants
  (N_bsf : ℕ) -- flowering plants bought on Saturday
  (N_bst : ℕ) -- fruiting plants bought on Saturday
  (N_gsf : ℕ) -- flowering plant given away on Sunday
  (N_total_remaining : ℕ) -- total plants remaining 
  (H₁ : N_ft = 2 * N_f) -- twice as many fruiting plants
  (H₂ : N_total_remaining = (N_f + N_bsf - N_gsf) + (N_ft + N_bst - N_gst)) -- total plants equation
  : ℕ :=
  4

theorem fruiting_plants_given_away (N_f : ℕ) (N_ft : ℕ) (N_bsf : ℕ) (N_bst : ℕ) (N_gsf : ℕ) (N_total_remaining : ℕ)
  (H₁ : N_ft = 2 * N_f) (H₂ : N_total_remaining = (N_f + N_bsf - N_gsf) + (N_ft + N_bst - N_gst)) : N_ft - (N_total_remaining - (N_f + N_bsf - N_gsf)) = 4 := 
by
  sorry

end fruiting_plants_given_away_l68_68180


namespace polynomial_characterization_l68_68548
open Polynomial

noncomputable def satisfies_functional_eq (P : Polynomial ℝ) :=
  ∀ (a b c : ℝ), 
  P.eval (a + b - 2*c) + P.eval (b + c - 2*a) + P.eval (c + a - 2*b) = 
  3 * P.eval (a - b) + 3 * P.eval (b - c) + 3 * P.eval (c - a)

theorem polynomial_characterization (P : Polynomial ℝ) :
  satisfies_functional_eq P ↔ 
  (∃ a b : ℝ, P = Polynomial.C a * Polynomial.X + Polynomial.C b) ∨
  (∃ a b : ℝ, P = Polynomial.C a * Polynomial.X^2 + Polynomial.C b * Polynomial.X) :=
sorry

end polynomial_characterization_l68_68548


namespace scenario_1_scenario_2_scenario_3_scenario_4_l68_68128

-- Definitions based on conditions
def prob_A_hit : ℚ := 2 / 3
def prob_B_hit : ℚ := 3 / 4

-- Scenario 1: Prove that the probability of A shooting 3 times and missing at least once is 19/27
theorem scenario_1 : 
  (1 - (prob_A_hit ^ 3)) = 19 / 27 :=
by sorry

-- Scenario 2: Prove that the probability of A hitting the target exactly 2 times and B hitting the target exactly 1 time after each shooting twice is 1/6
theorem scenario_2 : 
  (2 * ((prob_A_hit ^ 2) * (1 - prob_A_hit)) * (2 * (prob_B_hit * (1 - prob_B_hit)))) = 1 / 6 :=
by sorry

-- Scenario 3: Prove that the probability of A missing the target and B hitting the target 2 times after each shooting twice is 1/16
theorem scenario_3 :
  ((1 - prob_A_hit) ^ 2) * (prob_B_hit ^ 2) = 1 / 16 :=
by sorry

-- Scenario 4: Prove that the probability that both A and B hit the target once after each shooting twice is 1/6
theorem scenario_4 : 
  (2 * (prob_A_hit * (1 - prob_A_hit)) * 2 * (prob_B_hit * (1 - prob_B_hit))) = 1 / 6 :=
by sorry

end scenario_1_scenario_2_scenario_3_scenario_4_l68_68128


namespace robert_salary_loss_l68_68907

theorem robert_salary_loss (S : ℝ) : 
  let decreased_salary := S - 0.3 * S
  let increased_salary := decreased_salary + 0.3 * decreased_salary
  100 * (1 - increased_salary / S) = 9 :=
by
  let decreased_salary := S - 0.3 * S
  let increased_salary := decreased_salary + 0.3 * decreased_salary
  sorry

end robert_salary_loss_l68_68907


namespace no_ordered_triples_l68_68676

noncomputable def no_solution (x y z : ℝ) : Prop :=
  x^2 - 3 * x * y + 2 * y^2 - z^2 = 31 ∧
  -x^2 + 6 * y * z + 2 * z^2 = 44 ∧
  x^2 + x * y + 8 * z^2 = 100

theorem no_ordered_triples : ¬ ∃ (x y z : ℝ), no_solution x y z := 
by 
  sorry

end no_ordered_triples_l68_68676


namespace ratio_of_a_to_b_l68_68863

theorem ratio_of_a_to_b 
  (b c d a : ℚ)
  (h1 : b / c = 13 / 9)
  (h2 : c / d = 5 / 13)
  (h3 : a / d = 1 / 7.2) :
  a / b = 1 / 4 := 
by sorry

end ratio_of_a_to_b_l68_68863


namespace math_problem_l68_68991

def a : ℕ := 2013
def b : ℕ := 2014

theorem math_problem :
  (a^3 - 2 * a^2 * b + 3 * a * b^2 - b^3 + 1) / (a * b) = a := by
  sorry

end math_problem_l68_68991


namespace black_to_white_ratio_l68_68861

theorem black_to_white_ratio (initial_black initial_white new_black new_white : ℕ) 
  (h1 : initial_black = 7) (h2 : initial_white = 18)
  (h3 : new_black = 31) (h4 : new_white = 18) :
  (new_black : ℚ) / new_white = 31 / 18 :=
by
  sorry

end black_to_white_ratio_l68_68861


namespace system_of_equations_xy_l68_68951

theorem system_of_equations_xy (x y : ℝ)
  (h1 : 2 * x + y = 7)
  (h2 : x + 2 * y = 5) :
  x - y = 2 := sorry

end system_of_equations_xy_l68_68951


namespace find_s_range_l68_68140

variables {a b c s t y1 y2 : ℝ}

-- Conditions
def is_vertex (a b c s t : ℝ) : Prop := ∀ x : ℝ, (a * x^2 + b * x + c = a * (x - s)^2 + t)

def passes_points (a b c y1 y2 : ℝ) : Prop := 
  (a * (-2)^2 + b * (-2) + c = y1) ∧ (a * 4^2 + b * 4 + c = y2)

def valid_constants (a y1 y2 t : ℝ) : Prop := 
  (a ≠ 0) ∧ (y1 > y2) ∧ (y2 > t)

-- Theorem
theorem find_s_range {a b c s t y1 y2 : ℝ}
  (hv : is_vertex a b c s t)
  (hp : passes_points a b c y1 y2)
  (vc : valid_constants a y1 y2 t) : 
  s > 1 ∧ s ≠ 4 :=
sorry -- Proof skipped

end find_s_range_l68_68140


namespace problem_I_problem_II_l68_68875

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x + (4 / x) - m| + m

-- Proof problem (I): When m = 0, find the minimum value of the function f(x).
theorem problem_I : ∀ x : ℝ, (f x 0) ≥ 4 := by
  sorry

-- Proof problem (II): If the function f(x) ≤ 5 for all x ∈ [1, 4], find the range of m.
theorem problem_II (m : ℝ) : (∀ x : ℝ, 1 ≤ x → x ≤ 4 → f x m ≤ 5) ↔ m ≤ 9 / 2 := by
  sorry

end problem_I_problem_II_l68_68875


namespace no_100_roads_l68_68652

theorem no_100_roads (k : ℕ) (hk : 3 * k % 2 = 0) : 100 ≠ 3 * k / 2 := 
by
  sorry

end no_100_roads_l68_68652


namespace unique_solution_of_quadratic_l68_68637

theorem unique_solution_of_quadratic (b c x : ℝ) (h_eqn : 9 * x^2 + b * x + c = 0) (h_one_solution : ∀ y: ℝ, 9 * y^2 + b * y + c = 0 → y = x) (h_b2_4c : b^2 = 4 * c) : 
  x = -b / 18 := 
by 
  sorry

end unique_solution_of_quadratic_l68_68637


namespace purchase_probability_l68_68264

/--
A batch of products from a company has packages containing 10 components each.
Each package has either 1 or 2 second-grade components. 10% of the packages
contain 2 second-grade components. Xiao Zhang will decide to purchase
if all 4 randomly selected components from a package are first-grade.

We aim to prove the probability that Xiao Zhang decides to purchase the company's
products is \( \frac{43}{75} \).
-/
theorem purchase_probability : true := sorry

end purchase_probability_l68_68264


namespace prove_mutually_exclusive_and_exhaustive_events_l68_68516

-- Definitions of conditions
def number_of_boys : ℕ := 3
def number_of_girls : ℕ := 2

-- Definitions of options
def option_A : Prop := (∃ (b1 b2 b3 : Bool) (g1 g2 : Bool), b1 ∧ b2 ∧ ¬b3 ∧ ¬g1 ∧ g2)  -- Exactly 1 boy and exactly 2 girls
def option_B : Prop := (∃ (b1 b2 b3 : Bool), b1 ∧ b2 ∧ b3)  -- At least 1 boy and all boys
def option_C : Prop := (∃ (b1 b2 b3 : Bool) (g1 g2 : Bool), b1 ∧ b2 ∧ (b3 ∨ g1 ∨ g2))  -- At least 1 boy and at least 1 girl
def option_D : Prop := (∃ (b1 b2 : Bool) (g3 : Bool), b1 ∧ ¬b2 ∧ g3)  -- At least 1 boy and all girls

-- The proof statement showing that option_D == Mutually Exclusive and Exhaustive Events
theorem prove_mutually_exclusive_and_exhaustive_events : option_D :=
sorry

end prove_mutually_exclusive_and_exhaustive_events_l68_68516


namespace polynomial_difference_of_squares_l68_68035

theorem polynomial_difference_of_squares (x y : ℤ) :
  8 * x^2 + 2 * x * y - 3 * y^2 = (3 * x - y)^2 - (x + 2 * y)^2 :=
by
  sorry

end polynomial_difference_of_squares_l68_68035


namespace product_of_numbers_l68_68918

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 185) : x * y = 26 :=
sorry

end product_of_numbers_l68_68918


namespace solve_abc_l68_68452

theorem solve_abc (a b c : ℕ) (h1 : a > b ∧ b > c) 
  (h2 : 34 - 6 * (a + b + c) + (a * b + b * c + c * a) = 0) 
  (h3 : 79 - 9 * (a + b + c) + (a * b + b * c + c * a) = 0) : 
  a = 10 ∧ b = 3 ∧ c = 2 := 
by
  sorry

end solve_abc_l68_68452


namespace count_integers_between_cubes_l68_68322

theorem count_integers_between_cubes (a b : ℝ) (h1 : a = 10.5) (h2 : b = 10.6) : 
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  (last_integer - first_integer + 1) = 33 :=
by
  -- Definitions for clarity
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  
  -- Skipping the proof
  sorry

end count_integers_between_cubes_l68_68322


namespace smallest_x_such_that_sum_is_cubic_l68_68500

/-- 
  Given a positive integer x, the sum of the sequence x, x+3, x+6, x+9, and x+12 should be a perfect cube.
  Prove that the smallest such x is 19.
-/
theorem smallest_x_such_that_sum_is_cubic : 
  ∃ (x : ℕ), 0 < x ∧ (∃ k : ℕ, 5 * x + 30 = k^3) ∧ ∀ y : ℕ, 0 < y → (∃ m : ℕ, 5 * y + 30 = m^3) → y ≥ x :=
sorry

end smallest_x_such_that_sum_is_cubic_l68_68500


namespace red_ants_count_l68_68033

def total_ants : ℕ := 900
def black_ants : ℕ := 487
def red_ants (r : ℕ) : Prop := r + black_ants = total_ants

theorem red_ants_count : ∃ r : ℕ, red_ants r ∧ r = 413 := 
sorry

end red_ants_count_l68_68033


namespace power_mod_2040_l68_68528

theorem power_mod_2040 : (6^2040) % 13 = 1 := by
  -- Skipping the proof as the problem only requires the statement
  sorry

end power_mod_2040_l68_68528


namespace minimum_value_of_f_l68_68446

noncomputable def f (x : ℝ) : ℝ := sorry  -- define f such that f(x + 199) = 4x^2 + 4x + 3 for x ∈ ℝ

theorem minimum_value_of_f : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = 2 := by
  sorry  -- Prove that the minimum value of f(x) is 2

end minimum_value_of_f_l68_68446


namespace parallel_line_plane_no_common_points_l68_68615

noncomputable def line := Type
noncomputable def plane := Type

variable {l : line}
variable {α : plane}

-- Definitions for parallel lines and planes, and relations between lines and planes
def parallel_to_plane (l : line) (α : plane) : Prop := sorry -- Definition of line parallel to plane
def within_plane (m : line) (α : plane) : Prop := sorry -- Definition of line within plane
def no_common_points (l m : line) : Prop := sorry -- Definition of no common points between lines

theorem parallel_line_plane_no_common_points
  (h₁ : parallel_to_plane l α)
  (l2 : line)
  (h₂ : within_plane l2 α) :
  no_common_points l l2 :=
sorry

end parallel_line_plane_no_common_points_l68_68615


namespace find_k_and_b_l68_68196

variables (k b : ℝ)

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (k * p.1, p.2 + b)

theorem find_k_and_b
  (h : f k b (6, 2) = (3, 1)) :
  k = 2 ∧ b = -1 :=
by {
  -- proof steps would go here
  sorry
}

end find_k_and_b_l68_68196


namespace quad_eq_double_root_m_value_l68_68960

theorem quad_eq_double_root_m_value (m : ℝ) : 
  (∀ x : ℝ, x^2 + 6 * x + m = 0) → m = 9 := 
by 
  sorry

end quad_eq_double_root_m_value_l68_68960


namespace three_digit_divisible_by_7_iff_last_two_digits_equal_l68_68585

-- Define the conditions as given in the problem
variable (a b c : ℕ)

-- Ensure the sum of the digits is 7, as given by the problem conditions
def sum_of_digits_eq_7 := a + b + c = 7

-- Ensure that it is a three-digit number
def valid_three_digit_number := a ≠ 0

-- Define what it means to be divisible by 7
def divisible_by_7 (n : ℕ) := n % 7 = 0

-- Define the problem statement in Lean
theorem three_digit_divisible_by_7_iff_last_two_digits_equal (h1 : sum_of_digits_eq_7 a b c) (h2 : valid_three_digit_number a) :
  divisible_by_7 (100 * a + 10 * b + c) ↔ b = c :=
by sorry

end three_digit_divisible_by_7_iff_last_two_digits_equal_l68_68585
