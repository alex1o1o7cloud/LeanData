import Mathlib

namespace NUMINAMATH_GPT_horizontal_distance_l990_99084

theorem horizontal_distance (elev_initial elev_final v_ratio h_ratio distance: ℝ)
  (h1 : elev_initial = 100)
  (h2 : elev_final = 1450)
  (h3 : v_ratio = 1)
  (h4 : h_ratio = 2)
  (h5 : distance = 1350) :
  distance * h_ratio = 2700 := by
  sorry

end NUMINAMATH_GPT_horizontal_distance_l990_99084


namespace NUMINAMATH_GPT_milly_folds_count_l990_99098

theorem milly_folds_count (mixing_time baking_time total_minutes fold_time rest_time : ℕ) 
  (h : total_minutes = 360)
  (h_mixing_time : mixing_time = 10)
  (h_baking_time : baking_time = 30)
  (h_fold_time : fold_time = 5)
  (h_rest_time : rest_time = 75) : 
  (total_minutes - (mixing_time + baking_time)) / (fold_time + rest_time) = 4 := 
by
  sorry

end NUMINAMATH_GPT_milly_folds_count_l990_99098


namespace NUMINAMATH_GPT_alice_probability_l990_99068

noncomputable def probability_picking_exactly_three_green_marbles : ℚ :=
  let binom : ℚ := 35 -- binomial coefficient (7 choose 3)
  let prob_green : ℚ := 8 / 15 -- probability of picking a green marble
  let prob_purple : ℚ := 7 / 15 -- probability of picking a purple marble
  binom * (prob_green ^ 3) * (prob_purple ^ 4)

theorem alice_probability :
  probability_picking_exactly_three_green_marbles = 34454336 / 136687500 := by
  sorry

end NUMINAMATH_GPT_alice_probability_l990_99068


namespace NUMINAMATH_GPT_rachel_points_product_l990_99011

-- Define the scores in the first 10 games
def scores_first_10_games := [9, 5, 7, 4, 8, 6, 2, 3, 5, 6]

-- Define the conditions as given in the problem
def total_score_first_10_games := scores_first_10_games.sum = 55
def points_scored_in_game_11 (P₁₁ : ℕ) : Prop := P₁₁ < 10 ∧ (55 + P₁₁) % 11 = 0
def points_scored_in_game_12 (P₁₁ P₁₂ : ℕ) : Prop := P₁₂ < 10 ∧ (55 + P₁₁ + P₁₂) % 12 = 0

-- Prove the product of the points scored in eleventh and twelfth games
theorem rachel_points_product : ∃ P₁₁ P₁₂ : ℕ, total_score_first_10_games ∧ points_scored_in_game_11 P₁₁ ∧ points_scored_in_game_12 P₁₁ P₁₂ ∧ P₁₁ * P₁₂ = 0 :=
by 
  sorry -- proof not required

end NUMINAMATH_GPT_rachel_points_product_l990_99011


namespace NUMINAMATH_GPT_acrobats_count_l990_99056

theorem acrobats_count
  (a e c : ℕ)
  (h1 : 2 * a + 4 * e + 2 * c = 58)
  (h2 : a + e + c = 25) :
  a = 11 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_acrobats_count_l990_99056


namespace NUMINAMATH_GPT_count_valid_n_l990_99069

theorem count_valid_n : ∃ (n : ℕ), n < 200 ∧ (∃ (m : ℕ), (m % 4 = 0) ∧ (∃ (k : ℤ), n = 4 * k + 2 ∧ m = 4 * k * (k + 1))) ∧ (∃ k_range : ℕ, k_range = 50) :=
sorry

end NUMINAMATH_GPT_count_valid_n_l990_99069


namespace NUMINAMATH_GPT_garden_perimeter_is_44_l990_99080

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

end NUMINAMATH_GPT_garden_perimeter_is_44_l990_99080


namespace NUMINAMATH_GPT_determine_pairs_of_positive_integers_l990_99049

open Nat

theorem determine_pairs_of_positive_integers (n p : ℕ) (hp : Nat.Prime p) (hn_le_2p : n ≤ 2 * p)
    (hdiv : (p - 1)^n + 1 ∣ n^(p - 1)) : (n = 1) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
  sorry

end NUMINAMATH_GPT_determine_pairs_of_positive_integers_l990_99049


namespace NUMINAMATH_GPT_value_of_expression_l990_99002

theorem value_of_expression : 
  103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l990_99002


namespace NUMINAMATH_GPT_amount_cut_off_l990_99032

def initial_length : ℕ := 11
def final_length : ℕ := 7

theorem amount_cut_off : (initial_length - final_length) = 4 :=
by
  sorry

end NUMINAMATH_GPT_amount_cut_off_l990_99032


namespace NUMINAMATH_GPT_find_line_equation_l990_99063

variable (x y : ℝ)

theorem find_line_equation (hx : x = -5) (hy : y = 2)
  (line_through_point : ∃ a b c : ℝ, a * x + b * y + c = 0)
  (x_intercept_twice_y_intercept : ∀ a b c : ℝ, c ≠ 0 → b ≠ 0 → (a / c) = 2 * (c / b)) :
  ∃ a b c : ℝ, (a * x + b * y + c = 0 ∧ (a = 2 ∧ b = 5 ∧ c = 0) ∨ (a = 1 ∧ b = 2 ∧ c = 1)) :=
sorry

end NUMINAMATH_GPT_find_line_equation_l990_99063


namespace NUMINAMATH_GPT_weavers_problem_l990_99059

theorem weavers_problem 
  (W : ℕ) 
  (H1 : 1 = W / 4) 
  (H2 : 3.5 = 49 / 14) :
  W = 4 :=
by
  sorry

end NUMINAMATH_GPT_weavers_problem_l990_99059


namespace NUMINAMATH_GPT_original_proposition_inverse_proposition_converse_proposition_contrapositive_proposition_l990_99047

variable (a_n : ℕ → ℝ) (n : ℕ+)

-- To prove the original proposition
theorem original_proposition : (a_n n + a_n (n + 1)) / 2 < a_n n → (∀ m, a_n m ≥ a_n (m + 1)) := 
sorry

-- To prove the inverse proposition
theorem inverse_proposition : ((a_n n + a_n (n + 1)) / 2 ≥ a_n n → ¬ ∀ m, a_n m ≥ a_n (m + 1)) := 
sorry

-- To prove the converse proposition
theorem converse_proposition : (∀ m, a_n m ≥ a_n (m + 1)) → (a_n n + a_n (n + 1)) / 2 < a_n n := 
sorry

-- To prove the contrapositive proposition
theorem contrapositive_proposition : (¬ ∀ m, a_n m ≥ a_n (m + 1)) → (a_n n + a_n (n + 1)) / 2 ≥ a_n n :=
sorry

end NUMINAMATH_GPT_original_proposition_inverse_proposition_converse_proposition_contrapositive_proposition_l990_99047


namespace NUMINAMATH_GPT_trader_gain_percentage_l990_99062

variable (x : ℝ) (cost_of_one_pen : ℝ := x) (selling_cost_90_pens : ℝ := 90 * x) (gain : ℝ := 30 * x)

theorem trader_gain_percentage :
  30 * cost_of_one_pen / (90 * cost_of_one_pen) * 100 = 33.33 := by
  sorry

end NUMINAMATH_GPT_trader_gain_percentage_l990_99062


namespace NUMINAMATH_GPT_tan_bounds_l990_99021

theorem tan_bounds (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 1) :
    (2 / Real.pi) * (x / (1 - x)) ≤ Real.tan ((Real.pi * x) / 2) ∧
    Real.tan ((Real.pi * x) / 2) ≤ (Real.pi / 2) * (x / (1 - x)) :=
by
    sorry

end NUMINAMATH_GPT_tan_bounds_l990_99021


namespace NUMINAMATH_GPT_odd_squarefree_integers_1_to_199_l990_99097

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

end NUMINAMATH_GPT_odd_squarefree_integers_1_to_199_l990_99097


namespace NUMINAMATH_GPT_sin_double_angle_identity_l990_99064

variable (α : Real)

theorem sin_double_angle_identity (h : Real.sin (α + π / 6) = 1 / 3) : 
  Real.sin (2 * α + 5 * π / 6) = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_identity_l990_99064


namespace NUMINAMATH_GPT_scientific_notation_of_53_96_billion_l990_99054

theorem scientific_notation_of_53_96_billion :
  (53.96 * 10^9) = (5.396 * 10^10) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_53_96_billion_l990_99054


namespace NUMINAMATH_GPT_find_triplets_l990_99050

theorem find_triplets (u v w : ℝ):
  (u + v * w = 12) ∧ 
  (v + w * u = 12) ∧ 
  (w + u * v = 12) ↔ 
  (u = 3 ∧ v = 3 ∧ w = 3) ∨ 
  (u = -4 ∧ v = -4 ∧ w = -4) ∨ 
  (u = 1 ∧ v = 1 ∧ w = 11) ∨ 
  (u = 11 ∧ v = 1 ∧ w = 1) ∨ 
  (u = 1 ∧ v = 11 ∧ w = 1) := 
sorry

end NUMINAMATH_GPT_find_triplets_l990_99050


namespace NUMINAMATH_GPT_vegetables_harvest_problem_l990_99057

theorem vegetables_harvest_problem
  (same_area : ∀ (a b : ℕ), a = b)
  (first_field_harvest : ℕ := 900)
  (second_field_harvest : ℕ := 1500)
  (less_harvest_per_acre : ∀ (x : ℕ), x - 300 = y) :
  x = y ->
  900 / x = 1500 / (x + 300) :=
by
  sorry

end NUMINAMATH_GPT_vegetables_harvest_problem_l990_99057


namespace NUMINAMATH_GPT_range_of_t_for_obtuse_triangle_l990_99088

def is_obtuse_triangle (a b c : ℝ) : Prop := ∃t : ℝ, a = t - 1 ∧ b = t + 1 ∧ c = t + 3

theorem range_of_t_for_obtuse_triangle :
  ∀ t : ℝ, is_obtuse_triangle (t-1) (t+1) (t+3) → (3 < t ∧ t < 7) :=
by
  intros t ht
  sorry

end NUMINAMATH_GPT_range_of_t_for_obtuse_triangle_l990_99088


namespace NUMINAMATH_GPT_vector_norm_sq_sum_l990_99043

theorem vector_norm_sq_sum (a b : ℝ × ℝ) (m : ℝ × ℝ) (h_m : m = (4, 6))
  (h_midpoint : m = ((2 * a.1 + 2 * b.1) / 2, (2 * a.2 + 2 * b.2) / 2))
  (h_dot : a.1 * b.1 + a.2 * b.2 = 10) :
  a.1^2 + a.2^2 + b.1^2 + b.2^2 = 32 :=
by 
  sorry

end NUMINAMATH_GPT_vector_norm_sq_sum_l990_99043


namespace NUMINAMATH_GPT_isosceles_triangle_angles_l990_99046

theorem isosceles_triangle_angles 
  (α r R : ℝ)
  (isosceles : α ∈ {β : ℝ | β = α})
  (circumference_relation : R = 3 * r) :
  (α = Real.arccos (1 / 2 + 1 / (2 * Real.sqrt 3)) ∨ 
   α = Real.arccos (1 / 2 - 1 / (2 * Real.sqrt 3))) ∧ 
  (
    180 = 2 * (Real.arccos (1 / 2 + 1 / (2 * Real.sqrt 3))) + 2 * α ∨
    180 = 2 * (Real.arccos (1 / 2 - 1 / (2 * Real.sqrt 3))) + 2 * α 
  ) :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_angles_l990_99046


namespace NUMINAMATH_GPT_relationship_among_abc_l990_99018

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 3) ^ 2
noncomputable def c : ℝ := Real.log (1 / 30) / Real.log (1 / 3)

theorem relationship_among_abc : c > a ∧ a > b :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l990_99018


namespace NUMINAMATH_GPT_total_votes_cast_l990_99074

theorem total_votes_cast (V : ℝ) (h1 : ∃ x : ℝ, x = 0.31 * V) (h2 : ∃ y : ℝ, y = x + 2451) :
  V = 6450 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_cast_l990_99074


namespace NUMINAMATH_GPT_smallest_number_sum_of_three_squares_distinct_ways_l990_99015

theorem smallest_number_sum_of_three_squares_distinct_ways :
  ∃ n : ℤ, n = 30 ∧
  (∃ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℤ),
    a1^2 + b1^2 + c1^2 = n ∧
    a2^2 + b2^2 + c2^2 = n ∧
    a3^2 + b3^2 + c3^2 = n ∧
    (a1, b1, c1) ≠ (a2, b2, c2) ∧
    (a1, b1, c1) ≠ (a3, b3, c3) ∧
    (a2, b2, c2) ≠ (a3, b3, c3)) := sorry

end NUMINAMATH_GPT_smallest_number_sum_of_three_squares_distinct_ways_l990_99015


namespace NUMINAMATH_GPT_value_of_w_l990_99076

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

end NUMINAMATH_GPT_value_of_w_l990_99076


namespace NUMINAMATH_GPT_first_step_induction_l990_99036

theorem first_step_induction (n : ℕ) (h : 1 < n) : 1 + 1/2 + 1/3 < 2 :=
by
  sorry

end NUMINAMATH_GPT_first_step_induction_l990_99036


namespace NUMINAMATH_GPT_smallest_solution_proof_l990_99053

noncomputable def smallest_solution (x : ℝ) : ℝ :=
  if x = (1 - Real.sqrt 65) / 4 then x else x

theorem smallest_solution_proof :
  ∃ x : ℝ, (2 * x / (x - 2) + (2 * x^2 - 24) / x = 11) ∧
           (∀ y : ℝ, 2 * y / (y - 2) + (2 * y^2 - 24) / y = 11 → y ≥ (1 - Real.sqrt 65) / 4) ∧
           x = (1 - Real.sqrt 65) /4 :=
sorry

end NUMINAMATH_GPT_smallest_solution_proof_l990_99053


namespace NUMINAMATH_GPT_sally_jolly_money_sum_l990_99027

/-- Prove the combined amount of money of Sally and Jolly is $150 given the conditions. -/
theorem sally_jolly_money_sum (S J x : ℝ) (h1 : S - x = 80) (h2 : J + 20 = 70) (h3 : S + J = 150) : S + J = 150 :=
by
  sorry

end NUMINAMATH_GPT_sally_jolly_money_sum_l990_99027


namespace NUMINAMATH_GPT_Eunji_total_wrong_questions_l990_99060

theorem Eunji_total_wrong_questions 
  (solved_A : ℕ) (solved_B : ℕ) (wrong_A : ℕ) (right_diff : ℕ) 
  (h1 : solved_A = 12) 
  (h2 : solved_B = 15) 
  (h3 : wrong_A = 4) 
  (h4 : right_diff = 2) :
  (solved_A - (solved_A - (solved_A - wrong_A) + right_diff) + (solved_A - wrong_A) + right_diff - solved_B - (solved_B - (solved_A - (solved_A - wrong_A) + right_diff))) = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_Eunji_total_wrong_questions_l990_99060


namespace NUMINAMATH_GPT_opposite_face_number_l990_99039

theorem opposite_face_number (sum_faces : ℕ → ℕ → ℕ) (face_number : ℕ → ℕ) :
  (face_number 1 = 6) ∧ (face_number 2 = 7) ∧ (face_number 3 = 8) ∧ 
  (face_number 4 = 9) ∧ (face_number 5 = 10) ∧ (face_number 6 = 11) →
  (sum_faces 1 2 + sum_faces 3 4 + sum_faces 5 6 = 33 + 18) →
  (sum_faces 1 2 + sum_faces 3 4 + sum_faces 5 6 = 35 + 16) →
  (face_number 2 ≠ 9 ∨ face_number 2 ≠ 11) → 
  face_number 2 = 9 ∨ face_number 2 = 11 :=
by
  intros hface_numbers hsum1 hsum2 hnot_possible
  sorry

end NUMINAMATH_GPT_opposite_face_number_l990_99039


namespace NUMINAMATH_GPT_intersecting_lines_l990_99033

theorem intersecting_lines (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 3 * x * y ↔ (x = 0 ∨ y = 0) := by
  sorry

end NUMINAMATH_GPT_intersecting_lines_l990_99033


namespace NUMINAMATH_GPT_number_of_girls_l990_99034

variable (total_children : ℕ) (boys : ℕ)

theorem number_of_girls (h1 : total_children = 117) (h2 : boys = 40) : 
  total_children - boys = 77 := by
  sorry

end NUMINAMATH_GPT_number_of_girls_l990_99034


namespace NUMINAMATH_GPT_cyclic_quadrilateral_fourth_side_length_l990_99020

theorem cyclic_quadrilateral_fourth_side_length
  (r : ℝ) (a b c d : ℝ) (r_eq : r = 300 * Real.sqrt 2) (a_eq : a = 300) (b_eq : b = 400)
  (c_eq : c = 300) :
  d = 500 := 
by 
  sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_fourth_side_length_l990_99020


namespace NUMINAMATH_GPT_tiles_finite_initial_segment_l990_99093

theorem tiles_finite_initial_segment (S : ℕ → Prop) (hTiling : ∀ n : ℕ, ∃ m : ℕ, m ≥ n ∧ S m) :
  ∃ k : ℕ, ∀ n : ℕ, n ≤ k → S n :=
by
  sorry

end NUMINAMATH_GPT_tiles_finite_initial_segment_l990_99093


namespace NUMINAMATH_GPT_apple_eating_contest_l990_99095

theorem apple_eating_contest (a z : ℕ) (h_most : a = 8) (h_fewest : z = 1) : a - z = 7 :=
by
  sorry

end NUMINAMATH_GPT_apple_eating_contest_l990_99095


namespace NUMINAMATH_GPT_meal_cost_l990_99042

theorem meal_cost :
  ∃ (s c p : ℝ),
  (5 * s + 8 * c + 2 * p = 5.40) ∧
  (3 * s + 11 * c + 2 * p = 4.95) ∧
  (s + c + p = 1.55) :=
sorry

end NUMINAMATH_GPT_meal_cost_l990_99042


namespace NUMINAMATH_GPT_triangles_with_equal_angles_and_two_equal_sides_not_necessarily_congruent_l990_99026

-- Define the triangle type
structure Triangle :=
(SideA : ℝ)
(SideB : ℝ)
(SideC : ℝ)
(AngleA : ℝ)
(AngleB : ℝ)
(AngleC : ℝ)
(h1 : SideA > 0)
(h2 : SideB > 0)
(h3 : SideC > 0)
(h4 : AngleA + AngleB + AngleC = 180)

-- Define what it means for two triangles to have three equal angles
def have_equal_angles (T1 T2 : Triangle) : Prop :=
(T1.AngleA = T2.AngleA ∧ T1.AngleB = T2.AngleB ∧ T1.AngleC = T2.AngleC)

-- Define what it means for two triangles to have two equal sides
def have_two_equal_sides (T1 T2 : Triangle) : Prop :=
(T1.SideA = T2.SideA ∧ T1.SideB = T2.SideB) ∨
(T1.SideA = T2.SideA ∧ T1.SideC = T2.SideC) ∨
(T1.SideB = T2.SideB ∧ T1.SideC = T2.SideC)

-- Define what it means for two triangles to be congruent
def congruent (T1 T2 : Triangle) : Prop :=
(T1.SideA = T2.SideA ∧ T1.SideB = T2.SideB ∧ T1.SideC = T2.SideC ∧
 T1.AngleA = T2.AngleA ∧ T1.AngleB = T2.AngleB ∧ T1.AngleC = T2.AngleC)

-- The final theorem
theorem triangles_with_equal_angles_and_two_equal_sides_not_necessarily_congruent 
  (T1 T2 : Triangle) 
  (h_angles : have_equal_angles T1 T2)
  (h_sides : have_two_equal_sides T1 T2) : ¬ congruent T1 T2 :=
sorry

end NUMINAMATH_GPT_triangles_with_equal_angles_and_two_equal_sides_not_necessarily_congruent_l990_99026


namespace NUMINAMATH_GPT_bacon_percentage_l990_99061

theorem bacon_percentage (total_calories : ℕ) (bacon_calories : ℕ) (strips_of_bacon : ℕ) :
  total_calories = 1250 →
  bacon_calories = 125 →
  strips_of_bacon = 2 →
  (strips_of_bacon * bacon_calories * 100 / total_calories) = 20 :=
by sorry

end NUMINAMATH_GPT_bacon_percentage_l990_99061


namespace NUMINAMATH_GPT_chocolates_eaten_by_robert_l990_99081

theorem chocolates_eaten_by_robert (nickel_ate : ℕ) (robert_ate_more : ℕ) (H1 : nickel_ate = 3) (H2 : robert_ate_more = 4) :
  nickel_ate + robert_ate_more = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_chocolates_eaten_by_robert_l990_99081


namespace NUMINAMATH_GPT_magazines_per_bookshelf_l990_99014

noncomputable def total_books : ℕ := 23
noncomputable def total_books_and_magazines : ℕ := 2436
noncomputable def total_bookshelves : ℕ := 29

theorem magazines_per_bookshelf : (total_books_and_magazines - total_books) / total_bookshelves = 83 :=
by
  sorry

end NUMINAMATH_GPT_magazines_per_bookshelf_l990_99014


namespace NUMINAMATH_GPT_coordinates_of_B_l990_99037

def pointA : Prod Int Int := (-3, 2)
def moveRight (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1 + units, p.2)
def moveDown (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1, p.2 - units)
def pointB : Prod Int Int := moveDown (moveRight pointA 1) 2

theorem coordinates_of_B :
  pointB = (-2, 0) :=
sorry

end NUMINAMATH_GPT_coordinates_of_B_l990_99037


namespace NUMINAMATH_GPT_sqrt_200_eq_10_l990_99044

theorem sqrt_200_eq_10 (h : 200 = 2^2 * 5^2) : Real.sqrt 200 = 10 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_200_eq_10_l990_99044


namespace NUMINAMATH_GPT_new_volume_correct_l990_99017

-- Define the conditions
def original_volume : ℝ := 60
def length_factor : ℝ := 3
def width_factor : ℝ := 2
def height_factor : ℝ := 1.20

-- Define the new volume as a result of the above factors
def new_volume : ℝ := original_volume * length_factor * width_factor * height_factor

-- Proof statement for the new volume being 432 cubic feet
theorem new_volume_correct : new_volume = 432 :=
by 
    -- Directly state the desired equality
    sorry

end NUMINAMATH_GPT_new_volume_correct_l990_99017


namespace NUMINAMATH_GPT_log_product_l990_99065

theorem log_product :
  (Real.log 100 / Real.log 10) * (Real.log (1 / 10) / Real.log 10) = -2 := by
  sorry

end NUMINAMATH_GPT_log_product_l990_99065


namespace NUMINAMATH_GPT_quadratic_has_real_roots_iff_l990_99086

theorem quadratic_has_real_roots_iff (a : ℝ) :
  (∃ (x : ℝ), a * x^2 - 4 * x - 2 = 0) ↔ (a ≥ -2 ∧ a ≠ 0) := by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_iff_l990_99086


namespace NUMINAMATH_GPT_ratio_f_l990_99029

variable (f : ℝ → ℝ)

-- Hypothesis: For all x in ℝ^+, f'(x) = 3/x * f(x)
axiom hyp1 : ∀ x : ℝ, x > 0 → deriv f x = (3 / x) * f x

-- Hypothesis: f(2^2016) ≠ 0
axiom hyp2 : f (2^2016) ≠ 0

-- Prove that f(2^2017) / f(2^2016) = 8
theorem ratio_f : f (2^2017) / f (2^2016) = 8 :=
sorry

end NUMINAMATH_GPT_ratio_f_l990_99029


namespace NUMINAMATH_GPT_geometric_seq_sum_l990_99092

theorem geometric_seq_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, S n = a 0 * (1 - q^n) / (1 - q))
    (h2 : S 10 = 10) (h3 : S 30 = 70) (hq_pos : 0 < q) :
    S 40 = 150 := by
  sorry

end NUMINAMATH_GPT_geometric_seq_sum_l990_99092


namespace NUMINAMATH_GPT_union_M_N_l990_99072

def M : Set ℝ := { x | -3 < x ∧ x ≤ 5 }
def N : Set ℝ := { x | x > 3 }

theorem union_M_N : M ∪ N = { x | x > -3 } :=
by
  sorry

end NUMINAMATH_GPT_union_M_N_l990_99072


namespace NUMINAMATH_GPT_complement_union_correct_l990_99099

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

end NUMINAMATH_GPT_complement_union_correct_l990_99099


namespace NUMINAMATH_GPT_length_of_room_l990_99007

theorem length_of_room 
  (width : ℝ) (cost : ℝ) (rate : ℝ) (area : ℝ) (length : ℝ) 
  (h1 : width = 3.75) 
  (h2 : cost = 24750) 
  (h3 : rate = 1200) 
  (h4 : area = cost / rate) 
  (h5 : area = length * width) : 
  length = 5.5 :=
sorry

end NUMINAMATH_GPT_length_of_room_l990_99007


namespace NUMINAMATH_GPT_florist_sold_roses_l990_99075

theorem florist_sold_roses (x : ℕ) (h1 : 5 - x + 34 = 36) : x = 3 :=
by sorry

end NUMINAMATH_GPT_florist_sold_roses_l990_99075


namespace NUMINAMATH_GPT_height_of_brick_l990_99085

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

end NUMINAMATH_GPT_height_of_brick_l990_99085


namespace NUMINAMATH_GPT_valid_parameterizations_l990_99048

theorem valid_parameterizations (y x : ℝ) (t : ℝ) :
  let A := (⟨0, 4⟩ : ℝ × ℝ) + t • (⟨3, 1⟩ : ℝ × ℝ)
  let B := (⟨-4/3, 0⟩ : ℝ × ℝ) + t • (⟨-1, -3⟩ : ℝ × ℝ)
  let C := (⟨1, 7⟩ : ℝ × ℝ) + t • (⟨9, 3⟩ : ℝ × ℝ)
  let D := (⟨2, 10⟩ : ℝ × ℝ) + t • (⟨1/3, 1⟩ : ℝ × ℝ)
  let E := (⟨-4, -8⟩ : ℝ × ℝ) + t • (⟨1/9, 1/3⟩ : ℝ × ℝ)
  (B = (x, y) ∧ D = (x, y) ∧ E = (x, y)) ↔ y = 3 * x + 4 :=
sorry

end NUMINAMATH_GPT_valid_parameterizations_l990_99048


namespace NUMINAMATH_GPT_lcm_18_30_l990_99079

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end NUMINAMATH_GPT_lcm_18_30_l990_99079


namespace NUMINAMATH_GPT_maximum_value_of_function_l990_99016

theorem maximum_value_of_function (a : ℕ) (ha : 0 < a) : 
  ∃ x : ℝ, x + Real.sqrt (13 - 2 * a * x) = 7 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_function_l990_99016


namespace NUMINAMATH_GPT_evaluate_expression_l990_99019

theorem evaluate_expression (a b : ℕ) (ha : a = 3) (hb : b = 2) : ((a^b)^a + (b^a)^b = 793) := by
  -- The following lines skip the proof but outline the structure:
  sorry

end NUMINAMATH_GPT_evaluate_expression_l990_99019


namespace NUMINAMATH_GPT_trajectory_of_P_l990_99052

-- Define points F1 and F2
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)

-- Define the condition |PF2| - |PF1| = 4 for a moving point P
def condition (P : ℝ × ℝ) : Prop :=
  let PF1 := Real.sqrt ((P.1 + 4)^2 + P.2^2)
  let PF2 := Real.sqrt ((P.1 - 4)^2 + P.2^2)
  abs (PF2 - PF1) = 4

-- The target equation of the trajectory
def target_eq (P : ℝ × ℝ) : Prop :=
  P.1 * P.1 / 4 - P.2 * P.2 / 12 = 1 ∧ P.1 ≤ -2

theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, condition P → target_eq P := by
  sorry

end NUMINAMATH_GPT_trajectory_of_P_l990_99052


namespace NUMINAMATH_GPT_g_symmetric_l990_99004

theorem g_symmetric (g : ℝ → ℝ) (h₀ : ∀ x, x ≠ 0 → (g x + 3 * g (1 / x) = 4 * x ^ 2)) : 
  ∀ x : ℝ, x ≠ 0 → g x = g (-x) :=
by 
  sorry

end NUMINAMATH_GPT_g_symmetric_l990_99004


namespace NUMINAMATH_GPT_circles_intersect_l990_99094

def circle_eq1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 3 = 0
def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y + 4 = 0

theorem circles_intersect :
  (∃ (x y : ℝ), circle_eq1 x y ∧ circle_eq2 x y) :=
sorry

end NUMINAMATH_GPT_circles_intersect_l990_99094


namespace NUMINAMATH_GPT_inequality_8xyz_leq_1_equality_cases_8xyz_eq_1_l990_99013

theorem inequality_8xyz_leq_1 (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z ≤ 1 :=
sorry

theorem equality_cases_8xyz_eq_1 (x y z : ℝ) (h1 : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z = 1 ↔ 
  (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨ 
  (x = -1/2 ∧ y = -1/2 ∧ z = 1/2) ∨ 
  (x = -1/2 ∧ y = 1/2 ∧ z = -1/2) ∨ 
  (x = 1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end NUMINAMATH_GPT_inequality_8xyz_leq_1_equality_cases_8xyz_eq_1_l990_99013


namespace NUMINAMATH_GPT_newspaper_target_l990_99083

theorem newspaper_target (total_collected_2_weeks : Nat) (needed_more : Nat) (sections : Nat) (kilos_per_section_2_weeks : Nat)
  (h1 : sections = 6)
  (h2 : kilos_per_section_2_weeks = 280)
  (h3 : total_collected_2_weeks = sections * kilos_per_section_2_weeks)
  (h4 : needed_more = 320)
  : total_collected_2_weeks + needed_more = 2000 :=
by
  sorry

end NUMINAMATH_GPT_newspaper_target_l990_99083


namespace NUMINAMATH_GPT_find_value_of_expression_l990_99009

variable (p q r s : ℝ)

def g (x : ℝ) : ℝ := p * x ^ 3 + q * x ^ 2 + r * x + s

-- We state the condition that g(1) = 1
axiom g_at_one : g p q r s 1 = 1

-- Now, we state the problem we need to prove:
theorem find_value_of_expression : 5 * p - 3 * q + 2 * r - s = 5 :=
by
  -- We skip the proof here
  exact sorry

end NUMINAMATH_GPT_find_value_of_expression_l990_99009


namespace NUMINAMATH_GPT_total_paved_1120_l990_99067

-- Definitions based on given problem conditions
def workers_paved_april : ℕ := 480
def less_than_march : ℕ := 160
def workers_paved_march : ℕ := workers_paved_april + less_than_march
def total_paved : ℕ := workers_paved_april + workers_paved_march

-- The statement to prove
theorem total_paved_1120 : total_paved = 1120 := by
  sorry

end NUMINAMATH_GPT_total_paved_1120_l990_99067


namespace NUMINAMATH_GPT_average_annual_population_increase_l990_99073

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

end NUMINAMATH_GPT_average_annual_population_increase_l990_99073


namespace NUMINAMATH_GPT_digits_sum_is_23_l990_99077

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

end NUMINAMATH_GPT_digits_sum_is_23_l990_99077


namespace NUMINAMATH_GPT_maria_cookies_left_l990_99090

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

end NUMINAMATH_GPT_maria_cookies_left_l990_99090


namespace NUMINAMATH_GPT_universal_proposition_example_l990_99066

theorem universal_proposition_example :
  (∀ n : ℕ, n % 2 = 0 → ∃ k : ℕ, n = 2 * k) :=
sorry

end NUMINAMATH_GPT_universal_proposition_example_l990_99066


namespace NUMINAMATH_GPT_heartbeats_during_race_l990_99045

-- Define the conditions as constants
def heart_rate := 150 -- beats per minute
def race_distance := 26 -- miles
def pace := 5 -- minutes per mile

-- Formulate the statement
theorem heartbeats_during_race :
  heart_rate * (race_distance * pace) = 19500 :=
by
  sorry

end NUMINAMATH_GPT_heartbeats_during_race_l990_99045


namespace NUMINAMATH_GPT_initial_water_amount_l990_99022

open Real

theorem initial_water_amount (W : ℝ)
  (h1 : ∀ (d : ℝ), d = 0.03 * 20)
  (h2 : ∀ (W : ℝ) (d : ℝ), d = 0.06 * W) :
  W = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_water_amount_l990_99022


namespace NUMINAMATH_GPT_intersection_point_unique_l990_99030

theorem intersection_point_unique (k : ℝ) :
  (∃ y : ℝ, k = -2 * y^2 - 3 * y + 5) ∧ (∀ y₁ y₂ : ℝ, y₁ ≠ y₂ → -2 * y₁^2 - 3 * y₁ + 5 ≠ k ∨ -2 * y₂^2 - 3 * y₂ + 5 ≠ k)
  ↔ k = 49 / 8 := 
by sorry

end NUMINAMATH_GPT_intersection_point_unique_l990_99030


namespace NUMINAMATH_GPT_ratio_A_B_share_l990_99031

-- Define the capital contributions and time in months
def A_capital : ℕ := 3500
def B_capital : ℕ := 15750
def A_months: ℕ := 12
def B_months: ℕ := 4

-- Effective capital contributions
def A_contribution : ℕ := A_capital * A_months
def B_contribution : ℕ := B_capital * B_months

-- Declare the theorem to prove the ratio 2:3
theorem ratio_A_B_share : A_contribution / 21000 = 2 ∧ B_contribution / 21000 = 3 :=
by
  -- Calculate and simplify the ratios
  have hA : A_contribution = 42000 := rfl
  have hB : B_contribution = 63000 := rfl
  have hGCD : Nat.gcd 42000 63000 = 21000 := rfl
  sorry

end NUMINAMATH_GPT_ratio_A_B_share_l990_99031


namespace NUMINAMATH_GPT_balloon_arrangement_count_l990_99005

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

end NUMINAMATH_GPT_balloon_arrangement_count_l990_99005


namespace NUMINAMATH_GPT_max_probability_first_black_ace_l990_99087

def probability_first_black_ace(k : ℕ) : ℚ :=
  if 1 ≤ k ∧ k ≤ 51 then (52 - k) / 1326 else 0

theorem max_probability_first_black_ace : 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 51 → probability_first_black_ace k ≤ probability_first_black_ace 1 :=
by
  sorry

end NUMINAMATH_GPT_max_probability_first_black_ace_l990_99087


namespace NUMINAMATH_GPT_gcd_polynomial_l990_99008

theorem gcd_polynomial (n : ℕ) (h : n > 2^5) :
  Nat.gcd (n^5 + 125) (n + 5) = if n % 5 = 0 then 5 else 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_l990_99008


namespace NUMINAMATH_GPT_perfect_square_trinomial_l990_99091

theorem perfect_square_trinomial (m : ℤ) (h : ∃ b : ℤ, (x : ℤ) → x^2 - 10 * x + m = (x + b)^2) : m = 25 :=
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l990_99091


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l990_99028

theorem min_value_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 12) : 
  (1 / x) + (1 / y) ≥ 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l990_99028


namespace NUMINAMATH_GPT_find_sample_size_l990_99082

theorem find_sample_size : ∃ n : ℕ, n ∣ 36 ∧ (n + 1) ∣ 35 ∧ n = 6 := by
  use 6
  simp
  sorry

end NUMINAMATH_GPT_find_sample_size_l990_99082


namespace NUMINAMATH_GPT_anne_distance_l990_99001

-- Definitions based on conditions
def Time : ℕ := 5
def Speed : ℕ := 4
def Distance : ℕ := Speed * Time

-- Proof statement
theorem anne_distance : Distance = 20 := by
  sorry

end NUMINAMATH_GPT_anne_distance_l990_99001


namespace NUMINAMATH_GPT_cottage_cheese_quantity_l990_99025

theorem cottage_cheese_quantity (x : ℝ) 
    (milk_fat : ℝ := 0.05) 
    (curd_fat : ℝ := 0.155) 
    (whey_fat : ℝ := 0.005) 
    (milk_mass : ℝ := 1) 
    (h : (curd_fat * x + whey_fat * (milk_mass - x)) = milk_fat * milk_mass) : 
    x = 0.3 :=
    sorry

end NUMINAMATH_GPT_cottage_cheese_quantity_l990_99025


namespace NUMINAMATH_GPT_solve_for_f_1988_l990_99040

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom functional_eq (m n : ℕ+) : f (f m + f n) = m + n

theorem solve_for_f_1988 : f 1988 = 1988 :=
sorry

end NUMINAMATH_GPT_solve_for_f_1988_l990_99040


namespace NUMINAMATH_GPT_total_profit_l990_99024

theorem total_profit (A_investment : ℝ) (B_investment : ℝ) (C_investment : ℝ) 
                     (A_months : ℝ) (B_months : ℝ) (C_months : ℝ)
                     (C_share : ℝ) (A_profit_percentage : ℝ) : ℝ :=
  let A_capital_months := A_investment * A_months
  let B_capital_months := B_investment * B_months
  let C_capital_months := C_investment * C_months
  let total_capital_months := A_capital_months + B_capital_months + C_capital_months
  let P := (C_share * total_capital_months) / (C_capital_months * (1 - A_profit_percentage))
  P

example : total_profit 6500 8400 10000 6 5 3 1900 0.05 = 24667 := by
  sorry

end NUMINAMATH_GPT_total_profit_l990_99024


namespace NUMINAMATH_GPT_lines_intersect_l990_99089

def line1 (s : ℚ) : ℚ × ℚ :=
  (1 + 2 * s, 4 - 3 * s)

def line2 (v : ℚ) : ℚ × ℚ :=
  (3 + 4 * v, 9 - v)

theorem lines_intersect :
  ∃ s v : ℚ, (line1 s) = (line2 v) ∧ (line1 s) = (-17/5, 53/5) := 
sorry

end NUMINAMATH_GPT_lines_intersect_l990_99089


namespace NUMINAMATH_GPT_find_g_expression_l990_99000

theorem find_g_expression (f g : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f x = 2 * x + 3)
  (h2 : ∀ x : ℝ, g (x + 2) = f x) :
  ∀ x : ℝ, g x = 2 * x - 1 :=
by
  sorry

end NUMINAMATH_GPT_find_g_expression_l990_99000


namespace NUMINAMATH_GPT_remainder_9_plus_y_mod_31_l990_99078

theorem remainder_9_plus_y_mod_31 (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (9 + y) % 31 = 18 :=
sorry

end NUMINAMATH_GPT_remainder_9_plus_y_mod_31_l990_99078


namespace NUMINAMATH_GPT_vector_sum_is_correct_l990_99096

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

end NUMINAMATH_GPT_vector_sum_is_correct_l990_99096


namespace NUMINAMATH_GPT_square_area_l990_99023

theorem square_area (x1 x2 : ℝ) (hx1 : x1^2 + 4 * x1 + 3 = 8) (hx2 : x2^2 + 4 * x2 + 3 = 8) (h_eq : y = 8) : 
  (|x1 - x2|) ^ 2 = 36 :=
sorry

end NUMINAMATH_GPT_square_area_l990_99023


namespace NUMINAMATH_GPT_max_repeating_sequence_length_l990_99070

theorem max_repeating_sequence_length (p q n α β d : ℕ) (h_prime: Nat.gcd p q = 1)
  (hq : q = (2 ^ α) * (5 ^ β) * d) (hd_coprime: Nat.gcd d 10 = 1) (h_repeat: 10 ^ n ≡ 1 [MOD d]) :
  ∃ s, s ≤ n * (10 ^ n - 1) ∧ (10 ^ s ≡ 1 [MOD d^2]) :=
by
  sorry

end NUMINAMATH_GPT_max_repeating_sequence_length_l990_99070


namespace NUMINAMATH_GPT_denis_sum_of_numbers_l990_99035

theorem denis_sum_of_numbers :
  ∃ a b c d : ℕ, a < b ∧ b < c ∧ c < d ∧ a*d = 32 ∧ b*c = 14 ∧ a + b + c + d = 42 :=
sorry

end NUMINAMATH_GPT_denis_sum_of_numbers_l990_99035


namespace NUMINAMATH_GPT_remainder_is_210_l990_99071

-- Define necessary constants and theorems
def x : ℕ := 2^35
def dividend : ℕ := 2^210 + 210
def divisor : ℕ := 2^105 + 2^63 + 1

theorem remainder_is_210 : (dividend % divisor) = 210 :=
by 
  -- Assume the calculation steps in the preceding solution are correct.
  -- No need to manually re-calculate as we've directly taken from the solution.
  sorry

end NUMINAMATH_GPT_remainder_is_210_l990_99071


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l990_99058

-- Definitions of the variables and their values
def x : ℤ := -2
def y : ℚ := 1 / 2

-- Theorem statement
theorem simplify_and_evaluate_expression : 
  2 * (x^2 * y + x * y^2) - 2 * (x^2 * y - 1) - 3 * x * y^2 - 2 = 
  (1 : ℚ) / 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l990_99058


namespace NUMINAMATH_GPT_possible_values_of_y_l990_99055

theorem possible_values_of_y (x : ℝ) (hx : x^2 + 5 * (x / (x - 3)) ^ 2 = 50) :
  ∃ (y : ℝ), y = (x - 3)^2 * (x + 4) / (3 * x - 4) ∧ (y = 0 ∨ y = 15 ∨ y = 49) :=
sorry

end NUMINAMATH_GPT_possible_values_of_y_l990_99055


namespace NUMINAMATH_GPT_scientific_notation_of_diameter_l990_99041

theorem scientific_notation_of_diameter :
  0.00000258 = 2.58 * 10^(-6) :=
by sorry

end NUMINAMATH_GPT_scientific_notation_of_diameter_l990_99041


namespace NUMINAMATH_GPT_max_value_of_expression_l990_99006

-- Define the variables and condition.
variable (x y z : ℝ)
variable (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1)

-- State the theorem.
theorem max_value_of_expression :
  (8 * x + 5 * y + 15 * z) ≤ 4.54 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l990_99006


namespace NUMINAMATH_GPT_volume_in_region_l990_99003

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

end NUMINAMATH_GPT_volume_in_region_l990_99003


namespace NUMINAMATH_GPT_primary_college_employee_relation_l990_99012

theorem primary_college_employee_relation
  (P C N : ℕ)
  (hN : N = 20 + P + C)
  (h_illiterate_wages_before : 20 * 25 = 500)
  (h_illiterate_wages_after : 20 * 10 = 200)
  (h_primary_wages_before : P * 40 = P * 40)
  (h_primary_wages_after : P * 25 = P * 25)
  (h_college_wages_before : C * 50 = C * 50)
  (h_college_wages_after : C * 60 = C * 60)
  (h_avg_decrease : (500 + 40 * P + 50 * C) / N - (200 + 25 * P + 60 * C) / N = 10) :
  15 * P - 10 * C = 10 * N - 300 := 
by
  sorry

end NUMINAMATH_GPT_primary_college_employee_relation_l990_99012


namespace NUMINAMATH_GPT_max_value_expression_l990_99038

theorem max_value_expression (r : ℝ) : ∃ r : ℝ, -5 * r^2 + 40 * r - 12 = 68 ∧ (∀ s : ℝ, -5 * s^2 + 40 * s - 12 ≤ 68) :=
sorry

end NUMINAMATH_GPT_max_value_expression_l990_99038


namespace NUMINAMATH_GPT_no_good_number_exists_l990_99010

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_good (n : ℕ) : Prop :=
  (n % sum_of_digits n = 0) ∧
  ((n + 1) % sum_of_digits (n + 1) = 0) ∧
  ((n + 2) % sum_of_digits (n + 2) = 0) ∧
  ((n + 3) % sum_of_digits (n + 3) = 0)

theorem no_good_number_exists : ¬ ∃ n : ℕ, is_good n :=
by sorry

end NUMINAMATH_GPT_no_good_number_exists_l990_99010


namespace NUMINAMATH_GPT_expression_evaluation_l990_99051

theorem expression_evaluation : (16^3 + 3 * 16^2 + 3 * 16 + 1 = 4913) :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l990_99051
