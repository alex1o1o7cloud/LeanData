import Mathlib

namespace NUMINAMATH_GPT_remainders_equality_l1236_123697

open Nat

theorem remainders_equality (P P' D R R' r r': ℕ) 
  (hP : P > P')
  (hP_R : P % D = R)
  (hP'_R' : P' % D = R')
  (hPP' : (P * P') % D = r)
  (hRR' : (R * R') % D = r') : r = r' := 
sorry

end NUMINAMATH_GPT_remainders_equality_l1236_123697


namespace NUMINAMATH_GPT_quadrilateral_area_is_11_l1236_123624

def point := (ℤ × ℤ)

def A : point := (0, 0)
def B : point := (1, 4)
def C : point := (4, 3)
def D : point := (3, 0)

def area_of_quadrilateral (p1 p2 p3 p4 : point) : ℤ :=
  let ⟨x1, y1⟩ := p1
  let ⟨x2, y2⟩ := p2
  let ⟨x3, y3⟩ := p3
  let ⟨x4, y4⟩ := p4
  (|x1*y2 - y1*x2 + x2*y3 - y2*x3 + x3*y4 - y3*x4 + x4*y1 - y4*x1|) / 2

theorem quadrilateral_area_is_11 : area_of_quadrilateral A B C D = 11 := by 
  sorry

end NUMINAMATH_GPT_quadrilateral_area_is_11_l1236_123624


namespace NUMINAMATH_GPT_tangent_line_equation_l1236_123613

/-- Prove that the equation of the tangent line to the curve y = x^3 - 4x^2 + 4 at the point (1,1) is y = -5x + 6 -/
theorem tangent_line_equation (x y : ℝ)
  (h_curve : y = x^3 - 4 * x^2 + 4)
  (h_point : x = 1 ∧ y = 1) :
  y = -5 * x + 6 := by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l1236_123613


namespace NUMINAMATH_GPT_cubic_solution_identity_l1236_123674

theorem cubic_solution_identity {a b c : ℕ} 
  (h1 : a + b + c = 6) 
  (h2 : ab + bc + ca = 11) 
  (h3 : abc = 6) : 
  (ab / c) + (bc / a) + (ca / b) = 49 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_cubic_solution_identity_l1236_123674


namespace NUMINAMATH_GPT_probability_A_and_B_same_last_hour_l1236_123603
open Classical

-- Define the problem conditions
def attraction_count : ℕ := 6
def total_scenarios : ℕ := attraction_count * attraction_count
def favorable_scenarios : ℕ := attraction_count

-- Define the probability calculation
def probability_same_attraction : ℚ := favorable_scenarios / total_scenarios

-- The proof problem statement
theorem probability_A_and_B_same_last_hour : 
  probability_same_attraction = 1 / 6 :=
sorry

end NUMINAMATH_GPT_probability_A_and_B_same_last_hour_l1236_123603


namespace NUMINAMATH_GPT_vertical_asymptote_at_neg_two_over_three_l1236_123698

theorem vertical_asymptote_at_neg_two_over_three : 
  ∃ x : ℝ, 6 * x + 4 = 0 ∧ x = -2 / 3 := 
by
  use -2 / 3
  sorry

end NUMINAMATH_GPT_vertical_asymptote_at_neg_two_over_three_l1236_123698


namespace NUMINAMATH_GPT_maxValue_is_6084_over_17_l1236_123684

open Real

noncomputable def maxValue (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4

theorem maxValue_is_6084_over_17 (x y : ℝ) (h : x + y = 5) :
  maxValue x y h ≤ 6084 / 17 := 
sorry

end NUMINAMATH_GPT_maxValue_is_6084_over_17_l1236_123684


namespace NUMINAMATH_GPT_find_a_for_exponential_function_l1236_123661

theorem find_a_for_exponential_function (a : ℝ) :
  a - 2 = 1 ∧ a > 0 ∧ a ≠ 1 → a = 3 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end NUMINAMATH_GPT_find_a_for_exponential_function_l1236_123661


namespace NUMINAMATH_GPT_ratio_of_logs_l1236_123620

theorem ratio_of_logs (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : Real.log a / Real.log 4 = Real.log b / Real.log 18 ∧ Real.log b / Real.log 18 = Real.log (a + b) / Real.log 32) :
  b / a = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_logs_l1236_123620


namespace NUMINAMATH_GPT_correct_operation_l1236_123651

theorem correct_operation :
  (3 * m^2 + 4 * m^2 ≠ 7 * m^4) ∧
  (4 * m^3 * 5 * m^3 ≠ 20 * m^3) ∧
  ((-2 * m)^3 ≠ -6 * m^3) ∧
  (m^10 / m^5 = m^5) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1236_123651


namespace NUMINAMATH_GPT_eggs_for_husband_is_correct_l1236_123601

-- Define the conditions
def eggs_per_child : Nat := 2
def num_children : Nat := 4
def eggs_for_herself : Nat := 2
def total_eggs_per_year : Nat := 3380
def days_per_week : Nat := 5
def weeks_per_year : Nat := 52

-- Define the total number of eggs Lisa makes for her husband per year
def eggs_for_husband : Nat :=
  total_eggs_per_year - 
  (num_children * eggs_per_child + eggs_for_herself) * (days_per_week * weeks_per_year)

-- Prove the main statement
theorem eggs_for_husband_is_correct : eggs_for_husband = 780 := by
  sorry

end NUMINAMATH_GPT_eggs_for_husband_is_correct_l1236_123601


namespace NUMINAMATH_GPT_min_value_expr_l1236_123686

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b) + (b / c) + (c / a) + (a / c) ≥ 4 := 
sorry

end NUMINAMATH_GPT_min_value_expr_l1236_123686


namespace NUMINAMATH_GPT_zoo_rabbits_count_l1236_123666

theorem zoo_rabbits_count (parrots rabbits : ℕ) (h_ratio : parrots * 4 = rabbits * 3) (h_parrots_count : parrots = 21) : rabbits = 28 :=
by
  sorry

end NUMINAMATH_GPT_zoo_rabbits_count_l1236_123666


namespace NUMINAMATH_GPT_girls_more_than_boys_l1236_123649

/-- 
In a class with 42 students, where the ratio of boys to girls is 3:4, 
prove that there are 6 more girls than boys.
-/
theorem girls_more_than_boys (students total_students : ℕ) (boys girls : ℕ) (ratio_boys_girls : 3 * girls = 4 * boys)
  (total_students_count : boys + girls = total_students)
  (total_students_value : total_students = 42) : girls - boys = 6 :=
by
  sorry

end NUMINAMATH_GPT_girls_more_than_boys_l1236_123649


namespace NUMINAMATH_GPT_rope_segment_equation_l1236_123691

theorem rope_segment_equation (x : ℝ) (h1 : 2 - x > 0) :
  x^2 = 2 * (2 - x) :=
by
  sorry

end NUMINAMATH_GPT_rope_segment_equation_l1236_123691


namespace NUMINAMATH_GPT_calculation_equivalence_l1236_123640

theorem calculation_equivalence : 3000 * (3000 ^ 2999) = 3000 ^ 3000 := 
by
  sorry

end NUMINAMATH_GPT_calculation_equivalence_l1236_123640


namespace NUMINAMATH_GPT_initial_distance_proof_l1236_123652

noncomputable def initial_distance (V_A V_B T : ℝ) : ℝ :=
  (V_A * T) + (V_B * T)

theorem initial_distance_proof 
  (V_A V_B : ℝ) 
  (T : ℝ) 
  (h1 : V_A / V_B = 5 / 6)
  (h2 : V_B = 90)
  (h3 : T = 8 / 15) :
  initial_distance V_A V_B T = 88 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_initial_distance_proof_l1236_123652


namespace NUMINAMATH_GPT_rick_group_division_l1236_123668

theorem rick_group_division :
  ∀ (total_books : ℕ), total_books = 400 → 
  (∃ n : ℕ, (∀ (books_per_category : ℕ) (divisions : ℕ), books_per_category = total_books / (2 ^ divisions) → books_per_category = 25 → divisions = n) ∧ n = 4) :=
by
  sorry

end NUMINAMATH_GPT_rick_group_division_l1236_123668


namespace NUMINAMATH_GPT_stock_price_rise_l1236_123602

theorem stock_price_rise {P : ℝ} (h1 : P > 0)
    (h2007 : P * 1.20 = 1.20 * P)
    (h2008 : 1.20 * P * 0.75 = P * 0.90)
    (hCertainYear : P * 1.17 = P * 0.90 * (1 + 30 / 100)) :
  30 = 30 :=
by sorry

end NUMINAMATH_GPT_stock_price_rise_l1236_123602


namespace NUMINAMATH_GPT_trig_identity_proof_l1236_123641

theorem trig_identity_proof
  (α : ℝ)
  (h : Real.sin (α - π / 6) = 3 / 5) :
  Real.cos (2 * π / 3 - α) = 3 / 5 :=
sorry

end NUMINAMATH_GPT_trig_identity_proof_l1236_123641


namespace NUMINAMATH_GPT_Amanda_tickets_third_day_l1236_123687

theorem Amanda_tickets_third_day :
  (let total_tickets := 80
   let first_day_tickets := 5 * 4
   let second_day_tickets := 32

   total_tickets - (first_day_tickets + second_day_tickets) = 28) :=
by
  sorry

end NUMINAMATH_GPT_Amanda_tickets_third_day_l1236_123687


namespace NUMINAMATH_GPT_cauliflower_production_proof_l1236_123636

theorem cauliflower_production_proof (x y : ℕ) 
  (h1 : y^2 - x^2 = 401)
  (hx : x > 0)
  (hy : y > 0) :
  y^2 = 40401 :=
by
  sorry

end NUMINAMATH_GPT_cauliflower_production_proof_l1236_123636


namespace NUMINAMATH_GPT_equilateral_triangle_in_ellipse_l1236_123694

def ellipse_equation (x y a b : ℝ) : Prop := 
  ((x - y)^2 / a^2) + ((x + y)^2 / b^2) = 1

theorem equilateral_triangle_in_ellipse 
  {a b x y : ℝ}
  (A B C : ℝ × ℝ)
  (hA : A.1 = 0 ∧ A.2 = b)
  (hBC_parallel : ∃ k : ℝ, B.2 = k * B.1 ∧ C.2 = k * C.1 ∧ k = 1)
  (hF : ∃ F : ℝ × ℝ, F = C)
  (hEllipseA : ellipse_equation A.1 A.2 a b) 
  (hEllipseB : ellipse_equation B.1 B.2 a b)
  (hEllipseC : ellipse_equation C.1 C.2 a b) 
  (equilateral : dist A B = dist B C ∧ dist B C = dist C A) :
  AB / b = 8 / 5 :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_in_ellipse_l1236_123694


namespace NUMINAMATH_GPT_combined_river_length_estimate_l1236_123646

def river_length_GSA := 402 
def river_error_GSA := 0.5 
def river_prob_error_GSA := 0.04 

def river_length_AWRA := 403 
def river_error_AWRA := 0.5 
def river_prob_error_AWRA := 0.04 

/-- 
Given the measurements from GSA and AWRA, 
the combined estimate of the river's length, Rio-Coralio, is 402.5 km,
and the probability of error for this combined estimate is 0.04.
-/
theorem combined_river_length_estimate :
  ∃ l : ℝ, l = 402.5 ∧ ∀ p : ℝ, (p = 0.04) :=
sorry

end NUMINAMATH_GPT_combined_river_length_estimate_l1236_123646


namespace NUMINAMATH_GPT_simplify_expression_l1236_123628

theorem simplify_expression : ( (2^8 + 4^5) * (2^3 - (-2)^3) ^ 8 ) = 0 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l1236_123628


namespace NUMINAMATH_GPT_compute_fraction_l1236_123642

theorem compute_fraction : 
  (1 - 2 + 4 - 8 + 16 - 32 + 64) / (2 - 4 + 8 - 16 + 32 - 64 + 128) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_compute_fraction_l1236_123642


namespace NUMINAMATH_GPT_amount_used_to_pay_l1236_123673

noncomputable def the_cost_of_football : ℝ := 9.14
noncomputable def the_cost_of_baseball : ℝ := 6.81
noncomputable def the_change_received : ℝ := 4.05

theorem amount_used_to_pay : 
    (the_cost_of_football + the_cost_of_baseball + the_change_received) = 20.00 := 
by
  sorry

end NUMINAMATH_GPT_amount_used_to_pay_l1236_123673


namespace NUMINAMATH_GPT_rankings_are_correct_l1236_123645

-- Define teams:
inductive Team
| A | B | C | D

-- Define the type for ranking
structure Ranking :=
  (first : Team)
  (second : Team)
  (third : Team)
  (last : Team)

-- Define the predictions of Jia, Yi, and Bing
structure Predictions := 
  (Jia : Ranking)
  (Yi : Ranking)
  (Bing : Ranking)

-- Define the condition that each prediction is half right, half wrong
def isHalfRightHalfWrong (pred : Ranking) (actual : Ranking) : Prop :=
  (pred.first = actual.first ∨ pred.second = actual.second ∨ pred.third = actual.third ∨ pred.last = actual.last) ∧
  (pred.first ≠ actual.first ∨ pred.second ≠ actual.second ∨ pred.third ≠ actual.third ∨ pred.last ≠ actual.last)

-- Define the actual rankings
def actualRanking : Ranking := { first := Team.C, second := Team.A, third := Team.D, last := Team.B }

-- Define Jia's Predictions 
def JiaPrediction : Ranking := { first := Team.C, second := Team.C, third := Team.D, last := Team.D }

-- Define Yi's Predictions 
def YiPrediction : Ranking := { first := Team.B, second := Team.A, third := Team.C, last := Team.D }

-- Define Bing's Predictions 
def BingPrediction : Ranking := { first := Team.C, second := Team.B, third := Team.A, last := Team.D }

-- Create an instance of predictions
def pred : Predictions := { Jia := JiaPrediction, Yi := YiPrediction, Bing := BingPrediction }

-- The theorem to be proved
theorem rankings_are_correct :
  isHalfRightHalfWrong pred.Jia actualRanking ∧ 
  isHalfRightHalfWrong pred.Yi actualRanking ∧ 
  isHalfRightHalfWrong pred.Bing actualRanking →
  actualRanking.first = Team.C ∧ actualRanking.second = Team.A ∧ actualRanking.third = Team.D ∧ 
  actualRanking.last = Team.B :=
by
  sorry -- Proof is not required.

end NUMINAMATH_GPT_rankings_are_correct_l1236_123645


namespace NUMINAMATH_GPT_solve_diophantine_l1236_123688

theorem solve_diophantine : ∃ (x y : ℕ) (t : ℤ), x = 4 - 43 * t ∧ y = 6 - 65 * t ∧ t ≤ 0 ∧ 65 * x - 43 * y = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_diophantine_l1236_123688


namespace NUMINAMATH_GPT_find_A_l1236_123629

theorem find_A (A : ℤ) (h : A + 10 = 15) : A = 5 :=
sorry

end NUMINAMATH_GPT_find_A_l1236_123629


namespace NUMINAMATH_GPT_sobhas_parents_age_difference_l1236_123699

def difference_in_ages (F M : ℕ) : ℕ := F - M

theorem sobhas_parents_age_difference
  (S F M : ℕ)
  (h1 : F = S + 38)
  (h2 : M = S + 32) :
  difference_in_ages F M = 6 := by
  sorry

end NUMINAMATH_GPT_sobhas_parents_age_difference_l1236_123699


namespace NUMINAMATH_GPT_degrees_of_interior_angles_l1236_123669

-- Definitions for the problem conditions
variables {a b c h_a h_b S : ℝ} 
variables (ABC : Triangle) 
variables (height_to_bc height_to_ac : ℝ)
variables (le_a_ha : a ≤ height_to_bc)
variables (le_b_hb : b ≤ height_to_ac)
variables (area : S = 1 / 2 * a * height_to_bc)
variables (area_eq : S = 1 / 2 * b * height_to_ac)
variables (ha_eq : height_to_bc = 2 * S / a)
variables (hb_eq : height_to_ac = 2 * S / b)
variables (height_pos : 0 < 2 * S)
variables (length_pos : 0 < a ∧ 0 < b ∧ 0 < c)

-- Conclude the degrees of the interior angles
theorem degrees_of_interior_angles : 
  ∃ A B C : ℝ, A = 45 ∧ B = 45 ∧ C = 90 :=
sorry

end NUMINAMATH_GPT_degrees_of_interior_angles_l1236_123669


namespace NUMINAMATH_GPT_paige_team_total_players_l1236_123619

theorem paige_team_total_players 
    (total_points : ℕ)
    (paige_points : ℕ)
    (other_points_per_player : ℕ)
    (other_players : ℕ) :
    total_points = paige_points + other_points_per_player * other_players →
    (other_players + 1) = 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_paige_team_total_players_l1236_123619


namespace NUMINAMATH_GPT_honors_students_count_l1236_123667

variable {total_students : ℕ}
variable {total_girls total_boys : ℕ}
variable {honors_girls honors_boys : ℕ}

axiom class_size_constraint : total_students < 30
axiom prob_girls_honors : (honors_girls : ℝ) / total_girls = 3 / 13
axiom prob_boys_honors : (honors_boys : ℝ) / total_boys = 4 / 11
axiom total_students_eq : total_students = total_girls + total_boys
axiom honors_girls_value : honors_girls = 3
axiom honors_boys_value : honors_boys = 4

theorem honors_students_count : 
  honors_girls + honors_boys = 7 :=
by
  sorry

end NUMINAMATH_GPT_honors_students_count_l1236_123667


namespace NUMINAMATH_GPT_min_value_expression_l1236_123639

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  4 * x^3 + 8 * y^3 + 18 * z^3 + 1 / (6 * x * y * z) ≥ 4 := by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1236_123639


namespace NUMINAMATH_GPT_factor_expression_l1236_123671

theorem factor_expression (x : ℝ) : 
  4 * x * (x - 5) + 6 * (x - 5) = (4 * x + 6) * (x - 5) :=
by 
  sorry

end NUMINAMATH_GPT_factor_expression_l1236_123671


namespace NUMINAMATH_GPT_squares_are_equal_l1236_123654

theorem squares_are_equal (a b c d : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0) 
    (h₄ : a * (b + c + d) = b * (a + c + d)) 
    (h₅ : a * (b + c + d) = c * (a + b + d)) 
    (h₆ : a * (b + c + d) = d * (a + b + c)) : 
    a^2 = b^2 ∧ b^2 = c^2 ∧ c^2 = d^2 := 
by
  sorry

end NUMINAMATH_GPT_squares_are_equal_l1236_123654


namespace NUMINAMATH_GPT_greatest_divisor_l1236_123600

theorem greatest_divisor (n : ℕ) (h1 : 3461 % n = 23) (h2 : 4783 % n = 41) : n = 2 := by {
  sorry
}

end NUMINAMATH_GPT_greatest_divisor_l1236_123600


namespace NUMINAMATH_GPT_frame_interior_edge_sum_l1236_123662

theorem frame_interior_edge_sum (y : ℝ) :
  ( ∀ outer_edge1 : ℝ, outer_edge1 = 7 →
    ∀ frame_width : ℝ, frame_width = 2 →
    ∀ frame_area : ℝ, frame_area = 30 →
    7 * y - (3 * (y - 4)) = 30) → 
  (7 * y - (4 * y - 12) ) / 4 = 4.5 → 
  (3 + (y - 4)) * 2 = 7 :=
sorry

end NUMINAMATH_GPT_frame_interior_edge_sum_l1236_123662


namespace NUMINAMATH_GPT_negation_of_p_is_correct_l1236_123657

variable (c : ℝ)

-- Proposition p defined as: there exists c > 0 such that x^2 - x + c = 0 has a solution
def proposition_p : Prop :=
  ∃ c > 0, ∃ x : ℝ, x^2 - x + c = 0

-- Negation of proposition p
def neg_proposition_p : Prop :=
  ∀ c > 0, ¬ ∃ x : ℝ, x^2 - x + c = 0

-- The Lean statement to prove
theorem negation_of_p_is_correct :
  neg_proposition_p ↔ (∀ c > 0, ¬ ∃ x : ℝ, x^2 - x + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_is_correct_l1236_123657


namespace NUMINAMATH_GPT_extra_flowers_correct_l1236_123632

variable (pickedTulips : ℕ) (pickedRoses : ℕ) (usedFlowers : ℕ)

def totalFlowers : ℕ := pickedTulips + pickedRoses
def extraFlowers : ℕ := totalFlowers pickedTulips pickedRoses - usedFlowers

theorem extra_flowers_correct : 
  pickedTulips = 39 → pickedRoses = 49 → usedFlowers = 81 → extraFlowers pickedTulips pickedRoses usedFlowers = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_extra_flowers_correct_l1236_123632


namespace NUMINAMATH_GPT_match_Tile_C_to_Rectangle_III_l1236_123665

-- Define the structure for a Tile
structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the given tiles
def Tile_A : Tile := { top := 5, right := 3, bottom := 7, left := 2 }
def Tile_B : Tile := { top := 3, right := 6, bottom := 2, left := 8 }
def Tile_C : Tile := { top := 7, right := 9, bottom := 1, left := 3 }
def Tile_D : Tile := { top := 1, right := 8, bottom := 5, left := 9 }

-- The proof problem: Prove that Tile C should be matched to Rectangle III
theorem match_Tile_C_to_Rectangle_III : (Tile_C = { top := 7, right := 9, bottom := 1, left := 3 }) → true := 
by
  intros
  sorry

end NUMINAMATH_GPT_match_Tile_C_to_Rectangle_III_l1236_123665


namespace NUMINAMATH_GPT_card_prob_ace_of_hearts_l1236_123643

def problem_card_probability : Prop :=
  let deck_size := 52
  let draw_size := 2
  let ace_hearts := 1
  let total_combinations := Nat.choose deck_size draw_size
  let favorable_combinations := deck_size - ace_hearts
  let probability := favorable_combinations / total_combinations
  probability = 1 / 26

theorem card_prob_ace_of_hearts : problem_card_probability := by
  sorry

end NUMINAMATH_GPT_card_prob_ace_of_hearts_l1236_123643


namespace NUMINAMATH_GPT_toys_secured_in_25_minutes_l1236_123663

def net_toy_gain_per_minute (toys_mom_puts : ℕ) (toys_mia_takes : ℕ) : ℕ :=
  toys_mom_puts - toys_mia_takes

def total_minutes (total_toys : ℕ) (toys_mom_puts : ℕ) (toys_mia_takes : ℕ) : ℕ :=
  (total_toys - 1) / net_toy_gain_per_minute toys_mom_puts toys_mia_takes + 1

theorem toys_secured_in_25_minutes :
  total_minutes 50 5 3 = 25 :=
by
  sorry

end NUMINAMATH_GPT_toys_secured_in_25_minutes_l1236_123663


namespace NUMINAMATH_GPT_arithmetic_seq_k_l1236_123658

theorem arithmetic_seq_k (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ) 
  (h1 : a 1 = -3)
  (h2 : a (k + 1) = 3 / 2)
  (h3 : S k = -12)
  (h4 : ∀ n, S n = n * (a 1 + a (n+1)) / 2):
  k = 13 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_k_l1236_123658


namespace NUMINAMATH_GPT_tiles_needed_to_cover_floor_l1236_123608

/-- 
A floor 10 feet by 15 feet is to be tiled with 3-inch-by-9-inch tiles. 
This theorem verifies that the necessary number of tiles is 800. 
-/
theorem tiles_needed_to_cover_floor
  (floor_length : ℝ)
  (floor_width : ℝ)
  (tile_length_inch : ℝ)
  (tile_width_inch : ℝ)
  (conversion_factor : ℝ)
  (num_tiles : ℕ) 
  (h_floor_length : floor_length = 10)
  (h_floor_width : floor_width = 15)
  (h_tile_length_inch : tile_length_inch = 3)
  (h_tile_width_inch : tile_width_inch = 9)
  (h_conversion_factor : conversion_factor = 12)
  (h_num_tiles : num_tiles = 800) :
  (floor_length * floor_width) / ((tile_length_inch / conversion_factor) * (tile_width_inch / conversion_factor)) = num_tiles :=
by
  -- The proof is not included, using sorry to mark this part
  sorry

end NUMINAMATH_GPT_tiles_needed_to_cover_floor_l1236_123608


namespace NUMINAMATH_GPT_car_drive_distance_l1236_123693

-- Define the conditions as constants
def driving_speed : ℕ := 8 -- miles per hour
def driving_hours_before_cool : ℕ := 5 -- hours of constant driving
def cooling_hours : ℕ := 1 -- hours needed for cooling down
def total_time : ℕ := 13 -- hours available

-- Define the calculation for distance driven in cycles
def distance_per_cycle : ℕ := driving_speed * driving_hours_before_cool

-- Calculate the duration of one complete cycle
def cycle_duration : ℕ := driving_hours_before_cool + cooling_hours

-- Theorem statement: the car can drive 88 miles in 13 hours
theorem car_drive_distance : distance_per_cycle * (total_time / cycle_duration) + driving_speed * (total_time % cycle_duration) = 88 :=
by
  sorry

end NUMINAMATH_GPT_car_drive_distance_l1236_123693


namespace NUMINAMATH_GPT_prime_divisor_form_l1236_123625

theorem prime_divisor_form (n : ℕ) (q : ℕ) (hq : (2^(2^n) + 1) % q = 0) (prime_q : Nat.Prime q) :
  ∃ k : ℕ, q = 2^(n+1) * k + 1 :=
sorry

end NUMINAMATH_GPT_prime_divisor_form_l1236_123625


namespace NUMINAMATH_GPT_min_perimeter_is_676_l1236_123615

-- Definitions and conditions based on the problem statement
def equal_perimeter (a b c : ℕ) : Prop :=
  2 * a + 14 * c = 2 * b + 16 * c

def equal_area (a b c : ℕ) : Prop :=
  7 * Real.sqrt (a^2 - 49 * c^2) = 8 * Real.sqrt (b^2 - 64 * c^2)

def base_ratio (b : ℕ) : ℕ := b * 8 / 7

theorem min_perimeter_is_676 :
  ∃ a b c : ℕ, equal_perimeter a b c ∧ equal_area a b c ∧ base_ratio b = a - b ∧ 
  2 * a + 14 * c = 676 :=
sorry

end NUMINAMATH_GPT_min_perimeter_is_676_l1236_123615


namespace NUMINAMATH_GPT_find_certain_number_l1236_123696

theorem find_certain_number (x : ℕ) (h : 220025 = (x + 445) * (2 * (x - 445)) + 25) : x = 555 :=
sorry

end NUMINAMATH_GPT_find_certain_number_l1236_123696


namespace NUMINAMATH_GPT_correct_quotient_is_32_l1236_123683

-- Definitions based on the conditions
def incorrect_divisor := 12
def correct_divisor := 21
def incorrect_quotient := 56
def dividend := incorrect_divisor * incorrect_quotient -- Given as 672

-- Statement of the theorem
theorem correct_quotient_is_32 :
  dividend / correct_divisor = 32 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_correct_quotient_is_32_l1236_123683


namespace NUMINAMATH_GPT_cookie_count_l1236_123633

theorem cookie_count (C : ℕ) 
  (h1 : 3 * C / 4 + 1 * (C / 4) / 5 + 1 * (C / 4) * 4 / 20 = 10) 
  (h2: 1 * (5 * 4 / 20) / 10 = 1): 
  C = 100 :=
by 
sorry

end NUMINAMATH_GPT_cookie_count_l1236_123633


namespace NUMINAMATH_GPT_surface_area_of_4cm_cube_after_corner_removal_l1236_123679

noncomputable def surface_area_after_corner_removal (cube_side original_surface_length corner_cube_side : ℝ) : ℝ := 
  let num_faces : ℕ := 6
  let num_corners : ℕ := 8
  let surface_area_one_face := cube_side * cube_side
  let original_surface_area := num_faces * surface_area_one_face
  let corner_surface_area_one_face := 3 * (corner_cube_side * corner_cube_side)
  let exposed_surface_area_one_face := 3 * (corner_cube_side * corner_cube_side)
  let net_change_per_corner_cube := -corner_surface_area_one_face + exposed_surface_area_one_face
  let total_change := num_corners * net_change_per_corner_cube
  original_surface_area + total_change

theorem surface_area_of_4cm_cube_after_corner_removal : 
  ∀ (cube_side original_surface_length corner_cube_side : ℝ), 
  cube_side = 4 ∧ original_surface_length = 4 ∧ corner_cube_side = 2 →
  surface_area_after_corner_removal cube_side original_surface_length corner_cube_side = 96 :=
by
  intros cube_side original_surface_length corner_cube_side h
  rcases h with ⟨hs, ho, hc⟩
  rw [hs, ho, hc]
  sorry

end NUMINAMATH_GPT_surface_area_of_4cm_cube_after_corner_removal_l1236_123679


namespace NUMINAMATH_GPT_number_of_girls_in_school_l1236_123644

theorem number_of_girls_in_school (total_students : ℕ) (sample_size : ℕ) (x : ℕ) :
  total_students = 2400 →
  sample_size = 200 →
  2 * x + 10 = sample_size →
  (95 / 200 : ℚ) * (total_students : ℚ) = 1140 :=
by
  intros h_total h_sample h_sampled
  rw [h_total, h_sample] at *
  sorry

end NUMINAMATH_GPT_number_of_girls_in_school_l1236_123644


namespace NUMINAMATH_GPT_identify_nearly_regular_polyhedra_l1236_123623

structure Polyhedron :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

def nearlyRegularPolyhedra : List Polyhedron :=
  [ 
    ⟨8, 12, 6⟩,   -- Properties of Tetrahedron-octahedron intersection
    ⟨14, 24, 12⟩, -- Properties of Cuboctahedron
    ⟨32, 60, 30⟩  -- Properties of Dodecahedron-Icosahedron
  ]

theorem identify_nearly_regular_polyhedra :
  nearlyRegularPolyhedra = [
    ⟨8, 12, 6⟩,  -- Tetrahedron-octahedron intersection
    ⟨14, 24, 12⟩, -- Cuboctahedron
    ⟨32, 60, 30⟩  -- Dodecahedron-icosahedron intersection
  ] :=
by
  sorry

end NUMINAMATH_GPT_identify_nearly_regular_polyhedra_l1236_123623


namespace NUMINAMATH_GPT_condition_A_is_necessary_but_not_sufficient_for_condition_B_l1236_123664

-- Define conditions
variables (a b : ℝ)

-- Condition A: ab > 0
def condition_A : Prop := a * b > 0

-- Condition B: a > 0 and b > 0
def condition_B : Prop := a > 0 ∧ b > 0

-- Prove that condition_A is a necessary but not sufficient condition for condition_B
theorem condition_A_is_necessary_but_not_sufficient_for_condition_B :
  (condition_A a b → condition_B a b) ∧ ¬(condition_B a b → condition_A a b) :=
by
  sorry

end NUMINAMATH_GPT_condition_A_is_necessary_but_not_sufficient_for_condition_B_l1236_123664


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1236_123695

theorem quadratic_inequality_solution_set {x : ℝ} :
  (x^2 + x - 2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1236_123695


namespace NUMINAMATH_GPT_power_function_zeros_l1236_123605

theorem power_function_zeros :
  ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = x ^ 3) ∧ (f 2 = 8) ∧ (∀ y : ℝ, (f y - y = 0) ↔ (y = 0 ∨ y = 1 ∨ y = -1)) := by
  sorry

end NUMINAMATH_GPT_power_function_zeros_l1236_123605


namespace NUMINAMATH_GPT_find_constant_l1236_123659

theorem find_constant (N : ℝ) (C : ℝ) (h1 : N = 12.0) (h2 : C + 0.6667 * N = 0.75 * N) : C = 0.9996 :=
by
  sorry

end NUMINAMATH_GPT_find_constant_l1236_123659


namespace NUMINAMATH_GPT_probability_remainder_is_4_5_l1236_123630

def probability_remainder_1 (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 2020 → (N^16 % 5 = 1)

theorem probability_remainder_is_4_5 : 
  ∀ N, N ≥ 1 ∧ N ≤ 2020 → (N^16 % 5 = 1) → (number_of_successful_outcomes / total_outcomes = 4 / 5) :=
sorry

end NUMINAMATH_GPT_probability_remainder_is_4_5_l1236_123630


namespace NUMINAMATH_GPT_race_distance_l1236_123634

/-- Given that Sasha, Lesha, and Kolya start a 100m race simultaneously and run at constant velocities,
when Sasha finishes, Lesha is 10m behind, and when Lesha finishes, Kolya is 10m behind.
Prove that the distance between Sasha and Kolya when Sasha finishes is 19 meters. -/
theorem race_distance
    (v_S v_L v_K : ℝ)
    (h1 : 100 / v_S - 100 / v_L = 10 / v_L)
    (h2 : 100 / v_L - 100 / v_K = 10 / v_K) :
    100 - 81 = 19 :=
by
  sorry

end NUMINAMATH_GPT_race_distance_l1236_123634


namespace NUMINAMATH_GPT_sector_angle_degree_measure_l1236_123660

-- Define the variables and conditions
variables (θ r : ℝ)
axiom h1 : (1 / 2) * θ * r^2 = 1
axiom h2 : 2 * r + θ * r = 4

-- Define the theorem to be proved
theorem sector_angle_degree_measure (θ r : ℝ) (h1 : (1 / 2) * θ * r^2 = 1) (h2 : 2 * r + θ * r = 4) : θ = 2 :=
sorry

end NUMINAMATH_GPT_sector_angle_degree_measure_l1236_123660


namespace NUMINAMATH_GPT_cost_first_third_hour_l1236_123610

theorem cost_first_third_hour 
  (c : ℝ) 
  (h1 : 0 < c) 
  (h2 : ∀ t : ℝ, t > 1/4 → (t - 1/4) * 12 + c = 31)
  : c = 5 :=
by
  sorry

end NUMINAMATH_GPT_cost_first_third_hour_l1236_123610


namespace NUMINAMATH_GPT_decimal_to_fraction_l1236_123678

theorem decimal_to_fraction : 2.36 = 59 / 25 :=
by
  sorry

end NUMINAMATH_GPT_decimal_to_fraction_l1236_123678


namespace NUMINAMATH_GPT_starting_player_wins_by_taking_2_white_first_l1236_123618

-- Define initial setup
def initial_blue_balls : ℕ := 15
def initial_white_balls : ℕ := 12

-- Define conditions of the game
def can_take_blue_balls (n : ℕ) : Prop := n % 3 = 0
def can_take_white_balls (n : ℕ) : Prop := n % 2 = 0
def player_win_condition (blue white : ℕ) : Prop := 
  (blue = 0 ∧ white = 0)

-- Define the game strategy to establish and maintain the ratio 3/2
def maintain_ratio (blue white : ℕ) : Prop := blue * 2 = white * 3

-- Prove that the starting player should take 2 white balls first to ensure winning
theorem starting_player_wins_by_taking_2_white_first :
  (can_take_white_balls 2) →
  maintain_ratio initial_blue_balls (initial_white_balls - 2) →
  ∀ (blue white : ℕ), player_win_condition blue white :=
by
  intros h_take_white h_maintain_ratio blue white
  sorry

end NUMINAMATH_GPT_starting_player_wins_by_taking_2_white_first_l1236_123618


namespace NUMINAMATH_GPT_range_of_m_l1236_123637

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + (m + 2) * x + (m + 5) = 0 → 0 < x) → (-5 < m ∧ m ≤ -4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1236_123637


namespace NUMINAMATH_GPT_sum_infinite_geometric_series_l1236_123607

theorem sum_infinite_geometric_series : 
  let a : ℝ := 2
  let r : ℝ := -5/8
  a / (1 - r) = 16/13 :=
by
  sorry

end NUMINAMATH_GPT_sum_infinite_geometric_series_l1236_123607


namespace NUMINAMATH_GPT_next_year_multiple_of_6_8_9_l1236_123611

theorem next_year_multiple_of_6_8_9 (n : ℕ) (h₀ : n = 2016) (h₁ : n % 6 = 0) (h₂ : n % 8 = 0) (h₃ : n % 9 = 0) : ∃ m > n, m % 6 = 0 ∧ m % 8 = 0 ∧ m % 9 = 0 ∧ m = 2088 :=
by
  sorry

end NUMINAMATH_GPT_next_year_multiple_of_6_8_9_l1236_123611


namespace NUMINAMATH_GPT_range_of_m_l1236_123656

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, x^2 - x - m = 0) : m ≥ -1/4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1236_123656


namespace NUMINAMATH_GPT_circle_radius_l1236_123612

theorem circle_radius (x y : ℝ) : (x^2 + y^2 + 2*x = 0) → ∃ r, r = 1 :=
by sorry

end NUMINAMATH_GPT_circle_radius_l1236_123612


namespace NUMINAMATH_GPT_find_triples_l1236_123627

theorem find_triples :
  { (a, b, c) : ℕ × ℕ × ℕ | (c-1) * (a * b - b - a) = a + b - 2 } =
  { (2, 1, 0), (1, 2, 0), (3, 4, 2), (4, 3, 2), (1, 0, 2), (0, 1, 2), (2, 4, 3), (4, 2, 3) } :=
by
  sorry

end NUMINAMATH_GPT_find_triples_l1236_123627


namespace NUMINAMATH_GPT_winning_candidate_percentage_l1236_123626

def percentage_votes (votes1 votes2 votes3 : ℕ) : ℚ := 
  let total_votes := votes1 + votes2 + votes3
  let winning_votes := max (max votes1 votes2) votes3
  (winning_votes * 100) / total_votes

theorem winning_candidate_percentage :
  percentage_votes 3000 5000 15000 = (15000 * 100) / (3000 + 5000 + 15000) :=
by 
  -- This computation should give us the exact percentage fraction.
  -- Simplifying it would yield the result approximately 65.22%
  -- Proof steps can be provided here.
  sorry

end NUMINAMATH_GPT_winning_candidate_percentage_l1236_123626


namespace NUMINAMATH_GPT_original_number_is_correct_l1236_123685

theorem original_number_is_correct (x : ℝ) (h : 10 * x = x + 34.65) : x = 3.85 :=
sorry

end NUMINAMATH_GPT_original_number_is_correct_l1236_123685


namespace NUMINAMATH_GPT_continuity_at_three_l1236_123650

noncomputable def f (x : ℝ) : ℝ := -2 * x ^ 2 - 4

theorem continuity_at_three (ε : ℝ) (hε : 0 < ε) :
  ∃ δ > 0, ∀ x : ℝ, |x - 3| < δ → |f x - f 3| < ε :=
sorry

end NUMINAMATH_GPT_continuity_at_three_l1236_123650


namespace NUMINAMATH_GPT_number_of_seniors_in_statistics_l1236_123690

theorem number_of_seniors_in_statistics (total_students : ℕ) (half_enrolled_in_statistics : ℕ) (percentage_seniors : ℚ) (students_in_statistics seniors_in_statistics : ℕ) 
(h1 : total_students = 120)
(h2 : half_enrolled_in_statistics = total_students / 2)
(h3 : students_in_statistics = half_enrolled_in_statistics)
(h4 : percentage_seniors = 0.90)
(h5 : seniors_in_statistics = students_in_statistics * percentage_seniors) : 
seniors_in_statistics = 54 := 
by sorry

end NUMINAMATH_GPT_number_of_seniors_in_statistics_l1236_123690


namespace NUMINAMATH_GPT_bruce_goals_l1236_123635

theorem bruce_goals (B M : ℕ) (h1 : M = 3 * B) (h2 : B + M = 16) : B = 4 :=
by {
  -- Omitted proof
  sorry
}

end NUMINAMATH_GPT_bruce_goals_l1236_123635


namespace NUMINAMATH_GPT_find_primes_a_l1236_123675

theorem find_primes_a :
  ∀ (a : ℕ), (∀ n : ℕ, n < a → Nat.Prime (4 * n * n + a)) → (a = 3 ∨ a = 7) :=
by
  sorry

end NUMINAMATH_GPT_find_primes_a_l1236_123675


namespace NUMINAMATH_GPT_max_good_triplets_l1236_123648

-- Define the problem's conditions
variables (k : ℕ) (h_pos : 0 < k)

-- The statement to be proven
theorem max_good_triplets : ∃ T, T = 12 * k ^ 4 := 
sorry

end NUMINAMATH_GPT_max_good_triplets_l1236_123648


namespace NUMINAMATH_GPT_power_point_relative_to_circle_l1236_123621

noncomputable def circle_power (a b R x1 y1 : ℝ) : ℝ :=
  (x1 - a) ^ 2 + (y1 - b) ^ 2 - R ^ 2

theorem power_point_relative_to_circle (a b R x1 y1 : ℝ) :
  (x1 - a) ^ 2 + (y1 - b) ^ 2 - R ^ 2 = circle_power a b R x1 y1 := by
  unfold circle_power
  sorry

end NUMINAMATH_GPT_power_point_relative_to_circle_l1236_123621


namespace NUMINAMATH_GPT_inequality_solution_l1236_123631

theorem inequality_solution :
  ∀ x : ℝ, ( (x - 3) / ( (x - 2) ^ 2 ) < 0 ) ↔ ( x < 2 ∨ (2 < x ∧ x < 3) ) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1236_123631


namespace NUMINAMATH_GPT_find_j_of_scaled_quadratic_l1236_123682

/- Define the given condition -/
def quadratic_expressed (p q r : ℝ) : Prop :=
  ∀ x : ℝ, p * x^2 + q * x + r = 5 * (x - 3)^2 + 15

/- State the theorem to be proved -/
theorem find_j_of_scaled_quadratic (p q r m j l : ℝ) (h_quad : quadratic_expressed p q r) :
  (∀ x : ℝ, 2 * p * x^2 + 2 * q * x + 2 * r = m * (x - j)^2 + l) → j = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_j_of_scaled_quadratic_l1236_123682


namespace NUMINAMATH_GPT_number_power_eq_l1236_123676

theorem number_power_eq (x : ℕ) (h : x^10 = 16^5) : x = 4 :=
by {
  -- Add supporting calculations here if needed
  sorry
}

end NUMINAMATH_GPT_number_power_eq_l1236_123676


namespace NUMINAMATH_GPT_black_squares_count_l1236_123606

def checkerboard_size : Nat := 32
def total_squares : Nat := checkerboard_size * checkerboard_size
def black_squares (n : Nat) : Nat := n / 2

theorem black_squares_count : black_squares total_squares = 512 := by
  let n := total_squares
  show black_squares n = 512
  sorry

end NUMINAMATH_GPT_black_squares_count_l1236_123606


namespace NUMINAMATH_GPT_sand_removal_l1236_123672

theorem sand_removal :
  let initial_weight := (8 / 3 : ℚ)
  let first_removal := (1 / 4 : ℚ)
  let second_removal := (5 / 6 : ℚ)
  initial_weight - (first_removal + second_removal) = (13 / 12 : ℚ) := by
  -- sorry is used here to skip the proof as instructed
  sorry

end NUMINAMATH_GPT_sand_removal_l1236_123672


namespace NUMINAMATH_GPT_calculate_expression_l1236_123680

theorem calculate_expression (y : ℤ) (hy : y = 2) : (3 * y + 4)^2 = 100 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1236_123680


namespace NUMINAMATH_GPT_netSalePrice_correct_l1236_123609

-- Definitions for item costs and fees
def purchaseCostA : ℝ := 650
def handlingFeeA : ℝ := 0.02 * purchaseCostA
def totalCostA : ℝ := purchaseCostA + handlingFeeA

def purchaseCostB : ℝ := 350
def restockingFeeB : ℝ := 0.03 * purchaseCostB
def totalCostB : ℝ := purchaseCostB + restockingFeeB

def purchaseCostC : ℝ := 400
def transportationFeeC : ℝ := 0.015 * purchaseCostC
def totalCostC : ℝ := purchaseCostC + transportationFeeC

-- Desired profit percentages
def profitPercentageA : ℝ := 0.40
def profitPercentageB : ℝ := 0.25
def profitPercentageC : ℝ := 0.30

-- Net sale prices for achieving the desired profit percentages
def netSalePriceA : ℝ := totalCostA + (profitPercentageA * totalCostA)
def netSalePriceB : ℝ := totalCostB + (profitPercentageB * totalCostB)
def netSalePriceC : ℝ := totalCostC + (profitPercentageC * totalCostC)

-- Expected values
def expectedNetSalePriceA : ℝ := 928.20
def expectedNetSalePriceB : ℝ := 450.63
def expectedNetSalePriceC : ℝ := 527.80

-- Theorem to prove the net sale prices match the expected values
theorem netSalePrice_correct :
  netSalePriceA = expectedNetSalePriceA ∧
  netSalePriceB = expectedNetSalePriceB ∧
  netSalePriceC = expectedNetSalePriceC :=
by
  unfold netSalePriceA netSalePriceB netSalePriceC totalCostA totalCostB totalCostC
         handlingFeeA restockingFeeB transportationFeeC
  sorry

end NUMINAMATH_GPT_netSalePrice_correct_l1236_123609


namespace NUMINAMATH_GPT_express_as_terminating_decimal_l1236_123653

section terminating_decimal

theorem express_as_terminating_decimal
  (a b : ℚ)
  (h1 : a = 125)
  (h2 : b = 144)
  (h3 : b = 2^4 * 3^2): 
  a / b = 0.78125 := 
by 
  sorry

end terminating_decimal

end NUMINAMATH_GPT_express_as_terminating_decimal_l1236_123653


namespace NUMINAMATH_GPT_pow_two_gt_cube_l1236_123689

theorem pow_two_gt_cube (n : ℕ) (h : 10 ≤ n) : 2^n > n^3 := sorry

end NUMINAMATH_GPT_pow_two_gt_cube_l1236_123689


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1236_123670

theorem boat_speed_in_still_water 
  (rate_of_current : ℝ) 
  (time_in_hours : ℝ) 
  (distance_downstream : ℝ)
  (h_rate : rate_of_current = 5) 
  (h_time : time_in_hours = 15 / 60) 
  (h_distance : distance_downstream = 6.25) : 
  ∃ x : ℝ, (distance_downstream = (x + rate_of_current) * time_in_hours) ∧ x = 20 :=
by 
  -- Main theorem statement, proof omitted for brevity.
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1236_123670


namespace NUMINAMATH_GPT_ladder_distance_from_wall_l1236_123614

theorem ladder_distance_from_wall (θ : ℝ) (L : ℝ) (d : ℝ) 
  (h_angle : θ = 60) (h_length : L = 19) (h_cos : Real.cos (θ * Real.pi / 180) = 0.5) : 
  d = 9.5 :=
by
  sorry

end NUMINAMATH_GPT_ladder_distance_from_wall_l1236_123614


namespace NUMINAMATH_GPT_find_a7_l1236_123622

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def Sn_for_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem find_a7 (h_arith : arithmetic_sequence a)
  (h_sum_property : Sn_for_arithmetic_sequence a S)
  (h1 : a 2 + a 5 = 4)
  (h2 : S 7 = 21) :
  a 7 = 9 :=
sorry

end NUMINAMATH_GPT_find_a7_l1236_123622


namespace NUMINAMATH_GPT_columbus_discovered_america_in_1492_l1236_123617

theorem columbus_discovered_america_in_1492 :
  ∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x ≠ 1 ∧ y ≠ 1 ∧ z ≠ 1 ∧
  1 + x + y + z = 16 ∧ y + 1 = 5 * z ∧
  1000 + 100 * x + 10 * y + z = 1492 :=
by
  sorry

end NUMINAMATH_GPT_columbus_discovered_america_in_1492_l1236_123617


namespace NUMINAMATH_GPT_quadratic_roots_quadratic_roots_one_quadratic_roots_two_l1236_123604

open scoped Classical

variables {p : Type*} [Field p] {a b c x : p}

theorem quadratic_roots (h_a : a ≠ 0) :
  (¬ ∃ y : p, y^2 = b^2 - 4 * a * c) → ∀ x : p, ¬ a * x^2 + b * x + c = 0 :=
by sorry

theorem quadratic_roots_one (h_a : a ≠ 0) :
  (b^2 - 4 * a * c = 0) → ∃ x : p, a * x^2 + b * x + c = 0 ∧ ∀ y : p, a * y^2 + b * y + c = 0 → y = x :=
by sorry

theorem quadratic_roots_two (h_a : a ≠ 0) :
  (∃ y : p, y^2 = b^2 - 4 * a * c) ∧ (b^2 - 4 * a * c ≠ 0) → ∃ x1 x2 : p, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_quadratic_roots_one_quadratic_roots_two_l1236_123604


namespace NUMINAMATH_GPT_eunseo_change_correct_l1236_123638

-- Define the given values
def r : ℕ := 3
def p_r : ℕ := 350
def b : ℕ := 2
def p_b : ℕ := 180
def P : ℕ := 2000

-- Define the total cost of candies and the change
def total_cost := r * p_r + b * p_b
def change := P - total_cost

-- Theorem statement
theorem eunseo_change_correct : change = 590 := by
  -- proof not required, so using sorry
  sorry

end NUMINAMATH_GPT_eunseo_change_correct_l1236_123638


namespace NUMINAMATH_GPT_original_radius_of_cylinder_in_inches_l1236_123692

theorem original_radius_of_cylinder_in_inches
  (r : ℝ) (h : ℝ) (V : ℝ → ℝ → ℝ → ℝ) 
  (h_increased_radius : V (r + 4) h π = V r (h + 4) π) 
  (h_original_height : h = 3) :
  r = 8 :=
by
  sorry

end NUMINAMATH_GPT_original_radius_of_cylinder_in_inches_l1236_123692


namespace NUMINAMATH_GPT_farmland_acres_l1236_123616

theorem farmland_acres (x y : ℝ) 
  (h1 : x + y = 100) 
  (h2 : 300 * x + (500 / 7) * y = 10000) : 
  true :=
sorry

end NUMINAMATH_GPT_farmland_acres_l1236_123616


namespace NUMINAMATH_GPT_can_still_row_probability_l1236_123677

/-- Define the probabilities for the left and right oars --/
def P_left1_work : ℚ := 3 / 5
def P_left2_work : ℚ := 2 / 5
def P_right1_work : ℚ := 4 / 5 
def P_right2_work : ℚ := 3 / 5

/-- Define the probabilities of the failures as complementary probabilities --/
def P_left1_fail : ℚ := 1 - P_left1_work
def P_left2_fail : ℚ := 1 - P_left2_work
def P_right1_fail : ℚ := 1 - P_right1_work
def P_right2_fail : ℚ := 1 - P_right2_work

/-- Define the probability of both left oars failing --/
def P_both_left_fail : ℚ := P_left1_fail * P_left2_fail

/-- Define the probability of both right oars failing --/
def P_both_right_fail : ℚ := P_right1_fail * P_right2_fail

/-- Define the probability of all four oars failing --/
def P_all_fail : ℚ := P_both_left_fail * P_both_right_fail

/-- Calculate the probability that at least one oar on each side works --/
def P_can_row : ℚ := 1 - (P_both_left_fail + P_both_right_fail - P_all_fail)

theorem can_still_row_probability :
  P_can_row = 437 / 625 :=
by {
  -- The proof is to be completed
  sorry
}

end NUMINAMATH_GPT_can_still_row_probability_l1236_123677


namespace NUMINAMATH_GPT_ellipse_focal_distance_l1236_123681

theorem ellipse_focal_distance :
  let a := 9
  let b := 5
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 4 * Real.sqrt 14 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_focal_distance_l1236_123681


namespace NUMINAMATH_GPT_roots_opposite_eq_minus_one_l1236_123655

theorem roots_opposite_eq_minus_one (k : ℝ) 
  (h_real_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ + x₂ = 0 ∧ x₁ * x₂ = k + 1) :
  k = -1 :=
by
  sorry

end NUMINAMATH_GPT_roots_opposite_eq_minus_one_l1236_123655


namespace NUMINAMATH_GPT_Jake_has_fewer_peaches_l1236_123647

def Steven_peaches := 14
def Jill_peaches := 5
def Jake_peaches := Jill_peaches + 3

theorem Jake_has_fewer_peaches : Steven_peaches - Jake_peaches = 6 :=
by
  sorry

end NUMINAMATH_GPT_Jake_has_fewer_peaches_l1236_123647
