import Mathlib

namespace train_distance_covered_l478_47846

-- Definitions based on the given conditions
def average_speed := 3   -- in meters per second
def total_time := 9      -- in seconds

-- Theorem statement: Given the average speed and total time, the total distance covered is 27 meters
theorem train_distance_covered : average_speed * total_time = 27 := 
by
  sorry

end train_distance_covered_l478_47846


namespace find_point_B_find_line_BC_l478_47838

-- Define the coordinates of point A
def point_A : ℝ × ℝ := (2, -1)

-- Define the equation of the median on side AB
def median_AB (x y : ℝ) : Prop := x + 3 * y = 6

-- Define the equation of the internal angle bisector of ∠ABC
def bisector_BC (x y : ℝ) : Prop := x - y = -1

-- Prove the coordinates of point B
theorem find_point_B :
  (a b : ℝ) →
  (median_AB ((a + 2) / 2) ((b - 1) / 2)) →
  (a - b = -1) →
  a = 5 / 2 ∧ b = 7 / 2 :=
sorry

-- Define the line equation BC
def line_BC (x y : ℝ) : Prop := x - 9 * y + 29 = 0

-- Prove the equation of the line containing side BC
theorem find_line_BC :
  (x0 y0 : ℝ) →
  bisector_BC x0 y0 →
  (x0, y0) = (-2, 3) →
  line_BC x0 y0 :=
sorry

end find_point_B_find_line_BC_l478_47838


namespace simplify_div_expression_l478_47888

theorem simplify_div_expression (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (x - 1) / (x^2 + 2 * x + 1) / (1 - 2 / (x + 1)) = Real.sqrt 3 / 3 :=
sorry

end simplify_div_expression_l478_47888


namespace maximum_perimeter_triangle_area_l478_47849

-- Part 1: Maximum Perimeter
theorem maximum_perimeter (a b c : ℝ) (A B C : ℝ) 
  (h_c : c = 2) 
  (h_C : C = Real.pi / 3) :
  (a + b + c) ≤ 6 :=
sorry

-- Part 2: Area under given trigonometric condition
theorem triangle_area (A B C a b c : ℝ) 
  (h_c : 2 * Real.sin (2 * A) + Real.sin (2 * B + C) = Real.sin C) :
  (1/2 * a * b * Real.sin C) = (2 * Real.sqrt 6) / 3 :=
sorry

end maximum_perimeter_triangle_area_l478_47849


namespace fraction_expression_l478_47890

theorem fraction_expression :
  ((3 / 7) + (5 / 8)) / ((5 / 12) + (2 / 9)) = (531 / 322) :=
by
  sorry

end fraction_expression_l478_47890


namespace eliot_votes_l478_47878

theorem eliot_votes (randy_votes shaun_votes eliot_votes : ℕ)
                    (h1 : randy_votes = 16)
                    (h2 : shaun_votes = 5 * randy_votes)
                    (h3 : eliot_votes = 2 * shaun_votes) :
                    eliot_votes = 160 :=
by {
  -- Proof will be conducted here
  sorry
}

end eliot_votes_l478_47878


namespace find_original_price_l478_47891

-- Define the original price and conditions
def original_price (P : ℝ) : Prop :=
  ∃ discount final_price, discount = 0.55 ∧ final_price = 450000 ∧ ((1 - discount) * P = final_price)

-- The theorem to prove the original price before discount
theorem find_original_price (P : ℝ) (h : original_price P) : P = 1000000 :=
by
  sorry

end find_original_price_l478_47891


namespace range_of_a_l478_47864

noncomputable def min_expr (x: ℝ) : ℝ := x + 2/(x - 2)

theorem range_of_a (a: ℝ) : 
  (∀ x > 2, a ≤ min_expr x) ↔ a ≤ 2 + 2 * Real.sqrt 2 := 
by
  sorry

end range_of_a_l478_47864


namespace solution_sets_l478_47874

-- These are the hypotheses derived from the problem conditions.
structure Conditions (a b c d : ℕ) : Prop :=
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (positive_even : ∃ u v w x : ℕ, a = 2*u ∧ b = 2*v ∧ c = 2*w ∧ d = 2*x ∧ 
                   u > 0 ∧ v > 0 ∧ w > 0 ∧ x > 0)
  (sum_100 : a + b + c + d = 100)
  (third_fourth_single_digit : c < 20 ∧ d < 20)
  (sum_2000 : 12 * a + 30 * b + 52 * c = 2000)

-- The main theorem in Lean asserting that these are the only possible sets of numbers.
theorem solution_sets :
  ∃ (a b c d : ℕ), Conditions a b c d ∧
  ( 
    (a = 62 ∧ b = 14 ∧ c = 4 ∧ d = 1) ∨ 
    (a = 48 ∧ b = 22 ∧ c = 2 ∧ d = 3)
  ) :=
  sorry

end solution_sets_l478_47874


namespace f_is_monotonic_decreasing_l478_47805

noncomputable def f (x : ℝ) : ℝ := Real.sin (1/2 * x + Real.pi / 6)

theorem f_is_monotonic_decreasing : ∀ x y : ℝ, (2 * Real.pi / 3 ≤ x ∧ x ≤ 8 * Real.pi / 3) → (2 * Real.pi / 3 ≤ y ∧ y ≤ 8 * Real.pi / 3) → x < y → f y ≤ f x :=
sorry

end f_is_monotonic_decreasing_l478_47805


namespace original_cards_l478_47825

-- Define the number of cards Jason gave away
def cards_given_away : ℕ := 9

-- Define the number of cards Jason now has
def cards_now : ℕ := 4

-- Prove the original number of Pokemon cards Jason had
theorem original_cards (x : ℕ) : x = cards_given_away + cards_now → x = 13 :=
by {
    sorry
}

end original_cards_l478_47825


namespace cost_of_bought_movie_l478_47857

theorem cost_of_bought_movie 
  (ticket_cost : ℝ)
  (ticket_count : ℕ)
  (rental_cost : ℝ)
  (total_spent : ℝ)
  (bought_movie_cost : ℝ) :
  ticket_cost = 10.62 →
  ticket_count = 2 →
  rental_cost = 1.59 →
  total_spent = 36.78 →
  bought_movie_cost = total_spent - (ticket_cost * ticket_count + rental_cost) →
  bought_movie_cost = 13.95 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cost_of_bought_movie_l478_47857


namespace penalty_kicks_l478_47836

-- Define the soccer team data
def total_players : ℕ := 16
def goalkeepers : ℕ := 2
def players_shooting : ℕ := total_players - goalkeepers -- 14

-- Function to calculate total penalty kicks
def total_penalty_kicks (total_players goalkeepers : ℕ) : ℕ :=
  let players_shooting := total_players - goalkeepers
  players_shooting * goalkeepers

-- Theorem stating the number of penalty kicks
theorem penalty_kicks : total_penalty_kicks total_players goalkeepers = 30 :=
by
  sorry

end penalty_kicks_l478_47836


namespace inverse_proportion_indeterminate_l478_47803

theorem inverse_proportion_indeterminate (k : ℝ) (x1 x2 y1 y2 : ℝ) (h1 : x1 < x2)
  (h2 : y1 = k / x1) (h3 : y2 = k / x2) : 
  (y1 > 0 ∧ y2 > 0) ∨ (y1 < 0 ∧ y2 < 0) ∨ (y1 * y2 < 0) → false :=
sorry

end inverse_proportion_indeterminate_l478_47803


namespace arithmetic_sequence_sum_l478_47810

theorem arithmetic_sequence_sum (a_n : ℕ → ℝ) (h1 : a_n 1 + a_n 2 + a_n 3 + a_n 4 = 30) 
                               (h2 : a_n 1 + a_n 4 = a_n 2 + a_n 3) :
  a_n 2 + a_n 3 = 15 := 
by 
  sorry

end arithmetic_sequence_sum_l478_47810


namespace max_right_angle_triangles_in_pyramid_l478_47877

noncomputable def pyramid_max_right_angle_triangles : Nat :=
  let pyramid : Type := { faces : Nat // faces = 4 }
  1

theorem max_right_angle_triangles_in_pyramid (p : pyramid) : pyramid_max_right_angle_triangles = 1 :=
  sorry

end max_right_angle_triangles_in_pyramid_l478_47877


namespace geometric_sequence_value_l478_47875

theorem geometric_sequence_value 
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_nonzero_diff : d ≠ 0)
  (h_condition : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0)
  (h_geom_seq : ∀ n, b (n + 1) = b n * (b 1 / b 0))
  (h_b7_eq_a7 : b 7 = a 7) :
  b 6 * b 8 = 16 :=
sorry

end geometric_sequence_value_l478_47875


namespace max_sqrt_sum_l478_47863

theorem max_sqrt_sum (x y : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hxy : x + y = 8) :
  abs (Real.sqrt (x - 1 / y) + Real.sqrt (y - 1 / x)) ≤ Real.sqrt 15 :=
sorry

end max_sqrt_sum_l478_47863


namespace impossible_15_cents_l478_47824

theorem impossible_15_cents (a b c d : ℕ) (ha : a ≤ 4) (hb : b ≤ 4) (hc : c ≤ 4) (hd : d ≤ 4) (h : a + b + c + d = 4) : 
  1 * a + 5 * b + 10 * c + 25 * d ≠ 15 :=
by
  sorry

end impossible_15_cents_l478_47824


namespace real_part_zero_implies_a_eq_one_l478_47886

open Complex

theorem real_part_zero_implies_a_eq_one (a : ℝ) : 
  (1 + (1 : ℂ) * I) * (1 + a * I) = 0 ↔ a = 1 := by
  sorry

end real_part_zero_implies_a_eq_one_l478_47886


namespace workshop_worker_allocation_l478_47856

theorem workshop_worker_allocation :
  ∃ (x y : ℕ), 
    x + y = 22 ∧
    6 * x = 5 * y ∧
    x = 10 ∧ y = 12 :=
by
  sorry

end workshop_worker_allocation_l478_47856


namespace original_number_of_players_l478_47828

theorem original_number_of_players 
    (n : ℕ) (W : ℕ)
    (h1 : W = n * 112)
    (h2 : W + 110 + 60 = (n + 2) * 106) : 
    n = 7 :=
by
  sorry

end original_number_of_players_l478_47828


namespace seq_eleven_l478_47840

noncomputable def seq (n : ℕ) : ℤ := sorry

axiom seq_add (p q : ℕ) (hp : 0 < p) (hq : 0 < q) : seq (p + q) = seq p + seq q
axiom seq_two : seq 2 = -6

theorem seq_eleven : seq 11 = -33 := by
  sorry

end seq_eleven_l478_47840


namespace good_carrots_l478_47804

-- Definitions
def vanessa_carrots : ℕ := 17
def mother_carrots : ℕ := 14
def bad_carrots : ℕ := 7

-- Proof statement
theorem good_carrots : (vanessa_carrots + mother_carrots) - bad_carrots = 24 := by
  sorry

end good_carrots_l478_47804


namespace probability_of_matching_pair_l478_47835
-- Import the necessary library for probability and combinatorics

def probability_matching_pair (pairs : ℕ) (total_shoes : ℕ) : ℚ :=
  if total_shoes = 2 * pairs then
    (pairs : ℚ) / ((total_shoes * (total_shoes - 1) / 2) : ℚ)
  else 0

theorem probability_of_matching_pair (pairs := 6) (total_shoes := 12) : 
  probability_matching_pair pairs total_shoes = 1 / 11 := 
by
  sorry

end probability_of_matching_pair_l478_47835


namespace vertex_x_coordinate_l478_47850

theorem vertex_x_coordinate (a b c : ℝ) :
  (∀ x, x = 0 ∨ x = 4 ∨ x = 7 →
    (0 ≤ x ∧ x ≤ 7 →
      (x = 0 → c = 1) ∧
      (x = 4 → 16 * a + 4 * b + c = 1) ∧
      (x = 7 → 49 * a + 7 * b + c = 5))) →
  (2 * x = 2 * 2 - b / a) ∧ (0 ≤ x ∧ x ≤ 7) :=
sorry

end vertex_x_coordinate_l478_47850


namespace square_area_max_l478_47841

theorem square_area_max (perimeter : ℝ) (h_perimeter : perimeter = 32) : 
  ∃ (area : ℝ), area = 64 :=
by
  sorry

end square_area_max_l478_47841


namespace compute_d_for_ellipse_l478_47872

theorem compute_d_for_ellipse
  (in_first_quadrant : true)
  (is_tangent_x_axis : true)
  (is_tangent_y_axis : true)
  (focus1 : (ℝ × ℝ) := (5, 4))
  (focus2 : (ℝ × ℝ) := (d, 4)) :
  d = 3.2 := by
  sorry

end compute_d_for_ellipse_l478_47872


namespace maple_tree_taller_than_pine_tree_pine_tree_height_in_one_year_l478_47866

def pine_tree_height : ℚ := 37 / 4  -- 9 1/4 feet
def maple_tree_height : ℚ := 62 / 4  -- 15 1/2 feet (converted directly to common denominator)
def growth_rate : ℚ := 7 / 4  -- 1 3/4 feet per year

theorem maple_tree_taller_than_pine_tree : maple_tree_height - pine_tree_height = 25 / 4 := 
by sorry

theorem pine_tree_height_in_one_year : pine_tree_height + growth_rate = 44 / 4 := 
by sorry

end maple_tree_taller_than_pine_tree_pine_tree_height_in_one_year_l478_47866


namespace find_hours_spent_l478_47802

/-- Let 
  h : ℝ := hours Ed stayed in the hotel last night
  morning_hours : ℝ := 4 -- hours Ed stayed in the hotel this morning
  
  conditions:
  night_cost_per_hour : ℝ := 1.50 -- the cost per hour for staying at night
  morning_cost_per_hour : ℝ := 2 -- the cost per hour for staying in the morning
  initial_amount : ℝ := 80 -- initial amount Ed had
  remaining_amount : ℝ := 63 -- remaining amount after stay
  
  Then the total cost calculated by Ed is:
  total_cost : ℝ := (night_cost_per_hour * h) + (morning_cost_per_hour * morning_hours)
  spent_amount : ℝ := initial_amount - remaining_amount

  We need to prove that h = 6 given the above conditions.
-/
theorem find_hours_spent {h morning_hours night_cost_per_hour morning_cost_per_hour initial_amount remaining_amount total_cost spent_amount : ℝ}
  (hc1 : night_cost_per_hour = 1.50)
  (hc2 : morning_cost_per_hour = 2)
  (hc3 : initial_amount = 80)
  (hc4 : remaining_amount = 63)
  (hc5 : morning_hours = 4)
  (hc6 : spent_amount = initial_amount - remaining_amount)
  (hc7 : total_cost = night_cost_per_hour * h + morning_cost_per_hour * morning_hours)
  (hc8 : spent_amount = 17)
  (hc9 : total_cost = spent_amount) :
  h = 6 :=
by 
  sorry

end find_hours_spent_l478_47802


namespace sum_of_digits_l478_47868

theorem sum_of_digits (a b c d : ℕ) (h1 : a + c = 11) (h2 : b + c = 9) (h3 : a + d = 10) (h_d : d - c = 1) : 
  a + b + c + d = 21 :=
sorry

end sum_of_digits_l478_47868


namespace system_sampling_arithmetic_sequence_l478_47859

theorem system_sampling_arithmetic_sequence :
  ∃ (seq : Fin 5 → ℕ), seq 0 = 8 ∧ seq 3 = 104 ∧ seq 1 = 40 ∧ seq 2 = 72 ∧ seq 4 = 136 ∧ 
    (∀ n m : Fin 5, 0 < n.val - m.val → seq n.val = seq m.val + 32 * (n.val - m.val)) :=
sorry

end system_sampling_arithmetic_sequence_l478_47859


namespace coverage_is_20_l478_47832

noncomputable def cost_per_kg : ℝ := 60
noncomputable def total_cost : ℝ := 1800
noncomputable def side_length : ℝ := 10

-- Surface area of one side of the cube
noncomputable def area_side : ℝ := side_length * side_length

-- Total surface area of the cube
noncomputable def total_area : ℝ := 6 * area_side

-- Kilograms of paint used
noncomputable def kg_paint_used : ℝ := total_cost / cost_per_kg

-- Coverage per kilogram of paint
noncomputable def coverage_per_kg (total_area : ℝ) (kg_paint_used : ℝ) : ℝ := total_area / kg_paint_used

theorem coverage_is_20 : coverage_per_kg total_area kg_paint_used = 20 := by
  sorry

end coverage_is_20_l478_47832


namespace city_cleaning_total_l478_47870

variable (A B C D : ℕ)

theorem city_cleaning_total : 
  A = 54 →
  A = B + 17 →
  C = 2 * B →
  D = A / 3 →
  A + B + C + D = 183 := 
by 
  intros hA hAB hC hD
  sorry

end city_cleaning_total_l478_47870


namespace union_of_sets_l478_47880

def A := { x : ℝ | -1 ≤ x ∧ x < 3 }
def B := { x : ℝ | 2 < x ∧ x ≤ 5 }

theorem union_of_sets : A ∪ B = { x : ℝ | -1 ≤ x ∧ x ≤ 5 } := 
by sorry

end union_of_sets_l478_47880


namespace total_time_preparing_games_l478_47827

def time_A_game : ℕ := 15
def time_B_game : ℕ := 25
def time_C_game : ℕ := 30
def num_each_type : ℕ := 5

theorem total_time_preparing_games : 
  (num_each_type * time_A_game + num_each_type * time_B_game + num_each_type * time_C_game) = 350 := 
  by sorry

end total_time_preparing_games_l478_47827


namespace max_rect_area_l478_47801

theorem max_rect_area (l w : ℤ) (h1 : 2 * l + 2 * w = 40) (h2 : 0 < l) (h3 : 0 < w) : 
  l * w ≤ 100 :=
by sorry

end max_rect_area_l478_47801


namespace abc_sum_equals_9_l478_47809

theorem abc_sum_equals_9 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a * b + c = 57) (h5 : b * c + a = 57) (h6 : a * c + b = 57) :
  a + b + c = 9 := 
sorry

end abc_sum_equals_9_l478_47809


namespace roots_of_quadratic_eq_l478_47822

theorem roots_of_quadratic_eq:
  (8 * γ^3 + 15 * δ^2 = 179) ↔ (γ^2 - 3 * γ + 1 = 0 ∧ δ^2 - 3 * δ + 1 = 0) :=
sorry

end roots_of_quadratic_eq_l478_47822


namespace s_mores_graham_crackers_l478_47845

def graham_crackers_per_smore (total_graham_crackers total_marshmallows : ℕ) : ℕ :=
total_graham_crackers / total_marshmallows

theorem s_mores_graham_crackers :
  let total_graham_crackers := 48
  let available_marshmallows := 6
  let additional_marshmallows := 18
  let total_marshmallows := available_marshmallows + additional_marshmallows
  graham_crackers_per_smore total_graham_crackers total_marshallows = 2 := sorry

end s_mores_graham_crackers_l478_47845


namespace find_y_l478_47847

theorem find_y (x y : ℝ) (h1 : x = 202) (h2 : x^3 * y - 4 * x^2 * y + 2 * x * y = 808080) : y = 1 / 10 := by
  sorry

end find_y_l478_47847


namespace actual_diameter_of_tissue_l478_47816

theorem actual_diameter_of_tissue (magnification: ℝ) (magnified_diameter: ℝ) :
  magnification = 1000 ∧ magnified_diameter = 1 → magnified_diameter / magnification = 0.001 :=
by
  intro h
  sorry

end actual_diameter_of_tissue_l478_47816


namespace no_digit_satisfies_equations_l478_47889

-- Define the conditions as predicates.
def is_digit (x : ℤ) : Prop := 0 ≤ x ∧ x < 10

-- Formulate the proof problem based on the given problem conditions and conclusion
theorem no_digit_satisfies_equations : 
  ¬ (∃ x : ℤ, is_digit x ∧ (x - (10 * x + x) = 801 ∨ x - (10 * x + x) = 812)) :=
by
  sorry

end no_digit_satisfies_equations_l478_47889


namespace complement_intersection_eq_l478_47894

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_eq :
  (U \ (M ∩ N)) = {1, 3, 4} := by
  sorry

end complement_intersection_eq_l478_47894


namespace coefficient_B_is_1_l478_47892

-- Definitions based on the conditions
def g (A B C D : ℝ) (x : ℝ) : ℝ := A * x^3 + B * x^2 + C * x + D

-- Given conditions
def condition1 (A B C D : ℝ) := g A B C D (-2) = 0 
def condition2 (A B C D : ℝ) := g A B C D 0 = -1
def condition3 (A B C D : ℝ) := g A B C D 2 = 0

-- The main theorem to prove
theorem coefficient_B_is_1 (A B C D : ℝ) 
  (h1 : condition1 A B C D) 
  (h2 : condition2 A B C D) 
  (h3 : condition3 A B C D) : 
  B = 1 :=
sorry

end coefficient_B_is_1_l478_47892


namespace negative_values_of_x_l478_47885

theorem negative_values_of_x : 
  let f (x : ℤ) := Int.sqrt (x + 196)
  ∃ (n : ℕ), (f (n ^ 2 - 196) > 0 ∧ f (n ^ 2 - 196) = n) ∧ ∃ k : ℕ, k = 13 :=
by
  sorry

end negative_values_of_x_l478_47885


namespace base_8_subtraction_l478_47817

def subtract_in_base_8 (a b : ℕ) : ℕ := 
  -- Implementing the base 8 subtraction
  sorry

theorem base_8_subtraction : subtract_in_base_8 0o652 0o274 = 0o356 :=
by 
  -- Faking the proof to ensure it can compile.
  sorry

end base_8_subtraction_l478_47817


namespace trajectory_of_point_l478_47808

theorem trajectory_of_point (x y k : ℝ) (hx : x ≠ 0) (hk : k ≠ 0) (h : |y| / |x| = k) : y = k * x ∨ y = -k * x :=
by
  sorry

end trajectory_of_point_l478_47808


namespace problem_statement_l478_47815

open Set

def M : Set ℝ := {x | x^2 - 2008 * x - 2009 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}

theorem problem_statement (a b : ℝ) :
  (M ∪ N a b = univ) →
  (M ∩ N a b = {x | 2009 < x ∧ x ≤ 2010}) →
  (a = 2009 ∧ b = 2010) :=
by
  sorry

end problem_statement_l478_47815


namespace square_side_length_l478_47831

theorem square_side_length :
  ∀ (w l : ℕ) (area : ℕ),
  w = 9 → l = 27 → area = w * l →
  ∃ s : ℝ, s^2 = area ∧ s = 9 * Real.sqrt 3 :=
by
  intros w l area hw hl harea
  sorry

end square_side_length_l478_47831


namespace wrong_value_l478_47821

-- Definitions based on the conditions
def initial_mean : ℝ := 32
def corrected_mean : ℝ := 32.5
def num_observations : ℕ := 50
def correct_observation : ℝ := 48

-- We need to prove that the wrong value of the observation was 23
theorem wrong_value (sum_initial : ℝ) (sum_corrected : ℝ) : 
  sum_initial = num_observations * initial_mean ∧ 
  sum_corrected = num_observations * corrected_mean →
  48 - (sum_corrected - sum_initial) = 23 :=
by
  sorry

end wrong_value_l478_47821


namespace black_area_after_six_transformations_l478_47820

noncomputable def remaining_fraction_after_transformations (initial_fraction : ℚ) (transforms : ℕ) (reduction_factor : ℚ) : ℚ :=
  reduction_factor ^ transforms * initial_fraction

theorem black_area_after_six_transformations :
  remaining_fraction_after_transformations 1 6 (2 / 3) = 64 / 729 := 
by
  sorry

end black_area_after_six_transformations_l478_47820


namespace mutually_exclusive_not_opposite_l478_47865

-- Define the given conditions
def boys := 6
def girls := 5
def total_students := boys + girls
def selection := 3

-- Define the mutually exclusive and not opposite events
def event_at_least_2_boys := ∃ (b: ℕ), ∃ (g: ℕ), (b + g = selection) ∧ (b ≥ 2) ∧ (g ≤ (selection - b))
def event_at_least_2_girls := ∃ (b: ℕ), ∃ (g: ℕ), (b + g = selection) ∧ (g ≥ 2) ∧ (b ≤ (selection - g))

-- Statement that these events are mutually exclusive but not opposite
theorem mutually_exclusive_not_opposite :
  (event_at_least_2_boys ∧ event_at_least_2_girls) → 
  (¬ ((∃ (b: ℕ) (g: ℕ), b + g = selection ∧ b ≥ 2 ∧ g ≥ 2) ∧ ¬(event_at_least_2_boys))) :=
sorry

end mutually_exclusive_not_opposite_l478_47865


namespace simplify_trig_expression_l478_47884

theorem simplify_trig_expression (A : ℝ) :
  (2 - (Real.cos A / Real.sin A) + (1 / Real.sin A)) * (3 - (Real.sin A / Real.cos A) - (1 / Real.cos A)) = 
  7 * Real.sin A * Real.cos A - 2 * Real.cos A ^ 2 - 3 * Real.sin A ^ 2 - 3 * Real.cos A + Real.sin A + 1 :=
by
  sorry

end simplify_trig_expression_l478_47884


namespace find_number_l478_47812

variable (x : ℕ)
variable (result : ℕ)

theorem find_number (h : x * 9999 = 4690640889) : x = 469131 :=
by
  sorry

end find_number_l478_47812


namespace bucket_proof_l478_47814

variable (CA : ℚ) -- capacity of Bucket A
variable (CB : ℚ) -- capacity of Bucket B
variable (SA_init : ℚ) -- initial amount of sand in Bucket A
variable (SB_init : ℚ) -- initial amount of sand in Bucket B

def bucket_conditions : Prop := 
  CB = (1 / 2) * CA ∧
  SA_init = (1 / 4) * CA ∧
  SB_init = (3 / 8) * CB

theorem bucket_proof (h : bucket_conditions CA CB SA_init SB_init) : 
  (SA_init + SB_init) / CA = 7 / 16 := 
  by sorry

end bucket_proof_l478_47814


namespace students_in_high_school_l478_47811

-- Definitions from conditions
def H (L: ℝ) : ℝ := 4 * L
def middleSchoolStudents : ℝ := 300
def combinedStudents (H: ℝ) (L: ℝ) : ℝ := H + L
def combinedIsSevenTimesMiddle (H: ℝ) (L: ℝ) : Prop := combinedStudents H L = 7 * middleSchoolStudents

-- The main goal to prove
theorem students_in_high_school (L H: ℝ) (h1: H = 4 * L) (h2: combinedIsSevenTimesMiddle H L) : H = 1680 := by
  sorry

end students_in_high_school_l478_47811


namespace exponent_of_4_l478_47851

theorem exponent_of_4 (x : ℕ) (h₁ : (1 / 4 : ℚ) ^ 2 = 1 / 16) (h₂ : 16384 * (1 / 16 : ℚ) = 1024) :
  4 ^ x = 1024 → x = 5 :=
by
  sorry

end exponent_of_4_l478_47851


namespace expression_divisible_by_1968_l478_47873

theorem expression_divisible_by_1968 (n : ℕ) : 
  ( -1 ^ (2 * n) +  9 ^ (4 * n) - 6 ^ (8 * n) + 8 ^ (16 * n) ) % 1968 = 0 :=
by
  sorry

end expression_divisible_by_1968_l478_47873


namespace percentage_games_won_l478_47867

def total_games_played : ℕ := 75
def win_rate_first_100_games : ℝ := 0.65

theorem percentage_games_won : 
  (win_rate_first_100_games * total_games_played / total_games_played * 100) = 65 := 
by
  sorry

end percentage_games_won_l478_47867


namespace units_digit_of_fraction_l478_47879

theorem units_digit_of_fraction :
  let numer := 30 * 31 * 32 * 33 * 34 * 35
  let denom := 1000
  (numer / denom) % 10 = 6 :=
by
  sorry

end units_digit_of_fraction_l478_47879


namespace gcd_1680_1683_l478_47826

theorem gcd_1680_1683 :
  ∀ (n : ℕ), n = 1683 →
  (∀ m, (m = 5 ∨ m = 67 ∨ m = 8) → n % m = 3) →
  (∃ d, d > 1 ∧ d ∣ 1683 ∧ d = Nat.gcd 1680 n ∧ Nat.gcd 1680 n = 3) :=
by
  sorry

end gcd_1680_1683_l478_47826


namespace no_C_makes_2C7_even_and_multiple_of_5_l478_47848

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

theorem no_C_makes_2C7_even_and_multiple_of_5 : ∀ C : ℕ, ¬(C < 10) ∨ ¬(is_even (2 * 100 + C * 10 + 7) ∧ is_multiple_of_5 (2 * 100 + C * 10 + 7)) :=
by
  intro C
  sorry

end no_C_makes_2C7_even_and_multiple_of_5_l478_47848


namespace expression_evaluation_l478_47837

theorem expression_evaluation (k : ℚ) (h : 3 * k = 10) : (6 / 5) * k - 2 = 2 :=
by
  sorry

end expression_evaluation_l478_47837


namespace max_value_of_function_l478_47871

theorem max_value_of_function : ∀ x : ℝ, (0 < x ∧ x < 1) → x * (1 - x) ≤ 1 / 4 :=
sorry

end max_value_of_function_l478_47871


namespace incident_ray_slope_in_circle_problem_l478_47806

noncomputable def slope_of_incident_ray : ℚ := sorry

theorem incident_ray_slope_in_circle_problem :
  ∃ (P : ℝ × ℝ) (C : ℝ × ℝ) (D : ℝ × ℝ),
  P = (-1, -3) ∧
  C = (2, -1) ∧
  (D = (C.1, -C.2)) ∧
  (D = (2, 1)) ∧
  ∀ (m : ℚ), (m = (D.2 - P.2) / (D.1 - P.1)) → m = 4 / 3 := 
sorry

end incident_ray_slope_in_circle_problem_l478_47806


namespace speed_of_current_l478_47844

-- Definitions
def downstream_speed (m current : ℝ) := m + current
def upstream_speed (m current : ℝ) := m - current

-- Theorem
theorem speed_of_current 
  (m : ℝ) (current : ℝ) 
  (h1 : downstream_speed m current = 20) 
  (h2 : upstream_speed m current = 14) : 
  current = 3 :=
by
  -- proof goes here
  sorry

end speed_of_current_l478_47844


namespace chess_tournament_proof_l478_47876

-- Define the conditions
variables (i g n I G : ℕ)
variables (VI VG VD : ℕ)

-- Condition 1: The number of GMs is ten times the number of IMs
def condition1 : Prop := g = 10 * i
  
-- Condition 2: The sum of the points of all GMs is 4.5 times the sum of the points of all IMs
def condition2 : Prop := G = 5 * I + I / 2

-- Condition 3: The total number of players is the sum of IMs and GMs
def condition3 : Prop := n = i + g

-- Condition 4: Each player played only once against all other opponents
def condition4 : Prop := n * (n - 1) = 2 * (VI + VG + VD)

-- Condition 5: The sum of the points of all games is 5.5 times the sum of the points of all IMs
def condition5 : Prop := I + G = 11 * I / 2

-- Condition 6: Total games played
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The questions to be proven given the conditions
theorem chess_tournament_proof:
  condition1 i g →
  condition2 I G →
  condition3 i g n →
  condition4 n VI VG VD →
  condition5 I G →
  i = 1 ∧ g = 10 ∧ total_games n = 55 :=
by
  -- The proof is left as an exercise
  sorry

end chess_tournament_proof_l478_47876


namespace time_equal_l478_47869

noncomputable def S : ℝ := sorry 
noncomputable def S_flat : ℝ := S
noncomputable def S_uphill : ℝ := (1 / 3) * S
noncomputable def S_downhill : ℝ := (2 / 3) * S
noncomputable def V_flat : ℝ := sorry 
noncomputable def V_uphill : ℝ := (1 / 2) * V_flat
noncomputable def V_downhill : ℝ := 2 * V_flat
noncomputable def t_flat: ℝ := S / V_flat
noncomputable def t_uphill: ℝ := S_uphill / V_uphill
noncomputable def t_downhill: ℝ := S_downhill / V_downhill
noncomputable def t_hill: ℝ := t_uphill + t_downhill

theorem time_equal: t_flat = t_hill := 
  by sorry

end time_equal_l478_47869


namespace problem_solution_l478_47818

theorem problem_solution (k a b : ℝ) (h1 : k = a + Real.sqrt b) 
  (h2 : abs (Real.logb 5 k - Real.logb 5 (k^2 + 3)) = 0.6) : 
  a + b = 15 :=
sorry

end problem_solution_l478_47818


namespace solve_abs_equation_l478_47895

theorem solve_abs_equation (x : ℝ) :
  (|2 * x + 1| - |x - 5| = 6) ↔ (x = -12 ∨ x = 10 / 3) :=
by sorry

end solve_abs_equation_l478_47895


namespace ship_passengers_round_trip_tickets_l478_47860

theorem ship_passengers_round_trip_tickets (total_passengers : ℕ) (p1 : ℝ) (p2 : ℝ) :
  (p1 = 0.25 * total_passengers) ∧ (p2 = 0.6 * (p * total_passengers)) →
  (p * total_passengers = 62.5 / 100 * total_passengers) :=
by
  sorry

end ship_passengers_round_trip_tickets_l478_47860


namespace distance_between_points_l478_47862

/-- Given points P1 and P2 in the plane, prove that the distance between 
P1 and P2 is 5 units. -/
theorem distance_between_points : 
  let P1 : ℝ × ℝ := (-1, 1)
  let P2 : ℝ × ℝ := (2, 5)
  dist P1 P2 = 5 :=
by 
  sorry

end distance_between_points_l478_47862


namespace simplify_expression_l478_47854

theorem simplify_expression (a b c d x y : ℝ) (h : cx ≠ -dy) :
  (cx * (b^2 * x^2 + 3 * b^2 * y^2 + a^2 * y^2) + dy * (b^2 * x^2 + 3 * a^2 * x^2 + a^2 * y^2)) / (cx + dy)
  = (b^2 + 3 * a^2) * x^2 + (a^2 + 3 * b^2) * y^2 := by
  sorry

end simplify_expression_l478_47854


namespace total_turtles_l478_47852

theorem total_turtles (G H L : ℕ) (h_G : G = 800) (h_H : H = 2 * G) (h_L : L = 3 * G) : G + H + L = 4800 :=
by
  sorry

end total_turtles_l478_47852


namespace find_coordinates_of_M_l478_47887

-- Definitions of the points A, B, C
def A : (ℝ × ℝ) := (2, -4)
def B : (ℝ × ℝ) := (-1, 3)
def C : (ℝ × ℝ) := (3, 4)

-- Definitions of vectors CA and CB
def vector_CA : (ℝ × ℝ) := (A.1 - C.1, A.2 - C.2)
def vector_CB : (ℝ × ℝ) := (B.1 - C.1, B.2 - C.2)

-- Definition of the point M
def M : (ℝ × ℝ) := (-11, -15)

-- Definition of vector CM
def vector_CM : (ℝ × ℝ) := (M.1 - C.1, M.2 - C.2)

-- The condition to prove
theorem find_coordinates_of_M : vector_CM = (2 * vector_CA.1 + 3 * vector_CB.1, 2 * vector_CA.2 + 3 * vector_CB.2) :=
by
  sorry

end find_coordinates_of_M_l478_47887


namespace altered_solution_water_amount_l478_47823

def initial_bleach_ratio := 2
def initial_detergent_ratio := 40
def initial_water_ratio := 100

def new_bleach_to_detergent_ratio := 3 * initial_bleach_ratio
def new_detergent_to_water_ratio := initial_detergent_ratio / 2

def detergent_amount := 60
def water_amount := 75

theorem altered_solution_water_amount :
  (initial_detergent_ratio / new_detergent_to_water_ratio) * detergent_amount / new_bleach_to_detergent_ratio = water_amount :=
by
  sorry

end altered_solution_water_amount_l478_47823


namespace students_in_class_l478_47833

theorem students_in_class (n : ℕ) 
  (h1 : 15 = 15)
  (h2 : ∃ m, n = m + 20 - 1)
  (h3 : ∃ x : ℕ, x = 3) :
  n = 38 :=
by
  sorry

end students_in_class_l478_47833


namespace distribute_tourists_l478_47819

theorem distribute_tourists (guides tourists : ℕ) (hguides : guides = 3) (htourists : tourists = 8) :
  ∃ k, k = 5796 := by
  sorry

end distribute_tourists_l478_47819


namespace cameron_books_ratio_l478_47883

theorem cameron_books_ratio (Boris_books : ℕ) (Cameron_books : ℕ)
  (Boris_after_donation : ℕ) (Cameron_after_donation : ℕ)
  (total_books_after_donation : ℕ) (ratio : ℚ) :
  Boris_books = 24 → 
  Cameron_books = 30 → 
  Boris_after_donation = Boris_books - (Boris_books / 4) →
  total_books_after_donation = 38 →
  Cameron_after_donation = total_books_after_donation - Boris_after_donation →
  ratio = (Cameron_books - Cameron_after_donation) / Cameron_books →
  ratio = 1 / 3 :=
by
  -- Proof goes here.
  sorry

end cameron_books_ratio_l478_47883


namespace total_yarn_length_is_1252_l478_47839

/-- Defining the lengths of the yarns according to the conditions --/
def green_yarn : ℕ := 156
def red_yarn : ℕ := 3 * green_yarn + 8
def blue_yarn : ℕ := (green_yarn + red_yarn) / 2
def average_yarn_length : ℕ := (green_yarn + red_yarn + blue_yarn) / 3
def yellow_yarn : ℕ := average_yarn_length - 12

/-- Proving the total length of the four pieces of yarn is 1252 cm --/
theorem total_yarn_length_is_1252 :
  green_yarn + red_yarn + blue_yarn + yellow_yarn = 1252 := by
  sorry

end total_yarn_length_is_1252_l478_47839


namespace sugar_cubes_left_l478_47893

theorem sugar_cubes_left (h w d : ℕ) (hd1 : w * d = 77) (hd2 : h * d = 55) :
  (h - 1) * w * (d - 1) = 300 ∨ (h - 1) * w * (d - 1) = 0 :=
by
  sorry

end sugar_cubes_left_l478_47893


namespace find_x_l478_47842

-- Definitions based on the conditions
def remaining_scores_after_removal (s: List ℕ) : List ℕ :=
  s.erase 87 |>.erase 94

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

-- Converting the given problem into a Lean 4 theorem statement
theorem find_x (x : ℕ) (s : List ℕ) :
  s = [94, 87, 89, 88, 92, 90, x, 93, 92, 91] →
  average (remaining_scores_after_removal s) = 91 →
  x = 2 :=
by
  intros h1 h2
  sorry

end find_x_l478_47842


namespace sales_tax_difference_l478_47834

theorem sales_tax_difference :
  let price : ℝ := 30
  let tax_rate1 : ℝ := 0.0675
  let tax_rate2 : ℝ := 0.055
  let sales_tax1 : ℝ := price * tax_rate1
  let sales_tax2 : ℝ := price * tax_rate2
  let difference : ℝ := sales_tax1 - sales_tax2
  difference = 0.375 :=
by
  let price : ℝ := 30
  let tax_rate1 : ℝ := 0.0675
  let tax_rate2 : ℝ := 0.055
  let sales_tax1 : ℝ := price * tax_rate1
  let sales_tax2 : ℝ := price * tax_rate2
  let difference : ℝ := sales_tax1 - sales_tax2
  exact sorry

end sales_tax_difference_l478_47834


namespace find_x_range_l478_47896

noncomputable def p (x : ℝ) := x^2 + 2*x - 3 > 0
noncomputable def q (x : ℝ) := 1/(3 - x) > 1

theorem find_x_range (x : ℝ) : (¬q x ∧ p x) → (x ≥ 3 ∨ (1 < x ∧ x ≤ 2) ∨ x < -3) :=
by
  intro h
  sorry

end find_x_range_l478_47896


namespace remainder_div_l478_47898

theorem remainder_div (N : ℤ) (k : ℤ) (h : N = 39 * k + 18) :
  N % 13 = 5 := 
by
  sorry

end remainder_div_l478_47898


namespace min_value_of_y_min_value_achieved_l478_47807

noncomputable def y (x : ℝ) : ℝ := x + 1/x + 16*x / (x^2 + 1)

theorem min_value_of_y : ∀ x > 1, y x ≥ 8 :=
  sorry

theorem min_value_achieved : ∃ x, (x > 1) ∧ (y x = 8) :=
  sorry

end min_value_of_y_min_value_achieved_l478_47807


namespace final_price_percentage_l478_47861

theorem final_price_percentage (original_price sale_price final_price : ℝ) (h1 : sale_price = 0.9 * original_price) 
(h2 : final_price = sale_price - 0.1 * sale_price) : final_price / original_price = 0.81 :=
by
  sorry

end final_price_percentage_l478_47861


namespace infinitely_many_solutions_l478_47882

def circ (x y : ℝ) : ℝ := 4 * x - 3 * y + x * y

theorem infinitely_many_solutions : ∀ y : ℝ, circ 3 y = 12 := by
  sorry

end infinitely_many_solutions_l478_47882


namespace exponent_rule_example_l478_47853

theorem exponent_rule_example {a : ℝ} : (a^3)^4 = a^12 :=
by {
  sorry
}

end exponent_rule_example_l478_47853


namespace andrew_age_l478_47855

/-- 
Andrew and his five cousins are ages 4, 6, 8, 10, 12, and 14. 
One afternoon two of his cousins whose ages sum to 18 went to the movies. 
Two cousins younger than 12 but not including the 8-year-old went to play baseball. 
Andrew and the 6-year-old stayed home. How old is Andrew?
-/
theorem andrew_age (ages : Finset ℕ) (andrew_age: ℕ)
  (h_ages : ages = {4, 6, 8, 10, 12, 14})
  (movies : Finset ℕ) (baseball : Finset ℕ)
  (h_movies1 : movies.sum id = 18)
  (h_baseball1 : ∀ x ∈ baseball, x < 12 ∧ x ≠ 8)
  (home : Finset ℕ) (h_home : home = {6, andrew_age}) :
  andrew_age = 12 :=
sorry

end andrew_age_l478_47855


namespace general_formula_l478_47813

def sum_of_terms (a : ℕ → ℕ) (n : ℕ) : ℕ := 3 / 2 * a n - 3

def sequence_term (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 0 then 6 
  else a (n - 1) * 3

theorem general_formula (a : ℕ → ℕ) (n : ℕ) :
  (∀ n, sum_of_terms a n = 3 / 2 * a n - 3) →
  (∀ n, n = 0 → a n = 6) →
  (∀ n, n > 0 → a n = a (n - 1) * 3) →
  a n = 2 * 3^n := by
  sorry

end general_formula_l478_47813


namespace geometric_mean_unique_solution_l478_47843

-- Define the conditions
variable (k : ℕ) -- k is a natural number
variable (hk_pos : 0 < k) -- k is a positive natural number

-- The geometric mean condition translated to Lean
def geometric_mean_condition (k : ℕ) : Prop :=
  (2 * k)^2 = (k + 9) * (6 - k)

-- The main statement to prove
theorem geometric_mean_unique_solution (k : ℕ) (hk_pos : 0 < k) (h: geometric_mean_condition k) : k = 3 :=
sorry -- proof placeholder

end geometric_mean_unique_solution_l478_47843


namespace negB_sufficient_for_A_l478_47881

variables {A B : Prop}

theorem negB_sufficient_for_A (h : ¬A → B) (hnotsuff : ¬(B → ¬A)) : ¬ B → A :=
by
  sorry

end negB_sufficient_for_A_l478_47881


namespace water_tank_capacity_l478_47830

theorem water_tank_capacity (rate : ℝ) (time : ℝ) (fraction : ℝ) (capacity : ℝ) : 
(rate = 10) → (time = 300) → (fraction = 3/4) → 
(rate * time = fraction * capacity) → 
capacity = 4000 := 
by
  intros h_rate h_time h_fraction h_equation
  rw [h_rate, h_time, h_fraction] at h_equation
  linarith

end water_tank_capacity_l478_47830


namespace total_yards_thrown_l478_47897

-- Definitions for the conditions
def distance_50_degrees : ℕ := 20
def distance_80_degrees : ℕ := distance_50_degrees * 2

def throws_on_saturday : ℕ := 20
def throws_on_sunday : ℕ := 30

def headwind_penalty : ℕ := 5
def tailwind_bonus : ℕ := 10

-- Theorem for the total yards thrown in two days
theorem total_yards_thrown :
  ((distance_50_degrees - headwind_penalty) * throws_on_saturday) + 
  ((distance_80_degrees + tailwind_bonus) * throws_on_sunday) = 1800 :=
by
  sorry

end total_yards_thrown_l478_47897


namespace problem1_problem2_problem3_l478_47800

-- Proof statement for Problem 1
theorem problem1 : 23 * (-5) - (-3) / (3 / 108) = -7 := 
by 
  sorry

-- Proof statement for Problem 2
theorem problem2 : (-7) * (-3) * (-0.5) + (-12) * (-2.6) = 20.7 := 
by 
  sorry

-- Proof statement for Problem 3
theorem problem3 : ((-1 / 2) - (1 / 12) + (3 / 4) - (1 / 6)) * (-48) = 0 := 
by 
  sorry

end problem1_problem2_problem3_l478_47800


namespace find_other_number_l478_47899

theorem find_other_number
  (a b : ℕ)
  (HCF : ℕ)
  (LCM : ℕ)
  (h1 : HCF = 12)
  (h2 : LCM = 396)
  (h3 : a = 36)
  (h4 : HCF * LCM = a * b) :
  b = 132 :=
by
  sorry

end find_other_number_l478_47899


namespace distance_along_stream_l478_47858
-- Define the problem in Lean 4

noncomputable def speed_boat_still : ℝ := 11   -- Speed of the boat in still water
noncomputable def distance_against_stream : ℝ := 9  -- Distance traveled against the stream in one hour

theorem distance_along_stream : 
  ∃ (v_s : ℝ), (speed_boat_still - v_s = distance_against_stream) ∧ (11 + v_s) * 1 = 13 := 
by
  use 2
  sorry

end distance_along_stream_l478_47858


namespace find_AX_l478_47829

theorem find_AX (AB AC BC : ℝ) (CX_bisects_ACB : Prop) (h1 : AB = 50) (h2 : AC = 28) (h3 : BC = 56) : AX = 50 / 3 :=
by
  -- Proof can be added here
  sorry

end find_AX_l478_47829
