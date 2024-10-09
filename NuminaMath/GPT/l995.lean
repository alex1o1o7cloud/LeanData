import Mathlib

namespace triangle_is_isosceles_right_l995_99592

theorem triangle_is_isosceles_right
  (a b c : ℝ)
  (A B C : ℕ)
  (h1 : c = a * Real.cos B)
  (h2 : b = a * Real.sin C) :
  C = 90 ∧ B = 90 ∧ A = 90 :=
sorry

end triangle_is_isosceles_right_l995_99592


namespace find_number_l995_99519

theorem find_number (N : ℝ)
  (h1 : 5 / 6 * N = 5 / 16 * N + 250) :
  N = 480 :=
sorry

end find_number_l995_99519


namespace number_of_students_in_class_l995_99517

theorem number_of_students_in_class
  (G : ℕ) (E_and_G : ℕ) (E_only: ℕ)
  (h1 : G = 22)
  (h2 : E_and_G = 12)
  (h3 : E_only = 23) :
  ∃ S : ℕ, S = 45 :=
by
  sorry

end number_of_students_in_class_l995_99517


namespace max_value_of_a_l995_99581

theorem max_value_of_a {a : ℝ} (h : ∀ x ≥ 1, -3 * x^2 + a ≤ 0) : a ≤ 3 :=
sorry

end max_value_of_a_l995_99581


namespace product_of_points_l995_99575

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 2 ≠ 0 then 8
  else if n % 2 = 0 ∧ n % 3 ≠ 0 then 3
  else 0

def Chris_rolls : List ℕ := [5, 2, 1, 6]
def Dana_rolls : List ℕ := [6, 2, 3, 3]

def Chris_points : ℕ := (Chris_rolls.map f).sum
def Dana_points : ℕ := (Dana_rolls.map f).sum

theorem product_of_points : Chris_points * Dana_points = 297 := by
  sorry

end product_of_points_l995_99575


namespace num_occupied_third_floor_rooms_l995_99560

-- Definitions based on conditions
def first_floor_rent : Int := 15
def second_floor_rent : Int := 20
def third_floor_rent : Int := 2 * first_floor_rent
def rooms_per_floor : Int := 3
def monthly_earnings : Int := 165

-- The proof statement
theorem num_occupied_third_floor_rooms : 
  let total_full_occupancy_cost := rooms_per_floor * first_floor_rent + rooms_per_floor * second_floor_rent + rooms_per_floor * third_floor_rent
  let revenue_difference := total_full_occupancy_cost - monthly_earnings
  revenue_difference / third_floor_rent = 1 → rooms_per_floor - revenue_difference / third_floor_rent = 2 :=
by
  sorry

end num_occupied_third_floor_rooms_l995_99560


namespace line_always_intersects_circle_shortest_chord_line_equation_l995_99522

open Real

noncomputable def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y + 9 = 0

noncomputable def line_eqn (m x y : ℝ) : Prop := 2 * m * x - 3 * m * y + x - y - 1 = 0

theorem line_always_intersects_circle (m : ℝ) : 
  ∀ (x y : ℝ), circle_eqn x y → line_eqn m x y → True := 
by
  sorry

theorem shortest_chord_line_equation : 
  ∃ (m x y : ℝ), line_eqn m x y ∧ (∀ x y, line_eqn m x y → x - y - 1 = 0) :=
by
  sorry

end line_always_intersects_circle_shortest_chord_line_equation_l995_99522


namespace axis_of_symmetry_l995_99548

theorem axis_of_symmetry (x : ℝ) : 
  ∀ y, y = x^2 - 2 * x - 3 → (∃ k : ℝ, k = 1 ∧ ∀ x₀ : ℝ, y = (x₀ - k)^2 + C) := 
sorry

end axis_of_symmetry_l995_99548


namespace smallest_number_of_students_l995_99500

theorem smallest_number_of_students (n : ℕ) (x : ℕ) 
  (h_total : n = 5 * x + 3) 
  (h_more_than_50 : n > 50) : 
  n = 53 :=
by {
  sorry
}

end smallest_number_of_students_l995_99500


namespace negation_of_proposition_l995_99501

variable (l : ℝ)

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x + l ≥ 0) ↔ (∀ x : ℝ, x + l < 0) := by
  sorry

end negation_of_proposition_l995_99501


namespace landscape_length_l995_99588

theorem landscape_length (b l : ℕ) (playground_area : ℕ) (total_area : ℕ) 
  (h1 : l = 4 * b) (h2 : playground_area = 1200) (h3 : total_area = 3 * playground_area) (h4 : total_area = l * b) :
  l = 120 := 
by 
  sorry

end landscape_length_l995_99588


namespace min_value_F_l995_99561

theorem min_value_F :
  ∀ (x y : ℝ), (x^2 + y^2 - 2*x - 2*y + 1 = 0) → (x + 1) / y ≥ 3 / 4 :=
by
  intro x y h
  sorry

end min_value_F_l995_99561


namespace ratio_of_volumes_l995_99512

theorem ratio_of_volumes (r : ℝ) (π : ℝ) (V1 V2 : ℝ) 
  (h1 : V2 = (4 / 3) * π * r^3) 
  (h2 : V1 = 2 * π * r^3) : 
  V1 / V2 = 3 / 2 :=
by
  sorry

end ratio_of_volumes_l995_99512


namespace rowing_time_one_hour_l995_99557

noncomputable def total_time_to_travel (Vm Vr distance : ℝ) : ℝ :=
  let upstream_speed := Vm - Vr
  let downstream_speed := Vm + Vr
  let one_way_distance := distance / 2
  let time_upstream := one_way_distance / upstream_speed
  let time_downstream := one_way_distance / downstream_speed
  time_upstream + time_downstream

theorem rowing_time_one_hour : 
  total_time_to_travel 8 1.8 7.595 = 1 := 
sorry

end rowing_time_one_hour_l995_99557


namespace perfect_square_trinomial_k_l995_99535

theorem perfect_square_trinomial_k (k : ℤ) :
  (∀ x : ℝ, 9 * x^2 + 6 * x + k = (3 * x + 1) ^ 2) → (k = 1) :=
by
  sorry

end perfect_square_trinomial_k_l995_99535


namespace decagon_area_l995_99536

theorem decagon_area (perimeter : ℝ) (n : ℕ) (side_length : ℝ)
  (segments : ℕ) (area : ℝ) :
  perimeter = 200 ∧ n = 4 ∧ side_length = perimeter / n ∧ segments = 5 ∧ 
  area = (side_length / segments)^2 * (1 - (1/2)) * 4 * segments  →
  area = 2300 := 
by
  sorry

end decagon_area_l995_99536


namespace algebraic_expression_value_l995_99510

theorem algebraic_expression_value (x y : ℕ) (h : 3 * x - y = 1) : (8^x : ℝ) / (2^y) / 2 = 1 := 
by 
  sorry

end algebraic_expression_value_l995_99510


namespace probability_G_is_one_fourth_l995_99587

-- Definitions and conditions
variables (p_E p_F p_G p_H : ℚ)
axiom probability_E : p_E = 1/3
axiom probability_F : p_F = 1/6
axiom prob_G_eq_H : p_G = p_H
axiom total_prob_sum : p_E + p_F + p_G + p_G = 1

-- Theorem statement
theorem probability_G_is_one_fourth : p_G = 1/4 :=
by 
  -- Lean proof omitted, only the statement required
  sorry

end probability_G_is_one_fourth_l995_99587


namespace find_wrongly_written_height_l995_99537

variable (n : ℕ := 35)
variable (average_height_incorrect : ℚ := 184)
variable (actual_height_one_boy : ℚ := 106)
variable (actual_average_height : ℚ := 182)
variable (x : ℚ)

theorem find_wrongly_written_height
  (h_incorrect_total : n * average_height_incorrect = 6440)
  (h_correct_total : n * actual_average_height = 6370) :
  6440 - x + actual_height_one_boy = 6370 ↔ x = 176 := by
  sorry

end find_wrongly_written_height_l995_99537


namespace sum_of_legs_of_larger_triangle_l995_99513

def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def similar_triangles {a1 b1 c1 a2 b2 c2 : ℝ} (h1 : right_triangle a1 b1 c1) (h2 : right_triangle a2 b2 c2) :=
  ∃ k : ℝ, k > 0 ∧ (a2 = k * a1 ∧ b2 = k * b1)

theorem sum_of_legs_of_larger_triangle 
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h1 : right_triangle a1 b1 c1)
  (h2 : right_triangle a2 b2 c2)
  (h_sim : similar_triangles h1 h2)
  (area1 : ℝ) (area2 : ℝ)
  (hyp1 : c1 = 6) 
  (area_cond1 : (a1 * b1) / 2 = 8)
  (area_cond2 : (a2 * b2) / 2 = 200) :
  a2 + b2 = 40 := by
  sorry

end sum_of_legs_of_larger_triangle_l995_99513


namespace fourth_number_pascal_row_l995_99578

theorem fourth_number_pascal_row : (Nat.choose 12 3) = 220 := sorry

end fourth_number_pascal_row_l995_99578


namespace can_all_mushrooms_become_good_l995_99585

def is_bad (w : Nat) : Prop := w ≥ 10
def is_good (w : Nat) : Prop := w < 10

def mushrooms_initially_bad := 90
def mushrooms_initially_good := 10

def total_mushrooms := mushrooms_initially_bad + mushrooms_initially_good
def total_worms_initial := mushrooms_initially_bad * 10

theorem can_all_mushrooms_become_good :
  ∃ worms_distribution : Fin total_mushrooms → Nat,
  (∀ i : Fin total_mushrooms, is_good (worms_distribution i)) :=
sorry

end can_all_mushrooms_become_good_l995_99585


namespace binomial_seven_four_l995_99533

noncomputable def binomial (n k : Nat) : Nat := n.choose k

theorem binomial_seven_four : binomial 7 4 = 35 := by
  sorry

end binomial_seven_four_l995_99533


namespace arrangements_with_gap_l995_99526

theorem arrangements_with_gap :
  ∃ (arrangements : ℕ), arrangements = 36 :=
by
  sorry

end arrangements_with_gap_l995_99526


namespace quadratic_roots_two_l995_99540

theorem quadratic_roots_two (m : ℝ) :
  let a := 1
  let b := -m
  let c := m - 2
  let Δ := b^2 - 4 * a * c
  Δ > 0 :=
by
  let a := 1
  let b := -m
  let c := m - 2
  let Δ := b^2 - 4 * a * c
  sorry

end quadratic_roots_two_l995_99540


namespace complement_U_A_l995_99570

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | |x - 1| > 1 }

theorem complement_U_A : (U \ A) = { x | 0 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end complement_U_A_l995_99570


namespace determine_xyz_l995_99577

theorem determine_xyz (x y z : ℂ) (h1 : x * y + 3 * y = -9) (h2 : y * z + 3 * z = -9) (h3 : z * x + 3 * x = -9) : 
  x * y * z = 27 := 
by
  sorry

end determine_xyz_l995_99577


namespace max_rooks_max_rooks_4x4_max_rooks_8x8_l995_99551

theorem max_rooks (n : ℕ) : ℕ :=
  2 * (2 * n / 3)

theorem max_rooks_4x4 :
  max_rooks 4 = 4 :=
  sorry

theorem max_rooks_8x8 :
  max_rooks 8 = 10 :=
  sorry

end max_rooks_max_rooks_4x4_max_rooks_8x8_l995_99551


namespace average_headcount_is_11033_l995_99538

def average_headcount (count1 count2 count3 : ℕ) : ℕ :=
  (count1 + count2 + count3) / 3

theorem average_headcount_is_11033 :
  average_headcount 10900 11500 10700 = 11033 :=
by
  sorry

end average_headcount_is_11033_l995_99538


namespace minimum_voters_for_tall_win_l995_99550

-- Definitions based on the conditions
def voters : ℕ := 135
def districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3
def majority_precinct_voters : ℕ := 2
def majority_precincts_per_district : ℕ := 5
def majority_districts : ℕ := 3
def tall_won : Prop := true

-- Problem statement
theorem minimum_voters_for_tall_win : 
  tall_won → (∃ n : ℕ, n = 3 * 5 * 2 ∧ n ≤ voters) :=
by
  sorry

end minimum_voters_for_tall_win_l995_99550


namespace amusement_park_line_l995_99559

theorem amusement_park_line (h1 : Eunji_position = 6) (h2 : people_behind_Eunji = 7) : total_people_in_line = 13 :=
by
  sorry

end amusement_park_line_l995_99559


namespace triangle_median_inequality_l995_99573

variable (a b c m_a m_b m_c D : ℝ)

-- Assuming the conditions are required to make the proof valid
axiom median_formula_m_a : 4 * m_a^2 + a^2 = 2 * b^2 + 2 * c^2
axiom median_formula_m_b : 4 * m_b^2 + b^2 = 2 * c^2 + 2 * a^2
axiom median_formula_m_c : 4 * m_c^2 + c^2 = 2 * a^2 + 2 * b^2

theorem triangle_median_inequality : 
  a^2 + b^2 <= m_c * 6 * D ∧ b^2 + c^2 <= m_a * 6 * D ∧ c^2 + a^2 <= m_b * 6 * D → 
  (a^2 + b^2) / m_c + (b^2 + c^2) / m_a + (c^2 + a^2) / m_b <= 6 * D := 
by
  sorry

end triangle_median_inequality_l995_99573


namespace find_even_increasing_l995_99572

theorem find_even_increasing (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x y : ℝ, 0 < x → x < y → 0 < y → f x < f y) ↔
  f = (fun x => 3 * x^2 - 1) ∨ f = (fun x => 2^|x|) :=
by
  sorry

end find_even_increasing_l995_99572


namespace plates_usage_when_parents_join_l995_99530

theorem plates_usage_when_parents_join
  (total_plates : ℕ)
  (plates_per_day_matt_and_son : ℕ)
  (days_matt_and_son : ℕ)
  (days_with_parents : ℕ)
  (total_days_in_week : ℕ)
  (total_plates_needed : total_plates = 38)
  (plates_used_matt_and_son : plates_per_day_matt_and_son = 2)
  (days_matt_and_son_eq : days_matt_and_son = 3)
  (days_with_parents_eq : days_with_parents = 4)
  (total_days_in_week_eq : total_days_in_week = 7)
  (plates_used_when_parents_join : total_plates - plates_per_day_matt_and_son * days_matt_and_son = days_with_parents * 8) :
  true :=
sorry

end plates_usage_when_parents_join_l995_99530


namespace find_x_l995_99528

def hash_op (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_x (x : ℕ) (h : hash_op x 6 = 48) : x = 3 :=
by
  sorry

end find_x_l995_99528


namespace solution_set_l995_99545

theorem solution_set (x : ℝ) : 
  (x * (x + 2) > 0 ∧ |x| < 1) ↔ (0 < x ∧ x < 1) := 
by 
  sorry

end solution_set_l995_99545


namespace goldfinch_percentage_l995_99511

noncomputable def percentage_of_goldfinches 
  (goldfinches : ℕ) (sparrows : ℕ) (grackles : ℕ) : ℚ :=
  (goldfinches : ℚ) / (goldfinches + sparrows + grackles) * 100

theorem goldfinch_percentage (goldfinches sparrows grackles : ℕ)
  (h_goldfinches : goldfinches = 6)
  (h_sparrows : sparrows = 9)
  (h_grackles : grackles = 5) :
  percentage_of_goldfinches goldfinches sparrows grackles = 30 :=
by
  rw [h_goldfinches, h_sparrows, h_grackles]
  show percentage_of_goldfinches 6 9 5 = 30
  sorry

end goldfinch_percentage_l995_99511


namespace prob_first_two_same_color_expected_value_eta_l995_99586

-- Definitions and conditions
def num_white : ℕ := 4
def num_black : ℕ := 3
def total_pieces : ℕ := num_white + num_black

-- Probability of drawing two pieces of the same color
def prob_same_color : ℚ :=
  (4/7 * 3/6) + (3/7 * 2/6)

-- Expected value of the number of white pieces drawn in the first four draws
def E_eta : ℚ :=
  1 * (4 / 35) + 2 * (18 / 35) + 3 * (12 / 35) + 4 * (1 / 35)

-- Proof statements
theorem prob_first_two_same_color : prob_same_color = 3 / 7 :=
  by sorry

theorem expected_value_eta : E_eta = 16 / 7 :=
  by sorry

end prob_first_two_same_color_expected_value_eta_l995_99586


namespace ms_perez_class_total_students_l995_99567

/-- Half the students in Ms. Perez's class collected 12 cans each, two students didn't collect any cans,
    and the remaining 13 students collected 4 cans each. The total number of cans collected is 232. 
    Prove that the total number of students in Ms. Perez's class is 30. -/
theorem ms_perez_class_total_students (S : ℕ) :
  (S / 2) * 12 + 13 * 4 + 2 * 0 = 232 →
  S = S / 2 + 13 + 2 →
  S = 30 :=
by {
  sorry
}

end ms_perez_class_total_students_l995_99567


namespace tan_theta_minus_pi_over4_l995_99524

theorem tan_theta_minus_pi_over4 (θ : Real) (h : Real.cos θ - 3 * Real.sin θ = 0) : 
  Real.tan (θ - Real.pi / 4) = -1 / 2 :=
sorry

end tan_theta_minus_pi_over4_l995_99524


namespace minimum_value_l995_99541

theorem minimum_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) : 
  (∃ x, (∀ y, y = (1 / a) + (4 / b) → y ≥ x) ∧ x = 9 / 2) :=
by
  sorry

end minimum_value_l995_99541


namespace price_of_pen_l995_99529

theorem price_of_pen (price_pen : ℚ) (price_notebook : ℚ) :
  (price_pen + 3 * price_notebook = 36.45) →
  (price_notebook = 15 / 4 * price_pen) →
  price_pen = 3 :=
by
  intros h1 h2
  sorry

end price_of_pen_l995_99529


namespace joe_spent_on_fruits_l995_99544

theorem joe_spent_on_fruits (total_money amount_left : ℝ) (spent_on_chocolates : ℝ)
  (h1 : total_money = 450)
  (h2 : spent_on_chocolates = (1/9) * total_money)
  (h3 : amount_left = 220)
  : (total_money - spent_on_chocolates - amount_left) / total_money = 2 / 5 :=
by
  sorry

end joe_spent_on_fruits_l995_99544


namespace sum_of_center_coordinates_eq_neg2_l995_99507

theorem sum_of_center_coordinates_eq_neg2 
  (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 7)
  (h2 : y1 = -8)
  (h3 : x2 = -5)
  (h4 : y2 = 2) 
  : (x1 + x2) / 2 + (y1 + y2) / 2 = -2 :=
by
  -- Insert proof here
  sorry

end sum_of_center_coordinates_eq_neg2_l995_99507


namespace find_a_from_limit_l995_99516

theorem find_a_from_limit (a : ℝ) (h : (Filter.Tendsto (fun n : ℕ => (a * n - 2) / (n + 1)) Filter.atTop (Filter.principal {1}))) :
    a = 1 := 
sorry

end find_a_from_limit_l995_99516


namespace find_width_of_first_tract_l995_99589

-- Definitions based on given conditions
noncomputable def area_first_tract (W : ℝ) : ℝ := 300 * W
def area_second_tract : ℝ := 250 * 630
def combined_area : ℝ := 307500

-- The theorem we need to prove: width of the first tract is 500 meters
theorem find_width_of_first_tract (W : ℝ) (h : area_first_tract W + area_second_tract = combined_area) : W = 500 :=
by
  sorry

end find_width_of_first_tract_l995_99589


namespace range_of_x_satisfying_inequality_l995_99563

theorem range_of_x_satisfying_inequality (x : ℝ) : 
  (|x+1| + |x| < 2) ↔ (-3/2 < x ∧ x < 1/2) :=
by sorry

end range_of_x_satisfying_inequality_l995_99563


namespace ending_number_of_sequence_divisible_by_11_l995_99598

theorem ending_number_of_sequence_divisible_by_11 : 
  ∃ (n : ℕ), 19 < n ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 6 → n = 19 + 11 * k) ∧ n = 77 :=
by
  sorry

end ending_number_of_sequence_divisible_by_11_l995_99598


namespace find_valid_triples_l995_99597

-- Define the theorem to prove the conditions and results
theorem find_valid_triples :
  ∀ (a b c : ℕ), 
    (2^a + 2^b + 1) % (2^c - 1) = 0 ↔ (a = 0 ∧ b = 0 ∧ c = 2) ∨ 
                                      (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
                                      (a = 2 ∧ b = 1 ∧ c = 3) := 
sorry  -- Proof omitted

end find_valid_triples_l995_99597


namespace isosceles_triangle_base_angle_l995_99515

theorem isosceles_triangle_base_angle (a b c : ℝ) (h : a + b + c = 180) (h_isosceles : b = c) (h_angle_a : a = 120) : b = 30 := 
by
  sorry

end isosceles_triangle_base_angle_l995_99515


namespace waiting_time_probability_l995_99518

-- Given conditions
def dep1 := 7 * 60 -- 7:00 in minutes
def dep2 := 7 * 60 + 30 -- 7:30 in minutes
def dep3 := 8 * 60 -- 8:00 in minutes

def arrival_start := 7 * 60 + 25 -- 7:25 in minutes
def arrival_end := 8 * 60 -- 8:00 in minutes
def total_time_window := arrival_end - arrival_start -- 35 minutes

def favorable_window1_start := 7 * 60 + 25 -- 7:25 in minutes
def favorable_window1_end := 7 * 60 + 30 -- 7:30 in minutes
def favorable_window2_start := 8 * 60 -- 8:00 in minutes
def favorable_window2_end := 8 * 60 + 10 -- 8:10 in minutes

def favorable_time_window := 
  (favorable_window1_end - favorable_window1_start) + 
  (favorable_window2_end - favorable_window2_start) -- 15 minutes

-- Probability calculation
theorem waiting_time_probability : 
  (favorable_time_window : ℚ) / (total_time_window : ℚ) = 3 / 7 :=
by
  sorry

end waiting_time_probability_l995_99518


namespace money_spent_correct_l995_99565

-- Define the number of plays, acts per play, wigs per act, and the cost of each wig
def num_plays := 3
def acts_per_play := 5
def wigs_per_act := 2
def wig_cost := 5
def sell_price := 4

-- Given the total number of wigs he drops and sells from one play
def dropped_plays := 1
def total_wigs_dropped := dropped_plays * acts_per_play * wigs_per_act
def money_from_selling_dropped_wigs := total_wigs_dropped * sell_price

-- Calculate the initial cost
def total_wigs := num_plays * acts_per_play * wigs_per_act
def initial_cost := total_wigs * wig_cost

-- The final spent money should be calculated by subtracting money made from selling the wigs of the dropped play
def final_spent_money := initial_cost - money_from_selling_dropped_wigs

-- Specify the expected amount of money John spent
def expected_final_spent_money := 110

theorem money_spent_correct :
  final_spent_money = expected_final_spent_money := by
  sorry

end money_spent_correct_l995_99565


namespace remainder_of_n_l995_99566

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 7 = 3) (h2 : n^3 % 7 = 6) : n % 7 = 5 :=
by
  sorry

end remainder_of_n_l995_99566


namespace michaels_brother_money_end_l995_99596

theorem michaels_brother_money_end 
  (michael_money : ℕ)
  (brother_money : ℕ)
  (gives_half : ℕ)
  (buys_candy : ℕ) 
  (h1 : michael_money = 42)
  (h2 : brother_money = 17)
  (h3 : gives_half = michael_money / 2)
  (h4 : buys_candy = 3) : 
  brother_money + gives_half - buys_candy = 35 :=
by {
  sorry
}

end michaels_brother_money_end_l995_99596


namespace employee_saves_l995_99514

-- Given conditions
def cost_price : ℝ := 500
def markup_percentage : ℝ := 0.15
def employee_discount_percentage : ℝ := 0.15

-- Definitions
def final_retail_price : ℝ := cost_price * (1 + markup_percentage)
def employee_discount_amount : ℝ := final_retail_price * employee_discount_percentage

-- Assertion
theorem employee_saves :
  employee_discount_amount = 86.25 := by
  sorry

end employee_saves_l995_99514


namespace last_digit_2008_pow_2005_l995_99568

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_2008_pow_2005 : last_digit (2008 ^ 2005) = 8 :=
by
  sorry

end last_digit_2008_pow_2005_l995_99568


namespace sqrt_expression_meaningful_l995_99571

theorem sqrt_expression_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) :=
by
  -- Proof will be skipped
  sorry

end sqrt_expression_meaningful_l995_99571


namespace range_a_A_intersect_B_empty_range_a_A_union_B_eq_B_l995_99594

-- Definition of the sets A and B
def A (a : ℝ) (x : ℝ) : Prop := a - 1 < x ∧ x < 2 * a + 1
def B (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Proving range of a for A ∩ B = ∅
theorem range_a_A_intersect_B_empty (a : ℝ) :
  (¬ ∃ x : ℝ, A a x ∧ B x) ↔ (a ≤ -2 ∨ a ≥ 2 ∨ (-2 < a ∧ a ≤ -1/2)) := sorry

-- Proving range of a for A ∪ B = B
theorem range_a_A_union_B_eq_B (a : ℝ) :
  (∀ x : ℝ, A a x ∨ B x → B x) ↔ (a ≤ -2) := sorry

end range_a_A_intersect_B_empty_range_a_A_union_B_eq_B_l995_99594


namespace triangle_centroid_l995_99555

theorem triangle_centroid :
  let (x1, y1) := (2, 6)
  let (x2, y2) := (6, 2)
  let (x3, y3) := (4, 8)
  let centroid_x := (x1 + x2 + x3) / 3
  let centroid_y := (y1 + y2 + y3) / 3
  (centroid_x, centroid_y) = (4, 16 / 3) :=
by
  let x1 := 2
  let y1 := 6
  let x2 := 6
  let y2 := 2
  let x3 := 4
  let y3 := 8
  let centroid_x := (x1 + x2 + x3) / 3
  let centroid_y := (y1 + y2 + y3) / 3
  show (centroid_x, centroid_y) = (4, 16 / 3)
  sorry

end triangle_centroid_l995_99555


namespace compute_ns_l995_99508

noncomputable def f : ℝ → ℝ :=
sorry

-- Defining the functional equation as a condition
def functional_equation (f : ℝ → ℝ) :=
∀ x y z : ℝ, f (x^2 + y^2 * f z) = x * f x + z * f (y^2)

-- Proving that the number of possible values of f(5) is 2
-- and their sum is 5, thus n * s = 10
theorem compute_ns (f : ℝ → ℝ) (hf : functional_equation f) : 2 * 5 = 10 :=
sorry

end compute_ns_l995_99508


namespace sequence_properties_l995_99520

-- Define the sequence according to the problem
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ (∀ n : ℕ, n ≥ 2 → a n = (n * a (n - 1)) / (n - 1))

-- State the theorem to be proved
theorem sequence_properties :
  ∃ (a : ℕ → ℕ), 
    seq a ∧ a 2 = 6 ∧ a 3 = 9 ∧ (∀ n : ℕ, n ≥ 1 → a n = 3 * n) :=
by
  -- Existence quantifier and properties (sequence definition, first three terms, and general term)
  sorry

end sequence_properties_l995_99520


namespace meal_total_cost_l995_99580

theorem meal_total_cost (x : ℝ) (h_initial: x/5 - 15 = x/8) : x = 200 :=
by sorry

end meal_total_cost_l995_99580


namespace roots_transformation_l995_99582

noncomputable def poly_with_roots (r₁ r₂ r₃ : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 - 5 * Polynomial.X ^ 2 + 10

noncomputable def transformed_poly_with_roots (r₁ r₂ r₃ : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 - 15 * Polynomial.X ^ 2 + 270

theorem roots_transformation (r₁ r₂ r₃ : ℝ) (h : poly_with_roots r₁ r₂ r₃ = 0) :
  transformed_poly_with_roots (3 * r₁) (3 * r₂) (3 * r₃) = Polynomial.X ^ 3 - 15 * Polynomial.X ^ 2 + 270 :=
by
  sorry

end roots_transformation_l995_99582


namespace ratio_of_trout_l995_99574

-- Definition of the conditions
def trout_caught_by_Sara : Nat := 5
def trout_caught_by_Melanie : Nat := 10

-- Theorem stating the main claim to be proved
theorem ratio_of_trout : trout_caught_by_Melanie / trout_caught_by_Sara = 2 := by
  sorry

end ratio_of_trout_l995_99574


namespace perpendicular_condition_l995_99542

def is_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_condition (a : ℝ) :
  is_perpendicular (a^2) (1/a) ↔ a = -1 :=
sorry

end perpendicular_condition_l995_99542


namespace num_unique_triangle_areas_correct_l995_99505

noncomputable def num_unique_triangle_areas : ℕ :=
  let A := 0
  let B := 1
  let C := 3
  let D := 6
  let E := 0
  let F := 2
  let base_lengths := [1, 2, 3, 5, 6]
  (base_lengths.eraseDups).length

theorem num_unique_triangle_areas_correct : num_unique_triangle_areas = 5 :=
  by sorry

end num_unique_triangle_areas_correct_l995_99505


namespace correct_factorization_l995_99521

-- Define the polynomial expressions
def polyA (x : ℝ) := x^3 - x
def factorA1 (x : ℝ) := x * (x^2 - 1)
def factorA2 (x : ℝ) := x * (x + 1) * (x - 1)

def polyB (a : ℝ) := 4 * a^2 - 4 * a + 1
def factorB (a : ℝ) := 4 * a * (a - 1) + 1

def polyC (x y : ℝ) := x^2 + y^2
def factorC (x y : ℝ) := (x + y)^2

def polyD (x : ℝ) := -3 * x + 6 * x^2 - 3 * x^3
def factorD (x : ℝ) := -3 * x * (x - 1)^2

-- Statement of the correctness of factorization D
theorem correct_factorization : ∀ (x : ℝ), polyD x = factorD x :=
by
  intro x
  sorry

end correct_factorization_l995_99521


namespace intersection_result_l995_99546

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | 0 ≤ x ∧ x < 5 }
def M_compl : Set ℝ := { x | x < 1 }

theorem intersection_result : N ∩ M_compl = { x | 0 ≤ x ∧ x < 1 } :=
by sorry

end intersection_result_l995_99546


namespace num_supervisors_correct_l995_99562

theorem num_supervisors_correct (S : ℕ) 
  (avg_sal_total : ℕ) (avg_sal_supervisor : ℕ) (avg_sal_laborer : ℕ) (num_laborers : ℕ)
  (h1 : avg_sal_total = 1250) 
  (h2 : avg_sal_supervisor = 2450) 
  (h3 : avg_sal_laborer = 950) 
  (h4 : num_laborers = 42) 
  (h5 : avg_sal_total = (39900 + S * avg_sal_supervisor) / (num_laborers + S)) : 
  S = 10 := by sorry

end num_supervisors_correct_l995_99562


namespace side_length_of_cloth_l995_99564

namespace ClothProblem

def original_side_length (trimming_x_sides trimming_y_sides remaining_area : ℤ) :=
  let x : ℤ := 12
  x

theorem side_length_of_cloth (x_trim y_trim remaining_area : ℤ) (h_trim_x : x_trim = 4) 
                             (h_trim_y : y_trim = 3) (h_area : remaining_area = 120) :
  original_side_length x_trim y_trim remaining_area = 12 :=
by
  sorry

end ClothProblem

end side_length_of_cloth_l995_99564


namespace rectangle_area_l995_99590

theorem rectangle_area (x : ℝ) (w l : ℝ) (h₁ : l = 3 * w) (h₂ : l^2 + w^2 = x^2) :
    l * w = (3 / 10) * x^2 :=
by
  sorry

end rectangle_area_l995_99590


namespace gf_3_eq_495_l995_99543

def f (x : ℝ) : ℝ := x^2 + 4
def g (x : ℝ) : ℝ := 3 * x^2 - x + 1

theorem gf_3_eq_495 : g (f 3) = 495 := by
  sorry

end gf_3_eq_495_l995_99543


namespace g_of_2_l995_99569

noncomputable def g : ℝ → ℝ := sorry

axiom cond1 (x y : ℝ) : x * g y = y * g x
axiom cond2 : g 10 = 30

theorem g_of_2 : g 2 = 6 := by
  sorry

end g_of_2_l995_99569


namespace no_simultaneous_squares_l995_99523

theorem no_simultaneous_squares (x y : ℕ) :
  ¬ (∃ a b : ℤ, x^2 + 2 * y = a^2 ∧ y^2 + 2 * x = b^2) :=
by
  sorry

end no_simultaneous_squares_l995_99523


namespace equations_neither_directly_nor_inversely_proportional_l995_99531

-- Definitions for equations
def equation1 (x y : ℝ) : Prop := 2 * x + 3 * y = 6
def equation2 (x y : ℝ) : Prop := 4 * x * y = 12
def equation3 (x y : ℝ) : Prop := y = 1/2 * x
def equation4 (x y : ℝ) : Prop := 5 * x - 2 * y = 20
def equation5 (x y : ℝ) : Prop := x / y = 5

-- Theorem stating that y is neither directly nor inversely proportional to x for the given equations
theorem equations_neither_directly_nor_inversely_proportional (x y : ℝ) :
  (¬∃ k : ℝ, x = k * y) ∧ (¬∃ k : ℝ, x * y = k) ↔ 
  (equation1 x y ∨ equation4 x y) :=
sorry

end equations_neither_directly_nor_inversely_proportional_l995_99531


namespace sample_freq_0_40_l995_99534

def total_sample_size : ℕ := 100
def freq_group_0_10 : ℕ := 12
def freq_group_10_20 : ℕ := 13
def freq_group_20_30 : ℕ := 24
def freq_group_30_40 : ℕ := 15
def freq_group_40_50 : ℕ := 16
def freq_group_50_60 : ℕ := 13
def freq_group_60_70 : ℕ := 7

theorem sample_freq_0_40 : (freq_group_0_10 + freq_group_10_20 + freq_group_20_30 + freq_group_30_40) / (total_sample_size : ℝ) = 0.64 := by
  sorry

end sample_freq_0_40_l995_99534


namespace CombinedHeightOfTowersIsCorrect_l995_99502

-- Define the heights as non-negative reals for clarity.
noncomputable def ClydeTowerHeight : ℝ := 5.0625
noncomputable def GraceTowerHeight : ℝ := 40.5
noncomputable def SarahTowerHeight : ℝ := 2 * ClydeTowerHeight
noncomputable def LindaTowerHeight : ℝ := (ClydeTowerHeight + GraceTowerHeight + SarahTowerHeight) / 3
noncomputable def CombinedHeight : ℝ := ClydeTowerHeight + GraceTowerHeight + SarahTowerHeight + LindaTowerHeight

-- State the theorem to be proven
theorem CombinedHeightOfTowersIsCorrect : CombinedHeight = 74.25 := 
by
  sorry

end CombinedHeightOfTowersIsCorrect_l995_99502


namespace smallest_n_for_Tn_gt_2006_over_2016_l995_99576

-- Definitions from the given problem
def Sn (n : ℕ) : ℚ := n^2 / (n + 1)
def an (n : ℕ) : ℚ := if n = 1 then 1 / 2 else Sn n - Sn (n - 1)
def bn (n : ℕ) : ℚ := an n / (n^2 + n - 1)

-- Definition of Tn sum
def Tn (n : ℕ) : ℚ := (Finset.range n).sum (λ k => bn (k + 1))

-- The main statement
theorem smallest_n_for_Tn_gt_2006_over_2016 : ∃ n : ℕ, Tn n > 2006 / 2016 := by
  sorry

end smallest_n_for_Tn_gt_2006_over_2016_l995_99576


namespace convex_pentagon_angle_greater_than_36_l995_99558

theorem convex_pentagon_angle_greater_than_36
  (α γ : ℝ)
  (h_sum : 5 * α + 10 * γ = 3 * Real.pi)
  (h_convex : ∀ i : Fin 5, (α + i.val * γ < Real.pi)) :
  α > Real.pi / 5 :=
sorry

end convex_pentagon_angle_greater_than_36_l995_99558


namespace x_cubed_plus_y_cubed_l995_99549

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^3 + y^3 = 85 / 2 :=
by
  sorry

end x_cubed_plus_y_cubed_l995_99549


namespace determine_g_10_l995_99552

noncomputable def g : ℝ → ℝ := sorry

-- Given condition
axiom g_condition : ∀ x y : ℝ, g x + g (2 * x + y) + 7 * x * y = g (3 * x - y) + 3 * x ^ 2 + 4

-- Theorem to prove
theorem determine_g_10 : g 10 = -46 := 
by
  -- skipping the proof here
  sorry

end determine_g_10_l995_99552


namespace kiwis_to_add_for_25_percent_oranges_l995_99591

theorem kiwis_to_add_for_25_percent_oranges :
  let oranges := 24
  let kiwis := 30
  let apples := 15
  let bananas := 20
  let total_fruits := oranges + kiwis + apples + bananas
  let target_total_fruits := (oranges : ℝ) / 0.25
  let fruits_to_add := target_total_fruits - (total_fruits : ℝ)
  fruits_to_add = 7 := by
  sorry

end kiwis_to_add_for_25_percent_oranges_l995_99591


namespace sequence_terms_l995_99509

/-- Given the sequence {a_n} with the sum of the first n terms S_n = n^2 - 3, 
    prove that a_1 = -2 and a_n = 2n - 1 for n ≥ 2. --/
theorem sequence_terms (a : ℕ → ℤ) (S : ℕ → ℤ)
  (hS : ∀ n : ℕ, S n = n^2 - 3)
  (h1 : ∀ n : ℕ, a n = S n - S (n - 1)) :
  a 1 = -2 ∧ (∀ n : ℕ, n ≥ 2 → a n = 2 * n - 1) :=
by {
  sorry
}

end sequence_terms_l995_99509


namespace value_two_stds_less_than_mean_l995_99554

theorem value_two_stds_less_than_mean (μ σ : ℝ) (hμ : μ = 16.5) (hσ : σ = 1.5) : (μ - 2 * σ) = 13.5 :=
by
  rw [hμ, hσ]
  norm_num

end value_two_stds_less_than_mean_l995_99554


namespace jackie_break_duration_l995_99579

noncomputable def push_ups_no_breaks : ℕ := 30

noncomputable def push_ups_with_breaks : ℕ := 22

noncomputable def total_breaks : ℕ := 2

theorem jackie_break_duration :
  (5 * 6 - push_ups_with_breaks) * (10 / 5) / total_breaks = 8 := by
-- Given that
-- 1) Jackie does 5 push-ups in 10 seconds
-- 2) Jackie takes 2 breaks in one minute and performs 22 push-ups
-- We need to prove the duration of each break
sorry

end jackie_break_duration_l995_99579


namespace number_of_parallelograms_l995_99547

-- Given conditions
def num_horizontal_lines : ℕ := 4
def num_vertical_lines : ℕ := 4

-- Mathematical function for combinations
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Proof statement
theorem number_of_parallelograms :
  binom num_horizontal_lines 2 * binom num_vertical_lines 2 = 36 :=
by
  sorry

end number_of_parallelograms_l995_99547


namespace count_valid_m_values_l995_99503

theorem count_valid_m_values : ∃ (count : ℕ), count = 72 ∧
  (∀ m : ℕ, 1 ≤ m ∧ m ≤ 5000 →
     (⌊Real.sqrt m⌋ = ⌊Real.sqrt (m+125)⌋)) ↔ count = 72 :=
by
  sorry

end count_valid_m_values_l995_99503


namespace arithmetic_sequence_sum_S15_l995_99584

theorem arithmetic_sequence_sum_S15 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hs5 : S 5 = 10) (hs10 : S 10 = 30) 
  (has : ∀ n, S n = n * (2 * a 1 + (n - 1) * a 2) / 2) : 
  S 15 = 60 := 
sorry

end arithmetic_sequence_sum_S15_l995_99584


namespace median_avg_scores_compare_teacher_avg_scores_l995_99595

-- Definitions of conditions
def class1_students (a : ℕ) := a
def class2_students (b : ℕ) := b
def class3_students (c : ℕ) := c
def class4_students (c : ℕ) := c

def avg_score_1 := 68
def avg_score_2 := 78
def avg_score_3 := 74
def avg_score_4 := 72

-- Part 1: Prove the median of the average scores.
theorem median_avg_scores : 
  let scores := [68, 72, 74, 78]
  ∃ m, m = 73 :=
by 
  sorry

-- Part 2: Prove that the average scores for Teacher Wang and Teacher Li are not necessarily the same.
theorem compare_teacher_avg_scores (a b c : ℕ) (h_ab : a ≠ 0 ∧ b ≠ 0) : 
  let wang_avg := (68 * a + 78 * b) / (a + b)
  let li_avg := 73
  wang_avg ≠ li_avg :=
by
  sorry

end median_avg_scores_compare_teacher_avg_scores_l995_99595


namespace determine_mass_l995_99583

noncomputable def mass_of_water 
  (P : ℝ) (t1 t2 : ℝ) (deltaT : ℝ) (cw : ℝ) : ℝ :=
  P * t1 / ((cw * deltaT) + ((cw * deltaT) / t2) * t1)

theorem determine_mass (P : ℝ) (t1 : ℝ) (deltaT : ℝ) (t2 : ℝ) (cw : ℝ) :
  P = 1000 → t1 = 120 → deltaT = 2 → t2 = 60 → cw = 4200 →
  mass_of_water P t1 deltaT t2 cw = 4.76 :=
by
  intros hP ht1 hdeltaT ht2 hcw
  sorry

end determine_mass_l995_99583


namespace minimum_moves_black_white_swap_l995_99525

-- Define an initial setup of the chessboard
def initial_positions_black := [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8)]
def initial_positions_white := [(8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (8,8)]

-- Define chess rules, positions, and switching places
def black_to_white_target := initial_positions_white
def white_to_black_target := initial_positions_black

-- Define a function to count minimal moves (trivial here just for the purpose of this statement)
def min_moves_to_switch_positions := 23

-- The main theorem statement proving necessity of at least 23 moves
theorem minimum_moves_black_white_swap :
  ∀ (black_positions white_positions : List (ℕ × ℕ)),
  black_positions = initial_positions_black →
  white_positions = initial_positions_white →
  min_moves_to_switch_positions ≥ 23 :=
by
  sorry

end minimum_moves_black_white_swap_l995_99525


namespace sum_of_youngest_and_oldest_l995_99553

-- Let a1, a2, a3, a4 be the ages of Janet's 4 children arranged in non-decreasing order.
-- Given conditions:
variable (a₁ a₂ a₃ a₄ : ℕ)
variable (h_mean : (a₁ + a₂ + a₃ + a₄) / 4 = 10)
variable (h_median : (a₂ + a₃) / 2 = 7)

-- Proof problem:
theorem sum_of_youngest_and_oldest :
  a₁ + a₄ = 26 :=
sorry

end sum_of_youngest_and_oldest_l995_99553


namespace proof_problem_l995_99539

variable {R : Type} [LinearOrderedField R]

def is_increasing (f : R → R) : Prop :=
  ∀ x y : R, x < y → f x < f y

theorem proof_problem (f : R → R) (a b : R) 
  (inc_f : is_increasing f) 
  (h : f a + f b > f (-a) + f (-b)) : 
  a + b > 0 := 
by
  sorry

end proof_problem_l995_99539


namespace total_acorns_l995_99504

theorem total_acorns (s_a : ℕ) (s_b : ℕ) (d : ℕ)
  (h1 : s_a = 7)
  (h2 : s_b = 5 * s_a)
  (h3 : s_b + 3 = d) :
  s_a + s_b + d = 80 :=
by
  sorry

end total_acorns_l995_99504


namespace natural_number_divisor_problem_l995_99532

theorem natural_number_divisor_problem (x y z : ℕ) (h1 : (y+1)*(z+1) = 30) 
    (h2 : (x+1)*(z+1) = 42) (h3 : (x+1)*(y+1) = 35) :
    (2^x * 3^y * 5^z = 2^6 * 3^5 * 5^4) :=
sorry

end natural_number_divisor_problem_l995_99532


namespace fraction_division_l995_99599

theorem fraction_division : (3/4) / (5/8) = (6/5) := by
  sorry

end fraction_division_l995_99599


namespace purely_imaginary_a_l995_99527

theorem purely_imaginary_a (a : ℝ) (h : (a^3 - a) = 0) (h2 : (a / (1 - a)) ≠ 0) : a = -1 := 
sorry

end purely_imaginary_a_l995_99527


namespace sum_of_solutions_eq_0_l995_99593

-- Define the conditions
def y : ℝ := 6
def main_eq (x : ℝ) : Prop := x^2 + y^2 = 145

-- State the theorem
theorem sum_of_solutions_eq_0 : 
  let x1 := Real.sqrt 109
  let x2 := -Real.sqrt 109
  x1 + x2 = 0 :=
by {
  sorry
}

end sum_of_solutions_eq_0_l995_99593


namespace select_rows_and_columns_l995_99556

theorem select_rows_and_columns (n : Nat) (pieces : Fin (2 * n) × Fin (2 * n) → Bool) :
  (∃ rows cols : Finset (Fin (2 * n)),
    rows.card = n ∧ cols.card = n ∧
    (∀ r c, r ∈ rows → c ∈ cols → pieces (r, c))) :=
sorry

end select_rows_and_columns_l995_99556


namespace perimeter_after_adding_tiles_l995_99506

-- Definition of the initial configuration
def initial_perimeter := 16

-- Definition of the number of additional tiles
def additional_tiles := 3

-- Statement of the problem: to prove that the new perimeter is 22
theorem perimeter_after_adding_tiles : initial_perimeter + 2 * additional_tiles = 22 := 
by 
  -- The number initially added each side exposed would increase the perimeter incremented by 6
  -- You can also assume the boundary conditions for the shared sides reducing.
  sorry

end perimeter_after_adding_tiles_l995_99506
