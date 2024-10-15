import Mathlib

namespace NUMINAMATH_GPT_largest_divisor_power_of_ten_l2017_201738

theorem largest_divisor_power_of_ten (N : ℕ) (m : ℕ) (k : ℕ) 
  (h1 : m ∣ N)
  (h2 : m < N)
  (h3 : N + m = 10^k) : N = 75 := sorry

end NUMINAMATH_GPT_largest_divisor_power_of_ten_l2017_201738


namespace NUMINAMATH_GPT_length_of_second_platform_is_correct_l2017_201701

-- Define the constants
def lt : ℕ := 70  -- Length of the train
def l1 : ℕ := 170  -- Length of the first platform
def t1 : ℕ := 15  -- Time to cross the first platform
def t2 : ℕ := 20  -- Time to cross the second platform

-- Calculate the speed of the train
def v : ℕ := (lt + l1) / t1

-- Define the length of the second platform
def l2 : ℕ := 250

-- The proof statement
theorem length_of_second_platform_is_correct : lt + l2 = v * t2 := sorry

end NUMINAMATH_GPT_length_of_second_platform_is_correct_l2017_201701


namespace NUMINAMATH_GPT_geom_prog_min_third_term_l2017_201739

theorem geom_prog_min_third_term :
  ∃ (d : ℝ), (-4 + 10 * Real.sqrt 6 = d ∨ -4 - 10 * Real.sqrt 6 = d) ∧
  (∀ x, x = 37 + 2 * d → x ≤ 29 - 20 * Real.sqrt 6) := 
sorry

end NUMINAMATH_GPT_geom_prog_min_third_term_l2017_201739


namespace NUMINAMATH_GPT_regression_estimate_l2017_201786

theorem regression_estimate (x : ℝ) (h : x = 28) : 4.75 * x + 257 = 390 :=
by
  rw [h]
  norm_num

end NUMINAMATH_GPT_regression_estimate_l2017_201786


namespace NUMINAMATH_GPT_distance_from_center_to_plane_correct_l2017_201788

noncomputable def distance_from_center_to_plane (O A B C : ℝ × ℝ × ℝ) (radius : ℝ) (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * K)
  let OD := Real.sqrt (radius^2 - R^2)
  OD

theorem distance_from_center_to_plane_correct (O A B C : ℝ × ℝ × ℝ) :
  (dist O A = 20) →
  (dist O B = 20) →
  (dist O C = 20) →
  (dist A B = 13) →
  (dist B C = 14) →
  (dist C A = 15) →
  let m := 15
  let n := 95
  let k := 8
  m + n + k = 118 := by
  sorry

end NUMINAMATH_GPT_distance_from_center_to_plane_correct_l2017_201788


namespace NUMINAMATH_GPT_geometric_sequence_term_l2017_201792

/-
Prove that the 303rd term in a geometric sequence with the first term a1 = 5 and the second term a2 = -10 is 5 * 2^302.
-/

theorem geometric_sequence_term :
  let a1 := 5
  let a2 := -10
  let r := a2 / a1
  let n := 303
  let a_n := a1 * r^(n-1)
  a_n = 5 * 2^302 :=
by
  let a1 := 5
  let a2 := -10
  let r := a2 / a1
  let n := 303
  have h1 : a1 * r^(n-1) = 5 * 2^302 := sorry
  exact h1

end NUMINAMATH_GPT_geometric_sequence_term_l2017_201792


namespace NUMINAMATH_GPT_tank_depth_is_six_l2017_201760

-- Definitions derived from the conditions
def tank_length : ℝ := 25
def tank_width : ℝ := 12
def plastering_cost_per_sq_meter : ℝ := 0.45
def total_cost : ℝ := 334.8

-- Compute the surface area to be plastered
def surface_area (d : ℝ) : ℝ := (tank_length * tank_width) + 2 * (tank_length * d) + 2 * (tank_width * d)

-- Equation relating the plastering cost to the surface area
def cost_equation (d : ℝ) : ℝ := plastering_cost_per_sq_meter * (surface_area d)

-- The mathematical result we need to prove
theorem tank_depth_is_six : ∃ d : ℝ, cost_equation d = total_cost ∧ d = 6 := by
  sorry

end NUMINAMATH_GPT_tank_depth_is_six_l2017_201760


namespace NUMINAMATH_GPT_debate_team_boys_l2017_201781

theorem debate_team_boys (total_groups : ℕ) (members_per_group : ℕ) (num_girls : ℕ) (total_members : ℕ) :
  total_groups = 8 →
  members_per_group = 4 →
  num_girls = 4 →
  total_members = total_groups * members_per_group →
  total_members - num_girls = 28 :=
by
  sorry

end NUMINAMATH_GPT_debate_team_boys_l2017_201781


namespace NUMINAMATH_GPT_dvd_packs_l2017_201756

theorem dvd_packs (cost_per_pack : ℕ) (discount_per_pack : ℕ) (money_available : ℕ) 
  (h_cost : cost_per_pack = 107) 
  (h_discount : discount_per_pack = 106) 
  (h_money : money_available = 93) : 
  (money_available / (cost_per_pack - discount_per_pack)) = 93 := 
by 
  -- Implementation of the proof goes here
  sorry

end NUMINAMATH_GPT_dvd_packs_l2017_201756


namespace NUMINAMATH_GPT_number_of_stones_l2017_201765

theorem number_of_stones (hall_length_m : ℕ) (hall_breadth_m : ℕ)
  (stone_length_dm : ℕ) (stone_breadth_dm : ℕ)
  (hall_length_dm_eq : hall_length_m * 10 = 360)
  (hall_breadth_dm_eq : hall_breadth_m * 10 = 150)
  (stone_length_eq : stone_length_dm = 6)
  (stone_breadth_eq : stone_breadth_dm = 5) :
  ((hall_length_m * 10) * (hall_breadth_m * 10)) / (stone_length_dm * stone_breadth_dm) = 1800 :=
by
  sorry

end NUMINAMATH_GPT_number_of_stones_l2017_201765


namespace NUMINAMATH_GPT_weight_of_new_student_l2017_201705

-- Define some constants for the problem
def avg_weight_29_students : ℝ := 28
def number_of_students_29 : ℕ := 29
def new_avg_weight_30_students : ℝ := 27.5
def number_of_students_30 : ℕ := 30

-- Calculate total weights
def total_weight_29_students : ℝ := avg_weight_29_students * number_of_students_29
def new_total_weight_30_students : ℝ := new_avg_weight_30_students * number_of_students_30

-- The proposition we need to prove
theorem weight_of_new_student :
  new_total_weight_30_students - total_weight_29_students = 13 := by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_weight_of_new_student_l2017_201705


namespace NUMINAMATH_GPT_total_banana_produce_correct_l2017_201719

-- Defining the conditions as variables and constants
def B_nearby : ℕ := 9000
def B_Jakies : ℕ := 10 * B_nearby
def T : ℕ := B_nearby + B_Jakies

-- Theorem statement
theorem total_banana_produce_correct : T = 99000 := by
  sorry  -- Proof placeholder

end NUMINAMATH_GPT_total_banana_produce_correct_l2017_201719


namespace NUMINAMATH_GPT_point_of_tangency_l2017_201732

def parabola1 (x y : ℝ) : Prop := y = x^2 + 15*x + 32
def parabola2 (x y : ℝ) : Prop := x = y^2 + 49*y + 593

theorem point_of_tangency :
  parabola1 (-7) (-24) ∧ parabola2 (-7) (-24) := by
  sorry

end NUMINAMATH_GPT_point_of_tangency_l2017_201732


namespace NUMINAMATH_GPT_compute_expression_l2017_201737

theorem compute_expression : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l2017_201737


namespace NUMINAMATH_GPT_find_a_l2017_201758

noncomputable def f (t : ℝ) (a : ℝ) : ℝ := (1 / (Real.cos t)) + (a / (1 - (Real.cos t)))

theorem find_a (t : ℝ) (a : ℝ) (h1 : 0 < t) (h2 : t < (Real.pi / 2)) (h3 : 0 < a) (h4 : ∀ t, 0 < t ∧ t < (Real.pi / 2) → f t a = 16) :
  a = 9 :=
sorry

end NUMINAMATH_GPT_find_a_l2017_201758


namespace NUMINAMATH_GPT_friend_selling_price_l2017_201745

-- Define the conditions
def CP : ℝ := 51136.36
def loss_percent : ℝ := 0.12
def gain_percent : ℝ := 0.20

-- Define the selling prices SP1 and SP2
def SP1 := CP * (1 - loss_percent)
def SP2 := SP1 * (1 + gain_percent)

-- State the theorem
theorem friend_selling_price : SP2 = 54000 := 
by sorry

end NUMINAMATH_GPT_friend_selling_price_l2017_201745


namespace NUMINAMATH_GPT_yao_ming_mcgrady_probability_l2017_201770

theorem yao_ming_mcgrady_probability
        (p : ℝ) (q : ℝ)
        (h1 : p = 0.8)
        (h2 : q = 0.7) :
        (2 * p * (1 - p)) * (2 * q * (1 - q)) = 0.1344 := 
by
  sorry

end NUMINAMATH_GPT_yao_ming_mcgrady_probability_l2017_201770


namespace NUMINAMATH_GPT_kite_perimeter_l2017_201730

-- Given the kite's diagonals, shorter sides, and longer sides
def diagonals : ℕ × ℕ := (12, 30)
def shorter_sides : ℕ := 10
def longer_sides : ℕ := 15

-- Problem statement: Prove that the perimeter is 50 inches
theorem kite_perimeter (diag1 diag2 short_len long_len : ℕ) 
                       (h_diag : diag1 = 12 ∧ diag2 = 30)
                       (h_short : short_len = 10)
                       (h_long : long_len = 15) : 
                       2 * short_len + 2 * long_len = 50 :=
by
  -- We provide no proof, only the statement
  sorry

end NUMINAMATH_GPT_kite_perimeter_l2017_201730


namespace NUMINAMATH_GPT_pool_perimeter_l2017_201799

theorem pool_perimeter (garden_length : ℝ) (plot_area : ℝ) (plot_count : ℕ) : 
  garden_length = 9 ∧ plot_area = 20 ∧ plot_count = 4 →
  ∃ (pool_perimeter : ℝ), pool_perimeter = 18 :=
by
  intros h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end NUMINAMATH_GPT_pool_perimeter_l2017_201799


namespace NUMINAMATH_GPT_min_value_of_2a_plus_3b_l2017_201750

theorem min_value_of_2a_plus_3b
  (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_perpendicular : (x - (2 * b - 3) * y + 6 = 0) ∧ (2 * b * x + a * y - 5 = 0)) :
  2 * a + 3 * b = 25 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_2a_plus_3b_l2017_201750


namespace NUMINAMATH_GPT_legs_in_room_l2017_201729

def total_legs_in_room (tables4 : Nat) (sofa : Nat) (chairs4 : Nat) (tables3 : Nat) (table1 : Nat) (rocking_chair2 : Nat) : Nat :=
  (tables4 * 4) + (sofa * 4) + (chairs4 * 4) + (tables3 * 3) + (table1 * 1) + (rocking_chair2 * 2)

theorem legs_in_room :
  total_legs_in_room 4 1 2 3 1 1 = 40 :=
by
  -- Skipping proof steps
  sorry

end NUMINAMATH_GPT_legs_in_room_l2017_201729


namespace NUMINAMATH_GPT_point_A_in_fourth_quadrant_l2017_201713

-- Defining the coordinates of point A
def x_A : ℝ := 2
def y_A : ℝ := -3

-- Defining the property of the quadrant
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Proposition stating point A is in the fourth quadrant
theorem point_A_in_fourth_quadrant : in_fourth_quadrant x_A y_A :=
by
  sorry

end NUMINAMATH_GPT_point_A_in_fourth_quadrant_l2017_201713


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2017_201767

-- Define the condition
variable (a : ℝ)

-- Theorem statement: $a > 0$ is a sufficient but not necessary condition for $a^2 > 0$
theorem sufficient_but_not_necessary_condition : 
  (a > 0 → a^2 > 0) ∧ (¬ (a > 0) → a^2 > 0) :=
  by
    sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2017_201767


namespace NUMINAMATH_GPT_find_m_of_hyperbola_l2017_201703

theorem find_m_of_hyperbola (m : ℝ) (h : mx^2 + y^2 = 1) (s : ∃ x : ℝ, x = 2) : m = -4 := 
by
  sorry

end NUMINAMATH_GPT_find_m_of_hyperbola_l2017_201703


namespace NUMINAMATH_GPT_average_age_of_adults_l2017_201785

theorem average_age_of_adults (n_total n_girls n_boys n_adults : ℕ) 
                              (avg_age_total avg_age_girls avg_age_boys avg_age_adults : ℕ)
                              (h1 : n_total = 60)
                              (h2 : avg_age_total = 18)
                              (h3 : n_girls = 30)
                              (h4 : avg_age_girls = 16)
                              (h5 : n_boys = 20)
                              (h6 : avg_age_boys = 17)
                              (h7 : n_adults = 10) :
                              avg_age_adults = 26 :=
sorry

end NUMINAMATH_GPT_average_age_of_adults_l2017_201785


namespace NUMINAMATH_GPT_valid_pairs_iff_l2017_201768

noncomputable def valid_pairs (a b : ℝ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ a * (⌊ b * n ⌋ : ℝ) = b * (⌊ a * n ⌋ : ℝ)

theorem valid_pairs_iff (a b : ℝ) : valid_pairs a b ↔
  (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ (m n : ℤ), a = m ∧ b = n)) :=
by sorry

end NUMINAMATH_GPT_valid_pairs_iff_l2017_201768


namespace NUMINAMATH_GPT_gcd_square_product_l2017_201769

theorem gcd_square_product (x y z : ℕ) (h : 1 / (x : ℝ) - 1 / (y : ℝ) = 1 / (z : ℝ)) : 
    ∃ n : ℕ, gcd x (gcd y z) * x * y * z = n * n := 
sorry

end NUMINAMATH_GPT_gcd_square_product_l2017_201769


namespace NUMINAMATH_GPT_number_of_yellow_marbles_l2017_201716

theorem number_of_yellow_marbles 
  (total_marbles : ℕ) 
  (red_marbles : ℕ) 
  (blue_marbles : ℕ) 
  (yellow_marbles : ℕ)
  (h1 : total_marbles = 85) 
  (h2 : red_marbles = 14) 
  (h3 : blue_marbles = 3 * red_marbles) 
  (h4 : yellow_marbles = total_marbles - (red_marbles + blue_marbles)) :
  yellow_marbles = 29 :=
  sorry

end NUMINAMATH_GPT_number_of_yellow_marbles_l2017_201716


namespace NUMINAMATH_GPT_amount_of_flour_per_new_bread_roll_l2017_201754

theorem amount_of_flour_per_new_bread_roll :
  (24 * (1 / 8) = 3) → (16 * f = 3) → (f = 3 / 16) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_amount_of_flour_per_new_bread_roll_l2017_201754


namespace NUMINAMATH_GPT_number_with_at_least_two_zeros_l2017_201727

-- A 6-digit number can have for its leftmost digit anything from 1 to 9 inclusive,
-- and for each of its next five digits anything from 0 through 9 inclusive.
def total_6_digit_numbers : ℕ := 9 * 10^5

-- A 6-digit number with no zeros consists solely of digits from 1 to 9
def no_zero : ℕ := 9^6

-- A 6-digit number with exactly one zero
def exactly_one_zero : ℕ := 5 * 9^5

-- The number of 6-digit numbers with less than two zeros is the sum of no_zero and exactly_one_zero
def less_than_two_zeros : ℕ := no_zero + exactly_one_zero

-- The number of 6-digit numbers with at least two zeros is the difference between total_6_digit_numbers and less_than_two_zeros
def at_least_two_zeros : ℕ := total_6_digit_numbers - less_than_two_zeros

-- The theorem that states the number of 6-digit numbers with at least two zeros is 73,314
theorem number_with_at_least_two_zeros : at_least_two_zeros = 73314 := 
by
  sorry

end NUMINAMATH_GPT_number_with_at_least_two_zeros_l2017_201727


namespace NUMINAMATH_GPT_number_of_selected_in_interval_l2017_201771

noncomputable def systematic_sampling_group := (420: ℕ)
noncomputable def selected_people := (21: ℕ)
noncomputable def interval_start := (241: ℕ)
noncomputable def interval_end := (360: ℕ)
noncomputable def sampling_interval := systematic_sampling_group / selected_people
noncomputable def interval_length := interval_end - interval_start + 1

theorem number_of_selected_in_interval :
  interval_length / sampling_interval = 6 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_number_of_selected_in_interval_l2017_201771


namespace NUMINAMATH_GPT_A_min_votes_for_victory_l2017_201766

theorem A_min_votes_for_victory:
  ∀ (initial_votes_A initial_votes_B initial_votes_C total_votes remaining_votes min_votes_A: ℕ),
  initial_votes_A = 350 →
  initial_votes_B = 370 →
  initial_votes_C = 280 →
  total_votes = 1500 →
  remaining_votes = 500 →
  min_votes_A = 261 →
  initial_votes_A + min_votes_A > initial_votes_B + (remaining_votes - min_votes_A) :=
by
  intros _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_A_min_votes_for_victory_l2017_201766


namespace NUMINAMATH_GPT_box_dimensions_l2017_201708

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) : 
  a = 5 ∧ b = 8 ∧ c = 12 := 
by
  sorry

end NUMINAMATH_GPT_box_dimensions_l2017_201708


namespace NUMINAMATH_GPT_percent_decrease_l2017_201774

theorem percent_decrease (original_price sale_price : ℝ) (h₀ : original_price = 100) (h₁ : sale_price = 30) :
  (original_price - sale_price) / original_price * 100 = 70 :=
by
  rw [h₀, h₁]
  norm_num

end NUMINAMATH_GPT_percent_decrease_l2017_201774


namespace NUMINAMATH_GPT_triangles_combined_area_is_96_l2017_201780

noncomputable def combined_area_of_triangles : Prop :=
  let length_rectangle : ℝ := 6
  let width_rectangle : ℝ := 4
  let area_rectangle : ℝ := length_rectangle * width_rectangle
  let ratio_rectangle_to_first_triangle : ℝ := 2 / 5
  let area_first_triangle : ℝ := (5 / 2) * area_rectangle
  let x : ℝ := area_first_triangle / 5
  let base_second_triangle : ℝ := 8
  let height_second_triangle : ℝ := 9  -- calculated height based on the area ratio
  let area_second_triangle : ℝ := (base_second_triangle * height_second_triangle) / 2
  let combined_area : ℝ := area_first_triangle + area_second_triangle
  combined_area = 96

theorem triangles_combined_area_is_96 : combined_area_of_triangles := by
  sorry

end NUMINAMATH_GPT_triangles_combined_area_is_96_l2017_201780


namespace NUMINAMATH_GPT_evaluate_expression_l2017_201749

theorem evaluate_expression : ∀ (a b c d : ℤ), 
  a = 3 →
  b = a + 3 →
  c = b - 8 →
  d = a + 5 →
  (a + 2 ≠ 0) →
  (b - 4 ≠ 0) →
  (c + 5 ≠ 0) →
  (d - 3 ≠ 0) →
  ((a + 3) * (b - 2) * (c + 9) * (d + 1) = 1512 * (a + 2) * (b - 4) * (c + 5) * (d - 3)) :=
by
  intros a b c d ha hb hc hd ha2 hb4 hc5 hd3
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2017_201749


namespace NUMINAMATH_GPT_simplify_evaluate_l2017_201722

noncomputable def a := (1 / 2) + Real.sqrt (1 / 2)

theorem simplify_evaluate (a : ℝ) (h : a = (1 / 2) + Real.sqrt (1 / 2)) :
  (a + Real.sqrt 3) * (a - Real.sqrt 3) - a * (a - 6) = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_simplify_evaluate_l2017_201722


namespace NUMINAMATH_GPT_cistern_fill_time_l2017_201707

/--
  A cistern can be filled by tap A in 4 hours,
  emptied by tap B in 6 hours,
  and filled by tap C in 3 hours.
  If all the taps are opened simultaneously,
  then the cistern will be filled in exactly 2.4 hours.
-/
theorem cistern_fill_time :
  let rate_A := 1 / 4
  let rate_B := -1 / 6
  let rate_C := 1 / 3
  let combined_rate := rate_A + rate_B + rate_C
  let fill_time := 1 / combined_rate
  fill_time = 2.4 := by
  sorry

end NUMINAMATH_GPT_cistern_fill_time_l2017_201707


namespace NUMINAMATH_GPT_cost_of_carpeting_l2017_201796

noncomputable def cost_per_meter_in_paise (cost : ℝ) (length_in_meters : ℝ) : ℝ :=
  cost * 100 / length_in_meters

theorem cost_of_carpeting (room_length room_breadth carpet_width_m cost_total : ℝ) (h1 : room_length = 15) 
  (h2 : room_breadth = 6) (h3 : carpet_width_m = 0.75) (h4 : cost_total = 36) :
  cost_per_meter_in_paise cost_total (room_length * room_breadth / carpet_width_m) = 30 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_carpeting_l2017_201796


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l2017_201706

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 1) (h2 : 0 < m) (h3 : 0 < n) : 
  (1/m + 1/n) = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l2017_201706


namespace NUMINAMATH_GPT_dice_probability_sum_12_l2017_201714

open Nat

/-- Probability that the sum of three six-faced dice rolls equals 12 is 10 / 216 --/
theorem dice_probability_sum_12 : 
  let outcomes := 6^3
  let favorable := 10
  (favorable : ℚ) / outcomes = 10 / 216 := 
by
  let outcomes := 6^3
  let favorable := 10
  sorry

end NUMINAMATH_GPT_dice_probability_sum_12_l2017_201714


namespace NUMINAMATH_GPT_steve_correct_operations_l2017_201772

theorem steve_correct_operations (x : ℕ) (h1 : x / 8 - 20 = 12) : ((x * 8) + 20) = 2068 :=
by
  sorry

end NUMINAMATH_GPT_steve_correct_operations_l2017_201772


namespace NUMINAMATH_GPT_jason_quarters_l2017_201717

def quarters_original := 49
def quarters_added := 25
def quarters_total := 74

theorem jason_quarters : quarters_original + quarters_added = quarters_total :=
by
  sorry

end NUMINAMATH_GPT_jason_quarters_l2017_201717


namespace NUMINAMATH_GPT_triangle_area_is_two_l2017_201784

noncomputable def triangle_area (b c : ℝ) (angle_A : ℝ) : ℝ :=
  (1 / 2) * b * c * Real.sin angle_A

theorem triangle_area_is_two
  (A B C : ℝ) (a b c : ℝ)
  (hA : A = π / 4)
  (hCondition : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B)
  (hBC : b * c = 4 * Real.sqrt 2) : 
  triangle_area b c A = 2 :=
by
  -- actual proof omitted
  sorry

end NUMINAMATH_GPT_triangle_area_is_two_l2017_201784


namespace NUMINAMATH_GPT_find_10th_integer_l2017_201779

-- Defining the conditions
def avg_20_consecutive_integers (avg : ℝ) : Prop :=
  avg = 23.65

def consecutive_integer_sequence (n : ℤ) (a : ℤ) : Prop :=
  a = n + 9

-- The main theorem statement
theorem find_10th_integer (n : ℤ) (avg : ℝ) (h_avg : avg_20_consecutive_integers avg) (h_seq : consecutive_integer_sequence n 23) :
  n = 14 :=
sorry

end NUMINAMATH_GPT_find_10th_integer_l2017_201779


namespace NUMINAMATH_GPT_roots_square_difference_l2017_201734

theorem roots_square_difference (a b : ℚ)
  (ha : 6 * a^2 + 13 * a - 28 = 0)
  (hb : 6 * b^2 + 13 * b - 28 = 0) : (a - b)^2 = 841 / 36 :=
sorry

end NUMINAMATH_GPT_roots_square_difference_l2017_201734


namespace NUMINAMATH_GPT_fuel_consumption_new_model_l2017_201795

variable (d_old : ℝ) (d_new : ℝ) (c_old : ℝ) (c_new : ℝ)

theorem fuel_consumption_new_model :
  (d_new = d_old + 4.4) →
  (c_new = c_old - 2) →
  (c_old = 100 / d_old) →
  d_old = 12.79 →
  c_new = 5.82 :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_fuel_consumption_new_model_l2017_201795


namespace NUMINAMATH_GPT_ratio_is_five_to_three_l2017_201755

variable (g b : ℕ)

def girls_more_than_boys : Prop := g - b = 6
def total_pupils : Prop := g + b = 24
def ratio_girls_to_boys : ℚ := g / b

theorem ratio_is_five_to_three (h1 : girls_more_than_boys g b) (h2 : total_pupils g b) : ratio_girls_to_boys g b = 5 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_is_five_to_three_l2017_201755


namespace NUMINAMATH_GPT_find_x_when_y4_l2017_201747

theorem find_x_when_y4 
  (k : ℝ) 
  (h_var : ∀ y : ℝ, ∃ x : ℝ, x = k * y^2)
  (h_initial : ∃ x : ℝ, x = 6 ∧ 1 = k) :
  ∃ x : ℝ, x = 96 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_when_y4_l2017_201747


namespace NUMINAMATH_GPT_total_number_of_toys_is_105_l2017_201775

-- Definitions
variables {a k : ℕ}

-- Conditions
def condition_1 (a k : ℕ) : Prop := k ≥ 2
def katya_toys (a : ℕ) : ℕ := a
def lena_toys (a k : ℕ) : ℕ := k * a
def masha_toys (a k : ℕ) : ℕ := k^2 * a

def after_katya_gave_toys (a : ℕ) : ℕ := a - 2
def after_lena_received_toys (a k : ℕ) : ℕ := k * a + 5
def after_masha_gave_toys (a k : ℕ) : ℕ := k^2 * a - 3

def arithmetic_progression (x1 x2 x3 : ℕ) : Prop :=
  2 * x2 = x1 + x3

-- Problem statement to prove
theorem total_number_of_toys_is_105 (a k : ℕ) (h1 : condition_1 a k)
  (h2 : arithmetic_progression (after_katya_gave_toys a) (after_lena_received_toys a k) (after_masha_gave_toys a k)) :
  katya_toys a + lena_toys a k + masha_toys a k = 105 :=
sorry

end NUMINAMATH_GPT_total_number_of_toys_is_105_l2017_201775


namespace NUMINAMATH_GPT_flour_maximum_weight_l2017_201710

/-- Given that the bag of flour is marked with 25kg + 50g, prove that the maximum weight of the flour is 25.05kg. -/
theorem flour_maximum_weight :
  let weight_kg := 25
  let weight_g := 50
  (weight_kg + (weight_g / 1000 : ℝ)) = 25.05 :=
by 
  -- provide definitions
  let weight_kg := 25
  let weight_g := 50
  have : (weight_kg + (weight_g / 1000 : ℝ)) = 25.05 := sorry
  exact this

end NUMINAMATH_GPT_flour_maximum_weight_l2017_201710


namespace NUMINAMATH_GPT_sequence_general_term_l2017_201759

theorem sequence_general_term {a : ℕ → ℚ} 
  (h₀ : a 1 = 1) 
  (h₁ : ∀ n ≥ 2, a n = 3 * a (n - 1) / (a (n - 1) + 3)) : 
  ∀ n, a n = 3 / (n + 2) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l2017_201759


namespace NUMINAMATH_GPT_probability_of_one_each_color_is_two_fifths_l2017_201748

/-- Definition for marbles bag containing 2 red, 2 blue, and 2 green marbles -/
structure MarblesBag where
  red : ℕ
  blue : ℕ
  green : ℕ
  total : ℕ := red + blue + green

/-- Initial setup for the problem -/
def initialBag : MarblesBag := { red := 2, blue := 2, green := 2 }

/-- Represents the outcome of selecting marbles without replacement -/
def selectMarbles (bag : MarblesBag) (count : ℕ) : ℕ :=
  Nat.choose bag.total count

/-- The number of ways to select one marble of each color -/
def selectOneOfEachColor (bag : MarblesBag) : ℕ :=
  bag.red * bag.blue * bag.green

/-- Calculate the probability of selecting one marble of each color -/
def probabilityOneOfEachColor (bag : MarblesBag) (selectCount : ℕ) : ℚ :=
  selectOneOfEachColor bag / selectMarbles bag selectCount

/-- Theorem stating the answer to the probability problem -/
theorem probability_of_one_each_color_is_two_fifths (bag : MarblesBag) :
  probabilityOneOfEachColor bag 3 = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_probability_of_one_each_color_is_two_fifths_l2017_201748


namespace NUMINAMATH_GPT_angle_conversion_l2017_201728

theorem angle_conversion :
  (12 * (Real.pi / 180)) = (Real.pi / 15) := by
  sorry

end NUMINAMATH_GPT_angle_conversion_l2017_201728


namespace NUMINAMATH_GPT_arith_seq_sum_correct_l2017_201797

-- Define the arithmetic sequence given the first term and common difference
def arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def arith_seq_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

-- Given Problem Conditions
def a₁ := -5
def d := 3
def n := 20

-- Theorem: Sum of the first 20 terms of the arithmetic sequence is 470
theorem arith_seq_sum_correct : arith_seq_sum a₁ d n = 470 :=
  sorry

end NUMINAMATH_GPT_arith_seq_sum_correct_l2017_201797


namespace NUMINAMATH_GPT_smaller_angle_at_8_15_l2017_201700

def angle_minute_hand_at_8_15: ℝ := 90
def angle_hour_hand_at_8: ℝ := 240
def additional_angle_hour_hand_at_8_15: ℝ := 7.5
def total_angle_hour_hand_at_8_15 := angle_hour_hand_at_8 + additional_angle_hour_hand_at_8_15

theorem smaller_angle_at_8_15 :
  min (abs (total_angle_hour_hand_at_8_15 - angle_minute_hand_at_8_15))
      (abs (360 - (total_angle_hour_hand_at_8_15 - angle_minute_hand_at_8_15))) = 157.5 :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_at_8_15_l2017_201700


namespace NUMINAMATH_GPT_c_rent_share_l2017_201757

-- Definitions based on conditions
def a_oxen := 10
def a_months := 7
def b_oxen := 12
def b_months := 5
def c_oxen := 15
def c_months := 3
def total_rent := 105

-- Calculate the shares in ox-months
def share_a := a_oxen * a_months
def share_b := b_oxen * b_months
def share_c := c_oxen * c_months

-- Calculate the total ox-months
def total_ox_months := share_a + share_b + share_c

-- Calculate the rent per ox-month
def rent_per_ox_month := total_rent / total_ox_months

-- Calculate the amount C should pay
def amount_c_should_pay := share_c * rent_per_ox_month

-- Prove the statement
theorem c_rent_share : amount_c_should_pay = 27 := by
  sorry

end NUMINAMATH_GPT_c_rent_share_l2017_201757


namespace NUMINAMATH_GPT_polynomial_value_l2017_201751

theorem polynomial_value :
  let a := -4
  let b := 23
  let c := -17
  let d := 10
  5 * a + 3 * b + 2 * c + d = 25 :=
by
  let a := -4
  let b := 23
  let c := -17
  let d := 10
  sorry

end NUMINAMATH_GPT_polynomial_value_l2017_201751


namespace NUMINAMATH_GPT_sum_of_first_five_terms_is_31_l2017_201726

variable (a : ℕ → ℝ) (q : ℝ)

-- The geometric sequence a_n
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Condition 1: a_2 * a_3 = 2 * a_1
def condition1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 3 = 2 * a 1

-- Condition 2: The arithmetic mean of a_4 and 2 * a_7 is 5/4
def condition2 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 4 + 2 * a 7) / 2 = 5 / 4

-- Sum of the first 5 terms of the geometric sequence
def S_5 (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3 + a 4

-- The theorem to prove
theorem sum_of_first_five_terms_is_31 (a : ℕ → ℝ) (q : ℝ) 
  (hg : is_geometric_sequence a q) 
  (hc1 : condition1 a q) 
  (hc2 : condition2 a q) : 
  S_5 a = 31 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_five_terms_is_31_l2017_201726


namespace NUMINAMATH_GPT_total_meters_examined_l2017_201725

theorem total_meters_examined (total_meters : ℝ) (h : 0.10 * total_meters = 12) :
  total_meters = 120 :=
sorry

end NUMINAMATH_GPT_total_meters_examined_l2017_201725


namespace NUMINAMATH_GPT_probability_of_union_l2017_201793

-- Define the range of two-digit numbers
def digit_count : ℕ := 90

-- Define events A and B
def event_a (n : ℕ) : Prop := n % 2 = 0
def event_b (n : ℕ) : Prop := n % 5 = 0

-- Define the probabilities P(A), P(B), and P(A ∩ B)
def P_A : ℚ := 45 / digit_count
def P_B : ℚ := 18 / digit_count
def P_A_and_B : ℚ := 9 / digit_count

-- Prove the final probability using inclusion-exclusion principle
theorem probability_of_union : P_A + P_B - P_A_and_B = 0.6 := by
  sorry

end NUMINAMATH_GPT_probability_of_union_l2017_201793


namespace NUMINAMATH_GPT_pattern_D_cannot_form_tetrahedron_l2017_201763

theorem pattern_D_cannot_form_tetrahedron :
  (¬ ∃ (f : ℝ × ℝ → ℝ × ℝ),
      f (0, 0) = (1, 1) ∧ f (1, 0) = (1, -1) ∧ f (2, 0) = (-1, 1) ∧ f (3, 0) = (-1, -1)) :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_pattern_D_cannot_form_tetrahedron_l2017_201763


namespace NUMINAMATH_GPT_extra_apples_l2017_201746

-- Defining the given conditions
def redApples : Nat := 60
def greenApples : Nat := 34
def studentsWantFruit : Nat := 7

-- Defining the theorem to prove the number of extra apples
theorem extra_apples : redApples + greenApples - studentsWantFruit = 87 := by
  sorry

end NUMINAMATH_GPT_extra_apples_l2017_201746


namespace NUMINAMATH_GPT_infinite_solutions_b_value_l2017_201711

-- Given condition for the equation to hold
def equation_condition (x b : ℤ) : Prop :=
  4 * (3 * x - b) = 3 * (4 * x + 16)

-- The statement we need to prove: b = -12
theorem infinite_solutions_b_value :
  (∀ x : ℤ, equation_condition x b) → b = -12 :=
sorry

end NUMINAMATH_GPT_infinite_solutions_b_value_l2017_201711


namespace NUMINAMATH_GPT_marissas_sunflower_height_in_meters_l2017_201704

-- Define the conversion factors
def inches_per_foot : ℝ := 12
def cm_per_inch : ℝ := 2.54
def cm_per_meter : ℝ := 100

-- Define the given data
def sister_height_feet : ℝ := 4.15
def additional_height_cm : ℝ := 37
def height_difference_inches : ℝ := 63

-- Calculate the height of Marissa's sunflower in meters
theorem marissas_sunflower_height_in_meters :
  let sister_height_inches := sister_height_feet * inches_per_foot
  let sister_height_cm := sister_height_inches * cm_per_inch
  let total_sister_height_cm := sister_height_cm + additional_height_cm
  let height_difference_cm := height_difference_inches * cm_per_inch
  let marissas_sunflower_height_cm := total_sister_height_cm + height_difference_cm
  let marissas_sunflower_height_m := marissas_sunflower_height_cm / cm_per_meter
  marissas_sunflower_height_m = 3.23512 :=
by
  sorry

end NUMINAMATH_GPT_marissas_sunflower_height_in_meters_l2017_201704


namespace NUMINAMATH_GPT_solve_for_q_l2017_201720

theorem solve_for_q : 
  let n : ℤ := 63
  let m : ℤ := 14
  ∀ (q : ℤ),
  (7 : ℤ) / 9 = n / 81 ∧
  (7 : ℤ) / 9 = (m + n) / 99 ∧
  (7 : ℤ) / 9 = (q - m) / 135 → 
  q = 119 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_q_l2017_201720


namespace NUMINAMATH_GPT_unique_intersection_of_line_and_parabola_l2017_201736

theorem unique_intersection_of_line_and_parabola :
  ∃! k : ℚ, ∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k → k = 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_unique_intersection_of_line_and_parabola_l2017_201736


namespace NUMINAMATH_GPT_car_speed_decrease_l2017_201741

theorem car_speed_decrease (d : ℝ) (speed_first : ℝ) (distance_fifth : ℝ) (time_interval : ℝ) :
  speed_first = 45 ∧ distance_fifth = 4.4 ∧ time_interval = 8 / 60 ∧ speed_first - 4 * d = distance_fifth / time_interval -> d = 3 :=
by
  intros h
  obtain ⟨_, _, _, h_eq⟩ := h
  sorry

end NUMINAMATH_GPT_car_speed_decrease_l2017_201741


namespace NUMINAMATH_GPT_tom_sold_4_books_l2017_201773

-- Definitions based on conditions from the problem
def initial_books : ℕ := 5
def new_books : ℕ := 38
def final_books : ℕ := 39

-- The number of books Tom sold
def books_sold (S : ℕ) : Prop := initial_books - S + new_books = final_books

-- Our goal is to prove that Tom sold 4 books
theorem tom_sold_4_books : books_sold 4 :=
  by
    -- Implicitly here would be the proof, but we use sorry to skip it
    sorry

end NUMINAMATH_GPT_tom_sold_4_books_l2017_201773


namespace NUMINAMATH_GPT_balance_equation_l2017_201764

variable (G Y W B : ℝ)
variable (balance1 : 4 * G = 8 * B)
variable (balance2 : 3 * Y = 7.5 * B)
variable (balance3 : 8 * B = 6 * W)

theorem balance_equation : 5 * G + 3 * Y + 4 * W = 23.5 * B := by
  sorry

end NUMINAMATH_GPT_balance_equation_l2017_201764


namespace NUMINAMATH_GPT_largest_multiple_of_11_less_than_minus_150_l2017_201762

theorem largest_multiple_of_11_less_than_minus_150 : 
  ∃ n : ℤ, (n * 11 < -150) ∧ (∀ m : ℤ, (m * 11 < -150) →  n * 11 ≥ m * 11) ∧ (n * 11 = -154) :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_11_less_than_minus_150_l2017_201762


namespace NUMINAMATH_GPT_acme_vowel_soup_l2017_201724

-- Define the set of vowels
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

-- Define the number of each vowel
def num_vowels (v : Char) : ℕ := 5

-- Define a function to count the number of five-letter words
def count_five_letter_words : ℕ :=
  (vowels.card) ^ 5

-- Theorem to be proven
theorem acme_vowel_soup :
  count_five_letter_words = 3125 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_acme_vowel_soup_l2017_201724


namespace NUMINAMATH_GPT_wall_volume_is_128512_l2017_201787

noncomputable def wall_volume (width : ℝ) (height : ℝ) (length : ℝ) : ℝ :=
  width * height * length

theorem wall_volume_is_128512 : 
  ∀ (w : ℝ) (h : ℝ) (l : ℝ), 
  h = 6 * w ∧ l = 7 * h ∧ w = 8 → 
  wall_volume w h l = 128512 := 
by
  sorry

end NUMINAMATH_GPT_wall_volume_is_128512_l2017_201787


namespace NUMINAMATH_GPT_intersection_points_of_lines_l2017_201731

theorem intersection_points_of_lines :
  (∃ (x y : ℚ), 2 * y - 3 * x = 4 ∧ x + 3 * y = 3 ∧ x = 10 / 11 ∧ y = 13 / 11) ∧
  (∃ (x y : ℚ), 2 * y - 3 * x = 4 ∧ 5 * x - 3 * y = 6 ∧ x = 24 ∧ y = 38) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_of_lines_l2017_201731


namespace NUMINAMATH_GPT_distance_between_A_and_B_l2017_201789

-- Definitions and conditions
variables {A B C : Type}    -- Locations
variables {v1 v2 : ℕ}       -- Speeds of person A and person B
variables {distanceAB : ℕ}  -- Distance we want to find

noncomputable def first_meet_condition (v1 v2 : ℕ) : Prop :=
  ∃ t : ℕ, (v1 * t - 108 = v2 * t - 100)

noncomputable def second_meet_condition (v1 v2 distanceAB : ℕ) : Prop :=
  distanceAB = 3750

-- Theorem statement
theorem distance_between_A_and_B (v1 v2 distanceAB : ℕ) :
  first_meet_condition v1 v2 → second_meet_condition v1 v2 distanceAB →
  distanceAB = 3750 :=
by
  intros _ _ 
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l2017_201789


namespace NUMINAMATH_GPT_number_of_friends_shared_with_l2017_201702

-- Conditions and given data
def doughnuts_samuel : ℕ := 2 * 12
def doughnuts_cathy : ℕ := 3 * 12
def total_doughnuts : ℕ := doughnuts_samuel + doughnuts_cathy
def each_person_doughnuts : ℕ := 6
def total_people := total_doughnuts / each_person_doughnuts
def samuel_and_cathy : ℕ := 2

-- Statement to prove - Number of friends they shared with
theorem number_of_friends_shared_with : (total_people - samuel_and_cathy) = 8 := by
  sorry

end NUMINAMATH_GPT_number_of_friends_shared_with_l2017_201702


namespace NUMINAMATH_GPT_abs_neg_eq_iff_nonpos_l2017_201761

theorem abs_neg_eq_iff_nonpos (a : ℝ) : |a| = -a ↔ a ≤ 0 :=
by sorry

end NUMINAMATH_GPT_abs_neg_eq_iff_nonpos_l2017_201761


namespace NUMINAMATH_GPT_sequence_inequality_l2017_201753

-- Define the problem
theorem sequence_inequality (a : ℕ → ℕ) (h0 : ∀ n, 0 < a n) (h1 : a 1 > a 0) (h2 : ∀ n ≥ 2, a n = 3 * a (n-1) - 2 * a (n-2)) : a 100 > 2^99 :=
by
  sorry

end NUMINAMATH_GPT_sequence_inequality_l2017_201753


namespace NUMINAMATH_GPT_min_buses_needed_l2017_201715

theorem min_buses_needed (students : ℕ) (cap1 cap2 : ℕ) (h_students : students = 530) (h_cap1 : cap1 = 40) (h_cap2 : cap2 = 45) :
  min (Nat.ceil (students / cap1)) (Nat.ceil (students / cap2)) = 12 :=
  sorry

end NUMINAMATH_GPT_min_buses_needed_l2017_201715


namespace NUMINAMATH_GPT_dice_sum_probability_l2017_201733

-- Define a noncomputable function to calculate the number of ways to get a sum of 15
noncomputable def dice_sum_ways (dices : ℕ) (sides : ℕ) (target_sum : ℕ) : ℕ :=
  sorry

-- Define the Lean 4 statement
theorem dice_sum_probability :
  dice_sum_ways 5 6 15 = 2002 :=
sorry

end NUMINAMATH_GPT_dice_sum_probability_l2017_201733


namespace NUMINAMATH_GPT_express_w_l2017_201744

theorem express_w (w a b c : ℝ) (x y z : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ w ∧ b ≠ w ∧ c ≠ w)
  (h1 : x + y + z = 1)
  (h2 : x * a^2 + y * b^2 + z * c^2 = w^2)
  (h3 : x * a^3 + y * b^3 + z * c^3 = w^3)
  (h4 : x * a^4 + y * b^4 + z * c^4 = w^4) :
  w = - (a * b * c) / (a * b + b * c + c * a) :=
sorry

end NUMINAMATH_GPT_express_w_l2017_201744


namespace NUMINAMATH_GPT_triangle_rational_segments_l2017_201776

theorem triangle_rational_segments (a b c : ℚ) (h : a + b > c ∧ a + c > b ∧ b + c > a):
  ∃ (ab1 cb1 : ℚ), (ab1 + cb1 = b) := sorry

end NUMINAMATH_GPT_triangle_rational_segments_l2017_201776


namespace NUMINAMATH_GPT_avg_age_of_children_l2017_201794

theorem avg_age_of_children 
  (participants : ℕ) (women : ℕ) (men : ℕ) (children : ℕ)
  (overall_avg_age : ℕ) (avg_age_women : ℕ) (avg_age_men : ℕ)
  (hp : participants = 50) (hw : women = 22) (hm : men = 18) (hc : children = 10)
  (ho : overall_avg_age = 20) (haw : avg_age_women = 24) (ham : avg_age_men = 19) :
  ∃ (avg_age_children : ℕ), avg_age_children = 13 :=
by
  -- Proof will be here.
  sorry

end NUMINAMATH_GPT_avg_age_of_children_l2017_201794


namespace NUMINAMATH_GPT_second_bill_late_fee_l2017_201743

def first_bill_amount : ℕ := 200
def first_bill_interest_rate : ℝ := 0.10
def first_bill_months : ℕ := 2
def second_bill_amount : ℕ := 130
def second_bill_months : ℕ := 6
def third_bill_first_month_fee : ℕ := 40
def third_bill_second_month_fee : ℕ := 80
def total_amount_owed : ℕ := 1234

theorem second_bill_late_fee (x : ℕ) 
(h : first_bill_amount * (first_bill_interest_rate * first_bill_months) + first_bill_amount + third_bill_first_month_fee + third_bill_second_month_fee + second_bill_amount + second_bill_months * x = total_amount_owed) : x = 124 :=
sorry

end NUMINAMATH_GPT_second_bill_late_fee_l2017_201743


namespace NUMINAMATH_GPT_abc_product_l2017_201709

theorem abc_product (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b = 13) (h2 : b * c = 52) (h3 : c * a = 4) : a * b * c = 52 := 
  sorry

end NUMINAMATH_GPT_abc_product_l2017_201709


namespace NUMINAMATH_GPT_chengdu_chongqing_scientific_notation_l2017_201778

theorem chengdu_chongqing_scientific_notation:
  (185000 : ℝ) = 1.85 * 10^5 :=
sorry

end NUMINAMATH_GPT_chengdu_chongqing_scientific_notation_l2017_201778


namespace NUMINAMATH_GPT_coloring_scheme_formula_l2017_201790

noncomputable def number_of_coloring_schemes (m n : ℕ) : ℕ :=
  if h : (m ≥ 2) ∧ (n ≥ 2) then
    m * ((-1 : ℤ)^n * (m - 2 : ℤ)).natAbs + (m - 2)^n
  else 0

-- Formal statement verifying the formula for coloring schemes
theorem coloring_scheme_formula (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) :
  number_of_coloring_schemes m n = m * ((-1 : ℤ)^n * (m - 2 : ℤ)).natAbs + (m - 2)^n :=
by sorry

end NUMINAMATH_GPT_coloring_scheme_formula_l2017_201790


namespace NUMINAMATH_GPT_alpha_minus_beta_l2017_201721

theorem alpha_minus_beta {α β : ℝ} (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
    (h_cos_alpha : Real.cos α = 2 * Real.sqrt 5 / 5) 
    (h_cos_beta : Real.cos β = Real.sqrt 10 / 10) : 
    α - β = -π / 4 := 
sorry

end NUMINAMATH_GPT_alpha_minus_beta_l2017_201721


namespace NUMINAMATH_GPT_gcd_polynomial_l2017_201798

theorem gcd_polynomial (b : ℤ) (hb : ∃ k : ℤ, b = 570 * k) :
  Int.gcd (5 * b^3 + b^2 + 6 * b + 95) b = 95 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_l2017_201798


namespace NUMINAMATH_GPT_positive_integer_satisfies_condition_l2017_201742

theorem positive_integer_satisfies_condition : 
  ∃ n : ℕ, (12 * n = n^2 + 36) ∧ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_satisfies_condition_l2017_201742


namespace NUMINAMATH_GPT_cliff_rock_collection_l2017_201723

theorem cliff_rock_collection (S I : ℕ) 
  (h1 : I = S / 2) 
  (h2 : 2 * I / 3 = 40) : S + I = 180 := by
  sorry

end NUMINAMATH_GPT_cliff_rock_collection_l2017_201723


namespace NUMINAMATH_GPT_total_grapes_is_157_l2017_201735

def number_of_grapes_in_robs_bowl : ℕ := 25

def number_of_grapes_in_allies_bowl : ℕ :=
  number_of_grapes_in_robs_bowl + 5

def number_of_grapes_in_allyns_bowl : ℕ :=
  2 * number_of_grapes_in_allies_bowl - 2

def number_of_grapes_in_sams_bowl : ℕ :=
  (number_of_grapes_in_allies_bowl + number_of_grapes_in_allyns_bowl) / 2

def total_number_of_grapes : ℕ :=
  number_of_grapes_in_robs_bowl +
  number_of_grapes_in_allies_bowl +
  number_of_grapes_in_allyns_bowl +
  number_of_grapes_in_sams_bowl

theorem total_grapes_is_157 : total_number_of_grapes = 157 :=
  sorry

end NUMINAMATH_GPT_total_grapes_is_157_l2017_201735


namespace NUMINAMATH_GPT_arccos_neg_one_eq_pi_l2017_201777

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end NUMINAMATH_GPT_arccos_neg_one_eq_pi_l2017_201777


namespace NUMINAMATH_GPT_sum_in_base_4_l2017_201791

theorem sum_in_base_4 : 
  let n1 := 2
  let n2 := 23
  let n3 := 132
  let n4 := 1320
  let sum := 20200
  n1 + n2 + n3 + n4 = sum := 
by
  sorry

end NUMINAMATH_GPT_sum_in_base_4_l2017_201791


namespace NUMINAMATH_GPT_largest_r_l2017_201783

theorem largest_r (p q r : ℝ) (h1 : p + q + r = 6) (h2 : p*q + p*r + q*r = 8) : 
  r ≤ 2 + Real.sqrt (20/3) := 
sorry

end NUMINAMATH_GPT_largest_r_l2017_201783


namespace NUMINAMATH_GPT_min_value_of_a_is_five_l2017_201718

-- Given: a, b, c in table satisfying the conditions
-- We are to prove that the minimum value of a is 5.
theorem min_value_of_a_is_five
  {a b c: ℤ} (h_pos: 0 < a) (hx_distinct: 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧ 
                               a*x₁^2 + b*x₁ + c = 0 ∧ 
                               a*x₂^2 + b*x₂ + c = 0) (hb_neg: b < 0) 
                               (h_disc_pos: (b^2 - 4*a*c) > 0) : a = 5 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_is_five_l2017_201718


namespace NUMINAMATH_GPT_Gwen_avg_speed_trip_l2017_201752

theorem Gwen_avg_speed_trip : 
  ∀ (d1 d2 s1 s2 t1 t2 : ℝ), 
  d1 = 40 → d2 = 40 → s1 = 15 → s2 = 30 →
  d1 / s1 = t1 → d2 / s2 = t2 →
  (d1 + d2) / (t1 + t2) = 20 :=
by 
  intros d1 d2 s1 s2 t1 t2 hd1 hd2 hs1 hs2 ht1 ht2
  sorry

end NUMINAMATH_GPT_Gwen_avg_speed_trip_l2017_201752


namespace NUMINAMATH_GPT_tangent_line_at_one_l2017_201712

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + Real.log x

theorem tangent_line_at_one (a : ℝ)
  (h : ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → |(f a x - f a 1) / (x - 1) - 3| < ε) :
  ∃ m b, m = 3 ∧ b = -2 ∧ (∀ x y, y = f a x → m * x = y + b) := sorry

end NUMINAMATH_GPT_tangent_line_at_one_l2017_201712


namespace NUMINAMATH_GPT_number_of_female_students_l2017_201782

theorem number_of_female_students
    (F : ℕ)  -- Number of female students
    (avg_all : ℝ)  -- Average score for all students
    (avg_male : ℝ)  -- Average score for male students
    (avg_female : ℝ)  -- Average score for female students
    (num_male : ℕ)  -- Number of male students
    (h_avg_all : avg_all = 90)
    (h_avg_male : avg_male = 82)
    (h_avg_female : avg_female = 92)
    (h_num_male : num_male = 8)
    (h_avg : avg_all * (num_male + F) = avg_male * num_male + avg_female * F) :
  F = 32 :=
by
  sorry

end NUMINAMATH_GPT_number_of_female_students_l2017_201782


namespace NUMINAMATH_GPT_garden_ratio_2_l2017_201740

theorem garden_ratio_2 :
  ∃ (P C k R : ℤ), 
      P = 237 ∧ 
      C = P - 60 ∧ 
      P + C + k = 768 ∧ 
      R = k / C ∧ 
      R = 2 := 
by
  sorry

end NUMINAMATH_GPT_garden_ratio_2_l2017_201740
