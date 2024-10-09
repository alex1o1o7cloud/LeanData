import Mathlib

namespace B_subset_A_implies_m_le_5_l2427_242783

variable (A B : Set ℝ)
variable (m : ℝ)

def setA : Set ℝ := {x | 4 ≤ x ∧ x ≤ 8}
def setB (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2 * m - 2}

theorem B_subset_A_implies_m_le_5 :
  B ⊆ A → (∀ k : ℝ, k ∈ setB m → k ∈ setA) → m ≤ 5 :=
by
  sorry

end B_subset_A_implies_m_le_5_l2427_242783


namespace james_points_l2427_242758

theorem james_points (x : ℕ) :
  13 * 3 + 20 * x = 79 → x = 2 :=
by
  sorry

end james_points_l2427_242758


namespace set_diff_M_N_l2427_242753

def set_diff {α : Type} (A B : Set α) : Set α := {x | x ∈ A ∧ x ∉ B}

def M : Set ℝ := {x | |x + 1| ≤ 2}

def N : Set ℝ := {x | ∃ α : ℝ, x = |Real.sin α| }

theorem set_diff_M_N :
  set_diff M N = {x | -3 ≤ x ∧ x < 0} :=
by
  sorry

end set_diff_M_N_l2427_242753


namespace total_chocolate_bars_in_large_box_l2427_242733

-- Define the given conditions
def small_boxes : ℕ := 16
def chocolate_bars_per_box : ℕ := 25

-- State the proof problem
theorem total_chocolate_bars_in_large_box :
  small_boxes * chocolate_bars_per_box = 400 :=
by
  -- The proof is omitted
  sorry

end total_chocolate_bars_in_large_box_l2427_242733


namespace tennis_player_games_l2427_242719

theorem tennis_player_games (b : ℕ → ℕ) (h1 : ∀ k, b k ≥ k) (h2 : ∀ k, b k ≤ 12 * (k / 7)) :
  ∃ i j : ℕ, i < j ∧ b j - b i = 20 :=
by
  sorry

end tennis_player_games_l2427_242719


namespace max_distance_unit_circle_l2427_242746

open Complex

theorem max_distance_unit_circle : 
  ∀ (z : ℂ), abs z = 1 → ∃ M : ℝ, M = abs (z - (1 : ℂ) - I) ∧ ∀ w : ℂ, abs w = 1 → abs (w - 1 - I) ≤ M :=
by
  sorry

end max_distance_unit_circle_l2427_242746


namespace p_sq_plus_q_sq_l2427_242716

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 :=
by
  sorry

end p_sq_plus_q_sq_l2427_242716


namespace no_valid_conference_division_l2427_242795

theorem no_valid_conference_division (num_teams : ℕ) (matches_per_team : ℕ) :
  num_teams = 30 → matches_per_team = 82 → 
  ¬ ∃ (k : ℕ) (x y z : ℕ), k + (num_teams - k) = num_teams ∧
                          x + y + z = (num_teams * matches_per_team) / 2 ∧
                          z = ((x + y + z) / 2) := 
by
  sorry

end no_valid_conference_division_l2427_242795


namespace max_area_of_garden_l2427_242799

theorem max_area_of_garden (total_fence : ℝ) (gate : ℝ) (remaining_fence := total_fence - gate) :
  total_fence = 60 → gate = 4 → (remaining_fence / 2) * (remaining_fence / 2) = 196 :=
by 
  sorry

end max_area_of_garden_l2427_242799


namespace sufficient_but_not_necessary_condition_l2427_242749

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, |2*x - 1| ≤ x → x^2 + x - 2 ≤ 0) ∧ 
  ¬(∀ x : ℝ, x^2 + x - 2 ≤ 0 → |2 * x - 1| ≤ x) := sorry

end sufficient_but_not_necessary_condition_l2427_242749


namespace P_roots_implies_Q_square_roots_l2427_242770

noncomputable def P (x : ℝ) : ℝ := x^3 - 2 * x + 1

noncomputable def Q (x : ℝ) : ℝ := x^3 - 4 * x^2 + 4 * x - 1

theorem P_roots_implies_Q_square_roots (r : ℝ) (h : P r = 0) : Q (r^2) = 0 := sorry

end P_roots_implies_Q_square_roots_l2427_242770


namespace unoccupied_seats_in_business_class_l2427_242782

/-
Define the numbers for each class and the number of people in each.
-/
def first_class_seats : Nat := 10
def business_class_seats : Nat := 30
def economy_class_seats : Nat := 50
def people_in_first_class : Nat := 3
def people_in_economy_class : Nat := economy_class_seats / 2
def people_in_business_and_first_class : Nat := people_in_economy_class
def people_in_business_class : Nat := people_in_business_and_first_class - people_in_first_class

/-
Prove that the number of unoccupied seats in business class is 8.
-/
theorem unoccupied_seats_in_business_class :
  business_class_seats - people_in_business_class = 8 :=
sorry

end unoccupied_seats_in_business_class_l2427_242782


namespace books_left_to_read_l2427_242789

theorem books_left_to_read (total_books : ℕ) (books_mcgregor : ℕ) (books_floyd : ℕ) : total_books = 89 → books_mcgregor = 34 → books_floyd = 32 → 
  (total_books - (books_mcgregor + books_floyd) = 23) :=
by
  intros h1 h2 h3
  sorry

end books_left_to_read_l2427_242789


namespace line_through_points_l2427_242754

-- Define the conditions and the required proof statement
theorem line_through_points (x1 y1 z1 x2 y2 z2 x y z m n p : ℝ) :
  (∃ m n p, (x-x1) / m = (y-y1) / n ∧ (y-y1) / n = (z-z1) / p) → 
  (x-x1) / (x2 - x1) = (y-y1) / (y2 - y1) ∧ 
  (y-y1) / (y2 - y1) = (z-z1) / (z2 - z1) :=
sorry

end line_through_points_l2427_242754


namespace pure_water_to_achieve_desired_concentration_l2427_242744

theorem pure_water_to_achieve_desired_concentration :
  ∀ (w : ℝ), (50 + w ≠ 0) → (0.4 * 50 / (50 + w) = 0.25) → w = 30 := 
by
  intros w h_nonzero h_concentration
  sorry

end pure_water_to_achieve_desired_concentration_l2427_242744


namespace cube_surface_area_l2427_242773

-- Define the volume condition
def volume (s : ℕ) : ℕ := s^3

-- Define the surface area function
def surface_area (s : ℕ) : ℕ := 6 * s^2

-- State the theorem to be proven
theorem cube_surface_area (s : ℕ) (h : volume s = 729) : surface_area s = 486 :=
by
  sorry

end cube_surface_area_l2427_242773


namespace Q_equals_10_04_l2427_242720
-- Import Mathlib for mathematical operations and equivalence checking

-- Define the given conditions
def a := 6
def b := 3
def c := 2

-- Define the expression to be evaluated
def Q : ℚ := (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 - b*c + c^2)

-- Prove that the expression equals 10.04
theorem Q_equals_10_04 : Q = 10.04 := by
  -- Proof goes here
  sorry

end Q_equals_10_04_l2427_242720


namespace largest_prime_factor_of_1729_l2427_242785

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l2427_242785


namespace blue_chip_value_l2427_242790

noncomputable def yellow_chip_value := 2
noncomputable def green_chip_value := 5
noncomputable def total_product_value := 16000
noncomputable def num_yellow_chips := 4

def blue_chip_points (b n : ℕ) :=
  yellow_chip_value ^ num_yellow_chips * b ^ n * green_chip_value ^ n = total_product_value

theorem blue_chip_value (b : ℕ) (n : ℕ) (h : blue_chip_points b n) (hn : b^n = 8) : b = 8 :=
by
  have h1 : ∀ k : ℕ, k ^ n = 8 → k = 8 ∧ n = 3 := sorry
  exact (h1 b hn).1

end blue_chip_value_l2427_242790


namespace susan_age_is_11_l2427_242762

theorem susan_age_is_11 (S A : ℕ) 
  (h1 : A = S + 5) 
  (h2 : A + S = 27) : 
  S = 11 := 
by 
  sorry

end susan_age_is_11_l2427_242762


namespace log_problem_l2427_242776

open Real

noncomputable def lg (x : ℝ) := log x / log 10

theorem log_problem :
  lg 2 ^ 2 + lg 2 * lg 5 + lg 5 = 1 :=
by
  sorry

end log_problem_l2427_242776


namespace walking_representation_l2427_242703

-- Definitions based on conditions
def represents_walking_eastward (m : ℤ) : Prop := m > 0

-- The theorem to prove based on the problem statement
theorem walking_representation :
  represents_walking_eastward 5 →
  ¬ represents_walking_eastward (-10) ∧ abs (-10) = 10 :=
by
  sorry

end walking_representation_l2427_242703


namespace circumscribed_circle_radius_l2427_242712

noncomputable def radius_of_circumcircle (a b c : ℚ) (h_a : a = 15/2) (h_b : b = 10) (h_c : c = 25/2) : ℚ :=
if h_triangle : a^2 + b^2 = c^2 then (c / 2) else 0

theorem circumscribed_circle_radius :
  radius_of_circumcircle (15/2 : ℚ) 10 (25/2 : ℚ) (by norm_num) (by norm_num) (by norm_num) = 25 / 4 := 
by
  sorry

end circumscribed_circle_radius_l2427_242712


namespace find_h_l2427_242775

noncomputable def h (x : ℝ) : ℝ := -x^4 - 2 * x^3 + 4 * x^2 + 9 * x - 5

def f (x : ℝ) : ℝ := x^4 + 2 * x^3 - x^2 - 4 * x + 1

def p (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 4

theorem find_h (x : ℝ) : (f x) + (h x) = p x :=
by sorry

end find_h_l2427_242775


namespace relationship_among_a_b_c_l2427_242769

variable (x y : ℝ)
variable (hx_pos : x > 0) (hy_pos : y > 0) (hxy_ne : x ≠ y)

noncomputable def a := (x + y) / 2
noncomputable def b := Real.sqrt (x * y)
noncomputable def c := 2 / ((1 / x) + (1 / y))

theorem relationship_among_a_b_c :
    a > b ∧ b > c := by
    sorry

end relationship_among_a_b_c_l2427_242769


namespace seeds_per_can_l2427_242774

theorem seeds_per_can (total_seeds : ℝ) (number_of_cans : ℝ) (h1 : total_seeds = 54.0) (h2 : number_of_cans = 9.0) : (total_seeds / number_of_cans = 6.0) :=
by
  rw [h1, h2]
  norm_num
  -- sorry

end seeds_per_can_l2427_242774


namespace symmetric_circle_eq_of_given_circle_eq_l2427_242798

theorem symmetric_circle_eq_of_given_circle_eq
  (x y : ℝ)
  (eq1 : (x - 1)^2 + (y - 2)^2 = 1)
  (line_eq : y = x) :
  (x - 2)^2 + (y - 1)^2 = 1 := by
  sorry

end symmetric_circle_eq_of_given_circle_eq_l2427_242798


namespace fraction_simplification_l2427_242792

theorem fraction_simplification :
    1 + (1 / (1 + (1 / (2 + (1 / 3))))) = 17 / 10 := by
  sorry

end fraction_simplification_l2427_242792


namespace smallest_w_l2427_242702

theorem smallest_w (w : ℕ) (h1 : 1916 = 2^2 * 479) (h2 : w > 0) : w = 74145392000 ↔ 
  (∀ p e, (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11) → (∃ k, (1916 * w = p^e * k ∧ e ≥ if p = 2 then 6 else 3))) :=
sorry

end smallest_w_l2427_242702


namespace tom_seashells_l2427_242751

theorem tom_seashells (days : ℕ) (seashells_per_day : ℕ) (h1 : days = 5) (h2 : seashells_per_day = 7) : 
  seashells_per_day * days = 35 := 
by
  sorry

end tom_seashells_l2427_242751


namespace triangle_equilateral_l2427_242778

variables {A B C : ℝ} -- angles of the triangle
variables {a b c : ℝ} -- sides opposite to the angles

-- Given conditions
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C = c * Real.cos A ∧ (b * b = a * c)

-- The proof goal
theorem triangle_equilateral (A B C : ℝ) (a b c : ℝ) :
  triangle A B C a b c → a = b ∧ b = c :=
sorry

end triangle_equilateral_l2427_242778


namespace home_run_difference_l2427_242767

def hank_aaron_home_runs : ℕ := 755
def dave_winfield_home_runs : ℕ := 465

theorem home_run_difference :
  2 * dave_winfield_home_runs - hank_aaron_home_runs = 175 := by
  sorry

end home_run_difference_l2427_242767


namespace factor_expression_l2427_242755

theorem factor_expression (z : ℂ) : 
  75 * z^12 + 162 * z^24 + 27 = 3 * (9 + z^12 * (25 + 54 * z^12)) :=
sorry

end factor_expression_l2427_242755


namespace evaluate_polynomial_l2427_242788

theorem evaluate_polynomial (x : ℝ) (h : x^2 - 3 * x - 9 = 0) (hx : 0 < x) : 
  x^4 - 3 * x^3 - 9 * x^2 + 27 * x - 8 = -8 := 
sorry

end evaluate_polynomial_l2427_242788


namespace sequence_is_arithmetic_max_value_a_n_b_n_l2427_242731

open Real

theorem sequence_is_arithmetic (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (Sn : ℕ → ℝ) 
  (h_Sn : ∀ n, Sn n = (a n ^ 2 + a n) / 2) :
    ∀ n, a n = n := sorry 

theorem max_value_a_n_b_n (a b : ℕ → ℝ)
  (h_b : ∀ n, b n = - n + 5)
  (h_a : ∀ n, a n = n) :
    ∀ n, n ≥ 2 → n ≤ 3 → 
    ∃ k, a k * b k = 25 / 4 := by 
      sorry

end sequence_is_arithmetic_max_value_a_n_b_n_l2427_242731


namespace basic_computer_price_l2427_242760

variables (C P : ℕ)

theorem basic_computer_price (h1 : C + P = 2500)
                            (h2 : C + 500 + P = 6 * P) : C = 2000 :=
by
  sorry

end basic_computer_price_l2427_242760


namespace min_value_of_expr_l2427_242700

noncomputable def min_expr (a b c : ℝ) := (2 * a / b) + (3 * b / c) + (4 * c / a)

theorem min_value_of_expr (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) 
    (habc : a * b * c = 1) : 
  min_expr a b c ≥ 9 := 
sorry

end min_value_of_expr_l2427_242700


namespace cannot_inscribe_good_tetrahedron_in_good_parallelepiped_l2427_242729

-- Definitions related to the problem statements
def good_tetrahedron (V S : ℝ) := V = S

def good_parallelepiped (V' S1 S2 S3 : ℝ) := V' = 2 * (S1 + S2 + S3)

-- Theorem statement
theorem cannot_inscribe_good_tetrahedron_in_good_parallelepiped
  (V V' S : ℝ) (S1 S2 S3 : ℝ) (h1 h2 h3 : ℝ)
  (HT : good_tetrahedron V S)
  (HP : good_parallelepiped V' S1 S2 S3)
  (Hheights : S1 ≥ S2 ∧ S2 ≥ S3) :
  ¬ (V = S ∧ V' = 2 * (S1 + S2 + S3) ∧ h1 > 6 * S1 ∧ h2 > 6 * S2 ∧ h3 > 6 * S3) := 
sorry

end cannot_inscribe_good_tetrahedron_in_good_parallelepiped_l2427_242729


namespace real_solutions_count_l2427_242706

theorem real_solutions_count :
  (∃ x : ℝ, |x - 2| - 4 = 1 / |x - 3|) ∧
  (∃ y : ℝ, |y - 2| - 4 = 1 / |y - 3| ∧ x ≠ y) :=
sorry

end real_solutions_count_l2427_242706


namespace solution_set_inequalities_l2427_242796

theorem solution_set_inequalities (a b x : ℝ) (h1 : ∃ x, x > a ∧ x < b) :
  (x < 1 - a ∧ x < 1 - b) ↔ x < 1 - b :=
by
  sorry

end solution_set_inequalities_l2427_242796


namespace problem_part1_problem_part2_l2427_242737

theorem problem_part1 (k m : ℝ) :
  (∀ x : ℝ, (|k|-3) * x^2 - (k-3) * x + 2*m + 1 = 0 → (|k|-3 = 0 ∧ k ≠ 3)) →
  k = -3 :=
sorry

theorem problem_part2 (k m : ℝ) :
  ((∃ x1 x2 : ℝ, 
     ((|k|-3) * x1^2 - (k-3) * x1 + 2*m + 1 = 0) ∧
     (3 * x2 - 2 = 4 - 5 * x2 + 2 * x2) ∧
     x1 = -x2) →
  (∀ x : ℝ, (|k|-3) * x^2 - (k-3) * x + 2*m + 1 = 0 → (|k|-3 = 0 ∧ x = -1)) →
  (k = -3 ∧ m = 5/2)) :=
sorry

end problem_part1_problem_part2_l2427_242737


namespace xiao_ming_returns_and_distance_is_correct_l2427_242718

theorem xiao_ming_returns_and_distance_is_correct :
  ∀ (walk_distance : ℝ) (turn_angle : ℝ), 
  walk_distance = 5 ∧ turn_angle = 20 → 
  (∃ n : ℕ, (360 % turn_angle = 0) ∧ n = 360 / turn_angle ∧ walk_distance * n = 90) :=
by
  sorry

end xiao_ming_returns_and_distance_is_correct_l2427_242718


namespace factor_expression_l2427_242704

theorem factor_expression (x : ℝ) : 3 * x^2 + 12 * x + 12 = 3 * (x + 2) ^ 2 :=
by sorry

end factor_expression_l2427_242704


namespace initial_paint_amount_l2427_242791

theorem initial_paint_amount (P : ℝ) (h1 : P > 0) (h2 : (1 / 4) * P + (1 / 3) * (3 / 4) * P = 180) : P = 360 := by
  sorry

end initial_paint_amount_l2427_242791


namespace six_digit_numbers_with_zero_count_l2427_242722

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l2427_242722


namespace xiao_dong_actual_jump_distance_l2427_242734

-- Conditions are defined here
def standard_jump_distance : ℝ := 4.00
def xiao_dong_recorded_result : ℝ := -0.32

-- Here we structure our problem
theorem xiao_dong_actual_jump_distance :
  standard_jump_distance + xiao_dong_recorded_result = 3.68 :=
by
  sorry

end xiao_dong_actual_jump_distance_l2427_242734


namespace twice_x_minus_three_lt_zero_l2427_242757

theorem twice_x_minus_three_lt_zero (x : ℝ) : (2 * x - 3 < 0) ↔ (2 * x < 3) :=
by
  sorry

end twice_x_minus_three_lt_zero_l2427_242757


namespace find_p_q_coprime_sum_l2427_242745

theorem find_p_q_coprime_sum (x y n m: ℕ) (h_sum: x + y = 30)
  (h_prob: ((n/x) * (n-1)/(x-1) * (n-2)/(x-2)) * ((m/y) * (m-1)/(y-1) * (m-2)/(y-2)) = 18/25)
  : ∃ p q : ℕ, p.gcd q = 1 ∧ p + q = 1006 :=
by
  sorry

end find_p_q_coprime_sum_l2427_242745


namespace probability_of_opposite_middle_vertex_l2427_242708

noncomputable def ant_moves_to_opposite_middle_vertex_prob : ℚ := 1 / 2

-- Specification of the problem conditions
structure Octahedron :=
  (middle_vertices : Finset ℕ) -- Assume some identification of middle vertices
  (adjacent_vertices : ℕ → Finset ℕ) -- Function mapping a vertex to its adjacent vertices
  (is_middle_vertex : ℕ → Prop) -- Predicate to check if a vertex is a middle vertex
  (is_top_or_bottom_vertex : ℕ → Prop) -- Predicate to check if a vertex is a top or bottom vertex
  (start_vertex : ℕ)

variables (O : Octahedron)

-- Main theorem statement
theorem probability_of_opposite_middle_vertex :
  ∃ A B : ℕ, A ∈ O.adjacent_vertices O.start_vertex ∧ B ∈ O.adjacent_vertices A ∧ B ≠ O.start_vertex ∧ (∃ x ∈ O.middle_vertices, x = B) →
  (∀ (A B : ℕ), (A ∈ O.adjacent_vertices O.start_vertex ∧ B ∈ O.adjacent_vertices A ∧ B ≠ O.start_vertex ∧ (∃ x ∈ O.middle_vertices, x = B)) →
    ant_moves_to_opposite_middle_vertex_prob = 1 / 2) := sorry

end probability_of_opposite_middle_vertex_l2427_242708


namespace tape_length_division_l2427_242707

theorem tape_length_division (n_pieces : ℕ) (length_piece overlap : ℝ) (n_parts : ℕ) 
  (h_pieces : n_pieces = 5) (h_length : length_piece = 2.7) (h_overlap : overlap = 0.3) 
  (h_parts : n_parts = 6) : 
  ((n_pieces * length_piece) - ((n_pieces - 1) * overlap)) / n_parts = 2.05 :=
  by
    sorry

end tape_length_division_l2427_242707


namespace original_price_of_car_l2427_242724

-- Let P be the original price of the car
variable (P : ℝ)

-- Condition: The car's value is reduced by 30%
-- Condition: The car's current value is $2800, which means 70% of the original price
def car_current_value_reduced (P : ℝ) : Prop :=
  0.70 * P = 2800

-- Theorem: Prove that the original price of the car is $4000
theorem original_price_of_car (P : ℝ) (h : car_current_value_reduced P) : P = 4000 := by
  sorry

end original_price_of_car_l2427_242724


namespace roots_quadratic_expression_l2427_242740

theorem roots_quadratic_expression (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0) 
  (sum_roots : m + n = -2) (product_roots : m * n = -5) : m^2 + m * n + 3 * m + n = -2 :=
sorry

end roots_quadratic_expression_l2427_242740


namespace problem_l2427_242793

theorem problem : 3^128 + 8^5 / 8^3 = 65 := sorry

end problem_l2427_242793


namespace solve_equation_l2427_242787

variable {x y : ℝ}

theorem solve_equation (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2: y ≠ 4) (h : (3 / x) + (2 / y) = 5 / 6) :
  x = 18 * y / (5 * y - 12) :=
sorry

end solve_equation_l2427_242787


namespace stewart_farm_horse_food_l2427_242743

def sheep_to_horse_ratio := 3 / 7
def horses_needed (sheep : ℕ) := (sheep * 7) / 3 
def daily_food_per_horse := 230
def sheep_count := 24
def total_horses := horses_needed sheep_count
def total_daily_horse_food := total_horses * daily_food_per_horse

theorem stewart_farm_horse_food : total_daily_horse_food = 12880 := by
  have num_horses : horses_needed 24 = 56 := by
    unfold horses_needed
    sorry -- Omitted for brevity, this would be solved

  have food_needed : 56 * 230 = 12880 := by
    sorry -- Omitted for brevity, this would be solved

  exact food_needed

end stewart_farm_horse_food_l2427_242743


namespace square_side_to_diagonal_ratio_l2427_242777

theorem square_side_to_diagonal_ratio (s : ℝ) : 
  s / (s * Real.sqrt 2) = Real.sqrt 2 / 2 :=
by
  sorry

end square_side_to_diagonal_ratio_l2427_242777


namespace fractional_eq_has_root_l2427_242727

theorem fractional_eq_has_root (x : ℝ) (m : ℝ) (h : x ≠ 4) :
    (3 / (x - 4) + (x + m) / (4 - x) = 1) → m = -1 :=
by
    intros h_eq
    sorry

end fractional_eq_has_root_l2427_242727


namespace total_buttons_needed_l2427_242701

def shirts_sewn_on_monday := 4
def shirts_sewn_on_tuesday := 3
def shirts_sewn_on_wednesday := 2
def buttons_per_shirt := 5

theorem total_buttons_needed : 
  (shirts_sewn_on_monday + shirts_sewn_on_tuesday + shirts_sewn_on_wednesday) * buttons_per_shirt = 45 :=
by 
  sorry

end total_buttons_needed_l2427_242701


namespace product_is_in_A_l2427_242779

def is_sum_of_squares (z : Int) : Prop :=
  ∃ t s : Int, z = t^2 + s^2

variable {x y : Int}

theorem product_is_in_A (hx : is_sum_of_squares x) (hy : is_sum_of_squares y) :
  is_sum_of_squares (x * y) :=
sorry

end product_is_in_A_l2427_242779


namespace unique_10_digit_number_property_l2427_242750

def ten_digit_number (N : ℕ) : Prop :=
  10^9 ≤ N ∧ N < 10^10

def first_digits_coincide (N : ℕ) : Prop :=
  ∀ M : ℕ, N^2 < 10^M → N^2 / 10^(M - 10) = N

theorem unique_10_digit_number_property :
  ∀ (N : ℕ), ten_digit_number N ∧ first_digits_coincide N → N = 1000000000 := 
by
  intros N hN
  sorry

end unique_10_digit_number_property_l2427_242750


namespace distance_from_center_to_line_l2427_242710

-- Define the circle and its center
def is_circle (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def center : (ℝ × ℝ) := (1, 0)

-- Define the line equation y = tan(30°) * x
def is_line (x y : ℝ) : Prop := y = (1 / Real.sqrt 3) * x

-- Function to compute the distance from a point to a line
noncomputable def distance_point_to_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  (abs (A * p.1 + B * p.2 + C)) / Real.sqrt (A^2 + B^2)

-- The main theorem to be proven:
theorem distance_from_center_to_line : 
  distance_point_to_line center (1 / Real.sqrt 3) (-1) 0 = 1 / 2 :=
  sorry

end distance_from_center_to_line_l2427_242710


namespace dinitrogen_monoxide_molecular_weight_l2427_242732

def atomic_weight_N : Real := 14.01
def atomic_weight_O : Real := 16.00

def chemical_formula_N2O_weight : Real :=
  (2 * atomic_weight_N) + (1 * atomic_weight_O)

theorem dinitrogen_monoxide_molecular_weight :
  chemical_formula_N2O_weight = 44.02 :=
by
  sorry

end dinitrogen_monoxide_molecular_weight_l2427_242732


namespace find_y_given_conditions_l2427_242715

theorem find_y_given_conditions (k : ℝ) (h1 : ∀ (x y : ℝ), xy = k) (h2 : ∀ (x y : ℝ), x + y = 30) (h3 : ∀ (x y : ℝ), x - y = 10) :
    ∀ x y, x = 8 → y = 25 :=
by
  sorry

end find_y_given_conditions_l2427_242715


namespace min_value_4x_3y_l2427_242771

theorem min_value_4x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y = 5 * x * y) : 
  4 * x + 3 * y ≥ 5 :=
sorry

end min_value_4x_3y_l2427_242771


namespace simplify_expression_l2427_242736

theorem simplify_expression (a b c : ℝ) (h : a + b + c = 3) : 
  (a ≠ 0) → (b ≠ 0) → (c ≠ 0) →
  (1 / (b^2 + c^2 - 3 * a^2) + 1 / (a^2 + c^2 - 3 * b^2) + 1 / (a^2 + b^2 - 3 * c^2) = -3) :=
by
  intros
  sorry

end simplify_expression_l2427_242736


namespace megan_popsicles_l2427_242752

variable (t_rate : ℕ) (t_hours : ℕ)

def popsicles_eaten (rate: ℕ) (hours: ℕ) : ℕ :=
  60 * hours / rate

theorem megan_popsicles : popsicles_eaten 20 5 = 15 := by
  sorry

end megan_popsicles_l2427_242752


namespace restaurant_meals_l2427_242794

theorem restaurant_meals (k a : ℕ) (ratio_kids_to_adults : k / a = 10 / 7) (kids_meals_sold : k = 70) : a = 49 :=
by
  sorry

end restaurant_meals_l2427_242794


namespace express_in_scientific_notation_l2427_242725

-- Definition for expressing number in scientific notation
def scientific_notation (n : ℝ) (a : ℝ) (b : ℕ) : Prop :=
  n = a * 10 ^ b

-- Condition of the problem
def condition : ℝ := 1300000

-- Stating the theorem to be proved
theorem express_in_scientific_notation : scientific_notation condition 1.3 6 :=
by
  -- Placeholder for the proof
  sorry

end express_in_scientific_notation_l2427_242725


namespace locus_of_midpoint_of_chord_l2427_242766

theorem locus_of_midpoint_of_chord 
  (A B C : ℝ) (h_arith_seq : A - 2 * B + C = 0) 
  (h_passing_through : ∀ t : ℝ,  t*A + -2*B + C = 0) :
  ∀ (x y : ℝ), 
    (Ax + By + C = 0) → 
    (h_on_parabola : y = -2 * x ^ 2) 
    → y + 1 = -(2 * x - 1) ^ 2 :=
sorry

end locus_of_midpoint_of_chord_l2427_242766


namespace sphere_radius_is_five_l2427_242763

theorem sphere_radius_is_five
    (π : ℝ)
    (r r_cylinder h : ℝ)
    (A_sphere A_cylinder : ℝ)
    (h1 : A_sphere = 4 * π * r ^ 2)
    (h2 : A_cylinder = 2 * π * r_cylinder * h)
    (h3 : h = 10)
    (h4 : r_cylinder = 5)
    (h5 : A_sphere = A_cylinder) :
    r = 5 :=
by
  sorry

end sphere_radius_is_five_l2427_242763


namespace new_ratio_alcohol_water_l2427_242747

theorem new_ratio_alcohol_water (alcohol water: ℕ) (initial_ratio: alcohol * 3 = water * 4) 
  (extra_water: ℕ) (extra_water_added: extra_water = 4) (alcohol_given: alcohol = 20):
  20 * 19 = alcohol * (water + extra_water) :=
by
  sorry

end new_ratio_alcohol_water_l2427_242747


namespace age_ratio_l2427_242723

theorem age_ratio (R D : ℕ) (h1 : R + 2 = 26) (h2 : D = 18) : R / D = 4 / 3 :=
sorry

end age_ratio_l2427_242723


namespace skillful_hands_award_prob_cannot_enter_finals_after_training_l2427_242735

noncomputable def combinatorial_probability : ℚ :=
  let P1 := (4 * 3) / (10 * 10)    -- P1: 1 specified, 2 creative
  let P2 := (6 * 3) / (10 * 10)    -- P2: 2 specified, 1 creative
  let P3 := (6 * 3) / (10 * 10)    -- P3: 2 specified, 2 creative
  P1 + P2 + P3

theorem skillful_hands_award_prob : combinatorial_probability = 33 / 50 := 
  sorry

def after_training_probability := 3 / 4
theorem cannot_enter_finals_after_training : after_training_probability * 5 < 4 := 
  sorry

end skillful_hands_award_prob_cannot_enter_finals_after_training_l2427_242735


namespace simplify_and_evaluate_l2427_242741

/-- 
Given the expression (1 + 1 / (x - 2)) ÷ ((x ^ 2 - 2 * x + 1) / (x - 2)), 
prove that it evaluates to -1 when x = 0.
-/
theorem simplify_and_evaluate (x : ℝ) (h : x = 0) :
  (1 + 1 / (x - 2)) / ((x^2 - 2 * x + 1) / (x - 2)) = -1 :=
by
  sorry

end simplify_and_evaluate_l2427_242741


namespace checkerboard_black_squares_l2427_242728

theorem checkerboard_black_squares (n : ℕ) (hn : n = 33) :
  let black_squares : ℕ := (n * n + 1) / 2
  black_squares = 545 :=
by
  sorry

end checkerboard_black_squares_l2427_242728


namespace cone_volume_from_half_sector_l2427_242786

theorem cone_volume_from_half_sector (r l : ℝ) (h : ℝ) 
    (h_r : r = 3) (h_l : l = 6) (h_h : h = 3 * Real.sqrt 3) : 
    (1 / 3) * Real.pi * r^2 * h = 9 * Real.pi * Real.sqrt 3 := 
by
  -- Sorry to skip the proof
  sorry

end cone_volume_from_half_sector_l2427_242786


namespace max_value_char_l2427_242709

theorem max_value_char (m x a b : ℕ) (h_sum : 28 * m + x + a + 2 * b = 368)
  (h1 : x ≤ 23) (h2 : x > a) (h3 : a > b) (h4 : b ≥ 0) :
  m + x ≤ 35 := 
sorry

end max_value_char_l2427_242709


namespace cone_volume_l2427_242759

theorem cone_volume (l : ℝ) (circumference : ℝ) (radius : ℝ) (height : ℝ) (volume : ℝ) 
  (h1 : l = 8) 
  (h2 : circumference = 6 * Real.pi) 
  (h3 : radius = circumference / (2 * Real.pi))
  (h4 : height = Real.sqrt (l^2 - radius^2)) 
  (h5 : volume = (1 / 3) * Real.pi * radius^2 * height) :
  volume = 3 * Real.sqrt 55 * Real.pi := 
  by 
    sorry

end cone_volume_l2427_242759


namespace distance_focus_directrix_parabola_l2427_242748

theorem distance_focus_directrix_parabola (p : ℝ) (h : y^2 = 20 * x) : 
  2 * p = 10 :=
by
  -- h represents the given condition y^2 = 20x.
  sorry

end distance_focus_directrix_parabola_l2427_242748


namespace total_length_segments_in_figure2_l2427_242711

-- Define the original dimensions of the figure
def vertical_side : ℕ := 10
def bottom_horizontal_side : ℕ := 3
def middle_horizontal_side : ℕ := 4
def topmost_horizontal_side : ℕ := 2

-- Define the lengths that are removed to form Figure 2
def removed_sides_length : ℕ :=
  bottom_horizontal_side + topmost_horizontal_side + vertical_side

-- Define the remaining lengths in Figure 2
def remaining_vertical_side : ℕ := vertical_side
def remaining_horizontal_side : ℕ := middle_horizontal_side

-- Total length of segments in Figure 2
def total_length_figure2 : ℕ :=
  remaining_vertical_side + remaining_horizontal_side

-- Conjecture that this total length is 14 units
theorem total_length_segments_in_figure2 : total_length_figure2 = 14 := by
  -- Proof goes here
  sorry

end total_length_segments_in_figure2_l2427_242711


namespace average_headcount_11600_l2427_242739

theorem average_headcount_11600 : 
  let h02_03 := 11700
  let h03_04 := 11500
  let h04_05 := 11600
  (h02_03 + h03_04 + h04_05) / 3 = 11600 := 
by
  sorry

end average_headcount_11600_l2427_242739


namespace greatest_integer_value_l2427_242797

theorem greatest_integer_value (x : ℤ) : ∃ x, (∀ y, (x^2 + 2 * x + 10) % (x - 3) = 0 → x ≥ y) → x = 28 :=
by
  sorry

end greatest_integer_value_l2427_242797


namespace sugar_for_cake_l2427_242761

-- Definitions of given values
def sugar_for_frosting : ℝ := 0.6
def total_sugar_required : ℝ := 0.8

-- Proof statement
theorem sugar_for_cake : (total_sugar_required - sugar_for_frosting) = 0.2 :=
by
  sorry

end sugar_for_cake_l2427_242761


namespace determine_F_value_l2427_242713

theorem determine_F_value (D E F : ℕ) (h1 : (9 + 6 + D + 1 + E + 8 + 2) % 3 = 0) (h2 : (5 + 4 + E + D + 2 + 1 + F) % 3 = 0) : 
  F = 2 := 
by
  sorry

end determine_F_value_l2427_242713


namespace can_form_all_numbers_l2427_242730

noncomputable def domino_tiles : List (ℕ × ℕ) := [(1, 3), (6, 6), (6, 2), (3, 2)]

def form_any_number (n : ℕ) : Prop :=
  ∃ (comb : List (ℕ × ℕ)), comb ⊆ domino_tiles ∧ (comb.bind (λ p => [p.1, p.2])).sum = n

theorem can_form_all_numbers : ∀ n, 1 ≤ n → n ≤ 23 → form_any_number n :=
by sorry

end can_form_all_numbers_l2427_242730


namespace leak_emptying_time_l2427_242717

theorem leak_emptying_time (A_rate L_rate : ℚ) 
  (hA : A_rate = 1 / 4)
  (hCombined : A_rate - L_rate = 1 / 8) :
  1 / L_rate = 8 := 
by
  sorry

end leak_emptying_time_l2427_242717


namespace equation_represents_single_point_l2427_242784

theorem equation_represents_single_point (d : ℝ) :
  (∀ x y : ℝ, 3*x^2 + 4*y^2 + 6*x - 8*y + d = 0 ↔ (x = -1 ∧ y = 1)) → d = 7 :=
sorry

end equation_represents_single_point_l2427_242784


namespace range_of_m_l2427_242705

def M := {y : ℝ | ∃ (x : ℝ), y = (1/2)^x}
def N (m : ℝ) := {y : ℝ | ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ y = ((1/(m-1) + 1) * (x - 1) + (|m| - 1) * (x - 2))}

theorem range_of_m (m : ℝ) : (∀ y ∈ N m, y ∈ M) ↔ -1 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l2427_242705


namespace nonagon_side_length_l2427_242768

theorem nonagon_side_length (perimeter : ℝ) (n : ℕ) (h_reg_nonagon : n = 9) (h_perimeter : perimeter = 171) :
  perimeter / n = 19 := by
  sorry

end nonagon_side_length_l2427_242768


namespace sufficient_condition_for_inequality_l2427_242764

theorem sufficient_condition_for_inequality (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by
  sorry

end sufficient_condition_for_inequality_l2427_242764


namespace problem_l2427_242765

def op (x y : ℝ) : ℝ := x^2 - y

theorem problem (h : ℝ) : op h (op h h) = h :=
by
  sorry

end problem_l2427_242765


namespace ratio_x_y_l2427_242738

-- Definitions based on conditions
variables (a b c x y : ℝ) 

-- Conditions
def right_triangle (a b c : ℝ) := (a^2 + b^2 = c^2)
def a_b_ratio (a b : ℝ) := (a / b = 2 / 5)
def segments_ratio (a b c x y : ℝ) := (x = a^2 / c) ∧ (y = b^2 / c)
def perpendicular_division (x y a b : ℝ) := ((a^2 / x) = c) ∧ ((b^2 / y) = c)

-- The proof statement we need
theorem ratio_x_y : 
  ∀ (a b c x y : ℝ),
    right_triangle a b c → 
    a_b_ratio a b → 
    segments_ratio a b c x y → 
    (x / y = 4 / 25) :=
by sorry

end ratio_x_y_l2427_242738


namespace sum_of_x_y_l2427_242780

theorem sum_of_x_y :
  ∀ (x y : ℚ), (1 / x + 1 / y = 4) → (1 / x - 1 / y = -8) → x + y = -1 / 3 := 
by
  intros x y h1 h2
  sorry

end sum_of_x_y_l2427_242780


namespace tamia_total_slices_and_pieces_l2427_242714

-- Define the conditions
def num_bell_peppers : ℕ := 5
def slices_per_pepper : ℕ := 20
def num_large_slices : ℕ := num_bell_peppers * slices_per_pepper
def num_half_slices : ℕ := num_large_slices / 2
def small_pieces_per_slice : ℕ := 3
def num_small_pieces : ℕ := num_half_slices * small_pieces_per_slice
def num_uncut_slices : ℕ := num_half_slices

-- Define the total number of pieces and slices
def total_pieces_and_slices : ℕ := num_uncut_slices + num_small_pieces

-- State the theorem and provide a placeholder for the proof
theorem tamia_total_slices_and_pieces : total_pieces_and_slices = 200 :=
by
  sorry

end tamia_total_slices_and_pieces_l2427_242714


namespace opposite_of_negative_five_l2427_242772

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l2427_242772


namespace minimum_purchase_price_mod6_l2427_242756

theorem minimum_purchase_price_mod6 
  (coin_values : List ℕ)
  (h1 : (1 : ℕ) ∈ coin_values)
  (h15 : (15 : ℕ) ∈ coin_values)
  (h50 : (50 : ℕ) ∈ coin_values)
  (A C : ℕ)
  (k : ℕ)
  (hA : A ≡ k [MOD 7])
  (hC : C ≡ k + 1 [MOD 7])
  (hP : ∃ P, P = A - C) : 
  ∃ P, P ≡ 6 [MOD 7] ∧ P > 0 :=
by
  sorry

end minimum_purchase_price_mod6_l2427_242756


namespace find_principal_l2427_242781

variable (P R : ℝ)
variable (condition1 : P + (P * R * 2) / 100 = 660)
variable (condition2 : P + (P * R * 7) / 100 = 1020)

theorem find_principal : P = 516 := by
  sorry

end find_principal_l2427_242781


namespace trig_identity_l2427_242726

open Real

theorem trig_identity :
  (1 - 1 / cos (23 * π / 180)) *
  (1 + 1 / sin (67 * π / 180)) *
  (1 - 1 / sin (23 * π / 180)) * 
  (1 + 1 / cos (67 * π / 180)) = 1 :=
by
  sorry

end trig_identity_l2427_242726


namespace problem_a_problem_b_l2427_242721

noncomputable def gini_coefficient_separate_operations : ℝ := 
  let population_north := 24
  let population_south := population_north / 4
  let income_per_north_inhabitant := (6000 * 18) / population_north
  let income_per_south_inhabitant := (6000 * 12) / population_south
  let total_population := population_north + population_south
  let total_income := 6000 * (18 + 12)
  let share_pop_north := population_north / total_population
  let share_income_north := (income_per_north_inhabitant * population_north) / total_income
  share_pop_north - share_income_north

theorem problem_a : gini_coefficient_separate_operations = 0.2 := 
  by sorry

noncomputable def change_in_gini_coefficient_after_collaboration : ℝ :=
  let previous_income_north := 6000 * 18
  let compensation := previous_income_north + 1983
  let total_combined_income := 6000 * 30.5
  let remaining_income_south := total_combined_income - compensation
  let population := 24 + 6
  let income_per_capita_north := compensation / 24
  let income_per_capita_south := remaining_income_south / 6
  let new_gini_coefficient := 
    let share_pop_north := 24 / population
    let share_income_north := compensation / total_combined_income
    share_pop_north - share_income_north
  (0.2 - new_gini_coefficient)

theorem problem_b : change_in_gini_coefficient_after_collaboration = 0.001 := 
  by sorry

end problem_a_problem_b_l2427_242721


namespace sum_of_quarter_circle_arcs_l2427_242742

-- Define the main variables and problem statement.
variable (D : ℝ) -- Diameter of the original circle.
variable (n : ℕ) (hn : 0 < n) -- Number of parts (positive integer).

-- Define a theorem stating that the sum of quarter-circle arcs is greater than D, but less than (pi D / 2) as n tends to infinity.
theorem sum_of_quarter_circle_arcs (hn : 0 < n) :
  D < (π * D) / 4 ∧ (π * D) / 4 < (π * D) / 2 :=
by
  sorry -- Proof of the theorem goes here.

end sum_of_quarter_circle_arcs_l2427_242742
