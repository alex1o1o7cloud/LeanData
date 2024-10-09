import Mathlib

namespace machine_p_vs_machine_q_l256_25699

variable (MachineA_rate MachineQ_rate MachineP_rate : ℝ)
variable (Total_sprockets : ℝ := 550)
variable (Production_rate_A : ℝ := 5)
variable (Production_rate_Q : ℝ := MachineA_rate + 0.1 * MachineA_rate)
variable (Time_Q : ℝ := Total_sprockets / Production_rate_Q)
variable (Time_P : ℝ)
variable (Difference : ℝ)

noncomputable def production_times_difference (MachineA_rate MachineQ_rate MachineP_rate : ℝ) : ℝ :=
  let Production_rate_Q := MachineA_rate + 0.1 * MachineA_rate
  let Time_Q := Total_sprockets / Production_rate_Q
  let Difference := Time_P - Time_Q
  Difference

theorem machine_p_vs_machine_q : 
  Production_rate_A = 5 → 
  Total_sprockets = 550 →
  Production_rate_Q = 5.5 →
  Time_Q = 100 →
  MachineP_rate = MachineP_rate →
  Time_P = Time_P →
  Difference = (Time_P - Time_Q) :=
by
  intros
  sorry

end machine_p_vs_machine_q_l256_25699


namespace arctan_sum_l256_25655

theorem arctan_sum (a b : ℝ) (h1 : a = 2 / 3) (h2 : (a + 1) * (b + 1) = 8 / 3) :
  Real.arctan a + Real.arctan b = Real.arctan (19 / 9) := by
  sorry

end arctan_sum_l256_25655


namespace calculate_stripes_l256_25613

theorem calculate_stripes :
  let olga_stripes_per_shoe := 3
  let rick_stripes_per_shoe := olga_stripes_per_shoe - 1
  let hortense_stripes_per_shoe := olga_stripes_per_shoe * 2
  let ethan_stripes_per_shoe := hortense_stripes_per_shoe + 2
  (olga_stripes_per_shoe * 2 + rick_stripes_per_shoe * 2 + hortense_stripes_per_shoe * 2 + ethan_stripes_per_shoe * 2) / 2 = 19 := 
by
  sorry

end calculate_stripes_l256_25613


namespace measure_of_angle_C_l256_25678

theorem measure_of_angle_C (m l : ℝ) (angle_A angle_B angle_D angle_C : ℝ)
  (h_parallel : l = m)
  (h_angle_A : angle_A = 130)
  (h_angle_B : angle_B = 140)
  (h_angle_D : angle_D = 100) :
  angle_C = 90 :=
by
  sorry

end measure_of_angle_C_l256_25678


namespace ways_to_insert_plus_l256_25600

-- Definition of the problem conditions
def num_ones : ℕ := 15
def target_sum : ℕ := 0 

-- Binomial coefficient calculation
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to be proven
theorem ways_to_insert_plus :
  binomial 14 9 = 2002 :=
by
  sorry

end ways_to_insert_plus_l256_25600


namespace cost_of_each_muffin_l256_25615

-- Define the cost of juice
def juice_cost : ℝ := 1.45

-- Define the total cost paid by Kevin
def total_cost : ℝ := 3.70

-- Assume the cost of each muffin
def muffin_cost (M : ℝ) : Prop := 
  3 * M + juice_cost = total_cost

-- The theorem we aim to prove
theorem cost_of_each_muffin : muffin_cost 0.75 :=
by
  -- Here the proof would go
  sorry

end cost_of_each_muffin_l256_25615


namespace volume_of_inscribed_tetrahedron_l256_25607

theorem volume_of_inscribed_tetrahedron (r h : ℝ) (V : ℝ) (tetrahedron_inscribed : Prop) 
  (cylinder_condition : π * r^2 * h = 1) 
  (inscribed : tetrahedron_inscribed → True) : 
  V ≤ 2 / (3 * π) :=
sorry

end volume_of_inscribed_tetrahedron_l256_25607


namespace ladder_alley_width_l256_25674

theorem ladder_alley_width (l : ℝ) (m : ℝ) (w : ℝ) (h : m = l / 2) :
  w = (l * (Real.sqrt 3 + 1)) / 2 :=
by
  sorry

end ladder_alley_width_l256_25674


namespace tina_wins_more_than_losses_l256_25690

theorem tina_wins_more_than_losses 
  (initial_wins : ℕ)
  (additional_wins : ℕ)
  (first_loss : ℕ)
  (doubled_wins : ℕ)
  (second_loss : ℕ)
  (total_wins : ℕ)
  (total_losses : ℕ)
  (final_difference : ℕ) :
  initial_wins = 10 →
  additional_wins = 5 →
  first_loss = 1 →
  doubled_wins = 30 →
  second_loss = 1 →
  total_wins = initial_wins + additional_wins + doubled_wins →
  total_losses = first_loss + second_loss →
  final_difference = total_wins - total_losses →
  final_difference = 43 :=
by
  sorry

end tina_wins_more_than_losses_l256_25690


namespace frog_jump_distance_l256_25695

theorem frog_jump_distance (grasshopper_jump : ℕ) (extra_jump : ℕ) (frog_jump : ℕ) :
  grasshopper_jump = 9 → extra_jump = 3 → frog_jump = grasshopper_jump + extra_jump → frog_jump = 12 :=
by
  intros h_grasshopper h_extra h_frog
  rw [h_grasshopper, h_extra] at h_frog
  exact h_frog

end frog_jump_distance_l256_25695


namespace factorization_example_l256_25641

theorem factorization_example (C D : ℤ) (h : 20 * y^2 - 122 * y + 72 = (C * y - 8) * (D * y - 9)) : C * D + C = 25 := by
  sorry

end factorization_example_l256_25641


namespace solve_equation_l256_25676

theorem solve_equation (x : ℝ) : (2*x - 1)^2 = 81 ↔ (x = 5 ∨ x = -4) :=
by
  sorry

end solve_equation_l256_25676


namespace gardening_project_cost_l256_25620

def cost_rose_bushes (number_of_bushes: ℕ) (cost_per_bush: ℕ) : ℕ := number_of_bushes * cost_per_bush
def cost_gardener (hourly_rate: ℕ) (hours_per_day: ℕ) (days: ℕ) : ℕ := hourly_rate * hours_per_day * days
def cost_soil (cubic_feet: ℕ) (cost_per_cubic_foot: ℕ) : ℕ := cubic_feet * cost_per_cubic_foot

theorem gardening_project_cost :
  cost_rose_bushes 20 150 + cost_gardener 30 5 4 + cost_soil 100 5 = 4100 :=
by
  sorry

end gardening_project_cost_l256_25620


namespace crayons_difference_l256_25603

def initial_crayons : ℕ := 8597
def crayons_given : ℕ := 7255
def crayons_lost : ℕ := 3689

theorem crayons_difference : crayons_given - crayons_lost = 3566 := by
  sorry

end crayons_difference_l256_25603


namespace price_increase_percentage_l256_25638

-- Define the problem conditions
def lowest_price := 12
def highest_price := 21

-- Formulate the goal as a theorem
theorem price_increase_percentage :
  ((highest_price - lowest_price) / lowest_price : ℚ) * 100 = 75 := by
  sorry

end price_increase_percentage_l256_25638


namespace monotonicity_f_parity_f_max_value_f_min_value_f_l256_25677

noncomputable def f (x : ℝ) : ℝ := x / (x^2 - 4)

-- Monotonicity Proof
theorem monotonicity_f : ∀ {x1 x2 : ℝ}, 2 < x1 → 2 < x2 → x1 < x2 → f x1 > f x2 :=
sorry

-- Parity Proof
theorem parity_f : ∀ x : ℝ, f (-x) = -f x :=
sorry

-- Maximum Value Proof
theorem max_value_f : ∀ {x : ℝ}, x = -6 → f x = -3/16 :=
sorry

-- Minimum Value Proof
theorem min_value_f : ∀ {x : ℝ}, x = -3 → f x = -3/5 :=
sorry

end monotonicity_f_parity_f_max_value_f_min_value_f_l256_25677


namespace find_side_length_of_square_l256_25640

theorem find_side_length_of_square (n k : ℕ) (hk : k ≥ 1) (h : (n + k) * (n + k) - n * n = 47) : n = 23 :=
  sorry

end find_side_length_of_square_l256_25640


namespace total_number_of_cows_l256_25623

theorem total_number_of_cows (n : ℕ) 
  (h1 : n > 0) 
  (h2 : (1/3) * n + (1/6) * n + (1/8) * n + 9 = n) : n = 216 :=
sorry

end total_number_of_cows_l256_25623


namespace lollipop_problem_l256_25672

def arithmetic_sequence_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem lollipop_problem
  (a : ℕ) (h1 : arithmetic_sequence_sum a 5 7 = 175) :
  (a + 15) = 25 :=
by
  sorry

end lollipop_problem_l256_25672


namespace find_value_of_expression_l256_25673

theorem find_value_of_expression
  (k m : ℕ)
  (hk : 3^(k - 1) = 9)
  (hm : 4^(m + 2) = 64) :
  2^(3*k + 2*m) = 2^11 :=
by 
  sorry

end find_value_of_expression_l256_25673


namespace neither_outstanding_nor_young_pioneers_is_15_l256_25660

-- Define the conditions
def total_students : ℕ := 87
def outstanding_students : ℕ := 58
def young_pioneers : ℕ := 63
def both_outstanding_and_young_pioneers : ℕ := 49

-- Define the function to calculate the number of students who are neither
def neither_outstanding_nor_young_pioneers
: ℕ :=
total_students - (outstanding_students - both_outstanding_and_young_pioneers) - (young_pioneers - both_outstanding_and_young_pioneers) - both_outstanding_and_young_pioneers

-- The theorem to prove
theorem neither_outstanding_nor_young_pioneers_is_15
: neither_outstanding_nor_young_pioneers = 15 :=
by
  sorry

end neither_outstanding_nor_young_pioneers_is_15_l256_25660


namespace fraction_addition_l256_25649

theorem fraction_addition (a b : ℕ) (hb : b ≠ 0) (h : a / (b : ℚ) = 3 / 5) : (a + b) / (b : ℚ) = 8 / 5 := 
by
sorry

end fraction_addition_l256_25649


namespace range_of_a_l256_25621

noncomputable def p (x : ℝ) : Prop := abs (3 * x - 4) > 2
noncomputable def q (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0
noncomputable def r (x a : ℝ) : Prop := (x - a) * (x - a - 1) < 0

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, ¬ r x a → ¬ p x) → (a ≥ 2 ∨ a ≤ -1/3) :=
sorry

end range_of_a_l256_25621


namespace gcd_of_sum_and_product_l256_25644

theorem gcd_of_sum_and_product (x y : ℕ) (h1 : x + y = 1130) (h2 : x * y = 100000) : Int.gcd x y = 2 := 
sorry

end gcd_of_sum_and_product_l256_25644


namespace slope_y_intercept_product_l256_25671

theorem slope_y_intercept_product (m b : ℝ) (hm : m = -1/2) (hb : b = 4/5) : -1 < m * b ∧ m * b < 0 :=
by
  sorry

end slope_y_intercept_product_l256_25671


namespace circle_center_radius_l256_25618

theorem circle_center_radius (x y : ℝ) :
  x^2 - 6*x + y^2 + 2*y - 9 = 0 ↔ (x-3)^2 + (y+1)^2 = 19 :=
sorry

end circle_center_radius_l256_25618


namespace water_depth_is_12_feet_l256_25681

variable (Ron_height Dean_height Water_depth : ℕ)

-- Given conditions
axiom H1 : Ron_height = 14
axiom H2 : Dean_height = Ron_height - 8
axiom H3 : Water_depth = 2 * Dean_height

-- Prove that the water depth is 12 feet
theorem water_depth_is_12_feet : Water_depth = 12 :=
by
  sorry

end water_depth_is_12_feet_l256_25681


namespace lemmings_distance_average_l256_25650

noncomputable def diagonal_length (side: ℝ) : ℝ :=
  Real.sqrt (side^2 + side^2)

noncomputable def fraction_traveled (side: ℝ) (distance: ℝ) : ℝ :=
  distance / (Real.sqrt 2 * side)

noncomputable def final_coordinates (side: ℝ) (distance1: ℝ) (angle: ℝ) (distance2: ℝ) : (ℝ × ℝ) :=
  let frac := fraction_traveled side distance1
  let initial_pos := (frac * side, frac * side)
  let move_dist := distance2 * (Real.sqrt 2 / 2)
  (initial_pos.1 + move_dist, initial_pos.2 + move_dist)

noncomputable def average_shortest_distances (side: ℝ) (coords: ℝ × ℝ) : ℝ :=
  let x_dist := min coords.1 (side - coords.1)
  let y_dist := min coords.2 (side - coords.2)
  (x_dist + (side - x_dist) + y_dist + (side - y_dist)) / 4

theorem lemmings_distance_average :
  let side := 15
  let distance1 := 9.3
  let angle := 45 / 180 * Real.pi -- convert to radians
  let distance2 := 3
  let coords := final_coordinates side distance1 angle distance2
  average_shortest_distances side coords = 7.5 :=
by
  sorry

end lemmings_distance_average_l256_25650


namespace sphere_surface_area_l256_25686

theorem sphere_surface_area (V : ℝ) (h : V = 72 * Real.pi) : ∃ A, A = 36 * Real.pi * (2 ^ (2 / 3)) := 
by
  sorry

end sphere_surface_area_l256_25686


namespace diagonal_length_of_regular_hexagon_l256_25634

theorem diagonal_length_of_regular_hexagon (
  side_length : ℝ
) (h_side_length : side_length = 12) : 
  ∃ DA, DA = 12 * Real.sqrt 3 :=
by 
  sorry

end diagonal_length_of_regular_hexagon_l256_25634


namespace solution_to_problem_l256_25691

-- Definitions of conditions
def condition_1 (x : ℝ) : Prop := 2 * x - 6 ≠ 0
def condition_2 (x : ℝ) : Prop := 5 ≤ x / (2 * x - 6) ∧ x / (2 * x - 6) < 10

-- Definition of solution set
def solution_set (x : ℝ) : Prop := 3 < x ∧ x < 60 / 19

-- The theorem to be proven
theorem solution_to_problem (x : ℝ) (h1 : condition_1 x) : condition_2 x ↔ solution_set x :=
by sorry

end solution_to_problem_l256_25691


namespace correct_mark_l256_25608

theorem correct_mark
  (n : ℕ)
  (initial_avg : ℝ)
  (wrong_mark : ℝ)
  (correct_avg : ℝ)
  (correct_total_marks : ℝ)
  (actual_total_marks : ℝ)
  (final_mark : ℝ) :
  n = 25 →
  initial_avg = 100 →
  wrong_mark = 60 →
  correct_avg = 98 →
  correct_total_marks = (n * correct_avg) →
  actual_total_marks = (n * initial_avg - wrong_mark + final_mark) →
  correct_total_marks = actual_total_marks →
  final_mark = 10 :=
by
  intros h_n h_initial_avg h_wrong_mark h_correct_avg h_correct_total_marks h_actual_total_marks h_eq
  sorry

end correct_mark_l256_25608


namespace average_probable_weight_l256_25604

-- Define the conditions
def Arun_opinion (w : ℝ) : Prop := 64 < w ∧ w < 72
def Brother_opinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def Mother_opinion (w : ℝ) : Prop := w ≤ 67

-- The proof problem statement
theorem average_probable_weight :
  ∃ (w : ℝ), Arun_opinion w ∧ Brother_opinion w ∧ Mother_opinion w →
  (64 + 67) / 2 = 65.5 :=
by
  sorry

end average_probable_weight_l256_25604


namespace different_movies_count_l256_25664

theorem different_movies_count 
    (d_movies : ℕ) (h_movies : ℕ) (a_movies : ℕ) (b_movies : ℕ) (c_movies : ℕ) 
    (together_movies : ℕ) (dha_movies : ℕ) (bc_movies : ℕ) 
    (db_movies : ℕ) (ac_movies : ℕ)
    (H_d : d_movies = 20) (H_h : h_movies = 26) (H_a : a_movies = 35) 
    (H_b : b_movies = 29) (H_c : c_movies = 16)
    (H_together : together_movies = 5)
    (H_dha : dha_movies = 4) (H_bc : bc_movies = 3) 
    (H_db : db_movies = 2) (H_ac : ac_movies = 4) :
    d_movies + h_movies + a_movies + b_movies + c_movies 
    - 4 * together_movies - 3 * dha_movies - 2 * bc_movies - db_movies - 3 * ac_movies = 74 := by sorry

end different_movies_count_l256_25664


namespace two_sectors_area_l256_25645

theorem two_sectors_area {r : ℝ} {θ : ℝ} (h_radius : r = 15) (h_angle : θ = 45) : 
  2 * (θ / 360) * (π * r^2) = 56.25 * π := 
by
  rw [h_radius, h_angle]
  norm_num
  sorry

end two_sectors_area_l256_25645


namespace initial_men_is_250_l256_25661

-- Define the given conditions
def provisions (initial_men remaining_men initial_days remaining_days : ℕ) : Prop :=
  initial_men * initial_days = remaining_men * remaining_days

-- Define the problem statement
theorem initial_men_is_250 (initial_days remaining_days : ℕ) (remaining_men_leaving : ℕ) :
  provisions initial_men (initial_men - remaining_men_leaving) initial_days remaining_days → initial_men = 250 :=
by
  intros h
  -- Requirement to solve the theorem.
  -- This is where the proof steps would go, but we put sorry to satisfy the statement requirement.
  sorry

end initial_men_is_250_l256_25661


namespace profit_percentage_is_ten_l256_25617

-- Define the cost price (CP) and selling price (SP) as constants
def CP : ℝ := 90.91
def SP : ℝ := 100

-- Define a theorem to prove the profit percentage is 10%
theorem profit_percentage_is_ten : ((SP - CP) / CP) * 100 = 10 := 
by 
  -- Skip the proof.
  sorry

end profit_percentage_is_ten_l256_25617


namespace solution_l256_25684

noncomputable def triangle_perimeter (AB BC AC : ℕ) (lA lB lC : ℕ) : ℕ :=
  -- This represents the proof problem using the given conditions
  if (AB = 130) ∧ (BC = 240) ∧ (AC = 190)
     ∧ (lA = 65) ∧ (lB = 50) ∧ (lC = 20)
  then
    130  -- The correct answer
  else
    0    -- If the conditions are not met, return 0 

theorem solution :
  triangle_perimeter 130 240 190 65 50 20 = 130 :=
by
  -- This theorem states that with the given conditions, the perimeter of the triangle is 130
  sorry

end solution_l256_25684


namespace solve_for_x_l256_25606

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 := 
by
  sorry

end solve_for_x_l256_25606


namespace min_value_func_y_l256_25601

noncomputable def geometric_sum (t : ℝ) (n : ℕ) : ℝ :=
  t * 3^(n-1) - (1 / 3)

noncomputable def func_y (x t : ℝ) : ℝ :=
  (x + 2) * (x + 10) / (x + t)

theorem min_value_func_y :
  ∀ (t : ℝ), (∀ n : ℕ, geometric_sum t n = (1) → (∀ x > 0, func_y x t ≥ 16)) :=
  sorry

end min_value_func_y_l256_25601


namespace maria_money_left_l256_25667

def ticket_cost : ℕ := 300
def hotel_cost : ℕ := ticket_cost / 2
def transportation_cost : ℕ := 80
def num_days : ℕ := 5
def avg_meal_cost_per_day : ℕ := 40
def tourist_tax_rate : ℚ := 0.10
def starting_amount : ℕ := 760

def total_meal_cost : ℕ := num_days * avg_meal_cost_per_day
def expenses_subject_to_tax := hotel_cost + transportation_cost
def tourist_tax := tourist_tax_rate * expenses_subject_to_tax
def total_expenses := ticket_cost + hotel_cost + transportation_cost + total_meal_cost + tourist_tax
def money_left := starting_amount - total_expenses

theorem maria_money_left : money_left = 7 := by
  sorry

end maria_money_left_l256_25667


namespace day_crew_fraction_loaded_l256_25658

-- Let D be the number of boxes loaded by each worker on the day crew
-- Let W_d be the number of workers on the day crew
-- Let W_n be the number of workers on the night crew
-- Let B_d be the total number of boxes loaded by the day crew
-- Let B_n be the total number of boxes loaded by the night crew

variable (D W_d : ℕ) 
variable (B_d := D * W_d)
variable (W_n := (4 / 9 : ℚ) * W_d)
variable (B_n := (3 / 4 : ℚ) * D * W_n)
variable (total_boxes := B_d + B_n)

theorem day_crew_fraction_loaded : 
  (D * W_d) / (D * W_d + (3 / 4 : ℚ) * D * ((4 / 9 : ℚ) * W_d)) = (3 / 4 : ℚ) := sorry

end day_crew_fraction_loaded_l256_25658


namespace proof_problem_l256_25659

def sum_even_ints (n : ℕ) : ℕ := n * (n + 1)
def sum_odd_ints (n : ℕ) : ℕ := n^2
def sum_specific_primes : ℕ := [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97].sum

theorem proof_problem : (sum_even_ints 100 - sum_odd_ints 100) + sum_specific_primes = 1063 :=
by
  sorry

end proof_problem_l256_25659


namespace percentage_difference_l256_25616

theorem percentage_difference :
    let A := (40 / 100) * ((50 / 100) * 60)
    let B := (50 / 100) * ((60 / 100) * 70)
    (B - A) = 9 :=
by
    sorry

end percentage_difference_l256_25616


namespace rhombus_longer_diagonal_l256_25692

theorem rhombus_longer_diagonal (a b d_1 : ℝ) (h_side : a = 60) (h_d1 : d_1 = 56) :
  ∃ d_2, d_2 = 106 := by
  sorry

end rhombus_longer_diagonal_l256_25692


namespace min_dot_product_of_vectors_at_fixed_point_l256_25652

noncomputable def point := ℝ × ℝ

def on_ellipse (x y : ℝ) : Prop := 
  (x^2) / 36 + (y^2) / 9 = 1

def dot_product (p q : point) : ℝ := 
  p.1 * q.1 + p.2 * q.2

def vector_magnitude_squared (p : point) : ℝ := 
  p.1^2 + p.2^2

def KM (M : point) : point := 
  (M.1 - 2, M.2)

def NM (N M : point) : point := 
  (M.1 - N.1, M.2 - N.2)

def fixed_point_K : point := 
  (2, 0)

theorem min_dot_product_of_vectors_at_fixed_point (M N : point) 
  (hM_on_ellipse : on_ellipse M.1 M.2) 
  (hN_on_ellipse : on_ellipse N.1 N.2) 
  (h_orthogonal : dot_product (KM M) (KM N) = 0) : 
  ∃ (α : ℝ), dot_product (KM M) (NM N M) = 23 / 3 :=
sorry

end min_dot_product_of_vectors_at_fixed_point_l256_25652


namespace misha_darts_score_l256_25668

theorem misha_darts_score (x : ℕ) 
  (h1 : x >= 24)
  (h2 : x * 3 <= 72) : 
  2 * x = 48 :=
by
  sorry

end misha_darts_score_l256_25668


namespace find_number_l256_25625

noncomputable def least_common_multiple (a b : ℕ) : ℕ := Nat.lcm a b

theorem find_number (n : ℕ) (h1 : least_common_multiple (least_common_multiple n 16) (least_common_multiple 18 24) = 144) : n = 9 :=
sorry

end find_number_l256_25625


namespace initial_tree_height_l256_25662

-- Definition of the problem conditions as Lean definitions.
def quadruple (x : ℕ) : ℕ := 4 * x

-- Given conditions of the problem
def final_height : ℕ := 256
def height_increase_each_year (initial_height : ℕ) : Prop :=
  quadruple (quadruple (quadruple (quadruple initial_height))) = final_height

-- The proof statement that we need to prove
theorem initial_tree_height 
  (initial_height : ℕ)
  (h : height_increase_each_year initial_height)
  : initial_height = 1 := sorry

end initial_tree_height_l256_25662


namespace find_breadth_of_rectangular_plot_l256_25626

-- Define the conditions
def length_is_thrice_breadth (b l : ℕ) : Prop := l = 3 * b
def area_is_363 (b l : ℕ) : Prop := l * b = 363

-- State the theorem
theorem find_breadth_of_rectangular_plot : ∃ b : ℕ, ∀ l : ℕ, length_is_thrice_breadth b l ∧ area_is_363 b l → b = 11 := 
by
  sorry

end find_breadth_of_rectangular_plot_l256_25626


namespace Jillian_had_200_friends_l256_25657

def oranges : ℕ := 80
def pieces_per_orange : ℕ := 10
def pieces_per_friend : ℕ := 4
def number_of_friends : ℕ := oranges * pieces_per_orange / pieces_per_friend

theorem Jillian_had_200_friends :
  number_of_friends = 200 :=
sorry

end Jillian_had_200_friends_l256_25657


namespace great_eighteen_hockey_league_games_l256_25609

theorem great_eighteen_hockey_league_games :
  (let teams_per_division := 9
   let games_intra_division_per_team := 8 * 3
   let games_inter_division_per_team := teams_per_division * 2
   let total_games_per_team := games_intra_division_per_team + games_inter_division_per_team
   let total_game_instances := 18 * total_games_per_team
   let unique_games := total_game_instances / 2
   unique_games = 378) :=
by
  sorry

end great_eighteen_hockey_league_games_l256_25609


namespace find_rate_percent_l256_25632

-- Definitions based on the given conditions
def principal : ℕ := 800
def time : ℕ := 4
def simple_interest : ℕ := 192
def si_formula (P R T : ℕ) : ℕ := P * R * T / 100

-- Statement: prove that the rate percent (R) is 6%
theorem find_rate_percent (R : ℕ) (h : simple_interest = si_formula principal R time) : R = 6 :=
sorry

end find_rate_percent_l256_25632


namespace ratio_of_altitude_to_radius_l256_25611

theorem ratio_of_altitude_to_radius (r R h : ℝ)
  (hR : R = 2 * r)
  (hV : (1/3) * π * R^2 * h = (1/3) * (4/3) * π * r^3) :
  h / R = 1 / 6 := by
  sorry

end ratio_of_altitude_to_radius_l256_25611


namespace rem_frac_l256_25651

def rem (x y : ℚ) : ℚ := x - y * (⌊x / y⌋ : ℤ)

theorem rem_frac : rem (7 / 12) (-3 / 4) = -1 / 6 :=
by
  sorry

end rem_frac_l256_25651


namespace equilateral_triangle_side_length_l256_25683

theorem equilateral_triangle_side_length (a : ℝ) (h : 3 * a = 18) : a = 6 :=
by
  sorry

end equilateral_triangle_side_length_l256_25683


namespace percentage_increase_l256_25642

variable (A B C : ℝ)
variable (h1 : A = 0.71 * C)
variable (h2 : A = 0.05 * B)

theorem percentage_increase (A B C : ℝ) (h1 : A = 0.71 * C) (h2 : A = 0.05 * B) : (B - C) / C = 13.2 :=
by
  sorry

end percentage_increase_l256_25642


namespace rectangle_length_l256_25663

theorem rectangle_length (P B L : ℕ) (h1 : P = 800) (h2 : B = 300) (h3 : P = 2 * (L + B)) : L = 100 :=
by
  sorry

end rectangle_length_l256_25663


namespace factorize_polynomial_l256_25688

theorem factorize_polynomial (x y : ℝ) : 
  (x^2 - y^2 - 2 * x - 4 * y - 3) = (x + y + 1) * (x - y - 3) :=
  sorry

end factorize_polynomial_l256_25688


namespace cube_volume_l256_25689

theorem cube_volume (h : 12 * l = 72) : l^3 = 216 :=
sorry

end cube_volume_l256_25689


namespace evaluate_expression_l256_25631

theorem evaluate_expression : 
  let e := 3 + 2 * Real.sqrt 3 + 1 / (3 + 2 * Real.sqrt 3) + 1 / (2 * Real.sqrt 3 - 3)
  e = 3 + 10 * Real.sqrt 3 / 3 :=
by
  let e := 3 + 2 * Real.sqrt 3 + 1 / (3 + 2 * Real.sqrt 3) + 1 / (2 * Real.sqrt 3 - 3)
  have h : e = 3 + 10 * Real.sqrt 3 / 3 := sorry
  exact h

end evaluate_expression_l256_25631


namespace quadratic_roots_range_l256_25656

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    (x1^2 - 2 * x1 + m - 2 = 0) ∧ 
    (x2^2 - 2 * x2 + m - 2 = 0)) → m < 3 := 
by 
  sorry

end quadratic_roots_range_l256_25656


namespace find_Y_value_l256_25637

theorem find_Y_value : ∃ Y : ℤ, 80 - (Y - (6 + 2 * (7 - 8 - 5))) = 89 ∧ Y = -15 := by
  sorry

end find_Y_value_l256_25637


namespace problem_inequality_l256_25628

theorem problem_inequality (a b : ℝ) (hab : 1 / a + 1 / b = 1) : 
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) := 
by
  sorry

end problem_inequality_l256_25628


namespace math_problem_l256_25622

theorem math_problem :
  (Int.ceil ((16 / 5 : ℚ) * (-34 / 4 : ℚ)) - Int.floor ((16 / 5 : ℚ) * Int.floor (-34 / 4 : ℚ))) = 2 :=
by
  sorry

end math_problem_l256_25622


namespace measure_of_α_l256_25666

variables (α β : ℝ)
-- Condition 1: α and β are complementary angles
def complementary := α + β = 180

-- Condition 2: Half of angle β is 30° less than α
def half_less_30 := α - (1 / 2) * β = 30

-- Theorem: Measure of angle α
theorem measure_of_α (α β : ℝ) (h1 : complementary α β) (h2 : half_less_30 α β) :
  α = 80 :=
by
  sorry

end measure_of_α_l256_25666


namespace percentage_increase_l256_25694

theorem percentage_increase (x : ℝ) (y : ℝ) (h1 : x = 114.4) (h2 : y = 88) : 
  ((x - y) / y) * 100 = 30 := 
by 
  sorry

end percentage_increase_l256_25694


namespace polygon_sides_arithmetic_progression_l256_25633

theorem polygon_sides_arithmetic_progression
  (angles_in_arithmetic_progression : ∃ (a d : ℝ) (angles : ℕ → ℝ), ∀ (k : ℕ), angles k = a + k * d)
  (common_difference : ∃ (d : ℝ), d = 3)
  (largest_angle : ∃ (n : ℕ) (angles : ℕ → ℝ), angles n = 150) :
  ∃ (n : ℕ), n = 15 :=
sorry

end polygon_sides_arithmetic_progression_l256_25633


namespace nonneg_integer_solution_l256_25679

theorem nonneg_integer_solution (a b c : ℕ) (h : 5^a * 7^b + 4 = 3^c) : (a, b, c) = (1, 0, 2) := 
sorry

end nonneg_integer_solution_l256_25679


namespace smallest_enclosing_sphere_radius_l256_25680

theorem smallest_enclosing_sphere_radius :
  let r := 2
  let d := 4 * Real.sqrt 3
  let total_diameter := d + 2*r
  let radius_enclosing_sphere := total_diameter / 2
  radius_enclosing_sphere = 2 + 2 * Real.sqrt 3 := by
  -- Define the radius of the smaller spheres
  let r : ℝ := 2
  -- Space diagonal of the cube which is 4√3 where 4 is the side length
  let d : ℝ := 4 * Real.sqrt 3
  -- Total diameter of the sphere containing the cube (space diagonal + 2 radius of one sphere)
  let total_diameter : ℝ := d + 2 * r
  -- Radius of the enclosing sphere
  let radius_enclosing_sphere : ℝ := total_diameter / 2
  -- We need to prove that this radius equals 2 + 2√3
  sorry

end smallest_enclosing_sphere_radius_l256_25680


namespace product_of_roots_l256_25653

theorem product_of_roots (a b c d : ℝ)
  (h1 : a = 16 ^ (1 / 5))
  (h2 : 16 = 2 ^ 4)
  (h3 : b = 64 ^ (1 / 6))
  (h4 : 64 = 2 ^ 6):
  a * b = 2 * (16 ^ (1 / 5)) := by
  sorry

end product_of_roots_l256_25653


namespace max_volume_of_prism_l256_25610

theorem max_volume_of_prism (a b c s : ℝ) (h : a + b + c = 3 * s) : a * b * c ≤ s^3 :=
by {
    -- placeholder for the proof
    sorry
}

end max_volume_of_prism_l256_25610


namespace negative_half_power_zero_l256_25624

theorem negative_half_power_zero : (- (1 / 2)) ^ 0 = 1 :=
by
  sorry

end negative_half_power_zero_l256_25624


namespace find_number_of_girls_l256_25654

noncomputable def B (G : ℕ) : ℕ := (8 * G) / 5

theorem find_number_of_girls (B G : ℕ) (h_ratio : B = (8 * G) / 5) (h_total : B + G = 312) : G = 120 :=
by
  -- the proof would be done here
  sorry

end find_number_of_girls_l256_25654


namespace train_crossing_time_l256_25670

theorem train_crossing_time
  (length : ℝ) (speed : ℝ) (time : ℝ)
  (h1 : length = 100) (h2 : speed = 30.000000000000004) :
  time = length / speed :=
by
  sorry

end train_crossing_time_l256_25670


namespace max_has_two_nickels_l256_25636

theorem max_has_two_nickels (n : ℕ) (nickels : ℕ) (coins_value_total : ℕ) :
  (coins_value_total = 15 * n) -> (coins_value_total + 10 = 16 * (n + 1)) -> 
  coins_value_total - nickels * 5 + nickels + 25 = 90 -> 
  n = 6 -> 
  2 = nickels := 
by 
  sorry

end max_has_two_nickels_l256_25636


namespace tangerine_count_l256_25643

-- Definitions based directly on the conditions
def initial_oranges : ℕ := 5
def remaining_oranges : ℕ := initial_oranges - 2
def remaining_tangerines (T : ℕ) : ℕ := T - 10
def condition1 (T : ℕ) : Prop := remaining_tangerines T = remaining_oranges + 4

-- Theorem to prove the number of tangerines in the bag
theorem tangerine_count (T : ℕ) (h : condition1 T) : T = 17 :=
by
  sorry

end tangerine_count_l256_25643


namespace frac_m_q_eq_one_l256_25635

theorem frac_m_q_eq_one (m n p q : ℕ) 
  (h1 : m = 40 * n)
  (h2 : p = 5 * n)
  (h3 : p = q / 8) : (m / q = 1) :=
by
  sorry

end frac_m_q_eq_one_l256_25635


namespace tanA_over_tanB_l256_25648

noncomputable def tan_ratios (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A + 2 * c = 0

theorem tanA_over_tanB {A B C a b c : ℝ} (h : tan_ratios A B C a b c) : 
  Real.tan A / Real.tan B = -1 / 3 :=
by
  sorry

end tanA_over_tanB_l256_25648


namespace prob_of_target_hit_l256_25685

noncomputable def probability_target_hit : ℚ :=
  let pA := (1 : ℚ) / 2
  let pB := (1 : ℚ) / 3
  let pC := (1 : ℚ) / 4
  let pA' := 1 - pA
  let pB' := 1 - pB
  let pC' := 1 - pC
  let pNoneHit := pA' * pB' * pC'
  1 - pNoneHit

-- Statement to be proved
theorem prob_of_target_hit : probability_target_hit = 3 / 4 :=
  sorry

end prob_of_target_hit_l256_25685


namespace tangent_slope_at_1_0_l256_25646

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_slope_at_1_0 : (deriv f 1) = 3 := by
  sorry

end tangent_slope_at_1_0_l256_25646


namespace maximize_xyz_l256_25614

theorem maximize_xyz (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 60) :
    (x, y, z) = (20, 40 / 3, 80 / 3) → x^3 * y^2 * z^4 ≤ 20^3 * (40 / 3)^2 * (80 / 3)^4 :=
by
  sorry

end maximize_xyz_l256_25614


namespace al_original_portion_l256_25675

theorem al_original_portion (a b c : ℝ) (h1 : a + b + c = 1200) (h2 : 0.75 * a + 2 * b + 2 * c = 1800) : a = 480 :=
by
  sorry

end al_original_portion_l256_25675


namespace division_then_multiplication_l256_25602

theorem division_then_multiplication : (180 / 6) * 3 = 90 := 
by
  have step1 : 180 / 6 = 30 := sorry
  have step2 : 30 * 3 = 90 := sorry
  sorry

end division_then_multiplication_l256_25602


namespace find_denominator_of_second_fraction_l256_25696

theorem find_denominator_of_second_fraction (y : ℝ) (h : y > 0) (x : ℝ) :
  (2 * y) / 5 + (3 * y) / x = 0.7 * y → x = 10 :=
by
  sorry

end find_denominator_of_second_fraction_l256_25696


namespace rooster_ratio_l256_25697

theorem rooster_ratio (R H : ℕ) 
  (h1 : R + H = 80)
  (h2 : R + (1 / 4) * H = 35) :
  R / 80 = 1 / 4 :=
  sorry

end rooster_ratio_l256_25697


namespace arccos_one_over_sqrt_two_eq_pi_four_l256_25612

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l256_25612


namespace sum_of_possible_values_l256_25605

variable {S : ℝ} (h : S ≠ 0)

theorem sum_of_possible_values (h : S ≠ 0) : ∃ N : ℝ, N ≠ 0 ∧ 6 * N + 2 / N = S → ∀ N1 N2 : ℝ, (6 * N1 + 2 / N1 = S ∧ 6 * N2 + 2 / N2 = S) → (N1 + N2) = S / 6 :=
by
  sorry

end sum_of_possible_values_l256_25605


namespace tom_boxes_needed_l256_25669

-- Definitions of given conditions
def room_length : ℕ := 16
def room_width : ℕ := 20
def box_coverage : ℕ := 10
def already_covered : ℕ := 250

-- The total area of the living room
def total_area : ℕ := room_length * room_width

-- The remaining area that needs to be covered
def remaining_area : ℕ := total_area - already_covered

-- The number of boxes required to cover the remaining area
def boxes_needed : ℕ := remaining_area / box_coverage

-- The theorem statement
theorem tom_boxes_needed : boxes_needed = 7 := by
  -- The proof will go here
  sorry

end tom_boxes_needed_l256_25669


namespace arithmetic_sequence_sum_l256_25687

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n+1) = a n + d)
  (h1 : a 2 + a 3 = 1)
  (h2 : a 10 + a 11 = 9) :
  a 5 + a 6 = 4 :=
sorry

end arithmetic_sequence_sum_l256_25687


namespace contrapositive_example_l256_25665

theorem contrapositive_example (x : ℝ) :
  (x ^ 2 < 1 → -1 < x ∧ x < 1) ↔ (x ≥ 1 ∨ x ≤ -1 → x ^ 2 ≥ 1) :=
sorry

end contrapositive_example_l256_25665


namespace original_price_of_shirts_l256_25693

theorem original_price_of_shirts 
  (sale_price : ℝ) 
  (fraction_of_original : ℝ) 
  (original_price : ℝ) 
  (h1 : sale_price = 6) 
  (h2 : fraction_of_original = 0.25) 
  (h3 : sale_price = fraction_of_original * original_price) 
  : original_price = 24 := 
by 
  sorry

end original_price_of_shirts_l256_25693


namespace partition_natural_numbers_l256_25629

theorem partition_natural_numbers :
  ∃ (f : ℕ → ℕ), (∀ n, 1 ≤ f n ∧ f n ≤ 100) ∧
  (∀ a b c, a + 99 * b = c → f a = f c ∨ f a = f b ∨ f b = f c) :=
sorry

end partition_natural_numbers_l256_25629


namespace transport_cost_l256_25639

theorem transport_cost (weight_g : ℕ) (cost_per_kg : ℕ) (weight_kg : ℕ) (total_cost : ℕ)
  (h1 : weight_g = 2000)
  (h2 : cost_per_kg = 15000)
  (h3 : weight_kg = weight_g / 1000)
  (h4 : total_cost = weight_kg * cost_per_kg) :
  total_cost = 30000 :=
by
  sorry

end transport_cost_l256_25639


namespace area_of_quadrilateral_ABDE_l256_25619

-- Definitions for the given problem
variable (AB CE AC DE : ℝ)
variable (parABCE parACDE : Prop)
variable (areaCOD : ℝ)

-- Lean 4 statement for the proof problem
theorem area_of_quadrilateral_ABDE
  (h1 : parABCE)
  (h2 : parACDE)
  (h3 : AB = 5)
  (h4 : AC = 5)
  (h5 : CE = 10)
  (h6 : DE = 10)
  (h7 : areaCOD = 10)
  : (AB + AC + CE + DE) / 2 + areaCOD = 52.5 := 
sorry

end area_of_quadrilateral_ABDE_l256_25619


namespace original_movie_length_l256_25630

theorem original_movie_length (final_length cut_scene original_length : ℕ) 
    (h1 : cut_scene = 3) (h2 : final_length = 57) (h3 : final_length + cut_scene = original_length) : 
  original_length = 60 := 
by 
  -- Proof omitted
  sorry

end original_movie_length_l256_25630


namespace find_a_l256_25647

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : (a^2 + a = 6)) : a = 2 :=
sorry

end find_a_l256_25647


namespace member_number_property_l256_25627

theorem member_number_property :
  ∃ (country : Fin 6) (member_number : Fin 1978),
    (∀ (i j : Fin 1978), i ≠ j → member_number ≠ i + j) ∨
    (∀ (k : Fin 1978), member_number ≠ 2 * k) :=
by
  sorry

end member_number_property_l256_25627


namespace evaluate_expression_l256_25698

theorem evaluate_expression (b : ℤ) (x : ℤ) (h : x = b + 9) : (x - b + 5 = 14) :=
by
  sorry

end evaluate_expression_l256_25698


namespace find_x2_y2_l256_25682

theorem find_x2_y2 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x * y + 2 * x + 2 * y = 152)
  (h2 : x^2 * y + x * y^2 = 1512) :
  x^2 + y^2 = 1136 ∨ x^2 + y^2 = 221 := by
  sorry

end find_x2_y2_l256_25682
