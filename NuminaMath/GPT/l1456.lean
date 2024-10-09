import Mathlib

namespace chairs_to_remove_l1456_145638

/-- A conference hall is setting up seating for a lecture with specific conditions.
    Given the total number of chairs, chairs per row, and participants expected to attend,
    prove the number of chairs to be removed to have complete rows with the least number of empty seats. -/
theorem chairs_to_remove
  (chairs_per_row : ℕ) (total_chairs : ℕ) (expected_participants : ℕ)
  (h1 : chairs_per_row = 15)
  (h2 : total_chairs = 225)
  (h3 : expected_participants = 140) :
  total_chairs - (chairs_per_row * ((expected_participants + chairs_per_row - 1) / chairs_per_row)) = 75 :=
by
  sorry

end chairs_to_remove_l1456_145638


namespace increasing_interval_of_f_l1456_145646

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 15 * x ^ 2 + 36 * x - 24

theorem increasing_interval_of_f : (∀ x : ℝ, x = 2 → deriv f x = 0) → ∀ x : ℝ, 3 < x → 0 < deriv f x :=
by
  intro h x hx
  -- We know that the function has an extreme value at x = 2
  have : deriv f 2 = 0 := h 2 rfl
  -- Require to prove the function is increasing in interval (3, +∞)
  sorry

end increasing_interval_of_f_l1456_145646


namespace sweets_distribution_l1456_145636

theorem sweets_distribution (S : ℕ) (N : ℕ) (h1 : N - 70 > 0) (h2 : S = N * 24) (h3 : S = (N - 70) * 38) : N = 190 :=
by
  sorry

end sweets_distribution_l1456_145636


namespace sum_eq_two_l1456_145620

theorem sum_eq_two (x y : ℝ) (hx : x^3 - 3 * x^2 + 5 * x = 1) (hy : y^3 - 3 * y^2 + 5 * y = 5) : x + y = 2 := 
sorry

end sum_eq_two_l1456_145620


namespace graph_comparison_l1456_145692

theorem graph_comparison :
  (∀ x : ℝ, (x^2 - x + 3) < (x^2 - x + 5)) :=
by
  sorry

end graph_comparison_l1456_145692


namespace find_initial_crayons_l1456_145649

namespace CrayonProblem

variable (gave : ℕ) (lost : ℕ) (additional_lost : ℕ) 

def correct_answer (gave lost additional_lost : ℕ) :=
  gave + lost = gave + (gave + additional_lost) ∧ gave + lost = 502

theorem find_initial_crayons
  (gave := 90)
  (lost := 412)
  (additional_lost := 322)
  : correct_answer gave lost additional_lost :=
by 
  sorry

end CrayonProblem

end find_initial_crayons_l1456_145649


namespace sum_of_five_consecutive_squares_not_perfect_square_l1456_145659

theorem sum_of_five_consecutive_squares_not_perfect_square (n : ℤ) : 
  ¬ ∃ (k : ℤ), k^2 = (n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 := 
by
  sorry

end sum_of_five_consecutive_squares_not_perfect_square_l1456_145659


namespace simplify_fractions_l1456_145648

theorem simplify_fractions : (360 / 32) * (10 / 240) * (16 / 6) = 10 := by
  sorry

end simplify_fractions_l1456_145648


namespace number_of_integer_pairs_l1456_145643

theorem number_of_integer_pairs (n : ℕ) : 
  ∃ (count : ℕ), count = 2 * n^2 + 2 * n + 1 ∧ 
  ∀ x y : ℤ, abs x + abs y ≤ n ↔
  count = 2 * n^2 + 2 * n + 1 :=
by
  sorry

end number_of_integer_pairs_l1456_145643


namespace valid_starting_lineups_correct_l1456_145619

-- Define the parameters from the problem
def volleyball_team : Finset ℕ := Finset.range 18
def quadruplets : Finset ℕ := {0, 1, 2, 3}

-- Define the main computation: total lineups excluding those where all quadruplets are chosen
noncomputable def valid_starting_lineups : ℕ :=
  (volleyball_team.card.choose 7) - ((volleyball_team \ quadruplets).card.choose 3)

-- The theorem states that the number of valid starting lineups is 31460
theorem valid_starting_lineups_correct : valid_starting_lineups = 31460 := by
  sorry

end valid_starting_lineups_correct_l1456_145619


namespace num_people_price_item_equation_l1456_145661

theorem num_people_price_item_equation
  (x y : ℕ)
  (h1 : 8 * x = y + 3)
  (h2 : 7 * x = y - 4) :
  (y + 3) / 8 = (y - 4) / 7 :=
by
  sorry

end num_people_price_item_equation_l1456_145661


namespace quadratic_root_exists_l1456_145694

theorem quadratic_root_exists {a b c d : ℝ} (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
by
  sorry

end quadratic_root_exists_l1456_145694


namespace total_earnings_from_selling_working_games_l1456_145698

-- Conditions definition
def total_games : ℕ := 16
def broken_games : ℕ := 8
def working_games : ℕ := total_games - broken_games
def game_prices : List ℕ := [6, 7, 9, 5, 8, 10, 12, 11]

-- Proof problem statement
theorem total_earnings_from_selling_working_games : List.sum game_prices = 68 := by
  sorry

end total_earnings_from_selling_working_games_l1456_145698


namespace maximum_unique_walks_l1456_145686

-- Define the conditions
def starts_at_A : Prop := true
def crosses_bridge_1_first : Prop := true
def finishes_at_B : Prop := true
def six_bridges_linking_two_islands_and_banks : Prop := true

-- Define the theorem to prove the maximum number of unique walks is 6
theorem maximum_unique_walks : starts_at_A ∧ crosses_bridge_1_first ∧ finishes_at_B ∧ six_bridges_linking_two_islands_and_banks → ∃ n, n = 6 :=
by
  intros
  existsi 6
  sorry

end maximum_unique_walks_l1456_145686


namespace find_k_circle_radius_l1456_145666

theorem find_k_circle_radius (k : ℝ) :
  (∀ x y : ℝ, (x^2 + 8 * x + y^2 + 4 * y - k = 0) → ((x + 4)^2 + (y + 2)^2 = 7^2)) → k = 29 :=
sorry

end find_k_circle_radius_l1456_145666


namespace remainder_of_x13_plus_1_by_x_minus_1_l1456_145639

-- Define the polynomial f(x) = x^13 + 1
def f (x : ℕ) : ℕ := x ^ 13 + 1

-- State the theorem using the Polynomial Remainder Theorem
theorem remainder_of_x13_plus_1_by_x_minus_1 : f 1 = 2 := by
  -- Skip the proof
  sorry

end remainder_of_x13_plus_1_by_x_minus_1_l1456_145639


namespace standing_in_a_row_standing_in_a_row_AB_adj_CD_not_adj_assign_to_classes_l1456_145632

theorem standing_in_a_row (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 → 
  ∃ (ways : ℕ), ways = 120 :=
by
  sorry

theorem standing_in_a_row_AB_adj_CD_not_adj (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 →
  ∃ (ways : ℕ), ways = 24 :=
by
  sorry

theorem assign_to_classes (students : Finset String) (h : students = {"A", "B", "C", "D", "E"}) :
  students.card = 5 →
  ∃ (ways : ℕ), ways = 150 :=
by
  sorry

end standing_in_a_row_standing_in_a_row_AB_adj_CD_not_adj_assign_to_classes_l1456_145632


namespace scientific_notation_of_distance_l1456_145685

theorem scientific_notation_of_distance :
  ∃ a n, (1 ≤ a ∧ a < 10) ∧ 384000 = a * 10^n ∧ a = 3.84 ∧ n = 5 :=
by
  sorry

end scientific_notation_of_distance_l1456_145685


namespace sqrt8_same_type_as_sqrt2_l1456_145609

def same_type_sqrt_2 (x : Real) : Prop := ∃ k : Real, k * Real.sqrt 2 = x

theorem sqrt8_same_type_as_sqrt2 : same_type_sqrt_2 (Real.sqrt 8) :=
  sorry

end sqrt8_same_type_as_sqrt2_l1456_145609


namespace weight_of_six_moles_BaF2_l1456_145612

variable (atomic_weight_Ba : ℝ := 137.33) -- Atomic weight of Barium in g/mol
variable (atomic_weight_F : ℝ := 19.00) -- Atomic weight of Fluorine in g/mol
variable (moles_BaF2 : ℝ := 6) -- Number of moles of BaF2

theorem weight_of_six_moles_BaF2 :
  moles_BaF2 * (atomic_weight_Ba + 2 * atomic_weight_F) = 1051.98 :=
by sorry

end weight_of_six_moles_BaF2_l1456_145612


namespace solve_for_y_l1456_145629

theorem solve_for_y (y : ℝ) : 5 * y - 100 = 125 ↔ y = 45 := by
  sorry

end solve_for_y_l1456_145629


namespace sandwiches_ordered_l1456_145647

-- Definitions of the given conditions
def sandwichCost : ℕ := 5
def payment : ℕ := 20
def change : ℕ := 5

-- Statement to prove how many sandwiches Jack ordered
theorem sandwiches_ordered : (payment - change) / sandwichCost = 3 := by
  -- Sorry to skip the proof
  sorry

end sandwiches_ordered_l1456_145647


namespace quadratic_passes_through_origin_quadratic_symmetric_about_y_axis_l1456_145627

-- Define the quadratic function
def quadratic (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + m - 2

-- Problem 1: Prove that the quadratic function passes through the origin for m = 1 or m = -2
theorem quadratic_passes_through_origin :
  ∃ m : ℝ, (m = 1 ∨ m = -2) ∧ quadratic m 0 = 0 := by
  sorry

-- Problem 2: Prove that the quadratic function is symmetric about the y-axis for m = 0
theorem quadratic_symmetric_about_y_axis :
  ∃ m : ℝ, m = 0 ∧ ∀ x : ℝ, quadratic m x = quadratic m (-x) := by
  sorry

end quadratic_passes_through_origin_quadratic_symmetric_about_y_axis_l1456_145627


namespace molecular_weight_of_one_mole_l1456_145682

theorem molecular_weight_of_one_mole 
  (molicular_weight_9_moles : ℕ) 
  (weight_9_moles : ℕ)
  (h : molicular_weight_9_moles = 972 ∧ weight_9_moles = 9) : 
  molicular_weight_9_moles / weight_9_moles = 108 := 
  by
    sorry

end molecular_weight_of_one_mole_l1456_145682


namespace determine_A_l1456_145669

open Real

theorem determine_A (A B C : ℝ)
  (h_decomposition : ∀ x, x ≠ 4 ∧ x ≠ -2 -> (x + 2) / (x^3 - 9 * x^2 + 14 * x + 24) = A / (x - 4) + B / (x - 3) + C / (x + 2)^2)
  (h_factorization : ∀ x, (x^3 - 9 * x^2 + 14 * x + 24) = (x - 4) * (x - 3) * (x + 2)^2) :
  A = 1 / 6 := 
sorry

end determine_A_l1456_145669


namespace geom_prog_terms_exist_l1456_145679

theorem geom_prog_terms_exist (b3 b6 : ℝ) (h1 : b3 = -1) (h2 : b6 = 27 / 8) :
  ∃ (b1 q : ℝ), b1 = -4 / 9 ∧ q = -3 / 2 :=
by
  sorry

end geom_prog_terms_exist_l1456_145679


namespace quadratic_factorization_l1456_145624

theorem quadratic_factorization (a b : ℕ) (h1 : x^2 - 20 * x + 96 = (x - a) * (x - b)) (h2 : a > b) : 2 * b - a = 4 :=
sorry

end quadratic_factorization_l1456_145624


namespace find_required_water_amount_l1456_145608

-- Definitions based on the conditions
def sanitizer_volume : ℝ := 12
def initial_alcohol_concentration : ℝ := 0.60
def desired_alcohol_concentration : ℝ := 0.40

-- Statement of the proof problem
theorem find_required_water_amount : 
  ∃ (x : ℝ), x = 6 ∧ sanitizer_volume * initial_alcohol_concentration = desired_alcohol_concentration * (sanitizer_volume + x) :=
sorry

end find_required_water_amount_l1456_145608


namespace find_g_values_l1456_145696

variables (f g : ℝ → ℝ)

-- Conditions
axiom cond1 : ∀ x y, g (x - y) = g x * g y + f x * f y
axiom cond2 : f (-1) = -1
axiom cond3 : f 0 = 0
axiom cond4 : f 1 = 1

-- Goal
theorem find_g_values : g 0 = 1 ∧ g 1 = 0 ∧ g 2 = -1 :=
by
  sorry

end find_g_values_l1456_145696


namespace find_factor_l1456_145615

theorem find_factor (x : ℕ) (f : ℕ) (h1 : x = 9)
  (h2 : (2 * x + 6) * f = 72) : f = 3 := by
  sorry

end find_factor_l1456_145615


namespace no_pieces_left_impossible_l1456_145621

/-- Starting with 100 pieces and 1 pile, and given the ability to either:
1. Remove one piece from a pile of at least 3 pieces and divide the remaining pile into two non-empty piles,
2. Eliminate a pile containing a single piece,
prove that it is impossible to reach a situation with no pieces left. -/
theorem no_pieces_left_impossible :
  ∀ (p t : ℕ), p = 100 → t = 1 →
  (∀ (p' t' : ℕ),
    (p' = p - 1 ∧ t' = t + 1 ∧ 3 ≤ p) ∨
    (p' = p - 1 ∧ t' = t - 1 ∧ ∃ k, k = 1 ∧ t ≠ 0) →
    false) :=
by
  intros
  sorry

end no_pieces_left_impossible_l1456_145621


namespace sonic_leads_by_19_2_meters_l1456_145658

theorem sonic_leads_by_19_2_meters (v_S v_D : ℝ)
  (h1 : ∀ t, t = 200 / v_S → 200 = v_S * t)
  (h2 : ∀ t, t = 184 / v_D → 184 = v_D * t)
  (h3 : v_S / v_D = 200 / 184)
  :  240 / v_S - (200 / v_S / (200 / 184) * 240) = 19.2 := by
  sorry

end sonic_leads_by_19_2_meters_l1456_145658


namespace find_m_value_l1456_145681

open Real

-- Define the vectors a and b as specified in the problem
def vec_a (m : ℝ) : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (3, -2)

-- Define the sum of vectors a and b
def vec_sum (m : ℝ) : ℝ × ℝ := (1 + 3, m - 2)

-- Define the dot product of the vector sum with vector b to be zero as the given condition
def dot_product (m : ℝ) : ℝ := (vec_sum m).1 * vec_b.1 + (vec_sum m).2 * vec_b.2

-- The theorem to prove that given the defined conditions, m equals 8
theorem find_m_value (m : ℝ) (h : dot_product m = 0) : m = 8 := by
  sorry

end find_m_value_l1456_145681


namespace num_games_round_robin_l1456_145652

-- There are 10 classes in the second grade, each class forms one team.
def num_teams := 10

-- A round-robin format means each team plays against every other team once.
def num_games (n : Nat) := n * (n - 1) / 2

-- Proving the total number of games played with num_teams equals to 45
theorem num_games_round_robin : num_games num_teams = 45 := by
  sorry

end num_games_round_robin_l1456_145652


namespace number_of_people_entered_l1456_145604

-- Define the total number of placards
def total_placards : ℕ := 5682

-- Define the number of placards each person takes
def placards_per_person : ℕ := 2

-- The Lean theorem to prove the number of people who entered the stadium
theorem number_of_people_entered : total_placards / placards_per_person = 2841 :=
by
  -- Proof will be inserted here
  sorry

end number_of_people_entered_l1456_145604


namespace cricketer_average_after_19_innings_l1456_145687

theorem cricketer_average_after_19_innings
  (A : ℝ) 
  (total_runs_after_18 : ℝ := 18 * A) 
  (runs_in_19th : ℝ := 99) 
  (new_avg : ℝ := A + 4) 
  (total_runs_after_19 : ℝ := total_runs_after_18 + runs_in_19th) 
  (equation : 19 * new_avg = total_runs_after_19) : 
  new_avg = 27 :=
by
  sorry

end cricketer_average_after_19_innings_l1456_145687


namespace janice_work_days_l1456_145655

variable (dailyEarnings : Nat)
variable (overtimeEarnings : Nat)
variable (numOvertimeShifts : Nat)
variable (totalEarnings : Nat)

theorem janice_work_days
    (h1 : dailyEarnings = 30)
    (h2 : overtimeEarnings = 15)
    (h3 : numOvertimeShifts = 3)
    (h4 : totalEarnings = 195)
    : let overtimeTotal := numOvertimeShifts * overtimeEarnings
      let regularEarnings := totalEarnings - overtimeTotal
      let workDays := regularEarnings / dailyEarnings
      workDays = 5 :=
by
  sorry

end janice_work_days_l1456_145655


namespace hotel_ticket_ratio_l1456_145653

theorem hotel_ticket_ratio (initial_amount : ℕ) (remaining_amount : ℕ) (ticket_cost : ℕ) (hotel_cost : ℕ) :
  initial_amount = 760 →
  remaining_amount = 310 →
  ticket_cost = 300 →
  initial_amount - remaining_amount - ticket_cost = hotel_cost →
  (hotel_cost : ℚ) / (ticket_cost : ℚ) = 1 / 2 :=
by
  intros h_initial h_remaining h_ticket h_hotel
  sorry

end hotel_ticket_ratio_l1456_145653


namespace part_1_relationship_part_2_solution_part_2_preferred_part_3_max_W_part_3_max_at_28_l1456_145664

noncomputable def y (x : ℝ) : ℝ := -10 * x + 400
noncomputable def W (x : ℝ) : ℝ := -10 * x^2 + 500 * x - 4000

theorem part_1_relationship (x : ℝ) (h₀ : 0 < x) (h₁ : x ≤ 40) :
  W x = -10 * x^2 + 500 * x - 4000 := by
  sorry

theorem part_2_solution (x : ℝ) (h₀ : W x = 1250) :
  x = 15 ∨ x = 35 := by
  sorry

theorem part_2_preferred (x : ℝ) (h₀ : W x = 1250) (h₁ : y 15 ≥ y 35) :
  x = 15 := by
  sorry

theorem part_3_max_W (x : ℝ) (h₀ : 28 ≤ x) (h₁ : x ≤ 35) :
  W x ≤ 2160 := by
  sorry

theorem part_3_max_at_28 :
  W 28 = 2160 := by
  sorry

end part_1_relationship_part_2_solution_part_2_preferred_part_3_max_W_part_3_max_at_28_l1456_145664


namespace intersection_lines_l1456_145622

theorem intersection_lines (x y : ℝ) :
  (2 * x - y - 10 = 0) ∧ (3 * x + 4 * y - 4 = 0) → (x = 4) ∧ (y = -2) :=
by
  -- The proof is provided here
  sorry

end intersection_lines_l1456_145622


namespace son_present_age_l1456_145672

-- Definitions
variables (S M : ℕ)
-- Conditions
def age_diff : Prop := M = S + 22
def future_age_condition : Prop := M + 2 = 2 * (S + 2)

-- Theorem statement with proof placeholder
theorem son_present_age (H1 : age_diff S M) (H2 : future_age_condition S M) : S = 20 :=
by sorry

end son_present_age_l1456_145672


namespace grid_game_winner_l1456_145654

theorem grid_game_winner {m n : ℕ} :
  (if (m + n) % 2 = 0 then "Second player wins" else "First player wins") = (if (m + n) % 2 = 0 then "Second player wins" else "First player wins") := by
  sorry

end grid_game_winner_l1456_145654


namespace veggies_minus_fruits_l1456_145651

-- Definitions of quantities as given in the conditions
def cucumbers : ℕ := 6
def tomatoes : ℕ := 8
def apples : ℕ := 2
def bananas : ℕ := 4

-- Problem Statement
theorem veggies_minus_fruits : (cucumbers + tomatoes) - (apples + bananas) = 8 :=
by 
  -- insert proof here
  sorry

end veggies_minus_fruits_l1456_145651


namespace Sheila_attendance_probability_l1456_145611

-- Definitions as per given conditions
def P_rain := 0.5
def P_sunny := 0.3
def P_cloudy := 0.2
def P_Sheila_goes_given_rain := 0.3
def P_Sheila_goes_given_sunny := 0.7
def P_Sheila_goes_given_cloudy := 0.5

-- Define the probability calculation
def P_Sheila_attends := 
  (P_rain * P_Sheila_goes_given_rain) + 
  (P_sunny * P_Sheila_goes_given_sunny) + 
  (P_cloudy * P_Sheila_goes_given_cloudy)

-- Final theorem statement
theorem Sheila_attendance_probability : P_Sheila_attends = 0.46 := by
  sorry

end Sheila_attendance_probability_l1456_145611


namespace min_value_of_expression_l1456_145677

theorem min_value_of_expression (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 10) : 
  ∃ B, B = x^2 + y^2 + z^2 + x^2 * y ∧ B ≥ 4 :=
by
  sorry

end min_value_of_expression_l1456_145677


namespace sqrt_prod_plus_one_equals_341_l1456_145637

noncomputable def sqrt_prod_plus_one : ℕ :=
  Nat.sqrt ((20 * 19 * 18 * 17) + 1)

theorem sqrt_prod_plus_one_equals_341 :
  sqrt_prod_plus_one = 341 := 
by
  sorry

end sqrt_prod_plus_one_equals_341_l1456_145637


namespace area_ratio_proof_l1456_145602

variables (BE CE DE AE : ℝ)
variables (S_alpha S_beta S_gamma S_delta : ℝ)
variables (x : ℝ)

-- Definitions for the given conditions
def BE_val := 80
def CE_val := 60
def DE_val := 40
def AE_val := 30

-- Expressing the ratios
def S_alpha_ratio := 2
def S_beta_ratio := 2

-- Assuming areas in terms of x
def S_alpha_val := 2 * x
def S_beta_val := 2 * x
def S_delta_val := x
def S_gamma_val := 2 * x

-- Problem statement
theorem area_ratio_proof
  (BE := BE_val)
  (CE := CE_val)
  (DE := DE_val)
  (AE := AE_val)
  (S_alpha := S_alpha_val)
  (S_beta := S_beta_val)
  (S_gamma := S_gamma_val)
  (S_delta := S_delta_val) :
  (S_gamma + S_delta) / (S_alpha + S_beta) = 5 / 4 :=
by
  sorry

end area_ratio_proof_l1456_145602


namespace zero_point_interval_l1456_145641

noncomputable def f (x : ℝ) := 6 / x - x ^ 2

theorem zero_point_interval : ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  sorry

end zero_point_interval_l1456_145641


namespace decreasing_linear_function_l1456_145603

theorem decreasing_linear_function (k : ℝ) : 
  (∀ x1 x2 : ℝ, x1 < x2 → (k - 3) * x1 + 2 > (k - 3) * x2 + 2) → k < 3 := 
by 
  sorry

end decreasing_linear_function_l1456_145603


namespace quadratic_solution_l1456_145600

theorem quadratic_solution :
  (∀ x : ℝ, 3 * x^2 - 13 * x + 5 = 0 → 
           x = (13 + Real.sqrt 109) / 6 ∨ x = (13 - Real.sqrt 109) / 6) 
  := by
  sorry

end quadratic_solution_l1456_145600


namespace students_not_picked_l1456_145644

/-- There are 36 students trying out for the school's trivia teams. 
If some of them didn't get picked and the rest were put into 3 groups with 9 students in each group,
prove that the number of students who didn't get picked is 9. -/

theorem students_not_picked (total_students groups students_per_group picked_students not_picked_students : ℕ)
    (h1 : total_students = 36)
    (h2 : groups = 3)
    (h3 : students_per_group = 9)
    (h4 : picked_students = groups * students_per_group)
    (h5 : not_picked_students = total_students - picked_students) :
    not_picked_students = 9 :=
by
  sorry

end students_not_picked_l1456_145644


namespace shaded_area_eight_l1456_145616

-- Definitions based on given conditions
def arcAQB (r : ℝ) : Prop := r = 2
def arcBRC (r : ℝ) : Prop := r = 2
def midpointQ (r : ℝ) : Prop := arcAQB r
def midpointR (r : ℝ) : Prop := arcBRC r
def midpointS (r : ℝ) : Prop := arcAQB r ∧ arcBRC r ∧ (arcAQB r ∨ arcBRC r)
def arcQRS (r : ℝ) : Prop := r = 2 ∧ midpointS r

-- The theorem to prove
theorem shaded_area_eight (r : ℝ) : arcAQB r ∧ arcBRC r ∧ arcQRS r → area_shaded_region = 8 := by
  sorry

end shaded_area_eight_l1456_145616


namespace carriage_and_people_l1456_145674

variable {x y : ℕ}

theorem carriage_and_people :
  (3 * (x - 2) = y) ∧ (2 * x + 9 = y) :=
sorry

end carriage_and_people_l1456_145674


namespace fraction_power_equiv_l1456_145630

theorem fraction_power_equiv : (75000^4) / (25000^4) = 81 := by
  sorry

end fraction_power_equiv_l1456_145630


namespace number_of_tires_l1456_145662

theorem number_of_tires (n : ℕ)
  (repair_cost : ℕ → ℝ)
  (sales_tax : ℕ → ℝ)
  (total_cost : ℝ) :
  (∀ t, repair_cost t = 7) →
  (∀ t, sales_tax t = 0.5) →
  (total_cost = n * (repair_cost 0 + sales_tax 0)) →
  total_cost = 30 →
  n = 4 :=
by 
  sorry

end number_of_tires_l1456_145662


namespace range_of_a_l1456_145660

variable (a : ℝ)
def p := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def q := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (hpq_or : p a ∨ q a) (hpq_and_false : ¬ (p a ∧ q a)) : 
    a ∈ Set.Iio 0 ∪ Set.Ioo (1/4) 4 :=
by
  sorry

end range_of_a_l1456_145660


namespace inequality_transitive_l1456_145617

theorem inequality_transitive (a b c : ℝ) : a * c^2 > b * c^2 → a > b :=
sorry

end inequality_transitive_l1456_145617


namespace square_AP_square_equals_2000_l1456_145683

noncomputable def square_side : ℝ := 100
noncomputable def midpoint_AB : ℝ := square_side / 2
noncomputable def distance_MP : ℝ := 50
noncomputable def distance_PC : ℝ := square_side

/-- Given a square ABCD with side length 100, midpoint M of AB, MP = 50, and PC = 100, prove AP^2 = 2000 -/
theorem square_AP_square_equals_2000 :
  ∃ (P : ℝ × ℝ), (dist (P.1, P.2) (midpoint_AB, 0) = distance_MP) ∧ (dist (P.1, P.2) (square_side, square_side) = distance_PC) ∧ ((P.1) ^ 2 + (P.2) ^ 2 = 2000) := 
sorry


end square_AP_square_equals_2000_l1456_145683


namespace fraction_of_total_amount_l1456_145657

-- Conditions
variable (p q r : ℕ)
variable (total_amount amount_r : ℕ)
variable (total_amount_eq : total_amount = 6000)
variable (amount_r_eq : amount_r = 2400)

-- Mathematical statement
theorem fraction_of_total_amount :
  amount_r / total_amount = 2 / 5 :=
by
  -- Sorry to skip the proof, as instructed
  sorry

end fraction_of_total_amount_l1456_145657


namespace max_value_of_expression_l1456_145684

noncomputable def max_expression_value (x y : ℝ) :=
  x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  max_expression_value x y ≤ 961 / 8 :=
sorry

end max_value_of_expression_l1456_145684


namespace sum_of_solutions_eq_zero_l1456_145670

theorem sum_of_solutions_eq_zero (x : ℝ) (h : 6 * x / 30 = 7 / x) :
  (∃ x₁ x₂ : ℝ, x₁^2 = 35 ∧ x₂^2 = 35 ∧ x₁ + x₂ = 0) :=
sorry

end sum_of_solutions_eq_zero_l1456_145670


namespace meaningful_fraction_iff_l1456_145690

theorem meaningful_fraction_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (2 - x)) ↔ x ≠ 2 := by
  sorry

end meaningful_fraction_iff_l1456_145690


namespace solve_for_x_l1456_145663

-- We define that the condition and what we need to prove.
theorem solve_for_x (x : ℝ) : (x + 7) / (x - 4) = (x - 3) / (x + 6) → x = -3 / 2 :=
by sorry

end solve_for_x_l1456_145663


namespace parabola_vertex_l1456_145695

theorem parabola_vertex :
  ∃ (x y : ℝ), (∀ t : ℝ, t^2 + 2 * t - 2 ≥ y) ∧ (x^2 + 2 * x - 2 = y) ∧ (x = -1) ∧ (y = -3) :=
by sorry

end parabola_vertex_l1456_145695


namespace cost_of_double_burger_l1456_145676

-- Definitions based on conditions
def total_cost : ℝ := 64.50
def total_burgers : ℕ := 50
def single_burger_cost : ℝ := 1.00
def double_burgers : ℕ := 29

-- Proof goal
theorem cost_of_double_burger : (total_cost - single_burger_cost * (total_burgers - double_burgers)) / double_burgers = 1.50 :=
by
  sorry

end cost_of_double_burger_l1456_145676


namespace smallest_positive_integer_solution_l1456_145699

theorem smallest_positive_integer_solution : ∃ n : ℕ, 23 * n % 9 = 310 % 9 ∧ n = 8 :=
by
  sorry

end smallest_positive_integer_solution_l1456_145699


namespace range_of_m_l1456_145618

-- Definitions
def is_circle_eqn (d e f : ℝ) : Prop :=
  d^2 + e^2 - 4 * f > 0

-- Main statement 
theorem range_of_m (m : ℝ) : 
  is_circle_eqn (-2) (-4) m → m < 5 :=
by
  intro h
  sorry

end range_of_m_l1456_145618


namespace anne_distance_diff_l1456_145673

def track_length := 300
def min_distance := 100

-- Define distances functions as described
def distance_AB (t : ℝ) : ℝ := sorry  -- Distance function between Anne and Beth over time 
def distance_AC (t : ℝ) : ℝ := sorry  -- Distance function between Anne and Carmen over time 

theorem anne_distance_diff (Anne_speed Beth_speed Carmen_speed : ℝ) 
  (hneA : Anne_speed ≠ Beth_speed)
  (hneC : Anne_speed ≠ Carmen_speed) :
  ∃ α ≥ 0, min_distance ≤ distance_AB α ∧ min_distance ≤ distance_AC α :=
sorry

end anne_distance_diff_l1456_145673


namespace quadratic_solution_l1456_145605

theorem quadratic_solution (x : ℝ) : 2 * x * (x + 1) = 3 * (x + 1) ↔ (x = -1 ∨ x = 3 / 2) := by
  sorry

end quadratic_solution_l1456_145605


namespace complement_union_correct_l1456_145640

open Set

variable (U : Set Int)
variable (A B : Set Int)

theorem complement_union_correct (hU : U = {-2, -1, 0, 1, 2}) (hA : A = {1, 2}) (hB : B = {-2, 1, 2}) :
  A ∪ (U \ B) = {-1, 0, 1, 2} := by
  rw [hU, hA, hB]
  simp
  sorry

end complement_union_correct_l1456_145640


namespace inequality_inequality_l1456_145688

theorem inequality_inequality (a b c d : ℝ) (h1 : a^2 + b^2 = 4) (h2 : c^2 + d^2 = 16) :
  ac + bd ≤ 8 :=
sorry

end inequality_inequality_l1456_145688


namespace sqrt_11_custom_op_l1456_145626

noncomputable def sqrt := Real.sqrt

def custom_op (x y : Real) := (x + y) ^ 2 - (x - y) ^ 2

theorem sqrt_11_custom_op : custom_op (sqrt 11) (sqrt 11) = 44 :=
by
  sorry

end sqrt_11_custom_op_l1456_145626


namespace machine_production_l1456_145642

theorem machine_production
  (rate_per_minute : ℕ)
  (machines_total : ℕ)
  (production_minute : ℕ)
  (machines_sub : ℕ)
  (time_minutes : ℕ)
  (total_production : ℕ) :
  machines_total * rate_per_minute = production_minute →
  rate_per_minute = production_minute / machines_total →
  machines_sub * rate_per_minute = total_production / time_minutes →
  time_minutes * total_production / time_minutes = 900 :=
by
  sorry

end machine_production_l1456_145642


namespace black_squares_covered_by_trominoes_l1456_145697

theorem black_squares_covered_by_trominoes (n : ℕ) (h_odd : n % 2 = 1) :
  (∃ (k : ℕ), k * k = (n + 1) / 2 ∧ n ≥ 7) ↔ n ≥ 7 :=
by
  sorry

end black_squares_covered_by_trominoes_l1456_145697


namespace problem1_problem2_l1456_145633

variable (a b : ℝ)

-- (1) Prove a + b = 2 given the conditions
theorem problem1 (h1 : a > 0) (h2 : b > 0) (h3 : ∀ x : ℝ, abs (x - a) + abs (x + b) ≥ 2) : a + b = 2 :=
sorry

-- (2) Prove it is not possible for both a^2 + a > 2 and b^2 + b > 2 to hold simultaneously
theorem problem2 (h1: a + b = 2) (h2 : a^2 + a > 2) (h3 : b^2 + b > 2) : False :=
sorry

end problem1_problem2_l1456_145633


namespace harvest_rate_l1456_145634

def days := 3
def total_sacks := 24
def sacks_per_day := total_sacks / days

theorem harvest_rate :
  sacks_per_day = 8 :=
by
  sorry

end harvest_rate_l1456_145634


namespace ellipse_intersection_l1456_145613

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem ellipse_intersection (f1 f2 : ℝ × ℝ)
    (h1 : f1 = (0, 5))
    (h2 : f2 = (4, 0))
    (origin_intersection : distance (0, 0) f1 + distance (0, 0) f2 = 5) :
    ∃ x : ℝ, (distance (x, 0) f1 + distance (x, 0) f2 = 5 ∧ x > 0 ∧ x ≠ 0 → x = 28 / 9) :=
by 
  sorry

end ellipse_intersection_l1456_145613


namespace bacteria_doubling_time_l1456_145665

noncomputable def doubling_time_population 
    (initial final : ℝ) 
    (time : ℝ) 
    (growth_factor : ℕ) : ℝ :=
    time / (Real.log growth_factor / Real.log 2)

theorem bacteria_doubling_time :
  doubling_time_population 1000 500000 26.897352853986263 500 = 0.903 :=
by
  sorry

end bacteria_doubling_time_l1456_145665


namespace root_condition_l1456_145680

noncomputable def f (x t : ℝ) := x^2 + t * x - t

theorem root_condition {t : ℝ} : (t ≥ 0 → ∃ x : ℝ, f x t = 0) ∧ (∃ x : ℝ, f x t = 0 → t ≥ 0 ∨ t ≤ -4) := 
  sorry

end root_condition_l1456_145680


namespace min_x2_y2_l1456_145675

theorem min_x2_y2 (x y : ℝ) (h : x * y - x - y = 1) : x^2 + y^2 ≥ 6 - 4 * Real.sqrt 2 :=
by
  sorry

end min_x2_y2_l1456_145675


namespace repeating_decimals_subtraction_l1456_145606

def x : Rat := 1 / 3
def y : Rat := 2 / 99

theorem repeating_decimals_subtraction :
  x - y = 31 / 99 :=
sorry

end repeating_decimals_subtraction_l1456_145606


namespace min_value_of_inverse_sum_l1456_145614

theorem min_value_of_inverse_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) : 
  ∃ c : ℝ, c = 4 + 2 * Real.sqrt 3 ∧ ∀x : ℝ, (x = (1 / a + 1 / b)) → x ≥ c :=
by
  sorry

end min_value_of_inverse_sum_l1456_145614


namespace radio_show_length_l1456_145631

theorem radio_show_length :
  let s3 := 10
  let s2 := s3 + 5
  let s4 := s2 / 2
  let s5 := 2 * s4
  let s1 := 2 * (s2 + s3 + s4 + s5)
  s1 + s2 + s3 + s4 + s5 = 142.5 :=
by
  sorry

end radio_show_length_l1456_145631


namespace actual_revenue_percent_of_projected_l1456_145645

noncomputable def projected_revenue (R : ℝ) : ℝ := 1.2 * R
noncomputable def actual_revenue (R : ℝ) : ℝ := 0.75 * R

theorem actual_revenue_percent_of_projected (R : ℝ) :
  (actual_revenue R / projected_revenue R) * 100 = 62.5 :=
  sorry

end actual_revenue_percent_of_projected_l1456_145645


namespace length_of_square_side_l1456_145656

theorem length_of_square_side 
  (r : ℝ) 
  (A : ℝ) 
  (h : A = 42.06195997410015) 
  (side_length : ℝ := 2 * r)
  (area_of_square : ℝ := side_length ^ 2)
  (segment_area : ℝ := 4 * (π * r * r / 4))
  (enclosed_area: ℝ := area_of_square - segment_area)
  (h2 : enclosed_area = A) :
  side_length = 14 :=
by sorry

end length_of_square_side_l1456_145656


namespace val_need_33_stamps_l1456_145610

def valerie_needs_total_stamps 
    (thank_you_cards : ℕ) 
    (bills_water : ℕ) 
    (bills_electric : ℕ) 
    (bills_internet : ℕ) 
    (rebate_addition : ℕ) 
    (rebate_stamps : ℕ) 
    (job_apps_multiplier : ℕ) 
    (job_app_stamps : ℕ) 
    (total_stamps : ℕ) : Prop :=
    thank_you_cards = 3 ∧
    bills_water = 1 ∧
    bills_electric = 2 ∧
    bills_internet = 3 ∧
    rebate_addition = 3 ∧
    rebate_stamps = 2 ∧
    job_apps_multiplier = 2 ∧
    job_app_stamps = 1 ∧
    total_stamps = 33

theorem val_need_33_stamps : 
  valerie_needs_total_stamps 3 1 2 3 3 2 2 1 33 :=
by 
  -- proof skipped
  sorry

end val_need_33_stamps_l1456_145610


namespace annual_interest_rate_Paul_annual_interest_rate_Emma_annual_interest_rate_Harry_l1456_145693

-- Define principal amounts for Paul, Emma and Harry
def principalPaul : ℚ := 5000
def principalEmma : ℚ := 3000
def principalHarry : ℚ := 7000

-- Define time periods for Paul, Emma and Harry
def timePaul : ℚ := 2
def timeEmma : ℚ := 4
def timeHarry : ℚ := 3

-- Define interests received from Paul, Emma and Harry
def interestPaul : ℚ := 2200
def interestEmma : ℚ := 3400
def interestHarry : ℚ := 3900

-- Define the simple interest formula 
def simpleInterest (P : ℚ) (R : ℚ) (T : ℚ) : ℚ := P * R * T

-- Prove the annual interest rates for each loan 
theorem annual_interest_rate_Paul : 
  ∃ (R : ℚ), simpleInterest principalPaul R timePaul = interestPaul ∧ R = 0.22 := 
by
  sorry

theorem annual_interest_rate_Emma : 
  ∃ (R : ℚ), simpleInterest principalEmma R timeEmma = interestEmma ∧ R = 0.2833 := 
by
  sorry

theorem annual_interest_rate_Harry : 
  ∃ (R : ℚ), simpleInterest principalHarry R timeHarry = interestHarry ∧ R = 0.1857 := 
by
  sorry

end annual_interest_rate_Paul_annual_interest_rate_Emma_annual_interest_rate_Harry_l1456_145693


namespace complement_union_covers_until_1_l1456_145607

open Set

noncomputable def S := {x : ℝ | x > -2}
noncomputable def T := {x : ℝ | x^2 + 3*x - 4 ≤ 0}
noncomputable def complement_R_S := {x : ℝ | x ≤ -2}
noncomputable def union := complement_R_S ∪ T

theorem complement_union_covers_until_1 : union = {x : ℝ | x ≤ 1} := by
  sorry

end complement_union_covers_until_1_l1456_145607


namespace gcd_40_120_45_l1456_145625

theorem gcd_40_120_45 : Nat.gcd (Nat.gcd 40 120) 45 = 5 :=
by
  sorry

end gcd_40_120_45_l1456_145625


namespace sum_of_positive_factors_of_90_eq_234_l1456_145628

theorem sum_of_positive_factors_of_90_eq_234 : 
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  List.sum factors = 234 :=
by
  -- List the positive factors of 90
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  -- Prove that the sum of these factors is 234
  have h_sum_factors : List.sum factors = 234 := sorry
  exact h_sum_factors

end sum_of_positive_factors_of_90_eq_234_l1456_145628


namespace solve_abs_inequality_l1456_145667

theorem solve_abs_inequality (x : ℝ) (h : x ≠ 1) : 
  abs ((3 * x - 2) / (x - 1)) > 3 ↔ (5 / 6 < x ∧ x < 1) ∨ (x > 1) := 
by 
  sorry

end solve_abs_inequality_l1456_145667


namespace lucy_apples_per_week_l1456_145671

-- Define the conditions
def chandler_apples_per_week := 23
def total_apples_per_month := 168
def weeks_per_month := 4
def chandler_apples_per_month := chandler_apples_per_week * weeks_per_month
def lucy_apples_per_month := total_apples_per_month - chandler_apples_per_month

-- Define the proof problem statement
theorem lucy_apples_per_week :
  lucy_apples_per_month / weeks_per_month = 19 :=
  by sorry

end lucy_apples_per_week_l1456_145671


namespace cost_price_of_toy_l1456_145678

theorem cost_price_of_toy 
  (cost_price : ℝ)
  (SP : ℝ := 120000)
  (num_toys : ℕ := 40)
  (profit_per_toy : ℝ := 500)
  (gain_per_toy : ℝ := cost_price + profit_per_toy)
  (total_gain : ℝ := 8 * cost_price + profit_per_toy * num_toys)
  (total_cost_price : ℝ := num_toys * cost_price)
  (SP_eq_cost_plus_gain : SP = total_cost_price + total_gain) :
  cost_price = 2083.33 :=
by
  sorry

end cost_price_of_toy_l1456_145678


namespace distance_light_travels_500_years_l1456_145689

-- Define the given conditions
def distance_in_one_year_miles : ℝ := 5.87e12
def years_traveling : ℝ := 500
def miles_to_kilometers : ℝ := 1.60934

-- Define the expected distance in kilometers after 500 years
def expected_distance_in_kilometers : ℝ  := 4.723e15

-- State the theorem: the distance light travels in 500 years in kilometers
theorem distance_light_travels_500_years :
  (distance_in_one_year_miles * years_traveling * miles_to_kilometers) 
    = expected_distance_in_kilometers := 
by
  sorry

end distance_light_travels_500_years_l1456_145689


namespace cubic_roots_natural_numbers_l1456_145691

theorem cubic_roots_natural_numbers (p : ℝ) :
  (∃ x1 x2 x3 : ℕ, (5 * (x1 : ℝ)^3 - 5 * (p + 1) * (x1 : ℝ)^2 + (71 * p - 1) * (x1 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x2 : ℝ)^3 - 5 * (p + 1) * (x2 : ℝ)^2 + (71 * p - 1) * (x2 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x3 : ℝ)^3 - 5 * (p + 1) * (x3 : ℝ)^2 + (71 * p - 1) * (x3 : ℝ) + 1 = 66 * p)) →
  p = 76 :=
sorry

end cubic_roots_natural_numbers_l1456_145691


namespace manager_salary_calculation_l1456_145623

theorem manager_salary_calculation :
  let percent_marketers := 0.60
  let salary_marketers := 50000
  let percent_engineers := 0.20
  let salary_engineers := 80000
  let percent_sales_reps := 0.10
  let salary_sales_reps := 70000
  let percent_managers := 0.10
  let total_average_salary := 75000
  let total_contribution := percent_marketers * salary_marketers + percent_engineers * salary_engineers + percent_sales_reps * salary_sales_reps
  let managers_total_contribution := total_average_salary - total_contribution
  let manager_salary := managers_total_contribution / percent_managers
  manager_salary = 220000 :=
by
  sorry

end manager_salary_calculation_l1456_145623


namespace point_above_line_l1456_145601

theorem point_above_line (t : ℝ) : (∃ y : ℝ, y = (2 : ℝ)/3) → (t > (2 : ℝ)/3) :=
  by
  intro h
  sorry

end point_above_line_l1456_145601


namespace find_x_y_sum_l1456_145635

def is_perfect_square (n : ℕ) : Prop := ∃ (k : ℕ), k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

theorem find_x_y_sum (n x y : ℕ) (hn : n = 450) (hx : x > 0) (hy : y > 0)
  (hxsq : is_perfect_square (n * x))
  (hycube : is_perfect_cube (n * y)) :
  x + y = 62 :=
  sorry

end find_x_y_sum_l1456_145635


namespace no_maximum_y_coordinate_for_hyperbola_l1456_145668

theorem no_maximum_y_coordinate_for_hyperbola :
  ∀ y : ℝ, ∃ x : ℝ, y = 3 + (3 / 5) * x :=
by
  sorry

end no_maximum_y_coordinate_for_hyperbola_l1456_145668


namespace ellen_legos_final_count_l1456_145650

-- Definitions based on conditions
def initial_legos : ℕ := 380
def lost_legos_first_week : ℕ := 57
def additional_legos_second_week (remaining_legos : ℕ) : ℕ := 32
def borrowed_legos_third_week (total_legos : ℕ) : ℕ := 88

-- Computed values based on conditions
def legos_after_first_week (initial : ℕ) (lost : ℕ) : ℕ := initial - lost
def legos_after_second_week (remaining : ℕ) (additional : ℕ) : ℕ := remaining + additional
def legos_after_third_week (total : ℕ) (borrowed : ℕ) : ℕ := total - borrowed

-- Proof statement
theorem ellen_legos_final_count : 
  legos_after_third_week 
    (legos_after_second_week 
      (legos_after_first_week initial_legos lost_legos_first_week)
      (additional_legos_second_week (legos_after_first_week initial_legos lost_legos_first_week)))
    (borrowed_legos_third_week (legos_after_second_week 
                                  (legos_after_first_week initial_legos lost_legos_first_week)
                                  (additional_legos_second_week (legos_after_first_week initial_legos lost_legos_first_week)))) 
  = 267 :=
by 
  sorry

end ellen_legos_final_count_l1456_145650
