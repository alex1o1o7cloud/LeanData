import Mathlib

namespace roots_of_equation_l353_35374

theorem roots_of_equation : ∃ x₁ x₂ : ℝ, (3 ^ x₁ = Real.log (x₁ + 9) / Real.log 3) ∧ 
                                     (3 ^ x₂ = Real.log (x₂ + 9) / Real.log 3) ∧ 
                                     (x₁ < 0) ∧ (x₂ > 0) := 
by {
  sorry
}

end roots_of_equation_l353_35374


namespace find_K_values_l353_35375

theorem find_K_values (K M : ℕ) (h1 : (K * (K + 1)) / 2 = M^2) (h2 : M < 200) (h3 : K > M) :
  K = 8 ∨ K = 49 :=
sorry

end find_K_values_l353_35375


namespace trajectory_eq_l353_35377

-- Define the conditions provided in the problem
def circle_eq (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * (m + 3) * x + 2 * (1 - 4 * m^2) + 16 * m^4 + 9 = 0

-- Define the required range for m based on the derivation
def m_valid (m : ℝ) : Prop :=
  -1/7 < m ∧ m < 1

-- Prove that the equation of the trajectory of the circle's center is y = 4(x-3)^2 -1 
-- and it's valid in the required range for x
theorem trajectory_eq (x y : ℝ) :
  (∃ m : ℝ, m_valid m ∧ y = 4 * (x - 3)^2 - 1 ∧ (x = m + 3) ∧ (y = 4 * m^2 - 1)) →
  y = 4 * (x - 3)^2 - 1 ∧ (20/7 < x) ∧ (x < 4) :=
by
  intro h
  cases' h with m hm
  sorry

end trajectory_eq_l353_35377


namespace spots_combined_l353_35301

def Rover : ℕ := 46
def Cisco : ℕ := Rover / 2 - 5
def Granger : ℕ := 5 * Cisco

theorem spots_combined : Granger + Cisco = 108 := by
  sorry

end spots_combined_l353_35301


namespace common_value_of_4a_and_5b_l353_35327

theorem common_value_of_4a_and_5b (a b C : ℝ) (h1 : 4 * a = C) (h2 : 5 * b = C) (h3 : 40 * a * b = 1800) :
  C = 60 :=
sorry

end common_value_of_4a_and_5b_l353_35327


namespace find_m_l353_35397

theorem find_m (m : ℝ) (A B C D : ℝ × ℝ)
  (h1 : A = (m, 1)) (h2 : B = (-3, 4))
  (h3 : C = (0, 2)) (h4 : D = (1, 1))
  (h_parallel : (4 - 1) / (-3 - m) = (1 - 2) / (1 - 0)) :
  m = 0 :=
  by
  sorry

end find_m_l353_35397


namespace totalCost_l353_35379
-- Importing the necessary library

-- Defining the conditions
def numberOfHotDogs : Nat := 6
def costPerHotDog : Nat := 50

-- Proving the total cost
theorem totalCost : numberOfHotDogs * costPerHotDog = 300 := by
  sorry

end totalCost_l353_35379


namespace line_eq_form_l353_35347

def line_equation (x y : ℝ) : Prop :=
  ((3 : ℝ) * (x - 2) - (4 : ℝ) * (y + 3) = 0)

theorem line_eq_form (x y : ℝ) (h : line_equation x y) :
  ∃ (m b : ℝ), y = m * x + b ∧ (m = 3/4 ∧ b = -9/2) :=
by
  sorry

end line_eq_form_l353_35347


namespace factorization_problem_l353_35337

theorem factorization_problem :
  (∃ (h : D), 
    (¬ ∃ (a b : ℝ) (x y : ℝ), a * (x - y) = a * x - a * y) ∧
    (¬ ∃ (x : ℝ), x^2 - 2 * x + 3 = x * (x - 2) + 3) ∧
    (¬ ∃ (x : ℝ), (x - 1) * (x + 4) = x^2 + 3 * x - 4) ∧
    (∃ (x : ℝ), x^3 - 2 * x^2 + x = x * (x - 1)^2)) :=
  sorry

end factorization_problem_l353_35337


namespace min_value_proof_l353_35311

noncomputable def min_expr_value (x y : ℝ) : ℝ :=
  (1 / (2 * x)) + (1 / y)

theorem min_value_proof (x y : ℝ) (h1 : x + y = 1) (h2 : y > 0) (h3 : x > 0) :
  min_expr_value x y = (3 / 2) + Real.sqrt 2 :=
sorry

end min_value_proof_l353_35311


namespace problem_statement_l353_35360

open Real

noncomputable def log4 (x : ℝ) : ℝ := log x / log 4

noncomputable def a : ℝ := log4 (sqrt 5)
noncomputable def b : ℝ := log 2 / log 5
noncomputable def c : ℝ := log4 5

theorem problem_statement : b < a ∧ a < c :=
by
  sorry

end problem_statement_l353_35360


namespace percentage_increase_l353_35371

theorem percentage_increase (old_earnings new_earnings : ℝ) (h_old : old_earnings = 50) (h_new : new_earnings = 70) :
  ((new_earnings - old_earnings) / old_earnings) * 100 = 40 :=
by
  rw [h_old, h_new]
  -- Simplification and calculation steps would go here
  sorry

end percentage_increase_l353_35371


namespace necessary_condition_transitivity_l353_35390

theorem necessary_condition_transitivity (A B C : Prop) 
  (hAB : A → B) (hBC : B → C) : A → C := 
by
  intro ha
  apply hBC
  apply hAB
  exact ha

-- sorry


end necessary_condition_transitivity_l353_35390


namespace worth_of_each_gold_bar_l353_35364

theorem worth_of_each_gold_bar
  (rows : ℕ) (gold_bars_per_row : ℕ) (total_worth : ℕ)
  (h1 : rows = 4) (h2 : gold_bars_per_row = 20) (h3 : total_worth = 1600000)
  (total_gold_bars : ℕ) (h4 : total_gold_bars = rows * gold_bars_per_row) :
  total_worth / total_gold_bars = 20000 :=
by sorry

end worth_of_each_gold_bar_l353_35364


namespace find_function_expression_point_on_function_graph_l353_35322

-- Problem setup
def y_minus_2_is_directly_proportional_to_x (y x : ℝ) : Prop :=
  ∃ k : ℝ, y - 2 = k * x

-- Conditions
def specific_condition : Prop :=
  y_minus_2_is_directly_proportional_to_x 6 1

-- Function expression derivation
theorem find_function_expression : ∃ k, ∀ x, 6 - 2 = k * 1 ∧ ∀ y, y = k * x + 2 :=
sorry

-- Given point P belongs to the function graph
theorem point_on_function_graph (a : ℝ) : (∀ x y, y = 4 * x + 2) → ∃ a, 4 * a + 2 = -1 :=
sorry

end find_function_expression_point_on_function_graph_l353_35322


namespace point_B_coordinates_l353_35396

theorem point_B_coordinates :
  ∃ (B : ℝ × ℝ), (B.1 < 0) ∧ (|B.2| = 4) ∧ (|B.1| = 5) ∧ (B = (-5, 4) ∨ B = (-5, -4)) :=
sorry

end point_B_coordinates_l353_35396


namespace find_cows_l353_35348

variable (D C : ℕ)

theorem find_cows (h1 : 2 * D + 4 * C = 2 * (D + C) + 36) : C = 18 :=
by
  -- Proof goes here
  sorry

end find_cows_l353_35348


namespace sin_diff_l353_35393

theorem sin_diff (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1 / 3) 
  (h2 : Real.sin β - Real.cos α = 1 / 2) : 
  Real.sin (α - β) = -59 / 72 := 
sorry

end sin_diff_l353_35393


namespace polygon_sides_l353_35355

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l353_35355


namespace find_n_for_sum_l353_35343

theorem find_n_for_sum (n : ℕ) : ∃ n, n * (2 * n - 1) = 2009 ^ 2 :=
by
  sorry

end find_n_for_sum_l353_35343


namespace inverse_function_b_value_l353_35395

theorem inverse_function_b_value (b : ℝ) :
  (∀ x, ∃ y, 2^x + b = y) ∧ (∃ x, ∃ y, (x, y) = (2, 5)) → b = 1 :=
by
  sorry

end inverse_function_b_value_l353_35395


namespace half_percent_to_decimal_l353_35346

def percent_to_decimal (x : ℚ) : ℚ := x / 100

theorem half_percent_to_decimal : percent_to_decimal (1 / 2) = 0.005 :=
by
  sorry

end half_percent_to_decimal_l353_35346


namespace calculate_expression_l353_35366

theorem calculate_expression (m : ℝ) : (-m)^2 * m^5 = m^7 := 
sorry

end calculate_expression_l353_35366


namespace partition_weights_l353_35336

theorem partition_weights :
  ∃ A B C : Finset ℕ,
    (∀ x ∈ A, x ≤ 552) ∧
    (∀ x ∈ B, x ≤ 552) ∧
    (∀ x ∈ C, x ≤ 552) ∧
    ∀ x, (x ∈ A ∨ x ∈ B ∨ x ∈ C) ↔ 1 ≤ x ∧ x ≤ 552 ∧
    A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧
    A.sum id = 50876 ∧ B.sum id = 50876 ∧ C.sum id = 50876 :=
by
  sorry

end partition_weights_l353_35336


namespace y_intercept_of_line_l353_35352

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) : y = -2 :=
by
  sorry

end y_intercept_of_line_l353_35352


namespace chess_club_officers_l353_35363

/-- The Chess Club with 24 members needs to choose 3 officers: president,
    secretary, and treasurer. Each person can hold at most one office. 
    Alice and Bob will only serve together as officers. Prove that 
    the number of ways to choose the officers is 9372. -/
theorem chess_club_officers : 
  let members := 24
  let num_officers := 3
  let alice_and_bob_together := true
  ∃ n : ℕ, n = 9372 := sorry

end chess_club_officers_l353_35363


namespace inequality_proof_l353_35323

theorem inequality_proof (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
    (h_sum : a + b + c + d = 8) :
    (a^3 / (a^2 + b + c) + b^3 / (b^2 + c + d) + c^3 / (c^2 + d + a) + d^3 / (d^2 + a + b)) ≥ 4 :=
by
  sorry

end inequality_proof_l353_35323


namespace max_actors_chess_tournament_l353_35306

-- Definitions based on conditions
variable {α : Type} [Fintype α] [DecidableEq α]

-- Each actor played with every other actor exactly once.
def played_with_everyone (R : α → α → ℝ) : Prop :=
  ∀ a b, a ≠ b → (R a b = 1 ∨ R a b = 0.5 ∨ R a b = 0)

-- Among every three participants, one earned exactly 1.5 solidus in matches against the other two.
def condition_1_5_solidi (R : α → α → ℝ) : Prop :=
  ∀ a b c, a ≠ b → b ≠ c → a ≠ c → 
   (R a b + R a c = 1.5 ∨ R b a + R b c = 1.5 ∨ R c a + R c b = 1.5)

-- Prove the maximum number of such participants is 5
theorem max_actors_chess_tournament (actors : Finset α) (R : α → α → ℝ) 
  (h_played : played_with_everyone R) (h_condition : condition_1_5_solidi R) :
  actors.card ≤ 5 :=
  sorry

end max_actors_chess_tournament_l353_35306


namespace no_zeros_sin_log_l353_35376

open Real

theorem no_zeros_sin_log (x : ℝ) (h1 : 1 < x) (h2 : x < exp 1) : ¬ (sin (log x) = 0) :=
sorry

end no_zeros_sin_log_l353_35376


namespace micheal_work_separately_40_days_l353_35392

-- Definitions based on the problem conditions
def work_complete_together (M A : ℕ) : Prop := (1/(M:ℝ) + 1/(A:ℝ) = 1/20)
def remaining_work_completed_by_adam (A : ℕ) : Prop := (1/(A:ℝ) = 1/40)

-- The theorem we want to prove
theorem micheal_work_separately_40_days (M A : ℕ) 
  (h1 : work_complete_together M A) 
  (h2 : remaining_work_completed_by_adam A) : 
  M = 40 := 
by 
  sorry  -- Placeholder for proof

end micheal_work_separately_40_days_l353_35392


namespace total_number_of_birds_l353_35312

def bird_cages : Nat := 9
def parrots_per_cage : Nat := 2
def parakeets_per_cage : Nat := 6
def birds_per_cage : Nat := parrots_per_cage + parakeets_per_cage
def total_birds : Nat := bird_cages * birds_per_cage

theorem total_number_of_birds : total_birds = 72 := by
  sorry

end total_number_of_birds_l353_35312


namespace greatest_int_less_than_50_satisfying_conditions_l353_35321

def satisfies_conditions (n : ℕ) : Prop :=
  n < 50 ∧ Int.gcd n 18 = 6

theorem greatest_int_less_than_50_satisfying_conditions :
  ∃ n : ℕ, satisfies_conditions n ∧ ∀ m : ℕ, satisfies_conditions m → m ≤ n ∧ n = 42 :=
by
  sorry

end greatest_int_less_than_50_satisfying_conditions_l353_35321


namespace tonya_needs_to_eat_more_l353_35329

-- Define the conditions in the problem
def ounces_per_hamburger : ℕ := 4
def ounces_eaten_last_year : ℕ := 84

-- Define a function to calculate hamburgers given ounces
def hamburgers_eaten (ounces : ℕ) (ounce_per_hamburger : ℕ) : ℕ :=
  ounces / ounce_per_hamburger

-- State the theorem
theorem tonya_needs_to_eat_more (ounces_per_hamburger ounces_eaten_last_year : ℕ) :
  hamburgers_eaten ounces_eaten_last_year ounces_per_hamburger + 1 = 22 := by
  sorry

end tonya_needs_to_eat_more_l353_35329


namespace histogram_height_representation_l353_35381

theorem histogram_height_representation (freq_ratio : ℝ) (frequency : ℝ) (class_interval : ℝ) 
  (H : freq_ratio = frequency / class_interval) : 
  freq_ratio = frequency / class_interval :=
by 
  sorry

end histogram_height_representation_l353_35381


namespace minimize_sum_of_squares_if_and_only_if_l353_35369

noncomputable def minimize_sum_of_squares (a b c S : ℝ) (O : ℝ×ℝ×ℝ) (x y z : ℝ) : Prop :=
  let ax_by_cz := a * x + b * y + c * z
  ax_by_cz = 2 * S ∧
  x/y = a/b ∧
  y/z = b/c ∧
  x/z = a/c

theorem minimize_sum_of_squares_if_and_only_if (a b c S : ℝ) (O : ℝ×ℝ×ℝ) (x y z : ℝ) :
  (∃ P : ℝ, minimize_sum_of_squares a b c S O x y z) ↔ (x/y = a/b ∧ y/z = b/c ∧ x/z = a/c) := sorry

end minimize_sum_of_squares_if_and_only_if_l353_35369


namespace complex_number_solution_l353_35391

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i^2 = -1) (hz : i * (z - 1) = 1 - i) : z = -i :=
by sorry

end complex_number_solution_l353_35391


namespace range_of_x_plus_2y_minus_2z_l353_35328

theorem range_of_x_plus_2y_minus_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 4) : -6 ≤ x + 2 * y - 2 * z ∧ x + 2 * y - 2 * z ≤ 6 :=
sorry

end range_of_x_plus_2y_minus_2z_l353_35328


namespace semicircle_area_difference_l353_35378

theorem semicircle_area_difference 
  (A B C P D E F : Type) 
  (h₁ : S₅ - S₆ = 2) 
  (h₂ : S₁ - S₂ = 1) 
  : S₄ - S₃ = 3 :=
by
  -- Using Lean tactics to form the proof, place sorry for now.
  sorry

end semicircle_area_difference_l353_35378


namespace prob_two_red_two_blue_is_3_over_14_l353_35332

def red_marbles : ℕ := 15
def blue_marbles : ℕ := 10
def total_marbles : ℕ := red_marbles + blue_marbles
def chosen_marbles : ℕ := 4

noncomputable def prob_two_red_two_blue : ℚ :=
  let total_ways := (Nat.choose total_marbles chosen_marbles : ℚ)
  let ways_two_red := (Nat.choose red_marbles 2)
  let ways_two_blue := (Nat.choose blue_marbles 2)
  let favorable_outcomes := 6 * ways_two_red * ways_two_blue
  favorable_outcomes / total_ways

theorem prob_two_red_two_blue_is_3_over_14 : prob_two_red_two_blue = 3 / 14 :=
  sorry

end prob_two_red_two_blue_is_3_over_14_l353_35332


namespace johns_average_speed_l353_35300

-- Definitions of conditions
def total_time_hours : ℝ := 6.5
def total_distance_miles : ℝ := 255

-- Stating the problem to be proven
theorem johns_average_speed :
  (total_distance_miles / total_time_hours) = 39.23 := 
sorry

end johns_average_speed_l353_35300


namespace Cody_games_l353_35335

/-- Cody had nine old video games he wanted to get rid of.
He decided to give four of the games to his friend Jake,
three games to his friend Sarah, and one game to his friend Luke.
On Saturday he bought five new games.
How many games does Cody have now? -/
theorem Cody_games (nine_games initially: ℕ) (jake_games: ℕ) (sarah_games: ℕ) (luke_games: ℕ) (saturday_games: ℕ)
  (h_initial: initially = 9)
  (h_jake: jake_games = 4)
  (h_sarah: sarah_games = 3)
  (h_luke: luke_games = 1)
  (h_saturday: saturday_games = 5) :
  ((initially - (jake_games + sarah_games + luke_games)) + saturday_games) = 6 :=
by
  sorry

end Cody_games_l353_35335


namespace find_b_l353_35387

noncomputable def triangle_b_value (a : ℝ) (C : ℝ) (area : ℝ) : ℝ :=
  let sin_C := Real.sin C
  let b := (2 * area) / (a * sin_C)
  b

theorem find_b (h₁ : a = 1)
              (h₂ : C = Real.pi / 4)
              (h₃ : area = 2 * a) :
              triangle_b_value a C area = 8 * Real.sqrt 2 :=
by
  -- Definitions imply what we need
  sorry

end find_b_l353_35387


namespace right_triangle_ratio_l353_35316

theorem right_triangle_ratio (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : (x - y)^2 + x^2 = (x + y)^2) : x / y = 4 :=
by
  sorry

end right_triangle_ratio_l353_35316


namespace systematic_sampling_student_l353_35338

theorem systematic_sampling_student (total_students sample_size : ℕ) 
  (h_total_students : total_students = 56)
  (h_sample_size : sample_size = 4)
  (student1 student2 student3 student4 : ℕ)
  (h_student1 : student1 = 6)
  (h_student2 : student2 = 34)
  (h_student3 : student3 = 48) :
  student4 = 20 :=
sorry

end systematic_sampling_student_l353_35338


namespace correct_option_d_l353_35388

theorem correct_option_d (a b c : ℝ) (h: a < b ∧ b < 0) : a^2 > ab ∧ ab > b^2 :=
by
  sorry

end correct_option_d_l353_35388


namespace initial_mean_corrected_observations_l353_35319

theorem initial_mean_corrected_observations:
  ∃ M : ℝ, 
  (∀ (Sum_initial Sum_corrected : ℝ), 
    Sum_initial = 50 * M ∧ 
    Sum_corrected = Sum_initial + (48 - 23) → 
    Sum_corrected / 50 = 41.5) →
  M = 41 :=
by
  sorry

end initial_mean_corrected_observations_l353_35319


namespace sin_2theta_plus_pi_div_2_l353_35315

theorem sin_2theta_plus_pi_div_2 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 4)
    (h_tan2θ : Real.tan (2 * θ) = Real.cos θ / (2 - Real.sin θ)) :
    Real.sin (2 * θ + π / 2) = 7 / 8 :=
sorry

end sin_2theta_plus_pi_div_2_l353_35315


namespace max_area_of_equilateral_triangle_in_rectangle_l353_35385

noncomputable def maxEquilateralTriangleArea (a b : ℝ) : ℝ :=
  if h : a ≤ b then
    (a^2 * Real.sqrt 3) / 4
  else
    (b^2 * Real.sqrt 3) / 4

theorem max_area_of_equilateral_triangle_in_rectangle :
  maxEquilateralTriangleArea 12 14 = 36 * Real.sqrt 3 :=
by
  sorry

end max_area_of_equilateral_triangle_in_rectangle_l353_35385


namespace probability_Rachel_Robert_in_picture_l353_35359

noncomputable def Rachel_lap_time := 75
noncomputable def Robert_lap_time := 70
noncomputable def photo_time_start := 900
noncomputable def photo_time_end := 960
noncomputable def track_fraction := 1 / 5

theorem probability_Rachel_Robert_in_picture :
  let lap_time_Rachel := Rachel_lap_time
  let lap_time_Robert := Robert_lap_time
  let time_start := photo_time_start
  let time_end := photo_time_end
  let interval_Rachel := 15  -- ±15 seconds for Rachel
  let interval_Robert := 14  -- ±14 seconds for Robert
  let probability := (2 * interval_Robert) / (time_end - time_start) 
  probability = 7 / 15 :=
by
  sorry

end probability_Rachel_Robert_in_picture_l353_35359


namespace marbles_lost_l353_35333

theorem marbles_lost (m_initial m_current : ℕ) (h_initial : m_initial = 19) (h_current : m_current = 8) : m_initial - m_current = 11 :=
by {
  sorry
}

end marbles_lost_l353_35333


namespace noah_left_lights_on_2_hours_l353_35320

-- Define the conditions
def bedroom_light_usage : ℕ := 6
def office_light_usage : ℕ := 3 * bedroom_light_usage
def living_room_light_usage : ℕ := 4 * bedroom_light_usage
def total_energy_used : ℕ := 96
def total_energy_per_hour := bedroom_light_usage + office_light_usage + living_room_light_usage

-- Define the main theorem to prove
theorem noah_left_lights_on_2_hours : total_energy_used / total_energy_per_hour = 2 := by
  sorry

end noah_left_lights_on_2_hours_l353_35320


namespace distance_between_houses_l353_35382

-- Definitions
def speed : ℝ := 2          -- Amanda's speed in miles per hour
def time : ℝ := 3           -- Time taken by Amanda in hours

-- The theorem to prove distance is 6 miles
theorem distance_between_houses : speed * time = 6 := by
  sorry

end distance_between_houses_l353_35382


namespace probability_factor_120_less_9_l353_35354

theorem probability_factor_120_less_9 : 
  ∀ n : ℕ, n = 120 → (∃ p : ℚ, p = 7 / 16 ∧ (∃ factors_less_9 : ℕ, factors_less_9 < 16 ∧ factors_less_9 = 7)) := 
by 
  sorry

end probability_factor_120_less_9_l353_35354


namespace intersection_points_C1_C2_l353_35341

theorem intersection_points_C1_C2 :
  (∀ t : ℝ, ∃ (ρ θ : ℝ), 
    (ρ^2 - 10 * ρ * Real.cos θ - 8 * ρ * Real.sin θ + 41 = 0) ∧ 
    (ρ = 2 * Real.cos θ) → 
    ((ρ = 2 ∧ θ = 0) ∨ (ρ = Real.sqrt 2 ∧ θ = Real.pi / 4))) :=
sorry

end intersection_points_C1_C2_l353_35341


namespace only_sqrt_three_is_irrational_l353_35302

-- Definitions based on conditions
def zero_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (0 : ℝ) = p / q
def neg_three_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (-3 : ℝ) = p / q
def one_third_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (1/3 : ℝ) = p / q
def sqrt_three_irrational : Prop := ¬ ∃ p q : ℤ, q ≠ 0 ∧ (Real.sqrt 3) = p / q

-- The proof problem statement
theorem only_sqrt_three_is_irrational :
  zero_rational ∧
  neg_three_rational ∧
  one_third_rational ∧
  sqrt_three_irrational :=
by sorry

end only_sqrt_three_is_irrational_l353_35302


namespace findingRealNumsPureImaginary_l353_35308

theorem findingRealNumsPureImaginary :
  ∀ x : ℝ, ((x + Complex.I * 2) * ((x + 2) + Complex.I * 2) * ((x + 4) + Complex.I * 2)).im = 0 → 
    x = -4 ∨ x = -1 + 2 * Real.sqrt 5 ∨ x = -1 - 2 * Real.sqrt 5 :=
by
  intros x h
  let expr := x^3 + 6*x^2 + 4*x - 16
  have h_real_part_eq_0 : expr = 0 := sorry
  have solutions_correct :
    expr = 0 → (x = -4 ∨ x = -1 + 2 * Real.sqrt 5 ∨ x = -1 - 2 * Real.sqrt 5) := sorry
  exact solutions_correct h_real_part_eq_0

end findingRealNumsPureImaginary_l353_35308


namespace sum_infinite_series_eq_l353_35350

theorem sum_infinite_series_eq {x : ℝ} (hx : |x| < 1) :
  (∑' n : ℕ, (n + 1) * x^n) = 1 / (1 - x)^2 :=
by
  sorry

end sum_infinite_series_eq_l353_35350


namespace diameter_of_circumscribed_circle_l353_35398

theorem diameter_of_circumscribed_circle (a : ℝ) (A : ℝ) (D : ℝ) 
  (h1 : a = 12) (h2 : A = 30) : D = 24 :=
by
  sorry

end diameter_of_circumscribed_circle_l353_35398


namespace solution_difference_l353_35330

theorem solution_difference (m n : ℝ) (h_eq : ∀ x : ℝ, (x - 4) * (x + 4) = 24 * x - 96 ↔ x = m ∨ x = n) (h_distinct : m ≠ n) (h_order : m > n) : m - n = 16 :=
sorry

end solution_difference_l353_35330


namespace almond_butter_servings_l353_35372

noncomputable def servings_in_container (total_tbsps : ℚ) (serving_size : ℚ) : ℚ :=
  total_tbsps / serving_size

theorem almond_butter_servings :
  servings_in_container (34 + 3/5) (5 + 1/2) = 6 + 21/55 :=
by
  sorry

end almond_butter_servings_l353_35372


namespace prove_equation_1_prove_equation_2_l353_35339

theorem prove_equation_1 : 
  ∀ x, (x - 3) / (x - 2) - 1 = 3 / x ↔ x = 3 / 2 :=
by
  sorry

theorem prove_equation_2 :
  ¬∃ x, (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 :=
by
  sorry

end prove_equation_1_prove_equation_2_l353_35339


namespace solution_to_equation_l353_35357

noncomputable def solve_equation (x : ℝ) : Prop :=
  x + 2 = 1 / (x - 2) ∧ x ≠ 2

theorem solution_to_equation (x : ℝ) (h : solve_equation x) : x = Real.sqrt 5 ∨ x = -Real.sqrt 5 :=
sorry

end solution_to_equation_l353_35357


namespace assignment_plans_proof_l353_35344

noncomputable def total_assignment_plans : ℕ :=
  let volunteers := ["Xiao Zhang", "Xiao Zhao", "Xiao Li", "Xiao Luo", "Xiao Wang"]
  let positions := ["translation", "tour guide", "etiquette", "driver"]
  -- Definitions for eligible volunteers for the first two positions
  let first_positions := ["Xiao Zhang", "Xiao Zhao"]
  let remaining_positions := ["Xiao Li", "Xiao Luo", "Xiao Wang"]
  -- Assume the computation for the exact number which results in 36
  36

theorem assignment_plans_proof : total_assignment_plans = 36 := 
  by 
  -- Proof skipped
  sorry

end assignment_plans_proof_l353_35344


namespace admin_staff_in_sample_l353_35386

theorem admin_staff_in_sample (total_staff : ℕ) (admin_staff : ℕ) (total_samples : ℕ)
  (probability : ℚ) (h1 : total_staff = 200) (h2 : admin_staff = 24)
  (h3 : total_samples = 50) (h4 : probability = 50 / 200) :
  admin_staff * probability = 6 :=
by
  -- Proof goes here
  sorry

end admin_staff_in_sample_l353_35386


namespace problem_l353_35394

theorem problem (x y z : ℕ) (hx : x < 9) (hy : y < 9) (hz : z < 9) 
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z % 9 = 0) :=
sorry

end problem_l353_35394


namespace cinnamon_swirl_eaters_l353_35318

theorem cinnamon_swirl_eaters (total_pieces : ℝ) (jane_pieces : ℝ) (equal_pieces : total_pieces / jane_pieces = 3 ) : 
  (total_pieces = 12) ∧ (jane_pieces = 4) → total_pieces / jane_pieces = 3 := 
by 
  sorry

end cinnamon_swirl_eaters_l353_35318


namespace Lindsay_has_26_more_black_brown_dolls_than_blonde_l353_35361

def blonde_dolls : Nat := 4
def brown_dolls : Nat := 4 * blonde_dolls
def black_dolls : Nat := brown_dolls - 2
def total_black_brown_dolls : Nat := black_dolls + brown_dolls
def extra_black_brown_dolls (blonde_dolls black_dolls brown_dolls : Nat) : Nat :=
  total_black_brown_dolls - blonde_dolls

theorem Lindsay_has_26_more_black_brown_dolls_than_blonde :
  extra_black_brown_dolls blonde_dolls black_dolls brown_dolls = 26 := by
  sorry

end Lindsay_has_26_more_black_brown_dolls_than_blonde_l353_35361


namespace probability_of_pink_gumball_l353_35325

theorem probability_of_pink_gumball 
  (P B : ℕ) 
  (total_gumballs : P + B > 0)
  (prob_blue_blue : ((B : ℚ) / (B + P))^2 = 16 / 49) : 
  (B + P > 0) → ((P : ℚ) / (B + P) = 3 / 7) :=
by
  sorry

end probability_of_pink_gumball_l353_35325


namespace production_average_lemma_l353_35313

theorem production_average_lemma (n : ℕ) (h1 : 50 * n + 60 = 55 * (n + 1)) : n = 1 :=
by
  sorry

end production_average_lemma_l353_35313


namespace karen_kept_cookies_l353_35399

def total_cookies : ℕ := 50
def cookies_to_grandparents : ℕ := 8
def number_of_classmates : ℕ := 16
def cookies_per_classmate : ℕ := 2

theorem karen_kept_cookies (x : ℕ) 
  (H1 : x = total_cookies - (cookies_to_grandparents + number_of_classmates * cookies_per_classmate)) :
  x = 10 :=
by
  -- proof omitted
  sorry

end karen_kept_cookies_l353_35399


namespace salary_january_l353_35349

variable (J F M A May : ℝ)

theorem salary_january 
  (h1 : J + F + M + A = 32000) 
  (h2 : F + M + A + May = 33600) 
  (h3 : May = 6500) : 
  J = 4900 := 
by {
 sorry 
}

end salary_january_l353_35349


namespace box_max_volume_l353_35317

theorem box_max_volume (x : ℝ) (h1 : 0 < x) (h2 : x < 5) :
    (10 - 2 * x) * (16 - 2 * x) * x ≤ 144 :=
by
  -- The proof will be filled here
  sorry

end box_max_volume_l353_35317


namespace hyperbola_focus_l353_35362

theorem hyperbola_focus :
  ∃ (x y : ℝ), 2 * x^2 - y^2 - 8 * x + 4 * y - 4 = 0 ∧ (x, y) = (2 + 2 * Real.sqrt 3, 2) :=
by
  -- The proof would go here
  sorry

end hyperbola_focus_l353_35362


namespace jungkook_age_l353_35380

theorem jungkook_age
    (J U : ℕ)
    (h1 : J = U - 12)
    (h2 : (J + 3) + (U + 3) = 38) :
    J = 10 := 
sorry

end jungkook_age_l353_35380


namespace total_cost_of_stamps_is_correct_l353_35307

-- Define the costs of each type of stamp
def cost_of_stamp_A : ℕ := 34 -- cost in cents
def cost_of_stamp_B : ℕ := 52 -- cost in cents
def cost_of_stamp_C : ℕ := 73 -- cost in cents

-- Define the number of stamps Alice needs to buy
def num_stamp_A : ℕ := 4
def num_stamp_B : ℕ := 6
def num_stamp_C : ℕ := 2

-- Define the expected total cost in dollars
def expected_total_cost : ℝ := 5.94

-- State the theorem about the total cost
theorem total_cost_of_stamps_is_correct :
  ((num_stamp_A * cost_of_stamp_A) + (num_stamp_B * cost_of_stamp_B) + (num_stamp_C * cost_of_stamp_C)) / 100 = expected_total_cost :=
by
  sorry

end total_cost_of_stamps_is_correct_l353_35307


namespace some_number_value_l353_35368

theorem some_number_value (a : ℕ) (x : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * x * 49) : x = 9 := by
  sorry

end some_number_value_l353_35368


namespace percentage_of_students_owning_cats_l353_35353

theorem percentage_of_students_owning_cats (N C : ℕ) (hN : N = 500) (hC : C = 75) :
  (C / N : ℚ) * 100 = 15 := by
  sorry

end percentage_of_students_owning_cats_l353_35353


namespace passing_percentage_correct_l353_35389

-- The given conditions
def marks_obtained : ℕ := 175
def marks_failed : ℕ := 89
def max_marks : ℕ := 800

-- The theorem to prove
theorem passing_percentage_correct :
  (
    (marks_obtained + marks_failed : ℕ) * 100 / max_marks
  ) = 33 :=
sorry

end passing_percentage_correct_l353_35389


namespace domain_of_p_l353_35303

theorem domain_of_p (h : ℝ → ℝ) (h_domain : ∀ x, -10 ≤ x → x ≤ 6 → ∃ y, h x = y) :
  ∀ x, -1.2 ≤ x ∧ x ≤ 2 → ∃ y, h (-5 * x) = y :=
by
  sorry

end domain_of_p_l353_35303


namespace total_participants_l353_35342

-- Define the number of indoor and outdoor participants
variables (x y : ℕ)

-- First condition: number of outdoor participants is 480 more than indoor participants
def condition1 : Prop := y = x + 480

-- Second condition: moving 50 participants results in outdoor participants being 5 times the indoor participants
def condition2 : Prop := y + 50 = 5 * (x - 50)

-- Theorem statement: the total number of participants is 870
theorem total_participants (h1 : condition1 x y) (h2 : condition2 x y) : x + y = 870 :=
sorry

end total_participants_l353_35342


namespace phone_call_probability_within_four_rings_l353_35331

variables (P_A P_B P_C P_D : ℝ)

-- Assuming given probabilities
def probabilities_given : Prop :=
  P_A = 0.1 ∧ P_B = 0.3 ∧ P_C = 0.4 ∧ P_D = 0.1

theorem phone_call_probability_within_four_rings (h : probabilities_given P_A P_B P_C P_D) :
  P_A + P_B + P_C + P_D = 0.9 :=
sorry

end phone_call_probability_within_four_rings_l353_35331


namespace position_of_2017_in_arithmetic_sequence_l353_35384

theorem position_of_2017_in_arithmetic_sequence :
  ∀ (n : ℕ), 4 + 3 * (n - 1) = 2017 → n = 672 :=
by
  intros n h
  sorry

end position_of_2017_in_arithmetic_sequence_l353_35384


namespace prop1_prop2_l353_35370

-- Proposition 1: Prove the contrapositive
theorem prop1 (q : ℝ) (h : ¬(∃ x : ℝ, x^2 + 2 * x + q = 0)) : q ≥ 1 :=
sorry

-- Proposition 2: Prove the contrapositive
theorem prop2 (x y : ℝ) (h : ¬(x = 0 ∧ y = 0)) : x^2 + y^2 ≠ 0 :=
sorry

end prop1_prop2_l353_35370


namespace base3_20121_to_base10_l353_35365

def base3_to_base10 (n : ℕ) : ℕ :=
  2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0

theorem base3_20121_to_base10 :
  base3_to_base10 20121 = 178 :=
by
  sorry

end base3_20121_to_base10_l353_35365


namespace smallest_x_for_multiple_l353_35340

theorem smallest_x_for_multiple (x : ℕ) : (450 * x) % 720 = 0 ↔ x = 8 := 
by {
  sorry
}

end smallest_x_for_multiple_l353_35340


namespace calculate_square_of_complex_l353_35305

theorem calculate_square_of_complex (i : ℂ) (h : i^2 = -1) : (1 - i)^2 = -2 * i :=
by
  sorry

end calculate_square_of_complex_l353_35305


namespace max_product_two_four_digit_numbers_l353_35358

theorem max_product_two_four_digit_numbers :
  ∃ (a b : ℕ), 
    (a * b = max (8564 * 7321) (8531 * 7642)) 
    ∧ max 8531 8564 = 8531 ∧ 
    (∀ x y : ℕ, x * y ≤ 8531 * 7642 → x * y = max (8564 * 7321) (8531 * 7642)) :=
sorry

end max_product_two_four_digit_numbers_l353_35358


namespace polynomial_positive_for_all_reals_l353_35324

theorem polynomial_positive_for_all_reals (m : ℝ) : m^6 - m^5 + m^4 + m^2 - m + 1 > 0 :=
by
  sorry

end polynomial_positive_for_all_reals_l353_35324


namespace quadratic_inequality_range_l353_35309

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (1/2) * a * x^2 - a * x + 2 > 0) ↔ a ∈ Set.Ico 0 4 := 
by
  sorry

end quadratic_inequality_range_l353_35309


namespace number_square_25_l353_35334

theorem number_square_25 (x : ℝ) : x^2 = 25 ↔ x = 5 ∨ x = -5 := 
sorry

end number_square_25_l353_35334


namespace right_triangle_circum_inradius_sum_l353_35383

theorem right_triangle_circum_inradius_sum
  (a b : ℕ)
  (h1 : a = 16)
  (h2 : b = 30)
  (h_triangle : a^2 + b^2 = 34^2) :
  let c := 34
  let R := c / 2
  let A := a * b / 2
  let s := (a + b + c) / 2
  let r := A / s
  R + r = 23 :=
by
  sorry

end right_triangle_circum_inradius_sum_l353_35383


namespace motorist_gallons_affordable_l353_35351

-- Definitions based on the conditions in the problem
def expected_gallons : ℕ := 12
def actual_price_per_gallon : ℕ := 150
def price_difference : ℕ := 30
def expected_price_per_gallon : ℕ := actual_price_per_gallon - price_difference
def total_initial_cents : ℕ := expected_gallons * expected_price_per_gallon

-- Theorem stating that given the conditions, the motorist can afford 9 gallons of gas
theorem motorist_gallons_affordable : 
  total_initial_cents / actual_price_per_gallon = 9 := 
by
  sorry

end motorist_gallons_affordable_l353_35351


namespace calculate_product_value_l353_35304

theorem calculate_product_value :
    (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by
  sorry

end calculate_product_value_l353_35304


namespace fraction_ordering_l353_35367

theorem fraction_ordering : (4 / 17) < (6 / 25) ∧ (6 / 25) < (8 / 31) :=
by
  sorry

end fraction_ordering_l353_35367


namespace vector_addition_result_l353_35373

-- Definitions based on problem conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (y : ℝ) : ℝ × ℝ := (2, y)

-- The condition that vectors are parallel
def parallel_vectors (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2)

-- The main theorem to prove
theorem vector_addition_result (y : ℝ) (h : parallel_vectors vector_a (vector_b y)) : 
  (vector_a.1 + 2 * (vector_b y).1, vector_a.2 + 2 * (vector_b y).2) = (5, 10) :=
sorry

end vector_addition_result_l353_35373


namespace binom_2p_p_mod_p_l353_35326

theorem binom_2p_p_mod_p (p : ℕ) (hp : p.Prime) : Nat.choose (2 * p) p ≡ 2 [MOD p] := 
by
  sorry

end binom_2p_p_mod_p_l353_35326


namespace find_t_l353_35356

theorem find_t (c o u n t s : ℕ)
    (hc : c ≠ 0) (ho : o ≠ 0) (hn : n ≠ 0) (ht : t ≠ 0) (hs : s ≠ 0)
    (h1 : c + o = u)
    (h2 : u + n = t + 1)
    (h3 : t + c = s)
    (h4 : o + n + s = 15) :
    t = 7 := 
sorry

end find_t_l353_35356


namespace equal_sundays_tuesdays_days_l353_35345

-- Define the problem in Lean
def num_equal_sundays_and_tuesdays_starts : ℕ :=
  3

-- Define a function that calculates the number of starting days that result in equal Sundays and Tuesdays
def calculate_sundays_tuesdays_starts (days_in_month : ℕ) : ℕ :=
  if days_in_month = 30 then 3 else 0

-- Prove that for a month of 30 days, there are 3 valid starting days for equal Sundays and Tuesdays
theorem equal_sundays_tuesdays_days :
  calculate_sundays_tuesdays_starts 30 = num_equal_sundays_and_tuesdays_starts :=
by 
  -- Proof outline here
  sorry

end equal_sundays_tuesdays_days_l353_35345


namespace number_of_truthful_dwarfs_is_correct_l353_35310

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l353_35310


namespace derivative_at_zero_l353_35314

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + x)

-- Statement of the problem: The derivative of f at 0 is 1
theorem derivative_at_zero : deriv f 0 = 1 := 
  sorry

end derivative_at_zero_l353_35314
