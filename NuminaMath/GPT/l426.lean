import Mathlib

namespace NUMINAMATH_GPT_area_enclosed_curves_l426_42646

theorem area_enclosed_curves (a : ℝ) (h1 : (1 + 1/a)^5 = 1024) :
  ∫ x in (0 : ℝ)..1, (x^(1/3) - x^2) = 5/12 :=
sorry

end NUMINAMATH_GPT_area_enclosed_curves_l426_42646


namespace NUMINAMATH_GPT_roof_ratio_l426_42641

theorem roof_ratio (L W : ℕ) (h1 : L * W = 768) (h2 : L - W = 32) : L / W = 3 := 
sorry

end NUMINAMATH_GPT_roof_ratio_l426_42641


namespace NUMINAMATH_GPT_initial_money_l426_42653

-- Definitions based on conditions in the problem
def money_left_after_purchase : ℕ := 3
def cost_of_candy_bar : ℕ := 1

-- Theorem statement to prove the initial amount of money
theorem initial_money (initial_amount : ℕ) :
  initial_amount - cost_of_candy_bar = money_left_after_purchase → initial_amount = 4 :=
sorry

end NUMINAMATH_GPT_initial_money_l426_42653


namespace NUMINAMATH_GPT_first_term_geometric_series_l426_42613

theorem first_term_geometric_series (r a S : ℝ) (h1 : r = 1 / 4) (h2 : S = 40) (h3 : S = a / (1 - r)) : a = 30 :=
by
  sorry

end NUMINAMATH_GPT_first_term_geometric_series_l426_42613


namespace NUMINAMATH_GPT_total_players_on_team_l426_42606

theorem total_players_on_team (M W : ℕ) (h1 : W = M + 2) (h2 : (M : ℝ) / W = 0.7777777777777778) : M + W = 16 :=
by 
  sorry

end NUMINAMATH_GPT_total_players_on_team_l426_42606


namespace NUMINAMATH_GPT_find_n_squares_l426_42685

theorem find_n_squares (n : ℤ) : 
  (∃ a : ℤ, n^2 + 6 * n + 24 = a^2) ↔ n = 4 ∨ n = -2 ∨ n = -4 ∨ n = -10 :=
by
  sorry

end NUMINAMATH_GPT_find_n_squares_l426_42685


namespace NUMINAMATH_GPT_bacteria_after_10_hours_l426_42625

def bacteria_count (hours : ℕ) : ℕ :=
  2^hours

theorem bacteria_after_10_hours : bacteria_count 10 = 1024 := by
  sorry

end NUMINAMATH_GPT_bacteria_after_10_hours_l426_42625


namespace NUMINAMATH_GPT_find_abs_product_abc_l426_42621

theorem find_abs_product_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h : a + 1 / b = b + 1 / c ∧ b + 1 / c = c + 1 / a) : |a * b * c| = 1 :=
sorry

end NUMINAMATH_GPT_find_abs_product_abc_l426_42621


namespace NUMINAMATH_GPT_Isabelle_ticket_cost_l426_42630

theorem Isabelle_ticket_cost :
  (∀ (week_salary : ℕ) (weeks_worked : ℕ) (brother_ticket_cost : ℕ) (brothers_saved : ℕ) (Isabelle_saved : ℕ),
  week_salary = 3 ∧ weeks_worked = 10 ∧ brother_ticket_cost = 10 ∧ brothers_saved = 5 ∧ Isabelle_saved = 5 →
  Isabelle_saved + (week_salary * weeks_worked) - ((brother_ticket_cost * 2) - brothers_saved) = 15) :=
by
  sorry

end NUMINAMATH_GPT_Isabelle_ticket_cost_l426_42630


namespace NUMINAMATH_GPT_third_studio_students_l426_42643

theorem third_studio_students 
  (total_students : ℕ)
  (first_studio : ℕ)
  (second_studio : ℕ) 
  (third_studio : ℕ) 
  (h1 : total_students = 376) 
  (h2 : first_studio = 110) 
  (h3 : second_studio = 135) 
  (h4 : total_students = first_studio + second_studio + third_studio) :
  third_studio = 131 := 
sorry

end NUMINAMATH_GPT_third_studio_students_l426_42643


namespace NUMINAMATH_GPT_intersection_M_N_l426_42686

-- Definitions of the sets M and N
def M : Set ℝ := { -1, 0, 1 }
def N : Set ℝ := { x | x^2 ≤ x }

-- The theorem to be proven
theorem intersection_M_N : M ∩ N = { 0, 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l426_42686


namespace NUMINAMATH_GPT_cost_of_apple_is_two_l426_42611

-- Define the costs and quantities
def cost_of_apple (A : ℝ) : Prop :=
  let total_cost := 12 * A + 4 * 1 + 4 * 3
  let total_pieces := 12 + 4 + 4
  let average_cost := 2
  total_cost = total_pieces * average_cost

theorem cost_of_apple_is_two : cost_of_apple 2 :=
by
  -- Skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_cost_of_apple_is_two_l426_42611


namespace NUMINAMATH_GPT_degree_of_g_l426_42652

open Polynomial

theorem degree_of_g (f g : Polynomial ℂ) (h1 : f = -3 * X^5 + 4 * X^4 - X^2 + C 2) (h2 : degree (f + g) = 2) : degree g = 5 :=
sorry

end NUMINAMATH_GPT_degree_of_g_l426_42652


namespace NUMINAMATH_GPT_max_weight_of_flock_l426_42650

def MaxWeight (A E Af: ℕ): ℕ := A * 5 + E * 10 + Af * 15

theorem max_weight_of_flock :
  ∀ (A E Af: ℕ),
    A = 2 * E →
    Af = 3 * A →
    A + E + Af = 120 →
    MaxWeight A E Af = 1415 :=
by
  sorry

end NUMINAMATH_GPT_max_weight_of_flock_l426_42650


namespace NUMINAMATH_GPT_zoo_animals_left_l426_42657

noncomputable def totalAnimalsLeft (x : ℕ) : ℕ := 
  let initialFoxes := 2 * x
  let initialRabbits := 3 * x
  let foxesAfterMove := initialFoxes - 10
  let rabbitsAfterMove := initialRabbits / 2
  foxesAfterMove + rabbitsAfterMove

theorem zoo_animals_left (x : ℕ) (h : 20 * x - 100 = 39 * x / 2) : totalAnimalsLeft x = 690 := by
  sorry

end NUMINAMATH_GPT_zoo_animals_left_l426_42657


namespace NUMINAMATH_GPT_cuboid_dimensions_l426_42661

-- Define the problem conditions and the goal
theorem cuboid_dimensions (x y v : ℕ) :
  (v * (x * y - 1) = 602) ∧ (x * (v * y - 1) = 605) →
  v = x + 3 →
  x = 11 ∧ y = 4 ∧ v = 14 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_dimensions_l426_42661


namespace NUMINAMATH_GPT_min_product_of_positive_numbers_l426_42683

theorem min_product_of_positive_numbers {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : a * b = a + b) : a * b = 4 :=
sorry

end NUMINAMATH_GPT_min_product_of_positive_numbers_l426_42683


namespace NUMINAMATH_GPT_plastering_cost_correct_l426_42607

noncomputable def tank_length : ℝ := 25
noncomputable def tank_width : ℝ := 12
noncomputable def tank_depth : ℝ := 6
noncomputable def cost_per_sqm_paise : ℝ := 75
noncomputable def cost_per_sqm_rupees : ℝ := cost_per_sqm_paise / 100

noncomputable def total_cost_plastering : ℝ :=
  let long_wall_area := 2 * (tank_length * tank_depth)
  let short_wall_area := 2 * (tank_width * tank_depth)
  let bottom_area := tank_length * tank_width
  let total_area := long_wall_area + short_wall_area + bottom_area
  total_area * cost_per_sqm_rupees

theorem plastering_cost_correct : total_cost_plastering = 558 := by
  sorry

end NUMINAMATH_GPT_plastering_cost_correct_l426_42607


namespace NUMINAMATH_GPT_trail_length_is_48_meters_l426_42642

noncomputable def length_of_trail (d: ℝ) : Prop :=
  let normal_speed := 8 -- normal speed in m/s
  let mud_speed := normal_speed / 4 -- speed in mud in m/s

  let time_mud := (1 / 3 * d) / mud_speed -- time through the mud in seconds
  let time_normal := (2 / 3 * d) / normal_speed -- time through the normal trail in seconds

  let total_time := 12 -- total time in seconds

  total_time = time_mud + time_normal

theorem trail_length_is_48_meters : ∃ d: ℝ, length_of_trail d ∧ d = 48 :=
sorry

end NUMINAMATH_GPT_trail_length_is_48_meters_l426_42642


namespace NUMINAMATH_GPT_calculate_expression_l426_42648

theorem calculate_expression : (2^1234 + 5^1235)^2 - (2^1234 - 5^1235)^2 = 20 * 10^1234 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l426_42648


namespace NUMINAMATH_GPT_age_sum_is_ninety_l426_42677

theorem age_sum_is_ninety (a b c : ℕ)
  (h1 : a = 20 + b + c)
  (h2 : a^2 = 1800 + (b + c)^2) :
  a + b + c = 90 := 
sorry

end NUMINAMATH_GPT_age_sum_is_ninety_l426_42677


namespace NUMINAMATH_GPT_inverse_function_of_f_l426_42694

noncomputable def f (x : ℝ) : ℝ := (x - 1) ^ 2

noncomputable def f_inv (y : ℝ) : ℝ := 1 - Real.sqrt y

theorem inverse_function_of_f :
  ∀ x, x ≤ 1 → f_inv (f x) = x ∧ ∀ y, 0 ≤ y → f (f_inv y) = y :=
by
  intros
  sorry

end NUMINAMATH_GPT_inverse_function_of_f_l426_42694


namespace NUMINAMATH_GPT_possible_values_of_A_l426_42627

theorem possible_values_of_A :
  ∃ (A : ℕ), (A ≤ 4 ∧ A < 10) ∧ A = 5 :=
sorry

end NUMINAMATH_GPT_possible_values_of_A_l426_42627


namespace NUMINAMATH_GPT_tim_points_l426_42654

theorem tim_points (J T K : ℝ) (h1 : T = J + 20) (h2 : T = K / 2) (h3 : J + T + K = 100) : T = 30 := 
by 
  sorry

end NUMINAMATH_GPT_tim_points_l426_42654


namespace NUMINAMATH_GPT_casper_entry_exit_ways_correct_l426_42620

-- Define the total number of windows
def num_windows : Nat := 8

-- Define the number of ways Casper can enter and exit through different windows
def casper_entry_exit_ways (num_windows : Nat) : Nat :=
  num_windows * (num_windows - 1)

-- Create a theorem to state the problem and its solution
theorem casper_entry_exit_ways_correct : casper_entry_exit_ways num_windows = 56 := by
  sorry

end NUMINAMATH_GPT_casper_entry_exit_ways_correct_l426_42620


namespace NUMINAMATH_GPT_max_profit_l426_42680

theorem max_profit : ∃ v p : ℝ, 
  v + p ≤ 5 ∧
  v + 3 * p ≤ 12 ∧
  100000 * v + 200000 * p = 850000 :=
by
  sorry

end NUMINAMATH_GPT_max_profit_l426_42680


namespace NUMINAMATH_GPT_exists_triangle_with_prime_angles_l426_42604

-- Definition of prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Definition of being an angle of a triangle
def is_valid_angle (α : ℕ) : Prop := α > 0 ∧ α < 180

-- Main statement
theorem exists_triangle_with_prime_angles :
  ∃ (α β γ : ℕ), is_prime α ∧ is_prime β ∧ is_prime γ ∧ is_valid_angle α ∧ is_valid_angle β ∧ is_valid_angle γ ∧ α + β + γ = 180 :=
by
  sorry

end NUMINAMATH_GPT_exists_triangle_with_prime_angles_l426_42604


namespace NUMINAMATH_GPT_length_of_AB_l426_42674

variables (AB CD : ℝ)

-- Given conditions
def area_ratio (h : ℝ) : Prop := (1/2 * AB * h) / (1/2 * CD * h) = 4
def sum_condition : Prop := AB + CD = 200

-- The proof problem: proving the length of AB
theorem length_of_AB (h : ℝ) (h_area_ratio : area_ratio AB CD h) 
  (h_sum_condition : sum_condition AB CD) : AB = 160 :=
sorry

end NUMINAMATH_GPT_length_of_AB_l426_42674


namespace NUMINAMATH_GPT_Chloe_total_points_l426_42695

-- Define the points scored in each round
def first_round_points : ℕ := 40
def second_round_points : ℕ := 50
def last_round_points : ℤ := -4

-- Define total points calculation
def total_points := first_round_points + second_round_points + last_round_points

-- The final statement to prove
theorem Chloe_total_points : total_points = 86 := by
  -- This proof is to be completed
  sorry

end NUMINAMATH_GPT_Chloe_total_points_l426_42695


namespace NUMINAMATH_GPT_negation_statement_l426_42679

variable (x y : ℝ)

theorem negation_statement :
  ¬ (x > 1 ∧ y > 2) ↔ (x ≤ 1 ∨ y ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_negation_statement_l426_42679


namespace NUMINAMATH_GPT_journey_time_approx_24_hours_l426_42691

noncomputable def journey_time_in_hours : ℝ :=
  let t1 := 70 / 60  -- time for destination 1
  let t2 := 50 / 35  -- time for destination 2
  let t3 := 20 / 60 + 20 / 30  -- time for destination 3
  let t4 := 30 / 40 + 60 / 70  -- time for destination 4
  let t5 := 60 / 35  -- time for destination 5
  let return_distance := 70 + 50 + 40 + 90 + 60 + 100  -- total return distance
  let return_time := return_distance / 55  -- time for return journey
  let stay_time := 1 + 3 + 2 + 2.5 + 0.75  -- total stay time
  t1 + t2 + t3 + t4 + t5 + return_time + stay_time  -- total journey time

theorem journey_time_approx_24_hours : abs (journey_time_in_hours - 24) < 1 :=
by
  sorry

end NUMINAMATH_GPT_journey_time_approx_24_hours_l426_42691


namespace NUMINAMATH_GPT_find_A_l426_42660

-- Define the condition as an axiom
axiom A : ℝ
axiom condition : A + 10 = 15 

-- Prove that given the condition, A must be 5
theorem find_A : A = 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_A_l426_42660


namespace NUMINAMATH_GPT_find_a_l426_42676

theorem find_a (a : ℝ) (x : ℝ) (h : x^2 + a * x + 4 = (x + 2)^2) : a = 4 :=
sorry

end NUMINAMATH_GPT_find_a_l426_42676


namespace NUMINAMATH_GPT_find_fourth_vertex_l426_42689

open Complex

theorem find_fourth_vertex (A B C: ℂ) (hA: A = 2 + 3 * Complex.I) 
                            (hB: B = -3 + 2 * Complex.I) 
                            (hC: C = -2 - 3 * Complex.I) : 
                            ∃ D : ℂ, D = 2.5 + 0.5 * Complex.I :=
by 
  sorry

end NUMINAMATH_GPT_find_fourth_vertex_l426_42689


namespace NUMINAMATH_GPT_number_of_trousers_given_l426_42698

-- Define the conditions
def shirts_given : Nat := 589
def total_clothing_given : Nat := 934

-- Define the expected answer
def expected_trousers_given : Nat := 345

-- The theorem statement to prove the number of trousers given
theorem number_of_trousers_given : total_clothing_given - shirts_given = expected_trousers_given :=
by
  sorry

end NUMINAMATH_GPT_number_of_trousers_given_l426_42698


namespace NUMINAMATH_GPT_four_real_solutions_l426_42622

-- Definitions used in the problem
def P (x : ℝ) : Prop := (6 * x) / (x^2 + 2 * x + 5) + (4 * x) / (x^2 - 4 * x + 5) = -2 / 3

-- Statement of the problem
theorem four_real_solutions : ∃ (x1 x2 x3 x4 : ℝ), P x1 ∧ P x2 ∧ P x3 ∧ P x4 ∧ 
  ∀ x, P x → (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4) :=
sorry

end NUMINAMATH_GPT_four_real_solutions_l426_42622


namespace NUMINAMATH_GPT_points_symmetric_about_x_axis_l426_42669

def point := ℝ × ℝ

def symmetric_x_axis (A B : point) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

theorem points_symmetric_about_x_axis : symmetric_x_axis (-1, 3) (-1, -3) :=
by
  sorry

end NUMINAMATH_GPT_points_symmetric_about_x_axis_l426_42669


namespace NUMINAMATH_GPT_baseball_team_groups_l426_42632

theorem baseball_team_groups (new_players returning_players players_per_group : ℕ) (h_new : new_players = 48) (h_return : returning_players = 6) (h_per_group : players_per_group = 6) : (new_players + returning_players) / players_per_group = 9 :=
by
  sorry

end NUMINAMATH_GPT_baseball_team_groups_l426_42632


namespace NUMINAMATH_GPT_total_cost_bicycle_helmet_l426_42692

-- Let h represent the cost of the helmet
def helmet_cost := 40

-- Let b represent the cost of the bicycle
def bicycle_cost := 5 * helmet_cost

-- We need to prove that the total cost (bicycle + helmet) is equal to 240
theorem total_cost_bicycle_helmet : bicycle_cost + helmet_cost = 240 := 
by
  -- This will skip the proof, we only need the statement
  sorry

end NUMINAMATH_GPT_total_cost_bicycle_helmet_l426_42692


namespace NUMINAMATH_GPT_hotpot_total_cost_l426_42665

def table_cost : ℝ := 280
def table_limit : ℕ := 8
def extra_person_cost : ℝ := 29.9
def total_people : ℕ := 12

theorem hotpot_total_cost : 
  total_people > table_limit →
  table_cost + (total_people - table_limit) * extra_person_cost = 369.7 := 
by 
  sorry

end NUMINAMATH_GPT_hotpot_total_cost_l426_42665


namespace NUMINAMATH_GPT_sum_of_vertical_asymptotes_l426_42610

noncomputable def sum_of_roots (a b c : ℝ) (h_discriminant : b^2 - 4*a*c ≠ 0) : ℝ :=
-(b/a)

theorem sum_of_vertical_asymptotes :
  let f := (6 * (x^2) - 8) / (4 * (x^2) + 7*x + 3)
  ∃ c d, c ≠ d ∧ (4*c^2 + 7*c + 3 = 0) ∧ (4*d^2 + 7*d + 3 = 0)
  ∧ c + d = -7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_vertical_asymptotes_l426_42610


namespace NUMINAMATH_GPT_seq_common_max_l426_42614

theorem seq_common_max : ∃ a, a ≤ 250 ∧ 1 ≤ a ∧ a % 8 = 1 ∧ a % 9 = 4 ∧ ∀ b, b ≤ 250 ∧ 1 ≤ b ∧ b % 8 = 1 ∧ b % 9 = 4 → b ≤ a :=
by 
  sorry

end NUMINAMATH_GPT_seq_common_max_l426_42614


namespace NUMINAMATH_GPT_total_outfits_l426_42697

def numRedShirts : ℕ := 7
def numGreenShirts : ℕ := 5
def numPants : ℕ := 6
def numRedHats : ℕ := 7
def numGreenHats : ℕ := 9

theorem total_outfits : 
  ((numRedShirts * numPants * numGreenHats) + 
   (numGreenShirts * numPants * numRedHats) + 
   ((numRedShirts * numRedHats + numGreenShirts * numGreenHats) * numPants)
  ) = 1152 := 
by
  sorry

end NUMINAMATH_GPT_total_outfits_l426_42697


namespace NUMINAMATH_GPT_sum_abs_frac_geq_frac_l426_42684

theorem sum_abs_frac_geq_frac (n : ℕ) (h1 : n ≥ 3) (a : Fin n → ℝ) (hnz : ∀ i : Fin n, a i ≠ 0) 
(hsum : (Finset.univ.sum a) = S) : 
  (Finset.univ.sum (fun i => |(S - a i) / a i|)) ≥ (n - 1) / (n - 2) :=
sorry

end NUMINAMATH_GPT_sum_abs_frac_geq_frac_l426_42684


namespace NUMINAMATH_GPT_tan_alpha_plus_beta_mul_tan_alpha_l426_42601

theorem tan_alpha_plus_beta_mul_tan_alpha (α β : ℝ) (h : 2 * Real.cos (2 * α + β) + 3 * Real.cos β = 0) :
  Real.tan (α + β) * Real.tan α = -5 := 
by
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_beta_mul_tan_alpha_l426_42601


namespace NUMINAMATH_GPT_trig_expression_value_l426_42678

theorem trig_expression_value : 
  (2 * (Real.sin (25 * Real.pi / 180))^2 - 1) / 
  (Real.sin (20 * Real.pi / 180) * Real.cos (20 * Real.pi / 180)) = -2 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_trig_expression_value_l426_42678


namespace NUMINAMATH_GPT_total_hours_A_ascending_and_descending_l426_42672

theorem total_hours_A_ascending_and_descending
  (ascending_speed_A ascending_speed_B descending_speed_A descending_speed_B distance summit_distance : ℝ)
  (h1 : descending_speed_A = 1.5 * ascending_speed_A)
  (h2 : descending_speed_B = 1.5 * ascending_speed_B)
  (h3 : ascending_speed_A > ascending_speed_B)
  (h4 : 1/ascending_speed_A + 1/ascending_speed_B = 1/hour - 600/summit_distance)
  (h5 : 0.5 * summit_distance/ascending_speed_A = (summit_distance - 600)/ascending_speed_B) :
  (summit_distance / ascending_speed_A) + (summit_distance / descending_speed_A) = 1.5 := 
sorry

end NUMINAMATH_GPT_total_hours_A_ascending_and_descending_l426_42672


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l426_42633

theorem sum_of_roots_of_quadratic (x1 x2 : ℝ) (h : x1 * x2 + -(x1 + x2) * 6 + 5 = 0) : x1 + x2 = 6 :=
by
-- Vieta's formulas for the sum of the roots of a quadratic equation state that x1 + x2 = -b / a.
sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l426_42633


namespace NUMINAMATH_GPT_exists_distinct_abc_sum_l426_42615

theorem exists_distinct_abc_sum (n : ℕ) (h : n ≥ 1) (X : Finset ℤ)
  (h_card : X.card = n + 2)
  (h_abs : ∀ x ∈ X, abs x ≤ n) :
  ∃ (a b c : ℤ), a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b = c :=
sorry

end NUMINAMATH_GPT_exists_distinct_abc_sum_l426_42615


namespace NUMINAMATH_GPT_length_of_top_side_l426_42616

def height_of_trapezoid : ℝ := 8
def area_of_trapezoid : ℝ := 72
def top_side_is_shorter (b : ℝ) : Prop := ∃ t : ℝ, t = b - 6

theorem length_of_top_side (b t : ℝ) (h_height : height_of_trapezoid = 8)
  (h_area : area_of_trapezoid = 72) 
  (h_top_side : top_side_is_shorter b)
  (h_area_formula : (1/2) * (b + t) * 8 = 72) : t = 6 := 
by 
  sorry

end NUMINAMATH_GPT_length_of_top_side_l426_42616


namespace NUMINAMATH_GPT_smallest_possible_N_l426_42631

theorem smallest_possible_N (p q r s t : ℕ) (hp : p > 0) (hq : q > 0) 
  (hr : r > 0) (hs : s > 0) (ht : t > 0) (h_sum : p + q + r + s + t = 4020) :
  ∃ N, N = max (max (p + q) (q + r)) (max (r + s) (s + t)) ∧ N = 1005 :=
sorry

end NUMINAMATH_GPT_smallest_possible_N_l426_42631


namespace NUMINAMATH_GPT_find_legs_of_triangle_l426_42640

theorem find_legs_of_triangle (a b : ℝ) (h : a / b = 3 / 4) (h_sum : a^2 + b^2 = 70^2) : 
  (a = 42) ∧ (b = 56) :=
sorry

end NUMINAMATH_GPT_find_legs_of_triangle_l426_42640


namespace NUMINAMATH_GPT_inequality_solution_empty_l426_42634

theorem inequality_solution_empty {a x: ℝ} : 
  (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0 → 
  (-2 < a) ∧ (a < 6 / 5) :=
sorry

end NUMINAMATH_GPT_inequality_solution_empty_l426_42634


namespace NUMINAMATH_GPT_find_ordered_pair_l426_42699

theorem find_ordered_pair (x y : ℝ) :
  (2 * x + 3 * y = (6 - x) + (6 - 3 * y)) ∧ (x - 2 * y = (x - 2) - (y + 2)) ↔ (x = -4) ∧ (y = 4) := by
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l426_42699


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l426_42663

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l426_42663


namespace NUMINAMATH_GPT_quadratic_intersection_l426_42619

def quadratic (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

theorem quadratic_intersection:
  ∃ a b c : ℝ, 
  quadratic a b c (-3) = 16 ∧ 
  quadratic a b c 0 = -5 ∧ 
  quadratic a b c 3 = -8 ∧ 
  quadratic a b c (-1) = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_intersection_l426_42619


namespace NUMINAMATH_GPT_determine_linear_relation_l426_42609

-- Define the set of options
inductive PlotType
| Scatter
| StemAndLeaf
| FrequencyHistogram
| FrequencyLineChart

-- Define the question and state the expected correct answer
def correctPlotTypeForLinearRelation : PlotType :=
  PlotType.Scatter

-- Prove that the correct method for determining linear relation in a set of data is a Scatter plot
theorem determine_linear_relation :
  correctPlotTypeForLinearRelation = PlotType.Scatter :=
by
  sorry

end NUMINAMATH_GPT_determine_linear_relation_l426_42609


namespace NUMINAMATH_GPT_product_of_five_consecutive_divisible_by_30_l426_42623

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end NUMINAMATH_GPT_product_of_five_consecutive_divisible_by_30_l426_42623


namespace NUMINAMATH_GPT_terms_before_five_l426_42647

theorem terms_before_five (a₁ : ℤ) (d : ℤ) (n : ℤ) :
  a₁ = 75 → d = -5 → (a₁ + (n - 1) * d = 5) → n - 1 = 14 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_terms_before_five_l426_42647


namespace NUMINAMATH_GPT_kamal_marks_in_english_l426_42664

theorem kamal_marks_in_english :
  ∀ (E Math Physics Chemistry Biology Average : ℕ), 
    Math = 65 → 
    Physics = 82 → 
    Chemistry = 67 → 
    Biology = 85 → 
    Average = 79 → 
    (Math + Physics + Chemistry + Biology + E) / 5 = Average → 
    E = 96 :=
by
  intros E Math Physics Chemistry Biology Average
  intros hMath hPhysics hChemistry hBiology hAverage hTotal
  sorry

end NUMINAMATH_GPT_kamal_marks_in_english_l426_42664


namespace NUMINAMATH_GPT_average_marks_math_chem_l426_42659

theorem average_marks_math_chem (M P C : ℝ) (h1 : M + P = 60) (h2 : C = P + 20) : 
  (M + C) / 2 = 40 := 
by
  sorry

end NUMINAMATH_GPT_average_marks_math_chem_l426_42659


namespace NUMINAMATH_GPT_roots_sum_powers_l426_42673

theorem roots_sum_powers (t : ℕ → ℝ) (b d f : ℝ)
  (ht0 : t 0 = 3)
  (ht1 : t 1 = 6)
  (ht2 : t 2 = 11)
  (hrec : ∀ k ≥ 2, t (k + 1) = b * t k + d * t (k - 1) + f * t (k - 2))
  (hpoly : ∀ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6 = 0) :
  b + d + f = 13 :=
sorry

end NUMINAMATH_GPT_roots_sum_powers_l426_42673


namespace NUMINAMATH_GPT_certain_number_l426_42668

theorem certain_number (x : ℝ) (h : 7125 / x = 5700) : x = 1.25 := 
sorry

end NUMINAMATH_GPT_certain_number_l426_42668


namespace NUMINAMATH_GPT_max_squares_fitting_l426_42644

theorem max_squares_fitting (L S : ℕ) (hL : L = 8) (hS : S = 2) : (L / S) * (L / S) = 16 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_max_squares_fitting_l426_42644


namespace NUMINAMATH_GPT_B_finishes_in_10_days_l426_42693

noncomputable def B_remaining_work_days (A_work_days : ℕ := 15) (A_initial_days_worked : ℕ := 5) (B_work_days : ℝ := 14.999999999999996) : ℝ :=
  let A_rate := 1 / A_work_days
  let B_rate := 1 / B_work_days
  let remaining_work := 1 - (A_rate * A_initial_days_worked)
  let days_for_B := remaining_work / B_rate
  days_for_B

theorem B_finishes_in_10_days :
  B_remaining_work_days 15 5 14.999999999999996 = 10 :=
by
  sorry

end NUMINAMATH_GPT_B_finishes_in_10_days_l426_42693


namespace NUMINAMATH_GPT_pastries_made_correct_l426_42696

-- Definitions based on conditions
def cakes_made := 14
def cakes_sold := 97
def pastries_sold := 8
def cakes_more_than_pastries := 89

-- Definition of the function to compute pastries made
def pastries_made (cakes_made cakes_sold pastries_sold cakes_more_than_pastries : ℕ) : ℕ :=
  cakes_sold - cakes_more_than_pastries

-- The statement to prove
theorem pastries_made_correct : pastries_made cakes_made cakes_sold pastries_sold cakes_more_than_pastries = 8 := by
  unfold pastries_made
  norm_num
  sorry

end NUMINAMATH_GPT_pastries_made_correct_l426_42696


namespace NUMINAMATH_GPT_polygon_sides_l426_42628

theorem polygon_sides (x : ℕ) (h : 180 * (x - 2) = 1080) : x = 8 :=
by sorry

end NUMINAMATH_GPT_polygon_sides_l426_42628


namespace NUMINAMATH_GPT_bus_full_problem_l426_42649

theorem bus_full_problem
      (cap : ℕ := 80)
      (first_pickup_ratio : ℚ := 3/5)
      (second_pickup_exit : ℕ := 15)
      (waiting_people : ℕ := 50) :
      waiting_people - (cap - (first_pickup_ratio * cap - second_pickup_exit)) = 3 := by
  sorry

end NUMINAMATH_GPT_bus_full_problem_l426_42649


namespace NUMINAMATH_GPT_darla_total_payment_l426_42656

-- Definitions of the conditions
def rate_per_watt : ℕ := 4
def energy_usage : ℕ := 300
def late_fee : ℕ := 150

-- Definition of the expected total cost
def expected_total_cost : ℕ := 1350

-- Theorem stating the problem
theorem darla_total_payment :
  rate_per_watt * energy_usage + late_fee = expected_total_cost := 
by 
  sorry

end NUMINAMATH_GPT_darla_total_payment_l426_42656


namespace NUMINAMATH_GPT_additional_time_due_to_leak_l426_42635

theorem additional_time_due_to_leak (fill_time_no_leak: ℝ) (leak_empty_time: ℝ) (fill_rate_no_leak: fill_time_no_leak ≠ 0):
  (fill_time_no_leak = 3) → 
  (leak_empty_time = 12) → 
  (1 / fill_time_no_leak - 1 / leak_empty_time ≠ 0) → 
  ((1 / fill_time_no_leak - 1 / leak_empty_time) / (1 / (1 / fill_time_no_leak - 1 / leak_empty_time)) - fill_time_no_leak = 1) := 
by
  intro h_fill h_leak h_effective_rate
  sorry

end NUMINAMATH_GPT_additional_time_due_to_leak_l426_42635


namespace NUMINAMATH_GPT_company_fund_initial_amount_l426_42618

theorem company_fund_initial_amount
  (n : ℕ) -- number of employees
  (initial_bonus_per_employee : ℕ := 60)
  (shortfall : ℕ := 10)
  (revised_bonus_per_employee : ℕ := 50)
  (fund_remaining : ℕ := 150)
  (initial_fund : ℕ := initial_bonus_per_employee * n - shortfall) -- condition that the fund was $10 short when planning the initial bonus
  (revised_fund : ℕ := revised_bonus_per_employee * n + fund_remaining) -- condition after distributing the $50 bonuses

  (eqn : initial_fund = revised_fund) -- equating initial and revised budget calculations
  
  : initial_fund = 950 := 
sorry

end NUMINAMATH_GPT_company_fund_initial_amount_l426_42618


namespace NUMINAMATH_GPT_households_using_neither_brands_l426_42617

def total_households : Nat := 240
def only_brand_A_households : Nat := 60
def both_brands_households : Nat := 25
def ratio_B_to_both : Nat := 3
def only_brand_B_households : Nat := ratio_B_to_both * both_brands_households
def either_brand_households : Nat := only_brand_A_households + only_brand_B_households + both_brands_households
def neither_brand_households : Nat := total_households - either_brand_households

theorem households_using_neither_brands :
  neither_brand_households = 80 :=
by
  -- Proof can be filled out here
  sorry

end NUMINAMATH_GPT_households_using_neither_brands_l426_42617


namespace NUMINAMATH_GPT_perpendicular_MP_MQ_l426_42637

variable (k m : ℝ)

def ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1

def line (x y : ℝ) := y = k*x + m

def fixed_point_exists (k m : ℝ) : Prop :=
  let P := (-(4 * k) / m, 3 / m)
  let Q := (4, 4 * k + m)
  ∃ (M : ℝ), (M = 1 ∧ ((P.1 - M) * (Q.1 - M) + P.2 * Q.2 = 0))

theorem perpendicular_MP_MQ : fixed_point_exists k m := sorry

end NUMINAMATH_GPT_perpendicular_MP_MQ_l426_42637


namespace NUMINAMATH_GPT_manu_wins_probability_l426_42624

def prob_manu_wins : ℚ :=
  let a := (1/2) ^ 5
  let r := (1/2) ^ 4
  a / (1 - r)

theorem manu_wins_probability : prob_manu_wins = 1 / 30 :=
  by
  -- here we would have the proof steps
  sorry

end NUMINAMATH_GPT_manu_wins_probability_l426_42624


namespace NUMINAMATH_GPT_find_special_two_digit_integer_l426_42602

theorem find_special_two_digit_integer (n : ℕ) (h1 : 10 ≤ n ∧ n < 100)
  (h2 : (n + 3) % 3 = 0)
  (h3 : (n + 4) % 4 = 0)
  (h4 : (n + 5) % 5 = 0) :
  n = 60 := by
  sorry

end NUMINAMATH_GPT_find_special_two_digit_integer_l426_42602


namespace NUMINAMATH_GPT_chord_intersection_probability_l426_42682

noncomputable def probability_chord_intersection : ℚ :=
1 / 3

theorem chord_intersection_probability 
    (A B C D : ℕ) 
    (total_points : ℕ) 
    (adjacent : A + 1 = B ∨ A = B + 1)
    (distinct : ∀ (A B C D : ℕ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (points_on_circle : total_points = 2023) :
    ∃ p : ℚ, p = probability_chord_intersection :=
by sorry

end NUMINAMATH_GPT_chord_intersection_probability_l426_42682


namespace NUMINAMATH_GPT_nonneg_integer_solutions_l426_42681

theorem nonneg_integer_solutions :
  { x : ℕ | 5 * x + 3 < 3 * (2 + x) } = {0, 1} :=
by
  sorry

end NUMINAMATH_GPT_nonneg_integer_solutions_l426_42681


namespace NUMINAMATH_GPT_membership_relation_l426_42662

-- Definitions of M and N
def M (x : ℝ) : Prop := abs (x + 1) < 4
def N (x : ℝ) : Prop := x / (x - 3) < 0

theorem membership_relation (a : ℝ) (h : M a) : N a → M a := by
  sorry

end NUMINAMATH_GPT_membership_relation_l426_42662


namespace NUMINAMATH_GPT_product_of_solutions_of_abs_equation_l426_42636

theorem product_of_solutions_of_abs_equation : 
  (∃ x1 x2 : ℝ, |5 * x1| + 2 = 47 ∧ |5 * x2| + 2 = 47 ∧ x1 ≠ x2 ∧ x1 * x2 = -81) :=
sorry

end NUMINAMATH_GPT_product_of_solutions_of_abs_equation_l426_42636


namespace NUMINAMATH_GPT_caterer_cheapest_option_l426_42608

theorem caterer_cheapest_option :
  ∃ x : ℕ, x ≥ 42 ∧ (∀ y : ℕ, y ≥ x → (20 * y < 120 + 18 * y) ∧ (20 * y < 250 + 14 * y)) := 
by
  sorry

end NUMINAMATH_GPT_caterer_cheapest_option_l426_42608


namespace NUMINAMATH_GPT_count_valid_pairs_l426_42688

theorem count_valid_pairs : 
  ∃! n : ℕ, 
  n = 2 ∧ 
  (∀ (a b : ℕ), (0 < a ∧ 0 < b) → 
    (a * b + 97 = 18 * Nat.lcm a b + 14 * Nat.gcd a b) → 
    n = 2)
:= sorry

end NUMINAMATH_GPT_count_valid_pairs_l426_42688


namespace NUMINAMATH_GPT_marbles_leftover_l426_42600

theorem marbles_leftover (r p j : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 7) (hj : j % 8 = 2) : (r + p + j) % 8 = 6 := 
sorry

end NUMINAMATH_GPT_marbles_leftover_l426_42600


namespace NUMINAMATH_GPT_max_elevation_reached_l426_42690

theorem max_elevation_reached 
  (t : ℝ) 
  (s : ℝ) 
  (h : s = 200 * t - 20 * t^2) : 
  ∃ t_max : ℝ, ∃ s_max : ℝ, t_max = 5 ∧ s_max = 500 ∧ s_max = 200 * t_max - 20 * t_max^2 := sorry

end NUMINAMATH_GPT_max_elevation_reached_l426_42690


namespace NUMINAMATH_GPT_prevent_four_digit_number_l426_42655

theorem prevent_four_digit_number (N : ℕ) (n : ℕ) :
  n = 123 + 102 * N ∧ ∀ x : ℕ, (3 + 2 * x) % 10 < 1000 → x < 1000 := 
sorry

end NUMINAMATH_GPT_prevent_four_digit_number_l426_42655


namespace NUMINAMATH_GPT_Im_abcd_eq_zero_l426_42603

noncomputable def normalized (z : ℂ) : ℂ := z / Complex.abs z

theorem Im_abcd_eq_zero (a b c d : ℂ)
  (h1 : ∃ α : ℝ, ∃ w : ℂ, w = Complex.cos α + Complex.sin α * Complex.I ∧ (normalized b = w * normalized a) ∧ (normalized d = w * normalized c)) :
  Complex.im (a * b * c * d) = 0 :=
by
  sorry

end NUMINAMATH_GPT_Im_abcd_eq_zero_l426_42603


namespace NUMINAMATH_GPT_find_rate_per_kg_mangoes_l426_42670

-- Definitions based on the conditions
def rate_per_kg_grapes : ℕ := 70
def quantity_grapes : ℕ := 8
def total_payment : ℕ := 1000
def quantity_mangoes : ℕ := 8

-- Proposition stating what we want to prove
theorem find_rate_per_kg_mangoes (r : ℕ) (H : total_payment = (rate_per_kg_grapes * quantity_grapes) + (r * quantity_mangoes)) : r = 55 := sorry

end NUMINAMATH_GPT_find_rate_per_kg_mangoes_l426_42670


namespace NUMINAMATH_GPT_converse_proposition_l426_42687

-- Define the predicate variables p and q
variables (p q : Prop)

-- State the theorem about the converse of the proposition
theorem converse_proposition (hpq : p → q) : q → p :=
sorry

end NUMINAMATH_GPT_converse_proposition_l426_42687


namespace NUMINAMATH_GPT_find_x_value_l426_42667

theorem find_x_value (PQ_is_straight_line : True) 
  (angles_on_line : List ℕ) (h : angles_on_line = [x, x, x, x, x])
  (sum_of_angles : angles_on_line.sum = 180) :
  x = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_x_value_l426_42667


namespace NUMINAMATH_GPT_extreme_point_inequality_l426_42651

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x - a / x - 2 * Real.log x

theorem extreme_point_inequality (x₁ x₂ a : ℝ) (h1 : x₁ < x₂) (h2 : f x₁ a = 0) (h3 : f x₂ a = 0) 
(h_a_range : 0 < a) (h_a_lt_1 : a < 1) :
  f x₂ a < x₂ - 1 :=
sorry

end NUMINAMATH_GPT_extreme_point_inequality_l426_42651


namespace NUMINAMATH_GPT_range_of_m_l426_42629

-- Define the polynomial p(x)
def p (x : ℝ) (m : ℝ) := x^2 + 2*x - m

-- Given conditions: p(1) is false and p(2) is true
theorem range_of_m (m : ℝ) : 
  (p 1 m ≤ 0) ∧ (p 2 m > 0) → (3 ≤ m ∧ m < 8) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l426_42629


namespace NUMINAMATH_GPT_range_of_a_l426_42626

noncomputable def condition_p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
noncomputable def condition_q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) :
  (¬(∀ x, condition_p x)) → (¬(∀ x, condition_q x a)) → 
  (∀ x, condition_p x ↔ condition_q x a) → (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l426_42626


namespace NUMINAMATH_GPT_inverse_tangent_line_l426_42639

theorem inverse_tangent_line
  (f : ℝ → ℝ)
  (hf₁ : ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ f (g x) = x) 
  (hf₂ : ∀ x, deriv f x ≠ 0)
  (h_tangent : ∀ x₀, (2 * x₀ - f x₀ + 3) = 0) :
  ∀ x₀, (x₀ - 2 * f x₀ - 3) = 0 :=
by
  sorry

end NUMINAMATH_GPT_inverse_tangent_line_l426_42639


namespace NUMINAMATH_GPT_initial_pieces_l426_42658

-- Definitions of the conditions
def pieces_eaten : ℕ := 7
def pieces_given : ℕ := 21
def pieces_now : ℕ := 37

-- The proposition to prove
theorem initial_pieces (C : ℕ) (h : C - pieces_eaten + pieces_given = pieces_now) : C = 23 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_initial_pieces_l426_42658


namespace NUMINAMATH_GPT_carol_extra_invitations_l426_42675

theorem carol_extra_invitations : 
  let invitations_per_pack := 3
  let packs_bought := 2
  let friends_to_invite := 9
  packs_bought * invitations_per_pack < friends_to_invite → 
  friends_to_invite - (packs_bought * invitations_per_pack) = 3 :=
by 
  intros _  -- Introduce the condition
  exact sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_carol_extra_invitations_l426_42675


namespace NUMINAMATH_GPT_quadratic_function_conditions_l426_42666

noncomputable def quadratic_function_example (x : ℝ) : ℝ :=
  -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_conditions :
  quadratic_function_example 1 = 0 ∧
  quadratic_function_example 5 = 0 ∧
  quadratic_function_example 3 = 10 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_conditions_l426_42666


namespace NUMINAMATH_GPT_meeting_time_l426_42671

-- Definitions for the problem conditions.
def track_length : ℕ := 1800
def speed_A_kmph : ℕ := 36
def speed_B_kmph : ℕ := 54

-- Conversion factor from kmph to mps.
def kmph_to_mps (speed_kmph : ℕ) : ℕ := (speed_kmph * 1000) / 3600

-- Calculate the speeds in mps.
def speed_A_mps : ℕ := kmph_to_mps speed_A_kmph
def speed_B_mps : ℕ := kmph_to_mps speed_B_kmph

-- Calculate the time to complete one lap for A and B.
def time_lap_A : ℕ := track_length / speed_A_mps
def time_lap_B : ℕ := track_length / speed_B_mps

-- Prove the time to meet at the starting point.
theorem meeting_time : (Nat.lcm time_lap_A time_lap_B) = 360 := by
  -- Skipping the proof with sorry placeholder
  sorry

end NUMINAMATH_GPT_meeting_time_l426_42671


namespace NUMINAMATH_GPT_number_of_children_l426_42605

theorem number_of_children (total_crayons children_crayons children : ℕ) 
  (h1 : children_crayons = 3) 
  (h2 : total_crayons = 18) 
  (h3 : total_crayons = children_crayons * children) : 
  children = 6 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_children_l426_42605


namespace NUMINAMATH_GPT_volume_ratio_surface_area_ratio_l426_42645

theorem volume_ratio_surface_area_ratio (V1 V2 S1 S2 : ℝ) (h : V1 / V2 = 8 / 27) :
  S1 / S2 = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_surface_area_ratio_l426_42645


namespace NUMINAMATH_GPT_football_game_attendance_l426_42612

theorem football_game_attendance :
  let saturday : ℕ := 80
  let monday : ℕ := saturday - 20
  let friday : ℕ := saturday + monday
  let expected_total_audience : ℕ := 350
  let actual_total_audience : ℕ := expected_total_audience + 40
  let known_attendance : ℕ := saturday + monday + friday
  let wednesday : ℕ := actual_total_audience - known_attendance
  wednesday - monday = 50 :=
by
  let saturday : ℕ := 80
  let monday : ℕ := saturday - 20
  let friday : ℕ := saturday + monday
  let expected_total_audience : ℕ := 350
  let actual_total_audience : ℕ := expected_total_audience + 40
  let known_attendance : ℕ := saturday + monday + friday
  let wednesday : ℕ := actual_total_audience - known_attendance
  show wednesday - monday = 50
  sorry

end NUMINAMATH_GPT_football_game_attendance_l426_42612


namespace NUMINAMATH_GPT_max_xy_l426_42638

theorem max_xy (x y : ℕ) (h : 7 * x + 4 * y = 150) : x * y ≤ 200 :=
sorry

end NUMINAMATH_GPT_max_xy_l426_42638
