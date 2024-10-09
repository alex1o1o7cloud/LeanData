import Mathlib

namespace train_length_l634_63409

theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (h_speed : speed_kmh = 30) (h_time : time_sec = 6) :
  ∃ length_meters : ℝ, abs (length_meters - 50) < 1 :=
by
  -- Converting speed from km/hr to m/s
  let speed_ms := speed_kmh * (1000 / 3600)
  
  -- Calculating length of the train using the distance formula
  let length_meters := speed_ms * time_sec

  use length_meters
  -- Proof would go here showing abs (length_meters - 50) < 1
  sorry

end train_length_l634_63409


namespace sum_of_ages_is_37_l634_63451

def maries_age : ℕ := 12
def marcos_age (M : ℕ) : ℕ := 2 * M + 1

theorem sum_of_ages_is_37 : maries_age + marcos_age maries_age = 37 := 
by
  -- Inserting the proof details
  sorry

end sum_of_ages_is_37_l634_63451


namespace reciprocal_of_sum_l634_63475

-- Define the fractions
def a := (1: ℚ) / 2
def b := (1: ℚ) / 3

-- Define their sum
def c := a + b

-- Define the expected reciprocal
def reciprocal := (6: ℚ) / 5

-- The theorem we want to prove:
theorem reciprocal_of_sum : (c⁻¹ = reciprocal) :=
by 
  sorry

end reciprocal_of_sum_l634_63475


namespace zeros_of_f_l634_63416

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 3 * x + 2

-- State the theorem about its roots
theorem zeros_of_f : ∃ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end zeros_of_f_l634_63416


namespace ratio_x_y_l634_63465

theorem ratio_x_y (x y : ℝ) (h1 : x * y = 9) (h2 : 0 < x) (h3 : 0 < y) (h4 : y = 0.5) : x / y = 36 :=
by
  sorry

end ratio_x_y_l634_63465


namespace number_of_molecules_correct_l634_63456

-- Define Avogadro's number
def avogadros_number : ℝ := 6.022 * 10^23

-- Define the given number of molecules
def given_number_of_molecules : ℝ := 3 * 10^26

-- State the problem
theorem number_of_molecules_correct :
  (number_of_molecules = given_number_of_molecules) :=
by
  sorry

end number_of_molecules_correct_l634_63456


namespace redistribute_marbles_l634_63400

theorem redistribute_marbles :
  let d := 14
  let m := 20
  let p := 19
  let v := 7
  let n := 4
  (d + m + p + v) / n = 15 :=
by
  let d := 14
  let m := 20
  let p := 19
  let v := 7
  let n := 4
  sorry

end redistribute_marbles_l634_63400


namespace water_added_to_solution_l634_63446

theorem water_added_to_solution :
  let initial_volume := 340
  let initial_sugar := 0.20 * initial_volume
  let added_sugar := 3.2
  let added_kola := 6.8
  let final_sugar := initial_sugar + added_sugar
  let final_percentage_sugar := 19.66850828729282 / 100
  let final_volume := final_sugar / final_percentage_sugar
  let added_water := final_volume - initial_volume - added_sugar - added_kola
  added_water = 12 :=
by
  sorry

end water_added_to_solution_l634_63446


namespace Sam_memorized_more_digits_l634_63482

variable (MinaDigits SamDigits CarlosDigits : ℕ)
variable (h1 : MinaDigits = 6 * CarlosDigits)
variable (h2 : MinaDigits = 24)
variable (h3 : SamDigits = 10)
 
theorem Sam_memorized_more_digits :
  SamDigits - CarlosDigits = 6 :=
by
  -- Let's unfold the statements and perform basic arithmetic.
  sorry

end Sam_memorized_more_digits_l634_63482


namespace initial_cabinets_l634_63401

theorem initial_cabinets (C : ℤ) (h1 : 26 = C + 6 * C + 5) : C = 3 := 
by 
  sorry

end initial_cabinets_l634_63401


namespace solve_for_y_l634_63440

theorem solve_for_y (x y : ℝ) (h : 4 * x + y = 9) : y = 9 - 4 * x :=
by sorry

end solve_for_y_l634_63440


namespace find_a_l634_63474

-- Condition: Define a * b as 2a - b^2
def star (a b : ℝ) := 2 * a - b^2

-- Proof problem: Prove the value of a given the condition and that a * 7 = 16.
theorem find_a : ∃ a : ℝ, star a 7 = 16 ∧ a = 32.5 :=
by
  sorry

end find_a_l634_63474


namespace total_vases_l634_63443

theorem total_vases (vases_per_day : ℕ) (days : ℕ) (total_vases : ℕ) 
  (h1 : vases_per_day = 16) 
  (h2 : days = 16) 
  (h3 : total_vases = vases_per_day * days) : 
  total_vases = 256 := 
by 
  sorry

end total_vases_l634_63443


namespace functional_equation_solution_l634_63427

theorem functional_equation_solution (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
sorry

end functional_equation_solution_l634_63427


namespace village_current_population_l634_63425

def initial_population : ℕ := 4675
def died_by_bombardment : ℕ := (5*initial_population + 99) / 100 -- Equivalent to rounding (5/100) * 4675
def remaining_after_bombardment : ℕ := initial_population - died_by_bombardment
def left_due_to_fear : ℕ := (20*remaining_after_bombardment + 99) / 100 -- Equivalent to rounding (20/100) * remaining
def current_population : ℕ := remaining_after_bombardment - left_due_to_fear

theorem village_current_population : current_population = 3553 := by
  sorry

end village_current_population_l634_63425


namespace abs_le_and_interval_iff_l634_63402

variable (x : ℝ)

theorem abs_le_and_interval_iff :
  (|x - 2| ≤ 5) ↔ (-3 ≤ x ∧ x ≤ 7) :=
by
  sorry

end abs_le_and_interval_iff_l634_63402


namespace fraction_product_l634_63470

theorem fraction_product :
  (2 / 3) * (5 / 7) * (9 / 11) * (4 / 13) = 360 / 3003 := by
  sorry

end fraction_product_l634_63470


namespace Jenny_reading_days_l634_63459

theorem Jenny_reading_days :
  let words_per_hour := 100
  let book1_words := 200
  let book2_words := 400
  let book3_words := 300
  let total_words := book1_words + book2_words + book3_words
  let total_hours := total_words / words_per_hour
  let minutes_per_day := 54
  let hours_per_day := minutes_per_day / 60
  total_hours / hours_per_day = 10 :=
by
  sorry

end Jenny_reading_days_l634_63459


namespace find_intersection_l634_63460

noncomputable def intersection_of_lines : Prop :=
  ∃ (x y : ℚ), (5 * x - 3 * y = 15) ∧ (6 * x + 2 * y = 14) ∧ (x = 11 / 4) ∧ (y = -5 / 4)

theorem find_intersection : intersection_of_lines :=
  sorry

end find_intersection_l634_63460


namespace same_color_combination_probability_l634_63469

-- Defining the number of each color candy 
def num_red : Nat := 12
def num_blue : Nat := 12
def num_green : Nat := 6

-- Terry and Mary each pick 3 candies at random
def total_pick : Nat := 3

-- The total number of candies in the jar
def total_candies : Nat := num_red + num_blue + num_green

-- Probability of Terry and Mary picking the same color combination
def probability_same_combination : ℚ := 2783 / 847525

-- The theorem statement
theorem same_color_combination_probability :
  let terry_picks_red := (num_red * (num_red - 1) * (num_red - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_red := num_red - total_pick
  let mary_picks_red := (remaining_red * (remaining_red - 1) * (remaining_red - 2)) / (27 * 26 * 25)
  let combined_red := terry_picks_red * mary_picks_red

  let terry_picks_blue := (num_blue * (num_blue - 1) * (num_blue - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_blue := num_blue - total_pick
  let mary_picks_blue := (remaining_blue * (remaining_blue - 1) * (remaining_blue - 2)) / (27 * 26 * 25)
  let combined_blue := terry_picks_blue * mary_picks_blue

  let terry_picks_green := (num_green * (num_green - 1) * (num_green - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let remaining_green := num_green - total_pick
  let mary_picks_green := (remaining_green * (remaining_green - 1) * (remaining_green - 2)) / (27 * 26 * 25)
  let combined_green := terry_picks_green * mary_picks_green

  let total_probability := 2 * combined_red + 2 * combined_blue + combined_green
  total_probability = probability_same_combination := sorry

end same_color_combination_probability_l634_63469


namespace union_of_A_and_B_l634_63444

variable {α : Type*}

def A (x : ℝ) : Prop := x - 1 > 0
def B (x : ℝ) : Prop := 0 < x ∧ x ≤ 3

theorem union_of_A_and_B : ∀ x : ℝ, (A x ∨ B x) ↔ (0 < x) :=
by
  sorry

end union_of_A_and_B_l634_63444


namespace find_value_of_complex_fraction_l634_63494

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem find_value_of_complex_fraction :
  (1 - 2 * i) / (1 + i) = -1 / 2 - 3 / 2 * i := 
sorry

end find_value_of_complex_fraction_l634_63494


namespace total_distance_traveled_in_12_hours_l634_63463

variable (n a1 d : ℕ) (u : ℕ → ℕ)

def arithmetic_seq_sum (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  n * a1 + (n * (n - 1) * d) / 2

theorem total_distance_traveled_in_12_hours :
  arithmetic_seq_sum 12 55 2 = 792 := by
  sorry

end total_distance_traveled_in_12_hours_l634_63463


namespace inequality_conditions_l634_63413

variable (a b : ℝ)

theorem inequality_conditions (ha : 1 / a < 1 / b) (hb : 1 / b < 0) : 
  (1 / (a + b) < 1 / (a * b)) ∧ ¬(a * - (1 / a) > b * - (1 / b)) := 
by 
  sorry

end inequality_conditions_l634_63413


namespace hyperbola_asymptotes_l634_63407

theorem hyperbola_asymptotes :
  ∀ x y : ℝ, x^2 - y^2 / 4 = 1 → (y = 2 * x ∨ y = -2 * x) :=
by
  intros x y h
  sorry

end hyperbola_asymptotes_l634_63407


namespace initial_stock_decaf_percentage_l634_63478

-- Definitions as conditions of the problem
def initial_coffee_stock : ℕ := 400
def purchased_coffee_stock : ℕ := 100
def percentage_decaf_purchased : ℕ := 60
def total_percentage_decaf : ℕ := 32

/-- The proof problem statement -/
theorem initial_stock_decaf_percentage : 
  ∃ x : ℕ, x * initial_coffee_stock / 100 + percentage_decaf_purchased * purchased_coffee_stock / 100 = total_percentage_decaf * (initial_coffee_stock + purchased_coffee_stock) / 100 ∧ x = 25 :=
sorry

end initial_stock_decaf_percentage_l634_63478


namespace quadratic_has_at_most_two_solutions_l634_63480

theorem quadratic_has_at_most_two_solutions (a b c : ℝ) (h : a ≠ 0) :
  ¬(∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
    a * x1^2 + b * x1 + c = 0 ∧ 
    a * x2^2 + b * x2 + c = 0 ∧ 
    a * x3^2 + b * x3 + c = 0) := 
by {
  sorry
}

end quadratic_has_at_most_two_solutions_l634_63480


namespace sum_arithmetic_sequence_S12_l634_63420

variable {a : ℕ → ℝ} -- Arithmetic sequence a_n
variable {S : ℕ → ℝ} -- Sum of the first n terms S_n

-- Conditions given in the problem
axiom condition1 (n : ℕ) : S n = (n / 2) * (a 1 + a n)
axiom condition2 : a 4 + a 9 = 10

-- Proving that S 12 = 60 given the conditions
theorem sum_arithmetic_sequence_S12 : S 12 = 60 := by
  sorry

end sum_arithmetic_sequence_S12_l634_63420


namespace max_sum_arithmetic_sequence_terms_l634_63487

theorem max_sum_arithmetic_sequence_terms (d : ℝ) (a : ℕ → ℝ) (n : ℕ) 
  (h0 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h1 : d < 0)
  (h2 : a 1 ^ 2 = a 11 ^ 2) : 
  (n = 5) ∨ (n = 6) :=
sorry

end max_sum_arithmetic_sequence_terms_l634_63487


namespace triangle_perimeter_l634_63457

theorem triangle_perimeter (a b : ℕ) (h1 : a = 2) (h2 : b = 3) (x : ℕ) 
  (x_odd : x % 2 = 1) (triangle_ineq : 1 < x ∧ x < 5) : a + b + x = 8 :=
by
  sorry

end triangle_perimeter_l634_63457


namespace zoo_people_l634_63495

def number_of_people (cars : ℝ) (people_per_car : ℝ) : ℝ :=
  cars * people_per_car

theorem zoo_people (h₁ : cars = 3.0) (h₂ : people_per_car = 63.0) :
  number_of_people cars people_per_car = 189.0 :=
by
  rw [h₁, h₂]
  -- multiply the numbers directly after substitution
  norm_num
  -- left this as a placeholder for now, can use calc or norm_num for final steps
  exact sorry

end zoo_people_l634_63495


namespace distance_from_tangency_to_tangent_theorem_l634_63450

noncomputable def distance_from_tangency_to_tangent (R r : ℝ) : ℝ :=
  2 * R * r / (R + r)

theorem distance_from_tangency_to_tangent_theorem (R r : ℝ) :
  ∃ d : ℝ, d = distance_from_tangency_to_tangent R r :=
by
  use 2 * R * r / (R + r)
  sorry

end distance_from_tangency_to_tangent_theorem_l634_63450


namespace max_n_for_coloring_l634_63429

noncomputable def maximum_n : ℕ :=
  11

theorem max_n_for_coloring :
  ∃ n : ℕ, (n = maximum_n) ∧ ∀ k ∈ Finset.range n, 
  (∃ x y : ℕ, 1 ≤ x ∧ x ≤ 14 ∧ 1 ≤ y ∧ y ≤ 14 ∧ (x - y = k ∨ y - x = k) ∧ x ≠ y) ∧
  (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 14 ∧ 1 ≤ b ∧ b ≤ 14 ∧ (a - b = k ∨ b - a = k) ∧ a ≠ b) :=
sorry

end max_n_for_coloring_l634_63429


namespace sequence_x_value_l634_63490

theorem sequence_x_value
  (z y x : ℤ)
  (h1 : z + (-2) = -1)
  (h2 : y + 1 = -2)
  (h3 : x + (-3) = 1) :
  x = 4 := 
sorry

end sequence_x_value_l634_63490


namespace marble_problem_solution_l634_63464

noncomputable def probability_two_marbles (red_marble_initial white_marble_initial total_drawn : ℕ) : ℚ :=
  let total_initial := red_marble_initial + white_marble_initial
  let probability_first_white := (white_marble_initial : ℚ) / total_initial
  let red_marble_after_first_draw := red_marble_initial
  let total_after_first_draw := total_initial - 1
  let probability_second_red := (red_marble_after_first_draw : ℚ) / total_after_first_draw
  probability_first_white * probability_second_red

theorem marble_problem_solution :
  probability_two_marbles 4 6 2 = 4 / 15 := by
  sorry

end marble_problem_solution_l634_63464


namespace outlet_two_rate_l634_63461

/-- Definitions and conditions for the problem -/
def tank_volume_feet : ℝ := 20
def inlet_rate_cubic_inches_per_min : ℝ := 5
def outlet_one_rate_cubic_inches_per_min : ℝ := 9
def empty_time_minutes : ℝ := 2880
def cubic_feet_to_cubic_inches : ℝ := 1728
def tank_volume_cubic_inches := tank_volume_feet * cubic_feet_to_cubic_inches

/-- Statement to prove the rate of the other outlet pipe -/
theorem outlet_two_rate (x : ℝ) :
  tank_volume_cubic_inches / empty_time_minutes = outlet_one_rate_cubic_inches_per_min + x - inlet_rate_cubic_inches_per_min → 
  x = 8 :=
by
  sorry

end outlet_two_rate_l634_63461


namespace parabola_focus_directrix_l634_63430

noncomputable def parabola_distance_property (p : ℝ) (hp : 0 < p) : Prop :=
  let focus := (2 * p, 0)
  let directrix := -2 * p
  let distance := 4 * p
  p = distance / 4

-- Theorem: Given a parabola with equation y^2 = 8px (p > 0), p represents 1/4 of the distance from the focus to the directrix.
theorem parabola_focus_directrix (p : ℝ) (hp : 0 < p) : parabola_distance_property p hp :=
by
  sorry

end parabola_focus_directrix_l634_63430


namespace find_cd_l634_63479

theorem find_cd : 
  (∀ x : ℝ, (4 * x - 3) / (x^2 - 3 * x - 18) = ((7 / 3) / (x - 6)) + ((5 / 3) / (x + 3))) :=
by
  intro x
  have h : x^2 - 3 * x - 18 = (x - 6) * (x + 3) := by
    sorry
  rw [h]
  sorry

end find_cd_l634_63479


namespace fraction_power_rule_l634_63436

theorem fraction_power_rule :
  (5 / 6) ^ 4 = (625 : ℚ) / 1296 := 
by sorry

end fraction_power_rule_l634_63436


namespace james_ate_eight_slices_l634_63489

-- Define the conditions
def num_pizzas := 2
def slices_per_pizza := 6
def fraction_james_ate := 2 / 3
def total_slices := num_pizzas * slices_per_pizza

-- Define the statement to prove
theorem james_ate_eight_slices : fraction_james_ate * total_slices = 8 :=
by
  sorry

end james_ate_eight_slices_l634_63489


namespace average_age_of_boys_l634_63472

theorem average_age_of_boys
  (N : ℕ) (G : ℕ) (A_G : ℕ) (A_S : ℚ) (B : ℕ)
  (hN : N = 652)
  (hG : G = 163)
  (hA_G : A_G = 11)
  (hA_S : A_S = 11.75)
  (hB : B = N - G) :
  (163 * 11 + 489 * x = 11.75 * 652) → x = 12 := by
  sorry

end average_age_of_boys_l634_63472


namespace anie_days_to_complete_l634_63455

def normal_work_hours : ℕ := 10
def extra_hours : ℕ := 5
def total_project_hours : ℕ := 1500

theorem anie_days_to_complete :
  (total_project_hours / (normal_work_hours + extra_hours)) = 100 :=
by
  sorry

end anie_days_to_complete_l634_63455


namespace convert_to_base_8_l634_63426

theorem convert_to_base_8 (n : ℕ) (hn : n = 3050) : 
  ∃ d1 d2 d3 d4 : ℕ, d1 = 5 ∧ d2 = 7 ∧ d3 = 5 ∧ d4 = 2 ∧ n = d1 * 8^3 + d2 * 8^2 + d3 * 8^1 + d4 * 8^0 :=
by 
  use 5, 7, 5, 2
  sorry

end convert_to_base_8_l634_63426


namespace jack_last_10_shots_made_l634_63462

theorem jack_last_10_shots_made (initial_shots : ℕ) (initial_percentage : ℚ)
  (additional_shots : ℕ) (new_percentage : ℚ)
  (initial_successful_shots : initial_shots * initial_percentage = 18)
  (total_shots : initial_shots + additional_shots = 40)
  (total_successful_shots : (initial_shots + additional_shots) * new_percentage = 25) :
  ∃ x : ℕ, x = 7 := by
sorry

end jack_last_10_shots_made_l634_63462


namespace find_number_l634_63404

theorem find_number (x : ℝ) (h : 45 * 7 = 0.35 * x) : x = 900 :=
by
  -- Proof (skipped with sorry)
  sorry

end find_number_l634_63404


namespace shortest_distance_between_circles_zero_l634_63415

noncomputable def center_radius_circle1 : (ℝ × ℝ) × ℝ :=
  let c1 := (3, -5)
  let r1 := Real.sqrt 20
  (c1, r1)

noncomputable def center_radius_circle2 : (ℝ × ℝ) × ℝ :=
  let c2 := (-4, 1)
  let r2 := Real.sqrt 1
  (c2, r2)

theorem shortest_distance_between_circles_zero :
  let c1 := center_radius_circle1.1
  let r1 := center_radius_circle1.2
  let c2 := center_radius_circle2.1
  let r2 := center_radius_circle2.2
  let dist := Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2)
  dist < r1 + r2 → 0 = 0 :=
by
  intros
  -- Add appropriate steps for the proof (skipping by using sorry for now)
  sorry

end shortest_distance_between_circles_zero_l634_63415


namespace range_of_ab_min_value_of_ab_plus_inv_ab_l634_63419

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 0 < a * b ∧ a * b ≤ 1 / 4 :=
sorry

theorem min_value_of_ab_plus_inv_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (∃ ab, ab = a * b ∧ ab + 1 / ab = 17 / 4) :=
sorry

end range_of_ab_min_value_of_ab_plus_inv_ab_l634_63419


namespace initial_video_files_l634_63432

theorem initial_video_files (V : ℕ) (h1 : 26 + V - 48 = 14) : V = 36 := 
by
  sorry

end initial_video_files_l634_63432


namespace Alyssa_missed_games_l634_63466

theorem Alyssa_missed_games (total_games attended_games : ℕ) (h1 : total_games = 31) (h2 : attended_games = 13) : total_games - attended_games = 18 :=
by sorry

end Alyssa_missed_games_l634_63466


namespace solve_equation_l634_63428

theorem solve_equation (x : ℝ) (h : 3 * x ≠ 0) (h2 : x + 2 ≠ 0) : (2 / (3 * x) = 1 / (x + 2)) ↔ x = 4 := by
  sorry

end solve_equation_l634_63428


namespace linear_system_substitution_correct_l634_63452

theorem linear_system_substitution_correct (x y : ℝ)
  (h1 : y = x - 1)
  (h2 : x + 2 * y = 7) :
  x + 2 * x - 2 = 7 :=
by
  sorry

end linear_system_substitution_correct_l634_63452


namespace inequality_proof_l634_63492

noncomputable def inequality (x y z : ℝ) : Prop :=
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7

theorem inequality_proof (x y z : ℝ) (hx : x ≥ y + z) (hx_pos: 0 < x) (hy_pos: 0 < y) (hz_pos: 0 < z) :
  inequality x y z :=
by
  sorry

end inequality_proof_l634_63492


namespace parallel_lines_coefficient_l634_63498

theorem parallel_lines_coefficient (a : ℝ) :
  (x + 2*a*y - 1 = 0) → (3*a - 1)*x - a*y - 1 = 0 → (a = 0 ∨ a = 1/6) :=
by
  sorry

end parallel_lines_coefficient_l634_63498


namespace fifteenth_even_multiple_of_5_l634_63423

theorem fifteenth_even_multiple_of_5 : 15 * 2 * 5 = 150 := by
  sorry

end fifteenth_even_multiple_of_5_l634_63423


namespace range_of_a_l634_63424

theorem range_of_a (a : ℝ) (h : a ≤ 1) :
  (∃! n : ℕ, n = (2 - a) - a + 1) → -1 < a ∧ a ≤ 0 :=
by 
  sorry

end range_of_a_l634_63424


namespace negation_of_abs_x_minus_2_lt_3_l634_63410

theorem negation_of_abs_x_minus_2_lt_3 :
  ¬ (∀ x : ℝ, |x - 2| < 3) ↔ ∃ x : ℝ, |x - 2| ≥ 3 :=
by
  sorry

end negation_of_abs_x_minus_2_lt_3_l634_63410


namespace percentage_excess_calculation_l634_63491

theorem percentage_excess_calculation (A B : ℝ) (x : ℝ) 
  (h1 : (A * (1 + x / 100)) * (B * 0.95) = A * B * 1.007) : 
  x = 6.05 :=
by
  sorry

end percentage_excess_calculation_l634_63491


namespace proof_problem_l634_63458

noncomputable def problem (a b c d : ℝ) : Prop :=
(a + b + c = 3) ∧ 
(a + b + d = -1) ∧ 
(a + c + d = 8) ∧ 
(b + c + d = 0) ∧ 
(a * b + c * d = -127 / 9)

theorem proof_problem (a b c d : ℝ) : 
  (a + b + c = 3) → 
  (a + b + d = -1) →
  (a + c + d = 8) → 
  (b + c + d = 0) → 
  (a * b + c * d = -127 / 9) :=
by 
  intro h1 h2 h3 h4
  -- Proof is omitted, "sorry" indicates it is to be filled in
  admit

end proof_problem_l634_63458


namespace maximize_sequence_l634_63412

theorem maximize_sequence (n : ℕ) (an : ℕ → ℝ) (h : ∀ n, an n = (10/11)^n * (3 * n + 13)) : 
  (∃ n_max, (∀ m, an m ≤ an n_max) ∧ n_max = 6) :=
by
  sorry

end maximize_sequence_l634_63412


namespace paving_cost_l634_63422

variable (L : ℝ) (W : ℝ) (R : ℝ)

def area (L W : ℝ) := L * W
def cost (A R : ℝ) := A * R

theorem paving_cost (hL : L = 5) (hW : W = 4.75) (hR : R = 900) : cost (area L W) R = 21375 :=
by
  sorry

end paving_cost_l634_63422


namespace remainder_sum_first_150_l634_63484

-- Definitions based on the conditions
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Lean statement equivalent to the mathematical problem
theorem remainder_sum_first_150 :
  (sum_first_n 150) % 11250 = 75 :=
by 
sorry

end remainder_sum_first_150_l634_63484


namespace necessary_sufficient_condition_geometric_sequence_l634_63499

noncomputable def an_geometric (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem necessary_sufficient_condition_geometric_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ) (p q : ℝ) (h_sum : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
  (h_eq : ∀ n : ℕ, a (n + 1) = p * S n + q) :
  (a 1 = q) ↔ (∃ r : ℝ, an_geometric a r) :=
sorry

end necessary_sufficient_condition_geometric_sequence_l634_63499


namespace eighteen_gon_vertex_number_l634_63497

theorem eighteen_gon_vertex_number (a b : ℕ) (P : ℕ) (h₁ : a = 20) (h₂ : b = 18) (h₃ : P = a + b) : P = 38 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end eighteen_gon_vertex_number_l634_63497


namespace max_value_is_5_l634_63468

noncomputable def max_value (θ φ : ℝ) : ℝ :=
  3 * Real.sin θ * Real.cos φ + 2 * Real.sin φ ^ 2

theorem max_value_is_5 (θ φ : ℝ) (h1 : 0 ≤ θ) (h2 : θ ≤ Real.pi / 2) (h3 : 0 ≤ φ) (h4 : φ ≤ Real.pi / 2) :
  max_value θ φ ≤ 5 :=
sorry

end max_value_is_5_l634_63468


namespace smallest_number_among_10_11_12_l634_63411

theorem smallest_number_among_10_11_12 : min (min 10 11) 12 = 10 :=
by sorry

end smallest_number_among_10_11_12_l634_63411


namespace root_of_quadratic_eq_is_two_l634_63431

theorem root_of_quadratic_eq_is_two (k : ℝ) : (2^2 - 3 * 2 + k = 0) → k = 2 :=
by
  intro h
  sorry

end root_of_quadratic_eq_is_two_l634_63431


namespace solution_set_f_pos_min_a2_b2_c2_l634_63406

def f (x : ℝ) : ℝ := |2 * x + 3| - |x - 1|

theorem solution_set_f_pos : 
  { x : ℝ | f x > 0 } = { x : ℝ | x < -3 / 2 ∨ -2 / 3 < x } := 
sorry

theorem min_a2_b2_c2 (a b c : ℝ) (h : a + 2 * b + 3 * c = 5) : 
  a^2 + b^2 + c^2 ≥ 25 / 14 :=
sorry

end solution_set_f_pos_min_a2_b2_c2_l634_63406


namespace ff_two_eq_three_l634_63442

noncomputable def f (x : ℝ) : ℝ :=
  if x < 6 then x^3 else Real.log x / Real.log x

theorem ff_two_eq_three : f (f 2) = 3 := by
  sorry

end ff_two_eq_three_l634_63442


namespace find_value_l634_63439

def star (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 2010

variable (a b : ℝ)

axiom h1 : 3 * a + 5 * b = 1
axiom h2 : 4 * a + 9 * b = -1

theorem find_value : star a b 1 2 = 2010 := 
by 
  sorry

end find_value_l634_63439


namespace julies_birthday_day_of_week_l634_63447

theorem julies_birthday_day_of_week
    (fred_birthday_monday : Nat)
    (pat_birthday_before_fred : Nat)
    (julie_birthday_before_pat : Nat)
    (fred_birthday_after_pat : fred_birthday_monday - pat_birthday_before_fred = 37)
    (julie_birthday_before_pat_eq : pat_birthday_before_fred - julie_birthday_before_pat = 67)
    : (julie_birthday_before_pat - julie_birthday_before_pat % 7 + ((julie_birthday_before_pat % 7) - fred_birthday_monday % 7)) % 7 = 2 :=
by
  sorry

end julies_birthday_day_of_week_l634_63447


namespace inlet_pipe_filling_rate_l634_63476

def leak_rate (volume : ℕ) (time_hours : ℕ) : ℕ :=
  volume / time_hours

def net_emptying_rate (volume : ℕ) (time_hours : ℕ) : ℕ :=
  volume / time_hours

def inlet_rate_per_hour (net_rate : ℕ) (leak_rate : ℕ) : ℕ :=
  leak_rate - net_rate

def convert_to_minutes (rate_per_hour : ℕ) : ℕ :=
  rate_per_hour / 60

theorem inlet_pipe_filling_rate :
  let volume := 4320
  let time_to_empty_with_leak := 6
  let net_time_to_empty := 12
  let leak_rate := leak_rate volume time_to_empty_with_leak
  let net_rate := net_emptying_rate volume net_time_to_empty
  let fill_rate_per_hour := inlet_rate_per_hour net_rate leak_rate
  convert_to_minutes fill_rate_per_hour = 6 := by
    -- Proof ends with a placeholder 'sorry'
    sorry

end inlet_pipe_filling_rate_l634_63476


namespace factorize_expression_l634_63448

theorem factorize_expression (x y : ℝ) : 
  (x + y)^2 - 14 * (x + y) + 49 = (x + y - 7)^2 := 
by
  sorry

end factorize_expression_l634_63448


namespace test_point_selection_l634_63441

theorem test_point_selection (x_1 x_2 : ℝ)
    (interval_begin interval_end : ℝ) (h_interval : interval_begin = 2 ∧ interval_end = 4)
    (h_better_result : x_1 < x_2 ∨ x_1 > x_2)
    (h_test_points : (x_1 = interval_begin + 0.618 * (interval_end - interval_begin) ∧ 
                     x_2 = interval_begin + interval_end - x_1) ∨ 
                    (x_1 = interval_begin + interval_end - (interval_begin + 0.618 * (interval_end - interval_begin)) ∧ 
                     x_2 = interval_begin + 0.618 * (interval_end - interval_begin)))
  : ∃ x_3, x_3 = 3.528 ∨ x_3 = 2.472 := by
    sorry

end test_point_selection_l634_63441


namespace books_sold_in_january_l634_63434

theorem books_sold_in_january (J : ℕ) 
  (h_avg : (J + 16 + 17) / 3 = 16) : J = 15 :=
sorry

end books_sold_in_january_l634_63434


namespace binary_addition_correct_l634_63408

-- define the binary numbers as natural numbers using their binary representations
def bin_1010 : ℕ := 0b1010
def bin_10 : ℕ := 0b10
def bin_sum : ℕ := 0b1100

-- state the theorem that needs to be proved
theorem binary_addition_correct : bin_1010 + bin_10 = bin_sum := by
  sorry

end binary_addition_correct_l634_63408


namespace investment_amount_l634_63477

noncomputable def annual_income (investment : ℝ) (percent_stock : ℝ) (market_price : ℝ) : ℝ :=
  (investment * percent_stock / 100) / market_price * market_price

theorem investment_amount (annual_income_value : ℝ) (percent_stock : ℝ) (market_price : ℝ) (investment : ℝ) :
  annual_income investment percent_stock market_price = annual_income_value →
  investment = 6800 :=
by
  intros
  sorry

end investment_amount_l634_63477


namespace remainder_when_x_squared_div_30_l634_63467

theorem remainder_when_x_squared_div_30 (x : ℤ) 
  (h1 : 5 * x ≡ 15 [ZMOD 30]) 
  (h2 : 7 * x ≡ 13 [ZMOD 30]) : 
  (x^2) % 30 = 21 := 
by 
  sorry

end remainder_when_x_squared_div_30_l634_63467


namespace Janet_sold_six_action_figures_l634_63433

variable {x : ℕ}

theorem Janet_sold_six_action_figures
  (h₁ : 10 - x + 4 + 2 * (10 - x + 4) = 24) :
  x = 6 :=
by
  sorry

end Janet_sold_six_action_figures_l634_63433


namespace intersection_A_B_l634_63483

-- Define set A
def A : Set ℤ := {-1, 1, 2, 3, 4}

-- Define set B with the given condition
def B : Set ℤ := {x : ℤ | 1 ≤ x ∧ x < 3}

-- The main theorem statement showing the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 2} :=
    sorry -- Placeholder for the proof

end intersection_A_B_l634_63483


namespace two_rooks_non_attacking_two_kings_non_attacking_two_bishops_non_attacking_two_knights_non_attacking_two_queens_non_attacking_l634_63445

noncomputable def rooks_non_attacking : Nat :=
  8 * 8 * 7 * 7 / 2

theorem two_rooks_non_attacking : rooks_non_attacking = 1568 := by
  sorry

noncomputable def kings_non_attacking : Nat :=
  (4 * 60 + 24 * 58 + 36 * 55 + 24 * 55 + 4 * 50) / 2

theorem two_kings_non_attacking : kings_non_attacking = 1806 := by
  sorry

noncomputable def bishops_non_attacking : Nat :=
  (28 * 25 + 20 * 54 + 12 * 52 + 4 * 50) / 2

theorem two_bishops_non_attacking : bishops_non_attacking = 1736 := by
  sorry

noncomputable def knights_non_attacking : Nat :=
  (4 * 61 + 8 * 60 + 20 * 59 + 16 * 57 + 15 * 55) / 2

theorem two_knights_non_attacking : knights_non_attacking = 1848 := by
  sorry

noncomputable def queens_non_attacking : Nat :=
  (28 * 42 + 20 * 40 + 12 * 38 + 4 * 36) / 2

theorem two_queens_non_attacking : queens_non_attacking = 1288 := by
  sorry

end two_rooks_non_attacking_two_kings_non_attacking_two_bishops_non_attacking_two_knights_non_attacking_two_queens_non_attacking_l634_63445


namespace binomial_coefficient_30_3_l634_63454

theorem binomial_coefficient_30_3 :
  Nat.choose 30 3 = 4060 := 
by 
  sorry

end binomial_coefficient_30_3_l634_63454


namespace rice_in_each_container_ounces_l634_63449

-- Given conditions
def total_rice_pounds : ℚ := 25 / 4
def num_containers : ℕ := 4
def pounds_to_ounces : ℚ := 16

-- Problem statement: proving the amount of rice in each container in ounces
theorem rice_in_each_container_ounces :
  (total_rice_pounds / num_containers) * pounds_to_ounces = 25 :=
by sorry

end rice_in_each_container_ounces_l634_63449


namespace sum_of_digits_l634_63405

theorem sum_of_digits (a b : ℕ) (h1 : 10 * a + b + 10 * b + a = 202) (h2 : a < 10) (h3 : b < 10) :
  a + b = 12 :=
sorry

end sum_of_digits_l634_63405


namespace complement_M_l634_63437

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x^2 - 4 ≤ 0 }

theorem complement_M (U M : Set ℝ) : (U \ M) = { x : ℝ | x < -2 ∨ x > 2 } :=
by
  sorry

end complement_M_l634_63437


namespace vector_dot_product_l634_63414

open Matrix

section VectorDotProduct

variables (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)
variables (E : ℝ × ℝ) (F : ℝ × ℝ)

def vector_sub (P Q : ℝ × ℝ) : ℝ × ℝ := (P.1 - Q.1, P.2 - Q.2)
def vector_add (P Q : ℝ × ℝ) : ℝ × ℝ := (P.1 + Q.1, P.2 + Q.2)
def scalar_mul (k : ℝ) (P : ℝ × ℝ) : ℝ × ℝ := (k * P.1, k * P.2)
def dot_product (P Q : ℝ × ℝ) : ℝ := P.1 * Q.1 + P.2 * Q.2

axiom A_coord : A = (1, 2)
axiom B_coord : B = (2, -1)
axiom C_coord : C = (2, 2)
axiom E_is_trisection : vector_add (vector_sub B A) (scalar_mul (1/3) (vector_sub C B)) = E
axiom F_is_trisection : vector_add (vector_sub B A) (scalar_mul (2/3) (vector_sub C B)) = F

theorem vector_dot_product : dot_product (vector_sub E A) (vector_sub F A) = 3 := by
  sorry

end VectorDotProduct

end vector_dot_product_l634_63414


namespace unit_digit_of_six_consecutive_product_is_zero_l634_63493

theorem unit_digit_of_six_consecutive_product_is_zero (n : ℕ) (h : n > 0) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5)) % 10 = 0 := 
by sorry

end unit_digit_of_six_consecutive_product_is_zero_l634_63493


namespace find_number_l634_63473

theorem find_number (x : ℝ) : (1.12 * x) / 4.98 = 528.0642570281125 → x = 2350 :=
  by 
  sorry

end find_number_l634_63473


namespace proportion_correct_l634_63481

theorem proportion_correct {a b : ℝ} (h : 2 * a = 5 * b) : a / 5 = b / 2 :=
by {
  sorry
}

end proportion_correct_l634_63481


namespace find_a_for_quadratic_l634_63496

theorem find_a_for_quadratic (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20 * x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20 * y) + 20 = 0) ↔ a = 20 := 
sorry

end find_a_for_quadratic_l634_63496


namespace choir_population_l634_63418

theorem choir_population 
  (female_students : ℕ) 
  (male_students : ℕ) 
  (choir_multiple : ℕ) 
  (total_students_orchestra : ℕ := female_students + male_students)
  (total_students_choir : ℕ := choir_multiple * total_students_orchestra)
  (h_females : female_students = 18) 
  (h_males : male_students = 25) 
  (h_multiple : choir_multiple = 3) : 
  total_students_choir = 129 := 
by
  -- The proof of the theorem will be done here.
  sorry

end choir_population_l634_63418


namespace square_side_length_l634_63421

theorem square_side_length (A : ℝ) (h : A = 100) : ∃ s : ℝ, s * s = A ∧ s = 10 := by
  sorry

end square_side_length_l634_63421


namespace fraction_inequality_l634_63486

theorem fraction_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / b) + (b / c) + (c / a) ≤ (a^2 / b^2) + (b^2 / c^2) + (c^2 / a^2) := 
by
  sorry

end fraction_inequality_l634_63486


namespace parabola_range_l634_63438

theorem parabola_range (x : ℝ) (h : 0 < x ∧ x < 3) : 
  1 ≤ (x^2 - 4*x + 5) ∧ (x^2 - 4*x + 5) < 5 :=
sorry

end parabola_range_l634_63438


namespace value_of_x_l634_63488

theorem value_of_x (x : ℕ) (M : Set ℕ) :
  M = {0, 1, 2} →
  M ∪ {x} = {0, 1, 2, 3} →
  x = 3 :=
by
  sorry

end value_of_x_l634_63488


namespace ripe_oranges_count_l634_63403

/-- They harvest 52 sacks of unripe oranges per day. -/
def unripe_oranges_per_day : ℕ := 52

/-- After 26 days of harvest, they will have 2080 sacks of oranges. -/
def total_oranges_after_26_days : ℕ := 2080

/-- Define the number of sacks of ripe oranges harvested per day. -/
def ripe_oranges_per_day (R : ℕ) : Prop :=
  26 * (R + unripe_oranges_per_day) = total_oranges_after_26_days

/-- Prove that they harvest 28 sacks of ripe oranges per day. -/
theorem ripe_oranges_count : ripe_oranges_per_day 28 :=
by {
  -- This is where the proof would go
  sorry
}

end ripe_oranges_count_l634_63403


namespace rectangle_area_l634_63485

theorem rectangle_area (P l w : ℝ) (h1 : P = 60) (h2 : l / w = 3 / 2) (h3 : P = 2 * l + 2 * w) : l * w = 216 :=
by
  sorry

end rectangle_area_l634_63485


namespace age_of_b_is_6_l634_63435

theorem age_of_b_is_6 (x : ℕ) (h1 : 5 * x / 3 * x = 5 / 3)
                         (h2 : (5 * x + 2) / (3 * x + 2) = 3 / 2) : 3 * x = 6 := 
by
  sorry

end age_of_b_is_6_l634_63435


namespace smallest_integer_value_l634_63417

theorem smallest_integer_value (x : ℤ) (h : 7 - 3 * x < 22) : x ≥ -4 := 
sorry

end smallest_integer_value_l634_63417


namespace a_x1_x2_x13_eq_zero_l634_63453

theorem a_x1_x2_x13_eq_zero {a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ}
  (h1: a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) *
             (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2: a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) *
             (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := by
  sorry

end a_x1_x2_x13_eq_zero_l634_63453


namespace number_of_teachers_l634_63471

theorem number_of_teachers
  (students : ℕ) (lessons_per_student_per_day : ℕ) (lessons_per_teacher_per_day : ℕ) (students_per_class : ℕ)
  (h1 : students = 1200)
  (h2 : lessons_per_student_per_day = 5)
  (h3 : lessons_per_teacher_per_day = 4)
  (h4 : students_per_class = 30) :
  ∃ teachers : ℕ, teachers = 50 :=
by
  have total_lessons : ℕ := lessons_per_student_per_day * students
  have classes : ℕ := total_lessons / students_per_class
  have teachers : ℕ := classes / lessons_per_teacher_per_day
  use teachers
  sorry

end number_of_teachers_l634_63471
