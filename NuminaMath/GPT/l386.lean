import Mathlib

namespace fg_difference_l386_38600

def f (x : ℝ) : ℝ := x^2 - 4 * x + 7
def g (x : ℝ) : ℝ := x + 4

theorem fg_difference : f (g 3) - g (f 3) = 20 :=
by
  sorry

end fg_difference_l386_38600


namespace sum_of_squares_and_product_pos_ints_l386_38613

variable (x y : ℕ)

theorem sum_of_squares_and_product_pos_ints :
  x^2 + y^2 = 289 ∧ x * y = 120 → x + y = 23 :=
by
  intro h
  sorry

end sum_of_squares_and_product_pos_ints_l386_38613


namespace min_expression_value_l386_38608

theorem min_expression_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + 2 * b = 1) : 
  ∃ x, (x = (a^2 + 1) / a + (2 * b^2 + 1) / b) ∧ x = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end min_expression_value_l386_38608


namespace swim_speed_in_still_water_l386_38611

-- Definitions from conditions
def downstream_speed (v_man v_stream : ℝ) : ℝ := v_man + v_stream
def upstream_speed (v_man v_stream : ℝ) : ℝ := v_man - v_stream

-- Question formatted as a proof problem
theorem swim_speed_in_still_water (v_man v_stream : ℝ)
  (h1 : downstream_speed v_man v_stream = 6)
  (h2 : upstream_speed v_man v_stream = 10) : v_man = 8 :=
by
  -- The proof will come here
  sorry

end swim_speed_in_still_water_l386_38611


namespace largest_base5_three_digit_to_base10_l386_38628

theorem largest_base5_three_digit_to_base10 :
  (4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 124 :=
by
  sorry

end largest_base5_three_digit_to_base10_l386_38628


namespace extreme_points_inequality_l386_38668

noncomputable def f (x : ℝ) (m : ℝ) := (1 / 2) * x^2 + m * Real.log (1 - x)

theorem extreme_points_inequality (m x1 x2 : ℝ) 
  (h_m1 : 0 < m) (h_m2 : m < 1 / 4)
  (h_x1 : 0 < x1) (h_x2: x1 < 1 / 2)
  (h_x3: x2 > 1 / 2) (h_x4: x2 < 1)
  (h_x5 : x1 < x2)
  (h_sum : x1 + x2 = 1)
  (h_prod : x1 * x2 = m)
  : (1 / 4) - (1 / 2) * Real.log 2 < (f x1 m) / x2 ∧ (f x1 m) / x2 < 0 :=
by
  sorry

end extreme_points_inequality_l386_38668


namespace competition_end_time_l386_38699

def time_in_minutes := 24 * 60  -- Total minutes in 24 hours

def competition_start_time := 14 * 60 + 30  -- 2:30 p.m. in minutes from midnight

theorem competition_end_time :
  competition_start_time + 1440 = competition_start_time :=
by 
  sorry

end competition_end_time_l386_38699


namespace triangle_area_in_circle_l386_38655

theorem triangle_area_in_circle (r : ℝ) (arc1 arc2 arc3 : ℝ) 
  (circumference_eq : arc1 + arc2 + arc3 = 24)
  (radius_eq : 2 * Real.pi * r = 24) : 
  1 / 2 * (r ^ 2) * (Real.sin (105 * Real.pi / 180) + Real.sin (120 * Real.pi / 180) + Real.sin (135 * Real.pi / 180)) = 364.416 / (Real.pi ^ 2) :=
by
  sorry

end triangle_area_in_circle_l386_38655


namespace lions_at_sanctuary_l386_38677

variable (L C : ℕ)

noncomputable def is_solution :=
  C = 1 / 2 * (L + 14) ∧
  L + 14 + C = 39 ∧
  L = 12

theorem lions_at_sanctuary : is_solution L C :=
sorry

end lions_at_sanctuary_l386_38677


namespace people_who_didnt_show_up_l386_38687

-- Definitions based on the conditions
def invited_people : ℕ := 68
def people_per_table : ℕ := 3
def tables_needed : ℕ := 6

-- Theorem statement
theorem people_who_didnt_show_up : 
  (invited_people - tables_needed * people_per_table = 50) :=
by 
  sorry

end people_who_didnt_show_up_l386_38687


namespace at_least_one_not_less_than_two_l386_38632

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) :=
sorry

end at_least_one_not_less_than_two_l386_38632


namespace solution_set_of_inequality_l386_38630

theorem solution_set_of_inequality (x : ℝ) : x * (9 - x) > 0 ↔ 0 < x ∧ x < 9 := by
  sorry

end solution_set_of_inequality_l386_38630


namespace parabola_intersection_square_l386_38649

theorem parabola_intersection_square (p : ℝ) :
   (∃ (x : ℝ), (x = 1 ∨ x = 2) ∧ x^2 * p = 1 ∨ x^2 * p = 2)
   → (1 / 4 ≤ p ∧ p ≤ 2) :=
by
  sorry

end parabola_intersection_square_l386_38649


namespace least_integer_greater_than_sqrt_450_l386_38636

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end least_integer_greater_than_sqrt_450_l386_38636


namespace white_area_l386_38661

/-- The area of a 5 by 17 rectangular sign. -/
def sign_area : ℕ := 5 * 17

/-- The area covered by the letter L. -/
def L_area : ℕ := 5 * 1 + 1 * 2

/-- The area covered by the letter O. -/
def O_area : ℕ := (3 * 3) - (1 * 1)

/-- The area covered by the letter V. -/
def V_area : ℕ := 2 * (3 * 1)

/-- The area covered by the letter E. -/
def E_area : ℕ := 3 * (1 * 3)

/-- The total area covered by the letters L, O, V, E. -/
def sum_black_area : ℕ := L_area + O_area + V_area + E_area

/-- The problem statement: Calculate the area of the white portion of the sign. -/
theorem white_area : sign_area - sum_black_area = 55 :=
by
  -- Place the proof here
  sorry

end white_area_l386_38661


namespace number_of_restaurants_l386_38625

def units_in_building : ℕ := 300
def residential_units := units_in_building / 2
def remaining_units := units_in_building - residential_units
def restaurants := remaining_units / 2

theorem number_of_restaurants : restaurants = 75 :=
by
  sorry

end number_of_restaurants_l386_38625


namespace induction_inequality_l386_38685

variable (n : ℕ) (h₁ : n ∈ Set.Icc 2 (2^n - 1))

theorem induction_inequality : 1 + 1/2 + 1/3 < 2 := 
  sorry

end induction_inequality_l386_38685


namespace ratio_of_scores_l386_38601

theorem ratio_of_scores (Lizzie Nathalie Aimee teammates : ℕ) (combinedLN : ℕ)
    (team_total : ℕ) (m : ℕ) :
    Lizzie = 4 →
    Nathalie = Lizzie + 3 →
    combinedLN = Lizzie + Nathalie →
    Aimee = m * combinedLN →
    teammates = 17 →
    team_total = Lizzie + Nathalie + Aimee + teammates →
    team_total = 50 →
    (Aimee / combinedLN) = 2 :=
by 
    sorry

end ratio_of_scores_l386_38601


namespace necessary_and_sufficient_condition_l386_38680

theorem necessary_and_sufficient_condition (a b : ℝ) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := 
by 
  sorry

end necessary_and_sufficient_condition_l386_38680


namespace find_circumference_l386_38647

theorem find_circumference
  (C : ℕ)
  (h1 : ∃ (vA vB : ℕ), C > 0 ∧ vA > 0 ∧ vB > 0 ∧ 
                        (120 * (C/2 + 80)) = ((C - 80) * (C/2 - 120)) ∧
                        (C - 240) / vA = (C + 240) / vB) :
  C = 520 := 
  sorry

end find_circumference_l386_38647


namespace complex_number_property_l386_38675

noncomputable def imaginary_unit : Complex := Complex.I

theorem complex_number_property (n : ℕ) (hn : 4^n = 256) : (1 + imaginary_unit)^n = -4 :=
by
  sorry

end complex_number_property_l386_38675


namespace animals_per_aquarium_l386_38642

theorem animals_per_aquarium
  (saltwater_aquariums : ℕ)
  (saltwater_animals : ℕ)
  (h1 : saltwater_aquariums = 56)
  (h2 : saltwater_animals = 2184)
  : saltwater_animals / saltwater_aquariums = 39 := by
  sorry

end animals_per_aquarium_l386_38642


namespace correct_calculation_l386_38618

theorem correct_calculation (a b : ℝ) :
  (6 * a - 5 * a ≠ 1) ∧
  (a + 2 * a^2 ≠ 3 * a^3) ∧
  (- (a - b) = -a + b) ∧
  (2 * (a + b) ≠ 2 * a + b) :=
by 
  sorry

end correct_calculation_l386_38618


namespace students_in_each_group_is_9_l386_38621

-- Define the number of students trying out for the trivia teams
def total_students : ℕ := 36

-- Define the number of students who didn't get picked for the team
def students_not_picked : ℕ := 9

-- Define the number of groups the remaining students are divided into
def number_of_groups : ℕ := 3

-- Define the function that calculates the number of students in each group
def students_per_group (total students_not_picked number_of_groups : ℕ) : ℕ :=
  (total - students_not_picked) / number_of_groups

-- Theorem: Given the conditions, the number of students in each group is 9
theorem students_in_each_group_is_9 : students_per_group total_students students_not_picked number_of_groups = 9 := 
by 
  -- proof skipped
  sorry

end students_in_each_group_is_9_l386_38621


namespace proof_problem_l386_38682

open Real

noncomputable def set_A : Set ℝ :=
  {x | x = tan (-19 * π / 6) ∨ x = sin (-19 * π / 6)}

noncomputable def set_B : Set ℝ :=
  {m | 0 <= m ∧ m <= 4}

noncomputable def set_C (a : ℝ) : Set ℝ :=
  {x | a + 1 < x ∧ x < 2 * a}

theorem proof_problem (a : ℝ) :
  set_A = {-sqrt 3 / 3, -1 / 2} ∧
  set_B = {m | 0 <= m ∧ m <= 4} ∧
  (set_A ∪ set_B) = {-sqrt 3 / 3, -1 / 2, 0, 4} →
  (∀ a, set_C a ⊆ (set_A ∪ set_B) → 1 < a ∧ a < 2) :=
sorry

end proof_problem_l386_38682


namespace lcm_of_two_numbers_l386_38676

theorem lcm_of_two_numbers (a b : ℕ) (h_prod : a * b = 145862784) (h_hcf : Nat.gcd a b = 792) : Nat.lcm a b = 184256 :=
by {
  sorry
}

end lcm_of_two_numbers_l386_38676


namespace participation_arrangements_l386_38619

def num_students : ℕ := 5
def num_competitions : ℕ := 3
def eligible_dance_students : ℕ := 4

def arrangements_singing : ℕ := num_students
def arrangements_chess : ℕ := num_students
def arrangements_dance : ℕ := eligible_dance_students

def total_arrangements : ℕ := arrangements_singing * arrangements_chess * arrangements_dance

theorem participation_arrangements :
  total_arrangements = 100 := by
  sorry

end participation_arrangements_l386_38619


namespace probability_of_cold_given_rhinitis_l386_38612

/-- Define the events A and B as propositions --/
def A : Prop := sorry -- A represents having rhinitis
def B : Prop := sorry -- B represents having a cold

/-- Define the given probabilities as assumptions --/
axiom P_A : ℝ -- P(A) = 0.8
axiom P_A_and_B : ℝ -- P(A ∩ B) = 0.6

/-- Adding the conditions --/
axiom P_A_val : P_A = 0.8
axiom P_A_and_B_val : P_A_and_B = 0.6

/-- Define the conditional probability --/
noncomputable def P_B_given_A : ℝ := P_A_and_B / P_A

/-- The main theorem which states the problem --/
theorem probability_of_cold_given_rhinitis : P_B_given_A = 0.75 :=
by 
  sorry

end probability_of_cold_given_rhinitis_l386_38612


namespace intersection_of_sets_l386_38678

def setA : Set ℝ := {x | -2 < x ∧ x < 3}
def setB : Set ℝ := {x | 0 < x ∧ x < 4}

theorem intersection_of_sets :
  setA ∩ setB = {x | 0 < x ∧ x < 3} :=
by
  sorry

end intersection_of_sets_l386_38678


namespace sum_of_roots_l386_38645

-- Define the main condition
def equation (x : ℝ) : Prop :=
  (x + 3) * (x - 4) = 22

-- Define the statement we want to prove
theorem sum_of_roots : ∀ x1 x2 : ℝ, (equation x1 ∧ equation x2) → x1 + x2 = 1 := by
  intros x1 x2 h
  sorry

end sum_of_roots_l386_38645


namespace count_paths_l386_38657

theorem count_paths (m n : ℕ) : (n + m).choose m = (n + m).choose n :=
by
  sorry

end count_paths_l386_38657


namespace work_completion_time_l386_38684

theorem work_completion_time (d : ℕ) (h : d = 9) : 3 * d = 27 := by
  sorry

end work_completion_time_l386_38684


namespace jorge_total_goals_l386_38673

theorem jorge_total_goals (last_season_goals current_season_goals : ℕ) (h_last : last_season_goals = 156) (h_current : current_season_goals = 187) : 
  last_season_goals + current_season_goals = 343 :=
by
  sorry

end jorge_total_goals_l386_38673


namespace lucy_fish_moved_l386_38670

theorem lucy_fish_moved (original_count moved_count remaining_count : ℝ)
  (h1: original_count = 212.0)
  (h2: remaining_count = 144.0) :
  moved_count = original_count - remaining_count :=
by sorry

end lucy_fish_moved_l386_38670


namespace temperature_on_friday_is_72_l386_38652

-- Define the temperatures on specific days
def temp_sunday := 40
def temp_monday := 50
def temp_tuesday := 65
def temp_wednesday := 36
def temp_thursday := 82
def temp_saturday := 26

-- Average temperature over the week
def average_temp := 53

-- Total number of days in a week
def days_in_week := 7

-- Calculate the total sum of temperatures given the average temperature
def total_sum_temp : ℤ := average_temp * days_in_week

-- Sum of known temperatures from specific days
def known_sum_temp : ℤ := temp_sunday + temp_monday + temp_tuesday + temp_wednesday + temp_thursday + temp_saturday

-- Define the temperature on Friday
def temp_friday : ℤ := total_sum_temp - known_sum_temp

theorem temperature_on_friday_is_72 : temp_friday = 72 :=
by
  -- Placeholder for the proof
  sorry

end temperature_on_friday_is_72_l386_38652


namespace sum_of_number_and_conjugate_l386_38616

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l386_38616


namespace solution_set_l386_38638

theorem solution_set (x : ℝ) : 
  1 < |x + 2| ∧ |x + 2| < 5 ↔ 
  (-7 < x ∧ x < -3) ∨ (-1 < x ∧ x < 3) := 
by 
  sorry

end solution_set_l386_38638


namespace turtle_minimum_distance_l386_38688

theorem turtle_minimum_distance 
  (constant_speed : ℝ)
  (turn_angle : ℝ)
  (total_time : ℕ) :
  constant_speed = 5 →
  turn_angle = 90 →
  total_time = 11 →
  ∃ (final_position : ℝ × ℝ), 
    (final_position = (5, 0) ∨ final_position = (-5, 0) ∨ final_position = (0, 5) ∨ final_position = (0, -5)) ∧
    dist final_position (0, 0) = 5 :=
by
  intros
  sorry

end turtle_minimum_distance_l386_38688


namespace problem1_problem2_problem3_l386_38681

theorem problem1 (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + a + 3 = 0) → (a ≤ -2 ∨ a ≥ 6) :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a + 3 ≥ 4) → 
    (if a > 2 then 
      ∀ x : ℝ, ((x ≤ 1) ∨ (x ≥ a-1)) 
    else if a = 2 then 
      ∀ x : ℝ, true
    else 
      ∀ x : ℝ, ((x ≤ a - 1) ∨ (x ≥ 1))) :=
sorry

theorem problem3 (a : ℝ) :
  (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ x^2 - a*x + a + 3 = 0) → (6 ≤ a ∧ a ≤ 7) :=
sorry

end problem1_problem2_problem3_l386_38681


namespace speed_of_man_in_still_water_l386_38689

-- Define the conditions as given in step (a)
axiom conditions :
  ∃ (v_m v_s : ℝ),
    (40 / 5 = v_m + v_s) ∧
    (30 / 5 = v_m - v_s)

-- State the theorem that proves the speed of the man in still water
theorem speed_of_man_in_still_water : ∃ v_m : ℝ, v_m = 7 :=
by
  obtain ⟨v_m, v_s, h1, h2⟩ := conditions
  have h3 : v_m + v_s = 8 := by sorry
  have h4 : v_m - v_s = 6 := by sorry
  have h5 : 2 * v_m = 14 := by sorry
  exact ⟨7, by linarith⟩

end speed_of_man_in_still_water_l386_38689


namespace infinite_integer_solutions_iff_l386_38643

theorem infinite_integer_solutions_iff
  (a b c d : ℤ) :
  (∃ inf_int_sol : (ℤ → ℤ) → Prop, ∀ (f : (ℤ → ℤ)), inf_int_sol f) ↔ (a^2 - 4*b = c^2 - 4*d) :=
by
  sorry

end infinite_integer_solutions_iff_l386_38643


namespace fraction_power_multiplication_l386_38629

theorem fraction_power_multiplication :
  ( (5 / 8: ℚ) ^ 2 * (3 / 4) ^ 2 * (2 / 3) = 75 / 512) := 
  by
  sorry

end fraction_power_multiplication_l386_38629


namespace radius_of_circle_with_area_3_14_l386_38637

theorem radius_of_circle_with_area_3_14 (A : ℝ) (π : ℝ) (hA : A = 3.14) (hπ : π = 3.14) (h_area : A = π * r^2) : r = 1 :=
by
  sorry

end radius_of_circle_with_area_3_14_l386_38637


namespace arithmetic_seq_middle_term_l386_38693

theorem arithmetic_seq_middle_term (a1 a3 y : ℤ) (h1 : a1 = 3^2) (h2 : a3 = 3^4)
    (h3 : y = (a1 + a3) / 2) : y = 45 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end arithmetic_seq_middle_term_l386_38693


namespace midpoint_s2_l386_38635

structure Point where
  x : ℤ
  y : ℤ

def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

def translate (p : Point) (dx dy : ℤ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem midpoint_s2 :
  let s1_p1 := ⟨6, -2⟩
  let s1_p2 := ⟨-4, 6⟩
  let s1_mid := midpoint s1_p1 s1_p2
  let s2_mid_translated := translate s1_mid (-3) (-4)
  s2_mid_translated = ⟨-2, -2⟩ := 
by
  sorry

end midpoint_s2_l386_38635


namespace storybook_pages_l386_38651

theorem storybook_pages :
  (10 + 5) / (1 - (1 / 5) * 2) = 25 := by
  sorry

end storybook_pages_l386_38651


namespace prisoners_freedom_guaranteed_l386_38697

-- Definition of the problem strategy
def prisoners_strategy (n : ℕ) : Prop :=
  ∃ counter regular : ℕ → ℕ,
    (∀ i, i < n - 1 → regular i < 2) ∧ -- Each regular prisoner turns on the light only once
    (∃ count : ℕ, 
      counter count = 99 ∧  -- The counter counts to 99 based on the strategy
      (∀ k, k < 99 → (counter (k + 1) = counter k + 1))) -- Each turn off increases the count by one

-- The main proof statement that there is a strategy ensuring the prisoners' release
theorem prisoners_freedom_guaranteed : ∀ (n : ℕ), n = 100 →
  prisoners_strategy n :=
by {
  sorry -- The actual proof is omitted
}

end prisoners_freedom_guaranteed_l386_38697


namespace convert_kmph_to_mps_l386_38662

theorem convert_kmph_to_mps (speed_kmph : ℝ) (km_to_m : ℝ) (hr_to_s : ℝ) : 
  speed_kmph = 56 → km_to_m = 1000 → hr_to_s = 3600 → 
  (speed_kmph * (km_to_m / hr_to_s) : ℝ) = 15.56 :=
by
  intros
  sorry

end convert_kmph_to_mps_l386_38662


namespace prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5_l386_38631

-- Definition of the inequalities to be proven using the rearrangement inequality
def inequality1 (a b : ℝ) : Prop := a^2 + b^2 ≥ 2 * a * b
def inequality2 (a b c : ℝ) : Prop := a^2 + b^2 + c^2 ≥ a * b + b * c + c * a
def inequality3 (a b : ℝ) : Prop := a^2 + b^2 + 1 ≥ a * b + b + a
def inequality5 (x y : ℝ) : Prop := x^3 + y^3 ≥ x^2 * y + x * y^2

-- Proofs required for each inequality
theorem prove_inequality1 (a b : ℝ) : inequality1 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality2 (a b c : ℝ) : inequality2 a b c := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality3 (a b : ℝ) : inequality3 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality5 (x y : ℝ) (hx : x ≥ y) (hy : 0 < y) : inequality5 x y := 
by sorry  -- This can be proved using the rearrangement inequality

end prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5_l386_38631


namespace team_problem_solved_probability_l386_38663

-- Defining the probabilities
def P_A : ℚ := 1 / 5
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Defining the probability that the problem is solved
def P_s : ℚ := 3 / 5

-- Lean 4 statement to prove that the calculated probability matches the expected solution
theorem team_problem_solved_probability :
  1 - (1 - P_A) * (1 - P_B) * (1 - P_C) = P_s :=
by
  sorry

end team_problem_solved_probability_l386_38663


namespace mul_101_101_l386_38698

theorem mul_101_101 : 101 * 101 = 10201 := 
by
  sorry

end mul_101_101_l386_38698


namespace find_a_l386_38622

noncomputable def tangent_to_circle_and_parallel (a : ℝ) : Prop := 
  let P := (2, 2)
  let circle_center := (1, 0)
  let on_circle := (P.1 - 1)^2 + P.2^2 = 5
  let perpendicular_slope := (P.2 - circle_center.2) / (P.1 - circle_center.1) * (1 / a) = -1
  on_circle ∧ perpendicular_slope

theorem find_a (a : ℝ) : tangent_to_circle_and_parallel a ↔ a = -2 :=
by
  sorry

end find_a_l386_38622


namespace total_marbles_l386_38639

theorem total_marbles (r b g : ℕ) (h_ratio : r = 1 ∧ b = 5 ∧ g = 3) (h_green : g = 27) :
  (r + b + g) * 3 = 81 :=
  sorry

end total_marbles_l386_38639


namespace converse_and_inverse_false_l386_38683

variable (Polygon : Type)
variable (RegularHexagon : Polygon → Prop)
variable (AllSidesEqual : Polygon → Prop)

theorem converse_and_inverse_false (p : Polygon → Prop) (q : Polygon → Prop)
  (h : ∀ x, RegularHexagon x → AllSidesEqual x) :
  ¬ (∀ x, AllSidesEqual x → RegularHexagon x) ∧ ¬ (∀ x, ¬ RegularHexagon x → ¬ AllSidesEqual x) :=
by
  sorry

end converse_and_inverse_false_l386_38683


namespace distance_halfway_along_orbit_l386_38694

variable {Zeta : Type}  -- Zeta is a type representing the planet
variable (distance_from_focus : Zeta → ℝ)  -- Function representing the distance from the sun (focus)

-- Conditions
variable (perigee_distance : ℝ := 3)
variable (apogee_distance : ℝ := 15)
variable (a : ℝ := (perigee_distance + apogee_distance) / 2)  -- semi-major axis

theorem distance_halfway_along_orbit (z : Zeta) (h1 : distance_from_focus z = perigee_distance) (h2 : distance_from_focus z = apogee_distance) :
  distance_from_focus z = a :=
sorry

end distance_halfway_along_orbit_l386_38694


namespace option_a_is_odd_l386_38667

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem option_a_is_odd (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_odd (a + 2 * b + 1) :=
by sorry

end option_a_is_odd_l386_38667


namespace problem1_problem2_l386_38627

theorem problem1 : (40 * Real.sqrt 3 - 18 * Real.sqrt 3 + 8 * Real.sqrt 3) / 6 = 5 * Real.sqrt 3 := 
by sorry

theorem problem2 : (Real.sqrt 3 - 2)^2023 * (Real.sqrt 3 + 2)^2023
                 - Real.sqrt 4 * Real.sqrt (1 / 2)
                 - (Real.pi - 1)^0
                = -2 - Real.sqrt 2 :=
by sorry

end problem1_problem2_l386_38627


namespace KochCurve_MinkowskiDimension_l386_38633

noncomputable def minkowskiDimensionOfKochCurve : ℝ :=
  let N (n : ℕ) := 3 * (4 ^ (n - 1))
  (Real.log 4) / (Real.log 3)

theorem KochCurve_MinkowskiDimension : minkowskiDimensionOfKochCurve = (Real.log 4) / (Real.log 3) := by
  sorry

end KochCurve_MinkowskiDimension_l386_38633


namespace distance_swim_against_current_l386_38690

-- Definitions based on problem conditions
def swimmer_speed_still_water : ℝ := 4 -- km/h
def water_current_speed : ℝ := 1 -- km/h
def time_swimming_against_current : ℝ := 2 -- hours

-- Calculation of effective speed against the current
def effective_speed_against_current : ℝ :=
  swimmer_speed_still_water - water_current_speed

-- Proof statement
theorem distance_swim_against_current :
  effective_speed_against_current * time_swimming_against_current = 6 :=
by
  -- By substituting values from the problem,
  -- effective_speed_against_current * time_swimming_against_current = 3 * 2
  -- which equals 6.
  sorry

end distance_swim_against_current_l386_38690


namespace temperature_on_Tuesday_l386_38606

variable (T W Th F : ℝ)

theorem temperature_on_Tuesday :
  (T + W + Th) / 3 = 52 →
  (W + Th + F) / 3 = 54 →
  F = 53 →
  T = 47 := by
  intros h₁ h₂ h₃
  sorry

end temperature_on_Tuesday_l386_38606


namespace quadratic_sequence_exists_l386_38623

theorem quadratic_sequence_exists (b c : ℤ) : 
  ∃ (n : ℕ) (a : ℕ → ℤ), 
  a 0 = b ∧ 
  a n = c ∧ 
  ∀ i, 1 ≤ i → i ≤ n → |a i - a (i - 1)| = i^2 :=
sorry

end quadratic_sequence_exists_l386_38623


namespace find_a1_l386_38640

-- Definitions of the conditions
def Sn (n : ℕ) : ℕ := sorry  -- Sum of the first n terms of the sequence
def a₁ : ℤ := sorry          -- First term of the sequence

axiom S_2016_eq_2016 : Sn 2016 = 2016
axiom diff_seq_eq_2000 : (Sn 2016 / 2016) - (Sn 16 / 16) = 2000

-- Proof statement
theorem find_a1 : a₁ = -2014 :=
by
  -- The proof would go here
  sorry

end find_a1_l386_38640


namespace probability_of_selected_member_l386_38634

section Probability

variables {N : ℕ} -- Total number of members in the group

-- Conditions
-- Probabilities of selecting individuals by gender
def P_woman : ℝ := 0.70
def P_man : ℝ := 0.20
def P_non_binary : ℝ := 0.10

-- Conditional probabilities of occupations given gender
def P_engineer_given_woman : ℝ := 0.20
def P_doctor_given_man : ℝ := 0.20
def P_translator_given_non_binary : ℝ := 0.20

-- The main proof statement
theorem probability_of_selected_member :
  (P_woman * P_engineer_given_woman) + (P_man * P_doctor_given_man) + (P_non_binary * P_translator_given_non_binary) = 0.20 :=
by
  sorry

end Probability

end probability_of_selected_member_l386_38634


namespace reduced_price_per_kg_l386_38614

-- Define the conditions
def reduction_factor : ℝ := 0.80
def extra_kg : ℝ := 4
def total_cost : ℝ := 684

-- Assume the original price P and reduced price R
variables (P R : ℝ)

-- Define the equations derived from the conditions
def original_cost_eq := (P * 16 = total_cost)
def reduced_cost_eq := (0.80 * P * (16 + extra_kg) = total_cost)

-- The final theorem stating the reduced price per kg of oil is 34.20 Rs
theorem reduced_price_per_kg : R = 34.20 :=
by
  have h1: P * 16 = total_cost := sorry -- This will establish the original cost
  have h2: 0.80 * P * (16 + extra_kg) = total_cost := sorry -- This will establish the reduced cost
  have Q: 16 = 16 := sorry -- Calculation of Q (original quantity)
  have h3: P = 42.75 := sorry -- Calculation of original price
  have h4: R = 0.80 * P := sorry -- Calculation of reduced price
  have h5: R = 34.20 := sorry -- Final calculation matching the required answer
  exact h5

end reduced_price_per_kg_l386_38614


namespace cost_price_of_watch_l386_38641

theorem cost_price_of_watch (CP : ℝ) (h_loss : 0.54 * CP = SP_loss)
                            (h_gain : 1.04 * CP = SP_gain)
                            (h_diff : SP_gain - SP_loss = 140) :
                            CP = 280 :=
by {
    sorry
}

end cost_price_of_watch_l386_38641


namespace problem1_problem2_l386_38695

-- Problem 1: Proove that the given expression equals 1
theorem problem1 : (2021 * 2023) / (2022^2 - 1) = 1 :=
  by
  sorry

-- Problem 2: Proove that the given expression equals 45000
theorem problem2 : 2 * 101^2 + 2 * 101 * 98 + 2 * 49^2 = 45000 :=
  by
  sorry

end problem1_problem2_l386_38695


namespace carol_initial_peanuts_l386_38654

theorem carol_initial_peanuts (p_initial p_additional p_total : Nat) (h1 : p_additional = 5) (h2 : p_total = 7) (h3 : p_initial + p_additional = p_total) : p_initial = 2 :=
by
  sorry

end carol_initial_peanuts_l386_38654


namespace age_ratio_proof_l386_38692

-- Define the ages
def sonAge := 22
def manAge := sonAge + 24

-- Define the ratio computation statement
def ageRatioInTwoYears : ℚ := 
  let sonAgeInTwoYears := sonAge + 2
  let manAgeInTwoYears := manAge + 2
  manAgeInTwoYears / sonAgeInTwoYears

-- The theorem to prove
theorem age_ratio_proof : ageRatioInTwoYears = 2 :=
by
  sorry

end age_ratio_proof_l386_38692


namespace y_intercept_of_line_l386_38664

theorem y_intercept_of_line (x y : ℝ) (h : 5 * x - 3 * y = 15) : (0, -5) = (0, (-5 : ℝ)) :=
by
  sorry

end y_intercept_of_line_l386_38664


namespace fraction_to_decimal_l386_38620

/-- The decimal equivalent of 1/4 is 0.25. -/
theorem fraction_to_decimal : (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end fraction_to_decimal_l386_38620


namespace minimum_x2y3z_l386_38605

theorem minimum_x2y3z (x y z : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : x^3 + y^3 + z^3 - 3 * x * y * z = 607) : 
  x + 2 * y + 3 * z ≥ 1215 :=
sorry

end minimum_x2y3z_l386_38605


namespace john_twice_as_old_in_x_years_l386_38656

def frank_is_younger (john_age frank_age : ℕ) : Prop :=
  frank_age = john_age - 15

def frank_future_age (frank_age : ℕ) : ℕ :=
  frank_age + 4

def john_future_age (john_age : ℕ) : ℕ :=
  john_age + 4

theorem john_twice_as_old_in_x_years (john_age frank_age x : ℕ) 
  (h1 : frank_is_younger john_age frank_age)
  (h2 : frank_future_age frank_age = 16)
  (h3 : john_age = frank_age + 15) :
  (john_age + x) = 2 * (frank_age + x) → x = 3 :=
by 
  -- Skip the proof part
  sorry

end john_twice_as_old_in_x_years_l386_38656


namespace Tony_slices_left_after_week_l386_38696

-- Define the conditions and problem statement
def Tony_slices_per_day (days : ℕ) : ℕ := days * 2
def Tony_slices_on_Saturday : ℕ := 3 + 2
def Tony_slice_on_Sunday : ℕ := 1
def Total_slices_used (days : ℕ) : ℕ := Tony_slices_per_day days + Tony_slices_on_Saturday + Tony_slice_on_Sunday
def Initial_loaf : ℕ := 22
def Slices_left (days : ℕ) : ℕ := Initial_loaf - Total_slices_used days

-- Prove that Tony has 6 slices left after a week
theorem Tony_slices_left_after_week : Slices_left 5 = 6 := by
  sorry

end Tony_slices_left_after_week_l386_38696


namespace percentage_music_l386_38679

variable (students_total : ℕ)
variable (percent_dance percent_art percent_drama percent_sports percent_photography percent_music : ℝ)

-- Define the problem conditions
def school_conditions : Prop :=
  students_total = 3000 ∧
  percent_dance = 0.125 ∧
  percent_art = 0.22 ∧
  percent_drama = 0.135 ∧
  percent_sports = 0.15 ∧
  percent_photography = 0.08 ∧
  percent_music = 1 - (percent_dance + percent_art + percent_drama + percent_sports + percent_photography)

-- Define the proof statement
theorem percentage_music : school_conditions students_total percent_dance percent_art percent_drama percent_sports percent_photography percent_music → percent_music = 0.29 :=
by
  intros h
  rw [school_conditions] at h
  sorry

end percentage_music_l386_38679


namespace binomial_expansion_coeff_x10_sub_x5_eq_251_l386_38666

open BigOperators Polynomial

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_expansion_coeff_x10_sub_x5_eq_251 :
  ∀ (a : Fin 11 → ℤ), (fun (x : ℤ) =>
    x^10 - x^5 - (a 0 + a 1 * (x - 1) + a 2 * (x - 1)^2 + 
                  a 3 * (x - 1)^3 + a 4 * (x - 1)^4 + 
                  a 5 * (x - 1)^5 + a 6 * (x - 1)^6 + 
                  a 7 * (x - 1)^7 + a 8 * (x - 1)^8 + 
                  a 9 * (x - 1)^9 + a 10 * (x - 1)^10)) = 0 → 
  a 5 = 251 := 
by 
  sorry

end binomial_expansion_coeff_x10_sub_x5_eq_251_l386_38666


namespace increasing_iff_a_le_0_l386_38624

variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 - a * x + 1

theorem increasing_iff_a_le_0 : (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 0 :=
by
  sorry

end increasing_iff_a_le_0_l386_38624


namespace apples_picked_per_tree_l386_38615

-- Definitions
def num_trees : Nat := 4
def total_apples_picked : Nat := 28

-- Proving how many apples Rachel picked from each tree if the same number were picked from each tree
theorem apples_picked_per_tree (h : num_trees ≠ 0) :
  total_apples_picked / num_trees = 7 :=
by
  sorry

end apples_picked_per_tree_l386_38615


namespace geometric_sequence_common_ratio_l386_38604

theorem geometric_sequence_common_ratio (S : ℕ → ℝ) (a : ℕ → ℝ)
  (q : ℝ) (h1 : a 1 = 2) (h2 : S 3 = 6)
  (geo_sum : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  q = 1 ∨ q = -2 :=
by
  sorry

end geometric_sequence_common_ratio_l386_38604


namespace train_B_speed_l386_38617

noncomputable def train_speed_B (V_A : ℕ) (T_A : ℕ) (T_B : ℕ) : ℕ :=
  V_A * T_A / T_B

theorem train_B_speed
  (V_A : ℕ := 60)
  (T_A : ℕ := 9)
  (T_B : ℕ := 4) :
  train_speed_B V_A T_A T_B = 135 := 
by
  sorry

end train_B_speed_l386_38617


namespace reflection_of_P_across_y_axis_l386_38646

-- Define the initial point P as a tuple
def P : ℝ × ℝ := (1, -2)

-- Define the reflection across the y-axis function
def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

-- State the theorem that we want to prove
theorem reflection_of_P_across_y_axis :
  reflect_y_axis P = (-1, -2) :=
by
  -- placeholder for the proof steps
  sorry

end reflection_of_P_across_y_axis_l386_38646


namespace term_2012_of_T_is_2057_l386_38644

-- Define a function that checks if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the sequence T as all natural numbers which are not perfect squares
def T (n : ℕ) : ℕ :=
  (n + Nat.sqrt (4 * n)) 

-- The theorem to state the mathematical proof problem
theorem term_2012_of_T_is_2057 :
  T 2012 = 2057 :=
sorry

end term_2012_of_T_is_2057_l386_38644


namespace average_height_of_trees_l386_38607

-- Define the heights of the trees
def height_tree1: ℕ := 1000
def height_tree2: ℕ := height_tree1 / 2
def height_tree3: ℕ := height_tree1 / 2
def height_tree4: ℕ := height_tree1 + 200

-- Calculate the total number of trees
def number_of_trees: ℕ := 4

-- Compute the total height climbed
def total_height: ℕ := height_tree1 + height_tree2 + height_tree3 + height_tree4

-- Define the average height
def average_height: ℕ := total_height / number_of_trees

-- The theorem statement
theorem average_height_of_trees: average_height = 800 := by
  sorry

end average_height_of_trees_l386_38607


namespace find_eighth_number_l386_38610

theorem find_eighth_number (x : ℕ) (h1 : (1 + 2 + 4 + 5 + 6 + 9 + 9 + x + 12) / 9 = 7) : x = 27 :=
sorry

end find_eighth_number_l386_38610


namespace gcd_at_most_3_digits_l386_38671

/-- If the least common multiple of two 7-digit integers has 12 digits, 
  then their greatest common divisor has at most 3 digits. -/
theorem gcd_at_most_3_digits (a b : ℕ)
  (h1 : 10^6 ≤ a ∧ a < 10^7)
  (h2 : 10^6 ≤ b ∧ b < 10^7)
  (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b ≤ 999 := 
sorry

end gcd_at_most_3_digits_l386_38671


namespace prove_a4_plus_1_div_a4_l386_38669

theorem prove_a4_plus_1_div_a4 (a : ℝ) (h : (a + 1/a)^2 = 5) : a^4 + 1/(a^4) = 7 :=
by
  sorry

end prove_a4_plus_1_div_a4_l386_38669


namespace cylinder_volume_rotation_l386_38609

theorem cylinder_volume_rotation (length width : ℝ) (π : ℝ) (h : length = 4) (w : width = 2) (V : ℝ) :
  (V = π * (4^2) * width ∨ V = π * (2^2) * length) :=
by
  sorry

end cylinder_volume_rotation_l386_38609


namespace sum_of_first_3n_terms_l386_38660

theorem sum_of_first_3n_terms (S_n S_2n S_3n : ℕ) (h1 : S_n = 48) (h2 : S_2n = 60) :
  S_3n = 63 :=
by
  sorry

end sum_of_first_3n_terms_l386_38660


namespace smallest_integer_condition_l386_38653

theorem smallest_integer_condition (x : ℝ) (hz : 9 = 9) (hineq : 27^9 > x^24) : x < 27 :=
  by {
    sorry
  }

end smallest_integer_condition_l386_38653


namespace smallest_n_divides_24_and_1024_l386_38650

theorem smallest_n_divides_24_and_1024 : ∃ n : ℕ, n > 0 ∧ (24 ∣ n^2) ∧ (1024 ∣ n^3) ∧ (∀ m : ℕ, (m > 0 ∧ (24 ∣ m^2) ∧ (1024 ∣ m^3)) → n ≤ m) :=
by
  sorry

end smallest_n_divides_24_and_1024_l386_38650


namespace point_B_between_A_and_C_l386_38665

theorem point_B_between_A_and_C (a b c : ℚ) (h_abc : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : |a - b| + |b - c| = |a - c|) : 
  (a < b ∧ b < c) ∨ (c < b ∧ b < a) :=
sorry

end point_B_between_A_and_C_l386_38665


namespace problem1_problem2_l386_38603

/-- Problem 1 -/
theorem problem1 (a b : ℝ) : (a^2 - b)^2 = a^4 - 2 * a^2 * b + b^2 :=
by
  sorry

/-- Problem 2 -/
theorem problem2 (x : ℝ) : (2 * x + 1) * (4 * x^2 - 1) * (2 * x - 1) = 16 * x^4 - 8 * x^2 + 1 :=
by
  sorry

end problem1_problem2_l386_38603


namespace company_stores_l386_38691

theorem company_stores (total_uniforms : ℕ) (uniforms_per_store : ℕ) 
  (h1 : total_uniforms = 121) (h2 : uniforms_per_store = 4) : 
  total_uniforms / uniforms_per_store = 30 :=
by
  sorry

end company_stores_l386_38691


namespace trigonometric_identity_l386_38674

theorem trigonometric_identity (α : ℝ)
  (h1 : Real.sin (π + α) = 3 / 5)
  (h2 : π < α ∧ α < 3 * π / 2) :
  (Real.sin ((π + α) / 2) - Real.cos ((π + α) / 2)) / 
  (Real.sin ((π - α) / 2) - Real.cos ((π - α) / 2)) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l386_38674


namespace obtuse_triangle_condition_l386_38648

theorem obtuse_triangle_condition
  (a b c : ℝ) 
  (h : ∃ A B C : ℝ, A + B + C = 180 ∧ A > 90 ∧ a^2 + b^2 - c^2 < 0)
  : (∃ A B C : ℝ, A + B + C = 180 ∧ A > 90 → a^2 + b^2 - c^2 < 0) := 
sorry

end obtuse_triangle_condition_l386_38648


namespace all_statements_true_l386_38659

noncomputable def g : ℝ → ℝ := sorry

axiom g_defined (x : ℝ) : ∃ y, g x = y
axiom g_positive (x : ℝ) : g x > 0
axiom g_multiplicative (a b : ℝ) : g (a) * g (b) = g (a + b)
axiom g_div (a b : ℝ) (h : a > b) : g (a - b) = g (a) / g (b)

theorem all_statements_true :
  (g 0 = 1) ∧
  (∀ a, g (-a) = 1 / g (a)) ∧
  (∀ a, g (a) = (g (3 * a))^(1 / 3)) ∧
  (∀ a b, b > a → g (b - a) < g (b)) :=
by
  sorry

end all_statements_true_l386_38659


namespace inequality_xyz_geq_3_l386_38658

theorem inequality_xyz_geq_3
  (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_not_all_zero : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :
  (2 * x^2 - x + y + z) / (x + y^2 + z^2) +
  (2 * y^2 + x - y + z) / (x^2 + y + z^2) +
  (2 * z^2 + x + y - z) / (x^2 + y^2 + z) ≥ 3 := 
sorry

end inequality_xyz_geq_3_l386_38658


namespace negation_statement_l386_38672

variables (Students : Type) (LeftHanded InChessClub : Students → Prop)

theorem negation_statement :
  (¬ ∃ x, LeftHanded x ∧ InChessClub x) ↔ (∃ x, LeftHanded x ∧ InChessClub x) :=
by
  sorry

end negation_statement_l386_38672


namespace commute_time_abs_diff_l386_38626

theorem commute_time_abs_diff (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2) :
  |x - y| = 4 := by
  sorry

end commute_time_abs_diff_l386_38626


namespace divisibility_by_65_product_of_four_natural_numbers_l386_38686

def N : ℕ := 2^2022 + 1

theorem divisibility_by_65 : ∃ k : ℕ, N = 65 * k := by
  sorry

theorem product_of_four_natural_numbers :
  ∃ a b c d : ℕ, 1 < a ∧ 1 < b ∧ 1 < c ∧ 1 < d ∧ N = a * b * c * d :=
  by sorry

end divisibility_by_65_product_of_four_natural_numbers_l386_38686


namespace prank_combinations_l386_38602

theorem prank_combinations :
  let monday := 1
  let tuesday := 4
  let wednesday := 7
  let thursday := 5
  let friday := 1
  (monday * tuesday * wednesday * thursday * friday) = 140 :=
by
  sorry

end prank_combinations_l386_38602
